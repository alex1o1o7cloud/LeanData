import Mathlib

namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l241_24166

theorem purely_imaginary_complex_number (m : ℝ) : 
  (Complex.I * Complex.I = -1) →
  (∃ (z : ℂ), z = m * (m + 1) + Complex.I * (m^2 - 1) ∧ z.re = 0 ∧ z.im ≠ 0) →
  m = 0 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l241_24166


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_quadratic_inequality_range_set_l241_24193

/-- For a real number a, if ax^2 + ax + a + 3 > 0 for all real x, then a ≥ 0 -/
theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + a + 3 > 0) → a ≥ 0 := by
  sorry

/-- The set of all real numbers a satisfying the quadratic inequality for all x is [0, +∞) -/
theorem quadratic_inequality_range_set : 
  {a : ℝ | ∀ x : ℝ, a * x^2 + a * x + a + 3 > 0} = Set.Ici (0 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_quadratic_inequality_range_set_l241_24193


namespace NUMINAMATH_CALUDE_sequence_term_l241_24133

def S (n : ℕ) := 2 * n^2 + n

theorem sequence_term (n : ℕ) (h : n > 0) : 
  (∀ k, k > 0 → S k - S (k-1) = 4*k - 1) → 
  S n - S (n-1) = 4*n - 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_term_l241_24133


namespace NUMINAMATH_CALUDE_sin_cos_15_deg_l241_24171

theorem sin_cos_15_deg : 
  Real.sin (15 * π / 180) ^ 4 - Real.cos (15 * π / 180) ^ 4 = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_15_deg_l241_24171


namespace NUMINAMATH_CALUDE_unique_digit_for_divisibility_l241_24176

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def four_digit_number (B : ℕ) : ℕ := 5000 + 100 * B + 10 * B + 3

theorem unique_digit_for_divisibility :
  ∃! B : ℕ, B ≤ 9 ∧ is_divisible_by_9 (four_digit_number B) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_digit_for_divisibility_l241_24176


namespace NUMINAMATH_CALUDE_rohan_farm_size_l241_24164

/-- Represents the characteristics of Rohan's coconut farm and its earnings -/
structure CoconutFarm where
  trees_per_sqm : ℕ := 2
  coconuts_per_tree : ℕ := 6
  harvest_period_months : ℕ := 3
  coconut_price : ℚ := 1/2
  total_earnings : ℚ := 240
  total_period_months : ℕ := 6

/-- Calculates the size of the coconut farm based on given parameters -/
def farm_size (farm : CoconutFarm) : ℚ :=
  farm.total_earnings / (farm.trees_per_sqm * farm.coconuts_per_tree * farm.coconut_price * (farm.total_period_months / farm.harvest_period_months))

/-- Theorem stating that Rohan's coconut farm size is 20 square meters -/
theorem rohan_farm_size (farm : CoconutFarm) : farm_size farm = 20 := by
  sorry

end NUMINAMATH_CALUDE_rohan_farm_size_l241_24164


namespace NUMINAMATH_CALUDE_factor_calculation_l241_24159

theorem factor_calculation (initial_number : ℕ) (factor : ℚ) : 
  initial_number = 5 → 
  factor * (2 * initial_number + 15) = 75 → 
  factor = 3 := by sorry

end NUMINAMATH_CALUDE_factor_calculation_l241_24159


namespace NUMINAMATH_CALUDE_min_value_theorem_l241_24109

theorem min_value_theorem (a b : ℝ) (ha : a > 0) 
  (h : ∀ x > 0, (a * x - 1) * (x^2 + b * x - 4) ≥ 0) : 
  (∀ c, b + 2 / a ≥ c) → c = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l241_24109


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l241_24119

theorem algebraic_expression_equality (y : ℝ) : 
  2 * y^2 + 3 * y + 7 = 8 → 4 * y^2 + 6 * y - 9 = -7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l241_24119


namespace NUMINAMATH_CALUDE_expression_evaluation_l241_24105

theorem expression_evaluation :
  let x : ℝ := -1
  let y : ℝ := Real.sqrt 2
  (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y) = 5 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l241_24105


namespace NUMINAMATH_CALUDE_arbitrary_across_classes_most_representative_l241_24156

/-- Represents a sampling method for a student survey --/
inductive SamplingMethod
  | GradeSpecific
  | GenderSpecific
  | ActivitySpecific
  | ArbitraryAcrossClasses

/-- Determines if a sampling method is representative of the entire student population --/
def is_representative (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.ArbitraryAcrossClasses => true
  | _ => false

/-- Theorem stating that the arbitrary across classes method is the most representative --/
theorem arbitrary_across_classes_most_representative :
  ∀ (method : SamplingMethod),
    is_representative method →
    method = SamplingMethod.ArbitraryAcrossClasses :=
by
  sorry

#check arbitrary_across_classes_most_representative

end NUMINAMATH_CALUDE_arbitrary_across_classes_most_representative_l241_24156


namespace NUMINAMATH_CALUDE_m_plus_3_interpretation_l241_24138

/-- Represents the possible interpretations of an assignment statement -/
inductive AssignmentInterpretation
  | AssignToSum
  | AddAndReassign
  | Equality
  | None

/-- Defines the meaning of an assignment statement -/
def assignmentMeaning (left : String) (right : String) : AssignmentInterpretation :=
  if left = right.take (right.length - 2) && right.takeRight 2 = "+3" then
    AssignmentInterpretation.AddAndReassign
  else
    AssignmentInterpretation.None

/-- Theorem stating the correct interpretation of M=M+3 -/
theorem m_plus_3_interpretation :
  assignmentMeaning "M" "M+3" = AssignmentInterpretation.AddAndReassign :=
by sorry

end NUMINAMATH_CALUDE_m_plus_3_interpretation_l241_24138


namespace NUMINAMATH_CALUDE_skittles_division_l241_24153

theorem skittles_division (total_skittles : ℕ) (num_students : ℕ) (skittles_per_student : ℕ) :
  total_skittles = 27 →
  num_students = 9 →
  total_skittles = num_students * skittles_per_student →
  skittles_per_student = 3 := by
  sorry

end NUMINAMATH_CALUDE_skittles_division_l241_24153


namespace NUMINAMATH_CALUDE_speed_in_still_water_l241_24129

/-- Theorem: Given a man's upstream and downstream speeds, his speed in still water
    is the average of these two speeds. -/
theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) :
  upstream_speed = 55 →
  downstream_speed = 65 →
  (upstream_speed + downstream_speed) / 2 = 60 := by
sorry

end NUMINAMATH_CALUDE_speed_in_still_water_l241_24129


namespace NUMINAMATH_CALUDE_perpendicular_vectors_a_equals_two_l241_24160

-- Define the vectors m and n
def m : ℝ × ℝ := (1, 2)
def n (a : ℝ) : ℝ × ℝ := (a, -1)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem perpendicular_vectors_a_equals_two :
  ∀ a : ℝ, dot_product m (n a) = 0 → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_vectors_a_equals_two_l241_24160


namespace NUMINAMATH_CALUDE_regular_soda_bottles_l241_24167

theorem regular_soda_bottles (total_bottles : ℕ) (diet_bottles : ℕ) 
  (h1 : total_bottles = 17) 
  (h2 : diet_bottles = 8) : 
  total_bottles - diet_bottles = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_soda_bottles_l241_24167


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l241_24101

theorem complex_modulus_problem (a : ℝ) (i : ℂ) : 
  i ^ 2 = -1 →
  (Complex.I : ℂ) ^ 2 = -1 →
  ((a - Real.sqrt 2 + i) / i).im = 0 →
  Complex.abs (2 * a + Complex.I * Real.sqrt 2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l241_24101


namespace NUMINAMATH_CALUDE_binomial_coefficient_23_5_l241_24188

theorem binomial_coefficient_23_5 (h1 : Nat.choose 21 3 = 1330)
                                  (h2 : Nat.choose 21 4 = 5985)
                                  (h3 : Nat.choose 21 5 = 20349) :
  Nat.choose 23 5 = 33649 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_23_5_l241_24188


namespace NUMINAMATH_CALUDE_hex_351_equals_849_l241_24162

/-- Converts a hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | '0' => 0 | '1' => 1 | '2' => 2 | '3' => 3
  | '4' => 4 | '5' => 5 | '6' => 6 | '7' => 7
  | '8' => 8 | '9' => 9 | 'A' => 10 | 'B' => 11
  | 'C' => 12 | 'D' => 13 | 'E' => 14 | 'F' => 15
  | _ => 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.foldr (fun c acc => 16 * acc + hex_to_dec c) 0

/-- Theorem: The hexadecimal number 351 is equal to 849 in decimal -/
theorem hex_351_equals_849 : hex_string_to_dec "351" = 849 := by
  sorry

end NUMINAMATH_CALUDE_hex_351_equals_849_l241_24162


namespace NUMINAMATH_CALUDE_min_coefficient_value_l241_24185

theorem min_coefficient_value (c d box : ℤ) : 
  (∀ x : ℝ, (c * x + d) * (d * x + c) = 29 * x^2 + box * x + 29) →
  c ≠ d ∧ c ≠ box ∧ d ≠ box →
  ∀ b : ℤ, (∀ x : ℝ, (c * x + d) * (d * x + c) = 29 * x^2 + b * x + 29) → box ≤ b →
  box = 842 :=
by sorry

end NUMINAMATH_CALUDE_min_coefficient_value_l241_24185


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l241_24113

theorem longest_side_of_triangle (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  Real.tan A = 1/4 ∧
  Real.tan B = 3/5 ∧
  a = min a (min b c) ∧
  a = Real.sqrt 2 ∧
  c = max a (max b c) →
  c = Real.sqrt 17 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l241_24113


namespace NUMINAMATH_CALUDE_complex_fraction_value_l241_24182

theorem complex_fraction_value (a : ℝ) (z : ℂ) :
  z = (a^2 - 1 : ℂ) + (a + 1 : ℂ) * I →
  z.re = 0 →
  (a + I^2016) / (1 + I) = 1 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_value_l241_24182


namespace NUMINAMATH_CALUDE_range_of_a_for_solution_a_value_for_minimum_l241_24147

-- Define the function f
def f (a x : ℝ) : ℝ := |2*x - a| + |x - 1|

-- Part 1
theorem range_of_a_for_solution (a : ℝ) :
  (∃ x, f a x ≤ 2 - |x - 1|) ↔ 0 ≤ a ∧ a ≤ 4 :=
sorry

-- Part 2
theorem a_value_for_minimum (a : ℝ) :
  a < 2 → (∀ x, f a x ≥ 3) → (∃ x, f a x = 3) → a = -4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_solution_a_value_for_minimum_l241_24147


namespace NUMINAMATH_CALUDE_simplify_square_root_sum_l241_24125

theorem simplify_square_root_sum : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 98 / Real.sqrt 56) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_root_sum_l241_24125


namespace NUMINAMATH_CALUDE_shortest_distance_circle_to_line_l241_24178

/-- The shortest distance from a point on a circle to a line -/
theorem shortest_distance_circle_to_line :
  let center : ℝ × ℝ := (3, -3)
  let radius : ℝ := 3
  let line := {p : ℝ × ℝ | p.1 = p.2}
  ∃ (shortest : ℝ), 
    shortest = 3 * (Real.sqrt 2 - 1) ∧
    ∀ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 →
      shortest ≤ Real.sqrt ((p.1 - p.2)^2 + (p.2 - p.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_circle_to_line_l241_24178


namespace NUMINAMATH_CALUDE_smallest_possible_a_l241_24108

def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem smallest_possible_a :
  ∀ (a b c : ℝ),
  a > 0 →
  parabola a b c (-1/3) = -4/3 →
  (∃ n : ℤ, a + b + c = n) →
  (∀ a' : ℝ, a' > 0 ∧ 
    (∃ b' c' : ℝ, parabola a' b' c' (-1/3) = -4/3 ∧ 
    (∃ n : ℤ, a' + b' + c' = n)) → 
  a' ≥ 3/16) →
  a = 3/16 := by
sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l241_24108


namespace NUMINAMATH_CALUDE_side_view_area_is_four_l241_24161

/-- Represents a triangular prism -/
structure TriangularPrism where
  lateral_edge_length : ℝ
  base_side_length : ℝ
  main_view_side_length : ℝ

/-- The area of the side view of a triangular prism -/
def side_view_area (prism : TriangularPrism) : ℝ :=
  prism.lateral_edge_length * prism.base_side_length

/-- Theorem: The area of the side view of a specific triangular prism is 4 -/
theorem side_view_area_is_four :
  ∀ (prism : TriangularPrism),
    prism.lateral_edge_length = 2 →
    prism.base_side_length = 2 →
    prism.main_view_side_length = 2 →
    side_view_area prism = 4 := by
  sorry

end NUMINAMATH_CALUDE_side_view_area_is_four_l241_24161


namespace NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l241_24121

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem union_of_A_and_complement_of_B :
  A ∪ (U \ B) = {1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l241_24121


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l241_24144

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 1) :
  (1 : ℝ) / (2*a + 3*b) + 1 / (2*b + 3*c) + 1 / (2*c + 3*a) ≥ 9/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l241_24144


namespace NUMINAMATH_CALUDE_final_probability_l241_24173

/-- Represents the number of operations performed -/
def num_operations : ℕ := 5

/-- Represents the initial number of red balls -/
def initial_red : ℕ := 2

/-- Represents the initial number of blue balls -/
def initial_blue : ℕ := 1

/-- Represents the final number of red balls -/
def final_red : ℕ := 4

/-- Represents the final number of blue balls -/
def final_blue : ℕ := 4

/-- Calculates the probability of drawing a specific sequence of balls -/
def sequence_probability (red_draws blue_draws : ℕ) : ℚ := sorry

/-- Calculates the number of possible sequences -/
def num_sequences : ℕ := sorry

/-- The main theorem stating the probability of the final outcome -/
theorem final_probability : 
  sequence_probability (final_red - initial_red) (final_blue - initial_blue) * num_sequences = 2/7 := by sorry

end NUMINAMATH_CALUDE_final_probability_l241_24173


namespace NUMINAMATH_CALUDE_total_students_correct_l241_24140

/-- The number of students who tried out for the trivia teams -/
def total_students : ℕ := 17

/-- The number of students who didn't get picked -/
def not_picked : ℕ := 5

/-- The number of groups formed -/
def num_groups : ℕ := 3

/-- The number of students in each group -/
def students_per_group : ℕ := 4

/-- Theorem stating that the total number of students who tried out is correct -/
theorem total_students_correct : 
  total_students = not_picked + num_groups * students_per_group := by
  sorry

end NUMINAMATH_CALUDE_total_students_correct_l241_24140


namespace NUMINAMATH_CALUDE_total_students_surveyed_l241_24143

/-- Represents the number of students speaking different combinations of languages --/
structure LanguageCounts where
  french : ℕ
  english : ℕ
  spanish : ℕ
  frenchEnglish : ℕ
  frenchSpanish : ℕ
  englishSpanish : ℕ
  allThree : ℕ
  none : ℕ

/-- The conditions given in the problem --/
def languageConditions (counts : LanguageCounts) : Prop :=
  -- 230 students speak only one language
  counts.french + counts.english + counts.spanish = 230 ∧
  -- 190 students speak exactly two languages
  counts.frenchEnglish + counts.frenchSpanish + counts.englishSpanish = 190 ∧
  -- 40 students speak all three languages
  counts.allThree = 40 ∧
  -- 60 students do not speak any of the three languages
  counts.none = 60 ∧
  -- Among French speakers, 25% speak English, 15% speak Spanish, and 10% speak both English and Spanish
  4 * (counts.frenchEnglish + counts.allThree) = (counts.french + counts.frenchEnglish + counts.frenchSpanish + counts.allThree) ∧
  20 * (counts.frenchSpanish + counts.allThree) = 3 * (counts.french + counts.frenchEnglish + counts.frenchSpanish + counts.allThree) ∧
  10 * counts.allThree = (counts.french + counts.frenchEnglish + counts.frenchSpanish + counts.allThree) ∧
  -- Among English speakers, 20% also speak Spanish
  5 * (counts.englishSpanish + counts.allThree) = (counts.english + counts.frenchEnglish + counts.englishSpanish + counts.allThree)

/-- The theorem to be proved --/
theorem total_students_surveyed (counts : LanguageCounts) :
  languageConditions counts →
  counts.french + counts.english + counts.spanish +
  counts.frenchEnglish + counts.frenchSpanish + counts.englishSpanish +
  counts.allThree + counts.none = 520 :=
by sorry

end NUMINAMATH_CALUDE_total_students_surveyed_l241_24143


namespace NUMINAMATH_CALUDE_railway_theorem_l241_24103

structure City where
  id : Nat

structure DirectedGraph where
  cities : Set City
  connections : City → City → Prop

def reachable (g : DirectedGraph) (a b : City) : Prop :=
  g.connections a b ∨ ∃ c, g.connections a c ∧ g.connections c b

theorem railway_theorem (g : DirectedGraph) 
  (h₁ : ∀ a b : City, a ∈ g.cities → b ∈ g.cities → a ≠ b → (g.connections a b ∨ g.connections b a)) :
  ∃ n : City, n ∈ g.cities ∧ ∀ m : City, m ∈ g.cities → m ≠ n → reachable g m n :=
sorry

end NUMINAMATH_CALUDE_railway_theorem_l241_24103


namespace NUMINAMATH_CALUDE_magnitude_a_plus_2b_eq_sqrt_2_l241_24180

def a : Fin 2 → ℝ := ![(-1 : ℝ), 3]
def b : Fin 2 → ℝ := ![1, -2]

theorem magnitude_a_plus_2b_eq_sqrt_2 :
  Real.sqrt ((a 0 + 2 * b 0)^2 + (a 1 + 2 * b 1)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_a_plus_2b_eq_sqrt_2_l241_24180


namespace NUMINAMATH_CALUDE_power_2023_preserves_order_l241_24194

theorem power_2023_preserves_order (a b : ℝ) (h : a > b) : a^2023 > b^2023 := by
  sorry

end NUMINAMATH_CALUDE_power_2023_preserves_order_l241_24194


namespace NUMINAMATH_CALUDE_product_mod_fifty_l241_24102

theorem product_mod_fifty : ∃ m : ℕ, 0 ≤ m ∧ m < 50 ∧ (289 * 673) % 50 = m ∧ m = 47 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_fifty_l241_24102


namespace NUMINAMATH_CALUDE_shells_added_correct_l241_24114

/-- Given an initial amount of shells and a final amount of shells,
    calculate the amount of shells added. -/
def shells_added (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that given the initial amount of 5 pounds and
    final amount of 28 pounds, the amount of shells added is 23 pounds. -/
theorem shells_added_correct :
  shells_added 5 28 = 23 := by
  sorry

end NUMINAMATH_CALUDE_shells_added_correct_l241_24114


namespace NUMINAMATH_CALUDE_total_matchsticks_l241_24148

def boxes : ℕ := 4
def matchboxes_per_box : ℕ := 20
def sticks_per_matchbox : ℕ := 300

theorem total_matchsticks :
  boxes * matchboxes_per_box * sticks_per_matchbox = 24000 :=
by sorry

end NUMINAMATH_CALUDE_total_matchsticks_l241_24148


namespace NUMINAMATH_CALUDE_function_property_result_l241_24124

theorem function_property_result (g : ℝ → ℝ) 
    (h : ∀ a c : ℝ, c^3 * g a = a^3 * g c) 
    (h_nonzero : g 3 ≠ 0) : 
  (g 6 - g 2) / g 3 = 208/27 := by
sorry

end NUMINAMATH_CALUDE_function_property_result_l241_24124


namespace NUMINAMATH_CALUDE_prime_sequence_l241_24187

def is_increasing (s : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → s n < s m

theorem prime_sequence (a p : ℕ → ℕ) 
  (h_inc : is_increasing a)
  (h_prime : ∀ n, Nat.Prime (p n))
  (h_div : ∀ n, (p n) ∣ (a n))
  (h_diff : ∀ n k, a n - a k = p n - p k) :
  ∀ n, a n = p n :=
sorry

end NUMINAMATH_CALUDE_prime_sequence_l241_24187


namespace NUMINAMATH_CALUDE_absolute_value_difference_l241_24134

theorem absolute_value_difference : |-3 * (7 - 15)| - |(5 - 7)^2 + (-4)^2| = 4 := by sorry

end NUMINAMATH_CALUDE_absolute_value_difference_l241_24134


namespace NUMINAMATH_CALUDE_impossibility_theorem_l241_24131

/-- Represents the number of boxes -/
def n : ℕ := 100

/-- Represents the initial number of stones in each box -/
def initial_stones (i : ℕ) : ℕ := i

/-- Represents the condition for moving stones between boxes -/
def can_move (a b : ℕ) : Prop := a + b = 101

/-- Represents the desired final configuration -/
def desired_config (stones : ℕ → ℕ) : Prop :=
  stones 70 = 69 ∧ stones 50 = 51 ∧ 
  ∀ i, i ≠ 70 ∧ i ≠ 50 → stones i = initial_stones i

/-- Main theorem: It's impossible to achieve the desired configuration -/
theorem impossibility_theorem :
  ¬ ∃ (stones : ℕ → ℕ), 
    (∀ i j, i ≠ j → can_move (stones i) (stones j) → 
      ∃ k l, k ≠ l ∧ stones k + stones l = 101) ∧
    desired_config stones :=
sorry

end NUMINAMATH_CALUDE_impossibility_theorem_l241_24131


namespace NUMINAMATH_CALUDE_third_month_sale_l241_24135

/-- Proves that the sale in the third month is 6855 given the conditions of the problem -/
theorem third_month_sale (sales : Fin 6 → ℕ) : 
  (sales 0 = 6335) → 
  (sales 1 = 6927) → 
  (sales 3 = 7230) → 
  (sales 4 = 6562) → 
  (sales 5 = 5091) → 
  ((sales 0 + sales 1 + sales 2 + sales 3 + sales 4 + sales 5) / 6 = 6500) → 
  sales 2 = 6855 := by
sorry

end NUMINAMATH_CALUDE_third_month_sale_l241_24135


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l241_24132

theorem profit_percent_calculation (selling_price : ℝ) (cost_price : ℝ) 
  (h : cost_price = 0.9 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = (1 / 9) * 100 := by
sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l241_24132


namespace NUMINAMATH_CALUDE_retail_markup_percentage_l241_24111

theorem retail_markup_percentage 
  (wholesale : ℝ) 
  (retail : ℝ) 
  (h1 : retail > 0) 
  (h2 : wholesale > 0) 
  (h3 : retail * 0.75 = wholesale * 1.3500000000000001) : 
  (retail / wholesale - 1) * 100 = 80.00000000000002 := by
sorry

end NUMINAMATH_CALUDE_retail_markup_percentage_l241_24111


namespace NUMINAMATH_CALUDE_fraction_integer_iff_q_in_set_l241_24118

theorem fraction_integer_iff_q_in_set (q : ℕ+) :
  (∃ (k : ℕ+), (5 * q + 35 : ℤ) = k * (3 * q - 7)) ↔ 
  q ∈ ({3, 4, 5, 7, 9, 15, 21, 31} : Set ℕ+) := by
  sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_q_in_set_l241_24118


namespace NUMINAMATH_CALUDE_factory_production_l241_24190

/-- Calculates the number of toys produced per day in a factory -/
def toys_per_day (total_toys : ℕ) (work_days : ℕ) : ℕ :=
  total_toys / work_days

theorem factory_production :
  let total_weekly_production := 6000
  let work_days_per_week := 4
  toys_per_day total_weekly_production work_days_per_week = 1500 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_l241_24190


namespace NUMINAMATH_CALUDE_r_daily_earning_l241_24174

/-- The daily earnings of p, q, and r satisfy the given conditions and r earns 70 per day -/
theorem r_daily_earning (p q r : ℚ) : 
  (9 * (p + q + r) = 1620) → 
  (5 * (p + r) = 600) → 
  (7 * (q + r) = 910) → 
  r = 70 := by
  sorry

end NUMINAMATH_CALUDE_r_daily_earning_l241_24174


namespace NUMINAMATH_CALUDE_triangle_side_range_l241_24183

theorem triangle_side_range (x : ℝ) : 
  (x > 0) →  -- Ensure positive side lengths
  (x + (x + 1) > (x + 2)) →  -- Triangle inequality
  (x + (x + 1) + (x + 2) ≤ 12) →  -- Perimeter condition
  (1 < x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l241_24183


namespace NUMINAMATH_CALUDE_poem_distribution_theorem_l241_24175

def distribute_poems (n : ℕ) (k : ℕ) (min_poems : ℕ) : ℕ :=
  let case1 := (n.choose 2) * ((n - 2).choose 2) * 3
  let case2 := (n.choose 2) * ((n - 2).choose 3) * 3
  case1 + case2

theorem poem_distribution_theorem :
  distribute_poems 8 3 2 = 2940 := by
  sorry

end NUMINAMATH_CALUDE_poem_distribution_theorem_l241_24175


namespace NUMINAMATH_CALUDE_carols_pool_water_carols_pool_water_proof_l241_24117

/-- Calculates the amount of water left in Carol's pool after five hours of filling and a leak -/
theorem carols_pool_water (first_hour_rate : ℕ) (second_third_hour_rate : ℕ) (fourth_hour_rate : ℕ) (leak_amount : ℕ) : ℕ :=
  let total_added := first_hour_rate + 2 * second_third_hour_rate + fourth_hour_rate
  total_added - leak_amount

/-- Proves that the amount of water left in Carol's pool after five hours is 34 gallons -/
theorem carols_pool_water_proof :
  carols_pool_water 8 10 14 8 = 34 := by
  sorry

end NUMINAMATH_CALUDE_carols_pool_water_carols_pool_water_proof_l241_24117


namespace NUMINAMATH_CALUDE_spheres_in_base_of_165_pyramid_l241_24191

/-- The number of spheres in a regular triangular pyramid with n levels -/
def pyramid_spheres (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- The number of spheres in the base of a regular triangular pyramid with n levels -/
def base_spheres (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: In a regular triangular pyramid with exactly 165 identical spheres,
    the number of spheres in the base is 45 -/
theorem spheres_in_base_of_165_pyramid :
  ∃ n : ℕ, pyramid_spheres n = 165 ∧ base_spheres n = 45 :=
sorry

end NUMINAMATH_CALUDE_spheres_in_base_of_165_pyramid_l241_24191


namespace NUMINAMATH_CALUDE_stans_paper_words_per_page_l241_24112

/-- Calculates the number of words per page in Stan's paper. -/
theorem stans_paper_words_per_page 
  (typing_speed : ℕ)        -- Stan's typing speed in words per minute
  (pages : ℕ)               -- Number of pages in the paper
  (water_per_hour : ℕ)      -- Water consumption rate in ounces per hour
  (total_water : ℕ)         -- Total water consumed while writing the paper
  (h1 : typing_speed = 50)  -- Stan types 50 words per minute
  (h2 : pages = 5)          -- The paper is 5 pages long
  (h3 : water_per_hour = 15) -- Stan drinks 15 ounces of water per hour while typing
  (h4 : total_water = 10)   -- Stan drinks 10 ounces of water while writing his paper
  : (typing_speed * (total_water * 60 / water_per_hour)) / pages = 400 := by
  sorry

#check stans_paper_words_per_page

end NUMINAMATH_CALUDE_stans_paper_words_per_page_l241_24112


namespace NUMINAMATH_CALUDE_not_both_perfect_squares_l241_24189

theorem not_both_perfect_squares (x y z t : ℕ+) 
  (h1 : x.val * y.val - z.val * t.val = x.val + y.val)
  (h2 : x.val + y.val = z.val + t.val) :
  ¬(∃ (a c : ℕ), x.val * y.val = a^2 ∧ z.val * t.val = c^2) :=
by sorry

end NUMINAMATH_CALUDE_not_both_perfect_squares_l241_24189


namespace NUMINAMATH_CALUDE_log_625_squared_base_5_l241_24100

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_625_squared_base_5 : log 5 (625^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_log_625_squared_base_5_l241_24100


namespace NUMINAMATH_CALUDE_sum_with_gap_l241_24110

theorem sum_with_gap (x : ℝ) (h1 : |x - 5.46| = 3.97) (h2 : x < 5.46) : x + 5.46 = 6.95 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_gap_l241_24110


namespace NUMINAMATH_CALUDE_rational_function_sum_l241_24152

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_quadratic : ∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c
  q_cubic : ∃ a b c d : ℝ, ∀ x, q x = a * x^3 + b * x^2 + c * x + d
  p_cond : p 4 = 4
  q_cond1 : q 1 = 0
  q_cond2 : q 3 = 3
  q_factor : ∃ r : ℝ → ℝ, ∀ x, q x = (x - 2) * r x

/-- The main theorem -/
theorem rational_function_sum (f : RationalFunction) :
  ∃ p q : ℝ → ℝ, (∀ x, f.p x = p x ∧ f.q x = q x) ∧
  (∀ x, p x + q x = (1/2) * x^3 - (5/4) * x^2 + (17/4) * x) := by
  sorry

end NUMINAMATH_CALUDE_rational_function_sum_l241_24152


namespace NUMINAMATH_CALUDE_dividend_calculation_l241_24115

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17)
  (h2 : quotient = 8)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 141 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l241_24115


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l241_24127

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + a 16 = 6) →
  (a 2 * a 16 = 1) →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l241_24127


namespace NUMINAMATH_CALUDE_min_value_of_fraction_sum_l241_24168

theorem min_value_of_fraction_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : 2*m + n = 1) :
  (1/m + 2/n) ≥ 8 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2*m₀ + n₀ = 1 ∧ 1/m₀ + 2/n₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_sum_l241_24168


namespace NUMINAMATH_CALUDE_sqrt_25_l241_24177

theorem sqrt_25 : {x : ℝ | x^2 = 25} = {5, -5} := by sorry

end NUMINAMATH_CALUDE_sqrt_25_l241_24177


namespace NUMINAMATH_CALUDE_tom_free_lessons_l241_24154

/-- Calculates the number of free dance lessons given the total number of lessons,
    cost per lesson, and total amount paid. -/
def free_lessons (total_lessons : ℕ) (cost_per_lesson : ℕ) (total_paid : ℕ) : ℕ :=
  total_lessons - (total_paid / cost_per_lesson)

/-- Proves that Tom received 2 free dance lessons given the problem conditions. -/
theorem tom_free_lessons :
  let total_lessons : ℕ := 10
  let cost_per_lesson : ℕ := 10
  let total_paid : ℕ := 80
  free_lessons total_lessons cost_per_lesson total_paid = 2 := by
  sorry


end NUMINAMATH_CALUDE_tom_free_lessons_l241_24154


namespace NUMINAMATH_CALUDE_baseball_team_wins_l241_24137

theorem baseball_team_wins (total_games wins : ℕ) (h1 : total_games = 130) (h2 : wins = 101) :
  let losses := total_games - wins
  wins - 3 * losses = 14 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_wins_l241_24137


namespace NUMINAMATH_CALUDE_total_students_l241_24128

/-- Given a student's position from right and left in a line, calculate the total number of students -/
theorem total_students (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 6)
  (h2 : rank_from_left = 5) :
  rank_from_right + rank_from_left - 1 = 10 := by
  sorry

#check total_students

end NUMINAMATH_CALUDE_total_students_l241_24128


namespace NUMINAMATH_CALUDE_range_of_b_l241_24198

theorem range_of_b (a b : ℝ) (h : a * b^2 > a ∧ a > a * b) : b < -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_b_l241_24198


namespace NUMINAMATH_CALUDE_distribute_four_to_three_eq_36_l241_24186

/-- The number of ways to distribute four distinct objects into three non-empty groups -/
def distribute_four_to_three : ℕ := 36

/-- Theorem stating that the number of ways to distribute four distinct objects 
    into three non-empty groups is 36 -/
theorem distribute_four_to_three_eq_36 : 
  distribute_four_to_three = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_four_to_three_eq_36_l241_24186


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l241_24120

theorem exponential_equation_solution :
  ∃ x : ℝ, (16 : ℝ) ^ x * (16 : ℝ) ^ x * (16 : ℝ) ^ x = (256 : ℝ) ^ 3 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l241_24120


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l241_24145

/-- Given a sphere with surface area 12π, its volume is 4√3π -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 4 * π * r^2 = 12 * π → (4 / 3) * π * r^3 = 4 * Real.sqrt 3 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l241_24145


namespace NUMINAMATH_CALUDE_binomial_seven_four_l241_24126

theorem binomial_seven_four : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_seven_four_l241_24126


namespace NUMINAMATH_CALUDE_hundredth_digit_is_one_l241_24192

/-- The decimal representation of 7/33 has a repeating pattern of length 2 -/
def decimal_rep_period (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 
    (7 : ℚ) / 33 = (a * 10 + b : ℚ) / 100 + (7 : ℚ) / (33 * 100)

/-- The 100th digit after the decimal point in 7/33 -/
def hundredth_digit : ℕ :=
  sorry

theorem hundredth_digit_is_one :
  decimal_rep_period 2 → hundredth_digit = 1 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_digit_is_one_l241_24192


namespace NUMINAMATH_CALUDE_trig_identity_l241_24149

theorem trig_identity (α : Real) (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos (2 * α) + Real.sin (2 * α)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l241_24149


namespace NUMINAMATH_CALUDE_triangle_altitude_l241_24195

theorem triangle_altitude (a b : ℝ) (B : ℝ) (h : ℝ) : 
  a = 2 → b = Real.sqrt 7 → B = π / 3 → h = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_l241_24195


namespace NUMINAMATH_CALUDE_regular_polygon_with_36_degree_central_angle_l241_24136

theorem regular_polygon_with_36_degree_central_angle (n : ℕ) 
  (h : n > 0) 
  (central_angle : ℝ) 
  (h_central_angle : central_angle = 36) : 
  (360 : ℝ) / central_angle = 10 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_36_degree_central_angle_l241_24136


namespace NUMINAMATH_CALUDE_residue_mod_14_l241_24199

theorem residue_mod_14 : (182 * 12 - 15 * 7 + 3) % 14 = 10 := by sorry

end NUMINAMATH_CALUDE_residue_mod_14_l241_24199


namespace NUMINAMATH_CALUDE_prime_divisors_theorem_l241_24104

def f (p : ℕ) : ℕ := 3^p + 4^p + 5^p + 9^p - 98

theorem prime_divisors_theorem (p : ℕ) :
  Prime p ↔ (Nat.card (Nat.divisors (f p)) ≤ 6 ↔ p = 2 ∨ p = 3) := by sorry

end NUMINAMATH_CALUDE_prime_divisors_theorem_l241_24104


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l241_24123

-- Define the curve
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k + 2) - y^2 / (6 - 2*k) = 1

-- Define the range of k
def k_range (k : ℝ) : Prop := -2 < k ∧ k < 3

-- Theorem statement
theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k ↔ k_range k := by sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l241_24123


namespace NUMINAMATH_CALUDE_cosine_graph_shift_l241_24130

theorem cosine_graph_shift (x : ℝ) :
  4 * Real.cos (2 * (x - π/8) + π/4) = 4 * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_cosine_graph_shift_l241_24130


namespace NUMINAMATH_CALUDE_eggs_per_unit_is_twelve_l241_24155

/-- Represents the number of eggs in one unit -/
def eggs_per_unit : ℕ := 12

/-- Represents the number of units supplied to the first store daily -/
def units_to_first_store : ℕ := 5

/-- Represents the number of eggs supplied to the second store daily -/
def eggs_to_second_store : ℕ := 30

/-- Represents the total number of eggs supplied to both stores in a week -/
def total_eggs_per_week : ℕ := 630

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem stating that the number of eggs in one unit is 12 -/
theorem eggs_per_unit_is_twelve :
  eggs_per_unit * units_to_first_store * days_in_week +
  eggs_to_second_store * days_in_week = total_eggs_per_week :=
by sorry

end NUMINAMATH_CALUDE_eggs_per_unit_is_twelve_l241_24155


namespace NUMINAMATH_CALUDE_inequality_solution_l241_24158

theorem inequality_solution : 
  ∃! a : ℝ, a > 0 ∧ ∀ x > 0, (2 * x - 2 * a + Real.log (x / a)) * (-2 * x^2 + a * x + 5) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l241_24158


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l241_24184

theorem decimal_to_fraction :
  (3.75 : ℚ) = 15 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l241_24184


namespace NUMINAMATH_CALUDE_mack_writes_sixteen_pages_l241_24141

/-- Calculates the total number of pages Mack writes from Monday to Thursday -/
def total_pages (T1 R1 T2 R2 P3 T4 T5 R3 R4 : ℕ) : ℕ :=
  let monday_pages := T1 / R1
  let tuesday_pages := T2 / R2
  let wednesday_pages := P3
  let thursday_first_part := T5 / R3
  let thursday_second_part := (T4 - T5) / R4
  let thursday_pages := thursday_first_part + thursday_second_part
  monday_pages + tuesday_pages + wednesday_pages + thursday_pages

/-- Theorem stating that given the specified conditions, Mack writes 16 pages in total -/
theorem mack_writes_sixteen_pages :
  total_pages 60 30 45 15 5 90 30 10 20 = 16 := by
  sorry

end NUMINAMATH_CALUDE_mack_writes_sixteen_pages_l241_24141


namespace NUMINAMATH_CALUDE_rectangular_hall_dimension_difference_l241_24181

theorem rectangular_hall_dimension_difference 
  (length width : ℝ) 
  (width_half_length : width = length / 2)
  (area_constraint : length * width = 578) :
  length - width = 17 := by
sorry

end NUMINAMATH_CALUDE_rectangular_hall_dimension_difference_l241_24181


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l241_24197

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 3 * y = 1 → 1 / a + 1 / b ≤ 1 / x + 1 / y) ∧
  (1 / a + 1 / b = 4 + 2 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l241_24197


namespace NUMINAMATH_CALUDE_binomial_20_17_l241_24179

theorem binomial_20_17 : (Nat.choose 20 17) = 1140 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_17_l241_24179


namespace NUMINAMATH_CALUDE_exists_m_intersecting_line_and_circle_l241_24139

/-- A line intersects a circle if and only if the distance from the center of the circle to the line is less than the radius of the circle. -/
axiom line_intersects_circle_iff_distance_lt_radius {a b c x₀ y₀ r : ℝ} :
  (∃ x y, a * x + b * y + c = 0 ∧ (x - x₀)^2 + (y - y₀)^2 = r^2) ↔
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2) < r

/-- The theorem stating that there exists an integer m between 2 and 7 (exclusive) such that the line 4x + 3y + 2m = 0 intersects with the circle (x + 3)² + (y - 1)² = 1. -/
theorem exists_m_intersecting_line_and_circle :
  ∃ m : ℤ, 2 < m ∧ m < 7 ∧
  (∃ x y : ℝ, 4 * x + 3 * y + 2 * (m : ℝ) = 0 ∧ (x + 3)^2 + (y - 1)^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_m_intersecting_line_and_circle_l241_24139


namespace NUMINAMATH_CALUDE_cos_210_degrees_l241_24170

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l241_24170


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_difference_l241_24106

theorem quadratic_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (2 * x₁^2 - 7 * x₁ + 1 = x₁ + 31) ∧
  (2 * x₂^2 - 7 * x₂ + 1 = x₂ + 31) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 2 * Real.sqrt 19 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_difference_l241_24106


namespace NUMINAMATH_CALUDE_smallest_of_three_l241_24165

theorem smallest_of_three : 
  ∀ (x y z : ℝ), x = -Real.sqrt 2 ∧ y = 0 ∧ z = -1 → 
  x < y ∧ x < z := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_three_l241_24165


namespace NUMINAMATH_CALUDE_range_of_R_l241_24116

/-- The polar equation of curve C1 is ρ = R (R > 0) -/
def C1 (R : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = R^2 ∧ R > 0}

/-- The parametric equation of curve C2 is x = 2 + sin²α, y = sin²α -/
def C2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ α : ℝ, p.1 = 2 + Real.sin α ^ 2 ∧ p.2 = Real.sin α ^ 2}

/-- C1 and C2 have common points -/
def have_common_points (R : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ C1 R ∧ p ∈ C2

theorem range_of_R :
  ∀ R : ℝ, have_common_points R ↔ 2 ≤ R ∧ R ≤ Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_range_of_R_l241_24116


namespace NUMINAMATH_CALUDE_dance_event_relationship_l241_24150

/-- Represents a dance event with boys and girls. -/
structure DanceEvent where
  boys : ℕ
  girls : ℕ
  first_boy_dances : ℕ
  increment : ℕ

/-- The relationship between boys and girls in a specific dance event. -/
def dance_relationship (event : DanceEvent) : Prop :=
  event.boys = (event.girls - 4) / 2

/-- Theorem stating the relationship between boys and girls in the dance event. -/
theorem dance_event_relationship :
  ∀ (event : DanceEvent),
  event.first_boy_dances = 6 →
  event.increment = 2 →
  (∀ n : ℕ, n < event.boys → event.first_boy_dances + n * event.increment ≤ event.girls) →
  event.first_boy_dances + (event.boys - 1) * event.increment = event.girls →
  dance_relationship event :=
sorry

end NUMINAMATH_CALUDE_dance_event_relationship_l241_24150


namespace NUMINAMATH_CALUDE_equation_solution_l241_24172

theorem equation_solution : ∃ x : ℝ, x > 0 ∧ 5 * (x^(1/4))^2 - (3*x)/(x^(3/4)) = 10 + 2 * x^(1/4) ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l241_24172


namespace NUMINAMATH_CALUDE_delay_calculation_cottage_to_station_delay_l241_24107

theorem delay_calculation (usual_time : ℝ) (speed_increase : ℝ) (lateness : ℝ) : ℝ :=
  let normal_distance := usual_time
  let increased_speed_time := normal_distance / speed_increase
  let total_time := increased_speed_time - lateness
  usual_time - total_time

theorem cottage_to_station_delay : delay_calculation 18 1.2 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_delay_calculation_cottage_to_station_delay_l241_24107


namespace NUMINAMATH_CALUDE_exactly_three_combinations_l241_24196

/-- Represents the number of games played -/
def total_games : ℕ := 15

/-- Represents the total points scored -/
def total_points : ℕ := 33

/-- Represents the points earned for a win -/
def win_points : ℕ := 3

/-- Represents the points earned for a draw -/
def draw_points : ℕ := 1

/-- Represents the points earned for a loss -/
def loss_points : ℕ := 0

/-- A combination of wins, draws, and losses -/
structure GameCombination where
  wins : ℕ
  draws : ℕ
  losses : ℕ

/-- Checks if a combination is valid according to the given conditions -/
def is_valid_combination (c : GameCombination) : Prop :=
  c.wins + c.draws + c.losses = total_games ∧
  c.wins * win_points + c.draws * draw_points + c.losses * loss_points = total_points

/-- The theorem to be proved -/
theorem exactly_three_combinations :
  ∃! (combinations : List GameCombination),
    (∀ c ∈ combinations, is_valid_combination c) ∧
    combinations.length = 3 :=
sorry

end NUMINAMATH_CALUDE_exactly_three_combinations_l241_24196


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_2_l241_24151

/-- Hyperbola eccentricity theorem -/
theorem hyperbola_eccentricity_sqrt_2 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (asymptote_eq : ∀ x, b / a * x = x)
  (F2 : ℝ × ℝ)
  (hF2 : F2 = (c, 0))
  (M : ℝ × ℝ)
  (hM : M.1 = 0)
  (N : ℝ × ℝ)
  (perpendicular : (M.2 - N.2) * (b / a) = -(M.1 - N.1))
  (midpoint : N = ((F2.1 + M.1) / 2, (F2.2 + M.2) / 2))
  : c / a = Real.sqrt 2 := by
  sorry

#check hyperbola_eccentricity_sqrt_2

end NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_2_l241_24151


namespace NUMINAMATH_CALUDE_tangent_lines_with_equal_intercepts_l241_24163

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0

-- Define a tangent line
def is_tangent_line (a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_C x y ∧ a*x + b*y + c = 0 ∧
  ∀ (x' y' : ℝ), circle_C x' y' → a*x' + b*y' + c ≥ 0

-- Define the condition for equal absolute intercepts
def equal_abs_intercepts (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ |c/a| = |c/b|

-- Theorem statement
theorem tangent_lines_with_equal_intercepts :
  ∀ (a b c : ℝ),
    is_tangent_line a b c ∧ equal_abs_intercepts a b c →
    ((a = 1 ∧ b = 1 ∧ c = -3) ∨
     (a = 1 ∧ b = 1 ∧ c = 1) ∨
     (a = 1 ∧ b = -1 ∧ c = -5) ∨
     (a = 1 ∧ b = -1 ∧ c = -1) ∨
     (∃ k : ℝ, k^2 = 10 ∧ a = k ∧ b = -1 ∧ c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_with_equal_intercepts_l241_24163


namespace NUMINAMATH_CALUDE_sum_of_product_of_roots_l241_24142

theorem sum_of_product_of_roots (p q r : ℂ) : 
  (4 * p^3 - 8 * p^2 + 16 * p - 12 = 0) ∧ 
  (4 * q^3 - 8 * q^2 + 16 * q - 12 = 0) ∧ 
  (4 * r^3 - 8 * r^2 + 16 * r - 12 = 0) →
  p * q + q * r + r * p = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_product_of_roots_l241_24142


namespace NUMINAMATH_CALUDE_rectangle_side_difference_l241_24157

theorem rectangle_side_difference (A d : ℝ) (h_A : A > 0) (h_d : d > 0) :
  ∃ x y : ℝ, x > y ∧ x * y = A ∧ x^2 + y^2 = d^2 ∧ x - y = Real.sqrt (d^2 - 4 * A) :=
sorry

end NUMINAMATH_CALUDE_rectangle_side_difference_l241_24157


namespace NUMINAMATH_CALUDE_horses_added_correct_horses_added_l241_24169

theorem horses_added (initial_horses : ℕ) (drinking_water : ℕ) (bathing_water : ℕ) 
  (total_days : ℕ) (total_water : ℕ) : ℕ :=
  let water_per_horse := drinking_water + bathing_water
  let initial_daily_water := initial_horses * water_per_horse
  let initial_total_water := initial_daily_water * total_days
  let new_horses_water := total_water - initial_total_water
  let new_horses_daily_water := new_horses_water / total_days
  new_horses_daily_water / water_per_horse

theorem correct_horses_added :
  horses_added 3 5 2 28 1568 = 5 := by
  sorry

end NUMINAMATH_CALUDE_horses_added_correct_horses_added_l241_24169


namespace NUMINAMATH_CALUDE_f_not_monotonic_iff_l241_24122

noncomputable def f (x : ℝ) : ℝ := x^2 - (1/2) * Real.log x + 1

def is_not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a < x ∧ x < y ∧ y < z ∧ z < b ∧
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

theorem f_not_monotonic_iff (k : ℝ) :
  is_not_monotonic f (k - 1) (k + 1) ↔ 1 ≤ k ∧ k < 3/2 := by sorry

end NUMINAMATH_CALUDE_f_not_monotonic_iff_l241_24122


namespace NUMINAMATH_CALUDE_a_value_l241_24146

def U : Set ℤ := {3, 4, 5}

def M (a : ℤ) : Set ℤ := {|a - 3|, 3}

theorem a_value (a : ℤ) (h : (U \ M a) = {5}) : a = -1 ∨ a = 7 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l241_24146
