import Mathlib

namespace NUMINAMATH_CALUDE_otimes_nested_l1377_137789

/-- Custom binary operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^2 - 2*y

/-- Theorem stating the result of k ⊗ (k ⊗ k) -/
theorem otimes_nested (k : ℝ) : otimes k (otimes k k) = -k^2 + 4*k := by
  sorry

end NUMINAMATH_CALUDE_otimes_nested_l1377_137789


namespace NUMINAMATH_CALUDE_sin_two_pi_thirds_l1377_137769

theorem sin_two_pi_thirds : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_pi_thirds_l1377_137769


namespace NUMINAMATH_CALUDE_zero_point_of_f_l1377_137752

def f (x : ℝ) : ℝ := x + 1

theorem zero_point_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_zero_point_of_f_l1377_137752


namespace NUMINAMATH_CALUDE_total_trees_l1377_137733

/-- Represents the tree-planting task for a school --/
structure TreePlantingTask where
  total : ℕ
  ninth_grade : ℕ
  eighth_grade : ℕ
  seventh_grade : ℕ

/-- The conditions of the tree-planting task --/
def tree_planting_conditions (a : ℕ) (task : TreePlantingTask) : Prop :=
  task.ninth_grade = task.total / 2 ∧
  task.eighth_grade = (task.total - task.ninth_grade) * 2 / 3 ∧
  task.seventh_grade = a ∧
  task.total = task.ninth_grade + task.eighth_grade + task.seventh_grade

/-- The theorem stating that the total number of trees is 6a --/
theorem total_trees (a : ℕ) (task : TreePlantingTask) 
  (h : tree_planting_conditions a task) : task.total = 6 * a :=
sorry

end NUMINAMATH_CALUDE_total_trees_l1377_137733


namespace NUMINAMATH_CALUDE_sqrt_cos_squared_660_l1377_137724

theorem sqrt_cos_squared_660 : Real.sqrt (Real.cos (660 * π / 180) ^ 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_cos_squared_660_l1377_137724


namespace NUMINAMATH_CALUDE_handshake_count_l1377_137745

theorem handshake_count (n : ℕ) (h : n = 8) : 
  (n * (n - 2)) / 2 = 24 := by
  sorry

#check handshake_count

end NUMINAMATH_CALUDE_handshake_count_l1377_137745


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1377_137764

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | (1 - a) * x^2 - 2 * x + 1 < 0}
  if a > 1 then
    S = {x : ℝ | x < (1 - Real.sqrt a) / (a - 1) ∨ x > (1 + Real.sqrt a) / (a - 1)}
  else if a = 1 then
    S = {x : ℝ | x > 1 / 2}
  else if 0 < a ∧ a < 1 then
    S = {x : ℝ | (1 - Real.sqrt a) / (1 - a) < x ∧ x < (1 + Real.sqrt a) / (1 - a)}
  else
    S = ∅ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1377_137764


namespace NUMINAMATH_CALUDE_problem_solution_l1377_137729

-- Define the sets A, B, and C
def A : Set ℝ := {x | -2 < x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 3 ∨ x < 2}
def C (a : ℝ) : Set ℝ := {x | x < 2 * a + 1}

-- State the theorem
theorem problem_solution :
  (∃ a : ℝ, B ∩ C a = C a) →
  ((A ∩ B = {x : ℝ | -2 < x ∧ x < 2}) ∧
   (∃ a : ℝ, ∀ x : ℝ, x ≤ 1/2 ↔ B ∩ C x = C x)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1377_137729


namespace NUMINAMATH_CALUDE_area_difference_l1377_137751

/-- The difference in area between a square and a rectangle -/
theorem area_difference (square_side : ℝ) (rect_length rect_width : ℝ) : 
  square_side = 5 → rect_length = 3 → rect_width = 6 → 
  square_side * square_side - rect_length * rect_width = 7 := by
  sorry

#check area_difference

end NUMINAMATH_CALUDE_area_difference_l1377_137751


namespace NUMINAMATH_CALUDE_star_op_equation_has_two_distinct_real_roots_l1377_137718

/-- Custom operation ※ -/
def star_op (a b : ℝ) : ℝ := a^2 * b + a * b - 1

/-- Theorem stating that x※1 = 0 has two distinct real roots -/
theorem star_op_equation_has_two_distinct_real_roots :
  ∃ x y : ℝ, x ≠ y ∧ star_op x 1 = 0 ∧ star_op y 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_op_equation_has_two_distinct_real_roots_l1377_137718


namespace NUMINAMATH_CALUDE_coefficient_of_3x_squared_l1377_137717

/-- Definition of a coefficient in a monomial term -/
def coefficient (term : ℝ → ℝ) : ℝ :=
  term 1

/-- The term 3x^2 -/
def term (x : ℝ) : ℝ := 3 * x^2

/-- Theorem: The coefficient of 3x^2 is 3 -/
theorem coefficient_of_3x_squared :
  coefficient term = 3 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_3x_squared_l1377_137717


namespace NUMINAMATH_CALUDE_license_plate_count_l1377_137710

/-- The set of odd single-digit numbers -/
def oddDigits : Finset Nat := {1, 3, 5, 7, 9}

/-- The set of prime numbers less than 10 -/
def primesLessThan10 : Finset Nat := {2, 3, 5, 7}

/-- The set of even single-digit numbers -/
def evenDigits : Finset Nat := {0, 2, 4, 6, 8}

/-- The number of letters in the alphabet -/
def alphabetSize : Nat := 26

theorem license_plate_count :
  (alphabetSize ^ 2) * oddDigits.card * primesLessThan10.card * evenDigits.card = 67600 := by
  sorry

#eval (alphabetSize ^ 2) * oddDigits.card * primesLessThan10.card * evenDigits.card

end NUMINAMATH_CALUDE_license_plate_count_l1377_137710


namespace NUMINAMATH_CALUDE_total_cellphones_sold_l1377_137757

/-- Calculates the number of cell phones sold given initial and final inventories and damaged/defective phones. -/
def cellphonesSold (initialSamsung : ℕ) (finalSamsung : ℕ) (initialIphone : ℕ) (finalIphone : ℕ) (damagedSamsung : ℕ) (defectiveIphone : ℕ) : ℕ :=
  (initialSamsung - damagedSamsung - finalSamsung) + (initialIphone - defectiveIphone - finalIphone)

/-- Proves that the total number of cell phones sold is 4 given the inventory information. -/
theorem total_cellphones_sold :
  cellphonesSold 14 10 8 5 2 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_cellphones_sold_l1377_137757


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l1377_137732

def arithmetic_sequence_sum (a : ℕ) (d : ℕ) (last : ℕ) : ℕ :=
  let n := (last - a) / d + 1
  n * (a + last) / 2

theorem arithmetic_sequence_sum_remainder
  (h1 : arithmetic_sequence_sum 3 6 273 % 6 = 0) : 
  arithmetic_sequence_sum 3 6 273 % 6 = 0 := by
  sorry

#check arithmetic_sequence_sum_remainder

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l1377_137732


namespace NUMINAMATH_CALUDE_counterexample_existence_l1377_137776

theorem counterexample_existence : ∃ (S : Finset ℝ), 
  (Finset.card S = 25) ∧ 
  (∀ (a b c : ℝ), a ∈ S → b ∈ S → c ∈ S → 
    ∃ (d : ℝ), d ∈ S ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c ∧ a + b + c + d > 0) ∧
  (Finset.sum S id ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_existence_l1377_137776


namespace NUMINAMATH_CALUDE_essay_section_length_l1377_137762

theorem essay_section_length 
  (intro_length : ℕ) 
  (conclusion_length : ℕ) 
  (body_sections : ℕ) 
  (total_length : ℕ) :
  intro_length = 450 →
  conclusion_length = 3 * intro_length →
  body_sections = 4 →
  total_length = 5000 →
  (total_length - (intro_length + conclusion_length)) / body_sections = 800 :=
by
  sorry

end NUMINAMATH_CALUDE_essay_section_length_l1377_137762


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l1377_137711

-- Define the sum of the first n terms of an arithmetic sequence
def T (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 5) / 2

-- State the theorem
theorem arithmetic_sequence_constant_ratio (a : ℚ) :
  (∀ n : ℕ, n > 0 → ∃ k : ℚ, T a (4 * n) / T a n = k) →
  a = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l1377_137711


namespace NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l1377_137799

theorem other_root_of_complex_quadratic (z : ℂ) :
  z^2 = -20 + 15*I ∧ (4 + 3*I)^2 = -20 + 15*I →
  (-4 - 3*I)^2 = -20 + 15*I :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l1377_137799


namespace NUMINAMATH_CALUDE_basketball_probability_l1377_137778

theorem basketball_probability (p : ℝ) (n : ℕ) (h1 : p = 1/3) (h2 : n = 3) :
  (1 - p)^n + n * p * (1 - p)^(n-1) = 20/27 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probability_l1377_137778


namespace NUMINAMATH_CALUDE_modular_inverse_13_mod_101_l1377_137705

theorem modular_inverse_13_mod_101 : ∃ x : ℕ, x ≤ 100 ∧ (13 * x) % 101 = 1 :=
by
  use 70
  sorry

end NUMINAMATH_CALUDE_modular_inverse_13_mod_101_l1377_137705


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1377_137795

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1377_137795


namespace NUMINAMATH_CALUDE_min_value_of_S_l1377_137737

theorem min_value_of_S (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_S_l1377_137737


namespace NUMINAMATH_CALUDE_total_prime_factors_l1377_137755

-- Define the expression
def expression := (4 : ℕ) ^ 13 * 7 ^ 5 * 11 ^ 2

-- Define the prime factorization of 4
axiom four_eq_two_squared : (4 : ℕ) = 2 ^ 2

-- Define 7 and 11 as prime numbers
axiom seven_prime : Nat.Prime 7
axiom eleven_prime : Nat.Prime 11

-- Theorem statement
theorem total_prime_factors : 
  (Nat.factors expression).length = 33 :=
sorry

end NUMINAMATH_CALUDE_total_prime_factors_l1377_137755


namespace NUMINAMATH_CALUDE_binary_representation_properties_l1377_137754

def has_exactly_three_ones (n : ℕ) : Prop :=
  (n.digits 2).count 1 = 3

def is_multiple_of_617 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 617 * k

theorem binary_representation_properties (n : ℕ) 
  (h1 : is_multiple_of_617 n) 
  (h2 : has_exactly_three_ones n) : 
  ((n.digits 2).length ≥ 9) ∧ 
  ((n.digits 2).length = 10 → Even n) :=
sorry

end NUMINAMATH_CALUDE_binary_representation_properties_l1377_137754


namespace NUMINAMATH_CALUDE_correct_answer_probability_l1377_137706

/-- The probability of correctly answering one MCQ question with 3 options -/
def mcq_prob : ℚ := 1 / 3

/-- The probability of correctly answering a true/false question -/
def tf_prob : ℚ := 1 / 2

/-- The number of true/false questions -/
def num_tf_questions : ℕ := 2

/-- The probability of correctly answering all questions -/
def total_prob : ℚ := mcq_prob * tf_prob ^ num_tf_questions

theorem correct_answer_probability :
  total_prob = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_correct_answer_probability_l1377_137706


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1377_137744

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 3
  let θ : ℝ := 3 * π / 2
  let φ : ℝ := π / 3
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) := by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1377_137744


namespace NUMINAMATH_CALUDE_min_value_perpendicular_vectors_l1377_137765

theorem min_value_perpendicular_vectors (x y : ℝ) :
  let m : ℝ × ℝ := (x - 1, 1)
  let n : ℝ × ℝ := (1, y)
  (m.1 * n.1 + m.2 * n.2 = 0) →
  (∀ a b : ℝ, m = (a - 1, 1) ∧ n = (1, b) → 2^a + 2^b ≥ 2 * Real.sqrt 2) ∧
  (∃ a b : ℝ, m = (a - 1, 1) ∧ n = (1, b) ∧ 2^a + 2^b = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_perpendicular_vectors_l1377_137765


namespace NUMINAMATH_CALUDE_power_six_expression_l1377_137746

theorem power_six_expression (m n : ℕ) (P Q : ℕ) 
  (h1 : P = 2^m) (h2 : Q = 5^n) : 
  6^(m+n) = P * 2^n * 3^(m+n) := by
  sorry

end NUMINAMATH_CALUDE_power_six_expression_l1377_137746


namespace NUMINAMATH_CALUDE_man_swimming_speed_l1377_137796

/-- The swimming speed of a man in still water, given that it takes him twice as long to swim upstream
    than downstream in a stream with a speed of 1 km/h. -/
def swimming_speed : ℝ := 2

/-- The speed of the stream in km/h. -/
def stream_speed : ℝ := 1

/-- The time ratio of swimming upstream to downstream. -/
def upstream_downstream_ratio : ℝ := 2

theorem man_swimming_speed :
  swimming_speed = 2 ∧
  stream_speed = 1 ∧
  upstream_downstream_ratio = 2 →
  swimming_speed + stream_speed = upstream_downstream_ratio * (swimming_speed - stream_speed) :=
by sorry

end NUMINAMATH_CALUDE_man_swimming_speed_l1377_137796


namespace NUMINAMATH_CALUDE_probability_three_tails_l1377_137731

/-- The probability of getting exactly k successes in n trials
    with probability p for each trial -/
def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The number of coin flips -/
def numFlips : ℕ := 8

/-- The probability of getting tails -/
def probTails : ℚ := 2/3

/-- The number of tails we're interested in -/
def numTails : ℕ := 3

theorem probability_three_tails :
  binomialProbability numFlips numTails probTails = 448/6561 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_tails_l1377_137731


namespace NUMINAMATH_CALUDE_max_value_when_m_neg_four_range_of_m_for_condition_l1377_137727

-- Define the function f
def f (x m : ℝ) : ℝ := x - |x + 2| - |x - 3| - m

-- Theorem for part I
theorem max_value_when_m_neg_four :
  ∃ (x_max : ℝ), ∀ (x : ℝ), f x (-4) ≤ f x_max (-4) ∧ f x_max (-4) = 2 :=
sorry

-- Theorem for part II
theorem range_of_m_for_condition (m : ℝ) :
  (∃ (x₀ : ℝ), f x₀ m ≥ 1 / m - 4) ↔ m ∈ Set.Ioi 0 ∪ {1} :=
sorry

end NUMINAMATH_CALUDE_max_value_when_m_neg_four_range_of_m_for_condition_l1377_137727


namespace NUMINAMATH_CALUDE_constant_phi_forms_cone_l1377_137794

/-- Represents a point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Defines the set of points satisfying φ = d -/
def ConstantPhiSet (d : ℝ) : Set SphericalPoint :=
  {p : SphericalPoint | p.φ = d}

/-- Theorem: The set of points with constant φ forms a cone -/
theorem constant_phi_forms_cone (d : ℝ) :
  ∃ (cone : Set SphericalPoint), ConstantPhiSet d = cone :=
sorry

end NUMINAMATH_CALUDE_constant_phi_forms_cone_l1377_137794


namespace NUMINAMATH_CALUDE_oranges_remaining_proof_l1377_137728

/-- The number of oranges Michaela needs to eat until she gets full -/
def michaela_oranges : ℕ := 30

/-- The number of oranges Cassandra needs to eat until she gets full -/
def cassandra_oranges : ℕ := 3 * michaela_oranges

/-- The total number of oranges picked from the farm -/
def total_oranges : ℕ := 200

/-- The number of oranges remaining after Michaela and Cassandra eat until they're full -/
def remaining_oranges : ℕ := total_oranges - (michaela_oranges + cassandra_oranges)

theorem oranges_remaining_proof : remaining_oranges = 80 := by
  sorry

end NUMINAMATH_CALUDE_oranges_remaining_proof_l1377_137728


namespace NUMINAMATH_CALUDE_jorge_total_goals_l1377_137773

/-- The total number of goals Jorge scored over two seasons -/
def total_goals (last_season_goals this_season_goals : ℕ) : ℕ :=
  last_season_goals + this_season_goals

/-- Theorem stating that Jorge's total goals over two seasons is 343 -/
theorem jorge_total_goals :
  total_goals 156 187 = 343 := by
  sorry

end NUMINAMATH_CALUDE_jorge_total_goals_l1377_137773


namespace NUMINAMATH_CALUDE_two_unique_pairs_for_15_l1377_137707

/-- The number of unique pairs of nonnegative integers (a, b) satisfying a^2 - b^2 = n, for n = 15 -/
def uniquePairsCount (n : Nat) : Nat :=
  (Finset.filter (fun p : Nat × Nat => 
    p.1 ≥ p.2 ∧ p.1^2 - p.2^2 = n) (Finset.product (Finset.range (n+1)) (Finset.range (n+1)))).card

/-- Theorem stating that there are exactly 2 unique pairs for n = 15 -/
theorem two_unique_pairs_for_15 : uniquePairsCount 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_unique_pairs_for_15_l1377_137707


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1377_137761

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 3) : 
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1377_137761


namespace NUMINAMATH_CALUDE_focal_distance_l1377_137743

/-- Represents an ellipse with focal points and vertices -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Focal distance from center

/-- Properties of the ellipse based on given conditions -/
def ellipse_properties (e : Ellipse) : Prop :=
  e.a - e.c = 1.5 ∧  -- |F₂A| = 1.5
  2 * e.a = 5.4 ∧    -- |BC| = 5.4
  e.a^2 = e.b^2 + e.c^2

/-- Theorem stating the distance between focal points -/
theorem focal_distance (e : Ellipse) 
  (h : ellipse_properties e) : 2 * e.a - (e.a - e.c) = 13.5 := by
  sorry

#check focal_distance

end NUMINAMATH_CALUDE_focal_distance_l1377_137743


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_ge_negation_of_quadratic_inequality_l1377_137766

theorem negation_of_existence (P : ℕ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem negation_of_inequality_ge (a b : ℝ) :
  (¬ (a ≥ b)) ↔ (a < b) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℕ, x^2 + 2*x ≥ 3) ↔ (∀ x : ℕ, x^2 + 2*x < 3) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_ge_negation_of_quadratic_inequality_l1377_137766


namespace NUMINAMATH_CALUDE_cosine_power_relation_l1377_137786

theorem cosine_power_relation (z : ℂ) (α : ℝ) (h : z + 1/z = 2 * Real.cos α) :
  ∀ n : ℕ, z^n + 1/z^n = 2 * Real.cos (n * α) := by
  sorry

end NUMINAMATH_CALUDE_cosine_power_relation_l1377_137786


namespace NUMINAMATH_CALUDE_square_roots_of_nine_l1377_137714

theorem square_roots_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_square_roots_of_nine_l1377_137714


namespace NUMINAMATH_CALUDE_book_sale_gain_percentage_l1377_137748

/-- Represents the problem of calculating the gain percentage on a book sale --/
theorem book_sale_gain_percentage 
  (total_cost : ℝ) 
  (cost_book1 : ℝ) 
  (loss_percentage : ℝ) 
  (total_cost_eq : total_cost = 360) 
  (cost_book1_eq : cost_book1 = 210) 
  (loss_percentage_eq : loss_percentage = 15) 
  (cost_book2_eq : total_cost = cost_book1 + cost_book2) 
  (same_selling_price : 
    cost_book1 * (1 - loss_percentage / 100) = 
    cost_book2 * (1 + gain_percentage / 100)) : 
  gain_percentage = 19 := by sorry

end NUMINAMATH_CALUDE_book_sale_gain_percentage_l1377_137748


namespace NUMINAMATH_CALUDE_right_triangle_345_l1377_137712

def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

theorem right_triangle_345 :
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle (Real.sqrt 3) (Real.sqrt 4) (Real.sqrt 5) ∧
  ¬ is_right_triangle 4 6 9 ∧
  is_right_triangle 3 4 5 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_345_l1377_137712


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l1377_137709

/-- Given a rectangular plot where the length is thrice the breadth and the breadth is 15 meters,
    prove that the area of the plot is 675 square meters. -/
theorem rectangular_plot_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 15 →
  length = 3 * breadth →
  area = length * breadth →
  area = 675 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l1377_137709


namespace NUMINAMATH_CALUDE_least_possible_y_l1377_137782

theorem least_possible_y (x y z : ℤ) 
  (h_x_even : Even x)
  (h_y_odd : Odd y)
  (h_z_odd : Odd z)
  (h_y_minus_x : y - x > 5)
  (h_z_minus_x : ∀ w : ℤ, (Odd w ∧ w - x ≥ 9) → z - x ≤ w - x) : 
  y ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_y_l1377_137782


namespace NUMINAMATH_CALUDE_odd_divisor_of_power_plus_one_l1377_137726

theorem odd_divisor_of_power_plus_one (n : ℕ) :
  n > 0 ∧ Odd n ∧ n ∣ (3^n + 1) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_divisor_of_power_plus_one_l1377_137726


namespace NUMINAMATH_CALUDE_infinitely_many_primes_6n_plus_5_l1377_137763

theorem infinitely_many_primes_6n_plus_5 : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℕ, p = 6 * n + 5} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_6n_plus_5_l1377_137763


namespace NUMINAMATH_CALUDE_least_square_tiles_l1377_137771

def room_length : ℕ := 720
def room_width : ℕ := 432

theorem least_square_tiles (l w : ℕ) (h1 : l = room_length) (h2 : w = room_width) :
  ∃ (tile_size : ℕ), 
    tile_size > 0 ∧
    l % tile_size = 0 ∧ 
    w % tile_size = 0 ∧
    (l / tile_size) * (w / tile_size) = 15 ∧
    ∀ (other_size : ℕ), 
      (other_size > 0 ∧ l % other_size = 0 ∧ w % other_size = 0) →
      (l / other_size) * (w / other_size) ≥ 15 := by
  sorry

#check least_square_tiles

end NUMINAMATH_CALUDE_least_square_tiles_l1377_137771


namespace NUMINAMATH_CALUDE_quadratic_always_positive_range_l1377_137702

theorem quadratic_always_positive_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_range_l1377_137702


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l1377_137722

theorem existence_of_special_sequence :
  ∃ (a : ℕ → ℕ),
    (∀ k, a k < a (k + 1)) ∧
    (∀ n : ℤ, ∃ N : ℕ, ∀ k > N, ¬ Prime (a k + n)) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l1377_137722


namespace NUMINAMATH_CALUDE_blue_pen_cost_is_ten_cents_l1377_137768

/-- The cost of a blue pen given the conditions of Maci's pen purchase. -/
def blue_pen_cost (blue_pens red_pens : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (blue_pens + 2 * red_pens)

/-- Theorem stating that the cost of a blue pen is $0.10 under the given conditions. -/
theorem blue_pen_cost_is_ten_cents :
  blue_pen_cost 10 15 4 = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_blue_pen_cost_is_ten_cents_l1377_137768


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1377_137791

theorem complex_equation_sum (x y : ℝ) :
  (x - Complex.I) * Complex.I = y + 2 * Complex.I →
  x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1377_137791


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1377_137747

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the point of tangency
def point : ℝ × ℝ := (1, 3)

-- State the theorem
theorem tangent_line_equation :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, y = f x → (a * x + b * y + c = 0 ↔ (x, y) = point ∨ 
      ∃ h > 0, ∀ t : ℝ, 0 < |t| → |t| < h → 
        (a * (point.1 + t) + b * f (point.1 + t) + c) * (a * point.1 + b * point.2 + c) > 0)) ∧
    a = 2 ∧ b = -1 ∧ c = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1377_137747


namespace NUMINAMATH_CALUDE_log_ratio_squared_l1377_137777

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1) 
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log y) (h2 : x * y = 243) : 
  (Real.log (x / y) / Real.log 3)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l1377_137777


namespace NUMINAMATH_CALUDE_guessing_game_score_sum_l1377_137750

/-- The guessing game score problem -/
theorem guessing_game_score_sum :
  ∀ (hajar_score farah_score : ℕ),
  hajar_score = 24 →
  farah_score - hajar_score = 21 →
  farah_score > hajar_score →
  hajar_score + farah_score = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_guessing_game_score_sum_l1377_137750


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1377_137790

theorem solution_set_quadratic_inequality :
  let f : ℝ → ℝ := λ x => -x^2 + 3*x - 2
  {x : ℝ | f x ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by
sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1377_137790


namespace NUMINAMATH_CALUDE_equation_value_l1377_137792

theorem equation_value (x y : ℝ) (h : x - 3*y = 4) : 
  (x - 3*y)^2 + 2*x - 6*y - 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_equation_value_l1377_137792


namespace NUMINAMATH_CALUDE_f_geq_1_solution_set_g_max_value_l1377_137719

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Define the function g(x)
def g (x : ℝ) : ℝ := f x - x^2 + x

theorem f_geq_1_solution_set (x : ℝ) :
  f x ≥ 1 ↔ x ≥ 1 := by sorry

theorem g_max_value :
  ∃ x₀ : ℝ, ∀ x : ℝ, g x ≤ g x₀ ∧ g x₀ = 5/4 := by sorry

end NUMINAMATH_CALUDE_f_geq_1_solution_set_g_max_value_l1377_137719


namespace NUMINAMATH_CALUDE_height_estimate_correct_l1377_137787

/-- Represents the regression line for student height based on foot length -/
structure HeightRegression where
  n : ℕ              -- number of students in the sample
  sum_x : ℝ          -- sum of foot lengths
  sum_y : ℝ          -- sum of heights
  slope : ℝ          -- slope of the regression line
  intercept : ℝ      -- y-intercept of the regression line

/-- Calculates the estimated height for a given foot length -/
def estimate_height (reg : HeightRegression) (x : ℝ) : ℝ :=
  reg.slope * x + reg.intercept

/-- Theorem stating that the estimated height for a foot length of 24 cm is 166 cm -/
theorem height_estimate_correct (reg : HeightRegression) : 
  reg.n = 10 ∧ 
  reg.sum_x = 225 ∧ 
  reg.sum_y = 1600 ∧ 
  reg.slope = 4 ∧
  reg.intercept = reg.sum_y / reg.n - reg.slope * (reg.sum_x / reg.n) →
  estimate_height reg 24 = 166 := by
  sorry

end NUMINAMATH_CALUDE_height_estimate_correct_l1377_137787


namespace NUMINAMATH_CALUDE_prob_no_match_three_picks_correct_l1377_137725

/-- The probability of not having a matching pair after 3 picks from 3 pairs of socks -/
def prob_no_match_three_picks : ℚ := 2 / 5

/-- The number of pairs of socks -/
def num_pairs : ℕ := 3

/-- The total number of socks -/
def total_socks : ℕ := 2 * num_pairs

/-- The probability of picking a non-matching sock on the second draw -/
def prob_second_draw : ℚ := 4 / 5

/-- The probability of picking a non-matching sock on the third draw -/
def prob_third_draw : ℚ := 1 / 2

theorem prob_no_match_three_picks_correct :
  prob_no_match_three_picks = prob_second_draw * prob_third_draw :=
sorry

end NUMINAMATH_CALUDE_prob_no_match_three_picks_correct_l1377_137725


namespace NUMINAMATH_CALUDE_square_of_two_minus_x_l1377_137788

theorem square_of_two_minus_x (x : ℝ) : (2 - x)^2 = 4 - 4*x + x^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_two_minus_x_l1377_137788


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1377_137774

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_roots : a 2 ^ 2 - 13 * a 2 + 14 = 0 ∧ a 10 ^ 2 - 13 * a 10 + 14 = 0) :
  a 6 = Real.sqrt 14 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1377_137774


namespace NUMINAMATH_CALUDE_fraction_equality_l1377_137715

theorem fraction_equality (a : ℕ+) :
  (a : ℚ) / (a + 45 : ℚ) = 3 / 4 → a = 135 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1377_137715


namespace NUMINAMATH_CALUDE_workshop_average_salary_l1377_137793

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (technician_avg_salary : ℚ)
  (other_avg_salary : ℚ)
  (h1 : total_workers = 21)
  (h2 : technicians = 7)
  (h3 : technician_avg_salary = 12000)
  (h4 : other_avg_salary = 6000) :
  (technicians * technician_avg_salary + (total_workers - technicians) * other_avg_salary) / total_workers = 8000 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l1377_137793


namespace NUMINAMATH_CALUDE_smallest_integer_l1377_137780

theorem smallest_integer (a b : ℕ) (ha : a = 60) (hb : b > 0) 
  (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 60) : 
  (∀ c : ℕ, c > 0 ∧ c < b → ¬(Nat.lcm a c / Nat.gcd a c = 60)) → b = 16 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_l1377_137780


namespace NUMINAMATH_CALUDE_fibonacci_sum_odd_equals_next_l1377_137735

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def sum_odd_fibonacci (n : ℕ) : ℕ :=
  1 + (List.range n).foldl (λ acc i => acc + fibonacci (2 * i + 3)) 0

theorem fibonacci_sum_odd_equals_next (n : ℕ) :
  sum_odd_fibonacci n = fibonacci (2 * n + 2) := by
  sorry

#eval fibonacci 2018
#eval sum_odd_fibonacci 1008

end NUMINAMATH_CALUDE_fibonacci_sum_odd_equals_next_l1377_137735


namespace NUMINAMATH_CALUDE_max_true_statements_l1377_137741

theorem max_true_statements (a b : ℝ) : 
  ¬(∃ (s1 s2 s3 s4 : Bool), 
    (s1 → 1/a > 1/b) ∧
    (s2 → abs a > abs b) ∧
    (s3 → a > b) ∧
    (s4 → a < 0) ∧
    (¬s1 ∨ ¬s2 ∨ ¬s3 ∨ ¬s4 → b > 0) ∧
    s1 ∧ s2 ∧ s3 ∧ s4) :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l1377_137741


namespace NUMINAMATH_CALUDE_arithmetic_contains_geometric_l1377_137738

-- Define the arithmetic progression
def arithmetic_progression (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Theorem statement
theorem arithmetic_contains_geometric (a d : ℕ) (h : d > 0) :
  ∃ (r : ℕ) (b : ℕ → ℕ), 
    (∀ n : ℕ, ∃ k : ℕ, b n = arithmetic_progression a d k) ∧
    (∀ n : ℕ, b (n + 1) = r * b n) ∧
    (∀ n : ℕ, b n > 0) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_contains_geometric_l1377_137738


namespace NUMINAMATH_CALUDE_banana_pie_angle_l1377_137700

theorem banana_pie_angle (total : ℕ) (chocolate apple blueberry : ℕ) :
  total = 48 →
  chocolate = 15 →
  apple = 10 →
  blueberry = 9 →
  let remaining := total - (chocolate + apple + blueberry)
  let banana := remaining / 2
  (banana : ℝ) / total * 360 = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_banana_pie_angle_l1377_137700


namespace NUMINAMATH_CALUDE_three_integer_chords_l1377_137781

/-- A circle with a point P inside --/
structure CircleWithPoint where
  radius : ℝ
  distanceToCenter : ℝ

/-- Count of integer-length chords through P --/
def integerChordCount (c : CircleWithPoint) : ℕ :=
  sorry

theorem three_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 12)
  (h2 : c.distanceToCenter = 5) : 
  integerChordCount c = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_integer_chords_l1377_137781


namespace NUMINAMATH_CALUDE_clock_gains_five_minutes_per_hour_l1377_137713

/-- A clock that gains time -/
structure GainingClock where
  start_time : ℕ  -- Start time in hours (24-hour format)
  end_time : ℕ    -- End time in hours (24-hour format)
  total_gain : ℕ  -- Total minutes gained

/-- Calculate the minutes gained per hour -/
def minutes_gained_per_hour (clock : GainingClock) : ℚ :=
  clock.total_gain / (clock.end_time - clock.start_time)

/-- Theorem: A clock that starts at 9 a.m. and gains 45 minutes by 6 p.m. gains 5 minutes per hour -/
theorem clock_gains_five_minutes_per_hour (clock : GainingClock) 
    (h1 : clock.start_time = 9)
    (h2 : clock.end_time = 18)
    (h3 : clock.total_gain = 45) :
  minutes_gained_per_hour clock = 5 := by
  sorry

end NUMINAMATH_CALUDE_clock_gains_five_minutes_per_hour_l1377_137713


namespace NUMINAMATH_CALUDE_sampling_methods_correct_l1377_137783

-- Define the sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

-- Define a scenario
structure Scenario where
  total_population : ℕ
  sample_size : ℕ
  has_distinct_groups : Bool
  is_ordered : Bool

-- Define a function to determine the most suitable sampling method
def most_suitable_method (s : Scenario) : SamplingMethod :=
  if s.total_population ≤ 15 then SamplingMethod.SimpleRandom
  else if s.has_distinct_groups then SamplingMethod.Stratified
  else if s.is_ordered then SamplingMethod.Systematic
  else SamplingMethod.SimpleRandom

-- Theorem to prove
theorem sampling_methods_correct :
  (most_suitable_method ⟨15, 5, false, false⟩ = SamplingMethod.SimpleRandom) ∧
  (most_suitable_method ⟨240, 20, true, false⟩ = SamplingMethod.Stratified) ∧
  (most_suitable_method ⟨950, 25, false, true⟩ = SamplingMethod.Systematic) :=
by sorry

end NUMINAMATH_CALUDE_sampling_methods_correct_l1377_137783


namespace NUMINAMATH_CALUDE_solve_complex_equation_l1377_137736

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := 2 - 3 * i * z = -4 + 5 * i * z

-- State the theorem
theorem solve_complex_equation :
  ∃ z : ℂ, equation z ∧ z = -3/4 * i :=
sorry

end NUMINAMATH_CALUDE_solve_complex_equation_l1377_137736


namespace NUMINAMATH_CALUDE_walking_scenario_solution_l1377_137723

/-- Represents the walking scenario between two people --/
structure WalkingScenario where
  speed_A : ℝ  -- Speed of person A in meters per minute
  time_A_start : ℝ  -- Time person A has been walking when B starts
  speed_diff : ℝ  -- Speed difference between person B and person A
  time_diff_AC_CB : ℝ  -- Time difference between A to C and C to B for person A
  time_diff_CA_BC : ℝ  -- Time difference between C to A and B to C for person B

/-- The main theorem about the walking scenario --/
theorem walking_scenario_solution (w : WalkingScenario) 
  (h_speed_diff : w.speed_diff = 30)
  (h_time_A_start : w.time_A_start = 5.5)
  (h_time_diff_AC_CB : w.time_diff_AC_CB = 4)
  (h_time_diff_CA_BC : w.time_diff_CA_BC = 3) :
  ∃ (time_A_to_C : ℝ) (distance_AB : ℝ),
    time_A_to_C = 10 ∧ distance_AB = 1440 := by
  sorry


end NUMINAMATH_CALUDE_walking_scenario_solution_l1377_137723


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1377_137797

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => 3 * x^2 + 6 * x - |-21 + 5|
  ∃ x₁ x₂ : ℝ, x₁ = -1 + Real.sqrt 19 ∧ x₂ = -1 - Real.sqrt 19 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1377_137797


namespace NUMINAMATH_CALUDE_homes_cleaned_l1377_137740

-- Define the given conditions
def earnings_per_home : ℕ := 46
def total_earnings : ℕ := 276

-- Define the theorem to prove
theorem homes_cleaned (earnings_per_home : ℕ) (total_earnings : ℕ) : 
  total_earnings / earnings_per_home = 6 :=
sorry

end NUMINAMATH_CALUDE_homes_cleaned_l1377_137740


namespace NUMINAMATH_CALUDE_parallel_lines_theorem_l1377_137767

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- State the theorem
theorem parallel_lines_theorem 
  (m n : Line) (α β : Plane) 
  (hm_neq_n : m ≠ n)
  (hα_neq_β : α ≠ β)
  (hm_parallel_β : parallel_line_plane m β)
  (hm_in_α : contained_in m α)
  (hα_intersect_β : intersect α β n) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_theorem_l1377_137767


namespace NUMINAMATH_CALUDE_cube_triangle_areas_sum_l1377_137760

/-- Represents a 3D point in space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle in 3D space -/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- The vertices of a 2 × 2 × 2 cube -/
def cubeVertices : List Point3D := [
  ⟨0, 0, 0⟩, ⟨0, 0, 2⟩, ⟨0, 2, 0⟩, ⟨0, 2, 2⟩,
  ⟨2, 0, 0⟩, ⟨2, 0, 2⟩, ⟨2, 2, 0⟩, ⟨2, 2, 2⟩
]

/-- All possible triangles formed by the vertices of the cube -/
def cubeTriangles : List Triangle3D := sorry

/-- Calculates the area of a triangle in 3D space -/
def triangleArea (t : Triangle3D) : ℝ := sorry

/-- The sum of areas of all triangles formed by the cube vertices -/
def totalArea : ℝ := (cubeTriangles.map triangleArea).sum

/-- The main theorem -/
theorem cube_triangle_areas_sum :
  ∃ (m n p : ℕ), totalArea = m + Real.sqrt n + Real.sqrt p ∧ m + n + p = 972 := by
  sorry

end NUMINAMATH_CALUDE_cube_triangle_areas_sum_l1377_137760


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l1377_137753

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l1377_137753


namespace NUMINAMATH_CALUDE_even_multiple_six_sum_properties_l1377_137742

theorem even_multiple_six_sum_properties (a b : ℤ) 
  (h_a_even : Even a) (h_b_multiple_six : ∃ k, b = 6 * k) :
  Even (a + b) ∧ 
  (∃ m, a + b = 3 * m) ∧ 
  ¬(∀ (a b : ℤ), Even a → (∃ k, b = 6 * k) → ∃ n, a + b = 6 * n) ∧
  ∃ (a b : ℤ), Even a ∧ (∃ k, b = 6 * k) ∧ (∃ n, a + b = 6 * n) :=
by sorry

end NUMINAMATH_CALUDE_even_multiple_six_sum_properties_l1377_137742


namespace NUMINAMATH_CALUDE_hypotenuse_length_hypotenuse_length_proof_l1377_137756

/-- Represents a right triangle with one leg of 10 inches and the angle opposite that leg being 60° --/
structure RightTriangle where
  leg : ℝ
  angle : ℝ
  leg_eq : leg = 10
  angle_eq : angle = 60

/-- Theorem stating that the hypotenuse of the described right triangle is (20√3)/3 inches --/
theorem hypotenuse_length (t : RightTriangle) : ℝ :=
  (20 * Real.sqrt 3) / 3

/-- Proof of the theorem --/
theorem hypotenuse_length_proof (t : RightTriangle) : 
  hypotenuse_length t = (20 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_hypotenuse_length_proof_l1377_137756


namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l1377_137701

theorem perfect_square_divisibility (m n : ℕ) (h : m * n ∣ m^2 + n^2 + m) : 
  ∃ k : ℕ, m = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l1377_137701


namespace NUMINAMATH_CALUDE_actual_distance_between_towns_l1377_137720

-- Define the map distance between towns
def map_distance : ℝ := 18

-- Define the scale
def scale_inches : ℝ := 0.3
def scale_miles : ℝ := 5

-- Theorem to prove
theorem actual_distance_between_towns :
  (map_distance * scale_miles) / scale_inches = 300 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_between_towns_l1377_137720


namespace NUMINAMATH_CALUDE_minimum_walnuts_l1377_137721

/-- Represents the process of a child dividing and taking walnuts -/
def childProcess (n : ℕ) : ℕ := (n - 1) * 4 / 5

/-- Represents the final division process -/
def finalDivision (n : ℕ) : ℕ := n - 1

/-- Represents the entire walnut distribution process -/
def walnutDistribution (initial : ℕ) : ℕ :=
  finalDivision (childProcess (childProcess (childProcess (childProcess (childProcess initial)))))

theorem minimum_walnuts :
  ∃ (n : ℕ), n > 0 ∧ walnutDistribution n = 0 ∧ ∀ m : ℕ, m > 0 ∧ m < n → walnutDistribution m ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_minimum_walnuts_l1377_137721


namespace NUMINAMATH_CALUDE_no_odd_integer_solution_l1377_137785

theorem no_odd_integer_solution :
  ¬∃ (x y z : ℤ), Odd x ∧ Odd y ∧ Odd z ∧ (x + y)^2 + (x + z)^2 = (y + z)^2 := by
  sorry

end NUMINAMATH_CALUDE_no_odd_integer_solution_l1377_137785


namespace NUMINAMATH_CALUDE_frame_interior_perimeter_l1377_137775

theorem frame_interior_perimeter
  (frame_width : ℝ)
  (frame_area : ℝ)
  (outer_edge : ℝ)
  (h1 : frame_width = 2)
  (h2 : frame_area = 60)
  (h3 : outer_edge = 10) :
  let inner_length := outer_edge - 2 * frame_width
  let inner_width := (frame_area / (outer_edge - inner_length)) - frame_width
  inner_length * 2 + inner_width * 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_frame_interior_perimeter_l1377_137775


namespace NUMINAMATH_CALUDE_calendar_cost_l1377_137734

/-- The cost of each calendar given the promotional item quantities and costs -/
theorem calendar_cost (total_items : ℕ) (num_calendars : ℕ) (num_datebooks : ℕ) 
  (datebook_cost : ℚ) (total_spent : ℚ) :
  total_items = 500 →
  num_calendars = 300 →
  num_datebooks = 200 →
  datebook_cost = 1/2 →
  total_spent = 300 →
  (total_spent - num_datebooks * datebook_cost) / num_calendars = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_calendar_cost_l1377_137734


namespace NUMINAMATH_CALUDE_reflection_line_correct_l1377_137704

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The reflection line that transforms one point to another -/
def reflection_line (p1 p2 : Point) : Line :=
  sorry

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- The given points in the problem -/
def point1 : Point := ⟨5, 3⟩
def point2 : Point := ⟨1, -1⟩

/-- The proposed reflection line -/
def line_l : Line := ⟨-1, 4⟩

theorem reflection_line_correct :
  reflection_line point1 point2 = line_l ∧
  point_on_line (⟨3, 1⟩ : Point) line_l :=
sorry

end NUMINAMATH_CALUDE_reflection_line_correct_l1377_137704


namespace NUMINAMATH_CALUDE_prime_equality_l1377_137784

theorem prime_equality (p q r n : ℕ) : 
  Prime p → Prime q → Prime r → n > 0 →
  (∃ k₁ k₂ k₃ : ℕ, (p + n) = k₁ * q * r ∧ 
                   (q + n) = k₂ * r * p ∧ 
                   (r + n) = k₃ * p * q) →
  p = q ∧ q = r :=
sorry

end NUMINAMATH_CALUDE_prime_equality_l1377_137784


namespace NUMINAMATH_CALUDE_sum_of_solutions_x_minus_4_squared_equals_16_l1377_137759

theorem sum_of_solutions_x_minus_4_squared_equals_16 : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 4)^2 = 16 ∧ (x₂ - 4)^2 = 16 ∧ x₁ + x₂ = 8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_x_minus_4_squared_equals_16_l1377_137759


namespace NUMINAMATH_CALUDE_chapatis_ordered_l1377_137770

/-- The number of chapatis ordered by Alok -/
def chapatis : ℕ := sorry

/-- The cost of each chapati in rupees -/
def chapati_cost : ℕ := 6

/-- The cost of each plate of rice in rupees -/
def rice_cost : ℕ := 45

/-- The cost of each plate of mixed vegetable in rupees -/
def vegetable_cost : ℕ := 70

/-- The cost of each ice-cream cup in rupees -/
def icecream_cost : ℕ := 40

/-- The number of plates of rice ordered -/
def rice_plates : ℕ := 5

/-- The number of plates of mixed vegetable ordered -/
def vegetable_plates : ℕ := 7

/-- The number of ice-cream cups ordered -/
def icecream_cups : ℕ := 6

/-- The total amount paid in rupees -/
def total_paid : ℕ := 1051

theorem chapatis_ordered : chapatis = 16 := by
  sorry

end NUMINAMATH_CALUDE_chapatis_ordered_l1377_137770


namespace NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l1377_137749

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digit_factorial_sum (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  factorial hundreds + factorial tens + factorial ones

theorem unique_three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
  (∃ d, d ∈ [n / 100, (n / 10) % 10, n % 10] ∧ d = 6) ∧
  n = digit_factorial_sum n :=
by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l1377_137749


namespace NUMINAMATH_CALUDE_bicycle_price_problem_l1377_137779

theorem bicycle_price_problem (cost_price_A : ℝ) : 
  let selling_price_B := cost_price_A * 1.25
  let selling_price_C := selling_price_B * 1.5
  selling_price_C = 225 → cost_price_A = 120 := by
sorry

end NUMINAMATH_CALUDE_bicycle_price_problem_l1377_137779


namespace NUMINAMATH_CALUDE_sum_of_coordinates_A_l1377_137708

/-- Given three points A, B, and C in a 2D plane satisfying specific conditions,
    prove that the sum of the coordinates of A is 3. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/2 →
  (C.2 - A.2) / (B.2 - A.2) = 1/2 →
  B = (2, 5) →
  C = (6, -3) →
  A.1 + A.2 = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_A_l1377_137708


namespace NUMINAMATH_CALUDE_book_cost_calculation_l1377_137798

theorem book_cost_calculation (initial_amount : ℕ) (num_books : ℕ) (remaining_amount : ℕ) :
  initial_amount = 79 →
  num_books = 9 →
  remaining_amount = 16 →
  (initial_amount - remaining_amount) / num_books = 7 :=
by sorry

end NUMINAMATH_CALUDE_book_cost_calculation_l1377_137798


namespace NUMINAMATH_CALUDE_square_floor_tiles_l1377_137703

theorem square_floor_tiles (s : ℕ) (h_odd : Odd s) (h_middle : (s + 1) / 2 = 49) :
  s * s = 9409 := by
  sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l1377_137703


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l1377_137730

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 180 →
  a = 30 →
  c^2 = a^2 + b^2 →
  a + b + c = 42 + 2 * Real.sqrt 261 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l1377_137730


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1377_137739

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 12*a^2 + 27*a - 18 = 0 →
  b^3 - 12*b^2 + 27*b - 18 = 0 →
  c^3 - 12*c^2 + 27*c - 18 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^3 + 1/b^3 + 1/c^3 = 13/24 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1377_137739


namespace NUMINAMATH_CALUDE_m_one_sufficient_not_necessary_l1377_137758

def is_perpendicular (m : ℝ) : Prop :=
  let line1_slope := -m
  let line2_slope := 1 / m
  line1_slope * line2_slope = -1

theorem m_one_sufficient_not_necessary :
  (∃ m : ℝ, m ≠ 1 ∧ is_perpendicular m) ∧
  (is_perpendicular 1) :=
sorry

end NUMINAMATH_CALUDE_m_one_sufficient_not_necessary_l1377_137758


namespace NUMINAMATH_CALUDE_store_inventory_sale_l1377_137772

theorem store_inventory_sale (total_items : ℕ) (original_price : ℝ) 
  (discount_percent : ℝ) (debt : ℝ) (leftover : ℝ) :
  total_items = 2000 →
  original_price = 50 →
  discount_percent = 80 →
  debt = 15000 →
  leftover = 3000 →
  (((debt + leftover) / (original_price * (1 - discount_percent / 100))) / total_items) * 100 = 90 := by
  sorry


end NUMINAMATH_CALUDE_store_inventory_sale_l1377_137772


namespace NUMINAMATH_CALUDE_pentagonal_figure_segment_length_pentagonal_figure_segment_length_proof_l1377_137716

/-- The length of segment AB in a folded pentagonal figure --/
theorem pentagonal_figure_segment_length : ℝ :=
  -- Define the side length of the regular pentagons
  let side_length : ℝ := 1

  -- Define the number of pentagons
  let num_pentagons : ℕ := 4

  -- Define the internal angle of a regular pentagon (in radians)
  let pentagon_angle : ℝ := 3 * Real.pi / 5

  -- Define the angle between the square base and the pentagon face (in radians)
  let folding_angle : ℝ := Real.pi / 2 - pentagon_angle / 2

  -- The length of segment AB
  2

/-- Proof of the pentagonal_figure_segment_length theorem --/
theorem pentagonal_figure_segment_length_proof :
  pentagonal_figure_segment_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_figure_segment_length_pentagonal_figure_segment_length_proof_l1377_137716
