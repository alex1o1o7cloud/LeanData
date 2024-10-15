import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_roots_of_equation_l2384_238495

theorem sum_of_roots_of_equation (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁ - 3)^2 = 16 ∧ (x₂ - 3)^2 = 16 ∧ x₁ + x₂ = 6) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_of_equation_l2384_238495


namespace NUMINAMATH_CALUDE_geometric_probability_models_l2384_238460

-- Define the characteristics of a geometric probability model
structure GeometricProbabilityModel where
  infiniteOutcomes : Bool
  equallyLikely : Bool

-- Define the four probability models
def model1 : GeometricProbabilityModel :=
  { infiniteOutcomes := true,
    equallyLikely := true }

def model2 : GeometricProbabilityModel :=
  { infiniteOutcomes := true,
    equallyLikely := true }

def model3 : GeometricProbabilityModel :=
  { infiniteOutcomes := false,
    equallyLikely := true }

def model4 : GeometricProbabilityModel :=
  { infiniteOutcomes := true,
    equallyLikely := true }

-- Function to check if a model is a geometric probability model
def isGeometricProbabilityModel (model : GeometricProbabilityModel) : Bool :=
  model.infiniteOutcomes ∧ model.equallyLikely

-- Theorem stating which models are geometric probability models
theorem geometric_probability_models :
  isGeometricProbabilityModel model1 ∧
  isGeometricProbabilityModel model2 ∧
  ¬isGeometricProbabilityModel model3 ∧
  isGeometricProbabilityModel model4 :=
sorry

end NUMINAMATH_CALUDE_geometric_probability_models_l2384_238460


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2384_238412

/-- An arithmetic sequence with first term a and common difference d -/
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1 : ℝ) * d

theorem arithmetic_sequence_properties (a d : ℝ) :
  let seq := arithmetic_sequence a d
  (∀ n : ℕ, n > 0 → seq (n + 1) - seq n = d) ∧
  (seq 4 = 15 ∧ seq 15 = 59) := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2384_238412


namespace NUMINAMATH_CALUDE_max_value_of_expression_achievable_max_value_l2384_238486

theorem max_value_of_expression (n : ℕ) : 
  10 ≤ n ∧ n ≤ 99 → 3 * (300 - n) ≤ 870 := by
  sorry

theorem achievable_max_value : 
  ∃ (n : ℕ), 10 ≤ n ∧ n ≤ 99 ∧ 3 * (300 - n) = 870 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_achievable_max_value_l2384_238486


namespace NUMINAMATH_CALUDE_sin_plus_cos_value_l2384_238413

theorem sin_plus_cos_value (x : ℝ) 
  (h1 : 0 < x ∧ x < π/2) 
  (h2 : Real.sin (2*x - π/4) = -Real.sqrt 2/10) : 
  Real.sin x + Real.cos x = 2*Real.sqrt 10/5 := by
sorry

end NUMINAMATH_CALUDE_sin_plus_cos_value_l2384_238413


namespace NUMINAMATH_CALUDE_students_without_A_count_l2384_238405

/-- Represents the number of students who received an A in a specific combination of subjects -/
structure GradeDistribution where
  total : Nat
  history : Nat
  math : Nat
  computing : Nat
  historyAndMath : Nat
  historyAndComputing : Nat
  mathAndComputing : Nat
  allThree : Nat

/-- Calculates the number of students who didn't receive an A in any subject -/
def studentsWithoutA (g : GradeDistribution) : Nat :=
  g.total - (g.history + g.math + g.computing - g.historyAndMath - g.historyAndComputing - g.mathAndComputing + g.allThree)

theorem students_without_A_count (g : GradeDistribution) 
  (h_total : g.total = 40)
  (h_history : g.history = 10)
  (h_math : g.math = 18)
  (h_computing : g.computing = 9)
  (h_historyAndMath : g.historyAndMath = 5)
  (h_historyAndComputing : g.historyAndComputing = 3)
  (h_mathAndComputing : g.mathAndComputing = 4)
  (h_allThree : g.allThree = 2) :
  studentsWithoutA g = 13 := by
  sorry

end NUMINAMATH_CALUDE_students_without_A_count_l2384_238405


namespace NUMINAMATH_CALUDE_min_value_theorem_l2384_238410

theorem min_value_theorem (x : ℝ) (h : x > 2) :
  x + 2 / (x - 2) ≥ 2 + 2 * Real.sqrt 2 ∧
  (x + 2 / (x - 2) = 2 + 2 * Real.sqrt 2 ↔ x = 2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2384_238410


namespace NUMINAMATH_CALUDE_average_distance_is_17_l2384_238445

-- Define the distances traveled on each day
def monday_distance : ℝ := 12
def tuesday_distance : ℝ := 18
def wednesday_distance : ℝ := 21

-- Define the number of days
def num_days : ℝ := 3

-- Define the total distance
def total_distance : ℝ := monday_distance + tuesday_distance + wednesday_distance

-- Theorem: The average distance traveled per day is 17 miles
theorem average_distance_is_17 : total_distance / num_days = 17 := by
  sorry

end NUMINAMATH_CALUDE_average_distance_is_17_l2384_238445


namespace NUMINAMATH_CALUDE_perpendicular_iff_m_eq_two_l2384_238437

/-- Two vectors in R² -/
def Vector2 := ℝ × ℝ

/-- Dot product of two vectors in R² -/
def dot_product (v w : Vector2) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Definition of vector AB -/
def AB (m : ℝ) : Vector2 := (m + 3, 2 * m + 1)

/-- Definition of vector CD -/
def CD (m : ℝ) : Vector2 := (m + 3, -5)

/-- Theorem stating that AB and CD are perpendicular if and only if m = 2 -/
theorem perpendicular_iff_m_eq_two :
  ∀ m : ℝ, dot_product (AB m) (CD m) = 0 ↔ m = 2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_iff_m_eq_two_l2384_238437


namespace NUMINAMATH_CALUDE_length_AE_is_seven_l2384_238403

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the lengths of the sides
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (13, 14, 15)

-- Define the altitude from A
def altitude_A (t : Triangle) : ℝ × ℝ := sorry

-- Define point D where altitude intersects BC
def point_D (t : Triangle) : ℝ × ℝ := sorry

-- Define incircles of ABD and ACD
def incircle_ABD (t : Triangle) : Circle := sorry
def incircle_ACD (t : Triangle) : Circle := sorry

-- Define the common external tangent
def common_external_tangent (c1 c2 : Circle) : Line := sorry

-- Define point E where the common external tangent intersects AD
def point_E (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of AE
def length_AE (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem length_AE_is_seven (t : Triangle) :
  side_lengths t = (13, 14, 15) →
  length_AE t = 7 := by sorry

end NUMINAMATH_CALUDE_length_AE_is_seven_l2384_238403


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l2384_238417

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 10) 
  (h2 : a^2 + b^2 = 210) : 
  a * b = 55 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l2384_238417


namespace NUMINAMATH_CALUDE_max_pies_without_ingredients_l2384_238404

/-- Given a set of pies with specific ingredient distributions, 
    calculate the maximum number of pies without any of the specified ingredients -/
theorem max_pies_without_ingredients 
  (total_pies : ℕ) 
  (chocolate_pies : ℕ) 
  (blueberry_pies : ℕ) 
  (vanilla_pies : ℕ) 
  (almond_pies : ℕ) 
  (h_total : total_pies = 60)
  (h_chocolate : chocolate_pies ≥ 20)
  (h_blueberry : blueberry_pies = 45)
  (h_vanilla : vanilla_pies ≥ 24)
  (h_almond : almond_pies ≥ 6) :
  total_pies - blueberry_pies ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_max_pies_without_ingredients_l2384_238404


namespace NUMINAMATH_CALUDE_unique_prime_permutation_residue_system_l2384_238479

theorem unique_prime_permutation_residue_system : ∃! (p : ℕ), 
  Nat.Prime p ∧ 
  p % 2 = 1 ∧
  ∃ (b : Fin (p - 1) → Fin (p - 1)), Function.Bijective b ∧
    (∀ (x : Fin (p - 1)), 
      ∃ (y : Fin (p - 1)), (x.val + 1) ^ (b y).val ≡ (y.val + 1) [ZMOD p]) ∧
  p = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_permutation_residue_system_l2384_238479


namespace NUMINAMATH_CALUDE_no_lcm_83_l2384_238414

theorem no_lcm_83 (a b c : ℕ) : 
  a = 23 → b = 46 → Nat.lcm a (Nat.lcm b c) = 83 → False :=
by
  sorry

#check no_lcm_83

end NUMINAMATH_CALUDE_no_lcm_83_l2384_238414


namespace NUMINAMATH_CALUDE_jim_gas_spending_l2384_238423

/-- The amount of gas Jim bought in each state, in gallons -/
def gas_amount : ℝ := 10

/-- The price of gas per gallon in North Carolina, in dollars -/
def nc_price : ℝ := 2

/-- The additional price per gallon in Virginia compared to North Carolina, in dollars -/
def price_difference : ℝ := 1

/-- The total amount Jim spent on gas in both states -/
def total_spent : ℝ := gas_amount * nc_price + gas_amount * (nc_price + price_difference)

theorem jim_gas_spending :
  total_spent = 50 := by sorry

end NUMINAMATH_CALUDE_jim_gas_spending_l2384_238423


namespace NUMINAMATH_CALUDE_trigonometric_sum_product_form_l2384_238457

open Real

theorem trigonometric_sum_product_form :
  ∃ (a b c d : ℕ+),
    (∀ x : ℝ, cos (2 * x) + cos (6 * x) + cos (10 * x) + cos (14 * x) = 
      (a : ℝ) * cos (b * x) * cos (c * x) * cos (d * x)) ∧
    (a : ℕ) + b + c + d = 18 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_sum_product_form_l2384_238457


namespace NUMINAMATH_CALUDE_rotation_result_l2384_238478

/-- Rotation of a vector about the origin -/
def rotate90 (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Check if a vector passes through the y-axis -/
def passesYAxis (v : ℝ × ℝ × ℝ) : Prop := sorry

theorem rotation_result :
  let v₀ : ℝ × ℝ × ℝ := (2, 1, 1)
  let v₁ := rotate90 v₀
  passesYAxis v₁ →
  v₁ = (Real.sqrt (6/11), -3 * Real.sqrt (6/11), Real.sqrt (6/11)) :=
by sorry

end NUMINAMATH_CALUDE_rotation_result_l2384_238478


namespace NUMINAMATH_CALUDE_cos_pi_half_minus_two_alpha_l2384_238406

theorem cos_pi_half_minus_two_alpha (α : ℝ) (h : Real.tan α = 2) : 
  Real.cos (π / 2 - 2 * α) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_half_minus_two_alpha_l2384_238406


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l2384_238476

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (triangle_base : ℝ) : 
  square_perimeter = 64 →
  triangle_height = 32 →
  (square_perimeter / 4)^2 = (1/2) * triangle_height * triangle_base →
  triangle_base = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l2384_238476


namespace NUMINAMATH_CALUDE_min_b_value_l2384_238462

noncomputable section

variables (a b : ℝ) (x x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := Real.log x - a * x + (1 - a) / x - 1

def g (x : ℝ) : ℝ := x^2 - 2 * b * x + 4 / 3

theorem min_b_value (h1 : a = 1/3) 
  (h2 : ∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Icc 1 3, f x₁ ≥ g x₂) :
  b ≥ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_min_b_value_l2384_238462


namespace NUMINAMATH_CALUDE_odd_terms_in_binomial_expansion_l2384_238429

theorem odd_terms_in_binomial_expansion (p q : ℤ) (hp : Odd p) (hq : Odd q) :
  (Finset.range 9).filter (fun k => Odd (Nat.choose 8 k * (p + q)^(8 - k) * p^k)) = {0, 8} := by
  sorry

end NUMINAMATH_CALUDE_odd_terms_in_binomial_expansion_l2384_238429


namespace NUMINAMATH_CALUDE_johns_umbrella_cost_l2384_238456

/-- The total cost of John's umbrellas -/
def total_cost (house_umbrellas car_umbrellas unit_cost : ℕ) : ℕ :=
  (house_umbrellas + car_umbrellas) * unit_cost

/-- Proof that John's total umbrella cost is $24 -/
theorem johns_umbrella_cost :
  total_cost 2 1 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_johns_umbrella_cost_l2384_238456


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2384_238435

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_first : a 1 = 1)
  (h_fifth : a 5 = 5) :
  a 3 = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2384_238435


namespace NUMINAMATH_CALUDE_total_students_agreed_l2384_238438

def third_grade_students : ℕ := 256
def fourth_grade_students : ℕ := 525
def third_grade_agreement_rate : ℚ := 60 / 100
def fourth_grade_agreement_rate : ℚ := 45 / 100

theorem total_students_agreed :
  ⌊third_grade_agreement_rate * third_grade_students⌋ +
  ⌊fourth_grade_agreement_rate * fourth_grade_students⌋ = 390 := by
  sorry

end NUMINAMATH_CALUDE_total_students_agreed_l2384_238438


namespace NUMINAMATH_CALUDE_cube_root_simplification_l2384_238465

theorem cube_root_simplification :
  (20^3 + 30^3 + 60^3 : ℝ)^(1/3) = 10 * 251^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l2384_238465


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l2384_238436

theorem pure_imaginary_product (m : ℝ) : 
  (Complex.I : ℂ).im * ((1 + m * Complex.I) * (1 - Complex.I)).re = 0 → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l2384_238436


namespace NUMINAMATH_CALUDE_fraction_meaningful_l2384_238448

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / ((x - 1) * (x + 2))) ↔ (x ≠ 1 ∧ x ≠ -2) := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l2384_238448


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2384_238492

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B : Set ℝ := {x | -x^2 + 6*x - 8 > 0}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = Set.Ioc 2 3 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2384_238492


namespace NUMINAMATH_CALUDE_eventual_bounded_groups_l2384_238475

/-- Represents a group distribution in the society -/
def GroupDistribution := List Nat

/-- The redistribution process for a week -/
def redistribute : GroupDistribution → GroupDistribution := sorry

/-- Checks if all groups in a distribution have size at most 1 + √(2n) -/
def all_groups_bounded (n : Nat) (dist : GroupDistribution) : Prop := sorry

/-- Theorem: Eventually, all groups will be bounded by 1 + √(2n) -/
theorem eventual_bounded_groups (n : Nat) :
  ∃ (k : Nat), all_groups_bounded n ((redistribute^[k]) [n]) := by
  sorry

end NUMINAMATH_CALUDE_eventual_bounded_groups_l2384_238475


namespace NUMINAMATH_CALUDE_count_quadrilaterals_with_equidistant_point_l2384_238444

/-- A quadrilateral in a plane -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- A point is equidistant from all vertices of a quadrilateral -/
def has_equidistant_point (q : Quadrilateral) : Prop :=
  ∃ p : ℝ × ℝ, ∀ i : Fin 4, dist p (q.vertices i) = dist p (q.vertices 0)

/-- A kite with two consecutive right angles -/
def is_kite_with_two_right_angles (q : Quadrilateral) : Prop :=
  sorry

/-- A rectangle with sides in the ratio 3:1 -/
def is_rectangle_3_1 (q : Quadrilateral) : Prop :=
  sorry

/-- A rhombus with an angle of 120 degrees -/
def is_rhombus_120 (q : Quadrilateral) : Prop :=
  sorry

/-- A general quadrilateral with perpendicular diagonals -/
def has_perpendicular_diagonals (q : Quadrilateral) : Prop :=
  sorry

/-- An isosceles trapezoid where the non-parallel sides are equal in length -/
def is_isosceles_trapezoid (q : Quadrilateral) : Prop :=
  sorry

/-- The main theorem -/
theorem count_quadrilaterals_with_equidistant_point :
  ∃ (q1 q2 q3 : Quadrilateral),
    (is_kite_with_two_right_angles q1 ∨ 
     is_rectangle_3_1 q1 ∨ 
     is_rhombus_120 q1 ∨ 
     has_perpendicular_diagonals q1 ∨ 
     is_isosceles_trapezoid q1) ∧
    (is_kite_with_two_right_angles q2 ∨ 
     is_rectangle_3_1 q2 ∨ 
     is_rhombus_120 q2 ∨ 
     has_perpendicular_diagonals q2 ∨ 
     is_isosceles_trapezoid q2) ∧
    (is_kite_with_two_right_angles q3 ∨ 
     is_rectangle_3_1 q3 ∨ 
     is_rhombus_120 q3 ∨ 
     has_perpendicular_diagonals q3 ∨ 
     is_isosceles_trapezoid q3) ∧
    has_equidistant_point q1 ∧
    has_equidistant_point q2 ∧
    has_equidistant_point q3 ∧
    (∀ q : Quadrilateral, 
      (is_kite_with_two_right_angles q ∨ 
       is_rectangle_3_1 q ∨ 
       is_rhombus_120 q ∨ 
       has_perpendicular_diagonals q ∨ 
       is_isosceles_trapezoid q) →
      has_equidistant_point q →
      (q = q1 ∨ q = q2 ∨ q = q3)) :=
by
  sorry

end NUMINAMATH_CALUDE_count_quadrilaterals_with_equidistant_point_l2384_238444


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_is_two_l2384_238400

/-- Represents an isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The number of small triangles along each base -/
  num_triangles : ℕ
  /-- The area of each small triangle -/
  small_triangle_area : ℝ
  /-- Assumption that each small triangle has an area of 1 -/
  h_area_is_one : small_triangle_area = 1

/-- Calculates the area of the isosceles trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ :=
  2 * t.num_triangles * t.small_triangle_area

/-- Theorem stating that the area of the isosceles trapezoid is 2 -/
theorem isosceles_trapezoid_area_is_two (t : IsoscelesTrapezoid) :
  trapezoid_area t = 2 := by
  sorry

#check isosceles_trapezoid_area_is_two

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_is_two_l2384_238400


namespace NUMINAMATH_CALUDE_polynomial_sum_theorem_l2384_238484

theorem polynomial_sum_theorem (p q r s : ℤ) :
  (∀ x : ℝ, (x^2 + p*x + q) * (x^2 + r*x + s) = x^4 + 3*x^3 - 4*x^2 + 9*x + 7) →
  p + q + r + s = 11 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_theorem_l2384_238484


namespace NUMINAMATH_CALUDE_list_price_problem_l2384_238453

theorem list_price_problem (list_price : ℝ) : 
  (0.15 * (list_price - 15) = 0.30 * (list_price - 25)) → list_price = 35 := by
  sorry

end NUMINAMATH_CALUDE_list_price_problem_l2384_238453


namespace NUMINAMATH_CALUDE_train_length_l2384_238473

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length -/
theorem train_length 
  (speed : ℝ) 
  (time_to_cross : ℝ) 
  (bridge_length : ℝ) 
  (h1 : speed = 45) -- km/hr
  (h2 : time_to_cross = 30 / 3600) -- 30 seconds converted to hours
  (h3 : bridge_length = 265) -- meters
  : ∃ (train_length : ℝ), train_length = 110 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2384_238473


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2384_238416

theorem arithmetic_calculation : (120 / 6 * 2 / 3 : ℚ) = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2384_238416


namespace NUMINAMATH_CALUDE_equation_containing_2012_l2384_238482

theorem equation_containing_2012 (n : ℕ) : 
  (n^2 ≤ 2012 ∧ ∀ m : ℕ, m > n → m^2 > 2012) → 
  (n = 44 ∧ 2012 ∈ Finset.range (2*n^2 - n^2 + 1) \ Finset.range (n^2)) := by
  sorry

end NUMINAMATH_CALUDE_equation_containing_2012_l2384_238482


namespace NUMINAMATH_CALUDE_dot_product_calculation_l2384_238446

def vector_a : ℝ × ℝ := (-2, -6)

theorem dot_product_calculation (b : ℝ × ℝ) 
  (angle_condition : Real.cos (120 * π / 180) = -1/2)
  (magnitude_b : Real.sqrt ((b.1)^2 + (b.2)^2) = Real.sqrt 10) :
  (vector_a.1 * b.1 + vector_a.2 * b.2) = -10 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_calculation_l2384_238446


namespace NUMINAMATH_CALUDE_lovely_class_size_l2384_238441

/-- Proves that the number of students in Mrs. Lovely's class is 29 given the jelly bean distribution conditions. -/
theorem lovely_class_size :
  ∀ (g : ℕ),
  let b := g + 3
  let total_jelly_beans := 420
  let remaining_jelly_beans := 18
  let distributed_jelly_beans := total_jelly_beans - remaining_jelly_beans
  g * g + b * b = distributed_jelly_beans →
  g + b = 29 := by
  sorry

end NUMINAMATH_CALUDE_lovely_class_size_l2384_238441


namespace NUMINAMATH_CALUDE_prime_product_sum_proper_fractions_l2384_238415

/-- Sum of proper fractions with denominator k -/
def sum_proper_fractions (k : ℕ) : ℚ :=
  (k - 1) / 2

theorem prime_product_sum_proper_fractions : 
  ∀ m n : ℕ, 
  m.Prime → n.Prime → m < n → 
  (sum_proper_fractions m) * (sum_proper_fractions n) = 5 → 
  m = 3 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_prime_product_sum_proper_fractions_l2384_238415


namespace NUMINAMATH_CALUDE_monotonic_decreasing_odd_function_property_l2384_238467

-- Define a monotonically decreasing function on ℝ
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- Define an odd function on ℝ
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem monotonic_decreasing_odd_function_property
  (f : ℝ → ℝ) (h1 : MonoDecreasing f) (h2 : OddFunction f) :
  -f (-3) < f (-4) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_odd_function_property_l2384_238467


namespace NUMINAMATH_CALUDE_table_covering_l2384_238474

/-- Represents a cell in the table -/
inductive Cell
| Zero
| One

/-- Represents the 1000x1000 table -/
def Table := Fin 1000 → Fin 1000 → Cell

/-- Checks if a set of rows covers all columns with at least one 1 -/
def coversColumnsWithOnes (t : Table) (rows : Finset (Fin 1000)) : Prop :=
  ∀ j : Fin 1000, ∃ i ∈ rows, t i j = Cell.One

/-- Checks if a set of columns covers all rows with at least one 0 -/
def coversRowsWithZeros (t : Table) (cols : Finset (Fin 1000)) : Prop :=
  ∀ i : Fin 1000, ∃ j ∈ cols, t i j = Cell.Zero

/-- The main theorem -/
theorem table_covering (t : Table) :
  (∃ rows : Finset (Fin 1000), rows.card = 10 ∧ coversColumnsWithOnes t rows) ∨
  (∃ cols : Finset (Fin 1000), cols.card = 10 ∧ coversRowsWithZeros t cols) :=
sorry

end NUMINAMATH_CALUDE_table_covering_l2384_238474


namespace NUMINAMATH_CALUDE_new_cards_count_l2384_238455

def cards_per_page : ℕ := 3
def old_cards : ℕ := 10
def pages_used : ℕ := 6

theorem new_cards_count : 
  pages_used * cards_per_page - old_cards = 8 := by sorry

end NUMINAMATH_CALUDE_new_cards_count_l2384_238455


namespace NUMINAMATH_CALUDE_hyperbola_parabola_focus_l2384_238461

theorem hyperbola_parabola_focus (a : ℝ) : 
  a > 0 → 
  (∃ (x y : ℝ), x^2 - y^2 = a^2) → 
  (∃ (x y : ℝ), y^2 = 4*x) → 
  (∃ (c : ℝ), c > 0 ∧ c^2 - a^2 = a^2) →
  (∃ (f : ℝ × ℝ), f = (1, 0) ∧ f.1 = c) →
  a = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_focus_l2384_238461


namespace NUMINAMATH_CALUDE_range_of_p_l2384_238450

noncomputable def p (x : ℝ) : ℝ := x^6 + 6*x^3 + 9

theorem range_of_p :
  Set.range (fun (x : ℝ) ↦ p x) = Set.Ici 9 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l2384_238450


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l2384_238440

-- Define the function f
def f : ℝ → ℝ := fun x ↦ 2 * x^2 - 10 * x

-- State the theorem
theorem quadratic_function_unique :
  (∀ x, f x < 0 ↔ 0 < x ∧ x < 5) →
  (∀ x ∈ Set.Icc (-1) 4, f x ≤ 12) →
  (∃ x ∈ Set.Icc (-1) 4, f x = 12) →
  (∀ x, f x = 2 * x^2 - 10 * x) :=
by sorry

-- Note: The condition a < 0 is not used in this theorem as it's only relevant for part II of the original problem

end NUMINAMATH_CALUDE_quadratic_function_unique_l2384_238440


namespace NUMINAMATH_CALUDE_distance_between_locations_l2384_238487

theorem distance_between_locations (speed_A speed_B : ℝ) (time : ℝ) (remaining_distance : ℝ) :
  speed_B = (4/5) * speed_A →
  time = 3 →
  remaining_distance = 3 →
  ∃ (distance_AB : ℝ),
    distance_AB = speed_A * time + speed_B * time + remaining_distance :=
by sorry

end NUMINAMATH_CALUDE_distance_between_locations_l2384_238487


namespace NUMINAMATH_CALUDE_circle_center_l2384_238481

/-- The center of the circle x^2 + y^2 - 2x + 4y + 3 = 0 is at the point (1, -2). -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 - 2*x + 4*y + 3 = 0) → 
  ∃ (h : ℝ), (x - 1)^2 + (y + 2)^2 = h^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l2384_238481


namespace NUMINAMATH_CALUDE_nonzero_terms_count_l2384_238419

/-- The number of nonzero terms in the expansion of (2x+3)(x^2 + 2x + 4) - 2(x^3 + x^2 - 3x + 1) + (x-2)(x+5) is 2 -/
theorem nonzero_terms_count (x : ℝ) : 
  let expansion := (2*x+3)*(x^2 + 2*x + 4) - 2*(x^3 + x^2 - 3*x + 1) + (x-2)*(x+5)
  ∃ (a b : ℝ), expansion = a*x^2 + b*x ∧ a ≠ 0 ∧ b ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_nonzero_terms_count_l2384_238419


namespace NUMINAMATH_CALUDE_sum_a_plus_d_equals_six_l2384_238470

theorem sum_a_plus_d_equals_six (a b c d e : ℝ)
  (eq1 : a + b = 12)
  (eq2 : b + c = 9)
  (eq3 : c + d = 3)
  (eq4 : d + e = 7)
  (eq5 : e + a = 10) :
  a + d = 6 := by sorry

end NUMINAMATH_CALUDE_sum_a_plus_d_equals_six_l2384_238470


namespace NUMINAMATH_CALUDE_line_intersects_x_axis_l2384_238452

theorem line_intersects_x_axis :
  ∃ (x : ℝ), 5 * 0 - 6 * x = 15 ∧ x = -2.5 := by sorry

end NUMINAMATH_CALUDE_line_intersects_x_axis_l2384_238452


namespace NUMINAMATH_CALUDE_eighth_fibonacci_term_l2384_238485

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem eighth_fibonacci_term :
  fibonacci 7 = 21 :=
by sorry

end NUMINAMATH_CALUDE_eighth_fibonacci_term_l2384_238485


namespace NUMINAMATH_CALUDE_trajectory_of_P_l2384_238422

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

/-- The line equation -/
def line (k m x y : ℝ) : Prop := y = k*x + m

/-- Condition that k is not equal to ±2 -/
def k_condition (k : ℝ) : Prop := k ≠ 2 ∧ k ≠ -2

/-- The trajectory equation -/
def trajectory (x y : ℝ) : Prop := x^2/25 - 4*y^2/25 = 1

/-- Main theorem: The trajectory of point P -/
theorem trajectory_of_P (k m x y : ℝ) :
  k_condition k →
  (∃ (x₀ y₀ : ℝ), hyperbola x₀ y₀ ∧ line k m x₀ y₀) →
  y ≠ 0 →
  trajectory x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_P_l2384_238422


namespace NUMINAMATH_CALUDE_trace_equality_for_cubed_matrices_l2384_238449

open Matrix

theorem trace_equality_for_cubed_matrices
  (A B : Matrix (Fin 2) (Fin 2) ℝ)
  (h_not_commute : A * B ≠ B * A)
  (h_cubed_equal : A^3 = B^3) :
  ∀ n : ℕ, Matrix.trace (A^n) = Matrix.trace (B^n) :=
by sorry

end NUMINAMATH_CALUDE_trace_equality_for_cubed_matrices_l2384_238449


namespace NUMINAMATH_CALUDE_base_representation_digit_difference_l2384_238498

theorem base_representation_digit_difference : 
  let n : ℕ := 1234
  let base_5_digits := (Nat.log n 5).succ
  let base_9_digits := (Nat.log n 9).succ
  base_5_digits - base_9_digits = 1 := by
sorry

end NUMINAMATH_CALUDE_base_representation_digit_difference_l2384_238498


namespace NUMINAMATH_CALUDE_cow_count_l2384_238427

/-- Represents a group of ducks and cows -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (g : AnimalGroup) : ℕ := 2 * g.ducks + 4 * g.cows

/-- The total number of heads in the group -/
def totalHeads (g : AnimalGroup) : ℕ := g.ducks + g.cows

/-- Theorem: If the total number of legs is 30 more than twice the number of heads,
    then there are 15 cows in the group -/
theorem cow_count (g : AnimalGroup) :
  totalLegs g = 2 * totalHeads g + 30 → g.cows = 15 := by
  sorry


end NUMINAMATH_CALUDE_cow_count_l2384_238427


namespace NUMINAMATH_CALUDE_largest_ratio_in_arithmetic_sequence_l2384_238494

/-- Given an arithmetic sequence {a_n} with S_n as the sum of its first n terms,
    if S_15 > 0 and S_16 < 0, then S_8/a_8 is the largest among S_1/a_1, S_2/a_2, ..., S_15/a_15 -/
theorem largest_ratio_in_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2) 
  (h_S15 : S 15 > 0) 
  (h_S16 : S 16 < 0) : 
  ∀ k ∈ Finset.range 15, S 8 / a 8 ≥ S (k + 1) / a (k + 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_ratio_in_arithmetic_sequence_l2384_238494


namespace NUMINAMATH_CALUDE_quadratic_root_triple_relation_l2384_238459

theorem quadratic_root_triple_relation (a b c : ℝ) (α β : ℝ) : 
  a ≠ 0 →
  a * α^2 + b * α + c = 0 →
  a * β^2 + b * β + c = 0 →
  β = 3 * α →
  3 * b^2 = 16 * a * c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_triple_relation_l2384_238459


namespace NUMINAMATH_CALUDE_complex_modulus_inequality_l2384_238409

theorem complex_modulus_inequality (x y : ℝ) : 
  let z : ℂ := Complex.mk x y
  ‖z‖ ≤ |x| + |y| := by sorry

end NUMINAMATH_CALUDE_complex_modulus_inequality_l2384_238409


namespace NUMINAMATH_CALUDE_total_boys_across_grades_l2384_238463

/-- Represents the number of students in each grade level -/
structure GradeLevel where
  girls : ℕ
  boys : ℕ

/-- Calculates the total number of boys across all grade levels -/
def totalBoys (gradeA gradeB gradeC : GradeLevel) : ℕ :=
  gradeA.boys + gradeB.boys + gradeC.boys

/-- Theorem stating the total number of boys across three grade levels -/
theorem total_boys_across_grades (gradeA gradeB gradeC : GradeLevel) 
  (hA : gradeA.girls = 256 ∧ gradeA.girls = gradeA.boys + 52)
  (hB : gradeB.girls = 360 ∧ gradeB.boys = gradeB.girls - 40)
  (hC : gradeC.girls = 168 ∧ gradeC.boys = gradeC.girls) : 
  totalBoys gradeA gradeB gradeC = 692 := by
  sorry


end NUMINAMATH_CALUDE_total_boys_across_grades_l2384_238463


namespace NUMINAMATH_CALUDE_extreme_value_and_monotonicity_l2384_238418

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 3

theorem extreme_value_and_monotonicity :
  (f 1 = -1 ∧ f' 1 = 0) ∧
  (∀ x, x < -1 → f' x > 0) ∧
  (∀ x, x > 1 → f' x > 0) ∧
  (∀ x, -1 < x ∧ x < 1 → f' x < 0) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_and_monotonicity_l2384_238418


namespace NUMINAMATH_CALUDE_mms_per_pack_is_40_l2384_238490

/-- The number of sundaes made on Monday -/
def monday_sundaes : ℕ := 40

/-- The number of m&ms per sundae on Monday -/
def monday_mms_per_sundae : ℕ := 6

/-- The number of sundaes made on Tuesday -/
def tuesday_sundaes : ℕ := 20

/-- The number of m&ms per sundae on Tuesday -/
def tuesday_mms_per_sundae : ℕ := 10

/-- The total number of m&m packs used -/
def total_packs : ℕ := 11

/-- The number of m&ms in each pack -/
def mms_per_pack : ℕ := (monday_sundaes * monday_mms_per_sundae + tuesday_sundaes * tuesday_mms_per_sundae) / total_packs

theorem mms_per_pack_is_40 : mms_per_pack = 40 := by
  sorry

end NUMINAMATH_CALUDE_mms_per_pack_is_40_l2384_238490


namespace NUMINAMATH_CALUDE_chameleon_distance_l2384_238425

/-- A chameleon is a sequence of letters a, b, and c. -/
structure Chameleon (n : ℕ) where
  sequence : List Char
  length_eq : sequence.length = 3 * n
  count_a : sequence.count 'a' = n
  count_b : sequence.count 'b' = n
  count_c : sequence.count 'c' = n

/-- A swap is a transposition of two adjacent letters in a chameleon. -/
def swap (c : Chameleon n) (i : ℕ) : Chameleon n :=
  sorry

/-- The minimum number of swaps required to transform one chameleon into another. -/
def min_swaps (x y : Chameleon n) : ℕ :=
  sorry

/-- For any chameleon, there exists another chameleon that requires at least 3n²/2 swaps to reach. -/
theorem chameleon_distance (n : ℕ) (hn : 0 < n) (x : Chameleon n) :
  ∃ y : Chameleon n, 3 * n^2 / 2 ≤ min_swaps x y :=
  sorry

end NUMINAMATH_CALUDE_chameleon_distance_l2384_238425


namespace NUMINAMATH_CALUDE_wednesday_temperature_l2384_238468

/-- The temperature on Wednesday given the temperatures for the other days of the week and the average temperature --/
theorem wednesday_temperature
  (sunday : ℝ) (monday : ℝ) (tuesday : ℝ) (thursday : ℝ) (friday : ℝ) (saturday : ℝ) 
  (average : ℝ)
  (h_sunday : sunday = 40)
  (h_monday : monday = 50)
  (h_tuesday : tuesday = 65)
  (h_thursday : thursday = 82)
  (h_friday : friday = 72)
  (h_saturday : saturday = 26)
  (h_average : average = 53)
  : ∃ (wednesday : ℝ), 
    (sunday + monday + tuesday + wednesday + thursday + friday + saturday) / 7 = average ∧ 
    wednesday = 36 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_temperature_l2384_238468


namespace NUMINAMATH_CALUDE_table_height_l2384_238443

/-- Represents the configuration of two identical blocks on a table -/
structure BlockConfiguration where
  l : ℝ  -- length of each block
  w : ℝ  -- width of each block
  h : ℝ  -- height of the table

/-- The length measurement in Figure 1 -/
def figure1_length (config : BlockConfiguration) : ℝ :=
  config.l + config.h - config.w

/-- The length measurement in Figure 2 -/
def figure2_length (config : BlockConfiguration) : ℝ :=
  config.w + config.h - config.l

/-- The main theorem stating the height of the table -/
theorem table_height (config : BlockConfiguration) 
  (h1 : figure1_length config = 32)
  (h2 : figure2_length config = 28) : 
  config.h = 30 := by
  sorry


end NUMINAMATH_CALUDE_table_height_l2384_238443


namespace NUMINAMATH_CALUDE_cricket_average_score_l2384_238466

/-- Given the average score for 10 matches and the average score for the first 6 matches,
    calculate the average score for the last 4 matches. -/
theorem cricket_average_score (total_matches : ℕ) (first_matches : ℕ) 
    (total_average : ℚ) (first_average : ℚ) :
  total_matches = 10 →
  first_matches = 6 →
  total_average = 389/10 →
  first_average = 41 →
  (total_average * total_matches - first_average * first_matches) / (total_matches - first_matches) = 143/4 :=
by sorry

end NUMINAMATH_CALUDE_cricket_average_score_l2384_238466


namespace NUMINAMATH_CALUDE_expression_equality_l2384_238454

theorem expression_equality : (-3)^2 - Real.sqrt 4 + (1/2)⁻¹ = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2384_238454


namespace NUMINAMATH_CALUDE_inequality_solution_interval_l2384_238472

theorem inequality_solution_interval (a : ℝ) : 
  (∀ x, Real.sqrt (x + a) ≥ x) ∧ 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    Real.sqrt (x₁ + a) = x₁ ∧ 
    Real.sqrt (x₂ + a) = x₂ ∧ 
    |x₁ - x₂| = 4 * |a|) →
  a = 4/9 ∨ a = (1 - Real.sqrt 5) / 8 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_interval_l2384_238472


namespace NUMINAMATH_CALUDE_league_members_count_l2384_238408

/-- Represents the cost of items and total expenditure in the Rockham Soccer League --/
structure LeagueCosts where
  sock_cost : ℕ
  tshirt_cost_difference : ℕ
  set_discount : ℕ
  total_expenditure : ℕ

/-- Calculates the number of members in the Rockham Soccer League --/
def calculate_members (costs : LeagueCosts) : ℕ :=
  sorry

/-- Theorem stating that the number of members in the league is 150 --/
theorem league_members_count (costs : LeagueCosts)
  (h1 : costs.sock_cost = 5)
  (h2 : costs.tshirt_cost_difference = 6)
  (h3 : costs.set_discount = 3)
  (h4 : costs.total_expenditure = 3100) :
  calculate_members costs = 150 :=
sorry

end NUMINAMATH_CALUDE_league_members_count_l2384_238408


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l2384_238442

theorem meaningful_fraction_range :
  ∀ x : ℝ, (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l2384_238442


namespace NUMINAMATH_CALUDE_min_days_to_plant_trees_l2384_238401

def trees_planted (n : ℕ) : ℕ := 2 * (2^n - 1)

theorem min_days_to_plant_trees : 
  ∀ n : ℕ, n > 0 → (trees_planted n ≥ 100 → n ≥ 6) ∧ (trees_planted 6 ≥ 100) :=
by sorry

end NUMINAMATH_CALUDE_min_days_to_plant_trees_l2384_238401


namespace NUMINAMATH_CALUDE_positive_real_solution_l2384_238447

theorem positive_real_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a ^ b = b ^ a) (h2 : b = 4 * a) : a = (4 : ℝ) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_positive_real_solution_l2384_238447


namespace NUMINAMATH_CALUDE_a_range_l2384_238407

/-- The piecewise function f(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x - a

/-- Theorem stating that if f(0) is the minimum value of f(x), then a is in [0,1] --/
theorem a_range (a : ℝ) : 
  (∀ x : ℝ, f a 0 ≤ f a x) → 0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l2384_238407


namespace NUMINAMATH_CALUDE_statement_implies_innocence_statement_proves_innocence_l2384_238421

-- Define the possible roles of a person
inductive Role
  | Knight
  | Liar
  | Normal

-- Define the statement made by the defendant
def statement (role : Role) (guilty : Bool) : Prop :=
  (role = Role.Knight ∧ ¬guilty) ∨ (role = Role.Liar ∧ guilty)

-- Define what it means to be a criminal
def isCriminal (role : Role) : Prop :=
  role = Role.Knight ∨ role = Role.Liar

-- Theorem: The statement implies innocence for all possible roles
theorem statement_implies_innocence (role : Role) :
  (∀ r, isCriminal r → (statement r true ↔ ¬statement r false)) →
  statement role false →
  ¬isCriminal role ∨ ¬statement role true :=
by sorry

-- The main theorem: The statement proves innocence
theorem statement_proves_innocence :
  ∀ role, ¬isCriminal role ∨ ¬statement role true :=
by sorry

end NUMINAMATH_CALUDE_statement_implies_innocence_statement_proves_innocence_l2384_238421


namespace NUMINAMATH_CALUDE_complex_equality_implies_ratio_l2384_238489

theorem complex_equality_implies_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b * Complex.I) ^ 4 = (a - b * Complex.I) ^ 4 →
  b / a = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_implies_ratio_l2384_238489


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l2384_238420

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, |3*y - 4| ≤ 25 → y ≥ -7) ∧ |3*(-7) - 4| ≤ 25 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l2384_238420


namespace NUMINAMATH_CALUDE_smallest_integer_in_A_l2384_238469

def A : Set ℝ := {x | |x - 2| ≤ 5}

theorem smallest_integer_in_A : 
  ∃ (n : ℤ), (n : ℝ) ∈ A ∧ ∀ (m : ℤ), (m : ℝ) ∈ A → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_A_l2384_238469


namespace NUMINAMATH_CALUDE_triangle_theorem_l2384_238424

open Real

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * cos (2 * t.C) + 2 * t.c * cos t.A * cos t.C + t.a + t.b = 0)
  (h2 : t.b = 4 * sin t.B) : 
  t.C = 2 * π / 3 ∧ 
  (∀ S : ℝ, S = 1/2 * t.a * t.b * sin t.C → S ≤ Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2384_238424


namespace NUMINAMATH_CALUDE_flower_bed_fraction_l2384_238496

/-- Represents a rectangular yard with flower beds -/
structure FlowerYard where
  /-- Length of the yard -/
  length : ℝ
  /-- Width of the yard -/
  width : ℝ
  /-- Radius of the circular flower bed -/
  circle_radius : ℝ
  /-- Length of the shorter parallel side of the trapezoidal remainder -/
  trapezoid_short_side : ℝ
  /-- Length of the longer parallel side of the trapezoidal remainder -/
  trapezoid_long_side : ℝ

/-- Theorem stating the fraction of the yard occupied by flower beds -/
theorem flower_bed_fraction (yard : FlowerYard) 
  (h1 : yard.trapezoid_short_side = 20)
  (h2 : yard.trapezoid_long_side = 40)
  (h3 : yard.circle_radius = 2)
  (h4 : yard.length = yard.trapezoid_long_side)
  (h5 : yard.width = (yard.trapezoid_long_side - yard.trapezoid_short_side) / 2) :
  (100 + 4 * Real.pi) / 400 = 
    ((yard.trapezoid_long_side - yard.trapezoid_short_side)^2 / 4 + yard.circle_radius^2 * Real.pi) / 
    (yard.length * yard.width) :=
by sorry

end NUMINAMATH_CALUDE_flower_bed_fraction_l2384_238496


namespace NUMINAMATH_CALUDE_optimal_cylinder_ratio_l2384_238483

/-- The optimal ratio of height to radius for a cylinder with minimal surface area --/
theorem optimal_cylinder_ratio (V : ℝ) (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  V = π * r^2 * h → (∀ h' r', h' > 0 → r' > 0 → V = π * r'^2 * h' → 
    2 * π * r^2 + 2 * π * r * h ≤ 2 * π * r'^2 + 2 * π * r' * h') → 
  h / r = 2 :=
sorry

end NUMINAMATH_CALUDE_optimal_cylinder_ratio_l2384_238483


namespace NUMINAMATH_CALUDE_inequality_proof_l2384_238432

theorem inequality_proof (a₁ a₂ a₃ a₄ : ℝ) 
  (h_pos₁ : a₁ > 0) (h_pos₂ : a₂ > 0) (h_pos₃ : a₃ > 0) (h_pos₄ : a₄ > 0)
  (h_distinct₁₂ : a₁ ≠ a₂) (h_distinct₁₃ : a₁ ≠ a₃) (h_distinct₁₄ : a₁ ≠ a₄)
  (h_distinct₂₃ : a₂ ≠ a₃) (h_distinct₂₄ : a₂ ≠ a₄) (h_distinct₃₄ : a₃ ≠ a₄) :
  a₁^3 / (a₂ - a₃)^2 + a₂^3 / (a₃ - a₄)^2 + a₃^3 / (a₄ - a₁)^2 + a₄^3 / (a₁ - a₂)^2 
  > a₁ + a₂ + a₃ + a₄ :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2384_238432


namespace NUMINAMATH_CALUDE_polynomial_identity_l2384_238433

-- Define g as a polynomial function
variable (g : ℝ → ℝ)

-- State the theorem
theorem polynomial_identity 
  (h : ∀ x, g (x^2 + 2) = x^4 + 6*x^2 + 8) :
  ∀ x, g (x^2 - 1) = x^4 - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2384_238433


namespace NUMINAMATH_CALUDE_max_y_diff_intersection_points_l2384_238499

/-- The maximum difference between the y-coordinates of the intersection points
    of y = 4 - 2x^2 + x^3 and y = 2 + x^2 + x^3 is 4√2/9. -/
theorem max_y_diff_intersection_points :
  let f (x : ℝ) := 4 - 2 * x^2 + x^3
  let g (x : ℝ) := 2 + x^2 + x^3
  let intersection_points := {x : ℝ | f x = g x}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ intersection_points ∧ x₂ ∈ intersection_points ∧
    |f x₁ - f x₂| = 4 * Real.sqrt 2 / 9 ∧
    ∀ (y₁ y₂ : ℝ), y₁ ∈ intersection_points → y₂ ∈ intersection_points →
      |f y₁ - f y₂| ≤ 4 * Real.sqrt 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_max_y_diff_intersection_points_l2384_238499


namespace NUMINAMATH_CALUDE_unique_valid_number_l2384_238430

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    b = 3 ∧
    a + c = 5 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ∃ (d : ℕ), n + 124 = 111 * d

theorem unique_valid_number : ∃! n : ℕ, is_valid_number n ∧ n = 431 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l2384_238430


namespace NUMINAMATH_CALUDE_box_weight_l2384_238411

/-- Given a pallet with boxes, calculate the weight of each box. -/
theorem box_weight (total_weight : ℝ) (num_boxes : ℕ) (h1 : total_weight = 267) (h2 : num_boxes = 3) :
  total_weight / num_boxes = 89 := by
  sorry

end NUMINAMATH_CALUDE_box_weight_l2384_238411


namespace NUMINAMATH_CALUDE_inequality_always_true_implies_a_less_than_seven_l2384_238426

theorem inequality_always_true_implies_a_less_than_seven (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 4| > a) → a < 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_implies_a_less_than_seven_l2384_238426


namespace NUMINAMATH_CALUDE_cream_ratio_l2384_238491

-- Define the initial quantities
def initial_coffee : ℚ := 20
def john_drink : ℚ := 3
def john_cream : ℚ := 4
def jane_cream : ℚ := 3
def jane_drink : ℚ := 5

-- Calculate the amounts of cream in each cup
def john_cream_amount : ℚ := john_cream

def jane_mixture : ℚ := initial_coffee + jane_cream
def jane_cream_ratio : ℚ := jane_cream / jane_mixture
def jane_remaining : ℚ := jane_mixture - jane_drink
def jane_cream_amount : ℚ := jane_cream_ratio * jane_remaining

-- State the theorem
theorem cream_ratio : john_cream_amount / jane_cream_amount = 46 / 27 := by
  sorry

end NUMINAMATH_CALUDE_cream_ratio_l2384_238491


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l2384_238458

theorem sqrt_sum_equals_seven (x : ℝ) (h : Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) :
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l2384_238458


namespace NUMINAMATH_CALUDE_complex_square_value_l2384_238471

theorem complex_square_value : ((1 - Complex.I * Real.sqrt 3) / Complex.I) ^ 2 = 2 + Complex.I * (2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_square_value_l2384_238471


namespace NUMINAMATH_CALUDE_monotonic_exp_minus_mx_l2384_238488

/-- If f(x) = e^x - mx is monotonically increasing on [0, +∞), then m ≤ 1 -/
theorem monotonic_exp_minus_mx (m : ℝ) :
  (∀ x : ℝ, x ≥ 0 → Monotone (fun x : ℝ ↦ Real.exp x - m * x)) →
  m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_exp_minus_mx_l2384_238488


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l2384_238434

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l2384_238434


namespace NUMINAMATH_CALUDE_unique_k_value_l2384_238431

/-- The polynomial expression -/
def polynomial (k : ℚ) (x y : ℚ) : ℚ := x^2 + 4*x*y + 2*x + k*y - 3*k

/-- Condition for integer factorization -/
def has_integer_factorization (k : ℚ) : Prop :=
  ∃ (A B C D E F : ℤ), 
    ∀ (x y : ℚ), polynomial k x y = (A*x + B*y + C) * (D*x + E*y + F)

/-- Condition for non-negative discriminant of the quadratic part -/
def has_nonnegative_discriminant (k : ℚ) : Prop :=
  (4:ℚ)^2 - 4*1*0 ≥ 0

/-- The main theorem -/
theorem unique_k_value : 
  (∃! k : ℚ, has_integer_factorization k ∧ has_nonnegative_discriminant k) ∧
  (∀ k : ℚ, has_integer_factorization k ∧ has_nonnegative_discriminant k → k = 0) :=
sorry

end NUMINAMATH_CALUDE_unique_k_value_l2384_238431


namespace NUMINAMATH_CALUDE_dot_product_v_w_l2384_238497

def v : Fin 3 → ℝ := ![(-5 : ℝ), 2, -3]
def w : Fin 3 → ℝ := ![7, -4, 6]

theorem dot_product_v_w :
  (Finset.univ.sum fun i => v i * w i) = -61 := by sorry

end NUMINAMATH_CALUDE_dot_product_v_w_l2384_238497


namespace NUMINAMATH_CALUDE_two_distinct_roots_root_one_case_l2384_238464

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := x^2 - (m - 3) * x - m

-- Theorem stating that the equation has two distinct real roots for all m
theorem two_distinct_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ = 0 ∧ quadratic_equation m x₂ = 0 :=
sorry

-- Theorem for the case when one root is 1
theorem root_one_case :
  ∃ m : ℝ, quadratic_equation m 1 = 0 ∧ 
  (∃ x : ℝ, x ≠ 1 ∧ quadratic_equation m x = 0) ∧
  m = 2 ∧
  (∃ x : ℝ, x = -2 ∧ quadratic_equation m x = 0) :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_root_one_case_l2384_238464


namespace NUMINAMATH_CALUDE_polygon_existence_l2384_238477

/-- A polygon on a unit grid --/
structure GridPolygon where
  sides : ℕ
  area : ℕ
  vertices : List (ℕ × ℕ)

/-- Predicate to check if a GridPolygon is valid --/
def isValidGridPolygon (p : GridPolygon) : Prop :=
  p.sides = p.vertices.length ∧
  p.area ≤ (List.maximum (p.vertices.map Prod.fst)).getD 0 * 
           (List.maximum (p.vertices.map Prod.snd)).getD 0

theorem polygon_existence : 
  (∃ p : GridPolygon, p.sides = 20 ∧ p.area = 9 ∧ isValidGridPolygon p) ∧
  (∃ p : GridPolygon, p.sides = 100 ∧ p.area = 49 ∧ isValidGridPolygon p) := by
  sorry

end NUMINAMATH_CALUDE_polygon_existence_l2384_238477


namespace NUMINAMATH_CALUDE_rectangle_area_breadth_ratio_l2384_238493

/-- Proves that for a rectangular plot with breadth 8 meters and length 10 meters more than its breadth, the ratio of its area to its breadth is 18:1. -/
theorem rectangle_area_breadth_ratio :
  let b : ℝ := 8  -- breadth
  let l : ℝ := b + 10  -- length
  let A : ℝ := l * b  -- area
  A / b = 18 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_breadth_ratio_l2384_238493


namespace NUMINAMATH_CALUDE_license_plate_ratio_l2384_238439

/-- The number of possible letters in a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate. -/
def num_digits : ℕ := 10

/-- The number of letters in an old license plate. -/
def old_letters : ℕ := 2

/-- The number of digits in an old license plate. -/
def old_digits : ℕ := 5

/-- The number of letters in a new license plate. -/
def new_letters : ℕ := 4

/-- The number of digits in a new license plate. -/
def new_digits : ℕ := 2

/-- The ratio of new possible license plates to old possible license plates. -/
theorem license_plate_ratio :
  (num_letters ^ new_letters * num_digits ^ new_digits) /
  (num_letters ^ old_letters * num_digits ^ old_digits) =
  (num_letters ^ 2 : ℚ) / (num_digits ^ 3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_license_plate_ratio_l2384_238439


namespace NUMINAMATH_CALUDE_sandy_age_l2384_238451

/-- Given two people, Sandy and Molly, where Sandy is 18 years younger than Molly
    and the ratio of their ages is 7:9, prove that Sandy is 63 years old. -/
theorem sandy_age (sandy_age molly_age : ℕ) 
    (h1 : molly_age = sandy_age + 18)
    (h2 : sandy_age * 9 = molly_age * 7) : 
  sandy_age = 63 := by
sorry

end NUMINAMATH_CALUDE_sandy_age_l2384_238451


namespace NUMINAMATH_CALUDE_sine_is_odd_and_has_zero_point_l2384_238428

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to have a zero point
def has_zero_point (f : ℝ → ℝ) : Prop := ∃ x, f x = 0

theorem sine_is_odd_and_has_zero_point :
  is_odd Real.sin ∧ has_zero_point Real.sin :=
sorry

end NUMINAMATH_CALUDE_sine_is_odd_and_has_zero_point_l2384_238428


namespace NUMINAMATH_CALUDE_wang_hua_practice_days_l2384_238480

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in the Gregorian calendar -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

def daysInMonth (year : Nat) (month : Nat) : Nat :=
  match month with
  | 2 => if isLeapYear year then 29 else 28
  | 4 | 6 | 9 | 11 => 30
  | _ => 31

def dayOfWeek (date : Date) : DayOfWeek :=
  sorry

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def countPracticeDays (year : Nat) (month : Nat) : Nat :=
  sorry

theorem wang_hua_practice_days :
  let newYearsDay2016 := Date.mk 2016 1 1
  let augustFirst2016 := Date.mk 2016 8 1
  dayOfWeek newYearsDay2016 = DayOfWeek.Friday →
  isLeapYear 2016 = true →
  countPracticeDays 2016 8 = 9 :=
by sorry

end NUMINAMATH_CALUDE_wang_hua_practice_days_l2384_238480


namespace NUMINAMATH_CALUDE_reciprocal_of_2023_l2384_238402

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_2023 : reciprocal 2023 = 1 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_2023_l2384_238402
