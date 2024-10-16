import Mathlib

namespace NUMINAMATH_CALUDE_least_sum_with_constraint_l1768_176826

theorem least_sum_with_constraint (x y z : ℕ+) : 
  (∀ a b c : ℕ+, x + y + z ≤ a + b + c) → 
  (x + y + z = 37) → 
  (5 * y = 6 * z) → 
  x = 21 := by
sorry

end NUMINAMATH_CALUDE_least_sum_with_constraint_l1768_176826


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1768_176881

theorem product_of_three_numbers (x y z : ℝ) 
  (sum_eq_20 : x + y + z = 20)
  (first_eq_four_times_sum_others : x = 4 * (y + z))
  (second_eq_seven_times_third : y = 7 * z) : 
  x * y * z = 28 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1768_176881


namespace NUMINAMATH_CALUDE_gcd_5005_11011_l1768_176874

theorem gcd_5005_11011 : Nat.gcd 5005 11011 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5005_11011_l1768_176874


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l1768_176892

/-- The function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem extreme_value_implies_a_equals_5 (a : ℝ) :
  (f' a (-3) = 0) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l1768_176892


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1768_176848

/-- The radius of the largest circle inscribed in a square, given specific distances from a point on the circle to two adjacent sides of the square. -/
theorem inscribed_circle_radius (square_side : ℝ) (dist_to_side1 : ℝ) (dist_to_side2 : ℝ) :
  square_side > 20 →
  dist_to_side1 = 8 →
  dist_to_side2 = 9 →
  ∃ (radius : ℝ),
    radius > 10 ∧
    (radius - dist_to_side1)^2 + (radius - dist_to_side2)^2 = radius^2 ∧
    radius = 29 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1768_176848


namespace NUMINAMATH_CALUDE_differences_of_geometric_progression_l1768_176822

/-- Given a geometric progression with first term a₁ and common ratio q,
    the sequence of differences between consecutive terms forms a geometric progression
    with first term a₁(q - 1) and common ratio q. -/
theorem differences_of_geometric_progression
  (a₁ : ℝ) (q : ℝ) (hq : q ≠ 1) :
  let gp : ℕ → ℝ := λ n => a₁ * q^(n - 1)
  let diff : ℕ → ℝ := λ n => gp (n + 1) - gp n
  ∀ n : ℕ, diff (n + 1) = q * diff n :=
by sorry

end NUMINAMATH_CALUDE_differences_of_geometric_progression_l1768_176822


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1768_176829

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (2, -3)
  let b : ℝ × ℝ := (m, 6)
  parallel a b → m = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1768_176829


namespace NUMINAMATH_CALUDE_major_axis_length_l1768_176816

def ellipse_equation (x y : ℝ) : Prop := y^2 / 25 + x^2 / 15 = 1

theorem major_axis_length :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), ellipse_equation x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
  max a b = 5 :=
sorry

end NUMINAMATH_CALUDE_major_axis_length_l1768_176816


namespace NUMINAMATH_CALUDE_night_temperature_l1768_176851

def noon_temperature : Int := -2
def temperature_drop : Int := 4

theorem night_temperature : 
  noon_temperature - temperature_drop = -6 := by sorry

end NUMINAMATH_CALUDE_night_temperature_l1768_176851


namespace NUMINAMATH_CALUDE_teacher_raise_percentage_l1768_176815

def former_salary : ℕ := 45000
def num_kids : ℕ := 9
def payment_per_kid : ℕ := 6000

def total_new_salary : ℕ := num_kids * payment_per_kid

def raise_amount : ℕ := total_new_salary - former_salary

def raise_percentage : ℚ := (raise_amount : ℚ) / (former_salary : ℚ) * 100

theorem teacher_raise_percentage :
  raise_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_teacher_raise_percentage_l1768_176815


namespace NUMINAMATH_CALUDE_partner_count_l1768_176805

theorem partner_count (P A : ℕ) (h1 : P / A = 2 / 63) (h2 : P / (A + 50) = 1 / 34) : P = 20 := by
  sorry

end NUMINAMATH_CALUDE_partner_count_l1768_176805


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1768_176814

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  1 / (a + 1) + 1 / (b + 3) ≥ 28 / 49 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1768_176814


namespace NUMINAMATH_CALUDE_drunk_driving_wait_time_l1768_176891

theorem drunk_driving_wait_time (p₀ r : ℝ) (h1 : p₀ = 89) (h2 : 61 = 89 * Real.exp (2 * r)) : 
  ⌈Real.log (20 / 89) / r⌉ = 8 := by
  sorry

end NUMINAMATH_CALUDE_drunk_driving_wait_time_l1768_176891


namespace NUMINAMATH_CALUDE_circle_equation_l1768_176808

/-- The ellipse with equation x²/16 + y²/4 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 16) + (p.2^2 / 4) = 1}

/-- The vertices of the ellipse -/
def EllipseVertices : Set (ℝ × ℝ) :=
  {p | p ∈ Ellipse ∧ (p.1 = 0 ∨ p.2 = 0)}

/-- The circle C passing through (6,0) and the vertices of the ellipse -/
def CircleC : Set (ℝ × ℝ) :=
  {p | ∃ (c : ℝ), (c, 0) ∈ Ellipse ∧ 
    ((p.1 - c)^2 + p.2^2 = (6 - c)^2) ∧
    (∀ v ∈ EllipseVertices, (p.1 - c)^2 + p.2^2 = (v.1 - c)^2 + v.2^2)}

theorem circle_equation : 
  CircleC = {p | (p.1 - 8/3)^2 + p.2^2 = 100/9} := by
  sorry


end NUMINAMATH_CALUDE_circle_equation_l1768_176808


namespace NUMINAMATH_CALUDE_cricket_score_problem_l1768_176821

theorem cricket_score_problem (a b c d e : ℕ) : 
  (a + b + c + d + e) / 5 = 36 ∧  -- average score
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, a = 4 * k₁ ∧ b = 4 * k₂ ∧ c = 4 * k₃ ∧ d = 4 * k₄ ∧ e = 4 * k₅) ∧  -- scores are multiples of 4
  d = e + 12 ∧  -- D scored 12 more than E
  e = a - 8 ∧  -- E scored 8 fewer than A
  b = d + e ∧  -- B scored as many as D and E combined
  b + c = 107 ∧  -- B and C scored 107 between them
  a > b ∧ a > c ∧ a > d ∧ a > e  -- A scored the maximum runs
  →
  e = 20 := by
sorry

end NUMINAMATH_CALUDE_cricket_score_problem_l1768_176821


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1768_176817

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def passesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def hasEqualIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

theorem line_through_point_with_equal_intercepts :
  ∃ (l₁ l₂ : Line),
    (passesThrough l₁ ⟨1, 2⟩ ∧ hasEqualIntercepts l₁) ∧
    (passesThrough l₂ ⟨1, 2⟩ ∧ hasEqualIntercepts l₂) ∧
    ((l₁.a = 2 ∧ l₁.b = -1 ∧ l₁.c = 0) ∨
     (l₂.a = 1 ∧ l₂.b = 1 ∧ l₂.c = -3)) :=
  sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1768_176817


namespace NUMINAMATH_CALUDE_distance_between_vertices_l1768_176838

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 121 - y^2 / 49 = 1

-- Define the vertices of the hyperbola
def vertices : Set (ℝ × ℝ) :=
  {(11, 0), (-11, 0)}

-- Theorem statement
theorem distance_between_vertices :
  ∀ (v1 v2 : ℝ × ℝ), v1 ∈ vertices → v2 ∈ vertices → v1 ≠ v2 →
  Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = 22 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l1768_176838


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1768_176852

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given conditions
  (2 * a + b) * Real.cos C + c * Real.cos B = 0 →
  c = 2 * Real.sqrt 6 / 3 →
  Real.sin A * Real.cos B = (Real.sqrt 3 - 1) / 4 →
  -- Conclusions
  C = 2 * π / 3 ∧
  (1/2 * b * c * Real.sin A : Real) = (6 - 2 * Real.sqrt 3) / 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1768_176852


namespace NUMINAMATH_CALUDE_intersection_implies_b_range_l1768_176812

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the line equation
def line_equation (x y b : ℝ) : Prop := y = -2 * x + b

-- Define the condition for line intersection with segment AB
def intersects_AB (b : ℝ) : Prop :=
  ∃ (x y : ℝ), line_equation x y b ∧ 
  ((x ≥ A.1 ∧ x ≤ B.1) ∨ (x ≤ A.1 ∧ x ≥ B.1)) ∧
  ((y ≥ A.2 ∧ y ≤ B.2) ∨ (y ≤ A.2 ∧ y ≥ B.2))

-- Theorem statement
theorem intersection_implies_b_range :
  ∀ b : ℝ, intersects_AB b → b ∈ Set.Icc (-2 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_b_range_l1768_176812


namespace NUMINAMATH_CALUDE_prob_compatible_donor_is_65_percent_l1768_176868

/-- Represents the blood types --/
inductive BloodType
  | O
  | A
  | B
  | AB

/-- Distribution of blood types in the population --/
def bloodTypeDistribution : BloodType → ℝ
  | BloodType.O  => 0.50
  | BloodType.A  => 0.15
  | BloodType.B  => 0.30
  | BloodType.AB => 0.05

/-- Predicate for blood types compatible with Type A --/
def compatibleWithA : BloodType → Prop
  | BloodType.O => True
  | BloodType.A => True
  | _ => False

/-- The probability of selecting a compatible donor for a Type A patient --/
def probCompatibleDonor : ℝ :=
  (bloodTypeDistribution BloodType.O) + (bloodTypeDistribution BloodType.A)

theorem prob_compatible_donor_is_65_percent :
  probCompatibleDonor = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_prob_compatible_donor_is_65_percent_l1768_176868


namespace NUMINAMATH_CALUDE_field_trip_van_capacity_l1768_176833

theorem field_trip_van_capacity (students : ℕ) (adults : ℕ) (vans : ℕ) :
  students = 25 → adults = 5 → vans = 6 →
  (students + adults) / vans = 5 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_van_capacity_l1768_176833


namespace NUMINAMATH_CALUDE_four_digit_to_two_digit_ratio_l1768_176847

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≥ 1 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a TwoDigitNumber to its numerical value -/
def TwoDigitNumber.toNat (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

/-- Converts a TwoDigitNumber to a four-digit number by repeating it -/
def TwoDigitNumber.toFourDigitNumber (n : TwoDigitNumber) : Nat :=
  1000 * n.tens + 100 * n.ones + 10 * n.tens + n.ones

/-- Theorem stating the ratio of the four-digit number to the original two-digit number is 101 -/
theorem four_digit_to_two_digit_ratio (n : TwoDigitNumber) :
    (n.toFourDigitNumber : ℚ) / (n.toNat : ℚ) = 101 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_to_two_digit_ratio_l1768_176847


namespace NUMINAMATH_CALUDE_andy_remaining_demerits_l1768_176863

/-- The maximum number of demerits Andy can get in a month before getting fired -/
def max_demerits : ℕ := 50

/-- The number of times Andy showed up late in the first week -/
def late_instances : ℕ := 6

/-- The number of demerits Andy gets for each late instance -/
def late_demerits : ℕ := 2

/-- The number of demerits Andy got for making an inappropriate joke in the second week -/
def joke_demerits : ℕ := 15

/-- The number of times Andy used his phone during work hours in the third week -/
def phone_instances : ℕ := 4

/-- The number of demerits Andy gets for each phone use instance -/
def phone_demerits : ℕ := 3

/-- The number of days Andy didn't tidy up his work area in the fourth week -/
def untidy_days : ℕ := 5

/-- The number of demerits Andy gets for each day of not tidying up -/
def untidy_demerits : ℕ := 1

/-- The total number of demerits Andy has accumulated so far -/
def total_demerits : ℕ := 
  late_instances * late_demerits + 
  joke_demerits + 
  phone_instances * phone_demerits + 
  untidy_days * untidy_demerits

/-- The number of additional demerits Andy can receive before getting fired -/
def additional_demerits : ℕ := max_demerits - total_demerits

theorem andy_remaining_demerits : additional_demerits = 6 := by
  sorry

end NUMINAMATH_CALUDE_andy_remaining_demerits_l1768_176863


namespace NUMINAMATH_CALUDE_matrix_product_result_l1768_176845

def odd_matrix (k : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  !![1, k; 0, 1]

def matrix_product : Matrix (Fin 2) (Fin 2) ℕ :=
  (List.range 50).foldl (λ acc i => acc * odd_matrix (2 * i + 1)) (odd_matrix 1)

theorem matrix_product_result :
  matrix_product = !![1, 2500; 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_result_l1768_176845


namespace NUMINAMATH_CALUDE_boat_speed_problem_l1768_176835

/-- Proves that given a boat traveling upstream at 3 km/h and having an average
    round-trip speed of 4.2 km/h, its downstream speed is 7 km/h. -/
theorem boat_speed_problem (upstream_speed downstream_speed average_speed : ℝ) 
    (h1 : upstream_speed = 3)
    (h2 : average_speed = 4.2)
    (h3 : average_speed = (2 * upstream_speed * downstream_speed) / (upstream_speed + downstream_speed)) :
  downstream_speed = 7 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_problem_l1768_176835


namespace NUMINAMATH_CALUDE_scenario_one_count_scenario_two_count_l1768_176887

/-- Represents the number of products --/
def total_products : ℕ := 10

/-- Represents the number of defective products --/
def defective_products : ℕ := 4

/-- Calculates the number of testing methods for scenario 1 --/
def scenario_one_methods : ℕ := sorry

/-- Calculates the number of testing methods for scenario 2 --/
def scenario_two_methods : ℕ := sorry

/-- Theorem for scenario 1 --/
theorem scenario_one_count :
  scenario_one_methods = 103680 :=
sorry

/-- Theorem for scenario 2 --/
theorem scenario_two_count :
  scenario_two_methods = 576 :=
sorry

end NUMINAMATH_CALUDE_scenario_one_count_scenario_two_count_l1768_176887


namespace NUMINAMATH_CALUDE_parabola_translation_l1768_176875

-- Define the base parabola
def base_parabola (x : ℝ) : ℝ := x^2

-- Define the transformed parabola
def transformed_parabola (x : ℝ) : ℝ := (x + 4)^2 - 5

-- Theorem stating the translation process
theorem parabola_translation :
  ∀ x : ℝ, transformed_parabola x = base_parabola (x + 4) - 5 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l1768_176875


namespace NUMINAMATH_CALUDE_sugar_concentration_mixture_l1768_176879

/-- Given two solutions with different sugar concentrations, calculate the sugar concentration of the resulting mixture --/
theorem sugar_concentration_mixture (original_concentration : ℝ) (replacement_concentration : ℝ)
  (replacement_fraction : ℝ) (h1 : original_concentration = 0.12)
  (h2 : replacement_concentration = 0.28000000000000004) (h3 : replacement_fraction = 0.25) :
  (1 - replacement_fraction) * original_concentration + replacement_fraction * replacement_concentration = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_sugar_concentration_mixture_l1768_176879


namespace NUMINAMATH_CALUDE_total_fish_l1768_176809

def gold_fish : ℕ := 15
def blue_fish : ℕ := 7

theorem total_fish : gold_fish + blue_fish = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_l1768_176809


namespace NUMINAMATH_CALUDE_max_real_axis_length_l1768_176855

/-- Represents a hyperbola with given asymptotes and passing through a specific point -/
structure Hyperbola where
  /-- The asymptotes of the hyperbola are of the form 2x ± y = 0 -/
  asymptotes : Unit
  /-- The hyperbola passes through the intersection of two lines -/
  intersection_point : ℝ × ℝ
  /-- The parameter t determines the intersection point -/
  t : ℝ
  /-- The intersection point satisfies the equations of both lines -/
  satisfies_line1 : intersection_point.1 + intersection_point.2 = 3
  satisfies_line2 : 2 * intersection_point.1 - intersection_point.2 = -3 * t
  /-- The parameter t is within the specified range -/
  t_range : -2 ≤ t ∧ t ≤ 5

/-- The length of the real axis of the hyperbola -/
def real_axis_length (h : Hyperbola) : ℝ := sorry

/-- Theorem stating the maximum possible length of the real axis -/
theorem max_real_axis_length (h : Hyperbola) : 
  real_axis_length h ≤ 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_max_real_axis_length_l1768_176855


namespace NUMINAMATH_CALUDE_volume_difference_rectangular_prisms_volume_difference_specific_bowls_l1768_176883

/-- The volume difference between two rectangular prisms with the same width and length
    but different heights is equal to the product of the width, length, and the difference in heights. -/
theorem volume_difference_rectangular_prisms
  (w : ℝ) (l : ℝ) (h₁ : ℝ) (h₂ : ℝ)
  (hw : w > 0) (hl : l > 0) (hh₁ : h₁ > 0) (hh₂ : h₂ > 0) :
  w * l * h₁ - w * l * h₂ = w * l * (h₁ - h₂) :=
by sorry

/-- The volume difference between two specific bowls -/
theorem volume_difference_specific_bowls :
  (16 : ℝ) * 14 * 9 - (16 : ℝ) * 14 * 4 = 1120 :=
by sorry

end NUMINAMATH_CALUDE_volume_difference_rectangular_prisms_volume_difference_specific_bowls_l1768_176883


namespace NUMINAMATH_CALUDE_hall_covering_cost_l1768_176861

/-- Calculates the total expenditure for covering the interior of a rectangular hall with mat -/
def total_expenditure (length width height cost_per_sqm : ℝ) : ℝ :=
  let floor_area := length * width
  let wall_area := 2 * (length * height + width * height)
  let total_area := 2 * floor_area + wall_area
  total_area * cost_per_sqm

/-- Proves that the total expenditure for covering a specific hall is Rs. 47,500 -/
theorem hall_covering_cost :
  total_expenditure 20 15 5 50 = 47500 := by
  sorry

end NUMINAMATH_CALUDE_hall_covering_cost_l1768_176861


namespace NUMINAMATH_CALUDE_new_student_weight_l1768_176873

theorem new_student_weight (initial_count : ℕ) (initial_avg : ℚ) (new_avg : ℚ) :
  initial_count = 29 →
  initial_avg = 28 →
  new_avg = 27.5 →
  (initial_count * initial_avg + (initial_count + 1) * new_avg - initial_count * initial_avg) / (initial_count + 1) = 13 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_l1768_176873


namespace NUMINAMATH_CALUDE_circle_line_distance_l1768_176894

theorem circle_line_distance (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + 1 = 0) → 
  (∀ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 4) →
  (|a + 1| / Real.sqrt (a^2 + 1) = 1) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_circle_line_distance_l1768_176894


namespace NUMINAMATH_CALUDE_unique_solution_l1768_176867

/-- 
Given two positive integers x and y, prove that if they satisfy the equations
x^y + 4 = y^x and 3x^y = y^x + 10, then x = 7 and y = 1.
-/
theorem unique_solution (x y : ℕ+) 
  (h1 : x^(y:ℕ) + 4 = y^(x:ℕ)) 
  (h2 : 3 * x^(y:ℕ) = y^(x:ℕ) + 10) : 
  x = 7 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1768_176867


namespace NUMINAMATH_CALUDE_largest_unexpressible_sum_l1768_176824

def min_num : ℕ := 135
def max_num : ℕ := 144
def target : ℕ := 2024

theorem largest_unexpressible_sum : 
  (∀ n : ℕ, n > target → ∃ k : ℕ, k * min_num ≤ n ∧ n ≤ k * max_num) ∧
  (∀ k : ℕ, k * min_num > target ∨ target > k * max_num) :=
sorry

end NUMINAMATH_CALUDE_largest_unexpressible_sum_l1768_176824


namespace NUMINAMATH_CALUDE_units_digit_of_product_l1768_176862

theorem units_digit_of_product (a b c : ℕ) : (4^1001 * 8^1002 * 12^1003) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l1768_176862


namespace NUMINAMATH_CALUDE_total_cost_is_100_l1768_176818

/-- Calculates the total cost in dollars for using whiteboards in all classes for one day -/
def whiteboard_cost (num_classes : ℕ) (boards_per_class : ℕ) (ink_per_board : ℕ) (cost_per_ml : ℚ) : ℚ :=
  num_classes * boards_per_class * ink_per_board * cost_per_ml

/-- Proves that the total cost for using whiteboards in all classes for one day is $100 -/
theorem total_cost_is_100 : 
  whiteboard_cost 5 2 20 (1/2) = 100 := by
  sorry

#eval whiteboard_cost 5 2 20 (1/2)

end NUMINAMATH_CALUDE_total_cost_is_100_l1768_176818


namespace NUMINAMATH_CALUDE_cube_sum_root_l1768_176860

theorem cube_sum_root : Real.sqrt (3^3 + 3^3 + 3^3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_root_l1768_176860


namespace NUMINAMATH_CALUDE_scenic_area_ticket_sales_l1768_176802

/-- Scenic area ticket sales problem -/
theorem scenic_area_ticket_sales 
  (parent_child_price : ℝ) 
  (family_price : ℝ) 
  (parent_child_presale : ℝ) 
  (family_presale : ℝ) 
  (volume_difference : ℕ) 
  (parent_child_planned : ℕ) 
  (family_planned : ℕ) 
  (a : ℝ) :
  family_price = 2 * parent_child_price →
  parent_child_presale = 21000 →
  family_presale = 10500 →
  (parent_child_presale / parent_child_price) - (family_presale / family_price) = volume_difference →
  parent_child_planned = 1600 →
  family_planned = 400 →
  (parent_child_price + 3/4 * a) * (parent_child_planned - 32 * a) + 
    (family_price + a) * family_planned = 
    parent_child_price * parent_child_planned + family_price * family_planned →
  parent_child_price = 35 ∧ a = 20 := by
  sorry

end NUMINAMATH_CALUDE_scenic_area_ticket_sales_l1768_176802


namespace NUMINAMATH_CALUDE_centroid_coincides_with_inscribed_sphere_center_l1768_176899

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents the centroids of faces opposite to vertices -/
structure FaceCentroids where
  SA : Point3D
  SB : Point3D
  SC : Point3D
  SD : Point3D

/-- Calculates the centroid of a system of homogeneous thin plates -/
def systemCentroid (t : Tetrahedron) (fc : FaceCentroids) : Point3D :=
  sorry

/-- Calculates the center of the inscribed sphere of a tetrahedron -/
def inscribedSphereCenter (t : Tetrahedron) : Point3D :=
  sorry

/-- Main theorem: The centroid of the system coincides with the center of the inscribed sphere -/
theorem centroid_coincides_with_inscribed_sphere_center 
  (t : Tetrahedron) (fc : FaceCentroids) :
  systemCentroid t fc = inscribedSphereCenter (Tetrahedron.mk fc.SA fc.SB fc.SC fc.SD) :=
by
  sorry

end NUMINAMATH_CALUDE_centroid_coincides_with_inscribed_sphere_center_l1768_176899


namespace NUMINAMATH_CALUDE_ten_player_tournament_matches_l1768_176853

/-- Calculates the number of matches in a round-robin tournament. -/
def roundRobinMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A 10-person round-robin tennis tournament has 45 matches. -/
theorem ten_player_tournament_matches :
  roundRobinMatches 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_player_tournament_matches_l1768_176853


namespace NUMINAMATH_CALUDE_switch_pairs_relation_l1768_176890

/-- Represents a row in the sequence --/
structure Row where
  switchPairs : ℕ
  oddBlocks : ℕ

/-- The relationship between switch pairs and odd blocks in a row --/
axiom switch_pairs_odd_blocks (r : Row) : r.switchPairs = 2 * r.oddBlocks

/-- The existence of at least one switch pair above each odd block --/
axiom switch_pair_above_odd_block (rn : Row) (rn_minus_1 : Row) :
  rn.oddBlocks ≤ rn_minus_1.switchPairs

/-- Theorem: The number of switch pairs in row n is at most twice 
    the number of switch pairs in row n-1 --/
theorem switch_pairs_relation (rn : Row) (rn_minus_1 : Row) :
  rn.switchPairs ≤ 2 * rn_minus_1.switchPairs := by
  sorry

end NUMINAMATH_CALUDE_switch_pairs_relation_l1768_176890


namespace NUMINAMATH_CALUDE_max_value_f_neg_one_range_of_a_l1768_176828

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x * f a x

-- Define the function h
def h (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - (2*a - 1) * x + a - 1

-- Theorem for the maximum value of f when a = -1
theorem max_value_f_neg_one :
  ∃ (max : ℝ), max = -1 ∧ ∀ x > 0, f (-1) x ≤ max :=
sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∀ x ≥ 1, g a x ≤ h a x) → a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_f_neg_one_range_of_a_l1768_176828


namespace NUMINAMATH_CALUDE_y_work_time_l1768_176858

/-- The time it takes for y to complete the work alone, given the conditions -/
def time_y_alone (time_x time_yz time_xz : ℝ) : ℝ :=
  24

/-- Theorem stating that y takes 24 hours to complete the work alone -/
theorem y_work_time (time_x time_yz time_xz : ℝ) 
  (hx : time_x = 8) 
  (hyz : time_yz = 6) 
  (hxz : time_xz = 4) : 
  time_y_alone time_x time_yz time_xz = 24 := by
  sorry

#check y_work_time

end NUMINAMATH_CALUDE_y_work_time_l1768_176858


namespace NUMINAMATH_CALUDE_line_segment_ratio_l1768_176898

theorem line_segment_ratio (x y z s : ℝ) 
  (h1 : x < y ∧ y < z)
  (h2 : x / y = y / z)
  (h3 : x + y + z = s)
  (h4 : x + y = z) :
  x / y = (-1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_line_segment_ratio_l1768_176898


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l1768_176807

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = a * b - 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = x * y - 1 → a + 2 * b ≤ x + 2 * y ∧
  ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = a₀ * b₀ - 1 ∧ a₀ + 2 * b₀ = 5 + 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l1768_176807


namespace NUMINAMATH_CALUDE_negation_equivalence_l1768_176810

theorem negation_equivalence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1768_176810


namespace NUMINAMATH_CALUDE_max_min_sum_difference_l1768_176841

def three_digit_integer (a b c : ℕ) : Prop :=
  100 ≤ a * 100 + b * 10 + c ∧ a * 100 + b * 10 + c < 1000

def all_different (a b c d e f g h i : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i

theorem max_min_sum_difference :
  ∀ (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ : ℕ),
  three_digit_integer a₁ b₁ c₁ →
  three_digit_integer a₂ b₂ c₂ →
  three_digit_integer a₃ b₃ c₃ →
  all_different a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ →
  (∀ (x₁ y₁ z₁ x₂ y₂ z₂ x₃ y₃ z₃ : ℕ),
    three_digit_integer x₁ y₁ z₁ →
    three_digit_integer x₂ y₂ z₂ →
    three_digit_integer x₃ y₃ z₃ →
    all_different x₁ y₁ z₁ x₂ y₂ z₂ x₃ y₃ z₃ →
    (a₁ * 100 + b₁ * 10 + c₁) + (a₂ * 100 + b₂ * 10 + c₂) + (a₃ * 100 + b₃ * 10 + c₃) ≥
    (x₁ * 100 + y₁ * 10 + z₁) + (x₂ * 100 + y₂ * 10 + z₂) + (x₃ * 100 + y₃ * 10 + z₃)) →
  (∀ (p₁ q₁ r₁ p₂ q₂ r₂ p₃ q₃ r₃ : ℕ),
    three_digit_integer p₁ q₁ r₁ →
    three_digit_integer p₂ q₂ r₂ →
    three_digit_integer p₃ q₃ r₃ →
    all_different p₁ q₁ r₁ p₂ q₂ r₂ p₃ q₃ r₃ →
    (p₁ * 100 + q₁ * 10 + r₁) + (p₂ * 100 + q₂ * 10 + r₂) + (p₃ * 100 + q₃ * 10 + r₃) ≥
    (a₁ * 100 + b₁ * 10 + c₁) + (a₂ * 100 + b₂ * 10 + c₂) + (a₃ * 100 + b₃ * 10 + c₃)) →
  ((a₁ * 100 + b₁ * 10 + c₁) + (a₂ * 100 + b₂ * 10 + c₂) + (a₃ * 100 + b₃ * 10 + c₃)) -
  ((p₁ * 100 + q₁ * 10 + r₁) + (p₂ * 100 + q₂ * 10 + r₂) + (p₃ * 100 + q₃ * 10 + r₃)) = 1845 :=
by sorry

end NUMINAMATH_CALUDE_max_min_sum_difference_l1768_176841


namespace NUMINAMATH_CALUDE_games_to_reach_target_win_rate_l1768_176811

def initial_games : ℕ := 20
def initial_win_rate : ℚ := 95 / 100
def target_win_rate : ℚ := 96 / 100

theorem games_to_reach_target_win_rate :
  let initial_wins := (initial_games : ℚ) * initial_win_rate
  ∃ (additional_games : ℕ),
    (initial_wins + additional_games) / (initial_games + additional_games) = target_win_rate ∧
    additional_games = 5 := by
  sorry

end NUMINAMATH_CALUDE_games_to_reach_target_win_rate_l1768_176811


namespace NUMINAMATH_CALUDE_sum_product_bound_l1768_176897

theorem sum_product_bound (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  -1/2 ≤ a*b + b*c + c*a ∧ a*b + b*c + c*a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_bound_l1768_176897


namespace NUMINAMATH_CALUDE_birth_rate_calculation_l1768_176825

/-- Represents the average birth rate in people per two seconds -/
def average_birth_rate : ℝ := 4

/-- Represents the death rate in people per two seconds -/
def death_rate : ℝ := 2

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- Represents the net population increase in one day -/
def net_increase_per_day : ℕ := 86400

theorem birth_rate_calculation :
  average_birth_rate = 4 :=
by
  sorry

#check birth_rate_calculation

end NUMINAMATH_CALUDE_birth_rate_calculation_l1768_176825


namespace NUMINAMATH_CALUDE_f_explicit_function_l1768_176885

-- Define the function f
def f : ℝ → ℝ := fun x => x^2 - 1

-- State the theorem
theorem f_explicit_function (x : ℝ) (h : x ≥ 0) : 
  f (Real.sqrt x + 1) = x + 2 * Real.sqrt x ↔ (∀ y ≥ 1, f y = y^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_f_explicit_function_l1768_176885


namespace NUMINAMATH_CALUDE_lcm_gcd_sum_theorem_l1768_176880

theorem lcm_gcd_sum_theorem : 
  (Nat.lcm 12 18 * Nat.gcd 12 18) + (Nat.lcm 10 15 * Nat.gcd 10 15) = 366 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_sum_theorem_l1768_176880


namespace NUMINAMATH_CALUDE_employee_pay_percentage_l1768_176865

/-- Given two employees X and Y with a total pay of 572, where Y is paid 260,
    prove that X's pay as a percentage of Y's pay is 120%. -/
theorem employee_pay_percentage (X Y : ℝ) : 
  Y = 260 → X + Y = 572 → (X / Y) * 100 = 120 := by sorry

end NUMINAMATH_CALUDE_employee_pay_percentage_l1768_176865


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l1768_176866

/-- Two lines in the plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l₁ l₂ : Line) : Prop := l₁.slope = l₂.slope

theorem parallel_lines_a_value :
  ∀ a : ℝ,
  let l₁ : Line := ⟨1, a/2⟩
  let l₂ : Line := ⟨a^2 - 3, 1⟩
  parallel l₁ l₂ → a = -2 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l1768_176866


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1768_176859

theorem arithmetic_sequence_sum : 
  ∀ (a l d n : ℤ),
    a = -41 →
    l = 1 →
    d = 2 →
    n = 22 →
    a + (n - 1) * d = l →
    (n * (a + l)) / 2 = -440 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1768_176859


namespace NUMINAMATH_CALUDE_sparrow_population_decline_l1768_176800

/-- Proves that the smallest integer t satisfying (0.6^t ≤ 0.05) is 6 -/
theorem sparrow_population_decline (t : ℕ) : 
  (∀ k : ℕ, k < t → (0.6 : ℝ)^k > 0.05) ∧ (0.6 : ℝ)^t ≤ 0.05 → t = 6 :=
by sorry

end NUMINAMATH_CALUDE_sparrow_population_decline_l1768_176800


namespace NUMINAMATH_CALUDE_donald_oranges_l1768_176864

theorem donald_oranges (initial_oranges found_oranges : ℕ) 
  (h1 : initial_oranges = 4)
  (h2 : found_oranges = 5) :
  initial_oranges + found_oranges = 9 := by
  sorry

end NUMINAMATH_CALUDE_donald_oranges_l1768_176864


namespace NUMINAMATH_CALUDE_nap_duration_l1768_176837

/-- Proves that if a person takes 3 naps per week for 70 days, 
    and the total duration of naps is 60 hours, then each nap is 2 hours long. -/
theorem nap_duration (naps_per_week : ℕ) (total_days : ℕ) (total_hours : ℕ) :
  naps_per_week = 3 →
  total_days = 70 →
  total_hours = 60 →
  (total_hours : ℚ) / ((naps_per_week * total_days : ℚ) / 7) = 2 := by
  sorry

end NUMINAMATH_CALUDE_nap_duration_l1768_176837


namespace NUMINAMATH_CALUDE_range_of_b_l1768_176869

theorem range_of_b (a b c : ℝ) (sum_eq : a + b + c = 9) (prod_eq : a * b + b * c + c * a = 24) :
  1 ≤ b ∧ b ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_range_of_b_l1768_176869


namespace NUMINAMATH_CALUDE_annas_money_l1768_176836

def chewing_gum_cost : ℝ := 3 * 1.00
def chocolate_bars_cost : ℝ := 5 * 1.00
def candy_canes_cost : ℝ := 2 * 0.50
def money_left : ℝ := 1.00

def total_spent : ℝ := chewing_gum_cost + chocolate_bars_cost + candy_canes_cost

theorem annas_money (money_from_mom : ℝ) : 
  money_from_mom = total_spent + money_left :=
by sorry

end NUMINAMATH_CALUDE_annas_money_l1768_176836


namespace NUMINAMATH_CALUDE_hydrogen_moles_formed_l1768_176878

/-- Represents a chemical element --/
structure Element where
  name : String
  atomic_mass : Float

/-- Represents a chemical compound --/
structure Compound where
  formula : String
  elements : List (Element × Nat)

/-- Represents a chemical reaction --/
structure Reaction where
  reactants : List (Compound × Float)
  products : List (Compound × Float)

/-- Calculate the molar mass of a compound --/
def molar_mass (c : Compound) : Float :=
  c.elements.foldl (fun acc (elem, count) => acc + elem.atomic_mass * count.toFloat) 0

/-- Calculate the number of moles given mass and molar mass --/
def moles (mass : Float) (molar_mass : Float) : Float :=
  mass / molar_mass

/-- The main theorem --/
theorem hydrogen_moles_formed
  (carbon : Element)
  (hydrogen : Element)
  (benzene : Compound)
  (methane : Compound)
  (toluene : Compound)
  (h2 : Compound)
  (reaction : Reaction)
  (benzene_mass : Float) :
  carbon.atomic_mass = 12.01 →
  hydrogen.atomic_mass = 1.008 →
  benzene.elements = [(carbon, 6), (hydrogen, 6)] →
  methane.elements = [(carbon, 1), (hydrogen, 4)] →
  toluene.elements = [(carbon, 7), (hydrogen, 8)] →
  h2.elements = [(hydrogen, 2)] →
  reaction.reactants = [(benzene, 1), (methane, 1)] →
  reaction.products = [(toluene, 1), (h2, 1)] →
  benzene_mass = 156 →
  moles benzene_mass (molar_mass benzene) = 2 →
  moles benzene_mass (molar_mass benzene) = moles 2 (molar_mass h2) :=
by sorry

end NUMINAMATH_CALUDE_hydrogen_moles_formed_l1768_176878


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1768_176834

theorem simplify_and_rationalize (x y z : ℝ) :
  x > 0 → y > 0 → z > 0 →
  (x / Real.sqrt 7) * (Real.sqrt 9 / Real.sqrt 11) * (z / Real.sqrt 13) =
  15 * Real.sqrt 1001 / 1001 →
  x = Real.sqrt 5 ∧ y = Real.sqrt 9 ∧ z = Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1768_176834


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l1768_176843

theorem chess_tournament_participants (x : ℕ) (y : ℕ) : 
  (2 * y + 8 = (x + 2) * (x + 1) / 2) →
  (x * y + 8 = (x + 2) * (x + 1) / 2) →
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l1768_176843


namespace NUMINAMATH_CALUDE_perfect_square_between_prime_sums_l1768_176806

def S (n : ℕ) : ℕ := sorry

theorem perfect_square_between_prime_sums (n : ℕ) :
  ∃ k : ℕ, S n < k^2 ∧ k^2 < S (n + 1) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_between_prime_sums_l1768_176806


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1768_176830

theorem complex_equation_solution (a : ℝ) : (Complex.I * a + 1) * (a - Complex.I) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1768_176830


namespace NUMINAMATH_CALUDE_room_width_l1768_176820

/-- Given a rectangular room with area 10 square feet and length 5 feet, prove the width is 2 feet -/
theorem room_width (area : ℝ) (length : ℝ) (width : ℝ) : 
  area = 10 → length = 5 → area = length * width → width = 2 := by
  sorry

end NUMINAMATH_CALUDE_room_width_l1768_176820


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l1768_176877

/-- A geometric sequence with common ratio q -/
def geometric_sequence (q : ℝ) : ℕ → ℝ := fun n ↦ q ^ (n - 1)

theorem geometric_sequence_properties (q : ℝ) (h_q : 0 < q ∧ q < 1) :
  let a := geometric_sequence q
  (∀ n : ℕ, a (n + 1) < a n) ∧
  (∃ k : ℕ+, a (k + 1) = (a k + a (k + 2)) / 2 → q = (1 - Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l1768_176877


namespace NUMINAMATH_CALUDE_golden_ratio_greater_than_half_l1768_176870

theorem golden_ratio_greater_than_half : (Real.sqrt 5 - 1) / 2 > 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_greater_than_half_l1768_176870


namespace NUMINAMATH_CALUDE_calculator_game_sum_l1768_176857

/-- Represents the operations performed on the calculators. -/
def calculatorOperations (n : ℕ) (a b c : ℕ) : ℕ × ℕ × ℕ :=
  (a^3, b^2, c + 1)

/-- Applies the operations n times to the initial values. -/
def applyNTimes (n : ℕ) : ℕ × ℕ × ℕ :=
  match n with
  | 0 => (2, 1, 0)
  | m + 1 => calculatorOperations m (applyNTimes m).1 (applyNTimes m).2.1 (applyNTimes m).2.2

/-- The main theorem to be proved. -/
theorem calculator_game_sum :
  let (a, b, c) := applyNTimes 50
  a + b + c = 307 := by sorry

end NUMINAMATH_CALUDE_calculator_game_sum_l1768_176857


namespace NUMINAMATH_CALUDE_area_enclosed_by_g_l1768_176801

open Real MeasureTheory

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + π / 3)

theorem area_enclosed_by_g : 
  ∫ (x : ℝ) in (0)..(π / 3), g x = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_g_l1768_176801


namespace NUMINAMATH_CALUDE_identify_genuine_coin_l1768_176840

/-- Represents the result of a weighing -/
inductive WeighingResult
  | Equal : WeighingResult
  | Unequal : WeighingResult

/-- Represents a coin -/
inductive Coin
  | Genuine : Coin
  | Counterfeit : Coin

/-- Represents a weighing operation -/
def weighing (a b : Coin) : WeighingResult :=
  match a, b with
  | Coin.Genuine, Coin.Genuine => WeighingResult.Equal
  | Coin.Counterfeit, Coin.Counterfeit => WeighingResult.Equal
  | _, _ => WeighingResult.Unequal

/-- Theorem stating that at least one genuine coin can be identified in at most 2 weighings -/
theorem identify_genuine_coin
  (coins : Fin 5 → Coin)
  (h_genuine : ∃ i j k, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ coins i = Coin.Genuine ∧ coins j = Coin.Genuine ∧ coins k = Coin.Genuine)
  (h_counterfeit : ∃ i j, i ≠ j ∧ coins i = Coin.Counterfeit ∧ coins j = Coin.Counterfeit) :
  ∃ (w₁ w₂ : Fin 5 × Fin 5), ∃ (i : Fin 5), coins i = Coin.Genuine :=
sorry

end NUMINAMATH_CALUDE_identify_genuine_coin_l1768_176840


namespace NUMINAMATH_CALUDE_cheryl_material_calculation_l1768_176844

/-- The amount of the second type of material Cheryl needed for her project -/
def second_material_amount : ℚ := 1 / 8

/-- The amount of the first type of material Cheryl bought -/
def first_material_amount : ℚ := 2 / 9

/-- The amount of material Cheryl had left after the project -/
def leftover_amount : ℚ := 4 / 18

/-- The total amount of material Cheryl used -/
def total_used : ℚ := 1 / 8

theorem cheryl_material_calculation :
  second_material_amount = 
    (first_material_amount + leftover_amount + total_used) - first_material_amount := by
  sorry

end NUMINAMATH_CALUDE_cheryl_material_calculation_l1768_176844


namespace NUMINAMATH_CALUDE_equivalence_condition_l1768_176872

/-- Hyperbola C with equation x² - y²/3 = 1 -/
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- Left vertex of the hyperbola -/
def A₁ : ℝ × ℝ := (-1, 0)

/-- Right vertex of the hyperbola -/
def A₂ : ℝ × ℝ := (1, 0)

/-- Moving line l with equation x = my + n -/
def line_l (m n y : ℝ) : ℝ := m * y + n

/-- Intersection point T of A₁M and A₂N -/
structure Point_T (m n : ℝ) where
  x₀ : ℝ
  y₀ : ℝ
  on_A₁M : ∃ (x₁ y₁ : ℝ), hyperbola_C x₁ y₁ ∧ y₀ = (y₁ / (x₁ + 1)) * (x₀ + 1)
  on_A₂N : ∃ (x₂ y₂ : ℝ), hyperbola_C x₂ y₂ ∧ y₀ = (y₂ / (x₂ - 1)) * (x₀ - 1)
  on_line_l : x₀ = line_l m n y₀

/-- The main theorem to prove -/
theorem equivalence_condition (m : ℝ) :
  ∀ (n : ℝ), (∃ (T : Point_T m n), n = 2 ↔ T.x₀ = 1/2) := by sorry

end NUMINAMATH_CALUDE_equivalence_condition_l1768_176872


namespace NUMINAMATH_CALUDE_sum_f_odd_points_l1768_176849

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_zero : f 0 = 2
axiom f_translated_odd : ∀ x, f (x - 1) = -f (-x - 1)

-- State the theorem
theorem sum_f_odd_points :
  f 1 + f 3 + f 5 + f 7 + f 9 = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_f_odd_points_l1768_176849


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1768_176854

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (6 * a^3 - 803 * a + 1606 = 0) → 
  (6 * b^3 - 803 * b + 1606 = 0) → 
  (6 * c^3 - 803 * c + 1606 = 0) → 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 803 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1768_176854


namespace NUMINAMATH_CALUDE_number_divisibility_l1768_176850

theorem number_divisibility (N : ℕ) : 
  N % 7 = 0 ∧ N % 11 = 2 → N / 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_divisibility_l1768_176850


namespace NUMINAMATH_CALUDE_gauss_family_mean_age_l1768_176871

def gauss_family_ages : List ℕ := [7, 7, 7, 14, 15]

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem gauss_family_mean_age :
  mean gauss_family_ages = 10 := by
  sorry

end NUMINAMATH_CALUDE_gauss_family_mean_age_l1768_176871


namespace NUMINAMATH_CALUDE_job_completion_time_l1768_176804

theorem job_completion_time (a_time b_time : ℝ) (combined_time : ℝ) (combined_work : ℝ) : 
  a_time = 15 →
  combined_time = 8 →
  combined_work = 0.9333333333333333 →
  combined_work = combined_time * (1 / a_time + 1 / b_time) →
  b_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l1768_176804


namespace NUMINAMATH_CALUDE_pen_ratio_problem_l1768_176884

theorem pen_ratio_problem (blue_pens green_pens : ℕ) : 
  (blue_pens : ℚ) / green_pens = 4 / 3 →
  blue_pens = 16 →
  green_pens = 12 := by
sorry

end NUMINAMATH_CALUDE_pen_ratio_problem_l1768_176884


namespace NUMINAMATH_CALUDE_bruce_fruit_purchase_total_l1768_176886

/-- Calculates the discounted price for a fruit purchase -/
def discountedPrice (quantity : ℕ) (pricePerKg : ℚ) (discountPercentage : ℚ) : ℚ :=
  let originalPrice := quantity * pricePerKg
  originalPrice - (originalPrice * discountPercentage / 100)

/-- Represents Bruce's fruit purchases -/
structure FruitPurchase where
  grapes : ℕ × ℚ × ℚ
  mangoes : ℕ × ℚ × ℚ
  oranges : ℕ × ℚ × ℚ
  apples : ℕ × ℚ × ℚ

/-- Calculates the total amount paid for all fruit purchases -/
def totalAmountPaid (purchase : FruitPurchase) : ℚ :=
  discountedPrice purchase.grapes.1 purchase.grapes.2.1 purchase.grapes.2.2 +
  discountedPrice purchase.mangoes.1 purchase.mangoes.2.1 purchase.mangoes.2.2 +
  discountedPrice purchase.oranges.1 purchase.oranges.2.1 purchase.oranges.2.2 +
  discountedPrice purchase.apples.1 purchase.apples.2.1 purchase.apples.2.2

theorem bruce_fruit_purchase_total :
  let purchase : FruitPurchase := {
    grapes := (9, 70, 10),
    mangoes := (7, 55, 5),
    oranges := (5, 45, 15),
    apples := (3, 80, 20)
  }
  totalAmountPaid purchase = 1316.25 := by
  sorry

end NUMINAMATH_CALUDE_bruce_fruit_purchase_total_l1768_176886


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l1768_176827

def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - 5

theorem f_monotonicity_and_extrema :
  (∀ x y, x < y ∧ x < 1 → f x < f y) ∧
  (∀ x y, 3 < x ∧ x < y → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y ∧ y < 3 → f x > f y) ∧
  (∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → f x < f 1) ∧
  (∃ δ > 0, ∀ x, 0 < |x - 3| ∧ |x - 3| < δ → f x > f 3) ∧
  f 1 = -1 ∧
  f 3 = -5 := by
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l1768_176827


namespace NUMINAMATH_CALUDE_meg_cat_weight_l1768_176813

theorem meg_cat_weight (meg_weight anne_weight : ℝ) 
  (h1 : meg_weight / anne_weight = 5 / 7)
  (h2 : anne_weight = meg_weight + 8) : 
  meg_weight = 20 := by
sorry

end NUMINAMATH_CALUDE_meg_cat_weight_l1768_176813


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l1768_176896

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

theorem f_max_min_on_interval :
  let a := 1
  let b := 3
  ∃ (x_max x_min : ℝ), a ≤ x_max ∧ x_max ≤ b ∧ a ≤ x_min ∧ x_min ≤ b ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f x_max) ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x_min ≤ f x) ∧
    f x_max = 4 ∧ x_max = 3 ∧ f x_min = 0 ∧ x_min = 2 :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l1768_176896


namespace NUMINAMATH_CALUDE_hidden_square_exists_l1768_176803

theorem hidden_square_exists (ℓ : ℕ) : ∃ (x y : ℤ) (p : Fin ℓ → Fin ℓ → ℕ), 
  (∀ (i j : Fin ℓ), Nat.Prime (p i j)) ∧ 
  (∀ (i j k m : Fin ℓ), i ≠ k ∨ j ≠ m → p i j ≠ p k m) ∧
  (∀ (i j : Fin ℓ), x ≡ -i.val [ZMOD (p i j)] ∧ y ≡ -j.val [ZMOD (p i j)]) :=
sorry

end NUMINAMATH_CALUDE_hidden_square_exists_l1768_176803


namespace NUMINAMATH_CALUDE_max_odd_sequence_length_l1768_176856

/-- The type of sequences where each term is obtained by adding the largest digit of the previous term --/
def DigitAddSequence := ℕ → ℕ

/-- The largest digit of a natural number --/
def largest_digit (n : ℕ) : ℕ := sorry

/-- The property that a sequence follows the digit addition rule --/
def is_digit_add_sequence (s : DigitAddSequence) : Prop :=
  ∀ n, s (n + 1) = s n + largest_digit (s n)

/-- The property that a number is odd --/
def is_odd (n : ℕ) : Prop := n % 2 = 1

/-- The length of a sequence of successive odd terms starting from a given index --/
def odd_sequence_length (s : DigitAddSequence) (start : ℕ) : ℕ := sorry

/-- The theorem stating that the maximal number of successive odd terms is 5 --/
theorem max_odd_sequence_length (s : DigitAddSequence) (h : is_digit_add_sequence s) :
  ∀ start, odd_sequence_length s start ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_odd_sequence_length_l1768_176856


namespace NUMINAMATH_CALUDE_michael_spending_l1768_176882

def fair_spending (initial_amount snack_cost : ℕ) : ℕ :=
  let game_cost := 3 * snack_cost
  let total_spent := snack_cost + game_cost
  initial_amount - total_spent

theorem michael_spending :
  fair_spending 80 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_michael_spending_l1768_176882


namespace NUMINAMATH_CALUDE_product_of_odd_is_even_correct_propositions_count_l1768_176876

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The product of two odd functions is even -/
theorem product_of_odd_is_even (f g : ℝ → ℝ) (hf : IsOdd f) (hg : IsOdd g) :
    IsEven (fun x ↦ f x * g x) := by
  sorry

/-- There are exactly two correct propositions among the original, converse, negation, and contrapositive -/
theorem correct_propositions_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_product_of_odd_is_even_correct_propositions_count_l1768_176876


namespace NUMINAMATH_CALUDE_hari_investment_is_8280_l1768_176888

/-- Represents the business partnership between Praveen and Hari --/
structure Partnership where
  praveen_investment : ℕ
  praveen_months : ℕ
  hari_months : ℕ
  total_months : ℕ
  profit_ratio_praveen : ℕ
  profit_ratio_hari : ℕ

/-- Calculates Hari's investment given the partnership details --/
def calculate_hari_investment (p : Partnership) : ℕ :=
  (3 * p.praveen_investment * p.total_months) / (2 * p.hari_months)

/-- Theorem stating that Hari's investment is 8280 Rs given the specific partnership conditions --/
theorem hari_investment_is_8280 :
  let p : Partnership := {
    praveen_investment := 3220,
    praveen_months := 12,
    hari_months := 7,
    total_months := 12,
    profit_ratio_praveen := 2,
    profit_ratio_hari := 3
  }
  calculate_hari_investment p = 8280 := by
  sorry


end NUMINAMATH_CALUDE_hari_investment_is_8280_l1768_176888


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1768_176842

theorem rationalize_denominator : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1768_176842


namespace NUMINAMATH_CALUDE_cookie_difference_l1768_176846

theorem cookie_difference (alyssa_cookies aiyanna_cookies : ℕ) 
  (h1 : alyssa_cookies = 129) (h2 : aiyanna_cookies = 140) : 
  aiyanna_cookies - alyssa_cookies = 11 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l1768_176846


namespace NUMINAMATH_CALUDE_sixth_score_for_target_mean_l1768_176819

def emily_scores : List ℕ := [88, 90, 85, 92, 97]

def target_mean : ℚ := 91

theorem sixth_score_for_target_mean :
  let all_scores := emily_scores ++ [94]
  (all_scores.sum : ℚ) / all_scores.length = target_mean := by sorry

end NUMINAMATH_CALUDE_sixth_score_for_target_mean_l1768_176819


namespace NUMINAMATH_CALUDE_final_selling_price_l1768_176839

/-- Given an original price and a first discount, calculate the final selling price after an additional 20% discount -/
theorem final_selling_price (m n : ℝ) : 
  let original_price := m
  let first_discount := n
  let price_after_first_discount := original_price - first_discount
  let second_discount_rate := (20 : ℝ) / 100
  let final_price := price_after_first_discount * (1 - second_discount_rate)
  final_price = (4/5) * (m - n) := by
sorry

end NUMINAMATH_CALUDE_final_selling_price_l1768_176839


namespace NUMINAMATH_CALUDE_derivative_f_at_negative_one_l1768_176893

def f (x : ℝ) : ℝ := x^6

theorem derivative_f_at_negative_one :
  deriv f (-1) = -6 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_negative_one_l1768_176893


namespace NUMINAMATH_CALUDE_probability_same_group_is_one_fourth_l1768_176895

def number_of_groups : ℕ := 4

def probability_same_group : ℚ :=
  (number_of_groups : ℚ) / ((number_of_groups : ℚ) * (number_of_groups : ℚ))

theorem probability_same_group_is_one_fourth :
  probability_same_group = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_group_is_one_fourth_l1768_176895


namespace NUMINAMATH_CALUDE_square_difference_65_35_l1768_176831

theorem square_difference_65_35 : 65^2 - 35^2 = 3000 := by sorry

end NUMINAMATH_CALUDE_square_difference_65_35_l1768_176831


namespace NUMINAMATH_CALUDE_max_value_f_l1768_176889

theorem max_value_f (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  let f := fun (a b c : ℝ) => (1 - b*c + c) * (1 - a*c + a) * (1 - a*b + b)
  (∀ a b c, a > 0 → b > 0 → c > 0 → a * b * c = 1 → f a b c ≤ 1) ∧
  f x y z = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_f_l1768_176889


namespace NUMINAMATH_CALUDE_three_digit_palindrome_squares_l1768_176823

/-- A number is a 3-digit palindrome square if it satisfies these conditions:
1. It is between 100 and 999 (inclusive).
2. It is a perfect square.
3. It reads the same forward and backward. -/
def is_three_digit_palindrome_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  ∃ k, n = k^2 ∧
  (n / 100 = n % 10) ∧ (n / 10 % 10 = (n / 10) % 10)

/-- There are exactly 3 numbers that are 3-digit palindrome squares. -/
theorem three_digit_palindrome_squares :
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_three_digit_palindrome_square n) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_three_digit_palindrome_squares_l1768_176823


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1768_176832

theorem algebraic_expression_value (a b : ℝ) (h : a - 2*b = -1) :
  1 - 2*a + 4*b = 3 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1768_176832
