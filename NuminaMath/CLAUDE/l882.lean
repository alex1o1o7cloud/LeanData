import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l882_88258

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 4 + a 7 = 45) →
  (a 2 + a 5 + a 8 = 29) →
  (a 3 + a 6 + a 9 = 13) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l882_88258


namespace NUMINAMATH_CALUDE_bicycle_price_problem_l882_88285

theorem bicycle_price_problem (profit_a_to_b : ℝ) (profit_b_to_c : ℝ) (final_price : ℝ) :
  profit_a_to_b = 0.25 →
  profit_b_to_c = 0.5 →
  final_price = 225 →
  ∃ (cost_price_a : ℝ),
    cost_price_a * (1 + profit_a_to_b) * (1 + profit_b_to_c) = final_price ∧
    cost_price_a = 120 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_price_problem_l882_88285


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_1739_l882_88297

theorem smallest_prime_factor_of_1739 : Nat.Prime 1739 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_1739_l882_88297


namespace NUMINAMATH_CALUDE_jason_has_21_toys_l882_88250

/-- The number of toys Rachel has -/
def rachel_toys : ℕ := 1

/-- The number of toys John has -/
def john_toys : ℕ := rachel_toys + 6

/-- The number of toys Jason has -/
def jason_toys : ℕ := 3 * john_toys

/-- Theorem: Jason has 21 toys -/
theorem jason_has_21_toys : jason_toys = 21 := by
  sorry

end NUMINAMATH_CALUDE_jason_has_21_toys_l882_88250


namespace NUMINAMATH_CALUDE_final_expression_l882_88242

theorem final_expression (x : ℝ) : ((3 * x + 5) - 5 * x) / 3 = (-2 * x + 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_final_expression_l882_88242


namespace NUMINAMATH_CALUDE_zach_rental_cost_l882_88275

/-- Calculates the total cost of a car rental given the base cost, cost per mile, and miles driven. -/
def rental_cost (base_cost : ℚ) (cost_per_mile : ℚ) (miles_driven : ℚ) : ℚ :=
  base_cost + cost_per_mile * miles_driven

/-- Proves that the total cost of Zach's car rental is $832. -/
theorem zach_rental_cost :
  let base_cost : ℚ := 150
  let cost_per_mile : ℚ := 1/2
  let monday_miles : ℚ := 620
  let thursday_miles : ℚ := 744
  let total_miles : ℚ := monday_miles + thursday_miles
  rental_cost base_cost cost_per_mile total_miles = 832 := by
  sorry

end NUMINAMATH_CALUDE_zach_rental_cost_l882_88275


namespace NUMINAMATH_CALUDE_sin_plus_cos_shift_l882_88220

theorem sin_plus_cos_shift (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.cos (3 * x - π / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_shift_l882_88220


namespace NUMINAMATH_CALUDE_union_of_sets_l882_88257

theorem union_of_sets (A B : Set ℤ) (hA : A = {0, 1}) (hB : B = {-1, 1}) :
  A ∪ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l882_88257


namespace NUMINAMATH_CALUDE_doughnuts_left_l882_88222

def doughnuts_problem (total_doughnuts : ℕ) (total_staff : ℕ) 
  (staff_3 : ℕ) (staff_2 : ℕ) (doughnuts_3 : ℕ) (doughnuts_2 : ℕ) (doughnuts_4 : ℕ) : Prop :=
  total_doughnuts = 120 ∧
  total_staff = 35 ∧
  staff_3 = 15 ∧
  staff_2 = 10 ∧
  doughnuts_3 = 3 ∧
  doughnuts_2 = 2 ∧
  doughnuts_4 = 4 ∧
  total_staff = staff_3 + staff_2 + (total_staff - staff_3 - staff_2)

theorem doughnuts_left (total_doughnuts : ℕ) (total_staff : ℕ) 
  (staff_3 : ℕ) (staff_2 : ℕ) (doughnuts_3 : ℕ) (doughnuts_2 : ℕ) (doughnuts_4 : ℕ) :
  doughnuts_problem total_doughnuts total_staff staff_3 staff_2 doughnuts_3 doughnuts_2 doughnuts_4 →
  total_doughnuts - (staff_3 * doughnuts_3 + staff_2 * doughnuts_2 + (total_staff - staff_3 - staff_2) * doughnuts_4) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_doughnuts_left_l882_88222


namespace NUMINAMATH_CALUDE_purple_balls_count_l882_88216

theorem purple_balls_count (total : ℕ) (white green yellow red : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 ∧
  white = 50 ∧
  green = 20 ∧
  yellow = 10 ∧
  red = 17 ∧
  prob_not_red_purple = 4/5 →
  total - (white + green + yellow + red) = 3 := by
sorry

end NUMINAMATH_CALUDE_purple_balls_count_l882_88216


namespace NUMINAMATH_CALUDE_sphere_radius_equals_eight_l882_88284

-- Define constants for the cylinder dimensions
def cylinder_height : ℝ := 16
def cylinder_diameter : ℝ := 16

-- Define the theorem
theorem sphere_radius_equals_eight :
  ∀ r : ℝ,
  (4 * Real.pi * r^2 = 2 * Real.pi * (cylinder_diameter / 2) * cylinder_height) →
  r = 8 := by
  sorry


end NUMINAMATH_CALUDE_sphere_radius_equals_eight_l882_88284


namespace NUMINAMATH_CALUDE_min_union_cardinality_l882_88236

theorem min_union_cardinality (A B : Finset ℕ) (hA : A.card = 30) (hB : B.card = 20) :
  35 ≤ (A ∪ B).card := by sorry

end NUMINAMATH_CALUDE_min_union_cardinality_l882_88236


namespace NUMINAMATH_CALUDE_game_cost_l882_88240

theorem game_cost (initial_amount allowance final_amount : ℕ) : 
  initial_amount = 5 → 
  allowance = 26 → 
  final_amount = 29 → 
  initial_amount + allowance - final_amount = 2 := by
sorry

end NUMINAMATH_CALUDE_game_cost_l882_88240


namespace NUMINAMATH_CALUDE_quartic_inequality_l882_88295

theorem quartic_inequality (a b : ℝ) : 
  (∃ x : ℝ, x^4 - a*x^3 + 2*x^2 - b*x + 1 = 0) → a^2 + b^2 ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_quartic_inequality_l882_88295


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_00001_l882_88244

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_0_00001 :
  toScientificNotation 0.00001 = ScientificNotation.mk 1 (-5) sorry :=
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_00001_l882_88244


namespace NUMINAMATH_CALUDE_fourth_animal_is_sheep_l882_88293

def animals : List String := ["Horses", "Cows", "Pigs", "Sheep", "Rabbits", "Squirrels"]

theorem fourth_animal_is_sheep : animals[3] = "Sheep" := by
  sorry

end NUMINAMATH_CALUDE_fourth_animal_is_sheep_l882_88293


namespace NUMINAMATH_CALUDE_cubic_root_sum_reciprocal_squares_l882_88212

theorem cubic_root_sum_reciprocal_squares : 
  ∀ (α β γ : ℝ), 
    (α^3 - 15*α^2 + 26*α - 8 = 0) → 
    (β^3 - 15*β^2 + 26*β - 8 = 0) → 
    (γ^3 - 15*γ^2 + 26*γ - 8 = 0) → 
    α ≠ β → β ≠ γ → γ ≠ α →
    1/α^2 + 1/β^2 + 1/γ^2 = 916/64 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_reciprocal_squares_l882_88212


namespace NUMINAMATH_CALUDE_wilson_payment_l882_88231

def hamburger_price : ℕ := 5
def cola_price : ℕ := 2
def hamburger_quantity : ℕ := 2
def cola_quantity : ℕ := 3
def discount : ℕ := 4

def total_cost : ℕ := hamburger_price * hamburger_quantity + cola_price * cola_quantity - discount

theorem wilson_payment : total_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_wilson_payment_l882_88231


namespace NUMINAMATH_CALUDE_inverse_proportion_constant_l882_88279

/-- Given an inverse proportion function y = k/x passing through the point (-2, -3),
    prove that the value of k is equal to 6. -/
theorem inverse_proportion_constant (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x ≠ 0, f x = k / x) ∧ f (-2) = -3) → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_constant_l882_88279


namespace NUMINAMATH_CALUDE_sphere_radius_in_pyramid_l882_88280

theorem sphere_radius_in_pyramid (h : ℝ) (a b c : ℝ) (r : ℝ) : 
  h = 5 → 
  a = 7 ∧ b = 8 ∧ c = 9 → 
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    x + y + z = a + b + c ∧
    x^2 + h^2 = (a/2)^2 + (b/2)^2 + r^2 ∧
    y^2 + h^2 = (b/2)^2 + (c/2)^2 + r^2 ∧
    z^2 + h^2 = (c/2)^2 + (a/2)^2 + r^2) →
  r = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_in_pyramid_l882_88280


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l882_88204

theorem cubic_equation_solutions :
  let z₁ : ℂ := -3
  let z₂ : ℂ := (3 + 3 * Complex.I * Real.sqrt 3) / 2
  let z₃ : ℂ := (3 - 3 * Complex.I * Real.sqrt 3) / 2
  (z₁^3 = -27 ∧ z₂^3 = -27 ∧ z₃^3 = -27) ∧
  ∀ z : ℂ, z^3 = -27 → (z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l882_88204


namespace NUMINAMATH_CALUDE_prime_property_l882_88217

theorem prime_property (p : ℕ) : 
  Prime p → (∃ q : ℕ, Prime q ∧ q = 2^(p+1) + p^3 - p^2 - p) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_prime_property_l882_88217


namespace NUMINAMATH_CALUDE_removed_triangles_area_l882_88201

theorem removed_triangles_area (r s : ℝ) : 
  (r + s)^2 + (r - s)^2 = 16^2 → 
  2 * (r^2 + s^2) = 256 := by
  sorry

end NUMINAMATH_CALUDE_removed_triangles_area_l882_88201


namespace NUMINAMATH_CALUDE_abc_inequality_l882_88291

theorem abc_inequality (a b c : ℝ) (sum_eq_one : a + b + c = 1) (prod_pos : a * b * c > 0) :
  a * b + b * c + c * a < Real.sqrt (a * b * c) / 2 + 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l882_88291


namespace NUMINAMATH_CALUDE_donut_problem_l882_88278

theorem donut_problem (initial_donuts : ℕ) (h1 : initial_donuts = 50) : 
  let after_bill_eats := initial_donuts - 2
  let after_secretary_takes := after_bill_eats - 4
  let stolen_by_coworkers := after_secretary_takes / 2
  initial_donuts - 2 - 4 - stolen_by_coworkers = 22 := by
  sorry

end NUMINAMATH_CALUDE_donut_problem_l882_88278


namespace NUMINAMATH_CALUDE_triangle_area_l882_88255

/-- Given a triangle with perimeter 20 and inradius 3, prove its area is 30 -/
theorem triangle_area (T : Set ℝ) (perimeter inradius : ℝ) : 
  perimeter = 20 →
  inradius = 3 →
  (∃ (area : ℝ), area = inradius * (perimeter / 2) ∧ area = 30) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l882_88255


namespace NUMINAMATH_CALUDE_sum_of_segments_9_9_l882_88292

/-- The sum of lengths of all line segments formed by dividing a line segment into equal parts -/
def sum_of_segments (total_length : ℕ) (num_divisions : ℕ) : ℕ :=
  let unit_length := total_length / num_divisions
  let sum_short_segments := (num_divisions - 1) * num_divisions * unit_length
  let sum_long_segments := (num_divisions * (num_divisions + 1) * unit_length) / 2
  sum_short_segments + sum_long_segments

/-- Theorem: The sum of lengths of all line segments formed by dividing a line segment of length 9 into 9 equal parts is equal to 165 -/
theorem sum_of_segments_9_9 :
  sum_of_segments 9 9 = 165 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_segments_9_9_l882_88292


namespace NUMINAMATH_CALUDE_watch_cost_price_l882_88202

theorem watch_cost_price (CP : ℝ) : 
  (1.04 * CP - 0.90 * CP = 280) → CP = 2000 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l882_88202


namespace NUMINAMATH_CALUDE_triangle_angle_not_all_greater_than_60_l882_88272

theorem triangle_angle_not_all_greater_than_60 :
  ∀ (a b c : ℝ),
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Angles are positive
  (a + b + c = 180) →        -- Sum of angles in a triangle is 180 degrees
  ¬(a > 60 ∧ b > 60 ∧ c > 60) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_not_all_greater_than_60_l882_88272


namespace NUMINAMATH_CALUDE_inequalities_satisfied_l882_88203

theorem inequalities_satisfied (a b c x y z : ℝ) 
  (h1 : x ≤ a) (h2 : y ≤ b) (h3 : z ≤ c) : 
  (x * y + y * z + z * x ≤ a * b + b * c + c * a) ∧ 
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧ 
  (x * y * z ≤ a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_satisfied_l882_88203


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l882_88225

/-- Given vectors OA, OB, OC in ℝ², prove that if A, B, C are collinear, then the x-coordinate of OA is 6. -/
theorem collinear_points_k_value (OA OB OC : ℝ × ℝ) :
  OA.1 = k ∧ OA.2 = 11 ∧
  OB = (4, 5) ∧
  OC = (5, 8) ∧
  ∃ (t : ℝ), (OC.1 - OB.1, OC.2 - OB.2) = t • (OB.1 - OA.1, OB.2 - OA.2) →
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l882_88225


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l882_88218

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l882_88218


namespace NUMINAMATH_CALUDE_xiao_ming_calculation_l882_88227

theorem xiao_ming_calculation (a : ℚ) : 
  (37 + 31 * a = 37 + 31 + a) → (a = 31 / 30) := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_calculation_l882_88227


namespace NUMINAMATH_CALUDE_alcohol_quantity_in_mixture_l882_88205

theorem alcohol_quantity_in_mixture (initial_alcohol : ℝ) (initial_water : ℝ) :
  initial_alcohol / initial_water = 4 / 3 →
  initial_alcohol / (initial_water + 8) = 4 / 5 →
  initial_alcohol = 16 := by
sorry

end NUMINAMATH_CALUDE_alcohol_quantity_in_mixture_l882_88205


namespace NUMINAMATH_CALUDE_birthday_gift_cost_l882_88247

def boss_contribution : ℕ := 15
def todd_contribution : ℕ := 2 * boss_contribution
def remaining_employees : ℕ := 5
def employee_contribution : ℕ := 11

theorem birthday_gift_cost :
  boss_contribution + todd_contribution + (remaining_employees * employee_contribution) = 100 := by
  sorry

end NUMINAMATH_CALUDE_birthday_gift_cost_l882_88247


namespace NUMINAMATH_CALUDE_bill_increase_proof_l882_88235

/-- Calculates the total monthly bill after a percentage increase -/
def total_bill_after_increase (original_bill : ℚ) (percent_increase : ℚ) : ℚ :=
  original_bill * (1 + percent_increase / 100)

/-- Proves that given an original monthly bill of $60 and a 30% increase, 
    the total monthly bill after the increase is $78 -/
theorem bill_increase_proof :
  total_bill_after_increase 60 30 = 78 := by
  sorry

end NUMINAMATH_CALUDE_bill_increase_proof_l882_88235


namespace NUMINAMATH_CALUDE_intersection_condition_l882_88271

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt (2 * p.1 - p.1^2)}
def N (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * (p.1 + 1)}

-- State the theorem
theorem intersection_condition (k : ℝ) :
  (∃ p, p ∈ M ∩ N k) ↔ 0 ≤ k ∧ k ≤ Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l882_88271


namespace NUMINAMATH_CALUDE_intersection_condition_l882_88265

/-- The set M in ℝ² -/
def M : Set (ℝ × ℝ) := {p | p.2 ≥ p.1^2}

/-- The set N in ℝ² parameterized by a -/
def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - a)^2 ≤ 1}

/-- The necessary and sufficient condition for M ∩ N = N -/
theorem intersection_condition (a : ℝ) : M ∩ N a = N a ↔ a ≥ 5/4 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l882_88265


namespace NUMINAMATH_CALUDE_correct_recommendation_plans_l882_88233

def total_students : ℕ := 7
def students_to_recommend : ℕ := 4
def sports_talented : ℕ := 2
def artistic_talented : ℕ := 2
def other_talented : ℕ := 3

def recommendation_plans : ℕ := sorry

theorem correct_recommendation_plans : recommendation_plans = 25 := by sorry

end NUMINAMATH_CALUDE_correct_recommendation_plans_l882_88233


namespace NUMINAMATH_CALUDE_ab_range_l882_88200

def f (x : ℝ) : ℝ := |2 - x^2|

theorem ab_range (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  ∃ (l u : ℝ), l = 0 ∧ u = 2 ∧ ∀ x, a * b = x → l < x ∧ x < u :=
sorry

end NUMINAMATH_CALUDE_ab_range_l882_88200


namespace NUMINAMATH_CALUDE_total_sheets_used_l882_88234

theorem total_sheets_used (total_classes : ℕ) (first_class_count : ℕ) (last_class_count : ℕ)
  (first_class_students : ℕ) (last_class_students : ℕ)
  (first_class_sheets_per_student : ℕ) (last_class_sheets_per_student : ℕ) :
  total_classes = first_class_count + last_class_count →
  first_class_count = 3 →
  last_class_count = 3 →
  first_class_students = 22 →
  last_class_students = 18 →
  first_class_sheets_per_student = 6 →
  last_class_sheets_per_student = 4 →
  (first_class_count * first_class_students * first_class_sheets_per_student) +
  (last_class_count * last_class_students * last_class_sheets_per_student) = 612 :=
by sorry

end NUMINAMATH_CALUDE_total_sheets_used_l882_88234


namespace NUMINAMATH_CALUDE_plane_line_propositions_l882_88229

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def parallel_to_plane (l : Line) (p : Plane) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def perpendicular_to_plane (l : Line) (p : Plane) : Prop :=
  sorry

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

/-- Two lines are skew -/
def skew (l1 l2 : Line) : Prop :=
  sorry

/-- A line intersects a plane -/
def intersects_plane (l : Line) (p : Plane) : Prop :=
  sorry

/-- The angles formed by two lines with a plane are equal -/
def equal_angles_with_plane (l1 l2 : Line) (p : Plane) : Prop :=
  sorry

theorem plane_line_propositions (α : Plane) (m n : Line) :
  (∃! prop : Prop, prop = true ∧
    (prop = (parallel m n → equal_angles_with_plane m n α) ∨
     prop = (parallel_to_plane m α → parallel_to_plane n α → parallel m n) ∨
     prop = (perpendicular_to_plane m α → perpendicular m n → parallel_to_plane n α) ∨
     prop = (skew m n → parallel_to_plane m α → intersects_plane n α))) :=
  sorry

end NUMINAMATH_CALUDE_plane_line_propositions_l882_88229


namespace NUMINAMATH_CALUDE_geometric_and_arithmetic_sequences_l882_88248

/-- A geometric sequence with a₁ = 2 and a₄ = 16 -/
def geometric_sequence (n : ℕ) : ℝ :=
  2 * (2 : ℝ) ^ (n - 1)

/-- An arithmetic sequence with b₃ = a₃ and b₅ = a₅ -/
def arithmetic_sequence (n : ℕ) : ℝ :=
  -16 + 12 * (n - 1)

theorem geometric_and_arithmetic_sequences :
  (∀ n : ℕ, geometric_sequence n = 2^n) ∧
  (arithmetic_sequence 3 = geometric_sequence 3 ∧
   arithmetic_sequence 5 = geometric_sequence 5) ∧
  (∀ n : ℕ, arithmetic_sequence n = 12*n - 28) := by
  sorry

end NUMINAMATH_CALUDE_geometric_and_arithmetic_sequences_l882_88248


namespace NUMINAMATH_CALUDE_perfect_square_consecutive_integers_l882_88214

theorem perfect_square_consecutive_integers (n : ℤ) : 
  (∃ k : ℤ, n * (n + 1) = k^2) ↔ (n = 0 ∨ n = -1) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_consecutive_integers_l882_88214


namespace NUMINAMATH_CALUDE_reset_counters_in_eleven_moves_l882_88289

/-- Represents a counter with a value between 1 and 2017 -/
def Counter := { n : ℕ // 1 ≤ n ∧ n ≤ 2017 }

/-- Represents a configuration of 28 counters -/
def Configuration := Fin 28 → Counter

/-- Represents a move that decreases some counters by a certain value -/
def Move := Configuration → ℕ → Configuration

/-- Predicate to check if a configuration has all counters reset to zero -/
def AllZero (config : Configuration) : Prop :=
  ∀ i, (config i).val = 0

/-- The main theorem stating that any configuration can be reset in at most 11 moves -/
theorem reset_counters_in_eleven_moves :
  ∀ (initial_config : Configuration),
  ∃ (moves : Fin 11 → Move),
  AllZero ((moves 10) ((moves 9) ((moves 8) ((moves 7) ((moves 6) ((moves 5) ((moves 4) ((moves 3) ((moves 2) ((moves 1) ((moves 0) initial_config 0) 0) 0) 0) 0) 0) 0) 0) 0) 0) 0) :=
sorry

end NUMINAMATH_CALUDE_reset_counters_in_eleven_moves_l882_88289


namespace NUMINAMATH_CALUDE_triangle_dot_product_l882_88239

-- Define the triangle ABC
theorem triangle_dot_product (A B C : ℝ × ℝ) :
  -- Given conditions
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let S := abs ((A.1 - C.1) * (B.2 - C.2) - (A.2 - C.2) * (B.1 - C.1)) / 2
  -- Hypothesis
  AC = 8 →
  BC = 5 →
  S = 10 * Real.sqrt 3 →
  -- Conclusion
  ((B.1 - C.1) * (C.1 - A.1) + (B.2 - C.2) * (C.2 - A.2) = 20 ∨
   (B.1 - C.1) * (C.1 - A.1) + (B.2 - C.2) * (C.2 - A.2) = -20) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_dot_product_l882_88239


namespace NUMINAMATH_CALUDE_limit_at_one_l882_88226

noncomputable def f (x : ℝ) : ℝ := (5/3) * x - Real.log (2*x + 1)

theorem limit_at_one (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ Δx : ℝ, 0 < |Δx| ∧ |Δx| < δ →
    |(f (1 + Δx) - f 1) / Δx - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_at_one_l882_88226


namespace NUMINAMATH_CALUDE_two_plus_three_equals_eight_is_proposition_l882_88290

/-- A statement is a proposition if it can be judged as either true or false. -/
def is_proposition (statement : Prop) : Prop :=
  (statement ∨ ¬statement) ∧ ¬(statement ∧ ¬statement)

/-- The statement "2 + 3 = 8" is a proposition. -/
theorem two_plus_three_equals_eight_is_proposition :
  is_proposition (2 + 3 = 8) := by
  sorry

end NUMINAMATH_CALUDE_two_plus_three_equals_eight_is_proposition_l882_88290


namespace NUMINAMATH_CALUDE_harriett_found_three_dollars_l882_88219

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The number of quarters Harriett found -/
def quarters_found : ℕ := 10

/-- The number of dimes Harriett found -/
def dimes_found : ℕ := 3

/-- The number of nickels Harriett found -/
def nickels_found : ℕ := 3

/-- The number of pennies Harriett found -/
def pennies_found : ℕ := 5

/-- The total value of the coins Harriett found -/
def total_value : ℚ := 
  quarters_found * quarter_value + 
  dimes_found * dime_value + 
  nickels_found * nickel_value + 
  pennies_found * penny_value

theorem harriett_found_three_dollars : total_value = 3 := by
  sorry

end NUMINAMATH_CALUDE_harriett_found_three_dollars_l882_88219


namespace NUMINAMATH_CALUDE_condo_units_l882_88206

/-- Calculates the total number of units in a condo building -/
def total_units (total_floors : ℕ) (regular_units : ℕ) (penthouse_units : ℕ) (penthouse_floors : ℕ) : ℕ :=
  (total_floors - penthouse_floors) * regular_units + penthouse_floors * penthouse_units

/-- Theorem stating that a condo with the given specifications has 256 units -/
theorem condo_units : total_units 23 12 2 2 = 256 := by
  sorry

end NUMINAMATH_CALUDE_condo_units_l882_88206


namespace NUMINAMATH_CALUDE_total_peaches_l882_88211

theorem total_peaches (red yellow green : ℕ) 
  (h1 : red = 7) 
  (h2 : yellow = 15) 
  (h3 : green = 8) : 
  red + yellow + green = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_l882_88211


namespace NUMINAMATH_CALUDE_sequence_sum_l882_88208

/-- Given a sequence {a_n} where a_1 = 1 and S_n = n^2 * a_n for all positive integers n,
    prove that the sum of the first n terms (S_n) is equal to 2n / (n + 1). -/
theorem sequence_sum (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) (h1 : a 1 = 1)
    (h2 : ∀ n : ℕ+, S n = n^2 * a n) :
    ∀ n : ℕ+, S n = 2 * n / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l882_88208


namespace NUMINAMATH_CALUDE_no_hexagon_for_19_and_20_l882_88274

theorem no_hexagon_for_19_and_20 : 
  (¬ ∃ (ℓ : ℤ), 19 = 2 * ℓ^2 + ℓ) ∧ (¬ ∃ (ℓ : ℤ), 20 = 2 * ℓ^2 + ℓ) := by
  sorry

end NUMINAMATH_CALUDE_no_hexagon_for_19_and_20_l882_88274


namespace NUMINAMATH_CALUDE_f_properties_l882_88266

noncomputable section

def f (x : ℝ) := Real.log x - (x - 1)^2 / 2

theorem f_properties :
  let φ := (1 + Real.sqrt 5) / 2
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < φ → f x₁ < f x₂) ∧
  (∀ x, x > 1 → f x < x - 1) ∧
  (∀ k, k < 1 → ∃ x₀ > 1, ∀ x, 1 < x ∧ x < x₀ → f x > k * (x - 1)) ∧
  (∀ k, k ≥ 1 → ¬∃ x₀ > 1, ∀ x, 1 < x ∧ x < x₀ → f x > k * (x - 1)) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l882_88266


namespace NUMINAMATH_CALUDE_equation_equivalence_l882_88282

theorem equation_equivalence (a b : ℝ) (h : a + 2 * b + 2 = Real.sqrt 2) : 
  4 * a + 8 * b + 5 = 4 * Real.sqrt 2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l882_88282


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l882_88263

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (x^2 - 9) / (x + 3) = 0 → x = 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l882_88263


namespace NUMINAMATH_CALUDE_dino_money_theorem_l882_88287

/-- Calculates the money Dino has left at the end of the month based on his work hours, rates, and expenses. -/
def dino_money_left (hours1 hours2 hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (expenses : ℕ) : ℕ :=
  hours1 * rate1 + hours2 * rate2 + hours3 * rate3 - expenses

/-- Theorem stating that Dino has $500 left at the end of the month. -/
theorem dino_money_theorem : dino_money_left 20 30 5 10 20 40 500 = 500 := by
  sorry

end NUMINAMATH_CALUDE_dino_money_theorem_l882_88287


namespace NUMINAMATH_CALUDE_second_number_calculation_l882_88264

theorem second_number_calculation (A B : ℝ) (h1 : A = 680) (h2 : 0.2 * A = 0.4 * B + 80) : B = 140 := by
  sorry

end NUMINAMATH_CALUDE_second_number_calculation_l882_88264


namespace NUMINAMATH_CALUDE_acute_triangle_sine_sum_l882_88277

theorem acute_triangle_sine_sum (α β γ : Real) 
  (acute_triangle : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = Real.pi)
  (acute_angles : α < Real.pi/2 ∧ β < Real.pi/2 ∧ γ < Real.pi/2) : 
  Real.sin α + Real.sin β + Real.sin γ > 2 := by
sorry

end NUMINAMATH_CALUDE_acute_triangle_sine_sum_l882_88277


namespace NUMINAMATH_CALUDE_total_goals_after_five_matches_l882_88213

/-- A football player's goal scoring record -/
structure FootballPlayer where
  goals_before_fifth : ℕ  -- Total goals before the fifth match
  matches_before_fifth : ℕ -- Number of matches before the fifth match (should be 4)

/-- The problem statement -/
theorem total_goals_after_five_matches (player : FootballPlayer) 
  (h1 : player.matches_before_fifth = 4)
  (h2 : (player.goals_before_fifth : ℚ) / 4 + 0.2 = 
        ((player.goals_before_fifth + 4) : ℚ) / 5) : 
  player.goals_before_fifth + 4 = 16 := by
  sorry

#check total_goals_after_five_matches

end NUMINAMATH_CALUDE_total_goals_after_five_matches_l882_88213


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l882_88210

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l882_88210


namespace NUMINAMATH_CALUDE_macaron_ratio_l882_88283

theorem macaron_ratio (mitch joshua miles renz : ℕ) : 
  mitch = 20 →
  joshua = mitch + 6 →
  (∃ k : ℚ, joshua = k * miles) →
  renz = (3 * miles) / 4 - 1 →
  mitch + joshua + miles + renz = 68 * 2 →
  joshua * 2 = miles * 1 := by
  sorry

end NUMINAMATH_CALUDE_macaron_ratio_l882_88283


namespace NUMINAMATH_CALUDE_uniform_transformation_l882_88209

theorem uniform_transformation (a₁ : ℝ) : 
  a₁ ∈ Set.Icc 0 1 → (8 * a₁ - 2) ∈ Set.Icc (-2) 6 := by
  sorry

end NUMINAMATH_CALUDE_uniform_transformation_l882_88209


namespace NUMINAMATH_CALUDE_greatest_power_of_ten_dividing_twenty_factorial_l882_88262

theorem greatest_power_of_ten_dividing_twenty_factorial : 
  (∃ m : ℕ, (20 : ℕ).factorial % (10 ^ m) = 0 ∧ 
    ∀ k : ℕ, k > m → (20 : ℕ).factorial % (10 ^ k) ≠ 0) → 
  (∃ m : ℕ, m = 4 ∧ (20 : ℕ).factorial % (10 ^ m) = 0 ∧ 
    ∀ k : ℕ, k > m → (20 : ℕ).factorial % (10 ^ k) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_ten_dividing_twenty_factorial_l882_88262


namespace NUMINAMATH_CALUDE_two_buckets_water_amount_l882_88223

/-- A container for water -/
structure Container where
  capacity : ℕ

/-- A jug is a container with a capacity of 5 liters -/
def Jug : Container :=
  { capacity := 5 }

/-- A bucket is a container that can hold 4 jugs worth of water -/
def Bucket : Container :=
  { capacity := 4 * Jug.capacity }

/-- The amount of water in multiple containers -/
def water_amount (n : ℕ) (c : Container) : ℕ :=
  n * c.capacity

theorem two_buckets_water_amount :
  water_amount 2 Bucket = 40 :=
by sorry

end NUMINAMATH_CALUDE_two_buckets_water_amount_l882_88223


namespace NUMINAMATH_CALUDE_difference_of_squares_l882_88281

theorem difference_of_squares (x y : ℝ) 
  (sum_eq : x + y = 20) 
  (diff_eq : x - y = 4) : 
  x^2 - y^2 = 80 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_l882_88281


namespace NUMINAMATH_CALUDE_milkshake_leftover_l882_88276

/-- Calculates the amount of milk left over after making milkshakes -/
theorem milkshake_leftover (milk_per_shake ice_cream_per_shake total_milk total_ice_cream : ℕ) :
  milk_per_shake = 4 →
  ice_cream_per_shake = 12 →
  total_milk = 72 →
  total_ice_cream = 192 →
  total_milk - (total_ice_cream / ice_cream_per_shake * milk_per_shake) = 8 := by
  sorry

#check milkshake_leftover

end NUMINAMATH_CALUDE_milkshake_leftover_l882_88276


namespace NUMINAMATH_CALUDE_marked_circles_alignment_l882_88252

/-- Two identical circles, each marked with k arcs -/
structure MarkedCircle where
  k : ℕ
  arcs : Fin k → ℝ
  arc_measure : ∀ i, arcs i < 180 / (k^2 - k + 1)
  alignment : ∃ r : ℝ, ∀ i, ∃ j, arcs i = (fun x => (x + r) % 360) (arcs j)

/-- The theorem statement -/
theorem marked_circles_alignment (c1 c2 : MarkedCircle) (h : c1 = c2) :
  ∃ r : ℝ, ∀ i, ∀ j, c1.arcs i ≠ (fun x => (x + r) % 360) (c2.arcs j) := by
  sorry

end NUMINAMATH_CALUDE_marked_circles_alignment_l882_88252


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_nine_l882_88249

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_nine :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 9 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 9 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_nine_l882_88249


namespace NUMINAMATH_CALUDE_magnitude_of_BC_l882_88251

/-- Given vectors BA and AC in R², prove that the magnitude of BC is 5 -/
theorem magnitude_of_BC (BA AC : ℝ × ℝ) : 
  BA = (3, -2) → AC = (0, 6) → ‖BA + AC‖ = 5 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_BC_l882_88251


namespace NUMINAMATH_CALUDE_min_value_theorem_l882_88207

noncomputable def f (x : ℝ) : ℝ := min (3^x - 1) (-x^2 + 2*x + 1)

theorem min_value_theorem (m a b : ℝ) :
  (∀ x, f x ≤ m) ∧  -- m is the maximum value of f
  (∃ x, f x = m) ∧  -- m is attained for some x
  (a > 0) ∧ (b > 0) ∧ (a + 2*b = m) →  -- conditions on a and b
  (∀ a' b', a' > 0 → b' > 0 → a' + 2*b' = m → 
    2 / (a' + 1) + 1 / b' ≥ 8/3) ∧  -- 8/3 is the minimum value
  (∃ a' b', a' > 0 ∧ b' > 0 ∧ a' + 2*b' = m ∧ 
    2 / (a' + 1) + 1 / b' = 8/3)  -- minimum is attained for some a' and b'
  := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l882_88207


namespace NUMINAMATH_CALUDE_pet_store_birds_pet_store_birds_after_changes_l882_88294

/-- The number of birds in a pet store after sales and additions --/
theorem pet_store_birds (num_cages : ℕ) (initial_parrots : ℕ) (initial_parakeets : ℕ) (initial_canaries : ℕ)
  (sold_parrots : ℕ) (sold_canaries : ℕ) (added_parakeets : ℕ) : ℕ :=
  let total_initial_parrots := num_cages * initial_parrots
  let total_initial_parakeets := num_cages * initial_parakeets
  let total_initial_canaries := num_cages * initial_canaries
  let final_parrots := total_initial_parrots - sold_parrots
  let final_parakeets := total_initial_parakeets + added_parakeets
  let final_canaries := total_initial_canaries - sold_canaries
  final_parrots + final_parakeets + final_canaries

/-- The number of birds in the pet store after changes is 235 --/
theorem pet_store_birds_after_changes : pet_store_birds 15 3 8 5 5 2 2 = 235 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_pet_store_birds_after_changes_l882_88294


namespace NUMINAMATH_CALUDE_negative_difference_l882_88273

theorem negative_difference (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end NUMINAMATH_CALUDE_negative_difference_l882_88273


namespace NUMINAMATH_CALUDE_crease_lines_form_annulus_l882_88238

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the folding operation
def Fold (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
    p = (center.1 + t * (point.1 - center.1), center.2 + t * (point.2 - center.2))}

-- Define the set of all crease lines
def CreaseLines (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  ⋃ (point ∈ Circle center radius), Fold center radius point

-- Define the annulus
def Annulus (center : ℝ × ℝ) (innerRadius outerRadius : ℝ) : Set (ℝ × ℝ) :=
  {p | innerRadius^2 ≤ (p.1 - center.1)^2 + (p.2 - center.2)^2 ∧ 
       (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ outerRadius^2}

-- The theorem to prove
theorem crease_lines_form_annulus (center : ℝ × ℝ) :
  CreaseLines center 10 = Annulus center 5 10 := by sorry

end NUMINAMATH_CALUDE_crease_lines_form_annulus_l882_88238


namespace NUMINAMATH_CALUDE_regular_hexagon_vector_relation_l882_88298

-- Define a regular hexagon
structure RegularHexagon (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D E F : V)
  (is_regular : sorry)  -- This would typically include conditions that define a regular hexagon

-- Theorem statement
theorem regular_hexagon_vector_relation 
  {V : Type*} [AddCommGroup V] [Module ℝ V] 
  (hex : RegularHexagon V) 
  (a b : V) 
  (h1 : hex.B - hex.A = a) 
  (h2 : hex.E - hex.A = b) : 
  hex.C - hex.B = (1/2 : ℝ) • a + (1/2 : ℝ) • b := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_vector_relation_l882_88298


namespace NUMINAMATH_CALUDE_reverse_digits_sum_l882_88256

/-- Two-digit integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem reverse_digits_sum (x y : ℕ) (hx : TwoDigitInt x) (hy : TwoDigitInt y)
  (h_reverse : y = 10 * (x % 10) + (x / 10))
  (a b : ℕ) (hx_digits : x = 10 * a + b)
  (hab : a - b = 3)
  (m : ℕ) (hm : x^2 - y^2 = m^2) :
  x + y + m = 178 := by
sorry

end NUMINAMATH_CALUDE_reverse_digits_sum_l882_88256


namespace NUMINAMATH_CALUDE_distance_to_park_is_correct_l882_88228

/-- The distance from point A to the forest amusement park -/
def distance_to_park : ℕ := 2370

/-- The rabbit's starting time in minutes after midnight -/
def rabbit_start : ℕ := 9 * 60

/-- The turtle's starting time in minutes after midnight -/
def turtle_start : ℕ := 6 * 60 + 40

/-- The rabbit's speed in meters per minute -/
def rabbit_speed : ℕ := 40

/-- The turtle's speed in meters per minute -/
def turtle_speed : ℕ := 10

/-- The rabbit's jumping time in minutes -/
def rabbit_jump_time : ℕ := 3

/-- The rabbit's resting time in minutes -/
def rabbit_rest_time : ℕ := 2

/-- The time difference between rabbit and turtle arrival in seconds -/
def arrival_time_diff : ℕ := 15

theorem distance_to_park_is_correct : 
  ∀ (t : ℕ), 
  t * turtle_speed = distance_to_park ∧ 
  t = (rabbit_start - turtle_start) + 
      (distance_to_park - (rabbit_start - turtle_start) * turtle_speed) / 
      (rabbit_speed * rabbit_jump_time / (rabbit_jump_time + rabbit_rest_time) - turtle_speed) + 
      arrival_time_diff / 60 :=
sorry

end NUMINAMATH_CALUDE_distance_to_park_is_correct_l882_88228


namespace NUMINAMATH_CALUDE_rational_equation_solution_l882_88268

theorem rational_equation_solution :
  let x : ℚ := -26/9
  (2*x + 18) / (x - 6) = (2*x - 4) / (x + 10) := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l882_88268


namespace NUMINAMATH_CALUDE_log_equation_solution_l882_88299

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  (Real.log x^3 / Real.log 3) + (Real.log x / Real.log (1/3)) = 8 → x = 81 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l882_88299


namespace NUMINAMATH_CALUDE_ship_departure_theorem_l882_88243

/-- Represents the total transit time for a cargo shipment -/
def total_transit_time (navigation_time customs_time delivery_time : ℕ) : ℕ :=
  navigation_time + customs_time + delivery_time

/-- Calculates the departure date given the expected arrival and total transit time -/
def departure_date (days_until_arrival total_transit : ℕ) : ℕ :=
  days_until_arrival + total_transit

/-- Theorem: Given the specified conditions, the ship should have departed 34 days ago -/
theorem ship_departure_theorem (navigation_time customs_time delivery_time days_until_arrival : ℕ)
  (h1 : navigation_time = 21)
  (h2 : customs_time = 4)
  (h3 : delivery_time = 7)
  (h4 : days_until_arrival = 2) :
  departure_date days_until_arrival (total_transit_time navigation_time customs_time delivery_time) = 34 := by
  sorry

#check ship_departure_theorem

end NUMINAMATH_CALUDE_ship_departure_theorem_l882_88243


namespace NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l882_88296

/-- The area of a square sheet of wrapping paper for a box with base side length s -/
def wrapping_paper_area (s : ℝ) : ℝ := 4 * s^2

/-- Theorem: The area of the square sheet of wrapping paper is 4s² -/
theorem wrapping_paper_area_theorem (s : ℝ) (h : s > 0) :
  wrapping_paper_area s = 4 * s^2 := by
  sorry

#check wrapping_paper_area_theorem

end NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l882_88296


namespace NUMINAMATH_CALUDE_equal_reading_time_l882_88215

/-- Represents the reading scenario in Mrs. Reed's English class -/
structure ReadingScenario where
  total_pages : ℕ
  mia_speed : ℕ  -- seconds per page
  leo_speed : ℕ  -- seconds per page
  mia_pages : ℕ

/-- The specific reading scenario from the problem -/
def problem_scenario : ReadingScenario :=
  { total_pages := 840
  , mia_speed := 60
  , leo_speed := 40
  , mia_pages := 336 }

/-- Calculates the total reading time for a given number of pages and reading speed -/
def reading_time (pages : ℕ) (speed : ℕ) : ℕ := pages * speed

/-- Theorem stating that Mia and Leo spend equal time reading in the given scenario -/
theorem equal_reading_time (s : ReadingScenario) (h : s = problem_scenario) :
  reading_time s.mia_pages s.mia_speed = reading_time (s.total_pages - s.mia_pages) s.leo_speed := by
  sorry

#check equal_reading_time

end NUMINAMATH_CALUDE_equal_reading_time_l882_88215


namespace NUMINAMATH_CALUDE_abc_base16_to_base4_l882_88267

/-- Converts a base 16 digit to its decimal representation -/
def hexToDecimal (x : Char) : ℕ :=
  match x with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | _ => 0  -- This case should not occur for our specific problem

/-- Converts a decimal number to its base 4 representation -/
def decimalToBase4 (x : ℕ) : List ℕ :=
  [x / 4, x % 4]

/-- Converts a base 16 number to base 4 -/
def hexToBase4 (x : String) : List ℕ :=
  x.data.map hexToDecimal |>.bind decimalToBase4

theorem abc_base16_to_base4 :
  hexToBase4 "ABC" = [2, 2, 2, 3, 3, 0] := by sorry

end NUMINAMATH_CALUDE_abc_base16_to_base4_l882_88267


namespace NUMINAMATH_CALUDE_bicycle_price_reduction_l882_88241

theorem bicycle_price_reduction (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  original_price = 200 → 
  discount1 = 0.40 → 
  discount2 = 0.25 → 
  original_price * (1 - discount1) * (1 - discount2) = 90 := by
sorry

end NUMINAMATH_CALUDE_bicycle_price_reduction_l882_88241


namespace NUMINAMATH_CALUDE_max_t_value_l882_88232

theorem max_t_value (t : ℝ) (h : t > 0) :
  (∀ u v : ℝ, (u + 5 - 2*v)^2 + (u - v^2)^2 ≥ t^2) →
  t ≤ 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_t_value_l882_88232


namespace NUMINAMATH_CALUDE_cube_of_negative_four_equals_negative_cube_of_four_l882_88286

theorem cube_of_negative_four_equals_negative_cube_of_four : (-4)^3 = -4^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_four_equals_negative_cube_of_four_l882_88286


namespace NUMINAMATH_CALUDE_lowest_digit_change_l882_88269

/-- The correct sum of the addition -/
def correct_sum : ℕ := 1179

/-- The first addend in the incorrect addition -/
def addend1 : ℕ := 374

/-- The second addend in the incorrect addition -/
def addend2 : ℕ := 519

/-- The third addend in the incorrect addition -/
def addend3 : ℕ := 286

/-- The incorrect sum displayed in the problem -/
def incorrect_sum : ℕ := 1229

/-- Function to check if a digit change makes the addition correct -/
def is_correct_change (digit : ℕ) (position : ℕ) : Prop :=
  ∃ (new_addend : ℕ),
    (position = 1 ∧ new_addend + addend2 + addend3 = correct_sum) ∨
    (position = 2 ∧ addend1 + new_addend + addend3 = correct_sum) ∨
    (position = 3 ∧ addend1 + addend2 + new_addend = correct_sum)

/-- The lowest digit that can be changed to make the addition correct -/
def lowest_changeable_digit : ℕ := 4

theorem lowest_digit_change :
  (∀ d : ℕ, d < lowest_changeable_digit → ¬∃ p : ℕ, is_correct_change d p) ∧
  (∃ p : ℕ, is_correct_change lowest_changeable_digit p) :=
sorry

end NUMINAMATH_CALUDE_lowest_digit_change_l882_88269


namespace NUMINAMATH_CALUDE_excluded_angle_measure_l882_88245

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180° -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: In a polygon where the sum of all interior angles except one is 1680°,
    the measure of the excluded interior angle is 120°. -/
theorem excluded_angle_measure (n : ℕ) (h : sum_interior_angles n - 120 = 1680) :
  120 = sum_interior_angles n - 1680 := by
  sorry

end NUMINAMATH_CALUDE_excluded_angle_measure_l882_88245


namespace NUMINAMATH_CALUDE_restaurant_bill_l882_88237

theorem restaurant_bill (n : ℕ) (extra : ℝ) (discount : ℝ) (original_bill : ℝ) :
  n = 10 →
  extra = 3 →
  discount = 10 →
  (n - 1) * ((original_bill - discount) / n + extra) = original_bill - discount →
  original_bill = 180 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_l882_88237


namespace NUMINAMATH_CALUDE_max_area_inscribed_triangle_ellipse_l882_88288

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    prove that the maximum area of a triangle inscribed in a quarter of the ellipse,
    with one vertex at the endpoint of the minor axis, another at the focus,
    and the third moving along the ellipse, is equal to b/2. -/
theorem max_area_inscribed_triangle_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≥ b) :
  let c := Real.sqrt (a^2 - b^2)
  let max_area := (fun (x y : ℝ) => (1/2) * (b * x + c * y - b * c))
  let ellipse := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}
  ∃ (p : ℝ × ℝ), p ∈ ellipse ∧
    ∀ (q : ℝ × ℝ), q ∈ ellipse →
      max_area p.1 p.2 ≥ max_area q.1 q.2 ∧
      max_area p.1 p.2 = b / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_area_inscribed_triangle_ellipse_l882_88288


namespace NUMINAMATH_CALUDE_dave_added_sixty_apps_l882_88260

/-- Calculates the number of apps Dave added to his phone -/
def apps_added (initial : ℕ) (removed : ℕ) (final : ℕ) : ℕ :=
  final - (initial - removed)

/-- Proves that Dave added 60 apps to his phone -/
theorem dave_added_sixty_apps :
  apps_added 50 10 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_dave_added_sixty_apps_l882_88260


namespace NUMINAMATH_CALUDE_dictation_mistakes_l882_88253

theorem dictation_mistakes (n : ℕ) (max_mistakes : ℕ) 
  (h1 : n = 30) 
  (h2 : max_mistakes = 12) : 
  ∃ k : ℕ, ∃ (s : Finset (Fin n)), s.card ≥ 3 ∧ 
  ∀ i ∈ s, ∃ f : Fin n → ℕ, f i = k ∧ f i ≤ max_mistakes :=
by sorry

end NUMINAMATH_CALUDE_dictation_mistakes_l882_88253


namespace NUMINAMATH_CALUDE_percentage_calculation_l882_88221

theorem percentage_calculation (x : ℝ) : 
  (0.20 * x = 80) → (0.40 * x = 160) := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l882_88221


namespace NUMINAMATH_CALUDE_survey_order_correct_l882_88246

-- Define the steps of the survey process
inductive SurveyStep
  | CollectData
  | OrganizeData
  | DrawPieChart
  | AnalyzeData

-- Define a function to represent the correct order of steps
def correctOrder : List SurveyStep :=
  [SurveyStep.CollectData, SurveyStep.OrganizeData, SurveyStep.DrawPieChart, SurveyStep.AnalyzeData]

-- Define a function to check if a given order is correct
def isCorrectOrder (order : List SurveyStep) : Prop :=
  order = correctOrder

-- Theorem stating that the given order is correct
theorem survey_order_correct :
  isCorrectOrder [SurveyStep.CollectData, SurveyStep.OrganizeData, SurveyStep.DrawPieChart, SurveyStep.AnalyzeData] :=
by sorry

end NUMINAMATH_CALUDE_survey_order_correct_l882_88246


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l882_88270

theorem exponential_equation_solution (x y : ℝ) :
  (5 : ℝ) ^ (x + y + 4) = 625 ^ x → y = 3 * x - 4 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l882_88270


namespace NUMINAMATH_CALUDE_product_absolute_value_l882_88224

theorem product_absolute_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hdistinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (heq : x + 2 / y = y + 2 / z ∧ y + 2 / z = z + 2 / x) :
  |x * y * z| = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_product_absolute_value_l882_88224


namespace NUMINAMATH_CALUDE_quadrilateral_propositions_l882_88230

-- Define a quadrilateral
structure Quadrilateral :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side4 : ℝ)

-- Define a property for quadrilaterals with four equal sides
def has_equal_sides (q : Quadrilateral) : Prop :=
  q.side1 = q.side2 ∧ q.side2 = q.side3 ∧ q.side3 = q.side4

-- Define a property for squares
def is_square (q : Quadrilateral) : Prop :=
  has_equal_sides q ∧ q.side1 = q.side2 -- This is a simplified definition

theorem quadrilateral_propositions :
  (∃ q : Quadrilateral, has_equal_sides q ∧ ¬is_square q) ∧
  (∀ q : Quadrilateral, is_square q → has_equal_sides q) ∧
  (∀ q : Quadrilateral, ¬is_square q → ¬has_equal_sides q) ∧
  (∃ q : Quadrilateral, ¬is_square q ∧ has_equal_sides q) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_propositions_l882_88230


namespace NUMINAMATH_CALUDE_star_3_7_equals_16_l882_88259

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

-- Theorem statement
theorem star_3_7_equals_16 : star 3 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_star_3_7_equals_16_l882_88259


namespace NUMINAMATH_CALUDE_solve_dancers_earnings_l882_88261

def dancers_earnings (total : ℚ) (d1 d2 d3 d4 : ℚ) : Prop :=
  d1 + d2 + d3 + d4 = total ∧
  d2 = d1 - 16 ∧
  d3 = d1 + d2 - 24 ∧
  d4 = d1 + d3

theorem solve_dancers_earnings :
  ∃ d1 d2 d3 d4 : ℚ,
    dancers_earnings 280 d1 d2 d3 d4 ∧
    d1 = 53 + 5/7 ∧
    d2 = 37 + 5/7 ∧
    d3 = 67 + 3/7 ∧
    d4 = 121 + 1/7 :=
by sorry

end NUMINAMATH_CALUDE_solve_dancers_earnings_l882_88261


namespace NUMINAMATH_CALUDE_dennis_floor_l882_88254

/-- Given the floor arrangements of Frank, Charlie, and Dennis, prove that Dennis lives on the 6th floor. -/
theorem dennis_floor : 
  ∀ (frank_floor charlie_floor dennis_floor : ℕ),
  frank_floor = 16 →
  charlie_floor = frank_floor / 4 →
  dennis_floor = charlie_floor + 2 →
  dennis_floor = 6 := by
sorry

end NUMINAMATH_CALUDE_dennis_floor_l882_88254
