import Mathlib

namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1510_151072

def equation (a b : ℕ) : Prop := 4 * a + b = 6

theorem min_reciprocal_sum :
  ∀ a b : ℕ, equation a b →
  (a ≠ 0 ∧ b ≠ 0) →
  (1 : ℚ) / a + (1 : ℚ) / b ≥ (1 : ℚ) / 1 + (1 : ℚ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1510_151072


namespace NUMINAMATH_CALUDE_subset_implies_membership_condition_l1510_151096

theorem subset_implies_membership_condition (A B : Set α) (h : A ⊆ B) :
  ∀ x, x ∈ A → x ∈ B := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_membership_condition_l1510_151096


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_trigonometric_identities_l1510_151021

-- Part 1
theorem trigonometric_expression_equality : 
  (Real.sqrt 3 * Real.sin (-20 / 3 * Real.pi)) / Real.tan (11 / 3 * Real.pi) - 
  Real.cos (13 / 4 * Real.pi) * Real.tan (-37 / 4 * Real.pi) = 
  (Real.sqrt 3 - Real.sqrt 2) / 2 := by sorry

-- Part 2
theorem trigonometric_identities (a : Real) (h : Real.tan a = 4 / 3) : 
  (Real.sin a ^ 2 + 2 * Real.sin a * Real.cos a) / (2 * Real.cos a ^ 2 - Real.sin a ^ 2) = 20 ∧
  Real.sin a * Real.cos a = 12 / 25 := by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_trigonometric_identities_l1510_151021


namespace NUMINAMATH_CALUDE_farm_field_solution_l1510_151078

/-- Represents the farm field ploughing problem -/
structure FarmField where
  planned_hectares_per_day : ℕ
  actual_hectares_per_day : ℕ
  extra_days : ℕ
  hectares_left : ℕ

/-- Calculates the total area and initial planned days for the farm field -/
def calculate_farm_area_and_days (f : FarmField) : ℕ × ℕ :=
  let initial_days := (f.actual_hectares_per_day * (f.extra_days + 1) + f.hectares_left) / f.planned_hectares_per_day
  let total_area := f.planned_hectares_per_day * initial_days + f.hectares_left
  (total_area, initial_days)

/-- Theorem stating the solution to the farm field problem -/
theorem farm_field_solution (f : FarmField) 
  (h1 : f.planned_hectares_per_day = 160)
  (h2 : f.actual_hectares_per_day = 85)
  (h3 : f.extra_days = 2)
  (h4 : f.hectares_left = 40) :
  calculate_farm_area_and_days f = (520, 3) := by
  sorry

#eval calculate_farm_area_and_days { planned_hectares_per_day := 160, actual_hectares_per_day := 85, extra_days := 2, hectares_left := 40 }

end NUMINAMATH_CALUDE_farm_field_solution_l1510_151078


namespace NUMINAMATH_CALUDE_vector_properties_l1510_151091

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-2, 4)

theorem vector_properties :
  let cos_theta := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))
  let proj_b_on_a := (Real.sqrt (b.1^2 + b.2^2) * cos_theta)
  cos_theta = (4 * Real.sqrt 65) / 65 ∧
  proj_b_on_a = (8 * Real.sqrt 13) / 13 := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l1510_151091


namespace NUMINAMATH_CALUDE_sector_radius_l1510_151076

theorem sector_radius (A : ℝ) (θ : ℝ) (r : ℝ) : 
  A = 6 * Real.pi → θ = (4 * Real.pi) / 3 → A = (1/2) * r^2 * θ → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l1510_151076


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_l1510_151015

/-- Two vectors in R² are perpendicular if and only if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The theorem states that if vectors (m, 4) and (m+4, 1) are perpendicular, then m = -2 -/
theorem perpendicular_vectors_m (m : ℝ) :
  perpendicular (m, 4) (m + 4, 1) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_l1510_151015


namespace NUMINAMATH_CALUDE_rectangle_frame_area_l1510_151075

theorem rectangle_frame_area (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  a * b = ((a + 2) * (b + 2) - a * b) → 
  ((a = 3 ∧ b = 10) ∨ (a = 10 ∧ b = 3) ∨ (a = 4 ∧ b = 6) ∨ (a = 6 ∧ b = 4)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_frame_area_l1510_151075


namespace NUMINAMATH_CALUDE_debby_total_messages_l1510_151025

/-- The total number of text messages Debby received -/
def total_messages (before_noon after_noon : ℕ) : ℕ := before_noon + after_noon

/-- Proof that Debby received 39 text messages in total -/
theorem debby_total_messages :
  total_messages 21 18 = 39 := by
  sorry

end NUMINAMATH_CALUDE_debby_total_messages_l1510_151025


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1510_151065

theorem completing_square_equivalence (x : ℝ) :
  (x^2 - 6*x + 7 = 0) ↔ ((x - 3)^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1510_151065


namespace NUMINAMATH_CALUDE_abs_equation_sufficient_not_necessary_l1510_151011

/-- The distance from a point to the x-axis --/
def dist_to_x_axis (y : ℝ) : ℝ := |y|

/-- The distance from a point to the y-axis --/
def dist_to_y_axis (x : ℝ) : ℝ := |x|

/-- The condition that distances to both axes are equal --/
def equal_dist_to_axes (x y : ℝ) : Prop :=
  dist_to_x_axis y = dist_to_y_axis x

/-- The equation y = |x| --/
def abs_equation (x y : ℝ) : Prop := y = |x|

/-- Theorem stating that y = |x| is a sufficient but not necessary condition --/
theorem abs_equation_sufficient_not_necessary :
  (∀ x y : ℝ, abs_equation x y → equal_dist_to_axes x y) ∧
  ¬(∀ x y : ℝ, equal_dist_to_axes x y → abs_equation x y) :=
sorry

end NUMINAMATH_CALUDE_abs_equation_sufficient_not_necessary_l1510_151011


namespace NUMINAMATH_CALUDE_polynomial_properties_l1510_151000

-- Define the polynomial equation
def polynomial_equation (x a₀ a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  (2*x - 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4

-- Theorem statement
theorem polynomial_properties 
  (a₀ a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x, polynomial_equation x a₀ a₁ a₂ a₃ a₄) : 
  (a₁ + a₂ + a₃ + a₄ = -80) ∧ 
  ((a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 625) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_properties_l1510_151000


namespace NUMINAMATH_CALUDE_min_value_inequality_l1510_151009

theorem min_value_inequality (a b c d : ℝ) 
  (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≥ 2) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (d/c - 1)^2 ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1510_151009


namespace NUMINAMATH_CALUDE_carbon_count_in_compound_l1510_151092

/-- Represents the atomic weights of elements in atomic mass units (amu) -/
structure AtomicWeights where
  copper : ℝ
  carbon : ℝ
  oxygen : ℝ

/-- Calculates the molecular weight of a compound given its composition -/
def molecularWeight (weights : AtomicWeights) (copperCount : ℕ) (carbonCount : ℕ) (oxygenCount : ℕ) : ℝ :=
  weights.copper * copperCount + weights.carbon * carbonCount + weights.oxygen * oxygenCount

/-- Theorem stating that a compound with 1 Copper, n Carbon, and 3 Oxygen atoms
    with a molecular weight of 124 amu has 1 Carbon atom -/
theorem carbon_count_in_compound (weights : AtomicWeights) 
    (h1 : weights.copper = 63.55)
    (h2 : weights.carbon = 12.01)
    (h3 : weights.oxygen = 16.00) :
  ∃ (n : ℕ), molecularWeight weights 1 n 3 = 124 ∧ n = 1 := by
  sorry


end NUMINAMATH_CALUDE_carbon_count_in_compound_l1510_151092


namespace NUMINAMATH_CALUDE_work_time_relation_l1510_151032

/-- Represents the amount of work that can be done by a group of people in a given time -/
structure WorkCapacity where
  people : ℕ
  work : ℝ
  days : ℝ

/-- The work rate is constant for a given group size -/
axiom work_rate_constant (w : WorkCapacity) : w.work / w.days = w.people

/-- The theorem stating the relationship between work, people, and time -/
theorem work_time_relation (w1 w2 : WorkCapacity) 
  (h1 : w1.people = 3 ∧ w1.work = 3)
  (h2 : w2.people = 5 ∧ w2.work = 5)
  (h3 : w1.days = w2.days) :
  ∃ (original_work : WorkCapacity), 
    original_work.people = 3 ∧ 
    original_work.work = 1 ∧ 
    original_work.days = w1.days / 3 :=
sorry

end NUMINAMATH_CALUDE_work_time_relation_l1510_151032


namespace NUMINAMATH_CALUDE_x_value_when_z_64_l1510_151083

/-- Given that x is directly proportional to y^4 and y is inversely proportional to z^2,
    prove that x = 1/4 when z = 64, given that x = 4 when z = 16. -/
theorem x_value_when_z_64 
  (h1 : ∃ (k₁ : ℝ), ∀ (x y : ℝ), x = k₁ * y^4)
  (h2 : ∃ (k₂ : ℝ), ∀ (y z : ℝ), y * z^2 = k₂)
  (h3 : ∃ (x y : ℝ), x = 4 ∧ y^4 = 1/16^4)
  : ∃ (x y : ℝ), x = 1/4 ∧ y^4 = 1/64^4 :=
sorry

end NUMINAMATH_CALUDE_x_value_when_z_64_l1510_151083


namespace NUMINAMATH_CALUDE_figure2_segment_length_l1510_151066

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square with given side length -/
structure Square where
  side : ℝ

/-- Calculates the total length of visible segments after cutting a square from a rectangle -/
def visibleSegmentsLength (rect : Rectangle) (sq : Square) : ℝ :=
  rect.length + (rect.width - sq.side) + (rect.length - sq.side) + sq.side

/-- Theorem stating that the total length of visible segments in Figure 2 is 23 units -/
theorem figure2_segment_length :
  let rect : Rectangle := { length := 10, width := 6 }
  let sq : Square := { side := 3 }
  visibleSegmentsLength rect sq = 23 := by
  sorry

end NUMINAMATH_CALUDE_figure2_segment_length_l1510_151066


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_neg_one_range_of_a_for_p_sufficient_not_necessary_for_q_l1510_151034

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 + 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - 6*x - 72 ≤ 0 ∧ x^2 + x - 6 > 0

-- Part 1: Range of x when a = -1
theorem range_of_x_when_a_is_neg_one :
  ∀ x : ℝ, (p x (-1) ∨ q x) ↔ x ∈ Set.Ioc (-6) (-3) ∪ Set.Icc 1 12 :=
sorry

-- Part 2: Range of a when p is sufficient but not necessary for q
theorem range_of_a_for_p_sufficient_not_necessary_for_q :
  {a : ℝ | ∀ x : ℝ, p x a → q x} ∩ {a : ℝ | ∃ x : ℝ, q x ∧ ¬p x a} = Set.Icc (-4) (-2) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_neg_one_range_of_a_for_p_sufficient_not_necessary_for_q_l1510_151034


namespace NUMINAMATH_CALUDE_triangle_angle_b_value_l1510_151037

/-- Given a triangle ABC with side lengths a and b, and angle A, proves that angle B has a specific value. -/
theorem triangle_angle_b_value 
  (a b : ℝ) 
  (A B : ℝ) 
  (h1 : a = 2 * Real.sqrt 3)
  (h2 : b = Real.sqrt 6)
  (h3 : A = π/4)  -- 45° in radians
  (h4 : 0 < A ∧ A < π)  -- A is a valid angle
  (h5 : 0 < B ∧ B < π)  -- B is a valid angle
  : B = π/6  -- 30° in radians
:= by sorry

end NUMINAMATH_CALUDE_triangle_angle_b_value_l1510_151037


namespace NUMINAMATH_CALUDE_sqrt_3_simplest_l1510_151093

def is_simplest_sqrt (x : ℝ) (options : List ℝ) : Prop :=
  ∀ y ∈ options, (∃ z : ℝ, z ^ 2 = x) → (∃ w : ℝ, w ^ 2 = y) → x ≤ y

theorem sqrt_3_simplest : 
  is_simplest_sqrt 3 [0.1, 8, (abs a), 3] := by sorry

end NUMINAMATH_CALUDE_sqrt_3_simplest_l1510_151093


namespace NUMINAMATH_CALUDE_lauren_jane_equation_l1510_151008

theorem lauren_jane_equation (x : ℝ) :
  (∀ x, |x - 4| = 3 ↔ x^2 + b*x + c = 0) →
  (b : ℝ) = -8 ∧ (c : ℝ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_lauren_jane_equation_l1510_151008


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1510_151031

theorem inequality_solution_set (x : ℝ) : 
  (4 * x^2 - 3 * x > 5) ↔ (x < -5/4 ∨ x > 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1510_151031


namespace NUMINAMATH_CALUDE_ratio_closest_to_ten_l1510_151059

theorem ratio_closest_to_ten : 
  let r := (10^3000 + 10^3004) / (10^3001 + 10^3003)
  ∀ n : ℤ, n ≠ 10 → |r - 10| < |r - n| := by
  sorry

end NUMINAMATH_CALUDE_ratio_closest_to_ten_l1510_151059


namespace NUMINAMATH_CALUDE_negation_equivalence_l1510_151055

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 3*x + 2 < 0) ↔ (∀ x : ℝ, x^2 + 3*x + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1510_151055


namespace NUMINAMATH_CALUDE_worker_arrival_delay_l1510_151017

theorem worker_arrival_delay (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_time = 60 → 
  let new_speed := (4/5) * usual_speed
  let new_time := usual_time * (usual_speed / new_speed)
  new_time - usual_time = 15 := by sorry

end NUMINAMATH_CALUDE_worker_arrival_delay_l1510_151017


namespace NUMINAMATH_CALUDE_seashells_given_proof_l1510_151088

def seashells_given_to_jessica (initial_seashells remaining_seashells : ℝ) : ℝ :=
  initial_seashells - remaining_seashells

theorem seashells_given_proof (initial_seashells remaining_seashells : ℝ) 
  (h1 : initial_seashells = 62.5) 
  (h2 : remaining_seashells = 30.75) : 
  seashells_given_to_jessica initial_seashells remaining_seashells = 31.75 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_proof_l1510_151088


namespace NUMINAMATH_CALUDE_complex_square_eq_neg100_minus_64i_l1510_151060

theorem complex_square_eq_neg100_minus_64i (z : ℂ) :
  z^2 = -100 - 64*I ↔ z = 4 - 8*I ∨ z = -4 + 8*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_eq_neg100_minus_64i_l1510_151060


namespace NUMINAMATH_CALUDE_max_log_sum_l1510_151043

theorem max_log_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 4 * y = 40) :
  ∃ (max_val : ℝ), max_val = 2 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + 4 * b = 40 → Real.log a + Real.log b ≤ max_val := by
  sorry

end NUMINAMATH_CALUDE_max_log_sum_l1510_151043


namespace NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l1510_151016

theorem midpoint_sum_equals_vertex_sum (d e f : ℝ) :
  let vertex_sum := d + e + f
  let midpoint_sum := (d + e) / 2 + (d + f) / 2 + (e + f) / 2
  vertex_sum = midpoint_sum := by sorry

end NUMINAMATH_CALUDE_midpoint_sum_equals_vertex_sum_l1510_151016


namespace NUMINAMATH_CALUDE_expression_evaluations_l1510_151035

theorem expression_evaluations :
  -- Part 1
  (25 ^ (1/3) - 125 ^ (1/2)) / (5 ^ (1/4)) = 5 ^ (5/12) - 5 * (5 ^ (1/4)) ∧
  -- Part 2
  ∀ a : ℝ, a > 0 → a^2 / (a^(1/2) * a^(2/3)) = a^(5/6) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluations_l1510_151035


namespace NUMINAMATH_CALUDE_sally_quarters_l1510_151022

theorem sally_quarters (initial : ℕ) (spent : ℕ) (remaining : ℕ) : 
  initial = 760 → spent = 418 → remaining = initial - spent → remaining = 342 := by
sorry

end NUMINAMATH_CALUDE_sally_quarters_l1510_151022


namespace NUMINAMATH_CALUDE_jerry_bacon_strips_l1510_151024

/-- Represents the number of calories in Jerry's breakfast items and total breakfast --/
structure BreakfastCalories where
  pancakeCalories : ℕ
  baconCalories : ℕ
  cerealCalories : ℕ
  totalCalories : ℕ

/-- Calculates the number of bacon strips in Jerry's breakfast --/
def calculateBaconStrips (b : BreakfastCalories) : ℕ :=
  (b.totalCalories - (6 * b.pancakeCalories + b.cerealCalories)) / b.baconCalories

/-- Theorem stating that Jerry had 2 strips of bacon for breakfast --/
theorem jerry_bacon_strips :
  let b : BreakfastCalories := {
    pancakeCalories := 120,
    baconCalories := 100,
    cerealCalories := 200,
    totalCalories := 1120
  }
  calculateBaconStrips b = 2 := by
  sorry


end NUMINAMATH_CALUDE_jerry_bacon_strips_l1510_151024


namespace NUMINAMATH_CALUDE_rope_length_problem_l1510_151084

theorem rope_length_problem (total_ropes : ℕ) (avg_length : ℝ) 
  (subset_ropes : ℕ) (subset_avg_length : ℝ) 
  (ratio_a ratio_b ratio_c : ℝ) :
  total_ropes = 9 →
  avg_length = 90 →
  subset_ropes = 3 →
  subset_avg_length = 70 →
  ratio_a = 2 ∧ ratio_b = 3 ∧ ratio_c = 5 →
  let remaining_ropes := total_ropes - subset_ropes
  let total_length := total_ropes * avg_length
  let subset_length := subset_ropes * subset_avg_length
  let remaining_length := total_length - subset_length
  (remaining_length / remaining_ropes : ℝ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_rope_length_problem_l1510_151084


namespace NUMINAMATH_CALUDE_abc_relationship_l1510_151068

theorem abc_relationship : ∀ (a b c : ℕ),
  a = 3^44 → b = 4^33 → c = 5^22 → c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_abc_relationship_l1510_151068


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1510_151099

/-- Simple interest calculation --/
theorem simple_interest_rate_calculation 
  (principal : ℝ) 
  (interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 5000)
  (h2 : interest = 2500)
  (h3 : time = 5)
  : (interest * 100) / (principal * time) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1510_151099


namespace NUMINAMATH_CALUDE_square_root_problem_l1510_151063

theorem square_root_problem (x y : ℝ) 
  (h1 : (x - 1) = 9) 
  (h2 : (2 * x + y + 7) = 8) : 
  (7 - x - y) = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l1510_151063


namespace NUMINAMATH_CALUDE_sports_camp_coach_age_l1510_151052

theorem sports_camp_coach_age (total_members : ℕ) (avg_age : ℕ) 
  (num_girls num_boys num_coaches : ℕ) (avg_age_girls avg_age_boys : ℕ) :
  total_members = 30 →
  avg_age = 20 →
  num_girls = 10 →
  num_boys = 15 →
  num_coaches = 5 →
  avg_age_girls = 18 →
  avg_age_boys = 19 →
  (total_members * avg_age - num_girls * avg_age_girls - num_boys * avg_age_boys) / num_coaches = 27 :=
by sorry

end NUMINAMATH_CALUDE_sports_camp_coach_age_l1510_151052


namespace NUMINAMATH_CALUDE_sequence_a_11_l1510_151019

theorem sequence_a_11 (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) 
  (h : ∀ n : ℕ+, 4 * S n = 2 * a n - n.val^2 + 7 * n.val) : 
  a 11 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_11_l1510_151019


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l1510_151026

theorem rectangular_solid_diagonal 
  (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 24) 
  (h2 : a + b + c = 6) : 
  Real.sqrt (a^2 + b^2 + c^2) = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l1510_151026


namespace NUMINAMATH_CALUDE_total_hotdogs_is_480_l1510_151010

/-- The number of hotdogs Helen's mother brought -/
def helen_hotdogs : ℕ := 101

/-- The number of hotdogs Dylan's mother brought -/
def dylan_hotdogs : ℕ := 379

/-- The total number of hotdogs -/
def total_hotdogs : ℕ := helen_hotdogs + dylan_hotdogs

/-- Theorem stating that the total number of hotdogs is 480 -/
theorem total_hotdogs_is_480 : total_hotdogs = 480 := by
  sorry

end NUMINAMATH_CALUDE_total_hotdogs_is_480_l1510_151010


namespace NUMINAMATH_CALUDE_complex_sum_powers_l1510_151046

theorem complex_sum_powers (x : ℂ) (h : x^2 + x + 1 = 0) :
  x^49 + x^50 + x^51 + x^52 + x^53 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_powers_l1510_151046


namespace NUMINAMATH_CALUDE_x_gt_3_sufficient_not_necessary_l1510_151071

theorem x_gt_3_sufficient_not_necessary :
  (∃ x : ℝ, x ≤ 3 ∧ 1 / x < 1 / 3) ∧
  (∀ x : ℝ, x > 3 → 1 / x < 1 / 3) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_3_sufficient_not_necessary_l1510_151071


namespace NUMINAMATH_CALUDE_baking_time_l1510_151029

/-- 
Given:
- It takes 7 minutes to bake 1 pan of cookies
- The total time to bake 4 pans is 28 minutes

Prove that the time to bake 4 pans of cookies is 28 minutes.
-/
theorem baking_time (time_for_one_pan : ℕ) (total_time : ℕ) (num_pans : ℕ) :
  time_for_one_pan = 7 →
  total_time = 28 →
  num_pans = 4 →
  total_time = 28 := by
sorry

end NUMINAMATH_CALUDE_baking_time_l1510_151029


namespace NUMINAMATH_CALUDE_grade_change_impossibility_l1510_151077

theorem grade_change_impossibility : ¬ ∃ (n₁ n₂ n₃ n₄ : ℤ),
  2 * n₁ + n₂ - 2 * n₃ - n₄ = 27 ∧
  -n₁ + 2 * n₂ + n₃ - 2 * n₄ = -27 :=
sorry

end NUMINAMATH_CALUDE_grade_change_impossibility_l1510_151077


namespace NUMINAMATH_CALUDE_diamond_two_three_l1510_151023

/-- The diamond operation defined for real numbers -/
def diamond (a b : ℝ) : ℝ := a * b^2 - b + 1

/-- Theorem stating that 2 ◇ 3 = 16 -/
theorem diamond_two_three : diamond 2 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_three_l1510_151023


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l1510_151027

theorem binomial_coefficient_sum (a : ℝ) : 
  (∃ n : ℕ, ∃ x : ℝ, (2 : ℝ)^n = 256 ∧ (∀ k : ℕ, k ≤ n → ∃ c : ℝ, c * (a/x + 3)^(n-k) * x^k = c * (a + 3*x)^(n-k))) → 
  (a = -1 ∨ a = -5) := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l1510_151027


namespace NUMINAMATH_CALUDE_smallest_divisor_of_930_l1510_151058

theorem smallest_divisor_of_930 : ∃ (d : ℕ), d > 1 ∧ d ∣ 930 ∧ ∀ (k : ℕ), 1 < k ∧ k < d → ¬(k ∣ 930) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_930_l1510_151058


namespace NUMINAMATH_CALUDE_flower_bed_properties_l1510_151079

/-- Represents a rectangular flower bed with specific properties -/
structure FlowerBed where
  length : ℝ
  width : ℝ
  area : ℝ
  theta : ℝ

/-- Theorem about the properties of a specific flower bed -/
theorem flower_bed_properties :
  ∃ (fb : FlowerBed),
    fb.area = 50 ∧
    fb.width = (2/3) * fb.length ∧
    Real.tan fb.theta = (fb.length - fb.width) / (fb.length + fb.width) ∧
    fb.length = 75 ∧
    fb.width = 50 ∧
    Real.tan fb.theta = 1/5 := by
  sorry


end NUMINAMATH_CALUDE_flower_bed_properties_l1510_151079


namespace NUMINAMATH_CALUDE_selection_test_results_l1510_151094

/-- Represents the probability of A answering a question correctly -/
def prob_A_correct : ℚ := 3/5

/-- Represents the number of questions B can answer correctly out of 10 -/
def B_correct_answers : ℕ := 5

/-- Represents the total number of questions in the pool -/
def total_questions : ℕ := 10

/-- Represents the number of questions in each exam -/
def exam_questions : ℕ := 3

/-- Represents the score for a correct answer -/
def correct_score : ℤ := 10

/-- Represents the score deduction for an incorrect answer -/
def incorrect_score : ℤ := -5

/-- Represents the minimum score required for selection -/
def selection_threshold : ℤ := 15

/-- The expected score for A -/
def expected_score_A : ℚ := 12

/-- The probability that both A and B are selected -/
def prob_both_selected : ℚ := 81/250

theorem selection_test_results :
  (prob_A_correct = 3/5) →
  (B_correct_answers = 5) →
  (total_questions = 10) →
  (exam_questions = 3) →
  (correct_score = 10) →
  (incorrect_score = -5) →
  (selection_threshold = 15) →
  (expected_score_A = 12) ∧
  (prob_both_selected = 81/250) := by
  sorry

end NUMINAMATH_CALUDE_selection_test_results_l1510_151094


namespace NUMINAMATH_CALUDE_line_contains_point_l1510_151006

/-- 
Given a line represented by the equation 1-kx = -3y that contains the point (4, -3),
prove that the value of k is -2.
-/
theorem line_contains_point (k : ℝ) : 
  (1 - k * 4 = -3 * (-3)) → k = -2 := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l1510_151006


namespace NUMINAMATH_CALUDE_smallest_n_and_b_over_a_l1510_151074

theorem smallest_n_and_b_over_a : ∃ (n : ℕ+) (a b : ℝ),
  (∀ m : ℕ+, m < n → ¬∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + 3*y*Complex.I)^(m:ℕ) = (x - 3*y*Complex.I)^(m:ℕ)) ∧
  a > 0 ∧ b > 0 ∧
  (a + 3*b*Complex.I)^(n:ℕ) = (a - 3*b*Complex.I)^(n:ℕ) ∧
  b/a = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_and_b_over_a_l1510_151074


namespace NUMINAMATH_CALUDE_original_denominator_problem_l1510_151030

theorem original_denominator_problem (d : ℤ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 7 : ℚ) / (d + 7) = 1 / 3 →
  d = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_original_denominator_problem_l1510_151030


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1510_151057

theorem triangle_perimeter (a b c : ℝ) :
  |a - 2 * Real.sqrt 2| + Real.sqrt (b - 5) + (c - 3 * Real.sqrt 2)^2 = 0 →
  a + b + c = 5 + 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1510_151057


namespace NUMINAMATH_CALUDE_rectangle_existence_l1510_151070

theorem rectangle_existence (m : ℕ) (h : m > 12) :
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * y > m ∧ x * (y - 1) < m ∧ x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_rectangle_existence_l1510_151070


namespace NUMINAMATH_CALUDE_u_converges_to_L_least_k_for_bound_l1510_151002

def u : ℕ → ℚ
  | 0 => 1/8
  | n + 1 => 2 * u n - 2 * (u n)^2

def L : ℚ := 1/2

def converges_to (a : ℕ → ℚ) (l : ℚ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - l| < ε

theorem u_converges_to_L : converges_to u L := sorry

theorem least_k_for_bound :
  (∃ k, |u k - L| ≤ 1/2^10) ∧
  (∀ k < 4, |u k - L| > 1/2^10) ∧
  |u 4 - L| ≤ 1/2^10 := sorry

end NUMINAMATH_CALUDE_u_converges_to_L_least_k_for_bound_l1510_151002


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_lines_l1510_151042

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for lines and planes
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the property of a line being outside a plane
variable (outside : Line → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes_from_perpendicular_lines
  (α β : Plane) (m n : Line)
  (different_planes : α ≠ β)
  (distinct_lines : m ≠ n)
  (m_outside_α : outside m α)
  (m_outside_β : outside m β)
  (n_outside_α : outside n α)
  (n_outside_β : outside n β)
  (h1 : perpendicular_lines m n)
  (h3 : perpendicular_line_plane n β)
  (h4 : perpendicular_line_plane m α) :
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_lines_l1510_151042


namespace NUMINAMATH_CALUDE_divisibility_by_1995_l1510_151062

theorem divisibility_by_1995 (n : ℕ) : 
  1995 ∣ 256^(2*n) * 7^(2*n) - 168^(2*n) - 32^(2*n) + 3^(2*n) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_1995_l1510_151062


namespace NUMINAMATH_CALUDE_quadratic_roots_l1510_151090

theorem quadratic_roots (a b : ℝ) (h1 : a * b ≠ 0) 
  (h2 : a^2 + 2*b*a + a = 0) (h3 : b^2 + 2*b*b + a = 0) : 
  a = -3 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1510_151090


namespace NUMINAMATH_CALUDE_greatest_number_in_set_l1510_151044

/-- A set of consecutive multiples of 2 -/
def ConsecutiveMultiplesOf2 (n : ℕ) (start : ℕ) : Set ℕ :=
  {x : ℕ | ∃ k : ℕ, k < n ∧ x = start + 2 * k}

theorem greatest_number_in_set (s : Set ℕ) :
  s = ConsecutiveMultiplesOf2 50 56 →
  ∃ m : ℕ, m ∈ s ∧ ∀ x ∈ s, x ≤ m ∧ m = 154 :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_in_set_l1510_151044


namespace NUMINAMATH_CALUDE_tangent_line_equation_chord_line_equation_l1510_151047

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 12 = 0

-- Define the point P
def P : ℝ × ℝ := (-2, 0)

-- Define a line passing through P
def line_through_P (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 2)

-- Define tangent line condition
def is_tangent (k : ℝ) : Prop :=
  ∃ (x y : ℝ), C x y ∧ line_through_P k x y ∧
  ∀ (x' y' : ℝ), C x' y' ∧ line_through_P k x' y' → (x', y') = (x, y)

-- Define chord length condition
def has_chord_length (k : ℝ) (len : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧
  line_through_P k x₁ y₁ ∧ line_through_P k x₂ y₂ ∧
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = len^2

-- Theorem for part (1)
theorem tangent_line_equation :
  ∀ k : ℝ, is_tangent k ↔ (k = 0 ∨ 3 * k = 4) :=
sorry

-- Theorem for part (2)
theorem chord_line_equation :
  ∀ k : ℝ, has_chord_length k (2 * Real.sqrt 2) ↔ (k = 1 ∨ k = 7) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_chord_line_equation_l1510_151047


namespace NUMINAMATH_CALUDE_monotonic_function_property_l1510_151033

/-- A monotonic function f satisfying f(f(x) - 3^x) = 4 for all x ∈ ℝ has f(2) = 10 -/
theorem monotonic_function_property (f : ℝ → ℝ) 
  (h_mono : Monotone f) 
  (h_prop : ∀ x, f (f x - 3^x) = 4) : 
  f 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_function_property_l1510_151033


namespace NUMINAMATH_CALUDE_equation_solutions_l1510_151041

open Real

theorem equation_solutions (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    sin (x₁ - a) + cos (x₁ + 3 * a) = 0 ∧
    sin (x₂ - a) + cos (x₂ + 3 * a) = 0 ∧
    ∀ k : ℤ, x₁ - x₂ ≠ π * k) ↔
  ∃ t : ℤ, a = π * (4 * t + 1) / 8 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1510_151041


namespace NUMINAMATH_CALUDE_popsicle_sticks_left_l1510_151039

/-- Calculates the number of popsicle sticks Miss Davis has left after distribution -/
theorem popsicle_sticks_left (initial_sticks : ℕ) (sticks_per_group : ℕ) (num_groups : ℕ) : 
  initial_sticks = 170 → sticks_per_group = 15 → num_groups = 10 → 
  initial_sticks - (sticks_per_group * num_groups) = 20 := by
sorry

end NUMINAMATH_CALUDE_popsicle_sticks_left_l1510_151039


namespace NUMINAMATH_CALUDE_problem_solution_l1510_151064

theorem problem_solution (t s x : ℝ) : 
  t = 15 * s^2 → t = 3.75 → x = s / 2 → s = 0.5 ∧ x = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1510_151064


namespace NUMINAMATH_CALUDE_power_function_range_l1510_151067

-- Define the power function f
def f (x : ℝ) : ℝ := x^2

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + 1

-- State the theorem
theorem power_function_range (m : ℝ) : 
  (f (Real.sqrt 3) = 3) → 
  (∀ x ∈ Set.Icc m 2, g x ∈ Set.Icc 1 5) → 
  (m ∈ Set.Icc (-2) 0) :=
by sorry

end NUMINAMATH_CALUDE_power_function_range_l1510_151067


namespace NUMINAMATH_CALUDE_routes_from_bristol_to_birmingham_l1510_151086

theorem routes_from_bristol_to_birmingham :
  ∀ (bristol_to_birmingham birmingham_to_sheffield sheffield_to_carlisle bristol_to_carlisle : ℕ),
    birmingham_to_sheffield = 3 →
    sheffield_to_carlisle = 2 →
    bristol_to_carlisle = 36 →
    bristol_to_carlisle = bristol_to_birmingham * birmingham_to_sheffield * sheffield_to_carlisle →
    bristol_to_birmingham = 6 := by
  sorry

end NUMINAMATH_CALUDE_routes_from_bristol_to_birmingham_l1510_151086


namespace NUMINAMATH_CALUDE_vanessa_chocolate_sales_l1510_151080

/-- Calculates the money made from selling chocolate bars -/
def money_made (total_bars : ℕ) (price_per_bar : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Proves that Vanessa made $16 from selling chocolate bars -/
theorem vanessa_chocolate_sales :
  money_made 11 4 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_chocolate_sales_l1510_151080


namespace NUMINAMATH_CALUDE_rebecca_earnings_l1510_151005

/-- Rebecca's hair salon earnings calculation -/
theorem rebecca_earnings (haircut_price perm_price dye_job_price dye_cost : ℕ)
  (num_haircuts num_perms num_dye_jobs tips : ℕ) :
  haircut_price = 30 →
  perm_price = 40 →
  dye_job_price = 60 →
  dye_cost = 10 →
  num_haircuts = 4 →
  num_perms = 1 →
  num_dye_jobs = 2 →
  tips = 50 →
  (haircut_price * num_haircuts + 
   perm_price * num_perms + 
   dye_job_price * num_dye_jobs + 
   tips - 
   dye_cost * num_dye_jobs) = 310 :=
by sorry

end NUMINAMATH_CALUDE_rebecca_earnings_l1510_151005


namespace NUMINAMATH_CALUDE_larger_number_proof_l1510_151004

theorem larger_number_proof (x y : ℕ) (h1 : x > y) (h2 : x + y = 363) (h3 : x = 16 * y + 6) : x = 342 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1510_151004


namespace NUMINAMATH_CALUDE_function_properties_l1510_151014

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x + a) / (2^x - 1)

theorem function_properties (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → f a x = -f a (-x)) →
  (∀ x : ℝ, x ≠ 0 → f a x = f a x) ∧
  (a = 1) ∧
  (∀ x y : ℝ, 0 < x → x < y → f a y < f a x) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1510_151014


namespace NUMINAMATH_CALUDE_return_journey_percentage_l1510_151053

/-- Represents the distance of a one-way trip -/
def one_way_distance : ℝ := 1

/-- Represents the total round-trip distance -/
def round_trip_distance : ℝ := 2 * one_way_distance

/-- Represents the percentage of the round-trip completed -/
def round_trip_completed_percentage : ℝ := 0.75

/-- Represents the distance traveled in the round-trip -/
def distance_traveled : ℝ := round_trip_completed_percentage * round_trip_distance

/-- Represents the distance traveled on the return journey -/
def return_journey_traveled : ℝ := distance_traveled - one_way_distance

theorem return_journey_percentage :
  return_journey_traveled / one_way_distance = 0.5 := by sorry

end NUMINAMATH_CALUDE_return_journey_percentage_l1510_151053


namespace NUMINAMATH_CALUDE_expand_and_simplify_fraction_l1510_151097

theorem expand_and_simplify_fraction (y : ℝ) (h : y ≠ 0) :
  (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_fraction_l1510_151097


namespace NUMINAMATH_CALUDE_equation_solution_l1510_151087

theorem equation_solution : ∃! x : ℝ, (3 : ℝ) / (x - 3) = (4 : ℝ) / (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1510_151087


namespace NUMINAMATH_CALUDE_new_person_weight_l1510_151054

/-- Given a group of 8 people, if replacing one person weighing 65 kg with a new person
    increases the average weight by 3.5 kg, then the weight of the new person is 93 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 3.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 93 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1510_151054


namespace NUMINAMATH_CALUDE_gcd_lcm_product_240_l1510_151061

theorem gcd_lcm_product_240 : 
  ∃! (s : Finset Nat), 
    (∀ d ∈ s, ∃ a b : Nat, a > 0 ∧ b > 0 ∧ Nat.gcd a b = d ∧ Nat.gcd a b * Nat.lcm a b = 240) ∧ 
    (∀ d : Nat, (∃ a b : Nat, a > 0 ∧ b > 0 ∧ Nat.gcd a b = d ∧ Nat.gcd a b * Nat.lcm a b = 240) → d ∈ s) ∧
    s.card = 10 :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_240_l1510_151061


namespace NUMINAMATH_CALUDE_binomial_product_l1510_151051

/-- The product of (2x² + 3y - 4) and (y + 6) is equal to 2x²y + 12x² + 3y² + 14y - 24 -/
theorem binomial_product (x y : ℝ) :
  (2 * x^2 + 3 * y - 4) * (y + 6) = 2 * x^2 * y + 12 * x^2 + 3 * y^2 + 14 * y - 24 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l1510_151051


namespace NUMINAMATH_CALUDE_total_stripes_is_34_l1510_151073

/-- The total number of stripes on Vaishali's hats -/
def total_stripes : ℕ :=
  let hats_with_3_stripes := 4
  let hats_with_4_stripes := 3
  let hats_with_0_stripes := 6
  let hats_with_5_stripes := 2
  hats_with_3_stripes * 3 +
  hats_with_4_stripes * 4 +
  hats_with_0_stripes * 0 +
  hats_with_5_stripes * 5

/-- Theorem stating that the total number of stripes on Vaishali's hats is 34 -/
theorem total_stripes_is_34 : total_stripes = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_stripes_is_34_l1510_151073


namespace NUMINAMATH_CALUDE_greatest_non_expressible_as_sum_of_composites_l1510_151048

-- Define what it means for a number to be composite
def IsComposite (n : ℕ) : Prop := n > 1 ∧ ¬(Nat.Prime n)

-- Define the property of being expressible as the sum of two composite numbers
def ExpressibleAsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ (a b : ℕ), IsComposite a ∧ IsComposite b ∧ n = a + b

-- State the theorem
theorem greatest_non_expressible_as_sum_of_composites :
  (∀ n > 11, ExpressibleAsSumOfTwoComposites n) ∧
  ¬(ExpressibleAsSumOfTwoComposites 11) := by sorry

end NUMINAMATH_CALUDE_greatest_non_expressible_as_sum_of_composites_l1510_151048


namespace NUMINAMATH_CALUDE_new_person_weight_l1510_151038

/-- Given a group of 12 people where one person weighing 62 kg is replaced by a new person,
    causing the average weight to increase by 4.8 kg, prove that the new person weighs 119.6 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 12 →
  weight_increase = 4.8 →
  replaced_weight = 62 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 119.6 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1510_151038


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1510_151085

theorem polynomial_factorization (a b : ℤ) : 
  (∀ x : ℝ, x^2 + a*x + b = (x+1)*(x-3)) → (a = -2 ∧ b = -3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1510_151085


namespace NUMINAMATH_CALUDE_recurrence_2004_values_l1510_151036

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * (a n + 2)

/-- The set of possible values for the 2004th term of the sequence -/
def PossibleValues (a : ℕ → ℝ) : Set ℝ :=
  {x : ℝ | ∃ (seq : ℕ → ℝ), RecurrenceSequence seq ∧ seq 2004 = x}

/-- The theorem stating that the set of possible values for a₂₀₀₄ is [-1, ∞) -/
theorem recurrence_2004_values :
  ∀ a : ℕ → ℝ, RecurrenceSequence a →
  PossibleValues a = Set.Ici (-1) :=
sorry

end NUMINAMATH_CALUDE_recurrence_2004_values_l1510_151036


namespace NUMINAMATH_CALUDE_f_derivative_at_negative_one_l1510_151001

noncomputable def f (x : ℝ) : ℝ := Real.log (-2 * x) + 3 * x

theorem f_derivative_at_negative_one :
  deriv f (-1) = 2 :=
sorry

end NUMINAMATH_CALUDE_f_derivative_at_negative_one_l1510_151001


namespace NUMINAMATH_CALUDE_two_adults_in_group_l1510_151020

/-- Represents the restaurant bill problem --/
def restaurant_bill_problem (num_children : ℕ) (meal_cost : ℕ) (total_bill : ℕ) : Prop :=
  ∃ (num_adults : ℕ), 
    num_adults * meal_cost + num_children * meal_cost = total_bill

/-- Proves that there are 2 adults in the group --/
theorem two_adults_in_group : 
  restaurant_bill_problem 5 3 21 → 
  ∃ (num_adults : ℕ), num_adults = 2 ∧ restaurant_bill_problem 5 3 21 :=
by sorry

end NUMINAMATH_CALUDE_two_adults_in_group_l1510_151020


namespace NUMINAMATH_CALUDE_three_non_congruent_triangles_l1510_151069

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  perimeter_eq : a + b + c = 11
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- Two triangles are congruent if they have the same side lengths (up to permutation) -/
def congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of all valid integer triangles with perimeter 11 -/
def valid_triangles : Set IntTriangle :=
  {t : IntTriangle | True}

/-- The theorem to be proved -/
theorem three_non_congruent_triangles :
  ∃ (t1 t2 t3 : IntTriangle),
    t1 ∈ valid_triangles ∧ t2 ∈ valid_triangles ∧ t3 ∈ valid_triangles ∧
    ¬(congruent t1 t2) ∧ ¬(congruent t2 t3) ∧ ¬(congruent t1 t3) ∧
    ∀ (t : IntTriangle), t ∈ valid_triangles →
      congruent t t1 ∨ congruent t t2 ∨ congruent t t3 :=
by sorry

end NUMINAMATH_CALUDE_three_non_congruent_triangles_l1510_151069


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1510_151012

/-- Given a line L1 with equation x - 3y + 2 = 0 and a point P(1, 2),
    prove that the line L2 with equation 3x + y - 5 = 0 passes through P
    and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => x - 3*y + 2 = 0
  let L2 : ℝ → ℝ → Prop := λ x y => 3*x + y - 5 = 0
  let P : ℝ × ℝ := (1, 2)
  (L2 P.1 P.2) ∧                        -- L2 passes through P
  (∀ x1 y1 x2 y2 : ℝ,                   -- L2 is perpendicular to L1
    L1 x1 y1 → L1 x2 y2 → L2 x1 y1 → L2 x2 y2 →
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    ((x2 - x1) * 1 + (y2 - y1) * (-3)) * ((x2 - x1) * 3 + (y2 - y1) * 1) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1510_151012


namespace NUMINAMATH_CALUDE_b_fourth_congruence_l1510_151028

theorem b_fourth_congruence (n : ℕ+) (b : ℤ) (h : b^2 ≡ 1 [ZMOD n]) :
  b^4 ≡ 1 [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_b_fourth_congruence_l1510_151028


namespace NUMINAMATH_CALUDE_eric_required_bike_speed_l1510_151081

/-- Represents the triathlon components --/
structure Triathlon :=
  (swim_distance : ℚ)
  (swim_speed : ℚ)
  (run_distance : ℚ)
  (run_speed : ℚ)
  (bike_distance : ℚ)
  (total_time : ℚ)

/-- Calculates the required bike speed for a given triathlon --/
def required_bike_speed (t : Triathlon) : ℚ :=
  let swim_time := t.swim_distance / t.swim_speed
  let run_time := t.run_distance / t.run_speed
  let bike_time := t.total_time - (swim_time + run_time)
  t.bike_distance / bike_time

/-- The triathlon problem --/
def eric_triathlon : Triathlon :=
  { swim_distance := 1/4
  , swim_speed := 2
  , run_distance := 3
  , run_speed := 6
  , bike_distance := 15
  , total_time := 2 }

/-- Theorem stating that the required bike speed for Eric's triathlon is 120/11 --/
theorem eric_required_bike_speed :
  required_bike_speed eric_triathlon = 120/11 := by sorry


end NUMINAMATH_CALUDE_eric_required_bike_speed_l1510_151081


namespace NUMINAMATH_CALUDE_equation_has_one_solution_l1510_151056

theorem equation_has_one_solution :
  ∃! x : ℝ, x - 8 / (x - 2) = 4 - 8 / (x - 2) ∧ x ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_one_solution_l1510_151056


namespace NUMINAMATH_CALUDE_floor_length_percentage_l1510_151050

-- Define the parameters of the problem
def floor_length : ℝ := 18.9999683334125
def total_cost : ℝ := 361
def paint_rate : ℝ := 3.00001

-- Define the theorem
theorem floor_length_percentage (l b : ℝ) (h1 : l = floor_length) 
  (h2 : l * b = total_cost / paint_rate) : 
  (l - b) / b * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_floor_length_percentage_l1510_151050


namespace NUMINAMATH_CALUDE_jenga_players_l1510_151098

/-- The number of players in a Jenga game -/
def num_players : ℕ := 5

/-- The initial number of blocks in the Jenga tower -/
def initial_blocks : ℕ := 54

/-- The number of full rounds played -/
def full_rounds : ℕ := 5

/-- The number of blocks remaining after 5 full rounds and one additional move -/
def remaining_blocks : ℕ := 28

/-- The number of blocks removed in the 6th round before the tower falls -/
def extra_blocks_removed : ℕ := 1

theorem jenga_players :
  initial_blocks - remaining_blocks = full_rounds * num_players + extra_blocks_removed :=
sorry

end NUMINAMATH_CALUDE_jenga_players_l1510_151098


namespace NUMINAMATH_CALUDE_smallest_congruent_difference_l1510_151013

theorem smallest_congruent_difference : ∃ m n : ℕ,
  (m ≥ 100 ∧ m < 1000 ∧ m % 13 = 3 ∧ ∀ k, k ≥ 100 ∧ k < 1000 ∧ k % 13 = 3 → m ≤ k) ∧
  (n ≥ 1000 ∧ n < 10000 ∧ n % 13 = 3 ∧ ∀ l, l ≥ 1000 ∧ l < 10000 ∧ l % 13 = 3 → n ≤ l) ∧
  n - m = 896 :=
by sorry

end NUMINAMATH_CALUDE_smallest_congruent_difference_l1510_151013


namespace NUMINAMATH_CALUDE_T_equals_eleven_l1510_151045

/-- Given a natural number S, we define F as the sum of powers of 2 from 0 to S -/
def F (S : ℕ) : ℝ := (2^(S+1) - 1)

/-- T is defined as the square root of the ratio of logarithms -/
noncomputable def T (S : ℕ) : ℝ := Real.sqrt (Real.log (1 + F S) / Real.log 2)

/-- The theorem states that for S = 120, T equals 11 -/
theorem T_equals_eleven : T 120 = 11 := by sorry

end NUMINAMATH_CALUDE_T_equals_eleven_l1510_151045


namespace NUMINAMATH_CALUDE_dinosaur_count_correct_l1510_151003

/-- Represents the number of dinosaurs in the flock -/
def num_dinosaurs : ℕ := 5

/-- Represents the number of legs each dinosaur has -/
def legs_per_dinosaur : ℕ := 3

/-- Represents the total number of heads and legs in the flock -/
def total_heads_and_legs : ℕ := 20

/-- Proves that the number of dinosaurs in the flock is correct -/
theorem dinosaur_count_correct :
  num_dinosaurs * (legs_per_dinosaur + 1) = total_heads_and_legs :=
by sorry

end NUMINAMATH_CALUDE_dinosaur_count_correct_l1510_151003


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l1510_151089

/-- The y-intercept of the line x - 2y = 5 is -5/2 -/
theorem y_intercept_of_line (x y : ℝ) : x - 2*y = 5 → y = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l1510_151089


namespace NUMINAMATH_CALUDE_next_meeting_after_105_days_l1510_151007

/-- Represents the number of days between cinema visits for each boy -/
structure VisitIntervals :=
  (kolya : ℕ)
  (seryozha : ℕ)
  (vanya : ℕ)

/-- The least number of days after which all three boys meet at the cinema again -/
def nextMeeting (intervals : VisitIntervals) : ℕ :=
  Nat.lcm intervals.kolya (Nat.lcm intervals.seryozha intervals.vanya)

/-- Theorem stating that the next meeting of all three boys occurs after 105 days -/
theorem next_meeting_after_105_days :
  let intervals : VisitIntervals := { kolya := 3, seryozha := 7, vanya := 5 }
  nextMeeting intervals = 105 := by sorry

end NUMINAMATH_CALUDE_next_meeting_after_105_days_l1510_151007


namespace NUMINAMATH_CALUDE_average_difference_l1510_151018

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

end NUMINAMATH_CALUDE_average_difference_l1510_151018


namespace NUMINAMATH_CALUDE_congruence_problem_l1510_151095

theorem congruence_problem (n : ℤ) : 
  0 ≤ n ∧ n < 200 ∧ (150 * n) % 199 = 110 % 199 → n % 199 = 157 % 199 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1510_151095


namespace NUMINAMATH_CALUDE_ratio_and_equation_solution_l1510_151040

theorem ratio_and_equation_solution (a b : ℝ) : 
  b / a = 4 → b = 16 - 6 * a + a^2 → (a = -5 + Real.sqrt 41 ∨ a = -5 - Real.sqrt 41) :=
by sorry

end NUMINAMATH_CALUDE_ratio_and_equation_solution_l1510_151040


namespace NUMINAMATH_CALUDE_pool_filling_time_l1510_151082

-- Define the rates of the valves
variable (a b c d : ℝ)

-- Define the conditions
def condition1 : Prop := 1/a + 1/b + 1/c = 1/12
def condition2 : Prop := 1/b + 1/c + 1/d = 1/15
def condition3 : Prop := 1/a + 1/d = 1/20

-- Theorem statement
theorem pool_filling_time 
  (h1 : condition1 a b c)
  (h2 : condition2 b c d)
  (h3 : condition3 a d) :
  1/a + 1/b + 1/c + 1/d = 1/10 := by
  sorry


end NUMINAMATH_CALUDE_pool_filling_time_l1510_151082


namespace NUMINAMATH_CALUDE_inequality_preservation_l1510_151049

theorem inequality_preservation (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1510_151049
