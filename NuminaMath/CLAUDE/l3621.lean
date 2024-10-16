import Mathlib

namespace NUMINAMATH_CALUDE_negative_two_less_than_negative_two_thirds_l3621_362153

theorem negative_two_less_than_negative_two_thirds : -2 < -(2/3) := by
  sorry

end NUMINAMATH_CALUDE_negative_two_less_than_negative_two_thirds_l3621_362153


namespace NUMINAMATH_CALUDE_rotation_and_scaling_l3621_362178

def rotate90Clockwise (z : ℂ) : ℂ := -z.im + z.re * Complex.I

theorem rotation_and_scaling :
  let z : ℂ := 3 + 4 * Complex.I
  let rotated := rotate90Clockwise z
  let scaled := 2 * rotated
  scaled = -8 - 6 * Complex.I := by sorry

end NUMINAMATH_CALUDE_rotation_and_scaling_l3621_362178


namespace NUMINAMATH_CALUDE_chemical_quantity_problem_l3621_362146

theorem chemical_quantity_problem (x : ℤ) : 
  532 * x - 325 * x = 1065430 → x = 5148 := by sorry

end NUMINAMATH_CALUDE_chemical_quantity_problem_l3621_362146


namespace NUMINAMATH_CALUDE_dogs_with_neither_l3621_362130

/-- Given a kennel with dogs, prove the number of dogs wearing neither tags nor flea collars -/
theorem dogs_with_neither (total : ℕ) (with_tags : ℕ) (with_collars : ℕ) (with_both : ℕ) 
  (h1 : total = 80)
  (h2 : with_tags = 45)
  (h3 : with_collars = 40)
  (h4 : with_both = 6) :
  total - (with_tags + with_collars - with_both) = 1 := by
  sorry

#check dogs_with_neither

end NUMINAMATH_CALUDE_dogs_with_neither_l3621_362130


namespace NUMINAMATH_CALUDE_swing_slide_wait_time_difference_l3621_362192

theorem swing_slide_wait_time_difference :
  let swingKids : ℕ := 6
  let slideKids : ℕ := 4 * swingKids
  let swingWaitTime1 : ℝ := 3.5 * 60  -- 3.5 minutes in seconds
  let slideWaitTime1 : ℝ := 45  -- 45 seconds
  let rounds : ℕ := 3
  
  let swingTotalWait : ℝ := swingKids * (swingWaitTime1 * (1 - 2^rounds) / (1 - 2))
  let slideTotalWait : ℝ := slideKids * (slideWaitTime1 * (1 - 2^rounds) / (1 - 2))
  
  swingTotalWait - slideTotalWait = 1260
  := by sorry

end NUMINAMATH_CALUDE_swing_slide_wait_time_difference_l3621_362192


namespace NUMINAMATH_CALUDE_five_letter_words_same_start_end_l3621_362115

theorem five_letter_words_same_start_end (alphabet_size : ℕ) (word_length : ℕ) : 
  alphabet_size = 26 → word_length = 5 → 
  (alphabet_size ^ (word_length - 2)) * alphabet_size = 456976 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_words_same_start_end_l3621_362115


namespace NUMINAMATH_CALUDE_expression_evaluation_l3621_362145

theorem expression_evaluation : (3 * 4 * 6) * (1/3 + 1/4 + 1/6) = 54 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3621_362145


namespace NUMINAMATH_CALUDE_function_properties_l3621_362182

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1)

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := a^(2*x) - 4*a^x + 8

-- State the theorem
theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 3 = 1/9) :
  a = 1/3 ∧ Set.Icc 4 53 = Set.image (g (1/3)) (Set.Icc (-2) 1) := by sorry

end

end NUMINAMATH_CALUDE_function_properties_l3621_362182


namespace NUMINAMATH_CALUDE_f_odd_and_periodic_l3621_362138

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom cond1 : ∀ x, f (10 + x) = f (10 - x)
axiom cond2 : ∀ x, f (20 - x) = -f (20 + x)

-- Define odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define periodic function
def is_periodic (f : ℝ → ℝ) : Prop := ∃ T > 0, ∀ x, f (x + T) = f x

-- Theorem statement
theorem f_odd_and_periodic : is_odd f ∧ is_periodic f := by sorry

end NUMINAMATH_CALUDE_f_odd_and_periodic_l3621_362138


namespace NUMINAMATH_CALUDE_evaluate_expression_l3621_362124

theorem evaluate_expression : -(18 / 3 * 8 - 72 + 4^2 * 3) = -24 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3621_362124


namespace NUMINAMATH_CALUDE_students_without_A_l3621_362126

theorem students_without_A (total : ℕ) (history_A : ℕ) (math_A : ℕ) (both_A : ℕ) :
  total = 30 →
  history_A = 7 →
  math_A = 13 →
  both_A = 4 →
  total - ((history_A + math_A) - both_A) = 14 :=
by sorry

end NUMINAMATH_CALUDE_students_without_A_l3621_362126


namespace NUMINAMATH_CALUDE_system_solution_implies_m_value_l3621_362179

theorem system_solution_implies_m_value (x y m : ℝ) : 
  (2 * x + y = 6 * m) →
  (3 * x - 2 * y = 2 * m) →
  (x / 3 - y / 5 = 4) →
  m = 15 := by
sorry

end NUMINAMATH_CALUDE_system_solution_implies_m_value_l3621_362179


namespace NUMINAMATH_CALUDE_not_both_bidirectional_l3621_362102

-- Define the two proof methods
inductive ProofMethod
| Synthetic
| Analytic

-- Define the reasoning direction
inductive ReasoningDirection
| CauseToEffect
| EffectToCause

-- Define the properties of the proof methods
def methodDirection (m : ProofMethod) : ReasoningDirection :=
  match m with
  | ProofMethod.Synthetic => ReasoningDirection.CauseToEffect
  | ProofMethod.Analytic => ReasoningDirection.EffectToCause

-- Theorem statement
theorem not_both_bidirectional :
  ¬(∀ (m : ProofMethod), 
    (methodDirection m = ReasoningDirection.CauseToEffect ∧
     methodDirection m = ReasoningDirection.EffectToCause)) :=
by
  sorry

end NUMINAMATH_CALUDE_not_both_bidirectional_l3621_362102


namespace NUMINAMATH_CALUDE_percent_of_itself_l3621_362123

theorem percent_of_itself (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 4) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_itself_l3621_362123


namespace NUMINAMATH_CALUDE_equal_digit_probability_l3621_362158

def num_dice : ℕ := 6
def sides_per_die : ℕ := 16
def one_digit_prob : ℚ := 9 / 16
def two_digit_prob : ℚ := 7 / 16

theorem equal_digit_probability : 
  (num_dice.choose (num_dice / 2)) * (one_digit_prob ^ (num_dice / 2)) * (two_digit_prob ^ (num_dice / 2)) = 3115125 / 10485760 := by
  sorry

end NUMINAMATH_CALUDE_equal_digit_probability_l3621_362158


namespace NUMINAMATH_CALUDE_find_F_when_C_is_35_l3621_362166

-- Define the relationship between C and F
def C_F_relation (C F : ℝ) : Prop := C = (4/7) * (F - 40)

-- State the theorem
theorem find_F_when_C_is_35 :
  ∃ F : ℝ, C_F_relation 35 F ∧ F = 101.25 := by
  sorry

end NUMINAMATH_CALUDE_find_F_when_C_is_35_l3621_362166


namespace NUMINAMATH_CALUDE_odd_function_negative_x_l3621_362180

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_x 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_positive : ∀ x > 0, f x = -x + 1) :
  ∀ x < 0, f x = -x - 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_x_l3621_362180


namespace NUMINAMATH_CALUDE_inequality_not_always_hold_l3621_362155

theorem inequality_not_always_hold (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 ∧ Real.sqrt a + Real.sqrt b > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_always_hold_l3621_362155


namespace NUMINAMATH_CALUDE_inscribed_circle_exists_l3621_362157

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a circle with center and radius -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Given configuration satisfies the problem conditions -/
def ValidConfiguration (rect : Rectangle) (circleA : Circle) (circleB : Circle) (circleC : Circle) (circleD : Circle) (e : ℝ) : Prop :=
  let a := circleA.radius
  let b := circleB.radius
  let c := circleC.radius
  let d := circleD.radius
  (a + c = b + d) ∧ (a + c < e) ∧
  (rect.A = circleA.center) ∧ (rect.B = circleB.center) ∧ (rect.C = circleC.center) ∧ (rect.D = circleD.center)

/-- Theorem: A circle can be inscribed in the quadrilateral formed by outer common tangents -/
theorem inscribed_circle_exists (rect : Rectangle) (circleA : Circle) (circleB : Circle) (circleC : Circle) (circleD : Circle) (e : ℝ) 
  (h : ValidConfiguration rect circleA circleB circleC circleD e) : 
  ∃ (inscribedCircle : Circle), true :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_exists_l3621_362157


namespace NUMINAMATH_CALUDE_total_colors_over_two_hours_l3621_362112

/-- Represents the number of colors the sky changes through in a 10-minute period -/
structure ColorChange where
  quick : ℕ
  slow : ℕ

/-- Calculates the total number of colors for one hour given a ColorChange pattern -/
def colorsPerHour (change : ColorChange) : ℕ :=
  (change.quick + change.slow) * 3

/-- The color change pattern for the first hour -/
def firstHourPattern : ColorChange :=
  { quick := 5, slow := 2 }

/-- The color change pattern for the second hour (doubled rate) -/
def secondHourPattern : ColorChange :=
  { quick := firstHourPattern.quick * 2, slow := firstHourPattern.slow * 2 }

/-- The main theorem stating the total number of colors over two hours -/
theorem total_colors_over_two_hours :
  colorsPerHour firstHourPattern + colorsPerHour secondHourPattern = 63 := by
  sorry


end NUMINAMATH_CALUDE_total_colors_over_two_hours_l3621_362112


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l3621_362176

theorem arithmetic_square_root_of_16 : ∃ (x : ℝ), x ≥ 0 ∧ x^2 = 16 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l3621_362176


namespace NUMINAMATH_CALUDE_percentage_with_no_conditions_is_10_percent_l3621_362105

-- Define the total number of teachers
def total_teachers : ℕ := 150

-- Define the number of teachers with each condition
def high_blood_pressure : ℕ := 90
def heart_trouble : ℕ := 60
def diabetes : ℕ := 30

-- Define the number of teachers with combinations of conditions
def high_blood_pressure_and_heart_trouble : ℕ := 25
def heart_trouble_and_diabetes : ℕ := 10
def high_blood_pressure_and_diabetes : ℕ := 15
def all_three_conditions : ℕ := 5

-- Define the function to calculate the percentage
def percentage_with_no_conditions : ℚ :=
  let teachers_with_conditions := high_blood_pressure + heart_trouble + diabetes
    - high_blood_pressure_and_heart_trouble - heart_trouble_and_diabetes - high_blood_pressure_and_diabetes
    + all_three_conditions
  let teachers_with_no_conditions := total_teachers - teachers_with_conditions
  (teachers_with_no_conditions : ℚ) / (total_teachers : ℚ) * 100

-- Theorem statement
theorem percentage_with_no_conditions_is_10_percent :
  percentage_with_no_conditions = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_with_no_conditions_is_10_percent_l3621_362105


namespace NUMINAMATH_CALUDE_interest_rate_is_four_percent_l3621_362140

/-- Given a loan with simple interest, prove that the interest rate is 4% per annum -/
theorem interest_rate_is_four_percent
  (P : ℚ) -- Principal amount
  (t : ℚ) -- Time in years
  (I : ℚ) -- Interest amount
  (h1 : P = 250) -- Sum lent is Rs. 250
  (h2 : t = 8) -- Time period is 8 years
  (h3 : I = P - 170) -- Interest is Rs. 170 less than sum lent
  (h4 : I = P * r * t / 100) -- Simple interest formula
  : r = 4 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_four_percent_l3621_362140


namespace NUMINAMATH_CALUDE_string_average_length_l3621_362194

theorem string_average_length : 
  let lengths : List ℚ := [2, 5, 7]
  (lengths.sum / lengths.length : ℚ) = 14 / 3 := by sorry

end NUMINAMATH_CALUDE_string_average_length_l3621_362194


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l3621_362114

theorem quadratic_solution_difference : 
  let f : ℝ → ℝ := λ x => x^2 + 5*x - 4 - (x + 66)
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = 2 * Real.sqrt 74 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l3621_362114


namespace NUMINAMATH_CALUDE_double_inequality_solution_l3621_362161

theorem double_inequality_solution (x : ℝ) : 
  -1 < (x^2 - 16*x + 24) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 16*x + 24) / (x^2 - 4*x + 8) < 1 ↔ 
  (3/2 < x ∧ x < 4) ∨ (8 < x) := by
  sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l3621_362161


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l3621_362116

-- Define the cubic polynomial
def q (x : ℚ) : ℚ := 7/4 * x^3 - 19 * x^2 + 149/4 * x + 6

-- Theorem statement
theorem cubic_polynomial_satisfies_conditions :
  q 1 = -6 ∧ q 3 = -20 ∧ q 4 = -42 ∧ q 5 = -60 := by
  sorry


end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l3621_362116


namespace NUMINAMATH_CALUDE_problem_statement_l3621_362184

open Real

theorem problem_statement (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ (x₀ - a)^2 + (log (x₀^2) - 2*a)^2 ≤ 4/5) → a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3621_362184


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l3621_362125

theorem opposite_of_negative_three :
  ∃ y : ℤ, ((-3 : ℤ) + y = 0) ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l3621_362125


namespace NUMINAMATH_CALUDE_f_injective_l3621_362101

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def injective {α β : Type} (f : α → β) : Prop :=
  ∀ a b : α, f a = f b → a = b

theorem f_injective (f : ℕ → ℕ) 
  (h : ∀ (x y : ℕ), is_perfect_square (f x + y) ↔ is_perfect_square (x + f y)) :
  injective f := by
sorry

end NUMINAMATH_CALUDE_f_injective_l3621_362101


namespace NUMINAMATH_CALUDE_sufficient_condition_l3621_362186

theorem sufficient_condition (a : ℝ) : a > 0 → a^2 + a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_l3621_362186


namespace NUMINAMATH_CALUDE_no_fixed_points_iff_a_in_open_interval_l3621_362147

theorem no_fixed_points_iff_a_in_open_interval
  (f : ℝ → ℝ)
  (h : ∀ x, f x = x^2 + a*x + 1) :
  (∀ x, f x ≠ x) ↔ a ∈ Set.Ioo (-1 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_no_fixed_points_iff_a_in_open_interval_l3621_362147


namespace NUMINAMATH_CALUDE_frosting_cans_needed_l3621_362159

def cakes_day1 : ℕ := 7
def cakes_day2 : ℕ := 12
def cakes_day3 : ℕ := 8
def cakes_day4 : ℕ := 10
def cakes_day5 : ℕ := 15
def cakes_eaten : ℕ := 18
def frosting_per_cake : ℕ := 3

def total_cakes : ℕ := cakes_day1 + cakes_day2 + cakes_day3 + cakes_day4 + cakes_day5
def remaining_cakes : ℕ := total_cakes - cakes_eaten

theorem frosting_cans_needed : remaining_cakes * frosting_per_cake = 102 := by
  sorry

end NUMINAMATH_CALUDE_frosting_cans_needed_l3621_362159


namespace NUMINAMATH_CALUDE_triangle_angle_not_all_less_than_60_l3621_362143

theorem triangle_angle_not_all_less_than_60 : 
  ¬ (∀ (a b c : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) → 
    (a + b + c = 180) → 
    (a < 60 ∧ b < 60 ∧ c < 60)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_not_all_less_than_60_l3621_362143


namespace NUMINAMATH_CALUDE_polyhedron_special_value_l3621_362103

/-- Represents a convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces
  T : ℕ  -- Number of triangular faces meeting at each vertex
  P : ℕ  -- Number of pentagonal faces meeting at each vertex
  euler_formula : V - E + F = 2
  face_count : F = 32
  vertex_face_relation : V * (P / 5 + T / 3 : ℚ) = 32

/-- Theorem stating the specific value of 100P + 10T + V for the given polyhedron -/
theorem polyhedron_special_value (poly : ConvexPolyhedron) : 
  100 * poly.P + 10 * poly.T + poly.V = 250 := by
  sorry


end NUMINAMATH_CALUDE_polyhedron_special_value_l3621_362103


namespace NUMINAMATH_CALUDE_ceiling_floor_calculation_l3621_362136

theorem ceiling_floor_calculation : 
  ⌈(18 : ℚ) / 5 * (-25 : ℚ) / 4⌉ - ⌊(18 : ℚ) / 5 * ⌊(-25 : ℚ) / 4⌋⌋ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_calculation_l3621_362136


namespace NUMINAMATH_CALUDE_first_group_weight_proof_l3621_362165

-- Define the number of girls in the second group
def second_group_count : ℕ := 8

-- Define the average weights
def first_group_avg : ℝ := 50.25
def second_group_avg : ℝ := 45.15
def total_avg : ℝ := 48.55

-- Define the theorem
theorem first_group_weight_proof :
  ∃ (first_group_count : ℕ),
    (first_group_count * first_group_avg + second_group_count * second_group_avg) / 
    (first_group_count + second_group_count) = total_avg →
    first_group_avg = 50.25 := by
  sorry


end NUMINAMATH_CALUDE_first_group_weight_proof_l3621_362165


namespace NUMINAMATH_CALUDE_delta_takes_five_hours_prove_delta_time_l3621_362106

/-- Represents the time taken by Delta and Epsilon to complete a landscaping job -/
structure LandscapingJob where
  delta : ℝ  -- Time taken by Delta alone
  epsilon : ℝ  -- Time taken by Epsilon alone
  together : ℝ  -- Time taken by both working together

/-- The conditions of the landscaping job -/
def job_conditions (job : LandscapingJob) : Prop :=
  job.together = job.delta - 3 ∧
  job.together = job.epsilon - 4 ∧
  job.together = 2

/-- The theorem stating that Delta takes 5 hours to complete the job alone -/
theorem delta_takes_five_hours (job : LandscapingJob) 
  (h : job_conditions job) : job.delta = 5 := by
  sorry

/-- The main theorem proving Delta's time based on the given conditions -/
theorem prove_delta_time : ∃ (job : LandscapingJob), job_conditions job ∧ job.delta = 5 := by
  sorry

end NUMINAMATH_CALUDE_delta_takes_five_hours_prove_delta_time_l3621_362106


namespace NUMINAMATH_CALUDE_homework_problems_l3621_362162

theorem homework_problems (t p : ℕ) (ht : t > 0) (hp : p > 10) : 
  (∀ (t' : ℕ), t' > 0 → p * t = (2 * p - 2) * (t' - 1) → t' = t) →
  p * t = 48 := by
sorry

end NUMINAMATH_CALUDE_homework_problems_l3621_362162


namespace NUMINAMATH_CALUDE_line_through_vectors_l3621_362107

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem line_through_vectors (a b : V) (k : ℝ) (h : a ≠ b) :
  (∃ t : ℝ, k • a + (2/3 : ℝ) • b = a + t • (b - a)) →
  k = 1/3 := by
sorry

end NUMINAMATH_CALUDE_line_through_vectors_l3621_362107


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_l3621_362141

/-- A triangular pyramid with specific properties -/
structure SpecialPyramid where
  -- Base side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Condition: base side lengths are √85, √58, and √45
  ha : a = Real.sqrt 85
  hb : b = Real.sqrt 58
  hc : c = Real.sqrt 45
  -- Condition: lateral edges are mutually perpendicular
  lateral_edges_perpendicular : Bool

/-- The sphere inscribed in the special pyramid -/
structure InscribedSphere (p : SpecialPyramid) where
  -- The radius of the sphere
  radius : ℝ
  -- Condition: The sphere touches all lateral faces
  touches_all_faces : Bool
  -- Condition: The center of the sphere lies on the base
  center_on_base : Bool

/-- The main theorem stating the radius of the inscribed sphere -/
theorem inscribed_sphere_radius (p : SpecialPyramid) (s : InscribedSphere p) :
  s.touches_all_faces ∧ s.center_on_base ∧ p.lateral_edges_perpendicular →
  s.radius = 14 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_l3621_362141


namespace NUMINAMATH_CALUDE_triangle_properties_l3621_362148

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.b * Real.sin abc.A = Real.sqrt 3 * abc.a * Real.cos abc.B)
  (h2 : abc.b = 3)
  (h3 : Real.sin abc.C = 2 * Real.sin abc.A) :
  abc.B = π / 3 ∧ 
  abc.a = Real.sqrt 3 ∧ 
  abc.c = 2 * Real.sqrt 3 ∧ 
  (1 / 2 : ℝ) * abc.a * abc.c * Real.sin abc.B = 3 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3621_362148


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3621_362154

theorem complex_fraction_equality : Complex.I * Complex.I = -1 → (3 : ℂ) / (1 - Complex.I)^2 = (3 / 2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3621_362154


namespace NUMINAMATH_CALUDE_cosine_triangle_condition_l3621_362118

/-- A cubic equation with real coefficients -/
structure CubicEquation where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The roots of a cubic equation are all real and positive -/
def has_real_positive_roots (eq : CubicEquation) : Prop := sorry

/-- The roots of a cubic equation are the cosines of the angles of a triangle -/
def roots_are_triangle_cosines (eq : CubicEquation) : Prop := sorry

/-- The necessary and sufficient condition for the roots to be the cosines of the angles of a triangle -/
theorem cosine_triangle_condition (eq : CubicEquation) :
  roots_are_triangle_cosines eq ↔
    eq.p^2 - 2*eq.q - 2*eq.r - 1 = 0 ∧ eq.p < 0 ∧ eq.q > 0 ∧ eq.r < 0 :=
by sorry

end NUMINAMATH_CALUDE_cosine_triangle_condition_l3621_362118


namespace NUMINAMATH_CALUDE_solution_satisfies_conditions_l3621_362150

/-- Represents the number of bicycles, cars, and carts passing in front of a house. -/
structure VehicleCount where
  bicycles : ℕ
  cars : ℕ
  carts : ℕ

/-- Checks if the given vehicle count satisfies the problem conditions. -/
def satisfiesConditions (vc : VehicleCount) : Prop :=
  (vc.bicycles + vc.cars = 3 * vc.carts) ∧
  (2 * vc.bicycles + 4 * vc.cars + vc.bicycles + vc.cars = 100)

/-- The solution to the vehicle counting problem. -/
def solution : VehicleCount :=
  { bicycles := 10, cars := 14, carts := 8 }

/-- Theorem stating that the given solution satisfies the problem conditions. -/
theorem solution_satisfies_conditions : satisfiesConditions solution := by
  sorry


end NUMINAMATH_CALUDE_solution_satisfies_conditions_l3621_362150


namespace NUMINAMATH_CALUDE_specific_normal_distribution_mean_l3621_362198

/-- A normal distribution with given properties -/
structure NormalDistribution where
  μ : ℝ  -- arithmetic mean
  σ : ℝ  -- standard deviation
  value_2sd_below : ℝ  -- value 2 standard deviations below the mean

/-- Theorem stating the properties of the specific normal distribution -/
theorem specific_normal_distribution_mean 
  (d : NormalDistribution) 
  (h1 : d.σ = 2.3)
  (h2 : d.value_2sd_below = 11.6)
  (h3 : d.value_2sd_below = d.μ - 2 * d.σ) : 
  d.μ = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_specific_normal_distribution_mean_l3621_362198


namespace NUMINAMATH_CALUDE_min_rubles_to_win_l3621_362177

/-- Represents the state of the game machine --/
structure GameState :=
  (points : ℕ)
  (rubles : ℕ)

/-- Defines the possible moves in the game --/
inductive Move
  | AddOne
  | Double

/-- Applies a move to the current game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.AddOne => { points := state.points + 1, rubles := state.rubles + 1 }
  | Move.Double => { points := state.points * 2, rubles := state.rubles + 2 }

/-- Checks if the given state is valid (not exceeding 50 points) --/
def isValidState (state : GameState) : Prop :=
  state.points ≤ 50

/-- Checks if the given state is a winning state (exactly 50 points) --/
def isWinningState (state : GameState) : Prop :=
  state.points = 50

/-- Theorem stating that 11 rubles is the minimum amount needed to win --/
theorem min_rubles_to_win :
  ∃ (moves : List Move),
    let finalState := moves.foldl applyMove { points := 0, rubles := 0 }
    isValidState finalState ∧
    isWinningState finalState ∧
    finalState.rubles = 11 ∧
    (∀ (otherMoves : List Move),
      let otherFinalState := otherMoves.foldl applyMove { points := 0, rubles := 0 }
      isValidState otherFinalState → isWinningState otherFinalState →
      otherFinalState.rubles ≥ 11) :=
by sorry


end NUMINAMATH_CALUDE_min_rubles_to_win_l3621_362177


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_with_diff_217_l3621_362189

theorem smallest_sum_of_squares_with_diff_217 :
  ∃ (x y : ℕ), 
    x^2 - y^2 = 217 ∧
    ∀ (a b : ℕ), a^2 - b^2 = 217 → x^2 + y^2 ≤ a^2 + b^2 ∧
    x^2 + y^2 = 505 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_with_diff_217_l3621_362189


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l3621_362121

def p (x : ℝ) : ℝ := x^5 - 2*x^4 + 4*x^3 - 5*x + 2
def q (x : ℝ) : ℝ := 3*x^4 - x^3 + x^2 + 4*x - 1

theorem coefficient_of_x_squared :
  ∃ (a b c d e f : ℝ),
    p x * q x = a*x^9 + b*x^8 + c*x^7 + d*x^6 + e*x^5 + (-18)*x^2 + f :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l3621_362121


namespace NUMINAMATH_CALUDE_pond_water_theorem_l3621_362172

/-- Calculates the amount of water remaining in a pond after a certain number of days,
    given initial water amount, evaporation rate, and rain addition rate. -/
def water_remaining (initial_water : ℝ) (evaporation_rate : ℝ) (rain_rate : ℝ) (days : ℕ) : ℝ :=
  initial_water - (evaporation_rate - rain_rate) * days

theorem pond_water_theorem (initial_water : ℝ) (evaporation_rate : ℝ) (rain_rate : ℝ) (days : ℕ) :
  initial_water = 500 ∧ evaporation_rate = 4 ∧ rain_rate = 2 ∧ days = 40 →
  water_remaining initial_water evaporation_rate rain_rate days = 420 := by
  sorry

#eval water_remaining 500 4 2 40

end NUMINAMATH_CALUDE_pond_water_theorem_l3621_362172


namespace NUMINAMATH_CALUDE_monic_cubic_polynomial_uniqueness_l3621_362100

/-- A monic cubic polynomial with real coefficients -/
def MonicCubicPolynomial (a b c : ℝ) : ℝ → ℂ :=
  fun x => (x : ℂ)^3 + a * (x : ℂ)^2 + b * (x : ℂ) + c

theorem monic_cubic_polynomial_uniqueness (a b c : ℝ) :
  let p := MonicCubicPolynomial a b c
  p (1 + 3*I) = 0 ∧ p 0 = -108 →
  a = -12.8 ∧ b = 31 ∧ c = -108 := by
  sorry

end NUMINAMATH_CALUDE_monic_cubic_polynomial_uniqueness_l3621_362100


namespace NUMINAMATH_CALUDE_jacob_number_problem_l3621_362120

theorem jacob_number_problem : 
  ∃! x : ℕ, 10 ≤ x ∧ x < 100 ∧ 
  (∃ a b : ℕ, 4 * x - 8 = 10 * a + b ∧ 
              25 ≤ 10 * b + a ∧ 10 * b + a ≤ 30) ∧
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_jacob_number_problem_l3621_362120


namespace NUMINAMATH_CALUDE_new_average_income_after_death_l3621_362181

/-- Calculates the new average income after a member's death -/
def new_average_income (original_members : ℕ) (original_average : ℚ) (deceased_income : ℚ) : ℚ :=
  (original_members * original_average - deceased_income) / (original_members - 1)

/-- Theorem: The new average income after a member's death is 650 -/
theorem new_average_income_after_death :
  new_average_income 4 782 1178 = 650 := by
  sorry

end NUMINAMATH_CALUDE_new_average_income_after_death_l3621_362181


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_300_l3621_362131

def sum_of_divisors (n : ℕ) : ℕ := sorry

def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_300 :
  largest_prime_factor (sum_of_divisors 300) = 31 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_300_l3621_362131


namespace NUMINAMATH_CALUDE_car_pedestrian_speed_ratio_l3621_362133

/-- The ratio of car speed to pedestrian speed on a bridge -/
theorem car_pedestrian_speed_ratio :
  ∀ (L : ℝ) (vp vc : ℝ),
  L > 0 →  -- The bridge has positive length
  vp > 0 →  -- The pedestrian's speed is positive
  vc > 0 →  -- The car's speed is positive
  (2 / 5 * L) / vp = L / vc →  -- Time for pedestrian to return equals time for car to reach start
  (3 / 5 * L) / vp = L / vc →  -- Time for pedestrian to finish equals time for car to finish
  vc / vp = 5 := by
sorry

end NUMINAMATH_CALUDE_car_pedestrian_speed_ratio_l3621_362133


namespace NUMINAMATH_CALUDE_adjacent_pair_arrangements_l3621_362135

def number_of_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial n

theorem adjacent_pair_arrangements :
  let total_people : ℕ := 5
  let adjacent_pair : ℕ := 2
  let remaining_people : ℕ := total_people - adjacent_pair + 1
  number_of_arrangements remaining_people remaining_people = 24 :=
by sorry

end NUMINAMATH_CALUDE_adjacent_pair_arrangements_l3621_362135


namespace NUMINAMATH_CALUDE_prob_select_seventh_grade_prob_select_one_from_each_grade_l3621_362160

structure School :=
  (seventh_grade : Finset Nat)
  (eighth_grade : Finset Nat)
  (h1 : seventh_grade.card = 2)
  (h2 : eighth_grade.card = 2)
  (h3 : seventh_grade ∩ eighth_grade = ∅)

def total_students (s : School) : Finset Nat :=
  s.seventh_grade ∪ s.eighth_grade

theorem prob_select_seventh_grade (s : School) :
  (s.seventh_grade.card : ℚ) / (total_students s).card = 1 / 2 := by sorry

theorem prob_select_one_from_each_grade (s : School) :
  let total_pairs := (total_students s).card.choose 2
  let mixed_pairs := s.seventh_grade.card * s.eighth_grade.card * 2
  (mixed_pairs : ℚ) / total_pairs = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_select_seventh_grade_prob_select_one_from_each_grade_l3621_362160


namespace NUMINAMATH_CALUDE_town_hall_eggs_l3621_362111

/-- The number of eggs Joe found in different locations --/
structure EggCount where
  clubHouse : ℕ
  park : ℕ
  townHall : ℕ
  total : ℕ

/-- Theorem stating that Joe found 3 eggs in the town hall garden --/
theorem town_hall_eggs (e : EggCount) 
  (h1 : e.clubHouse = 12)
  (h2 : e.park = 5)
  (h3 : e.total = 20)
  (h4 : e.total = e.clubHouse + e.park + e.townHall) :
  e.townHall = 3 := by
  sorry

#check town_hall_eggs

end NUMINAMATH_CALUDE_town_hall_eggs_l3621_362111


namespace NUMINAMATH_CALUDE_molecular_weight_K3AlC2O4_3_l3621_362199

-- Define atomic weights
def atomic_weight_K : ℝ := 39.10
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms in the compound
def num_K : ℕ := 3
def num_Al : ℕ := 1
def num_C : ℕ := 6  -- 3 oxalate ions * 2 C atoms per oxalate
def num_O : ℕ := 12 -- 3 oxalate ions * 4 O atoms per oxalate

-- Theorem statement
theorem molecular_weight_K3AlC2O4_3 :
  num_K * atomic_weight_K +
  num_Al * atomic_weight_Al +
  num_C * atomic_weight_C +
  num_O * atomic_weight_O = 408.34 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_K3AlC2O4_3_l3621_362199


namespace NUMINAMATH_CALUDE_mans_walking_speed_l3621_362108

-- Define the given conditions
def walking_time : ℝ := 8
def running_time : ℝ := 2
def running_speed : ℝ := 36

-- Define the walking speed as a variable
def walking_speed : ℝ := sorry

-- Theorem statement
theorem mans_walking_speed :
  walking_speed * walking_time = running_speed * running_time →
  walking_speed = 9 := by
  sorry

end NUMINAMATH_CALUDE_mans_walking_speed_l3621_362108


namespace NUMINAMATH_CALUDE_cos_sum_of_complex_on_unit_circle_l3621_362110

/-- Given complex numbers on the unit circle represented by their real and imaginary parts,
    prove that the cosine of the sum of their arguments is as specified. -/
theorem cos_sum_of_complex_on_unit_circle
  (γ δ : ℝ)
  (h1 : Complex.exp (Complex.I * γ) = Complex.ofReal (8/17) + Complex.I * (15/17))
  (h2 : Complex.exp (Complex.I * δ) = Complex.ofReal (3/5) - Complex.I * (4/5)) :
  Real.cos (γ + δ) = 84/85 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_of_complex_on_unit_circle_l3621_362110


namespace NUMINAMATH_CALUDE_sandhill_football_club_members_l3621_362104

/-- Represents the Sandhill Football Club problem --/
theorem sandhill_football_club_members :
  let sock_cost : ℕ := 5
  let tshirt_cost : ℕ := sock_cost + 6
  let home_game_socks : ℕ := 1
  let home_game_tshirts : ℕ := 1
  let away_game_socks : ℕ := 2
  let away_game_tshirts : ℕ := 1
  let total_expenditure : ℕ := 4150
  let member_cost : ℕ := 
    (home_game_socks + away_game_socks) * sock_cost + 
    (home_game_tshirts + away_game_tshirts) * tshirt_cost
  let number_of_members : ℕ := total_expenditure / member_cost
  number_of_members = 112 :=
by
  sorry


end NUMINAMATH_CALUDE_sandhill_football_club_members_l3621_362104


namespace NUMINAMATH_CALUDE_exercise_minutes_proof_l3621_362149

/-- The number of minutes Javier exercised daily -/
def javier_daily_minutes : ℕ := 50

/-- The number of days Javier exercised in a week -/
def javier_days : ℕ := 7

/-- The number of minutes Sanda exercised on each day she exercised -/
def sanda_daily_minutes : ℕ := 90

/-- The number of days Sanda exercised -/
def sanda_days : ℕ := 3

/-- The total number of minutes Javier and Sanda exercised -/
def total_exercise_minutes : ℕ := javier_daily_minutes * javier_days + sanda_daily_minutes * sanda_days

theorem exercise_minutes_proof : total_exercise_minutes = 620 := by
  sorry

end NUMINAMATH_CALUDE_exercise_minutes_proof_l3621_362149


namespace NUMINAMATH_CALUDE_half_job_days_l3621_362129

/-- 
Proves that the number of days to complete half a job is 6, 
given that it takes 6 more days to finish the entire job after completing half of it.
-/
theorem half_job_days : 
  ∀ (x : ℝ), (x + 6 = 2*x) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_half_job_days_l3621_362129


namespace NUMINAMATH_CALUDE_triangle_angle_A_l3621_362137

theorem triangle_angle_A (a b : ℝ) (A B : ℝ) :
  a = Real.sqrt 2 →
  b = 2 →
  Real.sin B + Real.cos B = Real.sqrt 2 →
  0 < A →
  A < π / 4 →
  0 < B →
  B < π →
  Real.sin A = a / b * Real.sin B →
  A = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l3621_362137


namespace NUMINAMATH_CALUDE_square_remainder_mod_five_l3621_362109

theorem square_remainder_mod_five (n : ℤ) (h : n % 5 = 3) : n^2 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_remainder_mod_five_l3621_362109


namespace NUMINAMATH_CALUDE_ratio_problem_l3621_362144

theorem ratio_problem (first_number second_number : ℚ) : 
  (first_number / second_number = 15) → 
  (first_number = 150) → 
  (second_number = 10) := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3621_362144


namespace NUMINAMATH_CALUDE_problem_solution_l3621_362142

theorem problem_solution : 2.017 * 2016 - 10.16 * 201.7 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3621_362142


namespace NUMINAMATH_CALUDE_angle_with_complement_quarter_supplement_l3621_362164

theorem angle_with_complement_quarter_supplement (x : ℝ) :
  (90 - x = (1 / 4) * (180 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_complement_quarter_supplement_l3621_362164


namespace NUMINAMATH_CALUDE_exponential_inequality_minimum_value_of_f_l3621_362183

-- Proposition 2
theorem exponential_inequality (x₁ x₂ : ℝ) :
  Real.exp ((x₁ + x₂) / 2) ≤ (Real.exp x₁ + Real.exp x₂) / 2 := by
  sorry

-- Proposition 4
def f (x : ℝ) : ℝ := (x - 2014)^2 - 2

theorem minimum_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = -2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_minimum_value_of_f_l3621_362183


namespace NUMINAMATH_CALUDE_ratio_of_x_to_y_l3621_362188

theorem ratio_of_x_to_y (x y : ℝ) (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0) :
  (1/5 * x) / (1/6 * y) = 18/25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_x_to_y_l3621_362188


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l3621_362119

/-- Given a line with equation y - 6 = -3(x - 5), 
    the sum of its x-intercept and y-intercept is 28 -/
theorem line_intercepts_sum : 
  ∀ (x y : ℝ), y - 6 = -3 * (x - 5) → 
  ∃ (x_int y_int : ℝ),
    (y_int - 6 = -3 * (x_int - 5) ∧ y_int = 0) ∧
    (0 - 6 = -3 * (0 - 5) ∧ y = y_int) ∧
    x_int + y_int = 28 :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l3621_362119


namespace NUMINAMATH_CALUDE_quadratic_root_fraction_l3621_362132

theorem quadratic_root_fraction (a b : ℝ) (h1 : a ≠ b) (h2 : a + b - 20 = 0) :
  (a^2 - b^2) / (2*a - 2*b) = 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_fraction_l3621_362132


namespace NUMINAMATH_CALUDE_garden_area_l3621_362185

theorem garden_area (length width : ℝ) (h1 : length = 350) (h2 : width = 50) :
  (length * width) / 10000 = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l3621_362185


namespace NUMINAMATH_CALUDE_line_through_center_and_perpendicular_l3621_362127

/-- The circle C in the xy-plane -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0

/-- The given line L1 -/
def line_L1 (x y : ℝ) : Prop := 3*x + y - 2 = 0

/-- The line to be proved L2 -/
def line_L2 (x y : ℝ) : Prop := x - 3*y + 1 = 0

/-- The center of a circle given by its equation -/
def circle_center (f : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (f g : ℝ → ℝ → Prop) : Prop := sorry

theorem line_through_center_and_perpendicular :
  let center := circle_center circle_C
  line_L2 center.1 center.2 ∧ perpendicular line_L1 line_L2 := by sorry

end NUMINAMATH_CALUDE_line_through_center_and_perpendicular_l3621_362127


namespace NUMINAMATH_CALUDE_arrangement_counts_l3621_362171

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- Calculates the number of permutations of n elements taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of arrangements where the three girls must stand together -/
def arrangements_girls_together : ℕ := 
  permutations num_girls num_girls * permutations (num_boys + 1) (num_boys + 1)

/-- The number of arrangements where no two girls are next to each other -/
def arrangements_girls_apart : ℕ := 
  permutations num_boys num_boys * permutations (num_boys + 1) num_girls

/-- The number of arrangements where there are exactly three people between person A and person B -/
def arrangements_three_between : ℕ := 
  permutations 2 2 * permutations (total_people - 2) 3 * permutations 5 5

/-- The number of arrangements where persons A and B are adjacent, but neither is next to person C -/
def arrangements_ab_adjacent_not_c : ℕ := 
  permutations 2 2 * permutations (total_people - 3) (total_people - 3) * permutations 5 2

theorem arrangement_counts :
  arrangements_girls_together = 720 ∧
  arrangements_girls_apart = 1440 ∧
  arrangements_three_between = 720 ∧
  arrangements_ab_adjacent_not_c = 960 := by sorry

end NUMINAMATH_CALUDE_arrangement_counts_l3621_362171


namespace NUMINAMATH_CALUDE_problem_solution_l3621_362195

theorem problem_solution (y : ℝ) (h1 : y > 0) (h2 : y * y / 100 = 9) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3621_362195


namespace NUMINAMATH_CALUDE_reading_time_difference_l3621_362167

theorem reading_time_difference 
  (xanthia_speed : ℝ) 
  (molly_speed : ℝ) 
  (book_pages : ℝ) 
  (h1 : xanthia_speed = 120) 
  (h2 : molly_speed = 60) 
  (h3 : book_pages = 300) : 
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 150 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_l3621_362167


namespace NUMINAMATH_CALUDE_min_value_x_plus_3y_l3621_362170

theorem min_value_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + 4 * y = x * y) :
  x + 3 * y ≥ 25 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 3 * x + 4 * y = x * y ∧ x + 3 * y = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_3y_l3621_362170


namespace NUMINAMATH_CALUDE_triangle_angle_sum_identity_l3621_362173

theorem triangle_angle_sum_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = 
  -4 * Real.cos (3/2 * A) * Real.cos (3/2 * B) * Real.cos (3/2 * C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_identity_l3621_362173


namespace NUMINAMATH_CALUDE_intersection_implies_k_zero_l3621_362174

/-- Line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem intersection_implies_k_zero (m n : Line) (h1 : m.slope = 3) (h2 : m.intercept = 5)
    (h3 : n.intercept = -7) (h4 : m.equation (-4) (-7)) (h5 : n.equation (-4) (-7)) :
    n.slope = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_k_zero_l3621_362174


namespace NUMINAMATH_CALUDE_birds_in_cage_l3621_362134

theorem birds_in_cage (birds_taken_out birds_left : ℕ) 
  (h1 : birds_taken_out = 10)
  (h2 : birds_left = 9) : 
  birds_taken_out + birds_left = 19 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_cage_l3621_362134


namespace NUMINAMATH_CALUDE_complex_roots_of_unity_real_sixth_power_l3621_362113

theorem complex_roots_of_unity_real_sixth_power :
  ∃! (S : Finset ℂ), 
    (∀ z ∈ S, z^24 = 1 ∧ (∃ r : ℝ, z^6 = r)) ∧ 
    Finset.card S = 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_of_unity_real_sixth_power_l3621_362113


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3621_362156

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 3*x + 1) * (y^2 + 3*y + 1) * (z^2 + 3*z + 1) / (x*y*z) ≥ 125 :=
by sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x^2 + 3*x + 1) * (y^2 + 3*y + 1) * (z^2 + 3*z + 1) / (x*y*z) = 125 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3621_362156


namespace NUMINAMATH_CALUDE_two_propositions_correct_l3621_362196

-- Define the original proposition
def original (x : ℝ) : Prop := x = 3 → x^2 - 7*x + 12 = 0

-- Define the converse proposition
def converse (x : ℝ) : Prop := x^2 - 7*x + 12 = 0 → x = 3

-- Define the inverse proposition
def inverse (x : ℝ) : Prop := x ≠ 3 → x^2 - 7*x + 12 ≠ 0

-- Define the contrapositive proposition
def contrapositive (x : ℝ) : Prop := x^2 - 7*x + 12 ≠ 0 → x ≠ 3

-- Theorem stating that exactly two propositions are correct
theorem two_propositions_correct :
  (∃! (n : ℕ), n = 2 ∧
    (∀ (x : ℝ), original x) ∧
    (∀ (x : ℝ), contrapositive x) ∧
    ¬(∀ (x : ℝ), converse x) ∧
    ¬(∀ (x : ℝ), inverse x)) :=
sorry

end NUMINAMATH_CALUDE_two_propositions_correct_l3621_362196


namespace NUMINAMATH_CALUDE_ellipse_parameter_sum_l3621_362152

def ellipse (h k a b : ℝ) := fun (x y : ℝ) ↦ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

theorem ellipse_parameter_sum :
  ∃ h k a b : ℝ,
    (∀ x y : ℝ, ellipse h k a b x y ↔ 
      Real.sqrt ((x - 0)^2 + (y - 0)^2) + Real.sqrt ((x - 6)^2 + (y - 0)^2) = 10) ∧
    h + k + a + b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parameter_sum_l3621_362152


namespace NUMINAMATH_CALUDE_larger_quadrilateral_cyclic_l3621_362117

/-- A configuration of five cyclic quadrilaterals forming a larger quadrilateral -/
structure FiveQuadrilateralsConfig where
  /-- The angles of the five smaller quadrilaterals -/
  angles : Fin 10 → ℝ
  /-- Each smaller quadrilateral is cyclic -/
  cyclic_small : ∀ i : Fin 5, angles (2*i) + angles (2*i+1) = 180
  /-- The sum of angles around each internal vertex is 360° -/
  vertex_sum : angles 1 + angles 7 + angles 8 = 360 ∧ angles 3 + angles 5 + angles 9 = 360

/-- The theorem stating that the larger quadrilateral is cyclic -/
theorem larger_quadrilateral_cyclic (config : FiveQuadrilateralsConfig) :
  config.angles 0 + config.angles 2 + config.angles 4 + config.angles 6 = 180 :=
sorry

end NUMINAMATH_CALUDE_larger_quadrilateral_cyclic_l3621_362117


namespace NUMINAMATH_CALUDE_m_range_l3621_362187

def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*(m+1)*x + m*(m+1) > 0

theorem m_range (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) :
  m > 2 ∨ (-2 ≤ m ∧ m < -1) :=
sorry

end NUMINAMATH_CALUDE_m_range_l3621_362187


namespace NUMINAMATH_CALUDE_smallest_n_with_abc_property_l3621_362175

def has_abc_property (n : ℕ) : Prop :=
  ∀ (A B : Set ℕ), A ∪ B = Finset.range (n + 1) → A ∩ B = ∅ →
    (∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b = c) ∨
    (∃ (a b c : ℕ), a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b = c)

theorem smallest_n_with_abc_property :
  (∀ m < 96, ¬(has_abc_property m)) ∧ has_abc_property 96 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_abc_property_l3621_362175


namespace NUMINAMATH_CALUDE_abs_neg_five_eq_five_l3621_362163

theorem abs_neg_five_eq_five : abs (-5 : ℤ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_eq_five_l3621_362163


namespace NUMINAMATH_CALUDE_solution_approximation_l3621_362169

/-- The solution to the equation (0.625 * 0.0729 * 28.9) / (x * 0.025 * 8.1) = 382.5 is approximately 0.01689 -/
theorem solution_approximation : ∃ x : ℝ, 
  (0.625 * 0.0729 * 28.9) / (x * 0.025 * 8.1) = 382.5 ∧ 
  abs (x - 0.01689) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_solution_approximation_l3621_362169


namespace NUMINAMATH_CALUDE_no_solution_inequality_l3621_362139

theorem no_solution_inequality (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 5| + |x + 3| < a)) → a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_inequality_l3621_362139


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3621_362190

theorem right_triangle_segment_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  r * s = c^2 →      -- Geometric mean theorem
  r * c = a^2 →      -- Geometric mean theorem
  s * c = b^2 →      -- Geometric mean theorem
  a / b = 2 / 5 →    -- Given ratio of legs
  r / s = 4 / 25 :=  -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3621_362190


namespace NUMINAMATH_CALUDE_linear_term_coefficient_l3621_362128

/-- The coefficient of the linear term in the polynomial x^2 - 2x - 3 is -2 -/
theorem linear_term_coefficient (x : ℝ) : ∃ a b c : ℝ, 
  x^2 - 2*x - 3 = a*x^2 + b*x + c ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_linear_term_coefficient_l3621_362128


namespace NUMINAMATH_CALUDE_function_value_at_three_l3621_362193

/-- Given a positive real number and a function satisfying certain conditions,
    prove that the function evaluated at 3 equals 1/3. -/
theorem function_value_at_three (x : ℝ) (f : ℝ → ℝ) 
    (h1 : x > 0)
    (h2 : x + 17 = 60 * f x)
    (h3 : x = 3) : 
  f 3 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_three_l3621_362193


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3621_362151

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 + a 3 = 1) →
  (a 2 + a 3 + a 4 = 2) →
  (a 5 + a 6 + a 7 = 16) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3621_362151


namespace NUMINAMATH_CALUDE_max_disks_in_rectangle_l3621_362122

/-- The maximum number of circular disks that can be cut from a rectangular sheet. -/
def max_disks (rect_width rect_height disk_diameter : ℝ) : ℕ :=
  32

/-- Theorem stating the maximum number of 5 cm diameter circular disks 
    that can be cut from a 9 × 100 cm rectangular sheet. -/
theorem max_disks_in_rectangle : 
  max_disks 9 100 5 = 32 := by sorry

end NUMINAMATH_CALUDE_max_disks_in_rectangle_l3621_362122


namespace NUMINAMATH_CALUDE_crackers_distribution_l3621_362191

/-- The number of crackers Matthew had initially -/
def total_crackers : ℕ := 32

/-- The number of friends Matthew gave crackers to -/
def num_friends : ℕ := 4

/-- The number of crackers each friend received -/
def crackers_per_friend : ℕ := total_crackers / num_friends

theorem crackers_distribution :
  crackers_per_friend = 8 := by sorry

end NUMINAMATH_CALUDE_crackers_distribution_l3621_362191


namespace NUMINAMATH_CALUDE_square_perimeter_l3621_362197

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 625) (h2 : side * side = area) :
  4 * side = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3621_362197


namespace NUMINAMATH_CALUDE_division_theorem_l3621_362168

theorem division_theorem (M q : ℤ) (h : M = 54 * q + 37) :
  ∃ (k : ℤ), M = 18 * k + 1 ∧ k = 3 * q + 2 := by
  sorry

end NUMINAMATH_CALUDE_division_theorem_l3621_362168
