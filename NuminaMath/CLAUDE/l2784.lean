import Mathlib

namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l2784_278428

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def circle1 : Circle := ⟨(3, 0), 3⟩
def circle2 : Circle := ⟨(7, 0), 2⟩
def circle3 : Circle := ⟨(11, 0), 1⟩

-- Define the tangent line
structure TangentLine where
  slope : ℝ
  yIntercept : ℝ

-- Function to check if a line is tangent to a circle
def isTangent (l : TangentLine) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  let r := c.radius
  let m := l.slope
  let b := l.yIntercept
  (y₀ - m * x₀ - b)^2 = (m^2 + 1) * r^2

-- Theorem statement
theorem tangent_line_y_intercept :
  ∃ l : TangentLine,
    isTangent l circle1 ∧
    isTangent l circle2 ∧
    isTangent l circle3 ∧
    l.yIntercept = 36 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l2784_278428


namespace NUMINAMATH_CALUDE_average_mpg_calculation_l2784_278423

theorem average_mpg_calculation (initial_reading final_reading : ℕ) (fuel_used : ℕ) :
  initial_reading = 56200 →
  final_reading = 57150 →
  fuel_used = 50 →
  (final_reading - initial_reading : ℚ) / fuel_used = 19 := by
  sorry

end NUMINAMATH_CALUDE_average_mpg_calculation_l2784_278423


namespace NUMINAMATH_CALUDE_probability_positive_sum_is_one_third_l2784_278440

/-- The set of card values in the bag -/
def card_values : Finset Int := {-2, -1, 2}

/-- The sample space of all possible outcomes when drawing two cards with replacement -/
def sample_space : Finset (Int × Int) :=
  card_values.product card_values

/-- The set of favorable outcomes (sum of drawn cards is positive) -/
def favorable_outcomes : Finset (Int × Int) :=
  sample_space.filter (fun p => p.1 + p.2 > 0)

/-- The probability of drawing two cards with a positive sum -/
def probability_positive_sum : ℚ :=
  (favorable_outcomes.card : ℚ) / (sample_space.card : ℚ)

theorem probability_positive_sum_is_one_third :
  probability_positive_sum = 1/3 := by sorry

end NUMINAMATH_CALUDE_probability_positive_sum_is_one_third_l2784_278440


namespace NUMINAMATH_CALUDE_intersection_count_l2784_278427

-- Define the two curves
def curve1 (x y : ℝ) : Prop := 3 * x^2 + 2 * y^2 = 6
def curve2 (x y : ℝ) : Prop := x^2 - 2 * y^2 = 1

-- Define an intersection point
def is_intersection_point (x y : ℝ) : Prop :=
  curve1 x y ∧ curve2 x y

-- Define a function to count distinct intersection points
def count_distinct_intersections : ℕ :=
  -- Implementation details omitted
  sorry

-- Theorem statement
theorem intersection_count :
  count_distinct_intersections = 4 :=
sorry

end NUMINAMATH_CALUDE_intersection_count_l2784_278427


namespace NUMINAMATH_CALUDE_cyclist_motorcyclist_speed_l2784_278431

theorem cyclist_motorcyclist_speed : ∀ (motorcyclist_speed : ℝ) (cyclist_speed : ℝ),
  motorcyclist_speed > 0 ∧
  cyclist_speed > 0 ∧
  cyclist_speed = motorcyclist_speed - 30 ∧
  120 / motorcyclist_speed + 2 = 120 / cyclist_speed →
  motorcyclist_speed = 60 ∧ cyclist_speed = 30 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_motorcyclist_speed_l2784_278431


namespace NUMINAMATH_CALUDE_caterpillar_climb_days_l2784_278473

/-- The number of days it takes for a caterpillar to climb a pole -/
def climbingDays (poleHeight : ℕ) (dayClimb : ℕ) (nightSlide : ℕ) : ℕ :=
  let netClimbPerDay := dayClimb - nightSlide
  let daysToAlmostTop := (poleHeight - dayClimb) / netClimbPerDay
  daysToAlmostTop + 1

/-- Theorem stating that it takes 16 days for the caterpillar to reach the top -/
theorem caterpillar_climb_days :
  climbingDays 20 5 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_caterpillar_climb_days_l2784_278473


namespace NUMINAMATH_CALUDE_triangle_problem_l2784_278477

theorem triangle_problem (A B C : ℝ) (AC BC : ℝ) (h1 : AC = 2) (h2 : BC = 3) (h3 : Real.cos A = -4/5) :
  Real.sin B = 2/5 ∧ Real.sin (2*B + π/6) = (12*Real.sqrt 7 + 17) / 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2784_278477


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_15_l2784_278474

/-- An arithmetic sequence with common difference d ≠ 0 -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_15 
  (a : ℕ → ℚ) (d : ℚ) 
  (h_arith : arithmetic_sequence a d)
  (h_eq : a 5^2 + a 7^2 + 16*d = a 9^2 + a 11^2) :
  sum_arithmetic a 15 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_15_l2784_278474


namespace NUMINAMATH_CALUDE_expression_simplification_l2784_278413

theorem expression_simplification (x : ℝ) 
  (h1 : x ≠ 2) (h2 : x ≠ 3) (h3 : x ≠ 4) (h4 : x ≠ 5) : 
  ((x^2 - 4*x + 3) / (x^2 - 6*x + 9)) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  (x - 1)*(x - 5) / ((x - 2)*(x - 4)) := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2784_278413


namespace NUMINAMATH_CALUDE_inequality_addition_l2784_278421

theorem inequality_addition (m n c : ℝ) : m > n → m + c > n + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l2784_278421


namespace NUMINAMATH_CALUDE_problem_l2784_278471

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem problem (a b : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  (∀ n, a n ≠ 0) →
  2 * a 3 - a 1 ^ 2 = 0 →
  a 1 = d →
  b 13 = a 2 →
  b 1 = a 1 →
  b 6 * b 8 = 72 := by
sorry

end NUMINAMATH_CALUDE_problem_l2784_278471


namespace NUMINAMATH_CALUDE_umars_age_l2784_278404

/-- Given the ages of Ali, Yusaf, and Umar, prove Umar's age -/
theorem umars_age (ali_age yusaf_age umar_age : ℕ) : 
  ali_age = 8 →
  ali_age = yusaf_age + 3 →
  umar_age = 2 * yusaf_age →
  umar_age = 10 := by
sorry

end NUMINAMATH_CALUDE_umars_age_l2784_278404


namespace NUMINAMATH_CALUDE_salad_dressing_weight_is_700_l2784_278498

/-- Calculates the weight of salad dressing given bowl capacity, oil and vinegar proportions, and their densities. -/
def salad_dressing_weight (bowl_capacity : ℝ) (oil_proportion : ℝ) (vinegar_proportion : ℝ) (oil_density : ℝ) (vinegar_density : ℝ) : ℝ :=
  (bowl_capacity * oil_proportion * oil_density) + (bowl_capacity * vinegar_proportion * vinegar_density)

/-- Theorem stating that the weight of the salad dressing is 700 grams given the specified conditions. -/
theorem salad_dressing_weight_is_700 :
  salad_dressing_weight 150 (2/3) (1/3) 5 4 = 700 := by
  sorry

end NUMINAMATH_CALUDE_salad_dressing_weight_is_700_l2784_278498


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocal_squares_l2784_278401

/-- Given two circles with equations (x^2 + y^2 + 2ax + a^2 - 4 = 0) and (x^2 + y^2 - 4by - 1 + 4b^2 = 0),
    where a ∈ ℝ, ab ≠ 0, and the circles have exactly three common tangents,
    prove that the minimum value of (1/a^2 + 1/b^2) is 1. -/
theorem min_value_sum_reciprocal_squares (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x + a^2 - 4 = 0 ∨ x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0) →
  (∃! (t1 t2 t3 : ℝ × ℝ → ℝ), t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    (∀ x y : ℝ, (t1 (x, y) = 0 ∨ t2 (x, y) = 0 ∨ t3 (x, y) = 0) ↔ 
      ((x^2 + y^2 + 2*a*x + a^2 - 4 = 0) ∨ (x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0)))) →
  a ≠ 0 →
  b ≠ 0 →
  ∃ (m : ℝ), m = 1 ∧ ∀ (k : ℝ), k ≥ 0 → (1 / a^2 + 1 / b^2) ≥ m + k :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocal_squares_l2784_278401


namespace NUMINAMATH_CALUDE_max_constant_inequality_l2784_278447

theorem max_constant_inequality (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  ∀ k : ℝ, (∀ a b c d : ℝ, 0 ≤ a ∧ a ≤ 1 → 0 ≤ b ∧ b ≤ 1 → 0 ≤ c ∧ c ≤ 1 → 0 ≤ d ∧ d ≤ 1 →
    a^2*b + b^2*c + c^2*d + d^2*a + 4 ≥ k*(a^3 + b^3 + c^3 + d^3)) → k ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_constant_inequality_l2784_278447


namespace NUMINAMATH_CALUDE_circle_coverage_theorem_l2784_278492

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the configuration of circles used to cover the main circle -/
structure CoverConfiguration where
  main_circle : Circle
  covering_circles : List Circle

/-- Checks if a point is covered by a circle -/
def is_point_covered (point : ℝ × ℝ) (circle : Circle) : Prop :=
  let (x, y) := point
  let (cx, cy) := circle.center
  (x - cx)^2 + (y - cy)^2 ≤ circle.radius^2

/-- Checks if all points in the main circle are covered by at least one of the covering circles -/
def is_circle_covered (config : CoverConfiguration) : Prop :=
  ∀ point, is_point_covered point config.main_circle →
    ∃ cover_circle ∈ config.covering_circles, is_point_covered point cover_circle

/-- The main theorem stating that a circle with diameter 81.9 can be covered by 5 circles of diameter 50 -/
theorem circle_coverage_theorem :
  ∃ config : CoverConfiguration,
    config.main_circle.radius = 81.9 / 2 ∧
    config.covering_circles.length = 5 ∧
    (∀ circle ∈ config.covering_circles, circle.radius = 50 / 2) ∧
    is_circle_covered config :=
  sorry


end NUMINAMATH_CALUDE_circle_coverage_theorem_l2784_278492


namespace NUMINAMATH_CALUDE_video_recorder_wholesale_cost_l2784_278408

theorem video_recorder_wholesale_cost :
  ∀ (wholesale_cost : ℝ),
    let retail_price := wholesale_cost * 1.2
    let employee_price := retail_price * 0.8
    employee_price = 192 →
    wholesale_cost = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_video_recorder_wholesale_cost_l2784_278408


namespace NUMINAMATH_CALUDE_dany_farm_bushels_l2784_278420

/-- The number of bushels needed for Dany's farm animals for one day -/
def bushels_needed (cows sheep : ℕ) (cow_sheep_bushels : ℕ) (chickens : ℕ) (chicken_bushels : ℕ) : ℕ :=
  (cows + sheep) * cow_sheep_bushels + chicken_bushels

/-- Theorem: Dany needs 17 bushels for his animals for one day -/
theorem dany_farm_bushels :
  bushels_needed 4 3 2 7 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_dany_farm_bushels_l2784_278420


namespace NUMINAMATH_CALUDE_trig_identities_for_point_l2784_278483

/-- Given a point P(1, -3) on the terminal side of angle α, prove trigonometric identities. -/
theorem trig_identities_for_point (α : Real) :
  let P : ℝ × ℝ := (1, -3)
  (P.1 = Real.cos α * Real.sqrt (P.1^2 + P.2^2) ∧ 
   P.2 = Real.sin α * Real.sqrt (P.1^2 + P.2^2)) →
  Real.sin α = -3 * Real.sqrt 10 / 10 ∧ 
  Real.sqrt 10 * Real.cos α + Real.tan α = -2 := by
sorry

end NUMINAMATH_CALUDE_trig_identities_for_point_l2784_278483


namespace NUMINAMATH_CALUDE_meaningful_sqrt_range_l2784_278448

theorem meaningful_sqrt_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 5) → x ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_meaningful_sqrt_range_l2784_278448


namespace NUMINAMATH_CALUDE_factors_of_N_l2784_278412

/-- The number of natural-number factors of N, where N = 2^4 * 3^2 * 5^1 * 7^2 -/
def number_of_factors (N : ℕ) : ℕ :=
  (5 : ℕ) * (3 : ℕ) * (2 : ℕ) * (3 : ℕ)

/-- N is defined as 2^4 * 3^2 * 5^1 * 7^2 -/
def N : ℕ := 2^4 * 3^2 * 5^1 * 7^2

theorem factors_of_N : number_of_factors N = 90 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_N_l2784_278412


namespace NUMINAMATH_CALUDE_complement_of_union_equals_set_l2784_278455

-- Define the universal set U
def U : Finset Nat := {1,2,3,4,5}

-- Define set A
def A : Finset Nat := {1,2}

-- Define set B
def B : Finset Nat := {2,3}

-- Theorem statement
theorem complement_of_union_equals_set (h : U = {1,2,3,4,5} ∧ A = {1,2} ∧ B = {2,3}) :
  (U \ (A ∪ B)) = {4,5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_set_l2784_278455


namespace NUMINAMATH_CALUDE_expected_digits_is_correct_l2784_278419

/-- A fair 20-sided die with numbers 1 to 20 -/
def icosahedral_die : Finset ℕ := Finset.range 20

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 2

/-- The expected number of digits when rolling the die -/
def expected_digits : ℚ :=
  (icosahedral_die.sum (λ i => num_digits (i + 1))) / icosahedral_die.card

/-- Theorem: The expected number of digits is 1.55 -/
theorem expected_digits_is_correct :
  expected_digits = 31 / 20 := by sorry

end NUMINAMATH_CALUDE_expected_digits_is_correct_l2784_278419


namespace NUMINAMATH_CALUDE_faye_age_l2784_278439

/-- Represents the ages of Chad, Diana, Eduardo, and Faye --/
structure Ages where
  chad : ℕ
  diana : ℕ
  eduardo : ℕ
  faye : ℕ

/-- The conditions given in the problem --/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.diana = ages.eduardo - 5 ∧
  ages.eduardo = ages.chad + 3 ∧
  ages.faye = ages.chad + 2 ∧
  ages.diana = 18

theorem faye_age (ages : Ages) :
  satisfiesConditions ages → ages.faye = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_faye_age_l2784_278439


namespace NUMINAMATH_CALUDE_original_machines_work_hours_l2784_278468

/-- The number of original machines in the factory -/
def original_machines : ℕ := 3

/-- The number of hours the new machine works per day -/
def new_machine_hours : ℕ := 12

/-- The production rate of each machine in kg per hour -/
def production_rate : ℕ := 2

/-- The selling price of the material in dollars per kg -/
def selling_price : ℕ := 50

/-- The total earnings of the factory in one day in dollars -/
def total_earnings : ℕ := 8100

/-- Theorem stating that the original machines work 23 hours a day -/
theorem original_machines_work_hours : 
  ∃ h : ℕ, 
    (original_machines * production_rate * h + new_machine_hours * production_rate) * selling_price = total_earnings ∧ 
    h = 23 := by
  sorry

end NUMINAMATH_CALUDE_original_machines_work_hours_l2784_278468


namespace NUMINAMATH_CALUDE_murtha_pebble_collection_l2784_278443

/-- The sum of an arithmetic sequence with n terms, first term a, and common difference d -/
def arithmetic_sum (n : ℕ) (a d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Murtha's pebble collection problem -/
theorem murtha_pebble_collection :
  arithmetic_sum 12 1 1 = 78 := by
  sorry

end NUMINAMATH_CALUDE_murtha_pebble_collection_l2784_278443


namespace NUMINAMATH_CALUDE_lacy_correct_percentage_l2784_278472

theorem lacy_correct_percentage (x : ℕ) (x_pos : x > 0) :
  let total_problems := 6 * x
  let missed_problems := 2 * x
  let correct_problems := total_problems - missed_problems
  (correct_problems : ℚ) / (total_problems : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_lacy_correct_percentage_l2784_278472


namespace NUMINAMATH_CALUDE_cos_theta_minus_phi_l2784_278435

theorem cos_theta_minus_phi (θ φ : ℝ) :
  Complex.exp (θ * Complex.I) = (4 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I →
  Complex.exp (φ * Complex.I) = (5 / 13 : ℂ) - (12 / 13 : ℂ) * Complex.I →
  Real.cos (θ - φ) = -16 / 65 := by
sorry

end NUMINAMATH_CALUDE_cos_theta_minus_phi_l2784_278435


namespace NUMINAMATH_CALUDE_sphere_radius_from_hole_l2784_278400

theorem sphere_radius_from_hole (hole_width : ℝ) (hole_depth : ℝ) (sphere_radius : ℝ) : 
  hole_width = 30 ∧ hole_depth = 10 → 
  sphere_radius^2 = (hole_width/2)^2 + (sphere_radius - hole_depth)^2 →
  sphere_radius = 16.25 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_from_hole_l2784_278400


namespace NUMINAMATH_CALUDE_pencil_distribution_l2784_278436

/-- Given 1204 pens and an unknown number of pencils distributed equally among 28 students,
    prove that the total number of pencils must be a multiple of 28. -/
theorem pencil_distribution (total_pencils : ℕ) : 
  (1204 % 28 = 0) → 
  (∃ (pencils_per_student : ℕ), total_pencils = 28 * pencils_per_student) :=
by sorry

end NUMINAMATH_CALUDE_pencil_distribution_l2784_278436


namespace NUMINAMATH_CALUDE_min_value_implies_t_l2784_278445

-- Define the function f
def f (x t : ℝ) : ℝ := |x - t| + |5 - x|

-- State the theorem
theorem min_value_implies_t (t : ℝ) : 
  (∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), f x t ≥ m) → t = 2 ∨ t = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_t_l2784_278445


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2784_278449

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2784_278449


namespace NUMINAMATH_CALUDE_product_evaluation_l2784_278403

theorem product_evaluation (a : ℕ) (h : a = 7) : 
  (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 5040 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2784_278403


namespace NUMINAMATH_CALUDE_coefficient_x4_is_10_l2784_278479

/-- The coefficient of x^4 in the expansion of (x^2 + 1/x)^5 -/
def coefficient_x4 : ℕ :=
  (Nat.choose 5 2)

/-- Theorem: The coefficient of x^4 in the expansion of (x^2 + 1/x)^5 is 10 -/
theorem coefficient_x4_is_10 : coefficient_x4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_is_10_l2784_278479


namespace NUMINAMATH_CALUDE_watch_cost_price_l2784_278476

theorem watch_cost_price (loss_percentage : ℝ) (gain_percentage : ℝ) (additional_amount : ℝ) :
  loss_percentage = 10 →
  gain_percentage = 4 →
  additional_amount = 168 →
  ∃ (cost_price : ℝ),
    cost_price * (1 - loss_percentage / 100) + additional_amount = cost_price * (1 + gain_percentage / 100) ∧
    cost_price = 1200 :=
by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l2784_278476


namespace NUMINAMATH_CALUDE_min_value_is_884_l2784_278432

/-- A type representing a permutation of the numbers 1 to 9 -/
def Perm9 := { f : Fin 9 → Fin 9 // Function.Bijective f }

/-- The expression we want to minimize -/
def expr (p : Perm9) : ℕ :=
  let x₁ := (p.val 0).val + 1
  let x₂ := (p.val 1).val + 1
  let x₃ := (p.val 2).val + 1
  let y₁ := (p.val 3).val + 1
  let y₂ := (p.val 4).val + 1
  let y₃ := (p.val 5).val + 1
  let z₁ := (p.val 6).val + 1
  let z₂ := (p.val 7).val + 1
  let z₃ := (p.val 8).val + 1
  x₁ * x₂ * x₃ + y₁ * y₂ * y₃ + z₁ * z₂ * z₃ + x₁ * y₁ * z₁

/-- The theorem stating that the minimum value of the expression is 884 -/
theorem min_value_is_884 : ∀ p : Perm9, expr p ≥ 884 := by
  sorry

end NUMINAMATH_CALUDE_min_value_is_884_l2784_278432


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_sum_l2784_278422

theorem quadratic_root_ratio_sum (x₁ x₂ : ℝ) : 
  x₁^2 - x₁ - 1 = 0 → 
  x₂^2 - x₂ - 1 = 0 → 
  x₁ ≠ 0 → 
  x₂ ≠ 0 → 
  (x₂ / x₁) + (x₁ / x₂) = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_sum_l2784_278422


namespace NUMINAMATH_CALUDE_statement_truth_condition_l2784_278409

theorem statement_truth_condition (g : ℝ → ℝ) (c d : ℝ) :
  (∀ x, g x = 4 * x + 5) →
  c > 0 →
  d > 0 →
  (∀ x, |x + 3| < d → |g x + 7| < c) ↔
  d ≤ c / 4 :=
by sorry

end NUMINAMATH_CALUDE_statement_truth_condition_l2784_278409


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_l2784_278442

theorem consecutive_even_numbers (x y z : ℕ) : 
  (∃ n : ℕ, x = 2*n ∧ y = 2*n + 2 ∧ z = 2*n + 4) →  -- x, y, z are consecutive even numbers
  (x + y + z = (x + y + z) / 3 + 44) →               -- sum is 44 more than average
  z = 24                                             -- largest number is 24
:= by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_l2784_278442


namespace NUMINAMATH_CALUDE_minimum_area_of_rectangle_l2784_278488

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Checks if the actual dimensions are within the reported range --/
def withinReportedRange (reported : Rectangle) (actual : Rectangle) : Prop :=
  (actual.length ≥ reported.length - 0.5) ∧
  (actual.length ≤ reported.length + 0.5) ∧
  (actual.width ≥ reported.width - 0.5) ∧
  (actual.width ≤ reported.width + 0.5)

/-- Checks if the length is at least twice the width --/
def lengthAtLeastTwiceWidth (r : Rectangle) : Prop :=
  r.length ≥ 2 * r.width

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ :=
  r.length * r.width

/-- The reported dimensions of the tile --/
def reportedDimensions : Rectangle :=
  { length := 4, width := 6 }

theorem minimum_area_of_rectangle :
  ∃ (minRect : Rectangle),
    withinReportedRange reportedDimensions minRect ∧
    lengthAtLeastTwiceWidth minRect ∧
    area minRect = 60.5 ∧
    ∀ (r : Rectangle),
      withinReportedRange reportedDimensions r →
      lengthAtLeastTwiceWidth r →
      area r ≥ 60.5 := by
  sorry

end NUMINAMATH_CALUDE_minimum_area_of_rectangle_l2784_278488


namespace NUMINAMATH_CALUDE_f_composition_at_two_l2784_278460

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem f_composition_at_two : f (f (f 2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_at_two_l2784_278460


namespace NUMINAMATH_CALUDE_intersection_when_a_is_3_range_of_a_when_subset_l2784_278478

def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ a}
def B : Set ℝ := {x | x ≥ 2}

theorem intersection_when_a_is_3 : A 3 ∩ B = {x | 2 ≤ x ∧ x ≤ 3} := by sorry

theorem range_of_a_when_subset : 
  (∀ a : ℝ, A a ⊆ (Set.univ \ B)) → 
  {a : ℝ | ∃ x, x ∈ A a} = Set.Iio 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_3_range_of_a_when_subset_l2784_278478


namespace NUMINAMATH_CALUDE_unique_number_l2784_278462

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  ∃ (d : ℕ), d < 10 ∧
    (n - d * 10000 + n) = 54321 ∨
    (n - d * 1000 + n) = 54321 ∨
    (n - d * 100 + n) = 54321 ∨
    (n - d * 10 + n) = 54321 ∨
    (n - d + n) = 54321

theorem unique_number : ∀ n : ℕ, is_valid_number n ↔ n = 49383 := by sorry

end NUMINAMATH_CALUDE_unique_number_l2784_278462


namespace NUMINAMATH_CALUDE_parity_condition_l2784_278441

theorem parity_condition (n : ℕ) : n ≥ 2 →
  (∀ i j : ℕ, 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n →
    Even (i + j) ↔ Even (Nat.choose n i + Nat.choose n j)) ↔
  ∃ k : ℕ, k ≥ 1 ∧ n = 2^k - 2 :=
by sorry

end NUMINAMATH_CALUDE_parity_condition_l2784_278441


namespace NUMINAMATH_CALUDE_weight_of_raisins_l2784_278458

/-- Given that Kelly bought peanuts and raisins, prove the weight of raisins. -/
theorem weight_of_raisins 
  (total_weight : ℝ) 
  (peanut_weight : ℝ) 
  (h1 : total_weight = 0.5) 
  (h2 : peanut_weight = 0.1) : 
  total_weight - peanut_weight = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_raisins_l2784_278458


namespace NUMINAMATH_CALUDE_expression_simplification_l2784_278430

theorem expression_simplification (x y k : ℝ) 
  (hk : k ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = k * y) :
  (x - k / x) * (y + 1 / (k * y)) = (x^2 * k - k^3) / x^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2784_278430


namespace NUMINAMATH_CALUDE_two_truthful_students_l2784_278497

/-- Represents the performance of a student in the exam -/
inductive Performance
| Good
| NotGood

/-- Represents a student -/
inductive Student
| A
| B
| C
| D

/-- The statement made by each student -/
def statement (s : Student) (performances : Student → Performance) : Prop :=
  match s with
  | Student.A => ∀ s, performances s = Performance.NotGood
  | Student.B => ∃ s, performances s = Performance.Good
  | Student.C => performances Student.B = Performance.NotGood ∨ performances Student.D = Performance.NotGood
  | Student.D => performances Student.D = Performance.NotGood

/-- Checks if a student's statement is true -/
def isTruthful (s : Student) (performances : Student → Performance) : Prop :=
  statement s performances

theorem two_truthful_students :
  ∃ (performances : Student → Performance),
    (isTruthful Student.B performances ∧ isTruthful Student.C performances) ∧
    (¬isTruthful Student.A performances ∧ ¬isTruthful Student.D performances) ∧
    (∀ (s1 s2 : Student), isTruthful s1 performances ∧ isTruthful s2 performances ∧ s1 ≠ s2 →
      ∀ (s : Student), s ≠ s1 ∧ s ≠ s2 → ¬isTruthful s performances) :=
  sorry

end NUMINAMATH_CALUDE_two_truthful_students_l2784_278497


namespace NUMINAMATH_CALUDE_division_with_remainder_l2784_278438

theorem division_with_remainder (A : ℕ) (h : 14 = A * 3 + 2) : A = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_with_remainder_l2784_278438


namespace NUMINAMATH_CALUDE_equal_coin_count_l2784_278407

def coin_count (t : ℕ) : ℕ := t / 3

theorem equal_coin_count (t : ℕ) (h : t % 3 = 0) :
  let one_dollar_count := coin_count t
  let two_dollar_count := coin_count t
  one_dollar_count * 1 + two_dollar_count * 2 = t ∧
  one_dollar_count = two_dollar_count :=
by sorry

end NUMINAMATH_CALUDE_equal_coin_count_l2784_278407


namespace NUMINAMATH_CALUDE_factorization_of_x_squared_minus_four_l2784_278446

theorem factorization_of_x_squared_minus_four (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x_squared_minus_four_l2784_278446


namespace NUMINAMATH_CALUDE_herman_bird_feeding_l2784_278434

/-- The number of days Herman feeds the birds -/
def feeding_days : ℕ := 90

/-- The amount of food Herman gives per feeding in cups -/
def food_per_feeding : ℚ := 1/2

/-- The number of feedings per day -/
def feedings_per_day : ℕ := 2

/-- Calculates the total amount of food needed for the feeding period -/
def total_food_needed (days : ℕ) (food_per_feeding : ℚ) (feedings_per_day : ℕ) : ℚ :=
  (days : ℚ) * food_per_feeding * (feedings_per_day : ℚ)

theorem herman_bird_feeding :
  total_food_needed feeding_days food_per_feeding feedings_per_day = 90 := by
  sorry

end NUMINAMATH_CALUDE_herman_bird_feeding_l2784_278434


namespace NUMINAMATH_CALUDE_vaccine_cost_reduction_l2784_278475

/-- The cost reduction for vaccine production over one year -/
def costReduction (initialCost : ℝ) (decreaseRate : ℝ) : ℝ :=
  initialCost * decreaseRate - initialCost * decreaseRate^2

/-- Theorem: The cost reduction for producing 1 set of vaccines this year
    compared to last year, given an initial cost of 5000 yuan two years ago
    and an annual average decrease rate of x, is 5000x - 5000x^2 yuan. -/
theorem vaccine_cost_reduction (x : ℝ) :
  costReduction 5000 x = 5000 * x - 5000 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_vaccine_cost_reduction_l2784_278475


namespace NUMINAMATH_CALUDE_percentage_of_amount_twenty_five_percent_of_300_l2784_278402

theorem percentage_of_amount (amount : ℝ) (percentage : ℝ) :
  (percentage / 100) * amount = (percentage * amount) / 100 := by sorry

theorem twenty_five_percent_of_300 :
  (25 : ℝ) / 100 * 300 = 75 := by sorry

end NUMINAMATH_CALUDE_percentage_of_amount_twenty_five_percent_of_300_l2784_278402


namespace NUMINAMATH_CALUDE_stating_mooncake_packing_solution_l2784_278463

/-- Represents the number of mooncakes in a large bag -/
def large_bag : ℕ := 9

/-- Represents the number of mooncakes in a small package -/
def small_package : ℕ := 4

/-- Represents the total number of mooncakes -/
def total_mooncakes : ℕ := 35

/-- 
Theorem stating that there exist non-negative integers x and y 
such that 9x + 4y = 35, and x + y is minimized
-/
theorem mooncake_packing_solution :
  ∃ x y : ℕ, large_bag * x + small_package * y = total_mooncakes ∧
  ∀ a b : ℕ, large_bag * a + small_package * b = total_mooncakes → x + y ≤ a + b :=
sorry

end NUMINAMATH_CALUDE_stating_mooncake_packing_solution_l2784_278463


namespace NUMINAMATH_CALUDE_integer_between_sqrt_twelve_l2784_278425

theorem integer_between_sqrt_twelve : ∃ (m : ℤ), m < 2 * Real.sqrt 3 ∧ 2 * Real.sqrt 3 < m + 1 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_integer_between_sqrt_twelve_l2784_278425


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_l2784_278459

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical (x y z : ℝ) (h_x : x = 4) (h_y : y = 4 * Real.sqrt 3) (h_z : z = 5) :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = 8 ∧ θ = Real.pi / 3 ∧ z = 5 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_l2784_278459


namespace NUMINAMATH_CALUDE_simplify_expression_l2784_278457

theorem simplify_expression : 18 * (8 / 15) * (1 / 12) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2784_278457


namespace NUMINAMATH_CALUDE_washing_machine_loads_l2784_278414

theorem washing_machine_loads (machine_capacity : ℕ) (total_clothes : ℕ) : 
  machine_capacity = 5 → total_clothes = 53 → 
  (total_clothes + machine_capacity - 1) / machine_capacity = 11 := by
sorry

end NUMINAMATH_CALUDE_washing_machine_loads_l2784_278414


namespace NUMINAMATH_CALUDE_impossibleToGet2015Stacks_l2784_278469

/-- Represents a collection of token stacks -/
structure TokenStacks where
  stacks : List Nat
  inv : stacks.sum = 2014

/-- Represents the allowed operations on token stacks -/
inductive Operation
  | Split : Nat → Nat → Operation  -- Split a stack into two
  | Merge : Nat → Nat → Operation  -- Merge two stacks

/-- Applies an operation to the token stacks -/
def applyOperation (ts : TokenStacks) (op : Operation) : TokenStacks :=
  match op with
  | Operation.Split i j => { stacks := i :: j :: ts.stacks.tail, inv := sorry }
  | Operation.Merge i j => { stacks := (i + j) :: ts.stacks.tail.tail, inv := sorry }

/-- The main theorem to prove -/
theorem impossibleToGet2015Stacks (ts : TokenStacks) :
  ¬∃ (ops : List Operation), (ops.foldl applyOperation ts).stacks = List.replicate 2015 1 :=
sorry

end NUMINAMATH_CALUDE_impossibleToGet2015Stacks_l2784_278469


namespace NUMINAMATH_CALUDE_coeff_x3_in_expansion_l2784_278493

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in (x-4)^5
def coeff_x3 (x : ℝ) : ℝ := binomial 5 2 * (-4)^2

-- Theorem statement
theorem coeff_x3_in_expansion :
  coeff_x3 x = 160 := by sorry

end NUMINAMATH_CALUDE_coeff_x3_in_expansion_l2784_278493


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l2784_278482

theorem cylinder_height_relationship (r1 h1 r2 h2 : ℝ) :
  r1 > 0 ∧ h1 > 0 ∧ r2 > 0 ∧ h2 > 0 →
  r2 = 1.2 * r1 →
  π * r1^2 * h1 = π * r2^2 * h2 →
  h1 = 1.44 * h2 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l2784_278482


namespace NUMINAMATH_CALUDE_favorite_numbers_product_l2784_278494

/-- Sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- Definition of a favorite number -/
def is_favorite (n : ℕ+) : Prop :=
  n * sum_of_digits n = 10 * n

/-- Theorem statement -/
theorem favorite_numbers_product :
  ∃ (a b c : ℕ+),
    a * b * c = 71668 ∧
    is_favorite a ∧
    is_favorite b ∧
    is_favorite c := by sorry

end NUMINAMATH_CALUDE_favorite_numbers_product_l2784_278494


namespace NUMINAMATH_CALUDE_square_sum_equals_eighteen_l2784_278487

theorem square_sum_equals_eighteen (a b : ℝ) (h1 : a - b = Real.sqrt 2) (h2 : a * b = 4) :
  (a + b)^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_eighteen_l2784_278487


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2784_278465

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 10*p - 3 = 0 →
  q^3 - 8*q^2 + 10*q - 3 = 0 →
  r^3 - 8*r^2 + 10*r - 3 = 0 →
  p / (q*r + 2) + q / (p*r + 2) + r / (p*q + 2) = 8/5 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2784_278465


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l2784_278486

theorem x_range_for_inequality (x : ℝ) :
  (∀ m : ℝ, |m| ≤ 1 → x^2 - 2 > m*x) →
  x < -2 ∨ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l2784_278486


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2784_278467

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.side + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * Real.sqrt ((t.side : ℝ)^2 - ((t.base : ℝ) / 2)^2) / 2

theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    (t1.base : ℚ) / (t2.base : ℚ) = 5 / 4 ∧
    area t1 = area t2 ∧
    perimeter t1 = perimeter t2 ∧
    perimeter t1 = 192 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      (s1.base : ℚ) / (s2.base : ℚ) = 5 / 4 →
      area s1 = area s2 →
      perimeter s1 = perimeter s2 →
      perimeter s1 ≥ 192) :=
by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2784_278467


namespace NUMINAMATH_CALUDE_trigonometric_expressions_equal_half_l2784_278484

theorem trigonometric_expressions_equal_half :
  let expr1 := Real.sin (15 * π / 180) * Real.cos (15 * π / 180)
  let expr2 := Real.cos (π / 8)^2 - Real.sin (π / 8)^2
  let expr3 := Real.tan (22.5 * π / 180) / (1 - Real.tan (22.5 * π / 180)^2)
  (expr1 ≠ 1/2 ∧ expr2 ≠ 1/2 ∧ expr3 = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_expressions_equal_half_l2784_278484


namespace NUMINAMATH_CALUDE_sin_15_cos_15_eq_one_fourth_l2784_278405

theorem sin_15_cos_15_eq_one_fourth : 4 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_eq_one_fourth_l2784_278405


namespace NUMINAMATH_CALUDE_cost_price_per_metre_l2784_278495

/-- Proves that given a cloth length of 200 metres sold for Rs. 12000 with a loss of Rs. 12 per metre, the cost price for one metre of cloth is Rs. 72. -/
theorem cost_price_per_metre (total_length : ℕ) (selling_price : ℕ) (loss_per_metre : ℕ) :
  total_length = 200 →
  selling_price = 12000 →
  loss_per_metre = 12 →
  (selling_price + total_length * loss_per_metre) / total_length = 72 := by
sorry

end NUMINAMATH_CALUDE_cost_price_per_metre_l2784_278495


namespace NUMINAMATH_CALUDE_no_real_solutions_l2784_278464

theorem no_real_solutions :
  ¬ ∃ x : ℝ, (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2784_278464


namespace NUMINAMATH_CALUDE_min_value_and_solution_set_l2784_278418

def f (a : ℝ) (x : ℝ) : ℝ := |x + 2| - |x - a|

theorem min_value_and_solution_set (a : ℝ) (h1 : a > 0) :
  (∃ (m : ℝ), m = -3 ∧ ∀ x, f a x ≥ m) →
  (a = 1 ∧ 
   ∀ x, |f a x| ≤ 2 ↔ a / 2 - 2 ≤ x ∧ x < a / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_solution_set_l2784_278418


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2784_278454

/-- The eccentricity of an ellipse with specific geometric properties -/
theorem ellipse_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let e := Real.sqrt ((a^2 - b^2) / a^2)
  let AF := a - (e * a)
  let AB := Real.sqrt (a^2 + b^2)
  let BF := a
  (∃ (r : ℝ), AF * r = AB ∧ AB * r = 3 * BF) →
  e = (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2784_278454


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l2784_278456

theorem negation_of_proposition (P : ℝ → Prop) : 
  (¬∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬P x) :=
by sorry

theorem negation_of_specific_proposition : 
  (¬∀ x : ℝ, x^2 + x - 1 < 0) ↔ (∃ x : ℝ, x^2 + x - 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l2784_278456


namespace NUMINAMATH_CALUDE_smallest_better_discount_l2784_278461

theorem smallest_better_discount (x : ℝ) (h : x > 0) : ∃ (n : ℕ), n = 38 ∧ 
  (∀ (m : ℕ), m < n → 
    ((1 - m / 100) * x < (1 - 0.2) * (1 - 0.2) * x ∨
     (1 - m / 100) * x < (1 - 0.12) * (1 - 0.12) * (1 - 0.12) * x ∨
     (1 - m / 100) * x < (1 - 0.3) * (1 - 0.1) * x)) ∧
  (1 - n / 100) * x > (1 - 0.2) * (1 - 0.2) * x ∧
  (1 - n / 100) * x > (1 - 0.12) * (1 - 0.12) * (1 - 0.12) * x ∧
  (1 - n / 100) * x > (1 - 0.3) * (1 - 0.1) * x :=
sorry

end NUMINAMATH_CALUDE_smallest_better_discount_l2784_278461


namespace NUMINAMATH_CALUDE_inequality_solution_l2784_278415

theorem inequality_solution (x : ℝ) : 5 * x - 12 ≤ 2 * (4 * x - 3) ↔ x ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2784_278415


namespace NUMINAMATH_CALUDE_optimal_deposit_rate_l2784_278451

/-- The bank's profit function -/
def profit (k : ℝ) (x : ℝ) : ℝ := 0.048 * k * x^2 - k * x^3

/-- The derivative of the profit function -/
def profit_derivative (k : ℝ) (x : ℝ) : ℝ := 0.096 * k * x - 3 * k * x^2

theorem optimal_deposit_rate (k : ℝ) (h : k > 0) :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 0.048 ∧ 
  (∀ (y : ℝ), y ∈ Set.Ioo 0 0.048 → profit k x ≥ profit k y) ∧
  x = 0.032 := by
  sorry

#eval (0.032 : ℝ) * 100  -- Should output 3.2

end NUMINAMATH_CALUDE_optimal_deposit_rate_l2784_278451


namespace NUMINAMATH_CALUDE_set_swept_equals_parabola_l2784_278444

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line A_t B_t for a given t -/
def line_A_t_B_t (t : ℝ) (p : Point) : Prop :=
  p.y = t * p.x - t^2 + 1

/-- The set of all points on or below any line A_t B_t -/
def set_swept_by_lines (p : Point) : Prop :=
  ∃ t : ℝ, line_A_t_B_t t p

/-- The parabola y = x^2/4 + 1 -/
def parabola (p : Point) : Prop :=
  p.y ≤ p.x^2 / 4 + 1

theorem set_swept_equals_parabola :
  ∀ p : Point, set_swept_by_lines p ↔ parabola p := by
  sorry


end NUMINAMATH_CALUDE_set_swept_equals_parabola_l2784_278444


namespace NUMINAMATH_CALUDE_binomial_expansion_equality_l2784_278470

theorem binomial_expansion_equality (x : ℝ) : 
  (x - 1)^4 - 4*x*(x - 1)^3 + 6*x^2*(x - 1)^2 - 4*x^3*(x - 1) * x^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_equality_l2784_278470


namespace NUMINAMATH_CALUDE_solution_existence_l2784_278424

theorem solution_existence (k : ℕ+) :
  (∃ x y : ℕ+, x * (x + k) = y * (y + 1)) ↔ (k = 1 ∨ k ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_solution_existence_l2784_278424


namespace NUMINAMATH_CALUDE_return_trip_duration_l2784_278426

/-- Represents the flight scenario between two towns -/
structure FlightScenario where
  d : ℝ  -- distance between towns
  p : ℝ  -- speed of the plane in still air
  w₁ : ℝ  -- wind speed against the plane
  w₂ : ℝ  -- wind speed with the plane

/-- The conditions of the flight scenario -/
def flight_conditions (f : FlightScenario) : Prop :=
  f.d / (f.p - f.w₁) = 120 ∧  -- outbound trip takes 120 minutes
  f.d / (f.p + f.w₂) = f.d / f.p - 10  -- return trip is 10 minutes faster than in still air

/-- The theorem to prove -/
theorem return_trip_duration (f : FlightScenario) 
  (h : flight_conditions f) : f.d / (f.p + f.w₂) = 72 := by
  sorry


end NUMINAMATH_CALUDE_return_trip_duration_l2784_278426


namespace NUMINAMATH_CALUDE_telephone_answered_probability_l2784_278406

theorem telephone_answered_probability :
  let p1 : ℝ := 0.1  -- Probability of answering at first ring
  let p2 : ℝ := 0.3  -- Probability of answering at second ring
  let p3 : ℝ := 0.4  -- Probability of answering at third ring
  let p4 : ℝ := 0.1  -- Probability of answering at fourth ring
  1 - (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 0.9
  := by sorry

end NUMINAMATH_CALUDE_telephone_answered_probability_l2784_278406


namespace NUMINAMATH_CALUDE_height_difference_l2784_278499

/-- Given the heights of Anne, her sister, and Bella, prove the height difference between Bella and Anne's sister. -/
theorem height_difference (anne_height : ℝ) (sister_ratio : ℝ) (bella_ratio : ℝ)
  (h1 : anne_height = 80)
  (h2 : sister_ratio = 2)
  (h3 : bella_ratio = 3) :
  bella_ratio * anne_height - anne_height / sister_ratio = 200 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l2784_278499


namespace NUMINAMATH_CALUDE_trig_product_equals_one_sixteenth_l2784_278416

theorem trig_product_equals_one_sixteenth :
  (1 - Real.sin (π / 12)) * (1 - Real.sin (5 * π / 12)) *
  (1 - Real.sin (7 * π / 12)) * (1 - Real.sin (11 * π / 12)) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_equals_one_sixteenth_l2784_278416


namespace NUMINAMATH_CALUDE_investment_strategy_l2784_278491

/-- Represents the investment and profit parameters for a manufacturing company's production lines. -/
structure ProductionParameters where
  initialInvestmentA : ℝ  -- Initial investment in production line A (in million yuan)
  profitRateA : ℝ         -- Profit rate for production line A (profit per 10,000 yuan invested)
  investmentReduction : ℝ -- Reduction in investment for A (in million yuan)
  profitIncreaseRate : ℝ  -- Rate of profit increase for A (as a percentage)
  profitRateB : ℝ → ℝ     -- Profit rate function for production line B
  a : ℝ                   -- Parameter a for production line B's profit rate

/-- The main theorem about the manufacturing company's investment strategy. -/
theorem investment_strategy 
  (params : ProductionParameters) 
  (h_initialInvestmentA : params.initialInvestmentA = 50)
  (h_profitRateA : params.profitRateA = 1.5)
  (h_profitIncreaseRate : params.profitIncreaseRate = 0.005)
  (h_profitRateB : params.profitRateB = fun x => 1.5 * (params.a - 0.013 * x))
  (h_a_positive : params.a > 0) :
  (∃ x_range : Set ℝ, x_range = {x | 0 < x ∧ x ≤ 300} ∧ 
    ∀ x ∈ x_range, 
      (params.initialInvestmentA - x) * params.profitRateA * (1 + x * params.profitIncreaseRate) ≥ 
      params.initialInvestmentA * params.profitRateA) ∧
  (∃ a_max : ℝ, a_max = 5.5 ∧
    ∀ x > 0, x * params.profitRateB x ≤ 
      (params.initialInvestmentA - x) * params.profitRateA * (1 + x * params.profitIncreaseRate) ∧
    params.a ≤ a_max) := by
  sorry

end NUMINAMATH_CALUDE_investment_strategy_l2784_278491


namespace NUMINAMATH_CALUDE_total_fish_count_l2784_278433

/-- Represents the number of fish in Jonah's aquariums -/
def total_fish (x y : ℕ) : ℤ :=
  let first_aquarium := 14 + 2 - 2 * x + 3
  let second_aquarium := 18 + 4 - 4 * y + 5
  first_aquarium + second_aquarium

/-- The theorem stating the total number of fish in both aquariums -/
theorem total_fish_count (x y : ℕ) : total_fish x y = 46 - 2 * x - 4 * y := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l2784_278433


namespace NUMINAMATH_CALUDE_equation_solutions_l2784_278480

theorem equation_solutions :
  (∃ x : ℚ, 2*x + 1 = -2 - 3*x ∧ x = -3/5) ∧
  (∃ x : ℚ, x + (1-2*x)/3 = 2 - (x+2)/2 ∧ x = 4/5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2784_278480


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2784_278437

open Set

-- Define the sets M and N
def M : Set ℝ := {x | 2 - x < 0}
def N : Set ℝ := {x | x - 3 ≤ 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Ioo 2 3 ∪ Iic 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2784_278437


namespace NUMINAMATH_CALUDE_rainville_rainfall_2006_l2784_278481

/-- The total rainfall in Rainville in 2006 given the average monthly rainfall in 2005 and the increase in 2006 -/
theorem rainville_rainfall_2006 (rainfall_2005 rainfall_increase : ℝ) : 
  rainfall_2005 = 50.0 →
  rainfall_increase = 3 →
  (rainfall_2005 + rainfall_increase) * 12 = 636 := by
  sorry

end NUMINAMATH_CALUDE_rainville_rainfall_2006_l2784_278481


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2784_278490

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The given line 2x - 3y + 4 = 0 -/
def givenLine : Line := { a := 2, b := -3, c := 4 }

/-- The point (-1, 2) -/
def givenPoint : Point := { x := -1, y := 2 }

/-- The line we want to prove -/
def targetLine : Line := { a := 2, b := -3, c := 8 }

theorem parallel_line_through_point :
  (targetLine.isParallelTo givenLine) ∧
  (givenPoint.liesOn targetLine) := by
  sorry

#check parallel_line_through_point

end NUMINAMATH_CALUDE_parallel_line_through_point_l2784_278490


namespace NUMINAMATH_CALUDE_quadratic_polynomial_sequence_bound_l2784_278411

/-- A real quadratic polynomial with positive leading coefficient and no fixed point -/
structure QuadraticPolynomial where
  f : ℝ → ℝ
  is_quadratic : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c
  positive_leading : ∃ (a b c : ℝ), (∀ x, f x = a * x^2 + b * x + c) ∧ a > 0
  no_fixed_point : ∀ α : ℝ, f α ≠ α

/-- The theorem statement -/
theorem quadratic_polynomial_sequence_bound (f : QuadraticPolynomial) :
  ∃ n : ℕ+, ∀ (a : ℕ → ℝ),
    (∀ i : ℕ, i ≥ 1 → i ≤ n → a i = f.f (a (i-1))) →
    a n > 2021 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_sequence_bound_l2784_278411


namespace NUMINAMATH_CALUDE_negation_equivalence_l2784_278429

theorem negation_equivalence :
  (¬ ∀ (n : ℕ), ∃ (x : ℝ), n^2 < x) ↔ (∃ (n : ℕ), ∀ (x : ℝ), n^2 ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2784_278429


namespace NUMINAMATH_CALUDE_sum_reciprocals_l2784_278452

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) :
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^4 = 1 →
  ω ≠ 1 →
  (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 4 / ω →
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l2784_278452


namespace NUMINAMATH_CALUDE_candy_bar_calories_l2784_278453

theorem candy_bar_calories (total_calories : ℕ) (total_bars : ℕ) (h1 : total_calories = 2016) (h2 : total_bars = 42) :
  (total_calories / total_bars) / 12 = 4 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_calories_l2784_278453


namespace NUMINAMATH_CALUDE_exists_m_n_for_any_d_l2784_278450

theorem exists_m_n_for_any_d (d : ℤ) : ∃ (m n : ℤ), d = (n - 2*m + 1) / (m^2 - n) := by
  sorry

end NUMINAMATH_CALUDE_exists_m_n_for_any_d_l2784_278450


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2784_278496

/-- The volume of a cube with surface area 150 cm² is 125 cm³. -/
theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 150 → s^3 = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2784_278496


namespace NUMINAMATH_CALUDE_casey_savings_l2784_278417

/-- Represents the weekly savings when hiring the cheaper employee --/
def weeklySavings (hourlyRate1 hourlyRate2 subsidy hoursPerWeek : ℝ) : ℝ :=
  (hourlyRate1 * hoursPerWeek) - ((hourlyRate2 - subsidy) * hoursPerWeek)

/-- Proves that Casey saves $160 per week by hiring the cheaper employee --/
theorem casey_savings :
  let hourlyRate1 : ℝ := 20
  let hourlyRate2 : ℝ := 22
  let subsidy : ℝ := 6
  let hoursPerWeek : ℝ := 40
  weeklySavings hourlyRate1 hourlyRate2 subsidy hoursPerWeek = 160 := by
  sorry

end NUMINAMATH_CALUDE_casey_savings_l2784_278417


namespace NUMINAMATH_CALUDE_equal_sum_sequence_definition_l2784_278466

def is_equal_sum_sequence (a : ℕ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = k

theorem equal_sum_sequence_definition (a : ℕ → ℝ) :
  is_equal_sum_sequence a ↔
  ∃ k : ℝ, ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = k :=
by sorry

end NUMINAMATH_CALUDE_equal_sum_sequence_definition_l2784_278466


namespace NUMINAMATH_CALUDE_typist_salary_problem_l2784_278489

theorem typist_salary_problem (original_salary : ℝ) : 
  (original_salary * 1.1 * 0.95 = 3135) → original_salary = 3000 := by
  sorry

end NUMINAMATH_CALUDE_typist_salary_problem_l2784_278489


namespace NUMINAMATH_CALUDE_smallest_integer_y_l2784_278410

theorem smallest_integer_y : ∃ y : ℤ, (y : ℚ) / 4 + 3 / 7 > 2 / 3 ∧ ∀ z : ℤ, z < y → (z : ℚ) / 4 + 3 / 7 ≤ 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_l2784_278410


namespace NUMINAMATH_CALUDE_hotel_room_pricing_and_schemes_l2784_278485

theorem hotel_room_pricing_and_schemes :
  ∀ (price_A price_B : ℕ) (schemes : List (ℕ × ℕ)),
  (∃ n : ℕ, 6000 = n * price_A ∧ 4400 = n * price_B) →
  price_A = price_B + 80 →
  (∀ (a b : ℕ), (a, b) ∈ schemes → a + b = 30) →
  (∀ (a b : ℕ), (a, b) ∈ schemes → 2 * a ≥ b) →
  (∀ (a b : ℕ), (a, b) ∈ schemes → a * price_A + b * price_B ≤ 7600) →
  price_A = 300 ∧ price_B = 220 ∧ schemes = [(10, 20), (11, 19), (12, 18)] := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_pricing_and_schemes_l2784_278485
