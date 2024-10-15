import Mathlib

namespace NUMINAMATH_CALUDE_slope_intercept_sum_l2143_214361

/-- Given two points A and B on a line, prove that the sum of the line's slope and y-intercept is 10. -/
theorem slope_intercept_sum (A B : ℝ × ℝ) : 
  A = (5, 6) → B = (8, 3) → 
  let m := (B.2 - A.2) / (B.1 - A.1)
  let b := A.2 - m * A.1
  m + b = 10 := by sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_l2143_214361


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l2143_214343

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) : 
  2 * (x - 2)^2 + 2 * (y - 3)^2 + 2 * (z - 6)^2 = 0 → x + y + z = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l2143_214343


namespace NUMINAMATH_CALUDE_eighty_nine_degrees_is_acute_l2143_214381

-- Define what an acute angle is
def is_acute_angle (angle : ℝ) : Prop := 0 < angle ∧ angle < 90

-- State the theorem
theorem eighty_nine_degrees_is_acute : is_acute_angle 89 := by
  sorry

end NUMINAMATH_CALUDE_eighty_nine_degrees_is_acute_l2143_214381


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l2143_214333

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define a point on the parabola
def point_on_parabola (p : ℝ) (x y : ℝ) : Prop := parabola p x y

-- Define the fixed points A and B
def point_A (a b : ℝ) : ℝ × ℝ := (a, b)
def point_B (a : ℝ) : ℝ × ℝ := (-a, 0)

-- Define a line passing through two points
def line_through_points (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (y2 - y1) * (x - x1)

-- Define the theorem
theorem fixed_point_theorem (p a b : ℝ) (M M1 M2 : ℝ × ℝ) 
  (h1 : a * b ≠ 0)
  (h2 : b^2 ≠ 2 * p * a)
  (h3 : point_on_parabola p M.1 M.2)
  (h4 : point_on_parabola p M1.1 M1.2)
  (h5 : point_on_parabola p M2.1 M2.2)
  (h6 : line_through_points a b M.1 M.2 M1.1 M1.2)
  (h7 : line_through_points (-a) 0 M.1 M.2 M2.1 M2.2)
  (h8 : M1 ≠ M2) :
  line_through_points M1.1 M1.2 M2.1 M2.2 a (2 * p * a / b) :=
sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l2143_214333


namespace NUMINAMATH_CALUDE_decimal_difference_value_l2143_214390

/-- The repeating decimal 0.0̅6̅ -/
def repeating_decimal : ℚ := 2 / 33

/-- The terminating decimal 0.06 -/
def terminating_decimal : ℚ := 6 / 100

/-- The difference between the repeating decimal 0.0̅6̅ and the terminating decimal 0.06 -/
def decimal_difference : ℚ := repeating_decimal - terminating_decimal

theorem decimal_difference_value : decimal_difference = 2 / 3300 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_value_l2143_214390


namespace NUMINAMATH_CALUDE_inequality_solution_l2143_214338

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 6) ≥ 1) ↔ (x ∈ Set.Icc (-4) (-2) ∪ Set.Icc 2 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2143_214338


namespace NUMINAMATH_CALUDE_function_property_Z_function_property_Q_l2143_214389

-- For integers
theorem function_property_Z (f : ℤ → ℤ) :
  (∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))) →
  (∀ x : ℤ, f x = 2 * x ∨ f x = 0) :=
sorry

-- For rationals (bonus)
theorem function_property_Q (f : ℚ → ℚ) :
  (∀ a b : ℚ, f (2 * a) + 2 * f b = f (f (a + b))) →
  (∀ x : ℚ, f x = 2 * x ∨ f x = 0) :=
sorry

end NUMINAMATH_CALUDE_function_property_Z_function_property_Q_l2143_214389


namespace NUMINAMATH_CALUDE_chord_length_l2143_214306

/-- The length of the chord cut by a circle on a line --/
theorem chord_length (x y : ℝ) : 
  (x^2 + y^2 = 9) →  -- Circle equation
  (x + y = 2 * Real.sqrt 2) →  -- Line equation
  ∃ (a b : ℝ), (a - x)^2 + (b - y)^2 = 25 ∧  -- Chord endpoints
               (a^2 + b^2 = 9) ∧  -- Endpoints on circle
               (a + b = 2 * Real.sqrt 2) :=  -- Endpoints on line
by sorry

end NUMINAMATH_CALUDE_chord_length_l2143_214306


namespace NUMINAMATH_CALUDE_subset_P_l2143_214310

def P : Set ℝ := {x | x ≤ 3}

theorem subset_P : {-1} ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_subset_P_l2143_214310


namespace NUMINAMATH_CALUDE_jane_age_proof_l2143_214324

/-- Represents Jane's age when she started babysitting -/
def start_age : ℕ := 20

/-- Represents the number of years since Jane stopped babysitting -/
def years_since_stop : ℕ := 10

/-- Represents the current age of the oldest person Jane could have babysat -/
def oldest_babysat_current_age : ℕ := 22

/-- Represents Jane's current age -/
def jane_current_age : ℕ := 34

theorem jane_age_proof :
  ∀ (jane_age : ℕ),
    jane_age ≥ start_age →
    (oldest_babysat_current_age - years_since_stop) * 2 ≤ jane_age - years_since_stop →
    jane_age = jane_current_age := by
  sorry

end NUMINAMATH_CALUDE_jane_age_proof_l2143_214324


namespace NUMINAMATH_CALUDE_dad_steps_l2143_214380

/-- Represents the number of steps taken by each person -/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- Defines the relationship between Dad's and Masha's steps -/
def dad_masha_ratio (s : Steps) : Prop :=
  5 * s.dad = 3 * s.masha

/-- Defines the relationship between Masha's and Yasha's steps -/
def masha_yasha_ratio (s : Steps) : Prop :=
  5 * s.masha = 3 * s.yasha

/-- States that Masha and Yasha together took 400 steps -/
def total_masha_yasha (s : Steps) : Prop :=
  s.masha + s.yasha = 400

theorem dad_steps (s : Steps) 
  (h1 : dad_masha_ratio s)
  (h2 : masha_yasha_ratio s)
  (h3 : total_masha_yasha s) :
  s.dad = 90 := by
  sorry

end NUMINAMATH_CALUDE_dad_steps_l2143_214380


namespace NUMINAMATH_CALUDE_system_solution_l2143_214399

theorem system_solution (a b : ℝ) : 
  a^2 + b^2 = 25 ∧ 3*(a + b) - a*b = 15 ↔ 
  ((a = 0 ∧ b = 5) ∨ (a = 5 ∧ b = 0) ∨ (a = 4 ∧ b = -3) ∨ (a = -3 ∧ b = 4)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2143_214399


namespace NUMINAMATH_CALUDE_log_equation_solution_l2143_214339

theorem log_equation_solution (b x : ℝ) (hb_pos : b > 0) (hb_neq_one : b ≠ 1) (hx_neq_one : x ≠ 1) :
  (Real.log x) / (Real.log (b^3)) + 3 * (Real.log b) / (Real.log x) = 2 → x = b^3 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2143_214339


namespace NUMINAMATH_CALUDE_intersection_condition_l2143_214323

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) := {p | p.2 = -p.1^2 + m*p.1 - 1}
def B : Set (ℝ × ℝ) := {p | p.1 + p.2 = 3 ∧ 0 ≤ p.1 ∧ p.1 ≤ 3}

-- Define the condition for exactly one intersection
def exactly_one_intersection (m : ℝ) : Prop :=
  ∃! p, p ∈ A m ∩ B

-- State the theorem
theorem intersection_condition (m : ℝ) :
  exactly_one_intersection m ↔ (m = 3 ∨ m > 10/3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l2143_214323


namespace NUMINAMATH_CALUDE_touring_plans_count_l2143_214328

def num_destinations : Nat := 3
def num_students : Nat := 4

def total_assignments : Nat := num_destinations ^ num_students

def assignments_without_specific_destination : Nat := (num_destinations - 1) ^ num_students

theorem touring_plans_count : 
  total_assignments - assignments_without_specific_destination = 65 := by
  sorry

end NUMINAMATH_CALUDE_touring_plans_count_l2143_214328


namespace NUMINAMATH_CALUDE_squirrel_travel_time_l2143_214358

/-- Proves that a squirrel traveling 3 miles at 6 miles per hour takes 30 minutes -/
theorem squirrel_travel_time :
  let speed : ℝ := 6 -- miles per hour
  let distance : ℝ := 3 -- miles
  let time_hours : ℝ := distance / speed
  let time_minutes : ℝ := time_hours * 60
  time_minutes = 30 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_travel_time_l2143_214358


namespace NUMINAMATH_CALUDE_paul_total_crayons_l2143_214367

/-- The number of crayons Paul initially had -/
def initial_crayons : ℝ := 479.0

/-- The number of additional crayons Paul received -/
def additional_crayons : ℝ := 134.0

/-- The total number of crayons Paul now has -/
def total_crayons : ℝ := initial_crayons + additional_crayons

/-- Theorem stating that Paul now has 613.0 crayons -/
theorem paul_total_crayons : total_crayons = 613.0 := by
  sorry

end NUMINAMATH_CALUDE_paul_total_crayons_l2143_214367


namespace NUMINAMATH_CALUDE_sum_exterior_angles_regular_pentagon_sum_exterior_angles_regular_pentagon_proof_l2143_214317

/-- The sum of the exterior angles of a regular pentagon is 360 degrees. -/
theorem sum_exterior_angles_regular_pentagon : ℝ :=
  360

/-- A polygon is a closed plane figure with straight sides. -/
def Polygon : Type := sorry

/-- A regular polygon is a polygon with all sides and angles equal. -/
def RegularPolygon (p : Polygon) : Prop := sorry

/-- A pentagon is a polygon with five sides. -/
def Pentagon (p : Polygon) : Prop := sorry

/-- The sum of the exterior angles of any polygon is constant. -/
axiom sum_exterior_angles_constant (p : Polygon) : ℝ

/-- The sum of the exterior angles of any polygon is 360 degrees. -/
axiom sum_exterior_angles_360 (p : Polygon) : sum_exterior_angles_constant p = 360

/-- Theorem: The sum of the exterior angles of a regular pentagon is 360 degrees. -/
theorem sum_exterior_angles_regular_pentagon_proof (p : Polygon) 
  (h1 : RegularPolygon p) (h2 : Pentagon p) : 
  sum_exterior_angles_constant p = 360 := by
  sorry

end NUMINAMATH_CALUDE_sum_exterior_angles_regular_pentagon_sum_exterior_angles_regular_pentagon_proof_l2143_214317


namespace NUMINAMATH_CALUDE_N_mod_45_l2143_214318

/-- N is the number formed by concatenating integers from 1 to 52 -/
def N : ℕ := sorry

theorem N_mod_45 : N % 45 = 37 := by sorry

end NUMINAMATH_CALUDE_N_mod_45_l2143_214318


namespace NUMINAMATH_CALUDE_greatest_value_cubic_inequality_l2143_214320

theorem greatest_value_cubic_inequality :
  let f : ℝ → ℝ := λ b => -b^3 + b^2 + 7*b - 10
  ∃ (max_b : ℝ), max_b = 4 + Real.sqrt 6 ∧
    (∀ b : ℝ, f b ≥ 0 → b ≤ max_b) ∧
    f max_b ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_cubic_inequality_l2143_214320


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l2143_214311

theorem maintenance_check_increase (original_time new_time : ℝ) 
  (h1 : original_time = 50)
  (h2 : new_time = 60) :
  (new_time - original_time) / original_time * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l2143_214311


namespace NUMINAMATH_CALUDE_billy_has_ten_fish_l2143_214341

def fish_problem (billy_fish : ℕ) : Prop :=
  let tony_fish := 3 * billy_fish
  let sarah_fish := tony_fish + 5
  let bobby_fish := 2 * sarah_fish
  billy_fish + tony_fish + sarah_fish + bobby_fish = 145

theorem billy_has_ten_fish :
  ∃ (billy_fish : ℕ), fish_problem billy_fish ∧ billy_fish = 10 := by
  sorry

end NUMINAMATH_CALUDE_billy_has_ten_fish_l2143_214341


namespace NUMINAMATH_CALUDE_sequence_properties_l2143_214352

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sequence_properties (a b : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 3 →
  a 4 = 24 →
  b 1 = 0 →
  (∀ n : ℕ, b n + b (n + 1) = a n) →
  (∀ n : ℕ, a n = 3 * 2^(n - 1)) ∧
  (∀ n : ℕ, b n = 2^(n - 1) + (-1)^n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2143_214352


namespace NUMINAMATH_CALUDE_total_problems_l2143_214357

def marvin_yesterday : ℕ := 40

def marvin_today (x : ℕ) : ℕ := 3 * x

def arvin_daily (x : ℕ) : ℕ := 2 * x

theorem total_problems :
  marvin_yesterday + marvin_today marvin_yesterday +
  arvin_daily marvin_yesterday + arvin_daily (marvin_today marvin_yesterday) = 480 := by
  sorry

end NUMINAMATH_CALUDE_total_problems_l2143_214357


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_quadrilateral_inequality_equality_condition_l2143_214385

/-- Theorem: Quadrilateral Inequality
For any quadrilateral with sides a₁, a₂, a₃, a₄ and semi-perimeter s,
the sum of reciprocals of (aᵢ + s) is less than or equal to 2/9 times
the sum of reciprocals of square roots of (s-aᵢ)(s-aⱼ) for all pairs i,j. -/
theorem quadrilateral_inequality (a₁ a₂ a₃ a₄ s : ℝ) 
  (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : a₄ > 0) (h_s : s > 0)
  (h_perimeter : a₁ + a₂ + a₃ + a₄ = 2 * s) : 
  (1 / (a₁ + s) + 1 / (a₂ + s) + 1 / (a₃ + s) + 1 / (a₄ + s)) ≤ 
  (2 / 9) * (1 / Real.sqrt ((s - a₁) * (s - a₂)) + 
             1 / Real.sqrt ((s - a₁) * (s - a₃)) + 
             1 / Real.sqrt ((s - a₁) * (s - a₄)) + 
             1 / Real.sqrt ((s - a₂) * (s - a₃)) + 
             1 / Real.sqrt ((s - a₂) * (s - a₄)) + 
             1 / Real.sqrt ((s - a₃) * (s - a₄))) :=
by sorry

/-- Corollary: Equality condition for the quadrilateral inequality -/
theorem quadrilateral_inequality_equality_condition (a₁ a₂ a₃ a₄ s : ℝ) 
  (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : a₄ > 0) (h_s : s > 0)
  (h_perimeter : a₁ + a₂ + a₃ + a₄ = 2 * s) : 
  (1 / (a₁ + s) + 1 / (a₂ + s) + 1 / (a₃ + s) + 1 / (a₄ + s) = 
   (2 / 9) * (1 / Real.sqrt ((s - a₁) * (s - a₂)) + 
              1 / Real.sqrt ((s - a₁) * (s - a₃)) + 
              1 / Real.sqrt ((s - a₁) * (s - a₄)) + 
              1 / Real.sqrt ((s - a₂) * (s - a₃)) + 
              1 / Real.sqrt ((s - a₂) * (s - a₄)) + 
              1 / Real.sqrt ((s - a₃) * (s - a₄)))) ↔ 
  (a₁ = a₂ ∧ a₂ = a₃ ∧ a₃ = a₄) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_quadrilateral_inequality_equality_condition_l2143_214385


namespace NUMINAMATH_CALUDE_zaras_estimate_l2143_214362

theorem zaras_estimate (x y z : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : z > 0) :
  (x + z) - (y + z) = x - y := by sorry

end NUMINAMATH_CALUDE_zaras_estimate_l2143_214362


namespace NUMINAMATH_CALUDE_arithmetic_progression_probability_l2143_214388

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The total number of possible outcomes when tossing three dice -/
def total_outcomes : ℕ := num_faces ^ 3

/-- A function that checks if three numbers form an arithmetic progression with common difference 2 -/
def is_arithmetic_progression (a b c : ℕ) : Prop :=
  (b = a + 2 ∧ c = b + 2) ∨ (b = a - 2 ∧ c = b - 2) ∨
  (a = b + 2 ∧ c = a + 2) ∨ (c = b + 2 ∧ a = c + 2) ∨
  (a = b - 2 ∧ c = a - 2) ∨ (c = b - 2 ∧ a = c - 2)

/-- The number of favorable outcomes (i.e., outcomes that form an arithmetic progression) -/
def favorable_outcomes : ℕ := 12

/-- The theorem stating the probability of getting an arithmetic progression -/
theorem arithmetic_progression_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 18 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_probability_l2143_214388


namespace NUMINAMATH_CALUDE_proposition_p_sufficient_not_necessary_for_q_l2143_214377

theorem proposition_p_sufficient_not_necessary_for_q :
  (∀ a b : ℝ, a * b ≠ 0 → a^2 + b^2 ≠ 0) ∧
  (∃ a b : ℝ, a^2 + b^2 ≠ 0 ∧ a * b = 0) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_sufficient_not_necessary_for_q_l2143_214377


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l2143_214342

theorem binomial_coefficient_two (n : ℕ) (h : n > 0) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l2143_214342


namespace NUMINAMATH_CALUDE_eBook_readers_count_l2143_214370

/-- The number of eBook readers Anna bought -/
def anna_readers : ℕ := 50

/-- The number of eBook readers John bought initially -/
def john_initial_readers : ℕ := anna_readers - 15

/-- The number of eBook readers John lost -/
def john_lost_readers : ℕ := 3

/-- The number of eBook readers John has after losing some -/
def john_final_readers : ℕ := john_initial_readers - john_lost_readers

/-- The total number of eBook readers John and Anna have together -/
def total_readers : ℕ := anna_readers + john_final_readers

theorem eBook_readers_count : total_readers = 82 := by
  sorry

end NUMINAMATH_CALUDE_eBook_readers_count_l2143_214370


namespace NUMINAMATH_CALUDE_remainder_theorem_l2143_214372

/-- Given a polynomial q(x) satisfying specific conditions, 
    prove properties about its remainder when divided by (x - 3)(x + 2)(x - 4) -/
theorem remainder_theorem (q : ℝ → ℝ) (h1 : q 3 = 2) (h2 : q (-2) = -3) (h3 : q 4 = 6) :
  ∃ (s : ℝ → ℝ), 
    (∀ x, q x = (x - 3) * (x + 2) * (x - 4) * (q x / ((x - 3) * (x + 2) * (x - 4))) + s x) ∧
    (∀ x, s x = 1/2 * x^2 + 1/2 * x - 4) ∧
    (s 5 = 11) :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2143_214372


namespace NUMINAMATH_CALUDE_circle_line_intersection_l2143_214363

/-- The number of distinct points common to the circle x^2 + y^2 = 16 and the vertical line x = 4 is one. -/
theorem circle_line_intersection :
  ∃! p : ℝ × ℝ, (p.1^2 + p.2^2 = 16) ∧ (p.1 = 4) := by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l2143_214363


namespace NUMINAMATH_CALUDE_total_students_agreed_l2143_214369

def third_grade_total : ℕ := 256
def fourth_grade_total : ℕ := 525
def fifth_grade_total : ℕ := 410
def sixth_grade_total : ℕ := 600

def third_grade_percentage : ℚ := 60 / 100
def fourth_grade_percentage : ℚ := 45 / 100
def fifth_grade_percentage : ℚ := 35 / 100
def sixth_grade_percentage : ℚ := 55 / 100

def round_to_nearest (x : ℚ) : ℕ := 
  (x + 1/2).floor.toNat

theorem total_students_agreed : 
  round_to_nearest (third_grade_percentage * third_grade_total) +
  round_to_nearest (fourth_grade_percentage * fourth_grade_total) +
  round_to_nearest (fifth_grade_percentage * fifth_grade_total) +
  round_to_nearest (sixth_grade_percentage * sixth_grade_total) = 864 := by
  sorry

end NUMINAMATH_CALUDE_total_students_agreed_l2143_214369


namespace NUMINAMATH_CALUDE_birds_storks_difference_l2143_214329

/-- Given the initial conditions of birds and storks on a fence, prove that there are 3 more birds than storks. -/
theorem birds_storks_difference :
  let initial_birds : ℕ := 2
  let additional_birds : ℕ := 5
  let storks : ℕ := 4
  let total_birds : ℕ := initial_birds + additional_birds
  total_birds - storks = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_birds_storks_difference_l2143_214329


namespace NUMINAMATH_CALUDE_f_property_l2143_214397

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 3

-- State the theorem
theorem f_property (a b : ℝ) : f a b (-2) = 7 → f a b 2 = -13 := by
  sorry

end NUMINAMATH_CALUDE_f_property_l2143_214397


namespace NUMINAMATH_CALUDE_equation_solution_l2143_214344

theorem equation_solution :
  ∀ y : ℝ, (2012 + y)^2 = 2*y^2 ↔ y = 2012*(Real.sqrt 2 + 1) ∨ y = -2012*(Real.sqrt 2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2143_214344


namespace NUMINAMATH_CALUDE_prob_exactly_one_of_two_independent_l2143_214351

/-- The probability of exactly one of two independent events occurring -/
theorem prob_exactly_one_of_two_independent (p₁ p₂ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) : 
  p₁ * (1 - p₂) + p₂ * (1 - p₁) = 
  (p₁ + p₂) - (p₁ * p₂) := by sorry

end NUMINAMATH_CALUDE_prob_exactly_one_of_two_independent_l2143_214351


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l2143_214382

theorem unique_congruence_in_range : ∃! n : ℤ,
  5 ≤ n ∧ n ≤ 10 ∧ n ≡ 10543 [ZMOD 7] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l2143_214382


namespace NUMINAMATH_CALUDE_line_translation_l2143_214325

/-- Given a line y = -2x + 1, translating it upwards by 2 units results in y = -2x + 3 -/
theorem line_translation (x y : ℝ) : 
  (y = -2*x + 1) → (y + 2 = -2*x + 3) := by sorry

end NUMINAMATH_CALUDE_line_translation_l2143_214325


namespace NUMINAMATH_CALUDE_parabola_directrix_l2143_214346

/-- Given a parabola with equation 16y^2 = x, its directrix equation is x = -1/64 -/
theorem parabola_directrix (x y : ℝ) : 
  (16 * y^2 = x) → (∃ (k : ℝ), k = -1/64 ∧ k = x) := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2143_214346


namespace NUMINAMATH_CALUDE_transportation_puzzle_l2143_214301

def is_valid_assignment (T R A N S P O B K : ℕ) : Prop :=
  T > R ∧ R > A ∧ A > N ∧ N < S ∧ S < P ∧ P < O ∧ O < R ∧ R < T ∧
  T > R ∧ R > O ∧ O < A ∧ A > B ∧ B < K ∧ K < A ∧
  T ≠ R ∧ T ≠ A ∧ T ≠ N ∧ T ≠ S ∧ T ≠ P ∧ T ≠ O ∧ T ≠ B ∧ T ≠ K ∧
  R ≠ A ∧ R ≠ N ∧ R ≠ S ∧ R ≠ P ∧ R ≠ O ∧ R ≠ B ∧ R ≠ K ∧
  A ≠ N ∧ A ≠ S ∧ A ≠ P ∧ A ≠ O ∧ A ≠ B ∧ A ≠ K ∧
  N ≠ S ∧ N ≠ P ∧ N ≠ O ∧ N ≠ B ∧ N ≠ K ∧
  S ≠ P ∧ S ≠ O ∧ S ≠ B ∧ S ≠ K ∧
  P ≠ O ∧ P ≠ B ∧ P ≠ K ∧
  O ≠ B ∧ O ≠ K ∧
  B ≠ K

theorem transportation_puzzle :
  ∃! (T R A N S P O B K : ℕ), is_valid_assignment T R A N S P O B K :=
sorry

end NUMINAMATH_CALUDE_transportation_puzzle_l2143_214301


namespace NUMINAMATH_CALUDE_chocolate_percentage_proof_l2143_214316

/-- Represents the number of each type of chocolate bar -/
def chocolate_count : ℕ := 25

/-- Represents the number of different types of chocolate bars -/
def chocolate_types : ℕ := 4

/-- Calculates the total number of chocolate bars -/
def total_chocolates : ℕ := chocolate_count * chocolate_types

/-- Represents the percentage as a rational number -/
def percentage_per_type : ℚ := chocolate_count / total_chocolates

theorem chocolate_percentage_proof :
  percentage_per_type = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_chocolate_percentage_proof_l2143_214316


namespace NUMINAMATH_CALUDE_mikes_muffins_l2143_214348

/-- The number of muffins in a dozen -/
def dozen : ℕ := 12

/-- The number of boxes needed to pack all muffins -/
def boxes : ℕ := 8

/-- The total number of muffins Mike has -/
def total_muffins : ℕ := boxes * dozen

theorem mikes_muffins : total_muffins = 96 := by
  sorry

end NUMINAMATH_CALUDE_mikes_muffins_l2143_214348


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_sum_of_roots_l2143_214368

theorem sum_of_fractions_equals_sum_of_roots : 
  let T := 1 / (Real.sqrt 10 - Real.sqrt 8) + 
           1 / (Real.sqrt 8 - Real.sqrt 6) + 
           1 / (Real.sqrt 6 - Real.sqrt 4)
  T = (Real.sqrt 10 + 2 * Real.sqrt 8 + 2 * Real.sqrt 6 + 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_sum_of_roots_l2143_214368


namespace NUMINAMATH_CALUDE_subtraction_mistake_l2143_214314

/-- Given two two-digit numbers, if the first number is misread by increasing both digits by 3
    and the incorrect subtraction results in 44, then the correct subtraction equals 11. -/
theorem subtraction_mistake (A B C D : ℕ) : 
  (A < 10) → (B < 10) → (C < 10) → (D < 10) →
  ((10 * (A + 3) + (B + 3)) - (10 * C + D) = 44) →
  ((10 * A + B) - (10 * C + D) = 11) := by
sorry

end NUMINAMATH_CALUDE_subtraction_mistake_l2143_214314


namespace NUMINAMATH_CALUDE_point_movement_l2143_214376

theorem point_movement (A B : ℝ × ℝ) : 
  A = (-3, 2) → 
  B.1 = A.1 + 1 → 
  B.2 = A.2 - 2 → 
  B = (-2, 0) := by
sorry

end NUMINAMATH_CALUDE_point_movement_l2143_214376


namespace NUMINAMATH_CALUDE_julio_is_ten_l2143_214321

-- Define the ages as natural numbers
def zipporah_age : ℕ := 7
def dina_age : ℕ := 51 - zipporah_age
def julio_age : ℕ := 54 - dina_age

-- State the theorem
theorem julio_is_ten : julio_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_julio_is_ten_l2143_214321


namespace NUMINAMATH_CALUDE_fence_overlap_calculation_l2143_214386

theorem fence_overlap_calculation (num_planks : ℕ) (plank_length : ℝ) (total_length : ℝ) 
  (h1 : num_planks = 25)
  (h2 : plank_length = 30)
  (h3 : total_length = 690) :
  ∃ overlap : ℝ, 
    overlap = 2.5 ∧ 
    total_length = (13 * plank_length) + (12 * (plank_length - 2 * overlap)) :=
by sorry

end NUMINAMATH_CALUDE_fence_overlap_calculation_l2143_214386


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2143_214327

def repeating_decimal : ℚ := 0.157142857142857

theorem repeating_decimal_as_fraction :
  repeating_decimal = 10690 / 68027 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2143_214327


namespace NUMINAMATH_CALUDE_exists_divisible_pair_l2143_214330

/-- A function that checks if a number is three-digit -/
def isThreeDigit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

/-- A function that checks if a number is two-digit -/
def isTwoDigit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

/-- A function that checks if a number uses only the digits 1, 2, 3, 4, 5 -/
def usesOnlyGivenDigits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [1, 2, 3, 4, 5]

/-- The main theorem -/
theorem exists_divisible_pair :
  ∃ (a b : ℕ),
    isThreeDigit a ∧
    isTwoDigit b ∧
    usesOnlyGivenDigits a ∧
    usesOnlyGivenDigits b ∧
    a % b = 0 :=
  sorry

end NUMINAMATH_CALUDE_exists_divisible_pair_l2143_214330


namespace NUMINAMATH_CALUDE_smallest_marble_count_l2143_214396

/-- Represents the number of marbles of each color -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the total number of marbles -/
def total_marbles (mc : MarbleCount) : ℕ :=
  mc.red + mc.white + mc.blue + mc.green

/-- Checks if the given marble count satisfies the equal probability conditions -/
def satisfies_conditions (mc : MarbleCount) : Prop :=
  let r := mc.red
  let w := mc.white
  let b := mc.blue
  let g := mc.green
  (r * (r - 1) * (r - 2) * (r - 3)) / 24 = 
    (w * r * (r - 1) * (r - 2)) / 6 ∧
  (r * (r - 1) * (r - 2) * (r - 3)) / 24 = 
    (w * b * r * (r - 1)) / 2 ∧
  (w * b * r * (r - 1)) / 2 = 
    w * b * g * r

theorem smallest_marble_count : 
  ∃ (mc : MarbleCount), 
    satisfies_conditions mc ∧ 
    total_marbles mc = 21 ∧ 
    (∀ (mc' : MarbleCount), satisfies_conditions mc' → total_marbles mc' ≥ 21) := by
  sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l2143_214396


namespace NUMINAMATH_CALUDE_triangle_n_values_l2143_214371

theorem triangle_n_values :
  let valid_n (n : ℕ) : Prop :=
    3*n + 15 > 3*n + 10 ∧ 
    3*n + 10 > 4*n ∧ 
    4*n + (3*n + 10) > 3*n + 15 ∧ 
    4*n + (3*n + 15) > 3*n + 10 ∧ 
    (3*n + 10) + (3*n + 15) > 4*n
  ∃! (s : Finset ℕ), (∀ n ∈ s, valid_n n) ∧ s.card = 8 :=
by sorry

end NUMINAMATH_CALUDE_triangle_n_values_l2143_214371


namespace NUMINAMATH_CALUDE_original_number_proof_l2143_214335

theorem original_number_proof (x : ℝ) : x * 1.1 = 550 ↔ x = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2143_214335


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_greater_than_one_l2143_214354

-- Define the function f(x) = 2ax^2 - x - 1
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - x - 1

-- State the theorem
theorem unique_solution_implies_a_greater_than_one :
  ∀ a : ℝ, (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ f a x = 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_greater_than_one_l2143_214354


namespace NUMINAMATH_CALUDE_complex_modulus_inequality_l2143_214332

theorem complex_modulus_inequality (x y : ℝ) : 
  let z : ℂ := Complex.mk x y
  ‖z‖ ≤ |x| + |y| := by sorry

end NUMINAMATH_CALUDE_complex_modulus_inequality_l2143_214332


namespace NUMINAMATH_CALUDE_break_even_price_correct_l2143_214319

/-- The price per kilogram to sell fruits without loss or profit -/
def break_even_price : ℝ := 2.6

/-- The price per jin that results in a loss -/
def loss_price : ℝ := 1.2

/-- The price per jin that results in a profit -/
def profit_price : ℝ := 1.5

/-- The amount of loss when selling at loss_price -/
def loss_amount : ℝ := 4

/-- The amount of profit when selling at profit_price -/
def profit_amount : ℝ := 8

/-- Conversion factor from jin to kilogram -/
def jin_to_kg : ℝ := 0.5

theorem break_even_price_correct :
  ∃ (weight : ℝ),
    weight * (break_even_price * jin_to_kg) = weight * loss_price + loss_amount ∧
    weight * (break_even_price * jin_to_kg) = weight * profit_price - profit_amount :=
by sorry

end NUMINAMATH_CALUDE_break_even_price_correct_l2143_214319


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l2143_214350

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation (1-i)z = 2i
def equation (z : ℂ) : Prop := (1 - i) * z = 2 * i

-- Define what it means for a complex number to be in the second quadrant
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- State the theorem
theorem z_in_second_quadrant :
  ∃ z : ℂ, equation z ∧ in_second_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l2143_214350


namespace NUMINAMATH_CALUDE_total_sum_is_71_rupees_l2143_214373

/-- Calculates the total sum of money in rupees given the number of 20 paise and 25 paise coins -/
def total_sum_rupees (total_coins : ℕ) (coins_20_paise : ℕ) : ℚ :=
  let coins_25_paise := total_coins - coins_20_paise
  let sum_20_paise := (coins_20_paise : ℚ) * (20 : ℚ) / 100
  let sum_25_paise := (coins_25_paise : ℚ) * (25 : ℚ) / 100
  sum_20_paise + sum_25_paise

/-- Theorem stating that given 324 total coins with 200 coins of 20 paise, the total sum is 71 rupees -/
theorem total_sum_is_71_rupees :
  total_sum_rupees 324 200 = 71 := by
  sorry

end NUMINAMATH_CALUDE_total_sum_is_71_rupees_l2143_214373


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l2143_214393

def selling_price : ℝ := 1110
def cost_price : ℝ := 925

theorem shopkeeper_profit_percentage :
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l2143_214393


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2143_214305

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {-2, -1, 0}

theorem intersection_A_complement_B (x : Int) : 
  x ∈ (A ∩ (U \ B)) ↔ x = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2143_214305


namespace NUMINAMATH_CALUDE_water_intake_glasses_l2143_214355

/-- Calculates the number of glasses of water needed to meet a daily water intake goal -/
theorem water_intake_glasses (daily_goal : ℝ) (glass_capacity : ℝ) : 
  daily_goal = 1.5 → glass_capacity = 0.250 → (daily_goal * 1000) / glass_capacity = 6 := by
  sorry

end NUMINAMATH_CALUDE_water_intake_glasses_l2143_214355


namespace NUMINAMATH_CALUDE_money_collection_l2143_214326

theorem money_collection (households_per_day : ℕ) (days : ℕ) (total_amount : ℕ) :
  households_per_day = 20 →
  days = 5 →
  total_amount = 2000 →
  (households_per_day * days) / 2 * (total_amount / ((households_per_day * days) / 2)) = 40 :=
by sorry

end NUMINAMATH_CALUDE_money_collection_l2143_214326


namespace NUMINAMATH_CALUDE_polynomial_equality_l2143_214387

theorem polynomial_equality (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (3 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2143_214387


namespace NUMINAMATH_CALUDE_special_triangle_properties_l2143_214312

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Given conditions
  c_eq : c = 7/2
  area_eq : 1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2
  tan_eq : Real.tan A + Real.tan B = Real.sqrt 3 * (Real.tan A * Real.tan B - 1)

/-- Theorem about the properties of the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  t.C = π/3 ∧ t.a + t.b = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l2143_214312


namespace NUMINAMATH_CALUDE_line_through_parabola_vertex_l2143_214375

theorem line_through_parabola_vertex (a : ℝ) : 
  (∃! (x y : ℝ), y = x^2 + a^2 ∧ y = 2*x + a ∧ ∀ (x' : ℝ), x'^2 + a^2 ≥ y) ↔ (a = 0 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_line_through_parabola_vertex_l2143_214375


namespace NUMINAMATH_CALUDE_line_equation_proof_l2143_214379

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if it satisfies the line's equation -/
def lies_on (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_equation_proof (l : Line) :
  parallel l (Line.mk 2 (-1) 1) →
  lies_on (Point.mk 1 2) l →
  l = Line.mk 2 (-1) 0 :=
by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2143_214379


namespace NUMINAMATH_CALUDE_math_club_teams_l2143_214384

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of girls in the math club -/
def num_girls : ℕ := 4

/-- The number of boys in the math club -/
def num_boys : ℕ := 6

/-- The number of girls to be selected for each team -/
def girls_per_team : ℕ := 2

/-- The number of boys to be selected for each team -/
def boys_per_team : ℕ := 2

theorem math_club_teams : 
  (choose num_girls girls_per_team) * (choose num_boys boys_per_team) = 90 := by
  sorry

end NUMINAMATH_CALUDE_math_club_teams_l2143_214384


namespace NUMINAMATH_CALUDE_circle_tangency_radius_l2143_214349

theorem circle_tangency_radius (r_P r_Q r_R : ℝ) : 
  r_P = 4 ∧ 
  r_Q = 4 * r_R ∧ 
  r_P > r_Q ∧ 
  r_P > r_R ∧
  r_Q > r_R ∧
  r_P = r_Q + r_R →
  r_Q = 16 ∧ 
  r_Q = Real.sqrt 256 - 0 := by
sorry

end NUMINAMATH_CALUDE_circle_tangency_radius_l2143_214349


namespace NUMINAMATH_CALUDE_triangle_hyperbola_ratio_l2143_214378

-- Define the right triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let (xA, yA) := A
  let (xB, yB) := B
  let (xC, yC) := C
  (xB - xA)^2 + (yB - yA)^2 = 3^2 ∧
  (xC - xA)^2 + (yC - yA)^2 = 1^2 ∧
  (xB - xA) * (xC - xA) + (yB - yA) * (yC - yA) = 0

-- Define the hyperbola passing through A and intersecting AB at D
def Hyperbola (A B C D : ℝ × ℝ) (a b : ℝ) : Prop :=
  let (xA, yA) := A
  let (xB, yB) := B
  let (xC, yC) := C
  let (xD, yD) := D
  a > 0 ∧ b > 0 ∧
  xA^2 / a^2 - yA^2 / b^2 = 1 ∧
  xD^2 / a^2 - yD^2 / b^2 = 1 ∧
  (xD - xA) * (yB - yA) = (yD - yA) * (xB - xA)

-- Theorem statement
theorem triangle_hyperbola_ratio 
  (A B C D : ℝ × ℝ) (a b : ℝ) :
  Triangle A B C → Hyperbola A B C D a b →
  let (xA, yA) := A
  let (xB, yB) := B
  let (xD, yD) := D
  Real.sqrt ((xD - xA)^2 + (yD - yA)^2) / Real.sqrt ((xB - xD)^2 + (yB - yD)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_hyperbola_ratio_l2143_214378


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l2143_214391

theorem power_mod_seventeen : 7^2048 % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l2143_214391


namespace NUMINAMATH_CALUDE_profit_equation_l2143_214313

/-- Given a profit equation P = (1/m)S - (1/n)C, prove that P = (m-n)/(mn) * S -/
theorem profit_equation (m n : ℝ) (m_ne_zero : m ≠ 0) (n_ne_zero : n ≠ 0) :
  ∀ (S C P : ℝ), P = (1/m) * S - (1/n) * C → P = (m-n)/(m*n) * S :=
by sorry

end NUMINAMATH_CALUDE_profit_equation_l2143_214313


namespace NUMINAMATH_CALUDE_function_value_at_negative_one_l2143_214304

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_value_at_negative_one
  (h1 : ∀ x : ℝ, f (x + 2009) = -f (x + 2008))
  (h2 : f 2009 = -2009) :
  f (-1) = -2009 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_one_l2143_214304


namespace NUMINAMATH_CALUDE_mork_tax_rate_l2143_214337

theorem mork_tax_rate (mork_income : ℝ) (mork_rate : ℝ) : 
  mork_rate > 0 →
  mork_income > 0 →
  (mork_rate / 100 * mork_income + 0.15 * (4 * mork_income)) / (5 * mork_income) = 0.21 →
  mork_rate = 45 := by
sorry

end NUMINAMATH_CALUDE_mork_tax_rate_l2143_214337


namespace NUMINAMATH_CALUDE_min_jumps_proof_l2143_214300

/-- The distance of each jump in millimeters -/
def jump_distance : ℝ := 19

/-- The distance between points A and B in centimeters -/
def total_distance : ℝ := 1812

/-- The minimum number of jumps required -/
def min_jumps : ℕ := 954

/-- Theorem stating the minimum number of jumps required -/
theorem min_jumps_proof :
  ∃ (n : ℕ), n = min_jumps ∧ 
  (n : ℝ) * jump_distance ≥ total_distance * 10 ∧
  ∀ (m : ℕ), (m : ℝ) * jump_distance ≥ total_distance * 10 → m ≥ n :=
sorry

end NUMINAMATH_CALUDE_min_jumps_proof_l2143_214300


namespace NUMINAMATH_CALUDE_money_distribution_inconsistency_l2143_214383

/-- Represents the money distribution problem with aunts and children --/
theorem money_distribution_inconsistency 
  (jade_money : ℕ) 
  (julia_money : ℕ) 
  (jack_money : ℕ) 
  (john_money : ℕ) 
  (jane_money : ℕ) 
  (total_after : ℕ) 
  (aunt_mary_gift : ℕ) 
  (aunt_susan_gift : ℕ) 
  (h1 : jade_money = 38)
  (h2 : julia_money = jade_money / 2)
  (h3 : jack_money = 12)
  (h4 : john_money = 15)
  (h5 : jane_money = 20)
  (h6 : total_after = 225)
  (h7 : aunt_mary_gift = 65)
  (h8 : aunt_susan_gift = 70) : 
  ¬(∃ (aunt_lucy_gift : ℕ) (individual_gift : ℕ),
    jade_money + julia_money + jack_money + john_money + jane_money + 
    aunt_mary_gift + aunt_susan_gift + aunt_lucy_gift = total_after ∧
    aunt_mary_gift + aunt_susan_gift + aunt_lucy_gift = 5 * individual_gift) :=
sorry


end NUMINAMATH_CALUDE_money_distribution_inconsistency_l2143_214383


namespace NUMINAMATH_CALUDE_triangle_side_length_simplification_l2143_214302

theorem triangle_side_length_simplification 
  (a b c : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) : 
  |a + b - c| - |b - a - c| = 2*b - 2*c := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_simplification_l2143_214302


namespace NUMINAMATH_CALUDE_fruit_cost_theorem_l2143_214345

def fruit_prices (o g w f : ℝ) : Prop :=
  o + g + w + f = 24 ∧ f = 3 * o ∧ w = o - 2 * g

theorem fruit_cost_theorem :
  ∀ o g w f : ℝ, fruit_prices o g w f → g + w = 4.8 :=
by sorry

end NUMINAMATH_CALUDE_fruit_cost_theorem_l2143_214345


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2143_214394

/-- The sum of the geometric series 15 + 15r + 15r^2 + 15r^3 + ... for -1 < r < 1 -/
noncomputable def T (r : ℝ) : ℝ :=
  15 / (1 - r)

/-- For -1 < b < 1, if T(b)T(-b) = 3240, then T(b) + T(-b) = 432 -/
theorem geometric_series_sum (b : ℝ) (hb1 : -1 < b) (hb2 : b < 1) 
    (h : T b * T (-b) = 3240) : T b + T (-b) = 432 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2143_214394


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2143_214315

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- The dimensions of the carton -/
def carton_dimensions : Dimensions := ⟨30, 42, 60⟩

/-- The dimensions of a soap box -/
def soap_box_dimensions : Dimensions := ⟨7, 6, 5⟩

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (volume carton_dimensions) / (volume soap_box_dimensions) = 360 := by
  sorry

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2143_214315


namespace NUMINAMATH_CALUDE_sum_three_numbers_l2143_214307

/-- Given three numbers a, b, and c, and a value T, satisfying the following conditions:
    1. a + b + c = 84
    2. a - 5 = T
    3. b + 9 = T
    4. 5 * c = T
    Prove that T = 40 -/
theorem sum_three_numbers (a b c T : ℝ) 
  (sum_eq : a + b + c = 84)
  (a_minus : a - 5 = T)
  (b_plus : b + 9 = T)
  (c_times : 5 * c = T) : 
  T = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_three_numbers_l2143_214307


namespace NUMINAMATH_CALUDE_reservoir_after_storm_l2143_214308

/-- Represents the capacity of the reservoir in billion gallons -/
def reservoir_capacity : ℝ := 400

/-- Represents the initial amount of water in the reservoir in billion gallons -/
def initial_water : ℝ := 200

/-- Represents the amount of water added by the storm in billion gallons -/
def storm_water : ℝ := 120

/-- Theorem stating that the reservoir is 80% full after the storm -/
theorem reservoir_after_storm :
  (initial_water + storm_water) / reservoir_capacity = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_after_storm_l2143_214308


namespace NUMINAMATH_CALUDE_divisibility_of_quadratic_form_l2143_214331

theorem divisibility_of_quadratic_form (p a b k : ℤ) : 
  Prime p → 
  p = 3*k + 2 → 
  p ∣ (a^2 + a*b + b^2) → 
  p ∣ a ∧ p ∣ b :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_quadratic_form_l2143_214331


namespace NUMINAMATH_CALUDE_min_value_theorem_l2143_214322

theorem min_value_theorem (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + 2*y = 1) :
  (1 / (x + 1) + 1 / y) ≥ (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2143_214322


namespace NUMINAMATH_CALUDE_car_selection_average_l2143_214364

theorem car_selection_average (num_cars : ℕ) (num_clients : ℕ) (selections_per_client : ℕ) 
  (h1 : num_cars = 18) 
  (h2 : num_clients = 18) 
  (h3 : selections_per_client = 3) : 
  (num_clients * selections_per_client) / num_cars = 3 := by
  sorry

end NUMINAMATH_CALUDE_car_selection_average_l2143_214364


namespace NUMINAMATH_CALUDE_parabola_and_tangent_lines_l2143_214395

-- Define the parabola
structure Parabola where
  -- Standard form equation: x² = 2py
  p : ℝ
  -- Vertex at origin
  vertex : (ℝ × ℝ) := (0, 0)
  -- Focus on y-axis
  focus : (ℝ × ℝ) := (0, p)

-- Define a point on the parabola
def point_on_parabola (par : Parabola) (x y : ℝ) : Prop :=
  x^2 = 2 * par.p * y

-- Define a line
structure Line where
  m : ℝ
  b : ℝ

-- Define a point on a line
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  y = l.m * x + l.b

-- Define when a line intersects a parabola at a single point
def single_intersection (par : Parabola) (l : Line) : Prop :=
  ∃! (x y : ℝ), point_on_parabola par x y ∧ point_on_line l x y

-- Theorem statement
theorem parabola_and_tangent_lines :
  ∃ (par : Parabola),
    -- Parabola passes through (2, 1)
    point_on_parabola par 2 1 ∧
    -- Standard equation is x² = 4y
    par.p = 2 ∧
    -- Lines x = 2 and x - y - 1 = 0 are the only lines through (2, 1)
    -- that intersect the parabola at a single point
    (∀ (l : Line),
      point_on_line l 2 1 →
      single_intersection par l ↔ (l.m = 0 ∧ l.b = 2) ∨ (l.m = 1 ∧ l.b = -1)) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_tangent_lines_l2143_214395


namespace NUMINAMATH_CALUDE_maaza_liters_l2143_214398

theorem maaza_liters (pepsi sprite cans : ℕ) (h1 : pepsi = 144) (h2 : sprite = 368) (h3 : cans = 281) :
  ∃ (M : ℕ), M + pepsi + sprite = cans * (M + pepsi + sprite) / cans ∧
  ∀ (M' : ℕ), M' + pepsi + sprite = cans * (M' + pepsi + sprite) / cans → M ≤ M' :=
by sorry

end NUMINAMATH_CALUDE_maaza_liters_l2143_214398


namespace NUMINAMATH_CALUDE_stacy_berries_l2143_214360

theorem stacy_berries (total : ℕ) (x : ℕ) : total = 1100 → x + 2*x + 8*x = total → 8*x = 800 :=
by
  sorry

end NUMINAMATH_CALUDE_stacy_berries_l2143_214360


namespace NUMINAMATH_CALUDE_students_neither_art_nor_music_l2143_214340

theorem students_neither_art_nor_music 
  (total : ℕ) (art : ℕ) (music : ℕ) (both : ℕ) :
  total = 75 →
  art = 45 →
  music = 50 →
  both = 30 →
  total - (art + music - both) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_students_neither_art_nor_music_l2143_214340


namespace NUMINAMATH_CALUDE_not_always_prime_l2143_214309

theorem not_always_prime : ∃ n : ℤ, ¬(Nat.Prime (Int.natAbs (n^2 + n + 41))) := by sorry

end NUMINAMATH_CALUDE_not_always_prime_l2143_214309


namespace NUMINAMATH_CALUDE_greatest_common_divisor_420_90_under_50_l2143_214303

theorem greatest_common_divisor_420_90_under_50 : 
  ∀ n : ℕ, n ∣ 420 ∧ n < 50 ∧ n ∣ 90 → n ≤ 30 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_420_90_under_50_l2143_214303


namespace NUMINAMATH_CALUDE_average_monthly_sales_l2143_214374

def january_sales : ℝ := 110
def february_sales : ℝ := 90
def march_sales : ℝ := 70
def april_sales : ℝ := 130
def may_sales : ℝ := 50
def total_months : ℕ := 5

def total_sales : ℝ := january_sales + february_sales + march_sales + april_sales + may_sales

theorem average_monthly_sales :
  total_sales / total_months = 90 := by sorry

end NUMINAMATH_CALUDE_average_monthly_sales_l2143_214374


namespace NUMINAMATH_CALUDE_zoo_line_theorem_l2143_214347

/-- The number of ways to arrange 6 people in a line with specific conditions -/
def zoo_line_arrangements : ℕ := 24

/-- Two fathers in a group of 6 people -/
def fathers : ℕ := 2

/-- Two mothers in a group of 6 people -/
def mothers : ℕ := 2

/-- Two children in a group of 6 people -/
def children : ℕ := 2

/-- Total number of people in the group -/
def total_people : ℕ := fathers + mothers + children

theorem zoo_line_theorem :
  zoo_line_arrangements = 24 :=
sorry

end NUMINAMATH_CALUDE_zoo_line_theorem_l2143_214347


namespace NUMINAMATH_CALUDE_concert_attendance_l2143_214334

theorem concert_attendance (adults : ℕ) 
  (h1 : 3 * adults = children)
  (h2 : 7 * adults + 3 * children = 6000) :
  adults + children = 1500 :=
by sorry

end NUMINAMATH_CALUDE_concert_attendance_l2143_214334


namespace NUMINAMATH_CALUDE_not_perfect_cube_l2143_214353

theorem not_perfect_cube (n : ℕ) : ¬ ∃ k : ℤ, 2^(2^n) + 1 = k^3 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_cube_l2143_214353


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l2143_214359

theorem lcm_gcd_product (a b : ℕ) (ha : a = 30) (hb : b = 75) :
  Nat.lcm a b * Nat.gcd a b = 2250 ∧ Nat.lcm a b * Nat.gcd a b = a * b := by
  sorry

#check lcm_gcd_product

end NUMINAMATH_CALUDE_lcm_gcd_product_l2143_214359


namespace NUMINAMATH_CALUDE_perfect_square_floor_l2143_214366

theorem perfect_square_floor (a b : ℝ) : 
  (∀ n : ℕ+, ∃ k : ℕ, ⌊a * n + b⌋ = k^2) ↔ 
  (a = 0 ∧ ∃ k : ℕ, ∃ u : ℝ, b = k^2 + u ∧ 0 ≤ u ∧ u < 1) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_floor_l2143_214366


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2143_214392

theorem quadratic_inequality (a x : ℝ) : 
  a * x^2 - (a + 2) * x + 2 < 0 ↔ 
    ((a < 0 ∧ (x < 2/a ∨ x > 1)) ∨
     (a = 0 ∧ x > 1) ∨
     (0 < a ∧ a < 2 ∧ 1 < x ∧ x < 2/a) ∨
     (a > 2 ∧ 2/a < x ∧ x < 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2143_214392


namespace NUMINAMATH_CALUDE_divisibility_conditions_divisibility_conditions_2_divisibility_conditions_3_l2143_214365

theorem divisibility_conditions (n : ℤ) :
  (∃ k : ℤ, n = 225 * k + 99) ↔ (9 ∣ n ∧ 25 ∣ (n + 1)) :=
sorry

theorem divisibility_conditions_2 (n : ℤ) :
  (∃ k : ℤ, n = 3465 * k + 1649) ↔ (21 ∣ n ∧ 165 ∣ (n + 1)) :=
sorry

theorem divisibility_conditions_3 (n : ℤ) :
  (∃ m : ℤ, n = 900 * m + 774) ↔ (9 ∣ n ∧ 25 ∣ (n + 1) ∧ 4 ∣ (n + 2)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_conditions_divisibility_conditions_2_divisibility_conditions_3_l2143_214365


namespace NUMINAMATH_CALUDE_part_one_part_two_l2143_214356

/-- Definition of proposition p -/
def p (x a : ℝ) : Prop := x^2 - 2*a*x - 3*a^2 < 0

/-- Definition of proposition q -/
def q (x : ℝ) : Prop := 2 ≤ x ∧ x < 4

/-- Theorem for part (1) -/
theorem part_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 ≤ x ∧ x < 3) :=
sorry

/-- Theorem for part (2) -/
theorem part_two :
  ∀ a : ℝ, (a > 0 ∧ (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x)) ↔ (4/3 ≤ a) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2143_214356


namespace NUMINAMATH_CALUDE_pocket_money_mode_and_median_l2143_214336

def pocket_money : List ℕ := [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 5, 5, 5, 6]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℕ := sorry

theorem pocket_money_mode_and_median :
  mode pocket_money = 2 ∧ median pocket_money = 3 := by sorry

end NUMINAMATH_CALUDE_pocket_money_mode_and_median_l2143_214336
