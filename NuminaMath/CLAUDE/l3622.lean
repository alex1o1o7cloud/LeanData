import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_farthest_apart_l3622_362223

/-- The quadratic equation x^2 - 4ax + 5a^2 - 6a = 0 has roots that are farthest apart when a = 3 -/
theorem quadratic_roots_farthest_apart (a : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 4*a*x + 5*a^2 - 6*a
  let discriminant := 4*a*(6 - a)
  (∀ b : ℝ, discriminant ≥ 4*b*(6 - b)) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_farthest_apart_l3622_362223


namespace NUMINAMATH_CALUDE_dave_tickets_for_toys_l3622_362271

/-- The number of tickets Dave used to buy toys -/
def tickets_for_toys : ℕ := 12

/-- The number of tickets Dave had initially -/
def initial_tickets : ℕ := 19

/-- The number of tickets Dave used to buy clothes -/
def tickets_for_clothes : ℕ := 7

/-- The difference between tickets used for toys and clothes -/
def ticket_difference : ℕ := 5

theorem dave_tickets_for_toys :
  tickets_for_toys = tickets_for_clothes + ticket_difference :=
by sorry

end NUMINAMATH_CALUDE_dave_tickets_for_toys_l3622_362271


namespace NUMINAMATH_CALUDE_digit_sum_multiple_of_nine_l3622_362268

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Theorem: If a number n and 3n have the same digit sum, then n is divisible by 9 -/
theorem digit_sum_multiple_of_nine (n : ℕ) : digitSum n = digitSum (3 * n) → 9 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_multiple_of_nine_l3622_362268


namespace NUMINAMATH_CALUDE_sin_two_phi_l3622_362287

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_phi_l3622_362287


namespace NUMINAMATH_CALUDE_milburg_population_l3622_362200

theorem milburg_population :
  let children : ℕ := 2987
  let adults : ℕ := 2269
  let total_population : ℕ := children + adults
  total_population = 5256 := by sorry

end NUMINAMATH_CALUDE_milburg_population_l3622_362200


namespace NUMINAMATH_CALUDE_two_digit_number_square_equals_cube_of_digit_sum_l3622_362253

theorem two_digit_number_square_equals_cube_of_digit_sum :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧
  (∃ a b : ℕ, a ≠ b ∧ a < 10 ∧ b < 10 ∧ n = 10 * a + b) ∧
  n^2 = (n / 10 + n % 10)^3 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_square_equals_cube_of_digit_sum_l3622_362253


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l3622_362227

def phone_cost : ℝ := 2
def service_plan_monthly_cost : ℝ := 7
def service_plan_duration : ℕ := 4
def insurance_fee : ℝ := 10
def first_phone_tax_rate : ℝ := 0.05
def second_phone_tax_rate : ℝ := 0.03
def service_plan_discount_rate : ℝ := 0.20
def num_phones : ℕ := 2

def total_cost : ℝ :=
  let phone_total := phone_cost * num_phones
  let service_plan_total := service_plan_monthly_cost * service_plan_duration * num_phones
  let service_plan_discount := service_plan_total * service_plan_discount_rate
  let discounted_service_plan := service_plan_total - service_plan_discount
  let tax_total := (first_phone_tax_rate * phone_cost) + (second_phone_tax_rate * phone_cost)
  phone_total + discounted_service_plan + tax_total + insurance_fee

theorem total_cost_is_correct : total_cost = 58.96 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l3622_362227


namespace NUMINAMATH_CALUDE_shortest_side_in_triangle_l3622_362288

/-- Given a triangle with side lengths a, b, and c, if a^2 + b^2 > 5c^2, then c is the length of the shortest side. -/
theorem shortest_side_in_triangle (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_inequality : a^2 + b^2 > 5*c^2) : 
  c ≤ a ∧ c ≤ b :=
sorry

end NUMINAMATH_CALUDE_shortest_side_in_triangle_l3622_362288


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3622_362279

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 + 5x = 0 -/
def f (x : ℝ) : ℝ := x^2 + 5*x

/-- Theorem: f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f :=
sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3622_362279


namespace NUMINAMATH_CALUDE_initial_shoe_pairs_l3622_362224

/-- Represents the number of shoes in a pair -/
def shoes_per_pair : ℕ := 2

/-- Represents the number of individual shoes lost -/
def shoes_lost : ℕ := 9

/-- Represents the number of matching pairs left after losing shoes -/
def matching_pairs_left : ℕ := 15

/-- Represents the initial number of pairs of shoes -/
def initial_pairs : ℕ := matching_pairs_left + shoes_lost

theorem initial_shoe_pairs :
  initial_pairs = 24 :=
by sorry

end NUMINAMATH_CALUDE_initial_shoe_pairs_l3622_362224


namespace NUMINAMATH_CALUDE_shooter_probability_l3622_362204

theorem shooter_probability (p : ℝ) (n k : ℕ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  let prob_hit := p
  let num_shots := n
  let num_hits := k
  Nat.choose num_shots num_hits * prob_hit ^ num_hits * (1 - prob_hit) ^ (num_shots - num_hits) =
  Nat.choose 5 4 * (0.8 : ℝ) ^ 4 * (0.2 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_shooter_probability_l3622_362204


namespace NUMINAMATH_CALUDE_smallest_m_for_square_inequality_l3622_362244

theorem smallest_m_for_square_inequality : ∃ (m : ℕ+), 
  (m = 16144325) ∧ 
  (∀ (n : ℕ+), n ≥ m → ∃ (l : ℕ+), (n : ℝ) < (l : ℝ)^2 ∧ (l : ℝ)^2 < (1 + 1/2009) * (n : ℝ)) ∧
  (∀ (m' : ℕ+), m' < m → ∃ (n : ℕ+), n ≥ m' ∧ ∀ (l : ℕ+), ((n : ℝ) ≥ (l : ℝ)^2 ∨ (l : ℝ)^2 ≥ (1 + 1/2009) * (n : ℝ))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_square_inequality_l3622_362244


namespace NUMINAMATH_CALUDE_shortest_horizontal_distance_l3622_362206

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - x - 6

/-- Theorem stating the shortest horizontal distance -/
theorem shortest_horizontal_distance :
  ∃ (x₁ x₂ : ℝ),
    f x₁ = 6 ∧
    f x₂ = -6 ∧
    ∀ (y₁ y₂ : ℝ),
      f y₁ = 6 →
      f y₂ = -6 →
      |x₁ - x₂| ≤ |y₁ - y₂| ∧
      |x₁ - x₂| = 3 :=
sorry

end NUMINAMATH_CALUDE_shortest_horizontal_distance_l3622_362206


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l3622_362280

/-- In a Cartesian coordinate system, the coordinates of a point (-1, 2) with respect to the origin are (-1, 2). -/
theorem coordinates_wrt_origin (x y : ℝ) : x = -1 ∧ y = 2 → (x, y) = (-1, 2) := by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l3622_362280


namespace NUMINAMATH_CALUDE_even_sum_condition_l3622_362231

theorem even_sum_condition (m n : ℤ) : 
  (∃ (k l : ℤ), m = 2 * k ∧ n = 2 * l → ∃ (p : ℤ), m + n = 2 * p) ∧ 
  (∃ (m n : ℤ), ∃ (q : ℤ), m + n = 2 * q ∧ ¬(∃ (r s : ℤ), m = 2 * r ∧ n = 2 * s)) :=
by sorry

end NUMINAMATH_CALUDE_even_sum_condition_l3622_362231


namespace NUMINAMATH_CALUDE_fano_plane_properties_l3622_362210

/-- A point in the Fano plane. -/
inductive Point
| P1 | P2 | P3 | P4 | P5 | P6 | P7

/-- A line in the Fano plane. -/
inductive Line
| L1 | L2 | L3 | L4 | L5 | L6 | L7

/-- The incidence relation between points and lines in the Fano plane. -/
def incidence : Point → Line → Prop
| Point.P1, Line.L1 => True
| Point.P1, Line.L2 => True
| Point.P1, Line.L3 => True
| Point.P2, Line.L1 => True
| Point.P2, Line.L4 => True
| Point.P2, Line.L5 => True
| Point.P3, Line.L1 => True
| Point.P3, Line.L6 => True
| Point.P3, Line.L7 => True
| Point.P4, Line.L2 => True
| Point.P4, Line.L4 => True
| Point.P4, Line.L6 => True
| Point.P5, Line.L2 => True
| Point.P5, Line.L5 => True
| Point.P5, Line.L7 => True
| Point.P6, Line.L3 => True
| Point.P6, Line.L4 => True
| Point.P6, Line.L7 => True
| Point.P7, Line.L3 => True
| Point.P7, Line.L5 => True
| Point.P7, Line.L6 => True
| _, _ => False

/-- The theorem stating that the Fano plane satisfies the required properties. -/
theorem fano_plane_properties :
  (∀ l : Line, ∃! (p1 p2 p3 : Point), p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    incidence p1 l ∧ incidence p2 l ∧ incidence p3 l ∧
    (∀ p : Point, incidence p l → p = p1 ∨ p = p2 ∨ p = p3)) ∧
  (∀ p : Point, ∃! (l1 l2 l3 : Line), l1 ≠ l2 ∧ l1 ≠ l3 ∧ l2 ≠ l3 ∧
    incidence p l1 ∧ incidence p l2 ∧ incidence p l3 ∧
    (∀ l : Line, incidence p l → l = l1 ∨ l = l2 ∨ l = l3)) :=
by sorry

end NUMINAMATH_CALUDE_fano_plane_properties_l3622_362210


namespace NUMINAMATH_CALUDE_doubled_roots_quadratic_l3622_362273

theorem doubled_roots_quadratic (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 5 * x₁ - 8 = 0 ∧ 2 * x₂^2 - 5 * x₂ - 8 = 0) →
  ((2 * x₁)^2 - 5 * (2 * x₁) - 16 = 0 ∧ (2 * x₂)^2 - 5 * (2 * x₂) - 16 = 0) :=
by sorry

end NUMINAMATH_CALUDE_doubled_roots_quadratic_l3622_362273


namespace NUMINAMATH_CALUDE_discount_difference_is_978_75_l3622_362228

/-- The initial invoice amount -/
def initial_amount : ℝ := 15000

/-- The single discount rate -/
def single_discount_rate : ℝ := 0.5

/-- The successive discount rates -/
def successive_discount_rates : List ℝ := [0.3, 0.15, 0.05]

/-- Calculate the amount after applying a single discount -/
def amount_after_single_discount (amount : ℝ) (rate : ℝ) : ℝ :=
  amount * (1 - rate)

/-- Calculate the amount after applying successive discounts -/
def amount_after_successive_discounts (amount : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (fun acc rate => acc * (1 - rate)) amount

/-- The difference between single discount and successive discounts -/
def discount_difference : ℝ :=
  amount_after_successive_discounts initial_amount successive_discount_rates -
  amount_after_single_discount initial_amount single_discount_rate

theorem discount_difference_is_978_75 :
  discount_difference = 978.75 := by sorry

end NUMINAMATH_CALUDE_discount_difference_is_978_75_l3622_362228


namespace NUMINAMATH_CALUDE_tangent_segment_difference_l3622_362207

/-- Represents a quadrilateral inscribed in a circle with an inscribed circle --/
structure InscribedQuadrilateral where
  /-- Side lengths of the quadrilateral --/
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  /-- Proof that the quadrilateral is inscribed in a circle --/
  inscribed_in_circle : True
  /-- Proof that there's a circle inscribed in the quadrilateral --/
  has_inscribed_circle : True

/-- Theorem about the difference of segments created by the inscribed circle's tangency point --/
theorem tangent_segment_difference (q : InscribedQuadrilateral)
    (h1 : q.side1 = 50)
    (h2 : q.side2 = 80)
    (h3 : q.side3 = 140)
    (h4 : q.side4 = 120) :
    ∃ (x y : ℝ), x + y = 140 ∧ |x - y| = 19 := by
  sorry


end NUMINAMATH_CALUDE_tangent_segment_difference_l3622_362207


namespace NUMINAMATH_CALUDE_faculty_reduction_l3622_362257

theorem faculty_reduction (initial_faculty : ℕ) : 
  (initial_faculty : ℝ) * 0.85 * 0.75 = 180 → 
  initial_faculty = 282 :=
by
  sorry

end NUMINAMATH_CALUDE_faculty_reduction_l3622_362257


namespace NUMINAMATH_CALUDE_jamies_mothers_age_twice_l3622_362289

/-- 
Given:
- Jamie's age in 2010 is 10 years
- Jamie's mother's age in 2010 is 5 times Jamie's age
Prove that the year when Jamie's mother's age will be twice Jamie's age is 2040
-/
theorem jamies_mothers_age_twice (jamie_age_2010 : ℕ) (mother_age_multiplier : ℕ) : 
  jamie_age_2010 = 10 →
  mother_age_multiplier = 5 →
  ∃ (years_passed : ℕ),
    (jamie_age_2010 + years_passed) * 2 = (jamie_age_2010 * mother_age_multiplier + years_passed) ∧
    2010 + years_passed = 2040 := by
  sorry

#check jamies_mothers_age_twice

end NUMINAMATH_CALUDE_jamies_mothers_age_twice_l3622_362289


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3622_362296

-- Define set A
def A : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define set B
def B : Set ℝ := {-1, 0, 1, 2, 3}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3622_362296


namespace NUMINAMATH_CALUDE_remainder_not_always_power_of_four_l3622_362270

theorem remainder_not_always_power_of_four : 
  ∃ n : ℕ, n ≥ 2 ∧ ∃ k : ℕ, 2^(2^n) ≡ k [MOD 2^n - 1] ∧ ¬∃ m : ℕ, k = 4^m := by
  sorry

end NUMINAMATH_CALUDE_remainder_not_always_power_of_four_l3622_362270


namespace NUMINAMATH_CALUDE_pickle_barrel_problem_l3622_362266

theorem pickle_barrel_problem (B M T G S : ℚ) : 
  M + T + G + S = B →
  B - M / 2 = B / 10 →
  B - T / 2 = B / 8 →
  B - G / 2 = B / 4 →
  B - S / 2 = B / 40 := by
sorry

end NUMINAMATH_CALUDE_pickle_barrel_problem_l3622_362266


namespace NUMINAMATH_CALUDE_problem_solution_l3622_362256

theorem problem_solution (x : ℝ) : 0.8 * x - 20 = 60 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3622_362256


namespace NUMINAMATH_CALUDE_triangle_tangent_sum_inequality_l3622_362252

/-- For any acute-angled triangle ABC with perimeter p and inradius r,
    the sum of the tangents of its angles is greater than or equal to
    the ratio of its perimeter to twice its inradius. -/
theorem triangle_tangent_sum_inequality (A B C : ℝ) (p r : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute angles
  A + B + C = Real.pi ∧    -- Sum of angles in a triangle
  p > 0 ∧ r > 0 →          -- Positive perimeter and inradius
  Real.tan A + Real.tan B + Real.tan C ≥ p / (2 * r) := by
sorry

end NUMINAMATH_CALUDE_triangle_tangent_sum_inequality_l3622_362252


namespace NUMINAMATH_CALUDE_embankment_build_time_l3622_362292

/-- Represents the time taken to build an embankment given a number of workers -/
def build_time (workers : ℕ) (days : ℚ) : Prop :=
  workers * days = 300

theorem embankment_build_time :
  build_time 75 4 → build_time 50 6 := by
  sorry

end NUMINAMATH_CALUDE_embankment_build_time_l3622_362292


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cos_sum_l3622_362226

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cos_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 5 + a 9 = 5 * Real.pi →
  Real.cos (a 2 + a 8) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cos_sum_l3622_362226


namespace NUMINAMATH_CALUDE_birds_in_tree_l3622_362250

/-- The number of birds left in a tree after some fly away -/
def birds_left (initial : ℝ) (flew_away : ℝ) : ℝ :=
  initial - flew_away

/-- Theorem: Given 21.0 initial birds and 14.0 birds that flew away, 7.0 birds are left -/
theorem birds_in_tree : birds_left 21.0 14.0 = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l3622_362250


namespace NUMINAMATH_CALUDE_log_one_over_twentyfive_base_five_l3622_362284

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_one_over_twentyfive_base_five : log 5 (1 / 25) = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_one_over_twentyfive_base_five_l3622_362284


namespace NUMINAMATH_CALUDE_descent_time_calculation_l3622_362201

theorem descent_time_calculation (climb_time : ℝ) (avg_speed_total : ℝ) (avg_speed_climb : ℝ) :
  climb_time = 4 →
  avg_speed_total = 2 →
  avg_speed_climb = 1.5 →
  ∃ (descent_time : ℝ),
    descent_time = 2 ∧
    avg_speed_total = (2 * avg_speed_climb * climb_time) / (climb_time + descent_time) :=
by sorry

end NUMINAMATH_CALUDE_descent_time_calculation_l3622_362201


namespace NUMINAMATH_CALUDE_brothers_combined_age_theorem_l3622_362205

/-- Represents the ages of two brothers -/
structure BrothersAges where
  adam : ℕ
  tom : ℕ

/-- Calculates the number of years until the brothers' combined age reaches a target -/
def yearsUntilCombinedAge (ages : BrothersAges) (targetAge : ℕ) : ℕ :=
  (targetAge - (ages.adam + ages.tom)) / 2

/-- Theorem: The number of years until Adam and Tom's combined age is 44 is 12 -/
theorem brothers_combined_age_theorem (ages : BrothersAges) 
  (h1 : ages.adam = 8) 
  (h2 : ages.tom = 12) : 
  yearsUntilCombinedAge ages 44 = 12 := by
  sorry

end NUMINAMATH_CALUDE_brothers_combined_age_theorem_l3622_362205


namespace NUMINAMATH_CALUDE_equal_angle_vector_l3622_362238

theorem equal_angle_vector (a b c : ℝ × ℝ) : 
  a = (1, 2) → 
  b = (4, 2) → 
  c ≠ (0, 0) → 
  (c.1 * a.1 + c.2 * a.2) / (Real.sqrt (c.1^2 + c.2^2) * Real.sqrt (a.1^2 + a.2^2)) = 
  (c.1 * b.1 + c.2 * b.2) / (Real.sqrt (c.1^2 + c.2^2) * Real.sqrt (b.1^2 + b.2^2)) → 
  ∃ (k : ℝ), k ≠ 0 ∧ c = (k, k) := by
sorry

end NUMINAMATH_CALUDE_equal_angle_vector_l3622_362238


namespace NUMINAMATH_CALUDE_number_problem_l3622_362291

theorem number_problem (N : ℚ) : 
  (N / (4/5) = (4/5) * N + 27) → N = 60 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3622_362291


namespace NUMINAMATH_CALUDE_racetrack_probability_l3622_362241

/-- Represents a circular racetrack -/
structure Racetrack where
  length : ℝ
  isCircular : Bool

/-- Represents a car on the racetrack -/
structure Car where
  position : ℝ
  travelDistance : ℝ

/-- Calculates the probability of the car ending within the specified range -/
def probabilityOfEndingInRange (track : Racetrack) (car : Car) (targetPosition : ℝ) (range : ℝ) : ℝ :=
  sorry

theorem racetrack_probability (track : Racetrack) (car : Car) : 
  track.length = 3 →
  track.isCircular = true →
  car.travelDistance = 0.5 →
  probabilityOfEndingInRange track car 2.5 0.5 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_racetrack_probability_l3622_362241


namespace NUMINAMATH_CALUDE_coordinates_of_point_B_l3622_362269

/-- Given a 2D coordinate system with origin O, point A at (-1, 2), 
    and vector BA = (3, 3), prove that the coordinates of point B are (-4, -1) -/
theorem coordinates_of_point_B (O A B : ℝ × ℝ) : 
  O = (0, 0) → 
  A = (-1, 2) → 
  B - A = (3, 3) →
  B = (-4, -1) := by
sorry

end NUMINAMATH_CALUDE_coordinates_of_point_B_l3622_362269


namespace NUMINAMATH_CALUDE_average_shift_l3622_362219

theorem average_shift (a b c : ℝ) : 
  (a + b + c) / 3 = 5 → ((a - 2) + (b - 2) + (c - 2)) / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_shift_l3622_362219


namespace NUMINAMATH_CALUDE_danny_thrice_jane_age_l3622_362283

/-- Proves that Danny was thrice as old as Jane 19 years ago -/
theorem danny_thrice_jane_age (danny_age : ℕ) (jane_age : ℕ) 
  (h1 : danny_age = 40) (h2 : jane_age = 26) : 
  ∃ x : ℕ, x = 19 ∧ (danny_age - x) = 3 * (jane_age - x) :=
by sorry

end NUMINAMATH_CALUDE_danny_thrice_jane_age_l3622_362283


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l3622_362222

theorem greatest_integer_radius (r : ℝ) : r > 0 → r * r * Real.pi < 75 * Real.pi → ∃ n : ℕ, n = 8 ∧ (∀ m : ℕ, m * m * Real.pi < 75 * Real.pi → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l3622_362222


namespace NUMINAMATH_CALUDE_equation_solution_l3622_362235

theorem equation_solution :
  ∃ y : ℚ, y ≠ -2 ∧ (6 * y / (y + 2) - 2 / (y + 2) = 5 / (y + 2)) ∧ y = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3622_362235


namespace NUMINAMATH_CALUDE_syllogistic_reasoning_incorrect_l3622_362262

-- Define the complex number z
def z : ℂ := 2 + 3 * Complex.I

-- Define the major premise (which is false)
def major_premise : Prop := ∀ (c : ℂ), ∃ (r i : ℝ), c = r + i * Complex.I

-- Define the minor premise
def minor_premise : Prop := z.re = 2

-- Define the conclusion
def conclusion : Prop := z.im = 3

-- Theorem stating that the syllogistic reasoning is incorrect due to a false major premise
theorem syllogistic_reasoning_incorrect :
  ¬major_premise → minor_premise → conclusion → 
  ∃ (error : String), error = "The syllogistic reasoning is incorrect due to a false major premise" :=
by
  sorry

end NUMINAMATH_CALUDE_syllogistic_reasoning_incorrect_l3622_362262


namespace NUMINAMATH_CALUDE_nathan_ate_100_gumballs_l3622_362202

/-- The number of gumballs in each package -/
def gumballs_per_package : ℝ := 5.0

/-- The number of packages Nathan ate -/
def packages_eaten : ℝ := 20.0

/-- The total number of gumballs Nathan ate -/
def total_gumballs : ℝ := gumballs_per_package * packages_eaten

theorem nathan_ate_100_gumballs : total_gumballs = 100.0 := by
  sorry

end NUMINAMATH_CALUDE_nathan_ate_100_gumballs_l3622_362202


namespace NUMINAMATH_CALUDE_basketball_tournament_games_l3622_362251

theorem basketball_tournament_games (x : ℕ) 
  (h1 : x > 0)
  (h2 : (3 * x) / 4 = (2 * (x + 4)) / 3 - 8) :
  x = 48 := by
sorry

end NUMINAMATH_CALUDE_basketball_tournament_games_l3622_362251


namespace NUMINAMATH_CALUDE_complement_A_inter_B_range_of_a_l3622_362298

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 6}
def B : Set ℝ := {x | 3 * x - 7 ≥ 8 - 2 * x}
def C (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Theorem for the complement of A ∩ B
theorem complement_A_inter_B :
  (A ∩ B)ᶜ = {x | x < 3 ∨ x > 6} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) (h : A ∪ C a = C a) :
  a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_range_of_a_l3622_362298


namespace NUMINAMATH_CALUDE_place_face_value_difference_l3622_362218

def number : ℕ := 856973

def digit_of_interest : ℕ := 7

def place_value (n : ℕ) (d : ℕ) : ℕ :=
  if n / 100 % 10 = d then d * 10 else 0

def face_value (d : ℕ) : ℕ := d

theorem place_face_value_difference :
  place_value number digit_of_interest - face_value digit_of_interest = 63 := by
  sorry

end NUMINAMATH_CALUDE_place_face_value_difference_l3622_362218


namespace NUMINAMATH_CALUDE_sum_of_factors_144_l3622_362246

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_factors_144 : sum_of_factors 144 = 403 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_144_l3622_362246


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3622_362215

theorem smallest_number_satisfying_conditions : ∃ n : ℕ,
  (n % 6 = 1) ∧ (n % 8 = 3) ∧ (n % 9 = 2) ∧
  (∀ m : ℕ, m < n → ¬((m % 6 = 1) ∧ (m % 8 = 3) ∧ (m % 9 = 2))) ∧
  n = 107 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3622_362215


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3622_362220

theorem binomial_expansion_coefficient (a : ℝ) : 
  (Nat.choose 6 3 : ℝ) * a^3 * 2^3 = 5/2 → a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3622_362220


namespace NUMINAMATH_CALUDE_battery_current_at_12_ohms_l3622_362242

/-- A battery with voltage 48V and current-resistance relationship I = 48 / R -/
structure Battery where
  voltage : ℝ
  current : ℝ → ℝ
  resistance : ℝ
  h1 : voltage = 48
  h2 : ∀ r, current r = 48 / r

/-- When resistance is 12Ω, the current is 4A -/
theorem battery_current_at_12_ohms (b : Battery) (h : b.resistance = 12) : b.current b.resistance = 4 := by
  sorry

end NUMINAMATH_CALUDE_battery_current_at_12_ohms_l3622_362242


namespace NUMINAMATH_CALUDE_power_equality_l3622_362213

theorem power_equality (n : ℕ) : 9^4 = 3^n → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3622_362213


namespace NUMINAMATH_CALUDE_area_of_rectangle_S_l3622_362211

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- The configuration of shapes within the larger square -/
structure Configuration where
  largerSquare : Square
  rectangle : Rectangle
  smallerSquare : Square
  rectangleS : Rectangle

/-- The conditions of the problem -/
def validConfiguration (c : Configuration) : Prop :=
  c.rectangle.width = 2 ∧
  c.rectangle.height = 4 ∧
  c.smallerSquare.side = 2 ∧
  c.largerSquare.side ≥ 4 ∧
  c.largerSquare.side ^ 2 = 
    c.rectangle.width * c.rectangle.height +
    c.smallerSquare.side ^ 2 +
    c.rectangleS.width * c.rectangleS.height

theorem area_of_rectangle_S (c : Configuration) 
  (h : validConfiguration c) : 
  c.rectangleS.width * c.rectangleS.height = 4 :=
sorry

end NUMINAMATH_CALUDE_area_of_rectangle_S_l3622_362211


namespace NUMINAMATH_CALUDE_raccoon_lock_problem_l3622_362293

theorem raccoon_lock_problem (first_lock_duration second_lock_duration : ℕ) : 
  first_lock_duration = 5 →
  second_lock_duration < 3 * first_lock_duration →
  5 * second_lock_duration = 60 →
  3 * first_lock_duration - second_lock_duration = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_raccoon_lock_problem_l3622_362293


namespace NUMINAMATH_CALUDE_library_shelving_l3622_362259

theorem library_shelving (jason_books_per_time lexi_books_per_time total_books : ℕ) :
  jason_books_per_time = 6 →
  total_books = 102 →
  total_books % jason_books_per_time = 0 →
  total_books % lexi_books_per_time = 0 →
  total_books / jason_books_per_time = total_books / lexi_books_per_time →
  lexi_books_per_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_library_shelving_l3622_362259


namespace NUMINAMATH_CALUDE_blue_given_not_red_probability_l3622_362258

-- Define the total number of balls
def total_balls : ℕ := 20

-- Define the number of red balls
def red_balls : ℕ := 5

-- Define the number of yellow balls
def yellow_balls : ℕ := 5

-- Define the number of blue balls
def blue_balls : ℕ := 10

-- Define the number of non-red balls
def non_red_balls : ℕ := yellow_balls + blue_balls

-- Theorem: The probability of drawing a blue ball given that it's not red is 2/3
theorem blue_given_not_red_probability : 
  (blue_balls : ℚ) / (non_red_balls : ℚ) = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_blue_given_not_red_probability_l3622_362258


namespace NUMINAMATH_CALUDE_extremum_point_implies_a_value_max_min_values_l3622_362265

-- Define the function f(x) = x^3 - ax
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- Theorem 1: If x=1 is an extremum point of f(x), then a = 3
theorem extremum_point_implies_a_value (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1 ∨ f a x ≥ f a 1) →
  a = 3 :=
sorry

-- Theorem 2: For f(x) = x^3 - 3x and x ∈ [0, 2], the maximum value is 2 and the minimum value is -2
theorem max_min_values :
  (∀ x ∈ Set.Icc 0 2, f 3 x ≤ 2) ∧
  (∀ x ∈ Set.Icc 0 2, f 3 x ≥ -2) ∧
  (∃ x ∈ Set.Icc 0 2, f 3 x = 2) ∧
  (∃ x ∈ Set.Icc 0 2, f 3 x = -2) :=
sorry

end NUMINAMATH_CALUDE_extremum_point_implies_a_value_max_min_values_l3622_362265


namespace NUMINAMATH_CALUDE_insects_in_laboratory_l3622_362229

/-- The number of insects in a laboratory given the total number of insect legs and legs per insect. -/
def number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) : ℕ :=
  total_legs / legs_per_insect

/-- Theorem stating that there are 9 insects in the laboratory given the conditions. -/
theorem insects_in_laboratory : number_of_insects 54 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_insects_in_laboratory_l3622_362229


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_given_l3622_362234

-- Define the given line
def given_line (x y : ℝ) : Prop := x - 2 * y + 3 = 0

-- Define the point that the new line passes through
def point : ℝ × ℝ := (-1, 3)

-- Define the equation of the new line
def new_line (x y : ℝ) : Prop := x - 2 * y + 7 = 0

-- Theorem statement
theorem line_through_point_parallel_to_given : 
  (∀ (x y : ℝ), new_line x y ↔ ∃ (k : ℝ), x - point.1 = k * 1 ∧ y - point.2 = k * (-1/2)) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), given_line x₁ y₁ ∧ given_line x₂ y₂ → 
    (x₂ - x₁) * (-1/2) = (y₂ - y₁) * 1) ∧
  new_line point.1 point.2 :=
sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_given_l3622_362234


namespace NUMINAMATH_CALUDE_emilys_waist_size_conversion_l3622_362297

/-- Conversion of Emily's waist size from inches to centimeters -/
theorem emilys_waist_size_conversion (inches_per_foot : ℝ) (cm_per_foot : ℝ) (waist_inches : ℝ) :
  inches_per_foot = 12 →
  cm_per_foot = 30.48 →
  waist_inches = 28 →
  ∃ (waist_cm : ℝ), abs (waist_cm - (waist_inches / inches_per_foot * cm_per_foot)) < 0.1 ∧ waist_cm = 71.1 := by
  sorry

end NUMINAMATH_CALUDE_emilys_waist_size_conversion_l3622_362297


namespace NUMINAMATH_CALUDE_volume_conversion_l3622_362274

-- Define conversion factors
def feet_to_meters : ℝ := 0.3048
def meters_to_yards : ℝ := 1.09361

-- Define the volume in cubic feet
def volume_cubic_feet : ℝ := 216

-- Define the conversion function from cubic feet to cubic meters
def cubic_feet_to_cubic_meters (v : ℝ) : ℝ := v * (feet_to_meters ^ 3)

-- Define the conversion function from cubic meters to cubic yards
def cubic_meters_to_cubic_yards (v : ℝ) : ℝ := v * (meters_to_yards ^ 3)

-- Theorem statement
theorem volume_conversion :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |cubic_meters_to_cubic_yards (cubic_feet_to_cubic_meters volume_cubic_feet) - 8| < ε :=
sorry

end NUMINAMATH_CALUDE_volume_conversion_l3622_362274


namespace NUMINAMATH_CALUDE_square_ratio_proof_l3622_362290

theorem square_ratio_proof : 
  ∀ (s₁ s₂ : ℝ), s₁ > 0 ∧ s₂ > 0 →
  (s₁^2 / s₂^2 = 45 / 64) →
  ∃ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0) ∧
  (s₁ / s₂ = (a : ℝ) * Real.sqrt b / c) ∧
  (a + b + c = 16) :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_proof_l3622_362290


namespace NUMINAMATH_CALUDE_all_children_receive_candy_candy_distribution_works_l3622_362232

/-- Represents the candy distribution function -/
def candyDistribution (n : ℕ+) (k : ℕ) : ℕ :=
  (k * (k + 1) / 2) % n

/-- Theorem stating that all children receive candy iff n is a power of 2 -/
theorem all_children_receive_candy (n : ℕ+) :
  (∀ i : ℕ, i < n → ∃ k : ℕ, candyDistribution n k = i) ↔ ∃ m : ℕ, n = 2^m := by
  sorry

/-- Corollary: The number of children for which the candy distribution works -/
theorem candy_distribution_works (n : ℕ+) :
  (∀ i : ℕ, i < n → ∃ k : ℕ, candyDistribution n k = i) → ∃ m : ℕ, n = 2^m := by
  sorry

end NUMINAMATH_CALUDE_all_children_receive_candy_candy_distribution_works_l3622_362232


namespace NUMINAMATH_CALUDE_equation_solution_l3622_362286

theorem equation_solution :
  ∃ y : ℝ, (7 * (4 * y + 3) - 3 = -3 * (2 - 9 * y)) ∧ (y = -24) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3622_362286


namespace NUMINAMATH_CALUDE_arithmetic_square_root_l3622_362240

theorem arithmetic_square_root (a : ℝ) (h : a > 0) : Real.sqrt a = (a ^ (1/2 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_l3622_362240


namespace NUMINAMATH_CALUDE_current_speed_l3622_362285

/-- Given a man's speed with and against a current, calculate the speed of the current. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 15)
  (h2 : speed_against_current = 10) :
  ∃ (current_speed : ℝ), current_speed = 2.5 ∧ 
    speed_with_current = speed_against_current + 2 * current_speed :=
by sorry

end NUMINAMATH_CALUDE_current_speed_l3622_362285


namespace NUMINAMATH_CALUDE_cubic_inequality_implies_value_range_l3622_362203

theorem cubic_inequality_implies_value_range (y : ℝ) : 
  y^3 - 6*y^2 + 11*y - 6 < 0 → 
  24 < y^3 + 6*y^2 + 11*y + 6 ∧ y^3 + 6*y^2 + 11*y + 6 < 120 := by
sorry

end NUMINAMATH_CALUDE_cubic_inequality_implies_value_range_l3622_362203


namespace NUMINAMATH_CALUDE_expo_visitors_scientific_notation_l3622_362243

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Rounds a real number to a given number of significant figures -/
def roundToSignificantFigures (x : ℝ) (figures : ℕ) : ℝ :=
  sorry

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem expo_visitors_scientific_notation :
  let visitors : ℝ := 8.0327 * 1000000
  let rounded := roundToSignificantFigures visitors 2
  let scientific := toScientificNotation rounded
  scientific.coefficient = 8.0 ∧ scientific.exponent = 6 := by
  sorry

end NUMINAMATH_CALUDE_expo_visitors_scientific_notation_l3622_362243


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3622_362225

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 4 + a 8 = 16 → a 2 + a 10 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3622_362225


namespace NUMINAMATH_CALUDE_parkway_elementary_soccer_l3622_362230

theorem parkway_elementary_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) (boys_soccer_percent : ℚ) :
  total_students = 470 →
  boys = 300 →
  soccer_players = 250 →
  boys_soccer_percent = 86 / 100 →
  (total_students - boys) - (soccer_players - (boys_soccer_percent * soccer_players).floor) = 135 := by
sorry

end NUMINAMATH_CALUDE_parkway_elementary_soccer_l3622_362230


namespace NUMINAMATH_CALUDE_max_value_of_sum_l3622_362236

noncomputable def f (x : ℝ) : ℝ := 3^(x-1) + x - 1

def is_inverse (f g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

theorem max_value_of_sum (f : ℝ → ℝ) (f_inv : ℝ → ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x = 3^(x-1) + x - 1) →
  is_inverse f f_inv →
  (∃ y, ∀ x ∈ Set.Icc 0 1, f x + f_inv x ≤ y) ∧
  (∃ x ∈ Set.Icc 0 1, f x + f_inv x = 2) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l3622_362236


namespace NUMINAMATH_CALUDE_simplify_expression_l3622_362249

/-- Given a = 1 and b = -4, prove that 4(a²b+ab²)-3(a²b-1)+2ab²-6 = 89 -/
theorem simplify_expression (a b : ℝ) (ha : a = 1) (hb : b = -4) :
  4*(a^2*b + a*b^2) - 3*(a^2*b - 1) + 2*a*b^2 - 6 = 89 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3622_362249


namespace NUMINAMATH_CALUDE_team_total_catch_l3622_362221

/-- Represents the number of days in the fishing competition -/
def competition_days : ℕ := 5

/-- Represents Jackson's daily catch -/
def jackson_daily_catch : ℕ := 6

/-- Represents Jonah's daily catch -/
def jonah_daily_catch : ℕ := 4

/-- Represents George's daily catch -/
def george_daily_catch : ℕ := 8

/-- Theorem stating the total catch of the team during the competition -/
theorem team_total_catch : 
  competition_days * (jackson_daily_catch + jonah_daily_catch + george_daily_catch) = 90 := by
  sorry

end NUMINAMATH_CALUDE_team_total_catch_l3622_362221


namespace NUMINAMATH_CALUDE_petes_bottle_return_l3622_362254

/-- Represents the number of bottles Pete needs to return to the store -/
def bottles_to_return (total_owed : ℚ) (cash_in_wallet : ℚ) (cash_in_pockets : ℚ) (bottle_return_rate : ℚ) : ℕ :=
  sorry

/-- The theorem stating the number of bottles Pete needs to return -/
theorem petes_bottle_return : 
  bottles_to_return 90 40 40 (1/2) = 20 := by sorry

end NUMINAMATH_CALUDE_petes_bottle_return_l3622_362254


namespace NUMINAMATH_CALUDE_fourth_number_is_28_l3622_362295

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def sequence_property (a b c d : ℕ) : Prop :=
  is_two_digit a ∧ is_two_digit b ∧ is_two_digit c ∧ is_two_digit d ∧
  (digit_sum a + digit_sum b + digit_sum c + digit_sum d) * 4 = a + b + c + d

theorem fourth_number_is_28 :
  ∃ (d : ℕ), sequence_property 46 19 63 d ∧ d = 28 :=
sorry

end NUMINAMATH_CALUDE_fourth_number_is_28_l3622_362295


namespace NUMINAMATH_CALUDE_lost_card_sum_l3622_362263

theorem lost_card_sum (a b c d : ℝ) : 
  let sums := [a + b, a + c, a + d, b + c, b + d, c + d]
  (∃ (s : Finset ℝ), s ⊆ sums.toFinset ∧ s.card = 5 ∧ 
    (270 ∈ s ∧ 360 ∈ s ∧ 390 ∈ s ∧ 500 ∈ s ∧ 620 ∈ s)) →
  530 ∈ sums.toFinset :=
by sorry

end NUMINAMATH_CALUDE_lost_card_sum_l3622_362263


namespace NUMINAMATH_CALUDE_range_of_a_l3622_362299

/-- The range of real number a when "x=1" is a sufficient but not necessary condition for "(x-a)[x-(a+2)]≤0" -/
theorem range_of_a : ∃ (a_min a_max : ℝ), a_min = -1 ∧ a_max = 1 ∧
  ∀ (a : ℝ), (∀ (x : ℝ), x = 1 → (x - a) * (x - (a + 2)) ≤ 0) ∧
             (∃ (x : ℝ), x ≠ 1 ∧ (x - a) * (x - (a + 2)) ≤ 0) ↔
             a_min ≤ a ∧ a ≤ a_max := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3622_362299


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l3622_362255

theorem line_not_in_second_quadrant (α : Real) (h : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi) :
  ∃ (x y : Real), x > 0 ∧ y < 0 ∧ x / Real.cos α + y / Real.sin α = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l3622_362255


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l3622_362248

theorem fraction_sum_simplification : (1 : ℚ) / 210 + 17 / 30 = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l3622_362248


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3622_362247

/-- The intersection points of a line and a circle -/
theorem line_circle_intersection :
  let line := { p : ℝ × ℝ | p.1 + p.2 = 1 }
  let circle := { p : ℝ × ℝ | p.1^2 + p.2^2 = 9 }
  let point1 := ((1 + Real.sqrt 17) / 2, (1 - Real.sqrt 17) / 2)
  let point2 := ((1 - Real.sqrt 17) / 2, (1 + Real.sqrt 17) / 2)
  (point1 ∈ line ∧ point1 ∈ circle) ∧ 
  (point2 ∈ line ∧ point2 ∈ circle) ∧
  (∀ p ∈ line ∩ circle, p = point1 ∨ p = point2) :=
by
  sorry


end NUMINAMATH_CALUDE_line_circle_intersection_l3622_362247


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3622_362217

/-- A quadratic function f(x) = x^2 + px + q -/
def f (p q x : ℝ) : ℝ := x^2 + p*x + q

/-- The theorem stating the conditions for f(p) = f(q) = 0 -/
theorem quadratic_roots_condition (p q : ℝ) :
  (f p q p = 0 ∧ f p q q = 0) ↔ 
  ((p = 0 ∧ q = 0) ∨ (p = -1/2 ∧ q = -1/2) ∨ (p = 1 ∧ q = -2)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3622_362217


namespace NUMINAMATH_CALUDE_bales_in_barn_l3622_362294

/-- The number of bales in the barn after Tim added a couple more -/
def total_bales (initial_bales : ℕ) (added_bales : ℕ) : ℕ :=
  initial_bales + added_bales

/-- A couple is defined as 2 -/
def couple : ℕ := 2

theorem bales_in_barn (initial_bales : ℕ) (h : initial_bales = 540) :
  total_bales initial_bales couple = 542 := by
  sorry

end NUMINAMATH_CALUDE_bales_in_barn_l3622_362294


namespace NUMINAMATH_CALUDE_wizard_concoction_combinations_l3622_362214

/-- Represents the number of herbs available --/
def num_herbs : ℕ := 4

/-- Represents the number of crystals available --/
def num_crystals : ℕ := 6

/-- Represents the number of incompatible combinations --/
def num_incompatible : ℕ := 3

/-- Theorem stating the number of valid combinations for the wizard's concoction --/
theorem wizard_concoction_combinations : 
  num_herbs * num_crystals - num_incompatible = 21 := by
  sorry

end NUMINAMATH_CALUDE_wizard_concoction_combinations_l3622_362214


namespace NUMINAMATH_CALUDE_joes_steakhouse_wages_l3622_362282

/-- Proves that the manager's hourly wage is $6.50 given the conditions from Joe's Steakhouse --/
theorem joes_steakhouse_wages (manager_wage dishwasher_wage chef_wage : ℝ) :
  chef_wage = dishwasher_wage + 0.2 * dishwasher_wage →
  dishwasher_wage = 0.5 * manager_wage →
  chef_wage = manager_wage - 2.6 →
  manager_wage = 6.5 := by
sorry

end NUMINAMATH_CALUDE_joes_steakhouse_wages_l3622_362282


namespace NUMINAMATH_CALUDE_equal_parts_in_one_to_one_mix_l3622_362212

/-- Represents a substrate composition with parts of bark, peat, and sand -/
structure Substrate :=
  (bark : ℚ)
  (peat : ℚ)
  (sand : ℚ)

/-- Orchid-1 substrate composition -/
def orchid1 : Substrate :=
  { bark := 3
    peat := 2
    sand := 1 }

/-- Orchid-2 substrate composition -/
def orchid2 : Substrate :=
  { bark := 1
    peat := 2
    sand := 3 }

/-- Mixes two substrates in given proportions -/
def mixSubstrates (s1 s2 : Substrate) (r1 r2 : ℚ) : Substrate :=
  { bark := r1 * s1.bark + r2 * s2.bark
    peat := r1 * s1.peat + r2 * s2.peat
    sand := r1 * s1.sand + r2 * s2.sand }

/-- Checks if all components of a substrate are equal -/
def hasEqualParts (s : Substrate) : Prop :=
  s.bark = s.peat ∧ s.peat = s.sand

theorem equal_parts_in_one_to_one_mix :
  hasEqualParts (mixSubstrates orchid1 orchid2 1 1) :=
by sorry


end NUMINAMATH_CALUDE_equal_parts_in_one_to_one_mix_l3622_362212


namespace NUMINAMATH_CALUDE_vector_add_scale_l3622_362261

/-- Given two 3D vectors, prove that adding them and scaling the result by 2 yields the expected vector -/
theorem vector_add_scale (v1 v2 : Fin 3 → ℝ) (h1 : v1 = ![- 3, 2, 5]) (h2 : v2 = ![4, 7, - 3]) :
  (2 • (v1 + v2)) = ![2, 18, 4] := by
  sorry

end NUMINAMATH_CALUDE_vector_add_scale_l3622_362261


namespace NUMINAMATH_CALUDE_company_production_theorem_l3622_362245

/-- Represents the production schedule of a company making parts --/
structure ProductionSchedule where
  initialRate : ℕ            -- Initial production rate (parts per day)
  initialDays : ℕ            -- Number of days at initial rate
  increasedRate : ℕ          -- Increased production rate (parts per day)
  extraParts : ℕ             -- Extra parts produced beyond the plan

/-- Calculates the total number of parts produced given a production schedule --/
def totalPartsProduced (schedule : ProductionSchedule) : ℕ :=
  sorry

/-- Theorem stating that given the specific production schedule, 675 parts are produced --/
theorem company_production_theorem :
  let schedule := ProductionSchedule.mk 25 3 30 100
  totalPartsProduced schedule = 675 :=
by sorry

end NUMINAMATH_CALUDE_company_production_theorem_l3622_362245


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3622_362275

/-- Given an algebraic expression ax-2, if the value of the expression is 4 when x=2, then a=3 -/
theorem algebraic_expression_value (a : ℝ) : (a * 2 - 2 = 4) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3622_362275


namespace NUMINAMATH_CALUDE_gas_fill_friday_l3622_362267

/-- Calculates the number of liters of gas Mr. Deane will fill on Friday given the conditions of the problem. -/
theorem gas_fill_friday 
  (today_liters : ℝ) 
  (today_price : ℝ) 
  (price_rollback : ℝ) 
  (total_cost : ℝ) 
  (total_liters : ℝ) 
  (h1 : today_liters = 10)
  (h2 : today_price = 1.4)
  (h3 : price_rollback = 0.4)
  (h4 : total_cost = 39)
  (h5 : total_liters = 35) :
  total_liters - today_liters = 25 := by
sorry

end NUMINAMATH_CALUDE_gas_fill_friday_l3622_362267


namespace NUMINAMATH_CALUDE_equation_condition_l3622_362208

theorem equation_condition (a b c : ℤ) : 
  a * (a - b) + b * (b - c) + c * (c - a) = 2 → (a > b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_equation_condition_l3622_362208


namespace NUMINAMATH_CALUDE_amusement_park_cost_per_trip_l3622_362239

/-- The cost per trip to an amusement park given the following conditions:
  * Two season passes are purchased
  * Each pass costs 100 (in some currency unit)
  * One person uses their pass 35 times
  * Another person uses their pass 15 times
-/
theorem amusement_park_cost_per_trip 
  (pass_cost : ℝ) 
  (num_passes : ℕ) 
  (trips_person1 : ℕ) 
  (trips_person2 : ℕ) 
  (h1 : pass_cost = 100) 
  (h2 : num_passes = 2) 
  (h3 : trips_person1 = 35) 
  (h4 : trips_person2 = 15) : 
  (num_passes * pass_cost) / (trips_person1 + trips_person2 : ℝ) = 4 := by
  sorry

#check amusement_park_cost_per_trip

end NUMINAMATH_CALUDE_amusement_park_cost_per_trip_l3622_362239


namespace NUMINAMATH_CALUDE_greatest_power_under_600_l3622_362278

theorem greatest_power_under_600 (a b : ℕ) : 
  a > 0 → b > 1 → a^b < 600 → 
  (∀ c d : ℕ, c > 0 → d > 1 → c^d < 600 → c^d ≤ a^b) →
  a + b = 26 := by
sorry

end NUMINAMATH_CALUDE_greatest_power_under_600_l3622_362278


namespace NUMINAMATH_CALUDE_min_value_f_when_m_1_existence_of_m_l3622_362277

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x + m / (2 * x)

def g (m : ℝ) (x : ℝ) : ℝ := x - 2 * m

theorem min_value_f_when_m_1 :
  ∃ x₀ > 0, ∀ x > 0, f 1 x₀ ≤ f 1 x ∧ f 1 x₀ = 1 - Real.log 2 := by sorry

theorem existence_of_m :
  ∃ m ∈ Set.Ioo (4/5 : ℝ) 1, ∀ x ∈ Set.Icc (Real.exp (-1)) 1,
    f m x > g m x + 1 := by sorry

end NUMINAMATH_CALUDE_min_value_f_when_m_1_existence_of_m_l3622_362277


namespace NUMINAMATH_CALUDE_unit_digit_of_seven_to_fourteen_l3622_362216

theorem unit_digit_of_seven_to_fourteen (n : ℕ) : n = 7^14 → n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_seven_to_fourteen_l3622_362216


namespace NUMINAMATH_CALUDE_arrangement_count_l3622_362281

/-- Represents the number of students -/
def num_students : ℕ := 4

/-- Represents the number of schools -/
def num_schools : ℕ := 3

/-- Represents the total number of arrangements without restrictions -/
def total_arrangements : ℕ := (num_students.choose 2) * num_schools.factorial

/-- Represents the number of arrangements where A and B are in the same school -/
def arrangements_ab_together : ℕ := num_schools.factorial

/-- Represents the number of valid arrangements -/
def valid_arrangements : ℕ := total_arrangements - arrangements_ab_together

theorem arrangement_count : valid_arrangements = 30 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_l3622_362281


namespace NUMINAMATH_CALUDE_N_subset_M_l3622_362272

def M : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | x - 2 = 0}

theorem N_subset_M : N ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_N_subset_M_l3622_362272


namespace NUMINAMATH_CALUDE_backpack_price_change_l3622_362260

theorem backpack_price_change (P : ℝ) (x : ℝ) (h : P > 0) :
  P * (1 + x / 100) * (1 - x / 100) = 0.64 * P →
  x = 60 := by
  sorry

end NUMINAMATH_CALUDE_backpack_price_change_l3622_362260


namespace NUMINAMATH_CALUDE_solution_set_f_intersection_condition_l3622_362276

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - |x + 1|

-- Define the function g(x)
def g (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Theorem 1: Solution set of f(x) > 2 when m = 5
theorem solution_set_f (x : ℝ) : f 5 x > 2 ↔ -3/2 < x ∧ x < 3/2 := by sorry

-- Theorem 2: Condition for f(x) and g(x) to always intersect
theorem intersection_condition (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, f m y = g y) ↔ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_intersection_condition_l3622_362276


namespace NUMINAMATH_CALUDE_water_tank_problem_l3622_362264

theorem water_tank_problem (c : ℝ) (h1 : c > 0) : 
  let w := c / 3
  let w' := w + 5
  let w'' := w' + 4
  (w / c = 1 / 3) ∧ (w' / c = 2 / 5) → w'' / c = 34 / 75 := by
sorry

end NUMINAMATH_CALUDE_water_tank_problem_l3622_362264


namespace NUMINAMATH_CALUDE_wheel_radii_problem_l3622_362233

theorem wheel_radii_problem (x : ℝ) : 
  (2 * x > 0) →  -- Ensure positive radii
  (1500 / (2 * Real.pi * x + 5) = 1875 / (4 * Real.pi * x - 5)) → 
  (x = 15 / (2 * Real.pi) ∧ 2 * x = 15 / Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_wheel_radii_problem_l3622_362233


namespace NUMINAMATH_CALUDE_chewing_gums_count_l3622_362209

/-- Given the total number of treats, chocolate bars, and candies, prove the number of chewing gums. -/
theorem chewing_gums_count 
  (total_treats : ℕ) 
  (chocolate_bars : ℕ) 
  (candies : ℕ) 
  (h1 : total_treats = 155) 
  (h2 : chocolate_bars = 55) 
  (h3 : candies = 40) : 
  total_treats - (chocolate_bars + candies) = 60 := by
  sorry

#check chewing_gums_count

end NUMINAMATH_CALUDE_chewing_gums_count_l3622_362209


namespace NUMINAMATH_CALUDE_parallelogram_inequality_l3622_362237

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Parallelogram P -/
structure Parallelogram (n : ℕ) (t : ℝ) where
  v1 : ℝ × ℝ := (0, 0)
  v2 : ℝ × ℝ := (0, t)
  v3 : ℝ × ℝ := (t * fib (2 * n + 1), t * fib (2 * n))
  v4 : ℝ × ℝ := (t * fib (2 * n + 1), t * fib (2 * n) + t)

/-- Number of integer points inside P -/
def L (n : ℕ) (t : ℝ) : ℕ := sorry

/-- Area of P -/
def M (n : ℕ) (t : ℝ) : ℝ := t^2 * fib (2 * n + 1)

/-- Main theorem -/
theorem parallelogram_inequality (n : ℕ) (t : ℝ) (hn : n > 1) (ht : t ≥ 1) :
  |Real.sqrt (L n t) - Real.sqrt (M n t)| ≤ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_parallelogram_inequality_l3622_362237
