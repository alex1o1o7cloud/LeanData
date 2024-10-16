import Mathlib

namespace NUMINAMATH_CALUDE_power_three_mod_thirteen_l2663_266380

theorem power_three_mod_thirteen : 3^39 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_thirteen_l2663_266380


namespace NUMINAMATH_CALUDE_grandmothers_age_is_52_l2663_266367

/-- The age of the grandmother given the average age of the family and the ages of the children -/
def grandmothers_age (average_age : ℝ) (child1_age child2_age child3_age : ℕ) : ℝ :=
  4 * average_age - (child1_age + child2_age + child3_age)

/-- Theorem stating that the grandmother's age is 52 given the problem conditions -/
theorem grandmothers_age_is_52 :
  grandmothers_age 20 5 10 13 = 52 := by
  sorry

end NUMINAMATH_CALUDE_grandmothers_age_is_52_l2663_266367


namespace NUMINAMATH_CALUDE_cubic_function_increasing_l2663_266334

theorem cubic_function_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁^3 < x₂^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_increasing_l2663_266334


namespace NUMINAMATH_CALUDE_sandy_age_l2663_266313

theorem sandy_age (S M : ℕ) (h1 : S = M - 16) (h2 : S * 9 = M * 7) : S = 56 := by
  sorry

end NUMINAMATH_CALUDE_sandy_age_l2663_266313


namespace NUMINAMATH_CALUDE_triangle_max_area_l2663_266356

/-- Given a triangle ABC where:
  - The sides a, b, c are opposite to angles A, B, C respectively
  - a = 2
  - tan A / tan B = 4/3
  The maximum area of the triangle is 1/2 -/
theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) : 
  a = 2 → 
  (Real.tan A) / (Real.tan B) = 4/3 →
  0 < a ∧ 0 < b ∧ 0 < c →
  A + B + C = Real.pi →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  (∃ (S : ℝ), S = (1/2) * b * c * (Real.sin A) ∧ 
    ∀ (S' : ℝ), S' = (1/2) * b * c * (Real.sin A) → S' ≤ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2663_266356


namespace NUMINAMATH_CALUDE_matt_twice_james_age_l2663_266378

/-- 
Given:
- James turned 27 three years ago
- Matt is now 65 years old

Prove that in 5 years, Matt will be twice James' age.
-/
theorem matt_twice_james_age (james_age_three_years_ago : ℕ) (matt_current_age : ℕ) :
  james_age_three_years_ago = 27 →
  matt_current_age = 65 →
  ∃ (years_from_now : ℕ), 
    years_from_now = 5 ∧
    matt_current_age + years_from_now = 2 * (james_age_three_years_ago + 3 + years_from_now) :=
by sorry

end NUMINAMATH_CALUDE_matt_twice_james_age_l2663_266378


namespace NUMINAMATH_CALUDE_cubic_cm_in_cubic_meter_proof_l2663_266383

/-- The number of cubic centimeters in one cubic meter -/
def cubic_cm_in_cubic_meter : ℕ := 1000000

/-- The number of centimeters in one meter -/
def cm_in_meter : ℕ := 100

/-- Theorem stating that the number of cubic centimeters in one cubic meter is 1,000,000,
    given that one meter is equal to one hundred centimeters -/
theorem cubic_cm_in_cubic_meter_proof :
  cubic_cm_in_cubic_meter = cm_in_meter ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_cm_in_cubic_meter_proof_l2663_266383


namespace NUMINAMATH_CALUDE_trig_identity_l2663_266312

theorem trig_identity (α : ℝ) 
  (h1 : Real.cos (7 * Real.pi / 2 + α) = 4 / 7)
  (h2 : Real.tan α < 0) :
  Real.cos (Real.pi - α) + Real.sin (Real.pi / 2 - α) * Real.tan α = (4 + Real.sqrt 33) / 7 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l2663_266312


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2663_266363

/-- Given an arithmetic sequence {aₙ} where the sum of the first n terms
    is Sₙ = 3n² + 2n, prove that the general term aₙ = 6n - 1 for all
    positive integers n. -/
theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (h_sum : ∀ n : ℕ, S n = 3 * n^2 + 2 * n)  -- Given condition
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Arithmetic sequence property
  : ∀ n : ℕ, n > 0 → a n = 6 * n - 1 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2663_266363


namespace NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_lines_perpendicular_to_same_plane_are_parallel_l2663_266300

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_to_plane : Plane → Plane → Prop)
variable (perpendicular_to_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1: Two planes parallel to the same plane are parallel
theorem planes_parallel_to_same_plane_are_parallel 
  (P Q R : Plane) 
  (h1 : parallel_to_plane P R) 
  (h2 : parallel_to_plane Q R) : 
  parallel_planes P Q :=
sorry

-- Theorem 2: Two lines perpendicular to the same plane are parallel
theorem lines_perpendicular_to_same_plane_are_parallel 
  (l1 l2 : Line) 
  (P : Plane) 
  (h1 : perpendicular_to_plane l1 P) 
  (h2 : perpendicular_to_plane l2 P) : 
  parallel_lines l1 l2 :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_lines_perpendicular_to_same_plane_are_parallel_l2663_266300


namespace NUMINAMATH_CALUDE_specific_l_shape_area_l2663_266327

/-- The area of an "L" shape formed by removing a smaller rectangle from a larger rectangle --/
def l_shape_area (length width subtract_length subtract_width : ℕ) : ℕ :=
  length * width - (length - subtract_length) * (width - subtract_width)

/-- Theorem: The area of the specific "L" shape is 42 square units --/
theorem specific_l_shape_area : l_shape_area 10 7 3 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_specific_l_shape_area_l2663_266327


namespace NUMINAMATH_CALUDE_root_value_theorem_l2663_266348

theorem root_value_theorem (a : ℝ) (h : a^2 + 2*a - 1 = 0) : 2*a^2 + 4*a - 2024 = -2022 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l2663_266348


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_geq_one_l2663_266345

/-- The function f(x) = x³ - ax² - x + 6 --/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - x + 6

/-- f is monotonically decreasing in the interval (0,1) --/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y ∧ y < 1 → f a x > f a y

theorem monotone_decreasing_implies_a_geq_one (a : ℝ) :
  is_monotone_decreasing a → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_geq_one_l2663_266345


namespace NUMINAMATH_CALUDE_scalene_polygon_existence_l2663_266347

theorem scalene_polygon_existence (n : ℕ) : 
  (n ≥ 13) → 
  (∀ (S : Finset ℝ), 
    (S.card = n) → 
    (∀ x ∈ S, 1 ≤ x ∧ x ≤ 2013) → 
    ∃ (a b c : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
      a + b > c ∧ b + c > a ∧ a + c > b) ∧
  (n = 13) :=
sorry

end NUMINAMATH_CALUDE_scalene_polygon_existence_l2663_266347


namespace NUMINAMATH_CALUDE_triple_sharp_fifty_l2663_266373

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.6 * N - 2

-- Theorem statement
theorem triple_sharp_fifty : sharp (sharp (sharp 50)) = 6.88 := by
  sorry

end NUMINAMATH_CALUDE_triple_sharp_fifty_l2663_266373


namespace NUMINAMATH_CALUDE_largest_number_l2663_266399

theorem largest_number : ∀ (a b c d : ℝ), 
  a = -|(-4)| ∧ b = 0 ∧ c = 1 ∧ d = -(-3) →
  d ≥ a ∧ d ≥ b ∧ d ≥ c :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l2663_266399


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2663_266351

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2663_266351


namespace NUMINAMATH_CALUDE_count_divisible_numbers_main_result_l2663_266368

theorem count_divisible_numbers (n : ℕ) (m : ℕ) : 
  (Finset.filter (fun k => (k^2 - 1) % m = 0) (Finset.range (n + 1))).card = 4 * (n / m) :=
by
  sorry

theorem main_result : 
  (Finset.filter (fun k => (k^2 - 1) % 485 = 0) (Finset.range 485001)).card = 4000 :=
by
  sorry

end NUMINAMATH_CALUDE_count_divisible_numbers_main_result_l2663_266368


namespace NUMINAMATH_CALUDE_find_g_x_l2663_266381

/-- Given that 4x^4 - 6x^2 + 2 + g(x) = 7x^3 - 3x^2 + 4x - 1 for all x,
    prove that g(x) = -4x^4 + 7x^3 + 3x^2 + 4x - 3 -/
theorem find_g_x (g : ℝ → ℝ) :
  (∀ x, 4 * x^4 - 6 * x^2 + 2 + g x = 7 * x^3 - 3 * x^2 + 4 * x - 1) →
  (∀ x, g x = -4 * x^4 + 7 * x^3 + 3 * x^2 + 4 * x - 3) := by
  sorry

end NUMINAMATH_CALUDE_find_g_x_l2663_266381


namespace NUMINAMATH_CALUDE_spade_calculation_l2663_266350

/-- Definition of the ♠ operation for real numbers -/
def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- Theorem stating that 6 ♠ (7 ♠ 7) = 36 -/
theorem spade_calculation : spade 6 (spade 7 7) = 36 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l2663_266350


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_ratio_l2663_266324

/-- Given two circles A and B where A is inside B, this theorem proves the diameter of A
    given the diameter of B and the ratio of areas. -/
theorem circle_diameter_from_area_ratio (dB : ℝ) (r : ℝ) :
  dB = 20 →  -- Diameter of circle B is 20 cm
  r = 1/7 →  -- Ratio of area of A to shaded area is 1:7
  ∃ dA : ℝ,  -- There exists a diameter for circle A
    (π * (dA/2)^2) / (π * (dB/2)^2 - π * (dA/2)^2) = r ∧  -- Area ratio condition
    abs (dA - 7.08) < 0.01  -- Diameter of A is approximately 7.08 cm
    := by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_ratio_l2663_266324


namespace NUMINAMATH_CALUDE_square_area_error_l2663_266394

theorem square_area_error (S : ℝ) (h : S > 0) :
  let measured_side := 1.05 * S
  let actual_area := S^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 10.25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l2663_266394


namespace NUMINAMATH_CALUDE_range_of_m_l2663_266398

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x - 4 < 0 → (x - m)*(x - m - 3) > 0) ∧ 
  (∃ x : ℝ, (x - m)*(x - m - 3) > 0 ∧ x^2 + 3*x - 4 ≥ 0) →
  m ≤ -7 ∨ m ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2663_266398


namespace NUMINAMATH_CALUDE_nikolai_faster_l2663_266355

/-- Represents a mountain goat with its jump distance and number of jumps in a given time -/
structure Goat where
  name : String
  jumpDistance : ℕ
  jumpsPerTime : ℕ

/-- Calculates the distance covered by a goat in one time unit -/
def distancePerTime (g : Goat) : ℕ := g.jumpDistance * g.jumpsPerTime

/-- Calculates the number of jumps needed to cover a given distance -/
def jumpsNeeded (g : Goat) (distance : ℕ) : ℕ :=
  (distance + g.jumpDistance - 1) / g.jumpDistance

theorem nikolai_faster (gennady nikolai : Goat) (totalDistance : ℕ) : 
  gennady.name = "Gennady" →
  nikolai.name = "Nikolai" →
  gennady.jumpDistance = 6 →
  gennady.jumpsPerTime = 2 →
  nikolai.jumpDistance = 4 →
  nikolai.jumpsPerTime = 3 →
  totalDistance = 2000 →
  distancePerTime gennady = distancePerTime nikolai →
  jumpsNeeded nikolai totalDistance < jumpsNeeded gennady totalDistance := by
  sorry

#eval jumpsNeeded (Goat.mk "Gennady" 6 2) 2000
#eval jumpsNeeded (Goat.mk "Nikolai" 4 3) 2000

end NUMINAMATH_CALUDE_nikolai_faster_l2663_266355


namespace NUMINAMATH_CALUDE_division_value_problem_l2663_266323

theorem division_value_problem (x : ℝ) : (4 / x) * 12 = 8 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_value_problem_l2663_266323


namespace NUMINAMATH_CALUDE_arrangements_with_specific_people_at_ends_l2663_266374

/-- The number of permutations of n distinct objects. -/
def permutations (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange m objects out of n distinct objects. -/
def arrangements (n m : ℕ) : ℕ := 
  if m ≤ n then
    permutations n / permutations (n - m)
  else
    0

theorem arrangements_with_specific_people_at_ends (total_people : ℕ) 
  (specific_people : ℕ) (h : total_people = 6 ∧ specific_people = 2) : 
  permutations total_people - 
  (arrangements (total_people - 2) specific_people * permutations (total_people - specific_people)) = 432 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_specific_people_at_ends_l2663_266374


namespace NUMINAMATH_CALUDE_joes_team_draws_l2663_266391

/-- Represents a team's performance in a soccer tournament --/
structure TeamPerformance where
  wins : ℕ
  draws : ℕ

/-- Calculates the total points for a team --/
def calculatePoints (team : TeamPerformance) : ℕ :=
  3 * team.wins + team.draws

theorem joes_team_draws : 
  ∀ (joes_team first_place : TeamPerformance),
    joes_team.wins = 1 →
    first_place.wins = 2 →
    first_place.draws = 2 →
    calculatePoints first_place = calculatePoints joes_team + 2 →
    joes_team.draws = 3 := by
  sorry


end NUMINAMATH_CALUDE_joes_team_draws_l2663_266391


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2663_266372

/-- 
Given a rectangular field with one side uncovered and three sides fenced,
prove that the area of the field is 720 square feet when the uncovered side
is 20 feet and the total fencing is 92 feet.
-/
theorem rectangular_field_area (L W : ℝ) : 
  L = 20 →  -- The uncovered side is 20 feet
  2 * W + L = 92 →  -- Total fencing equation
  L * W = 720  -- Area of the field
:= by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2663_266372


namespace NUMINAMATH_CALUDE_sqrt_five_decomposition_l2663_266354

theorem sqrt_five_decomposition (a : ℤ) (b : ℝ) 
  (h1 : Real.sqrt 5 = a + b) 
  (h2 : 0 < b) 
  (h3 : b < 1) : 
  (a - b) * (4 + Real.sqrt 5) = 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_decomposition_l2663_266354


namespace NUMINAMATH_CALUDE_cubic_roots_difference_squared_l2663_266317

theorem cubic_roots_difference_squared (r s : ℝ) : 
  (∃ c : ℝ, r^3 - 2*r + c = 0 ∧ s^3 - 2*s + c = 0 ∧ 1^3 - 2*1 + c = 0) →
  (r - s)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_difference_squared_l2663_266317


namespace NUMINAMATH_CALUDE_phase_shift_cosine_l2663_266303

/-- The phase shift of y = 3 cos(4x - π/4) is π/16 -/
theorem phase_shift_cosine (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 3 * Real.cos (4 * x - π / 4)
  ∃ (shift : ℝ), shift = π / 16 ∧ 
    ∀ (t : ℝ), f (t + shift) = 3 * Real.cos (4 * t) := by
  sorry

end NUMINAMATH_CALUDE_phase_shift_cosine_l2663_266303


namespace NUMINAMATH_CALUDE_closest_fraction_to_37_57_l2663_266332

theorem closest_fraction_to_37_57 :
  ∀ n : ℤ, n ≠ 15 → |37/57 - 15/23| < |37/57 - n/23| := by
sorry

end NUMINAMATH_CALUDE_closest_fraction_to_37_57_l2663_266332


namespace NUMINAMATH_CALUDE_one_fourth_of_8_4_l2663_266397

theorem one_fourth_of_8_4 : 
  ∃ (n d : ℕ), n ≠ 0 ∧ d ≠ 0 ∧ (8.4 / 4 : ℚ) = n / d ∧ Nat.gcd n d = 1 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_8_4_l2663_266397


namespace NUMINAMATH_CALUDE_average_age_problem_l2663_266386

theorem average_age_problem (a c : ℝ) : 
  (a + c) / 2 = 32 →
  ((a + c) + 23) / 3 = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_average_age_problem_l2663_266386


namespace NUMINAMATH_CALUDE_lord_moneybag_problem_l2663_266328

theorem lord_moneybag_problem :
  ∃! n : ℕ, 300 ≤ n ∧ n ≤ 500 ∧
  6 ∣ n ∧
  5 ∣ (n - 1) ∧
  4 ∣ (n - 2) ∧
  3 ∣ (n - 3) ∧
  2 ∣ (n - 4) ∧
  Nat.Prime (n - 5) := by
  sorry

end NUMINAMATH_CALUDE_lord_moneybag_problem_l2663_266328


namespace NUMINAMATH_CALUDE_indeterminate_relation_l2663_266310

theorem indeterminate_relation (x y : ℝ) (h : Real.exp (-x) + Real.log y < Real.exp (-y) + Real.log x) :
  ¬ (∀ p : Prop, p ∨ ¬p) := by
  sorry

end NUMINAMATH_CALUDE_indeterminate_relation_l2663_266310


namespace NUMINAMATH_CALUDE_line_properties_l2663_266388

/-- A line passing through a point with given conditions -/
structure Line where
  P : ℝ × ℝ
  α : ℝ
  intersects_positive_axes : Bool
  PA_PB_product : ℝ

/-- The main theorem stating the properties of the line -/
theorem line_properties (l : Line) 
  (h1 : l.P = (2, 1))
  (h2 : l.intersects_positive_axes = true)
  (h3 : l.PA_PB_product = 4) :
  (l.α = 3 * Real.pi / 4) ∧ 
  (∃ (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) = 3) := by
  sorry

#check line_properties

end NUMINAMATH_CALUDE_line_properties_l2663_266388


namespace NUMINAMATH_CALUDE_find_a_l2663_266319

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := (a * x - 1) / (x + 1) > 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | x < -1 ∨ x > 1/2}

-- Theorem statement
theorem find_a : ∃ a : ℝ, ∀ x : ℝ, inequality a x ↔ x ∈ solution_set a :=
sorry

end NUMINAMATH_CALUDE_find_a_l2663_266319


namespace NUMINAMATH_CALUDE_polygon_area_bounds_l2663_266337

/-- Represents a polygon in 2D space -/
structure Polygon where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents the projections of a polygon -/
structure Projections where
  x_axis : ℝ
  y_axis : ℝ
  bisector_1_3 : ℝ
  bisector_2_4 : ℝ

/-- Given a polygon, return its projections -/
def get_projections (p : Polygon) : Projections :=
  sorry

/-- Calculate the area of a polygon -/
def area (p : Polygon) : ℝ :=
  sorry

/-- Check if a polygon is convex -/
def is_convex (p : Polygon) : Prop :=
  sorry

theorem polygon_area_bounds (M : Polygon) 
    (h_proj : get_projections M = Projections.mk 4 5 (3 * Real.sqrt 2) (4 * Real.sqrt 2)) : 
  (area M ≤ 17.5) ∧ (is_convex M → area M ≥ 10) := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_bounds_l2663_266337


namespace NUMINAMATH_CALUDE_equation_solution_l2663_266393

theorem equation_solution :
  ∃! x : ℝ, ∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0 :=
by
  -- The unique value of x that satisfies the equation for all y is 3/2
  use 3/2
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2663_266393


namespace NUMINAMATH_CALUDE_identity_function_unique_l2663_266311

def PositiveInt := {n : ℤ // n > 0}

def DivisibilityCondition (f : PositiveInt → PositiveInt) : Prop :=
  ∀ a b : PositiveInt, (a.val - (f b).val) ∣ (a.val * (f a).val - b.val * (f b).val)

theorem identity_function_unique :
  ∀ f : PositiveInt → PositiveInt,
    DivisibilityCondition f →
    ∀ x : PositiveInt, f x = x :=
by
  sorry

end NUMINAMATH_CALUDE_identity_function_unique_l2663_266311


namespace NUMINAMATH_CALUDE_school_club_revenue_l2663_266389

/-- Represents the revenue from full-price tickets in a school club event. -/
def revenue_full_price (total_tickets : ℕ) (total_revenue : ℚ) : ℚ :=
  let full_price : ℚ := 30
  let full_price_tickets : ℕ := 45
  full_price * full_price_tickets

/-- Proves that the revenue from full-price tickets is $1350 given the conditions. -/
theorem school_club_revenue 
  (total_tickets : ℕ) 
  (total_revenue : ℚ) 
  (h_tickets : total_tickets = 160)
  (h_revenue : total_revenue = 2500) :
  revenue_full_price total_tickets total_revenue = 1350 := by
  sorry

#eval revenue_full_price 160 2500

end NUMINAMATH_CALUDE_school_club_revenue_l2663_266389


namespace NUMINAMATH_CALUDE_book_pages_book_has_120_pages_l2663_266362

theorem book_pages : ℕ → Prop :=
  fun total_pages =>
    let pages_yesterday : ℕ := 12
    let pages_today : ℕ := 2 * pages_yesterday
    let pages_read : ℕ := pages_yesterday + pages_today
    let pages_tomorrow : ℕ := 42
    let remaining_pages : ℕ := 2 * pages_tomorrow
    total_pages = pages_read + remaining_pages ∧ total_pages = 120

-- The proof of the theorem
theorem book_has_120_pages : ∃ (n : ℕ), book_pages n := by
  sorry

end NUMINAMATH_CALUDE_book_pages_book_has_120_pages_l2663_266362


namespace NUMINAMATH_CALUDE_problem_solution_l2663_266326

def f (a x : ℝ) : ℝ := a * x^2 + x - a

theorem problem_solution :
  (∀ x : ℝ, f 1 x ≥ 1 ↔ x ≤ -2 ∨ x ≥ 1) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x > -2*x^2 - 3*x + 1 - 2*a) ↔ a > 2) ∧
  (∀ a : ℝ, a < 0 →
    ((-1/2 < a ∧ a < 0 ∧ ∀ x : ℝ, f a x > 1 ↔ 1 < x ∧ x < -(a+1)/a) ∨
     (a = -1/2 ∧ ∀ x : ℝ, ¬(f a x > 1)) ∨
     (a < -1/2 ∧ ∀ x : ℝ, f a x > 1 ↔ -(a+1)/a < x ∧ x < 1))) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2663_266326


namespace NUMINAMATH_CALUDE_no_sequence_satisfying_condition_l2663_266301

theorem no_sequence_satisfying_condition :
  ¬ (∃ (a : ℝ) (a_n : ℕ → ℝ), 
    (0 < a ∧ a < 1) ∧
    (∀ n : ℕ, n > 0 → a_n n > 0) ∧
    (∀ n : ℕ, n > 0 → 1 + a_n (n + 1) ≤ a_n n + (a / n) * a_n n)) :=
by sorry

end NUMINAMATH_CALUDE_no_sequence_satisfying_condition_l2663_266301


namespace NUMINAMATH_CALUDE_ticket_sales_total_l2663_266376

/-- Calculates the total money collected from ticket sales -/
def total_money_collected (student_price general_price : ℕ) (total_tickets general_tickets : ℕ) : ℕ :=
  let student_tickets := total_tickets - general_tickets
  student_tickets * student_price + general_tickets * general_price

/-- Theorem stating that the total money collected is 2876 given the specific conditions -/
theorem ticket_sales_total :
  total_money_collected 4 6 525 388 = 2876 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_total_l2663_266376


namespace NUMINAMATH_CALUDE_xiao_ming_distance_l2663_266396

/-- The distance between Xiao Ming's house and school -/
def distance : ℝ := 1500

/-- The original planned speed in meters per minute -/
def original_speed : ℝ := 200

/-- The reduced speed due to rain in meters per minute -/
def reduced_speed : ℝ := 120

/-- The additional time taken due to reduced speed in minutes -/
def additional_time : ℝ := 5

theorem xiao_ming_distance :
  distance = original_speed * (distance / reduced_speed - additional_time) :=
sorry

end NUMINAMATH_CALUDE_xiao_ming_distance_l2663_266396


namespace NUMINAMATH_CALUDE_student_selection_and_advancement_probability_l2663_266305

-- Define the scores for students A and B
def scores_A : List ℕ := [100, 90, 120, 130, 105, 115]
def scores_B : List ℕ := [95, 125, 110, 95, 100, 135]

-- Define a function to calculate the average score
def average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

-- Define a function to calculate the variance
def variance (scores : List ℕ) : ℚ :=
  let avg := average scores
  (scores.map (λ x => ((x : ℚ) - avg) ^ 2)).sum / scores.length

-- Define a function to select the student with lower variance
def select_student (scores_A scores_B : List ℕ) : Bool :=
  variance scores_A < variance scores_B

-- Define the probability of advancing to the final round
def probability_advance : ℚ := 7 / 10

-- Theorem statement
theorem student_selection_and_advancement_probability 
  (scores_A scores_B : List ℕ) 
  (h_scores_A : scores_A = [100, 90, 120, 130, 105, 115])
  (h_scores_B : scores_B = [95, 125, 110, 95, 100, 135]) :
  select_student scores_A scores_B = true ∧ 
  probability_advance = 7 / 10 := by
  sorry


end NUMINAMATH_CALUDE_student_selection_and_advancement_probability_l2663_266305


namespace NUMINAMATH_CALUDE_lawn_area_l2663_266375

/-- Calculates the area of a lawn in a rectangular park with crossroads -/
theorem lawn_area (park_length park_width road_width : ℝ) 
  (h1 : park_length = 60)
  (h2 : park_width = 40)
  (h3 : road_width = 3) : 
  park_length * park_width - 
  (park_length * road_width + park_width * road_width - road_width * road_width) = 2109 :=
by sorry

end NUMINAMATH_CALUDE_lawn_area_l2663_266375


namespace NUMINAMATH_CALUDE_paper_clip_distribution_l2663_266361

theorem paper_clip_distribution (total_clips : ℕ) (clips_per_box : ℕ) (boxes_needed : ℕ) :
  total_clips = 81 →
  clips_per_box = 9 →
  total_clips = clips_per_box * boxes_needed →
  boxes_needed = 9 := by
  sorry

end NUMINAMATH_CALUDE_paper_clip_distribution_l2663_266361


namespace NUMINAMATH_CALUDE_multiply_specific_numbers_l2663_266315

theorem multiply_specific_numbers : 469138 * 9999 = 4690692862 := by
  sorry

end NUMINAMATH_CALUDE_multiply_specific_numbers_l2663_266315


namespace NUMINAMATH_CALUDE_pool_filling_times_l2663_266325

theorem pool_filling_times (t₁ t₂ : ℝ) : 
  (t₁ > 0 ∧ t₂ > 0) →  -- Ensure positive times
  (1 / t₁ + 1 / t₂ = 1 / 2.4) →  -- Combined filling rate
  (t₂ / (4 * t₁) + t₁ / (4 * t₂) = 11 / 24) →  -- Fraction filled by individual operations
  (t₁ = 4 ∧ t₂ = 6) :=
by sorry

end NUMINAMATH_CALUDE_pool_filling_times_l2663_266325


namespace NUMINAMATH_CALUDE_chef_apple_pies_l2663_266308

theorem chef_apple_pies (total pies : ℕ) (pecan pumpkin apple : ℕ) : 
  total = 13 → pecan = 4 → pumpkin = 7 → total = apple + pecan + pumpkin → apple = 2 := by
  sorry

end NUMINAMATH_CALUDE_chef_apple_pies_l2663_266308


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_one_l2663_266306

theorem factorization_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_one_l2663_266306


namespace NUMINAMATH_CALUDE_vacation_group_size_l2663_266371

def airbnb_cost : ℕ := 3200
def car_cost : ℕ := 800
def share_per_person : ℕ := 500

theorem vacation_group_size :
  (airbnb_cost + car_cost) / share_per_person = 8 :=
by sorry

end NUMINAMATH_CALUDE_vacation_group_size_l2663_266371


namespace NUMINAMATH_CALUDE_james_lifting_weight_l2663_266358

/-- Calculates the weight James can lift with straps for 10 meters given initial conditions -/
def weight_with_straps (initial_weight : ℝ) (distance_increase : ℝ) (short_distance_factor : ℝ) (strap_factor : ℝ) : ℝ :=
  let base_weight := initial_weight + distance_increase
  let short_distance_weight := base_weight * (1 + short_distance_factor)
  short_distance_weight * (1 + strap_factor)

/-- Theorem stating the final weight James can lift with straps for 10 meters -/
theorem james_lifting_weight :
  weight_with_straps 300 50 0.3 0.2 = 546 := by
  sorry

#eval weight_with_straps 300 50 0.3 0.2

end NUMINAMATH_CALUDE_james_lifting_weight_l2663_266358


namespace NUMINAMATH_CALUDE_gwen_recycling_points_l2663_266329

/-- Calculate points earned for recycling bags with increasing rewards -/
def calculate_points (base_points : ℕ) (increase_percent : ℚ) (bags_recycled : ℕ) : ℕ :=
  let points_list := List.range bags_recycled |>.map (fun i =>
    (base_points : ℚ) * (1 + increase_percent) ^ i)
  (points_list.sum).ceil.toNat

/-- Theorem stating the number of points Gwen would earn -/
theorem gwen_recycling_points : 
  let base_points : ℕ := 8
  let increase_percent : ℚ := 1/10
  let total_bags : ℕ := 4
  let unrecycled_bags : ℕ := 2
  let recycled_bags : ℕ := total_bags - unrecycled_bags
  calculate_points base_points increase_percent recycled_bags = 17 := by
  sorry

end NUMINAMATH_CALUDE_gwen_recycling_points_l2663_266329


namespace NUMINAMATH_CALUDE_pencil_distribution_l2663_266320

theorem pencil_distribution (num_pens : ℕ) (num_pencils : ℕ) (num_students : ℕ) :
  num_pens = 781 →
  num_students = 71 →
  num_pens % num_students = 0 →
  num_pencils % num_students = 0 →
  ∃ k : ℕ, num_pencils = 71 * k :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l2663_266320


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2663_266369

theorem angle_between_vectors (a b : ℝ × ℝ) :
  a = (Real.sqrt 3, 1) →
  b = (Real.sqrt 3, -1) →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2663_266369


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l2663_266395

def z : ℂ := Complex.I * (Complex.I + 2)

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l2663_266395


namespace NUMINAMATH_CALUDE_smallest_r_is_pi_over_two_l2663_266379

theorem smallest_r_is_pi_over_two :
  ∃ (r : ℝ) (f g : ℝ → ℝ), r > 0 ∧
    Differentiable ℝ f ∧ Differentiable ℝ g ∧
    f 0 > 0 ∧
    g 0 = 0 ∧
    (∀ x, |deriv f x| ≤ |g x|) ∧
    (∀ x, |deriv g x| ≤ |f x|) ∧
    f r = 0 ∧
    (∀ r' > 0, (∃ f' g' : ℝ → ℝ,
      Differentiable ℝ f' ∧ Differentiable ℝ g' ∧
      f' 0 > 0 ∧
      g' 0 = 0 ∧
      (∀ x, |deriv f' x| ≤ |g' x|) ∧
      (∀ x, |deriv g' x| ≤ |f' x|) ∧
      f' r' = 0) → r' ≥ r) ∧
    r = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_r_is_pi_over_two_l2663_266379


namespace NUMINAMATH_CALUDE_identity_function_proof_l2663_266353

theorem identity_function_proof (f : ℕ → ℕ) 
  (h : ∀ n : ℕ, f (n + 1) > f (f n)) : 
  ∀ n : ℕ, f n = n := by
  sorry

end NUMINAMATH_CALUDE_identity_function_proof_l2663_266353


namespace NUMINAMATH_CALUDE_max_y_value_l2663_266342

theorem max_y_value (x y : ℤ) (h : 2*x*y + 8*x + 2*y = -14) : 
  ∃ (max_y : ℤ), (∃ (x' : ℤ), 2*x'*max_y + 8*x' + 2*max_y = -14) ∧ 
  (∀ (y' : ℤ), (∃ (x'' : ℤ), 2*x''*y' + 8*x'' + 2*y' = -14) → y' ≤ max_y) ∧
  max_y = 5 := by
sorry

end NUMINAMATH_CALUDE_max_y_value_l2663_266342


namespace NUMINAMATH_CALUDE_tic_tac_toe_rounds_l2663_266330

/-- Given that William won 10 rounds of tic-tac-toe and 5 more rounds than Harry,
    prove that the total number of rounds played is 15. -/
theorem tic_tac_toe_rounds (william_rounds harry_rounds total_rounds : ℕ) 
  (h1 : william_rounds = 10)
  (h2 : william_rounds = harry_rounds + 5) : 
  total_rounds = 15 := by
  sorry

end NUMINAMATH_CALUDE_tic_tac_toe_rounds_l2663_266330


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l2663_266346

theorem complement_intersection_problem (U A B : Set ℕ) : 
  U = {1, 2, 3, 4, 5} →
  A = {2, 3, 4} →
  B = {3, 4, 5} →
  (Aᶜ ∩ Bᶜ : Set ℕ) = {1, 2, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l2663_266346


namespace NUMINAMATH_CALUDE_two_perpendicular_points_l2663_266384

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define a point on the ellipse
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

-- Define the foci of the ellipse
def focus1 : ℝ × ℝ := (2, 0)
def focus2 : ℝ × ℝ := (-2, 0)

-- Define perpendicularity condition
def perpendicularToFoci (p : PointOnEllipse) : Prop :=
  let v1 := (p.x - focus1.1, p.y - focus1.2)
  let v2 := (p.x - focus2.1, p.y - focus2.2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Theorem statement
theorem two_perpendicular_points :
  ∃! (s : Finset PointOnEllipse),
    (∀ p ∈ s, perpendicularToFoci p) ∧ s.card = 2 := by sorry

end NUMINAMATH_CALUDE_two_perpendicular_points_l2663_266384


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l2663_266302

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x + a| + 3

-- State the theorem
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x y, 1 < x ∧ x < y → f a x < f a y) → a ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l2663_266302


namespace NUMINAMATH_CALUDE_quadratic_minimum_min_value_is_zero_l2663_266339

theorem quadratic_minimum (x : ℝ) : 
  (∀ y : ℝ, x^2 - 12*x + 36 ≤ y^2 - 12*y + 36) ↔ x = 6 :=
by sorry

theorem min_value_is_zero : 
  (6:ℝ)^2 - 12*(6:ℝ) + 36 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_min_value_is_zero_l2663_266339


namespace NUMINAMATH_CALUDE_book_chapters_l2663_266344

theorem book_chapters (total_pages : ℕ) (pages_per_chapter : ℕ) 
  (h1 : total_pages = 1891) 
  (h2 : pages_per_chapter = 61) : 
  total_pages / pages_per_chapter = 31 := by
  sorry

end NUMINAMATH_CALUDE_book_chapters_l2663_266344


namespace NUMINAMATH_CALUDE_prime_solution_equation_l2663_266321

theorem prime_solution_equation : 
  ∃! (p q : ℕ), Prime p ∧ Prime q ∧ p^2 - 6*p*q + q^2 + 3*q - 1 = 0 ∧ p = 17 ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_solution_equation_l2663_266321


namespace NUMINAMATH_CALUDE_locus_of_centers_l2663_266357

/-- The locus of centers of circles externally tangent to C1 and internally tangent to C2 -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, 
    (a^2 + b^2 = (r + 1)^2) ∧ 
    ((a - 3)^2 + b^2 = (5 - r)^2)) ↔ 
  (4*a^2 + 4*b^2 - 6*a - 25 = 0) := by sorry

end NUMINAMATH_CALUDE_locus_of_centers_l2663_266357


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_third_and_fraction_l2663_266341

noncomputable def α : ℝ := Real.arctan 3 * 2

theorem tan_alpha_plus_pi_third_and_fraction (h : Real.tan (α/2) = 3) :
  Real.tan (α + π/3) = (48 - 25 * Real.sqrt 3) / 11 ∧
  (Real.sin α + 2 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = -5/17 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_third_and_fraction_l2663_266341


namespace NUMINAMATH_CALUDE_sock_drawing_probability_l2663_266340

/-- The number of colors of socks --/
def num_colors : ℕ := 5

/-- The number of socks per color --/
def socks_per_color : ℕ := 2

/-- The total number of socks --/
def total_socks : ℕ := num_colors * socks_per_color

/-- The number of socks drawn --/
def socks_drawn : ℕ := 5

/-- The probability of drawing exactly one pair of socks with the same color
    and the rest all different colors --/
theorem sock_drawing_probability : 
  (num_colors * (Nat.choose (num_colors - 1) (socks_drawn - 2)) * 
   (socks_per_color ^ 2) * (socks_per_color ^ (socks_drawn - 2))) /
  (Nat.choose total_socks socks_drawn) = 40 / 63 :=
by sorry

end NUMINAMATH_CALUDE_sock_drawing_probability_l2663_266340


namespace NUMINAMATH_CALUDE_z_absolute_value_range_l2663_266304

open Complex

theorem z_absolute_value_range (t : ℝ) :
  let z : ℂ := (sin t / Real.sqrt 2 + I * cos t) / (sin t - I * cos t / Real.sqrt 2)
  1 / Real.sqrt 2 ≤ abs z ∧ abs z ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_z_absolute_value_range_l2663_266304


namespace NUMINAMATH_CALUDE_x_equals_neg_x_valid_l2663_266390

/-- Represents a variable in a programming context -/
structure Variable where
  name : String

/-- Represents an expression in a programming context -/
inductive Expression where
  | Var : Variable → Expression
  | Num : Int → Expression
  | Neg : Expression → Expression
  | Add : Expression → Expression → Expression
  | Str : String → Expression

/-- Represents an assignment statement -/
structure Assignment where
  lhs : Expression
  rhs : Expression

/-- Predicate to check if an assignment is valid -/
def is_valid_assignment (a : Assignment) : Prop :=
  match a.lhs with
  | Expression.Var _ => True
  | _ => False

/-- Theorem stating that x = -x is a valid assignment -/
theorem x_equals_neg_x_valid :
  ∃ (x : Variable),
    is_valid_assignment { lhs := Expression.Var x, rhs := Expression.Neg (Expression.Var x) } ∧
    ¬is_valid_assignment { lhs := Expression.Num 5, rhs := Expression.Str "M" } ∧
    ¬is_valid_assignment { lhs := Expression.Add (Expression.Var ⟨"x"⟩) (Expression.Var ⟨"y"⟩), rhs := Expression.Num 0 } :=
by
  sorry


end NUMINAMATH_CALUDE_x_equals_neg_x_valid_l2663_266390


namespace NUMINAMATH_CALUDE_correct_observation_value_l2663_266338

theorem correct_observation_value 
  (n : ℕ) 
  (initial_mean : ℝ) 
  (wrong_value : ℝ) 
  (corrected_mean : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 36)
  (h3 : wrong_value = 23)
  (h4 : corrected_mean = 36.5) : 
  ∃ (correct_value : ℝ), correct_value = 48 ∧ 
    n * corrected_mean = n * initial_mean - wrong_value + correct_value :=
by sorry

end NUMINAMATH_CALUDE_correct_observation_value_l2663_266338


namespace NUMINAMATH_CALUDE_simplify_radicals_l2663_266314

theorem simplify_radicals : 
  Real.sqrt 10 - Real.sqrt 40 + Real.sqrt 90 + Real.sqrt 160 = 6 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radicals_l2663_266314


namespace NUMINAMATH_CALUDE_system_solution_l2663_266318

-- Define the system of equations
def equation1 (x y : ℝ) : Prop :=
  (x^4 / y^2)^(Real.log y) = (-x)^(Real.log (-x*y))

def equation2 (x y : ℝ) : Prop :=
  2*y^2 - x*y - x^2 - 4*x - 8*y = 0

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {(-4, 2), (-2, 2), ((Real.sqrt 17 - 9)/2, (Real.sqrt 17 - 1)/2)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solution_l2663_266318


namespace NUMINAMATH_CALUDE_unit_digit_sum_factorials_l2663_266316

def factorial (n : ℕ) : ℕ := Nat.factorial n

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def unit_digit (n : ℕ) : ℕ := n % 10

theorem unit_digit_sum_factorials :
  unit_digit (sum_factorials 2012) = unit_digit (sum_factorials 4) :=
sorry

end NUMINAMATH_CALUDE_unit_digit_sum_factorials_l2663_266316


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2663_266352

/-- The equation of an ellipse with parameter k -/
def is_ellipse (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (3 + k) + y^2 / (2 - k) = 1 ∧
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

/-- The range of k for which the equation represents an ellipse -/
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ k ∈ Set.Ioo (-3 : ℝ) (-1/2) ∪ Set.Ioo (-1/2 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2663_266352


namespace NUMINAMATH_CALUDE_base_7_digits_of_2000_l2663_266385

/-- The number of digits in the base-7 representation of a positive integer -/
def num_digits_base_7 (n : ℕ+) : ℕ :=
  Nat.log 7 n.val + 1

/-- Theorem: The base-7 representation of 2000 has 4 digits -/
theorem base_7_digits_of_2000 : num_digits_base_7 2000 = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_7_digits_of_2000_l2663_266385


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2663_266331

theorem geometric_sequence_sum (n : ℕ) : 
  let a : ℚ := 1/3
  let r : ℚ := 2/3
  let sum : ℚ := a * (1 - r^n) / (1 - r)
  sum = 80/243 → n = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2663_266331


namespace NUMINAMATH_CALUDE_negation_of_false_is_true_l2663_266322

theorem negation_of_false_is_true (p q : Prop) 
  (hp : p) (hq : ¬q) : ¬q := by sorry

end NUMINAMATH_CALUDE_negation_of_false_is_true_l2663_266322


namespace NUMINAMATH_CALUDE_john_burritos_days_l2663_266370

theorem john_burritos_days (boxes : ℕ) (burritos_per_box : ℕ) (fraction_given_away : ℚ)
  (burritos_eaten_per_day : ℕ) (burritos_left : ℕ) :
  boxes = 3 →
  burritos_per_box = 20 →
  fraction_given_away = 1 / 3 →
  burritos_eaten_per_day = 3 →
  burritos_left = 10 →
  (boxes * burritos_per_box * (1 - fraction_given_away) - burritos_left) / burritos_eaten_per_day = 10 := by
  sorry

#check john_burritos_days

end NUMINAMATH_CALUDE_john_burritos_days_l2663_266370


namespace NUMINAMATH_CALUDE_xy_value_l2663_266392

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : Real.sqrt (Real.log x) + Real.sqrt (Real.log y) + 
          Real.log (Real.sqrt x) + Real.log (Real.sqrt y) + 
          Real.log (x^(1/4)) + Real.log (y^(1/4)) = 150)
  (h_int1 : ∃ n : ℤ, Real.sqrt (Real.log x) = n)
  (h_int2 : ∃ n : ℤ, Real.sqrt (Real.log y) = n)
  (h_int3 : ∃ n : ℤ, Real.log (Real.sqrt x) = n)
  (h_int4 : ∃ n : ℤ, Real.log (Real.sqrt y) = n)
  (h_int5 : ∃ n : ℤ, Real.log (x^(1/4)) = n)
  (h_int6 : ∃ n : ℤ, Real.log (y^(1/4)) = n) :
  x * y = Real.exp 340 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2663_266392


namespace NUMINAMATH_CALUDE_staff_members_count_correct_staff_count_l2663_266377

theorem staff_members_count (allowance_days : ℕ) (allowance_rate : ℕ) 
  (accountant_amount : ℕ) (petty_cash : ℕ) : ℕ :=
  let allowance_per_staff := allowance_days * allowance_rate
  let total_amount := accountant_amount + petty_cash
  total_amount / allowance_per_staff

theorem correct_staff_count : 
  staff_members_count 30 100 65000 1000 = 22 := by sorry

end NUMINAMATH_CALUDE_staff_members_count_correct_staff_count_l2663_266377


namespace NUMINAMATH_CALUDE_non_sum_sequence_inequality_l2663_266309

/-- A sequence of positive integers where no element can be represented as the sum of two or more different elements from the sequence -/
def NonSumSequence (m : Nat → Nat) : Prop :=
  ∀ (i j k : Nat), i < j → j < k → m i + m j ≠ m k

theorem non_sum_sequence_inequality
  (m : Nat → Nat)  -- The sequence
  (s : Nat)        -- The length of the sequence
  (h_s : s ≥ 2)    -- s is at least 2
  (h_m : ∀ i j, i < j → j ≤ s → m i < m j)  -- The sequence is strictly increasing
  (h_non_sum : NonSumSequence m)  -- The non-sum property
  (r : Nat)        -- The parameter r
  (h_r : 1 ≤ r ∧ r < s)  -- r is between 1 and s-1
  : r * m r + m s ≥ (r + 1) * (s - 1) :=
sorry

end NUMINAMATH_CALUDE_non_sum_sequence_inequality_l2663_266309


namespace NUMINAMATH_CALUDE_cos_minus_sin_for_point_l2663_266307

theorem cos_minus_sin_for_point (α : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = Real.sqrt 3 ∧ r * Real.sin α = -1) →
  Real.cos α - Real.sin α = (Real.sqrt 3 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_minus_sin_for_point_l2663_266307


namespace NUMINAMATH_CALUDE_cheryl_material_calculation_l2663_266349

/-- The amount of the second type of material Cheryl needed -/
def second_material : ℚ := 1 / 8

/-- The amount of material Cheryl has left unused -/
def unused_material : ℚ := 4 / 18

/-- The total amount of material Cheryl used -/
def used_material : ℚ := 125 / 1000

/-- The amount of the first type of material Cheryl needed -/
def first_material : ℚ := 2222 / 10000

theorem cheryl_material_calculation :
  first_material = (unused_material + used_material) - second_material :=
by sorry

end NUMINAMATH_CALUDE_cheryl_material_calculation_l2663_266349


namespace NUMINAMATH_CALUDE_dimes_per_quarter_l2663_266366

/-- Represents the number of coins traded for a quarter -/
structure TradeRatio :=
  (dimes : ℚ)
  (nickels : ℚ)

/-- Calculates the total value of coins traded -/
def totalValue (ratio : TradeRatio) : ℚ :=
  20 * (ratio.dimes * (1/10) + ratio.nickels * (1/20))

/-- Theorem: The number of dimes traded for each quarter is 4 -/
theorem dimes_per_quarter :
  ∃ (ratio : TradeRatio),
    totalValue ratio = 10 + 3 ∧
    ratio.nickels = 5 ∧
    ratio.dimes = 4 := by
  sorry

end NUMINAMATH_CALUDE_dimes_per_quarter_l2663_266366


namespace NUMINAMATH_CALUDE_triangle_isosceles_condition_l2663_266343

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a/cos(A) = b/cos(B), then the triangle is isosceles. -/
theorem triangle_isosceles_condition (a b c A B C : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- Angles are in (0, π)
  A + B + C = π →  -- Sum of angles in a triangle
  a / Real.cos A = b / Real.cos B →  -- Given condition
  a = b  -- Conclusion: triangle is isosceles
  := by sorry

end NUMINAMATH_CALUDE_triangle_isosceles_condition_l2663_266343


namespace NUMINAMATH_CALUDE_bing_duan_duan_properties_l2663_266387

/-- Represents the production and sales of "Bing Duan Duan" mascots --/
structure BingDuanDuan where
  feb_production : ℕ
  apr_production : ℕ
  daily_sales : ℕ
  profit_per_item : ℕ
  sales_increase : ℕ
  max_price_reduction : ℕ
  target_daily_profit : ℕ

/-- Calculates the monthly growth rate given February and April production --/
def monthly_growth_rate (b : BingDuanDuan) : ℚ :=
  ((b.apr_production : ℚ) / b.feb_production) ^ (1/2) - 1

/-- Calculates the optimal price reduction --/
def optimal_price_reduction (b : BingDuanDuan) : ℕ :=
  sorry -- The actual calculation would go here

/-- Theorem stating the properties of BingDuanDuan production and sales --/
theorem bing_duan_duan_properties (b : BingDuanDuan) 
  (h1 : b.feb_production = 500)
  (h2 : b.apr_production = 720)
  (h3 : b.daily_sales = 20)
  (h4 : b.profit_per_item = 40)
  (h5 : b.sales_increase = 5)
  (h6 : b.max_price_reduction = 10)
  (h7 : b.target_daily_profit = 1440) :
  monthly_growth_rate b = 1/5 ∧ 
  optimal_price_reduction b = 4 ∧ 
  optimal_price_reduction b ≤ b.max_price_reduction :=
by sorry


end NUMINAMATH_CALUDE_bing_duan_duan_properties_l2663_266387


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2663_266364

theorem solution_set_quadratic_inequality :
  {x : ℝ | -x^2 + 5*x > 6} = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2663_266364


namespace NUMINAMATH_CALUDE_snow_leopard_arrangement_l2663_266335

theorem snow_leopard_arrangement (n : ℕ) (h : n = 7) : 
  2 * Nat.factorial (n - 2) = 240 := by
  sorry

end NUMINAMATH_CALUDE_snow_leopard_arrangement_l2663_266335


namespace NUMINAMATH_CALUDE_orange_balls_count_l2663_266360

theorem orange_balls_count (total green red blue yellow pink orange purple : ℕ) :
  total = 120 ∧
  green = 5 ∧
  red = 30 ∧
  blue = 20 ∧
  yellow = 10 ∧
  pink = 2 * green ∧
  orange = 3 * pink ∧
  purple = orange - pink ∧
  total = red + blue + yellow + green + pink + orange + purple →
  orange = 30 := by
sorry

end NUMINAMATH_CALUDE_orange_balls_count_l2663_266360


namespace NUMINAMATH_CALUDE_henry_returned_half_l2663_266365

/-- The portion of catch Henry returned -/
def henryReturnedPortion (willCatfish : ℕ) (willEels : ℕ) (henryTroutPerCatfish : ℕ) (totalFishAfterReturn : ℕ) : ℚ :=
  let willTotal := willCatfish + willEels
  let henryTotal := willCatfish * henryTroutPerCatfish
  let totalBeforeReturn := willTotal + henryTotal
  let returnedFish := totalBeforeReturn - totalFishAfterReturn
  returnedFish / henryTotal

/-- Theorem stating that Henry returned half of his catch -/
theorem henry_returned_half :
  henryReturnedPortion 16 10 3 50 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_henry_returned_half_l2663_266365


namespace NUMINAMATH_CALUDE_angle_bisector_length_l2663_266382

/-- Given a triangle ABC with sides b and c, and angle A between them,
    prove that the length of the angle bisector of A is (2bc cos(A/2)) / (b + c) -/
theorem angle_bisector_length (b c A : ℝ) (hb : b > 0) (hc : c > 0) (hA : 0 < A ∧ A < π) :
  let S := (1/2) * b * c * Real.sin A
  let l_a := (2 * b * c * Real.cos (A/2)) / (b + c)
  ∀ S', S' = S → l_a = (2 * b * c * Real.cos (A/2)) / (b + c) :=
by sorry

end NUMINAMATH_CALUDE_angle_bisector_length_l2663_266382


namespace NUMINAMATH_CALUDE_system_solution_l2663_266336

theorem system_solution (x y z : ℝ) : 
  x^2 + y^2 + z^2 = 1 ∧ x^3 + y^3 + z^3 = 1 → 
  (x = 1 ∧ y = 0 ∧ z = 0) ∨ (x = 0 ∧ y = 1 ∧ z = 0) ∨ (x = 0 ∧ y = 0 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2663_266336


namespace NUMINAMATH_CALUDE_total_lemons_picked_l2663_266333

theorem total_lemons_picked (sally_lemons mary_lemons : ℕ) 
  (h1 : sally_lemons = 7)
  (h2 : mary_lemons = 9) :
  sally_lemons + mary_lemons = 16 := by
sorry

end NUMINAMATH_CALUDE_total_lemons_picked_l2663_266333


namespace NUMINAMATH_CALUDE_transformation_of_point_l2663_266359

/-- Given a point A and a transformation φ, prove that the transformed point A' has specific coordinates -/
theorem transformation_of_point (x y x' y' : ℚ) : 
  x = 1/3 ∧ y = -2 ∧ x' = 3*x ∧ 2*y' = y → x' = 1 ∧ y' = -1 := by
  sorry

end NUMINAMATH_CALUDE_transformation_of_point_l2663_266359
