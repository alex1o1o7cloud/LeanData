import Mathlib

namespace NUMINAMATH_CALUDE_soap_brand_usage_ratio_l2350_235002

/-- Given a survey of households and their soap brand usage, prove the ratio of households
    using only brand B to those using both brands A and B. -/
theorem soap_brand_usage_ratio
  (total_households : ℕ)
  (neither_brand : ℕ)
  (only_brand_A : ℕ)
  (both_brands : ℕ)
  (h1 : total_households = 180)
  (h2 : neither_brand = 80)
  (h3 : only_brand_A = 60)
  (h4 : both_brands = 10)
  (h5 : total_households = neither_brand + only_brand_A + (total_households - neither_brand - only_brand_A - both_brands) + both_brands) :
  (total_households - neither_brand - only_brand_A - both_brands) / both_brands = 3 := by
  sorry

end NUMINAMATH_CALUDE_soap_brand_usage_ratio_l2350_235002


namespace NUMINAMATH_CALUDE_average_b_c_l2350_235062

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45) 
  (h2 : c - a = 30) : 
  (b + c) / 2 = 60 := by
sorry

end NUMINAMATH_CALUDE_average_b_c_l2350_235062


namespace NUMINAMATH_CALUDE_halloween_decorations_l2350_235059

/-- Calculates the number of plastic skulls in Danai's Halloween decorations. -/
theorem halloween_decorations (total_decorations : ℕ) (broomsticks : ℕ) (spiderwebs : ℕ) 
  (cauldron : ℕ) (budget_left : ℕ) (left_to_put_up : ℕ) 
  (h1 : total_decorations = 83)
  (h2 : broomsticks = 4)
  (h3 : spiderwebs = 12)
  (h4 : cauldron = 1)
  (h5 : budget_left = 20)
  (h6 : left_to_put_up = 10) :
  total_decorations - (broomsticks + spiderwebs + 2 * spiderwebs + cauldron + budget_left + left_to_put_up) = 12 := by
  sorry

end NUMINAMATH_CALUDE_halloween_decorations_l2350_235059


namespace NUMINAMATH_CALUDE_hyperbola_line_inclination_l2350_235063

/-- Given a hyperbola with equation x²/m² - y²/n² = 1 and eccentricity 2,
    prove that the angle of inclination of the line mx + ny - 1 = 0
    is either π/6 or 5π/6 -/
theorem hyperbola_line_inclination (m n : ℝ) (h_eccentricity : m^2 + n^2 = 4 * m^2) :
  let θ := Real.arctan (-m / n)
  θ = π / 6 ∨ θ = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_line_inclination_l2350_235063


namespace NUMINAMATH_CALUDE_valid_numbers_count_l2350_235071

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    b = 2 * c ∧
    b = (a + c) / 2

theorem valid_numbers_count :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, is_valid_number n) ∧
    s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_count_l2350_235071


namespace NUMINAMATH_CALUDE_triangle_with_unit_inradius_is_right_angled_l2350_235083

/-- A triangle with integer side lengths and inradius 1 is right-angled with sides (3, 4, 5) -/
theorem triangle_with_unit_inradius_is_right_angled (a b c : ℕ) (r : ℝ) :
  r = 1 →
  (a : ℝ) + (b : ℝ) + (c : ℝ) = 2 * ((a : ℝ) * (b : ℝ) * (c : ℝ)) / ((a : ℝ) + (b : ℝ) + (c : ℝ)) →
  (a = 3 ∧ b = 4 ∧ c = 5) ∨ (a = 3 ∧ b = 5 ∧ c = 4) ∨
  (a = 4 ∧ b = 3 ∧ c = 5) ∨ (a = 4 ∧ b = 5 ∧ c = 3) ∨
  (a = 5 ∧ b = 3 ∧ c = 4) ∨ (a = 5 ∧ b = 4 ∧ c = 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_unit_inradius_is_right_angled_l2350_235083


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_2_range_of_a_l2350_235064

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (a x : ℝ) : ℝ := 2 * |x - a|

-- Statement for the first part of the problem
theorem solution_set_when_a_eq_2 :
  {x : ℝ | f x - g 2 x ≤ x - 3} = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := by sorry

-- Statement for the second part of the problem
theorem range_of_a :
  (∀ m > 1, ∃ x₀ : ℝ, f x₀ + g a x₀ ≤ (m^2 + m + 4) / (m - 1)) →
  a ∈ Set.Icc (-2 * Real.sqrt 6 - 2) (2 * Real.sqrt 6 + 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_2_range_of_a_l2350_235064


namespace NUMINAMATH_CALUDE_min_third_highest_score_is_95_l2350_235056

def min_third_highest_score (scores : List ℤ) : Prop :=
  scores.length = 6 ∧
  scores.Nodup ∧
  scores.sum / scores.length = 555 / 6 ∧
  scores.maximum = some 99 ∧
  scores.minimum = some 76 ∧
  ∃ (third_highest : ℤ), third_highest ∈ scores ∧
    (scores.filter (λ x => x > third_highest)).length = 2 ∧
    third_highest ≥ 95

theorem min_third_highest_score_is_95 :
  ∀ scores : List ℤ, min_third_highest_score scores →
    ∃ (third_highest : ℤ), third_highest ∈ scores ∧
      (scores.filter (λ x => x > third_highest)).length = 2 ∧
      third_highest = 95 :=
by
  sorry

end NUMINAMATH_CALUDE_min_third_highest_score_is_95_l2350_235056


namespace NUMINAMATH_CALUDE_stable_yield_promotion_l2350_235008

/-- Represents a type of red rice -/
structure RedRice where
  typeName : String
  averageYield : ℝ
  variance : ℝ

/-- Determines if a type of red rice is suitable for promotion based on yield stability -/
def isSuitableForPromotion (rice1 rice2 : RedRice) : Prop :=
  rice1.averageYield = rice2.averageYield ∧ 
  rice1.variance < rice2.variance

theorem stable_yield_promotion (A B : RedRice) 
  (h_yield : A.averageYield = B.averageYield)
  (h_variance : A.variance < B.variance) : 
  isSuitableForPromotion A B := by
  sorry

#check stable_yield_promotion

end NUMINAMATH_CALUDE_stable_yield_promotion_l2350_235008


namespace NUMINAMATH_CALUDE_scientific_notation_32000000_l2350_235091

theorem scientific_notation_32000000 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 32000000 = a * (10 : ℝ) ^ n ∧ a = 3.2 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_32000000_l2350_235091


namespace NUMINAMATH_CALUDE_arrangement_remainder_l2350_235066

/-- The number of blue marbles --/
def blue_marbles : ℕ := 6

/-- The maximum number of yellow marbles that can be arranged with the blue marbles
    such that the number of marbles with same-color neighbors equals the number of
    marbles with different-color neighbors --/
def max_yellow_marbles : ℕ := 17

/-- The total number of marbles --/
def total_marbles : ℕ := blue_marbles + max_yellow_marbles

/-- The number of possible arrangements of the marbles --/
def num_arrangements : ℕ := Nat.choose total_marbles blue_marbles

theorem arrangement_remainder :
  num_arrangements % 1000 = 376 := by sorry

end NUMINAMATH_CALUDE_arrangement_remainder_l2350_235066


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l2350_235049

/-- The number of large seats on the Ferris wheel -/
def num_large_seats : ℕ := 7

/-- The total number of people that can be accommodated on large seats -/
def total_people_large_seats : ℕ := 84

/-- The number of people each large seat can hold -/
def people_per_large_seat : ℕ := total_people_large_seats / num_large_seats

theorem ferris_wheel_capacity : people_per_large_seat = 12 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l2350_235049


namespace NUMINAMATH_CALUDE_production_difference_formula_l2350_235094

/-- Represents the widget production scenario for David --/
structure WidgetProduction where
  /-- Widgets produced per hour on Monday --/
  w : ℕ
  /-- Hours worked on Monday --/
  t : ℕ
  /-- Relationship between w and t --/
  w_eq_3t : w = 3 * t

/-- Calculates the difference in widget production between Monday and Tuesday --/
def productionDifference (p : WidgetProduction) : ℕ :=
  let monday_production := p.w * p.t
  let tuesday_production := (p.w + 6) * (p.t - 3)
  monday_production - tuesday_production

/-- Theorem stating the difference in widget production --/
theorem production_difference_formula (p : WidgetProduction) :
  productionDifference p = 3 * p.t + 18 := by
  sorry

#check production_difference_formula

end NUMINAMATH_CALUDE_production_difference_formula_l2350_235094


namespace NUMINAMATH_CALUDE_rectangular_box_diagonals_l2350_235042

theorem rectangular_box_diagonals 
  (a b c : ℝ) 
  (surface_area : 2 * (a * b + b * c + c * a) = 166) 
  (edge_sum : 4 * (a + b + c) = 64) : 
  4 * Real.sqrt (a^2 + b^2 + c^2) = 12 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_diagonals_l2350_235042


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l2350_235046

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l2350_235046


namespace NUMINAMATH_CALUDE_derivative_bounded_l2350_235043

open Real

/-- Given a function f: ℝ → ℝ with continuous second derivative, 
    and both f and f'' are bounded, prove that f' is also bounded. -/
theorem derivative_bounded (f : ℝ → ℝ) (hf'' : Continuous (deriv (deriv f))) 
  (hf_bdd : ∃ M, ∀ x, |f x| ≤ M) (hf''_bdd : ∃ M, ∀ x, |(deriv (deriv f)) x| ≤ M) :
  ∃ K, ∀ x, |deriv f x| ≤ K := by
  sorry

end NUMINAMATH_CALUDE_derivative_bounded_l2350_235043


namespace NUMINAMATH_CALUDE_transform_f_to_g_l2350_235007

/-- The original function -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- The resulting function after transformation -/
def g (x : ℝ) : ℝ := (x - 5)^2 + 5

/-- Vertical shift transformation -/
def vertical_shift (h : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := h x + k

/-- Horizontal shift transformation -/
def horizontal_shift (h : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := h (x - k)

/-- Theorem stating that the transformation of f results in g -/
theorem transform_f_to_g : 
  ∀ x, horizontal_shift (vertical_shift f 3) 4 x = g x :=
sorry

end NUMINAMATH_CALUDE_transform_f_to_g_l2350_235007


namespace NUMINAMATH_CALUDE_sin_30_degrees_l2350_235057

theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l2350_235057


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equations_l2350_235000

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem unique_function_satisfying_equations (f : RealFunction) :
  (∀ x : ℝ, f (x + 1) = 1 + f x) ∧
  (∀ x : ℝ, f (x^4 - x^2) = f x^4 - f x^2) →
  ∀ x : ℝ, f x = x :=
by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equations_l2350_235000


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l2350_235053

/-- A sequence (a, b, c) is geometric if there exists a non-zero real number r such that b = a * r and c = b * r. -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ b = a * r ∧ c = b * r

/-- The condition ac = b^2 is necessary but not sufficient for a, b, c to form a geometric sequence. -/
theorem geometric_sequence_condition (a b c : ℝ) :
  (IsGeometricSequence a b c → a * c = b ^ 2) ∧
  ¬(a * c = b ^ 2 → IsGeometricSequence a b c) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l2350_235053


namespace NUMINAMATH_CALUDE_derivative_at_three_l2350_235013

/-- Given a function f(x) = -x^2 + 10, prove that its derivative at x = 3 is -3. -/
theorem derivative_at_three (f : ℝ → ℝ) (h : ∀ x, f x = -x^2 + 10) :
  deriv f 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_three_l2350_235013


namespace NUMINAMATH_CALUDE_bob_muffins_l2350_235096

theorem bob_muffins (total : ℕ) (days : ℕ) (increment : ℕ) (second_day : ℚ) : 
  total = 55 → 
  days = 4 → 
  increment = 2 → 
  (∃ (first_day : ℚ), 
    first_day + (first_day + ↑increment) + (first_day + 2 * ↑increment) + (first_day + 3 * ↑increment) = total ∧
    second_day = first_day + ↑increment) →
  second_day = 12.75 := by sorry

end NUMINAMATH_CALUDE_bob_muffins_l2350_235096


namespace NUMINAMATH_CALUDE_square_roots_problem_l2350_235058

theorem square_roots_problem (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (2*a + 6)^2 = x ∧ (3 - a)^2 = x) → a = -9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l2350_235058


namespace NUMINAMATH_CALUDE_camp_cedar_counselors_l2350_235040

/-- The number of counselors needed at Camp Cedar -/
def counselors_needed (num_boys : ℕ) (girl_to_boy_ratio : ℕ) (children_per_counselor : ℕ) : ℕ :=
  let num_girls := num_boys * girl_to_boy_ratio
  let total_children := num_boys + num_girls
  total_children / children_per_counselor

/-- Theorem stating the number of counselors needed at Camp Cedar -/
theorem camp_cedar_counselors :
  counselors_needed 40 3 8 = 20 := by
  sorry

#eval counselors_needed 40 3 8

end NUMINAMATH_CALUDE_camp_cedar_counselors_l2350_235040


namespace NUMINAMATH_CALUDE_diving_class_capacity_l2350_235038

/-- The number of people that can be accommodated in each diving class -/
def people_per_class : ℕ := 5

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The number of classes per weekday -/
def classes_per_weekday : ℕ := 2

/-- The number of classes per weekend day -/
def classes_per_weekend_day : ℕ := 4

/-- The number of weeks -/
def weeks : ℕ := 3

/-- The total number of people that can take classes in 3 weeks -/
def total_people : ℕ := 270

/-- Theorem stating that the number of people per class is 5 -/
theorem diving_class_capacity :
  people_per_class = 
    total_people / (weeks * (weekdays * classes_per_weekday + weekend_days * classes_per_weekend_day)) :=
by sorry

end NUMINAMATH_CALUDE_diving_class_capacity_l2350_235038


namespace NUMINAMATH_CALUDE_average_temperature_l2350_235009

/-- The average temperature of three days with recorded temperatures of -14°F, -8°F, and +1°F is -7°F. -/
theorem average_temperature (temp1 temp2 temp3 : ℚ) : 
  temp1 = -14 → temp2 = -8 → temp3 = 1 → (temp1 + temp2 + temp3) / 3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_l2350_235009


namespace NUMINAMATH_CALUDE_seating_arrangements_l2350_235098

-- Define the number of people excluding the fixed person
def n : ℕ := 4

-- Define the function to calculate the total number of permutations
def total_permutations (n : ℕ) : ℕ := n.factorial

-- Define the function to calculate the number of permutations where two specific people are adjacent
def adjacent_permutations (n : ℕ) : ℕ := 2 * (n - 1).factorial

-- Theorem statement
theorem seating_arrangements :
  total_permutations n - adjacent_permutations n = 12 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l2350_235098


namespace NUMINAMATH_CALUDE_female_students_count_l2350_235026

theorem female_students_count (total_students sample_size : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : ∃ (sampled_girls sampled_boys : ℕ), 
    sampled_girls + sampled_boys = sample_size ∧ 
    sampled_boys = sampled_girls + 10) :
  ∃ (female_students : ℕ), female_students = 760 ∧ 
    female_students * sample_size = sampled_girls * total_students :=
by
  sorry


end NUMINAMATH_CALUDE_female_students_count_l2350_235026


namespace NUMINAMATH_CALUDE_perpendicular_slope_l2350_235010

theorem perpendicular_slope (x₁ y₁ x₂ y₂ : ℚ) (hx : x₁ ≠ x₂) :
  let m₁ := (y₂ - y₁) / (x₂ - x₁)
  let m₂ := -1 / m₁
  x₁ = 3 ∧ y₁ = -3 ∧ x₂ = -4 ∧ y₂ = 2 → m₂ = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l2350_235010


namespace NUMINAMATH_CALUDE_i_pow_2006_l2350_235078

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the properties of i
axiom i_pow_1 : i^1 = i
axiom i_pow_2 : i^2 = -1
axiom i_pow_3 : i^3 = -i
axiom i_pow_4 : i^4 = 1
axiom i_pow_5 : i^5 = i

-- Theorem to prove
theorem i_pow_2006 : i^2006 = -1 := by
  sorry

end NUMINAMATH_CALUDE_i_pow_2006_l2350_235078


namespace NUMINAMATH_CALUDE_area_triangle_on_hyperbola_l2350_235097

/-- The area of a triangle formed by three points on the curve xy = 1 -/
theorem area_triangle_on_hyperbola (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h₁ : x₁ * y₁ = 1) 
  (h₂ : x₂ * y₂ = 1) 
  (h₃ : x₃ * y₃ = 1) 
  (h₄ : x₁ ≠ 0) 
  (h₅ : x₂ ≠ 0) 
  (h₆ : x₃ ≠ 0) :
  let t := abs ((x₁ - x₂) * (x₂ - x₃) * (x₃ - x₁)) / (2 * x₁ * x₂ * x₃)
  t = abs (1/2 * (x₁ * y₂ + x₂ * y₃ + x₃ * y₁ - x₂ * y₁ - x₃ * y₂ - x₁ * y₃)) := by
  sorry


end NUMINAMATH_CALUDE_area_triangle_on_hyperbola_l2350_235097


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l2350_235084

/-- Calculates the total wet surface area of a rectangular cistern -/
def cisternWetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem: The wet surface area of a cistern with given dimensions is 134 square meters -/
theorem cistern_wet_surface_area :
  cisternWetSurfaceArea 10 8 1.5 = 134 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l2350_235084


namespace NUMINAMATH_CALUDE_seminar_duration_is_428_l2350_235030

/-- Represents the duration of a seminar session in minutes -/
def seminar_duration (first_part_hours : ℕ) (first_part_minutes : ℕ) (second_part_minutes : ℕ) (closing_event_seconds : ℕ) : ℕ :=
  (first_part_hours * 60 + first_part_minutes) + second_part_minutes + (closing_event_seconds / 60)

/-- Theorem stating that the seminar duration is 428 minutes given the specified conditions -/
theorem seminar_duration_is_428 :
  seminar_duration 4 45 135 500 = 428 := by
  sorry

end NUMINAMATH_CALUDE_seminar_duration_is_428_l2350_235030


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2350_235020

theorem necessary_not_sufficient_condition 
  (A B C : Set α) 
  (hAnonempty : A.Nonempty) 
  (hBnonempty : B.Nonempty) 
  (hCnonempty : C.Nonempty) 
  (hUnion : A ∪ B = C) 
  (hNotSubset : ¬(B ⊆ A)) :
  (∀ x, x ∈ A → x ∈ C) ∧ 
  (∃ y, y ∈ C ∧ y ∉ A) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2350_235020


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l2350_235044

theorem smallest_common_multiple_of_6_and_15 (b : ℕ) : 
  (b % 6 = 0 ∧ b % 15 = 0) → b ≥ 30 :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l2350_235044


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l2350_235015

/-- Two lines are parallel if their slopes are equal and they are not coincident -/
def parallel (a b c d e f : ℝ) : Prop :=
  a / d = b / e ∧ a / d ≠ c / f

/-- First line equation: ax + 2y + 3a = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 3 * a = 0

/-- Second line equation: 3x + (a-1)y + a^2 - a + 3 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  3 * x + (a - 1) * y + a^2 - a + 3 = 0

/-- Theorem stating that a=3 is neither sufficient nor necessary for the lines to be parallel -/
theorem not_sufficient_nor_necessary :
  ¬(∀ a : ℝ, a = 3 → parallel a 2 (3*a) 3 (a-1) (a^2 - a + 3)) ∧
  ¬(∀ a : ℝ, parallel a 2 (3*a) 3 (a-1) (a^2 - a + 3) → a = 3) :=
sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l2350_235015


namespace NUMINAMATH_CALUDE_quadratic_coincidence_l2350_235023

-- Define the type for 2D points
def Point := ℝ × ℝ

-- Define a line in 2D
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a quadratic function
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the concept of a segment cut by a quadratic function on a line
def SegmentCut (f : QuadraticFunction) (l : Line) : ℝ :=
  sorry

-- Non-parallel lines
def NonParallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b ≠ l₁.b * l₂.a

-- Theorem statement
theorem quadratic_coincidence (f₁ f₂ : QuadraticFunction) (l₁ l₂ : Line) :
  NonParallel l₁ l₂ →
  SegmentCut f₁ l₁ = SegmentCut f₂ l₁ →
  SegmentCut f₁ l₂ = SegmentCut f₂ l₂ →
  f₁ = f₂ :=
sorry

end NUMINAMATH_CALUDE_quadratic_coincidence_l2350_235023


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l2350_235055

/-- Given polynomials f, g, and h, prove their sum equals the simplified polynomial -/
theorem sum_of_polynomials (x : ℝ) :
  let f := fun x : ℝ => -4 * x^2 + 2 * x - 5
  let g := fun x : ℝ => -6 * x^2 + 4 * x - 9
  let h := fun x : ℝ => 6 * x^2 + 6 * x + 2
  f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l2350_235055


namespace NUMINAMATH_CALUDE_license_plate_increase_l2350_235039

theorem license_plate_increase : 
  let old_format := 26^2 * 10^3
  let new_format := 26^4 * 10^4
  (new_format / old_format : ℚ) = 2600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_increase_l2350_235039


namespace NUMINAMATH_CALUDE_even_function_condition_l2350_235014

/-- A function f is even if f(-x) = f(x) for all x in ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x+a)(x-4) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * (x - 4)

theorem even_function_condition (a : ℝ) : IsEven (f a) ↔ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_condition_l2350_235014


namespace NUMINAMATH_CALUDE_triple_area_right_triangle_l2350_235077

/-- Given a right triangle with hypotenuse a+b and legs a and b, 
    the area of a triangle that is three times the area of this right triangle is 3/2ab. -/
theorem triple_area_right_triangle (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  3 * (1/2 * a * b) = 3/2 * a * b := by sorry

end NUMINAMATH_CALUDE_triple_area_right_triangle_l2350_235077


namespace NUMINAMATH_CALUDE_deepak_present_age_l2350_235093

-- Define the ages as natural numbers
variable (R D : ℕ)

-- Define the conditions
def ratio_condition : Prop := 4 * D = 3 * R
def future_age_condition : Prop := R + 6 = 26

-- Theorem statement
theorem deepak_present_age 
  (h1 : ratio_condition R D) 
  (h2 : future_age_condition R) : 
  D = 15 := by sorry

end NUMINAMATH_CALUDE_deepak_present_age_l2350_235093


namespace NUMINAMATH_CALUDE_quadratic_nonnegative_conditions_l2350_235017

theorem quadratic_nonnegative_conditions (a b c : ℝ) (ha : a ≠ 0)
  (hf : ∀ x : ℝ, a * x^2 + 2 * b * x + c ≥ 0) :
  a > 0 ∧ c ≥ 0 ∧ a * c - b^2 ≥ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_nonnegative_conditions_l2350_235017


namespace NUMINAMATH_CALUDE_division_problem_l2350_235033

theorem division_problem : (160 : ℝ) / (10 + 11 * 2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2350_235033


namespace NUMINAMATH_CALUDE_nested_root_simplification_l2350_235029

theorem nested_root_simplification :
  (81 * Real.sqrt (27 * Real.sqrt 9)) ^ (1/4) = 3 * 9 ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_nested_root_simplification_l2350_235029


namespace NUMINAMATH_CALUDE_hawk_percentage_is_25_percent_l2350_235001

/-- Represents the percentage of hawks in the bird population -/
def hawk_percentage : ℝ := sorry

/-- Represents the percentage of paddyfield-warblers in the bird population -/
def paddyfield_warbler_percentage : ℝ := sorry

/-- Represents the percentage of kingfishers in the bird population -/
def kingfisher_percentage : ℝ := sorry

/-- The percentage of non-hawks that are paddyfield-warblers -/
def paddyfield_warbler_ratio : ℝ := 0.4

/-- The ratio of kingfishers to paddyfield-warblers -/
def kingfisher_to_warbler_ratio : ℝ := 0.25

/-- The percentage of birds that are not hawks, paddyfield-warblers, or kingfishers -/
def other_birds_percentage : ℝ := 0.35

theorem hawk_percentage_is_25_percent :
  hawk_percentage = 0.25 ∧
  paddyfield_warbler_percentage = paddyfield_warbler_ratio * (1 - hawk_percentage) ∧
  kingfisher_percentage = kingfisher_to_warbler_ratio * paddyfield_warbler_percentage ∧
  hawk_percentage + paddyfield_warbler_percentage + kingfisher_percentage + other_birds_percentage = 1 :=
by sorry

end NUMINAMATH_CALUDE_hawk_percentage_is_25_percent_l2350_235001


namespace NUMINAMATH_CALUDE_lock_combination_l2350_235024

/-- Represents a digit in the cryptarithmetic problem -/
structure Digit where
  value : Nat
  is_valid : value < 10

/-- Represents the base of the number system -/
structure Base where
  value : Nat
  is_valid : value > 1

/-- Function to convert a number from base b to base 10 -/
def to_decimal (digits : List Digit) (b : Base) : Nat :=
  sorry

/-- The cryptarithmetic equation -/
def cryptarithmetic_equation (T I D E : Digit) (b : Base) : Prop :=
  to_decimal [T, I, D, E] b + to_decimal [E, D, I, T] b + to_decimal [T, I, D, E] b
  = to_decimal [D, I, E, T] b

/-- All digits are distinct -/
def all_distinct (T I D E : Digit) : Prop :=
  T.value ≠ I.value ∧ T.value ≠ D.value ∧ T.value ≠ E.value ∧
  I.value ≠ D.value ∧ I.value ≠ E.value ∧ D.value ≠ E.value

theorem lock_combination :
  ∃ (T I D E : Digit) (b : Base),
    cryptarithmetic_equation T I D E b ∧
    all_distinct T I D E ∧
    to_decimal [T, I, D] (Base.mk 10 sorry) = 984 :=
  sorry

end NUMINAMATH_CALUDE_lock_combination_l2350_235024


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2350_235034

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a ≥ 6 ∧
  ((a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a = 6 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2350_235034


namespace NUMINAMATH_CALUDE_borrowed_amount_with_interest_l2350_235092

/-- Calculates the total amount to be returned given a borrowed amount and an interest rate. -/
def totalAmount (borrowed : ℝ) (interestRate : ℝ) : ℝ :=
  borrowed * (1 + interestRate)

/-- Proves that given a borrowed amount of $100 and an agreed increase of 10%, 
    the total amount to be returned is $110. -/
theorem borrowed_amount_with_interest : 
  totalAmount 100 0.1 = 110 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_amount_with_interest_l2350_235092


namespace NUMINAMATH_CALUDE_solution_ratio_proof_l2350_235035

/-- Proves that the ratio of solutions A and B is 1:1 when mixed to form a 45% alcohol solution --/
theorem solution_ratio_proof (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) : 
  (4/9 * a + 5/11 * b) / (a + b) = 9/20 → a = b :=
by
  sorry

end NUMINAMATH_CALUDE_solution_ratio_proof_l2350_235035


namespace NUMINAMATH_CALUDE_lineman_drinks_eight_ounces_l2350_235027

/-- Represents the water consumption scenario of a football team -/
structure WaterConsumption where
  linemen_count : ℕ
  skill_players_count : ℕ
  cooler_capacity : ℕ
  skill_player_consumption : ℕ
  waiting_skill_players : ℕ

/-- Calculates the amount of water each lineman drinks -/
def lineman_consumption (wc : WaterConsumption) : ℚ :=
  let skill_players_drinking := wc.skill_players_count - wc.waiting_skill_players
  let total_skill_consumption := skill_players_drinking * wc.skill_player_consumption
  (wc.cooler_capacity - total_skill_consumption) / wc.linemen_count

/-- Theorem stating that each lineman drinks 8 ounces of water -/
theorem lineman_drinks_eight_ounces (wc : WaterConsumption) 
  (h1 : wc.linemen_count = 12)
  (h2 : wc.skill_players_count = 10)
  (h3 : wc.cooler_capacity = 126)
  (h4 : wc.skill_player_consumption = 6)
  (h5 : wc.waiting_skill_players = 5) :
  lineman_consumption wc = 8 := by
  sorry

#eval lineman_consumption {
  linemen_count := 12,
  skill_players_count := 10,
  cooler_capacity := 126,
  skill_player_consumption := 6,
  waiting_skill_players := 5
}

end NUMINAMATH_CALUDE_lineman_drinks_eight_ounces_l2350_235027


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2350_235032

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 2) :
  ∃ (m : ℝ), (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 2 → x + y ≤ a + b) ∧ x + y = m ∧ m = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2350_235032


namespace NUMINAMATH_CALUDE_candy_mixture_cost_l2350_235068

/-- Given a mixture of two types of candy, prove the cost of the first type. -/
theorem candy_mixture_cost
  (total_mixture : ℝ)
  (selling_price : ℝ)
  (expensive_candy_amount : ℝ)
  (expensive_candy_price : ℝ)
  (h1 : total_mixture = 80)
  (h2 : selling_price = 2.20)
  (h3 : expensive_candy_amount = 16)
  (h4 : expensive_candy_price = 3) :
  ∃ (cheap_candy_price : ℝ),
    cheap_candy_price * (total_mixture - expensive_candy_amount) +
    expensive_candy_price * expensive_candy_amount =
    selling_price * total_mixture ∧
    cheap_candy_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_l2350_235068


namespace NUMINAMATH_CALUDE_charitable_distribution_result_l2350_235021

def charitable_distribution (initial : ℕ) : ℕ :=
  let to_farmer := initial / 2 + 1
  let after_farmer := initial - to_farmer
  let to_beggar := after_farmer / 2 + 2
  let after_beggar := after_farmer - to_beggar
  let to_boy := after_beggar / 2 + 3
  after_beggar - to_boy

theorem charitable_distribution_result :
  charitable_distribution 42 = 1 := by
  sorry

end NUMINAMATH_CALUDE_charitable_distribution_result_l2350_235021


namespace NUMINAMATH_CALUDE_probability_of_pair_l2350_235070

/-- Represents a deck of cards with their counts -/
def Deck := List (Nat × Nat)

/-- The initial deck configuration -/
def initial_deck : Deck := List.replicate 10 (5, 5)

/-- Remove a matching pair from the deck -/
def remove_pair (d : Deck) : Deck :=
  match d with
  | (n, count) :: rest => if count ≥ 2 then (n, count - 2) :: rest else d
  | [] => []

/-- Calculate the total number of cards in the deck -/
def total_cards (d : Deck) : Nat :=
  d.foldr (fun (_, count) acc => acc + count) 0

/-- Calculate the number of ways to choose 2 cards from n cards -/
def choose_2 (n : Nat) : Nat := n * (n - 1) / 2

/-- Calculate the number of possible pairs in the deck -/
def count_pairs (d : Deck) : Nat :=
  d.foldr (fun (_, count) acc => acc + choose_2 count) 0

theorem probability_of_pair (d : Deck) :
  let remaining_deck := remove_pair initial_deck
  let total := total_cards remaining_deck
  let pairs := count_pairs remaining_deck
  (pairs : Rat) / (choose_2 total) = 31 / 376 := by sorry

end NUMINAMATH_CALUDE_probability_of_pair_l2350_235070


namespace NUMINAMATH_CALUDE_furniture_markup_proof_l2350_235004

/-- Calculates the percentage markup given the selling price and cost price -/
def percentage_markup (selling_price cost_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Proves that the percentage markup is 25% for the given selling and cost prices -/
theorem furniture_markup_proof (selling_price cost_price : ℚ) 
  (h1 : selling_price = 4800)
  (h2 : cost_price = 3840) : 
  percentage_markup selling_price cost_price = 25 := by
  sorry

end NUMINAMATH_CALUDE_furniture_markup_proof_l2350_235004


namespace NUMINAMATH_CALUDE_line_intercept_sum_l2350_235075

/-- Given a line 5x + 8y + c = 0, if the sum of its x-intercept and y-intercept is 26, then c = -80 -/
theorem line_intercept_sum (c : ℝ) : 
  (∃ x y : ℝ, 5*x + 8*y + c = 0 ∧ 5*x + c = 0 ∧ 8*y + c = 0 ∧ x + y = 26) → 
  c = -80 :=
by sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l2350_235075


namespace NUMINAMATH_CALUDE_heather_aprons_l2350_235074

/-- The number of aprons Heather sewed before today -/
def aprons_before_today : ℕ := by sorry

/-- The total number of aprons to be sewn -/
def total_aprons : ℕ := 150

/-- The number of aprons Heather sewed today -/
def aprons_today : ℕ := 3 * aprons_before_today

/-- The number of aprons Heather will sew tomorrow -/
def aprons_tomorrow : ℕ := 49

/-- The number of remaining aprons after sewing tomorrow -/
def remaining_aprons : ℕ := aprons_tomorrow

theorem heather_aprons : 
  aprons_before_today = 13 ∧
  aprons_before_today + aprons_today + aprons_tomorrow + remaining_aprons = total_aprons := by
  sorry

end NUMINAMATH_CALUDE_heather_aprons_l2350_235074


namespace NUMINAMATH_CALUDE_largest_prime_factor_3434_l2350_235054

def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_3434 : largest_prime_factor 3434 = 7 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_3434_l2350_235054


namespace NUMINAMATH_CALUDE_zoom_setup_ratio_l2350_235022

/-- Represents the time spent on various activities during Mary's Zoom setup and call -/
structure ZoomSetup where
  mac_download : ℕ
  windows_download : ℕ
  audio_glitch_duration : ℕ
  audio_glitch_count : ℕ
  video_glitch_duration : ℕ
  total_time : ℕ

/-- Calculates the ratio of time spent talking without glitches to time spent with glitches -/
def talkTimeRatio (setup : ZoomSetup) : Rat :=
  let total_download_time := setup.mac_download + setup.windows_download
  let total_glitch_time := setup.audio_glitch_duration * setup.audio_glitch_count + setup.video_glitch_duration
  let total_talk_time := setup.total_time - total_download_time
  let talk_time_without_glitches := total_talk_time - total_glitch_time
  talk_time_without_glitches / total_glitch_time

/-- Theorem stating that given the specific conditions, the talk time ratio is 2:1 -/
theorem zoom_setup_ratio : 
  ∀ (setup : ZoomSetup), 
    setup.mac_download = 10 ∧ 
    setup.windows_download = 3 * setup.mac_download ∧
    setup.audio_glitch_duration = 4 ∧
    setup.audio_glitch_count = 2 ∧
    setup.video_glitch_duration = 6 ∧
    setup.total_time = 82 →
    talkTimeRatio setup = 2 := by
  sorry

end NUMINAMATH_CALUDE_zoom_setup_ratio_l2350_235022


namespace NUMINAMATH_CALUDE_polynomial_real_root_l2350_235082

theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, x^4 - a*x^3 - x^2 - a*x + 1 = 0) ↔ a ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l2350_235082


namespace NUMINAMATH_CALUDE_robins_gum_problem_l2350_235081

/-- Given that Robin initially had 18 pieces of gum and now has 44 pieces in total,
    prove that Robin's brother gave her 26 pieces of gum. -/
theorem robins_gum_problem (initial : ℕ) (total : ℕ) (h1 : initial = 18) (h2 : total = 44) :
  total - initial = 26 := by
  sorry

end NUMINAMATH_CALUDE_robins_gum_problem_l2350_235081


namespace NUMINAMATH_CALUDE_train_speed_l2350_235012

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length crossing_time : Real) 
  (h1 : train_length = 140)
  (h2 : bridge_length = 235.03)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45.0036 := by
  sorry

#eval (140 + 235.03) / 30 * 3.6

end NUMINAMATH_CALUDE_train_speed_l2350_235012


namespace NUMINAMATH_CALUDE_promotions_equivalent_l2350_235076

/-- Calculates the discount percentage for a given promotion --/
def discount_percentage (items_taken : ℕ) (items_paid : ℕ) : ℚ :=
  (items_taken - items_paid : ℚ) / items_taken * 100

/-- The original promotion "Buy one and get another for half price" --/
def original_promotion : ℚ := discount_percentage 2 (3/2)

/-- The alternative promotion "Take four and pay for three" --/
def alternative_promotion : ℚ := discount_percentage 4 3

/-- Theorem stating that both promotions offer the same discount --/
theorem promotions_equivalent : original_promotion = alternative_promotion := by
  sorry

end NUMINAMATH_CALUDE_promotions_equivalent_l2350_235076


namespace NUMINAMATH_CALUDE_division_problem_l2350_235073

theorem division_problem : (72 : ℚ) / ((6 : ℚ) / 3) = 36 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2350_235073


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2350_235019

/-- The sum of an arithmetic sequence with first term a₁ = k^2 - k + 1 and common difference d = 2 for k^2 terms -/
theorem arithmetic_sequence_sum (k : ℕ) : 
  let a₁ : ℤ := k^2 - k + 1
  let d : ℤ := 2
  let n : ℕ := k^2
  let Sₙ : ℤ := n * (2 * a₁ + (n - 1) * d) / 2
  Sₙ = 2 * k^4 - k^3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2350_235019


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2350_235072

theorem complex_modulus_problem (z : ℂ) : z = (3 - Complex.I) / (1 + 2 * Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2350_235072


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l2350_235089

theorem consecutive_pages_sum (n : ℕ) : n > 0 ∧ n + (n + 1) = 185 → n = 92 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l2350_235089


namespace NUMINAMATH_CALUDE_quadratic_roots_bound_l2350_235067

-- Define the quadratic function
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_roots_bound (a b : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f a b (f a b x₁) = 0 ∧ f a b (f a b x₂) = 0 ∧ f a b (f a b x₃) = 0 ∧ f a b (f a b x₄) = 0) →
  (∃ y₁ y₂ : ℝ, f a b (f a b y₁) = 0 ∧ f a b (f a b y₂) = 0 ∧ y₁ + y₂ = -1) →
  b ≤ -1/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_bound_l2350_235067


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l2350_235045

theorem simplify_radical_expression :
  ∃ (a b c : ℕ+), 
    (((Real.sqrt 2 - 1) ^ (2 - Real.sqrt 3)) / ((Real.sqrt 2 + 1) ^ (2 + Real.sqrt 3)) = 
     (3 + 2 * Real.sqrt 2) ^ Real.sqrt 3) ∧
    (∀ (p : ℕ), Nat.Prime p → ¬(p ^ 2 ∣ c.val)) :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l2350_235045


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_problem_l2350_235061

theorem consecutive_odd_numbers_problem :
  ∀ x y z : ℤ,
  (y = x + 2) →
  (z = x + 4) →
  (8 * x = 3 * z + 2 * y + 5) →
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_problem_l2350_235061


namespace NUMINAMATH_CALUDE_stratified_sampling_arts_students_l2350_235005

theorem stratified_sampling_arts_students 
  (total_students : ℕ) 
  (arts_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 1000)
  (h2 : arts_students = 200)
  (h3 : sample_size = 100) :
  (arts_students : ℚ) / total_students * sample_size = 20 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_arts_students_l2350_235005


namespace NUMINAMATH_CALUDE_missing_number_equation_l2350_235086

theorem missing_number_equation (x : ℤ) : 10010 - 12 * 3 * x = 9938 ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_equation_l2350_235086


namespace NUMINAMATH_CALUDE_custom_operations_simplification_specific_case_l2350_235069

/-- Custom addition operation for rational numbers -/
def star (a b : ℚ) : ℚ := a + b

/-- Custom subtraction operation for rational numbers -/
def otimes (a b : ℚ) : ℚ := a - b

/-- The main theorem -/
theorem custom_operations_simplification (a b : ℚ) :
  star (a^2 * b) (3 * a * b) + otimes (5 * a^2 * b) (4 * a * b) = 6 * a^2 * b - a * b :=
sorry

/-- The specific case for a = 5 and b = 3 -/
theorem specific_case :
  star ((5:ℚ)^2 * 3) (3 * 5 * 3) + otimes (5 * (5:ℚ)^2 * 3) (4 * 5 * 3) = 435 :=
sorry

end NUMINAMATH_CALUDE_custom_operations_simplification_specific_case_l2350_235069


namespace NUMINAMATH_CALUDE_fraction_addition_l2350_235087

theorem fraction_addition : (2 : ℚ) / 5 + (3 : ℚ) / 9 = (11 : ℚ) / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2350_235087


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2350_235065

theorem sufficient_but_not_necessary (x : ℝ) :
  (|x - 2| < 1 → x^2 + x - 2 > 0) ∧ 
  ∃ y : ℝ, (y^2 + y - 2 > 0 ∧ ¬(|y - 2| < 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2350_235065


namespace NUMINAMATH_CALUDE_bert_spent_nine_at_dry_cleaners_l2350_235088

/-- Represents Bert's spending problem -/
def BertSpending (initial_amount dry_cleaner_amount : ℚ) : Prop :=
  let hardware_store_amount := initial_amount / 4
  let after_hardware := initial_amount - hardware_store_amount
  let after_dry_cleaner := after_hardware - dry_cleaner_amount
  let grocery_store_amount := after_dry_cleaner / 2
  let final_amount := after_dry_cleaner - grocery_store_amount
  (initial_amount = 52) ∧
  (final_amount = 15) ∧
  (dry_cleaner_amount > 0)

/-- Proves that Bert spent $9 at the dry cleaners -/
theorem bert_spent_nine_at_dry_cleaners :
  ∃ (dry_cleaner_amount : ℚ), BertSpending 52 dry_cleaner_amount ∧ dry_cleaner_amount = 9 := by
  sorry

end NUMINAMATH_CALUDE_bert_spent_nine_at_dry_cleaners_l2350_235088


namespace NUMINAMATH_CALUDE_problem_statement_l2350_235031

theorem problem_statement (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 152) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 154 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2350_235031


namespace NUMINAMATH_CALUDE_max_value_of_f_on_I_l2350_235018

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*x - 1

-- Define the interval
def I : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem max_value_of_f_on_I :
  ∃ (M : ℝ), M = 2 ∧ ∀ x ∈ I, f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_I_l2350_235018


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l2350_235048

theorem triangle_third_side_length
  (a b c : ℕ)
  (h1 : a = 2)
  (h2 : b = 5)
  (h3 : Odd c)
  (h4 : a + b > c)
  (h5 : b + c > a)
  (h6 : c + a > b) :
  c = 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l2350_235048


namespace NUMINAMATH_CALUDE_z_percentage_of_x_l2350_235095

theorem z_percentage_of_x (x y z : ℝ) 
  (h1 : 0.45 * z = 1.2 * y) 
  (h2 : y = 0.75 * x) : 
  z = 2 * x := by
sorry

end NUMINAMATH_CALUDE_z_percentage_of_x_l2350_235095


namespace NUMINAMATH_CALUDE_age_of_b_l2350_235090

/-- Given three people a, b, and c, with their ages represented as natural numbers. -/
def problem (a b c : ℕ) : Prop :=
  -- The average age of a, b, and c is 26 years
  (a + b + c) / 3 = 26 ∧
  -- The average age of a and c is 29 years
  (a + c) / 2 = 29 →
  -- The age of b is 20 years
  b = 20

/-- Theorem stating that under the given conditions, the age of b must be 20 years -/
theorem age_of_b (a b c : ℕ) : problem a b c := by
  sorry

end NUMINAMATH_CALUDE_age_of_b_l2350_235090


namespace NUMINAMATH_CALUDE_arthur_walked_four_point_five_miles_l2350_235016

/-- The distance Arthur walked in miles -/
def arthurs_distance (blocks_west : ℕ) (blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Theorem stating that Arthur walked 4.5 miles -/
theorem arthur_walked_four_point_five_miles :
  arthurs_distance 8 10 (1/4) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walked_four_point_five_miles_l2350_235016


namespace NUMINAMATH_CALUDE_sum_squares_consecutive_integers_l2350_235051

theorem sum_squares_consecutive_integers (a : ℤ) :
  let S := (a - 2)^2 + (a - 1)^2 + a^2 + (a + 1)^2 + (a + 2)^2
  ∃ k : ℤ, S = 5 * k ∧ ¬∃ m : ℤ, S = 25 * m :=
by sorry

end NUMINAMATH_CALUDE_sum_squares_consecutive_integers_l2350_235051


namespace NUMINAMATH_CALUDE_football_team_throwers_l2350_235041

/-- Represents the number of throwers on a football team. -/
def num_throwers : ℕ := 52

/-- Represents the total number of players on the football team. -/
def total_players : ℕ := 70

/-- Represents the total number of right-handed players on the team. -/
def right_handed_players : ℕ := 64

theorem football_team_throwers :
  num_throwers = 52 ∧
  total_players = 70 ∧
  right_handed_players = 64 ∧
  num_throwers ≤ total_players ∧
  num_throwers ≤ right_handed_players ∧
  (total_players - num_throwers) % 3 = 0 ∧
  right_handed_players = num_throwers + 2 * ((total_players - num_throwers) / 3) :=
by sorry

end NUMINAMATH_CALUDE_football_team_throwers_l2350_235041


namespace NUMINAMATH_CALUDE_stuffed_animals_theorem_l2350_235085

/-- Given the number of stuffed animals for McKenna (M), Kenley (K), and Tenly (T),
    prove various properties about their stuffed animal collection. -/
theorem stuffed_animals_theorem (M K T : ℕ) (S : ℕ) (A F : ℚ) 
    (hM : M = 34)
    (hK : K = 2 * M)
    (hT : T = K + 5)
    (hS : S = M + K + T)
    (hA : A = S / 3)
    (hF : F = M / S) : 
  K = 68 ∧ 
  T = 73 ∧ 
  S = 175 ∧ 
  A = 175 / 3 ∧ 
  F = 34 / 175 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_animals_theorem_l2350_235085


namespace NUMINAMATH_CALUDE_ten_customers_miss_sunday_paper_l2350_235028

/-- Represents Kyle's newspaper delivery route -/
structure NewspaperRoute where
  totalHouses : ℕ
  dailyDeliveries : ℕ
  sundayOnlyDeliveries : ℕ
  weeklyTotalDeliveries : ℕ

/-- Calculates the number of customers who do not get the Sunday paper -/
def customersMissingSundayPaper (route : NewspaperRoute) : ℕ :=
  route.totalHouses - (route.totalHouses - (route.weeklyTotalDeliveries - 6 * route.totalHouses - route.sundayOnlyDeliveries))

/-- Theorem stating that 10 customers do not get the Sunday paper -/
theorem ten_customers_miss_sunday_paper (route : NewspaperRoute) 
  (h1 : route.totalHouses = 100)
  (h2 : route.dailyDeliveries = 100)
  (h3 : route.sundayOnlyDeliveries = 30)
  (h4 : route.weeklyTotalDeliveries = 720) :
  customersMissingSundayPaper route = 10 := by
  sorry

#eval customersMissingSundayPaper { totalHouses := 100, dailyDeliveries := 100, sundayOnlyDeliveries := 30, weeklyTotalDeliveries := 720 }

end NUMINAMATH_CALUDE_ten_customers_miss_sunday_paper_l2350_235028


namespace NUMINAMATH_CALUDE_quadratic_sum_l2350_235037

theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, -3 * x^2 + 15 * x + 75 = a * (x + b)^2 + c) →
  a + b + c = 353/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2350_235037


namespace NUMINAMATH_CALUDE_toothpicks_15th_stage_l2350_235003

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ :=
  3 + 2 * (n - 1)

/-- Theorem stating that the 15th stage of the pattern has 31 toothpicks -/
theorem toothpicks_15th_stage :
  toothpicks 15 = 31 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_15th_stage_l2350_235003


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l2350_235025

theorem polynomial_root_sum (c d : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + c * (Complex.I * Real.sqrt 2 + 2) + d = 0 → 
  c + d = 14 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l2350_235025


namespace NUMINAMATH_CALUDE_onions_in_basket_l2350_235079

/-- Given a basket of onions with initial count S, prove that after
    Sara adds 4, Sally removes 5, and Fred adds F onions, 
    resulting in 8 more onions than the initial count,
    Fred must have added 9 onions. -/
theorem onions_in_basket (S : ℤ) : ∃ F : ℤ, 
  S - 1 + F = S + 8 ∧ F = 9 := by
  sorry

end NUMINAMATH_CALUDE_onions_in_basket_l2350_235079


namespace NUMINAMATH_CALUDE_power_multiplication_l2350_235050

theorem power_multiplication (a : ℝ) : a^3 * a^4 = a^7 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_l2350_235050


namespace NUMINAMATH_CALUDE_jellybean_distribution_l2350_235052

theorem jellybean_distribution (total_jellybeans : ℕ) (total_recipients : ℕ) 
  (h1 : total_jellybeans = 70) (h2 : total_recipients = 5) :
  total_jellybeans / total_recipients = 14 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_distribution_l2350_235052


namespace NUMINAMATH_CALUDE_max_product_constrained_l2350_235006

theorem max_product_constrained (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_constraint : 3 * x + 8 * y = 72) : 
  x * y ≤ 54 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 * x₀ + 8 * y₀ = 72 ∧ x₀ * y₀ = 54 :=
sorry

end NUMINAMATH_CALUDE_max_product_constrained_l2350_235006


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l2350_235060

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 675)
  (h2 : selling_price = 1080) : 
  (selling_price - cost_price) / cost_price * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l2350_235060


namespace NUMINAMATH_CALUDE_inequality_proof_l2350_235099

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d)
  (h5 : a^5 + b^5 ≤ 1) (h6 : c^5 + d^5 ≤ 1) : 
  a^2 * c^3 + b^2 * d^3 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2350_235099


namespace NUMINAMATH_CALUDE_similar_cube_volume_l2350_235036

theorem similar_cube_volume (original_volume : ℝ) (scale_factor : ℝ) : 
  original_volume = 343 → scale_factor = 2 → 
  (scale_factor ^ 3) * original_volume = 2744 := by
  sorry

end NUMINAMATH_CALUDE_similar_cube_volume_l2350_235036


namespace NUMINAMATH_CALUDE_ellipse_focus_coordinates_l2350_235011

/-- Given an ellipse with specified major and minor axis endpoints, 
    prove that the focus with the smaller y-coordinate has coordinates (5 - √5, 2) -/
theorem ellipse_focus_coordinates 
  (major_endpoint1 : ℝ × ℝ)
  (major_endpoint2 : ℝ × ℝ)
  (minor_endpoint1 : ℝ × ℝ)
  (minor_endpoint2 : ℝ × ℝ)
  (h1 : major_endpoint1 = (2, 2))
  (h2 : major_endpoint2 = (8, 2))
  (h3 : minor_endpoint1 = (5, 4))
  (h4 : minor_endpoint2 = (5, 0)) :
  ∃ (focus : ℝ × ℝ), focus = (5 - Real.sqrt 5, 2) ∧ 
  (∀ (other_focus : ℝ × ℝ), other_focus.2 ≤ focus.2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_coordinates_l2350_235011


namespace NUMINAMATH_CALUDE_perimeter_ABF₂_is_24_l2350_235080

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 25 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points A and B on the ellipse that intersect with F₁
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Assume A and B are on the ellipse
axiom A_on_ellipse : ellipse A.1 A.2
axiom B_on_ellipse : ellipse B.1 B.2

-- Assume F₁ intersects the ellipse at A and B
axiom F₁_intersect_A : F₁ = A
axiom F₁_intersect_B : F₁ = B

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define the perimeter of triangle ABF₂
def perimeter_ABF₂ : ℝ := distance A F₂ + distance B F₂ + distance A B

-- Theorem: The perimeter of triangle ABF₂ is 24
theorem perimeter_ABF₂_is_24 : perimeter_ABF₂ = 24 := by sorry

end NUMINAMATH_CALUDE_perimeter_ABF₂_is_24_l2350_235080


namespace NUMINAMATH_CALUDE_largest_difference_l2350_235047

def A : ℕ := 3 * 1003^1004
def B : ℕ := 1003^1004
def C : ℕ := 1002 * 1003^1003
def D : ℕ := 3 * 1003^1003
def E : ℕ := 1003^1003
def F : ℕ := 1003^1002

theorem largest_difference :
  A - B > max (B - C) (max (C - D) (max (D - E) (E - F))) :=
by sorry

end NUMINAMATH_CALUDE_largest_difference_l2350_235047
