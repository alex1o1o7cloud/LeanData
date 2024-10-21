import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_interest_rate_is_five_percent_l842_84281

/-- 
Given a principal sum, time period, and the difference between compound and simple interest,
calculate the interest rate.
-/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (interest_difference : ℝ) : ℝ :=
  Real.sqrt (interest_difference / principal)

/-- 
Theorem stating that for the given conditions, the calculated interest rate is 0.05 (5%).
-/
theorem interest_rate_is_five_percent :
  let principal : ℝ := 24000
  let time : ℝ := 2
  let interest_difference : ℝ := 60
  calculate_interest_rate principal time interest_difference = 0.05 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_interest_rate 24000 2 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_interest_rate_is_five_percent_l842_84281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_form_sqrt_a_squared_plus_one_l842_84278

/-- A square root expression is in its simplest form if it cannot be further simplified. -/
noncomputable def IsSimplestForm (x : ℝ → ℝ) : Prop :=
  ∀ y : ℝ → ℝ, (∀ t, x t = y t) → x = y

/-- Given options for square root expressions -/
noncomputable def Options : List (ℝ → ℝ) :=
  [λ _ ↦ Real.sqrt 12, λ a ↦ Real.sqrt (a / 2), λ a ↦ Real.sqrt (a^2 + 1), λ a ↦ Real.sqrt (4*a + 4)]

theorem simplest_form_sqrt_a_squared_plus_one :
  ∃ x ∈ Options, IsSimplestForm x ∧ ∀ y ∈ Options, IsSimplestForm y → x = y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_form_sqrt_a_squared_plus_one_l842_84278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_minimum_l842_84246

/-- The total cost function for the protective cover -/
noncomputable def total_cost (x : ℝ) : ℝ := 500 * x + 8000 / x - 250

/-- The minimum value of the total cost -/
def min_cost : ℝ := 3750

/-- Theorem stating that the minimum value of the total cost is 3750 yuan -/
theorem total_cost_minimum (x : ℝ) (h : x > 0.5) : 
  total_cost x ≥ min_cost := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_minimum_l842_84246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_I_part_II_l842_84285

noncomputable section

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := Real.log x + k / x

-- Define e as a constant (base of natural logarithm)
noncomputable def e : ℝ := Real.exp 1

-- Part I
theorem part_I (k : ℝ) :
  (∀ x : ℝ, x > 0 → (deriv (f k)) e = 0) →
  (∀ x : ℝ, 0 < x ∧ x < e → deriv (f k) x < 0) ∧
  (∀ x : ℝ, x > e → deriv (f k) x > 0) ∧
  (f k e = 2) :=
sorry

-- Part II
theorem part_II :
  (∀ k : ℝ, (∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ > 0 → f k x₁ - f k x₂ < x₁ - x₂) ↔ k ≥ 1/4) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_I_part_II_l842_84285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_failed_l842_84277

def total_students : ℕ := 32
def a_percentage : ℚ := 1/4
def bc_fraction : ℚ := 1/4

theorem students_failed : 
  total_students - 
  (a_percentage * total_students).floor.toNat - 
  (bc_fraction * (total_students - (a_percentage * total_students).floor.toNat)).floor.toNat = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_failed_l842_84277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_trinomial_divisibility_l842_84268

/-- A quadratic trinomial with integer coefficients -/
def QuadraticTrinomial := ℤ → ℤ

/-- Proper divisor predicate -/
def IsProperDivisor (d n : ℤ) : Prop := 1 < d ∧ d < n ∧ n % d = 0

/-- Increasing sequence predicate -/
def IsIncreasing (s : ℕ → ℤ) : Prop := ∀ n : ℕ, s n < s (n + 1)

/-- Product of two linear polynomials with integer coefficients -/
def IsProductOfLinear (P : QuadraticTrinomial) : Prop :=
  ∃ a b c d : ℤ, ∀ n : ℤ, P n = (a * n + b) * (c * n + d)

/-- All values divisible by the same integer m > 1 -/
def AllDivisibleBy (P : QuadraticTrinomial) : Prop :=
  ∃ m : ℤ, m > 1 ∧ ∀ n : ℕ, (P n) % m = 0

theorem quadratic_trinomial_divisibility (P : QuadraticTrinomial)
    (h1 : ∀ n : ℕ, ∃ d : ℤ, IsProperDivisor d (P n))
    (h2 : IsIncreasing (fun n ↦ Classical.choose (h1 n))) :
    IsProductOfLinear P ∨ AllDivisibleBy P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_trinomial_divisibility_l842_84268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_scores_l842_84239

def scores : List ℕ := [64, 68, 74, 77, 85, 90]

def is_integer_average (sublist : List ℕ) : Prop :=
  ∃ n : ℕ, n * sublist.length = sublist.sum

def last_score_is_68 : Prop :=
  ∀ perm : List ℕ, perm.length = 6 → perm.toFinset = scores.toFinset →
    (∀ k : ℕ, k ≤ 6 → is_integer_average (perm.take k)) →
    perm.getLast? = some 68

theorem quiz_scores :
  last_score_is_68 := by sorry

#check quiz_scores

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_scores_l842_84239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_theorem_l842_84215

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_function_theorem (f : ℝ → ℝ) 
  (h1 : is_odd f)
  (h2 : is_even (fun x ↦ f (x + 1)))
  (h3 : ∀ x, x ∈ Set.Icc 0 1 → f x = x * (3 - 2 * x)) :
  f (31 / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_theorem_l842_84215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_reaches_b_time_l842_84214

/-- Represents a robot moving on a circular track -/
structure Robot where
  name : String
  clockwise : Bool
  startPoint : String

/-- Represents the circular track and robot movements -/
structure CircularTrack where
  length : ℝ
  robots : List Robot
  jToBTime : ℝ
  jMeetsBYTime : ℝ

/-- Helper function to calculate the time Y reaches B -/
def time_y_reaches_b (track : CircularTrack) : ℝ :=
  56  -- Simplified implementation for the proof

/-- The main theorem statement -/
theorem y_reaches_b_time (track : CircularTrack) 
  (h1 : track.robots.length = 3)
  (h2 : ∃ j y b, j ∈ track.robots ∧ y ∈ track.robots ∧ b ∈ track.robots ∧ 
                 j.name = "J" ∧ y.name = "Y" ∧ b.name = "B")
  (h3 : ∃ j y, j ∈ track.robots ∧ y ∈ track.robots ∧ 
               j.startPoint = "A" ∧ y.startPoint = "A")
  (h4 : ∃ b, b ∈ track.robots ∧ b.startPoint = "B")
  (h5 : ∃ y, y ∈ track.robots ∧ y.clockwise = true)
  (h6 : ∃ j b, j ∈ track.robots ∧ b ∈ track.robots ∧ 
               j.clockwise = false ∧ b.clockwise = false)
  (h7 : track.jToBTime = 12)
  (h8 : track.jMeetsBYTime = track.jToBTime + 9)
  : ∃ t : ℝ, t = 56 ∧ t = time_y_reaches_b track :=
by
  use 56
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_reaches_b_time_l842_84214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_to_hundredth_l842_84206

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The sum of 74.6893 and 23.152 rounded to the nearest hundredth is 97.84 -/
theorem sum_and_round_to_hundredth :
  roundToHundredth (74.6893 + 23.152) = 97.84 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_to_hundredth_l842_84206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_whole_number_to_shaded_area_l842_84296

noncomputable section

-- Define the rectangle dimensions
def rectangle_width : ℝ := 2
def rectangle_height : ℝ := 3

-- Define the circle diameter
def circle_diameter : ℝ := 1

-- Define the shaded area
noncomputable def shaded_area : ℝ := rectangle_width * rectangle_height - Real.pi * (circle_diameter / 2)^2

-- Theorem statement
theorem closest_whole_number_to_shaded_area :
  ∃ (n : ℕ), n = 5 ∧ ∀ (m : ℕ), |shaded_area - ↑n| ≤ |shaded_area - ↑m| := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_whole_number_to_shaded_area_l842_84296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_l842_84212

def M : Set Int := {-2, -1, 0, 1}

def N : Set Int := {x : Int | x^2 - x - 6 < 0}

theorem M_intersect_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_intersect_N_l842_84212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l842_84232

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

/-- Theorem: The area of the triangle with vertices (7.5, 12.5), (13.5, 2.6), and (9.4, 18.8) is approximately 28.305 -/
theorem triangle_area_example : 
  abs (triangleArea 7.5 12.5 13.5 2.6 9.4 18.8 - 28.305) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l842_84232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_minimum_f_at_one_inflection_point_l842_84292

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + 27 * x + 2000

-- State the theorems
theorem no_minimum : ¬ ∃ (a : ℝ), ∀ (x : ℝ), f x ≥ f a := by sorry

theorem f_at_one : f 1 = 2021 := by sorry

theorem inflection_point :
  ∃ (ε : ℝ), ε > 0 ∧
  (∀ (x : ℝ), x > 1 - ε ∧ x < 1 → (deriv (deriv f)) x < 0) ∧
  (∀ (x : ℝ), x > 1 ∧ x < 1 + ε → (deriv (deriv f)) x > 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_minimum_f_at_one_inflection_point_l842_84292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baseball_hours_l842_84260

/-- Represents the number of hours played or present at the ballpark -/
def TotalHours (nathan : ℕ) (tobias : ℕ) (leo : ℕ) (maddison : ℕ) : ℕ :=
  nathan + tobias + leo + maddison

/-- Calculates Nathan's hours given days played and hours per day -/
def NathanHours (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  days * hours_per_day

/-- Calculates Tobias's hours given days played and hours per day -/
def TobiasHours (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  days * hours_per_day

/-- Calculates Leo's hours given days played and hours per day -/
def LeoHours (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  days * hours_per_day

/-- Calculates Maddison's hours given weekdays present and hours per day -/
def MaddisonHours (weekdays : ℕ) (hours_per_day : ℕ) : ℕ :=
  weekdays * hours_per_day

theorem baseball_hours :
  let days_in_week : ℕ := 7
  let nathan_days : ℕ := 2 * days_in_week
  let nathan_hours_per_day : ℕ := 3
  let tobias_days : ℕ := days_in_week
  let tobias_hours_per_day : ℕ := 5
  let leo_days : ℕ := 10
  let leo_hours_per_day : ℕ := 5 / 2
  let maddison_weekdays : ℕ := 3 * 5
  let maddison_hours_per_day : ℕ := 6
  TotalHours
    (NathanHours nathan_days nathan_hours_per_day)
    (TobiasHours tobias_days tobias_hours_per_day)
    (LeoHours leo_days leo_hours_per_day)
    (MaddisonHours maddison_weekdays maddison_hours_per_day) = 192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baseball_hours_l842_84260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_d_inequality_l842_84231

theorem smallest_d_inequality (d : ℝ) : d = 1/2 ↔ 
  (∀ (x y z : ℝ), x ≥ 0 → y ≥ 0 → z > 0 → 
    Real.sqrt (x * y * z) + d * abs (x - y) * z ≥ (x * z + y * z) / 2) ∧
  (∀ (d' : ℝ), d' > 0 → d' < d → 
    ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z > 0 ∧
      Real.sqrt (x * y * z) + d' * abs (x - y) * z < (x * z + y * z) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_d_inequality_l842_84231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_in_interval_l842_84279

noncomputable def f (x : ℝ) := 1 / x - 2 * x

theorem min_value_of_f_in_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-2) (-1/2) ∧
  f x = -1 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-1/2) (-2) → f y ≥ f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_in_interval_l842_84279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_pi_x_over_5_l842_84209

theorem coefficient_of_pi_x_over_5 : 
  let expression : ℝ → ℝ := λ x => (π * x) / 5
  (π / 5 : ℝ) = π / 5 := by
  -- The coefficient is already π / 5, so we just need to show it's equal to itself
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_pi_x_over_5_l842_84209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cherry_pie_approx_25_degrees_l842_84228

/-- Represents the pie chart for Elaina's class preferences -/
structure PieChart where
  total_students : ℕ
  chocolate_pref : ℕ
  apple_pref : ℕ
  blueberry_pref : ℕ

/-- Calculates the degrees for cherry pie in Elaina's pie chart -/
noncomputable def cherry_pie_degrees (chart : PieChart) : ℝ :=
  let remaining_students := chart.total_students - (chart.chocolate_pref + chart.apple_pref + chart.blueberry_pref)
  let cherry_students := (remaining_students : ℝ) / 3
  (cherry_students / chart.total_students) * 360

/-- Theorem stating that the degrees for cherry pie is approximately 25° -/
theorem cherry_pie_approx_25_degrees (chart : PieChart)
  (h1 : chart.total_students = 48)
  (h2 : chart.chocolate_pref = 18)
  (h3 : chart.apple_pref = 12)
  (h4 : chart.blueberry_pref = 8) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |cherry_pie_degrees chart - 25| < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cherry_pie_approx_25_degrees_l842_84228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chromium_percentage_in_combined_alloy_l842_84218

/-- Represents an alloy with its chromium percentage and weight -/
structure Alloy where
  chromium_percentage : ℝ
  weight : ℝ

/-- Calculates the chromium content of an alloy -/
noncomputable def chromium_content (a : Alloy) : ℝ :=
  a.chromium_percentage * a.weight / 100

/-- Calculates the percentage of chromium in a new alloy formed by combining two alloys -/
noncomputable def new_alloy_chromium_percentage (a1 a2 : Alloy) : ℝ :=
  let total_chromium := chromium_content a1 + chromium_content a2
  let total_weight := a1.weight + a2.weight
  (total_chromium / total_weight) * 100

theorem chromium_percentage_in_combined_alloy :
  let alloy1 : Alloy := { chromium_percentage := 10, weight := 15 }
  let alloy2 : Alloy := { chromium_percentage := 8, weight := 35 }
  new_alloy_chromium_percentage alloy1 alloy2 = 8.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chromium_percentage_in_combined_alloy_l842_84218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coat_distribution_l842_84240

theorem coat_distribution (total_coats : ℝ) (num_boxes : ℝ) 
  (h1 : total_coats = 385.5) 
  (h2 : num_boxes = 7.5) : 
  Int.floor (total_coats - num_boxes * ↑(Int.floor (total_coats / num_boxes))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coat_distribution_l842_84240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_elements_l842_84207

def f (x : ℝ) : ℝ := 2 * x^2 + 2 * x - 4

def g (x : ℝ) : ℝ := x^2 - x + 2

def A : Set ℝ := {x | ∃ (n : ℕ), n > 0 ∧ f x = n * g x}

theorem set_A_elements : A = {2, (-3 + Real.sqrt 33) / 2, (-3 - Real.sqrt 33) / 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_elements_l842_84207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_in_small_disk_l842_84274

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a square with side length 1
def UnitSquare : Set Point := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define a disk with radius 1/7
def Disk (center : Point) : Set Point := {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ (1/7)^2}

-- State the theorem
theorem three_points_in_small_disk (points : Finset Point) :
  points.card = 51 → (↑points : Set Point) ⊆ UnitSquare →
  ∃ (center : Point), ((Disk center ∩ ↑points) : Set Point).ncard ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_in_small_disk_l842_84274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_range_l842_84271

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 4*x + 3 else -x^2 - 2*x + 3

-- State the theorem
theorem function_inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 1), f (x + a) > f (2 * a - x)) →
  a < -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_range_l842_84271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l842_84213

/-- Represents an equilateral triangle with a given side length -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Calculate the perimeter of an equilateral triangle -/
def perimeter (t : EquilateralTriangle) : ℝ :=
  3 * t.side_length

/-- Calculate the area of an equilateral triangle -/
noncomputable def area (t : EquilateralTriangle) : ℝ :=
  (Real.sqrt 3 / 4) * t.side_length^2

theorem equilateral_triangle_properties :
  let t : EquilateralTriangle := { side_length := 10, side_length_pos := by norm_num }
  perimeter t = 30 ∧ area t = 25 * Real.sqrt 3 := by
  sorry

#eval perimeter { side_length := 10, side_length_pos := by norm_num }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_properties_l842_84213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_algorithms_exist_not_only_one_algorithm_per_problem_l842_84291

/-- Represents a step in an algorithm --/
structure Step where
  description : String
  is_clearly_defined : Bool
  is_executable : Bool

/-- Represents an algorithm --/
structure Algorithm where
  steps : List Step
  is_finite : Bool
  has_definitive_outcome : Bool
  produces : ∀ (input : Type) (output : Type), input → output → Prop

/-- Represents a problem that can be solved by an algorithm --/
structure Problem where
  input : Type
  output : Type
  is_valid_solution : input → output → Prop

/-- States that multiple algorithms can exist for a single problem --/
theorem multiple_algorithms_exist (p : Problem) :
  ∃ (a1 a2 : Algorithm), a1 ≠ a2 ∧
    (∀ (i : p.input), ∃ (o : p.output), 
      p.is_valid_solution i o ∧ 
      a1.produces p.input p.output i o ∧ 
      a2.produces p.input p.output i o) :=
by
  sorry

/-- Proves that the statement "There can only be one algorithm for a problem" is false --/
theorem not_only_one_algorithm_per_problem :
  ¬ (∀ (p : Problem), ∃! (a : Algorithm), ∀ (i : p.input) (o : p.output),
    p.is_valid_solution i o ↔ a.produces p.input p.output i o) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_algorithms_exist_not_only_one_algorithm_per_problem_l842_84291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_equality_l842_84264

theorem cosine_sine_equality (α : ℝ) 
  (h1 : Real.tan α = Real.sqrt 3) 
  (h2 : π < α) 
  (h3 : α < 3 * π / 2) : 
  Real.cos (2 * α) - Real.sin (π / 2 + α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_equality_l842_84264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_minus_alpha_l842_84205

theorem tan_pi_4_minus_alpha (α : ℝ) :
  Real.tan (π / 4 - α) = 1 / 2 →
  (Real.sin (2 * α) + Real.sin α ^ 2) / (1 + Real.cos (2 * α)) = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_minus_alpha_l842_84205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_volume_l842_84253

/-- The volume of water filled in the first hour -/
def b : ℝ := sorry

/-- The constant factor by which the water volume decreases each hour -/
def q : ℝ := sorry

/-- The total volume of the tank -/
def V : ℝ := sorry

/-- Conditions of the problem -/
axiom water_decrease : 0 < q ∧ q < 1
axiom first_four_twice_last_four : b + b*q + b*q^2 + b*q^3 = 2*(b*q + b*q^2 + b*q^3 + b*q^4)
axiom first_two_hours : b + b*q = 48

/-- The theorem to prove -/
theorem tank_volume : V = 62 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_volume_l842_84253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l842_84234

/-- The circle (x-1)² + (y-2)² = 4 is always tangent to the line x*cos(θ) + y*sin(θ) - cos(θ) - 2*sin(θ) - 2 = 0 for θ ∈ [0, 2π] -/
theorem circle_tangent_to_line :
  ∀ θ : ℝ, θ ∈ Set.Icc 0 (2 * Real.pi) →
  ∃! d : ℝ, d = |Real.cos θ + 2 * Real.sin θ + 1| ∧ d = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l842_84234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_major_axis_increase_l842_84261

-- Define the Ramanujan's approximation for ellipse perimeter
noncomputable def ellipse_perimeter (a b : ℝ) : ℝ :=
  Real.pi * (3 * (a + b) - Real.sqrt ((3 * a + b) * (a + 3 * b)))

-- Define the problem parameters
def initial_perimeter : ℝ := 30
def final_perimeter : ℝ := 40

-- Theorem statement
theorem semi_major_axis_increase 
  (b : ℝ) -- Constant semi-minor axis
  (a₁ a₂ : ℝ) -- Initial and final semi-major axes
  (h₁ : ellipse_perimeter a₁ b = initial_perimeter)
  (h₂ : ellipse_perimeter a₂ b = final_perimeter)
  : (a₂ - a₁ = 2) ∨ (a₂ - a₁ = 3) ∨ (a₂ - a₁ = 4) ∨ (a₂ - a₁ = 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_major_axis_increase_l842_84261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_eight_hundred_thousand_is_last_integer_l842_84220

def sequenceterm (n : ℕ) : ℚ := (800000 : ℚ) / 3^n

theorem last_integer_in_sequence : 
  ∀ n : ℕ, n > 0 → ¬(sequenceterm n).isInt :=
by
  sorry

theorem eight_hundred_thousand_is_last_integer :
  (sequenceterm 0).isInt ∧ ∀ n : ℕ, n > 0 → ¬(sequenceterm n).isInt :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_eight_hundred_thousand_is_last_integer_l842_84220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_2_range_of_a_for_two_zeros_l842_84269

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - a^2 * Real.log x

-- Part 1: Monotonicity when a = 2
theorem monotonicity_when_a_2 :
  ∀ x₁ x₂ ε, 0 < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ ε > 0 →
  (f 2 x₁ > f 2 (x₁ + ε) ∧ f 2 x₂ < f 2 (x₂ + ε)) :=
by
  sorry

-- Part 2: Range of a for two distinct zeros
theorem range_of_a_for_two_zeros :
  ∀ a, (∃ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔
  (a > 2 * Real.exp (3/4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_2_range_of_a_for_two_zeros_l842_84269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_theorem_l842_84273

theorem cosine_theorem (a b c A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  (a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)) ∧
  (b^2 = a^2 + c^2 - 2*a*c*(Real.cos B)) ∧
  (c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_theorem_l842_84273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l842_84283

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * (x + 4)

-- Define the domain of f
def domain : Set ℝ := {x | x > 0}

-- Define the inequality condition
def inequality_condition (a : ℝ) (x : ℝ) : Prop :=
  (x + 1) * Real.exp x + a * (x^2 + 4*x) ≥ 0

theorem range_of_a (a : ℝ) :
  (∀ x ∈ domain, inequality_condition a x) →
  a ≥ -(1/2) * Real.exp (Real.sqrt 3 - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l842_84283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_at_20_l842_84222

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_odd : a 1 + a 3 + a 5 = 105
  sum_even : a 2 + a 4 + a 6 = 99

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1)) / 2

/-- The theorem stating that the sum reaches its maximum when n = 20 -/
theorem sum_max_at_20 (seq : ArithmeticSequence) :
  ∀ n : ℕ, sum_n seq n ≤ sum_n seq 20 := by
  sorry

#check sum_max_at_20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_at_20_l842_84222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_latte_cost_is_four_l842_84290

/-- The cost of a latte that Martha buys -/
def latte_cost : ℝ := sorry

/-- The number of days per week Martha buys a latte -/
def latte_days_per_week : ℕ := 5

/-- The cost of an iced coffee -/
def iced_coffee_cost : ℝ := 2

/-- The number of days per week Martha buys an iced coffee -/
def iced_coffee_days_per_week : ℕ := 3

/-- The number of weeks in a year -/
def weeks_in_year : ℕ := 52

/-- The percentage of spending reduction Martha aims for -/
def spending_reduction_percentage : ℝ := 0.25

/-- The amount Martha will save in a year -/
def yearly_savings : ℝ := 338

theorem latte_cost_is_four :
  latte_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_latte_cost_is_four_l842_84290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_and_area_l842_84284

theorem inscribed_sphere_volume_and_area (cube_edge : ℝ) 
  (h : cube_edge = 10) : 
  (4 / 3) * Real.pi * (cube_edge / 2) ^ 3 = (500 / 3) * Real.pi ∧ 
  4 * Real.pi * (cube_edge / 2) ^ 2 = 100 * Real.pi :=
by
  -- Convert the cube edge to the sphere radius
  have sphere_radius : ℝ := cube_edge / 2
  
  -- Calculate the volume
  have sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  
  -- Calculate the surface area
  have sphere_area : ℝ := 4 * Real.pi * sphere_radius ^ 2
  
  -- Prove the equalities
  sorry  -- We'll use sorry to skip the actual proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_and_area_l842_84284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_of_f_six_times_l842_84294

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := -1 / x

-- State the theorem
theorem composition_of_f_six_times :
  f (f (f (f (f (f 8))))) = 8 := by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_of_f_six_times_l842_84294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_problem_l842_84247

/-- The time (in minutes) it takes for two people walking in opposite directions 
    around a circular path to meet again for the first time. -/
noncomputable def meeting_time (circumference : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  circumference / (speed1 + speed2)

/-- Theorem stating that two people walking in opposite directions on a circular path
    with a circumference of 3000 meters, at speeds of 100 m/min and 150 m/min respectively,
    will meet again for the first time after 12 minutes. -/
theorem meeting_time_problem :
  let circumference : ℝ := 3000
  let speed_cheolsu : ℝ := 100
  let speed_younghee : ℝ := 150
  meeting_time circumference speed_cheolsu speed_younghee = 12 := by
  -- Unfold the definition of meeting_time
  unfold meeting_time
  -- Simplify the expression
  simp
  -- Check that the equation holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_problem_l842_84247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l842_84289

/-- The volume of a regular tetrahedron with circumscribed circle radius R -/
noncomputable def tetrahedronVolume (R : ℝ) : ℝ := (R^3 * Real.sqrt 6) / 4

/-- Theorem: The volume of a regular tetrahedron with a circumscribed circle 
    of radius R around one of its faces is equal to (R³√6)/4 -/
theorem regular_tetrahedron_volume (R : ℝ) (h : R > 0) :
  tetrahedronVolume R = (R^3 * Real.sqrt 6) / 4 :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_volume_l842_84289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_henry_paint_price_l842_84245

/-- The amount Henry is paid to paint a bike -/
def paint_price : ℚ := sorry

/-- The amount Henry is paid to sell a bike -/
def sell_price : ℚ := sorry

/-- The total amount Henry is paid for painting and selling 8 bikes -/
def total_price : ℚ := sorry

/-- Henry is paid $8 more to sell a bike than to paint it -/
axiom sell_paint_difference : sell_price = paint_price + 8

/-- Henry gets paid $144 to sell and paint 8 bikes -/
axiom total_for_8_bikes : total_price = 144

/-- The total price is the sum of painting and selling 8 bikes -/
axiom price_composition : total_price = 8 * (paint_price + sell_price)

theorem henry_paint_price : paint_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_henry_paint_price_l842_84245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ajay_walking_time_l842_84248

noncomputable section

-- Define Ajay's walking speeds
def flat_speed : ℝ := 3
def uphill_speed : ℝ := flat_speed * (1 - 0.2)
def downhill_speed : ℝ := flat_speed * (1 + 0.1)

-- Define distances for each terrain type
def uphill_distance : ℝ := 15
def flat_distance : ℝ := 25
def downhill_distance : ℝ := 20

-- Define the total time function
def total_time : ℝ :=
  uphill_distance / uphill_speed +
  flat_distance / flat_speed +
  downhill_distance / downhill_speed

-- Theorem statement
theorem ajay_walking_time : ∃ ε > 0, |total_time - 20.64| < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ajay_walking_time_l842_84248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pedro_difference_theorem_l842_84263

/-- Represents a player in the game -/
structure Player where
  name : String
  initialSquares : ℕ
  multiplier : ℚ
deriving Inhabited

/-- Calculates the number of squares a player has after applying their multiplier -/
def squaresAfterMultiplier (p : Player) : ℚ :=
  p.initialSquares * p.multiplier

/-- The set of players in the game -/
def players : List Player :=
  [⟨"Pedro", 200, 4⟩, ⟨"Linden", 75, 3⟩, ⟨"Jesus", 60, 2⟩, ⟨"Martha", 120, (3/2)⟩, ⟨"Nancy", 90, (7/2)⟩]

/-- Pedro's player object -/
def pedro : Player := players.head!

/-- The list of players excluding Pedro -/
def otherPlayers : List Player := players.tail

/-- Theorem: The difference between Pedro's squares after multiplication and the average
    of the other players' squares after multiplication is 590 -/
theorem pedro_difference_theorem :
  squaresAfterMultiplier pedro - (otherPlayers.map squaresAfterMultiplier).sum / otherPlayers.length = 590 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pedro_difference_theorem_l842_84263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_implication_l842_84280

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem negation_of_implication :
  (¬ (∀ f : ℝ → ℝ, is_odd f → is_odd (λ x ↦ f (-x)))) ↔
  (∃ f : ℝ → ℝ, ¬ is_odd f ∧ ¬ is_odd (λ x ↦ f (-x))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_implication_l842_84280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l842_84251

noncomputable def g (x : ℝ) : ℝ := 1 / (x ^ 2) + 5

theorem range_of_g :
  Set.range g = Set.Ioi 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l842_84251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_october_order_theorem_l842_84257

/-- Represents the monthly order quantities for a type of hammer -/
structure HammerOrders :=
  (june july august september : ℕ)

/-- Calculates the next month's order based on the pattern -/
def nextMonthOrder (orders : HammerOrders) : ℕ := sorry

/-- Represents the order quantities for both types of hammers -/
structure StoreOrders :=
  (claw : HammerOrders)
  (ballPeen : HammerOrders)

/-- Calculates the total number of hammers ordered for October -/
noncomputable def octoberOrderTotal (orders : StoreOrders) : ℕ :=
  let clawOct := nextMonthOrder orders.claw
  let ballPeenOct := nextMonthOrder orders.ballPeen
  let totalBeforeIncrease := clawOct + ballPeenOct
  ⌈(totalBeforeIncrease : ℝ) * 1.05⌉.toNat

theorem october_order_theorem (orders : StoreOrders) 
  (h1 : orders.claw = ⟨3, 4, 6, 9⟩) 
  (h2 : orders.ballPeen = ⟨2, 3, 7, 11⟩) : 
  octoberOrderTotal orders = 30 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_october_order_theorem_l842_84257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l842_84258

theorem problem_solution (m n : ℝ) (hm : |m| = 1) (hn : |n| = 4) :
  (∃ x ∈ ({-3, 3} : Set ℝ), m * n < 0 → m + n = x) ∧ 
  (∀ x : ℝ, m - n ≤ 5) ∧ (∃ y z : ℝ, y - z = 5 ∧ |y| = 1 ∧ |z| = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l842_84258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_lambda_mu_l842_84259

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 1)

-- Define the intersection points
noncomputable def intersection_points (k : ℝ) : 
  Σ' (A B : ℝ × ℝ), parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
    line_through_focus k A.1 A.2 ∧ line_through_focus k B.1 B.2 ∧ A ≠ B :=
  sorry

-- Define point N on y-axis
def N (k : ℝ) : ℝ × ℝ := (0, -k)

-- Define λ and μ
noncomputable def lambda_mu (k : ℝ) : ℝ × ℝ := by
  let ⟨A, B, _⟩ := intersection_points k
  let NA := (A.1 - (N k).1, A.2 - (N k).2)
  let NB := (B.1 - (N k).1, B.2 - (N k).2)
  let AF := (focus.1 - A.1, focus.2 - A.2)
  let BF := (focus.1 - B.1, focus.2 - B.2)
  exact (NA.1 / AF.1, NB.1 / BF.1)

-- Theorem statement
theorem constant_sum_lambda_mu :
  ∀ k : ℝ, (lambda_mu k).1 + (lambda_mu k).2 = -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_lambda_mu_l842_84259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volvo_acquisition_theorem_l842_84241

def sales_volume (n : ℕ) : ℕ := 20000 + 10000 * n

def profit_per_unit (n : ℕ) : ℝ := 20000 * (0.9 ^ n)

def yearly_profit (n : ℕ) : ℝ := (sales_volume n : ℝ) * (profit_per_unit n)

def total_profit (years : ℕ) : ℝ := (Finset.range years).sum (λ i ↦ yearly_profit i)

theorem volvo_acquisition_theorem :
  (∀ n : ℕ, yearly_profit n = (10000 + 10000 * n : ℝ) * (2 * 0.9^(n - 1))) ∧
  (total_profit 5 < 3.8 * 10^9) := by
  sorry

#eval total_profit 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volvo_acquisition_theorem_l842_84241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_focus_dist_product_l842_84293

/-- The ellipse with equation x²/25 + y²/9 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 25) + (p.2^2 / 9) = 1}

/-- The foci of the ellipse -/
def Foci : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((4, 0), (-4, 0))

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Product of distances from a point to the foci -/
noncomputable def focusDistProduct (p : ℝ × ℝ) : ℝ :=
  distance p Foci.1 * distance p Foci.2

theorem max_focus_dist_product :
  ∃ (max : ℝ), max = 25 ∧ ∀ p ∈ Ellipse, focusDistProduct p ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_focus_dist_product_l842_84293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solar_panel_optimization_l842_84216

/-- The annual electricity cost function after solar panel installation -/
noncomputable def C (k : ℝ) (x : ℝ) : ℝ := k / (20 * x + 100)

/-- The total cost function over 15 years -/
noncomputable def F (x : ℝ) : ℝ := 15 * C 2400 x + 0.5 * x

theorem solar_panel_optimization :
  ∃ (x : ℝ), x ≥ 0 ∧ F x ≤ F y ∧ F x = 57.5 ∧ x = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solar_panel_optimization_l842_84216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_parabola_l842_84299

/-- The length of a chord AB intersecting the parabola y^2 = 8x at an angle of 135° from the focus --/
theorem chord_length_parabola (A B : ℝ × ℝ) : 
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 8*x}
  let focus := (2, 0)
  let chord_angle := 135 * Real.pi / 180
  (A ∈ parabola) → 
  (B ∈ parabola) → 
  (A ≠ B) →
  (Real.arctan ((A.2 - focus.2) / (A.1 - focus.1)) = chord_angle) →
  (∃ t : ℝ, A = (focus.1 + t * Real.cos chord_angle, focus.2 + t * Real.sin chord_angle)) →
  (∃ s : ℝ, B = (focus.1 + s * Real.cos chord_angle, focus.2 + s * Real.sin chord_angle)) →
  ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2 : ℝ) = 16 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_parabola_l842_84299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l842_84235

noncomputable def f (x : ℝ) : ℝ := (4 * x - 2) / Real.sqrt (x - 7)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 7} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l842_84235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_2_l842_84227

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2 * x - 1 else 1 / x

theorem f_composition_at_2 : f (f 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_2_l842_84227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_l842_84267

-- Define the point in rectangular coordinates
noncomputable def point : ℝ × ℝ := (8, -8)

-- Define the polar coordinates
noncomputable def polar_coords : ℝ × ℝ := (8 * Real.sqrt 2, 7 * Real.pi / 4)

-- Theorem statement
theorem rectangular_to_polar :
  let (x, y) := point
  let (r, θ) := polar_coords
  (r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) ∧
  r * (Real.cos θ) = x ∧
  r * (Real.sin θ) = y :=
by
  -- Proof goes here
  sorry

#check rectangular_to_polar

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_l842_84267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_odd_counts_l842_84250

/-- Represents a building with floors and entrances -/
structure Building where
  floors : Nat
  entrances : Nat

/-- Represents the number of residents on each floor and in each entrance -/
structure ResidentCount (b : Building) where
  floor_counts : Fin b.floors → Nat
  entrance_counts : Fin b.entrances → Nat

/-- Theorem: In a 5-story building with 4 entrances, it's impossible for all resident counts to be odd -/
theorem not_all_odd_counts (b : Building) (rc : ResidentCount b)
  (h_floors : b.floors = 5)
  (h_entrances : b.entrances = 4)
  (h_total : (Finset.sum (Finset.univ : Finset (Fin b.floors)) rc.floor_counts) =
             (Finset.sum (Finset.univ : Finset (Fin b.entrances)) rc.entrance_counts)) :
  ¬(∀ (i : Fin b.floors), Odd (rc.floor_counts i) ∧
    ∀ (j : Fin b.entrances), Odd (rc.entrance_counts j)) := by
  sorry

#check not_all_odd_counts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_odd_counts_l842_84250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_rate_is_nine_percent_l842_84254

/-- Represents the investment scenario --/
structure Investment where
  total : ℚ
  first_amount : ℚ
  first_rate : ℚ
  second_amount : ℚ
  second_rate : ℚ
  desired_income : ℚ

/-- Calculates the required rate for the remaining investment --/
def required_rate (i : Investment) : ℚ :=
  let remaining_amount := i.total - i.first_amount - i.second_amount
  let remaining_income := i.desired_income - (i.first_amount * i.first_rate / 100) - (i.second_amount * i.second_rate / 100)
  (remaining_income / remaining_amount) * 100

/-- Theorem stating that the required rate for the given investment scenario is 9% --/
theorem investment_rate_is_nine_percent (i : Investment) 
  (h1 : i.total = 12000)
  (h2 : i.first_amount = 5000)
  (h3 : i.first_rate = 3)
  (h4 : i.second_amount = 4000)
  (h5 : i.second_rate = 9/2)
  (h6 : i.desired_income = 600) :
  required_rate i = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_rate_is_nine_percent_l842_84254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_maximum_l842_84237

theorem triangle_ratio_maximum (a b c : ℝ) (h : ℝ) (A : ℝ) :
  h = a / 2 →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π →
  a * h = b * c * Real.sin A →
  ∃ x : ℝ, ∀ y : ℝ, (c / b + b / c) ≤ x ∧ x ≤ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_maximum_l842_84237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_18_l842_84223

/-- Represents the distance swam upstream given the man's speed in still water,
    downstream distance, and time spent swimming in each direction. -/
noncomputable def distanceUpstream (speedStillWater : ℝ) (downstreamDistance : ℝ) (time : ℝ) : ℝ :=
  let streamSpeed := downstreamDistance / time - speedStillWater
  (speedStillWater - streamSpeed) * time

/-- Theorem stating that given the specific conditions of the problem,
    the distance swam upstream is 18 km. -/
theorem upstream_distance_is_18 :
  distanceUpstream 10.5 45 3 = 18 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval distanceUpstream 10.5 45 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_18_l842_84223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l842_84262

/-- Calculates simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Calculates compound interest -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate) ^ time - principal

theorem principal_calculation (P : ℝ) :
  let rate : ℝ := 0.05
  let time : ℝ := 2
  compoundInterest P rate time - simpleInterest P rate time = 15 →
  P = 6000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l842_84262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_concentric_circles_l842_84225

theorem shaded_area_of_concentric_circles (R : ℝ) : 
  R^2 = 100 → 
  (π * R^2 / 2) + (π * (R - 2)^2 / 2) - (π * (R - 1)^2 / 2) = 36 * π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_concentric_circles_l842_84225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_a_range_l842_84217

-- Define the points A and B
def A : ℝ × ℝ := (-2, 3)
def B (a : ℝ) : ℝ × ℝ := (0, a)

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x + 3)^2 + (y + 2)^2 = 1

-- Define the symmetrical line
def symmetrical_line (a x y : ℝ) : Prop := 
  (3 - a) * x - 2 * y + 2 * a = 0

-- Main theorem
theorem intersection_implies_a_range (a : ℝ) :
  (∃ x y : ℝ, symmetrical_line a x y ∧ my_circle x y) →
  a ∈ Set.Icc (1/3 : ℝ) (3/2 : ℝ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_a_range_l842_84217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_point_inside_circle_l842_84276

theorem ellipse_point_inside_circle 
  (a b c : ℝ) 
  (x₁ x₂ : ℝ) 
  (h_pos : a > b ∧ b > 0) 
  (h_ecc : c / a = 1 / 2) 
  (h_ellipse : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → 
    ∃ t, x = a * Real.cos t ∧ y = b * Real.sin t) 
  (h_roots : x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ - c = 0 ∧ a * x₂^2 + b * x₂ - c = 0) :
  x₁^2 + x₂^2 < 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_point_inside_circle_l842_84276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_of_n_squared_l842_84210

-- Define the number
def n : ℕ := 10^10 - 3

-- Define a function to count trailing zeros
def countTrailingZeros (m : ℕ) : ℕ :=
  if m = 0 then 0
  else if m % 10 = 0 then 1 + countTrailingZeros (m / 10)
  else 0

-- Theorem statement
theorem trailing_zeros_of_n_squared :
  countTrailingZeros (n^2) = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_of_n_squared_l842_84210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_l842_84204

theorem angle_of_inclination (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x => a * Real.sin x - b * Real.cos x
  ∃ θ, θ = 135 * (π / 180) ∧ Real.tan θ = a / b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_l842_84204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_five_l842_84211

/-- A function that checks if a natural number contains the digit 5 -/
def containsFive (n : ℕ) : Bool :=
  let digits := n.repr.data
  '5' ∈ digits

/-- The count of numbers from 1 to 600 containing the digit 5 -/
def countWithFive : ℕ :=
  (List.range 600).filter (fun n => containsFive (n + 1)) |>.length

/-- Theorem stating that the count of numbers from 1 to 600 containing the digit 5 is 195 -/
theorem count_numbers_with_five : countWithFive = 195 := by
  sorry

#eval countWithFive

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_five_l842_84211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_inequality_l842_84252

-- Define the polynomial
def polynomial (a b c d x : ℝ) : ℝ := x^5 - a*x^4 + b*x^3 - c*x^2 + d*x - 1

-- State the theorem
theorem roots_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_roots : ∀ x : ℝ, polynomial a b c d x = 0 → x ∈ Set.univ) :
  1/a + 1/b + 1/c + 1/d ≤ 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_inequality_l842_84252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_square_division_l842_84287

/-- A square containing four smaller shaded squares -/
structure ShadedSquare :=
  (side : ℝ)
  (shaded_squares : Finset (ℝ × ℝ))
  (is_valid : shaded_squares.card = 4)

/-- A division of a square into four equal parts -/
structure SquareDivision :=
  (parts : Finset (Finset (ℝ × ℝ)))
  (is_valid : parts.card = 4)

/-- Each part of the division contains exactly one shaded square -/
def contains_one_shaded (sq : ShadedSquare) (div : SquareDivision) : Prop :=
  ∀ part, part ∈ div.parts → (sq.shaded_squares ∩ part).card = 1

/-- The division results in four equal parts -/
def equal_parts (sq : ShadedSquare) (div : SquareDivision) : Prop :=
  ∀ part₁ part₂, part₁ ∈ div.parts → part₂ ∈ div.parts → part₁.card = part₂.card

/-- Main theorem: A square with four shaded squares can be divided into four equal parts,
    each containing one shaded square -/
theorem shaded_square_division (sq : ShadedSquare) :
  ∃ (div : SquareDivision), contains_one_shaded sq div ∧ equal_parts sq div :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_square_division_l842_84287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_average_speed_l842_84219

/-- Calculates the average speed given two separate trips -/
noncomputable def averageSpeed (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) (time2 : ℝ) : ℝ :=
  (distance1 + distance2) / (time1 + time2)

theorem linda_average_speed :
  let distance1 : ℝ := 450
  let time1 : ℝ := 7.5
  let distance2 : ℝ := 480
  let time2 : ℝ := 8
  averageSpeed distance1 time1 distance2 time2 = 60 := by
  -- Unfold the definition of averageSpeed
  unfold averageSpeed
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_average_speed_l842_84219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_number_of_digits_l842_84200

theorem same_number_of_digits (n : ℕ) : 
  ∃ k : ℕ, (10 ^ (k - 1) : ℝ) ≤ (2002 : ℝ) ^ n ∧ (2002 : ℝ) ^ n < 10 ^ k ∧ 
            (10 ^ (k - 1) : ℝ) ≤ (2002 : ℝ) ^ n + (2 : ℝ) ^ n ∧ (2002 : ℝ) ^ n + (2 : ℝ) ^ n < 10 ^ k :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_number_of_digits_l842_84200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_equation_solution_l842_84272

-- Define the operation ⊙
noncomputable def odot (a b : ℝ) : ℝ := (Real.sqrt (3 * a + 2 * b)) ^ 2

-- Theorem statement
theorem odot_equation_solution :
  ∀ y : ℝ, odot 7 y = 64 → y = 43 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odot_equation_solution_l842_84272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_surrounding_tokens_no_more_than_six_tokens_l842_84244

/-- The radius of each circular token -/
def tokenRadius : ℝ := 2

/-- The number of tokens that can be placed around the central token -/
def surroundingTokens : ℕ := 6

/-- Theorem stating the maximum number of surrounding tokens -/
theorem max_surrounding_tokens : 
  ∀ n : ℕ, n ≤ surroundingTokens →
  ∃ (positions : Fin n → ℝ × ℝ),
    (∀ i : Fin n, ‖positions i‖ = 2 * tokenRadius) ∧
    (∀ i j : Fin n, i ≠ j → ‖positions i - positions j‖ ≥ 2 * tokenRadius) ∧
    (∀ i : Fin n, ∃ j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
      ‖positions i - positions j‖ = 2 * tokenRadius ∧
      ‖positions i - positions k‖ = 2 * tokenRadius) :=
by sorry

/-- Theorem stating that more than 6 tokens cannot be placed -/
theorem no_more_than_six_tokens :
  ∀ n : ℕ, n > surroundingTokens →
  ¬∃ (positions : Fin n → ℝ × ℝ),
    (∀ i : Fin n, ‖positions i‖ = 2 * tokenRadius) ∧
    (∀ i j : Fin n, i ≠ j → ‖positions i - positions j‖ ≥ 2 * tokenRadius) ∧
    (∀ i : Fin n, ∃ j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
      ‖positions i - positions j‖ = 2 * tokenRadius ∧
      ‖positions i - positions k‖ = 2 * tokenRadius) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_surrounding_tokens_no_more_than_six_tokens_l842_84244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_slope_range_l842_84233

-- Define the hyperbola C
noncomputable def hyperbola_C (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 8 = 1

-- Define the line that intersects the hyperbola
noncomputable def intersecting_line (t x : ℝ) : ℝ :=
  t * x + 1

-- Define the slope of line PM
noncomputable def slope_PM (t : ℝ) : ℝ :=
  2 / (t^2 + t - 2)

-- Define the range of the slope k
def slope_range (k : ℝ) : Prop :=
  k ∈ Set.Iic (-8/9) ∪ Set.Ioo (8/7) (Real.sqrt 2) ∪ Set.Ioi (Real.sqrt 2)

-- Theorem statement
theorem hyperbola_intersection_slope_range :
  ∀ t : ℝ,
  -3/2 < t → t < 3/2 →
  t ≠ Real.sqrt 2 → t ≠ -Real.sqrt 2 → t ≠ 1 →
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁ ≠ x₂ ∧
    hyperbola_C x₁ y₁ ∧ hyperbola_C x₂ y₂ ∧
    y₁ = intersecting_line t x₁ ∧ y₂ = intersecting_line t x₂) →
  slope_range (slope_PM t) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_slope_range_l842_84233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_sum_l842_84202

theorem rationalize_and_sum :
  ∃ (A B C D E F : ℤ),
    (F > 0) ∧
    (¬ ∃ (p : ℕ), Prime p ∧ (p * p ∣ B.natAbs)) ∧
    (¬ ∃ (p : ℕ), Prime p ∧ (p * p ∣ D.natAbs)) ∧
    (Nat.gcd (Nat.gcd (Nat.gcd A.natAbs C.natAbs) E.natAbs) F.natAbs = 1) ∧
    (7 / (3 + 2 * Real.sqrt 2 + Real.sqrt 3) =
      (A * Real.sqrt B.toNat + C * Real.sqrt D.toNat + E) / F) ∧
    (A + C + E + F + B + D = 51) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_and_sum_l842_84202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_initial_balloons_l842_84288

theorem sam_initial_balloons 
  (fred_balloons : ℕ) 
  (dan_balloons : ℕ) 
  (total_balloons : ℕ) 
  (sam_initial_balloons : ℕ)
  (h1 : fred_balloons = 10)
  (h2 : dan_balloons = 16)
  (h3 : total_balloons = 52)
  (h4 : total_balloons = (sam_initial_balloons - fred_balloons) + dan_balloons) :
  sam_initial_balloons = 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_initial_balloons_l842_84288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_n_l842_84255

/-- Given a positive integer n that satisfies the equation n*(n+1)/2 = 781,
    prove that the sum of the digits of n is 12. -/
theorem sum_of_digits_of_n (n : ℕ+) 
  (h : n.val * (n.val + 1) / 2 = 781) : 
  (n.val.repr.toList.map (fun c => c.toNat - 48)).sum = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_n_l842_84255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_phi_l842_84282

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)

theorem min_abs_phi :
  ∃ φ_min : ℝ, 
    (∀ φ : ℝ, (∀ x : ℝ, f φ x = f φ (4 * π / 3 - x)) → |φ| ≥ |φ_min|) ∧ 
    φ_min = π / 3 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_phi_l842_84282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_447_l842_84298

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Converts a three-digit number to its decimal representation -/
def toDecimal (a b c : Digit) : ℚ :=
  (a.val : ℚ) * 100 + (b.val : ℚ) * 10 + (c.val : ℚ)

/-- Converts a two-digit repeating decimal to a rational number -/
def repeatingDecimalAB (a b : Digit) : ℚ :=
  (10 * (a.val : ℚ) + (b.val : ℚ)) / 99

/-- Converts a three-digit repeating decimal to a rational number -/
def repeatingDecimalABC (a b c : Digit) : ℚ :=
  (100 * (a.val : ℚ) + 10 * (b.val : ℚ) + (c.val : ℚ)) / 999

/-- The main theorem stating that the only solution to the equation is 447 -/
theorem unique_solution_is_447 :
  ∃! (a b c : Digit), 
    repeatingDecimalAB a b + repeatingDecimalABC a b c = 33 / 37 ∧
    toDecimal a b c = 447 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_447_l842_84298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_2_range_of_a_for_f_non_negative_l842_84266

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1) / (x + 1)

-- Part 1: f is increasing when a = 2
theorem f_increasing_when_a_2 :
  ∀ x > 0, deriv (f 2) x ≥ 0 := by sorry

-- Part 2: Range of a for which f(x) ≥ 0 on [1, +∞) and f(1) = 0
theorem range_of_a_for_f_non_negative :
  ∀ a : ℝ, (∀ x ≥ 1, f a x ≥ 0) ∧ (f a 1 = 0) ↔ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_when_a_2_range_of_a_for_f_non_negative_l842_84266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_plus_alpha_l842_84203

theorem tan_pi_4_plus_alpha (α : ℝ) : 
  Real.tan (π / 4 + α) = 2 → 1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 15 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_4_plus_alpha_l842_84203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_new_reading_total_l842_84256

/-- Jim's reading habits --/
structure ReadingHabits where
  originalRate : ℚ  -- pages per hour
  originalTotal : ℚ  -- pages per week
  speedIncrease : ℚ  -- percentage increase
  hourDecrease : ℚ  -- hours decreased per week

/-- Calculate the new total pages read per week --/
def newTotalPages (habits : ReadingHabits) : ℚ :=
  let originalHours := habits.originalTotal / habits.originalRate
  let newHours := originalHours - habits.hourDecrease
  let newRate := habits.originalRate * (1 + habits.speedIncrease)
  newHours * newRate

/-- Theorem stating that Jim now reads 660 pages per week --/
theorem jim_new_reading_total : 
  let habits : ReadingHabits := {
    originalRate := 40,
    originalTotal := 600,
    speedIncrease := 1/2,
    hourDecrease := 4
  }
  newTotalPages habits = 660 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_new_reading_total_l842_84256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clever_value_points_existence_l842_84265

def clever_value_point (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = f' x

theorem clever_value_points_existence :
  (clever_value_point (λ x ↦ x^2) (λ x ↦ 2*x)) ∧
  (¬ clever_value_point (λ x ↦ 1/Real.exp x) (λ x ↦ -1/Real.exp x)) ∧
  (clever_value_point Real.log (λ x ↦ 1/x)) ∧
  (¬ clever_value_point Real.tan (λ x ↦ 1/(Real.cos x)^2)) ∧
  (clever_value_point (λ x ↦ x + 1/x) (λ x ↦ 1 - 1/x^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clever_value_points_existence_l842_84265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identity_l842_84242

theorem triangle_trig_identity (A B C : ℝ) (a b c h : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  c = a * Real.sin B / Real.sin A ∧
  b = a * Real.sin C / Real.sin A ∧
  c - a = h →
  Real.sin ((C - A) / 2) + Real.cos ((C + A) / 2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identity_l842_84242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_720_l842_84226

theorem divisors_of_720 : 
  (Finset.filter (λ x => 720 % x = 0) (Finset.range 721)).card = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_720_l842_84226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_cost_prices_l842_84229

noncomputable def cost_price_with_markup (selling_price : ℝ) (markup_percent : ℝ) : ℝ :=
  selling_price / (1 + markup_percent / 100)

noncomputable def cost_price_with_discount_and_tax (selling_price : ℝ) (discount_percent : ℝ) (tax_percent : ℝ) : ℝ :=
  let price_before_tax := selling_price / (1 + tax_percent / 100)
  price_before_tax / (1 - discount_percent / 100)

noncomputable def cost_price_with_markup_and_tax (selling_price : ℝ) (markup_percent : ℝ) (tax_percent : ℝ) : ℝ :=
  let price_before_tax := selling_price / (1 + tax_percent / 100)
  price_before_tax / (1 + markup_percent / 100)

theorem furniture_cost_prices (ε : ℝ) (ε_pos : ε > 0) :
  let computer_table := cost_price_with_markup 8500 35
  let bookshelf := cost_price_with_markup 6200 28
  let chair := cost_price_with_discount_and_tax 3600 18 10
  let lamp := cost_price_with_markup_and_tax 1600 12 5
  (abs (computer_table - 6296.30) < ε) ∧
  (abs (bookshelf - 4843.75) < ε) ∧
  (abs (chair - 3991.13) < ε) ∧
  (abs (lamp - 1360.72) < ε) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_cost_prices_l842_84229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_example_l842_84249

noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

theorem geometric_series_sum_example :
  let a : ℝ := -2
  let r : ℝ := 3
  let n : ℕ := 10
  geometric_series_sum a r n = -59048 :=
by
  -- Unfold the definition and simplify
  unfold geometric_series_sum
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_example_l842_84249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_christopher_stroll_time_l842_84297

/-- The time (in hours) taken to travel a given distance at a constant speed -/
noncomputable def travel_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

/-- Christopher's stroll -/
theorem christopher_stroll_time : 
  let distance : ℝ := 5
  let speed : ℝ := 4
  travel_time distance speed = 1.25 := by
  -- Unfold the definition of travel_time
  unfold travel_time
  -- Simplify the expression
  simp
  -- Check that 5 / 4 = 1.25
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_christopher_stroll_time_l842_84297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painted_cubes_count_l842_84275

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  sideLength : n > 0

/-- Calculates the number of corner cubes in a cube -/
def cornerCubes : ℕ := 8

/-- Calculates the number of edge cubes in a cube -/
def edgeCubes (c : Cube n) : ℕ := 12 * (n - 2)

/-- Calculates the number of face center cubes in a cube -/
def faceCenterCubes (c : Cube n) : ℕ := 6 * (n - 2) * (n - 2)

/-- Calculates the total number of cubes with at least one painted face -/
def totalPaintedCubes (c : Cube n) : ℕ :=
  cornerCubes + edgeCubes c + faceCenterCubes c

theorem painted_cubes_count (c : Cube 10) :
  totalPaintedCubes c = 488 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_painted_cubes_count_l842_84275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_entertainment_expenses_calculation_l842_84243

/-- Calculates the total amount spent on entertainment and additional expenses --/
noncomputable def total_entertainment_expenses (game_discounted_price : ℝ) 
                                 (game_discount_percentage : ℝ)
                                 (ticket_original_price : ℝ)
                                 (number_of_tickets : ℕ)
                                 (entertainment_tax_percentage : ℝ)
                                 (snacks_cost : ℝ)
                                 (transportation_cost : ℝ)
                                 (number_of_trips : ℕ) : ℝ :=
  let game_original_price := game_discounted_price / (1 - game_discount_percentage / 100)
  let ticket_with_tax := ticket_original_price * (1 + entertainment_tax_percentage / 100)
  let total_ticket_cost := ticket_with_tax * (number_of_tickets : ℝ)
  let additional_expenses := (snacks_cost + transportation_cost) * (number_of_trips : ℝ)
  game_discounted_price + total_ticket_cost + additional_expenses

theorem entertainment_expenses_calculation :
  total_entertainment_expenses 66 15 12 3 10 7 5 1 = 117.60 := by
  -- Unfold the definition and simplify
  unfold total_entertainment_expenses
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_entertainment_expenses_calculation_l842_84243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_when_angle_minimized_l842_84295

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define the point M
def M : ℝ × ℝ := (2, 1)

-- Define a line passing through M
def line_through_M (m : ℝ) (x y : ℝ) : Prop := y - M.2 = m * (x - M.1)

-- Define the intersection of a line with the circle
def intersects_circle (m : ℝ) : Prop := ∃ (x₁ y₁ x₂ y₂ : ℝ), 
  x₁ ≠ x₂ ∧ 
  line_through_M m x₁ y₁ ∧ 
  line_through_M m x₂ y₂ ∧ 
  my_circle x₁ y₁ ∧ 
  my_circle x₂ y₂

-- Define the center of the circle
def C : ℝ × ℝ := (1, 0)

-- Define the angle minimization condition
def angle_minimized (m : ℝ) : Prop := m = -1

-- The main theorem
theorem line_equation_when_angle_minimized :
  ∀ m : ℝ, intersects_circle m → angle_minimized m →
  ∀ x y : ℝ, line_through_M m x y ↔ x + y - 3 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_when_angle_minimized_l842_84295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_M_l842_84286

def M : ℕ := 35^5 + 5*35^4 + 10*35^3 + 10*35^2 + 5*35 + 1

theorem number_of_factors_M : (Finset.filter (fun x => x ∣ M) (Finset.range (M + 1))).card = 121 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_M_l842_84286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_line_equation_l842_84236

/-- A line passing through a point with equal distances from two other points -/
structure EqualDistanceLine where
  -- The slope of the line
  k : ℝ
  -- The line passes through point (3,4)
  passes_through : k * 3 - 4 + 4 - 3 * k = 0
  -- The distances from (-2,2) and (4,-2) to the line are equal
  equal_distances : 
    |k * -2 - 2 + 4 - 3 * k| / Real.sqrt (1 + k^2) = 
    |k * 4 + -2 + 4 - 3 * k| / Real.sqrt (1 + k^2)

/-- The equation of the line is either 2x+3y-18=0 or 2x-y-2=0 -/
theorem equal_distance_line_equation (l : EqualDistanceLine) :
  (l.k = 2 ∧ ∀ x y, 2 * x - y - 2 = 0 → True) ∨
  (l.k = -2/3 ∧ ∀ x y, 2 * x + 3 * y - 18 = 0 → True) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_distance_line_equation_l842_84236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_cracking_theorem_l842_84224

/-- A type representing a seven-digit code with all different digits -/
def GoodCode := { c : Fin 10 → Fin 10 // Function.Injective c ∧ (∀ i, i.val < 7 → c i < 10) }

/-- A function that checks if two codes match at least one digit in the same position -/
def matchCodes (c1 c2 : GoodCode) : Prop :=
  ∃ i : Fin 7, c1.val i = c2.val i

/-- The main theorem stating that it's possible to open the safe in six attempts -/
theorem safe_cracking_theorem :
  ∀ (password : GoodCode),
  ∃ (attempts : Fin 6 → GoodCode),
  ∃ (i : Fin 6), matchCodes (attempts i) password :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_cracking_theorem_l842_84224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_values_l842_84201

theorem min_values (x y z : ℝ) 
  (hx : x ∈ Set.Icc (-3) 7) 
  (hy : y ∈ Set.Icc (-2) 5) 
  (hz : z ∈ Set.Icc (-5) 3) : 
  (∀ a b, a ∈ Set.Icc (-3) 7 → b ∈ Set.Icc (-2) 5 → x^2 + y^2 ≤ a^2 + b^2) ∧
  (∀ a b c, a ∈ Set.Icc (-3) 7 → b ∈ Set.Icc (-2) 5 → c ∈ Set.Icc (-5) 3 → x*y*z - z^2 ≤ a*b*c - c^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_values_l842_84201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_l842_84208

/-- Calculates the length of a bridge given the length of a train, its speed, and the time it takes to cross the bridge. -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 120)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time = 30) :
  (train_speed_kmh * (1000 / 3600) * crossing_time - train_length) = 255 :=
by
  -- Convert speed from km/h to m/s
  have train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  
  -- Calculate total distance
  have total_distance : ℝ := train_speed_ms * crossing_time
  
  -- Calculate bridge length
  have bridge_length : ℝ := total_distance - train_length
  
  -- Prove the equality
  sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_l842_84208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_iff_m_eq_three_l842_84238

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (2 * m^2 - 17) * x^(m - 2)

-- State the theorem
theorem monotonically_increasing_iff_m_eq_three :
  ∃ (m : ℝ), ∀ (x y : ℝ), 0 < x → 0 < y → x < y → f m x < f m y ↔ m = 3 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_iff_m_eq_three_l842_84238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inspection_journey_l842_84230

def travel_records : List Int := [-4, 7, -9, 8, 6, -5, -2, -4]

def fuel_rate : Rat := 1/2

theorem inspection_journey :
  let net_distance := travel_records.sum
  let total_distance := (travel_records.map Int.natAbs).sum
  let total_fuel := fuel_rate * (total_distance : Rat)
  (net_distance = -3 ∧
   total_distance = 45 ∧
   total_fuel = 45/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inspection_journey_l842_84230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l842_84221

/-- The probability of getting tails on a single flip of a fair coin -/
noncomputable def p_tails : ℝ := 1/2

/-- The number of times the coin is flipped -/
def n_flips : ℕ := 5

/-- The number of initial flips that should be tails -/
def n_tails : ℕ := 2

/-- The number of final flips that should not be tails -/
def n_not_tails : ℕ := 3

/-- 
Theorem: The probability of getting tails on the first 2 flips 
and not tails on the last 3 flips of a fair coin is 1/32
-/
theorem coin_flip_probability : 
  p_tails ^ n_tails * (1 - p_tails) ^ n_not_tails = 1/32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l842_84221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_times_l842_84270

/-- Represents a traveler with a constant speed -/
structure Traveler where
  speed : ℝ
  startPosition : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  distanceBetweenCities : ℝ
  mikhail : Traveler
  hariton : Traveler
  nikolai : Traveler

/-- Checks if the problem setup satisfies the given conditions -/
def validSetup (setup : ProblemSetup) : Prop :=
  let t₁ := 1.5  -- Time from start to 9:30 AM in hours
  let t₂ := 2    -- Time from start to 10:00 AM in hours
  let S := setup.distanceBetweenCities
  let x := setup.mikhail.speed
  let y := setup.hariton.speed
  let z := setup.nikolai.speed
  
  -- Condition at 9:30 AM
  S - (t₁ * y) - (t₁ * x) = t₁ * y - t₁ * z ∧
  -- Condition at 10:00 AM
  t₂ * x = S - t₂ * y - (S - t₂ * x - t₂ * z)

/-- Calculates the meeting time between two travelers -/
noncomputable def meetingTime (s : ℝ) (v₁ v₂ : ℝ) : ℝ :=
  s / (v₁ + v₂)

/-- The main theorem to prove -/
theorem meeting_times (setup : ProblemSetup) 
  (h : validSetup setup) :
  let t_mh := meetingTime setup.distanceBetweenCities setup.mikhail.speed setup.hariton.speed
  let t_mn := meetingTime setup.distanceBetweenCities setup.mikhail.speed setup.nikolai.speed
  t_mh = 9/5 ∧ t_mn = 9/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_times_l842_84270
