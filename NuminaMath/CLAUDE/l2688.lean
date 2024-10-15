import Mathlib

namespace NUMINAMATH_CALUDE_drain_time_to_half_l2688_268876

/-- Represents the remaining water volume in the pool after draining for a given time. -/
def remaining_water (t : ℝ) : ℝ := 300 - 25 * t

/-- Proves that it takes 6 hours to drain the pool from 300 m³ to 150 m³. -/
theorem drain_time_to_half : ∃ t : ℝ, t = 6 ∧ remaining_water t = 150 := by
  sorry

end NUMINAMATH_CALUDE_drain_time_to_half_l2688_268876


namespace NUMINAMATH_CALUDE_abs_func_differentiable_l2688_268817

-- Define the absolute value function
def abs_func (x : ℝ) : ℝ := |x|

-- State the theorem
theorem abs_func_differentiable :
  ∀ x : ℝ, x ≠ 0 →
    (DifferentiableAt ℝ abs_func x) ∧
    (deriv abs_func x = if x > 0 then 1 else -1) :=
by sorry

end NUMINAMATH_CALUDE_abs_func_differentiable_l2688_268817


namespace NUMINAMATH_CALUDE_largest_negative_integer_congruence_l2688_268855

theorem largest_negative_integer_congruence :
  ∃ (x : ℤ), x = -6 ∧
  (34 * x + 6) % 20 = 2 % 20 ∧
  ∀ (y : ℤ), y < 0 → (34 * y + 6) % 20 = 2 % 20 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_integer_congruence_l2688_268855


namespace NUMINAMATH_CALUDE_point_on_linear_graph_l2688_268887

theorem point_on_linear_graph (a : ℝ) : (1 : ℝ) = 3 * a + 4 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_linear_graph_l2688_268887


namespace NUMINAMATH_CALUDE_max_regions_five_lines_l2688_268898

/-- The number of regions created by n intersecting lines in a plane --/
def num_regions (n : ℕ) : ℕ := sorry

/-- The maximum number of regions created by n intersecting lines in a rectangle --/
def max_regions_rectangle (n : ℕ) : ℕ := num_regions n

theorem max_regions_five_lines : 
  max_regions_rectangle 5 = 16 := by sorry

end NUMINAMATH_CALUDE_max_regions_five_lines_l2688_268898


namespace NUMINAMATH_CALUDE_largest_valid_three_digit_number_l2688_268873

/-- Represents a three-digit number as a tuple of its digits -/
def ThreeDigitNumber := (Nat × Nat × Nat)

/-- Converts a ThreeDigitNumber to its numerical value -/
def toNumber (n : ThreeDigitNumber) : Nat :=
  100 * n.1 + 10 * n.2.1 + n.2.2

/-- Calculates the sum of digits of a ThreeDigitNumber -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.1 + n.2.1 + n.2.2

/-- Checks if a ThreeDigitNumber satisfies all conditions -/
def isValid (n : ThreeDigitNumber) : Prop :=
  n.1 ≠ 0 ∧  -- Ensures it's a three-digit number
  n.1 = n.2.2 ∧  -- First digit matches third digit
  n.1 ≠ n.2.1 ∧  -- First digit doesn't match second digit
  (toNumber n) % (digitSum n) = 0  -- Number is divisible by sum of its digits

theorem largest_valid_three_digit_number :
  ∀ n : ThreeDigitNumber, isValid n → toNumber n ≤ 828 :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_three_digit_number_l2688_268873


namespace NUMINAMATH_CALUDE_jessica_bank_balance_l2688_268891

theorem jessica_bank_balance (B : ℝ) 
  (h1 : B * (2/5) = 200)  -- Condition: 2/5 of initial balance equals $200
  (h2 : B > 200)          -- Implicit condition: initial balance is greater than withdrawal
  : B - 200 + (B - 200) / 2 = 450 := by
  sorry

end NUMINAMATH_CALUDE_jessica_bank_balance_l2688_268891


namespace NUMINAMATH_CALUDE_spirangle_length_is_10301_l2688_268867

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def spirangle_length (a₁ : ℕ) (d : ℕ) (last_seq : ℕ) (final_seg : ℕ) : ℕ :=
  let n := (last_seq - a₁) / d + 1
  arithmetic_sequence_sum a₁ d n + final_seg

theorem spirangle_length_is_10301 :
  spirangle_length 2 2 200 201 = 10301 :=
by sorry

end NUMINAMATH_CALUDE_spirangle_length_is_10301_l2688_268867


namespace NUMINAMATH_CALUDE_fox_max_berries_l2688_268826

/-- The number of bear cubs -/
def num_cubs : ℕ := 100

/-- The total number of berries initially -/
def total_berries : ℕ := 2^num_cubs - 1

/-- The maximum number of berries the fox can eat -/
def max_fox_berries : ℕ := 1

/-- Theorem stating the maximum number of berries the fox can eat -/
theorem fox_max_berries :
  max_fox_berries = (total_berries % num_cubs) :=
by sorry

end NUMINAMATH_CALUDE_fox_max_berries_l2688_268826


namespace NUMINAMATH_CALUDE_land_area_calculation_l2688_268813

/-- The total area of 9 square-shaped plots of land, each measuring 6 meters in length and width, is 324 square meters. -/
theorem land_area_calculation (num_plots : ℕ) (side_length : ℝ) : 
  num_plots = 9 → side_length = 6 → num_plots * (side_length * side_length) = 324 := by
  sorry

end NUMINAMATH_CALUDE_land_area_calculation_l2688_268813


namespace NUMINAMATH_CALUDE_friends_weight_loss_l2688_268889

/-- The combined weight loss of two friends over different periods -/
theorem friends_weight_loss (aleesia_weekly_loss : ℝ) (aleesia_weeks : ℕ)
                             (alexei_weekly_loss : ℝ) (alexei_weeks : ℕ) :
  aleesia_weekly_loss = 1.5 ∧ 
  aleesia_weeks = 10 ∧
  alexei_weekly_loss = 2.5 ∧ 
  alexei_weeks = 8 →
  aleesia_weekly_loss * aleesia_weeks + alexei_weekly_loss * alexei_weeks = 35 := by
  sorry

end NUMINAMATH_CALUDE_friends_weight_loss_l2688_268889


namespace NUMINAMATH_CALUDE_line_properties_l2688_268834

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

def isDirectionalVector (v : Vector2D) (l : Line2D) : Prop :=
  v.y / v.x = -l.a / l.b

def hasEqualIntercepts (l : Line2D) : Prop :=
  -l.c / l.a = -l.c / l.b

def passesThrough (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def yIntercept (m : ℝ) (b : ℝ) : ℝ := b

theorem line_properties :
  let l1 : Line2D := ⟨2, 1, 3⟩
  let l2 : Line2D := ⟨1, 1, -6⟩
  let v : Vector2D := ⟨1, -2⟩
  let p : Point2D := ⟨2, 4⟩
  isDirectionalVector v l1 ∧
  hasEqualIntercepts l2 ∧
  passesThrough l2 p ∧
  yIntercept 3 (-2) = -2 := by sorry

end NUMINAMATH_CALUDE_line_properties_l2688_268834


namespace NUMINAMATH_CALUDE_smallest_a1_l2688_268888

/-- Given a sequence of positive real numbers where aₙ = 8aₙ₋₁ - n² for all n > 1,
    the smallest possible value of a₁ is 2/7 -/
theorem smallest_a1 (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
    (h_rec : ∀ n > 1, a n = 8 * a (n - 1) - n^2) :
  ∀ a₁ > 0, (∀ n > 1, a n = 8 * a (n - 1) - n^2) → a₁ ≥ 2/7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a1_l2688_268888


namespace NUMINAMATH_CALUDE_root_product_theorem_l2688_268838

theorem root_product_theorem (a b : ℂ) : 
  a ≠ b →
  a^4 + a^3 - 1 = 0 →
  b^4 + b^3 - 1 = 0 →
  (a*b)^6 + (a*b)^4 + (a*b)^3 - (a*b)^2 - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_product_theorem_l2688_268838


namespace NUMINAMATH_CALUDE_remainder_equality_l2688_268897

theorem remainder_equality (A B C S T s t : ℕ) 
  (h1 : A > B)
  (h2 : A^2 % C = S)
  (h3 : B^2 % C = T)
  (h4 : (A^2 * B^2) % C = s)
  (h5 : (S * T) % C = t) :
  s = t := by
  sorry

end NUMINAMATH_CALUDE_remainder_equality_l2688_268897


namespace NUMINAMATH_CALUDE_max_product_of_three_numbers_l2688_268831

theorem max_product_of_three_numbers (n : ℕ) :
  let S := Finset.range (3 * n + 1) \ {0}
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
    a < b ∧ b < c ∧
    a + b + c = 3 * n ∧
    ∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S →
      x < y → y < z →
      x + y + z = 3 * n →
      x * y * z ≤ a * b * c ∧
    a * b * c = n^3 - n :=
by sorry

end NUMINAMATH_CALUDE_max_product_of_three_numbers_l2688_268831


namespace NUMINAMATH_CALUDE_owen_sleep_hours_l2688_268860

theorem owen_sleep_hours (total_hours work_hours chore_hours sleep_hours : ℕ) :
  total_hours = 24 ∧ work_hours = 6 ∧ chore_hours = 7 ∧ 
  sleep_hours = total_hours - (work_hours + chore_hours) →
  sleep_hours = 11 := by
  sorry

end NUMINAMATH_CALUDE_owen_sleep_hours_l2688_268860


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2688_268871

def i : ℂ := Complex.I

theorem complex_fraction_simplification :
  (2 - 3 * i) / (1 + 4 * i) = -10/17 - 11/17 * i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2688_268871


namespace NUMINAMATH_CALUDE_initial_average_age_l2688_268852

theorem initial_average_age (n : ℕ) (new_person_age : ℕ) (new_average : ℚ) :
  n = 17 ∧ new_person_age = 32 ∧ new_average = 15 →
  ∃ initial_average : ℚ, 
    initial_average * n + new_person_age = new_average * (n + 1) ∧
    initial_average = 14 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_age_l2688_268852


namespace NUMINAMATH_CALUDE_garden_area_l2688_268803

theorem garden_area (width length perimeter area : ℝ) : 
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 72 →
  area = width * length →
  area = 243 := by
sorry

end NUMINAMATH_CALUDE_garden_area_l2688_268803


namespace NUMINAMATH_CALUDE_legislation_approval_probability_l2688_268850

/-- The probability of a voter approving the legislation -/
def p_approve : ℝ := 0.6

/-- The number of voters surveyed -/
def n : ℕ := 4

/-- The number of approving voters we're interested in -/
def k : ℕ := 2

/-- The probability of exactly k out of n voters approving the legislation -/
def prob_k_approve (p : ℝ) (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem legislation_approval_probability :
  prob_k_approve p_approve n k = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_legislation_approval_probability_l2688_268850


namespace NUMINAMATH_CALUDE_new_students_count_l2688_268835

theorem new_students_count (n : ℕ) : n < 600 → n % 28 = 27 → n % 26 = 20 → n = 615 :=
by
  sorry

end NUMINAMATH_CALUDE_new_students_count_l2688_268835


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_range_of_m_l2688_268875

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((x + a) * Real.log x) / (x + 1)

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ :=
  ((Real.log x + (x + a) / x) * (x + 1) - (x + a) * Real.log x) / ((x + 1)^2)

-- Theorem for part (I)
theorem tangent_line_perpendicular (a : ℝ) :
  f_derivative a 1 = 1/2 → a = 0 :=
by sorry

-- Theorem for part (II)
theorem range_of_m (m : ℝ) :
  (∀ x ≥ 1, f 0 x ≤ m * (x - 1)) ↔ m ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_range_of_m_l2688_268875


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l2688_268805

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Theorem: For a parabola x² = 2py (p > 0 and constant), if a line with slope 1
    passing through the focus intersects the parabola at points A and B,
    then the length of AB is 4p. -/
theorem parabola_intersection_length
  (p : ℝ)
  (hp : p > 0)
  (A B : ParabolaPoint)
  (h_parabola_A : A.x^2 = 2*p*A.y)
  (h_parabola_B : B.x^2 = 2*p*B.y)
  (h_line : B.y - A.y = B.x - A.x)
  (h_focus : ∃ (f : ℝ), A.y = A.x + f ∧ B.y = B.x + f ∧ f = p/2) :
  Real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2) = 4*p :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l2688_268805


namespace NUMINAMATH_CALUDE_function_from_derivative_and_point_l2688_268870

open Real

theorem function_from_derivative_and_point (f : ℝ → ℝ) 
  (h1 : ∀ x, HasDerivAt f (4 * x^3) x) 
  (h2 : f 1 = -1) : 
  ∀ x, f x = x^4 - 2 := by
  sorry

end NUMINAMATH_CALUDE_function_from_derivative_and_point_l2688_268870


namespace NUMINAMATH_CALUDE_distance_in_15_minutes_l2688_268877

/-- Given a constant speed calculated from driving 80 miles in 2 hours, 
    prove that the distance traveled in 15 minutes is 10 miles. -/
theorem distance_in_15_minutes (total_distance : ℝ) (total_time : ℝ) 
  (travel_time : ℝ) (h1 : total_distance = 80) (h2 : total_time = 2) 
  (h3 : travel_time = 15 / 60) : 
  (total_distance / total_time) * travel_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_in_15_minutes_l2688_268877


namespace NUMINAMATH_CALUDE_karen_picked_up_three_cases_l2688_268874

/-- The number of boxes of Tagalongs Karen sold -/
def boxes_sold : ℕ := 36

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 12

/-- The number of cases Karen picked up -/
def cases_picked_up : ℕ := boxes_sold / boxes_per_case

theorem karen_picked_up_three_cases : cases_picked_up = 3 := by
  sorry

end NUMINAMATH_CALUDE_karen_picked_up_three_cases_l2688_268874


namespace NUMINAMATH_CALUDE_no_rectangle_from_five_distinct_squares_l2688_268848

/-- A configuration of five squares with side lengths q₁, q₂, q₃, q₄, q₅ -/
structure FiveSquares where
  q₁ : ℝ
  q₂ : ℝ
  q₃ : ℝ
  q₄ : ℝ
  q₅ : ℝ
  h₁ : 0 < q₁
  h₂ : q₁ < q₂
  h₃ : q₂ < q₃
  h₄ : q₃ < q₄
  h₅ : q₄ < q₅

/-- Predicate to check if the five squares can form a rectangle -/
def CanFormRectangle (s : FiveSquares) : Prop :=
  ∃ (w h : ℝ), w > 0 ∧ h > 0 ∧ w * h = s.q₁^2 + s.q₂^2 + s.q₃^2 + s.q₄^2 + s.q₅^2

/-- Theorem stating that it's impossible to form a rectangle with five squares of distinct sizes -/
theorem no_rectangle_from_five_distinct_squares :
  ¬∃ (s : FiveSquares), CanFormRectangle s := by
  sorry

end NUMINAMATH_CALUDE_no_rectangle_from_five_distinct_squares_l2688_268848


namespace NUMINAMATH_CALUDE_factorization_cubic_l2688_268880

theorem factorization_cubic (a : ℝ) : a^3 - 10*a^2 + 25*a = a*(a-5)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_l2688_268880


namespace NUMINAMATH_CALUDE_integer_sum_problem_l2688_268802

theorem integer_sum_problem : ∃ (a b : ℕ+), 
  (a * b + a + b = 167) ∧ 
  (Nat.gcd a.val b.val = 1) ∧ 
  (a < 30) ∧ (b < 30) ∧ 
  (a + b = 24) := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l2688_268802


namespace NUMINAMATH_CALUDE_range_of_b_l2688_268886

def solution_set (b : ℝ) : Set ℝ := {x : ℝ | |3*x - b| < 4}

theorem range_of_b :
  (∃ b : ℝ, solution_set b = {1, 2, 3}) →
  (∀ b : ℝ, solution_set b = {1, 2, 3} → b ∈ Set.Ioo 5 7) ∧
  (∀ b : ℝ, b ∈ Set.Ioo 5 7 → solution_set b = {1, 2, 3}) :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_l2688_268886


namespace NUMINAMATH_CALUDE_athlete_shots_l2688_268808

theorem athlete_shots (x y z : ℕ) : 
  x > 0 → y > 0 → z > 0 →  -- Each point value scored at least once
  x + y + z > 11 →         -- More than 11 shots
  8*x + 9*y + 10*z = 100 → -- Total score is 100
  x = 9                    -- Number of 8-point shots is 9
  := by sorry

end NUMINAMATH_CALUDE_athlete_shots_l2688_268808


namespace NUMINAMATH_CALUDE_solve_for_c_l2688_268801

theorem solve_for_c (y : ℝ) (h1 : y > 0) : 
  ∃ c : ℝ, (7 * y) / 20 + (c * y) / 10 = 0.6499999999999999 * y ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_c_l2688_268801


namespace NUMINAMATH_CALUDE_sample_survey_suitability_l2688_268825

-- Define the set of all surveys
def Surveys : Set Nat := {1, 2, 3, 4}

-- Define the characteristics of each survey
def is_destructive_testing (s : Nat) : Prop :=
  s = 1 ∨ s = 4

def has_large_scope (s : Nat) : Prop :=
  s = 2

def has_small_scope (s : Nat) : Prop :=
  s = 3

-- Define what makes a survey suitable for sampling
def suitable_for_sampling (s : Nat) : Prop :=
  is_destructive_testing s ∨ has_large_scope s

-- Theorem to prove
theorem sample_survey_suitability :
  {s ∈ Surveys | suitable_for_sampling s} = {1, 2, 4} := by
  sorry


end NUMINAMATH_CALUDE_sample_survey_suitability_l2688_268825


namespace NUMINAMATH_CALUDE_sum_x_y_equals_seven_a_l2688_268863

theorem sum_x_y_equals_seven_a (a x y : ℝ) (h1 : a / x = 1 / 3) (h2 : a / y = 1 / 4) :
  x + y = 7 * a := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_seven_a_l2688_268863


namespace NUMINAMATH_CALUDE_xiaoMingCarbonEmissions_l2688_268800

/-- The carbon dioxide emissions formula for household tap water usage -/
def carbonEmissions (x : ℝ) : ℝ := 0.9 * x

/-- Xiao Ming's household tap water usage in tons -/
def xiaoMingWaterUsage : ℝ := 10

theorem xiaoMingCarbonEmissions :
  carbonEmissions xiaoMingWaterUsage = 9 := by
  sorry

end NUMINAMATH_CALUDE_xiaoMingCarbonEmissions_l2688_268800


namespace NUMINAMATH_CALUDE_int_endomorphisms_characterization_l2688_268869

/-- An endomorphism of the additive group of integers -/
def IntEndomorphism : Type := ℤ → ℤ

/-- The homomorphism property for integer endomorphisms -/
def IsHomomorphism (φ : IntEndomorphism) : Prop :=
  ∀ a b : ℤ, φ (a + b) = φ a + φ b

/-- The set of all endomorphisms of the additive group of integers -/
def IntEndomorphisms : Set IntEndomorphism :=
  {φ : IntEndomorphism | IsHomomorphism φ}

/-- A linear function with integer coefficient -/
def LinearIntFunction (d : ℤ) : IntEndomorphism :=
  fun x => d * x

theorem int_endomorphisms_characterization :
  ∀ φ : IntEndomorphism, φ ∈ IntEndomorphisms ↔ ∃ d : ℤ, φ = LinearIntFunction d :=
by sorry

end NUMINAMATH_CALUDE_int_endomorphisms_characterization_l2688_268869


namespace NUMINAMATH_CALUDE_system_solution_l2688_268823

theorem system_solution (x y : ℝ) 
  (eq1 : 4 * x - y = 2) 
  (eq2 : 3 * x - 2 * y = -1) : 
  x - y = -1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2688_268823


namespace NUMINAMATH_CALUDE_function_transformation_l2688_268896

theorem function_transformation (g : ℝ → ℝ) (h : ∀ x, g (x + 2) = 2 * x + 3) :
  ∀ x, g x = 2 * x - 1 := by
sorry

end NUMINAMATH_CALUDE_function_transformation_l2688_268896


namespace NUMINAMATH_CALUDE_parabola_translation_l2688_268811

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2 + 1

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := (x - 3)^2 - 1

-- Theorem statement
theorem parabola_translation :
  ∀ x y : ℝ, y = original_parabola (x - 3) - 2 ↔ y = translated_parabola x :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2688_268811


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2688_268832

theorem absolute_value_inequality_solution_set :
  {x : ℝ | 1 ≤ |x + 2| ∧ |x + 2| ≤ 5} = {x : ℝ | (-7 ≤ x ∧ x ≤ -3) ∨ (-1 ≤ x ∧ x ≤ 3)} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2688_268832


namespace NUMINAMATH_CALUDE_point_transformation_l2688_268816

/-- Given a point B(5, -1) moved 3 units upwards to point A(a+1, 1-b), prove that a = 4 and b = -1 -/
theorem point_transformation (a b : ℝ) : 
  (5 : ℝ) = a + 1 ∧ 
  (1 : ℝ) - b = -1 + 3 → 
  a = 4 ∧ b = -1 := by sorry

end NUMINAMATH_CALUDE_point_transformation_l2688_268816


namespace NUMINAMATH_CALUDE_situp_difference_l2688_268895

/-- The number of sit-ups Ken can do -/
def ken_situps : ℕ := 20

/-- The number of sit-ups Nathan can do -/
def nathan_situps : ℕ := 2 * ken_situps

/-- The number of sit-ups Bob can do -/
def bob_situps : ℕ := (ken_situps + nathan_situps) / 2

/-- The number of sit-ups Emma can do -/
def emma_situps : ℕ := bob_situps / 3

/-- The theorem stating the difference in sit-ups between the group (Nathan, Bob, Emma) and Ken -/
theorem situp_difference : nathan_situps + bob_situps + emma_situps - ken_situps = 60 := by
  sorry

end NUMINAMATH_CALUDE_situp_difference_l2688_268895


namespace NUMINAMATH_CALUDE_circle_equation_l2688_268857

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

def is_tangent_to (c : Circle) (l : Line) : Prop :=
  let (cx, cy) := c.center
  |l.a * cx + l.b * cy + l.c| / Real.sqrt (l.a^2 + l.b^2) = c.radius

def center_on_line (c : Circle) (l : Line) : Prop :=
  let (cx, cy) := c.center
  l.a * cx + l.b * cy + l.c = 0

-- Theorem statement
theorem circle_equation (c : Circle) :
  passes_through c (0, -1) ∧
  is_tangent_to c { a := 1, b := 1, c := -1 } ∧
  center_on_line c { a := 2, b := 1, c := 0 } →
  ((∀ x y, (x - 1)^2 + (y + 2)^2 = 2 ↔ passes_through c (x, y)) ∨
   (∀ x y, (x - 1/9)^2 + (y + 2/9)^2 = 50/81 ↔ passes_through c (x, y))) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2688_268857


namespace NUMINAMATH_CALUDE_earthwork_transport_theorem_prove_earthwork_transport_l2688_268868

/-- Represents the capacity of earthwork transport vehicles -/
structure VehicleCapacity where
  large : ℕ
  small : ℕ

/-- Represents a dispatch plan for earthwork transport vehicles -/
structure DispatchPlan where
  large : ℕ
  small : ℕ

/-- Theorem stating the correct vehicle capacities and possible dispatch plans -/
theorem earthwork_transport_theorem 
  (capacity : VehicleCapacity)
  (plans : List DispatchPlan) : Prop :=
  -- Conditions
  (3 * capacity.large + 4 * capacity.small = 44) ∧
  (4 * capacity.large + 6 * capacity.small = 62) ∧
  (∀ plan ∈ plans, 
    plan.large + plan.small = 12 ∧
    plan.small ≥ 4 ∧
    plan.large * capacity.large + plan.small * capacity.small ≥ 78) ∧
  -- Conclusions
  (capacity.large = 8 ∧ capacity.small = 5) ∧
  (plans = [
    DispatchPlan.mk 8 4,
    DispatchPlan.mk 7 5,
    DispatchPlan.mk 6 6
  ])

/-- Proof of the earthwork transport theorem -/
theorem prove_earthwork_transport : 
  ∃ (capacity : VehicleCapacity) (plans : List DispatchPlan),
    earthwork_transport_theorem capacity plans := by
  sorry

end NUMINAMATH_CALUDE_earthwork_transport_theorem_prove_earthwork_transport_l2688_268868


namespace NUMINAMATH_CALUDE_nested_sum_equals_geometric_sum_l2688_268809

def nested_sum : ℕ → ℕ
  | 0 => 5
  | n + 1 => 5 * (1 + nested_sum n)

theorem nested_sum_equals_geometric_sum : nested_sum 11 = 305175780 := by
  sorry

end NUMINAMATH_CALUDE_nested_sum_equals_geometric_sum_l2688_268809


namespace NUMINAMATH_CALUDE_surface_area_difference_l2688_268858

theorem surface_area_difference (large_cube_volume : ℝ) (small_cube_volume : ℝ) (num_small_cubes : ℕ) :
  large_cube_volume = 64 →
  small_cube_volume = 1 →
  num_small_cubes = 64 →
  (num_small_cubes : ℝ) * (6 * small_cube_volume ^ (2/3)) - (6 * large_cube_volume ^ (2/3)) = 288 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_difference_l2688_268858


namespace NUMINAMATH_CALUDE_right_triangle_area_l2688_268861

theorem right_triangle_area (DF EF : ℝ) (angle_DEF : ℝ) :
  DF = 4 →
  angle_DEF = π / 4 →
  DF = EF →
  (1 / 2) * DF * EF = 8 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2688_268861


namespace NUMINAMATH_CALUDE_sector_central_angle_l2688_268806

/-- Given a sector with perimeter 8 and area 4, its central angle is 2 radians -/
theorem sector_central_angle (l r : ℝ) (h1 : 2 * r + l = 8) (h2 : (1 / 2) * l * r = 4) :
  l / r = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2688_268806


namespace NUMINAMATH_CALUDE_incorrect_expression_for_repeating_decimal_l2688_268822

/-- Represents a repeating decimal number -/
structure RepeatingDecimal where
  nonRepeating : ℕ → ℕ  -- X: non-repeating part
  repeating : ℕ → ℕ     -- Y: repeating part
  a : ℕ                 -- length of non-repeating part
  b : ℕ                 -- length of repeating part

/-- Converts a RepeatingDecimal to a real number -/
def toReal (v : RepeatingDecimal) : ℝ :=
  sorry

/-- Theorem stating that the given expression is incorrect for repeating decimals -/
theorem incorrect_expression_for_repeating_decimal (v : RepeatingDecimal) :
  ∃ (x y : ℕ), 10^v.a * (10^v.b - 1) * (toReal v) ≠ x * (y - 1) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_for_repeating_decimal_l2688_268822


namespace NUMINAMATH_CALUDE_correct_missile_sampling_l2688_268845

/-- Represents a systematic sampling of missiles -/
structure MissileSampling where
  total : ℕ
  sample_size : ℕ
  first : ℕ
  interval : ℕ

/-- Generates the sequence of sampled missile numbers -/
def generate_sequence (ms : MissileSampling) : List ℕ :=
  List.range ms.sample_size |>.map (λ i => ms.first + i * ms.interval)

/-- Checks if all elements in the list are within the valid range -/
def valid_range (l : List ℕ) (max : ℕ) : Prop :=
  l.all (λ x => x > 0 ∧ x ≤ max)

theorem correct_missile_sampling :
  let ms : MissileSampling := {
    total := 60,
    sample_size := 6,
    first := 3,
    interval := 10
  }
  let sequence := generate_sequence ms
  (sequence = [3, 13, 23, 33, 43, 53]) ∧
  (ms.interval = ms.total / ms.sample_size) ∧
  (valid_range sequence ms.total) :=
by sorry

end NUMINAMATH_CALUDE_correct_missile_sampling_l2688_268845


namespace NUMINAMATH_CALUDE_trajectory_of_tangent_circles_l2688_268833

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 25
def C2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define the trajectory equation
def trajectory_equation (x y : ℝ) : Prop := y^2 / 9 + x^2 / 5 = 1 ∧ y ≠ 3

-- Theorem statement
theorem trajectory_of_tangent_circles :
  ∀ x y : ℝ, 
  (∃ r : ℝ, (x - 0)^2 + (y - (-1))^2 = (5 - r)^2 ∧ (x - 0)^2 + (y - 2)^2 = (r + 1)^2) →
  trajectory_equation x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_tangent_circles_l2688_268833


namespace NUMINAMATH_CALUDE_candidates_per_state_l2688_268851

theorem candidates_per_state : 
  ∀ x : ℝ,
  (x * 0.07 = x * 0.06 + 79) →
  x = 7900 := by
sorry

end NUMINAMATH_CALUDE_candidates_per_state_l2688_268851


namespace NUMINAMATH_CALUDE_rice_qualification_condition_l2688_268829

/-- The maximum number of chaff grains allowed in a qualified rice sample -/
def max_chaff_grains : ℕ := 7

/-- The total number of grains in the rice sample -/
def total_grains : ℕ := 235

/-- The maximum allowed percentage of chaff for qualified rice -/
def max_chaff_percentage : ℚ := 3 / 100

/-- Theorem stating the condition for qualified rice -/
theorem rice_qualification_condition (n : ℕ) :
  (n : ℚ) / total_grains ≤ max_chaff_percentage ↔ n ≤ max_chaff_grains :=
by sorry

end NUMINAMATH_CALUDE_rice_qualification_condition_l2688_268829


namespace NUMINAMATH_CALUDE_arithmetic_proof_l2688_268899

theorem arithmetic_proof : (100 + 20 / 90) * 90 = 120 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l2688_268899


namespace NUMINAMATH_CALUDE_ceiling_product_equation_l2688_268841

theorem ceiling_product_equation : ∃ x : ℝ, ⌈x⌉ * x = 210 ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_l2688_268841


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2688_268846

theorem fraction_multiplication (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (5*x + 5*y) / ((5*x) * (5*y)) = (1 / 5) * ((x + y) / (x * y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2688_268846


namespace NUMINAMATH_CALUDE_lawn_mowing_theorem_l2688_268830

/-- Represents the time (in hours) it takes to mow the entire lawn -/
def MaryTime : ℚ := 4
def TomTime : ℚ := 5

/-- Represents the fraction of the lawn mowed per hour -/
def MaryRate : ℚ := 1 / MaryTime
def TomRate : ℚ := 1 / TomTime

/-- Represents the time Tom works alone -/
def TomAloneTime : ℚ := 3

/-- Represents the time Mary and Tom work together -/
def TogetherTime : ℚ := 1

/-- The fraction of lawn remaining to be mowed -/
def RemainingFraction : ℚ := 1 / 20

theorem lawn_mowing_theorem :
  1 - (TomRate * TomAloneTime + (MaryRate + TomRate) * TogetherTime) = RemainingFraction := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_theorem_l2688_268830


namespace NUMINAMATH_CALUDE_log_absolute_equality_l2688_268862

/-- Given a function f(x) = |log x|, prove that if 0 < a < b and f(a) = f(b), then ab = 1 -/
theorem log_absolute_equality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) 
  (h3 : |Real.log a| = |Real.log b|) : a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_absolute_equality_l2688_268862


namespace NUMINAMATH_CALUDE_polynomial_equation_sum_l2688_268885

theorem polynomial_equation_sum (a b : ℤ) : 
  (∀ x : ℝ, 2 * x^3 - a * x^2 - 5 * x + 5 = (2 * x^2 + a * x - 1) * (x - b) + 3) → 
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equation_sum_l2688_268885


namespace NUMINAMATH_CALUDE_subtraction_problem_l2688_268872

theorem subtraction_problem : 2000000000000 - 1111111111111 = 888888888889 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l2688_268872


namespace NUMINAMATH_CALUDE_sequence_characterization_l2688_268890

theorem sequence_characterization (a : ℕ → ℕ) :
  (∀ n : ℕ, n ≥ 1 → a (n + 2) * (a (n + 1) - 1) = a n * (a (n + 1) + 1)) →
  ∃ k : ℕ, ∀ n : ℕ, n ≥ 1 → a n = k + n :=
sorry

end NUMINAMATH_CALUDE_sequence_characterization_l2688_268890


namespace NUMINAMATH_CALUDE_train_speed_l2688_268866

/-- The speed of a train traveling between two points, given the conditions of the problem -/
theorem train_speed (distance : ℝ) (return_speed : ℝ) (time_difference : ℝ) :
  distance = 480 ∧ 
  return_speed = 120 ∧ 
  time_difference = 1 →
  ∃ speed : ℝ, 
    speed = 160 ∧ 
    distance / speed + time_difference = distance / return_speed :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2688_268866


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l2688_268812

theorem set_equality_implies_sum (a b : ℝ) : 
  ({0, b, b/a} : Set ℝ) = {1, a, a+b} → a + 2*b = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l2688_268812


namespace NUMINAMATH_CALUDE_history_paper_pages_l2688_268836

/-- Given a paper due in 6 days with a required writing pace of 11 pages per day,
    the total number of pages in the paper is 66. -/
theorem history_paper_pages (days : ℕ) (pages_per_day : ℕ) (h1 : days = 6) (h2 : pages_per_day = 11) :
  days * pages_per_day = 66 := by
  sorry

end NUMINAMATH_CALUDE_history_paper_pages_l2688_268836


namespace NUMINAMATH_CALUDE_complex_square_root_l2688_268828

theorem complex_square_root (z : ℂ) : 
  z^2 = -100 - 48*I ↔ z = 2 - 12*I ∨ z = -2 + 12*I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_root_l2688_268828


namespace NUMINAMATH_CALUDE_total_stuffed_animals_l2688_268856

theorem total_stuffed_animals (mckenna kenley tenly : ℕ) : 
  mckenna = 34 → 
  kenley = 2 * mckenna → 
  tenly = kenley + 5 → 
  mckenna + kenley + tenly = 175 :=
by
  sorry

end NUMINAMATH_CALUDE_total_stuffed_animals_l2688_268856


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2688_268814

theorem sufficient_but_not_necessary (a : ℝ) :
  (∀ a ≥ 0, ∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ∧
  (∃ a < 0, ∃ x : ℝ, a * x^2 + x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2688_268814


namespace NUMINAMATH_CALUDE_dante_remaining_coconuts_l2688_268840

/-- Paolo's number of coconuts -/
def paolo_coconuts : ℕ := 14

/-- Dante's initial number of coconuts in terms of Paolo's -/
def dante_initial_coconuts : ℕ := 3 * paolo_coconuts

/-- Number of coconuts Dante sold -/
def dante_sold_coconuts : ℕ := 10

/-- Theorem: Dante has 32 coconuts left after selling -/
theorem dante_remaining_coconuts : 
  dante_initial_coconuts - dante_sold_coconuts = 32 := by sorry

end NUMINAMATH_CALUDE_dante_remaining_coconuts_l2688_268840


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2688_268859

/-- Given a line mx + y + √3 = 0 intersecting a circle (x+1)² + y² = 2 with a chord length of 2,
    prove that m = √3/3 -/
theorem line_circle_intersection (m : ℝ) : 
  (∃ x y : ℝ, mx + y + Real.sqrt 3 = 0 ∧ (x + 1)^2 + y^2 = 2) → 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    mx₁ + y₁ + Real.sqrt 3 = 0 ∧ (x₁ + 1)^2 + y₁^2 = 2 ∧
    mx₂ + y₂ + Real.sqrt 3 = 0 ∧ (x₂ + 1)^2 + y₂^2 = 2 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4) →
  m = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2688_268859


namespace NUMINAMATH_CALUDE_solve_equation_l2688_268820

theorem solve_equation (n m q : ℚ) : 
  (7 / 8 : ℚ) = n / 96 ∧ 
  (7 / 8 : ℚ) = (m + n) / 112 ∧ 
  (7 / 8 : ℚ) = (q - m) / 144 → 
  q = 140 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2688_268820


namespace NUMINAMATH_CALUDE_system_solution_l2688_268883

theorem system_solution :
  ∃ (x y z : ℚ),
    (4 * x - 6 * y + 2 * z = -14) ∧
    (8 * x + 3 * y - z = -15) ∧
    (3 * x + z = 7) ∧
    (x = 100 / 33) ∧
    (y = 146 / 33) ∧
    (z = 29 / 11) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2688_268883


namespace NUMINAMATH_CALUDE_abs_minus_one_lt_two_iff_product_lt_zero_l2688_268892

theorem abs_minus_one_lt_two_iff_product_lt_zero (x : ℝ) :
  |x - 1| < 2 ↔ (x + 1) * (x - 3) < 0 := by sorry

end NUMINAMATH_CALUDE_abs_minus_one_lt_two_iff_product_lt_zero_l2688_268892


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2688_268865

theorem quadratic_equation_properties (a b : ℝ) (h1 : a > 0) 
  (h2 : ∃! x y : ℝ, x ≠ y ∧ x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0) :
  (a^2 - b^2 ≤ 4) ∧ 
  (a^2 + 1/b ≥ 4) ∧
  (∀ c x₁ x₂ : ℝ, (x₁^2 + a*x₁ + b < c ∧ x₂^2 + a*x₂ + b < c ∧ |x₁ - x₂| = 4) → c = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2688_268865


namespace NUMINAMATH_CALUDE_Michael_birth_year_l2688_268827

def IMO_start_year : ℕ := 1959

def Michael_age_at_10th_IMO : ℕ := 15

def IMO_held_annually : Prop := ∀ n : ℕ, n ≥ IMO_start_year → ∃ m : ℕ, m = n - IMO_start_year + 1

theorem Michael_birth_year :
  IMO_held_annually →
  ∃ year : ℕ, year = IMO_start_year + 9 - Michael_age_at_10th_IMO ∧ year = 1953 :=
by sorry

end NUMINAMATH_CALUDE_Michael_birth_year_l2688_268827


namespace NUMINAMATH_CALUDE_cricketer_average_score_l2688_268839

theorem cricketer_average_score 
  (total_matches : Nat) 
  (matches_with_known_average : Nat) 
  (known_average : ℝ) 
  (total_average : ℝ) 
  (h1 : total_matches = 5)
  (h2 : matches_with_known_average = 3)
  (h3 : known_average = 10)
  (h4 : total_average = 22) :
  let remaining_matches := total_matches - matches_with_known_average
  let remaining_average := (total_matches * total_average - matches_with_known_average * known_average) / remaining_matches
  remaining_average = 40 := by
sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l2688_268839


namespace NUMINAMATH_CALUDE_runs_by_running_percentage_l2688_268884

def total_runs : ℕ := 120
def num_boundaries : ℕ := 3
def num_sixes : ℕ := 8
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

theorem runs_by_running_percentage : 
  (total_runs - (num_boundaries * runs_per_boundary + num_sixes * runs_per_six)) / total_runs * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_runs_by_running_percentage_l2688_268884


namespace NUMINAMATH_CALUDE_jane_drawing_paper_l2688_268824

/-- The number of old, brown sheets of drawing paper Jane has. -/
def brown_sheets : ℕ := 28

/-- The number of old, yellow sheets of drawing paper Jane has. -/
def yellow_sheets : ℕ := 27

/-- The total number of sheets of drawing paper Jane has. -/
def total_sheets : ℕ := brown_sheets + yellow_sheets

theorem jane_drawing_paper :
  total_sheets = 55 := by sorry

end NUMINAMATH_CALUDE_jane_drawing_paper_l2688_268824


namespace NUMINAMATH_CALUDE_frood_game_theorem_l2688_268882

/-- Score for dropping n froods -/
def drop_score (n : ℕ) : ℕ := n * (n + 1)

/-- Score for eating n froods -/
def eat_score (n : ℕ) : ℕ := 8 * n

/-- The least number of froods for which dropping them earns more points than eating them -/
def least_frood_number : ℕ := 8

theorem frood_game_theorem :
  least_frood_number = 8 ∧
  ∀ n : ℕ, n < least_frood_number → drop_score n ≤ eat_score n ∧
  drop_score least_frood_number > eat_score least_frood_number :=
by sorry

end NUMINAMATH_CALUDE_frood_game_theorem_l2688_268882


namespace NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l2688_268893

noncomputable section

def a : ℝ := Real.log 2 / 2
def b : ℝ := Real.log 3 / 3
def c : ℝ := Real.log Real.pi / Real.pi
def d : ℝ := Real.log 2.72 / 2.72
def f : ℝ := (Real.sqrt 10 * Real.log 10) / 20

theorem order_of_logarithmic_expressions :
  a < f ∧ f < c ∧ c < b ∧ b < d := by sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l2688_268893


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2688_268818

theorem complex_fraction_equality (z : ℂ) (h : z = 1 + I) :
  (3 * I) / (z + 1) = 3/5 + 6/5 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2688_268818


namespace NUMINAMATH_CALUDE_margos_walking_distance_l2688_268864

/-- Margo's Walking Problem -/
theorem margos_walking_distance
  (outbound_time : Real) (return_time : Real)
  (outbound_speed : Real) (return_speed : Real)
  (average_speed : Real)
  (h1 : outbound_time = 15 / 60)
  (h2 : return_time = 30 / 60)
  (h3 : outbound_speed = 5)
  (h4 : return_speed = 3)
  (h5 : average_speed = 3.6)
  (h6 : average_speed = (outbound_time + return_time) / 
        ((outbound_time / outbound_speed) + (return_time / return_speed))) :
  outbound_speed * outbound_time + return_speed * return_time = 2.75 := by
  sorry

#check margos_walking_distance

end NUMINAMATH_CALUDE_margos_walking_distance_l2688_268864


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_sum_l2688_268815

/-- The quadratic function f(x) = 4x^2 - 8x + 6 -/
def f (x : ℝ) : ℝ := 4 * x^2 - 8 * x + 6

/-- The vertex form of the quadratic function -/
def vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

theorem quadratic_vertex_form_sum :
  ∃ (a h k : ℝ), (∀ x, f x = vertex_form a h k x) ∧ (a + h + k = 7) := by
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_sum_l2688_268815


namespace NUMINAMATH_CALUDE_ellipse_problem_l2688_268879

-- Define the ellipse C
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line (x y k : ℝ) : Prop := y = k * (x - 1)

-- Define the theorem
theorem ellipse_problem (a b c k : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : k > 0) :
  -- Condition 2: C passes through Q(√2, 1)
  ellipse (Real.sqrt 2) 1 a b →
  -- Condition 3: Right focus at F(√2, 0)
  c = Real.sqrt 2 →
  a^2 - b^2 = c^2 →
  -- Condition 6: CN = MD (implicitly used in the solution)
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ a b ∧ 
    ellipse x₂ y₂ a b ∧
    line x₁ y₁ k ∧ 
    line x₂ y₂ k ∧
    x₂ - 1 = -x₁ ∧ 
    y₂ = -k - y₁) →
  -- Conclusion I: Equation of ellipse C
  (a = 2 ∧ b = Real.sqrt 2) ∧
  -- Conclusion II: Value of k and length of MN
  (k = Real.sqrt 2 / 2 ∧ 
   ∃ x₁ x₂ : ℝ, 
     ellipse x₁ (k * (x₁ - 1)) 2 (Real.sqrt 2) ∧
     ellipse x₂ (k * (x₂ - 1)) 2 (Real.sqrt 2) ∧
     Real.sqrt ((x₂ - x₁)^2 + (k * (x₂ - x₁))^2) = Real.sqrt 42 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_problem_l2688_268879


namespace NUMINAMATH_CALUDE_variance_mean_preserved_l2688_268819

def initial_set : List Int := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

def mean (xs : List Int) : ℚ := (xs.sum : ℚ) / xs.length

def variance (xs : List Int) : ℚ :=
  let m := mean xs
  (xs.map (fun x => ((x : ℚ) - m) ^ 2)).sum / xs.length

def replace_4_with_neg1_and_5 (xs : List Int) : List Int :=
  xs.filter (· ≠ 4) ++ [-1, 5]

def replace_neg4_with_1_and_neg5 (xs : List Int) : List Int :=
  xs.filter (· ≠ -4) ++ [1, -5]

theorem variance_mean_preserved :
  (mean initial_set = mean (replace_4_with_neg1_and_5 initial_set) ∧
   variance initial_set = variance (replace_4_with_neg1_and_5 initial_set)) ∨
  (mean initial_set = mean (replace_neg4_with_1_and_neg5 initial_set) ∧
   variance initial_set = variance (replace_neg4_with_1_and_neg5 initial_set)) :=
by sorry

end NUMINAMATH_CALUDE_variance_mean_preserved_l2688_268819


namespace NUMINAMATH_CALUDE_number_of_values_l2688_268854

theorem number_of_values (initial_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) (correct_mean : ℚ) : 
  initial_mean = 250 →
  incorrect_value = 135 →
  correct_value = 165 →
  correct_mean = 251 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℚ) * initial_mean + correct_value - incorrect_value = (n : ℚ) * correct_mean ∧
    n = 30 :=
by sorry

end NUMINAMATH_CALUDE_number_of_values_l2688_268854


namespace NUMINAMATH_CALUDE_borrowing_ten_sheets_avg_49_l2688_268821

/-- Represents a notebook with a given number of sheets --/
structure Notebook where
  sheets : ℕ
  pages : ℕ
  h_pages : pages = 2 * sheets

/-- Represents a borrowing of consecutive sheets from a notebook --/
structure Borrowing where
  notebook : Notebook
  borrowed_sheets : ℕ
  start_sheet : ℕ
  h_consecutive : start_sheet + borrowed_sheets ≤ notebook.sheets

/-- Calculates the average page number of remaining sheets --/
def average_remaining_pages (b : Borrowing) : ℚ :=
  let total_pages := b.notebook.pages
  let borrowed_pages := 2 * b.borrowed_sheets
  let remaining_pages := total_pages - borrowed_pages
  let sum_before := b.start_sheet * (2 * b.start_sheet + 1)
  let sum_after := ((total_pages - 2 * (b.start_sheet + b.borrowed_sheets) + 1) * 
                    (2 * (b.start_sheet + b.borrowed_sheets) + total_pages)) / 2
  (sum_before + sum_after) / remaining_pages

/-- Theorem stating that borrowing 10 sheets results in an average of 49 for remaining pages --/
theorem borrowing_ten_sheets_avg_49 (n : Notebook) (h_100_pages : n.pages = 100) :
  ∃ b : Borrowing, b.notebook = n ∧ b.borrowed_sheets = 10 ∧ average_remaining_pages b = 49 := by
  sorry


end NUMINAMATH_CALUDE_borrowing_ten_sheets_avg_49_l2688_268821


namespace NUMINAMATH_CALUDE_choose_15_4_l2688_268807

theorem choose_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_choose_15_4_l2688_268807


namespace NUMINAMATH_CALUDE_hulk_jump_exceeds_1500_l2688_268881

def hulk_jump (n : ℕ) : ℝ := 2 * (3 : ℝ) ^ (n - 1)

theorem hulk_jump_exceeds_1500 :
  ∀ k < 8, hulk_jump k ≤ 1500 ∧ hulk_jump 8 > 1500 := by sorry

end NUMINAMATH_CALUDE_hulk_jump_exceeds_1500_l2688_268881


namespace NUMINAMATH_CALUDE_x_one_minus_f_equals_four_power_500_l2688_268843

/-- Given x = (3 + √5)^500, n = ⌊x⌋, and f = x - n, prove that x(1 - f) = 4^500 -/
theorem x_one_minus_f_equals_four_power_500 :
  let x : ℝ := (3 + Real.sqrt 5) ^ 500
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 4 ^ 500 := by
  sorry

end NUMINAMATH_CALUDE_x_one_minus_f_equals_four_power_500_l2688_268843


namespace NUMINAMATH_CALUDE_nested_inverse_expression_l2688_268847

theorem nested_inverse_expression : ((((3+2)⁻¹ - 1)⁻¹ - 1)⁻¹ - 1 : ℚ) = -13/9 := by
  sorry

end NUMINAMATH_CALUDE_nested_inverse_expression_l2688_268847


namespace NUMINAMATH_CALUDE_miss_two_consecutive_probability_l2688_268810

/-- The probability of hitting a target in one shot. -/
def hit_probability : ℝ := 0.8

/-- The probability of missing a target in one shot. -/
def miss_probability : ℝ := 1 - hit_probability

/-- The probability of missing a target in two consecutive shots. -/
def miss_two_consecutive : ℝ := miss_probability * miss_probability

theorem miss_two_consecutive_probability :
  miss_two_consecutive = 0.04 := by sorry

end NUMINAMATH_CALUDE_miss_two_consecutive_probability_l2688_268810


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2688_268804

theorem negation_of_universal_proposition 
  (f : ℝ → ℝ) (m : ℝ) : 
  (¬ ∀ x, f x ≥ m) ↔ (∃ x, f x < m) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2688_268804


namespace NUMINAMATH_CALUDE_puppies_sold_l2688_268842

theorem puppies_sold (initial_puppies : ℕ) (puppies_per_cage : ℕ) (cages_used : ℕ) : 
  initial_puppies = 13 → puppies_per_cage = 2 → cages_used = 3 →
  initial_puppies - (puppies_per_cage * cages_used) = 7 := by
  sorry

end NUMINAMATH_CALUDE_puppies_sold_l2688_268842


namespace NUMINAMATH_CALUDE_eulers_formula_modulus_l2688_268853

theorem eulers_formula_modulus (i : ℂ) (π : ℝ) : 
  Complex.abs (Complex.exp (i * π / 3)) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_modulus_l2688_268853


namespace NUMINAMATH_CALUDE_part_I_part_II_l2688_268837

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}

-- Part I
theorem part_I (a : ℝ) (h : a = 3) :
  (A ∪ B a = {x | 1 ≤ x ∧ x ≤ 5}) ∧
  (B a ∩ (Set.univ \ A) = {x | 4 < x ∧ x ≤ 5}) := by
  sorry

-- Part II
theorem part_II (a : ℝ) :
  B a ⊆ A ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_part_I_part_II_l2688_268837


namespace NUMINAMATH_CALUDE_count_valid_digits_l2688_268849

def is_valid_digit (n : ℕ) : Prop :=
  n < 10

def appended_number (n : ℕ) : ℕ :=
  7580 + n

theorem count_valid_digits :
  ∃ (valid_digits : Finset ℕ),
    (∀ d ∈ valid_digits, is_valid_digit d ∧ (appended_number d).mod 4 = 0) ∧
    (∀ d, is_valid_digit d ∧ (appended_number d).mod 4 = 0 → d ∈ valid_digits) ∧
    valid_digits.card = 3 :=
by sorry

end NUMINAMATH_CALUDE_count_valid_digits_l2688_268849


namespace NUMINAMATH_CALUDE_andrew_work_hours_l2688_268844

theorem andrew_work_hours : 
  let day1 : ℝ := 1.5
  let day2 : ℝ := 2.75
  let day3 : ℝ := 3.25
  day1 + day2 + day3 = 7.5 := by
sorry

end NUMINAMATH_CALUDE_andrew_work_hours_l2688_268844


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2688_268894

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  sum_property : a 1 + a 3 = 8
  geometric_mean : a 4 ^ 2 = a 2 * a 9
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term (seq : ArithmeticSequence) : seq.a 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2688_268894


namespace NUMINAMATH_CALUDE_base_2_representation_of_236_l2688_268878

theorem base_2_representation_of_236 :
  ∃ (a : List Bool),
    a.length = 9 ∧
    a = [true, true, true, false, true, false, true, false, false] ∧
    (a.foldr (λ (b : Bool) (acc : Nat) => 2 * acc + if b then 1 else 0) 0) = 236 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_236_l2688_268878
