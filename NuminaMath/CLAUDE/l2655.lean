import Mathlib

namespace NUMINAMATH_CALUDE_power_equation_solution_l2655_265550

theorem power_equation_solution : ∃ x : ℝ, (5^5 * 9^3 : ℝ) = 3 * 15^x ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2655_265550


namespace NUMINAMATH_CALUDE_twenty_sided_polygon_selection_l2655_265500

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : Fin n → Set (Fin n × Fin n)
  convex : sorry -- Additional property to ensure convexity

/-- The condition that two sides have at least k sides between them -/
def HasKSidesBetween (n : ℕ) (k : ℕ) (s₁ s₂ : Fin n × Fin n) : Prop :=
  sorry

/-- The number of ways to choose m sides from an n-sided polygon with k sides between each pair -/
def CountValidSelections (n m k : ℕ) : ℕ :=
  sorry

theorem twenty_sided_polygon_selection :
  CountValidSelections 20 3 2 = 520 :=
sorry

end NUMINAMATH_CALUDE_twenty_sided_polygon_selection_l2655_265500


namespace NUMINAMATH_CALUDE_prime_abs_nsquared_minus_6n_minus_27_l2655_265537

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem prime_abs_nsquared_minus_6n_minus_27 (n : ℤ) :
  is_prime (Int.natAbs (n^2 - 6*n - 27)) ↔ n = -4 ∨ n = -2 ∨ n = 8 ∨ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_prime_abs_nsquared_minus_6n_minus_27_l2655_265537


namespace NUMINAMATH_CALUDE_max_value_xyz_l2655_265591

theorem max_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  x^4 * y^3 * z^2 ≤ 1024/14348907 :=
sorry

end NUMINAMATH_CALUDE_max_value_xyz_l2655_265591


namespace NUMINAMATH_CALUDE_equation_roots_l2655_265563

theorem equation_roots : ∀ x : ℝ, 3 * (x - 3) = (x - 3)^2 ↔ x = 3 ∨ x = 6 := by sorry

end NUMINAMATH_CALUDE_equation_roots_l2655_265563


namespace NUMINAMATH_CALUDE_two_year_increase_l2655_265562

/-- Calculates the final amount after a given number of years with a fixed annual increase rate. -/
def finalAmount (initialValue : ℝ) (increaseRate : ℝ) (years : ℕ) : ℝ :=
  initialValue * (1 + increaseRate) ^ years

/-- Theorem stating that an initial amount of 59,000, increasing by 1/8 of itself annually, 
    will result in 74,671.875 after 2 years. -/
theorem two_year_increase : 
  let initialValue : ℝ := 59000
  let increaseRate : ℝ := 1/8
  let years : ℕ := 2
  finalAmount initialValue increaseRate years = 74671.875 := by
sorry

end NUMINAMATH_CALUDE_two_year_increase_l2655_265562


namespace NUMINAMATH_CALUDE_exists_point_sum_distances_gt_perimeter_l2655_265509

/-- A convex n-gon in a 2D plane -/
structure ConvexNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_convex : sorry -- Axiom for convexity

/-- The perimeter of a convex n-gon -/
def perimeter (polygon : ConvexNGon n) : ℝ := sorry

/-- The sum of distances from a point to all vertices of a convex n-gon -/
def sum_distances (polygon : ConvexNGon n) (point : ℝ × ℝ) : ℝ := sorry

/-- For any convex n-gon with n ≥ 7, there exists a point inside the n-gon
    such that the sum of distances from this point to all vertices
    is greater than the perimeter of the n-gon -/
theorem exists_point_sum_distances_gt_perimeter (n : ℕ) (h : n ≥ 7) (polygon : ConvexNGon n) :
  ∃ (point : ℝ × ℝ), sum_distances polygon point > perimeter polygon := by
  sorry

end NUMINAMATH_CALUDE_exists_point_sum_distances_gt_perimeter_l2655_265509


namespace NUMINAMATH_CALUDE_smallest_multiple_of_2_3_5_l2655_265526

theorem smallest_multiple_of_2_3_5 : ∀ n : ℕ, n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n → n ≥ 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_2_3_5_l2655_265526


namespace NUMINAMATH_CALUDE_opposite_sides_line_range_l2655_265584

/-- Given that the points (-2, 1) and (1, 1) are on opposite sides of the line 3x-2y-a=0,
    prove that the range of values for a is -8 < a < 1 -/
theorem opposite_sides_line_range (a : ℝ) : 
  (∀ (x y : ℝ), 3*x - 2*y - a = 0 → 
    ((3*(-2) - 2*1 - a) * (3*1 - 2*1 - a) < 0)) → 
  -8 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_line_range_l2655_265584


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_1_simplify_and_evaluate_2_l2655_265516

-- Part 1
theorem simplify_and_evaluate_1 (x : ℝ) (h : x = 3) :
  3 * x^2 - (5 * x - (6 * x - 4) - 2 * x^2) = 44 := by sorry

-- Part 2
theorem simplify_and_evaluate_2 (m n : ℝ) (h1 : m = -1) (h2 : n = 2) :
  (8 * m * n - 3 * m^2) - 5 * m * n - 2 * (3 * m * n - 2 * m^2) = 7 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_1_simplify_and_evaluate_2_l2655_265516


namespace NUMINAMATH_CALUDE_days_took_capsules_l2655_265508

/-- The number of days in July -/
def days_in_july : ℕ := 31

/-- The number of days Isla forgot to take capsules -/
def days_forgot : ℕ := 2

/-- Theorem: The number of days Isla took capsules in July is 29 -/
theorem days_took_capsules : days_in_july - days_forgot = 29 := by
  sorry

end NUMINAMATH_CALUDE_days_took_capsules_l2655_265508


namespace NUMINAMATH_CALUDE_bridge_building_time_l2655_265595

theorem bridge_building_time
  (workers₁ workers₂ : ℕ)
  (days₁ : ℕ)
  (h_workers₁ : workers₁ = 60)
  (h_workers₂ : workers₂ = 30)
  (h_days₁ : days₁ = 6)
  (h_positive : workers₁ > 0 ∧ workers₂ > 0 ∧ days₁ > 0)
  (h_same_rate : ∀ w : ℕ, w > 0 → ∃ r : ℚ, r > 0 ∧ w * r * days₁ = 1) :
  ∃ days₂ : ℕ, days₂ = 12 ∧ workers₂ * days₂ = workers₁ * days₁ :=
by sorry


end NUMINAMATH_CALUDE_bridge_building_time_l2655_265595


namespace NUMINAMATH_CALUDE_sum_equality_l2655_265557

theorem sum_equality (a b c d : ℝ) 
  (hab : a + b = 4)
  (hbc : b + c = 5)
  (had : a + d = 2) :
  c + d = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_equality_l2655_265557


namespace NUMINAMATH_CALUDE_store_purchase_count_l2655_265598

def num_cookie_flavors : ℕ := 6
def num_milk_flavors : ℕ := 4

def gamma_purchase_options (n : ℕ) : ℕ :=
  Nat.choose (num_cookie_flavors + num_milk_flavors) n

def delta_cookie_options (n : ℕ) : ℕ :=
  if n = 1 then
    num_cookie_flavors
  else if n = 2 then
    Nat.choose num_cookie_flavors 2 + num_cookie_flavors
  else if n = 3 then
    Nat.choose num_cookie_flavors 3 + num_cookie_flavors * (num_cookie_flavors - 1) + num_cookie_flavors
  else
    0

def total_purchase_options : ℕ :=
  gamma_purchase_options 3 +
  gamma_purchase_options 2 * delta_cookie_options 1 +
  gamma_purchase_options 1 * delta_cookie_options 2 +
  delta_cookie_options 3

theorem store_purchase_count : total_purchase_options = 656 := by
  sorry

end NUMINAMATH_CALUDE_store_purchase_count_l2655_265598


namespace NUMINAMATH_CALUDE_total_sections_after_admission_l2655_265572

/-- Proves that the total number of sections after admitting new students is 16 -/
theorem total_sections_after_admission (
  initial_students_per_section : ℕ) 
  (new_sections : ℕ)
  (students_per_section_after : ℕ)
  (new_students : ℕ)
  (h1 : initial_students_per_section = 24)
  (h2 : new_sections = 3)
  (h3 : students_per_section_after = 21)
  (h4 : new_students = 24) :
  ∃ (initial_sections : ℕ),
    (initial_sections + new_sections) * students_per_section_after = 
    initial_sections * initial_students_per_section + new_students ∧
    initial_sections + new_sections = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_sections_after_admission_l2655_265572


namespace NUMINAMATH_CALUDE_angle_sum_is_ninety_degrees_l2655_265532

theorem angle_sum_is_ninety_degrees (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β)) : 
  α + β = π/2 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_is_ninety_degrees_l2655_265532


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l2655_265506

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (x ^ 2 - 1) / (x - 1) = 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l2655_265506


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2655_265551

theorem infinite_series_sum : 
  (∑' n : ℕ, 1 / (n * (n + 3))) = 11 / 18 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2655_265551


namespace NUMINAMATH_CALUDE_number_of_roses_l2655_265531

theorem number_of_roses (total : ℕ) (rose_lily_diff : ℕ) (tulip_rose_diff : ℕ)
  (h1 : total = 100)
  (h2 : rose_lily_diff = 22)
  (h3 : tulip_rose_diff = 20) :
  ∃ (roses lilies tulips : ℕ),
    roses + lilies + tulips = total ∧
    roses = lilies + rose_lily_diff ∧
    tulips = roses + tulip_rose_diff ∧
    roses = 34 := by
  sorry

end NUMINAMATH_CALUDE_number_of_roses_l2655_265531


namespace NUMINAMATH_CALUDE_kirin_990_calculations_l2655_265564

-- Define the number of calculations per second
def calculations_per_second : ℝ := 10^11

-- Define the number of seconds
def seconds : ℝ := 2022

-- Theorem to prove
theorem kirin_990_calculations :
  calculations_per_second * seconds = 2.022 * 10^13 := by
  sorry

end NUMINAMATH_CALUDE_kirin_990_calculations_l2655_265564


namespace NUMINAMATH_CALUDE_opposite_silver_is_yellow_l2655_265573

-- Define the colors
inductive Color
| Blue | Yellow | Orange | Black | Silver | Violet

-- Define the faces of the cube
inductive Face
| Top | Bottom | Front | Back | Left | Right

-- Define the cube as a function from Face to Color
def Cube := Face → Color

-- Define the three views
def view1 (c : Cube) : Prop :=
  c Face.Top = Color.Blue ∧ c Face.Front = Color.Yellow ∧ c Face.Right = Color.Orange

def view2 (c : Cube) : Prop :=
  c Face.Top = Color.Blue ∧ c Face.Front = Color.Black ∧ c Face.Right = Color.Orange

def view3 (c : Cube) : Prop :=
  c Face.Top = Color.Blue ∧ c Face.Front = Color.Violet ∧ c Face.Right = Color.Orange

-- Define the theorem
theorem opposite_silver_is_yellow (c : Cube) :
  view1 c → view2 c → view3 c →
  (∃ f : Face, c f = Color.Silver) →
  (∃ f : Face, c f = Color.Yellow) →
  c Face.Front = Color.Yellow :=
by sorry

end NUMINAMATH_CALUDE_opposite_silver_is_yellow_l2655_265573


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l2655_265539

theorem absolute_value_simplification (a b : ℝ) (h1 : a < 0) (h2 : a * b < 0) :
  |b - a + 1| - |a - b - 5| = -4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l2655_265539


namespace NUMINAMATH_CALUDE_overlapping_squares_area_l2655_265549

/-- Given two squares with side length a that overlap such that one pair of vertices coincide,
    and the overlapping part forms a right triangle with an angle of 30°,
    the area of the non-overlapping part is 2(1 - √3/3)a². -/
theorem overlapping_squares_area (a : ℝ) (h : a > 0) :
  let overlap_angle : ℝ := 30 * π / 180
  let overlap_area : ℝ := a^2 * (Real.sin overlap_angle * Real.cos overlap_angle)
  let non_overlap_area : ℝ := 2 * (a^2 - overlap_area)
  non_overlap_area = 2 * (1 - Real.sqrt 3 / 3) * a^2 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_squares_area_l2655_265549


namespace NUMINAMATH_CALUDE_equation_solution_l2655_265580

theorem equation_solution : 
  ∃ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) ∧ x = 105 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2655_265580


namespace NUMINAMATH_CALUDE_triangle_properties_l2655_265567

/-- Given a triangle ABC with specific properties, prove that A = π/3 and AB = 2 -/
theorem triangle_properties (A B C : ℝ) (AB BC AC : ℝ) :
  (0 < A) ∧ (A < π) →
  (0 < B) ∧ (B < π) →
  (0 < C) ∧ (C < π) →
  A + B + C = π →
  2 * Real.sin B * Real.cos A = Real.sin (A + C) →
  BC = 2 →
  (1/2) * AB * AC * Real.sin A = Real.sqrt 3 →
  A = π/3 ∧ AB = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2655_265567


namespace NUMINAMATH_CALUDE_inequality_proof_l2655_265513

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^3 + b^3 + a + b ≥ 4*a*b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2655_265513


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2655_265507

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_sum : 1 / (a * b) + 1 / (b * c) + 1 / (c * d) + 1 / (d * a) = 1) :
  a * b * c * d + 16 ≥ 8 * Real.sqrt ((a + c) * (1 / a + 1 / c)) +
    8 * Real.sqrt ((b + d) * (1 / b + 1 / d)) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2655_265507


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2655_265561

/-- Given a geometric sequence of positive integers with first term 3 and fourth term 240,
    prove that the fifth term is 768. -/
theorem geometric_sequence_fifth_term :
  ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 3 →                            -- first term is 3
  a 4 = 240 →                          -- fourth term is 240
  a 5 = 768 :=                         -- conclusion: fifth term is 768
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2655_265561


namespace NUMINAMATH_CALUDE_xia_initial_stickers_l2655_265576

/-- The number of stickers Xia shared with her friends -/
def shared_stickers : ℕ := 100

/-- The number of sheets of stickers Xia had left after sharing -/
def remaining_sheets : ℕ := 5

/-- The number of stickers on each sheet -/
def stickers_per_sheet : ℕ := 10

/-- Xia's initial number of stickers -/
def initial_stickers : ℕ := shared_stickers + remaining_sheets * stickers_per_sheet

theorem xia_initial_stickers : initial_stickers = 150 := by
  sorry

end NUMINAMATH_CALUDE_xia_initial_stickers_l2655_265576


namespace NUMINAMATH_CALUDE_linear_function_increasing_l2655_265583

/-- Given a linear function f(x) = (k^2 + 1)x - 5, prove that f(-3) < f(4) for any real k -/
theorem linear_function_increasing (k : ℝ) :
  let f : ℝ → ℝ := λ x => (k^2 + 1) * x - 5
  f (-3) < f 4 := by sorry

end NUMINAMATH_CALUDE_linear_function_increasing_l2655_265583


namespace NUMINAMATH_CALUDE_bag_problem_l2655_265597

theorem bag_problem (total_slips : ℕ) (value1 value2 : ℝ) (expected_value : ℝ) :
  total_slips = 12 →
  value1 = 2 →
  value2 = 7 →
  expected_value = 3.25 →
  ∃ (slips_with_value1 : ℕ),
    slips_with_value1 ≤ total_slips ∧
    (slips_with_value1 : ℝ) / total_slips * value1 +
    ((total_slips - slips_with_value1) : ℝ) / total_slips * value2 = expected_value ∧
    slips_with_value1 = 9 :=
by sorry

end NUMINAMATH_CALUDE_bag_problem_l2655_265597


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l2655_265541

theorem largest_number_in_ratio (a b c d : ℕ) (h_ratio : a * 3 = b * 2 ∧ b * 4 = c * 3 ∧ c * 5 = d * 4) 
  (h_sum : a + b + c + d = 1344) : d = 480 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l2655_265541


namespace NUMINAMATH_CALUDE_right_to_left_grouping_l2655_265517

/-- A function that represents the right-to-left grouping evaluation of expressions -/
noncomputable def rightToLeftEval (a b c d : ℝ) : ℝ := a * (b + (c - d))

/-- The theorem stating that the right-to-left grouping evaluation is correct -/
theorem right_to_left_grouping (a b c d : ℝ) :
  rightToLeftEval a b c d = a * (b + c - d) := by sorry

end NUMINAMATH_CALUDE_right_to_left_grouping_l2655_265517


namespace NUMINAMATH_CALUDE_same_weaving_rate_first_group_weavers_count_l2655_265569

/-- The number of weavers in the first group -/
def first_group_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of weavers in the second group -/
def second_group_weavers : ℕ := 14

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 49

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 14

/-- The rate of weaving is the same for both groups -/
theorem same_weaving_rate :
  (first_group_mats : ℚ) / (first_group_days * first_group_weavers) =
  (second_group_mats : ℚ) / (second_group_days * second_group_weavers) :=
sorry

/-- The number of weavers in the first group is 4 -/
theorem first_group_weavers_count :
  first_group_weavers = 4 :=
sorry

end NUMINAMATH_CALUDE_same_weaving_rate_first_group_weavers_count_l2655_265569


namespace NUMINAMATH_CALUDE_contractor_engagement_days_l2655_265586

/-- Represents the engagement of a contractor --/
structure ContractorEngagement where
  daysWorked : ℕ
  daysAbsent : ℕ
  dailyWage : ℚ
  dailyFine : ℚ
  totalAmount : ℚ

/-- Theorem: Given the conditions, the contractor was engaged for 22 days --/
theorem contractor_engagement_days (c : ContractorEngagement) 
  (h1 : c.dailyWage = 25)
  (h2 : c.dailyFine = 7.5)
  (h3 : c.totalAmount = 490)
  (h4 : c.daysAbsent = 8) :
  c.daysWorked = 22 := by
  sorry


end NUMINAMATH_CALUDE_contractor_engagement_days_l2655_265586


namespace NUMINAMATH_CALUDE_max_flight_time_l2655_265559

/-- The maximum flight time for a projectile launched under specific conditions -/
theorem max_flight_time (V₀ g : ℝ) (h₁ : V₀ > 0) (h₂ : g > 0) : 
  ∃ (τ : ℝ), τ = (2 * V₀ / g) * (2 / Real.sqrt 12.5) ∧ 
  ∀ (α : ℝ), 0 ≤ α ∧ α ≤ π / 2 ∧ Real.sin (2 * α) ≥ 0.96 → 
  (2 * V₀ * Real.sin α) / g ≤ τ :=
sorry

end NUMINAMATH_CALUDE_max_flight_time_l2655_265559


namespace NUMINAMATH_CALUDE_square_side_length_l2655_265585

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) :
  ∃ (s : ℝ), s * s = d * d / 2 ∧ s = 2 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l2655_265585


namespace NUMINAMATH_CALUDE_locus_of_Y_l2655_265528

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents a trapezoid -/
structure Trapezoid :=
  (A B C D : Point)

/-- Defines a perpendicular line to the bases of a trapezoid -/
def perpendicularLine (t : Trapezoid) : Line := sorry

/-- Defines a point on a given line -/
def pointOnLine (l : Line) : Point := sorry

/-- Constructs perpendiculars from points to lines -/
def perpendicular (p : Point) (l : Line) : Line := sorry

/-- Finds the intersection of two lines -/
def lineIntersection (l1 l2 : Line) : Point := sorry

/-- Checks if a point is on a line -/
def isPointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Checks if a line divides a segment in a given ratio -/
def dividesSameRatio (l1 l2 : Line) (seg1 seg2 : Point × Point) : Prop := sorry

/-- Main theorem: The locus of point Y is a line perpendicular to the bases -/
theorem locus_of_Y (t : Trapezoid) (l : Line) : 
  ∃ l' : Line, 
    (∀ X : Point, isPointOnLine X l → 
      let BX := Line.mk sorry sorry sorry
      let CX := Line.mk sorry sorry sorry
      let perp1 := perpendicular t.A BX
      let perp2 := perpendicular t.D CX
      let Y := lineIntersection perp1 perp2
      isPointOnLine Y l') ∧ 
    dividesSameRatio l' l (t.A, t.D) (t.B, t.C) := by
  sorry

end NUMINAMATH_CALUDE_locus_of_Y_l2655_265528


namespace NUMINAMATH_CALUDE_jordan_read_more_than_maxime_l2655_265502

-- Define the number of novels read by each person
def jordan_french : ℕ := 130
def jordan_spanish : ℕ := 20
def alexandre_french : ℕ := jordan_french / 10
def alexandre_spanish : ℕ := 3 * jordan_spanish
def camille_french : ℕ := 2 * alexandre_french
def camille_spanish : ℕ := jordan_spanish / 2

-- Define the total number of French novels read by Jordan, Alexandre, and Camille
def total_french : ℕ := jordan_french + alexandre_french + camille_french

-- Define Maxime's French and Spanish novels
def maxime_french : ℕ := total_french / 2 - 5
def maxime_spanish : ℕ := 2 * camille_spanish

-- Define the total novels read by Jordan and Maxime
def jordan_total : ℕ := jordan_french + jordan_spanish
def maxime_total : ℕ := maxime_french + maxime_spanish

-- Theorem statement
theorem jordan_read_more_than_maxime :
  jordan_total = maxime_total + 51 := by sorry

end NUMINAMATH_CALUDE_jordan_read_more_than_maxime_l2655_265502


namespace NUMINAMATH_CALUDE_angle_C_value_l2655_265590

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)
  (a b c : Real)

-- Define the theorem
theorem angle_C_value (t : Triangle) 
  (h1 : t.b = Real.sqrt 2)
  (h2 : t.c = 1)
  (h3 : t.B = π / 4) : 
  t.C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_value_l2655_265590


namespace NUMINAMATH_CALUDE_third_person_investment_range_l2655_265527

theorem third_person_investment_range (total : ℝ) (ratio_high_low : ℝ) :
  total = 143 ∧ ratio_high_low = 5 / 3 →
  ∃ (max min : ℝ),
    max = 55 ∧ min = 39 ∧
    ∀ (third : ℝ),
      (∃ (high low : ℝ),
        high + low + third = total ∧
        high / low = ratio_high_low ∧
        high ≥ third ∧ third ≥ low) →
      third ≤ max ∧ third ≥ min :=
by sorry

end NUMINAMATH_CALUDE_third_person_investment_range_l2655_265527


namespace NUMINAMATH_CALUDE_fish_population_estimate_l2655_265593

/-- Represents the number of fish in a pond given certain sampling conditions -/
def fish_in_pond (initial_caught : ℕ) (second_caught : ℕ) (marked_in_second : ℕ) : ℕ :=
  (initial_caught * second_caught) / marked_in_second

/-- Theorem stating that under given conditions, there are approximately 1200 fish in the pond -/
theorem fish_population_estimate :
  let initial_caught := 120
  let second_caught := 100
  let marked_in_second := 10
  fish_in_pond initial_caught second_caught marked_in_second = 1200 := by
  sorry

#eval fish_in_pond 120 100 10

end NUMINAMATH_CALUDE_fish_population_estimate_l2655_265593


namespace NUMINAMATH_CALUDE_perimeter_of_hundred_rectangles_l2655_265552

/-- The perimeter of a shape formed by arranging rectangles edge-to-edge -/
def perimeter_of_arranged_rectangles (n : ℕ) (length width : ℝ) : ℝ :=
  let single_rectangle_perimeter := 2 * (length + width)
  let total_perimeter_without_overlap := n * single_rectangle_perimeter
  let number_of_joins := n - 1
  let overlap_per_join := 2 * width
  total_perimeter_without_overlap - (number_of_joins * overlap_per_join)

/-- Theorem stating that the perimeter of a shape formed by 100 rectangles 
    (each 3 cm by 1 cm) arranged edge-to-edge is 602 cm -/
theorem perimeter_of_hundred_rectangles : 
  perimeter_of_arranged_rectangles 100 3 1 = 602 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_hundred_rectangles_l2655_265552


namespace NUMINAMATH_CALUDE_max_value_properties_l2655_265529

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem max_value_properties (x₀ : ℝ) 
  (h₁ : ∀ x > 0, f x ≤ f x₀) 
  (h₂ : x₀ > 0) :
  f x₀ = x₀ ∧ f x₀ < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_properties_l2655_265529


namespace NUMINAMATH_CALUDE_hyperbola_k_squared_l2655_265560

/-- A hyperbola centered at the origin, opening vertically -/
structure VerticalHyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point (x, y) lies on the hyperbola -/
def VerticalHyperbola.contains (h : VerticalHyperbola) (x y : ℝ) : Prop :=
  y^2 / h.a^2 - x^2 / h.b^2 = 1

/-- The main theorem -/
theorem hyperbola_k_squared (h : VerticalHyperbola) 
  (h1 : h.contains 4 3)
  (h2 : h.contains 0 2)
  (h3 : h.contains 2 k) : k^2 = 17/4 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_k_squared_l2655_265560


namespace NUMINAMATH_CALUDE_willie_stickers_l2655_265556

theorem willie_stickers (initial : ℕ) (given_away : ℕ) (final : ℕ) : 
  initial = 36 → given_away = 7 → final = initial - given_away → final = 29 := by
  sorry

end NUMINAMATH_CALUDE_willie_stickers_l2655_265556


namespace NUMINAMATH_CALUDE_min_e_value_l2655_265558

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit

-- Define the diameter of the circle
def diameter : ℝ := 4

-- Define the points
def P : Point := sorry
def Q : Point := sorry
def X : Point := sorry
def Y : Point := sorry
def Z : Point := sorry

-- Define the properties of the points
def is_diameter (p q : Point) : Prop := sorry
def on_semicircle (p : Point) : Prop := sorry
def is_midpoint (x : Point) : Prop := sorry
def distance (p q : Point) : ℝ := sorry
def symmetric_to (z p q : Point) : Prop := sorry

-- Define the intersection points
def A : Point := sorry
def B : Point := sorry

-- Define the length of AB
def e : ℝ := sorry

-- State the theorem
theorem min_e_value (c : Circle) :
  is_diameter P Q →
  on_semicircle X →
  on_semicircle Y →
  is_midpoint X →
  distance P Y = 5 / 4 →
  symmetric_to Z P Q →
  ∃ (min_e : ℝ), (∀ e', e' ≥ min_e) ∧ min_e = 6 - 5 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_e_value_l2655_265558


namespace NUMINAMATH_CALUDE_white_triangle_pairs_count_l2655_265566

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : Nat
  blue : Nat
  white : Nat

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  red_red : Nat
  blue_blue : Nat
  red_white : Nat
  white_white : Nat

/-- The main theorem statement -/
theorem white_triangle_pairs_count 
  (counts : TriangleCounts)
  (pairs : CoincidingPairs)
  (h1 : counts.red = 3)
  (h2 : counts.blue = 5)
  (h3 : counts.white = 8)
  (h4 : pairs.red_red = 2)
  (h5 : pairs.blue_blue = 3)
  (h6 : pairs.red_white = 2) :
  pairs.white_white = 5 := by
  sorry

end NUMINAMATH_CALUDE_white_triangle_pairs_count_l2655_265566


namespace NUMINAMATH_CALUDE_right_triangle_cosine_sine_l2655_265555

-- Define the right triangle XYZ
def RightTriangleXYZ (X Y Z : ℝ) : Prop :=
  X^2 + Y^2 = Z^2 ∧ X = 8 ∧ Z = 17

-- Theorem statement
theorem right_triangle_cosine_sine 
  (X Y Z : ℝ) (h : RightTriangleXYZ X Y Z) : 
  Real.cos (Real.arccos (X / Z)) = 15 / 17 ∧ Real.sin (Real.arcsin 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cosine_sine_l2655_265555


namespace NUMINAMATH_CALUDE_alex_escalator_time_l2655_265553

/-- The time it takes Alex to walk down the non-moving escalator -/
def time_not_moving : ℝ := 75

/-- The time it takes Alex to walk down the moving escalator -/
def time_moving : ℝ := 30

/-- The time it takes Alex to ride the escalator without walking -/
def time_riding : ℝ := 50

theorem alex_escalator_time :
  (time_not_moving * time_moving) / (time_not_moving - time_moving) = time_riding := by
  sorry

end NUMINAMATH_CALUDE_alex_escalator_time_l2655_265553


namespace NUMINAMATH_CALUDE_subset_implies_c_equals_two_l2655_265514

theorem subset_implies_c_equals_two :
  {p : ℝ × ℝ | p.1 + p.2 - 2 = 0 ∧ p.1 - 2*p.2 + 4 = 0} ⊆ {p : ℝ × ℝ | p.2 = 3*p.1 + c} →
  c = 2 :=
by sorry

end NUMINAMATH_CALUDE_subset_implies_c_equals_two_l2655_265514


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2655_265589

noncomputable def f (x : ℝ) : ℝ := (x - 2) * (x^3 - 1)

theorem tangent_line_equation :
  let p : ℝ × ℝ := (1, 0)
  let m : ℝ := deriv f p.1
  (λ (x y : ℝ) ↦ m * (x - p.1) - (y - p.2)) = (λ (x y : ℝ) ↦ 3 * x + y - 3) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2655_265589


namespace NUMINAMATH_CALUDE_correct_change_amount_and_composition_l2655_265503

def initial_money : ℚ := 20.40
def avocado_prices : List ℚ := [1.50, 2.25, 3.00]
def water_price : ℚ := 1.75
def water_quantity : ℕ := 2
def apple_price : ℚ := 0.75
def apple_quantity : ℕ := 4

def total_cost : ℚ := (List.sum avocado_prices) + (water_price * water_quantity) + (apple_price * apple_quantity)

def change : ℚ := initial_money - total_cost

theorem correct_change_amount_and_composition :
  change = 7.15 ∧
  ∃ (five_dollar : ℕ) (one_dollar : ℕ) (dime : ℕ) (nickel : ℕ),
    five_dollar = 1 ∧
    one_dollar = 2 ∧
    dime = 1 ∧
    nickel = 1 ∧
    5 * five_dollar + one_dollar + 0.1 * dime + 0.05 * nickel = change :=
by sorry

end NUMINAMATH_CALUDE_correct_change_amount_and_composition_l2655_265503


namespace NUMINAMATH_CALUDE_measurable_eq_set_l2655_265575

open MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω]
variable (F : MeasurableSpace Ω)
variable (ξ η : Ω → ℝ)

theorem measurable_eq_set (hξ : Measurable ξ) (hη : Measurable η) :
  MeasurableSet {ω | ξ ω = η ω} :=
by
  sorry

end NUMINAMATH_CALUDE_measurable_eq_set_l2655_265575


namespace NUMINAMATH_CALUDE_repeating_decimal_35_equals_fraction_l2655_265574

/-- The repeating decimal 0.3535... is equal to 35/99 -/
theorem repeating_decimal_35_equals_fraction : ∃ (x : ℚ), x = 35 / 99 ∧ 100 * x - x = 35 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_35_equals_fraction_l2655_265574


namespace NUMINAMATH_CALUDE_locus_of_midpoints_l2655_265581

/-- The locus of midpoints theorem -/
theorem locus_of_midpoints 
  (A : ℝ × ℝ) 
  (h_A : A = (4, -2))
  (B : ℝ × ℝ → Prop)
  (h_B : ∀ x y, B (x, y) ↔ x^2 + y^2 = 4)
  (P : ℝ × ℝ)
  (h_P : ∃ x y, B (x, y) ∧ P = ((A.1 + x) / 2, (A.2 + y) / 2)) :
  (P.1 - 2)^2 + (P.2 + 1)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_locus_of_midpoints_l2655_265581


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2655_265545

theorem polynomial_simplification (r : ℝ) : 
  (2 * r^3 + r^2 + 4*r - 3) - (r^3 + r^2 + 6*r - 8) = r^3 - 2*r + 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2655_265545


namespace NUMINAMATH_CALUDE_coffee_machine_price_l2655_265523

/-- The original price of a coffee machine given certain conditions -/
theorem coffee_machine_price (discount : ℕ) (payback_days : ℕ) (old_daily_cost new_daily_cost : ℕ) : 
  discount = 20 →
  payback_days = 36 →
  old_daily_cost = 8 →
  new_daily_cost = 3 →
  (payback_days * (old_daily_cost - new_daily_cost)) + discount = 200 :=
by sorry

end NUMINAMATH_CALUDE_coffee_machine_price_l2655_265523


namespace NUMINAMATH_CALUDE_radical_equality_l2655_265582

theorem radical_equality (a b c : ℕ+) :
  Real.sqrt (a * (b + c)) = a * Real.sqrt (b + c) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_radical_equality_l2655_265582


namespace NUMINAMATH_CALUDE_integer_solution_inequalities_l2655_265533

theorem integer_solution_inequalities :
  ∀ x : ℤ, (x + 7 > 5 ∧ -3*x > -9) ↔ x ∈ ({-1, 0, 1, 2} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_inequalities_l2655_265533


namespace NUMINAMATH_CALUDE_input_is_input_command_l2655_265571

-- Define the type for programming commands
inductive ProgrammingCommand
  | PRINT
  | INPUT
  | THEN
  | END

-- Define a function to check if a command is used for input
def isInputCommand (cmd : ProgrammingCommand) : Prop :=
  match cmd with
  | ProgrammingCommand.INPUT => True
  | _ => False

-- Theorem: INPUT is the only command used for receiving user input
theorem input_is_input_command :
  ∀ (cmd : ProgrammingCommand),
    isInputCommand cmd ↔ cmd = ProgrammingCommand.INPUT :=
  sorry

end NUMINAMATH_CALUDE_input_is_input_command_l2655_265571


namespace NUMINAMATH_CALUDE_rotten_apples_l2655_265587

theorem rotten_apples (apples_per_crate : ℕ) (num_crates : ℕ) (boxes : ℕ) (apples_per_box : ℕ) :
  apples_per_crate = 180 →
  num_crates = 12 →
  boxes = 100 →
  apples_per_box = 20 →
  apples_per_crate * num_crates - boxes * apples_per_box = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_rotten_apples_l2655_265587


namespace NUMINAMATH_CALUDE_factory_scrap_rate_l2655_265577

/-- The overall scrap rate of a factory with two machines -/
def overall_scrap_rate (output_a output_b scrap_rate_a scrap_rate_b : ℝ) : ℝ :=
  output_a * scrap_rate_a + output_b * scrap_rate_b

theorem factory_scrap_rate :
  overall_scrap_rate 0.45 0.55 0.02 0.03 = 0.0255 := by
  sorry

end NUMINAMATH_CALUDE_factory_scrap_rate_l2655_265577


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2655_265521

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (1 + 3*i) / (1 + i) = 2 + i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2655_265521


namespace NUMINAMATH_CALUDE_unique_number_exists_l2655_265540

theorem unique_number_exists : ∃! x : ℝ, x / 2 + x + 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_exists_l2655_265540


namespace NUMINAMATH_CALUDE_derivative_problems_l2655_265568

open Real

theorem derivative_problems :
  (∀ x : ℝ, x > 0 → deriv (λ x => x * log x) x = log x + 1) ∧
  (∀ x : ℝ, x ≠ 0 → deriv (λ x => sin x / x) x = (x * cos x - sin x) / x^2) :=
by sorry

end NUMINAMATH_CALUDE_derivative_problems_l2655_265568


namespace NUMINAMATH_CALUDE_travel_time_with_reduced_speed_l2655_265519

/-- Proves that the time taken to travel a given distance with reduced speed is as expected -/
theorem travel_time_with_reduced_speed 
  (distance : ℝ) 
  (no_traffic_time : ℝ) 
  (speed_reduction : ℝ) 
  (heavy_traffic_time : ℝ) 
  (h1 : distance = 200) 
  (h2 : no_traffic_time = 4) 
  (h3 : speed_reduction = 10) 
  (h4 : heavy_traffic_time = 5) : 
  heavy_traffic_time = distance / (distance / no_traffic_time - speed_reduction) := by
  sorry

#check travel_time_with_reduced_speed

end NUMINAMATH_CALUDE_travel_time_with_reduced_speed_l2655_265519


namespace NUMINAMATH_CALUDE_winning_numbers_are_correct_l2655_265518

def winning_numbers : Set Nat :=
  {n : Nat | n ≥ 1 ∧ n ≤ 999 ∧ n % 100 = 88}

theorem winning_numbers_are_correct :
  winning_numbers = {88, 188, 288, 388, 488, 588, 688, 788, 888, 988} := by
  sorry

end NUMINAMATH_CALUDE_winning_numbers_are_correct_l2655_265518


namespace NUMINAMATH_CALUDE_greatest_b_value_l2655_265501

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 8*x - 15 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 8*5 - 15 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l2655_265501


namespace NUMINAMATH_CALUDE_problem_solution_l2655_265534

theorem problem_solution (a b c : ℚ) : 
  8 = (2 / 100) * a → 
  2 = (8 / 100) * b → 
  c = b / a → 
  c = 1 / 16 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2655_265534


namespace NUMINAMATH_CALUDE_rational_square_property_l2655_265538

theorem rational_square_property (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ (z : ℚ), 1 - x*y = z^2 := by
  sorry

end NUMINAMATH_CALUDE_rational_square_property_l2655_265538


namespace NUMINAMATH_CALUDE_cone_frustum_sphere_ratio_l2655_265520

/-- A cone frustum with given height and base radius -/
structure ConeFrustum where
  height : ℝ
  baseRadius : ℝ

/-- The radius of the inscribed sphere of a cone frustum -/
def inscribedSphereRadius (cf : ConeFrustum) : ℝ := sorry

/-- The radius of the circumscribed sphere of a cone frustum -/
def circumscribedSphereRadius (cf : ConeFrustum) : ℝ := sorry

theorem cone_frustum_sphere_ratio :
  let cf : ConeFrustum := { height := 12, baseRadius := 5 }
  (inscribedSphereRadius cf) / (circumscribedSphereRadius cf) = 80 / 169 := by
  sorry

end NUMINAMATH_CALUDE_cone_frustum_sphere_ratio_l2655_265520


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l2655_265504

theorem no_integer_solutions_for_equation : 
  ¬ ∃ (x y z : ℤ), 4 * x^2 + 77 * y^2 = 487 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l2655_265504


namespace NUMINAMATH_CALUDE_no_integer_solution_l2655_265548

theorem no_integer_solution : ¬ ∃ (n : ℤ), (n + 15 > 18) ∧ (-3*n > -9) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2655_265548


namespace NUMINAMATH_CALUDE_image_and_preimage_l2655_265578

def f (x : ℝ) : ℝ := x^2 - 2*x - 1

theorem image_and_preimage :
  (f (1 + Real.sqrt 2) = 0) ∧
  ({x : ℝ | f x = -1} = {0, 2}) := by
sorry

end NUMINAMATH_CALUDE_image_and_preimage_l2655_265578


namespace NUMINAMATH_CALUDE_fraction_value_l2655_265522

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  a * c / (b * d) = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2655_265522


namespace NUMINAMATH_CALUDE_nonreal_roots_product_l2655_265511

theorem nonreal_roots_product (x : ℂ) : 
  (x^4 - 4*x^3 + 6*x^2 - 4*x = 2047) → 
  (∃ a b : ℂ, a ≠ b ∧ a.im ≠ 0 ∧ b.im ≠ 0 ∧ 
   (x = a ∨ x = b) ∧ (x^4 - 4*x^3 + 6*x^2 - 4*x = 2047) ∧
   (a * b = 257)) := by
sorry

end NUMINAMATH_CALUDE_nonreal_roots_product_l2655_265511


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l2655_265543

theorem fraction_equality_sum (p q : ℚ) : p / q = 2 / 7 → 2 * p + q = 11 * p / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l2655_265543


namespace NUMINAMATH_CALUDE_prob_two_aces_full_deck_prob_two_aces_after_two_kings_l2655_265547

/-- Represents a deck of cards with Aces, Kings, and Queens -/
structure Deck :=
  (num_aces : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)

/-- Calculates the probability of drawing two Aces from a given deck -/
def prob_two_aces (d : Deck) : ℚ :=
  (d.num_aces.choose 2 : ℚ) / (d.num_aces + d.num_kings + d.num_queens).choose 2

/-- The full deck with 4 each of Aces, Kings, and Queens -/
def full_deck : Deck := ⟨4, 4, 4⟩

/-- The deck after two Kings have been drawn -/
def deck_after_two_kings : Deck := ⟨4, 2, 4⟩

theorem prob_two_aces_full_deck :
  prob_two_aces full_deck = 1 / 11 :=
sorry

theorem prob_two_aces_after_two_kings :
  prob_two_aces deck_after_two_kings = 2 / 15 :=
sorry

end NUMINAMATH_CALUDE_prob_two_aces_full_deck_prob_two_aces_after_two_kings_l2655_265547


namespace NUMINAMATH_CALUDE_arithmetic_mean_inequality_negative_l2655_265579

theorem arithmetic_mean_inequality_negative (m n : ℝ) (h1 : m < n) (h2 : n < 0) : n / m + m / n > 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_inequality_negative_l2655_265579


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l2655_265505

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem fourth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_diff : a 2 - a 1 = 2)
  (h_arithmetic : 2 * a 2 = (3 * a 1 + a 3) / 2) :
  a 4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l2655_265505


namespace NUMINAMATH_CALUDE_line_equivalence_l2655_265570

/-- Given a line in the form (3, -7) · ((x, y) - (2, 8)) = 0, prove it's equivalent to y = (3/7)x + 50/7 -/
theorem line_equivalence (x y : ℝ) :
  (3 : ℝ) * (x - 2) + (-7 : ℝ) * (y - 8) = 0 ↔ y = (3/7)*x + 50/7 :=
by sorry

end NUMINAMATH_CALUDE_line_equivalence_l2655_265570


namespace NUMINAMATH_CALUDE_gcf_2835_9150_l2655_265594

theorem gcf_2835_9150 : Nat.gcd 2835 9150 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_2835_9150_l2655_265594


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_0000064_l2655_265542

theorem scientific_notation_of_0_0000064 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 0.0000064 = a * (10 : ℝ) ^ n ∧ a = 6.4 ∧ n = -6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_0000064_l2655_265542


namespace NUMINAMATH_CALUDE_greatest_n_for_perfect_square_T_l2655_265546

def g (x : ℕ) : ℕ := 
  if x % 2 = 1 then 1 else 0

def T (n : ℕ) : ℕ := 2^n

def is_perfect_square (x : ℕ) : Prop :=
  ∃ y : ℕ, y * y = x

theorem greatest_n_for_perfect_square_T : 
  (∃ n : ℕ, n < 500 ∧ is_perfect_square (T n) ∧ 
    ∀ m : ℕ, m < 500 → is_perfect_square (T m) → m ≤ n) →
  (∃ n : ℕ, n = 498 ∧ n < 500 ∧ is_perfect_square (T n) ∧ 
    ∀ m : ℕ, m < 500 → is_perfect_square (T m) → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_n_for_perfect_square_T_l2655_265546


namespace NUMINAMATH_CALUDE_pi_approximation_after_three_tiaoRi_l2655_265565

def tiaoRiMethod (a b c d : ℕ) : ℚ := (b + d) / (a + c)

theorem pi_approximation_after_three_tiaoRi :
  let initial_lower : ℚ := 31 / 10
  let initial_upper : ℚ := 49 / 15
  let first_upper : ℚ := tiaoRiMethod 10 31 15 49
  let second_upper : ℚ := tiaoRiMethod 15 47 5 16
  let third_upper : ℚ := tiaoRiMethod 15 47 5 16
  initial_lower < Real.pi ∧ Real.pi < initial_upper →
  third_upper = 63 / 20 :=
by sorry

end NUMINAMATH_CALUDE_pi_approximation_after_three_tiaoRi_l2655_265565


namespace NUMINAMATH_CALUDE_product_equals_eight_l2655_265535

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = r * b n

/-- The main theorem -/
theorem product_equals_eight
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_nonzero : ∀ n, a n ≠ 0)
  (h_condition : a 3 - 2 * (a 6)^2 + 3 * a 7 = 0)
  (h_equal : b 6 = a 6) :
  b 1 * b 7 * b 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_eight_l2655_265535


namespace NUMINAMATH_CALUDE_marts_income_percentage_l2655_265554

theorem marts_income_percentage (juan tim mart : ℝ) 
  (h1 : mart = 1.3 * tim) 
  (h2 : tim = 0.6 * juan) : 
  mart = 0.78 * juan := by
sorry

end NUMINAMATH_CALUDE_marts_income_percentage_l2655_265554


namespace NUMINAMATH_CALUDE_train_meeting_probability_l2655_265536

-- Define the time intervals
def train_arrival_interval : ℝ := 60  -- 60 minutes between 1:00 and 2:00
def alex_arrival_interval : ℝ := 75   -- 75 minutes between 1:00 and 2:15
def train_wait_time : ℝ := 15         -- 15 minutes wait time

-- Define the probability calculation function
def calculate_probability (train_interval : ℝ) (alex_interval : ℝ) (wait_time : ℝ) : ℚ :=
  -- The actual calculation is not implemented, just the type signature
  0

-- Theorem statement
theorem train_meeting_probability :
  calculate_probability train_arrival_interval alex_arrival_interval train_wait_time = 7/40 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_probability_l2655_265536


namespace NUMINAMATH_CALUDE_kayla_kimiko_age_ratio_l2655_265596

/-- Proves that the ratio of Kayla's age to Kimiko's age is 1:2 -/
theorem kayla_kimiko_age_ratio :
  let kimiko_age : ℕ := 26
  let min_driving_age : ℕ := 18
  let years_until_driving : ℕ := 5
  let kayla_age : ℕ := min_driving_age - years_until_driving
  (kayla_age : ℚ) / kimiko_age = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_kayla_kimiko_age_ratio_l2655_265596


namespace NUMINAMATH_CALUDE_sequence_increasing_l2655_265599

def a (n : ℕ) : ℚ := (n - 1) / (n + 1)

theorem sequence_increasing : ∀ k j : ℕ, k > j → j ≥ 1 → a k > a j := by
  sorry

end NUMINAMATH_CALUDE_sequence_increasing_l2655_265599


namespace NUMINAMATH_CALUDE_quadrilateral_vector_sum_l2655_265515

/-- Given a quadrilateral ABCD in a real inner product space, with M as the intersection of its diagonals,
    prove that for any point O not equal to M, the sum of vectors from O to each vertex
    is equal to four times the vector from O to M. -/
theorem quadrilateral_vector_sum (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (A B C D M O : V) : 
  M ≠ O →  -- O is not equal to M
  (A - C) = (D - B) →  -- M is the midpoint of AC
  (B - D) = (C - A) →  -- M is the midpoint of BD
  (O - A) + (O - B) + (O - C) + (O - D) = 4 • (O - M) := by sorry

end NUMINAMATH_CALUDE_quadrilateral_vector_sum_l2655_265515


namespace NUMINAMATH_CALUDE_triangle_square_perimeter_l2655_265524

theorem triangle_square_perimeter (a b c : ℝ) (s : ℝ) : 
  a = 5 → b = 12 → c = 13 → 
  (1/2) * a * b = s^2 → 
  4 * s = 4 * Real.sqrt 30 :=
by sorry

end NUMINAMATH_CALUDE_triangle_square_perimeter_l2655_265524


namespace NUMINAMATH_CALUDE_train_crossing_bridge_l2655_265592

/-- A train crossing a bridge problem -/
theorem train_crossing_bridge (train_length bridge_length : ℝ) (train_speed_kmph : ℝ) :
  train_length = 100 →
  bridge_length = 150 →
  train_speed_kmph = 18 →
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_bridge_l2655_265592


namespace NUMINAMATH_CALUDE_base_8_to_10_367_l2655_265510

-- Define the base-8 number as a list of digits
def base_8_number : List Nat := [3, 6, 7]

-- Define the function to convert base-8 to base-10
def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem base_8_to_10_367 :
  base_8_to_10 base_8_number = 247 := by sorry

end NUMINAMATH_CALUDE_base_8_to_10_367_l2655_265510


namespace NUMINAMATH_CALUDE_third_degree_polynomial_property_l2655_265525

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- The property that the absolute value of g at specified points is 15 -/
def has_abs_value_15 (g : ThirdDegreePolynomial) : Prop :=
  |g 1| = 15 ∧ |g 3| = 15 ∧ |g 4| = 15 ∧ |g 5| = 15 ∧ |g 6| = 15 ∧ |g 7| = 15

theorem third_degree_polynomial_property (g : ThirdDegreePolynomial) 
  (h : has_abs_value_15 g) : |g 0| = 645/8 := by
  sorry


end NUMINAMATH_CALUDE_third_degree_polynomial_property_l2655_265525


namespace NUMINAMATH_CALUDE_only_odd_solution_is_one_l2655_265588

theorem only_odd_solution_is_one :
  ∀ y : ℤ, ∃ x : ℤ, x^2 + 2*y^2 = y*x^2 + y + 1 ∧ Odd y → y = 1 :=
by sorry

end NUMINAMATH_CALUDE_only_odd_solution_is_one_l2655_265588


namespace NUMINAMATH_CALUDE_correct_operation_l2655_265530

theorem correct_operation (a : ℝ) : 4 * a - a = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2655_265530


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2655_265512

theorem min_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y^2 = 4) :
  x + 2*y ≥ 3 * (4 : ℝ)^(1/3) ∧ 
  ∃ (x₀ y₀ : ℝ), 0 < x₀ ∧ 0 < y₀ ∧ x₀ * y₀^2 = 4 ∧ x₀ + 2*y₀ = 3 * (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2655_265512


namespace NUMINAMATH_CALUDE_jose_bottle_caps_l2655_265544

def initial_bottle_caps : ℕ := 26
def additional_bottle_caps : ℕ := 13

theorem jose_bottle_caps : 
  initial_bottle_caps + additional_bottle_caps = 39 := by
  sorry

end NUMINAMATH_CALUDE_jose_bottle_caps_l2655_265544
