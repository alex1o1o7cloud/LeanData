import Mathlib

namespace NUMINAMATH_CALUDE_days_without_class_total_course_days_course_duration_proof_l2709_270928

/- Define the parameters of the problem -/
def total_hours : ℕ := 30
def class_duration : ℕ := 1
def afternoons_without_class : ℕ := 20
def mornings_without_class : ℕ := 18

/- Define the theorems to be proved -/
theorem days_without_class : ℕ := by sorry

theorem total_course_days : ℕ := by sorry

/- Main theorem combining both results -/
theorem course_duration_proof :
  (days_without_class = 4) ∧ (total_course_days = 34) := by
  sorry

end NUMINAMATH_CALUDE_days_without_class_total_course_days_course_duration_proof_l2709_270928


namespace NUMINAMATH_CALUDE_cylinder_radius_theorem_l2709_270902

/-- The original radius of a cylinder with the given properties -/
def original_radius : ℝ := 8

/-- The original height of the cylinder -/
def original_height : ℝ := 3

/-- The increase in either radius or height -/
def increase : ℝ := 4

theorem cylinder_radius_theorem :
  (π * (original_radius + increase)^2 * original_height = 
   π * original_radius^2 * (original_height + increase)) ∧
  original_radius > 0 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_theorem_l2709_270902


namespace NUMINAMATH_CALUDE_johns_pace_l2709_270939

/-- Given the conditions of a race between John and Steve, prove that John's pace during his final push was 178 / 42.5 m/s. -/
theorem johns_pace (john_initial_behind : ℝ) (steve_speed : ℝ) (john_final_ahead : ℝ) (push_duration : ℝ) :
  john_initial_behind = 15 →
  steve_speed = 3.8 →
  john_final_ahead = 2 →
  push_duration = 42.5 →
  (john_initial_behind + john_final_ahead + steve_speed * push_duration) / push_duration = 178 / 42.5 := by
  sorry

#eval (178 : ℚ) / 42.5

end NUMINAMATH_CALUDE_johns_pace_l2709_270939


namespace NUMINAMATH_CALUDE_system_solution_l2709_270960

theorem system_solution (x y : ℝ) : x + y = -5 ∧ 2*y = -2 → x = -4 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2709_270960


namespace NUMINAMATH_CALUDE_lateral_area_of_specific_prism_l2709_270919

/-- A prism with a square base and a circumscribed sphere -/
structure SquareBasePrism where
  /-- Side length of the square base -/
  baseSide : ℝ
  /-- Height of the prism -/
  height : ℝ
  /-- Volume of the circumscribed sphere -/
  sphereVolume : ℝ

/-- Theorem: The lateral area of a square-based prism with circumscribed sphere volume 4π/3 and base side length 1 is 4√2 -/
theorem lateral_area_of_specific_prism (p : SquareBasePrism) 
  (h1 : p.baseSide = 1)
  (h2 : p.sphereVolume = 4 * Real.pi / 3) : 
  4 * p.baseSide * p.height = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_lateral_area_of_specific_prism_l2709_270919


namespace NUMINAMATH_CALUDE_B_k_closed_form_l2709_270955

/-- B_k(n) is the largest possible number of elements in a 2-separable k-configuration of a set with 2n elements -/
def B_k (k n : ℕ) : ℕ := Nat.choose (2*n) k - 2 * Nat.choose n k

/-- Theorem stating the closed-form expression for B_k(n) -/
theorem B_k_closed_form (k n : ℕ) (h1 : 2 ≤ k) (h2 : k ≤ n) :
  B_k k n = Nat.choose (2*n) k - 2 * Nat.choose n k := by
  sorry

end NUMINAMATH_CALUDE_B_k_closed_form_l2709_270955


namespace NUMINAMATH_CALUDE_classroom_pencils_l2709_270912

/-- The number of pencils a teacher gives out to a classroom of students. -/
def pencils_given_out (num_students : ℕ) (dozens_per_student : ℕ) (pencils_per_dozen : ℕ) : ℕ :=
  num_students * dozens_per_student * pencils_per_dozen

/-- Theorem stating the total number of pencils given out in the classroom scenario. -/
theorem classroom_pencils : 
  pencils_given_out 96 7 12 = 8064 := by
  sorry

end NUMINAMATH_CALUDE_classroom_pencils_l2709_270912


namespace NUMINAMATH_CALUDE_new_boy_weight_l2709_270926

theorem new_boy_weight (original_count : ℕ) (original_average : ℝ) (new_average : ℝ) : 
  original_count = 5 →
  original_average = 35 →
  new_average = 36 →
  (original_count + 1) * new_average - original_count * original_average = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_new_boy_weight_l2709_270926


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l2709_270991

/-- Represents an event in a probability space -/
structure Event (Ω : Type) :=
  (set : Set Ω)

/-- Two events are mutually exclusive if their intersection is empty -/
def mutually_exclusive {Ω : Type} (A B : Event Ω) : Prop :=
  A.set ∩ B.set = ∅

/-- Represents the sample space for shooting at a target -/
inductive ShootingTarget
  | ring7
  | ring8
  | miss

/-- Represents the sample space for two people shooting -/
inductive TwoPeopleShooting
  | bothHit
  | AHitBMiss
  | AMissBHit
  | bothMiss

/-- Represents the sample space for drawing two balls -/
inductive TwoBallDraw
  | redRed
  | redBlack
  | blackRed
  | blackBlack

/-- Event 1: Hitting the 7th ring -/
def hit7th : Event ShootingTarget :=
  ⟨{ShootingTarget.ring7}⟩

/-- Event 1: Hitting the 8th ring -/
def hit8th : Event ShootingTarget :=
  ⟨{ShootingTarget.ring8}⟩

/-- Event 2: At least one person hits the target -/
def atLeastOneHit : Event TwoPeopleShooting :=
  ⟨{TwoPeopleShooting.bothHit, TwoPeopleShooting.AHitBMiss, TwoPeopleShooting.AMissBHit}⟩

/-- Event 2: A hits, B misses -/
def AHitBMiss : Event TwoPeopleShooting :=
  ⟨{TwoPeopleShooting.AHitBMiss}⟩

/-- Event 3: At least one black ball -/
def atLeastOneBlack : Event TwoBallDraw :=
  ⟨{TwoBallDraw.redBlack, TwoBallDraw.blackRed, TwoBallDraw.blackBlack}⟩

/-- Event 3: Both balls are red -/
def bothRed : Event TwoBallDraw :=
  ⟨{TwoBallDraw.redRed}⟩

/-- Event 4: No black balls -/
def noBlack : Event TwoBallDraw :=
  ⟨{TwoBallDraw.redRed}⟩

/-- Event 4: Exactly one red ball -/
def oneRed : Event TwoBallDraw :=
  ⟨{TwoBallDraw.redBlack, TwoBallDraw.blackRed}⟩

theorem mutually_exclusive_events :
  (mutually_exclusive hit7th hit8th) ∧
  (¬mutually_exclusive atLeastOneHit AHitBMiss) ∧
  (mutually_exclusive atLeastOneBlack bothRed) ∧
  (mutually_exclusive noBlack oneRed) := by
  sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l2709_270991


namespace NUMINAMATH_CALUDE_smallest_multiple_of_nine_l2709_270997

theorem smallest_multiple_of_nine (x y : ℤ) 
  (hx : ∃ k : ℤ, x + 2 = 9 * k) 
  (hy : ∃ k : ℤ, y - 2 = 9 * k) : 
  (∃ n : ℕ, n > 0 ∧ ∃ k : ℤ, x^2 - x*y + y^2 + n = 9 * k) ∧ 
  (∀ m : ℕ, m > 0 → (∃ k : ℤ, x^2 - x*y + y^2 + m = 9 * k) → m ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_nine_l2709_270997


namespace NUMINAMATH_CALUDE_a_neither_sufficient_nor_necessary_for_b_l2709_270976

/-- Proposition A: The complex number z satisfies |z-3|+|z+3| is a constant -/
def propositionA (z : ℂ) : Prop :=
  ∃ c : ℝ, ∀ z : ℂ, Complex.abs (z - 3) + Complex.abs (z + 3) = c

/-- Proposition B: The trajectory of the point corresponding to the complex number z in the complex plane is an ellipse -/
def propositionB (z : ℂ) : Prop :=
  ∃ a b : ℝ, ∃ f₁ f₂ : ℂ, ∀ z : ℂ, Complex.abs (z - f₁) + Complex.abs (z - f₂) = a + b

/-- A is neither sufficient nor necessary for B -/
theorem a_neither_sufficient_nor_necessary_for_b :
  (¬∀ z : ℂ, propositionA z → propositionB z) ∧
  (¬∀ z : ℂ, propositionB z → propositionA z) :=
sorry

end NUMINAMATH_CALUDE_a_neither_sufficient_nor_necessary_for_b_l2709_270976


namespace NUMINAMATH_CALUDE_sum_squares_inequality_sum_squares_equality_l2709_270971

theorem sum_squares_inequality (a b c : ℝ) 
  (ha : a ≥ -1) (hb : b ≥ -1) (hc : c ≥ -1) 
  (h_sum_cubes : a^3 + b^3 + c^3 = 1) :
  a + b + c + a^2 + b^2 + c^2 ≤ 4 := by
  sorry

-- Equality case
theorem sum_squares_equality (a b c : ℝ) 
  (ha : a ≥ -1) (hb : b ≥ -1) (hc : c ≥ -1) 
  (h_sum_cubes : a^3 + b^3 + c^3 = 1) :
  (a + b + c + a^2 + b^2 + c^2 = 4) ↔ 
  ((a = 1 ∧ b = 1 ∧ c = -1) ∨ 
   (a = 1 ∧ b = -1 ∧ c = 1) ∨ 
   (a = -1 ∧ b = 1 ∧ c = 1)) := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_inequality_sum_squares_equality_l2709_270971


namespace NUMINAMATH_CALUDE_parallelepiped_theorem_l2709_270945

/-- Represents a parallelepiped with a sphere inscribed --/
structure ParallelepipedWithSphere where
  -- Edge length of the base square
  base_edge : ℝ
  -- Height of the parallelepiped (length of A₁A)
  height : ℝ
  -- Distance from C to K on edge CD
  ck : ℝ
  -- Distance from K to D on edge CD
  kd : ℝ

/-- Properties of the parallelepiped and inscribed sphere --/
def parallelepiped_properties (p : ParallelepipedWithSphere) : Prop :=
  -- Edge A₁A is perpendicular to face ABCD (implied by the structure)
  -- Sphere Ω touches edges BB₁, B₁C₁, C₁C, CB, CD (implied by the structure)
  -- Sphere Ω touches edge CD at point K
  p.ck + p.kd = p.base_edge ∧
  -- Given values for CK and KD
  p.ck = 9 ∧ p.kd = 1 ∧
  -- Sphere Ω touches edge A₁D₁ (implied by the structure)
  -- The base is a square (implied by the problem description)
  p.base_edge = p.height

/-- Main theorem stating the properties to be proven --/
theorem parallelepiped_theorem (p : ParallelepipedWithSphere) 
  (h : parallelepiped_properties p) : 
  p.height = 18 ∧ 
  p.height * p.base_edge * p.base_edge = 1944 ∧ 
  ∃ (r : ℝ), r * r = 90 ∧ r = 3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_theorem_l2709_270945


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2709_270987

theorem quadratic_roots_sum (p q : ℝ) : 
  p^2 - 6*p + 8 = 0 → q^2 - 6*q + 8 = 0 → p^3 + p^4*q^2 + p^2*q^4 + q^3 = 1352 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2709_270987


namespace NUMINAMATH_CALUDE_length_AB_line_MN_fixed_point_min_distance_PM_l2709_270951

-- Define the circles and line
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + 4 = 0
def line_l (x y : ℝ) : Prop := x - 2*y + 5 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_O A.1 A.2 ∧ circle_C A.1 A.2 ∧
  circle_O B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Define the tangent points M and N
def tangent_points (P M N : ℝ × ℝ) : Prop :=
  line_l P.1 P.2 ∧
  circle_O M.1 M.2 ∧ circle_O N.1 N.2 ∧
  (P.1 - M.1) * M.1 + (P.2 - M.2) * M.2 = 0 ∧
  (P.1 - N.1) * N.1 + (P.2 - N.2) * N.2 = 0

-- Theorem statements
theorem length_AB : ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 5 / 5 := 
sorry

theorem line_MN_fixed_point : ∀ P M N : ℝ × ℝ, tangent_points P M N →
  ∃ t : ℝ, M.1 + t * (N.1 - M.1) = -4/5 ∧ M.2 + t * (N.2 - M.2) = 8/5 :=
sorry

theorem min_distance_PM : ∀ P : ℝ × ℝ, line_l P.1 P.2 →
  (∃ M : ℝ × ℝ, circle_O M.1 M.2 ∧ 
    ∀ N : ℝ × ℝ, circle_O N.1 N.2 → 
      Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) ≤ 
      Real.sqrt ((P.1 - N.1)^2 + (P.2 - N.2)^2)) ∧
  (∃ M : ℝ × ℝ, circle_O M.1 M.2 ∧ 
    Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) = 1) :=
sorry

end NUMINAMATH_CALUDE_length_AB_line_MN_fixed_point_min_distance_PM_l2709_270951


namespace NUMINAMATH_CALUDE_kyle_to_grant_ratio_l2709_270972

def parker_distance : ℝ := 16

def grant_distance : ℝ := parker_distance * 1.25

def kyle_distance : ℝ := parker_distance + 24

theorem kyle_to_grant_ratio : kyle_distance / grant_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_kyle_to_grant_ratio_l2709_270972


namespace NUMINAMATH_CALUDE_closed_convex_curve_length_l2709_270927

/-- A closed convex curve in 2D space -/
structure ClosedConvexCurve where
  -- We assume the curve is represented by some internal structure
  -- The details of this structure are not important for this theorem
  curve : Unit

/-- The length of a curve -/
noncomputable def length (c : ClosedConvexCurve) : ℝ := sorry

/-- The length of the projection of a curve onto a line -/
noncomputable def projectionLength (c : ClosedConvexCurve) (l : Line) : ℝ := sorry

/-- A straight line in 2D space -/
structure Line where
  -- We assume the line is represented by some internal structure
  -- The details of this structure are not important for this theorem
  line : Unit

theorem closed_convex_curve_length (c : ClosedConvexCurve) :
  (∀ l : Line, projectionLength c l = 1) → length c = π := by
  sorry

end NUMINAMATH_CALUDE_closed_convex_curve_length_l2709_270927


namespace NUMINAMATH_CALUDE_hourly_charge_is_correct_l2709_270989

/-- The hourly charge for renting a bike -/
def hourly_charge : ℝ := 7

/-- The fixed fee for renting a bike -/
def fixed_fee : ℝ := 17

/-- The number of hours Tom rented the bike -/
def rental_hours : ℝ := 9

/-- The total cost Tom paid for renting the bike -/
def total_cost : ℝ := 80

/-- Theorem stating that the hourly charge is correct given the conditions -/
theorem hourly_charge_is_correct : 
  fixed_fee + rental_hours * hourly_charge = total_cost := by sorry

end NUMINAMATH_CALUDE_hourly_charge_is_correct_l2709_270989


namespace NUMINAMATH_CALUDE_equation_solution_l2709_270994

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (x - 15))) = 54 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2709_270994


namespace NUMINAMATH_CALUDE_product_equals_zero_l2709_270901

theorem product_equals_zero (n : ℤ) (h : n = 1) : (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l2709_270901


namespace NUMINAMATH_CALUDE_cubic_expression_equality_l2709_270904

theorem cubic_expression_equality : 103^3 - 3 * 103^2 + 3 * 103 - 1 = 1061208 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_equality_l2709_270904


namespace NUMINAMATH_CALUDE_median_of_special_list_l2709_270908

/-- Represents the sum of integers from 1 to n -/
def triangularSum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents our special list where each number n appears n times, up to 250 -/
def specialList : List ℕ := sorry

/-- The length of our special list -/
def listLength : ℕ := triangularSum 250

/-- The index of the median element in our list -/
def medianIndex : ℕ := (listLength + 1) / 2

/-- Function to find the smallest n such that triangularSum n ≥ target -/
def findSmallestN (target : ℕ) : ℕ := sorry

theorem median_of_special_list :
  let n := findSmallestN medianIndex
  n = 177 := by sorry

end NUMINAMATH_CALUDE_median_of_special_list_l2709_270908


namespace NUMINAMATH_CALUDE_linear_function_decreasing_l2709_270952

/-- A linear function y = mx + b where m is the slope and b is the y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- The linear function y = (k-3)x + 2 -/
def f (k : ℝ) : LinearFunction :=
  { slope := k - 3, intercept := 2 }

/-- A function is decreasing if for any x1 < x2, f(x1) > f(x2) -/
def isDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2

/-- The main theorem: The linear function y = (k-3)x + 2 is decreasing iff k < 3 -/
theorem linear_function_decreasing (k : ℝ) :
  isDecreasing (fun x ↦ (f k).slope * x + (f k).intercept) ↔ k < 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_l2709_270952


namespace NUMINAMATH_CALUDE_steves_nickels_l2709_270979

theorem steves_nickels (nickels dimes : ℕ) : 
  dimes = nickels + 4 →
  5 * nickels + 10 * dimes = 70 →
  nickels = 2 := by
sorry

end NUMINAMATH_CALUDE_steves_nickels_l2709_270979


namespace NUMINAMATH_CALUDE_perpendicular_slope_l2709_270916

theorem perpendicular_slope (x y : ℝ) :
  (4 * x - 6 * y = 12) →
  (∃ m : ℝ, m = -3/2 ∧ m * (2/3) = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l2709_270916


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l2709_270905

theorem greatest_q_minus_r : ∃ (q r : ℕ+), 
  839 = 19 * q + r ∧ 
  ∀ (q' r' : ℕ+), 839 = 19 * q' + r' → (q - r : ℤ) ≥ (q' - r' : ℤ) ∧
  (q - r : ℤ) = 41 := by
  sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l2709_270905


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l2709_270918

/-- The number of trailing zeros in a positive integer -/
def trailingZeros (n : ℕ+) : ℕ := sorry

/-- The product of 30 and 450 -/
def product : ℕ+ := 30 * 450

theorem product_trailing_zeros :
  trailingZeros product = 3 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l2709_270918


namespace NUMINAMATH_CALUDE_betty_oranges_l2709_270984

theorem betty_oranges (emily sandra betty : ℕ) 
  (h1 : emily = 7 * sandra) 
  (h2 : sandra = 3 * betty) 
  (h3 : emily = 252) : 
  betty = 12 := by
  sorry

end NUMINAMATH_CALUDE_betty_oranges_l2709_270984


namespace NUMINAMATH_CALUDE_x_minus_y_equals_six_l2709_270903

theorem x_minus_y_equals_six (x y : ℚ) 
  (eq1 : 3 * x + 2 * y = 16) 
  (eq2 : x + 3 * y = 26/5) : 
  x - y = 6 := by sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_six_l2709_270903


namespace NUMINAMATH_CALUDE_circle_equation_through_points_l2709_270966

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) -/
theorem circle_equation_through_points :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 4*x - 6*y = 0) ↔
  ((x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_through_points_l2709_270966


namespace NUMINAMATH_CALUDE_specific_window_side_length_l2709_270950

/-- Represents a square window with glass panes -/
structure SquareWindow where
  /-- Number of panes in each row/column -/
  panes_per_side : ℕ
  /-- Width of a single pane -/
  pane_width : ℝ
  /-- Width of borders between panes and around the window -/
  border_width : ℝ

/-- Calculates the side length of the square window -/
def window_side_length (w : SquareWindow) : ℝ :=
  w.panes_per_side * w.pane_width + (w.panes_per_side + 1) * w.border_width

/-- Theorem stating the side length of the specific window described in the problem -/
theorem specific_window_side_length :
  ∃ w : SquareWindow,
    w.panes_per_side = 3 ∧
    w.pane_width * 3 = w.pane_width * w.panes_per_side ∧
    w.border_width = 3 ∧
    window_side_length w = 42 := by
  sorry

end NUMINAMATH_CALUDE_specific_window_side_length_l2709_270950


namespace NUMINAMATH_CALUDE_set_size_from_averages_l2709_270929

theorem set_size_from_averages (S : Finset ℝ) (sum : ℝ) (n : ℕ) :
  sum = S.sum (λ x => x) →
  n = S.card →
  sum / n = 6.2 →
  (sum + 7) / n = 6.9 →
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_set_size_from_averages_l2709_270929


namespace NUMINAMATH_CALUDE_same_solution_implies_k_value_l2709_270940

theorem same_solution_implies_k_value (x : ℝ) (k : ℝ) : 
  (2 * x - 1 = 3) ∧ (3 * x + k = 0) → k = -6 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_k_value_l2709_270940


namespace NUMINAMATH_CALUDE_museum_survey_visitors_l2709_270911

/-- Represents the survey results of visitors to a modern art museum --/
structure MuseumSurvey where
  total : ℕ
  enjoyed_and_understood : ℕ
  neither_enjoyed_nor_understood : ℕ

/-- The conditions of the survey --/
def survey_conditions (s : MuseumSurvey) : Prop :=
  s.neither_enjoyed_nor_understood = 110 ∧
  s.enjoyed_and_understood = (3 : ℚ) / 4 * s.total

/-- The theorem to be proved --/
theorem museum_survey_visitors (s : MuseumSurvey) :
  survey_conditions s → s.total = 440 := by
  sorry


end NUMINAMATH_CALUDE_museum_survey_visitors_l2709_270911


namespace NUMINAMATH_CALUDE_gas_cost_theorem_l2709_270924

/-- Calculates the total cost of gas for Ryosuke's travels in a day -/
def calculate_gas_cost (trip1_start trip1_end trip2_start trip2_end : ℕ) 
                       (fuel_efficiency : ℚ) (gas_price : ℚ) : ℚ :=
  let total_distance : ℕ := (trip1_end - trip1_start) + (trip2_end - trip2_start)
  let gallons_used : ℚ := total_distance / fuel_efficiency
  let total_cost : ℚ := gallons_used * gas_price
  total_cost

/-- The total cost of gas for Ryosuke's travels is approximately $10.11 -/
theorem gas_cost_theorem : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  |calculate_gas_cost 63102 63135 63135 63166 25 (395/100) - (1011/100)| < ε :=
sorry

end NUMINAMATH_CALUDE_gas_cost_theorem_l2709_270924


namespace NUMINAMATH_CALUDE_convex_polyhedron_volume_relation_l2709_270917

/-- A convex polyhedron with an inscribed sphere -/
structure ConvexPolyhedron where
  volume : ℝ
  surfaceArea : ℝ
  inscribedSphereRadius : ℝ

/-- The relationship between volume, surface area, and inscribed sphere radius for a convex polyhedron -/
theorem convex_polyhedron_volume_relation (P : ConvexPolyhedron) :
  P.volume = (1 / 3) * P.surfaceArea * P.inscribedSphereRadius := by
  sorry

end NUMINAMATH_CALUDE_convex_polyhedron_volume_relation_l2709_270917


namespace NUMINAMATH_CALUDE_line_graph_most_suitable_l2709_270938

/-- Represents types of graphs --/
inductive GraphType
  | Bar
  | Pie
  | Line

/-- Represents a geographical direction --/
inductive Direction
  | West
  | East

/-- Represents the characteristics of terrain elevation --/
structure TerrainElevation where
  higher : Direction
  lower : Direction

/-- Represents the requirement for visual representation --/
structure VisualRepresentation where
  showChanges : Bool
  alongLatitude : Bool

/-- Determines the most suitable graph type for representing elevation changes --/
def mostSuitableGraphType (terrain : TerrainElevation) (requirement : VisualRepresentation) : GraphType :=
  sorry

/-- Theorem stating that a line graph is the most suitable for the given conditions --/
theorem line_graph_most_suitable 
  (terrain : TerrainElevation)
  (requirement : VisualRepresentation)
  (h1 : terrain.higher = Direction.West)
  (h2 : terrain.lower = Direction.East)
  (h3 : requirement.showChanges = true)
  (h4 : requirement.alongLatitude = true) :
  mostSuitableGraphType terrain requirement = GraphType.Line :=
  sorry

end NUMINAMATH_CALUDE_line_graph_most_suitable_l2709_270938


namespace NUMINAMATH_CALUDE_new_student_height_l2709_270964

def original_heights : List ℝ := [145, 139, 155, 160, 143]

def average_increase : ℝ := 1.2

theorem new_student_height :
  let original_sum := original_heights.sum
  let original_count := original_heights.length
  let original_average := original_sum / original_count
  let new_average := original_average + average_increase
  let new_count := original_count + 1
  let new_sum := new_average * new_count
  new_sum - original_sum = 155.6 := by sorry

end NUMINAMATH_CALUDE_new_student_height_l2709_270964


namespace NUMINAMATH_CALUDE_element_in_set_l2709_270936

theorem element_in_set (a b : Type) : a ∈ ({a, b} : Set Type) := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l2709_270936


namespace NUMINAMATH_CALUDE_expansion_properties_l2709_270999

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion term
def expansionTerm (n r : ℕ) (x : ℝ) : ℝ := 
  (binomial n r : ℝ) * (2^(n-r)) * (3^r) * (x^(n - (4/3)*r))

theorem expansion_properties :
  ∃ (n : ℕ) (x : ℝ),
  -- Condition: ratio of binomial coefficients
  (binomial n 2 : ℝ) / (binomial n 1 : ℝ) = 5/2 →
  -- 1. n = 6
  n = 6 ∧
  -- 2. Coefficient of x^2 term
  (∃ (r : ℕ), n - (4/3)*r = 2 ∧ 
    expansionTerm n r 1 = 4320) ∧
  -- 3. Term with maximum coefficient
  (∃ (r : ℕ), ∀ (k : ℕ), 
    expansionTerm n r x ≥ expansionTerm n k x ∧
    expansionTerm n r 1 = 4860 ∧
    n - (4/3)*r = 2/3) :=
sorry

end NUMINAMATH_CALUDE_expansion_properties_l2709_270999


namespace NUMINAMATH_CALUDE_comparison_sqrt_l2709_270921

theorem comparison_sqrt : 2 * Real.sqrt 3 < Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_comparison_sqrt_l2709_270921


namespace NUMINAMATH_CALUDE_exam_girls_count_l2709_270923

theorem exam_girls_count (total : ℕ) (pass_rate_boys : ℚ) (pass_rate_girls : ℚ) (fail_rate_total : ℚ) :
  total = 2000 ∧
  pass_rate_boys = 30 / 100 ∧
  pass_rate_girls = 32 / 100 ∧
  fail_rate_total = 691 / 1000 →
  ∃ (girls : ℕ), girls = 900 ∧ girls ≤ total ∧
    (girls : ℚ) * pass_rate_girls + (total - girls : ℚ) * pass_rate_boys = (1 - fail_rate_total) * total :=
by sorry

end NUMINAMATH_CALUDE_exam_girls_count_l2709_270923


namespace NUMINAMATH_CALUDE_markers_given_l2709_270943

theorem markers_given (initial : ℕ) (total : ℕ) (given : ℕ) : 
  initial = 217 → total = 326 → given = total - initial → given = 109 := by
sorry

end NUMINAMATH_CALUDE_markers_given_l2709_270943


namespace NUMINAMATH_CALUDE_max_value_theorem_l2709_270974

/-- The function f(x) = x^3 + x -/
def f (x : ℝ) : ℝ := x^3 + x

/-- The theorem stating the maximum value of a√(1 + b^2) -/
theorem max_value_theorem (a b : ℝ) (h : f (a^2) + f (2 * b^2 - 3) = 0) :
  ∃ (M : ℝ), M = (5 * Real.sqrt 2) / 4 ∧ a * Real.sqrt (1 + b^2) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2709_270974


namespace NUMINAMATH_CALUDE_triangular_sequence_start_fifteenth_triangular_number_l2709_270941

/-- Triangular number function -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sequence of triangular numbers starts with 1, 3, 6, 10, ... -/
theorem triangular_sequence_start :
  [triangular_number 1, triangular_number 2, triangular_number 3, triangular_number 4] = [1, 3, 6, 10] := by sorry

/-- The 15th triangular number is 120 -/
theorem fifteenth_triangular_number :
  triangular_number 15 = 120 := by sorry

end NUMINAMATH_CALUDE_triangular_sequence_start_fifteenth_triangular_number_l2709_270941


namespace NUMINAMATH_CALUDE_tan_value_fourth_quadrant_l2709_270985

theorem tan_value_fourth_quadrant (α : Real) 
  (h1 : α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) 
  (h2 : Real.sin α + Real.cos α = 1/5) : 
  Real.tan α = -3/4 := by
sorry

end NUMINAMATH_CALUDE_tan_value_fourth_quadrant_l2709_270985


namespace NUMINAMATH_CALUDE_math_team_selection_l2709_270937

theorem math_team_selection (n : ℕ) (k : ℕ) (total : ℕ) :
  n = 10 →
  k = 3 →
  total = 10 →
  (Nat.choose (total - 1) k) - (Nat.choose (total - 3) k) = 49 := by
  sorry

end NUMINAMATH_CALUDE_math_team_selection_l2709_270937


namespace NUMINAMATH_CALUDE_jessy_jewelry_count_l2709_270956

def initial_necklaces : ℕ := 10
def initial_earrings : ℕ := 15
def bought_necklaces : ℕ := 10
def bought_earrings : ℕ := (2 * initial_earrings) / 3
def mother_gift_earrings : ℕ := bought_earrings / 5 + bought_earrings

def total_jewelry : ℕ := initial_necklaces + initial_earrings + bought_necklaces + bought_earrings + mother_gift_earrings

theorem jessy_jewelry_count : total_jewelry = 57 := by
  sorry

end NUMINAMATH_CALUDE_jessy_jewelry_count_l2709_270956


namespace NUMINAMATH_CALUDE_range_of_a_l2709_270930

theorem range_of_a (a : ℝ) : 
  (∀ x, (a - 4 < x ∧ x < a + 4) → (x - 2) * (x - 3) > 0) →
  (a ≤ -2 ∨ a ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2709_270930


namespace NUMINAMATH_CALUDE_triangle_inequality_third_side_bounds_l2709_270914

theorem triangle_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ c : ℝ, 0 < c ∧ a - b < c ∧ c < a + b :=
by
  sorry

theorem third_side_bounds (side1 side2 : ℝ) 
  (h1 : side1 = 6) (h2 : side2 = 10) :
  ∃ side3 : ℝ, 4 < side3 ∧ side3 < 16 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_third_side_bounds_l2709_270914


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l2709_270910

def product : ℕ := 91 * 92 * 93 * 94

theorem distinct_prime_factors_count :
  (Nat.factors product).toFinset.card = 7 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l2709_270910


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2709_270998

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a2 : a 2 = 2)
  (h_sum : 2 * a 3 + a 4 = 16) :
  a 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2709_270998


namespace NUMINAMATH_CALUDE_divisibility_condition_l2709_270992

theorem divisibility_condition (k n : ℕ+) :
  (∃ (p : ℕ), Prime p ∧ p ∣ (4 * k ^ 2 - 1) ^ 2 ∧ p = 8 * k * n - 1) ↔ Even k :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2709_270992


namespace NUMINAMATH_CALUDE_circle_C_equation_l2709_270944

-- Define the circles and line
def circle_M (r : ℝ) (x y : ℝ) : Prop := (x + 2)^2 + (y + 2)^2 = r^2
def line_symmetry (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the symmetry condition
def symmetric_circles (C_center : ℝ × ℝ) (r : ℝ) : Prop :=
  let (a, b) := C_center
  (a - (-2)) / 2 + (b - (-2)) / 2 + 2 = 0 ∧ (b + 2) / (a + 2) = 1

-- Theorem statement
theorem circle_C_equation :
  ∀ (r : ℝ), r > 0 →
  ∃ (C_center : ℝ × ℝ),
    (symmetric_circles C_center r) ∧
    ((1 : ℝ) - C_center.1)^2 + ((1 : ℝ) - C_center.2)^2 = 
    C_center.1^2 + C_center.2^2 →
    ∀ (x y : ℝ), x^2 + y^2 = 2 ↔ 
      ((x - C_center.1)^2 + (y - C_center.2)^2 = C_center.1^2 + C_center.2^2) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_C_equation_l2709_270944


namespace NUMINAMATH_CALUDE_aquarium_height_is_three_l2709_270963

/-- Represents an aquarium with given dimensions and water filling process --/
structure Aquarium where
  length : ℝ
  width : ℝ
  height : ℝ
  initialFillFraction : ℝ
  spillFraction : ℝ
  finalMultiplier : ℝ

/-- Calculates the final volume of water in the aquarium after the described process --/
def finalVolume (a : Aquarium) : ℝ :=
  a.length * a.width * a.height * a.initialFillFraction * (1 - a.spillFraction) * a.finalMultiplier

/-- Theorem stating that an aquarium with the given properties has a height of 3 feet --/
theorem aquarium_height_is_three :
  ∀ (a : Aquarium),
    a.length = 4 →
    a.width = 6 →
    a.initialFillFraction = 1/2 →
    a.spillFraction = 1/2 →
    a.finalMultiplier = 3 →
    finalVolume a = 54 →
    a.height = 3 := by sorry

end NUMINAMATH_CALUDE_aquarium_height_is_three_l2709_270963


namespace NUMINAMATH_CALUDE_max_abs_quadratic_function_bound_l2709_270980

theorem max_abs_quadratic_function_bound (a b : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + a*x + b
  let M := ⨆ (x : ℝ) (hx : x ∈ Set.Icc (-1) 1), |f x|
  M ≥ (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_max_abs_quadratic_function_bound_l2709_270980


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l2709_270965

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = 6 ∧ b = 8) ∨ (a = 6 ∧ c = 8) ∨ (b = 6 ∧ c = 8) →
  (a^2 + b^2 = c^2) →
  c = 10 ∨ c = 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l2709_270965


namespace NUMINAMATH_CALUDE_integer_representation_l2709_270958

theorem integer_representation (N : ℕ+) : 
  ∃ (p q u v : ℤ), (N : ℤ) = p * q + u * v ∧ u - v = 2 * (p - q) := by
  sorry

end NUMINAMATH_CALUDE_integer_representation_l2709_270958


namespace NUMINAMATH_CALUDE_morning_afternoon_difference_l2709_270962

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 44

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 39

/-- The number of campers who went rowing in the evening -/
def evening_campers : ℕ := 31

theorem morning_afternoon_difference :
  morning_campers - afternoon_campers = 5 := by
  sorry

end NUMINAMATH_CALUDE_morning_afternoon_difference_l2709_270962


namespace NUMINAMATH_CALUDE_count_numbers_with_remainder_l2709_270906

theorem count_numbers_with_remainder (n : ℕ) : 
  (Finset.filter (fun N : ℕ => N > 17 ∧ 2017 % N = 17) (Finset.range (2017 + 1))).card = 13 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_remainder_l2709_270906


namespace NUMINAMATH_CALUDE_min_max_cubic_minus_xy_squared_l2709_270988

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := |x^3 - x*y^2|

/-- The theorem statement -/
theorem min_max_cubic_minus_xy_squared :
  (∃ (m : ℝ), ∀ (y : ℝ), m ≤ (⨆ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2), f x y)) ∧
  (∀ (m : ℝ), (∀ (y : ℝ), m ≤ (⨆ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2), f x y)) → 8 ≤ m) :=
sorry

end NUMINAMATH_CALUDE_min_max_cubic_minus_xy_squared_l2709_270988


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l2709_270932

/-- The number of wrapping paper varieties -/
def wrapping_paper_varieties : ℕ := 10

/-- The number of ribbon colors -/
def ribbon_colors : ℕ := 3

/-- The number of gift card types -/
def gift_card_types : ℕ := 4

/-- The number of gift card styles -/
def gift_card_styles : ℕ := 2

/-- The number of decorative bow options -/
def decorative_bow_options : ℕ := 2

/-- Theorem stating the total number of gift wrapping combinations -/
theorem gift_wrapping_combinations :
  wrapping_paper_varieties * ribbon_colors * gift_card_types * gift_card_styles * decorative_bow_options = 480 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l2709_270932


namespace NUMINAMATH_CALUDE_quadratic_equation_1_l2709_270909

theorem quadratic_equation_1 : 
  ∃ x₁ x₂ : ℝ, (x₁ + 1)^2 - 144 = 0 ∧ (x₂ + 1)^2 - 144 = 0 ∧ x₁ ≠ x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_1_l2709_270909


namespace NUMINAMATH_CALUDE_sandy_kim_age_multiple_l2709_270961

/-- Proves that Sandy will be 3 times as old as Kim in two years -/
theorem sandy_kim_age_multiple :
  ∀ (sandy_age kim_age : ℕ) (sandy_bill : ℕ),
    sandy_bill = 10 * sandy_age →
    sandy_bill = 340 →
    kim_age = 10 →
    (sandy_age + 2) / (kim_age + 2) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_kim_age_multiple_l2709_270961


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l2709_270942

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width +  -- bottom area
  2 * length * depth +  -- long sides area
  2 * width * depth  -- short sides area

/-- Theorem: The total wet surface area of a cistern with given dimensions is 62 square meters -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 8 4 1.25 = 62 := by
  sorry


end NUMINAMATH_CALUDE_cistern_wet_surface_area_l2709_270942


namespace NUMINAMATH_CALUDE_max_square_side_length_56_24_l2709_270975

/-- The maximum side length of squares that can be cut from a rectangular paper -/
def max_square_side_length (length width : ℕ) : ℕ := Nat.gcd length width

theorem max_square_side_length_56_24 :
  max_square_side_length 56 24 = 8 := by sorry

end NUMINAMATH_CALUDE_max_square_side_length_56_24_l2709_270975


namespace NUMINAMATH_CALUDE_point_on_line_l2709_270900

/-- Given three points M, N, and P in the 2D plane, where P lies on the line passing through M and N,
    prove that the y-coordinate of P is 2. -/
theorem point_on_line (M N P : ℝ × ℝ) : 
  M = (2, -1) → N = (4, 5) → P.1 = 3 → 
  (P.2 - M.2) / (P.1 - M.1) = (N.2 - M.2) / (N.1 - M.1) → 
  P.2 = 2 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l2709_270900


namespace NUMINAMATH_CALUDE_complex_power_problem_l2709_270947

theorem complex_power_problem (z : ℂ) (i : ℂ) (h1 : i^2 = -1) (h2 : z * (1 - i) = 1 + i) : z^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_problem_l2709_270947


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2709_270907

theorem geometric_sequence_fourth_term 
  (a₁ a₂ a₃ a₄ : ℝ) 
  (h1 : a₁ = 2^(1/4)) 
  (h2 : a₂ = 2^(1/5)) 
  (h3 : a₃ = 2^(1/10)) 
  (h_geometric : ∃ r : ℝ, r ≠ 0 ∧ a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r) : 
  a₄ = 2^(1/10) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2709_270907


namespace NUMINAMATH_CALUDE_special_set_characterization_l2709_270995

/-- The set of integers n ≥ 1 such that 2^n - 1 has exactly n positive integer divisors -/
def special_set : Set ℕ+ :=
  {n | (Nat.card (Nat.divisors ((2:ℕ)^(n:ℕ) - 1))) = n}

/-- Theorem stating that the special set is equal to {1, 2, 4, 6, 8, 16, 32} -/
theorem special_set_characterization :
  special_set = {1, 2, 4, 6, 8, 16, 32} := by sorry

end NUMINAMATH_CALUDE_special_set_characterization_l2709_270995


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l2709_270983

-- Define the sets A and S
def A : Set ℝ := {x | -7 ≤ 2*x - 5 ∧ 2*x - 5 ≤ 9}
def S (k : ℝ) : Set ℝ := {x | k + 1 ≤ x ∧ x ≤ 2*k - 1}

-- Statement 1
theorem subset_condition (k : ℝ) : 
  (S k).Nonempty ∧ S k ⊆ A ↔ 2 ≤ k ∧ k ≤ 4 := by sorry

-- Statement 2
theorem disjoint_condition (k : ℝ) : 
  A ∩ S k = ∅ ↔ k < 2 ∨ k > 6 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l2709_270983


namespace NUMINAMATH_CALUDE_ellipse_C_equation_min_OP_OQ_sum_l2709_270978

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Theorem for the equation of ellipse C
theorem ellipse_C_equation :
  ∀ a b : ℝ, (ellipse_C a b 1 (Real.sqrt 6 / 3)) →
  (∀ x y : ℝ, hyperbola x y ↔ hyperbola x y) →
  (∀ x y : ℝ, ellipse_C a b x y ↔ x^2 / 3 + y^2 = 1) :=
sorry

-- Define a line passing through two points on the ellipse
def line_through_ellipse (a b : ℝ) (x1 y1 x2 y2 : ℝ) : Prop :=
  ellipse_C a b x1 y1 ∧ ellipse_C a b x2 y2

-- Define points P and Q on the x-axis
def point_P (x : ℝ) : Prop := x ≠ 0
def point_Q (x : ℝ) : Prop := x ≠ 0

-- Theorem for the minimum value of |OP| + |OQ|
theorem min_OP_OQ_sum :
  ∀ a b x1 y1 x2 y2 p q : ℝ,
  line_through_ellipse a b x1 y1 x2 y2 →
  point_P p → point_Q q →
  |p| + |q| ≥ 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_C_equation_min_OP_OQ_sum_l2709_270978


namespace NUMINAMATH_CALUDE_ingrid_income_calculation_l2709_270935

-- Define the given constants
def john_tax_rate : ℝ := 0.30
def ingrid_tax_rate : ℝ := 0.40
def john_income : ℝ := 58000
def combined_tax_rate : ℝ := 0.3554

-- Define Ingrid's income as a variable
def ingrid_income : ℝ := 72000

-- Theorem statement
theorem ingrid_income_calculation :
  ingrid_income = 72000 ∧
  (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income) = combined_tax_rate :=
by sorry

end NUMINAMATH_CALUDE_ingrid_income_calculation_l2709_270935


namespace NUMINAMATH_CALUDE_triangle_inequality_cube_l2709_270922

theorem triangle_inequality_cube (a b c : ℝ) 
  (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^3 + b^3 + 3*a*b*c > c^3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_cube_l2709_270922


namespace NUMINAMATH_CALUDE_greatest_five_digit_with_product_90_l2709_270931

/-- A function that returns true if n is a five-digit number -/
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

/-- A function that returns the product of digits of a natural number -/
def digit_product (n : ℕ) : ℕ :=
  sorry

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- The greatest five-digit number whose digits have a product of 90 -/
def M : ℕ :=
  sorry

theorem greatest_five_digit_with_product_90 :
  is_five_digit M ∧
  digit_product M = 90 ∧
  (∀ n : ℕ, is_five_digit n → digit_product n = 90 → n ≤ M) ∧
  digit_sum M = 18 :=
sorry

end NUMINAMATH_CALUDE_greatest_five_digit_with_product_90_l2709_270931


namespace NUMINAMATH_CALUDE_shopkeeper_profit_calculation_l2709_270959

/-- Represents the profit percentage calculation for a shopkeeper's sale --/
theorem shopkeeper_profit_calculation 
  (cost_price : ℝ) 
  (discount_percent : ℝ) 
  (profit_with_discount : ℝ) 
  (h_positive_cp : cost_price > 0)
  (h_discount : discount_percent = 5)
  (h_profit : profit_with_discount = 20.65) :
  let selling_price_with_discount := cost_price * (1 - discount_percent / 100)
  let selling_price_no_discount := cost_price * (1 + profit_with_discount / 100)
  let profit_no_discount := (selling_price_no_discount - cost_price) / cost_price * 100
  profit_no_discount = profit_with_discount := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_calculation_l2709_270959


namespace NUMINAMATH_CALUDE_price_change_percentage_l2709_270970

theorem price_change_percentage (P : ℝ) (x : ℝ) (h : P > 0) :
  P * (1 + x / 100) * (1 - x / 100) = 0.84 * P → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_price_change_percentage_l2709_270970


namespace NUMINAMATH_CALUDE_inequality_proof_l2709_270953

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 
    2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2709_270953


namespace NUMINAMATH_CALUDE_fruit_arrangement_theorem_l2709_270969

def num_apples : ℕ := 4
def num_oranges : ℕ := 3
def num_bananas : ℕ := 2
def total_fruits : ℕ := num_apples + num_oranges + num_bananas

-- Function to calculate the number of ways to arrange fruits
-- with the constraint that not all apples are consecutive
def arrange_fruits (a o b : ℕ) : ℕ := sorry

theorem fruit_arrangement_theorem :
  arrange_fruits num_apples num_oranges num_bananas = 150 := by sorry

end NUMINAMATH_CALUDE_fruit_arrangement_theorem_l2709_270969


namespace NUMINAMATH_CALUDE_p_false_and_q_true_l2709_270915

-- Define proposition p
def p : Prop := ∀ x : ℝ, 2 * x^2 - 2 * x + 1 ≤ 0

-- Define proposition q
def q : Prop := ∃ x : ℝ, Real.sin x + Real.cos x = Real.sqrt 2

-- Theorem stating that p is false and q is true
theorem p_false_and_q_true : ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_p_false_and_q_true_l2709_270915


namespace NUMINAMATH_CALUDE_weight_of_new_person_l2709_270977

/-- Theorem: Weight of new person in group weight change scenario -/
theorem weight_of_new_person
  (n : ℕ)  -- Number of persons in the group
  (w : ℝ)  -- Initial total weight of the group
  (r : ℝ)  -- Weight of the person being replaced
  (d : ℝ)  -- Increase in average weight after replacement
  (h1 : n = 10)  -- There are 10 persons
  (h2 : r = 65)  -- The replaced person weighs 65 kg
  (h3 : d = 3.7)  -- The average weight increases by 3.7 kg
  : ∃ x : ℝ, (w - r + x) / n = w / n + d ∧ x = 102 :=
sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l2709_270977


namespace NUMINAMATH_CALUDE_trig_identity_l2709_270981

theorem trig_identity : 
  (2 * Real.sin (10 * π / 180) - Real.cos (20 * π / 180)) / Real.cos (70 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2709_270981


namespace NUMINAMATH_CALUDE_cubic_function_properties_l2709_270925

-- Define the function f
def f (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

-- Define the derivative of f
def f' (b c : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*b*x + c

theorem cubic_function_properties (b c d : ℝ) :
  (∀ k, (k < 0 ∨ k > 4) → (∃! x, f b c d x = k)) ∧
  (∀ k, (0 < k ∧ k < 4) → (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f b c d x = k ∧ f b c d y = k ∧ f b c d z = k)) →
  (∃ x, f b c d x = 4 ∧ f' b c x = 0) ∧
  (∃ x, f b c d x = 0 ∧ f' b c x = 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l2709_270925


namespace NUMINAMATH_CALUDE_bookstore_inventory_l2709_270968

/-- The number of books acquired by the bookstore. -/
def total_books : ℕ := 1000

/-- The number of books sold on the first day. -/
def first_day_sales : ℕ := total_books / 2

/-- The number of books sold on the second day. -/
def second_day_sales : ℕ := first_day_sales / 2 + first_day_sales + 50

/-- The number of books remaining after both days of sales. -/
def remaining_books : ℕ := 200

/-- Theorem stating that the total number of books is 1000, given the sales conditions. -/
theorem bookstore_inventory :
  total_books = 1000 ∧
  first_day_sales = total_books / 2 ∧
  second_day_sales = first_day_sales / 2 + first_day_sales + 50 ∧
  remaining_books = 200 ∧
  total_books = first_day_sales + second_day_sales + remaining_books :=
by sorry

end NUMINAMATH_CALUDE_bookstore_inventory_l2709_270968


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2709_270954

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | x^2 - 5*x + 4 < 0}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2709_270954


namespace NUMINAMATH_CALUDE_unique_solution_l2709_270948

theorem unique_solution : ∃! (x y : ℕ+), x^(y:ℕ) + 1 = y^(x:ℕ) ∧ 2*(x^(y:ℕ)) = y^(x:ℕ) + 13 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2709_270948


namespace NUMINAMATH_CALUDE_log_10_14_in_terms_of_r_and_s_l2709_270920

theorem log_10_14_in_terms_of_r_and_s (r s : ℝ) 
  (h1 : Real.log 2 / Real.log 9 = r) 
  (h2 : Real.log 7 / Real.log 2 = s) : 
  Real.log 14 / Real.log 10 = (s + 1) / (3 + 1 / (2 * r)) := by
  sorry

end NUMINAMATH_CALUDE_log_10_14_in_terms_of_r_and_s_l2709_270920


namespace NUMINAMATH_CALUDE_prob_same_color_7_8_l2709_270982

/-- The probability of drawing two balls of the same color from a bag containing black and white balls. -/
def prob_same_color (black : ℕ) (white : ℕ) : ℚ :=
  let total := black + white
  let prob_both_black := (black * (black - 1)) / (total * (total - 1))
  let prob_both_white := (white * (white - 1)) / (total * (total - 1))
  prob_both_black + prob_both_white

/-- Theorem stating that the probability of drawing two balls of the same color
    from a bag with 7 black balls and 8 white balls is 7/15. -/
theorem prob_same_color_7_8 :
  prob_same_color 7 8 = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_7_8_l2709_270982


namespace NUMINAMATH_CALUDE_probability_calculation_l2709_270949

theorem probability_calculation (total_students : ℕ) (eliminated : ℕ) (selected : ℕ) 
  (remaining : ℕ) (h1 : total_students = 2006) (h2 : eliminated = 6) 
  (h3 : selected = 50) (h4 : remaining = total_students - eliminated) :
  (eliminated : ℚ) / (total_students : ℚ) = 3 / 1003 ∧ 
  (selected : ℚ) / (remaining : ℚ) = 25 / 1003 := by
  sorry

#check probability_calculation

end NUMINAMATH_CALUDE_probability_calculation_l2709_270949


namespace NUMINAMATH_CALUDE_N_equals_set_l2709_270967

def M (m : ℝ) := {x : ℝ | m * x^2 + 2 * x + m = 0}

def N : Set ℝ := {m : ℝ | ∃! x : ℝ, x ∈ M m}

theorem N_equals_set : N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_N_equals_set_l2709_270967


namespace NUMINAMATH_CALUDE_circle_area_increase_l2709_270934

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_r := 2 * r
  let original_area := π * r^2
  let new_area := π * new_r^2
  (new_area - original_area) / original_area = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l2709_270934


namespace NUMINAMATH_CALUDE_sum_of_consecutive_integers_l2709_270973

theorem sum_of_consecutive_integers (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 30 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_integers_l2709_270973


namespace NUMINAMATH_CALUDE_fruit_store_theorem_l2709_270913

def fruit_problem (total_kg : ℕ) (total_cost : ℕ) 
                  (purchase_price_A purchase_price_B : ℕ)
                  (selling_price_A selling_price_B : ℕ) :=
  ∃ (kg_A kg_B : ℕ),
    -- Total kg constraint
    kg_A + kg_B = total_kg ∧ 
    -- Total cost constraint
    kg_A * purchase_price_A + kg_B * purchase_price_B = total_cost ∧
    -- Specific kg values
    kg_A = 65 ∧ kg_B = 75 ∧
    -- Profit calculation
    (kg_A * (selling_price_A - purchase_price_A) + 
     kg_B * (selling_price_B - purchase_price_B)) = 495

theorem fruit_store_theorem : 
  fruit_problem 140 1000 5 9 8 13 := by
  sorry

end NUMINAMATH_CALUDE_fruit_store_theorem_l2709_270913


namespace NUMINAMATH_CALUDE_rachel_age_2009_l2709_270946

/-- Rachel's age at the end of 2004 -/
def rachel_age_2004 : ℝ := 47.5

/-- Rachel's uncle's age at the end of 2004 -/
def uncle_age_2004 : ℝ := 3 * rachel_age_2004

/-- The sum of Rachel's and her uncle's birth years -/
def birth_years_sum : ℕ := 3818

/-- The year for which we're calculating Rachel's age -/
def target_year : ℕ := 2009

/-- The base year from which we're calculating -/
def base_year : ℕ := 2004

theorem rachel_age_2009 :
  rachel_age_2004 + (target_year - base_year) = 52.5 ∧
  rachel_age_2004 = uncle_age_2004 / 3 ∧
  (base_year - rachel_age_2004) + (base_year - uncle_age_2004) = birth_years_sum :=
by sorry

end NUMINAMATH_CALUDE_rachel_age_2009_l2709_270946


namespace NUMINAMATH_CALUDE_cube_volume_after_removal_l2709_270996

/-- Theorem: Volume of a cube with edge sum 72 cm after removing a 1 cm cube corner -/
theorem cube_volume_after_removal (edge_sum : ℝ) (small_cube_edge : ℝ) : 
  edge_sum = 72 → small_cube_edge = 1 → 
  (edge_sum / 12)^3 - small_cube_edge^3 = 215 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_after_removal_l2709_270996


namespace NUMINAMATH_CALUDE_vector_operation_l2709_270986

/-- Given planar vectors a and b, prove that 1/2a - 3/2b equals (-1,2) -/
theorem vector_operation (a b : ℝ × ℝ) :
  a = (1, 1) →
  b = (1, -1) →
  (1/2 : ℝ) • a - (3/2 : ℝ) • b = (-1, 2) := by
sorry

end NUMINAMATH_CALUDE_vector_operation_l2709_270986


namespace NUMINAMATH_CALUDE_two_cars_problem_l2709_270957

/-- Two cars problem -/
theorem two_cars_problem 
  (distance_between_villages : ℝ) 
  (speed_car_A speed_car_B : ℝ) 
  (target_distance : ℝ) :
  distance_between_villages = 18 →
  speed_car_A = 54 →
  speed_car_B = 36 →
  target_distance = 45 →
  -- Case 1: Cars driving towards each other
  (distance_between_villages + target_distance) / (speed_car_A + speed_car_B) = 0.7 ∧
  -- Case 2a: Cars driving in same direction, faster car behind
  (target_distance + distance_between_villages) / (speed_car_A - speed_car_B) = 3.5 ∧
  -- Case 2b: Cars driving in same direction, faster car ahead
  (target_distance - distance_between_villages) / (speed_car_A - speed_car_B) = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_two_cars_problem_l2709_270957


namespace NUMINAMATH_CALUDE_factorial_difference_quotient_l2709_270993

theorem factorial_difference_quotient : (Nat.factorial 13 - Nat.factorial 12) / Nat.factorial 10 = 1584 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_quotient_l2709_270993


namespace NUMINAMATH_CALUDE_eleventh_term_is_25_l2709_270990

/-- An arithmetic sequence with a given sum of first seven terms and first term -/
structure ArithmeticSequence where
  sum_seven : ℝ  -- Sum of first seven terms
  first_term : ℝ  -- First term
  nth_term : ℕ → ℝ  -- Function to calculate the nth term

/-- The eleventh term of the arithmetic sequence is 25 -/
theorem eleventh_term_is_25 (seq : ArithmeticSequence)
    (h1 : seq.sum_seven = 77)
    (h2 : seq.first_term = 5) :
    seq.nth_term 11 = 25 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_is_25_l2709_270990


namespace NUMINAMATH_CALUDE_max_value_problem_l2709_270933

theorem max_value_problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 12 ∧
  ∃ (a b c : ℝ), (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) = 12 ∧
                  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l2709_270933
