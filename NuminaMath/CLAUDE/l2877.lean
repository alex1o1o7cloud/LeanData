import Mathlib

namespace NUMINAMATH_CALUDE_sun_radius_scientific_notation_l2877_287727

theorem sun_radius_scientific_notation :
  696000 = 6.96 * (10 ^ 5) := by sorry

end NUMINAMATH_CALUDE_sun_radius_scientific_notation_l2877_287727


namespace NUMINAMATH_CALUDE_f_f_zero_l2877_287772

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 3 * x^2 - 4
  else if x = 0 then Real.pi
  else 0

theorem f_f_zero : f (f 0) = 3 * Real.pi^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_f_f_zero_l2877_287772


namespace NUMINAMATH_CALUDE_marbles_left_l2877_287714

theorem marbles_left (total_marbles : ℕ) (num_bags : ℕ) (removed_bags : ℕ) : 
  total_marbles = 28 → 
  num_bags = 4 → 
  removed_bags = 1 → 
  total_marbles % num_bags = 0 → 
  total_marbles - (total_marbles / num_bags * removed_bags) = 21 := by
sorry

end NUMINAMATH_CALUDE_marbles_left_l2877_287714


namespace NUMINAMATH_CALUDE_largest_change_first_digit_l2877_287718

def original_number : ℚ := 0.05123

def change_digit (n : ℚ) (pos : ℕ) (new_digit : ℕ) : ℚ :=
  sorry

theorem largest_change_first_digit :
  ∀ pos : ℕ, pos > 0 → pos ≤ 5 →
    change_digit original_number 1 8 > change_digit original_number pos 8 :=
  sorry

end NUMINAMATH_CALUDE_largest_change_first_digit_l2877_287718


namespace NUMINAMATH_CALUDE_x_axis_symmetry_y_axis_symmetry_l2877_287789

-- Define the region
def region (x y : ℝ) : Prop := abs (x + 2*y) + abs (2*x - y) ≤ 8

-- Theorem: The region is symmetric about the x-axis
theorem x_axis_symmetry :
  ∀ x y : ℝ, region x y ↔ region x (-y) :=
sorry

-- Theorem: The region is symmetric about the y-axis
theorem y_axis_symmetry :
  ∀ x y : ℝ, region x y ↔ region (-x) y :=
sorry

end NUMINAMATH_CALUDE_x_axis_symmetry_y_axis_symmetry_l2877_287789


namespace NUMINAMATH_CALUDE_womans_age_multiple_l2877_287782

theorem womans_age_multiple (W S k : ℕ) : 
  W = k * S + 3 →
  W + S = 84 →
  S = 27 →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_womans_age_multiple_l2877_287782


namespace NUMINAMATH_CALUDE_annulus_area_l2877_287798

/-- An annulus is the region between two concentric circles. -/
structure Annulus where
  b : ℝ
  c : ℝ
  a : ℝ
  h1 : b > c
  h2 : a^2 + c^2 = b^2

/-- The area of an annulus is πa², where a is the length of a line segment
    tangent to the inner circle and extending from the outer circle to the
    point of tangency. -/
theorem annulus_area (ann : Annulus) : Real.pi * ann.a^2 = Real.pi * (ann.b^2 - ann.c^2) := by
  sorry

end NUMINAMATH_CALUDE_annulus_area_l2877_287798


namespace NUMINAMATH_CALUDE_parabola_hyperbola_focus_coincidence_l2877_287703

/-- The value of 'a' for which the focus of the parabola y = ax^2 (a > 0) 
    coincides with one of the foci of the hyperbola y^2 - x^2 = 2 -/
theorem parabola_hyperbola_focus_coincidence (a : ℝ) : 
  a > 0 → 
  (∃ (x y : ℝ), y = a * x^2 ∧ y^2 - x^2 = 2 ∧ 
    ((x = 0 ∧ y = 1 / (4 * a)) ∨ (x = 0 ∧ y = 2) ∨ (x = 0 ∧ y = -2))) → 
  a = 1/8 := by
sorry


end NUMINAMATH_CALUDE_parabola_hyperbola_focus_coincidence_l2877_287703


namespace NUMINAMATH_CALUDE_walking_rate_ratio_l2877_287787

theorem walking_rate_ratio (usual_time faster_time : ℝ) (h1 : usual_time = 28) 
  (h2 : faster_time = usual_time - 4) : 
  (usual_time / faster_time) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_walking_rate_ratio_l2877_287787


namespace NUMINAMATH_CALUDE_third_book_words_l2877_287722

-- Define the given constants
def days : ℕ := 10
def books : ℕ := 3
def reading_speed : ℕ := 100 -- words per hour
def first_book_words : ℕ := 200
def second_book_words : ℕ := 400
def average_reading_time : ℕ := 54 -- minutes per day

-- Define the theorem
theorem third_book_words :
  let total_reading_time : ℕ := days * average_reading_time
  let total_reading_hours : ℕ := total_reading_time / 60
  let total_words : ℕ := total_reading_hours * reading_speed
  let first_two_books_words : ℕ := first_book_words + second_book_words
  total_words - first_two_books_words = 300 := by
  sorry

end NUMINAMATH_CALUDE_third_book_words_l2877_287722


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l2877_287783

theorem difference_of_squares_special_case : (831 : ℤ) * 831 - 830 * 832 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l2877_287783


namespace NUMINAMATH_CALUDE_sector_max_area_l2877_287761

/-- Given a sector with perimeter 20 cm, prove that the area is maximized when the central angle is 2 radians and the maximum area is 25 cm². -/
theorem sector_max_area (r : ℝ) (θ : ℝ) :
  r > 0 →
  r * θ + 2 * r = 20 →
  0 < θ →
  θ ≤ 2 * π →
  (∀ r' θ', r' > 0 → r' * θ' + 2 * r' = 20 → 0 < θ' → θ' ≤ 2 * π → 
    1/2 * r * r * θ ≥ 1/2 * r' * r' * θ') →
  θ = 2 ∧ 1/2 * r * r * θ = 25 :=
sorry

end NUMINAMATH_CALUDE_sector_max_area_l2877_287761


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_angle_l2877_287720

theorem isosceles_right_triangle_angle (a h : ℝ) (θ : Real) : 
  a > 0 → -- leg length is positive
  h > 0 → -- hypotenuse length is positive
  h = a * Real.sqrt 2 → -- Pythagorean theorem for isosceles right triangle
  h^2 = 4 * a * Real.cos θ → -- given condition
  0 < θ ∧ θ < Real.pi / 2 → -- θ is an acute angle
  θ = Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_angle_l2877_287720


namespace NUMINAMATH_CALUDE_figure_100_squares_l2877_287788

def f (n : ℕ) : ℕ := n^3 + 2*n^2 + 2*n + 1

theorem figure_100_squares :
  f 0 = 1 ∧ f 1 = 6 ∧ f 2 = 20 ∧ f 3 = 50 → f 100 = 1020201 := by
  sorry

end NUMINAMATH_CALUDE_figure_100_squares_l2877_287788


namespace NUMINAMATH_CALUDE_line_equation_through_M_intersecting_C_l2877_287762

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  ∃ θ : ℝ, x = -1 + 2 * Real.cos θ ∧ y = 1 + 2 * Real.sin θ

-- Define the point M
def point_M : ℝ × ℝ := (-1, 2)

-- Define a line passing through two points
def line_through_points (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

-- State the theorem
theorem line_equation_through_M_intersecting_C :
  ∀ A B : ℝ × ℝ,
  curve_C A.1 A.2 →
  curve_C B.1 B.2 →
  A ≠ B →
  line_through_points point_M.1 point_M.2 A.1 A.2 B.1 B.2 →
  point_M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  ∃ x y : ℝ,
    (Real.sqrt 15 * x - 5 * y + Real.sqrt 15 + 10 = 0) ∨
    (Real.sqrt 15 * x + 5 * y + Real.sqrt 15 - 10 = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_M_intersecting_C_l2877_287762


namespace NUMINAMATH_CALUDE_super_ball_distance_l2877_287775

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundRatio : ℝ) (bounces : ℕ) : ℝ :=
  -- Definition of total distance calculation
  sorry

/-- Theorem stating the total distance traveled by the ball -/
theorem super_ball_distance :
  let initialHeight : ℝ := 150
  let reboundRatio : ℝ := 2/3
  let bounces : ℕ := 5
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
    |totalDistance initialHeight reboundRatio bounces - 591.67| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_super_ball_distance_l2877_287775


namespace NUMINAMATH_CALUDE_unchanged_100th_is_100_l2877_287729

def is_valid_sequence (s : List ℕ) : Prop :=
  s.length = 1982 ∧ s.toFinset = Finset.range 1983 \ {0}

def swap_adjacent (s : List ℕ) : List ℕ :=
  s.zipWith (λ a b => if a > b then b else a) (s.tail.append [0])

def left_to_right_pass (s : List ℕ) : List ℕ :=
  (s.length - 1).fold (λ _ s' => swap_adjacent s') s

def right_to_left_pass (s : List ℕ) : List ℕ :=
  (left_to_right_pass s.reverse).reverse

def double_pass (s : List ℕ) : List ℕ :=
  right_to_left_pass (left_to_right_pass s)

theorem unchanged_100th_is_100 (s : List ℕ) :
  is_valid_sequence s →
  (double_pass s).nthLe 99 (by sorry) = s.nthLe 99 (by sorry) →
  s.nthLe 99 (by sorry) = 100 := by sorry

end NUMINAMATH_CALUDE_unchanged_100th_is_100_l2877_287729


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2877_287797

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_properties (a b c : ℝ) 
  (h1 : QuadraticFunction a b c 1 = -a/2)
  (h2 : a > 0)
  (h3 : ∀ x, QuadraticFunction a b c x < 1 ↔ 0 < x ∧ x < 3) :
  (QuadraticFunction a b c = fun x ↦ (2/3) * x^2 - 2 * x + 1) ∧
  (∃ x, 0 < x ∧ x < 2 ∧ QuadraticFunction a b c x = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2877_287797


namespace NUMINAMATH_CALUDE_multiplication_subtraction_equality_l2877_287719

theorem multiplication_subtraction_equality : 75 * 3030 - 35 * 3030 = 121200 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_equality_l2877_287719


namespace NUMINAMATH_CALUDE_second_meeting_at_5_4_minutes_l2877_287780

/-- Represents the race scenario between George and Henry --/
structure RaceScenario where
  pool_length : ℝ
  george_start_time : ℝ
  henry_start_time : ℝ
  first_meeting_time : ℝ
  first_meeting_distance : ℝ

/-- Calculates the time of the second meeting given a race scenario --/
def second_meeting_time (scenario : RaceScenario) : ℝ :=
  sorry

/-- The main theorem stating that the second meeting occurs 5.4 minutes after George's start --/
theorem second_meeting_at_5_4_minutes (scenario : RaceScenario) 
  (h1 : scenario.pool_length = 50)
  (h2 : scenario.george_start_time = 0)
  (h3 : scenario.henry_start_time = 1)
  (h4 : scenario.first_meeting_time = 3)
  (h5 : scenario.first_meeting_distance = 25) : 
  second_meeting_time scenario = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_second_meeting_at_5_4_minutes_l2877_287780


namespace NUMINAMATH_CALUDE_expression_factorization_l2877_287745

theorem expression_factorization (a b c : ℝ) :
  (((a^2 + 1) - (b^2 + 1))^3 + ((b^2 + 1) - (c^2 + 1))^3 + ((c^2 + 1) - (a^2 + 1))^3) /
  ((a - b)^3 + (b - c)^3 + (c - a)^3) = (a + b) * (b + c) * (c + a) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l2877_287745


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2877_287700

def second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_number_in_second_quadrant :
  let z : ℂ := -1 + 2*I
  second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2877_287700


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_84_and_n_l2877_287771

theorem greatest_common_divisor_of_84_and_n (n : ℕ) : 
  (∃ (d₁ d₂ d₃ : ℕ), d₁ < d₂ ∧ d₂ < d₃ ∧ 
    {d | d > 0 ∧ d ∣ 84 ∧ d ∣ n} = {d₁, d₂, d₃}) →
  (∃ (d : ℕ), d > 0 ∧ d ∣ 84 ∧ d ∣ n ∧ 
    ∀ (k : ℕ), k > 0 ∧ k ∣ 84 ∧ k ∣ n → k ≤ d) →
  4 = (Nat.gcd 84 n) :=
sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_84_and_n_l2877_287771


namespace NUMINAMATH_CALUDE_distance_traveled_l2877_287701

/-- 
Given a skater's speed and time spent skating, calculate the total distance traveled.
-/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 10) (h2 : time = 8) :
  speed * time = 80 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l2877_287701


namespace NUMINAMATH_CALUDE_water_tank_capacity_l2877_287763

theorem water_tank_capacity (c : ℝ) (h1 : c > 0) :
  (1 / 5 : ℝ) * c + 6 = (1 / 3 : ℝ) * c → c = 45 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l2877_287763


namespace NUMINAMATH_CALUDE_parabola_point_ordinate_l2877_287726

/-- The y-coordinate of a point on the parabola y = 4x^2 that is at a distance of 1 from the focus -/
theorem parabola_point_ordinate : ∀ (x y : ℝ),
  y = 4 * x^2 →  -- Point is on the parabola
  (x - 0)^2 + (y - 1/16)^2 = 1 →  -- Distance from focus is 1
  y = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_ordinate_l2877_287726


namespace NUMINAMATH_CALUDE_students_watching_l2877_287715

theorem students_watching (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 33 → 
  total = boys + girls → 
  (2 * boys + 2 * girls) / 3 = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_students_watching_l2877_287715


namespace NUMINAMATH_CALUDE_unique_solution_l2877_287784

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the addition problem -/
def AdditionProblem (A B C : Digit) : Prop :=
  (C.val * 100 + C.val * 10 + A.val) + (B.val * 100 + 2 * 10 + B.val) = A.val * 100 + 8 * 10 + 8

theorem unique_solution :
  ∃! (A B C : Digit), AdditionProblem A B C ∧ A.val * B.val * C.val = 42 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2877_287784


namespace NUMINAMATH_CALUDE_yarn_length_problem_l2877_287707

theorem yarn_length_problem (green_length red_length total_length : ℕ) : 
  green_length = 156 →
  red_length = 3 * green_length + 8 →
  total_length = green_length + red_length →
  total_length = 632 := by
  sorry

end NUMINAMATH_CALUDE_yarn_length_problem_l2877_287707


namespace NUMINAMATH_CALUDE_line_intersection_with_axes_l2877_287740

/-- A line passing through two given points intersects the x-axis and y-axis at specific points -/
theorem line_intersection_with_axes (x₁ y₁ x₂ y₂ : ℝ) :
  let m : ℝ := (y₂ - y₁) / (x₂ - x₁)
  let b : ℝ := y₁ - m * x₁
  let line : ℝ → ℝ := λ x => m * x + b
  x₁ = 8 ∧ y₁ = 2 ∧ x₂ = 4 ∧ y₂ = 6 →
  (∃ x : ℝ, line x = 0 ∧ x = 10) ∧
  (∃ y : ℝ, line 0 = y ∧ y = 10) :=
by sorry

#check line_intersection_with_axes

end NUMINAMATH_CALUDE_line_intersection_with_axes_l2877_287740


namespace NUMINAMATH_CALUDE_abhay_speed_l2877_287799

theorem abhay_speed (distance : ℝ) (a s : ℝ → ℝ) :
  distance = 30 →
  (∀ x, a x > 0 ∧ s x > 0) →
  (∀ x, distance / (a x) = distance / (s x) + 2) →
  (∀ x, distance / (2 * a x) = distance / (s x) - 1) →
  (∃ x, a x = 5 * Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_abhay_speed_l2877_287799


namespace NUMINAMATH_CALUDE_total_rainfall_three_days_l2877_287779

/-- Calculates the total rainfall over three days given specific conditions --/
theorem total_rainfall_three_days 
  (monday_hours : ℕ) 
  (monday_rate : ℕ) 
  (tuesday_hours : ℕ) 
  (tuesday_rate : ℕ) 
  (wednesday_hours : ℕ) 
  (h_monday : monday_hours = 7 ∧ monday_rate = 1)
  (h_tuesday : tuesday_hours = 4 ∧ tuesday_rate = 2)
  (h_wednesday : wednesday_hours = 2)
  (h_wednesday_rate : wednesday_hours * (2 * tuesday_rate) = 8) :
  monday_hours * monday_rate + 
  tuesday_hours * tuesday_rate + 
  wednesday_hours * (2 * tuesday_rate) = 23 := by
sorry


end NUMINAMATH_CALUDE_total_rainfall_three_days_l2877_287779


namespace NUMINAMATH_CALUDE_complement_equivalence_l2877_287725

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event "at least one item is defective"
def at_least_one_defective : Set Ω :=
  {ω | ω.1 = true ∨ ω.2 = true}

-- Define the event "neither of the items is defective"
def neither_defective : Set Ω :=
  {ω | ω.1 = false ∧ ω.2 = false}

-- Theorem: The complement of "at least one item is defective" 
-- is equivalent to "neither of the items is defective"
theorem complement_equivalence :
  at_least_one_defective.compl = neither_defective :=
sorry

end NUMINAMATH_CALUDE_complement_equivalence_l2877_287725


namespace NUMINAMATH_CALUDE_smallest_number_600_times_prime_divisors_l2877_287790

theorem smallest_number_600_times_prime_divisors :
  ∃ (N : ℕ), N > 1 ∧
  (∀ p : ℕ, Nat.Prime p → p ∣ N → N ≥ 600 * p) ∧
  (∀ M : ℕ, M > 1 → (∀ q : ℕ, Nat.Prime q → q ∣ M → M ≥ 600 * q) → M ≥ N) ∧
  N = 1944 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_600_times_prime_divisors_l2877_287790


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l2877_287709

theorem unique_solution_exponential_equation :
  ∀ x : ℝ, (4 : ℝ)^((9 : ℝ)^x) = (9 : ℝ)^((4 : ℝ)^x) ↔ x = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l2877_287709


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2877_287766

/-- Given a geometric sequence {a_n} with common ratio q and sum of first n terms S_n,
    prove that if q = 2 and S_5 = 1, then S_10 = 33. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- sum formula
  q = 2 →
  S 5 = 1 →
  S 10 = 33 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2877_287766


namespace NUMINAMATH_CALUDE_gensokyo_tennis_club_meeting_day_l2877_287792

/-- The Gensokyo Tennis Club problem -/
theorem gensokyo_tennis_club_meeting_day :
  let total_players : ℕ := 2016
  let total_courts : ℕ := 1008
  let reimu_start : ℕ := 123
  let marisa_start : ℕ := 876
  let winner_move (court : ℕ) : ℕ := if court > 1 then court - 1 else 1
  let loser_move (court : ℕ) : ℕ := if court < total_courts then court + 1 else total_courts
  let reimu_path (day : ℕ) : ℕ := if day < reimu_start then reimu_start - day else 1
  let marisa_path (day : ℕ) : ℕ :=
    if day ≤ (total_courts - marisa_start) then
      marisa_start + day
    else
      total_courts - (day - (total_courts - marisa_start))
  ∃ (n : ℕ), n > 0 ∧ reimu_path n = marisa_path n ∧ 
    ∀ (m : ℕ), m > 0 ∧ m < n → reimu_path m ≠ marisa_path m :=
by
  sorry

end NUMINAMATH_CALUDE_gensokyo_tennis_club_meeting_day_l2877_287792


namespace NUMINAMATH_CALUDE_tan_negative_405_degrees_l2877_287773

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_405_degrees_l2877_287773


namespace NUMINAMATH_CALUDE_triangle_properties_l2877_287768

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.A + t.B + t.C = Real.pi)
  (h2 : t.a * Real.cos t.B = (2 * t.c - t.b) * Real.cos t.A) : 
  t.A = Real.pi / 3 ∧ 
  (∃ (max : Real), max = Real.sqrt 3 ∧ 
    ∀ (x : Real), x = Real.sin t.B + Real.sin t.C → x ≤ max) ∧
  (t.A = t.B ∧ t.B = t.C) := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l2877_287768


namespace NUMINAMATH_CALUDE_specific_polyhedron_volume_l2877_287764

/-- A polyhedron formed by a unit square base and four points above its vertices -/
structure UnitSquarePolyhedron where
  -- Heights of the points above the unit square vertices
  h1 : ℝ
  h2 : ℝ
  h3 : ℝ
  h4 : ℝ

/-- The volume of the UnitSquarePolyhedron -/
def volume (p : UnitSquarePolyhedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific polyhedron is 4.5 -/
theorem specific_polyhedron_volume :
  ∃ (p : UnitSquarePolyhedron),
    p.h1 = 3 ∧ p.h2 = 4 ∧ p.h3 = 6 ∧ p.h4 = 5 ∧
    volume p = 4.5 :=
  sorry

end NUMINAMATH_CALUDE_specific_polyhedron_volume_l2877_287764


namespace NUMINAMATH_CALUDE_maria_berry_purchase_l2877_287754

/-- The number of cartons Maria needs to buy -/
def cartons_to_buy (total_needed : ℕ) (strawberries : ℕ) (blueberries : ℕ) : ℕ :=
  total_needed - (strawberries + blueberries)

/-- Theorem stating that Maria needs to buy 9 more cartons of berries -/
theorem maria_berry_purchase : cartons_to_buy 21 4 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_maria_berry_purchase_l2877_287754


namespace NUMINAMATH_CALUDE_find_A_l2877_287704

theorem find_A : ∀ A : ℕ, (A / 7 = 5) ∧ (A % 7 = 3) → A = 38 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l2877_287704


namespace NUMINAMATH_CALUDE_triangle_not_right_angle_l2877_287781

theorem triangle_not_right_angle (A B C : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0)
  (h4 : A + B + C = 180) (h5 : A / 3 = B / 4) (h6 : A / 3 = C / 5) : 
  ¬(A = 90 ∨ B = 90 ∨ C = 90) := by
sorry

end NUMINAMATH_CALUDE_triangle_not_right_angle_l2877_287781


namespace NUMINAMATH_CALUDE_time_after_1457_minutes_l2877_287769

/-- Represents a time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hValid : hours < 24 ∧ minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  sorry

/-- Converts a number to a 24-hour time -/
def minutesToTime (m : ℕ) : Time :=
  sorry

theorem time_after_1457_minutes :
  let start_time : Time := ⟨3, 0, sorry⟩
  let added_minutes : ℕ := 1457
  let end_time : Time := addMinutes start_time added_minutes
  end_time = ⟨3, 17, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_time_after_1457_minutes_l2877_287769


namespace NUMINAMATH_CALUDE_points_difference_l2877_287774

/-- The number of teams in the tournament -/
def num_teams : ℕ := 6

/-- The number of points awarded for a win -/
def win_points : ℕ := 3

/-- The number of points awarded for a tie -/
def tie_points : ℕ := 1

/-- The number of points awarded for a loss -/
def loss_points : ℕ := 0

/-- The total number of matches in a round-robin tournament -/
def total_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The maximum total points possible in the tournament -/
def max_total_points : ℕ := total_matches num_teams * win_points

/-- The minimum total points possible in the tournament -/
def min_total_points : ℕ := total_matches num_teams * 2 * tie_points

/-- The theorem stating the difference between maximum and minimum total points -/
theorem points_difference :
  max_total_points - min_total_points = 30 := by sorry

end NUMINAMATH_CALUDE_points_difference_l2877_287774


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_3_l2877_287770

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The first line: 3y - a = 9x + 1 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := 3 * y - a = 9 * x + 1

/-- The second line: y - 2 = (2a - 3)x -/
def line2 (a : ℝ) (x y : ℝ) : Prop := y - 2 = (2 * a - 3) * x

theorem parallel_lines_a_equals_3 :
  ∀ a : ℝ, (∀ x y : ℝ, line1 a x y ↔ line2 a x y) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_3_l2877_287770


namespace NUMINAMATH_CALUDE_factorization_equality_l2877_287767

theorem factorization_equality (a b x y : ℝ) :
  a^2 * b * (x - y)^3 - a * b^2 * (y - x)^2 = a * b * (x - y)^2 * (a * x - a * y - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2877_287767


namespace NUMINAMATH_CALUDE_max_sum_squares_l2877_287731

theorem max_sum_squares (m n : ℕ) : 
  m ∈ Finset.range 101 ∧ 
  n ∈ Finset.range 101 ∧ 
  (n^2 - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 10946 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squares_l2877_287731


namespace NUMINAMATH_CALUDE_count_twelve_digit_numbers_with_three_ones_l2877_287743

/-- Recursively defines the count of n-digit numbers with digits 1 or 2 without three consecutive 1's -/
def G : ℕ → ℕ
| 0 => 1  -- Base case for 0 digits (empty string)
| 1 => 2  -- Base case for 1 digit
| 2 => 3  -- Base case for 2 digits
| n + 3 => G (n + 2) + G (n + 1) + G n

/-- The count of 12-digit numbers with all digits 1 or 2 and at least three consecutive 1's -/
def count_with_three_ones : ℕ := 2^12 - G 12

theorem count_twelve_digit_numbers_with_three_ones : 
  count_with_three_ones = 3656 :=
sorry

end NUMINAMATH_CALUDE_count_twelve_digit_numbers_with_three_ones_l2877_287743


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l2877_287738

theorem pizza_toppings_combinations : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l2877_287738


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2877_287741

theorem arithmetic_mean_problem : 
  let sequence1 := List.range 15 |> List.map (λ x => x - 6)
  let sequence2 := List.range 10 |> List.map (λ x => x + 1)
  let combined_sequence := sequence1 ++ sequence2
  let sum := combined_sequence.sum
  let count := combined_sequence.length
  (sum : ℚ) / count = 35 / 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2877_287741


namespace NUMINAMATH_CALUDE_bus_ride_difference_l2877_287706

theorem bus_ride_difference (vince_ride zachary_ride : ℝ) 
  (h1 : vince_ride = 0.62)
  (h2 : zachary_ride = 0.5) :
  vince_ride - zachary_ride = 0.12 := by
sorry

end NUMINAMATH_CALUDE_bus_ride_difference_l2877_287706


namespace NUMINAMATH_CALUDE_simplify_expression_l2877_287734

theorem simplify_expression : (5 + 7 + 3 - 2) / 3 - 1 / 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2877_287734


namespace NUMINAMATH_CALUDE_tax_rate_calculation_l2877_287759

theorem tax_rate_calculation (total_value tax_free_allowance tax_paid : ℝ) : 
  total_value = 1720 →
  tax_free_allowance = 600 →
  tax_paid = 78.4 →
  (tax_paid / (total_value - tax_free_allowance)) * 100 = 7 := by
sorry

end NUMINAMATH_CALUDE_tax_rate_calculation_l2877_287759


namespace NUMINAMATH_CALUDE_rectangle_sides_when_perimeter_equals_area_l2877_287716

theorem rectangle_sides_when_perimeter_equals_area :
  ∀ w l : ℝ,
  w > 0 →
  l = 3 * w →
  2 * (w + l) = w * l →
  w = 8 / 3 ∧ l = 8 := by
sorry

end NUMINAMATH_CALUDE_rectangle_sides_when_perimeter_equals_area_l2877_287716


namespace NUMINAMATH_CALUDE_expression_value_l2877_287713

theorem expression_value (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (x^5 + 3*y^2 + 7) / (x + 4) = 298 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2877_287713


namespace NUMINAMATH_CALUDE_range_of_m_l2877_287730

-- Define a decreasing function on [-1, 1]
def IsDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ [-1, 1] → y ∈ [-1, 1] → x < y → f x > f y

-- Main theorem
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h_decreasing : IsDecreasingOn f)
  (h_inequality : f (m - 1) > f (2 * m - 1)) :
  m ∈ Set.Ioo 0 1 := by
  sorry

#check range_of_m

end NUMINAMATH_CALUDE_range_of_m_l2877_287730


namespace NUMINAMATH_CALUDE_sphere_volume_area_ratio_l2877_287711

theorem sphere_volume_area_ratio (r R : ℝ) (h : r > 0) (H : R > 0) :
  (4 / 3 * Real.pi * r^3) / (4 / 3 * Real.pi * R^3) = 1 / 8 →
  (4 * Real.pi * r^2) / (4 * Real.pi * R^2) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_area_ratio_l2877_287711


namespace NUMINAMATH_CALUDE_balloon_distribution_l2877_287728

/-- Given a total number of balloons and the ratios between different colors,
    calculate the number of balloons for each color. -/
theorem balloon_distribution (total : ℕ) (red_ratio blue_ratio black_ratio : ℕ) 
    (h_total : total = 180)
    (h_red : red_ratio = 3)
    (h_black : black_ratio = 2)
    (h_blue : blue_ratio = 1) :
    ∃ (red blue black : ℕ),
      red = 90 ∧ blue = 30 ∧ black = 60 ∧
      red = red_ratio * blue ∧
      black = black_ratio * blue ∧
      red + blue + black = total :=
by
  sorry

#check balloon_distribution

end NUMINAMATH_CALUDE_balloon_distribution_l2877_287728


namespace NUMINAMATH_CALUDE_bills_average_speed_day2_l2877_287750

/-- Represents the driving scenario of Bill's two-day journey --/
structure DrivingScenario where
  speed_day2 : ℝ  -- Average speed on the second day
  time_day2 : ℝ   -- Time spent driving on the second day
  total_distance : ℝ  -- Total distance driven over two days
  total_time : ℝ      -- Total time spent driving over two days

/-- Defines the conditions of Bill's journey --/
def journey_conditions (s : DrivingScenario) : Prop :=
  s.total_distance = 680 ∧
  s.total_time = 18 ∧
  s.total_distance = (s.speed_day2 + 5) * (s.time_day2 + 2) + s.speed_day2 * s.time_day2 ∧
  s.total_time = (s.time_day2 + 2) + s.time_day2

/-- Theorem stating that given the journey conditions, Bill's average speed on the second day was 35 mph --/
theorem bills_average_speed_day2 (s : DrivingScenario) : 
  journey_conditions s → s.speed_day2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_bills_average_speed_day2_l2877_287750


namespace NUMINAMATH_CALUDE_value_of_a_l2877_287710

theorem value_of_a (a : ℝ) (h1 : a < 0) (h2 : |a| = 3) : a = -3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2877_287710


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2877_287793

theorem simplify_and_evaluate : 
  let x : ℚ := -4
  let y : ℚ := 1/2
  (x + 2*y)^2 - x*(x + 3*y) - 4*y^2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2877_287793


namespace NUMINAMATH_CALUDE_f_composition_at_two_l2877_287786

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -Real.sqrt x
  else (x - 1/x)^4

theorem f_composition_at_two : f (f 2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_at_two_l2877_287786


namespace NUMINAMATH_CALUDE_min_students_with_glasses_and_scarf_l2877_287708

theorem min_students_with_glasses_and_scarf (n : ℕ) 
  (h1 : n > 0)
  (h2 : ∃ k : ℕ, n * 3 = k * 7)
  (h3 : ∃ m : ℕ, n * 5 = m * 6)
  (h4 : ∀ p : ℕ, p > 0 → (∃ q : ℕ, p * 3 = q * 7) → (∃ r : ℕ, p * 5 = r * 6) → p ≥ n) :
  ∃ x : ℕ, x = 11 ∧ 
    x = n * 3 / 7 + n * 5 / 6 - n :=
by sorry

end NUMINAMATH_CALUDE_min_students_with_glasses_and_scarf_l2877_287708


namespace NUMINAMATH_CALUDE_min_red_chips_l2877_287751

theorem min_red_chips (r w b : ℕ) : 
  b ≥ (w : ℚ) / 3 →
  (b : ℚ) ≤ r / 4 →
  w + b ≥ 70 →
  r ≥ 72 ∧ ∀ (r' : ℕ), (∃ (w' b' : ℕ), 
    b' ≥ (w' : ℚ) / 3 ∧
    (b' : ℚ) ≤ r' / 4 ∧
    w' + b' ≥ 70) → 
  r' ≥ 72 := by
sorry

end NUMINAMATH_CALUDE_min_red_chips_l2877_287751


namespace NUMINAMATH_CALUDE_cupcake_cost_split_l2877_287765

theorem cupcake_cost_split (num_cupcakes : ℕ) (cost_per_cupcake : ℚ) (num_people : ℕ) :
  num_cupcakes = 12 →
  cost_per_cupcake = 3/2 →
  num_people = 2 →
  (num_cupcakes : ℚ) * cost_per_cupcake / (num_people : ℚ) = 9 :=
by sorry

end NUMINAMATH_CALUDE_cupcake_cost_split_l2877_287765


namespace NUMINAMATH_CALUDE_factorial_calculation_l2877_287748

-- Define the factorial function
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_calculation :
  (5 * factorial 6 + 30 * factorial 5) / factorial 7 = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l2877_287748


namespace NUMINAMATH_CALUDE_tree_leaves_problem_l2877_287796

/-- The number of leaves remaining after dropping 1/10 of leaves n times -/
def leavesRemaining (initialLeaves : ℕ) (n : ℕ) : ℚ :=
  initialLeaves * (9/10)^n

/-- The proposition that a tree with the given leaf-dropping pattern initially had 311 leaves -/
theorem tree_leaves_problem : ∃ (initialLeaves : ℕ),
  (leavesRemaining initialLeaves 4).num = 204 * (leavesRemaining initialLeaves 4).den ∧
  initialLeaves = 311 := by
  sorry


end NUMINAMATH_CALUDE_tree_leaves_problem_l2877_287796


namespace NUMINAMATH_CALUDE_cantaloupes_total_l2877_287742

/-- The number of cantaloupes grown by Fred -/
def fred_cantaloupes : ℕ := 38

/-- The number of cantaloupes grown by Tim -/
def tim_cantaloupes : ℕ := 44

/-- The total number of cantaloupes grown by Fred and Tim -/
def total_cantaloupes : ℕ := fred_cantaloupes + tim_cantaloupes

theorem cantaloupes_total : total_cantaloupes = 82 := by
  sorry

end NUMINAMATH_CALUDE_cantaloupes_total_l2877_287742


namespace NUMINAMATH_CALUDE_rattlesnake_count_l2877_287747

theorem rattlesnake_count (total_snakes : ℕ) (boa_constrictors : ℕ) : 
  total_snakes = 200 →
  boa_constrictors = 40 →
  total_snakes = boa_constrictors + 3 * boa_constrictors + (total_snakes - (boa_constrictors + 3 * boa_constrictors)) →
  total_snakes - (boa_constrictors + 3 * boa_constrictors) = 40 :=
by sorry

end NUMINAMATH_CALUDE_rattlesnake_count_l2877_287747


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l2877_287795

theorem painted_cube_theorem (n : ℕ) (h1 : n > 2) :
  (6 * (n - 2)^2 = (n - 2)^3) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l2877_287795


namespace NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l2877_287712

theorem probability_at_least_one_multiple_of_four :
  let range := Finset.range 100
  let multiples_of_four := range.filter (λ n => n % 4 = 0)
  let prob_not_multiple := (range.card - multiples_of_four.card : ℚ) / range.card
  1 - prob_not_multiple ^ 2 = 7 / 16 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l2877_287712


namespace NUMINAMATH_CALUDE_circle_division_sum_l2877_287794

/-- The sum of numbers on a circle after n steps of division -/
def circleSum (n : ℕ) : ℕ :=
  2 * 3^n

/-- The process of dividing the circle and summing numbers -/
def divideAndSum : ℕ → ℕ
  | 0 => 2  -- Initial sum: 1 + 1
  | n + 1 => 3 * divideAndSum n

theorem circle_division_sum (n : ℕ) :
  divideAndSum n = circleSum n := by
  sorry

end NUMINAMATH_CALUDE_circle_division_sum_l2877_287794


namespace NUMINAMATH_CALUDE_single_digit_equation_l2877_287717

theorem single_digit_equation (a b : ℕ) : 
  (0 < a ∧ a < 10) →
  (0 < b ∧ b < 10) →
  82 * 10 * a + 7 + 6 * b = 190 →
  a + 2 * b = 6 →
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_single_digit_equation_l2877_287717


namespace NUMINAMATH_CALUDE_rectangle_count_l2877_287756

def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem rectangle_count : 
  choose horizontal_lines 2 * choose vertical_lines 2 = 100 := by sorry

end NUMINAMATH_CALUDE_rectangle_count_l2877_287756


namespace NUMINAMATH_CALUDE_parabola_axis_l2877_287702

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := x^2 = -8*y

/-- The equation of the axis of a parabola -/
def axis_equation (y : ℝ) : Prop := y = 2

/-- Theorem: The axis of the parabola x^2 = -8y is y = 2 -/
theorem parabola_axis :
  (∀ x y : ℝ, parabola_equation x y) →
  (∀ y : ℝ, axis_equation y) :=
sorry

end NUMINAMATH_CALUDE_parabola_axis_l2877_287702


namespace NUMINAMATH_CALUDE_snowflake_weight_scientific_notation_l2877_287705

theorem snowflake_weight_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.00003 = a * 10^n ∧ 1 ≤ a ∧ a < 10 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_snowflake_weight_scientific_notation_l2877_287705


namespace NUMINAMATH_CALUDE_hitAtMostOnce_mutually_exclusive_hitBothTimes_l2877_287735

/-- Represents the outcome of a single shot -/
inductive ShotOutcome
| Hit
| Miss

/-- Represents the outcome of two shots -/
def TwoShotOutcome := (ShotOutcome × ShotOutcome)

/-- The event of hitting the target at most once -/
def hitAtMostOnce (outcome : TwoShotOutcome) : Prop :=
  match outcome with
  | (ShotOutcome.Miss, ShotOutcome.Miss) => True
  | (ShotOutcome.Hit, ShotOutcome.Miss) => True
  | (ShotOutcome.Miss, ShotOutcome.Hit) => True
  | (ShotOutcome.Hit, ShotOutcome.Hit) => False

/-- The event of hitting the target both times -/
def hitBothTimes (outcome : TwoShotOutcome) : Prop :=
  match outcome with
  | (ShotOutcome.Hit, ShotOutcome.Hit) => True
  | _ => False

theorem hitAtMostOnce_mutually_exclusive_hitBothTimes :
  ∀ (outcome : TwoShotOutcome), ¬(hitAtMostOnce outcome ∧ hitBothTimes outcome) :=
by sorry

end NUMINAMATH_CALUDE_hitAtMostOnce_mutually_exclusive_hitBothTimes_l2877_287735


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l2877_287760

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![-2, 4]

theorem vector_difference_magnitude : ‖a - b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l2877_287760


namespace NUMINAMATH_CALUDE_percentage_fraction_equality_l2877_287778

theorem percentage_fraction_equality : 
  (85 / 100 * 40) - (4 / 5 * 25) = 14 := by
sorry

end NUMINAMATH_CALUDE_percentage_fraction_equality_l2877_287778


namespace NUMINAMATH_CALUDE_infinite_primes_l2877_287753

theorem infinite_primes : ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p) → ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_l2877_287753


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2877_287736

theorem arithmetic_geometric_mean_inequality 
  (a b k : ℝ) 
  (h1 : b = k * a) 
  (h2 : k > 0) 
  (h3 : 1 ≤ k) 
  (h4 : k ≤ 3) : 
  ((a + b) / 2)^2 ≥ (Real.sqrt (a * b))^2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2877_287736


namespace NUMINAMATH_CALUDE_eleventh_term_is_320_l2877_287733

/-- A geometric sequence with given 5th and 8th terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  fifth_term : a 5 = 5
  eighth_term : a 8 = 40

/-- The 11th term of the geometric sequence is 320 -/
theorem eleventh_term_is_320 (seq : GeometricSequence) : seq.a 11 = 320 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_is_320_l2877_287733


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2877_287721

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 2}

theorem complement_of_A_in_U :
  (U \ A) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2877_287721


namespace NUMINAMATH_CALUDE_excellent_students_probability_l2877_287724

/-- The probability of selecting exactly 4 excellent students when randomly choosing 7 students from a class of 10 students, where 6 are excellent, is equal to 0.5. -/
theorem excellent_students_probability :
  let total_students : ℕ := 10
  let excellent_students : ℕ := 6
  let selected_students : ℕ := 7
  let target_excellent : ℕ := 4
  (Nat.choose excellent_students target_excellent * Nat.choose (total_students - excellent_students) (selected_students - target_excellent)) / Nat.choose total_students selected_students = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_excellent_students_probability_l2877_287724


namespace NUMINAMATH_CALUDE_m_range_l2877_287758

def p (m : ℝ) : Prop := ∀ x : ℝ, 2^x - m + 1 > 0

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (5 - 2*m)^x < (5 - 2*m)^y

theorem m_range (m : ℝ) (h : p m ∧ q m) : m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2877_287758


namespace NUMINAMATH_CALUDE_six_digit_repeat_gcd_l2877_287737

theorem six_digit_repeat_gcd : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ x : ℕ, 100 ≤ x ∧ x < 1000 → 
    n = Nat.gcd (1000 * x + x) (1000 * (x + 1) + (x + 1))) ∧ 
  n = 1001 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_repeat_gcd_l2877_287737


namespace NUMINAMATH_CALUDE_clock_angle_at_8_30_clock_angle_at_8_30_is_75_l2877_287749

/-- The angle between clock hands at 8:30 -/
theorem clock_angle_at_8_30 : ℝ :=
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute : ℝ := 360 / 60
  let hours : ℝ := 8.5
  let minutes : ℝ := 30
  let hour_hand_angle : ℝ := hours * degrees_per_hour
  let minute_hand_angle : ℝ := minutes * degrees_per_minute
  |hour_hand_angle - minute_hand_angle|

theorem clock_angle_at_8_30_is_75 : clock_angle_at_8_30 = 75 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_8_30_clock_angle_at_8_30_is_75_l2877_287749


namespace NUMINAMATH_CALUDE_sum_reciprocal_products_l2877_287746

theorem sum_reciprocal_products (x y z : ℝ) 
  (sum_eq : x + y + z = 6)
  (sum_products_eq : x*y + y*z + z*x = 11)
  (product_eq : x*y*z = 6) :
  x/(y*z) + y/(z*x) + z/(x*y) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_products_l2877_287746


namespace NUMINAMATH_CALUDE_divisibility_condition_implies_prime_relation_l2877_287752

theorem divisibility_condition_implies_prime_relation (m n : ℕ) : 
  m ≥ 2 → n ≥ 2 → 
  (∀ a : ℕ, a ∈ Finset.range n → (a^n - 1) % m = 0) →
  Nat.Prime m ∧ n = m - 1 := by
sorry

end NUMINAMATH_CALUDE_divisibility_condition_implies_prime_relation_l2877_287752


namespace NUMINAMATH_CALUDE_estimate_fish_population_l2877_287723

/-- Estimates the total number of fish in a lake using the mark and recapture method. -/
theorem estimate_fish_population (m n k : ℕ) (h : k > 0) :
  let estimated_total := m * n / k
  ∃ x : ℚ, x = estimated_total ∧ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_estimate_fish_population_l2877_287723


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2877_287777

-- Define a proposition P to represent the given condition
variable (P : Prop)

-- Define a proposition Q to represent the conclusion
variable (Q : Prop)

-- Theorem stating that P is sufficient but not necessary for Q
theorem sufficient_but_not_necessary : (P → Q) ∧ ¬(Q → P) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2877_287777


namespace NUMINAMATH_CALUDE_percentage_calculation_l2877_287757

theorem percentage_calculation : (1 / 8 / 100 * 160) + 0.5 = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2877_287757


namespace NUMINAMATH_CALUDE_max_value_fraction_l2877_287744

theorem max_value_fraction (x : ℝ) : 
  (4 * x^2 + 12 * x + 19) / (4 * x^2 + 12 * x + 9) ≤ 11 ∧ 
  ∀ ε > 0, ∃ y : ℝ, (4 * y^2 + 12 * y + 19) / (4 * y^2 + 12 * y + 9) > 11 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2877_287744


namespace NUMINAMATH_CALUDE_initial_cookies_l2877_287739

/-- Given that 2 cookies were eaten and 5 cookies remain, prove that the initial number of cookies was 7. -/
theorem initial_cookies (eaten : ℕ) (remaining : ℕ) (h1 : eaten = 2) (h2 : remaining = 5) :
  eaten + remaining = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_cookies_l2877_287739


namespace NUMINAMATH_CALUDE_toothpick_grid_theorem_l2877_287732

/-- Represents a toothpick grid -/
structure ToothpickGrid where
  length : ℕ
  width : ℕ

/-- Calculates the total number of toothpicks in a grid -/
def total_toothpicks (grid : ToothpickGrid) : ℕ :=
  (grid.length + 1) * grid.width + (grid.width + 1) * grid.length

/-- Calculates the area enclosed by a grid -/
def enclosed_area (grid : ToothpickGrid) : ℕ :=
  grid.length * grid.width

theorem toothpick_grid_theorem (grid : ToothpickGrid) 
    (h1 : grid.length = 30) (h2 : grid.width = 50) : 
    total_toothpicks grid = 3080 ∧ enclosed_area grid = 1500 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_grid_theorem_l2877_287732


namespace NUMINAMATH_CALUDE_product_seven_consecutive_divisible_by_ten_l2877_287776

/-- The product of any seven consecutive positive integers is divisible by 10 -/
theorem product_seven_consecutive_divisible_by_ten (n : ℕ) : 
  10 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) := by
sorry

end NUMINAMATH_CALUDE_product_seven_consecutive_divisible_by_ten_l2877_287776


namespace NUMINAMATH_CALUDE_shooting_probability_l2877_287785

/-- The probability of person A hitting the target in a single shot -/
def prob_A : ℚ := 3/4

/-- The probability of person B hitting the target in a single shot -/
def prob_B : ℚ := 4/5

/-- The probability that A has taken two shots when they stop shooting -/
def prob_A_two_shots : ℚ := 19/400

theorem shooting_probability :
  let p1 := (1 - prob_A) * (1 - prob_B) * prob_A
  let p2 := (1 - prob_A) * (1 - prob_B) * (1 - prob_A) * prob_B
  p1 + p2 = prob_A_two_shots := by sorry

end NUMINAMATH_CALUDE_shooting_probability_l2877_287785


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l2877_287791

theorem equation_is_quadratic : ∃ (a b c : ℝ), a ≠ 0 ∧ 
  ∀ x, 3 * (x + 1)^2 = 2 * (x - 2) ↔ a * x^2 + b * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_is_quadratic_l2877_287791


namespace NUMINAMATH_CALUDE_frame_sales_ratio_l2877_287755

/-- Given:
  - Dorothy sells glass frames at half the price of Jemma
  - Jemma sells glass frames at 5 dollars each
  - Jemma sold 400 frames
  - They made 2500 dollars together in total
Prove that the ratio of frames Jemma sold to frames Dorothy sold is 2:1 -/
theorem frame_sales_ratio (jemma_price : ℚ) (jemma_sold : ℕ) (total_revenue : ℚ) 
    (h1 : jemma_price = 5)
    (h2 : jemma_sold = 400)
    (h3 : total_revenue = 2500) : 
  ∃ (dorothy_sold : ℕ), jemma_sold = 2 * dorothy_sold := by
  sorry

#check frame_sales_ratio

end NUMINAMATH_CALUDE_frame_sales_ratio_l2877_287755
