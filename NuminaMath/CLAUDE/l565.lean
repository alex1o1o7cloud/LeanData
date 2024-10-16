import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_properties_l565_56531

-- Define the given hyperbola
def given_hyperbola (x y : ℝ) : Prop := x^2 - 2*y^2 = 2

-- Define the desired hyperbola
def desired_hyperbola (x y : ℝ) : Prop := y^2/2 - x^2/4 = 1

-- Define a function to represent the asymptotes
def asymptote (x y : ℝ) : Prop := y = (Real.sqrt 2 / 2) * x ∨ y = -(Real.sqrt 2 / 2) * x

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y : ℝ, asymptote x y ↔ (∃ k : ℝ, given_hyperbola x y ∧ k ≠ 0 ∧ y = k*x)) ∧
  (∀ x y : ℝ, asymptote x y ↔ (∃ k : ℝ, desired_hyperbola x y ∧ k ≠ 0 ∧ y = k*x)) ∧
  desired_hyperbola 2 (-2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l565_56531


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l565_56515

/-- A predicate that determines if three positive real numbers can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating the triangle inequality for forming a triangle -/
theorem triangle_inequality_theorem (a b c : ℝ) :
  can_form_triangle a b c ↔ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l565_56515


namespace NUMINAMATH_CALUDE_balls_in_boxes_l565_56504

/-- The number of ways to choose 2 boxes out of 4 -/
def choose_empty_boxes : ℕ := 6

/-- The number of ways to place 4 different balls into 2 boxes, with at least one ball in each box -/
def place_balls : ℕ := 14

/-- The total number of ways to place 4 different balls into 4 numbered boxes such that exactly two boxes are empty -/
def total_ways : ℕ := choose_empty_boxes * place_balls

theorem balls_in_boxes :
  total_ways = 84 :=
sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l565_56504


namespace NUMINAMATH_CALUDE_levi_has_five_lemons_l565_56505

/-- The number of lemons each person has -/
structure LemonCounts where
  levi : ℕ
  jayden : ℕ
  eli : ℕ
  ian : ℕ

/-- The conditions of the lemon problem -/
def LemonProblem (counts : LemonCounts) : Prop :=
  counts.jayden = counts.levi + 6 ∧
  counts.jayden * 3 = counts.eli ∧
  counts.eli * 2 = counts.ian ∧
  counts.levi + counts.jayden + counts.eli + counts.ian = 115

/-- Theorem stating that under the given conditions, Levi has 5 lemons -/
theorem levi_has_five_lemons :
  ∃ (counts : LemonCounts), LemonProblem counts ∧ counts.levi = 5 := by
  sorry

end NUMINAMATH_CALUDE_levi_has_five_lemons_l565_56505


namespace NUMINAMATH_CALUDE_right_triangle_square_areas_l565_56513

theorem right_triangle_square_areas : ∀ (A B C : ℝ),
  (A = 6^2) →
  (B = 8^2) →
  (C = 10^2) →
  A + B = C :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_square_areas_l565_56513


namespace NUMINAMATH_CALUDE_quadratic_equation_from_sum_and_difference_l565_56503

theorem quadratic_equation_from_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_abs : |x - y| = 12) : 
  (∀ z : ℝ, z^2 - 10*z - 11 = 0 ↔ z = x ∨ z = y) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_sum_and_difference_l565_56503


namespace NUMINAMATH_CALUDE_greatest_length_of_rope_pieces_l565_56574

theorem greatest_length_of_rope_pieces : Nat.gcd 28 (Nat.gcd 42 70) = 7 := by sorry

end NUMINAMATH_CALUDE_greatest_length_of_rope_pieces_l565_56574


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l565_56535

/-- The trajectory of a point equidistant from a fixed point and a line is a parabola -/
theorem trajectory_is_parabola (x y : ℝ) : 
  (x^2 + (y + 3)^2)^(1/2) = |y - 3| → x^2 = -12*y :=
by sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l565_56535


namespace NUMINAMATH_CALUDE_sarah_savings_l565_56528

def savings_schedule : List ℕ := [5, 5, 5, 5, 10, 10, 10, 10, 20, 20, 20, 20]

theorem sarah_savings : (savings_schedule.sum = 140) := by
  sorry

end NUMINAMATH_CALUDE_sarah_savings_l565_56528


namespace NUMINAMATH_CALUDE_only_valid_numbers_l565_56553

/-- A six-digit number starting with 523 that is divisible by 7, 8, and 9 -/
def validNumber (n : ℕ) : Prop :=
  523000 ≤ n ∧ n < 524000 ∧ 7 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n

/-- The theorem stating that 523656 and 523152 are the only valid numbers -/
theorem only_valid_numbers :
  ∀ n : ℕ, validNumber n ↔ n = 523656 ∨ n = 523152 :=
by sorry

end NUMINAMATH_CALUDE_only_valid_numbers_l565_56553


namespace NUMINAMATH_CALUDE_school_average_gpa_l565_56572

theorem school_average_gpa (gpa_6th : ℝ) (gpa_7th : ℝ) (gpa_8th : ℝ)
  (h1 : gpa_6th = 93)
  (h2 : gpa_7th = gpa_6th + 2)
  (h3 : gpa_8th = 91) :
  (gpa_6th + gpa_7th + gpa_8th) / 3 = 93 := by
  sorry

end NUMINAMATH_CALUDE_school_average_gpa_l565_56572


namespace NUMINAMATH_CALUDE_mountain_hike_l565_56588

theorem mountain_hike (rate_up : ℝ) (time : ℝ) (rate_down_factor : ℝ) :
  rate_up = 8 →
  time = 2 →
  rate_down_factor = 1.5 →
  (rate_up * time) * rate_down_factor = 24 := by
sorry

end NUMINAMATH_CALUDE_mountain_hike_l565_56588


namespace NUMINAMATH_CALUDE_inequality_equivalence_l565_56552

theorem inequality_equivalence (x y : ℝ) : 
  (y - x < Real.sqrt (x^2 + 1)) ↔ (y < x + Real.sqrt (x^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l565_56552


namespace NUMINAMATH_CALUDE_tim_marbles_l565_56559

/-- Given that Fred has 110 blue marbles and 22 times more blue marbles than Tim,
    prove that Tim has 5 blue marbles. -/
theorem tim_marbles (fred_marbles : ℕ) (ratio : ℕ) (h1 : fred_marbles = 110) (h2 : ratio = 22) :
  fred_marbles / ratio = 5 := by
  sorry

end NUMINAMATH_CALUDE_tim_marbles_l565_56559


namespace NUMINAMATH_CALUDE_total_grapes_is_83_l565_56597

/-- The number of grapes in Rob's bowl -/
def robs_grapes : ℕ := 25

/-- The number of grapes in Allie's bowl -/
def allies_grapes : ℕ := robs_grapes + 2

/-- The number of grapes in Allyn's bowl -/
def allyns_grapes : ℕ := allies_grapes + 4

/-- The total number of grapes in all three bowls -/
def total_grapes : ℕ := robs_grapes + allies_grapes + allyns_grapes

theorem total_grapes_is_83 : total_grapes = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_grapes_is_83_l565_56597


namespace NUMINAMATH_CALUDE_square_sum_constant_l565_56599

theorem square_sum_constant (x : ℝ) : (2*x + 3)^2 + 2*(2*x + 3)*(5 - 2*x) + (5 - 2*x)^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_constant_l565_56599


namespace NUMINAMATH_CALUDE_gcd_repeated_digit_numbers_l565_56579

/-- A six-digit integer formed by repeating a positive three-digit integer -/
def repeatedDigitNumber (m : ℕ) : Prop :=
  100 ≤ m ∧ m < 1000 ∧ ∃ n : ℕ, n = 1001 * m

/-- The greatest common divisor of all six-digit integers formed by repeating a positive three-digit integer is 1001 -/
theorem gcd_repeated_digit_numbers :
  ∃ d : ℕ, d > 0 ∧ (∀ n : ℕ, repeatedDigitNumber n → d ∣ n) ∧
  ∀ k : ℕ, k > 0 → (∀ n : ℕ, repeatedDigitNumber n → k ∣ n) → k ∣ d :=
by sorry

end NUMINAMATH_CALUDE_gcd_repeated_digit_numbers_l565_56579


namespace NUMINAMATH_CALUDE_at_least_half_eligible_l565_56540

/-- Represents a team of sailors --/
structure Team where
  heights : List ℝ
  nonempty : heights ≠ []

/-- The median of a list of real numbers --/
def median (l : List ℝ) : ℝ := sorry

/-- The count of elements in a list satisfying a predicate --/
def count_if (l : List ℝ) (p : ℝ → Bool) : ℕ := sorry

theorem at_least_half_eligible (t : Team) (h_median : median t.heights = 167) :
  2 * (count_if t.heights (λ x => x ≤ 168)) ≥ t.heights.length := by sorry

end NUMINAMATH_CALUDE_at_least_half_eligible_l565_56540


namespace NUMINAMATH_CALUDE_farm_tax_percentage_l565_56529

/-- Given a village's farm tax collection and information about Mr. Willam's tax and land,
    prove that the percentage of cultivated land taxed is 12.5%. -/
theorem farm_tax_percentage (total_tax village_tax willam_tax : ℚ) (willam_land_percentage : ℚ) :
  total_tax = 4000 →
  willam_tax = 500 →
  willam_land_percentage = 20833333333333332 / 100000000000000000 →
  (willam_tax / total_tax) * 100 = 125 / 10 :=
by sorry

end NUMINAMATH_CALUDE_farm_tax_percentage_l565_56529


namespace NUMINAMATH_CALUDE_boxwood_trim_charge_l565_56570

/-- Calculates the total charge for trimming boxwoods with various shapes -/
def total_charge (basic_trim_cost sphere_cost pyramid_cost cube_cost : ℚ)
                 (total_boxwoods spheres pyramids cubes : ℕ) : ℚ :=
  basic_trim_cost * total_boxwoods +
  sphere_cost * spheres +
  pyramid_cost * pyramids +
  cube_cost * cubes

/-- Theorem stating the total charge for the given scenario -/
theorem boxwood_trim_charge :
  total_charge 5 15 20 25 30 4 3 2 = 320 := by
  sorry

end NUMINAMATH_CALUDE_boxwood_trim_charge_l565_56570


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_fourths_l565_56583

theorem sin_thirteen_pi_fourths : Real.sin (13 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_fourths_l565_56583


namespace NUMINAMATH_CALUDE_notebook_cost_l565_56545

theorem notebook_cost (notebook_cost pencil_cost : ℝ) 
  (h1 : notebook_cost + pencil_cost = 2.20)
  (h2 : notebook_cost = pencil_cost + 2) : 
  notebook_cost = 2.10 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l565_56545


namespace NUMINAMATH_CALUDE_max_n_is_26_l565_56571

/-- The number of non-congruent trapezoids formed by four points out of n equally spaced points on a circle's circumference -/
def num_trapezoids (n : ℕ) : ℕ := sorry

/-- The maximum value of n such that the number of non-congruent trapezoids is no more than 2012 -/
def max_n : ℕ := sorry

theorem max_n_is_26 :
  (∀ n : ℕ, n > 0 → num_trapezoids n ≤ 2012) ∧
  (∀ m : ℕ, m > max_n → num_trapezoids m > 2012) ∧
  max_n = 26 := by sorry

end NUMINAMATH_CALUDE_max_n_is_26_l565_56571


namespace NUMINAMATH_CALUDE_truncated_cone_radius_l565_56586

/-- Represents a cone with its base radius -/
structure Cone :=
  (radius : ℝ)

/-- Represents a truncated cone with its smaller base radius -/
structure TruncatedCone :=
  (smallerRadius : ℝ)

/-- 
  Given three touching cones and a truncated cone sharing a common generatrix with each,
  the radius of the smaller base of the truncated cone is 6.
-/
theorem truncated_cone_radius 
  (cone1 cone2 cone3 : Cone) 
  (truncCone : TruncatedCone) 
  (h1 : cone1.radius = 23) 
  (h2 : cone2.radius = 46) 
  (h3 : cone3.radius = 69) 
  (h4 : ∃ (x y : ℝ), 
    (x^2 + y^2 = (cone1.radius + truncCone.smallerRadius)^2) ∧ 
    ((x - (cone1.radius + cone2.radius))^2 + y^2 = (cone2.radius + truncCone.smallerRadius)^2) ∧
    (x^2 + (y - (cone1.radius + cone3.radius))^2 = (cone3.radius + truncCone.smallerRadius)^2)) :
  truncCone.smallerRadius = 6 := by
  sorry

end NUMINAMATH_CALUDE_truncated_cone_radius_l565_56586


namespace NUMINAMATH_CALUDE_mike_remaining_nickels_l565_56523

/-- Given Mike's initial number of nickels and the number of nickels his dad borrowed,
    calculate the number of nickels Mike has left. -/
def nickels_remaining (initial : ℕ) (borrowed : ℕ) : ℕ :=
  initial - borrowed

/-- Theorem stating that Mike has 12 nickels remaining after his dad's borrowing. -/
theorem mike_remaining_nickels :
  nickels_remaining 87 75 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mike_remaining_nickels_l565_56523


namespace NUMINAMATH_CALUDE_min_colors_for_distribution_centers_l565_56568

def combinations (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem min_colors_for_distribution_centers : 
  (∃ (n : ℕ), n ≥ 6 ∧ combinations n 3 ≥ 20) ∧
  (∀ (m : ℕ), m < 6 → combinations m 3 < 20) := by
  sorry

end NUMINAMATH_CALUDE_min_colors_for_distribution_centers_l565_56568


namespace NUMINAMATH_CALUDE_total_dogs_l565_56539

theorem total_dogs (brown : ℕ) (white : ℕ) (black : ℕ)
  (h1 : brown = 20)
  (h2 : white = 10)
  (h3 : black = 15) :
  brown + white + black = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_dogs_l565_56539


namespace NUMINAMATH_CALUDE_perfect_square_expression_l565_56537

theorem perfect_square_expression : ∃ x : ℝ, (11.98 * 11.98 + 11.98 * 0.04 + 0.02 * 0.02) = x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_expression_l565_56537


namespace NUMINAMATH_CALUDE_square_from_equation_l565_56567

theorem square_from_equation (x y z : ℕ) 
  (h : x^2 + y^2 + z^2 = 2*(x*y + y*z + z*x)) :
  ∃ (a b c : ℕ), x = a^2 ∧ y = b^2 ∧ z = c^2 := by
  sorry

end NUMINAMATH_CALUDE_square_from_equation_l565_56567


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_plus_square_nonnegative_l565_56562

theorem negation_of_absolute_value_plus_square_nonnegative :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x : ℝ, |x| + x^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_plus_square_nonnegative_l565_56562


namespace NUMINAMATH_CALUDE_min_triangle_area_l565_56573

/-- Triangle ABC with A at origin, B at (30, 18), and C with integer coordinates -/
structure Triangle :=
  (p : ℤ)
  (q : ℤ)

/-- Calculate the area of the triangle using the Shoelace formula -/
def triangleArea (t : Triangle) : ℚ :=
  (1 / 2 : ℚ) * |30 * t.q - 18 * t.p|

/-- Theorem: The minimum area of triangle ABC is 3 -/
theorem min_triangle_area :
  ∃ (t : Triangle), ∀ (t' : Triangle), triangleArea t ≤ triangleArea t' ∧ triangleArea t = 3 :=
sorry

end NUMINAMATH_CALUDE_min_triangle_area_l565_56573


namespace NUMINAMATH_CALUDE_unique_subset_existence_l565_56542

theorem unique_subset_existence : 
  ∃ (X : Set ℤ), ∀ (n : ℤ), ∃! (pair : ℤ × ℤ), 
    pair.1 ∈ X ∧ pair.2 ∈ X ∧ pair.1 + 2 * pair.2 = n := by
  sorry

end NUMINAMATH_CALUDE_unique_subset_existence_l565_56542


namespace NUMINAMATH_CALUDE_right_triangle_complex_roots_l565_56560

theorem right_triangle_complex_roots : 
  ∃! (S : Finset ℂ), 
    (∀ z ∈ S, z ≠ 0 ∧ 
      (z.re * (z^6 - z).re + z.im * (z^6 - z).im = 0)) ∧ 
    S.card = 5 := by sorry

end NUMINAMATH_CALUDE_right_triangle_complex_roots_l565_56560


namespace NUMINAMATH_CALUDE_new_year_after_10_years_l565_56514

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year in the 21st century -/
structure Year21stCentury where
  year : Nat
  is_21st_century : 2001 ≤ year ∧ year ≤ 2100

/-- Function to determine if a year is a leap year -/
def isLeapYear (y : Year21stCentury) : Bool :=
  y.year % 4 = 0 && (y.year % 100 ≠ 0 || y.year % 400 = 0)

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

/-- Theorem stating that New Year's Day 10 years after a Friday is a Thursday -/
theorem new_year_after_10_years 
  (start_year : Year21stCentury)
  (h1 : DayOfWeek.Friday = advanceDays DayOfWeek.Friday 0)  -- New Year's Day is Friday in start_year
  (h2 : ∀ d : DayOfWeek, (advanceDays d (5 * 365 + 2)) = d) -- All days occur equally often in 5 years
  : DayOfWeek.Thursday = advanceDays DayOfWeek.Friday (10 * 365 + 3) :=
by sorry


end NUMINAMATH_CALUDE_new_year_after_10_years_l565_56514


namespace NUMINAMATH_CALUDE_upstream_speed_l565_56534

/-- 
Given a man's rowing speed in still water and his speed downstream, 
this theorem proves that his speed upstream can be calculated as the 
difference between his speed in still water and half the difference 
between his downstream speed and his speed in still water.
-/
theorem upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still > 0) 
  (h2 : speed_downstream > speed_still) : 
  speed_still - (speed_downstream - speed_still) / 2 = 
  speed_still - (speed_downstream - speed_still) / 2 :=
by sorry

end NUMINAMATH_CALUDE_upstream_speed_l565_56534


namespace NUMINAMATH_CALUDE_divisible_by_72_digits_l565_56578

theorem divisible_by_72_digits (a b : Nat) : 
  a < 10 → b < 10 → 
  (42000 + 1000 * a + 40 + b) % 72 = 0 → 
  ((a = 8 ∧ b = 0) ∨ (a = 0 ∧ b = 8)) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_72_digits_l565_56578


namespace NUMINAMATH_CALUDE_income_comparison_l565_56536

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : mary = 1.5 * tim) 
  (h2 : tim = 0.6 * juan) : 
  mary = 0.9 * juan := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l565_56536


namespace NUMINAMATH_CALUDE_sequence_squares_l565_56557

theorem sequence_squares (n : ℕ) : 
  let a : ℕ → ℕ := λ k => k^2
  (a 1 = 1) ∧ (a 2 = 4) ∧ (a 3 = 9) ∧ (a 4 = 16) ∧ (a 5 = 25) := by
  sorry

end NUMINAMATH_CALUDE_sequence_squares_l565_56557


namespace NUMINAMATH_CALUDE_circular_track_length_l565_56532

/-- The length of a circular track given cycling conditions. -/
theorem circular_track_length
  (ivanov_initial_speed : ℝ)
  (petrov_speed : ℝ)
  (track_length : ℝ)
  (h1 : 2 * ivanov_initial_speed - 2 * petrov_speed = 3 * track_length)
  (h2 : 3 * ivanov_initial_speed + 10 - 3 * petrov_speed = 7 * track_length) :
  track_length = 4 := by
sorry

end NUMINAMATH_CALUDE_circular_track_length_l565_56532


namespace NUMINAMATH_CALUDE_star_equation_equiv_two_distinct_real_roots_l565_56555

/-- The star operation defined as m ☆ n = mn² - mn - 1 -/
def star (m n : ℝ) : ℝ := m * n^2 - m * n - 1

/-- The equation 1 ☆ x = 0 is equivalent to x² - x - 1 = 0 -/
theorem star_equation_equiv (x : ℝ) : star 1 x = 0 ↔ x^2 - x - 1 = 0 := by sorry

/-- The equation x² - x - 1 = 0 has two distinct real roots -/
theorem two_distinct_real_roots :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ r₁^2 - r₁ - 1 = 0 ∧ r₂^2 - r₂ - 1 = 0 := by sorry

end NUMINAMATH_CALUDE_star_equation_equiv_two_distinct_real_roots_l565_56555


namespace NUMINAMATH_CALUDE_job_completion_time_l565_56561

/-- Given that A can complete a job in 10 hours alone and A and D together can complete a job in 5 hours, prove that D can complete the job in 10 hours alone. -/
theorem job_completion_time (a_time : ℝ) (ad_time : ℝ) (d_time : ℝ) 
    (ha : a_time = 10) 
    (had : ad_time = 5) : 
  d_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l565_56561


namespace NUMINAMATH_CALUDE_sammy_has_eight_caps_l565_56591

/-- The number of bottle caps Billie has -/
def billies_caps : ℕ := 2

/-- The number of bottle caps Janine has -/
def janines_caps : ℕ := 3 * billies_caps

/-- The number of bottle caps Sammy has -/
def sammys_caps : ℕ := janines_caps + 2

/-- Theorem stating that Sammy has 8 bottle caps -/
theorem sammy_has_eight_caps : sammys_caps = 8 := by
  sorry

end NUMINAMATH_CALUDE_sammy_has_eight_caps_l565_56591


namespace NUMINAMATH_CALUDE_sequence_properties_l565_56502

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

-- Define the geometric sequence
def geometric_sequence (b : ℕ → ℝ) := ∀ n m, b (n + m) = b n * b m

theorem sequence_properties 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a_cond : 2 * a 5 - a 3 = 3)
  (h_b_2 : b 2 = 1)
  (h_b_4 : b 4 = 4) :
  (a 7 = 3) ∧ 
  ((b 3 = 2 ∨ b 3 = -2) ∧ b 6 = 16) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l565_56502


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l565_56554

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * r^(n - 1)

theorem fifth_term_of_geometric_sequence
  (a₁ a₂ : ℚ)
  (h₁ : a₁ = 2)
  (h₂ : a₂ = 1/4)
  (h₃ : a₂ = a₁ * (a₂ / a₁)) :
  geometric_sequence a₁ (a₂ / a₁) 5 = 1/2048 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l565_56554


namespace NUMINAMATH_CALUDE_fishbowl_count_l565_56594

theorem fishbowl_count (fish_per_bowl : ℕ) (total_fish : ℕ) (h1 : fish_per_bowl = 23) (h2 : total_fish = 6003) :
  total_fish / fish_per_bowl = 261 := by
  sorry

end NUMINAMATH_CALUDE_fishbowl_count_l565_56594


namespace NUMINAMATH_CALUDE_algebraic_simplification_l565_56576

theorem algebraic_simplification (a b : ℝ) : 3 * a * b - 2 * a * b = a * b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l565_56576


namespace NUMINAMATH_CALUDE_saucer_area_l565_56550

/-- The area of a circular saucer with radius 3 centimeters is 9π square centimeters. -/
theorem saucer_area (π : ℝ) (h : π > 0) : 
  let r : ℝ := 3
  let area : ℝ := π * r^2
  area = 9 * π := by sorry

end NUMINAMATH_CALUDE_saucer_area_l565_56550


namespace NUMINAMATH_CALUDE_horners_method_operations_l565_56500

/-- The polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x + 1 -/
def f (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1

/-- Horner's method representation of the polynomial -/
def horner_f (x : ℝ) : ℝ := ((((5*x + 4)*x + 3)*x + 2)*x + 1)*x + 1

/-- The number of multiplication operations in Horner's method for this polynomial -/
def mult_ops : ℕ := 5

/-- The number of addition operations in Horner's method for this polynomial -/
def add_ops : ℕ := 5

theorem horners_method_operations :
  f 5 = horner_f 5 ∧ mult_ops = 5 ∧ add_ops = 5 := by sorry

end NUMINAMATH_CALUDE_horners_method_operations_l565_56500


namespace NUMINAMATH_CALUDE_triangle_inequality_l565_56512

/-- Proves that for any triangle with side lengths a, b, c, and area S,
    the inequality (ab + ac + bc) / (4S) ≥ √3 holds true. -/
theorem triangle_inequality (a b c S : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ S > 0) 
    (h_triangle : S = 1/4 * Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))) :
  (a * b + a * c + b * c) / (4 * S) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l565_56512


namespace NUMINAMATH_CALUDE_attendance_proof_l565_56593

/-- Calculates the total attendance given the number of adults and children -/
def total_attendance (adults : ℕ) (children : ℕ) : ℕ :=
  adults + children

/-- Theorem: The total attendance for 280 adults and 120 children is 400 -/
theorem attendance_proof :
  total_attendance 280 120 = 400 := by
  sorry

end NUMINAMATH_CALUDE_attendance_proof_l565_56593


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l565_56520

/-- A parabola of the form y = ax^2 + 6 is tangent to the line y = 2x + 4 if and only if a = 1/2 -/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃ x : ℝ, ax^2 + 6 = 2*x + 4 ∧ 
   ∀ y : ℝ, y ≠ x → ay^2 + 6 ≠ 2*y + 4) ↔ 
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l565_56520


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_squares_l565_56546

theorem product_of_sum_and_sum_of_squares (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^2 + b^2 = 6) : 
  a * b = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_squares_l565_56546


namespace NUMINAMATH_CALUDE_equation_solution_l565_56533

theorem equation_solution (x : ℝ) : 
  (x / 5) / 3 = 5 / (x / 3) → x = 15 ∨ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l565_56533


namespace NUMINAMATH_CALUDE_derivative_of_f_l565_56543

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- State the theorem
theorem derivative_of_f :
  deriv f = λ x => 2 * x - 1 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l565_56543


namespace NUMINAMATH_CALUDE_range_of_a_l565_56507

/-- Proposition p: The real number x satisfies x^2 - 4ax + 3a^2 < 0 -/
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

/-- Proposition q: The real number x satisfies x^2 - x - 6 ≤ 0 and x^2 + 2x - 8 > 0 -/
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

/-- ¬p is a necessary but not sufficient condition for ¬q -/
def not_p_necessary_not_sufficient_for_not_q (a : ℝ) : Prop :=
  (∀ x, q x → p x a) ∧ ∃ x, ¬(q x) ∧ p x a

theorem range_of_a :
  ∀ a : ℝ, a > 0 → not_p_necessary_not_sufficient_for_not_q a → 1 < a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l565_56507


namespace NUMINAMATH_CALUDE_project_completion_time_l565_56526

/-- Represents the completion of an engineering project --/
structure Project where
  initialWorkers : ℕ
  initialWorkCompleted : ℚ
  initialDuration : ℕ
  additionalWorkers : ℕ

/-- Calculates the total days required to complete the project --/
def totalDays (p : Project) : ℕ :=
  let totalWorkers := p.initialWorkers + p.additionalWorkers
  let remainingWork := 1 - p.initialWorkCompleted
  let initialWorkRate := p.initialWorkCompleted / p.initialDuration
  let totalWorkRate := initialWorkRate * totalWorkers / p.initialWorkers
  p.initialDuration + (remainingWork / totalWorkRate).ceil.toNat

/-- Theorem stating that for the given project parameters, the total days to complete is 70 --/
theorem project_completion_time (p : Project) 
  (h1 : p.initialWorkers = 6)
  (h2 : p.initialWorkCompleted = 1/3)
  (h3 : p.initialDuration = 35)
  (h4 : p.additionalWorkers = 6) :
  totalDays p = 70 := by
  sorry

#eval totalDays { initialWorkers := 6, initialWorkCompleted := 1/3, initialDuration := 35, additionalWorkers := 6 }

end NUMINAMATH_CALUDE_project_completion_time_l565_56526


namespace NUMINAMATH_CALUDE_pyramid_numbers_l565_56524

theorem pyramid_numbers (a b : ℕ) : 
  (42 = a * 6) → 
  (72 = 6 * b) → 
  (504 = 42 * 72) → 
  (a = 7 ∧ b = 12) := by
sorry

end NUMINAMATH_CALUDE_pyramid_numbers_l565_56524


namespace NUMINAMATH_CALUDE_quadratic_ratio_l565_56521

/-- Given a quadratic function f(x) = ax² + bx + c where a > 0,
    if the solution set of f(x) > 0 is (-∞, -2) ∪ (-1, +∞),
    then the ratio a:b:c is 1:3:2 -/
theorem quadratic_ratio (a b c : ℝ) : 
  a > 0 → 
  (∀ x, a * x^2 + b * x + c > 0 ↔ x < -2 ∨ x > -1) → 
  ∃ (k : ℝ), k ≠ 0 ∧ a = k ∧ b = 3*k ∧ c = 2*k :=
sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l565_56521


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l565_56598

theorem geometric_sequence_fifth_term 
  (t : ℕ → ℝ) 
  (h_geometric : ∃ (a r : ℝ), ∀ n, t n = a * r^(n-1))
  (h_t1 : t 1 = 3)
  (h_t2 : t 2 = 6) :
  t 5 = 48 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l565_56598


namespace NUMINAMATH_CALUDE_gcf_72_108_l565_56519

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcf_72_108_l565_56519


namespace NUMINAMATH_CALUDE_polar_coords_of_negative_two_plus_two_i_l565_56548

/-- The polar coordinates of a complex number z = -(2+2i) -/
theorem polar_coords_of_negative_two_plus_two_i :
  ∃ (r : ℝ) (θ : ℝ) (k : ℤ),
    r = 2 * Real.sqrt 2 ∧
    θ = 5 * Real.pi / 4 + 2 * k * Real.pi ∧
    Complex.exp (θ * Complex.I) * r = -(2 + 2 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_polar_coords_of_negative_two_plus_two_i_l565_56548


namespace NUMINAMATH_CALUDE_operation_five_times_on_1024_l565_56508

/-- The operation that keeps only the numbers at positions of the form 4k + 3 in a sequence -/
def keepOperation (s : List ℕ) : List ℕ :=
  (s.enum.filter (fun (i, _) => i % 4 = 3)).map Prod.snd

/-- Applies the keepOperation n times to the given sequence -/
def applyOperationNTimes (s : List ℕ) (n : ℕ) : List ℕ :=
  match n with
  | 0 => s
  | m + 1 => applyOperationNTimes (keepOperation s) m

theorem operation_five_times_on_1024 :
  applyOperationNTimes (List.range 1024) 5 = [1023] := by
  sorry

end NUMINAMATH_CALUDE_operation_five_times_on_1024_l565_56508


namespace NUMINAMATH_CALUDE_traffic_light_probabilities_l565_56582

/-- Represents the duration of each traffic light state in seconds -/
structure TrafficLightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total duration of a traffic light cycle -/
def cycle_duration (d : TrafficLightDuration) : ℕ :=
  d.red + d.yellow + d.green

/-- Calculates the probability of encountering a specific light state -/
def light_probability (duration : ℕ) (total : ℕ) : ℚ :=
  ↑duration / ↑total

/-- Theorem stating the probabilities of encountering each traffic light state -/
theorem traffic_light_probabilities (d : TrafficLightDuration)
  (h_red : d.red = 40)
  (h_yellow : d.yellow = 5)
  (h_green : d.green = 50) :
  let total := cycle_duration d
  (light_probability d.red total = 8 / 19) ∧
  (light_probability d.yellow total = 1 / 19) ∧
  (light_probability (d.yellow + d.green) total = 11 / 19) := by
  sorry


end NUMINAMATH_CALUDE_traffic_light_probabilities_l565_56582


namespace NUMINAMATH_CALUDE_negative_multiplication_result_l565_56509

theorem negative_multiplication_result : (-4 : ℚ) * (-3/2 : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_multiplication_result_l565_56509


namespace NUMINAMATH_CALUDE_inscribed_triangle_inequality_l565_56510

/-- A triangle inscribed in a circle -/
structure InscribedTriangle :=
  (A B C : ℝ × ℝ)  -- Vertices of the triangle
  (center : ℝ × ℝ)  -- Center of the circumscribed circle
  (radius : ℝ)  -- Radius of the circumscribed circle

/-- Ratio of internal angle bisector to its extension -/
def angle_bisector_ratio (t : InscribedTriangle) (v : ℝ × ℝ) : ℝ := sorry

/-- Sine of an angle in the triangle -/
def triangle_angle_sin (t : InscribedTriangle) (v : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem inscribed_triangle_inequality (t : InscribedTriangle) :
  let l_a := angle_bisector_ratio t t.A
  let l_b := angle_bisector_ratio t t.B
  let l_c := angle_bisector_ratio t t.C
  let sin_A := triangle_angle_sin t t.A
  let sin_B := triangle_angle_sin t t.B
  let sin_C := triangle_angle_sin t t.C
  l_a / (sin_A * sin_A) + l_b / (sin_B * sin_B) + l_c / (sin_C * sin_C) ≥ 3 ∧
  (l_a / (sin_A * sin_A) + l_b / (sin_B * sin_B) + l_c / (sin_C * sin_C) = 3 ↔ 
   t.A = t.B ∧ t.B = t.C) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_triangle_inequality_l565_56510


namespace NUMINAMATH_CALUDE_max_q_plus_2r_l565_56516

theorem max_q_plus_2r (q r : ℕ+) (h : 1230 = 28 * q + r) : 
  (∀ q' r' : ℕ+, 1230 = 28 * q' + r' → q' + 2 * r' ≤ q + 2 * r) ∧ q + 2 * r = 95 := by
  sorry

end NUMINAMATH_CALUDE_max_q_plus_2r_l565_56516


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l565_56585

-- System 1
theorem system_one_solution :
  ∃ (x y : ℚ), 3 * x + 2 * y = 8 ∧ y = 2 * x - 3 ∧ x = 2 ∧ y = 1 := by sorry

-- System 2
theorem system_two_solution :
  ∃ (x y : ℚ), 2 * x + 3 * y = 6 ∧ 3 * x - 2 * y = -2 ∧ x = 6/13 ∧ y = 22/13 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l565_56585


namespace NUMINAMATH_CALUDE_a_value_l565_56575

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2

-- Define the derivative of f
def f_prime (a : ℝ) (x : ℝ) : ℝ := 2 * a * x

-- Theorem statement
theorem a_value (a : ℝ) : f_prime a 1 = 4 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l565_56575


namespace NUMINAMATH_CALUDE_actual_number_is_two_l565_56511

-- Define the set of people
inductive Person
| Natasha
| Boy1
| Boy2
| Girl1
| Girl2

-- Define a function to represent claims about the number
def claim (p : Person) (n : Nat) : Prop :=
  match p with
  | Person.Natasha => n % 15 = 0
  | _ => true  -- We don't have specific information about other claims

-- Define the conditions of the problem
axiom one_boy_correct : ∃ (b : Person), b = Person.Boy1 ∨ b = Person.Boy2
axiom one_girl_correct : ∃ (g : Person), g = Person.Girl1 ∨ g = Person.Girl2
axiom two_wrong : ∃ (p1 p2 : Person), p1 ≠ p2 ∧ ¬(claim p1 2) ∧ ¬(claim p2 2)

-- The theorem to prove
theorem actual_number_is_two : 
  ∃ (n : Nat), (claim Person.Natasha n = false) ∧ n = 2 :=
sorry

end NUMINAMATH_CALUDE_actual_number_is_two_l565_56511


namespace NUMINAMATH_CALUDE_sum_parity_eq_parity_of_M_l565_56595

/-- Represents the parity of a number -/
inductive Parity
  | Even
  | Odd

/-- The sum of N even numbers and M odd numbers -/
def sum_parity (N M : ℕ) : Parity :=
  match M % 2 with
  | 0 => Parity.Even
  | _ => Parity.Odd

/-- The parity of a natural number -/
def parity (n : ℕ) : Parity :=
  match n % 2 with
  | 0 => Parity.Even
  | _ => Parity.Odd

/-- Theorem: The parity of the sum of N even numbers and M odd numbers
    is equal to the parity of M -/
theorem sum_parity_eq_parity_of_M (N M : ℕ) :
  sum_parity N M = parity M := by sorry

end NUMINAMATH_CALUDE_sum_parity_eq_parity_of_M_l565_56595


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l565_56556

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (2 - 5 * x) = 5 → x = -4.6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l565_56556


namespace NUMINAMATH_CALUDE_yoongi_has_smallest_number_l565_56580

def yoongi_number : ℕ := 4
def jungkook_number : ℕ := 6 * 3
def yuna_number : ℕ := 5

theorem yoongi_has_smallest_number : 
  yoongi_number < jungkook_number ∧ yoongi_number < yuna_number :=
sorry

end NUMINAMATH_CALUDE_yoongi_has_smallest_number_l565_56580


namespace NUMINAMATH_CALUDE_existence_of_xy_l565_56566

theorem existence_of_xy (f g : ℝ → ℝ) : ∃ x y : ℝ, 
  x ∈ Set.Icc 0 1 ∧ 
  y ∈ Set.Icc 0 1 ∧ 
  |f x + g y - x * y| ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_xy_l565_56566


namespace NUMINAMATH_CALUDE_no_perfect_power_consecutive_product_l565_56587

theorem no_perfect_power_consecutive_product : 
  ∀ n : ℕ, ¬∃ (a k : ℕ), k > 1 ∧ n * (n + 1) = a ^ k :=
sorry

end NUMINAMATH_CALUDE_no_perfect_power_consecutive_product_l565_56587


namespace NUMINAMATH_CALUDE_car_trip_duration_l565_56589

/-- Represents the duration of a car trip with varying speeds -/
def car_trip (initial_speed initial_duration additional_speed average_speed : ℝ) : Prop :=
  ∃ (total_time additional_time : ℝ),
    total_time > 0 ∧
    additional_time ≥ 0 ∧
    total_time = initial_duration + additional_time ∧
    (initial_speed * initial_duration + additional_speed * additional_time) / total_time = average_speed

/-- The car trip lasts 12 hours given the specified conditions -/
theorem car_trip_duration :
  car_trip 45 4 75 65 → ∃ (total_time : ℝ), total_time = 12 :=
by sorry

end NUMINAMATH_CALUDE_car_trip_duration_l565_56589


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_six_exists_138_unique_greatest_l565_56517

theorem greatest_integer_with_gcf_six (n : ℕ) : n < 150 ∧ Nat.gcd n 18 = 6 → n ≤ 138 :=
by sorry

theorem exists_138 : 138 < 150 ∧ Nat.gcd 138 18 = 6 :=
by sorry

theorem unique_greatest : ∀ m : ℕ, m > 138 → m < 150 → Nat.gcd m 18 ≠ 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_six_exists_138_unique_greatest_l565_56517


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l565_56558

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x ≤ 1 ∧ x^2 > 1) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l565_56558


namespace NUMINAMATH_CALUDE_product_of_polynomials_l565_56530

theorem product_of_polynomials (g h : ℚ) : 
  (∀ x, (9*x^2 - 5*x + g) * (4*x^2 + h*x - 12) = 36*x^4 - 41*x^3 + 7*x^2 + 13*x - 72) →
  g + h = -11/6 := by sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l565_56530


namespace NUMINAMATH_CALUDE_integer_solutions_inequality_l565_56527

theorem integer_solutions_inequality :
  ∀ x y z : ℤ,
  x^2 * y^2 + y^2 * z^2 + x^2 + z^2 - 38*(x*y + z) - 40*(y*z + x) + 4*x*y*z + 761 ≤ 0 →
  ((x = 6 ∧ y = 2 ∧ z = 7) ∨ (x = 20 ∧ y = 0 ∧ z = 19)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_inequality_l565_56527


namespace NUMINAMATH_CALUDE_quadratic_decreasing_l565_56577

/-- Theorem: For a quadratic function y = ax² + 2ax + c where a < 0,
    and points A(1, y₁) and B(2, y₂) on this function, y₁ - y₂ > 0. -/
theorem quadratic_decreasing (a c y₁ y₂ : ℝ) (ha : a < 0) 
  (h1 : y₁ = a + 2*a + c) 
  (h2 : y₂ = 4*a + 4*a + c) : 
  y₁ - y₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_l565_56577


namespace NUMINAMATH_CALUDE_abs_neg_one_wrt_one_abs_wrt_one_2023_l565_56501

-- Define absolute value with respect to 1
def abs_wrt_one (a : ℝ) : ℝ := |a - 1|

-- Theorem 1
theorem abs_neg_one_wrt_one : abs_wrt_one (-1) = 2 := by sorry

-- Theorem 2
theorem abs_wrt_one_2023 (a : ℝ) : 
  abs_wrt_one a = 2023 → (a = 2024 ∨ a = -2022) := by sorry

end NUMINAMATH_CALUDE_abs_neg_one_wrt_one_abs_wrt_one_2023_l565_56501


namespace NUMINAMATH_CALUDE_points_form_hyperbola_l565_56592

/-- The set of points (x,y) defined by x = 2sinh(t) and y = 4cosh(t) for real t forms a hyperbola -/
theorem points_form_hyperbola :
  ∀ (x y t : ℝ), x = 2 * Real.sinh t ∧ y = 4 * Real.cosh t →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_points_form_hyperbola_l565_56592


namespace NUMINAMATH_CALUDE_arc_length_calculation_l565_56538

theorem arc_length_calculation (r : ℝ) (θ_central : ℝ) (θ_peripheral : ℝ) :
  r = 5 →
  θ_central = (2/3) * θ_peripheral →
  θ_peripheral = 2 * π →
  r * θ_central = (20 * π) / 3 :=
by sorry

end NUMINAMATH_CALUDE_arc_length_calculation_l565_56538


namespace NUMINAMATH_CALUDE_x_shape_is_line_segments_l565_56590

/-- The shape defined by θ = π/4 or θ = 5π/4 within 2 units of the origin -/
def X_shape : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 + p.2^2 ≤ 4) ∧ 
    (p.2 = p.1 ∨ p.2 = -p.1) ∧ 
    (p.1 ≠ 0 ∨ p.2 ≠ 0)}

theorem x_shape_is_line_segments : 
  ∃ (a b c d : ℝ × ℝ), 
    a ≠ b ∧ c ≠ d ∧
    X_shape = {p : ℝ × ℝ | ∃ (t : ℝ), (0 ≤ t ∧ t ≤ 1 ∧ 
      ((p = (1 - t) • a + t • b) ∨ (p = (1 - t) • c + t • d)))} :=
sorry

end NUMINAMATH_CALUDE_x_shape_is_line_segments_l565_56590


namespace NUMINAMATH_CALUDE_probability_different_families_l565_56522

/-- The number of families -/
def num_families : ℕ := 6

/-- The number of members in each family -/
def members_per_family : ℕ := 3

/-- The total number of people -/
def total_people : ℕ := num_families * members_per_family

/-- The size of each group in the game -/
def group_size : ℕ := 3

/-- The probability of selecting 3 people from different families -/
theorem probability_different_families : 
  (Nat.choose num_families group_size * (members_per_family ^ group_size)) / 
  (Nat.choose total_people group_size) = 45 / 68 := by sorry

end NUMINAMATH_CALUDE_probability_different_families_l565_56522


namespace NUMINAMATH_CALUDE_trapezium_side_length_l565_56569

theorem trapezium_side_length 
  (a b h area : ℝ) 
  (h1 : b = 28) 
  (h2 : h = 21) 
  (h3 : area = 504) 
  (h4 : area = (a + b) * h / 2) : 
  a = 20 := by sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l565_56569


namespace NUMINAMATH_CALUDE_borel_sets_closed_under_countable_operations_l565_56506

-- Define the σ-algebra of Borel sets
def BorelSets : Set (Set ℝ) := sorry

-- Define the property of being generated by open sets
def GeneratedByOpenSets (S : Set (Set ℝ)) : Prop := sorry

-- Define closure under countable union
def ClosedUnderCountableUnion (S : Set (Set ℝ)) : Prop := sorry

-- Define closure under countable intersection
def ClosedUnderCountableIntersection (S : Set (Set ℝ)) : Prop := sorry

-- Theorem statement
theorem borel_sets_closed_under_countable_operations :
  GeneratedByOpenSets BorelSets →
  ClosedUnderCountableUnion BorelSets ∧ ClosedUnderCountableIntersection BorelSets := by
  sorry

end NUMINAMATH_CALUDE_borel_sets_closed_under_countable_operations_l565_56506


namespace NUMINAMATH_CALUDE_minimize_f_minimum_l565_56544

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  |7 * x - 3 * a + 8| + |5 * x + 4 * a - 6| + |x - a - 8| - 24

/-- The theorem stating that 82/43 minimizes the minimum value of f(x) -/
theorem minimize_f_minimum (a : ℝ) :
  (∀ x, f (82/43) x ≤ f a x) → a = 82/43 := by
  sorry

end NUMINAMATH_CALUDE_minimize_f_minimum_l565_56544


namespace NUMINAMATH_CALUDE_ellipse_parabola_configuration_eccentricity_is_half_l565_56564

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a parabola with focal length c -/
structure Parabola where
  c : ℝ
  h_pos : 0 < c

/-- Configuration of the ellipse and parabola -/
structure Configuration where
  C₁ : Ellipse
  C₂ : Parabola
  h_focus : C₁.a * C₁.a - C₁.b * C₁.b = C₂.c * C₂.c  -- Right focus of C₁ coincides with focus of C₂
  h_center : True  -- Center of C₁ coincides with vertex of C₂ (implied by other conditions)
  h_chord_ratio : (2 * C₂.c) = 4/3 * (2 * C₁.b * C₁.b / C₁.a)  -- |CD| = 4/3 * |AB|
  h_vertices_sum : 2 * C₁.a + C₂.c = 12  -- Sum of distances from vertices of C₁ to directrix of C₂

/-- Main theorem statement -/
theorem ellipse_parabola_configuration (cfg : Configuration) :
  cfg.C₁.a * cfg.C₁.a = 16 ∧ 
  cfg.C₁.b * cfg.C₁.b = 12 ∧ 
  cfg.C₂.c = 2 :=
by sorry

/-- Corollary: Eccentricity of C₁ is 1/2 -/
theorem eccentricity_is_half (cfg : Configuration) :
  Real.sqrt (cfg.C₁.a * cfg.C₁.a - cfg.C₁.b * cfg.C₁.b) / cfg.C₁.a = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_configuration_eccentricity_is_half_l565_56564


namespace NUMINAMATH_CALUDE_parabola_focus_l565_56549

/-- The parabola equation: y^2 + 4x = 0 -/
def parabola_eq (x y : ℝ) : Prop := y^2 + 4*x = 0

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- The theorem stating that the focus of the parabola y^2 + 4x = 0 is at (-1, 0) -/
theorem parabola_focus :
  ∃ (f : Focus), (f.x = -1 ∧ f.y = 0) ∧
  ∀ (x y : ℝ), parabola_eq x y → 
    (y^2 = 4 * (f.x - x) ∧ f.y = 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l565_56549


namespace NUMINAMATH_CALUDE_problem_statement_l565_56565

theorem problem_statement (x y : ℝ) 
  (h1 : x + y = 3) 
  (h2 : x^2 + y^2 - x*y = 4) : 
  x^4 + y^4 + x^3*y + x*y^3 = 36 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l565_56565


namespace NUMINAMATH_CALUDE_triangle_altitude_l565_56551

theorem triangle_altitude (A b : ℝ) (h : A = 900 ∧ b = 45) :
  ∃ h : ℝ, A = (1/2) * b * h ∧ h = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_l565_56551


namespace NUMINAMATH_CALUDE_rotate90_neg4_plus_2i_l565_56563

def rotate90(z : ℂ) : ℂ := z * Complex.I

theorem rotate90_neg4_plus_2i :
  rotate90 (-4 + 2 * Complex.I) = -2 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_rotate90_neg4_plus_2i_l565_56563


namespace NUMINAMATH_CALUDE_halloween_candy_problem_l565_56541

theorem halloween_candy_problem (eaten : ℕ) (pile_size : ℕ) (num_piles : ℕ) :
  eaten = 30 →
  pile_size = 8 →
  num_piles = 6 →
  eaten + (pile_size * num_piles) = 78 :=
by sorry

end NUMINAMATH_CALUDE_halloween_candy_problem_l565_56541


namespace NUMINAMATH_CALUDE_tomorrow_is_saturday_l565_56584

-- Define the days of the week
inductive Day : Type
  | Sunday : Day
  | Monday : Day
  | Tuesday : Day
  | Wednesday : Day
  | Thursday : Day
  | Friday : Day
  | Saturday : Day

-- Define a function to get the next day
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

-- Define a function to add days
def addDays (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ k => nextDay (addDays d k)

-- Theorem statement
theorem tomorrow_is_saturday 
  (h : addDays Day.Wednesday 5 = Day.Monday) : 
  nextDay Day.Friday = Day.Saturday :=
by sorry

end NUMINAMATH_CALUDE_tomorrow_is_saturday_l565_56584


namespace NUMINAMATH_CALUDE_mathematics_encoding_l565_56596

def encode (c : Char) : ℕ :=
  match c with
  | 'M' => 22
  | 'A' => 32
  | 'T' => 33
  | 'E' => 11
  | 'I' => 23
  | 'K' => 13
  | _   => 0

def encodeWord (s : String) : List ℕ :=
  s.toList.map encode

theorem mathematics_encoding :
  encodeWord "MATHEMATICS" = [22, 32, 33, 11, 22, 32, 33, 23, 13, 32] :=
by sorry

end NUMINAMATH_CALUDE_mathematics_encoding_l565_56596


namespace NUMINAMATH_CALUDE_fraction_value_l565_56525

theorem fraction_value : (5 * 7) / 10 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l565_56525


namespace NUMINAMATH_CALUDE_tom_worked_eight_hours_l565_56581

/-- Represents the number of hours Tom worked on Monday -/
def hours : ℝ := 8

/-- Represents the number of customers Tom served per hour -/
def customers_per_hour : ℝ := 10

/-- Represents the bonus point percentage (20% = 0.20) -/
def bonus_percentage : ℝ := 0.20

/-- Represents the total bonus points Tom earned on Monday -/
def total_bonus_points : ℝ := 16

/-- Proves that Tom worked 8 hours on Monday given the conditions -/
theorem tom_worked_eight_hours :
  hours * customers_per_hour * bonus_percentage = total_bonus_points :=
sorry

end NUMINAMATH_CALUDE_tom_worked_eight_hours_l565_56581


namespace NUMINAMATH_CALUDE_rectangle_area_stage_8_l565_56518

/-- The area of a rectangle formed by n squares, each measuring s by s units -/
def rectangleArea (n : ℕ) (s : ℝ) : ℝ := n * (s * s)

/-- Theorem: The area of a rectangle formed by 8 squares, each measuring 4 inches by 4 inches, is 128 square inches -/
theorem rectangle_area_stage_8 : rectangleArea 8 4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_stage_8_l565_56518


namespace NUMINAMATH_CALUDE_retail_price_calculation_l565_56547

/-- The retail price of a machine given wholesale price, discount, and profit percentage -/
theorem retail_price_calculation (wholesale_price discount_percent profit_percent : ℝ) 
  (h_wholesale : wholesale_price = 90)
  (h_discount : discount_percent = 10)
  (h_profit : profit_percent = 20) :
  ∃ (retail_price : ℝ), 
    retail_price = 120 ∧ 
    (1 - discount_percent / 100) * retail_price = wholesale_price + (profit_percent / 100 * wholesale_price) := by
  sorry


end NUMINAMATH_CALUDE_retail_price_calculation_l565_56547
