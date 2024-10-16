import Mathlib

namespace NUMINAMATH_CALUDE_division_of_fractions_l1721_172115

theorem division_of_fractions : 
  (-1/24) / ((1/3) - (1/6) + (3/8)) = -1/13 := by sorry

end NUMINAMATH_CALUDE_division_of_fractions_l1721_172115


namespace NUMINAMATH_CALUDE_frog_dog_ratio_l1721_172162

theorem frog_dog_ratio (dogs : ℕ) (cats : ℕ) (frogs : ℕ) : 
  cats = (80 * dogs) / 100 →
  frogs = 160 →
  dogs + cats + frogs = 304 →
  frogs = 2 * dogs :=
by sorry

end NUMINAMATH_CALUDE_frog_dog_ratio_l1721_172162


namespace NUMINAMATH_CALUDE_joan_bought_six_dozens_l1721_172106

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The total number of eggs Joan bought -/
def total_eggs : ℕ := 72

/-- The number of dozens of eggs Joan bought -/
def dozens_bought : ℕ := total_eggs / eggs_per_dozen

theorem joan_bought_six_dozens : dozens_bought = 6 := by
  sorry

end NUMINAMATH_CALUDE_joan_bought_six_dozens_l1721_172106


namespace NUMINAMATH_CALUDE_tan_pi_over_a_equals_sqrt_three_l1721_172148

theorem tan_pi_over_a_equals_sqrt_three (a : ℝ) (h : a ^ 3 = 27) : 
  Real.tan (π / a) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_pi_over_a_equals_sqrt_three_l1721_172148


namespace NUMINAMATH_CALUDE_factory_work_hours_l1721_172187

/-- Calculates the number of hours a factory works per day given its production rates and total output. -/
theorem factory_work_hours 
  (refrigerators_per_hour : ℕ)
  (extra_coolers : ℕ)
  (total_products : ℕ)
  (days : ℕ)
  (h : refrigerators_per_hour = 90)
  (h' : extra_coolers = 70)
  (h'' : total_products = 11250)
  (h''' : days = 5) :
  (total_products / (days * (refrigerators_per_hour + (refrigerators_per_hour + extra_coolers)))) = 9 :=
by sorry

end NUMINAMATH_CALUDE_factory_work_hours_l1721_172187


namespace NUMINAMATH_CALUDE_square_binomial_constant_l1721_172104

theorem square_binomial_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 200*x + c = (x + a)^2) → c = 10000 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_constant_l1721_172104


namespace NUMINAMATH_CALUDE_sum_of_radii_tangent_circles_l1721_172129

/-- The sum of all possible radii of a circle tangent to both axes and externally tangent to another circle -/
theorem sum_of_radii_tangent_circles : ∃ (r₁ r₂ : ℝ),
  let c₁ : ℝ × ℝ := (r₁, r₁)  -- Center of the first circle
  let c₂ : ℝ × ℝ := (5, 0)    -- Center of the second circle
  let r₃ : ℝ := 3             -- Radius of the second circle
  (0 < r₁ ∧ 0 < r₂) ∧         -- Radii are positive
  (c₁.1 - c₂.1)^2 + (c₁.2 - c₂.2)^2 = (r₁ + r₃)^2 ∧  -- Circles are externally tangent
  r₁ + r₂ = 16 :=             -- Sum of radii is 16
by sorry

end NUMINAMATH_CALUDE_sum_of_radii_tangent_circles_l1721_172129


namespace NUMINAMATH_CALUDE_product_sum_and_reciprocals_geq_nine_l1721_172178

theorem product_sum_and_reciprocals_geq_nine (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_product_sum_and_reciprocals_geq_nine_l1721_172178


namespace NUMINAMATH_CALUDE_scott_runs_84_miles_per_month_l1721_172167

/-- Scott's weekly running schedule -/
structure RunningSchedule where
  mon_to_wed : ℕ  -- Miles run Monday through Wednesday (daily)
  thu_fri : ℕ     -- Miles run Thursday and Friday (daily)

/-- Calculate total miles run in a week -/
def weekly_miles (schedule : RunningSchedule) : ℕ :=
  schedule.mon_to_wed * 3 + schedule.thu_fri * 2

/-- Calculate total miles run in a month -/
def monthly_miles (schedule : RunningSchedule) (weeks : ℕ) : ℕ :=
  weekly_miles schedule * weeks

/-- Scott's actual running schedule -/
def scotts_schedule : RunningSchedule :=
  { mon_to_wed := 3, thu_fri := 6 }

/-- Theorem: Scott runs 84 miles in a month with 4 weeks -/
theorem scott_runs_84_miles_per_month : 
  monthly_miles scotts_schedule 4 = 84 := by sorry

end NUMINAMATH_CALUDE_scott_runs_84_miles_per_month_l1721_172167


namespace NUMINAMATH_CALUDE_equation_solutions_l1721_172142

theorem equation_solutions :
  (∃ x : ℝ, 2 * (x + 6) = 3 * (x - 1) ∧ x = 15) ∧
  (∃ x : ℝ, (x - 7) / 2 - (1 + x) / 3 = 1 ∧ x = 29) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1721_172142


namespace NUMINAMATH_CALUDE_slanted_line_angle_l1721_172170

/-- The angle between a slanted line segment and a plane, given that the slanted line segment
    is twice the length of its projection on the plane. -/
theorem slanted_line_angle (L l : ℝ) (h : L = 2 * l) :
  Real.arccos (l / L) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_slanted_line_angle_l1721_172170


namespace NUMINAMATH_CALUDE_cellar_water_pumping_time_l1721_172169

/-- Calculates the time needed to pump out water from a flooded cellar. -/
theorem cellar_water_pumping_time
  (length : ℝ)
  (width : ℝ)
  (depth : ℝ)
  (num_pumps : ℕ)
  (pump_rate : ℝ)
  (water_density : ℝ)
  (h_length : length = 30)
  (h_width : width = 40)
  (h_depth : depth = 2)
  (h_num_pumps : num_pumps = 4)
  (h_pump_rate : pump_rate = 10)
  (h_water_density : water_density = 7.5) :
  (length * width * depth * water_density) / (num_pumps * pump_rate) = 450 :=
sorry

end NUMINAMATH_CALUDE_cellar_water_pumping_time_l1721_172169


namespace NUMINAMATH_CALUDE_books_on_shelf_initial_books_count_l1721_172180

/-- The number of books on the shelf before Marta added more -/
def initial_books : ℕ := sorry

/-- The number of books Marta added to the shelf -/
def books_added : ℕ := 10

/-- The total number of books on the shelf after Marta added more -/
def total_books : ℕ := 48

/-- Theorem stating that the initial number of books plus the added books equals the total books -/
theorem books_on_shelf : initial_books + books_added = total_books := by sorry

/-- Theorem proving that the initial number of books is 38 -/
theorem initial_books_count : initial_books = 38 := by sorry

end NUMINAMATH_CALUDE_books_on_shelf_initial_books_count_l1721_172180


namespace NUMINAMATH_CALUDE_triangle_inequality_l1721_172122

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let s := (a + b + c) / 2
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * Real.sqrt (s * (s - a) * (s - b) * (s - c)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1721_172122


namespace NUMINAMATH_CALUDE_marbles_remaining_l1721_172105

/-- The number of marbles remaining in a pile after Chris and Ryan combine their marbles and each takes away 1/4 of the total. -/
theorem marbles_remaining (chris_marbles ryan_marbles : ℕ) 
  (h_chris : chris_marbles = 12)
  (h_ryan : ryan_marbles = 28) : 
  (chris_marbles + ryan_marbles) - 2 * ((chris_marbles + ryan_marbles) / 4) = 20 := by
  sorry

end NUMINAMATH_CALUDE_marbles_remaining_l1721_172105


namespace NUMINAMATH_CALUDE_range_of_a_l1721_172198

-- Define the proposition p
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, Real.exp x - a ≥ 0

-- State the theorem
theorem range_of_a (a : ℝ) : p a ↔ a ∈ Set.Iic (Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1721_172198


namespace NUMINAMATH_CALUDE_jean_initial_stuffies_l1721_172159

/-- Proves that Jean initially had 60 stuffies given the problem conditions -/
theorem jean_initial_stuffies :
  ∀ (initial : ℕ),
  (initial : ℚ) * (2/3) * (1/4) = 10 →
  initial = 60 := by
sorry

end NUMINAMATH_CALUDE_jean_initial_stuffies_l1721_172159


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_power_2017_l1721_172149

theorem last_four_digits_of_5_power_2017 (h1 : 5^5 % 10000 = 3125) 
                                         (h2 : 5^6 % 10000 = 5625) 
                                         (h3 : 5^7 % 10000 = 8125) : 
  5^2017 % 10000 = 3125 := by
sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_power_2017_l1721_172149


namespace NUMINAMATH_CALUDE_work_completion_time_l1721_172176

/-- Given two workers a and b, where a does half as much work as b in 3/4 of the time,
    and b takes 30 days to complete the work alone, prove that they take 18 days
    to complete the work together. -/
theorem work_completion_time (a b : ℝ) : 
  (a * (3/4 * 30) = (1/2) * b * 30) →  -- a does half as much work as b in 3/4 of the time
  (b * 30 = 1) →  -- b completes the work in 30 days
  (a + b) * 18 = 1  -- they complete the work together in 18 days
:= by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1721_172176


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l1721_172154

/-- Two points are symmetric about the y-axis if their y-coordinates are the same and their x-coordinates are opposites -/
def symmetric_about_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ = y₂ ∧ x₁ = -x₂

/-- The problem statement -/
theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_about_y_axis a 3 4 b →
  (a + b)^2012 = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l1721_172154


namespace NUMINAMATH_CALUDE_proposition_relationship_l1721_172195

theorem proposition_relationship :
  (∀ x : ℝ, (x - 3) * (x + 1) > 0 → x^2 - 2*x + 1 > 0) ∧
  (∃ x : ℝ, x^2 - 2*x + 1 > 0 ∧ (x - 3) * (x + 1) ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l1721_172195


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l1721_172146

/-- In three-dimensional space -/
structure Space3D where

/-- Represent a line in 3D space -/
structure Line (S : Space3D) where

/-- Represent a plane in 3D space -/
structure Plane (S : Space3D) where

/-- Perpendicular relation between two lines -/
def Line.perp (S : Space3D) (l1 l2 : Line S) : Prop :=
  sorry

/-- Perpendicular relation between a line and a plane -/
def Line.perpToPlane (S : Space3D) (l : Line S) (p : Plane S) : Prop :=
  sorry

/-- Perpendicular relation between two planes -/
def Plane.perp (S : Space3D) (p1 p2 : Plane S) : Prop :=
  sorry

/-- The main theorem -/
theorem perpendicular_transitivity (S : Space3D) (a b : Line S) (α β : Plane S) :
  a ≠ b → α ≠ β →
  Line.perp S a b →
  Line.perpToPlane S a α →
  Line.perpToPlane S b β →
  Plane.perp S α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l1721_172146


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1721_172119

/-- A geometric sequence with real number terms -/
def GeometricSequence := ℕ → ℝ

/-- Sum of the first n terms of a geometric sequence -/
def SumN (a : GeometricSequence) (n : ℕ) : ℝ := sorry

theorem geometric_sequence_sum (a : GeometricSequence) :
  SumN a 10 = 10 →
  SumN a 30 = 70 →
  SumN a 40 = 150 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1721_172119


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1721_172153

-- Define the inequality function
def f (x : ℝ) := (3*x + 1) * (2*x - 1)

-- Define the solution set
def solution_set := {x : ℝ | x < -1/3 ∨ x > 1/2}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x > 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1721_172153


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_eight_thirds_sqrt_three_l1721_172197

theorem sqrt_difference_equals_eight_thirds_sqrt_three :
  Real.sqrt 27 - Real.sqrt (1/3) = (8/3) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_eight_thirds_sqrt_three_l1721_172197


namespace NUMINAMATH_CALUDE_last_four_average_l1721_172128

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 60 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 4 = 63.75 := by
sorry

end NUMINAMATH_CALUDE_last_four_average_l1721_172128


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1721_172183

theorem quadratic_inequality_solution (z : ℝ) :
  z^2 - 42*z + 350 ≤ 4 ↔ 21 - Real.sqrt 95 ≤ z ∧ z ≤ 21 + Real.sqrt 95 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1721_172183


namespace NUMINAMATH_CALUDE_alien_energy_conversion_l1721_172134

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the alien's energy units --/
def alienEnergy : List Nat := [3, 6, 2]

theorem alien_energy_conversion :
  base7ToBase10 alienEnergy = 143 := by
  sorry

end NUMINAMATH_CALUDE_alien_energy_conversion_l1721_172134


namespace NUMINAMATH_CALUDE_special_circle_equation_l1721_172191

/-- A circle symmetric about the y-axis, passing through (1,0), 
    and divided by the x-axis into arc lengths with ratio 1:2 -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  symmetric_about_y_axis : center.1 = 0
  passes_through_1_0 : (1 - center.1)^2 + (0 - center.2)^2 = radius^2
  arc_ratio : Real.cos (Real.pi / 3) = center.2 / radius

/-- The equation of the special circle -/
def circle_equation (c : SpecialCircle) (x y : ℝ) : Prop :=
  x^2 + (y - c.center.2)^2 = c.radius^2

theorem special_circle_equation (c : SpecialCircle) :
  ∃ a : ℝ, a = Real.sqrt 3 / 3 ∧
    (∀ x y : ℝ, circle_equation c x y ↔ x^2 + (y - a)^2 = 4/3 ∨ x^2 + (y + a)^2 = 4/3) :=
sorry

end NUMINAMATH_CALUDE_special_circle_equation_l1721_172191


namespace NUMINAMATH_CALUDE_smallest_x_abs_equation_l1721_172184

theorem smallest_x_abs_equation : ∃ x : ℝ, x = -7 ∧ 
  (∀ y : ℝ, |4*y + 8| = 20 → y ≥ x) ∧ |4*x + 8| = 20 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_abs_equation_l1721_172184


namespace NUMINAMATH_CALUDE_gcd_180_480_l1721_172124

theorem gcd_180_480 : Nat.gcd 180 480 = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_480_l1721_172124


namespace NUMINAMATH_CALUDE_simple_interest_rate_is_five_percent_l1721_172109

/-- Calculates the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

/-- Theorem: The simple interest rate is 5% given the problem conditions -/
theorem simple_interest_rate_is_five_percent :
  simple_interest_rate 750 900 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_is_five_percent_l1721_172109


namespace NUMINAMATH_CALUDE_number_of_divisors_30030_l1721_172190

theorem number_of_divisors_30030 : Nat.card {d : ℕ | d > 0 ∧ 30030 % d = 0} = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_30030_l1721_172190


namespace NUMINAMATH_CALUDE_is_671st_term_l1721_172174

/-- The arithmetic sequence with first term 1 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℤ := 1 + 3 * (n - 1)

/-- 2011 is the 671st term in the arithmetic sequence -/
theorem is_671st_term : arithmetic_sequence 671 = 2011 := by sorry

end NUMINAMATH_CALUDE_is_671st_term_l1721_172174


namespace NUMINAMATH_CALUDE_evaluate_sqrt_fraction_l1721_172121

theorem evaluate_sqrt_fraction (x : ℝ) (h : x < -1) :
  Real.sqrt (x / (1 - (x - 2) / (x + 1))) = -x * (x + 1) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_sqrt_fraction_l1721_172121


namespace NUMINAMATH_CALUDE_problem_solution_l1721_172140

theorem problem_solution (a b : ℕ+) (q r : ℕ) :
  a^2 + b^2 = q * (a + b) + r ∧ q^2 + r = 1977 →
  ((a = 50 ∧ b = 7) ∨ (a = 50 ∧ b = 37) ∨ (a = 7 ∧ b = 50) ∨ (a = 37 ∧ b = 50)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1721_172140


namespace NUMINAMATH_CALUDE_fraction_equality_l1721_172116

theorem fraction_equality (p q r s : ℚ) 
  (h1 : p / q = 8)
  (h2 : r / q = 5)
  (h3 : r / s = 3 / 4) :
  s / p = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1721_172116


namespace NUMINAMATH_CALUDE_some_number_approximation_l1721_172107

/-- Given that (3.241 * 14) / x = 0.045374000000000005, prove that x ≈ 1000 -/
theorem some_number_approximation (x : ℝ) 
  (h : (3.241 * 14) / x = 0.045374000000000005) : 
  ∃ ε > 0, |x - 1000| < ε :=
sorry

end NUMINAMATH_CALUDE_some_number_approximation_l1721_172107


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l1721_172118

theorem quadratic_equation_proof (a : ℝ) :
  (a^2 - 4*a + 5 ≠ 0) ∧
  (∀ x : ℝ, (2^2 - 4*2 + 5)*x^2 + 2*2*x + 4 = 0 ↔ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l1721_172118


namespace NUMINAMATH_CALUDE_cos_2alpha_is_zero_l1721_172163

theorem cos_2alpha_is_zero (α : Real) (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.sin (2*α) = Real.cos (π/4 - α)) : Real.cos (2*α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_is_zero_l1721_172163


namespace NUMINAMATH_CALUDE_absolute_value_of_h_l1721_172185

theorem absolute_value_of_h (h : ℝ) : 
  (∃ x y : ℝ, x^2 + 2*h*x = 8 ∧ y^2 + 2*h*y = 8 ∧ x^2 + y^2 = 18) → 
  |h| = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_of_h_l1721_172185


namespace NUMINAMATH_CALUDE_parallel_angles_theorem_l1721_172131

/-- Two angles in space with parallel corresponding sides -/
structure ParallelAngles where
  a : Real
  b : Real
  parallel : Bool

/-- Theorem: If two angles with parallel corresponding sides have one angle of 60°, 
    then the other angle is either 60° or 120° -/
theorem parallel_angles_theorem (angles : ParallelAngles) 
  (h1 : angles.parallel = true) 
  (h2 : angles.a = 60) : 
  angles.b = 60 ∨ angles.b = 120 := by
  sorry

end NUMINAMATH_CALUDE_parallel_angles_theorem_l1721_172131


namespace NUMINAMATH_CALUDE_total_bananas_is_110_l1721_172155

/-- The total number of bananas Willie, Charles, and Lucy had originally -/
def total_bananas (willie_bananas charles_bananas lucy_bananas : ℕ) : ℕ :=
  willie_bananas + charles_bananas + lucy_bananas

/-- Theorem stating that the total number of bananas is 110 -/
theorem total_bananas_is_110 :
  total_bananas 48 35 27 = 110 := by
  sorry

end NUMINAMATH_CALUDE_total_bananas_is_110_l1721_172155


namespace NUMINAMATH_CALUDE_first_butcher_packages_correct_l1721_172126

/-- The number of packages delivered by the first butcher -/
def first_butcher_packages : ℕ := 10

/-- The weight of each package in pounds -/
def package_weight : ℕ := 4

/-- The number of packages delivered by the second butcher -/
def second_butcher_packages : ℕ := 7

/-- The number of packages delivered by the third butcher -/
def third_butcher_packages : ℕ := 8

/-- The total weight of all delivered packages in pounds -/
def total_weight : ℕ := 100

/-- Theorem stating that the number of packages delivered by the first butcher is correct -/
theorem first_butcher_packages_correct :
  package_weight * first_butcher_packages +
  package_weight * second_butcher_packages +
  package_weight * third_butcher_packages = total_weight :=
by sorry

end NUMINAMATH_CALUDE_first_butcher_packages_correct_l1721_172126


namespace NUMINAMATH_CALUDE_square_position_2010_l1721_172152

-- Define the possible positions of the square
inductive SquarePosition
  | ABCD
  | DABC
  | BDAC
  | ACBD
  | CABD
  | DCBA
  | CDAB
  | BADC
  | DBCA

def next_position (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.BDAC
  | SquarePosition.DABC => SquarePosition.BDAC
  | SquarePosition.BDAC => SquarePosition.ACBD
  | SquarePosition.ACBD => SquarePosition.CABD
  | SquarePosition.CABD => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.CDAB
  | SquarePosition.CDAB => SquarePosition.BADC
  | SquarePosition.BADC => SquarePosition.DBCA
  | SquarePosition.DBCA => SquarePosition.ABCD

def nth_position (n : Nat) : SquarePosition :=
  match n with
  | 0 => SquarePosition.ABCD
  | n + 1 => next_position (nth_position n)

theorem square_position_2010 :
  nth_position 2010 = SquarePosition.BDAC :=
by sorry

end NUMINAMATH_CALUDE_square_position_2010_l1721_172152


namespace NUMINAMATH_CALUDE_system_solution_l1721_172157

theorem system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ = 6 ∧ y₁ = 3 ∧ x₂ = 3 ∧ y₂ = 3/2) ∧
    (∀ x y : ℝ,
      3*x - 2*y > 0 ∧ x > 0 →
      (Real.sqrt ((3*x - 2*y)/(2*x)) + Real.sqrt ((2*x)/(3*x - 2*y)) = 2 ∧
       x^2 - 18 = 2*y*(4*y - 9)) →
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1721_172157


namespace NUMINAMATH_CALUDE_specific_cards_probability_l1721_172100

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (kings_per_suit : Nat)
  (queens_per_suit : Nat)
  (jacks_per_suit : Nat)

/-- Calculates the probability of drawing specific cards from a deck -/
def draw_probability (d : Deck) : Rat :=
  1 / (d.cards * (d.cards - 1) * (d.cards - 2) / (4 * d.queens_per_suit))

theorem specific_cards_probability :
  let standard_deck : Deck := {
    cards := 52,
    suits := 4,
    cards_per_suit := 13,
    kings_per_suit := 1,
    queens_per_suit := 1,
    jacks_per_suit := 1
  }
  draw_probability standard_deck = 1 / 33150 := by
  sorry

end NUMINAMATH_CALUDE_specific_cards_probability_l1721_172100


namespace NUMINAMATH_CALUDE_magic_shop_change_theorem_final_change_theorem_l1721_172175

/-- Represents the currency system in the magic shop -/
structure MagicShopCurrency where
  silver_to_gold_rate : ℚ
  cloak_price_gold : ℚ

/-- Calculate the change in silver coins when buying a cloak with gold coins -/
def change_in_silver (c : MagicShopCurrency) (gold_paid : ℚ) : ℚ :=
  (gold_paid - c.cloak_price_gold) * (1 / c.silver_to_gold_rate)

/-- Theorem: Buying a cloak with 14 gold coins results in 10 silver coins as change -/
theorem magic_shop_change_theorem (c : MagicShopCurrency) 
  (h1 : 20 = c.cloak_price_gold * c.silver_to_gold_rate + 4 * c.silver_to_gold_rate)
  (h2 : 15 = c.cloak_price_gold * c.silver_to_gold_rate + 1 * c.silver_to_gold_rate) :
  change_in_silver c 14 = 10 := by
  sorry

/-- The correct change is 10 silver coins -/
def correct_change : ℚ := 10

/-- The final theorem stating the correct change -/
theorem final_change_theorem (c : MagicShopCurrency) 
  (h1 : 20 = c.cloak_price_gold * c.silver_to_gold_rate + 4 * c.silver_to_gold_rate)
  (h2 : 15 = c.cloak_price_gold * c.silver_to_gold_rate + 1 * c.silver_to_gold_rate) :
  change_in_silver c 14 = correct_change := by
  sorry

end NUMINAMATH_CALUDE_magic_shop_change_theorem_final_change_theorem_l1721_172175


namespace NUMINAMATH_CALUDE_cube_face_sum_l1721_172165

theorem cube_face_sum (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ+) : 
  (a₁ * a₂ * a₅ + a₂ * a₃ * a₅ + a₃ * a₄ * a₅ + a₄ * a₁ * a₅ +
   a₁ * a₂ * a₆ + a₂ * a₃ * a₆ + a₃ * a₄ * a₆ + a₄ * a₁ * a₆ = 70) →
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ : ℕ) = 14 := by
sorry

end NUMINAMATH_CALUDE_cube_face_sum_l1721_172165


namespace NUMINAMATH_CALUDE_fifth_group_sample_l1721_172114

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population : ℕ
  num_groups : ℕ
  group_size : ℕ
  first_sample : ℕ

/-- Calculates the sample number for a given group in a systematic sampling scenario -/
def sample_number (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.first_sample + (group - 1) * s.group_size

/-- Theorem: In the given systematic sampling scenario, the sample number in the fifth group is 43 -/
theorem fifth_group_sample (s : SystematicSampling) 
  (h1 : s.population = 60)
  (h2 : s.num_groups = 6)
  (h3 : s.group_size = s.population / s.num_groups)
  (h4 : s.first_sample = 3) :
  sample_number s 5 = 43 := by
  sorry


end NUMINAMATH_CALUDE_fifth_group_sample_l1721_172114


namespace NUMINAMATH_CALUDE_courtyard_diagonal_length_l1721_172199

/-- Represents the length of the diagonal of a rectangular courtyard -/
def diagonal_length (side_ratio : ℚ) (paving_cost : ℚ) (cost_per_sqm : ℚ) : ℚ :=
  let longer_side := 4 * (paving_cost / cost_per_sqm / (12 * side_ratio)).sqrt
  let shorter_side := 3 * (paving_cost / cost_per_sqm / (12 * side_ratio)).sqrt
  (longer_side^2 + shorter_side^2).sqrt

/-- Theorem: The diagonal length of the courtyard is 50 meters -/
theorem courtyard_diagonal_length :
  diagonal_length (4/3) 600 0.5 = 50 := by
  sorry

#eval diagonal_length (4/3) 600 0.5

end NUMINAMATH_CALUDE_courtyard_diagonal_length_l1721_172199


namespace NUMINAMATH_CALUDE_vector_combination_equals_result_l1721_172172

/-- Given vectors a, b, and c in ℝ³, prove that 2a - 3b + 4c equals the specified result -/
theorem vector_combination_equals_result (a b c : ℝ × ℝ × ℝ) 
  (ha : a = (3, 5, -1)) 
  (hb : b = (2, 2, 3)) 
  (hc : c = (4, -1, -3)) : 
  (2 : ℝ) • a - (3 : ℝ) • b + (4 : ℝ) • c = (16, 0, -23) := by
  sorry

end NUMINAMATH_CALUDE_vector_combination_equals_result_l1721_172172


namespace NUMINAMATH_CALUDE_no_perfect_square_in_sequence_l1721_172138

def sequence_a : ℕ → ℚ
  | 0 => 2016
  | n + 1 => sequence_a n + 2 / sequence_a n

theorem no_perfect_square_in_sequence :
  ∀ n : ℕ, ¬∃ q : ℚ, sequence_a n = q ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_in_sequence_l1721_172138


namespace NUMINAMATH_CALUDE_student_journal_pages_l1721_172196

/-- Calculates the total number of journal pages written by a student over a given number of weeks. -/
def total_pages (sessions_per_week : ℕ) (pages_per_session : ℕ) (weeks : ℕ) : ℕ :=
  sessions_per_week * pages_per_session * weeks

/-- Theorem stating that given the specific conditions, a student writes 72 pages in 6 weeks. -/
theorem student_journal_pages :
  total_pages 3 4 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_student_journal_pages_l1721_172196


namespace NUMINAMATH_CALUDE_cargo_loaded_in_bahamas_l1721_172156

/-- The amount of cargo loaded in the Bahamas -/
def cargo_loaded (initial_cargo final_cargo : ℕ) : ℕ :=
  final_cargo - initial_cargo

/-- Theorem: The amount of cargo loaded in the Bahamas is 8723 tons -/
theorem cargo_loaded_in_bahamas :
  cargo_loaded 5973 14696 = 8723 := by
  sorry

end NUMINAMATH_CALUDE_cargo_loaded_in_bahamas_l1721_172156


namespace NUMINAMATH_CALUDE_tan_sum_zero_implies_tan_sqrt_three_l1721_172110

open Real

theorem tan_sum_zero_implies_tan_sqrt_three (θ : ℝ) :
  π/4 < θ ∧ θ < π/2 →
  tan θ + tan (2*θ) + tan (3*θ) + tan (4*θ) = 0 →
  tan θ = sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_zero_implies_tan_sqrt_three_l1721_172110


namespace NUMINAMATH_CALUDE_equation_roots_count_l1721_172177

-- Define the function f
def f (x : ℝ) : ℝ := |x| - 1

-- Define the iterative composition of f
def f_n (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => id
  | n + 1 => f ∘ (f_n n)

-- Theorem statement
theorem equation_roots_count :
  ∃! (roots : Finset ℝ), (∀ x ∈ roots, f_n 10 x = -1/2) ∧ (Finset.card roots = 20) := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_count_l1721_172177


namespace NUMINAMATH_CALUDE_no_real_roots_l1721_172182

theorem no_real_roots (a b : ℝ) (h1 : b/a > 1/4) (h2 : a > 0) :
  ∀ x : ℝ, x/a + b/x ≠ 1 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_l1721_172182


namespace NUMINAMATH_CALUDE_elsa_lost_marbles_l1721_172193

/-- The number of marbles Elsa lost at breakfast -/
def x : ℕ := sorry

/-- Elsa's initial number of marbles -/
def initial_marbles : ℕ := 40

/-- Number of marbles Elsa gave to Susie -/
def marbles_given_to_susie : ℕ := 5

/-- Number of new marbles Elsa's mom bought -/
def new_marbles : ℕ := 12

/-- Elsa's final number of marbles -/
def final_marbles : ℕ := 54

theorem elsa_lost_marbles : 
  initial_marbles - x - marbles_given_to_susie + new_marbles + 2 * marbles_given_to_susie = final_marbles ∧
  x = 3 := by sorry

end NUMINAMATH_CALUDE_elsa_lost_marbles_l1721_172193


namespace NUMINAMATH_CALUDE_mask_quality_most_suitable_l1721_172158

-- Define the survey types
inductive SurveyType
| SecurityCheck
| TeacherRecruitment
| MaskQuality
| StudentVision

-- Define a function to determine if a survey is suitable for sampling
def isSuitableForSampling (survey : SurveyType) : Prop :=
  match survey with
  | SurveyType.MaskQuality => True
  | _ => False

-- Theorem statement
theorem mask_quality_most_suitable :
  ∀ (survey : SurveyType), isSuitableForSampling survey → survey = SurveyType.MaskQuality :=
by sorry

end NUMINAMATH_CALUDE_mask_quality_most_suitable_l1721_172158


namespace NUMINAMATH_CALUDE_convex_polygon_covered_by_three_similar_l1721_172164

/-- A planar convex polygon. -/
structure PlanarConvexPolygon where
  -- Add necessary fields and properties here
  -- This is a placeholder definition

/-- Similarity between two planar convex polygons. -/
def IsSimilar (P Q : PlanarConvexPolygon) : Prop :=
  -- Define similarity condition here
  sorry

/-- One polygon covers another. -/
def Covers (P Q : PlanarConvexPolygon) : Prop :=
  -- Define covering condition here
  sorry

/-- Union of three polygons. -/
def Union3 (P Q R : PlanarConvexPolygon) : PlanarConvexPolygon :=
  -- Define union operation here
  sorry

/-- A polygon is smaller than another. -/
def IsSmaller (P Q : PlanarConvexPolygon) : Prop :=
  -- Define size comparison here
  sorry

/-- Theorem: Every planar convex polygon can be covered by three smaller similar polygons. -/
theorem convex_polygon_covered_by_three_similar :
  ∀ (M : PlanarConvexPolygon),
  ∃ (N₁ N₂ N₃ : PlanarConvexPolygon),
    IsSimilar N₁ M ∧ IsSimilar N₂ M ∧ IsSimilar N₃ M ∧
    IsSmaller N₁ M ∧ IsSmaller N₂ M ∧ IsSmaller N₃ M ∧
    Covers (Union3 N₁ N₂ N₃) M :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_covered_by_three_similar_l1721_172164


namespace NUMINAMATH_CALUDE_lcm_four_eight_l1721_172171

theorem lcm_four_eight : ∀ n : ℕ,
  (∃ m : ℕ, 4 ∣ m ∧ 8 ∣ m ∧ n ∣ m) →
  n ≥ 8 →
  Nat.lcm 4 8 = 8 :=
by sorry

end NUMINAMATH_CALUDE_lcm_four_eight_l1721_172171


namespace NUMINAMATH_CALUDE_road_trip_distance_l1721_172135

theorem road_trip_distance (D : ℝ) 
  (h1 : D > 0)
  (h2 : D * (1 - 1/3) * (1 - 1/5) * (1 - 1/4) = 400) : D = 1000 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_distance_l1721_172135


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l1721_172144

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  num_tiles : ℕ
  perimeter : ℕ

/-- Represents the process of adding tiles to a configuration -/
def add_tiles (initial : TileConfiguration) (added_tiles : ℕ) : TileConfiguration :=
  { num_tiles := initial.num_tiles + added_tiles,
    perimeter := initial.perimeter } -- Placeholder, actual calculation would depend on tile placement

/-- The theorem to be proved -/
theorem perimeter_after_adding_tiles :
  ∃ (final : TileConfiguration),
    let initial : TileConfiguration := { num_tiles := 9, perimeter := 16 }
    let with_added_tiles := add_tiles initial 3
    with_added_tiles.perimeter = 18 :=
sorry

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l1721_172144


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_345_l1721_172160

/-- The sum of the digits in the binary representation of a natural number -/
def sum_of_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

/-- The theorem stating that the sum of the digits in the binary representation of 345 is 5 -/
theorem sum_of_binary_digits_345 : sum_of_binary_digits 345 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_345_l1721_172160


namespace NUMINAMATH_CALUDE_perception_permutations_count_l1721_172108

/-- The number of letters in the word "PERCEPTION" -/
def total_letters : ℕ := 10

/-- The number of repeating letters (E, P, I, N) in "PERCEPTION" -/
def repeating_letters : Finset ℕ := {2, 2, 2, 2}

/-- The number of distinct permutations of the letters in "PERCEPTION" -/
def perception_permutations : ℕ := total_letters.factorial / (repeating_letters.prod (λ x => x.factorial))

theorem perception_permutations_count :
  perception_permutations = 226800 := by sorry

end NUMINAMATH_CALUDE_perception_permutations_count_l1721_172108


namespace NUMINAMATH_CALUDE_cosine_equality_l1721_172161

theorem cosine_equality (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → 
  Real.cos (n * π / 180) = Real.cos (820 * π / 180) → 
  n = 100 := by
sorry

end NUMINAMATH_CALUDE_cosine_equality_l1721_172161


namespace NUMINAMATH_CALUDE_student_ticket_cost_l1721_172189

theorem student_ticket_cost (num_students : ℕ) (num_teachers : ℕ) (adult_ticket_cost : ℚ) (total_cost : ℚ) :
  num_students = 12 →
  num_teachers = 4 →
  adult_ticket_cost = 3 →
  total_cost = 24 →
  ∃ (student_ticket_cost : ℚ),
    student_ticket_cost * num_students + adult_ticket_cost * num_teachers = total_cost ∧
    student_ticket_cost = 1 :=
by sorry

end NUMINAMATH_CALUDE_student_ticket_cost_l1721_172189


namespace NUMINAMATH_CALUDE_tank_emptied_in_two_minutes_l1721_172186

/-- Represents the state and properties of a water tank system -/
structure WaterTank where
  initialFill : ℚ  -- Initial fill level of the tank (1/5)
  pipeARate : ℚ    -- Rate at which pipe A fills the tank (1/15 per minute)
  pipeBRate : ℚ    -- Rate at which pipe B empties the tank (1/6 per minute)

/-- Calculates the time to empty or fill the tank completely -/
def timeToEmptyOrFill (tank : WaterTank) : ℚ :=
  tank.initialFill / (tank.pipeBRate - tank.pipeARate)

/-- Theorem stating that the tank will be emptied in 2 minutes -/
theorem tank_emptied_in_two_minutes (tank : WaterTank) 
  (h1 : tank.initialFill = 1/5)
  (h2 : tank.pipeARate = 1/15)
  (h3 : tank.pipeBRate = 1/6) : 
  timeToEmptyOrFill tank = 2 := by
  sorry

#eval timeToEmptyOrFill { initialFill := 1/5, pipeARate := 1/15, pipeBRate := 1/6 }

end NUMINAMATH_CALUDE_tank_emptied_in_two_minutes_l1721_172186


namespace NUMINAMATH_CALUDE_definite_integral_f_l1721_172123

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x + 2

-- State the theorem
theorem definite_integral_f : ∫ x in (0:ℝ)..(1:ℝ), f x = 11/6 := by sorry

end NUMINAMATH_CALUDE_definite_integral_f_l1721_172123


namespace NUMINAMATH_CALUDE_smallest_value_in_range_l1721_172151

theorem smallest_value_in_range (x : ℝ) (h : 0 < x ∧ x < 2) :
  x^2 ≤ min x (min (3*x) (min (Real.sqrt x) (1/x))) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_in_range_l1721_172151


namespace NUMINAMATH_CALUDE_cards_per_page_l1721_172150

theorem cards_per_page (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 8)
  (h2 : old_cards = 16)
  (h3 : pages = 8) :
  (new_cards + old_cards) / pages = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_per_page_l1721_172150


namespace NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l1721_172120

theorem shortest_altitude_of_triangle (a b c : ℝ) (h1 : a = 12) (h2 : b = 16) (h3 : c = 20) :
  ∃ h : ℝ, h = 9.6 ∧ h ≤ min a b ∧ h ≤ (2 * (a * b) / c) := by
  sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l1721_172120


namespace NUMINAMATH_CALUDE_square_of_linear_expression_l1721_172132

theorem square_of_linear_expression (n : ℚ) :
  (∃ a b : ℚ, ∀ x : ℚ, (7 * x^2 + 21 * x + 5 * n) / 7 = (a * x + b)^2) →
  n = 63/20 := by
sorry

end NUMINAMATH_CALUDE_square_of_linear_expression_l1721_172132


namespace NUMINAMATH_CALUDE_demand_exceeds_50k_july_august_l1721_172147

def S (n : ℕ) : ℚ := n / 27 * (21 * n - n^2 - 5)

def demand_exceeds_50k (n : ℕ) : Prop := S n - S (n-1) > 5

theorem demand_exceeds_50k_july_august :
  demand_exceeds_50k 7 ∧ demand_exceeds_50k 8 ∧
  ∀ m, m < 7 ∨ m > 8 → ¬demand_exceeds_50k m :=
sorry

end NUMINAMATH_CALUDE_demand_exceeds_50k_july_august_l1721_172147


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1721_172137

def A : Set ℝ := {x | -3 ≤ x ∧ x < 4}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1721_172137


namespace NUMINAMATH_CALUDE_earnings_difference_l1721_172166

def car_price : ℕ := 5200
def inspection_cost : ℕ := car_price / 10
def headlight_cost : ℕ := 80
def tire_cost : ℕ := 3 * headlight_cost

def first_offer_earnings : ℕ := car_price - inspection_cost
def second_offer_earnings : ℕ := car_price - (headlight_cost + tire_cost)

theorem earnings_difference : second_offer_earnings - first_offer_earnings = 200 := by
  sorry

end NUMINAMATH_CALUDE_earnings_difference_l1721_172166


namespace NUMINAMATH_CALUDE_diameter_in_scientific_notation_l1721_172103

def scientific_notation (n : ℝ) (a : ℝ) (b : ℤ) : Prop :=
  n = a * (10 : ℝ) ^ b ∧ 1 ≤ a ∧ a < 10

theorem diameter_in_scientific_notation :
  scientific_notation 0.0000077 7.7 (-6) :=
sorry

end NUMINAMATH_CALUDE_diameter_in_scientific_notation_l1721_172103


namespace NUMINAMATH_CALUDE_century_park_weed_removal_l1721_172179

/-- Represents the weed growth and removal scenario in Century Park --/
structure WeedScenario where
  weed_growth_rate : ℝ
  worker_removal_rate : ℝ
  day1_duration : ℕ
  day2_workers : ℕ
  day2_duration : ℕ

/-- Calculates the finish time for day 3 given a WeedScenario --/
def day3_finish_time (scenario : WeedScenario) : ℕ :=
  sorry

/-- The theorem states that given the specific scenario, 8 workers will finish at 8:38 AM on day 3 --/
theorem century_park_weed_removal 
  (scenario : WeedScenario)
  (h1 : scenario.day1_duration = 60)
  (h2 : scenario.day2_workers = 10)
  (h3 : scenario.day2_duration = 30) :
  day3_finish_time scenario = 38 :=
sorry

end NUMINAMATH_CALUDE_century_park_weed_removal_l1721_172179


namespace NUMINAMATH_CALUDE_problem_statement_l1721_172143

open Real

theorem problem_statement : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 3^x₀ + x₀ = 2016) ∧ 
  ¬(∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, |x| - a*x = |-x| - a*(-x)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1721_172143


namespace NUMINAMATH_CALUDE_equation_simplification_l1721_172113

theorem equation_simplification (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 1 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_simplification_l1721_172113


namespace NUMINAMATH_CALUDE_custom_mult_zero_l1721_172127

/-- Custom multiplication operation for real numbers -/
def custom_mult (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem stating that (x-y)^2 * (y-x)^2 = 0 under the custom multiplication -/
theorem custom_mult_zero (x y : ℝ) : custom_mult ((x - y)^2) ((y - x)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_zero_l1721_172127


namespace NUMINAMATH_CALUDE_sine_product_equality_l1721_172173

theorem sine_product_equality : 
  3.438 * Real.sin (84 * π / 180) * Real.sin (24 * π / 180) * 
  Real.sin (48 * π / 180) * Real.sin (12 * π / 180) = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_sine_product_equality_l1721_172173


namespace NUMINAMATH_CALUDE_rectangular_frame_properties_l1721_172102

/-- Calculates the total length of wire needed for a rectangular frame --/
def total_wire_length (a b c : ℕ) : ℕ := 4 * (a + b + c)

/-- Calculates the total area of paper needed to cover a rectangular frame --/
def total_paper_area (a b c : ℕ) : ℕ := 2 * (a * b + a * c + b * c)

theorem rectangular_frame_properties :
  total_wire_length 3 4 5 = 48 ∧ total_paper_area 3 4 5 = 94 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_frame_properties_l1721_172102


namespace NUMINAMATH_CALUDE_library_initial_books_l1721_172141

/-- The number of books purchased last year -/
def books_last_year : ℕ := 50

/-- The number of books purchased this year -/
def books_this_year : ℕ := 3 * books_last_year

/-- The total number of books in the library now -/
def total_books_now : ℕ := 300

/-- The number of books in the library before the new purchases last year -/
def initial_books : ℕ := total_books_now - books_last_year - books_this_year

theorem library_initial_books :
  initial_books = 100 := by sorry

end NUMINAMATH_CALUDE_library_initial_books_l1721_172141


namespace NUMINAMATH_CALUDE_ellipse_dot_product_min_l1721_172125

/-- An ellipse with center at origin and left focus at (-1, 0) -/
structure Ellipse where
  x : ℝ
  y : ℝ
  eq : x^2 / 4 + y^2 / 3 = 1

/-- The dot product of OP and FP is always greater than or equal to 2 -/
theorem ellipse_dot_product_min (P : Ellipse) : 
  P.x * (P.x + 1) + P.y * P.y ≥ 2 := by
  sorry

#check ellipse_dot_product_min

end NUMINAMATH_CALUDE_ellipse_dot_product_min_l1721_172125


namespace NUMINAMATH_CALUDE_x4_plus_y4_equals_7_l1721_172194

theorem x4_plus_y4_equals_7 (x y : ℝ) 
  (hx : x^4 + x^2 = 3) 
  (hy : y^4 - y^2 = 3) : 
  x^4 + y^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x4_plus_y4_equals_7_l1721_172194


namespace NUMINAMATH_CALUDE_glorys_favorite_number_l1721_172181

theorem glorys_favorite_number (glory misty : ℕ) : 
  misty = glory / 3 →
  misty + glory = 600 →
  glory = 450 := by sorry

end NUMINAMATH_CALUDE_glorys_favorite_number_l1721_172181


namespace NUMINAMATH_CALUDE_cos_two_alpha_value_l1721_172192

theorem cos_two_alpha_value (α : Real) (h : Real.tan (α + π/4) = 1/3) : 
  Real.cos (2*α) = 3/5 := by sorry

end NUMINAMATH_CALUDE_cos_two_alpha_value_l1721_172192


namespace NUMINAMATH_CALUDE_train_crossing_time_l1721_172145

/-- Given a train crossing two platforms of different lengths, prove the time taken to cross the second platform -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform1_length platform2_length : ℝ)
  (time1 : ℝ) 
  (h1 : train_length = 30)
  (h2 : platform1_length = 180)
  (h3 : platform2_length = 250)
  (h4 : time1 = 15)
  (h5 : (train_length + platform1_length) / time1 = (train_length + platform2_length) / (20 : ℝ)) :
  (train_length + platform2_length) / ((train_length + platform1_length) / time1) = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1721_172145


namespace NUMINAMATH_CALUDE_min_daily_pages_for_given_plan_l1721_172101

/-- Represents a reading plan for a book -/
structure ReadingPlan where
  total_pages : ℕ
  total_days : ℕ
  initial_days : ℕ
  initial_pages : ℕ

/-- Calculates the minimum pages to read daily for the remaining days -/
def min_daily_pages (plan : ReadingPlan) : ℕ :=
  ((plan.total_pages - plan.initial_pages) + (plan.total_days - plan.initial_days - 1)) / (plan.total_days - plan.initial_days)

/-- Theorem stating the minimum daily pages for the given reading plan -/
theorem min_daily_pages_for_given_plan :
  let plan := ReadingPlan.mk 400 10 5 100
  min_daily_pages plan = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_daily_pages_for_given_plan_l1721_172101


namespace NUMINAMATH_CALUDE_tolu_pencils_tolu_wants_three_pencils_l1721_172168

/-- The problem of determining the number of pencils Tolu wants -/
theorem tolu_pencils (pencil_price : ℚ) (robert_pencils melissa_pencils : ℕ) 
  (total_spent : ℚ) : ℕ :=
  let tolu_pencils := (total_spent - pencil_price * (robert_pencils + melissa_pencils)) / pencil_price
  3

/-- The main theorem stating that Tolu wants 3 pencils -/
theorem tolu_wants_three_pencils : 
  tolu_pencils (20 / 100) 5 2 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tolu_pencils_tolu_wants_three_pencils_l1721_172168


namespace NUMINAMATH_CALUDE_distance_between_first_and_last_trees_l1721_172117

/-- Given 30 trees along a straight road with 3 meters between each adjacent pair of trees,
    the distance between the first and last trees is 87 meters. -/
theorem distance_between_first_and_last_trees (num_trees : ℕ) (distance_between_trees : ℝ) :
  num_trees = 30 →
  distance_between_trees = 3 →
  (num_trees - 1) * distance_between_trees = 87 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_first_and_last_trees_l1721_172117


namespace NUMINAMATH_CALUDE_feb_7_is_saturday_l1721_172139

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in February -/
structure FebruaryDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Given that February 14 is a Saturday, February 7 is also a Saturday -/
theorem feb_7_is_saturday (feb14 : FebruaryDate) 
    (h14 : feb14.day = 14 ∧ feb14.dayOfWeek = DayOfWeek.Saturday) :
    ∃ (feb7 : FebruaryDate), feb7.day = 7 ∧ feb7.dayOfWeek = DayOfWeek.Saturday := by
  sorry

end NUMINAMATH_CALUDE_feb_7_is_saturday_l1721_172139


namespace NUMINAMATH_CALUDE_linda_earnings_l1721_172112

/-- Calculates the total money earned from selling jeans and tees -/
def total_money_earned (jeans_price : ℕ) (tees_price : ℕ) (jeans_sold : ℕ) (tees_sold : ℕ) : ℕ :=
  jeans_price * jeans_sold + tees_price * tees_sold

/-- Proves that Linda earned $100 from selling jeans and tees -/
theorem linda_earnings : total_money_earned 11 8 4 7 = 100 := by
  sorry

end NUMINAMATH_CALUDE_linda_earnings_l1721_172112


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l1721_172136

theorem opposite_of_negative_two :
  ∃ x : ℝ, x + (-2) = 0 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l1721_172136


namespace NUMINAMATH_CALUDE_veg_eaters_count_l1721_172130

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  only_veg : ℕ
  only_nonveg : ℕ
  both_veg_and_nonveg : ℕ

/-- Calculates the total number of people who eat veg in the family -/
def total_veg_eaters (fd : FamilyDiet) : ℕ :=
  fd.only_veg + fd.both_veg_and_nonveg

/-- Theorem: The number of people who eat veg in the given family is 26 -/
theorem veg_eaters_count (fd : FamilyDiet) 
  (h1 : fd.only_veg = 15)
  (h2 : fd.only_nonveg = 8)
  (h3 : fd.both_veg_and_nonveg = 11) : 
  total_veg_eaters fd = 26 := by
  sorry

end NUMINAMATH_CALUDE_veg_eaters_count_l1721_172130


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_l1721_172188

theorem rationalize_and_simplify :
  ∃ (A B C : ℤ), 
    (3 + Real.sqrt 2) / (2 - Real.sqrt 5) = 
      A + B * Real.sqrt C ∧ A * B * C = -24 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_l1721_172188


namespace NUMINAMATH_CALUDE_largest_minus_smallest_l1721_172111

def problem (A B C : ℤ) : Prop :=
  A = 10 * 2 + 9 ∧
  A = B + 16 ∧
  C = B * 3

theorem largest_minus_smallest (A B C : ℤ) 
  (h : problem A B C) : 
  max A (max B C) - min A (min B C) = 26 := by
  sorry

end NUMINAMATH_CALUDE_largest_minus_smallest_l1721_172111


namespace NUMINAMATH_CALUDE_commission_per_car_l1721_172133

/-- Proves that the commission per car is $200 given the specified conditions -/
theorem commission_per_car 
  (base_salary : ℕ) 
  (march_earnings : ℕ) 
  (cars_to_double : ℕ) 
  (h1 : base_salary = 1000)
  (h2 : march_earnings = 2000)
  (h3 : cars_to_double = 15) :
  (2 * march_earnings - base_salary) / cars_to_double = 200 := by
  sorry

end NUMINAMATH_CALUDE_commission_per_car_l1721_172133
