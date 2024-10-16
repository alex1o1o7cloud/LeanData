import Mathlib

namespace NUMINAMATH_CALUDE_boat_trip_theorem_l1390_139030

/-- Represents the boat trip scenario -/
structure BoatTrip where
  total_time : ℝ
  stream_velocity : ℝ
  boat_speed : ℝ
  distance : ℝ

/-- The specific boat trip instance from the problem -/
def problem_trip : BoatTrip where
  total_time := 38
  stream_velocity := 4
  boat_speed := 14
  distance := 360

/-- Theorem stating that the given boat trip satisfies the problem conditions -/
theorem boat_trip_theorem (trip : BoatTrip) : 
  trip.total_time = 38 ∧ 
  trip.stream_velocity = 4 ∧ 
  trip.boat_speed = 14 ∧
  trip.distance / (trip.boat_speed + trip.stream_velocity) + 
    (trip.distance / 2) / (trip.boat_speed - trip.stream_velocity) = trip.total_time →
  trip.distance = 360 := by
  sorry

#check boat_trip_theorem problem_trip

end NUMINAMATH_CALUDE_boat_trip_theorem_l1390_139030


namespace NUMINAMATH_CALUDE_least_positive_slope_line_l1390_139034

/-- The curve equation -/
def curve (x y : ℝ) : Prop := 4 * x^2 - y^2 - 8 * x = 12

/-- The line equation -/
def line (m : ℝ) (x y : ℝ) : Prop := y = m * x - m

/-- The line contains the point (1, 0) -/
def contains_point (m : ℝ) : Prop := line m 1 0

/-- The line does not intersect the curve -/
def no_intersection (m : ℝ) : Prop := ∀ x y : ℝ, line m x y → ¬curve x y

/-- The slope is positive -/
def positive_slope (m : ℝ) : Prop := m > 0

theorem least_positive_slope_line :
  ∃ m : ℝ, m = 2 ∧
    contains_point m ∧
    no_intersection m ∧
    positive_slope m ∧
    ∀ m' : ℝ, m' ≠ m → contains_point m' → no_intersection m' → positive_slope m' → m' > m :=
sorry

end NUMINAMATH_CALUDE_least_positive_slope_line_l1390_139034


namespace NUMINAMATH_CALUDE_original_hourly_wage_l1390_139081

/-- Given a worker's daily wage, increased wage, bonus, total new wage, and hours worked per day,
    calculate the original hourly wage. -/
theorem original_hourly_wage (W : ℝ) (h1 : 1.60 * W + 10 = 45) (h2 : 8 > 0) :
  W / 8 = (45 - 10) / (1.60 * 8) := by sorry

end NUMINAMATH_CALUDE_original_hourly_wage_l1390_139081


namespace NUMINAMATH_CALUDE_untouched_area_of_tetrahedron_l1390_139015

/-- The area of a regular tetrahedron's inner wall that cannot be touched by an inscribed sphere -/
theorem untouched_area_of_tetrahedron (r : ℝ) (a : ℝ) (h : a = 4 * Real.sqrt 6) :
  let total_surface_area := a^2 * Real.sqrt 3
  let touched_area := (a^2 * Real.sqrt 3) / 4
  total_surface_area - touched_area = 108 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_untouched_area_of_tetrahedron_l1390_139015


namespace NUMINAMATH_CALUDE_triangle_inequality_inside_l1390_139089

/-- A point is inside a triangle if it's in the interior of the triangle --/
def PointInsideTriangle (A B C M : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (α β γ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = 1 ∧
  M = α • A + β • B + γ • C

/-- The theorem statement --/
theorem triangle_inequality_inside (A B C M : EuclideanSpace ℝ (Fin 2)) 
  (h : PointInsideTriangle A B C M) : 
  dist M B + dist M C < dist A B + dist A C := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_inside_l1390_139089


namespace NUMINAMATH_CALUDE_all_propositions_false_l1390_139022

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relations between lines and planes
def parallel (x y : Line) : Prop := sorry
def perpendicular (x y : Line) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def plane_perpendicular (p q : Plane) : Prop := sorry

-- Define the given lines and planes
variable (a b l : Line)
variable (α β γ : Plane)

-- Axioms for different objects
axiom different_lines : a ≠ b ∧ b ≠ l ∧ a ≠ l
axiom different_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- The four propositions
def proposition1 : Prop := 
  ∀ a b α, parallel a b → contained_in b α → parallel a α

def proposition2 : Prop := 
  ∀ a b α, perpendicular a b → perpendicular b α → parallel a α

def proposition3 : Prop := 
  ∀ l α β, plane_perpendicular α β → contained_in l α → perpendicular l β

def proposition4 : Prop := 
  ∀ l a b α, perpendicular l a → perpendicular l b → 
    contained_in a α → contained_in b α → perpendicular l α

-- Theorem stating all propositions are false
theorem all_propositions_false : 
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4 := by
  sorry

end NUMINAMATH_CALUDE_all_propositions_false_l1390_139022


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1390_139050

theorem circle_area_ratio (R₁ R₂ : ℝ) (h : R₁ > 0 ∧ R₂ > 0) :
  let chord_length := R₁ * Real.sqrt 3
  chord_length = R₂ * Real.sqrt 3 →
  (π * R₁^2) / (π * R₂^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1390_139050


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1390_139008

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 4
  let θ : ℝ := π / 4
  let φ : ℝ := π / 6
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (Real.sqrt 2, Real.sqrt 2, 2 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1390_139008


namespace NUMINAMATH_CALUDE_min_m_for_inequality_l1390_139044

theorem min_m_for_inequality : 
  (∃ (m : ℝ), ∀ (x : ℝ), x ∈ Set.Icc (-1) 2 → x^2 - m ≤ 1) ∧ 
  (∀ (m' : ℝ), (∀ (x : ℝ), x ∈ Set.Icc (-1) 2 → x^2 - m' ≤ 1) → m' ≥ 3) :=
by sorry


end NUMINAMATH_CALUDE_min_m_for_inequality_l1390_139044


namespace NUMINAMATH_CALUDE_average_of_numbers_l1390_139045

def numbers : List ℝ := [10, 4, 8, 7, 6]

theorem average_of_numbers : (numbers.sum / numbers.length : ℝ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1390_139045


namespace NUMINAMATH_CALUDE_oplus_equation_solution_l1390_139080

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := 4 * a + 2 * b

-- Theorem statement
theorem oplus_equation_solution :
  ∃ y : ℝ, oplus 3 (oplus 4 y) = -14 ∧ y = -14.5 := by
sorry

end NUMINAMATH_CALUDE_oplus_equation_solution_l1390_139080


namespace NUMINAMATH_CALUDE_greatest_five_digit_divisible_by_63_11_with_reverse_divisible_by_63_l1390_139021

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is a five-digit integer -/
def isFiveDigit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem greatest_five_digit_divisible_by_63_11_with_reverse_divisible_by_63 :
  ∃ (p : ℕ), isFiveDigit p ∧ 
             p % 63 = 0 ∧
             (reverseDigits p) % 63 = 0 ∧
             p % 11 = 0 ∧
             ∀ (q : ℕ), isFiveDigit q ∧ 
                         q % 63 = 0 ∧ 
                         (reverseDigits q) % 63 = 0 ∧ 
                         q % 11 = 0 → 
                         q ≤ p ∧
             p = 99729 := by
  sorry

end NUMINAMATH_CALUDE_greatest_five_digit_divisible_by_63_11_with_reverse_divisible_by_63_l1390_139021


namespace NUMINAMATH_CALUDE_min_value_a_l1390_139058

theorem min_value_a (a b : ℤ) (m : ℕ) (h1 : a - b = m) (h2 : Nat.Prime m) 
  (h3 : ∃ n : ℕ, a * b = n * n) (h4 : a ≥ 2012) : 
  (∀ a' b' : ℤ, ∃ m' : ℕ, a' - b' = m' ∧ Nat.Prime m' ∧ (∃ n' : ℕ, a' * b' = n' * n') ∧ a' ≥ 2012 → a' ≥ a) ∧ 
  a = 2025 := by
sorry


end NUMINAMATH_CALUDE_min_value_a_l1390_139058


namespace NUMINAMATH_CALUDE_secant_slope_positive_l1390_139033

open Real

noncomputable def f (x : ℝ) : ℝ := 2^x + x^3

theorem secant_slope_positive (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  (f x₁ - f x₂) / (x₁ - x₂) > 0 :=
sorry

end NUMINAMATH_CALUDE_secant_slope_positive_l1390_139033


namespace NUMINAMATH_CALUDE_rectangle_inequality_l1390_139018

/-- Represents a rectangle with side lengths 3b and b -/
structure Rectangle (b : ℝ) where
  length : ℝ := 3 * b
  width : ℝ := b

/-- Represents a point P on the longer side of the rectangle -/
structure PointP (b : ℝ) where
  x : ℝ
  y : ℝ := 0
  h1 : 0 ≤ x ∧ x ≤ 3 * b

/-- Represents a point T inside the rectangle -/
structure PointT (b : ℝ) where
  x : ℝ
  y : ℝ
  h1 : 0 < x ∧ x < 3 * b
  h2 : 0 < y ∧ y < b
  h3 : y = b / 2

/-- The theorem to be proved -/
theorem rectangle_inequality (b : ℝ) (h : b > 0) (R : Rectangle b) (P : PointP b) (T : PointT b) :
  let s := (2 * b)^2 + b^2
  let rt := (T.x - 0)^2 + (T.y - 0)^2
  s > 2 * rt := by sorry

end NUMINAMATH_CALUDE_rectangle_inequality_l1390_139018


namespace NUMINAMATH_CALUDE_range_of_a_l1390_139002

-- Define the statements p and q
def p (a : ℝ) : Prop :=
  ∀ k : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x₁^2 + (k*x₁ + 1)^2/a = 1) ∧ 
    (x₂^2 + (k*x₂ + 1)^2/a = 1)

def q (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, 4^x₀ - 2^x₀ - a ≤ 0

-- Theorem statement
theorem range_of_a :
  (∀ a : ℝ, ¬(p a ∧ q a)) ∧ (∀ a : ℝ, p a ∨ q a) →
  ∀ a : ℝ, -1/4 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1390_139002


namespace NUMINAMATH_CALUDE_B_is_largest_l1390_139027

def A : ℚ := 2008/2007 + 2008/2009
def B : ℚ := 2010/2009 + 2 * (2010/2009)
def C : ℚ := 2009/2008 + 2009/2010

theorem B_is_largest : B > A ∧ B > C := by
  sorry

end NUMINAMATH_CALUDE_B_is_largest_l1390_139027


namespace NUMINAMATH_CALUDE_expression_is_perfect_square_l1390_139076

theorem expression_is_perfect_square (x y z : ℤ) (A : ℤ) :
  A = x * y + y * z + z * x →
  A = (x + 1) * (y - 2) + (y - 2) * (z - 2) + (z - 2) * (x + 1) →
  ∃ k : ℤ, (-1) * A = k^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_is_perfect_square_l1390_139076


namespace NUMINAMATH_CALUDE_prime_equation_solutions_l1390_139047

theorem prime_equation_solutions :
  ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p ∧ p^3 + p^2 - 18*p + 26 = 0) ∧ S.card = 2 :=
by sorry

end NUMINAMATH_CALUDE_prime_equation_solutions_l1390_139047


namespace NUMINAMATH_CALUDE_inequality_proof_l1390_139026

theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1390_139026


namespace NUMINAMATH_CALUDE_eliana_steps_l1390_139057

/-- The total number of steps Eliana walked during three days -/
def total_steps (first_day_initial : ℕ) (first_day_additional : ℕ) (third_day_additional : ℕ) : ℕ :=
  let first_day := first_day_initial + first_day_additional
  let second_day := 2 * first_day
  let third_day := second_day + third_day_additional
  first_day + second_day + third_day

/-- Theorem stating the total number of steps Eliana walked during three days -/
theorem eliana_steps : 
  total_steps 200 300 100 = 2600 := by
  sorry

end NUMINAMATH_CALUDE_eliana_steps_l1390_139057


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l1390_139053

/-- The volume of a sphere inscribed in a cube with edge length 10 inches -/
theorem inscribed_sphere_volume :
  let edge_length : ℝ := 10
  let radius : ℝ := edge_length / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * radius ^ 3
  sphere_volume = (500 / 3) * Real.pi := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l1390_139053


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1390_139014

theorem inequality_solution_set (x : ℝ) : x - 3 > 4*x ↔ x < -1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1390_139014


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_range_l1390_139083

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (1 - m) + y^2 / (m + 2) = 1
def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (2 * m) + y^2 / (2 - m) = 1

-- Define the condition that q represents an ellipse with foci on the x-axis
def q_ellipse (m : ℝ) : Prop := 2 * m > 0 ∧ 2 - m > 0

-- Define the theorem
theorem hyperbola_ellipse_range (m : ℝ) : 
  (¬(p m ∧ q m) ∧ q_ellipse m) ↔ (m ≤ 1 ∨ m ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_range_l1390_139083


namespace NUMINAMATH_CALUDE_units_digit_of_power_l1390_139025

/-- The units' digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The given number 4589 -/
def givenNumber : ℕ := 4589

/-- The given exponent 1276 -/
def givenExponent : ℕ := 1276

/-- Theorem: The units' digit of 4589^1276 is 1 -/
theorem units_digit_of_power : unitsDigit (givenNumber ^ givenExponent) = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_l1390_139025


namespace NUMINAMATH_CALUDE_cookie_ratio_proof_l1390_139073

def cookie_problem (initial_cookies brother_cookies final_cookies : ℕ) : Prop :=
  ∃ (mother_cookies : ℕ),
    let remaining_after_brother := initial_cookies - brother_cookies
    let total_before_sister := remaining_after_brother + mother_cookies
    let sister_cookies := (2 : ℚ) / 3 * total_before_sister
    (total_before_sister - sister_cookies = final_cookies) ∧
    (mother_cookies : ℚ) / brother_cookies = 1 / 2

theorem cookie_ratio_proof :
  cookie_problem 20 10 5 :=
sorry

end NUMINAMATH_CALUDE_cookie_ratio_proof_l1390_139073


namespace NUMINAMATH_CALUDE_rectangle_area_l1390_139007

/-- Given a rectangle where the length is four times the width and the perimeter is 250 cm,
    prove that its area is 2500 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 4 * w
  let perimeter := 2 * l + 2 * w
  perimeter = 250 → l * w = 2500 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1390_139007


namespace NUMINAMATH_CALUDE_divided_parallelogram_segments_l1390_139049

/-- A parallelogram with given side lengths and angle, divided by perpendicular lines -/
structure DividedParallelogram where
  a : ℝ  -- Length of one side
  b : ℝ  -- Length of the other side
  α : ℝ  -- Angle between sides
  x : ℝ  -- Length of one segment
  y : ℝ  -- Length of the other segment
  h_a : a = 5
  h_b : b = 13
  h_α : α = Real.arccos (6/13)
  h_perpendicular : x * y = a * b / 4  -- Condition for perpendicular lines
  h_equal_areas : (a - x) * y = (b - y) * x  -- Condition for equal quadrilaterals

/-- The lengths of the segments in the divided parallelogram -/
theorem divided_parallelogram_segments (p : DividedParallelogram) : 
  p.x = 3 ∧ p.y = 39/5 := by
  sorry

end NUMINAMATH_CALUDE_divided_parallelogram_segments_l1390_139049


namespace NUMINAMATH_CALUDE_average_percent_increase_per_year_l1390_139017

def initial_population : ℕ := 175000
def final_population : ℕ := 297500
def time_period : ℕ := 10

theorem average_percent_increase_per_year :
  let total_increase : ℕ := final_population - initial_population
  let average_annual_increase : ℚ := total_increase / time_period
  let percent_increase : ℚ := (average_annual_increase / initial_population) * 100
  percent_increase = 7 := by sorry

end NUMINAMATH_CALUDE_average_percent_increase_per_year_l1390_139017


namespace NUMINAMATH_CALUDE_circle_C_equation_l1390_139066

/-- A circle C with the following properties:
  - The center is on the positive x-axis
  - The radius is √2
  - The circle is tangent to the line x + y = 0
-/
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ
  center_on_x_axis : center.2 = 0
  center_positive : center.1 > 0
  radius_is_sqrt2 : radius = Real.sqrt 2
  tangent_to_line : ∃ (p : ℝ × ℝ), p.1 + p.2 = 0 ∧ 
    (center.1 - p.1)^2 + (center.2 - p.2)^2 = radius^2

/-- The standard equation of circle C is (x-2)² + y² = 2 -/
theorem circle_C_equation (c : CircleC) : 
  ∀ (x y : ℝ), (x - 2)^2 + y^2 = 2 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_C_equation_l1390_139066


namespace NUMINAMATH_CALUDE_old_selling_price_l1390_139019

/-- Given a product with an increased gross profit and new selling price, calculate the old selling price. -/
theorem old_selling_price (cost : ℝ) (new_selling_price : ℝ) : 
  (new_selling_price = cost * 1.15) →  -- New selling price is cost plus 15% profit
  (new_selling_price = 92) →           -- New selling price is $92.00
  (cost * 1.10 = 88) :=                -- Old selling price (cost plus 10% profit) is $88.00
by sorry

end NUMINAMATH_CALUDE_old_selling_price_l1390_139019


namespace NUMINAMATH_CALUDE_four_digit_sum_1989_l1390_139077

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digitSum (n / 10)

/-- Apply the digit sum transformation n times -/
def iterateDigitSum (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | k + 1 => iterateDigitSum (digitSum n) k

/-- The main theorem stating that applying digit sum 4 times to 1989 results in 9 -/
theorem four_digit_sum_1989 : iterateDigitSum 1989 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_sum_1989_l1390_139077


namespace NUMINAMATH_CALUDE_negation_of_or_implies_both_false_l1390_139020

theorem negation_of_or_implies_both_false (p q : Prop) :
  ¬(p ∨ q) → ¬p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_negation_of_or_implies_both_false_l1390_139020


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l1390_139024

/-- A quadratic function f(x) = ax² + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_unique 
  (f : ℝ → ℝ) 
  (h1 : ∃ a b c : ℝ, f = QuadraticFunction a b c) 
  (h2 : f (-2) = 0) 
  (h3 : f 4 = 0) 
  (h4 : ∃ x_max : ℝ, ∀ x : ℝ, f x ≤ f x_max ∧ f x_max = 9) :
  f = QuadraticFunction (-1) 2 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l1390_139024


namespace NUMINAMATH_CALUDE_spring_work_l1390_139094

/-- Work required to stretch a spring -/
theorem spring_work (F : ℝ) (x₀ : ℝ) (x₁ : ℝ) (x₂ : ℝ) (W : ℝ) :
  F = 60 →  -- Given force
  x₀ = 0.02 →  -- Given stretch
  x₁ = 0.14 →  -- Initial length
  x₂ = 0.20 →  -- Final length
  W = 5.4 →  -- Work done
  W = (F / x₀) * (x₂ - x₁)^2 / 2 :=
by sorry


end NUMINAMATH_CALUDE_spring_work_l1390_139094


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l1390_139043

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 1

-- Define the interval
def interval : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (a b : ℝ), a ∈ interval ∧ b ∈ interval ∧
  (∀ x ∈ interval, f x ≤ f a) ∧
  (∀ x ∈ interval, f b ≤ f x) ∧
  f a = 1 ∧ f b = -2 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l1390_139043


namespace NUMINAMATH_CALUDE_ball_count_proof_l1390_139005

theorem ball_count_proof (white green yellow red purple : ℕ)
  (h1 : white = 50)
  (h2 : green = 30)
  (h3 : yellow = 8)
  (h4 : red = 9)
  (h5 : purple = 3)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 88/100) :
  white + green + yellow + red + purple = 100 := by
sorry

end NUMINAMATH_CALUDE_ball_count_proof_l1390_139005


namespace NUMINAMATH_CALUDE_students_attending_game_l1390_139059

/-- Proves the number of students attending a football game -/
theorem students_attending_game (total_attendees : ℕ) (student_price non_student_price : ℕ) (total_revenue : ℕ) : 
  total_attendees = 3000 →
  student_price = 10 →
  non_student_price = 15 →
  total_revenue = 36250 →
  ∃ (students non_students : ℕ),
    students + non_students = total_attendees ∧
    students * student_price + non_students * non_student_price = total_revenue ∧
    students = 1750 :=
by sorry

end NUMINAMATH_CALUDE_students_attending_game_l1390_139059


namespace NUMINAMATH_CALUDE_triangle_areas_l1390_139067

-- Define the triangle ABC
structure Triangle :=
  (BC : ℝ)
  (AC : ℝ)
  (AB : ℝ)

-- Define the areas of the triangles formed by altitude and median
def AreaTriangles (t : Triangle) : (ℝ × ℝ × ℝ) :=
  sorry

-- Theorem statement
theorem triangle_areas (t : Triangle) 
  (h1 : t.BC = 3)
  (h2 : t.AC = 4)
  (h3 : t.AB = 5) :
  AreaTriangles t = (3, 0.84, 2.16) :=
sorry

end NUMINAMATH_CALUDE_triangle_areas_l1390_139067


namespace NUMINAMATH_CALUDE_repeating_decimal_is_rational_l1390_139051

def repeating_decimal (a b c : ℕ) : ℚ :=
  a + b / (10^c.succ * 99)

theorem repeating_decimal_is_rational (a b c : ℕ) :
  ∃ (p q : ℤ), repeating_decimal a b c = p / q ∧ q ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_is_rational_l1390_139051


namespace NUMINAMATH_CALUDE_range_of_b_given_false_proposition_l1390_139003

theorem range_of_b_given_false_proposition :
  (¬ ∃ a : ℝ, a < 0 ∧ a + 1/a > b) →
  ∀ b : ℝ, b ≥ -2 ↔ b ∈ Set.Ici (-2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_b_given_false_proposition_l1390_139003


namespace NUMINAMATH_CALUDE_pet_store_earnings_l1390_139036

theorem pet_store_earnings :
  let num_kittens : ℕ := 2
  let num_puppies : ℕ := 1
  let kitten_price : ℕ := 6
  let puppy_price : ℕ := 5
  (num_kittens * kitten_price + num_puppies * puppy_price : ℕ) = 17 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_earnings_l1390_139036


namespace NUMINAMATH_CALUDE_correct_operation_l1390_139097

theorem correct_operation (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1390_139097


namespace NUMINAMATH_CALUDE_train_distance_l1390_139011

/-- The distance traveled by a train in a given time, given its speed -/
def distance_traveled (speed : ℚ) (time : ℚ) : ℚ :=
  speed * time

/-- Convert hours to minutes -/
def hours_to_minutes (hours : ℚ) : ℚ :=
  hours * 60

theorem train_distance (train_speed : ℚ) (travel_time : ℚ) :
  train_speed = 2 / 2 →
  travel_time = 3 →
  distance_traveled train_speed (hours_to_minutes travel_time) = 180 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l1390_139011


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1390_139087

variable (x y : ℝ)

theorem problem_1 : ((x * y + 2) * (x * y - 2) - 2 * x^2 * y^2 + 4) / (x * y) = -x * y :=
by sorry

theorem problem_2 : (2 * x + y)^2 - (2 * x + 3 * y) * (2 * x - 3 * y) = 4 * x * y + 10 * y^2 :=
by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1390_139087


namespace NUMINAMATH_CALUDE_even_function_implies_a_plus_b_eq_four_l1390_139099

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 3) * x + 3

-- Define the property of being an even function on an interval
def is_even_on (f : ℝ → ℝ) (l r : ℝ) : Prop :=
  ∀ x, l ≤ x ∧ x ≤ r → f x = f (-x)

-- Theorem statement
theorem even_function_implies_a_plus_b_eq_four (a b : ℝ) :
  is_even_on (f a b) (a^2 - 2) a →
  a^2 - 2 ≤ a →
  a + b = 4 := by
  sorry


end NUMINAMATH_CALUDE_even_function_implies_a_plus_b_eq_four_l1390_139099


namespace NUMINAMATH_CALUDE_not_all_points_satisfy_equation_tan_eq_one_not_same_as_pi_over_four_rho_3_same_as_neg_3_l1390_139078

-- Define a polar coordinate system
structure PolarCoordinate where
  r : ℝ
  θ : ℝ

-- Define a curve in polar coordinates
def PolarCurve := PolarCoordinate → Prop

-- Statement 1
theorem not_all_points_satisfy_equation (C : PolarCurve) :
  ¬ ∀ (P : PolarCoordinate), C P → (∀ (eq : PolarCoordinate → Prop), (∀ Q, C Q → eq Q) → eq P) :=
sorry

-- Statement 2
theorem tan_eq_one_not_same_as_pi_over_four :
  ∃ (P : PolarCoordinate), (Real.tan P.θ = 1) ≠ (P.θ = π / 4) :=
sorry

-- Statement 3
theorem rho_3_same_as_neg_3 :
  ∀ (P : PolarCoordinate), P.r = 3 ↔ P.r = -3 :=
sorry

end NUMINAMATH_CALUDE_not_all_points_satisfy_equation_tan_eq_one_not_same_as_pi_over_four_rho_3_same_as_neg_3_l1390_139078


namespace NUMINAMATH_CALUDE_sales_tax_theorem_l1390_139082

/-- Calculates the sales tax paid given total purchase, tax rate, and cost of tax-free items -/
def calculate_sales_tax (total_purchase : ℝ) (tax_rate : ℝ) (tax_free_cost : ℝ) : ℝ :=
  let taxable_cost := total_purchase - tax_free_cost
  tax_rate * taxable_cost

/-- Theorem stating that under the given conditions, the sales tax paid is 0.3 -/
theorem sales_tax_theorem (total_purchase tax_rate tax_free_cost : ℝ) 
  (h1 : total_purchase = 25)
  (h2 : tax_rate = 0.06)
  (h3 : tax_free_cost = 19.7) :
  calculate_sales_tax total_purchase tax_rate tax_free_cost = 0.3 := by
  sorry

#eval calculate_sales_tax 25 0.06 19.7

end NUMINAMATH_CALUDE_sales_tax_theorem_l1390_139082


namespace NUMINAMATH_CALUDE_meeting_time_proof_l1390_139031

/-- 
Given two people traveling towards each other on a 600 km route, 
one at 70 km/hr and the other at 80 km/hr, prove that they meet 
after traveling for 4 hours.
-/
theorem meeting_time_proof (total_distance : ℝ) (speed1 speed2 : ℝ) (t : ℝ) : 
  total_distance = 600 →
  speed1 = 70 →
  speed2 = 80 →
  speed1 * t + speed2 * t = total_distance →
  t = 4 := by
sorry

end NUMINAMATH_CALUDE_meeting_time_proof_l1390_139031


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l1390_139098

/-- The number of ways to distribute n different balls into k different boxes -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n different balls into k different boxes with exactly m empty boxes -/
def distributeWithEmpty (n k m : ℕ) : ℕ := sorry

theorem ball_distribution_theorem (n k : ℕ) (hn : n = 4) (hk : k = 4) :
  distribute n k = 256 ∧
  distributeWithEmpty n k 1 = 144 ∧
  distributeWithEmpty n k 2 = 84 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_theorem_l1390_139098


namespace NUMINAMATH_CALUDE_product_divisible_by_504_l1390_139095

theorem product_divisible_by_504 (a : ℕ) : 
  504 ∣ (a^3 - 1) * a^3 * (a^3 + 1) := by
sorry

end NUMINAMATH_CALUDE_product_divisible_by_504_l1390_139095


namespace NUMINAMATH_CALUDE_school_pupils_count_l1390_139063

theorem school_pupils_count (girls : ℕ) (boys : ℕ) (teachers : ℕ) : girls = 308 → boys = 318 → teachers = 36 → girls + boys = 626 := by
  sorry

end NUMINAMATH_CALUDE_school_pupils_count_l1390_139063


namespace NUMINAMATH_CALUDE_linear_equation_implies_a_equals_negative_two_l1390_139064

theorem linear_equation_implies_a_equals_negative_two (a : ℝ) : 
  (∀ x, (a - 2) * x^(|a| - 1) - 2 = 0 → ∃ m k, (a - 2) * x^(|a| - 1) - 2 = m * x + k) → 
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_implies_a_equals_negative_two_l1390_139064


namespace NUMINAMATH_CALUDE_quadratic_function_passes_through_points_l1390_139041

/-- A quadratic function f(x) = x^2 + 2x - 3 -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

/-- The given points that the function should pass through -/
def points : List (ℝ × ℝ) := [(-2, -3), (-1, -4), (0, -3), (2, 5)]

theorem quadratic_function_passes_through_points :
  ∀ (p : ℝ × ℝ), p ∈ points → f p.1 = p.2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_passes_through_points_l1390_139041


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_neg_two_l1390_139074

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The line ax + y - 1 = 0 is parallel to the line 2x - y + 2 = 0 -/
def lines_are_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, y = -a * x + 1 ↔ y = 2 * x + 2

theorem parallel_lines_imply_a_eq_neg_two :
  ∀ a : ℝ, lines_are_parallel a → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_neg_two_l1390_139074


namespace NUMINAMATH_CALUDE_tape_shortage_l1390_139084

/-- Proves that 180 feet of tape is insufficient to wrap around a 35x80 foot field and three 5-foot circumference trees, requiring an additional 65 feet. -/
theorem tape_shortage (field_width : ℝ) (field_length : ℝ) (tree_circumference : ℝ) (num_trees : ℕ) (available_tape : ℝ) : 
  field_width = 35 → 
  field_length = 80 → 
  tree_circumference = 5 → 
  num_trees = 3 → 
  available_tape = 180 → 
  (2 * (field_width + field_length) + num_trees * tree_circumference) - available_tape = 65 := by
  sorry

end NUMINAMATH_CALUDE_tape_shortage_l1390_139084


namespace NUMINAMATH_CALUDE_simplify_expression_l1390_139096

theorem simplify_expression (x y : ℝ) :
  (25 * x + 70 * y) + (15 * x + 34 * y) - (13 * x + 55 * y) = 27 * x + 49 * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1390_139096


namespace NUMINAMATH_CALUDE_increase_in_average_goals_is_point_two_l1390_139023

/-- Calculates the increase in average goals score after the fifth match -/
def increase_in_average_goals (total_matches : ℕ) (total_goals : ℕ) (goals_in_fifth_match : ℕ) : ℚ :=
  let goals_before_fifth := total_goals - goals_in_fifth_match
  let matches_before_fifth := total_matches - 1
  let average_before := goals_before_fifth / matches_before_fifth
  let average_after := total_goals / total_matches
  average_after - average_before

/-- The increase in average goals score after the fifth match is 0.2 -/
theorem increase_in_average_goals_is_point_two :
  increase_in_average_goals 5 21 5 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_increase_in_average_goals_is_point_two_l1390_139023


namespace NUMINAMATH_CALUDE_flamingo_percentage_among_non_parrots_l1390_139090

/-- Given the distribution of birds in a wildlife reserve, this theorem proves
    that flamingos constitute 50% of the non-parrot birds. -/
theorem flamingo_percentage_among_non_parrots :
  let total_percentage : ℝ := 100
  let flamingo_percentage : ℝ := 40
  let parrot_percentage : ℝ := 20
  let eagle_percentage : ℝ := 15
  let owl_percentage : ℝ := total_percentage - flamingo_percentage - parrot_percentage - eagle_percentage
  let non_parrot_percentage : ℝ := total_percentage - parrot_percentage
  (flamingo_percentage / non_parrot_percentage) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_flamingo_percentage_among_non_parrots_l1390_139090


namespace NUMINAMATH_CALUDE_min_queries_theorem_l1390_139079

/-- Represents a card with either +1 or -1 written on it -/
inductive Card : Type
| plus_one : Card
| minus_one : Card

/-- Represents a deck of cards -/
def Deck := List Card

/-- Represents a query function that returns the product of three cards -/
def Query := Card → Card → Card → Int

/-- The minimum number of queries needed to determine the product of all cards in a deck -/
def min_queries (n : Nat) (circular : Bool) : Nat :=
  match n with
  | 30 => 10
  | 31 => 11
  | 32 => 12
  | 50 => if circular then 50 else 17  -- 17 is a placeholder for the non-circular case
  | _ => 0  -- placeholder for other cases

/-- Theorem stating the minimum number of queries needed for specific deck sizes -/
theorem min_queries_theorem (d : Deck) (q : Query) :
  (d.length = 30 → min_queries 30 false = 10) ∧
  (d.length = 31 → min_queries 31 false = 11) ∧
  (d.length = 32 → min_queries 32 false = 12) ∧
  (d.length = 50 → min_queries 50 true = 50) :=
sorry

end NUMINAMATH_CALUDE_min_queries_theorem_l1390_139079


namespace NUMINAMATH_CALUDE_commercial_time_calculation_l1390_139010

theorem commercial_time_calculation (num_programs : ℕ) (program_duration : ℕ) (commercial_fraction : ℚ) : 
  num_programs = 6 → 
  program_duration = 30 → 
  commercial_fraction = 1/4 → 
  (↑num_programs * ↑program_duration : ℚ) * commercial_fraction = 45 := by
  sorry

end NUMINAMATH_CALUDE_commercial_time_calculation_l1390_139010


namespace NUMINAMATH_CALUDE_dice_probability_l1390_139060

def num_dice : ℕ := 8
def num_sides : ℕ := 8
def num_favorable : ℕ := 4

theorem dice_probability :
  let p_first_die : ℚ := (num_favorable : ℚ) / num_sides
  let p_remaining : ℚ := 1 / 2
  let combinations : ℕ := Nat.choose (num_dice - 1) (num_favorable - 1)
  p_first_die * combinations * p_remaining ^ (num_dice - 1) = 35 / 256 := by
    sorry

end NUMINAMATH_CALUDE_dice_probability_l1390_139060


namespace NUMINAMATH_CALUDE_altitude_equation_l1390_139072

/-- Given points A, B, and C in a plane, this theorem states that 
    the equation x + 3y - 9 = 0 represents the altitude from A in triangle ABC. -/
theorem altitude_equation (A B C : ℝ × ℝ) : 
  A = (6, 1) → B = (-5, -4) → C = (-2, 5) → 
  ∀ (x y : ℝ), (x + 3*y - 9 = 0) ↔ 
  (∃ (t : ℝ), (x, y) = (6 + t, 1 - t/3) ∧ 
   ((x - 6) * ((-2) - (-5)) + (y - 1) * (5 - (-4)) = 0)) := by
  sorry

end NUMINAMATH_CALUDE_altitude_equation_l1390_139072


namespace NUMINAMATH_CALUDE_f_properties_l1390_139009

noncomputable def f (x : ℝ) : ℝ := x + Real.cos x

theorem f_properties :
  (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧
  (f (π / 2) = π / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1390_139009


namespace NUMINAMATH_CALUDE_popton_bus_toes_count_l1390_139091

/-- Represents the three races on planet Popton -/
inductive Race
  | Hoopit
  | Neglart
  | Zentorian

/-- Returns the number of toes per hand for a given race -/
def toesPerHand (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 3
  | Race.Neglart => 2
  | Race.Zentorian => 4

/-- Returns the number of hands for a given race -/
def handsCount (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 4
  | Race.Neglart => 5
  | Race.Zentorian => 6

/-- Returns the number of students of a given race on the bus -/
def studentsCount (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 7
  | Race.Neglart => 8
  | Race.Zentorian => 5

/-- Calculates the total number of toes for a given race on the bus -/
def totalToesForRace (r : Race) : ℕ :=
  toesPerHand r * handsCount r * studentsCount r

/-- Theorem: The total number of toes on the Popton school bus is 284 -/
theorem popton_bus_toes_count :
  (totalToesForRace Race.Hoopit) + (totalToesForRace Race.Neglart) + (totalToesForRace Race.Zentorian) = 284 := by
  sorry

end NUMINAMATH_CALUDE_popton_bus_toes_count_l1390_139091


namespace NUMINAMATH_CALUDE_percentage_calculation_l1390_139070

theorem percentage_calculation (x : ℝ) (h : 70 = 0.56 * x) : 1.25 * x = 156.25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1390_139070


namespace NUMINAMATH_CALUDE_letter_sum_equals_fifteen_l1390_139001

/-- Given a mapping of letters to numbers where A = 0, B = 1, C = 2, ..., Z = 25,
    prove that the sum of A + B + M + C equals 15. -/
theorem letter_sum_equals_fifteen :
  let letter_to_num : Char → ℕ := fun c => c.toNat - 65
  letter_to_num 'A' + letter_to_num 'B' + letter_to_num 'M' + letter_to_num 'C' = 15 := by
  sorry

end NUMINAMATH_CALUDE_letter_sum_equals_fifteen_l1390_139001


namespace NUMINAMATH_CALUDE_remaining_flowers_l1390_139048

/-- Represents the flower arrangement along the path --/
structure FlowerPath :=
  (peonies : Nat)
  (tulips : Nat)
  (watered : Nat)
  (unwatered : Nat)
  (picked_tulips : Nat)

/-- Theorem stating the number of remaining flowers after Neznayka's picking --/
theorem remaining_flowers (path : FlowerPath) 
  (h1 : path.peonies = 15)
  (h2 : path.tulips = 15)
  (h3 : path.unwatered = 10)
  (h4 : path.watered + path.unwatered = path.peonies + path.tulips)
  (h5 : path.picked_tulips = 6) :
  path.watered - path.picked_tulips = 19 := by
  sorry

#check remaining_flowers

end NUMINAMATH_CALUDE_remaining_flowers_l1390_139048


namespace NUMINAMATH_CALUDE_not_prime_n4_plus_n2_plus_1_l1390_139061

theorem not_prime_n4_plus_n2_plus_1 (n : ℕ) (h : n > 1) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + n^2 + 1 = a * b :=
by
  sorry

end NUMINAMATH_CALUDE_not_prime_n4_plus_n2_plus_1_l1390_139061


namespace NUMINAMATH_CALUDE_gcd_266_209_l1390_139006

theorem gcd_266_209 : Nat.gcd 266 209 = 19 := by
  sorry

end NUMINAMATH_CALUDE_gcd_266_209_l1390_139006


namespace NUMINAMATH_CALUDE_problem_solution_l1390_139042

theorem problem_solution (x y t : ℝ) 
  (h1 : 2^x = t) 
  (h2 : 7^y = t) 
  (h3 : 1/x + 1/y = 2) : 
  t = Real.sqrt 14 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1390_139042


namespace NUMINAMATH_CALUDE_circle_tangent_to_three_lines_l1390_139037

-- Define the types for lines and circles
variable (Line Circle : Type)

-- Define the tangent relation between a circle and a line
variable (tangent_to : Circle → Line → Prop)

-- Define the intersection angle between two lines
variable (intersection_angle : Line → Line → ℝ)

-- Define the main theorem
theorem circle_tangent_to_three_lines 
  (C : Circle) (l m n : Line) :
  (tangent_to C l ∧ tangent_to C m ∧ tangent_to C n) →
  (∃ (C' : Circle), 
    tangent_to C' l ∧ tangent_to C' m ∧ tangent_to C' n) ∧
  (intersection_angle l m = π/3 ∧ 
   intersection_angle m n = π/3 ∧ 
   intersection_angle n l = π/3) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_three_lines_l1390_139037


namespace NUMINAMATH_CALUDE_number_problem_l1390_139028

theorem number_problem : 
  ∃ x : ℝ, x - (3/5) * x = 50 → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1390_139028


namespace NUMINAMATH_CALUDE_ellipse_problem_l1390_139035

-- Define the points F₁ and F₂
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the distance condition for point P
def distance_condition (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 4

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 2 * (x + 1)

-- Define the perpendicular condition
def perpendicular_condition (M A : ℝ × ℝ) : Prop :=
  (A.2 - M.2) = -1/2 * (A.1 - M.1)

-- Theorem statement
theorem ellipse_problem :
  ∀ (P : ℝ × ℝ), distance_condition P →
  (∀ x y, trajectory_C x y ↔ (x, y) = P) ∧
  (∃ (M : ℝ × ℝ), trajectory_C M.1 M.2 ∧
    ∀ (A : ℝ × ℝ), line_l A.1 A.2 ∧ perpendicular_condition M A →
    Real.sqrt ((A.1 - F₁.1)^2 + (A.2 - F₁.2)^2) ≤ Real.sqrt 5) ∧
  (∃ (M : ℝ × ℝ), M = (1, 3/2) ∧ trajectory_C M.1 M.2 ∧
    ∃ (A : ℝ × ℝ), line_l A.1 A.2 ∧ perpendicular_condition M A ∧
    Real.sqrt ((A.1 - F₁.1)^2 + (A.2 - F₁.2)^2) = Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_problem_l1390_139035


namespace NUMINAMATH_CALUDE_min_points_to_guarantee_win_no_smaller_guarantee_l1390_139000

/-- Represents the points earned in a single race -/
inductive RaceResult
| First  : RaceResult
| Second : RaceResult
| Third  : RaceResult

/-- Converts a race result to points -/
def points (result : RaceResult) : Nat :=
  match result with
  | RaceResult.First  => 6
  | RaceResult.Second => 4
  | RaceResult.Third  => 2

/-- Calculates the total points from a list of race results -/
def totalPoints (results : List RaceResult) : Nat :=
  results.map points |>.sum

/-- Represents the results of three races -/
def ThreeRaces := (RaceResult × RaceResult × RaceResult)

/-- Theorem: 16 points is the minimum to guarantee winning -/
theorem min_points_to_guarantee_win :
  ∀ (other : ThreeRaces),
  ∃ (winner : ThreeRaces),
    totalPoints (winner.1 :: winner.2.1 :: [winner.2.2]) = 16 ∧
    totalPoints (winner.1 :: winner.2.1 :: [winner.2.2]) >
    totalPoints (other.1 :: other.2.1 :: [other.2.2]) :=
  sorry

/-- Theorem: No smaller number of points can guarantee winning -/
theorem no_smaller_guarantee :
  ∀ (n : Nat),
  n < 16 →
  ∃ (player1 player2 : ThreeRaces),
    totalPoints (player1.1 :: player1.2.1 :: [player1.2.2]) = n ∧
    totalPoints (player2.1 :: player2.2.1 :: [player2.2.2]) ≥ n :=
  sorry

end NUMINAMATH_CALUDE_min_points_to_guarantee_win_no_smaller_guarantee_l1390_139000


namespace NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l1390_139071

theorem greatest_whole_number_satisfying_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ (5 * x - 4 : ℝ) < (3 - 2 * x : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_greatest_whole_number_satisfying_inequality_l1390_139071


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l1390_139062

theorem square_plus_inverse_square (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l1390_139062


namespace NUMINAMATH_CALUDE_number_added_before_division_l1390_139004

theorem number_added_before_division (x n : ℤ) : 
  (∃ k : ℤ, x = 82 * k + 5) →
  (∃ m : ℤ, x + n = 41 * m + 22) →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_number_added_before_division_l1390_139004


namespace NUMINAMATH_CALUDE_total_puppies_l1390_139029

def puppies_week1 : ℕ := 20

def puppies_week2 : ℕ := (2 * puppies_week1) / 5

def puppies_week3 : ℕ := 2 * puppies_week2

def puppies_week4 : ℕ := puppies_week1 + 10

theorem total_puppies : 
  puppies_week1 + puppies_week2 + puppies_week3 + puppies_week4 = 74 := by
  sorry

end NUMINAMATH_CALUDE_total_puppies_l1390_139029


namespace NUMINAMATH_CALUDE_sum_of_squares_l1390_139069

theorem sum_of_squares (x y z a b c : ℝ) 
  (h1 : x * y = a) 
  (h2 : x * z = b) 
  (h3 : y * z = c) 
  (h4 : x ≠ 0) 
  (h5 : y ≠ 0) 
  (h6 : z ≠ 0) 
  (h7 : a ≠ 0) 
  (h8 : b ≠ 0) 
  (h9 : c ≠ 0) : 
  x^2 + y^2 + z^2 = ((a*b)^2 + (a*c)^2 + (b*c)^2) / (a*b*c) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1390_139069


namespace NUMINAMATH_CALUDE_repair_cost_percentage_l1390_139046

def apple_price : ℚ := 5/4
def bike_cost : ℚ := 80
def apples_sold : ℕ := 20
def remaining_fraction : ℚ := 1/5

theorem repair_cost_percentage :
  let total_earned : ℚ := apple_price * apples_sold
  let repair_cost : ℚ := total_earned * (1 - remaining_fraction)
  repair_cost / bike_cost = 1/4
:= by sorry

end NUMINAMATH_CALUDE_repair_cost_percentage_l1390_139046


namespace NUMINAMATH_CALUDE_tony_graduate_degree_time_l1390_139086

/-- Time spent on graduate degree in physics -/
def graduate_degree_time (first_degree_time additional_degree_time number_of_additional_degrees total_school_time : ℕ) : ℕ :=
  total_school_time - (first_degree_time + additional_degree_time * number_of_additional_degrees)

/-- Theorem stating that Tony's graduate degree time is 2 years -/
theorem tony_graduate_degree_time :
  graduate_degree_time 4 4 2 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tony_graduate_degree_time_l1390_139086


namespace NUMINAMATH_CALUDE_james_muffins_l1390_139065

theorem james_muffins (arthur_muffins : Float) (ratio : Float) (james_muffins : Float)
  (h1 : arthur_muffins = 115.0)
  (h2 : ratio = 12.0)
  (h3 : arthur_muffins = ratio * james_muffins) :
  james_muffins = arthur_muffins / ratio := by
sorry

end NUMINAMATH_CALUDE_james_muffins_l1390_139065


namespace NUMINAMATH_CALUDE_train_crossing_time_l1390_139092

/-- Calculates the time it takes for a train to cross a platform -/
theorem train_crossing_time (train_speed_kmph : ℝ) (man_crossing_time : ℝ) (platform_length : ℝ) : 
  train_speed_kmph = 72 →
  man_crossing_time = 18 →
  platform_length = 280 →
  (platform_length + train_speed_kmph * man_crossing_time * (5/18)) / (train_speed_kmph * (5/18)) = 32 := by
  sorry


end NUMINAMATH_CALUDE_train_crossing_time_l1390_139092


namespace NUMINAMATH_CALUDE_sin_36_degrees_l1390_139039

theorem sin_36_degrees : 
  Real.sin (36 * π / 180) = (1 / 4) * Real.sqrt (10 - 2 * Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_sin_36_degrees_l1390_139039


namespace NUMINAMATH_CALUDE_star_composition_l1390_139093

-- Define the star operations
def star_right (y : ℝ) : ℝ := 10 - y
def star_left (y : ℝ) : ℝ := y - 10

-- State the theorem
theorem star_composition : star_left (star_right 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_star_composition_l1390_139093


namespace NUMINAMATH_CALUDE_all_routes_have_eight_stations_l1390_139075

/-- Represents a bus route in the city -/
structure BusRoute where
  stations : Set Nat
  station_count : Nat

/-- Represents the city's bus network -/
structure BusNetwork where
  routes : Finset BusRoute
  route_count : Nat

/-- Conditions for the bus network -/
def valid_network (n : BusNetwork) : Prop :=
  -- There are 57 bus routes
  n.route_count = 57 ∧
  -- Any two routes share exactly one station
  ∀ r1 r2 : BusRoute, r1 ∈ n.routes ∧ r2 ∈ n.routes ∧ r1 ≠ r2 →
    ∃! s : Nat, s ∈ r1.stations ∧ s ∈ r2.stations ∧
  -- Each route has at least 3 stations
  ∀ r : BusRoute, r ∈ n.routes → r.station_count ≥ 3 ∧
  -- From any station, it's possible to reach any other station without changing buses
  ∀ s1 s2 : Nat, ∃ r : BusRoute, r ∈ n.routes ∧ s1 ∈ r.stations ∧ s2 ∈ r.stations

/-- The main theorem to prove -/
theorem all_routes_have_eight_stations (n : BusNetwork) (h : valid_network n) :
  ∀ r : BusRoute, r ∈ n.routes → r.station_count = 8 := by
  sorry

end NUMINAMATH_CALUDE_all_routes_have_eight_stations_l1390_139075


namespace NUMINAMATH_CALUDE_triangle_properties_l1390_139056

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.c * Real.cos t.B = (2 * t.a - t.b) * Real.cos t.C ∧ t.c = 4

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.C = π / 3 ∧
  (∀ s : ℝ, s = 1/2 * t.a * t.b * Real.sin t.C → s ≤ 4 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1390_139056


namespace NUMINAMATH_CALUDE_hyperbola_focus_l1390_139040

/-- Given a hyperbola with equation (x-1)^2/7^2 - (y+8)^2/3^2 = 1,
    the coordinates of the focus with the smaller x-coordinate are (1 - √58, -8) -/
theorem hyperbola_focus (x y : ℝ) :
  (x - 1)^2 / 7^2 - (y + 8)^2 / 3^2 = 1 →
  ∃ (focus_x focus_y : ℝ),
    focus_x = 1 - Real.sqrt 58 ∧
    focus_y = -8 ∧
    ∀ (other_focus_x : ℝ),
      ((other_focus_x - 1)^2 / 7^2 - (focus_y + 8)^2 / 3^2 = 1 →
       other_focus_x ≥ focus_x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l1390_139040


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1390_139016

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Theorem: In an arithmetic sequence where a_4 = 5, a_3 + a_5 = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) (h_a4 : a 4 = 5) : 
  a 3 + a 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1390_139016


namespace NUMINAMATH_CALUDE_binary_sum_equality_l1390_139055

/-- Prove that the binary sum 1111₂ + 110₂ - 1001₂ + 1110₂ equals 11100₂ --/
theorem binary_sum_equality : 
  (0b1111 : Nat) + 0b110 - 0b1001 + 0b1110 = 0b11100 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equality_l1390_139055


namespace NUMINAMATH_CALUDE_operations_correct_l1390_139013

-- Define the operations
def operation3 (x : ℝ) : Prop := x ≠ 0 → x^6 / x^3 = x^3
def operation4 (x : ℝ) : Prop := (x^3)^2 = x^6

-- Theorem stating that both operations are correct
theorem operations_correct : 
  (∀ x : ℝ, operation3 x) ∧ (∀ x : ℝ, operation4 x) := by sorry

end NUMINAMATH_CALUDE_operations_correct_l1390_139013


namespace NUMINAMATH_CALUDE_alice_painted_six_cuboids_l1390_139012

/-- The number of faces on a single cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The total number of faces Alice painted -/
def total_painted_faces : ℕ := 36

/-- The number of cuboids Alice painted -/
def num_cuboids : ℕ := total_painted_faces / faces_per_cuboid

theorem alice_painted_six_cuboids :
  num_cuboids = 6 :=
sorry

end NUMINAMATH_CALUDE_alice_painted_six_cuboids_l1390_139012


namespace NUMINAMATH_CALUDE_ajay_walking_distance_l1390_139032

/-- Ajay's walking problem -/
theorem ajay_walking_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (h1 : speed = 3) 
  (h2 : time = 16.666666666666668) : 
  speed * time = 50 := by
  sorry

end NUMINAMATH_CALUDE_ajay_walking_distance_l1390_139032


namespace NUMINAMATH_CALUDE_max_xy_value_l1390_139088

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 140) : x * y ≤ 168 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l1390_139088


namespace NUMINAMATH_CALUDE_vector_relations_l1390_139038

/-- Two vectors in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Define vector a -/
def a : Vector2D := ⟨1, 1⟩

/-- Define vector b with parameter m -/
def b (m : ℝ) : Vector2D := ⟨2, m⟩

/-- Two vectors are parallel if their components are proportional -/
def parallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : Vector2D) : Prop :=
  v.x * w.x + v.y * w.y = 0

/-- Main theorem -/
theorem vector_relations :
  (∀ m : ℝ, parallel a (b m) → m = 2) ∧
  (∀ m : ℝ, perpendicular a (b m) → m = -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_relations_l1390_139038


namespace NUMINAMATH_CALUDE_selling_prices_correct_l1390_139052

def calculate_selling_price (cost : ℚ) (profit_percent : ℚ) (tax_percent : ℚ) : ℚ :=
  let pre_tax_price := cost * (1 + profit_percent)
  pre_tax_price * (1 + tax_percent)

theorem selling_prices_correct : 
  let cost_A : ℚ := 650
  let cost_B : ℚ := 1200
  let cost_C : ℚ := 800
  let profit_A : ℚ := 1/10
  let profit_B : ℚ := 3/20
  let profit_C : ℚ := 1/5
  let tax : ℚ := 1/20
  
  (calculate_selling_price cost_A profit_A tax = 75075/100) ∧
  (calculate_selling_price cost_B profit_B tax = 1449) ∧
  (calculate_selling_price cost_C profit_C tax = 1008) :=
by sorry

end NUMINAMATH_CALUDE_selling_prices_correct_l1390_139052


namespace NUMINAMATH_CALUDE_parabola_theorem_l1390_139054

-- Define the parabola C
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus F
def F : ℝ × ℝ := (1, 0)

-- Define point K
def K : ℝ × ℝ := (-1, 0)

-- Define line l passing through K
def l (m : ℝ) (x y : ℝ) : Prop := x = m*y - 1

-- Define the condition for A and B being on C and l
def intersectionPoints (m : ℝ) (A B : ℝ × ℝ) : Prop :=
  C A.1 A.2 ∧ C B.1 B.2 ∧ l m A.1 A.2 ∧ l m B.1 B.2

-- Define the symmetry condition for A and D
def symmetricPoints (A D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ A.2 = -D.2

-- Define the dot product condition
def dotProductCondition (A B : ℝ × ℝ) : Prop :=
  (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = 8/9

-- State the theorem
theorem parabola_theorem (m : ℝ) (A B D : ℝ × ℝ) :
  intersectionPoints m A B →
  symmetricPoints A D →
  dotProductCondition A B →
  (∃ (t : ℝ), F.1 = D.1 + t * (B.1 - D.1) ∧ F.2 = D.2 + t * (B.2 - D.2)) ∧
  (∃ (M : ℝ × ℝ), M.1 = 1/9 ∧ M.2 = 0 ∧
    ∀ (x y : ℝ), (x - M.1)^2 + (y - M.2)^2 = 4/9 →
      (x - K.1)^2 + (y - K.2)^2 ≥ 4/9 ∧
      (x - B.1)^2 + (y - B.2)^2 ≥ 4/9 ∧
      (x - D.1)^2 + (y - D.2)^2 ≥ 4/9) :=
by sorry

end NUMINAMATH_CALUDE_parabola_theorem_l1390_139054


namespace NUMINAMATH_CALUDE_min_value_theorem_l1390_139085

open Real

theorem min_value_theorem (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  let f := fun x => 1/x + 9/(1-x)
  (∀ y, 0 < y ∧ y < 1 → f y ≥ 16) ∧ (∃ z, 0 < z ∧ z < 1 ∧ f z = 16) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1390_139085


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1390_139068

/-- A quadratic function with vertex (3, 2) passing through (-2, -18) has a = -4/5 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Condition 1
  (2 = a * 3^2 + b * 3 + c) →             -- Condition 2 (vertex)
  (3 = -b / (2 * a)) →                    -- Condition 2 (vertex x-coordinate)
  (-18 = a * (-2)^2 + b * (-2) + c) →     -- Condition 3
  a = -4/5 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1390_139068
