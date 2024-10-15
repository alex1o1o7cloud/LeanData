import Mathlib

namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l250_25050

/-- The number of ways to choose k items from a set of n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The problem statement -/
theorem pizza_toppings_combinations :
  choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l250_25050


namespace NUMINAMATH_CALUDE_three_numbers_sum_l250_25039

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- a, b, c are in ascending order
  (a + b + c) / 3 = a + 8 ∧  -- mean is 8 more than least
  (a + b + c) / 3 = c - 20 ∧  -- mean is 20 less than greatest
  b = 10  -- median is 10
  → a + b + c = 66 := by sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l250_25039


namespace NUMINAMATH_CALUDE_odd_numbers_with_difference_16_are_coprime_l250_25081

theorem odd_numbers_with_difference_16_are_coprime 
  (a b : ℤ) 
  (ha : Odd a) 
  (hb : Odd b) 
  (hdiff : |a - b| = 16) : 
  Int.gcd a b = 1 := by
sorry

end NUMINAMATH_CALUDE_odd_numbers_with_difference_16_are_coprime_l250_25081


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l250_25078

theorem simplify_sqrt_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l250_25078


namespace NUMINAMATH_CALUDE_value_of_c_l250_25014

theorem value_of_c (a c : ℕ) (h1 : a = 105) (h2 : a^5 = 3^3 * 5^2 * 7^2 * 11^2 * 13 * c) : c = 385875 := by
  sorry

end NUMINAMATH_CALUDE_value_of_c_l250_25014


namespace NUMINAMATH_CALUDE_cubic_inequality_l250_25035

theorem cubic_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 + a + b ≥ 4*a*b := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l250_25035


namespace NUMINAMATH_CALUDE_amanda_earnings_l250_25031

/-- Amanda's hourly rate in dollars -/
def hourly_rate : ℝ := 20

/-- Hours worked on Monday -/
def monday_hours : ℝ := 5 * 1.5

/-- Hours worked on Tuesday -/
def tuesday_hours : ℝ := 3

/-- Hours worked on Thursday -/
def thursday_hours : ℝ := 2 * 2

/-- Hours worked on Saturday -/
def saturday_hours : ℝ := 6

/-- Total hours worked in the week -/
def total_hours : ℝ := monday_hours + tuesday_hours + thursday_hours + saturday_hours

/-- Amanda's earnings for the week -/
def weekly_earnings : ℝ := total_hours * hourly_rate

theorem amanda_earnings : weekly_earnings = 410 := by
  sorry

end NUMINAMATH_CALUDE_amanda_earnings_l250_25031


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l250_25082

theorem smallest_square_containing_circle (r : ℝ) (h : r = 5) : 
  (2 * r) ^ 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l250_25082


namespace NUMINAMATH_CALUDE_probability_not_snowing_l250_25090

theorem probability_not_snowing (p_snowing : ℚ) (h : p_snowing = 5/8) : 
  1 - p_snowing = 3/8 := by
sorry

end NUMINAMATH_CALUDE_probability_not_snowing_l250_25090


namespace NUMINAMATH_CALUDE_journey_time_calculation_l250_25033

/-- Calculates the time spent on the road given start time, end time, and total stop time. -/
def timeOnRoad (startTime endTime stopTime : ℕ) : ℕ :=
  (endTime - startTime) - stopTime

/-- Proves that for a journey from 7:00 AM to 8:00 PM with 60 minutes of stops, the time on the road is 12 hours. -/
theorem journey_time_calculation :
  let startTime : ℕ := 7  -- 7:00 AM
  let endTime : ℕ := 20   -- 8:00 PM (20:00 in 24-hour format)
  let stopTime : ℕ := 1   -- 60 minutes = 1 hour
  timeOnRoad startTime endTime stopTime = 12 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_calculation_l250_25033


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l250_25052

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l250_25052


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_450_l250_25011

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_450 :
  let M := sum_of_divisors 450
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ M ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ M → q ≤ p ∧ p = 31 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_450_l250_25011


namespace NUMINAMATH_CALUDE_min_value_inequality_l250_25040

theorem min_value_inequality (a b c d : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (d/c - 1)^2 + (5/d - 1)^2 ≥ 5 * (5^(1/5) - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l250_25040


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l250_25012

/-- An isosceles right triangle with an inscribed square -/
structure IsoscelesRightTriangleWithSquare where
  /-- Length of the leg of the isosceles right triangle -/
  a : ℝ
  /-- Side length of the inscribed square -/
  s : ℝ
  /-- The triangle is isosceles and right-angled -/
  isIsoscelesRight : True
  /-- The square is inscribed with one vertex on the hypotenuse -/
  squareOnHypotenuse : True
  /-- The square has one vertex at the right angle of the triangle -/
  squareAtRightAngle : True
  /-- The square has two vertices on the legs of the triangle -/
  squareOnLegs : True
  /-- The leg length is positive -/
  a_pos : 0 < a

/-- The side length of the inscribed square is half the leg length of the triangle -/
theorem inscribed_square_side_length 
  (triangle : IsoscelesRightTriangleWithSquare) : 
  triangle.s = triangle.a / 2 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_square_side_length_l250_25012


namespace NUMINAMATH_CALUDE_class_average_calculation_l250_25077

theorem class_average_calculation (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) : 
  total_students = 20 → 
  excluded_students = 5 → 
  excluded_avg = 50 → 
  remaining_avg = 90 → 
  (total_students * (total_students * remaining_avg - excluded_students * remaining_avg + 
   excluded_students * excluded_avg)) / (total_students * total_students) = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_average_calculation_l250_25077


namespace NUMINAMATH_CALUDE_john_january_savings_l250_25094

theorem john_january_savings :
  let base_income : ℝ := 2000
  let bonus_rate : ℝ := 0.15
  let transport_rate : ℝ := 0.05
  let rent : ℝ := 500
  let utilities : ℝ := 100
  let food : ℝ := 300
  let misc_rate : ℝ := 0.10

  let total_income : ℝ := base_income * (1 + bonus_rate)
  let transport_expense : ℝ := total_income * transport_rate
  let misc_expense : ℝ := total_income * misc_rate
  let total_expenses : ℝ := transport_expense + rent + utilities + food + misc_expense
  let savings : ℝ := total_income - total_expenses

  savings = 1055 := by sorry

end NUMINAMATH_CALUDE_john_january_savings_l250_25094


namespace NUMINAMATH_CALUDE_invalid_domain_l250_25016

def f (x : ℝ) : ℝ := x^2

def N : Set ℝ := {1, 2}

theorem invalid_domain : ¬(∀ x ∈ ({1, Real.sqrt 2, 2} : Set ℝ), f x ∈ N) := by
  sorry

end NUMINAMATH_CALUDE_invalid_domain_l250_25016


namespace NUMINAMATH_CALUDE_pipe_speed_ratio_l250_25005

-- Define the rates of pipes A, B, and C
def rate_A : ℚ := 1 / 21
def rate_B : ℚ := 2 / 21
def rate_C : ℚ := 4 / 21

-- State the theorem
theorem pipe_speed_ratio :
  -- Conditions
  (rate_A + rate_B + rate_C = 1 / 3) →  -- All pipes fill the tank in 3 hours
  (rate_C = 2 * rate_B) →               -- Pipe C is twice as fast as B
  (rate_A = 1 / 21) →                   -- Pipe A alone takes 21 hours
  -- Conclusion
  (rate_B / rate_A = 2) :=
by sorry

end NUMINAMATH_CALUDE_pipe_speed_ratio_l250_25005


namespace NUMINAMATH_CALUDE_smaller_number_value_l250_25087

theorem smaller_number_value (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 3) : 
  min x y = 28.5 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_value_l250_25087


namespace NUMINAMATH_CALUDE_inequality_solution_set_l250_25024

theorem inequality_solution_set (x : ℝ) : 
  |x - 4| - |x + 1| < 3 ↔ x ∈ Set.Ioo (-1/2 : ℝ) 4 ∪ Set.Ici 4 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l250_25024


namespace NUMINAMATH_CALUDE_simplify_complex_root_expression_l250_25022

theorem simplify_complex_root_expression (a : ℝ) :
  (((a^16)^(1/8))^(1/4) + ((a^16)^(1/4))^(1/8))^2 = 4*a := by sorry

end NUMINAMATH_CALUDE_simplify_complex_root_expression_l250_25022


namespace NUMINAMATH_CALUDE_folded_paper_thickness_l250_25036

/-- The thickness of a folded paper stack -/
def folded_thickness (initial_thickness : ℝ) : ℝ := 2 * initial_thickness

/-- Theorem: Folding a 0.2 cm thick paper stack once results in a 0.4 cm thick stack -/
theorem folded_paper_thickness :
  folded_thickness 0.2 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_thickness_l250_25036


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l250_25002

def C : Set Nat := {62, 64, 65, 69, 71}

theorem smallest_prime_factor_in_C :
  ∃ x ∈ C, ∀ y ∈ C, ∀ p q : Nat,
    Prime p → Prime q → p ∣ x → q ∣ y → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l250_25002


namespace NUMINAMATH_CALUDE_production_equation_holds_l250_25066

/-- Represents the production rate of a factory -/
structure FactoryProduction where
  current_rate : ℝ
  original_rate : ℝ
  h_rate_increase : current_rate = original_rate + 50

/-- The equation representing the production scenario -/
def production_equation (fp : FactoryProduction) : Prop :=
  (450 / fp.original_rate) - (400 / fp.current_rate) = 1

/-- Theorem stating that the production equation holds for the given scenario -/
theorem production_equation_holds (fp : FactoryProduction) :
  production_equation fp := by
  sorry

#check production_equation_holds

end NUMINAMATH_CALUDE_production_equation_holds_l250_25066


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_properties_l250_25088

/-- Represents a hyperbola -/
structure Hyperbola where
  center : ℝ × ℝ
  focal_length : ℝ
  directrix : ℝ

/-- The equation of a hyperbola given its properties -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => 2 * x^2 - 2 * y^2 = 1

/-- Theorem: Given a hyperbola with center at the origin, focal length 2, 
    and one directrix at x = -1/2, its equation is 2x^2 - 2y^2 = 1 -/
theorem hyperbola_equation_from_properties 
  (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_focal_length : h.focal_length = 2)
  (h_directrix : h.directrix = -1/2) :
  ∀ x y, hyperbola_equation h x y ↔ 2 * x^2 - 2 * y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_properties_l250_25088


namespace NUMINAMATH_CALUDE_fraction_count_l250_25097

/-- A fraction is an expression of the form A/B where A and B are polynomials and B contains letters -/
def IsFraction (expr : String) : Prop := sorry

/-- The set of given expressions -/
def ExpressionSet : Set String := {"1/m", "b/3", "(x-1)/π", "2/(x+y)", "a+1/a"}

/-- Counts the number of fractions in a set of expressions -/
def CountFractions (s : Set String) : ℕ := sorry

theorem fraction_count : CountFractions ExpressionSet = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_count_l250_25097


namespace NUMINAMATH_CALUDE_smallest_n_cube_plus_2square_eq_odd_square_l250_25068

theorem smallest_n_cube_plus_2square_eq_odd_square : 
  (∀ n : ℕ, 0 < n → n < 7 → ¬∃ k : ℕ, k % 2 = 1 ∧ n^3 + 2*n^2 = k^2) ∧
  (∃ k : ℕ, k % 2 = 1 ∧ 7^3 + 2*7^2 = k^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_cube_plus_2square_eq_odd_square_l250_25068


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l250_25034

/-- A rectangular solid with prime edge lengths and volume 455 has surface area 382 -/
theorem rectangular_solid_surface_area : ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 455 →
  2 * (a * b + b * c + c * a) = 382 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l250_25034


namespace NUMINAMATH_CALUDE_rectangle_area_l250_25062

theorem rectangle_area (a b : ℕ) : 
  a ≠ b →                  -- rectangle is not a square
  a % 2 = 0 →              -- one side is even
  a * b = 3 * (2 * a + 2 * b) →  -- area is three times perimeter
  a * b = 162              -- area is 162
:= by sorry

end NUMINAMATH_CALUDE_rectangle_area_l250_25062


namespace NUMINAMATH_CALUDE_inequality_proof_l250_25083

theorem inequality_proof (x y z : ℝ) (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) (hz : z ∈ Set.Icc 0 1) :
  2 * (x^3 + y^3 + z^3) - (x^2 * y + y^2 * z + z^2 * x) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l250_25083


namespace NUMINAMATH_CALUDE_division_problem_l250_25095

theorem division_problem (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 15 ∧ divisor = 3 ∧ remainder = 3 →
  dividend = divisor * quotient + remainder →
  quotient = 4 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l250_25095


namespace NUMINAMATH_CALUDE_gcd_10293_29384_l250_25053

theorem gcd_10293_29384 : Nat.gcd 10293 29384 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_10293_29384_l250_25053


namespace NUMINAMATH_CALUDE_correct_initial_driving_time_l250_25060

/-- Represents the driving scenario with given conditions -/
structure DrivingScenario where
  totalDistance : ℝ
  initialSpeed : ℝ
  finalSpeed : ℝ
  lateTime : ℝ
  earlyTime : ℝ

/-- Calculates the time driven at the initial speed -/
def initialDrivingTime (scenario : DrivingScenario) : ℝ :=
  sorry

/-- Theorem stating the correct initial driving time for the given scenario -/
theorem correct_initial_driving_time (scenario : DrivingScenario) 
  (h1 : scenario.totalDistance = 45)
  (h2 : scenario.initialSpeed = 15)
  (h3 : scenario.finalSpeed = 60)
  (h4 : scenario.lateTime = 1)
  (h5 : scenario.earlyTime = 0.5) :
  initialDrivingTime scenario = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_correct_initial_driving_time_l250_25060


namespace NUMINAMATH_CALUDE_samara_tire_expense_l250_25021

/-- Calculates Samara's spending on tires given the other expenses -/
def samaras_tire_spending (alberto_total : ℕ) (samara_oil : ℕ) (samara_detailing : ℕ) (difference : ℕ) : ℕ :=
  alberto_total - (samara_oil + samara_detailing + difference)

theorem samara_tire_expense :
  samaras_tire_spending 2457 25 79 1886 = 467 := by
  sorry

end NUMINAMATH_CALUDE_samara_tire_expense_l250_25021


namespace NUMINAMATH_CALUDE_point_on_right_branch_l250_25018

/-- 
Given a point P(a, b) on the hyperbola x² - 4y² = m (m ≠ 0),
if a - 2b > 0 and a + 2b > 0, then a > 0.
-/
theorem point_on_right_branch 
  (m : ℝ) (hm : m ≠ 0)
  (a b : ℝ) 
  (h_hyperbola : a^2 - 4*b^2 = m)
  (h_diff : a - 2*b > 0)
  (h_sum : a + 2*b > 0) : 
  a > 0 := by sorry

end NUMINAMATH_CALUDE_point_on_right_branch_l250_25018


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l250_25085

theorem square_sum_reciprocal (w : ℝ) (hw : w > 0) (heq : w - 1/w = 5) :
  (w + 1/w)^2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l250_25085


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_vector_l250_25076

/-- A line in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if a point lies on a line given by its parametric equation -/
def lies_on_line (p : ℝ × ℝ) (l : Line2D) : Prop :=
  ∃ t : ℝ, p.1 = l.point.1 + t * l.direction.1 ∧ p.2 = l.point.2 + t * l.direction.2

theorem line_through_point_parallel_to_vector 
  (P : ℝ × ℝ) (d : ℝ × ℝ) :
  let l : Line2D := ⟨P, d⟩
  (∀ x y : ℝ, (x - P.1) / d.1 = (y - P.2) / d.2 ↔ lies_on_line (x, y) l) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_vector_l250_25076


namespace NUMINAMATH_CALUDE_a2_value_l250_25020

theorem a2_value (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^3 + x^10 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a₂ = 42 := by
sorry

end NUMINAMATH_CALUDE_a2_value_l250_25020


namespace NUMINAMATH_CALUDE_bd_length_is_15_l250_25019

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a kite
def is_kite (q : Quadrilateral) : Prop :=
  let AB := dist q.A q.B
  let BC := dist q.B q.C
  let CD := dist q.C q.D
  let DA := dist q.D q.A
  AB = CD ∧ BC = DA

-- Define the specific quadrilateral from the problem
def problem_quadrilateral : Quadrilateral :=
  { A := (0, 0),  -- Arbitrary placement
    B := (7, 0),  -- AB = 7
    C := (7, 19), -- BC = 19
    D := (0, 11)  -- DA = 11
  }

-- Theorem statement
theorem bd_length_is_15 (q : Quadrilateral) :
  is_kite q →
  dist q.A q.B = 7 →
  dist q.B q.C = 19 →
  dist q.C q.D = 7 →
  dist q.D q.A = 11 →
  dist q.B q.D = 15 :=
by sorry

#check bd_length_is_15

end NUMINAMATH_CALUDE_bd_length_is_15_l250_25019


namespace NUMINAMATH_CALUDE_total_notes_count_l250_25048

/-- Proves that given a total amount of 480 rupees in equal numbers of one-rupee, five-rupee, and ten-rupee notes, the total number of notes is 90. -/
theorem total_notes_count (total_amount : ℕ) (note_count : ℕ) : 
  total_amount = 480 →
  note_count * 1 + note_count * 5 + note_count * 10 = total_amount →
  3 * note_count = 90 :=
by
  sorry

#check total_notes_count

end NUMINAMATH_CALUDE_total_notes_count_l250_25048


namespace NUMINAMATH_CALUDE_initial_jasmine_percentage_l250_25093

/-- Proof of initial jasmine percentage in a solution --/
theorem initial_jasmine_percentage
  (initial_volume : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (final_jasmine_percentage : ℝ)
  (h1 : initial_volume = 100)
  (h2 : added_jasmine = 5)
  (h3 : added_water = 10)
  (h4 : final_jasmine_percentage = 8.695652173913043)
  : (100 * (initial_volume * (final_jasmine_percentage / 100) - added_jasmine) / initial_volume) = 5 := by
  sorry

end NUMINAMATH_CALUDE_initial_jasmine_percentage_l250_25093


namespace NUMINAMATH_CALUDE_triple_composition_even_l250_25086

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem triple_composition_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (fun x ↦ g (g (g x))) :=
by sorry

end NUMINAMATH_CALUDE_triple_composition_even_l250_25086


namespace NUMINAMATH_CALUDE_inequality_proof_l250_25043

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * a^2 * b^2 * c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l250_25043


namespace NUMINAMATH_CALUDE_red_peaches_count_l250_25080

/-- Represents a basket of peaches -/
structure Basket :=
  (total : ℕ)
  (green : ℕ)
  (h_green_le_total : green ≤ total)

/-- Calculates the number of red peaches in a basket -/
def red_peaches (b : Basket) : ℕ := b.total - b.green

/-- Theorem: The number of red peaches in a basket with 10 total peaches and 3 green peaches is 7 -/
theorem red_peaches_count (b : Basket) (h_total : b.total = 10) (h_green : b.green = 3) : 
  red_peaches b = 7 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_count_l250_25080


namespace NUMINAMATH_CALUDE_divisor_problem_l250_25007

theorem divisor_problem (n : ℕ) (h1 : n = 1025) (h2 : ¬ (n - 4) % 41 = 0) :
  ∀ d : ℕ, d > 41 → d ∣ n → d ∣ (n - 4) :=
sorry

end NUMINAMATH_CALUDE_divisor_problem_l250_25007


namespace NUMINAMATH_CALUDE_constant_term_expansion_l250_25000

def p (x : ℝ) : ℝ := x^3 + 2*x + 3
def q (x : ℝ) : ℝ := 2*x^4 + x^2 + 7

theorem constant_term_expansion : 
  (p 0) * (q 0) = 21 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l250_25000


namespace NUMINAMATH_CALUDE_shortest_player_height_l250_25029

/-- Given the height of the tallest player and the difference in height between
    the tallest and shortest players, calculate the height of the shortest player. -/
theorem shortest_player_height
  (tallest_height : ℝ)
  (height_difference : ℝ)
  (h1 : tallest_height = 77.75)
  (h2 : height_difference = 9.5) :
  tallest_height - height_difference = 68.25 := by
  sorry

#check shortest_player_height

end NUMINAMATH_CALUDE_shortest_player_height_l250_25029


namespace NUMINAMATH_CALUDE_salon_non_clients_l250_25015

theorem salon_non_clients (manicure_cost : ℝ) (total_earnings : ℝ) (total_fingers : ℕ) (fingers_per_person : ℕ) :
  manicure_cost = 20 →
  total_earnings = 200 →
  total_fingers = 210 →
  fingers_per_person = 10 →
  (total_fingers / fingers_per_person : ℝ) - (total_earnings / manicure_cost) = 11 :=
by sorry

end NUMINAMATH_CALUDE_salon_non_clients_l250_25015


namespace NUMINAMATH_CALUDE_ginos_popsicle_sticks_l250_25063

/-- Gino's popsicle stick problem -/
theorem ginos_popsicle_sticks (initial : Real) (given : Real) (remaining : Real) :
  initial = 63.0 →
  given = 50.0 →
  remaining = initial - given →
  remaining = 13.0 := by sorry

end NUMINAMATH_CALUDE_ginos_popsicle_sticks_l250_25063


namespace NUMINAMATH_CALUDE_trailing_zeros_30_factorial_l250_25099

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 30! is 7 -/
theorem trailing_zeros_30_factorial : trailingZeros 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_30_factorial_l250_25099


namespace NUMINAMATH_CALUDE_hank_aaron_home_runs_l250_25001

/-- The number of home runs hit by Dave Winfield -/
def dave_winfield_hr : ℕ := 465

/-- The number of home runs hit by Hank Aaron -/
def hank_aaron_hr : ℕ := 2 * dave_winfield_hr - 175

/-- Theorem stating that Hank Aaron hit 755 home runs -/
theorem hank_aaron_home_runs : hank_aaron_hr = 755 := by sorry

end NUMINAMATH_CALUDE_hank_aaron_home_runs_l250_25001


namespace NUMINAMATH_CALUDE_wall_width_calculation_l250_25042

/-- The width of a wall given a string length and a relation to that length -/
def wall_width (string_length_m : ℕ) (string_length_cm : ℕ) : ℕ :=
  let string_length_total_cm := string_length_m * 100 + string_length_cm
  5 * string_length_total_cm + 80

theorem wall_width_calculation :
  wall_width 1 70 = 930 := by sorry

end NUMINAMATH_CALUDE_wall_width_calculation_l250_25042


namespace NUMINAMATH_CALUDE_notebook_problem_l250_25089

theorem notebook_problem (x : ℕ) (h : x^2 + 20 = (x + 1)^2 - 9) : x^2 + 20 = 216 := by
  sorry

end NUMINAMATH_CALUDE_notebook_problem_l250_25089


namespace NUMINAMATH_CALUDE_triangle_inequality_l250_25008

theorem triangle_inequality (R r p : ℝ) (hR : R > 0) (hr : r > 0) (hp : p > 0) :
  16 * R * r - 5 * r^2 ≤ p^2 ∧ p^2 ≤ 4 * R^2 + 4 * R * r + 3 * r^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l250_25008


namespace NUMINAMATH_CALUDE_negation_equivalence_l250_25006

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 3*x + 3 < 0) ↔ (∀ x : ℝ, x^2 - 3*x + 3 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l250_25006


namespace NUMINAMATH_CALUDE_negation_of_exists_greater_than_one_l250_25045

theorem negation_of_exists_greater_than_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_greater_than_one_l250_25045


namespace NUMINAMATH_CALUDE_sundae_price_l250_25041

/-- Proves that the price of each sundae is $1.20 given the specified conditions -/
theorem sundae_price (ice_cream_bars sundaes : ℕ) (total_price ice_cream_price : ℚ) : 
  ice_cream_bars = 125 →
  sundaes = 125 →
  total_price = 225 →
  ice_cream_price = 0.60 →
  (total_price - ice_cream_bars * ice_cream_price) / sundaes = 1.20 := by
sorry

end NUMINAMATH_CALUDE_sundae_price_l250_25041


namespace NUMINAMATH_CALUDE_four_tuple_solution_l250_25037

theorem four_tuple_solution (x y z w : ℝ) 
  (h1 : x^2 + y^2 + z^2 + w^2 = 4)
  (h2 : 1/x^2 + 1/y^2 + 1/z^2 + 1/w^2 = 5 - 1/(x*y*z*w)^2) :
  (x = 1 ∨ x = -1) ∧ 
  (y = 1 ∨ y = -1) ∧ 
  (z = 1 ∨ z = -1) ∧ 
  (w = 1 ∨ w = -1) ∧
  (x*y*z*w = 1 ∨ x*y*z*w = -1) :=
by sorry

end NUMINAMATH_CALUDE_four_tuple_solution_l250_25037


namespace NUMINAMATH_CALUDE_min_voters_for_tall_win_l250_25003

/-- Structure representing the giraffe beauty contest voting system -/
structure GiraffeContest where
  total_voters : Nat
  num_districts : Nat
  precincts_per_district : Nat
  voters_per_precinct : Nat

/-- Definition of the specific contest configuration -/
def contest : GiraffeContest :=
  { total_voters := 135
  , num_districts := 5
  , precincts_per_district := 9
  , voters_per_precinct := 3 }

/-- Theorem stating the minimum number of voters needed for Tall to win -/
theorem min_voters_for_tall_win (c : GiraffeContest) 
  (h1 : c.total_voters = c.num_districts * c.precincts_per_district * c.voters_per_precinct)
  (h2 : c = contest) : 
  ∃ (min_voters : Nat), 
    min_voters = 30 ∧ 
    min_voters ≤ c.total_voters ∧
    min_voters = (c.num_districts / 2 + 1) * (c.precincts_per_district / 2 + 1) * (c.voters_per_precinct / 2 + 1) :=
by sorry


end NUMINAMATH_CALUDE_min_voters_for_tall_win_l250_25003


namespace NUMINAMATH_CALUDE_sam_distance_theorem_l250_25049

/-- Calculates the distance traveled given an average speed and time -/
def distanceTraveled (avgSpeed : ℝ) (time : ℝ) : ℝ := avgSpeed * time

theorem sam_distance_theorem (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) 
    (h1 : marguerite_distance = 100)
    (h2 : marguerite_time = 2.4)
    (h3 : sam_time = 3) :
  distanceTraveled (marguerite_distance / marguerite_time) sam_time = 125 := by
  sorry

#check sam_distance_theorem

end NUMINAMATH_CALUDE_sam_distance_theorem_l250_25049


namespace NUMINAMATH_CALUDE_cos_squared_30_minus_2_minus_pi_to_0_l250_25069

theorem cos_squared_30_minus_2_minus_pi_to_0 :
  Real.cos (30 * π / 180) ^ 2 - (2 - π) ^ 0 = -(1/4) := by sorry

end NUMINAMATH_CALUDE_cos_squared_30_minus_2_minus_pi_to_0_l250_25069


namespace NUMINAMATH_CALUDE_shoe_repair_time_l250_25058

theorem shoe_repair_time (heel_time shoe_count total_time : ℕ) (h1 : heel_time = 10) (h2 : shoe_count = 2) (h3 : total_time = 30) :
  (total_time - heel_time * shoe_count) / shoe_count = 5 :=
by sorry

end NUMINAMATH_CALUDE_shoe_repair_time_l250_25058


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l250_25032

/-- The surface area of a cylinder given its unfolded lateral surface dimensions -/
theorem cylinder_surface_area (h w : ℝ) (h_pos : h > 0) (w_pos : w > 0) 
  (h_eq : h = 6 * Real.pi) (w_eq : w = 4 * Real.pi) :
  ∃ (r : ℝ), (r = 3 ∨ r = 2) ∧ 
    (2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 18 * Real.pi ∨
     2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 8 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l250_25032


namespace NUMINAMATH_CALUDE_fuel_tank_capacities_solve_problem_l250_25027

/-- Represents the fuel tank capacities and prices for two cars -/
structure CarFuelData where
  small_capacity : ℝ
  large_capacity : ℝ
  small_fill_cost : ℝ
  large_fill_cost : ℝ
  price_difference : ℝ

/-- The theorem to be proved -/
theorem fuel_tank_capacities (data : CarFuelData) : 
  data.small_capacity = 30 ∧ data.large_capacity = 40 :=
by
  have total_capacity : data.small_capacity + data.large_capacity = 70 := by sorry
  have small_fill_equation : data.small_capacity * (data.large_fill_cost / data.large_capacity - data.price_difference) = data.small_fill_cost := by sorry
  have large_fill_equation : data.large_capacity * (data.large_fill_cost / data.large_capacity) = data.large_fill_cost := by sorry
  have price_relation : data.large_fill_cost / data.large_capacity = data.small_fill_cost / data.small_capacity + data.price_difference := by sorry
  
  sorry -- The proof would go here

/-- The specific instance of CarFuelData for our problem -/
def problem_data : CarFuelData := {
  small_capacity := 30,  -- to be proved
  large_capacity := 40,  -- to be proved
  small_fill_cost := 45,
  large_fill_cost := 68,
  price_difference := 0.29
}

/-- The main theorem applied to our specific problem -/
theorem solve_problem : 
  problem_data.small_capacity = 30 ∧ problem_data.large_capacity = 40 :=
fuel_tank_capacities problem_data

end NUMINAMATH_CALUDE_fuel_tank_capacities_solve_problem_l250_25027


namespace NUMINAMATH_CALUDE_shaded_area_half_circle_l250_25092

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the intersecting point
def intersectionPoint : ℝ × ℝ := sorry

-- Define the four lines
def lines : List (ℝ × ℝ → ℝ × ℝ → Prop) := sorry

-- Define the condition that the point is inside the circle
def pointInsideCircle (c : Circle) (p : ℝ × ℝ) : Prop := sorry

-- Define the condition that the lines form eight 45° angles
def formsEightFortyFiveAngles (p : ℝ × ℝ) (ls : List (ℝ × ℝ → ℝ × ℝ → Prop)) : Prop := sorry

-- Define the shaded sectors
def shadedSectors (c : Circle) (p : ℝ × ℝ) (ls : List (ℝ × ℝ → ℝ × ℝ → Prop)) : Set (ℝ × ℝ) := sorry

-- Define the area of a set in ℝ²
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- The theorem to be proved
theorem shaded_area_half_circle (c : Circle) :
  pointInsideCircle c intersectionPoint →
  formsEightFortyFiveAngles intersectionPoint lines →
  area (shadedSectors c intersectionPoint lines) = π * c.radius^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_half_circle_l250_25092


namespace NUMINAMATH_CALUDE_intersection_empty_union_equals_B_l250_25009

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem for the first part
theorem intersection_empty (a : ℝ) :
  A a ∩ B = ∅ ↔ a ≥ 2 ∨ a ≤ -1/2 :=
sorry

-- Theorem for the second part
theorem union_equals_B (a : ℝ) :
  A a ∪ B = B ↔ a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_intersection_empty_union_equals_B_l250_25009


namespace NUMINAMATH_CALUDE_soldier_difference_l250_25055

/-- Calculates the difference in the number of soldiers between two sides in a war scenario --/
theorem soldier_difference (
  daily_food : ℕ)  -- Daily food requirement per soldier on the first side
  (food_difference : ℕ)  -- Difference in food given to soldiers on the second side
  (first_side_soldiers : ℕ)  -- Number of soldiers on the first side
  (total_food : ℕ)  -- Total amount of food for both sides
  (h1 : daily_food = 10)  -- Each soldier needs 10 pounds of food per day
  (h2 : food_difference = 2)  -- Soldiers on the second side get 2 pounds less food
  (h3 : first_side_soldiers = 4000)  -- The first side has 4000 soldiers
  (h4 : total_food = 68000)  -- The total amount of food for both sides is 68000 pounds
  : (first_side_soldiers - (total_food - first_side_soldiers * daily_food) / (daily_food - food_difference) = 500) :=
by sorry

end NUMINAMATH_CALUDE_soldier_difference_l250_25055


namespace NUMINAMATH_CALUDE_factorization_equality_l250_25028

theorem factorization_equality (c : ℤ) : 
  (∀ x : ℤ, x^2 - x + c = (x + 2) * (x - 3)) → c = -6 := by
sorry

end NUMINAMATH_CALUDE_factorization_equality_l250_25028


namespace NUMINAMATH_CALUDE_number_puzzle_l250_25071

theorem number_puzzle (x : ℤ) : (x + 2)^2 = x^2 - 2016 → x = -505 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l250_25071


namespace NUMINAMATH_CALUDE_valid_seating_arrangements_l250_25098

-- Define the type for people
inductive Person : Type
| Alice : Person
| Bob : Person
| Carla : Person
| Derek : Person
| Eric : Person
| Fiona : Person

-- Define a seating arrangement as a function from position to person
def SeatingArrangement := Fin 6 → Person

-- Define the conditions for a valid seating arrangement
def IsValidArrangement (arrangement : SeatingArrangement) : Prop :=
  -- Alice is not next to Bob or Carla
  (∀ i : Fin 5, arrangement i = Person.Alice → 
    arrangement (i + 1) ≠ Person.Bob ∧ arrangement (i + 1) ≠ Person.Carla) ∧
  (∀ i : Fin 5, arrangement (i + 1) = Person.Alice → 
    arrangement i ≠ Person.Bob ∧ arrangement i ≠ Person.Carla) ∧
  -- Derek is not next to Eric
  (∀ i : Fin 5, arrangement i = Person.Derek → arrangement (i + 1) ≠ Person.Eric) ∧
  (∀ i : Fin 5, arrangement (i + 1) = Person.Derek → arrangement i ≠ Person.Eric) ∧
  -- Fiona is not at either end
  (arrangement 0 ≠ Person.Fiona) ∧ (arrangement 5 ≠ Person.Fiona) ∧
  -- All people are seated and each seat has exactly one person
  (∀ p : Person, ∃! i : Fin 6, arrangement i = p)

-- The theorem to be proved
theorem valid_seating_arrangements :
  (∃ (arrangements : Finset SeatingArrangement), 
    (∀ arr ∈ arrangements, IsValidArrangement arr) ∧ 
    arrangements.card = 16) :=
sorry

end NUMINAMATH_CALUDE_valid_seating_arrangements_l250_25098


namespace NUMINAMATH_CALUDE_solution_set_sqrt3_sin_eq_cos_l250_25072

theorem solution_set_sqrt3_sin_eq_cos :
  {x : ℝ | Real.sqrt 3 * Real.sin x = Real.cos x} =
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.pi / 6} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_sqrt3_sin_eq_cos_l250_25072


namespace NUMINAMATH_CALUDE_triangle_perimeter_l250_25030

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) : 
  c = 2 → b = 2 * a → C = π / 3 → a + b + c = 2 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l250_25030


namespace NUMINAMATH_CALUDE_staircase_perimeter_l250_25070

/-- Represents a staircase-shaped region with specific properties -/
structure StaircaseRegion where
  right_angles : Bool
  congruent_segments : ℕ
  segment_length : ℝ
  area : ℝ
  bottom_width : ℝ

/-- Calculates the perimeter of a staircase-shaped region -/
def perimeter (s : StaircaseRegion) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the specific staircase region -/
theorem staircase_perimeter :
  ∀ (s : StaircaseRegion),
    s.right_angles = true →
    s.congruent_segments = 8 →
    s.segment_length = 1 →
    s.area = 41 →
    s.bottom_width = 7 →
    perimeter s = 128 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l250_25070


namespace NUMINAMATH_CALUDE_debbys_store_inventory_l250_25059

/-- Represents a DVD rental store inventory --/
structure DVDStore where
  initial_count : ℕ
  rental_rate : ℚ
  sold_count : ℕ

/-- Calculates the remaining DVD count after sales --/
def remaining_dvds (store : DVDStore) : ℕ :=
  store.initial_count - store.sold_count

/-- Theorem stating the remaining DVD count for Debby's store --/
theorem debbys_store_inventory :
  let store : DVDStore := {
    initial_count := 150,
    rental_rate := 35 / 100,
    sold_count := 20
  }
  remaining_dvds store = 130 := by
  sorry

end NUMINAMATH_CALUDE_debbys_store_inventory_l250_25059


namespace NUMINAMATH_CALUDE_cooper_fence_bricks_l250_25067

/-- Represents the dimensions of a wall in bricks -/
structure WallDimensions where
  length : Nat
  height : Nat
  depth : Nat

/-- Calculates the number of bricks needed for a wall -/
def bricksForWall (wall : WallDimensions) : Nat :=
  wall.length * wall.height * wall.depth

/-- The dimensions of Cooper's four walls -/
def wall1 : WallDimensions := { length := 15, height := 6, depth := 3 }
def wall2 : WallDimensions := { length := 20, height := 4, depth := 2 }
def wall3 : WallDimensions := { length := 25, height := 5, depth := 3 }
def wall4 : WallDimensions := { length := 17, height := 7, depth := 2 }

/-- Theorem: The total number of bricks needed for Cooper's fence is 1043 -/
theorem cooper_fence_bricks :
  bricksForWall wall1 + bricksForWall wall2 + bricksForWall wall3 + bricksForWall wall4 = 1043 := by
  sorry

end NUMINAMATH_CALUDE_cooper_fence_bricks_l250_25067


namespace NUMINAMATH_CALUDE_negation_of_all_greater_than_sin_l250_25057

theorem negation_of_all_greater_than_sin :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x₀ : ℝ, x₀ ≤ Real.sin x₀) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_all_greater_than_sin_l250_25057


namespace NUMINAMATH_CALUDE_shares_owned_shares_owned_example_l250_25047

/-- Calculates the number of shares owned based on dividend payment and earnings -/
theorem shares_owned (expected_earnings dividend_ratio additional_dividend_rate actual_earnings total_dividend : ℚ) : ℚ :=
  let base_dividend := expected_earnings * dividend_ratio
  let additional_earnings := actual_earnings - expected_earnings
  let additional_dividend := (additional_earnings / 0.1) * additional_dividend_rate
  let total_dividend_per_share := base_dividend + additional_dividend
  total_dividend / total_dividend_per_share

/-- Proves that the number of shares owned is 600 given the specific conditions -/
theorem shares_owned_example : shares_owned 0.8 0.5 0.04 1.1 312 = 600 := by
  sorry

end NUMINAMATH_CALUDE_shares_owned_shares_owned_example_l250_25047


namespace NUMINAMATH_CALUDE_number_of_boys_l250_25073

/-- The number of boys in a school with the given conditions -/
theorem number_of_boys (total : ℕ) (boys : ℕ) : 
  total = 400 → 
  boys + (boys * total) / 100 = total →
  boys = 80 :=
by sorry

end NUMINAMATH_CALUDE_number_of_boys_l250_25073


namespace NUMINAMATH_CALUDE_faster_train_speed_l250_25065

/-- Proves that the speed of the faster train is 50 km/hr given the problem conditions -/
theorem faster_train_speed 
  (speed_diff : ℝ) 
  (faster_train_length : ℝ) 
  (passing_time : ℝ) :
  speed_diff = 32 →
  faster_train_length = 75 →
  passing_time = 15 →
  ∃ (slower_speed faster_speed : ℝ),
    faster_speed - slower_speed = speed_diff ∧
    faster_train_length / passing_time * 3.6 = speed_diff ∧
    faster_speed = 50 := by
  sorry

#check faster_train_speed

end NUMINAMATH_CALUDE_faster_train_speed_l250_25065


namespace NUMINAMATH_CALUDE_babysitting_time_calculation_l250_25064

/-- Calculates the time spent babysitting given the hourly rate and total earnings -/
def time_spent (hourly_rate : ℚ) (total_earnings : ℚ) : ℚ :=
  (total_earnings / hourly_rate) * 60

/-- Proves that given an hourly rate of $12 and total earnings of $10, the time spent babysitting is 50 minutes -/
theorem babysitting_time_calculation (hourly_rate : ℚ) (total_earnings : ℚ) 
  (h1 : hourly_rate = 12)
  (h2 : total_earnings = 10) :
  time_spent hourly_rate total_earnings = 50 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_time_calculation_l250_25064


namespace NUMINAMATH_CALUDE_corrected_mean_l250_25051

theorem corrected_mean (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 →
  initial_mean = 36 →
  incorrect_value = 23 →
  correct_value = 48 →
  let total_sum := n * initial_mean
  let corrected_sum := total_sum - incorrect_value + correct_value
  corrected_sum / n = 36.5 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l250_25051


namespace NUMINAMATH_CALUDE_import_tax_threshold_l250_25025

/-- Proves that the amount in excess of which import tax was applied is $1000 -/
theorem import_tax_threshold (total_value : ℝ) (tax_rate : ℝ) (tax_paid : ℝ) (threshold : ℝ) : 
  total_value = 2570 →
  tax_rate = 0.07 →
  tax_paid = 109.90 →
  tax_rate * (total_value - threshold) = tax_paid →
  threshold = 1000 := by
sorry

end NUMINAMATH_CALUDE_import_tax_threshold_l250_25025


namespace NUMINAMATH_CALUDE_profit_maximized_optimal_selling_price_l250_25026

/-- Profit function given the increase in selling price -/
def profit (x : ℝ) : ℝ := (2 + x) * (200 - 20 * x)

/-- The optimal price increase that maximizes profit -/
def optimal_price_increase : ℝ := 4

/-- The maximum profit achievable -/
def max_profit : ℝ := 720

/-- Theorem stating that the profit function reaches its maximum at the optimal price increase -/
theorem profit_maximized :
  (∀ x : ℝ, profit x ≤ profit optimal_price_increase) ∧
  profit optimal_price_increase = max_profit :=
sorry

/-- The initial selling price -/
def initial_price : ℝ := 10

/-- Theorem stating the optimal selling price -/
theorem optimal_selling_price :
  initial_price + optimal_price_increase = 14 :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_optimal_selling_price_l250_25026


namespace NUMINAMATH_CALUDE_no_equal_consecutive_digit_sums_l250_25074

def sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

def S (n : ℕ) : ℕ :=
  sum_of_digits (2^n)

theorem no_equal_consecutive_digit_sums :
  ∀ n : ℕ, n > 0 → S (n + 1) ≠ S n :=
sorry

end NUMINAMATH_CALUDE_no_equal_consecutive_digit_sums_l250_25074


namespace NUMINAMATH_CALUDE_stock_investment_net_increase_l250_25046

theorem stock_investment_net_increase (x : ℝ) (x_pos : x > 0) :
  x * 1.5 * 0.7 = 1.05 * x := by
  sorry

end NUMINAMATH_CALUDE_stock_investment_net_increase_l250_25046


namespace NUMINAMATH_CALUDE_x_plus_y_values_l250_25013

theorem x_plus_y_values (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 2) (h3 : x > y) :
  x + y = 5 ∨ x + y = 1 :=
by sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l250_25013


namespace NUMINAMATH_CALUDE_line_points_count_l250_25023

theorem line_points_count (n : ℕ) 
  (point1 : ∃ (a b : ℕ), a * b = 80 ∧ a + b + 1 = n)
  (point2 : ∃ (c d : ℕ), c * d = 90 ∧ c + d + 1 = n) :
  n = 22 := by
sorry

end NUMINAMATH_CALUDE_line_points_count_l250_25023


namespace NUMINAMATH_CALUDE_pentagon_triangle_side_ratio_l250_25075

theorem pentagon_triangle_side_ratio :
  ∀ (p t s : ℝ),
  p > 0 ∧ t > 0 ∧ s > 0 →
  5 * p = 3 * t →
  5 * p = 4 * s →
  p / t = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_pentagon_triangle_side_ratio_l250_25075


namespace NUMINAMATH_CALUDE_conference_handshakes_l250_25061

/-- The number of handshakes in a conference where each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a conference of 35 people where each person shakes hands with every other person exactly once, the total number of handshakes is 595. -/
theorem conference_handshakes :
  handshakes 35 = 595 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l250_25061


namespace NUMINAMATH_CALUDE_no_rain_probability_l250_25091

theorem no_rain_probability (rain_prob : ℚ) (days : ℕ) : 
  rain_prob = 2/3 → days = 5 → (1 - rain_prob)^days = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l250_25091


namespace NUMINAMATH_CALUDE_sherry_age_l250_25096

theorem sherry_age (randolph_age sydney_age sherry_age : ℕ) : 
  randolph_age = 55 →
  randolph_age = sydney_age + 5 →
  sydney_age = 2 * sherry_age →
  sherry_age = 25 := by sorry

end NUMINAMATH_CALUDE_sherry_age_l250_25096


namespace NUMINAMATH_CALUDE_function_composition_problem_l250_25044

theorem function_composition_problem (a b : ℝ) : 
  (∀ x, (3 * ((a * x) + b) - 4) = 4 * x + 5) → 
  a + b = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_problem_l250_25044


namespace NUMINAMATH_CALUDE_constant_ratio_locus_l250_25056

/-- The locus of points with a constant ratio of distances -/
theorem constant_ratio_locus (x y : ℝ) :
  (((x - 4)^2 + y^2) / (x - 3)^2 = 4) →
  (3 * x^2 - y^2 - 16 * x + 20 = 0) :=
by sorry

end NUMINAMATH_CALUDE_constant_ratio_locus_l250_25056


namespace NUMINAMATH_CALUDE_miran_has_least_paper_l250_25079

def miran_paper : ℕ := 6
def junga_paper : ℕ := 13
def minsu_paper : ℕ := 10

theorem miran_has_least_paper : 
  miran_paper ≤ junga_paper ∧ miran_paper ≤ minsu_paper :=
sorry

end NUMINAMATH_CALUDE_miran_has_least_paper_l250_25079


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l250_25010

theorem simplify_and_evaluate_expression :
  let x : ℝ := Real.sqrt 3 + 1
  let y : ℝ := Real.sqrt 3
  ((3 * x + y) / (x^2 - y^2) + (2 * x) / (y^2 - x^2)) / (2 / (x^2 * y - x * y^2)) = (3 + Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l250_25010


namespace NUMINAMATH_CALUDE_cylinder_height_l250_25084

/-- The height of a right cylinder with radius 3 feet and surface area 36π square feet is 3 feet. -/
theorem cylinder_height (π : ℝ) (h : ℝ) : 
  2 * π * 3^2 + 2 * π * 3 * h = 36 * π → h = 3 := by sorry

end NUMINAMATH_CALUDE_cylinder_height_l250_25084


namespace NUMINAMATH_CALUDE_cookie_problem_l250_25038

theorem cookie_problem : ∃! C : ℕ, 0 < C ∧ C < 80 ∧ C % 6 = 5 ∧ C % 9 = 7 ∧ C = 29 := by
  sorry

end NUMINAMATH_CALUDE_cookie_problem_l250_25038


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l250_25054

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) :
  (a / (1 - r) = 64 * (a * r^4) / (1 - r)) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l250_25054


namespace NUMINAMATH_CALUDE_division_problem_l250_25004

theorem division_problem (n : ℕ) : 
  n % 7 = 5 ∧ n / 7 = 12 → n / 8 = 11 := by sorry

end NUMINAMATH_CALUDE_division_problem_l250_25004


namespace NUMINAMATH_CALUDE_min_profit_is_128_l250_25017

/-- The profit function for a stationery item -/
def profit (x : ℝ) : ℝ :=
  let y := -2 * x + 60
  y * (x - 10)

/-- The theorem stating the minimum profit -/
theorem min_profit_is_128 :
  ∃ (x_min : ℝ), 15 ≤ x_min ∧ x_min ≤ 26 ∧
  ∀ (x : ℝ), 15 ≤ x → x ≤ 26 → profit x_min ≤ profit x ∧
  profit x_min = 128 :=
sorry

end NUMINAMATH_CALUDE_min_profit_is_128_l250_25017
