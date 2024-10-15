import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_three_squares_l1764_176411

theorem sum_of_three_squares (a k : ℕ) :
  ¬ ∃ x y z : ℤ, (4^a * (8*k + 7) : ℤ) = x^2 + y^2 + z^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l1764_176411


namespace NUMINAMATH_CALUDE_hypotenuse_length_of_isosceles_right_triangle_l1764_176443

def isosceles_right_triangle (a c : ℝ) : Prop :=
  a > 0 ∧ c > 0 ∧ c^2 = 2 * a^2

theorem hypotenuse_length_of_isosceles_right_triangle (a c : ℝ) :
  isosceles_right_triangle a c →
  2 * a + c = 8 + 8 * Real.sqrt 2 →
  c = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_length_of_isosceles_right_triangle_l1764_176443


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a8_l1764_176419

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a8 (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 2 = 2 → a 14 = 18 → a 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a8_l1764_176419


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l1764_176413

/-- Proves that the length of a rectangular plot is 63 meters given the specified conditions -/
theorem rectangular_plot_length : 
  ∀ (breadth length : ℝ),
  length = breadth + 26 →
  2 * (length + breadth) * 26.5 = 5300 →
  length = 63 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l1764_176413


namespace NUMINAMATH_CALUDE_min_sum_and_inequality_l1764_176436

-- Define the function f
def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

-- State the theorem
theorem min_sum_and_inequality (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∀ x, f x a b ≥ 4) : 
  a + b ≥ 4 ∧ (a + b = 4 → 1/a + 4/b ≥ 9/4) := by
  sorry


end NUMINAMATH_CALUDE_min_sum_and_inequality_l1764_176436


namespace NUMINAMATH_CALUDE_triangle_trig_identities_l1764_176441

/-- Given an acute triangle ABC with area 3√3, side lengths AB = 3 and AC = 4, 
    prove the following trigonometric identities involving its angles. -/
theorem triangle_trig_identities 
  (A B C : Real) 
  (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
  (h_area : (1/2) * 3 * 4 * Real.sin A = 3 * Real.sqrt 3)
  (h_AB : 3 = 3)
  (h_AC : 4 = 4) :
  Real.sin (π/2 + A) = 1/2 ∧ 
  Real.cos (A - B) = (7 * Real.sqrt 13) / 26 := by
sorry

end NUMINAMATH_CALUDE_triangle_trig_identities_l1764_176441


namespace NUMINAMATH_CALUDE_domain_of_composed_function_inequality_proof_l1764_176444

-- Definition of the function f
def f : Set ℝ := Set.Icc (1/2) 2

-- Theorem 1: Domain of y = f(2^x)
theorem domain_of_composed_function :
  {x : ℝ | 2^x ∈ f} = Set.Icc (-1) 1 := by sorry

-- Theorem 2: Inequality proof
theorem inequality_proof (x y : ℝ) (h1 : -2 < x) (h2 : x < y) (h3 : y < 1) :
  -3 < x - y ∧ x - y < 0 := by sorry

end NUMINAMATH_CALUDE_domain_of_composed_function_inequality_proof_l1764_176444


namespace NUMINAMATH_CALUDE_outbound_time_calculation_l1764_176498

/-- The time taken for John to drive to the distant city -/
def outbound_time : ℝ := 30

/-- The time taken for John to return from the distant city -/
def return_time : ℝ := 5

/-- The speed increase on the return trip -/
def speed_increase : ℝ := 12

/-- The speed on the outbound trip -/
def outbound_speed : ℝ := 60

/-- The speed on the return trip -/
def return_speed : ℝ := outbound_speed + speed_increase

theorem outbound_time_calculation :
  outbound_time * outbound_speed = return_time * return_speed := by sorry

#check outbound_time_calculation

end NUMINAMATH_CALUDE_outbound_time_calculation_l1764_176498


namespace NUMINAMATH_CALUDE_initial_speed_is_40_l1764_176465

/-- Represents a journey with increasing speed -/
structure Journey where
  totalDistance : ℝ
  totalTime : ℝ
  speedIncrease : ℝ
  intervalTime : ℝ

/-- Calculates the initial speed for a given journey -/
def calculateInitialSpeed (j : Journey) : ℝ :=
  sorry

/-- Theorem stating that for the given journey parameters, the initial speed is 40 km/h -/
theorem initial_speed_is_40 :
  let j : Journey := {
    totalDistance := 56,
    totalTime := 48 / 60, -- converting minutes to hours
    speedIncrease := 20,
    intervalTime := 12 / 60 -- converting minutes to hours
  }
  calculateInitialSpeed j = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_is_40_l1764_176465


namespace NUMINAMATH_CALUDE_hexagon_side_sum_l1764_176451

/-- Given a hexagon PQRSTU with the following properties:
  * The area of PQRSTU is 68
  * PQ = 10
  * QR = 7
  * TU = 6
  Prove that RS + ST = 3 -/
theorem hexagon_side_sum (PQRSTU : Set ℝ × ℝ) (area : ℝ) (PQ QR TU : ℝ) :
  area = 68 → PQ = 10 → QR = 7 → TU = 6 →
  ∃ (RS ST : ℝ), RS + ST = 3 := by
  sorry

#check hexagon_side_sum

end NUMINAMATH_CALUDE_hexagon_side_sum_l1764_176451


namespace NUMINAMATH_CALUDE_total_holiday_savings_l1764_176403

/-- The total money saved for holiday spending by Victory and Sam -/
theorem total_holiday_savings (sam_savings : ℕ) (victory_savings : ℕ) : 
  sam_savings = 1200 → 
  victory_savings = sam_savings - 200 →
  sam_savings + victory_savings = 2200 := by
sorry

end NUMINAMATH_CALUDE_total_holiday_savings_l1764_176403


namespace NUMINAMATH_CALUDE_cylinder_volume_l1764_176487

/-- Given a cylinder with lateral surface area 100π cm² and an inscribed rectangular solid
    with diagonal 10√2 cm, prove that the volume of the cylinder is 250π cm³. -/
theorem cylinder_volume (r h : ℝ) : 
  r > 0 → h > 0 →
  2 * Real.pi * r * h = 100 * Real.pi →
  4 * r^2 + h^2 = 200 →
  Real.pi * r^2 * h = 250 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l1764_176487


namespace NUMINAMATH_CALUDE_area_of_PQRS_l1764_176412

/-- Reflect a point (x, y) in the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflect a point (x, y) in the line y=x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

/-- Reflect a point (x, y) in the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Calculate the area of a quadrilateral given its four vertices -/
def quadrilateral_area (a b c d : ℝ × ℝ) : ℝ := sorry

theorem area_of_PQRS : 
  let P : ℝ × ℝ := (-1, 4)
  let Q := reflect_y P
  let R := reflect_y_eq_x Q
  let S := reflect_x R
  quadrilateral_area P Q R S = 8 := by sorry

end NUMINAMATH_CALUDE_area_of_PQRS_l1764_176412


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l1764_176435

/-- Given a circle with center (2,3) and one endpoint of a diameter at (-1,-1),
    the other endpoint of the diameter is at (5,7). -/
theorem circle_diameter_endpoint (O : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) : 
  O = (2, 3) → A = (-1, -1) → 
  (O.1 - A.1 = B.1 - O.1 ∧ O.2 - A.2 = B.2 - O.2) → 
  B = (5, 7) := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l1764_176435


namespace NUMINAMATH_CALUDE_curve_arc_length_l1764_176445

noncomputable def arcLength (t₁ t₂ : Real) : Real :=
  ∫ t in t₁..t₂, Real.sqrt ((12 * Real.cos t ^ 2 * Real.sin t) ^ 2 + (12 * Real.sin t ^ 2 * Real.cos t) ^ 2)

theorem curve_arc_length :
  arcLength (π / 6) (π / 4) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_curve_arc_length_l1764_176445


namespace NUMINAMATH_CALUDE_percentage_failed_hindi_l1764_176483

theorem percentage_failed_hindi (failed_english : Real) (failed_both : Real) (passed_both : Real) :
  failed_english = 35 →
  failed_both = 40 →
  passed_both = 80 →
  ∃ (failed_hindi : Real), failed_hindi = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_failed_hindi_l1764_176483


namespace NUMINAMATH_CALUDE_calculation_proof_l1764_176447

theorem calculation_proof :
  (1/2 + (-2/3) - 4/7 + (-1/2) - 1/3 = -11/7) ∧
  (-7^2 + 2*(-3)^2 - (-6)/((-1/3)^2) = 23) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1764_176447


namespace NUMINAMATH_CALUDE_inclination_angle_tangent_l1764_176424

theorem inclination_angle_tangent (α : ℝ) : 
  (∃ (x y : ℝ), 2 * x + y + 1 = 0 ∧ α = Real.arctan (-2)) → 
  Real.tan (α - π / 4) = 3 := by
sorry

end NUMINAMATH_CALUDE_inclination_angle_tangent_l1764_176424


namespace NUMINAMATH_CALUDE_carpet_area_calculation_l1764_176482

theorem carpet_area_calculation (rectangle_length rectangle_width triangle_base triangle_height : ℝ) 
  (h1 : rectangle_length = 12)
  (h2 : rectangle_width = 8)
  (h3 : triangle_base = 10)
  (h4 : triangle_height = 6) : 
  rectangle_length * rectangle_width + (triangle_base * triangle_height) / 2 = 126 := by
  sorry

end NUMINAMATH_CALUDE_carpet_area_calculation_l1764_176482


namespace NUMINAMATH_CALUDE_no_solution_exists_l1764_176408

theorem no_solution_exists : ¬∃ x : ℝ, x > 0 ∧ x * Real.sqrt (9 - x) + Real.sqrt (9 * x - x^3) ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1764_176408


namespace NUMINAMATH_CALUDE_problem_statement_l1764_176418

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x - a

-- Define the domain
def domain : Set ℝ := Set.Icc (-1) 1

theorem problem_statement (a : ℝ) :
  -- Part 1
  (f a 0 = f a 1) →
  (Set.Icc (-1) (1/2) = {x ∈ domain | |f a x - 1| < a * x + 3/4}) ∧
  -- Part 2
  (|a| ≤ 1) →
  (∀ x ∈ domain, |f a x| ≤ 5/4) :=
by sorry


end NUMINAMATH_CALUDE_problem_statement_l1764_176418


namespace NUMINAMATH_CALUDE_restaurant_group_l1764_176457

/-- Proves the number of kids in a group given the total number of people, 
    adult meal cost, and total cost. -/
theorem restaurant_group (total_people : ℕ) (adult_meal_cost : ℕ) (total_cost : ℕ) 
  (h1 : total_people = 9)
  (h2 : adult_meal_cost = 2)
  (h3 : total_cost = 14) :
  ∃ (num_kids : ℕ), 
    num_kids = total_people - (total_cost / adult_meal_cost) ∧ 
    num_kids = 2 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_group_l1764_176457


namespace NUMINAMATH_CALUDE_fifteenth_digit_sum_one_ninth_one_eleventh_l1764_176488

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

/-- The sum of decimal representations of two rational numbers -/
def sumDecimalRepresentations (q₁ q₂ : ℚ) : ℕ → ℕ := sorry

/-- The nth digit after the decimal point in a decimal representation -/
def nthDigitAfterDecimal (rep : ℕ → ℕ) (n : ℕ) : ℕ := sorry

theorem fifteenth_digit_sum_one_ninth_one_eleventh :
  nthDigitAfterDecimal (sumDecimalRepresentations (1/9) (1/11)) 15 = 1 := by sorry

end NUMINAMATH_CALUDE_fifteenth_digit_sum_one_ninth_one_eleventh_l1764_176488


namespace NUMINAMATH_CALUDE_initial_book_donations_l1764_176460

/-- Proves that the initial number of book donations is 300 given the conditions of the problem. -/
theorem initial_book_donations (
  people_donating : ℕ)
  (books_per_person : ℕ)
  (books_borrowed : ℕ)
  (remaining_books : ℕ)
  (h1 : people_donating = 10)
  (h2 : books_per_person = 5)
  (h3 : books_borrowed = 140)
  (h4 : remaining_books = 210) :
  people_donating * books_per_person + remaining_books + books_borrowed = 300 :=
by sorry


end NUMINAMATH_CALUDE_initial_book_donations_l1764_176460


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1764_176437

theorem cos_alpha_value (α : Real) (h : Real.sin (α / 2) = Real.sqrt 3 / 3) : 
  Real.cos α = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1764_176437


namespace NUMINAMATH_CALUDE_sector_area_rate_of_change_l1764_176469

/-- The rate of change of a circular sector's area --/
theorem sector_area_rate_of_change
  (r : ℝ)
  (θ : ℝ → ℝ)
  (h_r : r = 12)
  (h_θ : ∀ t, θ t = 38 + 5 * t) :
  ∀ t, (deriv (λ t => (1/2) * r^2 * (θ t * π / 180))) t = 2 * π :=
sorry

end NUMINAMATH_CALUDE_sector_area_rate_of_change_l1764_176469


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_comparison_l1764_176450

theorem geometric_arithmetic_sequence_comparison 
  (a b : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, ∃ q : ℝ, a (n + 1) = a n * q) 
  (h_arith : ∀ n : ℕ, ∃ d : ℝ, b (n + 1) = b n + d)
  (h_pos : a 1 > 0)
  (h_eq1 : a 1 = b 1)
  (h_eq3 : a 3 = b 3)
  (h_neq : a 1 ≠ a 3) :
  a 5 > b 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_comparison_l1764_176450


namespace NUMINAMATH_CALUDE_infinitely_many_nth_powers_l1764_176471

/-- An infinite arithmetic progression of positive integers -/
structure ArithmeticProgression :=
  (a : ℕ)  -- First term
  (d : ℕ)  -- Common difference

/-- Checks if a number is in the arithmetic progression -/
def ArithmeticProgression.contains (ap : ArithmeticProgression) (x : ℕ) : Prop :=
  ∃ k : ℕ, x = ap.a + k * ap.d

/-- Checks if a number is an nth power -/
def is_nth_power (x n : ℕ) : Prop :=
  ∃ m : ℕ, x = m^n

theorem infinitely_many_nth_powers
  (ap : ArithmeticProgression)
  (n : ℕ)
  (h : ∃ x : ℕ, ap.contains x ∧ is_nth_power x n) :
  ∀ N : ℕ, ∃ M : ℕ, M > N ∧ ap.contains M ∧ is_nth_power M n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_nth_powers_l1764_176471


namespace NUMINAMATH_CALUDE_project_hours_total_l1764_176499

/-- Represents the hours charged by Kate, Pat, and Mark to a project -/
structure ProjectHours where
  kate : ℕ
  pat : ℕ
  mark : ℕ

/-- Defines the conditions of the project hours -/
def validProjectHours (h : ProjectHours) : Prop :=
  h.pat = 2 * h.kate ∧
  h.pat = h.mark / 3 ∧
  h.mark = h.kate + 110

theorem project_hours_total (h : ProjectHours) (hValid : validProjectHours h) :
  h.kate + h.pat + h.mark = 198 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_total_l1764_176499


namespace NUMINAMATH_CALUDE_circle_equation_and_line_slope_l1764_176462

/-- A circle passing through three points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the form mx + y - 1 = 0 -/
structure Line where
  m : ℝ

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Check if a point lies on a circle -/
def onCircle (c : Circle) (p : ℝ × ℝ) : Prop := sorry

/-- Check if a point lies on a line -/
def onLine (l : Line) (p : ℝ × ℝ) : Prop := sorry

/-- The intersection points of a circle and a line -/
def intersectionPoints (c : Circle) (l : Line) : Set (ℝ × ℝ) := sorry

theorem circle_equation_and_line_slope 
  (c : Circle) 
  (l : Line) 
  (h1 : onCircle c (0, -4))
  (h2 : onCircle c (2, 0))
  (h3 : onCircle c (3, -1))
  (h4 : ∃ (A B : ℝ × ℝ), A ∈ intersectionPoints c l ∧ B ∈ intersectionPoints c l ∧ distance A B = 4) :
  c.center = (1, -2) ∧ c.radius^2 = 5 ∧ l.m = 4/3 := by sorry

end NUMINAMATH_CALUDE_circle_equation_and_line_slope_l1764_176462


namespace NUMINAMATH_CALUDE_g_max_value_l1764_176455

/-- The function g(x) = 4x - x^4 -/
def g (x : ℝ) := 4 * x - x^4

/-- The maximum value of g(x) on the interval [0, 2] is 3 -/
theorem g_max_value : ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ ∀ x ∈ Set.Icc 0 2, g x ≤ g c ∧ g c = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_max_value_l1764_176455


namespace NUMINAMATH_CALUDE_next_feb29_sunday_l1764_176401

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Checks if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  year % 4 == 0 && (year % 100 ≠ 0 || year % 400 == 0)

/-- Advances the day of the week by the given number of days -/
def advanceDayOfWeek (day : DayOfWeek) (days : Nat) : DayOfWeek :=
  match (day, days % 7) with
  | (DayOfWeek.Sunday, 0) => DayOfWeek.Sunday
  | (DayOfWeek.Sunday, 1) => DayOfWeek.Monday
  | (DayOfWeek.Sunday, 2) => DayOfWeek.Tuesday
  | (DayOfWeek.Sunday, 3) => DayOfWeek.Wednesday
  | (DayOfWeek.Sunday, 4) => DayOfWeek.Thursday
  | (DayOfWeek.Sunday, 5) => DayOfWeek.Friday
  | (DayOfWeek.Sunday, 6) => DayOfWeek.Saturday
  | _ => DayOfWeek.Sunday  -- Default case, should not occur

/-- Calculates the day of the week for February 29 in the given year, starting from 2004 -/
def feb29DayOfWeek (year : Nat) : DayOfWeek :=
  let daysAdvanced := (year - 2004) / 4 * 2  -- Each leap year advances by 2 days
  advanceDayOfWeek DayOfWeek.Sunday daysAdvanced

/-- Theorem: The next year after 2004 when February 29 falls on a Sunday is 2032 -/
theorem next_feb29_sunday : 
  (∀ y : Nat, 2004 < y → y < 2032 → feb29DayOfWeek y ≠ DayOfWeek.Sunday) ∧ 
  feb29DayOfWeek 2032 = DayOfWeek.Sunday :=
sorry

end NUMINAMATH_CALUDE_next_feb29_sunday_l1764_176401


namespace NUMINAMATH_CALUDE_not_always_parallel_lines_l1764_176480

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)

-- State the theorem
theorem not_always_parallel_lines 
  (α β : Plane) (m n : Line) 
  (h1 : parallel_line_plane m α) 
  (h2 : parallel_plane_plane α β) 
  (h3 : line_in_plane n β) : 
  ¬(∀ m n, parallel_line_line m n) := by
sorry


end NUMINAMATH_CALUDE_not_always_parallel_lines_l1764_176480


namespace NUMINAMATH_CALUDE_two_machines_total_copies_l1764_176473

/-- Represents a copy machine with a constant copying rate. -/
structure CopyMachine where
  rate : ℕ  -- Copies per minute

/-- Calculates the total number of copies made by a machine in a given time. -/
def copies_made (machine : CopyMachine) (minutes : ℕ) : ℕ :=
  machine.rate * minutes

/-- Represents the problem setup with two copy machines. -/
structure TwoMachineProblem where
  machine1 : CopyMachine
  machine2 : CopyMachine
  duration : ℕ  -- Duration in minutes

/-- The main theorem stating the total number of copies made by both machines. -/
theorem two_machines_total_copies (problem : TwoMachineProblem)
    (h1 : problem.machine1.rate = 40)
    (h2 : problem.machine2.rate = 55)
    (h3 : problem.duration = 30) :
    copies_made problem.machine1 problem.duration +
    copies_made problem.machine2 problem.duration = 2850 := by
  sorry

end NUMINAMATH_CALUDE_two_machines_total_copies_l1764_176473


namespace NUMINAMATH_CALUDE_emily_trivia_score_l1764_176459

/-- Emily's trivia game score calculation -/
theorem emily_trivia_score (first_round : ℤ) : 
  first_round + 33 - 48 = 1 → first_round = 16 := by
  sorry

end NUMINAMATH_CALUDE_emily_trivia_score_l1764_176459


namespace NUMINAMATH_CALUDE_simplify_expression_l1764_176434

theorem simplify_expression : ((- Real.sqrt 3) ^ 2) ^ (-1/2 : ℝ) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1764_176434


namespace NUMINAMATH_CALUDE_bank_investment_problem_l1764_176404

theorem bank_investment_problem (total_investment interest_rate1 interest_rate2 total_interest : ℝ)
  (h1 : total_investment = 5000)
  (h2 : interest_rate1 = 0.04)
  (h3 : interest_rate2 = 0.065)
  (h4 : total_interest = 282.5)
  (h5 : ∃ x y : ℝ, x + y = total_investment ∧ interest_rate1 * x + interest_rate2 * y = total_interest) :
  ∃ x : ℝ, x = 1700 ∧ interest_rate1 * x + interest_rate2 * (total_investment - x) = total_interest :=
by
  sorry

end NUMINAMATH_CALUDE_bank_investment_problem_l1764_176404


namespace NUMINAMATH_CALUDE_stating_max_principals_l1764_176466

/-- Represents the duration of the period in years -/
def period_duration : ℕ := 10

/-- Represents the duration of a principal's term in years -/
def term_duration : ℕ := 4

/-- 
Theorem stating that the maximum number of principals 
that can serve during the given period is 3
-/
theorem max_principals :
  ∀ (principal_count : ℕ),
  (∀ (year : ℕ), year ≤ period_duration → 
    ∃ (principal : ℕ), principal ≤ principal_count ∧ 
    ∃ (start_year : ℕ), start_year ≤ period_duration ∧ 
    year ∈ Set.Icc start_year (start_year + term_duration - 1)) →
  principal_count ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_stating_max_principals_l1764_176466


namespace NUMINAMATH_CALUDE_special_collection_returned_percentage_l1764_176442

/-- Calculates the percentage of returned books given initial count, final count, and loaned count. -/
def percentage_returned (initial : ℕ) (final : ℕ) (loaned : ℕ) : ℚ :=
  (1 - (initial - final : ℚ) / (loaned : ℚ)) * 100

/-- Theorem stating that the percentage of returned books is 65% given the problem conditions. -/
theorem special_collection_returned_percentage :
  percentage_returned 75 61 40 = 65 := by
  sorry

end NUMINAMATH_CALUDE_special_collection_returned_percentage_l1764_176442


namespace NUMINAMATH_CALUDE_overtime_rate_is_90_cents_l1764_176491

/-- Represents the worker's pay structure and work week --/
structure WorkerPay where
  ordinary_rate : ℚ
  total_hours : ℕ
  overtime_hours : ℕ
  total_pay : ℚ

/-- Calculates the overtime rate given the worker's pay structure --/
def overtime_rate (w : WorkerPay) : ℚ :=
  let ordinary_hours := w.total_hours - w.overtime_hours
  let ordinary_pay := (w.ordinary_rate * ordinary_hours : ℚ)
  let overtime_pay := w.total_pay - ordinary_pay
  overtime_pay / w.overtime_hours

/-- Theorem stating that the overtime rate is $0.90 per hour --/
theorem overtime_rate_is_90_cents (w : WorkerPay) 
  (h1 : w.ordinary_rate = 60 / 100)
  (h2 : w.total_hours = 50)
  (h3 : w.overtime_hours = 8)
  (h4 : w.total_pay = 3240 / 100) : 
  overtime_rate w = 90 / 100 := by
  sorry

#eval overtime_rate { 
  ordinary_rate := 60 / 100, 
  total_hours := 50, 
  overtime_hours := 8, 
  total_pay := 3240 / 100 
}

end NUMINAMATH_CALUDE_overtime_rate_is_90_cents_l1764_176491


namespace NUMINAMATH_CALUDE_spinster_cat_problem_l1764_176474

theorem spinster_cat_problem (S C : ℕ) 
  (h1 : S * 9 = C * 2)  -- Ratio of spinsters to cats is 2:9
  (h2 : C = S + 63)     -- There are 63 more cats than spinsters
  : S = 18 := by        -- Prove that the number of spinsters is 18
sorry

end NUMINAMATH_CALUDE_spinster_cat_problem_l1764_176474


namespace NUMINAMATH_CALUDE_initial_books_correct_l1764_176405

/-- Calculates the initial number of books in Mary's mystery book library --/
def initial_books : ℕ :=
  let books_received := 12 -- 1 book per month for 12 months
  let books_bought := 5 + 2 -- 5 from bookstore, 2 from yard sales
  let books_gifted := 1 + 4 -- 1 from daughter, 4 from mother
  let books_removed := 12 + 3 -- 12 donated, 3 sold
  let final_books := 81

  final_books - (books_received + books_bought + books_gifted) + books_removed

theorem initial_books_correct :
  initial_books = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_books_correct_l1764_176405


namespace NUMINAMATH_CALUDE_a_55_divisible_by_55_l1764_176453

/-- Concatenation of integers from 1 to n -/
def a (n : ℕ) : ℕ :=
  -- Definition of a_n goes here
  sorry

/-- Theorem: a_55 is divisible by 55 -/
theorem a_55_divisible_by_55 : 55 ∣ a 55 := by
  sorry

end NUMINAMATH_CALUDE_a_55_divisible_by_55_l1764_176453


namespace NUMINAMATH_CALUDE_parabola_properties_l1764_176476

/-- Represents a parabola in a 2D Cartesian coordinate system -/
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * a * x

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in a 2D Cartesian coordinate system -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  equation : ℝ → ℝ → Prop := fun x y => a * x + b * y + c = 0

/-- Given a parabola C with vertex at (0, 0) and focus at (1, 0), 
    prove that its standard equation is y^2 = 4x and that for any two points 
    M and N on its directrix with y-coordinates y₁ and y₂ such that y₁y₂ = -4, 
    the line passing through the intersections of OM and ON with C 
    always contains the point (1, 0) -/
theorem parabola_properties (C : Parabola) 
  (h_vertex : C.equation 0 0)
  (h_focus : C.equation 1 0) :
  (C.equation = fun x y => y^2 = 4 * x) ∧
  (∀ y₁ y₂ : ℝ, y₁ * y₂ = -4 →
    ∃ (L : Line), 
      (∀ x y, L.equation x y → C.equation x y) ∧
      L.equation 1 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l1764_176476


namespace NUMINAMATH_CALUDE_smallest_n_is_smallest_l1764_176439

/-- The smallest positive integer satisfying the given conditions -/
def smallest_n : ℕ := 46656

/-- n is divisible by 36 -/
axiom divisible_by_36 : smallest_n % 36 = 0

/-- n^2 is a perfect cube -/
axiom perfect_cube : ∃ k : ℕ, smallest_n^2 = k^3

/-- n^3 is a perfect square -/
axiom perfect_square : ∃ k : ℕ, smallest_n^3 = k^2

/-- Theorem stating that smallest_n is indeed the smallest positive integer satisfying all conditions -/
theorem smallest_n_is_smallest : 
  ∀ m : ℕ, m > 0 ∧ m % 36 = 0 ∧ (∃ k : ℕ, m^2 = k^3) ∧ (∃ k : ℕ, m^3 = k^2) → m ≥ smallest_n :=
sorry

end NUMINAMATH_CALUDE_smallest_n_is_smallest_l1764_176439


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l1764_176477

theorem final_sum_after_operations (S a b : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 7) = 3 * S + 36 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l1764_176477


namespace NUMINAMATH_CALUDE_infinite_solutions_for_diophantine_equation_l1764_176475

theorem infinite_solutions_for_diophantine_equation :
  ∃ (S : Set Nat), Set.Infinite S ∧ 
  (∀ p ∈ S, Prime p ∧ 
  ∃ x y : ℤ, x^2 + x + 1 = p * y) := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_for_diophantine_equation_l1764_176475


namespace NUMINAMATH_CALUDE_four_digit_divisor_cyclic_iff_abab_l1764_176431

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def cyclic_shift (n : ℕ) : ℕ := 
  let d := n % 10
  let r := n / 10
  d * 1000 + r

def is_divisor_of_cyclic (n : ℕ) : Prop :=
  ∃ k, k * n = cyclic_shift n ∨ k * n = cyclic_shift (cyclic_shift n) ∨ k * n = cyclic_shift (cyclic_shift (cyclic_shift n))

def is_abab_form (n : ℕ) : Prop :=
  ∃ a b, a ≠ 0 ∧ b ≠ 0 ∧ n = a * 1000 + b * 100 + a * 10 + b

theorem four_digit_divisor_cyclic_iff_abab (n : ℕ) :
  is_four_digit n ∧ is_divisor_of_cyclic n ↔ is_abab_form n := by sorry

end NUMINAMATH_CALUDE_four_digit_divisor_cyclic_iff_abab_l1764_176431


namespace NUMINAMATH_CALUDE_special_function_result_l1764_176416

/-- A function satisfying the given property for all real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, b^2 * f a = a^2 * f b

theorem special_function_result (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 2 ≠ 0) :
  (f 3 - f 4) / f 2 = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_special_function_result_l1764_176416


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1764_176426

universe u

def U : Set (Fin 4) := {1, 2, 3, 4}
def S : Set (Fin 4) := {1, 3}
def T : Set (Fin 4) := {4}

theorem complement_union_theorem : 
  (U \ S) ∪ T = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1764_176426


namespace NUMINAMATH_CALUDE_initial_weight_calculation_l1764_176494

/-- 
Given a person who:
1. Loses 10% of their initial weight
2. Then gains 2 pounds
3. Ends up weighing 200 pounds

Their initial weight was 220 pounds.
-/
theorem initial_weight_calculation (initial_weight : ℝ) : 
  (initial_weight * 0.9 + 2 = 200) → initial_weight = 220 := by
  sorry

end NUMINAMATH_CALUDE_initial_weight_calculation_l1764_176494


namespace NUMINAMATH_CALUDE_remainder_theorem_l1764_176430

def f (x : ℝ) : ℝ := 4*x^5 - 9*x^4 + 3*x^3 + 5*x^2 - x - 15

theorem remainder_theorem :
  f 4 = 2045 :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1764_176430


namespace NUMINAMATH_CALUDE_line_intersection_l1764_176478

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of a line being contained in a plane
variable (contains : Plane → Line → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the property of a line intersecting another line
variable (intersects : Line → Line → Prop)

-- Theorem statement
theorem line_intersection
  (a b c : Line) (α β : Plane)
  (h1 : skew a b)
  (h2 : contains α a)
  (h3 : contains β b)
  (h4 : c = intersect α β) :
  intersects c a ∨ intersects c b :=
sorry

end NUMINAMATH_CALUDE_line_intersection_l1764_176478


namespace NUMINAMATH_CALUDE_max_distance_product_l1764_176433

/-- Triangle with side lengths 3, 4, and 5 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_eq : a = 3
  b_eq : b = 4
  c_eq : c = 5

/-- Point inside the triangle -/
structure InteriorPoint (t : RightTriangle) where
  x : ℝ
  y : ℝ
  interior : 0 < x ∧ 0 < y ∧ x + y < 1

/-- Distances from a point to the sides of the triangle -/
def distances (t : RightTriangle) (p : InteriorPoint t) : ℝ × ℝ × ℝ :=
  (p.x, p.y, 1 - p.x - p.y)

/-- Product of distances from a point to the sides of the triangle -/
def distanceProduct (t : RightTriangle) (p : InteriorPoint t) : ℝ :=
  let (d₁, d₂, d₃) := distances t p
  d₁ * d₂ * d₃

theorem max_distance_product (t : RightTriangle) :
  ∀ p : InteriorPoint t, distanceProduct t p ≤ 1/125 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_product_l1764_176433


namespace NUMINAMATH_CALUDE_smallest_integer_divisible_by_18_with_sqrt_between_30_and_30_5_l1764_176492

theorem smallest_integer_divisible_by_18_with_sqrt_between_30_and_30_5 :
  ∃ n : ℕ+, (∀ m : ℕ+, m < n → ¬(18 ∣ m ∧ 30 < Real.sqrt m ∧ Real.sqrt m < 30.5)) ∧
            (18 ∣ n) ∧ (30 < Real.sqrt n) ∧ (Real.sqrt n < 30.5) ∧ n = 900 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_divisible_by_18_with_sqrt_between_30_and_30_5_l1764_176492


namespace NUMINAMATH_CALUDE_houses_around_square_l1764_176414

/-- The number of houses around the square. -/
def n : ℕ := 32

/-- Maria's starting position relative to João's. -/
def m_start : ℕ := 8

/-- Proposition that the given conditions imply there are 32 houses around the square. -/
theorem houses_around_square :
  (∀ k : ℕ, (k + 5 - m_start) % n = (k + 12) % n) ∧
  (∀ k : ℕ, (k + 30 - m_start) % n = (k + 5) % n) →
  n = 32 :=
by sorry

end NUMINAMATH_CALUDE_houses_around_square_l1764_176414


namespace NUMINAMATH_CALUDE_cube_sum_integer_l1764_176452

theorem cube_sum_integer (a : ℝ) (ha : a ≠ 0) :
  ∃ k : ℤ, (a + 1/a : ℝ) = k → ∃ m : ℤ, (a^3 + 1/a^3 : ℝ) = m := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_integer_l1764_176452


namespace NUMINAMATH_CALUDE_range_of_m_l1764_176468

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h_eq : 2/x + 3/y = 1) 
  (h_ineq : ∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 3/y = 1 → 3*x + 2*y > m^2 + 2*m) : 
  -6 < m ∧ m < 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1764_176468


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l1764_176438

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the statement
theorem perpendicular_transitivity
  (m n : Line) (α β : Plane)
  (diff_lines : m ≠ n)
  (diff_planes : α ≠ β)
  (n_perp_α : perp n α)
  (n_perp_β : perp n β)
  (m_perp_β : perp m β) :
  perp m α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l1764_176438


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1764_176422

theorem fraction_to_decimal : (49 : ℚ) / (2^3 * 5^4) = 6.125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1764_176422


namespace NUMINAMATH_CALUDE_zero_in_interval_l1764_176400

-- Define the function f(x) = x³ - 2x - 1
def f (x : ℝ) : ℝ := x^3 - 2*x - 1

-- State the theorem
theorem zero_in_interval :
  (f 1.5 < 0) → (f 2 > 0) → ∃ x, x ∈ Set.Ioo 1.5 2 ∧ f x = 0 := by
  sorry

-- Note: Set.Ioo represents an open interval (1.5, 2)

end NUMINAMATH_CALUDE_zero_in_interval_l1764_176400


namespace NUMINAMATH_CALUDE_discriminant_5x2_minus_8x_plus_1_l1764_176421

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 5x^2 - 8x + 1 is 44 -/
theorem discriminant_5x2_minus_8x_plus_1 : discriminant 5 (-8) 1 = 44 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_5x2_minus_8x_plus_1_l1764_176421


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l1764_176449

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem derivative_f_at_one : 
  deriv f 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l1764_176449


namespace NUMINAMATH_CALUDE_money_split_proof_l1764_176497

/-- The total amount of money found by Donna and her friends -/
def total_money : ℝ := 97.50

/-- Donna's share of the money as a percentage -/
def donna_share : ℝ := 0.40

/-- The amount Donna received in dollars -/
def donna_amount : ℝ := 39

/-- Theorem stating that if Donna received 40% of the total money and her share was $39, 
    then the total amount of money found was $97.50 -/
theorem money_split_proof : 
  donna_share * total_money = donna_amount → total_money = 97.50 := by
  sorry

end NUMINAMATH_CALUDE_money_split_proof_l1764_176497


namespace NUMINAMATH_CALUDE_fraction_of_number_l1764_176406

theorem fraction_of_number (original : ℕ) (target : ℚ) : 
  original = 5040 → target = 756.0000000000001 → 
  (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * original = target := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_number_l1764_176406


namespace NUMINAMATH_CALUDE_sandy_molly_age_difference_l1764_176496

theorem sandy_molly_age_difference :
  ∀ (sandy_age molly_age : ℕ),
  sandy_age = 56 →
  sandy_age * 9 = molly_age * 7 →
  molly_age - sandy_age = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_molly_age_difference_l1764_176496


namespace NUMINAMATH_CALUDE_cos_two_alpha_plus_beta_l1764_176470

theorem cos_two_alpha_plus_beta
  (α β : ℝ)
  (h1 : 3 * (Real.sin α)^2 + 2 * (Real.sin β)^2 = 1)
  (h2 : 3 * (Real.sin α + Real.cos α)^2 - 2 * (Real.sin β + Real.cos β)^2 = 1) :
  Real.cos (2 * (α + β)) = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_cos_two_alpha_plus_beta_l1764_176470


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1764_176446

theorem smaller_number_problem (x y : ℝ) 
  (eq1 : 3 * x - y = 20) 
  (eq2 : x + y = 48) : 
  min x y = 17 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1764_176446


namespace NUMINAMATH_CALUDE_chef_nuts_weight_l1764_176410

/-- The total weight of nuts bought by a chef -/
def total_weight (almonds pecans walnuts cashews pistachios : ℝ) : ℝ :=
  almonds + pecans + walnuts + cashews + pistachios

/-- Theorem stating that the total weight of nuts is 1.50 kg -/
theorem chef_nuts_weight :
  let almonds : ℝ := 0.14
  let pecans : ℝ := 0.38
  let walnuts : ℝ := 0.22
  let cashews : ℝ := 0.47
  let pistachios : ℝ := 0.29
  total_weight almonds pecans walnuts cashews pistachios = 1.50 := by
  sorry

end NUMINAMATH_CALUDE_chef_nuts_weight_l1764_176410


namespace NUMINAMATH_CALUDE_sequence_problem_l1764_176429

theorem sequence_problem (a : ℕ → ℕ) (n : ℕ) : 
  a 1 = 2 ∧ 
  (∀ k ≥ 1, a (k + 1) = a k + 3) ∧ 
  a n = 2009 →
  n = 670 := by sorry

end NUMINAMATH_CALUDE_sequence_problem_l1764_176429


namespace NUMINAMATH_CALUDE_alannah_extra_books_l1764_176467

/-- The number of books each person has -/
structure BookCount where
  alannah : ℕ
  beatrix : ℕ
  queen : ℕ

/-- The conditions of the book distribution problem -/
def BookProblem (bc : BookCount) : Prop :=
  bc.alannah > bc.beatrix ∧
  bc.queen = bc.alannah + bc.alannah / 5 ∧
  bc.beatrix = 30 ∧
  bc.alannah + bc.beatrix + bc.queen = 140

/-- The theorem stating that Alannah has 20 more books than Beatrix -/
theorem alannah_extra_books (bc : BookCount) (h : BookProblem bc) : 
  bc.alannah = bc.beatrix + 20 := by
  sorry


end NUMINAMATH_CALUDE_alannah_extra_books_l1764_176467


namespace NUMINAMATH_CALUDE_min_values_l1764_176420

-- Define the equation
def equation (x y : ℝ) : Prop := Real.log (3 * x) + Real.log y = Real.log (x + y + 1)

-- Theorem statement
theorem min_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : equation x y) :
  (∀ a b, equation a b → x * y ≤ a * b) ∧
  (∀ a b, equation a b → x + y ≤ a + b) ∧
  (∀ a b, equation a b → 1 / x + 1 / y ≤ 1 / a + 1 / b) ∧
  x * y = 1 ∧ x + y = 2 ∧ 1 / x + 1 / y = 2 :=
sorry

end NUMINAMATH_CALUDE_min_values_l1764_176420


namespace NUMINAMATH_CALUDE_matrix_identity_proof_l1764_176464

variables {n : Type*} [Fintype n] [DecidableEq n]

theorem matrix_identity_proof 
  (B : Matrix n n ℝ) 
  (h_inv : IsUnit B) 
  (h_eq : (B - 3 • 1) * (B - 5 • 1) = 0) : 
  B + 10 • B⁻¹ = 8 • 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_identity_proof_l1764_176464


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1764_176407

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def f (a : ℕ → ℝ) : ℝ := 1  -- Definition of f, which always returns 1

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 5 / a 3 = 5 / 9) : 
  f a = 1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1764_176407


namespace NUMINAMATH_CALUDE_sqrt_two_over_two_gt_sqrt_three_over_three_l1764_176493

theorem sqrt_two_over_two_gt_sqrt_three_over_three :
  (Real.sqrt 2) / 2 > (Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_over_two_gt_sqrt_three_over_three_l1764_176493


namespace NUMINAMATH_CALUDE_initial_weavers_count_l1764_176456

/-- The number of mat-weavers initially weaving -/
def initial_weavers : ℕ := sorry

/-- The number of mats woven by the initial weavers -/
def initial_mats : ℕ := 4

/-- The number of days taken by the initial weavers -/
def initial_days : ℕ := 4

/-- The number of mat-weavers in the second scenario -/
def second_weavers : ℕ := 8

/-- The number of mats woven in the second scenario -/
def second_mats : ℕ := 16

/-- The number of days taken in the second scenario -/
def second_days : ℕ := 8

/-- The rate of weaving is consistent across both scenarios -/
axiom consistent_rate : 
  (initial_mats : ℚ) / (initial_weavers * initial_days) = 
  (second_mats : ℚ) / (second_weavers * second_days)

theorem initial_weavers_count : initial_weavers = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_weavers_count_l1764_176456


namespace NUMINAMATH_CALUDE_sum_abcd_equals_1986_l1764_176495

theorem sum_abcd_equals_1986 
  (h1 : 6 * a + 2 * b = 3848) 
  (h2 : 6 * c + 3 * d = 4410) 
  (h3 : a + 3 * b + 2 * d = 3080) : 
  a + b + c + d = 1986 := by
  sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_1986_l1764_176495


namespace NUMINAMATH_CALUDE_min_sum_a_b_l1764_176458

theorem min_sum_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 9/b = 1) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 9/y = 1 → a + b ≤ x + y ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 1 ∧ x + y = 16 :=
sorry

end NUMINAMATH_CALUDE_min_sum_a_b_l1764_176458


namespace NUMINAMATH_CALUDE_can_capacity_proof_l1764_176415

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- The capacity of the can in liters -/
def canCapacity : ℝ := 8

theorem can_capacity_proof (initial : CanContents) (final : CanContents) :
  -- Initial ratio of milk to water is 1:5
  initial.milk / initial.water = 1 / 5 →
  -- Final contents after adding 2 liters of milk
  final.milk = initial.milk + 2 ∧
  final.water = initial.water →
  -- New ratio of milk to water is 3:5
  final.milk / final.water = 3 / 5 →
  -- The can is full after adding 2 liters of milk
  final.milk + final.water = canCapacity :=
by sorry


end NUMINAMATH_CALUDE_can_capacity_proof_l1764_176415


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l1764_176489

/-- Given a class with four more girls than boys and 30 total students, 
    prove the ratio of girls to boys is 17/13 -/
theorem girls_to_boys_ratio (g b : ℕ) : 
  g = b + 4 → g + b = 30 → g / b = 17 / 13 := by
  sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l1764_176489


namespace NUMINAMATH_CALUDE_treasure_count_conversion_l1764_176486

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- The deep-sea creature's treasure count in base 7 -/
def treasureCountBase7 : Nat := 245

theorem treasure_count_conversion :
  base7ToBase10 treasureCountBase7 = 131 := by
  sorry

end NUMINAMATH_CALUDE_treasure_count_conversion_l1764_176486


namespace NUMINAMATH_CALUDE_amount_to_return_l1764_176432

/-- Represents the exchange rate in rubles per dollar -/
def exchange_rate : ℝ := 58.15

/-- Represents the initial deposit in USD -/
def initial_deposit : ℝ := 10000

/-- Calculates the amount to be returned in rubles -/
def amount_in_rubles : ℝ := initial_deposit * exchange_rate

/-- Theorem stating that the amount to be returned is 581,500 rubles -/
theorem amount_to_return : amount_in_rubles = 581500 := by
  sorry

end NUMINAMATH_CALUDE_amount_to_return_l1764_176432


namespace NUMINAMATH_CALUDE_fourth_person_height_l1764_176440

/-- Given four people with heights in increasing order, prove that the fourth person is 84 inches tall -/
theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℝ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- heights in increasing order
  h₂ - h₁ = 2 →                 -- difference between first and second
  h₃ - h₂ = 2 →                 -- difference between second and third
  h₄ - h₃ = 6 →                 -- difference between third and fourth
  (h₁ + h₂ + h₃ + h₄) / 4 = 78  -- average height
  → h₄ = 84 := by
sorry

end NUMINAMATH_CALUDE_fourth_person_height_l1764_176440


namespace NUMINAMATH_CALUDE_quadratic_properties_l1764_176479

/-- Quadratic function f(x) = x^2 - 4x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem quadratic_properties :
  (∃ (a b : ℝ), ∀ x, f x = (x - a)^2 + b ∧ a = 2 ∧ b = -1) ∧
  (f 1 = 0 ∧ f 3 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1764_176479


namespace NUMINAMATH_CALUDE_quadratic_function_j_value_l1764_176463

theorem quadratic_function_j_value (a b c : ℤ) (j : ℤ) :
  let f := fun (x : ℤ) => a * x^2 + b * x + c
  (f 1 = 0) →
  (60 < f 7) →
  (f 7 < 70) →
  (80 < f 8) →
  (f 8 < 90) →
  (1000 * j < f 10) →
  (f 10 < 1000 * (j + 1)) →
  j = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_j_value_l1764_176463


namespace NUMINAMATH_CALUDE_value_of_R_l1764_176428

theorem value_of_R : ∀ P Q R : ℚ, 
  P = 4014 / 2 →
  Q = P / 4 →
  R = P - Q →
  R = 1505.25 := by
sorry

end NUMINAMATH_CALUDE_value_of_R_l1764_176428


namespace NUMINAMATH_CALUDE_acorn_theorem_l1764_176427

def acorn_problem (total_acorns : ℕ) 
                  (first_month_allocation : ℚ) 
                  (second_month_allocation : ℚ) 
                  (third_month_allocation : ℚ) 
                  (first_month_consumption : ℚ) 
                  (second_month_consumption : ℚ) 
                  (third_month_consumption : ℚ) : Prop :=
  let first_month := (first_month_allocation * total_acorns : ℚ)
  let second_month := (second_month_allocation * total_acorns : ℚ)
  let third_month := (third_month_allocation * total_acorns : ℚ)
  let remaining_first := first_month * (1 - first_month_consumption)
  let remaining_second := second_month * (1 - second_month_consumption)
  let remaining_third := third_month * (1 - third_month_consumption)
  let total_remaining := remaining_first + remaining_second + remaining_third
  total_acorns = 500 ∧
  first_month_allocation = 2/5 ∧
  second_month_allocation = 3/10 ∧
  third_month_allocation = 3/10 ∧
  first_month_consumption = 1/5 ∧
  second_month_consumption = 1/4 ∧
  third_month_consumption = 3/20 ∧
  total_remaining = 400

theorem acorn_theorem : 
  ∃ (total_acorns : ℕ) 
    (first_month_allocation second_month_allocation third_month_allocation : ℚ)
    (first_month_consumption second_month_consumption third_month_consumption : ℚ),
  acorn_problem total_acorns 
                first_month_allocation 
                second_month_allocation 
                third_month_allocation 
                first_month_consumption 
                second_month_consumption 
                third_month_consumption :=
by
  sorry

end NUMINAMATH_CALUDE_acorn_theorem_l1764_176427


namespace NUMINAMATH_CALUDE_inequality_proof_l1764_176448

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1764_176448


namespace NUMINAMATH_CALUDE_radius_of_circle_in_spherical_coordinates_l1764_176461

/-- The radius of the circle formed by points with spherical coordinates (2, θ, π/4) is √2 -/
theorem radius_of_circle_in_spherical_coordinates : 
  let ρ : ℝ := 2
  let φ : ℝ := π / 4
  Real.sqrt (ρ^2 * Real.sin φ^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_radius_of_circle_in_spherical_coordinates_l1764_176461


namespace NUMINAMATH_CALUDE_polygon_sides_l1764_176472

/-- A convex polygon with the sum of all angles except one equal to 2790° has 18 sides -/
theorem polygon_sides (n : ℕ) (angle_sum : ℝ) : 
  n ≥ 3 → -- convex polygon has at least 3 sides
  angle_sum = 2790 → -- sum of all angles except one is 2790°
  (n - 2) * 180 - angle_sum ≥ 0 → -- the missing angle is non-negative
  (n - 2) * 180 - angle_sum < 180 → -- the missing angle is less than 180°
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l1764_176472


namespace NUMINAMATH_CALUDE_exists_real_less_than_one_exists_natural_in_real_exists_real_between_two_and_three_forall_int_exists_real_outside_interval_l1764_176423

-- 1. Prove that there exists a real number less than 1
theorem exists_real_less_than_one : ∃ x : ℝ, x < 1 := by sorry

-- 2. Prove that there exists a natural number that is also a real number
theorem exists_natural_in_real : ∃ x : ℕ, ∃ y : ℝ, x = y := by sorry

-- 3. Prove that there exists a real number greater than 2 and less than 3
theorem exists_real_between_two_and_three : ∃ x : ℝ, x > 2 ∧ x < 3 := by sorry

-- 4. Prove that for all integers n, there exists a real number x that is either less than n or greater than or equal to n + 1
theorem forall_int_exists_real_outside_interval : ∀ n : ℤ, ∃ x : ℝ, x < n ∨ x ≥ n + 1 := by sorry

end NUMINAMATH_CALUDE_exists_real_less_than_one_exists_natural_in_real_exists_real_between_two_and_three_forall_int_exists_real_outside_interval_l1764_176423


namespace NUMINAMATH_CALUDE_mike_investment_l1764_176484

/-- Prove that Mike's investment is $350 given the partnership conditions --/
theorem mike_investment (mary_investment : ℝ) (total_profit : ℝ) (profit_difference : ℝ) :
  mary_investment = 650 →
  total_profit = 2999.9999999999995 →
  profit_difference = 600 →
  ∃ (mike_investment : ℝ),
    mike_investment = 350 ∧
    (1/3 * total_profit / 2 + 2/3 * total_profit * mary_investment / (mary_investment + mike_investment) =
     1/3 * total_profit / 2 + 2/3 * total_profit * mike_investment / (mary_investment + mike_investment) + profit_difference) :=
by sorry

end NUMINAMATH_CALUDE_mike_investment_l1764_176484


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1764_176454

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1764_176454


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1764_176485

def A : Set ℝ := {x | -2 < x ∧ x ≤ 2}
def B : Set ℝ := {-2, -1, 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1764_176485


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l1764_176490

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l1764_176490


namespace NUMINAMATH_CALUDE_total_students_third_and_fourth_grade_l1764_176417

theorem total_students_third_and_fourth_grade 
  (third_grade : ℕ) 
  (difference : ℕ) 
  (h1 : third_grade = 203)
  (h2 : difference = 125) :
  third_grade + (third_grade + difference) = 531 := by
  sorry

end NUMINAMATH_CALUDE_total_students_third_and_fourth_grade_l1764_176417


namespace NUMINAMATH_CALUDE_jen_profit_l1764_176425

/-- Calculates the profit in cents for a candy bar business -/
def candy_bar_profit (buy_price sell_price bought_quantity sold_quantity : ℕ) : ℤ :=
  (sell_price * sold_quantity : ℤ) - (buy_price * bought_quantity : ℤ)

/-- Proves that Jen's profit from her candy bar business is 800 cents -/
theorem jen_profit : candy_bar_profit 80 100 50 48 = 800 := by
  sorry

end NUMINAMATH_CALUDE_jen_profit_l1764_176425


namespace NUMINAMATH_CALUDE_train_length_train_length_is_120_l1764_176409

/-- The length of a train that overtakes a motorbike -/
theorem train_length (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) 
  (h1 : train_speed = 100) 
  (h2 : motorbike_speed = 64) 
  (h3 : overtake_time = 12) : ℝ :=
let train_speed_ms := train_speed * 1000 / 3600
let motorbike_speed_ms := motorbike_speed * 1000 / 3600
let relative_speed := train_speed_ms - motorbike_speed_ms
120

/-- The length of the train is 120 meters -/
theorem train_length_is_120 (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) 
  (h1 : train_speed = 100) 
  (h2 : motorbike_speed = 64) 
  (h3 : overtake_time = 12) : 
  train_length train_speed motorbike_speed overtake_time h1 h2 h3 = 120 := by
sorry

end NUMINAMATH_CALUDE_train_length_train_length_is_120_l1764_176409


namespace NUMINAMATH_CALUDE_cube_root_negative_three_l1764_176402

theorem cube_root_negative_three (x : ℝ) : x^(1/3) = (-3)^(1/3) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_negative_three_l1764_176402


namespace NUMINAMATH_CALUDE_factorization_equality_l1764_176481

theorem factorization_equality (x y : ℝ) :
  -1/2 * x^3 + 1/8 * x * y^2 = -1/8 * x * (2*x + y) * (2*x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1764_176481
