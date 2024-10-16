import Mathlib

namespace NUMINAMATH_CALUDE_tetrahedron_distance_sum_l946_94679

/-- Theorem about distances in a tetrahedron -/
theorem tetrahedron_distance_sum (V : ℝ) (S₁ S₂ S₃ S₄ : ℝ) (H₁ H₂ H₃ H₄ : ℝ) (k : ℝ) :
  V > 0 →
  S₁ > 0 → S₂ > 0 → S₃ > 0 → S₄ > 0 →
  H₁ > 0 → H₂ > 0 → H₃ > 0 → H₄ > 0 →
  S₁ = k → S₂ = 2*k → S₃ = 3*k → S₄ = 4*k →
  V = (1/3) * (S₁*H₁ + S₂*H₂ + S₃*H₃ + S₄*H₄) →
  H₁ + 2*H₂ + 3*H₃ + 4*H₄ = 3*V/k :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_distance_sum_l946_94679


namespace NUMINAMATH_CALUDE_rectangle_count_in_grid_l946_94624

/-- The number of dots in each row and column of the square array -/
def gridSize : Nat := 5

/-- The number of different rectangles that can be formed in the grid -/
def numRectangles : Nat := (gridSize.choose 2) * (gridSize.choose 2)

theorem rectangle_count_in_grid : numRectangles = 100 := by sorry

end NUMINAMATH_CALUDE_rectangle_count_in_grid_l946_94624


namespace NUMINAMATH_CALUDE_expression_evaluation_l946_94623

theorem expression_evaluation (a b c : ℚ) : 
  a = 5 → 
  b = a + 4 → 
  c = b - 12 → 
  a + 2 ≠ 0 → 
  b - 3 ≠ 0 → 
  c + 7 ≠ 0 → 
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 10) / (c + 7) = 7 / 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l946_94623


namespace NUMINAMATH_CALUDE_square_area_diagonal_relation_l946_94660

theorem square_area_diagonal_relation (d : ℝ) (h : d > 0) :
  ∃ (A : ℝ), A > 0 ∧ A = (1/2) * d^2 ∧ 
  (∃ (s : ℝ), s > 0 ∧ A = s^2 ∧ d^2 = 2 * s^2) := by
  sorry

end NUMINAMATH_CALUDE_square_area_diagonal_relation_l946_94660


namespace NUMINAMATH_CALUDE_percentage_equation_l946_94627

theorem percentage_equation (x : ℝ) : (65 / 100 * x = 20 / 100 * 617.50) → x = 190 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_l946_94627


namespace NUMINAMATH_CALUDE_expression_evaluation_l946_94605

theorem expression_evaluation (x : ℝ) (h : x = -1) :
  (((x - 2) / x - x / (x + 2)) / ((x + 2) / (x^2 + 4*x + 4))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l946_94605


namespace NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l946_94625

theorem neither_sufficient_nor_necessary :
  ¬(∀ x y : ℝ, (x > 1 ∧ y > 1) → (x + y > 3)) ∧
  ¬(∀ x y : ℝ, (x + y > 3) → (x > 1 ∧ y > 1)) := by
  sorry

end NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l946_94625


namespace NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_l946_94673

def number : ℕ := 264

-- Define a function to get the prime factors of a number
def prime_factors (n : ℕ) : List ℕ := sorry

-- Define a function to get the two smallest elements from a list
def two_smallest (l : List ℕ) : List ℕ := sorry

theorem sum_two_smallest_prime_factors :
  (two_smallest (prime_factors number)).sum = 5 := by sorry

end NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_l946_94673


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l946_94672

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_sum : a 1 + 2 * a 2 = 4)
  (h_product : (a 4) ^ 2 = 4 * a 3 * a 7) :
  a 5 = 1/8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l946_94672


namespace NUMINAMATH_CALUDE_alok_rice_order_l946_94681

def chapatis : ℕ := 16
def mixed_vegetable : ℕ := 7
def ice_cream_cups : ℕ := 6
def cost_chapati : ℕ := 6
def cost_rice : ℕ := 45
def cost_mixed_vegetable : ℕ := 70
def cost_ice_cream : ℕ := 40
def total_paid : ℕ := 1051

theorem alok_rice_order :
  ∃ (rice_plates : ℕ),
    rice_plates = 5 ∧
    total_paid = chapatis * cost_chapati +
                 rice_plates * cost_rice +
                 mixed_vegetable * cost_mixed_vegetable +
                 ice_cream_cups * cost_ice_cream :=
by sorry

end NUMINAMATH_CALUDE_alok_rice_order_l946_94681


namespace NUMINAMATH_CALUDE_axis_of_symmetry_point_relationship_t_range_max_t_value_l946_94685

-- Define the parabola
def parabola (t x y : ℝ) : Prop := y = x^2 - 2*t*x + 1

-- Theorem for the axis of symmetry
theorem axis_of_symmetry (t : ℝ) : 
  ∀ x y : ℝ, parabola t x y → parabola t (2*t - x) y := by sorry

-- Theorem for point relationship
theorem point_relationship (t m n : ℝ) :
  parabola t (t-2) m → parabola t (t+3) n → m < n := by sorry

-- Theorem for t range
theorem t_range (t : ℝ) :
  (∀ x₁ y₁ y₂ : ℝ, -1 ≤ x₁ ∧ x₁ < 3 ∧ parabola t x₁ y₁ ∧ parabola t 3 y₂ ∧ y₁ ≤ y₂) 
  → t ≤ 1 := by sorry

-- Theorem for maximum t value
theorem max_t_value :
  ∃ t_max : ℝ, t_max = 5 ∧ 
  ∀ t y₁ y₂ : ℝ, parabola t (t+1) y₁ ∧ parabola t (2*t-4) y₂ ∧ y₁ ≥ y₂ 
  → t ≤ t_max := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_point_relationship_t_range_max_t_value_l946_94685


namespace NUMINAMATH_CALUDE_rental_income_calculation_l946_94603

theorem rental_income_calculation (total_units : ℕ) (occupancy_rate : ℚ) (monthly_rent : ℕ) :
  total_units = 100 →
  occupancy_rate = 3/4 →
  monthly_rent = 400 →
  (total_units : ℚ) * occupancy_rate * (monthly_rent : ℚ) * 12 = 360000 := by
  sorry

end NUMINAMATH_CALUDE_rental_income_calculation_l946_94603


namespace NUMINAMATH_CALUDE_sum_of_cubes_l946_94653

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 1) : a^3 + b^3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l946_94653


namespace NUMINAMATH_CALUDE_frequency_distribution_best_for_proportions_l946_94637

-- Define the possible statistical measures
inductive StatisticalMeasure
  | Average
  | Variance
  | Mode
  | FrequencyDistribution

-- Define a function to determine if a measure can calculate proportions within ranges
def canCalculateProportionsInRange (measure : StatisticalMeasure) : Prop :=
  match measure with
  | StatisticalMeasure.FrequencyDistribution => True
  | _ => False

-- Theorem statement
theorem frequency_distribution_best_for_proportions :
  ∀ (measure : StatisticalMeasure),
    canCalculateProportionsInRange measure →
    measure = StatisticalMeasure.FrequencyDistribution :=
by sorry

end NUMINAMATH_CALUDE_frequency_distribution_best_for_proportions_l946_94637


namespace NUMINAMATH_CALUDE_writer_birthday_theorem_l946_94621

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the number of leap years in a given range -/
def leapYearsInRange (startYear endYear : Nat) : Nat :=
  sorry

/-- Calculates the day of the week given a number of days before Friday -/
def dayBeforeFriday (days : Nat) : DayOfWeek :=
  sorry

theorem writer_birthday_theorem :
  let startYear := 1780
  let endYear := 2020
  let yearsDiff := endYear - startYear
  let leapYears := leapYearsInRange startYear endYear
  let regularYears := yearsDiff - leapYears
  let totalDaysBackward := regularYears + 2 * leapYears
  dayBeforeFriday (totalDaysBackward % 7) = DayOfWeek.Sunday :=
by sorry

end NUMINAMATH_CALUDE_writer_birthday_theorem_l946_94621


namespace NUMINAMATH_CALUDE_sqrt_problem_1_sqrt_problem_2_sqrt_problem_3_sqrt_problem_4_l946_94620

-- 1. Prove that √18 - √32 + √2 = 0
theorem sqrt_problem_1 : Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0 := by sorry

-- 2. Prove that (√27 - √12) / √3 = 1
theorem sqrt_problem_2 : (Real.sqrt 27 - Real.sqrt 12) / Real.sqrt 3 = 1 := by sorry

-- 3. Prove that √(1/6) + √24 - √600 = -43/6 * √6
theorem sqrt_problem_3 : Real.sqrt (1/6) + Real.sqrt 24 - Real.sqrt 600 = -43/6 * Real.sqrt 6 := by sorry

-- 4. Prove that (√3 + 1)(√3 - 1) = 2
theorem sqrt_problem_4 : (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_problem_1_sqrt_problem_2_sqrt_problem_3_sqrt_problem_4_l946_94620


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l946_94684

/-- A regular polygon with an exterior angle of 10 degrees has 36 sides. -/
theorem regular_polygon_sides (n : ℕ) : n > 0 → (360 : ℝ) / n = 10 → n = 36 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l946_94684


namespace NUMINAMATH_CALUDE_expression_value_at_three_l946_94647

theorem expression_value_at_three :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 4 * x + 2
  f 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l946_94647


namespace NUMINAMATH_CALUDE_power_function_above_identity_l946_94629

theorem power_function_above_identity {α : ℝ} :
  (∀ x : ℝ, x ∈ (Set.Ioo 0 1) → x^α > x) ↔ α < 1 :=
sorry

end NUMINAMATH_CALUDE_power_function_above_identity_l946_94629


namespace NUMINAMATH_CALUDE_symmetry_axis_implies_equal_coefficients_l946_94669

/-- Given a function f(x) = a*sin(2x) + b*cos(2x) where ab ≠ 0,
    if f has a symmetry axis at x = π/8, then a = b -/
theorem symmetry_axis_implies_equal_coefficients
  (a b : ℝ) (hab : a * b ≠ 0)
  (h_symmetry : ∀ x : ℝ, a * Real.sin (2 * (π/8 + x)) + b * Real.cos (2 * (π/8 + x)) =
                         a * Real.sin (2 * (π/8 - x)) + b * Real.cos (2 * (π/8 - x))) :
  a = b :=
sorry

end NUMINAMATH_CALUDE_symmetry_axis_implies_equal_coefficients_l946_94669


namespace NUMINAMATH_CALUDE_perpendicular_when_a_neg_one_passes_through_zero_one_l946_94600

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop :=
  (a^2 + a + 1) * x - y + 1 = 0

-- Define perpendicularity of two lines given their slopes
def perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Theorem for statement A
theorem perpendicular_when_a_neg_one :
  perpendicular (-((-1)^2 + (-1) + 1)) 1 :=
sorry

-- Theorem for statement C
theorem passes_through_zero_one (a : ℝ) :
  line_l a 0 1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_when_a_neg_one_passes_through_zero_one_l946_94600


namespace NUMINAMATH_CALUDE_cars_meet_time_l946_94680

/-- Two cars meet on a highway -/
theorem cars_meet_time (highway_length : ℝ) (speed1 speed2 : ℝ) (h1 : highway_length = 500) 
  (h2 : speed1 = 40) (h3 : speed2 = 60) : 
  (highway_length / (speed1 + speed2) = 5) := by
sorry

end NUMINAMATH_CALUDE_cars_meet_time_l946_94680


namespace NUMINAMATH_CALUDE_theater_empty_showtime_l946_94670

/-- Represents a theater --/
structure Theater :=
  (id : Nat)

/-- Represents a student --/
structure Student :=
  (id : Nat)

/-- Represents a showtime --/
structure Showtime :=
  (id : Nat)

/-- Represents the attendance of students at a theater for a specific showtime --/
def Attendance := Theater → Showtime → Finset Student

theorem theater_empty_showtime 
  (students : Finset Student) 
  (theaters : Finset Theater) 
  (showtimes : Finset Showtime) 
  (attendance : Attendance) :
  (students.card = 7) →
  (theaters.card = 7) →
  (showtimes.card = 8) →
  (∀ s : Showtime, ∃! t : Theater, (attendance t s).card = 6) →
  (∀ s : Showtime, ∃! t : Theater, (attendance t s).card = 1) →
  (∀ stud : Student, ∀ t : Theater, ∃ s : Showtime, stud ∈ attendance t s) →
  (∀ t : Theater, ∃ s : Showtime, (attendance t s).card = 0) :=
by sorry

end NUMINAMATH_CALUDE_theater_empty_showtime_l946_94670


namespace NUMINAMATH_CALUDE_quadratic_form_completion_l946_94602

theorem quadratic_form_completion (z : ℝ) : ∃ (b : ℝ) (c : ℤ), z^2 - 6*z + 20 = (z + b)^2 + c ∧ c = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_completion_l946_94602


namespace NUMINAMATH_CALUDE_complex_multiplication_l946_94676

theorem complex_multiplication (i : ℂ) : i * i = -1 → 2 * i * (1 + i) = -2 + 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l946_94676


namespace NUMINAMATH_CALUDE_sequence_converges_to_ones_l946_94644

/-- The operation S applied to a sequence -/
def S (a : Fin (2^n) → Int) : Fin (2^n) → Int :=
  fun i => a i * a (i.succ)

/-- The result of applying S v times -/
def applyS (a : Fin (2^n) → Int) (v : Nat) : Fin (2^n) → Int :=
  match v with
  | 0 => a
  | v+1 => S (applyS a v)

theorem sequence_converges_to_ones 
  (n : Nat) (a : Fin (2^n) → Int) 
  (h : ∀ i, a i = 1 ∨ a i = -1) : 
  ∀ i, applyS a (2^n) i = 1 := by
  sorry

#check sequence_converges_to_ones

end NUMINAMATH_CALUDE_sequence_converges_to_ones_l946_94644


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l946_94687

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2 - 3*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l946_94687


namespace NUMINAMATH_CALUDE_meeting_participants_l946_94661

theorem meeting_participants :
  ∀ (F M : ℕ),
  F > 0 →
  M > 0 →
  F / 2 = 130 →
  F / 2 + M / 4 = (F + M) / 3 →
  F + M = 780 :=
by
  sorry

end NUMINAMATH_CALUDE_meeting_participants_l946_94661


namespace NUMINAMATH_CALUDE_orange_picking_theorem_l946_94651

/-- The total number of oranges picked over three days -/
def totalOranges (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) : ℕ :=
  day1 + day2 + day3

/-- Theorem stating the total number of oranges picked -/
theorem orange_picking_theorem :
  let day1 := 100
  let day2 := 3 * day1
  let day3 := 70
  totalOranges day1 day2 day3 = 470 := by
  sorry

end NUMINAMATH_CALUDE_orange_picking_theorem_l946_94651


namespace NUMINAMATH_CALUDE_hockey_team_starters_l946_94691

/-- The number of ways to choose starters from a hockey team with quadruplets -/
def chooseStarters (totalPlayers : ℕ) (quadruplets : ℕ) (starters : ℕ) (maxQuadruplets : ℕ) : ℕ :=
  (Nat.choose (totalPlayers - quadruplets) starters) +
  (quadruplets * Nat.choose (totalPlayers - quadruplets) (starters - 1)) +
  (Nat.choose quadruplets 2 * Nat.choose (totalPlayers - quadruplets) (starters - 2))

/-- The theorem stating the correct number of ways to choose starters -/
theorem hockey_team_starters :
  chooseStarters 18 4 7 2 = 27456 := by
  sorry

end NUMINAMATH_CALUDE_hockey_team_starters_l946_94691


namespace NUMINAMATH_CALUDE_certain_number_is_900_l946_94639

theorem certain_number_is_900 :
  ∃ x : ℝ, (45 * 9 = 0.45 * x) ∧ (x = 900) :=
by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_900_l946_94639


namespace NUMINAMATH_CALUDE_dolls_distribution_count_l946_94677

def distribute_dolls (n_dolls : ℕ) (n_houses : ℕ) : ℕ :=
  let choose_two := n_dolls.choose 2
  let select_house := n_houses
  let arrange_rest := (n_dolls - 2).factorial
  choose_two * select_house * arrange_rest

theorem dolls_distribution_count :
  distribute_dolls 7 6 = 15120 :=
by sorry

end NUMINAMATH_CALUDE_dolls_distribution_count_l946_94677


namespace NUMINAMATH_CALUDE_tangent_point_and_zeros_l946_94611

noncomputable def f (a x : ℝ) : ℝ := 2 * Real.exp x + 2 * a * x - x + 3 - a^2

theorem tangent_point_and_zeros (a : ℝ) :
  (∃ x : ℝ, f a x = 0 ∧ (∀ y : ℝ, f a y ≥ 0)) ↔ a = Real.log 3 - 3 ∧
  (∀ x : ℝ, x > 0 →
    (((a ≤ -Real.sqrt 5 ∨ a = Real.log 3 - 3 ∨ a > Real.sqrt 5) →
      (∃! y : ℝ, y > 0 ∧ f a y = 0)) ∧
    ((-Real.sqrt 5 < a ∧ a < Real.log 3 - 3) →
      (∃ y z : ℝ, 0 < y ∧ y < z ∧ f a y = 0 ∧ f a z = 0 ∧
        ∀ w : ℝ, 0 < w ∧ w ≠ y ∧ w ≠ z → f a w ≠ 0)) ∧
    ((Real.log 3 - 3 < a ∧ a ≤ Real.sqrt 5) →
      (∀ y : ℝ, y > 0 → f a y ≠ 0)))) :=
sorry

end NUMINAMATH_CALUDE_tangent_point_and_zeros_l946_94611


namespace NUMINAMATH_CALUDE_total_rent_is_245_l946_94608

-- Define the oxen-months for each person
def a_oxen_months : ℕ := 10 * 7
def b_oxen_months : ℕ := 12 * 5
def c_oxen_months : ℕ := 15 * 3

-- Define the total oxen-months
def total_oxen_months : ℕ := a_oxen_months + b_oxen_months + c_oxen_months

-- Define c's payment
def c_payment : ℚ := 62.99999999999999

-- Define the cost per oxen-month
def cost_per_oxen_month : ℚ := c_payment / c_oxen_months

-- Theorem to prove
theorem total_rent_is_245 : 
  ∃ (total_rent : ℚ), total_rent = cost_per_oxen_month * total_oxen_months ∧ 
                       total_rent = 245 := by
  sorry

end NUMINAMATH_CALUDE_total_rent_is_245_l946_94608


namespace NUMINAMATH_CALUDE_perpendicular_slope_l946_94659

/-- The slope of a line perpendicular to the line passing through (3, -4) and (-2, 5) is 5/9 -/
theorem perpendicular_slope : 
  let x₁ : ℚ := 3
  let y₁ : ℚ := -4
  let x₂ : ℚ := -2
  let y₂ : ℚ := 5
  let m : ℚ := (y₂ - y₁) / (x₂ - x₁)
  (- (1 / m)) = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l946_94659


namespace NUMINAMATH_CALUDE_rope_percentage_theorem_l946_94613

theorem rope_percentage_theorem (total_length used_length : ℝ) 
  (h1 : total_length = 20)
  (h2 : used_length = 15) :
  used_length / total_length = 0.75 ∧ (1 - used_length / total_length) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_rope_percentage_theorem_l946_94613


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l946_94664

theorem necessary_but_not_sufficient :
  (∀ a b c d : ℝ, (a > b ∧ c > d) → (a + c > b + d)) ∧
  (∃ a b c d : ℝ, (a + c > b + d) ∧ ¬(a > b ∧ c > d)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l946_94664


namespace NUMINAMATH_CALUDE_power_mul_l946_94656

theorem power_mul (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by sorry

end NUMINAMATH_CALUDE_power_mul_l946_94656


namespace NUMINAMATH_CALUDE_fraction_equality_l946_94628

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 5) : 
  (2 * m * r - 5 * n * t) / (5 * n * t - 4 * m * r) = -2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l946_94628


namespace NUMINAMATH_CALUDE_quadrilateral_area_l946_94617

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (quad : Quadrilateral) : Prop :=
  let (px, py) := quad.P
  let (qx, qy) := quad.Q
  let (rx, ry) := quad.R
  let (sx, sy) := quad.S
  -- PQ = 6
  (px - qx)^2 + (py - qy)^2 = 36 ∧
  -- QR = 8
  (qx - rx)^2 + (qy - ry)^2 = 64 ∧
  -- RS = 15
  (rx - sx)^2 + (ry - sy)^2 = 225 ∧
  -- PS = 17
  (px - sx)^2 + (py - sy)^2 = 289 ∧
  -- Angle PQR = 90°
  (px - qx) * (rx - qx) + (py - qy) * (ry - qy) = 0

-- Define the area calculation function
noncomputable def area (quad : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area (quad : Quadrilateral) 
  (h : is_valid_quadrilateral quad) : area quad = 98.5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l946_94617


namespace NUMINAMATH_CALUDE_perpendicular_length_l946_94698

/-- A parallelogram with a diagonal of length 'd' and area 'a' has a perpendicular of length 'h' dropped on that diagonal. -/
structure Parallelogram where
  d : ℝ  -- length of the diagonal
  a : ℝ  -- area of the parallelogram
  h : ℝ  -- length of the perpendicular dropped on the diagonal

/-- The area of a parallelogram is equal to the product of its diagonal and the perpendicular dropped on that diagonal. -/
axiom area_formula (p : Parallelogram) : p.a = p.d * p.h

/-- For a parallelogram with a diagonal of 30 meters and an area of 600 square meters, 
    the length of the perpendicular dropped on the diagonal is 20 meters. -/
theorem perpendicular_length : 
  ∀ (p : Parallelogram), p.d = 30 → p.a = 600 → p.h = 20 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_length_l946_94698


namespace NUMINAMATH_CALUDE_donut_selection_equals_object_distribution_l946_94649

/-- The number of ways to select n donuts from k types with at least one of a specific type -/
def donut_selections (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 2) (k - 1)

/-- The number of ways to distribute m objects into k distinct boxes -/
def object_distribution (m k : ℕ) : ℕ :=
  Nat.choose (m + k - 1) (k - 1)

theorem donut_selection_equals_object_distribution :
  donut_selections 5 4 = object_distribution 4 4 :=
by sorry

end NUMINAMATH_CALUDE_donut_selection_equals_object_distribution_l946_94649


namespace NUMINAMATH_CALUDE_number_manipulation_l946_94674

theorem number_manipulation (x : ℝ) : (x - 5) / 7 = 7 → (x - 14) / 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_manipulation_l946_94674


namespace NUMINAMATH_CALUDE_stone_slab_length_l946_94641

theorem stone_slab_length (total_area : Real) (num_slabs : Nat) (slab_length : Real) : 
  total_area = 58.8 ∧ 
  num_slabs = 30 ∧ 
  slab_length * slab_length * num_slabs = total_area * 10000 →
  slab_length = 140 := by
sorry

end NUMINAMATH_CALUDE_stone_slab_length_l946_94641


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l946_94619

theorem trigonometric_equation_solution (t : ℝ) : 
  (2 * (Real.sin (2 * t))^5 - (Real.sin (2 * t))^3 - 6 * (Real.sin (2 * t))^2 + 3 = 0) ↔ 
  (∃ k : ℤ, t = (π / 8) * (2 * ↑k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l946_94619


namespace NUMINAMATH_CALUDE_circle_P_equation_l946_94607

/-- The curve C defined by the distance ratio condition -/
def C (x y : ℝ) : Prop :=
  (x^2 / 3 + y^2 / 2 = 1)

/-- The line l intersecting curve C -/
def l (x y : ℝ) (k : ℝ) : Prop :=
  (y = k * (x - 1) - 1)

/-- Points A and B are on both C and l -/
def A_and_B_on_C_and_l (x₁ y₁ x₂ y₂ k : ℝ) : Prop :=
  C x₁ y₁ ∧ C x₂ y₂ ∧ l x₁ y₁ k ∧ l x₂ y₂ k

/-- AB is the diameter of circle P centered at (1, -1) -/
def P_diameter (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = -1

theorem circle_P_equation (x₁ y₁ x₂ y₂ k : ℝ) :
  A_and_B_on_C_and_l x₁ y₁ x₂ y₂ k →
  P_diameter x₁ y₁ x₂ y₂ →
  k = 2/3 →
  ∀ x y, (x - 1)^2 + (y + 1)^2 = 13/30 :=
sorry

end NUMINAMATH_CALUDE_circle_P_equation_l946_94607


namespace NUMINAMATH_CALUDE_overall_score_calculation_l946_94604

theorem overall_score_calculation (score1 score2 score3 : ℚ) 
  (problems1 problems2 problems3 : ℕ) : 
  score1 = 60 / 100 →
  score2 = 75 / 100 →
  score3 = 85 / 100 →
  problems1 = 15 →
  problems2 = 25 →
  problems3 = 20 →
  (score1 * problems1 + score2 * problems2 + score3 * problems3) / 
  (problems1 + problems2 + problems3 : ℚ) = 75 / 100 := by
  sorry

end NUMINAMATH_CALUDE_overall_score_calculation_l946_94604


namespace NUMINAMATH_CALUDE_sum_below_threshold_equals_14_tenths_l946_94646

def numbers : List ℚ := [14/10, 9/10, 12/10, 5/10, 13/10]
def threshold : ℚ := 11/10

def sum_below_threshold (nums : List ℚ) (t : ℚ) : ℚ :=
  (nums.filter (· ≤ t)).sum

theorem sum_below_threshold_equals_14_tenths :
  sum_below_threshold numbers threshold = 14/10 := by
  sorry

end NUMINAMATH_CALUDE_sum_below_threshold_equals_14_tenths_l946_94646


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l946_94693

theorem concentric_circles_ratio (r₁ r₂ r₃ : ℝ) (h₁ : 0 < r₁) (h₂ : r₁ < r₂) (h₃ : r₂ < r₃) :
  (π * r₁^2 / 4 = π * (r₂^2 - r₁^2) / 4) ∧ 
  (π * (r₂^2 - r₁^2) / 4 = π * (r₃^2 - r₂^2) / 4) →
  ∃ (k : ℝ), k > 0 ∧ r₁ = k ∧ r₂ = k * Real.sqrt 2 ∧ r₃ = k * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l946_94693


namespace NUMINAMATH_CALUDE_three_toppings_from_seven_l946_94638

theorem three_toppings_from_seven (n : ℕ) (k : ℕ) : n = 7 ∧ k = 3 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_three_toppings_from_seven_l946_94638


namespace NUMINAMATH_CALUDE_midpoint_sum_midpoint_sum_specific_l946_94631

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (3, -1) and (11, 21) is 17. -/
theorem midpoint_sum : ℝ × ℝ → ℝ × ℝ → ℝ
  | (x₁, y₁) => λ (x₂, y₂) => (x₁ + x₂) / 2 + (y₁ + y₂) / 2

#check midpoint_sum (3, -1) (11, 21) = 17

theorem midpoint_sum_specific :
  midpoint_sum (3, -1) (11, 21) = 17 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_midpoint_sum_specific_l946_94631


namespace NUMINAMATH_CALUDE_joe_money_left_l946_94689

def initial_amount : ℕ := 56
def notebooks_bought : ℕ := 7
def books_bought : ℕ := 2
def notebook_cost : ℕ := 4
def book_cost : ℕ := 7

theorem joe_money_left : 
  initial_amount - (notebooks_bought * notebook_cost + books_bought * book_cost) = 14 := by
  sorry

end NUMINAMATH_CALUDE_joe_money_left_l946_94689


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l946_94658

/-- The perimeter of a semicircle with radius 12 is approximately 61.7 -/
theorem semicircle_perimeter_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  abs ((12 * Real.pi + 24) - 61.7) < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l946_94658


namespace NUMINAMATH_CALUDE_midpoint_of_intersection_l946_94626

-- Define the curve
def curve (t : ℝ) : ℝ × ℝ := (t + 1, (t - 1)^2)

-- Define the ray at θ = π/4
def ray (x : ℝ) : ℝ × ℝ := (x, x)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, curve t = p ∧ ray p.1 = p}

-- Theorem statement
theorem midpoint_of_intersection :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
  (A.1 + B.1) / 2 = 2.5 ∧ (A.2 + B.2) / 2 = 2.5 :=
sorry

end NUMINAMATH_CALUDE_midpoint_of_intersection_l946_94626


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l946_94671

/-- The perpendicular bisector of a line segment with endpoints (x₁, y₁) and (x₂, y₂) -/
def perpendicular_bisector (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - x₁)^2 + (p.2 - y₁)^2 = (p.1 - x₂)^2 + (p.2 - y₂)^2}

theorem perpendicular_bisector_equation :
  perpendicular_bisector 1 3 5 (-1) = {p : ℝ × ℝ | p.1 - p.2 - 2 = 0} := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l946_94671


namespace NUMINAMATH_CALUDE_find_number_l946_94648

theorem find_number : ∃! x : ℝ, ((35 - x) * 2 + 12) / 8 = 9 := by sorry

end NUMINAMATH_CALUDE_find_number_l946_94648


namespace NUMINAMATH_CALUDE_circle_coloring_exists_l946_94678

/-- A type representing the two colors we can use -/
inductive Color
  | Red
  | Blue

/-- A type representing a region in the plane -/
structure Region

/-- A type representing a circle in the plane -/
structure Circle

/-- A function that determines if two regions are adjacent (separated by an arc of a circle) -/
def adjacent (r1 r2 : Region) : Prop := sorry

/-- A coloring function that assigns a color to each region -/
def coloring (r : Region) : Color := sorry

/-- The main theorem stating that a valid coloring exists for any number of circles -/
theorem circle_coloring_exists (n : ℕ) (h : n ≥ 1) :
  ∃ (circles : Finset Circle) (regions : Finset Region),
    circles.card = n ∧
    (∀ r1 r2 : Region, r1 ∈ regions → r2 ∈ regions → adjacent r1 r2 → coloring r1 ≠ coloring r2) :=
  sorry

end NUMINAMATH_CALUDE_circle_coloring_exists_l946_94678


namespace NUMINAMATH_CALUDE_min_distance_exp_ln_l946_94618

/-- The minimum distance between a point on y = e^x and a point on y = ln(x) -/
theorem min_distance_exp_ln : ∃ (d : ℝ), d = Real.sqrt 2 ∧ 
  ∀ (P Q : ℝ × ℝ), 
    (P.2 = Real.exp P.1) → (Q.2 = Real.log Q.1) → 
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_exp_ln_l946_94618


namespace NUMINAMATH_CALUDE_rooks_knight_move_theorem_l946_94665

/-- Represents a position on a chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a knight's move -/
structure KnightMove :=
  (drow : Int)
  (dcol : Int)

/-- Checks if a move is a valid knight's move -/
def isValidKnightMove (km : KnightMove) : Prop :=
  (km.drow.natAbs = 2 ∧ km.dcol.natAbs = 1) ∨ 
  (km.drow.natAbs = 1 ∧ km.dcol.natAbs = 2)

/-- Applies a knight's move to a position -/
def applyMove (p : Position) (km : KnightMove) : Position :=
  ⟨p.row + km.drow, p.col + km.dcol⟩

/-- Checks if two positions are non-attacking for rooks -/
def nonAttacking (p1 p2 : Position) : Prop :=
  p1.row ≠ p2.row ∧ p1.col ≠ p2.col

/-- The main theorem -/
theorem rooks_knight_move_theorem 
  (initial_positions : Fin 8 → Position)
  (h_initial_non_attacking : ∀ i j, i ≠ j → 
    nonAttacking (initial_positions i) (initial_positions j)) :
  ∃ (moves : Fin 8 → KnightMove),
    (∀ i, isValidKnightMove (moves i)) ∧
    (∀ i j, i ≠ j → 
      nonAttacking 
        (applyMove (initial_positions i) (moves i))
        (applyMove (initial_positions j) (moves j))) :=
  sorry


end NUMINAMATH_CALUDE_rooks_knight_move_theorem_l946_94665


namespace NUMINAMATH_CALUDE_expression_factorization_l946_94666

theorem expression_factorization (b : ℝ) :
  (3 * b^4 + 66 * b^3 - 14) - (-4 * b^4 + 2 * b^3 - 14) = b^3 * (7 * b + 64) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l946_94666


namespace NUMINAMATH_CALUDE_ray_fish_market_l946_94652

/-- Calculates the number of tuna needed to serve customers in a fish market -/
def tuna_needed (total_customers : ℕ) (unsatisfied_customers : ℕ) (pounds_per_customer : ℕ) (pounds_per_tuna : ℕ) : ℕ :=
  ((total_customers - unsatisfied_customers) * pounds_per_customer) / pounds_per_tuna

theorem ray_fish_market :
  tuna_needed 100 20 25 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ray_fish_market_l946_94652


namespace NUMINAMATH_CALUDE_dvd_book_total_capacity_l946_94609

def dvd_book_capacity (current_dvds : ℕ) (additional_dvds : ℕ) : ℕ :=
  current_dvds + additional_dvds

theorem dvd_book_total_capacity :
  dvd_book_capacity 81 45 = 126 := by
  sorry

end NUMINAMATH_CALUDE_dvd_book_total_capacity_l946_94609


namespace NUMINAMATH_CALUDE_train_length_l946_94632

/-- The length of a train given its speed, time to cross a platform, and the platform's length -/
theorem train_length (speed : ℝ) (time : ℝ) (platform_length : ℝ) : 
  speed = 72 → time = 25 → platform_length = 250.04 → 
  speed * (5/18) * time - platform_length = 249.96 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l946_94632


namespace NUMINAMATH_CALUDE_f_properties_l946_94610

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 then -x - 1
  else if -1 ≤ x ∧ x ≤ 1 then -x^2 + 1
  else x - 1

-- Theorem statement
theorem f_properties :
  (f 2 = 1 ∧ f (-2) = 1) ∧
  (∀ a : ℝ, f a = 1 ↔ a = -2 ∨ a = 0 ∨ a = 2) ∧
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, (-1 ≤ x ∧ x < y ∧ y ≤ 0) → f x < f y) ∧
  (∀ x y : ℝ, (1 ≤ x ∧ x < y) → f x < f y) ∧
  (∀ x y : ℝ, (x < y ∧ y ≤ -1) → f x > f y) ∧
  (∀ x y : ℝ, (0 < x ∧ x < y ∧ y ≤ 1) → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l946_94610


namespace NUMINAMATH_CALUDE_f_nonnegative_condition_f_two_zeros_condition_l946_94614

/-- The function f(x) defined as |x^2 - 1| + x^2 + kx -/
def f (k : ℝ) (x : ℝ) : ℝ := |x^2 - 1| + x^2 + k*x

theorem f_nonnegative_condition (k : ℝ) :
  (∀ x > 0, f k x ≥ 0) ↔ k ≥ -1 := by sorry

theorem f_two_zeros_condition (k : ℝ) (x₁ x₂ : ℝ) :
  (0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0) →
  (-7/2 < k ∧ k < -1 ∧ 2 < 1/x₁ + 1/x₂ ∧ 1/x₁ + 1/x₂ < 4) := by sorry

end NUMINAMATH_CALUDE_f_nonnegative_condition_f_two_zeros_condition_l946_94614


namespace NUMINAMATH_CALUDE_intersection_points_form_circle_l946_94636

-- Define the system of equations
def equation1 (s x y : ℝ) : Prop := 3 * s * x - 5 * y - 7 * s = 0
def equation2 (s x y : ℝ) : Prop := 2 * x - 5 * s * y + 4 = 0

-- Define the set of points satisfying both equations
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ s : ℝ, equation1 s p.1 p.2 ∧ equation2 s p.1 p.2}

-- Theorem stating that the intersection points form a circle
theorem intersection_points_form_circle :
  ∃ c : ℝ × ℝ, ∃ r : ℝ, ∀ p ∈ intersection_points,
    (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_form_circle_l946_94636


namespace NUMINAMATH_CALUDE_democrat_ratio_l946_94650

/-- Prove that the ratio of democrats to total participants is 1:3 -/
theorem democrat_ratio (total : ℕ) (female_democrats : ℕ) :
  total = 780 →
  female_democrats = 130 →
  (∃ (female male : ℕ),
    female + male = total ∧
    female = 2 * female_democrats ∧
    4 * (female_democrats + male / 4) = total / 3) :=
by sorry

end NUMINAMATH_CALUDE_democrat_ratio_l946_94650


namespace NUMINAMATH_CALUDE_point_movement_l946_94657

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given conditions and proof goal -/
theorem point_movement :
  let A : Point := ⟨a - 5, 2 * b - 1⟩
  let B : Point := ⟨3 * a + 2, b + 3⟩
  let C : Point := ⟨a, b⟩
  A.x = 0 →  -- A lies on y-axis
  B.y = 0 →  -- B lies on x-axis
  (⟨C.x + 2, C.y - 3⟩ : Point) = ⟨7, -6⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_movement_l946_94657


namespace NUMINAMATH_CALUDE_triangle_angle_sum_and_type_l946_94616

/-- A triangle with angles a, b, and c is right if its largest angle is 90 degrees --/
def is_right_triangle (a b c : ℝ) : Prop :=
  max a (max b c) = 90

theorem triangle_angle_sum_and_type 
  (a b : ℝ) 
  (ha : a = 56)
  (hb : b = 34) :
  let c := 180 - a - b
  ∃ (x : ℝ), x = c ∧ x = 90 ∧ is_right_triangle a b c :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_and_type_l946_94616


namespace NUMINAMATH_CALUDE_four_digit_sum_gcd_quotient_l946_94642

theorem four_digit_sum_gcd_quotient
  (a b c d : Nat)
  (h1 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10)
  (h2 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h3 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  let S := a + b + c + d
  let G := Nat.gcd a (Nat.gcd b (Nat.gcd c d))
  (33 * S - S * G) / S = 33 - G :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_gcd_quotient_l946_94642


namespace NUMINAMATH_CALUDE_prob_sum_gt_five_l946_94622

/-- The probability of rolling two dice and getting a sum greater than five -/
def prob_sum_greater_than_five : ℚ := 2/3

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of outcomes where the sum is less than or equal to five -/
def outcomes_sum_le_five : ℕ := 12

theorem prob_sum_gt_five :
  prob_sum_greater_than_five = 1 - (outcomes_sum_le_five : ℚ) / total_outcomes :=
sorry

end NUMINAMATH_CALUDE_prob_sum_gt_five_l946_94622


namespace NUMINAMATH_CALUDE_bryan_uninterested_offer_is_one_l946_94606

/-- Represents the record sale scenario --/
structure RecordSale where
  total_records : ℕ
  sammy_offer : ℚ
  bryan_offer_interested : ℚ
  bryan_interested_fraction : ℚ
  profit_difference : ℚ

/-- Calculates Bryan's offer for uninterested records --/
def bryan_uninterested_offer (sale : RecordSale) : ℚ :=
  let sammy_total := sale.total_records * sale.sammy_offer
  let bryan_interested_records := sale.total_records * sale.bryan_interested_fraction
  let bryan_uninterested_records := sale.total_records - bryan_interested_records
  let bryan_interested_total := bryan_interested_records * sale.bryan_offer_interested
  (sammy_total - bryan_interested_total - sale.profit_difference) / bryan_uninterested_records

/-- Theorem stating Bryan's offer for uninterested records is $1 --/
theorem bryan_uninterested_offer_is_one (sale : RecordSale)
    (h1 : sale.total_records = 200)
    (h2 : sale.sammy_offer = 4)
    (h3 : sale.bryan_offer_interested = 6)
    (h4 : sale.bryan_interested_fraction = 1/2)
    (h5 : sale.profit_difference = 100) :
    bryan_uninterested_offer sale = 1 := by
  sorry

end NUMINAMATH_CALUDE_bryan_uninterested_offer_is_one_l946_94606


namespace NUMINAMATH_CALUDE_largest_angle_in_circle_l946_94612

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the angle between three points
def angle (A B C : Point) : ℝ := sorry

-- Define a function to check if a point is inside a circle
def isInside (p : Point) (c : Circle) : Prop := sorry

-- Define a function to check if a point is on the circumference of a circle
def isOnCircumference (p : Point) (c : Circle) : Prop := sorry

-- Define a function to check if three points form a diameter of a circle
def formsDiameter (A B C : Point) (circle : Circle) : Prop := sorry

theorem largest_angle_in_circle (circle : Circle) (A B : Point) 
  (hA : isInside A circle) (hB : isInside B circle) :
  ∃ C, isOnCircumference C circle ∧ 
    (∀ D, isOnCircumference D circle → angle A B C ≥ angle A B D) ∧
    formsDiameter A B C circle := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_circle_l946_94612


namespace NUMINAMATH_CALUDE_janet_lives_lost_l946_94663

/-- The number of lives Janet lost in the hard part of the game -/
def lives_lost : ℕ := sorry

theorem janet_lives_lost :
  (∃ (initial_lives : ℕ) (gained_lives : ℕ),
    initial_lives = 38 ∧
    gained_lives = 32 ∧
    initial_lives - lives_lost + gained_lives = 54) →
  lives_lost = 16 := by sorry

end NUMINAMATH_CALUDE_janet_lives_lost_l946_94663


namespace NUMINAMATH_CALUDE_betty_needs_five_more_l946_94654

def wallet_cost : ℕ := 100
def betty_initial_savings : ℕ := wallet_cost / 2
def parents_contribution : ℕ := 15
def grandparents_contribution : ℕ := 2 * parents_contribution

theorem betty_needs_five_more :
  wallet_cost - (betty_initial_savings + parents_contribution + grandparents_contribution) = 5 := by
  sorry

end NUMINAMATH_CALUDE_betty_needs_five_more_l946_94654


namespace NUMINAMATH_CALUDE_sum_of_digits_3_plus_4_pow_17_l946_94645

/-- The sum of the tens digit and the ones digit of (3+4)^17 in integer form is 7 -/
theorem sum_of_digits_3_plus_4_pow_17 : 
  (((3 + 4)^17 / 10) % 10 + (3 + 4)^17 % 10) = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_3_plus_4_pow_17_l946_94645


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_inverse_squares_l946_94697

theorem quadratic_roots_sum_inverse_squares (a b c k : ℝ) (kr ks : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : a * kr^2 + k * c * kr + b = 0) 
  (h4 : a * ks^2 + k * c * ks + b = 0) 
  (h5 : kr ≠ 0) (h6 : ks ≠ 0) : 
  1 / kr^2 + 1 / ks^2 = (k^2 * c^2 - 2 * a * b) / b^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_inverse_squares_l946_94697


namespace NUMINAMATH_CALUDE_multiplication_result_l946_94667

theorem multiplication_result : 2.68 * 0.74 = 1.9832 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_result_l946_94667


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l946_94692

/-- Given a line with equation x + y + 1 = 0 and a point of symmetry (1, 2),
    the symmetric line has the equation x + y - 7 = 0 -/
theorem symmetric_line_equation :
  let original_line := {(x, y) : ℝ × ℝ | x + y + 1 = 0}
  let symmetry_point := (1, 2)
  let symmetric_line := {(x, y) : ℝ × ℝ | x + y - 7 = 0}
  ∀ (p : ℝ × ℝ), p ∈ symmetric_line ↔
    (2 * symmetry_point.1 - p.1, 2 * symmetry_point.2 - p.2) ∈ original_line :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l946_94692


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l946_94688

theorem smallest_undefined_inverse (b : ℕ) : b = 6 ↔ 
  (b > 0) ∧ 
  (∀ x : ℕ, x * b % 30 ≠ 1) ∧ 
  (∀ y : ℕ, y * b % 42 ≠ 1) ∧ 
  (∀ c < b, c > 0 → (∃ x : ℕ, x * c % 30 = 1) ∨ (∃ y : ℕ, y * c % 42 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l946_94688


namespace NUMINAMATH_CALUDE_binomial_coefficient_26_6_l946_94643

theorem binomial_coefficient_26_6 (h1 : Nat.choose 24 5 = 42504) (h2 : Nat.choose 24 6 = 134596) :
  Nat.choose 26 6 = 230230 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_26_6_l946_94643


namespace NUMINAMATH_CALUDE_volunteer_assignment_count_l946_94695

/-- The number of ways to assign volunteers to tasks -/
def assignment_count (n : ℕ) (k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n

/-- The problem statement -/
theorem volunteer_assignment_count :
  assignment_count 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_assignment_count_l946_94695


namespace NUMINAMATH_CALUDE_solution_value_l946_94655

-- Define the function E
def E (a b c : ℚ) : ℚ := a * b^2 + c

-- State the theorem
theorem solution_value :
  ∃ (a : ℚ), E a 3 10 = E a 5 (-2) ∧ a = 3/4 := by sorry

end NUMINAMATH_CALUDE_solution_value_l946_94655


namespace NUMINAMATH_CALUDE_function_value_difference_bound_l946_94633

theorem function_value_difference_bound
  (f : Set.Icc 0 1 → ℝ)
  (h₁ : f ⟨0, by norm_num⟩ = f ⟨1, by norm_num⟩)
  (h₂ : ∀ (x₁ x₂ : Set.Icc 0 1), x₁ ≠ x₂ → |f x₂ - f x₁| < |x₂.val - x₁.val|) :
  ∀ (x₁ x₂ : Set.Icc 0 1), |f x₂ - f x₁| < (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_difference_bound_l946_94633


namespace NUMINAMATH_CALUDE_total_amount_proof_l946_94694

def calculate_total_amount (plant_price tool_price soil_price : ℝ)
  (plant_discount tool_discount : ℝ) (tax_rate : ℝ) (surcharge : ℝ) : ℝ :=
  let discounted_plant := plant_price * (1 - plant_discount)
  let discounted_tool := tool_price * (1 - tool_discount)
  let subtotal := discounted_plant + discounted_tool + soil_price
  let total_with_tax := subtotal * (1 + tax_rate)
  total_with_tax + surcharge

theorem total_amount_proof :
  calculate_total_amount 467 85 38 0.15 0.10 0.08 12 = 564.37 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_proof_l946_94694


namespace NUMINAMATH_CALUDE_remainder_b_sixth_l946_94683

theorem remainder_b_sixth (n : ℕ+) (b : ℤ) (h : b^3 ≡ 1 [ZMOD n]) : b^6 ≡ 1 [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_remainder_b_sixth_l946_94683


namespace NUMINAMATH_CALUDE_total_canoes_by_april_l946_94640

def canoes_per_month (month : Nat) : Nat :=
  match month with
  | 0 => 4  -- January (0-indexed)
  | n + 1 => 3 * canoes_per_month n

theorem total_canoes_by_april : 
  (canoes_per_month 0) + (canoes_per_month 1) + (canoes_per_month 2) + (canoes_per_month 3) = 160 := by
  sorry

end NUMINAMATH_CALUDE_total_canoes_by_april_l946_94640


namespace NUMINAMATH_CALUDE_square_root_of_16_l946_94615

theorem square_root_of_16 : {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_16_l946_94615


namespace NUMINAMATH_CALUDE_field_purchase_problem_l946_94662

theorem field_purchase_problem :
  let good_field_value : ℚ := 300  -- value of 1 acre of good field
  let bad_field_value : ℚ := 500 / 7  -- value of 1 acre of bad field
  let total_area : ℚ := 100  -- total area in acres
  let total_cost : ℚ := 10000  -- total cost in coins
  let good_field_acres : ℚ := 25 / 2  -- solution for good field acres
  let bad_field_acres : ℚ := 175 / 2  -- solution for bad field acres
  (good_field_acres + bad_field_acres = total_area) ∧
  (good_field_value * good_field_acres + bad_field_value * bad_field_acres = total_cost) :=
by sorry


end NUMINAMATH_CALUDE_field_purchase_problem_l946_94662


namespace NUMINAMATH_CALUDE_always_odd_l946_94668

theorem always_odd (n : ℤ) : ∃ k : ℤ, n^2 + n + 5 = 2*k + 1 := by
  sorry

end NUMINAMATH_CALUDE_always_odd_l946_94668


namespace NUMINAMATH_CALUDE_joan_balloons_l946_94630

/-- Joan and Melanie's blue balloons problem -/
theorem joan_balloons (joan_balloons : ℕ) (melanie_balloons : ℕ) (total_balloons : ℕ)
    (h1 : melanie_balloons = 41)
    (h2 : total_balloons = 81)
    (h3 : joan_balloons + melanie_balloons = total_balloons) :
  joan_balloons = 40 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l946_94630


namespace NUMINAMATH_CALUDE_intersection_complement_problem_l946_94635

open Set

theorem intersection_complement_problem (U M N : Set ℕ) : 
  U = {0, 1, 2, 3, 4, 5} →
  M = {0, 3, 5} →
  N = {1, 4, 5} →
  M ∩ (U \ N) = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_problem_l946_94635


namespace NUMINAMATH_CALUDE_simplify_fraction_l946_94699

theorem simplify_fraction : (222 : ℚ) / 8888 * 22 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l946_94699


namespace NUMINAMATH_CALUDE_triangle_formation_count_l946_94601

/-- The number of ways to choose k elements from a set of n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of checkpoints on the first track -/
def checkpoints_track1 : ℕ := 6

/-- The number of checkpoints on the second track -/
def checkpoints_track2 : ℕ := 10

/-- The number of ways to form triangles by selecting one point from the first track
    and two points from the second track -/
def triangle_formations : ℕ := checkpoints_track1 * choose checkpoints_track2 2

theorem triangle_formation_count :
  triangle_formations = 270 := by sorry

end NUMINAMATH_CALUDE_triangle_formation_count_l946_94601


namespace NUMINAMATH_CALUDE_time_saved_by_bike_l946_94690

/-- Given that it takes Mike 98 minutes to walk to school and riding a bicycle saves him 64 minutes,
    prove that the time saved by Mike when riding a bicycle is 64 minutes. -/
theorem time_saved_by_bike (walking_time : ℕ) (time_saved : ℕ) 
  (h1 : walking_time = 98) 
  (h2 : time_saved = 64) : 
  time_saved = 64 := by
  sorry

end NUMINAMATH_CALUDE_time_saved_by_bike_l946_94690


namespace NUMINAMATH_CALUDE_box_volume_l946_94634

theorem box_volume (l w h : ℝ) 
  (side1 : l * w = 120)
  (side2 : w * h = 72)
  (top : l * h = 60) :
  l * w * h = 720 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l946_94634


namespace NUMINAMATH_CALUDE_gwen_gave_away_seven_games_l946_94682

/-- The number of games Gwen gave away -/
def games_given_away (initial_games : ℕ) (remaining_games : ℕ) : ℕ :=
  initial_games - remaining_games

/-- Proof that Gwen gave away 7 games -/
theorem gwen_gave_away_seven_games :
  let initial_games : ℕ := 98
  let remaining_games : ℕ := 91
  games_given_away initial_games remaining_games = 7 := by
  sorry

end NUMINAMATH_CALUDE_gwen_gave_away_seven_games_l946_94682


namespace NUMINAMATH_CALUDE_basketball_practice_time_ratio_l946_94675

theorem basketball_practice_time_ratio :
  ∀ (total_practice_time shooting_time weightlifting_time running_time : ℕ),
  total_practice_time = 120 →
  shooting_time = total_practice_time / 2 →
  weightlifting_time = 20 →
  running_time = total_practice_time - shooting_time - weightlifting_time →
  running_time / weightlifting_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_basketball_practice_time_ratio_l946_94675


namespace NUMINAMATH_CALUDE_table_price_is_84_l946_94696

/-- The price of a chair in dollars -/
def chair_price : ℝ := sorry

/-- The price of a table in dollars -/
def table_price : ℝ := sorry

/-- The condition that the price of 2 chairs and 1 table is 60% of the price of 1 chair and 2 tables -/
def price_ratio_condition : Prop :=
  2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price)

/-- The condition that the price of 1 table and 1 chair is $96 -/
def total_price_condition : Prop :=
  chair_price + table_price = 96

theorem table_price_is_84 
  (h1 : price_ratio_condition) 
  (h2 : total_price_condition) : 
  table_price = 84 := by sorry

end NUMINAMATH_CALUDE_table_price_is_84_l946_94696


namespace NUMINAMATH_CALUDE_larger_number_proof_l946_94686

theorem larger_number_proof (L S : ℕ) (hL : L > S) : 
  L - S = 1000 → L = 10 * S + 10 → L = 1110 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l946_94686
