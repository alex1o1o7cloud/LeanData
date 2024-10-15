import Mathlib

namespace NUMINAMATH_CALUDE_plane_perpendicularity_condition_l3446_344608

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (subset : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity_condition 
  (α β : Plane) (l : Line) 
  (h1 : α ≠ β) 
  (h2 : subset l α) :
  (∀ l, subset l α → perpendicular l β → plane_perpendicular α β) ∧ 
  (∃ l, subset l α ∧ plane_perpendicular α β ∧ ¬perpendicular l β) :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicularity_condition_l3446_344608


namespace NUMINAMATH_CALUDE_sum_squares_of_roots_l3446_344601

theorem sum_squares_of_roots (x₁ x₂ : ℝ) : 
  (3 * x₁^2 + 4 * x₁ - 9 = 0) →
  (3 * x₂^2 + 4 * x₂ - 9 = 0) →
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 70/9) := by
sorry

end NUMINAMATH_CALUDE_sum_squares_of_roots_l3446_344601


namespace NUMINAMATH_CALUDE_curve_is_parabola_l3446_344650

theorem curve_is_parabola (θ : Real) (r : Real → Real) (x y : Real) :
  (r θ = 1 / (1 - Real.sin θ)) →
  (x^2 + y^2 = r θ^2) →
  (y = r θ * Real.sin θ) →
  (x^2 = 2*y + 1) :=
by
  sorry

#check curve_is_parabola

end NUMINAMATH_CALUDE_curve_is_parabola_l3446_344650


namespace NUMINAMATH_CALUDE_correct_sum_l3446_344637

theorem correct_sum (a b : ℕ) (h1 : a % 10 = 1) (h2 : b / 10 % 10 = 8) 
  (h3 : (a - 1 + 7) + (b - 80 + 30) = 1946) : a + b = 1990 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_sum_l3446_344637


namespace NUMINAMATH_CALUDE_inequality_implications_l3446_344689

theorem inequality_implications (a b : ℝ) (h : a + 1 > b + 1) :
  (a > b) ∧ (a + 2 > b + 2) ∧ (-a < -b) ∧ ¬(∀ a b : ℝ, a + 1 > b + 1 → 2*a > 3*b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_implications_l3446_344689


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l3446_344679

def is_prime (n : ℕ) : Prop := sorry

def is_cube (n : ℕ) : Prop := sorry

def ends_with (a b : ℕ) : Prop := sorry

def digit_sum (n : ℕ) : ℕ := sorry

theorem smallest_number_with_conditions (p : ℕ) (hp_prime : is_prime p) (hp_cube : is_cube p) :
  ∃ (A : ℕ), 
    A = 11713 ∧ 
    p = 13 ∧
    p ∣ A ∧ 
    ends_with A p ∧ 
    digit_sum A = p ∧ 
    ∀ (B : ℕ), (p ∣ B ∧ ends_with B p ∧ digit_sum B = p) → A ≤ B :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l3446_344679


namespace NUMINAMATH_CALUDE_min_circles_6x3_min_circles_5x3_l3446_344610

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define a circle
structure Circle where
  radius : ℝ

-- Define a function to calculate the minimum number of circles needed to cover a rectangle
def minCircles (r : Rectangle) (c : Circle) : ℕ :=
  sorry

-- Theorem for 6 × 3 rectangle
theorem min_circles_6x3 :
  let r := Rectangle.mk 6 3
  let c := Circle.mk (Real.sqrt 2)
  minCircles r c = 6 :=
sorry

-- Theorem for 5 × 3 rectangle
theorem min_circles_5x3 :
  let r := Rectangle.mk 5 3
  let c := Circle.mk (Real.sqrt 2)
  minCircles r c = 5 :=
sorry

end NUMINAMATH_CALUDE_min_circles_6x3_min_circles_5x3_l3446_344610


namespace NUMINAMATH_CALUDE_otimes_inequality_solution_set_l3446_344677

-- Define the custom operation ⊗
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem otimes_inequality_solution_set :
  ∀ x : ℝ, (otimes (x - 2) (x + 2) < 2) ↔ (x < 0 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_otimes_inequality_solution_set_l3446_344677


namespace NUMINAMATH_CALUDE_ferris_wheel_cost_calculation_l3446_344662

/-- The cost of a Ferris wheel ride in tickets -/
def ferris_wheel_cost : ℕ := sorry

/-- The number of Ferris wheel rides -/
def ferris_wheel_rides : ℕ := 2

/-- The cost of a roller coaster ride in tickets -/
def roller_coaster_cost : ℕ := 5

/-- The number of roller coaster rides -/
def roller_coaster_rides : ℕ := 3

/-- The cost of a log ride in tickets -/
def log_ride_cost : ℕ := 1

/-- The number of log rides -/
def log_ride_rides : ℕ := 7

/-- The initial number of tickets Dolly has -/
def initial_tickets : ℕ := 20

/-- The number of additional tickets Dolly buys -/
def additional_tickets : ℕ := 6

theorem ferris_wheel_cost_calculation :
  ferris_wheel_cost = 2 :=
sorry

end NUMINAMATH_CALUDE_ferris_wheel_cost_calculation_l3446_344662


namespace NUMINAMATH_CALUDE_garment_fraction_l3446_344647

theorem garment_fraction (bikini_fraction trunks_fraction : ℝ) 
  (h1 : bikini_fraction = 0.38) 
  (h2 : trunks_fraction = 0.25) : 
  bikini_fraction + trunks_fraction = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_garment_fraction_l3446_344647


namespace NUMINAMATH_CALUDE_math_sequences_count_l3446_344606

theorem math_sequences_count : 
  let letters := ['M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'S']
  let n := letters.length
  let first_letter := 'M'
  let last_letter_options := (letters.filter (· ≠ 'A')).filter (· ≠ first_letter)
  let middle_letters_count := 2
  (n - 1 - middle_letters_count).factorial * 
  last_letter_options.length * 
  (Nat.choose (n - 2) middle_letters_count) = 392 := by
sorry

end NUMINAMATH_CALUDE_math_sequences_count_l3446_344606


namespace NUMINAMATH_CALUDE_power_multiplication_l3446_344626

theorem power_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3446_344626


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3446_344627

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 1 - Complex.I → z = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3446_344627


namespace NUMINAMATH_CALUDE_quiche_volume_l3446_344686

/-- Calculates the total volume of a quiche given the ingredients' volumes and spinach reduction factor. -/
theorem quiche_volume 
  (raw_spinach : ℝ) 
  (reduction_factor : ℝ) 
  (cream_cheese : ℝ) 
  (eggs : ℝ) 
  (h1 : 0 < reduction_factor) 
  (h2 : reduction_factor < 1) :
  raw_spinach * reduction_factor + cream_cheese + eggs = 
  (raw_spinach * reduction_factor + cream_cheese + eggs) := by
  sorry

end NUMINAMATH_CALUDE_quiche_volume_l3446_344686


namespace NUMINAMATH_CALUDE_cyclic_sum_equals_two_l3446_344613

theorem cyclic_sum_equals_two (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_prod : a * b * c * d = 1) : 
  (1 + a + a*b) / (1 + a + a*b + a*b*c) + 
  (1 + b + b*c) / (1 + b + b*c + b*c*d) + 
  (1 + c + c*d) / (1 + c + c*d + c*d*a) + 
  (1 + d + d*a) / (1 + d + d*a + d*a*b) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_equals_two_l3446_344613


namespace NUMINAMATH_CALUDE_parabola_through_points_with_parallel_tangent_l3446_344669

/-- A parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The slope of the tangent line to the parabola at a given x-coordinate -/
def Parabola.tangent_slope (p : Parabola) (x : ℝ) : ℝ :=
  2 * p.a * x + p.b

/-- Theorem stating the conditions and the result to be proved -/
theorem parabola_through_points_with_parallel_tangent 
  (p : Parabola) 
  (h1 : p.y_coord 1 = 1) 
  (h2 : p.y_coord 2 = -1) 
  (h3 : p.tangent_slope 2 = 1) : 
  p.a = 3 ∧ p.b = -11 ∧ p.c = 9 := by
  sorry


end NUMINAMATH_CALUDE_parabola_through_points_with_parallel_tangent_l3446_344669


namespace NUMINAMATH_CALUDE_wire_length_proof_l3446_344653

theorem wire_length_proof (shorter_piece : ℝ) (longer_piece : ℝ) : 
  shorter_piece = 14.285714285714285 →
  shorter_piece = (2/5) * longer_piece →
  shorter_piece + longer_piece = 50 := by
sorry

end NUMINAMATH_CALUDE_wire_length_proof_l3446_344653


namespace NUMINAMATH_CALUDE_quadratic_properties_l3446_344603

def f (x : ℝ) := -x^2 + 3*x + 1

theorem quadratic_properties :
  (∀ x y, x < y → f y < f x) ∧ 
  (3/2 = -(-3)/(2*(-1))) ∧
  (∀ x y, x < y → y < 3/2 → f x < f y) ∧
  (∀ x, f x = 0 → x < 4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3446_344603


namespace NUMINAMATH_CALUDE_total_money_collected_l3446_344675

/-- Calculates the total money collected from ticket sales given the prices and attendance. -/
theorem total_money_collected 
  (adult_price : ℚ) 
  (child_price : ℚ) 
  (total_attendance : ℕ) 
  (children_attendance : ℕ) 
  (h1 : adult_price = 60 / 100) 
  (h2 : child_price = 25 / 100) 
  (h3 : total_attendance = 280) 
  (h4 : children_attendance = 80) :
  (total_attendance - children_attendance) * adult_price + children_attendance * child_price = 140 / 100 := by
sorry

end NUMINAMATH_CALUDE_total_money_collected_l3446_344675


namespace NUMINAMATH_CALUDE_pasture_rent_problem_l3446_344654

/-- Represents the number of oxen each person puts in the pasture -/
structure OxenCount where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the number of months each person's oxen graze -/
structure GrazingMonths where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total oxen-months for all three people -/
def totalOxenMonths (oxen : OxenCount) (months : GrazingMonths) : ℕ :=
  oxen.a * months.a + oxen.b * months.b + oxen.c * months.c

/-- Calculates a person's share of the rent based on their oxen-months -/
def rentShare (totalRent : ℚ) (oxenMonths : ℕ) (totalOxenMonths : ℕ) : ℚ :=
  totalRent * (oxenMonths : ℚ) / (totalOxenMonths : ℚ)

theorem pasture_rent_problem (totalRent : ℚ) (oxen : OxenCount) (months : GrazingMonths) 
    (h1 : totalRent = 175)
    (h2 : oxen.a = 10 ∧ oxen.b = 12 ∧ oxen.c = 15)
    (h3 : months.a = 7 ∧ months.b = 5)
    (h4 : rentShare totalRent (oxen.c * months.c) (totalOxenMonths oxen months) = 45) :
    months.c = 3 := by
  sorry

end NUMINAMATH_CALUDE_pasture_rent_problem_l3446_344654


namespace NUMINAMATH_CALUDE_average_rounds_is_three_l3446_344643

/-- Represents the number of golfers who played a certain number of rounds -/
def GolferDistribution := List (ℕ × ℕ)

/-- Calculates the total number of rounds played by all golfers -/
def totalRounds (dist : GolferDistribution) : ℕ :=
  dist.foldl (fun acc (rounds, golfers) => acc + rounds * golfers) 0

/-- Calculates the total number of golfers -/
def totalGolfers (dist : GolferDistribution) : ℕ :=
  dist.foldl (fun acc (_, golfers) => acc + golfers) 0

/-- Rounds a rational number to the nearest integer -/
def roundToNearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem average_rounds_is_three (golfData : GolferDistribution) 
  (h : golfData = [(1, 4), (2, 3), (3, 6), (4, 2), (5, 4), (6, 1)]) : 
  roundToNearest (totalRounds golfData / totalGolfers golfData) = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_rounds_is_three_l3446_344643


namespace NUMINAMATH_CALUDE_river_road_cars_l3446_344681

/-- Proves that the number of cars on River Road is 60 -/
theorem river_road_cars :
  ∀ (buses cars motorcycles : ℕ),
    (buses : ℚ) / cars = 1 / 3 →
    cars = buses + 40 →
    buses + cars + motorcycles = 720 →
    cars = 60 := by
  sorry

end NUMINAMATH_CALUDE_river_road_cars_l3446_344681


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3446_344641

theorem p_necessary_not_sufficient_for_q :
  (∀ a b : ℝ, a^2 + b^2 = 0 → a + b = 0) ∧
  (∃ a b : ℝ, a + b = 0 ∧ a^2 + b^2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3446_344641


namespace NUMINAMATH_CALUDE_investment_theorem_l3446_344682

/-- Calculates the total investment with interest after one year -/
def total_investment_with_interest (total_investment : ℝ) (amount_at_3_percent : ℝ) : ℝ :=
  let amount_at_5_percent := total_investment - amount_at_3_percent
  let interest_at_3_percent := amount_at_3_percent * 0.03
  let interest_at_5_percent := amount_at_5_percent * 0.05
  total_investment + interest_at_3_percent + interest_at_5_percent

/-- Theorem stating that the total investment with interest is $1,046 -/
theorem investment_theorem :
  total_investment_with_interest 1000 199.99999999999983 = 1046 := by
  sorry

end NUMINAMATH_CALUDE_investment_theorem_l3446_344682


namespace NUMINAMATH_CALUDE_field_length_calculation_l3446_344628

theorem field_length_calculation (width length : ℝ) (pond_side : ℝ) : 
  length = 2 * width →
  pond_side = 8 →
  pond_side^2 = (1 / 50) * (length * width) →
  length = 80 := by
sorry

end NUMINAMATH_CALUDE_field_length_calculation_l3446_344628


namespace NUMINAMATH_CALUDE_travis_cereal_weeks_l3446_344620

/-- Proves the number of weeks Travis eats cereal given his consumption and spending habits -/
theorem travis_cereal_weeks (boxes_per_week : ℕ) (cost_per_box : ℚ) (total_spent : ℚ) :
  boxes_per_week = 2 →
  cost_per_box = 3 →
  total_spent = 312 →
  (total_spent / (boxes_per_week * cost_per_box) : ℚ) = 52 := by
  sorry

end NUMINAMATH_CALUDE_travis_cereal_weeks_l3446_344620


namespace NUMINAMATH_CALUDE_sum_of_digits_of_factorials_of_fib_l3446_344617

-- Define the first 10 Fibonacci numbers
def fib : List Nat := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

-- Function to calculate factorial
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Function to sum the digits of a number
def sumDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumDigits (n / 10)

-- Theorem statement
theorem sum_of_digits_of_factorials_of_fib : 
  (fib.map (λ x => sumDigits (factorial x))).sum = 240 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_digits_of_factorials_of_fib_l3446_344617


namespace NUMINAMATH_CALUDE_min_moves_correct_l3446_344631

/-- The minimum number of moves in Bethan's grid game -/
def min_moves (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n^2 / 2 + n
  else
    (n^2 + 1) / 2

/-- Theorem stating the minimum number of moves in Bethan's grid game -/
theorem min_moves_correct (n : ℕ) (h : n > 0) :
  min_moves n = if n % 2 = 0 then n^2 / 2 + n else (n^2 + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_moves_correct_l3446_344631


namespace NUMINAMATH_CALUDE_election_win_percentage_l3446_344623

theorem election_win_percentage 
  (total_voters : ℕ) 
  (republican_ratio : ℚ) 
  (democrat_ratio : ℚ) 
  (republican_for_x : ℚ) 
  (democrat_for_x : ℚ) :
  republican_ratio + democrat_ratio = 1 →
  republican_ratio / democrat_ratio = 3 / 2 →
  republican_for_x = 3 / 4 →
  democrat_for_x = 3 / 20 →
  let total_for_x := republican_ratio * republican_for_x + democrat_ratio * democrat_for_x
  let total_for_y := 1 - total_for_x
  (total_for_x - total_for_y) / (total_for_x + total_for_y) = 1 / 50 :=
by sorry

end NUMINAMATH_CALUDE_election_win_percentage_l3446_344623


namespace NUMINAMATH_CALUDE_parents_contribution_half_l3446_344609

/-- Represents the financial details for Nancy's university tuition --/
structure TuitionFinances where
  tuition : ℕ
  scholarship : ℕ
  workHours : ℕ
  hourlyWage : ℕ

/-- Calculates the ratio of parents' contribution to total tuition --/
def parentsContributionRatio (finances : TuitionFinances) : Rat :=
  let studentLoan := 2 * finances.scholarship
  let totalAid := finances.scholarship + studentLoan
  let workEarnings := finances.workHours * finances.hourlyWage
  let nancyContribution := totalAid + workEarnings
  let parentsContribution := finances.tuition - nancyContribution
  parentsContribution / finances.tuition

/-- Theorem stating that the parents' contribution ratio is 1/2 --/
theorem parents_contribution_half (finances : TuitionFinances) 
  (h1 : finances.tuition = 22000)
  (h2 : finances.scholarship = 3000)
  (h3 : finances.workHours = 200)
  (h4 : finances.hourlyWage = 10) :
  parentsContributionRatio finances = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parents_contribution_half_l3446_344609


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l3446_344625

theorem pure_imaginary_complex_fraction (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l3446_344625


namespace NUMINAMATH_CALUDE_school_students_count_prove_school_students_count_l3446_344618

theorem school_students_count : ℕ → Prop :=
  fun total_students =>
    let chess_students := (total_students : ℚ) * (1 / 10)
    let swimming_students := chess_students * (1 / 2)
    swimming_students = 100 →
    total_students = 2000

-- The proof is omitted
theorem prove_school_students_count :
  ∃ (n : ℕ), school_students_count n :=
sorry

end NUMINAMATH_CALUDE_school_students_count_prove_school_students_count_l3446_344618


namespace NUMINAMATH_CALUDE_work_completion_l3446_344612

theorem work_completion (days_first_group : ℝ) (men_second_group : ℕ) (days_second_group : ℝ) :
  days_first_group = 25 →
  men_second_group = 20 →
  days_second_group = 18.75 →
  ∃ (men_first_group : ℕ), 
    men_first_group * days_first_group = men_second_group * days_second_group ∧
    men_first_group = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_l3446_344612


namespace NUMINAMATH_CALUDE_quadratic_zero_in_interval_l3446_344644

/-- Given a quadratic function f(x) = ax^2 + bx + c, prove that it has a zero in the interval (-2, 0) under certain conditions. -/
theorem quadratic_zero_in_interval
  (a b c : ℝ)
  (h1 : 2 * a + c / 2 > b)
  (h2 : c < 0) :
  ∃ x : ℝ, -2 < x ∧ x < 0 ∧ a * x^2 + b * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_zero_in_interval_l3446_344644


namespace NUMINAMATH_CALUDE_xyz_value_l3446_344683

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 15 - x * y * z) :
  x * y * z = 15 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l3446_344683


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l3446_344690

theorem sphere_radius_ratio (v_large v_small : ℝ) (h1 : v_large = 432 * Real.pi) (h2 : v_small = 0.25 * v_large) :
  (∃ r_small r_large : ℝ, 
    v_small = (4/3) * Real.pi * r_small^3 ∧ 
    v_large = (4/3) * Real.pi * r_large^3 ∧ 
    r_small / r_large = 1 / (2^(2/3))) := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l3446_344690


namespace NUMINAMATH_CALUDE_festival_attendance_l3446_344688

theorem festival_attendance (total : ℕ) (first_day : ℕ) : 
  total = 2700 →
  first_day + (first_day / 2) + (3 * first_day) = total →
  first_day / 2 = 300 :=
by sorry

end NUMINAMATH_CALUDE_festival_attendance_l3446_344688


namespace NUMINAMATH_CALUDE_helen_raisin_cookies_l3446_344697

/-- The number of raisin cookies Helen baked yesterday -/
def raisin_cookies_yesterday : ℕ := sorry

/-- The number of chocolate chip cookies Helen baked yesterday -/
def choc_chip_cookies_yesterday : ℕ := 519

/-- The number of raisin cookies Helen baked today -/
def raisin_cookies_today : ℕ := 280

/-- The number of chocolate chip cookies Helen baked today -/
def choc_chip_cookies_today : ℕ := 359

/-- Helen baked 20 more raisin cookies yesterday compared to today -/
axiom raisin_cookies_difference : raisin_cookies_yesterday = raisin_cookies_today + 20

theorem helen_raisin_cookies : raisin_cookies_yesterday = 300 := by sorry

end NUMINAMATH_CALUDE_helen_raisin_cookies_l3446_344697


namespace NUMINAMATH_CALUDE_triangle_tangent_ratio_l3446_344615

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if tan A * tan B = 4(tan A + tan B) * tan C, then (a^2 + b^2) / c^2 = 9 -/
theorem triangle_tangent_ratio (a b c A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) → (0 < B) → (B < π) → (0 < C) → (C < π) →
  (A + B + C = π) →
  (Real.tan A * Real.tan B = 4 * (Real.tan A + Real.tan B) * Real.tan C) →
  ((a^2 + b^2) / c^2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_ratio_l3446_344615


namespace NUMINAMATH_CALUDE_all_chameleons_green_chameleon_color_convergence_l3446_344655

/-- Represents the colors of chameleons --/
inductive Color
| Yellow
| Red
| Green

/-- Represents the state of chameleons on the island --/
structure ChameleonState where
  yellow : Nat
  red : Nat
  green : Nat

/-- The initial state of chameleons --/
def initialState : ChameleonState :=
  { yellow := 7, red := 10, green := 17 }

/-- The total number of chameleons --/
def totalChameleons : Nat := 34

/-- Function to model the color change when two different colored chameleons meet --/
def colorChange (c1 c2 : Color) : Color :=
  match c1, c2 with
  | Color.Yellow, Color.Red => Color.Green
  | Color.Red, Color.Yellow => Color.Green
  | Color.Yellow, Color.Green => Color.Red
  | Color.Green, Color.Yellow => Color.Red
  | Color.Red, Color.Green => Color.Yellow
  | Color.Green, Color.Red => Color.Yellow
  | _, _ => c1  -- No change if same color

/-- Theorem stating that all chameleons will eventually be green --/
theorem all_chameleons_green (finalState : ChameleonState) : 
  (finalState.yellow + finalState.red + finalState.green = totalChameleons) →
  (finalState.yellow = 0 ∧ finalState.red = 0 ∧ finalState.green = totalChameleons) :=
sorry

/-- Main theorem to prove --/
theorem chameleon_color_convergence :
  ∃ (finalState : ChameleonState),
    (finalState.yellow + finalState.red + finalState.green = totalChameleons) ∧
    (finalState.yellow = 0 ∧ finalState.red = 0 ∧ finalState.green = totalChameleons) :=
sorry

end NUMINAMATH_CALUDE_all_chameleons_green_chameleon_color_convergence_l3446_344655


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l3446_344614

theorem simplify_radical_expression :
  Real.sqrt 10 - Real.sqrt 40 + Real.sqrt 90 = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l3446_344614


namespace NUMINAMATH_CALUDE_investment_rate_calculation_l3446_344646

theorem investment_rate_calculation 
  (total_investment : ℝ) 
  (first_rate : ℝ) 
  (first_amount : ℝ) 
  (total_interest : ℝ) :
  total_investment = 10000 →
  first_rate = 0.06 →
  first_amount = 7200 →
  total_interest = 684 →
  let second_amount := total_investment - first_amount
  let first_interest := first_amount * first_rate
  let second_interest := total_interest - first_interest
  let second_rate := second_interest / second_amount
  second_rate = 0.09 := by sorry

end NUMINAMATH_CALUDE_investment_rate_calculation_l3446_344646


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3446_344639

theorem trigonometric_simplification (α : ℝ) :
  (2 * Real.sin (π - α) + Real.sin (2 * α)) / (2 * (Real.cos (α / 2))^2) = 2 * Real.sin α :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3446_344639


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3446_344621

theorem rectangular_field_area (L W : ℝ) : 
  L = 40 →                 -- One side (length) is 40 feet
  2 * W + L = 74 →         -- Total fencing is 74 feet (two widths plus one length)
  L * W = 680 :=           -- The area of the field is 680 square feet
by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3446_344621


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l3446_344678

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l3446_344678


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3446_344638

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | x^2 - 2*a*x - 3*a^2 < 0} = {x : ℝ | 3*a < x ∧ x < -a} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3446_344638


namespace NUMINAMATH_CALUDE_population_growth_over_three_years_l3446_344668

/-- Represents the demographic rates for a given year -/
structure YearlyRates where
  birth_rate : ℝ
  death_rate : ℝ
  in_migration : ℝ
  out_migration : ℝ

/-- Calculates the net growth rate for a given year -/
def net_growth_rate (rates : YearlyRates) : ℝ :=
  rates.birth_rate + rates.in_migration - rates.death_rate - rates.out_migration

/-- Theorem stating the net percentage increase in population over three years -/
theorem population_growth_over_three_years 
  (year1 : YearlyRates)
  (year2 : YearlyRates)
  (year3 : YearlyRates)
  (h1 : year1 = { birth_rate := 0.025, death_rate := 0.01, in_migration := 0.03, out_migration := 0.02 })
  (h2 : year2 = { birth_rate := 0.02, death_rate := 0.015, in_migration := 0.04, out_migration := 0.035 })
  (h3 : year3 = { birth_rate := 0.022, death_rate := 0.008, in_migration := 0.025, out_migration := 0.01 })
  : ∃ (ε : ℝ), abs ((1 + net_growth_rate year1) * (1 + net_growth_rate year2) * (1 + net_growth_rate year3) - 1 - 0.065675) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_over_three_years_l3446_344668


namespace NUMINAMATH_CALUDE_f_positive_all_reals_f_positive_interval_l3446_344667

/-- The quadratic function f(x) = x^2 + 2(a-2)x + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-2)*x + 4

/-- Theorem 1: f(x) > 0 for all x ∈ ℝ if and only if 0 < a < 4 -/
theorem f_positive_all_reals (a : ℝ) :
  (∀ x : ℝ, f a x > 0) ↔ (0 < a ∧ a < 4) :=
sorry

/-- Theorem 2: f(x) > 0 for x ∈ [-3, 1] if and only if a ∈ (-1/2, 4) -/
theorem f_positive_interval (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-3) 1 → f a x > 0) ↔ (a > -1/2 ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_f_positive_all_reals_f_positive_interval_l3446_344667


namespace NUMINAMATH_CALUDE_root_intersection_l3446_344622

-- Define the original equation
def original_equation (x : ℝ) : Prop := x^2 - 2*x = 0

-- Define the roots of the original equation
def is_root (x : ℝ) : Prop := original_equation x

-- Define the pairs of equations
def pair_A (x y : ℝ) : Prop := (y = x^2 ∧ y = 2*x)
def pair_B (x y : ℝ) : Prop := (y = x^2 - 2*x ∧ y = 0)
def pair_C (x y : ℝ) : Prop := (y = x ∧ y = x - 2)
def pair_D (x y : ℝ) : Prop := (y = x^2 - 2*x + 1 ∧ y = 1)
def pair_E (x y : ℝ) : Prop := (y = x^2 - 1 ∧ y = 2*x - 1)

-- Theorem stating that pair C does not yield the roots while others do
theorem root_intersection :
  (∃ x y : ℝ, pair_C x y ∧ is_root x) = false ∧
  (∃ x y : ℝ, pair_A x y ∧ is_root x) = true ∧
  (∃ x y : ℝ, pair_B x y ∧ is_root x) = true ∧
  (∃ x y : ℝ, pair_D x y ∧ is_root x) = true ∧
  (∃ x y : ℝ, pair_E x y ∧ is_root x) = true :=
by sorry

end NUMINAMATH_CALUDE_root_intersection_l3446_344622


namespace NUMINAMATH_CALUDE_fraction_power_product_l3446_344648

theorem fraction_power_product : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l3446_344648


namespace NUMINAMATH_CALUDE_paradise_park_large_seats_l3446_344674

/-- Represents a Ferris wheel with small and large seats. -/
structure FerrisWheel where
  smallSeats : Nat
  largeSeats : Nat
  smallSeatCapacity : Nat
  largeSeatCapacity : Nat
  totalLargeSeatCapacity : Nat

/-- The Ferris wheel in paradise park -/
def paradiseParkFerrisWheel : FerrisWheel := {
  smallSeats := 3
  largeSeats := 0  -- We don't know this value yet
  smallSeatCapacity := 16
  largeSeatCapacity := 12
  totalLargeSeatCapacity := 84
}

/-- Theorem: The number of large seats on the paradise park Ferris wheel is 7 -/
theorem paradise_park_large_seats : 
  paradiseParkFerrisWheel.largeSeats = 
    paradiseParkFerrisWheel.totalLargeSeatCapacity / paradiseParkFerrisWheel.largeSeatCapacity := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_large_seats_l3446_344674


namespace NUMINAMATH_CALUDE_chicken_eggs_today_l3446_344632

theorem chicken_eggs_today (eggs_yesterday : ℕ) (eggs_difference : ℕ) : 
  eggs_yesterday = 10 → eggs_difference = 59 → eggs_yesterday + eggs_difference = 69 :=
by sorry

end NUMINAMATH_CALUDE_chicken_eggs_today_l3446_344632


namespace NUMINAMATH_CALUDE_current_speed_l3446_344633

theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 15)
  (h2 : speed_against_current = 9.4) :
  ∃ (man_speed current_speed : ℝ),
    speed_with_current = man_speed + current_speed ∧
    speed_against_current = man_speed - current_speed ∧
    current_speed = 2.8 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l3446_344633


namespace NUMINAMATH_CALUDE_triangle_abc_is_right_angled_l3446_344664

theorem triangle_abc_is_right_angled (A B C : ℝ) (h1 : A = 60) (h2 : B = 3 * C) 
  (h3 : A + B + C = 180) : B = 90 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_is_right_angled_l3446_344664


namespace NUMINAMATH_CALUDE_slope_of_line_l3446_344616

theorem slope_of_line (x y : ℝ) : 4 * x + 7 * y = 28 → (y - 4) / x = -4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l3446_344616


namespace NUMINAMATH_CALUDE_largest_three_digit_geometric_sequence_l3446_344696

def is_geometric_sequence (a b c : ℕ) : Prop :=
  ∃ r : ℚ, b = a * r ∧ c = b * r

def digits_are_distinct (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

theorem largest_three_digit_geometric_sequence :
  ∀ n : ℕ,
    100 ≤ n ∧ n < 1000 ∧
    (n / 100 = 8) ∧
    digits_are_distinct n ∧
    is_geometric_sequence (n / 100) ((n / 10) % 10) (n % 10) →
    n ≤ 842 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_geometric_sequence_l3446_344696


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3446_344665

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_def : ∀ n, S n = n * (a 1) + (n * (n - 1) / 2) * (a 2 - a 1)
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Theorem: If S_10 = S_20 in an arithmetic sequence, then S_30 = 0 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) (h : seq.S 10 = seq.S 20) :
  seq.S 30 = 0 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3446_344665


namespace NUMINAMATH_CALUDE_antonov_remaining_packs_l3446_344642

/-- Calculates the number of remaining candy packs given the initial number of candies,
    the number of candies per pack, and the number of packs given away. -/
def remaining_packs (initial_candies : ℕ) (candies_per_pack : ℕ) (packs_given : ℕ) : ℕ :=
  (initial_candies - packs_given * candies_per_pack) / candies_per_pack

/-- Proves that Antonov has 2 packs of candy remaining. -/
theorem antonov_remaining_packs :
  remaining_packs 60 20 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_antonov_remaining_packs_l3446_344642


namespace NUMINAMATH_CALUDE_no_line_exists_l3446_344651

-- Define the points and curve
def A : ℝ × ℝ := (8, 0)
def Q : ℝ × ℝ := (-1, 0)
def trajectory (x y : ℝ) : Prop := y^2 = -4*x

-- Define the line passing through A
def line_through_A (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 8)

-- Define the dot product of vectors QM and QN
def dot_product_QM_QN (M N : ℝ × ℝ) : ℝ :=
  (M.1 + 1) * (N.1 + 1) + M.2 * N.2

-- Theorem statement
theorem no_line_exists : ¬ ∃ (k : ℝ) (M N : ℝ × ℝ),
  M ≠ N ∧
  trajectory M.1 M.2 ∧
  trajectory N.1 N.2 ∧
  line_through_A k M.1 M.2 ∧
  line_through_A k N.1 N.2 ∧
  dot_product_QM_QN M N = 97 :=
by sorry

end NUMINAMATH_CALUDE_no_line_exists_l3446_344651


namespace NUMINAMATH_CALUDE_intersection_complement_problem_l3446_344624

open Set

theorem intersection_complement_problem :
  let U : Set ℝ := Set.univ
  let A : Set ℝ := {x | x > 0}
  let B : Set ℝ := {x | x > 1}
  A ∩ (U \ B) = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_problem_l3446_344624


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_47_l3446_344693

theorem gcd_of_powers_of_47 : Nat.gcd (47^5 + 1) (47^5 + 47^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_47_l3446_344693


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l3446_344671

theorem quadratic_root_implies_m (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 + 3 * x - 5 * m + 4 = 0 → x = 2 ∨ x ≠ 2) →
  ((m - 1) * 2^2 + 3 * 2 - 5 * m + 4 = 0) →
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l3446_344671


namespace NUMINAMATH_CALUDE_missy_watch_time_l3446_344659

/-- The total time Missy spends watching TV -/
def total_watch_time (num_reality_shows : ℕ) (reality_show_duration : ℕ) (num_cartoons : ℕ) (cartoon_duration : ℕ) : ℕ :=
  num_reality_shows * reality_show_duration + num_cartoons * cartoon_duration

/-- Theorem stating that Missy spends 150 minutes watching TV -/
theorem missy_watch_time :
  total_watch_time 5 28 1 10 = 150 := by
  sorry

end NUMINAMATH_CALUDE_missy_watch_time_l3446_344659


namespace NUMINAMATH_CALUDE_no_two_right_angles_l3446_344660

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_angles : a + b + c = 180

-- Define a right angle
def is_right_angle (angle : ℝ) : Prop := angle = 90

-- Theorem statement
theorem no_two_right_angles (t : Triangle) : 
  ¬(is_right_angle t.a ∧ is_right_angle t.b) ∧ 
  ¬(is_right_angle t.b ∧ is_right_angle t.c) ∧ 
  ¬(is_right_angle t.c ∧ is_right_angle t.a) :=
sorry

end NUMINAMATH_CALUDE_no_two_right_angles_l3446_344660


namespace NUMINAMATH_CALUDE_remainder_of_2745_base12_div_5_l3446_344656

/-- Converts a base 12 number to base 10 --/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

/-- The base 12 representation of 2745 --/
def number_base12 : List Nat := [2, 7, 4, 5]

theorem remainder_of_2745_base12_div_5 :
  (base12ToBase10 number_base12) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2745_base12_div_5_l3446_344656


namespace NUMINAMATH_CALUDE_free_younger_son_time_l3446_344685

/-- The time required to cut all strands of duct tape -/
def cut_time (total_strands : ℕ) (hannah_rate : ℕ) (son_rate : ℕ) : ℚ :=
  total_strands / (hannah_rate + son_rate)

/-- Theorem stating that it takes 2 minutes to cut 22 strands of duct tape -/
theorem free_younger_son_time :
  cut_time 22 8 3 = 2 := by sorry

end NUMINAMATH_CALUDE_free_younger_son_time_l3446_344685


namespace NUMINAMATH_CALUDE_square_plus_four_equals_54_l3446_344657

theorem square_plus_four_equals_54 (x : ℝ) (h : x = 5) : 2 * x^2 + 4 = 54 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_four_equals_54_l3446_344657


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3446_344698

/-- Proves that in a group of 9 players, if each player plays every other player
    the same number of times, and a total of 36 games are played, then each
    player must play every other player exactly once. -/
theorem chess_tournament_games (n : ℕ) (total_games : ℕ) 
    (h1 : n = 9)
    (h2 : total_games = 36)
    (h3 : ∀ i j : Fin n, i ≠ j → ∃ k : ℕ, k > 0) :
  ∀ i j : Fin n, i ≠ j → ∃ k : ℕ, k = 1 := by
  sorry

#check chess_tournament_games

end NUMINAMATH_CALUDE_chess_tournament_games_l3446_344698


namespace NUMINAMATH_CALUDE_slope_determines_m_l3446_344600

/-- Given two points A(-2, m) and B(m, 4), if the slope of line AB is -2, then m = -8 -/
theorem slope_determines_m (m : ℝ) : 
  let A : ℝ × ℝ := (-2, m)
  let B : ℝ × ℝ := (m, 4)
  (4 - m) / (m - (-2)) = -2 → m = -8 := by
sorry

end NUMINAMATH_CALUDE_slope_determines_m_l3446_344600


namespace NUMINAMATH_CALUDE_min_xy_value_least_xy_value_l3446_344687

theorem min_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 6) :
  ∀ (a b : ℕ+), ((1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 6) → (x : ℕ) * y ≤ (a : ℕ) * b :=
by
  sorry

theorem least_xy_value :
  ∃ (x y : ℕ+), ((1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 6) ∧ (x : ℕ) * y = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_min_xy_value_least_xy_value_l3446_344687


namespace NUMINAMATH_CALUDE_total_snake_owners_is_75_l3446_344630

/-- The number of people in the neighborhood who own pets -/
def total_population : ℕ := 200

/-- The number of people who own only dogs -/
def only_dogs : ℕ := 30

/-- The number of people who own only cats -/
def only_cats : ℕ := 25

/-- The number of people who own only birds -/
def only_birds : ℕ := 10

/-- The number of people who own only snakes -/
def only_snakes : ℕ := 7

/-- The number of people who own only fish -/
def only_fish : ℕ := 12

/-- The number of people who own both cats and dogs -/
def cats_and_dogs : ℕ := 15

/-- The number of people who own both birds and dogs -/
def birds_and_dogs : ℕ := 12

/-- The number of people who own both birds and cats -/
def birds_and_cats : ℕ := 8

/-- The number of people who own both snakes and dogs -/
def snakes_and_dogs : ℕ := 3

/-- The number of people who own both snakes and cats -/
def snakes_and_cats : ℕ := 4

/-- The number of people who own both snakes and birds -/
def snakes_and_birds : ℕ := 2

/-- The number of people who own both fish and dogs -/
def fish_and_dogs : ℕ := 9

/-- The number of people who own both fish and cats -/
def fish_and_cats : ℕ := 6

/-- The number of people who own both fish and birds -/
def fish_and_birds : ℕ := 14

/-- The number of people who own both fish and snakes -/
def fish_and_snakes : ℕ := 11

/-- The number of people who own cats, dogs, and snakes -/
def cats_dogs_snakes : ℕ := 5

/-- The number of people who own cats, dogs, and birds -/
def cats_dogs_birds : ℕ := 4

/-- The number of people who own cats, birds, and snakes -/
def cats_birds_snakes : ℕ := 6

/-- The number of people who own dogs, birds, and snakes -/
def dogs_birds_snakes : ℕ := 9

/-- The number of people who own cats, fish, and dogs -/
def cats_fish_dogs : ℕ := 7

/-- The number of people who own birds, fish, and dogs -/
def birds_fish_dogs : ℕ := 5

/-- The number of people who own birds, fish, and cats -/
def birds_fish_cats : ℕ := 3

/-- The number of people who own snakes, fish, and dogs -/
def snakes_fish_dogs : ℕ := 8

/-- The number of people who own snakes, fish, and cats -/
def snakes_fish_cats : ℕ := 4

/-- The number of people who own snakes, fish, and birds -/
def snakes_fish_birds : ℕ := 6

/-- The number of people who own all five pets -/
def all_five_pets : ℕ := 10

/-- The total number of snake owners in the neighborhood -/
def total_snake_owners : ℕ := 
  only_snakes + snakes_and_dogs + snakes_and_cats + snakes_and_birds + 
  fish_and_snakes + cats_dogs_snakes + cats_birds_snakes + dogs_birds_snakes + 
  snakes_fish_dogs + snakes_fish_cats + snakes_fish_birds + all_five_pets

theorem total_snake_owners_is_75 : total_snake_owners = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_snake_owners_is_75_l3446_344630


namespace NUMINAMATH_CALUDE_walking_distance_l3446_344670

theorem walking_distance (original_speed original_distance increased_speed additional_distance : ℝ) 
  (h1 : original_speed = 4)
  (h2 : increased_speed = 5)
  (h3 : additional_distance = 6)
  (h4 : original_distance / original_speed = (original_distance + additional_distance) / increased_speed) :
  original_distance = 24 := by
sorry

end NUMINAMATH_CALUDE_walking_distance_l3446_344670


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3446_344658

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricToOrigin (p q : Point2D) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

theorem symmetric_point_coordinates :
  let M : Point2D := ⟨1, -2⟩
  let N : Point2D := ⟨-1, 2⟩
  symmetricToOrigin M N → N = ⟨-1, 2⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3446_344658


namespace NUMINAMATH_CALUDE_average_weight_B_and_C_l3446_344619

theorem average_weight_B_and_C (A B C : ℝ) : 
  (A + B + C) / 3 = 45 →
  (A + B) / 2 = 40 →
  B = 31 →
  (B + C) / 2 = 43 := by
sorry

end NUMINAMATH_CALUDE_average_weight_B_and_C_l3446_344619


namespace NUMINAMATH_CALUDE_paige_finished_problems_l3446_344672

/-- Given that Paige had 43 math problems, 12 science problems, and 11 problems left to do for homework,
    prove that she finished 44 problems at school. -/
theorem paige_finished_problems (math_problems : ℕ) (science_problems : ℕ) (problems_left : ℕ)
  (h1 : math_problems = 43)
  (h2 : science_problems = 12)
  (h3 : problems_left = 11) :
  math_problems + science_problems - problems_left = 44 := by
  sorry

end NUMINAMATH_CALUDE_paige_finished_problems_l3446_344672


namespace NUMINAMATH_CALUDE_base8_addition_example_l3446_344607

/-- Addition in base 8 -/
def base8_add (a b : ℕ) : ℕ := sorry

/-- Conversion from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 8 -/
def base10_to_base8 (n : ℕ) : ℕ := sorry

theorem base8_addition_example : 
  base8_add (base10_to_base8 83) (base10_to_base8 46) = base10_to_base8 130 := by sorry

end NUMINAMATH_CALUDE_base8_addition_example_l3446_344607


namespace NUMINAMATH_CALUDE_log_equation_solution_l3446_344645

theorem log_equation_solution (a : ℕ) : 
  (10 - 2*a > 0) ∧ (a - 2 > 0) ∧ (a - 2 ≠ 1) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3446_344645


namespace NUMINAMATH_CALUDE_problem_statement_l3446_344695

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 3) :
  x * y ≤ 9/8 ∧ 
  4^x + 2^y ≥ 4 * Real.sqrt 2 ∧ 
  x / y + 1 / x ≥ 2/3 + 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3446_344695


namespace NUMINAMATH_CALUDE_sin_2x_equals_plus_minus_one_l3446_344694

/-- Given vectors a and b, if a is a non-zero scalar multiple of b, then sin(2x) = ±1 -/
theorem sin_2x_equals_plus_minus_one (x : ℝ) :
  let a : ℝ × ℝ := (Real.cos x, -Real.sin x)
  let b : ℝ × ℝ := (-Real.cos (π/2 - x), Real.cos x)
  ∀ t : ℝ, t ≠ 0 → a = t • b → Real.sin (2*x) = 1 ∨ Real.sin (2*x) = -1 :=
by sorry

end NUMINAMATH_CALUDE_sin_2x_equals_plus_minus_one_l3446_344694


namespace NUMINAMATH_CALUDE_percentage_commutation_l3446_344680

theorem percentage_commutation (x : ℝ) (h : 0.30 * 0.15 * x = 18) :
  0.15 * 0.30 * x = 18 := by
  sorry

end NUMINAMATH_CALUDE_percentage_commutation_l3446_344680


namespace NUMINAMATH_CALUDE_total_votes_is_129_l3446_344684

/-- The number of votes for each cake type in a baking contest. -/
structure CakeVotes where
  witch : ℕ
  unicorn : ℕ
  dragon : ℕ
  mermaid : ℕ
  fairy : ℕ
  phoenix : ℕ

/-- The conditions for the cake voting contest. -/
def contestConditions (votes : CakeVotes) : Prop :=
  votes.witch = 15 ∧
  votes.unicorn = 3 * votes.witch ∧
  votes.dragon = votes.witch + 7 ∧
  votes.dragon = (votes.mermaid * 5) / 4 ∧
  votes.mermaid = votes.dragon - 3 ∧
  votes.mermaid = 2 * votes.fairy ∧
  votes.fairy = votes.witch - 5 ∧
  votes.phoenix = votes.dragon - (votes.dragon / 5) ∧
  votes.phoenix = votes.fairy + 15

/-- The theorem stating that given the contest conditions, the total number of votes is 129. -/
theorem total_votes_is_129 (votes : CakeVotes) :
  contestConditions votes → votes.witch + votes.unicorn + votes.dragon + votes.mermaid + votes.fairy + votes.phoenix = 129 := by
  sorry


end NUMINAMATH_CALUDE_total_votes_is_129_l3446_344684


namespace NUMINAMATH_CALUDE_expenditure_problem_l3446_344652

theorem expenditure_problem (first_avg : ℝ) (second_avg : ℝ) (total_avg : ℝ) 
  (second_days : ℕ) (h1 : first_avg = 350) (h2 : second_avg = 420) 
  (h3 : total_avg = 390) (h4 : second_days = 4) : 
  ∃ (first_days : ℕ), first_days + second_days = 7 ∧ 
  (first_avg * first_days + second_avg * second_days) / (first_days + second_days) = total_avg :=
by
  sorry

#check expenditure_problem

end NUMINAMATH_CALUDE_expenditure_problem_l3446_344652


namespace NUMINAMATH_CALUDE_k_value_theorem_l3446_344673

theorem k_value_theorem (x y z k : ℝ) 
  (h1 : (y + z) / x = k)
  (h2 : (z + x) / y = k)
  (h3 : (x + y) / z = k)
  (h4 : x ≠ 0)
  (h5 : y ≠ 0)
  (h6 : z ≠ 0) :
  k = 2 ∨ k = -1 := by
sorry

end NUMINAMATH_CALUDE_k_value_theorem_l3446_344673


namespace NUMINAMATH_CALUDE_binomial_8_3_l3446_344663

theorem binomial_8_3 : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_3_l3446_344663


namespace NUMINAMATH_CALUDE_martha_ellen_age_ratio_l3446_344604

/-- The ratio of Martha's age to Ellen's age in six years -/
def age_ratio (martha_current_age ellen_current_age : ℕ) : ℚ :=
  (martha_current_age + 6) / (ellen_current_age + 6)

/-- Theorem stating the ratio of Martha's age to Ellen's age in six years -/
theorem martha_ellen_age_ratio :
  age_ratio 32 10 = 19 / 8 := by
  sorry

end NUMINAMATH_CALUDE_martha_ellen_age_ratio_l3446_344604


namespace NUMINAMATH_CALUDE_slope_60_degrees_l3446_344636

/-- The slope of a line with an angle of inclination of 60° is equal to √3 -/
theorem slope_60_degrees :
  let angle_of_inclination : ℝ := 60 * π / 180
  let slope : ℝ := Real.tan angle_of_inclination
  slope = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_60_degrees_l3446_344636


namespace NUMINAMATH_CALUDE_ellipse_properties_l3446_344602

-- Define the ellipse (E)
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 16 * x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 2

-- Define the focus of a parabola
def parabola_focus (x y : ℝ) : Prop :=
  parabola x y ∧ y = 0

-- Define a point on the ellipse
def point_on_ellipse (a b x y : ℝ) : Prop :=
  ellipse a b x y

-- Define a point on the major axis
def point_on_major_axis (m : ℝ) : Prop :=
  ∃ a b : ℝ, ellipse a b m 0

-- State the theorem
theorem ellipse_properties :
  ∃ a b : ℝ,
    (∀ x y : ℝ, ellipse a b x y →
      (∃ xf yf : ℝ, parabola_focus xf yf ∧ ellipse a b xf yf) ∧
      (∀ xh yh : ℝ, hyperbola xh yh → 
        ∃ c : ℝ, c^2 = a^2 - b^2 ∧ c^2 = xh^2 - yh^2 / 2)) →
    (a^2 = 16 ∧ b^2 = 12) ∧
    (∀ m : ℝ, point_on_major_axis m →
      (∀ x y : ℝ, point_on_ellipse a b x y →
        (x = 4 → (∀ x' y' : ℝ, point_on_ellipse a b x' y' →
          (x' - m)^2 + y'^2 ≥ (x - m)^2 + y^2))) →
      1 ≤ m ∧ m ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3446_344602


namespace NUMINAMATH_CALUDE_intersection_condition_union_condition_l3446_344661

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a-1)*x + (a^2-5) = 0}

-- Theorem for part (1)
theorem intersection_condition (a : ℝ) : A ∩ B a = {2} → a = -5 ∨ a = 1 := by sorry

-- Theorem for part (2)
theorem union_condition (a : ℝ) : A ∪ B a = A → a > 3 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_union_condition_l3446_344661


namespace NUMINAMATH_CALUDE_octal_calculation_l3446_344611

/-- Represents a number in base 8 --/
def OctalNumber := Nat

/-- Convert a decimal number to its octal representation --/
def toOctal (n : Nat) : OctalNumber :=
  sorry

/-- Add two octal numbers --/
def octalAdd (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Subtract two octal numbers --/
def octalSub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Theorem: 72₈ - 45₈ + 23₈ = 50₈ in base 8 --/
theorem octal_calculation :
  octalAdd (octalSub (toOctal 72) (toOctal 45)) (toOctal 23) = toOctal 50 := by
  sorry

end NUMINAMATH_CALUDE_octal_calculation_l3446_344611


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_b9_l3446_344666

def fibonacci_like_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → b (n + 2) = b (n + 1) + b n

theorem fibonacci_like_sequence_b9 (b : ℕ → ℕ) :
  fibonacci_like_sequence b →
  (∀ n m : ℕ, n < m → b n < b m) →
  b 8 = 100 →
  b 9 = 194 := by sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_b9_l3446_344666


namespace NUMINAMATH_CALUDE_circle_radius_problem_l3446_344635

/-- Given a circle with radius r and a point M at distance √7 from the center,
    if a secant from M intersects the circle such that the internal part
    of the secant is r and the external part is 2r, then r = 1. -/
theorem circle_radius_problem (r : ℝ) : 
  r > 0 →  -- r is positive (implicit condition for a circle's radius)
  (∃ (M : ℝ × ℝ) (C : ℝ × ℝ), 
    Real.sqrt ((M.1 - C.1)^2 + (M.2 - C.2)^2) = Real.sqrt 7 ∧  -- Distance from M to center is √7
    (∃ (A B : ℝ × ℝ),
      (A.1 - C.1)^2 + (A.2 - C.2)^2 = r^2 ∧  -- A is on the circle
      (B.1 - C.1)^2 + (B.2 - C.2)^2 = r^2 ∧  -- B is on the circle
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = r ∧  -- Internal part of secant is r
      Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) = 2*r  -- External part of secant is 2r
    )
  ) →
  r = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l3446_344635


namespace NUMINAMATH_CALUDE_amy_required_school_hours_per_week_l3446_344629

/-- Amy's work schedule and earnings --/
structure WorkSchedule where
  summer_weeks : ℕ
  summer_hours_per_week : ℕ
  summer_earnings : ℕ
  school_weeks : ℕ
  school_target_earnings : ℕ

/-- Calculate the required hours per week during school --/
def required_school_hours_per_week (schedule : WorkSchedule) : ℚ :=
  let hourly_rate : ℚ := schedule.summer_earnings / (schedule.summer_weeks * schedule.summer_hours_per_week)
  let total_school_hours : ℚ := schedule.school_target_earnings / hourly_rate
  total_school_hours / schedule.school_weeks

/-- Amy's specific work schedule --/
def amy_schedule : WorkSchedule := {
  summer_weeks := 8
  summer_hours_per_week := 40
  summer_earnings := 3200
  school_weeks := 32
  school_target_earnings := 4000
}

theorem amy_required_school_hours_per_week :
  required_school_hours_per_week amy_schedule = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_amy_required_school_hours_per_week_l3446_344629


namespace NUMINAMATH_CALUDE_necessary_condition_for_greater_than_five_l3446_344699

theorem necessary_condition_for_greater_than_five (x : ℝ) :
  x > 5 → x > 3 := by sorry

end NUMINAMATH_CALUDE_necessary_condition_for_greater_than_five_l3446_344699


namespace NUMINAMATH_CALUDE_concyclicity_equivalence_l3446_344605

-- Define the types for points and complex numbers
variable (P A B C D E F G H O₁ O₂ O₃ O₄ : ℂ)

-- Define the properties of the quadrilateral
def is_convex_quadrilateral (A B C D : ℂ) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect (A B C D P : ℂ) : Prop := sorry

-- Define midpoints
def is_midpoint (M A B : ℂ) : Prop := M = (A + B) / 2

-- Define circumcenter
def is_circumcenter (O P Q R : ℂ) : Prop := sorry

-- Define concyclicity
def are_concyclic (A B C D : ℂ) : Prop := sorry

-- State the theorem
theorem concyclicity_equivalence :
  is_convex_quadrilateral A B C D →
  diagonals_intersect A B C D P →
  is_midpoint E A B →
  is_midpoint F B C →
  is_midpoint G C D →
  is_midpoint H D A →
  is_circumcenter O₁ P H E →
  is_circumcenter O₂ P E F →
  is_circumcenter O₃ P F G →
  is_circumcenter O₄ P G H →
  (are_concyclic O₁ O₂ O₃ O₄ ↔ are_concyclic A B C D) :=
by sorry

end NUMINAMATH_CALUDE_concyclicity_equivalence_l3446_344605


namespace NUMINAMATH_CALUDE_OPRQ_shapes_l3446_344640

-- Define the points
def O : ℝ × ℝ := (0, 0)
def P (x₁ y₁ : ℝ) : ℝ × ℝ := (x₁, y₁)
def Q (x₂ y₂ : ℝ) : ℝ × ℝ := (x₂, y₂)
def R (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ := (x₁ - x₂, y₁ - y₂)

-- Define the quadrilateral OPRQ
def OPRQ (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {O, P x₁ y₁, Q x₂ y₂, R x₁ y₁ x₂ y₂}

-- Define conditions for parallelogram, straight line, and trapezoid
def isParallelogram (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  P x₁ y₁ + Q x₂ y₂ = R x₁ y₁ x₂ y₂

def isStraightLine (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * y₂ = x₂ * y₁

def isTrapezoid (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ (k : ℝ), x₂ = k * (x₁ - x₂) ∧ y₂ = k * (y₁ - y₂)

-- Theorem statement
theorem OPRQ_shapes (x₁ y₁ x₂ y₂ : ℝ) (h : P x₁ y₁ ≠ Q x₂ y₂) :
  (isParallelogram x₁ y₁ x₂ y₂) ∧
  (isStraightLine x₁ y₁ x₂ y₂ → OPRQ x₁ y₁ x₂ y₂ = {O, P x₁ y₁, Q x₂ y₂, R x₁ y₁ x₂ y₂}) ∧
  (∃ x₁' y₁' x₂' y₂', isTrapezoid x₁' y₁' x₂' y₂') :=
sorry

end NUMINAMATH_CALUDE_OPRQ_shapes_l3446_344640


namespace NUMINAMATH_CALUDE_necklaces_given_to_friends_l3446_344649

theorem necklaces_given_to_friends (initial : ℕ) (sold : ℕ) (remaining : ℕ) :
  initial = 60 →
  sold = 16 →
  remaining = 26 →
  initial - sold - remaining = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_necklaces_given_to_friends_l3446_344649


namespace NUMINAMATH_CALUDE_equation_roots_l3446_344692

theorem equation_roots (m : ℝ) :
  ((m - 2) ≠ 0) →  -- Condition for linear equation
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁ * (x₁ + 2*m) + m * (1 - x₁) - 1 = 0 ∧ 
    x₂ * (x₂ + 2*m) + m * (1 - x₂) - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l3446_344692


namespace NUMINAMATH_CALUDE_candy_bar_cost_l3446_344634

theorem candy_bar_cost (soft_drink_cost candy_bar_count total_spent : ℕ) :
  soft_drink_cost = 2 →
  candy_bar_count = 5 →
  total_spent = 27 →
  ∃ (candy_bar_cost : ℕ), candy_bar_cost * candy_bar_count + soft_drink_cost = total_spent ∧ candy_bar_cost = 5 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l3446_344634


namespace NUMINAMATH_CALUDE_inequality_proof_l3446_344676

theorem inequality_proof (x₁ x₂ x₃ y₁ y₂ y₃ z₁ z₂ z₃ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hx₃ : x₃ > 0)
  (hy₁ : y₁ > 0) (hy₂ : y₂ > 0) (hy₃ : y₃ > 0)
  (hz₁ : z₁ > 0) (hz₂ : z₂ > 0) (hz₃ : z₃ > 0) :
  (x₁^3 + x₂^3 + x₃^3 + 1) * (y₁^3 + y₂^3 + y₃^3 + 1) * (z₁^3 + z₂^3 + z₃^3 + 1) ≥ 
  (9/2) * (x₁ + y₁ + z₁) * (x₂ + y₂ + z₂) * (x₃ + y₃ + z₃) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3446_344676


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3446_344691

def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3446_344691
