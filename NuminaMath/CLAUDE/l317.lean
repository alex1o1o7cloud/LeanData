import Mathlib

namespace NUMINAMATH_CALUDE_function_minimum_implies_a_less_than_one_l317_31742

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a

-- State the theorem
theorem function_minimum_implies_a_less_than_one :
  ∀ a : ℝ, (∃ m : ℝ, ∀ x < 1, f a x ≥ f a m) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_implies_a_less_than_one_l317_31742


namespace NUMINAMATH_CALUDE_problem_statement_l317_31758

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 3) :
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2 * y₀ = 3 ∧ y₀ / x₀ + 3 / y₀ = 4 ∧ 
    ∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' + 2 * y' = 3 → y' / x' + 3 / y' ≥ 4) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2 * y₀ = 3 ∧ x₀ * y₀ = 9 / 8 ∧ 
    ∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' + 2 * y' = 3 → x' * y' ≤ 9 / 8) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2 * y₀ = 3 ∧ x₀^2 + 4 * y₀^2 = 9 / 2 ∧ 
    ∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' + 2 * y' = 3 → x'^2 + 4 * y'^2 ≥ 9 / 2) ∧
  ¬(∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' + 2 * y' = 3 → Real.sqrt x' + Real.sqrt (2 * y') ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l317_31758


namespace NUMINAMATH_CALUDE_minimal_points_double_star_l317_31763

/-- Represents a regular n-pointed double star polygon -/
structure DoubleStarPolygon where
  n : ℕ
  angleA : ℝ
  angleB : ℝ

/-- Conditions for a valid double star polygon -/
def isValidDoubleStarPolygon (d : DoubleStarPolygon) : Prop :=
  d.n > 0 ∧
  d.angleA > 0 ∧
  d.angleB > 0 ∧
  d.angleA = d.angleB + 15 ∧
  d.n * 15 = 360

theorem minimal_points_double_star :
  ∀ d : DoubleStarPolygon, isValidDoubleStarPolygon d → d.n ≥ 24 :=
by sorry

end NUMINAMATH_CALUDE_minimal_points_double_star_l317_31763


namespace NUMINAMATH_CALUDE_brick_volume_l317_31744

/-- The volume of a rectangular prism -/
def volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a brick with dimensions 9 cm × 4 cm × 7 cm is 252 cubic centimeters -/
theorem brick_volume :
  volume 4 9 7 = 252 := by
  sorry

end NUMINAMATH_CALUDE_brick_volume_l317_31744


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l317_31773

/-- Sum of first n terms of an arithmetic sequence -/
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := sorry

/-- The sequence a is arithmetic -/
def is_arithmetic (a : ℕ → ℝ) : Prop := sorry

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (h : is_arithmetic a) 
  (h1 : S 3 a / S 6 a = 1 / 3) : S 9 a / S 6 a = 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l317_31773


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l317_31713

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (6 * x₁^2 - 9 * x₁ + 5 = 0) → 
  (6 * x₂^2 - 9 * x₂ + 5 = 0) → 
  x₁^2 + x₂^2 = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l317_31713


namespace NUMINAMATH_CALUDE_meteorological_forecast_probability_l317_31743

theorem meteorological_forecast_probability 
  (p q : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) 
  (hq : 0 ≤ q ∧ q ≤ 1) : 
  (p * (1 - q) : ℝ) = 
  (p : ℝ) * (1 - (q : ℝ)) := by
sorry

end NUMINAMATH_CALUDE_meteorological_forecast_probability_l317_31743


namespace NUMINAMATH_CALUDE_loes_speed_l317_31722

/-- Proves that Loe's speed is 50 mph given the conditions of the problem -/
theorem loes_speed (teena_speed : ℝ) (initial_distance : ℝ) (time : ℝ) (final_distance : ℝ) :
  teena_speed = 55 →
  initial_distance = 7.5 →
  time = 1.5 →
  final_distance = 15 →
  ∃ (loe_speed : ℝ), loe_speed = 50 ∧
    teena_speed * time - loe_speed * time = final_distance + initial_distance :=
by sorry

end NUMINAMATH_CALUDE_loes_speed_l317_31722


namespace NUMINAMATH_CALUDE_max_lines_theorem_l317_31781

/-- Given n points on a plane where no three are collinear, 
    this function returns the maximum number of lines that can be drawn 
    through pairs of points without forming a triangle with vertices 
    among the given points. -/
def max_lines_without_triangle (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n^2 / 4
  else
    (n^2 - 1) / 4

/-- Theorem stating the maximum number of lines that can be drawn 
    through pairs of points without forming a triangle, 
    given n points on a plane where no three are collinear and n ≥ 3. -/
theorem max_lines_theorem (n : ℕ) (h : n ≥ 3) :
  max_lines_without_triangle n = 
    if n % 2 = 0 then
      n^2 / 4
    else
      (n^2 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_lines_theorem_l317_31781


namespace NUMINAMATH_CALUDE_sqrt_real_condition_l317_31721

theorem sqrt_real_condition (x : ℝ) : (∃ y : ℝ, y ^ 2 = (x - 1) / 9) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_real_condition_l317_31721


namespace NUMINAMATH_CALUDE_log_simplification_l317_31749

-- Define variables
variable (p q r s t z : ℝ)
variable (h₁ : p > 0)
variable (h₂ : q > 0)
variable (h₃ : r > 0)
variable (h₄ : s > 0)
variable (h₅ : t > 0)
variable (h₆ : z > 0)

-- State the theorem
theorem log_simplification :
  Real.log (p / q) + Real.log (q / r) + Real.log (r / s) - Real.log (p * t / (s * z)) = Real.log (z / t) :=
by sorry

end NUMINAMATH_CALUDE_log_simplification_l317_31749


namespace NUMINAMATH_CALUDE_captain_age_l317_31701

/-- Represents a cricket team with given properties -/
structure CricketTeam where
  size : ℕ
  captainAge : ℕ
  wicketKeeperAge : ℕ
  teamAverageAge : ℝ
  remainingPlayersAverageAge : ℝ

/-- Theorem stating the captain's age in the given cricket team scenario -/
theorem captain_age (team : CricketTeam) 
  (h1 : team.size = 11)
  (h2 : team.wicketKeeperAge = team.captainAge + 5)
  (h3 : team.teamAverageAge = 23)
  (h4 : team.remainingPlayersAverageAge = team.teamAverageAge - 1)
  : team.captainAge = 25 := by
  sorry

end NUMINAMATH_CALUDE_captain_age_l317_31701


namespace NUMINAMATH_CALUDE_first_agency_less_expensive_l317_31784

/-- The number of miles at which the first agency becomes less expensive than the second -/
def miles_threshold : ℝ := 25

/-- The daily rate for the first agency -/
def daily_rate_1 : ℝ := 20.25

/-- The per-mile rate for the first agency -/
def mile_rate_1 : ℝ := 0.14

/-- The daily rate for the second agency -/
def daily_rate_2 : ℝ := 18.25

/-- The per-mile rate for the second agency -/
def mile_rate_2 : ℝ := 0.22

/-- Theorem stating that the first agency is less expensive when miles driven exceed the threshold -/
theorem first_agency_less_expensive (miles : ℝ) (days : ℝ) 
  (h : miles > miles_threshold) : 
  daily_rate_1 * days + mile_rate_1 * miles < daily_rate_2 * days + mile_rate_2 * miles :=
by
  sorry


end NUMINAMATH_CALUDE_first_agency_less_expensive_l317_31784


namespace NUMINAMATH_CALUDE_photos_sum_equals_total_l317_31753

/-- The total number of photos collected by Tom, Tim, and Paul -/
def total_photos : ℕ := 152

/-- Tom's photos -/
def tom_photos : ℕ := 38

/-- Tim's photos -/
def tim_photos : ℕ := total_photos - 100

/-- Paul's photos -/
def paul_photos : ℕ := tim_photos + 10

/-- Theorem stating that the sum of individual photos equals the total photos -/
theorem photos_sum_equals_total : 
  tom_photos + tim_photos + paul_photos = total_photos := by sorry

end NUMINAMATH_CALUDE_photos_sum_equals_total_l317_31753


namespace NUMINAMATH_CALUDE_teacher_engineer_ratio_l317_31783

theorem teacher_engineer_ratio (t e : ℕ) (t_pos : t > 0) (e_pos : e > 0) :
  (40 * t + 55 * e) / (t + e) = 45 →
  t / e = 2 := by
sorry

end NUMINAMATH_CALUDE_teacher_engineer_ratio_l317_31783


namespace NUMINAMATH_CALUDE_distance_between_points_l317_31706

theorem distance_between_points : 
  let x1 : ℝ := 3
  let y1 : ℝ := 12
  let x2 : ℝ := 10
  let y2 : ℝ := 0
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = Real.sqrt 193 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l317_31706


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l317_31732

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l317_31732


namespace NUMINAMATH_CALUDE_complex_division_pure_imaginary_l317_31738

theorem complex_division_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 3 * Complex.I
  let z₂ : ℂ := 3 - 4 * Complex.I
  (∃ (b : ℝ), z₁ / z₂ = b * Complex.I) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_division_pure_imaginary_l317_31738


namespace NUMINAMATH_CALUDE_inequality_solution_set_l317_31702

theorem inequality_solution_set (x : ℝ) : 
  1 / (x + 2) + 8 / (x + 6) ≥ 1 ↔ -6 < x ∧ x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l317_31702


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_quadratic_equation_with_square_l317_31782

theorem quadratic_equation_solutions :
  (∃ x : ℝ, x^2 - 4*x - 8 = 0) ↔ 
  (∃ x : ℝ, x = 2 + 2*Real.sqrt 3 ∨ x = 2 - 2*Real.sqrt 3) :=
sorry

theorem quadratic_equation_with_square :
  (∃ x : ℝ, (x - 2)^2 = 2*x - 4) ↔ 
  (∃ x : ℝ, x = 2 ∨ x = 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_quadratic_equation_with_square_l317_31782


namespace NUMINAMATH_CALUDE_max_q_plus_r_for_1051_l317_31791

theorem max_q_plus_r_for_1051 :
  ∀ q r : ℕ+,
  1051 = 23 * q + r →
  ∀ q' r' : ℕ+,
  1051 = 23 * q' + r' →
  q + r ≤ 61 :=
by sorry

end NUMINAMATH_CALUDE_max_q_plus_r_for_1051_l317_31791


namespace NUMINAMATH_CALUDE_teacher_pay_per_period_l317_31797

/-- Calculates the pay per period for a teacher given their work schedule and total earnings --/
theorem teacher_pay_per_period 
  (periods_per_day : ℕ)
  (days_per_month : ℕ)
  (months_worked : ℕ)
  (total_earnings : ℕ)
  (h1 : periods_per_day = 5)
  (h2 : days_per_month = 24)
  (h3 : months_worked = 6)
  (h4 : total_earnings = 3600) :
  total_earnings / (periods_per_day * days_per_month * months_worked) = 5 := by
  sorry

#eval 3600 / (5 * 24 * 6)  -- This should output 5

end NUMINAMATH_CALUDE_teacher_pay_per_period_l317_31797


namespace NUMINAMATH_CALUDE_all_squares_similar_l317_31770

/-- A square is a quadrilateral with all sides equal and all angles 90 degrees. -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- Similarity of shapes means they have the same shape but not necessarily the same size. -/
def are_similar (s1 s2 : Square) : Prop :=
  ∃ k : ℝ, k > 0 ∧ s1.side = k * s2.side

/-- Any two squares are similar. -/
theorem all_squares_similar (s1 s2 : Square) : are_similar s1 s2 := by
  sorry

end NUMINAMATH_CALUDE_all_squares_similar_l317_31770


namespace NUMINAMATH_CALUDE_power_six_equivalence_l317_31717

theorem power_six_equivalence (m : ℝ) : m^2 * m^4 = m^6 := by
  sorry

end NUMINAMATH_CALUDE_power_six_equivalence_l317_31717


namespace NUMINAMATH_CALUDE_floor_width_proof_l317_31786

/-- Proves that the width of a rectangular floor is 120 cm given specific conditions --/
theorem floor_width_proof (floor_length tile_length tile_width max_tiles : ℕ) 
  (h1 : floor_length = 180)
  (h2 : tile_length = 25)
  (h3 : tile_width = 16)
  (h4 : max_tiles = 54)
  (h5 : floor_length % tile_width = 0)
  (h6 : floor_length / tile_width * (floor_length / tile_width) ≤ max_tiles) :
  ∃ (floor_width : ℕ), floor_width = 120 ∧ 
    floor_length * floor_width = max_tiles * tile_length * tile_width :=
by sorry

end NUMINAMATH_CALUDE_floor_width_proof_l317_31786


namespace NUMINAMATH_CALUDE_probability_second_shiny_penny_l317_31768

def total_pennies : ℕ := 7
def shiny_pennies : ℕ := 4
def dull_pennies : ℕ := 3

def probability_more_than_three_draws : ℚ :=
  (Nat.choose 3 1 * Nat.choose 4 1 + Nat.choose 3 0 * Nat.choose 4 2) / Nat.choose total_pennies shiny_pennies

theorem probability_second_shiny_penny :
  probability_more_than_three_draws = 18 / 35 := by sorry

end NUMINAMATH_CALUDE_probability_second_shiny_penny_l317_31768


namespace NUMINAMATH_CALUDE_distinct_sides_not_isosceles_l317_31711

-- Define a triangle with sides a, b, and c
structure Triangle (α : Type*) :=
  (a b c : α)

-- Define what it means for a triangle to be isosceles
def is_isosceles {α : Type*} [PartialOrder α] (t : Triangle α) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- Theorem statement
theorem distinct_sides_not_isosceles {α : Type*} [LinearOrder α] 
  (t : Triangle α) (h_distinct : t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.a ≠ t.c) :
  ¬(is_isosceles t) :=
sorry

end NUMINAMATH_CALUDE_distinct_sides_not_isosceles_l317_31711


namespace NUMINAMATH_CALUDE_power_equality_l317_31745

theorem power_equality (x : ℝ) : (1/8 : ℝ) * 2^50 = 4^x → x = 23.5 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l317_31745


namespace NUMINAMATH_CALUDE_rational_equation_solution_l317_31757

theorem rational_equation_solution :
  ∃ x : ℚ, (x + 11) / (x - 4) = (x - 3) / (x + 6) ↔ x = -9/4 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l317_31757


namespace NUMINAMATH_CALUDE_select_shoes_count_l317_31726

/-- The number of ways to select 4 shoes from 4 pairs of different shoes,
    with at least 2 shoes forming a pair -/
def select_shoes : ℕ :=
  Nat.choose 8 4 - 16

theorem select_shoes_count : select_shoes = 54 := by
  sorry

end NUMINAMATH_CALUDE_select_shoes_count_l317_31726


namespace NUMINAMATH_CALUDE_find_number_l317_31792

/-- Given two positive integers with specific LCM and HCF, prove one number given the other -/
theorem find_number (A B : ℕ+) (h1 : Nat.lcm A B = 2310) (h2 : Nat.gcd A B = 30) (h3 : B = 150) :
  A = 462 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l317_31792


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l317_31760

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sine law for triangles -/
axiom sine_law (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- The cosine law for triangles -/
axiom cosine_law (t : Triangle) : Real.cos t.C = (t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b)

theorem triangle_abc_properties (t : Triangle) 
  (ha : t.a = 4)
  (hc : t.c = Real.sqrt 13)
  (hsin : Real.sin t.A = 4 * Real.sin t.B) :
  t.b = 1 ∧ t.C = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l317_31760


namespace NUMINAMATH_CALUDE_lecture_scheduling_l317_31766

theorem lecture_scheduling (n : ℕ) (h : n = 6) :
  (n! / 2 : ℕ) = 360 := by
  sorry

#check lecture_scheduling

end NUMINAMATH_CALUDE_lecture_scheduling_l317_31766


namespace NUMINAMATH_CALUDE_rationalize_cube_root_difference_l317_31740

theorem rationalize_cube_root_difference : ∃ (A B C D : ℕ),
  (((1 : ℝ) / (5^(1/3) - 3^(1/3))) * ((5^(2/3) + 5^(1/3)*3^(1/3) + 3^(2/3)) / (5^(2/3) + 5^(1/3)*3^(1/3) + 3^(2/3)))) = 
  ((A : ℝ)^(1/3) + (B : ℝ)^(1/3) + (C : ℝ)^(1/3)) / (D : ℝ) ∧
  A + B + C + D = 51 := by
sorry

end NUMINAMATH_CALUDE_rationalize_cube_root_difference_l317_31740


namespace NUMINAMATH_CALUDE_car_both_ways_time_l317_31796

/-- Represents the time in hours for different travel scenarios -/
structure TravelTime where
  mixedTrip : ℝ  -- Time for walking one way and taking a car back
  walkingBothWays : ℝ  -- Time for walking both ways
  carBothWays : ℝ  -- Time for taking a car both ways

/-- Proves that given the conditions, the time taken if taking a car both ways is 30 minutes -/
theorem car_both_ways_time (t : TravelTime) 
  (h1 : t.mixedTrip = 1.5)
  (h2 : t.walkingBothWays = 2.5) : 
  t.carBothWays * 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_car_both_ways_time_l317_31796


namespace NUMINAMATH_CALUDE_sum_congruence_l317_31767

theorem sum_congruence (a b c : ℕ) : 
  a < 11 → b < 11 → c < 11 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 11 = 1 → 
  (7 * c) % 11 = 4 → 
  (8 * b) % 11 = (5 + b) % 11 → 
  (a + b + c) % 11 = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_congruence_l317_31767


namespace NUMINAMATH_CALUDE_laura_triathlon_speed_l317_31715

theorem laura_triathlon_speed :
  ∃ x : ℝ, x > 0 ∧ (20 / (2 * x + 1)) + (5 / x) + (5 / 60) = 110 / 60 := by
  sorry

end NUMINAMATH_CALUDE_laura_triathlon_speed_l317_31715


namespace NUMINAMATH_CALUDE_quadratic_root_range_l317_31719

theorem quadratic_root_range (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + a*x + a^2 - 1 = 0 ∧ y^2 + a*y + a^2 - 1 = 0) →
  -1 < a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l317_31719


namespace NUMINAMATH_CALUDE_final_sum_is_130_l317_31712

/-- Represents the financial state of Earl, Fred, and Greg --/
structure FinancialState where
  earl : Int
  fred : Int
  greg : Int

/-- Represents the debts between Earl, Fred, and Greg --/
structure Debts where
  earl_to_fred : Int
  fred_to_greg : Int
  greg_to_earl : Int

/-- Calculates the final amounts for Earl and Greg after settling all debts --/
def settle_debts (initial : FinancialState) (debts : Debts) : Int × Int :=
  let earl_final := initial.earl - debts.earl_to_fred + debts.greg_to_earl
  let greg_final := initial.greg + debts.fred_to_greg - debts.greg_to_earl
  (earl_final, greg_final)

/-- Theorem stating that Greg and Earl will have $130 together after settling all debts --/
theorem final_sum_is_130 (initial : FinancialState) (debts : Debts) :
  initial.earl = 90 →
  initial.fred = 48 →
  initial.greg = 36 →
  debts.earl_to_fred = 28 →
  debts.fred_to_greg = 32 →
  debts.greg_to_earl = 40 →
  let (earl_final, greg_final) := settle_debts initial debts
  earl_final + greg_final = 130 := by
  sorry

#check final_sum_is_130

end NUMINAMATH_CALUDE_final_sum_is_130_l317_31712


namespace NUMINAMATH_CALUDE_christinas_speed_limit_l317_31765

def total_distance : ℝ := 210
def friend_driving_time : ℝ := 3
def friend_speed_limit : ℝ := 40
def christina_driving_time : ℝ := 3  -- 180 minutes converted to hours

theorem christinas_speed_limit :
  ∃ (christina_speed : ℝ),
    christina_speed * christina_driving_time + 
    friend_speed_limit * friend_driving_time = total_distance ∧
    christina_speed = 30 := by
  sorry

end NUMINAMATH_CALUDE_christinas_speed_limit_l317_31765


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_range_l317_31752

theorem absolute_value_inequality_solution_range :
  (∃ (x : ℝ), |x - 5| + |x - 3| < m) → m > 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_range_l317_31752


namespace NUMINAMATH_CALUDE_combined_average_score_l317_31718

theorem combined_average_score (g₁ g₂ : ℕ) (avg₁ avg₂ : ℚ) :
  g₁ > 0 → g₂ > 0 →
  avg₁ = 88 →
  avg₂ = 76 →
  g₁ = (4 * g₂) / 5 →
  let total_score := g₁ * avg₁ + g₂ * avg₂
  let total_students := g₁ + g₂
  (total_score / total_students : ℚ) = 81 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_score_l317_31718


namespace NUMINAMATH_CALUDE_least_sum_with_conditions_l317_31710

theorem least_sum_with_conditions (m n : ℕ+) 
  (h1 : Nat.gcd (m + n) 330 = 1)
  (h2 : ∃ k : ℕ, m ^ m.val = k * n ^ n.val)
  (h3 : ¬ ∃ k : ℕ, m = k * n) :
  (∀ m' n' : ℕ+, 
    Nat.gcd (m' + n') 330 = 1 → 
    (∃ k : ℕ, m' ^ m'.val = k * n' ^ n'.val) → 
    (¬ ∃ k : ℕ, m' = k * n') → 
    m' + n' ≥ m + n) → 
  m + n = 429 := by
sorry

end NUMINAMATH_CALUDE_least_sum_with_conditions_l317_31710


namespace NUMINAMATH_CALUDE_large_glass_cost_l317_31735

def cost_of_large_glass (initial_money : ℕ) (small_glass_cost : ℕ) (num_small_glasses : ℕ) (num_large_glasses : ℕ) (money_left : ℕ) : ℕ :=
  let money_after_small := initial_money - (small_glass_cost * num_small_glasses)
  let total_large_cost := money_after_small - money_left
  total_large_cost / num_large_glasses

theorem large_glass_cost :
  cost_of_large_glass 50 3 8 5 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_large_glass_cost_l317_31735


namespace NUMINAMATH_CALUDE_not_p_and_q_implies_at_most_one_true_l317_31762

theorem not_p_and_q_implies_at_most_one_true (p q : Prop) :
  ¬(p ∧ q) → (¬p ∨ ¬q) :=
by
  sorry

end NUMINAMATH_CALUDE_not_p_and_q_implies_at_most_one_true_l317_31762


namespace NUMINAMATH_CALUDE_max_discount_rate_l317_31736

/-- The maximum discount rate that can be applied without incurring a loss,
    given an initial markup of 25% -/
theorem max_discount_rate : ∀ (m : ℝ) (x : ℝ),
  m > 0 →  -- Assuming positive cost
  (1.25 * m * (1 - x) ≥ m) ↔ (x ≤ 0.2) :=
by sorry

end NUMINAMATH_CALUDE_max_discount_rate_l317_31736


namespace NUMINAMATH_CALUDE_derivative_of_exp_ax_l317_31708

theorem derivative_of_exp_ax (a : ℝ) (x : ℝ) :
  deriv (fun x => Real.exp (a * x)) x = a * Real.exp (a * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_exp_ax_l317_31708


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l317_31703

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 144 - y^2 / 49 = 1

-- Define the distance between vertices
def distance_between_vertices : ℝ := 24

-- Theorem statement
theorem hyperbola_vertices_distance :
  ∀ x y : ℝ, hyperbola_equation x y →
  distance_between_vertices = 24 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l317_31703


namespace NUMINAMATH_CALUDE_least_number_remainder_l317_31754

theorem least_number_remainder : ∃ (r : ℕ), r > 0 ∧ r < 3 ∧ r < 38 ∧ 115 % 38 = r ∧ 115 % 3 = r := by
  sorry

end NUMINAMATH_CALUDE_least_number_remainder_l317_31754


namespace NUMINAMATH_CALUDE_sequence_properties_l317_31704

def sequence_a (n : ℕ+) : ℚ := sorry

def S (n : ℕ+) : ℚ := sorry

def T (n : ℕ+) : ℚ := sorry

theorem sequence_properties :
  (∀ n : ℕ+, 3 * S n = (n + 2) * sequence_a n) ∧
  sequence_a 1 = 2 →
  (∀ n : ℕ+, sequence_a n = n + 1) ∧
  ∃ M : Set ℕ+, Set.Infinite M ∧ ∀ n ∈ M, |T n - 1| < (1 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l317_31704


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_difference_l317_31751

theorem square_difference_of_sum_and_difference (x y : ℝ) 
  (h_sum : x + y = 20) (h_diff : x - y = 8) : x^2 - y^2 = 160 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_difference_l317_31751


namespace NUMINAMATH_CALUDE_lg_sqrt_sum_l317_31793

-- Define lg as the base 10 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_sqrt_sum : lg (Real.sqrt 5) + lg (Real.sqrt 20) = 1 := by
  sorry

end NUMINAMATH_CALUDE_lg_sqrt_sum_l317_31793


namespace NUMINAMATH_CALUDE_correct_average_l317_31714

theorem correct_average (n : ℕ) (initial_avg : ℚ) (wrong_num correct_num : ℕ) :
  n = 10 ∧ initial_avg = 14 ∧ wrong_num = 26 ∧ correct_num = 36 →
  (n * initial_avg + (correct_num - wrong_num)) / n = 15 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l317_31714


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l317_31750

/-- The number of ways to place distinguishable balls into distinguishable boxes -/
def place_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 2187 ways to place 7 distinguishable balls into 3 distinguishable boxes -/
theorem seven_balls_three_boxes : place_balls 7 3 = 2187 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l317_31750


namespace NUMINAMATH_CALUDE_inequality_solution_set_range_l317_31705

/-- The inequality we're working with -/
def inequality (m : ℝ) (x : ℝ) : Prop :=
  m * x^2 - m * x - 1 < 2 * x^2 - 2 * x

/-- The solution set of the inequality with respect to x is R -/
def solution_set_is_R (m : ℝ) : Prop :=
  ∀ x : ℝ, inequality m x

/-- The range of m values for which the solution set is R -/
def m_range : Set ℝ :=
  {m : ℝ | m > -2 ∧ m ≤ 2}

/-- The main theorem to prove -/
theorem inequality_solution_set_range :
  ∀ m : ℝ, solution_set_is_R m ↔ m ∈ m_range :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_range_l317_31705


namespace NUMINAMATH_CALUDE_unique_increasing_matrix_l317_31733

/-- A 4x4 matrix with entries from 1 to 16 in increasing order in rows and columns -/
def IncreasingMatrix : Type :=
  { M : Matrix (Fin 4) (Fin 4) ℕ // 
    (∀ i j, M i j ∈ Finset.range 16) ∧
    (∀ i j₁ j₂, j₁ < j₂ → M i j₁ < M i j₂) ∧
    (∀ i₁ i₂ j, i₁ < i₂ → M i₁ j < M i₂ j) ∧
    (∀ n, n ∈ Finset.range 16 → ∃ i j, M i j = n) }

/-- There is exactly one 4x4 matrix with entries from 1 to 16 in increasing order in rows and columns -/
theorem unique_increasing_matrix : ∃! M : IncreasingMatrix, True :=
sorry

end NUMINAMATH_CALUDE_unique_increasing_matrix_l317_31733


namespace NUMINAMATH_CALUDE_tan_half_sum_right_triangle_angles_l317_31769

theorem tan_half_sum_right_triangle_angles (A B : ℝ) : 
  0 < A → A < π/2 → 0 < B → B < π/2 → A + B = π/2 → 
  Real.tan ((A + B) / 2) = 1 := by
sorry

end NUMINAMATH_CALUDE_tan_half_sum_right_triangle_angles_l317_31769


namespace NUMINAMATH_CALUDE_gcd_g_x_eq_one_l317_31734

def g (x : ℤ) : ℤ := (3*x+4)*(9*x+5)*(17*x+11)*(x+17)

theorem gcd_g_x_eq_one (x : ℤ) (h : ∃ k : ℤ, x = 7263 * k) :
  Int.gcd (g x) x = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_x_eq_one_l317_31734


namespace NUMINAMATH_CALUDE_chord_length_l317_31779

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length (x y : ℝ) : 
  (x - y + 2 = 0) →  -- Line equation
  ((x - 1)^2 + (y - 2)^2 = 4) →  -- Circle equation
  ∃ A B : ℝ × ℝ,  -- Points of intersection
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 14  -- Length of AB squared
  := by sorry

end NUMINAMATH_CALUDE_chord_length_l317_31779


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l317_31787

theorem largest_integer_with_remainder : ∃ n : ℕ, n = 94 ∧ 
  (∀ m : ℕ, m < 100 ∧ m % 6 = 4 → m ≤ n) ∧ 
  n < 100 ∧ 
  n % 6 = 4 :=
sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l317_31787


namespace NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_congruences_l317_31789

theorem smallest_positive_integer_satisfying_congruences :
  ∃ x : ℕ+, 
    (45 * x.val + 15) % 25 = 5 ∧ 
    x.val % 4 = 3 ∧
    ∀ y : ℕ+, 
      ((45 * y.val + 15) % 25 = 5 ∧ y.val % 4 = 3) → 
      x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_congruences_l317_31789


namespace NUMINAMATH_CALUDE_vector_perpendicular_l317_31729

/-- Given plane vectors a, b, and c, where c is perpendicular to (a + b), prove that t = -6/5 -/
theorem vector_perpendicular (a b c : ℝ × ℝ) (t : ℝ) :
  a = (1, 2) →
  b = (3, 4) →
  c = (t, t + 2) →
  (c.1 * (a.1 + b.1) + c.2 * (a.2 + b.2) = 0) →
  t = -6/5 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l317_31729


namespace NUMINAMATH_CALUDE_least_multiple_of_13_greater_than_450_l317_31748

theorem least_multiple_of_13_greater_than_450 :
  (∀ n : ℕ, n * 13 > 450 → n * 13 ≥ 455) ∧ 455 % 13 = 0 ∧ 455 > 450 := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_13_greater_than_450_l317_31748


namespace NUMINAMATH_CALUDE_max_cookies_andy_l317_31723

/-- The number of cookies baked by the siblings -/
def total_cookies : ℕ := 36

/-- Andy's cookies -/
def andy_cookies : ℕ → ℕ := λ x => x

/-- Aaron's cookies -/
def aaron_cookies : ℕ → ℕ := λ x => 2 * x

/-- Alexa's cookies -/
def alexa_cookies : ℕ → ℕ := λ x => total_cookies - x - 2 * x

/-- The maximum number of cookies Andy could have eaten -/
def max_andy_cookies : ℕ := 12

theorem max_cookies_andy :
  ∀ x : ℕ,
  x ≤ max_andy_cookies ∧
  andy_cookies x + aaron_cookies x + alexa_cookies x = total_cookies ∧
  alexa_cookies x ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_max_cookies_andy_l317_31723


namespace NUMINAMATH_CALUDE_students_playing_sport_b_l317_31730

/-- Given that there are 6 students playing sport A, and the number of students
    playing sport B is 4 times the number of students playing sport A,
    prove that 24 students play sport B. -/
theorem students_playing_sport_b (students_a : ℕ) (students_b : ℕ) : 
  students_a = 6 →
  students_b = 4 * students_a →
  students_b = 24 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_sport_b_l317_31730


namespace NUMINAMATH_CALUDE_adjacent_pair_with_distinct_roots_l317_31709

/-- Represents a 6x6 grid containing integers from 1 to 36 --/
def Grid := Fin 6 → Fin 6 → Fin 36

/-- Checks if two numbers are adjacent in a row --/
def areAdjacent (grid : Grid) (i j : Fin 6) (k : Fin 5) : Prop :=
  grid i k = j ∧ grid i (k + 1) = j + 1 ∨ grid i k = j + 1 ∧ grid i (k + 1) = j

/-- Checks if a quadratic equation has two distinct real roots --/
def hasTwoDistinctRealRoots (p q : ℕ) : Prop :=
  p * p > 4 * q

theorem adjacent_pair_with_distinct_roots (grid : Grid) :
  ∃ (i : Fin 6) (j : Fin 36) (k : Fin 5),
    areAdjacent grid j (j + 1) k ∧
    hasTwoDistinctRealRoots j (j + 1) := by
  sorry

end NUMINAMATH_CALUDE_adjacent_pair_with_distinct_roots_l317_31709


namespace NUMINAMATH_CALUDE_closed_mul_l317_31788

structure SpecialSet (S : Set ℝ) : Prop where
  one_mem : (1 : ℝ) ∈ S
  closed_sub : ∀ a b : ℝ, a ∈ S → b ∈ S → (a - b) ∈ S
  closed_inv : ∀ a : ℝ, a ∈ S → a ≠ 0 → (1 / a) ∈ S

theorem closed_mul {S : Set ℝ} (h : SpecialSet S) :
  ∀ a b : ℝ, a ∈ S → b ∈ S → (a * b) ∈ S := by
  sorry

end NUMINAMATH_CALUDE_closed_mul_l317_31788


namespace NUMINAMATH_CALUDE_polynomial_expansion_l317_31747

theorem polynomial_expansion (x : ℝ) : 
  (7 * x + 3) * (5 * x^2 + 4) = 35 * x^3 + 15 * x^2 + 28 * x + 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l317_31747


namespace NUMINAMATH_CALUDE_continued_fraction_result_l317_31741

/-- Given x satisfying the infinite continued fraction equation,
    prove that 1/((x+2)(x-3)) equals (-√3 - 2) / 2 -/
theorem continued_fraction_result (x : ℝ) 
  (hx : x = 2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / x))) :
  1 / ((x + 2) * (x - 3)) = (-Real.sqrt 3 - 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_result_l317_31741


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l317_31756

theorem quadratic_equation_roots (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 3*x + k = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁ * x₂ + 2*x₁ + 2*x₂ = 1) →
  k = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l317_31756


namespace NUMINAMATH_CALUDE_triplet_satisfies_conditions_l317_31746

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- Checks if three numbers form a geometric sequence -/
def isGeometricSequence (a b c : ℕ) : Prop :=
  ∃ r : ℚ, r > 0 ∧ b = a * r ∧ c = b * r

theorem triplet_satisfies_conditions : 
  isPrime 17 ∧ isPrime 23 ∧ isPrime 31 ∧
  17 < 23 ∧ 23 < 31 ∧ 31 < 100 ∧
  isGeometricSequence 18 24 32 :=
by sorry

end NUMINAMATH_CALUDE_triplet_satisfies_conditions_l317_31746


namespace NUMINAMATH_CALUDE_vasyas_numbers_l317_31716

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l317_31716


namespace NUMINAMATH_CALUDE_g_3_6_neg1_eq_one_seventh_l317_31759

/-- The function g as defined in the problem -/
def g (a b c : ℚ) : ℚ := (2 * c + a) / (b - c)

/-- Theorem stating that g(3, 6, -1) = 1/7 -/
theorem g_3_6_neg1_eq_one_seventh : g 3 6 (-1) = 1/7 := by sorry

end NUMINAMATH_CALUDE_g_3_6_neg1_eq_one_seventh_l317_31759


namespace NUMINAMATH_CALUDE_dots_on_line_l317_31794

/-- The number of dots drawn on a line of given length at given intervals, excluding the beginning and end points. -/
def numDots (lineLength : ℕ) (interval : ℕ) : ℕ :=
  if interval = 0 then 0
  else (lineLength - interval) / interval

theorem dots_on_line (lineLength : ℕ) (interval : ℕ) 
  (h1 : lineLength = 30) 
  (h2 : interval = 5) : 
  numDots lineLength interval = 5 := by
  sorry

end NUMINAMATH_CALUDE_dots_on_line_l317_31794


namespace NUMINAMATH_CALUDE_sin_600_degrees_l317_31785

theorem sin_600_degrees : Real.sin (600 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_degrees_l317_31785


namespace NUMINAMATH_CALUDE_triangle_cosine_ratio_l317_31720

/-- In any triangle ABC, (b * cos C + c * cos B) / a = 1 -/
theorem triangle_cosine_ratio (A B C a b c : ℝ) : 
  0 < a → 0 < b → 0 < c →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  (b * Real.cos C + c * Real.cos B) / a = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_ratio_l317_31720


namespace NUMINAMATH_CALUDE_daily_rate_proof_l317_31724

/-- The daily rental rate for Jason's carriage house -/
def daily_rate : ℝ := 40

/-- The total cost for Eric's rental -/
def total_cost : ℝ := 800

/-- The number of days Eric is renting -/
def rental_days : ℕ := 20

/-- Theorem stating that the daily rate multiplied by the number of rental days equals the total cost -/
theorem daily_rate_proof : daily_rate * (rental_days : ℝ) = total_cost := by
  sorry

end NUMINAMATH_CALUDE_daily_rate_proof_l317_31724


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l317_31737

/-- 
Given an arithmetic sequence where the first term is 3^2 and the third term is 3^4,
prove that the second term is 45.
-/
theorem arithmetic_sequence_middle_term : 
  ∀ (a : ℕ → ℤ), 
  (a 0 = 3^2) → 
  (a 2 = 3^4) → 
  (∀ i j k, i < j → j < k → a j - a i = a k - a j) → 
  (a 1 = 45) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l317_31737


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l317_31755

/-- Given two plane vectors a and b, with the angle between them being 60°,
    a = (2,0), and |b| = 1, prove that |a + 2b| = 2√3 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  let angle := Real.pi / 3  -- 60° in radians
  a = (2, 0) ∧ 
  ‖b‖ = 1 ∧ 
  a.1 * b.1 + a.2 * b.2 = ‖a‖ * ‖b‖ * Real.cos angle →
  ‖a + 2 • b‖ = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l317_31755


namespace NUMINAMATH_CALUDE_trip_time_proof_l317_31728

-- Define the distances
def freeway_distance : ℝ := 60
def mountain_distance : ℝ := 20

-- Define the time spent on mountain pass
def mountain_time : ℝ := 40

-- Define the speed ratio
def speed_ratio : ℝ := 4

-- Define the total trip time
def total_trip_time : ℝ := 70

-- Theorem statement
theorem trip_time_proof :
  let mountain_speed := mountain_distance / mountain_time
  let freeway_speed := speed_ratio * mountain_speed
  let freeway_time := freeway_distance / freeway_speed
  mountain_time + freeway_time = total_trip_time := by sorry

end NUMINAMATH_CALUDE_trip_time_proof_l317_31728


namespace NUMINAMATH_CALUDE_range_of_y_over_x_l317_31790

theorem range_of_y_over_x (x y : ℝ) (h : (x - 2)^2 + y^2 = 3) :
  ∃ (k : ℝ), y / x = k ∧ -Real.sqrt 3 ≤ k ∧ k ≤ Real.sqrt 3 ∧
  (∀ (m : ℝ), y / x = m → -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_y_over_x_l317_31790


namespace NUMINAMATH_CALUDE_polynomial_simplification_l317_31707

theorem polynomial_simplification (x y : ℝ) :
  (4 * x^9 + 3 * y^8 + 5 * x^7) + (2 * x^10 + 6 * x^9 + y^8 + 4 * x^7 + 2 * y^4 + 7 * x + 9) =
  2 * x^10 + 10 * x^9 + 4 * y^8 + 9 * x^7 + 2 * y^4 + 7 * x + 9 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l317_31707


namespace NUMINAMATH_CALUDE_normal_hours_calculation_l317_31764

/-- Represents a worker's pay structure and a specific workday --/
structure WorkDay where
  regularRate : ℝ  -- Regular hourly rate
  overtimeMultiplier : ℝ  -- Overtime rate multiplier
  totalHours : ℝ  -- Total hours worked on a specific day
  totalEarnings : ℝ  -- Total earnings for the specific day
  normalHours : ℝ  -- Normal working hours per day

/-- Theorem stating that given the specific conditions, the normal working hours are 7.5 --/
theorem normal_hours_calculation (w : WorkDay) 
  (h1 : w.regularRate = 3.5)
  (h2 : w.overtimeMultiplier = 1.5)
  (h3 : w.totalHours = 10.5)
  (h4 : w.totalEarnings = 42)
  : w.normalHours = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_normal_hours_calculation_l317_31764


namespace NUMINAMATH_CALUDE_rectangle_area_l317_31727

/-- Given a rectangle with width m centimeters and length 1 centimeter more than twice its width,
    its area is equal to 2m^2 + m square centimeters. -/
theorem rectangle_area (m : ℝ) : 
  let width := m
  let length := 2 * m + 1
  width * length = 2 * m^2 + m := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l317_31727


namespace NUMINAMATH_CALUDE_trailer_cost_is_120000_l317_31778

/-- Represents the cost of a house in dollars -/
def house_cost : ℕ := 480000

/-- Represents the loan period in months -/
def loan_period : ℕ := 240

/-- Represents the additional monthly payment for the house compared to the trailer in dollars -/
def additional_house_payment : ℕ := 1500

/-- Calculates the cost of the trailer given the house cost, loan period, and additional house payment -/
def trailer_cost (h : ℕ) (l : ℕ) (a : ℕ) : ℕ := 
  h - l * a

/-- Theorem stating that the cost of the trailer is $120,000 -/
theorem trailer_cost_is_120000 : 
  trailer_cost house_cost loan_period additional_house_payment = 120000 := by
  sorry

end NUMINAMATH_CALUDE_trailer_cost_is_120000_l317_31778


namespace NUMINAMATH_CALUDE_prob_b_in_middle_l317_31774

def number_of_people : ℕ := 3

def total_arrangements (n : ℕ) : ℕ := n.factorial

def middle_arrangements (n : ℕ) : ℕ := (n - 1).factorial

def probability_in_middle (n : ℕ) : ℚ :=
  (middle_arrangements n : ℚ) / (total_arrangements n : ℚ)

theorem prob_b_in_middle :
  probability_in_middle number_of_people = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_b_in_middle_l317_31774


namespace NUMINAMATH_CALUDE_approximating_functions_theorem1_approximating_functions_theorem2_l317_31776

-- Define the concept of "approximating functions"
def approximating_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → -1 ≤ f x - g x ∧ f x - g x ≤ 1

-- Define the functions
def f1 (x : ℝ) : ℝ := x - 5
def f2 (x : ℝ) : ℝ := x^2 - 4*x
def g1 (x : ℝ) : ℝ := x^2 - 1
def g2 (x : ℝ) : ℝ := 2*x^2 - x

-- State the theorems to be proved
theorem approximating_functions_theorem1 :
  approximating_functions f1 f2 3 4 := by sorry

theorem approximating_functions_theorem2 :
  approximating_functions g1 g2 0 1 := by sorry

end NUMINAMATH_CALUDE_approximating_functions_theorem1_approximating_functions_theorem2_l317_31776


namespace NUMINAMATH_CALUDE_greatest_divisible_integer_l317_31761

theorem greatest_divisible_integer (m : ℕ+) :
  (∃ (n : ℕ), n > 0 ∧ (m^2 + n) ∣ (n^2 + m)) ∧
  (∀ (k : ℕ), k > (m^4 - m^2 + m) → ¬((m^2 + k) ∣ (k^2 + m))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisible_integer_l317_31761


namespace NUMINAMATH_CALUDE_book_arrangement_count_l317_31700

theorem book_arrangement_count : 
  let total_books : ℕ := 12
  let arabic_books : ℕ := 3
  let german_books : ℕ := 4
  let spanish_books : ℕ := 3
  let french_books : ℕ := 2
  let grouped_units : ℕ := 3  -- Arabic, Spanish, French groups
  let total_arrangements : ℕ := 
    (Nat.factorial (grouped_units + german_books)) * 
    (Nat.factorial arabic_books) * 
    (Nat.factorial spanish_books) * 
    (Nat.factorial french_books)
  total_books = arabic_books + german_books + spanish_books + french_books →
  total_arrangements = 362880 := by
sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l317_31700


namespace NUMINAMATH_CALUDE_product_of_y_values_l317_31772

theorem product_of_y_values (y : ℝ) : 
  (∃ y₁ y₂ : ℝ, 
    (|2 * y₁ * 3| + 5 = 47) ∧ 
    (|2 * y₂ * 3| + 5 = 47) ∧ 
    (y₁ ≠ y₂) ∧
    (y₁ * y₂ = -49)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_y_values_l317_31772


namespace NUMINAMATH_CALUDE_article_cost_l317_31771

/-- The cost of an article given specific selling prices and gains -/
theorem article_cost (sp1 sp2 : ℝ) (gain_increase : ℝ) :
  sp1 = 500 →
  sp2 = 570 →
  gain_increase = 0.15 →
  ∃ (cost gain : ℝ),
    cost + gain = sp1 ∧
    cost + gain * (1 + gain_increase) = sp2 ∧
    cost = 100 / 3 :=
sorry

end NUMINAMATH_CALUDE_article_cost_l317_31771


namespace NUMINAMATH_CALUDE_simple_compound_interest_equivalence_l317_31795

theorem simple_compound_interest_equivalence (P : ℝ) : 
  (P * 0.04 * 2 = 0.5 * (4000 * ((1 + 0.10)^2 - 1))) → P = 5250 :=
by sorry

end NUMINAMATH_CALUDE_simple_compound_interest_equivalence_l317_31795


namespace NUMINAMATH_CALUDE_awards_distribution_count_l317_31780

/-- The number of ways to distribute awards to students -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the correct number of ways to distribute 6 awards to 4 students -/
theorem awards_distribution_count :
  distribute_awards 6 4 = 3720 :=
by sorry

end NUMINAMATH_CALUDE_awards_distribution_count_l317_31780


namespace NUMINAMATH_CALUDE_anlu_temperature_difference_l317_31777

/-- Given a temperature range from -3°C to 3°C in Anlu on a winter day,
    the temperature difference is 6°C. -/
theorem anlu_temperature_difference :
  let min_temp : ℤ := -3
  let max_temp : ℤ := 3
  (max_temp - min_temp : ℤ) = 6 := by sorry

end NUMINAMATH_CALUDE_anlu_temperature_difference_l317_31777


namespace NUMINAMATH_CALUDE_part_one_part_two_l317_31775

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one : 
  {x : ℝ | |x - 1| ≥ |x + 1| + 1} = {x : ℝ | x ≤ -0.5} := by sorry

-- Part II
theorem part_two :
  {a : ℝ | ∀ x ≤ -1, f a x + 3 * x ≤ 0} = {a : ℝ | -4 ≤ a ∧ a ≤ 2} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l317_31775


namespace NUMINAMATH_CALUDE_number_of_routes_l317_31739

-- Define the cities
inductive City : Type
| A | B | C | D | F

-- Define the roads
inductive Road : Type
| AB | AD | AF | BC | BD | CD | DF

-- Define a route as a list of roads
def Route := List Road

-- Function to check if a route is valid (uses each road exactly once and starts at A and ends at B)
def isValidRoute (r : Route) : Prop := sorry

-- Function to count the number of valid routes
def countValidRoutes : Nat := sorry

-- Theorem to prove
theorem number_of_routes : countValidRoutes = 16 := by sorry

end NUMINAMATH_CALUDE_number_of_routes_l317_31739


namespace NUMINAMATH_CALUDE_x_fifth_minus_five_x_equals_3100_l317_31725

theorem x_fifth_minus_five_x_equals_3100 (x : ℝ) (h : x = 5) : x^5 - 5*x = 3100 := by
  sorry

end NUMINAMATH_CALUDE_x_fifth_minus_five_x_equals_3100_l317_31725


namespace NUMINAMATH_CALUDE_soccer_camp_ratio_l317_31798

theorem soccer_camp_ratio (total_kids : ℕ) (afternoon_kids : ℕ) 
  (h1 : total_kids = 2000)
  (h2 : afternoon_kids = 750)
  (h3 : ∃ (morning_kids : ℕ), 4 * morning_kids = total_soccer_kids - afternoon_kids) :
  ∃ (total_soccer_kids : ℕ), 
    2 * total_soccer_kids = total_kids ∧ 
    4 * afternoon_kids = 3 * total_soccer_kids := by
sorry


end NUMINAMATH_CALUDE_soccer_camp_ratio_l317_31798


namespace NUMINAMATH_CALUDE_point_movement_l317_31731

def move_point (start : ℤ) (distance : ℤ) : ℤ := start + distance

theorem point_movement (A B : ℤ) :
  A = -3 →
  move_point A 4 = B →
  B = 1 := by sorry

end NUMINAMATH_CALUDE_point_movement_l317_31731


namespace NUMINAMATH_CALUDE_max_boxes_of_paint_A_l317_31799

/-- The maximum number of boxes of paint A that can be purchased given the conditions -/
theorem max_boxes_of_paint_A : ℕ :=
  let price_A : ℕ := 24  -- Price of paint A in yuan
  let price_B : ℕ := 16  -- Price of paint B in yuan
  let total_boxes : ℕ := 200  -- Total number of boxes to be purchased
  let max_cost : ℕ := 3920  -- Maximum total cost in yuan
  let max_A : ℕ := 90  -- Maximum number of boxes of paint A (to be proved)

  have h1 : price_A + 2 * price_B = 56 := by sorry
  have h2 : 2 * price_A + price_B = 64 := by sorry
  have h3 : ∀ m : ℕ, m ≤ total_boxes → 
    price_A * m + price_B * (total_boxes - m) ≤ max_cost → 
    m ≤ max_A := by sorry

  max_A

end NUMINAMATH_CALUDE_max_boxes_of_paint_A_l317_31799
