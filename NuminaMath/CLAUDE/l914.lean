import Mathlib

namespace NUMINAMATH_CALUDE_orange_to_apple_ratio_l914_91472

/-- Given the total weight of fruits and the weight of oranges, proves the ratio of oranges to apples -/
theorem orange_to_apple_ratio
  (total_weight : ℕ)
  (orange_weight : ℕ)
  (h1 : total_weight = 12)
  (h2 : orange_weight = 10) :
  orange_weight / (total_weight - orange_weight) = 5 := by
  sorry

end NUMINAMATH_CALUDE_orange_to_apple_ratio_l914_91472


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l914_91457

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x > 0, x^2 + 1/x^2 ≥ 2) ∧
  (∃ x ≤ 0, x ≠ 0 ∧ x^2 + 1/x^2 ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l914_91457


namespace NUMINAMATH_CALUDE_restaurant_pies_theorem_l914_91487

/-- The number of pies sold in a week by a restaurant that sells 8 pies per day -/
def pies_sold_in_week (pies_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  pies_per_day * days_in_week

/-- Proof that a restaurant selling 8 pies per day for a week sells 56 pies in total -/
theorem restaurant_pies_theorem :
  pies_sold_in_week 8 7 = 56 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_pies_theorem_l914_91487


namespace NUMINAMATH_CALUDE_sum_of_extremes_l914_91467

def is_valid_number (n : ℕ) : Prop :=
  n > 100 ∧ n < 1000 ∧ ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [1, 2, 5]

def smallest_valid_number : ℕ := sorry

def largest_valid_number : ℕ := sorry

theorem sum_of_extremes :
  smallest_valid_number + largest_valid_number = 646 ∧
  is_valid_number smallest_valid_number ∧
  is_valid_number largest_valid_number ∧
  ∀ n : ℕ, is_valid_number n →
    smallest_valid_number ≤ n ∧ n ≤ largest_valid_number :=
sorry

end NUMINAMATH_CALUDE_sum_of_extremes_l914_91467


namespace NUMINAMATH_CALUDE_car_highway_efficiency_l914_91469

/-- The number of miles the car can travel on the highway with one gallon of gasoline. -/
def highway_miles_per_gallon : ℝ := 38

/-- The number of miles the car can travel in the city with one gallon of gasoline. -/
def city_miles_per_gallon : ℝ := 20

/-- Proves that the car can travel 38 miles on the highway with one gallon of gasoline,
    given the conditions stated in the problem. -/
theorem car_highway_efficiency :
  highway_miles_per_gallon = 38 ∧
  (4 / highway_miles_per_gallon + 4 / city_miles_per_gallon =
   8 / highway_miles_per_gallon * (1 + 0.45000000000000014)) :=
by sorry

end NUMINAMATH_CALUDE_car_highway_efficiency_l914_91469


namespace NUMINAMATH_CALUDE_negative_sqrt_three_squared_equals_negative_three_l914_91453

theorem negative_sqrt_three_squared_equals_negative_three :
  -Real.sqrt (3^2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_three_squared_equals_negative_three_l914_91453


namespace NUMINAMATH_CALUDE_root_negative_implies_inequality_l914_91495

theorem root_negative_implies_inequality (a : ℝ) : 
  (∃ x : ℝ, x - 2*a + 4 = 0 ∧ x < 0) → (a-3)*(a-4) > 0 := by
  sorry

end NUMINAMATH_CALUDE_root_negative_implies_inequality_l914_91495


namespace NUMINAMATH_CALUDE_four_solutions_to_g_composition_l914_91488

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 4*x

-- State the theorem
theorem four_solutions_to_g_composition :
  ∃! (s : Finset ℝ), (∀ c ∈ s, g (g (g (g c))) = 5) ∧ s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_four_solutions_to_g_composition_l914_91488


namespace NUMINAMATH_CALUDE_all_points_collinear_l914_91483

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  ∃ l : Line, p.onLine l ∧ q.onLine l ∧ r.onLine l

/-- The main theorem -/
theorem all_points_collinear (S : Set Point) (h_finite : Set.Finite S)
    (h_three_point : ∀ p q r : Point, p ∈ S → q ∈ S → r ∈ S → p ≠ q → 
      (∃ l : Line, p.onLine l ∧ q.onLine l) → r.onLine l) :
    ∀ p q r : Point, p ∈ S → q ∈ S → r ∈ S → collinear p q r :=
  sorry

end NUMINAMATH_CALUDE_all_points_collinear_l914_91483


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l914_91459

def f (x : ℝ) := x^3 + 3*x - 1

theorem root_exists_in_interval :
  Continuous f →
  f 0 < 0 →
  f 0.5 > 0 →
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo 0 0.5 ∧ f x₀ = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l914_91459


namespace NUMINAMATH_CALUDE_music_exam_songs_l914_91405

/-- Represents a girl participating in the music exam -/
inductive Girl
| Anna
| Bea
| Cili
| Dora

/-- The number of times each girl sang -/
def timesSang (g : Girl) : ℕ :=
  match g with
  | Girl.Anna => 8
  | Girl.Bea => 7  -- We assume 7 as it satisfies the conditions
  | Girl.Cili => 7 -- We assume 7 as it satisfies the conditions
  | Girl.Dora => 5

/-- The total number of individual singing assignments -/
def totalSingingAssignments : ℕ := 
  (timesSang Girl.Anna) + (timesSang Girl.Bea) + (timesSang Girl.Cili) + (timesSang Girl.Dora)

theorem music_exam_songs :
  (∀ g : Girl, timesSang g ≤ timesSang Girl.Anna) ∧ 
  (∀ g : Girl, g ≠ Girl.Anna → timesSang g < timesSang Girl.Anna) ∧
  (∀ g : Girl, g ≠ Girl.Dora → timesSang Girl.Dora < timesSang g) ∧
  (totalSingingAssignments % 3 = 0) →
  totalSingingAssignments / 3 = 9 := by
  sorry

#eval totalSingingAssignments / 3

end NUMINAMATH_CALUDE_music_exam_songs_l914_91405


namespace NUMINAMATH_CALUDE_inequality_system_solution_l914_91440

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (x - a ≥ b ∧ 2*x - a - 1 < 2*b) ↔ (3 ≤ x ∧ x < 5)) →
  a = -3 ∧ b = 6 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l914_91440


namespace NUMINAMATH_CALUDE_quadratic_completion_l914_91474

theorem quadratic_completion (c : ℝ) (n : ℝ) : 
  c < 0 → 
  (∀ x, x^2 + c*x + (1/4 : ℝ) = (x + n)^2 + (1/8 : ℝ)) → 
  c = -Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completion_l914_91474


namespace NUMINAMATH_CALUDE_cube_volume_scaling_l914_91479

theorem cube_volume_scaling (v : ℝ) (s : ℝ) :
  v > 0 →
  s > 0 →
  let original_side := v ^ (1/3)
  let scaled_side := s * original_side
  let scaled_volume := scaled_side ^ 3
  v = 64 ∧ s = 2 → scaled_volume = 512 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_scaling_l914_91479


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_negative_seven_l914_91402

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x : ℝ | ∃ (a b : ℝ), x^2 + a*x + b ≤ 0}

-- State the theorem
theorem sum_of_coefficients_is_negative_seven 
  (h_union : A ∪ B = Set.univ)
  (h_intersection : A ∩ B = Set.Ioc 3 4)
  : ∃ (a b : ℝ), B = {x : ℝ | x^2 + a*x + b ≤ 0} ∧ a + b = -7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_negative_seven_l914_91402


namespace NUMINAMATH_CALUDE_bucket_fill_time_l914_91456

/-- Given that two-thirds of a bucket is filled in 90 seconds,
    prove that the time taken to fill the bucket completely is 135 seconds. -/
theorem bucket_fill_time (fill_time : ℝ) (h : fill_time = 90) :
  (3 / 2) * fill_time = 135 := by
  sorry

end NUMINAMATH_CALUDE_bucket_fill_time_l914_91456


namespace NUMINAMATH_CALUDE_problem_statement_l914_91449

theorem problem_statement (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 - b^2 = -1) :
  a^2008 - b^2008 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l914_91449


namespace NUMINAMATH_CALUDE_total_current_ages_l914_91427

theorem total_current_ages (amar akbar anthony : ℕ) : 
  (amar - 4) + (akbar - 4) + (anthony - 4) = 54 → amar + akbar + anthony = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_current_ages_l914_91427


namespace NUMINAMATH_CALUDE_inequality_proof_l914_91421

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  (x / (1 - x^2)) + (y / (1 - y^2)) + (z / (1 - z^2)) ≥ (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l914_91421


namespace NUMINAMATH_CALUDE_satellite_altitude_scientific_notation_l914_91482

/-- The altitude of a Beidou satellite in meters -/
def satellite_altitude : ℝ := 21500000

/-- Scientific notation representation of the satellite altitude -/
def scientific_notation : ℝ := 2.15 * (10 ^ 7)

/-- Theorem stating that the satellite altitude is equal to its scientific notation representation -/
theorem satellite_altitude_scientific_notation : 
  satellite_altitude = scientific_notation := by sorry

end NUMINAMATH_CALUDE_satellite_altitude_scientific_notation_l914_91482


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l914_91424

theorem negation_of_existential_proposition :
  (¬ ∃ x₀ : ℝ, x₀ < 0 ∧ Real.exp x₀ - x₀ > 1) ↔ (∀ x : ℝ, x < 0 → Real.exp x - x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l914_91424


namespace NUMINAMATH_CALUDE_conference_attendance_theorem_l914_91436

/-- The percentage of attendees who paid their conference fee in full but did not register at least two weeks in advance -/
def late_payment_percentage : ℝ := 10

/-- The percentage of conference attendees who registered at least two weeks in advance -/
def early_registration_percentage : ℝ := 86.67

/-- The percentage of attendees who registered at least two weeks in advance and paid their conference fee in full -/
def early_registration_full_payment_percentage : ℝ := 96.3

theorem conference_attendance_theorem :
  (100 - late_payment_percentage) / 100 * early_registration_full_payment_percentage = early_registration_percentage :=
by sorry

end NUMINAMATH_CALUDE_conference_attendance_theorem_l914_91436


namespace NUMINAMATH_CALUDE_f_inequality_A_is_solution_set_l914_91446

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Define set A
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- State the theorem
theorem f_inequality (a b : ℝ) (ha : a ∈ A) (hb : b ∈ A) :
  f (a * b) > f a - f b := by
  sorry

-- Prove that A is indeed the solution set to f(x) < 3 - |2x + 1|
theorem A_is_solution_set (x : ℝ) :
  x ∈ A ↔ f x < 3 - |2 * x + 1| := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_A_is_solution_set_l914_91446


namespace NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_l914_91461

/-- Curve C defined by the equation x²/a + y²/b = 1 -/
structure CurveC (a b : ℝ) where
  equation : ∀ (x y : ℝ), x^2 / a + y^2 / b = 1

/-- Predicate for C being an ellipse with foci on the x-axis -/
def is_ellipse_x_foci (a b : ℝ) : Prop :=
  ∃ (c : ℝ), a > b ∧ b > 0 ∧ c^2 = a^2 - b^2

/-- Main theorem: "a > b" is necessary but not sufficient for C to be an ellipse with foci on x-axis -/
theorem a_gt_b_necessary_not_sufficient (a b : ℝ) :
  (is_ellipse_x_foci a b → a > b) ∧
  ¬(a > b → is_ellipse_x_foci a b) :=
sorry

end NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_l914_91461


namespace NUMINAMATH_CALUDE_sequence_matches_l914_91486

def a (n : ℕ) : ℤ := (-1)^n * (1 - 2*n)

theorem sequence_matches : 
  (a 1 = 1) ∧ (a 2 = -3) ∧ (a 3 = 5) ∧ (a 4 = -7) ∧ (a 5 = 9) := by
  sorry

end NUMINAMATH_CALUDE_sequence_matches_l914_91486


namespace NUMINAMATH_CALUDE_perpendicular_line_through_intersection_l914_91425

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - y - 3 = 0
def line2 (x y : ℝ) : Prop := 4*x - 3*y - 5 = 0
def line3 (x y : ℝ) : Prop := 2*x + 3*y + 5 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ := (2, 1)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 3*x - 2*y - 4 = 0

-- Theorem statement
theorem perpendicular_line_through_intersection :
  ∃ (x y : ℝ), 
    line1 x y ∧ 
    line2 x y ∧ 
    perpendicular_line x y ∧
    (∀ (m : ℝ), line3 x y → m = -2/3) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_intersection_l914_91425


namespace NUMINAMATH_CALUDE_power_of_power_l914_91416

-- Define the problem statement
theorem power_of_power : (3^4)^2 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l914_91416


namespace NUMINAMATH_CALUDE_quadratic_minimum_l914_91493

/-- Given a quadratic function f(x) = x^2 - 2x + m with a minimum value of -2 
    on the interval [2, +∞), prove that m = -2. -/
theorem quadratic_minimum (m : ℝ) : 
  (∀ x : ℝ, x ≥ 2 → x^2 - 2*x + m ≥ -2) ∧ 
  (∃ x : ℝ, x ≥ 2 ∧ x^2 - 2*x + m = -2) → 
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l914_91493


namespace NUMINAMATH_CALUDE_cylinder_volume_l914_91466

/-- The volume of a cylinder with equal base diameter and height, and lateral area π. -/
theorem cylinder_volume (r h : ℝ) (h1 : h = 2 * r) (h2 : 2 * π * r * h = π) : π * r^2 * h = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l914_91466


namespace NUMINAMATH_CALUDE_smallest_positive_number_l914_91445

theorem smallest_positive_number :
  let a := 8 - 3 * Real.sqrt 10
  let b := 3 * Real.sqrt 10 - 8
  let c := 23 - 6 * Real.sqrt 15
  let d := 58 - 12 * Real.sqrt 30
  let e := 12 * Real.sqrt 30 - 58
  (0 < b) ∧
  (a ≤ 0 ∨ b < a) ∧
  (c ≤ 0 ∨ b < c) ∧
  (d ≤ 0 ∨ b < d) ∧
  (e ≤ 0 ∨ b < e) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_number_l914_91445


namespace NUMINAMATH_CALUDE_fraction_simplification_l914_91431

theorem fraction_simplification : 
  (1/4 - 1/5) / (1/3 - 1/4) = 3/5 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l914_91431


namespace NUMINAMATH_CALUDE_factor_expression_1_l914_91412

theorem factor_expression_1 (m n : ℝ) :
  4/9 * m^2 + 4/3 * m * n + n^2 = (2/3 * m + n)^2 := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_1_l914_91412


namespace NUMINAMATH_CALUDE_intersection_point_of_two_lines_l914_91428

/-- Two lines in 2D space -/
structure Line2D where
  origin : ℝ × ℝ
  direction : ℝ × ℝ

/-- The point lies on the given line -/
def pointOnLine (p : ℝ × ℝ) (l : Line2D) : Prop :=
  ∃ t : ℝ, p = (l.origin.1 + t * l.direction.1, l.origin.2 + t * l.direction.2)

theorem intersection_point_of_two_lines :
  let l1 : Line2D := { origin := (2, 3), direction := (-1, 5) }
  let l2 : Line2D := { origin := (0, 7), direction := (-1, 4) }
  let p : ℝ × ℝ := (6, -17)
  (pointOnLine p l1 ∧ pointOnLine p l2) ∧
  ∀ q : ℝ × ℝ, pointOnLine q l1 ∧ pointOnLine q l2 → q = p :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_two_lines_l914_91428


namespace NUMINAMATH_CALUDE_solve_equation_l914_91455

/-- Given an equation 19(x + y) + 17 = 19(-x + y) - n where x = 1, prove that n = -55 -/
theorem solve_equation (y : ℝ) : 
  (∃ (n : ℝ), 19 * (1 + y) + 17 = 19 * (-1 + y) - n) → 
  (∃ (n : ℝ), 19 * (1 + y) + 17 = 19 * (-1 + y) - n ∧ n = -55) :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_l914_91455


namespace NUMINAMATH_CALUDE_total_soak_time_l914_91458

def grass_soak_time : ℕ := 3
def marinara_soak_time : ℕ := 7
def ink_soak_time : ℕ := 5
def coffee_soak_time : ℕ := 10

def num_grass_stains : ℕ := 3
def num_marinara_stains : ℕ := 1
def num_ink_stains : ℕ := 2
def num_coffee_stains : ℕ := 1

theorem total_soak_time :
  grass_soak_time * num_grass_stains +
  marinara_soak_time * num_marinara_stains +
  ink_soak_time * num_ink_stains +
  coffee_soak_time * num_coffee_stains = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_soak_time_l914_91458


namespace NUMINAMATH_CALUDE_det2_specific_values_det2_quadratic_relation_l914_91437

def det2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem det2_specific_values :
  det2 5 6 7 8 = -2 :=
sorry

theorem det2_quadratic_relation (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  det2 (x + 1) (3*x) (x - 2) (x - 1) = 6*x + 1 :=
sorry

end NUMINAMATH_CALUDE_det2_specific_values_det2_quadratic_relation_l914_91437


namespace NUMINAMATH_CALUDE_gum_cost_800_l914_91420

/-- The cost of gum pieces with a bulk discount -/
def gum_cost (pieces : ℕ) : ℚ :=
  let base_cost := pieces
  let discount_threshold := 500
  let discount_rate := 1 / 10
  let total_cents :=
    if pieces > discount_threshold
    then base_cost * (1 - discount_rate)
    else base_cost
  total_cents / 100

/-- The cost of 800 pieces of gum is $7.20 -/
theorem gum_cost_800 : gum_cost 800 = 72 / 10 := by
  sorry

end NUMINAMATH_CALUDE_gum_cost_800_l914_91420


namespace NUMINAMATH_CALUDE_jackson_holidays_l914_91418

/-- The number of holidays taken in a year given the number of days off per month and the number of months in a year -/
def holidays_in_year (days_off_per_month : ℕ) (months_in_year : ℕ) : ℕ :=
  days_off_per_month * months_in_year

/-- Theorem stating that taking 3 days off every month for 12 months results in 36 holidays in a year -/
theorem jackson_holidays :
  holidays_in_year 3 12 = 36 := by
  sorry

end NUMINAMATH_CALUDE_jackson_holidays_l914_91418


namespace NUMINAMATH_CALUDE_a_upper_bound_l914_91448

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - x ≤ 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2^(1-x) + a ≤ 0}

-- State the theorem
theorem a_upper_bound (a : ℝ) (h : A ⊆ B a) : a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_a_upper_bound_l914_91448


namespace NUMINAMATH_CALUDE_trapezoid_division_areas_l914_91409

/-- Given a trapezoid with base length a, parallel side length b, and height m,
    when divided into three equal parts, prove that the areas of the resulting
    trapezoids are as stated. -/
theorem trapezoid_division_areas (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0) :
  let s := m / 3
  let x := (2 * a + b) / 3
  let y := (a + 2 * b) / 3
  let t₁ := ((a + x) / 2) * s
  let t₂ := ((x + y) / 2) * s
  let t₃ := ((y + b) / 2) * s
  (t₁ = (5 * a + b) * m / 18) ∧
  (t₂ = (a + b) * m / 6) ∧
  (t₃ = (a + 5 * b) * m / 18) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_division_areas_l914_91409


namespace NUMINAMATH_CALUDE_game_price_calculation_l914_91439

theorem game_price_calculation (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ) : 
  total_games = 15 → non_working_games = 6 → total_earnings = 63 →
  total_earnings / (total_games - non_working_games) = 7 := by
sorry

end NUMINAMATH_CALUDE_game_price_calculation_l914_91439


namespace NUMINAMATH_CALUDE_horner_operations_l914_91499

-- Define the polynomial coefficients
def coeffs : List ℝ := [8, 7, 6, 5, 4, 3, 2]

-- Define Horner's method
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

-- Define a function to count operations in Horner's method
def count_operations (coeffs : List ℝ) : ℕ × ℕ :=
  (coeffs.length - 1, coeffs.length - 1)

-- Theorem statement
theorem horner_operations :
  let (mults, adds) := count_operations coeffs
  mults = 6 ∧ adds = 6 :=
sorry

end NUMINAMATH_CALUDE_horner_operations_l914_91499


namespace NUMINAMATH_CALUDE_simplify_square_roots_l914_91434

theorem simplify_square_roots :
  Real.sqrt 8 - Real.sqrt 32 + Real.sqrt 72 - Real.sqrt 50 = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l914_91434


namespace NUMINAMATH_CALUDE_ellipse_and_distance_l914_91477

/-- An ellipse with the given conditions -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The configuration of points and lines for the problem -/
structure Configuration (C : Ellipse) where
  right_focus : ℝ × ℝ
  passing_point : ℝ × ℝ
  M : ℝ
  l : ℝ → ℝ → Prop
  A : ℝ × ℝ
  B : ℝ × ℝ
  N : ℝ × ℝ
  h₁ : right_focus = (Real.sqrt 3, 0)
  h₂ : passing_point = (-1, Real.sqrt 3 / 2)
  h₃ : C.a^2 * (passing_point.1^2 / C.a^2 + passing_point.2^2 / C.b^2) = C.a^2
  h₄ : l M A.2 ∧ l M B.2
  h₅ : A.2 > 0 ∧ B.2 < 0
  h₆ : (A.1 - M)^2 + A.2^2 = 4 * ((B.1 - M)^2 + B.2^2)
  h₇ : N.1^2 + N.2^2 = 4/7
  h₈ : ∀ x y, l x y → (x - N.1)^2 + (y - N.2)^2 ≥ 4/7

/-- The main theorem to be proved -/
theorem ellipse_and_distance (C : Ellipse) (cfg : Configuration C) :
  C.a^2 = 4 ∧ C.b^2 = 1 ∧ 
  (cfg.M - cfg.N.1)^2 + cfg.N.2^2 = (4 * Real.sqrt 21 / 21)^2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_distance_l914_91477


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l914_91490

/-- Represents a repeating decimal with a non-repeating part and a repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℚ
  repeating : ℚ
  repeatingDigits : ℕ

/-- Converts a RepeatingDecimal to its fraction representation -/
def repeatingDecimalToFraction (x : RepeatingDecimal) : ℚ :=
  x.nonRepeating + x.repeating / (1 - (1 / 10 ^ x.repeatingDigits))

theorem repeating_decimal_equals_fraction :
  let x : RepeatingDecimal := {
    nonRepeating := 1/2,
    repeating := 23/1000,
    repeatingDigits := 3
  }
  repeatingDecimalToFraction x = 1045/1998 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l914_91490


namespace NUMINAMATH_CALUDE_two_color_theorem_l914_91480

/-- A line in a plane --/
structure Line where
  -- We don't need to define the specifics of a line for this problem

/-- A region in a plane --/
structure Region where
  -- We don't need to define the specifics of a region for this problem

/-- A color (we only need two colors) --/
inductive Color
  | A
  | B

/-- A function that determines if two regions are adjacent --/
def adjacent (r1 r2 : Region) : Prop :=
  sorry -- The specific implementation is not important for the statement

/-- A coloring of regions --/
def Coloring := Region → Color

/-- A valid coloring ensures adjacent regions have different colors --/
def valid_coloring (c : Coloring) : Prop :=
  ∀ r1 r2, adjacent r1 r2 → c r1 ≠ c r2

/-- The main theorem --/
theorem two_color_theorem (lines : List Line) :
  ∃ (regions : List Region) (c : Coloring), valid_coloring c :=
sorry

end NUMINAMATH_CALUDE_two_color_theorem_l914_91480


namespace NUMINAMATH_CALUDE_dividend_problem_l914_91463

theorem dividend_problem (dividend divisor quotient : ℕ) : 
  dividend + divisor + quotient = 103 →
  quotient = 3 →
  dividend % divisor = 0 →
  dividend / divisor = quotient →
  dividend = 75 := by
sorry

end NUMINAMATH_CALUDE_dividend_problem_l914_91463


namespace NUMINAMATH_CALUDE_power_four_inequality_l914_91481

theorem power_four_inequality (a b : ℝ) : (a^4 + b^4) / 2 ≥ ((a + b) / 2)^4 := by
  sorry

end NUMINAMATH_CALUDE_power_four_inequality_l914_91481


namespace NUMINAMATH_CALUDE_abs_x_lt_2_sufficient_not_necessary_l914_91411

theorem abs_x_lt_2_sufficient_not_necessary :
  (∃ x : ℝ, (abs x < 2 → x^2 - x - 6 < 0) ∧ 
            ¬(x^2 - x - 6 < 0 → abs x < 2)) :=
sorry

end NUMINAMATH_CALUDE_abs_x_lt_2_sufficient_not_necessary_l914_91411


namespace NUMINAMATH_CALUDE_ryans_initial_funds_l914_91430

/-- Proves that Ryan's initial funds equal the total cost minus crowdfunding amount -/
theorem ryans_initial_funds 
  (average_funding : ℕ) 
  (people_to_recruit : ℕ) 
  (total_cost : ℕ) 
  (h1 : average_funding = 10)
  (h2 : people_to_recruit = 80)
  (h3 : total_cost = 1000) :
  total_cost - (average_funding * people_to_recruit) = 200 := by
  sorry

#check ryans_initial_funds

end NUMINAMATH_CALUDE_ryans_initial_funds_l914_91430


namespace NUMINAMATH_CALUDE_donut_distribution_count_l914_91413

/-- The number of ways to distribute items into bins -/
def distribute_items (total_items : ℕ) (num_bins : ℕ) (items_to_distribute : ℕ) : ℕ :=
  Nat.choose (items_to_distribute + num_bins - 1) (num_bins - 1)

/-- Theorem stating the number of ways to distribute donuts -/
theorem donut_distribution_count :
  let total_donuts : ℕ := 10
  let donut_types : ℕ := 5
  let donuts_to_distribute : ℕ := total_donuts - donut_types
  distribute_items total_donuts donut_types donuts_to_distribute = 126 := by
  sorry

#eval distribute_items 10 5 5

end NUMINAMATH_CALUDE_donut_distribution_count_l914_91413


namespace NUMINAMATH_CALUDE_circle_equation_l914_91404

theorem circle_equation (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ (x - 2)^2 + y^2 = r^2) ∧ 
  ((-2)^2 + 0^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l914_91404


namespace NUMINAMATH_CALUDE_pitcher_problem_l914_91450

theorem pitcher_problem (C : ℝ) (h : C > 0) : 
  let juice_volume : ℝ := (3/4) * C
  let num_cups : ℕ := 5
  let juice_per_cup : ℝ := juice_volume / num_cups
  (juice_per_cup / C) * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_pitcher_problem_l914_91450


namespace NUMINAMATH_CALUDE_polynomial_equality_l914_91432

/-- Given that 4x^5 + 3x^3 + 2x^2 + p(x) = 6x^3 - 5x^2 + 4x - 2 for all x,
    prove that p(x) = -4x^5 + 3x^3 - 7x^2 + 4x - 2 -/
theorem polynomial_equality (p : ℝ → ℝ) :
  (∀ x, 4 * x^5 + 3 * x^3 + 2 * x^2 + p x = 6 * x^3 - 5 * x^2 + 4 * x - 2) →
  (∀ x, p x = -4 * x^5 + 3 * x^3 - 7 * x^2 + 4 * x - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l914_91432


namespace NUMINAMATH_CALUDE_max_value_2x_plus_y_max_value_2x_plus_y_achievable_l914_91492

theorem max_value_2x_plus_y (x y : ℝ) : 
  2 * x - y ≤ 0 → x + y ≤ 3 → x ≥ 0 → 2 * x + y ≤ 4 := by
  sorry

theorem max_value_2x_plus_y_achievable : 
  ∃ x y : ℝ, 2 * x - y ≤ 0 ∧ x + y ≤ 3 ∧ x ≥ 0 ∧ 2 * x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_2x_plus_y_max_value_2x_plus_y_achievable_l914_91492


namespace NUMINAMATH_CALUDE_specific_sampling_problem_l914_91451

/-- Systematic sampling function -/
def systematicSample (totalPopulation sampleSize firstDrawn nthGroup : ℕ) : ℕ :=
  let interval := totalPopulation / sampleSize
  firstDrawn + interval * (nthGroup - 1)

/-- Theorem for the specific sampling problem -/
theorem specific_sampling_problem :
  systematicSample 1000 50 15 21 = 415 := by
  sorry

end NUMINAMATH_CALUDE_specific_sampling_problem_l914_91451


namespace NUMINAMATH_CALUDE_product_xyz_equals_negative_one_l914_91406

theorem product_xyz_equals_negative_one 
  (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 2) 
  (h3 : z + 1/x = 2) : 
  x * y * z = -1 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_equals_negative_one_l914_91406


namespace NUMINAMATH_CALUDE_area_of_smaller_circle_l914_91468

/-- Two circles are externally tangent with common tangent lines -/
structure TangentCircles where
  center_small : ℝ × ℝ
  center_large : ℝ × ℝ
  radius_small : ℝ
  radius_large : ℝ
  tangent_point : ℝ × ℝ
  externally_tangent : (center_small.1 - center_large.1)^2 + (center_small.2 - center_large.2)^2 = (radius_small + radius_large)^2
  radius_ratio : radius_large = 3 * radius_small

/-- Common tangent line -/
structure CommonTangent where
  P : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  PA_length : ℝ
  AB_length : ℝ
  PA_eq_AB : PA_length = AB_length
  PA_eq_8 : PA_length = 8

/-- The main theorem -/
theorem area_of_smaller_circle (tc : TangentCircles) (ct : CommonTangent) : 
  π * tc.radius_small^2 = 16 * π :=
sorry

end NUMINAMATH_CALUDE_area_of_smaller_circle_l914_91468


namespace NUMINAMATH_CALUDE_oliver_battle_gremlins_count_l914_91485

/-- Oliver's card collection -/
structure CardCollection where
  monster_club : ℕ
  alien_baseball : ℕ
  battle_gremlins : ℕ

/-- Oliver's card collection satisfies the given conditions -/
def oliver_collection : CardCollection where
  monster_club := 32
  alien_baseball := 16
  battle_gremlins := 48

/-- Theorem: Oliver has 48 Battle Gremlins cards given the conditions -/
theorem oliver_battle_gremlins_count : 
  oliver_collection.battle_gremlins = 48 ∧
  oliver_collection.monster_club = 2 * oliver_collection.alien_baseball ∧
  oliver_collection.battle_gremlins = 3 * oliver_collection.alien_baseball :=
by sorry

end NUMINAMATH_CALUDE_oliver_battle_gremlins_count_l914_91485


namespace NUMINAMATH_CALUDE_carly_swimming_time_l914_91470

/-- Carly's swimming practice schedule and total time calculation -/
theorem carly_swimming_time :
  let butterfly_hours_per_day : ℕ := 3
  let butterfly_days_per_week : ℕ := 4
  let backstroke_hours_per_day : ℕ := 2
  let backstroke_days_per_week : ℕ := 6
  let weeks_per_month : ℕ := 4
  
  let butterfly_hours_per_week : ℕ := butterfly_hours_per_day * butterfly_days_per_week
  let backstroke_hours_per_week : ℕ := backstroke_hours_per_day * backstroke_days_per_week
  let total_hours_per_week : ℕ := butterfly_hours_per_week + backstroke_hours_per_week
  let total_hours_per_month : ℕ := total_hours_per_week * weeks_per_month
  
  total_hours_per_month = 96 :=
by
  sorry


end NUMINAMATH_CALUDE_carly_swimming_time_l914_91470


namespace NUMINAMATH_CALUDE_order_of_operations_4_times_20_plus_30_l914_91444

theorem order_of_operations_4_times_20_plus_30 : 
  let expression := 4 * (20 + 30)
  let correct_order := ["addition", "multiplication"]
  correct_order = ["addition", "multiplication"] := by sorry

end NUMINAMATH_CALUDE_order_of_operations_4_times_20_plus_30_l914_91444


namespace NUMINAMATH_CALUDE_triangle_parallelogram_altitude_l914_91462

theorem triangle_parallelogram_altitude (base : ℝ) (triangle_altitude parallelogram_altitude : ℝ) :
  base > 0 →
  parallelogram_altitude > 0 →
  parallelogram_altitude = 100 →
  (1 / 2 * base * triangle_altitude) = (base * parallelogram_altitude) →
  triangle_altitude = 200 := by
  sorry

end NUMINAMATH_CALUDE_triangle_parallelogram_altitude_l914_91462


namespace NUMINAMATH_CALUDE_souvenir_spending_l914_91422

/-- Given the total spending on souvenirs and the difference between
    key chains & bracelets and t-shirts, proves the amount spent on
    key chains and bracelets. -/
theorem souvenir_spending
  (total : ℚ)
  (difference : ℚ)
  (h1 : total = 548)
  (h2 : difference = 146) :
  let tshirts := (total - difference) / 2
  let keychains_bracelets := tshirts + difference
  keychains_bracelets = 347 := by
sorry

end NUMINAMATH_CALUDE_souvenir_spending_l914_91422


namespace NUMINAMATH_CALUDE_product_inequality_l914_91410

theorem product_inequality (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a + 1) * (b + 1) * (a + c) * (b + c) > 16 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l914_91410


namespace NUMINAMATH_CALUDE_price_restoration_l914_91408

theorem price_restoration (original_price : ℝ) (h : original_price > 0) :
  let reduced_price := 0.8 * original_price
  (reduced_price * 1.25 = original_price) := by
sorry

end NUMINAMATH_CALUDE_price_restoration_l914_91408


namespace NUMINAMATH_CALUDE_hyperbola_symmetric_intersection_l914_91491

/-- The hyperbola and its symmetric curve with respect to a line have common points for all real k -/
theorem hyperbola_symmetric_intersection (k : ℝ) : ∃ (x y : ℝ), 
  (x^2 - y^2 = 1) ∧ 
  (∃ (x' y' : ℝ), (x'^2 - y'^2 = 1) ∧ 
    ((x + x') / 2 = (y + y') / (2*k) - 1/k) ∧
    ((y + y') / 2 = k * ((x + x') / 2) - 1)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_symmetric_intersection_l914_91491


namespace NUMINAMATH_CALUDE_skateboard_distance_l914_91497

/-- The distance traveled by the skateboard in the nth second -/
def distance (n : ℕ) : ℕ := 8 + 9 * (n - 1)

/-- The total distance traveled by the skateboard after n seconds -/
def total_distance (n : ℕ) : ℕ := n * (distance 1 + distance n) / 2

theorem skateboard_distance :
  total_distance 20 = 1870 := by sorry

end NUMINAMATH_CALUDE_skateboard_distance_l914_91497


namespace NUMINAMATH_CALUDE_expression_value_l914_91433

theorem expression_value : 2^2 + (-3)^2 - 7^2 - 2*2*(-3) + 3*7 = -15 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l914_91433


namespace NUMINAMATH_CALUDE_age_ratio_in_two_years_l914_91452

def brother_age : ℕ := 10
def man_age : ℕ := brother_age + 12

theorem age_ratio_in_two_years :
  (man_age + 2) / (brother_age + 2) = 2 := by sorry

end NUMINAMATH_CALUDE_age_ratio_in_two_years_l914_91452


namespace NUMINAMATH_CALUDE_total_rods_for_fence_l914_91465

/-- Represents the types of metal used in the fence. -/
inductive Metal
| A  -- Aluminum
| B  -- Bronze
| C  -- Copper

/-- Represents the components of a fence panel. -/
inductive Component
| Sheet
| Beam

/-- The number of rods needed for each type of metal and component. -/
def rods_needed (m : Metal) (c : Component) : ℕ :=
  match m, c with
  | Metal.A, Component.Sheet => 10
  | Metal.B, Component.Sheet => 8
  | Metal.C, Component.Sheet => 12
  | Metal.A, Component.Beam => 6
  | Metal.B, Component.Beam => 4
  | Metal.C, Component.Beam => 5

/-- Represents a fence pattern. -/
structure Pattern :=
  (a_sheets : ℕ)
  (b_sheets : ℕ)
  (c_sheets : ℕ)
  (a_beams : ℕ)
  (b_beams : ℕ)
  (c_beams : ℕ)

/-- The composition of Pattern X. -/
def pattern_x : Pattern :=
  { a_sheets := 2
  , b_sheets := 1
  , c_sheets := 0
  , a_beams := 0
  , b_beams := 0
  , c_beams := 2 }

/-- The composition of Pattern Y. -/
def pattern_y : Pattern :=
  { a_sheets := 0
  , b_sheets := 2
  , c_sheets := 1
  , a_beams := 3
  , b_beams := 1
  , c_beams := 0 }

/-- Calculate the total number of rods needed for a given pattern and number of panels. -/
def total_rods (p : Pattern) (panels : ℕ) : ℕ :=
  (p.a_sheets * rods_needed Metal.A Component.Sheet +
   p.b_sheets * rods_needed Metal.B Component.Sheet +
   p.c_sheets * rods_needed Metal.C Component.Sheet +
   p.a_beams * rods_needed Metal.A Component.Beam +
   p.b_beams * rods_needed Metal.B Component.Beam +
   p.c_beams * rods_needed Metal.C Component.Beam) * panels

/-- The main theorem stating that the total number of rods needed is 416. -/
theorem total_rods_for_fence : 
  total_rods pattern_x 7 + total_rods pattern_y 3 = 416 := by
  sorry


end NUMINAMATH_CALUDE_total_rods_for_fence_l914_91465


namespace NUMINAMATH_CALUDE_factorization_proof_l914_91476

theorem factorization_proof (x y : ℝ) : x^2 + y^2 + 2*x*y - 1 = (x + y + 1) * (x + y - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l914_91476


namespace NUMINAMATH_CALUDE_max_zombies_after_four_days_l914_91403

/-- The maximum number of zombies in a mall after 4 days of doubling, given initial constraints -/
theorem max_zombies_after_four_days (initial_zombies : ℕ) : 
  initial_zombies < 50 → 
  (initial_zombies * 2^4 : ℕ) ≤ 48 :=
by sorry

end NUMINAMATH_CALUDE_max_zombies_after_four_days_l914_91403


namespace NUMINAMATH_CALUDE_trader_weighted_avg_gain_percentage_l914_91419

/-- Calculates the weighted average gain percentage for a trader selling three types of pens -/
theorem trader_weighted_avg_gain_percentage
  (quantity_A quantity_B quantity_C : ℕ)
  (cost_A cost_B cost_C : ℚ)
  (gain_quantity_A gain_quantity_B gain_quantity_C : ℕ)
  (h_quantity_A : quantity_A = 60)
  (h_quantity_B : quantity_B = 40)
  (h_quantity_C : quantity_C = 50)
  (h_cost_A : cost_A = 2)
  (h_cost_B : cost_B = 3)
  (h_cost_C : cost_C = 4)
  (h_gain_quantity_A : gain_quantity_A = 20)
  (h_gain_quantity_B : gain_quantity_B = 15)
  (h_gain_quantity_C : gain_quantity_C = 10) :
  let total_cost := quantity_A * cost_A + quantity_B * cost_B + quantity_C * cost_C
  let total_gain := gain_quantity_A * cost_A + gain_quantity_B * cost_B + gain_quantity_C * cost_C
  let weighted_avg_gain_percentage := (total_gain / total_cost) * 100
  weighted_avg_gain_percentage = 28.41 := by
  sorry

end NUMINAMATH_CALUDE_trader_weighted_avg_gain_percentage_l914_91419


namespace NUMINAMATH_CALUDE_ellipse_tangent_line_l914_91429

/-- The equation of the tangent line to an ellipse -/
theorem ellipse_tangent_line 
  (a b x₀ y₀ : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_on_ellipse : x₀^2 / a^2 + y₀^2 / b^2 = 1) :
  ∀ x y : ℝ, (x₀ * x / a^2 + y₀ * y / b^2 = 1) ↔ 
    (∃ t : ℝ, x = x₀ + t * (-b^2 * x₀) ∧ y = y₀ + t * (a^2 * y₀) ∧ 
    ∀ u : ℝ, (x₀ + u * (-b^2 * x₀))^2 / a^2 + (y₀ + u * (a^2 * y₀))^2 / b^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_line_l914_91429


namespace NUMINAMATH_CALUDE_cubic_value_l914_91423

theorem cubic_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2010 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_cubic_value_l914_91423


namespace NUMINAMATH_CALUDE_m_range_characterization_l914_91473

def f (x : ℝ) : ℝ := x^2 + 3

theorem m_range_characterization (m : ℝ) : 
  (∀ x ≥ 1, f x + m^2 * f x ≥ f (x - 1) + 3 * f m) ↔ 
  (m ≤ -1 ∨ m ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_m_range_characterization_l914_91473


namespace NUMINAMATH_CALUDE_zoo_animals_count_l914_91426

/-- The number of penguins in the zoo -/
def num_penguins : ℕ := 21

/-- The number of polar bears in the zoo -/
def num_polar_bears : ℕ := 2 * num_penguins

/-- The total number of animals in the zoo -/
def total_animals : ℕ := num_penguins + num_polar_bears

theorem zoo_animals_count : total_animals = 63 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_count_l914_91426


namespace NUMINAMATH_CALUDE_horseshoe_cost_per_set_l914_91489

/-- Proves that the cost per set of horseshoes is $20.75 given the initial outlay,
    selling price, number of sets sold, and profit. -/
theorem horseshoe_cost_per_set 
  (initial_outlay : ℝ)
  (selling_price : ℝ)
  (sets_sold : ℕ)
  (profit : ℝ)
  (h1 : initial_outlay = 12450)
  (h2 : selling_price = 50)
  (h3 : sets_sold = 950)
  (h4 : profit = 15337.5)
  (h5 : profit = selling_price * sets_sold - (initial_outlay + cost_per_set * sets_sold)) :
  cost_per_set = 20.75 :=
by
  sorry

#check horseshoe_cost_per_set

end NUMINAMATH_CALUDE_horseshoe_cost_per_set_l914_91489


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l914_91438

theorem consecutive_even_integers_sum (n : ℤ) : 
  (∃ k : ℤ, n = k^2) →
  (n - 2) + (n + 2) = 162 →
  (n - 2) + n + (n + 2) = 243 := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l914_91438


namespace NUMINAMATH_CALUDE_point_on_line_proof_l914_91417

def point_on_line (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

theorem point_on_line_proof : point_on_line 2 1 10 5 14 7 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_proof_l914_91417


namespace NUMINAMATH_CALUDE_school_population_theorem_l914_91441

theorem school_population_theorem (b g t s : ℕ) :
  b = 4 * g ∧ g = 8 * t ∧ t = 2 * s →
  b + g + t + s = (83 * g) / 16 := by
  sorry

end NUMINAMATH_CALUDE_school_population_theorem_l914_91441


namespace NUMINAMATH_CALUDE_tangent_line_and_minimum_value_and_a_range_l914_91407

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + (1 - x) * Real.exp x

noncomputable def g (a x : ℝ) : ℝ := x - (1 + a) * Real.log x - a / x

theorem tangent_line_and_minimum_value_and_a_range 
  (a : ℝ) 
  (h_a : a < 1) :
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 0, ∃ x₂ ∈ Set.Icc (Real.exp 1) 3, f x₁ > g a x₂) →
  (Real.exp 2 - 2 * Real.exp 1) / (Real.exp 1 + 1) < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_minimum_value_and_a_range_l914_91407


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_under_100_l914_91471

theorem largest_multiple_of_9_under_100 : ∃ n : ℕ, n = 99 ∧ 9 ∣ n ∧ n < 100 ∧ ∀ m : ℕ, 9 ∣ m → m < 100 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_under_100_l914_91471


namespace NUMINAMATH_CALUDE_cubic_sum_divisible_by_nine_l914_91401

theorem cubic_sum_divisible_by_nine (n : ℕ) :
  9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_divisible_by_nine_l914_91401


namespace NUMINAMATH_CALUDE_cos_D_is_zero_l914_91475

-- Define the triangle DEF
structure Triangle (DE EF : ℝ) where
  -- Ensure DE and EF are positive
  de_pos : DE > 0
  ef_pos : EF > 0

-- Define the right triangle DEF with given side lengths
def rightTriangleDEF : Triangle 9 40 where
  de_pos := by norm_num
  ef_pos := by norm_num

-- Theorem: In the right triangle DEF where angle D is 90°, cos D = 0
theorem cos_D_is_zero (t : Triangle 9 40) : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_D_is_zero_l914_91475


namespace NUMINAMATH_CALUDE_equation_solution_l914_91435

theorem equation_solution : 
  ∃! x : ℝ, Real.sqrt x + 3 * Real.sqrt (x^2 + 8*x) + Real.sqrt (x + 8) = 42 - 3*x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_equation_solution_l914_91435


namespace NUMINAMATH_CALUDE_rihanna_remaining_money_l914_91415

/-- Calculates the remaining money after shopping --/
def remaining_money (initial_amount : ℚ) 
  (mango_price : ℚ) (mango_count : ℕ)
  (juice_price : ℚ) (juice_count : ℕ)
  (chips_price : ℚ) (chips_count : ℕ)
  (chocolate_price : ℚ) (chocolate_count : ℕ) : ℚ :=
  initial_amount - 
  (mango_price * mango_count + 
   juice_price * juice_count + 
   chips_price * chips_count + 
   chocolate_price * chocolate_count)

/-- Theorem: Rihanna's remaining money after shopping --/
theorem rihanna_remaining_money : 
  remaining_money 50 3 6 3.5 4 2.25 2 1.75 3 = 8.25 := by
  sorry

end NUMINAMATH_CALUDE_rihanna_remaining_money_l914_91415


namespace NUMINAMATH_CALUDE_min_expression_l914_91447

theorem min_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x * y) / 2 + 18 / (x * y) ≥ 6 ∧
  ((x * y) / 2 + 18 / (x * y) = 6 → y / 2 + x / 3 ≥ 2) ∧
  ((x * y) / 2 + 18 / (x * y) = 6 ∧ y / 2 + x / 3 = 2 → x = 3 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_expression_l914_91447


namespace NUMINAMATH_CALUDE_unique_pair_l914_91460

/-- A function that returns the last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- A function that checks if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A theorem stating that the only pair of positive integers (a, b) satisfying
    all the given conditions is (9, 4) -/
theorem unique_pair : ∀ a b : ℕ+, 
  (lastDigit (a.val + b.val) = 3) →
  (∃ p : ℕ, Nat.Prime p ∧ a.val - b.val = p) →
  isPerfectSquare (a.val * b.val) →
  (a.val = 9 ∧ b.val = 4) ∨ (a.val = 4 ∧ b.val = 9) := by
  sorry

#check unique_pair

end NUMINAMATH_CALUDE_unique_pair_l914_91460


namespace NUMINAMATH_CALUDE_inequality_proof_l914_91414

theorem inequality_proof (t : Real) (h : 0 ≤ t ∧ t ≤ π / 2) :
  Real.sqrt 2 * (Real.sin t + Real.cos t) ≥ 2 * (Real.sin (2 * t))^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l914_91414


namespace NUMINAMATH_CALUDE_bicycle_inventory_problem_l914_91478

/-- Represents the bicycle inventory problem for Hank's store over three days --/
theorem bicycle_inventory_problem 
  (B : ℤ) -- Initial number of bicycles
  (S : ℤ) -- Number of bicycles sold on Friday
  (h1 : S ≥ 0) -- Number of bicycles sold is non-negative
  (h2 : B - S + 15 - 12 + 8 - 9 + 11 = B + 3) -- Net increase equation
  : S = 10 := by
  sorry


end NUMINAMATH_CALUDE_bicycle_inventory_problem_l914_91478


namespace NUMINAMATH_CALUDE_incorrect_equation_simplification_l914_91498

theorem incorrect_equation_simplification (x : ℝ) : 
  (1 / (x + 1) = 2 * x / (3 * x + 3) - 1) ≠ (3 = 2 * x - 3 * x + 3) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_equation_simplification_l914_91498


namespace NUMINAMATH_CALUDE_boutique_hats_count_l914_91484

/-- The total number of hats in the shipment -/
def total_hats : ℕ := 120

/-- The number of hats stored -/
def stored_hats : ℕ := 90

/-- The percentage of hats displayed -/
def displayed_percentage : ℚ := 25 / 100

theorem boutique_hats_count :
  total_hats = stored_hats / (1 - displayed_percentage) := by sorry

end NUMINAMATH_CALUDE_boutique_hats_count_l914_91484


namespace NUMINAMATH_CALUDE_square_diff_sum_eq_three_l914_91400

theorem square_diff_sum_eq_three (a b c : ℤ) 
  (ha : a = 2011) (hb : b = 2012) (hc : c = 2013) : 
  a^2 + b^2 + c^2 - a*b - b*c - a*c = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_sum_eq_three_l914_91400


namespace NUMINAMATH_CALUDE_jackies_tree_climbing_l914_91496

theorem jackies_tree_climbing (h : ℝ) : 
  (1000 + 500 + 500 + h) / 4 = 800 → h - 1000 = 200 := by sorry

end NUMINAMATH_CALUDE_jackies_tree_climbing_l914_91496


namespace NUMINAMATH_CALUDE_purple_marbles_fraction_l914_91464

theorem purple_marbles_fraction (total : ℚ) (h1 : total > 0) : 
  let yellow := (4/7) * total
  let green := (2/7) * total
  let initial_purple := total - yellow - green
  let new_purple := 3 * initial_purple
  let new_total := yellow + green + new_purple
  new_purple / new_total = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_purple_marbles_fraction_l914_91464


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l914_91454

theorem complex_fraction_evaluation :
  2 + (3 / (4 + (5 / 6))) = 76 / 29 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l914_91454


namespace NUMINAMATH_CALUDE_complex_product_modulus_l914_91494

theorem complex_product_modulus : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_modulus_l914_91494


namespace NUMINAMATH_CALUDE_inscribed_circumscribed_quadrilateral_relation_l914_91443

/-- A quadrilateral inscribed in one circle and circumscribed about another -/
structure InscribedCircumscribedQuadrilateral where
  R : ℝ  -- radius of the circumscribed circle
  r : ℝ  -- radius of the inscribed circle
  d : ℝ  -- distance between the centers of the circles
  R_pos : 0 < R
  r_pos : 0 < r
  d_pos : 0 < d
  d_lt_R : d < R

/-- The relationship between R, r, and d for an inscribed-circumscribed quadrilateral -/
theorem inscribed_circumscribed_quadrilateral_relation 
  (q : InscribedCircumscribedQuadrilateral) : 
  1 / (q.R + q.d)^2 + 1 / (q.R - q.d)^2 = 1 / q.r^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circumscribed_quadrilateral_relation_l914_91443


namespace NUMINAMATH_CALUDE_trains_return_to_initial_positions_l914_91442

/-- Represents a train on a circular track -/
structure Train where
  period : ℕ
  position : ℕ

/-- The state of the metro system -/
structure MetroSystem where
  trains : List Train

/-- Calculates the position of a train after a given number of minutes -/
def trainPosition (t : Train) (minutes : ℕ) : ℕ :=
  minutes % t.period

/-- Checks if all trains are at their initial positions -/
def allTrainsAtInitial (ms : MetroSystem) (minutes : ℕ) : Prop :=
  ∀ t ∈ ms.trains, trainPosition t minutes = 0

/-- The main theorem -/
theorem trains_return_to_initial_positions (ms : MetroSystem) : 
  ms.trains = [⟨14, 0⟩, ⟨16, 0⟩, ⟨18, 0⟩] → allTrainsAtInitial ms 2016 := by
  sorry


end NUMINAMATH_CALUDE_trains_return_to_initial_positions_l914_91442
