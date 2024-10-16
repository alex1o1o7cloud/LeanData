import Mathlib

namespace NUMINAMATH_CALUDE_parabola_properties_l837_83784

def parabola (x : ℝ) : ℝ := (x + 2)^2 - 1

theorem parabola_properties :
  (∀ x y : ℝ, parabola x = y → y = (x + 2)^2 - 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → parabola x₁ < parabola x₂) ∧
  (∀ x : ℝ, parabola x ≥ parabola (-2)) ∧
  (parabola (-2) = -1) ∧
  (∀ x₁ x₂ : ℝ, x₁ < -2 ∧ -2 < x₂ → parabola x₁ = parabola x₂ → x₁ + x₂ = -4) ∧
  (∀ x : ℝ, x > -2 → ∀ h : ℝ, h > 0 → parabola (x + h) > parabola x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l837_83784


namespace NUMINAMATH_CALUDE_sin_from_tan_l837_83756

theorem sin_from_tan (a b x : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 < x) (h4 : x < π / 2)
  (h5 : Real.tan x = 2 * a * b / (a^2 - b^2)) : 
  Real.sin x = 2 * a * b / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_sin_from_tan_l837_83756


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_4620_l837_83775

theorem largest_prime_factor_of_4620 : 
  (Nat.factors 4620).maximum? = some 11 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_4620_l837_83775


namespace NUMINAMATH_CALUDE_sum_mod_eleven_l837_83789

theorem sum_mod_eleven : (10555 + 10556 + 10557 + 10558 + 10559) % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_eleven_l837_83789


namespace NUMINAMATH_CALUDE_combined_salaries_BCDE_l837_83703

def salary_A : ℕ := 9000
def average_salary : ℕ := 8200
def num_employees : ℕ := 5

theorem combined_salaries_BCDE :
  salary_A + (num_employees - 1) * (average_salary * num_employees - salary_A) / (num_employees - 1) = average_salary * num_employees :=
by sorry

end NUMINAMATH_CALUDE_combined_salaries_BCDE_l837_83703


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l837_83725

-- Define the functions
def f (a b x : ℝ) : ℝ := -|x - a|^2 + b
def g (c d x : ℝ) : ℝ := |x - c|^2 + d

-- State the theorem
theorem intersection_implies_sum (a b c d : ℝ) :
  f a b 1 = 4 ∧ g c d 1 = 4 ∧ f a b 7 = 2 ∧ g c d 7 = 2 → a + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l837_83725


namespace NUMINAMATH_CALUDE_shop_distance_is_500_l837_83729

/-- Represents the configuration of camps and shop -/
structure CampConfig where
  girls_distance : ℝ  -- perpendicular distance from girls' camp to road
  boys_distance : ℝ   -- distance along road from perpendicular to boys' camp
  shop_distance : ℝ   -- distance from shop to each camp

/-- The shop is equidistant from both camps -/
def is_equidistant (config : CampConfig) : Prop :=
  config.shop_distance^2 = config.girls_distance^2 + (config.shop_distance - config.boys_distance)^2

/-- The theorem stating that given the conditions, the shop is 500 rods from each camp -/
theorem shop_distance_is_500 (config : CampConfig) 
    (h1 : config.girls_distance = 400)
    (h2 : config.boys_distance = 800)
    (h3 : is_equidistant config) : 
  config.shop_distance = 500 := by
  sorry

#check shop_distance_is_500

end NUMINAMATH_CALUDE_shop_distance_is_500_l837_83729


namespace NUMINAMATH_CALUDE_keith_picked_six_apples_l837_83735

/-- The number of apples picked by Mike -/
def mike_apples : ℕ := 7

/-- The number of apples picked by Nancy -/
def nancy_apples : ℕ := 3

/-- The total number of apples picked -/
def total_apples : ℕ := 16

/-- The number of apples picked by Keith -/
def keith_apples : ℕ := total_apples - (mike_apples + nancy_apples)

theorem keith_picked_six_apples : keith_apples = 6 := by
  sorry

end NUMINAMATH_CALUDE_keith_picked_six_apples_l837_83735


namespace NUMINAMATH_CALUDE_polynomial_roots_l837_83769

theorem polynomial_roots : ∃ (p : ℝ → ℝ), 
  (∀ x, p x = 6 * x^4 + 19 * x^3 - 51 * x^2 + 20 * x) ∧ 
  (p 0 = 0) ∧ 
  (p (1/2) = 0) ∧ 
  (p (4/3) = 0) ∧ 
  (p (-5) = 0) := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l837_83769


namespace NUMINAMATH_CALUDE_simplify_expression_l837_83783

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) :
  ((x + y)^2 - y * (2*x + y) - 6*x) / (2*x) = x/2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l837_83783


namespace NUMINAMATH_CALUDE_dow_jones_decrease_l837_83710

theorem dow_jones_decrease (initial_value end_value : ℝ) : 
  (end_value = initial_value * 0.98) → 
  (end_value = 8722) → 
  (initial_value = 8900) := by
sorry

end NUMINAMATH_CALUDE_dow_jones_decrease_l837_83710


namespace NUMINAMATH_CALUDE_d_value_for_four_roots_l837_83711

/-- The polynomial Q(x) -/
def Q (d : ℝ) (x : ℝ) : ℝ := (x^2 - 3*x + 5) * (x^2 - d*x + 7) * (x^2 - 6*x + 18)

/-- The number of distinct roots of Q(x) -/
def distinctRoots (d : ℝ) : ℕ := sorry

/-- Theorem stating that |d| = 9 when Q(x) has exactly 4 distinct roots -/
theorem d_value_for_four_roots :
  ∃ d : ℝ, distinctRoots d = 4 ∧ |d| = 9 := by sorry

end NUMINAMATH_CALUDE_d_value_for_four_roots_l837_83711


namespace NUMINAMATH_CALUDE_linear_equation_solution_l837_83772

theorem linear_equation_solution (x y a : ℝ) : 
  x = 1 → y = 2 → x - a * y = 3 → a = -1 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l837_83772


namespace NUMINAMATH_CALUDE_soccer_league_games_l837_83743

/-- The number of games played in a soccer league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem: In a league of 12 teams, where each team plays 4 games with every other team,
    the total number of games played is 264. -/
theorem soccer_league_games :
  total_games 12 4 = 264 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l837_83743


namespace NUMINAMATH_CALUDE_triangle_ratio_l837_83778

/-- Triangle PQR with angle bisector PS intersecting MN at X -/
structure Triangle (P Q R S M N X : ℝ × ℝ) : Prop where
  m_on_pq : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • P + t • Q
  n_on_pr : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ N = (1 - t) • P + t • R
  ps_bisector : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (1 - t) • P + t • ((2/3) • Q + (1/3) • R)
  x_on_mn : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (1 - t) • M + t • N
  x_on_ps : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (1 - t) • P + t • S

/-- Given lengths in the triangle -/
structure TriangleLengths (P Q R S M N X : ℝ × ℝ) : Prop where
  pm_eq : ‖M - P‖ = 2
  mq_eq : ‖Q - M‖ = 6
  pn_eq : ‖N - P‖ = 3
  nr_eq : ‖R - N‖ = 9

/-- The main theorem -/
theorem triangle_ratio 
  (P Q R S M N X : ℝ × ℝ) 
  (h1 : Triangle P Q R S M N X) 
  (h2 : TriangleLengths P Q R S M N X) : 
  ‖X - P‖ / ‖S - P‖ = 1/4 :=
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l837_83778


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_divisibility_l837_83737

theorem consecutive_odd_numbers_divisibility (a b c : ℤ) : 
  (∃ k : ℤ, b = 2 * k + 1) →  -- b is odd
  (a = b - 2) →              -- a is the previous odd number
  (c = b + 2) →              -- c is the next odd number
  ∃ m : ℤ, a * b * c + 4 * b = m * b^3 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_divisibility_l837_83737


namespace NUMINAMATH_CALUDE_son_dad_age_ratio_l837_83719

theorem son_dad_age_ratio : 
  ∀ (dad_age son_age : ℕ),
    dad_age = 36 →
    son_age = 9 →
    dad_age - son_age = 27 →
    dad_age / son_age = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_son_dad_age_ratio_l837_83719


namespace NUMINAMATH_CALUDE_bowling_team_weight_l837_83704

/-- Given a bowling team with the following properties:
  * 7 original players
  * Original average weight of 103 kg
  * 2 new players join
  * One new player weighs 60 kg
  * New average weight is 99 kg
  Prove that the other new player weighs 110 kg -/
theorem bowling_team_weight (original_players : ℕ) (original_avg : ℝ) 
  (new_players : ℕ) (known_new_weight : ℝ) (new_avg : ℝ) :
  original_players = 7 ∧ 
  original_avg = 103 ∧ 
  new_players = 2 ∧ 
  known_new_weight = 60 ∧ 
  new_avg = 99 →
  ∃ x : ℝ, x = 110 ∧ 
    (original_players * original_avg + known_new_weight + x) / 
    (original_players + new_players) = new_avg :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_weight_l837_83704


namespace NUMINAMATH_CALUDE_least_b_is_five_l837_83731

/-- A triangle with angles a, b, c in degrees, where a, b, c are prime numbers and a > b > c -/
structure PrimeAngleTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  a_prime : Nat.Prime a
  b_prime : Nat.Prime b
  c_prime : Nat.Prime c
  angle_sum : a + b + c = 180
  a_gt_b : a > b
  b_gt_c : b > c
  not_right : a ≠ 90 ∧ b ≠ 90 ∧ c ≠ 90

/-- The least possible value of b in a PrimeAngleTriangle is 5 -/
theorem least_b_is_five (t : PrimeAngleTriangle) : t.b ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_least_b_is_five_l837_83731


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l837_83780

theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 5) (hb : b = 12) (hc : c = 15) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 20 / 19 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l837_83780


namespace NUMINAMATH_CALUDE_sin_315_degrees_l837_83734

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l837_83734


namespace NUMINAMATH_CALUDE_sequence_representation_l837_83760

theorem sequence_representation (q : ℕ → ℕ) 
  (h_increasing : ∀ n, q n < q (n + 1))
  (h_bound : ∀ n, q n < 2 * n) :
  ∀ m : ℕ, ∃ k l : ℕ, q k = m ∨ q l - q k = m :=
sorry

end NUMINAMATH_CALUDE_sequence_representation_l837_83760


namespace NUMINAMATH_CALUDE_tara_dad_attendance_l837_83732

/-- The number of games Tara played each year -/
def games_per_year : ℕ := 20

/-- The number of games Tara's dad attended in the second year -/
def games_attended_second_year : ℕ := 14

/-- The difference in games attended between the first and second year -/
def games_difference : ℕ := 4

/-- The percentage of games Tara's dad attended in the first year -/
def attendance_percentage : ℚ := 90

theorem tara_dad_attendance :
  (games_attended_second_year + games_difference) / games_per_year * 100 = attendance_percentage := by
  sorry

end NUMINAMATH_CALUDE_tara_dad_attendance_l837_83732


namespace NUMINAMATH_CALUDE_power_function_sum_l837_83788

/-- A function f is a power function if it can be written as f(x) = k * x^c, 
    where k and c are constants, and c is not zero. -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (k c : ℝ), c ≠ 0 ∧ ∀ x, f x = k * x^c

/-- Given that f(x) = a*x^(2a+1) - b + 1 is a power function, prove that a + b = 2 -/
theorem power_function_sum (a b : ℝ) :
  isPowerFunction (fun x ↦ a * x^(2*a+1) - b + 1) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_sum_l837_83788


namespace NUMINAMATH_CALUDE_equation_solution_l837_83762

theorem equation_solution : 
  ∃! x : ℚ, (x - 15) / 3 = (3 * x + 11) / 8 :=
by
  use -153
  sorry

end NUMINAMATH_CALUDE_equation_solution_l837_83762


namespace NUMINAMATH_CALUDE_sin_shift_l837_83706

theorem sin_shift (x : ℝ) : Real.sin (x + π/3) = Real.sin (x + π/3) := by sorry

end NUMINAMATH_CALUDE_sin_shift_l837_83706


namespace NUMINAMATH_CALUDE_valid_starting_days_count_l837_83748

/-- Represents the days of the week -/
inductive DayOfWeek
  | sunday
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.sunday => DayOfWeek.monday
  | DayOfWeek.monday => DayOfWeek.tuesday
  | DayOfWeek.tuesday => DayOfWeek.wednesday
  | DayOfWeek.wednesday => DayOfWeek.thursday
  | DayOfWeek.thursday => DayOfWeek.friday
  | DayOfWeek.friday => DayOfWeek.saturday
  | DayOfWeek.saturday => DayOfWeek.sunday

/-- Counts the number of Tuesdays and Fridays in a 30-day month starting from a given day -/
def countTuesdaysAndFridays (startDay : DayOfWeek) : (Nat × Nat) :=
  let rec count (currentDay : DayOfWeek) (daysLeft : Nat) (tuesdays : Nat) (fridays : Nat) : (Nat × Nat) :=
    if daysLeft = 0 then
      (tuesdays, fridays)
    else
      match currentDay with
      | DayOfWeek.tuesday => count (nextDay currentDay) (daysLeft - 1) (tuesdays + 1) fridays
      | DayOfWeek.friday => count (nextDay currentDay) (daysLeft - 1) tuesdays (fridays + 1)
      | _ => count (nextDay currentDay) (daysLeft - 1) tuesdays fridays
  count startDay 30 0 0

/-- Checks if the number of Tuesdays equals the number of Fridays for a given starting day -/
def hasSameTuesdaysAndFridays (startDay : DayOfWeek) : Bool :=
  let (tuesdays, fridays) := countTuesdaysAndFridays startDay
  tuesdays = fridays

/-- Counts the number of valid starting days -/
def countValidStartingDays : Nat :=
  let allDays := [DayOfWeek.sunday, DayOfWeek.monday, DayOfWeek.tuesday, DayOfWeek.wednesday,
                  DayOfWeek.thursday, DayOfWeek.friday, DayOfWeek.saturday]
  allDays.filter hasSameTuesdaysAndFridays |>.length

/-- The main theorem to prove -/
theorem valid_starting_days_count :
  countValidStartingDays = 3 := by
  sorry


end NUMINAMATH_CALUDE_valid_starting_days_count_l837_83748


namespace NUMINAMATH_CALUDE_beth_graphic_novels_l837_83728

theorem beth_graphic_novels (total : ℕ) (novel_percent : ℚ) (comic_percent : ℚ) 
  (h_total : total = 120)
  (h_novel : novel_percent = 65 / 100)
  (h_comic : comic_percent = 20 / 100) :
  total - (novel_percent * total).floor - (comic_percent * total).floor = 18 := by
  sorry

end NUMINAMATH_CALUDE_beth_graphic_novels_l837_83728


namespace NUMINAMATH_CALUDE_initial_cards_l837_83736

theorem initial_cards (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 3 → total = 7 → initial + added = total → initial = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_cards_l837_83736


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l837_83779

def M : Set ℝ := {x | x < (1 : ℝ) / 2}
def N : Set ℝ := {x | x ≥ -4}

theorem intersection_of_M_and_N : M ∩ N = {x | -4 ≤ x ∧ x < (1 : ℝ) / 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l837_83779


namespace NUMINAMATH_CALUDE_lcm_problem_l837_83733

theorem lcm_problem (a b c : ℕ) (h1 : Nat.lcm a b = 16) (h2 : Nat.lcm b c = 21) :
  Nat.lcm a c ≥ 336 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l837_83733


namespace NUMINAMATH_CALUDE_sum_mod_nine_l837_83765

theorem sum_mod_nine : (9156 + 9157 + 9158 + 9159 + 9160) % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l837_83765


namespace NUMINAMATH_CALUDE_inequality_proof_l837_83761

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = 1/2) :
  1/(1-a) + 1/(1-b) ≥ 4 ∧ (1/(1-a) + 1/(1-b) = 4 ↔ a = 1/2 ∧ b = 1/2) := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l837_83761


namespace NUMINAMATH_CALUDE_average_hamburgers_per_day_l837_83702

-- Define the total number of hamburgers sold
def total_hamburgers : ℕ := 49

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the average number of hamburgers sold per day
def average_hamburgers : ℚ := total_hamburgers / days_in_week

-- Theorem statement
theorem average_hamburgers_per_day :
  average_hamburgers = 7 := by sorry

end NUMINAMATH_CALUDE_average_hamburgers_per_day_l837_83702


namespace NUMINAMATH_CALUDE_x_value_proof_l837_83751

theorem x_value_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h1 : x^2 / y = 2)
  (h2 : y^2 / z = 5)
  (h3 : z^2 / x = 7) : 
  x = (2800 : ℝ)^(1/7) := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l837_83751


namespace NUMINAMATH_CALUDE_eligible_age_range_max_different_ages_l837_83744

def average_age : ℕ := 31
def standard_deviation : ℕ := 5
def bachelor_degree_age : ℕ := 22

def lower_age_limit : ℕ := average_age - standard_deviation
def upper_age_limit : ℕ := average_age + standard_deviation

theorem eligible_age_range : ℕ :=
  upper_age_limit - lower_age_limit + 1

theorem max_different_ages (h : bachelor_degree_age ≤ lower_age_limit) : 
  eligible_age_range = 11 := by
  sorry

end NUMINAMATH_CALUDE_eligible_age_range_max_different_ages_l837_83744


namespace NUMINAMATH_CALUDE_last_two_digits_of_7_pow_2017_l837_83757

-- Define the pattern of last two digits
def lastTwoDigitsPattern : Fin 4 → Nat
  | 0 => 49
  | 1 => 43
  | 2 => 01
  | 3 => 07

-- Define the function to get the last two digits of 7^n
def lastTwoDigits (n : Nat) : Nat :=
  lastTwoDigitsPattern ((n - 2) % 4)

-- Theorem statement
theorem last_two_digits_of_7_pow_2017 :
  lastTwoDigits 2017 = 07 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_7_pow_2017_l837_83757


namespace NUMINAMATH_CALUDE_workshop_workers_count_l837_83754

/-- The total number of workers in a workshop given specific salary conditions -/
theorem workshop_workers_count : ℕ :=
  let average_salary : ℚ := 1000
  let technician_salary : ℚ := 1200
  let other_salary : ℚ := 820
  let technician_count : ℕ := 10
  let total_workers : ℕ := 21

  have h1 : average_salary * total_workers = 
    technician_salary * technician_count + other_salary * (total_workers - technician_count) := by sorry

  total_workers


end NUMINAMATH_CALUDE_workshop_workers_count_l837_83754


namespace NUMINAMATH_CALUDE_square_areas_product_equality_l837_83742

theorem square_areas_product_equality (α : Real) : 
  (Real.cos α)^4 * (Real.sin α)^4 = ((Real.cos α)^2 * (Real.sin α)^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_areas_product_equality_l837_83742


namespace NUMINAMATH_CALUDE_students_per_grade_l837_83767

theorem students_per_grade (total_students : ℕ) (total_grades : ℕ) 
  (h1 : total_students = 22800) 
  (h2 : total_grades = 304) : 
  total_students / total_grades = 75 := by
  sorry

end NUMINAMATH_CALUDE_students_per_grade_l837_83767


namespace NUMINAMATH_CALUDE_complex_number_problem_l837_83739

/-- Given a complex number z where z + 2i and z / (2 - i) are real numbers, 
    z = 4 - 2i and (z + ai)² is in the first quadrant when 2 < a < 6 -/
theorem complex_number_problem (z : ℂ) 
  (h1 : (z + 2*Complex.I).im = 0)
  (h2 : (z / (2 - Complex.I)).im = 0) :
  z = 4 - 2*Complex.I ∧ 
  ∀ a : ℝ, (z + a*Complex.I)^2 ∈ {w : ℂ | w.re > 0 ∧ w.im > 0} ↔ 2 < a ∧ a < 6 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l837_83739


namespace NUMINAMATH_CALUDE_subtracted_amount_l837_83771

theorem subtracted_amount (N : ℝ) (A : ℝ) : 
  N = 300 → 0.30 * N - A = 20 → A = 70 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_amount_l837_83771


namespace NUMINAMATH_CALUDE_product_xyz_is_zero_l837_83750

theorem product_xyz_is_zero 
  (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 1) 
  (h3 : y ≠ 0) 
  (h4 : z ≠ 0) : 
  x * y * z = 0 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_is_zero_l837_83750


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l837_83776

theorem quadratic_root_problem (a : ℝ) : 
  (2^2 + 2 - a = 0) → 
  (∃ x : ℝ, x ≠ 2 ∧ x^2 + x - a = 0) → 
  ((-3)^2 + (-3) - a = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l837_83776


namespace NUMINAMATH_CALUDE_recurring_decimal_fraction_sum_l837_83749

theorem recurring_decimal_fraction_sum (a b : ℕ+) :
  (a : ℚ) / (b : ℚ) = 36 / 99 →
  Nat.gcd a b = 1 →
  a + b = 15 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_fraction_sum_l837_83749


namespace NUMINAMATH_CALUDE_vector_equation_solution_l837_83714

theorem vector_equation_solution (a b : ℝ × ℝ) (m n : ℝ) : 
  a = (2, 1) → b = (1, -2) → m • a + n • b = (9, -8) → m - n = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l837_83714


namespace NUMINAMATH_CALUDE_soldier_hit_target_l837_83730

theorem soldier_hit_target (p q : Prop) : 
  (p ∨ q) ↔ (∃ shot : Fin 2, shot.val = 0 ∧ p ∨ shot.val = 1 ∧ q) :=
by sorry

end NUMINAMATH_CALUDE_soldier_hit_target_l837_83730


namespace NUMINAMATH_CALUDE_problem_solution_l837_83715

/-- The function f(x) as defined in the problem -/
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := (1/2) * (t * Real.log (x + 2) - Real.log (x - 2))

/-- The function F(x) as defined in the problem -/
noncomputable def F (a : ℝ) (t : ℝ) (x : ℝ) : ℝ := a * Real.log (x - 1) - f t x

/-- Theorem stating the main results of the problem -/
theorem problem_solution :
  ∃ (t : ℝ),
    (∀ x : ℝ, f t x ≥ f t 4) ∧
    (t = 3) ∧
    (∀ x ∈ Set.Icc 3 7, f t x ≤ f t 7) ∧
    (∀ a : ℝ, (∀ x > 2, Monotone (F a t)) ↔ a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l837_83715


namespace NUMINAMATH_CALUDE_stock_price_example_l837_83758

/-- Given a stock with income, dividend rate, and investment amount, calculate its price. -/
def stock_price (income : ℚ) (dividend_rate : ℚ) (investment : ℚ) : ℚ :=
  let face_value := (income * 100) / dividend_rate
  (investment / face_value) * 100

/-- Theorem: The price of a stock with income Rs. 650, 10% dividend rate, and Rs. 6240 investment is Rs. 96. -/
theorem stock_price_example : stock_price 650 10 6240 = 96 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_example_l837_83758


namespace NUMINAMATH_CALUDE_olivia_baseball_cards_l837_83718

/-- The number of decks of baseball cards Olivia bought -/
def baseball_decks : ℕ :=
  let basketball_packs : ℕ := 2
  let basketball_price : ℕ := 3
  let baseball_price : ℕ := 4
  let initial_money : ℕ := 50
  let change : ℕ := 24
  let total_spent : ℕ := initial_money - change
  let basketball_cost : ℕ := basketball_packs * basketball_price
  let baseball_cost : ℕ := total_spent - basketball_cost
  baseball_cost / baseball_price

theorem olivia_baseball_cards : baseball_decks = 5 := by
  sorry

end NUMINAMATH_CALUDE_olivia_baseball_cards_l837_83718


namespace NUMINAMATH_CALUDE_sqrt_200_simplification_l837_83792

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_simplification_l837_83792


namespace NUMINAMATH_CALUDE_initial_rulers_l837_83797

theorem initial_rulers (taken : ℕ) (remaining : ℕ) : taken = 25 → remaining = 21 → taken + remaining = 46 := by
  sorry

end NUMINAMATH_CALUDE_initial_rulers_l837_83797


namespace NUMINAMATH_CALUDE_smallest_three_digit_number_l837_83705

def digits : Finset Nat := {3, 0, 2, 5, 7}

def isValidNumber (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ 
  (Finset.card (Finset.filter (λ d => d ∈ digits) (Finset.image (λ i => (n / (10^i)) % 10) {0, 1, 2})) = 3)

def smallestValidNumber : Nat := 203

theorem smallest_three_digit_number :
  (isValidNumber smallestValidNumber) ∧
  (∀ n : Nat, isValidNumber n → n ≥ smallestValidNumber) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_number_l837_83705


namespace NUMINAMATH_CALUDE_min_value_quadratic_l837_83717

theorem min_value_quadratic (x y : ℝ) : 2 * x^2 + 3 * y^2 - 8 * x + 6 * y + 25 ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l837_83717


namespace NUMINAMATH_CALUDE_field_trip_girls_fraction_l837_83747

theorem field_trip_girls_fraction (total_students : ℕ) (h_total_positive : total_students > 0) :
  let total_girls : ℕ := total_students / 2
  let total_boys : ℕ := total_students / 2
  let girls_on_trip : ℚ := (4 : ℚ) / 5 * total_girls
  let boys_on_trip : ℚ := (3 : ℚ) / 4 * total_boys
  let total_on_trip : ℚ := girls_on_trip + boys_on_trip
  (girls_on_trip / total_on_trip) = (16 : ℚ) / 31 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_girls_fraction_l837_83747


namespace NUMINAMATH_CALUDE_students_per_group_l837_83798

theorem students_per_group 
  (total_students : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 30) 
  (h2 : num_groups = 6) 
  (h3 : total_students % num_groups = 0) :
  total_students / num_groups = 5 := by
sorry

end NUMINAMATH_CALUDE_students_per_group_l837_83798


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l837_83738

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_properties :
  ∀ d : ℤ,
  (arithmetic_sequence 23 d 6 > 0) →
  (arithmetic_sequence 23 d 7 < 0) →
  (d = -4) ∧
  (∀ n : ℕ, sum_arithmetic_sequence 23 d n ≤ 78) ∧
  (∀ n : ℕ, n ≤ 12 ↔ sum_arithmetic_sequence 23 d n > 0) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l837_83738


namespace NUMINAMATH_CALUDE_group_size_calculation_l837_83720

theorem group_size_calculation (n : ℕ) : 
  (n : ℝ) * 14 = n * 14 →                   -- Initial average age
  ((n : ℝ) * 14 + 32) / (n + 1) = 16 →      -- New average age
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_group_size_calculation_l837_83720


namespace NUMINAMATH_CALUDE_seven_eighths_of_48_l837_83774

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_48_l837_83774


namespace NUMINAMATH_CALUDE_concentric_circles_chords_l837_83795

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle between two adjacent chords is 60°, then the minimum number of such chords
    needed to complete a full circle is 3. -/
theorem concentric_circles_chords (angle_between_chords : ℝ) (n : ℕ) :
  angle_between_chords = 60 →
  (n : ℝ) * (180 - angle_between_chords) = 360 →
  n = 3 :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_chords_l837_83795


namespace NUMINAMATH_CALUDE_sixth_student_matches_l837_83770

/-- Represents the number of matches played by each student -/
structure MatchCounts where
  student1 : ℕ
  student2 : ℕ
  student3 : ℕ
  student4 : ℕ
  student5 : ℕ
  student6 : ℕ

/-- The total number of matches in a complete tournament with 6 players -/
def totalMatches : ℕ := 15

/-- Theorem stating that if 5 students have played 5, 4, 3, 2, and 1 matches respectively,
    then the 6th student must have played 3 matches -/
theorem sixth_student_matches (mc : MatchCounts) : 
  mc.student1 = 5 ∧ 
  mc.student2 = 4 ∧ 
  mc.student3 = 3 ∧ 
  mc.student4 = 2 ∧ 
  mc.student5 = 1 ∧
  (mc.student1 + mc.student2 + mc.student3 + mc.student4 + mc.student5 + mc.student6 = 2 * totalMatches) →
  mc.student6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sixth_student_matches_l837_83770


namespace NUMINAMATH_CALUDE_basketball_non_gymnastics_percentage_l837_83700

theorem basketball_non_gymnastics_percentage 
  (total : ℝ)
  (h_total_pos : total > 0)
  (h_basketball : total * (50 / 100) = total * 0.5)
  (h_gymnastics : total * (40 / 100) = total * 0.4)
  (h_both : (total * 0.5) * (30 / 100) = total * 0.15) :
  let non_gymnastics := total * 0.6
  let basketball_non_gymnastics := total * 0.35
  (basketball_non_gymnastics / non_gymnastics) * 100 = 58 := by
sorry

end NUMINAMATH_CALUDE_basketball_non_gymnastics_percentage_l837_83700


namespace NUMINAMATH_CALUDE_farm_hens_count_l837_83708

/-- Given a farm with roosters and hens, where the number of hens is 5 less than 9 times
    the number of roosters, and the total number of chickens is 75, prove that there are 67 hens. -/
theorem farm_hens_count (roosters hens : ℕ) : 
  hens = 9 * roosters - 5 →
  hens + roosters = 75 →
  hens = 67 := by
sorry

end NUMINAMATH_CALUDE_farm_hens_count_l837_83708


namespace NUMINAMATH_CALUDE_tens_digit_of_13_pow_1997_l837_83773

theorem tens_digit_of_13_pow_1997 :
  13^1997 % 100 = 53 := by sorry

end NUMINAMATH_CALUDE_tens_digit_of_13_pow_1997_l837_83773


namespace NUMINAMATH_CALUDE_power_two_plus_one_div_by_three_l837_83752

theorem power_two_plus_one_div_by_three (n : ℕ) : 
  3 ∣ (2^n + 1) ↔ n % 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_power_two_plus_one_div_by_three_l837_83752


namespace NUMINAMATH_CALUDE_simplify_product_of_square_roots_l837_83741

theorem simplify_product_of_square_roots (x : ℝ) (h : x > 0) :
  Real.sqrt (5 * 2 * x) * Real.sqrt (x^3 * 5^3) = 25 * x^2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_square_roots_l837_83741


namespace NUMINAMATH_CALUDE_calculate_first_train_length_l837_83753

/-- The length of the first train given the specified conditions -/
def first_train_length (first_train_speed second_train_speed : ℝ)
                       (second_train_length : ℝ)
                       (crossing_time : ℝ) : ℝ :=
  (first_train_speed - second_train_speed) * crossing_time - second_train_length

/-- Theorem stating the length of the first train under given conditions -/
theorem calculate_first_train_length :
  first_train_length 72 36 300 69.99440044796417 = 399.9440044796417 := by
  sorry

end NUMINAMATH_CALUDE_calculate_first_train_length_l837_83753


namespace NUMINAMATH_CALUDE_centroid_triangle_area_l837_83793

-- Define the rectangle ABCD
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define point E on side CD
def E (rect : Rectangle) : ℝ × ℝ :=
  sorry

-- Define the area of a triangle
def triangleArea (p q r : ℝ × ℝ) : ℝ :=
  sorry

-- Define the centroid of a triangle
def centroid (p q r : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem centroid_triangle_area (rect : Rectangle) :
  let G₁ := centroid rect.A rect.D (E rect)
  let G₂ := centroid rect.A rect.B (E rect)
  let G₃ := centroid rect.B rect.C (E rect)
  triangleArea G₁ G₂ G₃ = 1/18 :=
by
  sorry

end NUMINAMATH_CALUDE_centroid_triangle_area_l837_83793


namespace NUMINAMATH_CALUDE_soda_cost_calculation_l837_83786

/-- The cost of a single soda, given the total cost of sandwiches and sodas, and the cost of a single sandwich. -/
def soda_cost (total_cost sandwich_cost : ℚ) : ℚ :=
  (total_cost - 2 * sandwich_cost) / 4

theorem soda_cost_calculation (total_cost sandwich_cost : ℚ) 
  (h1 : total_cost = (8.36 : ℚ))
  (h2 : sandwich_cost = (2.44 : ℚ)) :
  soda_cost total_cost sandwich_cost = (0.87 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_calculation_l837_83786


namespace NUMINAMATH_CALUDE_impossible_grid_arrangement_l837_83726

/-- A type representing a 6x7 grid of natural numbers -/
def Grid := Fin 6 → Fin 7 → ℕ

/-- Predicate to check if a grid contains all numbers from 1 to 42 exactly once -/
def contains_all_numbers (g : Grid) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 42 → ∃! (i : Fin 6) (j : Fin 7), g i j = n

/-- Predicate to check if all vertical 1x2 rectangles in a grid have even sum -/
def all_vertical_sums_even (g : Grid) : Prop :=
  ∀ (i : Fin 5) (j : Fin 7), Even (g i j + g (i.succ) j)

/-- Theorem stating the impossibility of the desired grid arrangement -/
theorem impossible_grid_arrangement : 
  ¬ ∃ (g : Grid), contains_all_numbers g ∧ all_vertical_sums_even g :=
sorry

end NUMINAMATH_CALUDE_impossible_grid_arrangement_l837_83726


namespace NUMINAMATH_CALUDE_first_day_over_500_l837_83764

/-- Represents the number of markers Liam has on a given day -/
def markers (day : ℕ) : ℕ :=
  if day = 1 then 5
  else if day = 2 then 10
  else 5 * 3^(day - 2)

/-- The day of the week as a number from 1 to 7 -/
def dayOfWeek (day : ℕ) : ℕ :=
  (day - 1) % 7 + 1

theorem first_day_over_500 :
  ∃ d : ℕ, markers d > 500 ∧ 
    ∀ k < d, markers k ≤ 500 ∧
    dayOfWeek d = 6 :=
  sorry

end NUMINAMATH_CALUDE_first_day_over_500_l837_83764


namespace NUMINAMATH_CALUDE_range_of_a_minus_abs_b_l837_83713

theorem range_of_a_minus_abs_b (a b : ℝ) :
  1 < a ∧ a < 8 ∧ -4 < b ∧ b < 2 →
  ∃ x, -3 < x ∧ x < 8 ∧ x = a - |b| :=
sorry

end NUMINAMATH_CALUDE_range_of_a_minus_abs_b_l837_83713


namespace NUMINAMATH_CALUDE_divisibility_condition_l837_83745

/-- Sum of divisors of n -/
def A (n : ℕ+) : ℕ := sorry

/-- Sum of products of pairs of divisors of n -/
def B (n : ℕ+) : ℕ := sorry

/-- A positive integer n is a perfect square -/
def is_perfect_square (n : ℕ+) : Prop := ∃ m : ℕ+, n = m ^ 2

theorem divisibility_condition (n : ℕ+) : 
  (A n ∣ B n) ↔ is_perfect_square n := by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l837_83745


namespace NUMINAMATH_CALUDE_f_composition_negative_three_l837_83781

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 1 / Real.sqrt x else x^2

-- State the theorem
theorem f_composition_negative_three : f (f (-3)) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_three_l837_83781


namespace NUMINAMATH_CALUDE_meatballs_cost_is_five_l837_83723

/-- A dinner consisting of pasta, sauce, and meatballs -/
structure Dinner where
  total_cost : ℝ
  pasta_cost : ℝ
  sauce_cost : ℝ
  meatballs_cost : ℝ

/-- The cost of the dinner components add up to the total cost -/
def cost_sum (d : Dinner) : Prop :=
  d.total_cost = d.pasta_cost + d.sauce_cost + d.meatballs_cost

/-- Theorem: Given the total cost, pasta cost, and sauce cost, 
    prove that the meatballs cost $5 -/
theorem meatballs_cost_is_five (d : Dinner) 
  (h1 : d.total_cost = 8)
  (h2 : d.pasta_cost = 1)
  (h3 : d.sauce_cost = 2)
  (h4 : cost_sum d) : 
  d.meatballs_cost = 5 := by
  sorry


end NUMINAMATH_CALUDE_meatballs_cost_is_five_l837_83723


namespace NUMINAMATH_CALUDE_trig_inequality_l837_83785

theorem trig_inequality (a b c x : ℝ) :
  -(Real.sin (π/4 - (b-c)/2))^2 ≤ Real.sin (a*x + b) * Real.cos (a*x + c) ∧
  Real.sin (a*x + b) * Real.cos (a*x + c) ≤ (Real.cos (π/4 - (b-c)/2))^2 := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l837_83785


namespace NUMINAMATH_CALUDE_det_special_matrix_l837_83790

-- Define the matrix as a function of y
def matrix (y : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![y + 1, y, y],
    ![y, y + 1, y],
    ![y, y, y + 1]]

-- State the theorem
theorem det_special_matrix (y : ℝ) :
  Matrix.det (matrix y) = 3 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l837_83790


namespace NUMINAMATH_CALUDE_simplify_expression_l837_83763

theorem simplify_expression : 3000 * (3000 ^ 3000) + 3000 * (3000 ^ 3000) = 2 * 3000 ^ 3001 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l837_83763


namespace NUMINAMATH_CALUDE_marias_trip_l837_83794

theorem marias_trip (total_distance : ℝ) (remaining_distance : ℝ) 
  (h1 : total_distance = 360)
  (h2 : remaining_distance = 135)
  (h3 : remaining_distance = total_distance - (x * total_distance + 1/4 * (total_distance - x * total_distance)))
  : x = 1/2 :=
by
  sorry

#check marias_trip

end NUMINAMATH_CALUDE_marias_trip_l837_83794


namespace NUMINAMATH_CALUDE_sum_geq_three_l837_83727

theorem sum_geq_three (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a + 1) * (b + 1) * (c + 1) = 8) : a + b + c ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_three_l837_83727


namespace NUMINAMATH_CALUDE_product_of_numbers_l837_83796

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 42) (h2 : |x - y| = 4) : x * y = 437 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l837_83796


namespace NUMINAMATH_CALUDE_meeting_point_coordinates_l837_83716

/-- The point two-thirds of the way from one point to another -/
def two_thirds_point (x₁ y₁ x₂ y₂ : ℚ) : ℚ × ℚ :=
  (x₁ + 2/3 * (x₂ - x₁), y₁ + 2/3 * (y₂ - y₁))

/-- Prove that the meeting point is at (14/3, 11/3) -/
theorem meeting_point_coordinates :
  two_thirds_point 10 (-3) 2 7 = (14/3, 11/3) := by
  sorry

#check meeting_point_coordinates

end NUMINAMATH_CALUDE_meeting_point_coordinates_l837_83716


namespace NUMINAMATH_CALUDE_smallest_k_is_2010_l837_83709

/-- A sequence of natural numbers satisfying the given conditions -/
def ValidSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (∀ n, 1005 ∣ a n ∨ 1006 ∣ a n) ∧
  (∀ n, ¬(97 ∣ a n))

/-- The difference between consecutive terms is at most k -/
def BoundedDifference (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n, a (n + 1) - a n ≤ k

/-- The theorem stating the smallest possible k -/
theorem smallest_k_is_2010 :
  (∃ a, ValidSequence a ∧ BoundedDifference a 2010) ∧
  (∀ k < 2010, ¬∃ a, ValidSequence a ∧ BoundedDifference a k) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_is_2010_l837_83709


namespace NUMINAMATH_CALUDE_james_waiting_period_l837_83721

/-- Represents the timeline of James' injury and recovery process -/
structure InjuryTimeline where
  pain_duration : ℕ
  healing_multiplier : ℕ
  additional_wait : ℕ
  total_time : ℕ

/-- Calculates the number of days James waited to start working out after full healing -/
def waiting_period (timeline : InjuryTimeline) : ℕ :=
  timeline.total_time - (timeline.pain_duration * timeline.healing_multiplier) - (timeline.additional_wait * 7)

/-- Theorem stating that James waited 3 days to start working out after full healing -/
theorem james_waiting_period :
  let timeline : InjuryTimeline := {
    pain_duration := 3,
    healing_multiplier := 5,
    additional_wait := 3,
    total_time := 39
  }
  waiting_period timeline = 3 := by sorry

end NUMINAMATH_CALUDE_james_waiting_period_l837_83721


namespace NUMINAMATH_CALUDE_supremum_inequality_l837_83712

theorem supremum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  - (1 / (2 * a)) - (2 / b) ≤ - (9 / 2) ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ - (1 / (2 * a₀)) - (2 / b₀) = - (9 / 2) :=
sorry

end NUMINAMATH_CALUDE_supremum_inequality_l837_83712


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l837_83768

theorem line_segment_endpoint (x : ℝ) : 
  x < 0 → 
  (x - 1)^2 + (8 - 3)^2 = 15^2 → 
  x = 1 - 10 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l837_83768


namespace NUMINAMATH_CALUDE_circle_radius_d_value_l837_83755

theorem circle_radius_d_value (d : ℝ) :
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + d = 0 → (x - 4)^2 + (y + 5)^2 = 36) →
  d = 5 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_d_value_l837_83755


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l837_83722

/-- The atomic weight of Hydrogen in atomic mass units (amu) -/
def atomic_weight_H : ℝ := 1.008

/-- The atomic weight of Chromium in atomic mass units (amu) -/
def atomic_weight_Cr : ℝ := 51.996

/-- The atomic weight of Oxygen in atomic mass units (amu) -/
def atomic_weight_O : ℝ := 15.999

/-- The number of Hydrogen atoms in the compound -/
def num_H : ℕ := 2

/-- The number of Chromium atoms in the compound -/
def num_Cr : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 4

/-- The molecular weight of the compound in atomic mass units (amu) -/
def molecular_weight : ℝ := num_H * atomic_weight_H + num_Cr * atomic_weight_Cr + num_O * atomic_weight_O

theorem compound_molecular_weight : molecular_weight = 118.008 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l837_83722


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l837_83759

theorem trigonometric_equation_solution :
  ∀ x : ℝ,
  (Real.sin (2019 * x))^4 + (Real.cos (2022 * x))^2019 * (Real.cos (2019 * x))^2018 = 1 ↔
  (∃ n : ℤ, x = π / 4038 + π * n / 2019) ∨ (∃ k : ℤ, x = π * k / 3) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l837_83759


namespace NUMINAMATH_CALUDE_square_side_length_l837_83787

theorem square_side_length (rectangle_width rectangle_length : ℝ) 
  (h1 : rectangle_width = 4)
  (h2 : rectangle_length = 9)
  (h3 : rectangle_width > 0)
  (h4 : rectangle_length > 0) :
  ∃ (square_side : ℝ), 
    square_side * square_side = rectangle_width * rectangle_length ∧ 
    square_side = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l837_83787


namespace NUMINAMATH_CALUDE_fraction_simplification_l837_83740

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l837_83740


namespace NUMINAMATH_CALUDE_product_xy_equals_sqrt_30_6_l837_83777

/-- Represents a parallelogram EFGH with given side lengths -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ → ℝ
  GH : ℝ → ℝ
  HE : ℝ

/-- The product of x and y in the parallelogram EFGH -/
def product_xy (p : Parallelogram) : ℝ → ℝ → ℝ := fun x y => x * y

/-- Theorem: The product of x and y in the given parallelogram is √(30.6) -/
theorem product_xy_equals_sqrt_30_6 (p : Parallelogram) 
  (h1 : p.EF = 54)
  (h2 : ∀ x, p.FG x = 8 * x^2 + 2)
  (h3 : ∀ y, p.GH y = 5 * y^2 + 20)
  (h4 : p.HE = 38) :
  ∃ x y, product_xy p x y = Real.sqrt 30.6 := by
  sorry

#check product_xy_equals_sqrt_30_6

end NUMINAMATH_CALUDE_product_xy_equals_sqrt_30_6_l837_83777


namespace NUMINAMATH_CALUDE_no_smallest_rational_l837_83707

theorem no_smallest_rational : ¬ ∃ q : ℚ, ∀ r : ℚ, q ≤ r := by
  sorry

end NUMINAMATH_CALUDE_no_smallest_rational_l837_83707


namespace NUMINAMATH_CALUDE_fraction_subtraction_l837_83782

theorem fraction_subtraction (a : ℝ) (ha : a ≠ 0) : 1 / a - 3 / a = -2 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l837_83782


namespace NUMINAMATH_CALUDE_backpack_cost_is_fifteen_l837_83746

def total_spent : ℝ := 32
def pens_cost : ℝ := 1
def pencils_cost : ℝ := 1
def notebook_cost : ℝ := 3
def notebook_count : ℕ := 5

def backpack_cost : ℝ := total_spent - (pens_cost + pencils_cost + notebook_cost * notebook_count)

theorem backpack_cost_is_fifteen : backpack_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_backpack_cost_is_fifteen_l837_83746


namespace NUMINAMATH_CALUDE_sum_of_1006th_row_is_20112_l837_83724

/-- Calculates the sum of numbers in the nth row of the pattern -/
def row_sum (n : ℕ) : ℕ := n * (3 * n - 1) / 2

/-- The theorem states that the sum of numbers in the 1006th row equals 20112 -/
theorem sum_of_1006th_row_is_20112 : row_sum 1006 = 20112 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_1006th_row_is_20112_l837_83724


namespace NUMINAMATH_CALUDE_x_range_l837_83791

theorem x_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ x : ℝ, x^2 + 2*x < a/b + 16*b/a → -4 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_x_range_l837_83791


namespace NUMINAMATH_CALUDE_custom_op_result_l837_83701

-- Define the custom operation
def customOp (a b : ℚ) : ℚ := (a^2 + b^2) / (a - b)

-- State the theorem
theorem custom_op_result : customOp (customOp 7 5) 4 = 42 + 1/33 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l837_83701


namespace NUMINAMATH_CALUDE_expression_value_at_negative_three_l837_83766

theorem expression_value_at_negative_three :
  let x : ℤ := -3
  let expr := 5 * x - (3 * x - 2 * (2 * x - 3))
  expr = -24 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_negative_three_l837_83766


namespace NUMINAMATH_CALUDE_extreme_values_range_l837_83799

/-- Given a function f(x) = 2x³ - (1/2)ax² + ax + 1, where a is a real number,
    this theorem states that the range of values for a such that f(x) has two
    extreme values in the interval (0, +∞) is (0, +∞). -/
theorem extreme_values_range (a : ℝ) :
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x ≠ y ∧
    (∀ z : ℝ, 0 < z → (6 * z^2 - a * z + a = 0) ↔ (z = x ∨ z = y))) ↔
  (0 < a) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_range_l837_83799
