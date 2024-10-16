import Mathlib

namespace NUMINAMATH_CALUDE_female_managers_count_l2180_218098

/-- Represents a company with employees and managers -/
structure Company where
  total_employees : ℕ
  total_managers : ℕ
  male_employees : ℕ
  male_managers : ℕ
  female_employees : ℕ
  female_managers : ℕ

/-- The conditions of the company as described in the problem -/
def company_conditions (c : Company) : Prop :=
  c.female_employees = 500 ∧
  c.total_managers = (2 * c.total_employees) / 5 ∧
  c.male_managers = (2 * c.male_employees) / 5 ∧
  c.total_employees = c.male_employees + c.female_employees ∧
  c.total_managers = c.male_managers + c.female_managers

/-- The theorem to be proved -/
theorem female_managers_count (c : Company) :
  company_conditions c → c.female_managers = 200 := by
  sorry


end NUMINAMATH_CALUDE_female_managers_count_l2180_218098


namespace NUMINAMATH_CALUDE_common_tangent_sum_l2180_218085

/-- A line y = kx + b is a common tangent to the curves y = ln(1+x) and y = 2 + ln(x) -/
def isCommonTangent (k b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, 
    (k * x₁ + b = Real.log (1 + x₁)) ∧
    (k * x₂ + b = 2 + Real.log x₂) ∧
    (k = 1 / (1 + x₁)) ∧
    (k = 1 / x₂)

/-- If a line y = kx + b is a common tangent to the curves y = ln(1+x) and y = 2 + ln(x), 
    then k + b = 3 - ln(2) -/
theorem common_tangent_sum (k b : ℝ) : 
  isCommonTangent k b → k + b = 3 - Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l2180_218085


namespace NUMINAMATH_CALUDE_additive_fun_properties_l2180_218086

/-- A function satisfying f(x+y) = f(x) + f(y) for all x, y ∈ ℝ -/
def AdditiveFun (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem additive_fun_properties
  (f : ℝ → ℝ)
  (h_additive : AdditiveFun f)
  (h_increasing : Monotone f)
  (h_f1 : f 1 = 1)
  (h_f2a : ∀ a : ℝ, f (2 * a) > f (a - 1) + 2) :
  (f 0 = 0) ∧
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ a : ℝ, a > 1) :=
by sorry

end NUMINAMATH_CALUDE_additive_fun_properties_l2180_218086


namespace NUMINAMATH_CALUDE_system_real_solutions_l2180_218063

theorem system_real_solutions (k : ℝ) : 
  (∃ x y : ℝ, x - k * y = 0 ∧ x^2 + y = -1) ↔ -1/2 ≤ k ∧ k ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_system_real_solutions_l2180_218063


namespace NUMINAMATH_CALUDE_simplify_expression_l2180_218015

theorem simplify_expression (x y : ℝ) :
  (15 * x + 35 * y) + (20 * x + 45 * y) - (8 * x + 40 * y) = 27 * x + 40 * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2180_218015


namespace NUMINAMATH_CALUDE_split_99_into_four_numbers_l2180_218076

theorem split_99_into_four_numbers : ∃ (a b c d : ℚ),
  a + b + c + d = 99 ∧
  a + 2 = b - 2 ∧
  a + 2 = 2 * c ∧
  a + 2 = d / 2 ∧
  a = 20 ∧ b = 24 ∧ c = 11 ∧ d = 44 := by
  sorry

end NUMINAMATH_CALUDE_split_99_into_four_numbers_l2180_218076


namespace NUMINAMATH_CALUDE_afternoon_to_morning_ratio_is_two_to_one_l2180_218097

/-- Represents the sales of pears by a salesman in a day -/
structure PearSales where
  total : ℕ
  morning : ℕ
  afternoon : ℕ

/-- Theorem stating that the ratio of afternoon to morning pear sales is 2:1 -/
theorem afternoon_to_morning_ratio_is_two_to_one (sales : PearSales)
  (h_total : sales.total = 360)
  (h_morning : sales.morning = 120)
  (h_afternoon : sales.afternoon = 240) :
  sales.afternoon / sales.morning = 2 := by
  sorry

#check afternoon_to_morning_ratio_is_two_to_one

end NUMINAMATH_CALUDE_afternoon_to_morning_ratio_is_two_to_one_l2180_218097


namespace NUMINAMATH_CALUDE_boat_production_l2180_218059

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem boat_production : geometric_sum 5 3 4 = 200 := by
  sorry

end NUMINAMATH_CALUDE_boat_production_l2180_218059


namespace NUMINAMATH_CALUDE_problem_statement_l2180_218001

theorem problem_statement (x : ℝ) (h : x + 1/x = 5) :
  (x - 2)^2 + 25/((x - 2)^2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2180_218001


namespace NUMINAMATH_CALUDE_fraction_difference_equals_specific_fraction_l2180_218008

theorem fraction_difference_equals_specific_fraction : 
  (3^2 + 5^2 + 7^2) / (2^2 + 4^2 + 6^2) - (2^2 + 4^2 + 6^2) / (3^2 + 5^2 + 7^2) = 3753 / 4656 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_specific_fraction_l2180_218008


namespace NUMINAMATH_CALUDE_additional_people_for_lawn_mowing_l2180_218056

/-- The number of additional people needed to mow a lawn in a shorter time -/
theorem additional_people_for_lawn_mowing 
  (initial_people : ℕ) 
  (initial_time : ℕ) 
  (new_time : ℕ) 
  (h1 : initial_people > 0)
  (h2 : initial_time > 0)
  (h3 : new_time > 0)
  (h4 : new_time < initial_time) :
  let total_work := initial_people * initial_time
  let new_people := total_work / new_time
  new_people - initial_people = 10 :=
by sorry

end NUMINAMATH_CALUDE_additional_people_for_lawn_mowing_l2180_218056


namespace NUMINAMATH_CALUDE_salary_calculation_l2180_218080

theorem salary_calculation (salary : ℝ) 
  (h1 : salary / 5 + salary / 10 + 3 * salary / 5 + 14000 = salary) : 
  salary = 140000 := by
sorry

end NUMINAMATH_CALUDE_salary_calculation_l2180_218080


namespace NUMINAMATH_CALUDE_grill_coal_consumption_l2180_218067

theorem grill_coal_consumption (total_time : ℕ) (bags : ℕ) (coals_per_bag : ℕ) 
  (h1 : total_time = 240)
  (h2 : bags = 3)
  (h3 : coals_per_bag = 60) :
  (bags * coals_per_bag) / (total_time / 20) = 15 := by
  sorry

end NUMINAMATH_CALUDE_grill_coal_consumption_l2180_218067


namespace NUMINAMATH_CALUDE_length_of_CF_l2180_218073

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle ABCD with given properties -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point
  ab_length : ℝ
  bc_length : ℝ
  cd_length : ℝ
  da_length : ℝ
  is_rectangle : ab_length = cd_length ∧ bc_length = da_length

/-- Triangle DEF with B as its centroid -/
structure TriangleDEF where
  D : Point
  E : Point
  F : Point
  B : Point
  is_centroid : B.x = (2 * D.x + E.x) / 3 ∧ B.y = (2 * D.y + E.y) / 3

/-- The main theorem -/
theorem length_of_CF (rect : Rectangle) (tri : TriangleDEF) :
  rect.A = tri.D ∧
  rect.B = tri.B ∧
  rect.C.x = tri.F.x ∧
  rect.da_length = 7 ∧
  rect.ab_length = 6 ∧
  rect.cd_length = 8 →
  Real.sqrt ((rect.C.x - tri.F.x)^2 + (rect.C.y - tri.F.y)^2) = 10.66 := by
  sorry


end NUMINAMATH_CALUDE_length_of_CF_l2180_218073


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2180_218005

theorem at_least_one_not_less_than_two (a b : ℕ) (h : a + b ≥ 3) : max a b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2180_218005


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l2180_218011

theorem min_value_and_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∃ m : ℝ, m = 6 ∧
    (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 →
      1 / a'^3 + 1 / b'^3 + 1 / c'^3 + 3 * a' * b' * c' ≥ m) ∧
    (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 →
      1 / a'^3 + 1 / b'^3 + 1 / c'^3 + 3 * a' * b' * c' = m →
      a' = 1 ∧ b' = 1 ∧ c' = 1)) ∧
  (∀ x : ℝ, abs (x + 1) - 2 * x < 6 ↔ x > -7/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l2180_218011


namespace NUMINAMATH_CALUDE_candy_difference_l2180_218022

theorem candy_difference (sandra_bags : Nat) (sandra_pieces_per_bag : Nat)
  (roger_bag1 : Nat) (roger_bag2 : Nat) :
  sandra_bags = 2 →
  sandra_pieces_per_bag = 6 →
  roger_bag1 = 11 →
  roger_bag2 = 3 →
  (roger_bag1 + roger_bag2) - (sandra_bags * sandra_pieces_per_bag) = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_difference_l2180_218022


namespace NUMINAMATH_CALUDE_equation_solution_l2180_218049

theorem equation_solution : ∃ x : ℚ, 3 * (x - 2) = 2 - 5 * (x - 2) ∧ x = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2180_218049


namespace NUMINAMATH_CALUDE_last_digit_of_3_power_2023_l2180_218014

/-- The last digit of 3^n for n ≥ 1 -/
def lastDigitOf3Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | 0 => 1
  | _ => 0  -- This case should never occur

theorem last_digit_of_3_power_2023 :
  lastDigitOf3Power 2023 = 7 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_3_power_2023_l2180_218014


namespace NUMINAMATH_CALUDE_function_value_range_l2180_218071

theorem function_value_range :
  ∀ x : ℝ, -2 ≤ Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 - 1 ∧
           Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 - 1 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_range_l2180_218071


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2180_218039

/-- The equation of a line perpendicular to 2x - y + 4 = 0 and passing through (-2, 1) is x + 2y = 0 -/
theorem perpendicular_line_equation :
  let given_line : ℝ → ℝ → Prop := λ x y => 2 * x - y + 4 = 0
  let point : ℝ × ℝ := (-2, 1)
  let perpendicular_line : ℝ → ℝ → Prop := λ x y => x + 2 * y = 0
  (∀ x y, perpendicular_line x y ↔ 
    (∃ m b, y = m * x + b ∧ 
            m * 2 = -1 ∧ 
            point.2 = m * point.1 + b)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2180_218039


namespace NUMINAMATH_CALUDE_roots_equation_l2180_218094

theorem roots_equation (α β : ℝ) : 
  (α^2 - 3*α + 1 = 0) → 
  (β^2 - 3*β + 1 = 0) → 
  3*α^4 + 8*β^3 = 333 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_l2180_218094


namespace NUMINAMATH_CALUDE_nabla_calculation_l2180_218055

-- Define the ∇ operation
def nabla (a b : ℕ) : ℕ :=
  (b * (2 * a + b - 1)) / 2

-- State the theorem
theorem nabla_calculation : nabla 2 (nabla 0 (nabla 1 7)) = 71859 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l2180_218055


namespace NUMINAMATH_CALUDE_average_people_per_hour_rounded_l2180_218032

def people_moving : ℕ := 3000
def days : ℕ := 4
def hours_per_day : ℕ := 24

def average_per_hour : ℚ :=
  people_moving / (days * hours_per_day)

theorem average_people_per_hour_rounded :
  round average_per_hour = 31 := by
  sorry

end NUMINAMATH_CALUDE_average_people_per_hour_rounded_l2180_218032


namespace NUMINAMATH_CALUDE_unique_solution_complex_magnitude_and_inequality_l2180_218050

theorem unique_solution_complex_magnitude_and_inequality :
  ∃! (n : ℝ), n > 0 ∧ Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 13 ∧ n^2 + 5*n > 50 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_complex_magnitude_and_inequality_l2180_218050


namespace NUMINAMATH_CALUDE_halloween_candies_l2180_218087

/-- The total number of candies collected by a group of friends on Halloween. -/
def total_candies (bob : ℕ) (mary : ℕ) (john : ℕ) (sue : ℕ) (sam : ℕ) : ℕ :=
  bob + mary + john + sue + sam

/-- Theorem stating that the total number of candies collected by the friends is 50. -/
theorem halloween_candies : total_candies 10 5 5 20 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candies_l2180_218087


namespace NUMINAMATH_CALUDE_triangle_theorem_l2180_218030

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b * Real.tan t.A = 2 * t.a * Real.sin t.B ∧
  t.a = Real.sqrt 7 ∧
  2 * t.b - t.c = 4

-- Define the theorem
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.A = π / 3 ∧ (1/2 * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2180_218030


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l2180_218062

/-- Represents the outcome of a single shot -/
inductive ShotOutcome
| Hit
| Miss

/-- Represents the outcome of two shots -/
def TwoShotOutcome := ShotOutcome × ShotOutcome

/-- The event of hitting the target at least once in two shots -/
def hitAtLeastOnce (outcome : TwoShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Hit ∨ outcome.2 = ShotOutcome.Hit

/-- The event of missing the target both times in two shots -/
def missBothTimes (outcome : TwoShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Miss ∧ outcome.2 = ShotOutcome.Miss

theorem mutually_exclusive_events :
  ∀ (outcome : TwoShotOutcome), ¬(hitAtLeastOnce outcome ∧ missBothTimes outcome) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l2180_218062


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2180_218041

/-- The perimeter of a triangle with vertices at (1, 4), (-7, 0), and (1, 0) is equal to 4√5 + 12. -/
theorem triangle_perimeter : 
  let A : ℝ × ℝ := (1, 4)
  let B : ℝ × ℝ := (-7, 0)
  let C : ℝ × ℝ := (1, 0)
  let d₁ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let d₂ := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let d₃ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  d₁ + d₂ + d₃ = 4 * Real.sqrt 5 + 12 := by
  sorry


end NUMINAMATH_CALUDE_triangle_perimeter_l2180_218041


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sin_sum_l2180_218031

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sin_sum 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + a 5 + a 9 = 2 * Real.pi) : 
  Real.sin (a 2 + a 8) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sin_sum_l2180_218031


namespace NUMINAMATH_CALUDE_no_integer_solution_l2180_218095

theorem no_integer_solution : ¬∃ (k l m n x : ℤ),
  (x = k * l * m * n) ∧
  (x - k = 1966) ∧
  (x - l = 966) ∧
  (x - m = 66) ∧
  (x - n = 6) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2180_218095


namespace NUMINAMATH_CALUDE_range_of_m_l2180_218036

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → x^2 - 4*x - 2*m + 1 ≤ 0) → m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2180_218036


namespace NUMINAMATH_CALUDE_lilia_initial_peaches_l2180_218096

/-- The number of peaches Lilia sold to friends -/
def peaches_sold_to_friends : ℕ := 10

/-- The price of each peach sold to friends -/
def price_for_friends : ℚ := 2

/-- The number of peaches Lilia sold to relatives -/
def peaches_sold_to_relatives : ℕ := 4

/-- The price of each peach sold to relatives -/
def price_for_relatives : ℚ := 5/4

/-- The number of peaches Lilia kept for herself -/
def peaches_kept : ℕ := 1

/-- The total amount of money Lilia earned -/
def total_earned : ℚ := 25

/-- The initial number of peaches Lilia had -/
def initial_peaches : ℕ := peaches_sold_to_friends + peaches_sold_to_relatives + peaches_kept

theorem lilia_initial_peaches :
  initial_peaches = 15 ∧
  total_earned = peaches_sold_to_friends * price_for_friends + peaches_sold_to_relatives * price_for_relatives :=
by sorry

end NUMINAMATH_CALUDE_lilia_initial_peaches_l2180_218096


namespace NUMINAMATH_CALUDE_car_speed_calculation_l2180_218078

/-- Calculates the speed of a car given distance and time -/
theorem car_speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 360 ∧ time = 4.5 → speed = distance / time → speed = 80 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_calculation_l2180_218078


namespace NUMINAMATH_CALUDE_age_problem_l2180_218000

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 47 →
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l2180_218000


namespace NUMINAMATH_CALUDE_total_games_in_season_l2180_218006

theorem total_games_in_season (total_teams : ℕ) (teams_per_division : ℕ) 
  (h1 : total_teams = 16)
  (h2 : teams_per_division = 8)
  (h3 : total_teams = 2 * teams_per_division)
  (h4 : ∀ (division : Fin 2), ∀ (team : Fin teams_per_division),
    (division.val = 0 → 
      (teams_per_division - 1) * 2 + teams_per_division = 22) ∧
    (division.val = 1 → 
      (teams_per_division - 1) * 2 + teams_per_division = 22)) :
  total_teams * 22 / 2 = 176 := by
sorry

end NUMINAMATH_CALUDE_total_games_in_season_l2180_218006


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2180_218026

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 1 → 1/x + 1/y ≥ 3 + 2*Real.sqrt 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ 1/x + 1/y = 3 + 2*Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2180_218026


namespace NUMINAMATH_CALUDE_donut_selection_problem_l2180_218034

/-- The number of ways to select n items from k types with at least one of each type -/
def selectWithMinimum (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- The problem statement -/
theorem donut_selection_problem :
  selectWithMinimum 6 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_donut_selection_problem_l2180_218034


namespace NUMINAMATH_CALUDE_four_periods_required_l2180_218004

/-- The number of periods required for all students to present their projects -/
def required_periods (num_students : ℕ) (presentation_time : ℕ) (period_length : ℕ) : ℕ :=
  (num_students * presentation_time + period_length - 1) / period_length

/-- Proof that 4 periods are required for the given conditions -/
theorem four_periods_required :
  required_periods 32 5 40 = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_periods_required_l2180_218004


namespace NUMINAMATH_CALUDE_cereal_eating_time_l2180_218083

/-- The time it takes for Mr. Fat and Mr. Thin to eat 4 pounds of cereal together -/
def eating_time (fat_rate thin_rate total_cereal : ℚ) : ℕ :=
  (total_cereal / (fat_rate + thin_rate)).ceil.toNat

/-- Proves that Mr. Fat and Mr. Thin take 53 minutes to eat 4 pounds of cereal together -/
theorem cereal_eating_time :
  eating_time (1 / 20) (1 / 40) 4 = 53 := by
  sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l2180_218083


namespace NUMINAMATH_CALUDE_three_unique_circles_l2180_218089

/-- A square with vertices P, Q, R, and S -/
structure Square where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- A circle defined by two points as its diameter endpoints -/
structure Circle where
  endpoint1 : Point
  endpoint2 : Point

/-- Function to count unique circles defined by square vertices -/
def count_unique_circles (s : Square) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 3 unique circles -/
theorem three_unique_circles (s : Square) : 
  count_unique_circles s = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_unique_circles_l2180_218089


namespace NUMINAMATH_CALUDE_megan_popsicle_consumption_l2180_218084

/-- The number of Popsicles Megan can finish in a given time -/
def popsicles_finished (popsicle_interval : ℕ) (total_time : ℕ) : ℕ :=
  total_time / popsicle_interval

theorem megan_popsicle_consumption :
  let popsicle_interval : ℕ := 15  -- minutes
  let hours : ℕ := 4
  let additional_minutes : ℕ := 30
  let total_time : ℕ := hours * 60 + additional_minutes
  popsicles_finished popsicle_interval total_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_megan_popsicle_consumption_l2180_218084


namespace NUMINAMATH_CALUDE_no_eulerian_path_in_problem_graph_l2180_218010

/-- A region in the planar graph --/
structure Region where
  edges : ℕ

/-- A planar graph representation --/
structure PlanarGraph where
  regions : List Region
  total_edges : ℕ

/-- Check if a planar graph has an Eulerian path --/
def has_eulerian_path (g : PlanarGraph) : Prop :=
  (g.regions.filter (λ r => r.edges % 2 = 1)).length ≤ 2

/-- The specific planar graph from the problem --/
def problem_graph : PlanarGraph :=
  { regions := [
      { edges := 5 },
      { edges := 5 },
      { edges := 4 },
      { edges := 5 },
      { edges := 4 },
      { edges := 4 },
      { edges := 4 }
    ],
    total_edges := 16
  }

theorem no_eulerian_path_in_problem_graph :
  ¬ (has_eulerian_path problem_graph) :=
by sorry

end NUMINAMATH_CALUDE_no_eulerian_path_in_problem_graph_l2180_218010


namespace NUMINAMATH_CALUDE_game_show_probability_l2180_218077

def num_questions : ℕ := 4
def num_options : ℕ := 4
def min_correct : ℕ := 3

def prob_correct_one : ℚ := 1 / num_options

def prob_all_correct : ℚ := prob_correct_one ^ num_questions

def prob_exactly_three_correct : ℚ := num_questions * (prob_correct_one ^ 3) * (1 - prob_correct_one)

theorem game_show_probability :
  prob_all_correct + prob_exactly_three_correct = 13 / 256 := by
  sorry

end NUMINAMATH_CALUDE_game_show_probability_l2180_218077


namespace NUMINAMATH_CALUDE_sum_of_divisors_of_420_l2180_218075

-- Define the sum of divisors function
def sumOfDivisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

-- State the theorem
theorem sum_of_divisors_of_420 : sumOfDivisors 420 = 1344 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_of_420_l2180_218075


namespace NUMINAMATH_CALUDE_triangle_properties_l2180_218064

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The cosine rule for triangle ABC -/
axiom cosine_rule (t : Triangle) : t.a^2 + t.b^2 - t.c^2 = 2 * t.a * t.b * Real.cos t.C

/-- The area formula for triangle ABC -/
axiom area_formula (t : Triangle) (S : ℝ) : S = 1/2 * t.a * t.b * Real.sin t.C

theorem triangle_properties (t : Triangle) (S : ℝ) :
  (t.a^2 + t.b^2 - t.c^2 = t.a * t.b → t.C = π/3) ∧
  (t.a^2 + t.b^2 - t.c^2 = t.a * t.b ∧ t.c = Real.sqrt 7 ∧ S = 3 * Real.sqrt 3 / 2 → t.a + t.b = 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2180_218064


namespace NUMINAMATH_CALUDE_min_value_sin_cos_min_value_achievable_l2180_218065

theorem min_value_sin_cos (x : ℝ) : 
  Real.sin x ^ 6 + 2 * Real.cos x ^ 6 ≥ 2/3 :=
sorry

theorem min_value_achievable : 
  ∃ x : ℝ, Real.sin x ^ 6 + 2 * Real.cos x ^ 6 = 2/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_sin_cos_min_value_achievable_l2180_218065


namespace NUMINAMATH_CALUDE_rational_inequality_equivalence_l2180_218009

theorem rational_inequality_equivalence (x : ℝ) :
  (2 * x - 1) / (x + 1) > 1 ↔ x < -1 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_rational_inequality_equivalence_l2180_218009


namespace NUMINAMATH_CALUDE_tangent_line_implies_sum_l2180_218045

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 - x^2 - a*x + b

/-- The derivative of f with respect to x -/
def f' (a x : ℝ) : ℝ := 3*x^2 - 2*x - a

theorem tangent_line_implies_sum (a b : ℝ) :
  f a b 0 = 1 ∧ f' a 0 = 2 → a + b = -1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_implies_sum_l2180_218045


namespace NUMINAMATH_CALUDE_line_through_points_sum_of_coefficients_l2180_218068

-- Define the line equation
def line_equation (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem line_through_points_sum_of_coefficients :
  ∀ a b : ℝ,
  line_equation a b 2 = 3 →
  line_equation a b 10 = 19 →
  a + b = 1 := by
  sorry


end NUMINAMATH_CALUDE_line_through_points_sum_of_coefficients_l2180_218068


namespace NUMINAMATH_CALUDE_sandy_book_purchase_l2180_218003

theorem sandy_book_purchase (books_shop1 : ℕ) (cost_shop1 : ℕ) (cost_shop2 : ℕ) (avg_price : ℚ) :
  books_shop1 = 65 →
  cost_shop1 = 1280 →
  cost_shop2 = 880 →
  avg_price = 18 →
  ∃ (books_shop2 : ℕ), 
    (books_shop1 + books_shop2) * avg_price = cost_shop1 + cost_shop2 ∧
    books_shop2 = 55 :=
by sorry

end NUMINAMATH_CALUDE_sandy_book_purchase_l2180_218003


namespace NUMINAMATH_CALUDE_exists_x_squared_sum_l2180_218090

theorem exists_x_squared_sum : ∃ x : ℕ, 106 * 106 + x * x = 19872 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_squared_sum_l2180_218090


namespace NUMINAMATH_CALUDE_cubic_equation_c_value_l2180_218040

/-- Given a cubic equation with coefficients a, b, c, d, returns whether it has three distinct positive roots -/
def has_three_distinct_positive_roots (a b c d : ℝ) : Prop := sorry

/-- Given three real numbers, returns their sum of base-3 logarithms -/
def sum_of_log3 (x y z : ℝ) : ℝ := sorry

theorem cubic_equation_c_value (c d : ℝ) :
  has_three_distinct_positive_roots 4 (5 * c) (3 * d) c →
  ∃ (x y z : ℝ), sum_of_log3 x y z = 3 ∧ 
    4 * x^3 + 5 * c * x^2 + 3 * d * x + c = 0 ∧
    4 * y^3 + 5 * c * y^2 + 3 * d * y + c = 0 ∧
    4 * z^3 + 5 * c * z^2 + 3 * d * z + c = 0 →
  c = -108 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_c_value_l2180_218040


namespace NUMINAMATH_CALUDE_length_AG_is_3_sqrt_10_over_2_l2180_218019

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define properties of the triangle
def isRightAngled (t : Triangle) : Prop :=
  -- Right angle at A
  sorry

def hasGivenSides (t : Triangle) : Prop :=
  -- AB = 3 and AC = 3√5
  sorry

-- Define altitude AD
def altitudeAD (t : Triangle) : ℝ × ℝ :=
  sorry

-- Define median BE
def medianBE (t : Triangle) : ℝ × ℝ :=
  sorry

-- Define intersection point G
def intersectionG (t : Triangle) : ℝ × ℝ :=
  sorry

-- Define length of AG
def lengthAG (t : Triangle) : ℝ :=
  sorry

-- Theorem statement
theorem length_AG_is_3_sqrt_10_over_2 (t : Triangle) :
  isRightAngled t → hasGivenSides t →
  lengthAG t = (3 * Real.sqrt 10) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_length_AG_is_3_sqrt_10_over_2_l2180_218019


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2180_218018

theorem fraction_equation_solution (x y : ℝ) 
  (hx_nonzero : x ≠ 0) 
  (hx_not_one : x ≠ 1) 
  (hy_nonzero : y ≠ 0) 
  (hy_not_three : y ≠ 3) 
  (h_equation : (3 / x) + (2 / y) = 1 / 3) : 
  x = 9 * y / (y - 6) := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2180_218018


namespace NUMINAMATH_CALUDE_circles_intersect_l2180_218081

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 7 = 0
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 27 = 0

-- Theorem stating that the circles intersect
theorem circles_intersect : ∃ (x y : ℝ), circle_O1 x y ∧ circle_O2 x y :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l2180_218081


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2180_218053

/-- The perimeter of an equilateral triangle with side length 13/12 meters is 3.25 meters. -/
theorem equilateral_triangle_perimeter :
  let side_length : ℚ := 13 / 12
  let perimeter : ℚ := 3 * side_length
  perimeter = 13 / 4 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2180_218053


namespace NUMINAMATH_CALUDE_intersection_singleton_complement_intersection_l2180_218069

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + (a^2-5) = 0}

-- Part 1
theorem intersection_singleton (a : ℝ) : A ∩ B a = {2} → a = -1 ∨ a = -3 := by
  sorry

-- Part 2
theorem complement_intersection (a : ℝ) : A ∩ (Set.univ \ B a) = A →
  a < -3 ∨ (-3 < a ∧ a < -1 - Real.sqrt 3) ∨
  (-1 - Real.sqrt 3 < a ∧ a < -1) ∨
  (-1 < a ∧ a < -1 + Real.sqrt 3) ∨
  a > -1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_singleton_complement_intersection_l2180_218069


namespace NUMINAMATH_CALUDE_square_and_cube_roots_l2180_218044

theorem square_and_cube_roots :
  (∀ x : ℝ, x ^ 2 = 36 → x = 6 ∨ x = -6) ∧
  (Real.sqrt 16 = 4) ∧
  (∃ x : ℝ, x ^ 2 = 4 ∧ x > 0 ∧ x = 2) ∧
  (∃ x : ℝ, x ^ 3 = -27 ∧ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_square_and_cube_roots_l2180_218044


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l2180_218027

theorem quadratic_roots_product (b c : ℝ) : 
  (1 : ℝ) ∈ {x : ℝ | x^2 + b*x + c = 0} ∧ 
  (-2 : ℝ) ∈ {x : ℝ | x^2 + b*x + c = 0} → 
  b * c = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l2180_218027


namespace NUMINAMATH_CALUDE_combined_bus_capacity_l2180_218074

/-- The capacity of the train -/
def train_capacity : ℕ := 120

/-- The number of buses -/
def num_buses : ℕ := 2

/-- The capacity of each bus as a fraction of the train's capacity -/
def bus_capacity_fraction : ℚ := 1 / 6

/-- Theorem stating the combined capacity of the buses -/
theorem combined_bus_capacity :
  (↑train_capacity * bus_capacity_fraction * ↑num_buses : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_combined_bus_capacity_l2180_218074


namespace NUMINAMATH_CALUDE_three_layer_rug_area_l2180_218043

theorem three_layer_rug_area (total_area floor_area two_layer_area : ℝ) 
  (h1 : total_area = 200)
  (h2 : floor_area = 140)
  (h3 : two_layer_area = 22) :
  let three_layer_area := (total_area - floor_area - two_layer_area) / 2
  three_layer_area = 19 := by sorry

end NUMINAMATH_CALUDE_three_layer_rug_area_l2180_218043


namespace NUMINAMATH_CALUDE_stride_sync_l2180_218012

/-- The least common multiple of Jack and Jill's stride lengths -/
def stride_lcm (jack_stride jill_stride : ℕ) : ℕ :=
  Nat.lcm jack_stride jill_stride

/-- Theorem stating that the LCM of Jack and Jill's strides is 448 cm -/
theorem stride_sync (jack_stride jill_stride : ℕ) 
  (h1 : jack_stride = 64) 
  (h2 : jill_stride = 56) : 
  stride_lcm jack_stride jill_stride = 448 := by
  sorry

end NUMINAMATH_CALUDE_stride_sync_l2180_218012


namespace NUMINAMATH_CALUDE_water_needed_for_noah_l2180_218029

/-- Represents the recipe ratios and quantities for Noah's orange juice --/
structure OrangeJuiceRecipe where
  orange : ℝ  -- Amount of orange concentrate
  sugar : ℝ   -- Amount of sugar
  water : ℝ   -- Amount of water
  sugar_to_orange_ratio : sugar = 3 * orange
  water_to_sugar_ratio : water = 3 * sugar

/-- Theorem: Given Noah's recipe ratios and 4 cups of orange concentrate, 36 cups of water are needed --/
theorem water_needed_for_noah's_recipe : 
  ∀ (recipe : OrangeJuiceRecipe), 
  recipe.orange = 4 → 
  recipe.water = 36 := by
sorry


end NUMINAMATH_CALUDE_water_needed_for_noah_l2180_218029


namespace NUMINAMATH_CALUDE_a3_value_geometric_sequence_max_sum_value_l2180_218047

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define the conditions for the sequence
def SequenceConditions (a : Sequence) : Prop :=
  (∀ n ≥ 2, a n ≥ 0) ∧
  (∀ n ≥ 2, (2 * a n = a (n+1) + a (n-1)) ∨ (2 * a (n+1) = a n + a (n-1)))

-- Theorem 1
theorem a3_value (a : Sequence) (h : SequenceConditions a) :
  a 1 = 5 ∧ a 2 = 3 ∧ a 4 = 2 → a 3 = 1 :=
sorry

-- Theorem 2
theorem geometric_sequence (a : Sequence) (h : SequenceConditions a) :
  a 1 = 0 ∧ a 4 = 0 ∧ a 7 = 0 ∧ a 2 > 0 ∧ a 5 > 0 ∧ a 8 > 0 →
  ∃ q : ℝ, q = 1/4 ∧ a 5 = a 2 * q ∧ a 8 = a 5 * q :=
sorry

-- Theorem 3
theorem max_sum_value (a : Sequence) (h : SequenceConditions a) :
  a 1 = 1 ∧ a 2 = 2 ∧
  (∃ r s t : ℕ, 2 < r ∧ r < s ∧ s < t ∧ a r = 0 ∧ a s = 0 ∧ a t = 0 ∧
    (∀ n : ℕ, n ≠ r ∧ n ≠ s ∧ n ≠ t → a n ≠ 0)) →
  (∀ r s t : ℕ, 2 < r ∧ r < s ∧ s < t ∧ a r = 0 ∧ a s = 0 ∧ a t = 0 →
    a (r+1) + a (s+1) + a (t+1) ≤ 21/64) :=
sorry

end NUMINAMATH_CALUDE_a3_value_geometric_sequence_max_sum_value_l2180_218047


namespace NUMINAMATH_CALUDE_range_of_a_l2180_218038

def p (a : ℝ) : Prop := ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 * x^2 + a * x - 2 = 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0

theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a > 1 ∨ (-1 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2180_218038


namespace NUMINAMATH_CALUDE_cube_opposite_face_l2180_218025

structure Cube where
  faces : Finset Char
  adjacent : Char → Finset Char

def is_opposite (c : Cube) (face1 face2 : Char) : Prop :=
  face1 ∈ c.faces ∧ face2 ∈ c.faces ∧ face1 ≠ face2 ∧
  c.adjacent face1 ∩ c.adjacent face2 = ∅

theorem cube_opposite_face (c : Cube) :
  c.faces = {'A', 'B', 'C', 'D', 'E', 'F'} →
  c.adjacent 'E' = {'A', 'B', 'C', 'D'} →
  is_opposite c 'E' 'F' :=
by sorry

end NUMINAMATH_CALUDE_cube_opposite_face_l2180_218025


namespace NUMINAMATH_CALUDE_inequality_solution_transformation_l2180_218072

theorem inequality_solution_transformation (a c : ℝ) :
  (∀ x : ℝ, ax^2 + 2*x + c > 0 ↔ -1/3 < x ∧ x < 1/2) →
  (∀ x : ℝ, -c*x^2 + 2*x - a > 0 ↔ -2 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_transformation_l2180_218072


namespace NUMINAMATH_CALUDE_range_of_m_l2180_218092

/-- Proposition p: The solution set for |x| + |x + 1| > m is ℝ -/
def p (m : ℝ) : Prop := ∀ x, |x| + |x + 1| > m

/-- Proposition q: The function f(x) = x^2 - 2mx + 1 is increasing on the interval (2, +∞) -/
def q (m : ℝ) : Prop := ∀ x > 2, Monotone (fun x => x^2 - 2*m*x + 1)

/-- The main theorem stating the range of m given the conditions -/
theorem range_of_m (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → 1 ≤ m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2180_218092


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_relationship_l2180_218052

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships
variable (parallelToPlane : Line → Plane → Prop)
variable (withinPlane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane_relationship 
  (a b : Line) (α : Plane)
  (h1 : parallelToPlane a α)
  (h2 : withinPlane b α) :
  parallel a b ∨ skew a b :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_relationship_l2180_218052


namespace NUMINAMATH_CALUDE_multiplication_after_division_l2180_218028

theorem multiplication_after_division (x y : ℝ) : x = 6 → (x / 6) * y = 12 → y = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_after_division_l2180_218028


namespace NUMINAMATH_CALUDE_group_size_calculation_l2180_218020

theorem group_size_calculation (iceland : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : iceland = 55)
  (h2 : norway = 43)
  (h3 : both = 61)
  (h4 : neither = 63) :
  iceland + norway - both + neither = 161 :=
by sorry

end NUMINAMATH_CALUDE_group_size_calculation_l2180_218020


namespace NUMINAMATH_CALUDE_subset_P_l2180_218066

def P : Set ℝ := {x | x ≤ 3}

theorem subset_P : {-1} ⊆ P := by sorry

end NUMINAMATH_CALUDE_subset_P_l2180_218066


namespace NUMINAMATH_CALUDE_sum_equals_two_n_cubed_l2180_218007

/-- The sum of numbers in the nth group of positive integers -/
def A (n : ℕ+) : ℕ := sorry

/-- The difference between the latter and former number in the nth group of cubes of natural numbers -/
def B (n : ℕ+) : ℕ := sorry

/-- The theorem stating that A_n + B_n = 2n^3 for all positive integers n -/
theorem sum_equals_two_n_cubed (n : ℕ+) : A n + B n = 2 * n.val ^ 3 := by sorry

end NUMINAMATH_CALUDE_sum_equals_two_n_cubed_l2180_218007


namespace NUMINAMATH_CALUDE_f_properties_l2180_218013

noncomputable def f (x : ℝ) := 6 * (Real.cos x)^2 - Real.sqrt 3 * Real.sin (2 * x)

theorem f_properties :
  (∃ (max_value : ℝ), ∀ (x : ℝ), f x ≤ max_value ∧ max_value = 2 * Real.sqrt 3 + 3) ∧
  (∃ (period : ℝ), period > 0 ∧ ∀ (x : ℝ), f (x + period) = f x ∧ 
    ∀ (p : ℝ), p > 0 → (∀ (x : ℝ), f (x + p) = f x) → period ≤ p) ∧
  (∀ (α : ℝ), 0 < α ∧ α < Real.pi / 2 → 
    f α = 3 - 2 * Real.sqrt 3 → Real.tan (4 * α / 5) = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2180_218013


namespace NUMINAMATH_CALUDE_parallel_segments_length_l2180_218021

/-- Given three parallel line segments XY, UV, and PQ, where UV = 90 cm and XY = 120 cm,
    prove that the length of PQ is 360/7 cm. -/
theorem parallel_segments_length (XY UV PQ : ℝ) (h1 : XY = 120) (h2 : UV = 90)
    (h3 : ∃ (k : ℝ), XY = k * UV ∧ PQ = k * UV) : PQ = 360 / 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_segments_length_l2180_218021


namespace NUMINAMATH_CALUDE_harrys_age_l2180_218046

theorem harrys_age :
  ∀ (H : ℕ),
  (H + 24 : ℕ) - H / 25 = H + 22 →
  H = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_harrys_age_l2180_218046


namespace NUMINAMATH_CALUDE_hcf_problem_l2180_218091

def is_hcf (h : ℕ) (a b : ℕ) : Prop :=
  h ∣ a ∧ h ∣ b ∧ ∀ d : ℕ, d ∣ a → d ∣ b → d ≤ h

def is_lcm (l : ℕ) (a b : ℕ) : Prop :=
  a ∣ l ∧ b ∣ l ∧ ∀ m : ℕ, a ∣ m → b ∣ m → l ≤ m

theorem hcf_problem (a b : ℕ) (h : ℕ) :
  a = 345 →
  (∃ l : ℕ, is_lcm l a b ∧ l = h * 13 * 15) →
  is_hcf h a b →
  h = 15 := by sorry

end NUMINAMATH_CALUDE_hcf_problem_l2180_218091


namespace NUMINAMATH_CALUDE_random_event_identification_l2180_218057

theorem random_event_identification :
  -- Event ①
  (∀ x : ℝ, x^2 + 1 ≠ 0) ∧
  -- Event ②
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x > 1/x ∧ y ≤ 1/y) ∧
  -- Event ③
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → 1/x > 1/y) ∧
  -- Event ④
  (∀ a b : ℝ, a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_random_event_identification_l2180_218057


namespace NUMINAMATH_CALUDE_range_of_a_for_single_root_l2180_218042

-- Define the function f(x) = 2x³ - 3x² + a
def f (x a : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + a

-- State the theorem
theorem range_of_a_for_single_root :
  ∀ a : ℝ, (∃! x : ℝ, x ∈ Set.Icc (-2) 2 ∧ f x a = 0) →
  a ∈ Set.Ioo (-4) 0 ∪ Set.Ioo 1 28 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_single_root_l2180_218042


namespace NUMINAMATH_CALUDE_salary_problem_l2180_218037

theorem salary_problem (total_salary : ℝ) (a_spend_rate : ℝ) (b_spend_rate : ℝ)
  (h_total : total_salary = 5000)
  (h_a_spend : a_spend_rate = 0.95)
  (h_b_spend : b_spend_rate = 0.85)
  (h_equal_savings : (1 - a_spend_rate) * a_salary = (1 - b_spend_rate) * b_salary)
  (h_total_sum : a_salary + b_salary = total_salary) :
  a_salary = 3750 :=
by
  sorry

end NUMINAMATH_CALUDE_salary_problem_l2180_218037


namespace NUMINAMATH_CALUDE_place_value_ratio_in_53687_4921_l2180_218058

/-- The place value of a digit in a decimal number -/
def place_value (digit_position : Int) : ℚ :=
  10 ^ digit_position

/-- The position of a digit in a decimal number, counting from right to left,
    with the decimal point at position 0 -/
def digit_position (n : ℚ) (d : ℕ) : Int :=
  sorry

theorem place_value_ratio_in_53687_4921 :
  let n : ℚ := 53687.4921
  let pos_8 := digit_position n 8
  let pos_2 := digit_position n 2
  place_value pos_8 / place_value pos_2 = 1000 := by sorry

end NUMINAMATH_CALUDE_place_value_ratio_in_53687_4921_l2180_218058


namespace NUMINAMATH_CALUDE_numbers_with_2019_divisors_l2180_218016

theorem numbers_with_2019_divisors (n : ℕ) : 
  n < 128^97 → (Finset.card (Nat.divisors n) = 2019) → 
  (n = 2^672 * 3^2 ∨ n = 2^672 * 5^2 ∨ n = 2^672 * 7^2 ∨ n = 2^672 * 11^2) :=
by sorry

end NUMINAMATH_CALUDE_numbers_with_2019_divisors_l2180_218016


namespace NUMINAMATH_CALUDE_i_to_2016_l2180_218099

theorem i_to_2016 (i : ℂ) (h : i^2 = -1) : i^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_i_to_2016_l2180_218099


namespace NUMINAMATH_CALUDE_rabbit_pairs_rabbit_pairs_base_cases_rabbit_pairs_recurrence_l2180_218060

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem rabbit_pairs (n : ℕ) : 
  fib n = if n = 0 then 0 
          else if n = 1 then 1 
          else fib (n - 1) + fib (n - 2) := by
  sorry

theorem rabbit_pairs_base_cases :
  fib 1 = 1 ∧ fib 2 = 1 := by
  sorry

theorem rabbit_pairs_recurrence (n : ℕ) (h : n > 2) :
  fib n = fib (n - 1) + fib (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_rabbit_pairs_rabbit_pairs_base_cases_rabbit_pairs_recurrence_l2180_218060


namespace NUMINAMATH_CALUDE_least_number_of_grapes_l2180_218088

theorem least_number_of_grapes (n : ℕ) : 
  (n > 0) → 
  (n % 3 = 1) → 
  (n % 5 = 1) → 
  (n % 7 = 1) → 
  (∀ m : ℕ, m > 0 → m % 3 = 1 → m % 5 = 1 → m % 7 = 1 → m ≥ n) → 
  n = 106 := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_grapes_l2180_218088


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2180_218033

/-- Given a quadratic polynomial 7x^2 + 4x + 9, if α and β are the reciprocals of its roots,
    then their sum α + β equals -4/9 -/
theorem sum_of_reciprocals_of_roots (α β : ℝ) : 
  (∃ a b : ℝ, (7 * a^2 + 4 * a + 9 = 0) ∧ 
              (7 * b^2 + 4 * b + 9 = 0) ∧ 
              (α = 1 / a) ∧ 
              (β = 1 / b)) → 
  α + β = -4/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2180_218033


namespace NUMINAMATH_CALUDE_points_collinear_l2180_218017

-- Define the points
variable (A B C K : Point)

-- Define the shapes
variable (square1 square2 : Square)
variable (triangle : Triangle)

-- Define the properties
variable (triangle_isosceles : IsIsosceles triangle)
variable (K_on_triangle_side : OnSide K triangle)

-- Define the theorem
theorem points_collinear (h1 : triangle_isosceles) (h2 : K_on_triangle_side) : 
  Collinear A B C := by sorry

end NUMINAMATH_CALUDE_points_collinear_l2180_218017


namespace NUMINAMATH_CALUDE_largest_number_with_different_digits_summing_to_17_l2180_218048

def digits_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

def all_digits_different (n : ℕ) : Prop :=
  (n.digits 10).toFinset.card = (n.digits 10).length

theorem largest_number_with_different_digits_summing_to_17 :
  ∀ n : ℕ, 
    all_digits_different n → 
    digits_sum n = 17 → 
    n ≤ 7543210 :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_different_digits_summing_to_17_l2180_218048


namespace NUMINAMATH_CALUDE_distance_sum_between_22_and_23_l2180_218061

/-- Given points A, B, and D in a 2D plane, prove that the sum of distances AD and BD 
    is between 22 and 23. -/
theorem distance_sum_between_22_and_23 :
  let A : ℝ × ℝ := (15, 0)
  let B : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (6, 8)
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  22 < distance A D + distance B D ∧ distance A D + distance B D < 23 := by
  sorry


end NUMINAMATH_CALUDE_distance_sum_between_22_and_23_l2180_218061


namespace NUMINAMATH_CALUDE_ratio_equality_l2180_218023

theorem ratio_equality (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4 ∧ a / 2 ≠ 0) :
  (a - 2 * c) / (a - 2 * b) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2180_218023


namespace NUMINAMATH_CALUDE_original_plan_pages_l2180_218002

-- Define the total number of pages in the book
def total_pages : ℕ := 200

-- Define the number of days before changing the plan
def days_before_change : ℕ := 5

-- Define the additional pages read per day after changing the plan
def additional_pages : ℕ := 5

-- Define the number of days earlier the book was finished
def days_earlier : ℕ := 1

-- Define the function to calculate the total pages read
def total_pages_read (x : ℕ) : ℕ :=
  (days_before_change * x) + 
  ((x + additional_pages) * (total_pages / x - days_before_change - days_earlier))

-- Theorem stating that the original plan was to read 20 pages per day
theorem original_plan_pages : 
  ∃ (x : ℕ), x > 0 ∧ total_pages_read x = total_pages ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_original_plan_pages_l2180_218002


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2180_218051

theorem quadratic_inequality (x : ℝ) : 3 * x^2 - 4 * x - 4 < 0 ↔ -2/3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2180_218051


namespace NUMINAMATH_CALUDE_min_colors_l2180_218035

def is_divisor (a b : ℕ) : Prop := b % a = 0

def valid_coloring (f : ℕ → ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ 1000 ∧ 1 ≤ b ∧ b ≤ 1000 → 
    is_divisor a b → f a ≠ f b

theorem min_colors : 
  (∃ (n : ℕ) (f : ℕ → ℕ), n = 10 ∧ valid_coloring f ∧ 
    (∀ i, 1 ≤ i ∧ i ≤ 1000 → f i ≤ n)) ∧ 
  (∀ (m : ℕ) (g : ℕ → ℕ), m < 10 → 
    ¬(valid_coloring g ∧ (∀ i, 1 ≤ i ∧ i ≤ 1000 → g i ≤ m))) :=
by sorry

end NUMINAMATH_CALUDE_min_colors_l2180_218035


namespace NUMINAMATH_CALUDE_fifteen_distinct_configurations_l2180_218054

/-- Represents a 4x4x4 cube configuration with 63 white cubes and 1 black cube -/
def CubeConfiguration := Fin 4 → Fin 4 → Fin 4 → Bool

/-- Counts the number of distinct cube configurations -/
def countDistinctConfigurations : ℕ :=
  let corner_configs := 1
  let edge_configs := 2
  let face_configs := 1
  let inner_configs := 8
  corner_configs + edge_configs + face_configs + inner_configs

/-- Theorem stating that there are 15 distinct cube configurations -/
theorem fifteen_distinct_configurations :
  countDistinctConfigurations = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_distinct_configurations_l2180_218054


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l2180_218024

def initial_bottle_caps (current : ℕ) (lost : ℕ) : ℕ := current + lost

theorem danny_bottle_caps : initial_bottle_caps 25 66 = 91 := by
  sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l2180_218024


namespace NUMINAMATH_CALUDE_prime_divisibility_l2180_218082

theorem prime_divisibility (a b p q : ℕ) 
  (ha : 0 < a) (hb : 0 < b)
  (hp : Nat.Prime p) (hq : Nat.Prime q)
  (hpq : ¬(p ∣ (q - 1)))
  (hdiv : q ∣ (a^p - b^p)) : 
  q ∣ (a - b) := by
sorry

end NUMINAMATH_CALUDE_prime_divisibility_l2180_218082


namespace NUMINAMATH_CALUDE_academic_year_school_days_l2180_218079

/-- The number of school days in the academic year -/
def school_days : ℕ := sorry

/-- The number of days Aliyah packs lunch -/
def aliyah_lunch_days : ℕ := sorry

/-- The number of days Becky packs lunch -/
def becky_lunch_days : ℕ := 45

theorem academic_year_school_days :
  (aliyah_lunch_days = 2 * becky_lunch_days) →
  (school_days = 2 * aliyah_lunch_days) →
  (school_days = 180) :=
by sorry

end NUMINAMATH_CALUDE_academic_year_school_days_l2180_218079


namespace NUMINAMATH_CALUDE_geometric_sequence_constant_l2180_218070

/-- A geometric sequence with sum S_n = (a-2)⋅3^(n+1) + 2 -/
def GeometricSequence (a : ℝ) (n : ℕ) : ℝ := (a - 2) * 3^(n + 1) + 2

/-- The difference between consecutive sums gives the n-th term -/
def NthTerm (a : ℝ) (n : ℕ) : ℝ := GeometricSequence a n - GeometricSequence a (n - 1)

/-- Theorem stating that the constant a in the given geometric sequence is 4/3 -/
theorem geometric_sequence_constant : 
  ∃ (a : ℝ), (∀ n : ℕ, n ≥ 2 → (NthTerm a n) / (NthTerm a (n-1)) = (NthTerm a (n-1)) / (NthTerm a (n-2))) ∧ 
  a = 4/3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_constant_l2180_218070


namespace NUMINAMATH_CALUDE_term1_and_term2_not_like_terms_l2180_218093

/-- Two terms are like terms if they have the same variables with the same exponents -/
def are_like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ x y, ∃ k, term1 x y = k * term2 x y ∨ term2 x y = k * term1 x y

/-- The first term in our problem: x^3 * y^2 -/
def term1 (x y : ℕ) : ℚ := (x^3 * y^2 : ℚ)

/-- The second term in our problem: 3 * x^2 * y^3 -/
def term2 (x y : ℕ) : ℚ := (3 * x^2 * y^3 : ℚ)

/-- Theorem stating that term1 and term2 are not like terms -/
theorem term1_and_term2_not_like_terms : ¬(are_like_terms term1 term2) := by
  sorry

end NUMINAMATH_CALUDE_term1_and_term2_not_like_terms_l2180_218093
