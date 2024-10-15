import Mathlib

namespace NUMINAMATH_CALUDE_water_consumption_l2863_286362

/-- Proves that given a 1.5-quart bottle of water and a can of water,
    if the total amount of water drunk is 60 ounces,
    and 1 quart is equivalent to 32 ounces,
    then the can of water contains 12 ounces. -/
theorem water_consumption (bottle : ℚ) (can : ℚ) (total : ℚ) (quart_to_ounce : ℚ → ℚ) :
  bottle = 1.5 →
  total = 60 →
  quart_to_ounce 1 = 32 →
  can = total - quart_to_ounce bottle :=
by
  sorry

end NUMINAMATH_CALUDE_water_consumption_l2863_286362


namespace NUMINAMATH_CALUDE_inequality_relationship_l2863_286372

theorem inequality_relationship (x : ℝ) : 
  ¬(((x - 1) * (x + 3) < 0 → (x + 1) * (x - 3) < 0) ∧ 
    ((x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0)) :=
sorry

end NUMINAMATH_CALUDE_inequality_relationship_l2863_286372


namespace NUMINAMATH_CALUDE_f_range_l2863_286384

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.cos x ^ 2 - 3/4) + Real.sin x

theorem f_range : Set.range f = Set.Icc (-1/2) (Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_range_l2863_286384


namespace NUMINAMATH_CALUDE_triangle_side_and_angle_l2863_286305

open Real

structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

theorem triangle_side_and_angle (t : Triangle) :
  t.perimeter = Real.sqrt 3 + 1 →
  sin t.B + sin t.C = Real.sqrt 3 * sin t.A →
  t.c = 1 ∧
  (t.perimeter = Real.sqrt 3 + 1 →
   sin t.B + sin t.C = Real.sqrt 3 * sin t.A →
   (1/2) * t.a * t.b * sin t.A = (1/3) * sin t.A →
   t.A = π/3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_and_angle_l2863_286305


namespace NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l2863_286385

/-- The area of a triangle given its perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius 
  (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
  (h_perimeter : perimeter = 20) 
  (h_inradius : inradius = 2.5) : 
  area = perimeter / 2 * inradius ∧ area = 25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l2863_286385


namespace NUMINAMATH_CALUDE_homework_is_duration_l2863_286325

-- Define a type for time expressions
inductive TimeExpression
  | PointInTime (description : String)
  | Duration (description : String)

-- Define the given options
def option_a : TimeExpression := TimeExpression.PointInTime "Get up at 6:30"
def option_b : TimeExpression := TimeExpression.PointInTime "School ends at 3:40"
def option_c : TimeExpression := TimeExpression.Duration "It took 30 minutes to do the homework"

-- Define a function to check if a TimeExpression represents a duration
def is_duration (expr : TimeExpression) : Prop :=
  match expr with
  | TimeExpression.Duration _ => True
  | _ => False

-- Theorem to prove
theorem homework_is_duration :
  is_duration option_c ∧ ¬is_duration option_a ∧ ¬is_duration option_b :=
sorry

end NUMINAMATH_CALUDE_homework_is_duration_l2863_286325


namespace NUMINAMATH_CALUDE_correct_change_calculation_l2863_286351

/-- The change to be returned when mailing items with given costs and payment -/
def change_to_return (cost1 cost2 payment : ℚ) : ℚ :=
  payment - (cost1 + cost2)

/-- Theorem stating that the change to be returned is 1.2 yuan given the specific costs and payment -/
theorem correct_change_calculation :
  change_to_return (1.6) (12.2) (15) = (1.2) := by
  sorry

end NUMINAMATH_CALUDE_correct_change_calculation_l2863_286351


namespace NUMINAMATH_CALUDE_total_eggs_theorem_l2863_286334

/-- Represents the number of eggs used for a family member's breakfast on a given day type --/
structure EggUsage where
  children : ℕ  -- eggs per child
  husband : ℕ   -- eggs for husband
  lisa : ℕ      -- eggs for Lisa

/-- Represents the egg usage patterns for different days of the week --/
structure WeeklyEggUsage where
  monday_tuesday : EggUsage
  wednesday : EggUsage
  thursday : EggUsage
  friday : EggUsage

/-- Calculates the total eggs used in a year based on the given parameters --/
def total_eggs_per_year (
  num_children : ℕ
  ) (weekly_usage : WeeklyEggUsage
  ) (num_holidays : ℕ
  ) (holiday_usage : EggUsage
  ) : ℕ :=
  sorry

/-- The main theorem stating the total number of eggs used in a year --/
theorem total_eggs_theorem : 
  total_eggs_per_year 
    4  -- number of children
    {  -- weekly egg usage
      monday_tuesday := { children := 2, husband := 3, lisa := 2 },
      wednesday := { children := 3, husband := 4, lisa := 3 },
      thursday := { children := 1, husband := 2, lisa := 1 },
      friday := { children := 2, husband := 3, lisa := 2 }
    }
    8  -- number of holidays
    { children := 2, husband := 2, lisa := 2 }  -- holiday egg usage
  = 3476 := by
  sorry

end NUMINAMATH_CALUDE_total_eggs_theorem_l2863_286334


namespace NUMINAMATH_CALUDE_polynomial_division_result_l2863_286395

theorem polynomial_division_result :
  let f : Polynomial ℝ := 4 * X^4 + 12 * X^3 - 9 * X^2 + X + 3
  let d : Polynomial ℝ := X^2 + 3 * X - 2
  ∀ q r : Polynomial ℝ,
    f = q * d + r →
    (r.degree < d.degree) →
    q.eval 1 + r.eval (-1) = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_result_l2863_286395


namespace NUMINAMATH_CALUDE_solution_numbers_l2863_286391

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem solution_numbers : 
  {n : ℕ | n + sumOfDigits n = 2021} = {2014, 1996} := by sorry

end NUMINAMATH_CALUDE_solution_numbers_l2863_286391


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l2863_286339

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (5 * x - 4 < 3 - 2 * x) → x ≤ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l2863_286339


namespace NUMINAMATH_CALUDE_problem_statement_l2863_286378

theorem problem_statement (m n : ℝ) (h : 3 * m - n = 1) : 
  9 * m^2 - n^2 - 2 * n = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2863_286378


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2863_286306

theorem algebraic_expression_value :
  let a : ℝ := Real.sqrt 2 + 1
  let b : ℝ := Real.sqrt 2 - 1
  (a^2 - 2*a*b + b^2) / (a^2 - b^2) = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2863_286306


namespace NUMINAMATH_CALUDE_effervesces_arrangements_l2863_286309

def word := "EFFERVESCES"

/-- Number of ways to arrange letters in a line with no adjacent E's -/
def linear_arrangements (w : String) : ℕ := sorry

/-- Number of ways to arrange letters in a circle with no adjacent E's -/
def circular_arrangements (w : String) : ℕ := sorry

/-- No two E's are adjacent in the arrangement -/
def no_adjacent_es (arrangement : List Char) : Prop := sorry

/-- The arrangement preserves the letter count of the original word -/
def preserves_letter_count (w : String) (arrangement : List Char) : Prop := sorry

theorem effervesces_arrangements :
  (linear_arrangements word = 88200) ∧
  (circular_arrangements word = 6300) :=
by sorry

end NUMINAMATH_CALUDE_effervesces_arrangements_l2863_286309


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l2863_286357

theorem opposite_of_negative_two :
  ∀ x : ℝ, (x + (-2) = 0) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l2863_286357


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2863_286389

theorem complex_modulus_problem (x y : ℝ) (h : (Complex.I : ℂ) / (1 + Complex.I) = x + y * Complex.I) : 
  Complex.abs (x - y * Complex.I) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2863_286389


namespace NUMINAMATH_CALUDE_fruits_given_to_jane_l2863_286392

/- Define the initial number of each type of fruit -/
def plums : ℕ := 25
def guavas : ℕ := 30
def apples : ℕ := 36
def oranges : ℕ := 20
def bananas : ℕ := 15

/- Define the total number of fruits Jacqueline had initially -/
def initial_fruits : ℕ := plums + guavas + apples + oranges + bananas

/- Define the number of fruits Jacqueline had left -/
def fruits_left : ℕ := 38

/- Theorem: The number of fruits Jacqueline gave Jane is equal to 
   the difference between her initial fruits and the fruits left -/
theorem fruits_given_to_jane : 
  initial_fruits - fruits_left = 88 := by sorry

end NUMINAMATH_CALUDE_fruits_given_to_jane_l2863_286392


namespace NUMINAMATH_CALUDE_gardening_project_cost_l2863_286335

-- Define constants for the given conditions
def rose_bushes : Nat := 20
def fruit_trees : Nat := 10
def ornamental_shrubs : Nat := 5
def rose_bush_cost : Nat := 150
def fertilizer_cost : Nat := 25
def fruit_tree_cost : Nat := 75
def ornamental_shrub_cost : Nat := 50
def gardener_hourly_rate : Nat := 30
def soil_cost_per_cubic_foot : Nat := 5
def soil_needed : Nat := 100
def tiller_cost_per_day : Nat := 40
def wheelbarrow_cost_per_day : Nat := 10
def rental_days : Nat := 3

def gardener_hours : List Nat := [6, 5, 4, 7]

-- Define functions for calculations
def rose_bush_total_cost : Nat :=
  let base_cost := rose_bushes * rose_bush_cost
  let discount := base_cost * 5 / 100
  base_cost - discount

def fertilizer_total_cost : Nat :=
  let base_cost := rose_bushes * fertilizer_cost
  let discount := base_cost * 10 / 100
  base_cost - discount

def fruit_tree_total_cost : Nat :=
  let free_trees := fruit_trees / 3
  let paid_trees := fruit_trees - free_trees
  paid_trees * fruit_tree_cost

def ornamental_shrub_total_cost : Nat :=
  ornamental_shrubs * ornamental_shrub_cost

def gardener_total_cost : Nat :=
  (gardener_hours.sum) * gardener_hourly_rate

def soil_total_cost : Nat :=
  soil_needed * soil_cost_per_cubic_foot

def tools_rental_total_cost : Nat :=
  (tiller_cost_per_day + wheelbarrow_cost_per_day) * rental_days

-- Define the total cost of the gardening project
def total_gardening_cost : Nat :=
  rose_bush_total_cost +
  fertilizer_total_cost +
  fruit_tree_total_cost +
  ornamental_shrub_total_cost +
  gardener_total_cost +
  soil_total_cost +
  tools_rental_total_cost

-- Theorem statement
theorem gardening_project_cost :
  total_gardening_cost = 6385 := by sorry

end NUMINAMATH_CALUDE_gardening_project_cost_l2863_286335


namespace NUMINAMATH_CALUDE_jack_christina_lindy_meeting_l2863_286354

/-- The problem of Jack, Christina, and Lindy meeting --/
theorem jack_christina_lindy_meeting 
  (initial_distance : ℝ) 
  (christina_speed : ℝ) 
  (lindy_speed : ℝ) 
  (lindy_total_distance : ℝ) 
  (h1 : initial_distance = 240)
  (h2 : christina_speed = 3)
  (h3 : lindy_speed = 9)
  (h4 : lindy_total_distance = 270) :
  ∃ (jack_speed : ℝ), 
    jack_speed = 5 ∧ 
    (lindy_total_distance / lindy_speed) * jack_speed + 
    (lindy_total_distance / lindy_speed) * christina_speed = 
    initial_distance := by
  sorry


end NUMINAMATH_CALUDE_jack_christina_lindy_meeting_l2863_286354


namespace NUMINAMATH_CALUDE_files_deleted_l2863_286380

theorem files_deleted (initial_files remaining_files : ℕ) (h1 : initial_files = 25) (h2 : remaining_files = 2) :
  initial_files - remaining_files = 23 := by
  sorry

end NUMINAMATH_CALUDE_files_deleted_l2863_286380


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2863_286371

-- Define a rectangular solid with prime edge lengths
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ
  length_prime : Nat.Prime length
  width_prime : Nat.Prime width
  height_prime : Nat.Prime height
  different_edges : length ≠ width ∧ width ≠ height ∧ length ≠ height

-- Define the volume of the rectangular solid
def volume (r : RectangularSolid) : ℕ := r.length * r.width * r.height

-- Define the surface area of the rectangular solid
def surfaceArea (r : RectangularSolid) : ℕ :=
  2 * (r.length * r.width + r.width * r.height + r.length * r.height)

-- Theorem statement
theorem rectangular_solid_surface_area :
  ∀ r : RectangularSolid, volume r = 770 → surfaceArea r = 1098 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2863_286371


namespace NUMINAMATH_CALUDE_max_cars_ac_no_stripes_l2863_286333

theorem max_cars_ac_no_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (cars_with_stripes : ℕ) 
  (h1 : total_cars = 200)
  (h2 : cars_without_ac = 85)
  (h3 : cars_with_stripes ≥ 110) :
  ∃ (max_ac_no_stripes : ℕ),
    max_ac_no_stripes = 5 ∧
    max_ac_no_stripes ≤ total_cars - cars_without_ac ∧
    max_ac_no_stripes ≤ total_cars - cars_with_stripes :=
by
  sorry

end NUMINAMATH_CALUDE_max_cars_ac_no_stripes_l2863_286333


namespace NUMINAMATH_CALUDE_compound_weight_l2863_286394

/-- Given a compound with a molecular weight of 1098 and 9 moles of this compound,
    prove that the total weight is 9882 grams. -/
theorem compound_weight (molecular_weight : ℕ) (moles : ℕ) : 
  molecular_weight = 1098 → moles = 9 → molecular_weight * moles = 9882 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_l2863_286394


namespace NUMINAMATH_CALUDE_certain_number_proof_l2863_286313

theorem certain_number_proof : ∃ x : ℚ, x - (390 / 5) = (4 - (210 / 7)) + 114 ∧ x = 166 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2863_286313


namespace NUMINAMATH_CALUDE_number_scientific_notation_equality_l2863_286327

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff : 1 ≤ coefficient
  coeff_lt_ten : coefficient < 10

/-- The number to be represented in scientific notation -/
def number : ℕ := 11090000

/-- The scientific notation representation of the number -/
def scientific_rep : ScientificNotation :=
  { coefficient := 1.109
    exponent := 7
    one_le_coeff := by sorry
    coeff_lt_ten := by sorry }

theorem number_scientific_notation_equality :
  (number : ℝ) = scientific_rep.coefficient * (10 : ℝ) ^ scientific_rep.exponent := by
  sorry

end NUMINAMATH_CALUDE_number_scientific_notation_equality_l2863_286327


namespace NUMINAMATH_CALUDE_boys_share_calculation_l2863_286398

/-- Proves that in a family with a given boy-to-girl ratio and total children, 
    if a certain amount is shared among the boys, each boy receives the calculated amount. -/
theorem boys_share_calculation 
  (total_children : ℕ) 
  (boy_ratio girl_ratio : ℕ) 
  (total_money : ℕ) 
  (h1 : total_children = 180) 
  (h2 : boy_ratio = 5) 
  (h3 : girl_ratio = 7) 
  (h4 : total_money = 3900) :
  total_money / (total_children * boy_ratio / (boy_ratio + girl_ratio)) = 52 := by
sorry


end NUMINAMATH_CALUDE_boys_share_calculation_l2863_286398


namespace NUMINAMATH_CALUDE_max_pies_without_ingredients_l2863_286331

/-- Represents the number of pies with specific ingredients -/
structure PieCount where
  total : ℕ
  chocolate : ℕ
  marshmallow : ℕ
  cayenne : ℕ
  salted_soy_nut : ℕ

/-- Conditions for the pie problem -/
def pie_conditions (p : PieCount) : Prop :=
  p.total = 48 ∧
  p.chocolate = (5 * p.total) / 8 ∧
  p.marshmallow = (3 * p.total) / 4 ∧
  p.cayenne = (2 * p.total) / 3 ∧
  p.salted_soy_nut = p.total / 4 ∧
  p.salted_soy_nut ≤ p.marshmallow

/-- The theorem stating the maximum number of pies without any of the mentioned ingredients -/
theorem max_pies_without_ingredients (p : PieCount) 
  (h : pie_conditions p) : 
  p.total - max p.chocolate (max p.marshmallow (max p.cayenne p.salted_soy_nut)) ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_max_pies_without_ingredients_l2863_286331


namespace NUMINAMATH_CALUDE_consecutive_numbers_probability_l2863_286366

def choose (n k : ℕ) : ℕ := Nat.choose n k

def p : ℚ :=
  1 - (choose 40 6 + choose 5 1 * choose 39 5 + choose 4 2 * choose 38 4 + choose 37 3) / choose 45 6

theorem consecutive_numbers_probability : 
  ⌊1000 * p⌋ = 56 := by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_probability_l2863_286366


namespace NUMINAMATH_CALUDE_battle_station_staffing_l2863_286300

/-- The number of job openings --/
def num_openings : ℕ := 6

/-- The total number of resumes received --/
def total_resumes : ℕ := 36

/-- The number of suitable candidates after removing one-third --/
def suitable_candidates : ℕ := total_resumes - (total_resumes / 3)

/-- The number of ways to staff the battle station --/
def staffing_ways : ℕ := 255024240

theorem battle_station_staffing :
  (suitable_candidates.factorial) / ((suitable_candidates - num_openings).factorial) = staffing_ways := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l2863_286300


namespace NUMINAMATH_CALUDE_cos_two_pi_thirds_l2863_286338

theorem cos_two_pi_thirds : Real.cos (2 * Real.pi / 3) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_two_pi_thirds_l2863_286338


namespace NUMINAMATH_CALUDE_joan_total_seashells_l2863_286361

/-- Given that Joan found 79 seashells, received 63 from Mike, and 97 from Alicia,
    prove that the total number of seashells Joan has is 239. -/
theorem joan_total_seashells 
  (joan_found : ℕ) 
  (mike_gave : ℕ) 
  (alicia_gave : ℕ) 
  (h1 : joan_found = 79) 
  (h2 : mike_gave = 63) 
  (h3 : alicia_gave = 97) : 
  joan_found + mike_gave + alicia_gave = 239 := by
  sorry

end NUMINAMATH_CALUDE_joan_total_seashells_l2863_286361


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2863_286374

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_mean1 : (a 1 + a 2) / 2 = 1)
  (h_mean2 : (a 2 + a 3) / 2 = 2) :
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 1 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2863_286374


namespace NUMINAMATH_CALUDE_hyperbola_focus_to_asymptote_distance_l2863_286322

/-- Given a hyperbola and a parabola with coinciding foci, prove the distance from the hyperbola's focus to its asymptote -/
theorem hyperbola_focus_to_asymptote_distance 
  (a : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / 5 = 1 → (∃ c : ℝ, x^2 / c^2 - y^2 / 5 = 1 ∧ c^2 = 4)) 
  (h2 : ∀ x y : ℝ, y^2 = 12*x → (∃ p : ℝ × ℝ, p = (3, 0))) : 
  ∃ d : ℝ, d = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focus_to_asymptote_distance_l2863_286322


namespace NUMINAMATH_CALUDE_managers_salary_l2863_286319

/-- Proves that the manager's salary is 14100 given the conditions -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℕ) (salary_increase : ℕ) :
  num_employees = 20 →
  avg_salary = 1500 →
  salary_increase = 600 →
  (num_employees * avg_salary + (num_employees + 1) * salary_increase) = 14100 := by
  sorry

#check managers_salary

end NUMINAMATH_CALUDE_managers_salary_l2863_286319


namespace NUMINAMATH_CALUDE_octagon_area_l2863_286355

/-- Given a square with area 16 and a regular octagon with equal perimeter to the square,
    the area of the octagon is 8(1+√2) -/
theorem octagon_area (s : ℝ) (t : ℝ) : 
  s^2 = 16 →                        -- Square area is 16
  4*s = 8*t →                       -- Equal perimeters
  2*(1+Real.sqrt 2)*t^2 = 8*(1+Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_octagon_area_l2863_286355


namespace NUMINAMATH_CALUDE_bicycle_wheels_l2863_286316

theorem bicycle_wheels (num_bicycles num_tricycles tricycle_wheels total_wheels : ℕ) 
  (h1 : num_bicycles = 24)
  (h2 : num_tricycles = 14)
  (h3 : tricycle_wheels = 3)
  (h4 : total_wheels = 90)
  : ∃ bicycle_wheels : ℕ, 
    bicycle_wheels * num_bicycles + tricycle_wheels * num_tricycles = total_wheels ∧ 
    bicycle_wheels = 2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_wheels_l2863_286316


namespace NUMINAMATH_CALUDE_credit_card_balance_ratio_l2863_286311

theorem credit_card_balance_ratio : 
  ∀ (gold_limit : ℝ) (gold_balance : ℝ) (platinum_balance : ℝ),
  gold_limit > 0 →
  platinum_balance = (1/8) * (2 * gold_limit) →
  0.7083333333333334 * (2 * gold_limit) = 2 * gold_limit - (platinum_balance + gold_balance) →
  gold_balance / gold_limit = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_credit_card_balance_ratio_l2863_286311


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_y_negative_l2863_286343

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the Cartesian plane -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

theorem point_in_fourth_quadrant_y_negative (p : Point) 
  (h : p.x = 5) (h₂ : fourth_quadrant p) : p.y < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_y_negative_l2863_286343


namespace NUMINAMATH_CALUDE_right_triangle_area_l2863_286382

theorem right_triangle_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_angle : a = b) (h_side : a = 5) : (1/2) * a * b = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2863_286382


namespace NUMINAMATH_CALUDE_jumping_contest_l2863_286340

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : frog_jump = 58) :
  frog_jump - grasshopper_jump = 39 := by
  sorry

end NUMINAMATH_CALUDE_jumping_contest_l2863_286340


namespace NUMINAMATH_CALUDE_third_file_size_l2863_286302

theorem third_file_size 
  (internet_speed : ℝ) 
  (download_time : ℝ) 
  (file1_size : ℝ) 
  (file2_size : ℝ) 
  (h1 : internet_speed = 2) 
  (h2 : download_time = 2 * 60) 
  (h3 : file1_size = 80) 
  (h4 : file2_size = 90) : 
  ∃ (file3_size : ℝ), 
    file3_size = internet_speed * download_time - (file1_size + file2_size) ∧ 
    file3_size = 70 := by
  sorry

end NUMINAMATH_CALUDE_third_file_size_l2863_286302


namespace NUMINAMATH_CALUDE_group_frequency_l2863_286358

theorem group_frequency (sample_capacity : ℕ) (group_frequency_ratio : ℚ) : 
  sample_capacity = 20 →
  group_frequency_ratio = 1/4 →
  (sample_capacity : ℚ) * group_frequency_ratio = 5 := by
sorry

end NUMINAMATH_CALUDE_group_frequency_l2863_286358


namespace NUMINAMATH_CALUDE_triangle_area_is_three_l2863_286387

/-- Triangle ABC with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The line equation x - y = 5 -/
def LineEquation (p : ℝ × ℝ) : Prop :=
  p.1 - p.2 = 5

/-- The area of a triangle -/
def TriangleArea (t : Triangle) : ℝ :=
  sorry

/-- The theorem statement -/
theorem triangle_area_is_three :
  ∀ (t : Triangle),
    t.A = (3, 0) →
    t.B = (0, 3) →
    LineEquation t.C →
    TriangleArea t = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_three_l2863_286387


namespace NUMINAMATH_CALUDE_compound_weight_is_88_l2863_286348

/-- The molecular weight of the carbon part in the compound C4H8O2 -/
def carbon_weight : ℕ := 48

/-- The molecular weight of the hydrogen part in the compound C4H8O2 -/
def hydrogen_weight : ℕ := 8

/-- The molecular weight of the oxygen part in the compound C4H8O2 -/
def oxygen_weight : ℕ := 32

/-- The total molecular weight of the compound C4H8O2 -/
def total_molecular_weight : ℕ := carbon_weight + hydrogen_weight + oxygen_weight

theorem compound_weight_is_88 : total_molecular_weight = 88 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_is_88_l2863_286348


namespace NUMINAMATH_CALUDE_car_pedestrian_speed_ratio_l2863_286373

-- Define the bridge length
variable (L : ℝ)

-- Define the speeds of the pedestrian and the car
variable (v_p v_c : ℝ)

-- Assume positive speeds and bridge length
variable (h_pos_L : L > 0)
variable (h_pos_v_p : v_p > 0)
variable (h_pos_v_c : v_c > 0)

-- Define the theorem
theorem car_pedestrian_speed_ratio
  (h1 : v_c * (L / (5 * v_p)) = L) -- Car covers full bridge in time pedestrian covers 1/5
  (h2 : v_p * (L / (5 * v_p)) = L / 5) -- Pedestrian covers 1/5 bridge in same time
  : v_c / v_p = 5 :=
by sorry

end NUMINAMATH_CALUDE_car_pedestrian_speed_ratio_l2863_286373


namespace NUMINAMATH_CALUDE_angle_A_is_60_degrees_b_plus_c_range_l2863_286397

/- Define a triangle with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/- Define the condition √3a*sin(C) + a*cos(C) = c + b -/
def condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a * Real.sin t.C + t.a * Real.cos t.C = t.c + t.b

/- Theorem 1: If the condition holds, then angle A = 60° -/
theorem angle_A_is_60_degrees (t : Triangle) (h : condition t) : t.A = 60 * π / 180 := by
  sorry

/- Theorem 2: If a = √3 and the condition holds, then √3 < b + c ≤ 2√3 -/
theorem b_plus_c_range (t : Triangle) (h1 : t.a = Real.sqrt 3) (h2 : condition t) :
  Real.sqrt 3 < t.b + t.c ∧ t.b + t.c ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_60_degrees_b_plus_c_range_l2863_286397


namespace NUMINAMATH_CALUDE_max_sum_after_adding_pyramid_l2863_286347

/-- A rectangular prism -/
structure RectangularPrism :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)

/-- Properties of the resulting solid after adding a square pyramid -/
structure ResultingSolid :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)

/-- Function to calculate the resulting solid properties -/
def add_pyramid (prism : RectangularPrism) : ResultingSolid :=
  { faces := prism.faces - 1 + 4,
    edges := prism.edges + 4,
    vertices := prism.vertices + 1 }

/-- Theorem stating the maximum sum of faces, edges, and vertices -/
theorem max_sum_after_adding_pyramid (prism : RectangularPrism)
  (h1 : prism.faces = 6)
  (h2 : prism.edges = 12)
  (h3 : prism.vertices = 8) :
  let resulting := add_pyramid prism
  resulting.faces + resulting.edges + resulting.vertices = 34 :=
sorry

end NUMINAMATH_CALUDE_max_sum_after_adding_pyramid_l2863_286347


namespace NUMINAMATH_CALUDE_judgment_not_basic_structure_l2863_286328

/-- Represents the basic structures of flowcharts in algorithms -/
inductive FlowchartStructure
  | Sequential
  | Selection
  | Loop
  | Judgment

/-- The set of basic flowchart structures -/
def BasicStructures : Set FlowchartStructure :=
  {FlowchartStructure.Sequential, FlowchartStructure.Selection, FlowchartStructure.Loop}

/-- Theorem: The judgment structure is not one of the three basic structures of flowcharts -/
theorem judgment_not_basic_structure :
  FlowchartStructure.Judgment ∉ BasicStructures :=
by sorry

end NUMINAMATH_CALUDE_judgment_not_basic_structure_l2863_286328


namespace NUMINAMATH_CALUDE_mountain_paths_theorem_l2863_286337

/-- Number of paths to the mountain top -/
def num_paths : ℕ := 5

/-- Number of people ascending and descending -/
def num_people : ℕ := 2

/-- Calculates the number of ways to ascend and descend the mountain for scenario a -/
def scenario_a (n p : ℕ) : ℕ := 
  Nat.choose n p * Nat.choose (n - p) p

/-- Calculates the number of ways to ascend and descend the mountain for scenario b -/
def scenario_b (n p : ℕ) : ℕ := 
  Nat.choose n p * Nat.choose n p

/-- Calculates the number of ways to ascend and descend the mountain for scenario c -/
def scenario_c (n p : ℕ) : ℕ := 
  (n ^ p) * (n ^ p)

/-- Calculates the number of ways to ascend and descend the mountain for scenario d -/
def scenario_d (n p : ℕ) : ℕ := 
  (Nat.factorial n / Nat.factorial (n - p)) * (Nat.factorial (n - p) / Nat.factorial (n - 2*p))

/-- Calculates the number of ways to ascend and descend the mountain for scenario e -/
def scenario_e (n p : ℕ) : ℕ := 
  (Nat.factorial n / Nat.factorial (n - p)) * (Nat.factorial n / Nat.factorial (n - p))

/-- Calculates the number of ways to ascend and descend the mountain for scenario f -/
def scenario_f (n p : ℕ) : ℕ := 
  (n ^ p) * (n ^ p)

theorem mountain_paths_theorem :
  scenario_a num_paths num_people = 30 ∧
  scenario_b num_paths num_people = 100 ∧
  scenario_c num_paths num_people = 625 ∧
  scenario_d num_paths num_people = 120 ∧
  scenario_e num_paths num_people = 400 ∧
  scenario_f num_paths num_people = 625 :=
by sorry

end NUMINAMATH_CALUDE_mountain_paths_theorem_l2863_286337


namespace NUMINAMATH_CALUDE_binomial_15_12_l2863_286349

theorem binomial_15_12 : Nat.choose 15 12 = 455 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_12_l2863_286349


namespace NUMINAMATH_CALUDE_sum_of_variables_l2863_286363

theorem sum_of_variables (x y z : ℚ) 
  (eq1 : y + z = 20 - 5*x)
  (eq2 : x + z = -18 - 5*y)
  (eq3 : x + y = 10 - 5*z) :
  3*x + 3*y + 3*z = 36/7 := by sorry

end NUMINAMATH_CALUDE_sum_of_variables_l2863_286363


namespace NUMINAMATH_CALUDE_log_approximation_l2863_286359

-- Define the base of the logarithm
def base : ℝ := 8

-- Define the given logarithmic value
def log_value : ℝ := 2.75

-- Define the approximate result
def approx_result : ℝ := 215

-- Define a tolerance for the approximation
def tolerance : ℝ := 0.1

-- Theorem statement
theorem log_approximation (y : ℝ) (h : Real.log y / Real.log base = log_value) :
  |y - approx_result| < tolerance :=
sorry

end NUMINAMATH_CALUDE_log_approximation_l2863_286359


namespace NUMINAMATH_CALUDE_fruit_arrangement_count_l2863_286381

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem fruit_arrangement_count : 
  let total_fruits : ℕ := 8
  let apples : ℕ := 3
  let oranges : ℕ := 2
  let bananas : ℕ := 3
  factorial total_fruits / (factorial apples * factorial oranges * factorial bananas) = 560 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangement_count_l2863_286381


namespace NUMINAMATH_CALUDE_samples_per_box_l2863_286393

theorem samples_per_box (boxes_opened : ℕ) (samples_leftover : ℕ) (customers : ℕ) : 
  boxes_opened = 12 → samples_leftover = 5 → customers = 235 → 
  ∃ (samples_per_box : ℕ), samples_per_box * boxes_opened - samples_leftover = customers ∧ samples_per_box = 20 := by
  sorry

end NUMINAMATH_CALUDE_samples_per_box_l2863_286393


namespace NUMINAMATH_CALUDE_cricket_bat_profit_l2863_286388

/-- Calculates the profit amount for a cricket bat sale -/
theorem cricket_bat_profit (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 900 →
  profit_percentage = 33.33 →
  ∃ (cost_price : ℝ), 
    cost_price > 0 ∧
    selling_price = cost_price * (1 + profit_percentage / 100) ∧
    selling_price - cost_price = 225 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_l2863_286388


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2863_286346

/-- Given a hyperbola with equation x²/48 - y²/16 = 1, 
    the distance between its vertices is 8√3. -/
theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ), 
  x^2 / 48 - y^2 / 16 = 1 →
  ∃ (d : ℝ), d = 8 * Real.sqrt 3 ∧ d = 2 * Real.sqrt 48 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2863_286346


namespace NUMINAMATH_CALUDE_intersection_M_N_l2863_286365

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x^2 - 2*x ≥ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2863_286365


namespace NUMINAMATH_CALUDE_probability_specific_arrangement_l2863_286310

def total_tiles : ℕ := 7
def x_tiles : ℕ := 4
def o_tiles : ℕ := 3

theorem probability_specific_arrangement :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles : ℚ) = 1 / 35 :=
by sorry

end NUMINAMATH_CALUDE_probability_specific_arrangement_l2863_286310


namespace NUMINAMATH_CALUDE_coin_value_equality_l2863_286377

theorem coin_value_equality (n : ℕ) : 
  (20 * 25 + 10 * 10 = 10 * 25 + n * 10) → n = 35 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_equality_l2863_286377


namespace NUMINAMATH_CALUDE_spiral_stripe_length_l2863_286301

/-- The length of a spiral stripe on a right circular cylinder -/
theorem spiral_stripe_length (base_circumference height : ℝ) (h1 : base_circumference = 18) (h2 : height = 8) :
  Real.sqrt (height^2 + (2 * base_circumference)^2) = Real.sqrt 1360 := by
  sorry

end NUMINAMATH_CALUDE_spiral_stripe_length_l2863_286301


namespace NUMINAMATH_CALUDE_five_million_times_eight_million_l2863_286321

theorem five_million_times_eight_million :
  (5 * (10 : ℕ)^6) * (8 * (10 : ℕ)^6) = 40 * (10 : ℕ)^12 := by
  sorry

end NUMINAMATH_CALUDE_five_million_times_eight_million_l2863_286321


namespace NUMINAMATH_CALUDE_division_result_l2863_286390

theorem division_result : (4.036 : ℝ) / 0.04 = 100.9 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l2863_286390


namespace NUMINAMATH_CALUDE_digit_difference_729_l2863_286386

def base_3_digits (n : ℕ) : ℕ := 
  Nat.log 3 n + 1

def base_8_digits (n : ℕ) : ℕ := 
  Nat.log 8 n + 1

theorem digit_difference_729 : 
  base_3_digits 729 - base_8_digits 729 = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_729_l2863_286386


namespace NUMINAMATH_CALUDE_parabolas_intersection_k_l2863_286370

/-- Two different parabolas that intersect on the x-axis -/
def intersecting_parabolas (k : ℝ) : Prop :=
  ∃ x : ℝ, 
    (x^2 + k*x + 1 = 0) ∧ 
    (x^2 - x - k = 0) ∧
    (x^2 + k*x + 1 ≠ x^2 - x - k)

/-- The value of k for which the parabolas intersect on the x-axis -/
theorem parabolas_intersection_k : 
  ∃! k : ℝ, intersecting_parabolas k ∧ k = 2 :=
sorry

end NUMINAMATH_CALUDE_parabolas_intersection_k_l2863_286370


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2863_286342

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2863_286342


namespace NUMINAMATH_CALUDE_disease_mortality_percentage_l2863_286379

theorem disease_mortality_percentage (population : ℝ) (affected_percentage : ℝ) (mortality_rate : ℝ) 
  (h1 : affected_percentage = 15)
  (h2 : mortality_rate = 8) :
  (affected_percentage / 100) * (mortality_rate / 100) * 100 = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_disease_mortality_percentage_l2863_286379


namespace NUMINAMATH_CALUDE_fraction_equality_l2863_286399

theorem fraction_equality (w z : ℝ) (h : (1/w + 1/z) / (1/w - 1/z) = 2014) : 
  (w + z) / (w - z) = -2014 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2863_286399


namespace NUMINAMATH_CALUDE_complex_sum_argument_l2863_286326

/-- The argument of the sum of five complex exponentials -/
theorem complex_sum_argument :
  let z₁ := Complex.exp (11 * π * Complex.I / 120)
  let z₂ := Complex.exp (31 * π * Complex.I / 120)
  let z₃ := Complex.exp (51 * π * Complex.I / 120)
  let z₄ := Complex.exp (71 * π * Complex.I / 120)
  let z₅ := Complex.exp (91 * π * Complex.I / 120)
  Complex.arg (z₁ + z₂ + z₃ + z₄ + z₅) = 17 * π / 40 :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_argument_l2863_286326


namespace NUMINAMATH_CALUDE_moving_points_minimum_distance_l2863_286332

/-- Two points moving along perpendicular lines towards their intersection --/
theorem moving_points_minimum_distance 
  (a b v v₁ : ℝ) (ha : a > 0) (hb : b > 0) (hv : v > 0) (hv₁ : v₁ > 0) :
  let min_distance := |b * v - a * v₁| / Real.sqrt (v^2 + v₁^2)
  let vertex_distance_diff := |a * v^2 + a * b * v * v₁| / (v^2 + v₁^2)
  let equal_speed_min_distance := |a - b| / Real.sqrt 2
  let equal_speed_time := (a + b) / (2 * v)
  let equal_speed_distance_a := (a - b) / 2
  let equal_speed_distance_b := (b - a) / 2
  ∃ (t : ℝ), 
    (∀ (s : ℝ), 
      Real.sqrt ((a - v * s)^2 + (b - v₁ * s)^2) ≥ min_distance) ∧
    (Real.sqrt ((a - v * t)^2 + (b - v₁ * t)^2) = min_distance) ∧
    (|(a - v * t) - (b - v₁ * t)| = vertex_distance_diff) ∧
    (v = v₁ → 
      min_distance = equal_speed_min_distance ∧
      t = equal_speed_time ∧
      a - v * t = equal_speed_distance_a ∧
      b - v₁ * t = equal_speed_distance_b) := by sorry

end NUMINAMATH_CALUDE_moving_points_minimum_distance_l2863_286332


namespace NUMINAMATH_CALUDE_class_gender_ratio_l2863_286307

theorem class_gender_ratio (female_count : ℕ) (male_count : ℕ) 
  (h1 : female_count = 28)
  (h2 : female_count = male_count + 6) :
  let total_count := female_count + male_count
  (female_count : ℚ) / (male_count : ℚ) = 14 / 11 ∧ 
  (male_count : ℚ) / (total_count : ℚ) = 11 / 25 := by
sorry

end NUMINAMATH_CALUDE_class_gender_ratio_l2863_286307


namespace NUMINAMATH_CALUDE_tree_initial_height_l2863_286317

/-- Represents the height of a tree over time -/
def TreeHeight (initial_height : ℝ) (growth_rate : ℝ) (initial_age : ℝ) (current_age : ℝ) : ℝ :=
  initial_height + growth_rate * (current_age - initial_age)

theorem tree_initial_height :
  ∀ (initial_height : ℝ) (growth_rate : ℝ) (initial_age : ℝ) (current_age : ℝ) (current_height : ℝ),
  growth_rate = 3 →
  initial_age = 1 →
  current_age = 7 →
  current_height = 23 →
  TreeHeight initial_height growth_rate initial_age current_age = current_height →
  initial_height = 5 := by
sorry

end NUMINAMATH_CALUDE_tree_initial_height_l2863_286317


namespace NUMINAMATH_CALUDE_sequence_inequality_l2863_286356

theorem sequence_inequality (a : ℕ+ → ℝ) 
  (h : ∀ (k m : ℕ+), |a (k + m) - a k - a m| ≤ 1) :
  ∀ (k m : ℕ+), |a k / k.val - a m / m.val| < 1 / k.val + 1 / m.val := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l2863_286356


namespace NUMINAMATH_CALUDE_circumcircle_equation_l2863_286352

-- Define the vertices of the triangle
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (5, 3)
def C : ℝ × ℝ := (3, -1)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 1)^2 = 5

-- Theorem statement
theorem circumcircle_equation :
  (circle_equation A.1 A.2) ∧
  (circle_equation B.1 B.2) ∧
  (circle_equation C.1 C.2) ∧
  (∀ (a b r : ℝ), (
    ((A.1 - a)^2 + (A.2 - b)^2 = r^2) ∧
    ((B.1 - a)^2 + (B.2 - b)^2 = r^2) ∧
    ((C.1 - a)^2 + (C.2 - b)^2 = r^2)
  ) → a = 4 ∧ b = 1 ∧ r^2 = 5) :=
sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l2863_286352


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l2863_286360

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 = 20 → speed2 = 30 → (speed1 + speed2) / 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l2863_286360


namespace NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l2863_286369

theorem geometric_sum_first_six_terms :
  let a : ℚ := 1/2  -- First term
  let r : ℚ := 1/3  -- Common ratio
  let n : ℕ := 6    -- Number of terms
  let S : ℚ := a * (1 - r^n) / (1 - r)  -- Formula for sum of geometric series
  S = 364/243
  := by sorry

end NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l2863_286369


namespace NUMINAMATH_CALUDE_textbook_profit_l2863_286323

/-- The profit of a textbook sale -/
def profit (cost_price selling_price : ℝ) : ℝ :=
  selling_price - cost_price

/-- Theorem: The profit of a textbook is $11 given that its cost price is $44 and its selling price is $55 -/
theorem textbook_profit :
  let cost_price : ℝ := 44
  let selling_price : ℝ := 55
  profit cost_price selling_price = 11 := by
  sorry

end NUMINAMATH_CALUDE_textbook_profit_l2863_286323


namespace NUMINAMATH_CALUDE_belle_weekly_treat_cost_l2863_286368

def cost_brand_a : ℚ := 0.25
def cost_brand_b : ℚ := 0.35
def cost_small_rawhide : ℚ := 1
def cost_large_rawhide : ℚ := 1.5

def odd_day_cost : ℚ :=
  3 * cost_brand_a + 2 * cost_brand_b + cost_small_rawhide + cost_large_rawhide

def even_day_cost : ℚ :=
  4 * cost_brand_a + 2 * cost_small_rawhide

def days_in_week : ℕ := 7
def odd_days_in_week : ℕ := 4
def even_days_in_week : ℕ := 3

theorem belle_weekly_treat_cost :
  odd_days_in_week * odd_day_cost + even_days_in_week * even_day_cost = 24.8 := by
  sorry

end NUMINAMATH_CALUDE_belle_weekly_treat_cost_l2863_286368


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2863_286308

theorem arithmetic_expression_evaluation : 3 * 4 + (2 * 5)^2 - 6 * 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2863_286308


namespace NUMINAMATH_CALUDE_rationalization_sum_l2863_286396

/-- Represents a cube root expression of the form (a * ∛b) / c --/
structure CubeRootExpression where
  a : ℤ
  b : ℕ
  c : ℕ
  c_pos : c > 0
  b_not_perfect_cube : ∀ (p : ℕ), Prime p → ¬(p^3 ∣ b)

/-- Rationalizes the denominator of 5 / (3 * ∛7) --/
def rationalize_denominator : CubeRootExpression :=
  { a := 5
    b := 49
    c := 21
    c_pos := by sorry
    b_not_perfect_cube := by sorry }

/-- The sum of a, b, and c in the rationalized expression --/
def sum_of_parts (expr : CubeRootExpression) : ℤ :=
  expr.a + expr.b + expr.c

theorem rationalization_sum :
  sum_of_parts rationalize_denominator = 75 := by sorry

end NUMINAMATH_CALUDE_rationalization_sum_l2863_286396


namespace NUMINAMATH_CALUDE_quadratic_roots_relations_l2863_286345

theorem quadratic_roots_relations (a : ℝ) :
  let x₁ : ℝ := (1 + Real.sqrt (5 - 4*a)) / 2
  let x₂ : ℝ := (1 - Real.sqrt (5 - 4*a)) / 2
  (x₁*x₂ + x₁ + x₂ - a = 0) ∧ (x₁*x₂ - a*(x₁ + x₂) + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relations_l2863_286345


namespace NUMINAMATH_CALUDE_intersection_line_is_canonical_l2863_286314

-- Define the two planes
def plane1 (x y z : ℝ) : Prop := 2*x + 3*y + z + 6 = 0
def plane2 (x y z : ℝ) : Prop := x - 3*y - 2*z + 3 = 0

-- Define the canonical form of the line
def canonical_line (x y z : ℝ) : Prop := (x + 3)/(-3) = y/5 ∧ y/5 = z/(-9)

-- Theorem statement
theorem intersection_line_is_canonical :
  ∀ x y z : ℝ, plane1 x y z → plane2 x y z → canonical_line x y z :=
sorry

end NUMINAMATH_CALUDE_intersection_line_is_canonical_l2863_286314


namespace NUMINAMATH_CALUDE_julias_initial_money_l2863_286344

theorem julias_initial_money (initial_money : ℝ) : 
  initial_money / 2 - (initial_money / 2) / 4 = 15 → initial_money = 40 := by
  sorry

end NUMINAMATH_CALUDE_julias_initial_money_l2863_286344


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2863_286375

/-- The eccentricity of a hyperbola with equation x²/5 - y²/4 = 1 is 3√5/5 -/
theorem hyperbola_eccentricity : 
  let a : ℝ := Real.sqrt 5
  let b : ℝ := 2
  let c : ℝ := 3
  let e : ℝ := c / a
  (∀ x y : ℝ, x^2 / 5 - y^2 / 4 = 1) →
  e = 3 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2863_286375


namespace NUMINAMATH_CALUDE_geometric_sequence_cosine_l2863_286324

/-- 
Given a geometric sequence {an} with common ratio √2,
prove that if sn(a7a8) = 3/5, then cos(2a5) = 7/25
-/
theorem geometric_sequence_cosine (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * Real.sqrt 2) →  -- Common ratio is √2
  (∃ sn : ℝ, sn * (a 7 * a 8) = 3/5) →    -- sn(a7a8) = 3/5
  Real.cos (2 * a 5) = 7/25 :=             -- cos(2a5) = 7/25
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_cosine_l2863_286324


namespace NUMINAMATH_CALUDE_problem_solution_l2863_286303

def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| - |x + a|

theorem problem_solution :
  (∀ x : ℝ, f 1 x + x > 0 ↔ (-3 < x ∧ x < 1) ∨ x > 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x ≤ 3) ↔ -5 ≤ a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2863_286303


namespace NUMINAMATH_CALUDE_average_temperature_l2863_286320

def temperatures : List ℤ := [-36, 13, -15, -10]

theorem average_temperature (temps := temperatures) :
  (temps.sum : ℚ) / temps.length = -12 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_l2863_286320


namespace NUMINAMATH_CALUDE_camping_site_campers_l2863_286304

theorem camping_site_campers (total : ℕ) (last_week : ℕ) : 
  total = 150 → last_week = 80 → ∃ (three_weeks_ago two_weeks_ago : ℕ), 
    two_weeks_ago = three_weeks_ago + 10 ∧ 
    total = three_weeks_ago + two_weeks_ago + last_week ∧
    two_weeks_ago = 40 := by sorry

end NUMINAMATH_CALUDE_camping_site_campers_l2863_286304


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l2863_286353

theorem shaded_region_perimeter (r : ℝ) (h : r = 7) : 
  2 * r + 3 * π * r / 2 = 14 + 10.5 * π := by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l2863_286353


namespace NUMINAMATH_CALUDE_seashells_given_to_sam_l2863_286376

theorem seashells_given_to_sam (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 70) 
  (h2 : remaining_seashells = 27) : 
  initial_seashells - remaining_seashells = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_seashells_given_to_sam_l2863_286376


namespace NUMINAMATH_CALUDE_remainder_9876543210_mod_101_l2863_286350

theorem remainder_9876543210_mod_101 : 9876543210 % 101 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_9876543210_mod_101_l2863_286350


namespace NUMINAMATH_CALUDE_min_value_at_one_third_l2863_286364

def y (x : ℝ) : ℝ := |x - 1| + |2*x - 1| + |3*x - 1| + |4*x - 1| + |5*x - 1|

theorem min_value_at_one_third :
  ∀ x : ℝ, y (1/3 : ℝ) ≤ y x := by
  sorry

end NUMINAMATH_CALUDE_min_value_at_one_third_l2863_286364


namespace NUMINAMATH_CALUDE_binomial_10_3_l2863_286341

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l2863_286341


namespace NUMINAMATH_CALUDE_chess_club_election_l2863_286329

theorem chess_club_election (total_candidates : ℕ) (officer_positions : ℕ) (past_officers : ℕ) :
  total_candidates = 20 →
  officer_positions = 6 →
  past_officers = 8 →
  (Nat.choose total_candidates officer_positions - 
   Nat.choose (total_candidates - past_officers) officer_positions) = 37836 :=
by sorry

end NUMINAMATH_CALUDE_chess_club_election_l2863_286329


namespace NUMINAMATH_CALUDE_R_equals_triangle_interior_l2863_286367

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The polynomial z^2 + az + b -/
def polynomial (a b : ℝ) (z : ℂ) : ℂ := z^2 + a*z + b

/-- The region R -/
def R : Set (ℝ × ℝ) :=
  {p | ∀ z, polynomial p.1 p.2 z = 0 → Complex.abs z < 1}

/-- The triangle ABC -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {p | p.1 > -2 ∧ p.1 < 2 ∧ p.2 > -1 ∧ p.2 < 1 ∧ p.2 < (1 - p.1/2)}

/-- The theorem stating that R is equivalent to the interior of triangle ABC -/
theorem R_equals_triangle_interior : R = triangle_ABC := by sorry

end NUMINAMATH_CALUDE_R_equals_triangle_interior_l2863_286367


namespace NUMINAMATH_CALUDE_rowing_coach_votes_l2863_286318

theorem rowing_coach_votes (num_coaches : ℕ) (votes_per_rower : ℕ) (votes_per_coach : ℕ) :
  num_coaches = 50 →
  votes_per_rower = 4 →
  votes_per_coach = 7 →
  ∃ (num_rowers : ℕ), num_rowers * votes_per_rower = num_coaches * votes_per_coach ∧ 
                       num_rowers = 88 := by
  sorry

end NUMINAMATH_CALUDE_rowing_coach_votes_l2863_286318


namespace NUMINAMATH_CALUDE_employed_females_percentage_l2863_286336

theorem employed_females_percentage
  (total_population : ℕ)
  (employed_percentage : ℚ)
  (employed_males_percentage : ℚ)
  (h1 : employed_percentage = 60 / 100)
  (h2 : employed_males_percentage = 48 / 100)
  : (employed_percentage - employed_males_percentage) / employed_percentage = 20 / 100 := by
  sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l2863_286336


namespace NUMINAMATH_CALUDE_expected_replacement_seeds_l2863_286312

theorem expected_replacement_seeds :
  let germination_prob : ℝ := 0.9
  let initial_seeds : ℕ := 1000
  let replacement_per_failure : ℕ := 2
  let non_germination_prob : ℝ := 1 - germination_prob
  let expected_non_germinating : ℝ := initial_seeds * non_germination_prob
  let expected_replacements : ℝ := expected_non_germinating * replacement_per_failure
  expected_replacements = 200 := by sorry

end NUMINAMATH_CALUDE_expected_replacement_seeds_l2863_286312


namespace NUMINAMATH_CALUDE_words_with_consonants_l2863_286330

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 6

/-- The number of consonants in the alphabet -/
def consonant_count : ℕ := 4

/-- The number of vowels in the alphabet -/
def vowel_count : ℕ := 2

/-- The length of the words we're considering -/
def word_length : ℕ := 5

/-- The total number of possible words -/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of words containing only vowels -/
def vowel_only_words : ℕ := vowel_count ^ word_length

theorem words_with_consonants :
  total_words - vowel_only_words = 7744 :=
sorry

end NUMINAMATH_CALUDE_words_with_consonants_l2863_286330


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l2863_286315

/-- Given three consecutive odd integers where the sum of the first and third is 150,
    prove that the second integer is 75. -/
theorem consecutive_odd_integers_sum (n : ℤ) : 
  (∃ (a b c : ℤ), a + 2 = b ∧ b + 2 = c ∧ Odd a ∧ Odd b ∧ Odd c ∧ a + c = 150) →
  b = 75 := by
sorry


end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l2863_286315


namespace NUMINAMATH_CALUDE_intercept_plane_equation_point_on_intercept_plane_l2863_286383

/-- A plane in 3D space with intercepts a, b, c on x, y, z axes respectively --/
structure InterceptPlane where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

/-- The equation of a plane given its intercepts --/
def plane_equation (p : InterceptPlane) (x y z : ℝ) : Prop :=
  x / p.a + y / p.b + z / p.c = 1

/-- Theorem stating that the given equation represents the plane with given intercepts --/
theorem intercept_plane_equation (p : InterceptPlane) :
  ∀ x y z : ℝ, (x = p.a ∧ y = 0 ∧ z = 0) ∨ (x = 0 ∧ y = p.b ∧ z = 0) ∨ (x = 0 ∧ y = 0 ∧ z = p.c) →
  plane_equation p x y z := by
  sorry

/-- Theorem stating that any point satisfying the equation lies on the plane --/
theorem point_on_intercept_plane (p : InterceptPlane) :
  ∀ x y z : ℝ, plane_equation p x y z →
  ∃ t u v : ℝ, t + u + v = 1 ∧ x = t * p.a ∧ y = u * p.b ∧ z = v * p.c := by
  sorry

end NUMINAMATH_CALUDE_intercept_plane_equation_point_on_intercept_plane_l2863_286383
