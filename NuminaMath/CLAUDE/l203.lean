import Mathlib

namespace NUMINAMATH_CALUDE_highest_percentage_increase_city_H_l203_20338

structure City where
  name : String
  pop1990 : ℕ
  pop2000 : ℕ

def percentageIncrease (c : City) : ℚ :=
  ((c.pop2000 - c.pop1990) : ℚ) / (c.pop1990 : ℚ) * 100

def cities : List City := [
  ⟨"F", 60000, 78000⟩,
  ⟨"G", 80000, 96000⟩,
  ⟨"H", 70000, 91000⟩,
  ⟨"I", 85000, 94500⟩,
  ⟨"J", 95000, 114000⟩
]

theorem highest_percentage_increase_city_H :
  ∃ (c : City), c ∈ cities ∧ c.name = "H" ∧
  ∀ (other : City), other ∈ cities → percentageIncrease c ≥ percentageIncrease other :=
by sorry

end NUMINAMATH_CALUDE_highest_percentage_increase_city_H_l203_20338


namespace NUMINAMATH_CALUDE_parabola_vertex_l203_20326

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2

-- State the theorem
theorem parabola_vertex :
  ∃ (x y : ℝ), (x = 0 ∧ y = -2) ∧
  (∀ (x' : ℝ), parabola x' ≥ parabola x) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l203_20326


namespace NUMINAMATH_CALUDE_largest_eight_digit_with_even_digits_l203_20311

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop := n ≥ 10000000 ∧ n < 100000000

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k : Nat, n / (10^k) % 10 = d

theorem largest_eight_digit_with_even_digits :
  (∀ m : Nat, is_eight_digit m ∧ contains_all_even_digits m → m ≤ 99986420) ∧
  is_eight_digit 99986420 ∧
  contains_all_even_digits 99986420 :=
sorry

end NUMINAMATH_CALUDE_largest_eight_digit_with_even_digits_l203_20311


namespace NUMINAMATH_CALUDE_triangle_problem_l203_20330

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) : 
  (Real.cos t.B * Real.cos t.C - Real.sin t.B * Real.sin t.C = - (1/2)) →
  t.a = 2 →
  (1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 3) →
  (t.A = π/3 ∧ t.b = 2 ∧ t.c = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l203_20330


namespace NUMINAMATH_CALUDE_trig_sum_equals_one_l203_20393

theorem trig_sum_equals_one :
  let angle_to_real (θ : ℤ) : ℝ := (θ % 360 : ℝ) * Real.pi / 180
  Real.sin (angle_to_real (-120)) * Real.cos (angle_to_real 1290) +
  Real.cos (angle_to_real (-1020)) * Real.sin (angle_to_real (-1050)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equals_one_l203_20393


namespace NUMINAMATH_CALUDE_fencing_calculation_l203_20340

/-- Calculates the required fencing for a rectangular field -/
theorem fencing_calculation (area : ℝ) (uncovered_side : ℝ) (h1 : area = 680) (h2 : uncovered_side = 20) :
  let width := area / uncovered_side
  2 * width + uncovered_side = 88 :=
sorry

end NUMINAMATH_CALUDE_fencing_calculation_l203_20340


namespace NUMINAMATH_CALUDE_cookies_per_sheet_l203_20366

theorem cookies_per_sheet (members : ℕ) (sheets_per_member : ℕ) (total_cookies : ℕ) :
  members = 100 →
  sheets_per_member = 10 →
  total_cookies = 16000 →
  total_cookies / (members * sheets_per_member) = 16 :=
by sorry

end NUMINAMATH_CALUDE_cookies_per_sheet_l203_20366


namespace NUMINAMATH_CALUDE_not_algebraic_expression_l203_20398

-- Define what constitutes an algebraic expression
def is_algebraic_expression (e : Prop) : Prop :=
  ¬(∃ (x : ℝ), e ↔ x = 1)

-- Define the given expressions
def pi_expr : Prop := True
def x_equals_1 : Prop := ∃ (x : ℝ), x = 1
def one_over_x : Prop := True
def sqrt_3 : Prop := True

-- Theorem statement
theorem not_algebraic_expression :
  is_algebraic_expression pi_expr ∧
  is_algebraic_expression one_over_x ∧
  is_algebraic_expression sqrt_3 ∧
  ¬(is_algebraic_expression x_equals_1) :=
sorry

end NUMINAMATH_CALUDE_not_algebraic_expression_l203_20398


namespace NUMINAMATH_CALUDE_exactly_two_true_with_converse_l203_20360

-- Define the propositions
def proposition1 (a b : ℝ) : Prop := (a / b < 1) → (a < b)
def proposition2 (sides : Fin 4 → ℝ) : Prop := ∀ i j, sides i = sides j
def proposition3 (angles : Fin 3 → ℝ) : Prop := angles 0 = angles 1

-- Define the converses
def converse1 (a b : ℝ) : Prop := (a < b) → (a / b < 1)
def converse2 (sides : Fin 4 → ℝ) : Prop := (∀ i j, sides i = sides j) → (∃ r : ℝ, ∀ i, sides i = r)
def converse3 (angles : Fin 3 → ℝ) : Prop := (angles 0 = angles 1) → (∃ s : ℝ, angles 0 = s ∧ angles 1 = s)

-- Theorem statement
theorem exactly_two_true_with_converse :
  ∃! n : ℕ, n = 2 ∧
  (∀ a b : ℝ, proposition1 a b ∧ converse1 a b) ∨
  (∀ sides : Fin 4 → ℝ, proposition2 sides ∧ converse2 sides) ∨
  (∀ angles : Fin 3 → ℝ, proposition3 angles ∧ converse3 angles) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_true_with_converse_l203_20360


namespace NUMINAMATH_CALUDE_silverware_per_setting_l203_20313

/-- Proves the number of pieces of silverware per setting for a catering event -/
theorem silverware_per_setting :
  let silverware_weight : ℕ := 4  -- weight of each piece of silverware in ounces
  let plate_weight : ℕ := 12  -- weight of each plate in ounces
  let plates_per_setting : ℕ := 2  -- number of plates per setting
  let tables : ℕ := 15  -- number of tables
  let settings_per_table : ℕ := 8  -- number of settings per table
  let backup_settings : ℕ := 20  -- number of backup settings
  let total_weight : ℕ := 5040  -- total weight of all settings in ounces

  let total_settings : ℕ := tables * settings_per_table + backup_settings
  let plate_weight_per_setting : ℕ := plate_weight * plates_per_setting

  ∃ (silverware_per_setting : ℕ),
    silverware_per_setting * silverware_weight * total_settings +
    plate_weight_per_setting * total_settings = total_weight ∧
    silverware_per_setting = 3 :=
by sorry

end NUMINAMATH_CALUDE_silverware_per_setting_l203_20313


namespace NUMINAMATH_CALUDE_tv_cost_is_1060_l203_20333

-- Define the given values
def total_initial_purchase : ℝ := 3000
def returned_bike_cost : ℝ := 500
def toaster_cost : ℝ := 100
def total_out_of_pocket : ℝ := 2020

-- Define the TV cost as a variable
def tv_cost : ℝ := sorry

-- Define the sold bike cost
def sold_bike_cost : ℝ := returned_bike_cost * 1.2

-- Define the sale price of the sold bike
def sold_bike_sale_price : ℝ := sold_bike_cost * 0.8

-- Theorem stating that the TV cost is $1060
theorem tv_cost_is_1060 :
  tv_cost = 1060 :=
by
  sorry

#check tv_cost_is_1060

end NUMINAMATH_CALUDE_tv_cost_is_1060_l203_20333


namespace NUMINAMATH_CALUDE_inequality_solution_l203_20300

theorem inequality_solution (x : ℝ) (h : x ≥ 0) :
  2021 * (x^2020)^(1/202) - 1 ≥ 2020 * x ↔ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l203_20300


namespace NUMINAMATH_CALUDE_cube_edge_length_l203_20358

theorem cube_edge_length : ∃ s : ℝ, s > 0 ∧ s^3 = 6 * s^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l203_20358


namespace NUMINAMATH_CALUDE_altitude_segment_length_l203_20308

/-- Represents an acute triangle with two altitudes dividing the sides -/
structure AcuteTriangleWithAltitudes where
  /-- Length of one segment created by an altitude -/
  segment1 : ℝ
  /-- Length of another segment created by an altitude -/
  segment2 : ℝ
  /-- Length of a third segment created by an altitude -/
  segment3 : ℝ
  /-- Length of the fourth segment created by an altitude -/
  segment4 : ℝ
  /-- The triangle is acute -/
  acute : segment1 > 0 ∧ segment2 > 0 ∧ segment3 > 0 ∧ segment4 > 0

/-- The theorem stating that for the given acute triangle with altitudes, the fourth segment length is 4.5 -/
theorem altitude_segment_length (t : AcuteTriangleWithAltitudes) 
  (h1 : t.segment1 = 4) 
  (h2 : t.segment2 = 6) 
  (h3 : t.segment3 = 3) : 
  t.segment4 = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_altitude_segment_length_l203_20308


namespace NUMINAMATH_CALUDE_x_positive_necessary_not_sufficient_l203_20371

theorem x_positive_necessary_not_sufficient :
  (∀ x : ℝ, (x - 2) * (x - 4) < 0 → x > 0) ∧
  (∃ x : ℝ, x > 0 ∧ (x - 2) * (x - 4) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_positive_necessary_not_sufficient_l203_20371


namespace NUMINAMATH_CALUDE_product_of_quarters_l203_20384

theorem product_of_quarters : (0.25 : ℝ) * 0.75 = 0.1875 := by
  sorry

end NUMINAMATH_CALUDE_product_of_quarters_l203_20384


namespace NUMINAMATH_CALUDE_opposite_of_five_l203_20318

theorem opposite_of_five : 
  ∃ x : ℤ, (5 + x = 0) ∧ (x = -5) := by
sorry

end NUMINAMATH_CALUDE_opposite_of_five_l203_20318


namespace NUMINAMATH_CALUDE_feline_sanctuary_count_l203_20399

/-- The total number of big cats in a feline sanctuary -/
def total_big_cats (lions tigers cougars jaguars leopards : ℕ) : ℕ :=
  lions + tigers + cougars + jaguars + leopards

/-- Theorem: The total number of big cats in the feline sanctuary is 59 -/
theorem feline_sanctuary_count : ∃ (cougars leopards : ℕ),
  let lions : ℕ := 12
  let tigers : ℕ := 14
  let jaguars : ℕ := 8
  cougars = ((lions + tigers) + 2) / 3 ∧
  leopards = 2 * jaguars ∧
  total_big_cats lions tigers cougars jaguars leopards = 59 := by
  sorry


end NUMINAMATH_CALUDE_feline_sanctuary_count_l203_20399


namespace NUMINAMATH_CALUDE_product_condition_l203_20343

theorem product_condition (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) ∧
  (∃ a b : ℝ, a * b > 1 ∧ ¬(a > 1 ∧ b > 1)) :=
sorry

end NUMINAMATH_CALUDE_product_condition_l203_20343


namespace NUMINAMATH_CALUDE_overlapping_squares_area_l203_20329

/-- Represents a square sheet of paper --/
structure Square :=
  (side_length : ℝ)

/-- Represents the configuration of three overlapping squares --/
structure OverlappingSquares :=
  (base : Square)
  (middle_rotation : ℝ)
  (top_rotation : ℝ)

/-- Calculates the area of the resulting polygon --/
def polygon_area (config : OverlappingSquares) : ℝ :=
  sorry

/-- The main theorem --/
theorem overlapping_squares_area :
  let config := OverlappingSquares.mk (Square.mk 6) (30 * π / 180) (60 * π / 180)
  polygon_area config = 108 - 36 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_squares_area_l203_20329


namespace NUMINAMATH_CALUDE_survey_sampling_suitability_mainland_survey_suitable_l203_20357

/-- Represents a survey with its characteristics -/
structure Survey where
  population_size : ℕ
  requires_comprehensive : Bool
  is_safety_critical : Bool

/-- Defines when a survey is suitable for sampling -/
def suitable_for_sampling (s : Survey) : Prop :=
  s.population_size > 1000 ∧ ¬s.requires_comprehensive ∧ ¬s.is_safety_critical

/-- Theorem stating the condition for a survey to be suitable for sampling -/
theorem survey_sampling_suitability (s : Survey) :
  suitable_for_sampling s ↔
    s.population_size > 1000 ∧ ¬s.requires_comprehensive ∧ ¬s.is_safety_critical := by
  sorry

/-- The mainland population survey (Option C) -/
def mainland_survey : Survey :=
  { population_size := 1000000,  -- Large population
    requires_comprehensive := false,
    is_safety_critical := false }

/-- Theorem stating that the mainland survey is suitable for sampling -/
theorem mainland_survey_suitable :
  suitable_for_sampling mainland_survey := by
  sorry

end NUMINAMATH_CALUDE_survey_sampling_suitability_mainland_survey_suitable_l203_20357


namespace NUMINAMATH_CALUDE_dara_employment_age_l203_20344

/-- Proves that Dara will reach the minimum employment age in 14 years -/
theorem dara_employment_age (min_age : ℕ) (jane_age : ℕ) (years_until_half : ℕ) : 
  min_age = 25 →
  jane_age = 28 →
  years_until_half = 6 →
  min_age - (jane_age + years_until_half) / 2 + years_until_half = 14 :=
by sorry

end NUMINAMATH_CALUDE_dara_employment_age_l203_20344


namespace NUMINAMATH_CALUDE_point_on_line_with_given_x_l203_20321

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a straight line in the xy-plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

theorem point_on_line_with_given_x (l : Line) (x : ℝ) :
  l.slope = 2 →
  l.yIntercept = 2 →
  x = 269 →
  ∃ p : Point, p.x = x ∧ pointOnLine p l ∧ p.y = 540 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_with_given_x_l203_20321


namespace NUMINAMATH_CALUDE_number_puzzle_l203_20317

theorem number_puzzle : ∃! x : ℝ, x / 5 + 7 = x / 4 - 7 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l203_20317


namespace NUMINAMATH_CALUDE_sara_pumpkins_l203_20373

/-- The number of pumpkins eaten by rabbits -/
def pumpkins_eaten (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem sara_pumpkins : pumpkins_eaten 43 20 = 23 := by
  sorry

end NUMINAMATH_CALUDE_sara_pumpkins_l203_20373


namespace NUMINAMATH_CALUDE_min_c_value_l203_20323

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_order : a < b ∧ b < c) (h_sum : a + b + c = 2010)
  (h_unique : ∃! (x y : ℝ), 3 * x + y = 3005 ∧ y = |x - a| + |x - 2*b| + |x - c|) :
  c ≥ 1014 :=
sorry

end NUMINAMATH_CALUDE_min_c_value_l203_20323


namespace NUMINAMATH_CALUDE_min_participants_l203_20322

/-- Represents a participant in the race -/
structure Participant where
  name : String
  position : Nat

/-- Represents the race with its participants -/
structure Race where
  participants : List Participant

/-- Checks if the race satisfies the given conditions -/
def satisfiesConditions (race : Race) : Prop :=
  ∃ (andrei dima lenya : Participant),
    andrei ∈ race.participants ∧
    dima ∈ race.participants ∧
    lenya ∈ race.participants ∧
    (∀ p1 p2 : Participant, p1 ∈ race.participants → p2 ∈ race.participants → p1 ≠ p2 → p1.position ≠ p2.position) ∧
    (2 * (andrei.position - 1) = race.participants.length - andrei.position) ∧
    (3 * (dima.position - 1) = race.participants.length - dima.position) ∧
    (4 * (lenya.position - 1) = race.participants.length - lenya.position)

/-- The theorem stating the minimum number of participants -/
theorem min_participants : ∀ race : Race, satisfiesConditions race → race.participants.length ≥ 61 := by
  sorry

end NUMINAMATH_CALUDE_min_participants_l203_20322


namespace NUMINAMATH_CALUDE_function_properties_l203_20391

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (f a 1 + f a (-1) = 5/2 → f a 2 + f a (-2) = 17/4) ∧
  (∃ (max min : ℝ), (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≤ max ∧ f a x ≥ min) ∧
    max - min = 8/3 → a = 3 ∨ a = 1/3) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l203_20391


namespace NUMINAMATH_CALUDE_root_sum_fraction_values_l203_20332

theorem root_sum_fraction_values (α β γ : ℝ) : 
  (α^3 - α^2 - 2*α + 1 = 0) →
  (β^3 - β^2 - 2*β + 1 = 0) →
  (γ^3 - γ^2 - 2*γ + 1 = 0) →
  (α ≠ 0 ∧ β ≠ 0 ∧ γ ≠ 0) →
  (α/β + β/γ + γ/α = 3 ∨ α/β + β/γ + γ/α = -4) := by
sorry

end NUMINAMATH_CALUDE_root_sum_fraction_values_l203_20332


namespace NUMINAMATH_CALUDE_min_value_xy_expression_min_value_achievable_l203_20339

theorem min_value_xy_expression (x y : ℝ) : (x * y + 1)^2 + (x - y)^2 ≥ 1 := by sorry

theorem min_value_achievable : ∃ x y : ℝ, (x * y + 1)^2 + (x - y)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_min_value_xy_expression_min_value_achievable_l203_20339


namespace NUMINAMATH_CALUDE_privateer_overtakes_at_1730_l203_20368

/-- Represents the chase between a privateer and a merchantman -/
structure ChaseScenario where
  initial_distance : ℝ
  initial_time : ℕ  -- represented in minutes since midnight
  initial_privateer_speed : ℝ
  initial_merchantman_speed : ℝ
  initial_chase_duration : ℝ
  new_speed_ratio : ℚ

/-- Calculates the time when the privateer overtakes the merchantman -/
def overtake_time (scenario : ChaseScenario) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem privateer_overtakes_at_1730 : 
  let scenario : ChaseScenario := {
    initial_distance := 10,
    initial_time := 11 * 60 + 45,  -- 11:45 a.m. in minutes
    initial_privateer_speed := 11,
    initial_merchantman_speed := 8,
    initial_chase_duration := 2,
    new_speed_ratio := 17 / 15
  }
  overtake_time scenario = 17 * 60 + 30  -- 5:30 p.m. in minutes
:= by sorry


end NUMINAMATH_CALUDE_privateer_overtakes_at_1730_l203_20368


namespace NUMINAMATH_CALUDE_early_finish_hours_l203_20353

/-- Represents the number of workers -/
def num_workers : ℕ := 3

/-- Represents the normal working hours per day -/
def normal_hours : ℕ := 8

/-- Represents the number of customers served per hour by each worker -/
def customers_per_hour : ℕ := 7

/-- Represents the total number of customers served that day -/
def total_customers : ℕ := 154

/-- Theorem stating that the worker who finished early worked for 6 hours -/
theorem early_finish_hours :
  ∃ (h : ℕ),
    h < normal_hours ∧
    (2 * normal_hours * customers_per_hour + h * customers_per_hour = total_customers) ∧
    h = 6 :=
sorry

end NUMINAMATH_CALUDE_early_finish_hours_l203_20353


namespace NUMINAMATH_CALUDE_train_speed_equation_l203_20378

/-- Represents the equation for two trains traveling the same distance at different speeds -/
theorem train_speed_equation (distance : ℝ) (speed_difference : ℝ) (time_difference : ℝ) 
  (h1 : distance = 236)
  (h2 : speed_difference = 40)
  (h3 : time_difference = 1/4) :
  ∀ x : ℝ, x > speed_difference → 
    (distance / (x - speed_difference) - distance / x = time_difference) :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_equation_l203_20378


namespace NUMINAMATH_CALUDE_sum_inequality_l203_20331

theorem sum_inequality (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 6)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 12) :
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l203_20331


namespace NUMINAMATH_CALUDE_polynomial_factorization_l203_20386

theorem polynomial_factorization (x : ℝ) : x - x^3 = x * (1 - x) * (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l203_20386


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l203_20370

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 14 % 31 ∧
  ∀ (y : ℕ), y > 0 ∧ (5 * y) % 31 = 14 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l203_20370


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l203_20351

def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {3, 7, 9}
def B : Set ℕ := {1, 9}

theorem intersection_complement_equality : A ∩ (U \ B) = {3, 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l203_20351


namespace NUMINAMATH_CALUDE_order_of_abc_l203_20302

theorem order_of_abc (a b c : ℝ) 
  (h1 : 1.001 * Real.exp a = Real.exp 1.001)
  (h2 : b - Real.sqrt (1000 / 1001) = 1.001 - Real.sqrt 1.001)
  (h3 : c = 1.001) : 
  b < a ∧ a < c :=
sorry

end NUMINAMATH_CALUDE_order_of_abc_l203_20302


namespace NUMINAMATH_CALUDE_cost_for_23_days_l203_20325

/-- Calculates the total cost of staying in a student youth hostel for a given number of days. -/
def hostelCost (days : ℕ) : ℚ :=
  let firstWeekRate : ℚ := 18
  let additionalWeekRate : ℚ := 14
  let firstWeekDays : ℕ := min days 7
  let additionalDays : ℕ := days - firstWeekDays
  firstWeekRate * firstWeekDays + additionalWeekRate * additionalDays

/-- Theorem stating that the cost for a 23-day stay is $350.00 -/
theorem cost_for_23_days :
  hostelCost 23 = 350 := by
  sorry

#eval hostelCost 23

end NUMINAMATH_CALUDE_cost_for_23_days_l203_20325


namespace NUMINAMATH_CALUDE_distinct_sums_l203_20394

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  hd : d ≠ 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * seq.a 1 + (n * (n - 1) : ℚ) / 2 * seq.d

theorem distinct_sums (seq : ArithmeticSequence) 
  (h_sum_5 : S seq 5 = 0) : 
  Finset.card (Finset.image (S seq) (Finset.range 100)) = 98 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sums_l203_20394


namespace NUMINAMATH_CALUDE_arithmetic_sum_property_l203_20376

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 5 + a 7 = 2) →
  (a 4 + a 6 + a 8 = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_property_l203_20376


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_complementary_l203_20342

-- Define the set of cards
inductive Card : Type
| Hearts : Card
| Spades : Card
| Diamonds : Card
| Clubs : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "A gets clubs"
def A_gets_clubs (d : Distribution) : Prop := d Person.A = Card.Clubs

-- Define the event "B gets clubs"
def B_gets_clubs (d : Distribution) : Prop := d Person.B = Card.Clubs

-- Statement to prove
theorem events_mutually_exclusive_but_not_complementary :
  ∃ (d : Distribution),
    (∀ (p : Person), ∃! (c : Card), d p = c) →
    (¬(A_gets_clubs d ∧ B_gets_clubs d)) ∧
    (∃ (d' : Distribution), ¬(A_gets_clubs d') ∧ ¬(B_gets_clubs d')) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_complementary_l203_20342


namespace NUMINAMATH_CALUDE_front_view_length_l203_20395

/-- Given a line segment with length 5√2, side view 5, and top view √34, 
    its front view has length √41. -/
theorem front_view_length 
  (segment_length : ℝ) 
  (side_view : ℝ) 
  (top_view : ℝ) 
  (h1 : segment_length = 5 * Real.sqrt 2)
  (h2 : side_view = 5)
  (h3 : top_view = Real.sqrt 34) : 
  Real.sqrt (side_view^2 + top_view^2 + (Real.sqrt 41)^2) = segment_length :=
by sorry

end NUMINAMATH_CALUDE_front_view_length_l203_20395


namespace NUMINAMATH_CALUDE_parabola_points_order_l203_20375

/-- Given a parabola y = 2(x-2)^2 + 1 and three points on it, 
    prove that the y-coordinates are in a specific order -/
theorem parabola_points_order (y₁ y₂ y₃ : ℝ) : 
  (y₁ = 2*(-3-2)^2 + 1) →  -- Point A(-3, y₁)
  (y₂ = 2*(3-2)^2 + 1) →   -- Point B(3, y₂)
  (y₃ = 2*(4-2)^2 + 1) →   -- Point C(4, y₃)
  y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

end NUMINAMATH_CALUDE_parabola_points_order_l203_20375


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l203_20327

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the original point A
def A : Point := (-2, 3)

-- Define the symmetry operation about the x-axis
def symmetry_x_axis (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem statement
theorem symmetric_point_coordinates :
  symmetry_x_axis A = (-2, -3) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l203_20327


namespace NUMINAMATH_CALUDE_uv_value_l203_20349

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = -2/3 * x + 6

-- Define points P and Q
def P : ℝ × ℝ := (9, 0)
def Q : ℝ × ℝ := (0, 6)

-- Define point R
def R (u v : ℝ) : ℝ × ℝ := (u, v)

-- Define that R is on the line segment PQ
def R_on_PQ (u v : ℝ) : Prop :=
  line_equation u v ∧ 0 ≤ u ∧ u ≤ 9

-- Define the area ratio condition
def area_condition (u v : ℝ) : Prop :=
  (1/2 * 9 * 6) = 2 * (1/2 * 9 * v)

-- Theorem statement
theorem uv_value (u v : ℝ) 
  (h1 : R_on_PQ u v) 
  (h2 : area_condition u v) : 
  u * v = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_uv_value_l203_20349


namespace NUMINAMATH_CALUDE_poojas_speed_l203_20364

/-- 
Given:
- Roja moves in the opposite direction from Pooja at 5 km/hr
- After 4 hours, the distance between Roja and Pooja is 32 km

Prove that Pooja's speed is 3 km/hr
-/
theorem poojas_speed (roja_speed : ℝ) (time : ℝ) (distance : ℝ) :
  roja_speed = 5 →
  time = 4 →
  distance = 32 →
  ∃ (pooja_speed : ℝ), pooja_speed = 3 ∧ distance = (roja_speed + pooja_speed) * time :=
by sorry

end NUMINAMATH_CALUDE_poojas_speed_l203_20364


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_factor_difference_of_cubes_l203_20307

-- Theorem 1
theorem perfect_square_trinomial (m : ℝ) : m^2 - 10*m + 25 = (m - 5)^2 := by
  sorry

-- Theorem 2
theorem factor_difference_of_cubes (a b : ℝ) : a^3*b - a*b = a*b*(a + 1)*(a - 1) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_factor_difference_of_cubes_l203_20307


namespace NUMINAMATH_CALUDE_unique_valid_code_l203_20304

def is_valid_code (n : ℕ) : Prop :=
  -- The code is an eight-digit number
  100000000 > n ∧ n ≥ 10000000 ∧
  -- The code is a multiple of both 3 and 25
  n % 3 = 0 ∧ n % 25 = 0 ∧
  -- The code is between 20,000,000 and 30,000,000
  30000000 > n ∧ n > 20000000 ∧
  -- The digits in the millions and hundred thousand places are the same
  (n / 1000000) % 10 = (n / 100000) % 10 ∧
  -- The digit in the hundreds place is 2 less than the digit in the ten thousands place
  (n / 100) % 10 + 2 = (n / 10000) % 10 ∧
  -- The three-digit number formed by the digits in the hundred thousands, ten thousands, and thousands places,
  -- divided by the two-digit number formed by the digits in the ten millions and millions places, gives a quotient of 25
  ((n / 100000) % 1000) / ((n / 1000000) % 100) = 25

theorem unique_valid_code : ∃! n : ℕ, is_valid_code n ∧ n = 26650350 :=
  sorry

#check unique_valid_code

end NUMINAMATH_CALUDE_unique_valid_code_l203_20304


namespace NUMINAMATH_CALUDE_fraction_value_l203_20335

theorem fraction_value (p q : ℚ) (h : p / q = 7) : (p + q) / (p - q) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l203_20335


namespace NUMINAMATH_CALUDE_divisor_product_256_l203_20341

def divisor_product (n : ℕ+) : ℕ :=
  (List.range n.val).filter (λ i => i > 0 ∧ n.val % i = 0)
    |>.map (λ i => i + 1)
    |>.prod

theorem divisor_product_256 (n : ℕ+) :
  divisor_product n = 256 → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_divisor_product_256_l203_20341


namespace NUMINAMATH_CALUDE_square_not_always_positive_l203_20346

theorem square_not_always_positive : ¬ (∀ x : ℝ, x^2 > 0) := by sorry

end NUMINAMATH_CALUDE_square_not_always_positive_l203_20346


namespace NUMINAMATH_CALUDE_complex_equation_solution_l203_20316

theorem complex_equation_solution (z : ℂ) : (Complex.I / (z + Complex.I) = 2 - Complex.I) → z = -1/5 - 3/5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l203_20316


namespace NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l203_20390

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l203_20390


namespace NUMINAMATH_CALUDE_profit_maximizing_price_l203_20397

/-- The profit function based on price increase -/
def profit (x : ℝ) : ℝ := (90 + x - 80) * (400 - 10 * x)

/-- The initial purchase price -/
def initial_purchase_price : ℝ := 80

/-- The initial selling price -/
def initial_selling_price : ℝ := 90

/-- The initial sales volume -/
def initial_sales_volume : ℝ := 400

/-- The rate of decrease in sales volume per unit price increase -/
def sales_decrease_rate : ℝ := 10

/-- Theorem stating that the profit-maximizing selling price is 105 yuan -/
theorem profit_maximizing_price :
  ∃ (x : ℝ), x = 15 ∧ 
  ∀ (y : ℝ), profit y ≤ profit x ∧
  initial_selling_price + x = 105 := by
  sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_l203_20397


namespace NUMINAMATH_CALUDE_integer_set_range_l203_20396

theorem integer_set_range (a : ℝ) : 
  a ≤ 1 →
  (∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (↑x : ℝ) ∈ Set.Icc a (2 - a) ∧
    (↑y : ℝ) ∈ Set.Icc a (2 - a) ∧
    (↑z : ℝ) ∈ Set.Icc a (2 - a) ∧
    (∀ (w : ℤ), (↑w : ℝ) ∈ Set.Icc a (2 - a) → w = x ∨ w = y ∨ w = z)) →
  -1 < a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_integer_set_range_l203_20396


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l203_20372

theorem diophantine_equation_solutions :
  ∀ a b c d : ℕ, 2^a * 3^b - 5^c * 7^d = 1 ↔
    (a = 1 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
    (a = 3 ∧ b = 0 ∧ c = 0 ∧ d = 1) ∨
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 0) ∨
    (a = 2 ∧ b = 2 ∧ c = 1 ∧ d = 1) :=
by sorry


end NUMINAMATH_CALUDE_diophantine_equation_solutions_l203_20372


namespace NUMINAMATH_CALUDE_inequality_condition_l203_20367

theorem inequality_condition (p : ℝ) :
  (∀ x₁ x₂ x₃ : ℝ, x₁^2 + x₂^2 + x₃^2 ≥ p * (x₁ * x₂ + x₂ * x₃)) ↔ p ≤ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l203_20367


namespace NUMINAMATH_CALUDE_men_complete_nine_units_l203_20381

/-- The number of men in the committee -/
def num_men : ℕ := 250

/-- The number of women in the committee -/
def num_women : ℕ := 150

/-- The number of units completed per day when all men and women work -/
def total_units : ℕ := 12

/-- The number of units completed per day when only women work -/
def women_units : ℕ := 3

/-- The number of units completed per day by men -/
def men_units : ℕ := total_units - women_units

theorem men_complete_nine_units : men_units = 9 := by
  sorry

end NUMINAMATH_CALUDE_men_complete_nine_units_l203_20381


namespace NUMINAMATH_CALUDE_average_position_l203_20352

def fractions : List ℚ := [1/2, 1/3, 1/4, 1/5, 1/6, 1/7]

theorem average_position :
  let avg := (List.sum fractions) / fractions.length
  avg = 223 / 840 ∧ 1/4 < avg ∧ avg < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_average_position_l203_20352


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_eight_l203_20385

theorem sqrt_sum_equals_eight : 
  Real.sqrt (24 - 8 * Real.sqrt 3) + Real.sqrt (24 + 8 * Real.sqrt 3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_eight_l203_20385


namespace NUMINAMATH_CALUDE_equation_solution_l203_20319

theorem equation_solution :
  ∃ x : ℝ, (4 : ℝ)^x * (4 : ℝ)^x * (16 : ℝ)^(x + 1) = (1024 : ℝ)^2 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l203_20319


namespace NUMINAMATH_CALUDE_product_xyz_l203_20314

theorem product_xyz (x y z : ℝ) 
  (h1 : x + 2/y = 2) 
  (h2 : y + 2/z = 2) 
  (h3 : y ≠ 0) 
  (h4 : z ≠ 0) : x * y * z = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_l203_20314


namespace NUMINAMATH_CALUDE_f_positive_iff_l203_20347

-- Define the function
def f (x : ℝ) := x^2 + x - 12

-- State the theorem
theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x < -4 ∨ x > 3 := by sorry

end NUMINAMATH_CALUDE_f_positive_iff_l203_20347


namespace NUMINAMATH_CALUDE_find_M_l203_20337

theorem find_M : ∃ M : ℚ, (10 + 11 + 12) / 3 = (2022 + 2023 + 2024) / M ∧ M = 551 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l203_20337


namespace NUMINAMATH_CALUDE_line_intersects_y_axis_l203_20306

/-- A line passing through two points (1, 7) and (3, 11) -/
def line (x : ℝ) : ℝ := 2 * x + 5

/-- The y-axis is defined as the set of points with x-coordinate equal to 0 -/
def y_axis (x : ℝ) : Prop := x = 0

theorem line_intersects_y_axis :
  ∃ y : ℝ, y_axis 0 ∧ line 0 = y ∧ y = 5 := by sorry

end NUMINAMATH_CALUDE_line_intersects_y_axis_l203_20306


namespace NUMINAMATH_CALUDE_arrangement_count_l203_20365

/-- The number of ways to divide teachers and students into groups -/
def divide_groups (num_teachers num_students num_groups : ℕ) : ℕ :=
  if num_teachers = 2 ∧ num_students = 4 ∧ num_groups = 2 then
    12
  else
    0

/-- Theorem stating that the number of arrangements is 12 -/
theorem arrangement_count :
  divide_groups 2 4 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l203_20365


namespace NUMINAMATH_CALUDE_abc_solution_l203_20361

/-- Converts a base 7 number to its decimal representation -/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to its base 7 representation -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Represents a two-digit number in base 7 -/
def twoDigitBase7 (tens : ℕ) (ones : ℕ) : ℕ := 7 * tens + ones

/-- Represents a three-digit number in base 7 -/
def threeDigitBase7 (hundreds : ℕ) (tens : ℕ) (ones : ℕ) : ℕ := 49 * hundreds + 7 * tens + ones

theorem abc_solution (A B C : ℕ) : 
  (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) →  -- non-zero digits
  (A < 7 ∧ B < 7 ∧ C < 7) →  -- less than 7
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) →  -- distinct digits
  (twoDigitBase7 A B + C = twoDigitBase7 C 0) →  -- AB₇ + C₇ = C0₇
  (twoDigitBase7 A B + twoDigitBase7 B A = twoDigitBase7 C C) →  -- AB₇ + BA₇ = CC₇
  threeDigitBase7 A B C = 643  -- ABC = 643 in base 7
  := by sorry


end NUMINAMATH_CALUDE_abc_solution_l203_20361


namespace NUMINAMATH_CALUDE_weight_loss_calculation_l203_20309

theorem weight_loss_calculation (W : ℝ) (x : ℝ) : 
  W * (1 - x / 100 + 2 / 100) = W * (100 - 10.24) / 100 → x = 12.24 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_calculation_l203_20309


namespace NUMINAMATH_CALUDE_somu_age_problem_l203_20369

/-- Represents the problem of finding when Somu was one-fifth of his father's age --/
theorem somu_age_problem (somu_age : ℕ) (father_age : ℕ) (years_ago : ℕ) : 
  somu_age = 20 →
  3 * somu_age = father_age →
  5 * (somu_age - years_ago) = father_age - years_ago →
  years_ago = 10 := by
  sorry

#check somu_age_problem

end NUMINAMATH_CALUDE_somu_age_problem_l203_20369


namespace NUMINAMATH_CALUDE_magnitude_of_AB_l203_20377

def vector_AB : ℝ × ℝ := (3, -4)

theorem magnitude_of_AB : Real.sqrt ((vector_AB.1)^2 + (vector_AB.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_AB_l203_20377


namespace NUMINAMATH_CALUDE_basketball_weight_proof_l203_20359

/-- The weight of one basketball in pounds -/
def basketball_weight : ℝ := 16

/-- The weight of one kayak in pounds -/
def kayak_weight : ℝ := 24

theorem basketball_weight_proof : 
  (6 * basketball_weight = 4 * kayak_weight) ∧ 
  (3 * kayak_weight = 72) → 
  basketball_weight = 16 := by
  sorry

end NUMINAMATH_CALUDE_basketball_weight_proof_l203_20359


namespace NUMINAMATH_CALUDE_jerusha_earned_68_l203_20362

/-- Jerusha's earnings given Lottie's earnings and their total earnings -/
def jerushas_earnings (lotties_earnings : ℚ) (total_earnings : ℚ) : ℚ :=
  4 * lotties_earnings

theorem jerusha_earned_68 :
  ∃ (lotties_earnings : ℚ),
    jerushas_earnings lotties_earnings 85 = 68 ∧
    lotties_earnings + jerushas_earnings lotties_earnings 85 = 85 := by
  sorry

end NUMINAMATH_CALUDE_jerusha_earned_68_l203_20362


namespace NUMINAMATH_CALUDE_different_suit_probability_l203_20355

theorem different_suit_probability (total_cards : ℕ) (num_suits : ℕ) 
  (h1 : total_cards = 65) 
  (h2 : num_suits = 5) 
  (h3 : total_cards % num_suits = 0) :
  let cards_per_suit := total_cards / num_suits
  let remaining_cards := total_cards - 1
  let different_suit_cards := total_cards - cards_per_suit
  (different_suit_cards : ℚ) / remaining_cards = 13 / 16 := by
sorry


end NUMINAMATH_CALUDE_different_suit_probability_l203_20355


namespace NUMINAMATH_CALUDE_dust_storm_coverage_l203_20303

/-- Given a prairie and a dust storm, calculate the area covered by the storm -/
theorem dust_storm_coverage (total_prairie_area untouched_area : ℕ) 
  (h1 : total_prairie_area = 65057)
  (h2 : untouched_area = 522) :
  total_prairie_area - untouched_area = 64535 := by
  sorry

end NUMINAMATH_CALUDE_dust_storm_coverage_l203_20303


namespace NUMINAMATH_CALUDE_roots_of_equation_l203_20387

theorem roots_of_equation (x y : ℝ) (h1 : x + y = 10) (h2 : |x - y| = 14) :
  x^2 - 10*x - 24 = 0 ∧ y^2 - 10*y - 24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l203_20387


namespace NUMINAMATH_CALUDE_sum_difference_l203_20334

def mena_sequence : List Nat := List.range 30

def emily_sequence : List Nat :=
  mena_sequence.map (fun n => 
    let tens := n / 10
    let ones := n % 10
    if tens = 2 then 10 + ones
    else if ones = 2 then tens * 10 + 1
    else n)

theorem sum_difference : 
  mena_sequence.sum - emily_sequence.sum = 103 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_l203_20334


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l203_20356

theorem absolute_value_equation_solution :
  ∃! y : ℝ, abs (y - 4) + 3 * y = 15 :=
by
  -- The unique solution is y = 4.75
  use 4.75
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l203_20356


namespace NUMINAMATH_CALUDE_variance_of_specific_set_l203_20383

theorem variance_of_specific_set (a : ℝ) : 
  (5 + 8 + a + 7 + 4) / 5 = a → 
  ((5 - a)^2 + (8 - a)^2 + (a - a)^2 + (7 - a)^2 + (4 - a)^2) / 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_specific_set_l203_20383


namespace NUMINAMATH_CALUDE_yellow_jelly_bean_probability_l203_20312

/-- The probability of selecting a yellow jelly bean from a jar containing red, orange, yellow, and green jelly beans -/
theorem yellow_jelly_bean_probability
  (p_red : ℝ)
  (p_orange : ℝ)
  (p_green : ℝ)
  (h_red : p_red = 0.1)
  (h_orange : p_orange = 0.4)
  (h_green : p_green = 0.25)
  (h_sum : p_red + p_orange + p_green + (1 - p_red - p_orange - p_green) = 1) :
  1 - p_red - p_orange - p_green = 0.25 := by
sorry

end NUMINAMATH_CALUDE_yellow_jelly_bean_probability_l203_20312


namespace NUMINAMATH_CALUDE_influenza_spread_l203_20336

theorem influenza_spread (x : ℝ) : (1 + x)^2 = 100 → x = 9 := by sorry

end NUMINAMATH_CALUDE_influenza_spread_l203_20336


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_minus_i_l203_20379

theorem imaginary_part_of_i_times_one_minus_i (i : ℂ) : 
  (Complex.I * (1 - Complex.I)).im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_minus_i_l203_20379


namespace NUMINAMATH_CALUDE_min_plus_arg_is_pi_third_l203_20380

noncomputable def f (x : ℝ) : ℝ := 9 / (8 * Real.cos (2 * x) + 16) - Real.sin x ^ 2

def has_min (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, f x ≥ m

def is_smallest_positive_min (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  has_min f m ∧ f n = m ∧ n > 0 ∧ ∀ x, 0 < x ∧ x < n → f x > m

theorem min_plus_arg_is_pi_third :
  ∃ (m n : ℝ), is_smallest_positive_min f m n ∧ m + n = Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_min_plus_arg_is_pi_third_l203_20380


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l203_20328

theorem inequality_system_solution_set :
  {x : ℝ | 3 * x + 9 > 0 ∧ 2 * x < 6} = {x : ℝ | -3 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l203_20328


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l203_20310

theorem isosceles_triangle_base_length 
  (perimeter : ℝ) 
  (one_side : ℝ) 
  (h_perimeter : perimeter = 29) 
  (h_one_side : one_side = 7) 
  (h_isosceles : ∃ (base equal_side : ℝ), 
    base > 0 ∧ equal_side > 0 ∧ 
    base + 2 * equal_side = perimeter ∧ 
    (base = one_side ∨ equal_side = one_side)) :
  ∃ (base : ℝ), base = 7 ∧ 
    ∃ (equal_side : ℝ), 
      base > 0 ∧ equal_side > 0 ∧
      base + 2 * equal_side = perimeter ∧
      (base = one_side ∨ equal_side = one_side) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l203_20310


namespace NUMINAMATH_CALUDE_num_factors_of_2000_l203_20374

/-- The number of positive factors of a natural number n -/
def num_factors (n : ℕ) : ℕ := sorry

/-- 2000 expressed as a product of prime factors -/
def two_thousand_factorization : ℕ := 2^4 * 5^3

/-- Theorem stating that the number of positive factors of 2000 is 20 -/
theorem num_factors_of_2000 : num_factors two_thousand_factorization = 20 := by sorry

end NUMINAMATH_CALUDE_num_factors_of_2000_l203_20374


namespace NUMINAMATH_CALUDE_equation_roots_l203_20388

theorem equation_roots (k : ℝ) : 
  (∃ x y : ℂ, x ≠ y ∧ 
    (x / (x + 1) + 2 * x / (x + 3) = k * x) ∧ 
    (y / (y + 1) + 2 * y / (y + 3) = k * y) ∧
    (∀ z : ℂ, z / (z + 1) + 2 * z / (z + 3) = k * z → z = x ∨ z = y)) ↔ 
  k = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l203_20388


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l203_20382

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x + 3) → x ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l203_20382


namespace NUMINAMATH_CALUDE_chess_tournament_success_ratio_l203_20324

theorem chess_tournament_success_ratio (charlie_day1_score charlie_day1_attempted charlie_day2_score charlie_day2_attempted : ℕ) : 
  -- Total points for both players
  charlie_day1_attempted + charlie_day2_attempted = 600 →
  -- Charlie's scores are positive integers
  charlie_day1_score > 0 →
  charlie_day2_score > 0 →
  -- Charlie's daily success ratios are less than Alpha's
  charlie_day1_score * 360 < 180 * charlie_day1_attempted →
  charlie_day2_score * 240 < 120 * charlie_day2_attempted →
  -- Charlie did not attempt 360 points on day 1
  charlie_day1_attempted ≠ 360 →
  -- The maximum two-day success ratio for Charlie
  (charlie_day1_score + charlie_day2_score : ℚ) / 600 ≤ 299 / 600 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_success_ratio_l203_20324


namespace NUMINAMATH_CALUDE_dot_product_AB_normal_is_zero_l203_20363

def A : ℝ × ℝ := (3, -1)
def B : ℝ × ℝ := (6, 1)
def l (x y : ℝ) : Prop := 2 * x - 3 * y - 9 = 0

def normal_vector (l : (ℝ → ℝ → Prop)) : ℝ × ℝ := (2, -3)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

theorem dot_product_AB_normal_is_zero :
  (vector_AB.1 * (normal_vector l).1 + vector_AB.2 * (normal_vector l).2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_AB_normal_is_zero_l203_20363


namespace NUMINAMATH_CALUDE_sum_equation_l203_20348

theorem sum_equation (x y z : ℝ) (h1 : x + y = 4) (h2 : x * y = z^2 + 4) : 
  x + 2*y + 3*z = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_equation_l203_20348


namespace NUMINAMATH_CALUDE_james_hourly_rate_l203_20392

/-- Represents the car rental scenario for James --/
structure CarRental where
  hours_per_day : ℕ
  days_per_week : ℕ
  weekly_income : ℕ

/-- Calculates the hourly rate for car rental --/
def hourly_rate (rental : CarRental) : ℚ :=
  rental.weekly_income / (rental.hours_per_day * rental.days_per_week)

/-- Theorem stating that James' hourly rate is $20 --/
theorem james_hourly_rate :
  let james_rental : CarRental := {
    hours_per_day := 8,
    days_per_week := 4,
    weekly_income := 640
  }
  hourly_rate james_rental = 20 := by sorry

end NUMINAMATH_CALUDE_james_hourly_rate_l203_20392


namespace NUMINAMATH_CALUDE_find_w_l203_20305

theorem find_w : ∃ w : ℝ, ((2^5 : ℝ) * (9^2)) / ((8^2) * w) = 0.16666666666666666 ∧ w = 243 := by
  sorry

end NUMINAMATH_CALUDE_find_w_l203_20305


namespace NUMINAMATH_CALUDE_average_headcount_l203_20345

def fall_02_03 : ℕ := 11700
def fall_03_04 : ℕ := 11500
def fall_04_05 : ℕ := 11600

theorem average_headcount : 
  (fall_02_03 + fall_03_04 + fall_04_05) / 3 = 11600 := by
  sorry

end NUMINAMATH_CALUDE_average_headcount_l203_20345


namespace NUMINAMATH_CALUDE_jake_has_fewer_ball_difference_l203_20350

/-- The number of balls Audrey has -/
def audrey_balls : ℕ := 41

/-- The number of balls Jake has -/
def jake_balls : ℕ := 7

/-- Jake has fewer balls than Audrey -/
theorem jake_has_fewer : jake_balls < audrey_balls := by sorry

/-- The difference in the number of balls between Audrey and Jake is 34 -/
theorem ball_difference : audrey_balls - jake_balls = 34 := by sorry

end NUMINAMATH_CALUDE_jake_has_fewer_ball_difference_l203_20350


namespace NUMINAMATH_CALUDE_floor_fraction_equals_eight_l203_20354

theorem floor_fraction_equals_eight (n : ℕ) (h : n = 2006) : 
  ⌊(8 * (n^2 + 1 : ℝ)) / (n^2 - 1 : ℝ)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_fraction_equals_eight_l203_20354


namespace NUMINAMATH_CALUDE_joey_age_when_beth_was_joeys_current_age_l203_20389

/-- Represents a person's age at different points in time -/
structure Person where
  current_age : ℕ
  future_age : ℕ
  past_age : ℕ

/-- Given the conditions of the problem, prove that Joey was 4 years old when Beth was Joey's current age -/
theorem joey_age_when_beth_was_joeys_current_age 
  (joey : Person) 
  (beth : Person)
  (h1 : joey.current_age = 9)
  (h2 : joey.future_age = beth.current_age)
  (h3 : joey.future_age = joey.current_age + 5) :
  joey.past_age = 4 := by
sorry

end NUMINAMATH_CALUDE_joey_age_when_beth_was_joeys_current_age_l203_20389


namespace NUMINAMATH_CALUDE_l_shaped_area_l203_20301

/-- The area of the L-shaped region in a square arrangement --/
theorem l_shaped_area (large_square_side : ℝ) (small_square1 : ℝ) (small_square2 : ℝ) (small_square3 : ℝ) (small_square4 : ℝ)
  (h1 : large_square_side = 7)
  (h2 : small_square1 = 2)
  (h3 : small_square2 = 3)
  (h4 : small_square3 = 2)
  (h5 : small_square4 = 1) :
  large_square_side ^ 2 - (small_square1 ^ 2 + small_square2 ^ 2 + small_square3 ^ 2 + small_square4 ^ 2) = 31 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_area_l203_20301


namespace NUMINAMATH_CALUDE_tangent_line_and_max_value_l203_20320

-- Define the function f(x) = x³ - ax²
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

theorem tangent_line_and_max_value (a : ℝ) :
  f' a 1 = 3 →
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → f a x + x ≥ 0) →
  (∃ (m b : ℝ), m = 3 ∧ b = -2 ∧
    ∀ x : ℝ, f a x = m * (x - 1) + f a 1) ∧
  (∃ M : ℝ, M = 8 ∧
    ∀ x : ℝ, x ∈ Set.Icc 0 2 → f a x ≤ M) ∧
  a ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_tangent_line_and_max_value_l203_20320


namespace NUMINAMATH_CALUDE_angle_from_coordinates_l203_20315

theorem angle_from_coordinates (a : Real) (h1 : 0 < a) (h2 : a < π / 2) :
  ∃ (P : ℝ × ℝ), P.1 = 4 * Real.sin 3 ∧ P.2 = -4 * Real.cos 3 →
  a = 3 - π / 2 :=
sorry

end NUMINAMATH_CALUDE_angle_from_coordinates_l203_20315
