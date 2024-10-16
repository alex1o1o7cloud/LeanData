import Mathlib

namespace NUMINAMATH_CALUDE_jinsu_kicks_to_exceed_hoseok_l2405_240526

theorem jinsu_kicks_to_exceed_hoseok (hoseok_kicks : ℕ) (jinsu_first : ℕ) (jinsu_second : ℕ) :
  hoseok_kicks = 48 →
  jinsu_first = 15 →
  jinsu_second = 15 →
  ∃ (jinsu_third : ℕ), 
    jinsu_third = 19 ∧ 
    jinsu_first + jinsu_second + jinsu_third > hoseok_kicks ∧
    ∀ (x : ℕ), x < 19 → jinsu_first + jinsu_second + x ≤ hoseok_kicks :=
by sorry

end NUMINAMATH_CALUDE_jinsu_kicks_to_exceed_hoseok_l2405_240526


namespace NUMINAMATH_CALUDE_expectation_of_linear_combination_l2405_240533

variable (ξ η : ℝ → ℝ)
variable (E : (ℝ → ℝ) → ℝ)

axiom linearity_of_expectation : ∀ (a b : ℝ) (X Y : ℝ → ℝ), E (λ ω => a * X ω + b * Y ω) = a * E X + b * E Y

theorem expectation_of_linear_combination
  (h1 : E ξ = 10)
  (h2 : E η = 3) :
  E (λ ω => 3 * ξ ω + 5 * η ω) = 45 := by
sorry

end NUMINAMATH_CALUDE_expectation_of_linear_combination_l2405_240533


namespace NUMINAMATH_CALUDE_three_repeated_digit_sum_theorem_l2405_240501

theorem three_repeated_digit_sum_theorem : ∃ (a b c : ℕ), 
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  let sum := 11111 * a + 1111 * b + 111 * c
  10000 ≤ sum ∧ sum < 100000 ∧
  (∀ (d₁ d₂ : ℕ), 
    d₁ < 5 ∧ d₂ < 5 ∧ d₁ ≠ d₂ → 
    (sum / (10^d₁) % 10) ≠ (sum / (10^d₂) % 10)) :=
by sorry

end NUMINAMATH_CALUDE_three_repeated_digit_sum_theorem_l2405_240501


namespace NUMINAMATH_CALUDE_work_completion_time_l2405_240552

/-- Given that A can do a work in 6 days and B can do the same work in 12 days,
    prove that A and B working together can finish the work in 4 days. -/
theorem work_completion_time (work : ℝ) (days_A : ℝ) (days_B : ℝ)
    (h_work : work > 0)
    (h_days_A : days_A = 6)
    (h_days_B : days_B = 12) :
    work / (work / days_A + work / days_B) = 4 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2405_240552


namespace NUMINAMATH_CALUDE_power_fraction_equality_l2405_240524

theorem power_fraction_equality : (1 / ((-8^2)^3)) * (-8)^7 = 8 := by sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l2405_240524


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2405_240525

theorem polynomial_remainder_theorem (x : ℝ) : 
  (4 * x^3 - 8 * x^2 + 11 * x - 5) % (2 * x - 4) = 17 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2405_240525


namespace NUMINAMATH_CALUDE_paint_cost_per_quart_l2405_240591

/-- The cost of paint per quart given specific conditions -/
theorem paint_cost_per_quart : 
  ∀ (coverage_per_quart : ℝ) (cube_edge_length : ℝ) (cost_to_paint_cube : ℝ),
  coverage_per_quart = 1200 →
  cube_edge_length = 10 →
  cost_to_paint_cube = 1.6 →
  (∃ (cost_per_quart : ℝ),
    cost_per_quart = 3.2 ∧
    cost_per_quart * (6 * cube_edge_length^2 / coverage_per_quart) = cost_to_paint_cube) :=
by sorry


end NUMINAMATH_CALUDE_paint_cost_per_quart_l2405_240591


namespace NUMINAMATH_CALUDE_pigeons_on_pole_l2405_240553

theorem pigeons_on_pole (initial_pigeons : ℕ) (pigeons_flew_away : ℕ) (pigeons_left : ℕ) : 
  initial_pigeons = 8 → pigeons_flew_away = 3 → pigeons_left = initial_pigeons - pigeons_flew_away → pigeons_left = 5 := by
  sorry

end NUMINAMATH_CALUDE_pigeons_on_pole_l2405_240553


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2405_240504

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) (R : ℝ) :
  R > 0 →
  a > 0 →
  b > 0 →
  c > 0 →
  2 * R * (Real.sin A ^ 2 - Real.sin C ^ 2) = (Real.sqrt 2 * a - b) * Real.sin B →
  a = 2 * R * Real.sin A →
  b = 2 * R * Real.sin B →
  c = 2 * R * Real.sin C →
  C = Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2405_240504


namespace NUMINAMATH_CALUDE_symmetric_point_xoz_l2405_240573

/-- A point in 3D space represented by its coordinates -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoz plane in 3D space -/
def xoz_plane : Set Point3D :=
  {p : Point3D | p.y = 0}

/-- Two points are symmetric with respect to the xoz plane if their x and z coordinates are equal,
    and their y coordinates are opposite -/
def symmetric_wrt_xoz (p q : Point3D) : Prop :=
  p.x = q.x ∧ p.y = -q.y ∧ p.z = q.z

/-- Given a point P(1, 2, 3), its symmetric point Q with respect to the xoz plane
    has coordinates (1, -2, 3) -/
theorem symmetric_point_xoz :
  let P : Point3D := ⟨1, 2, 3⟩
  ∃ Q : Point3D, symmetric_wrt_xoz P Q ∧ Q = ⟨1, -2, 3⟩ :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_xoz_l2405_240573


namespace NUMINAMATH_CALUDE_hotel_air_conditioned_rooms_l2405_240500

theorem hotel_air_conditioned_rooms 
  (total_rooms : ℝ) 
  (air_conditioned_rooms : ℝ) 
  (h1 : 3 / 4 * total_rooms = total_rooms - (total_rooms - air_conditioned_rooms + (air_conditioned_rooms - 2 / 3 * air_conditioned_rooms)))
  (h2 : 2 / 3 * air_conditioned_rooms = air_conditioned_rooms - (air_conditioned_rooms - 2 / 3 * air_conditioned_rooms))
  (h3 : 4 / 5 * (1 / 4 * total_rooms) = 1 / 3 * air_conditioned_rooms) :
  air_conditioned_rooms / total_rooms = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_hotel_air_conditioned_rooms_l2405_240500


namespace NUMINAMATH_CALUDE_point_motion_l2405_240540

/-- The position function of a point moving in a straight line -/
def s (t : ℝ) : ℝ := 3 * t^2 - 3 * t + 8

/-- The velocity function derived from the position function -/
def v (t : ℝ) : ℝ := 6 * t - 3

/-- The acceleration function derived from the velocity function -/
def a : ℝ := 6

theorem point_motion :
  v 4 = 21 ∧ a = 6 := by sorry

end NUMINAMATH_CALUDE_point_motion_l2405_240540


namespace NUMINAMATH_CALUDE_custom_op_four_six_l2405_240597

def custom_op (a b : ℤ) : ℤ := 4*a - 2*b + a*b

theorem custom_op_four_six : custom_op 4 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_four_six_l2405_240597


namespace NUMINAMATH_CALUDE_water_sales_profit_profit_for_240_barrels_barrels_for_760_profit_l2405_240567

/-- Represents the daily sales and profit of a water sales department -/
structure WaterSales where
  fixed_costs : ℕ := 200
  cost_price : ℕ := 5
  selling_price : ℕ := 8

/-- Calculates the daily profit based on the number of barrels sold -/
def daily_profit (ws : WaterSales) (x : ℕ) : ℤ :=
  (ws.selling_price * x : ℤ) - (ws.cost_price * x : ℤ) - ws.fixed_costs

theorem water_sales_profit (ws : WaterSales) :
  ∀ x : ℕ, daily_profit ws x = 3 * x - 200 := by sorry

theorem profit_for_240_barrels (ws : WaterSales) :
  daily_profit ws 240 = 520 := by sorry

theorem barrels_for_760_profit (ws : WaterSales) :
  ∃ x : ℕ, daily_profit ws x = 760 ∧ x = 320 := by sorry

end NUMINAMATH_CALUDE_water_sales_profit_profit_for_240_barrels_barrels_for_760_profit_l2405_240567


namespace NUMINAMATH_CALUDE_inverse_variation_solution_l2405_240587

/-- The constant k in the inverse variation relationship -/
def k (a b : ℝ) : ℝ := a^3 * b^2

/-- The inverse variation relationship between a and b -/
def inverse_variation (a b : ℝ) : Prop := k a b = k 5 2

theorem inverse_variation_solution :
  ∀ a b : ℝ,
  inverse_variation a b →
  (a = 5 ∧ b = 2) ∨ (a = 2.5 ∧ b = 8) :=
sorry

end NUMINAMATH_CALUDE_inverse_variation_solution_l2405_240587


namespace NUMINAMATH_CALUDE_set_operations_and_range_of_a_l2405_240518

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a - 1}

-- State the theorem
theorem set_operations_and_range_of_a :
  ∀ a : ℝ,
  (B ∪ C a = B) →
  ((A ∩ B = {x | 3 < x ∧ x < 6}) ∧
   ((Set.compl A ∪ Set.compl B) = {x | x ≤ 3 ∨ x ≥ 6}) ∧
   (a ≤ 1 ∨ (2 ≤ a ∧ a ≤ 5))) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_range_of_a_l2405_240518


namespace NUMINAMATH_CALUDE_expression_evaluation_l2405_240559

theorem expression_evaluation (x y z : ℝ) 
  (hz : z = y - 11)
  (hy : y = x + 3)
  (hx : x = 5)
  (hd1 : x + 2 ≠ 0)
  (hd2 : y - 3 ≠ 0)
  (hd3 : z + 7 ≠ 0) :
  ((x + 3) / (x + 2)) * ((y - 2) / (y - 3)) * ((z + 9) / (z + 7)) = 72 / 35 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2405_240559


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2405_240507

def A : Set ℝ := {y | ∃ x : ℝ, y = x + 1}
def B : Set ℝ := {y | ∃ x : ℝ, y = 2 * x}

theorem intersection_of_A_and_B :
  A ∩ B = {y : ℝ | y ≥ 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2405_240507


namespace NUMINAMATH_CALUDE_unique_polynomial_property_l2405_240539

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The property that P satisfies for all real x and y -/
def SatisfiesProperty (P : RealPolynomial) : Prop :=
  ∀ x y : ℝ, |y^2 - P x| ≤ 2 * |x| ↔ |x^2 - P y| ≤ 2 * |y|

/-- The theorem stating that x^2 + 1 is the unique polynomial satisfying the given properties -/
theorem unique_polynomial_property : 
  ∃! P : RealPolynomial, 
    (P 0 = 1) ∧ 
    SatisfiesProperty P ∧ 
    ∀ x : ℝ, P x = x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_polynomial_property_l2405_240539


namespace NUMINAMATH_CALUDE_distinct_integer_quadruple_l2405_240595

theorem distinct_integer_quadruple : 
  ∀ a b c d : ℕ+, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    a + b = c * d →
    a * b = c + d →
    ((a = 1 ∧ b = 5 ∧ c = 3 ∧ d = 2) ∨
     (a = 1 ∧ b = 5 ∧ c = 2 ∧ d = 3) ∨
     (a = 5 ∧ b = 1 ∧ c = 3 ∧ d = 2) ∨
     (a = 5 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
     (a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 5) ∨
     (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 5) ∨
     (a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 1) ∨
     (a = 3 ∧ b = 2 ∧ c = 5 ∧ d = 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_distinct_integer_quadruple_l2405_240595


namespace NUMINAMATH_CALUDE_special_circle_equation_l2405_240554

/-- A circle passing through the origin and point (1, 1) with its center on the line 2x + 3y + 1 = 0 -/
def special_circle (x y : ℝ) : Prop :=
  (x - 4)^2 + (y + 3)^2 = 25

/-- The line on which the center of the circle lies -/
def center_line (x y : ℝ) : Prop :=
  2*x + 3*y + 1 = 0

theorem special_circle_equation :
  ∀ x y : ℝ,
  (special_circle x y ↔
    (x^2 + y^2 = 0 ∨ (x - 1)^2 + (y - 1)^2 = 0) ∧
    ∃ c_x c_y : ℝ, center_line c_x c_y ∧ (x - c_x)^2 + (y - c_y)^2 = (c_x^2 + c_y^2)) :=
by sorry

end NUMINAMATH_CALUDE_special_circle_equation_l2405_240554


namespace NUMINAMATH_CALUDE_subset_implies_a_in_set_l2405_240506

def A : Set ℝ := {x | x^2 = 1}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem subset_implies_a_in_set (a : ℝ) : B a ⊆ A → a ∈ ({-1, 0, 1} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_in_set_l2405_240506


namespace NUMINAMATH_CALUDE_total_savings_after_tax_l2405_240599

def total_income : ℝ := 18000

def income_ratio_a : ℝ := 3
def income_ratio_b : ℝ := 2
def income_ratio_c : ℝ := 1

def tax_rate_a : ℝ := 0.1
def tax_rate_b : ℝ := 0.15
def tax_rate_c : ℝ := 0

def expenditure_ratio : ℝ := 5
def income_ratio : ℝ := 9

theorem total_savings_after_tax :
  let income_a := (income_ratio_a / (income_ratio_a + income_ratio_b + income_ratio_c)) * total_income
  let income_b := (income_ratio_b / (income_ratio_a + income_ratio_b + income_ratio_c)) * total_income
  let income_c := (income_ratio_c / (income_ratio_a + income_ratio_b + income_ratio_c)) * total_income
  let tax_a := tax_rate_a * income_a
  let tax_b := tax_rate_b * income_b
  let tax_c := tax_rate_c * income_c
  let total_tax := tax_a + tax_b + tax_c
  let income_after_tax := total_income - total_tax
  let expenditure := (expenditure_ratio / income_ratio) * total_income
  let savings := income_after_tax - expenditure
  savings = 6200 := by sorry

end NUMINAMATH_CALUDE_total_savings_after_tax_l2405_240599


namespace NUMINAMATH_CALUDE_probability_of_identical_cubes_l2405_240531

/-- Represents the colors available for painting the cube faces -/
inductive Color
| Red
| Blue
| Green

/-- Represents a cube with painted faces -/
def Cube := Fin 6 → Color

/-- The total number of ways to paint a single cube -/
def totalWaysToPaintOneCube : ℕ := 729

/-- The total number of ways to paint two cubes -/
def totalWaysToPaintTwoCubes : ℕ := totalWaysToPaintOneCube * totalWaysToPaintOneCube

/-- Checks if two cubes are identical after rotation -/
def areIdenticalAfterRotation (cube1 cube2 : Cube) : Prop := sorry

/-- The number of ways two cubes can be painted to be identical after rotation -/
def waysToBeIdentical : ℕ := 66

/-- The probability that two independently painted cubes are identical after rotation -/
theorem probability_of_identical_cubes :
  (waysToBeIdentical : ℚ) / totalWaysToPaintTwoCubes = 2 / 16101 := by sorry

end NUMINAMATH_CALUDE_probability_of_identical_cubes_l2405_240531


namespace NUMINAMATH_CALUDE_gcd_not_eight_l2405_240503

theorem gcd_not_eight (x y : ℕ+) (h : y = x^2 + 8) : Nat.gcd x.val y.val ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_not_eight_l2405_240503


namespace NUMINAMATH_CALUDE_meal_distribution_theorem_l2405_240575

/-- The number of ways to derange 8 items -/
def derangement_8 : ℕ := 14833

/-- The number of ways to choose 2 items from 10 -/
def choose_2_from_10 : ℕ := 45

/-- The number of ways to distribute 10 meals of 4 types to 10 people
    such that exactly 2 people receive the correct meal type -/
def distribute_meals (d₈ : ℕ) (c₁₀₂ : ℕ) : ℕ := d₈ * c₁₀₂

theorem meal_distribution_theorem :
  distribute_meals derangement_8 choose_2_from_10 = 666885 := by
  sorry

end NUMINAMATH_CALUDE_meal_distribution_theorem_l2405_240575


namespace NUMINAMATH_CALUDE_platyfish_white_balls_l2405_240577

theorem platyfish_white_balls :
  let total_balls : ℕ := 80
  let num_goldfish : ℕ := 3
  let red_balls_per_goldfish : ℕ := 10
  let num_platyfish : ℕ := 10
  let total_red_balls : ℕ := num_goldfish * red_balls_per_goldfish
  let total_white_balls : ℕ := total_balls - total_red_balls
  let white_balls_per_platyfish : ℕ := total_white_balls / num_platyfish
  white_balls_per_platyfish = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_platyfish_white_balls_l2405_240577


namespace NUMINAMATH_CALUDE_effective_price_change_l2405_240542

theorem effective_price_change (P : ℝ) : 
  let price_after_first_discount := P * (1 - 0.3)
  let price_after_second_discount := price_after_first_discount * (1 - 0.2)
  let final_price := price_after_second_discount * (1 + 0.1)
  final_price = P * (1 - 0.384) :=
by sorry

end NUMINAMATH_CALUDE_effective_price_change_l2405_240542


namespace NUMINAMATH_CALUDE_lifeguard_swimming_distance_l2405_240512

/-- The problem of calculating the total swimming distance for a lifeguard test. -/
theorem lifeguard_swimming_distance 
  (front_crawl_speed : ℝ) 
  (breaststroke_speed : ℝ) 
  (total_time : ℝ) 
  (front_crawl_time : ℝ) 
  (h1 : front_crawl_speed = 45) 
  (h2 : breaststroke_speed = 35) 
  (h3 : total_time = 12) 
  (h4 : front_crawl_time = 8) :
  front_crawl_speed * front_crawl_time + breaststroke_speed * (total_time - front_crawl_time) = 500 := by
  sorry

#check lifeguard_swimming_distance

end NUMINAMATH_CALUDE_lifeguard_swimming_distance_l2405_240512


namespace NUMINAMATH_CALUDE_fraction_equality_l2405_240520

theorem fraction_equality (a b : ℝ) : |a^2 - b^2| / |(a - b)^2| = |a + b| / |a - b| := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2405_240520


namespace NUMINAMATH_CALUDE_toby_breakfast_pb_servings_l2405_240592

/-- Calculates the number of peanut butter servings needed for a target calorie count -/
def peanut_butter_servings (target_calories : ℕ) (bread_calories : ℕ) (pb_calories_per_serving : ℕ) : ℕ :=
  ((target_calories - bread_calories) / pb_calories_per_serving)

/-- Proves that 2 servings of peanut butter are needed for Toby's breakfast -/
theorem toby_breakfast_pb_servings :
  peanut_butter_servings 500 100 200 = 2 := by
  sorry

#eval peanut_butter_servings 500 100 200

end NUMINAMATH_CALUDE_toby_breakfast_pb_servings_l2405_240592


namespace NUMINAMATH_CALUDE_solution_y_l2405_240558

theorem solution_y (x y : ℝ) 
  (hx : x > 2) 
  (hy : y > 2) 
  (h1 : 1/x + 1/y = 3/4) 
  (h2 : x*y = 8) : 
  y = 4 :=
sorry

end NUMINAMATH_CALUDE_solution_y_l2405_240558


namespace NUMINAMATH_CALUDE_inequalities_for_M_l2405_240551

-- Define the set M
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- State the theorem
theorem inequalities_for_M (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2*|a - b|) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_for_M_l2405_240551


namespace NUMINAMATH_CALUDE_only_C_suitable_for_census_C_unique_suitable_for_census_l2405_240541

/-- Represents a survey option -/
inductive SurveyOption
| A  -- Understanding the vision of middle school students in our province
| B  -- Investigating the viewership of "The Reader"
| C  -- Inspecting the components of a newly developed fighter jet to ensure successful test flights
| D  -- Testing the lifespan of a batch of light bulbs

/-- Defines what makes a survey suitable for a census -/
def isSuitableForCensus (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.C => True
  | _ => False

/-- Theorem stating that only option C is suitable for a census -/
theorem only_C_suitable_for_census :
  ∀ (option : SurveyOption), isSuitableForCensus option ↔ option = SurveyOption.C :=
by
  sorry

/-- Corollary: Option C is the unique survey suitable for a census -/
theorem C_unique_suitable_for_census :
  ∃! (option : SurveyOption), isSuitableForCensus option :=
by
  sorry

end NUMINAMATH_CALUDE_only_C_suitable_for_census_C_unique_suitable_for_census_l2405_240541


namespace NUMINAMATH_CALUDE_linear_coefficient_of_example_quadratic_l2405_240508

/-- The coefficient of the linear term in a quadratic equation ax^2 + bx + c = 0 -/
def linearCoefficient (a b c : ℝ) : ℝ := b

theorem linear_coefficient_of_example_quadratic :
  linearCoefficient 1 (-5) (-2) = -5 := by sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_example_quadratic_l2405_240508


namespace NUMINAMATH_CALUDE_unique_digit_arrangement_l2405_240515

theorem unique_digit_arrangement :
  ∃! (A B C D E : ℕ),
    (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧ E ≠ 0) ∧
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
     C ≠ D ∧ C ≠ E ∧
     D ≠ E) ∧
    (A + B : ℚ) = (C + D + E : ℚ) / 7 ∧
    (A + C : ℚ) = (B + D + E : ℚ) / 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_arrangement_l2405_240515


namespace NUMINAMATH_CALUDE_jimin_english_score_l2405_240585

def jimin_scores (science social_studies english : ℕ) : Prop :=
  social_studies = science + 6 ∧
  science = 87 ∧
  (science + social_studies + english) / 3 = 92

theorem jimin_english_score :
  ∀ science social_studies english : ℕ,
  jimin_scores science social_studies english →
  english = 96 := by sorry

end NUMINAMATH_CALUDE_jimin_english_score_l2405_240585


namespace NUMINAMATH_CALUDE_circle_division_evenness_l2405_240594

theorem circle_division_evenness (N : ℕ) : 
  (∃ (chords : Fin N → Fin (2 * N) × Fin (2 * N)),
    (∀ i : Fin N, (chords i).1 ≠ (chords i).2) ∧ 
    (∀ i j : Fin N, i ≠ j → (chords i).1 ≠ (chords j).1 ∧ (chords i).1 ≠ (chords j).2 ∧
                            (chords i).2 ≠ (chords j).1 ∧ (chords i).2 ≠ (chords j).2) ∧
    (∀ i : Fin N, ∃ k l : ℕ, 
      (((chords i).2 - (chords i).1 : ℤ) % (2 * N : ℤ) = 2 * k ∨
       ((chords i).1 - (chords i).2 : ℤ) % (2 * N : ℤ) = 2 * k) ∧
      (((chords i).2 - (chords i).1 : ℤ) % (2 * N : ℤ) = 2 * l ∨
       ((chords i).1 - (chords i).2 : ℤ) % (2 * N : ℤ) = 2 * l) ∧
      k + l = N)) →
  Even N :=
by sorry

end NUMINAMATH_CALUDE_circle_division_evenness_l2405_240594


namespace NUMINAMATH_CALUDE_eighth_term_of_geometric_sequence_l2405_240548

/-- Given a geometric sequence with first term 3 and second term 9/2,
    prove that the eighth term is 6561/128 -/
theorem eighth_term_of_geometric_sequence (a : ℕ → ℚ)
  (h1 : a 1 = 3)
  (h2 : a 2 = 9/2)
  (h_geom : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * (a 2 / a 1)) :
  a 8 = 6561/128 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_of_geometric_sequence_l2405_240548


namespace NUMINAMATH_CALUDE_quarters_ratio_proof_l2405_240529

def initial_quarters : ℕ := 50
def doubled_quarters : ℕ := initial_quarters * 2
def collected_second_year : ℕ := 3 * 12
def collected_third_year : ℕ := 1 * (12 / 3)
def total_before_loss : ℕ := doubled_quarters + collected_second_year + collected_third_year
def quarters_remaining : ℕ := 105

theorem quarters_ratio_proof :
  (total_before_loss - quarters_remaining) * 4 = total_before_loss :=
by sorry

end NUMINAMATH_CALUDE_quarters_ratio_proof_l2405_240529


namespace NUMINAMATH_CALUDE_product_of_two_numbers_l2405_240522

theorem product_of_two_numbers (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (sum_squares_eq : x^2 + y^2 = 120) : 
  x * y = -20 := by
  sorry

end NUMINAMATH_CALUDE_product_of_two_numbers_l2405_240522


namespace NUMINAMATH_CALUDE_rip3_properties_l2405_240546

-- Define the basic concepts
def Cell : Type := sorry
def RIP3 : Type := sorry
def Gene : Type := sorry

-- Define the properties and relationships
def can_convert_apoptosis_to_necrosis (r : RIP3) : Prop := sorry
def controls_synthesis_of (g : Gene) (r : RIP3) : Prop := sorry
def exists_in_human_body (r : RIP3) : Prop := sorry
def can_regulate_cell_death_mode (r : RIP3) : Prop := sorry
def has_gene (c : Cell) (g : Gene) : Prop := sorry

-- State the theorem
theorem rip3_properties :
  ∃ (r : RIP3) (g : Gene),
    exists_in_human_body r ∧
    can_convert_apoptosis_to_necrosis r ∧
    controls_synthesis_of g r ∧
    can_regulate_cell_death_mode r ∧
    ∀ (c : Cell), has_gene c g :=
sorry

-- Note: This theorem encapsulates the main points about RIP3 from the problem statement,
-- without making claims about the correctness or incorrectness of the given statements.

end NUMINAMATH_CALUDE_rip3_properties_l2405_240546


namespace NUMINAMATH_CALUDE_novel_pages_count_l2405_240555

/-- Represents the number of pages in the novel -/
def total_pages : ℕ := 420

/-- Pages read on the first day -/
def pages_read_day1 (x : ℕ) : ℕ := x / 4 + 10

/-- Pages read on the second day -/
def pages_read_day2 (x : ℕ) : ℕ := (x - pages_read_day1 x) / 3 + 20

/-- Pages read on the third day -/
def pages_read_day3 (x : ℕ) : ℕ := (x - pages_read_day1 x - pages_read_day2 x) / 2 + 40

/-- Pages remaining after the third day -/
def pages_remaining (x : ℕ) : ℕ := x - pages_read_day1 x - pages_read_day2 x - pages_read_day3 x

theorem novel_pages_count : pages_remaining total_pages = 50 := by sorry

end NUMINAMATH_CALUDE_novel_pages_count_l2405_240555


namespace NUMINAMATH_CALUDE_fruit_vendor_sales_l2405_240519

/-- A fruit vendor's sales problem -/
theorem fruit_vendor_sales (apple_price orange_price : ℚ)
  (morning_apples morning_oranges : ℕ)
  (afternoon_oranges : ℕ)
  (total_sales : ℚ)
  (h1 : apple_price = 3/2)
  (h2 : orange_price = 1)
  (h3 : morning_apples = 40)
  (h4 : morning_oranges = 30)
  (h5 : afternoon_oranges = 40)
  (h6 : total_sales = 205) :
  ∃ (afternoon_apples : ℕ),
    afternoon_apples = 50 ∧
    total_sales = apple_price * (morning_apples + afternoon_apples) +
                  orange_price * (morning_oranges + afternoon_oranges) :=
by sorry

end NUMINAMATH_CALUDE_fruit_vendor_sales_l2405_240519


namespace NUMINAMATH_CALUDE_book_store_inventory_l2405_240544

theorem book_store_inventory (initial : ℝ) (first_addition : ℝ) (second_addition : ℝ) 
  (h1 : initial = 41.0)
  (h2 : first_addition = 33.0)
  (h3 : second_addition = 2.0) :
  initial + first_addition + second_addition = 76.0 := by
  sorry

end NUMINAMATH_CALUDE_book_store_inventory_l2405_240544


namespace NUMINAMATH_CALUDE_money_value_difference_l2405_240556

def euro_to_dollar : ℝ := 1.5
def diana_dollars : ℝ := 600
def etienne_euros : ℝ := 450

theorem money_value_difference : 
  let etienne_dollars := etienne_euros * euro_to_dollar
  let percentage_diff := (diana_dollars - etienne_dollars) / etienne_dollars * 100
  ∀ ε > 0, |percentage_diff + 11.11| < ε :=
sorry

end NUMINAMATH_CALUDE_money_value_difference_l2405_240556


namespace NUMINAMATH_CALUDE_power_of_three_plus_five_mod_ten_l2405_240534

theorem power_of_three_plus_five_mod_ten : (3^108 + 5) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_plus_five_mod_ten_l2405_240534


namespace NUMINAMATH_CALUDE_smallest_angle_theorem_l2405_240510

/-- The smallest positive angle y in degrees that satisfies the equation 
    9 sin(y) cos³(y) - 9 sin³(y) cos(y) = 3√2 is 22.5° -/
theorem smallest_angle_theorem : 
  ∃ y : ℝ, y > 0 ∧ y < 360 ∧ 
  (9 * Real.sin y * (Real.cos y)^3 - 9 * (Real.sin y)^3 * Real.cos y = 3 * Real.sqrt 2) ∧
  (∀ z : ℝ, z > 0 ∧ z < y → 
    9 * Real.sin z * (Real.cos z)^3 - 9 * (Real.sin z)^3 * Real.cos z ≠ 3 * Real.sqrt 2) ∧
  y = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_theorem_l2405_240510


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1384_l2405_240545

theorem rightmost_three_digits_of_7_to_1384 :
  7^1384 ≡ 401 [ZMOD 1000] :=
by sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1384_l2405_240545


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l2405_240550

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 3
  f 2 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l2405_240550


namespace NUMINAMATH_CALUDE_lottery_probability_theorem_l2405_240517

/-- The number of StarBalls in the lottery -/
def num_starballs : ℕ := 30

/-- The number of MagicBalls in the lottery -/
def num_magicballs : ℕ := 49

/-- The number of MagicBalls that need to be picked -/
def num_picked_magicballs : ℕ := 6

/-- The probability of winning the lottery -/
def lottery_win_probability : ℚ := 1 / 419514480

theorem lottery_probability_theorem :
  (1 : ℚ) / num_starballs * (1 : ℚ) / (num_magicballs.choose num_picked_magicballs) = lottery_win_probability := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_theorem_l2405_240517


namespace NUMINAMATH_CALUDE_first_equation_is_golden_second_equation_root_values_l2405_240530

/-- Definition of a golden equation -/
def is_golden_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ a - b + c = 0

/-- The first equation -/
def first_equation (x : ℝ) : Prop :=
  2 * x^2 + 5 * x + 3 = 0

/-- The second equation -/
def second_equation (x a b : ℝ) : Prop :=
  3 * x^2 - a * x + b = 0

/-- Theorem for the first part -/
theorem first_equation_is_golden :
  is_golden_equation 2 5 3 :=
sorry

/-- Theorem for the second part -/
theorem second_equation_root_values (a b : ℝ) :
  is_golden_equation 3 (-a) b →
  second_equation a a b →
  (a = -1 ∨ a = 3/2) :=
sorry

end NUMINAMATH_CALUDE_first_equation_is_golden_second_equation_root_values_l2405_240530


namespace NUMINAMATH_CALUDE_vehicle_license_count_l2405_240549

/-- The number of possible letters for a license -/
def num_letters : ℕ := 3

/-- The number of possible digits for each position in a license -/
def num_digits : ℕ := 10

/-- The number of digit positions in a license -/
def num_digit_positions : ℕ := 6

/-- The total number of possible vehicle licenses -/
def total_licenses : ℕ := num_letters * (num_digits ^ num_digit_positions)

theorem vehicle_license_count :
  total_licenses = 3000000 := by sorry

end NUMINAMATH_CALUDE_vehicle_license_count_l2405_240549


namespace NUMINAMATH_CALUDE_extreme_points_inequality_l2405_240516

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a * log x

theorem extreme_points_inequality (a : ℝ) (m : ℝ) :
  a > 0 →
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧
    (∀ x, x > 0 → (deriv (f a)) x = 0 ↔ x = x₁ ∨ x = x₂) →
    (∀ x, x > 0 → f a x₁ ≥ m * x₂) →
    m ≤ -3/2 - log 2 := by
  sorry

end NUMINAMATH_CALUDE_extreme_points_inequality_l2405_240516


namespace NUMINAMATH_CALUDE_danys_farm_bushels_l2405_240527

/-- Calculates the total number of bushels needed for animals on a farm for one day. -/
def total_bushels_needed (num_cows num_sheep num_chickens : ℕ) 
  (cow_sheep_consumption chicken_consumption : ℕ) : ℕ :=
  (num_cows + num_sheep) * cow_sheep_consumption + num_chickens * chicken_consumption

/-- Theorem stating the total number of bushels needed for Dany's farm animals for one day. -/
theorem danys_farm_bushels : 
  total_bushels_needed 4 3 7 2 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_danys_farm_bushels_l2405_240527


namespace NUMINAMATH_CALUDE_baking_powder_inventory_l2405_240574

/-- Given that Kelly had 0.4 box of baking powder yesterday and 0.1 more box
    yesterday compared to now, prove that she has 0.3 box of baking powder now. -/
theorem baking_powder_inventory (yesterday : ℝ) (difference : ℝ) (now : ℝ)
    (h1 : yesterday = 0.4)
    (h2 : difference = 0.1)
    (h3 : yesterday = now + difference) :
    now = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_baking_powder_inventory_l2405_240574


namespace NUMINAMATH_CALUDE_cube_power_eq_l2405_240589

theorem cube_power_eq : (3^3 * 6^3)^2 = 34062224 := by
  sorry

end NUMINAMATH_CALUDE_cube_power_eq_l2405_240589


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_sum_l2405_240532

theorem polynomial_equality_implies_sum (m n : ℝ) : 
  (∀ x : ℝ, (x + 5) * (x + n) = x^2 + m*x - 5) → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_sum_l2405_240532


namespace NUMINAMATH_CALUDE_recipe_total_cups_l2405_240547

theorem recipe_total_cups (butter baking_soda flour sugar : ℚ)
  (ratio : butter = 1 ∧ baking_soda = 2 ∧ flour = 5 ∧ sugar = 3)
  (flour_cups : flour * 3 = 15) :
  butter * 3 + baking_soda * 3 + 15 + sugar * 3 = 33 := by
sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l2405_240547


namespace NUMINAMATH_CALUDE_binomial_expansion_fourth_fifth_terms_sum_zero_l2405_240588

/-- Given a binomial expansion (a-b)^n where n ≥ 2, ab ≠ 0, and a = mb with m = k + 2 and k a positive integer,
    prove that n = 2m + 3 makes the sum of the fourth and fifth terms zero. -/
theorem binomial_expansion_fourth_fifth_terms_sum_zero 
  (n : ℕ) (a b : ℝ) (m k : ℕ) :
  n ≥ 2 →
  a ≠ 0 →
  b ≠ 0 →
  k > 0 →
  m = k + 2 →
  a = m * b →
  (n = 2 * m + 3 ↔ 
    (Nat.choose n 3) * (a - b)^(n - 3) * b^3 + 
    (Nat.choose n 4) * (a - b)^(n - 4) * b^4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_fourth_fifth_terms_sum_zero_l2405_240588


namespace NUMINAMATH_CALUDE_mary_flour_needed_l2405_240568

/-- The number of cups of flour required by the recipe -/
def recipe_flour : ℕ := 9

/-- The number of cups of flour Mary has already added -/
def added_flour : ℕ := 2

/-- The number of cups of flour Mary needs to add -/
def flour_needed : ℕ := recipe_flour - added_flour

theorem mary_flour_needed : flour_needed = 7 := by sorry

end NUMINAMATH_CALUDE_mary_flour_needed_l2405_240568


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l2405_240509

def cryptarithm (C L M O P S U W Y : ℕ) : Prop :=
  let MSU := 100 * M + 10 * S + U
  let OLYMP := 10000 * O + 1000 * L + 100 * Y + 10 * M + P
  let MOSCOW := 100000 * M + 10000 * O + 1000 * S + 100 * C + 10 * O + W
  4 * MSU + 2 * OLYMP = MOSCOW

theorem cryptarithm_solution :
  ∃ (C L M O P S U W Y : ℕ),
    C = 5 ∧ L = 7 ∧ M = 1 ∧ O = 9 ∧ P = 2 ∧ S = 4 ∧ U = 3 ∧ W = 6 ∧ Y = 0 ∧
    cryptarithm C L M O P S U W Y ∧
    C ≠ L ∧ C ≠ M ∧ C ≠ O ∧ C ≠ P ∧ C ≠ S ∧ C ≠ U ∧ C ≠ W ∧ C ≠ Y ∧
    L ≠ M ∧ L ≠ O ∧ L ≠ P ∧ L ≠ S ∧ L ≠ U ∧ L ≠ W ∧ L ≠ Y ∧
    M ≠ O ∧ M ≠ P ∧ M ≠ S ∧ M ≠ U ∧ M ≠ W ∧ M ≠ Y ∧
    O ≠ P ∧ O ≠ S ∧ O ≠ U ∧ O ≠ W ∧ O ≠ Y ∧
    P ≠ S ∧ P ≠ U ∧ P ≠ W ∧ P ≠ Y ∧
    S ≠ U ∧ S ≠ W ∧ S ≠ Y ∧
    U ≠ W ∧ U ≠ Y ∧
    W ≠ Y :=
by sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l2405_240509


namespace NUMINAMATH_CALUDE_min_disks_for_profit_l2405_240536

/-- The number of disks Maria buys for $7 -/
def buy_rate : ℕ := 5

/-- The number of disks Maria sells for $7 -/
def sell_rate : ℕ := 4

/-- The price in dollars for buying or selling the respective number of disks -/
def price : ℚ := 7

/-- The desired profit in dollars -/
def target_profit : ℚ := 125

/-- The cost of buying one disk -/
def cost_per_disk : ℚ := price / buy_rate

/-- The revenue from selling one disk -/
def revenue_per_disk : ℚ := price / sell_rate

/-- The profit made from selling one disk -/
def profit_per_disk : ℚ := revenue_per_disk - cost_per_disk

theorem min_disks_for_profit : 
  ∀ n : ℕ, (n : ℚ) * profit_per_disk ≥ target_profit → n ≥ 358 :=
by sorry

end NUMINAMATH_CALUDE_min_disks_for_profit_l2405_240536


namespace NUMINAMATH_CALUDE_remainder_problem_l2405_240543

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k < 41) 
  (h3 : k % 5 = 2) 
  (h4 : k % 6 = 5) : 
  k % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2405_240543


namespace NUMINAMATH_CALUDE_f_values_l2405_240514

def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x

theorem f_values : f 2 = 14 ∧ f (-2) = 2 := by sorry

end NUMINAMATH_CALUDE_f_values_l2405_240514


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l2405_240528

/-- Represents the ages of a father and son -/
structure Ages where
  son : ℕ
  father : ℕ

/-- The current ages of the father and son -/
def currentAges : Ages :=
  { son := 24, father := 72 }

/-- The ages of the father and son 8 years ago -/
def pastAges : Ages :=
  { son := currentAges.son - 8, father := currentAges.father - 8 }

/-- The ratio of the father's age to the son's age -/
def ageRatio (ages : Ages) : ℚ :=
  ages.father / ages.son

theorem father_son_age_ratio :
  (pastAges.father = 4 * pastAges.son) →
  ageRatio currentAges = 3 / 1 := by
  sorry

#eval ageRatio currentAges

end NUMINAMATH_CALUDE_father_son_age_ratio_l2405_240528


namespace NUMINAMATH_CALUDE_tank_filling_time_l2405_240535

/-- Given a tank and three hoses X, Y, and Z, prove that they together fill the tank in 24/13 hours. -/
theorem tank_filling_time (T X Y Z : ℝ) (hxy : T = 2 * (X + Y)) (hxz : T = 3 * (X + Z)) (hyz : T = 4 * (Y + Z)) :
  T / (X + Y + Z) = 24 / 13 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l2405_240535


namespace NUMINAMATH_CALUDE_initial_cookies_count_l2405_240584

/-- The number of cookies initially in the package -/
def initial_cookies : ℕ := sorry

/-- The number of cookies left after eating some -/
def cookies_left : ℕ := 9

/-- The number of cookies eaten -/
def cookies_eaten : ℕ := 9

/-- Theorem stating that the initial number of cookies is 18 -/
theorem initial_cookies_count : initial_cookies = cookies_left + cookies_eaten := by sorry

end NUMINAMATH_CALUDE_initial_cookies_count_l2405_240584


namespace NUMINAMATH_CALUDE_product_of_system_l2405_240502

theorem product_of_system (a b c d : ℚ) : 
  (4 * a + 2 * b + 5 * c + 8 * d = 67) →
  (4 * (d + c) = b) →
  (2 * b + 3 * c = a) →
  (c + 1 = d) →
  (a * b * c * d = (1201 * 572 * 19 * 124 : ℚ) / 105^4) := by
  sorry

end NUMINAMATH_CALUDE_product_of_system_l2405_240502


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l2405_240581

-- Define the product
def product : ℕ := 45 * 320 * 60

-- Define a function to count trailing zeros
def count_trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 10 = 0 then 1 + count_trailing_zeros (n / 10)
  else 0

-- Theorem statement
theorem product_trailing_zeros :
  count_trailing_zeros product = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l2405_240581


namespace NUMINAMATH_CALUDE_club_size_after_four_years_l2405_240570

/-- Represents the number of people in the club after k years -/
def club_size (k : ℕ) : ℕ :=
  match k with
  | 0 => 20
  | n + 1 => 3 * club_size n - 14

/-- The theorem stating the club size after 4 years -/
theorem club_size_after_four_years :
  club_size 4 = 1060 := by
  sorry

end NUMINAMATH_CALUDE_club_size_after_four_years_l2405_240570


namespace NUMINAMATH_CALUDE_polynomial_expansion_sum_l2405_240538

theorem polynomial_expansion_sum (m : ℝ) (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (1 + m * x)^6 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 63 →
  m = 1 ∨ m = -3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_sum_l2405_240538


namespace NUMINAMATH_CALUDE_inequality_proof_l2405_240593

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2405_240593


namespace NUMINAMATH_CALUDE_softball_team_ratio_l2405_240579

theorem softball_team_ratio :
  ∀ (men women : ℕ),
  women = men + 4 →
  men + women = 14 →
  (men : ℚ) / women = 5 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l2405_240579


namespace NUMINAMATH_CALUDE_opposite_greater_implies_negative_l2405_240565

theorem opposite_greater_implies_negative (x : ℝ) : -x > x → x < 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_greater_implies_negative_l2405_240565


namespace NUMINAMATH_CALUDE_complex_square_root_l2405_240576

theorem complex_square_root (z : ℂ) : z^2 = -5 - 12*I → z = 2 - 3*I ∨ z = -2 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_l2405_240576


namespace NUMINAMATH_CALUDE_sweet_shop_candy_cases_l2405_240513

/-- The number of cases of chocolate bars in the Sweet Shop -/
def chocolate_cases : ℕ := 80 - 55

/-- The total number of cases of candy in the Sweet Shop -/
def total_cases : ℕ := 80

/-- The number of cases of lollipops in the Sweet Shop -/
def lollipop_cases : ℕ := 55

theorem sweet_shop_candy_cases : chocolate_cases = 25 := by
  sorry

end NUMINAMATH_CALUDE_sweet_shop_candy_cases_l2405_240513


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l2405_240566

theorem max_value_of_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  (∀ x' y', -5 ≤ x' ∧ x' ≤ -3 ∧ 3 ≤ y' ∧ y' ≤ 5 → (x' + y' + 1) / x' ≤ (x + y + 1) / x) →
  (x + y + 1) / x = -1/5 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l2405_240566


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l2405_240578

theorem shopkeeper_profit_percentage 
  (cost_price : ℝ) 
  (cost_price_positive : cost_price > 0) :
  let markup_percentage : ℝ := 30
  let discount_percentage : ℝ := 18.461538461538467
  let marked_price : ℝ := cost_price * (1 + markup_percentage / 100)
  let selling_price : ℝ := marked_price * (1 - discount_percentage / 100)
  let profit : ℝ := selling_price - cost_price
  let profit_percentage : ℝ := (profit / cost_price) * 100
  profit_percentage = 6 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l2405_240578


namespace NUMINAMATH_CALUDE_function_property_l2405_240571

/-- Given a function f(x, y) = kx + 1/y, if f(a, b) = f(b, a) for a ≠ b, then f(ab, 1) = 0 -/
theorem function_property (k : ℝ) (a b : ℝ) (h : a ≠ b) :
  (k * a + 1 / b = k * b + 1 / a) → (k * (a * b) + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_function_property_l2405_240571


namespace NUMINAMATH_CALUDE_central_angle_specific_points_l2405_240563

/-- A point on a sphere represented by its latitude and longitude -/
structure SpherePoint where
  latitude : Real
  longitude : Real

/-- The central angle between two points on a sphere -/
def centralAngle (center : Point) (p1 p2 : SpherePoint) : Real :=
  sorry

theorem central_angle_specific_points :
  let center : Point := sorry
  let pointA : SpherePoint := { latitude := 0, longitude := 110 }
  let pointB : SpherePoint := { latitude := 45, longitude := -115 }
  centralAngle center pointA pointB = 120 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_specific_points_l2405_240563


namespace NUMINAMATH_CALUDE_joan_seashells_l2405_240590

/-- The number of seashells Joan has after a series of events -/
def final_seashells (initial : ℕ) (given : ℕ) (found : ℕ) (traded : ℕ) (received : ℕ) (lost : ℕ) : ℕ :=
  initial - given + found - traded + received - lost

/-- Theorem stating that Joan ends up with 51 seashells -/
theorem joan_seashells :
  final_seashells 79 63 45 20 15 5 = 51 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l2405_240590


namespace NUMINAMATH_CALUDE_discount_calculation_l2405_240560

theorem discount_calculation (original_price discount_percentage : ℝ) 
  (h1 : original_price = 80)
  (h2 : discount_percentage = 15) : 
  original_price * (1 - discount_percentage / 100) = 68 := by
sorry

end NUMINAMATH_CALUDE_discount_calculation_l2405_240560


namespace NUMINAMATH_CALUDE_total_money_l2405_240537

/-- Given three people A, B, and C with some money between them, 
    prove that their total amount is 450 under certain conditions. -/
theorem total_money (a b c : ℕ) : 
  a + c = 200 → b + c = 350 → c = 100 → a + b + c = 450 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l2405_240537


namespace NUMINAMATH_CALUDE_correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_D_l2405_240505

-- Define variables
variable (a b x y : ℝ)

-- Theorem for the correct calculation (Option C)
theorem correct_calculation : 6 * x^3 * y^2 / (3 * x) = 2 * x^2 * y^2 := by sorry

-- Theorems for the incorrect calculations
theorem incorrect_calculation_A : (a - b) * (-a - b) ≠ a^2 - b^2 := by sorry

theorem incorrect_calculation_B : 2 * a^3 + 3 * a^3 ≠ 5 * a^6 := by sorry

theorem incorrect_calculation_D : (-2 * x^2)^3 ≠ -6 * x^6 := by sorry

end NUMINAMATH_CALUDE_correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_D_l2405_240505


namespace NUMINAMATH_CALUDE_triangle_properties_l2405_240557

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

theorem triangle_properties (t : Triangle) 
  (h1 : t.a ≠ t.b)
  (h2 : Real.cos t.A ^ 2 - Real.cos t.B ^ 2 = Real.sqrt 3 * Real.sin t.A * Real.cos t.A - Real.sqrt 3 * Real.sin t.B * Real.cos t.B)
  (h3 : t.c = Real.sqrt 3)
  (h4 : Real.sin t.A = Real.sqrt 2 / 2) :
  t.C = π / 3 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B) = (3 + Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2405_240557


namespace NUMINAMATH_CALUDE_license_plate_count_l2405_240569

/-- The number of possible letters in each position of the license plate. -/
def num_letters : ℕ := 26

/-- The number of positions for digits in the license plate. -/
def num_digit_positions : ℕ := 3

/-- The number of ways to choose positions for odd digits. -/
def num_odd_digit_arrangements : ℕ := 3

/-- The number of possible odd digits. -/
def num_odd_digits : ℕ := 5

/-- The number of possible even digits. -/
def num_even_digits : ℕ := 5

/-- The total number of license plates with 3 letters followed by 3 digits,
    where exactly two digits are odd and one is even. -/
theorem license_plate_count : 
  (num_letters ^ 3) * num_odd_digit_arrangements * (num_odd_digits ^ 2 * num_even_digits) = 6591000 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l2405_240569


namespace NUMINAMATH_CALUDE_sin_plus_power_cos_pi_third_l2405_240598

theorem sin_plus_power_cos_pi_third :
  Real.sin 3 + 2^(8-3) * Real.cos (π/3) = Real.sin 3 + 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_power_cos_pi_third_l2405_240598


namespace NUMINAMATH_CALUDE_power_division_l2405_240583

theorem power_division (x : ℝ) : x^8 / x^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l2405_240583


namespace NUMINAMATH_CALUDE_large_box_length_l2405_240572

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Theorem: If a large box with dimensions L × 14 × 16 can fit exactly 64 small boxes
    with dimensions 3 × 7 × 2, then L must be 12. -/
theorem large_box_length (L : ℝ) : 
  let largeBox : BoxDimensions := ⟨L, 14, 16⟩
  let smallBox : BoxDimensions := ⟨3, 7, 2⟩
  (boxVolume largeBox) / (boxVolume smallBox) = 64 → L = 12 := by
sorry

end NUMINAMATH_CALUDE_large_box_length_l2405_240572


namespace NUMINAMATH_CALUDE_identity_condition_l2405_240582

theorem identity_condition (a b c : ℝ) : 
  (∀ x y z : ℝ, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) → 
  ((a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = 0 ∧ b = 0 ∧ c = -1)) := by
  sorry

end NUMINAMATH_CALUDE_identity_condition_l2405_240582


namespace NUMINAMATH_CALUDE_average_marks_of_passed_boys_l2405_240596

theorem average_marks_of_passed_boys
  (total_boys : ℕ)
  (overall_average : ℚ)
  (passed_boys : ℕ)
  (failed_average : ℚ)
  (h1 : total_boys = 120)
  (h2 : overall_average = 38)
  (h3 : passed_boys = 115)
  (h4 : failed_average = 15)
  : ∃ (passed_average : ℚ), passed_average = 39 ∧
    overall_average * total_boys = passed_average * passed_boys + failed_average * (total_boys - passed_boys) := by
  sorry

end NUMINAMATH_CALUDE_average_marks_of_passed_boys_l2405_240596


namespace NUMINAMATH_CALUDE_number_of_values_l2405_240511

theorem number_of_values (initial_mean correct_mean : ℝ) 
  (wrong_value correct_value : ℝ) : 
  initial_mean = 140 ∧ 
  wrong_value = 135 ∧ 
  correct_value = 145 ∧ 
  correct_mean = 140.33333333333334 → 
  ∃ n : ℕ, n = 30 ∧ 
    (n : ℝ) * initial_mean - (correct_value - wrong_value) = n * correct_mean :=
by sorry

end NUMINAMATH_CALUDE_number_of_values_l2405_240511


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l2405_240521

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_periodic : has_period f 5)
  (h1 : f 1 = 1)
  (h2 : f 2 = 2) :
  f 3 - f 4 = -1 := by
sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l2405_240521


namespace NUMINAMATH_CALUDE_tangent_lines_perpendicular_PQR_inequality_l2405_240564

-- Define the function f and its inverse g
noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.exp x

-- Define the slopes of the tangent lines
noncomputable def k₁ : ℝ := -1 / Real.exp 1
noncomputable def k₂ : ℝ := Real.exp 1

-- Define P, Q, and R
noncomputable def P (a b : ℝ) : ℝ := g ((a + b) / 2)
noncomputable def Q (a b : ℝ) : ℝ := (g a - g b) / (a - b)
noncomputable def R (a b : ℝ) : ℝ := (g a + g b) / 2

-- State the theorems to be proved
theorem tangent_lines_perpendicular : k₁ * k₂ = -1 := by sorry

theorem PQR_inequality (a b : ℝ) (h : a ≠ b) : P a b < Q a b ∧ Q a b < R a b := by sorry

end NUMINAMATH_CALUDE_tangent_lines_perpendicular_PQR_inequality_l2405_240564


namespace NUMINAMATH_CALUDE_tv_cost_is_250_l2405_240523

/-- The cost of the TV given Linda's savings and furniture expenditure -/
def tv_cost (savings : ℚ) (furniture_fraction : ℚ) : ℚ :=
  savings * (1 - furniture_fraction)

/-- Theorem stating that the TV cost is $250 given the problem conditions -/
theorem tv_cost_is_250 :
  tv_cost 1000 (3/4) = 250 := by
  sorry

end NUMINAMATH_CALUDE_tv_cost_is_250_l2405_240523


namespace NUMINAMATH_CALUDE_point_not_in_fourth_quadrant_l2405_240580

theorem point_not_in_fourth_quadrant :
  ¬ ∃ a : ℝ, (a - 3 > 0 ∧ a + 3 < 0) := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_fourth_quadrant_l2405_240580


namespace NUMINAMATH_CALUDE_exactly_three_solutions_exactly_two_solutions_l2405_240561

/-- The number of solutions for the system of equations:
    5|x| - 12|y| = 5
    x^2 + y^2 - 28x + 196 - a^2 = 0
-/
def numSolutions (a : ℝ) : ℕ :=
  sorry

/-- The system has exactly 3 solutions if and only if |a| = 13 or |a| = 15 -/
theorem exactly_three_solutions (a : ℝ) :
  numSolutions a = 3 ↔ (abs a = 13 ∨ abs a = 15) :=
sorry

/-- The system has exactly 2 solutions if and only if |a| = 5 or 13 < |a| < 15 -/
theorem exactly_two_solutions (a : ℝ) :
  numSolutions a = 2 ↔ (abs a = 5 ∨ (13 < abs a ∧ abs a < 15)) :=
sorry

end NUMINAMATH_CALUDE_exactly_three_solutions_exactly_two_solutions_l2405_240561


namespace NUMINAMATH_CALUDE_t_grid_sum_l2405_240562

/-- Represents a T-shaped grid with 6 distinct digits --/
structure TGrid where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
               d ≠ e ∧ d ≠ f ∧
               e ≠ f
  h_range : a ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
            b ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
            c ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
            d ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
            e ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
            f ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ)
  h_vertical_sum : a + b + c = 20
  h_horizontal_sum : d + f = 7

theorem t_grid_sum (g : TGrid) : a + b + c + d + e + f = 33 := by
  sorry


end NUMINAMATH_CALUDE_t_grid_sum_l2405_240562


namespace NUMINAMATH_CALUDE_rob_has_three_dimes_l2405_240586

/-- Represents the number of coins of each type Rob has -/
structure RobsCoins where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value of Rob's coins in cents -/
def totalValue (coins : RobsCoins) : ℕ :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- Theorem stating that given Rob's coin counts and total value, he must have 3 dimes -/
theorem rob_has_three_dimes :
  ∀ (coins : RobsCoins),
    coins.quarters = 7 →
    coins.nickels = 5 →
    coins.pennies = 12 →
    totalValue coins = 242 →
    coins.dimes = 3 := by
  sorry


end NUMINAMATH_CALUDE_rob_has_three_dimes_l2405_240586
