import Mathlib

namespace NUMINAMATH_CALUDE_perimeter_increase_first_to_fourth_l1141_114180

/-- Calculates the perimeter of an equilateral triangle given its side length -/
def trianglePerimeter (side : ℝ) : ℝ := 3 * side

/-- Calculates the side length of the nth triangle in the sequence -/
def nthTriangleSide (n : ℕ) : ℝ :=
  3 * (1.6 ^ n)

/-- Theorem stating the percent increase in perimeter from the first to the fourth triangle -/
theorem perimeter_increase_first_to_fourth :
  let first_perimeter := trianglePerimeter 3
  let fourth_perimeter := trianglePerimeter (nthTriangleSide 3)
  (fourth_perimeter - first_perimeter) / first_perimeter * 100 = 309.6 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_increase_first_to_fourth_l1141_114180


namespace NUMINAMATH_CALUDE_hotel_revenue_maximization_l1141_114179

/-- Represents the hotel revenue optimization problem -/
def HotelRevenueProblem (totalRooms : ℕ) (initialPrice : ℕ) (initialOccupancy : ℕ) 
  (priceReduction : ℕ) (occupancyIncrease : ℕ) : Prop :=
  ∃ (maxRevenue : ℕ),
    maxRevenue = 22500 ∧
    (∀ (x : ℕ),
      let newPrice := initialPrice - x * priceReduction
      let newOccupancy := initialOccupancy + x * occupancyIncrease
      newPrice * newOccupancy ≤ maxRevenue)

/-- Theorem stating that the hotel revenue problem has a solution -/
theorem hotel_revenue_maximization :
  HotelRevenueProblem 100 400 50 20 5 := by
  sorry

#check hotel_revenue_maximization

end NUMINAMATH_CALUDE_hotel_revenue_maximization_l1141_114179


namespace NUMINAMATH_CALUDE_blue_to_red_light_ratio_l1141_114123

/-- Proves that the ratio of blue lights to red lights is 3:1 given the problem conditions -/
theorem blue_to_red_light_ratio :
  let initial_white_lights : ℕ := 59
  let red_lights : ℕ := 12
  let green_lights : ℕ := 6
  let remaining_to_buy : ℕ := 5
  let total_colored_lights : ℕ := initial_white_lights - remaining_to_buy
  let blue_lights : ℕ := total_colored_lights - (red_lights + green_lights)
  (blue_lights : ℚ) / red_lights = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_blue_to_red_light_ratio_l1141_114123


namespace NUMINAMATH_CALUDE_zeros_product_greater_than_e_squared_l1141_114146

theorem zeros_product_greater_than_e_squared (k : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
  (Real.log x₁ = k * x₁) → (Real.log x₂ = k * x₂) →
  x₁ * x₂ > Real.exp 2 := by
sorry

end NUMINAMATH_CALUDE_zeros_product_greater_than_e_squared_l1141_114146


namespace NUMINAMATH_CALUDE_pursuer_catches_target_l1141_114197

/-- Represents a point on an infinite straight line --/
structure Point where
  position : ℝ

/-- Represents a moving object on the line --/
structure MovingObject where
  initialPosition : Point
  speed : ℝ
  direction : Bool  -- True for positive direction, False for negative

/-- The pursuer (police car) --/
def pursuer : MovingObject :=
  { initialPosition := { position := 0 },
    speed := 1,
    direction := true }

/-- The target (stolen car) --/
def target : MovingObject :=
  { initialPosition := { position := 0 },  -- Initial position unknown
    speed := 0.9,
    direction := true }  -- Direction unknown

/-- Theorem stating that the pursuer will eventually catch the target --/
theorem pursuer_catches_target :
  ∃ (t : ℝ), t > 0 ∧ 
  (pursuer.initialPosition.position + t * pursuer.speed = 
   target.initialPosition.position + t * target.speed ∨
   pursuer.initialPosition.position - t * pursuer.speed = 
   target.initialPosition.position - t * target.speed) :=
sorry

end NUMINAMATH_CALUDE_pursuer_catches_target_l1141_114197


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l1141_114130

theorem lcm_factor_problem (A B : ℕ+) (X : ℕ) : 
  A = 225 →
  Nat.gcd A B = 15 →
  Nat.lcm A B = 15 * X →
  X = 15 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l1141_114130


namespace NUMINAMATH_CALUDE_knights_in_company_l1141_114148

def is_knight (person : Nat) : Prop := sorry

def statement (n : Nat) : Prop :=
  ∃ k, k ∣ n ∧ (∀ p, is_knight p ↔ p ≤ k)

theorem knights_in_company :
  ∀ k : Nat, k ≤ 39 →
  (∀ n : Nat, n ≤ 39 → (∃ p, is_knight p ↔ statement n)) →
  (k = 0 ∨ k = 6) :=
sorry

end NUMINAMATH_CALUDE_knights_in_company_l1141_114148


namespace NUMINAMATH_CALUDE_race_heartbeats_l1141_114149

/-- Calculates the total number of heartbeats during a race with varying heart rates. -/
def total_heartbeats (base_rate : ℕ) (distance : ℕ) (pace : ℕ) (rate_increase : ℕ) (increase_start : ℕ) : ℕ :=
  let total_time := distance * pace
  let base_beats := base_rate * total_time
  let increased_distance := distance - increase_start
  let increased_beats := increased_distance * (increased_distance + 1) * rate_increase / 2
  base_beats + increased_beats

/-- Theorem stating the total number of heartbeats during a 20-mile race 
    with specific heart rate conditions. -/
theorem race_heartbeats : 
  total_heartbeats 160 20 6 5 10 = 11475 :=
sorry

end NUMINAMATH_CALUDE_race_heartbeats_l1141_114149


namespace NUMINAMATH_CALUDE_smallest_base_for_80_l1141_114139

theorem smallest_base_for_80 :
  ∀ b : ℕ, b ≥ 5 → b^2 ≤ 80 ∧ 80 < b^3 →
  ∀ c : ℕ, c < 5 → ¬(c^2 ≤ 80 ∧ 80 < c^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_80_l1141_114139


namespace NUMINAMATH_CALUDE_four_projectors_illuminate_plane_l1141_114100

/-- Represents a point on a plane with a projector --/
structure ProjectorPoint where
  x : ℝ
  y : ℝ
  direction : Nat -- 0: North, 1: East, 2: South, 3: West

/-- Represents the illuminated area by a projector --/
def illuminatedArea (p : ProjectorPoint) : Set (ℝ × ℝ) :=
  sorry

/-- The entire plane --/
def entirePlane : Set (ℝ × ℝ) :=
  sorry

/-- Theorem stating that four projector points can illuminate the entire plane --/
theorem four_projectors_illuminate_plane (p1 p2 p3 p4 : ProjectorPoint) :
  ∃ (d1 d2 d3 d4 : Nat), 
    (d1 < 4 ∧ d2 < 4 ∧ d3 < 4 ∧ d4 < 4) ∧
    (illuminatedArea {p1 with direction := d1} ∪ 
     illuminatedArea {p2 with direction := d2} ∪
     illuminatedArea {p3 with direction := d3} ∪
     illuminatedArea {p4 with direction := d4}) = entirePlane :=
  sorry

end NUMINAMATH_CALUDE_four_projectors_illuminate_plane_l1141_114100


namespace NUMINAMATH_CALUDE_sarah_car_robots_l1141_114142

/-- Prove that Sarah has 125 car robots given the conditions of the problem -/
theorem sarah_car_robots :
  ∀ (tom michael bob sarah : ℕ),
  tom = 15 →
  michael = 2 * tom →
  bob = 8 * michael →
  sarah = (bob / 2) + 5 →
  sarah = 125 := by
sorry

end NUMINAMATH_CALUDE_sarah_car_robots_l1141_114142


namespace NUMINAMATH_CALUDE_linear_equation_and_absolute_value_l1141_114138

theorem linear_equation_and_absolute_value (m a : ℝ) :
  (∀ x, (m^2 - 9) * x^2 - (m - 3) * x + 6 = 0 → (m^2 - 9 = 0 ∧ m - 3 ≠ 0)) →
  |a| ≤ |m| →
  |a + m| + |a - m| = 6 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_and_absolute_value_l1141_114138


namespace NUMINAMATH_CALUDE_square_sum_theorem_l1141_114158

theorem square_sum_theorem (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y + x + y = 11) : 
  x^2 + y^2 = 2893/36 := by
sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l1141_114158


namespace NUMINAMATH_CALUDE_smallest_max_sum_l1141_114184

theorem smallest_max_sum (p q r s t : ℕ+) 
  (sum_eq : p + q + r + s + t = 3015) :
  let N := max (p + q) (max (q + r) (max (r + s) (s + t)))
  ∃ (min_N : ℕ), 
    (∀ (p' q' r' s' t' : ℕ+), 
      p' + q' + r' + s' + t' = 3015 → 
      max (p' + q') (max (q' + r') (max (r' + s') (s' + t'))) ≥ min_N) ∧
    N = min_N ∧
    min_N = 1508 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l1141_114184


namespace NUMINAMATH_CALUDE_abs_diff_eq_diff_implies_leq_l1141_114132

theorem abs_diff_eq_diff_implies_leq (x y : ℝ) : |x - y| = y - x → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_eq_diff_implies_leq_l1141_114132


namespace NUMINAMATH_CALUDE_cars_meet_time_l1141_114157

-- Define the highway length
def highway_length : ℝ := 175

-- Define the speeds of the two cars
def speed_car1 : ℝ := 25
def speed_car2 : ℝ := 45

-- Define the meeting time
def meeting_time : ℝ := 2.5

-- Theorem statement
theorem cars_meet_time :
  speed_car1 * meeting_time + speed_car2 * meeting_time = highway_length :=
by sorry


end NUMINAMATH_CALUDE_cars_meet_time_l1141_114157


namespace NUMINAMATH_CALUDE_smallest_divisible_by_4_13_7_l1141_114183

theorem smallest_divisible_by_4_13_7 : ∀ n : ℕ, n > 0 ∧ 4 ∣ n ∧ 13 ∣ n ∧ 7 ∣ n → n ≥ 364 := by
  sorry

#check smallest_divisible_by_4_13_7

end NUMINAMATH_CALUDE_smallest_divisible_by_4_13_7_l1141_114183


namespace NUMINAMATH_CALUDE_carol_first_six_prob_l1141_114161

/-- The probability of rolling a number other than 6 on a fair six-sided die. -/
def prob_not_six : ℚ := 5/6

/-- The probability of rolling a 6 on a fair six-sided die. -/
def prob_six : ℚ := 1/6

/-- The number of players before Carol. -/
def players_before_carol : ℕ := 2

/-- The total number of players. -/
def total_players : ℕ := 4

/-- The probability that Carol is the first to roll a six in the dice game. -/
theorem carol_first_six_prob : 
  (prob_not_six^players_before_carol * prob_six) / (1 - prob_not_six^total_players) = 125/671 := by
  sorry

end NUMINAMATH_CALUDE_carol_first_six_prob_l1141_114161


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1141_114110

theorem solution_set_inequality (a b : ℝ) : 
  (∀ x, ax^2 - b*x - 1 ≥ 0 ↔ x ∈ Set.Icc (-1/2) (-1/3)) → 
  (∀ x, ax^2 - b*x - 1 < 0 ↔ x ∈ Set.Ioo 2 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1141_114110


namespace NUMINAMATH_CALUDE_perpendicular_iff_a_eq_pm_one_l1141_114168

def line1 (a x y : ℝ) : Prop := a * x + y + 2 = 0
def line2 (a x y : ℝ) : Prop := a * x - y + 4 = 0

def perpendicular (a : ℝ) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ), line1 a x1 y1 → line2 a x2 y2 →
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) *
    ((x2 - x1) * (a * (x2 - x1)) + (y2 - y1) * (-(y2 - y1))) = 0

theorem perpendicular_iff_a_eq_pm_one :
  ∀ a : ℝ, perpendicular a ↔ (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_iff_a_eq_pm_one_l1141_114168


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1141_114159

theorem sqrt_product_simplification : Real.sqrt 18 * Real.sqrt 72 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1141_114159


namespace NUMINAMATH_CALUDE_total_payment_for_bikes_l1141_114102

-- Define the payment for painting a bike
def paint_payment : ℕ := 5

-- Define the additional payment for selling a bike
def sell_additional : ℕ := 8

-- Define the number of bikes
def num_bikes : ℕ := 8

-- Theorem to prove
theorem total_payment_for_bikes : 
  (paint_payment + (paint_payment + sell_additional)) * num_bikes = 144 := by
  sorry

end NUMINAMATH_CALUDE_total_payment_for_bikes_l1141_114102


namespace NUMINAMATH_CALUDE_exam_score_theorem_l1141_114167

theorem exam_score_theorem (total_students : ℕ) 
                            (assigned_day_percentage : ℚ) 
                            (makeup_day_percentage : ℚ) 
                            (makeup_day_average : ℚ) 
                            (class_average : ℚ) :
  total_students = 100 →
  assigned_day_percentage = 70 / 100 →
  makeup_day_percentage = 30 / 100 →
  makeup_day_average = 80 / 100 →
  class_average = 66 / 100 →
  ∃ (assigned_day_average : ℚ),
    assigned_day_average = 60 / 100 ∧
    class_average * total_students = 
      (assigned_day_percentage * total_students * assigned_day_average) +
      (makeup_day_percentage * total_students * makeup_day_average) :=
by sorry

end NUMINAMATH_CALUDE_exam_score_theorem_l1141_114167


namespace NUMINAMATH_CALUDE_probability_of_one_each_l1141_114199

def shirts : ℕ := 6
def shorts : ℕ := 8
def socks : ℕ := 7
def hats : ℕ := 3

def total_items : ℕ := shirts + shorts + socks + hats

theorem probability_of_one_each : 
  (shirts.choose 1 * shorts.choose 1 * socks.choose 1 * hats.choose 1) / total_items.choose 4 = 72 / 91 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_one_each_l1141_114199


namespace NUMINAMATH_CALUDE_consecutive_integers_transformation_l1141_114107

/-- Sum of squares of first m positive integers -/
def sum_of_squares (m : ℕ) : ℕ := m * (m + 1) * (2 * m + 1) / 6

/-- Sum of squares of 2n consecutive integers starting from k -/
def consecutive_sum_of_squares (n k : ℕ) : ℕ :=
  sum_of_squares (k + 2*n - 1) - sum_of_squares (k - 1)

theorem consecutive_integers_transformation (n : ℕ) :
  ∀ k m : ℕ, ∃ t : ℕ, 2^t * consecutive_sum_of_squares n 1 ≠ consecutive_sum_of_squares n k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_transformation_l1141_114107


namespace NUMINAMATH_CALUDE_sqrt_65_minus_1_bound_l1141_114120

theorem sqrt_65_minus_1_bound (n : ℕ) (hn : 0 < n) :
  (n : ℝ) < Real.sqrt 65 - 1 ∧ Real.sqrt 65 - 1 < (n : ℝ) + 1 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_65_minus_1_bound_l1141_114120


namespace NUMINAMATH_CALUDE_complex_point_in_third_quadrant_l1141_114165

/-- Given a complex number z = 1 + i, prove that the real and imaginary parts of (5/z^2) - z are both negative -/
theorem complex_point_in_third_quadrant (z : ℂ) (h : z = 1 + Complex.I) :
  (Complex.re ((5 / z^2) - z) < 0) ∧ (Complex.im ((5 / z^2) - z) < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_point_in_third_quadrant_l1141_114165


namespace NUMINAMATH_CALUDE_math_exam_participants_l1141_114186

theorem math_exam_participants (sample_size : ℕ) (probability : ℝ) (total_students : ℕ) : 
  sample_size = 50 → 
  probability = 0.1 → 
  (sample_size : ℝ) / probability = total_students →
  total_students = 500 := by
sorry

end NUMINAMATH_CALUDE_math_exam_participants_l1141_114186


namespace NUMINAMATH_CALUDE_trapezoid_circle_properties_l1141_114125

/-- Represents a trapezoid ABCD with a circle centered at P on AB and tangent to BC and AD -/
structure Trapezoid :=
  (AB CD BC AD : ℝ)
  (AP : ℝ)
  (r : ℝ)

/-- The theorem stating the properties of the trapezoid and circle -/
theorem trapezoid_circle_properties (T : Trapezoid) :
  T.AB = 105 ∧
  T.BC = 65 ∧
  T.CD = 27 ∧
  T.AD = 80 ∧
  T.AP = 175 / 3 ∧
  T.r = 35 / 6 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_circle_properties_l1141_114125


namespace NUMINAMATH_CALUDE_jim_sara_savings_equality_l1141_114155

/-- Proves that Jim and Sara will have saved the same amount after 820 weeks -/
theorem jim_sara_savings_equality :
  let sara_initial : ℕ := 4100
  let sara_weekly : ℕ := 10
  let jim_weekly : ℕ := 15
  let weeks : ℕ := 820
  sara_initial + sara_weekly * weeks = jim_weekly * weeks :=
by sorry

end NUMINAMATH_CALUDE_jim_sara_savings_equality_l1141_114155


namespace NUMINAMATH_CALUDE_serezha_puts_more_berries_l1141_114166

/-- Represents the berry picking scenario -/
structure BerryPicking where
  total_berries : ℕ
  serezha_rate : ℕ → ℕ  -- Function representing Serezha's picking pattern
  dima_rate : ℕ → ℕ     -- Function representing Dima's picking pattern
  serezha_speed : ℕ
  dima_speed : ℕ

/-- The specific berry picking scenario from the problem -/
def berry_scenario : BerryPicking :=
  { total_berries := 450
  , serezha_rate := λ n => n / 2  -- 1 out of every 2
  , dima_rate := λ n => 2 * n / 3 -- 2 out of every 3
  , serezha_speed := 2
  , dima_speed := 1 }

/-- Theorem stating the difference in berries put in basket -/
theorem serezha_puts_more_berries (bp : BerryPicking) (h : bp = berry_scenario) :
  ∃ (s d : ℕ), s = bp.serezha_rate (bp.serezha_speed * bp.total_berries / (bp.serezha_speed + bp.dima_speed)) ∧
                d = bp.dima_rate (bp.dima_speed * bp.total_berries / (bp.serezha_speed + bp.dima_speed)) ∧
                s - d = 50 := by
  sorry


end NUMINAMATH_CALUDE_serezha_puts_more_berries_l1141_114166


namespace NUMINAMATH_CALUDE_solve_equation_l1141_114174

theorem solve_equation (x y : ℝ) : y = 1 / (2 * x + 2) → y = 2 → x = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1141_114174


namespace NUMINAMATH_CALUDE_child_height_calculation_l1141_114170

/-- Calculates a child's current height given their previous height and growth. -/
def current_height (previous_height growth : Float) : Float :=
  previous_height + growth

theorem child_height_calculation :
  let previous_height : Float := 38.5
  let growth : Float := 3.0
  current_height previous_height growth = 41.5 := by
  sorry

end NUMINAMATH_CALUDE_child_height_calculation_l1141_114170


namespace NUMINAMATH_CALUDE_cyclist_speed_proof_l1141_114160

/-- The distance between Town X and Town Y in miles -/
def distance : ℝ := 90

/-- The speed difference between cyclists D and C in miles per hour -/
def speed_difference : ℝ := 5

/-- The distance from Town Y where cyclists C and D meet on D's return trip in miles -/
def meeting_point : ℝ := 15

/-- The speed of Cyclist C in miles per hour -/
def speed_C : ℝ := 12.5

theorem cyclist_speed_proof :
  ∃ (speed_D : ℝ),
    speed_D = speed_C + speed_difference ∧
    distance / speed_C = (distance + meeting_point) / speed_D :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_proof_l1141_114160


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l1141_114131

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l1141_114131


namespace NUMINAMATH_CALUDE_focal_radii_common_points_l1141_114109

/-- An ellipse and hyperbola sharing the same foci -/
structure EllipseHyperbola where
  a : ℝ  -- semi-major axis of the ellipse
  e : ℝ  -- semi-major axis of the hyperbola

/-- The focal radii of the common points of an ellipse and hyperbola sharing the same foci -/
def focal_radii (eh : EllipseHyperbola) : ℝ × ℝ :=
  (eh.a + eh.e, eh.a - eh.e)

/-- Theorem: The focal radii of the common points of an ellipse and hyperbola 
    sharing the same foci are a + e and a - e -/
theorem focal_radii_common_points (eh : EllipseHyperbola) :
  focal_radii eh = (eh.a + eh.e, eh.a - eh.e) := by
  sorry

end NUMINAMATH_CALUDE_focal_radii_common_points_l1141_114109


namespace NUMINAMATH_CALUDE_equations_represent_same_curve_l1141_114126

-- Define the two equations
def equation1 (x y : ℝ) : Prop := |y| = |x|
def equation2 (x y : ℝ) : Prop := y^2 = x^2

-- Theorem statement
theorem equations_represent_same_curve :
  ∀ (x y : ℝ), equation1 x y ↔ equation2 x y := by
  sorry

end NUMINAMATH_CALUDE_equations_represent_same_curve_l1141_114126


namespace NUMINAMATH_CALUDE_periodic_function_l1141_114178

def is_periodic (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, f (x + c) = f x

theorem periodic_function (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, |f x| ≤ 1)
  (h2 : ∀ x : ℝ, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  is_periodic f :=
sorry

end NUMINAMATH_CALUDE_periodic_function_l1141_114178


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_for_given_solution_set_l1141_114143

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Theorem for part (I)
theorem solution_set_when_a_is_one (x : ℝ) :
  (f 1 x ≥ 3 * x + 2) ↔ (x ≥ 3 ∨ x ≤ -1) :=
sorry

-- Theorem for part (II)
theorem a_value_for_given_solution_set (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, f a x ≤ 0 ↔ x ≤ -1) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_for_given_solution_set_l1141_114143


namespace NUMINAMATH_CALUDE_flu_probability_l1141_114191

/-- The probability of a randomly selected person having the flu given the flu rates and population ratios for three areas -/
theorem flu_probability (flu_rate_A flu_rate_B flu_rate_C : ℝ) 
  (pop_ratio_A pop_ratio_B pop_ratio_C : ℕ) : 
  flu_rate_A = 0.06 →
  flu_rate_B = 0.05 →
  flu_rate_C = 0.04 →
  pop_ratio_A = 6 →
  pop_ratio_B = 5 →
  pop_ratio_C = 4 →
  (flu_rate_A * pop_ratio_A + flu_rate_B * pop_ratio_B + flu_rate_C * pop_ratio_C) / 
  (pop_ratio_A + pop_ratio_B + pop_ratio_C) = 77 / 1500 := by
sorry


end NUMINAMATH_CALUDE_flu_probability_l1141_114191


namespace NUMINAMATH_CALUDE_orange_count_orange_count_problem_l1141_114195

theorem orange_count (initial_apples : ℕ) (removed_oranges : ℕ) 
  (apple_percentage : ℚ) (initial_oranges : ℕ) : Prop :=
  initial_apples = 14 →
  removed_oranges = 20 →
  apple_percentage = 7/10 →
  initial_apples / (initial_apples + initial_oranges - removed_oranges) = apple_percentage →
  initial_oranges = 26

/-- The theorem states that given the conditions from the problem,
    the initial number of oranges in the box is 26. -/
theorem orange_count_problem : 
  ∃ (initial_oranges : ℕ), orange_count 14 20 (7/10) initial_oranges :=
sorry

end NUMINAMATH_CALUDE_orange_count_orange_count_problem_l1141_114195


namespace NUMINAMATH_CALUDE_distance_after_seven_seconds_l1141_114152

/-- The distance fallen by a freely falling body after t seconds -/
def distance_fallen (t : ℝ) : ℝ := 4.9 * t^2

/-- The time difference between the start of the two falling bodies -/
def time_difference : ℝ := 5

/-- The distance between the two falling bodies after t seconds -/
def distance_between (t : ℝ) : ℝ :=
  distance_fallen t - distance_fallen (t - time_difference)

/-- Theorem: The distance between the two falling bodies is 220.5 meters after 7 seconds -/
theorem distance_after_seven_seconds :
  distance_between 7 = 220.5 := by sorry

end NUMINAMATH_CALUDE_distance_after_seven_seconds_l1141_114152


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l1141_114188

theorem fraction_sum_equals_one (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h : x - y + z = x * y * z) : 1 / x - 1 / y + 1 / z = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l1141_114188


namespace NUMINAMATH_CALUDE_exchange_of_segments_is_structure_variation_l1141_114124

-- Define the basic concepts
def ChromosomalVariation : Type := sorry
def NonHomologousChromosome : Type := sorry
def ChromosomeStructure : Type := sorry
def Translocation : Type := sorry

-- Define the exchange of partial segments between non-homologous chromosomes
def PartialSegmentExchange (c1 c2 : NonHomologousChromosome) : Translocation := sorry

-- Define what constitutes a variation in chromosome structure
def IsChromosomeStructureVariation (t : Translocation) : Prop := sorry

-- Theorem to prove
theorem exchange_of_segments_is_structure_variation 
  (c1 c2 : NonHomologousChromosome) : 
  IsChromosomeStructureVariation (PartialSegmentExchange c1 c2) := by
  sorry

end NUMINAMATH_CALUDE_exchange_of_segments_is_structure_variation_l1141_114124


namespace NUMINAMATH_CALUDE_specific_journey_distance_l1141_114147

/-- A journey with two parts at different speeds -/
structure Journey where
  total_time : ℝ
  first_part_time : ℝ
  first_part_speed : ℝ
  second_part_speed : ℝ
  (first_part_time_valid : first_part_time > 0 ∧ first_part_time < total_time)
  (speeds_positive : first_part_speed > 0 ∧ second_part_speed > 0)

/-- Calculate the total distance of a journey -/
def total_distance (j : Journey) : ℝ :=
  j.first_part_speed * j.first_part_time + 
  j.second_part_speed * (j.total_time - j.first_part_time)

/-- The specific journey described in the problem -/
def specific_journey : Journey where
  total_time := 8
  first_part_time := 4
  first_part_speed := 4
  second_part_speed := 2
  first_part_time_valid := by sorry
  speeds_positive := by sorry

/-- Theorem stating that the total distance of the specific journey is 24 km -/
theorem specific_journey_distance : 
  total_distance specific_journey = 24 := by sorry

end NUMINAMATH_CALUDE_specific_journey_distance_l1141_114147


namespace NUMINAMATH_CALUDE_cubic_function_monotonicity_l1141_114192

/-- A cubic function with a parameter b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + x

/-- The derivative of f with respect to x -/
def f' (b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*b*x + 1

/-- The number of monotonic intervals of f -/
def monotonic_intervals (b : ℝ) : ℕ := sorry

theorem cubic_function_monotonicity (b : ℝ) :
  monotonic_intervals b = 3 → b ∈ Set.Iic (-Real.sqrt 3) ∪ Set.Ioi (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_monotonicity_l1141_114192


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_72_l1141_114115

theorem sqrt_18_times_sqrt_72 : Real.sqrt 18 * Real.sqrt 72 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_72_l1141_114115


namespace NUMINAMATH_CALUDE_election_percentage_l1141_114190

theorem election_percentage (total_members votes_cast : ℕ) 
  (percentage_of_total : ℚ) (h1 : total_members = 1600) 
  (h2 : votes_cast = 525) (h3 : percentage_of_total = 19.6875 / 100) : 
  (percentage_of_total * total_members) / votes_cast = 60 / 100 := by
  sorry

end NUMINAMATH_CALUDE_election_percentage_l1141_114190


namespace NUMINAMATH_CALUDE_sum_234_142_in_base4_l1141_114173

/-- Converts a natural number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list of digits represents a valid base 4 number -/
def isValidBase4 (digits : List ℕ) : Prop :=
  sorry

theorem sum_234_142_in_base4 :
  let sum := 234 + 142
  let base4Sum := toBase4 sum
  isValidBase4 base4Sum ∧ base4Sum = [1, 1, 0, 3, 0] := by
  sorry

end NUMINAMATH_CALUDE_sum_234_142_in_base4_l1141_114173


namespace NUMINAMATH_CALUDE_jills_first_bus_ride_time_l1141_114175

/-- Jill's journey to the library -/
def jills_journey (first_bus_wait : ℕ) (second_bus_ride : ℕ) (first_bus_ride : ℕ) : Prop :=
  second_bus_ride = (first_bus_wait + first_bus_ride) / 2

theorem jills_first_bus_ride_time :
  ∃ (first_bus_ride : ℕ),
    jills_journey 12 21 first_bus_ride ∧
    first_bus_ride = 30 := by
  sorry

end NUMINAMATH_CALUDE_jills_first_bus_ride_time_l1141_114175


namespace NUMINAMATH_CALUDE_basketball_team_starters_l1141_114198

theorem basketball_team_starters (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) (quad_starters : ℕ) :
  total_players = 12 →
  quadruplets = 4 →
  starters = 5 →
  quad_starters = 2 →
  (Nat.choose quadruplets quad_starters) * (Nat.choose (total_players - quadruplets) (starters - quad_starters)) = 336 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_starters_l1141_114198


namespace NUMINAMATH_CALUDE_square_perimeter_l1141_114129

theorem square_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 520 → perimeter = 8 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1141_114129


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1141_114119

theorem fixed_point_on_line (m : ℝ) : 
  let x : ℝ := -1
  let y : ℝ := -1/2
  (m^2 + 6*m + 3) * x - (2*m^2 + 18*m + 2) * y - 3*m + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1141_114119


namespace NUMINAMATH_CALUDE_report_card_recess_num_of_ds_l1141_114193

/-- Calculates the number of Ds on report cards given the recess rules and grades --/
theorem report_card_recess (normal_recess : ℕ) (a_bonus : ℕ) (b_bonus : ℕ) (d_penalty : ℕ)
  (num_a : ℕ) (num_b : ℕ) (num_c : ℕ) (total_recess : ℕ) : ℕ :=
  let extra_time := num_a * a_bonus + num_b * b_bonus
  let expected_time := normal_recess + extra_time
  let reduced_time := expected_time - total_recess
  reduced_time / d_penalty

/-- Proves that there are 5 Ds on the report cards --/
theorem num_of_ds : report_card_recess 20 2 1 1 10 12 14 47 = 5 := by
  sorry

end NUMINAMATH_CALUDE_report_card_recess_num_of_ds_l1141_114193


namespace NUMINAMATH_CALUDE_average_of_xyz_l1141_114103

theorem average_of_xyz (x y z : ℝ) : 
  x = 3 → y = 2 * x → z = 3 * y → (x + y + z) / 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_of_xyz_l1141_114103


namespace NUMINAMATH_CALUDE_school_population_l1141_114133

theorem school_population (b g t : ℕ) : 
  b = 3 * g → g = 9 * t → b + g + t = (37 * b) / 27 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l1141_114133


namespace NUMINAMATH_CALUDE_total_hot_dogs_is_fifteen_l1141_114150

/-- Represents the number of hot dogs served at each meal -/
structure HotDogMeals where
  breakfast : ℕ
  lunch : ℕ
  dinner : ℕ

/-- Proves that the total number of hot dogs served is 15 given the conditions -/
theorem total_hot_dogs_is_fifteen (h : HotDogMeals) :
  (h.breakfast = 2 * h.dinner) →
  (h.lunch = 9) →
  (h.lunch = h.breakfast + h.dinner + 3) →
  (h.breakfast + h.lunch + h.dinner = 15) := by
  sorry

end NUMINAMATH_CALUDE_total_hot_dogs_is_fifteen_l1141_114150


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1141_114122

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, 3)

theorem perpendicular_vectors (t : ℝ) : 
  (a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) → t = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1141_114122


namespace NUMINAMATH_CALUDE_liam_target_time_l1141_114121

-- Define Mia's run
def mia_distance : ℕ := 5
def mia_time : ℕ := 45

-- Define Liam's initial run
def liam_initial_distance : ℕ := 3

-- Define the relationship between Liam and Mia's times
def liam_initial_time : ℚ := mia_time / 3

-- Define Liam's target distance
def liam_target_distance : ℕ := 7

-- Theorem to prove
theorem liam_target_time : 
  (liam_target_distance : ℚ) * (liam_initial_time / liam_initial_distance) = 35 := by
  sorry

end NUMINAMATH_CALUDE_liam_target_time_l1141_114121


namespace NUMINAMATH_CALUDE_base_difference_theorem_l1141_114140

/-- Converts a number from base 5 to base 10 -/
def base5_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

theorem base_difference_theorem :
  let n1 := 543210
  let n2 := 43210
  (base5_to_base10 n1) - (base8_to_base10 n2) = 499 := by sorry

end NUMINAMATH_CALUDE_base_difference_theorem_l1141_114140


namespace NUMINAMATH_CALUDE_sqrt_twelve_over_sqrt_two_equals_sqrt_six_l1141_114169

theorem sqrt_twelve_over_sqrt_two_equals_sqrt_six : 
  (Real.sqrt 12) / (Real.sqrt 2) = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_over_sqrt_two_equals_sqrt_six_l1141_114169


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l1141_114185

theorem quadratic_root_relation (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ = 4 * x₁ ∧ x₁^2 + a*x₁ + 2*a = 0 ∧ x₂^2 + a*x₂ + 2*a = 0) → 
  a = 25/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l1141_114185


namespace NUMINAMATH_CALUDE_marks_buttons_l1141_114105

theorem marks_buttons (x : ℕ) : 
  (x + 3*x) / 2 = 28 → x = 14 := by
  sorry

end NUMINAMATH_CALUDE_marks_buttons_l1141_114105


namespace NUMINAMATH_CALUDE_dot_path_length_on_rotating_cube_l1141_114118

/-- The path length of a dot on a rotating cube -/
theorem dot_path_length_on_rotating_cube (cube_edge : ℝ) (h_edge : cube_edge = 2) :
  let dot_radius : ℝ := cube_edge / 2
  let path_length : ℝ := 2 * Real.pi * dot_radius
  path_length = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_dot_path_length_on_rotating_cube_l1141_114118


namespace NUMINAMATH_CALUDE_emma_savings_l1141_114151

theorem emma_savings (initial_savings withdrawal deposit final_savings : ℕ) : 
  initial_savings = 230 →
  final_savings = 290 →
  deposit = 2 * withdrawal →
  final_savings = initial_savings - withdrawal + deposit →
  withdrawal = 60 := by
sorry

end NUMINAMATH_CALUDE_emma_savings_l1141_114151


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l1141_114137

theorem smaller_circle_radius (R : ℝ) (r : ℝ) : 
  R = 2 →  -- Larger circle radius
  0 < r → r < R →  -- Smaller circle is inside the larger circle
  ∃ (k : ℝ), k > 0 ∧ π * R^2 - π * r^2 = π * r^2 - k →  -- Areas form an arithmetic progression
  π * R^2 ≥ π * r^2 →  -- Larger circle has the largest area
  r = Real.sqrt 2 :=  -- Radius of smaller circle is √2
by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l1141_114137


namespace NUMINAMATH_CALUDE_original_price_calculation_l1141_114163

/-- Proves that given an article sold at a 30% profit with a selling price of 715, 
    the original price (cost price) of the article is 550. -/
theorem original_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
    (h1 : selling_price = 715)
    (h2 : profit_percentage = 30) : 
  ∃ (original_price : ℝ), 
    selling_price = original_price * (1 + profit_percentage / 100) ∧ 
    original_price = 550 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l1141_114163


namespace NUMINAMATH_CALUDE_jack_email_difference_l1141_114172

theorem jack_email_difference : 
  let morning_emails : ℕ := 6
  let afternoon_emails : ℕ := 2
  morning_emails - afternoon_emails = 4 :=
by sorry

end NUMINAMATH_CALUDE_jack_email_difference_l1141_114172


namespace NUMINAMATH_CALUDE_intersection_complement_eq_l1141_114156

def M : Set ℝ := {-1, 0, 1, 3}
def N : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_complement_eq : M ∩ (Set.univ \ N) = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_eq_l1141_114156


namespace NUMINAMATH_CALUDE_geometric_sequence_roots_l1141_114112

theorem geometric_sequence_roots (m n : ℝ) : 
  (∃ a b c d : ℝ, 
    (a^2 - m*a + 2) * (a^2 - n*a + 2) = 0 ∧
    (b^2 - m*b + 2) * (b^2 - n*b + 2) = 0 ∧
    (c^2 - m*c + 2) * (c^2 - n*c + 2) = 0 ∧
    (d^2 - m*d + 2) * (d^2 - n*d + 2) = 0 ∧
    a = 1/2 ∧
    ∃ r : ℝ, b = a*r ∧ c = b*r ∧ d = c*r) →
  |m - n| = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_roots_l1141_114112


namespace NUMINAMATH_CALUDE_two_from_four_combination_l1141_114116

theorem two_from_four_combination : Nat.choose 4 2 = 6 := by sorry

end NUMINAMATH_CALUDE_two_from_four_combination_l1141_114116


namespace NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l1141_114181

theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) :
  cube_edge = 5 →
  pyramid_base = 10 →
  cube_edge ^ 3 = (1 / 3) * pyramid_base ^ 2 * pyramid_height →
  pyramid_height = 3.75 := by
sorry

end NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l1141_114181


namespace NUMINAMATH_CALUDE_calculate_expression_l1141_114117

theorem calculate_expression : ((-2 : ℤ)^2 : ℝ) - |(-5 : ℤ)| - Real.sqrt 144 = -13 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1141_114117


namespace NUMINAMATH_CALUDE_time_difference_l1141_114189

-- Define constants
def blocks : ℕ := 12
def walk_time_per_block : ℕ := 1  -- in minutes
def bike_time_per_block : ℕ := 20 -- in seconds

-- Define functions
def walk_time : ℕ := blocks * walk_time_per_block

def bike_time_seconds : ℕ := blocks * bike_time_per_block
def bike_time : ℕ := bike_time_seconds / 60

-- Theorem
theorem time_difference : walk_time - bike_time = 8 := by
  sorry


end NUMINAMATH_CALUDE_time_difference_l1141_114189


namespace NUMINAMATH_CALUDE_parallel_condition_perpendicular_condition_l1141_114153

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x + (m + 3) * y - 5 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := 6 * x + (2 * m - 1) * y = 5

-- Define parallel and perpendicular conditions
def parallel (m : ℝ) : Prop := ∀ x y, l₁ m x y ↔ l₂ m x y
def perpendicular (m : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂, 
  l₁ m x₁ y₁ ∧ l₂ m x₂ y₂ → (x₁ - x₂) * (x₁ - x₂) + (y₁ - y₂) * (y₁ - y₂) = 0

-- Theorem for parallel lines
theorem parallel_condition : 
  ∀ m : ℝ, parallel m ↔ m = -5/2 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_condition : 
  ∀ m : ℝ, perpendicular m ↔ (m = -1 ∨ m = -9/2) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_perpendicular_condition_l1141_114153


namespace NUMINAMATH_CALUDE_max_area_triangle_OPQ_l1141_114162

/-- The maximum area of triangle OPQ given the specified conditions -/
theorem max_area_triangle_OPQ :
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (2, 0)
  let O : ℝ × ℝ := (0, 0)
  ∃ (M P Q : ℝ × ℝ),
    (M.1 ≠ -2 ∧ M.1 ≠ 2) →  -- M is not on the same vertical line as A or B
    (M.2 / (M.1 + 2)) * (M.2 / (M.1 - 2)) = -3/4 →  -- Product of slopes AM and BM
    (P.2 - Q.2) / (P.1 - Q.1) = 1 →  -- PQ has slope 1
    (M.1^2 / 4 + M.2^2 / 3 = 1) →  -- M is on the locus
    (P.1^2 / 4 + P.2^2 / 3 = 1) →  -- P is on the locus
    (Q.1^2 / 4 + Q.2^2 / 3 = 1) →  -- Q is on the locus
    (∀ R : ℝ × ℝ, R.1^2 / 4 + R.2^2 / 3 = 1 →  -- For all points R on the locus
      abs ((P.1 - O.1) * (Q.2 - O.2) - (Q.1 - O.1) * (P.2 - O.2)) / 2 ≥
      abs ((P.1 - O.1) * (R.2 - O.2) - (R.1 - O.1) * (P.2 - O.2)) / 2) →
    abs ((P.1 - O.1) * (Q.2 - O.2) - (Q.1 - O.1) * (P.2 - O.2)) / 2 = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_OPQ_l1141_114162


namespace NUMINAMATH_CALUDE_june_found_17_eggs_l1141_114154

/-- The total number of bird eggs June found -/
def total_eggs (tree1_nests tree1_eggs_per_nest tree2_eggs frontyard_eggs : ℕ) : ℕ :=
  tree1_nests * tree1_eggs_per_nest + tree2_eggs + frontyard_eggs

/-- Theorem stating that June found 17 eggs in total -/
theorem june_found_17_eggs : 
  total_eggs 2 5 3 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_june_found_17_eggs_l1141_114154


namespace NUMINAMATH_CALUDE_no_integer_solution_l1141_114134

/-- The equation x^3 - 3xy^2 + y^3 = 2891 has no integer solutions -/
theorem no_integer_solution :
  ¬ ∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = 2891 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1141_114134


namespace NUMINAMATH_CALUDE_necessary_condition_equality_l1141_114144

theorem necessary_condition_equality (a b c : ℝ) : a = b → a * c = b * c := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_equality_l1141_114144


namespace NUMINAMATH_CALUDE_expression_evaluation_l1141_114114

theorem expression_evaluation :
  let x : ℤ := -1
  (x + 2) * (x - 2) - (x - 1)^2 = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1141_114114


namespace NUMINAMATH_CALUDE_action_figure_price_l1141_114182

/-- Given the cost of sneakers, initial savings, number of action figures sold, and money left after purchase, 
    prove the price per action figure. -/
theorem action_figure_price 
  (sneaker_cost : ℕ) 
  (initial_savings : ℕ) 
  (figures_sold : ℕ) 
  (money_left : ℕ) 
  (h1 : sneaker_cost = 90)
  (h2 : initial_savings = 15)
  (h3 : figures_sold = 10)
  (h4 : money_left = 25) :
  (sneaker_cost - initial_savings + money_left) / figures_sold = 10 := by
  sorry

end NUMINAMATH_CALUDE_action_figure_price_l1141_114182


namespace NUMINAMATH_CALUDE_min_distance_parabola_perpendicular_l1141_114187

/-- The minimum distance between a point on y = x² and the intersection of a perpendicular line with the curve -/
theorem min_distance_parabola_perpendicular : 
  let f : ℝ → ℝ := λ x => x^2
  let P : ℝ → ℝ × ℝ := λ x₀ => (x₀, f x₀)
  let tangent_slope : ℝ → ℝ := λ x₀ => 2 * x₀
  let perpendicular_line : ℝ → (ℝ → ℝ) := λ x₀ => λ x => 
    f x₀ - (1 / tangent_slope x₀) * (x - x₀)
  let Q : ℝ → ℝ × ℝ := λ x₀ => 
    let x₁ := -1 / (2 * x₀) - x₀
    (x₁, f x₁)
  let distance : ℝ → ℝ := λ x₀ => 
    Real.sqrt ((Q x₀).1 - (P x₀).1)^2 + ((Q x₀).2 - (P x₀).2)^2
  ∀ x₀ : ℝ, x₀ ≠ 0 → distance x₀ ≥ 3 * Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_distance_parabola_perpendicular_l1141_114187


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1141_114135

-- Define the polynomial
def f (x : ℝ) : ℝ := 9*x^3 - 5*x^2 - 48*x + 54

-- Define divisibility by (x - p)^2
def is_divisible_by_square (p : ℝ) : Prop :=
  ∃ (q : ℝ → ℝ), ∀ x, f x = (x - p)^2 * q x

-- Theorem statement
theorem polynomial_divisibility :
  ∀ p : ℝ, is_divisible_by_square p → p = 8/3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1141_114135


namespace NUMINAMATH_CALUDE_triangle_properties_l1141_114101

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.a * Real.cos t.B + t.b * Real.cos t.A = t.c / (2 * Real.cos t.C) ∧
  t.c = 6 ∧
  2 * Real.sqrt 3 = t.c * Real.sin t.C / 2

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.C = π / 3 ∧ t.a + t.b + t.c = 6 * Real.sqrt 3 + 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1141_114101


namespace NUMINAMATH_CALUDE_craig_travel_difference_l1141_114108

theorem craig_travel_difference : 
  let bus_distance : ℝ := 3.83
  let walk_distance : ℝ := 0.17
  bus_distance - walk_distance = 3.66 := by
  sorry

end NUMINAMATH_CALUDE_craig_travel_difference_l1141_114108


namespace NUMINAMATH_CALUDE_smallest_digit_for_divisibility_l1141_114127

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def number_with_d (d : ℕ) : ℕ := 437000 + d * 1000 + 3

theorem smallest_digit_for_divisibility :
  ∃ (d : ℕ), d < 10 ∧ 
    is_divisible_by_9 (number_with_d d) ∧ 
    (∀ (d' : ℕ), d' < d → ¬is_divisible_by_9 (number_with_d d')) ∧
    d = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_digit_for_divisibility_l1141_114127


namespace NUMINAMATH_CALUDE_max_integer_difference_l1141_114176

theorem max_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 10) :
  ∃ (n : ℤ), n = ⌊y - x⌋ ∧ n ≤ 4 ∧ ∀ (m : ℤ), m = ⌊y - x⌋ → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_max_integer_difference_l1141_114176


namespace NUMINAMATH_CALUDE_right_triangle_sin_A_l1141_114164

theorem right_triangle_sin_A (A B C : Real) (AB BC : Real) :
  -- Right triangle ABC with ∠BAC = 90°
  A + B + C = 180 →
  A = 90 →
  -- Side lengths
  AB = 15 →
  BC = 20 →
  -- Definition of sin A in a right triangle
  Real.sin A = Real.sqrt 7 / 4 := by sorry

end NUMINAMATH_CALUDE_right_triangle_sin_A_l1141_114164


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1141_114141

/-- For a regular polygon with an exterior angle of 36°, the number of sides is 10. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 36 → n * exterior_angle = 360 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1141_114141


namespace NUMINAMATH_CALUDE_fifth_term_geometric_progression_l1141_114136

theorem fifth_term_geometric_progression :
  ∀ (b : ℕ → ℝ),
  (∀ n, b (n + 1) = b (n + 2) - b n) →  -- Each term from the second is the difference of adjacent terms
  b 1 = 7 - 3 * Real.sqrt 5 →           -- First term
  (∀ n, b (n + 1) > b n) →              -- Increasing progression
  b 5 = 2 :=                            -- Fifth term is 2
by
  sorry

end NUMINAMATH_CALUDE_fifth_term_geometric_progression_l1141_114136


namespace NUMINAMATH_CALUDE_team_a_win_probability_l1141_114177

theorem team_a_win_probability (p : ℝ) (h : p = 2/3) : 
  p^2 + p^2 * (1 - p) = 20/27 := by
  sorry

end NUMINAMATH_CALUDE_team_a_win_probability_l1141_114177


namespace NUMINAMATH_CALUDE_log2_7_value_l1141_114196

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Given conditions
variable (m n : ℝ)
variable (h1 : lg 5 = m)
variable (h2 : lg 7 = n)

-- Theorem to prove
theorem log2_7_value : Real.log 7 / Real.log 2 = n / (1 - m) := by
  sorry

end NUMINAMATH_CALUDE_log2_7_value_l1141_114196


namespace NUMINAMATH_CALUDE_bisecting_plane_intersects_24_cubes_l1141_114145

/-- Represents a cube composed of unit cubes -/
structure LargeCube where
  side_length : ℕ
  total_cubes : ℕ

/-- Represents a plane that bisects an internal diagonal of a cube -/
structure BisectingPlane where
  cube : LargeCube
  
/-- The number of unit cubes intersected by the bisecting plane -/
def intersected_cubes (plane : BisectingPlane) : ℕ := sorry

/-- Main theorem: A plane bisecting an internal diagonal of a 4x4x4 cube intersects 24 unit cubes -/
theorem bisecting_plane_intersects_24_cubes 
  (cube : LargeCube) 
  (plane : BisectingPlane) 
  (h1 : cube.side_length = 4) 
  (h2 : cube.total_cubes = 64) 
  (h3 : plane.cube = cube) :
  intersected_cubes plane = 24 := by sorry

end NUMINAMATH_CALUDE_bisecting_plane_intersects_24_cubes_l1141_114145


namespace NUMINAMATH_CALUDE_max_late_all_days_l1141_114111

theorem max_late_all_days (total_late : ℕ) (late_monday : ℕ) (late_tuesday : ℕ) (late_wednesday : ℕ)
  (h_total : total_late = 30)
  (h_monday : late_monday = 20)
  (h_tuesday : late_tuesday = 13)
  (h_wednesday : late_wednesday = 7) :
  ∃ (x : ℕ), x ≤ 5 ∧ 
    x ≤ late_monday ∧ 
    x ≤ late_tuesday ∧ 
    x ≤ late_wednesday ∧
    (late_monday - x) + (late_tuesday - x) + (late_wednesday - x) + x ≤ total_late ∧
    ∀ (y : ℕ), y > x → 
      (y > late_monday ∨ y > late_tuesday ∨ y > late_wednesday ∨
       (late_monday - y) + (late_tuesday - y) + (late_wednesday - y) + y > total_late) :=
by sorry

end NUMINAMATH_CALUDE_max_late_all_days_l1141_114111


namespace NUMINAMATH_CALUDE_fenced_area_calculation_l1141_114104

/-- The area of a rectangle with two square cut-outs at opposite corners -/
def fencedArea (length width cutout1 cutout2 : ℝ) : ℝ :=
  length * width - cutout1^2 - cutout2^2

/-- Theorem stating that the area of the fenced region is 340 square feet -/
theorem fenced_area_calculation :
  fencedArea 20 18 4 2 = 340 := by
  sorry

end NUMINAMATH_CALUDE_fenced_area_calculation_l1141_114104


namespace NUMINAMATH_CALUDE_board_sum_l1141_114171

theorem board_sum : ∀ (numbers : List ℕ),
  (numbers.length = 9) →
  (∀ n ∈ numbers, 1 ≤ n ∧ n ≤ 5) →
  (numbers.filter (λ n => n ≥ 2)).length ≥ 7 →
  (numbers.filter (λ n => n > 2)).length ≥ 6 →
  (numbers.filter (λ n => n ≥ 4)).length ≥ 3 →
  (numbers.filter (λ n => n ≥ 5)).length ≥ 1 →
  numbers.sum = 26 := by
sorry

end NUMINAMATH_CALUDE_board_sum_l1141_114171


namespace NUMINAMATH_CALUDE_min_swaps_at_most_five_l1141_114128

/-- Represents a 4026-digit number composed of ones and twos -/
structure NumberConfig :=
  (ones_count : Nat)
  (twos_count : Nat)
  (total_digits : Nat)
  (h1 : ones_count = 2013)
  (h2 : twos_count = 2013)
  (h3 : total_digits = 4026)
  (h4 : ones_count + twos_count = total_digits)

/-- Represents the state of the number after some swaps -/
structure NumberState :=
  (config : NumberConfig)
  (ones_in_odd : Nat)
  (h : ones_in_odd ≤ config.ones_count)

/-- Checks if a NumberState is divisible by 11 -/
def isDivisibleBy11 (state : NumberState) : Prop :=
  (state.config.total_digits - 2 * state.ones_in_odd) % 11 = 0

/-- The minimum number of swaps required to make the number divisible by 11 -/
def minSwapsToDiv11 (state : NumberState) : Nat :=
  min (state.ones_in_odd % 11) ((11 - state.ones_in_odd % 11) % 11)

/-- The main theorem stating that the minimum number of swaps is at most 5 -/
theorem min_swaps_at_most_five (state : NumberState) : minSwapsToDiv11 state ≤ 5 := by
  sorry


end NUMINAMATH_CALUDE_min_swaps_at_most_five_l1141_114128


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1141_114106

theorem rectangle_ratio (w : ℝ) : 
  w > 0 → 
  2 * w + 2 * 10 = 30 → 
  w / 10 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1141_114106


namespace NUMINAMATH_CALUDE_difference_of_differences_l1141_114113

def arithmetic_sequence (start : ℕ) (diff : ℕ) (length : ℕ) : List ℕ :=
  List.range length |>.map (fun n => start + n * diff)

def common_terms (seq1 seq2 : List ℕ) : List ℕ :=
  seq1.filter (fun x => seq2.contains x)

theorem difference_of_differences
  (start end_val : ℕ)
  (common_count : ℕ)
  (ratio_a ratio_b : ℕ)
  (ha : ratio_a > 0)
  (hb : ratio_b > 0)
  (hcommon : common_count > 0)
  (hend : end_val > start) :
  ∃ (len_a len_b : ℕ),
    let diff_a := ratio_a * (end_val - start) / ((len_a - 1) * ratio_a)
    let diff_b := ratio_b * (end_val - start) / ((len_b - 1) * ratio_b)
    let seq_a := arithmetic_sequence start diff_a len_a
    let seq_b := arithmetic_sequence start diff_b len_b
    (seq_a.length > 0) ∧
    (seq_b.length > 0) ∧
    (seq_a.getLast? = some end_val) ∧
    (seq_b.getLast? = some end_val) ∧
    (common_terms seq_a seq_b).length = common_count ∧
    diff_a - diff_b = 12 :=
by
  sorry

#check difference_of_differences

end NUMINAMATH_CALUDE_difference_of_differences_l1141_114113


namespace NUMINAMATH_CALUDE_circle_radius_l1141_114194

theorem circle_radius (x y : ℝ) : 
  (16 * x^2 - 32 * x + 16 * y^2 + 64 * y + 64 = 0) → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l1141_114194
