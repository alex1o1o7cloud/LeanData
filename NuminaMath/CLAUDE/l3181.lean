import Mathlib

namespace NUMINAMATH_CALUDE_equilateral_hyperbola_equation_l3181_318168

/-- An equilateral hyperbola centered at the origin and passing through (0, 3) -/
structure EquilateralHyperbola where
  /-- The equation of the hyperbola in the form y² - x² = a -/
  equation : ℝ → ℝ → ℝ
  /-- The hyperbola passes through the point (0, 3) -/
  passes_through_point : equation 0 3 = equation 0 3
  /-- The hyperbola is centered at the origin -/
  centered_at_origin : ∀ x y, equation x y = equation (-x) (-y)
  /-- The hyperbola is equilateral -/
  equilateral : ∀ x y, equation x y = equation y x

/-- The equation of the equilateral hyperbola is y² - x² = 9 -/
theorem equilateral_hyperbola_equation (h : EquilateralHyperbola) :
  ∀ x y, h.equation x y = y^2 - x^2 - 9 := by sorry

end NUMINAMATH_CALUDE_equilateral_hyperbola_equation_l3181_318168


namespace NUMINAMATH_CALUDE_nellie_legos_l3181_318113

/-- Given an initial number of legos, calculate the remaining legos after losing some and giving some away. -/
def remaining_legos (initial : ℕ) (lost : ℕ) (given : ℕ) : ℕ :=
  initial - lost - given

/-- Prove that Nellie has 299 legos remaining -/
theorem nellie_legos : remaining_legos 380 57 24 = 299 := by
  sorry

end NUMINAMATH_CALUDE_nellie_legos_l3181_318113


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3181_318117

theorem quadratic_equation_solution (m n : ℤ) :
  m^2 + 2*m*n + 2*n^2 - 4*n + 4 = 0 → m = -2 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3181_318117


namespace NUMINAMATH_CALUDE_thursday_seeds_count_l3181_318198

/-- The number of seeds planted on Wednesday -/
def seeds_wednesday : ℕ := 20

/-- The total number of seeds planted -/
def total_seeds : ℕ := 22

/-- The number of seeds planted on Thursday -/
def seeds_thursday : ℕ := total_seeds - seeds_wednesday

theorem thursday_seeds_count : seeds_thursday = 2 := by
  sorry

end NUMINAMATH_CALUDE_thursday_seeds_count_l3181_318198


namespace NUMINAMATH_CALUDE_quadratic_function_behavior_l3181_318125

def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 4

theorem quadratic_function_behavior (b : ℝ) :
  (∀ x₁ x₂, x₁ ≤ x₂ ∧ x₂ ≤ -1 → f b x₁ ≥ f b x₂) ∧
  (∀ x₁ x₂, -1 ≤ x₁ ∧ x₁ ≤ x₂ → f b x₁ ≤ f b x₂) →
  b > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_behavior_l3181_318125


namespace NUMINAMATH_CALUDE_student_rabbit_difference_is_105_l3181_318174

/-- The number of students in each third-grade classroom -/
def students_per_classroom : ℕ := 24

/-- The number of rabbits in each third-grade classroom -/
def rabbits_per_classroom : ℕ := 3

/-- The number of third-grade classrooms -/
def number_of_classrooms : ℕ := 5

/-- The difference between the total number of students and rabbits in all classrooms -/
def student_rabbit_difference : ℕ :=
  (students_per_classroom * number_of_classrooms) - (rabbits_per_classroom * number_of_classrooms)

theorem student_rabbit_difference_is_105 :
  student_rabbit_difference = 105 := by
  sorry

end NUMINAMATH_CALUDE_student_rabbit_difference_is_105_l3181_318174


namespace NUMINAMATH_CALUDE_diamond_three_four_l3181_318133

/-- The diamond operation defined for real numbers -/
def diamond (a b : ℝ) : ℝ := 4*a + 5*b - a^2*b

/-- Theorem stating that 3 ⋄ 4 = -4 -/
theorem diamond_three_four : diamond 3 4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_four_l3181_318133


namespace NUMINAMATH_CALUDE_house_sale_profit_l3181_318118

-- Define the initial home value
def initial_value : ℝ := 12000

-- Define the profit percentage for the first sale
def profit_percentage : ℝ := 0.20

-- Define the loss percentage for the second sale
def loss_percentage : ℝ := 0.15

-- Define the net profit
def net_profit : ℝ := 2160

-- Theorem statement
theorem house_sale_profit :
  let first_sale_price := initial_value * (1 + profit_percentage)
  let second_sale_price := first_sale_price * (1 - loss_percentage)
  first_sale_price - second_sale_price = net_profit := by
sorry

end NUMINAMATH_CALUDE_house_sale_profit_l3181_318118


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l3181_318163

/-- Sarah's bowling score problem -/
theorem sarahs_bowling_score :
  ∀ (sarah_score greg_score : ℕ),
  sarah_score = greg_score + 50 →
  (sarah_score + greg_score) / 2 = 110 →
  sarah_score = 135 :=
by
  sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l3181_318163


namespace NUMINAMATH_CALUDE_wind_speed_calculation_l3181_318159

/-- The speed of the wind that satisfies the given conditions -/
def wind_speed : ℝ := 20

/-- The speed of the plane in still air -/
def plane_speed : ℝ := 180

/-- The distance flown with the wind -/
def distance_with_wind : ℝ := 400

/-- The distance flown against the wind -/
def distance_against_wind : ℝ := 320

theorem wind_speed_calculation :
  (distance_with_wind / (plane_speed + wind_speed) = 
   distance_against_wind / (plane_speed - wind_speed)) ∧
  wind_speed = 20 := by
  sorry

end NUMINAMATH_CALUDE_wind_speed_calculation_l3181_318159


namespace NUMINAMATH_CALUDE_sector_area_l3181_318128

/-- Given a sector with central angle α and arc length l, 
    the area S of the sector is (l * l) / (2 * α) -/
theorem sector_area (α : ℝ) (l : ℝ) (S : ℝ) 
  (h1 : α = 2) 
  (h2 : l = 3 * Real.pi) 
  (h3 : S = (l * l) / (2 * α)) : 
  S = (9 * Real.pi^2) / 4 := by
  sorry

#check sector_area

end NUMINAMATH_CALUDE_sector_area_l3181_318128


namespace NUMINAMATH_CALUDE_placement_theorem_l3181_318136

def number_of_placements (n : ℕ) : ℕ := 
  Nat.choose 4 2 * (n * (n - 1))

theorem placement_theorem : number_of_placements 4 = 72 := by
  sorry

end NUMINAMATH_CALUDE_placement_theorem_l3181_318136


namespace NUMINAMATH_CALUDE_equation_solution_l3181_318175

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), (x₁ = 2 ∧ x₂ = -1 - Real.sqrt 5) ∧ 
  (∀ x : ℝ, x^2 - 2 * |x - 1| - 2 = 0 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3181_318175


namespace NUMINAMATH_CALUDE_percentage_calculation_l3181_318177

theorem percentage_calculation (x y : ℝ) : 
  (0.003 = (x/100) * 0.09) → 
  (0.008 = (y/100) * 0.15) → 
  (x = (0.003 / 0.09) * 100) ∧ 
  (y = (0.008 / 0.15) * 100) := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3181_318177


namespace NUMINAMATH_CALUDE_min_value_theorem_l3181_318169

/-- The function f(x) = |2x - 1| -/
def f (x : ℝ) : ℝ := |2 * x - 1|

/-- The function g(x) = f(x) + f(x - 1) -/
def g (x : ℝ) : ℝ := f x + f (x - 1)

/-- The minimum value of g(x) -/
def a : ℝ := 2

/-- Theorem: The minimum value of (m^2 + 2)/m + (n^2 + 1)/n is (7 + 2√2)/2,
    given m + n = a and m, n > 0 -/
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = a) :
  (m^2 + 2)/m + (n^2 + 1)/n ≥ (7 + 2 * Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3181_318169


namespace NUMINAMATH_CALUDE_drinking_speed_ratio_l3181_318189

theorem drinking_speed_ratio 
  (total_volume : ℝ) 
  (mala_volume : ℝ) 
  (usha_volume : ℝ) 
  (drinking_time : ℝ) 
  (h1 : drinking_time > 0) 
  (h2 : total_volume > 0) 
  (h3 : mala_volume + usha_volume = total_volume) 
  (h4 : usha_volume = 2 / 10 * total_volume) : 
  (mala_volume / drinking_time) / (usha_volume / drinking_time) = 4 := by
sorry

end NUMINAMATH_CALUDE_drinking_speed_ratio_l3181_318189


namespace NUMINAMATH_CALUDE_line_equation_l3181_318131

/-- A line L in R² -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : (a * X + b * Y + c = 0)

/-- Point in R² -/
structure Point where
  x : ℝ
  y : ℝ

def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def intersects_at (l1 l2 : Line) (x : ℝ) : Prop :=
  ∃ y : ℝ, l1.a * x + l1.b * y + l1.c = 0 ∧ l2.a * x + l2.b * y + l2.c = 0

theorem line_equation (l : Line) (p : Point) (l2 l3 : Line) :
  passes_through l { x := 1, y := 5 } →
  perpendicular l { a := 2, b := -5, c := 3, eq := sorry } →
  intersects_at l { a := 3, b := 1, c := -1, eq := sorry } (-1) →
  l = { a := 5, b := 2, c := -15, eq := sorry } :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l3181_318131


namespace NUMINAMATH_CALUDE_angle_bisector_exists_l3181_318193

-- Define the basic structures
structure Point :=
  (x : ℝ) (y : ℝ)

structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

-- Define the given lines
def L1 : Line := sorry
def L2 : Line := sorry

-- Define the property of inaccessible intersection
def inaccessibleIntersection (l1 l2 : Line) : Prop := sorry

-- Define angle bisector
def isAngleBisector (bisector : Line) (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem angle_bisector_exists (h : inaccessibleIntersection L1 L2) :
  ∃ bisector : Line, isAngleBisector bisector L1 L2 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_exists_l3181_318193


namespace NUMINAMATH_CALUDE_concert_stay_probability_l3181_318164

/-- The probability that at least 4 people stay for an entire concert, given the conditions. -/
theorem concert_stay_probability (total : ℕ) (certain : ℕ) (uncertain : ℕ) (p : ℚ) : 
  total = 8 →
  certain = 5 →
  uncertain = 3 →
  p = 1/3 →
  ∃ (prob : ℚ), prob = 19/27 ∧ 
    prob = (uncertain.choose 1 * p * (1-p)^2 + 
            uncertain.choose 2 * p^2 * (1-p) + 
            uncertain.choose 3 * p^3) := by
  sorry


end NUMINAMATH_CALUDE_concert_stay_probability_l3181_318164


namespace NUMINAMATH_CALUDE_largest_possible_b_value_l3181_318156

theorem largest_possible_b_value (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (c = 3) →  -- c is the smallest odd prime number
  (∀ x : ℕ, (1 < x) ∧ (x < b) ∧ (x ≠ c) → (a * x * c ≠ 360)) →
  (b = 8) :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_b_value_l3181_318156


namespace NUMINAMATH_CALUDE_chord_length_l3181_318166

/-- The length of chord AB formed by the intersection of a line and a circle -/
theorem chord_length (A B : ℝ × ℝ) : 
  (∀ (x y : ℝ), x + Real.sqrt 3 * y - 2 = 0 → x^2 + y^2 = 4 → (x, y) = A ∨ (x, y) = B) →
  ((A.1 + Real.sqrt 3 * A.2 - 2 = 0 ∧ A.1^2 + A.2^2 = 4) ∧
   (B.1 + Real.sqrt 3 * B.2 - 2 = 0 ∧ B.1^2 + B.2^2 = 4)) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l3181_318166


namespace NUMINAMATH_CALUDE_coplanar_condition_l3181_318154

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V] [CompleteSpace V]

-- Define the origin and points
variable (O A B C D : V)

-- Define the scalar m
variable (m : ℝ)

-- Define the condition for coplanarity
def coplanar (A B C D : V) : Prop :=
  ∃ (a b c : ℝ), (A - D) = a • (B - D) + b • (C - D) + c • (0 : V)

-- State the theorem
theorem coplanar_condition (h : ∀ A B C D : V, 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + m • (D - O) = 0 → coplanar A B C D) : 
  m = -7 := by sorry

end NUMINAMATH_CALUDE_coplanar_condition_l3181_318154


namespace NUMINAMATH_CALUDE_point_value_theorem_l3181_318106

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Represents the number line -/
structure NumberLine where
  origin : Point
  pointA : Point
  pointB : Point
  pointC : Point

def NumberLine.sameSide (nl : NumberLine) : Prop :=
  (nl.pointA.value - nl.origin.value) * (nl.pointB.value - nl.origin.value) > 0

theorem point_value_theorem (nl : NumberLine) 
  (h1 : nl.sameSide)
  (h2 : nl.pointB.value = 1)
  (h3 : nl.pointC.value = nl.pointA.value - 3)
  (h4 : nl.pointC.value = -nl.pointB.value) :
  nl.pointA.value = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_value_theorem_l3181_318106


namespace NUMINAMATH_CALUDE_modulus_of_z_l3181_318191

-- Define the complex number z
def z : ℂ := (1 - Complex.I) * (1 + Complex.I)^2 + 1

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3181_318191


namespace NUMINAMATH_CALUDE_dividend_proof_l3181_318110

theorem dividend_proof (y : ℝ) (x : ℝ) (h : y > 3) :
  (x = (3 * y + 5) * (2 * y - 1) + (5 * y - 13)) →
  (x = 6 * y^2 + 12 * y - 18) := by
  sorry

end NUMINAMATH_CALUDE_dividend_proof_l3181_318110


namespace NUMINAMATH_CALUDE_ratio_equality_l3181_318126

theorem ratio_equality (a b c : ℝ) (h : a/2 = b/3 ∧ b/3 = c/4 ∧ a/2 ≠ 0) : 
  (a - 2*c) / (a - 2*b) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3181_318126


namespace NUMINAMATH_CALUDE_product_equals_seven_l3181_318155

theorem product_equals_seven : 
  (1 + 1/1) * (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_seven_l3181_318155


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l3181_318183

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt x = 18 / (11 - Real.sqrt x)) ↔ (x = 81 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l3181_318183


namespace NUMINAMATH_CALUDE_fraction_difference_l3181_318151

theorem fraction_difference (n d : ℤ) : 
  d = 5 → n > d → n + 6 = 3 * d → n - d = 4 := by sorry

end NUMINAMATH_CALUDE_fraction_difference_l3181_318151


namespace NUMINAMATH_CALUDE_anne_found_five_bottle_caps_l3181_318197

/-- The number of bottle caps Anne found -/
def bottle_caps_found (initial final : ℕ) : ℕ := final - initial

/-- Proof that Anne found 5 bottle caps -/
theorem anne_found_five_bottle_caps :
  bottle_caps_found 10 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_anne_found_five_bottle_caps_l3181_318197


namespace NUMINAMATH_CALUDE_intersection_M_N_l3181_318190

def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def N : Set ℝ := {-3, -1, 1, 3, 5}

theorem intersection_M_N : M ∩ N = {-1, 1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3181_318190


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l3181_318108

-- Define the curve
def curve (x a : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def curve_derivative (x a : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a (a : ℝ) : 
  (curve (-1) a = a + 2) →  -- Point (-1, a+2) is on the curve
  (curve_derivative (-1) a = 8) →  -- Slope at (-1, a+2) is 8
  a = -6 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_l3181_318108


namespace NUMINAMATH_CALUDE_food_product_shelf_life_l3181_318141

/-- Represents the shelf life function of a food product -/
noncomputable def shelf_life (k b x : ℝ) : ℝ := Real.exp (k * x + b)

/-- Theorem stating the shelf life at 30°C and the maximum temperature for 80 hours shelf life -/
theorem food_product_shelf_life 
  (k b : ℝ) 
  (h1 : shelf_life k b 0 = 160) 
  (h2 : shelf_life k b 20 = 40) : 
  shelf_life k b 30 = 20 ∧ 
  ∀ x, shelf_life k b x ≥ 80 → x ≤ 10 := by
sorry


end NUMINAMATH_CALUDE_food_product_shelf_life_l3181_318141


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3181_318122

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^4 + 3 * x^3 - 5 * x^2 + 6 * x - 8) + (-5 * x^4 - 2 * x^3 + 4 * x^2 - 6 * x + 7) = 
  -3 * x^4 + x^3 - x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3181_318122


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3181_318157

/-- The length of the major axis of the ellipse x^2 + 4y^2 = 100 is 20 -/
theorem ellipse_major_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 + 4*y^2 = 100}
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
    (∀ (x y : ℝ), (x, y) ∈ ellipse ↔ x^2/a^2 + y^2/b^2 = 1) ∧
    2*a = 20 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3181_318157


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3181_318111

theorem sum_of_three_numbers : 3.15 + 0.014 + 0.458 = 3.622 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3181_318111


namespace NUMINAMATH_CALUDE_jerry_action_figures_l3181_318145

theorem jerry_action_figures
  (complete_collection : ℕ)
  (cost_per_figure : ℕ)
  (total_cost_to_complete : ℕ)
  (h1 : complete_collection = 16)
  (h2 : cost_per_figure = 8)
  (h3 : total_cost_to_complete = 72) :
  complete_collection - (total_cost_to_complete / cost_per_figure) = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_jerry_action_figures_l3181_318145


namespace NUMINAMATH_CALUDE_new_car_distance_l3181_318165

theorem new_car_distance (old_car_speed : ℝ) (old_car_distance : ℝ) (new_car_speed : ℝ) : 
  old_car_distance = 180 →
  new_car_speed = old_car_speed * 1.15 →
  new_car_speed * (old_car_distance / old_car_speed) = 207 :=
by sorry

end NUMINAMATH_CALUDE_new_car_distance_l3181_318165


namespace NUMINAMATH_CALUDE_total_black_dots_l3181_318158

/-- The number of black dots on a Type A butterfly -/
def dotsA : ℝ := 12

/-- The number of black dots on a Type B butterfly -/
def dotsB : ℝ := 8.5

/-- The number of black dots on a Type C butterfly -/
def dotsC : ℝ := 19

/-- The number of Type A butterflies -/
def numA : ℕ := 145

/-- The number of Type B butterflies -/
def numB : ℕ := 112

/-- The number of Type C butterflies -/
def numC : ℕ := 140

/-- The total number of butterflies -/
def totalButterflies : ℕ := 397

/-- Theorem: The total number of black dots among all butterflies is 5352 -/
theorem total_black_dots :
  dotsA * numA + dotsB * numB + dotsC * numC = 5352 := by
  sorry

end NUMINAMATH_CALUDE_total_black_dots_l3181_318158


namespace NUMINAMATH_CALUDE_goat_grazing_area_l3181_318134

/-- The side length of a square plot given a goat tied to one corner -/
theorem goat_grazing_area (rope_length : ℝ) (graze_area : ℝ) (side_length : ℝ) : 
  rope_length = 7 →
  graze_area = 38.48451000647496 →
  side_length = 7 →
  (1 / 4) * Real.pi * rope_length ^ 2 = graze_area →
  side_length = rope_length :=
by sorry

end NUMINAMATH_CALUDE_goat_grazing_area_l3181_318134


namespace NUMINAMATH_CALUDE_sqrt_sum_bounds_l3181_318150

theorem sqrt_sum_bounds :
  let m := Real.sqrt 4 + Real.sqrt 3
  3 < m ∧ m < 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_bounds_l3181_318150


namespace NUMINAMATH_CALUDE_product_173_240_l3181_318116

theorem product_173_240 : 
  (∃ n : ℕ, n * 12 = 173 * 240 ∧ n = 3460) → 173 * 240 = 41520 := by
  sorry

end NUMINAMATH_CALUDE_product_173_240_l3181_318116


namespace NUMINAMATH_CALUDE_saree_stripe_ratio_l3181_318172

theorem saree_stripe_ratio :
  ∀ (gold brown blue : ℕ),
    gold = brown →
    blue = 5 * gold →
    brown = 4 →
    blue = 60 →
    (gold : ℚ) / brown = 3 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_saree_stripe_ratio_l3181_318172


namespace NUMINAMATH_CALUDE_bisector_quadrilateral_is_square_l3181_318149

/-- A rectangle that is not a square -/
structure NonSquareRectangle where
  length : ℝ
  width : ℝ
  length_positive : 0 < length
  width_positive : 0 < width
  not_square : length ≠ width

/-- The quadrilateral formed by the intersection of angle bisectors -/
structure BisectorQuadrilateral (r : NonSquareRectangle) where
  vertices : Fin 4 → ℝ × ℝ

/-- Theorem: The quadrilateral formed by the intersection of angle bisectors in a non-square rectangle is a square -/
theorem bisector_quadrilateral_is_square (r : NonSquareRectangle) (q : BisectorQuadrilateral r) :
  IsSquare q.vertices := by sorry

end NUMINAMATH_CALUDE_bisector_quadrilateral_is_square_l3181_318149


namespace NUMINAMATH_CALUDE_atmosphere_depth_for_specific_peak_l3181_318146

/-- Represents a cone-shaped peak on an alien planet -/
structure ConePeak where
  height : ℝ
  atmosphereVolumeFraction : ℝ

/-- Calculates the depth of the atmosphere at the base of a cone-shaped peak -/
def atmosphereDepth (peak : ConePeak) : ℝ :=
  peak.height * (1 - (peak.atmosphereVolumeFraction)^(1/3))

/-- Theorem stating the depth of the atmosphere for a specific cone-shaped peak -/
theorem atmosphere_depth_for_specific_peak :
  let peak : ConePeak := { height := 5000, atmosphereVolumeFraction := 4/5 }
  atmosphereDepth peak = 340 := by
  sorry

end NUMINAMATH_CALUDE_atmosphere_depth_for_specific_peak_l3181_318146


namespace NUMINAMATH_CALUDE_cos_48_degrees_l3181_318102

theorem cos_48_degrees :
  ∃ x : ℝ, 4 * x^3 - 3 * x - (1 + Real.sqrt 5) / 4 = 0 ∧
  Real.cos (48 * π / 180) = (1 / 2) * x + (Real.sqrt 3 / 2) * Real.sqrt (1 - x^2) := by
  sorry

end NUMINAMATH_CALUDE_cos_48_degrees_l3181_318102


namespace NUMINAMATH_CALUDE_distance_to_work_is_27_l3181_318130

/-- The distance to work in miles -/
def distance_to_work : ℝ := 27

/-- The total commute time in hours -/
def total_commute_time : ℝ := 1.5

/-- The average speed to work in mph -/
def speed_to_work : ℝ := 45

/-- The average speed from work in mph -/
def speed_from_work : ℝ := 30

/-- Theorem stating that the distance to work is 27 miles -/
theorem distance_to_work_is_27 :
  distance_to_work = 27 ∧
  total_commute_time = distance_to_work / speed_to_work + distance_to_work / speed_from_work :=
by sorry

end NUMINAMATH_CALUDE_distance_to_work_is_27_l3181_318130


namespace NUMINAMATH_CALUDE_cupcakes_eaten_l3181_318180

theorem cupcakes_eaten (total_baked : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) : 
  total_baked = 68 →
  packages = 6 →
  cupcakes_per_package = 6 →
  total_baked - (packages * cupcakes_per_package) = 
    total_baked - (total_baked - (packages * cupcakes_per_package)) := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_eaten_l3181_318180


namespace NUMINAMATH_CALUDE_greatest_x_value_l3181_318137

theorem greatest_x_value (x : ℤ) (h : (2.134 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 210000) :
  x ≤ 4 ∧ ∃ y : ℤ, y > 4 → (2.134 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 210000 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3181_318137


namespace NUMINAMATH_CALUDE_total_moving_time_l3181_318186

/-- The time (in minutes) spent filling the car for each trip. -/
def fill_time : ℕ := 15

/-- The time (in minutes) spent driving one-way for each trip. -/
def drive_time : ℕ := 30

/-- The total number of trips made. -/
def num_trips : ℕ := 6

/-- The total time spent moving, in hours. -/
def total_time : ℚ := (fill_time + drive_time) * num_trips / 60

/-- Proves that the total time spent moving is 4.5 hours. -/
theorem total_moving_time : total_time = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_total_moving_time_l3181_318186


namespace NUMINAMATH_CALUDE_max_gcd_of_sequence_l3181_318127

theorem max_gcd_of_sequence (n : ℕ+) : Nat.gcd (101 + n^3) (101 + (n + 1)^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_of_sequence_l3181_318127


namespace NUMINAMATH_CALUDE_Tricia_age_is_5_l3181_318135

-- Define the ages as natural numbers
def Vincent_age : ℕ := 22
def Rupert_age : ℕ := Vincent_age - 2
def Khloe_age : ℕ := Rupert_age - 10
def Eugene_age : ℕ := Khloe_age * 3
def Yorick_age : ℕ := Eugene_age * 2
def Amilia_age : ℕ := Yorick_age / 4
def Tricia_age : ℕ := Amilia_age / 3

-- Theorem statement
theorem Tricia_age_is_5 : Tricia_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_Tricia_age_is_5_l3181_318135


namespace NUMINAMATH_CALUDE_min_value_of_a_l3181_318192

theorem min_value_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x : ℝ, x > 1 → x + a / (x - 1) ≥ 5) : 
  (∀ b : ℝ, b > 0 → (∀ x : ℝ, x > 1 → x + b / (x - 1) ≥ 5) → b ≥ a) ∧ a = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l3181_318192


namespace NUMINAMATH_CALUDE_snowball_theorem_l3181_318115

def snowball_distribution (lucy_snowballs : ℕ) (charlie_extra : ℕ) : Prop :=
  let charlie_initial := lucy_snowballs + charlie_extra
  let linus_received := charlie_initial / 2
  let charlie_final := charlie_initial / 2
  let sally_received := linus_received / 3
  let linus_final := linus_received - sally_received
  charlie_final = 25 ∧ lucy_snowballs = 19 ∧ linus_final = 17 ∧ sally_received = 8

theorem snowball_theorem : snowball_distribution 19 31 := by
  sorry

end NUMINAMATH_CALUDE_snowball_theorem_l3181_318115


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l3181_318114

/-- Regular triangular pyramid with specific properties -/
structure RegularTriangularPyramid where
  /-- Radius of the circle inscribed in the base -/
  r : ℝ
  /-- Side length of the base triangle -/
  s : ℝ
  /-- Height of the pyramid -/
  h : ℝ
  /-- The area of the inscribed circle equals its radius -/
  circle_area_eq_radius : π * r^2 = r
  /-- The side length of the base in terms of r -/
  side_length : s = 2 * r * Real.sqrt 3
  /-- The lateral surface area is three times the base area -/
  lateral_area_eq_3base : 3 * (s^2 * Real.sqrt 3 / 4) = 3 * ((s/2) * h)

/-- The volume of the regular triangular pyramid is (2√6) / π³ -/
theorem regular_triangular_pyramid_volume 
  (p : RegularTriangularPyramid) : 
  (1/3) * (p.s^2 * Real.sqrt 3 / 4) * p.h = (2 * Real.sqrt 6) / π^3 := by
  sorry


end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l3181_318114


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3181_318123

theorem system_solution_ratio (x y c d : ℝ) (h1 : 4 * x - 2 * y = c)
    (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : c / d = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3181_318123


namespace NUMINAMATH_CALUDE_pot_height_problem_l3181_318104

/-- Given two similar right-angled triangles, where one triangle has a height of 20 inches and
    a base of 10 inches, and the other triangle has a base of 20 inches,
    prove that the height of the second triangle is 40 inches. -/
theorem pot_height_problem (h₁ h₂ : ℝ) (b₁ b₂ : ℝ) :
  h₁ = 20 → b₁ = 10 → b₂ = 20 → (h₁ / b₁ = h₂ / b₂) → h₂ = 40 :=
by sorry

end NUMINAMATH_CALUDE_pot_height_problem_l3181_318104


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_l3181_318121

/-- Given five identical rectangles each with an area of 8 cm², 
    when arranged into a large rectangle, 
    the perimeter of the large rectangle is 32 cm. -/
theorem large_rectangle_perimeter (small_rectangle_area : ℝ) 
  (h1 : small_rectangle_area = 8) 
  (h2 : ∃ (w h : ℝ), w * h = small_rectangle_area ∧ h = 2 * w) : 
  ∃ (W H : ℝ), W * H = 5 * small_rectangle_area ∧ 2 * (W + H) = 32 :=
by sorry

end NUMINAMATH_CALUDE_large_rectangle_perimeter_l3181_318121


namespace NUMINAMATH_CALUDE_solve_for_y_l3181_318184

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 3*x + 2 = y + 6) (h2 : x = -4) : y = 24 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3181_318184


namespace NUMINAMATH_CALUDE_wire_cutting_l3181_318178

/-- Given a wire of length 28 cm, if one piece is 2.00001/5 times the length of the other,
    then the shorter piece is 20 cm long. -/
theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 28 →
  ratio = 2.00001 / 5 →
  shorter_length + ratio * shorter_length = total_length →
  shorter_length = 20 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l3181_318178


namespace NUMINAMATH_CALUDE_lunch_percentage_theorem_l3181_318170

theorem lunch_percentage_theorem (total : ℕ) (boy_ratio girl_ratio : ℕ) 
  (boy_lunch_percent girl_lunch_percent : ℚ) :
  boy_ratio + girl_ratio > 0 →
  boy_lunch_percent ≥ 0 →
  boy_lunch_percent ≤ 1 →
  girl_lunch_percent ≥ 0 →
  girl_lunch_percent ≤ 1 →
  boy_ratio = 6 →
  girl_ratio = 4 →
  boy_lunch_percent = 6/10 →
  girl_lunch_percent = 4/10 →
  (((boy_ratio * boy_lunch_percent + girl_ratio * girl_lunch_percent) / 
    (boy_ratio + girl_ratio)) : ℚ) = 52/100 := by
  sorry

end NUMINAMATH_CALUDE_lunch_percentage_theorem_l3181_318170


namespace NUMINAMATH_CALUDE_three_digit_divisibility_l3181_318103

def is_valid (x y z : Nat) : Prop :=
  let n := 579000 + 100 * x + 10 * y + z
  n % 5 = 0 ∧ n % 7 = 0 ∧ n % 9 = 0

theorem three_digit_divisibility :
  ∀ x y z : Nat,
    x < 10 ∧ y < 10 ∧ z < 10 →
    is_valid x y z ↔ (x = 6 ∧ y = 0 ∧ z = 0) ∨ 
                     (x = 2 ∧ y = 8 ∧ z = 5) ∨ 
                     (x = 9 ∧ y = 1 ∧ z = 5) :=
by sorry

#check three_digit_divisibility

end NUMINAMATH_CALUDE_three_digit_divisibility_l3181_318103


namespace NUMINAMATH_CALUDE_lunch_break_duration_l3181_318167

/- Define the workshop as a unit (100%) -/
def workshop : ℝ := 1

/- Define the working rates -/
variable (p : ℝ) -- Paula's painting rate (workshop/hour)
variable (h : ℝ) -- Combined rate of helpers (workshop/hour)

/- Define the lunch break duration in hours -/
variable (L : ℝ)

/- Monday's work -/
axiom monday_work : (9 - L) * (p + h) = 0.6 * workshop

/- Tuesday's work -/
axiom tuesday_work : (7 - L) * h = 0.3 * workshop

/- Wednesday's work -/
axiom wednesday_work : (10 - L) * p = 0.1 * workshop

/- The sum of work done on all three days equals the whole workshop -/
axiom total_work : 0.6 * workshop + 0.3 * workshop + 0.1 * workshop = workshop

/- Theorem: The lunch break is 48 minutes -/
theorem lunch_break_duration : L * 60 = 48 := by
  sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l3181_318167


namespace NUMINAMATH_CALUDE_triangle_selection_probability_l3181_318144

theorem triangle_selection_probability (total_triangles shaded_triangles : ℕ) 
  (h1 : total_triangles = 6)
  (h2 : shaded_triangles = 3)
  (h3 : shaded_triangles ≤ total_triangles)
  (h4 : total_triangles > 0) :
  (shaded_triangles : ℚ) / total_triangles = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_selection_probability_l3181_318144


namespace NUMINAMATH_CALUDE_total_candidates_l3181_318194

theorem total_candidates (girls : ℕ) (boys_fail_rate : ℝ) (girls_fail_rate : ℝ) (total_fail_rate : ℝ) :
  girls = 900 →
  boys_fail_rate = 0.7 →
  girls_fail_rate = 0.68 →
  total_fail_rate = 0.691 →
  ∃ (total : ℕ), total = 2000 ∧ 
    (boys_fail_rate * (total - girls) + girls_fail_rate * girls) / total = total_fail_rate :=
by sorry

end NUMINAMATH_CALUDE_total_candidates_l3181_318194


namespace NUMINAMATH_CALUDE_determinant_zero_implies_ratio_four_l3181_318132

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_zero_implies_ratio_four (θ : ℝ) : 
  determinant (Real.sin θ) 2 (Real.cos θ) 3 = 0 → 
  (3 * Real.sin θ + 2 * Real.cos θ) / (3 * Real.sin θ - Real.cos θ) = 4 :=
by sorry

end NUMINAMATH_CALUDE_determinant_zero_implies_ratio_four_l3181_318132


namespace NUMINAMATH_CALUDE_glove_selection_count_l3181_318195

def num_glove_pairs : ℕ := 6
def num_gloves_to_choose : ℕ := 4
def num_paired_gloves : ℕ := 2

theorem glove_selection_count :
  (num_glove_pairs.choose 1) * ((2 * num_glove_pairs - 2).choose (num_gloves_to_choose - num_paired_gloves) - (num_glove_pairs - 1)) = 240 := by
  sorry

end NUMINAMATH_CALUDE_glove_selection_count_l3181_318195


namespace NUMINAMATH_CALUDE_inequality_proof_l3181_318199

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3181_318199


namespace NUMINAMATH_CALUDE_selection_methods_l3181_318138

theorem selection_methods (n_boys n_girls n_select : ℕ) : 
  n_boys = 4 → n_girls = 3 → n_select = 4 →
  (Nat.choose (n_boys + n_girls) n_select) - (Nat.choose n_boys n_select) = 34 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_l3181_318138


namespace NUMINAMATH_CALUDE_circle_equation_l3181_318101

/-- Given a circle with center (-1, 2) passing through the point (2, -2),
    its standard equation is (x+1)^2 + (y-2)^2 = 25 -/
theorem circle_equation (x y : ℝ) : 
  let center := (-1, 2)
  let point_on_circle := (2, -2)
  (x + 1)^2 + (y - 2)^2 = 25 ↔ 
    (∃ (r : ℝ), r > 0 ∧
      (x - center.1)^2 + (y - center.2)^2 = r^2 ∧
      (point_on_circle.1 - center.1)^2 + (point_on_circle.2 - center.2)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3181_318101


namespace NUMINAMATH_CALUDE_min_distance_curve_to_line_l3181_318160

/-- The minimum distance from any point on the curve xy = √3 to the line x + √3y = 0 is √3 -/
theorem min_distance_curve_to_line :
  let C := {P : ℝ × ℝ | P.1 * P.2 = Real.sqrt 3}
  let l := {P : ℝ × ℝ | P.1 + Real.sqrt 3 * P.2 = 0}
  ∃ d : ℝ, d = Real.sqrt 3 ∧
    ∀ P ∈ C, ∀ Q ∈ l, d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_curve_to_line_l3181_318160


namespace NUMINAMATH_CALUDE_sum_of_X_and_Y_is_12_l3181_318148

/-- Converts a single-digit number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := n

/-- Converts a two-digit number from base 6 to base 10 -/
def twoDigitBase6ToBase10 (tens : ℕ) (ones : ℕ) : ℕ := 
  6 * tens + ones

theorem sum_of_X_and_Y_is_12 (X Y : ℕ) : 
  (X < 6 ∧ Y < 6) →  -- Ensure X and Y are single digits in base 6
  twoDigitBase6ToBase10 1 3 + twoDigitBase6ToBase10 X Y = 
  twoDigitBase6ToBase10 2 0 + twoDigitBase6ToBase10 5 2 →
  X + Y = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_X_and_Y_is_12_l3181_318148


namespace NUMINAMATH_CALUDE_i_pow_45_plus_345_l3181_318143

-- Define the imaginary unit i
axiom i : ℂ
axiom i_squared : i^2 = -1

-- Define the properties of i
axiom i_pow_one : i^1 = i
axiom i_pow_two : i^2 = -1
axiom i_pow_three : i^3 = -i
axiom i_pow_four : i^4 = 1

-- Define the cyclic nature of i
axiom i_cyclic (n : ℕ) : i^(n + 4) = i^n

-- Theorem to prove
theorem i_pow_45_plus_345 : i^45 + i^345 = 2*i := by
  sorry

end NUMINAMATH_CALUDE_i_pow_45_plus_345_l3181_318143


namespace NUMINAMATH_CALUDE_probability_male_student_id_l3181_318147

theorem probability_male_student_id (male_count female_count : ℕ) 
  (h1 : male_count = 6) (h2 : female_count = 4) : 
  (male_count : ℚ) / ((male_count : ℚ) + (female_count : ℚ)) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_male_student_id_l3181_318147


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3181_318187

theorem sufficient_condition_for_inequality (a : ℝ) :
  a ≥ 5 → ∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3181_318187


namespace NUMINAMATH_CALUDE_pierre_cake_consumption_l3181_318171

theorem pierre_cake_consumption (cake_weight : ℝ) (num_parts : ℕ) 
  (h1 : cake_weight = 400)
  (h2 : num_parts = 8)
  (h3 : num_parts > 0) :
  let part_weight := cake_weight / num_parts
  let nathalie_ate := part_weight
  let pierre_ate := 2 * nathalie_ate
  pierre_ate = 100 := by
  sorry

end NUMINAMATH_CALUDE_pierre_cake_consumption_l3181_318171


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3181_318124

theorem angle_measure_proof (C D : ℝ) : 
  C + D = 180 →  -- Angles are supplementary
  C = 7 * D →    -- C is 7 times D
  C = 157.5 :=   -- Measure of angle C
by sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3181_318124


namespace NUMINAMATH_CALUDE_bob_salary_calculation_l3181_318142

def initial_salary : ℝ := 3000
def raise_percentage : ℝ := 0.15
def cut_percentage : ℝ := 0.10
def bonus : ℝ := 500

def final_salary : ℝ := 
  initial_salary * (1 + raise_percentage) * (1 - cut_percentage) + bonus

theorem bob_salary_calculation : final_salary = 3605 := by
  sorry

end NUMINAMATH_CALUDE_bob_salary_calculation_l3181_318142


namespace NUMINAMATH_CALUDE_gcd_2183_1947_l3181_318153

theorem gcd_2183_1947 : Nat.gcd 2183 1947 = 59 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2183_1947_l3181_318153


namespace NUMINAMATH_CALUDE_platform_length_l3181_318100

/-- Given a train of length 450 meters that crosses a platform in 60 seconds
    and a signal pole in 30 seconds, prove that the length of the platform is 450 meters. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 450)
  (h2 : platform_crossing_time = 60)
  (h3 : pole_crossing_time = 30) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 450 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3181_318100


namespace NUMINAMATH_CALUDE_carols_age_difference_l3181_318112

theorem carols_age_difference (bob_age carol_age : ℕ) : 
  bob_age = 16 → carol_age = 50 → carol_age - 3 * bob_age = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_carols_age_difference_l3181_318112


namespace NUMINAMATH_CALUDE_duty_arrangements_l3181_318196

/-- Represents the number of teachers -/
def num_teachers : ℕ := 3

/-- Represents the number of days in a week -/
def num_days : ℕ := 5

/-- Represents the number of teachers required on Monday -/
def teachers_on_monday : ℕ := 2

/-- Represents the number of duty days per teacher -/
def duty_days_per_teacher : ℕ := 2

/-- Theorem stating the number of possible duty arrangements -/
theorem duty_arrangements :
  (num_teachers.choose teachers_on_monday) * ((num_days - 1).choose (num_teachers - 1)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_duty_arrangements_l3181_318196


namespace NUMINAMATH_CALUDE_remainder_division_l3181_318107

theorem remainder_division (y : ℤ) (h : y % 288 = 45) : y % 24 = 21 := by
  sorry

end NUMINAMATH_CALUDE_remainder_division_l3181_318107


namespace NUMINAMATH_CALUDE_organization_growth_l3181_318152

def population_growth (initial : ℕ) (years : ℕ) : ℕ :=
  match years with
  | 0 => initial
  | n + 1 => 3 * (population_growth initial n - 5) + 5

theorem organization_growth :
  population_growth 20 6 = 10895 := by
  sorry

end NUMINAMATH_CALUDE_organization_growth_l3181_318152


namespace NUMINAMATH_CALUDE_negative_sqrt_four_equals_negative_two_l3181_318105

theorem negative_sqrt_four_equals_negative_two : -Real.sqrt 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_four_equals_negative_two_l3181_318105


namespace NUMINAMATH_CALUDE_hawk_percentage_is_65_percent_l3181_318120

/-- Represents the percentage of birds that are hawks in the nature reserve -/
def hawk_percentage : ℝ := sorry

/-- Represents the percentage of non-hawks that are paddyfield-warblers -/
def paddyfield_warbler_ratio : ℝ := 0.4

/-- Represents the ratio of kingfishers to paddyfield-warblers -/
def kingfisher_to_warbler_ratio : ℝ := 0.25

/-- Represents the percentage of birds that are not hawks, paddyfield-warblers, or kingfishers -/
def other_birds_percentage : ℝ := 0.35

theorem hawk_percentage_is_65_percent :
  hawk_percentage = 0.65 ∧
  paddyfield_warbler_ratio * (1 - hawk_percentage) +
  kingfisher_to_warbler_ratio * paddyfield_warbler_ratio * (1 - hawk_percentage) +
  hawk_percentage +
  other_birds_percentage = 1 :=
sorry

end NUMINAMATH_CALUDE_hawk_percentage_is_65_percent_l3181_318120


namespace NUMINAMATH_CALUDE_parabola_properties_l3181_318188

/-- A parabola that intersects the x-axis at (-3,0) and (1,0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_neg : a < 0
  h_root1 : a * (-3)^2 + b * (-3) + c = 0
  h_root2 : a * 1^2 + b * 1 + c = 0

/-- Properties of the parabola -/
theorem parabola_properties (p : Parabola) :
  (p.b^2 - 4*p.a*p.c > 0) ∧ (3*p.b + 2*p.c = 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3181_318188


namespace NUMINAMATH_CALUDE_coefficient_x4_is_160_l3181_318181

/-- The coefficient of x^4 in the expansion of (1+x) * (1+2x)^5 -/
def coefficient_x4 : ℕ :=
  -- Define the coefficient here
  sorry

/-- Theorem stating that the coefficient of x^4 in the expansion of (1+x) * (1+2x)^5 is 160 -/
theorem coefficient_x4_is_160 : coefficient_x4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_is_160_l3181_318181


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3181_318119

theorem square_area_from_diagonal (d : ℝ) (h : d = 40) :
  let s := d / Real.sqrt 2
  s * s = 800 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3181_318119


namespace NUMINAMATH_CALUDE_hua_optimal_selection_uses_golden_ratio_l3181_318109

/-- The mathematical concept used in Hua Luogeng's optimal selection method -/
inductive OptimalSelectionConcept
  | GoldenRatio
  | Mean
  | Mode
  | Median

/-- Hua Luogeng's optimal selection method -/
def huaOptimalSelectionMethod : OptimalSelectionConcept := OptimalSelectionConcept.GoldenRatio

/-- Theorem: The mathematical concept used in Hua Luogeng's optimal selection method is the Golden ratio -/
theorem hua_optimal_selection_uses_golden_ratio :
  huaOptimalSelectionMethod = OptimalSelectionConcept.GoldenRatio := by
  sorry

end NUMINAMATH_CALUDE_hua_optimal_selection_uses_golden_ratio_l3181_318109


namespace NUMINAMATH_CALUDE_power_multiply_l3181_318162

theorem power_multiply (x : ℝ) : x^2 * x^3 = x^5 := by sorry

end NUMINAMATH_CALUDE_power_multiply_l3181_318162


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3181_318140

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (m - 2) * (-3) - 8 + 3 * m + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3181_318140


namespace NUMINAMATH_CALUDE_smallest_w_proof_l3181_318179

/-- The product of 1452 and the smallest positive integer w that results in a number 
    with 3^3 and 13^3 as factors -/
def smallest_w : ℕ := 19773

theorem smallest_w_proof :
  ∀ w : ℕ, w > 0 →
  (∃ k : ℕ, 1452 * w = k * 3^3 * 13^3) →
  w ≥ smallest_w :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_proof_l3181_318179


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_9_l3181_318129

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def five_digit_number (x : ℕ) : ℕ := 24600 + 10 * x + 8

theorem missing_digit_divisible_by_9 :
  ∀ x : ℕ, x < 10 →
    (is_divisible_by_9 (five_digit_number x) ↔ x = 7) :=
by sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_9_l3181_318129


namespace NUMINAMATH_CALUDE_sin_120_degrees_l3181_318139

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l3181_318139


namespace NUMINAMATH_CALUDE_crayon_distribution_l3181_318182

theorem crayon_distribution (total_crayons : ℕ) (num_people : ℕ) (crayons_per_person : ℕ) :
  total_crayons = 24 →
  num_people = 3 →
  total_crayons = num_people * crayons_per_person →
  crayons_per_person = 8 := by
  sorry

end NUMINAMATH_CALUDE_crayon_distribution_l3181_318182


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3181_318173

theorem complex_equation_solution (a : ℝ) : (a - Complex.I)^2 = 2 * Complex.I → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3181_318173


namespace NUMINAMATH_CALUDE_square_divisibility_l3181_318161

theorem square_divisibility (a b : ℕ+) (h : (a * b + 1) ∣ (a ^ 2 + b ^ 2)) :
  ∃ k : ℕ, (a ^ 2 + b ^ 2) / (a * b + 1) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_divisibility_l3181_318161


namespace NUMINAMATH_CALUDE_otimes_difference_l3181_318176

-- Define the ⊗ operation
def otimes (a b : ℚ) : ℚ := a^3 / b

-- State the theorem
theorem otimes_difference : 
  (otimes (otimes 2 3) 4) - (otimes 2 (otimes 3 4)) = 80 / 27 := by
  sorry

end NUMINAMATH_CALUDE_otimes_difference_l3181_318176


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l3181_318185

/-- Given a line segment with one endpoint (6,4) and midpoint (3,10),
    the sum of the coordinates of the other endpoint is 16. -/
theorem endpoint_coordinate_sum : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop :=
  fun endpoint1 midpoint endpoint2 =>
    endpoint1 = (6, 4) ∧
    midpoint = (3, 10) ∧
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
    endpoint2.1 + endpoint2.2 = 16

/-- Proof of the theorem -/
theorem endpoint_coordinate_sum_proof : ∃ (endpoint2 : ℝ × ℝ),
  endpoint_coordinate_sum (6, 4) (3, 10) endpoint2 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l3181_318185
