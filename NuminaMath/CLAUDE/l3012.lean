import Mathlib

namespace NUMINAMATH_CALUDE_tim_travel_distance_l3012_301287

/-- Represents the problem of Tim and Élan moving towards each other with increasing speeds -/
structure MeetingProblem where
  initialDistance : ℝ
  timInitialSpeed : ℝ
  elanInitialSpeed : ℝ

/-- Calculates the distance Tim travels before meeting Élan -/
def distanceTraveled (p : MeetingProblem) : ℝ :=
  sorry

/-- Theorem stating that Tim travels 20 miles before meeting Élan -/
theorem tim_travel_distance (p : MeetingProblem) 
  (h1 : p.initialDistance = 30)
  (h2 : p.timInitialSpeed = 10)
  (h3 : p.elanInitialSpeed = 5) :
  distanceTraveled p = 20 :=
sorry

end NUMINAMATH_CALUDE_tim_travel_distance_l3012_301287


namespace NUMINAMATH_CALUDE_ball_radius_from_hole_l3012_301249

theorem ball_radius_from_hole (hole_diameter : ℝ) (hole_depth : ℝ) (ball_radius : ℝ) : 
  hole_diameter = 24 →
  hole_depth = 8 →
  (hole_diameter / 2) ^ 2 + (ball_radius - hole_depth) ^ 2 = ball_radius ^ 2 →
  ball_radius = 13 := by
sorry

end NUMINAMATH_CALUDE_ball_radius_from_hole_l3012_301249


namespace NUMINAMATH_CALUDE_sphere_properties_l3012_301269

/-- For a sphere with volume 72π cubic inches, prove its surface area and diameter -/
theorem sphere_properties (V : ℝ) (h : V = 72 * Real.pi) :
  let r := (3 * V / (4 * Real.pi)) ^ (1/3)
  (4 * Real.pi * r^2 = 36 * Real.pi * 2^(2/3)) ∧
  (2 * r = 6 * 2^(1/3)) := by
sorry

end NUMINAMATH_CALUDE_sphere_properties_l3012_301269


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3012_301276

/-- A quadratic equation ax^2 - 4x - 2 = 0 has real roots if and only if a ≥ -2 and a ≠ 0 -/
theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 4*x - 2 = 0) ↔ (a ≥ -2 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3012_301276


namespace NUMINAMATH_CALUDE_max_square_area_with_perimeter_34_l3012_301237

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def rectangle_area (l w : ℕ) : ℕ := l * w

def rectangle_perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem max_square_area_with_perimeter_34 :
  ∀ l w : ℕ,
    rectangle_perimeter l w = 34 →
    is_perfect_square (rectangle_area l w) →
    rectangle_area l w ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_max_square_area_with_perimeter_34_l3012_301237


namespace NUMINAMATH_CALUDE_invalid_formula_l3012_301200

def sequence_formula_a (n : ℕ) : ℚ := n
def sequence_formula_b (n : ℕ) : ℚ := n^3 - 6*n^2 + 12*n - 6
def sequence_formula_c (n : ℕ) : ℚ := (1/2)*n^2 - (1/2)*n + 1
def sequence_formula_d (n : ℕ) : ℚ := 6 / (n^2 - 6*n + 11)

theorem invalid_formula :
  (sequence_formula_a 1 = 1 ∧ sequence_formula_a 2 = 2 ∧ sequence_formula_a 3 = 3) ∧
  (sequence_formula_b 1 = 1 ∧ sequence_formula_b 2 = 2 ∧ sequence_formula_b 3 = 3) ∧
  (sequence_formula_d 1 = 1 ∧ sequence_formula_d 2 = 2 ∧ sequence_formula_d 3 = 3) ∧
  ¬(sequence_formula_c 1 = 1 ∧ sequence_formula_c 2 = 2 ∧ sequence_formula_c 3 = 3) :=
by sorry

end NUMINAMATH_CALUDE_invalid_formula_l3012_301200


namespace NUMINAMATH_CALUDE_inequality_proof_l3012_301243

theorem inequality_proof (x y : ℝ) : 5 * x^2 + y^2 + 4 - 4 * x - 4 * x * y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3012_301243


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l3012_301279

/-- The area of an equilateral triangle with altitude 2√3 is 4√3 -/
theorem equilateral_triangle_area (h : ℝ) (altitude : h = 2 * Real.sqrt 3) :
  let side := 2 * h / Real.sqrt 3
  let area := 1/2 * side * h
  area = 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l3012_301279


namespace NUMINAMATH_CALUDE_limestone_cost_proof_l3012_301264

/-- The cost of limestone per pound -/
def limestone_cost : ℝ := 3

/-- The total weight of the compound in pounds -/
def total_weight : ℝ := 100

/-- The total cost of the compound in dollars -/
def total_cost : ℝ := 425

/-- The weight of limestone used in the compound in pounds -/
def limestone_weight : ℝ := 37.5

/-- The weight of shale mix used in the compound in pounds -/
def shale_weight : ℝ := 62.5

/-- The cost of shale mix per pound in dollars -/
def shale_cost_per_pound : ℝ := 5

/-- The total cost of shale mix in the compound in dollars -/
def total_shale_cost : ℝ := 312.5

theorem limestone_cost_proof :
  limestone_cost * limestone_weight + total_shale_cost = total_cost ∧
  limestone_weight + shale_weight = total_weight ∧
  shale_cost_per_pound * shale_weight = total_shale_cost :=
by sorry

end NUMINAMATH_CALUDE_limestone_cost_proof_l3012_301264


namespace NUMINAMATH_CALUDE_number_puzzle_l3012_301258

theorem number_puzzle : ∃ x : ℤ, x - 2 + 4 = 9 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3012_301258


namespace NUMINAMATH_CALUDE_tangent_line_to_circleC_l3012_301214

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The circle C: x^2 + y^2 - 6y + 8 = 0 -/
def circleC : Circle := { center := (0, 3), radius := 1 }

/-- A line in the form y = kx -/
structure Line where
  k : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  ∃ p : ℝ × ℝ, p.2 = l.k * p.1 ∧ 
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
    ∀ q : ℝ × ℝ, q.2 = l.k * q.1 → 
      (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 ≥ c.radius^2

theorem tangent_line_to_circleC (l : Line) :
  isTangent l circleC ∧ 
  (∃ p : ℝ × ℝ, isTangent l circleC ∧ isInSecondQuadrant p) →
  l.k = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_circleC_l3012_301214


namespace NUMINAMATH_CALUDE_parallelogram_area_l3012_301251

/-- The area of a parallelogram with given side lengths and included angle -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (ha : a = 12) (hb : b = 18) (hθ : θ = 45 * π / 180) :
  abs (a * b * Real.sin θ - 152.73) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3012_301251


namespace NUMINAMATH_CALUDE_store_brand_butter_price_l3012_301268

/-- The price of a single 16 oz package of store-brand butter -/
def single_package_price : ℝ := 6

/-- The price of an 8 oz package of butter -/
def eight_oz_price : ℝ := 4

/-- The normal price of a 4 oz package of butter -/
def four_oz_normal_price : ℝ := 2

/-- The discount rate for 4 oz packages -/
def discount_rate : ℝ := 0.5

/-- The lowest price for 16 oz of butter -/
def lowest_price : ℝ := 6

theorem store_brand_butter_price :
  single_package_price = lowest_price ∧
  lowest_price ≤ eight_oz_price + 2 * (four_oz_normal_price * (1 - discount_rate)) :=
by sorry

end NUMINAMATH_CALUDE_store_brand_butter_price_l3012_301268


namespace NUMINAMATH_CALUDE_profit_increase_may_to_june_l3012_301229

theorem profit_increase_may_to_june
  (march_to_april : Real)
  (april_to_may : Real)
  (march_to_june : Real)
  (h1 : march_to_april = 0.30)
  (h2 : april_to_may = -0.20)
  (h3 : march_to_june = 0.5600000000000001)
  : ∃ may_to_june : Real,
    (1 + march_to_april) * (1 + april_to_may) * (1 + may_to_june) = 1 + march_to_june ∧
    may_to_june = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_profit_increase_may_to_june_l3012_301229


namespace NUMINAMATH_CALUDE_visit_all_points_prob_one_l3012_301285

/-- Represents a one-dimensional random walk --/
structure RandomWalk where
  p : ℝ  -- Probability of moving right or left
  r : ℝ  -- Probability of staying in place
  prob_sum : p + p + r = 1  -- Sum of probabilities equals 1

/-- The probability of eventually reaching any point from any starting position --/
def eventual_visit_prob (rw : RandomWalk) : ℝ → ℝ := sorry

/-- Theorem stating that if p > 0, the probability of visiting any point is 1 --/
theorem visit_all_points_prob_one (rw : RandomWalk) (h : rw.p > 0) :
  ∀ x, eventual_visit_prob rw x = 1 := by sorry

end NUMINAMATH_CALUDE_visit_all_points_prob_one_l3012_301285


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3012_301288

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -5/12
  let a₃ : ℚ := 35/144
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 2 → a₂ / a₁ = a₃ / a₂) →
  r = -10/21 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3012_301288


namespace NUMINAMATH_CALUDE_chord_length_l3012_301226

/-- Circle C with equation x^2 + y^2 - 4x - 4y + 4 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 4 = 0

/-- Line l passing through points (4,0) and (0,2) -/
def line_l (x y : ℝ) : Prop := x + 2*y = 4

/-- The length of the chord cut by line l on circle C is 8√5/5 -/
theorem chord_length : ∃ (x₁ y₁ x₂ y₂ : ℝ),
  circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ 
  line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
  ((x₁ - x₂)^2 + (y₁ - y₂)^2) = (8*Real.sqrt 5/5)^2 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l3012_301226


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l3012_301223

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : (4/3 * π * r₁^3) / (4/3 * π * r₂^3) = 8/27) :
  (4 * π * r₁^2) / (4 * π * r₂^2) = 4/9 := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l3012_301223


namespace NUMINAMATH_CALUDE_trajectory_of_point_P_l3012_301239

/-- The trajectory of point P satisfying given conditions -/
theorem trajectory_of_point_P (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (x - a) * b = y * a ∧  -- P lies on line AB
    (x - 0)^2 + (y - b)^2 = 4 * ((a - x)^2 + y^2) ∧  -- BP = 2PA
    (-x) * (-a) + y * b = 1)  -- OQ · AB = 1
  → 3/2 * x^2 + 3 * y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_point_P_l3012_301239


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3012_301204

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, 2 * x^2 - 9 * x + a < 0 ∧ (x^2 - 4 * x + 3 < 0 ∨ x^2 - 6 * x + 8 < 0)) ↔
  a < 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3012_301204


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3012_301271

/-- Given a quadratic equation ax^2 + 16x + c = 0 with exactly one solution,
    where a + c = 25 and a < c, prove that a = 3 and c = 22 -/
theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 16 * x + c = 0) →  -- exactly one solution
  (a + c = 25) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 3 ∧ c = 22) :=                 -- conclusion
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3012_301271


namespace NUMINAMATH_CALUDE_value_of_b_l3012_301275

theorem value_of_b (a b c : ℝ) 
  (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
  (h2 : 6 * b * 7 = 1.5) : 
  b = 15 := by sorry

end NUMINAMATH_CALUDE_value_of_b_l3012_301275


namespace NUMINAMATH_CALUDE_negation_relationship_l3012_301256

theorem negation_relationship (x : ℝ) :
  (¬(x^2 + x - 6 > 0) → ¬(16 - x^2 < 0)) ∧
  ¬(¬(16 - x^2 < 0) → ¬(x^2 + x - 6 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_relationship_l3012_301256


namespace NUMINAMATH_CALUDE_nabla_calculation_l3012_301232

def nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem nabla_calculation : nabla (nabla 4 3) 2 = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l3012_301232


namespace NUMINAMATH_CALUDE_total_guests_proof_l3012_301206

def number_of_guests (adults children seniors teenagers toddlers vip : ℕ) : ℕ :=
  adults + children + seniors + teenagers + toddlers + vip

theorem total_guests_proof :
  ∃ (adults children seniors teenagers toddlers vip : ℕ),
    adults = 58 ∧
    children = adults - 35 ∧
    seniors = 2 * children ∧
    teenagers = seniors - 15 ∧
    toddlers = teenagers / 2 ∧
    vip = teenagers - 20 ∧
    ∃ (n : ℕ), vip = n^2 ∧
    number_of_guests adults children seniors teenagers toddlers vip = 198 :=
by
  sorry

end NUMINAMATH_CALUDE_total_guests_proof_l3012_301206


namespace NUMINAMATH_CALUDE_parabola_point_relation_l3012_301208

-- Define the parabola function
def parabola (c : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 4 * x + c

-- Define the theorem
theorem parabola_point_relation (c : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h1 : parabola c (-4) = y₁)
  (h2 : parabola c (-2) = y₂)
  (h3 : parabola c (1/2) = y₃) :
  y₁ > y₂ ∧ y₂ > y₃ :=
sorry

end NUMINAMATH_CALUDE_parabola_point_relation_l3012_301208


namespace NUMINAMATH_CALUDE_vector_opposite_directions_x_value_l3012_301291

def vector_a (x : ℝ) : Fin 2 → ℝ := ![1, -x]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![x, -6]

def opposite_directions (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ v = fun i ↦ k * w i

theorem vector_opposite_directions_x_value :
  ∀ x : ℝ, opposite_directions (vector_a x) (vector_b x) → x = -Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_vector_opposite_directions_x_value_l3012_301291


namespace NUMINAMATH_CALUDE_collinear_points_sum_l3012_301255

/-- Given three collinear points in 3D space, prove that the sum of x and y coordinates of two points is -1/2 --/
theorem collinear_points_sum (x y : ℝ) : 
  let A : ℝ × ℝ × ℝ := (1, 2, 0)
  let B : ℝ × ℝ × ℝ := (x, 3, -1)
  let C : ℝ × ℝ × ℝ := (4, y, 2)
  (∃ (t : ℝ), B - A = t • (C - A)) → x + y = -1/2 := by
sorry


end NUMINAMATH_CALUDE_collinear_points_sum_l3012_301255


namespace NUMINAMATH_CALUDE_smallest_with_eight_factors_l3012_301247

/-- A function that returns the number of distinct positive factors of a positive integer -/
def number_of_factors (n : ℕ+) : ℕ := sorry

/-- A function that checks if a given number has exactly eight distinct positive factors -/
def has_eight_factors (n : ℕ+) : Prop := number_of_factors n = 8

/-- Theorem stating that 24 is the smallest positive integer with exactly eight distinct positive factors -/
theorem smallest_with_eight_factors :
  has_eight_factors 24 ∧ ∀ m : ℕ+, m < 24 → ¬(has_eight_factors m) :=
sorry

end NUMINAMATH_CALUDE_smallest_with_eight_factors_l3012_301247


namespace NUMINAMATH_CALUDE_rafael_net_pay_l3012_301213

/-- Calculates the total net pay for Rafael's work week --/
def calculate_net_pay (monday_hours : ℕ) (tuesday_hours : ℕ) (total_week_hours : ℕ) 
  (max_daily_hours : ℕ) (regular_rate : ℚ) (overtime_rate : ℚ) (bonus : ℚ) 
  (tax_rate : ℚ) (tax_credit : ℚ) : ℚ :=
  let remaining_days := 3
  let remaining_hours := total_week_hours - monday_hours - tuesday_hours
  let wednesday_hours := min max_daily_hours remaining_hours
  let thursday_hours := min max_daily_hours (remaining_hours - wednesday_hours)
  let friday_hours := remaining_hours - wednesday_hours - thursday_hours
  
  let monday_pay := regular_rate * min monday_hours max_daily_hours + 
    overtime_rate * max (monday_hours - max_daily_hours) 0
  let tuesday_pay := regular_rate * tuesday_hours
  let wednesday_pay := regular_rate * wednesday_hours
  let thursday_pay := regular_rate * thursday_hours
  let friday_pay := regular_rate * friday_hours
  
  let total_pay := monday_pay + tuesday_pay + wednesday_pay + thursday_pay + friday_pay + bonus
  let taxes := max (total_pay * tax_rate - tax_credit) 0
  
  total_pay - taxes

/-- Theorem stating that Rafael's net pay for the week is $878 --/
theorem rafael_net_pay : 
  calculate_net_pay 10 8 40 8 20 30 100 (1/10) 50 = 878 := by
  sorry

end NUMINAMATH_CALUDE_rafael_net_pay_l3012_301213


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l3012_301231

theorem smallest_angle_solution (x : ℝ) : 
  (∀ y ∈ {y : ℝ | 0 < y ∧ y < x}, ¬(Real.sin (2*y) * Real.sin (3*y) = Real.cos (2*y) * Real.cos (3*y))) ∧
  (Real.sin (2*x) * Real.sin (3*x) = Real.cos (2*x) * Real.cos (3*x)) ∧
  (x * (180 / Real.pi) = 18) := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l3012_301231


namespace NUMINAMATH_CALUDE_f₁_solution_set_f₂_min_value_l3012_301207

-- Part 1
def f₁ (x : ℝ) : ℝ := |3*x - 1| + |x + 3|

theorem f₁_solution_set : 
  {x : ℝ | f₁ x ≥ 4} = {x : ℝ | x ≤ -3 ∨ x ≥ 1/2} := by sorry

-- Part 2
def f₂ (b c x : ℝ) : ℝ := |x - b| + |x + c|

theorem f₂_min_value (b c : ℝ) (hb : b > 0) (hc : c > 0) 
  (hmin : ∃ x, ∀ y, f₂ b c x ≤ f₂ b c y) 
  (hval : ∃ x, f₂ b c x = 1) : 
  (1/b + 1/c) ≥ 4 ∧ ∃ b c, (1/b + 1/c = 4 ∧ b > 0 ∧ c > 0) := by sorry

end NUMINAMATH_CALUDE_f₁_solution_set_f₂_min_value_l3012_301207


namespace NUMINAMATH_CALUDE_any_amount_possible_large_amount_without_change_l3012_301221

/-- Represents the currency system of Bordavia -/
structure BordaviaCurrency where
  m : ℕ  -- value of silver coin
  n : ℕ  -- value of gold coin
  h1 : ∃ (a b : ℕ), a * m + b * n = 10000
  h2 : ∃ (a b : ℕ), a * m + b * n = 1875
  h3 : ∃ (a b : ℕ), a * m + b * n = 3072

/-- Any integer amount of Bourbakis can be obtained using gold and silver coins -/
theorem any_amount_possible (currency : BordaviaCurrency) :
  ∀ k : ℤ, ∃ (a b : ℤ), a * currency.m + b * currency.n = k :=
sorry

/-- Any amount over (mn - 2) Bourbakis can be paid without needing change -/
theorem large_amount_without_change (currency : BordaviaCurrency) :
  ∀ k : ℕ, k > currency.m * currency.n - 2 →
    ∃ (a b : ℕ), a * currency.m + b * currency.n = k :=
sorry

end NUMINAMATH_CALUDE_any_amount_possible_large_amount_without_change_l3012_301221


namespace NUMINAMATH_CALUDE_rachel_dvd_fraction_l3012_301228

def total_earnings : ℚ := 200
def lunch_fraction : ℚ := 1/4
def money_left : ℚ := 50

theorem rachel_dvd_fraction :
  let lunch_cost : ℚ := lunch_fraction * total_earnings
  let money_after_lunch : ℚ := total_earnings - lunch_cost
  let dvd_cost : ℚ := money_after_lunch - money_left
  dvd_cost / total_earnings = 1/2 := by sorry

end NUMINAMATH_CALUDE_rachel_dvd_fraction_l3012_301228


namespace NUMINAMATH_CALUDE_problem_statement_l3012_301293

theorem problem_statement (x y : ℝ) (h : x + 2 * y - 3 = 0) : 2 * x * 4 * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3012_301293


namespace NUMINAMATH_CALUDE_hem_length_is_three_feet_l3012_301224

/-- The length of a stitch in inches -/
def stitch_length : ℚ := 1/4

/-- The number of stitches Jenna makes per minute -/
def stitches_per_minute : ℕ := 24

/-- The time it takes Jenna to hem her dress in minutes -/
def hemming_time : ℕ := 6

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The length of the dress's hem in feet -/
def hem_length : ℚ := (stitches_per_minute * hemming_time * stitch_length) / inches_per_foot

theorem hem_length_is_three_feet : hem_length = 3 := by
  sorry

end NUMINAMATH_CALUDE_hem_length_is_three_feet_l3012_301224


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l3012_301261

/-- Given a cube with face perimeter of 20 cm, prove its volume is 125 cubic centimeters -/
theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 20) : 
  let side_length := face_perimeter / 4
  let volume := side_length ^ 3
  volume = 125 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l3012_301261


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3012_301227

theorem solve_exponential_equation :
  ∃ n : ℕ, (9 : ℝ)^n * (9 : ℝ)^n * (9 : ℝ)^n * (9 : ℝ)^n = (729 : ℝ)^4 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3012_301227


namespace NUMINAMATH_CALUDE_new_arithmetic_mean_l3012_301274

/-- Given a set of 60 numbers with arithmetic mean 42, prove that removing 50 and 60
    and increasing each remaining number by 2 results in a new arithmetic mean of 43.55 -/
theorem new_arithmetic_mean (S : Finset ℝ) (sum_S : ℝ) : 
  S.card = 60 →
  sum_S = S.sum id →
  sum_S / 60 = 42 →
  50 ∈ S →
  60 ∈ S →
  let S' := S.erase 50 ⊔ S.erase 60
  let sum_S' := S'.sum (fun x => x + 2)
  sum_S' / 58 = 43.55 := by
sorry

end NUMINAMATH_CALUDE_new_arithmetic_mean_l3012_301274


namespace NUMINAMATH_CALUDE_geometric_mean_of_4_and_9_l3012_301284

-- Define the geometric mean
def geometric_mean (a c : ℝ) : Set ℝ :=
  {b : ℝ | a * c = b^2}

-- Theorem statement
theorem geometric_mean_of_4_and_9 :
  geometric_mean 4 9 = {6, -6} := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_4_and_9_l3012_301284


namespace NUMINAMATH_CALUDE_peters_pizza_portion_pizza_problem_l3012_301215

theorem peters_pizza_portion (total_slices : ℕ) (whole_slices : ℕ) 
  (shared_with_paul : ℚ) (shared_with_paul_and_sarah : ℚ) : ℚ :=
  by
  sorry

theorem pizza_problem :
  let total_slices : ℕ := 16
  let whole_slices : ℕ := 2
  let shared_with_paul : ℚ := 1 / 2
  let shared_with_paul_and_sarah : ℚ := 1 / 3
  peters_pizza_portion total_slices whole_slices shared_with_paul shared_with_paul_and_sarah = 17 / 96 :=
by
  sorry

end NUMINAMATH_CALUDE_peters_pizza_portion_pizza_problem_l3012_301215


namespace NUMINAMATH_CALUDE_probability_different_colors_bags_l3012_301218

/-- Represents a bag of colored balls -/
structure Bag where
  white : ℕ
  red : ℕ
  black : ℕ

/-- Calculates the total number of balls in a bag -/
def Bag.total (b : Bag) : ℕ := b.white + b.red + b.black

/-- Calculates the probability of drawing a ball of a specific color from a bag -/
def probability_color (b : Bag) (color : ℕ) : ℚ :=
  color / b.total

/-- Calculates the probability of drawing balls of different colors from two bags -/
def probability_different_colors (a b : Bag) : ℚ :=
  1 - (probability_color a a.white * probability_color b b.white +
       probability_color a a.red * probability_color b b.red +
       probability_color a a.black * probability_color b b.black)

theorem probability_different_colors_bags :
  let bag_a : Bag := { white := 4, red := 5, black := 6 }
  let bag_b : Bag := { white := 7, red := 6, black := 2 }
  probability_different_colors bag_a bag_b = 31 / 45 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_colors_bags_l3012_301218


namespace NUMINAMATH_CALUDE_chess_group_games_l3012_301282

/-- Represents a chess group with alternating even-odd opponent play --/
structure ChessGroup where
  total_players : ℕ
  even_players : ℕ
  odd_players : ℕ
  alternating_play : Bool

/-- Calculates the total number of games played in the chess group --/
def total_games (cg : ChessGroup) : ℕ :=
  (cg.total_players * cg.even_players) / 2

/-- Theorem stating the total number of games played in a specific chess group setup --/
theorem chess_group_games :
  ∀ (cg : ChessGroup),
    cg.total_players = 12 ∧
    cg.even_players = 6 ∧
    cg.odd_players = 6 ∧
    cg.alternating_play = true →
    total_games cg = 36 := by
  sorry

end NUMINAMATH_CALUDE_chess_group_games_l3012_301282


namespace NUMINAMATH_CALUDE_composition_ratio_l3012_301244

def f (x : ℝ) : ℝ := 3 * x + 1

def g (x : ℝ) : ℝ := 4 * x - 3

theorem composition_ratio : f (g (f 3)) / g (f (g 3)) = 112 / 109 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l3012_301244


namespace NUMINAMATH_CALUDE_hyperbola_equation_proof_l3012_301283

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  /-- The general form of the hyperbola is x²/a² - y²/b² = 1 -/
  a : ℝ
  b : ℝ
  /-- One focus of the hyperbola is at (2,0) -/
  focus_x : a = 2
  /-- The equations of the asymptotes are y = ±√3x -/
  asymptote_slope : b / a = Real.sqrt 3

/-- The equation of the hyperbola with given properties -/
def hyperbola_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, x^2 - y^2 / 3 = 1 ↔ x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Theorem stating that the hyperbola with given properties has the equation x² - y²/3 = 1 -/
theorem hyperbola_equation_proof (h : Hyperbola) : hyperbola_equation h := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_proof_l3012_301283


namespace NUMINAMATH_CALUDE_city_H_highest_increase_l3012_301254

structure City where
  name : String
  pop1990 : ℕ
  pop2000 : ℕ
  event_factor : ℚ

def effective_increase (c : City) : ℚ :=
  (c.pop2000 * c.event_factor - c.pop1990) / c.pop1990

def cities : List City := [
  ⟨"F", 90000, 120000, 11/10⟩,
  ⟨"G", 80000, 110000, 19/20⟩,
  ⟨"H", 70000, 115000, 11/10⟩,
  ⟨"I", 65000, 100000, 49/50⟩,
  ⟨"J", 95000, 145000, 1⟩
]

theorem city_H_highest_increase :
  ∃ c ∈ cities, c.name = "H" ∧
    ∀ c' ∈ cities, effective_increase c ≥ effective_increase c' := by
  sorry

end NUMINAMATH_CALUDE_city_H_highest_increase_l3012_301254


namespace NUMINAMATH_CALUDE_f_monotone_iff_a_range_f_lower_bound_l3012_301201

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - x^2 - a*x

theorem f_monotone_iff_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≤ 2 - 2 * Real.log 2 :=
sorry

theorem f_lower_bound (x : ℝ) (hx : x > 0) :
  f 1 x > 1 - (Real.log 2) / 2 - ((Real.log 2) / 2)^2 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_iff_a_range_f_lower_bound_l3012_301201


namespace NUMINAMATH_CALUDE_min_sum_squares_l3012_301205

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), (∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 = m) ∧
             m = 4 := by
  sorry

#check min_sum_squares

end NUMINAMATH_CALUDE_min_sum_squares_l3012_301205


namespace NUMINAMATH_CALUDE_bowling_team_score_l3012_301262

theorem bowling_team_score (total_score : ℕ) (third_bowler_score : ℕ) : 
  total_score = 810 →
  (third_bowler_score + 3 * third_bowler_score + (3 * third_bowler_score) / 3 = total_score) →
  third_bowler_score = 162 := by
sorry

end NUMINAMATH_CALUDE_bowling_team_score_l3012_301262


namespace NUMINAMATH_CALUDE_sine_cosine_identity_l3012_301294

theorem sine_cosine_identity : Real.sin (20 * π / 180) * Real.cos (110 * π / 180) + Real.cos (160 * π / 180) * Real.sin (70 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_identity_l3012_301294


namespace NUMINAMATH_CALUDE_jennas_tanning_schedule_l3012_301296

/-- Jenna's tanning schedule problem -/
theorem jennas_tanning_schedule 
  (total_time : ℕ) 
  (daily_time : ℕ) 
  (last_two_weeks_time : ℕ) 
  (h1 : total_time = 200)
  (h2 : daily_time = 30)
  (h3 : last_two_weeks_time = 80) :
  (total_time - last_two_weeks_time) / (2 * daily_time) = 2 := by
  sorry

end NUMINAMATH_CALUDE_jennas_tanning_schedule_l3012_301296


namespace NUMINAMATH_CALUDE_count_cubic_polynomials_satisfying_property_l3012_301257

/-- A polynomial function of degree 3 -/
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := fun x ↦ a * x^3 + b * x^2 + c * x + d

/-- The property that f(x)f(-x) = f(x³) -/
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x, f x * f (-x) = f (x^3)

/-- The main theorem stating that there are exactly 16 cubic polynomials satisfying the property -/
theorem count_cubic_polynomials_satisfying_property :
  ∃! (s : Finset (ℝ → ℝ)),
    (∀ f ∈ s, ∃ a b c d, f = CubicPolynomial a b c d ∧ SatisfiesProperty f) ∧
    Finset.card s = 16 := by sorry

end NUMINAMATH_CALUDE_count_cubic_polynomials_satisfying_property_l3012_301257


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l3012_301272

theorem max_gcd_13n_plus_4_8n_plus_3 :
  ∃ (k : ℕ), k > 0 ∧ gcd (13 * k + 4) (8 * k + 3) = 9 ∧
  ∀ (n : ℕ), n > 0 → gcd (13 * n + 4) (8 * n + 3) ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l3012_301272


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3012_301234

theorem rectangular_field_area (L W : ℝ) : 
  L = 10 →                   -- One side is 10 feet
  2 * W + L = 146 →          -- Total fencing is 146 feet
  L * W = 680 :=             -- Area of the field is 680 square feet
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3012_301234


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l3012_301230

theorem min_value_quadratic_sum (x y s : ℝ) (h : x + y = s) :
  (∀ a b : ℝ, a + b = s → 3 * a^2 + 2 * b^2 ≥ (6/5) * s^2) ∧
  ∃ x₀ y₀ : ℝ, x₀ + y₀ = s ∧ 3 * x₀^2 + 2 * y₀^2 = (6/5) * s^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l3012_301230


namespace NUMINAMATH_CALUDE_tom_four_times_cindy_l3012_301252

/-- Tom's current age -/
def t : ℕ := sorry

/-- Cindy's current age -/
def c : ℕ := sorry

/-- In five years, Tom will be twice as old as Cindy -/
axiom future_condition : t + 5 = 2 * (c + 5)

/-- Thirteen years ago, Tom was three times as old as Cindy -/
axiom past_condition : t - 13 = 3 * (c - 13)

/-- The number of years ago when Tom was four times as old as Cindy -/
def years_ago : ℕ := sorry

theorem tom_four_times_cindy : years_ago = 19 := by sorry

end NUMINAMATH_CALUDE_tom_four_times_cindy_l3012_301252


namespace NUMINAMATH_CALUDE_log_equation_solution_l3012_301225

-- Define the logarithm function
noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- Define the property of being a non-square
def is_non_square (x : ℚ) : Prop := ∀ n : ℕ, x ≠ (n : ℚ)^2

-- Define the property of being a non-cube
def is_non_cube (x : ℚ) : Prop := ∀ n : ℕ, x ≠ (n : ℚ)^3

-- Define the property of being non-integral
def is_non_integral (x : ℚ) : Prop := ∀ n : ℤ, x ≠ n

-- Main theorem
theorem log_equation_solution :
  ∃ x : ℝ, 
    log_base (3 * x) 343 = x ∧ 
    x = 4 / 3 ∧
    (∃ q : ℚ, x = q) ∧
    is_non_square (4 / 3) ∧
    is_non_cube (4 / 3) ∧
    is_non_integral (4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3012_301225


namespace NUMINAMATH_CALUDE_basket_average_price_l3012_301241

/-- Given 4 baskets with an average cost of $4 and a fifth basket costing $8,
    the average price of all 5 baskets is $4.80. -/
theorem basket_average_price (num_initial_baskets : ℕ) (initial_avg_cost : ℚ) (fifth_basket_cost : ℚ) :
  num_initial_baskets = 4 →
  initial_avg_cost = 4 →
  fifth_basket_cost = 8 →
  (num_initial_baskets * initial_avg_cost + fifth_basket_cost) / (num_initial_baskets + 1) = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_basket_average_price_l3012_301241


namespace NUMINAMATH_CALUDE_function_range_theorem_l3012_301281

def monotone_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f y ≤ f x

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem function_range_theorem (f : ℝ → ℝ) 
    (h_mono : monotone_decreasing f)
    (h_odd : odd_function f)
    (h_f1 : f 1 = -1) :
    {x : ℝ | -1 ≤ f (x - 2) ∧ f (x - 2) ≤ 1} = Set.Icc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_function_range_theorem_l3012_301281


namespace NUMINAMATH_CALUDE_samantha_bedtime_l3012_301270

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  deriving Repr

/-- Calculates the bedtime given wake-up time and sleep duration -/
def calculateBedtime (wakeUpTime : Time) (sleepDuration : Nat) : Time :=
  let totalMinutes := wakeUpTime.hour * 60 + wakeUpTime.minute
  let bedtimeMinutes := (totalMinutes - sleepDuration * 60 + 24 * 60) % (24 * 60)
  { hour := bedtimeMinutes / 60, minute := bedtimeMinutes % 60 }

theorem samantha_bedtime :
  let wakeUpTime : Time := { hour := 11, minute := 0 }
  let sleepDuration : Nat := 6
  calculateBedtime wakeUpTime sleepDuration = { hour := 5, minute := 0 } := by
  sorry

end NUMINAMATH_CALUDE_samantha_bedtime_l3012_301270


namespace NUMINAMATH_CALUDE_two_faces_same_sides_l3012_301238

/-- A face of a polyhedron -/
structure Face where
  sides : ℕ
  sides_ge_3 : sides ≥ 3

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  faces : Finset Face
  nonempty : faces.Nonempty

theorem two_faces_same_sides (P : ConvexPolyhedron) : 
  ∃ f₁ f₂ : Face, f₁ ∈ P.faces ∧ f₂ ∈ P.faces ∧ f₁ ≠ f₂ ∧ f₁.sides = f₂.sides :=
sorry

end NUMINAMATH_CALUDE_two_faces_same_sides_l3012_301238


namespace NUMINAMATH_CALUDE_intersection_values_l3012_301202

-- Define the line
def line (k : ℝ) (x : ℝ) : ℝ := k * x - 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

-- Define the condition for intersection at a single point
def single_intersection (k : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, hyperbola p.1 p.2 ∧ p.2 = line k p.1

-- Theorem statement
theorem intersection_values :
  {k : ℝ | single_intersection k} = {-1, 1, -Real.sqrt 5 / 2, Real.sqrt 5 / 2} :=
sorry

end NUMINAMATH_CALUDE_intersection_values_l3012_301202


namespace NUMINAMATH_CALUDE_leap_year_statistics_l3012_301203

def leap_year_data : List ℕ := sorry

def median_of_modes (data : List ℕ) : ℚ := sorry

def median (data : List ℕ) : ℚ := sorry

def mean (data : List ℕ) : ℚ := sorry

theorem leap_year_statistics :
  let d := median_of_modes leap_year_data
  let M := median leap_year_data
  let μ := mean leap_year_data
  d < M ∧ M < μ := by sorry

end NUMINAMATH_CALUDE_leap_year_statistics_l3012_301203


namespace NUMINAMATH_CALUDE_smallest_n_for_more_than_half_remaining_l3012_301273

def outer_layer_cubes (n : ℕ) : ℕ := 6 * n^2 - 12 * n + 8

def remaining_cubes (n : ℕ) : ℕ := n^3 - outer_layer_cubes n

theorem smallest_n_for_more_than_half_remaining : 
  (∀ k : ℕ, k < 10 → 2 * remaining_cubes k ≤ k^3) ∧
  (2 * remaining_cubes 10 > 10^3) := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_more_than_half_remaining_l3012_301273


namespace NUMINAMATH_CALUDE_point_on_unit_sphere_l3012_301217

theorem point_on_unit_sphere (x y z : ℝ) : 
  let r := Real.sqrt (x^2 + y^2 + z^2)
  let s := y / r
  let c := x / r
  let t := z / r
  s^2 + c^2 + t^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_unit_sphere_l3012_301217


namespace NUMINAMATH_CALUDE_five_people_handshakes_l3012_301253

/-- The number of handshakes in a group of n people, where each person shakes hands with every other person exactly once. -/
def number_of_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that in a group of 5 people, where each person shakes hands with every other person exactly once, the total number of handshakes is 10. -/
theorem five_people_handshakes : number_of_handshakes 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_five_people_handshakes_l3012_301253


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3012_301250

theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * n - 2) / (n * (n + 1) * (n + 2))) = 1/2 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3012_301250


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3012_301263

theorem polynomial_division_remainder : ∃ q : Polynomial ℂ, 
  (Y : Polynomial ℂ)^55 + Y^40 + Y^25 + Y^10 + 1 = 
  (Y^5 + Y^4 + Y^3 + Y^2 + Y + 1) * q + (2 * Y + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3012_301263


namespace NUMINAMATH_CALUDE_modified_code_system_distinct_symbols_l3012_301297

/-- The number of possible symbols (dot, dash, or blank) -/
def num_symbols : ℕ := 3

/-- The maximum length of a sequence -/
def max_length : ℕ := 3

/-- The number of distinct symbols for a given sequence length -/
def distinct_symbols (length : ℕ) : ℕ := num_symbols ^ length

/-- The total number of distinct symbols for sequences of length 1 to max_length -/
def total_distinct_symbols : ℕ :=
  (Finset.range max_length).sum (λ i => distinct_symbols (i + 1))

theorem modified_code_system_distinct_symbols :
  total_distinct_symbols = 39 := by
  sorry

end NUMINAMATH_CALUDE_modified_code_system_distinct_symbols_l3012_301297


namespace NUMINAMATH_CALUDE_x_gt_3_sufficient_not_necessary_for_x_sq_gt_9_l3012_301277

theorem x_gt_3_sufficient_not_necessary_for_x_sq_gt_9 :
  (∀ x : ℝ, x > 3 → x^2 > 9) ∧ 
  ¬(∀ x : ℝ, x^2 > 9 → x > 3) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_3_sufficient_not_necessary_for_x_sq_gt_9_l3012_301277


namespace NUMINAMATH_CALUDE_farm_heads_count_l3012_301267

/-- Represents a farm with hens and cows -/
structure Farm where
  hens : ℕ
  cows : ℕ

/-- The total number of feet on the farm -/
def totalFeet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows

/-- The total number of heads (animals) on the farm -/
def totalHeads (f : Farm) : ℕ := f.hens + f.cows

/-- Theorem: Given a farm with 24 hens and 144 total feet, the total number of heads is 48 -/
theorem farm_heads_count (f : Farm) 
  (hen_count : f.hens = 24) 
  (feet_count : totalFeet f = 144) : 
  totalHeads f = 48 := by
  sorry


end NUMINAMATH_CALUDE_farm_heads_count_l3012_301267


namespace NUMINAMATH_CALUDE_cupcake_package_size_l3012_301222

theorem cupcake_package_size :
  ∀ (small_package_size : ℕ) (total_cupcakes : ℕ) (small_packages : ℕ) (larger_package_size : ℕ),
    small_package_size = 10 →
    total_cupcakes = 100 →
    small_packages = 4 →
    total_cupcakes = small_package_size * small_packages + larger_package_size →
    larger_package_size = 60 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_package_size_l3012_301222


namespace NUMINAMATH_CALUDE_five_digit_permutations_l3012_301242

/-- The number of permutations of a multiset with repeated elements -/
def multiset_permutations (total : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial total / (repetitions.map Nat.factorial).prod

/-- The number of different five-digit integers formed using 1, 1, 1, 8, and 8 -/
theorem five_digit_permutations : multiset_permutations 5 [3, 2] = 10 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_permutations_l3012_301242


namespace NUMINAMATH_CALUDE_remainder_theorem_l3012_301286

/-- The polynomial being divided -/
def p (x : ℝ) : ℝ := x^6 - x^5 - x^4 + x^3 + x^2

/-- The divisor -/
def d (x : ℝ) : ℝ := (x^2 - 4) * (x + 1)

/-- The remainder -/
def r (x : ℝ) : ℝ := 15 * x^2 - 12 * x - 24

/-- Theorem stating that r is the remainder when p is divided by d -/
theorem remainder_theorem : ∃ q : ℝ → ℝ, ∀ x : ℝ, p x = d x * q x + r x := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3012_301286


namespace NUMINAMATH_CALUDE_coprime_implies_divisible_power_minus_one_l3012_301209

theorem coprime_implies_divisible_power_minus_one (a n : ℕ) (h : Nat.Coprime a n) :
  ∃ m : ℕ, n ∣ (a^m - 1) := by
sorry

end NUMINAMATH_CALUDE_coprime_implies_divisible_power_minus_one_l3012_301209


namespace NUMINAMATH_CALUDE_total_wax_sticks_is_42_l3012_301233

/-- Calculates the total number of wax sticks used for animal sculptures --/
def total_wax_sticks (large_animal_wax : ℕ) (small_animal_wax : ℕ) (small_animal_total_wax : ℕ) : ℕ :=
  let small_animals := small_animal_total_wax / small_animal_wax
  let large_animals := small_animals / 5
  let large_animal_total_wax := large_animals * large_animal_wax
  small_animal_total_wax + large_animal_total_wax

/-- Theorem stating that the total number of wax sticks used is 42 --/
theorem total_wax_sticks_is_42 :
  total_wax_sticks 6 3 30 = 42 :=
by
  sorry

#eval total_wax_sticks 6 3 30

end NUMINAMATH_CALUDE_total_wax_sticks_is_42_l3012_301233


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3012_301278

-- Define the solution set
def solution_set : Set ℝ := {x | x < -1 ∨ x > 1}

-- Statement of the theorem
theorem inequality_equivalence (x : ℝ) : x > (1 / x) ↔ x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3012_301278


namespace NUMINAMATH_CALUDE_robert_books_read_l3012_301266

def reading_speed : ℕ := 120
def book_length : ℕ := 360
def available_time : ℕ := 8

def books_read (speed pages time : ℕ) : ℕ :=
  (speed * time) / pages

theorem robert_books_read :
  books_read reading_speed book_length available_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_books_read_l3012_301266


namespace NUMINAMATH_CALUDE_tram_speed_l3012_301280

/-- Proves that a tram with constant speed passing an observer in 2 seconds
    and traversing a 96-meter tunnel in 10 seconds has a speed of 12 m/s. -/
theorem tram_speed (v : ℝ) 
  (h1 : v * 2 = v * 2)  -- Tram passes observer in 2 seconds
  (h2 : v * 10 = 96 + v * 2)  -- Tram traverses 96-meter tunnel in 10 seconds
  : v = 12 := by
  sorry

end NUMINAMATH_CALUDE_tram_speed_l3012_301280


namespace NUMINAMATH_CALUDE_open_cells_are_perfect_squares_l3012_301211

/-- Represents whether a cell is open (true) or closed (false) -/
def CellState := Bool

/-- The state of a cell after the jailer's procedure -/
def final_cell_state (n : ℕ) : CellState :=
  sorry

/-- A number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- The main theorem: a cell remains open iff its number is a perfect square -/
theorem open_cells_are_perfect_squares (n : ℕ) :
  final_cell_state n = true ↔ is_perfect_square n :=
  sorry

end NUMINAMATH_CALUDE_open_cells_are_perfect_squares_l3012_301211


namespace NUMINAMATH_CALUDE_min_tea_time_l3012_301220

def wash_kettle : ℕ := 1
def boil_water : ℕ := 10
def wash_cups : ℕ := 2
def get_leaves : ℕ := 1
def brew_tea : ℕ := 1

theorem min_tea_time : 
  ∃ (arrangement : ℕ), 
    arrangement = max boil_water (wash_kettle + wash_cups + get_leaves) + brew_tea ∧
    arrangement = 11 ∧
    ∀ (other_arrangement : ℕ), other_arrangement ≥ arrangement :=
by sorry

end NUMINAMATH_CALUDE_min_tea_time_l3012_301220


namespace NUMINAMATH_CALUDE_sedans_sold_prediction_l3012_301245

/-- The ratio of sports cars to sedans -/
def car_ratio : ℚ := 3 / 5

/-- The number of sports cars predicted to be sold -/
def sports_cars_sold : ℕ := 36

/-- The number of sedans expected to be sold -/
def sedans_sold : ℕ := 60

/-- Theorem stating the relationship between sports cars and sedans sold -/
theorem sedans_sold_prediction :
  (car_ratio * sports_cars_sold : ℚ) = sedans_sold := by sorry

end NUMINAMATH_CALUDE_sedans_sold_prediction_l3012_301245


namespace NUMINAMATH_CALUDE_zero_exponent_rule_l3012_301248

theorem zero_exponent_rule (x : ℚ) (h : x ≠ 0) : x ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_rule_l3012_301248


namespace NUMINAMATH_CALUDE_line_in_first_third_quadrants_positive_slope_l3012_301236

/-- A line passing through the first and third quadrants -/
structure LineInFirstThirdQuadrants where
  k : ℝ
  k_neq_zero : k ≠ 0
  passes_through_first_third : ∀ x y : ℝ, y = k * x → 
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))

/-- Theorem: If a line y = kx passes through the first and third quadrants, then k > 0 -/
theorem line_in_first_third_quadrants_positive_slope 
  (line : LineInFirstThirdQuadrants) : line.k > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_in_first_third_quadrants_positive_slope_l3012_301236


namespace NUMINAMATH_CALUDE_probability_of_all_successes_l3012_301289

-- Define the number of trials
def n : ℕ := 7

-- Define the probability of success in each trial
def p : ℚ := 2/7

-- Define the number of successes we're interested in
def k : ℕ := 7

-- State the theorem
theorem probability_of_all_successes :
  (n.choose k) * p^k * (1 - p)^(n - k) = 128/823543 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_all_successes_l3012_301289


namespace NUMINAMATH_CALUDE_infinite_sum_of_digits_not_exceeding_two_l3012_301210

theorem infinite_sum_of_digits_not_exceeding_two (n : ℕ) :
  ∃ (x y z : ℤ), 4 * x^4 + y^4 - z^2 + 4 * x * y * z = 2 * (10 : ℤ)^(2 * n + 2) := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_of_digits_not_exceeding_two_l3012_301210


namespace NUMINAMATH_CALUDE_two_zeros_implies_a_range_l3012_301240

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a + 2^x else (1/2) * x + a

-- Theorem statement
theorem two_zeros_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧
    ∀ x : ℝ, x ≠ x₁ ∧ x ≠ x₂ → f a x ≠ 0) →
  a ∈ Set.Icc (-2) (-1/2) :=
sorry

end NUMINAMATH_CALUDE_two_zeros_implies_a_range_l3012_301240


namespace NUMINAMATH_CALUDE_cupcake_cost_l3012_301260

/-- Proves that the cost of a cupcake is 40 cents given initial amount, juice box cost, and remaining amount --/
theorem cupcake_cost (initial_amount : ℕ) (juice_cost : ℕ) (remaining : ℕ) :
  initial_amount = 75 →
  juice_cost = 27 →
  remaining = 8 →
  initial_amount - juice_cost - remaining = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_cupcake_cost_l3012_301260


namespace NUMINAMATH_CALUDE_not_outperformed_iff_ge_five_l3012_301298

/-- A directed graph representing a table tennis tournament. -/
structure TournamentGraph (n : ℕ) where
  (edges : Fin n → Fin n → Prop)
  (complete : ∀ i j : Fin n, i ≠ j → edges i j ∨ edges j i)

/-- Player i is not out-performed by player j. -/
def not_outperformed {n : ℕ} (G : TournamentGraph n) (i j : Fin n) : Prop :=
  ∃ k : Fin n, G.edges i k ∧ ¬G.edges j k

/-- The tournament satisfies the not out-performed condition for all players. -/
def all_not_outperformed (n : ℕ) : Prop :=
  ∃ G : TournamentGraph n, ∀ i j : Fin n, i ≠ j → not_outperformed G i j

/-- The main theorem: the not out-performed condition holds if and only if n ≥ 5. -/
theorem not_outperformed_iff_ge_five :
  ∀ n : ℕ, n ≥ 3 → (all_not_outperformed n ↔ n ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_not_outperformed_iff_ge_five_l3012_301298


namespace NUMINAMATH_CALUDE_enid_sweaters_count_l3012_301246

/-- The number of scarves Aaron makes -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Aaron makes -/
def aaron_sweaters : ℕ := 5

/-- The number of balls of wool used for one scarf -/
def wool_per_scarf : ℕ := 3

/-- The number of balls of wool used for one sweater -/
def wool_per_sweater : ℕ := 4

/-- The total number of balls of wool used by both Enid and Aaron -/
def total_wool : ℕ := 82

/-- The number of sweaters Enid makes -/
def enid_sweaters : ℕ := (total_wool - (aaron_scarves * wool_per_scarf + aaron_sweaters * wool_per_sweater)) / wool_per_sweater

theorem enid_sweaters_count :
  enid_sweaters = 8 := by sorry

end NUMINAMATH_CALUDE_enid_sweaters_count_l3012_301246


namespace NUMINAMATH_CALUDE_last_number_not_one_l3012_301235

def board_sum : ℕ := (2012 * 2013) / 2

theorem last_number_not_one :
  ∀ (operations : ℕ) (final_number : ℕ),
    (operations < 2011 → final_number ≠ 1) ∧
    (operations = 2011 → final_number % 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_last_number_not_one_l3012_301235


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3012_301259

theorem simplify_and_evaluate (x y : ℝ) (hx : x = 2) (hy : y = -0.5) :
  2 * (2 * x - 3 * y) - (3 * x + 2 * y + 1) = 5 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3012_301259


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3012_301295

/-- The polynomial function f(x) = x^10 + 5x^9 + 28x^8 + 145x^7 - 1897x^6 -/
def f (x : ℝ) : ℝ := x^10 + 5*x^9 + 28*x^8 + 145*x^7 - 1897*x^6

/-- Theorem: The equation f(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ f x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3012_301295


namespace NUMINAMATH_CALUDE_sport_formulation_water_amount_l3012_301212

/-- Represents the ratio of flavoring to corn syrup to water in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  ⟨1, 12, 30⟩

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio :=
  ⟨1, 4, 60⟩

/-- Calculates the amount of water given the amount of corn syrup and the drink ratio -/
def water_amount (corn_syrup_amount : ℚ) (ratio : DrinkRatio) : ℚ :=
  (corn_syrup_amount * ratio.water) / ratio.corn_syrup

theorem sport_formulation_water_amount :
  water_amount 3 sport_ratio = 45 := by
  sorry

end NUMINAMATH_CALUDE_sport_formulation_water_amount_l3012_301212


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l3012_301265

theorem no_real_roots_quadratic (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + 1 ≠ 0) ↔ -2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l3012_301265


namespace NUMINAMATH_CALUDE_distance_of_opposite_numbers_a_and_neg_a_are_opposite_l3012_301290

-- Define the concept of opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- Define the distance from origin to a point on the number line
def distance_from_origin (a : ℝ) : ℝ := |a|

-- Statement 1: The distance from the origin to the points corresponding to two opposite numbers on the number line is equal
theorem distance_of_opposite_numbers (a : ℝ) : 
  distance_from_origin a = distance_from_origin (-a) := by sorry

-- Statement 2: For any real number a, a and -a are opposite numbers to each other
theorem a_and_neg_a_are_opposite (a : ℝ) : 
  are_opposite a (-a) := by sorry

end NUMINAMATH_CALUDE_distance_of_opposite_numbers_a_and_neg_a_are_opposite_l3012_301290


namespace NUMINAMATH_CALUDE_shop_width_calculation_l3012_301216

/-- Given a rectangular shop with the following properties:
  - Monthly rent is 3600 (in some currency unit)
  - Length is 20 feet
  - Annual rent per square foot is 144 (in the same currency unit)
  This theorem proves that the width of the shop is 15 feet. -/
theorem shop_width_calculation (monthly_rent : ℕ) (length : ℕ) (annual_rent_per_sqft : ℕ) :
  monthly_rent = 3600 →
  length = 20 →
  annual_rent_per_sqft = 144 →
  (monthly_rent * 12) / annual_rent_per_sqft / length = 15 := by
  sorry

end NUMINAMATH_CALUDE_shop_width_calculation_l3012_301216


namespace NUMINAMATH_CALUDE_trig_identity_l3012_301299

theorem trig_identity (α : Real) (h : α ∈ Set.Ioo (-π) (-π/2)) : 
  Real.sqrt ((1 + Real.cos α) / (1 - Real.cos α)) - 
  Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = 
  2 / Real.tan α := by sorry

end NUMINAMATH_CALUDE_trig_identity_l3012_301299


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3012_301292

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 3| > 1} = Set.Iio (-4) ∪ Set.Ioi (-2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3012_301292


namespace NUMINAMATH_CALUDE_remaining_miles_l3012_301219

def total_journey : ℕ := 1200
def miles_driven : ℕ := 215

theorem remaining_miles :
  total_journey - miles_driven = 985 := by sorry

end NUMINAMATH_CALUDE_remaining_miles_l3012_301219
