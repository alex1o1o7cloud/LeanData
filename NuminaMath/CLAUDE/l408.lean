import Mathlib

namespace NUMINAMATH_CALUDE_complex_power_equality_l408_40818

theorem complex_power_equality : (3 * Complex.cos (π / 6) + 3 * Complex.I * Complex.sin (π / 6)) ^ 4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_equality_l408_40818


namespace NUMINAMATH_CALUDE_complex_magnitude_l408_40803

theorem complex_magnitude (z : ℂ) (h : z - 2 * Complex.I = 1 + z * Complex.I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l408_40803


namespace NUMINAMATH_CALUDE_intersection_equality_iff_t_range_l408_40819

/-- The set M -/
def M : Set ℝ := {x | -2 < x ∧ x < 5}

/-- The set N parameterized by t -/
def N (t : ℝ) : Set ℝ := {x | 2 - t < x ∧ x < 2 * t + 1}

/-- Theorem stating the equivalence between M ∩ N = N and t ∈ (-∞, 2] -/
theorem intersection_equality_iff_t_range :
  ∀ t : ℝ, (M ∩ N t = N t) ↔ t ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_iff_t_range_l408_40819


namespace NUMINAMATH_CALUDE_polynomial_factorization_l408_40814

theorem polynomial_factorization (a b c : ℝ) : 
  a*(b - c)^4 + b*(c - a)^4 + c*(a - b)^4 = (a - b)*(b - c)*(c - a)*(a*b^2 + a*c^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l408_40814


namespace NUMINAMATH_CALUDE_jenny_egg_distribution_l408_40800

theorem jenny_egg_distribution (n : ℕ) : 
  n ∣ 18 ∧ n ∣ 24 ∧ n ≥ 4 → n = 6 :=
by sorry

end NUMINAMATH_CALUDE_jenny_egg_distribution_l408_40800


namespace NUMINAMATH_CALUDE_equation_solution_is_origin_l408_40831

theorem equation_solution_is_origin (x y : ℝ) : 
  (x + y)^2 = 2 * (x^2 + y^2) ↔ x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_is_origin_l408_40831


namespace NUMINAMATH_CALUDE_parabola_single_intersection_parabola_y_decreases_l408_40860

def parabola (x m : ℝ) : ℝ := -2 * x^2 + 4 * x + m

theorem parabola_single_intersection (m : ℝ) :
  (∃! x, parabola x m = 0) ↔ m = -2 := sorry

theorem parabola_y_decreases (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  parabola x₁ m = y₁ →
  parabola x₂ m = y₂ →
  x₁ > x₂ →
  x₂ > 2 →
  y₁ < y₂ := sorry

end NUMINAMATH_CALUDE_parabola_single_intersection_parabola_y_decreases_l408_40860


namespace NUMINAMATH_CALUDE_odd_function_properties_l408_40870

-- Define an odd function f
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property f(x+1) = f(x-1)
def property_f (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f (x - 1)

-- Define periodicity with period 2
def periodic_2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

-- Define symmetry about (k, 0) for all integer k
def symmetric_about_int (f : ℝ → ℝ) : Prop :=
  ∀ (k : ℤ) (x : ℝ), f (2 * k - x) = -f x

theorem odd_function_properties (f : ℝ → ℝ) 
  (h_odd : odd_function f) (h_prop : property_f f) :
  periodic_2 f ∧ symmetric_about_int f := by
  sorry

end NUMINAMATH_CALUDE_odd_function_properties_l408_40870


namespace NUMINAMATH_CALUDE_perpendicular_lines_intersection_l408_40857

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℚ) : Prop := m₁ * m₂ = -1

/-- A point (x, y) lies on a line ax + by + c = 0 -/
def point_on_line (x y a b c : ℚ) : Prop := a * x + b * y + c = 0

/-- Given two perpendicular lines and their intersection point, prove p - m - n = 4 -/
theorem perpendicular_lines_intersection (m n p : ℚ) : 
  perpendicular (-2/m) (3/2) →
  point_on_line 2 p 2 m (-1) →
  point_on_line 2 p 3 (-2) n →
  p - m - n = 4 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_intersection_l408_40857


namespace NUMINAMATH_CALUDE_focal_distance_l408_40845

/-- Represents an ellipse with focal points and vertices -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Focal distance from center

/-- Properties of the ellipse based on given conditions -/
def ellipse_properties (e : Ellipse) : Prop :=
  e.a - e.c = 1.5 ∧  -- |F₂A| = 1.5
  2 * e.a = 5.4 ∧    -- |BC| = 5.4
  e.a^2 = e.b^2 + e.c^2

/-- Theorem stating the distance between focal points -/
theorem focal_distance (e : Ellipse) 
  (h : ellipse_properties e) : 2 * e.a - (e.a - e.c) = 13.5 := by
  sorry

#check focal_distance

end NUMINAMATH_CALUDE_focal_distance_l408_40845


namespace NUMINAMATH_CALUDE_house_representatives_difference_l408_40812

theorem house_representatives_difference (total : Nat) (democrats : Nat) :
  total = 434 →
  democrats = 202 →
  democrats < total - democrats →
  total - 2 * democrats = 30 := by
sorry

end NUMINAMATH_CALUDE_house_representatives_difference_l408_40812


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_of_nine_to_nine_minus_one_l408_40890

theorem sum_of_prime_factors_of_nine_to_nine_minus_one : 
  ∃ (p₁ p₂ p₃ p₄ p₅ p₆ : ℕ), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ Prime p₆ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ p₁ ≠ p₆ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ p₂ ≠ p₆ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ p₃ ≠ p₆ ∧
    p₄ ≠ p₅ ∧ p₄ ≠ p₆ ∧
    p₅ ≠ p₆ ∧
    (9^9 - 1 : ℕ) = p₁ * p₂ * p₃ * p₄ * p₅ * p₆ ∧
    p₁ + p₂ + p₃ + p₄ + p₅ + p₆ = 835 := by
  sorry

#eval 9^9 - 1

end NUMINAMATH_CALUDE_sum_of_prime_factors_of_nine_to_nine_minus_one_l408_40890


namespace NUMINAMATH_CALUDE_movie_trip_cost_l408_40801

/-- The total cost of a movie trip for a group of adults and children -/
def total_cost (num_adults num_children : ℕ) (adult_ticket_price child_ticket_price concession_cost : ℚ) : ℚ :=
  num_adults * adult_ticket_price + num_children * child_ticket_price + concession_cost

/-- Theorem stating that the total cost for the given group is $76 -/
theorem movie_trip_cost : 
  total_cost 5 2 10 7 12 = 76 := by
  sorry

end NUMINAMATH_CALUDE_movie_trip_cost_l408_40801


namespace NUMINAMATH_CALUDE_triangular_weight_is_60_l408_40837

/-- The weight of a rectangular weight in grams -/
def rectangular_weight : ℝ := 90

/-- The weight of a round weight in grams -/
def round_weight : ℝ := 30

/-- The weight of a triangular weight in grams -/
def triangular_weight : ℝ := 60

/-- First balance condition: 1 round + 1 triangular = 3 round -/
axiom balance1 : round_weight + triangular_weight = 3 * round_weight

/-- Second balance condition: 4 round + 1 triangular = 1 triangular + 1 round + 1 rectangular -/
axiom balance2 : 4 * round_weight + triangular_weight = triangular_weight + round_weight + rectangular_weight

theorem triangular_weight_is_60 : triangular_weight = 60 := by sorry

end NUMINAMATH_CALUDE_triangular_weight_is_60_l408_40837


namespace NUMINAMATH_CALUDE_train_length_calculation_l408_40888

theorem train_length_calculation (speed1 speed2 speed3 : ℝ) (time1 time2 time3 : ℝ) :
  let length1 := (speed1 / 3600) * time1
  let length2 := (speed2 / 3600) * time2
  let length3 := (speed3 / 3600) * time3
  speed1 = 300 ∧ speed2 = 250 ∧ speed3 = 350 ∧
  time1 = 33 ∧ time2 = 44 ∧ time3 = 28 →
  length1 + length2 + length3 = 8.52741 :=
by sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l408_40888


namespace NUMINAMATH_CALUDE_cricketer_new_average_l408_40854

/-- Represents a cricketer's performance -/
structure CricketerPerformance where
  innings : ℕ
  lastInningScore : ℕ
  averageIncrease : ℕ

/-- Calculates the new average score after the last inning -/
def newAverageScore (performance : CricketerPerformance) : ℚ :=
  sorry

/-- Theorem stating the cricketer's new average score -/
theorem cricketer_new_average
  (performance : CricketerPerformance)
  (h1 : performance.innings = 19)
  (h2 : performance.lastInningScore = 99)
  (h3 : performance.averageIncrease = 4) :
  newAverageScore performance = 27 :=
sorry

end NUMINAMATH_CALUDE_cricketer_new_average_l408_40854


namespace NUMINAMATH_CALUDE_rectangle_to_square_perimeter_l408_40882

/-- Given a rectangle that forms a square when its width is doubled and length is halved,
    this theorem relates the perimeter of the resulting square to the original rectangle's perimeter. -/
theorem rectangle_to_square_perimeter (w l P : ℝ) 
  (h1 : w > 0) 
  (h2 : l > 0)
  (h3 : 2 * w = l / 2)  -- Condition for forming a square
  (h4 : P = 4 * (2 * w)) -- Perimeter of the square
  : 2 * (w + l) = 5/4 * P := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_perimeter_l408_40882


namespace NUMINAMATH_CALUDE_min_y_intercept_l408_40898

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 12*x + 11

-- Define the y-intercept of the tangent line as a function of x
def r (x : ℝ) : ℝ := -2*x^3 + 6*x^2 - 6

-- Theorem statement
theorem min_y_intercept :
  ∀ x ∈ Set.Icc 0 2, r 0 ≤ r x :=
sorry

end NUMINAMATH_CALUDE_min_y_intercept_l408_40898


namespace NUMINAMATH_CALUDE_trig_comparison_l408_40852

open Real

theorem trig_comparison : 
  sin (π/5) = sin (4*π/5) ∧ cos (π/5) > cos (4*π/5) := by
  have h1 : 0 < π/5 ∧ π/5 < 4*π/5 ∧ 4*π/5 < π := by sorry
  have h2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π → cos x > cos y := by sorry
  sorry

end NUMINAMATH_CALUDE_trig_comparison_l408_40852


namespace NUMINAMATH_CALUDE_intersection_line_l408_40897

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := (x+4)^2 + (y+3)^2 = 8

-- Define the line
def line (x y : ℝ) : Prop := 4*x + 3*y + 13 = 0

-- Theorem statement
theorem intersection_line :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_l408_40897


namespace NUMINAMATH_CALUDE_decimal_addition_subtraction_l408_40844

theorem decimal_addition_subtraction : 0.5 + 0.03 - 0.004 + 0.007 = 0.533 := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_subtraction_l408_40844


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l408_40895

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 1 > 0} = {x : ℝ | x < -1/2 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l408_40895


namespace NUMINAMATH_CALUDE_steven_peach_count_l408_40858

/-- 
Given that:
- Jake has 84 more apples than Steven
- Jake has 10 fewer peaches than Steven
- Steven has 52 apples
- Jake has 3 peaches

Prove that Steven has 13 peaches.
-/
theorem steven_peach_count (jake_apple_diff : ℕ) (jake_peach_diff : ℕ) 
  (steven_apples : ℕ) (jake_peaches : ℕ) 
  (h1 : jake_apple_diff = 84)
  (h2 : jake_peach_diff = 10)
  (h3 : steven_apples = 52)
  (h4 : jake_peaches = 3) : 
  jake_peaches + jake_peach_diff = 13 := by
  sorry

end NUMINAMATH_CALUDE_steven_peach_count_l408_40858


namespace NUMINAMATH_CALUDE_river_crossing_trips_l408_40840

/-- Represents the number of trips required to transport one adult across the river -/
def trips_per_adult : ℕ := 4

/-- Represents the total number of adults to be transported -/
def total_adults : ℕ := 358

/-- Calculates the total number of trips required to transport all adults -/
def total_trips : ℕ := trips_per_adult * total_adults

/-- Theorem stating that the total number of trips is 1432 -/
theorem river_crossing_trips : total_trips = 1432 := by
  sorry

end NUMINAMATH_CALUDE_river_crossing_trips_l408_40840


namespace NUMINAMATH_CALUDE_min_value_implies_m_l408_40848

/-- Given a function f(x) = x + m / (x - 2) where x > 2 and m > 0,
    if the minimum value of f(x) is 6, then m = 4. -/
theorem min_value_implies_m (m : ℝ) (h_m_pos : m > 0) :
  (∀ x > 2, x + m / (x - 2) ≥ 6) ∧
  (∃ x > 2, x + m / (x - 2) = 6) →
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_implies_m_l408_40848


namespace NUMINAMATH_CALUDE_workshop_workers_l408_40815

theorem workshop_workers (avg_salary : ℝ) (tech_count : ℕ) (tech_avg_salary : ℝ) (non_tech_avg_salary : ℝ)
  (h1 : avg_salary = 8000)
  (h2 : tech_count = 7)
  (h3 : tech_avg_salary = 10000)
  (h4 : non_tech_avg_salary = 6000) :
  ∃ (total_workers : ℕ), total_workers = 14 ∧
  (tech_count * tech_avg_salary + (total_workers - tech_count) * non_tech_avg_salary) / total_workers = avg_salary :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_l408_40815


namespace NUMINAMATH_CALUDE_max_m_value_l408_40834

/-- Given a > 0, proves that the maximum value of m is e^(1/2) when the tangents of 
    y = x²/2 + ax and y = 2a²ln(x) + m coincide at their intersection point. -/
theorem max_m_value (a : ℝ) (h_a : a > 0) : 
  let C₁ : ℝ → ℝ := λ x => x^2 / 2 + a * x
  let C₂ : ℝ → ℝ → ℝ := λ x m => 2 * a^2 * Real.log x + m
  let tangent_C₁ : ℝ → ℝ := λ x => x + a
  let tangent_C₂ : ℝ → ℝ := λ x => 2 * a^2 / x
  ∃ x₀ m, C₁ x₀ = C₂ x₀ m ∧ tangent_C₁ x₀ = tangent_C₂ x₀ ∧ 
    (∀ m', C₁ x₀ = C₂ x₀ m' ∧ tangent_C₁ x₀ = tangent_C₂ x₀ → m' ≤ m) ∧
    m = Real.exp (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_max_m_value_l408_40834


namespace NUMINAMATH_CALUDE_game_strategies_l408_40833

/-- The game state -/
structure GameState where
  board : ℝ
  turn : ℕ

/-- The game rules -/
def valid_move (x y : ℝ) : Prop :=
  0 < y - x ∧ y - x < 1

/-- The winning condition for the first variant -/
def winning_condition_1 (s : GameState) : Prop :=
  s.board ≥ 2010

/-- The winning condition for the second variant -/
def winning_condition_2 (s : GameState) : Prop :=
  s.board ≥ 2010 ∧ s.turn ≥ 2011

/-- The losing condition for the second variant -/
def losing_condition_2 (s : GameState) : Prop :=
  s.board ≥ 2010 ∧ s.turn ≤ 2010

/-- The theorem statement -/
theorem game_strategies :
  (∃ (strategy : ℕ → ℝ → ℝ),
    (∀ (n : ℕ) (x : ℝ), valid_move x (strategy n x)) ∧
    (∀ (play : ℕ → ℝ),
      (∀ (n : ℕ), valid_move (play n) (play (n+1))) →
      ∃ (k : ℕ), winning_condition_1 ⟨play k, k⟩ ∧
        k % 2 = 0)) ∧
  (∃ (strategy : ℕ → ℝ → ℝ),
    (∀ (n : ℕ) (x : ℝ), valid_move x (strategy n x)) ∧
    (∀ (play : ℕ → ℝ),
      (∀ (n : ℕ), valid_move (play n) (play (n+1))) →
      (∃ (k : ℕ), winning_condition_2 ⟨play k, k⟩ ∧
        k % 2 = 1) ∧
      (∀ (k : ℕ), k ≤ 2010 → ¬losing_condition_2 ⟨play k, k⟩))) :=
by sorry

end NUMINAMATH_CALUDE_game_strategies_l408_40833


namespace NUMINAMATH_CALUDE_abc_inequality_l408_40846

theorem abc_inequality (a b c : ℝ) (ha : |a| < 1) (hb : |b| < 1) (hc : |c| < 1) :
  a * b * c + 2 > a + b + c := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l408_40846


namespace NUMINAMATH_CALUDE_bakery_children_count_l408_40838

theorem bakery_children_count (initial_count : ℕ) : 
  initial_count + 24 - 31 = 78 → initial_count = 85 := by
  sorry

end NUMINAMATH_CALUDE_bakery_children_count_l408_40838


namespace NUMINAMATH_CALUDE_simplify_and_abs_l408_40856

theorem simplify_and_abs (a : ℝ) (h : a = -2) : 
  |12 * a^5 / (72 * a^3)| = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_abs_l408_40856


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l408_40861

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 2}
def B : Set ℝ := {x | x > Real.sqrt 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | Real.sqrt 3 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l408_40861


namespace NUMINAMATH_CALUDE_prime_factors_of_30_factorial_l408_40891

theorem prime_factors_of_30_factorial (n : ℕ) : n = 30 →
  (Finset.filter (Nat.Prime) (Finset.range (n + 1))).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_of_30_factorial_l408_40891


namespace NUMINAMATH_CALUDE_isosceles_triangle_coordinates_l408_40806

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (4, 2)

def is_right_angle (p q r : ℝ × ℝ) : Prop :=
  (p.1 - q.1) * (r.1 - q.1) + (p.2 - q.2) * (r.2 - q.2) = 0

def is_isosceles (p q r : ℝ × ℝ) : Prop :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = (p.1 - r.1)^2 + (p.2 - r.2)^2

theorem isosceles_triangle_coordinates :
  ∀ B : ℝ × ℝ,
    is_isosceles O A B →
    is_right_angle O B A →
    (B = (1, 3) ∨ B = (3, -1)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_coordinates_l408_40806


namespace NUMINAMATH_CALUDE_ben_bonus_allocation_l408_40802

theorem ben_bonus_allocation (bonus : ℚ) (holiday_fraction : ℚ) (gift_fraction : ℚ) (remaining : ℚ) 
  (h1 : bonus = 1496)
  (h2 : holiday_fraction = 1/4)
  (h3 : gift_fraction = 1/8)
  (h4 : remaining = 867) :
  let kitchen_fraction := (bonus - remaining - holiday_fraction * bonus - gift_fraction * bonus) / bonus
  kitchen_fraction = 221/748 := by sorry

end NUMINAMATH_CALUDE_ben_bonus_allocation_l408_40802


namespace NUMINAMATH_CALUDE_square_difference_divided_by_eleven_l408_40832

theorem square_difference_divided_by_eleven : (131^2 - 120^2) / 11 = 251 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_eleven_l408_40832


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l408_40894

theorem complex_modulus_problem (z : ℂ) : z = (-1 + 2*I) / I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l408_40894


namespace NUMINAMATH_CALUDE_storm_rain_difference_l408_40864

/-- Amount of rain in the first hour -/
def first_hour_rain : ℝ := 5

/-- Total amount of rain in the first two hours -/
def total_rain : ℝ := 22

/-- Amount of rain in the second hour -/
def second_hour_rain : ℝ := total_rain - first_hour_rain

/-- The difference between the amount of rain in the second hour and twice the amount of rain in the first hour -/
def rain_difference : ℝ := second_hour_rain - 2 * first_hour_rain

theorem storm_rain_difference : rain_difference = 7 := by
  sorry

end NUMINAMATH_CALUDE_storm_rain_difference_l408_40864


namespace NUMINAMATH_CALUDE_triangle_max_area_l408_40899

/-- The maximum area of a triangle with medians satisfying certain conditions -/
theorem triangle_max_area (m_a m_b m_c : ℝ) 
  (h_a : m_a ≤ 2) (h_b : m_b ≤ 3) (h_c : m_c ≤ 4) : 
  (∃ (E : ℝ), E = (1/3) * Real.sqrt (2*(m_a^2 * m_b^2 + m_b^2 * m_c^2 + m_c^2 * m_a^2) - (m_a^4 + m_b^4 + m_c^4)) ∧
  (∀ (E' : ℝ), E' = (1/3) * Real.sqrt (2*(m_a^2 * m_b^2 + m_b^2 * m_c^2 + m_c^2 * m_a^2) - (m_a^4 + m_b^4 + m_c^4)) → E' ≤ E)) →
  (∃ (E_max : ℝ), E_max = 4 ∧
  (∀ (E : ℝ), E = (1/3) * Real.sqrt (2*(m_a^2 * m_b^2 + m_b^2 * m_c^2 + m_c^2 * m_a^2) - (m_a^4 + m_b^4 + m_c^4)) → E ≤ E_max)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l408_40899


namespace NUMINAMATH_CALUDE_problem_triangle_integer_segments_l408_40884

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- Counts the number of distinct integer lengths of line segments
    from vertex E to points on the hypotenuse DF -/
def countIntegerSegments (t : RightTriangle) : ℕ :=
  sorry

/-- The specific right triangle in the problem -/
def problemTriangle : RightTriangle :=
  { de := 24, ef := 25 }

/-- The main theorem stating that the number of distinct integer lengths
    of line segments from E to DF in the problem triangle is 14 -/
theorem problem_triangle_integer_segments :
  countIntegerSegments problemTriangle = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_triangle_integer_segments_l408_40884


namespace NUMINAMATH_CALUDE_parameterized_line_problem_l408_40842

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  -- The vector on the line at parameter t
  vector : ℝ → (Fin 3 → ℝ)

/-- The problem statement as a theorem -/
theorem parameterized_line_problem :
  ∀ (line : ParameterizedLine),
    (line.vector 1 = ![2, 4, 9]) →
    (line.vector 3 = ![1, 1, 2]) →
    (line.vector 4 = ![0.5, -0.5, -1.5]) := by
  sorry

end NUMINAMATH_CALUDE_parameterized_line_problem_l408_40842


namespace NUMINAMATH_CALUDE_oliver_vowel_learning_days_l408_40880

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of days Oliver takes to learn one alphabet -/
def days_per_alphabet : ℕ := 5

/-- The number of days Oliver needs to finish learning all vowels -/
def days_to_learn_vowels : ℕ := num_vowels * days_per_alphabet

/-- Theorem: Oliver needs 25 days to finish learning all vowels -/
theorem oliver_vowel_learning_days : days_to_learn_vowels = 25 := by
  sorry

end NUMINAMATH_CALUDE_oliver_vowel_learning_days_l408_40880


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l408_40869

theorem arithmetic_geometric_mean_problem (p q r s : ℝ) : 
  (p + q) / 2 = 10 →
  (q + r) / 2 = 22 →
  (p * q * s)^(1/3) = 20 →
  r - p = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l408_40869


namespace NUMINAMATH_CALUDE_total_turnips_l408_40828

def keith_turnips : ℕ := 6
def alyssa_turnips : ℕ := 9

theorem total_turnips : keith_turnips + alyssa_turnips = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_l408_40828


namespace NUMINAMATH_CALUDE_sum_of_qp_values_l408_40822

def p (x : ℝ) : ℝ := |x| + 1

def q (x : ℝ) : ℝ := -|x - 1|

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3]

theorem sum_of_qp_values :
  (x_values.map (λ x => q (p x))).sum = -21 := by sorry

end NUMINAMATH_CALUDE_sum_of_qp_values_l408_40822


namespace NUMINAMATH_CALUDE_forty_sheep_eat_forty_bags_l408_40877

/-- The number of bags of grass eaten by a group of sheep -/
def bags_eaten (num_sheep : ℕ) (num_days : ℕ) : ℕ :=
  num_sheep * (num_days / 40)

/-- Theorem: 40 sheep eat 40 bags of grass in 40 days -/
theorem forty_sheep_eat_forty_bags :
  bags_eaten 40 40 = 40 := by
  sorry

end NUMINAMATH_CALUDE_forty_sheep_eat_forty_bags_l408_40877


namespace NUMINAMATH_CALUDE_white_triangle_coincidence_l408_40851

/-- Represents the number of triangles of each color in each half of the diagram -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of each type when the diagram is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_blue : ℕ

/-- Calculates the number of coinciding white triangle pairs given the initial counts and other coinciding pairs -/
def coinciding_white_pairs (counts : TriangleCounts) (pairs : CoincidingPairs) : ℕ :=
  counts.white - (counts.red - 2 * pairs.red_red - pairs.red_blue) - (counts.blue - 2 * pairs.blue_blue - pairs.red_blue)

/-- Theorem stating that under the given conditions, 6 pairs of white triangles exactly coincide -/
theorem white_triangle_coincidence (counts : TriangleCounts) (pairs : CoincidingPairs) : 
  counts.red = 5 ∧ counts.blue = 4 ∧ counts.white = 7 ∧ 
  pairs.red_red = 3 ∧ pairs.blue_blue = 2 ∧ pairs.red_blue = 1 →
  coinciding_white_pairs counts pairs = 6 := by
  sorry

end NUMINAMATH_CALUDE_white_triangle_coincidence_l408_40851


namespace NUMINAMATH_CALUDE_combination_equality_l408_40850

theorem combination_equality (a : ℕ) : 
  (Nat.choose 17 (2*a - 1) + Nat.choose 17 (2*a) = Nat.choose 18 12) → 
  (a = 3 ∨ a = 6) := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l408_40850


namespace NUMINAMATH_CALUDE_wall_width_proof_l408_40823

def wall_height : ℝ := 4
def wall_area : ℝ := 16

theorem wall_width_proof :
  ∃ (width : ℝ), width * wall_height = wall_area ∧ width = 4 := by
sorry

end NUMINAMATH_CALUDE_wall_width_proof_l408_40823


namespace NUMINAMATH_CALUDE_like_terms_exponent_l408_40892

theorem like_terms_exponent (x y : ℝ) (m n : ℕ) :
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * x * y^(n + 1) = b * x^m * y^4) →
  m^n = 1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_l408_40892


namespace NUMINAMATH_CALUDE_constrained_line_generates_surface_l408_40885

/-- A line parallel to the plane y=z, intersecting two parabolas -/
structure ConstrainedLine where
  /-- The line is parallel to the plane y=z -/
  parallel_to_yz : ℝ → ℝ → ℝ → Prop
  /-- The line intersects the parabola 2x=y², z=0 -/
  meets_parabola1 : ℝ → ℝ → ℝ → Prop
  /-- The line intersects the parabola 3x=z², y=0 -/
  meets_parabola2 : ℝ → ℝ → ℝ → Prop

/-- The surface generated by the constrained line -/
def generated_surface (x y z : ℝ) : Prop :=
  x = (y - z) * (y / 2 - z / 3)

/-- Theorem stating that the constrained line generates the specified surface -/
theorem constrained_line_generates_surface (L : ConstrainedLine) :
  ∀ x y z, L.parallel_to_yz x y z → L.meets_parabola1 x y z → L.meets_parabola2 x y z →
  generated_surface x y z :=
sorry

end NUMINAMATH_CALUDE_constrained_line_generates_surface_l408_40885


namespace NUMINAMATH_CALUDE_value_of_2a_plus_b_l408_40855

theorem value_of_2a_plus_b (a b : ℝ) (h : |a + 2| + (b - 5)^2 = 0) : 2*a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_2a_plus_b_l408_40855


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l408_40843

theorem fraction_sum_theorem (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (75 - c) = 8) :
  6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l408_40843


namespace NUMINAMATH_CALUDE_unfair_die_expected_value_l408_40887

/-- An unfair eight-sided die with specific probabilities -/
structure UnfairDie where
  /-- The probability of rolling an 8 -/
  prob_eight : ℚ
  /-- The probability of rolling any number from 1 to 7 -/
  prob_others : ℚ
  /-- The probability of rolling an 8 is 3/7 -/
  h_prob_eight : prob_eight = 3/7
  /-- The probabilities sum to 1 -/
  h_sum_to_one : prob_eight + 7 * prob_others = 1

/-- The expected value of rolling the unfair die -/
def expected_value (d : UnfairDie) : ℚ :=
  d.prob_others * (1 + 2 + 3 + 4 + 5 + 6 + 7) + 8 * d.prob_eight

/-- Theorem stating the expected value of the unfair die -/
theorem unfair_die_expected_value (d : UnfairDie) :
  expected_value d = 40/7 := by
  sorry

#eval (40 : ℚ) / 7

end NUMINAMATH_CALUDE_unfair_die_expected_value_l408_40887


namespace NUMINAMATH_CALUDE_fruit_juice_needed_correct_problem_solution_l408_40847

/-- Represents the ratio of ingredients in a drink -/
structure DrinkRatio where
  milk : ℚ
  fruit_juice : ℚ

/-- Represents the amount of ingredients in a drink -/
structure DrinkAmount where
  milk : ℚ
  fruit_juice : ℚ

/-- Converts a ratio to normalized form where total parts sum to 1 -/
def normalize_ratio (r : DrinkRatio) : DrinkRatio :=
  let total := r.milk + r.fruit_juice
  { milk := r.milk / total, fruit_juice := r.fruit_juice / total }

/-- Calculates the amount of fruit juice needed to convert drink A to drink B -/
def fruit_juice_needed (amount_A : ℚ) (ratio_A ratio_B : DrinkRatio) : ℚ :=
  let norm_A := normalize_ratio ratio_A
  let norm_B := normalize_ratio ratio_B
  let milk_A := amount_A * norm_A.milk
  let fruit_juice_A := amount_A * norm_A.fruit_juice
  (milk_A - fruit_juice_A) / (norm_B.fruit_juice - norm_B.milk)

/-- Theorem: The amount of fruit juice needed is correct -/
theorem fruit_juice_needed_correct (amount_A : ℚ) (ratio_A ratio_B : DrinkRatio) :
  let juice_needed := fruit_juice_needed amount_A ratio_A ratio_B
  let total_amount := amount_A + juice_needed
  let final_amount := DrinkAmount.mk (amount_A * (normalize_ratio ratio_A).milk) (fruit_juice_needed amount_A ratio_A ratio_B + amount_A * (normalize_ratio ratio_A).fruit_juice)
  final_amount.milk / total_amount = (normalize_ratio ratio_B).milk ∧
  final_amount.fruit_juice / total_amount = (normalize_ratio ratio_B).fruit_juice :=
by sorry

/-- Specific problem instance -/
def drink_A : DrinkRatio := { milk := 4, fruit_juice := 3 }
def drink_B : DrinkRatio := { milk := 3, fruit_juice := 4 }

/-- Theorem: For the given problem, 14 liters of fruit juice are needed -/
theorem problem_solution : 
  fruit_juice_needed 98 drink_A drink_B = 14 :=
by sorry

end NUMINAMATH_CALUDE_fruit_juice_needed_correct_problem_solution_l408_40847


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l408_40816

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) : A = 4 * Real.pi → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l408_40816


namespace NUMINAMATH_CALUDE_fraction_meaningful_iff_not_negative_one_l408_40825

theorem fraction_meaningful_iff_not_negative_one (x : ℝ) : 
  (∃ y : ℝ, y = (x - 1) / (x + 1)) ↔ x ≠ -1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_iff_not_negative_one_l408_40825


namespace NUMINAMATH_CALUDE_sector_to_cone_l408_40871

/-- Represents a cone formed from a circular sector -/
structure SectorCone where
  sector_radius : ℝ
  sector_angle : ℝ
  base_radius : ℝ
  slant_height : ℝ

/-- Theorem: A 270° sector of a circle with radius 12 forms a cone with base radius 9 and slant height 12 -/
theorem sector_to_cone :
  ∀ (cone : SectorCone),
    cone.sector_radius = 12 ∧
    cone.sector_angle = 270 ∧
    cone.slant_height = cone.sector_radius →
    cone.base_radius = 9 ∧
    cone.slant_height = 12 := by
  sorry


end NUMINAMATH_CALUDE_sector_to_cone_l408_40871


namespace NUMINAMATH_CALUDE_equation_rewrite_l408_40875

theorem equation_rewrite (x y : ℝ) : (2 * x + y = 5) ↔ (y = 5 - 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_equation_rewrite_l408_40875


namespace NUMINAMATH_CALUDE_sequence_property_l408_40809

def sequence_a (n : ℕ+) : ℚ :=
  1 / (2 * n - 1)

theorem sequence_property (n : ℕ+) :
  let a : ℕ+ → ℚ := sequence_a
  (n = 1 → a n = 1) ∧
  (∀ k : ℕ+, a k ≠ 0) ∧
  (∀ k : ℕ+, k ≥ 2 → a k + 2 * a k * a (k - 1) - a (k - 1) = 0) →
  a n = 1 / (2 * n - 1) :=
by
  sorry

#check sequence_property

end NUMINAMATH_CALUDE_sequence_property_l408_40809


namespace NUMINAMATH_CALUDE_max_value_theorem_l408_40896

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^2 + b^2 - Real.sqrt 3 * a * b = 1) : 
  Real.sqrt 3 * a^2 - a * b ≤ 2 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l408_40896


namespace NUMINAMATH_CALUDE_power_of_product_l408_40810

theorem power_of_product (x y : ℝ) : (-2 * x^2 * y)^3 = -8 * x^6 * y^3 := by sorry

end NUMINAMATH_CALUDE_power_of_product_l408_40810


namespace NUMINAMATH_CALUDE_minimize_S_l408_40886

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℝ := 2 * n^2 - 30 * n

/-- n minimizes S if S(n) is less than or equal to S(k) for all natural numbers k -/
def Minimizes (n : ℕ) : Prop :=
  ∀ k : ℕ, S n ≤ S k

theorem minimize_S :
  ∃ n : ℕ, (n = 7 ∨ n = 8) ∧ Minimizes n :=
sorry

end NUMINAMATH_CALUDE_minimize_S_l408_40886


namespace NUMINAMATH_CALUDE_field_trip_buses_l408_40839

/-- Given a field trip scenario with vans and buses, calculate the number of buses required. -/
theorem field_trip_buses (total_people : ℕ) (num_vans : ℕ) (people_per_van : ℕ) (people_per_bus : ℕ)
  (h1 : total_people = 180)
  (h2 : num_vans = 6)
  (h3 : people_per_van = 6)
  (h4 : people_per_bus = 18) :
  (total_people - num_vans * people_per_van) / people_per_bus = 8 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_buses_l408_40839


namespace NUMINAMATH_CALUDE_optimal_price_for_target_profit_l408_40878

-- Define the cost to produce the souvenir
def production_cost : ℝ := 30

-- Define the lower and upper bounds of the selling price
def min_price : ℝ := production_cost
def max_price : ℝ := 54

-- Define the base price and corresponding daily sales
def base_price : ℝ := 40
def base_sales : ℝ := 80

-- Define the rate of change in sales per yuan increase in price
def sales_change_rate : ℝ := -2

-- Define the target daily profit
def target_profit : ℝ := 1200

-- Define the function for daily sales based on price
def daily_sales (price : ℝ) : ℝ :=
  base_sales + sales_change_rate * (price - base_price)

-- Define the function for daily profit based on price
def daily_profit (price : ℝ) : ℝ :=
  (price - production_cost) * daily_sales price

-- Theorem statement
theorem optimal_price_for_target_profit :
  ∃ (price : ℝ), min_price ≤ price ∧ price ≤ max_price ∧ daily_profit price = target_profit ∧ price = 50 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_for_target_profit_l408_40878


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_when_area_twice_perimeter_l408_40835

/-- Given a triangle where the area is twice the perimeter, 
    the radius of the inscribed circle is 4. -/
theorem inscribed_circle_radius_when_area_twice_perimeter 
  (T : Set ℝ × Set ℝ) -- T represents a triangle in 2D space
  (A : ℝ) -- A represents the area of the triangle
  (p : ℝ) -- p represents the perimeter of the triangle
  (r : ℝ) -- r represents the radius of the inscribed circle
  (h1 : A = 2 * p) -- condition that area is twice the perimeter
  : r = 4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_when_area_twice_perimeter_l408_40835


namespace NUMINAMATH_CALUDE_range_g_eq_range_f_l408_40805

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }

-- Define the range of f
def range_f : Set ℝ := { y | ∃ x ∈ domain_f, f x = y }

-- State that the range of f is [a, b]
axiom range_f_is_ab : ∃ a b : ℝ, range_f = { y | a ≤ y ∧ y ≤ b }

-- Define the function g(x) = f(x + 4)
def g (x : ℝ) : ℝ := f (x + 4)

-- Define the range of g
def range_g : Set ℝ := { y | ∃ x : ℝ, g x = y }

-- Theorem: The range of g is equal to the range of f
theorem range_g_eq_range_f : range_g = range_f := by sorry

end NUMINAMATH_CALUDE_range_g_eq_range_f_l408_40805


namespace NUMINAMATH_CALUDE_simplify_and_sum_exponents_l408_40853

def simplify_cube_root (x y z : ℝ) : ℝ := (40 * x^5 * y^9 * z^14) ^ (1/3)

theorem simplify_and_sum_exponents (x y z : ℝ) :
  ∃ (a e : ℝ) (b c d f g h : ℕ),
    simplify_cube_root x y z = a * x^b * y^c * z^d * (e * x^f * y^g * z^h)^(1/3) ∧
    b + c + d = 5 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_sum_exponents_l408_40853


namespace NUMINAMATH_CALUDE_distance_to_reflection_over_x_axis_distance_A_to_A_l408_40824

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection_over_x_axis (x y : ℝ) :
  let A : ℝ × ℝ := (x, y)
  let A' : ℝ × ℝ := (x, -y)
  Real.sqrt ((A'.1 - A.1)^2 + (A'.2 - A.2)^2) = 2 * abs y := by
  sorry

/-- The specific case for point A(2, -4) --/
theorem distance_A_to_A'_reflection :
  let A : ℝ × ℝ := (2, -4)
  let A' : ℝ × ℝ := (2, 4)
  Real.sqrt ((A'.1 - A.1)^2 + (A'.2 - A.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_reflection_over_x_axis_distance_A_to_A_l408_40824


namespace NUMINAMATH_CALUDE_max_teams_in_tournament_l408_40849

/-- Represents a chess tournament with teams of 3 players each --/
structure ChessTournament where
  numTeams : ℕ
  maxGames : ℕ := 250

/-- Calculate the total number of games in the tournament --/
def totalGames (t : ChessTournament) : ℕ :=
  (9 * t.numTeams * (t.numTeams - 1)) / 2

/-- Theorem stating the maximum number of teams in the tournament --/
theorem max_teams_in_tournament (t : ChessTournament) :
  (∀ n : ℕ, n ≤ t.numTeams → totalGames { numTeams := n, maxGames := t.maxGames } ≤ t.maxGames) →
  t.numTeams ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_teams_in_tournament_l408_40849


namespace NUMINAMATH_CALUDE_cube_volume_l408_40829

theorem cube_volume (cube_side : ℝ) (h1 : cube_side > 0) (h2 : cube_side ^ 2 = 36) :
  cube_side ^ 3 = 216 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_l408_40829


namespace NUMINAMATH_CALUDE_rooster_on_roof_no_egg_falls_l408_40821

/-- Represents a bird species -/
inductive BirdSpecies
  | Rooster
  | Hen

/-- Represents the ability to lay eggs -/
def canLayEggs (species : BirdSpecies) : Prop :=
  match species with
  | BirdSpecies.Rooster => False
  | BirdSpecies.Hen => True

/-- Represents a roof with two slopes -/
structure Roof :=
  (slope1 : ℝ)
  (slope2 : ℝ)

/-- Theorem: Given a roof with two slopes and a rooster on the ridge, no egg will fall -/
theorem rooster_on_roof_no_egg_falls (roof : Roof) (bird : BirdSpecies) :
  roof.slope1 = 60 → roof.slope2 = 70 → bird = BirdSpecies.Rooster → ¬(canLayEggs bird) :=
by sorry

end NUMINAMATH_CALUDE_rooster_on_roof_no_egg_falls_l408_40821


namespace NUMINAMATH_CALUDE_stop_after_seventh_shot_probability_value_l408_40841

/-- The maximum number of shots allowed -/
def max_shots : ℕ := 10

/-- The probability of making a shot for student A -/
def shot_probability : ℚ := 2/3

/-- Calculate the score based on the shot number when the student stops -/
def score (n : ℕ) : ℕ := 12 - n

/-- The probability of the specific sequence of shots leading to stopping after the 7th shot -/
def stop_after_seventh_shot_probability : ℚ :=
  (1 - shot_probability) * shot_probability * (1 - shot_probability) *
  1 * (1 - shot_probability) * shot_probability * shot_probability

theorem stop_after_seventh_shot_probability_value :
  stop_after_seventh_shot_probability = 8/729 :=
sorry

end NUMINAMATH_CALUDE_stop_after_seventh_shot_probability_value_l408_40841


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l408_40859

/-- Simplification of a trigonometric expression -/
theorem trig_expression_simplification :
  let expr := (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + 
               Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / 
              Real.cos (10 * π / 180)
  ∃ (k : ℝ), expr = (2 * Real.cos (40 * π / 180)) / Real.cos (10 * π / 180) * k :=
by
  sorry


end NUMINAMATH_CALUDE_trig_expression_simplification_l408_40859


namespace NUMINAMATH_CALUDE_gcd_problems_l408_40836

theorem gcd_problems :
  (Nat.gcd 840 1764 = 84) ∧ (Nat.gcd 440 556 = 4) := by sorry

end NUMINAMATH_CALUDE_gcd_problems_l408_40836


namespace NUMINAMATH_CALUDE_quadratic_sum_theorem_l408_40865

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The x-coordinate of the vertex of a quadratic function -/
def vertex_x (f : QuadraticFunction) : ℚ := -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
def vertex_y (f : QuadraticFunction) : ℚ := f.c - f.b^2 / (4 * f.a)

/-- Theorem: For a quadratic function with integer coefficients and vertex at (2, -3),
    the sum a + b - c equals -4 -/
theorem quadratic_sum_theorem (f : QuadraticFunction) 
  (h1 : vertex_x f = 2)
  (h2 : vertex_y f = -3) :
  f.a + f.b - f.c = -4 := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_theorem_l408_40865


namespace NUMINAMATH_CALUDE_oregon_migration_l408_40879

/-- The number of people moving to Oregon -/
def people_moving : ℕ := 3500

/-- The number of days over which people are moving -/
def days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the average number of people moving per hour -/
def average_per_hour : ℚ := people_moving / (days * hours_per_day)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem oregon_migration :
  round_to_nearest average_per_hour = 29 := by
  sorry

end NUMINAMATH_CALUDE_oregon_migration_l408_40879


namespace NUMINAMATH_CALUDE_project_completion_time_l408_40863

-- Define the individual work rates
def renu_rate : ℚ := 1 / 5
def suma_rate : ℚ := 1 / 8
def arun_rate : ℚ := 1 / 10

-- Define the combined work rate
def combined_rate : ℚ := renu_rate + suma_rate + arun_rate

-- Theorem statement
theorem project_completion_time :
  (1 : ℚ) / combined_rate = 40 / 17 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_l408_40863


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l408_40813

theorem fixed_point_parabola (k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 9 * x^2 + 3 * k * x - 6 * k
  f 2 = 36 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l408_40813


namespace NUMINAMATH_CALUDE_logarithm_sum_property_l408_40830

theorem logarithm_sum_property (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_property_l408_40830


namespace NUMINAMATH_CALUDE_triangle_inequality_with_semiperimeter_l408_40873

theorem triangle_inequality_with_semiperimeter (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  Real.sqrt (b + c - a) + Real.sqrt (c + a - b) + Real.sqrt (a + b - c) ≤ 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ∧ 
  (Real.sqrt (b + c - a) + Real.sqrt (c + a - b) + Real.sqrt (a + b - c) = 
   Real.sqrt a + Real.sqrt b + Real.sqrt c ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_semiperimeter_l408_40873


namespace NUMINAMATH_CALUDE_set_range_with_given_mean_median_l408_40874

/-- Given a set of three real numbers with mean and median both equal to 5,
    and the smallest number being 2, the range of the set is 6. -/
theorem set_range_with_given_mean_median (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →  -- Ordered set of three numbers
  a = 2 →  -- Smallest number is 2
  (a + b + c) / 3 = 5 →  -- Mean is 5
  b = 5 →  -- Median is 5 (for three numbers, the median is the middle number)
  c - a = 6 :=  -- Range is 6
by sorry

end NUMINAMATH_CALUDE_set_range_with_given_mean_median_l408_40874


namespace NUMINAMATH_CALUDE_distribution_scheme_count_l408_40808

/-- The number of ways to distribute spots among schools -/
def distribute_spots (total_spots : ℕ) (num_schools : ℕ) (distribution : List ℕ) : ℕ :=
  if total_spots = distribution.sum ∧ num_schools = distribution.length
  then Nat.factorial num_schools
  else 0

theorem distribution_scheme_count :
  distribute_spots 10 4 [1, 2, 3, 4] = 24 := by
  sorry

end NUMINAMATH_CALUDE_distribution_scheme_count_l408_40808


namespace NUMINAMATH_CALUDE_school_boys_count_l408_40883

theorem school_boys_count (girls : ℕ) (difference : ℕ) (boys : ℕ) : 
  girls = 739 → difference = 402 → girls = boys + difference → boys = 337 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l408_40883


namespace NUMINAMATH_CALUDE_number_of_balls_l408_40807

theorem number_of_balls (x : ℕ) : x - 92 = 156 - x → x = 124 := by
  sorry

end NUMINAMATH_CALUDE_number_of_balls_l408_40807


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_third_l408_40820

theorem opposite_of_negative_one_third :
  -(-(1/3 : ℚ)) = 1/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_third_l408_40820


namespace NUMINAMATH_CALUDE_expression_factorization_l408_40817

theorem expression_factorization (x : ℝ) :
  (8 * x^6 + 36 * x^4 - 5) - (2 * x^6 - 6 * x^4 + 5) = 2 * (3 * x^6 + 21 * x^4 - 5) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l408_40817


namespace NUMINAMATH_CALUDE_shirt_ratio_l408_40881

theorem shirt_ratio (brian_shirts andrew_shirts steven_shirts : ℕ) :
  brian_shirts = 3 →
  andrew_shirts = 6 * brian_shirts →
  steven_shirts = 72 →
  steven_shirts / andrew_shirts = 4 :=
by sorry

end NUMINAMATH_CALUDE_shirt_ratio_l408_40881


namespace NUMINAMATH_CALUDE_earnings_increase_l408_40893

theorem earnings_increase (last_year_earnings last_year_rent_percentage this_year_rent_percentage rent_increase_percentage : ℝ)
  (h1 : last_year_rent_percentage = 20)
  (h2 : this_year_rent_percentage = 30)
  (h3 : rent_increase_percentage = 187.5)
  (h4 : this_year_rent_percentage / 100 * (last_year_earnings * (1 + x / 100)) = 
        rent_increase_percentage / 100 * (last_year_rent_percentage / 100 * last_year_earnings)) :
  x = 25 := by sorry


end NUMINAMATH_CALUDE_earnings_increase_l408_40893


namespace NUMINAMATH_CALUDE_expansion_coefficient_l408_40811

theorem expansion_coefficient (m : ℤ) : 
  (Nat.choose 6 3 : ℤ) * m^3 = -160 → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l408_40811


namespace NUMINAMATH_CALUDE_consecutive_four_plus_one_is_square_l408_40804

theorem consecutive_four_plus_one_is_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_four_plus_one_is_square_l408_40804


namespace NUMINAMATH_CALUDE_sum_lower_bound_l408_40867

theorem sum_lower_bound (x : ℕ → ℝ) (h_incr : ∀ n, x n ≤ x (n + 1)) (h_x0 : x 0 = 1) :
  (∑' n, x (n + 1) / (x n)^3) ≥ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l408_40867


namespace NUMINAMATH_CALUDE_cube_coloring_count_dodecahedron_coloring_count_l408_40889

/-- The number of rotational symmetries of a cube -/
def cube_rotations : ℕ := 24

/-- The number of rotational symmetries of a dodecahedron -/
def dodecahedron_rotations : ℕ := 60

/-- The number of faces of a cube -/
def cube_faces : ℕ := 6

/-- The number of faces of a dodecahedron -/
def dodecahedron_faces : ℕ := 12

/-- Calculates the number of geometrically distinct colorings for a polyhedron -/
def distinct_colorings (faces : ℕ) (rotations : ℕ) : ℕ :=
  (Nat.factorial faces) / rotations

theorem cube_coloring_count :
  distinct_colorings cube_faces cube_rotations = 30 := by sorry

theorem dodecahedron_coloring_count :
  distinct_colorings dodecahedron_faces dodecahedron_rotations = 7983360 := by sorry

end NUMINAMATH_CALUDE_cube_coloring_count_dodecahedron_coloring_count_l408_40889


namespace NUMINAMATH_CALUDE_segment_length_l408_40876

/-- The length of a segment with endpoints (1,2) and (9,16) is 2√65 -/
theorem segment_length : Real.sqrt ((9 - 1)^2 + (16 - 2)^2) = 2 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_segment_length_l408_40876


namespace NUMINAMATH_CALUDE_father_age_is_27_l408_40826

/-- The present age of the son -/
def son_age : ℕ := sorry

/-- The present age of the father -/
def father_age : ℕ := sorry

/-- The father's age is 3 years more than 3 times the son's age -/
axiom condition1 : father_age = 3 * son_age + 3

/-- In 3 years, the father's age will be 8 years more than twice the son's age -/
axiom condition2 : father_age + 3 = 2 * (son_age + 3) + 8

theorem father_age_is_27 : father_age = 27 := by sorry

end NUMINAMATH_CALUDE_father_age_is_27_l408_40826


namespace NUMINAMATH_CALUDE_exists_hyperbola_segment_with_midpoint_l408_40872

/-- The hyperbola equation -/
def on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

/-- The midpoint of two points -/
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2

theorem exists_hyperbola_segment_with_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    on_hyperbola x₁ y₁ ∧
    on_hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ :=
  sorry

end NUMINAMATH_CALUDE_exists_hyperbola_segment_with_midpoint_l408_40872


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l408_40827

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l408_40827


namespace NUMINAMATH_CALUDE_magic_trick_possible_l408_40866

-- Define a coin as either Heads or Tails
inductive Coin : Type
| Heads : Coin
| Tails : Coin

-- Define a row of 27 coins
def CoinRow : Type := Fin 27 → Coin

-- Define a function to group coins into triplets
def groupIntoTriplets (row : CoinRow) : Fin 9 → Fin 3 → Coin :=
  fun i j => row (3 * i + j)

-- Define a strategy for the assistant to uncover 5 coins
def assistantStrategy (row : CoinRow) : Fin 5 → Fin 27 :=
  sorry

-- Define a strategy for the magician to identify 5 more coins
def magicianStrategy (row : CoinRow) (uncovered : Fin 5 → Fin 27) : Fin 5 → Fin 27 :=
  sorry

-- The main theorem
theorem magic_trick_possible (row : CoinRow) :
  ∃ (uncovered : Fin 5 → Fin 27) (identified : Fin 5 → Fin 27),
    (∀ i : Fin 5, row (uncovered i) = row (uncovered 0)) ∧
    (∀ i : Fin 5, row (identified i) = row (uncovered 0)) ∧
    (∀ i j : Fin 5, uncovered i ≠ identified j) :=
  sorry

end NUMINAMATH_CALUDE_magic_trick_possible_l408_40866


namespace NUMINAMATH_CALUDE_circle_equation_implies_sum_l408_40862

theorem circle_equation_implies_sum (x y : ℝ) :
  x^2 + y^2 - 2*x + 4*y + 5 = 0 → 2*x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_implies_sum_l408_40862


namespace NUMINAMATH_CALUDE_power_of_power_l408_40868

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l408_40868
