import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_of_f_l16_1619

/-- The function f(x) defined as sin(x) sin(2x) sin(3x) sin(4x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.sin (2*x) * Real.sin (3*x) * Real.sin (4*x)

/-- The statement that 2π is the smallest positive period of f(x) -/
theorem smallest_period_of_f : 
  ∀ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) → p ≥ 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_of_f_l16_1619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_problem_l16_1624

theorem division_remainder_problem (m n : ℕ) 
  (h1 : (m : ℝ) / (n : ℝ) = 24.2)
  (h2 : (n : ℝ) = 60.00000000000021)
  (hm : m > 0)
  (hn : n > 0) : 
  m % n = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_problem_l16_1624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_x_measure_l16_1687

-- Define the lines and angles
structure Geometry where
  m : ℝ → ℝ → Prop
  n : ℝ → ℝ → Prop
  x : ℝ
  transversal1 : ℝ → ℝ → Prop
  transversal2 : ℝ → ℝ → Prop
  adjacent_angle : ℝ

-- Define the properties
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry
def intersects (l : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop := sorry
def angle_measure (a : ℝ) : ℝ := sorry
def forms_right_angle (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry
def adjacent_angles (a1 a2 : ℝ) : Prop := sorry

-- State the theorem
theorem angle_x_measure (g : Geometry) :
  parallel g.m g.n →
  intersects g.transversal1 (0, 0) →
  intersects g.transversal1 (1, 0) →
  adjacent_angles g.x g.adjacent_angle →
  angle_measure g.adjacent_angle = 45 →
  intersects g.transversal2 (2, 0) →
  forms_right_angle g.transversal2 g.n →
  angle_measure g.x = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_x_measure_l16_1687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_printer_time_l16_1612

/-- Represents a printer with a given rate of labels per minute -/
structure Printer where
  rate : ℝ

/-- The time taken to print a given number of labels -/
noncomputable def print_time (p : Printer) (labels : ℝ) : ℝ := labels / p.rate

theorem second_printer_time (first_printer second_printer : Printer) 
  (h1 : print_time first_printer 1000 = 12)
  (h2 : 1000 / (first_printer.rate + second_printer.rate) = 3)
  (h3 : second_printer.rate = 1.2 * first_printer.rate) :
  print_time second_printer 1000 = 4 := by
  sorry

#check second_printer_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_printer_time_l16_1612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_theorem_l16_1614

def marble_problem (total : ℕ) (violet orange red yellow black white green : ℕ) : Prop :=
  -- Conditions
  violet + orange + red + yellow + black + white + green = total ∧
  violet = (25 * total) / 100 ∧
  orange = (15 * total) / 100 ∧
  red = (20 * total) / 100 ∧
  yellow = (10 * total) / 100 ∧
  black = (5 * total) / 100 ∧
  white = (10 * total) / 100 ∧
  green = 40 ∧
  -- Question
  let new_white := white + red / 3
  new_white = 45

theorem marble_theorem :
  ∃ total : ℕ, ∃ violet orange red yellow black white green : ℕ,
    marble_problem total violet orange red yellow black white green :=
by
  sorry

#check marble_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_theorem_l16_1614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_second_quadrant_l16_1686

/-- If a point P(tan α, cos α) is in the third quadrant, then the terminal side of angle α is in the second quadrant. -/
theorem terminal_side_in_second_quadrant (α : Real) :
  (Real.tan α < 0 ∧ Real.cos α < 0) → (Real.sin α > 0 ∧ Real.cos α < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_second_quadrant_l16_1686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ping_pong_probability_l16_1613

def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

theorem ping_pong_probability : 
  let total_balls : ℕ := 80
  let balls : Finset ℕ := Finset.range total_balls
  let multiples_of_4_or_6 : Finset ℕ := balls.filter (λ n => n > 0 ∧ (n % 4 = 0 ∨ n % 6 = 0))
  (multiples_of_4_or_6.card : ℚ) / total_balls = 27 / 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ping_pong_probability_l16_1613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_problem_solution_l16_1672

/-- A line in a rectangular coordinate system passing through two points. -/
structure Line where
  slope : ℝ
  intercept : ℝ
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The specific line described in the problem. -/
noncomputable def problem_line (m n a : ℝ) : Line where
  slope := 1/5
  intercept := 1
  point1 := (m, n)
  point2 := (m + 2, n + a)

/-- Theorem stating that the line passes through both points. -/
theorem line_through_points (l : Line) : 
  l.point1.1 = l.slope * l.point1.2 + l.intercept ∧ 
  l.point2.1 = l.slope * l.point2.2 + l.intercept :=
sorry

/-- Theorem representing the problem solution. -/
theorem problem_solution : 
  ∀ (m n : ℝ), ∃ (a : ℝ), 
    (problem_line m n a).point1 = (m, n) ∧ 
    (problem_line m n a).point2 = (m + 2, n + a) ∧ 
    a = 2/5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_problem_solution_l16_1672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_third_term_positive_implies_sum_positive_l16_1680

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_third_term_positive_implies_sum_positive
  (a₁ q : ℝ) :
  geometric_sequence a₁ q 3 > 0 →
  geometric_sum a₁ q 2017 > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_third_term_positive_implies_sum_positive_l16_1680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_l16_1663

/-- The time (in seconds) it takes for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: A train 140 meters long traveling at 45 km/hr takes 30 seconds to cross a 235-meter bridge -/
theorem train_bridge_crossing :
  train_crossing_time 140 45 235 = 30 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_l16_1663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_our_sequence_100_l16_1649

def our_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => our_sequence n + n

theorem our_sequence_100 : our_sequence 99 = 4951 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_our_sequence_100_l16_1649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_many_people_sharing_car_l16_1625

/-- The number of people in the "Many People Sharing Car" problem -/
def x : ℕ := sorry

/-- The number of cars in the "Many People Sharing Car" problem -/
def y : ℕ := sorry

/-- The first condition: If three people ride in one car, there will be two empty cars -/
axiom condition1 : 3 * (y - 2) = x

/-- The second condition: If two people ride in one car, nine people will have to walk -/
axiom condition2 : 2 * y + 9 = x

/-- The "Many People Sharing Car" problem has a unique solution satisfying both conditions -/
theorem many_people_sharing_car : ∃! (x y : ℕ), 3 * (y - 2) = x ∧ 2 * y + 9 = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_many_people_sharing_car_l16_1625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l16_1635

/-- Given a circle with radius 2√3, whose center lies on the line y = 2x, 
    and which has a chord of length 4 intercepted by the line x - y = 0, 
    prove that the equation of the circle is either 
    (x - 4)² + (y - 8)² = 12 or (x + 4)² + (y + 8)² = 12 -/
theorem circle_equation (x y : ℝ) :
  let r : ℝ := 2 * Real.sqrt 3
  let center_on_line : ℝ → ℝ → Prop := λ a b => b = 2 * a
  let chord_length : ℝ → ℝ → Prop := λ a b => 
    4 + (|a - b| / Real.sqrt 2)^2 = 12
  ∃ a b : ℝ, center_on_line a b ∧ chord_length a b →
    ((x - 4)^2 + (y - 8)^2 = 12) ∨ ((x + 4)^2 + (y + 8)^2 = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l16_1635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l16_1642

/-- Calculates the combined length of platforms crossed by two trains -/
theorem platform_length_calculation (length_A length_B : ℝ) (speed_A speed_B : ℝ) (time_A time_B : ℝ) :
  length_A = 650 →
  length_B = 450 →
  speed_A = 115 * 1000 / 3600 →
  speed_B = 108 * 1000 / 3600 →
  time_A = 30 →
  time_B = 25 →
  (speed_A * time_A - length_A) + (speed_B * time_B - length_B) = 608.32 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l16_1642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_departure_difference_l16_1678

/-- Represents the scenario of two trains traveling from the same station. -/
structure TrainScenario where
  speed_a : ℝ  -- Speed of Train A in mph
  speed_b : ℝ  -- Speed of Train B in mph
  overtake_distance : ℝ  -- Distance at which Train B overtakes Train A in miles

/-- The time difference between the departures of Train A and Train B. -/
noncomputable def time_difference (scenario : TrainScenario) : ℝ :=
  scenario.overtake_distance / scenario.speed_a - scenario.overtake_distance / scenario.speed_b

/-- Theorem stating that under the given conditions, Train B leaves 2 hours after Train A. -/
theorem train_departure_difference (scenario : TrainScenario) 
  (h1 : scenario.speed_a = 30)
  (h2 : scenario.speed_b = 36)
  (h3 : scenario.overtake_distance = 360) :
  time_difference scenario = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_departure_difference_l16_1678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l16_1601

theorem polynomial_division_remainder (x : ℝ) : 
  ∃ q : Polynomial ℝ, X^4 + 3*X + 2 = (X - 2)^2 * q + (32*X - 30) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l16_1601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_x_minus_y_l16_1608

theorem max_value_x_minus_y (x y : ℝ) (h : x^2 + 2*x*y + y^2 + 4*x^2*y^2 = 4) :
  ∃ (z : ℝ), z = Real.sqrt 17 / 2 ∧ ∀ (w : ℝ), x - y ≤ w → w ≤ z :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_x_minus_y_l16_1608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calories_in_200g_lemonade_l16_1651

noncomputable section

-- Define the ingredients and their properties
def lemon_juice_weight : ℚ := 100
def sugar_weight : ℚ := 100
def water_weight : ℚ := 400
def lemon_juice_calories_per_100g : ℚ := 25
def sugar_calories_per_100g : ℚ := 386
def water_calories_per_100g : ℚ := 0

-- Define the total weight of the lemonade
def total_weight : ℚ := lemon_juice_weight + sugar_weight + water_weight

-- Define the total calories in the lemonade
def total_calories : ℚ := 
  (lemon_juice_weight / 100) * lemon_juice_calories_per_100g +
  (sugar_weight / 100) * sugar_calories_per_100g +
  (water_weight / 100) * water_calories_per_100g

-- Theorem to prove
theorem calories_in_200g_lemonade : 
  (200 / total_weight) * total_calories = 137 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calories_in_200g_lemonade_l16_1651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_minimum_l16_1666

theorem arithmetic_sequence_minimum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (a 2015 + a 2017 = ∫ x in (0 : ℝ)..(2 : ℝ), Real.sqrt (4 - x^2)) →  -- given condition
  (∃ m, ∀ d, a 2016 * (a 2014 + a 2018) ≥ m ∧ 
             a 2016 * (a 2014 + a 2018) = m ↔ d = 0) ∧
  (a 2016 * (a 2014 + a 2018) = Real.pi^2 / 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_minimum_l16_1666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1996_not_divisible_by_3_l16_1622

/-- 
f(n) is the number of permutations of integers 1 to n 
satisfying a₁ = 1 and |aᵢ - aᵢ₊₁| ≤ 2 for i = 1, 2, ..., n-1
-/
def f : ℕ → ℕ := sorry

/-- The main theorem stating that f(1996) is not divisible by 3 -/
theorem f_1996_not_divisible_by_3 : ¬ (3 ∣ f 1996) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1996_not_divisible_by_3_l16_1622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_675_l16_1668

/-- Represents a trapezoid EFGH -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  altitude : ℝ

/-- The area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  (t.EF + t.GH) * t.altitude / 2

theorem trapezoid_area_is_675 (EFGH : Trapezoid) 
    (h1 : EFGH.EF = 60)
    (h2 : EFGH.GH = 30)
    (h3 : EFGH.altitude = 15) :
  trapezoidArea EFGH = 675 := by
  -- Unfold the definition of trapezoidArea
  unfold trapezoidArea
  -- Substitute the given values
  rw [h1, h2, h3]
  -- Simplify the arithmetic
  norm_num

#check trapezoid_area_is_675

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_675_l16_1668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_two_equals_fourteen_l16_1632

-- Define the function g
noncomputable def g : ℝ → ℝ := fun _ => sorry

-- State the theorem
theorem g_of_two_equals_fourteen :
  (∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) → g 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_two_equals_fourteen_l16_1632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_of_oscillating_sine_l16_1683

noncomputable def oscillating_sine_function (a b c d : ℝ) : ℝ → ℝ := 
  fun x ↦ a * Real.sin (b * x + c) + d

theorem amplitude_of_oscillating_sine (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_max : ∀ x, oscillating_sine_function a b c d x ≤ 4)
  (h_min : ∀ x, oscillating_sine_function a b c d x ≥ -2)
  (h_reaches_max : ∃ x, oscillating_sine_function a b c d x = 4)
  (h_reaches_min : ∃ x, oscillating_sine_function a b c d x = -2) :
  a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_of_oscillating_sine_l16_1683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_li_number_quadratic_ratio_l16_1685

noncomputable def is_li_number (p : ℝ × ℝ) : Prop :=
  p.2 = -p.1

noncomputable def hyperbola (x : ℝ) : ℝ := -16 / x

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem li_number_quadratic_ratio (a b c : ℝ) (h₁ : a ≠ 0) :
  (∃! p : ℝ × ℝ, is_li_number p ∧ quadratic a b c p.1 = p.2) →
  (∃ x : ℝ, x < 0 ∧ is_li_number (x, hyperbola x) ∧ quadratic a b c x = hyperbola x) →
  c / a = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_li_number_quadratic_ratio_l16_1685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_theorem_l16_1653

/-- Definition of the ellipse -/
def is_on_ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the left focus -/
def left_focus (c : ℝ) : ℝ × ℝ := (-c, 0)

/-- Definition of a line passing through a point -/
def line_through_point (k : ℝ) (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - p.2 = k * (x - p.1)

/-- Definition of perpendicular lines -/
def perpendicular (k1 k2 : ℝ) : Prop :=
  k1 * k2 = -1

/-- Main theorem -/
theorem ellipse_line_theorem (a b c : ℝ) (A B C D : ℝ × ℝ) (k : ℝ) :
  (∀ (x y : ℝ), is_on_ellipse x y a b →
    (x, y) = A ∨ (x, y) = B ∨ (x, y) = C ∨ (x, y) = D) →
  line_through_point k (left_focus c) A.1 A.2 →
  line_through_point k (left_focus c) B.1 B.2 →
  perpendicular k (-1/k) →
  perpendicular ((C.2 - A.2) / (C.1 - A.1)) ((D.2 - A.2) / (D.1 - A.1)) →
  k = 1 ∨ k = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_theorem_l16_1653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_point_on_side_l16_1664

/-- Given a triangle ABC and a point M on BC, proves that AM can be expressed in terms of AB and AC -/
theorem vector_point_on_side (A B C M : EuclideanSpace ℝ (Fin 3)) (a b : EuclideanSpace ℝ (Fin 3)) : 
  (B - A = a) → 
  (C - A = b) → 
  (∃ t : ℝ, M = B + t • (C - B)) → 
  (M - B = 3 • (C - M)) → 
  (M - A = (1/4 : ℝ) • a + (3/4 : ℝ) • b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_point_on_side_l16_1664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l16_1626

/-- The sum of the infinite series ∑(k=1 to ∞) (2^(2k) / (3^(2k) - 1)) is equal to 1 -/
theorem infinite_series_sum : 
  ∑' k : ℕ, (2^(2*k.succ) : ℝ) / ((3^(2*k.succ) : ℝ) - 1) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l16_1626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_reduction_is_twenty_percent_l16_1693

/-- The average percentage reduction in price after two reductions -/
noncomputable def average_reduction (initial_price final_price : ℝ) : ℝ :=
  1 - (final_price / initial_price) ^ (1/2 : ℝ)

/-- Theorem stating that the average reduction is 20% given the initial and final prices -/
theorem average_reduction_is_twenty_percent :
  let initial_price : ℝ := 50
  let final_price : ℝ := 32
  average_reduction initial_price final_price = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_reduction_is_twenty_percent_l16_1693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_A_l16_1677

/-- A positive even single digit -/
def EvenSingleDigit : Type := { n : ℕ // n > 0 ∧ n < 10 ∧ n % 2 = 0 }

/-- The discriminant of the quadratic equation x^2 - (1A)x + A1 = 0 -/
def discriminant (A : EvenSingleDigit) : ℤ :=
  (A.val)^2 + 16*(A.val) + 60

/-- A perfect square integer -/
def is_perfect_square (n : ℤ) : Prop :=
  ∃ m : ℤ, n = m^2

/-- The main theorem -/
theorem no_valid_A : ¬∃ (A : EvenSingleDigit), is_perfect_square (discriminant A) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_A_l16_1677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_n_powered_consecutive_two_not_n_powered_consecutive_n_l16_1603

/-- Definition of n-powered number -/
def is_n_powered (x : ℕ) (n : ℕ) : Prop :=
  ∃ p : ℕ, x = p^n ∧ p > 0

/-- Theorem: Product of two consecutive positive integers is not n-powered -/
theorem not_n_powered_consecutive_two (n : ℕ) (h : n ≥ 2) :
  ∀ x : ℕ, x > 0 → ¬ is_n_powered (x * (x + 1)) n :=
sorry

/-- Theorem: Product of n consecutive positive integers is not n-powered -/
theorem not_n_powered_consecutive_n (n : ℕ) (h : n ≥ 2) :
  ∀ k : ℕ, k > 0 → ¬ is_n_powered (Finset.prod (Finset.range n) (λ i => k + i + 1)) n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_n_powered_consecutive_two_not_n_powered_consecutive_n_l16_1603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_athlete_triple_jump_l16_1639

/-- Represents an athlete's jump distances -/
structure AthleteJumps where
  long_jump : ℝ
  triple_jump : ℝ
  high_jump : ℝ

/-- Calculates the average jump distance for an athlete -/
noncomputable def average_jump (jumps : AthleteJumps) : ℝ :=
  (jumps.long_jump + jumps.triple_jump + jumps.high_jump) / 3

/-- The problem statement -/
theorem first_athlete_triple_jump 
  (first_athlete : AthleteJumps) 
  (second_athlete : AthleteJumps) :
  first_athlete.long_jump = 26 ∧ 
  first_athlete.high_jump = 7 ∧
  second_athlete.long_jump = 24 ∧ 
  second_athlete.triple_jump = 34 ∧ 
  second_athlete.high_jump = 8 ∧
  average_jump second_athlete = 22 →
  first_athlete.triple_jump = 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_athlete_triple_jump_l16_1639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_bd_length_l16_1615

/-- Right triangle ABC with B as the right angle -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

/-- Circle with diameter BC intersecting AC at D -/
def circle_intersect (t : RightTriangle) (D : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), 0 < k ∧ k < 1 ∧ D = (k * t.A.1 + (1 - k) * t.C.1, k * t.A.2 + (1 - k) * t.C.2)

/-- Length of a line segment -/
noncomputable def length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Area of a triangle -/
noncomputable def triangle_area (p q r : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((q.1 - p.1) * (r.2 - p.2) - (r.1 - p.1) * (q.2 - p.2))

theorem right_triangle_bd_length 
  (t : RightTriangle) 
  (D : ℝ × ℝ) 
  (h_circle : circle_intersect t D)
  (h_area : triangle_area t.A t.B t.C = 180)
  (h_ac : length t.A t.C = 30) :
  length t.B D = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_bd_length_l16_1615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_marathon_time_l16_1679

/-- Calculates the time taken to complete a marathon given the runner's average speed and the marathon distance. -/
noncomputable def marathonTime (averageSpeed : ℝ) (marathonDistance : ℝ) : ℝ :=
  marathonDistance / averageSpeed

/-- Theorem stating that Jack's marathon time is 5.5 hours given the specified conditions. -/
theorem jack_marathon_time
  (marathonDistance : ℝ)
  (jillTime : ℝ)
  (speedRatio : ℝ)
  (h1 : marathonDistance = 42)
  (h2 : jillTime = 4.2)
  (h3 : speedRatio = 0.7636363636363637)
  : marathonTime (speedRatio * (marathonDistance / jillTime)) marathonDistance = 5.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_marathon_time_l16_1679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_coordinate_G_l16_1609

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-4, 0)

-- Define a line passing through the left focus
noncomputable def line_through_focus (k : ℝ) (x : ℝ) : ℝ := k * (x + 4)

-- Define the y-coordinate of point G
noncomputable def y_coordinate_G (k : ℝ) : ℝ := -64 / (9/k + 25*k)

-- Theorem statement
theorem range_of_y_coordinate_G :
  ∀ k : ℝ, k ≠ 0 →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    ellipse x₁ (line_through_focus k x₁) ∧
    ellipse x₂ (line_through_focus k x₂)) →
  (y_coordinate_G k ∈ Set.Ioc (-32/15) 0 ∪ Set.Ioc 0 (32/15)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_y_coordinate_G_l16_1609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_circle_inequality_l16_1646

/-- The area of a triangle given its vertices --/
noncomputable def area_triangle (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := 
  sorry

/-- The area of the inscribed circle of a triangle --/
noncomputable def area_inscribed_circle (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := 
  sorry

/-- The area of the circumscribed circle of a triangle --/
noncomputable def area_circumscribed_circle (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := 
  sorry

/-- Predicate to check if an angle is a right angle --/
def RightAngle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

theorem right_triangle_circle_inequality (A B C : EuclideanSpace ℝ (Fin 2)) 
  (S : ℝ) (S₁ : ℝ) (S₂ : ℝ) 
  (h_right_angle : RightAngle A B C)
  (h_S : S = area_triangle A B C)
  (h_S₁ : S₁ = area_inscribed_circle A B C)
  (h_S₂ : S₂ = area_circumscribed_circle A B C) :
  π * (S - S₁) / S₂ < 1 / (π - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_circle_inequality_l16_1646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l16_1690

/-- The line y = 2x - 6 -/
noncomputable def line (x : ℝ) : ℝ := 2 * x - 6

/-- The point we're finding the closest point to -/
noncomputable def point : ℝ × ℝ := (3, 4)

/-- The claimed closest point on the line -/
noncomputable def closest_point : ℝ × ℝ := (69/15, -2/15)

/-- Theorem stating that closest_point is indeed the closest point on the line to point -/
theorem closest_point_is_closest :
  ∀ x : ℝ, 
  (x - point.1)^2 + (line x - point.2)^2 ≥ 
  (closest_point.1 - point.1)^2 + (closest_point.2 - point.2)^2 :=
by
  sorry

#check closest_point_is_closest

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l16_1690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_common_books_l16_1676

/-- The number of books in Ms. Carr's reading list -/
def total_books : ℕ := 12

/-- The number of books each student selects -/
def books_selected : ℕ := 6

/-- The number of books Harold and Betty have in common -/
def common_books : ℕ := 3

/-- Calculates the probability of Harold and Betty selecting exactly 3 books in common -/
noncomputable def probability_common_books : ℚ :=
  (Nat.choose total_books common_books * Nat.choose (total_books - common_books) (books_selected - common_books) ^ 2) /
  (Nat.choose total_books books_selected ^ 2)

theorem probability_three_common_books :
  probability_common_books = 405 / 2223 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_common_books_l16_1676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_in_scientific_notation_l16_1647

/-- Converts a number to scientific notation -/
noncomputable def to_scientific_notation (n : ℝ) : ℝ × ℤ :=
  let exponent := Real.log n / Real.log 10
  let mantissa := n / (10 ^ ⌊exponent⌋)
  (mantissa, ⌊exponent⌋)

/-- The amount invested in billions -/
def amount_invested : ℝ := 477.2

theorem investment_in_scientific_notation :
  to_scientific_notation (amount_invested * 10^9) = (4.772, 11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_in_scientific_notation_l16_1647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_theorem_l16_1658

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is on the circle -/
def isOnCircle (p : Point) (c : Circle) : Prop :=
  distance p c.center = c.radius

/-- The shortest path between two points on a circle that doesn't go inside the circle -/
noncomputable def shortestPathOnCircle (p1 p2 : Point) (c : Circle) : ℝ :=
  c.radius * Real.pi

theorem shortest_path_theorem (A D O : Point) (c : Circle) :
  A.x = 0 ∧ A.y = 0 ∧
  D.x = 18 ∧ D.y = 24 ∧
  O.x = 9 ∧ O.y = 12 ∧
  c.center = O ∧
  c.radius = 15 ∧
  isOnCircle A c ∧
  isOnCircle D c →
  shortestPathOnCircle A D c = 15 * Real.pi := by
  sorry

#check shortest_path_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_theorem_l16_1658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l16_1600

/-- The circle equation: x^2 + y^2 - 2x - 6y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y + 1 = 0

/-- The line equation: y = kx -/
def line_equation (k x y : ℝ) : Prop :=
  y = k * x

/-- Distance from a point (x, y) to the line y = kx -/
noncomputable def distance_to_line (k x y : ℝ) : ℝ :=
  abs (y - k*x) / Real.sqrt (k^2 + 1)

/-- There are exactly three points on the circle at distance 2 from the line -/
def three_points_condition (k : ℝ) : Prop :=
  ∃! (p1 p2 p3 : ℝ × ℝ),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    circle_equation p1.1 p1.2 ∧
    circle_equation p2.1 p2.2 ∧
    circle_equation p3.1 p3.2 ∧
    distance_to_line k p1.1 p1.2 = 2 ∧
    distance_to_line k p2.1 p2.2 = 2 ∧
    distance_to_line k p3.1 p3.2 = 2

theorem circle_line_intersection (k : ℝ) :
  three_points_condition k → k = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l16_1600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l16_1602

-- Define the set of rhombuses
def Rhombus : Type := sorry

-- Define the property of being symmetric about an axis
def SymmetricAboutAxis (R : Rhombus) : Prop := sorry

-- Define a power function
def PowerFunction : (ℝ → ℝ) → Prop := sorry

-- Define the property of passing through the origin
def PassesThroughOrigin (f : ℝ → ℝ) : Prop := f 0 = 0

-- Define necessary and sufficient condition
def NecessaryAndSufficient (p q : Prop) : Prop := p ↔ q

-- Define necessary but not sufficient condition
def NecessaryButNotSufficient (p q : Prop) : Prop :=
  (q → p) ∧ ¬(p → q)

theorem problem_solution :
  -- 1. Universal quantifier proposition
  (∀ R : Rhombus, SymmetricAboutAxis R) ∧
  -- 2. Existence of a power function not passing through origin
  (∃ f : ℝ → ℝ, PowerFunction f ∧ ¬PassesThroughOrigin f) ∧
  -- 3. Existence of x satisfying the inequality
  (∃ x : ℝ, x^2 - 4*x + 3 < 0) ∧
  -- 4. Logical implication
  (∀ p q r : Prop,
    ((p → q) ∧ ¬(q → p)) →
    (NecessaryAndSufficient q r) →
    (NecessaryButNotSufficient r p)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l16_1602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_high_scores_l16_1648

noncomputable def score_distribution (scores : List ℝ) : Prop :=
  scores.length = 20 ∧
  scores.all (λ x => 80 ≤ x ∧ x ≤ 100) ∧
  (scores.filter (λ x => 80 ≤ x ∧ x < 85)).length = 3 ∧
  (scores.filter (λ x => 85 ≤ x ∧ x < 90)).length = 4 ∧
  (scores.filter (λ x => 95 ≤ x ∧ x ≤ 100)).length = 8

noncomputable def mean_is_92 (scores : List ℝ) : Prop :=
  scores.sum / scores.length = 92

theorem estimate_high_scores (scores : List ℝ) (total_students : ℕ) :
  score_distribution scores →
  mean_is_92 scores →
  total_students = 2700 →
  ⌊(total_students : ℝ) * ((scores.filter (λ x => x ≥ 90)).length : ℝ) / 20⌋ = 1755 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_high_scores_l16_1648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l16_1611

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 3*x else -x^2 + 3*x

-- State the theorem
theorem solution_set_of_inequality :
  ∀ x : ℝ, f (x - 2) + f (x^2 - 4) < 0 ↔ -3 < x ∧ x < 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l16_1611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_to_hundredth_l16_1617

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The sum of 45.23 and 29.678, when rounded to the nearest hundredth, equals 74.91 -/
theorem sum_and_round_to_hundredth : roundToHundredth (45.23 + 29.678) = 74.91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_round_to_hundredth_l16_1617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l16_1618

/-- Given a train with speed in km/hr and time to cross a pole in seconds, 
    calculate its length in meters. -/
noncomputable def trainLength (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (1000 / 3600) * time

/-- Theorem stating that a train with speed 72 km/hr crossing a pole in 9 seconds 
    has a length of 180 meters. -/
theorem train_length_calculation :
  trainLength 72 9 = 180 := by
  -- Unfold the definition of trainLength
  unfold trainLength
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_div_assoc]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l16_1618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_is_19_l16_1665

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

noncomputable def expression : ℚ :=
  (factorial 13 * factorial 12 + factorial 12 * factorial 11 - factorial 11 * factorial 10) / 171

theorem greatest_prime_factor_is_19 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ expression.num.natAbs ∧ p ≥ 19 ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ expression.num.natAbs → q ≤ p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_is_19_l16_1665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_85_over_7_l16_1640

def point_A : Fin 3 → ℝ := ![3, -3, 4]
def point_B : Fin 3 → ℝ := ![1, 0, -2]
def point_C : Fin 3 → ℝ := ![0, 4, 0]

def line_direction : Fin 3 → ℝ :=
  ![point_C 0 - point_B 0, point_C 1 - point_B 1, point_C 2 - point_B 2]

noncomputable def distance_point_to_line (A B C : Fin 3 → ℝ) : ℝ :=
  sorry -- Definition of distance from point to line

theorem distance_point_to_line_is_85_over_7 :
  distance_point_to_line point_A point_B point_C = 85 / 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_85_over_7_l16_1640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_cube_roots_sum_l16_1661

theorem opposite_cube_roots_sum (a b : ℝ) : 
  |a - 27| + (b + 8)^2 = 0 → (a ^ (1/3 : ℝ)) + (b ^ (1/3 : ℝ)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_cube_roots_sum_l16_1661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_circle_radius_l16_1633

-- Define the line
def line (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

-- Define the circle
def circle_eq (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Define the angle between OA and OB
noncomputable def angle_AOB (xA yA xB yB : ℝ) : ℝ := 
  Real.arccos ((xA * xB + yA * yB) / (Real.sqrt (xA^2 + yA^2) * Real.sqrt (xB^2 + yB^2)))

-- Theorem statement
theorem intersection_line_circle_radius 
  (xA yA xB yB r : ℝ) 
  (hr : r > 0)
  (hline_A : line xA yA)
  (hline_B : line xB yB)
  (hcircle_A : circle_eq xA yA r)
  (hcircle_B : circle_eq xB yB r)
  (hangle : angle_AOB xA yA xB yB = 2 * Real.pi / 3) : 
  r = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_circle_radius_l16_1633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_cube_triangle_area_bound_l16_1638

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition of a unit cube -/
def UnitCube : Set Point3D :=
  { p : Point3D | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1 ∧ 0 ≤ p.z ∧ p.z ≤ 1 }

/-- Check if three points are collinear -/
def areCollinear (p1 p2 p3 : Point3D) : Prop :=
  ∃ (t : ℝ), p3.x - p1.x = t * (p2.x - p1.x) ∧
             p3.y - p1.y = t * (p2.y - p1.y) ∧
             p3.z - p1.z = t * (p2.z - p1.z)

/-- Calculate the area of a triangle formed by three points -/
noncomputable def triangleArea (p1 p2 p3 : Point3D) : ℝ :=
  let v1 := Point3D.mk (p2.x - p1.x) (p2.y - p1.y) (p2.z - p1.z)
  let v2 := Point3D.mk (p3.x - p1.x) (p3.y - p1.y) (p3.z - p1.z)
  let crossProduct := Point3D.mk
    (v1.y * v2.z - v1.z * v2.y)
    (v1.z * v2.x - v1.x * v2.z)
    (v1.x * v2.y - v1.y * v2.x)
  Real.sqrt (crossProduct.x^2 + crossProduct.y^2 + crossProduct.z^2) / 2

/-- The main theorem -/
theorem unit_cube_triangle_area_bound
  (points : Finset Point3D)
  (h_in_cube : ∀ p ∈ points, p ∈ UnitCube)
  (h_count : points.card = 75)
  (h_not_collinear : ∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points →
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬areCollinear p1 p2 p3) :
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ triangleArea p1 p2 p3 ≤ 7/72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_cube_triangle_area_bound_l16_1638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l16_1662

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence (α : Type*) [AddCommGroup α] [Mul α] where
  a : ℕ → α
  d : α
  seq_def : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence ℝ) (n : ℕ) : ℝ :=
  n * (seq.a 1 + seq.a n) / 2

/-- Theorem stating the ratio of sums for an arithmetic sequence -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequence ℝ) 
  (h1 : seq.a 1 ≠ 0) 
  (h2 : seq.a 2 = 3 * seq.a 1) : 
  sum_n seq 10 / sum_n seq 5 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l16_1662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_selection_l16_1654

/-- Represents a table tennis association -/
structure Association where
  name : String
  athletes : ℕ

/-- Represents a selected athlete -/
inductive SelectedAthlete
| A1 | A2 | A3 | A4 | A5 | A6

/-- The main theorem -/
theorem table_tennis_selection 
  (assoc_A assoc_B assoc_C : Association)
  (total_selected : ℕ)
  (stratified_sampling : Association → ℕ)
  (possible_outcomes : Finset (SelectedAthlete × SelectedAthlete))
  (event_A : Finset (SelectedAthlete × SelectedAthlete)) :
  assoc_A.athletes = 27 →
  assoc_B.athletes = 9 →
  assoc_C.athletes = 18 →
  total_selected = 6 →
  (∀ a, stratified_sampling a = (a.athletes * total_selected) / (assoc_A.athletes + assoc_B.athletes + assoc_C.athletes)) →
  possible_outcomes.card = 15 →
  event_A = {p : SelectedAthlete × SelectedAthlete | p.1 = SelectedAthlete.A5 ∨ p.1 = SelectedAthlete.A6 ∨ p.2 = SelectedAthlete.A5 ∨ p.2 = SelectedAthlete.A6} →
  stratified_sampling assoc_A = 3 ∧
  stratified_sampling assoc_B = 1 ∧
  stratified_sampling assoc_C = 2 ∧
  (event_A.card : ℚ) / possible_outcomes.card = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_selection_l16_1654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l16_1628

/-- Given complex numbers z₁ and z₂ satisfying certain conditions, prove specific values for z₁, z₂, and their product. -/
theorem complex_number_properties (z₁ z₂ : ℂ) :
  (Complex.abs z₁ = 1 ∧ Complex.abs z₂ = 1) →
  (z₁ - z₂ = (Real.sqrt 6 / 3 : ℂ) + (Real.sqrt 3 / 3 : ℂ) * Complex.I) →
  (z₁ + z₂ = 12 / 13 - (5 / 13) * Complex.I) →
  (z₁ = ((Real.sqrt 6 / 6 + 1 / 2 : ℂ) + (Real.sqrt 3 / 6 - Real.sqrt 2 / 2 : ℂ) * Complex.I) ∧
   z₂ = ((-Real.sqrt 6 / 6 + 1 / 2 : ℂ) + (-Real.sqrt 3 / 6 - Real.sqrt 2 / 2 : ℂ) * Complex.I)) ∧
  (z₁ * z₂ = 119 / 169 - (120 / 169) * Complex.I) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l16_1628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_side_length_l16_1659

/-- Represents a right pyramid with a square base -/
structure RightPyramid where
  base_side : ℝ
  slant_height : ℝ

/-- Calculates the area of one lateral face of the pyramid -/
noncomputable def lateral_face_area (p : RightPyramid) : ℝ :=
  (1 / 2) * p.base_side * p.slant_height

theorem pyramid_base_side_length :
  ∃ (p : RightPyramid), p.slant_height = 40 ∧ lateral_face_area p = 120 ∧ p.base_side = 6 := by
  -- Construct the pyramid
  let p : RightPyramid := ⟨6, 40⟩
  -- Prove it satisfies the conditions
  have h1 : p.slant_height = 40 := rfl
  have h2 : lateral_face_area p = 120 := by
    simp [lateral_face_area]
    norm_num
  have h3 : p.base_side = 6 := rfl
  -- Conclude the proof
  exact ⟨p, h1, h2, h3⟩

#check pyramid_base_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_side_length_l16_1659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l16_1688

theorem election_votes (winning_percentage : Real) (vote_majority : ℕ) (total_votes : ℕ) : 
  winning_percentage = 0.7 →
  vote_majority = 172 →
  (winning_percentage - (1 - winning_percentage)) * total_votes = vote_majority →
  total_votes = 430 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l16_1688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cattle_transport_l16_1697

/-- Calculates the number of cattle moved given transport conditions -/
theorem cattle_transport (distance : ℕ) (truck_capacity : ℕ) (speed : ℕ) (total_time : ℕ) : 
  distance = 60 ∧ truck_capacity = 20 ∧ speed = 60 ∧ total_time = 40 →
  (total_time * speed * truck_capacity) / (2 * distance) = 400 := by
  intro h
  -- The proof steps would go here
  sorry

#check cattle_transport

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cattle_transport_l16_1697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partitioning_pairs_characterization_l16_1673

/-- A pair of positive integers (a, b) satisfies the partitioning property if there exists an integer N
    such that for any integers m ≥ N and n ≥ N, every m × n grid of unit squares can be partitioned
    into a × b rectangles and fewer than ab unit squares. -/
def satisfies_partitioning_property (a b : ℕ+) : Prop :=
  ∃ N : ℕ, ∀ m n : ℕ, m ≥ N → n ≥ N →
    ∃ (k r : ℕ), m * n = k * (a * b) + r ∧ r < a * b

/-- The set of all pairs (a, b) that satisfy the partitioning property. -/
def partitioning_pairs : Set (ℕ+ × ℕ+) :=
  {p | satisfies_partitioning_property p.1 p.2}

/-- The theorem stating that the set of pairs satisfying the partitioning property
    is exactly {(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)}. -/
theorem partitioning_pairs_characterization :
  partitioning_pairs = {(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partitioning_pairs_characterization_l16_1673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_implies_logarithm_l16_1670

theorem exponential_inequality_implies_logarithm (x y : ℝ) :
  (2 : ℝ)^x - (2 : ℝ)^y < (3 : ℝ)^(-x) - (3 : ℝ)^(-y) → Real.log (y - x + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_implies_logarithm_l16_1670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_l16_1607

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := Real.log (x * y) / Real.log (Real.sqrt x) = 8
def equation2 (x y : ℝ) : Prop := Real.log (Real.log (x / y) / Real.log (1/9)) / Real.log 3 = 0

-- State the theorem
theorem solution_satisfies_system :
  ∃ (x y : ℝ), x > 0 ∧ x ≠ 1 ∧ y > 0 ∧ 0 < x / y ∧ x / y < 1 ∧
  equation1 x y ∧ equation2 x y ∧ x = 3 ∧ y = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_l16_1607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_midline_and_angles_area_is_valid_triangle_l16_1634

/-- The area of a triangle given its midline length and angles with sides -/
theorem triangle_area_from_midline_and_angles 
  (k φ ψ : ℝ) (hk : k > 0) (hφ : 0 < φ ∧ φ < π) (hψ : 0 < ψ ∧ ψ < π) :
  ∃ t : ℝ, t = (2 * k^2 * Real.sin φ * Real.sin ψ) / Real.sin (φ + ψ) ∧ 
  t > 0 := by
  sorry

/-- Predicate to represent that a real number is the area of a triangle -/
def represents_triangle_area (t : ℝ) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  t = Real.sqrt (((a + b + c) / 2) * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))

/-- The area calculated from midline and angles represents a valid triangle area -/
theorem area_is_valid_triangle 
  (k φ ψ : ℝ) (hk : k > 0) (hφ : 0 < φ ∧ φ < π) (hψ : 0 < ψ ∧ ψ < π) :
  ∃ t : ℝ, t = (2 * k^2 * Real.sin φ * Real.sin ψ) / Real.sin (φ + ψ) ∧ 
  t > 0 ∧ represents_triangle_area t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_midline_and_angles_area_is_valid_triangle_l16_1634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l16_1682

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x^3 / Real.log 3 + Real.log x / Real.log 9 = 6 → x = 3^(12/7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l16_1682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_four_l16_1675

/-- A power function passing through (2, √2/2) -/
noncomputable def f (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

/-- The function passes through the point (2, √2/2) -/
axiom f_passes_through : ∃ α : ℝ, f α 2 = Real.sqrt 2 / 2

theorem f_at_four : ∃ α : ℝ, f α 2 = Real.sqrt 2 / 2 → f α 4 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_four_l16_1675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_angle_formulas_l16_1656

theorem triple_angle_formulas (θ : Real) (h : Real.tan θ = 5) :
  Real.tan (3 * θ) = 55 / 37 ∧
  (Real.sin (3 * θ) = (55 * Real.sqrt 1369) / (37 * Real.sqrt 4394) ∨
   Real.sin (3 * θ) = -(55 * Real.sqrt 1369) / (37 * Real.sqrt 4394)) ∧
  (Real.cos (3 * θ) = Real.sqrt (1369 / 4394) ∨
   Real.cos (3 * θ) = -Real.sqrt (1369 / 4394)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_angle_formulas_l16_1656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_from_focus_l16_1620

/-- A parabola with focus at (-2, 0) has the standard equation y^2 = -8x -/
theorem parabola_equation_from_focus (x y : ℝ) :
  (∃ (F : ℝ × ℝ), F = (-2, 0) ∧ (x - F.1)^2 + y^2 = (x + F.1)^2) →
  y^2 = -8*x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_from_focus_l16_1620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_even_decreasing_function_l16_1696

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is monotonically decreasing on (0,+∞) if
    for all x, y ∈ (0,+∞), x < y implies f(x) > f(y) -/
def IsMonoDecreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x > f y

theorem solution_set_of_even_decreasing_function
  (f : ℝ → ℝ)
  (heven : IsEven f)
  (hdecr : IsMonoDecreasingOn f (Set.Ioi 0))
  (hf1 : f 1 = 0) :
  {x : ℝ | f x > 0} = Set.Ioo (-1) 0 ∪ Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_even_decreasing_function_l16_1696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_spend_theorem_l16_1674

def item_prices : List Nat := [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]

def promotion (group : List Nat) : Nat :=
  if group.length = 3 then group.sum - (group.minimum?.getD 0) else group.sum

def min_spend (prices : List Nat) : Nat :=
  let groups := prices.reverse.groupBy 3
  (groups.map promotion).sum

theorem min_spend_theorem :
  min_spend item_prices = 4800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_spend_theorem_l16_1674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l16_1621

/-- An isosceles trapezoid with height h and lateral side seen from the center of the circumscribed circle at an angle of 60° -/
structure IsoscelesTrapezoid where
  h : ℝ
  angle : ℝ
  angle_eq : angle = 60 * π / 180

/-- The area of an isosceles trapezoid -/
noncomputable def area (t : IsoscelesTrapezoid) : ℝ :=
  t.h^2 * Real.sqrt 3

/-- Theorem: The area of an isosceles trapezoid with height h and lateral side seen from the center of the circumscribed circle at an angle of 60° is h²√3 -/
theorem isosceles_trapezoid_area (t : IsoscelesTrapezoid) : area t = t.h^2 * Real.sqrt 3 := by
  -- Unfold the definition of area
  unfold area
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l16_1621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l16_1627

def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  (b + c - a) * (Real.sin A + Real.sin B + Real.sin C) = c * Real.sin B ∧
  a = 2 * Real.sqrt 3 ∧
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem triangle_theorem (a b c A B C : ℝ) (h : Triangle a b c A B C) :
  A = 2 * Real.pi / 3 ∧
  ∃ (max : ℝ), max = 4 * Real.sqrt 3 ∧
    ∀ (B' C' : ℝ), Triangle a b c A B' C' →
      (1 / 2) * b * c * Real.sin A + 4 * Real.sqrt 3 * Real.cos B' * Real.cos C' ≤ max :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l16_1627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_subinterval_l16_1636

theorem zero_in_subinterval (g : ℝ → ℝ) (h1 : g 2013 = 0) (h2 : g 2014 = 0)
  (h3 : ∀ (a b : ℝ), a ∈ Set.Icc 2013 2014 → b ∈ Set.Icc 2013 2014 → g ((a + b) / 2) ≤ g a + g b) :
  ∀ (c d : ℝ), c ∈ Set.Ioo 2013 2014 → d ∈ Set.Ioo 2013 2014 → ∃ (x : ℝ), x ∈ Set.Ioo c d ∧ g x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_subinterval_l16_1636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_commutation_result_l16_1650

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, -1; 5, 2]

noncomputable def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]

theorem matrix_commutation_result (x y z w : ℝ) 
  (h1 : A * B x y z w = B x y z w * A)
  (h2 : 5 * y ≠ z) :
  (x - w) / (z - 5 * y) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_commutation_result_l16_1650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l16_1667

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The standard equation of the ellipse -/
theorem ellipse_equation (C : Ellipse) (A F : Point) :
  A.x = 2 ∧ A.y = 3 ∧  -- Point A(2,3) lies on the ellipse
  F.x = 2 ∧ F.y = 0 ∧  -- F(2,0) is the right focus
  pointOnEllipse A C ∧
  C.a^2 - C.b^2 = 4 →  -- Distance between foci is 4
  C.a^2 = 16 ∧ C.b^2 = 12 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l16_1667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matthew_holdings_satisfy_conditions_more_expensive_stock_price_l16_1691

/-- Represents the price of stocks and Matthew's holdings --/
structure StockHoldings where
  less_expensive_price : ℚ
  more_expensive_price : ℚ
  less_expensive_shares : ℕ
  more_expensive_shares : ℕ
  total_assets : ℚ

/-- The conditions of Matthew's stock holdings --/
def matthew_holdings : StockHoldings where
  less_expensive_price := 39
  more_expensive_price := 78
  less_expensive_shares := 26
  more_expensive_shares := 14
  total_assets := 2106

/-- Theorem stating that Matthew's stock holdings satisfy the given conditions --/
theorem matthew_holdings_satisfy_conditions :
  let h := matthew_holdings
  h.more_expensive_price = 2 * h.less_expensive_price ∧
  h.more_expensive_shares = 14 ∧
  h.less_expensive_shares = 26 ∧
  h.total_assets = 2106 ∧
  h.total_assets = h.more_expensive_price * h.more_expensive_shares +
                   h.less_expensive_price * h.less_expensive_shares :=
by sorry

/-- Theorem proving that the more expensive stock is $78 per share --/
theorem more_expensive_stock_price :
  matthew_holdings.more_expensive_price = 78 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matthew_holdings_satisfy_conditions_more_expensive_stock_price_l16_1691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_lateral_area_inequality_volume_lateral_area_equality_condition_l16_1606

/-- Definition of a right circular cone -/
structure RightCircularCone where
  radius : ℝ
  height : ℝ

/-- Volume of a right circular cone -/
noncomputable def volume (cone : RightCircularCone) : ℝ :=
  (1 / 3) * Real.pi * cone.radius^2 * cone.height

/-- Lateral area of a right circular cone -/
noncomputable def lateralArea (cone : RightCircularCone) : ℝ :=
  Real.pi * cone.radius * Real.sqrt (cone.radius^2 + cone.height^2)

/-- Theorem: The inequality between volume and lateral area of a right circular cone -/
theorem volume_lateral_area_inequality (cone : RightCircularCone) :
  ((6 * volume cone) / Real.pi)^2 ≤ ((2 * lateralArea cone) / (Real.pi * Real.sqrt 3))^3 := by
  sorry

/-- Theorem: Condition for equality in the volume-lateral area inequality -/
theorem volume_lateral_area_equality_condition (cone : RightCircularCone) :
  ((6 * volume cone) / Real.pi)^2 = ((2 * lateralArea cone) / (Real.pi * Real.sqrt 3))^3 ↔
  2 * cone.radius^2 = cone.height^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_lateral_area_inequality_volume_lateral_area_equality_condition_l16_1606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fragment_probability_l16_1681

/-- The length of the decimal code -/
def codeLength : ℕ := 21

/-- The length of the fragment "0123456789" -/
def fragmentLength : ℕ := 10

/-- The number of possible digits (0 to 9) -/
def digitCount : ℕ := 10

/-- Helper function to calculate the probability of fragment occurrence -/
noncomputable def probability_of_fragment_occurrence (codeLen fragLen digCount : ℕ) : ℚ := 
  (12 * digCount^11 - 30) / digCount^codeLen

/-- The probability of the fragment "0123456789" appearing in a 21-digit decimal code -/
theorem fragment_probability : 
  probability_of_fragment_occurrence codeLength fragmentLength digitCount = 
  (12 * digitCount^11 - 30 : ℚ) / digitCount^codeLength := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fragment_probability_l16_1681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l16_1644

-- Define the original function
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the proposed inverse function
noncomputable def f_inv (x : ℝ) : ℝ := -Real.sqrt (x - 2)

-- State the theorem
theorem inverse_function_theorem :
  (∀ x ∈ Set.Icc (-1 : ℝ) 0, f x ∈ Set.Icc 2 3) ∧
  (∀ y ∈ Set.Icc 2 3, f_inv y ∈ Set.Icc (-1 : ℝ) 0) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 0, f_inv (f x) = x) ∧
  (∀ y ∈ Set.Icc 2 3, f (f_inv y) = y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l16_1644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_power_negative_four_solutions_l16_1629

def solutions : Set ℂ := {1 + Complex.I, 1 - Complex.I, -1 + Complex.I, -1 - Complex.I}

theorem fourth_power_negative_four_solutions :
  ∀ z : ℂ, z^4 = -4 ↔ z ∈ solutions := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_power_negative_four_solutions_l16_1629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l16_1605

/-- The length of a train in meters, given its speed in km/h and the time it takes to cross a pole -/
noncomputable def train_length (speed_kmph : ℝ) (time_seconds : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600) * time_seconds

/-- Theorem: A train traveling at 90 km/h that crosses a pole in 5 seconds has a length of 125 meters -/
theorem train_length_calculation :
  train_length 90 5 = 125 := by
  -- Unfold the definition of train_length
  unfold train_length
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- Check that the result is equal to 125
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l16_1605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_meeting_time_l16_1695

/-- Calculates the least common multiple (LCM) of three natural numbers -/
def lcm3 (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

/-- Represents a racing vehicle with a lap time in seconds -/
structure RacingVehicle where
  lapTime : ℕ

/-- Calculates the time (in minutes) for multiple vehicles to meet at the starting point for the second time -/
def timeToSecondMeeting (vehicles : List RacingVehicle) : ℚ :=
  match vehicles with
  | [v1, v2, v3] => (2 * (lcm3 v1.lapTime v2.lapTime v3.lapTime)) / 60
  | _ => 0

theorem second_meeting_time :
  let racingMagic : RacingVehicle := { lapTime := 120 }
  let chargingBull : RacingVehicle := { lapTime := 90 }
  let thunderStorm : RacingVehicle := { lapTime := 150 }
  let vehicles := [racingMagic, chargingBull, thunderStorm]
  timeToSecondMeeting vehicles = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_meeting_time_l16_1695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_of_line_l16_1610

/-- Definition of InclinationAngle for a line in 2D space -/
def InclinationAngle (line : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The inclination angle of the line x sin(2θ) - y cos(2θ) = 0 is π/2 -/
theorem inclination_angle_of_line (θ : ℝ) :
  let line := {(x, y) : ℝ × ℝ | x * Real.sin (2 * θ) - y * Real.cos (2 * θ) = 0}
  InclinationAngle line = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_of_line_l16_1610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_event_proof_l16_1604

/-- Calculates the number of full recipes needed for a cookie event --/
noncomputable def cookie_recipes_needed (total_students : ℕ) (attendance_drop : ℚ) 
  (cookies_per_student : ℕ) (cookies_per_recipe : ℕ) : ℕ :=
let attending_students := (total_students : ℚ) * (1 - attendance_drop)
let total_cookies_needed := attending_students * (cookies_per_student : ℚ)
let recipes_needed := total_cookies_needed / (cookies_per_recipe : ℚ)
Int.ceil recipes_needed |>.toNat

/-- Proves that 9 full recipes are needed for the given conditions --/
theorem cookie_event_proof : 
  cookie_recipes_needed 125 (2/5) 2 18 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_event_proof_l16_1604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_is_270_l16_1645

/-- Represents the side lengths of the nine squares in the rectangle. -/
structure SquareSides where
  b1 : ℕ+
  b2 : ℕ+
  b3 : ℕ+
  b4 : ℕ+
  b5 : ℕ+
  b6 : ℕ+
  b7 : ℕ+
  b8 : ℕ+
  b9 : ℕ+

/-- Represents the dimensions of the rectangle. -/
structure RectangleDimensions where
  length : ℕ+
  width : ℕ+

/-- Checks if the square sides form the specific pattern described in the problem. -/
def valid_square_pattern (s : SquareSides) : Prop :=
  s.b1 + s.b2 = s.b3 ∧
  s.b1 + s.b3 = s.b4 ∧
  s.b3 + s.b4 = s.b5 ∧
  s.b4 + s.b5 = s.b6 ∧
  s.b2 + s.b3 + s.b5 = s.b7 ∧
  s.b2 + s.b7 = s.b8 ∧
  s.b1 + s.b4 + s.b6 = s.b9 ∧
  s.b6 + s.b9 = s.b7 + s.b8

/-- Checks if the rectangle dimensions match the square sides. -/
def valid_rectangle_dimensions (s : SquareSides) (r : RectangleDimensions) : Prop :=
  r.length = s.b6 + s.b9 ∧ r.width = s.b7 + s.b8

/-- The main theorem stating that a rectangle with the given properties has a perimeter of 270. -/
theorem rectangle_perimeter_is_270 
    (s : SquareSides) 
    (r : RectangleDimensions) 
    (h1 : valid_square_pattern s) 
    (h2 : valid_rectangle_dimensions s r) 
    (h3 : Nat.Coprime r.length.val r.width.val) : 
  2 * (r.length + r.width) = 270 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_is_270_l16_1645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_on_unit_segment_l16_1631

/-- The maximum number of points that can be placed on a unit segment 
    with the given density constraint -/
def max_points : ℕ := 32

/-- The constraint function for the number of points on a subsegment -/
def point_constraint (d : ℝ) : ℝ := 1 + 1000 * d^2

theorem max_points_on_unit_segment :
  ∀ (n : ℕ) (point_positions : Fin n → ℝ),
  (∀ i, 0 ≤ point_positions i ∧ point_positions i ≤ 1) →
  (∀ a b : ℝ, 0 ≤ a → a < b → b ≤ 1 →
    (Finset.filter (λ i ↦ a ≤ point_positions i ∧ point_positions i ≤ b) Finset.univ).card ≤ 
    ⌊point_constraint (b - a)⌋) →
  n ≤ max_points :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_points_on_unit_segment_l16_1631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_product_15625_l16_1657

def divisor_product (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem divisor_product_15625 (n : ℕ) :
  divisor_product n = 15625 → n = 3125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_product_15625_l16_1657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_square_plots_l16_1694

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℝ
  length : ℝ

/-- Represents the available internal fencing -/
def available_fencing : ℝ := 2430

/-- Calculates the number of square plots given the number of plots along the width -/
noncomputable def num_plots (n : ℝ) : ℝ := n * (3/2 * n)

/-- Calculates the required internal fencing for a given number of plots along the width -/
noncomputable def required_fencing (n : ℝ) : ℝ := 90 * n - 75

/-- The main theorem stating the maximum number of square test plots -/
theorem max_square_plots (field : FieldDimensions)
    (h_width : field.width = 30)
    (h_length : field.length = 45)
    (h_fencing : required_fencing 27 ≤ available_fencing)
    (h_max : ∀ m : ℝ, m > 27 → required_fencing m > available_fencing) :
    ⌊num_plots 27⌋ = 1093 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_square_plots_l16_1694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_difference_l16_1669

/-- Given vectors a and b, prove that the maximum value of their difference is 3 -/
theorem max_vector_difference (θ : ℝ) :
  let a : Fin 2 → ℝ := ![Real.cos θ, Real.sin θ]
  let b : Fin 2 → ℝ := ![Real.sqrt 3, 1]
  (∀ φ : ℝ, ‖a - b‖ ≤ ‖![Real.cos φ, Real.sin φ] - b‖) →
  ‖a - b‖ = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vector_difference_l16_1669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_greater_than_two_l16_1671

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log (x - 1) - a

-- State the theorem
theorem f_has_zero_greater_than_two (a : ℝ) (h : a > 0) :
  ∃ x₀ : ℝ, x₀ > 2 ∧ f a x₀ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_greater_than_two_l16_1671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_candidates_count_l16_1689

theorem exam_candidates_count :
  ∀ (total_candidates : ℕ) (boys : ℕ),
    -- Given conditions
    total_candidates = boys + 900 →
    (0.28 * (boys : ℝ) + 0.32 * 900) / (total_candidates : ℝ) = 1 - 0.702 →
    -- Conclusion
    total_candidates = 2000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_candidates_count_l16_1689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_formula_l16_1616

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  h_geom : ∀ n, a (n + 1) = q * a n  -- Geometric sequence property

/-- Sum of first n terms of a geometric sequence -/
noncomputable def GeometricSum (g : GeometricSequence) (n : ℕ) : ℝ :=
  (g.a 1) * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_formula (g : GeometricSequence) 
  (h_q : g.q < 1)
  (h_a3 : g.a 3 = 1)
  (h_sum : GeometricSum g 4 = 5 * GeometricSum g 2) :
  (∃ c, ∀ n, g.a n = c * (-1)^(n-1)) ∨ 
  (∃ c, ∀ n, g.a n = c * (-2)^(n-1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_formula_l16_1616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_ratio_l16_1660

theorem product_sum_ratio : (Nat.factorial 10) / ((List.range 10).map (· + 1)).sum * (1 / 2) = 33000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_ratio_l16_1660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_tethered_area_l16_1698

/-- The area a point can reach when tethered to the corner of a rectangle --/
noncomputable def tethered_area (rectangle_width : ℝ) (rectangle_length : ℝ) (tether_length : ℝ) : ℝ :=
  let full_circle_area := Real.pi * tether_length^2
  let large_arc_area := (3/4) * full_circle_area
  let small_sector_radius := tether_length - rectangle_length
  let small_sector_area := (1/4) * Real.pi * small_sector_radius^2
  large_arc_area + small_sector_area

/-- Theorem stating the area a point can reach when tethered to a specific rectangle --/
theorem specific_tethered_area :
  tethered_area 3 4 5 = 19 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_tethered_area_l16_1698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_freight_problem_l16_1641

-- Define the freight charge function for Train A
noncomputable def freight_charge_A (weight : ℝ) : ℝ :=
  if weight ≤ 1800 then 10 * weight
  else if weight ≤ 3000 then 18000 + 15 * (weight - 1800)
  else 36000 + 20 * (weight - 3000)

-- Define the freight charge function for Train B
def freight_charge_B (weight : ℝ) : ℝ := 18 * weight

theorem freight_problem :
  ∀ x y : ℝ,
  (x + 1.5 * x = 7500) →
  (freight_charge_B y - freight_charge_A y = 17400) →
  ((x = 3000 ∧ 1.5 * x = 4500) ∧ (y = 2800 ∨ y = 3300)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_freight_problem_l16_1641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l16_1637

-- Use the built-in complex number type
open Complex

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + i) = 2 - 2*i) : 
  z.im = -2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l16_1637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_b1_b2_l16_1699

def sequence_b (b₁ b₂ : ℕ+) : ℕ → ℕ+
  | 0 => b₁
  | 1 => b₂
  | (n + 2) => ⟨(sequence_b b₁ b₂ n + 2021) / (1 + sequence_b b₁ b₂ (n + 1)), sorry⟩

def all_distinct (b₁ b₂ : ℕ+) : Prop :=
  ∀ m n, m ≠ n → sequence_b b₁ b₂ m ≠ sequence_b b₁ b₂ n

theorem min_sum_b1_b2 :
  ∃ b₁ b₂ : ℕ+,
    (∀ n, (sequence_b b₁ b₂ n : ℕ) = sequence_b b₁ b₂ n) ∧
    all_distinct b₁ b₂ ∧
    ∀ b₁' b₂' : ℕ+,
      (∀ n, (sequence_b b₁' b₂' n : ℕ) = sequence_b b₁' b₂' n) →
      all_distinct b₁' b₂' →
      (b₁ : ℕ) + (b₂ : ℕ) ≤ (b₁' : ℕ) + (b₂' : ℕ) ∧
      (b₁ : ℕ) + (b₂ : ℕ) = 90 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_b1_b2_l16_1699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_l16_1692

-- Define the circle
variable (circle : Set (ℝ × ℝ))

-- Define the points on the circle
variable (M N P : ℝ × ℝ)

-- Define the property that a point is on the circle
def on_circle (point : ℝ × ℝ) (circle : Set (ℝ × ℝ)) : Prop :=
  point ∈ circle

-- Define the triangle
variable (A B C : ℝ × ℝ)

-- Define the property that a triangle is inscribed in the circle
def inscribed_triangle (A B C : ℝ × ℝ) (circle : Set (ℝ × ℝ)) : Prop :=
  on_circle A circle ∧ on_circle B circle ∧ on_circle C circle

-- Define the altitude, angle bisector, and median
def altitude (A B C M : ℝ × ℝ) : Prop := sorry
def angle_bisector (A B C N : ℝ × ℝ) : Prop := sorry
def median (A B C P : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem triangle_construction (h1 : on_circle M circle)
                              (h2 : on_circle N circle)
                              (h3 : on_circle P circle) :
  ∃ A B C, inscribed_triangle A B C circle ∧
           altitude A B C M ∧
           angle_bisector A B C N ∧
           median A B C P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_l16_1692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zero_position_l16_1655

/-- The function f(x) = (1/3)^x - ln(x) -/
noncomputable def f (x : ℝ) : ℝ := (1/3)^x - Real.log x

/-- The theorem statement -/
theorem function_zero_position 
  (a b c d : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : c > 0) 
  (h4 : f a * f b * f c < 0) 
  (h5 : f d = 0) : 
  (d < a) ∨ (d > b) ∨ (d < c) ∨ (d > c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zero_position_l16_1655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_A_incorrect_statement_D_incorrect_A_and_D_incorrect_l16_1643

-- Define the curve C
def C (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (3 - t) + p.2^2 / (t - 1) = 1}

-- Helper definitions
def IsEllipse (S : Set (ℝ × ℝ)) : Prop := sorry
def IsHyperbola (S : Set (ℝ × ℝ)) : Prop := sorry

-- Statement A is incorrect
theorem statement_A_incorrect :
  ¬ (∀ t : ℝ, 1 < t ∧ t < 3 → IsEllipse (C t)) := by
  sorry

-- Statement D is incorrect
theorem statement_D_incorrect :
  ¬ (∀ t : ℝ, IsHyperbola (C t) → t < 1) := by
  sorry

-- Proof that both A and D are incorrect
theorem A_and_D_incorrect :
  (¬ (∀ t : ℝ, 1 < t ∧ t < 3 → IsEllipse (C t))) ∧
  (¬ (∀ t : ℝ, IsHyperbola (C t) → t < 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_A_incorrect_statement_D_incorrect_A_and_D_incorrect_l16_1643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l16_1623

-- Define the power function as noncomputable
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem power_function_theorem (α : ℝ) :
  (f α 2 = 4) →
  (∀ x : ℝ, f α x = x^2) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f α x₁ < f α x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l16_1623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_specific_point_l16_1652

/-- Given a real number a, define the function f -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((x + 1)^2 + a * Real.sin x) / (x^2 + 1) + 3

/-- The main theorem to prove -/
theorem function_value_at_specific_point (a : ℝ) :
  f a (Real.log (Real.log 5 / Real.log 2)) = 5 →
  f a (Real.log (Real.log 2 / Real.log 5)) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_specific_point_l16_1652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_horizontal_asymptote_l16_1630

noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 8 * x - 10) / (x^2 - 5 * x + 4)

theorem cross_horizontal_asymptote :
  ∃ (x : ℝ), x = 22 / 7 ∧ g x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_horizontal_asymptote_l16_1630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_letter_pair_exists_l16_1684

/-- Represents an English letter -/
def Letter : Type := Char

/-- Represents a positive integer -/
def PositiveInteger : Type := Nat

/-- Checks if a letter is present in the English spelling of a number -/
def letterInSpelling (l : Letter) (n : PositiveInteger) : Prop := sorry

/-- Represents the factorization of a number into two positive integers -/
def numberFactorization (n : PositiveInteger) : PositiveInteger × PositiveInteger := sorry

/-- The main theorem stating the existence and uniqueness of the letter pair -/
theorem unique_letter_pair_exists :
  ∃! (pair : Letter × Letter),
    ∀ (n : PositiveInteger),
      let (a, b) := numberFactorization n
      (letterInSpelling pair.fst a ∨ letterInSpelling pair.fst b) ∧
      (letterInSpelling pair.snd a ∨ letterInSpelling pair.snd b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_letter_pair_exists_l16_1684
