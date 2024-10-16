import Mathlib

namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l2559_255906

/-- Given that x is inversely proportional to y, this theorem proves that
    if x₁/x₂ = 3/4, then y₁/y₂ = 4/3, where y₁ and y₂ are the corresponding y values. -/
theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (hx : x₁ ≠ 0 ∧ x₂ ≠ 0) (hy : y₁ ≠ 0 ∧ y₂ ≠ 0)
    (h_prop : ∃ k : ℝ, ∀ x y, x * y = k) (h_ratio : x₁ / x₂ = 3 / 4) :
    y₁ / y₂ = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l2559_255906


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2559_255949

theorem arithmetic_mean_of_fractions (x b : ℝ) (hx : x ≠ 0) :
  (((2 * x + b) / x + (2 * x - b) / x) / 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2559_255949


namespace NUMINAMATH_CALUDE_school_population_theorem_l2559_255943

theorem school_population_theorem (b g t s : ℕ) :
  b = 4 * g ∧ g = 8 * t ∧ t = 2 * s →
  b + g + t + s = (83 * g) / 16 := by
  sorry

end NUMINAMATH_CALUDE_school_population_theorem_l2559_255943


namespace NUMINAMATH_CALUDE_temperature_increase_proof_l2559_255919

/-- Given an initial temperature and a final temperature over a time interval,
    calculates the average hourly increase in temperature. -/
def averageHourlyIncrease (initialTemp finalTemp : Int) (timeInterval : Nat) : ℚ :=
  (finalTemp - initialTemp : ℚ) / timeInterval

/-- Theorem stating that under the given conditions, 
    the average hourly increase in temperature is 5 deg/hr. -/
theorem temperature_increase_proof :
  let initialTemp := -13
  let finalTemp := 32
  let timeInterval := 9
  averageHourlyIncrease initialTemp finalTemp timeInterval = 5 := by
  sorry

end NUMINAMATH_CALUDE_temperature_increase_proof_l2559_255919


namespace NUMINAMATH_CALUDE_total_songs_bought_l2559_255989

theorem total_songs_bought (country_albums : ℕ) (pop_albums : ℕ) 
  (songs_per_country_album : ℕ) (songs_per_pop_album : ℕ) : 
  country_albums = 4 → pop_albums = 7 → 
  songs_per_country_album = 5 → songs_per_pop_album = 6 → 
  country_albums * songs_per_country_album + pop_albums * songs_per_pop_album = 62 := by
  sorry

#check total_songs_bought

end NUMINAMATH_CALUDE_total_songs_bought_l2559_255989


namespace NUMINAMATH_CALUDE_inverse_function_sum_l2559_255942

/-- Given two real numbers a and b, define f and its inverse --/
def f (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b

def f_inv (a b : ℝ) : ℝ → ℝ := λ x ↦ b * x^2 + a

/-- Theorem stating that if f and f_inv are inverse functions, then a + b = 1 --/
theorem inverse_function_sum (a b : ℝ) : 
  (∀ x, f a b (f_inv a b x) = x) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_sum_l2559_255942


namespace NUMINAMATH_CALUDE_volunteers_in_2002_l2559_255997

/-- The number of volunteers after n years, given an initial number and annual increase rate -/
def volunteers (initial : ℕ) (rate : ℚ) (years : ℕ) : ℚ :=
  initial * (1 + rate) ^ years

/-- Theorem: The number of volunteers in 2002 will be 6075, given the initial conditions -/
theorem volunteers_in_2002 :
  volunteers 1200 (1/2) 4 = 6075 := by
  sorry

#eval volunteers 1200 (1/2) 4

end NUMINAMATH_CALUDE_volunteers_in_2002_l2559_255997


namespace NUMINAMATH_CALUDE_y1_less_than_y2_l2559_255915

/-- A linear function y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def onLine (p : Point) (f : LinearFunction) : Prop :=
  p.y = f.m * p.x + f.b

theorem y1_less_than_y2 
  (f : LinearFunction)
  (p1 p2 : Point)
  (h1 : f.m = 8)
  (h2 : f.b = -1)
  (h3 : p1.x = 3)
  (h4 : p2.x = 4)
  (h5 : onLine p1 f)
  (h6 : onLine p2 f) :
  p1.y < p2.y := by
  sorry

end NUMINAMATH_CALUDE_y1_less_than_y2_l2559_255915


namespace NUMINAMATH_CALUDE_helmet_store_theorem_l2559_255990

structure HelmetStore where
  wholesale_price_A : ℕ
  wholesale_price_B : ℕ
  day1_sales_A : ℕ
  day1_sales_B : ℕ
  day1_total : ℕ
  day2_sales_A : ℕ
  day2_sales_B : ℕ
  day2_total : ℕ
  budget : ℕ
  total_helmets : ℕ
  profit_target : ℕ

def selling_prices (store : HelmetStore) : ℕ × ℕ :=
  -- Placeholder for the function to calculate selling prices
  (0, 0)

def can_achieve_profit (store : HelmetStore) (prices : ℕ × ℕ) : Prop :=
  -- Placeholder for the function to check if profit target can be achieved
  false

theorem helmet_store_theorem (store : HelmetStore) 
  (h1 : store.wholesale_price_A = 40)
  (h2 : store.wholesale_price_B = 30)
  (h3 : store.day1_sales_A = 10)
  (h4 : store.day1_sales_B = 15)
  (h5 : store.day1_total = 1150)
  (h6 : store.day2_sales_A = 6)
  (h7 : store.day2_sales_B = 12)
  (h8 : store.day2_total = 810)
  (h9 : store.budget = 3400)
  (h10 : store.total_helmets = 100)
  (h11 : store.profit_target = 1300) :
  let prices := selling_prices store
  prices = (55, 40) ∧ ¬(can_achieve_profit store prices) := by
  sorry

end NUMINAMATH_CALUDE_helmet_store_theorem_l2559_255990


namespace NUMINAMATH_CALUDE_counterexample_exists_l2559_255961

theorem counterexample_exists : ∃ n : ℕ, 
  (¬ Nat.Prime n) ∧ (Nat.Prime (n - 3) ∨ Nat.Prime (n - 2)) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2559_255961


namespace NUMINAMATH_CALUDE_max_triangle_sum_l2559_255925

/-- Represents the arrangement of numbers on the vertices of the triangles -/
def TriangleArrangement := Fin 6 → Fin 6

/-- The sum of three numbers on a side of a triangle -/
def sideSum (arr : TriangleArrangement) (i j k : Fin 6) : ℕ :=
  (arr i).val + 12 + (arr j).val + 12 + (arr k).val + 12

/-- Predicate to check if an arrangement is valid -/
def isValidArrangement (arr : TriangleArrangement) : Prop :=
  (∀ i j, i ≠ j → arr i ≠ arr j) ∧
  (∀ i, arr i < 6)

/-- Predicate to check if all sides have the same sum -/
def allSidesEqual (arr : TriangleArrangement) (S : ℕ) : Prop :=
  sideSum arr 0 1 2 = S ∧
  sideSum arr 2 3 4 = S ∧
  sideSum arr 4 5 0 = S

theorem max_triangle_sum :
  ∃ (S : ℕ) (arr : TriangleArrangement),
    isValidArrangement arr ∧
    allSidesEqual arr S ∧
    (∀ (S' : ℕ) (arr' : TriangleArrangement),
      isValidArrangement arr' → allSidesEqual arr' S' → S' ≤ S) ∧
    S = 45 := by
  sorry

end NUMINAMATH_CALUDE_max_triangle_sum_l2559_255925


namespace NUMINAMATH_CALUDE_smallest_two_digit_reverse_diff_perfect_square_l2559_255909

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem smallest_two_digit_reverse_diff_perfect_square :
  ∃ N : ℕ, is_two_digit N ∧
    is_perfect_square (N - reverse_digits N) ∧
    (N - reverse_digits N > 0) ∧
    (∀ M : ℕ, is_two_digit M →
      is_perfect_square (M - reverse_digits M) →
      (M - reverse_digits M > 0) →
      N ≤ M) ∧
    N = 90 := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_reverse_diff_perfect_square_l2559_255909


namespace NUMINAMATH_CALUDE_trajectory_is_two_circles_l2559_255904

-- Define the set of complex numbers satisfying the equation
def S : Set ℂ := {z : ℂ | Complex.abs z ^ 2 - 3 * Complex.abs z + 2 = 0}

-- Define the trajectory of z
def trajectory (z : ℂ) : Set ℂ := {w : ℂ | Complex.abs w = Complex.abs z}

-- Theorem statement
theorem trajectory_is_two_circles :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ r₁ > 0 ∧ r₂ > 0 ∧
  (∀ z ∈ S, (trajectory z = {w : ℂ | Complex.abs w = r₁} ∨
             trajectory z = {w : ℂ | Complex.abs w = r₂})) :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_two_circles_l2559_255904


namespace NUMINAMATH_CALUDE_probability_h_in_mathematics_l2559_255960

def word : String := "Mathematics"

def count_letter (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

theorem probability_h_in_mathematics :
  (count_letter word 'h' : ℚ) / word.length = 1 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_h_in_mathematics_l2559_255960


namespace NUMINAMATH_CALUDE_man_speed_l2559_255937

/-- The speed of a man running opposite to a train, given the train's length, speed, and time to pass the man. -/
theorem man_speed (train_length : Real) (train_speed : Real) (time_to_pass : Real) :
  train_length = 110 ∧ 
  train_speed = 40 ∧ 
  time_to_pass = 9 →
  ∃ (man_speed : Real),
    man_speed > 0 ∧ 
    man_speed < train_speed ∧
    abs (man_speed - train_speed) * time_to_pass / 3600 = train_length / 1000 ∧
    abs (man_speed - 3.992) < 0.001 :=
sorry

end NUMINAMATH_CALUDE_man_speed_l2559_255937


namespace NUMINAMATH_CALUDE_cos_sin_identity_l2559_255931

theorem cos_sin_identity : 
  Real.cos (96 * π / 180) * Real.cos (24 * π / 180) - 
  Real.sin (96 * π / 180) * Real.sin (66 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_identity_l2559_255931


namespace NUMINAMATH_CALUDE_tom_marble_groups_l2559_255957

/-- Represents the types of marbles Tom has --/
inductive MarbleType
  | Red
  | Blue
  | Green
  | Yellow

/-- Represents Tom's marble collection --/
structure MarbleCollection where
  red : Nat
  blue : Nat
  green : Nat
  yellow : Nat

/-- Counts the number of different groups of 3 marbles that can be chosen --/
def countDifferentGroups (collection : MarbleCollection) : Nat :=
  sorry

/-- Theorem stating that Tom can choose 8 different groups of 3 marbles --/
theorem tom_marble_groups (tom_marbles : MarbleCollection) 
  (h_red : tom_marbles.red = 1)
  (h_blue : tom_marbles.blue = 1)
  (h_green : tom_marbles.green = 2)
  (h_yellow : tom_marbles.yellow = 3) :
  countDifferentGroups tom_marbles = 8 := by
  sorry

end NUMINAMATH_CALUDE_tom_marble_groups_l2559_255957


namespace NUMINAMATH_CALUDE_max_value_of_y_l2559_255938

-- Define the function y
def y (x a : ℝ) : ℝ := |x - a| + |x + 19| + |x - a - 96|

-- State the theorem
theorem max_value_of_y (a : ℝ) (h1 : 19 < a) (h2 : a < 96) :
  ∃ (max_y : ℝ), max_y = 211 ∧ ∀ x, a ≤ x → x ≤ 96 → y x a ≤ max_y :=
sorry

end NUMINAMATH_CALUDE_max_value_of_y_l2559_255938


namespace NUMINAMATH_CALUDE_right_triangle_min_perimeter_l2559_255916

theorem right_triangle_min_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 1 →
  a^2 + b^2 = c^2 →
  a + b + c ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_min_perimeter_l2559_255916


namespace NUMINAMATH_CALUDE_counterexample_exists_l2559_255939

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2559_255939


namespace NUMINAMATH_CALUDE_max_sides_equal_longest_diagonal_l2559_255920

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields here
  -- This is a placeholder structure

/-- The longest diagonal of a convex polygon -/
def longest_diagonal (p : ConvexPolygon) : ℝ :=
  sorry

/-- The number of sides equal to the longest diagonal in a convex polygon -/
def num_sides_equal_longest_diagonal (p : ConvexPolygon) : ℕ :=
  sorry

/-- Theorem: The maximum number of sides that can be equal to the longest diagonal in a convex polygon is 2 -/
theorem max_sides_equal_longest_diagonal (p : ConvexPolygon) :
  num_sides_equal_longest_diagonal p ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_sides_equal_longest_diagonal_l2559_255920


namespace NUMINAMATH_CALUDE_min_value_expression_l2559_255976

theorem min_value_expression (a b : ℝ) (hb : b ≠ 0) :
  a^2 + b^2 + a/b + 1/b^2 ≥ Real.sqrt 3 ∧
  ∃ (a₀ b₀ : ℝ) (hb₀ : b₀ ≠ 0), a₀^2 + b₀^2 + a₀/b₀ + 1/b₀^2 = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2559_255976


namespace NUMINAMATH_CALUDE_xyz_congruence_l2559_255934

theorem xyz_congruence (x y z : ℕ) : 
  x < 10 → y < 10 → z < 10 → x > 0 → y > 0 → z > 0 →
  (x * y * z) % 9 = 1 →
  (7 * z) % 9 = 4 →
  (8 * y) % 9 = (5 + y) % 9 →
  (x + y + z) % 9 = 2 := by sorry

end NUMINAMATH_CALUDE_xyz_congruence_l2559_255934


namespace NUMINAMATH_CALUDE_problem_solution_l2559_255994

def is_arithmetic_sequence (s : Fin 5 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Fin 4, s (i + 1) - s i = d

def is_geometric_sequence (s : Fin 5 → ℝ) : Prop :=
  ∃ r : ℝ, ∀ i : Fin 4, s (i + 1) / s i = r

theorem problem_solution (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) :
  is_arithmetic_sequence (λ i => match i with
    | 0 => 1
    | 1 => a₁
    | 2 => a₂
    | 3 => a₃
    | 4 => 9) →
  is_geometric_sequence (λ i => match i with
    | 0 => -9
    | 1 => b₁
    | 2 => b₂
    | 3 => b₃
    | 4 => -1) →
  b₂ / (a₁ + a₃) = -3/10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2559_255994


namespace NUMINAMATH_CALUDE_total_tv_time_l2559_255910

theorem total_tv_time : 
  let reality_shows := [28, 35, 42, 39, 29]
  let cartoons := [10, 10]
  let ad_breaks := [8, 6, 12]
  (reality_shows.sum + cartoons.sum + ad_breaks.sum) = 219 := by
  sorry

end NUMINAMATH_CALUDE_total_tv_time_l2559_255910


namespace NUMINAMATH_CALUDE_right_angle_points_exist_iff_l2559_255922

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  a : ℝ  -- length of one base
  c : ℝ  -- length of the other base
  h : ℝ  -- altitude
  a_pos : 0 < a  -- a is positive
  c_pos : 0 < c  -- c is positive
  h_pos : 0 < h  -- h is positive
  a_ge_c : a ≥ c  -- assume a is the longer base

/-- A point on the axis of symmetry of the trapezoid -/
def AxisPoint (t : IsoscelesTrapezoid) := ℝ

/-- Predicate for a point P where both legs subtend right angles -/
def IsRightAnglePoint (t : IsoscelesTrapezoid) (p : AxisPoint t) : Prop :=
  sorry  -- Definition of right angle condition

/-- Theorem: Existence of right angle points iff h^2 ≤ ac -/
theorem right_angle_points_exist_iff (t : IsoscelesTrapezoid) :
  (∃ p : AxisPoint t, IsRightAnglePoint t p) ↔ t.h^2 ≤ t.a * t.c :=
sorry

end NUMINAMATH_CALUDE_right_angle_points_exist_iff_l2559_255922


namespace NUMINAMATH_CALUDE_line_equation_proof_l2559_255996

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_equation_proof (given_line : Line) (point : Point) (result_line : Line) : 
  given_line.a = 1 ∧ given_line.b = -2 ∧ given_line.c = -2 ∧
  point.x = 1 ∧ point.y = 0 ∧
  result_line.a = 1 ∧ result_line.b = -2 ∧ result_line.c = -1 →
  result_line.contains point ∧ result_line.parallel given_line :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2559_255996


namespace NUMINAMATH_CALUDE_notebook_discount_rate_l2559_255991

/-- The maximum discount rate that can be applied to a notebook while maintaining a minimum profit margin. -/
theorem notebook_discount_rate (cost : ℝ) (original_price : ℝ) (min_profit_margin : ℝ) :
  cost = 6 →
  original_price = 9 →
  min_profit_margin = 0.05 →
  ∃ (max_discount : ℝ), 
    max_discount = 0.7 ∧ 
    ∀ (discount : ℝ), 
      discount ≤ max_discount →
      (original_price * (1 - discount) - cost) / cost ≥ min_profit_margin :=
by sorry

end NUMINAMATH_CALUDE_notebook_discount_rate_l2559_255991


namespace NUMINAMATH_CALUDE_vectors_opposite_directions_l2559_255966

def a (x : ℝ) : ℝ × ℝ := (1, -x)
def b (x : ℝ) : ℝ × ℝ := (x, -16)

theorem vectors_opposite_directions :
  ∃ (k : ℝ), k ≠ 0 ∧ a (-5) = k • b (-5) :=
by sorry

end NUMINAMATH_CALUDE_vectors_opposite_directions_l2559_255966


namespace NUMINAMATH_CALUDE_number_exceeding_45_percent_l2559_255914

theorem number_exceeding_45_percent : ∃ x : ℝ, x = 0.45 * x + 1000 ∧ x = 1000 / 0.55 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_45_percent_l2559_255914


namespace NUMINAMATH_CALUDE_smallest_group_size_l2559_255940

theorem smallest_group_size : ∃ n : ℕ, n > 0 ∧ 
  n % 3 = 2 ∧ 
  n % 6 = 5 ∧ 
  n % 8 = 7 ∧ 
  ∀ m : ℕ, m > 0 → 
    (m % 3 = 2 ∧ m % 6 = 5 ∧ m % 8 = 7) → 
    n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_group_size_l2559_255940


namespace NUMINAMATH_CALUDE_books_per_shelf_l2559_255926

theorem books_per_shelf (total_shelves : ℕ) (total_books : ℕ) (h1 : total_shelves = 8) (h2 : total_books = 32) :
  total_books / total_shelves = 4 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l2559_255926


namespace NUMINAMATH_CALUDE_sleeping_passenger_journey_l2559_255971

theorem sleeping_passenger_journey (total_journey : ℝ) (sleeping_distance : ℝ) :
  (sleeping_distance = total_journey / 3) ∧
  (total_journey / 2 = sleeping_distance + sleeping_distance / 2) →
  sleeping_distance / total_journey = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_sleeping_passenger_journey_l2559_255971


namespace NUMINAMATH_CALUDE_selenas_remaining_money_l2559_255921

/-- Calculates the remaining money after Selena's meal --/
def remaining_money (tip : ℚ) (steak_price : ℚ) (steak_count : ℕ) (steak_tax : ℚ)
                    (burger_price : ℚ) (burger_count : ℕ) (burger_tax : ℚ)
                    (icecream_price : ℚ) (icecream_count : ℕ) (icecream_tax : ℚ) : ℚ :=
  let steak_total := steak_price * steak_count * (1 + steak_tax)
  let burger_total := burger_price * burger_count * (1 + burger_tax)
  let icecream_total := icecream_price * icecream_count * (1 + icecream_tax)
  tip - (steak_total + burger_total + icecream_total)

/-- Theorem stating that Selena will be left with $33.74 after her meal --/
theorem selenas_remaining_money :
  remaining_money 99 24 2 (7/100) 3.5 2 (6/100) 2 3 (8/100) = 33.74 := by
  sorry

end NUMINAMATH_CALUDE_selenas_remaining_money_l2559_255921


namespace NUMINAMATH_CALUDE_square_sum_value_l2559_255972

theorem square_sum_value (x y : ℝ) :
  (x^2 + y^2 + 1) * (x^2 + y^2 + 2) = 6 → x^2 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l2559_255972


namespace NUMINAMATH_CALUDE_infinite_triples_exist_l2559_255975

theorem infinite_triples_exist : 
  ∀ y : ℝ, ∃ x z : ℝ, 
    (x^2 + y = y^2 + z) ∧ 
    (y^2 + z = z^2 + x) ∧ 
    (z^2 + x = x^2 + y) ∧ 
    x ≠ y ∧ y ≠ z ∧ z ≠ x :=
by sorry

end NUMINAMATH_CALUDE_infinite_triples_exist_l2559_255975


namespace NUMINAMATH_CALUDE_tile_border_ratio_l2559_255970

/-- Proves that for a square tiled surface with n^2 tiles, each tile of side length s,
    surrounded by a border of width d, if n = 30 and the tiles cover 81% of the total area,
    then d/s = 1/18. -/
theorem tile_border_ratio (n s d : ℝ) (h1 : n = 30) 
    (h2 : (n^2 * s^2) / ((n*s + 2*n*d)^2) = 0.81) : d/s = 1/18 := by
  sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l2559_255970


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2559_255995

theorem sufficient_not_necessary (x y : ℝ) :
  (x ≤ 2 ∧ y ≤ 3 → x + y ≤ 5) ∧
  ∃ x y : ℝ, x + y ≤ 5 ∧ (x > 2 ∨ y > 3) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2559_255995


namespace NUMINAMATH_CALUDE_triangle_relations_l2559_255984

-- Define the triangle ABC and point D
structure Triangle :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  let ⟨A, B, C, D⟩ := t
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 -- B is right angle

def BC_equals_2AB (t : Triangle) : Prop :=
  let ⟨A, B, C, D⟩ := t
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 4 * ((B.1 - A.1)^2 + (B.2 - A.2)^2)

def D_on_angle_bisector (t : Triangle) : Prop :=
  let ⟨A, B, C, D⟩ := t
  ∃ k : ℝ, 0 < k ∧ k < 1 ∧
    D.1 = A.1 + k * (C.1 - A.1) ∧
    D.2 = A.2 + k * (C.2 - A.2)

-- Theorem statement
theorem triangle_relations (t : Triangle) 
  (h1 : is_right_triangle t) 
  (h2 : BC_equals_2AB t) 
  (h3 : D_on_angle_bisector t) :
  let ⟨A, B, C, D⟩ := t
  (D.1 - B.1)^2 + (D.2 - B.2)^2 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) * (Real.sin (18 * π / 180))^2 ∧
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) * (Real.sin (36 * π / 180))^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_relations_l2559_255984


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2559_255953

/-- An isosceles triangle with two side lengths of 6 and 8 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 6 ∧ b = 8 ∧ (c = a ∨ c = b) →  -- Triangle is isosceles with sides 6 and 8
  a + b + c = 22 :=                  -- Perimeter is 22
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2559_255953


namespace NUMINAMATH_CALUDE_platform_length_l2559_255963

/-- The length of a platform given train specifications -/
theorem platform_length
  (train_length : ℝ)
  (time_platform : ℝ)
  (time_pole : ℝ)
  (h1 : train_length = 500)
  (h2 : time_platform = 45)
  (h3 : time_pole = 25) :
  let train_speed := train_length / time_pole
  let platform_length := train_speed * time_platform - train_length
  platform_length = 400 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l2559_255963


namespace NUMINAMATH_CALUDE_install_remaining_windows_time_l2559_255958

/-- Calculates the time needed to install remaining windows -/
def timeToInstallRemaining (totalWindows installedWindows timePerWindow : ℕ) : ℕ :=
  (totalWindows - installedWindows) * timePerWindow

/-- Proves that the time to install the remaining windows is 18 hours -/
theorem install_remaining_windows_time :
  timeToInstallRemaining 9 6 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_install_remaining_windows_time_l2559_255958


namespace NUMINAMATH_CALUDE_average_after_removal_l2559_255929

def originalList : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def removedNumber : ℕ := 1

def remainingList : List ℕ := originalList.filter (· ≠ removedNumber)

theorem average_after_removal :
  (remainingList.sum : ℚ) / remainingList.length = 15/2 := by sorry

end NUMINAMATH_CALUDE_average_after_removal_l2559_255929


namespace NUMINAMATH_CALUDE_multiply_powers_of_same_base_l2559_255978

theorem multiply_powers_of_same_base (a b : ℝ) : 2 * a * b * b^2 = 2 * a * b^3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_of_same_base_l2559_255978


namespace NUMINAMATH_CALUDE_quadrilateral_theorem_l2559_255912

structure Quadrilateral :=
  (C D X W P : ℝ × ℝ)
  (CD_parallel_WX : (D.1 - C.1) * (X.2 - W.2) = (D.2 - C.2) * (X.1 - W.1))
  (P_on_CW : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (C.1 + t * (W.1 - C.1), C.2 + t * (W.2 - C.2)))
  (CW_length : Real.sqrt ((W.1 - C.1)^2 + (W.2 - C.2)^2) = 56)
  (DP_length : Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2) = 16)
  (PX_length : Real.sqrt ((X.1 - P.1)^2 + (X.2 - P.2)^2) = 32)

theorem quadrilateral_theorem (q : Quadrilateral) :
  Real.sqrt ((q.W.1 - q.P.1)^2 + (q.W.2 - q.P.2)^2) = 112/3 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_theorem_l2559_255912


namespace NUMINAMATH_CALUDE_class_average_proof_l2559_255954

theorem class_average_proof (group1_percent : Real) (group1_avg : Real)
                            (group2_percent : Real) (group2_avg : Real)
                            (group3_percent : Real) (group3_avg : Real)
                            (group4_percent : Real) (group4_avg : Real)
                            (group5_percent : Real) (group5_avg : Real)
                            (h1 : group1_percent = 0.25)
                            (h2 : group1_avg = 80)
                            (h3 : group2_percent = 0.35)
                            (h4 : group2_avg = 65)
                            (h5 : group3_percent = 0.20)
                            (h6 : group3_avg = 90)
                            (h7 : group4_percent = 0.10)
                            (h8 : group4_avg = 75)
                            (h9 : group5_percent = 0.10)
                            (h10 : group5_avg = 85)
                            (h11 : group1_percent + group2_percent + group3_percent + group4_percent + group5_percent = 1) :
  group1_percent * group1_avg + group2_percent * group2_avg + group3_percent * group3_avg +
  group4_percent * group4_avg + group5_percent * group5_avg = 76.75 := by
  sorry

#check class_average_proof

end NUMINAMATH_CALUDE_class_average_proof_l2559_255954


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_theorem_l2559_255930

/-- Represents a hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- Represents a chord PQ -/
structure Chord where
  P : ℝ × ℝ
  Q : ℝ × ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Checks if a chord is perpendicular to the real axis -/
def is_perpendicular_to_real_axis (c : Chord) : Prop := sorry

/-- Checks if a chord passes through a given point -/
def passes_through (c : Chord) (p : ℝ × ℝ) : Prop := sorry

/-- Calculates the angle between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_eccentricity_theorem (h : Hyperbola) (c : Chord) :
  is_perpendicular_to_real_axis c →
  passes_through c h.F₂ →
  angle c.P h.F₁ c.Q = π / 2 →
  eccentricity h = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_theorem_l2559_255930


namespace NUMINAMATH_CALUDE_product_remainder_mod_17_l2559_255973

theorem product_remainder_mod_17 : (2011 * 2012 * 2013 * 2014 * 2015) % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_17_l2559_255973


namespace NUMINAMATH_CALUDE_range_when_p_true_range_when_one_true_one_false_l2559_255980

-- Define the propositions
def has_two_distinct_negative_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 ≠ 0

-- Theorem 1
theorem range_when_p_true (m : ℝ) :
  has_two_distinct_negative_roots m → m > 2 :=
sorry

-- Theorem 2
theorem range_when_one_true_one_false (m : ℝ) :
  (has_two_distinct_negative_roots m ↔ ¬has_no_real_roots m) →
  (m ∈ Set.Ioo 1 2 ∪ Set.Ici 3) :=
sorry

end NUMINAMATH_CALUDE_range_when_p_true_range_when_one_true_one_false_l2559_255980


namespace NUMINAMATH_CALUDE_prob_both_female_given_one_female_l2559_255928

/-- Represents the number of male students -/
def num_male : ℕ := 3

/-- Represents the number of female students -/
def num_female : ℕ := 2

/-- Represents the total number of students -/
def total_students : ℕ := num_male + num_female

/-- Represents the number of students drawn -/
def students_drawn : ℕ := 2

/-- The probability of drawing both female students given that one female student is drawn -/
theorem prob_both_female_given_one_female :
  (students_drawn = 2) →
  (num_male = 3) →
  (num_female = 2) →
  (∃ (p : ℚ), p = 1 / 7 ∧ 
    p = (1 : ℚ) / (total_students.choose students_drawn - num_male.choose students_drawn)) :=
sorry

end NUMINAMATH_CALUDE_prob_both_female_given_one_female_l2559_255928


namespace NUMINAMATH_CALUDE_mary_seashells_l2559_255956

theorem mary_seashells (jessica_seashells : ℕ) (total_seashells : ℕ) 
  (h1 : jessica_seashells = 41)
  (h2 : total_seashells = 59) :
  total_seashells - jessica_seashells = 18 :=
by sorry

end NUMINAMATH_CALUDE_mary_seashells_l2559_255956


namespace NUMINAMATH_CALUDE_library_visits_total_l2559_255969

/-- The number of times William goes to the library per week -/
def william_freq : ℕ := 2

/-- The number of times Jason goes to the library per week -/
def jason_freq : ℕ := 4 * william_freq

/-- The number of times Emma goes to the library per week -/
def emma_freq : ℕ := 3 * jason_freq

/-- The number of times Zoe goes to the library per week -/
def zoe_freq : ℕ := william_freq / 2

/-- The number of times Chloe goes to the library per week -/
def chloe_freq : ℕ := emma_freq / 3

/-- The number of weeks -/
def weeks : ℕ := 8

/-- The total number of times Jason, Emma, Zoe, and Chloe go to the library over 8 weeks -/
def total_visits : ℕ := (jason_freq + emma_freq + zoe_freq + chloe_freq) * weeks

theorem library_visits_total : total_visits = 328 := by
  sorry

end NUMINAMATH_CALUDE_library_visits_total_l2559_255969


namespace NUMINAMATH_CALUDE_average_string_length_l2559_255927

theorem average_string_length :
  let string1 : ℚ := 2
  let string2 : ℚ := 5
  let string3 : ℚ := 3
  let num_strings : ℕ := 3
  (string1 + string2 + string3) / num_strings = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_average_string_length_l2559_255927


namespace NUMINAMATH_CALUDE_play_seating_l2559_255911

/-- The number of chairs put out for a play, given the number of rows and chairs per row -/
def total_chairs (rows : ℕ) (chairs_per_row : ℕ) : ℕ := rows * chairs_per_row

/-- Theorem stating that 27 rows of 16 chairs each results in 432 chairs total -/
theorem play_seating : total_chairs 27 16 = 432 := by
  sorry

end NUMINAMATH_CALUDE_play_seating_l2559_255911


namespace NUMINAMATH_CALUDE_kiwi_fraction_l2559_255967

theorem kiwi_fraction (total : ℕ) (strawberries : ℕ) (h1 : total = 78) (h2 : strawberries = 52) :
  (total - strawberries : ℚ) / total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_kiwi_fraction_l2559_255967


namespace NUMINAMATH_CALUDE_ordering_of_special_values_l2559_255945

theorem ordering_of_special_values :
  let a := Real.exp (1/2)
  let b := Real.log (1/2)
  let c := Real.sin (1/2)
  a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_ordering_of_special_values_l2559_255945


namespace NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l2559_255907

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - 9*x - 1

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - 9

-- Theorem statement
theorem tangent_line_and_monotonicity 
  (a : ℝ) 
  (h1 : a < 0) 
  (h2 : ∃ x₀, ∀ x, f' a x₀ ≤ f' a x ∧ f' a x₀ = -12) :
  a = -3 ∧ 
  (∀ x₁ x₂, x₁ < x₂ → 
    ((x₂ < -1 → f a x₁ < f a x₂) ∧
     (x₁ > 3 → f a x₁ < f a x₂) ∧
     (-1 < x₁ ∧ x₂ < 3 → f a x₁ > f a x₂))) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l2559_255907


namespace NUMINAMATH_CALUDE_tangent_points_collinearity_l2559_255979

-- Define the structure for a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the structure for a point
structure Point where
  coords : ℝ × ℝ

-- Define the property of three circles being pairwise non-intersecting
def pairwise_non_intersecting (c1 c2 c3 : Circle) : Prop :=
  sorry

-- Define the property of a point being on the internal tangent of two circles
def on_internal_tangent (p : Point) (c1 c2 : Circle) : Prop :=
  sorry

-- Define the property of a point being on the external tangent of two circles
def on_external_tangent (p : Point) (c1 c2 : Circle) : Prop :=
  sorry

-- Define the property of three points being collinear
def collinear (p1 p2 p3 : Point) : Prop :=
  sorry

-- Main theorem
theorem tangent_points_collinearity 
  (c1 c2 c3 : Circle)
  (A1 A2 A3 B1 B2 B3 : Point)
  (h_non_intersecting : pairwise_non_intersecting c1 c2 c3)
  (h_A1 : on_internal_tangent A1 c2 c3)
  (h_A2 : on_internal_tangent A2 c1 c3)
  (h_A3 : on_internal_tangent A3 c1 c2)
  (h_B1 : on_external_tangent B1 c2 c3)
  (h_B2 : on_external_tangent B2 c1 c3)
  (h_B3 : on_external_tangent B3 c1 c2) :
  (collinear A1 A2 B3) ∧ 
  (collinear A1 B2 A3) ∧ 
  (collinear B1 A2 A3) ∧ 
  (collinear B1 B2 B3) :=
sorry

end NUMINAMATH_CALUDE_tangent_points_collinearity_l2559_255979


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l2559_255933

open Complex

theorem z_in_third_quadrant : 
  let z : ℂ := (2 + I) / (I^5 - 1)
  (z.re < 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l2559_255933


namespace NUMINAMATH_CALUDE_kid_tickets_sold_l2559_255902

/-- Proves the number of kid tickets sold given ticket prices, total tickets, and profit -/
theorem kid_tickets_sold 
  (adult_price : ℕ) 
  (kid_price : ℕ) 
  (total_tickets : ℕ) 
  (total_profit : ℕ) 
  (h1 : adult_price = 6)
  (h2 : kid_price = 2)
  (h3 : total_tickets = 175)
  (h4 : total_profit = 750) :
  ∃ (adult_tickets kid_tickets : ℕ), 
    adult_tickets + kid_tickets = total_tickets ∧
    adult_price * adult_tickets + kid_price * kid_tickets = total_profit ∧
    kid_tickets = 75 :=
by sorry

end NUMINAMATH_CALUDE_kid_tickets_sold_l2559_255902


namespace NUMINAMATH_CALUDE_pipe_filling_time_l2559_255917

/-- Given two pipes A and B that can fill a tank, this theorem proves the time it takes for pipe B to fill the tank alone. -/
theorem pipe_filling_time (fill_rate_A fill_rate_B : ℝ) : 
  fill_rate_A = (1 : ℝ) / 10 →  -- Pipe A fills the tank in 10 hours
  fill_rate_A + fill_rate_B = (1 : ℝ) / 6 →  -- Both pipes together fill the tank in 6 hours
  fill_rate_B = (1 : ℝ) / 15  -- Pipe B fills the tank in 15 hours
:= by sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l2559_255917


namespace NUMINAMATH_CALUDE_symmetric_arrangement_exists_l2559_255903

/-- Represents a grid figure -/
structure GridFigure where
  -- Add necessary fields to represent a grid figure
  asymmetric : Bool

/-- Represents an arrangement of grid figures -/
structure Arrangement where
  figures : List GridFigure
  symmetric : Bool

/-- Given three identical asymmetric grid figures, 
    there exists a symmetric arrangement -/
theorem symmetric_arrangement_exists : 
  ∀ (f : GridFigure), 
    f.asymmetric → 
    ∃ (a : Arrangement), 
      a.figures.length = 3 ∧ 
      (∀ fig ∈ a.figures, fig = f) ∧ 
      a.symmetric :=
by
  sorry


end NUMINAMATH_CALUDE_symmetric_arrangement_exists_l2559_255903


namespace NUMINAMATH_CALUDE_min_value_theorem_l2559_255992

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1/x + 9/y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 2 ∧ 1/x₀ + 9/y₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2559_255992


namespace NUMINAMATH_CALUDE_mrs_hilt_book_profit_l2559_255982

/-- Calculates the profit from buying and selling books -/
def book_profit (num_books : ℕ) (buy_price sell_price : ℚ) : ℚ :=
  num_books * (sell_price - buy_price)

/-- Theorem: Mrs. Hilt's profit from buying and selling books -/
theorem mrs_hilt_book_profit :
  book_profit 15 11 25 = 210 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_book_profit_l2559_255982


namespace NUMINAMATH_CALUDE_diamond_two_three_l2559_255948

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := a * b^2 - b + a^2 + 1

-- Theorem statement
theorem diamond_two_three : diamond 2 3 = 20 := by sorry

end NUMINAMATH_CALUDE_diamond_two_three_l2559_255948


namespace NUMINAMATH_CALUDE_complex_number_parts_opposite_l2559_255950

theorem complex_number_parts_opposite (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (3 + Complex.I)
  (z.re = -z.im) → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_parts_opposite_l2559_255950


namespace NUMINAMATH_CALUDE_water_content_in_fresh_grapes_fresh_grapes_water_percentage_l2559_255951

theorem water_content_in_fresh_grapes 
  (dried_water_content : Real) 
  (fresh_weight : Real) 
  (dried_weight : Real) : Real :=
  let solid_content := dried_weight * (1 - dried_water_content)
  let water_content := fresh_weight - solid_content
  let water_percentage := (water_content / fresh_weight) * 100
  90

theorem fresh_grapes_water_percentage :
  let dried_water_content := 0.20
  let fresh_weight := 10
  let dried_weight := 1.25
  water_content_in_fresh_grapes dried_water_content fresh_weight dried_weight = 90 := by
  sorry

end NUMINAMATH_CALUDE_water_content_in_fresh_grapes_fresh_grapes_water_percentage_l2559_255951


namespace NUMINAMATH_CALUDE_intersection_minimum_distance_l2559_255923

/-- Given a line y = b intersecting f(x) = 2x + 3 and g(x) = ax + ln x at points A and B respectively,
    if the minimum value of |AB| is 2, then a + b = 2 -/
theorem intersection_minimum_distance (a b : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    2 * x₁ + 3 = b ∧ 
    a * x₂ + Real.log x₂ = b ∧
    (∀ (y₁ y₂ : ℝ), 2 * y₁ + 3 = b → a * y₂ + Real.log y₂ = b → |y₂ - y₁| ≥ 2) ∧
    |x₂ - x₁| = 2) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_minimum_distance_l2559_255923


namespace NUMINAMATH_CALUDE_count_numbers_with_seven_800_l2559_255964

def contains_seven (n : Nat) : Bool :=
  let digits := n.digits 10
  7 ∈ digits

def count_numbers_with_seven (upper_bound : Nat) : Nat :=
  (List.range upper_bound).filter contains_seven |>.length

theorem count_numbers_with_seven_800 :
  count_numbers_with_seven 800 = 152 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_seven_800_l2559_255964


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2559_255941

/-- Given that x varies inversely with y³ and x = 8 when y = 1, prove that x = 1 when y = 2 -/
theorem inverse_variation_problem (x y : ℝ) (h : ∀ y : ℝ, y ≠ 0 → ∃ k : ℝ, x * y^3 = k) :
  (∃ k : ℝ, 8 * 1^3 = k) → (∃ x : ℝ, x * 2^3 = 8) → (∃ x : ℝ, x = 1) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2559_255941


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2559_255959

theorem quadratic_inequality_solution_set :
  {x : ℝ | 3 * x^2 - 5 * x - 2 < 0} = {x : ℝ | -1/3 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2559_255959


namespace NUMINAMATH_CALUDE_division_problem_l2559_255977

theorem division_problem : (120 : ℚ) / ((6 : ℚ) / 2) = 40 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2559_255977


namespace NUMINAMATH_CALUDE_base_b_square_l2559_255962

theorem base_b_square (b : ℕ) : b > 0 → (3 * b + 3)^2 = b^3 + 2 * b^2 + 3 * b ↔ b = 9 := by
  sorry

end NUMINAMATH_CALUDE_base_b_square_l2559_255962


namespace NUMINAMATH_CALUDE_electric_power_is_4_l2559_255905

-- Define the constants and variables
variable (k_star : ℝ) (e_tau : ℝ) (a_star : ℝ) (N_H : ℝ) (N_e : ℝ)

-- Define the conditions
axiom k_star_def : k_star = 1/3
axiom e_tau_a_star_def : e_tau * a_star = 0.15
axiom N_H_def : N_H = 80

-- Define the electric power equation
def electric_power (k_star e_tau a_star N_H : ℝ) : ℝ :=
  k_star * e_tau * a_star * N_H

-- State the theorem
theorem electric_power_is_4 :
  electric_power k_star e_tau a_star N_H = 4 :=
sorry

end NUMINAMATH_CALUDE_electric_power_is_4_l2559_255905


namespace NUMINAMATH_CALUDE_function_value_alternation_l2559_255987

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β) where a, b, α, β are non-zero real numbers,
    if f(2013) = -1, then f(2014) = 1 -/
theorem function_value_alternation (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β)
  f 2013 = -1 → f 2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_alternation_l2559_255987


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l2559_255988

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The main theorem stating that there are 67 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute 6 3 = 67 := by sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l2559_255988


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l2559_255901

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (2 * x - y + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l2559_255901


namespace NUMINAMATH_CALUDE_trains_return_to_initial_positions_l2559_255944

/-- Represents a train on a circular track -/
structure Train where
  period : ℕ
  position : ℕ

/-- The state of the metro system -/
structure MetroSystem where
  trains : List Train

/-- Calculates the position of a train after a given number of minutes -/
def trainPosition (t : Train) (minutes : ℕ) : ℕ :=
  minutes % t.period

/-- Checks if all trains are at their initial positions -/
def allTrainsAtInitial (ms : MetroSystem) (minutes : ℕ) : Prop :=
  ∀ t ∈ ms.trains, trainPosition t minutes = 0

/-- The main theorem -/
theorem trains_return_to_initial_positions (ms : MetroSystem) : 
  ms.trains = [⟨14, 0⟩, ⟨16, 0⟩, ⟨18, 0⟩] → allTrainsAtInitial ms 2016 := by
  sorry


end NUMINAMATH_CALUDE_trains_return_to_initial_positions_l2559_255944


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2559_255974

-- Problem 1
theorem problem_1 : (-1)^4 - 2 * Real.tan (60 * π / 180) + (Real.sqrt 3 - Real.sqrt 2)^0 + Real.sqrt 12 = 2 := by
  sorry

-- Problem 2
theorem problem_2 : ∀ x : ℝ, (x - 1) / 3 ≥ x / 2 - 2 ↔ x ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2559_255974


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2559_255913

theorem absolute_value_equation_solution (x z : ℝ) :
  |5 * x - Real.log z| = 5 * x + 3 * Real.log z →
  x = 0 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2559_255913


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l2559_255918

/-- Convert a list of binary digits to a natural number -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Convert a natural number to a list of binary digits -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Multiply two binary numbers represented as lists of booleans -/
def binary_multiply (a b : List Bool) : List Bool :=
  nat_to_binary ((binary_to_nat a) * (binary_to_nat b))

theorem binary_multiplication_theorem :
  let a := [true, false, true, true, false, true, true]  -- 1101101₂
  let b := [true, true, true]  -- 111₂
  let product := binary_multiply a b
  binary_to_nat product = 1267 ∧ 
  product = [true, true, false, false, true, true, true, true, false, false, true] :=
by sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l2559_255918


namespace NUMINAMATH_CALUDE_crayon_calculation_l2559_255998

/-- Calculates the final number of crayons and their percentage of the total items -/
theorem crayon_calculation (initial_crayons : ℕ) (initial_pencils : ℕ) 
  (removed_crayons : ℕ) (added_crayons : ℕ) (increase_percentage : ℚ) :
  initial_crayons = 41 →
  initial_pencils = 26 →
  removed_crayons = 8 →
  added_crayons = 12 →
  increase_percentage = 1/10 →
  let intermediate_crayons := initial_crayons - removed_crayons + added_crayons
  let final_crayons := (intermediate_crayons : ℚ) * (1 + increase_percentage)
  let rounded_final_crayons := round final_crayons
  let total_items := rounded_final_crayons + initial_pencils
  let percentage_crayons := (rounded_final_crayons : ℚ) / (total_items : ℚ) * 100
  rounded_final_crayons = 50 ∧ abs (percentage_crayons - 65.79) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_crayon_calculation_l2559_255998


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2559_255946

theorem min_distance_to_line (x y : ℝ) : 
  3 * x + 4 * y = 24 → x ≥ 0 → 
  ∃ (min_val : ℝ), min_val = 24 / 5 ∧ 
    ∀ (x' y' : ℝ), 3 * x' + 4 * y' = 24 → x' ≥ 0 → 
      Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2559_255946


namespace NUMINAMATH_CALUDE_find_other_number_l2559_255924

-- Define the given conditions
def n : ℕ := 48
def lcm_nm : ℕ := 56
def gcf_nm : ℕ := 12

-- Define the theorem
theorem find_other_number (m : ℕ) : 
  (Nat.lcm n m = lcm_nm) → 
  (Nat.gcd n m = gcf_nm) → 
  m = 14 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l2559_255924


namespace NUMINAMATH_CALUDE_negation_of_existence_implication_l2559_255968

theorem negation_of_existence_implication :
  ¬(∃ n : ℤ, ∀ m : ℤ, n^2 = m^2 → n = m) ↔
  (∀ n : ℤ, ∃ m : ℤ, n^2 = m^2 ∧ n ≠ m) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_implication_l2559_255968


namespace NUMINAMATH_CALUDE_dividend_problem_l2559_255985

theorem dividend_problem (dividend divisor quotient : ℕ) : 
  dividend + divisor + quotient = 103 →
  quotient = 3 →
  dividend % divisor = 0 →
  dividend / divisor = quotient →
  dividend = 75 := by
sorry

end NUMINAMATH_CALUDE_dividend_problem_l2559_255985


namespace NUMINAMATH_CALUDE_purple_marbles_fraction_l2559_255986

theorem purple_marbles_fraction (total : ℚ) (h1 : total > 0) : 
  let yellow := (4/7) * total
  let green := (2/7) * total
  let initial_purple := total - yellow - green
  let new_purple := 3 * initial_purple
  let new_total := yellow + green + new_purple
  new_purple / new_total = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_purple_marbles_fraction_l2559_255986


namespace NUMINAMATH_CALUDE_deepak_present_age_l2559_255993

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's present age -/
theorem deepak_present_age 
  (ratio_rahul : ℕ) 
  (ratio_deepak : ℕ) 
  (rahul_future_age : ℕ) 
  (years_difference : ℕ) : 
  ratio_rahul = 4 → 
  ratio_deepak = 3 → 
  rahul_future_age = 22 → 
  years_difference = 6 → 
  (ratio_deepak * (rahul_future_age - years_difference)) / ratio_rahul = 12 := by
  sorry

end NUMINAMATH_CALUDE_deepak_present_age_l2559_255993


namespace NUMINAMATH_CALUDE_circle_properties_l2559_255952

/-- Given a circle C with equation x^2 + 8x - 2y = 1 - y^2, 
    prove that its center is (-4, 1), its radius is 3√2, 
    and the sum of its center coordinates and radius is -3 + 3√2 -/
theorem circle_properties : 
  ∃ (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + 8*x - 2*y = 1 - y^2) ∧
    center = (-4, 1) ∧
    radius = 3 * Real.sqrt 2 ∧
    center.1 + center.2 + radius = -3 + 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l2559_255952


namespace NUMINAMATH_CALUDE_negative_one_times_negative_three_equals_three_l2559_255935

theorem negative_one_times_negative_three_equals_three :
  (-1 : ℤ) * (-3 : ℤ) = (3 : ℤ) := by sorry

end NUMINAMATH_CALUDE_negative_one_times_negative_three_equals_three_l2559_255935


namespace NUMINAMATH_CALUDE_potato_yield_difference_l2559_255936

/-- Represents the yield difference between varietal and non-varietal potatoes -/
def yield_difference (
  non_varietal_area : ℝ
  ) (varietal_area : ℝ
  ) (yield_difference : ℝ
  ) : Prop :=
  let total_area := non_varietal_area + varietal_area
  let x := non_varietal_area
  let y := varietal_area
  ∃ (non_varietal_yield varietal_yield : ℝ),
    (non_varietal_yield * x + varietal_yield * y) / total_area = 
    non_varietal_yield + yield_difference ∧
    varietal_yield - non_varietal_yield = yield_difference

/-- Theorem stating the yield difference between varietal and non-varietal potatoes -/
theorem potato_yield_difference :
  yield_difference 14 4 90 := by
  sorry

end NUMINAMATH_CALUDE_potato_yield_difference_l2559_255936


namespace NUMINAMATH_CALUDE_spurs_basketball_count_l2559_255900

/-- The number of players on the Spurs basketball team -/
def num_players : ℕ := 22

/-- The number of basketballs each player has -/
def balls_per_player : ℕ := 11

/-- The total number of basketballs -/
def total_basketballs : ℕ := num_players * balls_per_player

theorem spurs_basketball_count : total_basketballs = 242 := by
  sorry

end NUMINAMATH_CALUDE_spurs_basketball_count_l2559_255900


namespace NUMINAMATH_CALUDE_circles_position_l2559_255908

theorem circles_position (r₁ r₂ : ℝ) (h₁ : r₁ * r₂ = 3) (h₂ : r₁ + r₂ = 5) (h₃ : (r₁ - r₂)^2 = 13/4) :
  let d := 3
  r₁ + r₂ > d ∧ |r₁ - r₂| > d :=
by sorry

end NUMINAMATH_CALUDE_circles_position_l2559_255908


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2559_255947

def a : Fin 2 → ℝ := ![1, -3]
def b : Fin 2 → ℝ := ![2, 1]

theorem parallel_vectors_k_value : 
  ∃ (k : ℝ), ∃ (c : ℝ), c ≠ 0 ∧ 
    (∀ i : Fin 2, (k * a i + b i) = c * (a i - 2 * b i)) → 
    k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2559_255947


namespace NUMINAMATH_CALUDE_divisibility_condition_l2559_255965

theorem divisibility_condition (n : ℕ) : n ≥ 1 → (n ^ 2 ∣ 2 ^ n + 1) ↔ n = 1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2559_255965


namespace NUMINAMATH_CALUDE_fraction_denominator_problem_l2559_255981

theorem fraction_denominator_problem (y x : ℝ) (h1 : y > 0) 
  (h2 : (2 * y) / 5 + (3 * y) / x = 0.7 * y) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_problem_l2559_255981


namespace NUMINAMATH_CALUDE_nested_bracket_equals_two_l2559_255955

/-- Defines the operation [a,b,c] as (a+b)/c for c ≠ 0 -/
def bracket (a b c : ℚ) : ℚ := (a + b) / c

/-- The main theorem to prove -/
theorem nested_bracket_equals_two :
  bracket (bracket 120 60 180) (bracket 4 2 6) (bracket 20 10 30) = 2 := by
  sorry


end NUMINAMATH_CALUDE_nested_bracket_equals_two_l2559_255955


namespace NUMINAMATH_CALUDE_arithmetic_progression_equality_l2559_255983

theorem arithmetic_progression_equality (n : ℕ) 
  (hn : n ≥ 2018) 
  (a b : Fin n → ℕ) 
  (h_distinct : ∀ i j : Fin n, i ≠ j → (a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j))
  (h_bound : ∀ i : Fin n, a i ≤ 5*n ∧ b i ≤ 5*n)
  (h_positive : ∀ i : Fin n, a i > 0 ∧ b i > 0)
  (h_arithmetic : ∃ d : ℚ, ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) - (a j : ℚ) / (b j : ℚ) = (i.val - j.val : ℚ) * d) :
  ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) = (a j : ℚ) / (b j : ℚ) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_equality_l2559_255983


namespace NUMINAMATH_CALUDE_no_real_solutions_l2559_255999

theorem no_real_solutions :
  ¬∃ (x : ℝ), (2*x - 6)^2 + 4 = -2*|x| := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2559_255999


namespace NUMINAMATH_CALUDE_magazine_subscription_cost_l2559_255932

theorem magazine_subscription_cost (reduced_cost : ℝ) (reduction_percentage : ℝ) (original_cost : ℝ) : 
  reduced_cost = 752 ∧ 
  reduction_percentage = 0.20 ∧ 
  reduced_cost = original_cost * (1 - reduction_percentage) →
  original_cost = 940 := by
sorry

end NUMINAMATH_CALUDE_magazine_subscription_cost_l2559_255932
