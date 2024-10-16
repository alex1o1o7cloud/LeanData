import Mathlib

namespace NUMINAMATH_CALUDE_paint_room_time_l737_73732

/-- The time required for Doug, Dave, and Diana to paint a room together -/
theorem paint_room_time (t : ℝ) 
  (hDoug : (1 : ℝ) / 5 * t = 1)  -- Doug can paint the room in 5 hours
  (hDave : (1 : ℝ) / 7 * t = 1)  -- Dave can paint the room in 7 hours
  (hDiana : (1 : ℝ) / 6 * t = 1) -- Diana can paint the room in 6 hours
  (hLunch : ℝ) (hLunchTime : hLunch = 2) -- 2-hour lunch break
  : ((1 : ℝ) / 5 + 1 / 7 + 1 / 6) * (t - hLunch) = 1 :=
by sorry

end NUMINAMATH_CALUDE_paint_room_time_l737_73732


namespace NUMINAMATH_CALUDE_angle_between_asymptotes_l737_73742

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = Real.sqrt 3 * x
def asymptote2 (x y : ℝ) : Prop := y = -Real.sqrt 3 * x

-- Theorem statement
theorem angle_between_asymptotes :
  ∃ (θ : ℝ), θ = 60 * π / 180 ∧
  (∀ (x y : ℝ), hyperbola x y → 
    (asymptote1 x y ∨ asymptote2 x y) →
    ∃ (x1 y1 x2 y2 : ℝ), 
      asymptote1 x1 y1 ∧ asymptote2 x2 y2 ∧
      Real.cos θ = (x1 * x2 + y1 * y2) / 
        (Real.sqrt (x1^2 + y1^2) * Real.sqrt (x2^2 + y2^2))) :=
by sorry

end NUMINAMATH_CALUDE_angle_between_asymptotes_l737_73742


namespace NUMINAMATH_CALUDE_cube_root_eq_four_l737_73704

theorem cube_root_eq_four (y : ℝ) :
  (y * (y^5)^(1/2))^(1/3) = 4 → y = 4^(6/7) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_eq_four_l737_73704


namespace NUMINAMATH_CALUDE_largest_common_value_l737_73779

/-- The first arithmetic progression -/
def seq1 (n : ℕ) : ℕ := 4 + 5 * n

/-- The second arithmetic progression -/
def seq2 (m : ℕ) : ℕ := 5 + 10 * m

/-- A common term of both sequences -/
def common_term (k : ℕ) : ℕ := 4 + 10 * k

theorem largest_common_value :
  (∃ n m : ℕ, seq1 n = seq2 m ∧ seq1 n < 1000) ∧
  (∀ n m : ℕ, seq1 n = seq2 m → seq1 n < 1000 → seq1 n ≤ 994) ∧
  (∃ k : ℕ, common_term k = 994 ∧ common_term k = seq1 (2 * k) ∧ common_term k = seq2 k) :=
sorry

end NUMINAMATH_CALUDE_largest_common_value_l737_73779


namespace NUMINAMATH_CALUDE_coin_collection_values_l737_73782

/-- Represents a collection of coins -/
structure CoinCollection where
  nickels : ℕ
  quarters : ℕ
  half_dollars : ℕ

/-- Defines the conditions for the coin collection -/
def valid_collection (c : CoinCollection) : Prop :=
  c.quarters = c.nickels / 2 ∧ c.half_dollars = 2 * c.quarters

/-- Calculates the total value of the coin collection in cents -/
def total_value (c : CoinCollection) : ℕ :=
  5 * c.nickels + 25 * c.quarters + 50 * c.half_dollars

/-- Theorem stating that there exist valid collections with total values of $67.50 and $135.00 -/
theorem coin_collection_values : 
  ∃ (c1 c2 : CoinCollection), 
    valid_collection c1 ∧ valid_collection c2 ∧ 
    total_value c1 = 6750 ∧ total_value c2 = 13500 :=
by sorry

end NUMINAMATH_CALUDE_coin_collection_values_l737_73782


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l737_73774

theorem rectangle_dimensions (w l : ℕ) : 
  l = w + 5 →
  2 * l + 2 * w = 34 →
  w = 6 ∧ l = 11 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l737_73774


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l737_73733

/-- The speed of a boat in still water, given its speeds with and against a stream -/
theorem boat_speed_in_still_water (along_stream speed_along_stream : ℝ) 
  (against_stream speed_against_stream : ℝ) :
  along_stream = 9 → against_stream = 5 →
  speed_along_stream = along_stream / 1 →
  speed_against_stream = against_stream / 1 →
  ∃ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = speed_along_stream ∧
    boat_speed - stream_speed = speed_against_stream ∧
    boat_speed = 7 := by
  sorry


end NUMINAMATH_CALUDE_boat_speed_in_still_water_l737_73733


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l737_73719

theorem smallest_x_absolute_value_equation : 
  ∃ x : ℝ, x = -8.6 ∧ ∀ y : ℝ, |5 * y + 9| = 34 → y ≥ x := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l737_73719


namespace NUMINAMATH_CALUDE_rational_root_implies_even_coefficient_l737_73723

theorem rational_root_implies_even_coefficient 
  (a b c : ℤ) 
  (h : ∃ (p q : ℤ), q ≠ 0 ∧ a * (p / q)^2 + b * (p / q) + c = 0) : 
  a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_root_implies_even_coefficient_l737_73723


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l737_73731

theorem min_sum_of_squares (m n : ℕ) (h1 : n = m + 1) (h2 : n^2 - m^2 > 20) :
  ∃ (k : ℕ), k = n^2 + m^2 ∧ k ≥ 221 ∧ ∀ (j : ℕ), (∃ (p q : ℕ), q = p + 1 ∧ q^2 - p^2 > 20 ∧ j = q^2 + p^2) → j ≥ k :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l737_73731


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l737_73756

/-- Given a total number of marbles, with blue marbles being three times
    the number of red marbles, and a specific number of red marbles,
    prove the number of yellow marbles. -/
theorem yellow_marbles_count
  (total : ℕ)
  (red : ℕ)
  (h1 : total = 85)
  (h2 : red = 14) :
  total - (red + 3 * red) = 29 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l737_73756


namespace NUMINAMATH_CALUDE_unique_three_digit_number_with_three_divisors_l737_73759

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def starts_with_three (n : ℕ) : Prop := ∃ k, n = 300 + k ∧ 0 ≤ k ∧ k < 100

def has_exactly_three_divisors (n : ℕ) : Prop := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 3

theorem unique_three_digit_number_with_three_divisors :
  ∃! n : ℕ, is_three_digit n ∧ starts_with_three n ∧ has_exactly_three_divisors n ∧ n = 361 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_with_three_divisors_l737_73759


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l737_73781

/-- Defines whether an equation represents an ellipse -/
def IsEllipse (m : ℝ) : Prop :=
  (m - 1 > 0) ∧ (3 - m > 0) ∧ (m - 1 ≠ 3 - m)

/-- The condition on m -/
def Condition (m : ℝ) : Prop :=
  1 < m ∧ m < 3

/-- Theorem stating that the condition is necessary but not sufficient -/
theorem condition_necessary_not_sufficient :
  (∀ m : ℝ, IsEllipse m → Condition m) ∧
  (∃ m : ℝ, Condition m ∧ ¬IsEllipse m) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l737_73781


namespace NUMINAMATH_CALUDE_diagonal_length_is_13_l737_73734

/-- Represents an isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  AB : ℝ  -- Length of the longer parallel side
  CD : ℝ  -- Length of the shorter parallel side
  AD : ℝ  -- Length of a leg (equal to BC in an isosceles trapezoid)

/-- The diagonal length of the isosceles trapezoid -/
def diagonal_length (t : IsoscelesTrapezoid) : ℝ := 
  sorry

/-- Theorem stating that for the given trapezoid dimensions, the diagonal length is 13 -/
theorem diagonal_length_is_13 :
  let t : IsoscelesTrapezoid := { AB := 24, CD := 10, AD := 13 }
  diagonal_length t = 13 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_is_13_l737_73734


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l737_73767

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - 6*a < 0) →
  (∃ x₁ x₂ : ℝ, x₁^2 - a*x₁ - 6*a = 0 ∧ x₂^2 - a*x₂ - 6*a = 0 ∧ |x₁ - x₂| ≤ 5) →
  (-25 ≤ a ∧ a < -24) ∨ (0 < a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l737_73767


namespace NUMINAMATH_CALUDE_new_sequence_common_difference_l737_73718

theorem new_sequence_common_difference 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h : ∀ n : ℕ, a (n + 1) = a n + d) :
  let b : ℕ → ℝ := λ n => a n + a (n + 3)
  ∀ n : ℕ, b (n + 1) = b n + 2 * d :=
by sorry

end NUMINAMATH_CALUDE_new_sequence_common_difference_l737_73718


namespace NUMINAMATH_CALUDE_warehouse_capacity_prove_warehouse_capacity_l737_73757

/-- The total capacity of a grain storage warehouse --/
theorem warehouse_capacity : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun total_bins twenty_ton_bins twenty_ton_capacity fifteen_ton_capacity =>
    total_bins = 30 ∧
    twenty_ton_bins = 12 ∧
    twenty_ton_capacity = 20 ∧
    fifteen_ton_capacity = 15 →
    (twenty_ton_bins * twenty_ton_capacity) +
    ((total_bins - twenty_ton_bins) * fifteen_ton_capacity) = 510

/-- Proof of the warehouse capacity theorem --/
theorem prove_warehouse_capacity :
  warehouse_capacity 30 12 20 15 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_capacity_prove_warehouse_capacity_l737_73757


namespace NUMINAMATH_CALUDE_garden_area_bounds_l737_73764

/-- Represents a rectangular garden with given constraints -/
structure Garden where
  wall : ℝ
  fence : ℝ
  minParallelSide : ℝ

/-- The area of the garden as a function of the length perpendicular to the wall -/
def Garden.area (g : Garden) (x : ℝ) : ℝ :=
  x * (g.fence - 2 * x)

/-- Theorem stating the maximum and minimum areas of the garden -/
theorem garden_area_bounds (g : Garden) 
  (h_wall : g.wall = 12)
  (h_fence : g.fence = 40)
  (h_minSide : g.minParallelSide = 6) :
  (∃ x : ℝ, g.area x ≤ 168 ∧ 
   ∀ y : ℝ, g.minParallelSide ≤ g.fence - 2 * y → g.area y ≤ g.area x) ∧
  (∃ x : ℝ, g.area x ≥ 102 ∧ 
   ∀ y : ℝ, g.minParallelSide ≤ g.fence - 2 * y → g.area y ≥ g.area x) :=
sorry

end NUMINAMATH_CALUDE_garden_area_bounds_l737_73764


namespace NUMINAMATH_CALUDE_odd_digits_base4_157_l737_73741

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Theorem: The number of odd digits in the base-4 representation of 157₁₀ is 2 -/
theorem odd_digits_base4_157 : countOddDigits (toBase4 157) = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_base4_157_l737_73741


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_l737_73748

theorem sqrt_50_between_consecutive_integers : 
  ∃ n : ℕ, n > 0 ∧ n < Real.sqrt 50 ∧ Real.sqrt 50 < n + 1 ∧ n * (n + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_l737_73748


namespace NUMINAMATH_CALUDE_four_integer_pairs_satisfying_equation_l737_73758

theorem four_integer_pairs_satisfying_equation :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1 + p.2 = p.1 * p.2 - 1) ∧
    s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_integer_pairs_satisfying_equation_l737_73758


namespace NUMINAMATH_CALUDE_money_puzzle_l737_73726

theorem money_puzzle (x : ℝ) : x = 800 ↔ 4 * x - 2000 = 2000 - x := by sorry

end NUMINAMATH_CALUDE_money_puzzle_l737_73726


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_smallest_primes_l737_73780

def smallest_primes : List Nat := [2, 3, 5, 7, 11]

def is_divisible_by_all (n : Nat) (lst : List Nat) : Prop :=
  ∀ m ∈ lst, n % m = 0

theorem largest_four_digit_divisible_by_smallest_primes :
  ∀ n : Nat, n ≤ 9999 → n ≥ 1000 →
  is_divisible_by_all n smallest_primes →
  n ≤ 9240 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_smallest_primes_l737_73780


namespace NUMINAMATH_CALUDE_ellipse_condition_l737_73737

/-- 
A non-degenerate ellipse is represented by the equation 
3x^2 + 9y^2 - 12x + 27y = b if and only if b > -129/4
-/
theorem ellipse_condition (b : ℝ) : 
  (∃ (x y : ℝ), 3*x^2 + 9*y^2 - 12*x + 27*y = b ∧ 
    ∀ (x' y' : ℝ), 3*x'^2 + 9*y'^2 - 12*x' + 27*y' = b → (x', y') ≠ (x, y)) ↔ 
  b > -129/4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l737_73737


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l737_73751

theorem sqrt_equation_solution (a : ℝ) :
  Real.sqrt 3 * (a * Real.sqrt 6) = 6 * Real.sqrt 2 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l737_73751


namespace NUMINAMATH_CALUDE_triangle_area_l737_73716

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  (∀ x y z, (x = a ∧ y = b ∧ z = c) → 
    (x = 2 * (y * Real.sin C) * (z * Real.sin B) / (Real.sin A)) ∧
    (y^2 + z^2 - x^2 = 8)) →
  (b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C) →
  (b^2 + c^2 - a^2 = 8) →
  (1/2 * b * c * Real.sin A = 2 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l737_73716


namespace NUMINAMATH_CALUDE_sin_cos_sum_14_16_l737_73702

theorem sin_cos_sum_14_16 : 
  Real.sin (14 * π / 180) * Real.cos (16 * π / 180) + 
  Real.cos (14 * π / 180) * Real.sin (16 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_14_16_l737_73702


namespace NUMINAMATH_CALUDE_speaking_orders_eq_600_l737_73753

-- Define the total number of people in the group
def total_people : ℕ := 7

-- Define the number of people to be selected
def selected_people : ℕ := 4

-- Function to calculate the number of speaking orders
def speaking_orders : ℕ :=
  -- Case 1: Only one of leader or deputy participates
  (2 * (total_people - 2).choose (selected_people - 1) * (selected_people).factorial) +
  -- Case 2: Both leader and deputy participate (not adjacent)
  ((total_people - 2).choose (selected_people - 2) * selected_people.factorial -
   (total_people - 2).choose (selected_people - 2) * 2 * (selected_people - 1).factorial)

-- Theorem statement
theorem speaking_orders_eq_600 :
  speaking_orders = 600 :=
sorry

end NUMINAMATH_CALUDE_speaking_orders_eq_600_l737_73753


namespace NUMINAMATH_CALUDE_marathon_equation_l737_73722

/-- Represents the marathon race scenario -/
theorem marathon_equation (x : ℝ) (distance : ℝ) (speed_ratio : ℝ) (head_start : ℝ) :
  distance > 0 ∧ x > 0 ∧ speed_ratio > 1 ∧ head_start > 0 →
  (distance = 5) ∧ (speed_ratio = 1.5) ∧ (head_start = 12.5 / 60) →
  distance / x = distance / (speed_ratio * x) + head_start :=
by
  sorry

end NUMINAMATH_CALUDE_marathon_equation_l737_73722


namespace NUMINAMATH_CALUDE_square_pyramid_sum_l737_73725

/-- A square pyramid is a polyhedron with a square base and triangular lateral faces -/
structure SquarePyramid where
  base : Nat
  lateral_faces : Nat
  base_edges : Nat
  lateral_edges : Nat
  base_vertices : Nat
  apex : Nat

/-- Properties of a square pyramid -/
def square_pyramid : SquarePyramid :=
  { base := 1
  , lateral_faces := 4
  , base_edges := 4
  , lateral_edges := 4
  , base_vertices := 4
  , apex := 1 }

/-- The sum of faces, edges, and vertices of a square pyramid is 18 -/
theorem square_pyramid_sum :
  (square_pyramid.base + square_pyramid.lateral_faces) +
  (square_pyramid.base_edges + square_pyramid.lateral_edges) +
  (square_pyramid.base_vertices + square_pyramid.apex) = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_sum_l737_73725


namespace NUMINAMATH_CALUDE_jake_has_more_apples_l737_73709

def steven_apples : ℕ := 14

theorem jake_has_more_apples (jake_apples : ℕ) (h : jake_apples > steven_apples) :
  jake_apples > steven_apples := by sorry

end NUMINAMATH_CALUDE_jake_has_more_apples_l737_73709


namespace NUMINAMATH_CALUDE_height_comparison_l737_73735

theorem height_comparison (p q : ℝ) (h : p = 0.6 * q) :
  (q - p) / p = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_height_comparison_l737_73735


namespace NUMINAMATH_CALUDE_x_coordinate_of_R_is_one_l737_73705

/-- The curve on which point R lies -/
def curve (x y : ℝ) : Prop := y = -2 * x^2 + 5 * x - 2

/-- Predicate to check if OMRN is a square -/
def is_square (O M R N : ℝ × ℝ) : Prop := sorry

/-- Theorem stating that the x-coordinate of R is 1 -/
theorem x_coordinate_of_R_is_one 
  (R : ℝ × ℝ) 
  (h1 : curve R.1 R.2)
  (h2 : is_square (0, 0) (R.1, 0) R (0, R.2)) : 
  R.1 = 1 := by sorry

end NUMINAMATH_CALUDE_x_coordinate_of_R_is_one_l737_73705


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l737_73717

/-- The y-coordinate of the point on the y-axis equidistant from A(-3, 1) and B(2, 5) is 19/8 -/
theorem equidistant_point_y_coordinate : 
  ∃ y : ℝ, ((-3 - 0)^2 + (1 - y)^2 = (2 - 0)^2 + (5 - y)^2) ∧ y = 19/8 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l737_73717


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l737_73796

theorem pure_imaginary_complex_number (m : ℝ) : 
  (m^2 - 9 : ℂ) + (m + 3 : ℂ) * Complex.I = Complex.I * (m + 3 : ℂ) → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l737_73796


namespace NUMINAMATH_CALUDE_cone_base_diameter_l737_73710

theorem cone_base_diameter (r : ℝ) (h1 : r > 0) : 
  (π * r^2 + π * r * (2 * r) = 3 * π) → 2 * r = 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_diameter_l737_73710


namespace NUMINAMATH_CALUDE_function_characterization_l737_73711

theorem function_characterization 
  (f : ℕ → ℕ) 
  (h1 : ∀ x y : ℕ, (x + y) ∣ (f x + f y))
  (h2 : ∀ x : ℕ, x ≥ 1395 → x^3 ≥ 2 * f x) :
  ∃ k : ℕ, k ≤ 1395^2 / 2 ∧ ∀ n : ℕ, f n = k * n :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l737_73711


namespace NUMINAMATH_CALUDE_soccer_leagues_games_l737_73776

/-- Calculate the number of games in a round-robin tournament -/
def gamesInLeague (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of games played across three leagues -/
def totalGames (a b c : ℕ) : ℕ := gamesInLeague a + gamesInLeague b + gamesInLeague c

theorem soccer_leagues_games :
  totalGames 20 25 30 = 925 := by
  sorry

end NUMINAMATH_CALUDE_soccer_leagues_games_l737_73776


namespace NUMINAMATH_CALUDE_dima_numbers_l737_73752

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 4) ∨ (a = 6 ∧ b = 3)

theorem dima_numbers (a b : ℕ) :
  (4 * a = b + (a + b) + (a * b)) ∨ (4 * b = a + (a + b) + (a * b)) ∨
  (4 * (a + b) = a + b + (a * b)) →
  is_valid_pair a b := by
  sorry

end NUMINAMATH_CALUDE_dima_numbers_l737_73752


namespace NUMINAMATH_CALUDE_inheritance_calculation_l737_73729

theorem inheritance_calculation (x : ℝ) 
  (h1 : 0.25 * x + 0.1 * x = 15000) : 
  x = 42857 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l737_73729


namespace NUMINAMATH_CALUDE_det_sin_matrix_zero_l737_73707

theorem det_sin_matrix_zero : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := ![![Real.sin 1, Real.sin 2, Real.sin 3],
                                        ![Real.sin 2, Real.sin 3, Real.sin 4],
                                        ![Real.sin 3, Real.sin 4, Real.sin 5]]
  Matrix.det A = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_sin_matrix_zero_l737_73707


namespace NUMINAMATH_CALUDE_lcm_of_coprime_product_l737_73747

theorem lcm_of_coprime_product (a b : ℕ+) (h_coprime : Nat.Coprime a b) (h_product : a * b = 117) :
  Nat.lcm a b = 117 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_coprime_product_l737_73747


namespace NUMINAMATH_CALUDE_tangent_line_to_x_ln_x_l737_73713

/-- The line y = 2x - e is tangent to the curve y = x ln x -/
theorem tangent_line_to_x_ln_x : ∃ (x₀ : ℝ), 
  (x₀ * Real.log x₀ = 2 * x₀ - Real.exp 1) ∧ 
  (Real.log x₀ + 1 = 2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_x_ln_x_l737_73713


namespace NUMINAMATH_CALUDE_emily_scores_mean_l737_73727

def emily_scores : List ℕ := [84, 90, 93, 85, 91, 87]

theorem emily_scores_mean : 
  (emily_scores.sum : ℚ) / emily_scores.length = 530 / 6 := by
sorry

end NUMINAMATH_CALUDE_emily_scores_mean_l737_73727


namespace NUMINAMATH_CALUDE_child_share_proof_l737_73787

theorem child_share_proof (total_money : ℕ) (ratio : List ℕ) : 
  total_money = 4500 →
  ratio = [2, 4, 5, 4] →
  (ratio[0]! + ratio[1]!) * total_money / ratio.sum = 1800 := by
  sorry

end NUMINAMATH_CALUDE_child_share_proof_l737_73787


namespace NUMINAMATH_CALUDE_functional_equation_solution_l737_73749

-- Define a monotonic function f from real numbers to real numbers
def MonotonicFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y ∨ (∀ z : ℝ, f z = f x)

-- State the theorem
theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h_monotonic : MonotonicFunction f)
  (h_equation : ∀ x y : ℝ, f x * f y = f (x + y)) :
  ∃! a : ℝ, a > 0 ∧ a ≠ 1 ∧ (∀ x : ℝ, f x = a^x) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l737_73749


namespace NUMINAMATH_CALUDE_min_value_of_a_l737_73769

theorem min_value_of_a (a x y : ℤ) : 
  x ≠ y →
  x - y^2 = a →
  y - x^2 = a →
  |x| ≤ 10 →
  (∀ b : ℤ, (∃ x' y' : ℤ, x' ≠ y' ∧ x' - y'^2 = b ∧ y' - x'^2 = b ∧ |x'| ≤ 10) → b ≥ a) →
  a = -111 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l737_73769


namespace NUMINAMATH_CALUDE_parabola_ellipse_focus_coincide_l737_73760

/-- The value of 'a' for a parabola y^2 = ax whose focus coincides with 
    the left focus of the ellipse x^2/6 + y^2/2 = 1 -/
theorem parabola_ellipse_focus_coincide : ∃ (a : ℝ), 
  (∀ (x y : ℝ), y^2 = a*x → x^2/6 + y^2/2 = 1 → 
    (x = -2 ∧ y = 0)) → a = -8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_ellipse_focus_coincide_l737_73760


namespace NUMINAMATH_CALUDE_tan_seven_pi_sixths_l737_73703

theorem tan_seven_pi_sixths : Real.tan (7 * Real.pi / 6) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_sixths_l737_73703


namespace NUMINAMATH_CALUDE_infinite_linear_combinations_l737_73761

/-- An infinite sequence of positive integers with strictly increasing terms. -/
def StrictlyIncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ k, 0 < a k ∧ a k < a (k + 1)

/-- The property that infinitely many elements of the sequence can be written as a linear
    combination of two earlier terms with positive integer coefficients. -/
def InfinitelyManyLinearCombinations (a : ℕ → ℕ) : Prop :=
  ∀ N, ∃ m p q x y, N < m ∧ m > p ∧ p > q ∧ 0 < x ∧ 0 < y ∧ a m = x * a p + y * a q

/-- The main theorem stating that any strictly increasing sequence of positive integers
    has infinitely many elements that can be written as a linear combination of two earlier terms. -/
theorem infinite_linear_combinations
  (a : ℕ → ℕ) (h : StrictlyIncreasingSequence a) :
  InfinitelyManyLinearCombinations a := by
  sorry

end NUMINAMATH_CALUDE_infinite_linear_combinations_l737_73761


namespace NUMINAMATH_CALUDE_chris_average_speed_l737_73790

/-- Calculates the average speed given initial and final odometer readings and total time. -/
def average_speed (initial_reading : ℕ) (final_reading : ℕ) (total_time : ℕ) : ℚ :=
  (final_reading - initial_reading : ℚ) / total_time

/-- Proves that Chris's average speed is approximately 36.67 miles per hour. -/
theorem chris_average_speed :
  let initial_reading := 2332
  let final_reading := 2772
  let total_time := 12
  abs (average_speed initial_reading final_reading total_time - 36.67) < 0.01 := by
  sorry

#eval average_speed 2332 2772 12

end NUMINAMATH_CALUDE_chris_average_speed_l737_73790


namespace NUMINAMATH_CALUDE_inequality_proof_l737_73766

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0)
  (h4 : a + b + c = 2 * Real.sqrt (a * b * c)) : 
  b * c ≥ b + c := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l737_73766


namespace NUMINAMATH_CALUDE_min_daily_expense_l737_73743

/-- Represents the daily transport capacity of a truck --/
def DailyCapacity (capacity : ℕ) (trips : ℕ) : ℕ := capacity * trips

/-- Represents the total daily capacity of a fleet of trucks --/
def FleetCapacity (trucks : ℕ) (dailyCapacity : ℕ) : ℕ := trucks * dailyCapacity

/-- Represents the daily cost for a fleet of trucks --/
def FleetCost (trucks : ℕ) (cost : ℕ) : ℕ := trucks * cost

/-- The minimum daily expense problem --/
theorem min_daily_expense :
  let typeA_capacity : ℕ := 6
  let typeA_trips : ℕ := 4
  let typeA_available : ℕ := 8
  let typeA_cost : ℕ := 320
  let typeB_capacity : ℕ := 10
  let typeB_trips : ℕ := 3
  let typeB_available : ℕ := 4
  let typeB_cost : ℕ := 504
  let daily_requirement : ℕ := 180
  let typeA_daily_capacity := DailyCapacity typeA_capacity typeA_trips
  let typeB_daily_capacity := DailyCapacity typeB_capacity typeB_trips
  ∀ x y : ℕ,
    x ≤ typeA_available →
    y ≤ typeB_available →
    FleetCapacity x typeA_daily_capacity + FleetCapacity y typeB_daily_capacity ≥ daily_requirement →
    FleetCost x typeA_cost + FleetCost y typeB_cost ≥ FleetCost typeA_available typeA_cost :=
by sorry

end NUMINAMATH_CALUDE_min_daily_expense_l737_73743


namespace NUMINAMATH_CALUDE_tangent_sum_identity_l737_73746

theorem tangent_sum_identity (α β γ : ℝ) (h : α + β + γ = Real.pi / 2) :
  Real.tan α * Real.tan β + Real.tan α * Real.tan γ + Real.tan β * Real.tan γ = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_identity_l737_73746


namespace NUMINAMATH_CALUDE_division_of_fractions_l737_73789

theorem division_of_fractions : (3 : ℚ) / 7 / 5 = 3 / 35 := by sorry

end NUMINAMATH_CALUDE_division_of_fractions_l737_73789


namespace NUMINAMATH_CALUDE_problem_statement_l737_73721

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 2) :
  (1 < b ∧ b < 2) ∧ a * b < 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l737_73721


namespace NUMINAMATH_CALUDE_p_squared_plus_eight_composite_l737_73763

theorem p_squared_plus_eight_composite (p : ℕ) (h_prime : Nat.Prime p) (h_not_three : p ≠ 3) :
  ¬(Nat.Prime (p^2 + 8)) := by
  sorry

end NUMINAMATH_CALUDE_p_squared_plus_eight_composite_l737_73763


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l737_73799

theorem quadratic_coefficient (a : ℚ) : 
  (∀ x, (x + 4)^2 * a = (x + 4)^2 * (-8/9)) → 
  a = -8/9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l737_73799


namespace NUMINAMATH_CALUDE_cosine_roots_condition_l737_73744

theorem cosine_roots_condition (p q r : ℝ) : 
  (∃ a b c : ℝ, 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
    (a^3 + p*a^2 + q*a + r = 0) ∧
    (b^3 + p*b^2 + q*b + r = 0) ∧
    (c^3 + p*c^2 + q*c + r = 0) ∧
    (∃ α β γ : ℝ, 
      α + β + γ = Real.pi ∧
      a = Real.cos α ∧
      b = Real.cos β ∧
      c = Real.cos γ)) →
  p^2 = 2*q + 2*r + 1 :=
by sorry

end NUMINAMATH_CALUDE_cosine_roots_condition_l737_73744


namespace NUMINAMATH_CALUDE_largest_prime_less_than_5000_l737_73771

def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ d : Nat, d > 1 → d < p → ¬(p % d = 0)

def is_of_form (p : Nat) : Prop :=
  ∃ (a n : Nat), a > 0 ∧ n > 1 ∧ p = a^n - 1

theorem largest_prime_less_than_5000 :
  ∀ p : Nat, p < 5000 → is_prime p → is_of_form p →
  p ≤ 127 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_less_than_5000_l737_73771


namespace NUMINAMATH_CALUDE_specific_polygon_area_l737_73715

/-- A point in 2D space represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- A polygon represented by its vertices -/
structure Polygon where
  vertices : List Point

/-- Calculate the area of a rectangle given its width and height -/
def rectangleArea (width height : ℝ) : ℝ := width * height

/-- The theorem stating that the area of the specific polygon is 36 square units -/
theorem specific_polygon_area : 
  let vertices := [
    Point.mk 0 0,
    Point.mk 6 0,
    Point.mk 6 6,
    Point.mk 0 6
  ]
  let polygon := Polygon.mk vertices
  rectangleArea 6 6 = 36 := by
  sorry

#check specific_polygon_area

end NUMINAMATH_CALUDE_specific_polygon_area_l737_73715


namespace NUMINAMATH_CALUDE_max_cookies_andy_l737_73740

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem max_cookies_andy (total : ℕ) (x : ℕ) (p : ℕ) 
  (h_total : total = 30)
  (h_prime : is_prime p)
  (h_all_eaten : x + p * x = total) :
  x ≤ 10 ∧ ∃ (x₀ : ℕ) (p₀ : ℕ), x₀ = 10 ∧ is_prime p₀ ∧ x₀ + p₀ * x₀ = total :=
sorry

end NUMINAMATH_CALUDE_max_cookies_andy_l737_73740


namespace NUMINAMATH_CALUDE_james_water_storage_l737_73700

/-- Represents the water storage problem with different container types --/
structure WaterStorage where
  barrelCount : ℕ
  largeCaskCount : ℕ
  smallCaskCount : ℕ
  largeCaskCapacity : ℕ

/-- Calculates the total water storage capacity --/
def totalCapacity (storage : WaterStorage) : ℕ :=
  let barrelCapacity := 2 * storage.largeCaskCapacity + 3
  let smallCaskCapacity := storage.largeCaskCapacity / 2
  storage.barrelCount * barrelCapacity +
  storage.largeCaskCount * storage.largeCaskCapacity +
  storage.smallCaskCount * smallCaskCapacity

/-- Theorem stating that James' total water storage capacity is 282 gallons --/
theorem james_water_storage :
  let storage : WaterStorage := {
    barrelCount := 4,
    largeCaskCount := 3,
    smallCaskCount := 5,
    largeCaskCapacity := 20
  }
  totalCapacity storage = 282 := by
  sorry

end NUMINAMATH_CALUDE_james_water_storage_l737_73700


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_train_passes_jogger_in_37_seconds_l737_73712

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (initial_distance : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- The train passes the jogger in 37 seconds under the given conditions -/
theorem train_passes_jogger_in_37_seconds : 
  train_passing_jogger_time 9 45 120 250 = 37 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_train_passes_jogger_in_37_seconds_l737_73712


namespace NUMINAMATH_CALUDE_min_abc_value_l737_73777

-- Define the quadratic function P(x)
def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition for P(x) having exactly one real root
def has_one_root (a b c : ℝ) : Prop := ∃! x : ℝ, P a b c x = 0

-- Define the condition for P(P(P(x))) having exactly three different real roots
def triple_P_has_three_roots (a b c : ℝ) : Prop :=
  ∃! x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    P a b c (P a b c (P a b c x)) = 0 ∧
    P a b c (P a b c (P a b c y)) = 0 ∧
    P a b c (P a b c (P a b c z)) = 0

-- State the theorem
theorem min_abc_value (a b c : ℝ) :
  has_one_root a b c →
  triple_P_has_three_roots a b c →
  ∀ a' b' c' : ℝ, has_one_root a' b' c' → triple_P_has_three_roots a' b' c' →
    a * b * c ≤ a' * b' * c' →
    a * b * c = -2 :=
sorry

end NUMINAMATH_CALUDE_min_abc_value_l737_73777


namespace NUMINAMATH_CALUDE_age_double_time_l737_73784

/-- Given Julio's current age is 42 and James' current age is 8,
    this theorem proves that it will take 26 years for Julio's age to be twice James' age. -/
theorem age_double_time (julio_age : ℕ) (james_age : ℕ) (h1 : julio_age = 42) (h2 : james_age = 8) :
  ∃ (years : ℕ), julio_age + years = 2 * (james_age + years) ∧ years = 26 := by
  sorry

end NUMINAMATH_CALUDE_age_double_time_l737_73784


namespace NUMINAMATH_CALUDE_average_difference_l737_73786

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = (10 + 50 + x) / 3 + 5 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l737_73786


namespace NUMINAMATH_CALUDE_total_kids_played_tag_l737_73775

def monday : ℕ := 12
def tuesday : ℕ := 7
def wednesday : ℕ := 15
def thursday : ℕ := 10
def friday : ℕ := 18

theorem total_kids_played_tag : monday + tuesday + wednesday + thursday + friday = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_kids_played_tag_l737_73775


namespace NUMINAMATH_CALUDE_stair_steps_left_l737_73714

theorem stair_steps_left (total : ℕ) (climbed : ℕ) (h1 : total = 96) (h2 : climbed = 74) :
  total - climbed = 22 := by
  sorry

end NUMINAMATH_CALUDE_stair_steps_left_l737_73714


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l737_73762

theorem min_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 2) :
  (1/x + 1/y + 1/z) ≥ 4.5 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ x' + y' + z' = 2 ∧ 1/x' + 1/y' + 1/z' = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l737_73762


namespace NUMINAMATH_CALUDE_cube_root_cube_identity_l737_73765

theorem cube_root_cube_identity (x : ℝ) : (x^3)^(1/3) = x := by
  sorry

end NUMINAMATH_CALUDE_cube_root_cube_identity_l737_73765


namespace NUMINAMATH_CALUDE_julie_school_year_hours_l737_73768

/-- Calculates the number of hours Julie needs to work per week during the school year
    to maintain the same rate of pay as her summer job. -/
theorem julie_school_year_hours
  (summer_weeks : ℕ)
  (summer_hours_per_week : ℕ)
  (summer_earnings : ℕ)
  (school_year_weeks : ℕ)
  (school_year_earnings : ℕ)
  (h1 : summer_weeks = 15)
  (h2 : summer_hours_per_week = 40)
  (h3 : summer_earnings = 6000)
  (h4 : school_year_weeks = 30)
  (h5 : school_year_earnings = 7500)
  : (school_year_earnings * summer_weeks * summer_hours_per_week) / 
    (summer_earnings * school_year_weeks) = 25 := by
  sorry


end NUMINAMATH_CALUDE_julie_school_year_hours_l737_73768


namespace NUMINAMATH_CALUDE_class_average_theorem_l737_73793

theorem class_average_theorem (group1_percent : Real) (group1_avg : Real)
                              (group2_percent : Real) (group2_avg : Real)
                              (group3_percent : Real) (group3_avg : Real) :
  group1_percent = 0.45 →
  group1_avg = 0.95 →
  group2_percent = 0.50 →
  group2_avg = 0.78 →
  group3_percent = 1 - group1_percent - group2_percent →
  group3_avg = 0.60 →
  round ((group1_percent * group1_avg + group2_percent * group2_avg + group3_percent * group3_avg) * 100) = 85 :=
by
  sorry

#check class_average_theorem

end NUMINAMATH_CALUDE_class_average_theorem_l737_73793


namespace NUMINAMATH_CALUDE_area_of_closed_region_l737_73778

-- Define the functions
def f₀ (x : ℝ) := |x|
def f₁ (x : ℝ) := |f₀ x - 1|
def f₂ (x : ℝ) := |f₁ x - 2|

-- Define the area function
noncomputable def area_under_curve (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, f x

-- Theorem statement
theorem area_of_closed_region :
  area_under_curve f₂ (-3) 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_area_of_closed_region_l737_73778


namespace NUMINAMATH_CALUDE_original_rectangle_area_l737_73797

/-- Given a rectangle whose dimensions are doubled to form a new rectangle with an area of 32 square meters, 
    the area of the original rectangle is 8 square meters. -/
theorem original_rectangle_area (original_length original_width : ℝ) 
  (new_length new_width : ℝ) (new_area : ℝ) :
  new_length = 2 * original_length →
  new_width = 2 * original_width →
  new_area = new_length * new_width →
  new_area = 32 →
  original_length * original_width = 8 :=
by sorry

end NUMINAMATH_CALUDE_original_rectangle_area_l737_73797


namespace NUMINAMATH_CALUDE_distribute_volunteers_count_l737_73783

/-- The number of ways to distribute 5 volunteers into 4 groups -/
def distribute_volunteers : ℕ :=
  Nat.choose 5 2 * Nat.factorial 4

/-- Theorem stating that the number of distribution methods is 240 -/
theorem distribute_volunteers_count : distribute_volunteers = 240 := by
  sorry

end NUMINAMATH_CALUDE_distribute_volunteers_count_l737_73783


namespace NUMINAMATH_CALUDE_expand_and_simplify_l737_73736

theorem expand_and_simplify (x : ℝ) : 6 * (x - 3) * (x + 10) = 6 * x^2 + 42 * x - 180 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l737_73736


namespace NUMINAMATH_CALUDE_eva_total_marks_2019_l737_73724

/-- Eva's marks in different subjects and semesters -/
structure EvaMarks where
  maths_second : ℕ
  arts_second : ℕ
  science_second : ℕ
  maths_first : ℕ
  arts_first : ℕ
  science_first : ℕ

/-- Calculate the total marks for Eva in 2019 -/
def total_marks (marks : EvaMarks) : ℕ :=
  marks.maths_first + marks.arts_first + marks.science_first +
  marks.maths_second + marks.arts_second + marks.science_second

/-- Theorem stating Eva's total marks in 2019 -/
theorem eva_total_marks_2019 (marks : EvaMarks)
  (h1 : marks.maths_second = 80)
  (h2 : marks.arts_second = 90)
  (h3 : marks.science_second = 90)
  (h4 : marks.maths_first = marks.maths_second + 10)
  (h5 : marks.arts_first = marks.arts_second - 15)
  (h6 : marks.science_first = marks.science_second - marks.science_second / 3) :
  total_marks marks = 485 := by
  sorry


end NUMINAMATH_CALUDE_eva_total_marks_2019_l737_73724


namespace NUMINAMATH_CALUDE_tim_books_l737_73794

theorem tim_books (sam_books : ℕ) (total_books : ℕ) (h1 : sam_books = 52) (h2 : total_books = 96) :
  total_books - sam_books = 44 := by
  sorry

end NUMINAMATH_CALUDE_tim_books_l737_73794


namespace NUMINAMATH_CALUDE_min_value_of_expression_l737_73745

theorem min_value_of_expression (x y : ℝ) : (x*y - 1)^3 + (x + y)^3 ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l737_73745


namespace NUMINAMATH_CALUDE_daves_shirts_l737_73795

theorem daves_shirts (short_sleeve : ℕ) (washed : ℕ) (unwashed : ℕ) 
  (h1 : short_sleeve = 9)
  (h2 : washed = 20)
  (h3 : unwashed = 16) :
  washed + unwashed - short_sleeve = 27 := by
  sorry

end NUMINAMATH_CALUDE_daves_shirts_l737_73795


namespace NUMINAMATH_CALUDE_rowing_distance_problem_l737_73791

/-- Proves that given a man who can row 7.5 km/hr in still water, in a river flowing at 1.5 km/hr,
    if it takes him 50 minutes to row to a place and back, the distance to that place is 3 km. -/
theorem rowing_distance_problem (man_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) :
  man_speed = 7.5 →
  river_speed = 1.5 →
  total_time = 50 / 60 →
  ∃ (distance : ℝ),
    distance / (man_speed - river_speed) + distance / (man_speed + river_speed) = total_time ∧
    distance = 3 :=
by sorry

end NUMINAMATH_CALUDE_rowing_distance_problem_l737_73791


namespace NUMINAMATH_CALUDE_g_approaches_neg_inf_pos_g_approaches_neg_inf_neg_l737_73728

/-- The function g(x) = -3x^4 + 50x^2 - 1 -/
def g (x : ℝ) : ℝ := -3 * x^4 + 50 * x^2 - 1

/-- Theorem stating that g(x) approaches negative infinity as x approaches positive infinity -/
theorem g_approaches_neg_inf_pos (ε : ℝ) : ∃ M : ℝ, ∀ x : ℝ, x > M → g x < ε :=
sorry

/-- Theorem stating that g(x) approaches negative infinity as x approaches negative infinity -/
theorem g_approaches_neg_inf_neg (ε : ℝ) : ∃ M : ℝ, ∀ x : ℝ, x < -M → g x < ε :=
sorry

end NUMINAMATH_CALUDE_g_approaches_neg_inf_pos_g_approaches_neg_inf_neg_l737_73728


namespace NUMINAMATH_CALUDE_exp_five_factorial_30_l737_73706

/-- The exponent of 5 in the prime factorization of n! -/
def exp_five_factorial (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The exponent of 5 in the prime factorization of 30! is 7 -/
theorem exp_five_factorial_30 : exp_five_factorial 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_exp_five_factorial_30_l737_73706


namespace NUMINAMATH_CALUDE_kenzo_round_tables_l737_73770

/-- The number of round tables Kenzo initially had -/
def num_round_tables : ℕ := 20

/-- The number of office chairs Kenzo initially had -/
def initial_chairs : ℕ := 80

/-- The number of legs each office chair has -/
def legs_per_chair : ℕ := 5

/-- The number of legs each round table has -/
def legs_per_table : ℕ := 3

/-- The percentage of chairs that were damaged and disposed of -/
def damaged_chair_percentage : ℚ := 40 / 100

/-- The total number of remaining legs of furniture -/
def total_remaining_legs : ℕ := 300

theorem kenzo_round_tables :
  num_round_tables * legs_per_table = 
    total_remaining_legs - 
    (initial_chairs * (1 - damaged_chair_percentage) : ℚ).num * legs_per_chair :=
by sorry

end NUMINAMATH_CALUDE_kenzo_round_tables_l737_73770


namespace NUMINAMATH_CALUDE_floor_sqrt_5_minus_3_l737_73730

theorem floor_sqrt_5_minus_3 : ⌊Real.sqrt 5 - 3⌋ = -1 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_5_minus_3_l737_73730


namespace NUMINAMATH_CALUDE_downstream_speed_l737_73701

/-- Calculates the downstream speed of a rower given upstream and still water speeds -/
theorem downstream_speed (upstream_speed still_water_speed : ℝ) :
  upstream_speed = 20 →
  still_water_speed = 24 →
  still_water_speed + (still_water_speed - upstream_speed) = 28 := by
  sorry


end NUMINAMATH_CALUDE_downstream_speed_l737_73701


namespace NUMINAMATH_CALUDE_two_zeros_iff_a_in_set_l737_73739

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2*x - |x^2 - a*x + 1|

/-- The set of a values that satisfy the condition -/
def A : Set ℝ := {a | a < 0 ∨ (0 < a ∧ a < 1) ∨ 1 < a}

theorem two_zeros_iff_a_in_set (a : ℝ) : 
  (∃! (x y : ℝ), x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ a ∈ A := by sorry

end NUMINAMATH_CALUDE_two_zeros_iff_a_in_set_l737_73739


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l737_73773

theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y - 11 = 0 →
  ∃ (h k r : ℝ), h = -1 ∧ k = 2 ∧ r = 2 ∧
  (x - h)^2 + (y - k)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l737_73773


namespace NUMINAMATH_CALUDE_chord_length_is_four_l737_73788

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := y = x + 2

def C2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the chord length
def chord_length (C1 C2 : ℝ → ℝ → Prop) : ℝ :=
  4 -- The actual calculation is omitted, we just state the result

-- Theorem statement
theorem chord_length_is_four :
  chord_length C1 C2 = 4 := by sorry

end NUMINAMATH_CALUDE_chord_length_is_four_l737_73788


namespace NUMINAMATH_CALUDE_john_walked_four_miles_l737_73772

/-- Represents the distance John traveled in miles -/
structure JohnTravel where
  initial_skate : ℝ
  total_skate : ℝ
  walk : ℝ

/-- The conditions of John's travel -/
def travel_conditions (j : JohnTravel) : Prop :=
  j.initial_skate = 10 ∧ 
  j.total_skate = 24 ∧
  j.total_skate = 2 * j.initial_skate + j.walk

/-- Theorem stating that John walked 4 miles to the park -/
theorem john_walked_four_miles (j : JohnTravel) 
  (h : travel_conditions j) : j.walk = 4 := by
  sorry

end NUMINAMATH_CALUDE_john_walked_four_miles_l737_73772


namespace NUMINAMATH_CALUDE_dot_product_range_l737_73720

-- Define the set of points satisfying the given inequalities
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 + 2 * p.2 ≤ 6 ∧ 3 * p.1 + p.2 ≤ 12}

-- Define vector a
def a : ℝ × ℝ := (1, -1)

-- Define the dot product function
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the function to be maximized/minimized
def f (m n : ℝ × ℝ) : ℝ := dot_product (n.1 - m.1, n.2 - m.2) a

-- Theorem statement
theorem dot_product_range :
  ∀ (m n : ℝ × ℝ), m ∈ S → n ∈ S →
  -7 ≤ f m n ∧ f m n ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_dot_product_range_l737_73720


namespace NUMINAMATH_CALUDE_nine_rings_puzzle_l737_73755

def min_moves : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | (n + 3) => min_moves (n + 2) + 2 * min_moves (n + 1) + 1

theorem nine_rings_puzzle : min_moves 7 = 85 := by
  sorry

end NUMINAMATH_CALUDE_nine_rings_puzzle_l737_73755


namespace NUMINAMATH_CALUDE_unique_positive_solution_l737_73785

theorem unique_positive_solution : 
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x := by
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l737_73785


namespace NUMINAMATH_CALUDE_sum_of_integers_l737_73708

theorem sum_of_integers (x y : ℕ+) 
  (sum_of_squares : x^2 + y^2 = 245)
  (product : x * y = 120) : 
  (x : ℝ) + y = Real.sqrt 485 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l737_73708


namespace NUMINAMATH_CALUDE_celsius_to_fahrenheit_l737_73798

/-- Given the relationship between Celsius (C) and Fahrenheit (F) temperatures,
    prove that when C is 25, F is 75. -/
theorem celsius_to_fahrenheit (C F : ℚ) : 
  C = 25 → C = (5 / 9) * (F - 30) → F = 75 := by sorry

end NUMINAMATH_CALUDE_celsius_to_fahrenheit_l737_73798


namespace NUMINAMATH_CALUDE_maria_number_puzzle_l737_73738

theorem maria_number_puzzle (x : ℝ) : 
  (((x + 3) * 2 - 4) / 3 = 10) → x = 14 := by
  sorry

end NUMINAMATH_CALUDE_maria_number_puzzle_l737_73738


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l737_73792

/-- Given vectors a and b in ℝ², prove that if (a + kb) ⊥ (a - kb), then k = ±√5 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (2, -1))
  (h2 : b = (-Real.sqrt 3 / 2, -1 / 2))
  (h3 : (a.1 + k * b.1, a.2 + k * b.2) • (a.1 - k * b.1, a.2 - k * b.2) = 0) :
  k = Real.sqrt 5 ∨ k = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l737_73792


namespace NUMINAMATH_CALUDE_expression_simplification_l737_73754

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 3) :
  (3 * x) / (x^2 - 9) * (1 - 3 / x) - 2 / (x + 3) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l737_73754


namespace NUMINAMATH_CALUDE_profit_loss_equality_l737_73750

/-- Given an article with cost price C, prove that if the profit at selling price S_p
    equals the loss when sold at $448, then the selling price for 30% profit is 1.30C. -/
theorem profit_loss_equality (C : ℝ) (S_p : ℝ) :
  S_p - C = C - 448 →
  ∃ (S_30 : ℝ), S_30 = 1.30 * C ∧ S_30 - C = 0.30 * C :=
by sorry

end NUMINAMATH_CALUDE_profit_loss_equality_l737_73750
