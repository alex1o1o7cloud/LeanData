import Mathlib

namespace NUMINAMATH_CALUDE_prob_two_consecutive_accurate_value_l2793_279360

/-- The accuracy rate of the weather forecast for each day -/
def accuracy_rate : ℝ := 0.8

/-- The probability of having at least two consecutive days with accurate forecasts
    out of three days, given the accuracy rate for each day -/
def prob_two_consecutive_accurate (p : ℝ) : ℝ :=
  p^3 + p^2 * (1 - p) + (1 - p) * p^2

/-- Theorem stating that the probability of having at least two consecutive days
    with accurate forecasts out of three days, given an accuracy rate of 0.8,
    is equal to 0.768 -/
theorem prob_two_consecutive_accurate_value :
  prob_two_consecutive_accurate accuracy_rate = 0.768 := by
  sorry


end NUMINAMATH_CALUDE_prob_two_consecutive_accurate_value_l2793_279360


namespace NUMINAMATH_CALUDE_watermelon_duration_example_l2793_279323

/-- The number of weeks watermelons will last -/
def watermelon_duration (total : ℕ) (weekly_usage : ℕ) : ℕ :=
  total / weekly_usage

/-- Theorem: Given 30 watermelons and using 5 per week, they will last 6 weeks -/
theorem watermelon_duration_example : watermelon_duration 30 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_duration_example_l2793_279323


namespace NUMINAMATH_CALUDE_quarters_in_eighth_l2793_279340

theorem quarters_in_eighth : (1 : ℚ) / 8 / ((1 : ℚ) / 4) = (1 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_quarters_in_eighth_l2793_279340


namespace NUMINAMATH_CALUDE_equation_solution_l2793_279318

theorem equation_solution :
  ∀ x : ℚ, (1 / 4 : ℚ) + 7 / x = 13 / x + (1 / 9 : ℚ) → x = 216 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2793_279318


namespace NUMINAMATH_CALUDE_go_complexity_ratio_l2793_279373

/-- The upper limit of the state space complexity of Go -/
def M : ℝ := 3^361

/-- The total number of atoms of ordinary matter in the observable universe -/
def N : ℝ := 10^80

/-- Theorem stating that M/N is approximately equal to 10^93 -/
theorem go_complexity_ratio : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |M / N - 10^93| < ε := by
  sorry

end NUMINAMATH_CALUDE_go_complexity_ratio_l2793_279373


namespace NUMINAMATH_CALUDE_unique_number_with_gcd_l2793_279306

theorem unique_number_with_gcd (n : ℕ) : 
  70 < n ∧ n < 80 ∧ Nat.gcd 15 n = 5 → n = 75 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_gcd_l2793_279306


namespace NUMINAMATH_CALUDE_book_price_calculation_l2793_279313

/-- Represents the price of a single book -/
def book_price : ℝ := 20

/-- Represents the number of books bought per month -/
def books_per_month : ℕ := 3

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents the total sale price of all books at the end of the year -/
def total_sale_price : ℝ := 500

/-- Represents the total loss incurred -/
def total_loss : ℝ := 220

theorem book_price_calculation : 
  book_price * (books_per_month * months_in_year) - total_sale_price = total_loss :=
sorry

end NUMINAMATH_CALUDE_book_price_calculation_l2793_279313


namespace NUMINAMATH_CALUDE_alberts_cabbage_rows_l2793_279383

/-- Represents Albert's cabbage patch -/
structure CabbagePatch where
  total_heads : ℕ
  heads_per_row : ℕ

/-- Calculates the number of rows in the cabbage patch -/
def number_of_rows (patch : CabbagePatch) : ℕ :=
  patch.total_heads / patch.heads_per_row

/-- Theorem stating the number of rows in Albert's cabbage patch -/
theorem alberts_cabbage_rows :
  let patch : CabbagePatch := { total_heads := 180, heads_per_row := 15 }
  number_of_rows patch = 12 := by
  sorry

end NUMINAMATH_CALUDE_alberts_cabbage_rows_l2793_279383


namespace NUMINAMATH_CALUDE_triangle_existence_l2793_279301

-- Define the basic types and structures
structure Point := (x y : ℝ)

def Angle := ℝ

-- Define the given elements
variable (F T : Point) -- F is midpoint of AB, T is foot of altitude
variable (α : Angle) -- angle at vertex A

-- Define the properties of the triangle
def is_midpoint (F A B : Point) : Prop := F = Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2)

def is_altitude_foot (T A C : Point) : Prop := 
  (T.x - A.x) * (C.x - A.x) + (T.y - A.y) * (C.y - A.y) = 0

def angle_at_vertex (A B C : Point) (α : Angle) : Prop :=
  let v1 := Point.mk (B.x - A.x) (B.y - A.y)
  let v2 := Point.mk (C.x - A.x) (C.y - A.y)
  Real.cos α = (v1.x * v2.x + v1.y * v2.y) / 
    (Real.sqrt (v1.x^2 + v1.y^2) * Real.sqrt (v2.x^2 + v2.y^2))

-- The main theorem
theorem triangle_existence (F T : Point) (α : Angle) :
  ∃ (A B C : Point),
    is_midpoint F A B ∧
    is_altitude_foot T A C ∧
    angle_at_vertex A B C α ∧
    ¬(∀ (C' : Point), is_altitude_foot T A C' → C' = C) :=
by sorry

end NUMINAMATH_CALUDE_triangle_existence_l2793_279301


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l2793_279381

theorem complete_square_quadratic : ∀ x : ℝ, x^2 - 4*x + 2 = 0 ↔ (x - 2)^2 = 2 := by sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l2793_279381


namespace NUMINAMATH_CALUDE_optimal_greening_arrangement_l2793_279321

/-- Represents a construction team with daily greening area and cost -/
structure Team where
  daily_area : ℝ
  daily_cost : ℝ

/-- The optimal greening arrangement problem -/
def OptimalGreeningArrangement (total_area : ℝ) (max_days : ℕ) (team_a team_b : Team) : Prop :=
  -- Team A is 1.8 times more efficient than Team B
  team_a.daily_area = 1.8 * team_b.daily_area ∧
  -- Team A takes 4 days less than Team B for 450 m²
  (450 / team_a.daily_area) + 4 = 450 / team_b.daily_area ∧
  -- Optimal arrangement
  ∃ (days_a days_b : ℕ),
    -- Total area constraint
    team_a.daily_area * days_a + team_b.daily_area * days_b ≥ total_area ∧
    -- Time constraint
    days_a + days_b ≤ max_days ∧
    -- Optimal solution
    days_a = 30 ∧ days_b = 18 ∧
    -- Minimum cost
    team_a.daily_cost * days_a + team_b.daily_cost * days_b = 40.5

/-- Theorem stating the optimal greening arrangement -/
theorem optimal_greening_arrangement :
  OptimalGreeningArrangement 3600 48
    (Team.mk 90 1.05)
    (Team.mk 50 0.5) := by
  sorry

end NUMINAMATH_CALUDE_optimal_greening_arrangement_l2793_279321


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2793_279308

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x^2 - x - 6 > 0) ↔ (x < -3/2 ∨ x > 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2793_279308


namespace NUMINAMATH_CALUDE_square_of_difference_product_of_three_terms_l2793_279386

-- Problem 1
theorem square_of_difference (a b : ℝ) : (a^2 - b)^2 = a^4 - 2*a^2*b + b^2 := by sorry

-- Problem 2
theorem product_of_three_terms (x : ℝ) : (2*x + 1)*(4*x^2 - 1)*(2*x - 1) = 16*x^4 - 8*x^2 + 1 := by sorry

end NUMINAMATH_CALUDE_square_of_difference_product_of_three_terms_l2793_279386


namespace NUMINAMATH_CALUDE_sin_difference_inequality_l2793_279351

theorem sin_difference_inequality (a b : ℝ) :
  ((0 ≤ a ∧ a < b ∧ b ≤ π / 2) ∨ (π ≤ a ∧ a < b ∧ b ≤ 3 * π / 2)) →
  a - Real.sin a < b - Real.sin b :=
by sorry

end NUMINAMATH_CALUDE_sin_difference_inequality_l2793_279351


namespace NUMINAMATH_CALUDE_cos_18_minus_cos_54_l2793_279350

theorem cos_18_minus_cos_54 :
  Real.cos (18 * π / 180) - Real.cos (54 * π / 180) =
  -16 * (Real.cos (9 * π / 180))^4 + 24 * (Real.cos (9 * π / 180))^2 - 4 := by
sorry

end NUMINAMATH_CALUDE_cos_18_minus_cos_54_l2793_279350


namespace NUMINAMATH_CALUDE_unique_prime_between_30_and_40_with_remainder_7_mod_9_l2793_279367

theorem unique_prime_between_30_and_40_with_remainder_7_mod_9 :
  ∃! p : ℕ, Prime p ∧ 30 < p ∧ p < 40 ∧ p % 9 = 7 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_between_30_and_40_with_remainder_7_mod_9_l2793_279367


namespace NUMINAMATH_CALUDE_crucian_carps_count_l2793_279335

/-- The number of bags of feed each fish eats -/
def bags_per_fish : ℕ := 3

/-- The number of individual feed bags prepared -/
def individual_bags : ℕ := 60

/-- The number of 8-packet feed bags prepared -/
def multi_bags : ℕ := 15

/-- The number of packets in each multi-bag -/
def packets_per_multi_bag : ℕ := 8

/-- The number of colored carps in the tank -/
def colored_carps : ℕ := 52

/-- The total number of feed packets available -/
def total_packets : ℕ := individual_bags + multi_bags * packets_per_multi_bag

/-- The total number of fish that can be fed -/
def total_fish : ℕ := total_packets / bags_per_fish

/-- The number of crucian carps in the tank -/
def crucian_carps : ℕ := total_fish - colored_carps

theorem crucian_carps_count : crucian_carps = 8 := by
  sorry

end NUMINAMATH_CALUDE_crucian_carps_count_l2793_279335


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2793_279369

/-- Definition of the hyperbola -/
def hyperbola (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - (x + 2)^2 / 25 = 1

/-- Definition of an asymptote -/
def is_asymptote (m b : ℝ) : Prop :=
  ∀ ε > 0, ∃ M > 0, ∀ x y : ℝ, 
    hyperbola x y → (|x| > M → |y - (m * x + b)| < ε)

/-- Theorem: The asymptotes of the given hyperbola -/
theorem hyperbola_asymptotes :
  (is_asymptote (4/5) (13/5)) ∧ (is_asymptote (-4/5) (13/5)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2793_279369


namespace NUMINAMATH_CALUDE_candy_per_box_l2793_279372

/-- Given that Billy bought 7 boxes of candy and had a total of 21 pieces,
    prove that each box contained 3 pieces of candy. -/
theorem candy_per_box (num_boxes : ℕ) (total_pieces : ℕ) (h1 : num_boxes = 7) (h2 : total_pieces = 21) :
  total_pieces / num_boxes = 3 := by
sorry

end NUMINAMATH_CALUDE_candy_per_box_l2793_279372


namespace NUMINAMATH_CALUDE_experts_win_probability_l2793_279327

/-- The probability of Experts winning a single round -/
def p : ℝ := 0.6

/-- The probability of Viewers winning a single round -/
def q : ℝ := 1 - p

/-- The current score of Experts -/
def expertsScore : ℕ := 3

/-- The current score of Viewers -/
def viewersScore : ℕ := 4

/-- The number of rounds needed to win the game -/
def winningScore : ℕ := 6

/-- The probability that the Experts will eventually win the game -/
def expertsWinProbability : ℝ := p^4 + 4 * p^3 * q

theorem experts_win_probability : 
  expertsWinProbability = 0.4752 := by sorry

end NUMINAMATH_CALUDE_experts_win_probability_l2793_279327


namespace NUMINAMATH_CALUDE_stanley_distance_difference_l2793_279368

/-- Given Stanley's running and walking distances, prove the difference between them. -/
theorem stanley_distance_difference (run walk : ℝ) 
  (h1 : run = 0.4) 
  (h2 : walk = 0.2) : 
  run - walk = 0.2 := by
sorry

end NUMINAMATH_CALUDE_stanley_distance_difference_l2793_279368


namespace NUMINAMATH_CALUDE_password_decryption_probability_l2793_279348

theorem password_decryption_probability :
  let p₁ : ℚ := 1/5
  let p₂ : ℚ := 2/5
  let p₃ : ℚ := 1/2
  let prob_at_least_one_success : ℚ := 1 - (1 - p₁) * (1 - p₂) * (1 - p₃)
  prob_at_least_one_success = 19/25 :=
by sorry

end NUMINAMATH_CALUDE_password_decryption_probability_l2793_279348


namespace NUMINAMATH_CALUDE_fixed_point_of_line_family_l2793_279375

/-- The fixed point that a family of lines passes through -/
theorem fixed_point_of_line_family :
  ∃! p : ℝ × ℝ, ∀ m : ℝ, (2*m - 1) * p.1 + (m + 3) * p.2 - (m - 11) = 0 :=
by
  -- The unique point is (2, -3)
  use (2, -3)
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_family_l2793_279375


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2793_279309

theorem sqrt_inequality (x : ℝ) : 
  Real.sqrt (3 - x) - Real.sqrt (x + 1) > (1 : ℝ) / 2 ↔ 
  -1 ≤ x ∧ x < 1 - Real.sqrt 31 / 8 := by
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2793_279309


namespace NUMINAMATH_CALUDE_parking_lot_wheels_l2793_279331

/-- The number of wheels on a car -/
def car_wheels : ℕ := 4

/-- The number of wheels on a bike -/
def bike_wheels : ℕ := 2

/-- The number of cars in the parking lot -/
def num_cars : ℕ := 10

/-- The number of bikes in the parking lot -/
def num_bikes : ℕ := 2

/-- The total number of wheels in the parking lot -/
def total_wheels : ℕ := num_cars * car_wheels + num_bikes * bike_wheels

theorem parking_lot_wheels : total_wheels = 44 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_wheels_l2793_279331


namespace NUMINAMATH_CALUDE_valid_triangulations_l2793_279314

/-- A triangulation of a triangle is a division of the triangle into n smaller triangles
    such that no three vertices are collinear and each vertex belongs to the same number of segments -/
structure Triangulation :=
  (n : ℕ)  -- number of smaller triangles
  (no_collinear : Bool)  -- no three vertices are collinear
  (equal_vertex_degree : Bool)  -- each vertex belongs to the same number of segments

/-- The set of valid n values for triangulations -/
def ValidTriangulations : Set ℕ := {1, 3, 7, 19}

/-- Theorem stating that the only valid triangulations are those with n in ValidTriangulations -/
theorem valid_triangulations (t : Triangulation) :
  t.no_collinear ∧ t.equal_vertex_degree → t.n ∈ ValidTriangulations := by
  sorry

end NUMINAMATH_CALUDE_valid_triangulations_l2793_279314


namespace NUMINAMATH_CALUDE_tenth_number_with_digit_sum_12_l2793_279326

/-- A function that returns the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits add up to 12 -/
def nth_number_with_digit_sum_12 (n : ℕ) : ℕ := sorry

/-- The theorem stating that the 10th number with digit sum 12 is 147 -/
theorem tenth_number_with_digit_sum_12 : nth_number_with_digit_sum_12 10 = 147 := by
  sorry

end NUMINAMATH_CALUDE_tenth_number_with_digit_sum_12_l2793_279326


namespace NUMINAMATH_CALUDE_fraction_equality_l2793_279361

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4*x + y) / (x - 4*y) = -3) : 
  (x + 4*y) / (4*x - y) = 39/37 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2793_279361


namespace NUMINAMATH_CALUDE_average_speed_distance_expression_time_range_l2793_279366

-- Define the boat's movement
structure BoatMovement where
  distance : ℕ → ℝ
  time : ℕ → ℝ

-- Define the given data
def givenData : BoatMovement := {
  distance := λ n => match n with
    | 0 => 200
    | 1 => 150
    | 2 => 100
    | 3 => 50
    | _ => 0
  time := λ n => 2 * n
}

-- Theorem for the average speed
theorem average_speed (b : BoatMovement) : 
  (b.distance 0 - b.distance 3) / (b.time 3 - b.time 0) = 25 := by
  sorry

-- Theorem for the analytical expression
theorem distance_expression (b : BoatMovement) (x : ℝ) : 
  ∃ y : ℝ, y = 200 - 25 * x := by
  sorry

-- Theorem for the range of x
theorem time_range (b : BoatMovement) (x : ℝ) : 
  0 ≤ x ∧ x ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_distance_expression_time_range_l2793_279366


namespace NUMINAMATH_CALUDE_prime_sum_squares_l2793_279353

theorem prime_sum_squares (p q : ℕ) : 
  Prime p → Prime q → 
  ∃ (x y : ℕ), x^2 = p + q ∧ y^2 = p + 7*q → 
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_squares_l2793_279353


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l2793_279319

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, 2 * x^2 + b * x + 16 = 0 → x.im ≠ 0) ↔ b ∈ Set.Ioo (-8 * Real.sqrt 2) (8 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l2793_279319


namespace NUMINAMATH_CALUDE_dog_bones_found_l2793_279312

/-- Given a dog initially has 15 bones and ends up with 23 bones, 
    prove that the number of bones found is 23 - 15. -/
theorem dog_bones_found (initial_bones final_bones : ℕ) 
  (h1 : initial_bones = 15) 
  (h2 : final_bones = 23) : 
  final_bones - initial_bones = 23 - 15 := by
  sorry

end NUMINAMATH_CALUDE_dog_bones_found_l2793_279312


namespace NUMINAMATH_CALUDE_fifteen_factorial_base_twelve_zeroes_l2793_279329

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem fifteen_factorial_base_twelve_zeroes :
  ∃ k : ℕ, k = 5 ∧ 12^k ∣ factorial 15 ∧ ¬(12^(k+1) ∣ factorial 15) :=
by sorry

end NUMINAMATH_CALUDE_fifteen_factorial_base_twelve_zeroes_l2793_279329


namespace NUMINAMATH_CALUDE_equation_solution_l2793_279337

theorem equation_solution : ∃ x : ℝ, (6*x + 7)^2 * (3*x + 4) * (x + 1) = 6 :=
  have h1 : (6 * (-2/3) + 7)^2 * (3 * (-2/3) + 4) * (-2/3 + 1) = 6 := by sorry
  have h2 : (6 * (-5/3) + 7)^2 * (3 * (-5/3) + 4) * (-5/3 + 1) = 6 := by sorry
  ⟨-2/3, h1⟩

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l2793_279337


namespace NUMINAMATH_CALUDE_four_square_rectangle_exists_l2793_279305

/-- Represents a color --/
structure Color : Type

/-- Represents a square on the grid --/
structure Square : Type :=
  (x : ℤ)
  (y : ℤ)
  (color : Color)

/-- Represents an infinite grid of colored squares --/
def InfiniteGrid : Type := ℤ → ℤ → Color

/-- Checks if four squares form a rectangle parallel to grid lines --/
def IsRectangle (s1 s2 s3 s4 : Square) : Prop :=
  (s1.x = s2.x ∧ s3.x = s4.x ∧ s1.y = s3.y ∧ s2.y = s4.y) ∨
  (s1.x = s3.x ∧ s2.x = s4.x ∧ s1.y = s2.y ∧ s3.y = s4.y)

/-- Main theorem: There always exist four squares of the same color forming a rectangle --/
theorem four_square_rectangle_exists (n : ℕ) (h : n ≥ 2) (grid : InfiniteGrid) :
  ∃ (s1 s2 s3 s4 : Square),
    s1.color = s2.color ∧ s2.color = s3.color ∧ s3.color = s4.color ∧
    IsRectangle s1 s2 s3 s4 := by
  sorry

end NUMINAMATH_CALUDE_four_square_rectangle_exists_l2793_279305


namespace NUMINAMATH_CALUDE_expression_simplification_l2793_279370

theorem expression_simplification :
  let a : ℚ := 3 / 2015
  let b : ℚ := 11 / 2016
  (6 + a) * (8 + b) - (11 - a) * (3 - b) - 12 * a = 11 / 112 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2793_279370


namespace NUMINAMATH_CALUDE_max_k_value_l2793_279332

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 4 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ 4 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 = 4^2 * (x^2 / y^2 + y^2 / x^2) + 4 * (x / y + y / x) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l2793_279332


namespace NUMINAMATH_CALUDE_smallest_a_is_correct_l2793_279387

/-- The smallest positive integer a such that both 112 and 33 are factors of a * 43 * 62 * 1311 -/
def smallest_a : ℕ := 1848

/-- Predicate to check if a number divides the product a * 43 * 62 * 1311 -/
def is_factor (n : ℕ) (a : ℕ) : Prop :=
  (n : ℤ) ∣ (a * 43 * 62 * 1311 : ℤ)

theorem smallest_a_is_correct :
  (∀ a : ℕ, a > 0 → is_factor 112 a → is_factor 33 a → a ≥ smallest_a) ∧
  is_factor 112 smallest_a ∧
  is_factor 33 smallest_a :=
sorry

end NUMINAMATH_CALUDE_smallest_a_is_correct_l2793_279387


namespace NUMINAMATH_CALUDE_custom_operation_result_l2793_279322

def custom_operation (A B : Set ℕ) : Set ℕ :=
  {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

theorem custom_operation_result :
  let A : Set ℕ := {1, 2, 3, 4, 5}
  let B : Set ℕ := {4, 5, 6}
  custom_operation A B = {1, 2, 3, 6} := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_result_l2793_279322


namespace NUMINAMATH_CALUDE_ariel_fish_count_l2793_279393

theorem ariel_fish_count (total : ℕ) (male_fraction : ℚ) (female_count : ℕ) : 
  total = 45 → male_fraction = 2/3 → female_count = total - (total * male_fraction).num → female_count = 15 := by
  sorry

end NUMINAMATH_CALUDE_ariel_fish_count_l2793_279393


namespace NUMINAMATH_CALUDE_justice_ferns_l2793_279310

/-- Given the number of palms and succulents Justice has, the total number of plants she wants,
    and the number of additional plants she needs, prove that Justice has 3 ferns. -/
theorem justice_ferns (palms_and_succulents : ℕ) (desired_total : ℕ) (additional_needed : ℕ)
  (h1 : palms_and_succulents = 12)
  (h2 : desired_total = 24)
  (h3 : additional_needed = 9) :
  desired_total - additional_needed - palms_and_succulents = 3 :=
by sorry

end NUMINAMATH_CALUDE_justice_ferns_l2793_279310


namespace NUMINAMATH_CALUDE_range_of_a_for_M_subset_N_l2793_279388

/-- The set of real numbers m for which x^2 - x - m = 0 has solutions in (-1, 1) -/
def M : Set ℝ :=
  {m : ℝ | ∃ x, -1 < x ∧ x < 1 ∧ x^2 - x - m = 0}

/-- The solution set of (x - a)(x + a - 2) < 0 -/
def N (a : ℝ) : Set ℝ :=
  {x : ℝ | (x - a) * (x + a - 2) < 0}

/-- The theorem stating the range of a values for which M ⊆ N(a) -/
theorem range_of_a_for_M_subset_N :
  {a : ℝ | M ⊆ N a} = {a : ℝ | a < -1/4 ∨ a > 9/4} := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_M_subset_N_l2793_279388


namespace NUMINAMATH_CALUDE_school_gymnastics_ratio_l2793_279343

theorem school_gymnastics_ratio (total_students : ℕ) 
  (h_total : total_students = 120) : 
  ¬ ∃ (boys girls : ℕ), boys + girls = total_students ∧ 9 * girls = 2 * boys := by
  sorry

end NUMINAMATH_CALUDE_school_gymnastics_ratio_l2793_279343


namespace NUMINAMATH_CALUDE_M_properly_contains_N_l2793_279356

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 - 2*x > 0}
def N : Set ℝ := {x : ℝ | ∃ y, y = Real.log (x - 2)}

-- Theorem stating that M properly contains N
theorem M_properly_contains_N : M ⊃ N := by
  sorry

end NUMINAMATH_CALUDE_M_properly_contains_N_l2793_279356


namespace NUMINAMATH_CALUDE_sum_of_integers_l2793_279379

theorem sum_of_integers (x y : ℕ+) (h1 : x - y = 10) (h2 : x * y = 56) : x + y = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2793_279379


namespace NUMINAMATH_CALUDE_inequality_theorem_l2793_279315

theorem inequality_theorem (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  a / (a - c) > b / (b - c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2793_279315


namespace NUMINAMATH_CALUDE_sidney_adult_cats_l2793_279385

/-- Represents the number of adult cats Sidney has -/
def num_adult_cats : ℕ := sorry

/-- The number of kittens Sidney has -/
def num_kittens : ℕ := 4

/-- The number of cans of cat food Sidney has -/
def initial_cans : ℕ := 7

/-- The number of cans an adult cat eats per day -/
def adult_cat_consumption : ℚ := 1

/-- The number of cans a kitten eats per day -/
def kitten_consumption : ℚ := 3/4

/-- The number of additional cans Sidney needs to buy -/
def additional_cans : ℕ := 35

/-- The number of days Sidney needs to feed her cats -/
def days : ℕ := 7

theorem sidney_adult_cats : 
  num_adult_cats = 3 ∧
  (num_kittens : ℚ) * kitten_consumption * days + 
  (num_adult_cats : ℚ) * adult_cat_consumption * days = 
  (initial_cans : ℚ) + additional_cans :=
sorry

end NUMINAMATH_CALUDE_sidney_adult_cats_l2793_279385


namespace NUMINAMATH_CALUDE_composite_cube_three_diff_squares_l2793_279325

/-- A number is composite if it has a non-trivial factorization -/
def IsComposite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- The proposition that the cube of a composite number can be represented as the difference of two squares in at least three ways -/
theorem composite_cube_three_diff_squares (n : ℕ) (h : IsComposite n) : 
  ∃ (A₁ B₁ A₂ B₂ A₃ B₃ : ℕ), 
    (n^3 = A₁^2 - B₁^2) ∧ 
    (n^3 = A₂^2 - B₂^2) ∧ 
    (n^3 = A₃^2 - B₃^2) ∧ 
    (A₁, B₁) ≠ (A₂, B₂) ∧ 
    (A₁, B₁) ≠ (A₃, B₃) ∧ 
    (A₂, B₂) ≠ (A₃, B₃) :=
sorry

end NUMINAMATH_CALUDE_composite_cube_three_diff_squares_l2793_279325


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2793_279324

theorem sqrt_product_equality (a : ℝ) (h : a ≥ 0) : Real.sqrt (2 * a) * Real.sqrt (3 * a) = a * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2793_279324


namespace NUMINAMATH_CALUDE_triangle_side_length_l2793_279302

/-- Prove that in a triangle ABC with specific properties, the length of side a is 3√2 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  let f := λ x : ℝ => (Real.cos x, -1) • (Real.cos x + Real.sqrt 3 * Real.sin x, -3/2) - 2
  (f A = 1/2) →
  (2 * a = b + c) →
  (b * c / 2 = 9) →
  (a = 3 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2793_279302


namespace NUMINAMATH_CALUDE_cake_sector_angle_l2793_279341

theorem cake_sector_angle (total_sectors : ℕ) (probability : ℚ) : 
  total_sectors = 6 → probability = 1/8 → 
  ∃ (angle : ℚ), angle = 45 ∧ probability = angle / 360 := by
  sorry

end NUMINAMATH_CALUDE_cake_sector_angle_l2793_279341


namespace NUMINAMATH_CALUDE_cone_base_diameter_l2793_279378

/-- For a cone with surface area 3π and lateral surface that unfolds into a semicircle, 
    the diameter of its base is 2. -/
theorem cone_base_diameter (l r : ℝ) 
  (h1 : (1/2) * Real.pi * l^2 + Real.pi * r^2 = 3 * Real.pi) 
  (h2 : Real.pi * l = 2 * Real.pi * r) : 
  2 * r = 2 := by sorry

end NUMINAMATH_CALUDE_cone_base_diameter_l2793_279378


namespace NUMINAMATH_CALUDE_det_transformation_l2793_279389

/-- Given a 2x2 matrix with determinant -3, prove that a specific transformation of this matrix also has determinant -3 -/
theorem det_transformation (x y z w : ℝ) 
  (h : Matrix.det !![x, y; z, w] = -3) :
  Matrix.det !![x + 2*z, y + 2*w; z, w] = -3 := by
sorry

end NUMINAMATH_CALUDE_det_transformation_l2793_279389


namespace NUMINAMATH_CALUDE_polynomial_floor_property_l2793_279346

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The floor function -/
noncomputable def floor : ℝ → ℤ := sorry

/-- The property that P(⌊x⌋) = ⌊P(x)⌋ for all real x -/
def HasFloorProperty (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, P (↑(floor x)) = ↑(floor (P x))

/-- The main theorem -/
theorem polynomial_floor_property (P : RealPolynomial) :
  HasFloorProperty P → ∃ k : ℤ, ∀ x : ℝ, P x = x + k := by sorry

end NUMINAMATH_CALUDE_polynomial_floor_property_l2793_279346


namespace NUMINAMATH_CALUDE_point_on_line_m_range_l2793_279382

-- Define the function f
def f (x m n : ℝ) : ℝ := |x - m| + |x + n|

-- Part 1
theorem point_on_line (m n : ℝ) (h1 : m + n > 0) (h2 : ∀ x, f x m n ≥ 2) 
  (h3 : ∃ x, f x m n = 2) : m + n = 2 := by
  sorry

-- Part 2
theorem m_range (m : ℝ) (h : ∀ x ∈ Set.Icc 0 1, f x m 2 ≤ x + 5) : 
  m ∈ Set.Icc (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_m_range_l2793_279382


namespace NUMINAMATH_CALUDE_triangle_properties_l2793_279336

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Triangle inequality
  A + B + C = π ∧  -- Sum of angles in a triangle
  A < π/2 ∧ B < π/2 ∧ C < π/2 ∧  -- Acute triangle
  sqrt 3 * tan A * tan B - tan A - tan B = sqrt 3 ∧  -- Given condition
  c = 2 →  -- Given side length
  C = π/3 ∧ 20/3 < a^2 + b^2 ∧ a^2 + b^2 ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2793_279336


namespace NUMINAMATH_CALUDE_rational_numbers_countable_l2793_279363

theorem rational_numbers_countable : ∃ f : ℚ → ℕ+, Function.Bijective f := by
  sorry

end NUMINAMATH_CALUDE_rational_numbers_countable_l2793_279363


namespace NUMINAMATH_CALUDE_power_four_times_base_equals_power_five_l2793_279333

theorem power_four_times_base_equals_power_five (a : ℝ) : a^4 * a = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_four_times_base_equals_power_five_l2793_279333


namespace NUMINAMATH_CALUDE_farm_animal_ratio_l2793_279391

theorem farm_animal_ratio (cows sheep pigs : ℕ) : 
  cows = 12 →
  sheep = 2 * cows →
  cows + sheep + pigs = 108 →
  pigs / sheep = 3 := by
  sorry

end NUMINAMATH_CALUDE_farm_animal_ratio_l2793_279391


namespace NUMINAMATH_CALUDE_museum_trip_buses_l2793_279320

theorem museum_trip_buses : 
  ∀ (bus1 bus2 bus3 bus4 : ℕ),
  bus1 = 12 →
  bus2 = 2 * bus1 →
  bus3 = bus2 - 6 →
  bus4 = bus1 + 9 →
  bus1 + bus2 + bus3 + bus4 = 75 →
  4 = (if bus1 > 0 then 1 else 0) + 
      (if bus2 > 0 then 1 else 0) + 
      (if bus3 > 0 then 1 else 0) + 
      (if bus4 > 0 then 1 else 0) :=
by
  sorry

#check museum_trip_buses

end NUMINAMATH_CALUDE_museum_trip_buses_l2793_279320


namespace NUMINAMATH_CALUDE_oranges_taken_l2793_279347

theorem oranges_taken (initial : ℕ) (remaining : ℕ) (taken : ℕ) : 
  initial = 70 → remaining = 51 → taken = initial - remaining → taken = 19 := by
sorry

end NUMINAMATH_CALUDE_oranges_taken_l2793_279347


namespace NUMINAMATH_CALUDE_ratio_of_arithmetic_sequences_l2793_279399

def arithmetic_sequence_sum (a₁ : ℚ) (d : ℚ) (l : ℚ) : ℚ :=
  let n := (l - a₁) / d + 1
  n / 2 * (a₁ + l)

theorem ratio_of_arithmetic_sequences :
  let seq1_sum := arithmetic_sequence_sum 3 3 96
  let seq2_sum := arithmetic_sequence_sum 4 4 64
  seq1_sum / seq2_sum = 99 / 34 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_arithmetic_sequences_l2793_279399


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_implies_a_range_l2793_279380

-- Define the complex number z
def z (a : ℝ) : ℂ := (2 + a * Complex.I) * (a - Complex.I)

-- Define the condition for z to be in the third quadrant
def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- State the theorem
theorem z_in_third_quadrant_implies_a_range (a : ℝ) :
  in_third_quadrant (z a) → -Real.sqrt 2 < a ∧ a < 0 := by sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_implies_a_range_l2793_279380


namespace NUMINAMATH_CALUDE_local_minimum_of_f_l2793_279397

/-- The function f(x) = x³ - 4x² + 4x -/
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 4*x

/-- The local minimum value of f(x) is 0 -/
theorem local_minimum_of_f :
  ∃ (a : ℝ), ∀ (x : ℝ), ∃ (ε : ℝ), ε > 0 ∧ 
    (∀ (y : ℝ), |y - a| < ε → f y ≥ f a) ∧
    f a = 0 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_of_f_l2793_279397


namespace NUMINAMATH_CALUDE_count_equal_to_one_l2793_279311

theorem count_equal_to_one : 
  let numbers := [(-1)^2, (-1)^3, -(1^2), |(-1)|, -(-1), 1/(-1)]
  (numbers.filter (λ x => x = 1)).length = 3 := by
sorry

end NUMINAMATH_CALUDE_count_equal_to_one_l2793_279311


namespace NUMINAMATH_CALUDE_odd_function_interval_l2793_279354

/-- A function f is odd on an interval [a, b] if the interval is symmetric about the origin -/
def is_odd_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a = -b ∧ ∀ x ∈ Set.Icc a b, f (-x) = -f x

/-- The theorem stating that if f is an odd function on [t, t^2 - 3t - 3], then t = -1 -/
theorem odd_function_interval (f : ℝ → ℝ) (t : ℝ) :
  is_odd_on_interval f t (t^2 - 3*t - 3) → t = -1 := by
  sorry


end NUMINAMATH_CALUDE_odd_function_interval_l2793_279354


namespace NUMINAMATH_CALUDE_arithmetic_equalities_l2793_279328

theorem arithmetic_equalities : 
  (-(2^3) / 8 - 1/4 * (-2)^2 = -2) ∧ 
  ((-1/12 - 1/16 + 3/4 - 1/6) * (-48) = -21) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equalities_l2793_279328


namespace NUMINAMATH_CALUDE_x_range_l2793_279364

theorem x_range (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -2) : x > 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l2793_279364


namespace NUMINAMATH_CALUDE_goods_train_speed_l2793_279339

/-- The speed of a train given the speed of another train traveling in the opposite direction,
    the length of the first train, and the time it takes to pass an observer on the other train. -/
theorem goods_train_speed
  (speed_A : ℝ)
  (length_B : ℝ)
  (passing_time : ℝ)
  (h1 : speed_A = 100)  -- km/h
  (h2 : length_B = 0.28)  -- km (280 m converted to km)
  (h3 : passing_time = 9 / 3600)  -- hours (9 seconds converted to hours)
  : ∃ (speed_B : ℝ), speed_B = 12 := by
  sorry


end NUMINAMATH_CALUDE_goods_train_speed_l2793_279339


namespace NUMINAMATH_CALUDE_law_school_applicants_l2793_279396

theorem law_school_applicants (total : ℕ) (pol_sci : ℕ) (high_gpa : ℕ) (pol_sci_high_gpa : ℕ) 
  (h1 : total = 40)
  (h2 : pol_sci = 15)
  (h3 : high_gpa = 20)
  (h4 : pol_sci_high_gpa = 5) :
  total - pol_sci - high_gpa + pol_sci_high_gpa = 10 :=
by sorry

end NUMINAMATH_CALUDE_law_school_applicants_l2793_279396


namespace NUMINAMATH_CALUDE_min_sum_squares_l2793_279359

theorem min_sum_squares (a b c d e f g h : Int) : 
  a ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  b ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  c ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  d ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  e ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  f ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  g ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  h ∈ ({-6, -4, -1, 0, 3, 5, 7, 10} : Set Int) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
  f ≠ g ∧ f ≠ h ∧
  g ≠ h →
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 98 := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2793_279359


namespace NUMINAMATH_CALUDE_cubic_km_to_cubic_m_l2793_279357

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

/-- The number of cubic kilometers to convert -/
def cubic_km : ℝ := 5

/-- Theorem stating that 5 cubic kilometers is equal to 5,000,000,000 cubic meters -/
theorem cubic_km_to_cubic_m : 
  cubic_km * (km_to_m ^ 3) = 5000000000 := by
  sorry

end NUMINAMATH_CALUDE_cubic_km_to_cubic_m_l2793_279357


namespace NUMINAMATH_CALUDE_dividend_calculation_l2793_279371

/-- Calculates the dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (face_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ) :
  investment = 14400 ∧ 
  face_value = 100 ∧ 
  premium_rate = 0.20 ∧ 
  dividend_rate = 0.05 →
  (investment / (face_value * (1 + premium_rate))) * (face_value * dividend_rate) = 600 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2793_279371


namespace NUMINAMATH_CALUDE_age_difference_theorem_l2793_279304

/-- Represents a two-digit number --/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≤ 9
  h_ones : ones ≤ 9
  h_not_zero : tens ≠ 0

/-- The value of a two-digit number --/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

theorem age_difference_theorem (jack bill : TwoDigitNumber)
    (h_reversed : jack.tens = bill.ones ∧ jack.ones = bill.tens)
    (h_future : jack.value + 6 = 3 * (bill.value + 6)) :
    jack.value - bill.value = 36 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_theorem_l2793_279304


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l2793_279342

/-- The number of combinations of k items chosen from a set of n items. -/
def combinations (n k : ℕ) : ℕ :=
  if k ≤ n then
    Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  else
    0

/-- Theorem: The number of combinations of 3 toppings chosen from 7 available toppings is 35. -/
theorem pizza_toppings_combinations :
  combinations 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l2793_279342


namespace NUMINAMATH_CALUDE_cuboid_max_volume_l2793_279395

/-- The maximum volume of a cuboid with a total edge length of 60 units is 125 cubic units. -/
theorem cuboid_max_volume :
  ∀ x y z : ℝ,
  x > 0 → y > 0 → z > 0 →
  4 * (x + y + z) = 60 →
  x * y * z ≤ 125 :=
by sorry

end NUMINAMATH_CALUDE_cuboid_max_volume_l2793_279395


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2793_279392

theorem solution_set_of_inequality (x : ℝ) : 
  (x^2 - x < 0) ↔ (0 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2793_279392


namespace NUMINAMATH_CALUDE_scientific_notation_of_12400_l2793_279345

theorem scientific_notation_of_12400 :
  let num_athletes : ℕ := 12400
  1.24 * (10 : ℝ)^4 = num_athletes := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_12400_l2793_279345


namespace NUMINAMATH_CALUDE_watermelon_sales_l2793_279300

/-- Proves that the number of watermelons sold is 18, given the weight, price per pound, and total revenue -/
theorem watermelon_sales
  (weight : ℕ)
  (price_per_pound : ℕ)
  (total_revenue : ℕ)
  (h1 : weight = 23)
  (h2 : price_per_pound = 2)
  (h3 : total_revenue = 828) :
  total_revenue / (weight * price_per_pound) = 18 :=
by sorry

end NUMINAMATH_CALUDE_watermelon_sales_l2793_279300


namespace NUMINAMATH_CALUDE_border_area_l2793_279377

/-- The area of a border around a rectangular picture --/
theorem border_area (picture_height picture_width border_width : ℝ) : 
  picture_height = 12 →
  picture_width = 15 →
  border_width = 3 →
  (picture_height + 2 * border_width) * (picture_width + 2 * border_width) - picture_height * picture_width = 198 := by
  sorry

end NUMINAMATH_CALUDE_border_area_l2793_279377


namespace NUMINAMATH_CALUDE_max_min_triangle_area_l2793_279358

/-- A point on the 10x10 grid -/
structure GridPoint where
  x : Fin 11
  y : Fin 11

/-- The configuration of three pieces on the grid -/
structure Configuration where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- The area of a triangle formed by three points -/
def triangleArea (p1 p2 p3 : GridPoint) : ℚ :=
  sorry

/-- Check if two grid points are adjacent -/
def isAdjacent (p1 p2 : GridPoint) : Prop :=
  sorry

/-- A valid move between two configurations -/
def validMove (c1 c2 : Configuration) : Prop :=
  sorry

/-- A sequence of configurations representing a valid solution -/
def ValidSolution : Type :=
  sorry

/-- The minimum triangle area over all configurations in a solution -/
def minTriangleArea (sol : ValidSolution) : ℚ :=
  sorry

theorem max_min_triangle_area :
  (∃ (sol : ValidSolution), minTriangleArea sol = 5/2) ∧
  (∀ (sol : ValidSolution), minTriangleArea sol ≤ 5/2) := by
  sorry

end NUMINAMATH_CALUDE_max_min_triangle_area_l2793_279358


namespace NUMINAMATH_CALUDE_square_and_ln_exp_are_geometric_l2793_279349

/-- A function is geometric if it preserves geometric sequences -/
def IsGeometricFunction (f : ℝ → ℝ) : Prop :=
  ∀ (a : ℕ → ℝ), (∀ n : ℕ, a n ≠ 0) →
    (∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) →
    (∀ n : ℕ, f (a (n + 1)) / f (a n) = f (a (n + 2)) / f (a (n + 1)))

theorem square_and_ln_exp_are_geometric :
  IsGeometricFunction (fun x ↦ x^2) ∧
  IsGeometricFunction (fun x ↦ Real.log (2^x)) :=
sorry

end NUMINAMATH_CALUDE_square_and_ln_exp_are_geometric_l2793_279349


namespace NUMINAMATH_CALUDE_chess_player_win_loss_difference_l2793_279317

theorem chess_player_win_loss_difference
  (total_games : ℕ)
  (total_points : ℚ)
  (wins : ℕ)
  (draws : ℕ)
  (losses : ℕ)
  (h1 : total_games = 40)
  (h2 : total_points = 25)
  (h3 : wins + draws + losses = total_games)
  (h4 : wins + (1/2 : ℚ) * draws = total_points) :
  wins - losses = 10 := by
  sorry

end NUMINAMATH_CALUDE_chess_player_win_loss_difference_l2793_279317


namespace NUMINAMATH_CALUDE_turner_syndrome_classification_l2793_279307

-- Define the types of mutations
inductive MutationType
  | GeneMutation
  | ChromosomalNumberVariation
  | GeneRecombination
  | ChromosomalStructureVariation

-- Define a structure for chromosomes
structure Chromosome where
  isSexChromosome : Bool

-- Define a human genetic condition
structure GeneticCondition where
  name : String
  missingChromosome : Option Chromosome
  mutationType : MutationType

-- Define Turner syndrome
def TurnerSyndrome : GeneticCondition where
  name := "Turner syndrome"
  missingChromosome := some { isSexChromosome := true }
  mutationType := MutationType.ChromosomalNumberVariation

-- Theorem statement
theorem turner_syndrome_classification :
  TurnerSyndrome.mutationType = MutationType.ChromosomalNumberVariation :=
by
  sorry


end NUMINAMATH_CALUDE_turner_syndrome_classification_l2793_279307


namespace NUMINAMATH_CALUDE_min_value_problem_min_value_attained_l2793_279390

theorem min_value_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  (x^2 / (x + 2) + y^2 / (y + 1)) ≥ 1/4 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧
    (x^2 / (x + 2) + y^2 / (y + 1)) < 1/4 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_min_value_attained_l2793_279390


namespace NUMINAMATH_CALUDE_parabola_perpendicular_range_l2793_279303

/-- Given a parabola y² = x + 4 and points A(0,2), B(m² - 4, m), and C(x₀² - 4, x₀) where B and C are on the parabola and AB ⊥ BC, 
    the y-coordinate of C (x₀) satisfies: x₀ ≤ 2 - 2√2 or x₀ ≥ 2 + 2√2 -/
theorem parabola_perpendicular_range (m x₀ : ℝ) : 
  (m ^ 2 - 4 ≥ 0) →  -- B is on or above the x-axis
  (x₀ ^ 2 - 4 ≥ 0) →  -- C is on or above the x-axis
  ((m - 2) / (m ^ 2 - 4) * (x₀ - m) / (x₀ ^ 2 - m ^ 2) = -1) →  -- AB ⊥ BC
  (x₀ ≤ 2 - 2 * Real.sqrt 2 ∨ x₀ ≥ 2 + 2 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_parabola_perpendicular_range_l2793_279303


namespace NUMINAMATH_CALUDE_sequence_integer_count_l2793_279344

def sequence_term (n : ℕ) : ℚ :=
  9720 / 3^n

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem sequence_integer_count :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    ¬is_integer (sequence_term k)) →
  (∃! (k : ℕ), k = 6 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    ¬is_integer (sequence_term k)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_integer_count_l2793_279344


namespace NUMINAMATH_CALUDE_inequality_proof_l2793_279384

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  (1 / (1 - x^2)) + (1 / (1 - y^2)) ≥ 2 / (1 - x*y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2793_279384


namespace NUMINAMATH_CALUDE_cos_sqrt3_over_2_necessary_not_sufficient_l2793_279376

theorem cos_sqrt3_over_2_necessary_not_sufficient (α : ℝ) :
  (∃ k : ℤ, α = 2 * k * π + 5 * π / 6 → Real.cos α = -Real.sqrt 3 / 2) ∧
  (∃ α : ℝ, Real.cos α = -Real.sqrt 3 / 2 ∧ ∀ k : ℤ, α ≠ 2 * k * π + 5 * π / 6) :=
by sorry

end NUMINAMATH_CALUDE_cos_sqrt3_over_2_necessary_not_sufficient_l2793_279376


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l2793_279338

/-- The volume of a cube inscribed in a specific pyramid -/
theorem inscribed_cube_volume (base_side : ℝ) (h : base_side = 2) :
  let pyramid_height := 2 * Real.sqrt 3 / 3
  let cube_side := 2 * Real.sqrt 3 / 9
  let cube_volume := cube_side ^ 3
  cube_volume = 8 * Real.sqrt 3 / 243 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l2793_279338


namespace NUMINAMATH_CALUDE_jordan_read_more_than_maxime_l2793_279330

-- Define the number of novels read by each person
def jordan_french : ℕ := 130
def jordan_spanish : ℕ := 20
def alexandre_french : ℕ := jordan_french / 10
def alexandre_spanish : ℕ := 3 * jordan_spanish
def camille_french : ℕ := 2 * alexandre_french
def camille_spanish : ℕ := jordan_spanish / 2

-- Define the total number of French novels read by Jordan, Alexandre, and Camille
def total_french : ℕ := jordan_french + alexandre_french + camille_french

-- Define Maxime's French and Spanish novels
def maxime_french : ℕ := total_french / 2 - 5
def maxime_spanish : ℕ := 2 * camille_spanish

-- Define the total novels read by Jordan and Maxime
def jordan_total : ℕ := jordan_french + jordan_spanish
def maxime_total : ℕ := maxime_french + maxime_spanish

-- Theorem statement
theorem jordan_read_more_than_maxime :
  jordan_total = maxime_total + 51 := by sorry

end NUMINAMATH_CALUDE_jordan_read_more_than_maxime_l2793_279330


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2793_279362

theorem rationalize_denominator : 45 / Real.sqrt 45 = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2793_279362


namespace NUMINAMATH_CALUDE_ac_plus_one_lt_a_plus_c_l2793_279316

theorem ac_plus_one_lt_a_plus_c (a c : ℝ) (ha : 0 < a ∧ a < 1) (hc : c > 1) :
  a * c + 1 < a + c := by
  sorry

end NUMINAMATH_CALUDE_ac_plus_one_lt_a_plus_c_l2793_279316


namespace NUMINAMATH_CALUDE_complement_of_union_l2793_279355

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3, 4}
def N : Set Nat := {4, 5}

theorem complement_of_union :
  (U \ (M ∪ N)) = {1, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2793_279355


namespace NUMINAMATH_CALUDE_common_tangent_l2793_279352

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C2 (x y : ℝ) : Prop := (x-3)^2 + (y-4)^2 = 16

-- Define the line x = -1
def tangent_line (x : ℝ) : Prop := x = -1

-- Theorem statement
theorem common_tangent :
  (∀ x y : ℝ, C1 x y → tangent_line x → (x^2 + y^2 = 1 ∧ x = -1)) ∧
  (∀ x y : ℝ, C2 x y → tangent_line x → ((x-3)^2 + (y-4)^2 = 16 ∧ x = -1)) :=
sorry

end NUMINAMATH_CALUDE_common_tangent_l2793_279352


namespace NUMINAMATH_CALUDE_problem_1_l2793_279334

theorem problem_1 : 23 * (-5) - (-3) / (3/108) = -7 := by sorry

end NUMINAMATH_CALUDE_problem_1_l2793_279334


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l2793_279398

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle --/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents a T-shaped figure formed by two rectangles --/
structure TShape where
  vertical : Rectangle
  horizontal : Rectangle

/-- Calculates the perimeter of a T-shaped figure --/
def TShape.perimeter (t : TShape) : ℝ :=
  t.vertical.perimeter + t.horizontal.perimeter - 4 * t.horizontal.width

theorem t_shape_perimeter :
  let t : TShape := {
    vertical := { width := 2, height := 6 },
    horizontal := { width := 2, height := 4 }
  }
  t.perimeter = 24 := by sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_l2793_279398


namespace NUMINAMATH_CALUDE_find_B_l2793_279394

/-- The number represented by 2B8, where B is a single digit -/
def number (B : ℕ) : ℕ := 200 + 10 * B + 8

/-- The sum of digits of the number 2B8 -/
def digit_sum (B : ℕ) : ℕ := 2 + B + 8

theorem find_B : ∃ B : ℕ, B < 10 ∧ number B % 3 = 0 ∧ B = 2 :=
sorry

end NUMINAMATH_CALUDE_find_B_l2793_279394


namespace NUMINAMATH_CALUDE_teagan_total_payment_l2793_279365

def original_shirt_price : ℚ := 60
def original_jacket_price : ℚ := 90
def price_reduction : ℚ := 20 / 100
def num_shirts : ℕ := 5
def num_jackets : ℕ := 10

def reduced_price (original_price : ℚ) : ℚ :=
  original_price * (1 - price_reduction)

def total_cost (item_price : ℚ) (quantity : ℕ) : ℚ :=
  item_price * quantity

theorem teagan_total_payment :
  total_cost (reduced_price original_shirt_price) num_shirts +
  total_cost (reduced_price original_jacket_price) num_jackets = 960 := by
  sorry

end NUMINAMATH_CALUDE_teagan_total_payment_l2793_279365


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2793_279374

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hbc : b + c ≥ a) :
  b / c + c / (a + b) ≥ Real.sqrt 2 - 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2793_279374
