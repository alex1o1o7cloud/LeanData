import Mathlib

namespace NUMINAMATH_CALUDE_w_squared_values_l1606_160613

theorem w_squared_values (w : ℝ) :
  (2 * w + 17)^2 = (4 * w + 9) * (3 * w + 6) →
  w^2 = 19.69140625 ∨ w^2 = 43.06640625 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_values_l1606_160613


namespace NUMINAMATH_CALUDE_inequality_proof_l1606_160684

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_abc : a * b + b * c + c * a = a * b * c) :
  (a^a * (b^2 + c^2)) / ((a^a - 1)^2) +
  (b^b * (c^2 + a^2)) / ((b^b - 1)^2) +
  (c^c * (a^2 + b^2)) / ((c^c - 1)^2) ≥
  18 * ((a + b + c) / (a * b * c - 1))^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1606_160684


namespace NUMINAMATH_CALUDE_calculate_brokerage_percentage_l1606_160683

/-- Calculate the brokerage percentage for a stock investment --/
theorem calculate_brokerage_percentage
  (stock_rate : ℝ)
  (income : ℝ)
  (investment : ℝ)
  (market_value : ℝ)
  (h1 : stock_rate = 10.5)
  (h2 : income = 756)
  (h3 : investment = 8000)
  (h4 : market_value = 110.86111111111111)
  : ∃ (brokerage_percentage : ℝ),
    brokerage_percentage = 0.225 ∧
    brokerage_percentage = (investment - (income * 100 / stock_rate) * market_value / 100) / investment * 100 :=
by sorry

end NUMINAMATH_CALUDE_calculate_brokerage_percentage_l1606_160683


namespace NUMINAMATH_CALUDE_zoo_guide_problem_l1606_160612

/-- Given two zoo guides speaking to groups of children, where one guide spoke to 25 children
    and the total number of children spoken to is 44, prove that the number of children
    the other guide spoke to is 19. -/
theorem zoo_guide_problem (total_children : ℕ) (second_guide_children : ℕ) :
  total_children = 44 →
  second_guide_children = 25 →
  total_children - second_guide_children = 19 :=
by sorry

end NUMINAMATH_CALUDE_zoo_guide_problem_l1606_160612


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l1606_160690

theorem largest_integer_inequality : 
  ∀ x : ℤ, x ≤ 10 ↔ (x : ℚ) / 4 + 5 / 6 < 7 / 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_inequality_l1606_160690


namespace NUMINAMATH_CALUDE_loot_box_loss_l1606_160655

/-- Calculates the average amount lost when buying loot boxes --/
theorem loot_box_loss (loot_box_cost : ℝ) (average_item_value : ℝ) (total_spent : ℝ) :
  loot_box_cost = 5 →
  average_item_value = 3.5 →
  total_spent = 40 →
  total_spent - (total_spent / loot_box_cost * average_item_value) = 12 := by
  sorry

#check loot_box_loss

end NUMINAMATH_CALUDE_loot_box_loss_l1606_160655


namespace NUMINAMATH_CALUDE_projection_shape_theorem_l1606_160643

/-- Represents a plane in 3D space -/
structure Plane

/-- Represents a point in 3D space -/
structure Point

/-- Represents a triangle in 3D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents the projection of a point onto a plane -/
def project (p : Point) (plane : Plane) : Point :=
  sorry

/-- Determines if a point is outside a plane -/
def isOutside (p : Point) (plane : Plane) : Prop :=
  sorry

/-- Determines if a point is on a plane -/
def isOn (p : Point) (plane : Plane) : Prop :=
  sorry

/-- Determines if a triangle is right-angled -/
def isRightTriangle (t : Triangle) : Prop :=
  sorry

/-- Represents the shape formed by projections -/
inductive ProjectionShape
  | LineSegment
  | ObtuseTriangle

theorem projection_shape_theorem (ABC : Triangle) (a : Plane) :
  isRightTriangle ABC →
  isOn ABC.B a →
  isOn ABC.C a →
  isOutside ABC.A a →
  (project ABC.A a ≠ ABC.B ∧ project ABC.A a ≠ ABC.C) →
  (∃ shape : ProjectionShape, 
    (shape = ProjectionShape.LineSegment ∨ shape = ProjectionShape.ObtuseTriangle)) :=
  sorry

end NUMINAMATH_CALUDE_projection_shape_theorem_l1606_160643


namespace NUMINAMATH_CALUDE_distribution_schemes_count_l1606_160662

/-- The number of ways to distribute 3 people to 7 communities with at most 2 people per community -/
def distribution_schemes : ℕ := sorry

/-- The number of ways to choose 3 communities out of 7 -/
def three_single_communities : ℕ := sorry

/-- The number of ways to choose 2 communities out of 7 and distribute 3 people -/
def one_double_one_single : ℕ := sorry

theorem distribution_schemes_count :
  distribution_schemes = three_single_communities + one_double_one_single ∧
  distribution_schemes = 336 := by sorry

end NUMINAMATH_CALUDE_distribution_schemes_count_l1606_160662


namespace NUMINAMATH_CALUDE_parabola_point_x_coordinate_l1606_160623

/-- The x-coordinate of a point on a parabola with a given distance from its focus -/
theorem parabola_point_x_coordinate (x y : ℝ) :
  y^2 = 4*x →  -- Point (x, y) is on the parabola y² = 4x
  (x - 1)^2 + y^2 = 4^2 →  -- Distance from (x, y) to focus (1, 0) is 4
  x = 3 := by sorry

end NUMINAMATH_CALUDE_parabola_point_x_coordinate_l1606_160623


namespace NUMINAMATH_CALUDE_two_number_difference_l1606_160696

theorem two_number_difference (x y : ℝ) 
  (sum_eq : x + y = 40)
  (triple_minus_quad : 3 * y - 4 * x = 20) :
  |y - x| = 100 / 7 := by
sorry

end NUMINAMATH_CALUDE_two_number_difference_l1606_160696


namespace NUMINAMATH_CALUDE_rotation_equivalence_l1606_160657

/-- 
Given that:
1. A point A is rotated 550 degrees clockwise about a center point B to reach point C.
2. The same point A is rotated x degrees counterclockwise about the same center point B to reach point C.
3. x is less than 360 degrees.

Prove that x equals 170 degrees.
-/
theorem rotation_equivalence (x : ℝ) 
  (h1 : x < 360) 
  (h2 : (550 % 360 : ℝ) + x = 360) : x = 170 :=
by sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l1606_160657


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_nine_between_90_and_200_l1606_160624

theorem unique_square_divisible_by_nine_between_90_and_200 :
  ∃! y : ℕ, 
    90 < y ∧ 
    y < 200 ∧ 
    ∃ n : ℕ, y = n^2 ∧ 
    ∃ k : ℕ, y = 9 * k :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_nine_between_90_and_200_l1606_160624


namespace NUMINAMATH_CALUDE_hexagon_diagonals_l1606_160606

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- The number of diagonals in a hexagon is 9 -/
theorem hexagon_diagonals : num_diagonals hexagon_sides = 9 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_l1606_160606


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l1606_160691

theorem quadratic_equivalence : ∀ x : ℝ, x^2 - 8*x - 1 = 0 ↔ (x - 4)^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l1606_160691


namespace NUMINAMATH_CALUDE_probability_three_same_color_l1606_160692

def total_balls : ℕ := 27
def green_balls : ℕ := 15
def white_balls : ℕ := 12

def probability_same_color : ℚ := 3 / 13

theorem probability_three_same_color :
  let total_combinations := Nat.choose total_balls 3
  let green_combinations := Nat.choose green_balls 3
  let white_combinations := Nat.choose white_balls 3
  (green_combinations + white_combinations : ℚ) / total_combinations = probability_same_color :=
by sorry

end NUMINAMATH_CALUDE_probability_three_same_color_l1606_160692


namespace NUMINAMATH_CALUDE_dice_probabilities_l1606_160693

-- Define the sample space and events
def Ω : ℕ := 216  -- Total number of possible outcomes
def A : ℕ := 120  -- Number of outcomes where all dice show different numbers
def AB : ℕ := 75  -- Number of outcomes satisfying both A and B

-- Define the probabilities
def P_AB : ℚ := AB / Ω
def P_A : ℚ := A / Ω
def P_B_given_A : ℚ := P_AB / P_A

-- State the theorem
theorem dice_probabilities :
  P_AB = 75 / 216 ∧ P_B_given_A = 5 / 8 := by
  sorry


end NUMINAMATH_CALUDE_dice_probabilities_l1606_160693


namespace NUMINAMATH_CALUDE_greatest_fraction_with_same_digit_sum_l1606_160601

/-- A function that returns the sum of digits of a number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Predicate to check if a number is a four-digit number -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem greatest_fraction_with_same_digit_sum :
  ∀ m n : ℕ, isFourDigit m → isFourDigit n → sumOfDigits m = sumOfDigits n →
  (m : ℚ) / n ≤ 9900 / 1089 :=
sorry

end NUMINAMATH_CALUDE_greatest_fraction_with_same_digit_sum_l1606_160601


namespace NUMINAMATH_CALUDE_snacks_sold_l1606_160626

/-- Given the initial number of snacks and ramens in a market, and the final total after some transactions, 
    prove that the number of snacks sold is 599. -/
theorem snacks_sold (initial_snacks : ℕ) (initial_ramens : ℕ) (ramens_bought : ℕ) (final_total : ℕ) :
  initial_snacks = 1238 →
  initial_ramens = initial_snacks + 374 →
  ramens_bought = 276 →
  final_total = 2527 →
  (initial_snacks - (initial_snacks - (initial_ramens + ramens_bought - final_total))) = 599 := by
  sorry

end NUMINAMATH_CALUDE_snacks_sold_l1606_160626


namespace NUMINAMATH_CALUDE_dice_cube_surface_area_l1606_160663

theorem dice_cube_surface_area (num_dice : ℕ) (die_side_length : ℝ) (h1 : num_dice = 27) (h2 : die_side_length = 3) :
  let edge_length : ℝ := die_side_length * (num_dice ^ (1/3 : ℝ))
  let face_area : ℝ := edge_length ^ 2
  let surface_area : ℝ := 6 * face_area
  surface_area = 486 :=
by sorry

end NUMINAMATH_CALUDE_dice_cube_surface_area_l1606_160663


namespace NUMINAMATH_CALUDE_sum_of_even_indexed_coefficients_l1606_160644

theorem sum_of_even_indexed_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (2*x + 3)^8 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                      a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8) →
  a + a₂ + a₄ + a₆ + a₈ = 3281 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_indexed_coefficients_l1606_160644


namespace NUMINAMATH_CALUDE_train_length_l1606_160675

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 27 → time = 16 → speed * time * (5 / 18) = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1606_160675


namespace NUMINAMATH_CALUDE_f_negative_implies_a_range_l1606_160617

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * (x - 1)

theorem f_negative_implies_a_range (a : ℝ) :
  (∀ x > 1, f a x < 0) → a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_f_negative_implies_a_range_l1606_160617


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_max_consecutive_integers_sum_500_thirty_one_is_max_l1606_160607

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 31 ↔ n * (n + 1) ≤ 1000 := by sorry

theorem max_consecutive_integers_sum_500 : 
  ∀ k > 31, k * (k + 1) / 2 > 500 := by sorry

theorem thirty_one_is_max : 
  31 * 32 / 2 ≤ 500 ∧ ∀ n > 31, n * (n + 1) / 2 > 500 := by sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_max_consecutive_integers_sum_500_thirty_one_is_max_l1606_160607


namespace NUMINAMATH_CALUDE_range_of_a_l1606_160670

theorem range_of_a (a : ℝ) : 
  (a + 1)^(-1/2 : ℝ) < (3 - 2*a)^(-1/2 : ℝ) → 2/3 < a ∧ a < 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1606_160670


namespace NUMINAMATH_CALUDE_HN_passes_through_fixed_point_l1606_160633

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 / 4 = 1

-- Define the line segment AB
def line_AB (x y : ℝ) : Prop := y = 2/3 * x - 2

-- Define a point on the ellipse
def on_ellipse (p : ℝ × ℝ) : Prop := ellipse p.1 p.2

-- Define a point on line AB
def on_line_AB (p : ℝ × ℝ) : Prop := line_AB p.1 p.2

-- Define the property of T being on the line parallel to x-axis through M
def T_on_parallel_line (M T : ℝ × ℝ) : Prop := M.2 = T.2

-- Define the property of H satisfying MT = TH
def H_satisfies_MT_eq_TH (M T H : ℝ × ℝ) : Prop := 
  (H.1 - T.1 = T.1 - M.1) ∧ (H.2 - T.2 = T.2 - M.2)

-- Main theorem
theorem HN_passes_through_fixed_point :
  ∀ (M N T H : ℝ × ℝ),
  on_ellipse M → on_ellipse N →
  on_line_AB T →
  T_on_parallel_line M T →
  H_satisfies_MT_eq_TH M T H →
  ∃ (t : ℝ), (1 - t) * H.1 + t * N.1 = 0 ∧ (1 - t) * H.2 + t * N.2 = -2 :=
sorry

end NUMINAMATH_CALUDE_HN_passes_through_fixed_point_l1606_160633


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1606_160619

theorem cube_volume_from_surface_area :
  ∀ s V : ℝ,
  (6 * s^2 = 864) →  -- Surface area condition
  (V = s^3) →        -- Volume definition
  V = 1728 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1606_160619


namespace NUMINAMATH_CALUDE_tournament_distributions_l1606_160653

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 6

/-- Represents the number of games in the tournament -/
def num_games : ℕ := 5

/-- Represents the number of possible outcomes for each game -/
def outcomes_per_game : ℕ := 2

/-- Calculates the total number of possible prize distributions -/
def total_distributions : ℕ := outcomes_per_game ^ num_games

theorem tournament_distributions :
  total_distributions = 32 :=
sorry

end NUMINAMATH_CALUDE_tournament_distributions_l1606_160653


namespace NUMINAMATH_CALUDE_mission_duration_l1606_160610

theorem mission_duration (planned_duration : ℝ) (overtime_percentage : ℝ) (second_mission_duration : ℝ) : 
  planned_duration = 5 ∧ 
  overtime_percentage = 0.6 ∧ 
  second_mission_duration = 3 → 
  planned_duration * (1 + overtime_percentage) + second_mission_duration = 11 :=
by sorry

end NUMINAMATH_CALUDE_mission_duration_l1606_160610


namespace NUMINAMATH_CALUDE_cheryl_strawberries_l1606_160642

theorem cheryl_strawberries (num_buckets : ℕ) (removed_per_bucket : ℕ) (remaining_per_bucket : ℕ) : 
  num_buckets = 5 →
  removed_per_bucket = 20 →
  remaining_per_bucket = 40 →
  num_buckets * (removed_per_bucket + remaining_per_bucket) = 300 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_strawberries_l1606_160642


namespace NUMINAMATH_CALUDE_total_sleep_l1606_160622

def sleep_pattern (first_night : ℕ) : Fin 4 → ℕ
| 0 => first_night
| 1 => 2 * first_night
| 2 => 2 * first_night - 3
| 3 => 3 * (2 * first_night - 3)

theorem total_sleep (first_night : ℕ) (h : first_night = 6) : 
  (Finset.sum Finset.univ (sleep_pattern first_night)) = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_sleep_l1606_160622


namespace NUMINAMATH_CALUDE_max_value_f_l1606_160688

/-- Given positive real numbers x, y, z satisfying xyz = 1, 
    the maximum value of f(x, y, z) = (1 - yz + z)(1 - zx + x)(1 - xy + y) is 1, 
    and this maximum is achieved when x = y = z = 1. -/
theorem max_value_f (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  let f := fun (a b c : ℝ) => (1 - b*c + c) * (1 - c*a + a) * (1 - a*b + b)
  (∀ a b c, a > 0 → b > 0 → c > 0 → a * b * c = 1 → f a b c ≤ 1) ∧ 
  f x y z ≤ 1 ∧
  f 1 1 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_f_l1606_160688


namespace NUMINAMATH_CALUDE_P_symmetric_l1606_160676

variable (x y z : ℝ)

noncomputable def P : ℕ → ℝ → ℝ → ℝ → ℝ
| 0, _, _, _ => 1
| (m + 1), x, y, z => (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

theorem P_symmetric (m : ℕ) : 
  P m x y z = P m y x z ∧ 
  P m x y z = P m x z y ∧ 
  P m x y z = P m z y x :=
by sorry

end NUMINAMATH_CALUDE_P_symmetric_l1606_160676


namespace NUMINAMATH_CALUDE_equation_simplification_l1606_160679

theorem equation_simplification (x y z : ℕ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  9 * x - (10 / (2 * y) / 3 + 7 * z) * Real.pi = 9 * x - (5 * Real.pi / (3 * y)) - (7 * Real.pi * z) :=
by sorry

end NUMINAMATH_CALUDE_equation_simplification_l1606_160679


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1606_160673

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                                    a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 510 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1606_160673


namespace NUMINAMATH_CALUDE_function_properties_l1606_160665

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + a

theorem function_properties (a : ℝ) :
  (∃ x, ∀ y, f a y ≤ f a x) ∧ (∃ x, f a x = 6) →
  a = 6 ∧
  ∀ k, (∀ x t, x ∈ [-2, 2] → t ∈ [-1, 1] → f a x ≥ k * t - 25) ↔ k ∈ [-3, 3] :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1606_160665


namespace NUMINAMATH_CALUDE_parallel_lines_parallelograms_l1606_160603

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of parallelograms formed by intersecting parallel lines -/
def parallelograms_count (set1 : ℕ) (set2 : ℕ) : ℕ :=
  choose_two set1 * choose_two set2

theorem parallel_lines_parallelograms :
  parallelograms_count 3 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_parallelograms_l1606_160603


namespace NUMINAMATH_CALUDE_completing_square_proof_l1606_160660

theorem completing_square_proof (x : ℝ) : 
  (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_proof_l1606_160660


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_squared_l1606_160630

/-- Represents a parallelogram ABCD with specific properties -/
structure Parallelogram where
  -- Area of the parallelogram
  area : ℝ
  -- Length of PQ (projections of A and C onto BD)
  pq : ℝ
  -- Length of RS (projections of B and D onto AC)
  rs : ℝ
  -- Ensures area is positive
  area_pos : area > 0
  -- Ensures PQ is positive
  pq_pos : pq > 0
  -- Ensures RS is positive
  rs_pos : rs > 0

/-- The main theorem about the longer diagonal of the parallelogram -/
theorem parallelogram_diagonal_squared
  (abcd : Parallelogram)
  (h1 : abcd.area = 24)
  (h2 : abcd.pq = 8)
  (h3 : abcd.rs = 10) :
  ∃ (d : ℝ), d > 0 ∧ d^2 = 62 + 20 * Real.sqrt 61 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_squared_l1606_160630


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l1606_160640

theorem smallest_fraction_between (p q : ℕ+) : 
  (4 : ℚ) / 11 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (3 : ℚ) / 8 ∧ 
  (∀ p' q' : ℕ+, (4 : ℚ) / 11 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (3 : ℚ) / 8 → q ≤ q') →
  q - p = 12 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l1606_160640


namespace NUMINAMATH_CALUDE_cube_projection_sum_squares_zero_l1606_160649

/-- Represents a vertex of a cube -/
structure CubeVertex where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the orthogonal projection of a cube vertex onto a complex plane -/
def project (v : CubeVertex) : ℂ :=
  Complex.mk v.x v.y

/-- Given four vertices of a cube where three are adjacent to the fourth,
    and their orthogonal projections onto a complex plane,
    the sum of the squares of the projected complex numbers is zero. -/
theorem cube_projection_sum_squares_zero
  (V V₁ V₂ V₃ : CubeVertex)
  (adj₁ : V₁.x = V.x ∨ V₁.y = V.y ∨ V₁.z = V.z)
  (adj₂ : V₂.x = V.x ∨ V₂.y = V.y ∨ V₂.z = V.z)
  (adj₃ : V₃.x = V.x ∨ V₃.y = V.y ∨ V₃.z = V.z)
  (origin_proj : project V = 0)
  : (project V₁)^2 + (project V₂)^2 + (project V₃)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_projection_sum_squares_zero_l1606_160649


namespace NUMINAMATH_CALUDE_sales_difference_is_48_l1606_160656

/-- Represents the baker's sales data --/
structure BakerSales where
  usualPastries : ℕ
  usualBread : ℕ
  todayPastries : ℕ
  todayBread : ℕ
  pastryPrice : ℕ
  breadPrice : ℕ

/-- Calculates the difference between today's sales and the daily average sales --/
def salesDifference (sales : BakerSales) : ℕ :=
  let usualTotal := sales.usualPastries * sales.pastryPrice + sales.usualBread * sales.breadPrice
  let todayTotal := sales.todayPastries * sales.pastryPrice + sales.todayBread * sales.breadPrice
  todayTotal - usualTotal

/-- Theorem stating the difference in sales --/
theorem sales_difference_is_48 :
  ∃ (sales : BakerSales),
    sales.usualPastries = 20 ∧
    sales.usualBread = 10 ∧
    sales.todayPastries = 14 ∧
    sales.todayBread = 25 ∧
    sales.pastryPrice = 2 ∧
    sales.breadPrice = 4 ∧
    salesDifference sales = 48 := by
  sorry

end NUMINAMATH_CALUDE_sales_difference_is_48_l1606_160656


namespace NUMINAMATH_CALUDE_jeromes_contact_list_ratio_l1606_160627

/-- Proves that the ratio of out of school friends to classmates is 1:2 given the conditions in Jerome's contact list problem -/
theorem jeromes_contact_list_ratio : 
  ∀ (out_of_school_friends : ℕ),
    20 + out_of_school_friends + 2 + 1 = 33 →
    out_of_school_friends = 10 ∧ 
    (out_of_school_friends : ℚ) / 20 = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_jeromes_contact_list_ratio_l1606_160627


namespace NUMINAMATH_CALUDE_weeks_to_save_dress_l1606_160637

def original_price : ℚ := 150
def discount_rate : ℚ := 15 / 100
def initial_savings : ℚ := 35
def odd_week_allowance : ℚ := 30
def even_week_allowance : ℚ := 35
def weekly_arcade_expense : ℚ := 20
def weekly_snack_expense : ℚ := 10

def discounted_price : ℚ := original_price * (1 - discount_rate)
def amount_to_save : ℚ := discounted_price - initial_savings
def biweekly_allowance : ℚ := odd_week_allowance + even_week_allowance
def weekly_expenses : ℚ := weekly_arcade_expense + weekly_snack_expense
def biweekly_savings : ℚ := biweekly_allowance - 2 * weekly_expenses
def average_weekly_savings : ℚ := biweekly_savings / 2

theorem weeks_to_save_dress : 
  ⌈amount_to_save / average_weekly_savings⌉ = 37 := by sorry

end NUMINAMATH_CALUDE_weeks_to_save_dress_l1606_160637


namespace NUMINAMATH_CALUDE_car_profit_percentage_l1606_160674

/-- Calculates the profit percentage on the original price of a car, given specific buying and selling conditions. -/
theorem car_profit_percentage (P : ℝ) (h : P > 0) : 
  let purchase_price := 0.95 * P
  let taxes := 0.03 * P
  let maintenance := 0.02 * P
  let total_cost := purchase_price + taxes + maintenance
  let selling_price := purchase_price * 1.6
  let profit := selling_price - total_cost
  let profit_percentage := (profit / P) * 100
  profit_percentage = 52 := by sorry

end NUMINAMATH_CALUDE_car_profit_percentage_l1606_160674


namespace NUMINAMATH_CALUDE_stadium_entrance_count_l1606_160616

/-- The number of placards initially in the basket -/
def initial_placards : ℕ := 5682

/-- The number of placards each person takes -/
def placards_per_person : ℕ := 2

/-- The number of people who entered the stadium -/
def people_entered : ℕ := initial_placards / placards_per_person

theorem stadium_entrance_count :
  people_entered = 2841 :=
sorry

end NUMINAMATH_CALUDE_stadium_entrance_count_l1606_160616


namespace NUMINAMATH_CALUDE_total_sides_of_dice_l1606_160686

/-- The number of dice each person brought -/
def dice_per_person : ℕ := 4

/-- The number of sides on each die -/
def sides_per_die : ℕ := 6

/-- The number of people who brought dice -/
def number_of_people : ℕ := 2

/-- Theorem: The total number of sides on all dice brought by two people, 
    each bringing 4 six-sided dice, is 48. -/
theorem total_sides_of_dice : 
  number_of_people * dice_per_person * sides_per_die = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_sides_of_dice_l1606_160686


namespace NUMINAMATH_CALUDE_dog_food_theorem_l1606_160698

/-- The amount of food eaten by Hannah's three dogs -/
def dog_food_problem (first_dog_food second_dog_food third_dog_food : ℝ) : Prop :=
  -- Hannah has three dogs
  -- The first dog eats 1.5 cups of dog food a day
  first_dog_food = 1.5 ∧
  -- The second dog eats twice as much as the first dog
  second_dog_food = 2 * first_dog_food ∧
  -- Hannah prepares 10 cups of dog food in total for her three dogs
  first_dog_food + second_dog_food + third_dog_food = 10 ∧
  -- The difference between the third dog's food and the second dog's food is 2.5 cups
  third_dog_food - second_dog_food = 2.5

theorem dog_food_theorem :
  ∃ (first_dog_food second_dog_food third_dog_food : ℝ),
    dog_food_problem first_dog_food second_dog_food third_dog_food :=
by
  sorry

end NUMINAMATH_CALUDE_dog_food_theorem_l1606_160698


namespace NUMINAMATH_CALUDE_park_outer_diameter_l1606_160681

/-- Represents the diameter of the outer boundary of a circular park with concentric sections -/
def outer_diameter (fountain_diameter : ℝ) (garden_width : ℝ) (path_width : ℝ) : ℝ :=
  fountain_diameter + 2 * (garden_width + path_width)

/-- Theorem stating the diameter of the outer boundary of the jogging path -/
theorem park_outer_diameter :
  outer_diameter 14 12 10 = 58 := by
  sorry

end NUMINAMATH_CALUDE_park_outer_diameter_l1606_160681


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1606_160618

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a = 1/4 → ∀ x : ℝ, x > 0 → x + a/x ≥ 1) ∧
  (∃ a : ℝ, a > 1/4 ∧ ∀ x : ℝ, x > 0 → x + a/x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1606_160618


namespace NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l1606_160666

/-- Prove that the polar equation ρ²cos(2θ) = 16 is equivalent to the Cartesian equation x² - y² = 16 -/
theorem polar_to_cartesian_equivalence (ρ θ x y : ℝ) 
  (h1 : x = ρ * Real.cos θ) 
  (h2 : y = ρ * Real.sin θ) : 
  ρ^2 * Real.cos (2 * θ) = 16 ↔ x^2 - y^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l1606_160666


namespace NUMINAMATH_CALUDE_average_time_per_mile_l1606_160667

/-- Proves that the average time per mile is 9 minutes for a 24-mile run completed in 3 hours and 36 minutes -/
theorem average_time_per_mile (distance : ℕ) (hours : ℕ) (minutes : ℕ) : 
  distance = 24 ∧ hours = 3 ∧ minutes = 36 → 
  (hours * 60 + minutes) / distance = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_time_per_mile_l1606_160667


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1606_160650

theorem inequality_solution_set (x : ℝ) :
  (1 / x < 1 / 2) ↔ x ∈ (Set.Ioi 2 ∪ Set.Iio 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1606_160650


namespace NUMINAMATH_CALUDE_at_least_one_equation_has_two_roots_l1606_160658

theorem at_least_one_equation_has_two_roots (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (4 * b^2 - 4 * a * c > 0) ∨ 
  (4 * c^2 - 4 * a * b > 0) ∨ 
  (4 * a^2 - 4 * b * c > 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_equation_has_two_roots_l1606_160658


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l1606_160628

theorem rectangle_shorter_side 
  (area : ℝ) 
  (perimeter : ℝ) 
  (h_area : area = 91) 
  (h_perimeter : perimeter = 40) : 
  ∃ (length width : ℝ), 
    length * width = area ∧ 
    2 * (length + width) = perimeter ∧ 
    width = 7 ∧ 
    width ≤ length := by
  sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l1606_160628


namespace NUMINAMATH_CALUDE_blue_balls_count_l1606_160651

/-- The number of boxes a person has -/
def num_boxes : ℕ := 2

/-- The number of blue balls in each box -/
def blue_balls_per_box : ℕ := 5

/-- The total number of blue balls a person has -/
def total_blue_balls : ℕ := num_boxes * blue_balls_per_box

theorem blue_balls_count : total_blue_balls = 10 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_count_l1606_160651


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1606_160654

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) : 
  let angle := angle_between_vectors a b
  let a_magnitude := Real.sqrt (a.1^2 + a.2^2)
  let b_magnitude := Real.sqrt (b.1^2 + b.2^2)
  angle = π/3 ∧ a = (2, 0) ∧ b_magnitude = 1 →
  Real.sqrt ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1606_160654


namespace NUMINAMATH_CALUDE_choose_two_from_seven_eq_twentyone_l1606_160671

/-- The number of ways to choose 2 people from 7 -/
def choose_two_from_seven : ℕ := Nat.choose 7 2

/-- Theorem stating that choosing 2 from 7 results in 21 possibilities -/
theorem choose_two_from_seven_eq_twentyone : choose_two_from_seven = 21 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_seven_eq_twentyone_l1606_160671


namespace NUMINAMATH_CALUDE_square_sum_problem_l1606_160646

theorem square_sum_problem (a b : ℝ) (h1 : a * b = 30) (h2 : a + b = 10) :
  a^2 + b^2 = 40 := by sorry

end NUMINAMATH_CALUDE_square_sum_problem_l1606_160646


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l1606_160615

def geometric_sequence (a : ℕ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem sixth_term_of_geometric_sequence 
  (a₁ : ℕ) (a₅ : ℕ) (h₁ : a₁ = 3) (h₅ : a₅ = 375) :
  ∃ r : ℝ, 
    geometric_sequence a₁ r 5 = a₅ ∧ 
    geometric_sequence a₁ r 6 = 9375 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l1606_160615


namespace NUMINAMATH_CALUDE_expression_evaluation_l1606_160614

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 7
  (2*x + 3) * (2*x - 3) - (x + 2)^2 + 4*(x + 3) = 20 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1606_160614


namespace NUMINAMATH_CALUDE_egyptian_fraction_1991_l1606_160668

theorem egyptian_fraction_1991 : ∃ (k l m : ℕ), 
  Odd k ∧ Odd l ∧ Odd m ∧ 
  (1 : ℚ) / 1991 = 1 / k + 1 / l + 1 / m := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_1991_l1606_160668


namespace NUMINAMATH_CALUDE_weight_ratio_l1606_160685

def student_weight : ℝ := 79
def total_weight : ℝ := 116
def weight_loss : ℝ := 5

def sister_weight : ℝ := total_weight - student_weight
def student_new_weight : ℝ := student_weight - weight_loss

theorem weight_ratio : student_new_weight / sister_weight = 2 := by sorry

end NUMINAMATH_CALUDE_weight_ratio_l1606_160685


namespace NUMINAMATH_CALUDE_function_value_at_specific_point_l1606_160621

/-- The base-3 logarithm -/
noncomputable def log3 (x : ℝ) : ℝ := (Real.log x) / (Real.log 3)

/-- The base-10 logarithm -/
noncomputable def lg (x : ℝ) : ℝ := (Real.log x) / (Real.log 10)

/-- The given function f -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin x - b * log3 (Real.sqrt (x^2 + 1) - x) + 1

theorem function_value_at_specific_point
  (a b : ℝ) (h : f a b (lg (log3 10)) = 5) :
  f a b (lg (lg 3)) = -3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_specific_point_l1606_160621


namespace NUMINAMATH_CALUDE_equation_represents_parabola_l1606_160689

/-- The equation y^4 - 6x^2 = 3y^2 - 2 represents a parabola -/
theorem equation_represents_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ x y : ℝ, y^4 - 6*x^2 = 3*y^2 - 2 ↔ a*y^2 + b*x + c = 0) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_parabola_l1606_160689


namespace NUMINAMATH_CALUDE_percentage_increase_l1606_160600

theorem percentage_increase (original : ℝ) (final : ℝ) (increase : ℝ) :
  original = 90 →
  final = 135 →
  increase = (final - original) / original * 100 →
  increase = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l1606_160600


namespace NUMINAMATH_CALUDE_no_real_sqrt_negative_quadratic_l1606_160635

theorem no_real_sqrt_negative_quadratic : ∀ x : ℝ, ¬ ∃ y : ℝ, y^2 = -(x^2 + x + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_sqrt_negative_quadratic_l1606_160635


namespace NUMINAMATH_CALUDE_neon_sign_blink_interval_l1606_160647

theorem neon_sign_blink_interval (t1 t2 : ℕ) : 
  t1 = 9 → 
  t1.lcm t2 = 45 → 
  t2 = 15 := by
sorry

end NUMINAMATH_CALUDE_neon_sign_blink_interval_l1606_160647


namespace NUMINAMATH_CALUDE_price_increase_percentage_l1606_160605

theorem price_increase_percentage (P : ℝ) (h : P > 0) : 
  let cheaper_price := 0.8 * P
  let price_increase := P - cheaper_price
  let percentage_increase := (price_increase / cheaper_price) * 100
  percentage_increase = 25 := by sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l1606_160605


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1606_160677

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - 2*x - 3 < 0) → (-2 < x ∧ x < 3) ∧
  ∃ y : ℝ, -2 < y ∧ y < 3 ∧ ¬(y^2 - 2*y - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1606_160677


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l1606_160620

def N : ℕ := 36 * 72 * 50 * 81

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums : 
  (sum_odd_divisors N) * 126 = sum_even_divisors N := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l1606_160620


namespace NUMINAMATH_CALUDE_floor_sum_abcd_l1606_160652

theorem floor_sum_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^2 + b^2 = 2500) (h2 : c^2 + d^2 = 2500) (h3 : a*c + b*d = 1500) :
  ⌊a + b + c + d⌋ = 126 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_abcd_l1606_160652


namespace NUMINAMATH_CALUDE_max_crates_on_trailer_l1606_160648

/-- The maximum number of crates a trailer can carry given weight constraints -/
theorem max_crates_on_trailer (min_crate_weight max_total_weight : ℕ) 
  (h1 : min_crate_weight ≥ 120)
  (h2 : max_total_weight = 720) :
  (max_total_weight / min_crate_weight : ℕ) = 6 := by
  sorry

#check max_crates_on_trailer

end NUMINAMATH_CALUDE_max_crates_on_trailer_l1606_160648


namespace NUMINAMATH_CALUDE_same_heads_probability_l1606_160638

/-- The number of coins Keiko tosses -/
def keiko_coins : ℕ := 2

/-- The number of coins Ephraim tosses -/
def ephraim_coins : ℕ := 3

/-- The probability of getting the same number of heads -/
def same_heads_prob : ℚ := 3/16

/-- 
Theorem: Given that Keiko tosses 2 coins and Ephraim tosses 3 coins, 
the probability that Ephraim gets the same number of heads as Keiko is 3/16.
-/
theorem same_heads_probability : 
  let outcomes := 2^(keiko_coins + ephraim_coins)
  let favorable_outcomes := (keiko_coins + 1) * (ephraim_coins + 1) / 2
  (favorable_outcomes : ℚ) / outcomes = same_heads_prob := by
  sorry

end NUMINAMATH_CALUDE_same_heads_probability_l1606_160638


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1606_160629

theorem solution_set_of_inequality (x : ℝ) : 
  x * (x + 2) < 3 ↔ -3 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1606_160629


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1606_160631

theorem unique_solution_quadratic (m : ℚ) : 
  (∃! x : ℝ, (x + 6) * (x + 2) = m + 3 * x) ↔ m = 23 / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1606_160631


namespace NUMINAMATH_CALUDE_last_three_digits_perfect_square_l1606_160645

theorem last_three_digits_perfect_square (n : ℕ) : 
  ∃ (m : ℕ), m * m % 1000 = 689 ∧ 
  ∀ (k : ℕ), k * k % 1000 ≠ 759 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_perfect_square_l1606_160645


namespace NUMINAMATH_CALUDE_triangle_values_theorem_l1606_160632

def triangle (a b c : ℚ) (x y : ℚ) : ℚ := a * x + b * y + c * x * y

theorem triangle_values_theorem (a b c d : ℚ) :
  (∀ x : ℚ, triangle a b c x d = x) ∧
  (triangle a b c 1 2 = 3) ∧
  (triangle a b c 2 3 = 4) ∧
  (d ≠ 0) →
  a = 5 ∧ b = 0 ∧ c = -1 ∧ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_values_theorem_l1606_160632


namespace NUMINAMATH_CALUDE_polynomial_equality_sum_of_squares_l1606_160659

theorem polynomial_equality_sum_of_squares (p q r s t u : ℤ) :
  (∀ x : ℝ, 512 * x^3 + 125 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 6410 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_sum_of_squares_l1606_160659


namespace NUMINAMATH_CALUDE_min_value_cyclic_fraction_sum_l1606_160678

theorem min_value_cyclic_fraction_sum (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a / b + b / c + c / d + d / a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_cyclic_fraction_sum_l1606_160678


namespace NUMINAMATH_CALUDE_carpenter_tables_problem_l1606_160699

theorem carpenter_tables_problem (T : ℕ) : 
  T + (T - 3) = 17 → T = 10 := by sorry

end NUMINAMATH_CALUDE_carpenter_tables_problem_l1606_160699


namespace NUMINAMATH_CALUDE_seven_trees_planting_methods_l1606_160695

/-- The number of ways to plant n trees in a row, choosing from plane trees and willow trees,
    such that no two adjacent trees are both willows. -/
def valid_planting_methods (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 34 valid planting methods for 7 trees. -/
theorem seven_trees_planting_methods :
  valid_planting_methods 7 = 34 :=
sorry

end NUMINAMATH_CALUDE_seven_trees_planting_methods_l1606_160695


namespace NUMINAMATH_CALUDE_dental_bill_theorem_l1606_160641

/-- The cost of a dental filling -/
def filling_cost : ℕ := sorry

/-- The cost of a dental cleaning -/
def cleaning_cost : ℕ := 70

/-- The cost of a tooth extraction -/
def extraction_cost : ℕ := 290

/-- The total bill for dental services -/
def total_bill : ℕ := 5 * filling_cost

theorem dental_bill_theorem : 
  filling_cost = 120 ∧ 
  total_bill = cleaning_cost + 2 * filling_cost + extraction_cost := by
  sorry

end NUMINAMATH_CALUDE_dental_bill_theorem_l1606_160641


namespace NUMINAMATH_CALUDE_problem_solution_l1606_160609

def g (x : ℝ) : ℝ := |x - 1| + |2*x + 4|
def f (a x : ℝ) : ℝ := |x - a| + 2 + a

theorem problem_solution :
  (∀ a : ℝ, ∀ x₁ : ℝ, ∃ x₂ : ℝ, g x₁ = f a x₂) →
  (∀ x : ℝ, x ∈ {x : ℝ | g x < 6} ↔ x ∈ Set.Ioo (-3) 1) ∧
  (∀ a : ℝ, (∀ x₁ : ℝ, ∃ x₂ : ℝ, g x₁ = f a x₂) → a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1606_160609


namespace NUMINAMATH_CALUDE_price_reduction_sales_increase_l1606_160625

/-- Proves that a 30% price reduction and 80% sales increase results in a 26% revenue increase -/
theorem price_reduction_sales_increase (P S : ℝ) (P_pos : P > 0) (S_pos : S > 0) :
  let new_price := 0.7 * P
  let new_sales := 1.8 * S
  let original_revenue := P * S
  let new_revenue := new_price * new_sales
  (new_revenue - original_revenue) / original_revenue = 0.26 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_sales_increase_l1606_160625


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_values_l1606_160608

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The ratio of a_n to a_{2n} is constant -/
def constant_ratio (a : ℕ → ℝ) : Prop :=
  ∃ c, ∀ n, a n / a (2 * n) = c

theorem arithmetic_sequence_constant_ratio_values
  (a : ℕ → ℝ) (h1 : arithmetic_sequence a) (h2 : constant_ratio a) :
  ∃ c, (c = 1 ∨ c = 1/2) ∧ ∀ n, a n / a (2 * n) = c :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_values_l1606_160608


namespace NUMINAMATH_CALUDE_grape_crates_count_l1606_160604

/-- Proves that the number of grape crates is 13 given the total number of crates and the number of mango and passion fruit crates. -/
theorem grape_crates_count (total_crates mango_crates passion_fruit_crates : ℕ) 
  (h1 : total_crates = 50)
  (h2 : mango_crates = 20)
  (h3 : passion_fruit_crates = 17) :
  total_crates - (mango_crates + passion_fruit_crates) = 13 := by
  sorry

end NUMINAMATH_CALUDE_grape_crates_count_l1606_160604


namespace NUMINAMATH_CALUDE_box_percentage_difference_l1606_160672

theorem box_percentage_difference
  (stan_boxes : ℕ)
  (john_boxes : ℕ)
  (jules_boxes : ℕ)
  (joseph_boxes : ℕ)
  (h1 : stan_boxes = 100)
  (h2 : john_boxes = 30)
  (h3 : john_boxes = jules_boxes + jules_boxes / 5)
  (h4 : jules_boxes = joseph_boxes + 5) :
  (stan_boxes - joseph_boxes) / stan_boxes = 4/5 :=
sorry

end NUMINAMATH_CALUDE_box_percentage_difference_l1606_160672


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1606_160694

def polynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^3 + b₂*x^2 + b₁*x + 18

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  ∀ x : ℤ, polynomial b₂ b₁ x = 0 →
    x ∈ ({-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1606_160694


namespace NUMINAMATH_CALUDE_circle_point_x_value_l1606_160680

/-- Given a circle in the xy-plane with diameter endpoints (-3,0) and (21,0),
    if the point (x,12) is on the circle, then x = 9. -/
theorem circle_point_x_value (x : ℝ) : 
  let center : ℝ × ℝ := ((21 - 3) / 2 + -3, 0)
  let radius : ℝ := (21 - (-3)) / 2
  ((x - center.1)^2 + (12 - center.2)^2 = radius^2) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_point_x_value_l1606_160680


namespace NUMINAMATH_CALUDE_bracelet_large_beads_l1606_160661

/-- Proves the number of large beads per bracelet given the problem conditions -/
theorem bracelet_large_beads (total_beads : ℕ) (num_bracelets : ℕ) : 
  total_beads = 528 →
  num_bracelets = 11 →
  ∃ (large_beads_per_bracelet : ℕ),
    large_beads_per_bracelet * num_bracelets = total_beads / 2 ∧
    large_beads_per_bracelet = 24 := by
  sorry

#check bracelet_large_beads

end NUMINAMATH_CALUDE_bracelet_large_beads_l1606_160661


namespace NUMINAMATH_CALUDE_larger_number_proof_l1606_160664

theorem larger_number_proof (a b : ℕ+) (x y : ℕ+) 
  (hcf_eq : Nat.gcd a b = 30)
  (x_eq : x = 10)
  (y_eq : y = 15)
  (lcm_eq : Nat.lcm a b = 30 * x * y) :
  max a b = 450 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1606_160664


namespace NUMINAMATH_CALUDE_logarithm_equation_solution_l1606_160611

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithm_equation_solution : 
  ∃ (x : ℝ), x > 0 ∧ 
  (log_base (Real.sqrt 3) x + log_base (Real.sqrt 3) x + log_base (3 ^ (1/6 : ℝ)) x + 
   log_base (Real.sqrt 3) x + log_base (Real.sqrt 3) x + log_base (Real.sqrt 3) x + 
   log_base (Real.sqrt 3) x + log_base (Real.sqrt 3) x = 36) ∧
  x = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_logarithm_equation_solution_l1606_160611


namespace NUMINAMATH_CALUDE_integers_between_cubes_l1606_160682

theorem integers_between_cubes : ∃ n : ℕ, n = 26 ∧ 
  n = (⌊(9.3 : ℝ)^3⌋ - ⌈(9.2 : ℝ)^3⌉ + 1) := by
  sorry

end NUMINAMATH_CALUDE_integers_between_cubes_l1606_160682


namespace NUMINAMATH_CALUDE_intersection_sum_l1606_160634

theorem intersection_sum (c d : ℝ) : 
  (3 = (1/3) * 3 + c) → 
  (3 = (1/3) * 3 + d) → 
  c + d = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l1606_160634


namespace NUMINAMATH_CALUDE_complement_A_union_B_when_m_4_range_of_m_for_B_subset_A_l1606_160697

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem for part I
theorem complement_A_union_B_when_m_4 :
  (Set.univ : Set ℝ) \ (A ∪ B 4) = {x | x < -2 ∨ x > 7} := by sorry

-- Theorem for part II
theorem range_of_m_for_B_subset_A :
  {m : ℝ | (B m).Nonempty ∧ B m ⊆ A} = {m | 2 ≤ m ∧ m ≤ 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_when_m_4_range_of_m_for_B_subset_A_l1606_160697


namespace NUMINAMATH_CALUDE_percentage_to_total_l1606_160669

/-- If 25% of an amount is 75 rupees, then the total amount is 300 rupees. -/
theorem percentage_to_total (amount : ℝ) : (25 / 100) * amount = 75 → amount = 300 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_total_l1606_160669


namespace NUMINAMATH_CALUDE_kangaroo_hop_distance_l1606_160687

theorem kangaroo_hop_distance :
  let a : ℚ := 1/4  -- first term
  let r : ℚ := 3/4  -- common ratio
  let n : ℕ := 6    -- number of hops
  (a * (1 - r^n)) / (1 - r) = 3367/4096 := by
sorry

end NUMINAMATH_CALUDE_kangaroo_hop_distance_l1606_160687


namespace NUMINAMATH_CALUDE_sqrt_sum_zero_implies_both_zero_l1606_160636

theorem sqrt_sum_zero_implies_both_zero (x y : ℝ) :
  Real.sqrt x + Real.sqrt y = 0 → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_zero_implies_both_zero_l1606_160636


namespace NUMINAMATH_CALUDE_exam_questions_count_l1606_160602

/-- Calculates the total number of questions in an examination given specific conditions. -/
theorem exam_questions_count 
  (type_a_count : ℕ)
  (type_a_time : ℕ)
  (total_time : ℕ)
  (h1 : type_a_count = 50)
  (h2 : type_a_time = 72)
  (h3 : total_time = 180)
  (h4 : type_a_time * 2 ≤ total_time) :
  ∃ (type_b_count : ℕ),
    (type_a_count + type_b_count = 200) ∧
    (type_a_time + type_b_count * (type_a_time / type_a_count / 2) = total_time) :=
by sorry

end NUMINAMATH_CALUDE_exam_questions_count_l1606_160602


namespace NUMINAMATH_CALUDE_jane_shorter_than_sarah_l1606_160639

-- Define the lengths of the sticks and the covered portion
def pat_stick_length : ℕ := 30
def pat_covered_length : ℕ := 7
def jane_stick_length : ℕ := 22

-- Define Sarah's stick length based on Pat's uncovered portion
def sarah_stick_length : ℕ := 2 * (pat_stick_length - pat_covered_length)

-- State the theorem
theorem jane_shorter_than_sarah : sarah_stick_length - jane_stick_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_jane_shorter_than_sarah_l1606_160639
