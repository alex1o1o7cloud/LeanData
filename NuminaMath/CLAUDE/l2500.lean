import Mathlib

namespace NUMINAMATH_CALUDE_treasure_in_fourth_bag_l2500_250019

/-- Given four bags A, B, C, and D, prove that D is the heaviest bag. -/
theorem treasure_in_fourth_bag (A B C D : ℝ) 
  (h1 : A + B < C)
  (h2 : A + C = D)
  (h3 : A + D > B + C) :
  D > A ∧ D > B ∧ D > C := by
  sorry

end NUMINAMATH_CALUDE_treasure_in_fourth_bag_l2500_250019


namespace NUMINAMATH_CALUDE_problem_solution_l2500_250021

/-- A function representing the relationship between x and y -/
def f (x : ℝ) : ℝ := sorry

/-- The constant of proportionality -/
def k : ℝ := sorry

theorem problem_solution :
  (∀ x : ℝ, f x - 4 = k * (2 * x + 1)) →  -- y-4 is directly proportional to 2x+1
  (f (-1) = 6) →                          -- When x = -1, y = 6
  (∀ x : ℝ, f x = -4 * x + 2) ∧           -- Functional expression
  (f (3/2) = -4)                          -- When y = -4, x = 3/2
  := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2500_250021


namespace NUMINAMATH_CALUDE_cricket_bat_selling_price_l2500_250000

-- Define the profit amount
def profit : ℝ := 230

-- Define the profit percentage
def profitPercentage : ℝ := 37.096774193548384

-- Define the selling price
def sellingPrice : ℝ := 850

-- Theorem to prove
theorem cricket_bat_selling_price :
  (profit / (profitPercentage / 100) + profit) = sellingPrice := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_selling_price_l2500_250000


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l2500_250070

theorem square_garden_perimeter (q p : ℝ) : 
  q = 49 → -- Area of the garden is 49 square feet
  q = p + 21 → -- Given relationship between q and p
  (4 * Real.sqrt q) = 28 -- Perimeter of the garden is 28 feet
:= by sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l2500_250070


namespace NUMINAMATH_CALUDE_opposite_of_23_l2500_250029

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_23 : opposite 23 = -23 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_23_l2500_250029


namespace NUMINAMATH_CALUDE_pattern_paths_count_l2500_250089

/-- Represents a position in the diagram -/
structure Position :=
  (row : ℕ) (col : ℕ)

/-- Represents a letter in the diagram -/
inductive Letter
  | P | A | T | E | R | N | C | O

/-- The diagram of letters -/
def diagram : List (List Letter) := sorry

/-- Checks if two positions are adjacent -/
def adjacent (p1 p2 : Position) : Prop := sorry

/-- Checks if a path spells "PATTERN" -/
def spells_pattern (path : List Position) : Prop := sorry

/-- Counts the number of valid paths spelling "PATTERN" -/
def count_pattern_paths : ℕ := sorry

/-- The main theorem to prove -/
theorem pattern_paths_count :
  count_pattern_paths = 18 := by sorry

end NUMINAMATH_CALUDE_pattern_paths_count_l2500_250089


namespace NUMINAMATH_CALUDE_car_p_distance_l2500_250037

/-- The distance traveled by a car given its speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem car_p_distance (v : ℝ) : 
  let car_m_speed := v
  let car_m_time := 3
  let car_n_speed := 3 * v
  let car_n_time := 2
  let car_p_speed := 2 * v
  let car_p_time := 1.5
  distance car_p_speed car_p_time = 3 * v :=
by sorry

end NUMINAMATH_CALUDE_car_p_distance_l2500_250037


namespace NUMINAMATH_CALUDE_book_pricing_theorem_l2500_250008

-- Define the cost function
def C (n : ℕ) : ℕ :=
  if n ≤ 24 then 12 * n
  else if n ≤ 48 then 11 * n
  else 10 * n

-- Define the production cost
def production_cost : ℕ := 5

-- Define the theorem
theorem book_pricing_theorem :
  -- Part 1: Exactly 6 values of n where C(n+1) < C(n)
  (∃ (S : Finset ℕ), S.card = 6 ∧ ∀ n, n ∈ S ↔ C (n + 1) < C n) ∧
  -- Part 2: Profit range for two individuals buying 60 books
  (∀ a b : ℕ, a + b = 60 → a ≥ 1 → b ≥ 1 →
    302 ≤ C a + C b - 60 * production_cost ∧
    C a + C b - 60 * production_cost ≤ 384) :=
by sorry

end NUMINAMATH_CALUDE_book_pricing_theorem_l2500_250008


namespace NUMINAMATH_CALUDE_direction_vector_of_parameterized_line_l2500_250064

/-- The direction vector of a parameterized line satisfying given conditions -/
theorem direction_vector_of_parameterized_line :
  ∃ (v d : ℝ × ℝ),
    (∀ (x y t : ℝ), y = (2 * x + 1) / 3 → (x, y) = v + t • d) ∧
    (∀ (x y t : ℝ), x ≤ -1 → y = (2 * x + 1) / 3 → 
      ‖(x, y) - (-1, -1)‖ = t) →
    d = (-3 / Real.sqrt 13, -2 / Real.sqrt 13) := by
  sorry

end NUMINAMATH_CALUDE_direction_vector_of_parameterized_line_l2500_250064


namespace NUMINAMATH_CALUDE_decagon_cuts_to_two_regular_polygons_l2500_250092

/-- A regular polygon with n sides -/
structure RegularPolygon where
  sides : Nat
  isRegular : sides ≥ 3

/-- A decagon is a regular polygon with 10 sides -/
def Decagon : RegularPolygon where
  sides := 10
  isRegular := by norm_num

/-- Represent a cut of a polygon along its diagonals -/
structure DiagonalCut (p : RegularPolygon) where
  pieces : List RegularPolygon
  sum_sides : (pieces.map RegularPolygon.sides).sum = p.sides

/-- Theorem: A regular decagon can be cut into two regular polygons -/
theorem decagon_cuts_to_two_regular_polygons : 
  ∃ (cut : DiagonalCut Decagon), cut.pieces.length = 2 := by
  sorry

end NUMINAMATH_CALUDE_decagon_cuts_to_two_regular_polygons_l2500_250092


namespace NUMINAMATH_CALUDE_cos_180_degrees_l2500_250074

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l2500_250074


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2500_250003

theorem arithmetic_sequence_common_difference
  (a : ℝ)  -- first term
  (an : ℝ) -- last term
  (s : ℝ)  -- sum of all terms
  (h1 : a = 7)
  (h2 : an = 88)
  (h3 : s = 570)
  : ∃ n : ℕ, n > 1 ∧ (an - a) / (n - 1) = 81 / 11 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2500_250003


namespace NUMINAMATH_CALUDE_principal_is_300_l2500_250073

/-- Given a principal amount P and an interest rate R, 
    if increasing the rate by 6% for 5 years results in 90 more interest,
    then P must be 300. -/
theorem principal_is_300 (P R : ℝ) : 
  (P * (R + 6) * 5) / 100 = (P * R * 5) / 100 + 90 → P = 300 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_300_l2500_250073


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_theorem_l2500_250067

/-- Represents the financial data of a person -/
structure FinancialData where
  income : ℕ
  savings : ℕ

/-- Calculates the expenditure given income and savings -/
def calculateExpenditure (data : FinancialData) : ℕ :=
  data.income - data.savings

/-- Calculates the ratio of income to expenditure -/
def incomeToExpenditureRatio (data : FinancialData) : Rat :=
  data.income / (calculateExpenditure data)

/-- Theorem stating that for a person with an income of 20000 and savings of 4000,
    the ratio of income to expenditure is 5/4 -/
theorem income_expenditure_ratio_theorem (data : FinancialData)
    (h1 : data.income = 20000)
    (h2 : data.savings = 4000) :
    incomeToExpenditureRatio data = 5 / 4 := by
  sorry


end NUMINAMATH_CALUDE_income_expenditure_ratio_theorem_l2500_250067


namespace NUMINAMATH_CALUDE_total_time_is_four_hours_l2500_250051

def first_movie_length : ℚ := 3/2 -- 1.5 hours
def second_movie_length : ℚ := first_movie_length + 1/2 -- 30 minutes longer
def popcorn_time : ℚ := 1/6 -- 10 minutes in hours
def fries_time : ℚ := 2 * popcorn_time -- twice as long as popcorn time

def total_time : ℚ := first_movie_length + second_movie_length + popcorn_time + fries_time

theorem total_time_is_four_hours : total_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_four_hours_l2500_250051


namespace NUMINAMATH_CALUDE_geometric_sequence_product_roots_product_geometric_sequence_roots_product_l2500_250001

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) (h : geometric_sequence a) :
  ∀ i j k l : ℕ, i + j = k + l → a i * a j = a k * a l :=
sorry

theorem roots_product (p q r : ℝ) (x y : ℝ) (hx : p * x^2 + q * x + r = 0) (hy : p * y^2 + q * y + r = 0) :
  x * y = r / p :=
sorry

theorem geometric_sequence_roots_product (a : ℕ → ℝ) :
  geometric_sequence a →
  3 * (a 1)^2 + 7 * (a 1) - 9 = 0 →
  3 * (a 10)^2 + 7 * (a 10) - 9 = 0 →
  a 4 * a 7 = -3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_roots_product_geometric_sequence_roots_product_l2500_250001


namespace NUMINAMATH_CALUDE_sum_recurring_thirds_equals_one_l2500_250040

-- Define the recurring decimal 0.333...
def recurring_third : ℚ := 1 / 3

-- Define the recurring decimal 0.666...
def recurring_two_thirds : ℚ := 2 / 3

-- Theorem: The sum of 0.333... and 0.666... is equal to 1
theorem sum_recurring_thirds_equals_one : 
  recurring_third + recurring_two_thirds = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_recurring_thirds_equals_one_l2500_250040


namespace NUMINAMATH_CALUDE_bulb_selection_problem_l2500_250009

theorem bulb_selection_problem (total_bulbs : ℕ) (defective_bulbs : ℕ) 
  (prob_at_least_one_defective : ℝ) :
  total_bulbs = 22 →
  defective_bulbs = 4 →
  prob_at_least_one_defective = 0.33766233766233766 →
  ∃ n : ℕ, n = 2 ∧ 
    (1 - ((total_bulbs - defective_bulbs : ℝ) / total_bulbs) ^ n) = prob_at_least_one_defective :=
by sorry

end NUMINAMATH_CALUDE_bulb_selection_problem_l2500_250009


namespace NUMINAMATH_CALUDE_problem_solution_l2500_250015

theorem problem_solution : (120 / (6 / 3)) - 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2500_250015


namespace NUMINAMATH_CALUDE_walking_distance_l2500_250002

theorem walking_distance (initial_speed : ℝ) (faster_speed : ℝ) (additional_distance : ℝ) 
  (h1 : initial_speed = 12)
  (h2 : faster_speed = 16)
  (h3 : additional_distance = 20) :
  ∃ (actual_distance : ℝ) (time : ℝ),
    actual_distance = initial_speed * time ∧
    actual_distance + additional_distance = faster_speed * time ∧
    actual_distance = 60 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_l2500_250002


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2500_250062

theorem purely_imaginary_complex_number (x : ℝ) : 
  let z : ℂ := Complex.mk (x^2 - 2*x - 3) (x + 1)
  (z.re = 0 ∧ z.im ≠ 0) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2500_250062


namespace NUMINAMATH_CALUDE_circle_radius_l2500_250056

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 10*y + 34 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (4, 5)

/-- The theorem stating that the radius of the circle is √7 -/
theorem circle_radius :
  ∃ (r : ℝ), r > 0 ∧
  (∀ (x y : ℝ), circle_equation x y ↔ 
    (x - circle_center.1)^2 + (y - circle_center.2)^2 = r^2) ∧
  r = Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l2500_250056


namespace NUMINAMATH_CALUDE_tangent_product_special_angles_l2500_250030

theorem tangent_product_special_angles :
  let A : Real := 30 * π / 180
  let B : Real := 60 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) = 2 + 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_special_angles_l2500_250030


namespace NUMINAMATH_CALUDE_ab_plus_cd_eq_zero_l2500_250098

/-- Given real numbers a, b, c, d satisfying certain conditions, prove that ab + cd = 0 -/
theorem ab_plus_cd_eq_zero 
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) 
  (h3 : a*c + b*d = 0) : 
  a*b + c*d = 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_plus_cd_eq_zero_l2500_250098


namespace NUMINAMATH_CALUDE_unique_solution_system_l2500_250061

theorem unique_solution_system (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x * (x + y + z) = 26 ∧ y * (x + y + z) = 27 ∧ z * (x + y + z) = 28 →
  x = 26 / 9 ∧ y = 3 ∧ z = 28 / 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2500_250061


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2500_250031

/-- A geometric sequence with first term 1 and common ratio q -/
def geometric_sequence (q : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => q * geometric_sequence q n

theorem geometric_sequence_product (q : ℝ) (h : q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1) :
  ∃ m : ℕ, geometric_sequence q (m - 1) = (geometric_sequence q 0) *
    (geometric_sequence q 1) * (geometric_sequence q 2) *
    (geometric_sequence q 3) * (geometric_sequence q 4) ∧ m = 11 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2500_250031


namespace NUMINAMATH_CALUDE_line_parallel_plane_neither_necessary_nor_sufficient_l2500_250010

/-- Two lines are perpendicular -/
def perpendicular (l₁ l₂ : Line) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (p : Plane) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem line_parallel_plane_neither_necessary_nor_sufficient
  (m n : Line) (α : Plane) (h : perpendicular m n) :
  ¬(∀ (m n : Line) (α : Plane), perpendicular m n → (line_parallel_plane n α ↔ line_perp_plane m α)) :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_plane_neither_necessary_nor_sufficient_l2500_250010


namespace NUMINAMATH_CALUDE_nim_max_product_l2500_250082

/-- Nim-sum (bitwise XOR) of two natural numbers -/
def nim_sum (a b : ℕ) : ℕ := a ^^^ b

/-- Check if a given configuration is a losing position in 3-player Nim -/
def is_losing_position (a b c d : ℕ) : Prop :=
  nim_sum (nim_sum (nim_sum a b) c) d = 0

/-- The maximum product of x and y satisfying the game conditions -/
def max_product : ℕ := 7704

/-- The theorem stating the maximum product of x and y in the given Nim game -/
theorem nim_max_product :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧
  is_losing_position 43 99 x y ∧
  x * y = max_product ∧
  ∀ (a b : ℕ), a > 0 → b > 0 → is_losing_position 43 99 a b → a * b ≤ max_product :=
sorry

end NUMINAMATH_CALUDE_nim_max_product_l2500_250082


namespace NUMINAMATH_CALUDE_sum_a_d_equals_two_l2500_250033

theorem sum_a_d_equals_two (a b c d : ℝ) 
  (h1 : a + b = 4)
  (h2 : b + c = 7)
  (h3 : c + d = 5) :
  a + d = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_equals_two_l2500_250033


namespace NUMINAMATH_CALUDE_boxes_on_pallet_l2500_250023

/-- 
Given a pallet of boxes with a total weight and the weight of each box,
calculate the number of boxes on the pallet.
-/
theorem boxes_on_pallet (total_weight : ℕ) (box_weight : ℕ) 
  (h1 : total_weight = 267)
  (h2 : box_weight = 89) :
  total_weight / box_weight = 3 :=
by sorry

end NUMINAMATH_CALUDE_boxes_on_pallet_l2500_250023


namespace NUMINAMATH_CALUDE_min_value_theorem_l2500_250024

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hmin : ∀ x, |x + a| + |x - b| ≥ 4) : 
  (a + b = 4) ∧ (∀ a' b' : ℝ, a' > 0 → b' > 0 → (1/4) * a'^2 + (1/9) * b'^2 ≥ 16/13) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2500_250024


namespace NUMINAMATH_CALUDE_best_of_three_match_probability_l2500_250050

/-- The probability of player A winning a single game against player B. -/
def p_win_game : ℚ := 1/3

/-- The probability of player A winning a best-of-three match against player B. -/
def p_win_match : ℚ := 7/27

/-- Theorem stating that if the probability of player A winning each game is 1/3,
    then the probability of A winning a best-of-three match is 7/27. -/
theorem best_of_three_match_probability :
  p_win_game = 1/3 → p_win_match = 7/27 := by sorry

end NUMINAMATH_CALUDE_best_of_three_match_probability_l2500_250050


namespace NUMINAMATH_CALUDE_only_cone_no_rectangular_cross_section_l2500_250016

-- Define the geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | RectangularPrism
  | Cube

-- Define a function that checks if a solid can have a rectangular cross-section
def canHaveRectangularCrossSection (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => true
  | GeometricSolid.Cone => false
  | GeometricSolid.RectangularPrism => true
  | GeometricSolid.Cube => true

-- Theorem stating that only a cone cannot have a rectangular cross-section
theorem only_cone_no_rectangular_cross_section :
  ∀ (solid : GeometricSolid),
    ¬(canHaveRectangularCrossSection solid) ↔ solid = GeometricSolid.Cone :=
by sorry

end NUMINAMATH_CALUDE_only_cone_no_rectangular_cross_section_l2500_250016


namespace NUMINAMATH_CALUDE_prob_not_blue_from_odds_l2500_250065

-- Define the odds ratio
def odds_blue : ℚ := 5 / 6

-- Define the probability of not obtaining a blue ball
def prob_not_blue : ℚ := 6 / 11

-- Theorem statement
theorem prob_not_blue_from_odds :
  odds_blue = 5 / 6 → prob_not_blue = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_blue_from_odds_l2500_250065


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2500_250028

theorem trigonometric_identity (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.sin y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2500_250028


namespace NUMINAMATH_CALUDE_room_tiling_l2500_250068

theorem room_tiling (room_length room_width : ℕ) 
  (border_tile_size inner_tile_size : ℕ) : 
  room_length = 16 → 
  room_width = 12 → 
  border_tile_size = 1 → 
  inner_tile_size = 2 → 
  (2 * (room_length - 2 + room_width - 2) + 4) + 
  ((room_length - 2) * (room_width - 2)) / (inner_tile_size ^ 2) = 87 :=
by
  sorry

end NUMINAMATH_CALUDE_room_tiling_l2500_250068


namespace NUMINAMATH_CALUDE_exponent_division_l2500_250046

theorem exponent_division (a : ℝ) (m n : ℕ) : a ^ m / a ^ n = a ^ (m - n) := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2500_250046


namespace NUMINAMATH_CALUDE_intersection_count_l2500_250039

/-- The complementary curve C₂ -/
def complementary_curve (x y : ℝ) : Prop := 1 / x^2 - 1 / y^2 = 1

/-- The hyperbola C₁ -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- The line MN passing through (m,0) and (0,n) -/
def line_mn (m n x y : ℝ) : Prop := y = -n/m * x + n

theorem intersection_count (m n : ℝ) :
  complementary_curve m n →
  ∃! p : ℝ × ℝ, hyperbola p.1 p.2 ∧ line_mn m n p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_intersection_count_l2500_250039


namespace NUMINAMATH_CALUDE_jerry_shelves_theorem_l2500_250059

def shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : ℕ :=
  ((total_books - books_taken) + books_per_shelf - 1) / books_per_shelf

theorem jerry_shelves_theorem :
  shelves_needed 34 7 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jerry_shelves_theorem_l2500_250059


namespace NUMINAMATH_CALUDE_equation_proof_l2500_250025

theorem equation_proof : Real.sqrt (72 * 2) + (5568 / 87) ^ (1/3) = Real.sqrt 256 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2500_250025


namespace NUMINAMATH_CALUDE_regular_nonagon_perimeter_l2500_250094

/-- A regular polygon with 9 sides, each 2 centimeters long -/
structure RegularNonagon where
  side_length : ℝ
  num_sides : ℕ
  h1 : side_length = 2
  h2 : num_sides = 9

/-- The perimeter of a regular nonagon -/
def perimeter (n : RegularNonagon) : ℝ :=
  n.side_length * n.num_sides

/-- Theorem: The perimeter of a regular nonagon with side length 2 cm is 18 cm -/
theorem regular_nonagon_perimeter (n : RegularNonagon) : perimeter n = 18 := by
  sorry

#check regular_nonagon_perimeter

end NUMINAMATH_CALUDE_regular_nonagon_perimeter_l2500_250094


namespace NUMINAMATH_CALUDE_same_start_end_words_count_l2500_250006

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The length of the words we're considering -/
def word_length : ℕ := 5

/-- The number of freely chosen letters in each word -/
def free_choices : ℕ := word_length - 2

/-- The number of five-letter words that begin and end with the same letter -/
def same_start_end_words : ℕ := alphabet_size ^ free_choices

theorem same_start_end_words_count :
  same_start_end_words = 456976 :=
sorry

end NUMINAMATH_CALUDE_same_start_end_words_count_l2500_250006


namespace NUMINAMATH_CALUDE_tim_score_is_38000_l2500_250034

/-- The value of a single line in points -/
def single_line_value : ℕ := 1000

/-- The value of a tetris in points -/
def tetris_value : ℕ := 8 * single_line_value

/-- The number of singles Tim scored -/
def tim_singles : ℕ := 6

/-- The number of tetrises Tim scored -/
def tim_tetrises : ℕ := 4

/-- Tim's total score -/
def tim_total_score : ℕ := tim_singles * single_line_value + tim_tetrises * tetris_value

theorem tim_score_is_38000 : tim_total_score = 38000 := by
  sorry

end NUMINAMATH_CALUDE_tim_score_is_38000_l2500_250034


namespace NUMINAMATH_CALUDE_joey_age_is_12_l2500_250004

def ages : List ℕ := [4, 6, 8, 10, 12, 14]

def went_to_movies (a b : ℕ) : Prop := a + b = 18 ∧ a ∈ ages ∧ b ∈ ages ∧ a ≠ b

def went_to_soccer (a b : ℕ) : Prop := a < 12 ∧ b < 12 ∧ a ∈ ages ∧ b ∈ ages ∧ a ≠ b

def stayed_home (joey_age : ℕ) : Prop := joey_age ∈ ages ∧ 6 ∈ ages

theorem joey_age_is_12 :
  ∃! (joey_age : ℕ),
    (∃ (a b c d : ℕ),
      went_to_movies a b ∧
      went_to_soccer c d ∧
      stayed_home joey_age ∧
      {a, b, c, d, joey_age, 6} = ages.toFinset) ∧
    joey_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_joey_age_is_12_l2500_250004


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l2500_250032

theorem simplify_nested_roots (a : ℝ) (ha : a > 0) :
  (((a^16)^(1/8))^(1/4))^3 * (((a^16)^(1/4))^(1/8))^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l2500_250032


namespace NUMINAMATH_CALUDE_overtake_time_problem_l2500_250078

/-- Proves that under given conditions, k started 10 hours after a. -/
theorem overtake_time_problem (speed_a speed_b speed_k : ℝ) 
  (start_delay_b : ℝ) (overtake_time : ℝ) :
  speed_a = 30 →
  speed_b = 40 →
  speed_k = 60 →
  start_delay_b = 5 →
  speed_a * overtake_time = speed_b * (overtake_time - start_delay_b) →
  speed_a * overtake_time = speed_k * (overtake_time - (overtake_time - 10)) →
  overtake_time - (overtake_time - 10) = 10 :=
by sorry

end NUMINAMATH_CALUDE_overtake_time_problem_l2500_250078


namespace NUMINAMATH_CALUDE_xyz_product_l2500_250091

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x * (y + z) = 120)
  (eq2 : y * (z + x) = 156)
  (eq3 : z * (x + y) = 144) :
  x * y * z = 360 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l2500_250091


namespace NUMINAMATH_CALUDE_max_servings_emily_l2500_250084

/-- Represents the recipe for the smoothie --/
structure Recipe :=
  (servings : ℕ)
  (bananas : ℕ)
  (strawberries : ℕ)
  (yogurt : ℕ)
  (honey : ℕ)

/-- Represents Emily's available ingredients --/
structure Available :=
  (bananas : ℕ)
  (strawberries : ℕ)
  (yogurt : ℕ)

def recipe : Recipe :=
  { servings := 8
  , bananas := 3
  , strawberries := 2
  , yogurt := 1
  , honey := 4 }

def emily : Available :=
  { bananas := 9
  , strawberries := 8
  , yogurt := 3 }

/-- Calculates the maximum number of servings that can be made --/
def maxServings (r : Recipe) (a : Available) : ℕ :=
  min (a.bananas * r.servings / r.bananas)
      (min (a.strawberries * r.servings / r.strawberries)
           (a.yogurt * r.servings / r.yogurt))

theorem max_servings_emily :
  maxServings recipe emily = 24 := by
  sorry

end NUMINAMATH_CALUDE_max_servings_emily_l2500_250084


namespace NUMINAMATH_CALUDE_allan_initial_balloons_l2500_250020

theorem allan_initial_balloons :
  ∀ (allan_initial jake_balloons allan_bought total : ℕ),
    jake_balloons = 5 →
    allan_bought = 2 →
    total = 10 →
    allan_initial + allan_bought + jake_balloons = total →
    allan_initial = 3 := by
  sorry

end NUMINAMATH_CALUDE_allan_initial_balloons_l2500_250020


namespace NUMINAMATH_CALUDE_marthas_cat_rats_l2500_250035

theorem marthas_cat_rats (R : ℕ) : 
  (5 * (R + 7) - 3 = 47) → R = 3 := by
  sorry

end NUMINAMATH_CALUDE_marthas_cat_rats_l2500_250035


namespace NUMINAMATH_CALUDE_seven_couples_handshakes_l2500_250077

/-- Represents a gathering of couples -/
structure Gathering where
  couples : ℕ
  deriving Repr

/-- Calculates the number of handshakes in a gathering under specific conditions -/
def handshakes (g : Gathering) : ℕ :=
  let total_people := 2 * g.couples
  let handshakes_per_person := total_people - 3  -- Excluding self, spouse, and one other
  (total_people * handshakes_per_person) / 2 - g.couples

/-- Theorem stating that in a gathering of 7 couples, 
    with the given handshake conditions, there are 77 handshakes -/
theorem seven_couples_handshakes :
  handshakes { couples := 7 } = 77 := by
  sorry

#eval handshakes { couples := 7 }

end NUMINAMATH_CALUDE_seven_couples_handshakes_l2500_250077


namespace NUMINAMATH_CALUDE_one_positive_integer_satisfies_condition_l2500_250053

theorem one_positive_integer_satisfies_condition : 
  ∃! (x : ℕ+), 25 - (5 * x.val) > 15 := by sorry

end NUMINAMATH_CALUDE_one_positive_integer_satisfies_condition_l2500_250053


namespace NUMINAMATH_CALUDE_group_collection_theorem_l2500_250041

/-- Calculates the total collection amount in rupees for a group where each member
    contributes as many paise as there are members. -/
def totalCollectionInRupees (numberOfMembers : ℕ) : ℚ :=
  (numberOfMembers * numberOfMembers : ℚ) / 100

/-- Proves that for a group of 88 members, where each member contributes as many
    paise as there are members, the total collection amount is 77.44 rupees. -/
theorem group_collection_theorem :
  totalCollectionInRupees 88 = 77.44 := by
  sorry

#eval totalCollectionInRupees 88

end NUMINAMATH_CALUDE_group_collection_theorem_l2500_250041


namespace NUMINAMATH_CALUDE_translation_result_l2500_250071

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translates a point vertically -/
def translateVertical (p : Point2D) (dy : ℝ) : Point2D :=
  { x := p.x, y := p.y + dy }

/-- Translates a point horizontally -/
def translateHorizontal (p : Point2D) (dx : ℝ) : Point2D :=
  { x := p.x - dx, y := p.y }

/-- The theorem to be proved -/
theorem translation_result :
  let initial_point : Point2D := { x := 3, y := -2 }
  let after_vertical := translateVertical initial_point 3
  let final_point := translateHorizontal after_vertical 2
  final_point = { x := 1, y := 1 } := by
  sorry

end NUMINAMATH_CALUDE_translation_result_l2500_250071


namespace NUMINAMATH_CALUDE_parallelogram_theorem_l2500_250086

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Checks if a point is the midpoint of a line segment -/
def is_midpoint (M : Point) (A B : Point) : Prop := sorry

/-- Checks if a point is inside a quadrilateral -/
def is_inside (M : Point) (q : Quadrilateral) : Prop := sorry

/-- Checks if four points form a parallelogram -/
def is_parallelogram (A B C D : Point) : Prop := sorry

/-- The main theorem -/
theorem parallelogram_theorem (ABCD : Quadrilateral) (P Q R S M : Point) :
  is_convex ABCD →
  is_midpoint P ABCD.A ABCD.B →
  is_midpoint Q ABCD.B ABCD.C →
  is_midpoint R ABCD.C ABCD.D →
  is_midpoint S ABCD.D ABCD.A →
  is_inside M ABCD →
  is_parallelogram ABCD.A P M S →
  is_parallelogram ABCD.C R M Q := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_theorem_l2500_250086


namespace NUMINAMATH_CALUDE_product_sixty_sum_diff_equality_l2500_250079

theorem product_sixty_sum_diff_equality (A B C D : ℕ+) :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A * B = 60 →
  C * D = 60 →
  A - B = C + D →
  A = 20 := by
sorry

end NUMINAMATH_CALUDE_product_sixty_sum_diff_equality_l2500_250079


namespace NUMINAMATH_CALUDE_divisors_of_2_pow_18_minus_1_l2500_250088

theorem divisors_of_2_pow_18_minus_1 :
  ∃! (a b : ℕ), 20 < a ∧ a < 30 ∧ 20 < b ∧ b < 30 ∧
  (2^18 - 1) % a = 0 ∧ (2^18 - 1) % b = 0 ∧ a ≠ b ∧
  a = 19 ∧ b = 27 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_2_pow_18_minus_1_l2500_250088


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l2500_250087

/-- The quadratic equation x^2 - bx + c = 0 with roots 1 and -2 has b = -1 and c = -2 -/
theorem quadratic_roots_to_coefficients :
  ∀ (b c : ℝ),
  (∀ x : ℝ, x^2 - b*x + c = 0 ↔ x = 1 ∨ x = -2) →
  b = -1 ∧ c = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l2500_250087


namespace NUMINAMATH_CALUDE_inscribed_tetrahedron_volume_l2500_250081

/-- Represents a tetrahedron with a triangular base and square lateral faces -/
structure Tetrahedron where
  base_side_length : ℝ
  has_square_lateral_faces : Bool

/-- Represents a tetrahedron inscribed within another tetrahedron -/
structure InscribedTetrahedron where
  outer : Tetrahedron
  vertices_touch_midpoints : Bool
  base_parallel : Bool

/-- Calculates the volume of an inscribed tetrahedron -/
def volume_inscribed_tetrahedron (t : InscribedTetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the inscribed tetrahedron -/
theorem inscribed_tetrahedron_volume 
  (t : InscribedTetrahedron) 
  (h1 : t.outer.base_side_length = 2) 
  (h2 : t.outer.has_square_lateral_faces = true)
  (h3 : t.vertices_touch_midpoints = true)
  (h4 : t.base_parallel = true) : 
  volume_inscribed_tetrahedron t = Real.sqrt 2 / 12 := by sorry

end NUMINAMATH_CALUDE_inscribed_tetrahedron_volume_l2500_250081


namespace NUMINAMATH_CALUDE_unique_solution_l2500_250047

theorem unique_solution : ∀ x y : ℕ+, 
  (x : ℝ) ^ (y : ℝ) - 1 = (y : ℝ) ^ (x : ℝ) → 
  2 * (x : ℝ) ^ (y : ℝ) = (y : ℝ) ^ (x : ℝ) + 5 → 
  x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2500_250047


namespace NUMINAMATH_CALUDE_new_building_windows_l2500_250054

/-- The number of windows needed for a new building --/
def total_windows (installed : ℕ) (hours_per_window : ℕ) (remaining_hours : ℕ) : ℕ :=
  installed + remaining_hours / hours_per_window

/-- Proof that the total number of windows needed is 14 --/
theorem new_building_windows :
  total_windows 5 4 36 = 14 :=
by sorry

end NUMINAMATH_CALUDE_new_building_windows_l2500_250054


namespace NUMINAMATH_CALUDE_olivia_remaining_money_l2500_250018

/-- Given an initial amount of money and an amount spent, 
    calculate the remaining amount. -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem stating that given an initial amount of 78 dollars 
    and a spent amount of 15 dollars, the remaining amount is 63 dollars. -/
theorem olivia_remaining_money :
  remaining_money 78 15 = 63 := by
  sorry

end NUMINAMATH_CALUDE_olivia_remaining_money_l2500_250018


namespace NUMINAMATH_CALUDE_triangle_area_from_inradius_and_perimeter_l2500_250097

/-- Given a triangle with angles A and B, perimeter p, and inradius r, 
    proves that the area of the triangle is equal to r * (p / 2) -/
theorem triangle_area_from_inradius_and_perimeter 
  (A B : Real) (p r : Real) (h1 : A = 40) (h2 : B = 60) (h3 : p = 40) (h4 : r = 2.5) : 
  r * (p / 2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_inradius_and_perimeter_l2500_250097


namespace NUMINAMATH_CALUDE_square_root_of_nine_l2500_250075

theorem square_root_of_nine : 
  ∃ (x : ℝ), x^2 = 9 ↔ x = 3 ∨ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l2500_250075


namespace NUMINAMATH_CALUDE_smart_mart_science_kits_l2500_250045

/-- The number of puzzles sold by Smart Mart -/
def puzzles_sold : ℕ := 36

/-- The difference between science kits and puzzles sold -/
def difference : ℕ := 9

/-- The number of science kits sold by Smart Mart -/
def science_kits_sold : ℕ := puzzles_sold + difference

/-- Theorem stating that Smart Mart sold 45 science kits -/
theorem smart_mart_science_kits : science_kits_sold = 45 := by
  sorry

end NUMINAMATH_CALUDE_smart_mart_science_kits_l2500_250045


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2500_250076

theorem constant_term_expansion (x : ℝ) : 
  let expansion := (x + 2 / Real.sqrt x) ^ 6
  ∃ (coefficient : ℝ), coefficient = 240 ∧ 
    (∃ (other_terms : ℝ → ℝ), expansion = coefficient + other_terms x ∧ 
      (∀ y : ℝ, other_terms y ≠ 0 → y ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2500_250076


namespace NUMINAMATH_CALUDE_escalator_length_is_126_l2500_250085

/-- Calculates the length of an escalator given its speed, a person's walking speed on it, and the time taken to cover the entire length. -/
def escalator_length (escalator_speed : ℝ) (person_speed : ℝ) (time : ℝ) : ℝ :=
  (escalator_speed + person_speed) * time

/-- Proves that the length of the escalator is 126 feet under the given conditions. -/
theorem escalator_length_is_126 :
  escalator_length 11 3 9 = 126 := by
  sorry

#eval escalator_length 11 3 9

end NUMINAMATH_CALUDE_escalator_length_is_126_l2500_250085


namespace NUMINAMATH_CALUDE_symmetric_points_product_l2500_250055

/-- Two points (x₁, y₁) and (x₂, y₂) are symmetric with respect to the origin if x₁ = -x₂ and y₁ = -y₂ -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

theorem symmetric_points_product (a b : ℝ) :
  symmetric_wrt_origin (a + 2) 2 4 (-b) →
  a * b = -12 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_product_l2500_250055


namespace NUMINAMATH_CALUDE_hyperbola_equation_correct_l2500_250069

/-- A hyperbola is defined by its equation, asymptotes, and a point it passes through. -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form (x²/a²) - (y²/b²) = 1 -/
  equation : ℝ → ℝ → Prop
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- A point that the hyperbola passes through -/
  point : ℝ × ℝ

/-- The hyperbola satisfies its equation at the given point -/
def satisfies_equation (h : Hyperbola) : Prop :=
  h.equation h.point.1 h.point.2

/-- The asymptotes of the hyperbola have the correct slope -/
def has_correct_asymptotes (h : Hyperbola) : Prop :=
  h.asymptote_slope = 1 / 2

/-- Theorem stating that the given hyperbola equation is correct -/
theorem hyperbola_equation_correct (h : Hyperbola)
  (heq : h.equation = fun x y => x^2 / 8 - y^2 / 2 = 1)
  (hpoint : h.point = (4, Real.sqrt 2))
  (hslope : h.asymptote_slope = 1 / 2)
  : satisfies_equation h ∧ has_correct_asymptotes h := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_equation_correct_l2500_250069


namespace NUMINAMATH_CALUDE_base6_addition_l2500_250044

/-- Represents a number in base 6 as a list of digits (least significant first) -/
def Base6 := List Nat

/-- Addition of two base 6 numbers -/
def add_base6 (a b : Base6) : Base6 :=
  sorry

/-- Conversion of a natural number to base 6 -/
def to_base6 (n : Nat) : Base6 :=
  sorry

/-- Conversion of a base 6 number to a natural number -/
def from_base6 (b : Base6) : Nat :=
  sorry

theorem base6_addition :
  add_base6 [2, 3, 5, 4] [6, 4, 3, 5, 2] = [5, 2, 5, 2, 3] :=
sorry

end NUMINAMATH_CALUDE_base6_addition_l2500_250044


namespace NUMINAMATH_CALUDE_candy_bar_difference_l2500_250022

theorem candy_bar_difference : 
  ∀ (bob_candy : ℕ),
  let fred_candy : ℕ := 12
  let total_candy : ℕ := fred_candy + bob_candy
  let jacqueline_candy : ℕ := 10 * total_candy
  120 = (40 : ℕ) * jacqueline_candy / 100 →
  bob_candy - fred_candy = 6 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_difference_l2500_250022


namespace NUMINAMATH_CALUDE_circle_squares_inequality_l2500_250095

theorem circle_squares_inequality (x y : ℝ) : 
  abs x + abs y ≤ Real.sqrt (2 * (x^2 + y^2)) ∧ 
  Real.sqrt (2 * (x^2 + y^2)) ≤ 2 * max (abs x) (abs y) := by
  sorry

end NUMINAMATH_CALUDE_circle_squares_inequality_l2500_250095


namespace NUMINAMATH_CALUDE_cost_of_pens_l2500_250072

theorem cost_of_pens (box_size : ℕ) (box_cost : ℚ) (total_pens : ℕ) : 
  box_size = 150 → box_cost = 45 → total_pens = 4500 → 
  (total_pens : ℚ) * (box_cost / box_size) = 1350 := by
sorry

end NUMINAMATH_CALUDE_cost_of_pens_l2500_250072


namespace NUMINAMATH_CALUDE_aquarium_visitors_not_ill_l2500_250005

theorem aquarium_visitors_not_ill (total_visitors : ℕ) (ill_percentage : ℚ) : 
  total_visitors = 500 → 
  ill_percentage = 40 / 100 → 
  total_visitors - (total_visitors * ill_percentage).floor = 300 := by
sorry

end NUMINAMATH_CALUDE_aquarium_visitors_not_ill_l2500_250005


namespace NUMINAMATH_CALUDE_exam_score_calculation_l2500_250036

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) : 
  total_questions = 60 → 
  correct_answers = 34 → 
  marks_per_correct = 4 → 
  marks_lost_per_wrong = 1 → 
  (correct_answers * marks_per_correct) - 
  ((total_questions - correct_answers) * marks_lost_per_wrong) = 110 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l2500_250036


namespace NUMINAMATH_CALUDE_line_properties_l2500_250060

/-- A line passing through point A(4, -1) with equal intercepts on x and y axes --/
def line_with_equal_intercepts : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + p.2 = 3) ∨ (p.1 + 4 * p.2 = 0)}

/-- The point A(4, -1) --/
def point_A : ℝ × ℝ := (4, -1)

/-- Theorem stating that the line passes through point A and has equal intercepts --/
theorem line_properties :
  point_A ∈ line_with_equal_intercepts ∧
  ∃ a : ℝ, (a, 0) ∈ line_with_equal_intercepts ∧ (0, a) ∈ line_with_equal_intercepts :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l2500_250060


namespace NUMINAMATH_CALUDE_multiples_of_15_between_12_and_202_l2500_250058

theorem multiples_of_15_between_12_and_202 : 
  (Finset.filter (fun n => n % 15 = 0 ∧ n > 12 ∧ n < 202) (Finset.range 202)).card = 13 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_12_and_202_l2500_250058


namespace NUMINAMATH_CALUDE_min_points_dodecahedron_correct_min_points_icosahedron_correct_l2500_250011

/-- A dodecahedron is a polyhedron with 12 faces, where each face is a regular pentagon and each vertex belongs to 3 faces. -/
structure Dodecahedron where
  faces : ℕ
  faces_are_pentagons : Bool
  vertex_face_count : ℕ
  h_faces : faces = 12
  h_pentagons : faces_are_pentagons = true
  h_vertex : vertex_face_count = 3

/-- An icosahedron is a polyhedron with 20 faces and 12 vertices, where each face is an equilateral triangle. -/
structure Icosahedron where
  faces : ℕ
  vertices : ℕ
  faces_are_triangles : Bool
  h_faces : faces = 20
  h_vertices : vertices = 12
  h_triangles : faces_are_triangles = true

/-- The minimum number of points that must be marked on the surface of a dodecahedron
    so that there is at least one marked point on each face. -/
def min_points_dodecahedron (d : Dodecahedron) : ℕ := 4

/-- The minimum number of points that must be marked on the surface of an icosahedron
    so that there is at least one marked point on each face. -/
def min_points_icosahedron (i : Icosahedron) : ℕ := 6

/-- Theorem stating the minimum number of points for a dodecahedron. -/
theorem min_points_dodecahedron_correct (d : Dodecahedron) :
  min_points_dodecahedron d = 4 := by sorry

/-- Theorem stating the minimum number of points for an icosahedron. -/
theorem min_points_icosahedron_correct (i : Icosahedron) :
  min_points_icosahedron i = 6 := by sorry

end NUMINAMATH_CALUDE_min_points_dodecahedron_correct_min_points_icosahedron_correct_l2500_250011


namespace NUMINAMATH_CALUDE_trailing_zeros_of_square_l2500_250090

/-- The number of trailing zeros in (10^12 - 5)^2 is 12 -/
theorem trailing_zeros_of_square : ∃ n : ℕ, (10^12 - 5)^2 = n * 10^12 ∧ n % 10 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_square_l2500_250090


namespace NUMINAMATH_CALUDE_annes_distance_is_six_l2500_250014

/-- The distance traveled by Anne given her walking time and speed -/
def annes_distance (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Theorem stating that Anne's distance traveled is 6 miles -/
theorem annes_distance_is_six : annes_distance 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_annes_distance_is_six_l2500_250014


namespace NUMINAMATH_CALUDE_max_spheres_in_frustum_l2500_250043

/-- Represents a frustum with given height and two spheres placed inside it. -/
structure Frustum :=
  (height : ℝ)
  (sphere1_radius : ℝ)
  (sphere2_radius : ℝ)

/-- Calculates the maximum number of additional spheres that can be placed in the frustum. -/
def max_additional_spheres (f : Frustum) : ℕ :=
  sorry

/-- The main theorem stating the maximum number of additional spheres. -/
theorem max_spheres_in_frustum (f : Frustum) 
  (h1 : f.height = 8)
  (h2 : f.sphere1_radius = 2)
  (h3 : f.sphere2_radius = 3)
  : max_additional_spheres f = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_spheres_in_frustum_l2500_250043


namespace NUMINAMATH_CALUDE_paper_cutting_game_l2500_250083

theorem paper_cutting_game (n : ℕ) : 
  (8 * n + 1 = 2009) ↔ (n = 251) :=
by sorry

#check paper_cutting_game

end NUMINAMATH_CALUDE_paper_cutting_game_l2500_250083


namespace NUMINAMATH_CALUDE_no_alpha_sequence_exists_l2500_250027

theorem no_alpha_sequence_exists : ¬∃ (α : ℝ) (a : ℕ → ℝ), 
  (0 < α ∧ α < 1) ∧ 
  (∀ n, 0 < a n) ∧
  (∀ n, 1 + a (n + 1) ≤ a n + (α / n) * a n) := by
  sorry

end NUMINAMATH_CALUDE_no_alpha_sequence_exists_l2500_250027


namespace NUMINAMATH_CALUDE_problem_solution_l2500_250099

theorem problem_solution : ∃ x : ℝ, (x * 12) / (180 / 3) + 70 = 71 :=
by
  use 5
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2500_250099


namespace NUMINAMATH_CALUDE_book_purchase_savings_l2500_250013

theorem book_purchase_savings (full_price_book1 full_price_book2 : ℝ) : 
  full_price_book1 = 33 →
  full_price_book2 > 0 →
  let total_paid := full_price_book1 + (full_price_book2 / 2)
  let full_price := full_price_book1 + full_price_book2
  let savings_ratio := (full_price - total_paid) / full_price
  savings_ratio = 1/5 →
  full_price - total_paid = 11 :=
by sorry

end NUMINAMATH_CALUDE_book_purchase_savings_l2500_250013


namespace NUMINAMATH_CALUDE_upstream_speed_calculation_l2500_250093

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ  -- Speed in still water
  downstream : ℝ  -- Speed downstream

/-- Calculates the upstream speed given the rowing speeds in still water and downstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem stating that given the specific conditions, the upstream speed is 20 kmph -/
theorem upstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.stillWater = 40) 
  (h2 : s.downstream = 60) : 
  upstreamSpeed s = 20 := by
  sorry

#check upstream_speed_calculation

end NUMINAMATH_CALUDE_upstream_speed_calculation_l2500_250093


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2500_250017

/-- Given a geometric sequence with first term a₁ and common ratio q,
    if S₁, S₃, and 2a₃ form an arithmetic sequence, then q = -1/2 -/
theorem geometric_sequence_common_ratio
  (a₁ : ℝ) (q : ℝ) (S₁ S₃ : ℝ)
  (h₁ : S₁ = a₁)
  (h₂ : S₃ = a₁ + a₁ * q + a₁ * q^2)
  (h₃ : 2 * S₃ = S₁ + 2 * a₁ * q^2)
  (h₄ : a₁ ≠ 0) :
  q = -1/2 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2500_250017


namespace NUMINAMATH_CALUDE_lindas_lunchbox_theorem_l2500_250042

/-- Represents the cost calculation at Linda's Lunchbox -/
def lindas_lunchbox_cost (sandwich_price : ℝ) (soda_price : ℝ) (discount_rate : ℝ) 
  (discount_threshold : ℕ) (num_sandwiches : ℕ) (num_sodas : ℕ) : ℝ :=
  let total_items := num_sandwiches + num_sodas
  let subtotal := sandwich_price * num_sandwiches + soda_price * num_sodas
  if total_items ≥ discount_threshold then
    subtotal * (1 - discount_rate)
  else
    subtotal

/-- Theorem: The cost of 7 sandwiches and 5 sodas at Linda's Lunchbox is $38.7 -/
theorem lindas_lunchbox_theorem : 
  lindas_lunchbox_cost 4 3 0.1 10 7 5 = 38.7 := by
  sorry

end NUMINAMATH_CALUDE_lindas_lunchbox_theorem_l2500_250042


namespace NUMINAMATH_CALUDE_remaining_questions_to_write_l2500_250012

theorem remaining_questions_to_write
  (total_multiple_choice : ℕ)
  (total_problem_solving : ℕ)
  (total_true_false : ℕ)
  (fraction_multiple_choice_written : ℚ)
  (fraction_problem_solving_written : ℚ)
  (fraction_true_false_written : ℚ)
  (h1 : total_multiple_choice = 35)
  (h2 : total_problem_solving = 15)
  (h3 : total_true_false = 20)
  (h4 : fraction_multiple_choice_written = 3/7)
  (h5 : fraction_problem_solving_written = 1/5)
  (h6 : fraction_true_false_written = 1/4) :
  (total_multiple_choice - (fraction_multiple_choice_written * total_multiple_choice).num) +
  (total_problem_solving - (fraction_problem_solving_written * total_problem_solving).num) +
  (total_true_false - (fraction_true_false_written * total_true_false).num) = 47 :=
by sorry

end NUMINAMATH_CALUDE_remaining_questions_to_write_l2500_250012


namespace NUMINAMATH_CALUDE_jack_classics_books_jack_classics_total_l2500_250048

theorem jack_classics_books (num_authors : Nat) (books_per_author : Nat) : Nat :=
  num_authors * books_per_author

theorem jack_classics_total :
  jack_classics_books 6 33 = 198 := by
  sorry

end NUMINAMATH_CALUDE_jack_classics_books_jack_classics_total_l2500_250048


namespace NUMINAMATH_CALUDE_integer_expression_l2500_250038

theorem integer_expression (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
  (∃ m : ℕ, k * m = 3) ↔ 
  ∃ z : ℤ, z = (((n + 1)^2 - 3*k) / k^2) * (n.factorial / (k.factorial * (n - k).factorial)) :=
sorry

end NUMINAMATH_CALUDE_integer_expression_l2500_250038


namespace NUMINAMATH_CALUDE_max_value_product_l2500_250049

theorem max_value_product (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x/y + y/z + z/x + y/x + z/y + x/z = 9) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) ≤ 81/4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l2500_250049


namespace NUMINAMATH_CALUDE_new_average_weight_l2500_250080

def original_team_size : ℕ := 7
def original_team_avg_weight : ℚ := 94
def first_team_size : ℕ := 5
def first_team_avg_weight : ℚ := 100
def second_team_size : ℕ := 8
def second_team_avg_weight : ℚ := 90
def third_team_size : ℕ := 4
def third_team_avg_weight : ℚ := 120

theorem new_average_weight :
  let total_players := original_team_size + first_team_size + second_team_size + third_team_size
  let total_weight := original_team_size * original_team_avg_weight +
                      first_team_size * first_team_avg_weight +
                      second_team_size * second_team_avg_weight +
                      third_team_size * third_team_avg_weight
  (total_weight / total_players : ℚ) = 98.25 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l2500_250080


namespace NUMINAMATH_CALUDE_range_of_expression_l2500_250063

theorem range_of_expression (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) : 
  -π/6 < 2*α - β/3 ∧ 2*α - β/3 < π := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l2500_250063


namespace NUMINAMATH_CALUDE_problem_solution_l2500_250066

theorem problem_solution (x : ℝ) (h : x - Real.sqrt (x^2 - 4) + 1 / (x + Real.sqrt (x^2 - 4)) = 10) :
  x^2 - Real.sqrt (x^4 - 16) + 1 / (x^2 - Real.sqrt (x^4 - 16)) = 237/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2500_250066


namespace NUMINAMATH_CALUDE_bombardment_death_percentage_l2500_250026

/-- Represents the percentage of people who died by bombardment -/
def bombardment_percentage : ℝ := 10

/-- The initial population of the village -/
def initial_population : ℕ := 4200

/-- The final population after bombardment and departure -/
def final_population : ℕ := 3213

/-- The percentage of people who left after the bombardment -/
def departure_percentage : ℝ := 15

theorem bombardment_death_percentage :
  let remaining_after_bombardment := initial_population - (bombardment_percentage / 100) * initial_population
  let departed := (departure_percentage / 100) * remaining_after_bombardment
  initial_population - (bombardment_percentage / 100) * initial_population - departed = final_population :=
by sorry

end NUMINAMATH_CALUDE_bombardment_death_percentage_l2500_250026


namespace NUMINAMATH_CALUDE_box_area_is_2144_l2500_250057

/-- The surface area of a box formed by removing square corners from a rectangular sheet. -/
def box_surface_area (length width corner_size : ℕ) : ℕ :=
  length * width - 4 * (corner_size * corner_size)

/-- Theorem stating that the surface area of the box is 2144 square units. -/
theorem box_area_is_2144 :
  box_surface_area 60 40 8 = 2144 :=
by sorry

end NUMINAMATH_CALUDE_box_area_is_2144_l2500_250057


namespace NUMINAMATH_CALUDE_sphere_radius_when_area_equals_volume_l2500_250052

theorem sphere_radius_when_area_equals_volume (R : ℝ) : R > 0 →
  (4 * Real.pi * R^2 = (4 / 3) * Real.pi * R^3) → R = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_when_area_equals_volume_l2500_250052


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2500_250096

/-- The quadratic function f(x) = 2(x - 4)² + 6 has a minimum value of 6 -/
theorem quadratic_minimum (x : ℝ) : ∀ y : ℝ, 2 * (x - 4)^2 + 6 ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2500_250096


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l2500_250007

theorem boys_to_girls_ratio (total_students : ℕ) (girls : ℕ) 
  (h1 : girls = 135) (h2 : total_students = 351) :
  (total_students - girls) / girls = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l2500_250007
