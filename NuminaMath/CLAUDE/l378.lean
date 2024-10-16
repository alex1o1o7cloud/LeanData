import Mathlib

namespace NUMINAMATH_CALUDE_factor_expression_l378_37831

theorem factor_expression (a b c : ℝ) :
  ((a^2 + b^2)^3 + (b^2 + c^2)^3 + (c^2 + a^2)^3) / ((a + b)^3 + (b + c)^3 + (c + a)^3)
  = (a^2 + b^2) * (b^2 + c^2) * (c^2 + a^2) / ((a + b) * (b + c) * (c + a)) :=
by sorry

end NUMINAMATH_CALUDE_factor_expression_l378_37831


namespace NUMINAMATH_CALUDE_stating_interest_rate_calculation_l378_37811

/-- Represents the annual interest rate as a percentage -/
def annual_rate : ℝ := 15

/-- Represents the principal amount in rupees -/
def principal : ℝ := 147.69

/-- Represents the time period for the first deposit in years -/
def time1 : ℝ := 3.5

/-- Represents the time period for the second deposit in years -/
def time2 : ℝ := 10

/-- Represents the difference in interests in rupees -/
def interest_diff : ℝ := 144

/-- 
Theorem stating that given the conditions, the annual interest rate is approximately 15%.
-/
theorem interest_rate_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |annual_rate - (interest_diff * 100) / (principal * (time2 - time1))| < ε :=
sorry

end NUMINAMATH_CALUDE_stating_interest_rate_calculation_l378_37811


namespace NUMINAMATH_CALUDE_adam_bought_26_books_l378_37874

/-- The number of books Adam bought on his shopping trip -/
def books_bought (initial_books shelf_count books_per_shelf leftover_books : ℕ) : ℕ :=
  shelf_count * books_per_shelf + leftover_books - initial_books

/-- Theorem stating that Adam bought 26 books -/
theorem adam_bought_26_books : 
  books_bought 56 4 20 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_adam_bought_26_books_l378_37874


namespace NUMINAMATH_CALUDE_albert_oranges_l378_37864

/-- The number of boxes Albert has -/
def num_boxes : ℕ := 7

/-- The number of oranges in each box -/
def oranges_per_box : ℕ := 5

/-- The total number of oranges Albert has -/
def total_oranges : ℕ := num_boxes * oranges_per_box

theorem albert_oranges : total_oranges = 35 := by
  sorry

end NUMINAMATH_CALUDE_albert_oranges_l378_37864


namespace NUMINAMATH_CALUDE_three_lines_intersection_l378_37865

/-- Three distinct lines in 2D space -/
structure ThreeLines where
  a : ℝ
  b : ℝ
  l₁ : ℝ → ℝ → ℝ := λ x y => a * x + 2 * b * y + 3 * (a + b + 1)
  l₂ : ℝ → ℝ → ℝ := λ x y => b * x + 2 * (a + b + 1) * y + 3 * a
  l₃ : ℝ → ℝ → ℝ := λ x y => (a + b + 1) * x + 2 * a * y + 3 * b
  distinct : l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₃ ≠ l₁

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point lying on a line -/
def PointOnLine (p : Point) (l : ℝ → ℝ → ℝ) : Prop :=
  l p.x p.y = 0

/-- Definition of three lines intersecting at a single point -/
def IntersectAtSinglePoint (lines : ThreeLines) : Prop :=
  ∃! p : Point, PointOnLine p lines.l₁ ∧ PointOnLine p lines.l₂ ∧ PointOnLine p lines.l₃

/-- Theorem statement -/
theorem three_lines_intersection (lines : ThreeLines) :
  IntersectAtSinglePoint lines ↔ lines.a + lines.b = -1/2 := by sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l378_37865


namespace NUMINAMATH_CALUDE_irrational_approximation_l378_37871

theorem irrational_approximation (k : ℝ) (ε : ℝ) 
  (h_irr : Irrational k) (h_pos : ε > 0) :
  ∃ (m n : ℤ), |m * k - n| < ε :=
sorry

end NUMINAMATH_CALUDE_irrational_approximation_l378_37871


namespace NUMINAMATH_CALUDE_integer_root_values_l378_37886

def polynomial (x b : ℤ) : ℤ := x^3 + 2*x^2 + b*x + 8

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x b = 0

theorem integer_root_values :
  {b : ℤ | has_integer_root b} = {-81, -26, -12, -6, 4, 9, 47} := by sorry

end NUMINAMATH_CALUDE_integer_root_values_l378_37886


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l378_37836

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : ℕ
  is_convex : Bool
  right_angles : ℕ

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex 9-sided polygon with three right angles has 27 diagonals -/
theorem nonagon_diagonals (P : ConvexPolygon 9) (h1 : P.is_convex = true) (h2 : P.right_angles = 3) :
  num_diagonals P.sides = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l378_37836


namespace NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l378_37876

/-- Given points A(-1, 3) and B(2, 6), prove that P(5, 0) on the x-axis satisfies |PA| = |PB| -/
theorem equidistant_point_on_x_axis :
  let A : ℝ × ℝ := (-1, 3)
  let B : ℝ × ℝ := (2, 6)
  let P : ℝ × ℝ := (5, 0)
  (Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) = Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)) := by
  sorry

#check equidistant_point_on_x_axis

end NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l378_37876


namespace NUMINAMATH_CALUDE_cubic_roots_determinant_l378_37823

theorem cubic_roots_determinant (p q r : ℝ) (a b c : ℝ) : 
  (a^3 - p*a^2 + q*a - r = 0) →
  (b^3 - p*b^2 + q*b - r = 0) →
  (c^3 - p*c^2 + q*c - r = 0) →
  let matrix : Matrix (Fin 3) (Fin 3) ℝ := !![a, 1, 1; 1, b, 1; 1, 1, c]
  Matrix.det matrix = r - p + 2 := by sorry

end NUMINAMATH_CALUDE_cubic_roots_determinant_l378_37823


namespace NUMINAMATH_CALUDE_uncle_ben_eggs_l378_37893

theorem uncle_ben_eggs (total_chickens roosters non_laying_hens eggs_per_hen : ℕ) 
  (h1 : total_chickens = 440)
  (h2 : roosters = 39)
  (h3 : non_laying_hens = 15)
  (h4 : eggs_per_hen = 3) :
  total_chickens - roosters - non_laying_hens = 386 →
  (total_chickens - roosters - non_laying_hens) * eggs_per_hen = 1158 := by
  sorry

end NUMINAMATH_CALUDE_uncle_ben_eggs_l378_37893


namespace NUMINAMATH_CALUDE_annular_ring_area_l378_37807

/-- Given a circle and a chord AB divided by point C such that AC = a and BC = b,
    the area of the annular ring formed when C traces another circle as AB's position changes
    is π(a + b)²/4. -/
theorem annular_ring_area (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let chord_length := a + b
  ∃ (R : ℝ), R > chord_length / 2 →
    (π * (chord_length ^ 2) / 4 : ℝ) =
      π * R ^ 2 - π * (R ^ 2 - chord_length ^ 2 / 4) :=
by sorry

end NUMINAMATH_CALUDE_annular_ring_area_l378_37807


namespace NUMINAMATH_CALUDE_polynomial_simplification_l378_37895

theorem polynomial_simplification (x : ℝ) : 
  x * (4 * x^2 - 2) - 5 * (x^2 - 3*x + 5) = 4 * x^3 - 5 * x^2 + 13 * x - 25 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l378_37895


namespace NUMINAMATH_CALUDE_age_difference_eighteen_l378_37817

/-- Represents a person's age as a two-digit number -/
structure Age :=
  (tens : Nat)
  (ones : Nat)
  (h_odd_tens : Odd tens)
  (h_odd_ones : Odd ones)
  (h_range_tens : tens ≤ 9)
  (h_range_ones : ones ≤ 9)

/-- The age as a natural number -/
def Age.toNat (a : Age) : Nat := 10 * a.tens + a.ones

theorem age_difference_eighteen :
  ∀ (alice bob : Age),
    alice.tens = bob.ones ∧ 
    alice.ones = bob.tens ∧
    (alice.toNat + 7 = 3 * (bob.toNat + 7)) →
    bob.toNat - alice.toNat = 18 := by
  sorry

#check age_difference_eighteen

end NUMINAMATH_CALUDE_age_difference_eighteen_l378_37817


namespace NUMINAMATH_CALUDE_daniels_candies_l378_37810

theorem daniels_candies (x : ℕ) : 
  (x : ℚ) * 3/8 - 3/2 - 6 = 10 ↔ x = 93 := by
  sorry

end NUMINAMATH_CALUDE_daniels_candies_l378_37810


namespace NUMINAMATH_CALUDE_chord_length_theorem_l378_37890

/-- Represents a circle with a given radius and center point -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

/-- Checks if a smaller circle is internally tangent to a larger circle -/
def is_internally_tangent (small large : Circle) : Prop :=
  (small.center.1 - large.center.1)^2 + (small.center.2 - large.center.2)^2 = (large.radius - small.radius)^2

/-- Represents the common external tangent chord length -/
def common_external_tangent_chord_length_squared (c1 c2 c3 : Circle) : ℝ := 72

theorem chord_length_theorem (c1 c2 c3 : Circle)
  (h1 : c1.radius = 3)
  (h2 : c2.radius = 6)
  (h3 : c3.radius = 9)
  (h4 : are_externally_tangent c1 c2)
  (h5 : is_internally_tangent c1 c3)
  (h6 : is_internally_tangent c2 c3) :
  common_external_tangent_chord_length_squared c1 c2 c3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_theorem_l378_37890


namespace NUMINAMATH_CALUDE_average_stickers_per_pack_l378_37887

def sticker_counts : List ℕ := [5, 7, 7, 10, 11]

def num_packs : ℕ := 5

theorem average_stickers_per_pack :
  (sticker_counts.sum / num_packs : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_stickers_per_pack_l378_37887


namespace NUMINAMATH_CALUDE_proportional_function_quadrants_l378_37867

/-- A proportional function passing through quadrants II and IV -/
theorem proportional_function_quadrants :
  ∀ (x y : ℝ), y = -2 * x →
  (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0) := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_quadrants_l378_37867


namespace NUMINAMATH_CALUDE_packaging_waste_exceeds_target_l378_37841

/-- The year when the packaging waste exceeds 40 million tons -/
def exceed_year : ℕ := 2021

/-- The initial packaging waste in 2015 (in million tons) -/
def initial_waste : ℝ := 4

/-- The annual growth rate of packaging waste -/
def growth_rate : ℝ := 0.5

/-- The target waste amount to exceed (in million tons) -/
def target_waste : ℝ := 40

/-- Function to calculate the waste amount after n years -/
def waste_after_years (n : ℕ) : ℝ :=
  initial_waste * (1 + growth_rate) ^ n

theorem packaging_waste_exceeds_target :
  waste_after_years (exceed_year - 2015) > target_waste ∧
  ∀ y : ℕ, y < exceed_year - 2015 → waste_after_years y ≤ target_waste :=
by sorry

end NUMINAMATH_CALUDE_packaging_waste_exceeds_target_l378_37841


namespace NUMINAMATH_CALUDE_sum_of_roots_l378_37805

theorem sum_of_roots (c d : ℝ) 
  (hc : c^3 - 21*c^2 + 28*c - 70 = 0) 
  (hd : 10*d^3 - 75*d^2 - 350*d + 3225 = 0) : 
  c + d = 21/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l378_37805


namespace NUMINAMATH_CALUDE_only_48_satisfies_l378_37829

/-- A function that returns the digits of a positive integer -/
def digits (n : ℕ+) : List ℕ :=
  sorry

/-- A function that checks if all elements in a list are between 1 and 9 (inclusive) -/
def all_between_1_and_9 (l : List ℕ) : Prop :=
  sorry

/-- The main theorem -/
theorem only_48_satisfies : ∃! (n : ℕ+),
  (n : ℕ) = (3/2 : ℚ) * (digits n).prod ∧
  all_between_1_and_9 (digits n) :=
by
  sorry

end NUMINAMATH_CALUDE_only_48_satisfies_l378_37829


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_max_profit_at_optimal_price_profit_function_is_quadratic_l378_37854

/-- Represents the profit function for a product sale scenario -/
def profit_function (x : ℝ) : ℝ :=
  (x - 8) * (100 - 10 * (x - 10))

/-- The optimal selling price that maximizes profit -/
def optimal_price : ℝ := 14

/-- The maximum daily profit -/
def max_profit : ℝ := 360

/-- Theorem stating that the optimal price maximizes the profit function -/
theorem optimal_price_maximizes_profit :
  ∀ x, x > 10 → profit_function x ≤ profit_function optimal_price :=
sorry

/-- Theorem stating that the maximum profit is achieved at the optimal price -/
theorem max_profit_at_optimal_price :
  profit_function optimal_price = max_profit :=
sorry

/-- Theorem stating that the profit function is a quadratic function -/
theorem profit_function_is_quadratic :
  ∃ a b c, ∀ x, profit_function x = a * x^2 + b * x + c ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_max_profit_at_optimal_price_profit_function_is_quadratic_l378_37854


namespace NUMINAMATH_CALUDE_min_people_needed_is_30_l378_37849

/-- Represents the types of vehicles --/
inductive VehicleType
| SmallCar
| MediumCar
| LargeCar
| LightTruck
| HeavyTruck

/-- Returns the weight of a vehicle type in pounds --/
def vehicleWeight (v : VehicleType) : ℕ :=
  match v with
  | .SmallCar => 2000
  | .MediumCar => 3000
  | .LargeCar => 4000
  | .LightTruck => 10000
  | .HeavyTruck => 15000

/-- Represents the fleet of vehicles --/
def fleet : List (VehicleType × ℕ) :=
  [(VehicleType.SmallCar, 2), (VehicleType.MediumCar, 2), (VehicleType.LargeCar, 2),
   (VehicleType.LightTruck, 1), (VehicleType.HeavyTruck, 2)]

/-- The maximum lifting capacity of a person in pounds --/
def maxLiftingCapacity : ℕ := 1000

/-- Calculates the total weight of the fleet --/
def totalFleetWeight : ℕ :=
  fleet.foldl (fun acc (v, count) => acc + vehicleWeight v * count) 0

/-- Theorem: The minimum number of people needed to lift all vehicles is 30 --/
theorem min_people_needed_is_30 :
  ∃ (n : ℕ), n = 30 ∧
  n * maxLiftingCapacity ≥ totalFleetWeight ∧
  ∀ (m : ℕ), m * maxLiftingCapacity ≥ totalFleetWeight → m ≥ n :=
sorry

end NUMINAMATH_CALUDE_min_people_needed_is_30_l378_37849


namespace NUMINAMATH_CALUDE_sum_has_five_digits_l378_37806

/-- A nonzero digit is a natural number between 1 and 9, inclusive. -/
def NonzeroDigit : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- The function that constructs the second number (A75) from a nonzero digit A. -/
def secondNumber (A : NonzeroDigit) : ℕ := A.val * 100 + 75

/-- The function that constructs the third number (5B2) from a nonzero digit B. -/
def thirdNumber (B : NonzeroDigit) : ℕ := 500 + B.val * 10 + 2

/-- The theorem stating that the sum of the three numbers always has 5 digits. -/
theorem sum_has_five_digits (A B : NonzeroDigit) :
  ∃ n : ℕ, 10000 ≤ 9643 + secondNumber A + thirdNumber B ∧
           9643 + secondNumber A + thirdNumber B < 100000 := by
  sorry

end NUMINAMATH_CALUDE_sum_has_five_digits_l378_37806


namespace NUMINAMATH_CALUDE_sequence_properties_l378_37878

def S (n : ℕ) : ℤ := -n^2 + 7*n

def a (n : ℕ) : ℤ := -2*n + 8

theorem sequence_properties :
  (∀ n : ℕ, S n = -n^2 + 7*n) →
  (∀ n : ℕ, n ≥ 2 → S n - S (n-1) = a n) ∧
  (∀ n : ℕ, n > 4 → a n < 0) ∧
  (∀ n : ℕ, n ≠ 3 ∧ n ≠ 4 → S n ≤ S 3 ∧ S n ≤ S 4) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l378_37878


namespace NUMINAMATH_CALUDE_range_of_function_l378_37857

theorem range_of_function (x : ℝ) : 
  (1/2 : ℝ) ≤ ((14 * Real.cos (2 * x) + 28 * Real.sin x + 15) * Real.pi / 108) ∧ 
  ((14 * Real.cos (2 * x) + 28 * Real.sin x + 15) * Real.pi / 108) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l378_37857


namespace NUMINAMATH_CALUDE_magical_stack_with_157_has_470_cards_l378_37869

/-- Represents a stack of cards -/
structure CardStack :=
  (total : ℕ)  -- Total number of cards
  (m : ℕ)      -- Number of cards in each pile
  (isMagical : Bool)  -- Whether the stack is magical

/-- Defines the conditions for a magical stack -/
def isMagicalStack (stack : CardStack) : Prop :=
  stack.total = 2 * stack.m ∧
  stack.isMagical = true ∧
  ∃ (card : ℕ), card ≤ stack.total ∧ card = 157 ∧
    (card % 2 = 1 → card ≤ stack.m) ∧
    (card % 2 = 0 → card > stack.m)

/-- The main theorem to prove -/
theorem magical_stack_with_157_has_470_cards :
  ∀ (stack : CardStack), isMagicalStack stack →
  (157 ≤ stack.m ∧ 157 + (156 / 2) = stack.m) →
  stack.total = 470 := by
  sorry

end NUMINAMATH_CALUDE_magical_stack_with_157_has_470_cards_l378_37869


namespace NUMINAMATH_CALUDE_probability_point_in_circle_l378_37800

/-- The probability of a randomly selected point in a square with side length 6 
    being within 2 units of the center is π/9. -/
theorem probability_point_in_circle (square_side : ℝ) (circle_radius : ℝ) : 
  square_side = 6 → circle_radius = 2 → 
  (π * circle_radius^2) / (square_side^2) = π / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_in_circle_l378_37800


namespace NUMINAMATH_CALUDE_range_of_k_l378_37808

/-- Given x ∈ (0, 2), prove that x/(e^x) < 1/(k + 2x - x^2) holds if and only if k ∈ [0, e-1) -/
theorem range_of_k (x : ℝ) (hx : x ∈ Set.Ioo 0 2) :
  (∀ k : ℝ, x / Real.exp x < 1 / (k + 2 * x - x^2)) ↔ k ∈ Set.Icc 0 (Real.exp 1 - 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l378_37808


namespace NUMINAMATH_CALUDE_number_problem_l378_37872

theorem number_problem (X Y Z : ℝ) 
  (h1 : X - Y = 3500)
  (h2 : (3/5) * X = (2/3) * Y)
  (h3 : 0.097 * Y = Real.sqrt Z) :
  X = 35000 ∧ Y = 31500 ∧ Z = 9333580.25 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l378_37872


namespace NUMINAMATH_CALUDE_couponA_provides_greatest_discount_l378_37863

-- Define the coupon discount functions
def couponA (price : Real) : Real := 0.12 * price

def couponB (price : Real) : Real := 25

def couponC (price : Real) : Real := 0.15 * (price - 150)

def couponD (price : Real) : Real := 0.1 * price + 13.5

-- Define the listed price
def listedPrice : Real := 229.95

-- Theorem statement
theorem couponA_provides_greatest_discount :
  couponA listedPrice > couponB listedPrice ∧
  couponA listedPrice > couponC listedPrice ∧
  couponA listedPrice > couponD listedPrice := by
  sorry

end NUMINAMATH_CALUDE_couponA_provides_greatest_discount_l378_37863


namespace NUMINAMATH_CALUDE_pitcher_problem_l378_37885

theorem pitcher_problem (C : ℝ) (h : C > 0) :
  let juice_in_pitcher := (5 / 6) * C
  let juice_per_cup := juice_in_pitcher / 3
  juice_per_cup / C = 5 / 18 := by
sorry

end NUMINAMATH_CALUDE_pitcher_problem_l378_37885


namespace NUMINAMATH_CALUDE_sin_half_theta_l378_37819

theorem sin_half_theta (θ : Real) 
  (h1 : |Real.cos θ| = 1/5)
  (h2 : 5/2 * Real.pi < θ) 
  (h3 : θ < 3 * Real.pi) : 
  Real.sin (θ/2) = -Real.sqrt 15 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_half_theta_l378_37819


namespace NUMINAMATH_CALUDE_numbers_less_than_reciprocals_l378_37891

theorem numbers_less_than_reciprocals :
  let numbers : List ℚ := [-1/2, -3, 1/4, 4, 1/3]
  ∀ x ∈ numbers, (x < 1 / x) ↔ (x = -3 ∨ x = 1/4 ∨ x = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_numbers_less_than_reciprocals_l378_37891


namespace NUMINAMATH_CALUDE_dressing_ratio_l378_37833

def ranch_cases : ℕ := 28
def caesar_cases : ℕ := 4

theorem dressing_ratio : 
  (ranch_cases / caesar_cases : ℚ) = 7 / 1 := by
  sorry

end NUMINAMATH_CALUDE_dressing_ratio_l378_37833


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l378_37801

/-- A cyclic quadrilateral with angles in arithmetic progression -/
structure CyclicQuadrilateral where
  -- The smallest angle
  a : ℝ
  -- The common difference in the arithmetic progression
  d : ℝ
  -- Ensures the quadrilateral is cyclic (opposite angles sum to 180°)
  cyclic : a + (a + 3*d) = 180 ∧ (a + d) + (a + 2*d) = 180
  -- Ensures the angles form an arithmetic sequence
  arithmetic_seq : true
  -- The largest angle is 140°
  largest_angle : a + 3*d = 140

/-- 
In a cyclic quadrilateral where the angles form an arithmetic sequence 
and the largest angle is 140°, the smallest angle measures 40°
-/
theorem smallest_angle_measure (q : CyclicQuadrilateral) : q.a = 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l378_37801


namespace NUMINAMATH_CALUDE_salt_solution_volume_l378_37809

/-- Proves that given a solution with an initial salt concentration of 10%,
    if adding 18 gallons of water reduces the salt concentration to 8%,
    then the initial volume of the solution must be 72 gallons. -/
theorem salt_solution_volume 
  (initial_concentration : ℝ) 
  (final_concentration : ℝ) 
  (water_added : ℝ) 
  (initial_volume : ℝ) :
  initial_concentration = 0.10 →
  final_concentration = 0.08 →
  water_added = 18 →
  initial_concentration * initial_volume = 
    final_concentration * (initial_volume + water_added) →
  initial_volume = 72 :=
by sorry

end NUMINAMATH_CALUDE_salt_solution_volume_l378_37809


namespace NUMINAMATH_CALUDE_dollar_three_minus_four_l378_37879

-- Define the custom operation $
def dollar (x y : Int) : Int :=
  x * (y + 2) + x * y + x

-- Theorem statement
theorem dollar_three_minus_four : dollar 3 (-4) = -15 := by
  sorry

end NUMINAMATH_CALUDE_dollar_three_minus_four_l378_37879


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l378_37898

theorem isosceles_triangle_vertex_angle (α β : ℝ) : 
  α = 50 → -- base angle is 50°
  β = 180 - 2*α → -- vertex angle formula
  β = 80 -- vertex angle is 80°
:= by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l378_37898


namespace NUMINAMATH_CALUDE_chloe_profit_l378_37832

/-- Calculates Chloe's profit from selling chocolate-dipped strawberries during a 3-day Mother's Day celebration. -/
theorem chloe_profit (buy_price : ℝ) (sell_price : ℝ) (bulk_discount : ℝ) 
  (min_production_cost : ℝ) (max_production_cost : ℝ) 
  (day1_price_factor : ℝ) (day2_price_factor : ℝ) (day3_price_factor : ℝ)
  (total_dozens : ℕ) (day1_dozens : ℕ) (day2_dozens : ℕ) (day3_dozens : ℕ) :
  buy_price = 50 →
  sell_price = 60 →
  bulk_discount = 0.1 →
  min_production_cost = 40 →
  max_production_cost = 45 →
  day1_price_factor = 1 →
  day2_price_factor = 1.2 →
  day3_price_factor = 0.85 →
  total_dozens = 50 →
  day1_dozens = 12 →
  day2_dozens = 18 →
  day3_dozens = 20 →
  total_dozens ≥ 10 →
  day1_dozens + day2_dozens + day3_dozens = total_dozens →
  ∃ profit : ℝ, profit = 152 ∧ 
    profit = (day1_dozens * sell_price * day1_price_factor +
              day2_dozens * sell_price * day2_price_factor +
              day3_dozens * sell_price * day3_price_factor) * (1 - bulk_discount) -
             total_dozens * (min_production_cost + max_production_cost) / 2 :=
by sorry

end NUMINAMATH_CALUDE_chloe_profit_l378_37832


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l378_37835

theorem imaginary_part_of_z (z : ℂ) (h : (1 + z) / Complex.I = 1 - z) : 
  Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l378_37835


namespace NUMINAMATH_CALUDE_solution_set_1_correct_solution_set_2_correct_l378_37860

-- Define the solution set for the first inequality
def solution_set_1 : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 4}

-- Define the solution set for the second inequality
def solution_set_2 (a : ℝ) : Set ℝ :=
  if a = 0 then Set.univ
  else if a > 0 then {x : ℝ | x ≥ a - 1 ∨ x ≤ -a - 1}
  else {x : ℝ | x ≥ -a - 1 ∨ x ≤ a - 1}

-- Theorem for the first inequality
theorem solution_set_1_correct :
  ∀ x : ℝ, x ∈ solution_set_1 ↔ -x^2 + 3*x + 4 ≥ 0 := by sorry

-- Theorem for the second inequality
theorem solution_set_2_correct :
  ∀ a x : ℝ, x ∈ solution_set_2 a ↔ x^2 + 2*x + (1-a)*(1+a) ≥ 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_1_correct_solution_set_2_correct_l378_37860


namespace NUMINAMATH_CALUDE_hilt_share_money_l378_37839

/-- The number of people Mrs. Hilt will share the money with -/
def number_of_people (total_amount : ℚ) (amount_per_person : ℚ) : ℚ :=
  total_amount / amount_per_person

/-- Theorem stating that Mrs. Hilt will share the money with 3 people -/
theorem hilt_share_money : 
  number_of_people (3.75 : ℚ) (1.25 : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_hilt_share_money_l378_37839


namespace NUMINAMATH_CALUDE_not_square_sum_ceil_l378_37881

theorem not_square_sum_ceil (a b : ℕ+) : ¬∃ k : ℤ, (a : ℤ)^2 + ⌈(4 * (a : ℤ)^2) / (b : ℤ)⌉ = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_square_sum_ceil_l378_37881


namespace NUMINAMATH_CALUDE_same_number_probability_l378_37848

-- Define the number of sides on each die
def sides : ℕ := 8

-- Define the number of dice
def num_dice : ℕ := 4

-- Theorem stating the probability of all dice showing the same number
theorem same_number_probability : 
  (1 : ℚ) / (sides ^ (num_dice - 1)) = 1 / 512 :=
sorry

end NUMINAMATH_CALUDE_same_number_probability_l378_37848


namespace NUMINAMATH_CALUDE_desired_circle_properties_l378_37855

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0
def l (x y : ℝ) : Prop := x + 2*y = 0

-- Define the desired circle
def desiredCircle (x y : ℝ) : Prop := x^2 + y^2 - x - 2*y = 0

-- Theorem statement
theorem desired_circle_properties :
  ∀ (x y : ℝ),
    (C₁ x y ∧ C₂ x y → desiredCircle x y) ∧
    (∃ (x₀ y₀ : ℝ), desiredCircle x₀ y₀ ∧ l x₀ y₀ ∧
      ∀ (x y : ℝ), (x - x₀)^2 + (y - y₀)^2 < ε → ¬(desiredCircle x y ∧ l x y))
    := by sorry


end NUMINAMATH_CALUDE_desired_circle_properties_l378_37855


namespace NUMINAMATH_CALUDE_airplane_seats_l378_37803

theorem airplane_seats (total_seats : ℕ) (first_class : ℕ) (coach : ℕ) 
  (h1 : total_seats = 387)
  (h2 : first_class + coach = total_seats)
  (h3 : coach = 4 * first_class + 2) :
  first_class = 77 := by
  sorry

end NUMINAMATH_CALUDE_airplane_seats_l378_37803


namespace NUMINAMATH_CALUDE_sum_of_digits_3n_l378_37851

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: If n has sum of digits 100 and 44n has sum of digits 800, then 3n has sum of digits 300 -/
theorem sum_of_digits_3n 
  (n : ℕ) 
  (h1 : sum_of_digits n = 100) 
  (h2 : sum_of_digits (44 * n) = 800) : 
  sum_of_digits (3 * n) = 300 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_3n_l378_37851


namespace NUMINAMATH_CALUDE_business_investment_l378_37883

theorem business_investment (A B : ℕ) (t : ℕ) (r : ℚ) : 
  A = 45000 →
  t = 2 →
  r = 2 / 1 →
  (A * t : ℚ) / (B * t : ℚ) = r →
  B = 22500 := by
  sorry

end NUMINAMATH_CALUDE_business_investment_l378_37883


namespace NUMINAMATH_CALUDE_strange_number_theorem_l378_37820

theorem strange_number_theorem : ∃! x : ℝ, (x - 7) * 7 = (x - 11) * 11 := by
  sorry

end NUMINAMATH_CALUDE_strange_number_theorem_l378_37820


namespace NUMINAMATH_CALUDE_reaction_theorem_l378_37877

/-- Represents the bond enthalpy values in kJ/mol -/
structure BondEnthalpy where
  oh : ℝ
  hh : ℝ
  nah : ℝ
  ona : ℝ

/-- Calculates the amount of water required and enthalpy change for the reaction -/
def reaction_calculation (bond_enthalpies : BondEnthalpy) : ℝ × ℝ :=
  let water_amount := 2
  let enthalpy_change := 
    2 * bond_enthalpies.nah + 2 * 2 * bond_enthalpies.oh -
    (2 * bond_enthalpies.ona + 2 * bond_enthalpies.hh)
  (water_amount, enthalpy_change)

/-- Theorem stating the correctness of the reaction calculation -/
theorem reaction_theorem (bond_enthalpies : BondEnthalpy) 
  (h_oh : bond_enthalpies.oh = 463)
  (h_hh : bond_enthalpies.hh = 432)
  (h_nah : bond_enthalpies.nah = 283)
  (h_ona : bond_enthalpies.ona = 377) :
  reaction_calculation bond_enthalpies = (2, 800) := by
  sorry

end NUMINAMATH_CALUDE_reaction_theorem_l378_37877


namespace NUMINAMATH_CALUDE_x_root_of_quadratic_with_integer_coeff_l378_37870

/-- Given distinct real numbers x and y with equal fractional parts and equal fractional parts of their cubes,
    x is a root of a quadratic equation with integer coefficients. -/
theorem x_root_of_quadratic_with_integer_coeff
  (x y : ℝ)
  (h_distinct : x ≠ y)
  (h_frac_eq : x - ⌊x⌋ = y - ⌊y⌋)
  (h_frac_cube_eq : x^3 - ⌊x^3⌋ = y^3 - ⌊y^3⌋) :
  ∃ (a b c : ℤ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧ (a * x^2 + b * x + c : ℝ) = 0 :=
sorry

end NUMINAMATH_CALUDE_x_root_of_quadratic_with_integer_coeff_l378_37870


namespace NUMINAMATH_CALUDE_taxi_fare_equality_l378_37897

/-- Taxi fare calculation problem -/
theorem taxi_fare_equality (mike_base_fare annie_base_fare toll_fee : ℚ)
  (per_mile_rate : ℚ) (annie_miles : ℕ) :
  mike_base_fare = 2.5 ∧
  annie_base_fare = 2.5 ∧
  toll_fee = 5 ∧
  per_mile_rate = 0.25 ∧
  annie_miles = 16 →
  ∃ (mike_miles : ℕ),
    mike_base_fare + per_mile_rate * mike_miles =
    annie_base_fare + toll_fee + per_mile_rate * annie_miles ∧
    mike_miles = 36 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_equality_l378_37897


namespace NUMINAMATH_CALUDE_group_size_l378_37813

theorem group_size (average_increase : ℝ) (old_weight new_weight : ℝ) :
  average_increase = 2.5 ∧
  old_weight = 40 ∧
  new_weight = 60 →
  (new_weight - old_weight) / average_increase = 8 := by
sorry

end NUMINAMATH_CALUDE_group_size_l378_37813


namespace NUMINAMATH_CALUDE_cube_surface_area_ratio_l378_37822

theorem cube_surface_area_ratio :
  let original_volume : ℝ := 1000
  let removed_volume : ℝ := 64
  let original_side : ℝ := original_volume ^ (1/3)
  let removed_side : ℝ := removed_volume ^ (1/3)
  let shaded_area : ℝ := removed_side ^ 2
  let total_surface_area : ℝ := 
    3 * original_side ^ 2 + 
    3 * removed_side ^ 2 + 
    3 * (original_side ^ 2 - removed_side ^ 2)
  shaded_area / total_surface_area = 2 / 75
  := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_ratio_l378_37822


namespace NUMINAMATH_CALUDE_fall_spending_calculation_l378_37845

/-- Represents the spending of River Town government in millions of dollars -/
structure RiverTownSpending where
  july_start : ℝ
  october_end : ℝ

/-- Calculates the spending during September and October -/
def fall_spending (s : RiverTownSpending) : ℝ :=
  s.october_end - s.july_start

/-- Theorem stating that for the given spending data, the fall spending is 3.4 million dollars -/
theorem fall_spending_calculation (s : RiverTownSpending) 
  (h1 : s.july_start = 3.1)
  (h2 : s.october_end = 6.5) : 
  fall_spending s = 3.4 := by
  sorry

#eval fall_spending { july_start := 3.1, october_end := 6.5 }

end NUMINAMATH_CALUDE_fall_spending_calculation_l378_37845


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l378_37875

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 4)) ↔ x ≠ 4 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l378_37875


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l378_37844

/-- A point in the Cartesian plane is in the fourth quadrant if its x-coordinate is positive and its y-coordinate is negative -/
def fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

/-- The point (2, -4) is in the fourth quadrant -/
theorem point_in_fourth_quadrant : fourth_quadrant (2, -4) := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l378_37844


namespace NUMINAMATH_CALUDE_infection_probability_l378_37847

theorem infection_probability (malaria_rate : Real) (zika_rate : Real) 
  (vaccine_effectiveness : Real) (overall_infection_rate : Real) :
  malaria_rate = 0.40 →
  zika_rate = 0.20 →
  vaccine_effectiveness = 0.50 →
  overall_infection_rate = 0.15 →
  ∃ (p : Real), 
    p = overall_infection_rate / (malaria_rate * vaccine_effectiveness + zika_rate) ∧
    p = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_infection_probability_l378_37847


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l378_37862

def A : Set ℕ := {0, 1, 3, 5, 7}
def B : Set ℕ := {2, 4, 6, 8, 0}

theorem intersection_of_A_and_B : A ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l378_37862


namespace NUMINAMATH_CALUDE_fourth_term_is_eight_l378_37852

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  sum_first_three : a 1 + a 2 + a 3 = 7
  product_first_three : a 1 * a 2 * a 3 = 8
  increasing : ∀ n : ℕ, a n < a (n + 1)

/-- The fourth term of the geometric sequence is 8 -/
theorem fourth_term_is_eight (seq : GeometricSequence) : seq.a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_eight_l378_37852


namespace NUMINAMATH_CALUDE_balanced_quadruple_inequality_l378_37827

/-- A quadruple of real numbers is balanced if the sum of its elements
    equals the sum of their squares. -/
def IsBalanced (a b c d : ℝ) : Prop :=
  a + b + c + d = a^2 + b^2 + c^2 + d^2

/-- For any positive real number x greater than or equal to 3/2,
    the product (x - a)(x - b)(x - c)(x - d) is non-negative
    for all balanced quadruples (a, b, c, d). -/
theorem balanced_quadruple_inequality (x : ℝ) (hx : x > 0) (hx_ge : x ≥ 3/2) :
  ∀ a b c d : ℝ, IsBalanced a b c d →
  (x - a) * (x - b) * (x - c) * (x - d) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_balanced_quadruple_inequality_l378_37827


namespace NUMINAMATH_CALUDE_green_then_blue_probability_l378_37889

/-- The probability of drawing a green marble first and a blue marble second from a bag -/
theorem green_then_blue_probability 
  (total_marbles : ℕ) 
  (blue_marbles : ℕ) 
  (green_marbles : ℕ) 
  (h1 : total_marbles = blue_marbles + green_marbles)
  (h2 : blue_marbles = 4)
  (h3 : green_marbles = 6) :
  (green_marbles : ℚ) / total_marbles * (blue_marbles : ℚ) / (total_marbles - 1) = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_green_then_blue_probability_l378_37889


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_max_min_values_even_function_l378_37812

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x + 3

-- Part 1
theorem monotonic_increasing_interval (a : ℝ) (h : a = 2) :
  ∀ x y, x ≤ y ∧ y ≤ 1 → f a x ≤ f a y :=
sorry

-- Part 2
theorem max_min_values_even_function (a : ℝ) 
  (h : ∀ x, f a x = f a (-x)) :
  (∃ x₀ ∈ Set.Icc (-1) 3, f a x₀ = 3 ∧ 
    ∀ x ∈ Set.Icc (-1) 3, f a x ≤ 3) ∧
  (∃ x₁ ∈ Set.Icc (-1) 3, f a x₁ = -6 ∧ 
    ∀ x ∈ Set.Icc (-1) 3, f a x ≥ -6) :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_max_min_values_even_function_l378_37812


namespace NUMINAMATH_CALUDE_seven_representations_l378_37816

/-- An arithmetic expression using digits, operations, and parentheses -/
inductive ArithExpr
  | Digit (d : ℕ)
  | Add (e1 e2 : ArithExpr)
  | Sub (e1 e2 : ArithExpr)
  | Mul (e1 e2 : ArithExpr)
  | Div (e1 e2 : ArithExpr)

/-- Count the number of times a specific digit appears in an ArithExpr -/
def countDigit (e : ArithExpr) (d : ℕ) : ℕ := sorry

/-- Evaluate an ArithExpr to a rational number -/
def evaluate (e : ArithExpr) : ℚ := sorry

/-- Theorem: For each integer n from 1 to 10 inclusive, there exists an arithmetic
    expression using the digit 7 exactly four times, along with operation signs
    and parentheses, that evaluates to n. -/
theorem seven_representations :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 10 →
  ∃ e : ArithExpr, countDigit e 7 = 4 ∧ evaluate e = n := by sorry

end NUMINAMATH_CALUDE_seven_representations_l378_37816


namespace NUMINAMATH_CALUDE_cubic_root_sum_l378_37868

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 10*p - 3 = 0 ∧ 
  q^3 - 8*q^2 + 10*q - 3 = 0 ∧ 
  r^3 - 8*r^2 + 10*r - 3 = 0 →
  p / (q*r + 2) + q / (p*r + 2) + r / (p*q + 2) = 8/69 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l378_37868


namespace NUMINAMATH_CALUDE_calculate_expression_l378_37856

theorem calculate_expression : 
  let a := (5 + 5/9) - 0.8 + (2 + 4/9)
  let b := 7.6 / (4/5) + (2 + 2/5) * 1.25
  a * b = 90 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l378_37856


namespace NUMINAMATH_CALUDE_triangle_gp_length_l378_37814

-- Define the triangle DEF
structure Triangle :=
  (DE DF EF : ℝ)

-- Define the centroid G and point P
structure TrianglePoints (t : Triangle) :=
  (G P : ℝ × ℝ)

-- Define the length of GP
def lengthGP (t : Triangle) (tp : TrianglePoints t) : ℝ :=
  sorry

-- Theorem statement
theorem triangle_gp_length (t : Triangle) (tp : TrianglePoints t) 
  (h1 : t.DE = 10) (h2 : t.DF = 15) (h3 : t.EF = 17) : 
  lengthGP t tp = 4 * Real.sqrt 154 / 17 := by
  sorry

end NUMINAMATH_CALUDE_triangle_gp_length_l378_37814


namespace NUMINAMATH_CALUDE_blueberry_count_l378_37843

/-- Represents the number of berries in a box of a specific color -/
structure BerryBox where
  blue : ℕ
  red : ℕ
  green : ℕ

/-- The change in berry counts when replacing boxes -/
structure BerryChange where
  total : ℤ
  difference : ℤ

theorem blueberry_count (box : BerryBox) 
  (replace_blue_with_red : BerryChange)
  (replace_green_with_blue : BerryChange) :
  (replace_blue_with_red.total = 10) →
  (replace_blue_with_red.difference = 50) →
  (replace_green_with_blue.total = -5) →
  (replace_green_with_blue.difference = -30) →
  (box.red - box.blue = replace_blue_with_red.total) →
  (box.blue - box.green = -replace_green_with_blue.total) →
  (box.green - 2 * box.blue = -replace_green_with_blue.difference) →
  box.blue = 35 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_count_l378_37843


namespace NUMINAMATH_CALUDE_marthas_children_l378_37858

/-- Given that Martha needs to buy a total number of cakes and each child should receive a specific number of cakes, calculate the number of children Martha has. -/
theorem marthas_children (total_cakes : ℕ) (cakes_per_child : ℚ) : 
  total_cakes = 54 → cakes_per_child = 18 → (total_cakes : ℚ) / cakes_per_child = 3 := by
  sorry

end NUMINAMATH_CALUDE_marthas_children_l378_37858


namespace NUMINAMATH_CALUDE_add_like_terms_l378_37825

theorem add_like_terms (a : ℝ) : 3 * a + 2 * a = 5 * a := by
  sorry

end NUMINAMATH_CALUDE_add_like_terms_l378_37825


namespace NUMINAMATH_CALUDE_distance_after_10_hours_l378_37828

/-- The distance between two people walking in the same direction for a given time -/
def distance_between (speed1 speed2 time : ℝ) : ℝ :=
  (speed2 - speed1) * time

/-- Theorem: The distance between two people walking for 10 hours at 5.5 kmph and 7.5 kmph is 20 km -/
theorem distance_after_10_hours :
  let speed1 : ℝ := 5.5
  let speed2 : ℝ := 7.5
  let time : ℝ := 10
  distance_between speed1 speed2 time = 20 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_10_hours_l378_37828


namespace NUMINAMATH_CALUDE_inequality_proof_l378_37892

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  4 * (a^3 + b^3) > (a + b)^3 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l378_37892


namespace NUMINAMATH_CALUDE_greatest_of_four_consecutive_integers_l378_37826

theorem greatest_of_four_consecutive_integers (a b c d : ℤ) : 
  (a + 1 = b) ∧ (b + 1 = c) ∧ (c + 1 = d) ∧ (a + b + c + d = 102) → d = 27 := by
  sorry

end NUMINAMATH_CALUDE_greatest_of_four_consecutive_integers_l378_37826


namespace NUMINAMATH_CALUDE_isabel_piggy_bank_l378_37840

theorem isabel_piggy_bank (initial_amount : ℝ) : 
  initial_amount / 2 / 2 = 51 → initial_amount = 204 := by
  sorry

end NUMINAMATH_CALUDE_isabel_piggy_bank_l378_37840


namespace NUMINAMATH_CALUDE_woodworker_productivity_increase_l378_37899

/-- Woodworker's productivity increase problem -/
theorem woodworker_productivity_increase
  (normal_days : ℕ)
  (normal_parts : ℕ)
  (new_days : ℕ)
  (extra_parts : ℕ)
  (h1 : normal_days = 24)
  (h2 : normal_parts = 360)
  (h3 : new_days = 22)
  (h4 : extra_parts = 80) :
  (normal_parts + extra_parts) / new_days - normal_parts / normal_days = 5 :=
by sorry

end NUMINAMATH_CALUDE_woodworker_productivity_increase_l378_37899


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l378_37861

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l378_37861


namespace NUMINAMATH_CALUDE_gate_width_scientific_notation_l378_37834

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem gate_width_scientific_notation :
  toScientificNotation 0.000000007 = ScientificNotation.mk 7 (-9) sorry := by
  sorry

end NUMINAMATH_CALUDE_gate_width_scientific_notation_l378_37834


namespace NUMINAMATH_CALUDE_min_value_of_expression_l378_37815

theorem min_value_of_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let A := ((a + b) / c)^4 + ((b + c) / d)^4 + ((c + d) / a)^4 + ((d + a) / b)^4
  A ≥ 64 ∧ (A = 64 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l378_37815


namespace NUMINAMATH_CALUDE_chameleon_painter_cannot_create_checkerboard_l378_37818

structure Board :=
  (size : Nat)
  (initial_color : Bool)

structure Painter :=
  (initial_color : Bool)
  (can_change_self : Bool)
  (can_change_square : Bool)

def is_checkerboard (board : Board) : Prop :=
  sorry

theorem chameleon_painter_cannot_create_checkerboard 
  (board : Board) 
  (painter : Painter) : 
  board.size = 8 ∧ 
  board.initial_color = false ∧ 
  painter.initial_color = true ∧
  painter.can_change_self = true ∧
  painter.can_change_square = true →
  ¬ (is_checkerboard board) :=
sorry

end NUMINAMATH_CALUDE_chameleon_painter_cannot_create_checkerboard_l378_37818


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_eighty_l378_37804

theorem twenty_five_percent_less_than_eighty (x : ℝ) : x = 40 ↔ 80 - 0.25 * 80 = x + 0.5 * x := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_eighty_l378_37804


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l378_37873

/-- A quadratic function is a function of the form f(x) = ax² + bx + c where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_unique 
  (f : ℝ → ℝ) 
  (h1 : IsQuadratic f) 
  (h2 : f 0 = 3) 
  (h3 : ∀ x, f (x + 2) - f x = 4 * x + 2) : 
  ∀ x, f x = x^2 - x + 3 := by
sorry


end NUMINAMATH_CALUDE_quadratic_function_unique_l378_37873


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l378_37888

-- Define the fuel tank capacity
def C : ℝ := 200

-- Define the volume of fuel A added
def fuel_A_volume : ℝ := 349.99999999999994

-- Define the ethanol percentage in fuel A
def ethanol_A_percent : ℝ := 0.12

-- Define the ethanol percentage in fuel B
def ethanol_B_percent : ℝ := 0.16

-- Define the total ethanol volume in the full tank
def total_ethanol_volume : ℝ := 18

-- Theorem statement
theorem fuel_tank_capacity :
  C = 200 ∧
  fuel_A_volume = 349.99999999999994 ∧
  ethanol_A_percent = 0.12 ∧
  ethanol_B_percent = 0.16 ∧
  total_ethanol_volume = 18 →
  ethanol_A_percent * fuel_A_volume + ethanol_B_percent * (C - fuel_A_volume) = total_ethanol_volume :=
by sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l378_37888


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_l378_37882

theorem regular_polygon_interior_angle (n : ℕ) (h : n - 3 = 5) :
  (180 * (n - 2) : ℝ) / n = 135 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_l378_37882


namespace NUMINAMATH_CALUDE_savings_calculation_l378_37894

def income_expenditure_ratio : Rat := 10 / 4
def income : ℕ := 19000
def tax_rate : Rat := 15 / 100
def long_term_investment_rate : Rat := 10 / 100
def short_term_investment_rate : Rat := 20 / 100

def calculate_savings (income_expenditure_ratio : Rat) (income : ℕ) (tax_rate : Rat) 
  (long_term_investment_rate : Rat) (short_term_investment_rate : Rat) : ℕ :=
  sorry

theorem savings_calculation :
  calculate_savings income_expenditure_ratio income tax_rate 
    long_term_investment_rate short_term_investment_rate = 11628 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l378_37894


namespace NUMINAMATH_CALUDE_hotel_room_charge_comparison_l378_37842

theorem hotel_room_charge_comparison (P R G : ℝ) 
  (h1 : P = R - 0.25 * R) 
  (h2 : P = G - 0.10 * G) : 
  R = 1.2 * G := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_charge_comparison_l378_37842


namespace NUMINAMATH_CALUDE_smallest_multiple_of_45_with_0_and_8_l378_37859

def is_multiple_of_45 (n : ℕ) : Prop := n % 45 = 0

def consists_of_0_and_8 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 8

def is_smallest_satisfying (n : ℕ) : Prop :=
  is_multiple_of_45 n ∧ 
  consists_of_0_and_8 n ∧ 
  ∀ m, m < n → ¬(is_multiple_of_45 m ∧ consists_of_0_and_8 m)

theorem smallest_multiple_of_45_with_0_and_8 :
  is_smallest_satisfying 8888888880 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_45_with_0_and_8_l378_37859


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l378_37853

/-- 
If the quadratic equation 2x^2 - x + c = 0 has two equal real roots, 
then c = 1/8.
-/
theorem equal_roots_quadratic (c : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - x + c = 0 ∧ 
   ∀ y : ℝ, 2 * y^2 - y + c = 0 → y = x) → 
  c = 1/8 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l378_37853


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l378_37838

def total_players : ℕ := 15
def guaranteed_players : ℕ := 3
def lineup_size : ℕ := 5

theorem starting_lineup_combinations :
  Nat.choose (total_players - guaranteed_players) (lineup_size - guaranteed_players) = 66 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l378_37838


namespace NUMINAMATH_CALUDE_factorization_proof_l378_37880

theorem factorization_proof (x : ℝ) : 75 * x^11 + 225 * x^22 = 75 * x^11 * (1 + 3 * x^11) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l378_37880


namespace NUMINAMATH_CALUDE_functional_equation_solution_l378_37896

/-- The functional equation for f and g -/
def FunctionalEquation (f g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + g y) = x * f y - y * f x + g x

/-- The solution forms for f and g -/
def SolutionForms (f g : ℝ → ℝ) : Prop :=
  ∃ t : ℝ, t ≠ -1 ∧
    (∀ x : ℝ, f x = (t * (x - t)) / (t + 1)) ∧
    (∀ x : ℝ, g x = t * (x - t))

theorem functional_equation_solution :
    ∀ f g : ℝ → ℝ, FunctionalEquation f g → SolutionForms f g :=
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l378_37896


namespace NUMINAMATH_CALUDE_complex_modulus_l378_37846

theorem complex_modulus (z : ℂ) : (1 - Complex.I) * z = 1 + 2 * Complex.I → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l378_37846


namespace NUMINAMATH_CALUDE_three_student_committees_from_eight_l378_37802

theorem three_student_committees_from_eight (n : ℕ) (k : ℕ) : n = 8 ∧ k = 3 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_student_committees_from_eight_l378_37802


namespace NUMINAMATH_CALUDE_markup_calculation_l378_37850

/-- Given a purchase price, overhead percentage, and desired net profit, 
    calculate the required markup. -/
def calculate_markup (purchase_price : ℝ) (overhead_percentage : ℝ) (net_profit : ℝ) : ℝ :=
  let overhead_cost := overhead_percentage * purchase_price
  let total_cost := purchase_price + overhead_cost
  let selling_price := total_cost + net_profit
  selling_price - purchase_price

/-- Theorem stating that the markup for the given conditions is $14.40 -/
theorem markup_calculation : 
  calculate_markup 48 0.05 12 = 14.40 := by
  sorry

end NUMINAMATH_CALUDE_markup_calculation_l378_37850


namespace NUMINAMATH_CALUDE_cash_percentage_proof_l378_37824

def total_amount : ℝ := 7428.57
def raw_materials : ℝ := 5000
def machinery : ℝ := 200

theorem cash_percentage_proof :
  let spent := raw_materials + machinery
  let cash := total_amount - spent
  let percentage := (cash / total_amount) * 100
  ∀ ε > 0, |percentage - 29.99| < ε :=
by sorry

end NUMINAMATH_CALUDE_cash_percentage_proof_l378_37824


namespace NUMINAMATH_CALUDE_candy_count_solution_l378_37866

def is_valid_candy_count (x : ℕ) : Prop :=
  ∃ (brother_takes : ℕ),
    x % 4 = 0 ∧
    x % 2 = 0 ∧
    2 ≤ brother_takes ∧
    brother_takes ≤ 6 ∧
    (x / 4 * 3 / 3 * 2 - 40 - brother_takes = 10)

theorem candy_count_solution :
  ∀ x : ℕ, is_valid_candy_count x ↔ (x = 108 ∨ x = 112) :=
sorry

end NUMINAMATH_CALUDE_candy_count_solution_l378_37866


namespace NUMINAMATH_CALUDE_x_minus_y_squared_times_x_plus_y_l378_37830

theorem x_minus_y_squared_times_x_plus_y (x y : ℝ) (hx : x = 8) (hy : y = 3) :
  (x - y)^2 * (x + y) = 275 := by sorry

end NUMINAMATH_CALUDE_x_minus_y_squared_times_x_plus_y_l378_37830


namespace NUMINAMATH_CALUDE_martins_berry_consumption_l378_37884

/-- Given the cost of berries and Martin's spending habits, calculate his daily berry consumption --/
theorem martins_berry_consumption
  (package_cost : ℚ)
  (total_spent : ℚ)
  (num_days : ℕ)
  (h1 : package_cost = 2)
  (h2 : total_spent = 30)
  (h3 : num_days = 30)
  : (total_spent / package_cost) / num_days = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_martins_berry_consumption_l378_37884


namespace NUMINAMATH_CALUDE_max_x_value_l378_37821

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 10) (prod_eq : x*y + x*z + y*z = 20) :
  x ≤ 10/3 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ + y₀ + z₀ = 10 ∧ x₀*y₀ + x₀*z₀ + y₀*z₀ = 20 ∧ x₀ = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l378_37821


namespace NUMINAMATH_CALUDE_factorization_equality_l378_37837

theorem factorization_equality (a b : ℝ) :
  276 * a^2 * b^2 + 69 * a * b - 138 * a * b^3 = 69 * a * b * (4 * a * b + 1 - 2 * b^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l378_37837
