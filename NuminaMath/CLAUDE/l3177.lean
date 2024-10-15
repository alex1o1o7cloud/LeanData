import Mathlib

namespace NUMINAMATH_CALUDE_ball_in_hole_within_six_bounces_l3177_317776

/-- Represents a point on the table -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hole on the table -/
structure Hole where
  location : Point

/-- Represents a rectangular table with holes -/
structure Table where
  length : ℝ
  width : ℝ
  holes : List Hole

/-- Represents a ball's trajectory -/
structure Trajectory where
  start : Point
  bounces : List Point

/-- Function to check if a trajectory ends in a hole within n bounces -/
def endsInHole (traj : Trajectory) (table : Table) (n : ℕ) : Prop :=
  ∃ (h : Hole), h ∈ table.holes ∧ traj.bounces.length ≤ n ∧ traj.bounces.getLast? = some h.location

/-- The main theorem -/
theorem ball_in_hole_within_six_bounces 
  (table : Table) 
  (a b c : Point) : 
  table.length = 8 ∧ 
  table.width = 5 ∧ 
  table.holes.length = 4 →
  ∃ (start : Point) (traj : Trajectory), 
    (start = a ∨ start = b ∨ start = c) ∧
    traj.start = start ∧
    endsInHole traj table 6 :=
sorry

end NUMINAMATH_CALUDE_ball_in_hole_within_six_bounces_l3177_317776


namespace NUMINAMATH_CALUDE_certain_amount_of_seconds_l3177_317731

/-- Given that 12 is to a certain amount of seconds as 16 is to 8 minutes,
    prove that the certain amount of seconds is 360. -/
theorem certain_amount_of_seconds : ∃ X : ℝ, 
  (12 / X = 16 / (8 * 60)) ∧ (X = 360) := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_of_seconds_l3177_317731


namespace NUMINAMATH_CALUDE_four_at_six_equals_twenty_l3177_317729

-- Define the @ operation
def at_operation (a b : ℤ) : ℤ := 4*a - 2*b + a^2

-- Theorem statement
theorem four_at_six_equals_twenty : at_operation 4 6 = 20 := by
  sorry

end NUMINAMATH_CALUDE_four_at_six_equals_twenty_l3177_317729


namespace NUMINAMATH_CALUDE_min_tuple_c_value_l3177_317781

def is_valid_tuple (a b c d e f : ℕ) : Prop :=
  a + 2*b + 6*c + 30*d + 210*e + 2310*f = 2^15

def tuple_sum (a b c d e f : ℕ) : ℕ :=
  a + b + c + d + e + f

theorem min_tuple_c_value :
  ∃ (a b c d e f : ℕ),
    is_valid_tuple a b c d e f ∧
    (∀ (a' b' c' d' e' f' : ℕ),
      is_valid_tuple a' b' c' d' e' f' →
      tuple_sum a b c d e f ≤ tuple_sum a' b' c' d' e' f') ∧
    c = 1 := by sorry

end NUMINAMATH_CALUDE_min_tuple_c_value_l3177_317781


namespace NUMINAMATH_CALUDE_molecular_weight_Al_OH_3_value_l3177_317740

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The molecular weight of Al(OH)3 in g/mol -/
def molecular_weight_Al_OH_3 : ℝ := 
  atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H

/-- Theorem stating that the molecular weight of Al(OH)3 is 78.01 g/mol -/
theorem molecular_weight_Al_OH_3_value : 
  molecular_weight_Al_OH_3 = 78.01 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_Al_OH_3_value_l3177_317740


namespace NUMINAMATH_CALUDE_product_mod_seven_l3177_317712

theorem product_mod_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l3177_317712


namespace NUMINAMATH_CALUDE_area_above_x_axis_half_total_l3177_317727

-- Define the parallelogram PQRS
def P : ℝ × ℝ := (4, 4)
def Q : ℝ × ℝ := (-2, -2)
def R : ℝ × ℝ := (-8, -2)
def S : ℝ × ℝ := (-2, 4)

-- Define a function to calculate the area of a parallelogram
def parallelogramArea (a b c d : ℝ × ℝ) : ℝ := sorry

-- Define a function to calculate the area of the part of the parallelogram above the x-axis
def areaAboveXAxis (a b c d : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_above_x_axis_half_total : 
  areaAboveXAxis P Q R S = (1/2) * parallelogramArea P Q R S := by sorry

end NUMINAMATH_CALUDE_area_above_x_axis_half_total_l3177_317727


namespace NUMINAMATH_CALUDE_ratio_difference_theorem_l3177_317769

theorem ratio_difference_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / b = 2 / 3 → (a + 4) / (b + 4) = 5 / 7 → b - a = 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_theorem_l3177_317769


namespace NUMINAMATH_CALUDE_boulevard_painting_cost_l3177_317775

/-- Represents a side of the boulevard with house numbers -/
structure BoulevardSide where
  start : ℕ
  diff : ℕ
  count : ℕ

/-- Calculates the sum of digits for all numbers in an arithmetic sequence -/
def sumOfDigits (side : BoulevardSide) : ℕ :=
  sorry

/-- The total cost of painting house numbers on both sides of the boulevard -/
def totalCost (eastSide westSide : BoulevardSide) : ℕ :=
  sumOfDigits eastSide + sumOfDigits westSide

theorem boulevard_painting_cost :
  let eastSide : BoulevardSide := { start := 5, diff := 7, count := 25 }
  let westSide : BoulevardSide := { start := 2, diff := 5, count := 25 }
  totalCost eastSide westSide = 113 :=
sorry

end NUMINAMATH_CALUDE_boulevard_painting_cost_l3177_317775


namespace NUMINAMATH_CALUDE_marble_arrangement_l3177_317709

/-- Represents the number of green marbles -/
def green_marbles : Nat := 4

/-- Represents the number of red marbles -/
def red_marbles : Nat := 3

/-- Represents the maximum number of blue marbles that can be used to create a balanced arrangement -/
def m : Nat := 5

/-- Represents the total number of slots where blue marbles can be placed -/
def total_slots : Nat := green_marbles + red_marbles + 1

/-- Calculates the number of ways to arrange the marbles -/
def N : Nat := Nat.choose (m + total_slots - 1) m

/-- Theorem stating the properties of the marble arrangement -/
theorem marble_arrangement :
  (N % 1000 = 287) ∧
  (∀ k : Nat, k > m → Nat.choose (k + total_slots - 1) k % 1000 ≠ 287) := by
  sorry

end NUMINAMATH_CALUDE_marble_arrangement_l3177_317709


namespace NUMINAMATH_CALUDE_quadratic_sum_l3177_317778

-- Define the quadratic function
def f (x : ℝ) : ℝ := -8 * x^2 + 16 * x + 320

-- Define the completed square form
def g (a b c : ℝ) (x : ℝ) : ℝ := a * (x + b)^2 + c

theorem quadratic_sum : ∃ a b c : ℝ, 
  (∀ x, f x = g a b c x) ∧ 
  (a + b + c = 319) := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3177_317778


namespace NUMINAMATH_CALUDE_a_can_be_any_real_l3177_317771

theorem a_can_be_any_real : ∀ (a b c d : ℤ), 
  b > 0 → d < 0 → (a : ℚ) / b > (c : ℚ) / d → 
  (∃ (x : ℝ), x > 0 ∧ (∃ (a : ℤ), (a : ℚ) / b > (c : ℚ) / d ∧ (a : ℝ) = x)) ∧
  (∃ (y : ℝ), y < 0 ∧ (∃ (a : ℤ), (a : ℚ) / b > (c : ℚ) / d ∧ (a : ℝ) = y)) ∧
  (∃ (a : ℤ), (a : ℚ) / b > (c : ℚ) / d ∧ a = 0) :=
by sorry

end NUMINAMATH_CALUDE_a_can_be_any_real_l3177_317771


namespace NUMINAMATH_CALUDE_finite_solutions_factorial_difference_l3177_317745

theorem finite_solutions_factorial_difference (u : ℕ+) :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), ∀ (n a b : ℕ), 
    n! = u^a - u^b → (n, a, b) ∈ S :=
sorry

end NUMINAMATH_CALUDE_finite_solutions_factorial_difference_l3177_317745


namespace NUMINAMATH_CALUDE_cosine_BHD_value_l3177_317717

structure RectangularPrism where
  DHG : Real
  FHB : Real

def cosine_BHD (prism : RectangularPrism) : Real :=
  sorry

theorem cosine_BHD_value (prism : RectangularPrism) 
  (h1 : prism.DHG = Real.pi / 4)
  (h2 : prism.FHB = Real.pi / 3) :
  cosine_BHD prism = Real.sqrt 6 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_BHD_value_l3177_317717


namespace NUMINAMATH_CALUDE_inequality_transformation_l3177_317799

theorem inequality_transformation (a : ℝ) : 
  (∀ x : ℝ, a * x > 2 ↔ x < 2 / a) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_transformation_l3177_317799


namespace NUMINAMATH_CALUDE_polynomial_property_l3177_317774

-- Define the polynomial Q(x)
def Q (p q d : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x + d

-- State the theorem
theorem polynomial_property (p q d : ℝ) :
  -- The mean of zeros equals the product of zeros taken two at a time
  (-p/3 = q) →
  -- The mean of zeros equals the sum of coefficients
  (-p/3 = 1 + p + q + d) →
  -- The y-intercept is 5
  (Q p q d 0 = 5) →
  -- Then q = 2
  q = 2 := by sorry

end NUMINAMATH_CALUDE_polynomial_property_l3177_317774


namespace NUMINAMATH_CALUDE_twenty_four_divides_Q_largest_divisor_of_Q_l3177_317706

/-- The product of three consecutive positive even integers -/
def Q (n : ℕ) : ℕ := (2*n) * (2*n + 2) * (2*n + 4)

/-- 24 divides Q for all positive n -/
theorem twenty_four_divides_Q (n : ℕ) (h : n > 0) : 24 ∣ Q n := by sorry

/-- 24 is the largest integer that divides Q for all positive n -/
theorem largest_divisor_of_Q :
  ∀ d : ℕ, (∀ n : ℕ, n > 0 → d ∣ Q n) → d ≤ 24 := by sorry

end NUMINAMATH_CALUDE_twenty_four_divides_Q_largest_divisor_of_Q_l3177_317706


namespace NUMINAMATH_CALUDE_train_overtake_l3177_317785

/-- The speed of Train A in miles per hour -/
def speed_a : ℝ := 30

/-- The speed of Train B in miles per hour -/
def speed_b : ℝ := 38

/-- The time difference between Train A and Train B's departure in hours -/
def time_diff : ℝ := 2

/-- The distance at which Train B overtakes Train A -/
def overtake_distance : ℝ := 285

theorem train_overtake :
  ∃ t : ℝ, t > 0 ∧ speed_b * t = speed_a * (t + time_diff) ∧ 
  overtake_distance = speed_b * t :=
sorry

end NUMINAMATH_CALUDE_train_overtake_l3177_317785


namespace NUMINAMATH_CALUDE_negative_result_l3177_317754

theorem negative_result : 1 - 9 < 0 := by
  sorry

#check negative_result

end NUMINAMATH_CALUDE_negative_result_l3177_317754


namespace NUMINAMATH_CALUDE_angela_action_figures_l3177_317742

theorem angela_action_figures (initial : ℕ) (sold_fraction : ℚ) (given_fraction : ℚ) : 
  initial = 24 → 
  sold_fraction = 1/4 → 
  given_fraction = 1/3 → 
  initial - (initial * sold_fraction).floor - ((initial - (initial * sold_fraction).floor) * given_fraction).floor = 12 := by
  sorry

end NUMINAMATH_CALUDE_angela_action_figures_l3177_317742


namespace NUMINAMATH_CALUDE_problem_solution_l3177_317773

theorem problem_solution (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) :
  a^2 * b - 2 * a * b + a * b^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3177_317773


namespace NUMINAMATH_CALUDE_divisible_by_thirty_l3177_317744

theorem divisible_by_thirty (n : ℕ) (h : n > 0) : ∃ k : ℤ, n^19 - n^7 = 30 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_thirty_l3177_317744


namespace NUMINAMATH_CALUDE_non_monotonic_range_l3177_317724

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem non_monotonic_range (a : ℝ) : 
  (¬ is_monotonic (f a)) ↔ 
  (0 < a ∧ a < 1/7) ∨ (1/3 ≤ a ∧ a < 1) ∨ (a > 1) :=
sorry

end NUMINAMATH_CALUDE_non_monotonic_range_l3177_317724


namespace NUMINAMATH_CALUDE_probability_of_purple_marble_l3177_317780

theorem probability_of_purple_marble (blue_prob green_prob purple_prob : ℝ) : 
  blue_prob = 0.35 →
  green_prob = 0.45 →
  blue_prob + green_prob + purple_prob = 1 →
  purple_prob = 0.2 := by
sorry

end NUMINAMATH_CALUDE_probability_of_purple_marble_l3177_317780


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3177_317789

/-- Given the quadratic equation x^2 + 4x + 4 = 0, which can be transformed
    into the form (x + h)^2 = k, prove that h + k = 2. -/
theorem quadratic_transformation (h k : ℝ) : 
  (∀ x, x^2 + 4*x + 4 = 0 ↔ (x + h)^2 = k) → h + k = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3177_317789


namespace NUMINAMATH_CALUDE_remaining_volume_of_cube_with_hole_l3177_317753

/-- The remaining volume of a cube with a square hole cut through its center -/
theorem remaining_volume_of_cube_with_hole (cube_side : ℝ) (hole_side : ℝ) : 
  cube_side = 8 → hole_side = 4 → 
  cube_side ^ 3 - (hole_side ^ 2 * cube_side) = 384 := by
  sorry

end NUMINAMATH_CALUDE_remaining_volume_of_cube_with_hole_l3177_317753


namespace NUMINAMATH_CALUDE_abc_product_l3177_317770

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 30 * Real.rpow 4 (1/3))
  (hac : a * c = 40 * Real.rpow 4 (1/3))
  (hbc : b * c = 24 * Real.rpow 4 (1/3)) :
  a * b * c = 120 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l3177_317770


namespace NUMINAMATH_CALUDE_b_work_time_l3177_317779

/-- The time it takes for worker a to complete the work alone -/
def a_time : ℝ := 14

/-- The time it takes for workers a and b to complete the work together -/
def ab_time : ℝ := 5.833333333333333

/-- The time it takes for worker b to complete the work alone -/
def b_time : ℝ := 10

/-- The total amount of work to be completed -/
def total_work : ℝ := 1

theorem b_work_time : 
  (1 / a_time + 1 / b_time = 1 / ab_time) ∧
  (1 / b_time = 1 / total_work) := by
  sorry

end NUMINAMATH_CALUDE_b_work_time_l3177_317779


namespace NUMINAMATH_CALUDE_vegetable_planting_methods_l3177_317748

theorem vegetable_planting_methods :
  let total_vegetables : ℕ := 4
  let vegetables_to_choose : ℕ := 3
  let remaining_choices : ℕ := total_vegetables - 1  -- Cucumber is always chosen
  let remaining_to_choose : ℕ := vegetables_to_choose - 1
  let soil_types : ℕ := 3
  
  (remaining_choices.choose remaining_to_choose) * (vegetables_to_choose.factorial) = 18 :=
by sorry

end NUMINAMATH_CALUDE_vegetable_planting_methods_l3177_317748


namespace NUMINAMATH_CALUDE_quadratic_root_l3177_317787

theorem quadratic_root (k : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - 3 * x - k = 0 ∧ x = 1) → 
  (∃ y : ℝ, 2 * y^2 - 3 * y - k = 0 ∧ y = 1/2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_l3177_317787


namespace NUMINAMATH_CALUDE_total_hot_dogs_today_l3177_317722

def hot_dogs_lunch : ℕ := 9
def hot_dogs_dinner : ℕ := 2

theorem total_hot_dogs_today : hot_dogs_lunch + hot_dogs_dinner = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_hot_dogs_today_l3177_317722


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3177_317760

/-- A rectangle and a circle that intersect in a specific way -/
structure RectangleCircleIntersection where
  /-- The diameter of the circle -/
  d : ℝ
  /-- Assumption that the diameter is positive -/
  d_pos : d > 0
  /-- The length of the longer side of the rectangle -/
  long_side : ℝ
  /-- The length of the shorter side of the rectangle -/
  short_side : ℝ
  /-- The longer side of the rectangle is twice the diameter of the circle -/
  long_side_eq : long_side = 2 * d
  /-- The shorter side of the rectangle is equal to the diameter of the circle -/
  short_side_eq : short_side = d

/-- The ratio of the area of the rectangle to the area of the circle is 8/π -/
theorem rectangle_circle_area_ratio (rc : RectangleCircleIntersection) :
  (rc.long_side * rc.short_side) / (π * (rc.d / 2)^2) = 8 / π := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3177_317760


namespace NUMINAMATH_CALUDE_sum_of_integers_l3177_317772

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 10) (h3 : x * y = 80) : x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3177_317772


namespace NUMINAMATH_CALUDE_complex_absolute_value_problem_l3177_317752

theorem complex_absolute_value_problem : 
  let z₁ : ℂ := 3 - 5*I
  let z₂ : ℂ := 3 + 5*I
  Complex.abs z₁ * Complex.abs z₂ + 2 * Complex.abs z₁ = 34 + 2 * Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_complex_absolute_value_problem_l3177_317752


namespace NUMINAMATH_CALUDE_two_shirts_per_package_l3177_317762

/-- Given a number of packages and a total number of t-shirts,
    calculate the number of t-shirts per package. -/
def tShirtsPerPackage (packages : ℕ) (totalShirts : ℕ) : ℚ :=
  totalShirts / packages

/-- Theorem stating that given 28 packages and 56 total t-shirts,
    the number of t-shirts per package is 2. -/
theorem two_shirts_per_package :
  tShirtsPerPackage 28 56 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_shirts_per_package_l3177_317762


namespace NUMINAMATH_CALUDE_three_digit_number_divisibility_l3177_317732

theorem three_digit_number_divisibility (a b c : Nat) : 
  a ≠ 0 ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ a + b + c = 7 →
  (100 * a + 10 * b + c) % 7 = 0 ↔ b = c :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_divisibility_l3177_317732


namespace NUMINAMATH_CALUDE_faye_coloring_books_l3177_317741

theorem faye_coloring_books (given_away_first : ℝ) (given_away_second : ℝ) (remaining : ℕ) 
  (h1 : given_away_first = 34.0)
  (h2 : given_away_second = 3.0)
  (h3 : remaining = 11) :
  given_away_first + given_away_second + remaining = 48.0 := by
  sorry

end NUMINAMATH_CALUDE_faye_coloring_books_l3177_317741


namespace NUMINAMATH_CALUDE_gloria_cabin_theorem_l3177_317725

/-- Represents the problem of calculating Gloria's remaining money after buying a cabin --/
def gloria_cabin_problem (cabin_price cash_on_hand cypress_count pine_count maple_count cypress_price pine_price maple_price : ℕ) : Prop :=
  let total_from_trees := cypress_count * cypress_price + pine_count * pine_price + maple_count * maple_price
  let total_amount := total_from_trees + cash_on_hand
  let money_left := total_amount - cabin_price
  money_left = 350

/-- Theorem stating that Gloria will have $350 left after buying the cabin --/
theorem gloria_cabin_theorem : gloria_cabin_problem 129000 150 20 600 24 100 200 300 := by
  sorry

end NUMINAMATH_CALUDE_gloria_cabin_theorem_l3177_317725


namespace NUMINAMATH_CALUDE_initial_orchids_l3177_317796

/-- Proves that the initial number of orchids in the vase was 2, given that there are now 21 orchids
    in the vase after 19 orchids were added. -/
theorem initial_orchids (final_orchids : ℕ) (added_orchids : ℕ) 
  (h1 : final_orchids = 21) 
  (h2 : added_orchids = 19) : 
  final_orchids - added_orchids = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_orchids_l3177_317796


namespace NUMINAMATH_CALUDE_some_number_value_l3177_317711

theorem some_number_value (x : ℝ) (h : (55 + 113 / x) * x = 4403) : x = 78 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3177_317711


namespace NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_theorem_l3177_317795

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange 3 different mathematics books and 3 different Chinese books
    on a shelf, such that books of the same type are not adjacent. -/
theorem book_arrangement_count : ℕ := 
  2 * permutations 3 * permutations 3

/-- Prove that the number of ways to arrange 3 different mathematics books and 3 different Chinese books
    on a shelf, such that books of the same type are not adjacent, is equal to 72. -/
theorem book_arrangement_theorem : book_arrangement_count = 72 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_theorem_l3177_317795


namespace NUMINAMATH_CALUDE_prob_red_given_red_half_l3177_317786

/-- A bag with red and yellow balls -/
structure Bag where
  total : ℕ
  red : ℕ
  yellow : ℕ
  h_total : total = red + yellow

/-- The probability of drawing a red ball in the second draw given a red ball in the first draw -/
def prob_red_given_red (b : Bag) : ℚ :=
  (b.red - 1) / (b.total - 1)

/-- The theorem stating the probability is 1/2 for the given bag -/
theorem prob_red_given_red_half (b : Bag) 
  (h_total : b.total = 5)
  (h_red : b.red = 3)
  (h_yellow : b.yellow = 2) : 
  prob_red_given_red b = 1/2 := by
sorry

end NUMINAMATH_CALUDE_prob_red_given_red_half_l3177_317786


namespace NUMINAMATH_CALUDE_quadratic_point_value_l3177_317710

/-- If the point (1,a) lies on the graph of y = 2x^2, then a = 2 -/
theorem quadratic_point_value (a : ℝ) : (2 : ℝ) * (1 : ℝ)^2 = a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_value_l3177_317710


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l3177_317738

-- Define the cylinders
structure Cylinder where
  radius : ℝ
  height : ℝ

-- Define the theorem
theorem cylinder_height_relationship (c1 c2 : Cylinder) : 
  -- Conditions
  (c1.radius * c1.radius * c1.height = c2.radius * c2.radius * c2.height) →  -- Equal volumes
  (c2.radius = 1.2 * c1.radius) →                                            -- Second radius is 20% more
  -- Conclusion
  (c1.height = 1.44 * c2.height) :=                                          -- First height is 44% more
by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l3177_317738


namespace NUMINAMATH_CALUDE_son_shoveling_time_l3177_317736

/-- Given a driveway shoveling scenario with three people, this theorem proves
    the time it takes for the son to shovel the entire driveway alone. -/
theorem son_shoveling_time (wayne_rate son_rate neighbor_rate : ℝ) 
  (h1 : wayne_rate = 6 * son_rate) 
  (h2 : neighbor_rate = 2 * wayne_rate) 
  (h3 : son_rate + wayne_rate + neighbor_rate = 1 / 2) : 
  1 / son_rate = 38 := by
  sorry

end NUMINAMATH_CALUDE_son_shoveling_time_l3177_317736


namespace NUMINAMATH_CALUDE_tea_profit_percentage_l3177_317702

/-- Given a tea mixture and sale price, calculate the profit percentage -/
theorem tea_profit_percentage
  (tea1_weight : ℝ)
  (tea1_cost : ℝ)
  (tea2_weight : ℝ)
  (tea2_cost : ℝ)
  (sale_price : ℝ)
  (h1 : tea1_weight = 80)
  (h2 : tea1_cost = 15)
  (h3 : tea2_weight = 20)
  (h4 : tea2_cost = 20)
  (h5 : sale_price = 20.8) :
  let total_cost := tea1_weight * tea1_cost + tea2_weight * tea2_cost
  let total_weight := tea1_weight + tea2_weight
  let total_sale := total_weight * sale_price
  let profit := total_sale - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_tea_profit_percentage_l3177_317702


namespace NUMINAMATH_CALUDE_no_intersection_l3177_317788

def f (x : ℝ) := |3 * x + 6|
def g (x : ℝ) := -|4 * x - 3|

theorem no_intersection :
  ∀ x : ℝ, f x ≠ g x :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l3177_317788


namespace NUMINAMATH_CALUDE_find_divisor_l3177_317739

theorem find_divisor (n s : ℕ) (hn : n = 5264) (hs : s = 11) :
  let d := n - s
  (d ∣ d) ∧ (∀ m : ℕ, m < s → ¬(d ∣ (n - m))) → d = 5253 :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l3177_317739


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3177_317703

theorem quadratic_inequality_equivalence (x : ℝ) :
  3 * x^2 + x - 2 < 0 ↔ -1 < x ∧ x < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3177_317703


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3177_317797

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3177_317797


namespace NUMINAMATH_CALUDE_square_perimeter_l3177_317755

theorem square_perimeter (s : ℝ) (h1 : s > 0) : 
  let rectangle_perimeter := 2 * (s + s / 5)
  rectangle_perimeter = 48 → 4 * s = 80 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l3177_317755


namespace NUMINAMATH_CALUDE_min_value_of_sqrt_sums_l3177_317701

theorem min_value_of_sqrt_sums (a b c : ℝ) : 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c → 
  a * b + b * c + c * a = a + b + c → 
  0 < a + b + c → 
  2 ≤ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sqrt_sums_l3177_317701


namespace NUMINAMATH_CALUDE_triangle_on_axes_zero_volume_l3177_317750

/-- Given a triangle ABC with sides of length 8, 6, and 10, where each vertex is on a positive axis,
    prove that the volume of tetrahedron OABC (where O is the origin) is 0. -/
theorem triangle_on_axes_zero_volume (a b c : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 →  -- vertices on positive axes
  a^2 + b^2 = 64 →  -- AB = 8
  b^2 + c^2 = 36 →  -- BC = 6
  c^2 + a^2 = 100 →  -- CA = 10
  (1/6 : ℝ) * a * b * c = 0 :=  -- volume of tetrahedron OABC
by sorry


end NUMINAMATH_CALUDE_triangle_on_axes_zero_volume_l3177_317750


namespace NUMINAMATH_CALUDE_complex_difference_modulus_l3177_317743

theorem complex_difference_modulus (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 1)
  (h2 : Complex.abs z₂ = 1)
  (h3 : Complex.abs (z₁ + z₂) = Real.sqrt 3) :
  Complex.abs (z₁ - z₂) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_difference_modulus_l3177_317743


namespace NUMINAMATH_CALUDE_pumpkin_patch_pie_filling_l3177_317766

/-- Represents the number of cans of pie filling produced from small and large pumpkins -/
def cans_of_pie_filling (small_pumpkins : ℕ) (large_pumpkins : ℕ) : ℕ :=
  (small_pumpkins / 2) + large_pumpkins

theorem pumpkin_patch_pie_filling :
  let small_pumpkins : ℕ := 50
  let large_pumpkins : ℕ := 33
  let total_sales : ℕ := 120
  let small_price : ℕ := 3
  let large_price : ℕ := 5
  cans_of_pie_filling small_pumpkins large_pumpkins = 58 := by
  sorry

#eval cans_of_pie_filling 50 33

end NUMINAMATH_CALUDE_pumpkin_patch_pie_filling_l3177_317766


namespace NUMINAMATH_CALUDE_down_jacket_price_reduction_l3177_317723

/-- Represents the price reduction problem for down jackets --/
theorem down_jacket_price_reduction
  (initial_sales : ℕ)
  (initial_profit_per_piece : ℕ)
  (sales_increase_per_yuan : ℕ)
  (target_daily_profit : ℕ)
  (h1 : initial_sales = 20)
  (h2 : initial_profit_per_piece = 40)
  (h3 : sales_increase_per_yuan = 2)
  (h4 : target_daily_profit = 1200) :
  ∃ (price_reduction : ℕ),
    (initial_profit_per_piece - price_reduction) *
    (initial_sales + sales_increase_per_yuan * price_reduction) = target_daily_profit ∧
    price_reduction = 20 :=
by sorry

end NUMINAMATH_CALUDE_down_jacket_price_reduction_l3177_317723


namespace NUMINAMATH_CALUDE_carols_invitations_l3177_317737

/-- Given that Carol bought packages of invitations, prove that the number of friends she can invite is equal to the product of invitations per package and the number of packages. -/
theorem carols_invitations (invitations_per_package : ℕ) (num_packages : ℕ) :
  invitations_per_package = 9 →
  num_packages = 5 →
  invitations_per_package * num_packages = 45 := by
  sorry

#check carols_invitations

end NUMINAMATH_CALUDE_carols_invitations_l3177_317737


namespace NUMINAMATH_CALUDE_factorization_proof_l3177_317767

theorem factorization_proof (x y : ℝ) : -2*x*y^2 + 4*x*y - 2*x = -2*x*(y - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3177_317767


namespace NUMINAMATH_CALUDE_sum_70_terms_is_negative_350_l3177_317733

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference

/-- Sum of first n terms of an arithmetic progression -/
def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a + (n - 1 : ℚ) * ap.d)

/-- Theorem: For an arithmetic progression with specific properties, 
    the sum of its first 70 terms is -350 -/
theorem sum_70_terms_is_negative_350 
  (ap : ArithmeticProgression)
  (h1 : sum_n_terms ap 20 = 200)
  (h2 : sum_n_terms ap 50 = 50) :
  sum_n_terms ap 70 = -350 := by
  sorry

end NUMINAMATH_CALUDE_sum_70_terms_is_negative_350_l3177_317733


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3177_317719

/-- Two vectors in ℝ² are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (x, -6)
  parallel a b → x = -4 :=
by
  sorry

#check parallel_vectors_x_value

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3177_317719


namespace NUMINAMATH_CALUDE_m_range_when_exists_positive_root_l3177_317793

/-- The quadratic function f(x) = x^2 + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

/-- The proposition that there exists a positive x₀ such that f(x₀) < 0 -/
def exists_positive_root (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀ > 0 ∧ f m x₀ < 0

/-- Theorem stating that if there exists a positive x₀ such that f(x₀) < 0,
    then m is in the open interval (-∞, -2) -/
theorem m_range_when_exists_positive_root :
  ∀ m : ℝ, exists_positive_root m → m < -2 :=
by sorry

end NUMINAMATH_CALUDE_m_range_when_exists_positive_root_l3177_317793


namespace NUMINAMATH_CALUDE_increase_by_percentage_l3177_317791

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 70 →
  percentage = 50 →
  final = initial * (1 + percentage / 100) →
  final = 105 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l3177_317791


namespace NUMINAMATH_CALUDE_equation_satisfied_l3177_317713

theorem equation_satisfied (x y : ℝ) (hx : x = 5) (hy : y = 3) : 2 * x - 3 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l3177_317713


namespace NUMINAMATH_CALUDE_cube_triangles_area_sum_l3177_317728

/-- Represents a 3D point in a cube --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle in 3D space --/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- The vertices of a 2x2x2 cube --/
def cubeVertices : List Point3D := sorry

/-- Calculates the area of a triangle in 3D space --/
def triangleArea (t : Triangle3D) : ℝ := sorry

/-- Generates all possible triangles from the cube vertices --/
def allTriangles : List Triangle3D := sorry

/-- Expresses a real number in the form m + √n + √p --/
structure SqrtForm where
  m : ℤ
  n : ℤ
  p : ℤ

/-- Converts a real number to SqrtForm --/
def toSqrtForm (r : ℝ) : SqrtForm := sorry

/-- The main theorem --/
theorem cube_triangles_area_sum :
  let totalArea := (allTriangles.map triangleArea).sum
  let sqrtForm := toSqrtForm totalArea
  sqrtForm.m + sqrtForm.n + sqrtForm.p = 121 := by sorry

end NUMINAMATH_CALUDE_cube_triangles_area_sum_l3177_317728


namespace NUMINAMATH_CALUDE_bottles_not_in_crates_l3177_317715

/-- Represents the number of bottles that can be held by each crate size -/
structure CrateCapacity where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the number of crates of each size -/
structure CrateCount where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculate the total capacity of all crates -/
def totalCrateCapacity (capacity : CrateCapacity) (count : CrateCount) : Nat :=
  capacity.small * count.small + capacity.medium * count.medium + capacity.large * count.large

/-- Calculate the number of bottles that will not be placed in a crate -/
def bottlesNotInCrates (totalBottles : Nat) (capacity : CrateCapacity) (count : CrateCount) : Nat :=
  totalBottles - totalCrateCapacity capacity count

/-- Theorem stating that 50 bottles will not be placed in a crate -/
theorem bottles_not_in_crates : 
  let totalBottles : Nat := 250
  let capacity : CrateCapacity := { small := 8, medium := 12, large := 20 }
  let count : CrateCount := { small := 5, medium := 5, large := 5 }
  bottlesNotInCrates totalBottles capacity count = 50 := by
  sorry

end NUMINAMATH_CALUDE_bottles_not_in_crates_l3177_317715


namespace NUMINAMATH_CALUDE_equation_solution_l3177_317784

theorem equation_solution : ∃ x : ℝ, 
  x * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800.0000000000005 ∧ 
  abs (x - 1.4) < 0.00000000000001 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3177_317784


namespace NUMINAMATH_CALUDE_nines_in_hundred_l3177_317756

def count_nines (n : ℕ) : ℕ :=
  (n / 10) + (n - n / 10 * 10)

theorem nines_in_hundred : count_nines 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_nines_in_hundred_l3177_317756


namespace NUMINAMATH_CALUDE_parentheses_removal_equality_l3177_317721

theorem parentheses_removal_equality (a c : ℝ) : 3*a - (2*a - c) = 3*a - 2*a + c := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_equality_l3177_317721


namespace NUMINAMATH_CALUDE_company_median_salary_l3177_317798

/-- Represents a job position with its title, number of employees, and salary --/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- The list of positions in the company --/
def company_positions : List Position := [
  { title := "President", count := 1, salary := 140000 },
  { title := "Vice-President", count := 10, salary := 100000 },
  { title := "Director", count := 15, salary := 80000 },
  { title := "Manager", count := 5, salary := 55000 },
  { title := "Associate Director", count := 9, salary := 52000 },
  { title := "Administrative Specialist", count := 35, salary := 25000 }
]

/-- The total number of employees in the company --/
def total_employees : Nat := 75

/-- Calculates the median salary of the company --/
def median_salary (positions : List Position) (total : Nat) : Nat :=
  sorry

/-- Theorem stating that the median salary of the company is $52,000 --/
theorem company_median_salary :
  median_salary company_positions total_employees = 52000 := by
  sorry

end NUMINAMATH_CALUDE_company_median_salary_l3177_317798


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_y_equals_5_l3177_317720

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, prove that if they are parallel, then y = 5 -/
theorem parallel_vectors_imply_y_equals_5 :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (4, y + 1)
  parallel a b → y = 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_y_equals_5_l3177_317720


namespace NUMINAMATH_CALUDE_wilson_prime_l3177_317782

theorem wilson_prime (n : ℕ) (h : n > 1) (h_div : n ∣ (Nat.factorial (n - 1) + 1)) : Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_wilson_prime_l3177_317782


namespace NUMINAMATH_CALUDE_distance_difference_l3177_317707

/-- The distance biked by Bjorn after six hours -/
def bjorn_distance : ℕ := 75

/-- The distance biked by Alberto after six hours -/
def alberto_distance : ℕ := 105

/-- Alberto bikes faster than Bjorn -/
axiom alberto_faster : alberto_distance > bjorn_distance

/-- The difference in distance biked between Alberto and Bjorn after six hours is 30 miles -/
theorem distance_difference : alberto_distance - bjorn_distance = 30 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l3177_317707


namespace NUMINAMATH_CALUDE_sum_of_eleven_terms_l3177_317751

def a (n : ℕ) : ℤ := 1 - 2 * n

def S (n : ℕ) : ℤ := n * (a 1 + a n) / 2

def sequence_sum (n : ℕ) : ℚ := 
  Finset.sum (Finset.range n) (λ i => S (i + 1) / (i + 1))

theorem sum_of_eleven_terms : sequence_sum 11 = -66 := by sorry

end NUMINAMATH_CALUDE_sum_of_eleven_terms_l3177_317751


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3177_317777

theorem expand_and_simplify (x y : ℝ) : 12 * (3 * x + 4 * y - 2) = 36 * x + 48 * y - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3177_317777


namespace NUMINAMATH_CALUDE_sequence_inequality_l3177_317718

theorem sequence_inequality (k : ℝ) : 
  (∀ n : ℕ+, (n + 1)^2 + k*(n + 1) + 2 > n^2 + k*n + 2) ↔ k > -3 :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3177_317718


namespace NUMINAMATH_CALUDE_locus_of_midpoints_l3177_317708

/-- The locus of midpoints theorem -/
theorem locus_of_midpoints 
  (P : ℝ × ℝ) 
  (h_P : P = (4, -2)) 
  (Q : ℝ × ℝ) 
  (h_Q : Q.1^2 + Q.2^2 = 4) 
  (M : ℝ × ℝ) 
  (h_M : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1 - 2)^2 + (M.2 + 1)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_locus_of_midpoints_l3177_317708


namespace NUMINAMATH_CALUDE_same_suit_in_rows_l3177_317761

/-- Represents a playing card suit -/
inductive Suit
| clubs
| diamonds
| hearts
| spades

/-- Represents a card in the grid -/
structure Card where
  suit : Suit
  rank : Nat

/-- Represents the 13 × 4 grid of cards -/
def CardGrid := Fin 13 → Fin 4 → Card

/-- Checks if two cards are adjacent -/
def adjacent (c1 c2 : Card) : Prop :=
  c1.suit = c2.suit ∨ c1.rank = c2.rank

/-- The condition that adjacent cards in the grid are of the same suit or rank -/
def adjacency_condition (grid : CardGrid) : Prop :=
  ∀ i j, (i.val < 12 → adjacent (grid i j) (grid (i + 1) j)) ∧
         (j.val < 3 → adjacent (grid i j) (grid i (j + 1)))

/-- The statement to be proved -/
theorem same_suit_in_rows (grid : CardGrid) 
  (h : adjacency_condition grid) : 
  ∀ j, ∀ i1 i2, (grid i1 j).suit = (grid i2 j).suit :=
sorry

end NUMINAMATH_CALUDE_same_suit_in_rows_l3177_317761


namespace NUMINAMATH_CALUDE_at_op_sum_equals_six_l3177_317768

-- Define the @ operation for positive integers
def at_op (a b : ℕ+) : ℚ := (a * b : ℚ) / (a + b : ℚ)

-- State the theorem
theorem at_op_sum_equals_six :
  at_op 7 14 + at_op 2 4 = 6 := by sorry

end NUMINAMATH_CALUDE_at_op_sum_equals_six_l3177_317768


namespace NUMINAMATH_CALUDE_race_distance_difference_l3177_317700

theorem race_distance_difference (race_distance : ℝ) (a_time b_time : ℝ) : 
  race_distance = 80 →
  a_time = 20 →
  b_time = 25 →
  let a_speed := race_distance / a_time
  let b_speed := race_distance / b_time
  let b_distance := b_speed * a_time
  race_distance - b_distance = 16 := by sorry

end NUMINAMATH_CALUDE_race_distance_difference_l3177_317700


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3177_317726

def M : Set ℝ := {y | 0 < y ∧ y < 1}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem sufficient_not_necessary : 
  (∀ x, x ∈ M → x ∈ N) ∧ 
  (∃ x, x ∈ N ∧ x ∉ M) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3177_317726


namespace NUMINAMATH_CALUDE_solve_system_l3177_317735

theorem solve_system (a b : ℚ) 
  (eq1 : 5 + 2 * a = 6 - 3 * b) 
  (eq2 : 3 + 4 * b = 10 + 2 * a) : 
  5 - 2 * a = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3177_317735


namespace NUMINAMATH_CALUDE_value_of_b_l3177_317765

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 4) : b = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l3177_317765


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l3177_317716

/-- The cubic polynomial q(x) that satisfies given conditions -/
def q (x : ℝ) : ℝ := 4 * x^3 - 19 * x^2 + 5 * x + 6

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions :
  q 0 = 6 ∧ q 1 = -4 ∧ q 2 = 0 ∧ q 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l3177_317716


namespace NUMINAMATH_CALUDE_stating_alice_probability_after_two_turns_l3177_317705

/-- The probability that Alice passes the ball to Bob -/
def alice_pass_prob : ℚ := 2/3

/-- The probability that Bob passes the ball to Alice -/
def bob_pass_prob : ℚ := 1/2

/-- The probability that Alice has the ball after two turns -/
def alice_has_ball_after_two_turns : ℚ := 4/9

/-- 
Theorem stating that given the game rules, the probability 
that Alice has the ball after two turns is 4/9 
-/
theorem alice_probability_after_two_turns : 
  alice_has_ball_after_two_turns = 
    (alice_pass_prob * bob_pass_prob) + ((1 - alice_pass_prob) * (1 - alice_pass_prob)) := by
  sorry

end NUMINAMATH_CALUDE_stating_alice_probability_after_two_turns_l3177_317705


namespace NUMINAMATH_CALUDE_josh_marbles_l3177_317794

/-- The number of marbles Josh has after receiving marbles from Jack -/
def total_marbles (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Josh has 42 marbles after receiving marbles from Jack -/
theorem josh_marbles :
  let initial_marbles : ℕ := 22
  let marbles_from_jack : ℕ := 20
  total_marbles initial_marbles marbles_from_jack = 42 := by
sorry

end NUMINAMATH_CALUDE_josh_marbles_l3177_317794


namespace NUMINAMATH_CALUDE_callum_points_l3177_317714

theorem callum_points (total_matches : ℕ) (krishna_win_ratio : ℚ) (points_per_win : ℕ) : 
  total_matches = 8 →
  krishna_win_ratio = 3/4 →
  points_per_win = 10 →
  (total_matches - (krishna_win_ratio * total_matches).num) * points_per_win = 20 := by
  sorry

end NUMINAMATH_CALUDE_callum_points_l3177_317714


namespace NUMINAMATH_CALUDE_pickle_problem_l3177_317763

/-- Pickle Problem -/
theorem pickle_problem (jars cucumbers initial_vinegar pickles_per_cucumber pickles_per_jar remaining_vinegar : ℕ)
  (h1 : jars = 4)
  (h2 : cucumbers = 10)
  (h3 : initial_vinegar = 100)
  (h4 : pickles_per_cucumber = 6)
  (h5 : pickles_per_jar = 12)
  (h6 : remaining_vinegar = 60) :
  (initial_vinegar - remaining_vinegar) / jars = 10 := by
  sorry


end NUMINAMATH_CALUDE_pickle_problem_l3177_317763


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3177_317734

/-- Given two quadratic equations and a relationship between their roots, prove the value of k. -/
theorem quadratic_root_relation (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 9 = 0 → ∃ y : ℝ, y^2 - k*y + 9 = 0 ∧ y = x + 3) →
  k = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3177_317734


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3177_317730

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions, prove that the 12th term is 14. -/
theorem arithmetic_sequence_12th_term
  (seq : ArithmeticSequence)
  (h1 : seq.a 7 + seq.a 9 = 15)
  (h2 : seq.a 4 = 1) :
  seq.a 12 = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l3177_317730


namespace NUMINAMATH_CALUDE_peach_basket_problem_l3177_317790

theorem peach_basket_problem (x : ℕ) : 
  (x > 0) →
  (x - (x / 2 + 1) > 0) →
  (x - (x / 2 + 1) - ((x - (x / 2 + 1)) / 2 - 1) = 4) →
  (x = 14) :=
by
  sorry

#check peach_basket_problem

end NUMINAMATH_CALUDE_peach_basket_problem_l3177_317790


namespace NUMINAMATH_CALUDE_half_circle_roll_midpoint_path_length_l3177_317783

/-- The length of the path traveled by the midpoint of a half-circle's diameter when rolled along a straight line -/
theorem half_circle_roll_midpoint_path_length 
  (diameter : ℝ) 
  (h_diameter : diameter = 4 / Real.pi) : 
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let path_length := circumference / 2
  path_length = 2 := by sorry

end NUMINAMATH_CALUDE_half_circle_roll_midpoint_path_length_l3177_317783


namespace NUMINAMATH_CALUDE_box_weight_example_l3177_317758

/-- Calculates the weight of an open box given its dimensions, thickness, and metal density. -/
def box_weight (length width height thickness : ℝ) (metal_density : ℝ) : ℝ :=
  let outer_volume := length * width * height
  let inner_length := length - 2 * thickness
  let inner_width := width - 2 * thickness
  let inner_height := height - thickness
  let inner_volume := inner_length * inner_width * inner_height
  let metal_volume := outer_volume - inner_volume
  metal_volume * metal_density

/-- Theorem stating that the weight of the specified box is 5504 grams. -/
theorem box_weight_example : 
  box_weight 50 40 23 2 0.5 = 5504 := by
  sorry

end NUMINAMATH_CALUDE_box_weight_example_l3177_317758


namespace NUMINAMATH_CALUDE_binary_10011_equals_19_l3177_317759

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10011_equals_19 :
  binary_to_decimal [true, true, false, false, true] = 19 := by
  sorry

end NUMINAMATH_CALUDE_binary_10011_equals_19_l3177_317759


namespace NUMINAMATH_CALUDE_circle_area_tripled_radius_l3177_317792

theorem circle_area_tripled_radius (r : ℝ) (hr : r > 0) :
  let A := π * r^2
  let A' := π * (3*r)^2
  A' = 9 * A ∧ A' ≠ 3 * A :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_tripled_radius_l3177_317792


namespace NUMINAMATH_CALUDE_jenny_distance_difference_l3177_317746

theorem jenny_distance_difference (run_distance walk_distance : ℝ) 
  (h1 : run_distance = 0.6)
  (h2 : walk_distance = 0.4) : 
  run_distance - walk_distance = 0.2 := by sorry

end NUMINAMATH_CALUDE_jenny_distance_difference_l3177_317746


namespace NUMINAMATH_CALUDE_prob_blank_one_shot_prob_blank_three_shots_prob_away_from_vertices_l3177_317749

-- Define the number of bullets and the number of blanks
def total_bullets : ℕ := 4
def blank_bullets : ℕ := 1

-- Define the number of shots
def shots : ℕ := 3

-- Define the side length of the equilateral triangle
def side_length : ℝ := 10

-- Theorem for the probability of shooting a blank in one shot
theorem prob_blank_one_shot : 
  (blank_bullets : ℝ) / total_bullets = 1 / 4 := by sorry

-- Theorem for the probability of a blank appearing in 3 shots
theorem prob_blank_three_shots : 
  1 - (total_bullets - blank_bullets : ℝ) * (total_bullets - blank_bullets - 1) * (total_bullets - blank_bullets - 2) / 
    (total_bullets * (total_bullets - 1) * (total_bullets - 2)) = 3 / 4 := by sorry

-- Theorem for the probability of all shots being more than 1 unit away from vertices
theorem prob_away_from_vertices (triangle_area : ℝ) (h : triangle_area = side_length^2 * Real.sqrt 3 / 4) :
  1 - (3 * π / 2) / triangle_area = 1 - Real.sqrt 3 * π / 150 := by sorry

end NUMINAMATH_CALUDE_prob_blank_one_shot_prob_blank_three_shots_prob_away_from_vertices_l3177_317749


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3177_317704

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ≥ 10) →
  (∀ x ∈ Set.Icc 1 3, x^2 - a ≤ 0) ∧
  ¬(∀ a : ℝ, (∀ x ∈ Set.Icc 1 3, x^2 - a ≤ 0) → (a ≥ 10)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3177_317704


namespace NUMINAMATH_CALUDE_multiple_of_six_is_multiple_of_three_l3177_317764

theorem multiple_of_six_is_multiple_of_three (n : ℤ) :
  (∃ k : ℤ, n = 6 * k) → (∃ m : ℤ, n = 3 * m) := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_six_is_multiple_of_three_l3177_317764


namespace NUMINAMATH_CALUDE_vector_relations_l3177_317757

/-- Two vectors in R² -/
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (2*x + 3, -x)

/-- Perpendicular vectors have dot product zero -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Parallel vectors have proportional components -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_relations (x : ℝ) :
  (perpendicular (a x) (b x) → x = 3 ∨ x = -1) ∧
  (parallel (a x) (b x) → x = 0 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_relations_l3177_317757


namespace NUMINAMATH_CALUDE_packet_B_height_l3177_317747

/-- Growth rate of Packet A sunflowers -/
def R_A (x y : ℝ) : ℝ := 2 * x + y

/-- Growth rate of Packet B sunflowers -/
def R_B (x y : ℝ) : ℝ := 3 * x - y

/-- Theorem stating the height of Packet B sunflowers on day 10 -/
theorem packet_B_height (h_A : ℝ) (h_B : ℝ) :
  R_A 10 6 = 26 →
  R_B 10 6 = 24 →
  h_A = 192 →
  h_A = h_B + 0.2 * h_B →
  h_B = 160 := by
  sorry

#check packet_B_height

end NUMINAMATH_CALUDE_packet_B_height_l3177_317747
