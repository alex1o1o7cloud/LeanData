import Mathlib

namespace NUMINAMATH_CALUDE_sector_central_angle_l2990_299005

-- Define the sector
structure Sector where
  circumference : ℝ
  area : ℝ

-- Define the given sector
def given_sector : Sector := { circumference := 6, area := 2 }

-- Define the possible central angles
def possible_angles : Set ℝ := {1, 4}

-- Theorem statement
theorem sector_central_angle (s : Sector) (h1 : s = given_sector) :
  ∃ θ ∈ possible_angles, 
    ∃ r l : ℝ, 
      r > 0 ∧ 
      l > 0 ∧ 
      2 * r + l = s.circumference ∧ 
      1 / 2 * r * l = s.area ∧ 
      θ = l / r :=
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2990_299005


namespace NUMINAMATH_CALUDE_smallest_multiple_of_9_and_21_l2990_299073

theorem smallest_multiple_of_9_and_21 :
  ∃ (b : ℕ), b > 0 ∧ 9 ∣ b ∧ 21 ∣ b ∧ ∀ (x : ℕ), x > 0 ∧ 9 ∣ x ∧ 21 ∣ x → b ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_9_and_21_l2990_299073


namespace NUMINAMATH_CALUDE_simplify_expression_l2990_299056

theorem simplify_expression : 15 * (7 / 10) * (1 / 9) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2990_299056


namespace NUMINAMATH_CALUDE_volume_of_tetrahedron_OCDE_l2990_299095

/-- Square ABCD with side length 2 -/
def square_ABCD : Set (ℝ × ℝ) := sorry

/-- Point E is the midpoint of AB -/
def point_E : ℝ × ℝ := sorry

/-- Point O is formed when A and B coincide after folding -/
def point_O : ℝ × ℝ := sorry

/-- Triangle OCD formed after folding -/
def triangle_OCD : Set (ℝ × ℝ) := sorry

/-- Tetrahedron O-CDE formed after folding -/
def tetrahedron_OCDE : Set (ℝ × ℝ × ℝ) := sorry

/-- Volume of a tetrahedron -/
def tetrahedron_volume (t : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

theorem volume_of_tetrahedron_OCDE : 
  tetrahedron_volume tetrahedron_OCDE = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_tetrahedron_OCDE_l2990_299095


namespace NUMINAMATH_CALUDE_min_value_of_function_l2990_299075

theorem min_value_of_function (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (min_val : ℝ), min_val = (a^(2/3) + b^(2/3))^(3/2) ∧
  ∀ θ : ℝ, θ ∈ Set.Ioo 0 (π/2) →
    a / Real.sin θ + b / Real.cos θ ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2990_299075


namespace NUMINAMATH_CALUDE_number_calculation_l2990_299052

theorem number_calculation (x y : ℝ) (h1 : 0.5 * x = 0.25 * y - 30) (h2 : x = 690) : y = 1500 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l2990_299052


namespace NUMINAMATH_CALUDE_sum_product_inequality_l2990_299008

theorem sum_product_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) : 
  a * b + b * c + c * d + d * a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l2990_299008


namespace NUMINAMATH_CALUDE_sqrt_eighteen_div_sqrt_two_equals_three_l2990_299043

theorem sqrt_eighteen_div_sqrt_two_equals_three :
  Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eighteen_div_sqrt_two_equals_three_l2990_299043


namespace NUMINAMATH_CALUDE_age_sum_in_six_years_l2990_299064

/-- Melanie's current age -/
def melanie_age : ℕ := sorry

/-- Phil's current age -/
def phil_age : ℕ := sorry

/-- The sum of Melanie's and Phil's ages 6 years from now is 42, 
    given that in 10 years, the product of their ages will be 400 more than it is now. -/
theorem age_sum_in_six_years : 
  (melanie_age + 10) * (phil_age + 10) = melanie_age * phil_age + 400 →
  (melanie_age + 6) + (phil_age + 6) = 42 := by sorry

end NUMINAMATH_CALUDE_age_sum_in_six_years_l2990_299064


namespace NUMINAMATH_CALUDE_parentheses_removal_l2990_299059

theorem parentheses_removal (a b c d : ℝ) : a - (b - c + d) = a - b + c - d := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_l2990_299059


namespace NUMINAMATH_CALUDE_length_of_EG_l2990_299078

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit

-- Define the radius of the circle
def radius : ℝ := 7

-- Define the points E, F, and G on the circle
def E : Point := sorry
def F : Point := sorry
def G : Point := sorry

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define the central angle subtended by an arc
def centralAngle (p q : Point) (c : Circle) : ℝ := sorry

-- State the theorem
theorem length_of_EG (c : Circle) : 
  distance E F = 8 → 
  centralAngle E G c = π / 3 → 
  distance E G = 7 := by sorry

end NUMINAMATH_CALUDE_length_of_EG_l2990_299078


namespace NUMINAMATH_CALUDE_puppies_sold_l2990_299019

/-- Given a pet store scenario, prove the number of puppies sold -/
theorem puppies_sold (initial_puppies cages_used puppies_per_cage : ℕ) :
  initial_puppies - (cages_used * puppies_per_cage) =
  initial_puppies - cages_used * puppies_per_cage :=
by sorry

end NUMINAMATH_CALUDE_puppies_sold_l2990_299019


namespace NUMINAMATH_CALUDE_ab_one_sufficient_not_necessary_l2990_299012

theorem ab_one_sufficient_not_necessary (a b : ℝ) : 
  (a * b = 1 → a^2 + b^2 ≥ 2) ∧ 
  ∃ a b : ℝ, a^2 + b^2 ≥ 2 ∧ a * b ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_ab_one_sufficient_not_necessary_l2990_299012


namespace NUMINAMATH_CALUDE_expression_value_l2990_299047

theorem expression_value : (85 + 32 / 113) * 113 = 9637 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2990_299047


namespace NUMINAMATH_CALUDE_system_solution_is_e_l2990_299006

theorem system_solution_is_e (x y z : ℝ) : 
  x = Real.exp (Real.log y) ∧ 
  y = Real.exp (Real.log z) ∧ 
  z = Real.exp (Real.log x) → 
  x = y ∧ y = z ∧ x = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_is_e_l2990_299006


namespace NUMINAMATH_CALUDE_catch_up_distance_l2990_299020

/-- Proves that B catches up with A 100 km from the start given the specified conditions -/
theorem catch_up_distance (speed_a speed_b : ℝ) (delay : ℝ) (catch_up_dist : ℝ) : 
  speed_a = 10 →
  speed_b = 20 →
  delay = 5 →
  catch_up_dist = speed_b * (catch_up_dist / (speed_b - speed_a)) →
  catch_up_dist = speed_a * (delay + catch_up_dist / (speed_b - speed_a)) →
  catch_up_dist = 100 := by
  sorry

#check catch_up_distance

end NUMINAMATH_CALUDE_catch_up_distance_l2990_299020


namespace NUMINAMATH_CALUDE_white_washing_cost_l2990_299092

-- Define the room dimensions
def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12

-- Define the door dimensions
def door_height : ℝ := 6
def door_width : ℝ := 3

-- Define the window dimensions
def window_height : ℝ := 4
def window_width : ℝ := 3

-- Define the number of windows
def num_windows : ℕ := 3

-- Define the cost per square foot
def cost_per_sqft : ℝ := 7

-- Theorem statement
theorem white_washing_cost :
  let total_wall_area := 2 * (room_length + room_width) * room_height
  let door_area := door_height * door_width
  let window_area := window_height * window_width * num_windows
  let adjusted_wall_area := total_wall_area - door_area - window_area
  adjusted_wall_area * cost_per_sqft = 6342 := by
  sorry


end NUMINAMATH_CALUDE_white_washing_cost_l2990_299092


namespace NUMINAMATH_CALUDE_divisibility_when_prime_exists_counterexample_for_composite_l2990_299010

-- Define the divisibility relation
def divides (a b : ℕ) : Prop := ∃ k, b = a * k

-- Statement for the case when n is prime
theorem divisibility_when_prime (m n : ℕ) (h1 : n > 1) (h2 : ∀ k, 1 < k → k < n → ¬ divides k n) 
  (h3 : divides (m + n) (m * n)) : divides n m := by sorry

-- Statement for the case when n is a product of two distinct primes
theorem exists_counterexample_for_composite : 
  ∃ m n p q : ℕ, p ≠ q ∧ p.Prime ∧ q.Prime ∧ n = p * q ∧ 
  divides (m + n) (m * n) ∧ ¬ divides n m := by sorry

end NUMINAMATH_CALUDE_divisibility_when_prime_exists_counterexample_for_composite_l2990_299010


namespace NUMINAMATH_CALUDE_sarah_walk_probability_l2990_299091

/-- The number of gates at the airport -/
def num_gates : ℕ := 15

/-- The distance between adjacent gates in feet -/
def gate_distance : ℕ := 80

/-- The maximum distance Sarah is willing to walk in feet -/
def max_walk_distance : ℕ := 320

/-- The probability that Sarah walks 320 feet or less to her new gate -/
theorem sarah_walk_probability : 
  (num_gates : ℚ) * (max_walk_distance / gate_distance * 2 : ℚ) / 
  ((num_gates : ℚ) * (num_gates - 1 : ℚ)) = 4 / 7 := by sorry

end NUMINAMATH_CALUDE_sarah_walk_probability_l2990_299091


namespace NUMINAMATH_CALUDE_sum_binary_digits_350_1350_l2990_299044

/-- The number of digits in the binary representation of a positive integer -/
def binaryDigits (n : ℕ+) : ℕ :=
  Nat.log2 n + 1

/-- The sum of binary digits for 350 and 1350 -/
def sumBinaryDigits : ℕ := binaryDigits 350 + binaryDigits 1350

theorem sum_binary_digits_350_1350 : sumBinaryDigits = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_binary_digits_350_1350_l2990_299044


namespace NUMINAMATH_CALUDE_symmetric_intersection_theorem_l2990_299038

/-- A line that intersects a circle at two points symmetric about another line -/
structure SymmetricIntersection where
  /-- The coefficient of x in the line equation ax + 2y - 2 = 0 -/
  a : ℝ
  /-- The first intersection point -/
  A : ℝ × ℝ
  /-- The second intersection point -/
  B : ℝ × ℝ

/-- The line ax + 2y - 2 = 0 intersects the circle (x-1)² + (y+1)² = 6 -/
def intersects_circle (si : SymmetricIntersection) : Prop :=
  let (x₁, y₁) := si.A
  let (x₂, y₂) := si.B
  si.a * x₁ + 2 * y₁ - 2 = 0 ∧
  si.a * x₂ + 2 * y₂ - 2 = 0 ∧
  (x₁ - 1)^2 + (y₁ + 1)^2 = 6 ∧
  (x₂ - 1)^2 + (y₂ + 1)^2 = 6

/-- A and B are symmetric with respect to the line x + y = 0 -/
def symmetric_about_line (si : SymmetricIntersection) : Prop :=
  let (x₁, y₁) := si.A
  let (x₂, y₂) := si.B
  x₁ + y₁ = -(x₂ + y₂)

/-- The main theorem: if the conditions are met, then a = -2 -/
theorem symmetric_intersection_theorem (si : SymmetricIntersection) :
  intersects_circle si → symmetric_about_line si → si.a = -2 :=
sorry

end NUMINAMATH_CALUDE_symmetric_intersection_theorem_l2990_299038


namespace NUMINAMATH_CALUDE_ellipse_and_circle_intersection_l2990_299088

/-- The ellipse C₁ -/
def C₁ (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

/-- The parabola C₂ -/
def C₂ (x y : ℝ) : Prop := y^2 = 4*x

/-- The circle C₃ -/
def C₃ (x y x₀ y₀ r : ℝ) : Prop := (x - x₀)^2 + (y - y₀)^2 = r^2

/-- The point P -/
def P (x y : ℝ) : Prop := C₁ x y 2 (Real.sqrt 3) ∧ C₂ x y ∧ x > 0 ∧ y > 0

/-- The point T -/
def T (x y : ℝ) : Prop := C₂ x y

theorem ellipse_and_circle_intersection 
  (a b : ℝ) 
  (h₁ : a > b) 
  (h₂ : b > 0) 
  (h₃ : ∃ x y, P x y ∧ (x - 1)^2 + y^2 = (5/3)^2) 
  (h₄ : ∀ x₀ y₀ r, T x₀ y₀ → C₃ 0 2 x₀ y₀ r → C₃ 0 (-2) x₀ y₀ r → r^2 = 4 + x₀^2) :
  (∀ x y, C₁ x y a b ↔ C₁ x y 2 (Real.sqrt 3)) ∧ 
  (∀ x₀ y₀ r, T x₀ y₀ → C₃ 2 0 x₀ y₀ r) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_intersection_l2990_299088


namespace NUMINAMATH_CALUDE_cricket_average_l2990_299003

theorem cricket_average (innings : ℕ) (next_runs : ℕ) (increase : ℕ) (current_average : ℕ) : 
  innings = 12 → 
  next_runs = 178 → 
  increase = 10 → 
  (innings * current_average + next_runs) / (innings + 1) = current_average + increase → 
  current_average = 48 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_l2990_299003


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2990_299002

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A point on the ellipse -/
def PointOnEllipse (C : Ellipse) := ℝ × ℝ

/-- The angle F₁MF₂ for a point M on the ellipse -/
def angle (C : Ellipse) (M : PointOnEllipse C) : ℝ := sorry

/-- The eccentricity of an ellipse -/
def eccentricity (C : Ellipse) : ℝ := sorry

/-- Theorem: If there exists a point M on ellipse C such that ∠F₁MF₂ = π/3,
    then the eccentricity e of C satisfies 1/2 ≤ e < 1 -/
theorem ellipse_eccentricity_range (C : Ellipse) :
  (∃ M : PointOnEllipse C, angle C M = π / 3) →
  let e := eccentricity C
  1 / 2 ≤ e ∧ e < 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2990_299002


namespace NUMINAMATH_CALUDE_coin_problem_l2990_299098

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- The value of each coin type in cents --/
def coinValue : CoinType → Nat
  | CoinType.Penny => 1
  | CoinType.Nickel => 5
  | CoinType.Dime => 10
  | CoinType.Quarter => 25
  | CoinType.HalfDollar => 50

/-- A collection of coins --/
structure CoinCollection where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  quarters : Nat
  halfDollars : Nat

/-- The total number of coins in a collection --/
def CoinCollection.totalCoins (c : CoinCollection) : Nat :=
  c.pennies + c.nickels + c.dimes + c.quarters + c.halfDollars

/-- The total value of a coin collection in cents --/
def CoinCollection.totalValue (c : CoinCollection) : Nat :=
  c.pennies * coinValue CoinType.Penny +
  c.nickels * coinValue CoinType.Nickel +
  c.dimes * coinValue CoinType.Dime +
  c.quarters * coinValue CoinType.Quarter +
  c.halfDollars * coinValue CoinType.HalfDollar

/-- The main theorem to prove --/
theorem coin_problem :
  ∀ (c : CoinCollection),
    c.totalCoins = 12 ∧
    c.totalValue = 166 ∧
    c.pennies ≥ 1 ∧
    c.nickels ≥ 1 ∧
    c.dimes ≥ 1 ∧
    c.quarters ≥ 1 ∧
    c.halfDollars ≥ 1
    →
    c.quarters = 3 :=
by sorry

end NUMINAMATH_CALUDE_coin_problem_l2990_299098


namespace NUMINAMATH_CALUDE_perpendicular_and_tangent_l2990_299027

-- Define the given curve
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

-- Define the given line
def l₁ (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

-- Define the line we want to prove
def l₂ (x y : ℝ) : Prop := 3*x + y + 6 = 0

-- Theorem statement
theorem perpendicular_and_tangent :
  ∃ (x₀ y₀ : ℝ),
    -- l₂ is perpendicular to l₁
    (∀ (x₁ y₁ x₂ y₂ : ℝ), l₁ x₁ y₁ → l₁ x₂ y₂ → l₂ x₁ y₁ → l₂ x₂ y₂ →
      (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
      ((x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁)) *
      ((x₂ - x₁) * (3) + (y₂ - y₁) * (1)) = 0) ∧
    -- l₂ is tangent to f at (x₀, y₀)
    (l₂ x₀ y₀ ∧ f x₀ = y₀ ∧
      ∀ (x : ℝ), x ≠ x₀ → l₂ x (f x) → False) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_and_tangent_l2990_299027


namespace NUMINAMATH_CALUDE_age_ratio_l2990_299022

theorem age_ratio (a b : ℕ) : 
  (a - 4 = b + 4) → 
  ((a + 4) = 3 * (b - 4)) → 
  (a : ℚ) / b = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_l2990_299022


namespace NUMINAMATH_CALUDE_hash_difference_eight_five_l2990_299039

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

-- Theorem statement
theorem hash_difference_eight_five : hash 8 5 - hash 5 8 = -12 := by sorry

end NUMINAMATH_CALUDE_hash_difference_eight_five_l2990_299039


namespace NUMINAMATH_CALUDE_special_sequence_is_arithmetic_l2990_299087

/-- A sequence satisfying specific conditions -/
def SpecialSequence (a : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, a (n + 1) > a n ∧ a n > 0) ∧
  (∀ n : ℕ+, a n - 1 / a n < n ∧ n < a n + 1 / a n) ∧
  (∃ m : ℕ+, m ≥ 2 ∧ ∀ n : ℕ+, a (m * n) = m * a n)

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

/-- Theorem: A special sequence is an arithmetic sequence -/
theorem special_sequence_is_arithmetic (a : ℕ+ → ℝ) 
    (h : SpecialSequence a) : ArithmeticSequence a := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_is_arithmetic_l2990_299087


namespace NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l2990_299015

theorem circle_y_axis_intersection_sum (h k r : ℝ) : 
  h = -3 → k = 5 → r = 8 → 
  (k + (r^2 - h^2).sqrt) + (k - (r^2 - h^2).sqrt) = 10 := by sorry

end NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l2990_299015


namespace NUMINAMATH_CALUDE_f_max_min_l2990_299042

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

-- State the theorem
theorem f_max_min :
  (∀ x, f x ≤ 15) ∧ (∃ x, f x = 15) ∧
  (∀ x, f x ≥ -1) ∧ (∃ x, f x = -1) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_l2990_299042


namespace NUMINAMATH_CALUDE_cricket_average_l2990_299029

theorem cricket_average (initial_average : ℝ) : 
  (8 * initial_average + 90) / 9 = initial_average + 6 → 
  initial_average + 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_l2990_299029


namespace NUMINAMATH_CALUDE_asterisk_replacement_l2990_299032

theorem asterisk_replacement : ∃ x : ℝ, (x / 21) * (x / 84) = 1 ∧ x = 42 := by sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l2990_299032


namespace NUMINAMATH_CALUDE_tracy_art_fair_sales_l2990_299021

/-- The number of paintings sold at Tracy's art fair booth --/
def paintings_sold (group1_count group1_paintings group2_count group2_paintings group3_count group3_paintings : ℕ) : ℕ :=
  group1_count * group1_paintings + group2_count * group2_paintings + group3_count * group3_paintings

/-- Theorem stating the total number of paintings sold at Tracy's art fair booth --/
theorem tracy_art_fair_sales : paintings_sold 4 2 12 1 4 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_tracy_art_fair_sales_l2990_299021


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2990_299025

/-- The line (k+1)x-(2k-1)y+3k=0 always passes through the point (-1, 1) for all real k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k + 1) * (-1) - (2 * k - 1) * 1 + 3 * k = 0 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2990_299025


namespace NUMINAMATH_CALUDE_tetrahedron_properties_l2990_299085

def A₁ : ℝ × ℝ × ℝ := (2, -1, 2)
def A₂ : ℝ × ℝ × ℝ := (1, 2, -1)
def A₃ : ℝ × ℝ × ℝ := (3, 2, 1)
def A₄ : ℝ × ℝ × ℝ := (-4, 2, 5)

def tetrahedron_volume (A B C D : ℝ × ℝ × ℝ) : ℝ := sorry

def tetrahedron_height (A B C D : ℝ × ℝ × ℝ) : ℝ := sorry

theorem tetrahedron_properties :
  tetrahedron_volume A₁ A₂ A₃ A₄ = 11 ∧
  tetrahedron_height A₄ A₁ A₂ A₃ = 3 * Real.sqrt (11 / 2) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_properties_l2990_299085


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2990_299077

-- Define the polynomial
def polynomial (p : ℝ) (x : ℝ) : ℝ := 4 * x^3 - 12 * x^2 + p * x - 16

-- Define divisibility condition
def is_divisible_by (f : ℝ → ℝ) (a : ℝ) : Prop := f a = 0

-- Theorem statement
theorem polynomial_divisibility (p : ℝ) :
  (is_divisible_by (polynomial p) 2) →
  (is_divisible_by (polynomial p) 4 ↔ p = 16) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2990_299077


namespace NUMINAMATH_CALUDE_circle_equation_l2990_299086

/-- Given a circle with center at (a,1) tangent to two lines, prove its standard equation -/
theorem circle_equation (a : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 
    (∀ (x y : ℝ), (2*x - y + 4 = 0 ∨ 2*x - y - 6 = 0) → 
      ((x - a)^2 + (y - 1)^2 = r^2))) →
  (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2990_299086


namespace NUMINAMATH_CALUDE_sandwiches_per_person_l2990_299076

def mini_croissants_per_set : ℕ := 12
def cost_per_set : ℕ := 8
def committee_size : ℕ := 24
def total_spent : ℕ := 32

theorem sandwiches_per_person :
  (total_spent / cost_per_set) * mini_croissants_per_set / committee_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_sandwiches_per_person_l2990_299076


namespace NUMINAMATH_CALUDE_rate_squares_sum_l2990_299011

theorem rate_squares_sum : ∃ (b j s : ℕ), b + j + s = 34 ∧ b^2 + j^2 + s^2 = 406 := by
  sorry

end NUMINAMATH_CALUDE_rate_squares_sum_l2990_299011


namespace NUMINAMATH_CALUDE_two_triangles_max_parts_two_rectangles_max_parts_two_n_gons_max_parts_l2990_299031

/-- The maximum number of parts into which two polygons can divide a plane -/
def max_parts (sides : ℕ) : ℕ := 2 * sides + 2

/-- Two triangles can divide a plane into at most 8 parts -/
theorem two_triangles_max_parts : max_parts 3 = 8 := by sorry

/-- Two rectangles can divide a plane into at most 10 parts -/
theorem two_rectangles_max_parts : max_parts 4 = 10 := by sorry

/-- Two convex n-gons can divide a plane into at most 2n + 2 parts -/
theorem two_n_gons_max_parts (n : ℕ) : max_parts n = 2 * n + 2 := by sorry

end NUMINAMATH_CALUDE_two_triangles_max_parts_two_rectangles_max_parts_two_n_gons_max_parts_l2990_299031


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l2990_299049

theorem sine_cosine_inequality (n : ℕ+) (x : ℝ) :
  (Real.sin (2 * x))^(n : ℝ) + (Real.sin x^(n : ℝ) - Real.cos x^(n : ℝ))^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l2990_299049


namespace NUMINAMATH_CALUDE_two_six_digit_squares_decomposable_l2990_299084

/-- A function that checks if a number is a two-digit square -/
def isTwoDigitSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, 4 ≤ k ∧ k ≤ 9 ∧ n = k^2

/-- A function that checks if a 6-digit number can be decomposed into three two-digit squares -/
def isDecomposableIntoThreeTwoDigitSquares (n : ℕ) : Prop :=
  ∃ a b c : ℕ,
    isTwoDigitSquare a ∧
    isTwoDigitSquare b ∧
    isTwoDigitSquare c ∧
    n = a * 10000 + b * 100 + c

/-- The main theorem stating that there are exactly two 6-digit perfect squares
    that can be decomposed into three two-digit perfect squares -/
theorem two_six_digit_squares_decomposable :
  ∃! (s : Finset ℕ),
    s.card = 2 ∧
    (∀ n ∈ s, 100000 ≤ n ∧ n < 1000000) ∧
    (∀ n ∈ s, ∃ k : ℕ, n = k^2) ∧
    (∀ n ∈ s, isDecomposableIntoThreeTwoDigitSquares n) :=
  sorry

end NUMINAMATH_CALUDE_two_six_digit_squares_decomposable_l2990_299084


namespace NUMINAMATH_CALUDE_average_height_theorem_l2990_299035

/-- The height difference between Itzayana and Zora in inches -/
def height_diff_itzayana_zora : ℝ := 4

/-- The height difference between Brixton and Zora in inches -/
def height_diff_brixton_zora : ℝ := 8

/-- Zara's height in inches -/
def height_zara : ℝ := 64

/-- Jaxon's height in centimeters -/
def height_jaxon_cm : ℝ := 170

/-- Conversion factor from centimeters to inches -/
def cm_to_inch : ℝ := 2.54

/-- The number of people -/
def num_people : ℕ := 5

theorem average_height_theorem :
  let height_brixton : ℝ := height_zara
  let height_zora : ℝ := height_brixton - height_diff_brixton_zora
  let height_itzayana : ℝ := height_zora + height_diff_itzayana_zora
  let height_jaxon : ℝ := height_jaxon_cm / cm_to_inch
  (height_itzayana + height_zora + height_brixton + height_zara + height_jaxon) / num_people = 62.2 := by
  sorry

end NUMINAMATH_CALUDE_average_height_theorem_l2990_299035


namespace NUMINAMATH_CALUDE_smallest_sum_l2990_299097

theorem smallest_sum (E F G H : ℕ+) : 
  (∃ d : ℤ, (E : ℤ) + d = F ∧ (F : ℤ) + d = G) →  -- arithmetic sequence
  (∃ r : ℚ, F * r = G ∧ G * r = H) →  -- geometric sequence
  G = (7 : ℚ) / 4 * F →  -- G/F = 7/4
  E + F + G + H ≥ 97 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_l2990_299097


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_lines_l2990_299034

/-- Given three lines in a plane, prove that they form an isosceles triangle -/
theorem isosceles_triangle_from_lines :
  let line1 : ℝ → ℝ := λ x => 4 * x + 3
  let line2 : ℝ → ℝ := λ x => -4 * x + 3
  let line3 : ℝ → ℝ := λ _ => -3
  let point1 : ℝ × ℝ := (0, 3)
  let point2 : ℝ × ℝ := (-3/2, -3)
  let point3 : ℝ × ℝ := (3/2, -3)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  true →
  ∃ (a b : ℝ), a = distance point1 point2 ∧ 
                a = distance point1 point3 ∧ 
                b = distance point2 point3 ∧
                a ≠ b :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_lines_l2990_299034


namespace NUMINAMATH_CALUDE_population_characteristics_changeable_l2990_299068

/-- Represents a population of organisms -/
structure Population where
  species : Type
  individuals : Set species
  space : Type
  time : Type

/-- Characteristics of a population -/
structure PopulationCharacteristics where
  density : ℝ
  birth_rate : ℝ
  death_rate : ℝ
  immigration_rate : ℝ
  age_composition : Set ℕ
  sex_ratio : ℝ

/-- A population has characteristics that can change over time -/
def population_characteristics_can_change (p : Population) : Prop :=
  ∃ (t₁ t₂ : p.time) (c₁ c₂ : PopulationCharacteristics),
    t₁ ≠ t₂ → c₁ ≠ c₂

/-- The main theorem stating that population characteristics can change over time -/
theorem population_characteristics_changeable :
  ∀ (p : Population), population_characteristics_can_change p :=
sorry

end NUMINAMATH_CALUDE_population_characteristics_changeable_l2990_299068


namespace NUMINAMATH_CALUDE_abs_a_plus_inv_a_geq_two_l2990_299045

theorem abs_a_plus_inv_a_geq_two (a : ℝ) (h : a ≠ 0) : |a + 1/a| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_a_plus_inv_a_geq_two_l2990_299045


namespace NUMINAMATH_CALUDE_eleven_steps_seven_moves_l2990_299051

/-- The number of ways to climb a staircase with a given number of steps in a fixed number of moves. -/
def climbStairs (totalSteps : ℕ) (requiredMoves : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem stating that there are 35 ways to climb 11 steps in 7 moves -/
theorem eleven_steps_seven_moves : climbStairs 11 7 = 35 := by
  sorry

end NUMINAMATH_CALUDE_eleven_steps_seven_moves_l2990_299051


namespace NUMINAMATH_CALUDE_equation_solution_l2990_299036

theorem equation_solution : ∃! x : ℝ, (9 - x)^2 = x^2 ∧ x = (9 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2990_299036


namespace NUMINAMATH_CALUDE_ibrahim_savings_is_55_l2990_299016

/-- The amount of money Ibrahim has in savings -/
def ibrahimSavings (mp3Cost cdCost fatherContribution amountLacking : ℕ) : ℕ :=
  (mp3Cost + cdCost) - fatherContribution - amountLacking

/-- Theorem stating that Ibrahim's savings are 55 euros -/
theorem ibrahim_savings_is_55 :
  ibrahimSavings 120 19 20 64 = 55 := by
  sorry

end NUMINAMATH_CALUDE_ibrahim_savings_is_55_l2990_299016


namespace NUMINAMATH_CALUDE_intersection_points_imply_c_value_l2990_299080

def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

theorem intersection_points_imply_c_value :
  ∀ c : ℝ, (∃! (a b : ℝ), a ≠ b ∧ f c a = 0 ∧ f c b = 0 ∧ 
    (∀ x : ℝ, f c x = 0 → x = a ∨ x = b)) →
  c = -2 ∨ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_imply_c_value_l2990_299080


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l2990_299017

-- Define the function f(x) = 2|x|
def f (x : ℝ) : ℝ := 2 * abs x

-- State the theorem
theorem f_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧ -- f is even
  (∀ x y : ℝ, 0 < x ∧ x < y → f x < f y) -- f is increasing on (0, +∞)
  := by sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l2990_299017


namespace NUMINAMATH_CALUDE_l_shape_surface_area_l2990_299096

/-- Represents the "L" shaped solid formed by unit cubes -/
structure LShape where
  bottom_row : ℕ
  vertical_stack : ℕ
  total_cubes : ℕ

/-- Calculates the surface area of the L-shaped solid -/
def surface_area (shape : LShape) : ℕ :=
  let bottom_exposure := shape.bottom_row + (shape.bottom_row - 1)
  let vertical_stack_exposure := 4 * shape.vertical_stack + 1
  let bottom_sides := 2 + shape.bottom_row
  bottom_exposure + vertical_stack_exposure + bottom_sides

/-- Theorem stating that the surface area of the specific L-shaped solid is 26 square units -/
theorem l_shape_surface_area :
  let shape : LShape := ⟨4, 3, 7⟩
  surface_area shape = 26 := by
  sorry

end NUMINAMATH_CALUDE_l_shape_surface_area_l2990_299096


namespace NUMINAMATH_CALUDE_equal_angles_point_exists_l2990_299079

-- Define the structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the function to calculate the angle between three points
def angle (p1 p2 p3 : Point2D) : ℝ := sorry

-- Define the theorem
theorem equal_angles_point_exists (A B C D : Point2D) 
  (h_collinear : ∃ (t : ℝ), B.x = A.x + t * (D.x - A.x) ∧ 
                             B.y = A.y + t * (D.y - A.y) ∧ 
                             C.x = A.x + t * (D.x - A.x) ∧ 
                             C.y = A.y + t * (D.y - A.y)) :
  ∃ (M : Point2D), angle A M B = angle B M C ∧ angle B M C = angle C M D := by
  sorry

end NUMINAMATH_CALUDE_equal_angles_point_exists_l2990_299079


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_perpendicular_plane_and_contained_implies_perpendicular_l2990_299074

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (contained : Line → Plane → Prop)

-- Non-coincident lines and planes
variable (m n : Line)
variable (α β : Plane)
variable (h_diff_lines : m ≠ n)
variable (h_diff_planes : α ≠ β)

-- Theorem 1
theorem perpendicular_implies_parallel 
  (h1 : perpendicular_plane m α) 
  (h2 : perpendicular_plane m β) : 
  parallel_plane α β :=
sorry

-- Theorem 2
theorem perpendicular_plane_and_contained_implies_perpendicular 
  (h1 : perpendicular_plane m α) 
  (h2 : contained n α) : 
  perpendicular m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_perpendicular_plane_and_contained_implies_perpendicular_l2990_299074


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2990_299013

theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 2 = 4 →
  a 4 = 2 →
  a 8 = -2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2990_299013


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l2990_299061

theorem divisibility_implies_equality (a b n : ℕ) :
  (∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) →
  a = b^n :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l2990_299061


namespace NUMINAMATH_CALUDE_smallest_NPP_l2990_299050

def is_two_digit_with_equal_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n % 11 = 0

def is_one_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def last_two_digits_equal (n : ℕ) : Prop :=
  (n % 100) % 11 = 0

theorem smallest_NPP :
  ∃ (M N P : ℕ),
    is_two_digit_with_equal_digits (11 * M) ∧
    is_one_digit N ∧
    is_three_digit (100 * N + 10 * P + P) ∧
    11 * M * N = 100 * N + 10 * P + P ∧
    (∀ (M' N' P' : ℕ),
      is_two_digit_with_equal_digits (11 * M') →
      is_one_digit N' →
      is_three_digit (100 * N' + 10 * P' + P') →
      11 * M' * N' = 100 * N' + 10 * P' + P' →
      100 * N + 10 * P + P ≤ 100 * N' + 10 * P' + P') ∧
    M = 2 ∧ N = 3 ∧ P = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_NPP_l2990_299050


namespace NUMINAMATH_CALUDE_box_depth_calculation_l2990_299089

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents a cube -/
structure Cube where
  sideLength : ℕ

def BoxDimensions.volume (b : BoxDimensions) : ℕ :=
  b.length * b.width * b.depth

def Cube.volume (c : Cube) : ℕ :=
  c.sideLength ^ 3

/-- The theorem to be proved -/
theorem box_depth_calculation (box : BoxDimensions) (cube : Cube) :
  box.length = 30 →
  box.width = 48 →
  (80 * cube.volume = box.volume) →
  (box.length % cube.sideLength = 0) →
  (box.width % cube.sideLength = 0) →
  (box.depth % cube.sideLength = 0) →
  box.depth = 12 := by
  sorry

end NUMINAMATH_CALUDE_box_depth_calculation_l2990_299089


namespace NUMINAMATH_CALUDE_set_A_properties_l2990_299048

def A : Set ℝ := {x | x^2 - 1 = 0}

theorem set_A_properties : 
  (1 ∈ A) ∧ (∅ ⊆ A) ∧ ({1, -1} ⊆ A) := by sorry

end NUMINAMATH_CALUDE_set_A_properties_l2990_299048


namespace NUMINAMATH_CALUDE_adam_bought_26_books_l2990_299053

/-- Represents Adam's bookcase and book shopping scenario -/
structure Bookcase where
  shelves : Nat
  booksPerShelf : Nat
  initialBooks : Nat
  leftoverBooks : Nat

/-- Calculates the number of books Adam bought -/
def booksBought (b : Bookcase) : Nat :=
  b.shelves * b.booksPerShelf + b.leftoverBooks - b.initialBooks

/-- Theorem stating that Adam bought 26 books -/
theorem adam_bought_26_books (b : Bookcase) 
    (h1 : b.shelves = 4)
    (h2 : b.booksPerShelf = 20)
    (h3 : b.initialBooks = 56)
    (h4 : b.leftoverBooks = 2) : 
  booksBought b = 26 := by
  sorry

end NUMINAMATH_CALUDE_adam_bought_26_books_l2990_299053


namespace NUMINAMATH_CALUDE_factorial_equality_l2990_299026

theorem factorial_equality : 5 * 8 * 2 * 63 = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equality_l2990_299026


namespace NUMINAMATH_CALUDE_ln_inequality_l2990_299014

theorem ln_inequality (x : ℝ) (h : x > 0) : Real.log x ≤ x - 1 := by
  sorry

end NUMINAMATH_CALUDE_ln_inequality_l2990_299014


namespace NUMINAMATH_CALUDE_oil_in_partial_tank_l2990_299070

theorem oil_in_partial_tank (tank_capacity : ℕ) (total_oil : ℕ) : 
  tank_capacity = 32 → total_oil = 728 → 
  total_oil % tank_capacity = 24 := by sorry

end NUMINAMATH_CALUDE_oil_in_partial_tank_l2990_299070


namespace NUMINAMATH_CALUDE_range_of_2a_minus_b_l2990_299001

theorem range_of_2a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 3) (hb : 2 < b ∧ b < 4) :
  -2 < 2*a - b ∧ 2*a - b < 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2a_minus_b_l2990_299001


namespace NUMINAMATH_CALUDE_figure_to_square_l2990_299046

/-- Represents a figure on a graph paper -/
structure Figure where
  area : ℕ

/-- Represents a cut of the figure -/
structure Cut where
  parts : ℕ

/-- Represents the result of reassembling cut parts -/
structure Reassembly where
  isSquare : Bool

/-- A function that cuts a figure into parts -/
def cutFigure (f : Figure) (c : Cut) : Cut :=
  c

/-- A function that reassembles cut parts -/
def reassemble (c : Cut) : Reassembly :=
  { isSquare := true }

/-- Theorem stating that a figure with area 18 can be cut into 3 parts
    and reassembled into a square -/
theorem figure_to_square (f : Figure) (h : f.area = 18) :
  ∃ (c : Cut), c.parts = 3 ∧ (reassemble (cutFigure f c)).isSquare = true := by
  sorry

end NUMINAMATH_CALUDE_figure_to_square_l2990_299046


namespace NUMINAMATH_CALUDE_hamburgers_left_over_example_l2990_299066

/-- Given a restaurant that made some hamburgers and served some of them,
    calculate the number of hamburgers left over. -/
def hamburgers_left_over (made served : ℕ) : ℕ :=
  made - served

/-- Theorem stating that if 9 hamburgers were made and 3 were served,
    then 6 hamburgers were left over. -/
theorem hamburgers_left_over_example :
  hamburgers_left_over 9 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_left_over_example_l2990_299066


namespace NUMINAMATH_CALUDE_star_commutative_star_not_distributive_star_has_identity_star_identity_is_neg_one_l2990_299024

-- Define the binary operation
def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

-- Theorem for commutativity
theorem star_commutative : ∀ x y : ℝ, star x y = star y x := by sorry

-- Theorem for non-distributivity
theorem star_not_distributive : ¬(∀ x y z : ℝ, star x (y + z) = star x y + star x z) := by sorry

-- Theorem for existence of identity element
theorem star_has_identity : ∃ e : ℝ, ∀ x : ℝ, star x e = x := by sorry

-- Theorem that -1 is the identity element
theorem star_identity_is_neg_one : ∀ x : ℝ, star x (-1) = x := by sorry

end NUMINAMATH_CALUDE_star_commutative_star_not_distributive_star_has_identity_star_identity_is_neg_one_l2990_299024


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l2990_299094

/-- Represents the number of books Robert can read in a given time -/
def books_read (reading_speed : ℕ) (available_time : ℕ) (book_type1_pages : ℕ) (book_type2_pages : ℕ) : ℕ :=
  let books_type1 := available_time / (book_type1_pages / reading_speed)
  let books_type2 := available_time / (book_type2_pages / reading_speed)
  books_type1 + books_type2

/-- Theorem stating that Robert can read 5 books in 6 hours given the specified conditions -/
theorem robert_reading_capacity : 
  books_read 120 6 240 360 = 5 := by
  sorry

end NUMINAMATH_CALUDE_robert_reading_capacity_l2990_299094


namespace NUMINAMATH_CALUDE_copy_pages_with_discount_l2990_299093

/-- Calculates the number of pages that can be copied given a certain amount of cents,
    considering a discount where for every 100 pages, an additional 10 pages are free. -/
def pages_copied (cents : ℕ) : ℕ :=
  let base_pages := (cents * 5) / 10
  let free_pages := (base_pages / 100) * 10
  base_pages + free_pages

/-- Proves that 5000 cents allows copying 2750 pages with the given pricing and discount. -/
theorem copy_pages_with_discount :
  pages_copied 5000 = 2750 := by
  sorry

end NUMINAMATH_CALUDE_copy_pages_with_discount_l2990_299093


namespace NUMINAMATH_CALUDE_expected_value_of_game_l2990_299065

-- Define the die
def die := Finset.range 10

-- Define prime numbers on the die
def primes : Finset ℕ := {2, 3, 5, 7}

-- Define composite numbers on the die
def composites : Finset ℕ := {4, 6, 8, 9, 10}

-- Define the winnings function
def winnings (n : ℕ) : ℚ :=
  if n ∈ primes then n
  else if n ∈ composites then 0
  else -5

-- Theorem statement
theorem expected_value_of_game : 
  (die.sum (fun i => winnings i) : ℚ) / 10 = 18 / 100 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_game_l2990_299065


namespace NUMINAMATH_CALUDE_sum_to_base3_l2990_299040

def base10_to_base3 (n : ℕ) : List ℕ :=
  sorry

theorem sum_to_base3 :
  base10_to_base3 (36 + 25 + 2) = [2, 1, 0, 0] :=
sorry

end NUMINAMATH_CALUDE_sum_to_base3_l2990_299040


namespace NUMINAMATH_CALUDE_not_equal_1990_l2990_299060

/-- Count of positive integers ≤ pqn that have a common divisor with pq -/
def f (p q n : ℕ) : ℕ := 
  (n * p) + (n * q) - n

theorem not_equal_1990 (p q n : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) (hn : n > 0) :
  (f p q n : ℚ) / n ≠ 1990 := by
  sorry

end NUMINAMATH_CALUDE_not_equal_1990_l2990_299060


namespace NUMINAMATH_CALUDE_garden_area_l2990_299054

/-- Represents a rectangular garden with specific properties. -/
structure RectangularGarden where
  width : Real
  length : Real
  perimeter : Real
  area : Real
  length_condition : length = 3 * width + 10
  perimeter_condition : perimeter = 2 * (length + width)
  area_condition : area = length * width

/-- Theorem stating the area of a specific rectangular garden. -/
theorem garden_area (g : RectangularGarden) (h : g.perimeter = 400) :
  g.area = 7243.75 := by
  sorry


end NUMINAMATH_CALUDE_garden_area_l2990_299054


namespace NUMINAMATH_CALUDE_cone_radius_from_slant_height_and_surface_area_l2990_299072

theorem cone_radius_from_slant_height_and_surface_area :
  ∀ (slant_height curved_surface_area : ℝ),
    slant_height = 22 →
    curved_surface_area = 483.80526865282815 →
    curved_surface_area = Real.pi * (7 : ℝ) * slant_height :=
by
  sorry

end NUMINAMATH_CALUDE_cone_radius_from_slant_height_and_surface_area_l2990_299072


namespace NUMINAMATH_CALUDE_inverse_of_proposition_l2990_299090

-- Define the original proposition
def original_proposition (x : ℝ) : Prop := x < 0 → x^2 > 0

-- Define the inverse proposition
def inverse_proposition (x : ℝ) : Prop := x^2 > 0 → x < 0

-- Theorem stating that inverse_proposition is the inverse of original_proposition
theorem inverse_of_proposition :
  (∀ x : ℝ, original_proposition x) ↔ (∀ x : ℝ, inverse_proposition x) :=
sorry

end NUMINAMATH_CALUDE_inverse_of_proposition_l2990_299090


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2990_299062

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (|x₁ - 4| = 15 ∧ |x₂ - 4| = 15 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 30 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l2990_299062


namespace NUMINAMATH_CALUDE_probability_of_being_leader_l2990_299018

theorem probability_of_being_leader (total_people : ℕ) (num_groups : ℕ) 
  (h1 : total_people = 12) 
  (h2 : num_groups = 2) 
  (h3 : total_people % num_groups = 0) : 
  (1 : ℚ) / (total_people / num_groups) = 1/6 :=
sorry

end NUMINAMATH_CALUDE_probability_of_being_leader_l2990_299018


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l2990_299082

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_specific : 
  arithmetic_sequence_sum 1 2 17 = 289 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l2990_299082


namespace NUMINAMATH_CALUDE_amount_spent_first_shop_l2990_299030

/-- The amount spent on books from the first shop -/
def amount_first_shop : ℕ := 1500

/-- The number of books bought from the first shop -/
def books_first_shop : ℕ := 55

/-- The number of books bought from the second shop -/
def books_second_shop : ℕ := 60

/-- The amount spent on books from the second shop -/
def amount_second_shop : ℕ := 340

/-- The average price per book -/
def average_price : ℕ := 16

/-- Theorem stating that the amount spent on the first shop is 1500,
    given the conditions of the problem -/
theorem amount_spent_first_shop :
  amount_first_shop = 1500 :=
by
  sorry

end NUMINAMATH_CALUDE_amount_spent_first_shop_l2990_299030


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2990_299071

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + f (x + y)) + f (x * y) = x + f (x + y) + y * f x) →
  (∀ x : ℝ, f x = x ∨ f x = 2 - x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2990_299071


namespace NUMINAMATH_CALUDE_windows_preference_l2990_299028

/-- Given a survey of college students about computer brand preferences,
    this theorem proves the number of students preferring Windows. -/
theorem windows_preference (total : ℕ) (mac : ℕ) (no_pref : ℕ) : 
  total = 210 →
  mac = 60 →
  no_pref = 90 →
  ∃ (windows : ℕ), 
    windows = total - (mac + mac / 3 + no_pref) ∧
    windows = 40 :=
by sorry

end NUMINAMATH_CALUDE_windows_preference_l2990_299028


namespace NUMINAMATH_CALUDE_trivia_game_total_score_luke_total_score_l2990_299069

/-- 
Given a player in a trivia game who:
- Plays a certain number of rounds
- Scores the same number of points each round
- Scores a specific number of points per round

This theorem proves that the total points scored is equal to 
the product of the number of rounds and the points per round.
-/
theorem trivia_game_total_score 
  (rounds : ℕ) 
  (points_per_round : ℕ) : 
  rounds * points_per_round = rounds * points_per_round := by
  sorry

/-- 
This theorem applies the general trivia_game_total_score theorem 
to Luke's specific case, where he played 5 rounds and scored 60 points per round.
-/
theorem luke_total_score : 5 * 60 = 300 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_total_score_luke_total_score_l2990_299069


namespace NUMINAMATH_CALUDE_room_width_calculation_l2990_299041

theorem room_width_calculation (area : ℝ) (length : ℝ) (width : ℝ) :
  area = 10 ∧ length = 5 ∧ area = length * width → width = 2 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l2990_299041


namespace NUMINAMATH_CALUDE_percentage_sum_l2990_299058

theorem percentage_sum : (28 / 100) * 400 + (45 / 100) * 250 = 224.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sum_l2990_299058


namespace NUMINAMATH_CALUDE_symmetry_point_l2990_299023

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricToYAxis (p q : Point2D) : Prop :=
  q.x = -p.x ∧ q.y = p.y

theorem symmetry_point : 
  let M : Point2D := ⟨3, -4⟩
  let N : Point2D := ⟨-3, -4⟩
  symmetricToYAxis M N → N = ⟨-3, -4⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetry_point_l2990_299023


namespace NUMINAMATH_CALUDE_sequence_property_l2990_299000

theorem sequence_property (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n, S n = n * (n - 40)) →
  (∀ n, a n = S n - S (n - 1)) →
  a 19 < 0 ∧ a 21 > 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l2990_299000


namespace NUMINAMATH_CALUDE_largest_three_digit_square_base_seven_l2990_299067

/-- The largest integer whose square has exactly 3 digits when written in base 7 -/
def M : ℕ := 48

/-- Conversion of a natural number to its base 7 representation -/
def to_base_seven (n : ℕ) : ℕ :=
  if n < 7 then n
  else 10 * to_base_seven (n / 7) + n % 7

theorem largest_three_digit_square_base_seven :
  (M^2 ≥ 7^2) ∧ 
  (M^2 < 7^3) ∧ 
  (∀ n : ℕ, n > M → n^2 ≥ 7^3) ∧
  (to_base_seven M = 66) :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_square_base_seven_l2990_299067


namespace NUMINAMATH_CALUDE_work_days_per_week_l2990_299033

/-- Proves that Terry and Jordan work 7 days a week given their daily incomes and weekly income difference -/
theorem work_days_per_week 
  (terry_daily_income : ℕ) 
  (jordan_daily_income : ℕ) 
  (weekly_income_difference : ℕ) 
  (h1 : terry_daily_income = 24)
  (h2 : jordan_daily_income = 30)
  (h3 : weekly_income_difference = 42) :
  ∃ d : ℕ, d = 7 ∧ d * jordan_daily_income - d * terry_daily_income = weekly_income_difference := by
  sorry

end NUMINAMATH_CALUDE_work_days_per_week_l2990_299033


namespace NUMINAMATH_CALUDE_incenter_bisects_orthocenter_circumcenter_angle_l2990_299004

-- Define the types for points and triangles
variable (Point : Type)
variable (Triangle : Type)

-- Define the properties of the triangle
variable (is_acute : Triangle → Prop)
variable (orthocenter : Triangle → Point)
variable (circumcenter : Triangle → Point)
variable (incenter : Triangle → Point)

-- Define the angle bisector property
variable (bisects_angle : Point → Point → Point → Point → Prop)

theorem incenter_bisects_orthocenter_circumcenter_angle 
  (ABC : Triangle) (H O I : Point) :
  is_acute ABC →
  H = orthocenter ABC →
  O = circumcenter ABC →
  I = incenter ABC →
  bisects_angle I A H O :=
sorry

end NUMINAMATH_CALUDE_incenter_bisects_orthocenter_circumcenter_angle_l2990_299004


namespace NUMINAMATH_CALUDE_sqrt_450_simplified_l2990_299099

theorem sqrt_450_simplified : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplified_l2990_299099


namespace NUMINAMATH_CALUDE_comic_book_problem_l2990_299009

theorem comic_book_problem (initial_books : ℕ) : 
  (initial_books / 3 + 15 = 45) → initial_books = 90 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_problem_l2990_299009


namespace NUMINAMATH_CALUDE_smallest_number_is_three_l2990_299081

/-- Represents a systematic sampling of units. -/
structure SystematicSampling where
  total_units : Nat
  selected_units : Nat
  sum_of_selected : Nat

/-- Calculates the smallest number drawn in a systematic sampling. -/
def smallest_number_drawn (s : SystematicSampling) : Nat :=
  (s.sum_of_selected - (s.selected_units - 1) * s.selected_units * (s.total_units / s.selected_units) / 2) / s.selected_units

/-- Theorem stating that for the given systematic sampling, the smallest number drawn is 3. -/
theorem smallest_number_is_three :
  let s : SystematicSampling := ⟨28, 4, 54⟩
  smallest_number_drawn s = 3 := by
  sorry


end NUMINAMATH_CALUDE_smallest_number_is_three_l2990_299081


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2990_299055

theorem trigonometric_equation_solution (x : ℝ) :
  0.5 * (Real.cos (5 * x) + Real.cos (7 * x)) - (Real.cos (2 * x))^2 + (Real.sin (3 * x))^2 = 0 ↔
  (∃ k : ℤ, x = π / 2 * (2 * k + 1)) ∨ (∃ k : ℤ, x = 2 * k * π / 11) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2990_299055


namespace NUMINAMATH_CALUDE_function_extension_l2990_299057

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem function_extension (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_symmetry : ∀ x, f (2 - x) = f x)
  (h_base : ∀ x ∈ Set.Ioo 0 1, f x = Real.log x) :
  (∀ x ∈ Set.Icc (-1) 0, f x = -Real.log (-x)) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Ioo (4 * k) (4 * k + 1), f x = Real.log (x - 4 * k)) :=
sorry

end NUMINAMATH_CALUDE_function_extension_l2990_299057


namespace NUMINAMATH_CALUDE_quadratic_sum_l2990_299007

/-- Given a quadratic function f(x) = -3x^2 + 27x + 135, 
    prove that when written in the form a(x+b)^2 + c,
    the sum of a, b, and c is 197.75 -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = -3*x^2 + 27*x + 135) →
  (∀ x, f x = a*(x+b)^2 + c) →
  a + b + c = 197.75 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2990_299007


namespace NUMINAMATH_CALUDE_haley_cupcakes_l2990_299037

theorem haley_cupcakes (todd_ate : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (h1 : todd_ate = 11)
  (h2 : packages = 3)
  (h3 : cupcakes_per_package = 3) :
  todd_ate + packages * cupcakes_per_package = 20 := by
  sorry

end NUMINAMATH_CALUDE_haley_cupcakes_l2990_299037


namespace NUMINAMATH_CALUDE_combine_like_terms_l2990_299063

theorem combine_like_terms (x y : ℝ) :
  -x^2 * y + 3/4 * x^2 * y = -(1/4) * x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_l2990_299063


namespace NUMINAMATH_CALUDE_original_price_calculation_l2990_299083

theorem original_price_calculation (reduced_price : ℝ) (reduction_percentage : ℝ) 
  (h1 : reduced_price = 6)
  (h2 : reduction_percentage = 0.25)
  (h3 : reduced_price = reduction_percentage * original_price) :
  original_price = 24 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2990_299083
