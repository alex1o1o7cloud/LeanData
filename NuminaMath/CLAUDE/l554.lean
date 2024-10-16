import Mathlib

namespace NUMINAMATH_CALUDE_tribe_leadership_count_l554_55471

def tribe_size : ℕ := 15

def leadership_arrangements : ℕ :=
  tribe_size *
  (tribe_size - 1) *
  (tribe_size - 2) *
  (tribe_size - 3) *
  (tribe_size - 4) *
  (Nat.choose (tribe_size - 5) 2) *
  (Nat.choose (tribe_size - 7) 2)

theorem tribe_leadership_count :
  leadership_arrangements = 216216000 := by
  sorry

end NUMINAMATH_CALUDE_tribe_leadership_count_l554_55471


namespace NUMINAMATH_CALUDE_investment_profit_distribution_l554_55486

/-- Represents the investment and profit distribution problem -/
theorem investment_profit_distribution 
  (total_capital : ℕ) 
  (total_profit : ℕ) 
  (a_invest_diff : ℕ) 
  (b_invest_diff : ℕ) 
  (d_invest_diff : ℕ) 
  (a_duration b_duration c_duration d_duration : ℕ) 
  (h1 : total_capital = 100000)
  (h2 : total_profit = 50000)
  (h3 : a_invest_diff = 10000)
  (h4 : b_invest_diff = 5000)
  (h5 : d_invest_diff = 8000)
  (h6 : a_duration = 12)
  (h7 : b_duration = 10)
  (h8 : c_duration = 8)
  (h9 : d_duration = 6) :
  ∃ (c_invest : ℕ),
    let b_invest := c_invest + b_invest_diff
    let a_invest := b_invest + a_invest_diff
    let d_invest := a_invest + d_invest_diff
    c_invest + b_invest + a_invest + d_invest = total_capital ∧
    (b_invest * b_duration : ℚ) / ((c_invest * c_duration + b_invest * b_duration + a_invest * a_duration + d_invest * d_duration) : ℚ) * total_profit = 10925 := by
  sorry

end NUMINAMATH_CALUDE_investment_profit_distribution_l554_55486


namespace NUMINAMATH_CALUDE_orange_profit_problem_l554_55431

/-- Represents the fruit vendor's orange selling problem -/
theorem orange_profit_problem 
  (buy_quantity : ℕ) 
  (buy_price : ℚ) 
  (sell_quantity : ℕ) 
  (sell_price : ℚ) 
  (target_profit : ℚ) :
  buy_quantity = 8 →
  buy_price = 15 →
  sell_quantity = 6 →
  sell_price = 18 →
  target_profit = 150 →
  ∃ (n : ℕ), 
    n * (sell_price / sell_quantity - buy_price / buy_quantity) ≥ target_profit ∧
    ∀ (m : ℕ), m * (sell_price / sell_quantity - buy_price / buy_quantity) ≥ target_profit → m ≥ n ∧
    n = 134 :=
sorry

end NUMINAMATH_CALUDE_orange_profit_problem_l554_55431


namespace NUMINAMATH_CALUDE_divisor_problem_l554_55468

theorem divisor_problem (x : ℝ) (d : ℝ) : 
  x = 22.142857142857142 →
  (7 * (x + 5)) / d - 5 = 33 →
  d = 5 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l554_55468


namespace NUMINAMATH_CALUDE_first_candidate_percentage_l554_55410

theorem first_candidate_percentage (total_votes : ℕ) (invalid_percent : ℚ) (second_candidate_votes : ℕ) :
  total_votes = 5500 →
  invalid_percent = 20/100 →
  second_candidate_votes = 1980 →
  let valid_votes := total_votes * (1 - invalid_percent)
  let first_candidate_votes := valid_votes - second_candidate_votes
  (first_candidate_votes : ℚ) / valid_votes * 100 = 55 := by
  sorry

end NUMINAMATH_CALUDE_first_candidate_percentage_l554_55410


namespace NUMINAMATH_CALUDE_f_of_g_10_l554_55442

-- Define the functions g and f
def g (x : ℝ) : ℝ := 2 * x + 6
def f (x : ℝ) : ℝ := 4 * x - 8

-- State the theorem
theorem f_of_g_10 : f (g 10) = 96 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_10_l554_55442


namespace NUMINAMATH_CALUDE_complex_angle_90_degrees_l554_55455

theorem complex_angle_90_degrees (z₁ z₂ : ℂ) (hz₁ : z₁ ≠ 0) (hz₂ : z₂ ≠ 0) 
  (h : Complex.abs (z₁ + z₂) = Complex.abs (z₁ - z₂)) : 
  Real.cos (Complex.arg z₁ - Complex.arg z₂) = 0 :=
sorry

end NUMINAMATH_CALUDE_complex_angle_90_degrees_l554_55455


namespace NUMINAMATH_CALUDE_cylinder_max_volume_ratio_l554_55454

/-- Given a rectangle with perimeter 12 that forms a cylinder, prove that the ratio of the base circumference to height is 2:1 when volume is maximized -/
theorem cylinder_max_volume_ratio (l w : ℝ) : 
  l > 0 → w > 0 → 
  2 * l + 2 * w = 12 → 
  let r := l / (2 * Real.pi)
  let h := w
  let V := Real.pi * r^2 * h
  (∀ l' w', l' > 0 → w' > 0 → 2 * l' + 2 * w' = 12 → 
    let r' := l' / (2 * Real.pi)
    let h' := w'
    Real.pi * r'^2 * h' ≤ V) →
  l / w = 2 := by
sorry

end NUMINAMATH_CALUDE_cylinder_max_volume_ratio_l554_55454


namespace NUMINAMATH_CALUDE_lcm_of_18_28_45_65_l554_55407

theorem lcm_of_18_28_45_65 : Nat.lcm 18 (Nat.lcm 28 (Nat.lcm 45 65)) = 16380 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_18_28_45_65_l554_55407


namespace NUMINAMATH_CALUDE_unique_solution_system_l554_55482

theorem unique_solution_system (x y : ℝ) :
  (x + y = (5 - x) + (5 - y)) ∧ (x - y = (x - 1) + (y - 1)) →
  x = 4 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l554_55482


namespace NUMINAMATH_CALUDE_taylor_score_ratio_l554_55477

/-- Given the conditions for Taylor's score mixture, prove the ratio of white to black scores -/
theorem taylor_score_ratio :
  ∀ (white black : ℕ),
  white + black = 78 →
  2 * (black - white) = 3 * 4 →
  (white : ℚ) / black = 6 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_taylor_score_ratio_l554_55477


namespace NUMINAMATH_CALUDE_notebook_distribution_l554_55462

theorem notebook_distribution (C : ℕ) (H : ℕ) : 
  (C * (C / 8) = 512) →
  (H / 8 = 16) →
  (H : ℚ) / C = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_l554_55462


namespace NUMINAMATH_CALUDE_water_pumped_30_minutes_l554_55475

/-- Represents a water pumping system -/
structure WaterPump where
  gallons_per_hour : ℝ

/-- Calculates the amount of water pumped in a given time -/
def water_pumped (pump : WaterPump) (hours : ℝ) : ℝ :=
  pump.gallons_per_hour * hours

theorem water_pumped_30_minutes (pump : WaterPump) 
  (h : pump.gallons_per_hour = 500) : 
  water_pumped pump (30 / 60) = 250 := by
  sorry

#check water_pumped_30_minutes

end NUMINAMATH_CALUDE_water_pumped_30_minutes_l554_55475


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l554_55430

theorem min_value_sum_reciprocals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 5) : 
  (1/x + 4/y + 9/z) ≥ 36/5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l554_55430


namespace NUMINAMATH_CALUDE_unique_satisfying_polynomial_l554_55443

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The roots of a quadratic polynomial -/
def roots (p : QuadraticPolynomial) : Set ℝ :=
  {r : ℝ | p.a * r^2 + p.b * r + p.c = 0}

/-- The coefficients of a quadratic polynomial -/
def coefficients (p : QuadraticPolynomial) : Set ℝ :=
  {p.a, p.b, p.c}

/-- Predicate for a polynomial satisfying the problem conditions -/
def satisfies_conditions (p : QuadraticPolynomial) : Prop :=
  roots p = coefficients p ∧
  (p.a < 0 ∨ p.b < 0 ∨ p.c < 0)

/-- The main theorem stating that exactly one quadratic polynomial satisfies the conditions -/
theorem unique_satisfying_polynomial :
  ∃! p : QuadraticPolynomial, satisfies_conditions p :=
sorry

end NUMINAMATH_CALUDE_unique_satisfying_polynomial_l554_55443


namespace NUMINAMATH_CALUDE_average_gas_mileage_calculation_l554_55427

theorem average_gas_mileage_calculation (distance_to_university : ℝ) (sedan_efficiency : ℝ)
  (weekend_trip_distance : ℝ) (truck_efficiency : ℝ)
  (h1 : distance_to_university = 150)
  (h2 : sedan_efficiency = 25)
  (h3 : weekend_trip_distance = 200)
  (h4 : truck_efficiency = 15) :
  let total_distance := distance_to_university + weekend_trip_distance
  let sedan_gas_used := distance_to_university / sedan_efficiency
  let truck_gas_used := weekend_trip_distance / truck_efficiency
  let total_gas_used := sedan_gas_used + truck_gas_used
  total_distance / total_gas_used = 1050 / 58 := by sorry

end NUMINAMATH_CALUDE_average_gas_mileage_calculation_l554_55427


namespace NUMINAMATH_CALUDE_jason_current_is_sum_jason_has_63_dollars_l554_55408

/-- Represents the money situation for Fred and Jason --/
structure MoneySituation where
  fred_initial : ℕ
  jason_initial : ℕ
  fred_current : ℕ
  jason_earned : ℕ

/-- Calculates Jason's current amount of money --/
def jason_current (s : MoneySituation) : ℕ := s.jason_initial + s.jason_earned

/-- Theorem stating that Jason's current amount is the sum of his initial and earned amounts --/
theorem jason_current_is_sum (s : MoneySituation) :
  jason_current s = s.jason_initial + s.jason_earned := by sorry

/-- The specific money situation from the problem --/
def problem_situation : MoneySituation :=
  { fred_initial := 49
  , jason_initial := 3
  , fred_current := 112
  , jason_earned := 60 }

/-- Theorem proving that Jason now has 63 dollars --/
theorem jason_has_63_dollars :
  jason_current problem_situation = 63 := by sorry

end NUMINAMATH_CALUDE_jason_current_is_sum_jason_has_63_dollars_l554_55408


namespace NUMINAMATH_CALUDE_even_function_implies_a_squared_one_l554_55446

def f (x a : ℝ) : ℝ := x^2 + (a^2 - 1)*x + 6

theorem even_function_implies_a_squared_one (a : ℝ) :
  (∀ x, f x a = f (-x) a) → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_squared_one_l554_55446


namespace NUMINAMATH_CALUDE_birdhouse_earnings_l554_55458

/-- The price of a large birdhouse in dollars -/
def large_price : ℕ := 22

/-- The price of a medium birdhouse in dollars -/
def medium_price : ℕ := 16

/-- The price of a small birdhouse in dollars -/
def small_price : ℕ := 7

/-- The number of large birdhouses sold -/
def large_sold : ℕ := 2

/-- The number of medium birdhouses sold -/
def medium_sold : ℕ := 2

/-- The number of small birdhouses sold -/
def small_sold : ℕ := 3

/-- The total money earned from selling birdhouses -/
def total_earned : ℕ := large_price * large_sold + medium_price * medium_sold + small_price * small_sold

theorem birdhouse_earnings : total_earned = 97 := by
  sorry

end NUMINAMATH_CALUDE_birdhouse_earnings_l554_55458


namespace NUMINAMATH_CALUDE_max_rectangles_intersection_l554_55425

/-- A rectangle in a plane --/
structure Rectangle where
  -- We don't need to define the specifics of a rectangle for this problem

/-- The number of intersection points between two rectangles --/
def intersection_points (r1 r2 : Rectangle) : ℕ := sorry

/-- The maximum number of intersection points between any two rectangles --/
def max_intersection_points : ℕ := sorry

/-- Theorem: The maximum number of intersection points between any two rectangles is 8 --/
theorem max_rectangles_intersection :
  max_intersection_points = 8 := by sorry

end NUMINAMATH_CALUDE_max_rectangles_intersection_l554_55425


namespace NUMINAMATH_CALUDE_meeting_point_theorem_l554_55487

/-- The distance between two points A and B, where two people walk towards each other
    under specific conditions. -/
def distance_AB : ℝ := 2800

theorem meeting_point_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let S := distance_AB
  let meeting_point := 1200
  let B_double_speed := 2 * y
  S / 2 / x + (S / 2 - meeting_point) / x = 
    S / 2 / y + (meeting_point - S * y / (2 * x)) / B_double_speed ∧
  S - meeting_point = S / 2 →
  S = 2800 := by sorry

end NUMINAMATH_CALUDE_meeting_point_theorem_l554_55487


namespace NUMINAMATH_CALUDE_exists_x_iff_b_gt_min_sum_l554_55417

/-- The minimum value of the sum of absolute differences -/
def min_sum : ℝ := 4

/-- The function representing the sum of absolute differences -/
def f (x : ℝ) : ℝ := |x - 5| + |x - 3| + |x - 2|

/-- Theorem stating the condition for the existence of x satisfying the inequality -/
theorem exists_x_iff_b_gt_min_sum (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, f x < b) ↔ b > min_sum :=
sorry

end NUMINAMATH_CALUDE_exists_x_iff_b_gt_min_sum_l554_55417


namespace NUMINAMATH_CALUDE_be_length_l554_55461

structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

def is_right_angle (p q r : ℝ × ℝ) : Prop := sorry

def on_line (p q r : ℝ × ℝ) : Prop := sorry

def perpendicular (l1 l2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem be_length 
  (ABCD : Quadrilateral)
  (E F : ℝ × ℝ)
  (h1 : is_right_angle ABCD.A ABCD.B ABCD.C)
  (h2 : is_right_angle ABCD.B ABCD.C ABCD.D)
  (h3 : on_line ABCD.A E ABCD.C)
  (h4 : on_line ABCD.A F ABCD.C)
  (h5 : perpendicular (ABCD.D, F) (ABCD.A, ABCD.C))
  (h6 : perpendicular (ABCD.B, E) (ABCD.A, ABCD.C))
  (h7 : distance ABCD.A F = 4)
  (h8 : distance ABCD.D F = 6)
  (h9 : distance ABCD.C F = 8)
  : distance ABCD.B E = 16/3 := sorry

end NUMINAMATH_CALUDE_be_length_l554_55461


namespace NUMINAMATH_CALUDE_simplify_power_expression_l554_55497

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^5 = 243 * x^20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l554_55497


namespace NUMINAMATH_CALUDE_incorrect_factorization_l554_55459

theorem incorrect_factorization (x : ℝ) : x^2 - 7*x + 12 ≠ x*(x - 7) + 12 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_factorization_l554_55459


namespace NUMINAMATH_CALUDE_cubic_root_equality_l554_55463

theorem cubic_root_equality (p q r : ℝ) : 
  (∀ x : ℝ, x^3 - 3*p*x^2 + 3*q^2*x - r^3 = 0 ↔ (x = p ∨ x = q ∨ x = r)) →
  p = q ∧ q = r :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_equality_l554_55463


namespace NUMINAMATH_CALUDE_functional_equation_solution_l554_55428

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y) →
  (∀ x : ℝ, f x = 0 ∨ f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l554_55428


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l554_55413

theorem complex_subtraction_simplification :
  (-5 - 3*I : ℂ) - (2 + 6*I) = -7 - 9*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l554_55413


namespace NUMINAMATH_CALUDE_fermats_little_theorem_l554_55492

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (hp : Nat.Prime p) :
  a^p ≡ a [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_fermats_little_theorem_l554_55492


namespace NUMINAMATH_CALUDE_coefficient_of_monomial_l554_55438

theorem coefficient_of_monomial (a b : ℝ) :
  let expression := (4 * Real.pi * a^2 * b) / 5
  let coefficient := -(4 / 5) * Real.pi
  coefficient = expression / (a^2 * b) := by sorry

end NUMINAMATH_CALUDE_coefficient_of_monomial_l554_55438


namespace NUMINAMATH_CALUDE_quadratic_form_nonnegative_l554_55415

theorem quadratic_form_nonnegative
  (a b c x y z : ℝ)
  (sum_xyz : x + y + z = 0)
  (sum_abc_nonneg : a + b + c ≥ 0)
  (sum_products_nonneg : a * b + b * c + c * a ≥ 0) :
  a * x^2 + b * y^2 + c * z^2 ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_nonnegative_l554_55415


namespace NUMINAMATH_CALUDE_composite_rectangle_area_l554_55405

theorem composite_rectangle_area : 
  let rect1_area := 6 * 9
  let rect2_area := 4 * 6
  let rect3_area := 5 * 2
  rect1_area + rect2_area + rect3_area = 88 := by
  sorry

end NUMINAMATH_CALUDE_composite_rectangle_area_l554_55405


namespace NUMINAMATH_CALUDE_penguin_fish_distribution_l554_55466

theorem penguin_fish_distribution (days : ℕ) (fish_eaten_by_first_chick : ℕ) : 
  fish_eaten_by_first_chick = 44 →
  (days * 12 - fish_eaten_by_first_chick = 52) := by sorry

end NUMINAMATH_CALUDE_penguin_fish_distribution_l554_55466


namespace NUMINAMATH_CALUDE_cube_root_of_x_plus_3y_is_3_l554_55411

theorem cube_root_of_x_plus_3y_is_3 (x y : ℝ) 
  (h : y = Real.sqrt (x - 3) + Real.sqrt (3 - x) + 8) : 
  (x + 3 * y) ^ (1/3 : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_x_plus_3y_is_3_l554_55411


namespace NUMINAMATH_CALUDE_number_of_students_l554_55406

/-- Proves the number of students in a class given average ages and teacher's age -/
theorem number_of_students (avg_age : ℝ) (teacher_age : ℝ) (new_avg_age : ℝ) :
  avg_age = 22 →
  teacher_age = 46 →
  new_avg_age = 23 →
  (avg_age * n + teacher_age) / (n + 1) = new_avg_age →
  n = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l554_55406


namespace NUMINAMATH_CALUDE_palindrome_pairs_exist_l554_55420

/-- A function that checks if a positive integer is a palindrome -/
def is_palindrome (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), n = digits.foldl (λ acc d => 10 * acc + d) 0 ∧ digits = digits.reverse

/-- A function that generates a palindrome given three digits -/
def generate_palindrome (a b k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are at least 2005 palindrome pairs -/
theorem palindrome_pairs_exist : 
  ∃ (pairs : List (ℕ × ℕ)), pairs.length ≥ 2005 ∧ 
    ∀ (pair : ℕ × ℕ), pair ∈ pairs → 
      is_palindrome pair.1 ∧ is_palindrome pair.2 ∧ pair.2 = pair.1 + 110 :=
sorry

end NUMINAMATH_CALUDE_palindrome_pairs_exist_l554_55420


namespace NUMINAMATH_CALUDE_friendship_divisibility_criterion_l554_55448

/-- Represents a friendship relation between students -/
def FriendshipRelation (n : ℕ) := Fin n → Fin n → Prop

/-- The friendship relation is symmetric -/
def symmetric {n : ℕ} (r : FriendshipRelation n) :=
  ∀ i j, r i j ↔ r j i

/-- The friendship relation is irreflexive -/
def irreflexive {n : ℕ} (r : FriendshipRelation n) :=
  ∀ i, ¬(r i i)

/-- Theorem: For any finite set of students with a friendship relation,
    there exists a positive integer N and an assignment of integers to students
    such that two students are friends if and only if N divides the product of their assigned integers -/
theorem friendship_divisibility_criterion
  {n : ℕ} (r : FriendshipRelation n) (h_sym : symmetric r) (h_irr : irreflexive r) :
  ∃ (N : ℕ) (N_pos : 0 < N) (a : Fin n → ℤ),
    ∀ i j, r i j ↔ (N : ℤ) ∣ (a i * a j) :=
sorry

end NUMINAMATH_CALUDE_friendship_divisibility_criterion_l554_55448


namespace NUMINAMATH_CALUDE_shortest_path_length_l554_55483

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ

/-- Represents a circle -/
structure Circle where
  radius : ℝ

/-- Represents a path on a triangle and circle -/
def ShortestPath (t : EquilateralTriangle) (c : Circle) : ℝ := sorry

/-- The theorem stating the length of the shortest path -/
theorem shortest_path_length 
  (t : EquilateralTriangle) 
  (c : Circle) 
  (h1 : t.sideLength = 2) 
  (h2 : c.radius = 1/2) : 
  ShortestPath t c = Real.sqrt (28/3) - 1 := by sorry

end NUMINAMATH_CALUDE_shortest_path_length_l554_55483


namespace NUMINAMATH_CALUDE_smallest_A_for_divisibility_l554_55412

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def six_digit_number (A : ℕ) : ℕ := 4 * 100000 + A * 10000 + 88851

theorem smallest_A_for_divisibility :
  ∀ A : ℕ, A ≥ 1 →
    (is_divisible_by_3 (six_digit_number A) → A ≥ 1) ∧
    is_divisible_by_3 (six_digit_number 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_A_for_divisibility_l554_55412


namespace NUMINAMATH_CALUDE_sum_of_digits_1_to_1000_l554_55453

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sum_of_digits_up_to (n : ℕ) : ℕ := 
  (Finset.range n).sum (λ i => sum_of_digits (i + 1))

/-- Theorem: The sum of digits of all numbers from 1 to 1000 is 14446 -/
theorem sum_of_digits_1_to_1000 : sum_of_digits_up_to 1000 = 14446 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_1_to_1000_l554_55453


namespace NUMINAMATH_CALUDE_inverse_iff_horizontal_line_test_l554_55401

-- Define a type for our functions
def Function := ℝ → ℝ

-- Define what it means for a function to have an inverse
def has_inverse (f : Function) : Prop :=
  ∃ g : Function, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- Define the horizontal line test
def passes_horizontal_line_test (f : Function) : Prop :=
  ∀ y : ℝ, ∀ x₁ x₂ : ℝ, f x₁ = y ∧ f x₂ = y → x₁ = x₂

-- State the theorem
theorem inverse_iff_horizontal_line_test (f : Function) :
  has_inverse f ↔ passes_horizontal_line_test f :=
sorry

end NUMINAMATH_CALUDE_inverse_iff_horizontal_line_test_l554_55401


namespace NUMINAMATH_CALUDE_linear_function_property_l554_55451

/-- Given a linear function f(x) = ax + b that satisfies f(bx + a) = x for all real x,
    prove that a + b = -2 -/
theorem linear_function_property (a b : ℝ) 
  (h1 : ∀ x : ℝ, ∃ f : ℝ → ℝ, f x = a * x + b) 
  (h2 : ∀ x : ℝ, ∃ f : ℝ → ℝ, f (b * x + a) = x) : 
  a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l554_55451


namespace NUMINAMATH_CALUDE_max_value_of_a_l554_55494

theorem max_value_of_a : 
  (∃ a : ℝ, ∀ x : ℝ, x < a → x^2 - 2*x - 3 > 0) ∧ 
  (∀ a : ℝ, ∃ x : ℝ, x^2 - 2*x - 3 > 0 ∧ x ≥ a) →
  (∀ b : ℝ, (∀ x : ℝ, x < b → x^2 - 2*x - 3 > 0) → b ≤ -1) ∧
  (∀ x : ℝ, x < -1 → x^2 - 2*x - 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l554_55494


namespace NUMINAMATH_CALUDE_triangle_with_arithmetic_sides_l554_55476

/-- 
A triangle with sides forming an arithmetic sequence with common difference 1 and area 6 
has sides 3, 4, and 5, and one of its angles is a right angle.
-/
theorem triangle_with_arithmetic_sides (a b c : ℝ) (α β γ : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  b = a + 1 ∧ c = b + 1 →  -- sides form arithmetic sequence with difference 1
  (a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c) = 36 →  -- area is 6 (Heron's formula)
  α + β + γ = π →  -- sum of angles is π
  a * a = b * b + c * c - 2 * b * c * Real.cos α →  -- law of cosines for side a
  b * b = a * a + c * c - 2 * a * c * Real.cos β →  -- law of cosines for side b
  c * c = a * a + b * b - 2 * a * b * Real.cos γ →  -- law of cosines for side c
  (a = 3 ∧ b = 4 ∧ c = 5) ∧ γ = π / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_with_arithmetic_sides_l554_55476


namespace NUMINAMATH_CALUDE_lindas_calculation_l554_55440

theorem lindas_calculation (x y z : ℝ) 
  (h1 : x - (y + z) = 5) 
  (h2 : x - y + z = -1) : 
  x - y = 2 := by
  sorry

end NUMINAMATH_CALUDE_lindas_calculation_l554_55440


namespace NUMINAMATH_CALUDE_expression_evaluation_l554_55490

theorem expression_evaluation : 
  let x : ℝ := -2
  3 * (-2 * x^2 + 5 + 4 * x) - (5 * x - 4 - 7 * x^2) = 9 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l554_55490


namespace NUMINAMATH_CALUDE_alternative_basis_l554_55467

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (e₁ e₂ : V)

/-- Given that e₁ and e₂ form a basis for a plane, prove that e₁ + e₂ and e₁ - e₂ also form a basis for the same plane. -/
theorem alternative_basis (h : LinearIndependent ℝ ![e₁, e₂]) :
  LinearIndependent ℝ ![e₁ + e₂, e₁ - e₂] ∧ 
  Submodule.span ℝ {e₁, e₂} = Submodule.span ℝ {e₁ + e₂, e₁ - e₂} := by
  sorry

end NUMINAMATH_CALUDE_alternative_basis_l554_55467


namespace NUMINAMATH_CALUDE_log_equation_proof_l554_55493

theorem log_equation_proof (y : ℝ) (m : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → (Real.log 125 / Real.log 2 = m * y) → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_proof_l554_55493


namespace NUMINAMATH_CALUDE_magnified_diameter_calculation_l554_55447

/-- Given a circular piece of tissue with an actual diameter and a magnification factor,
    calculate the diameter of the magnified image. -/
theorem magnified_diameter_calculation
  (actual_diameter : ℝ)
  (magnification_factor : ℝ)
  (h1 : actual_diameter = 0.0002)
  (h2 : magnification_factor = 1000) :
  actual_diameter * magnification_factor = 0.2 := by
sorry

end NUMINAMATH_CALUDE_magnified_diameter_calculation_l554_55447


namespace NUMINAMATH_CALUDE_min_cost_29_disks_l554_55414

/-- Represents the cost of a package of disks -/
structure Package where
  quantity : Nat
  price : Nat

/-- Calculates the minimum cost to buy at least n disks given a list of packages -/
def minCost (packages : List Package) (n : Nat) : Nat :=
  sorry

/-- The available packages -/
def availablePackages : List Package :=
  [{ quantity := 1, price := 20 },
   { quantity := 10, price := 111 },
   { quantity := 25, price := 265 }]

theorem min_cost_29_disks :
  minCost availablePackages 29 = 333 :=
sorry

end NUMINAMATH_CALUDE_min_cost_29_disks_l554_55414


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l554_55479

theorem arithmetic_evaluation : 6 / 3 - 2 - 8 + 2 * 8 = 8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l554_55479


namespace NUMINAMATH_CALUDE_boat_current_rate_l554_55418

/-- Proves that the rate of the current is 4 km/hr given the conditions of the boat problem -/
theorem boat_current_rate (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 12 →
  downstream_distance = 4.8 →
  downstream_time = 18 / 60 →
  ∃ current_rate : ℝ,
    current_rate = 4 ∧
    downstream_distance = (boat_speed + current_rate) * downstream_time :=
by
  sorry


end NUMINAMATH_CALUDE_boat_current_rate_l554_55418


namespace NUMINAMATH_CALUDE_amc8_participants_l554_55437

/-- The number of mathematics students at Euclid Middle School taking the AMC 8 contest -/
def total_students (germain newton young gauss : ℕ) : ℕ :=
  germain + newton + young + gauss

/-- Theorem stating that the total number of students taking the AMC 8 contest is 38 -/
theorem amc8_participants : total_students 12 10 9 7 = 38 := by
  sorry

end NUMINAMATH_CALUDE_amc8_participants_l554_55437


namespace NUMINAMATH_CALUDE_equation_solution_l554_55480

theorem equation_solution : ∃ x : ℚ, (27 / 4 : ℚ) * x - 18 = 3 * x + 27 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l554_55480


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l554_55404

def repeating_decimal_6 : ℚ := 2/3
def repeating_decimal_3 : ℚ := 1/3

theorem sum_of_repeating_decimals : 
  repeating_decimal_6 + repeating_decimal_3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l554_55404


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l554_55464

/-- A function f is monotonically increasing on (a, +∞) if for all x, y > a, x < y implies f(x) < f(y) -/
def MonoIncreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, x > a → y > a → x < y → f x < f y

/-- The quadratic function f(x) = x^2 + mx - 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 2

theorem quadratic_monotonicity (m : ℝ) :
  MonoIncreasing (f m) 2 → m ≥ -4 := by sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l554_55464


namespace NUMINAMATH_CALUDE_square_units_digit_l554_55465

theorem square_units_digit (n : ℤ) : 
  (n^2 / 10) % 10 = 7 → n^2 % 10 = 6 := by sorry

end NUMINAMATH_CALUDE_square_units_digit_l554_55465


namespace NUMINAMATH_CALUDE_average_weight_increase_l554_55429

theorem average_weight_increase (original_count : ℕ) (original_weight replaced_weight new_weight : ℝ) :
  original_count = 9 →
  replaced_weight = 65 →
  new_weight = 87.5 →
  (new_weight - replaced_weight) / original_count = 2.5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l554_55429


namespace NUMINAMATH_CALUDE_equivalent_operations_l554_55423

theorem equivalent_operations (x : ℝ) : 
  (x * (2/5)) / (4/7) = x * (7/10) := by
sorry

end NUMINAMATH_CALUDE_equivalent_operations_l554_55423


namespace NUMINAMATH_CALUDE_abs_ratio_eq_sqrt_seven_halves_l554_55473

theorem abs_ratio_eq_sqrt_seven_halves (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 5*a*b) :
  |((a + b) / (a - b))| = Real.sqrt (7/2) := by
  sorry

end NUMINAMATH_CALUDE_abs_ratio_eq_sqrt_seven_halves_l554_55473


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l554_55409

theorem absolute_value_equation_solutions :
  ∀ x : ℝ, (3 * x + 9 = |(-20 + 4 * x)|) ↔ (x = 29 ∨ x = 11/7) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l554_55409


namespace NUMINAMATH_CALUDE_abs_sum_equals_sum_abs_necessary_not_sufficient_l554_55457

theorem abs_sum_equals_sum_abs_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a * b > 0 → |a + b| = |a| + |b|) ∧
  (∃ a b : ℝ, |a + b| = |a| + |b| ∧ a * b ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_abs_sum_equals_sum_abs_necessary_not_sufficient_l554_55457


namespace NUMINAMATH_CALUDE_triangle345_circle1_common_points_l554_55426

/-- Represents the number of common points between a triangle and a circle -/
inductive CommonPoints
  | Zero
  | One
  | Two
  | Four

/-- A triangle with side lengths 3, 4, and 5 -/
structure Triangle345 where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side1_eq : side1 = 3
  side2_eq : side2 = 4
  side3_eq : side3 = 5

/-- A circle with radius 1 -/
structure Circle1 where
  radius : ℝ
  radius_eq : radius = 1

/-- The theorem stating the possible numbers of common points -/
theorem triangle345_circle1_common_points (t : Triangle345) (c : Circle1) :
  {cp : CommonPoints | cp = CommonPoints.Zero ∨ cp = CommonPoints.One ∨ 
                       cp = CommonPoints.Two ∨ cp = CommonPoints.Four} = 
  {CommonPoints.Zero, CommonPoints.One, CommonPoints.Two, CommonPoints.Four} :=
sorry

end NUMINAMATH_CALUDE_triangle345_circle1_common_points_l554_55426


namespace NUMINAMATH_CALUDE_inequality_of_product_one_l554_55491

theorem inequality_of_product_one (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_product_one_l554_55491


namespace NUMINAMATH_CALUDE_negation_of_statement_l554_55495

theorem negation_of_statement :
  (¬ (∀ x : ℝ, (x = 0 ∨ x = 1) → x^2 - x = 0)) ↔
  (∀ x : ℝ, (x ≠ 0 ∧ x ≠ 1) → x^2 - x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_statement_l554_55495


namespace NUMINAMATH_CALUDE_millet_exceeds_half_on_fourth_day_l554_55421

/-- Represents the fraction of millet seeds in the feeder on a given day -/
def milletFraction (day : ℕ) : ℚ :=
  match day with
  | 0 => 3/10
  | n + 1 => (1/2 * milletFraction n + 3/10)

/-- Theorem stating that on the 4th day, the fraction of millet seeds exceeds 1/2 for the first time -/
theorem millet_exceeds_half_on_fourth_day :
  (milletFraction 4 > 1/2) ∧
  (∀ d : ℕ, d < 4 → milletFraction d ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_millet_exceeds_half_on_fourth_day_l554_55421


namespace NUMINAMATH_CALUDE_grass_cutting_expenditure_l554_55416

/-- Represents the four seasons --/
inductive Season
  | Spring
  | Summer
  | Fall
  | Winter

/-- Growth rate of grass per month for each season (in inches) --/
def growth_rate (s : Season) : Real :=
  match s with
  | Season.Spring => 0.6
  | Season.Summer => 0.5
  | Season.Fall => 0.4
  | Season.Winter => 0.2

/-- Number of months in each season --/
def months_per_season : Nat := 3

/-- Initial height of grass after cutting (in inches) --/
def initial_height : Real := 2

/-- Height at which grass needs to be cut (in inches) --/
def cut_height : Real := 4

/-- Initial cost to cut grass --/
def initial_cost : Nat := 100

/-- Cost increase per cut --/
def cost_increase : Nat := 5

/-- Calculate the total growth of grass in a season --/
def season_growth (s : Season) : Real :=
  growth_rate s * months_per_season

/-- Calculate the number of cuts needed in a year --/
def cuts_per_year : Nat := 2

/-- Calculate the total expenditure for cutting grass in a year --/
def total_expenditure : Nat :=
  initial_cost + (initial_cost + cost_increase)

theorem grass_cutting_expenditure :
  total_expenditure = 205 := by
  sorry

end NUMINAMATH_CALUDE_grass_cutting_expenditure_l554_55416


namespace NUMINAMATH_CALUDE_max_shadow_distance_l554_55488

/-- 
Given a projectile motion with:
- v: initial velocity
- t: time of flight
- y: vertical displacement
- g: gravitational acceleration
- a: constant horizontal acceleration due to air resistance

The maximum horizontal distance L of the projectile's shadow is 0.75 m.
-/
theorem max_shadow_distance 
  (v : ℝ) 
  (t : ℝ) 
  (y : ℝ) 
  (g : ℝ) 
  (a : ℝ) 
  (h1 : v = 5)
  (h2 : t = 1)
  (h3 : y = -1)
  (h4 : g = 10)
  (h5 : y = v * Real.sin α * t - (g * t^2) / 2)
  (h6 : 0 = v * Real.cos α * t - (a * t^2) / 2)
  (h7 : α = Real.arcsin (4/5))
  : ∃ L : ℝ, L = 0.75 ∧ L = (v^2 * (Real.cos α)^2) / (2 * a) := by
  sorry

end NUMINAMATH_CALUDE_max_shadow_distance_l554_55488


namespace NUMINAMATH_CALUDE_trapezoid_bc_length_l554_55496

/-- Trapezoid properties -/
structure Trapezoid :=
  (area : ℝ)
  (altitude : ℝ)
  (ab : ℝ)
  (cd : ℝ)

/-- Theorem: For a trapezoid with given properties, BC = 10 cm -/
theorem trapezoid_bc_length (t : Trapezoid) 
  (h1 : t.area = 200)
  (h2 : t.altitude = 10)
  (h3 : t.ab = 12)
  (h4 : t.cd = 22) :
  ∃ bc : ℝ, bc = 10 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_bc_length_l554_55496


namespace NUMINAMATH_CALUDE_two_person_travel_problem_l554_55478

/-- The problem of two people traveling between two locations --/
theorem two_person_travel_problem 
  (distance : ℝ) 
  (total_time : ℝ) 
  (speed_difference : ℝ) :
  distance = 25.5 ∧ 
  total_time = 3 ∧ 
  speed_difference = 2 →
  ∃ (speed_A speed_B : ℝ),
    speed_A = 2 * speed_B + speed_difference ∧
    speed_B * total_time + speed_A * total_time = 2 * distance ∧
    speed_A = 12 ∧
    speed_B = 5 := by sorry

end NUMINAMATH_CALUDE_two_person_travel_problem_l554_55478


namespace NUMINAMATH_CALUDE_grandson_height_prediction_l554_55489

/-- Predicts the height of the next generation using linear regression -/
def predict_next_height (heights : List ℝ) : ℝ :=
  sorry

theorem grandson_height_prediction 
  (heights : List ℝ) 
  (h1 : heights = [173, 170, 176, 182]) : 
  predict_next_height heights = 185 := by
  sorry

end NUMINAMATH_CALUDE_grandson_height_prediction_l554_55489


namespace NUMINAMATH_CALUDE_expression_simplification_l554_55470

theorem expression_simplification (q : ℝ) : 
  ((7*q + 3) - 3*q*5)*4 + (5 - 2/4)*(8*q - 12) = 4*q - 42 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l554_55470


namespace NUMINAMATH_CALUDE_expand_polynomial_l554_55450

theorem expand_polynomial (x : ℝ) : (x - 2) * (x + 2) * (x^2 + 4*x + 4) = x^4 + 4*x^3 - 16*x - 16 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l554_55450


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l554_55402

/-- A regular polygon with side length 7 and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) : 
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  (360 : ℝ) / n = exterior_angle → 
  n * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l554_55402


namespace NUMINAMATH_CALUDE_squirrel_walnut_theorem_l554_55434

/-- Calculates the final number of walnuts after squirrel activities -/
def final_walnut_count (initial : ℕ) (boy_gathered : ℕ) (boy_dropped : ℕ) (girl_brought : ℕ) (girl_ate : ℕ) : ℕ :=
  initial + boy_gathered - boy_dropped + girl_brought - girl_ate

/-- Theorem stating that given the squirrel activities, the final walnut count is 20 -/
theorem squirrel_walnut_theorem : 
  final_walnut_count 12 6 1 5 2 = 20 := by
  sorry

#eval final_walnut_count 12 6 1 5 2

end NUMINAMATH_CALUDE_squirrel_walnut_theorem_l554_55434


namespace NUMINAMATH_CALUDE_greatest_plants_per_row_l554_55485

theorem greatest_plants_per_row (sunflowers corn tomatoes : ℕ) 
  (h_sunflowers : sunflowers = 45)
  (h_corn : corn = 81)
  (h_tomatoes : tomatoes = 63) :
  Nat.gcd sunflowers (Nat.gcd corn tomatoes) = 9 := by
  sorry

end NUMINAMATH_CALUDE_greatest_plants_per_row_l554_55485


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l554_55449

theorem fraction_sum_equality : (3 + 6 + 9) / (2 + 5 + 8) + (2 + 5 + 8) / (3 + 6 + 9) = 61 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l554_55449


namespace NUMINAMATH_CALUDE_solution_difference_l554_55403

theorem solution_difference (r s : ℝ) : 
  (((6 * r - 18) / (r^2 + 3*r - 18) = r + 3) ∧
   ((6 * s - 18) / (s^2 + 3*s - 18) = s + 3) ∧
   (r ≠ s) ∧
   (r > s)) → 
  (r - s = 11) := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l554_55403


namespace NUMINAMATH_CALUDE_unique_p_for_three_positive_integer_roots_l554_55484

/-- The cubic equation with parameter p -/
def cubic_equation (p : ℝ) (x : ℝ) : ℝ :=
  5 * x^3 - 5 * (p + 1) * x^2 + (71 * p - 1) * x + 1 - 66 * p

/-- Predicate to check if a number is a positive integer -/
def is_positive_integer (x : ℝ) : Prop :=
  x > 0 ∧ ∃ n : ℕ, x = n

/-- The main theorem -/
theorem unique_p_for_three_positive_integer_roots :
  ∃! p : ℝ, ∃ x y z : ℝ,
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    is_positive_integer x ∧ is_positive_integer y ∧ is_positive_integer z ∧
    cubic_equation p x = 0 ∧ cubic_equation p y = 0 ∧ cubic_equation p z = 0 ∧
    p = 76 :=
sorry

end NUMINAMATH_CALUDE_unique_p_for_three_positive_integer_roots_l554_55484


namespace NUMINAMATH_CALUDE_work_hours_ratio_l554_55424

def total_hours : ℕ := 157
def rebecca_hours : ℕ := 56

theorem work_hours_ratio (thomas_hours toby_hours : ℕ) : 
  thomas_hours + toby_hours + rebecca_hours = total_hours →
  toby_hours = thomas_hours + 10 →
  rebecca_hours = toby_hours - 8 →
  (toby_hours : ℚ) / (thomas_hours : ℚ) = 32 / 27 := by
  sorry

end NUMINAMATH_CALUDE_work_hours_ratio_l554_55424


namespace NUMINAMATH_CALUDE_max_abs_Z_on_circle_l554_55456

open Complex

theorem max_abs_Z_on_circle (Z : ℂ) (h : abs (Z - (3 + 4*I)) = 1) :
  ∃ (M : ℝ), M = 6 ∧ ∀ (W : ℂ), abs (W - (3 + 4*I)) = 1 → abs W ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_abs_Z_on_circle_l554_55456


namespace NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l554_55498

theorem quadratic_reciprocal_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    (a^2 - 1) * x^2 - (a + 1) * x + 1 = 0 ∧
    (a^2 - 1) * y^2 - (a + 1) * y + 1 = 0 ∧
    x * y = 1) →
  a = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l554_55498


namespace NUMINAMATH_CALUDE_smallest_number_l554_55472

def jungkook_number : ℚ := 6 / 3
def yoongi_number : ℚ := 4
def yuna_number : ℚ := 5

theorem smallest_number : 
  jungkook_number ≤ yoongi_number ∧ jungkook_number ≤ yuna_number :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l554_55472


namespace NUMINAMATH_CALUDE_prob_ace_king_queen_same_suit_l554_55439

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards dealt -/
def CardsDealt : ℕ := 3

/-- Represents the probability of drawing a specific Ace from a standard deck -/
def ProbFirstAce : ℚ := 1 / StandardDeck

/-- Represents the probability of drawing a specific King after an Ace is drawn -/
def ProbSecondKing : ℚ := 1 / (StandardDeck - 1)

/-- Represents the probability of drawing a specific Queen after an Ace and a King are drawn -/
def ProbThirdQueen : ℚ := 1 / (StandardDeck - 2)

/-- The probability of dealing an Ace, King, and Queen of the same suit in that order -/
def ProbAceKingQueen : ℚ := ProbFirstAce * ProbSecondKing * ProbThirdQueen

theorem prob_ace_king_queen_same_suit :
  ProbAceKingQueen = 1 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_king_queen_same_suit_l554_55439


namespace NUMINAMATH_CALUDE_cost_price_calculation_l554_55469

theorem cost_price_calculation (C : ℝ) : 0.18 * C - 0.09 * C = 72 → C = 800 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l554_55469


namespace NUMINAMATH_CALUDE_combination_not_equal_permutation_div_n_factorial_l554_55474

/-- The number of combinations of n things taken m at a time -/
def C (n m : ℕ) : ℕ := sorry

/-- The number of permutations of n things taken m at a time -/
def A (n m : ℕ) : ℕ := sorry

theorem combination_not_equal_permutation_div_n_factorial (n m : ℕ) :
  C n m ≠ A n m / n! :=
sorry

end NUMINAMATH_CALUDE_combination_not_equal_permutation_div_n_factorial_l554_55474


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l554_55452

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other. -/
def symmetric_wrt_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- Given that point A(a, -2) is symmetric to point B(-3, b) with respect to the x-axis,
    prove that a + b = -1. -/
theorem symmetric_points_sum (a b : ℝ) : 
  symmetric_wrt_x_axis (a, -2) (-3, b) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l554_55452


namespace NUMINAMATH_CALUDE_cubic_root_coefficient_a_l554_55422

theorem cubic_root_coefficient_a (a b : ℚ) : 
  ((-1 - 4 * Real.sqrt 2)^3 + a * (-1 - 4 * Real.sqrt 2)^2 + b * (-1 - 4 * Real.sqrt 2) + 31 = 0) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_coefficient_a_l554_55422


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l554_55432

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a b : ℝ, ∀ x, x^2 - (m-3)*x + 16 = (a*x + b)^2) ↔ (m = -5 ∨ m = 11) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l554_55432


namespace NUMINAMATH_CALUDE_sugar_water_sweetness_l554_55460

theorem sugar_water_sweetness (a b m : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : m > 0) :
  (a + m) / (b + m) > a / b :=
by sorry

end NUMINAMATH_CALUDE_sugar_water_sweetness_l554_55460


namespace NUMINAMATH_CALUDE_train_bridge_time_l554_55436

/-- Time for a train to pass a bridge -/
theorem train_bridge_time (train_length bridge_length : ℝ) (train_speed_kmh : ℝ) :
  train_length = 360 →
  bridge_length = 140 →
  train_speed_kmh = 45 →
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_time_l554_55436


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l554_55435

theorem abs_inequality_equivalence (x : ℝ) :
  (1 ≤ |x - 2| ∧ |x - 2| ≤ 7) ↔ ((-5 ≤ x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x ≤ 9)) :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l554_55435


namespace NUMINAMATH_CALUDE_min_value_function_l554_55433

theorem min_value_function (x : ℝ) (h : x > -1) :
  (x^2 + 3*x + 4) / (x + 1) ≥ 2*Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_function_l554_55433


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l554_55419

theorem smallest_n_for_roots_of_unity : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (z : ℂ), z^6 - z^3 + 1 = 0 → z^n = 1) ∧
  (∀ (m : ℕ), m > 0 → (∀ (z : ℂ), z^6 - z^3 + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l554_55419


namespace NUMINAMATH_CALUDE_base_conversion_equality_l554_55444

/-- Given that 10b1₍₂₎ = a02₍₃₎, b ∈ {0, 1}, and a ∈ {0, 1, 2}, prove that a = 1 and b = 1 -/
theorem base_conversion_equality (a b : ℕ) : 
  (1 + 2 * b + 8 = 2 + 9 * a) → 
  (b = 0 ∨ b = 1) → 
  (a = 0 ∨ a = 1 ∨ a = 2) → 
  (a = 1 ∧ b = 1) := by
sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l554_55444


namespace NUMINAMATH_CALUDE_infinite_power_tower_four_implies_sqrt_two_l554_55445

/-- The limit of the infinite power tower x^(x^(x^...)) -/
noncomputable def infinitePowerTower (x : ℝ) : ℝ :=
  Real.log x / Real.log (Real.log x)

/-- Theorem: If the infinite power tower of x equals 4, then x equals √2 -/
theorem infinite_power_tower_four_implies_sqrt_two (x : ℝ) 
  (h_pos : x > 0) 
  (h_converge : infinitePowerTower x = 4) : 
  x = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_infinite_power_tower_four_implies_sqrt_two_l554_55445


namespace NUMINAMATH_CALUDE_growth_rate_equation_l554_55400

/-- Represents the average annual growth rate of a company's capital -/
def x : ℝ := sorry

/-- The initial capital of the company in millions of yuan -/
def initial_capital : ℝ := 10

/-- The final capital of the company after two years in millions of yuan -/
def final_capital : ℝ := 14.4

/-- The number of years over which the growth occurred -/
def years : ℕ := 2

/-- Theorem stating that the equation 1000(1+x)^2 = 1440 correctly represents 
    the average annual growth rate of the company's capital -/
theorem growth_rate_equation : 1000 * (1 + x)^years = 1440 := by sorry

end NUMINAMATH_CALUDE_growth_rate_equation_l554_55400


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l554_55441

def pizza_sharing (eva gwen noah mia : ℚ) : Prop :=
  eva = 1/4 ∧ gwen = 1/6 ∧ noah = 1/5 ∧ mia = 1 - (eva + gwen + noah)

theorem pizza_consumption_order (eva gwen noah mia : ℚ) 
  (h : pizza_sharing eva gwen noah mia) : 
  eva > mia ∧ mia > noah ∧ noah > gwen :=
by
  sorry

#check pizza_consumption_order

end NUMINAMATH_CALUDE_pizza_consumption_order_l554_55441


namespace NUMINAMATH_CALUDE_least_x_72_implies_n_8_l554_55481

theorem least_x_72_implies_n_8 (x : ℕ+) (p : ℕ) (n : ℕ+) :
  Nat.Prime p →
  (∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ (x : ℚ) / (n * p : ℚ) = q) →
  (∀ y : ℕ+, y < x → ¬∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ (y : ℚ) / (n * p : ℚ) = q) →
  x = 72 →
  n = 8 :=
by sorry

end NUMINAMATH_CALUDE_least_x_72_implies_n_8_l554_55481


namespace NUMINAMATH_CALUDE_inequality_proof_l554_55499

theorem inequality_proof (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : 1/x + 1/y + 1/z = 1) :
  a^x + b^y + c^z ≥ (4*a*b*c*x*y*z) / (x + y + z - 3)^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l554_55499
