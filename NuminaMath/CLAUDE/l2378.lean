import Mathlib

namespace NUMINAMATH_CALUDE_f_comparison_l2378_237835

def f (a b x : ℝ) := a * x^2 - 2 * b * x + 1

theorem f_comparison (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_increasing : ∀ x y, x ≤ y → y ≤ 0 → f a b x ≤ f a b y) :
  f a b (a - 2) < f a b (b + 1) :=
by sorry

end NUMINAMATH_CALUDE_f_comparison_l2378_237835


namespace NUMINAMATH_CALUDE_line_circle_intersect_l2378_237827

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a line in polar coordinates of the form ρsinθ = k -/
structure PolarLine where
  k : ℝ

/-- Represents a circle in polar coordinates of the form ρ = asinθ -/
structure PolarCircle where
  a : ℝ

/-- Check if a point lies on a polar line -/
def pointOnLine (p : PolarPoint) (l : PolarLine) : Prop :=
  p.ρ * Real.sin p.θ = l.k

/-- Check if a point lies on a polar circle -/
def pointOnCircle (p : PolarPoint) (c : PolarCircle) : Prop :=
  p.ρ = c.a * Real.sin p.θ

/-- Definition of intersection between a polar line and a polar circle -/
def intersect (l : PolarLine) (c : PolarCircle) : Prop :=
  ∃ p : PolarPoint, pointOnLine p l ∧ pointOnCircle p c

theorem line_circle_intersect (l : PolarLine) (c : PolarCircle) 
    (h1 : l.k = 2) (h2 : c.a = 4) : intersect l c := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersect_l2378_237827


namespace NUMINAMATH_CALUDE_simplify_expression_l2378_237847

theorem simplify_expression (α : ℝ) (h : π < α ∧ α < (3*π)/2) :
  Real.sqrt (1/2 + 1/2 * Real.sqrt (1/2 + 1/2 * Real.cos (2*α))) = Real.sin (α/2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2378_237847


namespace NUMINAMATH_CALUDE_mall_entrance_exit_ways_l2378_237844

theorem mall_entrance_exit_ways (n : Nat) (h : n = 4) : 
  (n * (n - 1) : Nat) = 12 := by
  sorry

#check mall_entrance_exit_ways

end NUMINAMATH_CALUDE_mall_entrance_exit_ways_l2378_237844


namespace NUMINAMATH_CALUDE_unique_solution_iff_a_eq_one_or_neg_one_l2378_237857

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*x ≤ 1 ∧ x - y = -a

-- Define what it means for the system to have a unique solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! x y, system x y a

-- Theorem statement
theorem unique_solution_iff_a_eq_one_or_neg_one :
  ∀ a : ℝ, has_unique_solution a ↔ (a = 1 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_a_eq_one_or_neg_one_l2378_237857


namespace NUMINAMATH_CALUDE_oranges_per_box_l2378_237864

/-- Given 56 oranges and 8 boxes, prove that the number of oranges per box is 7 -/
theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (h1 : total_oranges = 56) (h2 : num_boxes = 8) :
  total_oranges / num_boxes = 7 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l2378_237864


namespace NUMINAMATH_CALUDE_ratio_to_twelve_l2378_237840

theorem ratio_to_twelve : ∃ x : ℝ, (5 : ℝ) / 1 = x / 12 ∧ x = 60 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_twelve_l2378_237840


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2378_237860

theorem triangle_perimeter (a b c : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Positive side lengths
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
  ((a - 6) * (a - 3) = 0 ∨ (b - 6) * (b - 3) = 0 ∨ (c - 6) * (c - 3) = 0) →  -- At least one side satisfies the equation
  (a + b + c = 9 ∨ a + b + c = 15 ∨ a + b + c = 18) :=
by sorry


end NUMINAMATH_CALUDE_triangle_perimeter_l2378_237860


namespace NUMINAMATH_CALUDE_peanut_bags_needed_l2378_237899

-- Define the flight duration in hours
def flight_duration : ℕ := 2

-- Define the number of peanuts per bag
def peanuts_per_bag : ℕ := 30

-- Define the interval between eating peanuts in minutes
def eating_interval : ℕ := 1

-- Theorem statement
theorem peanut_bags_needed : 
  (flight_duration * 60) / peanuts_per_bag = 4 := by
  sorry

end NUMINAMATH_CALUDE_peanut_bags_needed_l2378_237899


namespace NUMINAMATH_CALUDE_unique_number_with_gcd_l2378_237806

theorem unique_number_with_gcd : ∃! n : ℕ, 70 ≤ n ∧ n < 80 ∧ Nat.gcd 30 n = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_gcd_l2378_237806


namespace NUMINAMATH_CALUDE_polynomial_sum_theorem_l2378_237879

theorem polynomial_sum_theorem (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 + x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_theorem_l2378_237879


namespace NUMINAMATH_CALUDE_two_digit_integers_mod_seven_l2378_237878

theorem two_digit_integers_mod_seven : 
  (Finset.filter (fun n => n ≥ 10 ∧ n < 100 ∧ n % 7 = 3) (Finset.range 100)).card = 13 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_integers_mod_seven_l2378_237878


namespace NUMINAMATH_CALUDE_wood_measurement_theorem_l2378_237843

/-- Represents the system of equations for the wood measurement problem -/
def wood_measurement_equations (x y : ℝ) : Prop :=
  (y - x = 4.5) ∧ (x - 1/2 * y = 1)

/-- Theorem stating that the given system of equations correctly represents the wood measurement problem -/
theorem wood_measurement_theorem (x y : ℝ) :
  (∃ wood_length : ℝ, wood_length = x) →
  (∃ rope_length : ℝ, rope_length = y) →
  (y - x = 4.5) →
  (x - 1/2 * y = 1) →
  wood_measurement_equations x y :=
by
  sorry

end NUMINAMATH_CALUDE_wood_measurement_theorem_l2378_237843


namespace NUMINAMATH_CALUDE_school_cafeteria_discussion_l2378_237825

theorem school_cafeteria_discussion (students_like : ℕ) (students_dislike : ℕ) : 
  students_like = 383 → students_dislike = 431 → students_like + students_dislike = 814 :=
by sorry

end NUMINAMATH_CALUDE_school_cafeteria_discussion_l2378_237825


namespace NUMINAMATH_CALUDE_football_lineup_count_l2378_237868

/-- The number of ways to choose a starting lineup from a football team. -/
def starting_lineup_count (total_players : ℕ) (offensive_linemen : ℕ) (lineup_size : ℕ) (linemen_in_lineup : ℕ) : ℕ :=
  (Nat.choose offensive_linemen linemen_in_lineup) *
  (Nat.choose (total_players - linemen_in_lineup) (lineup_size - linemen_in_lineup)) *
  (Nat.factorial (lineup_size - linemen_in_lineup))

/-- Theorem stating the number of ways to choose the starting lineup. -/
theorem football_lineup_count :
  starting_lineup_count 15 5 5 2 = 17160 := by
  sorry

end NUMINAMATH_CALUDE_football_lineup_count_l2378_237868


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l2378_237861

theorem sine_cosine_inequality (x : ℝ) (n : ℕ+) :
  (Real.sin (2 * x))^(n : ℕ) + ((Real.sin x)^(n : ℕ) - (Real.cos x)^(n : ℕ))^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l2378_237861


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2378_237842

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y = 1) :
  (1 / x + 1 / y) ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2378_237842


namespace NUMINAMATH_CALUDE_chocolate_solution_l2378_237854

/-- Calculates the amount Tom paid for chocolates and the number of pieces he has left --/
def chocolate_problem (total_boxes : ℕ) (price_per_box : ℚ) (boxes_given_away : ℕ) 
  (pieces_per_box : ℕ) (discount_percent : ℚ) : ℚ × ℕ :=
  let total_cost := total_boxes * price_per_box
  let discount_amount := discount_percent * total_cost
  let final_cost := total_cost - discount_amount
  let boxes_left := total_boxes - boxes_given_away
  let pieces_left := boxes_left * pieces_per_box
  (final_cost, pieces_left)

/-- Theorem stating the correct solution to the chocolate problem --/
theorem chocolate_solution : 
  chocolate_problem 12 4 7 6 (15/100) = (40.8, 30) := by sorry

end NUMINAMATH_CALUDE_chocolate_solution_l2378_237854


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2378_237815

theorem inscribed_circle_radius (PQ QR : Real) (h1 : PQ = 15) (h2 : QR = 8) : 
  let PR := Real.sqrt (PQ^2 + QR^2)
  let s := (PQ + QR + PR) / 2
  let area := PQ * QR / 2
  area / s = 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2378_237815


namespace NUMINAMATH_CALUDE_sqrt_expressions_equality_l2378_237884

theorem sqrt_expressions_equality :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    (Real.sqrt (24 * a) - Real.sqrt (18 * b)) - Real.sqrt (6 * c) = 
    Real.sqrt (6 * c) - 3 * Real.sqrt (2 * b)) ∧
  (∀ d e f : ℝ, d > 0 → e > 0 → f > 0 →
    2 * Real.sqrt (12 * d) * Real.sqrt ((1 / 8) * e) + 5 * Real.sqrt (2 * f) = 
    Real.sqrt (6 * d) + 5 * Real.sqrt (2 * f)) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_expressions_equality_l2378_237884


namespace NUMINAMATH_CALUDE_difference_at_negative_five_l2378_237855

-- Define the functions f and g
def f (x : ℝ) : ℝ := 5 * x^2 - 3 * x + 8
def g (k : ℤ) (x : ℝ) : ℝ := x^3 - k * x - 10

-- State the theorem
theorem difference_at_negative_five (k : ℤ) : f (-5) - g k (-5) = -24 → k = 61 := by
  sorry

end NUMINAMATH_CALUDE_difference_at_negative_five_l2378_237855


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2378_237828

theorem negation_of_universal_proposition 
  (f : ℝ → ℝ) (m : ℝ) : 
  (¬ ∀ x, f x ≥ m) ↔ (∃ x, f x < m) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2378_237828


namespace NUMINAMATH_CALUDE_function_value_at_2012_l2378_237874

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx + β) where f(2001) = 3, 
    prove that f(2012) = -3 -/
theorem function_value_at_2012 
  (a b α β : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β))
  (h2 : f 2001 = 3) :
  f 2012 = -3 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_2012_l2378_237874


namespace NUMINAMATH_CALUDE_sin_cube_identity_l2378_237813

theorem sin_cube_identity (θ : Real) : 
  Real.sin θ ^ 3 = -1/4 * Real.sin (3 * θ) + 3/4 * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_sin_cube_identity_l2378_237813


namespace NUMINAMATH_CALUDE_common_prime_root_quadratics_l2378_237822

theorem common_prime_root_quadratics (a b : ℤ) : 
  (∃ p : ℕ, Prime p ∧ 
    (p : ℤ)^2 + a * (p : ℤ) + b = 0 ∧ 
    (p : ℤ)^2 + b * (p : ℤ) + 1100 = 0) →
  a = 274 ∨ a = 40 := by
sorry

end NUMINAMATH_CALUDE_common_prime_root_quadratics_l2378_237822


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l2378_237817

theorem smallest_multiplier_for_perfect_square : ∃ (k : ℕ+), 
  (∀ (m : ℕ+), (∃ (n : ℕ), 2010 * m = n * n) → k ≤ m) ∧ 
  (∃ (n : ℕ), 2010 * k = n * n) ∧
  k = 2010 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l2378_237817


namespace NUMINAMATH_CALUDE_point_on_line_l2378_237890

theorem point_on_line (m n k : ℝ) : 
  (m = 2 * n + 5) ∧ (m + 4 = 2 * (n + k) + 5) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2378_237890


namespace NUMINAMATH_CALUDE_divisor_property_solutions_l2378_237867

/-- The number of positive divisors of a positive integer n -/
def num_divisors (n : ℕ+) : ℕ+ :=
  sorry

/-- The property that the fourth power of the number of divisors equals the number itself -/
def has_divisor_property (m : ℕ+) : Prop :=
  (num_divisors m) ^ 4 = m

/-- Theorem stating that only 625, 6561, and 4100625 satisfy the divisor property -/
theorem divisor_property_solutions :
  ∀ m : ℕ+, has_divisor_property m ↔ m ∈ ({625, 6561, 4100625} : Set ℕ+) :=
sorry

end NUMINAMATH_CALUDE_divisor_property_solutions_l2378_237867


namespace NUMINAMATH_CALUDE_exists_divisible_by_four_l2378_237887

def collatz_sequence (a₁ : ℕ+) : ℕ → ℕ
  | 0 => a₁.val
  | n + 1 => 
    let prev := collatz_sequence a₁ n
    if prev % 2 = 0 then prev / 2 else 3 * prev + 1

theorem exists_divisible_by_four (a₁ : ℕ+) : 
  ∃ n : ℕ, (collatz_sequence a₁ n) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_four_l2378_237887


namespace NUMINAMATH_CALUDE_percentage_of_150_l2378_237805

theorem percentage_of_150 : (1 / 5 : ℚ) / 100 * 150 = 0.3 := by sorry

end NUMINAMATH_CALUDE_percentage_of_150_l2378_237805


namespace NUMINAMATH_CALUDE_number_problem_l2378_237826

theorem number_problem : ∃ x : ℚ, (x / 6) * 12 = 12 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2378_237826


namespace NUMINAMATH_CALUDE_anniversary_number_is_counting_l2378_237880

/-- Represents the categories of numbers in context --/
inductive NumberCategory
  | Label
  | MeasurementResult
  | Counting

/-- Represents the context in which the number is used --/
structure AnniversaryContext where
  years : ℕ

/-- Determines the category of a number in the anniversary context --/
def categorizeAnniversaryNumber (context : AnniversaryContext) : NumberCategory :=
  NumberCategory.Counting

/-- Theorem stating that the number used for anniversary years is a counting number --/
theorem anniversary_number_is_counting (context : AnniversaryContext) :
  categorizeAnniversaryNumber context = NumberCategory.Counting :=
by sorry

end NUMINAMATH_CALUDE_anniversary_number_is_counting_l2378_237880


namespace NUMINAMATH_CALUDE_poster_placement_l2378_237881

/-- Given a wall of width 25 feet and a centrally placed poster of width 4 feet,
    the distance from the end of the wall to the nearest edge of the poster is 10.5 feet. -/
theorem poster_placement (wall_width : ℝ) (poster_width : ℝ) 
    (h1 : wall_width = 25) 
    (h2 : poster_width = 4) :
  (wall_width - poster_width) / 2 = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_poster_placement_l2378_237881


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2378_237834

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  (180 * (n - 2) : ℝ) / n = 150 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2378_237834


namespace NUMINAMATH_CALUDE_simplify_fraction_l2378_237876

theorem simplify_fraction : (210 : ℚ) / 7350 * 14 = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2378_237876


namespace NUMINAMATH_CALUDE_exists_rectangle_same_parity_l2378_237823

/-- Represents a rectangle on a grid -/
structure GridRectangle where
  length : ℕ
  width : ℕ

/-- Represents a square cut into rectangles -/
structure CutSquare where
  side_length : ℕ
  rectangles : List GridRectangle

/-- Checks if a number is even -/
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Checks if two numbers have the same parity -/
def same_parity (a b : ℕ) : Prop :=
  (is_even a ∧ is_even b) ∨ (¬is_even a ∧ ¬is_even b)

/-- Main theorem: In a square with side length 2009 cut into rectangles,
    there exists at least one rectangle with sides of the same parity -/
theorem exists_rectangle_same_parity (sq : CutSquare) 
    (h1 : sq.side_length = 2009) 
    (h2 : sq.rectangles.length > 0) : 
    ∃ (r : GridRectangle), r ∈ sq.rectangles ∧ same_parity r.length r.width := by
  sorry

end NUMINAMATH_CALUDE_exists_rectangle_same_parity_l2378_237823


namespace NUMINAMATH_CALUDE_max_grain_mass_l2378_237820

/-- The maximum mass of grain that can be loaded onto a rectangular platform -/
theorem max_grain_mass (length width : Real) (max_angle : Real) (density : Real) :
  length = 10 ∧ 
  width = 5 ∧ 
  max_angle = π / 4 ∧ 
  density = 1200 →
  ∃ (mass : Real),
    mass = 175000 ∧ 
    mass = density * (length * width * (width / 2) / 2 + length * width * (width / 4))
    := by sorry

end NUMINAMATH_CALUDE_max_grain_mass_l2378_237820


namespace NUMINAMATH_CALUDE_initial_water_fraction_in_larger_jar_l2378_237862

theorem initial_water_fraction_in_larger_jar 
  (small_capacity large_capacity : ℝ) 
  (h1 : small_capacity > 0) 
  (h2 : large_capacity > 0) 
  (h3 : small_capacity ≠ large_capacity) :
  let water_amount := (1/5) * small_capacity
  let initial_fraction := water_amount / large_capacity
  let combined_fraction := (water_amount + water_amount) / large_capacity
  (combined_fraction = 0.4) → (initial_fraction = 1/10) := by
  sorry

end NUMINAMATH_CALUDE_initial_water_fraction_in_larger_jar_l2378_237862


namespace NUMINAMATH_CALUDE_no_solution_to_inequality_l2378_237809

theorem no_solution_to_inequality : ¬ ∃ x : ℝ, |x - 3| + |x + 4| < 6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_inequality_l2378_237809


namespace NUMINAMATH_CALUDE_blue_pens_count_l2378_237877

/-- Given the prices of red and blue pens, the total amount spent, and the total number of pens,
    prove that the number of blue pens bought is 11. -/
theorem blue_pens_count (red_price blue_price total_spent total_pens : ℕ) 
    (h1 : red_price = 5)
    (h2 : blue_price = 7)
    (h3 : total_spent = 102)
    (h4 : total_pens = 16) : 
  ∃ (red_count blue_count : ℕ),
    red_count + blue_count = total_pens ∧
    red_count * red_price + blue_count * blue_price = total_spent ∧
    blue_count = 11 := by
  sorry

end NUMINAMATH_CALUDE_blue_pens_count_l2378_237877


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2378_237886

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - 2*x - 3 < 0 ↔ -1 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2378_237886


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2378_237812

theorem rationalize_denominator (x : ℝ) (hx : x > 0) :
  (1 : ℝ) / (x^(1/3) + (27 : ℝ)^(1/3)) = (4 : ℝ)^(1/3) / (2 + 3 * (4 : ℝ)^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2378_237812


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l2378_237845

theorem consecutive_integers_product_sum (n : ℤ) : 
  n * (n + 1) * (n + 2) * (n + 3) = 3024 → n + (n + 1) + (n + 2) + (n + 3) = 30 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l2378_237845


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_subset_condition_l2378_237893

-- Define the sets P and Q
def P : Set ℝ := {x | 2*x^2 - 3*x + 1 ≤ 0}
def Q (a : ℝ) : Set ℝ := {x | (x-a)*(x-a-1) ≤ 0}

-- Theorem 1: P ∩ Q = {1} when a = 1
theorem intersection_when_a_is_one : P ∩ (Q 1) = {1} := by sorry

-- Theorem 2: P ⊆ Q if and only if 0 ≤ a ≤ 1/2
theorem subset_condition (a : ℝ) : P ⊆ Q a ↔ 0 ≤ a ∧ a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_subset_condition_l2378_237893


namespace NUMINAMATH_CALUDE_uncovered_area_of_rectangles_l2378_237851

theorem uncovered_area_of_rectangles (small_length small_width large_length large_width : ℝ) 
  (h1 : small_length = 4)
  (h2 : small_width = 2)
  (h3 : large_length = 10)
  (h4 : large_width = 6)
  (h5 : small_length ≤ large_length)
  (h6 : small_width ≤ large_width) :
  large_length * large_width - small_length * small_width = 52 := by
sorry

end NUMINAMATH_CALUDE_uncovered_area_of_rectangles_l2378_237851


namespace NUMINAMATH_CALUDE_ratio_problem_l2378_237872

theorem ratio_problem (x y z w : ℚ) 
  (h1 : x / y = 24)
  (h2 : z / y = 8)
  (h3 : z / w = 1 / 12) :
  x / w = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2378_237872


namespace NUMINAMATH_CALUDE_coin_division_problem_l2378_237885

theorem coin_division_problem (n : ℕ) : 
  (n > 0) →
  (n % 8 = 5) → 
  (n % 7 = 2) → 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 5 ∨ m % 7 ≠ 2)) →
  (n % 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_coin_division_problem_l2378_237885


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_seven_sqrt_two_over_six_l2378_237801

theorem sqrt_difference_equals_seven_sqrt_two_over_six :
  Real.sqrt (9 / 2) - Real.sqrt (2 / 9) = (7 * Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_seven_sqrt_two_over_six_l2378_237801


namespace NUMINAMATH_CALUDE_katies_cupcakes_l2378_237883

theorem katies_cupcakes (cupcakes cookies left_over sold : ℕ) :
  cookies = 5 →
  left_over = 8 →
  sold = 4 →
  cupcakes + cookies = left_over + sold →
  cupcakes = 7 := by
sorry

end NUMINAMATH_CALUDE_katies_cupcakes_l2378_237883


namespace NUMINAMATH_CALUDE_two_numbers_with_given_means_l2378_237804

theorem two_numbers_with_given_means (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (Real.sqrt (a * b) = Real.sqrt 5) → 
  (2 / (1/a + 1/b) = 5/3) → 
  ((a = 1 ∧ b = 5) ∨ (a = 5 ∧ b = 1)) := by
sorry

end NUMINAMATH_CALUDE_two_numbers_with_given_means_l2378_237804


namespace NUMINAMATH_CALUDE_system_solution_iff_m_neq_one_l2378_237814

/-- The system of equations has at least one solution if and only if m ≠ 1 -/
theorem system_solution_iff_m_neq_one (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 2 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_iff_m_neq_one_l2378_237814


namespace NUMINAMATH_CALUDE_lawn_mowing_theorem_l2378_237850

/-- Represents the time (in hours) it takes to mow the entire lawn -/
def MaryTime : ℚ := 4
def TomTime : ℚ := 5

/-- Represents the fraction of the lawn mowed per hour -/
def MaryRate : ℚ := 1 / MaryTime
def TomRate : ℚ := 1 / TomTime

/-- Represents the time Tom works alone -/
def TomAloneTime : ℚ := 3

/-- Represents the time Mary and Tom work together -/
def TogetherTime : ℚ := 1

/-- The fraction of lawn remaining to be mowed -/
def RemainingFraction : ℚ := 1 / 20

theorem lawn_mowing_theorem :
  1 - (TomRate * TomAloneTime + (MaryRate + TomRate) * TogetherTime) = RemainingFraction := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_theorem_l2378_237850


namespace NUMINAMATH_CALUDE_part1_solution_set_part2_solution_set_l2378_237816

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part1_solution_set :
  {x : ℝ | f 1 x < |2*x - 1| - 1} = {x : ℝ | x < -1 ∨ x > 1} := by sorry

-- Part 2
theorem part2_solution_set :
  ∀ x ∈ Set.Ioo (-2) 1, {a : ℝ | |x - 1| > |2*x - a - 1| - f a x} = Set.Iic (-2) := by sorry

end NUMINAMATH_CALUDE_part1_solution_set_part2_solution_set_l2378_237816


namespace NUMINAMATH_CALUDE_solution_set_is_real_solution_set_is_empty_solution_set_has_element_l2378_237800

-- Define the quadratic expression
def f (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 * a - 3

-- Define the solution set for the inequality
def solution_set (a : ℝ) : Set ℝ := {x | f a x < 0}

-- Theorem 1: The solution set is ℝ iff a ∈ (-∞, 0]
theorem solution_set_is_real : ∀ a : ℝ, solution_set a = Set.univ ↔ a ≤ 0 := by sorry

-- Theorem 2: The solution set is ∅ iff a ∈ [3, +∞)
theorem solution_set_is_empty : ∀ a : ℝ, solution_set a = ∅ ↔ a ≥ 3 := by sorry

-- Theorem 3: There is at least one real solution iff a ∈ (-∞, 3)
theorem solution_set_has_element : ∀ a : ℝ, (∃ x : ℝ, x ∈ solution_set a) ↔ a < 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_real_solution_set_is_empty_solution_set_has_element_l2378_237800


namespace NUMINAMATH_CALUDE_probability_square_or_triangle_l2378_237838

theorem probability_square_or_triangle :
  let total_figures : ℕ := 10
  let num_triangles : ℕ := 4
  let num_squares : ℕ := 3
  let num_circles : ℕ := 3
  let favorable_outcomes : ℕ := num_triangles + num_squares
  (favorable_outcomes : ℚ) / total_figures = 7 / 10 :=
by sorry

end NUMINAMATH_CALUDE_probability_square_or_triangle_l2378_237838


namespace NUMINAMATH_CALUDE_exactly_three_valid_sets_l2378_237858

/-- A set of consecutive positive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  length_ge_3 : length ≥ 3

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- Predicate for a valid set (sum equals 150) -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  sum_consecutive s = 150

theorem exactly_three_valid_sets :
  ∃! (sets : Finset ConsecutiveSet), 
    (∀ s ∈ sets, is_valid_set s) ∧ 
    Finset.card sets = 3 := by sorry

end NUMINAMATH_CALUDE_exactly_three_valid_sets_l2378_237858


namespace NUMINAMATH_CALUDE_infinite_solutions_exist_l2378_237871

theorem infinite_solutions_exist :
  ∃ f : ℕ → ℕ → ℕ × ℕ × ℕ,
    ∀ u v : ℕ, u > 1 → v > 1 →
      let (x, y, z) := f u v
      x^2015 + y^2015 = z^2016 ∧
      x ≠ y ∧ y ≠ z ∧ x ≠ z :=
by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_exist_l2378_237871


namespace NUMINAMATH_CALUDE_compass_leg_swap_impossible_l2378_237848

/-- Represents a point on the integer grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents the state of the compass -/
structure CompassState where
  leg1 : GridPoint
  leg2 : GridPoint

/-- Calculates the squared distance between two grid points -/
def squaredDistance (p1 p2 : GridPoint) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Defines a valid move of the compass -/
def isValidMove (s1 s2 : CompassState) : Prop :=
  (s1.leg1 = s2.leg1 ∧ squaredDistance s1.leg1 s1.leg2 = squaredDistance s2.leg1 s2.leg2) ∨
  (s1.leg2 = s2.leg2 ∧ squaredDistance s1.leg1 s1.leg2 = squaredDistance s2.leg1 s2.leg2)

/-- Defines a sequence of valid moves -/
def isValidMoveSequence : List CompassState → Prop
  | [] => True
  | [_] => True
  | s1 :: s2 :: rest => isValidMove s1 s2 ∧ isValidMoveSequence (s2 :: rest)

theorem compass_leg_swap_impossible (start finish : CompassState) 
  (h_start_distance : squaredDistance start.leg1 start.leg2 = squaredDistance finish.leg1 finish.leg2)
  (h_swap : start.leg1 = finish.leg2 ∧ start.leg2 = finish.leg1) :
  ¬∃ (moves : List CompassState), isValidMoveSequence (start :: moves ++ [finish]) :=
sorry

end NUMINAMATH_CALUDE_compass_leg_swap_impossible_l2378_237848


namespace NUMINAMATH_CALUDE_min_value_3x_4y_l2378_237882

theorem min_value_3x_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = 5 * x₀ * y₀ ∧ 3 * x₀ + 4 * y₀ = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_3x_4y_l2378_237882


namespace NUMINAMATH_CALUDE_certain_number_problem_l2378_237898

theorem certain_number_problem (x : ℝ) (h : 0.6 * x = 0.4 * 30 + 18) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2378_237898


namespace NUMINAMATH_CALUDE_vector_problem_l2378_237891

def a : Fin 2 → ℝ := ![- 3, 1]
def b : Fin 2 → ℝ := ![1, -2]
def c : Fin 2 → ℝ := ![1, -1]

def m (k : ℝ) : Fin 2 → ℝ := fun i ↦ a i + k * b i

theorem vector_problem :
  (∃ k : ℝ, (∀ i : Fin 2, m k i * (2 * a i - b i) = 0) ∧ k = 5 / 3) ∧
  (∃ k : ℝ, (∀ i : Fin 2, ∃ t : ℝ, m k i = t * (k * b i + c i)) ∧ k = -1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l2378_237891


namespace NUMINAMATH_CALUDE_chair_difference_l2378_237859

theorem chair_difference (initial_chairs left_chairs : ℕ) : 
  initial_chairs = 15 → left_chairs = 3 → initial_chairs - left_chairs = 12 := by
  sorry

end NUMINAMATH_CALUDE_chair_difference_l2378_237859


namespace NUMINAMATH_CALUDE_sum_of_ages_l2378_237888

/-- Viggo's age when his brother was 2 years old -/
def viggos_age_when_brother_was_2 (brothers_age_when_2 : ℕ) : ℕ :=
  10 + 2 * brothers_age_when_2

/-- The current age of Viggo's younger brother -/
def brothers_current_age : ℕ := 10

/-- Viggo's current age -/
def viggos_current_age : ℕ :=
  brothers_current_age + (viggos_age_when_brother_was_2 2 - 2)

theorem sum_of_ages : 
  viggos_current_age + brothers_current_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2378_237888


namespace NUMINAMATH_CALUDE_triangular_plot_size_l2378_237829

/-- The size of a triangular plot of land in acres, given its dimensions on a map and conversion factors. -/
theorem triangular_plot_size (base height : ℝ) (scale_factor : ℝ) (acres_per_square_mile : ℝ) : 
  base = 8 → height = 12 → scale_factor = 1 → acres_per_square_mile = 320 →
  (1/2 * base * height) * scale_factor^2 * acres_per_square_mile = 15360 := by
  sorry

end NUMINAMATH_CALUDE_triangular_plot_size_l2378_237829


namespace NUMINAMATH_CALUDE_family_travel_info_l2378_237897

structure FamilyMember where
  name : String
  statement : String

structure TravelInfo where
  origin : String
  destination : String
  stopover : Option String

def father : FamilyMember :=
  { name := "Father", statement := "We are going to Spain (we are coming from Newcastle)." }

def mother : FamilyMember :=
  { name := "Mother", statement := "We are not going to Spain but are coming from Newcastle (we stopped in Paris and are not going to Spain)." }

def daughter : FamilyMember :=
  { name := "Daughter", statement := "We are not coming from Newcastle (we stopped in Paris)." }

def family : List FamilyMember := [father, mother, daughter]

def interpretStatements (family : List FamilyMember) : TravelInfo :=
  { origin := "Newcastle", destination := "", stopover := some "Paris" }

theorem family_travel_info (family : List FamilyMember) :
  interpretStatements family = { origin := "Newcastle", destination := "", stopover := some "Paris" } :=
sorry

end NUMINAMATH_CALUDE_family_travel_info_l2378_237897


namespace NUMINAMATH_CALUDE_problem_solution_l2378_237833

theorem problem_solution (a b : ℝ) (h_distinct : a ≠ b) (h_sum_squares : a^2 + b^2 = 5) :
  (ab = 2 → a + b = 3 ∨ a + b = -3) ∧
  (a^2 - 2*a = b^2 - 2*b → a + b = 2 ∧ a^2 - 2*a = (1/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2378_237833


namespace NUMINAMATH_CALUDE_group_sizes_min_group_a_size_l2378_237866

/-- Represents the ticket price based on the number of people -/
def ticket_price (m : ℕ) : ℕ :=
  if 10 ≤ m ∧ m ≤ 50 then 60
  else if 51 ≤ m ∧ m ≤ 100 then 50
  else 40

/-- The total number of people in both groups -/
def total_people : ℕ := 102

/-- The total amount paid when buying tickets separately -/
def total_amount : ℕ := 5580

/-- Theorem stating the number of people in each group -/
theorem group_sizes :
  ∃ (a b : ℕ), a < 50 ∧ b > 50 ∧ a + b = total_people ∧
  ticket_price a * a + ticket_price b * b = total_amount :=
sorry

/-- Theorem stating the minimum number of people in Group A for savings -/
theorem min_group_a_size :
  ∃ (min_a : ℕ), ∀ a : ℕ, a ≥ min_a →
  ticket_price a * a + ticket_price (total_people - a) * (total_people - a) - 
  ticket_price total_people * total_people ≥ 1200 :=
sorry

end NUMINAMATH_CALUDE_group_sizes_min_group_a_size_l2378_237866


namespace NUMINAMATH_CALUDE_soft_drink_price_l2378_237852

/-- The price increase of a soft drink over 10 years -/
def price_increase (initial_price : ℕ) (increase_5p : ℕ) (increase_2p : ℕ) : ℚ :=
  (initial_price + 5 * increase_5p + 2 * increase_2p) / 100

/-- Theorem stating the final price of the soft drink -/
theorem soft_drink_price :
  price_increase 70 4 6 = 102 / 100 := by sorry

end NUMINAMATH_CALUDE_soft_drink_price_l2378_237852


namespace NUMINAMATH_CALUDE_power_four_remainder_l2378_237810

theorem power_four_remainder (a : ℕ) (h1 : a > 0) (h2 : 2 ∣ a) : 4^a % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_four_remainder_l2378_237810


namespace NUMINAMATH_CALUDE_divisibility_of_consecutive_ones_l2378_237841

/-- A number consisting of n consecutive ones -/
def consecutive_ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

theorem divisibility_of_consecutive_ones :
  ∃ k : ℕ, consecutive_ones 1998 = 37 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_consecutive_ones_l2378_237841


namespace NUMINAMATH_CALUDE_inequality_proof_l2378_237889

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2378_237889


namespace NUMINAMATH_CALUDE_max_marks_calculation_l2378_237870

/-- The maximum marks in an exam where:
  - The passing mark is 35% of the maximum marks
  - A student got 185 marks
  - The student failed by 25 marks
-/
theorem max_marks_calculation : ∃ (M : ℝ), 
  (0.35 * M = 185 + 25) ∧ 
  (M = 600) := by
  sorry

end NUMINAMATH_CALUDE_max_marks_calculation_l2378_237870


namespace NUMINAMATH_CALUDE_joan_seashells_l2378_237895

/-- The number of seashells Joan gave to Sam -/
def seashells_given : ℕ := 43

/-- The number of seashells Joan has left -/
def seashells_left : ℕ := 27

/-- The total number of seashells Joan found originally -/
def total_seashells : ℕ := seashells_given + seashells_left

theorem joan_seashells : total_seashells = 70 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l2378_237895


namespace NUMINAMATH_CALUDE_fraction_reducibility_implies_divisibility_l2378_237831

theorem fraction_reducibility_implies_divisibility 
  (a b c n l p : ℤ) 
  (h_reducible : ∃ (k m : ℤ), a * l + b = p * k ∧ c * l + n = p * m) : 
  p ∣ (a * n - b * c) := by
sorry

end NUMINAMATH_CALUDE_fraction_reducibility_implies_divisibility_l2378_237831


namespace NUMINAMATH_CALUDE_gasoline_consumption_reduction_l2378_237853

theorem gasoline_consumption_reduction 
  (original_price original_quantity : ℝ) 
  (price_increase : ℝ) 
  (spending_increase : ℝ) 
  (h1 : price_increase = 0.20) 
  (h2 : spending_increase = 0.14) : 
  let new_price := original_price * (1 + price_increase)
  let new_spending := original_price * original_quantity * (1 + spending_increase)
  let new_quantity := new_spending / new_price
  (original_quantity - new_quantity) / original_quantity = 0.05 := by
sorry

end NUMINAMATH_CALUDE_gasoline_consumption_reduction_l2378_237853


namespace NUMINAMATH_CALUDE_triangle_theorem_l2378_237873

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sine_rule : a / Real.sin A = b / Real.sin B
  angle_sum : A + B + C = π

/-- The theorem to be proved -/
theorem triangle_theorem (t : AcuteTriangle) 
  (h1 : 2 * t.a * Real.sin t.B = Real.sqrt 3 * t.b)
  (h2 : t.a = 6)
  (h3 : t.b + t.c = 8) :
  t.A = π/3 ∧ 
  1/2 * t.b * t.c * Real.sin t.A = 7 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2378_237873


namespace NUMINAMATH_CALUDE_inequality_range_l2378_237865

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3*a) ↔ (a ≤ -1 ∨ a ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2378_237865


namespace NUMINAMATH_CALUDE_alberts_age_l2378_237846

theorem alberts_age (dad_age : ℕ) (h1 : dad_age = 48) : ∃ (albert_age : ℕ),
  (albert_age = 15) ∧ 
  (dad_age - 4 = 4 * (albert_age - 4)) :=
by
  sorry

end NUMINAMATH_CALUDE_alberts_age_l2378_237846


namespace NUMINAMATH_CALUDE_quadratic_expression_rewrite_l2378_237892

theorem quadratic_expression_rewrite (i j : ℂ) : 
  let expression := 8 * j^2 + (6 * i) * j + 16
  ∃ (c p q : ℂ), 
    expression = c * (j + p)^2 + q ∧ 
    q / p = -137 * I / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_rewrite_l2378_237892


namespace NUMINAMATH_CALUDE_four_digit_divisibility_l2378_237811

def is_two_digit_prime (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem four_digit_divisibility (p q : ℕ) : 
  is_two_digit_prime p ∧ 
  is_two_digit_prime q ∧ 
  p ≠ q ∧
  (100 * p + q) % ((p + q) / 2) = 0 ∧ 
  (100 * q + p) % ((p + q) / 2) = 0 →
  ({p, q} : Set ℕ) = {13, 53} ∨ 
  ({p, q} : Set ℕ) = {19, 47} ∨ 
  ({p, q} : Set ℕ) = {23, 43} ∨ 
  ({p, q} : Set ℕ) = {29, 37} :=
by sorry

end NUMINAMATH_CALUDE_four_digit_divisibility_l2378_237811


namespace NUMINAMATH_CALUDE_divisibility_condition_l2378_237837

theorem divisibility_condition (a b : ℤ) : 
  (∃ d : ℕ, d ≥ 2 ∧ ∀ n : ℕ, n > 0 → (d : ℤ) ∣ (a^n + b^n + 1)) ↔ 
  ((a % 2 = 0 ∧ b % 2 = 1) ∨ (a % 3 = 1 ∧ b % 3 = 1)) := by
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2378_237837


namespace NUMINAMATH_CALUDE_club_president_secretary_choices_l2378_237832

/-- A club with boys and girls -/
structure Club where
  total : ℕ
  boys : ℕ
  girls : ℕ

/-- The number of ways to choose a president (boy) and secretary (girl) from a club -/
def choosePresidentAndSecretary (c : Club) : ℕ :=
  c.boys * c.girls

/-- Theorem stating that for a club with 30 members (18 boys and 12 girls),
    the number of ways to choose a president and secretary is 216 -/
theorem club_president_secretary_choices :
  let c : Club := { total := 30, boys := 18, girls := 12 }
  choosePresidentAndSecretary c = 216 := by
  sorry

end NUMINAMATH_CALUDE_club_president_secretary_choices_l2378_237832


namespace NUMINAMATH_CALUDE_inequality_proof_l2378_237818

theorem inequality_proof (x y z w : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0)
  (hxy : x + y ≠ 0) (hzw : z + w ≠ 0) (hxyzw : x * y + z * w ≥ 0) :
  ((x + y) / (z + w) + (z + w) / (x + y))⁻¹ + 1 / 2 ≥ 
  (x / z + z / x)⁻¹ + (y / w + w / y)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2378_237818


namespace NUMINAMATH_CALUDE_weight_of_b_l2378_237821

/-- Given three weights a, b, and c, prove that b = 33 under the given conditions -/
theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 44 →
  b = 33 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l2378_237821


namespace NUMINAMATH_CALUDE_remainder_theorem_l2378_237819

/-- The polynomial f(x) = x^5 - 8x^4 + 15x^3 + 20x^2 - 5x - 20 -/
def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 15*x^3 + 20*x^2 - 5*x - 20

/-- The theorem statement -/
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, f = fun x ↦ (x - 4) * q x + 216 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2378_237819


namespace NUMINAMATH_CALUDE_unique_multiplication_solution_l2378_237808

/-- Represents a three-digit number in the form abb --/
def three_digit (a b : Nat) : Nat := 100 * a + 10 * b + b

/-- Represents a four-digit number in the form bcb1 --/
def four_digit (b c : Nat) : Nat := 1000 * b + 100 * c + 10 * b + 1

theorem unique_multiplication_solution :
  ∃! (a b c : Nat),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    three_digit a b * c = four_digit b c ∧
    a = 5 ∧ b = 3 ∧ c = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_multiplication_solution_l2378_237808


namespace NUMINAMATH_CALUDE_range_of_fraction_l2378_237807

theorem range_of_fraction (a b : ℝ) (ha : 1 < a ∧ a < 2) (hb : -2 < b ∧ b < -1) :
  ∃ x, -2 < x ∧ x < -1/2 ∧ x = a/b :=
by sorry

end NUMINAMATH_CALUDE_range_of_fraction_l2378_237807


namespace NUMINAMATH_CALUDE_petyas_friends_l2378_237869

theorem petyas_friends (total_stickers : ℕ) : 
  (∃ (x : ℕ), 5 * x + 8 = total_stickers ∧ 6 * x = total_stickers + 11) → 
  (∃ (x : ℕ), x = 19 ∧ 5 * x + 8 = total_stickers ∧ 6 * x = total_stickers + 11) :=
by sorry

end NUMINAMATH_CALUDE_petyas_friends_l2378_237869


namespace NUMINAMATH_CALUDE_A_B_red_mutually_exclusive_not_contradictory_l2378_237836

-- Define the set of cards
inductive Card : Type
| Black : Card
| Red : Card
| White : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "A gets the red card"
def A_gets_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "B gets the red card"
def B_gets_red (d : Distribution) : Prop := d Person.B = Card.Red

-- Theorem stating that "A gets the red card" and "B gets the red card" are mutually exclusive but not contradictory
theorem A_B_red_mutually_exclusive_not_contradictory :
  (∀ d : Distribution, ¬(A_gets_red d ∧ B_gets_red d)) ∧
  (∃ d1 d2 : Distribution, A_gets_red d1 ∧ B_gets_red d2) :=
sorry

end NUMINAMATH_CALUDE_A_B_red_mutually_exclusive_not_contradictory_l2378_237836


namespace NUMINAMATH_CALUDE_total_cash_realized_proof_l2378_237863

/-- Represents a stock with its value and brokerage rate -/
structure Stock where
  value : ℝ
  brokerage_rate : ℝ

/-- Calculates the cash realized for a single stock after brokerage -/
def cash_realized_single (stock : Stock) : ℝ :=
  stock.value * (1 - stock.brokerage_rate)

/-- Calculates the total cash realized for multiple stocks -/
def total_cash_realized (stocks : List Stock) : ℝ :=
  stocks.map cash_realized_single |>.sum

/-- Theorem stating that the total cash realized for the given stocks is 637.818125 -/
theorem total_cash_realized_proof (stockA stockB stockC : Stock)
  (hA : stockA = { value := 120.50, brokerage_rate := 0.0025 })
  (hB : stockB = { value := 210.75, brokerage_rate := 0.005 })
  (hC : stockC = { value := 310.25, brokerage_rate := 0.0075 }) :
  total_cash_realized [stockA, stockB, stockC] = 637.818125 := by
  sorry

end NUMINAMATH_CALUDE_total_cash_realized_proof_l2378_237863


namespace NUMINAMATH_CALUDE_divides_two_pow_plus_one_congruence_l2378_237803

theorem divides_two_pow_plus_one_congruence (p : ℕ) (n : ℤ) 
  (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) 
  (h_divides : n ∣ (2^p + 1) / 3) : 
  n ≡ 1 [ZMOD (2 * p)] := by
sorry

end NUMINAMATH_CALUDE_divides_two_pow_plus_one_congruence_l2378_237803


namespace NUMINAMATH_CALUDE_rice_qualification_condition_l2378_237849

/-- The maximum number of chaff grains allowed in a qualified rice sample -/
def max_chaff_grains : ℕ := 7

/-- The total number of grains in the rice sample -/
def total_grains : ℕ := 235

/-- The maximum allowed percentage of chaff for qualified rice -/
def max_chaff_percentage : ℚ := 3 / 100

/-- Theorem stating the condition for qualified rice -/
theorem rice_qualification_condition (n : ℕ) :
  (n : ℚ) / total_grains ≤ max_chaff_percentage ↔ n ≤ max_chaff_grains :=
by sorry

end NUMINAMATH_CALUDE_rice_qualification_condition_l2378_237849


namespace NUMINAMATH_CALUDE_custom_mul_solution_l2378_237802

/-- Custom multiplication operation -/
def custom_mul (a b : ℕ) : ℕ := 2 * a + b^2

/-- Theorem stating that if a * 3 = 21 under the custom multiplication, then a = 6 -/
theorem custom_mul_solution :
  ∃ a : ℕ, custom_mul a 3 = 21 ∧ a = 6 :=
by sorry

end NUMINAMATH_CALUDE_custom_mul_solution_l2378_237802


namespace NUMINAMATH_CALUDE_floor_product_equals_48_l2378_237856

theorem floor_product_equals_48 (x : ℝ) :
  ⌊x * ⌊x⌋⌋ = 48 ↔ x ∈ Set.Icc 8 (49/6) :=
sorry

end NUMINAMATH_CALUDE_floor_product_equals_48_l2378_237856


namespace NUMINAMATH_CALUDE_storm_rainfall_calculation_l2378_237824

/-- Represents the rainfall during a storm -/
structure StormRainfall where
  first_30min : ℝ
  second_30min : ℝ
  last_hour : ℝ
  average_total : ℝ
  duration : ℝ

/-- Theorem about the rainfall during a specific storm -/
theorem storm_rainfall_calculation (storm : StormRainfall) 
  (h1 : storm.first_30min = 5)
  (h2 : storm.second_30min = storm.first_30min / 2)
  (h3 : storm.duration = 2)
  (h4 : storm.average_total = 4) :
  storm.last_hour = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_storm_rainfall_calculation_l2378_237824


namespace NUMINAMATH_CALUDE_sum_of_three_smallest_solutions_l2378_237839

def is_solution (x : ℝ) : Prop :=
  x > 0 ∧ x - ⌊x⌋ = 1 / (⌊x⌋^2)

def smallest_solutions : Set ℝ :=
  {x | is_solution x ∧ ∀ y, is_solution y → x ≤ y}

theorem sum_of_three_smallest_solutions :
  ∃ (a b c : ℝ), a ∈ smallest_solutions ∧ b ∈ smallest_solutions ∧ c ∈ smallest_solutions ∧
  (∀ x ∈ smallest_solutions, x = a ∨ x = b ∨ x = c) ∧
  a + b + c = 9 + 17/36 :=
sorry

end NUMINAMATH_CALUDE_sum_of_three_smallest_solutions_l2378_237839


namespace NUMINAMATH_CALUDE_x_minus_y_values_l2378_237830

theorem x_minus_y_values (x y : ℝ) 
  (h1 : |x + 1| = 4)
  (h2 : (y + 2)^2 = 4)
  (h3 : x + y ≥ -5) :
  (x - y = -5) ∨ (x - y = 3) ∨ (x - y = 7) :=
by sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l2378_237830


namespace NUMINAMATH_CALUDE_space_diagonal_length_l2378_237875

/-- The length of the space diagonal in a rectangular prism with edge lengths 2, 3, and 4 is √29. -/
theorem space_diagonal_length (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 4) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 29 := by
  sorry


end NUMINAMATH_CALUDE_space_diagonal_length_l2378_237875


namespace NUMINAMATH_CALUDE_range_of_a_l2378_237896

/-- A decreasing function defined on (-∞, 3] -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x > f y) ∧ (∀ x, f x ≤ 3)

theorem range_of_a (f : ℝ → ℝ) (h_f : DecreasingFunction f)
    (h_ineq : ∀ x a : ℝ, f (a^2 - Real.sin x) ≤ f (a + 1 + Real.cos x ^ 2)) :
    ∀ a : ℝ, a ∈ Set.Icc (-Real.sqrt 2) ((1 - Real.sqrt 10) / 2) :=
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2378_237896


namespace NUMINAMATH_CALUDE_circle_properties_l2378_237894

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line1 (x y : ℝ) : Prop := x - 6 * y - 10 = 0
def line2 (x y : ℝ) : Prop := 5 * x - 3 * y = 0

-- Define the given conditions
axiom circle_intersects_line1 : ∃ (c : Circle), line1 4 (-1)
axiom center_on_line2 : ∀ (c : Circle), line2 c.center.1 c.center.2

-- Define the theorem to prove
theorem circle_properties (c : Circle) :
  (∀ (x y : ℝ), (x - 3)^2 + (y - 5)^2 = 37 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
  (∃ (chord : ℝ), chord = 2 * Real.sqrt 3 ∧
    ∀ (l : ℝ → ℝ → Prop),
      (∀ x y, l x y → x = 0 ∨ y = 0) →
      (∃ x₁ y₁ x₂ y₂, l x₁ y₁ ∧ l x₂ y₂ ∧
        (x₁ - c.center.1)^2 + (y₁ - c.center.2)^2 = c.radius^2 ∧
        (x₂ - c.center.1)^2 + (y₂ - c.center.2)^2 = c.radius^2 ∧
        (x₂ - x₁)^2 + (y₂ - y₁)^2 ≤ chord^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_properties_l2378_237894
