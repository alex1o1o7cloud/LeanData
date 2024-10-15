import Mathlib

namespace NUMINAMATH_CALUDE_tank_depth_l3267_326701

/-- Calculates the depth of a tank given its dimensions and plastering costs -/
theorem tank_depth (length width : ℝ) (plaster_cost_per_sqm total_cost : ℝ) : 
  length = 25 → 
  width = 12 → 
  plaster_cost_per_sqm = 0.75 → 
  total_cost = 558 → 
  ∃ depth : ℝ, 
    plaster_cost_per_sqm * (2 * (length * depth) + 2 * (width * depth) + (length * width)) = total_cost ∧ 
    depth = 6 := by
  sorry

end NUMINAMATH_CALUDE_tank_depth_l3267_326701


namespace NUMINAMATH_CALUDE_min_expression_l3267_326796

theorem min_expression (k : ℝ) (x y z t : ℝ) 
  (h1 : k ≥ 0) 
  (h2 : x > 0) (h3 : y > 0) (h4 : z > 0) (h5 : t > 0) 
  (h6 : x + y + z + t = k) : 
  x / (1 + y^2) + y / (1 + x^2) + z / (1 + t^2) + t / (1 + z^2) ≥ 4 * k / (4 + k^2) := by
  sorry

end NUMINAMATH_CALUDE_min_expression_l3267_326796


namespace NUMINAMATH_CALUDE_M_remainder_mod_45_l3267_326710

def M : ℕ := sorry

theorem M_remainder_mod_45 : M % 45 = 15 := by sorry

end NUMINAMATH_CALUDE_M_remainder_mod_45_l3267_326710


namespace NUMINAMATH_CALUDE_mardi_gras_necklaces_mardi_gras_necklaces_proof_l3267_326747

theorem mardi_gras_necklaces : Int → Int → Int → Prop :=
  fun boudreaux rhonda latch =>
    boudreaux = 12 ∧
    rhonda = boudreaux / 2 ∧
    latch = 3 * rhonda - 4 →
    latch = 14

-- The proof is omitted
theorem mardi_gras_necklaces_proof : mardi_gras_necklaces 12 6 14 := by
  sorry

end NUMINAMATH_CALUDE_mardi_gras_necklaces_mardi_gras_necklaces_proof_l3267_326747


namespace NUMINAMATH_CALUDE_candy_distribution_theorem_l3267_326758

/-- The number of ways to distribute candies into boxes. -/
def distribute_candy (candies boxes : ℕ) : ℕ := sorry

/-- The number of ways to distribute candies into boxes with no adjacent empty boxes. -/
def distribute_candy_no_adjacent_empty (candies boxes : ℕ) : ℕ := sorry

/-- Theorem: There are 34 ways to distribute 10 pieces of candy into 5 boxes
    such that no two adjacent boxes are empty. -/
theorem candy_distribution_theorem :
  distribute_candy_no_adjacent_empty 10 5 = 34 := by sorry

end NUMINAMATH_CALUDE_candy_distribution_theorem_l3267_326758


namespace NUMINAMATH_CALUDE_three_equal_differences_exist_l3267_326718

theorem three_equal_differences_exist (a : Fin 19 → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_bound : ∀ i, a i < 91) :
  ∃ i j k l m n, i ≠ j ∧ k ≠ l ∧ m ≠ n ∧
    i ≠ k ∧ i ≠ m ∧ k ≠ m ∧
    a j - a i = a l - a k ∧ a n - a m = a j - a i :=
sorry

end NUMINAMATH_CALUDE_three_equal_differences_exist_l3267_326718


namespace NUMINAMATH_CALUDE_pizza_toppings_l3267_326749

theorem pizza_toppings (total_slices cheese_slices onion_slices : ℕ) 
  (h_total : total_slices = 16)
  (h_cheese : cheese_slices = 9)
  (h_onion : onion_slices = 13)
  (h_at_least_one : cheese_slices + onion_slices ≥ total_slices) :
  ∃ (both_toppings : ℕ), 
    both_toppings = cheese_slices + onion_slices - total_slices ∧
    both_toppings = 6 :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l3267_326749


namespace NUMINAMATH_CALUDE_smallest_percent_increase_between_3_and_4_l3267_326775

def question_values : List ℕ := [100, 300, 600, 900, 1500, 2400]

def percent_increase (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def consecutive_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  List.zip l (List.tail l)

theorem smallest_percent_increase_between_3_and_4 :
  let pairs := consecutive_pairs question_values
  let increases := pairs.map (fun (a, b) => percent_increase a b)
  increases.argmin id = some 2 := by sorry

end NUMINAMATH_CALUDE_smallest_percent_increase_between_3_and_4_l3267_326775


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l3267_326708

theorem smallest_gcd_multiple (m n : ℕ) (h : m > 0 ∧ n > 0) (h_gcd : Nat.gcd m n = 18) :
  Nat.gcd (8 * m) (12 * n) ≥ 72 ∧ ∃ (m₀ n₀ : ℕ), m₀ > 0 ∧ n₀ > 0 ∧ Nat.gcd m₀ n₀ = 18 ∧ Nat.gcd (8 * m₀) (12 * n₀) = 72 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l3267_326708


namespace NUMINAMATH_CALUDE_remainder_sum_l3267_326744

theorem remainder_sum (c d : ℤ) 
  (hc : c % 60 = 53) 
  (hd : d % 42 = 35) : 
  (c + d) % 21 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3267_326744


namespace NUMINAMATH_CALUDE_at_least_one_not_greater_than_neg_one_l3267_326748

theorem at_least_one_not_greater_than_neg_one (a b c d : ℝ) 
  (sum_eq : a + b + c + d = -2)
  (sum_products_eq : a*b + a*c + a*d + b*c + b*d + c*d = 0) :
  min a (min b (min c d)) ≤ -1 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_not_greater_than_neg_one_l3267_326748


namespace NUMINAMATH_CALUDE_Q_divisible_by_three_l3267_326740

def Q (x p q : ℤ) : ℤ := x^3 - x + (p+1)*x + q

theorem Q_divisible_by_three (p q : ℤ) 
  (h1 : 3 ∣ (p + 1)) 
  (h2 : 3 ∣ q) : 
  ∀ x : ℤ, 3 ∣ Q x p q := by
sorry

end NUMINAMATH_CALUDE_Q_divisible_by_three_l3267_326740


namespace NUMINAMATH_CALUDE_date_statistics_order_l3267_326731

def date_counts : List (Nat × Nat) :=
  (List.range 29).map (λ n => (n + 1, 12)) ++
  [(30, 11), (31, 7)]

def total_count : Nat := date_counts.foldl (λ acc (_, count) => acc + count) 0

def sum_of_values : Nat := date_counts.foldl (λ acc (date, count) => acc + date * count) 0

def mean : ℚ := sum_of_values / total_count

def median : ℚ :=
  let mid_point := (total_count + 1) / 2
  16

def median_of_modes : Nat := 15

theorem date_statistics_order :
  (median_of_modes : ℚ) < mean ∧ mean < median := by sorry

end NUMINAMATH_CALUDE_date_statistics_order_l3267_326731


namespace NUMINAMATH_CALUDE_regular_square_pyramid_volume_l3267_326732

/-- The volume of a regular square pyramid with base edge length 2 and side edge length √6 is 8/3. -/
theorem regular_square_pyramid_volume :
  ∀ (base_edge side_edge volume : ℝ),
    base_edge = 2 →
    side_edge = Real.sqrt 6 →
    volume = (1 / 3) * base_edge ^ 2 * Real.sqrt (side_edge ^ 2 - (base_edge ^ 2 / 2)) →
    volume = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_square_pyramid_volume_l3267_326732


namespace NUMINAMATH_CALUDE_percent_more_and_less_equal_l3267_326772

theorem percent_more_and_less_equal (x : ℝ) : x = 138.67 →
  (80 + 0.3 * 80 : ℝ) = (x - 0.25 * x) := by sorry

end NUMINAMATH_CALUDE_percent_more_and_less_equal_l3267_326772


namespace NUMINAMATH_CALUDE_expression_equality_l3267_326715

theorem expression_equality : -2^2 + Real.sqrt 8 - 3 + 1/3 = -20/3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3267_326715


namespace NUMINAMATH_CALUDE_greatest_multiple_of_nine_with_unique_digits_mod_100_l3267_326705

/-- A function that checks if all digits of a natural number are unique -/
def has_unique_digits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 9 with all unique digits -/
def M : ℕ := sorry

theorem greatest_multiple_of_nine_with_unique_digits_mod_100 :
  M % 9 = 0 ∧ has_unique_digits M ∧ (∀ k : ℕ, k % 9 = 0 → has_unique_digits k → k ≤ M) →
  M % 100 = 81 := by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_nine_with_unique_digits_mod_100_l3267_326705


namespace NUMINAMATH_CALUDE_odd_even_sum_reciprocal_l3267_326728

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function g is even if g(-x) = g(x) for all x -/
def IsEven (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

/-- Given f is odd, g is even, and f(x) + g(x) = 1 / (x - 1), prove f(3) = 3/8 -/
theorem odd_even_sum_reciprocal (f g : ℝ → ℝ) 
    (hodd : IsOdd f) (heven : IsEven g) 
    (hsum : ∀ x ≠ 1, f x + g x = 1 / (x - 1)) : 
    f 3 = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_odd_even_sum_reciprocal_l3267_326728


namespace NUMINAMATH_CALUDE_certain_number_proof_l3267_326768

theorem certain_number_proof (h1 : 268 * 74 = 19732) (n : ℝ) (h2 : 2.68 * n = 1.9832) : n = 0.74 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3267_326768


namespace NUMINAMATH_CALUDE_connie_gave_juan_marbles_l3267_326743

/-- The number of marbles Connie gave to Juan -/
def marbles_given (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Proof that Connie gave 183 marbles to Juan -/
theorem connie_gave_juan_marbles : marbles_given 776 593 = 183 := by
  sorry

end NUMINAMATH_CALUDE_connie_gave_juan_marbles_l3267_326743


namespace NUMINAMATH_CALUDE_number_of_men_l3267_326760

theorem number_of_men (M : ℕ) (W : ℝ) : 
  (W / (M * 40) = W / ((M - 5) * 50)) → M = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_of_men_l3267_326760


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3267_326703

theorem fraction_evaluation : 
  (1 - (1/4 + 1/5)) / (1 - 2/3) = 33/20 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3267_326703


namespace NUMINAMATH_CALUDE_remainder_problem_l3267_326729

theorem remainder_problem (n : ℕ) (h1 : (1661 - 10) % n = 0) (h2 : (2045 - 13) % n = 0) (h3 : n = 127) : 
  13 = 2045 % n :=
by sorry

end NUMINAMATH_CALUDE_remainder_problem_l3267_326729


namespace NUMINAMATH_CALUDE_overlapping_circles_common_chord_l3267_326778

theorem overlapping_circles_common_chord 
  (r : ℝ) 
  (h1 : r = 12) 
  (h2 : r > 0) : 
  let d := r -- distance between centers
  let x := Real.sqrt (r^2 - (r/2)^2) -- half-length of common chord
  2 * x = 12 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_overlapping_circles_common_chord_l3267_326778


namespace NUMINAMATH_CALUDE_third_score_proof_l3267_326757

/-- Given three scores with an average of 122, where two scores are 118 and 125, prove the third score is 123. -/
theorem third_score_proof (average : ℝ) (score1 score2 : ℝ) (h_average : average = 122) 
  (h_score1 : score1 = 118) (h_score2 : score2 = 125) : 
  3 * average - (score1 + score2) = 123 := by
  sorry

end NUMINAMATH_CALUDE_third_score_proof_l3267_326757


namespace NUMINAMATH_CALUDE_roses_left_unsold_l3267_326762

theorem roses_left_unsold (price : ℕ) (initial : ℕ) (earned : ℕ) : 
  price = 7 → initial = 9 → earned = 35 → initial - (earned / price) = 4 := by
  sorry

end NUMINAMATH_CALUDE_roses_left_unsold_l3267_326762


namespace NUMINAMATH_CALUDE_transform_f_to_g_l3267_326739

-- Define the original function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := f ((1 - x) / 2) + 1

-- Theorem stating the transformations
theorem transform_f_to_g :
  ∀ x : ℝ,
  -- Reflection across y-axis
  g (-x) = f ((1 + x) / 2) + 1 ∧
  -- Horizontal stretch by factor 2
  g (2 * x) = f ((1 - 2*x) / 2) + 1 ∧
  -- Horizontal shift right by 0.5 units
  g (x - 0.5) = f ((1 - (x - 0.5)) / 2) + 1 ∧
  -- Vertical shift up by 1 unit
  g x = f ((1 - x) / 2) + 1 :=
by sorry

end NUMINAMATH_CALUDE_transform_f_to_g_l3267_326739


namespace NUMINAMATH_CALUDE_sum_and_product_identities_l3267_326756

theorem sum_and_product_identities (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -1) : 
  (a^2 + b^2 = 6) ∧ ((a - b)^2 = 8) := by sorry

end NUMINAMATH_CALUDE_sum_and_product_identities_l3267_326756


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l3267_326738

theorem difference_of_squares_special_case : (500 : ℤ) * 500 - 499 * 501 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l3267_326738


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_391_l3267_326773

theorem greatest_prime_factor_of_391 : ∃ p : ℕ, 
  Nat.Prime p ∧ p ∣ 391 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 391 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_391_l3267_326773


namespace NUMINAMATH_CALUDE_average_value_function_m_range_l3267_326781

/-- A function is an average value function on [a, b] if there exists x₀ ∈ (a, b) such that f(x₀) = (f(b) - f(a)) / (b - a) -/
def IsAverageValueFunction (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₀ : ℝ, a < x₀ ∧ x₀ < b ∧ f x₀ = (f b - f a) / (b - a)

/-- The quadratic function f(x) = x² - mx - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x - 1

theorem average_value_function_m_range :
  ∀ m : ℝ, IsAverageValueFunction (f m) (-1) 1 ↔ 0 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_average_value_function_m_range_l3267_326781


namespace NUMINAMATH_CALUDE_complex_product_equals_369_l3267_326751

theorem complex_product_equals_369 (x : ℂ) : 
  x = Complex.exp (2 * Real.pi * I / 9) →
  (3 * x + x^3) * (3 * x^3 + x^9) * (3 * x^6 + x^18) = 369 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_369_l3267_326751


namespace NUMINAMATH_CALUDE_blake_apples_cost_l3267_326714

/-- The amount Blake spent on apples -/
def apples_cost (total : ℕ) (change : ℕ) (oranges : ℕ) (mangoes : ℕ) : ℕ :=
  total - change - (oranges + mangoes)

/-- Theorem: Blake spent $50 on apples -/
theorem blake_apples_cost :
  apples_cost 300 150 40 60 = 50 := by
  sorry

end NUMINAMATH_CALUDE_blake_apples_cost_l3267_326714


namespace NUMINAMATH_CALUDE_rectangle_formation_count_l3267_326722

theorem rectangle_formation_count (h : ℕ) (v : ℕ) : h = 5 → v = 6 → Nat.choose h 2 * Nat.choose v 2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formation_count_l3267_326722


namespace NUMINAMATH_CALUDE_collinear_probability_in_5x5_grid_l3267_326737

/-- The number of dots in each row and column of the grid -/
def gridSize : ℕ := 5

/-- The total number of dots in the grid -/
def totalDots : ℕ := gridSize * gridSize

/-- The number of dots to be chosen -/
def chosenDots : ℕ := 4

/-- The number of ways to choose 4 collinear dots -/
def collinearWays : ℕ := 2 * gridSize + 2

/-- The total number of ways to choose 4 dots from 25 -/
def totalWays : ℕ := Nat.choose totalDots chosenDots

/-- The probability of choosing 4 collinear dots -/
def collinearProbability : ℚ := collinearWays / totalWays

theorem collinear_probability_in_5x5_grid :
  collinearProbability = 12 / 12650 :=
sorry

end NUMINAMATH_CALUDE_collinear_probability_in_5x5_grid_l3267_326737


namespace NUMINAMATH_CALUDE_circle_in_diamond_l3267_326782

-- Define the sets M and N
def M (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = a^2}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | |p.1 + p.2| + |p.1 - p.2| ≤ 2}

-- State the theorem
theorem circle_in_diamond (a : ℝ) (h : a > 0) :
  M a ⊆ N ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_circle_in_diamond_l3267_326782


namespace NUMINAMATH_CALUDE_orthocenter_proof_l3267_326726

/-- A point in a 2D plane -/
structure Point := (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral := (A B C D : Point)

/-- A triangle defined by three points -/
structure Triangle := (D P Q : Point)

/-- Checks if a quadrilateral is a rhombus -/
def is_rhombus (quad : Quadrilateral) : Prop := sorry

/-- Checks if a quadrilateral is a parallelogram -/
def is_parallelogram (quad : Quadrilateral) : Prop := sorry

/-- Checks if a point lies inside a quadrilateral -/
def point_inside (P : Point) (quad : Quadrilateral) : Prop := sorry

/-- Checks if two line segments have equal length -/
def segments_equal (A B C D : Point) : Prop := sorry

/-- Checks if a point is the orthocenter of a triangle -/
def is_orthocenter (P : Point) (tri : Triangle) : Prop := sorry

theorem orthocenter_proof (A B C D P Q : Point) :
  let ABCD := Quadrilateral.mk A B C D
  let APQC := Quadrilateral.mk A P Q C
  let DPQ := Triangle.mk D P Q
  is_rhombus ABCD →
  is_parallelogram APQC →
  point_inside B APQC →
  segments_equal A P A B →
  is_orthocenter B DPQ := by
  sorry

end NUMINAMATH_CALUDE_orthocenter_proof_l3267_326726


namespace NUMINAMATH_CALUDE_spinner_probability_l3267_326713

theorem spinner_probability (pA pB pC pD : ℚ) : 
  pA = 1/4 → pB = 1/3 → pD = 1/6 → pA + pB + pC + pD = 1 → pC = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3267_326713


namespace NUMINAMATH_CALUDE_blueberry_picking_total_l3267_326733

/-- The total number of pints of blueberries picked by Annie, Kathryn, and Ben -/
def total_pints (annie kathryn ben : ℕ) : ℕ := annie + kathryn + ben

/-- Theorem stating the total number of pints picked given the conditions -/
theorem blueberry_picking_total :
  ∀ (annie kathryn ben : ℕ),
  annie = 8 →
  kathryn = annie + 2 →
  ben = kathryn - 3 →
  total_pints annie kathryn ben = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_blueberry_picking_total_l3267_326733


namespace NUMINAMATH_CALUDE_quadratic_equations_distinct_roots_l3267_326783

theorem quadratic_equations_distinct_roots (n : ℕ) (a b : Fin n → ℝ) 
  (h_n : n ≥ 2)
  (h_distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ b j) :
  ¬ (∀ i : Fin n, ∃ j : Fin n, (a i)^2 - a j * (a i) + b j = 0 ∨ (b i)^2 - a j * (b i) + b j = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_distinct_roots_l3267_326783


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3267_326702

theorem inequality_solution_set (a : ℝ) :
  let solution_set := {x : ℝ | x^2 - (a^2 + a)*x + a^3 < 0}
  (a = 0 ∨ a = 1 → solution_set = ∅) ∧
  (0 < a ∧ a < 1 → solution_set = {x : ℝ | a^2 < x ∧ x < a}) ∧
  ((a < 0 ∨ a > 1) → solution_set = {x : ℝ | a < x ∧ x < a^2}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3267_326702


namespace NUMINAMATH_CALUDE_equation_solution_l3267_326736

theorem equation_solution : 
  ∃! x : ℚ, x ≠ 3 ∧ (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ∧ x = -7/6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3267_326736


namespace NUMINAMATH_CALUDE_first_class_average_mark_l3267_326727

theorem first_class_average_mark (x : ℝ) : 
  (25 * x + 30 * 60) / 55 = 50.90909090909091 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_first_class_average_mark_l3267_326727


namespace NUMINAMATH_CALUDE_jackson_running_distance_l3267_326709

/-- Calculate the final running distance after doubling the initial distance for a given number of weeks -/
def finalDistance (initialDistance : ℕ) (weeks : ℕ) : ℕ :=
  initialDistance * (2 ^ weeks)

/-- Theorem stating that starting with 3 miles and doubling for 4 weeks results in 24 miles -/
theorem jackson_running_distance : finalDistance 3 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_jackson_running_distance_l3267_326709


namespace NUMINAMATH_CALUDE_dataset_mode_l3267_326777

def dataset : List Nat := [3, 1, 3, 0, 3, 2, 1, 2]

def mode (l : List Nat) : Nat :=
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

theorem dataset_mode :
  mode dataset = 3 := by
  sorry

end NUMINAMATH_CALUDE_dataset_mode_l3267_326777


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l3267_326721

theorem boys_to_girls_ratio : 
  ∀ (boys girls : ℕ), 
    boys = 40 →
    girls = boys + 64 →
    (boys : ℚ) / (girls : ℚ) = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l3267_326721


namespace NUMINAMATH_CALUDE_monster_feast_l3267_326784

theorem monster_feast (sequence : Fin 3 → ℕ) 
  (double_next : ∀ i : Fin 2, sequence (Fin.succ i) = 2 * sequence i)
  (total_consumed : sequence 0 + sequence 1 + sequence 2 = 847) :
  sequence 0 = 121 := by
sorry

end NUMINAMATH_CALUDE_monster_feast_l3267_326784


namespace NUMINAMATH_CALUDE_zoo_fraction_l3267_326741

/-- Given a zoo with various animals, prove that the fraction of elephants
    to the sum of parrots and snakes is 1/2. -/
theorem zoo_fraction (parrots snakes monkeys elephants zebras : ℕ) 
  (h1 : parrots = 8)
  (h2 : snakes = 3 * parrots)
  (h3 : monkeys = 2 * snakes)
  (h4 : ∃ f : ℚ, elephants = f * (parrots + snakes))
  (h5 : zebras + 3 = elephants)
  (h6 : monkeys - zebras = 35) :
  ∃ f : ℚ, elephants = f * (parrots + snakes) ∧ f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_zoo_fraction_l3267_326741


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l3267_326730

theorem angle_sum_theorem (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 1/3) (h4 : Real.cos β = 3/5) :
  α + 2*β = π - Real.arctan (13/9) := by sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l3267_326730


namespace NUMINAMATH_CALUDE_cube_root_unity_product_l3267_326776

theorem cube_root_unity_product (ω : ℂ) : 
  ω ≠ 1 → ω^3 = 1 → (1 - ω + ω^2) * (1 + ω - ω^2) = 4 := by sorry

end NUMINAMATH_CALUDE_cube_root_unity_product_l3267_326776


namespace NUMINAMATH_CALUDE_seventy_fifth_term_is_298_l3267_326746

def arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

theorem seventy_fifth_term_is_298 : arithmetic_sequence 2 4 75 = 298 := by
  sorry

end NUMINAMATH_CALUDE_seventy_fifth_term_is_298_l3267_326746


namespace NUMINAMATH_CALUDE_train_speed_l3267_326779

theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 500) (h2 : crossing_time = 10) :
  train_length / crossing_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3267_326779


namespace NUMINAMATH_CALUDE_budget_allocation_l3267_326795

def total_budget : ℝ := 40000000

def policing_percentage : ℝ := 0.35
def education_percentage : ℝ := 0.25
def healthcare_percentage : ℝ := 0.15

def remaining_budget : ℝ := total_budget * (1 - (policing_percentage + education_percentage + healthcare_percentage))

theorem budget_allocation :
  remaining_budget = 10000000 := by sorry

end NUMINAMATH_CALUDE_budget_allocation_l3267_326795


namespace NUMINAMATH_CALUDE_min_values_ab_l3267_326723

theorem min_values_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ 1/x + 2/y < 9) = False ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ x^2 + y^2 < 1/5) = False :=
by sorry

end NUMINAMATH_CALUDE_min_values_ab_l3267_326723


namespace NUMINAMATH_CALUDE_nobel_prize_laureates_l3267_326734

theorem nobel_prize_laureates (total_scientists : ℕ) 
                               (wolf_prize : ℕ) 
                               (wolf_and_nobel : ℕ) 
                               (non_wolf_nobel_diff : ℕ) : 
  total_scientists = 50 → 
  wolf_prize = 31 → 
  wolf_and_nobel = 12 → 
  non_wolf_nobel_diff = 3 → 
  (total_scientists - wolf_prize + wolf_and_nobel : ℕ) = 23 := by
  sorry

end NUMINAMATH_CALUDE_nobel_prize_laureates_l3267_326734


namespace NUMINAMATH_CALUDE_eu_countries_2012_is_set_l3267_326720

/-- A type representing countries -/
def Country : Type := String

/-- A predicate that determines if a country was in the EU in 2012 -/
def WasEUMemberIn2012 (c : Country) : Prop := sorry

/-- The set of all EU countries in 2012 -/
def EUCountries2012 : Set Country :=
  {c : Country | WasEUMemberIn2012 c}

/-- A property that determines if a collection can form a set -/
def CanFormSet (S : Set α) : Prop :=
  ∀ x, x ∈ S → (∃ p : Prop, p ↔ x ∈ S)

theorem eu_countries_2012_is_set :
  CanFormSet EUCountries2012 :=
sorry

end NUMINAMATH_CALUDE_eu_countries_2012_is_set_l3267_326720


namespace NUMINAMATH_CALUDE_layla_babysitting_earnings_l3267_326764

theorem layla_babysitting_earnings :
  let donaldson_rate : ℕ := 15
  let merck_rate : ℕ := 18
  let hille_rate : ℕ := 20
  let johnson_rate : ℕ := 22
  let ramos_rate : ℕ := 25
  let donaldson_hours : ℕ := 7
  let merck_hours : ℕ := 6
  let hille_hours : ℕ := 3
  let johnson_hours : ℕ := 4
  let ramos_hours : ℕ := 2
  donaldson_rate * donaldson_hours +
  merck_rate * merck_hours +
  hille_rate * hille_hours +
  johnson_rate * johnson_hours +
  ramos_rate * ramos_hours = 411 :=
by sorry

end NUMINAMATH_CALUDE_layla_babysitting_earnings_l3267_326764


namespace NUMINAMATH_CALUDE_prime_sum_and_seven_sum_squares_l3267_326769

theorem prime_sum_and_seven_sum_squares (p q : ℕ) : 
  Prime p → Prime q → 
  ∃ x y : ℕ, x^2 = p + q ∧ y^2 = p + 7*q → 
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_and_seven_sum_squares_l3267_326769


namespace NUMINAMATH_CALUDE_cubic_equation_real_root_l3267_326724

theorem cubic_equation_real_root (k : ℝ) (hk : k ≠ 0) :
  ∃ x : ℝ, x^3 + k*x + k^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_real_root_l3267_326724


namespace NUMINAMATH_CALUDE_mangoes_harvested_l3267_326706

theorem mangoes_harvested (neighbors : ℕ) (mangoes_per_neighbor : ℕ) 
  (h1 : neighbors = 8)
  (h2 : mangoes_per_neighbor = 35)
  (h3 : ∃ (total : ℕ), total / 2 = neighbors * mangoes_per_neighbor) :
  ∃ (total : ℕ), total = 560 := by
  sorry

end NUMINAMATH_CALUDE_mangoes_harvested_l3267_326706


namespace NUMINAMATH_CALUDE_student_ratio_l3267_326759

theorem student_ratio (total : ℕ) (on_bleachers : ℕ) 
  (h1 : total = 26) (h2 : on_bleachers = 4) : 
  (total - on_bleachers : ℚ) / total = 11 / 13 := by
  sorry

end NUMINAMATH_CALUDE_student_ratio_l3267_326759


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3267_326711

-- Problem 1
theorem problem_1 (x y : ℝ) (h1 : x * y = 5) (h2 : x + y = 6) :
  (x - y)^2 = 16 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h : (2016 - a) * (2017 - a) = 5) :
  (a - 2016)^2 + (2017 - a)^2 = 11 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3267_326711


namespace NUMINAMATH_CALUDE_subtraction_problem_l3267_326793

theorem subtraction_problem :
  572 - 275 = 297 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3267_326793


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l3267_326799

/-- Given a square with two pairs of identical isosceles triangles cut off, leaving a rectangle,
    if the total area cut off is 250 m² and one side of the rectangle is 1.5 times the length of the other,
    then the length of the longer side of the rectangle is 7.5√5 meters. -/
theorem rectangle_longer_side (x y : ℝ) : 
  x^2 + y^2 = 250 →  -- Total area cut off
  x = y →            -- Isosceles triangles condition
  1.5 * y = max x (1.5 * y) →  -- One side is 1.5 times the other
  max x (1.5 * y) = 7.5 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l3267_326799


namespace NUMINAMATH_CALUDE_abs_even_and_increasing_l3267_326774

-- Define the function
def f (x : ℝ) : ℝ := |x|

-- State the theorem
theorem abs_even_and_increasing : 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_abs_even_and_increasing_l3267_326774


namespace NUMINAMATH_CALUDE_alices_money_is_64_dollars_l3267_326704

/-- Represents the value of Alice's money after exchanging quarters for nickels -/
def alices_money_value (num_quarters : ℕ) (iron_nickel_percentage : ℚ) 
  (iron_nickel_value : ℚ) (regular_nickel_value : ℚ) : ℚ :=
  let total_cents := num_quarters * 25
  let total_nickels := total_cents / 5
  let iron_nickels := iron_nickel_percentage * total_nickels
  let regular_nickels := total_nickels - iron_nickels
  iron_nickels * iron_nickel_value + regular_nickels * regular_nickel_value

/-- Theorem stating that Alice's money value after exchange is $64 -/
theorem alices_money_is_64_dollars :
  alices_money_value 20 (1/5) 300 (5/100) = 6400/100 := by
  sorry

end NUMINAMATH_CALUDE_alices_money_is_64_dollars_l3267_326704


namespace NUMINAMATH_CALUDE_special_function_properties_l3267_326798

/-- An increasing function f defined on (-1, +∞) with the property f(xy) = f(x) + f(y) -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > -1 ∧ y > -1 → f (x * y) = f x + f y) ∧
  (∀ x y, x > -1 ∧ y > -1 ∧ x < y → f x < f y)

theorem special_function_properties
    (f : ℝ → ℝ)
    (hf : SpecialFunction f)
    (h3 : f 3 = 1) :
  (f 9 = 2) ∧
  (∀ a, a > -1 → (f a > f (a - 1) + 2 ↔ 0 < a ∧ a < 9/8)) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l3267_326798


namespace NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_15_l3267_326750

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2010 ∧ digit_sum year = 15

theorem first_year_after_2010_with_digit_sum_15 :
  ∀ year : ℕ, is_valid_year year → year ≥ 2049 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_15_l3267_326750


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l3267_326735

/-- Given an angle θ formed by the positive x-axis and a line passing through 
    the origin and the point (-3,1), prove that cos(2θ) = 4/5 -/
theorem cos_double_angle_special_case : 
  ∀ θ : Real, 
  (∃ (x y : Real), x = -3 ∧ y = 1 ∧ x = Real.cos θ * Real.sqrt (x^2 + y^2) ∧ 
                    y = Real.sin θ * Real.sqrt (x^2 + y^2)) → 
  Real.cos (2 * θ) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l3267_326735


namespace NUMINAMATH_CALUDE_equation_system_proof_l3267_326792

theorem equation_system_proof (x y m : ℝ) 
  (eq1 : x + m = 4) 
  (eq2 : y - 3 = m) : 
  x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_proof_l3267_326792


namespace NUMINAMATH_CALUDE_picnic_men_count_l3267_326770

/-- Represents the number of people at a picnic -/
structure PicnicAttendance where
  total : Nat
  men : Nat
  women : Nat
  adults : Nat
  children : Nat

/-- Defines the conditions for a valid picnic attendance -/
def ValidPicnicAttendance (p : PicnicAttendance) : Prop :=
  p.total = 200 ∧
  p.men = p.women + 20 ∧
  p.adults = p.children + 20 ∧
  p.total = p.men + p.women + p.children ∧
  p.adults = p.men + p.women

theorem picnic_men_count (p : PicnicAttendance) (h : ValidPicnicAttendance p) : p.men = 65 := by
  sorry

end NUMINAMATH_CALUDE_picnic_men_count_l3267_326770


namespace NUMINAMATH_CALUDE_train_passing_time_l3267_326794

/-- The length of the high-speed train in meters -/
def high_speed_train_length : ℝ := 400

/-- The length of the regular train in meters -/
def regular_train_length : ℝ := 600

/-- The time in seconds it takes for a passenger on the high-speed train to see the regular train pass -/
def high_speed_observation_time : ℝ := 3

/-- The time in seconds it takes for a passenger on the regular train to see the high-speed train pass -/
def regular_observation_time : ℝ := 2

theorem train_passing_time :
  (regular_train_length / high_speed_observation_time) * regular_observation_time = high_speed_train_length :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l3267_326794


namespace NUMINAMATH_CALUDE_girls_in_class_l3267_326752

/-- Represents the number of girls in a class given the total number of students and the ratio of girls to boys. -/
def number_of_girls (total : ℕ) (girl_ratio : ℕ) (boy_ratio : ℕ) : ℕ :=
  (total * girl_ratio) / (girl_ratio + boy_ratio)

/-- Theorem stating that in a class of 63 students with a girl-to-boy ratio of 4:3, there are 36 girls. -/
theorem girls_in_class : number_of_girls 63 4 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_class_l3267_326752


namespace NUMINAMATH_CALUDE_distance_in_one_hour_l3267_326797

/-- The number of seconds in one hour -/
def seconds_per_hour : ℕ := 3600

/-- The speed of the object in feet per second -/
def speed : ℕ := 3

/-- The distance traveled by an object moving at a constant speed for a given time -/
def distance_traveled (speed : ℕ) (time : ℕ) : ℕ := speed * time

/-- Theorem: An object traveling at 3 feet per second will cover 10800 feet in one hour -/
theorem distance_in_one_hour :
  distance_traveled speed seconds_per_hour = 10800 := by
  sorry

end NUMINAMATH_CALUDE_distance_in_one_hour_l3267_326797


namespace NUMINAMATH_CALUDE_university_theater_ticket_sales_l3267_326788

theorem university_theater_ticket_sales 
  (total_tickets : ℕ) 
  (adult_price senior_price : ℕ) 
  (total_receipts : ℕ) 
  (h1 : total_tickets = 510)
  (h2 : adult_price = 21)
  (h3 : senior_price = 15)
  (h4 : total_receipts = 8748) :
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 327 :=
by sorry

end NUMINAMATH_CALUDE_university_theater_ticket_sales_l3267_326788


namespace NUMINAMATH_CALUDE_sin_plus_sin_sqrt2_not_periodic_l3267_326707

/-- The function x ↦ sin x + sin (√2 x) is not periodic -/
theorem sin_plus_sin_sqrt2_not_periodic :
  ¬ ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, Real.sin x + Real.sin (Real.sqrt 2 * x) = Real.sin (x + p) + Real.sin (Real.sqrt 2 * (x + p)) := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_sin_sqrt2_not_periodic_l3267_326707


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3267_326742

theorem smallest_integer_satisfying_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 13*m + 22 ≤ 0 → n ≤ m) ∧ n^2 - 13*n + 22 ≤ 0 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3267_326742


namespace NUMINAMATH_CALUDE_benny_work_hours_l3267_326700

/-- Given a person who works a fixed number of hours per day for a certain number of days,
    calculate the total number of hours worked. -/
def totalHoursWorked (hoursPerDay : ℕ) (numberOfDays : ℕ) : ℕ :=
  hoursPerDay * numberOfDays

/-- Theorem stating that working 3 hours per day for 6 days results in 18 total hours worked. -/
theorem benny_work_hours :
  totalHoursWorked 3 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_benny_work_hours_l3267_326700


namespace NUMINAMATH_CALUDE_sphere_triangle_distance_is_four_l3267_326719

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Represents a triangle with given side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Calculates the distance between the center of a sphere and the plane of a triangle tangent to it -/
def sphereTriangleDistance (s : Sphere) (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating the distance between the sphere's center and the triangle's plane -/
theorem sphere_triangle_distance_is_four :
  ∀ (s : Sphere) (t : Triangle),
    s.radius = 8 ∧
    t.side1 = 13 ∧ t.side2 = 14 ∧ t.side3 = 15 →
    sphereTriangleDistance s t = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_triangle_distance_is_four_l3267_326719


namespace NUMINAMATH_CALUDE_horse_distance_in_day_l3267_326716

/-- The distance a horse can run in one day -/
def horse_distance (speed : ℝ) (hours_per_day : ℝ) : ℝ :=
  speed * hours_per_day

/-- Theorem: A horse running at 10 miles/hour for 24 hours covers 240 miles -/
theorem horse_distance_in_day :
  horse_distance 10 24 = 240 := by
  sorry

end NUMINAMATH_CALUDE_horse_distance_in_day_l3267_326716


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3267_326780

def A : Set ℝ := {x | -1 < x ∧ x ≤ 4}
def B : Set ℝ := {x | -3 ≤ x ∧ x < 1}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -3 ≤ x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3267_326780


namespace NUMINAMATH_CALUDE_polynomial_difference_theorem_l3267_326763

/-- Given two polynomials that differ in terms of x^2 and y^2, 
    prove the values of m and n and the result of a specific expression. -/
theorem polynomial_difference_theorem (m n : ℝ) : 
  (∀ x y : ℝ, 2 * (m * x^2 - 2 * y^2) - (x - 2 * y) - (x - n * y^2 - 2 * x^2) = 0) →
  m = -1 ∧ n = 4 ∧ (4 * m^2 * n - 3 * m * n^2) - 2 * (m^2 * n + m * n^2) = -72 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_difference_theorem_l3267_326763


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3267_326789

theorem inequality_equivalence (x : ℝ) :
  (2 * x + 3) / (x + 3) > (4 * x + 5) / (3 * x + 8) ↔ 
  x < -3 ∨ x > -3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3267_326789


namespace NUMINAMATH_CALUDE_regina_farm_sale_price_l3267_326761

/-- Calculates the total sale price of animals on Regina's farm -/
def total_sale_price (num_cows : ℕ) (cow_price pig_price goat_price chicken_price rabbit_price : ℕ) : ℕ :=
  let num_pigs := 4 * num_cows
  let num_goats := num_pigs / 2
  let num_chickens := 2 * num_cows
  let num_rabbits := 30
  num_cows * cow_price + num_pigs * pig_price + num_goats * goat_price + 
  num_chickens * chicken_price + num_rabbits * rabbit_price

/-- Theorem stating that the total sale price of all animals on Regina's farm is $74,750 -/
theorem regina_farm_sale_price :
  total_sale_price 20 800 400 600 50 25 = 74750 := by
  sorry

end NUMINAMATH_CALUDE_regina_farm_sale_price_l3267_326761


namespace NUMINAMATH_CALUDE_simplify_polynomial_l3267_326771

theorem simplify_polynomial (r : ℝ) : (2 * r^2 + 5 * r - 3) - (r^2 + 4 * r - 6) = r^2 + r + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l3267_326771


namespace NUMINAMATH_CALUDE_acute_triangle_with_largest_five_times_smallest_l3267_326787

theorem acute_triangle_with_largest_five_times_smallest (α β γ : ℕ) : 
  α > 0 ∧ β > 0 ∧ γ > 0 →  -- All angles are positive
  α + β + γ = 180 →  -- Sum of angles in a triangle
  α ≤ 89 ∧ β ≤ 89 ∧ γ ≤ 89 →  -- Acute triangle condition
  α ≥ β ∧ β ≥ γ →  -- Ordering of angles
  α = 5 * γ →  -- Largest angle is five times the smallest
  (α = 85 ∧ β = 78 ∧ γ = 17) := by
  sorry

#check acute_triangle_with_largest_five_times_smallest

end NUMINAMATH_CALUDE_acute_triangle_with_largest_five_times_smallest_l3267_326787


namespace NUMINAMATH_CALUDE_previous_year_profit_percentage_l3267_326755

/-- Given a company's financial data over two years, calculate the profit percentage in the previous year. -/
theorem previous_year_profit_percentage
  (R : ℝ)  -- Revenues in the previous year
  (P : ℝ)  -- Profits in the previous year
  (h1 : 0.8 * R = R - 0.2 * R)  -- Revenues fell by 20% in 2009
  (h2 : 0.09 * (0.8 * R) = 0.072 * R)  -- Profits were 9% of revenues in 2009
  (h3 : 0.072 * R = 0.72 * P)  -- Profits in 2009 were 72% of previous year's profits
  : P / R = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_previous_year_profit_percentage_l3267_326755


namespace NUMINAMATH_CALUDE_complex_equality_implies_a_equals_three_l3267_326785

theorem complex_equality_implies_a_equals_three (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (2 - Complex.I)
  (z.re = z.im) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_a_equals_three_l3267_326785


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3267_326791

theorem sin_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = Real.sqrt 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3267_326791


namespace NUMINAMATH_CALUDE_inequality_bound_l3267_326767

theorem inequality_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (a / x + b / y) > M) →
  M < a + b + 2 * Real.sqrt (a * b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_bound_l3267_326767


namespace NUMINAMATH_CALUDE_max_area_rectangle_l3267_326790

theorem max_area_rectangle (perimeter : ℕ) (h : perimeter = 148) :
  ∃ (length width : ℕ),
    2 * (length + width) = perimeter ∧
    ∀ (l w : ℕ), 2 * (l + w) = perimeter → l * w ≤ length * width ∧
    length * width = 1369 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l3267_326790


namespace NUMINAMATH_CALUDE_balls_after_500_steps_l3267_326745

/-- Represents the state of boxes after a certain number of steps -/
def BoxState := Nat

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : Nat) : List Nat :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List Nat) : Nat :=
  sorry

/-- Simulates the ball placement process for a given number of steps -/
def simulateSteps (steps : Nat) : BoxState :=
  sorry

theorem balls_after_500_steps :
  simulateSteps 500 = sumDigits (toBase4 500) :=
sorry

end NUMINAMATH_CALUDE_balls_after_500_steps_l3267_326745


namespace NUMINAMATH_CALUDE_division_problem_l3267_326753

theorem division_problem (x y z : ℕ) : 
  x > 0 → 
  x = 7 * y + 3 → 
  2 * x = 3 * y * z + 2 → 
  11 * y - x = 1 → 
  z = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3267_326753


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l3267_326712

/-- The area of the region in a square of side length 5 bounded by lines from (0,0) to (2.5,5) and from (5,5) to (0,2.5) is half the area of the square. -/
theorem shaded_area_ratio (square_side : ℝ) (h : square_side = 5) : 
  let shaded_area := (1/2 * 2.5 * 2.5) + (2.5 * 2.5) + (1/2 * 2.5 * 2.5)
  shaded_area / (square_side ^ 2) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l3267_326712


namespace NUMINAMATH_CALUDE_abcdef_hex_bits_l3267_326754

def hex_to_decimal (h : String) : ℕ := 
  match h with
  | "A" => 10
  | "B" => 11
  | "C" => 12
  | "D" => 13
  | "E" => 14
  | "F" => 15
  | _ => 0  -- This case should never be reached for valid hex digits

theorem abcdef_hex_bits : 
  let decimal : ℕ := 
    (hex_to_decimal "A") * (16^5) +
    (hex_to_decimal "B") * (16^4) +
    (hex_to_decimal "C") * (16^3) +
    (hex_to_decimal "D") * (16^2) +
    (hex_to_decimal "E") * (16^1) +
    (hex_to_decimal "F")
  ∃ n : ℕ, 2^n ≤ decimal ∧ decimal < 2^(n+1) ∧ n + 1 = 24 :=
by sorry

end NUMINAMATH_CALUDE_abcdef_hex_bits_l3267_326754


namespace NUMINAMATH_CALUDE_intersected_cubes_count_l3267_326725

/-- Represents a cube composed of unit cubes -/
structure LargeCube where
  side_length : ℕ
  total_cubes : ℕ
  h_total : total_cubes = side_length ^ 3

/-- Represents a plane intersecting the cube -/
structure IntersectingPlane where
  perpendicular_to_diagonal : Bool
  bisects_diagonal : Bool

/-- Counts the number of unit cubes intersected by the plane -/
def count_intersected_cubes (cube : LargeCube) (plane : IntersectingPlane) : ℕ :=
  sorry

/-- Theorem: A plane perpendicular to and bisecting an internal diagonal 
    of a 4x4x4 cube intersects exactly 32 unit cubes -/
theorem intersected_cubes_count 
  (cube : LargeCube) 
  (plane : IntersectingPlane) 
  (h_side : cube.side_length = 4)
  (h_perp : plane.perpendicular_to_diagonal = true)
  (h_bisect : plane.bisects_diagonal = true) : 
  count_intersected_cubes cube plane = 32 := by
  sorry

end NUMINAMATH_CALUDE_intersected_cubes_count_l3267_326725


namespace NUMINAMATH_CALUDE_sum_in_B_l3267_326717

-- Define set A
def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

-- Define set B
def B : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}

-- Define set C (although not used in the theorem, it's part of the original problem)
def C : Set ℤ := {x | ∃ k : ℤ, x = 4 * k + 1}

-- Theorem statement
theorem sum_in_B (a b : ℤ) (ha : a ∈ A) (hb : b ∈ B) : a + b ∈ B := by
  sorry

end NUMINAMATH_CALUDE_sum_in_B_l3267_326717


namespace NUMINAMATH_CALUDE_rhombus_area_theorem_l3267_326786

/-- Represents a rhombus with side length and diagonal -/
structure Rhombus where
  side_length : ℝ
  diagonal1 : ℝ

/-- Calculates the area of a rhombus given its side length and one diagonal -/
def rhombus_area (r : Rhombus) : ℝ :=
  sorry

theorem rhombus_area_theorem (r : Rhombus) :
  r.side_length = 2 * Real.sqrt 5 →
  r.diagonal1 = 4 →
  rhombus_area r = 16 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_theorem_l3267_326786


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_neg_one_l3267_326765

open Set

theorem intersection_nonempty_implies_a_greater_than_neg_one (a : ℝ) :
  let M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
  let N : Set ℝ := {y | y < a}
  (M ∩ N).Nonempty → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_greater_than_neg_one_l3267_326765


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_l3267_326766

theorem floor_plus_self_unique (r : ℝ) : ⌊r⌋ + r = 15.75 ↔ r = 7.25 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_l3267_326766
