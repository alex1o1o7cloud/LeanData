import Mathlib

namespace NUMINAMATH_CALUDE_absolute_value_equals_opposite_implies_nonpositive_l1561_156156

theorem absolute_value_equals_opposite_implies_nonpositive (a : ℝ) :
  (abs a = -a) → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_opposite_implies_nonpositive_l1561_156156


namespace NUMINAMATH_CALUDE_number_of_pupils_l1561_156180

/-- Given a program with parents and pupils, calculate the number of pupils -/
theorem number_of_pupils (total_people parents : ℕ) (h1 : parents = 105) (h2 : total_people = 803) :
  total_people - parents = 698 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pupils_l1561_156180


namespace NUMINAMATH_CALUDE_area_of_bounded_region_l1561_156187

/-- The area of the region bounded by x = 2, y = 2, and the coordinate axes is 4 square units. -/
theorem area_of_bounded_region : 
  let region := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}
  ∃ (A : Set (ℝ × ℝ)), A = region ∧ MeasureTheory.volume A = 4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_bounded_region_l1561_156187


namespace NUMINAMATH_CALUDE_rice_pricing_problem_l1561_156192

/-- Represents the linear relationship between price and quantity sold --/
def quantity_sold (x : ℝ) : ℝ := -50 * x + 1200

/-- Represents the profit function --/
def profit (x : ℝ) : ℝ := (x - 4) * (quantity_sold x)

/-- Theorem stating the main results of the problem --/
theorem rice_pricing_problem 
  (x : ℝ) 
  (h1 : 4 ≤ x ∧ x ≤ 7) :
  (∃ x, profit x = 1800 ∧ x = 6) ∧
  (∀ y, 4 ≤ y ∧ y ≤ 7 → profit y ≤ profit 7) ∧
  profit 7 = 2550 := by
  sorry


end NUMINAMATH_CALUDE_rice_pricing_problem_l1561_156192


namespace NUMINAMATH_CALUDE_valid_pairs_l1561_156158

def is_valid_pair (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧
  (∃ k : ℤ, (k : ℚ) = (a^2 + b : ℚ) / (b^2 - a : ℚ)) ∧
  (∃ m : ℤ, (m : ℚ) = (b^2 + a : ℚ) / (a^2 - b : ℚ))

theorem valid_pairs :
  ∀ a b : ℕ, is_valid_pair a b ↔
    ((a = 2 ∧ b = 2) ∨
     (a = 3 ∧ b = 3) ∨
     (a = 1 ∧ b = 2) ∨
     (a = 2 ∧ b = 1) ∨
     (a = 2 ∧ b = 3) ∨
     (a = 3 ∧ b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l1561_156158


namespace NUMINAMATH_CALUDE_line_through_ellipse_vertex_l1561_156184

/-- The value of 'a' when a line passes through the right vertex of an ellipse --/
theorem line_through_ellipse_vertex (t θ : ℝ) (a : ℝ) : 
  (∀ t, ∃ x y, x = t ∧ y = t - a) →  -- Line equation
  (∀ θ, ∃ x y, x = 3 * Real.cos θ ∧ y = 2 * Real.sin θ) →  -- Ellipse equation
  (∃ t, t = 3 ∧ t - a = 0) →  -- Line passes through right vertex (3, 0)
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_line_through_ellipse_vertex_l1561_156184


namespace NUMINAMATH_CALUDE_paving_cost_l1561_156100

/-- The cost of paving a rectangular floor given its dimensions and the rate per square meter. -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 600) :
  length * width * rate = 12375 := by sorry

end NUMINAMATH_CALUDE_paving_cost_l1561_156100


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1561_156189

theorem matrix_equation_solution :
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]
  N ^ 2 - 3 • N + 2 • N = !![6, 12; 3, 6] := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1561_156189


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l1561_156193

/-- A complex number is pure imaginary if its real part is zero. -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0

/-- The problem statement -/
theorem complex_product_pure_imaginary (a : ℝ) :
  IsPureImaginary ((a : ℂ) + Complex.I * (2 - Complex.I)) → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l1561_156193


namespace NUMINAMATH_CALUDE_problem_solution_l1561_156127

theorem problem_solution : 
  let left_sum := 5 + 6 + 7 + 8 + 9
  let right_sum := 2005 + 2006 + 2007 + 2008 + 2009
  ∀ N : ℝ, (left_sum / 5 : ℝ) = (right_sum / N : ℝ) → N = 1433 :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1561_156127


namespace NUMINAMATH_CALUDE_median_to_mean_l1561_156149

theorem median_to_mean (m : ℝ) : 
  let s : Finset ℝ := {m, m + 4, m + 7, m + 10, m + 16}
  m + 7 = 12 →
  (s.sum id) / s.card = 12.4 := by
sorry

end NUMINAMATH_CALUDE_median_to_mean_l1561_156149


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1561_156185

theorem polynomial_evaluation :
  ∀ x : ℝ, x > 0 → x^2 - 3*x - 9 = 0 →
  x^4 - 3*x^3 - 9*x^2 + 27*x - 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1561_156185


namespace NUMINAMATH_CALUDE_village_income_growth_and_prediction_l1561_156176

/-- Represents the annual average growth rate calculation and prediction for a village's per capita income. -/
theorem village_income_growth_and_prediction 
  (initial_income : ℝ) 
  (final_income : ℝ) 
  (years : ℕ) 
  (growth_rate : ℝ) 
  (predicted_income : ℝ)
  (h1 : initial_income = 20000)
  (h2 : final_income = 24200)
  (h3 : years = 2) :
  (final_income = initial_income * (1 + growth_rate) ^ years ∧ 
   growth_rate = 0.1 ∧
   predicted_income = final_income * (1 + growth_rate)) := by
  sorry

#check village_income_growth_and_prediction

end NUMINAMATH_CALUDE_village_income_growth_and_prediction_l1561_156176


namespace NUMINAMATH_CALUDE_two_week_egg_consumption_l1561_156123

/-- Calculates the total number of eggs consumed over a given number of days,
    given a daily egg consumption rate. -/
def totalEggsConsumed (dailyConsumption : ℕ) (days : ℕ) : ℕ :=
  dailyConsumption * days

/-- Theorem stating that consuming 3 eggs daily for 14 days results in 42 eggs consumed. -/
theorem two_week_egg_consumption :
  totalEggsConsumed 3 14 = 42 := by
  sorry

end NUMINAMATH_CALUDE_two_week_egg_consumption_l1561_156123


namespace NUMINAMATH_CALUDE_first_apartment_rent_l1561_156175

theorem first_apartment_rent (R : ℝ) : 
  R + 260 + (31 * 20 * 0.58) - (900 + 200 + (21 * 20 * 0.58)) = 76 → R = 800 := by
  sorry

end NUMINAMATH_CALUDE_first_apartment_rent_l1561_156175


namespace NUMINAMATH_CALUDE_range_of_m_l1561_156151

/-- Given two predicates p and q on real numbers, where p states that there exists a real x such that
    mx² + 1 ≤ 0, and q states that for all real x, x² + mx + 1 > 0, if the disjunction of p and q
    is false, then m is greater than or equal to 2. -/
theorem range_of_m (m : ℝ) : 
  let p := ∃ x : ℝ, m * x^2 + 1 ≤ 0
  let q := ∀ x : ℝ, x^2 + m * x + 1 > 0
  ¬(p ∨ q) → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1561_156151


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l1561_156134

theorem power_mod_thirteen : 6^4032 ≡ 1 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l1561_156134


namespace NUMINAMATH_CALUDE_exactly_one_tail_in_three_flips_l1561_156157

-- Define a fair coin
def fair_coin_prob : ℝ := 0.5

-- Define the number of flips
def num_flips : ℕ := 3

-- Define the number of tails we want
def num_tails : ℕ := 1

-- Define the binomial coefficient function
def choose (n k : ℕ) : ℕ := 
  Nat.choose n k

-- Define the probability of exactly k successes in n trials
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (choose n k : ℝ) * p^k * (1 - p)^(n - k)

-- Theorem statement
theorem exactly_one_tail_in_three_flips :
  binomial_probability num_flips num_tails fair_coin_prob = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_tail_in_three_flips_l1561_156157


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l1561_156147

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- State the theorem
theorem opposite_of_negative_three : opposite (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l1561_156147


namespace NUMINAMATH_CALUDE_cyclic_sum_factorization_l1561_156159

theorem cyclic_sum_factorization (a b c : ℝ) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = (a - b) * (b - c) * (c - a) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_factorization_l1561_156159


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1561_156140

theorem exponent_multiplication (a : ℝ) : 2 * (a^2 * a^4) = 2 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1561_156140


namespace NUMINAMATH_CALUDE_garden_dimensions_possible_longest_side_l1561_156182

/-- Represents a rectangular garden with one side along a wall -/
structure RectangularGarden where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ
  total_fence : ℕ
  h_fence : side1 + side2 + side3 = total_fence

/-- The total fence length is 140 meters -/
def total_fence : ℕ := 140

theorem garden_dimensions (g : RectangularGarden) 
  (h1 : g.side1 = 40) (h2 : g.side2 = 40) (h_total : g.total_fence = total_fence) : 
  g.side3 = 60 := by
  sorry

theorem possible_longest_side (g : RectangularGarden) (h_total : g.total_fence = total_fence) :
  (∃ (g' : RectangularGarden), g'.side1 = 65 ∨ g'.side2 = 65 ∨ g'.side3 = 65) ∧
  (¬∃ (g' : RectangularGarden), g'.side1 = 85 ∨ g'.side2 = 85 ∨ g'.side3 = 85) := by
  sorry

end NUMINAMATH_CALUDE_garden_dimensions_possible_longest_side_l1561_156182


namespace NUMINAMATH_CALUDE_tangent_line_t_value_l1561_156113

/-- A line in polar coordinates defined by ρcosθ = t, where t > 0 -/
structure PolarLine where
  t : ℝ
  t_pos : t > 0

/-- A curve in polar coordinates defined by ρ = 2sinθ -/
def PolarCurve : Type := Unit

/-- Predicate to check if a line is tangent to the curve -/
def is_tangent (l : PolarLine) (c : PolarCurve) : Prop := sorry

theorem tangent_line_t_value (l : PolarLine) (c : PolarCurve) :
  is_tangent l c → l.t = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_t_value_l1561_156113


namespace NUMINAMATH_CALUDE_sin_2017pi_over_3_l1561_156102

theorem sin_2017pi_over_3 : Real.sin (2017 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2017pi_over_3_l1561_156102


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_sum_squares_l1561_156191

theorem square_sum_from_difference_and_sum_squares 
  (m n : ℝ) 
  (h1 : (m - n)^2 = 8) 
  (h2 : (m + n)^2 = 2) : 
  m^2 + n^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_sum_squares_l1561_156191


namespace NUMINAMATH_CALUDE_matching_times_correct_l1561_156188

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  hLt24 : hours < 24
  mLt60 : minutes < 60

/-- Calculates the total minutes elapsed since 00:00 -/
def totalMinutes (t : Time) : Nat :=
  t.hours * 60 + t.minutes

/-- Calculates the charge of the mortar at a given time -/
def charge (t : Time) : Nat :=
  100 - (totalMinutes t) / 6

/-- The list of times when the charge equals the number of minutes -/
def matchingTimes : List Time := [
  ⟨4, 52, by sorry, by sorry⟩,
  ⟨5, 43, by sorry, by sorry⟩,
  ⟨6, 35, by sorry, by sorry⟩,
  ⟨7, 26, by sorry, by sorry⟩,
  ⟨9, 9, by sorry, by sorry⟩
]

/-- Theorem stating that the matching times are correct -/
theorem matching_times_correct :
  ∀ t ∈ matchingTimes, charge t = t.minutes :=
by sorry

end NUMINAMATH_CALUDE_matching_times_correct_l1561_156188


namespace NUMINAMATH_CALUDE_expand_x_plus_y_seventh_third_to_fourth_term_ratio_p_plus_q_equals_three_p_and_q_positive_prove_p_value_l1561_156110

/-- The value of p in the expansion of (x+y)^7 -/
def p : ℚ :=
  30/13

/-- The value of q in the expansion of (x+y)^7 -/
def q : ℚ :=
  9/13

/-- The ratio of the third to fourth term in the expansion of (x+y)^7 when x=p and y=q -/
def ratio : ℚ :=
  2/1

theorem expand_x_plus_y_seventh (x y : ℚ) :
  (x + y)^7 = x^7 + 7*x^6*y + 21*x^5*y^2 + 35*x^4*y^3 + 35*x^3*y^4 + 21*x^2*y^5 + 7*x*y^6 + y^7 :=
sorry

theorem third_to_fourth_term_ratio :
  (21 * p^5 * q^2) / (35 * p^4 * q^3) = ratio :=
sorry

theorem p_plus_q_equals_three :
  p + q = 3 :=
sorry

theorem p_and_q_positive :
  p > 0 ∧ q > 0 :=
sorry

theorem prove_p_value :
  p = 30/13 :=
sorry

end NUMINAMATH_CALUDE_expand_x_plus_y_seventh_third_to_fourth_term_ratio_p_plus_q_equals_three_p_and_q_positive_prove_p_value_l1561_156110


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l1561_156168

theorem complex_number_coordinates : Complex.I * 2 / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l1561_156168


namespace NUMINAMATH_CALUDE_white_square_area_l1561_156155

/-- Given a cube with edge length 12 feet and 432 square feet of green paint used equally on all faces as a border, the area of the white square centered on each face is 72 square feet. -/
theorem white_square_area (cube_edge : ℝ) (green_paint_area : ℝ) (white_square_area : ℝ) : 
  cube_edge = 12 →
  green_paint_area = 432 →
  white_square_area = 72 →
  white_square_area = cube_edge^2 - green_paint_area / 6 :=
by sorry

end NUMINAMATH_CALUDE_white_square_area_l1561_156155


namespace NUMINAMATH_CALUDE_triangle_ratio_l1561_156186

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  b * Real.sin A * Real.sin B + a * (Real.cos B)^2 = 2 * c →
  a / c = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_ratio_l1561_156186


namespace NUMINAMATH_CALUDE_whitney_fish_books_l1561_156150

/-- The number of books about fish Whitney bought -/
def fish_books : ℕ := 7

/-- The number of books about whales Whitney bought -/
def whale_books : ℕ := 9

/-- The number of magazines Whitney bought -/
def magazines : ℕ := 3

/-- The cost of each book in dollars -/
def book_cost : ℕ := 11

/-- The cost of each magazine in dollars -/
def magazine_cost : ℕ := 1

/-- The total amount Whitney spent in dollars -/
def total_spent : ℕ := 179

theorem whitney_fish_books :
  whale_books * book_cost + fish_books * book_cost + magazines * magazine_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_whitney_fish_books_l1561_156150


namespace NUMINAMATH_CALUDE_degree_of_minus_five_x_squared_y_l1561_156142

def monomial_degree (m : ℤ → ℤ → ℤ) : ℕ :=
  sorry

theorem degree_of_minus_five_x_squared_y :
  monomial_degree (fun x y ↦ -5 * x^2 * y) = 3 :=
sorry

end NUMINAMATH_CALUDE_degree_of_minus_five_x_squared_y_l1561_156142


namespace NUMINAMATH_CALUDE_part_one_part_two_l1561_156171

-- Part 1
theorem part_one (a b : ℝ) (h1 : 1 ≤ a - b) (h2 : a - b ≤ 2) (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) :
  5 ≤ 4*a - 2*b ∧ 4*a - 2*b ≤ 10 := by sorry

-- Part 2
theorem part_two (m : ℝ) (h : ∀ x > (1/2 : ℝ), 2*x^2 - x ≥ 2*m*x - m - 8) :
  m ≤ 9/2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1561_156171


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1561_156136

/-- 
For a quadratic equation x^2 - 4x + m - 1 = 0, 
if it has two distinct real roots, then m < 5.
-/
theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 - 4*x + m - 1 = 0 ∧ 
    y^2 - 4*y + m - 1 = 0) → 
  m < 5 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1561_156136


namespace NUMINAMATH_CALUDE_sara_remaining_pears_l1561_156143

def pears_distribution (initial_pears : ℕ) (to_dan : ℕ) (to_monica : ℕ) : ℕ :=
  let remaining_after_dan := initial_pears - to_dan
  let remaining_after_monica := remaining_after_dan - to_monica
  let to_jenny := remaining_after_monica / 2
  remaining_after_monica - to_jenny

theorem sara_remaining_pears :
  pears_distribution 35 28 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sara_remaining_pears_l1561_156143


namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l1561_156133

theorem quadratic_equation_transformation (x : ℝ) :
  (4 * x^2 + 8 * x - 468 = 0) →
  ∃ p q : ℝ, ((x + p)^2 = q) ∧ (q = 116) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l1561_156133


namespace NUMINAMATH_CALUDE_train_length_l1561_156145

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 3 → ∃ (length : ℝ), abs (length - 50.01) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1561_156145


namespace NUMINAMATH_CALUDE_problem_solution_l1561_156162

def f (a x : ℝ) : ℝ := |2*x - a| + |x - 1|

theorem problem_solution :
  (∀ a : ℝ, (∀ x : ℝ, f a x + |x - 1| ≥ 2) → a ≤ 0 ∨ a ≥ 4) ∧
  (∀ a : ℝ, a < 2 → (∃ x : ℝ, ∀ y : ℝ, f a x ≤ f a y) → f a (a/2) = a - 1 → a = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1561_156162


namespace NUMINAMATH_CALUDE_puzzle_assembly_time_l1561_156196

/-- Represents the time taken to assemble a puzzle -/
def assemble_puzzle (initial_pieces : ℕ) (pieces_per_minute : ℕ) : ℕ :=
  (initial_pieces - 1) / (pieces_per_minute - 1)

theorem puzzle_assembly_time :
  let initial_pieces : ℕ := 121
  let two_piece_time : ℕ := 120  -- 2 hours in minutes
  let three_piece_time : ℕ := 60 -- 1 hour in minutes
  assemble_puzzle initial_pieces 2 = two_piece_time →
  assemble_puzzle initial_pieces 3 = three_piece_time :=
by sorry

end NUMINAMATH_CALUDE_puzzle_assembly_time_l1561_156196


namespace NUMINAMATH_CALUDE_polynomial_independence_l1561_156122

-- Define the polynomials A and B
def A (x a : ℝ) : ℝ := x^2 + a*x
def B (x b : ℝ) : ℝ := 2*b*x^2 - 4*x - 1

-- Define the combined polynomial 2A + B
def combined_polynomial (x a b : ℝ) : ℝ := 2 * A x a + B x b

-- Theorem statement
theorem polynomial_independence (a b : ℝ) : 
  (∀ x : ℝ, ∃ c : ℝ, combined_polynomial x a b = c) ↔ (a = 2 ∧ b = -1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_independence_l1561_156122


namespace NUMINAMATH_CALUDE_rectangle_width_l1561_156126

/-- Given a rectangle ABCD with length 25 yards and an inscribed rhombus AFCE with perimeter 82 yards, 
    the width of the rectangle is equal to √(420.25 / 2) yards. -/
theorem rectangle_width (length : ℝ) (perimeter : ℝ) (width : ℝ) : 
  length = 25 →
  perimeter = 82 →
  width = Real.sqrt (420.25 / 2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_l1561_156126


namespace NUMINAMATH_CALUDE_yellow_marble_probability_l1561_156121

/-- Represents a bag of marbles -/
structure Bag where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0

/-- Calculates the total number of marbles in a bag -/
def Bag.total (b : Bag) : ℕ := b.white + b.black + b.yellow + b.blue

/-- Defines the bags A, B, C, and D -/
def bagA : Bag := { white := 4, black := 5 }
def bagB : Bag := { yellow := 7, blue := 3 }
def bagC : Bag := { yellow := 3, blue := 6 }
def bagD : Bag := { yellow := 5, blue := 4 }

/-- Calculates the probability of drawing a yellow marble given the problem conditions -/
def yellowProbability : ℚ :=
  let pWhiteA := bagA.white / bagA.total
  let pBlackA := bagA.black / bagA.total
  let pYellowB := bagB.yellow / bagB.total
  let pBlueB := bagB.blue / bagB.total
  let pYellowC := bagC.yellow / bagC.total
  let pBlueC := bagC.blue / bagC.total
  let pYellowD := bagD.yellow / bagD.total
  pWhiteA * pYellowB + pBlackA * pYellowC + pWhiteA * pBlueB * pYellowD + pBlackA * pBlueC * pYellowD

/-- The main theorem stating that the probability of drawing a yellow marble is 1884/3645 -/
theorem yellow_marble_probability : yellowProbability = 1884 / 3645 := by
  sorry


end NUMINAMATH_CALUDE_yellow_marble_probability_l1561_156121


namespace NUMINAMATH_CALUDE_man_running_opposite_direction_l1561_156107

/-- Proves that the relative speed between a train and a man is equal to the sum of their speeds,
    indicating that the man is running in the opposite direction to the train. -/
theorem man_running_opposite_direction
  (train_length : ℝ)
  (train_speed : ℝ)
  (man_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 110)
  (h2 : train_speed = 40 * 1000 / 3600)
  (h3 : man_speed = 4 * 1000 / 3600)
  (h4 : passing_time = 9) :
  train_length / passing_time = train_speed + man_speed := by
  sorry

#check man_running_opposite_direction

end NUMINAMATH_CALUDE_man_running_opposite_direction_l1561_156107


namespace NUMINAMATH_CALUDE_f_increasing_iff_m_range_l1561_156160

def f (x m : ℝ) : ℝ := |x^2 + (m-1)*x + (m^2 - 3*m + 1)|

theorem f_increasing_iff_m_range :
  ∀ m : ℝ, (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 0 → f x₁ m < f x₂ m) ↔ (m = 1 ∨ m ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_iff_m_range_l1561_156160


namespace NUMINAMATH_CALUDE_max_pie_pieces_l1561_156144

/-- Represents a five-digit number with distinct digits -/
def DistinctFiveDigitNumber (x : ℕ) : Prop :=
  10000 ≤ x ∧ x < 100000 ∧ (∀ i j, i ≠ j → (x / 10^i) % 10 ≠ (x / 10^j) % 10)

/-- The maximum number of pieces that can be obtained when dividing a pie -/
theorem max_pie_pieces :
  ∃ (n : ℕ) (pie piece : ℕ),
    n = 7 ∧
    DistinctFiveDigitNumber pie ∧
    10000 ≤ piece ∧ piece < 100000 ∧
    pie = piece * n ∧
    (∀ m > n, ¬∃ (p q : ℕ),
      DistinctFiveDigitNumber p ∧
      10000 ≤ q ∧ q < 100000 ∧
      p = q * m) :=
by sorry

end NUMINAMATH_CALUDE_max_pie_pieces_l1561_156144


namespace NUMINAMATH_CALUDE_percentage_increase_l1561_156131

theorem percentage_increase (original : ℝ) (new : ℝ) : 
  original = 80 → new = 96 → (new - original) / original * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1561_156131


namespace NUMINAMATH_CALUDE_cost_23_days_l1561_156119

/-- Calculate the cost of staying in a student youth hostel for a given number of days. -/
def hostelCost (days : ℕ) : ℚ :=
  let firstWeekRate : ℚ := 18
  let additionalWeekRate : ℚ := 12
  let firstWeekDays : ℕ := min 7 days
  let additionalDays : ℕ := days - firstWeekDays
  let firstWeekCost : ℚ := firstWeekRate * firstWeekDays
  let additionalCost : ℚ := additionalWeekRate * additionalDays
  firstWeekCost + additionalCost

/-- Theorem stating that the cost for a 23-day stay is $318.00 -/
theorem cost_23_days :
  hostelCost 23 = 318 := by
  sorry

#eval hostelCost 23

end NUMINAMATH_CALUDE_cost_23_days_l1561_156119


namespace NUMINAMATH_CALUDE_minimum_value_of_F_l1561_156118

/-- A function is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem minimum_value_of_F (m n : ℝ) (f g : ℝ → ℝ) :
  (∀ x > 0, f x + n * g x + x + 2 ≤ 8) →
  OddFunction f →
  OddFunction g →
  ∃ c, c = -4 ∧ ∀ x < 0, m * f x + n * g x + x + 2 ≥ c :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_F_l1561_156118


namespace NUMINAMATH_CALUDE_carlos_total_earnings_l1561_156148

-- Define the problem parameters
def hours_week1 : ℕ := 18
def hours_week2 : ℕ := 30
def extra_earnings : ℕ := 54

-- Define Carlos's hourly wage as a rational number
def hourly_wage : ℚ := 54 / 12

-- Theorem statement
theorem carlos_total_earnings :
  (hours_week1 : ℚ) * hourly_wage + (hours_week2 : ℚ) * hourly_wage = 216 := by
  sorry

#eval (hours_week1 : ℚ) * hourly_wage + (hours_week2 : ℚ) * hourly_wage

end NUMINAMATH_CALUDE_carlos_total_earnings_l1561_156148


namespace NUMINAMATH_CALUDE_prob_win_series_4_1_l1561_156198

/-- Represents the location of a game -/
inductive GameLocation
  | Home
  | Away

/-- Represents the schedule of games for Team A -/
def schedule : List GameLocation :=
  [GameLocation.Home, GameLocation.Home, GameLocation.Away, GameLocation.Away, 
   GameLocation.Home, GameLocation.Away, GameLocation.Home]

/-- Probability of Team A winning a home game -/
def probWinHome : ℝ := 0.6

/-- Probability of Team A winning an away game -/
def probWinAway : ℝ := 0.5

/-- Calculates the probability of Team A winning a game based on its location -/
def probWin (loc : GameLocation) : ℝ :=
  match loc with
  | GameLocation.Home => probWinHome
  | GameLocation.Away => probWinAway

/-- Calculates the probability of a specific game outcome for Team A -/
def probOutcome (outcomes : List Bool) : ℝ :=
  List.zipWith (fun o l => if o then probWin l else 1 - probWin l) outcomes schedule
  |> List.prod

/-- Theorem: The probability of Team A winning the series with a 4:1 score is 0.18 -/
theorem prob_win_series_4_1 : 
  (probOutcome [false, true, true, true, true] +
   probOutcome [true, false, true, true, true] +
   probOutcome [true, true, false, true, true] +
   probOutcome [true, true, true, false, true]) = 0.18 := by
  sorry


end NUMINAMATH_CALUDE_prob_win_series_4_1_l1561_156198


namespace NUMINAMATH_CALUDE_sunset_time_calculation_l1561_156137

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Adds two Times together -/
def addTime (t1 t2 : Time) : Time :=
  let totalMinutes := t1.hours * 60 + t1.minutes + t2.hours * 60 + t2.minutes
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

/-- Converts 24-hour format to 12-hour format -/
def to12HourFormat (t : Time) : Time :=
  if t.hours ≥ 12 then
    { hours := if t.hours = 12 then 12 else t.hours - 12, minutes := t.minutes }
  else
    { hours := if t.hours = 0 then 12 else t.hours, minutes := t.minutes }

theorem sunset_time_calculation (sunrise : Time) (daylight : Time) : 
  to12HourFormat (addTime sunrise daylight) = { hours := 7, minutes := 40 } :=
  sorry

end NUMINAMATH_CALUDE_sunset_time_calculation_l1561_156137


namespace NUMINAMATH_CALUDE_sinusoidal_function_omega_l1561_156146

/-- Given a sinusoidal function y = 2sin(ωx + π/6) with ω > 0,
    if the distance between adjacent symmetry axes is π/2,
    then ω = 2. -/
theorem sinusoidal_function_omega (ω : ℝ) (h1 : ω > 0) :
  (∀ x : ℝ, 2 * Real.sin (ω * x + π / 6) = 2 * Real.sin (ω * (x + π / (2 * ω)) + π / 6)) →
  ω = 2 := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_function_omega_l1561_156146


namespace NUMINAMATH_CALUDE_total_rainbow_nerds_l1561_156170

/-- The number of rainbow nerds in a box with purple, yellow, and green candies. -/
def rainbow_nerds (purple : ℕ) (yellow : ℕ) (green : ℕ) : ℕ := purple + yellow + green

/-- Theorem: The total number of rainbow nerds in the box is 36. -/
theorem total_rainbow_nerds :
  ∃ (purple yellow green : ℕ),
    purple = 10 ∧
    yellow = purple + 4 ∧
    green = yellow - 2 ∧
    rainbow_nerds purple yellow green = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_rainbow_nerds_l1561_156170


namespace NUMINAMATH_CALUDE_six_chairs_three_people_l1561_156130

/-- The number of ways to arrange n people among m chairs in a row, with no two people adjacent -/
def nonadjacentArrangements (m n : ℕ) : ℕ :=
  if m ≤ n then 0
  else Nat.descFactorial (m - n + 1) n

theorem six_chairs_three_people :
  nonadjacentArrangements 6 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_six_chairs_three_people_l1561_156130


namespace NUMINAMATH_CALUDE_parallel_equidistant_lines_theorem_l1561_156128

/-- Represents a line segment with a length -/
structure LineSegment where
  length : ℝ

/-- Represents three parallel, equidistant line segments -/
structure ParallelEquidistantLines where
  line1 : LineSegment
  line2 : LineSegment
  line3 : LineSegment

/-- Given three parallel, equidistant lines where the first line is 120 cm and the second is 80 cm,
    the length of the third line is 160/3 cm -/
theorem parallel_equidistant_lines_theorem (lines : ParallelEquidistantLines) 
    (h1 : lines.line1.length = 120)
    (h2 : lines.line2.length = 80) :
    lines.line3.length = 160 / 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_equidistant_lines_theorem_l1561_156128


namespace NUMINAMATH_CALUDE_rotate90_neg_6_minus_3i_l1561_156129

def rotate90 (z : ℂ) : ℂ := z * Complex.I

theorem rotate90_neg_6_minus_3i :
  rotate90 (-6 - 3 * Complex.I) = (3 : ℂ) - 6 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_rotate90_neg_6_minus_3i_l1561_156129


namespace NUMINAMATH_CALUDE_impossibleToMakeAllEqual_l1561_156104

/-- Represents the possible values in a cell of the table -/
inductive CellValue
  | Zero
  | One
  deriving Repr

/-- Represents a 4x4 table of cell values -/
def Table := Fin 4 → Fin 4 → CellValue

/-- Represents the initial state of the table -/
def initialTable : Table := fun i j =>
  if i = 0 ∧ j = 1 then CellValue.One else CellValue.Zero

/-- Represents the allowed operations on the table -/
inductive Operation
  | AddToRow (row : Fin 4)
  | AddToColumn (col : Fin 4)
  | AddToDiagonal (startRow startCol : Fin 4)

/-- Applies an operation to a table -/
def applyOperation (t : Table) (op : Operation) : Table :=
  sorry

/-- Checks if all values in the table are equal -/
def allEqual (t : Table) : Prop :=
  ∀ i j k l, t i j = t k l

/-- The main theorem stating that it's impossible to make all numbers equal -/
theorem impossibleToMakeAllEqual :
  ¬∃ (ops : List Operation), allEqual (ops.foldl applyOperation initialTable) :=
sorry

end NUMINAMATH_CALUDE_impossibleToMakeAllEqual_l1561_156104


namespace NUMINAMATH_CALUDE_linear_system_solution_l1561_156174

theorem linear_system_solution : 
  ∀ (x y : ℝ), 
    (2 * x + 3 * y = 4) → 
    (x = -y) → 
    (x = -4 ∧ y = 4) := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1561_156174


namespace NUMINAMATH_CALUDE_two_distinct_real_roots_l1561_156152

def polynomial (a x : ℝ) : ℝ := x^4 + 3*a*x^3 + a*(1-5*a^2)*x - 3*a^4 + a^2 + 1

theorem two_distinct_real_roots (a : ℝ) :
  (∃ x : ℝ, polynomial a x = 0) ∧ 
  (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ polynomial a x₁ = 0 ∧ polynomial a x₂ = 0) →
  a = 2 * Real.sqrt 26 / 13 ∨ a = -2 * Real.sqrt 26 / 13 := by
  sorry

end NUMINAMATH_CALUDE_two_distinct_real_roots_l1561_156152


namespace NUMINAMATH_CALUDE_unshaded_area_of_intersecting_rectangles_l1561_156178

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the intersection of two rectangles -/
structure Intersection where
  width : ℝ
  height : ℝ

theorem unshaded_area_of_intersecting_rectangles
  (r1 : Rectangle)
  (r2 : Rectangle)
  (i : Intersection)
  (h1 : r1.width = 4 ∧ r1.height = 12)
  (h2 : r2.width = 5 ∧ r2.height = 10)
  (h3 : i.width = 4 ∧ i.height = 5) :
  area r1 + area r2 - (area r1 + area r2 - i.width * i.height) = 20 :=
sorry

end NUMINAMATH_CALUDE_unshaded_area_of_intersecting_rectangles_l1561_156178


namespace NUMINAMATH_CALUDE_pet_store_cages_l1561_156165

/-- Given a pet store scenario, prove the number of cages used -/
theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 78)
  (h2 : sold_puppies = 30)
  (h3 : puppies_per_cage = 8) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 6 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l1561_156165


namespace NUMINAMATH_CALUDE_rosie_pies_theorem_l1561_156132

/-- Represents the number of pies that can be made from a given number of apples -/
def pies_from_apples (apples : ℕ) : ℕ :=
  (apples / 12) * 3

theorem rosie_pies_theorem :
  pies_from_apples 36 = 9 :=
by sorry

end NUMINAMATH_CALUDE_rosie_pies_theorem_l1561_156132


namespace NUMINAMATH_CALUDE_reciprocal_of_five_l1561_156153

theorem reciprocal_of_five (x : ℚ) : x = 5 → (1 : ℚ) / x = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_five_l1561_156153


namespace NUMINAMATH_CALUDE_vector_at_t_4_l1561_156163

/-- A line in 3D space parameterized by t -/
structure ParametricLine where
  point : ℝ → ℝ × ℝ × ℝ

/-- Given conditions for the line -/
def line_conditions (L : ParametricLine) : Prop :=
  L.point (-2) = (2, 6, 16) ∧ L.point 1 = (-1, -5, -10)

/-- Theorem: The vector on the line at t = 4 is (-16, -60, -140) -/
theorem vector_at_t_4 (L : ParametricLine) 
  (h : line_conditions L) : L.point 4 = (-16, -60, -140) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_t_4_l1561_156163


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l1561_156166

theorem smallest_lcm_with_gcd_5 (k ℓ : ℕ) :
  k ≥ 1000 ∧ k < 10000 ∧ ℓ ≥ 1000 ∧ ℓ < 10000 ∧ Nat.gcd k ℓ = 5 →
  Nat.lcm k ℓ ≥ 201000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l1561_156166


namespace NUMINAMATH_CALUDE_speed_ratio_l1561_156164

/-- The speed of object A in meters per minute -/
def v_A : ℝ := sorry

/-- The speed of object B in meters per minute -/
def v_B : ℝ := sorry

/-- The initial distance of B from point O in meters -/
def initial_distance_B : ℝ := 600

/-- The time when A and B are first equidistant from O in minutes -/
def t1 : ℝ := 4

/-- The time when A and B are again equidistant from O in minutes -/
def t2 : ℝ := 9

/-- Theorem stating that the ratio of A's speed to B's speed is 2/3 -/
theorem speed_ratio :
  v_A / v_B = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_speed_ratio_l1561_156164


namespace NUMINAMATH_CALUDE_interchange_digits_theorem_l1561_156141

theorem interchange_digits_theorem (a b k : ℕ) (n : ℕ) :
  n = 10 * a + b →
  n = (k + 1) * (a + b) →
  10 * b + a = (10 - k) * (a + b) :=
by sorry

end NUMINAMATH_CALUDE_interchange_digits_theorem_l1561_156141


namespace NUMINAMATH_CALUDE_functional_equation_equivalence_l1561_156199

theorem functional_equation_equivalence (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f x + f y) ↔
  (∀ x y : ℝ, f (x + y + x * y) = f x + f y + f (x * y)) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_equivalence_l1561_156199


namespace NUMINAMATH_CALUDE_sum_of_cubes_equals_fourth_power_l1561_156103

theorem sum_of_cubes_equals_fourth_power : 5^3 + 5^3 + 5^3 + 5^3 = 5^4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equals_fourth_power_l1561_156103


namespace NUMINAMATH_CALUDE_plane_speed_against_wind_l1561_156114

/-- Calculates the ground speed of a plane flying against a tailwind, given its ground speed with the tailwind and the wind speed. -/
def ground_speed_against_wind (ground_speed_with_wind wind_speed : ℝ) : ℝ :=
  2 * ground_speed_with_wind - 2 * wind_speed - wind_speed

/-- Theorem stating that a plane with a ground speed of 460 mph with a 75 mph tailwind
    will have a ground speed of 310 mph against the same tailwind. -/
theorem plane_speed_against_wind :
  ground_speed_against_wind 460 75 = 310 := by
  sorry

end NUMINAMATH_CALUDE_plane_speed_against_wind_l1561_156114


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l1561_156109

/-- The number of cups of flour Mary has already put in -/
def flour_already_added : ℕ := 2

/-- The number of cups of flour Mary needs to add -/
def flour_to_add : ℕ := 5

/-- The total number of cups of flour in the recipe -/
def total_flour : ℕ := flour_already_added + flour_to_add

theorem recipe_flour_amount : total_flour = 7 := by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l1561_156109


namespace NUMINAMATH_CALUDE_line_point_k_value_l1561_156167

/-- A line contains the points (5,10), (-3,k), and (-11,5). This theorem proves that k = 7.5. -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (m b : ℝ), 
    (10 = m * 5 + b) ∧ 
    (k = m * (-3) + b) ∧ 
    (5 = m * (-11) + b)) → 
  k = 7.5 := by
sorry

end NUMINAMATH_CALUDE_line_point_k_value_l1561_156167


namespace NUMINAMATH_CALUDE_B_subset_A_l1561_156169

def A : Set ℝ := {x | x ≥ 1}
def B : Set ℝ := {x | x > 2}

theorem B_subset_A : B ⊆ A := by sorry

end NUMINAMATH_CALUDE_B_subset_A_l1561_156169


namespace NUMINAMATH_CALUDE_luther_clothing_line_l1561_156154

/-- The number of silk pieces in Luther's clothing line -/
def silk_pieces : ℕ := 7

/-- The number of cashmere pieces in Luther's clothing line -/
def cashmere_pieces : ℕ := silk_pieces / 2

/-- The number of blended pieces using both cashmere and silk -/
def blended_pieces : ℕ := 2

/-- The total number of pieces in Luther's clothing line -/
def total_pieces : ℕ := 13

theorem luther_clothing_line :
  silk_pieces + cashmere_pieces + blended_pieces = total_pieces ∧
  cashmere_pieces = silk_pieces / 2 ∧
  silk_pieces = 7 := by sorry

end NUMINAMATH_CALUDE_luther_clothing_line_l1561_156154


namespace NUMINAMATH_CALUDE_square_triangulation_l1561_156177

/-- A planar graph representing the configuration of points and lines in a square -/
structure SquareGraph where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces (regions)

/-- The number of triangles formed in a square with 20 internal points -/
def num_triangles (g : SquareGraph) : ℕ := g.F - 1

theorem square_triangulation :
  ∀ g : SquareGraph,
  g.V = 24 →  -- 20 internal points + 4 vertices of the square
  2 * g.E = 3 * g.F + 1 →  -- Relation between edges and faces
  g.V - g.E + g.F = 2 →  -- Euler's formula for planar graphs
  num_triangles g = 42 := by
sorry

end NUMINAMATH_CALUDE_square_triangulation_l1561_156177


namespace NUMINAMATH_CALUDE_inequality_proof_l1561_156116

theorem inequality_proof (a b c : ℝ) (h : a^6 + b^6 + c^6 = 3) :
  a^7 * b^2 + b^7 * c^2 + c^7 * a^2 ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1561_156116


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1561_156197

theorem sum_of_coefficients (A B : ℝ) :
  (∀ x : ℝ, x ≠ 4 → A / (x - 4) + B * (x + 1) = (-4 * x^2 + 16 * x + 24) / (x - 4)) →
  A + B = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1561_156197


namespace NUMINAMATH_CALUDE_wood_length_after_sawing_l1561_156125

theorem wood_length_after_sawing (original_length sawed_off_length : ℝ) 
  (h1 : original_length = 0.41)
  (h2 : sawed_off_length = 0.33) :
  original_length - sawed_off_length = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_wood_length_after_sawing_l1561_156125


namespace NUMINAMATH_CALUDE_range_of_a_l1561_156105

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (A ∪ B a = {x : ℝ | x < 4}) ↔ (-2 < a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1561_156105


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1561_156120

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1561_156120


namespace NUMINAMATH_CALUDE_path_length_for_73_l1561_156139

/-- The length of a path around squares constructed on segments of a line -/
def path_length (segment_length : ℝ) : ℝ :=
  3 * segment_length

theorem path_length_for_73 :
  path_length 73 = 219 := by sorry

end NUMINAMATH_CALUDE_path_length_for_73_l1561_156139


namespace NUMINAMATH_CALUDE_equation_solution_l1561_156117

theorem equation_solution : ∃ x : ℝ, (x + 1 ≠ 0 ∧ x^2 - 1 ≠ 0) ∧ 
  (2 * x / (x + 1) - 2 = 3 / (x^2 - 1)) ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1561_156117


namespace NUMINAMATH_CALUDE_abc_inequality_l1561_156138

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  a + b + c ≤ a^2 + b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1561_156138


namespace NUMINAMATH_CALUDE_root_preservation_l1561_156194

-- Define the polynomial p(x) = x³ - 5x + 3
def p (x : ℚ) : ℚ := x^3 - 5*x + 3

-- Define a type for polynomials with rational coefficients
def RationalPolynomial := ℚ → ℚ

-- Theorem statement
theorem root_preservation 
  (α : ℚ) 
  (f : RationalPolynomial) 
  (h1 : p α = 0) 
  (h2 : p (f α) = 0) : 
  p (f (f α)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_preservation_l1561_156194


namespace NUMINAMATH_CALUDE_some_value_proof_l1561_156183

theorem some_value_proof (a : ℝ) : 
  (∀ x : ℝ, |x - a| = 100 → (a + 100) + (a - 100) = 24) → 
  a = 12 := by
  sorry

end NUMINAMATH_CALUDE_some_value_proof_l1561_156183


namespace NUMINAMATH_CALUDE_floor_length_proof_l1561_156108

/-- Represents a rectangular tile -/
structure Tile :=
  (length : ℕ)
  (width : ℕ)

/-- Represents a rectangular floor -/
structure Floor :=
  (length : ℕ)
  (width : ℕ)

/-- Calculates the maximum number of tiles that can fit on the floor -/
def maxTiles (t : Tile) (f : Floor) : ℕ :=
  let tilesAcross := f.width / t.width
  let tilesDown := f.length / t.length
  tilesAcross * tilesDown

theorem floor_length_proof (t : Tile) (f : Floor) (h1 : t.length = 25) (h2 : t.width = 16) 
    (h3 : f.width = 120) (h4 : maxTiles t f = 54) : f.length = 175 :=
by
  sorry

#check floor_length_proof

end NUMINAMATH_CALUDE_floor_length_proof_l1561_156108


namespace NUMINAMATH_CALUDE_prime_product_sum_l1561_156173

theorem prime_product_sum (m n p : ℕ) : 
  Prime m ∧ Prime n ∧ Prime p ∧ m * n * p = 5 * (m + n + p) → m^2 + n^2 + p^2 = 78 :=
by sorry

end NUMINAMATH_CALUDE_prime_product_sum_l1561_156173


namespace NUMINAMATH_CALUDE_only_solution_is_three_l1561_156135

def sum_of_digits (n : ℕ) : ℕ :=
  sorry

theorem only_solution_is_three :
  ∃! n : ℕ, sum_of_digits (5^n) = 2^n ∧ n = 3 :=
sorry

end NUMINAMATH_CALUDE_only_solution_is_three_l1561_156135


namespace NUMINAMATH_CALUDE_james_birthday_stickers_l1561_156101

/-- The number of stickers James gets for his birthday -/
def birthday_stickers (initial_stickers total_stickers : ℕ) : ℕ :=
  total_stickers - initial_stickers

/-- Theorem: James got 22 stickers for his birthday -/
theorem james_birthday_stickers :
  birthday_stickers 39 61 = 22 := by
  sorry

end NUMINAMATH_CALUDE_james_birthday_stickers_l1561_156101


namespace NUMINAMATH_CALUDE_center_of_hyperbola_l1561_156172

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 891 = 0

-- Define the center of a hyperbola
def is_center (c : ℝ × ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    eq x y ↔ (x - c.1)^2 / a^2 - (y - c.2)^2 / b^2 = 1

-- Theorem stating that (3, 5) is the center of the given hyperbola
theorem center_of_hyperbola :
  is_center (3, 5) hyperbola_eq :=
sorry

end NUMINAMATH_CALUDE_center_of_hyperbola_l1561_156172


namespace NUMINAMATH_CALUDE_eighteen_wheel_truck_toll_l1561_156190

/-- Calculates the toll for a truck based on the number of axles -/
def toll (axles : ℕ) : ℚ :=
  1.50 + 0.50 * (axles - 2)

/-- Calculates the number of axles for a truck given the total number of wheels -/
def axles_count (wheels : ℕ) : ℕ :=
  wheels / 2

theorem eighteen_wheel_truck_toll :
  let wheels : ℕ := 18
  let axles : ℕ := axles_count wheels
  toll axles = 5 := by sorry

end NUMINAMATH_CALUDE_eighteen_wheel_truck_toll_l1561_156190


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1561_156106

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2 * Complex.I → z = -1 + (3/2) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1561_156106


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1561_156181

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  first_term : ℝ
  last_term : ℝ
  sum : ℝ
  num_terms : ℕ
  common_diff : ℝ

/-- Theorem stating the properties of the specific arithmetic sequence -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
  (h1 : seq.first_term = 3)
  (h2 : seq.last_term = 50)
  (h3 : seq.sum = 318) :
  seq.common_diff = 47 / 11 := by
  sorry

#check arithmetic_sequence_property

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1561_156181


namespace NUMINAMATH_CALUDE_pages_used_per_day_l1561_156179

/-- Given 5 notebooks with 40 pages each, lasting for 50 days, prove that 4 pages are used per day. -/
theorem pages_used_per_day (num_notebooks : ℕ) (pages_per_notebook : ℕ) (days_lasted : ℕ) :
  num_notebooks = 5 →
  pages_per_notebook = 40 →
  days_lasted = 50 →
  (num_notebooks * pages_per_notebook) / days_lasted = 4 :=
by sorry

end NUMINAMATH_CALUDE_pages_used_per_day_l1561_156179


namespace NUMINAMATH_CALUDE_problem_statement_l1561_156112

noncomputable def f (x : ℝ) := Real.exp x + x - 2
noncomputable def g (x : ℝ) := Real.log x + x^2 - 3

theorem problem_statement (a b : ℝ) (h1 : f a = 0) (h2 : g b = 0) :
  g a < 0 ∧ 0 < f b := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1561_156112


namespace NUMINAMATH_CALUDE_water_volume_in_cylinder_l1561_156161

theorem water_volume_in_cylinder (r : ℝ) (h : r = 2) : 
  let cylinder_base_area := π * r^2
  let ball_volume := (4/3) * π * r^3
  let water_height_with_ball := 2 * r
  let total_volume_with_ball := cylinder_base_area * water_height_with_ball
  let original_water_volume := total_volume_with_ball - ball_volume
  original_water_volume = (16 * π) / 3 := by
sorry

end NUMINAMATH_CALUDE_water_volume_in_cylinder_l1561_156161


namespace NUMINAMATH_CALUDE_teacher_student_arrangements_l1561_156111

/-- The number of arrangements for a teacher and students in a row --/
def arrangements (n : ℕ) : ℕ :=
  (n - 2) * n.factorial

/-- The problem statement --/
theorem teacher_student_arrangements :
  arrangements 6 = 480 := by
  sorry

end NUMINAMATH_CALUDE_teacher_student_arrangements_l1561_156111


namespace NUMINAMATH_CALUDE_bisector_line_equation_chord_length_at_pi_over_4_l1561_156115

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 24 + y^2 / 12 = 1

-- Define point M
def M : ℝ × ℝ := (3, 1)

-- Define a line passing through M
def line_through_M (m : ℝ) (x y : ℝ) : Prop :=
  y - M.2 = m * (x - M.1)

-- Define the intersection points of a line with the ellipse
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ line_through_M m p.1 p.2}

-- Part I: Equation of line when M bisects AB
theorem bisector_line_equation :
  ∃ (A B : ℝ × ℝ) (m : ℝ),
    A ∈ intersection_points m ∧
    B ∈ intersection_points m ∧
    M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
    line_through_M (-3/2) M.1 M.2 :=
sorry

-- Part II: Length of AB when angle of inclination is π/4
theorem chord_length_at_pi_over_4 :
  ∃ (A B : ℝ × ℝ),
    A ∈ intersection_points 1 ∧
    B ∈ intersection_points 1 →
    ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2) = 16/(3*2^(1/2)) :=
sorry

end NUMINAMATH_CALUDE_bisector_line_equation_chord_length_at_pi_over_4_l1561_156115


namespace NUMINAMATH_CALUDE_compare_roots_l1561_156124

theorem compare_roots : (2 * Real.sqrt 6 < 5) ∧ (-Real.sqrt 5 < -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_compare_roots_l1561_156124


namespace NUMINAMATH_CALUDE_worker_speed_ratio_l1561_156195

/-- Given two workers a and b, where a is k times as fast as b, prove that k = 3 
    under the given conditions. -/
theorem worker_speed_ratio (k : ℝ) : 
  (∃ (rate_b : ℝ), 
    (k * rate_b + rate_b = 1 / 30) ∧ 
    (k * rate_b = 1 / 40)) → 
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_worker_speed_ratio_l1561_156195
