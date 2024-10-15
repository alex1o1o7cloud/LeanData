import Mathlib

namespace NUMINAMATH_CALUDE_min_box_height_l242_24243

def box_height (x : ℝ) : ℝ := 2 * x + 2

def box_surface_area (x : ℝ) : ℝ := 9 * x^2 + 8 * x

theorem min_box_height :
  ∃ (x : ℝ), 
    x > 0 ∧
    box_surface_area x ≥ 110 ∧
    (∀ y : ℝ, y > 0 → box_surface_area y ≥ 110 → x ≤ y) ∧
    box_height x = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_box_height_l242_24243


namespace NUMINAMATH_CALUDE_least_integer_with_2035_divisors_l242_24278

/-- The number of divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is the least positive integer with exactly 2035 distinct positive divisors -/
def n : ℕ := sorry

/-- m and k are integers such that n = m * 6^k and 6 is not a divisor of m -/
def m : ℕ := sorry
def k : ℕ := sorry

theorem least_integer_with_2035_divisors :
  num_divisors n = 2035 ∧
  n = m * 6^k ∧
  ¬(6 ∣ m) ∧
  ∀ i : ℕ, i < n → num_divisors i < 2035 →
  m + k = 26 := by sorry

end NUMINAMATH_CALUDE_least_integer_with_2035_divisors_l242_24278


namespace NUMINAMATH_CALUDE_volumes_equal_l242_24242

-- Define the region for V₁
def region_V1 (x y : ℝ) : Prop :=
  (x^2 = 4*y ∨ x^2 = -4*y) ∧ x ≥ -4 ∧ x ≤ 4

-- Define the region for V₂
def region_V2 (x y : ℝ) : Prop :=
  x^2 * y^2 ≤ 16 ∧ x^2 + (y-2)^2 ≥ 4 ∧ x^2 + (y+2)^2 ≥ 4

-- Define the volume of revolution around y-axis
noncomputable def volume_of_revolution (region : ℝ → ℝ → Prop) : ℝ :=
  sorry

-- State the theorem
theorem volumes_equal :
  volume_of_revolution region_V1 = volume_of_revolution region_V2 :=
sorry

end NUMINAMATH_CALUDE_volumes_equal_l242_24242


namespace NUMINAMATH_CALUDE_max_value_of_expression_l242_24274

theorem max_value_of_expression (x y : ℝ) (h : x * y > 0) :
  (∃ (z : ℝ), ∀ (a b : ℝ), a * b > 0 → x / (x + y) + 2 * y / (x + 2 * y) ≤ z) ∧
  (x / (x + y) + 2 * y / (x + 2 * y) ≤ 4 - 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l242_24274


namespace NUMINAMATH_CALUDE_impossible_shape_l242_24273

/-- Represents a square sheet of paper -/
structure Paper :=
  (side : ℝ)
  (is_square : side > 0)

/-- Represents a folded paper -/
structure FoldedPaper :=
  (paper : Paper)
  (folds : ℕ)
  (is_valid : folds ≤ 2)

/-- Represents a shape cut from the paper -/
inductive Shape
  | CrossesBothFolds
  | CrossesOneFold
  | CrossesNoFolds
  | ContainsCenter

/-- Represents a cut made on the folded paper -/
structure Cut :=
  (folded_paper : FoldedPaper)
  (resulting_shape : Shape)

/-- Theorem stating that a shape crossing both folds without containing the center is impossible -/
theorem impossible_shape (p : Paper) (fp : FoldedPaper) (c : Cut) :
  fp.paper = p →
  fp.folds = 2 →
  c.folded_paper = fp →
  c.resulting_shape = Shape.CrossesBothFolds →
  ¬(c.resulting_shape = Shape.ContainsCenter) →
  False :=
sorry

end NUMINAMATH_CALUDE_impossible_shape_l242_24273


namespace NUMINAMATH_CALUDE_sum_odd_integers_21_to_51_l242_24204

/-- The sum of all odd integers from 21 through 51, inclusive, is 576. -/
theorem sum_odd_integers_21_to_51 : 
  (Finset.range 16).sum (fun i => 21 + 2 * i) = 576 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_integers_21_to_51_l242_24204


namespace NUMINAMATH_CALUDE_sequence_21st_term_l242_24259

theorem sequence_21st_term (a : ℕ → ℚ) :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = (4 * a n + 3) / 4) →
  a 1 = 1 →
  a 21 = 16 := by
sorry

end NUMINAMATH_CALUDE_sequence_21st_term_l242_24259


namespace NUMINAMATH_CALUDE_cherries_eaten_l242_24294

theorem cherries_eaten (initial : ℕ) (remaining : ℕ) (h1 : initial = 67) (h2 : remaining = 42) :
  initial - remaining = 25 := by
  sorry

end NUMINAMATH_CALUDE_cherries_eaten_l242_24294


namespace NUMINAMATH_CALUDE_polynomial_value_l242_24270

theorem polynomial_value (a : ℝ) (h : a^2 + 2*a = 1) : 
  2*a^5 + 7*a^4 + 5*a^3 + 2*a^2 + 5*a + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l242_24270


namespace NUMINAMATH_CALUDE_find_N_l242_24236

theorem find_N (X Y Z N : ℝ) 
  (h1 : 0.15 * X = 0.25 * N + Y) 
  (h2 : X + Y = Z) : 
  N = 4.6 * X - 4 * Z := by
sorry

end NUMINAMATH_CALUDE_find_N_l242_24236


namespace NUMINAMATH_CALUDE_cloth_cost_price_l242_24208

/-- Represents the cost price per metre of cloth -/
def cost_price_per_metre (total_length : ℕ) (total_selling_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  (total_selling_price + total_length * loss_per_metre) / total_length

/-- Theorem: Given a cloth of 300 metres sold for Rs. 9000 with a loss of Rs. 6 per metre,
    the cost price for one metre of cloth is Rs. 36 -/
theorem cloth_cost_price :
  cost_price_per_metre 300 9000 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l242_24208


namespace NUMINAMATH_CALUDE_supermarket_product_sales_l242_24260

-- Define the linear function
def sales_quantity (x : ℝ) : ℝ := -2 * x + 200

-- Define the profit function
def profit (x : ℝ) : ℝ := (sales_quantity x) * (x - 60)

-- Define the given data points
def data_points : List (ℝ × ℝ) := [(65, 70), (70, 60), (75, 50), (80, 40)]

theorem supermarket_product_sales :
  -- 1. The function fits the given data points
  (∀ (point : ℝ × ℝ), point ∈ data_points → sales_quantity point.1 = point.2) ∧
  -- 2. The selling price of 70 or 90 dollars per kilogram results in a daily profit of $600
  (profit 70 = 600 ∧ profit 90 = 600) ∧
  -- 3. The maximum daily profit is $800, achieved at a selling price of 80 dollars per kilogram
  (∀ (x : ℝ), profit x ≤ 800) ∧ (profit 80 = 800) :=
by sorry

end NUMINAMATH_CALUDE_supermarket_product_sales_l242_24260


namespace NUMINAMATH_CALUDE_ellipse_intersection_properties_l242_24250

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = x - 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-1, 0)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ellipse p.1 p.2 ∧ line p.1 p.2}

-- Theorem statement
theorem ellipse_intersection_properties :
  let A := (0, -1)
  let B := (4/3, 1/3)
  (A ∈ intersection_points) ∧
  (B ∈ intersection_points) ∧
  (∃ (AB : ℝ), AB = (4 * Real.sqrt 2) / 3 ∧
    AB = Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) ∧
  (∃ (S : ℝ), S = 4/3 ∧
    S = (1/2) * ((4 * Real.sqrt 2) / 3) * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_properties_l242_24250


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l242_24201

theorem complex_on_imaginary_axis (z : ℂ) : 
  Complex.abs (z - 1) = Complex.abs (z + 1) → z.re = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l242_24201


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l242_24293

theorem inscribed_circle_area_ratio (h r b : ℝ) : 
  h > 0 → r > 0 → b > 0 →
  (b + r)^2 + b^2 = h^2 →
  (2 * π * r^2) / ((b + r + h) * r) = 2 * π * r / (2 * b + r + h) := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l242_24293


namespace NUMINAMATH_CALUDE_expression_simplification_l242_24219

theorem expression_simplification (a : ℝ) (h : a = 2 * Real.cos (π / 3) + 1) :
  (a - a^2 / (a + 1)) / (a^2 / (a^2 - 1)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l242_24219


namespace NUMINAMATH_CALUDE_logarithm_domain_l242_24235

theorem logarithm_domain (a : ℝ) : 
  (∀ x : ℝ, x < 2 → ∃ y : ℝ, y = Real.log (a - 3 * x)) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_domain_l242_24235


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_l242_24217

theorem greatest_three_digit_number : ∃ (n : ℕ), 
  n = 982 ∧ 
  100 ≤ n ∧ n ≤ 999 ∧ 
  ∃ (k : ℕ), n = 7 * k + 2 ∧ 
  ∃ (m : ℕ), n = 6 * m + 4 ∧ 
  ∀ (x : ℕ), (100 ≤ x ∧ x ≤ 999 ∧ ∃ (a : ℕ), x = 7 * a + 2 ∧ ∃ (b : ℕ), x = 6 * b + 4) → x ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_l242_24217


namespace NUMINAMATH_CALUDE_sequence_general_term_l242_24255

theorem sequence_general_term (a : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n + 1) →
  (∀ n : ℕ, n ≥ 1 → a n = n^2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l242_24255


namespace NUMINAMATH_CALUDE_distinct_digit_count_is_5040_l242_24209

/-- The number of four-digit integers with distinct digits, including those starting with 0 -/
def distinctDigitCount : ℕ := 10 * 9 * 8 * 7

/-- Theorem stating that the count of four-digit integers with distinct digits is 5040 -/
theorem distinct_digit_count_is_5040 : distinctDigitCount = 5040 := by
  sorry

end NUMINAMATH_CALUDE_distinct_digit_count_is_5040_l242_24209


namespace NUMINAMATH_CALUDE_min_intersection_size_l242_24238

theorem min_intersection_size (total blue_eyes backpack : ℕ) 
  (h_total : total = 35)
  (h_blue : blue_eyes = 18)
  (h_backpack : backpack = 24) :
  blue_eyes + backpack - total ≤ (blue_eyes ⊓ backpack) :=
by sorry

end NUMINAMATH_CALUDE_min_intersection_size_l242_24238


namespace NUMINAMATH_CALUDE_james_savings_l242_24265

/-- Proves that James saved for 4 weeks given the problem conditions --/
theorem james_savings (w : ℕ) : 
  (10 : ℚ) * w - ((10 : ℚ) * w / 2) / 4 = 15 → w = 4 := by
  sorry

end NUMINAMATH_CALUDE_james_savings_l242_24265


namespace NUMINAMATH_CALUDE_data_transmission_time_l242_24229

theorem data_transmission_time (blocks : ℕ) (chunks_per_block : ℕ) (transmission_rate : ℕ) :
  blocks = 80 →
  chunks_per_block = 640 →
  transmission_rate = 160 →
  (blocks * chunks_per_block : ℚ) / transmission_rate = 320 :=
by sorry

end NUMINAMATH_CALUDE_data_transmission_time_l242_24229


namespace NUMINAMATH_CALUDE_homogeneous_polynomial_terms_l242_24248

/-- The number of distinct terms in a homogeneous polynomial -/
def num_distinct_terms (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: The number of distinct terms in a homogeneous polynomial of degree 6 with 5 variables is 210 -/
theorem homogeneous_polynomial_terms :
  num_distinct_terms 6 5 = 210 := by sorry

end NUMINAMATH_CALUDE_homogeneous_polynomial_terms_l242_24248


namespace NUMINAMATH_CALUDE_total_votes_l242_24295

theorem total_votes (veggies : ℕ) (meat : ℕ) (dairy : ℕ) (plant_protein : ℕ)
  (h1 : veggies = 337)
  (h2 : meat = 335)
  (h3 : dairy = 274)
  (h4 : plant_protein = 212) :
  veggies + meat + dairy + plant_protein = 1158 :=
by sorry

end NUMINAMATH_CALUDE_total_votes_l242_24295


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l242_24290

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 5, 7}
def B : Set Nat := {3, 4, 5}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {1, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l242_24290


namespace NUMINAMATH_CALUDE_print_time_calculation_l242_24221

/-- Represents a printer with warm-up time and printing speed -/
structure Printer where
  warmupTime : ℕ
  pagesPerMinute : ℕ

/-- Calculates the total time required to print a given number of pages -/
def totalPrintTime (printer : Printer) (pages : ℕ) : ℕ :=
  printer.warmupTime + (pages + printer.pagesPerMinute - 1) / printer.pagesPerMinute

theorem print_time_calculation (printer : Printer) (pages : ℕ) :
  printer.warmupTime = 2 →
  printer.pagesPerMinute = 15 →
  pages = 225 →
  totalPrintTime printer pages = 17 :=
by
  sorry

#eval totalPrintTime ⟨2, 15⟩ 225

end NUMINAMATH_CALUDE_print_time_calculation_l242_24221


namespace NUMINAMATH_CALUDE_inequality_solution_count_l242_24289

theorem inequality_solution_count : ∃! (n : ℤ), (n - 2) * (n + 4) * (n - 3) < 0 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_count_l242_24289


namespace NUMINAMATH_CALUDE_orange_juice_profit_l242_24276

/-- Represents the number of orange trees each sister has -/
def trees_per_sister : ℕ := 110

/-- Represents the number of oranges Gabriela's trees produce per tree -/
def gabriela_oranges_per_tree : ℕ := 600

/-- Represents the number of oranges Alba's trees produce per tree -/
def alba_oranges_per_tree : ℕ := 400

/-- Represents the number of oranges Maricela's trees produce per tree -/
def maricela_oranges_per_tree : ℕ := 500

/-- Represents the number of oranges needed to make one cup of juice -/
def oranges_per_cup : ℕ := 3

/-- Represents the price of one cup of juice in dollars -/
def price_per_cup : ℕ := 4

/-- Theorem stating the total money earned from selling orange juice -/
theorem orange_juice_profit : 
  (trees_per_sister * gabriela_oranges_per_tree + 
   trees_per_sister * alba_oranges_per_tree + 
   trees_per_sister * maricela_oranges_per_tree) / oranges_per_cup * price_per_cup = 220000 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_profit_l242_24276


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l242_24230

/-- Two 2D vectors are parallel if the cross product of their coordinates is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given two vectors a and b, prove that if they are parallel, then x = 3 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (3, x)
  let b : ℝ × ℝ := (2, x - 1)
  are_parallel a b → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l242_24230


namespace NUMINAMATH_CALUDE_largest_n_with_negative_sum_l242_24239

theorem largest_n_with_negative_sum 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arith : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n) 
  (h_sum : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2) 
  (h_a6_neg : a 6 < 0) 
  (h_a4_a9_pos : a 4 + a 9 > 0) : 
  (∀ n > 11, S n ≥ 0) ∧ S 11 < 0 :=
sorry

end NUMINAMATH_CALUDE_largest_n_with_negative_sum_l242_24239


namespace NUMINAMATH_CALUDE_perfect_square_identification_l242_24280

theorem perfect_square_identification (a b : ℝ) : 
  (∃ x : ℝ, a^2 - 4*a + 4 = x^2) ∧ 
  (¬∃ x : ℝ, 1 + 4*a^2 = x^2) ∧ 
  (¬∃ x : ℝ, 4*b^2 + 4*b - 1 = x^2) ∧ 
  (¬∃ x : ℝ, a^2 + a*b + b^2 = x^2) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_identification_l242_24280


namespace NUMINAMATH_CALUDE_angle_sum_triangle_l242_24286

theorem angle_sum_triangle (A B C : ℝ) (h : A + B = 110) : C = 70 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_triangle_l242_24286


namespace NUMINAMATH_CALUDE_league_games_l242_24241

theorem league_games (n : ℕ) (k : ℕ) (m : ℕ) (h1 : n = 30) (h2 : k = 2) (h3 : m = 6) :
  (n.choose k) * m = 2610 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l242_24241


namespace NUMINAMATH_CALUDE_different_signs_implies_range_l242_24234

theorem different_signs_implies_range (m : ℝ) : 
  ((2 - m) * (|m| - 3) < 0) → ((-3 < m ∧ m < 2) ∨ m > 3) := by
sorry

end NUMINAMATH_CALUDE_different_signs_implies_range_l242_24234


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l242_24220

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < B → B < π →
  (a - b + c) * (a + b + c) = 3 * a * c →
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  B = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l242_24220


namespace NUMINAMATH_CALUDE_same_color_probability_l242_24284

def total_balls : ℕ := 3
def red_balls : ℕ := 2
def white_balls : ℕ := 1

def prob_red : ℚ := red_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

theorem same_color_probability :
  let prob_same_color := prob_red^2 + prob_white^2
  prob_same_color = 5/9 := by sorry

end NUMINAMATH_CALUDE_same_color_probability_l242_24284


namespace NUMINAMATH_CALUDE_rental_cost_difference_l242_24212

/-- Calculates the difference in rental costs between a ski boat and a sailboat for a given duration. -/
theorem rental_cost_difference 
  (sailboat_cost_per_day : ℕ)
  (ski_boat_cost_per_hour : ℕ)
  (hours_per_day : ℕ)
  (num_days : ℕ)
  (h1 : sailboat_cost_per_day = 60)
  (h2 : ski_boat_cost_per_hour = 80)
  (h3 : hours_per_day = 3)
  (h4 : num_days = 2) :
  ski_boat_cost_per_hour * hours_per_day * num_days - sailboat_cost_per_day * num_days = 360 :=
by
  sorry

#check rental_cost_difference

end NUMINAMATH_CALUDE_rental_cost_difference_l242_24212


namespace NUMINAMATH_CALUDE_diamond_zero_not_always_double_l242_24252

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 2 * |x - y|

-- Statement to prove
theorem diamond_zero_not_always_double : ¬ (∀ x : ℝ, diamond x 0 = 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_diamond_zero_not_always_double_l242_24252


namespace NUMINAMATH_CALUDE_courier_delivery_patterns_l242_24249

/-- Represents the number of acceptable delivery patterns for n offices -/
def P : ℕ → ℕ
| 0 => 1  -- Base case: only one way to deliver to 0 offices
| 1 => 2  -- Can either deliver or not deliver to 1 office
| 2 => 4  -- All combinations for 2 offices
| 3 => 8  -- All combinations for 3 offices
| 4 => 15 -- All combinations for 4 offices, excluding all non-deliveries
| n + 5 => P (n + 4) + P (n + 3) + P (n + 2) + P (n + 1)

theorem courier_delivery_patterns :
  P 12 = 927 := by
  sorry


end NUMINAMATH_CALUDE_courier_delivery_patterns_l242_24249


namespace NUMINAMATH_CALUDE_product_cde_value_l242_24207

theorem product_cde_value (a b c d e f : ℝ) 
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 0.75) :
  c * d * e = 1000 := by
  sorry

end NUMINAMATH_CALUDE_product_cde_value_l242_24207


namespace NUMINAMATH_CALUDE_product_fraction_inequality_l242_24257

theorem product_fraction_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) 
  (sum_eq_two : a + b + c = 2) : 
  (a / (1 - a)) * (b / (1 - b)) * (c / (1 - c)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_product_fraction_inequality_l242_24257


namespace NUMINAMATH_CALUDE_lottery_jackpot_probability_l242_24206

/-- The number of balls for the MegaBall draw -/
def megaBallCount : ℕ := 30

/-- The number of balls for the WinnerBalls draw -/
def winnerBallCount : ℕ := 49

/-- The number of WinnerBalls drawn -/
def winnerBallsDraw : ℕ := 6

/-- The probability of winning the jackpot in the lottery -/
def jackpotProbability : ℚ := 1 / 419514480

/-- Theorem stating that the probability of winning the jackpot in the given lottery system
    is equal to 1/419,514,480 -/
theorem lottery_jackpot_probability :
  (1 : ℚ) / megaBallCount * (1 : ℚ) / (winnerBallCount.choose winnerBallsDraw) = jackpotProbability := by
  sorry


end NUMINAMATH_CALUDE_lottery_jackpot_probability_l242_24206


namespace NUMINAMATH_CALUDE_journey_time_proof_l242_24298

/-- Proves that the time taken to complete a 224 km journey, where the first half is traveled at 21 km/hr and the second half at 24 km/hr, is equal to 10 hours. -/
theorem journey_time_proof (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_distance = 224 →
  speed1 = 21 →
  speed2 = 24 →
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 10 := by
  sorry

#check journey_time_proof

end NUMINAMATH_CALUDE_journey_time_proof_l242_24298


namespace NUMINAMATH_CALUDE_set_operations_l242_24210

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B : Set ℝ := {x | x^2 - 4*x ≤ 0}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 3}) ∧
  (A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 4}) ∧
  ((Aᶜ ∩ Bᶜ) = {x : ℝ | x < -1 ∨ x > 4}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l242_24210


namespace NUMINAMATH_CALUDE_investment_sum_l242_24277

/-- Given a sum invested at different interest rates, prove the sum equals 8400 --/
theorem investment_sum (P : ℝ) : 
  (P * 0.15 * 2) - (P * 0.10 * 2) = 840 → P = 8400 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l242_24277


namespace NUMINAMATH_CALUDE_flag_arrangement_problem_l242_24224

/-- Number of blue flags -/
def blue_flags : ℕ := 10

/-- Number of green flags -/
def green_flags : ℕ := 9

/-- Total number of flags -/
def total_flags : ℕ := blue_flags + green_flags

/-- Number of flagpoles -/
def flagpoles : ℕ := 2

/-- Function to calculate the number of arrangements -/
def calculate_arrangements (a b : ℕ) : ℕ :=
  (a + 1) * Nat.choose (a + 2) b - 2 * Nat.choose (a + 1) b

/-- Theorem stating the result of the flag arrangement problem -/
theorem flag_arrangement_problem :
  calculate_arrangements blue_flags green_flags % 1000 = 310 := by
  sorry

end NUMINAMATH_CALUDE_flag_arrangement_problem_l242_24224


namespace NUMINAMATH_CALUDE_leifs_oranges_l242_24244

theorem leifs_oranges (apples : ℕ) (oranges : ℕ) : apples = 14 → oranges = apples + 10 → oranges = 24 := by
  sorry

end NUMINAMATH_CALUDE_leifs_oranges_l242_24244


namespace NUMINAMATH_CALUDE_students_registered_correct_registration_l242_24272

theorem students_registered (students_yesterday : ℕ) (absent_today : ℕ) : ℕ :=
  let twice_yesterday := 2 * students_yesterday
  let ten_percent := twice_yesterday / 10
  let attending_today := twice_yesterday - ten_percent
  let total_registered := attending_today + absent_today
  total_registered

theorem correct_registration : students_registered 70 30 = 156 := by
  sorry

end NUMINAMATH_CALUDE_students_registered_correct_registration_l242_24272


namespace NUMINAMATH_CALUDE_right_triangle_shorter_side_l242_24292

/-- A right triangle with perimeter 40 and area 30 has a shorter side of length 5.25 -/
theorem right_triangle_shorter_side : ∃ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive side lengths
  a^2 + b^2 = c^2 ∧        -- right triangle (Pythagorean theorem)
  a + b + c = 40 ∧         -- perimeter is 40
  (1/2) * a * b = 30 ∧     -- area is 30
  (a = 5.25 ∨ b = 5.25) :=  -- one shorter side is 5.25
by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_side_l242_24292


namespace NUMINAMATH_CALUDE_sum_of_quotient_dividend_divisor_l242_24279

theorem sum_of_quotient_dividend_divisor (n d : ℕ) (h1 : n = 45) (h2 : d = 3) :
  n / d + n + d = 63 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_quotient_dividend_divisor_l242_24279


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_sixty_integers_from_three_l242_24262

def arithmetic_sequence_sum (a₁ n d : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_mean_of_sixty_integers_from_three (a₁ n : ℕ) (h₁ : a₁ = 3) (h₂ : n = 60) :
  (arithmetic_sequence_sum a₁ n 1 : ℚ) / n = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_sixty_integers_from_three_l242_24262


namespace NUMINAMATH_CALUDE_rectangle_perimeter_10_l242_24281

/-- A rectangle with sides a and b. -/
structure Rectangle where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0

/-- The perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.a + r.b)

/-- The sum of three sides of a rectangle. -/
def sum_three_sides (r : Rectangle) : Set ℝ := {2 * r.a + r.b, r.a + 2 * r.b}

/-- Theorem stating that there exists a rectangle with perimeter 10,
    given that the sum of the lengths of three different sides can be equal to 6 or 9. -/
theorem rectangle_perimeter_10 :
  ∃ r : Rectangle, (6 ∈ sum_three_sides r ∨ 9 ∈ sum_three_sides r) ∧ perimeter r = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_10_l242_24281


namespace NUMINAMATH_CALUDE_triangle_is_obtuse_l242_24233

theorem triangle_is_obtuse (A : ℝ) (h1 : 0 < A ∧ A < π) 
  (h2 : Real.sin A + Real.cos A = 7/12) : 
  π/2 < A ∧ A < π :=
by sorry

end NUMINAMATH_CALUDE_triangle_is_obtuse_l242_24233


namespace NUMINAMATH_CALUDE_digits_of_expression_l242_24264

theorem digits_of_expression : ∃ n : ℕ, n = 12 ∧ n = (Nat.digits 10 (2^15 * 5^12 - 10^5)).length := by
  sorry

end NUMINAMATH_CALUDE_digits_of_expression_l242_24264


namespace NUMINAMATH_CALUDE_equation_solution_l242_24237

theorem equation_solution (x : ℝ) (h : x > 1) :
  (x^2 / (x - 1)) + Real.sqrt (x - 1) + (Real.sqrt (x - 1) / x^2) =
  ((x - 1) / x^2) + (1 / Real.sqrt (x - 1)) + (x^2 / Real.sqrt (x - 1)) ↔
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l242_24237


namespace NUMINAMATH_CALUDE_frog_flies_consumption_l242_24291

/-- Proves that each frog needs to eat 30 flies per day in a swamp ecosystem -/
theorem frog_flies_consumption
  (fish_frog_consumption : ℕ) -- Number of frogs each fish eats per day
  (gharial_fish_consumption : ℕ) -- Number of fish each gharial eats per day
  (gharial_count : ℕ) -- Number of gharials in the swamp
  (total_flies_eaten : ℕ) -- Total number of flies eaten per day
  (h1 : fish_frog_consumption = 8)
  (h2 : gharial_fish_consumption = 15)
  (h3 : gharial_count = 9)
  (h4 : total_flies_eaten = 32400) :
  total_flies_eaten / (gharial_count * gharial_fish_consumption * fish_frog_consumption) = 30 := by
  sorry


end NUMINAMATH_CALUDE_frog_flies_consumption_l242_24291


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l242_24215

theorem contrapositive_equivalence :
  (∀ a : ℝ, a > 1 → a^2 > 1) ↔ (∀ a : ℝ, a^2 ≤ 1 → a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l242_24215


namespace NUMINAMATH_CALUDE_average_rate_for_trip_l242_24223

/-- Given a trip with the following conditions:
  - Total distance is 640 miles
  - First half is driven at 80 miles per hour
  - Second half takes 200% longer than the first half
  Prove that the average rate for the entire trip is 40 miles per hour -/
theorem average_rate_for_trip (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  (total_distance / (total_distance / (2 * first_half_speed) * (1 + second_half_time_factor))) = 40 :=
by sorry

end NUMINAMATH_CALUDE_average_rate_for_trip_l242_24223


namespace NUMINAMATH_CALUDE_prob_laurent_ge_2chloe_l242_24225

/-- Represents a uniform distribution over a real interval -/
structure UniformDist (a b : ℝ) where
  (a_le_b : a ≤ b)

/-- The probability that a random variable from distribution Y is at least twice 
    a random variable from distribution X -/
noncomputable def prob_y_ge_2x (X : UniformDist 0 1000) (Y : UniformDist 0 2000) : ℝ :=
  (1000 * 1000 / 2) / (1000 * 2000)

/-- Theorem stating that the probability of Laurent's number being at least 
    twice Chloe's number is 1/4 -/
theorem prob_laurent_ge_2chloe :
  ∀ (X : UniformDist 0 1000) (Y : UniformDist 0 2000),
  prob_y_ge_2x X Y = 1/4 := by sorry

end NUMINAMATH_CALUDE_prob_laurent_ge_2chloe_l242_24225


namespace NUMINAMATH_CALUDE_f_composition_negative_three_equals_zero_l242_24227

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x + 2/x - 3
  else Real.log (x^2 + 1) / Real.log 10

-- State the theorem
theorem f_composition_negative_three_equals_zero :
  f (f (-3)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_three_equals_zero_l242_24227


namespace NUMINAMATH_CALUDE_perfect_squares_exist_l242_24275

theorem perfect_squares_exist : ∃ (a b c d : ℤ),
  (∃ (x : ℤ), (a + b) = x^2) ∧
  (∃ (y : ℤ), (a + c) = y^2) ∧
  (∃ (z : ℤ), (a + d) = z^2) ∧
  (∃ (w : ℤ), (b + c) = w^2) ∧
  (∃ (v : ℤ), (b + d) = v^2) ∧
  (∃ (u : ℤ), (c + d) = u^2) ∧
  (∃ (t : ℤ), (a + b + c + d) = t^2) :=
by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_exist_l242_24275


namespace NUMINAMATH_CALUDE_tangent_curves_l242_24256

/-- The value of α for which e^x is tangent to αx^2 -/
theorem tangent_curves (f g : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = Real.exp x) →
  (∀ x, g x = α * x^2) →
  (∃ x, f x = g x ∧ deriv f x = deriv g x) →
  α = Real.exp 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_curves_l242_24256


namespace NUMINAMATH_CALUDE_mountain_height_proof_l242_24261

def mountain_height (h : ℝ) : Prop :=
  h > 7900 ∧ h < 8000

theorem mountain_height_proof (h : ℝ) 
  (peter_false : ¬(h ≥ 8000))
  (mary_false : ¬(h ≤ 7900))
  (john_false : ¬(h ≤ 7500)) :
  mountain_height h :=
sorry

end NUMINAMATH_CALUDE_mountain_height_proof_l242_24261


namespace NUMINAMATH_CALUDE_sqrt_square_sum_zero_implies_diff_l242_24251

theorem sqrt_square_sum_zero_implies_diff (x y : ℝ) : 
  Real.sqrt (8 - x) + (y + 4)^2 = 0 → x - y = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_sum_zero_implies_diff_l242_24251


namespace NUMINAMATH_CALUDE_coloring_problem_l242_24263

/-- Represents the number of objects colored by each person -/
def objects_per_person (total_colors : ℕ) (num_people : ℕ) : ℕ :=
  total_colors / num_people

/-- Represents the total number of objects colored -/
def total_objects (objects_per_person : ℕ) (num_people : ℕ) : ℕ :=
  objects_per_person * num_people

theorem coloring_problem (total_colors : ℕ) (num_people : ℕ) 
  (h1 : total_colors = 24) 
  (h2 : num_people = 3) :
  total_objects (objects_per_person total_colors num_people) num_people = 24 := by
sorry

end NUMINAMATH_CALUDE_coloring_problem_l242_24263


namespace NUMINAMATH_CALUDE_probability_theorem_l242_24258

def total_marbles : ℕ := 30
def red_marbles : ℕ := 15
def blue_marbles : ℕ := 10
def green_marbles : ℕ := 5
def marbles_selected : ℕ := 4

def probability_two_red_one_blue_one_green : ℚ :=
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 1) /
  Nat.choose total_marbles marbles_selected

theorem probability_theorem :
  probability_two_red_one_blue_one_green = 350 / 1827 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l242_24258


namespace NUMINAMATH_CALUDE_money_problem_l242_24245

theorem money_problem (p q r s t : ℚ) : 
  p = q + r + 35 →
  q = (2/5) * p →
  r = (1/7) * p →
  s = 2 * p →
  t = (1/2) * (q + r) →
  p + q + r + s + t = 291.03125 := by
sorry

end NUMINAMATH_CALUDE_money_problem_l242_24245


namespace NUMINAMATH_CALUDE_sarahs_number_l242_24205

theorem sarahs_number :
  ∃! n : ℕ, 1000 < n ∧ n < 3000 ∧ 144 ∣ n ∧ 45 ∣ n ∧ n = 2880 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_number_l242_24205


namespace NUMINAMATH_CALUDE_base_4_to_16_digits_l242_24211

theorem base_4_to_16_digits : ∀ n : ℕ,
  (4^4 ≤ n) ∧ (n < 4^5) →
  (16^2 ≤ n) ∧ (n < 16^3) :=
by sorry

end NUMINAMATH_CALUDE_base_4_to_16_digits_l242_24211


namespace NUMINAMATH_CALUDE_roberto_outfits_l242_24203

/-- The number of pairs of trousers Roberto has -/
def trousers : ℕ := 5

/-- The number of shirts Roberto has -/
def shirts : ℕ := 6

/-- The number of jackets Roberto has -/
def jackets : ℕ := 4

/-- The number of pairs of shoes Roberto has -/
def shoes : ℕ := 3

/-- The number of jackets with shoe restrictions -/
def restricted_jackets : ℕ := 1

/-- The number of shoes that can be worn with the restricted jacket -/
def shoes_per_restricted_jacket : ℕ := 2

/-- The total number of outfits Roberto can put together -/
def total_outfits : ℕ := trousers * shirts * (
  (jackets - restricted_jackets) * shoes +
  restricted_jackets * shoes_per_restricted_jacket
)

theorem roberto_outfits :
  total_outfits = 330 :=
by sorry

end NUMINAMATH_CALUDE_roberto_outfits_l242_24203


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l242_24288

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 3) = 7 → x = 46 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l242_24288


namespace NUMINAMATH_CALUDE_circle_intersection_range_l242_24254

-- Define the circles
def circle1 (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 25

-- Define the theorem
theorem circle_intersection_range (a : ℝ) :
  (a ≥ 0) →
  (∃ x y : ℝ, circle1 a x y ∧ circle2 x y) →
  2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l242_24254


namespace NUMINAMATH_CALUDE_age_difference_correct_l242_24266

/-- The age difference between Emma and her sister -/
def age_difference : ℕ := 9

/-- Emma's age when her sister is 56 -/
def emma_future_age : ℕ := 47

/-- Emma's sister's age when Emma is 47 -/
def sister_future_age : ℕ := 56

/-- Proof that the age difference is correct -/
theorem age_difference_correct : 
  emma_future_age + age_difference = sister_future_age :=
by sorry

end NUMINAMATH_CALUDE_age_difference_correct_l242_24266


namespace NUMINAMATH_CALUDE_symmetry_of_lines_l242_24232

/-- Given two lines in a 2D plane, this function returns true if they are symmetric with respect to a third line. -/
def are_symmetric_lines (line1 line2 axis : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, line1 x y ↔ line2 (y - 1) (x + 1)

/-- The line y = 2x + 3 -/
def line1 (x y : ℝ) : Prop := y = 2 * x + 3

/-- The line y = x + 1 (the axis of symmetry) -/
def axis (x y : ℝ) : Prop := y = x + 1

/-- The line x = 2y (which is equivalent to x - 2y = 0) -/
def line2 (x y : ℝ) : Prop := x = 2 * y

theorem symmetry_of_lines : are_symmetric_lines line1 line2 axis := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_lines_l242_24232


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l242_24268

theorem largest_angle_in_triangle (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines holds
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given conditions
  Real.cos A = 3/4 →
  C = 2 * A →
  -- Conclusion
  C > A ∧ C > B :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l242_24268


namespace NUMINAMATH_CALUDE_no_ten_digit_divisor_with_different_digits_l242_24269

/-- The number consisting of 1000 ones -/
def number_of_ones : ℕ := 10^1000 - 1

/-- A function to check if a natural number has exactly 10 digits -/
def has_ten_digits (n : ℕ) : Prop := 10^9 ≤ n ∧ n < 10^10

/-- A function to check if all digits in a number are different -/
def all_digits_different (n : ℕ) : Prop :=
  ∀ d₁ d₂, 0 ≤ d₁ ∧ d₁ < 10 ∧ 0 ≤ d₂ ∧ d₂ < 10 → d₁ ≠ d₂ → (n / 10^d₁ % 10) ≠ (n / 10^d₂ % 10)

/-- The main theorem stating that the number of 1000 ones has no ten-digit divisor with all different digits -/
theorem no_ten_digit_divisor_with_different_digits : 
  ¬ ∃ (d : ℕ), d ∣ number_of_ones ∧ has_ten_digits d ∧ all_digits_different d := by
  sorry

end NUMINAMATH_CALUDE_no_ten_digit_divisor_with_different_digits_l242_24269


namespace NUMINAMATH_CALUDE_computer_factory_month_days_l242_24216

/-- Proves that given a factory producing 5376 computers per month at a constant rate,
    and 4 computers built every 30 minutes, the number of days in the month is 28. -/
theorem computer_factory_month_days : 
  ∀ (computers_per_month : ℕ) (computers_per_30min : ℕ),
    computers_per_month = 5376 →
    computers_per_30min = 4 →
    (computers_per_month / (48 * computers_per_30min) : ℕ) = 28 := by
  sorry

#check computer_factory_month_days

end NUMINAMATH_CALUDE_computer_factory_month_days_l242_24216


namespace NUMINAMATH_CALUDE_friendly_numbers_solution_l242_24202

/-- Two rational numbers are friendly if their sum is 66 -/
def friendly (m n : ℚ) : Prop := m + n = 66

/-- Given that 7x and -18 are friendly numbers, prove that x = 12 -/
theorem friendly_numbers_solution : 
  ∀ x : ℚ, friendly (7 * x) (-18) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_friendly_numbers_solution_l242_24202


namespace NUMINAMATH_CALUDE_race_outcomes_l242_24282

theorem race_outcomes (n : ℕ) (k : ℕ) (h : n = 6 ∧ k = 4) : 
  n * (n - 1) * (n - 2) * (n - 3) = 360 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_l242_24282


namespace NUMINAMATH_CALUDE_distribute_five_into_four_l242_24287

def distribute_objects (n : ℕ) (k : ℕ) : ℕ :=
  if n < k then 0
  else (n - k) + 1

theorem distribute_five_into_four :
  distribute_objects 5 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_into_four_l242_24287


namespace NUMINAMATH_CALUDE_map_area_ratio_map_area_ratio_not_scale_l242_24271

/-- Represents the scale of a map --/
structure MapScale where
  ratio : ℚ

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℚ
  width : ℚ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℚ := r.length * r.width

/-- Theorem: For a map with scale 1:500, the ratio of map area to actual area is 1:250000 --/
theorem map_area_ratio (scale : MapScale) (map_rect : Rectangle) (actual_rect : Rectangle)
    (h_scale : scale.ratio = 1 / 500)
    (h_length : map_rect.length * 500 = actual_rect.length)
    (h_width : map_rect.width * 500 = actual_rect.width) :
    area map_rect / area actual_rect = 1 / 250000 := by
  sorry

/-- The ratio of map area to actual area is not 1:500 --/
theorem map_area_ratio_not_scale (scale : MapScale) (map_rect : Rectangle) (actual_rect : Rectangle)
    (h_scale : scale.ratio = 1 / 500)
    (h_length : map_rect.length * 500 = actual_rect.length)
    (h_width : map_rect.width * 500 = actual_rect.width) :
    area map_rect / area actual_rect ≠ 1 / 500 := by
  sorry

end NUMINAMATH_CALUDE_map_area_ratio_map_area_ratio_not_scale_l242_24271


namespace NUMINAMATH_CALUDE_josh_marbles_remaining_l242_24297

/-- The number of marbles Josh has remaining after losing some. -/
def remaining_marbles (initial : ℝ) (lost : ℝ) : ℝ :=
  initial - lost

/-- Theorem stating that Josh has 7.75 marbles remaining. -/
theorem josh_marbles_remaining :
  remaining_marbles 19.5 11.75 = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_remaining_l242_24297


namespace NUMINAMATH_CALUDE_yanni_remaining_money_l242_24299

/-- Calculates the remaining money in cents after Yanni's transactions --/
def remaining_money_in_cents (initial_money : ℚ) (mother_gave : ℚ) (found_money : ℚ) (toy_cost : ℚ) : ℕ :=
  let total_money := initial_money + mother_gave + found_money
  let remaining_money := total_money - toy_cost
  (remaining_money * 100).floor.toNat

/-- Proves that Yanni has 15 cents left after his transactions --/
theorem yanni_remaining_money :
  remaining_money_in_cents 0.85 0.40 0.50 1.60 = 15 := by
  sorry

end NUMINAMATH_CALUDE_yanni_remaining_money_l242_24299


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l242_24296

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℚ := 3 * n

-- Define the sum of the first n terms of {a_n}
def S (n : ℕ) : ℚ := n * (3 + a n) / 2

-- Define the sequence {c_n}
def c (n : ℕ) : ℚ := 3 / (2 * S n)

-- Define the sum of the first n terms of {c_n}
def T (n : ℕ) : ℚ := n / (n + 1)

theorem arithmetic_sequence_properties :
  (a 1 = 3) ∧
  (a 3 + S 3 = 27) ∧
  (∀ n : ℕ, a n = 3 * n) ∧
  (∀ n : ℕ, T n = n / (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l242_24296


namespace NUMINAMATH_CALUDE_quadratic_max_l242_24240

/-- Given a quadratic function f(x) = ax^2 + bx + c where a < 0,
    and x₀ satisfies 2ax + b = 0, then for all x ∈ ℝ, f(x) ≤ f(x₀) -/
theorem quadratic_max (a b c : ℝ) (x₀ : ℝ) (h₁ : a < 0) (h₂ : 2 * a * x₀ + b = 0) :
  ∀ x : ℝ, a * x^2 + b * x + c ≤ a * x₀^2 + b * x₀ + c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_l242_24240


namespace NUMINAMATH_CALUDE_cubic_inequality_l242_24253

theorem cubic_inequality (x : ℝ) : x^3 - 10*x^2 > -25*x ↔ (0 < x ∧ x < 5) ∨ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l242_24253


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l242_24285

theorem arctan_equation_solution :
  ∀ x : ℝ, 2 * Real.arctan (1/5) + Real.arctan (1/15) + Real.arctan (1/x) = π/3 → x = -49 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l242_24285


namespace NUMINAMATH_CALUDE_quadratic_roots_constraint_l242_24200

theorem quadratic_roots_constraint (x y : ℤ) : 
  (∃ α β : ℝ, α^2 + β^2 < 4 ∧ ∀ t : ℝ, t^2 + x*t + y = 0 ↔ t = α ∨ t = β) →
  (x = -2 ∧ y = 1) ∨
  (x = -1 ∧ y = -1) ∨
  (x = -1 ∧ y = 0) ∨
  (x = 0 ∧ y = -1) ∨
  (x = 0 ∧ y = 0) ∨
  (x = 1 ∧ y = 0) ∨
  (x = 1 ∧ y = -1) ∨
  (x = 2 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_constraint_l242_24200


namespace NUMINAMATH_CALUDE_vasya_petya_notebooks_different_l242_24247

theorem vasya_petya_notebooks_different (S : Finset ℝ) (h : S.card = 10) :
  let vasya_set := Finset.image (fun (p : ℝ × ℝ) => (p.1 - p.2)^2) (S.product S)
  let petya_set := Finset.image (fun (p : ℝ × ℝ) => |p.1^2 - p.2^2|) (S.product S)
  vasya_set ≠ petya_set :=
by sorry

end NUMINAMATH_CALUDE_vasya_petya_notebooks_different_l242_24247


namespace NUMINAMATH_CALUDE_trajectory_equation_l242_24246

/-- The trajectory of point M(x, y) with distance ratio 2 from F(4,0) and line x = 3 -/
theorem trajectory_equation (x y : ℝ) : 
  (((x - 4)^2 + y^2) / ((x - 3)^2)) = 4 → 
  3 * x^2 - y^2 - 16 * x + 20 = 0 := by
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l242_24246


namespace NUMINAMATH_CALUDE_go_out_to_sea_is_better_l242_24267

/-- Represents the decision to go out to sea or not -/
inductive Decision
| GoOut
| StayIn

/-- Represents the weather condition -/
inductive Weather
| Good
| Bad

/-- The profit or loss for each scenario -/
def profit (d : Decision) (w : Weather) : ℤ :=
  match d, w with
  | Decision.GoOut, Weather.Good => 6000
  | Decision.GoOut, Weather.Bad => -8000
  | Decision.StayIn, _ => -1000

/-- The probability of each weather condition -/
def weather_prob (w : Weather) : ℚ :=
  match w with
  | Weather.Good => 1/2
  | Weather.Bad => 4/10

/-- The expected value of a decision -/
def expected_value (d : Decision) : ℚ :=
  (weather_prob Weather.Good * profit d Weather.Good) +
  (weather_prob Weather.Bad * profit d Weather.Bad)

/-- Theorem stating that going out to sea has a higher expected value -/
theorem go_out_to_sea_is_better :
  expected_value Decision.GoOut > expected_value Decision.StayIn :=
by sorry

end NUMINAMATH_CALUDE_go_out_to_sea_is_better_l242_24267


namespace NUMINAMATH_CALUDE_scientific_notation_of_105_9_billion_l242_24283

theorem scientific_notation_of_105_9_billion : 
  (105.9 : ℝ) * 1000000000 = 1.059 * (10 : ℝ)^10 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_105_9_billion_l242_24283


namespace NUMINAMATH_CALUDE_horner_method_correct_l242_24218

def horner_polynomial (x : ℚ) : ℚ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v4 (x : ℚ) : ℚ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  let v3 := v2 * x + 79
  v3 * x - 8

theorem horner_method_correct :
  horner_v4 (-4) = 220 :=
sorry

end NUMINAMATH_CALUDE_horner_method_correct_l242_24218


namespace NUMINAMATH_CALUDE_vertical_pairwise_sets_l242_24214

/-- Definition of a vertical pairwise set -/
def is_vertical_pairwise_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ p₁ ∈ M, ∃ p₂ ∈ M, p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

/-- Set M₁: y = 1/x² -/
def M₁ : Set (ℝ × ℝ) :=
  {p | p.2 = 1 / (p.1 ^ 2) ∧ p.1 ≠ 0}

/-- Set M₂: y = sin x + 1 -/
def M₂ : Set (ℝ × ℝ) :=
  {p | p.2 = Real.sin p.1 + 1}

/-- Set M₄: y = 2ˣ - 2 -/
def M₄ : Set (ℝ × ℝ) :=
  {p | p.2 = 2 ^ p.1 - 2}

/-- Theorem: M₁, M₂, and M₄ are vertical pairwise sets -/
theorem vertical_pairwise_sets :
  is_vertical_pairwise_set M₁ ∧
  is_vertical_pairwise_set M₂ ∧
  is_vertical_pairwise_set M₄ := by
  sorry

end NUMINAMATH_CALUDE_vertical_pairwise_sets_l242_24214


namespace NUMINAMATH_CALUDE_population_exceeds_target_in_2125_l242_24222

-- Define the initial year and population
def initialYear : ℕ := 1950
def initialPopulation : ℕ := 750

-- Define the doubling period
def doublingPeriod : ℕ := 35

-- Define the target population
def targetPopulation : ℕ := 15000

-- Function to calculate population after n doubling periods
def populationAfterPeriods (n : ℕ) : ℕ :=
  initialPopulation * 2^n

-- Function to calculate the year after n doubling periods
def yearAfterPeriods (n : ℕ) : ℕ :=
  initialYear + n * doublingPeriod

-- Theorem to prove
theorem population_exceeds_target_in_2125 :
  ∃ n : ℕ, yearAfterPeriods n = 2125 ∧ populationAfterPeriods n > targetPopulation ∧
  ∀ m : ℕ, m < n → populationAfterPeriods m ≤ targetPopulation :=
sorry

end NUMINAMATH_CALUDE_population_exceeds_target_in_2125_l242_24222


namespace NUMINAMATH_CALUDE_total_gratuity_is_23_02_l242_24226

-- Define the structure for menu items
structure MenuItem where
  name : String
  basePrice : Float
  taxRate : Float

-- Define the menu items
def nyStriploin : MenuItem := ⟨"NY Striploin", 80, 0.10⟩
def wineGlass : MenuItem := ⟨"Glass of wine", 10, 0.15⟩
def dessert : MenuItem := ⟨"Dessert", 12, 0.05⟩
def waterBottle : MenuItem := ⟨"Bottle of water", 3, 0⟩

-- Define the gratuity rate
def gratuityRate : Float := 0.20

-- Function to calculate the total price with tax for an item
def totalPriceWithTax (item : MenuItem) : Float :=
  item.basePrice * (1 + item.taxRate)

-- Function to calculate the gratuity for an item
def calculateGratuity (item : MenuItem) : Float :=
  totalPriceWithTax item * gratuityRate

-- Theorem stating that the total gratuity is $23.02
theorem total_gratuity_is_23_02 :
  calculateGratuity nyStriploin +
  calculateGratuity wineGlass +
  calculateGratuity dessert +
  calculateGratuity waterBottle = 23.02 := by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_total_gratuity_is_23_02_l242_24226


namespace NUMINAMATH_CALUDE_sphere_volume_surface_area_ratio_l242_24213

theorem sphere_volume_surface_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 / 3 * Real.pi * r₁^3) / (4 / 3 * Real.pi * r₂^3) = 8 / 27 →
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_surface_area_ratio_l242_24213


namespace NUMINAMATH_CALUDE_fair_coin_five_tosses_l242_24231

/-- A fair coin is a coin that has an equal probability of landing on either side when tossed. -/
def FairCoin : Type := Unit

/-- The probability of a fair coin landing on one specific side in a single toss. -/
def singleTossProbability (coin : FairCoin) : ℚ := 1 / 2

/-- The number of tosses. -/
def numTosses : ℕ := 5

/-- The probability of a fair coin landing on the same side for a given number of tosses. -/
def sameSideProbability (coin : FairCoin) (n : ℕ) : ℚ :=
  (singleTossProbability coin) ^ n

theorem fair_coin_five_tosses (coin : FairCoin) :
  sameSideProbability coin numTosses = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_five_tosses_l242_24231


namespace NUMINAMATH_CALUDE_intersection_points_line_l242_24228

theorem intersection_points_line (s : ℝ) :
  let x : ℝ := (41 * s + 13) / 11
  let y : ℝ := -(2 * s + 6) / 11
  (2 * x - 3 * y = 8 * s + 4) ∧ 
  (x + 4 * y = 3 * s - 1) →
  y = (-22 * x + 272) / 451 := by
sorry

end NUMINAMATH_CALUDE_intersection_points_line_l242_24228
