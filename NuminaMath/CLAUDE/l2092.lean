import Mathlib

namespace integer_roots_quadratic_l2092_209217

theorem integer_roots_quadratic (m n : ℤ) : 
  (∃ x y : ℤ, (2*m - 3)*(n - 1)*x^2 + (2*m - 3)*(n - 1)*(m - n - 4)*x - 2*(2*m - 3)*(n - 1)*(m - n - 2) - 1 = 0 ∧
               (2*m - 3)*(n - 1)*y^2 + (2*m - 3)*(n - 1)*(m - n - 4)*y - 2*(2*m - 3)*(n - 1)*(m - n - 2) - 1 = 0 ∧
               x ≠ y) ↔
  ((m = 2 ∧ n = 2) ∨ (m = 2 ∧ n = 0)) :=
by sorry

end integer_roots_quadratic_l2092_209217


namespace max_sum_of_squared_distances_l2092_209232

variables {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem max_sum_of_squared_distances (a b c d : E) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) :
  ‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2 ≤ 16 := by
  sorry

end max_sum_of_squared_distances_l2092_209232


namespace unique_cube_fourth_power_l2092_209265

theorem unique_cube_fourth_power : 
  ∃! (K : ℤ), ∃ (Z : ℤ),
    600 < Z ∧ Z < 2000 ∧ 
    Z = K^4 ∧ 
    ∃ (n : ℤ), Z = n^3 ∧
    K = 8 :=
sorry

end unique_cube_fourth_power_l2092_209265


namespace c_symmetric_l2092_209236

def c : ℕ → ℕ → ℤ
  | m, 0 => 1
  | 0, n => 1
  | m+1, n+1 => c m (n+1) - (n+1) * c m n

theorem c_symmetric (m n : ℕ) (hm : m > 0) (hn : n > 0) : c m n = c n m := by
  sorry

end c_symmetric_l2092_209236


namespace prob_at_least_one_target_l2092_209279

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards that are either hearts or kings -/
def target_cards : ℕ := 16

/-- The probability of drawing a card that is not a heart or king -/
def prob_not_target : ℚ := (deck_size - target_cards) / deck_size

/-- The number of draws -/
def num_draws : ℕ := 3

/-- The probability of drawing at least one heart or king in three draws with replacement -/
theorem prob_at_least_one_target :
  1 - prob_not_target ^ num_draws = 1468 / 2197 := by sorry

end prob_at_least_one_target_l2092_209279


namespace parallel_line_through_point_l2092_209276

-- Define the slope of the given line
def m : ℚ := 1/2

-- Define the given line
def given_line (x : ℚ) : ℚ := m * x - 1

-- Define the point that the new line passes through
def point : ℚ × ℚ := (1, 0)

-- Define the equation of the new line
def new_line (x : ℚ) : ℚ := m * x - 1/2

theorem parallel_line_through_point :
  (∀ x, new_line x - new_line point.1 = m * (x - point.1)) ∧
  new_line point.1 = point.2 ∧
  ∀ x, new_line x - given_line x = new_line 0 - given_line 0 :=
by sorry

end parallel_line_through_point_l2092_209276


namespace min_reciprocal_sum_l2092_209267

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hsum : x + y = 8) (hprod : x * y = 12) : 
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 8 → a * b = 12 → 1/x + 1/y ≤ 1/a + 1/b) ∧ 
  1/x + 1/y = 2/3 :=
sorry

end min_reciprocal_sum_l2092_209267


namespace x_value_when_y_is_4_l2092_209205

-- Define the inverse square relationship between x and y
def inverse_square_relation (x y : ℝ) : Prop :=
  ∃ k : ℝ, x = k / (y ^ 2)

-- State the theorem
theorem x_value_when_y_is_4 :
  ∀ x₀ y₀ x₁ y₁ : ℝ,
  inverse_square_relation x₀ y₀ →
  inverse_square_relation x₁ y₁ →
  x₀ = 1 →
  y₀ = 3 →
  y₁ = 4 →
  x₁ = 0.5625 := by
sorry

end x_value_when_y_is_4_l2092_209205


namespace area_between_line_and_curve_l2092_209266

/-- The area enclosed between the line y = 2x and the curve y = x^2 from x = 0 to x = 2 is 4/3 -/
theorem area_between_line_and_curve : 
  ∫ x in (0 : ℝ)..2, (2 * x - x^2) = 4/3 := by sorry

end area_between_line_and_curve_l2092_209266


namespace supplementary_angles_difference_l2092_209262

theorem supplementary_angles_difference (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a / b = 5 / 3 →  -- ratio of angles is 5:3
  abs (a - b) = 45 :=  -- positive difference is 45°
by sorry

end supplementary_angles_difference_l2092_209262


namespace polynomial_factorization_l2092_209201

theorem polynomial_factorization (x y : ℝ) : 
  x^2 - y^2 - 2*x - 4*y - 3 = (x+y+1)*(x-y-3) := by
  sorry

end polynomial_factorization_l2092_209201


namespace max_digits_product_3digit_2digit_l2092_209295

theorem max_digits_product_3digit_2digit :
  ∀ (a b : ℕ), 100 ≤ a ∧ a ≤ 999 ∧ 10 ≤ b ∧ b ≤ 99 →
  a * b < 100000 :=
sorry

end max_digits_product_3digit_2digit_l2092_209295


namespace centroid_trajectory_l2092_209286

/-- Given a triangle ABC with vertices A(-3, 0), B(3, 0), and C(m, n) on the parabola y² = 6x,
    the centroid (x, y) of the triangle satisfies the equation y² = 2x for x ≠ 0. -/
theorem centroid_trajectory (m n x y : ℝ) : 
  n^2 = 6*m →                   -- C is on the parabola y² = 6x
  3*x = m →                     -- x-coordinate of centroid
  3*y = n →                     -- y-coordinate of centroid
  x ≠ 0 →                       -- x is non-zero
  y^2 = 2*x                     -- equation of centroid's trajectory
  := by sorry

end centroid_trajectory_l2092_209286


namespace quadratic_inequality_range_l2092_209220

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end quadratic_inequality_range_l2092_209220


namespace fifteenth_odd_multiple_of_5_l2092_209299

/-- The nth positive integer that is both odd and a multiple of 5 -/
def oddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

/-- Prove that the 15th positive integer that is both odd and a multiple of 5 is 145 -/
theorem fifteenth_odd_multiple_of_5 : oddMultipleOf5 15 = 145 := by
  sorry

end fifteenth_odd_multiple_of_5_l2092_209299


namespace average_weight_problem_l2092_209244

theorem average_weight_problem (a b c : ℝ) : 
  (a + b) / 2 = 41 →
  (b + c) / 2 = 43 →
  b = 33 →
  (a + b + c) / 3 = 45 := by
sorry

end average_weight_problem_l2092_209244


namespace complex_expression_simplification_l2092_209289

theorem complex_expression_simplification :
  (7 - 3 * Complex.I) - 3 * (2 + 4 * Complex.I) + (1 - Complex.I) * (3 + Complex.I) = 5 - 17 * Complex.I :=
by sorry

end complex_expression_simplification_l2092_209289


namespace quadrilateral_perimeter_l2092_209250

/-- Quadrilateral EFGH with given properties -/
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)
  (EF_perp_FG : (F.1 - E.1) * (G.2 - F.2) + (F.2 - E.2) * (G.1 - F.1) = 0)
  (HG_perp_FG : (G.1 - H.1) * (G.2 - F.2) + (G.2 - H.2) * (G.1 - F.1) = 0)
  (EF_length : Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2) = 12)
  (HG_length : Real.sqrt ((G.1 - H.1)^2 + (G.2 - H.2)^2) = 3)
  (FG_length : Real.sqrt ((G.1 - F.1)^2 + (G.2 - F.2)^2) = 16)

/-- The perimeter of quadrilateral EFGH is 31 + √337 -/
theorem quadrilateral_perimeter (q : Quadrilateral) :
  Real.sqrt ((q.E.1 - q.F.1)^2 + (q.E.2 - q.F.2)^2) +
  Real.sqrt ((q.F.1 - q.G.1)^2 + (q.F.2 - q.G.2)^2) +
  Real.sqrt ((q.G.1 - q.H.1)^2 + (q.G.2 - q.H.2)^2) +
  Real.sqrt ((q.H.1 - q.E.1)^2 + (q.H.2 - q.E.2)^2) =
  31 + Real.sqrt 337 := by
  sorry

end quadrilateral_perimeter_l2092_209250


namespace factorization_equality_l2092_209280

theorem factorization_equality (a x : ℝ) : -a*x^2 + 2*a*x - a = -a*(x-1)^2 := by
  sorry

end factorization_equality_l2092_209280


namespace sum_of_digits_of_power_l2092_209248

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_of_power : 
  tens_digit ((3 + 4)^25) + ones_digit ((3 + 4)^25) = 11 := by
  sorry

end sum_of_digits_of_power_l2092_209248


namespace ab_value_when_sqrt_and_abs_sum_zero_l2092_209293

theorem ab_value_when_sqrt_and_abs_sum_zero (a b : ℝ) :
  Real.sqrt (a - 3) + |1 - b| = 0 → a * b = 3 := by
  sorry

end ab_value_when_sqrt_and_abs_sum_zero_l2092_209293


namespace jack_walking_distance_l2092_209214

/-- Calculates the distance walked given the time in hours and minutes and the walking rate in miles per hour -/
def distance_walked (hours : ℕ) (minutes : ℕ) (rate : ℚ) : ℚ :=
  rate * (hours + minutes / 60)

/-- Proves that walking for 1 hour and 15 minutes at a rate of 7.2 miles per hour results in a distance of 9 miles -/
theorem jack_walking_distance :
  distance_walked 1 15 (7.2 : ℚ) = 9 := by
  sorry

end jack_walking_distance_l2092_209214


namespace prob_third_batch_given_two_standard_l2092_209243

/-- Represents the number of parts in each batch -/
def batch_size : ℕ := 20

/-- Represents the number of standard parts in the first batch -/
def standard_parts_batch1 : ℕ := 20

/-- Represents the number of standard parts in the second batch -/
def standard_parts_batch2 : ℕ := 15

/-- Represents the number of standard parts in the third batch -/
def standard_parts_batch3 : ℕ := 10

/-- Represents the probability of selecting a batch -/
def prob_select_batch : ℚ := 1 / 3

/-- Theorem stating the probability of selecting two standard parts from the third batch,
    given that two standard parts were selected consecutively from a randomly chosen batch -/
theorem prob_third_batch_given_two_standard : 
  (prob_select_batch * (standard_parts_batch3 / batch_size)^2) /
  (prob_select_batch * (standard_parts_batch1 / batch_size)^2 +
   prob_select_batch * (standard_parts_batch2 / batch_size)^2 +
   prob_select_batch * (standard_parts_batch3 / batch_size)^2) = 4 / 29 := by
  sorry

end prob_third_batch_given_two_standard_l2092_209243


namespace total_price_theorem_l2092_209294

def refrigerator_price : ℝ := 4275
def washing_machine_price : ℝ := refrigerator_price - 1490
def sales_tax_rate : ℝ := 0.07

def total_price_with_tax : ℝ :=
  (refrigerator_price + washing_machine_price) * (1 + sales_tax_rate)

theorem total_price_theorem :
  total_price_with_tax = 7554.20 := by sorry

end total_price_theorem_l2092_209294


namespace expression_simplification_l2092_209211

theorem expression_simplification (x : ℝ) 
  (hx : x ≠ 0 ∧ x ≠ 3 ∧ x ≠ 2) : 
  (x - 5) / (x - 3) - ((x^2 + 2*x + 1) / (x^2 + x)) / ((x + 1) / (x - 2)) = 
  -6 / (x^2 - 3*x) := by
sorry

end expression_simplification_l2092_209211


namespace expression_value_l2092_209287

theorem expression_value (x y : ℝ) (h : (x - y) / y = 2) :
  ((1 / (x - y) + 1 / (x + y)) / (x / (x - y)^2)) = 1 := by
  sorry

end expression_value_l2092_209287


namespace inverse_proportion_l2092_209298

theorem inverse_proportion (x y : ℝ) (h : x ≠ 0) : 
  (3 * x * y = 1) ↔ ∃ k : ℝ, k ≠ 0 ∧ y = k / x := by
  sorry

end inverse_proportion_l2092_209298


namespace least_n_with_1987_zeros_l2092_209253

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The least natural number n such that n! ends in exactly 1987 zeros -/
theorem least_n_with_1987_zeros : ∃ (n : ℕ), trailingZeros n = 1987 ∧ ∀ m < n, trailingZeros m < 1987 :=
  sorry

end least_n_with_1987_zeros_l2092_209253


namespace gain_percent_calculation_l2092_209238

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 100)
  (h2 : selling_price = 120) : 
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end gain_percent_calculation_l2092_209238


namespace mod_sum_powers_l2092_209277

theorem mod_sum_powers (n : ℕ) : (44^1234 + 99^567) % 7 = 3 := by
  sorry

end mod_sum_powers_l2092_209277


namespace negation_zero_collinear_with_any_l2092_209213

open Set

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def IsCollinear (v w : V) : Prop := ∃ (c : ℝ), v = c • w ∨ w = c • v

theorem negation_zero_collinear_with_any :
  (¬ ∀ (v : V), IsCollinear (0 : V) v) ↔ ∃ (v : V), ¬ IsCollinear (0 : V) v :=
sorry

end negation_zero_collinear_with_any_l2092_209213


namespace rope_for_third_post_l2092_209225

theorem rope_for_third_post 
  (total_rope : ℕ) 
  (first_post second_post fourth_post : ℕ) 
  (h1 : total_rope = 70)
  (h2 : first_post = 24)
  (h3 : second_post = 20)
  (h4 : fourth_post = 12) :
  total_rope - (first_post + second_post + fourth_post) = 14 := by
  sorry

end rope_for_third_post_l2092_209225


namespace complex_sum_imaginary_l2092_209271

theorem complex_sum_imaginary (a : ℝ) : 
  let z₁ : ℂ := a^2 - 2 - 3*a*Complex.I
  let z₂ : ℂ := a + (a^2 + 2)*Complex.I
  (z₁ + z₂).re = 0 ∧ (z₁ + z₂).im ≠ 0 → a = -2 := by
  sorry

end complex_sum_imaginary_l2092_209271


namespace two_squares_five_points_arrangement_l2092_209257

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a square in 2D space
structure Square where
  center : Point
  side_length : ℝ

-- Define a function to check if a point is inside a square
def is_inside (p : Point) (s : Square) : Prop :=
  abs (p.x - s.center.x) ≤ s.side_length / 2 ∧
  abs (p.y - s.center.y) ≤ s.side_length / 2

-- Define the theorem
theorem two_squares_five_points_arrangement :
  ∃ (s1 s2 : Square) (p1 p2 p3 p4 p5 : Point),
    (is_inside p1 s1 ∧ is_inside p2 s1 ∧ is_inside p3 s1) ∧
    (is_inside p1 s2 ∧ is_inside p2 s2 ∧ is_inside p3 s2 ∧ is_inside p4 s2) :=
  sorry

end two_squares_five_points_arrangement_l2092_209257


namespace probability_open_path_correct_l2092_209259

/-- The probability of being able to go from the first floor to the last floor through only open doors in a building with n floors and randomly locked doors. -/
def probability_open_path (n : ℕ) : ℚ :=
  if n ≤ 1 then 1
  else (2 ^ (n - 1) : ℚ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℚ)

/-- Theorem stating the probability of an open path in the building. -/
theorem probability_open_path_correct (n : ℕ) (h : n > 1) :
  probability_open_path n =
    (2 ^ (n - 1) : ℚ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℚ) := by
  sorry

#eval probability_open_path 5

end probability_open_path_correct_l2092_209259


namespace batsman_average_increase_17_innings_l2092_209249

def batsman_average_increase (total_innings : ℕ) (last_innings_score : ℕ) (final_average : ℚ) : ℚ :=
  let previous_total := (total_innings - 1) * (total_innings * final_average - last_innings_score) / total_innings
  let previous_average := previous_total / (total_innings - 1)
  final_average - previous_average

theorem batsman_average_increase_17_innings 
  (h1 : total_innings = 17)
  (h2 : last_innings_score = 85)
  (h3 : final_average = 37) :
  batsman_average_increase total_innings last_innings_score final_average = 3 := by
  sorry

end batsman_average_increase_17_innings_l2092_209249


namespace possible_values_of_a_l2092_209240

theorem possible_values_of_a :
  ∀ (a b c : ℤ), (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) → (a = 3 ∨ a = 7) :=
by sorry

end possible_values_of_a_l2092_209240


namespace no_numbers_equal_seven_times_sum_of_digits_l2092_209212

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem no_numbers_equal_seven_times_sum_of_digits : 
  ∀ n : ℕ, n ≤ 1000 → n ≠ 7 * sum_of_digits n :=
sorry

end no_numbers_equal_seven_times_sum_of_digits_l2092_209212


namespace chalkboard_area_l2092_209230

theorem chalkboard_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 3.5 →
  length = 2.3 * width →
  area = length * width →
  area = 28.175 := by
sorry

end chalkboard_area_l2092_209230


namespace stamps_in_first_book_l2092_209241

theorem stamps_in_first_book (x : ℕ) : 
  (4 * x + 6 * 15 = 130) → x = 10 := by
  sorry

end stamps_in_first_book_l2092_209241


namespace sum_lent_is_400_l2092_209215

/-- Calculates the simple interest for a given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem sum_lent_is_400 :
  let rate : ℚ := 4
  let time : ℚ := 8
  let principal : ℚ := 400
  simpleInterest principal rate time = principal - 272 :=
by
  sorry

#check sum_lent_is_400

end sum_lent_is_400_l2092_209215


namespace number_of_pickers_l2092_209291

theorem number_of_pickers (drums_per_day : ℕ) (total_days : ℕ) (total_drums : ℕ) :
  drums_per_day = 221 →
  total_days = 77 →
  total_drums = 17017 →
  drums_per_day * total_days = total_drums →
  drums_per_day = 221 :=
by
  sorry

end number_of_pickers_l2092_209291


namespace hash_toy_difference_l2092_209246

theorem hash_toy_difference (bill_toys : ℕ) (total_toys : ℕ) : 
  bill_toys = 60 →
  total_toys = 99 →
  total_toys - bill_toys - bill_toys / 2 = 9 :=
by sorry

end hash_toy_difference_l2092_209246


namespace geometric_sequence_property_l2092_209208

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) 
  (h_prod : a 7 * a 9 = 4) 
  (h_a4 : a 4 = 1) : 
  a 12 = 4 := by
sorry

end geometric_sequence_property_l2092_209208


namespace diet_soda_bottles_l2092_209254

theorem diet_soda_bottles (total : ℕ) (regular : ℕ) (h1 : total = 30) (h2 : regular = 28) :
  total - regular = 2 := by
  sorry

end diet_soda_bottles_l2092_209254


namespace power_simplification_l2092_209237

theorem power_simplification (x : ℝ) : (5 * x^4)^3 = 125 * x^12 := by
  sorry

end power_simplification_l2092_209237


namespace complex_root_implies_positive_and_triangle_l2092_209222

theorem complex_root_implies_positive_and_triangle (a b c α β : ℝ) : 
  (α > 0) →
  (β ≠ 0) →
  (Complex.I : ℂ)^2 = -1 →
  (α + β * Complex.I : ℂ)^2 - (a + b + c) * (α + β * Complex.I : ℂ) + (a * b + b * c + c * a : ℂ) = 0 →
  (a > 0 ∧ b > 0 ∧ c > 0) ∧ (Real.sqrt a < Real.sqrt b + Real.sqrt c) :=
by sorry

end complex_root_implies_positive_and_triangle_l2092_209222


namespace trapezoid_area_in_hexagon_triangle_l2092_209263

/-- Given a regular hexagon with an inscribed equilateral triangle, this theorem calculates the area of one of the six congruent trapezoids formed between the hexagon and the triangle. -/
theorem trapezoid_area_in_hexagon_triangle (hexagon_area : ℝ) (triangle_area : ℝ) :
  hexagon_area = 24 →
  triangle_area = 4 →
  (hexagon_area - triangle_area) / 6 = 10 / 3 := by
  sorry

end trapezoid_area_in_hexagon_triangle_l2092_209263


namespace tangent_lines_to_circle_radius_of_circle_l2092_209297

-- Define the circle M
def circle_M (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*x + a = 0

-- Define the point P
def point_P : ℝ × ℝ := (4, 5)

-- Define the tangent line equations
def tangent_line_1 (x : ℝ) : Prop := x = 4
def tangent_line_2 (x y : ℝ) : Prop := 8*x - 15*y + 43 = 0

-- Theorem for part (1)
theorem tangent_lines_to_circle (x y : ℝ) :
  circle_M (-8) x y →
  (∃ (t : ℝ), (x = t * (point_P.1 - x) + x ∧ y = t * (point_P.2 - y) + y)) →
  (tangent_line_1 x ∨ tangent_line_2 x y) :=
sorry

-- Theorem for part (2)
theorem radius_of_circle (a : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  circle_M a x₁ y₁ →
  circle_M a x₂ y₂ →
  x₁ * x₂ + y₁ * y₂ = -6 →
  ∃ (r : ℝ), r^2 = 7 ∧ 
    ∀ (x y : ℝ), circle_M a x y → (x - 1)^2 + y^2 = r^2 :=
sorry

end tangent_lines_to_circle_radius_of_circle_l2092_209297


namespace fraction_equality_l2092_209202

theorem fraction_equality (a b c : ℝ) (hb : b ≠ 0) (hc : c^2 + 1 ≠ 0) :
  (a * (c^2 + 1)) / (b * (c^2 + 1)) = a / b :=
by sorry

end fraction_equality_l2092_209202


namespace tan_570_degrees_l2092_209200

theorem tan_570_degrees : Real.tan (570 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end tan_570_degrees_l2092_209200


namespace degree_of_specific_monomial_l2092_209268

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree_of_monomial (m : Polynomial ℚ) : ℕ :=
  sorry

/-- The monomial 2/3 * a^3 * b -/
def monomial : Polynomial ℚ :=
  sorry

theorem degree_of_specific_monomial :
  degree_of_monomial monomial = 4 := by
  sorry

end degree_of_specific_monomial_l2092_209268


namespace hexagon_largest_angle_l2092_209255

/-- Given a hexagon with internal angles in the ratio 2:3:3:4:5:6, 
    the measure of the largest angle is 4320°/23. -/
theorem hexagon_largest_angle (a b c d e f : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  b / a = 3 / 2 →
  c / a = 3 / 2 →
  d / a = 2 →
  e / a = 5 / 2 →
  f / a = 3 →
  a + b + c + d + e + f = 720 →
  f = 4320 / 23 := by
  sorry

end hexagon_largest_angle_l2092_209255


namespace inequality_proof_l2092_209252

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  (a - b + c) * (1/a - 1/b + 1/c) ≥ 1 := by
  sorry

end inequality_proof_l2092_209252


namespace university_subjects_overlap_l2092_209227

/-- The problem of students studying Physics and Chemistry at a university --/
theorem university_subjects_overlap (total : ℕ) (physics_min physics_max chem_min chem_max : ℕ) :
  total = 2500 →
  physics_min = 1750 →
  physics_max = 1875 →
  chem_min = 1000 →
  chem_max = 1125 →
  let m := physics_min + chem_min - total
  let M := physics_max + chem_max - total
  M - m = 250 := by
  sorry

end university_subjects_overlap_l2092_209227


namespace application_methods_eq_sixteen_l2092_209203

/-- The number of universities -/
def total_universities : ℕ := 6

/-- The number of universities to be chosen -/
def universities_to_choose : ℕ := 3

/-- The number of universities with overlapping schedules -/
def overlapping_universities : ℕ := 2

/-- The function to calculate the number of different application methods -/
def application_methods : ℕ := sorry

/-- Theorem stating that the number of different application methods is 16 -/
theorem application_methods_eq_sixteen :
  application_methods = 16 := by sorry

end application_methods_eq_sixteen_l2092_209203


namespace roots_of_polynomial_l2092_209245

/-- The polynomial function we're working with -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - x + 3

/-- Theorem stating that 1, -1, and 3 are the only roots of the polynomial -/
theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) := by
  sorry

#check roots_of_polynomial

end roots_of_polynomial_l2092_209245


namespace integer_root_values_l2092_209264

theorem integer_root_values (a : ℤ) : 
  (∃ x : ℤ, x^3 + a*x^2 + 3*x + 7 = 0) ↔ (a = -11 ∨ a = -3) :=
by sorry

end integer_root_values_l2092_209264


namespace power_function_through_point_l2092_209256

theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x > 0, f x = x ^ α) →
  f 2 = Real.sqrt 2 →
  α = 1 / 2 := by
sorry

end power_function_through_point_l2092_209256


namespace supplement_of_complement_of_57_degree_angle_l2092_209228

-- Define the original angle
def original_angle : ℝ := 57

-- Define complement
def complement (angle : ℝ) : ℝ := 90 - angle

-- Define supplement
def supplement (angle : ℝ) : ℝ := 180 - angle

-- Theorem statement
theorem supplement_of_complement_of_57_degree_angle :
  supplement (complement original_angle) = 147 := by
  sorry

end supplement_of_complement_of_57_degree_angle_l2092_209228


namespace max_pages_copied_l2092_209234

/-- The cost in cents to copy 4 pages -/
def cost_per_4_pages : ℚ := 7

/-- The budget in dollars -/
def budget : ℚ := 15

/-- The number of pages that can be copied with the given budget -/
def pages_copied : ℕ := 857

/-- Theorem stating the maximum number of complete pages that can be copied -/
theorem max_pages_copied : 
  ⌊(budget * 100 / cost_per_4_pages) * 4⌋ = pages_copied :=
sorry

end max_pages_copied_l2092_209234


namespace monochromatic_triangle_exists_l2092_209269

/-- A color type representing red or blue --/
inductive Color
  | Red
  | Blue

/-- A type representing a complete graph with 6 vertices --/
structure CompleteGraph6 where
  /-- A function assigning a color to each pair of distinct vertices --/
  edgeColor : Fin 6 → Fin 6 → Color
  /-- Ensure the graph is undirected --/
  symm : ∀ (i j : Fin 6), i ≠ j → edgeColor i j = edgeColor j i

/-- Definition of a monochromatic triangle in the graph --/
def hasMonochromaticTriangle (g : CompleteGraph6) : Prop :=
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    g.edgeColor i j = g.edgeColor j k ∧ g.edgeColor j k = g.edgeColor k i

/-- Theorem stating that every complete graph with 6 vertices and edges colored red or blue
    contains a monochromatic triangle --/
theorem monochromatic_triangle_exists (g : CompleteGraph6) : hasMonochromaticTriangle g :=
  sorry


end monochromatic_triangle_exists_l2092_209269


namespace special_function_properties_l2092_209233

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x > 0 → f x < 0)

/-- The main theorem stating the properties of the special function -/
theorem special_function_properties (f : ℝ → ℝ) (hf : special_function f) :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f x > f y) :=
by sorry

end special_function_properties_l2092_209233


namespace unique_six_digit_number_l2092_209278

/-- Represents a six-digit number -/
def SixDigitNumber := { n : ℕ // 100000 ≤ n ∧ n ≤ 999999 }

/-- Represents a five-digit number -/
def FiveDigitNumber := { n : ℕ // 10000 ≤ n ∧ n ≤ 99999 }

/-- Function that removes one digit from a six-digit number to form a five-digit number -/
def removeOneDigit (n : SixDigitNumber) : FiveDigitNumber :=
  sorry

/-- The problem statement -/
theorem unique_six_digit_number :
  ∃! (n : SixDigitNumber), 
    ∀ (m : FiveDigitNumber), 
      (m = removeOneDigit n) → (n.val - m.val = 654321) := by
  sorry

end unique_six_digit_number_l2092_209278


namespace stock_percentage_is_25_percent_l2092_209290

/-- Calculates the percentage of a stock given the investment amount and income. -/
def stock_percentage (investment income : ℚ) : ℚ :=
  (income * 100) / investment

/-- Theorem stating that the stock percentage is 25% given the specified investment and income. -/
theorem stock_percentage_is_25_percent (investment : ℚ) (income : ℚ) 
  (h1 : investment = 15200)
  (h2 : income = 3800) :
  stock_percentage investment income = 25 := by
  sorry

end stock_percentage_is_25_percent_l2092_209290


namespace juice_packs_fit_l2092_209251

/-- The number of juice packs that can fit in a box without gaps -/
def juice_packs_in_box (box_width box_length box_height juice_width juice_length juice_height : ℕ) : ℕ :=
  (box_width * box_length * box_height) / (juice_width * juice_length * juice_height)

/-- Theorem stating that 72 juice packs fit in the given box -/
theorem juice_packs_fit :
  juice_packs_in_box 24 15 28 4 5 7 = 72 := by
  sorry

#eval juice_packs_in_box 24 15 28 4 5 7

end juice_packs_fit_l2092_209251


namespace sqrt_equation_solution_l2092_209231

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- Proof goes here
  sorry

end sqrt_equation_solution_l2092_209231


namespace all_pies_have_ingredients_l2092_209274

theorem all_pies_have_ingredients (total_pies : ℕ) 
  (blueberry_fraction : ℚ) (strawberry_fraction : ℚ) 
  (raspberry_fraction : ℚ) (almond_fraction : ℚ) : 
  total_pies = 48 →
  blueberry_fraction = 1/3 →
  strawberry_fraction = 3/8 →
  raspberry_fraction = 1/2 →
  almond_fraction = 1/4 →
  ∃ (blueberry strawberry raspberry almond : Finset (Fin total_pies)),
    (blueberry.card : ℚ) ≥ blueberry_fraction * total_pies ∧
    (strawberry.card : ℚ) ≥ strawberry_fraction * total_pies ∧
    (raspberry.card : ℚ) ≥ raspberry_fraction * total_pies ∧
    (almond.card : ℚ) ≥ almond_fraction * total_pies ∧
    (blueberry ∪ strawberry ∪ raspberry ∪ almond).card = total_pies :=
by sorry

end all_pies_have_ingredients_l2092_209274


namespace lines_perpendicular_to_plane_are_parallel_l2092_209285

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end lines_perpendicular_to_plane_are_parallel_l2092_209285


namespace exists_m_satisfying_conditions_l2092_209223

-- Define the sequence of concatenated natural numbers
def concatNaturals : ℕ → ℕ
| 0 => 0
| (n + 1) => concatNaturals n * 10^(Nat.digits 10 (n + 1)).length + (n + 1)

-- Define A_k as the first k digits of the concatenated sequence
def A (k : ℕ) : ℕ := concatNaturals k % (10^k)

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

-- State the theorem
theorem exists_m_satisfying_conditions (n : ℕ+) :
  ∃ m : ℕ+, (A m.val ∣ n) ∧ (m ∣ n) ∧ (sumOfDigits (A m.val) ∣ n) := by
  sorry

end exists_m_satisfying_conditions_l2092_209223


namespace quadratic_form_sum_l2092_209296

theorem quadratic_form_sum (x : ℝ) : ∃ (d e : ℝ), x^2 - 18*x + 81 = (x + d)^2 + e ∧ d + e = -9 := by
  sorry

end quadratic_form_sum_l2092_209296


namespace sum_of_combinations_specific_combination_l2092_209204

-- Define the binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Statement for the first part
theorem sum_of_combinations : C 5 0 + C 6 5 + C 7 5 + C 8 5 + C 9 5 + C 10 5 = 462 := by sorry

-- Statement for the second part
theorem specific_combination (m : ℕ) :
  (1 / C 5 m : ℚ) - (1 / C 6 m : ℚ) = (7 : ℚ) / (10 * C 7 m) → C 8 m = 28 := by sorry

end sum_of_combinations_specific_combination_l2092_209204


namespace line_parallel_perpendicular_implies_planes_perpendicular_l2092_209292

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l m : Line) (α β : Plane)
  (h_diff_lines : l ≠ m)
  (h_diff_planes : α ≠ β)
  (h_parallel : parallel_line_plane l α)
  (h_perpendicular : perpendicular_line_plane l β) :
  perpendicular_plane_plane α β :=
sorry

end line_parallel_perpendicular_implies_planes_perpendicular_l2092_209292


namespace rationalize_sqrt_five_eighteenths_l2092_209239

theorem rationalize_sqrt_five_eighteenths : 
  Real.sqrt (5 / 18) = Real.sqrt 10 / 6 := by sorry

end rationalize_sqrt_five_eighteenths_l2092_209239


namespace perpendicular_bisector_c_l2092_209247

/-- The perpendicular bisector of a line segment connecting two points (x₁, y₁) and (x₂, y₂) 
    is defined by the equation x + 2y = c. -/
def is_perpendicular_bisector (x₁ y₁ x₂ y₂ c : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + 2 * midpoint_y = c

/-- The value of c for the perpendicular bisector of the line segment 
    connecting (2,4) and (8,16) is 25. -/
theorem perpendicular_bisector_c : 
  is_perpendicular_bisector 2 4 8 16 25 := by
  sorry

end perpendicular_bisector_c_l2092_209247


namespace common_chord_length_l2092_209282

theorem common_chord_length (r : ℝ) (d : ℝ) (h1 : r = 12) (h2 : d = 16) :
  let chord_length := 2 * Real.sqrt (r^2 - (d/2)^2)
  chord_length = 8 * Real.sqrt 5 := by sorry

end common_chord_length_l2092_209282


namespace vanessa_deleted_files_l2092_209288

/-- Calculates the number of deleted files given the initial number of music and video files and the number of files left after deletion. -/
def deleted_files (music_files : ℕ) (video_files : ℕ) (files_left : ℕ) : ℕ :=
  music_files + video_files - files_left

/-- Proves that the number of deleted files is 10 given the specific conditions in the problem. -/
theorem vanessa_deleted_files :
  deleted_files 13 30 33 = 10 := by
  sorry

#eval deleted_files 13 30 33

end vanessa_deleted_files_l2092_209288


namespace biking_distance_l2092_209283

/-- Calculates the distance traveled given a constant rate and time -/
def distance (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Proves that biking at 8 miles per hour for 2.5 hours results in a distance of 20 miles -/
theorem biking_distance :
  let rate : ℝ := 8
  let time : ℝ := 2.5
  distance rate time = 20 := by sorry

end biking_distance_l2092_209283


namespace rationalize_cube_root_seven_l2092_209270

def rationalize_denominator (a b : ℕ) : ℚ × ℕ × ℕ := sorry

theorem rationalize_cube_root_seven :
  let (frac, B, C) := rationalize_denominator 4 (3 * 7^(1/3))
  frac = 4 * (49^(1/3)) / 21 ∧ 
  B = 49 ∧ 
  C = 21 ∧
  4 + B + C = 74 := by sorry

end rationalize_cube_root_seven_l2092_209270


namespace polynomial_division_quotient_remainder_l2092_209207

theorem polynomial_division_quotient_remainder (z : ℚ) :
  3 * z^4 - 4 * z^3 + 5 * z^2 - 11 * z + 2 =
  (2 + 3 * z) * (z^3 - 2 * z^2 + 3 * z - 17/3) + 40/3 := by
  sorry

end polynomial_division_quotient_remainder_l2092_209207


namespace symmetry_about_x_equals_one_symmetry_about_x_equals_three_halves_odd_function_shift_l2092_209260

open Function

-- Define a function f from reals to reals
variable (f : ℝ → ℝ)

-- Statement ①
theorem symmetry_about_x_equals_one :
  (∀ x, f (x - 1) = f (1 - x)) ↔ 
  (∀ x, f (2 - x) = f x) :=
sorry

-- Statement ②
theorem symmetry_about_x_equals_three_halves 
  (h1 : ∀ x, f (-3/2 - x) = f x) 
  (h2 : ∀ x, f (x + 3/2) = -f x) :
  ∀ x, f (3 - x) = f x :=
sorry

-- Statement ③
theorem odd_function_shift
  (h : ∀ x, f (x + 2) = -f (-x + 4)) :
  ∀ x, f ((x + 3) + 3) = -f (-(x + 3) + 3) :=
sorry

end symmetry_about_x_equals_one_symmetry_about_x_equals_three_halves_odd_function_shift_l2092_209260


namespace imaginary_part_of_z_l2092_209216

theorem imaginary_part_of_z (z : ℂ) : z = 2 / (Complex.I - 1) → z.im = -1 := by
  sorry

end imaginary_part_of_z_l2092_209216


namespace ratio_equality_l2092_209281

theorem ratio_equality (a b c d : ℝ) 
  (h1 : a = 5 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 6 * d) : 
  (a + b * c) / (c + d * b) = 3 * (5 + 6 * d) / (1 + 3 * d) := by
  sorry

end ratio_equality_l2092_209281


namespace equal_sets_implies_a_equals_one_l2092_209275

theorem equal_sets_implies_a_equals_one (a : ℝ) : 
  ({2, -1} : Set ℝ) = {2, a^2 - 2*a} → a = 1 := by
  sorry

end equal_sets_implies_a_equals_one_l2092_209275


namespace baseball_hits_theorem_l2092_209235

theorem baseball_hits_theorem (total_hits home_runs triples doubles : ℕ) 
  (h1 : total_hits = 50)
  (h2 : home_runs = 2)
  (h3 : triples = 3)
  (h4 : doubles = 10) :
  let singles := total_hits - (home_runs + triples + doubles)
  let percentage := (singles : ℚ) / total_hits * 100
  singles = 35 ∧ percentage = 70 := by
sorry

end baseball_hits_theorem_l2092_209235


namespace symmetry_of_point_l2092_209224

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to X-axis -/
def symmetryXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

theorem symmetry_of_point :
  let P : Point3D := { x := -1, y := 8, z := 4 }
  symmetryXAxis P = { x := -1, y := -8, z := 4 } := by
  sorry

end symmetry_of_point_l2092_209224


namespace orange_shells_count_l2092_209209

theorem orange_shells_count (total shells purple pink yellow blue : ℕ) 
  (h1 : total = 65)
  (h2 : purple = 13)
  (h3 : pink = 8)
  (h4 : yellow = 18)
  (h5 : blue = 12) :
  total - (purple + pink + yellow + blue) = 14 := by
  sorry

end orange_shells_count_l2092_209209


namespace cylinder_height_ratio_l2092_209226

/-- 
Given two right circular cylinders V and B, where:
- The radius of V is twice the radius of B
- It costs $4 to fill half of B
- It costs $16 to fill V completely
This theorem proves that the ratio of the height of V to the height of B is 1:2
-/
theorem cylinder_height_ratio 
  (r_B : ℝ) -- radius of cylinder B
  (h_B : ℝ) -- height of cylinder B
  (h_V : ℝ) -- height of cylinder V
  (cost_half_B : ℝ) -- cost to fill half of cylinder B
  (cost_full_V : ℝ) -- cost to fill cylinder V completely
  (h_radius : r_B > 0) -- radius of B is positive
  (h_cost_half_B : cost_half_B = 4) -- cost to fill half of B is $4
  (h_cost_full_V : cost_full_V = 16) -- cost to fill V completely is $16
  : h_V / h_B = 1 / 2 := by
  sorry

end cylinder_height_ratio_l2092_209226


namespace alpha_plus_three_beta_range_l2092_209242

theorem alpha_plus_three_beta_range (α β : ℝ) 
  (h1 : -1 ≤ α + β ∧ α + β ≤ 1) 
  (h2 : 1 ≤ α + 2*β ∧ α + 2*β ≤ 3) : 
  1 ≤ α + 3*β ∧ α + 3*β ≤ 7 := by
sorry

end alpha_plus_three_beta_range_l2092_209242


namespace peggy_needs_825_stamps_l2092_209210

/-- The number of stamps Peggy needs to add to have as many as Bert -/
def stamps_to_add (peggy_stamps ernie_stamps bert_stamps : ℕ) : ℕ :=
  bert_stamps - peggy_stamps

/-- Proof that Peggy needs to add 825 stamps to have as many as Bert -/
theorem peggy_needs_825_stamps : 
  ∀ (peggy_stamps ernie_stamps bert_stamps : ℕ),
    peggy_stamps = 75 →
    ernie_stamps = 3 * peggy_stamps →
    bert_stamps = 4 * ernie_stamps →
    stamps_to_add peggy_stamps ernie_stamps bert_stamps = 825 := by
  sorry

end peggy_needs_825_stamps_l2092_209210


namespace largest_a_for_integer_solution_l2092_209258

theorem largest_a_for_integer_solution :
  ∃ (a : ℝ), ∀ (b : ℝ),
    (∃ (x y : ℤ), x - 4*y = 1 ∧ a*x + 3*y = 1) ∧
    (∀ (x y : ℤ), b > a → ¬(x - 4*y = 1 ∧ b*x + 3*y = 1)) →
    a = 1 := by
  sorry

end largest_a_for_integer_solution_l2092_209258


namespace variation_problem_l2092_209206

theorem variation_problem (R S T : ℚ) (c : ℚ) : 
  (∀ R S T, R = c * S / T) →  -- R varies directly as S and inversely as T
  (2 = c * 4 / (1/2)) →       -- When R = 2, T = 1/2, S = 4
  (8 = c * S / (1/3)) →       -- When R = 8 and T = 1/3
  S = 32/3 := by
sorry

end variation_problem_l2092_209206


namespace fruit_store_problem_l2092_209284

/-- The price of apples in yuan per kg -/
def apple_price : ℝ := 8

/-- The price of pears in yuan per kg -/
def pear_price : ℝ := 6

/-- The maximum number of kg of apples that can be purchased -/
def max_apple_kg : ℝ := 5

theorem fruit_store_problem :
  (∀ x y : ℝ, x + 3 * y = 26 ∧ 2 * x + y = 22 →
    x = apple_price ∧ y = pear_price) ∧
  (∀ m : ℝ, 8 * m + 6 * (15 - m) ≤ 100 → m ≤ max_apple_kg) :=
by sorry

end fruit_store_problem_l2092_209284


namespace arrangement_satisfies_condition_l2092_209261

def arrangement : List ℕ := [3, 1, 4, 1, 3, 0, 2, 4, 2, 0]

def count_between (list : List ℕ) (n : ℕ) : ℕ :=
  match list.indexOf? n, list.reverse.indexOf? n with
  | some i, some j => list.length - i - j - 2
  | _, _ => 0

def satisfies_condition (list : List ℕ) : Prop :=
  ∀ n ∈ list, count_between list n = n

theorem arrangement_satisfies_condition : 
  satisfies_condition arrangement :=
sorry

end arrangement_satisfies_condition_l2092_209261


namespace exists_arithmetic_right_triangle_with_81_l2092_209273

/-- A right triangle with integer side lengths forming an arithmetic sequence -/
structure ArithmeticRightTriangle where
  a : ℕ
  d : ℕ
  right_triangle : a^2 + (a + d)^2 = (a + 2*d)^2
  arithmetic_sequence : True

/-- The existence of an arithmetic right triangle with one side length equal to 81 -/
theorem exists_arithmetic_right_triangle_with_81 :
  ∃ (t : ArithmeticRightTriangle), t.a = 81 ∨ t.a + t.d = 81 ∨ t.a + 2*t.d = 81 := by
  sorry

end exists_arithmetic_right_triangle_with_81_l2092_209273


namespace paint_left_is_four_liters_l2092_209219

/-- The amount of paint Dexter used in gallons -/
def dexter_paint : ℚ := 3/8

/-- The amount of paint Jay used in gallons -/
def jay_paint : ℚ := 5/8

/-- The conversion factor from gallons to liters -/
def gallon_to_liter : ℚ := 4

/-- The total amount of paint in gallons -/
def total_paint : ℚ := 2

theorem paint_left_is_four_liters : 
  (total_paint * gallon_to_liter) - ((dexter_paint + jay_paint) * gallon_to_liter) = 4 := by
  sorry

end paint_left_is_four_liters_l2092_209219


namespace james_sales_problem_l2092_209221

theorem james_sales_problem :
  let houses_day1 : ℕ := 20
  let houses_day2 : ℕ := 40
  let sale_rate_day2 : ℚ := 4/5
  let total_items : ℕ := 104
  let items_per_house : ℕ := 2
  
  (houses_day1 * items_per_house + 
   (houses_day2 : ℚ) * sale_rate_day2 * (items_per_house : ℚ) = (total_items : ℚ)) ∧
  (houses_day2 = 2 * houses_day1) :=
by
  sorry

end james_sales_problem_l2092_209221


namespace actual_distance_is_82_l2092_209272

/-- Represents the distance between two towns on a map --/
def map_distance : ℝ := 9

/-- Represents the initial scale of the map --/
def initial_scale : ℝ := 10

/-- Represents the subsequent scale of the map --/
def subsequent_scale : ℝ := 8

/-- Represents the distance on the map where the initial scale applies --/
def initial_scale_distance : ℝ := 5

/-- Calculates the actual distance between two towns given the map distance and scales --/
def actual_distance : ℝ :=
  initial_scale * initial_scale_distance +
  subsequent_scale * (map_distance - initial_scale_distance)

/-- Theorem stating that the actual distance between the towns is 82 miles --/
theorem actual_distance_is_82 : actual_distance = 82 := by
  sorry

end actual_distance_is_82_l2092_209272


namespace ninth_term_of_geometric_sequence_l2092_209229

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem ninth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_sixth : a 6 = 16)
  (h_twelfth : a 12 = 4) :
  a 9 = 2 := by
  sorry


end ninth_term_of_geometric_sequence_l2092_209229


namespace smallest_x_value_l2092_209218

theorem smallest_x_value (x : ℝ) : 
  (2 * x^2 + 24 * x - 60 = x * (x + 13)) → x ≥ -15 :=
by sorry

end smallest_x_value_l2092_209218
