import Mathlib

namespace NUMINAMATH_CALUDE_binomial_coefficient_21_12_l3542_354236

theorem binomial_coefficient_21_12 :
  (Nat.choose 20 13 = 77520) →
  (Nat.choose 20 12 = 125970) →
  (Nat.choose 21 13 = 203490) →
  (Nat.choose 21 12 = 125970) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_21_12_l3542_354236


namespace NUMINAMATH_CALUDE_original_denominator_proof_l3542_354281

theorem original_denominator_proof (d : ℚ) : 
  (4 : ℚ) / d ≠ 0 →
  (4 + 3) / (d + 3) = 2 / 3 →
  d = 15 / 2 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l3542_354281


namespace NUMINAMATH_CALUDE_supplier_A_better_performance_l3542_354211

def supplier_A : List ℕ := [10, 9, 10, 10, 11, 11, 9, 11, 10, 10]
def supplier_B : List ℕ := [8, 10, 14, 7, 10, 11, 10, 8, 15, 12]

def mean (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ :=
  let m := mean l
  (l.map (fun x => ((x : ℚ) - m) ^ 2)).sum / l.length

theorem supplier_A_better_performance (A : List ℕ) (B : List ℕ)
  (hA : A = supplier_A) (hB : B = supplier_B) :
  mean A < mean B ∧ variance A < variance B := by
  sorry

end NUMINAMATH_CALUDE_supplier_A_better_performance_l3542_354211


namespace NUMINAMATH_CALUDE_min_value_inequality_l3542_354256

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 5) :
  (9 / x) + (25 / y) + (49 / z) ≥ 45 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 5 ∧ (9 / x) + (25 / y) + (49 / z) = 45 :=
by sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3542_354256


namespace NUMINAMATH_CALUDE_ab_value_l3542_354245

theorem ab_value (a b : ℝ) (h1 : a - b = 8) (h2 : a^2 + b^2 = 164) : a * b = 50 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3542_354245


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3542_354244

def complex_number : ℂ := Complex.I + Complex.I^2

theorem complex_number_in_second_quadrant :
  complex_number.re < 0 ∧ complex_number.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3542_354244


namespace NUMINAMATH_CALUDE_students_in_c_class_l3542_354240

theorem students_in_c_class (a b c : ℕ) : 
  a = 44 ∧ a + 2 = b ∧ b = c + 1 → c = 45 := by
  sorry

end NUMINAMATH_CALUDE_students_in_c_class_l3542_354240


namespace NUMINAMATH_CALUDE_alok_mixed_veg_order_l3542_354225

/-- Represents the order and payment details of Alok's meal --/
structure MealOrder where
  chapatis : ℕ
  rice_plates : ℕ
  ice_cream_cups : ℕ
  chapati_cost : ℕ
  rice_cost : ℕ
  mixed_veg_cost : ℕ
  total_paid : ℕ

/-- Calculates the number of mixed vegetable plates ordered --/
def mixed_veg_plates (order : MealOrder) : ℕ :=
  ((order.total_paid - (order.chapatis * order.chapati_cost + order.rice_plates * order.rice_cost)) / order.mixed_veg_cost)

/-- Theorem stating that Alok ordered 9 plates of mixed vegetable --/
theorem alok_mixed_veg_order : 
  ∀ (order : MealOrder), 
    order.chapatis = 16 ∧ 
    order.rice_plates = 5 ∧ 
    order.ice_cream_cups = 6 ∧
    order.chapati_cost = 6 ∧ 
    order.rice_cost = 45 ∧ 
    order.mixed_veg_cost = 70 ∧ 
    order.total_paid = 961 → 
    mixed_veg_plates order = 9 := by
  sorry


end NUMINAMATH_CALUDE_alok_mixed_veg_order_l3542_354225


namespace NUMINAMATH_CALUDE_total_candy_count_l3542_354232

theorem total_candy_count (chocolate_boxes : ℕ) (caramel_boxes : ℕ) (mint_boxes : ℕ) (berry_boxes : ℕ)
  (chocolate_caramel_pieces_per_box : ℕ) (mint_pieces_per_box : ℕ) (berry_pieces_per_box : ℕ)
  (h1 : chocolate_boxes = 7)
  (h2 : caramel_boxes = 3)
  (h3 : mint_boxes = 5)
  (h4 : berry_boxes = 4)
  (h5 : chocolate_caramel_pieces_per_box = 8)
  (h6 : mint_pieces_per_box = 10)
  (h7 : berry_pieces_per_box = 12) :
  chocolate_boxes * chocolate_caramel_pieces_per_box +
  caramel_boxes * chocolate_caramel_pieces_per_box +
  mint_boxes * mint_pieces_per_box +
  berry_boxes * berry_pieces_per_box = 178 := by
sorry

end NUMINAMATH_CALUDE_total_candy_count_l3542_354232


namespace NUMINAMATH_CALUDE_sin_330_degrees_l3542_354285

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l3542_354285


namespace NUMINAMATH_CALUDE_equation_solution_l3542_354286

theorem equation_solution : ∃! x : ℝ, Real.sqrt (7 * x - 3) + Real.sqrt (x^3 - 1) = 3 :=
  by sorry

end NUMINAMATH_CALUDE_equation_solution_l3542_354286


namespace NUMINAMATH_CALUDE_jebbs_take_home_pay_l3542_354228

/-- Calculates the take-home pay after tax -/
def takeHomePay (originalPay : ℝ) (taxRate : ℝ) : ℝ :=
  originalPay * (1 - taxRate)

/-- Proves that Jebb's take-home pay is $585 -/
theorem jebbs_take_home_pay :
  takeHomePay 650 0.1 = 585 := by
  sorry

end NUMINAMATH_CALUDE_jebbs_take_home_pay_l3542_354228


namespace NUMINAMATH_CALUDE_total_pages_read_l3542_354217

/-- Represents the number of pages in each chapter of the book --/
def pages_per_chapter : ℕ := 40

/-- Represents the number of chapters Mitchell read before 4 o'clock --/
def chapters_before_4 : ℕ := 10

/-- Represents the number of pages Mitchell read from the 11th chapter at 4 o'clock --/
def pages_at_4 : ℕ := 20

/-- Represents the number of additional chapters Mitchell read after 4 o'clock --/
def chapters_after_4 : ℕ := 2

/-- Theorem stating that the total number of pages Mitchell read is 500 --/
theorem total_pages_read : 
  pages_per_chapter * chapters_before_4 + 
  pages_at_4 + 
  pages_per_chapter * chapters_after_4 = 500 := by
sorry

end NUMINAMATH_CALUDE_total_pages_read_l3542_354217


namespace NUMINAMATH_CALUDE_distance_between_cities_l3542_354200

/-- Proves that the distance between two cities is 300 miles given specific travel conditions -/
theorem distance_between_cities (speed_david speed_lewis : ℝ) (meeting_point : ℝ) : 
  speed_david = 50 →
  speed_lewis = 70 →
  meeting_point = 250 →
  ∃ (time : ℝ), 
    time * speed_david = meeting_point ∧
    time * speed_lewis = 2 * 300 - meeting_point →
  300 = 300 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l3542_354200


namespace NUMINAMATH_CALUDE_intersection_A_B_quadratic_inequality_solution_l3542_354270

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 > 0}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem quadratic_inequality_solution (a b : ℝ) :
  ({x : ℝ | x^2 + a*x + b < 0} = {x | 2 < x ∧ x < 3}) ↔ (a = -5 ∧ b = 6) := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_quadratic_inequality_solution_l3542_354270


namespace NUMINAMATH_CALUDE_algebraic_simplification_l3542_354242

theorem algebraic_simplification (m n : ℝ) : 3 * m^2 * n - 3 * m^2 * n = 0 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l3542_354242


namespace NUMINAMATH_CALUDE_divisibility_of_fifth_power_differences_l3542_354212

theorem divisibility_of_fifth_power_differences (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (x - y) * (y - z) * (z - x)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_fifth_power_differences_l3542_354212


namespace NUMINAMATH_CALUDE_square_area_error_l3542_354218

theorem square_area_error (x : ℝ) (h : x > 0) :
  let measured_side := x * (1 + 0.17)
  let actual_area := x^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.3689 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l3542_354218


namespace NUMINAMATH_CALUDE_number_of_girls_l3542_354216

/-- Given a group of kids with boys and girls, prove the number of girls. -/
theorem number_of_girls (total : ℕ) (boys : ℕ) (girls : ℕ) : total = 9 ∧ boys = 6 → girls = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_l3542_354216


namespace NUMINAMATH_CALUDE_equation_solution_l3542_354221

open Real

theorem equation_solution (x : ℝ) :
  0 < x ∧ x < π →
  ((Real.sqrt 2014 - Real.sqrt 2013) ^ (tan x)^2 + 
   (Real.sqrt 2014 + Real.sqrt 2013) ^ (-(tan x)^2) = 
   2 * (Real.sqrt 2014 - Real.sqrt 2013)^3) ↔ 
  (x = π/3 ∨ x = 2*π/3) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l3542_354221


namespace NUMINAMATH_CALUDE_flour_for_nine_biscuits_l3542_354295

/-- The amount of flour needed to make a certain number of biscuits -/
def flour_needed (num_biscuits : ℕ) : ℝ :=
  sorry

theorem flour_for_nine_biscuits :
  let members : ℕ := 18
  let biscuits_per_member : ℕ := 2
  let total_flour : ℝ := 5
  flour_needed (members * biscuits_per_member) = total_flour →
  flour_needed 9 = 1.25 :=
by sorry

end NUMINAMATH_CALUDE_flour_for_nine_biscuits_l3542_354295


namespace NUMINAMATH_CALUDE_farmer_sheep_problem_l3542_354267

theorem farmer_sheep_problem (total : ℕ) 
  (h1 : total % 3 = 0)  -- First son's share is whole
  (h2 : total % 5 = 0)  -- Second son's share is whole
  (h3 : total % 6 = 0)  -- Third son's share is whole
  (h4 : total % 8 = 0)  -- Daughter's share is whole
  (h5 : total - (total / 3 + total / 5 + total / 6 + total / 8) = 12)  -- Charity's share
  : total = 68 := by
  sorry

end NUMINAMATH_CALUDE_farmer_sheep_problem_l3542_354267


namespace NUMINAMATH_CALUDE_solve_equation_l3542_354213

theorem solve_equation (x : ℝ) (h : x - 2*x + 3*x = 100) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3542_354213


namespace NUMINAMATH_CALUDE_expand_product_l3542_354252

theorem expand_product (x : ℝ) : 3 * (2 * x - 7) * (x + 9) = 6 * x^2 + 33 * x - 189 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3542_354252


namespace NUMINAMATH_CALUDE_garden_tilling_time_l3542_354254

/-- Represents a rectangular obstacle in the garden -/
structure Obstacle where
  length : ℝ
  width : ℝ

/-- Represents the garden plot and tilling parameters -/
structure GardenPlot where
  shortBase : ℝ
  longBase : ℝ
  height : ℝ
  tillerWidth : ℝ
  tillingRate : ℝ
  obstacles : List Obstacle
  extraTimePerObstacle : ℝ

/-- Calculates the time required to till the garden plot -/
def tillingTime (plot : GardenPlot) : ℝ :=
  sorry

/-- Theorem stating the tilling time for the given garden plot -/
theorem garden_tilling_time :
  let plot : GardenPlot := {
    shortBase := 135,
    longBase := 170,
    height := 90,
    tillerWidth := 2.5,
    tillingRate := 1.5 / 3,
    obstacles := [
      { length := 20, width := 10 },
      { length := 15, width := 30 },
      { length := 10, width := 15 }
    ],
    extraTimePerObstacle := 15
  }
  abs (tillingTime plot - 173.08) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_garden_tilling_time_l3542_354254


namespace NUMINAMATH_CALUDE_additional_coins_for_eight_friends_l3542_354276

/-- The minimum number of additional coins needed for unique distribution. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let total_needed := (num_friends * (num_friends + 1)) / 2
  if total_needed > initial_coins then
    total_needed - initial_coins
  else
    0

/-- Theorem: Given 8 friends and 28 initial coins, 8 additional coins are needed. -/
theorem additional_coins_for_eight_friends :
  min_additional_coins 8 28 = 8 := by
  sorry


end NUMINAMATH_CALUDE_additional_coins_for_eight_friends_l3542_354276


namespace NUMINAMATH_CALUDE_smallest_a_for_two_zeros_l3542_354253

/-- The function f(x) = x^2 - a*ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a * Real.log x

/-- The function g(x) = (a-2)*x -/
def g (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x

/-- The function F(x) = f(x) - g(x) -/
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

/-- The theorem stating that 3 is the smallest positive integer value of a 
    for which F(x) has exactly two zeros -/
theorem smallest_a_for_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ F 3 x₁ = 0 ∧ F 3 x₂ = 0 ∧
  ∀ (a : ℕ), a < 3 → ¬∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ F (a : ℝ) y₁ = 0 ∧ F (a : ℝ) y₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_for_two_zeros_l3542_354253


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l3542_354287

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - a

theorem max_value_implies_a_equals_one :
  (∃ (M : ℝ), M = 1 ∧ ∀ x ∈ Set.Icc 0 2, f a x ≤ M) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l3542_354287


namespace NUMINAMATH_CALUDE_radius_B_is_three_fifths_l3542_354203

/-- A structure representing the configuration of circles A, B, C, and D. -/
structure CircleConfiguration where
  /-- Radius of circle A -/
  radius_A : ℝ
  /-- Radius of circle B -/
  radius_B : ℝ
  /-- Radius of circle D -/
  radius_D : ℝ
  /-- Circles A, B, and C are externally tangent to each other -/
  externally_tangent : Prop
  /-- Circles A, B, and C are internally tangent to circle D -/
  internally_tangent : Prop
  /-- Circles B and C are congruent -/
  B_C_congruent : Prop
  /-- The center of D is tangent to circle A at one point -/
  D_center_tangent_A : Prop

/-- Theorem stating that given the specific configuration of circles, the radius of circle B is 3/5. -/
theorem radius_B_is_three_fifths (config : CircleConfiguration)
  (h1 : config.radius_A = 2)
  (h2 : config.radius_D = 3) :
  config.radius_B = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_radius_B_is_three_fifths_l3542_354203


namespace NUMINAMATH_CALUDE_x_negative_y_positive_l3542_354263

theorem x_negative_y_positive (x y : ℝ) 
  (h1 : 2 * x - y > 3 * x) 
  (h2 : x + 2 * y < 2 * y) : 
  x < 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_x_negative_y_positive_l3542_354263


namespace NUMINAMATH_CALUDE_sum_of_seven_step_palindromes_l3542_354280

/-- Reverses a natural number -/
def reverseNum (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a palindrome -/
def isPalindrome (n : ℕ) : Bool := sorry

/-- Performs one step of reversing and adding -/
def reverseAndAdd (n : ℕ) : ℕ := n + reverseNum n

/-- Checks if a number becomes a palindrome after exactly k steps -/
def isPalindromeAfterKSteps (n : ℕ) (k : ℕ) : Bool := sorry

/-- The set of three-digit numbers that become palindromes after exactly 7 steps -/
def sevenStepPalindromes : Finset ℕ := sorry

theorem sum_of_seven_step_palindromes :
  Finset.sum sevenStepPalindromes id = 1160 := by sorry

end NUMINAMATH_CALUDE_sum_of_seven_step_palindromes_l3542_354280


namespace NUMINAMATH_CALUDE_paige_finished_problems_l3542_354278

/-- Calculates the number of finished homework problems -/
def finished_problems (total : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  total - (remaining_pages * problems_per_page)

/-- Theorem: Paige finished 47 problems -/
theorem paige_finished_problems :
  finished_problems 110 7 9 = 47 := by
  sorry

end NUMINAMATH_CALUDE_paige_finished_problems_l3542_354278


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3542_354219

theorem rectangle_perimeter (x y : ℝ) 
  (h1 : 6 * x + 2 * y = 56)  -- perimeter of figure A
  (h2 : 4 * x + 6 * y = 56)  -- perimeter of figure B
  : 2 * x + 6 * y = 40 :=    -- perimeter of figure C
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3542_354219


namespace NUMINAMATH_CALUDE_expression_equivalence_l3542_354222

theorem expression_equivalence (a b c : ℝ) (hc : c ≠ 0) :
  ((a - 0.07 * a) + 0.05 * b) / c = (0.93 * a + 0.05 * b) / c := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l3542_354222


namespace NUMINAMATH_CALUDE_tic_tac_toe_ties_l3542_354201

theorem tic_tac_toe_ties (james_win_rate mary_win_rate : ℚ)
  (h1 : james_win_rate = 4 / 9)
  (h2 : mary_win_rate = 5 / 18) :
  1 - (james_win_rate + mary_win_rate) = 5 / 18 := by
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_ties_l3542_354201


namespace NUMINAMATH_CALUDE_smallest_two_digit_number_one_more_than_multiple_l3542_354251

theorem smallest_two_digit_number_one_more_than_multiple (n : ℕ) : n = 71 ↔ 
  (n ≥ 10 ∧ n < 100) ∧ 
  (∃ k : ℕ, n = 2 * k + 1 ∧ n = 5 * k + 1 ∧ n = 7 * k + 1) ∧
  (∀ m : ℕ, m < n → ¬(m ≥ 10 ∧ m < 100 ∧ ∃ k : ℕ, m = 2 * k + 1 ∧ m = 5 * k + 1 ∧ m = 7 * k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_number_one_more_than_multiple_l3542_354251


namespace NUMINAMATH_CALUDE_single_filter_price_l3542_354284

/-- The price of a camera lens filter kit containing 5 filters -/
def kit_price : ℝ := 87.50

/-- The price of the first type of filter -/
def filter1_price : ℝ := 16.45

/-- The price of the second type of filter -/
def filter2_price : ℝ := 14.05

/-- The discount rate when purchasing the kit -/
def discount_rate : ℝ := 0.08

/-- The number of filters of the first type -/
def num_filter1 : ℕ := 2

/-- The number of filters of the second type -/
def num_filter2 : ℕ := 2

/-- The number of filters of the unknown type -/
def num_filter3 : ℕ := 1

/-- The total number of filters in the kit -/
def total_filters : ℕ := num_filter1 + num_filter2 + num_filter3

theorem single_filter_price (x : ℝ) : 
  (num_filter1 : ℝ) * filter1_price + (num_filter2 : ℝ) * filter2_price + (num_filter3 : ℝ) * x = 
  kit_price / (1 - discount_rate) → x = 34.11 := by
  sorry

end NUMINAMATH_CALUDE_single_filter_price_l3542_354284


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_6_mod_17_l3542_354227

theorem least_five_digit_congruent_to_6_mod_17 :
  (∀ n : ℕ, 10000 ≤ n ∧ n < 10017 → ¬(n % 17 = 6)) ∧
  10017 % 17 = 6 := by
  sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_6_mod_17_l3542_354227


namespace NUMINAMATH_CALUDE_walking_distance_l3542_354266

/-- Proves that given a walking speed where 1 mile is covered in 20 minutes, 
    the distance covered in 40 minutes is 2 miles. -/
theorem walking_distance (speed : ℝ) (time : ℝ) : 
  speed = 1 / 20 → time = 40 → speed * time = 2 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_l3542_354266


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l3542_354202

/-- The equation (x-3)^2 = 9(y+2)^2 - 81 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b h k : ℝ) (A B : ℝ → ℝ → Prop),
    (∀ x y, A x y ↔ (x - 3)^2 = 9*(y + 2)^2 - 81) ∧
    (∀ x y, B x y ↔ ((x - h) / a)^2 - ((y - k) / b)^2 = 1) ∧
    (∀ x y, A x y ↔ B x y) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l3542_354202


namespace NUMINAMATH_CALUDE_different_tens_digit_probability_l3542_354291

theorem different_tens_digit_probability :
  let n : ℕ := 5  -- number of integers to choose
  let lower_bound : ℕ := 10  -- lower bound of the range
  let upper_bound : ℕ := 59  -- upper bound of the range
  let total_numbers : ℕ := upper_bound - lower_bound + 1  -- total numbers in the range
  let tens_digits : ℕ := 5  -- number of different tens digits in the range
  let numbers_per_tens : ℕ := 10  -- numbers available for each tens digit

  -- Probability of choosing n integers with different tens digits
  (numbers_per_tens ^ n : ℚ) / (total_numbers.choose n) = 2500 / 52969 :=
by sorry

end NUMINAMATH_CALUDE_different_tens_digit_probability_l3542_354291


namespace NUMINAMATH_CALUDE_shortest_time_to_camp_l3542_354220

/-- The shortest time to reach the camp across a river -/
theorem shortest_time_to_camp (river_width : ℝ) (camp_distance : ℝ) 
  (swim_speed : ℝ) (walk_speed : ℝ) (h1 : river_width = 1) 
  (h2 : camp_distance = 1) (h3 : swim_speed = 2) (h4 : walk_speed = 3) :
  ∃ (t : ℝ), t = (1 + Real.sqrt 13) / (3 * Real.sqrt 13) ∧ 
  (∀ (x : ℝ), x ≥ 0 ∧ x ≤ 1 → 
    t ≤ x / swim_speed + (camp_distance - Real.sqrt (river_width^2 - x^2)) / walk_speed) :=
by sorry

end NUMINAMATH_CALUDE_shortest_time_to_camp_l3542_354220


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3542_354241

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), 15 * x^2 - 7 * y^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3542_354241


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3542_354299

theorem quadratic_coefficient (b : ℝ) (p : ℝ) : 
  (∀ x, x^2 + b*x + 64 = (x + p)^2 + 16) → b = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3542_354299


namespace NUMINAMATH_CALUDE_union_A_B_l3542_354264

def A : Set ℝ := {x | (x - 2) / (x + 1) ≤ 0}

def B : Set ℝ := {x | -2 * x^2 + 7 * x + 4 > 0}

theorem union_A_B : A ∪ B = Set.Ioo (-1 : ℝ) 4 := by sorry

end NUMINAMATH_CALUDE_union_A_B_l3542_354264


namespace NUMINAMATH_CALUDE_max_product_of_roots_l3542_354204

theorem max_product_of_roots (m : ℝ) : 
  let product_of_roots := m / 5
  let discriminant := 100 - 20 * m
  (discriminant ≥ 0) →  -- Condition for real roots
  product_of_roots ≤ 1 ∧ 
  (product_of_roots = 1 ↔ m = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_product_of_roots_l3542_354204


namespace NUMINAMATH_CALUDE_complement_of_union_l3542_354268

def U : Set Nat := {1, 2, 3, 4}
def M : Set Nat := {1, 2}
def N : Set Nat := {2, 3}

theorem complement_of_union (U M N : Set Nat) 
  (hU : U = {1, 2, 3, 4}) 
  (hM : M = {1, 2}) 
  (hN : N = {2, 3}) : 
  (U \ (M ∪ N)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l3542_354268


namespace NUMINAMATH_CALUDE_device_marked_price_device_marked_price_is_59_l3542_354289

theorem device_marked_price (initial_price : ℝ) (purchase_discount : ℝ) 
  (profit_percentage : ℝ) (sale_discount : ℝ) : ℝ :=
  let purchase_price := initial_price * (1 - purchase_discount)
  let selling_price := purchase_price * (1 + profit_percentage)
  selling_price / (1 - sale_discount)

theorem device_marked_price_is_59 :
  device_marked_price 50 0.15 0.25 0.10 = 59 := by
  sorry

end NUMINAMATH_CALUDE_device_marked_price_device_marked_price_is_59_l3542_354289


namespace NUMINAMATH_CALUDE_correct_performance_calculation_l3542_354260

/-- Represents a batsman's performance in a cricket match -/
structure BatsmanPerformance where
  initialAverage : ℝ
  eleventhInningRuns : ℝ
  averageIncrease : ℝ
  teamHandicap : ℝ

/-- Calculates the new average and total team runs for a batsman -/
def calculatePerformance (performance : BatsmanPerformance) : ℝ × ℝ :=
  let newAverage := performance.initialAverage + performance.averageIncrease
  let totalBatsmanRuns := 11 * newAverage
  let totalTeamRuns := totalBatsmanRuns + performance.teamHandicap
  (newAverage, totalTeamRuns)

/-- Theorem stating the correct calculation of a batsman's performance -/
theorem correct_performance_calculation 
  (performance : BatsmanPerformance) 
  (h1 : performance.eleventhInningRuns = 85)
  (h2 : performance.averageIncrease = 5)
  (h3 : performance.teamHandicap = 75) :
  calculatePerformance performance = (35, 460) := by
  sorry

#check correct_performance_calculation

end NUMINAMATH_CALUDE_correct_performance_calculation_l3542_354260


namespace NUMINAMATH_CALUDE_baseball_bat_price_l3542_354257

-- Define the prices and quantities
def basketball_price : ℝ := 29
def basketball_quantity : ℕ := 10
def baseball_price : ℝ := 2.5
def baseball_quantity : ℕ := 14
def price_difference : ℝ := 237

-- Define the theorem
theorem baseball_bat_price :
  ∃ (bat_price : ℝ),
    (basketball_price * basketball_quantity) =
    (baseball_price * baseball_quantity + bat_price + price_difference) ∧
    bat_price = 18 := by
  sorry

end NUMINAMATH_CALUDE_baseball_bat_price_l3542_354257


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l3542_354271

/-- Given a hyperbola with specific properties, prove its standard equation. -/
theorem hyperbola_standard_equation (a : ℝ) (A : ℝ × ℝ) :
  a = 2 * Real.sqrt 5 →
  A = (2, -5) →
  ∃ (b : ℝ), ∀ (x y : ℝ),
    (y^2 / (a^2) - x^2 / b^2 = 1) ↔
    (y^2 / 20 - x^2 / 16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l3542_354271


namespace NUMINAMATH_CALUDE_otimes_equation_solution_l3542_354282

-- Define the ⊗ operation
noncomputable def otimes (a b : ℝ) : ℝ :=
  a * Real.sqrt (b + Real.sqrt (b + 1 + Real.sqrt (b + 2 + Real.sqrt (b + 3 + Real.sqrt (b + 4)))))

-- Theorem statement
theorem otimes_equation_solution (h : ℝ) :
  otimes 3 h = 15 → h = 20 := by
  sorry

end NUMINAMATH_CALUDE_otimes_equation_solution_l3542_354282


namespace NUMINAMATH_CALUDE_first_digit_base9_of_122012_base3_l3542_354247

/-- Converts a base 3 number to base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 3 * acc) 0

/-- Calculates the first digit of a number in base 9 -/
def firstDigitBase9 (n : Nat) : Nat :=
  if n < 9 then n else firstDigitBase9 (n / 9)

theorem first_digit_base9_of_122012_base3 :
  let y := base3ToBase10 [1, 2, 2, 0, 1, 2]
  firstDigitBase9 y = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_base9_of_122012_base3_l3542_354247


namespace NUMINAMATH_CALUDE_max_distance_42000km_l3542_354297

/-- Represents the maximum distance a car can travel with tire switching -/
def maxDistanceWithTireSwitch (frontTireLife rear_tire_life : ℕ) : ℕ :=
  min frontTireLife rear_tire_life

/-- Theorem stating the maximum distance a car can travel with specific tire lifespans -/
theorem max_distance_42000km (frontTireLife rearTireLife : ℕ) 
  (h1 : frontTireLife = 42000)
  (h2 : rearTireLife = 56000) :
  maxDistanceWithTireSwitch frontTireLife rearTireLife = 42000 :=
by
  sorry

#eval maxDistanceWithTireSwitch 42000 56000

end NUMINAMATH_CALUDE_max_distance_42000km_l3542_354297


namespace NUMINAMATH_CALUDE_third_year_compound_interest_l3542_354215

/-- Calculates compound interest for a given principal, rate, and number of years -/
def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

theorem third_year_compound_interest (P : ℝ) (r : ℝ) :
  r = 0.06 →
  compoundInterest P r 2 = 1200 →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |compoundInterest P r 3 - 1858.03| < ε :=
sorry

end NUMINAMATH_CALUDE_third_year_compound_interest_l3542_354215


namespace NUMINAMATH_CALUDE_pages_per_day_l3542_354259

theorem pages_per_day (total_pages : ℕ) (days : ℕ) (h1 : total_pages = 96) (h2 : days = 12) :
  total_pages / days = 8 := by
sorry

end NUMINAMATH_CALUDE_pages_per_day_l3542_354259


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l3542_354288

theorem smallest_positive_integer_with_remainders : ∃! x : ℕ+, 
  (x : ℤ) % 4 = 1 ∧ 
  (x : ℤ) % 5 = 2 ∧ 
  (x : ℤ) % 6 = 3 ∧ 
  ∀ y : ℕ+, 
    (y : ℤ) % 4 = 1 → 
    (y : ℤ) % 5 = 2 → 
    (y : ℤ) % 6 = 3 → 
    x ≤ y :=
by
  use 57
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l3542_354288


namespace NUMINAMATH_CALUDE_two_roots_theorem_l3542_354274

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y → f x < f y

def has_exactly_two_roots (f : ℝ → ℝ) : Prop :=
  ∃ a b, a < b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b

theorem two_roots_theorem (f : ℝ → ℝ) 
  (h1 : even_function f)
  (h2 : monotone_increasing_nonneg f)
  (h3 : f 1 * f 2 < 0) :
  has_exactly_two_roots f :=
sorry

end NUMINAMATH_CALUDE_two_roots_theorem_l3542_354274


namespace NUMINAMATH_CALUDE_special_isosceles_sine_l3542_354238

/-- An isosceles triangle with a special property on inscribed rectangles -/
structure SpecialIsoscelesTriangle where
  -- The vertex angle of the isosceles triangle
  vertex_angle : ℝ
  -- The base and height of the isosceles triangle
  base : ℝ
  height : ℝ
  -- The isosceles property
  isosceles : base = height
  -- The property that all inscribed rectangles with two vertices on the base have the same perimeter
  constant_perimeter : ∀ (x : ℝ), 0 ≤ x → x ≤ base → 
    2 * (x + (base * (height - x)) / height) = base + height

/-- The main theorem stating that the sine of the vertex angle is 4/5 -/
theorem special_isosceles_sine (t : SpecialIsoscelesTriangle) : 
  Real.sin t.vertex_angle = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_special_isosceles_sine_l3542_354238


namespace NUMINAMATH_CALUDE_max_regions_theorem_l3542_354265

/-- Represents a convex polygon with a given number of sides -/
structure ConvexPolygon where
  sides : ℕ

/-- Represents two convex polygons on a plane -/
structure TwoPolygonsOnPlane where
  polygon1 : ConvexPolygon
  polygon2 : ConvexPolygon
  sides_condition : polygon1.sides > polygon2.sides

/-- The maximum number of regions into which two convex polygons can divide a plane -/
def max_regions (polygons : TwoPolygonsOnPlane) : ℕ :=
  2 * polygons.polygon2.sides + 2

/-- Theorem stating the maximum number of regions formed by two convex polygons on a plane -/
theorem max_regions_theorem (polygons : TwoPolygonsOnPlane) :
  max_regions polygons = 2 * polygons.polygon2.sides + 2 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_theorem_l3542_354265


namespace NUMINAMATH_CALUDE_birthday_problem_l3542_354293

/-- The number of trainees -/
def n : ℕ := 62

/-- The number of days in a year -/
def d : ℕ := 365

/-- The probability of at least two trainees sharing a birthday -/
noncomputable def prob_shared_birthday : ℝ :=
  1 - (d.factorial / (d - n).factorial : ℝ) / d ^ n

theorem birthday_problem :
  ∃ (p : ℝ), prob_shared_birthday = p ∧ p > 0.9959095 ∧ p < 0.9959096 :=
sorry

end NUMINAMATH_CALUDE_birthday_problem_l3542_354293


namespace NUMINAMATH_CALUDE_plant_arrangement_count_l3542_354239

/-- Represents the number of basil plants -/
def basil_count : ℕ := 5

/-- Represents the number of tomato plants -/
def tomato_count : ℕ := 5

/-- Represents the total number of plant positions (basil + tomato block) -/
def total_positions : ℕ := basil_count + 1

/-- Calculates the number of ways to arrange the plants with given constraints -/
def plant_arrangements : ℕ :=
  (Nat.factorial total_positions) * (Nat.factorial tomato_count)

theorem plant_arrangement_count :
  plant_arrangements = 86400 := by sorry

end NUMINAMATH_CALUDE_plant_arrangement_count_l3542_354239


namespace NUMINAMATH_CALUDE_equality_of_powers_l3542_354272

theorem equality_of_powers (a b c d e f : ℕ+) 
  (h1 : 20^21 = 2^(a:ℕ) * 5^(b:ℕ))
  (h2 : 20^21 = 4^(c:ℕ) * 5^(d:ℕ))
  (h3 : 20^21 = 8^(e:ℕ) * 5^(f:ℕ)) :
  100 * (b:ℕ) * (d:ℕ) * (f:ℕ) / ((a:ℕ) * (c:ℕ) * (e:ℕ)) = 75 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_powers_l3542_354272


namespace NUMINAMATH_CALUDE_condition_equivalence_l3542_354229

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- Theorem statement
theorem condition_equivalence (a b : ℝ) :
  (a + b > 0) ↔ (f a + f b > 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_equivalence_l3542_354229


namespace NUMINAMATH_CALUDE_hike_distance_l3542_354223

/-- The distance between two points given specific movement conditions -/
theorem hike_distance (A B C : ℝ × ℝ) : 
  B.1 - A.1 = 5 ∧ 
  B.2 - A.2 = 0 ∧ 
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 64 ∧ 
  C.1 - B.1 = C.2 - B.2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 89 + 40 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hike_distance_l3542_354223


namespace NUMINAMATH_CALUDE_sum_of_ages_l3542_354279

theorem sum_of_ages (marie_age marco_age : ℕ) : 
  marie_age = 12 → 
  marco_age = 2 * marie_age + 1 → 
  marie_age + marco_age = 37 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3542_354279


namespace NUMINAMATH_CALUDE_gold_bar_worth_l3542_354243

/-- Proves that the worth of each gold bar is $20,000 given the specified conditions -/
theorem gold_bar_worth (rows : ℕ) (bars_per_row : ℕ) (total_worth : ℕ) : ℕ :=
  by
  -- Define the given conditions
  have h1 : rows = 4 := by sorry
  have h2 : bars_per_row = 20 := by sorry
  have h3 : total_worth = 1600000 := by sorry

  -- Calculate the total number of gold bars
  let total_bars := rows * bars_per_row

  -- Calculate the worth of each gold bar
  let bar_worth := total_worth / total_bars

  -- Prove that bar_worth equals 20000
  sorry

-- The theorem statement
#check gold_bar_worth

end NUMINAMATH_CALUDE_gold_bar_worth_l3542_354243


namespace NUMINAMATH_CALUDE_blocks_added_l3542_354277

def initial_blocks : ℕ := 35
def final_blocks : ℕ := 65

theorem blocks_added : final_blocks - initial_blocks = 30 := by
  sorry

end NUMINAMATH_CALUDE_blocks_added_l3542_354277


namespace NUMINAMATH_CALUDE_functional_equation_bound_l3542_354255

/-- Given real-valued functions f and g defined on ℝ satisfying certain conditions,
    prove that |g(y)| ≤ 1 for all y. -/
theorem functional_equation_bound (f g : ℝ → ℝ)
  (h1 : ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y)
  (h2 : ∀ x, f x ≠ 0)
  (h3 : ∀ x, |f x| ≤ 1) :
  ∀ y, |g y| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_bound_l3542_354255


namespace NUMINAMATH_CALUDE_tan_product_identity_l3542_354224

theorem tan_product_identity : (1 + Real.tan (3 * π / 180)) * (1 + Real.tan (42 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_identity_l3542_354224


namespace NUMINAMATH_CALUDE_train_crossing_time_l3542_354269

/-- Proves that a train 200 meters long, traveling at 144 km/hr, will take 5 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 200 →
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * (5/18)) →
  crossing_time = 5 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3542_354269


namespace NUMINAMATH_CALUDE_neighbor_eggs_taken_neighbor_took_12_eggs_l3542_354249

/-- Calculates the number of eggs taken by the neighbor given the conditions of Myrtle's hens and absence. -/
theorem neighbor_eggs_taken (num_hens : ℕ) (eggs_per_hen_per_day : ℕ) (days_gone : ℕ) 
  (eggs_dropped : ℕ) (eggs_remaining : ℕ) : ℕ :=
  let total_eggs := num_hens * eggs_per_hen_per_day * days_gone
  let eggs_collected := eggs_remaining + eggs_dropped
  total_eggs - eggs_collected

/-- Proves that the neighbor took 12 eggs given Myrtle's specific situation. -/
theorem neighbor_took_12_eggs : 
  neighbor_eggs_taken 3 3 7 5 46 = 12 := by
  sorry

end NUMINAMATH_CALUDE_neighbor_eggs_taken_neighbor_took_12_eggs_l3542_354249


namespace NUMINAMATH_CALUDE_two_solutions_for_x_squared_minus_y_squared_77_l3542_354246

theorem two_solutions_for_x_squared_minus_y_squared_77 :
  ∃! n : ℕ, n > 0 ∧ 
  (∃ s : Finset (ℕ × ℕ), s.card = n ∧ 
    (∀ p ∈ s, p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 - p.2^2 = 77) ∧
    (∀ x y : ℕ, x > 0 → y > 0 → x^2 - y^2 = 77 → (x, y) ∈ s)) :=
sorry

end NUMINAMATH_CALUDE_two_solutions_for_x_squared_minus_y_squared_77_l3542_354246


namespace NUMINAMATH_CALUDE_exam_score_distribution_l3542_354235

/-- Represents the score distribution of a class -/
structure ScoreDistribution where
  mean : ℝ
  stdDev : ℝ
  totalStudents : ℕ

/-- Calculates the number of students who scored at least a given threshold -/
def studentsAboveThreshold (dist : ScoreDistribution) (threshold : ℝ) : ℕ :=
  sorry

/-- The exam score distribution -/
def examScores : ScoreDistribution :=
  { mean := 110
    stdDev := 10
    totalStudents := 50 }

theorem exam_score_distribution :
  (studentsAboveThreshold examScores 90 = 49) ∧
  (studentsAboveThreshold examScores 120 = 8) := by
  sorry

end NUMINAMATH_CALUDE_exam_score_distribution_l3542_354235


namespace NUMINAMATH_CALUDE_equation_solution_l3542_354273

theorem equation_solution : 
  ∃! y : ℚ, (4 * y - 5) / (5 * y - 15) = 7 / 10 ∧ y = -11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3542_354273


namespace NUMINAMATH_CALUDE_complex_norm_problem_l3542_354296

theorem complex_norm_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - 2 * w) = 30)
  (h2 : Complex.abs (z + 3 * w) = 6)
  (h3 : Complex.abs (z - w) = 3) :
  Complex.abs z = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_norm_problem_l3542_354296


namespace NUMINAMATH_CALUDE_ratio_problem_l3542_354262

theorem ratio_problem (x : ℝ) : (x / 10 = 15 / 1) → x = 150 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3542_354262


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3542_354214

theorem quadratic_inequality_solution_set (x : ℝ) :
  (∃ y ∈ Set.Icc (24 - 2 * Real.sqrt 19) (24 + 2 * Real.sqrt 19), x = y) ↔ 
  x^2 - 48*x + 500 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3542_354214


namespace NUMINAMATH_CALUDE_solution_equations_solution_inequalities_l3542_354226

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3 * x - 4 * y = 1
def equation2 (x y : ℝ) : Prop := 5 * x + 2 * y = 6

-- Define the system of inequalities
def inequality1 (x : ℝ) : Prop := 3 * x + 6 > 0
def inequality2 (x : ℝ) : Prop := x - 2 < -x

-- Theorem for the system of equations
theorem solution_equations :
  ∃! (x y : ℝ), equation1 x y ∧ equation2 x y ∧ x = 1 ∧ y = 1/2 := by sorry

-- Theorem for the system of inequalities
theorem solution_inequalities :
  ∀ x : ℝ, inequality1 x ∧ inequality2 x ↔ -2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_equations_solution_inequalities_l3542_354226


namespace NUMINAMATH_CALUDE_office_paper_cost_l3542_354209

/-- Represents a type of bond paper -/
structure BondPaper where
  sheets_per_ream : ℕ
  cost_per_ream : ℕ

/-- Calculates the number of reams needed, rounding up -/
def reams_needed (sheets_required : ℕ) (paper : BondPaper) : ℕ :=
  (sheets_required + paper.sheets_per_ream - 1) / paper.sheets_per_ream

/-- Calculates the cost for a given number of reams -/
def cost_for_reams (reams : ℕ) (paper : BondPaper) : ℕ :=
  reams * paper.cost_per_ream

theorem office_paper_cost :
  let type_a : BondPaper := ⟨500, 27⟩
  let type_b : BondPaper := ⟨400, 24⟩
  let type_c : BondPaper := ⟨300, 18⟩
  let total_sheets : ℕ := 5000
  let min_a_sheets : ℕ := 2500
  let min_b_sheets : ℕ := 1500
  let remaining_sheets : ℕ := total_sheets - min_a_sheets - min_b_sheets
  let reams_a : ℕ := reams_needed min_a_sheets type_a
  let reams_b : ℕ := reams_needed min_b_sheets type_b
  let reams_c : ℕ := reams_needed remaining_sheets type_c
  let total_cost : ℕ := cost_for_reams reams_a type_a +
                        cost_for_reams reams_b type_b +
                        cost_for_reams reams_c type_c
  total_cost = 303 := by
  sorry

end NUMINAMATH_CALUDE_office_paper_cost_l3542_354209


namespace NUMINAMATH_CALUDE_mikes_music_store_spending_l3542_354250

/-- The amount Mike spent on the trumpet -/
def trumpet_cost : ℚ := 145.16

/-- The amount Mike spent on the song book -/
def song_book_cost : ℚ := 5.84

/-- The total amount Mike spent at the music store -/
def total_spent : ℚ := trumpet_cost + song_book_cost

/-- Theorem stating that the total amount Mike spent is $151.00 -/
theorem mikes_music_store_spending :
  total_spent = 151.00 := by sorry

end NUMINAMATH_CALUDE_mikes_music_store_spending_l3542_354250


namespace NUMINAMATH_CALUDE_min_value_problem_l3542_354258

theorem min_value_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1 / (a + 3) + 1 / (b + 3) + 1 / (c + 3) = 1 / 4) :
  22.75 ≤ a + 3 * b + 2 * c ∧ ∃ (a₀ b₀ c₀ : ℝ), 
    0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧
    1 / (a₀ + 3) + 1 / (b₀ + 3) + 1 / (c₀ + 3) = 1 / 4 ∧
    a₀ + 3 * b₀ + 2 * c₀ = 22.75 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l3542_354258


namespace NUMINAMATH_CALUDE_min_sum_factors_l3542_354294

theorem min_sum_factors (n : ℕ) (hn : n = 2025) :
  ∃ (a b : ℕ), a * b = n ∧ a > 0 ∧ b > 0 ∧
  (∀ (x y : ℕ), x * y = n → x > 0 → y > 0 → a + b ≤ x + y) ∧
  a + b = 90 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_factors_l3542_354294


namespace NUMINAMATH_CALUDE_integral_f_equals_pi_over_four_l3542_354290

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + Real.tan (Real.sqrt 2 * x))

theorem integral_f_equals_pi_over_four :
  ∫ x in (0)..(Real.pi / 2), f x = Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_integral_f_equals_pi_over_four_l3542_354290


namespace NUMINAMATH_CALUDE_altitude_polynomial_l3542_354237

/-- Given a cubic polynomial with rational coefficients whose roots are the side lengths of a triangle,
    the altitudes of this triangle are roots of a polynomial of sixth degree with rational coefficients. -/
theorem altitude_polynomial (a b c d : ℚ) (r₁ r₂ r₃ : ℝ) :
  (∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  (r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) →
  (r₁ + r₂ > r₃ ∧ r₂ + r₃ > r₁ ∧ r₃ + r₁ > r₂) →
  ∃ (p q s t u v w : ℚ),
    ∀ h₁ h₂ h₃ : ℝ,
      (h₁ = 2 * (Real.sqrt (((r₁ + r₂ + r₃) / 2) * ((r₁ + r₂ + r₃) / 2 - r₁) * ((r₁ + r₂ + r₃) / 2 - r₂) * ((r₁ + r₂ + r₃) / 2 - r₃))) / r₁ ∧
       h₂ = 2 * (Real.sqrt (((r₁ + r₂ + r₃) / 2) * ((r₁ + r₂ + r₃) / 2 - r₁) * ((r₁ + r₂ + r₃) / 2 - r₂) * ((r₁ + r₂ + r₃) / 2 - r₃))) / r₂ ∧
       h₃ = 2 * (Real.sqrt (((r₁ + r₂ + r₃) / 2) * ((r₁ + r₂ + r₃) / 2 - r₁) * ((r₁ + r₂ + r₃) / 2 - r₂) * ((r₁ + r₂ + r₃) / 2 - r₃))) / r₃) →
      p * h₁^6 + q * h₁^5 + s * h₁^4 + t * h₁^3 + u * h₁^2 + v * h₁ + w = 0 ∧
      p * h₂^6 + q * h₂^5 + s * h₂^4 + t * h₂^3 + u * h₂^2 + v * h₂ + w = 0 ∧
      p * h₃^6 + q * h₃^5 + s * h₃^4 + t * h₃^3 + u * h₃^2 + v * h₃ + w = 0 :=
by sorry

end NUMINAMATH_CALUDE_altitude_polynomial_l3542_354237


namespace NUMINAMATH_CALUDE_negative_expression_l3542_354298

theorem negative_expression : 
  -(-2) > 0 ∧ (-1)^2023 < 0 ∧ |-1^2| > 0 ∧ (-5)^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_expression_l3542_354298


namespace NUMINAMATH_CALUDE_mark_has_10_fewer_cards_l3542_354207

/-- The number of Pokemon cards each person has. -/
structure CardCounts where
  lloyd : ℕ
  mark : ℕ
  michael : ℕ

/-- The conditions of the Pokemon card problem. -/
def PokemonCardProblem (c : CardCounts) : Prop :=
  c.mark = 3 * c.lloyd ∧
  c.mark < c.michael ∧
  c.michael = 100 ∧
  c.lloyd + c.mark + c.michael + 80 = 300

/-- The theorem stating that Mark has 10 fewer cards than Michael. -/
theorem mark_has_10_fewer_cards (c : CardCounts) 
  (h : PokemonCardProblem c) : c.michael - c.mark = 10 := by
  sorry

end NUMINAMATH_CALUDE_mark_has_10_fewer_cards_l3542_354207


namespace NUMINAMATH_CALUDE_chessboard_division_theorem_l3542_354283

/-- Represents a 6x6 chessboard --/
def Chessboard := Fin 6 → Fin 6 → Bool

/-- Represents a 2x1 domino on the chessboard --/
structure Domino where
  x : Fin 6
  y : Fin 6
  horizontal : Bool

/-- A configuration of dominoes on the chessboard --/
def DominoConfiguration := List Domino

/-- Checks if a given line (horizontal or vertical) intersects any domino --/
def lineIntersectsDomino (line : Nat) (horizontal : Bool) (config : DominoConfiguration) : Bool :=
  sorry

/-- The main theorem --/
theorem chessboard_division_theorem (config : DominoConfiguration) :
  config.length = 18 → ∃ (line : Nat) (horizontal : Bool),
    line < 6 ∧ ¬lineIntersectsDomino line horizontal config :=
  sorry

end NUMINAMATH_CALUDE_chessboard_division_theorem_l3542_354283


namespace NUMINAMATH_CALUDE_horner_method_correct_f_3_equals_283_l3542_354292

def f (x : ℝ) : ℝ := x^5 + x^3 + x^2 + x + 1

def horner_eval (x : ℝ) : ℝ := ((((1 * x + 0) * x + 1) * x + 1) * x + 1) * x + 1

theorem horner_method_correct (x : ℝ) : f x = horner_eval x := by sorry

theorem f_3_equals_283 : f 3 = 283 := by sorry

end NUMINAMATH_CALUDE_horner_method_correct_f_3_equals_283_l3542_354292


namespace NUMINAMATH_CALUDE_smallest_mu_inequality_l3542_354275

theorem smallest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  (∀ μ : ℝ, (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + 4*b^2 + 4*c^2 + d^2 ≥ 2*a*b + μ*b*c + 2*c*d) → μ ≥ 6) ∧
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + 4*b^2 + 4*c^2 + d^2 ≥ 2*a*b + 6*b*c + 2*c*d) :=
by sorry

end NUMINAMATH_CALUDE_smallest_mu_inequality_l3542_354275


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3542_354210

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 5*x < 10 ↔ -5 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3542_354210


namespace NUMINAMATH_CALUDE_rubber_boat_fall_time_l3542_354248

/-- Represents the speed of the ship in still water -/
def ship_speed : ℝ := sorry

/-- Represents the speed of the water flow -/
def water_flow : ℝ := sorry

/-- Represents the time (in hours) when the rubber boat fell into the water, before 5 PM -/
def fall_time : ℝ := sorry

/-- Represents the fact that the ship catches up with the rubber boat after 1 hour -/
axiom catch_up_condition : (5 - fall_time) * (ship_speed - water_flow) + (6 - fall_time) * water_flow = ship_speed + water_flow

theorem rubber_boat_fall_time : fall_time = 4 := by sorry

end NUMINAMATH_CALUDE_rubber_boat_fall_time_l3542_354248


namespace NUMINAMATH_CALUDE_students_on_left_side_l3542_354261

theorem students_on_left_side (total : ℕ) (right : ℕ) (h1 : total = 63) (h2 : right = 27) :
  total - right = 36 := by
  sorry

end NUMINAMATH_CALUDE_students_on_left_side_l3542_354261


namespace NUMINAMATH_CALUDE_mothers_birthday_knowledge_l3542_354230

/-- Represents the distribution of students' knowledge about their parents' birthdays -/
structure BirthdayKnowledge where
  total : ℕ
  only_father : ℕ
  only_mother : ℕ
  both_parents : ℕ
  neither_parent : ℕ

/-- Theorem stating that 22 students know their mother's birthday -/
theorem mothers_birthday_knowledge (bk : BirthdayKnowledge) 
  (h1 : bk.total = 40)
  (h2 : bk.only_father = 10)
  (h3 : bk.only_mother = 12)
  (h4 : bk.both_parents = 22)
  (h5 : bk.neither_parent = 26)
  (h6 : bk.total = bk.only_father + bk.only_mother + bk.both_parents + bk.neither_parent) :
  bk.only_mother + bk.both_parents = 22 := by
  sorry

end NUMINAMATH_CALUDE_mothers_birthday_knowledge_l3542_354230


namespace NUMINAMATH_CALUDE_dilation_complex_mapping_l3542_354206

theorem dilation_complex_mapping :
  let center : ℂ := 2 - 3*I
  let scale_factor : ℝ := 3
  let original : ℂ := (5 - 4*I) / 3
  let image : ℂ := 1 + 2*I
  (image - center) = scale_factor • (original - center) := by sorry

end NUMINAMATH_CALUDE_dilation_complex_mapping_l3542_354206


namespace NUMINAMATH_CALUDE_parabola_properties_l3542_354234

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem parabola_properties (a b c : ℝ) 
  (h_a_nonzero : a ≠ 0)
  (h_c_gt_3 : c > 3)
  (h_passes_through : parabola a b c 5 = 0)
  (h_symmetry_axis : -b / (2 * a) = 2) :
  (a * b * c < 0) ∧ 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ parabola a b c x₁ = 2 ∧ parabola a b c x₂ = 2) ∧
  (a < -3/5) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3542_354234


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3542_354233

theorem complex_equation_solution (a : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a * i) / (2 - i) = (1 - 2*i) / 5) : a = -1 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3542_354233


namespace NUMINAMATH_CALUDE_a_periodic_with_period_5_l3542_354231

/-- The sequence a_n defined as 6^n mod 100 -/
def a (n : ℕ) : ℕ := (6^n) % 100

/-- The period of the sequence a_n -/
def period : ℕ := 5

theorem a_periodic_with_period_5 :
  (∀ n ≥ 2, a (n + period) = a n) ∧
  (∀ k < period, ∃ m ≥ 2, a (m + k) ≠ a m) :=
sorry

end NUMINAMATH_CALUDE_a_periodic_with_period_5_l3542_354231


namespace NUMINAMATH_CALUDE_sign_determination_l3542_354205

theorem sign_determination (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a / 5 > 0)
  (h2 : -b / (7*a) > 0)
  (h3 : 11 / (a*b*c) > 0)
  (h4 : -18 / (a*b*c*d) > 0) :
  a > 0 ∧ b < 0 ∧ c < 0 ∧ d < 0 := by
sorry

end NUMINAMATH_CALUDE_sign_determination_l3542_354205


namespace NUMINAMATH_CALUDE_expansion_and_binomial_coeff_l3542_354208

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the sum of binomial coefficients for (a + b)^n
def sumBinomialCoeff (n : ℕ) : ℕ := sorry

-- Define the coefficient of the third term in (a + b)^n
def thirdTermCoeff (n : ℕ) : ℕ := sorry

theorem expansion_and_binomial_coeff :
  -- Part I: The term containing 1/x^2 in (2x^2 + 1/x)^5
  (binomial 5 4) * 2 = 10 ∧
  -- Part II: If sum of binomial coefficients in (2x^2 + 1/x)^5 is 28 less than
  -- the coefficient of the third term in (√x + 2/x)^n, then n = 6
  ∃ n : ℕ, sumBinomialCoeff 5 = thirdTermCoeff n - 28 → n = 6 :=
by sorry

end NUMINAMATH_CALUDE_expansion_and_binomial_coeff_l3542_354208
