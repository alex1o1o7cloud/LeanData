import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_7_24_25_l1842_184226

theorem right_triangle_7_24_25 : 
  ∀ (a b c : ℝ), a = 7 ∧ b = 24 ∧ c = 25 → a^2 + b^2 = c^2 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_7_24_25_l1842_184226


namespace NUMINAMATH_CALUDE_transformation_theorem_l1842_184241

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the transformation g
def g : ℝ → ℝ := sorry

-- Theorem statement
theorem transformation_theorem :
  ∀ x : ℝ, g x = -f (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_transformation_theorem_l1842_184241


namespace NUMINAMATH_CALUDE_paving_stone_width_l1842_184213

/-- Proves that the width of each paving stone is 2 meters given the courtyard dimensions,
    number of paving stones, and length of each paving stone. -/
theorem paving_stone_width
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (num_stones : ℕ)
  (stone_length : ℝ)
  (h1 : courtyard_length = 40)
  (h2 : courtyard_width = 33/2)
  (h3 : num_stones = 132)
  (h4 : stone_length = 5/2)
  : ∃ (stone_width : ℝ), stone_width = 2 ∧ 
    courtyard_length * courtyard_width = (stone_length * stone_width) * num_stones :=
by
  sorry


end NUMINAMATH_CALUDE_paving_stone_width_l1842_184213


namespace NUMINAMATH_CALUDE_election_result_l1842_184269

theorem election_result (total_votes : ℕ) (invalid_percent : ℚ) (second_candidate_votes : ℕ) :
  total_votes = 7500 →
  invalid_percent = 20 / 100 →
  second_candidate_votes = 2700 →
  (↑((total_votes * (1 - invalid_percent)).floor - second_candidate_votes) / ↑((total_votes * (1 - invalid_percent)).floor) : ℚ) = 55 / 100 := by
  sorry

end NUMINAMATH_CALUDE_election_result_l1842_184269


namespace NUMINAMATH_CALUDE_ndfl_calculation_l1842_184291

/-- Calculates the personal income tax (NDFL) for a Russian resident --/
def calculate_ndfl (monthly_income : ℚ) (bonus : ℚ) (car_sale : ℚ) (land_purchase : ℚ) : ℚ :=
  let annual_income := monthly_income * 12 + bonus + car_sale
  let total_deductions := car_sale + land_purchase
  let taxable_income := max (annual_income - total_deductions) 0
  let tax_rate := 13 / 100
  taxable_income * tax_rate

/-- Theorem stating that the NDFL for the given conditions is 10400 rubles --/
theorem ndfl_calculation :
  calculate_ndfl 30000 20000 250000 300000 = 10400 := by
  sorry

end NUMINAMATH_CALUDE_ndfl_calculation_l1842_184291


namespace NUMINAMATH_CALUDE_difference_of_squares_form_l1842_184200

theorem difference_of_squares_form (x y : ℝ) :
  ∃ (a b : ℝ), (2*x + y) * (y - 2*x) = -(a^2 - b^2) :=
sorry

end NUMINAMATH_CALUDE_difference_of_squares_form_l1842_184200


namespace NUMINAMATH_CALUDE_twenty_four_bananas_cost_l1842_184280

/-- The cost of fruits at Lisa's Fruit Stand -/
structure FruitCost where
  banana_apple_ratio : ℚ  -- 4 bananas = 3 apples
  apple_orange_ratio : ℚ  -- 8 apples = 5 oranges

/-- Calculate the number of oranges equivalent in cost to a given number of bananas -/
def bananas_to_oranges (cost : FruitCost) (num_bananas : ℕ) : ℚ :=
  let apples := (num_bananas : ℚ) * cost.banana_apple_ratio
  apples * cost.apple_orange_ratio

/-- Theorem: 24 bananas cost approximately as much as 11 oranges -/
theorem twenty_four_bananas_cost (cost : FruitCost) 
  (h1 : cost.banana_apple_ratio = 3 / 4)
  (h2 : cost.apple_orange_ratio = 5 / 8) :
  ⌊bananas_to_oranges cost 24⌋ = 11 := by
  sorry

#eval ⌊(24 : ℚ) * (3 / 4) * (5 / 8)⌋  -- Expected output: 11

end NUMINAMATH_CALUDE_twenty_four_bananas_cost_l1842_184280


namespace NUMINAMATH_CALUDE_tom_tim_typing_ratio_l1842_184242

/-- 
Given that Tim and Tom can type 12 pages in one hour together,
and 14 pages when Tom increases his speed by 25%,
prove that the ratio of Tom's normal typing speed to Tim's is 2:1
-/
theorem tom_tim_typing_ratio :
  ∀ (tim_speed tom_speed : ℝ),
    tim_speed + tom_speed = 12 →
    tim_speed + (1.25 * tom_speed) = 14 →
    tom_speed / tim_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_tom_tim_typing_ratio_l1842_184242


namespace NUMINAMATH_CALUDE_dog_age_difference_l1842_184233

/-- Proves that the 5th fastest dog is 20 years older than the 4th fastest dog --/
theorem dog_age_difference :
  let dog1_age : ℕ := 10
  let dog2_age : ℕ := dog1_age - 2
  let dog3_age : ℕ := dog2_age + 4
  let dog4_age : ℕ := dog3_age / 2
  let dog5_age : ℕ := dog4_age + 20
  (dog1_age + dog5_age) / 2 = 18 →
  dog5_age - dog4_age = 20 := by
sorry

end NUMINAMATH_CALUDE_dog_age_difference_l1842_184233


namespace NUMINAMATH_CALUDE_movie_cost_ratio_l1842_184236

/-- Proves that the ratio of the cost per minute of the new movie to the previous movie is 1/5 -/
theorem movie_cost_ratio :
  let previous_length : ℝ := 2 * 60  -- in minutes
  let new_length : ℝ := previous_length * 1.6
  let previous_cost_per_minute : ℝ := 50
  let total_new_cost : ℝ := 1920
  let new_cost_per_minute : ℝ := total_new_cost / new_length
  new_cost_per_minute / previous_cost_per_minute = 1 / 5 := by
sorry


end NUMINAMATH_CALUDE_movie_cost_ratio_l1842_184236


namespace NUMINAMATH_CALUDE_ali_remaining_money_l1842_184287

def calculate_remaining_money (initial_amount : ℚ) : ℚ :=
  let after_food := initial_amount * (1 - 3/8)
  let after_glasses := after_food * (1 - 2/5)
  let after_gift := after_glasses * (1 - 1/4)
  after_gift

theorem ali_remaining_money :
  calculate_remaining_money 480 = 135 := by
  sorry

end NUMINAMATH_CALUDE_ali_remaining_money_l1842_184287


namespace NUMINAMATH_CALUDE_jesses_rooms_l1842_184288

theorem jesses_rooms (room_length : ℝ) (room_width : ℝ) (total_carpet : ℝ) :
  room_length = 19 →
  room_width = 18 →
  total_carpet = 6840 →
  (total_carpet / (room_length * room_width) : ℝ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_jesses_rooms_l1842_184288


namespace NUMINAMATH_CALUDE_distance_swum_against_current_l1842_184248

/-- The distance swum against the current given swimming speed, current speed, and time taken -/
theorem distance_swum_against_current 
  (swimming_speed : ℝ) 
  (current_speed : ℝ) 
  (time_taken : ℝ) 
  (h1 : swimming_speed = 4)
  (h2 : current_speed = 2)
  (h3 : time_taken = 6) : 
  (swimming_speed - current_speed) * time_taken = 12 := by
  sorry

#check distance_swum_against_current

end NUMINAMATH_CALUDE_distance_swum_against_current_l1842_184248


namespace NUMINAMATH_CALUDE_correct_d_value_l1842_184290

/-- The exchange rate from U.S. dollars to Mexican pesos -/
def exchange_rate : ℚ := 13 / 9

/-- The amount of pesos spent -/
def pesos_spent : ℕ := 117

/-- The function that calculates the remaining pesos after exchange and spending -/
def remaining_pesos (d : ℕ) : ℚ := exchange_rate * d - pesos_spent

/-- The theorem stating that 264 is the correct value for d -/
theorem correct_d_value : ∃ (d : ℕ), d = 264 ∧ remaining_pesos d = d := by sorry

end NUMINAMATH_CALUDE_correct_d_value_l1842_184290


namespace NUMINAMATH_CALUDE_f_xy_second_derivative_not_exists_l1842_184299

noncomputable def f (x y : ℝ) : ℝ :=
  if x^2 + y^4 ≠ 0 then (x * y^2) / (x^2 + y^4) else 0

theorem f_xy_second_derivative_not_exists :
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ,
    x^2 + y^2 < δ^2 → |((f (x + y) y - f x y) / y - (f x y - f x 0) / y) / x - L| < ε :=
sorry

end NUMINAMATH_CALUDE_f_xy_second_derivative_not_exists_l1842_184299


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l1842_184253

def y : ℕ := 2^3 * 3^4 * 5^6 * 7^8 * 8^9 * 9^10

theorem smallest_factor_for_perfect_square :
  (∀ m : ℕ, m > 0 ∧ m < 2 → ¬ ∃ k : ℕ, m * y = k^2) ∧
  ∃ k : ℕ, 2 * y = k^2 :=
sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_square_l1842_184253


namespace NUMINAMATH_CALUDE_phillips_remaining_money_l1842_184296

theorem phillips_remaining_money
  (initial_amount : ℕ)
  (spent_oranges : ℕ)
  (spent_apples : ℕ)
  (spent_candy : ℕ)
  (h1 : initial_amount = 95)
  (h2 : spent_oranges = 14)
  (h3 : spent_apples = 25)
  (h4 : spent_candy = 6) :
  initial_amount - (spent_oranges + spent_apples + spent_candy) = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_phillips_remaining_money_l1842_184296


namespace NUMINAMATH_CALUDE_brad_daily_reading_l1842_184221

/-- Brad's daily reading in pages -/
def brad_pages : ℕ := 26

/-- Greg's daily reading in pages -/
def greg_pages : ℕ := 18

/-- The difference in pages read between Brad and Greg -/
def page_difference : ℕ := 8

theorem brad_daily_reading :
  brad_pages = greg_pages + page_difference :=
by sorry

end NUMINAMATH_CALUDE_brad_daily_reading_l1842_184221


namespace NUMINAMATH_CALUDE_rectangle_ratio_theorem_l1842_184205

theorem rectangle_ratio_theorem (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_a_le_b : a ≤ b) :
  (a / b = (a + b) / Real.sqrt (a^2 + b^2)) →
  (a / b)^2 = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_theorem_l1842_184205


namespace NUMINAMATH_CALUDE_comic_book_frames_l1842_184209

/-- The number of frames in Julian's comic book -/
def total_frames : ℕ := 143

/-- The number of frames per page if Julian puts them equally on 13 pages -/
def frames_per_page : ℕ := 11

/-- The number of pages if Julian puts 11 frames on each page -/
def number_of_pages : ℕ := 13

/-- Theorem stating that the total number of frames is correct -/
theorem comic_book_frames : 
  total_frames = frames_per_page * number_of_pages :=
by sorry

end NUMINAMATH_CALUDE_comic_book_frames_l1842_184209


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1842_184225

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) := Real.log x / Real.log (1/2)

-- Statement of the theorem
theorem sufficient_not_necessary :
  (∀ x : ℝ, x > 1 → log_half (x + 2) < 0) ∧
  (∃ x : ℝ, x ≤ 1 ∧ log_half (x + 2) < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1842_184225


namespace NUMINAMATH_CALUDE_arccos_negative_one_equals_pi_l1842_184240

theorem arccos_negative_one_equals_pi : Real.arccos (-1) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_negative_one_equals_pi_l1842_184240


namespace NUMINAMATH_CALUDE_product_and_reciprocal_sum_l1842_184210

theorem product_and_reciprocal_sum (x y : ℝ) : 
  x > 0 → y > 0 → x * y = 12 → (1 / x) = 5 * (1 / y) → x + y = (6 * Real.sqrt 60) / 5 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_sum_l1842_184210


namespace NUMINAMATH_CALUDE_f_max_min_range_l1842_184247

/-- A cubic function with parameter a -/
def f (a x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f_derivative (a x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

/-- Theorem stating the range of a for which f has both maximum and minimum values -/
theorem f_max_min_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
    (∀ z : ℝ, f a z ≤ f a x) ∧ 
    (∀ z : ℝ, f a z ≥ f a y)) ↔ 
  a < -3 ∨ a > 6 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_range_l1842_184247


namespace NUMINAMATH_CALUDE_g_composition_l1842_184278

def g (x : ℝ) : ℝ := 3 * x + 2

theorem g_composition : g (g (g 3)) = 107 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_l1842_184278


namespace NUMINAMATH_CALUDE_hundredth_term_difference_l1842_184208

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  terms : ℕ
  min_value : ℝ
  max_value : ℝ
  sum : ℝ

/-- The properties of our specific arithmetic sequence -/
def our_sequence : ArithmeticSequence where
  terms := 350
  min_value := 5
  max_value := 150
  sum := 38500

/-- The 100th term of an arithmetic sequence -/
def hundredth_term (a d : ℝ) : ℝ := a + 99 * d

/-- Theorem stating the difference between max and min possible 100th terms -/
theorem hundredth_term_difference (seq : ArithmeticSequence) 
  (h_seq : seq = our_sequence) : 
  ∃ (L G : ℝ), 
    (∀ (a d : ℝ), 
      (seq.min_value ≤ a) ∧ 
      (a + (seq.terms - 1) * d ≤ seq.max_value) ∧
      (seq.sum = (seq.terms : ℝ) * (2 * a + (seq.terms - 1) * d) / 2) →
      (L ≤ hundredth_term a d ∧ hundredth_term a d ≤ G)) ∧
    (G - L = 60.225) := by
  sorry

end NUMINAMATH_CALUDE_hundredth_term_difference_l1842_184208


namespace NUMINAMATH_CALUDE_books_together_l1842_184211

/-- The number of books Sandy, Tim, and Benny have together after Benny lost some books. -/
def remaining_books (sandy_books tim_books lost_books : ℕ) : ℕ :=
  sandy_books + tim_books - lost_books

/-- Theorem stating the number of books Sandy, Tim, and Benny have together. -/
theorem books_together : remaining_books 10 33 24 = 19 := by
  sorry

end NUMINAMATH_CALUDE_books_together_l1842_184211


namespace NUMINAMATH_CALUDE_distance_minimized_at_eight_sevenths_l1842_184268

/-- Given two points A and B in 3D space, prove that their distance is minimized when x = 8/7 -/
theorem distance_minimized_at_eight_sevenths (x : ℝ) :
  let A := (x, 5 - x, 2*x - 1)
  let B := (1, x + 2, 2 - x)
  let distance := Real.sqrt ((x - 1)^2 + (x + 2 - (5 - x))^2 + (2 - x - (2*x - 1))^2)
  (∀ y : ℝ, distance ≤ Real.sqrt ((y - 1)^2 + (y + 2 - (5 - y))^2 + (2 - y - (2*y - 1))^2)) ↔
  x = 8/7 := by
sorry


end NUMINAMATH_CALUDE_distance_minimized_at_eight_sevenths_l1842_184268


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1842_184238

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 20 + 3*x + 15 + (3*x + 6)) / 5 = 30 → x = 99 / 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1842_184238


namespace NUMINAMATH_CALUDE_video_length_correct_l1842_184295

/-- The length of each video in minutes -/
def video_length : ℝ := 7

/-- The number of videos watched per day -/
def videos_per_day : ℝ := 2

/-- The time spent watching ads in minutes -/
def ad_time : ℝ := 3

/-- The total time spent on Youtube in minutes -/
def total_time : ℝ := 17

/-- Theorem stating that the video length is correct given the conditions -/
theorem video_length_correct :
  videos_per_day * video_length + ad_time = total_time :=
by sorry

end NUMINAMATH_CALUDE_video_length_correct_l1842_184295


namespace NUMINAMATH_CALUDE_additional_hamburgers_l1842_184215

theorem additional_hamburgers (initial : ℕ) (total : ℕ) (h1 : initial = 9) (h2 : total = 12) :
  total - initial = 3 := by
  sorry

end NUMINAMATH_CALUDE_additional_hamburgers_l1842_184215


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l1842_184222

theorem like_terms_exponent_sum (m n : ℕ) : 
  (∃ (x y : ℝ), 2 * x^(3*n) * y^(m+4) = -3 * x^9 * y^(2*n)) → m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l1842_184222


namespace NUMINAMATH_CALUDE_incorrect_expression_l1842_184284

theorem incorrect_expression (x y : ℝ) (h : x / y = 5 / 6) : 
  ¬((2 * x - y) / y = 4 / 3) := by
sorry

end NUMINAMATH_CALUDE_incorrect_expression_l1842_184284


namespace NUMINAMATH_CALUDE_translated_line_proof_l1842_184256

/-- Given a line y = 2x + 5 translated down by m units (m > 0) -/
def translated_line (x : ℝ) (m : ℝ) : ℝ := 2 * x + 5 - m

theorem translated_line_proof (m : ℝ) (h_m : m > 0) :
  (translated_line (-2) m = -6 → m = 7) ∧
  (∀ x : ℝ, translated_line x 7 < 0 ↔ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_translated_line_proof_l1842_184256


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1842_184250

theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) :
  (1 / m + 1 / n) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1842_184250


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l1842_184282

/-- Given a price reduction scenario, prove that the first reduction percentage is 25% -/
theorem price_reduction_percentage (P : ℝ) (x : ℝ) : 
  P * (1 - x / 100) * (1 - 60 / 100) = P * (1 - 70 / 100) → x = 25 := by
  sorry

#check price_reduction_percentage

end NUMINAMATH_CALUDE_price_reduction_percentage_l1842_184282


namespace NUMINAMATH_CALUDE_wendys_score_l1842_184246

/-- The score for each treasure found in the game. -/
def points_per_treasure : ℕ := 5

/-- The number of treasures Wendy found on the first level. -/
def treasures_level1 : ℕ := 4

/-- The number of treasures Wendy found on the second level. -/
def treasures_level2 : ℕ := 3

/-- Wendy's total score in the game. -/
def total_score : ℕ := points_per_treasure * (treasures_level1 + treasures_level2)

/-- Theorem stating that Wendy's total score is 35 points. -/
theorem wendys_score : total_score = 35 := by
  sorry

end NUMINAMATH_CALUDE_wendys_score_l1842_184246


namespace NUMINAMATH_CALUDE_average_balance_is_200_l1842_184276

/-- Represents the balance of a savings account for a given month -/
structure MonthlyBalance where
  month : String
  balance : ℕ

/-- Calculates the average monthly balance given a list of monthly balances -/
def averageMonthlyBalance (balances : List MonthlyBalance) : ℚ :=
  (balances.map (·.balance)).sum / balances.length

/-- Theorem stating that the average monthly balance is $200 -/
theorem average_balance_is_200 (balances : List MonthlyBalance) 
  (h1 : balances = [
    { month := "January", balance := 200 },
    { month := "February", balance := 300 },
    { month := "March", balance := 100 },
    { month := "April", balance := 250 },
    { month := "May", balance := 150 }
  ]) : 
  averageMonthlyBalance balances = 200 := by
  sorry


end NUMINAMATH_CALUDE_average_balance_is_200_l1842_184276


namespace NUMINAMATH_CALUDE_right_triangle_area_l1842_184216

/-- The area of a right triangle with hypotenuse 15 and one angle 45° --/
theorem right_triangle_area (h : ℝ) (α : ℝ) (area : ℝ) 
  (hyp : h = 15)
  (angle : α = 45 * Real.pi / 180)
  (right_angle : α + α + Real.pi / 2 = Real.pi) : 
  area = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1842_184216


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l1842_184214

/-- A line y = 2x + a is tangent to the circle x^2 + y^2 = 9 if and only if a = ±3√5 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, y = 2*x + a ∧ x^2 + y^2 = 9 → (∀ ε > 0, ∃ δ > 0, ∀ x' y', 
    x'^2 + y'^2 = 9 → (x' - x)^2 + (y' - y)^2 < δ^2 → y' ≠ 2*x' + a)) ↔ 
  a = 3 * Real.sqrt 5 ∨ a = -3 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l1842_184214


namespace NUMINAMATH_CALUDE_base_8_digit_product_l1842_184286

def base_10_num : ℕ := 7890

def to_base_8 (n : ℕ) : List ℕ :=
  sorry

def digit_product (digits : List ℕ) : ℕ :=
  sorry

theorem base_8_digit_product :
  digit_product (to_base_8 base_10_num) = 84 :=
sorry

end NUMINAMATH_CALUDE_base_8_digit_product_l1842_184286


namespace NUMINAMATH_CALUDE_b_over_c_value_l1842_184245

theorem b_over_c_value (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : a * b * c / (d * e * f) = 0.1875)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) :
  b / c = 3 := by
sorry

end NUMINAMATH_CALUDE_b_over_c_value_l1842_184245


namespace NUMINAMATH_CALUDE_function_properties_l1842_184251

noncomputable def f (a b x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 1

def f_derivative_symmetric (a b : ℝ) : Prop :=
  ∀ x : ℝ, (6 * x^2 + 2 * a * x + b) = (6 * (-x - 1)^2 + 2 * a * (-x - 1) + b)

theorem function_properties (a b : ℝ) 
  (h1 : f_derivative_symmetric a b)
  (h2 : 6 + 2 * a + b = 0) :
  (a = 3 ∧ b = -12) ∧
  (∀ x : ℝ, f a b x ≤ f a b (-2)) ∧
  (∀ x : ℝ, f a b x ≥ f a b 1) ∧
  (f a b (-2) = 21) ∧
  (f a b 1 = -6) := by sorry

end NUMINAMATH_CALUDE_function_properties_l1842_184251


namespace NUMINAMATH_CALUDE_base_conversion_sum_l1842_184272

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

def C : Nat := 12
def D : Nat := 13

theorem base_conversion_sum :
  let base_8_num := base_to_decimal [5, 3, 7] 8
  let base_14_num := base_to_decimal [5, C, D] 14
  base_8_num + base_14_num = 1512 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l1842_184272


namespace NUMINAMATH_CALUDE_gcd_of_30_and_45_l1842_184227

theorem gcd_of_30_and_45 : Nat.gcd 30 45 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_30_and_45_l1842_184227


namespace NUMINAMATH_CALUDE_eighteenth_over_fortyfirst_415th_digit_l1842_184281

def decimal_expansion (n d : ℕ) : List ℕ := sorry

def nth_digit (n : ℕ) (expansion : List ℕ) : ℕ := sorry

theorem eighteenth_over_fortyfirst_415th_digit :
  let expansion := decimal_expansion 18 41
  nth_digit 415 expansion = 3 := by sorry

end NUMINAMATH_CALUDE_eighteenth_over_fortyfirst_415th_digit_l1842_184281


namespace NUMINAMATH_CALUDE_sqrt_fourth_root_approx_l1842_184235

theorem sqrt_fourth_root_approx : 
  ∃ (x : ℝ), x^2 = (0.000625)^(1/4) ∧ |x - 0.4| < 0.05 := by sorry

end NUMINAMATH_CALUDE_sqrt_fourth_root_approx_l1842_184235


namespace NUMINAMATH_CALUDE_min_value_equals_gcd_l1842_184228

theorem min_value_equals_gcd (a b c : ℕ+) :
  (∃ (x y z : ℤ), ∀ (x' y' z' : ℤ), a * x + b * y + c * z ≤ a * x' + b * y' + c * z' ∧ 0 < a * x + b * y + c * z) →
  (∃ (x y z : ℤ), a * x + b * y + c * z = Nat.gcd a.val (Nat.gcd b.val c.val)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_equals_gcd_l1842_184228


namespace NUMINAMATH_CALUDE_candy_calculation_correct_l1842_184265

/-- Calculates the number of candy pieces Haley's sister gave her. -/
def candy_from_sister (initial : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial - eaten)

/-- Proves that the calculation of candy pieces from Haley's sister is correct. -/
theorem candy_calculation_correct (initial eaten final : ℕ) 
  (h1 : initial ≥ eaten) 
  (h2 : final ≥ initial - eaten) : 
  candy_from_sister initial eaten final = final - (initial - eaten) :=
by sorry

end NUMINAMATH_CALUDE_candy_calculation_correct_l1842_184265


namespace NUMINAMATH_CALUDE_cos_squared_half_angle_minus_pi_fourth_l1842_184275

theorem cos_squared_half_angle_minus_pi_fourth (α : Real) 
  (h : Real.sin α = 2/3) : 
  Real.cos (α/2 - π/4)^2 = 1/6 := by sorry

end NUMINAMATH_CALUDE_cos_squared_half_angle_minus_pi_fourth_l1842_184275


namespace NUMINAMATH_CALUDE_pets_lost_l1842_184292

/-- Proves the number of pets Anthony lost when he forgot to lock the door -/
theorem pets_lost (initial_pets : ℕ) (final_pets : ℕ) : 
  initial_pets = 16 → 
  final_pets = 8 → 
  (initial_pets - (initial_pets - (initial_pets - final_pets) * 4 / 5)) = final_pets →
  initial_pets - (initial_pets - final_pets) * 5 / 4 = 6 :=
by
  sorry

#check pets_lost

end NUMINAMATH_CALUDE_pets_lost_l1842_184292


namespace NUMINAMATH_CALUDE_clock_rotation_impossibility_l1842_184243

/-- Represents a clock face with 12 numbers -/
def ClockFace : Type := Fin 12

/-- The sum of all numbers on the clock face -/
def clockSum : ℕ := (List.range 12).sum + 12

/-- The target number to be achieved on all positions of the blackboard -/
def target : ℕ := 1984

/-- The number of positions on the clock face and blackboard -/
def numPositions : ℕ := 12

theorem clock_rotation_impossibility : 
  ¬ ∃ (n : ℕ), n * clockSum = numPositions * target := by
  sorry

end NUMINAMATH_CALUDE_clock_rotation_impossibility_l1842_184243


namespace NUMINAMATH_CALUDE_pair_five_cows_four_pigs_seven_horses_l1842_184202

/-- The number of ways to pair animals of different species -/
def pairAnimals (cows pigs horses : ℕ) : ℕ :=
  cows * pigs * (cows + pigs - 2).factorial

/-- Theorem stating the number of ways to pair 5 cows, 4 pigs, and 7 horses -/
theorem pair_five_cows_four_pigs_seven_horses :
  pairAnimals 5 4 7 = 100800 := by
  sorry

#eval pairAnimals 5 4 7

end NUMINAMATH_CALUDE_pair_five_cows_four_pigs_seven_horses_l1842_184202


namespace NUMINAMATH_CALUDE_product_scaling_l1842_184273

theorem product_scaling (a b c : ℝ) (h : (268 : ℝ) * 74 = 19832) :
  2.68 * 0.74 = 1.9832 := by
  sorry

end NUMINAMATH_CALUDE_product_scaling_l1842_184273


namespace NUMINAMATH_CALUDE_negation_equivalence_l1842_184207

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∃ x₀ : ℝ, x₀ > 0 ∧ (x₀ + 1) * Real.exp x₀ ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1842_184207


namespace NUMINAMATH_CALUDE_initial_oil_fraction_l1842_184274

/-- Proves that the initial fraction of oil in the cylinder was 3/4 -/
theorem initial_oil_fraction (total_capacity : ℕ) (added_bottles : ℕ) (final_fraction : ℚ) :
  total_capacity = 80 →
  added_bottles = 4 →
  final_fraction = 4/5 →
  (total_capacity : ℚ) * final_fraction - added_bottles = (3/4 : ℚ) * total_capacity := by
  sorry

end NUMINAMATH_CALUDE_initial_oil_fraction_l1842_184274


namespace NUMINAMATH_CALUDE_carolyn_final_marbles_l1842_184277

/-- Represents the number of marbles Carolyn has after sharing -/
def marbles_after_sharing (initial_marbles shared_marbles : ℕ) : ℕ :=
  initial_marbles - shared_marbles

/-- Theorem stating that Carolyn ends up with 5 marbles -/
theorem carolyn_final_marbles :
  marbles_after_sharing 47 42 = 5 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_final_marbles_l1842_184277


namespace NUMINAMATH_CALUDE_peanut_butter_jar_size_l1842_184204

theorem peanut_butter_jar_size (total_ounces : ℕ) (jar_size_1 jar_size_3 : ℕ) (total_jars : ℕ) :
  total_ounces = 252 →
  jar_size_1 = 16 →
  jar_size_3 = 40 →
  total_jars = 9 →
  ∃ (jar_size_2 : ℕ),
    jar_size_2 = 28 ∧
    total_ounces = (total_jars / 3) * (jar_size_1 + jar_size_2 + jar_size_3) :=
by sorry

end NUMINAMATH_CALUDE_peanut_butter_jar_size_l1842_184204


namespace NUMINAMATH_CALUDE_sports_and_literature_enthusiasts_l1842_184212

theorem sports_and_literature_enthusiasts
  (total_students : ℕ)
  (sports_enthusiasts : ℕ)
  (literature_enthusiasts : ℕ)
  (h_total : total_students = 100)
  (h_sports : sports_enthusiasts = 60)
  (h_literature : literature_enthusiasts = 65) :
  ∃ (m n : ℕ),
    m = max sports_enthusiasts literature_enthusiasts ∧
    n = max 0 (sports_enthusiasts + literature_enthusiasts - total_students) ∧
    m + n = 85 :=
by sorry

end NUMINAMATH_CALUDE_sports_and_literature_enthusiasts_l1842_184212


namespace NUMINAMATH_CALUDE_orange_pear_weight_equivalence_l1842_184267

/-- Given that 7 oranges weigh the same as 5 pears, 
    prove that 49 oranges weigh the same as 35 pears. -/
theorem orange_pear_weight_equivalence :
  ∀ (orange_weight pear_weight : ℝ),
  orange_weight > 0 → pear_weight > 0 →
  7 * orange_weight = 5 * pear_weight →
  49 * orange_weight = 35 * pear_weight :=
by
  sorry

#check orange_pear_weight_equivalence

end NUMINAMATH_CALUDE_orange_pear_weight_equivalence_l1842_184267


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1842_184258

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z := (3 + i) / (1 - i)
  Complex.im z = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1842_184258


namespace NUMINAMATH_CALUDE_successive_integers_product_l1842_184231

theorem successive_integers_product (n : ℕ) : 
  n * (n + 1) = 7832 → n = 88 := by sorry

end NUMINAMATH_CALUDE_successive_integers_product_l1842_184231


namespace NUMINAMATH_CALUDE_inscribed_right_triangle_exists_l1842_184260

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the condition for a point to be inside a circle
def inside_circle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 < c.radius^2

-- Define the condition for a point to be on the circumference of a circle
def on_circumference (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define a right triangle
structure RightTriangle where
  A : Point
  B : Point
  C : Point
  right_angle : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

-- Theorem statement
theorem inscribed_right_triangle_exists (c : Circle) (A B : Point)
  (h_A : inside_circle A c) (h_B : inside_circle B c) :
  ∃ (C : Point), on_circumference C c ∧
    ∃ (t : RightTriangle), t.A = A ∧ t.B = B ∧ t.C = C :=
sorry

end NUMINAMATH_CALUDE_inscribed_right_triangle_exists_l1842_184260


namespace NUMINAMATH_CALUDE_binomial_cube_seven_l1842_184266

theorem binomial_cube_seven : 7^3 + 3*(7^2) + 3*7 + 1 = 512 := by
  sorry

end NUMINAMATH_CALUDE_binomial_cube_seven_l1842_184266


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1842_184285

-- Define the vectors
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (0, 1)

-- Define the theorem
theorem min_distance_to_line 
  (m n : ℝ) 
  (h : (a.1 - m) * (-m) + (a.2 - n) * (b.2 - n) = 0) : 
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧ 
  ∀ (x y : ℝ), x + y + 1 = 0 → 
  Real.sqrt ((x - m)^2 + (y - n)^2) ≥ d :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1842_184285


namespace NUMINAMATH_CALUDE_no_real_roots_l1842_184289

theorem no_real_roots (a b c d : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + a*x + b ≠ 0)
  (h2 : ∀ x : ℝ, x^2 + c*x + d ≠ 0) :
  ∀ x : ℝ, x^2 + ((a+c)/2)*x + ((b+d)/2) ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_l1842_184289


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l1842_184237

theorem largest_angle_in_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 7/5 of a right angle
  a + b = 7 / 5 * 90 →
  -- One angle is 40° larger than the other
  b = a + 40 →
  -- All angles are non-negative
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c →
  -- Sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 83°
  max a (max b c) = 83 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l1842_184237


namespace NUMINAMATH_CALUDE_value_of_a_l1842_184232

theorem value_of_a (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
  (hab : a * b = 2)
  (hbc : b * c = 3)
  (hcd : c * d = 4)
  (hde : d * e = 15)
  (hea : e * a = 10) :
  a = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1842_184232


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l1842_184259

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y + 1) = x + y + 1

/-- The theorem stating that there exists exactly one function satisfying the equation -/
theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, SatisfiesEquation f :=
sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l1842_184259


namespace NUMINAMATH_CALUDE_hundredth_digit_of_seven_twenty_sixths_l1842_184264

theorem hundredth_digit_of_seven_twenty_sixths (n : ℕ) : n = 100 → 
  (7 : ℚ) / 26 * 10^n % 10 = 9 := by sorry

end NUMINAMATH_CALUDE_hundredth_digit_of_seven_twenty_sixths_l1842_184264


namespace NUMINAMATH_CALUDE_sign_sum_theorem_l1842_184229

theorem sign_sum_theorem (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (x : ℤ), x ∈ ({5, 3, 2, 0, -3} : Set ℤ) ∧
  (a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d| = x) := by
  sorry

end NUMINAMATH_CALUDE_sign_sum_theorem_l1842_184229


namespace NUMINAMATH_CALUDE_tuesday_sales_l1842_184298

/-- Proves the number of bottles sold on Tuesday given inventory and sales information --/
theorem tuesday_sales (initial_inventory : ℕ) (monday_sales : ℕ) (daily_sales : ℕ) 
  (saturday_delivery : ℕ) (final_inventory : ℕ) : 
  initial_inventory = 4500 →
  monday_sales = 2445 →
  daily_sales = 50 →
  saturday_delivery = 650 →
  final_inventory = 1555 →
  initial_inventory + saturday_delivery - monday_sales - (daily_sales * 5) - final_inventory = 900 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_sales_l1842_184298


namespace NUMINAMATH_CALUDE_expression_factorization_l1842_184283

theorem expression_factorization (x : ℝ) : 
  (16 * x^6 + 49 * x^4 - 9) - (4 * x^6 - 14 * x^4 - 9) = 3 * x^4 * (4 * x^2 + 21) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1842_184283


namespace NUMINAMATH_CALUDE_voter_distribution_l1842_184230

theorem voter_distribution (total_voters : ℝ) (dem_percent : ℝ) (rep_percent : ℝ) 
  (rep_vote_a : ℝ) (total_vote_a : ℝ) (dem_vote_a : ℝ) :
  dem_percent = 0.6 →
  rep_percent = 1 - dem_percent →
  rep_vote_a = 0.2 →
  total_vote_a = 0.5 →
  dem_vote_a * dem_percent + rep_vote_a * rep_percent = total_vote_a →
  dem_vote_a = 0.7 := by
sorry

end NUMINAMATH_CALUDE_voter_distribution_l1842_184230


namespace NUMINAMATH_CALUDE_second_to_first_layer_ratio_l1842_184262

/-- Given a three-layer cake recipe, this theorem proves the ratio of the second layer to the first layer. -/
theorem second_to_first_layer_ratio 
  (sugar_first_layer : ℝ) 
  (sugar_third_layer : ℝ) 
  (third_to_second_ratio : ℝ) 
  (h1 : sugar_first_layer = 2)
  (h2 : sugar_third_layer = 12)
  (h3 : third_to_second_ratio = 3) :
  (sugar_third_layer / sugar_first_layer) / third_to_second_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_to_first_layer_ratio_l1842_184262


namespace NUMINAMATH_CALUDE_apple_count_l1842_184294

/-- The number of apples initially in the basket -/
def initial_apples : ℕ := sorry

/-- The number of oranges initially in the basket -/
def initial_oranges : ℕ := 5

/-- The number of oranges added to the basket -/
def added_oranges : ℕ := 5

/-- The total number of fruits in the basket after adding oranges -/
def total_fruits : ℕ := initial_apples + initial_oranges + added_oranges

theorem apple_count : initial_apples = 10 :=
  by
    have h1 : initial_oranges = 5 := rfl
    have h2 : added_oranges = 5 := rfl
    have h3 : 2 * initial_apples = total_fruits := sorry
    sorry

end NUMINAMATH_CALUDE_apple_count_l1842_184294


namespace NUMINAMATH_CALUDE_unique_special_number_l1842_184263

/-- A three-digit number is represented by its digits a, b, and c -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h1 : a > 0
  h2 : a ≤ 9
  h3 : b ≤ 9
  h4 : c ≤ 9

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of digits of a three-digit number -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.a + n.b + n.c

/-- A three-digit number is special if it equals 11 times the sum of its digits -/
def isSpecial (n : ThreeDigitNumber) : Prop :=
  value n = 11 * digitSum n

theorem unique_special_number :
  ∃! n : ThreeDigitNumber, isSpecial n ∧ value n = 198 :=
sorry

end NUMINAMATH_CALUDE_unique_special_number_l1842_184263


namespace NUMINAMATH_CALUDE_fruit_drink_volume_l1842_184220

/-- Represents the composition of a fruit drink -/
structure FruitDrink where
  orange : ℝ
  watermelon : ℝ
  grape : ℝ
  apple : ℝ
  pineapple : ℝ

/-- Theorem stating the total volume of the fruit drink -/
theorem fruit_drink_volume (drink : FruitDrink)
  (h1 : drink.orange = 0.1)
  (h2 : drink.watermelon = 0.4)
  (h3 : drink.grape = 0.2)
  (h4 : drink.apple = 0.15)
  (h5 : drink.pineapple = 0.15)
  (h6 : drink.orange + drink.watermelon + drink.grape + drink.apple + drink.pineapple = 1)
  (h7 : 24 / drink.grape = 36 / drink.apple) :
  24 / drink.grape = 240 := by
  sorry

end NUMINAMATH_CALUDE_fruit_drink_volume_l1842_184220


namespace NUMINAMATH_CALUDE_unique_prime_cube_l1842_184254

theorem unique_prime_cube (p : ℕ) : 
  Prime p ∧ ∃ (a : ℕ), a > 0 ∧ 16 * p + 1 = a^3 ↔ p = 307 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_cube_l1842_184254


namespace NUMINAMATH_CALUDE_min_attempts_to_guarantee_two_charged_l1842_184203

/-- Represents a set of batteries -/
def Battery := Fin 8

/-- Represents a pair of batteries -/
def BatteryPair := (Battery × Battery)

/-- The set of all possible battery pairs -/
def allPairs : Finset BatteryPair := sorry

/-- The set of charged batteries -/
def chargedBatteries : Finset Battery := sorry

/-- A function that determines if a set of battery pairs guarantees finding two charged batteries -/
def guaranteesTwoCharged (pairs : Finset BatteryPair) : Prop := sorry

/-- The minimum number of attempts required -/
def minAttempts : ℕ := sorry

theorem min_attempts_to_guarantee_two_charged :
  (minAttempts = 12) ∧
  (∃ (pairs : Finset BatteryPair), pairs.card = minAttempts ∧ guaranteesTwoCharged pairs) ∧
  (∀ (pairs : Finset BatteryPair), pairs.card < minAttempts → ¬guaranteesTwoCharged pairs) := by
  sorry

end NUMINAMATH_CALUDE_min_attempts_to_guarantee_two_charged_l1842_184203


namespace NUMINAMATH_CALUDE_line_slope_is_two_l1842_184252

/-- Given a line ax + 3my + 2a = 0 with m ≠ 0 and the sum of its intercepts on the coordinate axes is 2, prove that its slope is 2 -/
theorem line_slope_is_two (m a : ℝ) (hm : m ≠ 0) :
  (∃ (x y : ℝ), a * x + 3 * m * y + 2 * a = 0 ∧ 
   (a * 0 + 3 * m * y + 2 * a = 0 → y = -2 * a / (3 * m)) ∧
   (a * x + 3 * m * 0 + 2 * a = 0 → x = -2) ∧
   y + x = 2) →
  (∃ (k b : ℝ), ∀ x y, a * x + 3 * m * y + 2 * a = 0 ↔ y = k * x + b) ∧
  k = 2 :=
sorry

end NUMINAMATH_CALUDE_line_slope_is_two_l1842_184252


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l1842_184257

/-- The value of m^2 for which the line y = mx + 2 is tangent to the ellipse x^2 + 9y^2 = 9 -/
theorem line_tangent_to_ellipse :
  ∃ (m : ℝ),
    (∀ (x y : ℝ), y = m * x + 2 → x^2 + 9 * y^2 = 9) →
    (∃! (x y : ℝ), y = m * x + 2 ∧ x^2 + 9 * y^2 = 9) →
    m^2 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l1842_184257


namespace NUMINAMATH_CALUDE_strictly_decreasing_exponential_range_l1842_184271

theorem strictly_decreasing_exponential_range (a : ℝ) :
  (∀ x y : ℝ, x < y → (2*a - 1)^x > (2*a - 1)^y) → a ∈ Set.Ioo (1/2) 1 :=
by sorry

end NUMINAMATH_CALUDE_strictly_decreasing_exponential_range_l1842_184271


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1842_184206

theorem repeating_decimal_sum (a b : ℕ) : 
  (5 : ℚ) / 13 = (a * 10 + b : ℚ) / 99 → a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1842_184206


namespace NUMINAMATH_CALUDE_females_over30_prefer_l1842_184224

/-- Represents the survey data from WebStream --/
structure WebStreamSurvey where
  total_surveyed : ℕ
  total_prefer : ℕ
  males_prefer : ℕ
  females_under30_not_prefer : ℕ
  females_over30_not_prefer : ℕ

/-- Theorem stating the number of females over 30 who prefer WebStream --/
theorem females_over30_prefer (survey : WebStreamSurvey)
  (h1 : survey.total_surveyed = 420)
  (h2 : survey.total_prefer = 200)
  (h3 : survey.males_prefer = 80)
  (h4 : survey.females_under30_not_prefer = 90)
  (h5 : survey.females_over30_not_prefer = 70) :
  ∃ (females_over30_prefer : ℕ), females_over30_prefer = 110 := by
  sorry


end NUMINAMATH_CALUDE_females_over30_prefer_l1842_184224


namespace NUMINAMATH_CALUDE_pizza_price_l1842_184244

theorem pizza_price (num_pizzas : ℕ) (tip : ℝ) (bill : ℝ) (change : ℝ) :
  num_pizzas = 4 ∧ tip = 5 ∧ bill = 50 ∧ change = 5 →
  ∃ (price : ℝ), price = 10 ∧ num_pizzas * price + tip = bill - change :=
by sorry

end NUMINAMATH_CALUDE_pizza_price_l1842_184244


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l1842_184219

theorem smallest_upper_bound (a b : ℤ) (h1 : a > 6) (h2 : ∀ (x y : ℤ), x > 6 → x - y ≥ 4) : 
  ∃ N : ℤ, (a + b < N) ∧ (∀ M : ℤ, M < N → ¬(a + b < M)) :=
sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_l1842_184219


namespace NUMINAMATH_CALUDE_pencils_per_box_correct_l1842_184201

/-- Represents the number of pencils in each box -/
def pencils_per_box : ℕ := 80

/-- Represents the number of boxes of pencils ordered -/
def boxes : ℕ := 15

/-- Represents the cost of a single pencil in dollars -/
def pencil_cost : ℕ := 4

/-- Represents the cost of a single pen in dollars -/
def pen_cost : ℕ := 5

/-- Represents the total cost of all stationery in dollars -/
def total_cost : ℕ := 18300

/-- Theorem stating that the number of pencils per box satisfies the given conditions -/
theorem pencils_per_box_correct : 
  let total_pencils := pencils_per_box * boxes
  let total_pens := 2 * total_pencils + 300
  total_pencils * pencil_cost + total_pens * pen_cost = total_cost := by
  sorry


end NUMINAMATH_CALUDE_pencils_per_box_correct_l1842_184201


namespace NUMINAMATH_CALUDE_train_crossing_time_l1842_184255

/-- The time taken for a train to cross a stationary point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 180 → 
  train_speed_kmh = 108 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1842_184255


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_graph_not_in_third_quadrant_l1842_184239

/-- A linear function f(x) = kx + b does not pass through the third quadrant
    if and only if k < 0 and b > 0 -/
theorem linear_function_not_in_third_quadrant (k b : ℝ) :
  k < 0 ∧ b > 0 → ∀ x y : ℝ, y = k * x + b → ¬(x < 0 ∧ y < 0) := by
  sorry

/-- The graph of y = -2x + 1 does not pass through the third quadrant -/
theorem graph_not_in_third_quadrant :
  ∀ x y : ℝ, y = -2 * x + 1 → ¬(x < 0 ∧ y < 0) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_graph_not_in_third_quadrant_l1842_184239


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1842_184297

/-- The equation represents a hyperbola if both coefficients are nonzero and have opposite signs -/
def is_hyperbola (k : ℝ) : Prop :=
  k - 3 > 0 ∧ k > 0

/-- k > 3 is a sufficient condition for the equation to represent a hyperbola -/
theorem sufficient_condition (k : ℝ) (h : k > 3) : is_hyperbola k :=
sorry

/-- k > 3 is not a necessary condition for the equation to represent a hyperbola -/
theorem not_necessary_condition : ∃ k : ℝ, is_hyperbola k ∧ ¬(k > 3) :=
sorry

/-- k > 3 is a sufficient but not necessary condition for the equation to represent a hyperbola -/
theorem sufficient_but_not_necessary (k : ℝ) : 
  (k > 3 → is_hyperbola k) ∧ ¬(is_hyperbola k → k > 3) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1842_184297


namespace NUMINAMATH_CALUDE_incircle_radius_inscribed_triangle_l1842_184217

theorem incircle_radius_inscribed_triangle (r : ℝ) (α β γ : ℝ) (h1 : 0 < r) (h2 : 0 < α) (h3 : 0 < β) (h4 : 0 < γ) 
  (h5 : α + β + γ = π) (h6 : Real.tan α = 1/3) (h7 : Real.sin β * Real.sin γ = 1/Real.sqrt 10) : 
  ∃ ρ : ℝ, ρ = (r * Real.sqrt 10) / (1 + Real.sqrt 2 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_incircle_radius_inscribed_triangle_l1842_184217


namespace NUMINAMATH_CALUDE_popsicle_stick_count_l1842_184223

/-- Represents the number of popsicle sticks in Gino's problem -/
structure PopsicleSticks where
  initial : ℕ
  given_away : ℕ
  left : ℕ

/-- Theorem stating that the initial number of popsicle sticks 
    is equal to the sum of those given away and those left -/
theorem popsicle_stick_count (p : PopsicleSticks) 
    (h1 : p.given_away = 50)
    (h2 : p.left = 13)
    : p.initial = p.given_away + p.left := by
  sorry

#check popsicle_stick_count

end NUMINAMATH_CALUDE_popsicle_stick_count_l1842_184223


namespace NUMINAMATH_CALUDE_tribe_assignment_l1842_184261

-- Define the two tribes
inductive Tribe
| Triussa
| La

-- Define a person as having a tribe
structure Person where
  tribe : Tribe

-- Define the three people
def person1 : Person := sorry
def person2 : Person := sorry
def person3 : Person := sorry

-- Define what it means for a statement to be true
def isTrueStatement (p : Person) (s : Prop) : Prop :=
  (p.tribe = Tribe.Triussa ∧ s) ∨ (p.tribe = Tribe.La ∧ ¬s)

-- Define the statements made by each person
def statement1 : Prop := 
  (person1.tribe = Tribe.Triussa ∧ person2.tribe = Tribe.La ∧ person3.tribe = Tribe.La) ∨
  (person1.tribe = Tribe.La ∧ person2.tribe = Tribe.Triussa ∧ person3.tribe = Tribe.La) ∨
  (person1.tribe = Tribe.La ∧ person2.tribe = Tribe.La ∧ person3.tribe = Tribe.Triussa)

def statement2 : Prop := person3.tribe = Tribe.La

def statement3 : Prop := person1.tribe = Tribe.La

-- Theorem to prove
theorem tribe_assignment :
  isTrueStatement person1 statement1 ∧
  isTrueStatement person2 statement2 ∧
  isTrueStatement person3 statement3 →
  person1.tribe = Tribe.La ∧ person2.tribe = Tribe.La ∧ person3.tribe = Tribe.Triussa :=
sorry

end NUMINAMATH_CALUDE_tribe_assignment_l1842_184261


namespace NUMINAMATH_CALUDE_expression_simplification_l1842_184270

theorem expression_simplification :
  let a := 3
  let b := 4
  let c := 5
  let d := 6
  (Real.sqrt (a + b + c + d) / 3) + ((a * b + 10) / 4) = 5.5 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1842_184270


namespace NUMINAMATH_CALUDE_additional_seashells_is_8_l1842_184249

/-- The number of additional seashells Carina puts in each week -/
def additional_seashells : ℕ := sorry

/-- The number of seashells in the jar this week -/
def initial_seashells : ℕ := 50

/-- The number of seashells in the jar after 4 weeks -/
def final_seashells : ℕ := 130

/-- The number of weeks -/
def weeks : ℕ := 4

/-- Formula for the total number of seashells after n weeks -/
def total_seashells (n : ℕ) : ℕ :=
  initial_seashells + n * additional_seashells + (n * (n - 1) / 2) * additional_seashells

/-- Theorem stating that the number of additional seashells per week is 8 -/
theorem additional_seashells_is_8 :
  additional_seashells = 8 ∧
  (∀ n : ℕ, n ≤ weeks → total_seashells n ≤ total_seashells (n + 1)) ∧
  total_seashells weeks = final_seashells :=
sorry

end NUMINAMATH_CALUDE_additional_seashells_is_8_l1842_184249


namespace NUMINAMATH_CALUDE_modulus_of_one_plus_i_l1842_184293

/-- The modulus of the complex number z = 1 + i is √2 -/
theorem modulus_of_one_plus_i : Complex.abs (1 + Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_one_plus_i_l1842_184293


namespace NUMINAMATH_CALUDE_l₂_slope_l1842_184234

-- Define the slope and y-intercept of line l₁
def m₁ : ℝ := 2
def b₁ : ℝ := 3

-- Define the equation of line l₁
def l₁ (x y : ℝ) : Prop := y = m₁ * x + b₁

-- Define the equation of the symmetry line
def symmetry_line (x y : ℝ) : Prop := y = -x

-- Define the symmetry relation between two points
def symmetric (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  symmetry_line ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)

-- Define line l₂ as symmetric to l₁ with respect to y = -x
def l₂ (x y : ℝ) : Prop :=
  ∃ (x₁ y₁ : ℝ), l₁ x₁ y₁ ∧ symmetric x₁ y₁ x y

-- State the theorem
theorem l₂_slope :
  ∃ (m₂ : ℝ), m₂ = 1/2 ∧ ∀ (x y : ℝ), l₂ x y → ∃ (b₂ : ℝ), y = m₂ * x + b₂ :=
sorry

end NUMINAMATH_CALUDE_l₂_slope_l1842_184234


namespace NUMINAMATH_CALUDE_polynomial_value_at_zero_l1842_184279

/-- A polynomial P(x) = x^2 + bx + c satisfying specific conditions -/
def P (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem polynomial_value_at_zero 
  (b c : ℝ) 
  (h1 : P b c (P b c 1) = 0)
  (h2 : P b c (P b c 2) = 0)
  (h3 : P b c 1 ≠ P b c 2) :
  P b c 0 = -3/2 :=
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_zero_l1842_184279


namespace NUMINAMATH_CALUDE_abc_inequality_and_fraction_sum_l1842_184218

theorem abc_inequality_and_fraction_sum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^2 + b^2 + c^2 = 9) : 
  a * b * c ≤ 3 * Real.sqrt 3 ∧ 
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b)) > (a + b + c) / 3 :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_and_fraction_sum_l1842_184218
