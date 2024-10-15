import Mathlib

namespace NUMINAMATH_CALUDE_vasyas_numbers_l509_50935

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x + y = x / y) (h3 : x * y = x / y) :
  x = 1/2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l509_50935


namespace NUMINAMATH_CALUDE_linda_savings_l509_50956

theorem linda_savings (savings : ℝ) : 
  (5 / 6 : ℝ) * savings + 500 = savings → savings = 3000 := by
  sorry

end NUMINAMATH_CALUDE_linda_savings_l509_50956


namespace NUMINAMATH_CALUDE_variety_show_probability_variety_show_probability_proof_l509_50938

/-- The probability of selecting 2 dance performances out of 3 for the first 3 slots
    in a randomly arranged program of 8 performances (5 singing, 3 dance) -/
theorem variety_show_probability : ℚ :=
  let total_performances : ℕ := 8
  let singing_performances : ℕ := 5
  let dance_performances : ℕ := 3
  let first_slots : ℕ := 3
  let required_dance : ℕ := 2

  3 / 28

theorem variety_show_probability_proof :
  variety_show_probability = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_variety_show_probability_variety_show_probability_proof_l509_50938


namespace NUMINAMATH_CALUDE_tenth_fib_is_55_l509_50989

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- The 10th Fibonacci number is 55 -/
theorem tenth_fib_is_55 : fib 9 = 55 := by
  sorry

end NUMINAMATH_CALUDE_tenth_fib_is_55_l509_50989


namespace NUMINAMATH_CALUDE_discount_calculation_l509_50904

/-- Calculates the percentage discount given the original price and sale price -/
def percentage_discount (original_price sale_price : ℚ) : ℚ :=
  (original_price - sale_price) / original_price * 100

/-- Proves that the percentage discount for an item with original price $25 and sale price $18.75 is 25% -/
theorem discount_calculation :
  let original_price : ℚ := 25
  let sale_price : ℚ := 37/2  -- Representing 18.75 as a rational number
  percentage_discount original_price sale_price = 25 := by
  sorry


end NUMINAMATH_CALUDE_discount_calculation_l509_50904


namespace NUMINAMATH_CALUDE_machine_selling_price_l509_50994

/-- Calculates the selling price of a machine given its costs and desired profit percentage. -/
def selling_price (purchase_price repair_cost transport_cost profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transport_cost
  let profit := total_cost * profit_percentage / 100
  total_cost + profit

/-- Proves that the selling price of the machine is 25500 Rs given the specified costs and profit percentage. -/
theorem machine_selling_price :
  selling_price 11000 5000 1000 50 = 25500 := by
  sorry

#eval selling_price 11000 5000 1000 50

end NUMINAMATH_CALUDE_machine_selling_price_l509_50994


namespace NUMINAMATH_CALUDE_geometric_series_sum_l509_50980

/-- The sum of a geometric series with first term a, common ratio r, and n terms -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of our geometric series -/
def a : ℚ := 1/2

/-- The common ratio of our geometric series -/
def r : ℚ := 1/2

/-- The number of terms in our geometric series -/
def n : ℕ := 8

theorem geometric_series_sum :
  geometricSum a r n = 255/256 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l509_50980


namespace NUMINAMATH_CALUDE_ball_game_bill_l509_50998

theorem ball_game_bill (num_adults num_children : ℕ) 
  (adult_price child_price : ℚ) : 
  num_adults = 10 → 
  num_children = 11 → 
  adult_price = 8 → 
  child_price = 4 → 
  (num_adults : ℚ) * adult_price + (num_children : ℚ) * child_price = 124 := by
  sorry

end NUMINAMATH_CALUDE_ball_game_bill_l509_50998


namespace NUMINAMATH_CALUDE_right_triangle_sin_value_l509_50932

theorem right_triangle_sin_value (A B C : Real) (h1 : 0 < A) (h2 : A < π / 2) :
  (Real.cos B = 0) →
  (3 * Real.sin A = 4 * Real.cos A) →
  Real.sin A = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sin_value_l509_50932


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_smallest_solution_is_3_minus_sqrt_3_l509_50950

theorem smallest_solution_of_equation (x : ℝ) :
  (1 / (x - 2) + 1 / (x - 4) = 3 / (x - 3)) ↔ (x = 3 - Real.sqrt 3 ∨ x = 3 + Real.sqrt 3) :=
by sorry

theorem smallest_solution_is_3_minus_sqrt_3 :
  ∃ x : ℝ, (1 / (x - 2) + 1 / (x - 4) = 3 / (x - 3)) ∧
           (∀ y : ℝ, (1 / (y - 2) + 1 / (y - 4) = 3 / (y - 3)) → x ≤ y) ∧
           x = 3 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_smallest_solution_is_3_minus_sqrt_3_l509_50950


namespace NUMINAMATH_CALUDE_triangle_abc_theorem_l509_50954

theorem triangle_abc_theorem (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C) →
  (a + b = 6) →
  (1/2 * a * b * Real.sin C = 2 * Real.sqrt 3) →
  -- Conclusions
  (C = 2 * π / 3 ∧ c = 2 * Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_theorem_l509_50954


namespace NUMINAMATH_CALUDE_hired_waiters_count_l509_50982

/-- Represents the number of waiters hired to change the ratio of cooks to waiters -/
def waiters_hired (initial_ratio_cooks initial_ratio_waiters new_ratio_cooks new_ratio_waiters num_cooks : ℕ) : ℕ :=
  let initial_waiters := (num_cooks * initial_ratio_waiters) / initial_ratio_cooks
  let total_new_waiters := (num_cooks * new_ratio_waiters) / new_ratio_cooks
  total_new_waiters - initial_waiters

/-- Theorem stating that given the conditions, the number of waiters hired is 12 -/
theorem hired_waiters_count :
  waiters_hired 3 8 1 4 9 = 12 :=
by sorry

end NUMINAMATH_CALUDE_hired_waiters_count_l509_50982


namespace NUMINAMATH_CALUDE_solve_for_b_l509_50990

theorem solve_for_b (m a k c d b : ℝ) (h : m = (k * c * a * b) / (k * a - d)) :
  b = (m * k * a - m * d) / (k * c * a) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l509_50990


namespace NUMINAMATH_CALUDE_cost_of_soft_drink_l509_50975

/-- The cost of a can of soft drink given the following conditions:
  * 5 boxes of pizza cost $50
  * 6 hamburgers cost $18
  * 20 cans of soft drinks were bought
  * Total spent is $106
-/
theorem cost_of_soft_drink :
  let pizza_cost : ℚ := 50
  let hamburger_cost : ℚ := 18
  let total_cans : ℕ := 20
  let total_spent : ℚ := 106
  let soft_drink_cost : ℚ := (total_spent - pizza_cost - hamburger_cost) / total_cans
  soft_drink_cost = 19/10 := by sorry

end NUMINAMATH_CALUDE_cost_of_soft_drink_l509_50975


namespace NUMINAMATH_CALUDE_circular_arrangement_size_l509_50961

/-- Represents a circular arrangement of students and a teacher. -/
structure CircularArrangement where
  total_positions : ℕ
  teacher_position : ℕ

/-- Defines the property of two positions being opposite in the circle. -/
def is_opposite (c : CircularArrangement) (pos1 pos2 : ℕ) : Prop :=
  (pos2 - pos1) % c.total_positions = c.total_positions / 2

/-- The main theorem stating the total number of positions in the arrangement. -/
theorem circular_arrangement_size :
  ∀ (c : CircularArrangement),
    (is_opposite c 6 16) →
    (c.teacher_position ≤ c.total_positions) →
    (c.total_positions = 23) :=
by sorry

end NUMINAMATH_CALUDE_circular_arrangement_size_l509_50961


namespace NUMINAMATH_CALUDE_count_valid_lists_l509_50952

/-- A structure representing a list of five integers with the given properties -/
structure IntegerList :=
  (a b : ℕ+)
  (h1 : a < b)
  (h2 : 2 * a.val + 3 * b.val = 124)

/-- The number of valid integer lists -/
def validListCount : ℕ := sorry

/-- Theorem stating that there are exactly 8 valid integer lists -/
theorem count_valid_lists : validListCount = 8 := by sorry

end NUMINAMATH_CALUDE_count_valid_lists_l509_50952


namespace NUMINAMATH_CALUDE_power_sum_equality_l509_50936

theorem power_sum_equality : 2^567 + 9^5 / 3^2 = 2^567 + 6561 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l509_50936


namespace NUMINAMATH_CALUDE_qr_equals_b_l509_50919

def curve (c : ℝ) (x y : ℝ) : Prop := y / c = Real.cosh (x / c)

theorem qr_equals_b (a b c : ℝ) (h1 : curve c a b) (h2 : curve c 0 c) :
  let normal_slope := -1 / (Real.sinh (a / c) / c)
  let r_x := c * Real.sinh (a / c) / 2
  Real.sqrt ((r_x - 0)^2 + (0 - c)^2) = b := by sorry

end NUMINAMATH_CALUDE_qr_equals_b_l509_50919


namespace NUMINAMATH_CALUDE_monotonic_cos_plus_linear_monotonic_cos_plus_linear_converse_l509_50900

/-- A function f : ℝ → ℝ is monotonic if it is either monotonically increasing or monotonically decreasing -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

/-- The main theorem: if f(x) = cos x + ax is monotonic, then a ∈ (-∞, -1] ∪ [1, +∞) -/
theorem monotonic_cos_plus_linear (a : ℝ) :
  Monotonic (fun x => Real.cos x + a * x) → a ≤ -1 ∨ a ≥ 1 := by
  sorry

/-- The converse: if a ∈ (-∞, -1] ∪ [1, +∞), then f(x) = cos x + ax is monotonic -/
theorem monotonic_cos_plus_linear_converse (a : ℝ) :
  (a ≤ -1 ∨ a ≥ 1) → Monotonic (fun x => Real.cos x + a * x) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_cos_plus_linear_monotonic_cos_plus_linear_converse_l509_50900


namespace NUMINAMATH_CALUDE_simplify_sqrt_quadratic_l509_50937

theorem simplify_sqrt_quadratic (x : ℝ) (h : x < 2) : 
  Real.sqrt (x^2 - 4*x + 4) = 2 - x := by
sorry

end NUMINAMATH_CALUDE_simplify_sqrt_quadratic_l509_50937


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l509_50984

/-- A regular polygon with side length 2 and interior angles measuring 135° has a perimeter of 16. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (interior_angle : ℝ) :
  n ≥ 3 ∧
  side_length = 2 ∧
  interior_angle = 135 ∧
  (n : ℝ) * (180 - interior_angle) = 360 →
  n * side_length = 16 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l509_50984


namespace NUMINAMATH_CALUDE_percent_within_one_std_dev_l509_50901

/-- A symmetric distribution about a mean -/
structure SymmetricDistribution where
  mean : ℝ
  std_dev : ℝ
  is_symmetric : Bool
  percent_less_than_mean_plus_std : ℝ

/-- Theorem: In a symmetric distribution where 80% is less than mean + std_dev,
    60% lies within one standard deviation of the mean -/
theorem percent_within_one_std_dev
  (dist : SymmetricDistribution)
  (h_symmetric : dist.is_symmetric = true)
  (h_eighty_percent : dist.percent_less_than_mean_plus_std = 80) :
  ∃ (percent_within : ℝ), percent_within = 60 :=
sorry

end NUMINAMATH_CALUDE_percent_within_one_std_dev_l509_50901


namespace NUMINAMATH_CALUDE_tens_digit_of_difference_l509_50925

/-- Given a single digit t, prove that the tens digit of (6t5 - 5t6) is 9 -/
theorem tens_digit_of_difference (t : ℕ) (h : t < 10) : 
  (6 * 100 + t * 10 + 5) - (5 * 100 + t * 10 + 6) = 94 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_difference_l509_50925


namespace NUMINAMATH_CALUDE_collinear_points_m_value_l509_50973

/-- Given three points A, B, and C in 2D space, determines if they are collinear -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem stating that if A(1,2), B(3,m), and C(7,m+6) are collinear, then m = 5 -/
theorem collinear_points_m_value :
  ∀ m : ℝ, collinear (1, 2) (3, m) (7, m + 6) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_m_value_l509_50973


namespace NUMINAMATH_CALUDE_ball_drawing_probabilities_l509_50957

/-- The total number of balls -/
def total_balls : ℕ := 6

/-- The number of white balls -/
def white_balls : ℕ := 3

/-- The number of black balls -/
def black_balls : ℕ := 3

/-- The number of balls drawn -/
def drawn_balls : ℕ := 2

/-- The probability of drawing two balls of the same color -/
def prob_same_color : ℚ := 2/5

/-- The probability of drawing two balls of different colors -/
def prob_diff_color : ℚ := 3/5

theorem ball_drawing_probabilities :
  (prob_same_color + prob_diff_color = 1) ∧
  (prob_same_color = 2/5) ∧
  (prob_diff_color = 3/5) :=
by sorry

end NUMINAMATH_CALUDE_ball_drawing_probabilities_l509_50957


namespace NUMINAMATH_CALUDE_gravel_weight_l509_50955

/-- Proves that the weight of gravel in a cement mixture is 10 pounds given the specified conditions. -/
theorem gravel_weight (total_weight : ℝ) (sand_fraction : ℝ) (water_fraction : ℝ) :
  total_weight = 23.999999999999996 →
  sand_fraction = 1 / 3 →
  water_fraction = 1 / 4 →
  total_weight - (sand_fraction * total_weight + water_fraction * total_weight) = 10 := by
  sorry

end NUMINAMATH_CALUDE_gravel_weight_l509_50955


namespace NUMINAMATH_CALUDE_gift_price_gift_price_exact_l509_50976

/-- The price of Lisa's gift given her savings and contributions from family and friends --/
theorem gift_price (lisa_savings : ℚ) (mother_fraction : ℚ) (brother_multiplier : ℚ) 
  (friend_fraction : ℚ) (short_amount : ℚ) : ℚ :=
  let mother_contribution := mother_fraction * lisa_savings
  let brother_contribution := brother_multiplier * mother_contribution
  let friend_contribution := friend_fraction * (mother_contribution + brother_contribution)
  let total_contributions := lisa_savings + mother_contribution + brother_contribution + friend_contribution
  total_contributions + short_amount

/-- The price of Lisa's gift is $3935.71 --/
theorem gift_price_exact : 
  gift_price 1600 (3/8) (5/4) (2/7) 600 = 3935.71 := by
  sorry

end NUMINAMATH_CALUDE_gift_price_gift_price_exact_l509_50976


namespace NUMINAMATH_CALUDE_digit_sum_problem_l509_50987

theorem digit_sum_problem (p q r : ℕ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 →
  p < 10 → q < 10 → r < 10 →
  100 * p + 10 * q + r + 10 * q + r + r = 912 →
  q = 5 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l509_50987


namespace NUMINAMATH_CALUDE_speech_competition_score_l509_50918

/-- Calculates the weighted average score for a speech competition --/
def weighted_average (content_score delivery_score effectiveness_score : ℚ) : ℚ :=
  (4 * content_score + 4 * delivery_score + 2 * effectiveness_score) / 10

/-- Theorem: The weighted average score for a student with scores 91, 94, and 90 is 92 --/
theorem speech_competition_score : weighted_average 91 94 90 = 92 := by
  sorry

end NUMINAMATH_CALUDE_speech_competition_score_l509_50918


namespace NUMINAMATH_CALUDE_no_integer_solution_l509_50986

theorem no_integer_solution : ¬∃ (n : ℕ+), (20 * n + 2) ∣ (2003 * n + 2002) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l509_50986


namespace NUMINAMATH_CALUDE_min_value_of_f_in_interval_l509_50968

def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 2

theorem min_value_of_f_in_interval :
  ∃ (x : ℝ), x ∈ Set.Icc (-2) (-1) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2) (-1) → f y ≥ f x) ∧
  f x = -4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_in_interval_l509_50968


namespace NUMINAMATH_CALUDE_correct_transformation_l509_50988

theorem correct_transformation (x : ℝ) : (2/3 * x - 1 = x) ↔ (2*x - 3 = 3*x) := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l509_50988


namespace NUMINAMATH_CALUDE_unanswered_questions_l509_50966

/-- Represents the scoring for a math competition participant --/
structure Scoring where
  correct : ℕ      -- number of correct answers
  incorrect : ℕ    -- number of incorrect answers
  unanswered : ℕ   -- number of unanswered questions

/-- Calculates the score using the first method --/
def score_method1 (s : Scoring) : ℕ :=
  5 * s.correct + 2 * s.unanswered

/-- Calculates the score using the second method --/
def score_method2 (s : Scoring) : ℕ :=
  39 + 3 * s.correct - s.incorrect

/-- Theorem stating the possible number of unanswered questions --/
theorem unanswered_questions (s : Scoring) :
  score_method1 s = 71 ∧ score_method2 s = 71 ∧ 
  s.correct + s.incorrect + s.unanswered = s.correct + s.incorrect →
  s.unanswered = 8 ∨ s.unanswered = 3 := by
  sorry


end NUMINAMATH_CALUDE_unanswered_questions_l509_50966


namespace NUMINAMATH_CALUDE_round_trip_distance_l509_50960

/-- Calculates the total distance of a round trip given the times for each direction and the average speed -/
theorem round_trip_distance 
  (time_to : ℝ) 
  (time_from : ℝ) 
  (avg_speed : ℝ) 
  (h1 : time_to > 0) 
  (h2 : time_from > 0) 
  (h3 : avg_speed > 0) : 
  ∃ (distance : ℝ), distance = avg_speed * (time_to + time_from) / 60 := by
  sorry

#check round_trip_distance

end NUMINAMATH_CALUDE_round_trip_distance_l509_50960


namespace NUMINAMATH_CALUDE_certain_number_problem_l509_50985

theorem certain_number_problem : ∃ x : ℕ, 
  220020 = (x + 445) * (2 * (x - 445)) + 20 ∧ x = 555 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l509_50985


namespace NUMINAMATH_CALUDE_probability_within_four_rings_l509_50929

def P_first_ring : ℚ := 1 / 10
def P_second_ring : ℚ := 3 / 10
def P_third_ring : ℚ := 2 / 5
def P_fourth_ring : ℚ := 1 / 10

theorem probability_within_four_rings :
  P_first_ring + P_second_ring + P_third_ring + P_fourth_ring = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_within_four_rings_l509_50929


namespace NUMINAMATH_CALUDE_quadratic_discriminant_zero_not_harmonic_l509_50983

/-- The discriminant of the quadratic equation 3ax^2 + bx + 2c = 0 is zero -/
def discriminant_zero (a b c : ℝ) : Prop :=
  b^2 = 24*a*c

/-- a, b, and c form a harmonic progression -/
def harmonic_progression (a b c : ℝ) : Prop :=
  2/b = 1/a + 1/c

theorem quadratic_discriminant_zero_not_harmonic
  (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  discriminant_zero a b c → ¬harmonic_progression a b c :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_zero_not_harmonic_l509_50983


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l509_50917

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l509_50917


namespace NUMINAMATH_CALUDE_shift_sine_graph_l509_50974

theorem shift_sine_graph (x : ℝ) :
  let f (x : ℝ) := 2 * Real.sin (2 * x + π / 6)
  let period := 2 * π / 2
  let shift := period / 4
  let g (x : ℝ) := f (x - shift)
  g x = 2 * Real.sin (2 * x - π / 3) := by sorry

end NUMINAMATH_CALUDE_shift_sine_graph_l509_50974


namespace NUMINAMATH_CALUDE_luke_rounds_played_l509_50913

/-- The number of points Luke scored in total -/
def total_points : ℕ := 154

/-- The number of points Luke gained in each round -/
def points_per_round : ℕ := 11

/-- The number of rounds Luke played -/
def rounds_played : ℕ := total_points / points_per_round

theorem luke_rounds_played :
  rounds_played = 14 :=
by sorry

end NUMINAMATH_CALUDE_luke_rounds_played_l509_50913


namespace NUMINAMATH_CALUDE_rams_weight_increase_percentage_l509_50912

theorem rams_weight_increase_percentage
  (weight_ratio : ℚ) -- Ratio of Ram's weight to Shyam's weight
  (total_weight_after : ℝ) -- Total weight after increase
  (total_increase_percentage : ℝ) -- Total weight increase percentage
  (shyam_increase_percentage : ℝ) -- Shyam's weight increase percentage
  (h1 : weight_ratio = 4 / 5) -- Condition: weight ratio is 4:5
  (h2 : total_weight_after = 82.8) -- Condition: total weight after increase is 82.8 kg
  (h3 : total_increase_percentage = 15) -- Condition: total weight increase is 15%
  (h4 : shyam_increase_percentage = 19) -- Condition: Shyam's weight increased by 19%
  : ∃ (ram_increase_percentage : ℝ), ram_increase_percentage = 10 :=
by sorry

end NUMINAMATH_CALUDE_rams_weight_increase_percentage_l509_50912


namespace NUMINAMATH_CALUDE_water_consumption_ratio_l509_50959

theorem water_consumption_ratio (initial_volume : ℝ) (first_drink_fraction : ℝ) (final_volume : ℝ) :
  initial_volume = 4 →
  first_drink_fraction = 1/4 →
  final_volume = 1 →
  let remaining_after_first := initial_volume - first_drink_fraction * initial_volume
  let second_drink := remaining_after_first - final_volume
  (second_drink / remaining_after_first) = 2/3 := by sorry

end NUMINAMATH_CALUDE_water_consumption_ratio_l509_50959


namespace NUMINAMATH_CALUDE_square_root_five_expansion_square_root_three_expansion_simplify_nested_square_root_l509_50921

-- Part 1
theorem square_root_five_expansion (a b m n : ℤ) :
  a + b * Real.sqrt 5 = (m + n * Real.sqrt 5)^2 →
  a = m^2 + 5 * n^2 ∧ b = 2 * m * n :=
sorry

-- Part 2
theorem square_root_three_expansion :
  ∃ (x m n : ℕ+), x + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 ∧
  ((m = 1 ∧ n = 2 ∧ x = 13) ∨ (m = 2 ∧ n = 1 ∧ x = 7)) :=
sorry

-- Part 3
theorem simplify_nested_square_root :
  Real.sqrt (5 + 2 * Real.sqrt 6) = Real.sqrt 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_square_root_five_expansion_square_root_three_expansion_simplify_nested_square_root_l509_50921


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l509_50911

/-- Two circles with equations x^2 + y^2 + 2ax + a^2 - 4 = 0 and x^2 + y^2 - 4by - 1 + 4b^2 = 0 -/
def circle1 (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*a*x + a^2 - 4 = 0
def circle2 (b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0

/-- The circles have exactly three common tangents -/
def have_three_common_tangents (a b : ℝ) : Prop := sorry

theorem min_value_sum_of_reciprocals (a b : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : have_three_common_tangents a b) : 
  ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), (1 / a^2 + 1 / b^2) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l509_50911


namespace NUMINAMATH_CALUDE_smallest_linear_combination_divides_l509_50962

theorem smallest_linear_combination_divides (a b x₀ y₀ : ℤ) 
  (h_not_zero : a ≠ 0 ∨ b ≠ 0)
  (h_smallest : ∀ x y : ℤ, a * x + b * y > 0 → a * x₀ + b * y₀ ≤ a * x + b * y) :
  ∀ x y : ℤ, ∃ k : ℤ, a * x + b * y = k * (a * x₀ + b * y₀) := by
sorry

end NUMINAMATH_CALUDE_smallest_linear_combination_divides_l509_50962


namespace NUMINAMATH_CALUDE_primes_not_sum_of_composites_l509_50948

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ a b, a > 1 ∧ b > 1 ∧ n = a * b

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d, d ∣ n → d = 1 ∨ d = n

def cannot_be_sum_of_two_composites (p : ℕ) : Prop :=
  is_prime p ∧ ¬∃ a b, is_composite a ∧ is_composite b ∧ p = a + b

theorem primes_not_sum_of_composites :
  {p : ℕ | cannot_be_sum_of_two_composites p} = {2, 3, 5, 7, 11} :=
sorry

end NUMINAMATH_CALUDE_primes_not_sum_of_composites_l509_50948


namespace NUMINAMATH_CALUDE_kath_group_admission_cost_l509_50943

/-- Calculates the total admission cost for a group watching a movie before 6 P.M. -/
def total_admission_cost (regular_price : ℕ) (discount : ℕ) (num_people : ℕ) : ℕ :=
  (regular_price - discount) * num_people

/-- The total admission cost for Kath's group is $30 -/
theorem kath_group_admission_cost :
  let regular_price := 8
  let discount := 3
  let num_people := 6
  total_admission_cost regular_price discount num_people = 30 := by
  sorry

end NUMINAMATH_CALUDE_kath_group_admission_cost_l509_50943


namespace NUMINAMATH_CALUDE_integer_difference_l509_50945

theorem integer_difference (S L : ℤ) : 
  S = 10 → 
  S + L = 30 → 
  5 * S > 2 * L → 
  5 * S - 2 * L = 10 := by
sorry

end NUMINAMATH_CALUDE_integer_difference_l509_50945


namespace NUMINAMATH_CALUDE_bicycle_sale_percentage_prove_bicycle_sale_percentage_l509_50996

/-- The percentage of the suggested retail price that John paid for a bicycle -/
theorem bicycle_sale_percentage : ℝ → ℝ → ℝ → Prop :=
  fun wholesale_price suggested_retail_price johns_price =>
    suggested_retail_price = wholesale_price * (1 + 0.4) →
    johns_price = suggested_retail_price / 3 →
    johns_price / suggested_retail_price = 1 / 3

/-- Proof of the bicycle sale percentage theorem -/
theorem prove_bicycle_sale_percentage :
  ∀ (wholesale_price suggested_retail_price johns_price : ℝ),
    bicycle_sale_percentage wholesale_price suggested_retail_price johns_price := by
  sorry

#check prove_bicycle_sale_percentage

end NUMINAMATH_CALUDE_bicycle_sale_percentage_prove_bicycle_sale_percentage_l509_50996


namespace NUMINAMATH_CALUDE_slope_value_l509_50992

theorem slope_value (m : ℝ) : 
  let A : ℝ × ℝ := (-m, 6)
  let B : ℝ × ℝ := (1, 3*m)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = 12 → m = -2 := by
sorry

end NUMINAMATH_CALUDE_slope_value_l509_50992


namespace NUMINAMATH_CALUDE_parabola_translation_l509_50942

/-- Represents a parabola in the Cartesian coordinate system -/
structure Parabola where
  f : ℝ → ℝ

/-- Translates a parabola horizontally -/
def translate_x (p : Parabola) (dx : ℝ) : Parabola :=
  { f := fun x => p.f (x - dx) }

/-- Translates a parabola vertically -/
def translate_y (p : Parabola) (dy : ℝ) : Parabola :=
  { f := fun x => p.f x + dy }

/-- The original parabola y = x^2 + 3 -/
def original_parabola : Parabola :=
  { f := fun x => x^2 + 3 }

/-- The resulting parabola after translation -/
def resulting_parabola : Parabola :=
  { f := fun x => (x+3)^2 - 1 }

theorem parabola_translation :
  (translate_y (translate_x original_parabola 3) (-4)).f =
  resulting_parabola.f := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l509_50942


namespace NUMINAMATH_CALUDE_smallest_base_for_inequality_l509_50933

theorem smallest_base_for_inequality (k : ℕ) (h : k = 7) : 
  (∃ (base : ℕ), base^k > 4^20 ∧ ∀ (b : ℕ), b < base → b^k ≤ 4^20) ↔ 64^k > 4^20 ∧ ∀ (b : ℕ), b < 64 → b^k ≤ 4^20 :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_inequality_l509_50933


namespace NUMINAMATH_CALUDE_ellipse_parameter_sum_l509_50940

/-- Given two points F₁ and F₂ in the plane, we define an ellipse as the set of points P
    such that PF₁ + PF₂ is constant. This theorem proves that for the specific points
    F₁ = (2, 3) and F₂ = (8, 3), and the constant sum PF₁ + PF₂ = 10, 
    the resulting ellipse has parameters h, k, a, and b whose sum is 17. -/
theorem ellipse_parameter_sum : 
  let F₁ : ℝ × ℝ := (2, 3)
  let F₂ : ℝ × ℝ := (8, 3)
  let distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let is_on_ellipse (P : ℝ × ℝ) : Prop := distance P F₁ + distance P F₂ = 10
  let h : ℝ := (F₁.1 + F₂.1) / 2
  let k : ℝ := F₁.2  -- since F₁.2 = F₂.2
  let c : ℝ := distance F₁ ((F₁.1 + F₂.1) / 2, F₁.2) / 2
  let a : ℝ := 5  -- half of the constant sum
  let b : ℝ := Real.sqrt (a^2 - c^2)
  h + k + a + b = 17
  := by sorry

end NUMINAMATH_CALUDE_ellipse_parameter_sum_l509_50940


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l509_50953

/-- Given a quadratic equation 5 * x^2 + 14 * x + 5 = 0 with two reciprocal roots,
    the coefficient of the squared term is 5. -/
theorem quadratic_coefficient (x : ℝ) :
  (5 * x^2 + 14 * x + 5 = 0) →
  (∃ (r₁ r₂ : ℝ), r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ r₁ * r₂ = 1 ∧ 
    (x = r₁ ∨ x = r₂) ∧ 5 * r₁^2 + 14 * r₁ + 5 = 0 ∧ 5 * r₂^2 + 14 * r₂ + 5 = 0) →
  5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l509_50953


namespace NUMINAMATH_CALUDE_intersection_M_N_l509_50947

open Set Real

-- Define the sets M and N
def M : Set ℝ := {x | 2 - x > 0}
def N : Set ℝ := Icc 1 3

-- State the theorem
theorem intersection_M_N : M ∩ N = Ico 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l509_50947


namespace NUMINAMATH_CALUDE_terrell_workout_equivalence_l509_50969

/-- Given Terrell's original workout and new weights, calculate the number of lifts needed to match the total weight. -/
theorem terrell_workout_equivalence (original_weight original_reps new_weight : ℕ) : 
  original_weight = 30 →
  original_reps = 10 →
  new_weight = 20 →
  (2 * new_weight * (600 / (2 * new_weight)) : ℕ) = 2 * original_weight * original_reps :=
by
  sorry

#check terrell_workout_equivalence

end NUMINAMATH_CALUDE_terrell_workout_equivalence_l509_50969


namespace NUMINAMATH_CALUDE_remainder_1493829_div_7_l509_50906

theorem remainder_1493829_div_7 : 1493829 % 7 = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_1493829_div_7_l509_50906


namespace NUMINAMATH_CALUDE_gyroscope_spin_rate_doubling_time_l509_50946

/-- The time interval for which a gyroscope's spin rate doubles -/
theorem gyroscope_spin_rate_doubling_time (v₀ v t : ℝ) (h₁ : v₀ = 6.25) (h₂ : v = 400) (h₃ : t = 90) :
  ∃ T : ℝ, v = v₀ * 2^(t/T) ∧ T = 15 := by
  sorry

end NUMINAMATH_CALUDE_gyroscope_spin_rate_doubling_time_l509_50946


namespace NUMINAMATH_CALUDE_f_greater_than_g_l509_50916

/-- The function f defined as f(x) = 3x^2 - x + 1 -/
def f (x : ℝ) : ℝ := 3 * x^2 - x + 1

/-- The function g defined as g(x) = 2x^2 + x - 1 -/
def g (x : ℝ) : ℝ := 2 * x^2 + x - 1

/-- For all real x, f(x) > g(x) -/
theorem f_greater_than_g : ∀ x : ℝ, f x > g x := by sorry

end NUMINAMATH_CALUDE_f_greater_than_g_l509_50916


namespace NUMINAMATH_CALUDE_relay_race_arrangements_l509_50965

/-- The number of students in the class -/
def total_students : Nat := 8

/-- The number of students needed for the relay race -/
def relay_team_size : Nat := 4

/-- The number of students that must be selected (A and B) -/
def must_select : Nat := 2

/-- The number of positions where A and B can be placed (first or last) -/
def fixed_positions : Nat := 2

/-- The number of remaining positions to be filled -/
def remaining_positions : Nat := relay_team_size - must_select

/-- The number of remaining students to choose from -/
def remaining_students : Nat := total_students - must_select

theorem relay_race_arrangements :
  (fixed_positions.factorial) *
  (remaining_students.choose remaining_positions) *
  (remaining_positions.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_l509_50965


namespace NUMINAMATH_CALUDE_computer_selection_count_l509_50991

def lenovo_count : ℕ := 4
def crsc_count : ℕ := 5
def total_selection : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem computer_selection_count :
  (choose lenovo_count 1 * choose crsc_count 2) + 
  (choose lenovo_count 2 * choose crsc_count 1) = 70 := by
  sorry

end NUMINAMATH_CALUDE_computer_selection_count_l509_50991


namespace NUMINAMATH_CALUDE_complex_parts_of_i_squared_plus_i_l509_50927

theorem complex_parts_of_i_squared_plus_i :
  let i : ℂ := Complex.I
  let z : ℂ := i^2 + i
  (z.re = -1) ∧ (z.im = 1) := by sorry

end NUMINAMATH_CALUDE_complex_parts_of_i_squared_plus_i_l509_50927


namespace NUMINAMATH_CALUDE_real_solutions_condition_l509_50951

theorem real_solutions_condition (x : ℝ) :
  (∃ y : ℝ, y^2 + 6*x*y + x + 8 = 0) ↔ (x ≤ -8/9 ∨ x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_real_solutions_condition_l509_50951


namespace NUMINAMATH_CALUDE_part_one_part_two_l509_50979

-- Define the sets A and B
def A : Set ℝ := {x | x < -3 ∨ x > 7}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Part (1)
theorem part_one (m : ℝ) : 
  (Set.univ \ A) ∪ B m = Set.univ \ A ↔ m ≤ 4 := by sorry

-- Part (2)
theorem part_two (m : ℝ) : 
  (∃ (a b : ℝ), (Set.univ \ A) ∩ B m = {x | a ≤ x ∧ x ≤ b} ∧ b - a ≥ 1) ↔ 
  (3 ≤ m ∧ m ≤ 5) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l509_50979


namespace NUMINAMATH_CALUDE_rhombus_area_l509_50934

/-- The area of a rhombus with side length 3 cm and an acute angle of 45 degrees is 9√2/2 square centimeters. -/
theorem rhombus_area (side_length : ℝ) (acute_angle : ℝ) :
  side_length = 3 →
  acute_angle = 45 * π / 180 →
  let area := side_length * side_length * Real.sin acute_angle
  area = 9 * Real.sqrt 2 / 2 := by
  sorry

#check rhombus_area

end NUMINAMATH_CALUDE_rhombus_area_l509_50934


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l509_50923

-- Problem 1
theorem problem_1 (x y : ℝ) : (x - y)^3 * (y - x)^2 = (x - y)^5 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (-3 * a^2)^3 = -27 * a^6 := by sorry

-- Problem 3
theorem problem_3 (x : ℝ) (h : x ≠ 0) : x^10 / (2*x)^2 = x^8 / 4 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l509_50923


namespace NUMINAMATH_CALUDE_ball_probabilities_l509_50972

-- Define the sample space
def Ω : Type := Unit

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define the events
def red : Set Ω := sorry
def black : Set Ω := sorry
def white : Set Ω := sorry
def green : Set Ω := sorry

-- State the theorem
theorem ball_probabilities 
  (h1 : P red = 5/12)
  (h2 : P black = 1/3)
  (h3 : P white = 1/6)
  (h4 : P green = 1/12)
  (h5 : Disjoint red black)
  (h6 : Disjoint red white)
  (h7 : Disjoint red green)
  (h8 : Disjoint black white)
  (h9 : Disjoint black green)
  (h10 : Disjoint white green) :
  (P (red ∪ black) = 3/4) ∧ 
  (P (red ∪ black ∪ white) = 11/12) := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l509_50972


namespace NUMINAMATH_CALUDE_inequality_system_solution_l509_50903

theorem inequality_system_solution (k : ℝ) : 
  (∀ x : ℤ, (x^2 - x - 2 > 0 ∧ 2*x^2 + (2*k+5)*x + 5*k < 0) ↔ x = -2) →
  -3 ≤ k ∧ k < 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l509_50903


namespace NUMINAMATH_CALUDE_computation_proof_l509_50902

theorem computation_proof : 8 * (250 / 3 + 50 / 6 + 16 / 32 + 2) = 2260 / 3 := by
  sorry

end NUMINAMATH_CALUDE_computation_proof_l509_50902


namespace NUMINAMATH_CALUDE_hyperbola_equation_l509_50939

/-- Given a hyperbola and conditions on its asymptote and focus, prove its equation -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) ∧ 
  (∃ (x y : ℝ), y = Real.sqrt 3 * x) ∧
  (∃ (x : ℝ), x^2 + b^2 = 144) →
  a^2 = 36 ∧ b^2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l509_50939


namespace NUMINAMATH_CALUDE_tetrahedron_edge_length_is_2_sqrt_5_l509_50971

-- Define the radius of each ball
def ball_radius : ℝ := 2

-- Define the arrangement of balls
structure BallArrangement where
  bottom_balls : Fin 4 → ℝ × ℝ × ℝ  -- Centers of the four bottom balls
  top_ball : ℝ × ℝ × ℝ              -- Center of the top ball

-- Define the properties of the arrangement
def valid_arrangement (arr : BallArrangement) : Prop :=
  -- Four bottom balls are mutually tangent
  ∀ i j, i ≠ j → ‖arr.bottom_balls i - arr.bottom_balls j‖ = 2 * ball_radius
  -- Top ball is tangent to all bottom balls
  ∧ ∀ i, ‖arr.top_ball - arr.bottom_balls i‖ = 2 * ball_radius

-- Define the tetrahedron circumscribed around the arrangement
def tetrahedron_edge_length (arr : BallArrangement) : ℝ :=
  ‖arr.top_ball - arr.bottom_balls 0‖

-- Theorem statement
theorem tetrahedron_edge_length_is_2_sqrt_5 (arr : BallArrangement) 
  (h : valid_arrangement arr) : 
  tetrahedron_edge_length arr = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_length_is_2_sqrt_5_l509_50971


namespace NUMINAMATH_CALUDE_pascal_triangle_specific_number_l509_50910

/-- The number of elements in the nth row of Pascal's triangle -/
def pascal_row_length (n : ℕ) : ℕ := n + 1

/-- The kth number in the nth row of Pascal's triangle -/
def pascal_number (n k : ℕ) : ℕ := Nat.choose n (k - 1)

theorem pascal_triangle_specific_number :
  pascal_number 50 10 = 2586948580 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_specific_number_l509_50910


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l509_50997

theorem least_number_with_remainder (n : ℕ) : 
  (n % 6 = 4 ∧ n % 7 = 4 ∧ n % 9 = 4 ∧ n % 18 = 4) →
  (∀ m : ℕ, m < n → ¬(m % 6 = 4 ∧ m % 7 = 4 ∧ m % 9 = 4 ∧ m % 18 = 4)) →
  n = 130 := by
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l509_50997


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l509_50967

/-- Given polynomials P, Q, R, and S with integer coefficients satisfying the equation
    P(x^5) + x Q(x^5) + x^2 R(x^5) = (1 + x + x^2 + x^3 + x^4) S(x),
    prove that there exists a polynomial p such that P(x) = (x - 1) * p(x). -/
theorem polynomial_divisibility 
  (P Q R S : Polynomial ℤ) 
  (h : P.comp (X^5 : Polynomial ℤ) + X * Q.comp (X^5 : Polynomial ℤ) + X^2 * R.comp (X^5 : Polynomial ℤ) = 
       (1 + X + X^2 + X^3 + X^4) * S) : 
  ∃ p : Polynomial ℤ, P = (X - 1) * p := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l509_50967


namespace NUMINAMATH_CALUDE_interest_rate_difference_l509_50926

/-- Proves that the difference in interest rates is 5% given the specified conditions -/
theorem interest_rate_difference
  (principal : ℝ)
  (time : ℝ)
  (interest_difference : ℝ)
  (h1 : principal = 800)
  (h2 : time = 10)
  (h3 : interest_difference = 400)
  : ∃ (r1 r2 : ℝ), r2 - r1 = 5 ∧ 
    principal * r2 * time / 100 = principal * r1 * time / 100 + interest_difference :=
sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l509_50926


namespace NUMINAMATH_CALUDE_hex_sum_equals_451A5_l509_50981

/-- Represents a hexadecimal digit --/
def HexDigit : Type := Fin 16

/-- Represents a hexadecimal number as a list of digits --/
def HexNumber := List HexDigit

/-- Convert a natural number to its hexadecimal representation --/
def toHex (n : ℕ) : HexNumber := sorry

/-- Convert a hexadecimal number to its natural number representation --/
def fromHex (h : HexNumber) : ℕ := sorry

/-- Addition of hexadecimal numbers --/
def hexAdd (a b : HexNumber) : HexNumber := sorry

theorem hex_sum_equals_451A5 :
  let a := toHex 25  -- 19₁₆
  let b := toHex 12  -- C₁₆
  let c := toHex 432 -- 1B0₁₆
  let d := toHex 929 -- 3A1₁₆
  let e := toHex 47  -- 2F₁₆
  hexAdd a (hexAdd b (hexAdd c (hexAdd d e))) = toHex 283045 -- 451A5₁₆
  := by sorry

end NUMINAMATH_CALUDE_hex_sum_equals_451A5_l509_50981


namespace NUMINAMATH_CALUDE_equal_perimeter_lines_concurrent_l509_50949

open Real

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a line through a vertex
structure VertexLine :=
  (vertex : ℝ × ℝ)
  (point : ℝ × ℝ)

-- Function to check if a line divides a triangle into two triangles with equal perimeter
def divides_equal_perimeter (t : Triangle) (l : VertexLine) : Prop :=
  sorry

-- Function to check if three lines are concurrent
def are_concurrent (l1 l2 l3 : VertexLine) : Prop :=
  sorry

-- Theorem statement
theorem equal_perimeter_lines_concurrent (t : Triangle) :
  ∀ (l1 l2 l3 : VertexLine),
    (divides_equal_perimeter t l1 ∧ 
     divides_equal_perimeter t l2 ∧ 
     divides_equal_perimeter t l3) →
    are_concurrent l1 l2 l3 :=
sorry

end NUMINAMATH_CALUDE_equal_perimeter_lines_concurrent_l509_50949


namespace NUMINAMATH_CALUDE_enchiladas_and_tacos_price_l509_50924

-- Define the prices of enchiladas and tacos
noncomputable def enchilada_price : ℝ := sorry
noncomputable def taco_price : ℝ := sorry

-- Define the conditions
axiom condition1 : enchilada_price + 4 * taco_price = 3
axiom condition2 : 4 * enchilada_price + taco_price = 3.2

-- State the theorem
theorem enchiladas_and_tacos_price :
  4 * enchilada_price + 5 * taco_price = 5.55 := by sorry

end NUMINAMATH_CALUDE_enchiladas_and_tacos_price_l509_50924


namespace NUMINAMATH_CALUDE_symmetric_line_problem_l509_50908

/-- Given a line with equation ax + by + c = 0, 
    returns the equation of the line symmetric to it with respect to y = x -/
def symmetric_line (a b c : ℝ) : ℝ × ℝ × ℝ := (b, a, c)

/-- The equation of the line symmetric to 3x-5y+1=0 with respect to y=x -/
theorem symmetric_line_problem : 
  symmetric_line 3 (-5) 1 = (5, -3, -1) := by sorry

end NUMINAMATH_CALUDE_symmetric_line_problem_l509_50908


namespace NUMINAMATH_CALUDE_max_vertex_coordinate_sum_l509_50928

/-- Given a parabola y = ax^2 + bx + c passing through (0,0), (2T,0), and (2T+1,35),
    where a and T are integers and T ≠ 0, the maximum sum of vertex coordinates is 34. -/
theorem max_vertex_coordinate_sum :
  ∀ (a T : ℤ) (b c : ℝ),
    T ≠ 0 →
    (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ 
      (x = 0 ∧ y = 0) ∨ 
      (x = 2 * T ∧ y = 0) ∨ 
      (x = 2 * T + 1 ∧ y = 35)) →
    (∃ (N : ℝ), N = T - a * T^2 ∧ 
      (∀ (N' : ℝ), (∃ (a' T' : ℤ) (b' c' : ℝ),
        T' ≠ 0 ∧
        (∀ x y : ℝ, y = a' * x^2 + b' * x + c' ↔ 
          (x = 0 ∧ y = 0) ∨ 
          (x = 2 * T' ∧ y = 0) ∨ 
          (x = 2 * T' + 1 ∧ y = 35)) ∧
        N' = T' - a' * T'^2) → N' ≤ N)) →
    (∃ (N : ℝ), N = 34 ∧
      (∀ (N' : ℝ), (∃ (a' T' : ℤ) (b' c' : ℝ),
        T' ≠ 0 ∧
        (∀ x y : ℝ, y = a' * x^2 + b' * x + c' ↔ 
          (x = 0 ∧ y = 0) ∨ 
          (x = 2 * T' ∧ y = 0) ∨ 
          (x = 2 * T' + 1 ∧ y = 35)) ∧
        N' = T' - a' * T'^2) → N' ≤ N)) :=
by sorry

end NUMINAMATH_CALUDE_max_vertex_coordinate_sum_l509_50928


namespace NUMINAMATH_CALUDE_shirt_and_coat_cost_l509_50964

/-- Given a shirt that costs $150 and is one-third the price of a coat,
    prove that the total cost of the shirt and coat is $600. -/
theorem shirt_and_coat_cost (shirt_cost : ℕ) (coat_cost : ℕ) : 
  shirt_cost = 150 → 
  shirt_cost * 3 = coat_cost →
  shirt_cost + coat_cost = 600 := by
  sorry

end NUMINAMATH_CALUDE_shirt_and_coat_cost_l509_50964


namespace NUMINAMATH_CALUDE_money_division_l509_50914

theorem money_division (p q r : ℕ) (total : ℕ) : 
  p * 4 = q * 5 →
  q * 10 = r * 9 →
  r = 400 →
  total = p + q + r →
  total = 1210 := by
sorry

end NUMINAMATH_CALUDE_money_division_l509_50914


namespace NUMINAMATH_CALUDE_marcos_lap_time_improvement_l509_50909

/-- Represents the improvement in lap time for Marcos after training -/
theorem marcos_lap_time_improvement :
  let initial_laps : ℕ := 15
  let initial_time : ℕ := 45
  let final_laps : ℕ := 18
  let final_time : ℕ := 42
  let initial_lap_time := initial_time / initial_laps
  let final_lap_time := final_time / final_laps
  let improvement := initial_lap_time - final_lap_time
  improvement = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_marcos_lap_time_improvement_l509_50909


namespace NUMINAMATH_CALUDE_dress_cost_calculation_l509_50920

def dresses : ℕ := 5
def pants : ℕ := 3
def jackets : ℕ := 4
def pants_cost : ℕ := 12
def jackets_cost : ℕ := 30
def transportation_cost : ℕ := 5
def initial_money : ℕ := 400
def remaining_money : ℕ := 139

theorem dress_cost_calculation (dress_cost : ℕ) : 
  dress_cost * dresses + pants * pants_cost + jackets * jackets_cost + transportation_cost = initial_money - remaining_money → 
  dress_cost = 20 := by
sorry

end NUMINAMATH_CALUDE_dress_cost_calculation_l509_50920


namespace NUMINAMATH_CALUDE_trapezoid_height_l509_50977

theorem trapezoid_height (upper_side lower_side area height : ℝ) : 
  upper_side = 5 →
  lower_side = 9 →
  area = 56 →
  area = (1/2) * (upper_side + lower_side) * height →
  height = 8 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_height_l509_50977


namespace NUMINAMATH_CALUDE_different_color_chip_probability_l509_50963

theorem different_color_chip_probability : 
  let total_chips : ℕ := 6 + 5 + 4
  let green_chips : ℕ := 6
  let blue_chips : ℕ := 5
  let red_chips : ℕ := 4
  let prob_green : ℚ := green_chips / total_chips
  let prob_blue : ℚ := blue_chips / total_chips
  let prob_red : ℚ := red_chips / total_chips
  let prob_not_green : ℚ := (blue_chips + red_chips) / total_chips
  let prob_not_blue : ℚ := (green_chips + red_chips) / total_chips
  let prob_not_red : ℚ := (green_chips + blue_chips) / total_chips
  prob_green * prob_not_green + prob_blue * prob_not_blue + prob_red * prob_not_red = 148 / 225 :=
by sorry

end NUMINAMATH_CALUDE_different_color_chip_probability_l509_50963


namespace NUMINAMATH_CALUDE_sqrt_point_five_equals_sqrt_two_over_two_l509_50915

theorem sqrt_point_five_equals_sqrt_two_over_two :
  Real.sqrt 0.5 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_point_five_equals_sqrt_two_over_two_l509_50915


namespace NUMINAMATH_CALUDE_ending_number_divisible_by_three_eleven_numbers_divisible_by_three_l509_50970

theorem ending_number_divisible_by_three (start : Nat) (count : Nat) (divisor : Nat) : Nat :=
  let first_divisible := start + (divisor - start % divisor) % divisor
  first_divisible + (count - 1) * divisor

theorem eleven_numbers_divisible_by_three : 
  ending_number_divisible_by_three 10 11 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_ending_number_divisible_by_three_eleven_numbers_divisible_by_three_l509_50970


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l509_50931

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

-- State the theorem
theorem complement_of_M_in_U : 
  Set.compl M = {x : ℝ | x < -2 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l509_50931


namespace NUMINAMATH_CALUDE_isabellas_houses_l509_50905

theorem isabellas_houses (green yellow red : ℕ) : 
  green = 3 * yellow →
  yellow = red - 40 →
  green + red = 160 →
  green + red = 160 := by
sorry

end NUMINAMATH_CALUDE_isabellas_houses_l509_50905


namespace NUMINAMATH_CALUDE_problem_solution_l509_50922

theorem problem_solution : 
  (∃ n : ℕ, 140 * 5 = n * 100 ∧ n % 10 ≠ 0) ∧ 
  (4 * 150 - 7 = 593) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l509_50922


namespace NUMINAMATH_CALUDE_profit_share_ratio_l509_50930

theorem profit_share_ratio (total_profit : ℚ) (difference : ℚ) 
  (h1 : total_profit = 800)
  (h2 : difference = 160) :
  ∃ (x y : ℚ), x + y = total_profit ∧ 
                |x - y| = difference ∧ 
                y / total_profit = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_ratio_l509_50930


namespace NUMINAMATH_CALUDE_triangle_problem_l509_50978

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) (T : ℝ) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  (2 * a - c) * Real.cos B = b * Real.cos C ∧
  b = Real.sqrt 3 ∧
  T = (1 / 2) * a * c * Real.sin B →
  B = π / 3 ∧ a + c = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l509_50978


namespace NUMINAMATH_CALUDE_symmetric_axis_of_shifted_quadratic_unique_symmetric_axis_l509_50944

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * (x + 3)^2 - 2

-- Define the symmetric axis
def symmetric_axis : ℝ := -3

-- Theorem statement
theorem symmetric_axis_of_shifted_quadratic :
  ∀ x : ℝ, f (symmetric_axis + x) = f (symmetric_axis - x) := by
  sorry

-- The symmetric axis is unique
theorem unique_symmetric_axis :
  ∀ h : ℝ, h ≠ symmetric_axis →
  ∃ x : ℝ, f (h + x) ≠ f (h - x) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_axis_of_shifted_quadratic_unique_symmetric_axis_l509_50944


namespace NUMINAMATH_CALUDE_triangle_prime_angles_l509_50907

theorem triangle_prime_angles (a b c : ℕ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c →  -- All angles are prime
  a = 2 ∨ b = 2 ∨ c = 2 :=  -- One angle must be 2 degrees
by
  sorry

end NUMINAMATH_CALUDE_triangle_prime_angles_l509_50907


namespace NUMINAMATH_CALUDE_remainder_theorem_l509_50941

theorem remainder_theorem (n : ℤ) : n % 9 = 3 → (4 * n - 9) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l509_50941


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l509_50999

theorem complex_fraction_equality (a b : ℂ) 
  (h : (a + b) / (a - b) + (a - b) / (a + b) = 4) :
  (a^4 + b^4) / (a^4 - b^4) + (a^4 - b^4) / (a^4 + b^4) = 41 / 20 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l509_50999


namespace NUMINAMATH_CALUDE_l_structure_surface_area_l509_50993

/-- Represents the L-shaped structure composed of unit cubes -/
structure LStructure where
  bottom_row : Nat
  first_stack : Nat
  second_stack : Nat

/-- Calculates the surface area of the L-shaped structure -/
def surface_area (l : LStructure) : Nat :=
  let bottom_area := 2 * l.bottom_row + 2
  let first_stack_area := 1 + 1 + 3 + 3 + 2
  let second_stack_area := 1 + 5 + 5 + 2
  bottom_area + first_stack_area + second_stack_area

/-- Theorem stating that the surface area of the specific L-shaped structure is 39 square units -/
theorem l_structure_surface_area :
  surface_area { bottom_row := 7, first_stack := 3, second_stack := 5 } = 39 := by
  sorry

#eval surface_area { bottom_row := 7, first_stack := 3, second_stack := 5 }

end NUMINAMATH_CALUDE_l_structure_surface_area_l509_50993


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l509_50958

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) :
  z.im = 1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l509_50958


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l509_50995

/-- The number of heartbeats during a race -/
def heartbeats_during_race (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

/-- Proof that the athlete's heart beats 19200 times during the race -/
theorem athlete_heartbeats :
  heartbeats_during_race 160 6 20 = 19200 := by
  sorry

#eval heartbeats_during_race 160 6 20

end NUMINAMATH_CALUDE_athlete_heartbeats_l509_50995
