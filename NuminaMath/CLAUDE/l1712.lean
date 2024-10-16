import Mathlib

namespace NUMINAMATH_CALUDE_circle_config_exists_l1712_171205

-- Define the type for our circle configuration
def CircleConfig := Fin 8 → Fin 8

-- Define a function to check if two numbers are connected in our configuration
def isConnected (i j : Fin 8) : Prop :=
  (i.val = j.val + 1 ∧ i.val % 2 = 0) ∨
  (j.val = i.val + 1 ∧ j.val % 2 = 0) ∨
  (i.val = j.val + 2 ∧ i.val % 4 = 0) ∨
  (j.val = i.val + 2 ∧ j.val % 4 = 0)

-- Define the property that the configuration satisfies the problem conditions
def validConfig (c : CircleConfig) : Prop :=
  (∀ i : Fin 8, c i ≠ 0) ∧
  (∀ i j : Fin 8, i ≠ j → c i ≠ c j) ∧
  (∀ d : Fin 7, ∃! (i j : Fin 8), isConnected i j ∧ |c i - c j| = d + 1)

-- State the theorem
theorem circle_config_exists : ∃ c : CircleConfig, validConfig c := by
  sorry

end NUMINAMATH_CALUDE_circle_config_exists_l1712_171205


namespace NUMINAMATH_CALUDE_pizza_group_size_l1712_171280

theorem pizza_group_size (slices_per_person : ℕ) (slices_per_pizza : ℕ) (num_pizzas : ℕ)
  (h1 : slices_per_person = 3)
  (h2 : slices_per_pizza = 9)
  (h3 : num_pizzas = 6) :
  (num_pizzas * (slices_per_pizza / slices_per_person)) = 18 :=
by sorry

end NUMINAMATH_CALUDE_pizza_group_size_l1712_171280


namespace NUMINAMATH_CALUDE_graphic_artist_pages_sum_l1712_171223

theorem graphic_artist_pages_sum (n : ℕ) (a₁ d : ℝ) : 
  n = 15 ∧ a₁ = 3 ∧ d = 2 → 
  (n / 2 : ℝ) * (2 * a₁ + (n - 1) * d) = 255 := by
  sorry

end NUMINAMATH_CALUDE_graphic_artist_pages_sum_l1712_171223


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l1712_171220

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The problem statement -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l1712_171220


namespace NUMINAMATH_CALUDE_complex_point_C_l1712_171246

/-- Given points A, B, and C in the complex plane, prove that C corresponds to 4-2i -/
theorem complex_point_C (A B C : ℂ) : 
  A = 2 + I →
  B - A = 1 + 2*I →
  C - B = 3 - I →
  C = 4 - 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_point_C_l1712_171246


namespace NUMINAMATH_CALUDE_acid_dilution_l1712_171295

/-- Given an initial acid solution with concentration p% and volume p ounces,
    adding y ounces of water results in a (p-15)% acid solution.
    This theorem proves that y = 15p / (p-15) when p > 30. -/
theorem acid_dilution (p : ℝ) (y : ℝ) (h : p > 30) :
  (p * p / 100 = (p - 15) / 100 * (p + y)) → y = 15 * p / (p - 15) := by
  sorry


end NUMINAMATH_CALUDE_acid_dilution_l1712_171295


namespace NUMINAMATH_CALUDE_variance_of_data_l1712_171202

def data : List ℝ := [3, 2, 1, 0, 0, 0, 1]

theorem variance_of_data : 
  let n : ℝ := data.length
  let mean := (data.sum) / n
  let variance := (data.map (fun x => (x - mean)^2)).sum / n
  variance = 8/7 := by sorry

end NUMINAMATH_CALUDE_variance_of_data_l1712_171202


namespace NUMINAMATH_CALUDE_composite_function_equality_l1712_171243

/-- Given two functions f and g, and a real number b, proves that if f(g(b)) = 3,
    then b = 1/2. -/
theorem composite_function_equality (f g : ℝ → ℝ) (b : ℝ) 
    (hf : ∀ x, f x = x / 4 + 2)
    (hg : ∀ x, g x = 5 - 2 * x)
    (h : f (g b) = 3) : b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_equality_l1712_171243


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_l1712_171253

/-- If the terminal side of angle α passes through point (-1, 2), then sin α = 2√5/5 -/
theorem sin_alpha_for_point (α : Real) : 
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -1 ∧ t * Real.sin α = 2) → 
  Real.sin α = 2 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_l1712_171253


namespace NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l1712_171294

theorem police_emergency_number_prime_divisor (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ k : ℕ, n = 1000 * k + 133) : 
  ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n := by
sorry

end NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l1712_171294


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1712_171219

theorem simplify_complex_fraction :
  1 / ((2 / (Real.sqrt 2 + 2)) + (3 / (Real.sqrt 3 - 2)) + (4 / (Real.sqrt 5 + 1))) =
  (Real.sqrt 2 + 3 * Real.sqrt 3 - Real.sqrt 5 + 5) / 27 := by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1712_171219


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l1712_171230

def number_of_people : ℕ := 5

-- Define a function to calculate permutations
def permutations (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / Nat.factorial (n - k)

theorem valid_arrangements_count :
  (permutations number_of_people number_of_people) -
  (permutations (number_of_people - 2) (number_of_people - 2)) -
  (permutations (number_of_people - 2) 1 * permutations (number_of_people - 2) (number_of_people - 2)) -
  (permutations (number_of_people - 1) 1 * permutations (number_of_people - 2) (number_of_people - 2)) = 72 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l1712_171230


namespace NUMINAMATH_CALUDE_non_pine_trees_count_l1712_171224

/-- Given a park with 350 trees, where 70% are pine trees, prove that 105 trees are not pine trees. -/
theorem non_pine_trees_count (total_trees : ℕ) (pine_percentage : ℚ) : 
  total_trees = 350 → pine_percentage = 70 / 100 →
  (total_trees : ℚ) - (pine_percentage * total_trees) = 105 := by
  sorry

end NUMINAMATH_CALUDE_non_pine_trees_count_l1712_171224


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1712_171209

theorem perpendicular_lines_b_value (b : ℚ) : 
  (∀ x y : ℚ, 2 * x - 3 * y + 6 = 0 → (∃ m₁ : ℚ, y = m₁ * x + 2)) ∧ 
  (∀ x y : ℚ, b * x - 3 * y + 6 = 0 → (∃ m₂ : ℚ, y = m₂ * x + 2)) ∧
  (∃ m₁ m₂ : ℚ, m₁ * m₂ = -1) →
  b = -9/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1712_171209


namespace NUMINAMATH_CALUDE_mike_marks_short_l1712_171281

def passing_threshold (max_marks : ℕ) : ℕ := (30 * max_marks) / 100

theorem mike_marks_short (max_marks mike_score : ℕ) 
  (h1 : max_marks = 760) 
  (h2 : mike_score = 212) : 
  passing_threshold max_marks - mike_score = 16 := by
  sorry

end NUMINAMATH_CALUDE_mike_marks_short_l1712_171281


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1712_171222

theorem perfect_square_condition (a b k : ℝ) :
  (∃ (c : ℝ), 4 * a^2 + k * a * b + 9 * b^2 = c^2) →
  k = 12 ∨ k = -12 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1712_171222


namespace NUMINAMATH_CALUDE_tangent_line_parallel_point_exists_l1712_171266

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_line_parallel_point_exists :
  ∃ (x y : ℝ), f x = y ∧ f' x = 4 ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_point_exists_l1712_171266


namespace NUMINAMATH_CALUDE_function_difference_l1712_171289

theorem function_difference (f : ℝ → ℝ) (h : ∀ x, f x = 9^x) :
  ∀ x, f (x + 1) - f x = 8 * f x := by
  sorry

end NUMINAMATH_CALUDE_function_difference_l1712_171289


namespace NUMINAMATH_CALUDE_road_signs_at_first_intersection_l1712_171277

/-- Represents the number of road signs at each intersection -/
structure RoadSigns where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Defines the relationship between road signs at different intersections -/
def valid_road_signs (rs : RoadSigns) : Prop :=
  rs.second = rs.first + rs.first / 4 ∧
  rs.third = 2 * rs.second ∧
  rs.fourth = rs.third - 20 ∧
  rs.first + rs.second + rs.third + rs.fourth = 270

theorem road_signs_at_first_intersection :
  ∃ (rs : RoadSigns), valid_road_signs rs ∧ rs.first = 40 :=
by sorry

end NUMINAMATH_CALUDE_road_signs_at_first_intersection_l1712_171277


namespace NUMINAMATH_CALUDE_garlic_cloves_used_l1712_171252

/-- Proves that the number of garlic cloves used for cooking is the difference between
    the initial number and the remaining number of cloves -/
theorem garlic_cloves_used (initial : ℕ) (remaining : ℕ) (used : ℕ) 
    (h1 : initial = 93)
    (h2 : remaining = 7)
    (h3 : used = initial - remaining) :
  used = 86 := by
  sorry

end NUMINAMATH_CALUDE_garlic_cloves_used_l1712_171252


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1712_171216

theorem sufficient_not_necessary (a b : ℝ) : 
  (∀ a b, b ≥ 0 → a^2 + b ≥ 0) ∧ 
  (∃ a b, a^2 + b ≥ 0 ∧ b < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1712_171216


namespace NUMINAMATH_CALUDE_prob_one_head_two_tails_l1712_171231

/-- The probability of getting one head and two tails when tossing three fair coins -/
theorem prob_one_head_two_tails : ℝ := by
  -- Define the number of possible outcomes when tossing three fair coins
  let total_outcomes : ℕ := 2^3

  -- Define the number of ways to get one head and two tails
  let favorable_outcomes : ℕ := 3

  -- Define the probability as the ratio of favorable outcomes to total outcomes
  let probability : ℝ := favorable_outcomes / total_outcomes

  -- Prove that this probability equals 3/8
  sorry

end NUMINAMATH_CALUDE_prob_one_head_two_tails_l1712_171231


namespace NUMINAMATH_CALUDE_savings_after_twelve_months_l1712_171264

/-- Represents the electricity pricing and consumption data for a user. -/
structure ElectricityData where
  originalPrice : ℚ
  valleyPrice : ℚ
  peakPrice : ℚ
  installationFee : ℚ
  monthlyConsumption : ℚ
  valleyConsumption : ℚ
  peakConsumption : ℚ
  months : ℕ

/-- Calculates the total savings after a given number of months for a user
    who has installed a peak-valley meter. -/
def totalSavings (data : ElectricityData) : ℚ :=
  let monthlyOriginalCost := data.monthlyConsumption * data.originalPrice
  let monthlyNewCost := data.valleyConsumption * data.valleyPrice + data.peakConsumption * data.peakPrice
  let monthlySavings := monthlyOriginalCost - monthlyNewCost
  let totalSavingsBeforeFee := monthlySavings * data.months
  totalSavingsBeforeFee - data.installationFee

/-- The main theorem stating that the total savings after 12 months is 236 yuan. -/
theorem savings_after_twelve_months :
  let data : ElectricityData := {
    originalPrice := 56/100,
    valleyPrice := 28/100,
    peakPrice := 56/100,
    installationFee := 100,
    monthlyConsumption := 200,
    valleyConsumption := 100,
    peakConsumption := 100,
    months := 12
  }
  totalSavings data = 236 := by sorry

end NUMINAMATH_CALUDE_savings_after_twelve_months_l1712_171264


namespace NUMINAMATH_CALUDE_new_person_weight_l1712_171227

def initial_persons : ℕ := 10
def average_weight_increase : ℚ := 63/10
def replaced_person_weight : ℚ := 65

theorem new_person_weight :
  let total_weight_increase : ℚ := initial_persons * average_weight_increase
  let new_person_weight : ℚ := replaced_person_weight + total_weight_increase
  new_person_weight = 128 := by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1712_171227


namespace NUMINAMATH_CALUDE_justin_reading_ratio_l1712_171276

/-- Proves that the ratio of pages read each day in the remaining 6 days to the first day is 2:1 -/
theorem justin_reading_ratio : ∀ (pages_first_day : ℕ) (total_pages : ℕ) (days_remaining : ℕ),
  pages_first_day = 10 →
  total_pages = 130 →
  days_remaining = 6 →
  (days_remaining * (pages_first_day * (total_pages - pages_first_day) / (pages_first_day * days_remaining)) = total_pages - pages_first_day) →
  (total_pages - pages_first_day) / (pages_first_day * days_remaining) = 2 := by
  sorry

end NUMINAMATH_CALUDE_justin_reading_ratio_l1712_171276


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1712_171233

def arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a + i * d)

def last_term (seq : List ℕ) : ℕ :=
  match seq.getLast? with
  | some x => x
  | none => 0

theorem arithmetic_sequence_sum (a d : ℕ) :
  ∀ seq : List ℕ, seq = arithmetic_sequence a d seq.length →
  last_term seq = 50 →
  seq.sum = 442 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1712_171233


namespace NUMINAMATH_CALUDE_min_new_respondents_l1712_171288

/-- Given the initial conditions of a survey, proves that the minimum number of new respondents
    needed to change the ratio from 4:7:14 to 6:9:16 is 75. -/
theorem min_new_respondents (initial_total : ℕ) (initial_ratio : Fin 3 → ℕ) 
    (new_ratio : Fin 3 → ℕ) : 
  initial_total = 700 →
  initial_ratio 0 = 4 →
  initial_ratio 1 = 7 →
  initial_ratio 2 = 14 →
  new_ratio 0 = 6 →
  new_ratio 1 = 9 →
  new_ratio 2 = 16 →
  (∃ (new_total : ℕ),
    new_total > initial_total ∧
    (∀ i : Fin 3, (new_total * new_ratio i) / (new_ratio 0 + new_ratio 1 + new_ratio 2) ≥ 
      (initial_total * initial_ratio i) / (initial_ratio 0 + initial_ratio 1 + initial_ratio 2)) ∧
    ∀ (other_total : ℕ),
      other_total > initial_total →
      other_total < new_total →
      (∃ i : Fin 3, (other_total * new_ratio i) / (new_ratio 0 + new_ratio 1 + new_ratio 2) < 
        (initial_total * initial_ratio i) / (initial_ratio 0 + initial_ratio 1 + initial_ratio 2))) →
  new_total - initial_total = 75 :=
by sorry

end NUMINAMATH_CALUDE_min_new_respondents_l1712_171288


namespace NUMINAMATH_CALUDE_compare_quadratic_expressions_compare_fraction_sum_and_sum_l1712_171239

-- Statement 1
theorem compare_quadratic_expressions (m : ℝ) :
  3 * m^2 - m + 1 > 2 * m^2 + m - 3 := by
  sorry

-- Statement 2
theorem compare_fraction_sum_and_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 / b + b^2 / a ≥ a + b := by
  sorry

end NUMINAMATH_CALUDE_compare_quadratic_expressions_compare_fraction_sum_and_sum_l1712_171239


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_ratio_l1712_171282

/-- Given an arithmetic progression where the k-th, n-th, and p-th terms form three consecutive terms
    of a geometric progression, the common ratio of the geometric progression is (n-p)/(k-n). -/
theorem arithmetic_geometric_progression_ratio
  (a : ℕ → ℝ) -- The arithmetic progression
  (k n p : ℕ) -- Indices of the terms
  (d : ℝ) -- Common difference of the arithmetic progression
  (h1 : ∀ i, a (i + 1) = a i + d) -- Definition of arithmetic progression
  (h2 : ∃ q : ℝ, a n = a k * q ∧ a p = a n * q) -- Geometric progression condition
  : ∃ q : ℝ, q = (n - p) / (k - n) ∧ a n = a k * q ∧ a p = a n * q :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_ratio_l1712_171282


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1712_171203

theorem geometric_sequence_ratio_sum (k p r : ℝ) (h1 : p ≠ 1) (h2 : r ≠ 1) (h3 : p ≠ r) 
  (h4 : k ≠ 0) (h5 : k * p^2 - k * r^2 = 3 * (k * p - k * r)) : p + r = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1712_171203


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1712_171257

theorem average_speed_calculation (distance : ℝ) (time : ℝ) (average_speed : ℝ) :
  distance = 210 →
  time = 4.5 →
  average_speed = distance / time →
  average_speed = 140 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1712_171257


namespace NUMINAMATH_CALUDE_sum_of_series_l1712_171200

/-- The sum of the infinite series ∑(k=1 to ∞) k/3^k is equal to 3/4 -/
theorem sum_of_series : ∑' k, (k : ℝ) / 3^k = 3/4 := by sorry

end NUMINAMATH_CALUDE_sum_of_series_l1712_171200


namespace NUMINAMATH_CALUDE_probability_theorem_l1712_171258

/-- The probability of selecting exactly one high-quality item and one defective item
    from a set of 4 high-quality items and 1 defective item, when two items are randomly selected. -/
def probability_one_high_quality_one_defective : ℚ := 2 / 5

/-- The number of high-quality items -/
def num_high_quality : ℕ := 4

/-- The number of defective items -/
def num_defective : ℕ := 1

/-- The total number of items -/
def total_items : ℕ := num_high_quality + num_defective

/-- The number of items to be selected -/
def items_to_select : ℕ := 2

/-- Theorem stating that the probability of selecting exactly one high-quality item
    and one defective item is 2/5 -/
theorem probability_theorem :
  probability_one_high_quality_one_defective =
    (num_high_quality.choose 1 * num_defective.choose 1 : ℚ) /
    (total_items.choose items_to_select : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l1712_171258


namespace NUMINAMATH_CALUDE_election_result_l1712_171211

theorem election_result (total_votes : ℕ) (winner_votes first_opponent_votes second_opponent_votes third_opponent_votes : ℕ)
  (h1 : total_votes = 963)
  (h2 : winner_votes = 195)
  (h3 : first_opponent_votes = 142)
  (h4 : second_opponent_votes = 116)
  (h5 : third_opponent_votes = 90)
  (h6 : total_votes = winner_votes + first_opponent_votes + second_opponent_votes + third_opponent_votes) :
  winner_votes - first_opponent_votes = 53 := by
  sorry

end NUMINAMATH_CALUDE_election_result_l1712_171211


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l1712_171238

/-- The area of a parallelogram with given base and height -/
def parallelogramArea (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 34 cm and height 18 cm is 612 square centimeters -/
theorem parallelogram_area_example : parallelogramArea 34 18 = 612 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l1712_171238


namespace NUMINAMATH_CALUDE_third_group_frequency_l1712_171293

/-- Given a sample of data distributed into groups, calculate the frequency of the unspecified group --/
theorem third_group_frequency 
  (total : ℕ) 
  (num_groups : ℕ) 
  (group1 : ℕ) 
  (group2 : ℕ) 
  (group4 : ℕ) 
  (h1 : total = 40) 
  (h2 : num_groups = 4) 
  (h3 : group1 = 5) 
  (h4 : group2 = 12) 
  (h5 : group4 = 8) : 
  total - (group1 + group2 + group4) = 15 := by
  sorry

#check third_group_frequency

end NUMINAMATH_CALUDE_third_group_frequency_l1712_171293


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l1712_171249

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l1712_171249


namespace NUMINAMATH_CALUDE_computer_price_l1712_171214

theorem computer_price (new_price : ℝ) (price_increase : ℝ) (double_original : ℝ) 
  (h1 : price_increase = 0.3)
  (h2 : new_price = 377)
  (h3 : double_original = 580) : 
  ∃ (original_price : ℝ), 
    original_price * (1 + price_increase) = new_price ∧ 
    2 * original_price = double_original ∧
    original_price = 290 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_l1712_171214


namespace NUMINAMATH_CALUDE_triangle_area_ratio_specific_triangle_area_ratio_l1712_171265

/-- The ratio of the areas of two triangles with the same base and different heights -/
theorem triangle_area_ratio (base : ℝ) (height1 height2 : ℝ) :
  base > 0 → height1 > 0 → height2 > 0 →
  (base * height1 / 2) / (base * height2 / 2) = height1 / height2 := by
  sorry

/-- The specific ratio of triangle areas for the given problem -/
theorem specific_triangle_area_ratio :
  let base := 3
  let height1 := 6.02
  let height2 := 2
  (base * height1 / 2) / (base * height2 / 2) = 3.01 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_specific_triangle_area_ratio_l1712_171265


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l1712_171210

theorem floor_equation_solutions :
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    ⌊(2.018 : ℝ) * p.1⌋ + ⌊(5.13 : ℝ) * p.2⌋ = 24) (Finset.product (Finset.range 100) (Finset.range 100))).card ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l1712_171210


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l1712_171232

/-- Represents a parabola y^2 = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- A point on the parabola -/
structure PointOnParabola (para : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * para.p * x

theorem parabola_focus_distance (para : Parabola) 
  (A : PointOnParabola para) (h_x : A.x = 2) (h_dist : Real.sqrt ((A.x - para.p/2)^2 + A.y^2) = 6) :
  para.p = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l1712_171232


namespace NUMINAMATH_CALUDE_parallel_vectors_not_always_same_direction_l1712_171298

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (v w : V) : Prop := ∃ k : ℝ, v = k • w

theorem parallel_vectors_not_always_same_direction :
  ∃ (v w : V), parallel v w ∧ ¬(∃ k : ℝ, k > 0 ∧ v = k • w) :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_not_always_same_direction_l1712_171298


namespace NUMINAMATH_CALUDE_sarah_bottle_caps_l1712_171215

/-- The total number of bottle caps Sarah has at the end of the week -/
def total_bottle_caps (initial : ℕ) (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) : ℕ :=
  initial + day1 + day2 + day3

/-- Theorem stating that Sarah's total bottle caps at the end of the week
    is equal to her initial count plus all purchased bottle caps -/
theorem sarah_bottle_caps : 
  total_bottle_caps 450 175 95 220 = 940 := by
  sorry

end NUMINAMATH_CALUDE_sarah_bottle_caps_l1712_171215


namespace NUMINAMATH_CALUDE_polynomial_coefficient_equality_l1712_171245

theorem polynomial_coefficient_equality (k d m : ℚ) : 
  (∀ x : ℚ, (6 * x^3 - 4 * x^2 + 9/4) * (d * x^3 + k * x^2 + m) = 
   18 * x^6 - 17 * x^5 + 34 * x^4 - (36/4) * x^3 + (18/4) * x^2) → 
  k = -5/6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_equality_l1712_171245


namespace NUMINAMATH_CALUDE_bike_riders_count_l1712_171292

theorem bike_riders_count (total : ℕ) (difference : ℕ) :
  total = 676 →
  difference = 178 →
  ∃ (bikers hikers : ℕ),
    total = bikers + hikers ∧
    hikers = bikers + difference ∧
    bikers = 249 := by
  sorry

end NUMINAMATH_CALUDE_bike_riders_count_l1712_171292


namespace NUMINAMATH_CALUDE_min_c_over_d_l1712_171256

theorem min_c_over_d (x C D : ℝ) (hx : x > 0) (hC : C > 0) (hD : D > 0)
  (eq1 : x^2 + 1/x^2 = C) (eq2 : x + 1/x = D) : 
  ∀ y : ℝ, y > 0 → y^2 + 1/y^2 = C → y + 1/y = D → C / D ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_min_c_over_d_l1712_171256


namespace NUMINAMATH_CALUDE_soup_weight_after_four_days_l1712_171271

/-- The weight of soup remaining after four days of reduction -/
def remaining_soup_weight (initial_weight : ℝ) (day1_reduction day2_reduction day3_reduction day4_reduction : ℝ) : ℝ :=
  initial_weight * (1 - day1_reduction) * (1 - day2_reduction) * (1 - day3_reduction) * (1 - day4_reduction)

/-- Theorem stating the remaining weight of soup after four days -/
theorem soup_weight_after_four_days :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |remaining_soup_weight 80 0.40 0.35 0.55 0.50 - 7.02| < ε :=
sorry

end NUMINAMATH_CALUDE_soup_weight_after_four_days_l1712_171271


namespace NUMINAMATH_CALUDE_eight_percent_of_1200_is_96_l1712_171237

theorem eight_percent_of_1200_is_96 : 
  (8 / 100) * 1200 = 96 := by sorry

end NUMINAMATH_CALUDE_eight_percent_of_1200_is_96_l1712_171237


namespace NUMINAMATH_CALUDE_cookie_problem_l1712_171251

theorem cookie_problem (initial_cookies : ℕ) : 
  (initial_cookies : ℚ) * (1/4) * (1/2) = 8 → initial_cookies = 64 := by
  sorry

end NUMINAMATH_CALUDE_cookie_problem_l1712_171251


namespace NUMINAMATH_CALUDE_double_first_triple_second_row_l1712_171217

def N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 3]

theorem double_first_triple_second_row (A : Matrix (Fin 2) (Fin 2) ℝ) :
  N • A = !![2 * A 0 0, 2 * A 0 1; 3 * A 1 0, 3 * A 1 1] := by sorry

end NUMINAMATH_CALUDE_double_first_triple_second_row_l1712_171217


namespace NUMINAMATH_CALUDE_inequality_holds_iff_p_in_interval_l1712_171244

-- Define the inequality function
def inequality (p q : Real) : Prop :=
  (5 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q

-- State the theorem
theorem inequality_holds_iff_p_in_interval :
  ∀ p : Real, p ≥ 0 →
  (∀ q : Real, q > 0 → inequality p q) ↔
  p ∈ Set.Icc 0 (355/100) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_p_in_interval_l1712_171244


namespace NUMINAMATH_CALUDE_problem_solution_l1712_171204

-- Define the function f
def f (x : ℝ) := |2*x - 2| + |x + 2|

-- Define the theorem
theorem problem_solution :
  (∃ (S : Set ℝ), S = {x : ℝ | -3 ≤ x ∧ x ≤ 3/2} ∧ ∀ x, f x ≤ 6 - x ↔ x ∈ S) ∧
  (∃ (T : ℝ), T = 3 ∧ ∀ x, f x ≥ T) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 3 → 1/a + 1/b + 4/c ≥ 16/3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1712_171204


namespace NUMINAMATH_CALUDE_expected_red_pairs_l1712_171279

/-- The expected number of pairs of adjacent red cards in a circular arrangement -/
theorem expected_red_pairs (total_cards : ℕ) (red_cards : ℕ) (black_cards : ℕ)
  (h1 : total_cards = 60)
  (h2 : red_cards = 30)
  (h3 : black_cards = 30)
  (h4 : total_cards = red_cards + black_cards) :
  (red_cards : ℚ) * (red_cards - 1 : ℚ) / (total_cards - 1 : ℚ) = 870 / 59 := by
  sorry

end NUMINAMATH_CALUDE_expected_red_pairs_l1712_171279


namespace NUMINAMATH_CALUDE_pet_store_combinations_l1712_171235

def num_puppies : ℕ := 20
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 12
def num_rabbits : ℕ := 5
def num_customers : ℕ := 4

theorem pet_store_combinations : 
  num_puppies * num_kittens * num_hamsters * num_rabbits * Nat.factorial num_customers = 288000 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l1712_171235


namespace NUMINAMATH_CALUDE_professor_newton_students_l1712_171299

theorem professor_newton_students (total : ℕ) (male : ℕ) (female : ℕ) : 
  total % 4 = 2 →
  total % 5 = 1 →
  female = 15 →
  female > male →
  total = male + female →
  male = 11 := by
sorry

end NUMINAMATH_CALUDE_professor_newton_students_l1712_171299


namespace NUMINAMATH_CALUDE_quintic_polynomial_minimum_value_l1712_171284

/-- A quintic polynomial with real coefficients -/
def QuinticPolynomial (P : ℝ → ℝ) : Prop :=
  ∃ a b c d e f : ℝ, ∀ x, P x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f

/-- All complex roots of P have magnitude 1 -/
def AllRootsOnUnitCircle (P : ℂ → ℂ) : Prop :=
  ∀ z : ℂ, P z = 0 → Complex.abs z = 1

theorem quintic_polynomial_minimum_value (P : ℝ → ℝ) 
  (h_quintic : QuinticPolynomial P)
  (h_P0 : P 0 = 2)
  (h_P1 : P 1 = 3)
  (h_roots : AllRootsOnUnitCircle (fun z => P z.re)) :
  (∀ Q : ℝ → ℝ, QuinticPolynomial Q → Q 0 = 2 → Q 1 = 3 → 
    AllRootsOnUnitCircle (fun z => Q z.re) → P 2 ≤ Q 2) ∧ P 2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_quintic_polynomial_minimum_value_l1712_171284


namespace NUMINAMATH_CALUDE_kiley_ate_quarter_cheesecake_l1712_171291

/-- Represents the properties of a cheesecake and Kiley's consumption -/
structure CheesecakeConsumption where
  calories_per_slice : ℕ
  total_calories : ℕ
  slices_eaten : ℕ

/-- Calculates the percentage of cheesecake eaten -/
def percentage_eaten (c : CheesecakeConsumption) : ℚ :=
  (c.calories_per_slice * c.slices_eaten : ℚ) / c.total_calories * 100

/-- Theorem stating that Kiley ate 25% of the cheesecake -/
theorem kiley_ate_quarter_cheesecake :
  let c : CheesecakeConsumption := {
    calories_per_slice := 350,
    total_calories := 2800,
    slices_eaten := 2
  }
  percentage_eaten c = 25 := by
  sorry


end NUMINAMATH_CALUDE_kiley_ate_quarter_cheesecake_l1712_171291


namespace NUMINAMATH_CALUDE_one_instrument_one_sport_probability_l1712_171260

def total_people : ℕ := 1500

def instrument_ratio : ℚ := 3/7
def sport_ratio : ℚ := 5/14
def both_ratio : ℚ := 1/6
def multi_instrument_ratio : ℚ := 19/200  -- 9.5% = 19/200

def probability_one_instrument_one_sport (total : ℕ) (instrument : ℚ) (sport : ℚ) (both : ℚ) (multi : ℚ) : ℚ :=
  both

theorem one_instrument_one_sport_probability :
  probability_one_instrument_one_sport total_people instrument_ratio sport_ratio both_ratio multi_instrument_ratio = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_one_instrument_one_sport_probability_l1712_171260


namespace NUMINAMATH_CALUDE_special_function_bound_l1712_171206

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≥ 0 → y ≥ 0 → f x * f y ≤ x^2 * f (y/2) + y^2 * f (x/2)) ∧
  (∃ M : ℝ, M > 0 ∧ ∀ x, 0 ≤ x → x ≤ 1 → |f x| ≤ M)

/-- The main theorem stating that f(x) ≤ x^2 for all x ≥ 0 -/
theorem special_function_bound {f : ℝ → ℝ} (hf : SpecialFunction f) :
  ∀ x, x ≥ 0 → f x ≤ x^2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_bound_l1712_171206


namespace NUMINAMATH_CALUDE_angle_x_value_l1712_171296

/-- Given a configuration where AB and CD are straight lines, with specific angle measurements, prove that angle x equals 35 degrees. -/
theorem angle_x_value (AXB CYX XYB : ℝ) (h1 : AXB = 150) (h2 : CYX = 130) (h3 : XYB = 55) : ∃ x : ℝ, x = 35 := by
  sorry

end NUMINAMATH_CALUDE_angle_x_value_l1712_171296


namespace NUMINAMATH_CALUDE_students_above_115_l1712_171255

/-- Represents the score distribution of a math test -/
structure ScoreDistribution where
  mean : ℝ
  variance : ℝ
  normal : Bool

/-- Represents a class of students who took a math test -/
structure MathClass where
  size : ℕ
  scores : ScoreDistribution
  prob_95_to_105 : ℝ

/-- Calculates the number of students who scored above a given threshold -/
def students_above_threshold (c : MathClass) (threshold : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of students who scored above 115 in the given conditions -/
theorem students_above_115 (c : MathClass) 
  (h1 : c.size = 50)
  (h2 : c.scores.mean = 105)
  (h3 : c.scores.variance = 100)
  (h4 : c.scores.normal = true)
  (h5 : c.prob_95_to_105 = 0.32) :
  students_above_threshold c 115 = 9 :=
sorry

end NUMINAMATH_CALUDE_students_above_115_l1712_171255


namespace NUMINAMATH_CALUDE_solve_for_C_l1712_171297

theorem solve_for_C : ∃ C : ℝ, (4 * C - 5 = 23) ∧ (C = 7) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_C_l1712_171297


namespace NUMINAMATH_CALUDE_initial_population_approximation_l1712_171283

/-- The initial population of a town given its final population after a decade of growth. -/
def initial_population (final_population : ℕ) (growth_rate : ℚ) (years : ℕ) : ℚ :=
  final_population / (1 + growth_rate) ^ years

theorem initial_population_approximation :
  let final_population : ℕ := 297500
  let growth_rate : ℚ := 7 / 100
  let years : ℕ := 10
  ⌊initial_population final_population growth_rate years⌋ = 151195 := by
  sorry

end NUMINAMATH_CALUDE_initial_population_approximation_l1712_171283


namespace NUMINAMATH_CALUDE_no_solution_implies_m_leq_one_l1712_171226

theorem no_solution_implies_m_leq_one :
  (∀ x : ℝ, ¬(2*x - 1 > 1 ∧ x < m)) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_leq_one_l1712_171226


namespace NUMINAMATH_CALUDE_particular_solution_l1712_171250

noncomputable def y (x : ℝ) : ℝ := Real.sqrt (1 + 2 * Real.log ((1 + Real.exp x) / 2))

theorem particular_solution (x : ℝ) :
  (1 + Real.exp x) * y x * (deriv y x) = Real.exp x ∧ y 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_particular_solution_l1712_171250


namespace NUMINAMATH_CALUDE_jerry_one_way_time_15_minutes_l1712_171212

-- Define the distance to school in miles
def distance_to_school : ℝ := 4

-- Define Carson's speed in miles per hour
def carson_speed : ℝ := 8

-- Define the relationship between Jerry's round trip and Carson's one-way trip
axiom jerry_carson_time_relation : 
  ∀ (jerry_round_trip_time carson_one_way_time : ℝ), 
    jerry_round_trip_time = carson_one_way_time

-- Theorem: Jerry's one-way trip time to school is 15 minutes
theorem jerry_one_way_time_15_minutes : 
  ∃ (jerry_one_way_time : ℝ), 
    jerry_one_way_time = 15 := by sorry

end NUMINAMATH_CALUDE_jerry_one_way_time_15_minutes_l1712_171212


namespace NUMINAMATH_CALUDE_four_layer_pyramid_blocks_l1712_171269

/-- Calculates the total number of blocks in a pyramid with given number of layers -/
def pyramidBlocks (layers : ℕ) : ℕ := 
  let ratio := 3
  (ratio ^ layers - 1) / (ratio - 1)

/-- The number of layers in the pyramid -/
def numLayers : ℕ := 4

/-- Theorem stating that a four-layer pyramid with the given conditions has 40 blocks -/
theorem four_layer_pyramid_blocks : pyramidBlocks numLayers = 40 := by
  sorry

end NUMINAMATH_CALUDE_four_layer_pyramid_blocks_l1712_171269


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1712_171228

theorem complex_equation_solution (a b : ℝ) (h : (Complex.I + a) * (1 + Complex.I) = b * Complex.I) : 
  Complex.mk a b = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1712_171228


namespace NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l1712_171213

theorem equidistant_point_on_y_axis : 
  ∃ y : ℝ, y > 0 ∧ 
  ((-3 - 0)^2 + (0 - y)^2 = (-2 - 0)^2 + (5 - y)^2) ∧ 
  y = 2 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l1712_171213


namespace NUMINAMATH_CALUDE_dice_sum_probability_l1712_171259

/-- The number of sides on each die -/
def n : ℕ := 8

/-- The number of dice rolled -/
def k : ℕ := 10

/-- The target sum -/
def target_sum : ℕ := 50

/-- The number of ways to distribute the remaining sum after subtracting the minimum roll from each die -/
def num_ways : ℕ := Nat.choose 49 9

/-- The total number of possible outcomes when rolling k n-sided dice -/
def total_outcomes : ℕ := n ^ k

/-- The probability of obtaining the target sum -/
def probability : ℚ := num_ways / total_outcomes

theorem dice_sum_probability :
  probability = 818809200 / 1073741824 := by sorry

end NUMINAMATH_CALUDE_dice_sum_probability_l1712_171259


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_l1712_171272

def M : Set ℝ := {x | -x^2 - 5*x + 6 > 0}

def N : Set ℝ := {x | |x + 1| < 1}

theorem M_intersect_N_eq : M ∩ N = {x : ℝ | -2 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_l1712_171272


namespace NUMINAMATH_CALUDE_X_prob_implies_n_10_l1712_171242

/-- A random variable X taking values from 1 to n with equal probability -/
def X (n : ℕ) := Fin n

/-- The probability of X being less than 4 -/
def prob_X_less_than_4 (n : ℕ) : ℚ := (3 : ℚ) / n

/-- Theorem stating that if P(X < 4) = 0.3, then n = 10 -/
theorem X_prob_implies_n_10 (n : ℕ) (h : prob_X_less_than_4 n = (3 : ℚ) / 10) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_X_prob_implies_n_10_l1712_171242


namespace NUMINAMATH_CALUDE_red_packages_count_l1712_171274

def blue_packages : ℕ := 3
def beads_per_package : ℕ := 40
def total_beads : ℕ := 320

theorem red_packages_count :
  ∃ (red_packages : ℕ), 
    blue_packages * beads_per_package + red_packages * beads_per_package = total_beads ∧
    red_packages = 5 :=
by sorry

end NUMINAMATH_CALUDE_red_packages_count_l1712_171274


namespace NUMINAMATH_CALUDE_barry_head_stand_theorem_l1712_171270

/-- The number of turns Barry can take standing on his head during a 2-hour period -/
def barry_head_stand_turns : ℕ :=
  let head_stand_time : ℕ := 10  -- minutes
  let sit_time : ℕ := 5  -- minutes
  let total_period : ℕ := 2 * 60  -- 2 hours in minutes
  let time_per_turn : ℕ := head_stand_time + sit_time
  total_period / time_per_turn

theorem barry_head_stand_theorem :
  barry_head_stand_turns = 8 := by
  sorry

end NUMINAMATH_CALUDE_barry_head_stand_theorem_l1712_171270


namespace NUMINAMATH_CALUDE_researcher_reading_rate_l1712_171286

theorem researcher_reading_rate 
  (total_pages : ℕ) 
  (total_hours : ℕ) 
  (h1 : total_pages = 30000) 
  (h2 : total_hours = 150) : 
  (total_pages : ℚ) / total_hours = 200 := by
  sorry

end NUMINAMATH_CALUDE_researcher_reading_rate_l1712_171286


namespace NUMINAMATH_CALUDE_area_increase_bound_l1712_171248

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields for a convex polygon
  perimeter : ℝ
  area : ℝ

/-- The result of moving all sides of a polygon outward by distance h -/
def moveOutward (poly : ConvexPolygon) (h : ℝ) : ConvexPolygon := sorry

theorem area_increase_bound (poly : ConvexPolygon) (h : ℝ) (h_pos : h > 0) :
  (moveOutward poly h).area - poly.area > poly.perimeter * h + π * h^2 := by
  sorry

end NUMINAMATH_CALUDE_area_increase_bound_l1712_171248


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l1712_171208

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_eq : a + b + c = 1)
  (sum_prod_eq : a * b + a * c + b * c = -3)
  (prod_eq : a * b * c = 4) :
  a^3 + b^3 + c^3 = 1 := by
    sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l1712_171208


namespace NUMINAMATH_CALUDE_expansion_max_coefficient_l1712_171290

/-- The coefficient of x^3 in the expansion of (x - a/x)^5 is -5 -/
def coefficient_condition (a : ℝ) : Prop :=
  (5 : ℝ) * a = 5

/-- The maximum coefficient in the expansion of (x - a/x)^5 -/
def max_coefficient (a : ℝ) : ℕ :=
  Nat.max (Nat.choose 5 0)
    (Nat.max (Nat.choose 5 1)
      (Nat.max (Nat.choose 5 2)
        (Nat.max (Nat.choose 5 3)
          (Nat.max (Nat.choose 5 4)
            (Nat.choose 5 5)))))

theorem expansion_max_coefficient :
  ∀ a : ℝ, coefficient_condition a → max_coefficient a = 10 := by
  sorry

end NUMINAMATH_CALUDE_expansion_max_coefficient_l1712_171290


namespace NUMINAMATH_CALUDE_seashells_given_to_mike_l1712_171261

/-- Given that Joan initially found 79 seashells and now has 16 seashells,
    prove that the number of seashells she gave to Mike is 63. -/
theorem seashells_given_to_mike 
  (initial_seashells : ℕ) 
  (current_seashells : ℕ) 
  (h1 : initial_seashells = 79) 
  (h2 : current_seashells = 16) : 
  initial_seashells - current_seashells = 63 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_to_mike_l1712_171261


namespace NUMINAMATH_CALUDE_trig_sum_thirty_degrees_l1712_171225

theorem trig_sum_thirty_degrees :
  let tan30 := Real.sqrt 3 / 3
  let sin30 := 1 / 2
  let cos30 := Real.sqrt 3 / 2
  tan30 + 4 * sin30 + 2 * cos30 = (2 * Real.sqrt 3 + 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_thirty_degrees_l1712_171225


namespace NUMINAMATH_CALUDE_expression_evaluation_l1712_171207

theorem expression_evaluation : 
  let x := Real.sqrt (6000 - (3^3 : ℝ))
  let y := (105 / 21 : ℝ)^2
  abs (x * y - 1932.25) < 0.01 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1712_171207


namespace NUMINAMATH_CALUDE_race_head_start_l1712_171234

/-- Proves that Cristina gave Nicky a 12-second head start in a 100-meter race -/
theorem race_head_start (race_distance : ℝ) (cristina_speed nicky_speed : ℝ) (catch_up_time : ℝ) :
  race_distance = 100 →
  cristina_speed = 5 →
  nicky_speed = 3 →
  catch_up_time = 30 →
  (catch_up_time - (nicky_speed * catch_up_time) / cristina_speed) = 12 := by
  sorry

end NUMINAMATH_CALUDE_race_head_start_l1712_171234


namespace NUMINAMATH_CALUDE_beth_sells_80_coins_l1712_171262

/-- Calculates the number of coins Beth sells given her initial coins and a gift -/
def coins_sold (initial : ℕ) (gift : ℕ) : ℕ :=
  (initial + gift) / 2

/-- Proves that Beth sells 80 coins given her initial 125 coins and Carl's gift of 35 coins -/
theorem beth_sells_80_coins : coins_sold 125 35 = 80 := by
  sorry

end NUMINAMATH_CALUDE_beth_sells_80_coins_l1712_171262


namespace NUMINAMATH_CALUDE_inequality_solution_l1712_171241

theorem inequality_solution (x : ℝ) : 2 * (x - 3) < 8 ↔ x < 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1712_171241


namespace NUMINAMATH_CALUDE_tv_weather_forecast_is_random_l1712_171267

/-- Represents an event in probability theory -/
structure Event where
  (description : String)

/-- Classifies an event as random, certain, or impossible -/
inductive EventClass
  | Random
  | Certain
  | Impossible

/-- An event is random if it can lead to different outcomes, doesn't have a guaranteed outcome, and is feasible to occur -/
def is_random_event (e : Event) : Prop :=
  (∃ (outcome1 outcome2 : String), outcome1 ≠ outcome2) ∧
  ¬(∃ (guaranteed_outcome : String), true) ∧
  (∃ (possible_occurrence : Bool), possible_occurrence = true)

/-- The main theorem: Turning on the TV and watching the weather forecast is a random event -/
theorem tv_weather_forecast_is_random :
  let e : Event := { description := "turning on the TV and watching the weather forecast" }
  is_random_event e → EventClass.Random = EventClass.Random :=
by
  sorry

end NUMINAMATH_CALUDE_tv_weather_forecast_is_random_l1712_171267


namespace NUMINAMATH_CALUDE_crayon_division_l1712_171240

theorem crayon_division (total_crayons : ℕ) (num_people : ℕ) (crayons_per_person : ℕ) :
  total_crayons = 24 →
  num_people = 3 →
  total_crayons = num_people * crayons_per_person →
  crayons_per_person = 8 :=
by sorry

end NUMINAMATH_CALUDE_crayon_division_l1712_171240


namespace NUMINAMATH_CALUDE_smallest_p_for_multiple_of_ten_l1712_171221

theorem smallest_p_for_multiple_of_ten (n : ℕ) (h1 : n % 2 = 1) (h2 : n % 7 = 5) :
  ∃ p : ℕ, p > 0 ∧ (n + p) % 10 = 0 ∧ ∀ q : ℕ, 0 < q → (n + q) % 10 = 0 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_p_for_multiple_of_ten_l1712_171221


namespace NUMINAMATH_CALUDE_unique_two_digit_number_mod_4_17_l1712_171247

theorem unique_two_digit_number_mod_4_17 : 
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 4 = 1 ∧ n % 17 = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_mod_4_17_l1712_171247


namespace NUMINAMATH_CALUDE_science_fair_participants_l1712_171273

/-- The number of unique students participating in the Science Fair --/
def unique_students (robotics astronomy chemistry all_three : ℕ) : ℕ :=
  robotics + astronomy + chemistry - 2 * all_three

/-- Theorem stating the number of unique students in the Science Fair --/
theorem science_fair_participants : unique_students 15 10 12 2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_science_fair_participants_l1712_171273


namespace NUMINAMATH_CALUDE_friends_coming_over_l1712_171275

theorem friends_coming_over (sandwiches_per_friend : ℕ) (total_sandwiches : ℕ) 
  (h1 : sandwiches_per_friend = 3) 
  (h2 : total_sandwiches = 12) : 
  total_sandwiches / sandwiches_per_friend = 4 :=
by sorry

end NUMINAMATH_CALUDE_friends_coming_over_l1712_171275


namespace NUMINAMATH_CALUDE_four_numbers_problem_l1712_171236

theorem four_numbers_problem (a b c d : ℝ) : 
  a + b + c + d = 45 ∧ 
  a + 2 = b - 2 ∧ 
  a + 2 = 2 * c ∧ 
  a + 2 = d / 2 → 
  a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20 := by
sorry

end NUMINAMATH_CALUDE_four_numbers_problem_l1712_171236


namespace NUMINAMATH_CALUDE_max_sum_of_sides_l1712_171254

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x ^ 2

theorem max_sum_of_sides (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a = Real.sqrt 3 →
  f A = 1 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b + c ≤ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_sides_l1712_171254


namespace NUMINAMATH_CALUDE_find_x_l1712_171268

theorem find_x (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) 
  (h : (2 * a) ^ (2 * b) = a ^ b * x ^ (2 * b)) : x = 2 * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1712_171268


namespace NUMINAMATH_CALUDE_condition_p_necessary_not_sufficient_for_q_l1712_171263

theorem condition_p_necessary_not_sufficient_for_q :
  (∀ x y : ℝ, Real.sqrt x > Real.sqrt y → x > y) ∧
  (∃ x y : ℝ, x > y ∧ ¬(Real.sqrt x > Real.sqrt y)) := by
  sorry

end NUMINAMATH_CALUDE_condition_p_necessary_not_sufficient_for_q_l1712_171263


namespace NUMINAMATH_CALUDE_october_birthdays_percentage_l1712_171287

theorem october_birthdays_percentage (total : ℕ) (october_births : ℕ) : 
  total = 120 → october_births = 18 → (october_births : ℚ) / total * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_october_birthdays_percentage_l1712_171287


namespace NUMINAMATH_CALUDE_rectangle_length_l1712_171201

/-- The length of a rectangle with given area and width -/
theorem rectangle_length (area width : ℝ) (h_area : area = 36.48) (h_width : width = 6.08) :
  area / width = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l1712_171201


namespace NUMINAMATH_CALUDE_double_discount_price_l1712_171218

-- Define the original price
def original_price : ℝ := 33.78

-- Define the discount rate
def discount_rate : ℝ := 0.25

-- Define the function to apply a discount
def apply_discount (price : ℝ) : ℝ := price * (1 - discount_rate)

-- Theorem statement
theorem double_discount_price :
  apply_discount (apply_discount original_price) = 19.00125 := by
  sorry

end NUMINAMATH_CALUDE_double_discount_price_l1712_171218


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1712_171229

/-- Given a quadratic equation ax² + bx + c = 0, returns the coefficient of x² (a) -/
def quadratic_coefficient (a b c : ℚ) : ℚ := a

/-- Given a quadratic equation ax² + bx + c = 0, returns the constant term (c) -/
def constant_term (a b c : ℚ) : ℚ := c

theorem quadratic_equation_coefficients :
  let a : ℚ := 3
  let b : ℚ := -6
  let c : ℚ := -7
  quadratic_coefficient a b c = 3 ∧ constant_term a b c = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1712_171229


namespace NUMINAMATH_CALUDE_number_count_in_average_calculation_l1712_171278

/-- Given an initial average, an incorrectly read number, and the correct average,
    prove the number of numbers in the original calculation. -/
theorem number_count_in_average_calculation
  (initial_avg : ℚ)
  (incorrect_num : ℚ)
  (correct_num : ℚ)
  (correct_avg : ℚ)
  (h1 : initial_avg = 19)
  (h2 : incorrect_num = 26)
  (h3 : correct_num = 76)
  (h4 : correct_avg = 24) :
  ∃ (n : ℕ) (S : ℚ),
    S + incorrect_num = initial_avg * n ∧
    S + correct_num = correct_avg * n ∧
    n = 10 :=
sorry

end NUMINAMATH_CALUDE_number_count_in_average_calculation_l1712_171278


namespace NUMINAMATH_CALUDE_prob_k_white_balls_correct_l1712_171285

/-- The probability of drawing exactly k white balls from an urn containing n white and n black balls,
    when drawing n balls in total. -/
def prob_k_white_balls (n k : ℕ) : ℚ :=
  (Nat.choose n k)^2 / Nat.choose (2*n) n

/-- Theorem stating that the probability of drawing exactly k white balls from an urn
    containing n white balls and n black balls, when drawing n balls in total,
    is equal to (n choose k)^2 / (2n choose n). -/
theorem prob_k_white_balls_correct (n k : ℕ) (h : k ≤ n) :
  prob_k_white_balls n k = (Nat.choose n k)^2 / Nat.choose (2*n) n :=
by sorry

end NUMINAMATH_CALUDE_prob_k_white_balls_correct_l1712_171285
