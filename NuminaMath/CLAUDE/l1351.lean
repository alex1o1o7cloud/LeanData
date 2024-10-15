import Mathlib

namespace NUMINAMATH_CALUDE_set_intersection_theorem_l1351_135107

theorem set_intersection_theorem (m : ℝ) : 
  let A := {x : ℝ | x ≥ 3}
  let B := {x : ℝ | x < m}
  (A ∪ B = Set.univ) ∧ (A ∩ B = ∅) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l1351_135107


namespace NUMINAMATH_CALUDE_marcel_potatoes_l1351_135104

theorem marcel_potatoes (marcel_corn : ℕ) (dale_potatoes : ℕ) (total_vegetables : ℕ)
  (h1 : marcel_corn = 10)
  (h2 : total_vegetables = 27)
  (h3 : dale_potatoes = 8) :
  total_vegetables - (marcel_corn + marcel_corn / 2 + dale_potatoes) = 4 := by
sorry

end NUMINAMATH_CALUDE_marcel_potatoes_l1351_135104


namespace NUMINAMATH_CALUDE_difference_of_squares_75_35_l1351_135165

theorem difference_of_squares_75_35 : 75^2 - 35^2 = 4400 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_75_35_l1351_135165


namespace NUMINAMATH_CALUDE_garden_area_ratio_l1351_135167

/-- Given a rectangular garden with initial length and width, and increase percentages for both dimensions, prove that the ratio of the original area to the redesigned area is 1/3. -/
theorem garden_area_ratio (initial_length initial_width : ℝ) 
  (length_increase width_increase : ℝ) : 
  initial_length = 10 →
  initial_width = 5 →
  length_increase = 0.5 →
  width_increase = 1 →
  (initial_length * initial_width) / 
  ((initial_length * (1 + length_increase)) * (initial_width * (1 + width_increase))) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_garden_area_ratio_l1351_135167


namespace NUMINAMATH_CALUDE_exponent_calculation_l1351_135162

theorem exponent_calculation : (64 : ℝ)^(1/4) * (16 : ℝ)^(3/8) = 8 := by
  have h1 : (64 : ℝ) = 2^6 := by sorry
  have h2 : (16 : ℝ) = 2^4 := by sorry
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l1351_135162


namespace NUMINAMATH_CALUDE_coordinate_sum_of_point_d_l1351_135197

/-- Given a point C at (0, 0) and a point D on the line y = 5,
    if the slope of segment CD is 3/4,
    then the sum of the x- and y-coordinates of point D is 35/3. -/
theorem coordinate_sum_of_point_d (D : ℝ × ℝ) : 
  D.2 = 5 →                  -- D is on the line y = 5
  (D.2 - 0) / (D.1 - 0) = 3/4 →  -- slope of CD is 3/4
  D.1 + D.2 = 35/3 := by
sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_point_d_l1351_135197


namespace NUMINAMATH_CALUDE_alternating_draw_probability_l1351_135115

/-- The number of white balls in the box -/
def num_white : ℕ := 6

/-- The number of black balls in the box -/
def num_black : ℕ := 6

/-- The total number of balls in the box -/
def total_balls : ℕ := num_white + num_black

/-- The number of ways to arrange all balls -/
def total_arrangements : ℕ := Nat.choose total_balls num_white

/-- The number of alternating color sequences -/
def alternating_sequences : ℕ := 2

/-- The probability of drawing balls with alternating colors -/
def alternating_probability : ℚ := alternating_sequences / total_arrangements

theorem alternating_draw_probability : alternating_probability = 1 / 462 := by
  sorry

end NUMINAMATH_CALUDE_alternating_draw_probability_l1351_135115


namespace NUMINAMATH_CALUDE_unique_perfect_square_p_l1351_135174

/-- The polynomial p(x) = x^4 + 6x^3 + 11x^2 + 3x + 31 -/
def p (x : ℤ) : ℤ := x^4 + 6*x^3 + 11*x^2 + 3*x + 31

/-- A function that checks if a given integer is a perfect square -/
def is_perfect_square (n : ℤ) : Prop :=
  ∃ m : ℤ, n = m^2

/-- Theorem stating that there exists exactly one integer x for which p(x) is a perfect square -/
theorem unique_perfect_square_p :
  ∃! x : ℤ, is_perfect_square (p x) :=
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_p_l1351_135174


namespace NUMINAMATH_CALUDE_total_bread_served_l1351_135188

-- Define the quantities of bread served
def wheat_bread : ℚ := 1.25
def white_bread : ℚ := 3/4
def rye_bread : ℚ := 0.6
def multigrain_bread : ℚ := 7/10

-- Theorem to prove
theorem total_bread_served :
  wheat_bread + white_bread + rye_bread + multigrain_bread = 3 + 3/10 := by
  sorry

end NUMINAMATH_CALUDE_total_bread_served_l1351_135188


namespace NUMINAMATH_CALUDE_specific_rectangle_triangles_l1351_135142

/-- Represents a rectangle with a grid and diagonals -/
structure GridRectangle where
  width : ℕ
  height : ℕ
  vertical_spacing : ℕ
  horizontal_spacing : ℕ

/-- Counts the number of triangles in a GridRectangle -/
def count_triangles (rect : GridRectangle) : ℕ :=
  sorry

/-- The main theorem stating the number of triangles in the specific configuration -/
theorem specific_rectangle_triangles :
  let rect : GridRectangle := {
    width := 40,
    height := 10,
    vertical_spacing := 10,
    horizontal_spacing := 5
  }
  count_triangles rect = 74 := by
  sorry

end NUMINAMATH_CALUDE_specific_rectangle_triangles_l1351_135142


namespace NUMINAMATH_CALUDE_ladder_problem_l1351_135130

theorem ladder_problem (h1 h2 l : Real) 
  (hyp1 : h1 = 12)
  (hyp2 : h2 = 9)
  (hyp3 : l = 15) :
  Real.sqrt (l^2 - h1^2) + Real.sqrt (l^2 - h2^2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l1351_135130


namespace NUMINAMATH_CALUDE_austin_work_hours_on_monday_l1351_135101

/-- Proves that Austin works 2 hours on Mondays to earn enough for a $180 bicycle in 6 weeks -/
theorem austin_work_hours_on_monday : 
  let hourly_rate : ℕ := 5
  let bicycle_cost : ℕ := 180
  let weeks : ℕ := 6
  let wednesday_hours : ℕ := 1
  let friday_hours : ℕ := 3
  ∃ (monday_hours : ℕ), 
    weeks * (hourly_rate * (monday_hours + wednesday_hours + friday_hours)) = bicycle_cost ∧ 
    monday_hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_austin_work_hours_on_monday_l1351_135101


namespace NUMINAMATH_CALUDE_six_people_arrangement_l1351_135134

def arrangement_count (n : ℕ) : ℕ := 
  (n.choose 2) * ((n-2).choose 2) * ((n-4).choose 2)

theorem six_people_arrangement : arrangement_count 6 = 90 := by
  sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l1351_135134


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1351_135112

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℤ),
  X^4 + X^2 = (X^2 + 3*X + 2) * q + r ∧ 
  r.degree < (X^2 + 3*X + 2).degree ∧ 
  r = -18*X - 16 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1351_135112


namespace NUMINAMATH_CALUDE_basketball_evaluation_theorem_l1351_135111

/-- The number of rounds in the basketball evaluation -/
def num_rounds : ℕ := 3

/-- The number of shots per round -/
def shots_per_round : ℕ := 2

/-- The probability of player A making a shot -/
def prob_make_shot : ℚ := 2/3

/-- The probability of passing a single round -/
def prob_pass_round : ℚ := 1 - (1 - prob_make_shot) ^ shots_per_round

/-- The expected number of rounds player A will pass -/
def expected_passed_rounds : ℚ := num_rounds * prob_pass_round

theorem basketball_evaluation_theorem :
  expected_passed_rounds = 8/3 := by sorry

end NUMINAMATH_CALUDE_basketball_evaluation_theorem_l1351_135111


namespace NUMINAMATH_CALUDE_sequence_sum_l1351_135159

theorem sequence_sum (a b : ℕ) : 
  let seq : List ℕ := [a, b, a + b, a + 2*b, 2*a + 3*b, a + 2*b + 7, a + 2*b + 14, 2*a + 4*b + 21, 3*a + 6*b + 35]
  2*a + 3*b = 7 → 
  3*a + 6*b + 35 = 47 → 
  seq.sum = 122 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l1351_135159


namespace NUMINAMATH_CALUDE_hexagon_area_sum_l1351_135157

-- Define the hexagon structure
structure Hexagon :=
  (sideLength : ℝ)
  (numSegments : ℕ)

-- Define the theorem
theorem hexagon_area_sum (h : Hexagon) (a b : ℕ) : 
  h.sideLength = 3 ∧ h.numSegments = 12 →
  ∃ (area : ℝ), area = a * Real.sqrt b ∧ a + b = 30 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_area_sum_l1351_135157


namespace NUMINAMATH_CALUDE_john_plays_three_times_a_month_l1351_135135

/-- The number of times John plays paintball in a month -/
def plays_per_month : ℕ := sorry

/-- The number of boxes John buys each time he plays -/
def boxes_per_play : ℕ := 3

/-- The cost of each box of paintballs in dollars -/
def cost_per_box : ℕ := 25

/-- The total amount John spends on paintballs per month in dollars -/
def total_spent_per_month : ℕ := 225

/-- Theorem stating that John plays paintball 3 times a month -/
theorem john_plays_three_times_a_month : 
  plays_per_month = 3 ∧ 
  plays_per_month * boxes_per_play * cost_per_box = total_spent_per_month := by
  sorry

end NUMINAMATH_CALUDE_john_plays_three_times_a_month_l1351_135135


namespace NUMINAMATH_CALUDE_reading_homework_pages_isabel_homework_l1351_135149

theorem reading_homework_pages (math_pages : ℕ) (problems_per_page : ℕ) (total_problems : ℕ) : ℕ :=
  let reading_pages := (total_problems - math_pages * problems_per_page) / problems_per_page
  reading_pages

theorem isabel_homework :
  reading_homework_pages 2 5 30 = 4 := by
  sorry

end NUMINAMATH_CALUDE_reading_homework_pages_isabel_homework_l1351_135149


namespace NUMINAMATH_CALUDE_distance_traveled_l1351_135102

/-- Proves that given a speed of 20 km/hr and a time of 2.5 hours, the distance traveled is 50 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 20) (h2 : time = 2.5) :
  speed * time = 50 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l1351_135102


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_13_squared_plus_84_squared_l1351_135124

theorem largest_prime_divisor_of_13_squared_plus_84_squared : 
  (Nat.factors (13^2 + 84^2)).maximum = some 17 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_13_squared_plus_84_squared_l1351_135124


namespace NUMINAMATH_CALUDE_height_difference_l1351_135100

/-- Heights of people in centimeters -/
structure Heights where
  janet : ℝ
  charlene : ℝ
  pablo : ℝ
  ruby : ℝ

/-- Problem conditions -/
def problem_conditions (h : Heights) : Prop :=
  h.janet = 62 ∧
  h.charlene = 2 * h.janet ∧
  h.pablo = h.charlene + 70 ∧
  h.ruby = 192 ∧
  h.pablo > h.ruby

/-- Theorem stating the height difference between Pablo and Ruby -/
theorem height_difference (h : Heights) 
  (hc : problem_conditions h) : h.pablo - h.ruby = 2 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l1351_135100


namespace NUMINAMATH_CALUDE_ml_to_litre_fraction_l1351_135185

theorem ml_to_litre_fraction (ml_per_litre : ℝ) (volume_ml : ℝ) :
  ml_per_litre = 1000 →
  volume_ml = 30 →
  volume_ml / ml_per_litre = 0.03 := by
sorry

end NUMINAMATH_CALUDE_ml_to_litre_fraction_l1351_135185


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_13_greatest_two_digit_multiple_of_13_is_91_l1351_135164

theorem greatest_two_digit_multiple_of_13 : ℕ → Prop :=
  fun n =>
    (n ≤ 99) ∧ 
    (n ≥ 10) ∧ 
    (∃ k : ℕ, n = 13 * k) ∧ 
    (∀ m : ℕ, m ≤ 99 ∧ m ≥ 10 ∧ (∃ j : ℕ, m = 13 * j) → m ≤ n) →
    n = 91

-- The proof would go here, but we'll skip it as requested
theorem greatest_two_digit_multiple_of_13_is_91 : greatest_two_digit_multiple_of_13 91 := by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_13_greatest_two_digit_multiple_of_13_is_91_l1351_135164


namespace NUMINAMATH_CALUDE_min_value_sum_product_equality_condition_l1351_135114

theorem min_value_sum_product (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b + d) + 1 / (a + c + d) + 1 / (b + c + d)) ≥ 4 :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b + d) + 1 / (a + c + d) + 1 / (b + c + d)) = 4 ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_product_equality_condition_l1351_135114


namespace NUMINAMATH_CALUDE_bushes_for_zucchinis_l1351_135194

/-- The number of containers of blueberries yielded by one bush -/
def containers_per_bush : ℕ := 10

/-- The number of containers of blueberries needed to trade for one zucchini -/
def containers_per_zucchini : ℕ := 3

/-- The number of zucchinis Natalie wants to obtain -/
def target_zucchinis : ℕ := 60

/-- The number of bushes needed to obtain the target number of zucchinis -/
def bushes_needed : ℕ := 18

theorem bushes_for_zucchinis :
  bushes_needed * containers_per_bush = target_zucchinis * containers_per_zucchini :=
by sorry

end NUMINAMATH_CALUDE_bushes_for_zucchinis_l1351_135194


namespace NUMINAMATH_CALUDE_fifth_road_length_l1351_135172

/-- Represents a road network with four cities and five roads -/
structure RoadNetwork where
  road1 : ℕ
  road2 : ℕ
  road3 : ℕ
  road4 : ℕ
  road5 : ℕ

/-- The given road network satisfies the triangle inequality -/
def satisfiesTriangleInequality (rn : RoadNetwork) : Prop :=
  rn.road5 < rn.road1 + rn.road2 ∧
  rn.road5 + rn.road3 > rn.road4

/-- Theorem: Given the specific road lengths, the fifth road must be 17 km long -/
theorem fifth_road_length (rn : RoadNetwork) 
  (h1 : rn.road1 = 10)
  (h2 : rn.road2 = 8)
  (h3 : rn.road3 = 5)
  (h4 : rn.road4 = 21)
  (h5 : satisfiesTriangleInequality rn) :
  rn.road5 = 17 := by
  sorry


end NUMINAMATH_CALUDE_fifth_road_length_l1351_135172


namespace NUMINAMATH_CALUDE_original_price_of_meat_pack_original_price_is_40_l1351_135195

/-- The original price of a 4 pack of fancy, sliced meat, given rush delivery conditions -/
theorem original_price_of_meat_pack : ℝ :=
  let rush_delivery_factor : ℝ := 1.3
  let price_with_rush : ℝ := 13
  let pack_size : ℕ := 4
  let single_meat_price : ℝ := price_with_rush / rush_delivery_factor
  pack_size * single_meat_price

/-- Proof that the original price of the 4 pack is $40 -/
theorem original_price_is_40 : original_price_of_meat_pack = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_price_of_meat_pack_original_price_is_40_l1351_135195


namespace NUMINAMATH_CALUDE_gcd_product_is_square_l1351_135138

theorem gcd_product_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, (Nat.gcd x y).gcd z * x * y * z = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_product_is_square_l1351_135138


namespace NUMINAMATH_CALUDE_pizza_slices_left_l1351_135170

theorem pizza_slices_left (total_slices : ℕ) (people : ℕ) (slices_per_person : ℕ) :
  total_slices = 16 →
  people = 6 →
  slices_per_person = 2 →
  total_slices - (people * slices_per_person) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_left_l1351_135170


namespace NUMINAMATH_CALUDE_exam_failure_percentage_l1351_135113

theorem exam_failure_percentage 
  (pass_english : ℝ) 
  (pass_math : ℝ) 
  (pass_either : ℝ) 
  (h1 : pass_english = 0.63) 
  (h2 : pass_math = 0.65) 
  (h3 : pass_either = 0.55) : 
  1 - pass_either = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_exam_failure_percentage_l1351_135113


namespace NUMINAMATH_CALUDE_min_b_minus_a_l1351_135154

noncomputable def f (a b x : ℝ) : ℝ := (2 * x^2 + x) * Real.log x - (2 * a + 1) * x^2 - (a + 1) * x + b

theorem min_b_minus_a (a b : ℝ) :
  (∀ x > 0, f a b x ≥ 0) → 
  ∃ m, m = 3/4 + Real.log 2 ∧ b - a ≥ m ∧ ∀ ε > 0, ∃ a' b', b' - a' < m + ε :=
sorry

end NUMINAMATH_CALUDE_min_b_minus_a_l1351_135154


namespace NUMINAMATH_CALUDE_m_eq_neg_one_iff_pure_imaginary_l1351_135179

/-- A complex number z is defined as m² - 1 + (m² - 3m + 2)i, where m is a real number and i is the imaginary unit. -/
def z (m : ℝ) : ℂ := (m^2 - 1) + (m^2 - 3*m + 2)*Complex.I

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Theorem: m = -1 is both sufficient and necessary for z to be a pure imaginary number. -/
theorem m_eq_neg_one_iff_pure_imaginary (m : ℝ) :
  m = -1 ↔ is_pure_imaginary (z m) := by sorry

end NUMINAMATH_CALUDE_m_eq_neg_one_iff_pure_imaginary_l1351_135179


namespace NUMINAMATH_CALUDE_ratio_problem_l1351_135171

theorem ratio_problem (a b : ℝ) (h1 : a / b = 5) (h2 : a = 65) : b = 13 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1351_135171


namespace NUMINAMATH_CALUDE_fourth_berry_count_l1351_135106

/-- A sequence of berry counts where the difference between consecutive terms increases by 2 -/
def BerrySequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) = (a (n + 1) - a n) + 2

theorem fourth_berry_count
  (a : ℕ → ℕ)
  (seq : BerrySequence a)
  (first : a 0 = 3)
  (second : a 1 = 4)
  (third : a 2 = 7)
  (fifth : a 4 = 19) :
  a 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_fourth_berry_count_l1351_135106


namespace NUMINAMATH_CALUDE_sum_of_ac_l1351_135120

theorem sum_of_ac (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 48) 
  (h2 : b + d = 6) : 
  a + c = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ac_l1351_135120


namespace NUMINAMATH_CALUDE_race_time_difference_l1351_135108

/-- Represents the time difference in minutes between two runners finishing a race -/
def timeDifference (malcolmSpeed Joshua : ℝ) (raceDistance : ℝ) : ℝ :=
  raceDistance * Joshua - raceDistance * malcolmSpeed

theorem race_time_difference :
  let malcolmSpeed := 6
  let Joshua := 8
  let raceDistance := 10
  timeDifference malcolmSpeed Joshua raceDistance = 20 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l1351_135108


namespace NUMINAMATH_CALUDE_a_max_value_l1351_135147

def a (n : ℕ+) : ℚ := n / (n^2 + 90)

theorem a_max_value : ∀ n : ℕ+, a n ≤ 1/19 := by
  sorry

end NUMINAMATH_CALUDE_a_max_value_l1351_135147


namespace NUMINAMATH_CALUDE_geometric_sequence_terms_l1351_135143

theorem geometric_sequence_terms (a : ℕ → ℝ) (n : ℕ) (S_n : ℝ) : 
  (∀ k, a (k + 1) / a k = a 2 / a 1) →  -- geometric sequence condition
  a 1 + a n = 82 →
  a 3 * a (n - 2) = 81 →
  S_n = 121 →
  (∀ k, S_k = (a 1 * (1 - (a 2 / a 1)^k)) / (1 - (a 2 / a 1))) →  -- sum formula for geometric sequence
  n = 5 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_terms_l1351_135143


namespace NUMINAMATH_CALUDE_decreasing_function_property_l1351_135181

/-- A function f is decreasing on ℝ if for any x₁ < x₂, we have f(x₁) > f(x₂) -/
def DecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

theorem decreasing_function_property (f : ℝ → ℝ) (a : ℝ) 
  (h : DecreasingOn f) : f (a^2 + 1) < f a := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_property_l1351_135181


namespace NUMINAMATH_CALUDE_initial_number_proof_l1351_135110

theorem initial_number_proof : 
  ∃ x : ℝ, (3 * (2 * x + 9) = 81) ∧ (x = 9) := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l1351_135110


namespace NUMINAMATH_CALUDE_twentieth_stage_toothpicks_l1351_135166

/-- Number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ := 3 + 3 * (n - 1)

/-- The 20th stage of the toothpick pattern has 60 toothpicks -/
theorem twentieth_stage_toothpicks : toothpicks 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_stage_toothpicks_l1351_135166


namespace NUMINAMATH_CALUDE_xiaomin_house_coordinates_l1351_135186

/-- Represents a 2D point with x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- The school's position -/
def school : Point := { x := 0, y := 0 }

/-- Xiaomin's house position relative to the school -/
def house_relative : Point := { x := 200, y := -150 }

/-- Theorem stating that Xiaomin's house coordinates are (200, -150) -/
theorem xiaomin_house_coordinates :
  ∃ (p : Point), p.x = school.x + house_relative.x ∧ p.y = school.y + house_relative.y ∧ 
  p.x = 200 ∧ p.y = -150 := by
  sorry

end NUMINAMATH_CALUDE_xiaomin_house_coordinates_l1351_135186


namespace NUMINAMATH_CALUDE_solution_set_implies_k_value_l1351_135121

theorem solution_set_implies_k_value (k : ℝ) : 
  (∀ x : ℝ, |k * x - 4| ≤ 2 ↔ 1 ≤ x ∧ x ≤ 3) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_k_value_l1351_135121


namespace NUMINAMATH_CALUDE_no_valid_grid_l1351_135156

/-- Represents a grid of stars -/
def StarGrid := Fin 10 → Fin 10 → Bool

/-- Counts the number of stars in a 2x2 square starting at (i, j) -/
def countStars2x2 (grid : StarGrid) (i j : Fin 10) : Nat :=
  (grid i j).toNat + (grid i (j+1)).toNat + (grid (i+1) j).toNat + (grid (i+1) (j+1)).toNat

/-- Counts the number of stars in a 3x1 rectangle starting at (i, j) -/
def countStars3x1 (grid : StarGrid) (i j : Fin 10) : Nat :=
  (grid i j).toNat + (grid i (j+1)).toNat + (grid i (j+2)).toNat

/-- Checks if the grid satisfies the conditions -/
def isValidGrid (grid : StarGrid) : Prop :=
  (∀ i j, i < 9 ∧ j < 9 → countStars2x2 grid i j = 2) ∧
  (∀ i j, j < 8 → countStars3x1 grid i j = 1)

theorem no_valid_grid : ¬∃ (grid : StarGrid), isValidGrid grid := by
  sorry

end NUMINAMATH_CALUDE_no_valid_grid_l1351_135156


namespace NUMINAMATH_CALUDE_f_properties_l1351_135198

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 1/4 + Real.log x / Real.log 4
  else 2^(-x) - 1/4

theorem f_properties :
  (∀ x, f x ≥ 1/4) ∧
  (∀ x, f x = 3/4 ↔ x = 0 ∨ x = 2) := by
sorry

end NUMINAMATH_CALUDE_f_properties_l1351_135198


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l1351_135109

/-- A function with specific symmetry and derivative properties -/
class SpecialFunction (f : ℝ → ℝ) where
  symmetric : ∀ x, f x = f (-2 - x)
  derivative_property : ∀ x, x < -1 → (x + 1) * (f x + (x + 1) * (deriv f) x) < 0

/-- The solution set of the inequality xf(x-1) > f(0) -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | x * f (x - 1) > f 0}

/-- The theorem stating the solution set of the inequality -/
theorem solution_set_is_open_interval
  (f : ℝ → ℝ) [SpecialFunction f] :
  SolutionSet f = Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l1351_135109


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l1351_135119

theorem floor_negative_seven_fourths :
  ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l1351_135119


namespace NUMINAMATH_CALUDE_right_triangle_stable_l1351_135184

/-- A shape is considered stable if it maintains its form without deformation under normal conditions. -/
def Stable (shape : Type) : Prop := sorry

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
def Parallelogram : Type := sorry

/-- A square is a quadrilateral with four equal sides and four right angles. -/
def Square : Type := sorry

/-- A rectangle is a quadrilateral with four right angles. -/
def Rectangle : Type := sorry

/-- A right triangle is a triangle with one right angle. -/
def RightTriangle : Type := sorry

/-- Theorem stating that among the given shapes, only the right triangle is inherently stable. -/
theorem right_triangle_stable :
  ¬Stable Parallelogram ∧
  ¬Stable Square ∧
  ¬Stable Rectangle ∧
  Stable RightTriangle :=
sorry

end NUMINAMATH_CALUDE_right_triangle_stable_l1351_135184


namespace NUMINAMATH_CALUDE_special_triangle_sides_l1351_135116

/-- Represents a triangle with known height, base, and sum of two sides --/
structure SpecialTriangle where
  height : ℝ
  base : ℝ
  sum_of_sides : ℝ

/-- The two unknown sides of the triangle --/
structure TriangleSides where
  side1 : ℝ
  side2 : ℝ

/-- Theorem stating that for a triangle with height 24, base 28, and sum of two sides 56,
    the lengths of these two sides are 26 and 30 --/
theorem special_triangle_sides (t : SpecialTriangle) 
    (h1 : t.height = 24)
    (h2 : t.base = 28)
    (h3 : t.sum_of_sides = 56) :
  ∃ (s : TriangleSides), s.side1 = 26 ∧ s.side2 = 30 ∧ s.side1 + s.side2 = t.sum_of_sides :=
by
  sorry


end NUMINAMATH_CALUDE_special_triangle_sides_l1351_135116


namespace NUMINAMATH_CALUDE_prism_volume_l1351_135169

/-- The volume of a right rectangular prism with face areas 40, 50, and 100 square centimeters -/
theorem prism_volume (x y z : ℝ) (hxy : x * y = 40) (hxz : x * z = 50) (hyz : y * z = 100) :
  x * y * z = 100 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1351_135169


namespace NUMINAMATH_CALUDE_yellow_balls_count_l1351_135161

/-- Given a bag with 50 balls of two colors, if the frequency of picking one color (yellow)
    stabilizes around 0.3, then the number of yellow balls is 15. -/
theorem yellow_balls_count (total_balls : ℕ) (yellow_frequency : ℚ) 
  (h1 : total_balls = 50)
  (h2 : yellow_frequency = 3/10) : 
  ∃ (yellow_balls : ℕ), yellow_balls = 15 ∧ yellow_balls / total_balls = yellow_frequency := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l1351_135161


namespace NUMINAMATH_CALUDE_expression_equality_l1351_135192

theorem expression_equality : (2 + Real.sqrt 3) ^ 0 + 3 * Real.tan (π / 6) - |Real.sqrt 3 - 2| + (1 / 2)⁻¹ = 1 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1351_135192


namespace NUMINAMATH_CALUDE_pencil_box_sequence_l1351_135182

theorem pencil_box_sequence (a : ℕ → ℕ) (h1 : a 0 = 78) (h2 : a 1 = 87) (h3 : a 2 = 96) (h4 : a 3 = 105)
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) : a 4 = 114 := by
  sorry

end NUMINAMATH_CALUDE_pencil_box_sequence_l1351_135182


namespace NUMINAMATH_CALUDE_soccer_substitution_remainder_l1351_135145

/-- Represents the number of ways to make substitutions in a soccer game -/
def substitution_ways (total_players : ℕ) (starting_players : ℕ) (max_substitutions : ℕ) : ℕ :=
  let substitute_players := total_players - starting_players
  let rec ways_for_n (n : ℕ) : ℕ :=
    if n = 0 then 1
    else starting_players * (substitute_players - n + 1) * ways_for_n (n - 1)
  (List.range (max_substitutions + 1)).map ways_for_n |> List.sum

/-- The main theorem stating the remainder of substitution ways when divided by 1000 -/
theorem soccer_substitution_remainder :
  substitution_ways 22 11 4 % 1000 = 25 := by
  sorry


end NUMINAMATH_CALUDE_soccer_substitution_remainder_l1351_135145


namespace NUMINAMATH_CALUDE_geometric_arithmetic_mean_ratio_sum_l1351_135173

/-- Given a geometric sequence a, b, c and their arithmetic means m and n,
    prove that a/m + c/n = 2 -/
theorem geometric_arithmetic_mean_ratio_sum (a b c m n : ℝ) 
  (h1 : b ^ 2 = a * c)  -- geometric sequence condition
  (h2 : m = (a + b) / 2)  -- arithmetic mean of a and b
  (h3 : n = (b + c) / 2)  -- arithmetic mean of b and c
  : a / m + c / n = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_mean_ratio_sum_l1351_135173


namespace NUMINAMATH_CALUDE_greene_family_spending_l1351_135137

theorem greene_family_spending (admission_cost food_cost total_cost : ℕ) : 
  admission_cost = 45 →
  food_cost = admission_cost - 13 →
  total_cost = admission_cost + food_cost →
  total_cost = 77 := by
sorry

end NUMINAMATH_CALUDE_greene_family_spending_l1351_135137


namespace NUMINAMATH_CALUDE_inequality_implication_l1351_135128

theorem inequality_implication (a b : ℝ) : a < b → -a + 3 > -b + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1351_135128


namespace NUMINAMATH_CALUDE_volleyball_starters_count_l1351_135140

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

/-- The number of ways to choose 6 starters from 15 players, with at least one of 3 specific players -/
def volleyball_starters : ℕ :=
  binomial 15 6 - binomial 12 6

theorem volleyball_starters_count : volleyball_starters = 4081 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_starters_count_l1351_135140


namespace NUMINAMATH_CALUDE_no_solution_for_inequalities_l1351_135118

theorem no_solution_for_inequalities :
  ¬∃ x : ℝ, (4 * x^2 + 7 * x - 2 < 0) ∧ (3 * x - 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_inequalities_l1351_135118


namespace NUMINAMATH_CALUDE_units_digit_sum_base8_l1351_135151

/-- The units digit of a number in base 8 -/
def units_digit_base8 (n : ℕ) : ℕ := n % 8

/-- Addition in base 8 -/
def add_base8 (a b : ℕ) : ℕ := (a + b) % 8

theorem units_digit_sum_base8 :
  units_digit_base8 (add_base8 67 54) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_base8_l1351_135151


namespace NUMINAMATH_CALUDE_circle_plus_92_composed_thrice_l1351_135189

def circle_plus (N : ℝ) : ℝ := 0.75 * N + 2

theorem circle_plus_92_composed_thrice :
  circle_plus (circle_plus (circle_plus 92)) = 43.4375 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_92_composed_thrice_l1351_135189


namespace NUMINAMATH_CALUDE_shifted_roots_l1351_135178

variable (x : ℝ)

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ := x^3 - 5*x + 7

-- Define the roots a, b, c of the original polynomial
axiom roots_exist : ∃ a b c : ℝ, original_poly a = 0 ∧ original_poly b = 0 ∧ original_poly c = 0

-- Define the shifted polynomial
def shifted_poly (x : ℝ) : ℝ := x^3 + 6*x^2 + 7*x + 5

theorem shifted_roots (a b c : ℝ) : 
  original_poly a = 0 → original_poly b = 0 → original_poly c = 0 →
  shifted_poly (a - 2) = 0 ∧ shifted_poly (b - 2) = 0 ∧ shifted_poly (c - 2) = 0 :=
sorry

end NUMINAMATH_CALUDE_shifted_roots_l1351_135178


namespace NUMINAMATH_CALUDE_circuit_reliability_l1351_135132

-- Define the probabilities of element failures
def p1 : ℝ := 0.2
def p2 : ℝ := 0.3
def p3 : ℝ := 0.4

-- Define the probability of the circuit not breaking
def circuit_not_break : ℝ := (1 - p1) * (1 - p2) * (1 - p3)

-- Theorem statement
theorem circuit_reliability : circuit_not_break = 0.336 := by
  sorry

end NUMINAMATH_CALUDE_circuit_reliability_l1351_135132


namespace NUMINAMATH_CALUDE_work_completion_proof_l1351_135103

/-- The number of men initially planned to complete the work -/
def initial_men : ℕ := 38

/-- The number of days it takes the initial group to complete the work -/
def initial_days : ℕ := 10

/-- The number of men sent to another project -/
def men_sent_away : ℕ := 25

/-- The number of days it takes to complete the work after sending men away -/
def new_days : ℕ := 30

/-- The total amount of work in man-days -/
def total_work : ℕ := initial_men * initial_days

theorem work_completion_proof :
  initial_men * initial_days = (initial_men - men_sent_away) * new_days :=
by sorry

end NUMINAMATH_CALUDE_work_completion_proof_l1351_135103


namespace NUMINAMATH_CALUDE_common_external_tangent_intercept_l1351_135125

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The common external tangent problem --/
theorem common_external_tangent_intercept 
  (c1 : Circle) 
  (c2 : Circle) 
  (h1 : c1.center = (3, 2)) 
  (h2 : c1.radius = 5) 
  (h3 : c2.center = (12, 10)) 
  (h4 : c2.radius = 7) :
  ∃ (m b : ℝ), m > 0 ∧ 
    (∀ (x y : ℝ), y = m * x + b → 
      ((x - c1.center.1)^2 + (y - c1.center.2)^2 = c1.radius^2 ∨
       (x - c2.center.1)^2 + (y - c2.center.2)^2 = c2.radius^2)) ∧
    b = -313/17 := by
  sorry

end NUMINAMATH_CALUDE_common_external_tangent_intercept_l1351_135125


namespace NUMINAMATH_CALUDE_sasha_floor_problem_l1351_135122

theorem sasha_floor_problem (total_floors : ℕ) :
  (∃ (floors_descended : ℕ),
    floors_descended = total_floors / 3 ∧
    floors_descended + 1 = total_floors - (total_floors / 2)) →
  total_floors + 1 = 7 :=
by sorry

end NUMINAMATH_CALUDE_sasha_floor_problem_l1351_135122


namespace NUMINAMATH_CALUDE_moon_permutations_l1351_135168

-- Define the word as a list of characters
def moon : List Char := ['M', 'O', 'O', 'N']

-- Define the number of unique permutations
def uniquePermutations (word : List Char) : ℕ :=
  Nat.factorial word.length / (Nat.factorial (word.count 'O'))

-- Theorem statement
theorem moon_permutations :
  uniquePermutations moon = 12 := by
  sorry

end NUMINAMATH_CALUDE_moon_permutations_l1351_135168


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l1351_135199

/-- Represents a truncated cone with given radii of horizontal bases -/
structure TruncatedCone where
  bottomRadius : ℝ
  topRadius : ℝ

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Predicate to check if a sphere is tangent to a truncated cone -/
def isTangent (cone : TruncatedCone) (sphere : Sphere) : Prop :=
  -- The actual implementation of this predicate is complex and would depend on geometric calculations
  sorry

/-- The main theorem stating the radius of the sphere tangent to the truncated cone -/
theorem sphere_radius_in_truncated_cone (cone : TruncatedCone) 
  (h1 : cone.bottomRadius = 24)
  (h2 : cone.topRadius = 8) :
  ∃ (sphere : Sphere), isTangent cone sphere ∧ sphere.radius = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l1351_135199


namespace NUMINAMATH_CALUDE_equal_distribution_l1351_135183

/-- Represents the weight of a mouse's cheese slice -/
structure CheeseSlice where
  weight : ℝ

/-- Represents the total cheese and its distribution -/
structure Cheese where
  total_weight : ℝ
  white : CheeseSlice
  gray : CheeseSlice
  fat : CheeseSlice
  thin : CheeseSlice

/-- The conditions of the cheese distribution problem -/
def cheese_distribution (c : Cheese) : Prop :=
  c.thin.weight = c.fat.weight - 20 ∧
  c.white.weight = c.gray.weight - 8 ∧
  c.white.weight = c.total_weight / 4 ∧
  c.total_weight = c.white.weight + c.gray.weight + c.fat.weight + c.thin.weight

/-- The theorem stating the equal distribution of surplus cheese -/
theorem equal_distribution (c : Cheese) (h : cheese_distribution c) :
  ∃ (new_c : Cheese),
    cheese_distribution new_c ∧
    new_c.white.weight = new_c.gray.weight ∧
    new_c.fat.weight = new_c.thin.weight ∧
    new_c.fat.weight = c.fat.weight - 6 ∧
    new_c.thin.weight = c.thin.weight + 14 :=
  sorry

end NUMINAMATH_CALUDE_equal_distribution_l1351_135183


namespace NUMINAMATH_CALUDE_f_properties_l1351_135105

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 11

-- State the theorem
theorem f_properties :
  -- Part 1: Tangent line at x = 1 is y = 5
  (∀ y, (y - f 1 = 0 * (x - 1)) ↔ y = 5) ∧
  -- Part 2: Monotonicity intervals
  (∀ x, x < -1 → (deriv f) x > 0) ∧
  (∀ x, x > 1 → (deriv f) x > 0) ∧
  (∀ x, -1 < x ∧ x < 1 → (deriv f) x < 0) ∧
  -- Part 3: Maximum value on [-1, 1] is 17
  (∀ x, -1 ≤ x ∧ x ≤ 1 → f x ≤ 17) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 1 ∧ f x = 17) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1351_135105


namespace NUMINAMATH_CALUDE_committee_formation_ways_l1351_135141

theorem committee_formation_ways (n m : ℕ) (hn : n = 8) (hm : m = 4) :
  Nat.choose n m = 70 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_ways_l1351_135141


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l1351_135196

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of tiles that can fit in a given direction -/
def maxTilesInDirection (floor : Rectangle) (tile : Rectangle) : ℕ :=
  (floor.width / tile.width) * (floor.height / tile.height)

/-- Theorem: The maximum number of 20x30 tiles on a 100x150 floor is 25 -/
theorem max_tiles_on_floor :
  let floor := Rectangle.mk 100 150
  let tile := Rectangle.mk 20 30
  let maxTiles := max (maxTilesInDirection floor tile) (maxTilesInDirection floor (Rectangle.mk tile.height tile.width))
  maxTiles = 25 := by
  sorry

#check max_tiles_on_floor

end NUMINAMATH_CALUDE_max_tiles_on_floor_l1351_135196


namespace NUMINAMATH_CALUDE_lychee_theorem_l1351_135153

def lychee_yield (n : ℕ) : ℕ → ℕ
  | 0 => 1
  | i + 1 =>
    if i < 9 then 2 * lychee_yield n i + 1
    else if i < 15 then lychee_yield n 9
    else (lychee_yield n i) / 2

def total_yield (n : ℕ) : ℕ :=
  (List.range n).map (lychee_yield n) |>.sum

theorem lychee_theorem : total_yield 25 = 8173 := by
  sorry

end NUMINAMATH_CALUDE_lychee_theorem_l1351_135153


namespace NUMINAMATH_CALUDE_marble_problem_l1351_135129

theorem marble_problem (a : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ) 
  (h1 : angela = a) 
  (h2 : brian = 3 * a) 
  (h3 : caden = 4 * brian) 
  (h4 : daryl = 6 * caden) 
  (h5 : angela + brian + caden + daryl = 186) : 
  a = 93 / 44 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l1351_135129


namespace NUMINAMATH_CALUDE_engineer_number_theorem_l1351_135175

def proper_divisors (n : ℕ) : Set ℕ :=
  {d | d ∣ n ∧ d ≠ 1 ∧ d ≠ n}

def increased_divisors (n : ℕ) : Set ℕ :=
  {d + 1 | d ∈ proper_divisors n}

theorem engineer_number_theorem :
  {n : ℕ | ∃ m : ℕ, increased_divisors n = proper_divisors m} = {4, 8} := by
sorry

end NUMINAMATH_CALUDE_engineer_number_theorem_l1351_135175


namespace NUMINAMATH_CALUDE_playlist_duration_l1351_135160

/-- Given a playlist with three songs of durations 3, 2, and 3 minutes respectively,
    prove that listening to this playlist 5 times takes 40 minutes. -/
theorem playlist_duration (song1 song2 song3 : ℕ) (repetitions : ℕ) :
  song1 = 3 ∧ song2 = 2 ∧ song3 = 3 ∧ repetitions = 5 →
  (song1 + song2 + song3) * repetitions = 40 :=
by sorry

end NUMINAMATH_CALUDE_playlist_duration_l1351_135160


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l1351_135190

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_five : opposite (-5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l1351_135190


namespace NUMINAMATH_CALUDE_total_vegetables_bought_l1351_135152

/-- The number of vegetables bought by Marcel and Dale -/
def total_vegetables (marcel_corn : ℕ) (dale_corn : ℕ) (marcel_potatoes : ℕ) (dale_potatoes : ℕ) : ℕ :=
  marcel_corn + dale_corn + marcel_potatoes + dale_potatoes

/-- Theorem stating the total number of vegetables bought by Marcel and Dale -/
theorem total_vegetables_bought :
  ∃ (marcel_corn marcel_potatoes dale_potatoes : ℕ),
    marcel_corn = 10 ∧
    marcel_potatoes = 4 ∧
    dale_potatoes = 8 ∧
    total_vegetables marcel_corn (marcel_corn / 2) marcel_potatoes dale_potatoes = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_vegetables_bought_l1351_135152


namespace NUMINAMATH_CALUDE_solution_set_absolute_value_inequality_l1351_135146

-- Define the set of real numbers less than 2
def lessThanTwo : Set ℝ := {x | x < 2}

-- State the theorem
theorem solution_set_absolute_value_inequality :
  {x : ℝ | |x - 2| > x - 2} = lessThanTwo := by
  sorry

end NUMINAMATH_CALUDE_solution_set_absolute_value_inequality_l1351_135146


namespace NUMINAMATH_CALUDE_five_hundredth_term_is_negative_one_l1351_135144

def sequence_term (n : ℕ) : ℚ :=
  match n % 3 with
  | 1 => 2
  | 2 => -1
  | 0 => 1/2
  | _ => 0 -- This case should never occur

theorem five_hundredth_term_is_negative_one :
  sequence_term 500 = -1 := by
  sorry

end NUMINAMATH_CALUDE_five_hundredth_term_is_negative_one_l1351_135144


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1351_135187

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - (1/6)*x - 1/6 < 0} = Set.Ioo (-1/3 : ℝ) (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1351_135187


namespace NUMINAMATH_CALUDE_sqrt_y_squared_range_l1351_135155

theorem sqrt_y_squared_range (y : ℝ) (h : (y + 16) ^ (1/3) - (y - 16) ^ (1/3) = 4) :
  15 < Real.sqrt (y^2) ∧ Real.sqrt (y^2) < 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_y_squared_range_l1351_135155


namespace NUMINAMATH_CALUDE_min_triangular_faces_l1351_135117

/-- Represents a convex polyhedron --/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  non_triangular_faces : ℕ
  euler : faces + vertices = edges + 2
  more_faces : faces > vertices
  face_sum : faces = triangular_faces + non_triangular_faces
  edge_inequality : edges ≥ (3 * triangular_faces + 4 * non_triangular_faces) / 2

/-- The minimum number of triangular faces in a convex polyhedron with more faces than vertices is 6 --/
theorem min_triangular_faces (p : ConvexPolyhedron) : p.triangular_faces ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_triangular_faces_l1351_135117


namespace NUMINAMATH_CALUDE_extra_bananas_l1351_135176

theorem extra_bananas (total_children absent_children original_bananas : ℕ) 
  (h1 : total_children = 640)
  (h2 : absent_children = 320)
  (h3 : original_bananas = 2) : 
  (total_children * original_bananas) / (total_children - absent_children) - original_bananas = 2 := by
  sorry

end NUMINAMATH_CALUDE_extra_bananas_l1351_135176


namespace NUMINAMATH_CALUDE_integral_even_function_l1351_135136

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- State the theorem
theorem integral_even_function 
  (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_integral : ∫ x in (0:ℝ)..6, f x = 8) : 
  ∫ x in (-6:ℝ)..6, f x = 16 := by
  sorry

end NUMINAMATH_CALUDE_integral_even_function_l1351_135136


namespace NUMINAMATH_CALUDE_eighteenth_replacement_in_december_l1351_135133

def months_in_year : ℕ := 12
def replacement_interval : ℕ := 7
def target_replacement : ℕ := 18

def month_of_replacement (n : ℕ) : ℕ :=
  ((n - 1) * replacement_interval) % months_in_year + 1

theorem eighteenth_replacement_in_december :
  month_of_replacement target_replacement = 12 := by
  sorry

end NUMINAMATH_CALUDE_eighteenth_replacement_in_december_l1351_135133


namespace NUMINAMATH_CALUDE_prime_odd_sum_2009_l1351_135123

theorem prime_odd_sum_2009 :
  ∃! (a b : ℕ), Prime a ∧ Odd b ∧ a^2 + b = 2009 ∧ (a + b : ℕ) = 45 := by
  sorry

end NUMINAMATH_CALUDE_prime_odd_sum_2009_l1351_135123


namespace NUMINAMATH_CALUDE_diagonals_bisect_in_special_quadrilaterals_l1351_135193

-- Define a type for quadrilaterals
inductive Quadrilateral
  | Parallelogram
  | Rectangle
  | Rhombus
  | Square

-- Define a function to check if diagonals bisect each other
def diagonalsBisectEachOther (q : Quadrilateral) : Prop :=
  match q with
  | Quadrilateral.Parallelogram => true
  | Quadrilateral.Rectangle => true
  | Quadrilateral.Rhombus => true
  | Quadrilateral.Square => true

-- Theorem statement
theorem diagonals_bisect_in_special_quadrilaterals (q : Quadrilateral) :
  diagonalsBisectEachOther q := by
  sorry

end NUMINAMATH_CALUDE_diagonals_bisect_in_special_quadrilaterals_l1351_135193


namespace NUMINAMATH_CALUDE_xy_value_l1351_135150

theorem xy_value (x y : ℝ) 
  (h1 : (8 : ℝ)^x / (4 : ℝ)^(x + y) = 16)
  (h2 : (16 : ℝ)^(x + y) / (4 : ℝ)^(7 * y) = 1024) : 
  x * y = 30 := by sorry

end NUMINAMATH_CALUDE_xy_value_l1351_135150


namespace NUMINAMATH_CALUDE_total_marbles_is_27_l1351_135177

/-- The total number of green and red marbles owned by Sara, Tom, and Lisa -/
def total_green_red_marbles (sara_green sara_red : ℕ) (tom_green tom_red : ℕ) (lisa_green lisa_red : ℕ) : ℕ :=
  sara_green + sara_red + tom_green + tom_red + lisa_green + lisa_red

/-- Theorem stating that the total number of green and red marbles is 27 -/
theorem total_marbles_is_27 :
  total_green_red_marbles 3 5 4 7 5 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_is_27_l1351_135177


namespace NUMINAMATH_CALUDE_pump_water_in_half_hour_l1351_135139

/-- Given a pump that moves 560 gallons of water per hour, 
    prove that it will move 280 gallons in 30 minutes. -/
theorem pump_water_in_half_hour (pump_rate : ℝ) (time : ℝ) : 
  pump_rate = 560 → time = 0.5 → pump_rate * time = 280 := by
  sorry

end NUMINAMATH_CALUDE_pump_water_in_half_hour_l1351_135139


namespace NUMINAMATH_CALUDE_ellipse_equation_1_l1351_135127

/-- Given an ellipse with semi-major axis a = 6 and eccentricity e = 1/3,
    prove that its standard equation is x²/36 + y²/32 = 1 -/
theorem ellipse_equation_1 (x y : ℝ) (a b c : ℝ) (h1 : a = 6) (h2 : c/a = 1/3) :
  x^2/a^2 + y^2/b^2 = 1 ↔ x^2/36 + y^2/32 = 1 := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_1_l1351_135127


namespace NUMINAMATH_CALUDE_train_length_calculation_l1351_135191

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length_calculation (train_speed : ℝ) (cross_time : ℝ) (bridge_length : ℝ) :
  train_speed = 65 * (1000 / 3600) →
  cross_time = 13.568145317605362 →
  bridge_length = 145 →
  ∃ (train_length : ℝ), abs (train_length - 100) < 0.1 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l1351_135191


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1351_135148

-- Define the original proposition
def original_proposition (m : ℝ) : Prop :=
  m > 0 → ∃ x : ℝ, x^2 + x - m = 0

-- Define the contrapositive
def contrapositive (m : ℝ) : Prop :=
  (¬∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0

-- Theorem stating the equivalence of the contrapositive to the original proposition
theorem contrapositive_equivalence :
  ∀ m : ℝ, (¬original_proposition m) ↔ contrapositive m :=
by
  sorry


end NUMINAMATH_CALUDE_contrapositive_equivalence_l1351_135148


namespace NUMINAMATH_CALUDE_total_dogs_l1351_135158

theorem total_dogs (num_boxes : ℕ) (dogs_per_box : ℕ) (h1 : num_boxes = 7) (h2 : dogs_per_box = 4) :
  num_boxes * dogs_per_box = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_dogs_l1351_135158


namespace NUMINAMATH_CALUDE_complement_of_A_wrt_U_l1351_135131

def U : Set Int := {-1, 2, 4}
def A : Set Int := {-1, 4}

theorem complement_of_A_wrt_U :
  (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_wrt_U_l1351_135131


namespace NUMINAMATH_CALUDE_hostel_problem_l1351_135126

/-- The number of days the provisions would last for the initial number of men -/
def initial_days : ℕ := 32

/-- The number of days the provisions would last if 50 men left -/
def reduced_days : ℕ := 40

/-- The number of men that left the hostel -/
def men_left : ℕ := 50

/-- The initial number of men in the hostel -/
def initial_men : ℕ := 250

theorem hostel_problem :
  initial_men = 250 ∧
  (initial_days : ℚ) * initial_men = reduced_days * (initial_men - men_left) :=
sorry

end NUMINAMATH_CALUDE_hostel_problem_l1351_135126


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l1351_135163

theorem arithmetic_simplification : 
  (427 / 2.68) * 16 * 26.8 / 42.7 * 16 = 25600 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l1351_135163


namespace NUMINAMATH_CALUDE_sum_of_digits_7ab_l1351_135180

/-- Integer consisting of 1234 sevens in base 10 -/
def a : ℕ := 7 * (10^1234 - 1) / 9

/-- Integer consisting of 1234 twos in base 10 -/
def b : ℕ := 2 * (10^1234 - 1) / 9

/-- Sum of digits in the base 10 representation of a number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_7ab : sum_of_digits (7 * a * b) = 11100 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_7ab_l1351_135180
