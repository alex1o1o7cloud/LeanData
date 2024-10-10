import Mathlib

namespace sum_base6_series_l2731_273100

/-- Converts a number from base 6 to base 10 -/
def base6To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 6 -/
def base10To6 (n : ℕ) : ℕ := sorry

/-- Sum of arithmetic series in base 10 -/
def sumArithmeticSeries (a l n : ℕ) : ℕ := n * (a + l) / 2

theorem sum_base6_series :
  let first := base6To10 3
  let last := base6To10 100
  let n := last - first + 1
  base10To6 (sumArithmeticSeries first last n) = 3023 :=
by sorry

end sum_base6_series_l2731_273100


namespace inserted_numbers_sum_l2731_273120

/-- Given four positive numbers in sequence where the first is 4 and the last is 16,
    with two numbers inserted between them such that the first three form a geometric progression
    and the last three form a harmonic progression, prove that the sum of the inserted numbers is 8. -/
theorem inserted_numbers_sum (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧  -- x and y are positive
  (∃ r : ℝ, r > 0 ∧ x = 4 * r ∧ y = 4 * r^2) ∧  -- geometric progression
  2 / y = 1 / x + 1 / 16 →  -- harmonic progression
  x + y = 8 := by
sorry

end inserted_numbers_sum_l2731_273120


namespace convex_ngon_divided_into_equal_triangles_l2731_273141

/-- A convex n-gon that is circumscribed and divided into equal triangles by non-intersecting diagonals -/
structure ConvexNGon (n : ℕ) :=
  (convex : Bool)
  (circumscribed : Bool)
  (equal_triangles : Bool)
  (non_intersecting_diagonals : Bool)

/-- Theorem stating that the only possible value for n is 4 -/
theorem convex_ngon_divided_into_equal_triangles
  (n : ℕ) (ngon : ConvexNGon n) (h1 : n > 3)
  (h2 : ngon.convex = true)
  (h3 : ngon.circumscribed = true)
  (h4 : ngon.equal_triangles = true)
  (h5 : ngon.non_intersecting_diagonals = true) :
  n = 4 :=
sorry

end convex_ngon_divided_into_equal_triangles_l2731_273141


namespace prob_same_length_is_one_fifth_l2731_273129

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℕ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The number of diagonals of each distinct length -/
def num_diagonals_per_length : ℕ := 3

/-- The probability of selecting two elements of the same length from T -/
def prob_same_length : ℚ := sorry

theorem prob_same_length_is_one_fifth :
  prob_same_length = 1 / 5 := by sorry

end prob_same_length_is_one_fifth_l2731_273129


namespace power_inequality_l2731_273126

theorem power_inequality (x y : ℝ) (h : x^2013 + y^2013 > x^2012 + y^2012) :
  x^2014 + y^2014 > x^2013 + y^2013 :=
by
  sorry

end power_inequality_l2731_273126


namespace restricted_photo_arrangements_l2731_273184

/-- The number of ways to arrange n people in a line -/
def lineArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line where one specific person is restricted -/
def restrictedArrangements (n : ℕ) : ℕ := (n - 2) * Nat.factorial (n - 1)

/-- Theorem stating that for 5 people, with one person restricted from ends, there are 72 arrangements -/
theorem restricted_photo_arrangements :
  restrictedArrangements 5 = 72 := by
  sorry

end restricted_photo_arrangements_l2731_273184


namespace picture_processing_time_l2731_273165

/-- Given 960 pictures and a processing time of 2 minutes per picture, 
    the total processing time in hours is equal to 32. -/
theorem picture_processing_time : 
  let num_pictures : ℕ := 960
  let processing_time_per_picture : ℕ := 2
  let minutes_per_hour : ℕ := 60
  (num_pictures * processing_time_per_picture) / minutes_per_hour = 32 := by
sorry

end picture_processing_time_l2731_273165


namespace f_derivative_l2731_273197

def f (x : ℝ) : ℝ := -3 * x - 1

theorem f_derivative : 
  deriv f = λ x => -3 := by sorry

end f_derivative_l2731_273197


namespace ralph_tv_hours_l2731_273127

/-- The number of hours Ralph watches TV on weekdays (Monday to Friday) -/
def weekday_hours : ℕ := 4

/-- The number of hours Ralph watches TV on weekend days (Saturday and Sunday) -/
def weekend_hours : ℕ := 6

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The total number of hours Ralph watches TV in one week -/
def total_hours : ℕ := weekday_hours * weekdays + weekend_hours * weekend_days

theorem ralph_tv_hours : total_hours = 32 := by
  sorry

end ralph_tv_hours_l2731_273127


namespace remainder_two_power_33_minus_one_mod_9_l2731_273161

theorem remainder_two_power_33_minus_one_mod_9 : 2^33 - 1 ≡ 7 [ZMOD 9] := by
  sorry

end remainder_two_power_33_minus_one_mod_9_l2731_273161


namespace equation_solution_l2731_273119

theorem equation_solution : ∃ x : ℚ, (5*x + 2*x = 450 - 10*(x - 5) + 4) ∧ (x = 504/17) := by
  sorry

end equation_solution_l2731_273119


namespace roots_sum_of_squares_l2731_273168

theorem roots_sum_of_squares (a b : ℝ) : 
  (∀ x, x^2 - 4*x + 4 = 0 ↔ x = a ∨ x = b) → a^2 + b^2 = 8 := by
sorry

end roots_sum_of_squares_l2731_273168


namespace existence_of_indistinguishable_arrangements_l2731_273175

/-- Represents the type of a tree -/
inductive TreeType
| Oak
| Baobab

/-- Represents a row of trees -/
def TreeRow := List TreeType

/-- Counts the number of oaks in a group of three adjacent trees -/
def countOaks (trees : TreeRow) (index : Nat) : Nat :=
  match trees.get? index, trees.get? (index + 1), trees.get? (index + 2) with
  | some TreeType.Oak, _, _ => 1
  | _, some TreeType.Oak, _ => 1
  | _, _, some TreeType.Oak => 1
  | _, _, _ => 0

/-- Generates the sequence of tag numbers for a given row of trees -/
def generateTags (trees : TreeRow) : List Nat :=
  List.range trees.length |>.map (countOaks trees)

/-- Theorem stating that there exist two different arrangements of trees
    with the same tag sequence -/
theorem existence_of_indistinguishable_arrangements :
  ∃ (row1 row2 : TreeRow),
    row1.length = 2000 ∧
    row2.length = 2000 ∧
    row1 ≠ row2 ∧
    generateTags row1 = generateTags row2 :=
sorry

end existence_of_indistinguishable_arrangements_l2731_273175


namespace expression_value_l2731_273177

theorem expression_value (a b : ℚ) (ha : a = -1) (hb : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end expression_value_l2731_273177


namespace solution_difference_l2731_273178

/-- Given that r and s are distinct solutions to the equation (6x-18)/(x^2+4x-21) = x+3,
    and r > s, prove that r - s = 10. -/
theorem solution_difference (r s : ℝ) : 
  r ≠ s →
  (6*r - 18) / (r^2 + 4*r - 21) = r + 3 →
  (6*s - 18) / (s^2 + 4*s - 21) = s + 3 →
  r > s →
  r - s = 10 := by
sorry

end solution_difference_l2731_273178


namespace equilateral_triangle_division_exists_l2731_273122

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Represents a division of an equilateral triangle into smaller equilateral triangles -/
structure TriangleDivision where
  original : EquilateralTriangle
  num_divisions : ℕ
  side_lengths : Finset ℝ
  all_positive : ∀ l ∈ side_lengths, l > 0

/-- Theorem stating that there exists a division of an equilateral triangle into 2011 smaller equilateral triangles with only two different side lengths -/
theorem equilateral_triangle_division_exists : 
  ∃ (div : TriangleDivision), div.num_divisions = 2011 ∧ div.side_lengths.card = 2 :=
sorry

end equilateral_triangle_division_exists_l2731_273122


namespace remainder_of_power_minus_digit_l2731_273166

theorem remainder_of_power_minus_digit (x : ℕ) : 
  x < 10 → (Nat.pow 2 200 - x) % 7 = 1 → x = 3 := by sorry

end remainder_of_power_minus_digit_l2731_273166


namespace rectangle_area_l2731_273189

theorem rectangle_area (x : ℝ) : 
  x > 0 → 
  ∃ w l : ℝ, w > 0 ∧ l > 0 ∧ 
  l = 3 * w ∧ 
  x^2 = l^2 + w^2 ∧
  w * l = (3/10) * x^2 :=
by sorry

end rectangle_area_l2731_273189


namespace binomial_sum_divides_power_of_two_l2731_273111

theorem binomial_sum_divides_power_of_two (n : ℕ) : 
  n > 3 →
  (1 + Nat.choose n 1 + Nat.choose n 2 + Nat.choose n 3 ∣ 2^2000) ↔ 
  (n = 7 ∨ n = 23) := by
sorry

end binomial_sum_divides_power_of_two_l2731_273111


namespace negation_false_l2731_273194

theorem negation_false : ¬∃ (x y : ℝ), x > 2 ∧ y > 3 ∧ x + y ≤ 5 := by sorry

end negation_false_l2731_273194


namespace cookie_sales_proof_l2731_273195

/-- Represents the total number of boxes of cookies sold -/
def total_boxes (chocolate_chip_boxes : ℝ) (plain_boxes : ℝ) : ℝ :=
  chocolate_chip_boxes + plain_boxes

/-- Represents the total sales value -/
def total_sales (chocolate_chip_boxes : ℝ) (plain_boxes : ℝ) : ℝ :=
  1.25 * chocolate_chip_boxes + 0.75 * plain_boxes

theorem cookie_sales_proof :
  ∀ (chocolate_chip_boxes : ℝ) (plain_boxes : ℝ),
    plain_boxes = 793.375 →
    total_sales chocolate_chip_boxes plain_boxes = 1586.75 →
    total_boxes chocolate_chip_boxes plain_boxes = 1586.75 :=
by
  sorry

#check cookie_sales_proof

end cookie_sales_proof_l2731_273195


namespace intersection_equals_interval_l2731_273149

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Define the half-open interval [1, 2)
def interval : Set ℝ := {x | 1 ≤ x ∧ x < 2}

-- Theorem statement
theorem intersection_equals_interval : M ∩ N = interval := by sorry

end intersection_equals_interval_l2731_273149


namespace lcm_48_75_l2731_273138

theorem lcm_48_75 : Nat.lcm 48 75 = 1200 := by
  sorry

end lcm_48_75_l2731_273138


namespace square_difference_equality_l2731_273103

theorem square_difference_equality : 1012^2 - 992^2 - 1008^2 + 996^2 = 16032 := by
  sorry

end square_difference_equality_l2731_273103


namespace count_valid_sequences_l2731_273196

/-- Represents a sequence of non-negative integers -/
def Sequence := ℕ → ℕ

/-- Checks if a sequence satisfies the given conditions -/
def ValidSequence (a : Sequence) : Prop :=
  a 0 = 2016 ∧
  (∀ n, a (n + 1) ≤ Real.sqrt (a n)) ∧
  (∀ m n, m ≠ n → a m ≠ a n)

/-- Counts the number of valid sequences -/
def CountValidSequences : ℕ := sorry

/-- The main theorem stating that the count of valid sequences is 948 -/
theorem count_valid_sequences :
  CountValidSequences = 948 := by sorry

end count_valid_sequences_l2731_273196


namespace average_rate_of_change_f_l2731_273180

-- Define the function f(x) = x^2 + 2
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the interval [1, 3]
def a : ℝ := 1
def b : ℝ := 3

-- Theorem: The average rate of change of f(x) on [1, 3] is 4
theorem average_rate_of_change_f : (f b - f a) / (b - a) = 4 := by
  sorry

end average_rate_of_change_f_l2731_273180


namespace square_side_length_l2731_273155

/-- The side length of a square with area equal to a 3 cm × 27 cm rectangle is 9 cm. -/
theorem square_side_length (square_area rectangle_area : ℝ) (square_side : ℝ) : 
  square_area = rectangle_area →
  rectangle_area = 3 * 27 →
  square_area = square_side ^ 2 →
  square_side = 9 := by
  sorry

end square_side_length_l2731_273155


namespace unit_vector_of_difference_l2731_273130

/-- Given vectors a and b in ℝ², prove that the unit vector of a - b is (-4/5, 3/5) -/
theorem unit_vector_of_difference (a b : ℝ × ℝ) (ha : a = (3, 1)) (hb : b = (7, -2)) :
  let diff := a - b
  let norm := Real.sqrt ((diff.1)^2 + (diff.2)^2)
  (diff.1 / norm, diff.2 / norm) = (-4/5, 3/5) := by
sorry

end unit_vector_of_difference_l2731_273130


namespace largest_prime_divisor_of_sum_of_squares_l2731_273171

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : Nat, p.Prime ∧ p ∣ (36^2 + 49^2) ∧ ∀ q : Nat, q.Prime → q ∣ (36^2 + 49^2) → q ≤ p :=
by sorry

end largest_prime_divisor_of_sum_of_squares_l2731_273171


namespace largest_n_for_product_2210_l2731_273115

/-- An arithmetic sequence with integer terms -/
def ArithmeticSeq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem largest_n_for_product_2210 :
  ∀ a b : ℕ → ℕ,
  ArithmeticSeq a → ArithmeticSeq b →
  a 1 = 1 → b 1 = 1 →
  a 2 ≤ b 2 →
  (∃ n : ℕ, a n * b n = 2210) →
  (∀ m : ℕ, (∃ k : ℕ, a k * b k = 2210) → m ≤ 170) :=
sorry

end largest_n_for_product_2210_l2731_273115


namespace vertical_distance_traveled_l2731_273101

/-- Calculate the total vertical distance traveled in a week -/
theorem vertical_distance_traveled (story : Nat) (trips_per_day : Nat) (feet_per_story : Nat) (days_in_week : Nat) : 
  story = 5 → trips_per_day = 3 → feet_per_story = 10 → days_in_week = 7 →
  2 * story * feet_per_story * trips_per_day * days_in_week = 2100 :=
by
  sorry

end vertical_distance_traveled_l2731_273101


namespace power_calculation_l2731_273162

theorem power_calculation : 2^345 - 8^3 / 8^2 + 3^2 = 2^345 + 1 := by
  sorry

end power_calculation_l2731_273162


namespace log_expression_equality_l2731_273142

theorem log_expression_equality : 
  2 * Real.log 10 / Real.log 5 + Real.log (1/4) / Real.log 5 + (2 : ℝ) ^ (Real.log 3 / Real.log 4) = 2 + Real.sqrt 3 := by
  sorry

end log_expression_equality_l2731_273142


namespace nyc_streetlights_l2731_273144

/-- The number of streetlights bought by the New York City Council -/
theorem nyc_streetlights (num_squares : ℕ) (lights_per_square : ℕ) (unused_lights : ℕ) :
  num_squares = 15 →
  lights_per_square = 12 →
  unused_lights = 20 →
  num_squares * lights_per_square + unused_lights = 200 := by
  sorry


end nyc_streetlights_l2731_273144


namespace robot_center_not_necessarily_on_line_l2731_273183

/-- Represents a circular robot -/
structure CircularRobot where
  center : ℝ × ℝ
  radius : ℝ
  deriving Inhabited

/-- Represents a movement of the robot -/
def RobotMovement := ℝ → CircularRobot

/-- A point remains on a line throughout the movement -/
def PointRemainsOnLine (p : ℝ × ℝ) (m : RobotMovement) : Prop :=
  ∃ (a b c : ℝ), ∀ t, a * (m t).center.1 + b * (m t).center.2 + c = 0

/-- The theorem statement -/
theorem robot_center_not_necessarily_on_line :
  ∃ (m : RobotMovement),
    (∀ θ : ℝ, PointRemainsOnLine ((m 0).center.1 + (m 0).radius * Real.cos θ,
                                  (m 0).center.2 + (m 0).radius * Real.sin θ) m) ∧
    ¬ PointRemainsOnLine (m 0).center m :=
  sorry


end robot_center_not_necessarily_on_line_l2731_273183


namespace distinct_convex_polygons_l2731_273153

/-- The number of points marked on the circle -/
def num_points : ℕ := 12

/-- The total number of subsets of the points -/
def total_subsets : ℕ := 2^num_points

/-- The number of subsets with 0 members -/
def subsets_0 : ℕ := (num_points.choose 0)

/-- The number of subsets with 1 member -/
def subsets_1 : ℕ := (num_points.choose 1)

/-- The number of subsets with 2 members -/
def subsets_2 : ℕ := (num_points.choose 2)

/-- The number of distinct convex polygons with three or more sides -/
def num_polygons : ℕ := total_subsets - subsets_0 - subsets_1 - subsets_2

theorem distinct_convex_polygons :
  num_polygons = 4017 :=
by sorry

end distinct_convex_polygons_l2731_273153


namespace stratified_sampling_school_a_l2731_273102

theorem stratified_sampling_school_a (total_sample : ℕ) 
  (school_a : ℕ) (school_b : ℕ) (school_c : ℕ) : 
  total_sample = 90 → 
  school_a = 3600 → 
  school_b = 5400 → 
  school_c = 1800 → 
  (school_a * total_sample) / (school_a + school_b + school_c) = 30 := by
  sorry

end stratified_sampling_school_a_l2731_273102


namespace condition_necessary_not_sufficient_l2731_273157

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x^2 - x - 2 < 0 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ ¬(x^2 - x - 2 < 0)) :=
by sorry

end condition_necessary_not_sufficient_l2731_273157


namespace l_shaped_count_is_even_l2731_273167

/-- A centrally symmetric figure on a grid --/
structure CentrallySymmetricFigure where
  n : ℕ  -- number of "L-shaped" figures
  k : ℕ  -- number of 1 × 4 rectangles

/-- Theorem: The number of "L-shaped" figures in a centrally symmetric figure is even --/
theorem l_shaped_count_is_even (figure : CentrallySymmetricFigure) : Even figure.n := by
  sorry

end l_shaped_count_is_even_l2731_273167


namespace angle_through_point_l2731_273182

theorem angle_through_point (α : Real) :
  0 ≤ α → α < 2 * Real.pi →
  let P : ℝ × ℝ := (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))
  (Real.cos α = P.1 ∧ Real.sin α = P.2) →
  α = 11 * Real.pi / 6 := by
  sorry

end angle_through_point_l2731_273182


namespace reciprocal_roots_imply_a_eq_neg_one_l2731_273117

theorem reciprocal_roots_imply_a_eq_neg_one (a : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ 
    x^2 + (a-1)*x + a^2 = 0 ∧ 
    y^2 + (a-1)*y + a^2 = 0 ∧ 
    x*y = 1) → 
  a = -1 :=
by sorry

end reciprocal_roots_imply_a_eq_neg_one_l2731_273117


namespace tony_cheese_purchase_l2731_273135

theorem tony_cheese_purchase (initial_amount : ℕ) (cheese_cost : ℕ) (beef_cost : ℕ) (remaining_amount : ℕ)
  (h1 : initial_amount = 87)
  (h2 : cheese_cost = 7)
  (h3 : beef_cost = 5)
  (h4 : remaining_amount = 61) :
  (initial_amount - remaining_amount - beef_cost) / cheese_cost = 3 := by
  sorry

end tony_cheese_purchase_l2731_273135


namespace crate_stack_probability_l2731_273198

-- Define the dimensions of a crate
def CrateDimensions : Fin 3 → ℕ
  | 0 => 4
  | 1 => 5
  | 2 => 7

-- Define the number of crates
def NumCrates : ℕ := 15

-- Define the target height
def TargetHeight : ℕ := 50

-- Define the total number of possible arrangements
def TotalArrangements : ℕ := 3^NumCrates

-- Define the number of favorable arrangements
def FavorableArrangements : ℕ := 560

theorem crate_stack_probability :
  (FavorableArrangements : ℚ) / TotalArrangements = 560 / 14348907 := by
  sorry

#eval FavorableArrangements -- Should output 560

end crate_stack_probability_l2731_273198


namespace total_questions_answered_l2731_273143

/-- Represents a tour group with the number of tourists asking different amounts of questions -/
structure TourGroup where
  usual : ℕ  -- number of tourists asking the usual 2 questions
  zero : ℕ   -- number of tourists asking 0 questions
  one : ℕ    -- number of tourists asking 1 question
  three : ℕ  -- number of tourists asking 3 questions
  five : ℕ   -- number of tourists asking 5 questions
  double : ℕ -- number of tourists asking double the usual (4 questions)
  triple : ℕ -- number of tourists asking triple the usual (6 questions)
  quad : ℕ   -- number of tourists asking quadruple the usual (8 questions)

/-- Calculates the total number of questions for a tour group -/
def questionsForGroup (g : TourGroup) : ℕ :=
  2 * g.usual + 0 * g.zero + 1 * g.one + 3 * g.three + 5 * g.five +
  4 * g.double + 6 * g.triple + 8 * g.quad

/-- The six tour groups as described in the problem -/
def tourGroups : List TourGroup := [
  ⟨3, 0, 2, 0, 1, 0, 0, 0⟩,  -- Group A
  ⟨4, 1, 0, 6, 0, 0, 0, 0⟩,  -- Group B
  ⟨4, 2, 1, 0, 0, 0, 1, 0⟩,  -- Group C
  ⟨3, 1, 0, 0, 0, 0, 0, 1⟩,  -- Group D
  ⟨3, 2, 0, 0, 1, 3, 0, 0⟩,  -- Group E
  ⟨4, 1, 0, 2, 0, 0, 0, 0⟩   -- Group F
]

theorem total_questions_answered (groups := tourGroups) :
  (groups.map questionsForGroup).sum = 105 := by sorry

end total_questions_answered_l2731_273143


namespace solution_set_inequality_l2731_273124

theorem solution_set_inequality (x : ℝ) : 
  (((1 - x) / (x + 1) ≤ 0) ↔ (x ∈ Set.Iic (-1) ∪ Set.Ici 1)) := by
  sorry

end solution_set_inequality_l2731_273124


namespace subtraction_for_complex_equality_l2731_273192

theorem subtraction_for_complex_equality : ∃ (z : ℂ), (7 - 3*I) - z = 3 * ((2 + I) + (4 - 2*I)) ∧ z = -11 := by
  sorry

end subtraction_for_complex_equality_l2731_273192


namespace green_bean_to_onion_ratio_l2731_273105

def potato_count : ℕ := 2
def carrot_to_potato_ratio : ℕ := 6
def onion_to_carrot_ratio : ℕ := 2
def green_bean_count : ℕ := 8

def carrot_count : ℕ := potato_count * carrot_to_potato_ratio
def onion_count : ℕ := carrot_count * onion_to_carrot_ratio

theorem green_bean_to_onion_ratio :
  (green_bean_count : ℚ) / onion_count = 1 / 3 := by
  sorry

end green_bean_to_onion_ratio_l2731_273105


namespace power_ranger_stickers_l2731_273150

theorem power_ranger_stickers (total : ℕ) (first_box : ℕ) : 
  total = 58 → first_box = 23 → (total - first_box) - first_box = 12 := by
  sorry

end power_ranger_stickers_l2731_273150


namespace exactly_three_tangent_lines_l2731_273163

/-- A line passing through (0, 1) that intersects the parabola y^2 = 4x at only one point -/
structure TangentLine where
  slope : ℝ
  intersects_once : ∃! (x y : ℝ), y^2 = 4*x ∧ y = slope * x + 1

/-- The number of lines passing through (0, 1) that intersect y^2 = 4x at only one point -/
def num_tangent_lines : ℕ := sorry

/-- Theorem stating that there are exactly 3 such lines -/
theorem exactly_three_tangent_lines : num_tangent_lines = 3 := by sorry

end exactly_three_tangent_lines_l2731_273163


namespace total_rain_time_l2731_273112

/-- Given rain durations over three days, prove the total rain time -/
theorem total_rain_time (first_day_start : Nat) (first_day_end : Nat)
  (h1 : first_day_end - first_day_start = 10)
  (h2 : ∃ second_day_duration : Nat, second_day_duration = (first_day_end - first_day_start) + 2)
  (h3 : ∃ third_day_duration : Nat, third_day_duration = 2 * (first_day_end - first_day_start + 2)) :
  ∃ total_duration : Nat, total_duration = 46 := by
  sorry

end total_rain_time_l2731_273112


namespace sara_earnings_l2731_273152

/-- Sara's cake-making and selling scenario --/
def sara_cake_scenario (weekdays_per_week : ℕ) (cakes_per_day : ℕ) (price_per_cake : ℕ) (num_weeks : ℕ) : ℕ :=
  weekdays_per_week * cakes_per_day * price_per_cake * num_weeks

/-- Theorem: Sara's earnings over 4 weeks --/
theorem sara_earnings : sara_cake_scenario 5 4 8 4 = 640 := by
  sorry

end sara_earnings_l2731_273152


namespace class_fraction_proof_l2731_273145

/-- 
Given a class of students where:
1) The ratio of boys to girls is 2
2) Half the number of girls is equal to some fraction of the total number of students
This theorem proves that the fraction in condition 2 is 1/6
-/
theorem class_fraction_proof (G : ℚ) (h1 : G > 0) : 
  let B := 2 * G
  let total := G + B
  ∃ (x : ℚ), (1/2) * G = x * total ∧ x = 1/6 := by
sorry

end class_fraction_proof_l2731_273145


namespace remainder_sum_equals_27_l2731_273158

theorem remainder_sum_equals_27 (a : ℕ) (h : a > 0) : 
  (50 % a + 72 % a + 157 % a = 27) → a = 21 :=
by sorry

end remainder_sum_equals_27_l2731_273158


namespace A_n_nonempty_finite_l2731_273116

/-- The set A_n for a positive integer n -/
def A_n (n : ℕ+) : Set (ℕ × ℕ) :=
  {p : ℕ × ℕ | ∃ k : ℕ, (Real.sqrt (p.1^2 + p.2 + n) + Real.sqrt (p.2^2 + p.1 + n) : ℝ) = k}

/-- Theorem stating that A_n is non-empty and finite for any positive integer n -/
theorem A_n_nonempty_finite (n : ℕ+) : Set.Nonempty (A_n n) ∧ Set.Finite (A_n n) := by
  sorry

end A_n_nonempty_finite_l2731_273116


namespace sum_f_positive_l2731_273176

noncomputable def f (x : ℝ) : ℝ := x^3 / Real.cos x

theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
  (h₁ : |x₁| < π/2) (h₂ : |x₂| < π/2) (h₃ : |x₃| < π/2)
  (h₄ : x₁ + x₂ > 0) (h₅ : x₂ + x₃ > 0) (h₆ : x₁ + x₃ > 0) :
  f x₁ + f x₂ + f x₃ > 0 := by
  sorry

end sum_f_positive_l2731_273176


namespace josh_marbles_remaining_l2731_273123

def initial_marbles : ℕ := 16
def lost_marbles : ℕ := 7

theorem josh_marbles_remaining : initial_marbles - lost_marbles = 9 := by
  sorry

end josh_marbles_remaining_l2731_273123


namespace glasses_purchase_price_l2731_273107

/-- The purchase price of the glasses in yuan -/
def purchase_price : ℝ := 80

/-- The selling price after the initial increase -/
def increased_price (x : ℝ) : ℝ := 10 * x

/-- The selling price after applying the discount -/
def discounted_price (x : ℝ) : ℝ := 0.5 * increased_price x

/-- The profit made from selling the glasses -/
def profit (x : ℝ) : ℝ := discounted_price x - 20 - x

theorem glasses_purchase_price :
  profit purchase_price = 300 :=
sorry

end glasses_purchase_price_l2731_273107


namespace abs_is_even_and_increasing_l2731_273156

-- Define the absolute value function
def f (x : ℝ) := abs x

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define what it means for a function to be increasing on an interval
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem abs_is_even_and_increasing :
  is_even f ∧ is_increasing_on f 0 1 :=
sorry

end abs_is_even_and_increasing_l2731_273156


namespace M_values_l2731_273173

theorem M_values (a b : ℚ) (h : a * b ≠ 0) :
  let M := |a| / a + b / |b|
  M = 0 ∨ M = 2 ∨ M = -2 := by
sorry

end M_values_l2731_273173


namespace albert_running_laps_l2731_273108

theorem albert_running_laps 
  (total_distance : ℕ) 
  (track_length : ℕ) 
  (laps_run : ℕ) 
  (h1 : total_distance = 99)
  (h2 : track_length = 9)
  (h3 : laps_run = 6) :
  (total_distance / track_length) - laps_run = 5 :=
by
  sorry

#eval (99 / 9) - 6  -- This should output 5

end albert_running_laps_l2731_273108


namespace arithmetic_sequence_sum_l2731_273188

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 5 + a 13 = 40) →
  (a 8 + a 9 + a 10 = 60) := by
  sorry

end arithmetic_sequence_sum_l2731_273188


namespace race_time_comparison_l2731_273154

theorem race_time_comparison 
  (a : ℝ) (V : ℝ) 
  (h1 : a > 0) (h2 : V > 0) : 
  let planned_time := a / V
  let first_half_time := a / (2 * 1.25 * V)
  let second_half_time := a / (2 * 0.8 * V)
  let actual_time := first_half_time + second_half_time
  actual_time > planned_time :=
by sorry

end race_time_comparison_l2731_273154


namespace complex_equality_l2731_273172

/-- Given a real number b, if the real part is equal to the imaginary part
    for the complex number (1+i)/(1-i) + (1/2)b, then b = 2 -/
theorem complex_equality (b : ℝ) : 
  (((1 : ℂ) + Complex.I) / ((1 : ℂ) - Complex.I) + (1 / 2 : ℂ) * b).re = 
  (((1 : ℂ) + Complex.I) / ((1 : ℂ) - Complex.I) + (1 / 2 : ℂ) * b).im → b = 2 := by
  sorry

end complex_equality_l2731_273172


namespace circle_through_ellipse_vertices_l2731_273139

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point lies on an ellipse -/
def Point.onEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Check if a point lies on a circle -/
def Point.onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The main theorem to prove -/
theorem circle_through_ellipse_vertices (e : Ellipse) (c : Circle) : 
  e.a = 4 ∧ e.b = 2 ∧ 
  c.center.x = 3/2 ∧ c.center.y = 0 ∧ c.radius = 5/2 →
  (∃ (p1 p2 p3 : Point), 
    p1.onEllipse e ∧ p2.onEllipse e ∧ p3.onEllipse e ∧
    p1.onCircle c ∧ p2.onCircle c ∧ p3.onCircle c) :=
by
  sorry

end circle_through_ellipse_vertices_l2731_273139


namespace sin_neg_pi_l2731_273106

theorem sin_neg_pi : Real.sin (-π) = 0 := by
  sorry

end sin_neg_pi_l2731_273106


namespace f_decreasing_range_l2731_273174

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else a/x

/-- Theorem stating the range of a for f(x) to be decreasing -/
theorem f_decreasing_range (a : ℝ) :
  (∀ x y, x < y → f a x > f a y) →
  1/6 ≤ a ∧ a < 1/3 := by sorry

end f_decreasing_range_l2731_273174


namespace baker_cake_difference_l2731_273133

/-- Given the initial number of cakes, the number of cakes sold, and the number of cakes bought,
    prove that the difference between cakes sold and cakes bought is 47. -/
theorem baker_cake_difference (initial : ℕ) (sold : ℕ) (bought : ℕ)
    (h1 : initial = 170)
    (h2 : sold = 78)
    (h3 : bought = 31) :
  sold - bought = 47 := by
  sorry

end baker_cake_difference_l2731_273133


namespace pool_filling_buckets_l2731_273169

theorem pool_filling_buckets 
  (george_buckets : ℕ) 
  (harry_buckets : ℕ) 
  (total_rounds : ℕ) :
  george_buckets = 2 →
  harry_buckets = 3 →
  total_rounds = 22 →
  (george_buckets + harry_buckets) * total_rounds = 110 := by
sorry

end pool_filling_buckets_l2731_273169


namespace fuel_station_service_cost_l2731_273109

/-- Fuel station problem -/
theorem fuel_station_service_cost
  (fuel_cost_per_liter : Real)
  (num_minivans : Nat)
  (num_trucks : Nat)
  (total_cost : Real)
  (minivan_tank_capacity : Real)
  (truck_tank_multiplier : Real)
  (h1 : fuel_cost_per_liter = 0.70)
  (h2 : num_minivans = 4)
  (h3 : num_trucks = 2)
  (h4 : total_cost = 395.4)
  (h5 : minivan_tank_capacity = 65)
  (h6 : truck_tank_multiplier = 2.2)
  : (total_cost - (fuel_cost_per_liter * 
      (num_minivans * minivan_tank_capacity + 
       num_trucks * (minivan_tank_capacity * truck_tank_multiplier)))) / 
    (num_minivans + num_trucks) = 2.2 := by
  sorry

end fuel_station_service_cost_l2731_273109


namespace expression_equals_one_l2731_273147

theorem expression_equals_one : 
  (121^2 - 13^2) / (91^2 - 17^2) * ((91-17)*(91+17)) / ((121-13)*(121+13)) = 1 := by
  sorry

end expression_equals_one_l2731_273147


namespace maries_socks_l2731_273121

theorem maries_socks (x y z : ℕ) : 
  x + y + z = 15 →
  2 * x + 3 * y + 5 * z = 36 →
  x ≥ 1 →
  y ≥ 1 →
  z ≥ 1 →
  x = 11 := by
sorry

end maries_socks_l2731_273121


namespace earnings_difference_main_theorem_l2731_273187

/-- Given investment ratios, return ratios, and total earnings, 
    calculate the difference between earnings of b and a -/
theorem earnings_difference 
  (invest_a invest_b invest_c : ℚ) 
  (return_a return_b return_c : ℚ) 
  (total_earnings : ℚ) : ℚ :=
  let earnings_a := invest_a * return_a
  let earnings_b := invest_b * return_b
  let earnings_c := invest_c * return_c
  by
    have h1 : invest_b / invest_a = 4 / 3 := by sorry
    have h2 : invest_c / invest_a = 5 / 3 := by sorry
    have h3 : return_b / return_a = 5 / 6 := by sorry
    have h4 : return_c / return_a = 4 / 6 := by sorry
    have h5 : earnings_a + earnings_b + earnings_c = total_earnings := by sorry
    have h6 : total_earnings = 4350 := by sorry
    exact 150

/-- The main theorem stating the difference in earnings -/
theorem main_theorem : earnings_difference 3 4 5 6 5 4 4350 = 150 := by sorry

end earnings_difference_main_theorem_l2731_273187


namespace probability_convex_quadrilateral_l2731_273186

-- Define the number of points on the circle
def num_points : ℕ := 6

-- Define the number of chords to be selected
def num_chords : ℕ := 4

-- Define the total number of possible chords
def total_chords : ℕ := num_points.choose 2

-- Define the number of ways to select chords
def ways_to_select_chords : ℕ := total_chords.choose num_chords

-- Define the number of ways to form a convex quadrilateral
def convex_quadrilaterals : ℕ := num_points.choose 4

-- State the theorem
theorem probability_convex_quadrilateral :
  (convex_quadrilaterals : ℚ) / ways_to_select_chords = 1 / 91 := by
  sorry

end probability_convex_quadrilateral_l2731_273186


namespace number_divisibility_l2731_273110

theorem number_divisibility (x : ℝ) : (x / 6) * 12 = 18 → x = 9 := by
  sorry

end number_divisibility_l2731_273110


namespace car_repair_cost_proof_l2731_273114

/-- Calculates the total cost for a car repair given the hourly rate, hours worked per day,
    number of days worked, and cost of parts. -/
def total_repair_cost (hourly_rate : ℕ) (hours_per_day : ℕ) (days_worked : ℕ) (parts_cost : ℕ) : ℕ :=
  hourly_rate * hours_per_day * days_worked + parts_cost

/-- Proves that given the specified conditions, the total cost for the car's owner is $9220. -/
theorem car_repair_cost_proof :
  total_repair_cost 60 8 14 2500 = 9220 := by
  sorry

end car_repair_cost_proof_l2731_273114


namespace jose_share_of_profit_l2731_273193

/-- Calculates the share of profit for an investor given the total profit and investment ratios. -/
def calculate_share_of_profit (total_profit : ℚ) (investor_ratio : ℚ) (total_ratio : ℚ) : ℚ :=
  (investor_ratio / total_ratio) * total_profit

/-- Represents the problem of calculating Jose's share of profit in a business partnership. -/
theorem jose_share_of_profit 
  (tom_investment : ℚ) (tom_duration : ℕ) 
  (jose_investment : ℚ) (jose_duration : ℕ) 
  (total_profit : ℚ) : 
  tom_investment = 30000 → 
  tom_duration = 12 → 
  jose_investment = 45000 → 
  jose_duration = 10 → 
  total_profit = 54000 → 
  calculate_share_of_profit total_profit (jose_investment * jose_duration) 
    (tom_investment * tom_duration + jose_investment * jose_duration) = 30000 := by
  sorry

end jose_share_of_profit_l2731_273193


namespace hiker_distance_hiker_distance_proof_l2731_273170

/-- The straight-line distance a hiker travels after walking 8 miles east,
    turning 45 degrees north, and walking another 8 miles. -/
theorem hiker_distance : ℝ :=
  let initial_east_distance : ℝ := 8
  let turn_angle : ℝ := 45
  let second_walk_distance : ℝ := 8
  let final_distance : ℝ := 4 * Real.sqrt (6 + 4 * Real.sqrt 2)
  final_distance

/-- Proof that the hiker's final straight-line distance from the starting point
    is 4√(6 + 4√2) miles. -/
theorem hiker_distance_proof :
  hiker_distance = 4 * Real.sqrt (6 + 4 * Real.sqrt 2) := by
  sorry

end hiker_distance_hiker_distance_proof_l2731_273170


namespace circle_symmetry_l2731_273159

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 7 = 0

-- Define a line in the plane
def line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define symmetry with respect to a line
def symmetric_wrt_line (C1 C2 : (ℝ → ℝ → Prop)) (l : ℝ → ℝ → Prop) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ), C1 x1 y1 ∧ C2 x2 y2 → 
    ∃ (x y : ℝ), l x y ∧ 
      (x - x1)^2 + (y - y1)^2 = (x - x2)^2 + (y - y2)^2

-- Theorem statement
theorem circle_symmetry :
  symmetric_wrt_line circle1 circle2 (line 1 (-1) 2) := by sorry

end circle_symmetry_l2731_273159


namespace combined_cost_price_is_430_95_l2731_273151

-- Define the parameters for each stock
def stock1_face_value : ℝ := 100
def stock1_discount_rate : ℝ := 0.04
def stock1_brokerage_rate : ℝ := 0.002

def stock2_face_value : ℝ := 200
def stock2_discount_rate : ℝ := 0.06
def stock2_brokerage_rate : ℝ := 0.0025

def stock3_face_value : ℝ := 150
def stock3_discount_rate : ℝ := 0.03
def stock3_brokerage_rate : ℝ := 0.005

-- Define a function to calculate the cost price of a stock
def cost_price (face_value discount_rate brokerage_rate : ℝ) : ℝ :=
  (face_value - face_value * discount_rate) + face_value * brokerage_rate

-- Define the total cost price
def total_cost_price : ℝ :=
  cost_price stock1_face_value stock1_discount_rate stock1_brokerage_rate +
  cost_price stock2_face_value stock2_discount_rate stock2_brokerage_rate +
  cost_price stock3_face_value stock3_discount_rate stock3_brokerage_rate

-- Theorem statement
theorem combined_cost_price_is_430_95 :
  total_cost_price = 430.95 := by
  sorry

end combined_cost_price_is_430_95_l2731_273151


namespace equipment_maintenance_cost_calculation_l2731_273164

def equipment_maintenance_cost (initial_balance cheque_payment received_payment final_balance : ℕ) : ℕ :=
  initial_balance - cheque_payment + received_payment - final_balance

theorem equipment_maintenance_cost_calculation :
  equipment_maintenance_cost 2000 600 800 1000 = 1200 := by
  sorry

end equipment_maintenance_cost_calculation_l2731_273164


namespace no_decreasing_nat_function_exists_decreasing_int_function_l2731_273118

-- Define φ as a function from ℕ to ℕ
variable (φ : ℕ → ℕ)

-- Theorem 1: No such function f : ℕ → ℕ exists
theorem no_decreasing_nat_function : 
  ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f x > f (φ x) := by sorry

-- Theorem 2: Such a function f : ℕ → ℤ exists
theorem exists_decreasing_int_function : 
  ∃ f : ℕ → ℤ, ∀ x : ℕ, f x > f (φ x) := by sorry

end no_decreasing_nat_function_exists_decreasing_int_function_l2731_273118


namespace arithmetic_sequence_a15_l2731_273128

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a15
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 3 + a 13 = 20)
  (h_a2 : a 2 = -2) :
  a 15 = 24 := by
sorry

end arithmetic_sequence_a15_l2731_273128


namespace inequality_solution_set_l2731_273160

theorem inequality_solution_set (x : ℝ) : (1 - x > x - 1) ↔ (x < 1) := by
  sorry

end inequality_solution_set_l2731_273160


namespace apples_in_crate_l2731_273137

/-- The number of apples in a crate -/
def apples_per_crate : ℕ := sorry

/-- The number of crates delivered -/
def crates_delivered : ℕ := 12

/-- The number of rotten apples -/
def rotten_apples : ℕ := 4

/-- The number of apples that fit in each box -/
def apples_per_box : ℕ := 10

/-- The number of boxes filled with good apples -/
def filled_boxes : ℕ := 50

theorem apples_in_crate :
  apples_per_crate * crates_delivered = filled_boxes * apples_per_box + rotten_apples ∧
  apples_per_crate = 42 := by sorry

end apples_in_crate_l2731_273137


namespace divisibility_problem_l2731_273146

theorem divisibility_problem (x y : ℤ) (h : 5 ∣ (x + 9*y)) : 5 ∣ (8*x + 7*y) := by
  sorry

end divisibility_problem_l2731_273146


namespace prime_difference_theorem_l2731_273132

theorem prime_difference_theorem (m n : ℕ) : 
  Nat.Prime m → Nat.Prime n → m - n^2 = 2007 → m * n = 4022 := by
  sorry

end prime_difference_theorem_l2731_273132


namespace nine_times_2010_equals_201_l2731_273134

-- Define the operation
def diamond (a b : ℚ) : ℚ := (a * b) / (a + b)

-- Define a function that applies the operation n times
def apply_n_times (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => diamond (apply_n_times n x) x

-- Theorem statement
theorem nine_times_2010_equals_201 :
  apply_n_times 9 2010 = 201 := by sorry

end nine_times_2010_equals_201_l2731_273134


namespace youngest_child_age_l2731_273191

/-- Given 5 children born at intervals of 3 years, if the sum of their ages is 50 years,
    then the age of the youngest child is 4 years. -/
theorem youngest_child_age (children : ℕ) (interval : ℕ) (total_age : ℕ) :
  children = 5 →
  interval = 3 →
  total_age = 50 →
  total_age = (children - 1) * children / 2 * interval + children * (youngest_age : ℕ) →
  youngest_age = 4 := by
  sorry

end youngest_child_age_l2731_273191


namespace fraction_simplification_l2731_273179

theorem fraction_simplification :
  (30 : ℚ) / 45 * 75 / 128 * 256 / 150 = 1 / 6 := by
  sorry

end fraction_simplification_l2731_273179


namespace division_remainder_problem_l2731_273125

theorem division_remainder_problem (a b q r : ℕ) 
  (h1 : a - b = 1335)
  (h2 : a = 1584)
  (h3 : a = q * b + r)
  (h4 : q = 6)
  (h5 : r < b) :
  r = 90 := by
  sorry

end division_remainder_problem_l2731_273125


namespace min_box_value_l2731_273140

theorem min_box_value (a b : ℤ) (box : ℤ) :
  (∀ x : ℝ, (a * x + b) * (b * x + a) = 30 * x^2 + box * x + 30) →
  a ≠ b ∧ a ≠ box ∧ b ≠ box →
  ∃ (min_box : ℤ), (min_box = 61 ∧ box ≥ min_box) := by
  sorry

end min_box_value_l2731_273140


namespace defective_probability_l2731_273104

/-- The probability of a randomly chosen unit being defective in a factory with two machines --/
theorem defective_probability (total_output : ℝ) (machine_a_output : ℝ) (machine_b_output : ℝ)
  (machine_a_defective_rate : ℝ) (machine_b_defective_rate : ℝ) :
  machine_a_output = 0.4 * total_output →
  machine_b_output = 0.6 * total_output →
  machine_a_defective_rate = 9 / 1000 →
  machine_b_defective_rate = 1 / 50 →
  (machine_a_output / total_output) * machine_a_defective_rate +
  (machine_b_output / total_output) * machine_b_defective_rate = 0.0156 := by
  sorry


end defective_probability_l2731_273104


namespace negation_of_universal_proposition_l2731_273181

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 5*x = 4) ↔ (∃ x : ℝ, x^2 + 5*x ≠ 4) := by
  sorry

end negation_of_universal_proposition_l2731_273181


namespace derivative_value_at_five_l2731_273190

theorem derivative_value_at_five (f : ℝ → ℝ) (hf : ∀ x, f x = 3 * x^2 + 2 * x * (deriv f 2)) :
  deriv f 5 = 6 := by
  sorry

end derivative_value_at_five_l2731_273190


namespace pet_calculation_l2731_273131

theorem pet_calculation (taylor_pets : ℕ) (total_pets : ℕ) : 
  taylor_pets = 4 → 
  total_pets = 32 → 
  ∃ (other_friends_pets : ℕ),
    total_pets = taylor_pets + 3 * (2 * taylor_pets) + 2 * other_friends_pets ∧ 
    other_friends_pets = 2 := by
  sorry

end pet_calculation_l2731_273131


namespace time_BC_is_five_hours_l2731_273185

/-- Represents the train's journey between stations A, B, and C -/
structure TrainJourney where
  M : ℝ  -- Distance unit
  speed : ℝ  -- Constant speed of the train
  time_AB : ℝ  -- Time from A to B
  dist_AC : ℝ  -- Total distance from A to C

/-- The theorem stating the time taken from B to C -/
theorem time_BC_is_five_hours (journey : TrainJourney) 
  (h1 : journey.time_AB = 7)
  (h2 : journey.dist_AC = 6 * journey.M)
  : ∃ (time_BC : ℝ), time_BC = 5 := by
  sorry

end time_BC_is_five_hours_l2731_273185


namespace sum_of_products_l2731_273199

theorem sum_of_products (a b c d : ℝ) 
  (eq1 : a + b + c = 1)
  (eq2 : a + b + d = 5)
  (eq3 : a + c + d = 14)
  (eq4 : b + c + d = 9) :
  a * b + c * d = 338 / 9 := by
sorry

end sum_of_products_l2731_273199


namespace opposite_of_negative_two_l2731_273113

theorem opposite_of_negative_two :
  ∀ x : ℤ, x + (-2) = 0 → x = 2 := by
  sorry

end opposite_of_negative_two_l2731_273113


namespace solution_set_ln_inequality_l2731_273136

theorem solution_set_ln_inequality :
  {x : ℝ | x > 0 ∧ 2 - Real.log x ≥ 0} = Set.Ioo 0 (Real.exp 2) := by sorry

end solution_set_ln_inequality_l2731_273136


namespace alice_max_plates_l2731_273148

/-- Represents the shopping problem with pans, pots, and plates. -/
structure Shopping where
  pan_price : ℕ
  pot_price : ℕ
  plate_price : ℕ
  total_budget : ℕ
  min_pans : ℕ
  min_pots : ℕ

/-- Calculates the maximum number of plates that can be bought. -/
def max_plates (s : Shopping) : ℕ :=
  sorry

/-- The shopping problem instance as described in the question. -/
def alice_shopping : Shopping :=
  { pan_price := 3
  , pot_price := 5
  , plate_price := 11
  , total_budget := 100
  , min_pans := 2
  , min_pots := 2
  }

/-- Theorem stating that the maximum number of plates Alice can buy is 7. -/
theorem alice_max_plates :
  max_plates alice_shopping = 7 := by
  sorry

end alice_max_plates_l2731_273148
