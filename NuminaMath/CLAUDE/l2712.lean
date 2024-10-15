import Mathlib

namespace NUMINAMATH_CALUDE_triangle_properties_l2712_271290

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  2 * a * Real.sin (C + π / 6) = b + c →
  B = π / 4 →
  b - a = Real.sqrt 2 - Real.sqrt 3 →
  A = π / 3 ∧
  (1 / 2) * a * b * Real.sin C = (3 + Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2712_271290


namespace NUMINAMATH_CALUDE_range_of_m_range_of_x_l2712_271265

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 1) * (x - 5) ≤ 0
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0

-- Part 1: Range of m when p is a sufficient condition for q
theorem range_of_m : 
  (∀ x, p x → ∀ m, q x m) → 
  ∀ m, m ≥ 4 := 
sorry

-- Part 2: Range of x when m=5, "p ∨ q" is true, and "p ∧ q" is false
theorem range_of_x : 
  ∀ x, (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) → 
  (x ∈ Set.Icc (-4) (-1) ∪ Set.Ioc 5 6) := 
sorry

end NUMINAMATH_CALUDE_range_of_m_range_of_x_l2712_271265


namespace NUMINAMATH_CALUDE_jr_high_selection_theorem_l2712_271208

/-- Represents the structure of a school with different grade levels and classes --/
structure School where
  elem_grades : Nat
  elem_classes_per_grade : Nat
  jr_high_grades : Nat
  jr_high_classes_per_grade : Nat
  high_grades : Nat
  high_classes_per_grade : Nat

/-- Calculates the total number of classes in the school --/
def total_classes (s : School) : Nat :=
  s.elem_grades * s.elem_classes_per_grade +
  s.jr_high_grades * s.jr_high_classes_per_grade +
  s.high_grades * s.high_classes_per_grade

/-- Calculates the number of classes to be selected from each grade in junior high --/
def jr_high_classes_selected (s : School) (total_selected : Nat) : Nat :=
  (total_selected * s.jr_high_classes_per_grade) / (total_classes s)

theorem jr_high_selection_theorem (s : School) (total_selected : Nat) :
  s.elem_grades = 6 →
  s.elem_classes_per_grade = 6 →
  s.jr_high_grades = 3 →
  s.jr_high_classes_per_grade = 8 →
  s.high_grades = 3 →
  s.high_classes_per_grade = 12 →
  total_selected = 36 →
  jr_high_classes_selected s total_selected = 2 := by
  sorry

end NUMINAMATH_CALUDE_jr_high_selection_theorem_l2712_271208


namespace NUMINAMATH_CALUDE_gcd_binomial_divisibility_l2712_271282

theorem gcd_binomial_divisibility (m n : ℕ) (h1 : 0 < m) (h2 : m ≤ n) : 
  ∃ k : ℤ, (Int.gcd m n : ℚ) / n * (n.choose m) = k := by
  sorry

end NUMINAMATH_CALUDE_gcd_binomial_divisibility_l2712_271282


namespace NUMINAMATH_CALUDE_sum_of_digits_of_product_80_sevens_80_threes_l2712_271212

/-- A number consisting of n repeated digits d -/
def repeated_digit (n : ℕ) (d : ℕ) : ℕ := sorry

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_product_80_sevens_80_threes : 
  sum_of_digits (repeated_digit 80 7 * repeated_digit 80 3) = 240 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_product_80_sevens_80_threes_l2712_271212


namespace NUMINAMATH_CALUDE_certain_number_proof_l2712_271238

theorem certain_number_proof (x : ℝ) : x / 14.5 = 177 → x = 2566.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2712_271238


namespace NUMINAMATH_CALUDE_sum_of_multiples_l2712_271200

def largest_three_digit_multiple_of_4 : ℕ := 996

def smallest_four_digit_multiple_of_3 : ℕ := 1002

theorem sum_of_multiples : 
  largest_three_digit_multiple_of_4 + smallest_four_digit_multiple_of_3 = 1998 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l2712_271200


namespace NUMINAMATH_CALUDE_triangle_sine_comparison_l2712_271267

/-- For a triangle ABC, compare the sum of reciprocals of sines of doubled angles
    with the sum of reciprocals of sines of angles. -/
theorem triangle_sine_comparison (A B C : ℝ) (h_triangle : A + B + C = π) :
  (A > π / 2 ∨ B > π / 2 ∨ C > π / 2) →
    1 / Real.sin (2 * A) + 1 / Real.sin (2 * B) + 1 / Real.sin (2 * C) <
    1 / Real.sin A + 1 / Real.sin B + 1 / Real.sin C ∧
  (A ≤ π / 2 ∧ B ≤ π / 2 ∧ C ≤ π / 2) →
    1 / Real.sin (2 * A) + 1 / Real.sin (2 * B) + 1 / Real.sin (2 * C) ≥
    1 / Real.sin A + 1 / Real.sin B + 1 / Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_comparison_l2712_271267


namespace NUMINAMATH_CALUDE_decimal_point_problem_l2712_271268

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 / x) : x = Real.sqrt 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l2712_271268


namespace NUMINAMATH_CALUDE_thirty_percent_passed_l2712_271299

/-- The swim club scenario -/
structure SwimClub where
  total_members : ℕ
  not_passed_with_course : ℕ
  not_passed_without_course : ℕ

/-- Calculate the percentage of members who passed the lifesaving test -/
def passed_percentage (club : SwimClub) : ℚ :=
  1 - (club.not_passed_with_course + club.not_passed_without_course : ℚ) / club.total_members

/-- The theorem stating that 30% of members passed the test -/
theorem thirty_percent_passed (club : SwimClub) 
  (h1 : club.total_members = 50)
  (h2 : club.not_passed_with_course = 5)
  (h3 : club.not_passed_without_course = 30) : 
  passed_percentage club = 30 / 100 := by
  sorry

#eval passed_percentage ⟨50, 5, 30⟩

end NUMINAMATH_CALUDE_thirty_percent_passed_l2712_271299


namespace NUMINAMATH_CALUDE_algebraic_simplification_l2712_271217

theorem algebraic_simplification (m n : ℝ) :
  9 * m^2 - 4 * (2 * m^2 - 3 * m * n + n^2) + 4 * n^2 = m^2 + 12 * m * n := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l2712_271217


namespace NUMINAMATH_CALUDE_net_effect_on_revenue_l2712_271243

theorem net_effect_on_revenue 
  (original_price original_sales : ℝ) 
  (price_reduction : ℝ) 
  (sales_increase : ℝ) 
  (h1 : price_reduction = 0.2) 
  (h2 : sales_increase = 0.8) : 
  let new_price := original_price * (1 - price_reduction)
  let new_sales := original_sales * (1 + sales_increase)
  let original_revenue := original_price * original_sales
  let new_revenue := new_price * new_sales
  (new_revenue - original_revenue) / original_revenue = 0.44 := by
sorry

end NUMINAMATH_CALUDE_net_effect_on_revenue_l2712_271243


namespace NUMINAMATH_CALUDE_simplify_fraction_l2712_271281

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) :
  (x^2 + x) / (x^2 - 2*x + 1) / ((x + 1) / (x - 1)) = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2712_271281


namespace NUMINAMATH_CALUDE_dimes_in_jar_l2712_271272

/-- Represents the number of coins of each type in the jar -/
structure CoinCount where
  dimes : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.dimes * 10 + coins.quarters * 25

/-- Theorem stating that given the conditions, there are 15 dimes in the jar -/
theorem dimes_in_jar : ∃ (coins : CoinCount),
  coins.dimes = 3 * coins.quarters / 2 ∧
  totalValue coins = 400 ∧
  coins.dimes = 15 := by
  sorry

end NUMINAMATH_CALUDE_dimes_in_jar_l2712_271272


namespace NUMINAMATH_CALUDE_gwen_birthday_money_l2712_271287

/-- The amount of money Gwen received from her mom -/
def mom_money : ℕ := 8

/-- The difference between the money Gwen received from her mom and dad -/
def difference : ℕ := 3

/-- The amount of money Gwen received from her dad -/
def dad_money : ℕ := mom_money - difference

theorem gwen_birthday_money : dad_money = 5 := by
  sorry

end NUMINAMATH_CALUDE_gwen_birthday_money_l2712_271287


namespace NUMINAMATH_CALUDE_difference_d_minus_b_l2712_271241

theorem difference_d_minus_b (a b c d : ℕ+) 
  (h1 : a^5 = b^4) 
  (h2 : c^3 = d^2) 
  (h3 : c - a = 19) : 
  d - b = 757 := by sorry

end NUMINAMATH_CALUDE_difference_d_minus_b_l2712_271241


namespace NUMINAMATH_CALUDE_dagger_example_l2712_271283

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (q / n)

-- Theorem statement
theorem dagger_example : dagger (7/12) (8/3) = 14 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l2712_271283


namespace NUMINAMATH_CALUDE_total_spent_is_72_l2712_271271

/-- The cost of a single trick deck in dollars -/
def deck_cost : ℕ := 9

/-- The number of decks Edward bought -/
def edward_decks : ℕ := 4

/-- The number of decks Edward's friend bought -/
def friend_decks : ℕ := 4

/-- The total amount spent by Edward and his friend -/
def total_spent : ℕ := deck_cost * (edward_decks + friend_decks)

theorem total_spent_is_72 : total_spent = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_72_l2712_271271


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l2712_271288

theorem quadratic_rewrite (b : ℝ) : 
  (∃ n : ℝ, ∀ x : ℝ, x^2 + b*x + 72 = (x + n)^2 + 20) → 
  (b = 4 * Real.sqrt 13 ∨ b = -4 * Real.sqrt 13) := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l2712_271288


namespace NUMINAMATH_CALUDE_parabola_vertex_l2712_271289

-- Define the parabola
def f (x : ℝ) : ℝ := -3 * (x + 1)^2 + 1

-- State the theorem
theorem parabola_vertex : 
  ∃ (x y : ℝ), (∀ t : ℝ, f t ≤ f x) ∧ y = f x ∧ x = -1 ∧ y = 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2712_271289


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_range_l2712_271242

def integer_range : List Int := List.range 10 |>.map (λ x => x - 3)

theorem arithmetic_mean_of_range : 
  (integer_range.sum : ℚ) / integer_range.length = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_range_l2712_271242


namespace NUMINAMATH_CALUDE_parabola_vertex_relationship_l2712_271240

/-- Given a parabola y = x^2 - 2mx + 2m^2 - 3m + 1, prove that the functional relationship
    between the vertical coordinate y and the horizontal coordinate x of its vertex
    is y = x^2 - 3x + 1, regardless of the value of m. -/
theorem parabola_vertex_relationship (m x y : ℝ) :
  y = x^2 - 2*m*x + 2*m^2 - 3*m + 1 →
  (x = m ∧ y = m^2 - 3*m + 1) →
  y = x^2 - 3*x + 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_relationship_l2712_271240


namespace NUMINAMATH_CALUDE_amanda_camila_hike_ratio_l2712_271286

/-- Proves that the ratio of Amanda's hikes to Camila's hikes is 8:1 --/
theorem amanda_camila_hike_ratio :
  let camila_hikes : ℕ := 7
  let steven_hikes : ℕ := camila_hikes + 4 * 16
  let amanda_hikes : ℕ := steven_hikes - 15
  amanda_hikes / camila_hikes = 8 := by
sorry

end NUMINAMATH_CALUDE_amanda_camila_hike_ratio_l2712_271286


namespace NUMINAMATH_CALUDE_qqLive_higher_score_l2712_271239

structure SoftwareRating where
  name : String
  studentRatings : List Nat
  studentAverage : Float
  teacherAverage : Float

def comprehensiveScore (rating : SoftwareRating) : Float :=
  rating.studentAverage * 0.4 + rating.teacherAverage * 0.6

def dingtalk : SoftwareRating := {
  name := "DingTalk",
  studentRatings := [1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5],
  studentAverage := 3.4,
  teacherAverage := 3.9
}

def qqLive : SoftwareRating := {
  name := "QQ Live",
  studentRatings := [1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5],
  studentAverage := 3.35,
  teacherAverage := 4.0
}

theorem qqLive_higher_score : comprehensiveScore qqLive > comprehensiveScore dingtalk := by
  sorry

end NUMINAMATH_CALUDE_qqLive_higher_score_l2712_271239


namespace NUMINAMATH_CALUDE_distribute_six_students_two_activities_l2712_271280

/-- The number of ways to distribute n students between 2 activities,
    where each activity can have at most k students. -/
def distribute_students (n k : ℕ) : ℕ :=
  Nat.choose n k + Nat.choose n (n / 2)

/-- Theorem stating that the number of ways to distribute 6 students
    between 2 activities, where each activity can have at most 4 students,
    is equal to 35. -/
theorem distribute_six_students_two_activities :
  distribute_students 6 4 = 35 := by
  sorry

#eval distribute_students 6 4

end NUMINAMATH_CALUDE_distribute_six_students_two_activities_l2712_271280


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l2712_271291

/-- Represents a batsman's score data -/
structure BatsmanScore where
  initialAverage : ℝ
  inningsPlayed : ℕ
  newInningScore : ℝ
  averageIncrease : ℝ

/-- Theorem: If a batsman's average increases by 5 after scoring 100 in the 11th inning, 
    then his new average is 50 -/
theorem batsman_average_theorem (b : BatsmanScore) 
  (h1 : b.inningsPlayed = 10)
  (h2 : b.newInningScore = 100)
  (h3 : b.averageIncrease = 5)
  : b.initialAverage + b.averageIncrease = 50 := by
  sorry

#check batsman_average_theorem

end NUMINAMATH_CALUDE_batsman_average_theorem_l2712_271291


namespace NUMINAMATH_CALUDE_power_calculation_l2712_271221

theorem power_calculation (m n : ℕ) (h1 : 2^m = 3) (h2 : 4^n = 8) :
  2^(3*m - 2*n + 3) = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2712_271221


namespace NUMINAMATH_CALUDE_percent_equality_l2712_271232

theorem percent_equality (x : ℝ) : (75 / 100 * 600 = 50 / 100 * x) → x = 900 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l2712_271232


namespace NUMINAMATH_CALUDE_soccer_games_played_l2712_271274

theorem soccer_games_played (win_percentage : ℝ) (games_won : ℝ) (total_games : ℝ) : 
  win_percentage = 0.40 → games_won = 63.2 → win_percentage * total_games = games_won → total_games = 158 := by
  sorry

end NUMINAMATH_CALUDE_soccer_games_played_l2712_271274


namespace NUMINAMATH_CALUDE_function_range_l2712_271275

/-- The range of the function f(x) = (e^(3x) - 2) / (e^(3x) + 2) is (-1, 1) -/
theorem function_range (x : ℝ) : 
  -1 < (Real.exp (3 * x) - 2) / (Real.exp (3 * x) + 2) ∧ 
  (Real.exp (3 * x) - 2) / (Real.exp (3 * x) + 2) < 1 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l2712_271275


namespace NUMINAMATH_CALUDE_ellipse_equation_l2712_271277

/-- The equation of an ellipse with given properties -/
theorem ellipse_equation (e : ℝ) (c : ℝ) (h1 : e = 1/2) (h2 : c = 1) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), (x^2 / (a^2) + y^2 / (b^2) = 1) ↔ (x^2 / 4 + y^2 / 3 = 1)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2712_271277


namespace NUMINAMATH_CALUDE_expression_simplification_l2712_271229

theorem expression_simplification (b x : ℝ) 
  (hb : b ≠ 0) (hx : x ≠ 0) (hxb : x ≠ b/2) (hxb2 : x ≠ -2/b) :
  ((b*x + 4 + 4/(b*x)) / (2*b + (b^2 - 4)*x - 2*b*x^2) + 
   ((4*x^2 - b^2) / b) / ((b + 2*x)^2 - 8*b*x)) * (b*x/2) = 
  (x^2 - 1) / (2*x - b) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l2712_271229


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2712_271296

theorem complex_fraction_equality (z : ℂ) (h : z = 2 + I) : 
  (2 * I) / (z - 1) = 1 + I := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2712_271296


namespace NUMINAMATH_CALUDE_isabellas_final_hair_length_l2712_271262

/-- The final hair length given an initial length and growth --/
def finalHairLength (initialLength growth : ℕ) : ℕ :=
  initialLength + growth

/-- Theorem: Isabella's final hair length is 24 inches --/
theorem isabellas_final_hair_length :
  finalHairLength 18 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_final_hair_length_l2712_271262


namespace NUMINAMATH_CALUDE_cone_height_for_right_angle_vertex_l2712_271247

/-- Represents a cone with given volume and vertex angle -/
structure Cone where
  volume : ℝ
  vertexAngle : ℝ

/-- The height of a cone given its volume and vertex angle -/
def coneHeight (c : Cone) : ℝ :=
  sorry

theorem cone_height_for_right_angle_vertex (c : Cone) 
  (h_volume : c.volume = 20000 * Real.pi)
  (h_angle : c.vertexAngle = Real.pi / 2) :
  ∃ (r : ℝ), coneHeight c = r * Real.sqrt 2 ∧ 
  r^3 * Real.sqrt 2 = 60000 :=
sorry

end NUMINAMATH_CALUDE_cone_height_for_right_angle_vertex_l2712_271247


namespace NUMINAMATH_CALUDE_equation_solution_l2712_271228

theorem equation_solution :
  ∃! x : ℝ, x ≠ 3 ∧ x + 60 / (x - 3) = -13 :=
by
  -- The unique solution is x = -7
  use -7
  constructor
  · -- Prove that x = -7 satisfies the equation
    constructor
    · -- Prove -7 ≠ 3
      linarith
    · -- Prove -7 + 60 / (-7 - 3) = -13
      ring
  · -- Prove uniqueness
    intro y hy
    -- Assume y satisfies the equation
    have h1 : y ≠ 3 := hy.1
    have h2 : y + 60 / (y - 3) = -13 := hy.2
    -- Derive that y must equal -7
    sorry


end NUMINAMATH_CALUDE_equation_solution_l2712_271228


namespace NUMINAMATH_CALUDE_negation_existence_inequality_l2712_271264

theorem negation_existence_inequality :
  (¬ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_existence_inequality_l2712_271264


namespace NUMINAMATH_CALUDE_slices_in_large_pizza_l2712_271273

/-- Given that Mary orders 2 large pizzas, eats 7 slices, and has 9 slices remaining,
    prove that there are 8 slices in a large pizza. -/
theorem slices_in_large_pizza :
  ∀ (total_pizzas : ℕ) (slices_eaten : ℕ) (slices_remaining : ℕ),
    total_pizzas = 2 →
    slices_eaten = 7 →
    slices_remaining = 9 →
    (slices_remaining + slices_eaten) / total_pizzas = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_slices_in_large_pizza_l2712_271273


namespace NUMINAMATH_CALUDE_snow_at_brecknock_l2712_271235

/-- The amount of snow at Mrs. Hilt's house in inches -/
def mrs_hilt_snow : ℕ := 29

/-- The difference in snow between Mrs. Hilt's house and Brecknock Elementary School in inches -/
def snow_difference : ℕ := 12

/-- The amount of snow at Brecknock Elementary School in inches -/
def brecknock_snow : ℕ := mrs_hilt_snow - snow_difference

theorem snow_at_brecknock : brecknock_snow = 17 := by
  sorry

end NUMINAMATH_CALUDE_snow_at_brecknock_l2712_271235


namespace NUMINAMATH_CALUDE_non_similar_500_pointed_stars_l2712_271206

/-- A regular n-pointed star is the union of n line segments. -/
def RegularStar (n : ℕ) (m : ℕ) : Prop :=
  (n > 0) ∧ (m > 0) ∧ (m < n) ∧ (Nat.gcd m n = 1)

/-- Two stars are similar if they have the same number of points and
    their m values are either equal or complementary modulo n. -/
def SimilarStars (n : ℕ) (m1 m2 : ℕ) : Prop :=
  RegularStar n m1 ∧ RegularStar n m2 ∧ (m1 = m2 ∨ m1 + m2 = n)

/-- The number of non-similar regular n-pointed stars -/
def NonSimilarStarCount (n : ℕ) : ℕ :=
  (Nat.totient n - 2) / 2 + 1

theorem non_similar_500_pointed_stars :
  NonSimilarStarCount 500 = 99 := by
  sorry

#eval NonSimilarStarCount 500  -- This should output 99

end NUMINAMATH_CALUDE_non_similar_500_pointed_stars_l2712_271206


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l2712_271225

theorem fifteenth_student_age
  (total_students : ℕ)
  (total_average_age : ℝ)
  (group1_students : ℕ)
  (group1_average_age : ℝ)
  (group2_students : ℕ)
  (group2_average_age : ℝ)
  (h1 : total_students = 15)
  (h2 : total_average_age = 15)
  (h3 : group1_students = 4)
  (h4 : group1_average_age = 14)
  (h5 : group2_students = 10)
  (h6 : group2_average_age = 16)
  : ℝ := by
  sorry

#check fifteenth_student_age

end NUMINAMATH_CALUDE_fifteenth_student_age_l2712_271225


namespace NUMINAMATH_CALUDE_acid_dilution_l2712_271270

/-- Given an initial acid solution of m ounces at m% concentration, 
    prove that adding x ounces of water to reach (m-15)% concentration
    results in x = 15m / (m-15) for m > 30 -/
theorem acid_dilution (m : ℝ) (x : ℝ) (h₁ : m > 30) :
  (m * m / 100 = (m - 15) * (m + x) / 100) → x = 15 * m / (m - 15) := by
  sorry


end NUMINAMATH_CALUDE_acid_dilution_l2712_271270


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2712_271244

-- Define set P
def P : Set ℝ := {x | x^2 - x - 2 ≥ 0}

-- Define set Q
def Q : Set ℝ := {y | ∃ x ∈ P, y = (1/2) * x^2 - 1}

-- Theorem statement
theorem intersection_of_P_and_Q :
  P ∩ Q = {m : ℝ | m ≥ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2712_271244


namespace NUMINAMATH_CALUDE_max_green_lily_students_l2712_271218

-- Define variables
variable (x : ℝ) -- Cost of green lily
variable (y : ℝ) -- Cost of spider plant
variable (m : ℝ) -- Number of students taking care of green lilies

-- Define conditions
axiom condition1 : 2 * x + 3 * y = 36
axiom condition2 : x + 2 * y = 21
axiom total_students : m + (48 - m) = 48
axiom cost_constraint : m * x + (48 - m) * y ≤ 378

-- Theorem to prove
theorem max_green_lily_students : 
  ∃ m : ℝ, m ≤ 30 ∧ 
  ∀ n : ℝ, (n * x + (48 - n) * y ≤ 378 → n ≤ m) :=
sorry

end NUMINAMATH_CALUDE_max_green_lily_students_l2712_271218


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_neg_one_evaluate_zero_undefined_for_one_undefined_for_two_l2712_271202

theorem simplify_expression (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ 2) :
  (1 + 3 / (a - 1)) / ((a^2 - 4) / (a - 1)) = 1 / (a - 2) := by
  sorry

-- Evaluation for a = -1
theorem evaluate_neg_one :
  (1 + 3 / (-1 - 1)) / ((-1^2 - 4) / (-1 - 1)) = -1/3 := by
  sorry

-- Evaluation for a = 0
theorem evaluate_zero :
  (1 + 3 / (0 - 1)) / ((0^2 - 4) / (0 - 1)) = -1/2 := by
  sorry

-- Undefined for a = 1
theorem undefined_for_one (h : (1 : ℝ) ≠ 2) :
  ¬∃x, (1 + 3 / (1 - 1)) / ((1^2 - 4) / (1 - 1)) = x := by
  sorry

-- Undefined for a = 2
theorem undefined_for_two (h : (2 : ℝ) ≠ 1) :
  ¬∃x, (1 + 3 / (2 - 1)) / ((2^2 - 4) / (2 - 1)) = x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_neg_one_evaluate_zero_undefined_for_one_undefined_for_two_l2712_271202


namespace NUMINAMATH_CALUDE_percentage_problem_l2712_271224

theorem percentage_problem (x : ℝ) : 0.15 * 0.30 * 0.50 * x = 99 → x = 4400 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2712_271224


namespace NUMINAMATH_CALUDE_apples_added_to_pile_l2712_271279

/-- Given an initial pile of apples and a final pile of apples,
    calculate the number of apples added. -/
def applesAdded (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that 5 apples were added to the pile -/
theorem apples_added_to_pile :
  let initial := 8
  let final := 13
  applesAdded initial final = 5 := by sorry

end NUMINAMATH_CALUDE_apples_added_to_pile_l2712_271279


namespace NUMINAMATH_CALUDE_best_of_three_prob_l2712_271210

/-- The probability of winning a single set -/
def p : ℝ := 0.6

/-- The probability of winning a best of 3 sets match -/
def match_win_prob : ℝ := p^2 + 3 * p^2 * (1 - p)

theorem best_of_three_prob : match_win_prob = 0.648 := by sorry

end NUMINAMATH_CALUDE_best_of_three_prob_l2712_271210


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l2712_271255

/-- Definition of the ⋄ operation -/
noncomputable def diamond (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

/-- Theorem stating that if 3 ⋄ y = 12, then y = 72 -/
theorem diamond_equation_solution :
  ∃ y : ℝ, diamond 3 y = 12 ∧ y = 72 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l2712_271255


namespace NUMINAMATH_CALUDE_polynomial_coefficients_sum_l2712_271284

theorem polynomial_coefficients_sum (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, a₁ * (x - 1)^4 + a₂ * (x - 1)^3 + a₃ * (x - 1)^2 + a₄ * (x - 1) + a₅ = x^4) →
  a₂ + a₃ + a₄ = 14 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_sum_l2712_271284


namespace NUMINAMATH_CALUDE_simplified_multiplication_l2712_271254

def factor1 : Nat := 20213
def factor2 : Nat := 732575

theorem simplified_multiplication (f1 f2 : Nat) (h1 : f1 = factor1) (h2 : f2 = factor2) :
  ∃ (partial_products : List Nat),
    f1 * f2 = partial_products.sum ∧
    partial_products.length < 5 :=
sorry

end NUMINAMATH_CALUDE_simplified_multiplication_l2712_271254


namespace NUMINAMATH_CALUDE_extreme_point_iff_a_eq_zero_l2712_271213

/-- The function f(x) as defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2 - (a + 2) * x + 1

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * x - (a + 2)

/-- Definition of an extreme point -/
def is_extreme_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), y ≠ x → f y ≥ f x ∨ f y ≤ f x

/-- The main theorem stating that x=1 is an extreme point of f(x) iff a=0 -/
theorem extreme_point_iff_a_eq_zero (a : ℝ) :
  is_extreme_point (f a) 1 ↔ a = 0 := by sorry

end NUMINAMATH_CALUDE_extreme_point_iff_a_eq_zero_l2712_271213


namespace NUMINAMATH_CALUDE_linear_function_not_in_second_quadrant_l2712_271220

/-- A linear function f(x) = mx + b does not pass through the second quadrant
    if its slope m is positive and its y-intercept b is negative. -/
theorem linear_function_not_in_second_quadrant 
  (f : ℝ → ℝ) (m b : ℝ) (h1 : ∀ x, f x = m * x + b) (h2 : m > 0) (h3 : b < 0) :
  ∃ x y, f x = y ∧ (x ≤ 0 ∨ y ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_second_quadrant_l2712_271220


namespace NUMINAMATH_CALUDE_inequality_proof_l2712_271231

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  (a + 1/b)^2 > (b + 1/a)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2712_271231


namespace NUMINAMATH_CALUDE_not_perfect_squares_l2712_271226

theorem not_perfect_squares : ∃ (n : ℕ → ℕ), 
  (n 1 = 2048) ∧ 
  (n 2 = 2049) ∧ 
  (n 3 = 2050) ∧ 
  (n 4 = 2051) ∧ 
  (n 5 = 2052) ∧ 
  (∃ (a : ℕ), 1^(n 1) = a^2) ∧ 
  (¬∃ (b : ℕ), 2^(n 2) = b^2) ∧ 
  (∃ (c : ℕ), 3^(n 3) = c^2) ∧ 
  (¬∃ (d : ℕ), 4^(n 4) = d^2) ∧ 
  (∃ (e : ℕ), 5^(n 5) = e^2) :=
by sorry


end NUMINAMATH_CALUDE_not_perfect_squares_l2712_271226


namespace NUMINAMATH_CALUDE_unopened_box_cards_l2712_271203

theorem unopened_box_cards (initial_cards given_away_cards final_total_cards : ℕ) :
  initial_cards = 26 →
  given_away_cards = 18 →
  final_total_cards = 48 →
  final_total_cards = (initial_cards - given_away_cards) + (final_total_cards - (initial_cards - given_away_cards)) :=
by
  sorry

end NUMINAMATH_CALUDE_unopened_box_cards_l2712_271203


namespace NUMINAMATH_CALUDE_rectangular_prism_dimensions_sum_l2712_271250

theorem rectangular_prism_dimensions_sum (A B C : ℝ) 
  (h1 : A * B = 30)
  (h2 : A * C = 40)
  (h3 : B * C = 60) :
  A + B + C = 9 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_dimensions_sum_l2712_271250


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_four_l2712_271260

theorem smallest_four_digit_divisible_by_four : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 4 = 0 → n ≥ 1000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_four_l2712_271260


namespace NUMINAMATH_CALUDE_k_range_l2712_271251

theorem k_range (k : ℝ) : (1 - k > -1 ∧ 1 - k ≤ 3) ↔ -2 ≤ k ∧ k < 2 := by
  sorry

end NUMINAMATH_CALUDE_k_range_l2712_271251


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2712_271223

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) : ℂ).re = a^2 - 1 ∧ (a - 1 ≠ 0) → a = -1 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2712_271223


namespace NUMINAMATH_CALUDE_ultramarathon_training_l2712_271257

theorem ultramarathon_training (initial_time initial_speed : ℝ)
  (time_increase_percent speed_increase : ℝ)
  (h1 : initial_time = 8)
  (h2 : initial_speed = 8)
  (h3 : time_increase_percent = 75)
  (h4 : speed_increase = 4) :
  let new_time := initial_time * (1 + time_increase_percent / 100)
  let new_speed := initial_speed + speed_increase
  new_time * new_speed = 168 := by
  sorry

end NUMINAMATH_CALUDE_ultramarathon_training_l2712_271257


namespace NUMINAMATH_CALUDE_problem_statement_l2712_271234

theorem problem_statement :
  ∀ d : ℕ, (5 ^ 5) * (9 ^ 3) = 3 * d ∧ d = 15 ^ 5 → d = 759375 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2712_271234


namespace NUMINAMATH_CALUDE_tan_alpha_two_implies_expression_equals_one_l2712_271201

theorem tan_alpha_two_implies_expression_equals_one (α : Real) 
  (h : Real.tan α = 2) : 
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_implies_expression_equals_one_l2712_271201


namespace NUMINAMATH_CALUDE_min_correct_answers_for_target_score_l2712_271249

/-- A math competition with specific scoring rules -/
structure MathCompetition where
  total_questions : Nat
  correct_points : Int
  incorrect_points : Int
  unanswered_points : Int

/-- A participant in the math competition -/
structure Participant where
  answered_questions : Nat
  unanswered_questions : Nat

/-- Calculate the score based on the number of correct answers -/
def calculate_score (comp : MathCompetition) (part : Participant) (correct_answers : Nat) : Int :=
  correct_answers * comp.correct_points +
  (part.answered_questions - correct_answers) * comp.incorrect_points +
  part.unanswered_questions * comp.unanswered_points

/-- The main theorem to prove -/
theorem min_correct_answers_for_target_score 
  (comp : MathCompetition)
  (part : Participant)
  (target_score : Int) : Nat :=
  have h1 : comp.total_questions = 30 := by sorry
  have h2 : comp.correct_points = 8 := by sorry
  have h3 : comp.incorrect_points = -2 := by sorry
  have h4 : comp.unanswered_points = 2 := by sorry
  have h5 : part.answered_questions = 25 := by sorry
  have h6 : part.unanswered_questions = 5 := by sorry
  have h7 : target_score = 160 := by sorry

  20

#check min_correct_answers_for_target_score


end NUMINAMATH_CALUDE_min_correct_answers_for_target_score_l2712_271249


namespace NUMINAMATH_CALUDE_wendy_furniture_time_l2712_271233

/-- The time Wendy spent putting together all the furniture -/
def total_time (num_chairs num_tables time_per_piece : ℕ) : ℕ :=
  (num_chairs + num_tables) * time_per_piece

/-- Proof that Wendy spent 48 minutes putting together all the furniture -/
theorem wendy_furniture_time :
  total_time 4 4 6 = 48 := by
  sorry

end NUMINAMATH_CALUDE_wendy_furniture_time_l2712_271233


namespace NUMINAMATH_CALUDE_pie_chart_most_suitable_for_air_l2712_271261

/-- Represents different types of statistical graphs -/
inductive StatGraph
  | PieChart
  | LineChart
  | BarChart

/-- Represents a substance composed of various components -/
structure Substance where
  components : List String

/-- Determines if a statistical graph is suitable for representing the composition of a substance -/
def is_suitable (graph : StatGraph) (substance : Substance) : Prop :=
  match graph with
  | StatGraph.PieChart => substance.components.length > 1
  | _ => False

/-- Air is a substance composed of various gases -/
def air : Substance :=
  { components := ["nitrogen", "oxygen", "argon", "carbon dioxide", "other gases"] }

/-- Theorem stating that a pie chart is the most suitable graph for representing air composition -/
theorem pie_chart_most_suitable_for_air :
  is_suitable StatGraph.PieChart air ∧
  ∀ (graph : StatGraph), graph ≠ StatGraph.PieChart → ¬(is_suitable graph air) :=
sorry

end NUMINAMATH_CALUDE_pie_chart_most_suitable_for_air_l2712_271261


namespace NUMINAMATH_CALUDE_jake_money_left_jake_final_amount_l2712_271297

theorem jake_money_left (initial_amount : ℝ) (motorcycle_percent : ℝ) 
  (concert_percent : ℝ) (investment_percent : ℝ) (investment_loss_percent : ℝ) : ℝ :=
  let after_motorcycle := initial_amount * (1 - motorcycle_percent)
  let after_concert := after_motorcycle * (1 - concert_percent)
  let investment := after_concert * investment_percent
  let investment_loss := investment * investment_loss_percent
  let final_amount := after_concert - investment + (investment - investment_loss)
  final_amount

theorem jake_final_amount : 
  jake_money_left 5000 0.35 0.25 0.40 0.20 = 1462.50 := by
  sorry

end NUMINAMATH_CALUDE_jake_money_left_jake_final_amount_l2712_271297


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l2712_271269

theorem binomial_expansion_problem (n : ℕ) (h : (2 : ℝ)^n = 256) :
  n = 8 ∧ (Nat.choose n (n / 2) : ℝ) = 70 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l2712_271269


namespace NUMINAMATH_CALUDE_min_value_theorem_l2712_271253

theorem min_value_theorem (c : ℝ) (hc : c > 0) (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (heq : a^2 - 2*a*b + 2*b^2 - c = 0) (hmax : ∀ a' b' : ℝ, a'^2 - 2*a'*b' + 2*b'^2 - c = 0 → a' + b' ≤ a + b) :
  ∃ (m : ℝ), m = -1/4 ∧ ∀ a' b' : ℝ, a' ≠ 0 → b' ≠ 0 → a'^2 - 2*a'*b' + 2*b'^2 - c = 0 →
    m ≤ (3/a' - 4/b' + 5/c) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2712_271253


namespace NUMINAMATH_CALUDE_cos_150_degrees_l2712_271248

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l2712_271248


namespace NUMINAMATH_CALUDE_distance_between_points_distance_X_to_Y_l2712_271266

/-- The distance between two points X and Y, given the walking speeds of two people
    and the distance one person has walked when they meet. -/
theorem distance_between_points (yolanda_speed bob_speed : ℝ) 
  (time_difference : ℝ) (bob_distance : ℝ) : ℝ :=
  let total_time := bob_distance / bob_speed
  let yolanda_time := total_time + time_difference
  let yolanda_distance := yolanda_time * yolanda_speed
  bob_distance + yolanda_distance

/-- The specific problem statement -/
theorem distance_X_to_Y : distance_between_points 1 2 1 20 = 31 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_distance_X_to_Y_l2712_271266


namespace NUMINAMATH_CALUDE_water_consumption_per_person_per_hour_l2712_271292

theorem water_consumption_per_person_per_hour 
  (num_people : ℕ) 
  (total_hours : ℕ) 
  (total_bottles : ℕ) 
  (h1 : num_people = 4) 
  (h2 : total_hours = 16) 
  (h3 : total_bottles = 32) : 
  (total_bottles : ℚ) / total_hours / num_people = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_per_person_per_hour_l2712_271292


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l2712_271263

theorem reciprocal_sum_of_quadratic_roots (γ δ : ℝ) : 
  (∃ r s : ℝ, 7 * r^2 + 5 * r + 3 = 0 ∧ 
              7 * s^2 + 5 * s + 3 = 0 ∧ 
              γ = 1 / r ∧ 
              δ = 1 / s) → 
  γ + δ = -5/3 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_quadratic_roots_l2712_271263


namespace NUMINAMATH_CALUDE_hydroflow_pump_calculation_l2712_271219

/-- The rate at which the Hydroflow system pumps water, in gallons per hour -/
def pump_rate : ℝ := 360

/-- The time in minutes for which we want to calculate the amount of water pumped -/
def pump_time : ℝ := 30

/-- Theorem stating that the Hydroflow system pumps 180 gallons in 30 minutes -/
theorem hydroflow_pump_calculation : 
  pump_rate * (pump_time / 60) = 180 := by sorry

end NUMINAMATH_CALUDE_hydroflow_pump_calculation_l2712_271219


namespace NUMINAMATH_CALUDE_number_divisible_by_nine_missing_digit_correct_l2712_271211

/-- The missing digit in the five-digit number 385_7 that makes it divisible by 9 -/
def missing_digit : ℕ := 4

/-- The five-digit number with the missing digit filled in -/
def number : ℕ := 38547

theorem number_divisible_by_nine :
  number % 9 = 0 :=
sorry

theorem missing_digit_correct :
  ∃ (d : ℕ), d < 10 ∧ 38500 + d * 10 + 7 = number ∧ (38500 + d * 10 + 7) % 9 = 0 → d = missing_digit :=
sorry

end NUMINAMATH_CALUDE_number_divisible_by_nine_missing_digit_correct_l2712_271211


namespace NUMINAMATH_CALUDE_quadrupled_base_exponent_l2712_271215

theorem quadrupled_base_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (4*a)^(4*b) = a^b * x^(2*b) → x = 16 * a^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_quadrupled_base_exponent_l2712_271215


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l2712_271230

/-- Calculates the average runs for a batsman over multiple sets of matches -/
def average_runs (runs_per_set : List ℕ) (matches_per_set : List ℕ) : ℚ :=
  (runs_per_set.zip matches_per_set).map (fun (r, m) => r * m)
    |> List.sum
    |> (fun total_runs => total_runs / matches_per_set.sum)

theorem batsman_average_theorem (first_10_avg : ℕ) (next_10_avg : ℕ) :
  first_10_avg = 40 →
  next_10_avg = 30 →
  average_runs [first_10_avg, next_10_avg] [10, 10] = 35 := by
  sorry

#eval average_runs [40, 30] [10, 10]

end NUMINAMATH_CALUDE_batsman_average_theorem_l2712_271230


namespace NUMINAMATH_CALUDE_sum_of_xy_l2712_271216

theorem sum_of_xy (x y : ℕ) : 
  0 < x ∧ x < 30 ∧ 0 < y ∧ y < 30 ∧ x + y + x * y = 143 → 
  x + y = 22 ∨ x + y = 23 ∨ x + y = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_l2712_271216


namespace NUMINAMATH_CALUDE_trig_equation_solution_l2712_271237

theorem trig_equation_solution (a b c α β : ℝ) 
  (h1 : a * Real.cos α + b * Real.sin α = c)
  (h2 : a * Real.cos β + b * Real.sin β = c)
  (h3 : a^2 + b^2 ≠ 0)
  (h4 : ∀ k : ℤ, α ≠ β + 2 * k * Real.pi) :
  (Real.cos ((α - β) / 2))^2 = c^2 / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l2712_271237


namespace NUMINAMATH_CALUDE_horner_evaluation_approx_l2712_271252

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List Float) (x : Float) : Float :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial function f(x) = 1 + x + 0.5x^2 + 0.16667x^3 + 0.04167x^4 + 0.00833x^5 -/
def f (x : Float) : Float :=
  horner [1, 1, 0.5, 0.16667, 0.04167, 0.00833] x

theorem horner_evaluation_approx :
  (f (-0.2) - 0.81873).abs < 1e-5 := by
  sorry

end NUMINAMATH_CALUDE_horner_evaluation_approx_l2712_271252


namespace NUMINAMATH_CALUDE_expression_simplification_l2712_271205

theorem expression_simplification (a : ℝ) (h : a ≠ 0) :
  (a * (a + 1) + (a - 1)^2 - 1) / (-a) = -2 * a + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2712_271205


namespace NUMINAMATH_CALUDE_original_apples_in_B_l2712_271278

/-- Represents the number of apples in each basket -/
structure AppleBaskets where
  A : ℕ  -- Number of apples in basket A
  B : ℕ  -- Number of apples in basket B
  C : ℕ  -- Number of apples in basket C

/-- The conditions of the apple basket problem -/
def apple_basket_conditions (baskets : AppleBaskets) : Prop :=
  -- Condition 1: The number of apples in basket C is twice the number of apples in basket A
  baskets.C = 2 * baskets.A ∧
  -- Condition 2: After transferring 12 apples from B to A, A has 24 less than C
  baskets.A + 12 = baskets.C - 24 ∧
  -- Condition 3: After the transfer, B has 6 more than C
  baskets.B - 12 = baskets.C + 6

theorem original_apples_in_B (baskets : AppleBaskets) :
  apple_basket_conditions baskets → baskets.B = 90 := by
  sorry

end NUMINAMATH_CALUDE_original_apples_in_B_l2712_271278


namespace NUMINAMATH_CALUDE_twenty_paise_coins_count_l2712_271258

theorem twenty_paise_coins_count 
  (total_coins : ℕ) 
  (total_value : ℚ) 
  (h_total_coins : total_coins = 334)
  (h_total_value : total_value = 71)
  : ∃ (coins_20p coins_25p : ℕ), 
    coins_20p + coins_25p = total_coins ∧ 
    (1/5 : ℚ) * coins_20p + (1/4 : ℚ) * coins_25p = total_value ∧
    coins_20p = 250 := by
  sorry

end NUMINAMATH_CALUDE_twenty_paise_coins_count_l2712_271258


namespace NUMINAMATH_CALUDE_problem_statement_l2712_271259

theorem problem_statement (x y : ℝ) (h : x + 2*y = 30) : 
  x/5 + 2*y/3 + 2*y/5 + x/3 = 16 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2712_271259


namespace NUMINAMATH_CALUDE_banana_arrangements_count_l2712_271298

/-- The number of unique arrangements of the letters in BANANA -/
def banana_arrangements : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of unique arrangements of BANANA is 60 -/
theorem banana_arrangements_count : banana_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_count_l2712_271298


namespace NUMINAMATH_CALUDE_geometry_class_ratio_l2712_271245

theorem geometry_class_ratio (total_students : ℕ) (boys_under_6ft : ℕ) :
  total_students = 38 →
  (2 : ℚ) / 3 * total_students = 25 →
  boys_under_6ft = 19 →
  (boys_under_6ft : ℚ) / 25 = 19 / 25 := by
  sorry

end NUMINAMATH_CALUDE_geometry_class_ratio_l2712_271245


namespace NUMINAMATH_CALUDE_square_difference_l2712_271294

theorem square_difference (x : ℤ) (h : x^2 = 3136) : (x + 2) * (x - 2) = 3132 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2712_271294


namespace NUMINAMATH_CALUDE_overlap_area_l2712_271293

-- Define the points on a 2D grid
def Point := ℕ × ℕ

-- Define the rectangle
def rectangle : List Point := [(0, 0), (3, 0), (3, 2), (0, 2)]

-- Define the triangle
def triangle : List Point := [(2, 0), (2, 2), (4, 2)]

-- Function to calculate the area of a right triangle
def rightTriangleArea (base height : ℕ) : ℚ :=
  (base * height) / 2

-- Theorem stating that the overlapping area is 1 square unit
theorem overlap_area :
  let overlapBase := 1
  let overlapHeight := 2
  rightTriangleArea overlapBase overlapHeight = 1 := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_l2712_271293


namespace NUMINAMATH_CALUDE_matrix_equality_l2712_271285

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = !![5, 2; -2, 3]) : 
  B * A = !![5, 2; -2, 3] := by sorry

end NUMINAMATH_CALUDE_matrix_equality_l2712_271285


namespace NUMINAMATH_CALUDE_equal_price_sheets_is_12_l2712_271222

/-- The number of sheets for which two photo companies charge the same amount -/
def equal_price_sheets : ℕ :=
  let john_per_sheet : ℚ := 275 / 100
  let john_sitting_fee : ℚ := 125
  let sam_per_sheet : ℚ := 150 / 100
  let sam_sitting_fee : ℚ := 140
  ⌊(sam_sitting_fee - john_sitting_fee) / (john_per_sheet - sam_per_sheet)⌋₊

theorem equal_price_sheets_is_12 : equal_price_sheets = 12 := by
  sorry

#eval equal_price_sheets

end NUMINAMATH_CALUDE_equal_price_sheets_is_12_l2712_271222


namespace NUMINAMATH_CALUDE_sports_classes_theorem_l2712_271256

/-- The number of students in different sports classes -/
def sports_classes (x : ℕ) : ℕ × ℕ × ℕ :=
  let basketball := x
  let soccer := 2 * x - 2
  let volleyball := (soccer / 2) + 2
  (basketball, soccer, volleyball)

theorem sports_classes_theorem (x : ℕ) (h : 2 * x - 6 = 34) :
  sports_classes x = (20, 34, 19) := by
  sorry

end NUMINAMATH_CALUDE_sports_classes_theorem_l2712_271256


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2712_271276

theorem perfect_square_condition (a b k : ℝ) : 
  (∃ (c : ℝ), a^2 + 2*(k-3)*a*b + 9*b^2 = c^2) → (k = 0 ∨ k = 6) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2712_271276


namespace NUMINAMATH_CALUDE_min_value_function_min_value_attained_l2712_271295

theorem min_value_function (x : ℝ) (h : x > 0) : 3 * x + 12 / x^2 ≥ 9 :=
sorry

theorem min_value_attained : ∃ x : ℝ, x > 0 ∧ 3 * x + 12 / x^2 = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_function_min_value_attained_l2712_271295


namespace NUMINAMATH_CALUDE_remaining_pennies_l2712_271236

theorem remaining_pennies (initial : ℝ) (spent : ℝ) (remaining : ℝ) 
  (h1 : initial = 98.5) 
  (h2 : spent = 93.25) 
  (h3 : remaining = initial - spent) : 
  remaining = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pennies_l2712_271236


namespace NUMINAMATH_CALUDE_x_coordinate_range_l2712_271227

-- Define the line L
def L (x y : ℝ) : Prop := x + y - 9 = 0

-- Define the circle M
def M (x y : ℝ) : Prop := 2*x^2 + 2*y^2 - 8*x - 8*y - 1 = 0

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  A_on_L : L A.1 A.2
  B_on_M : M B.1 B.2
  C_on_M : M C.1 C.2
  angle_BAC : Real.cos (45 * π / 180) = (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) /
    (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))

-- Define that AB passes through the center of M
def AB_through_center (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 2 = A.1 + t * (B.1 - A.1) ∧ 2 = A.2 + t * (B.2 - A.2)

-- Main theorem
theorem x_coordinate_range (A B C : ℝ × ℝ) 
  (h_triangle : Triangle A B C) (h_AB : AB_through_center A B) : 
  3 ≤ A.1 ∧ A.1 ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_x_coordinate_range_l2712_271227


namespace NUMINAMATH_CALUDE_translation_transforms_function_l2712_271209

/-- The translation vector -/
def translation_vector : ℝ × ℝ := (2, -3)

/-- The original function -/
def original_function (x : ℝ) : ℝ := x^2 + 4*x + 7

/-- The translated function -/
def translated_function (x : ℝ) : ℝ := x^2

theorem translation_transforms_function :
  ∀ x y : ℝ,
  original_function (x - translation_vector.1) + translation_vector.2 = translated_function x :=
by sorry

end NUMINAMATH_CALUDE_translation_transforms_function_l2712_271209


namespace NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_l2712_271214

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  v1 : Point
  v2 : Point
  v3 : Point

/-- Check if a point is in the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Check if three points form an equilateral triangle -/
def isEquilateral (t : EquilateralTriangle) : Prop :=
  let d12 := ((t.v1.x - t.v2.x)^2 + (t.v1.y - t.v2.y)^2)
  let d23 := ((t.v2.x - t.v3.x)^2 + (t.v2.y - t.v3.y)^2)
  let d31 := ((t.v3.x - t.v1.x)^2 + (t.v3.y - t.v1.y)^2)
  d12 = d23 ∧ d23 = d31

theorem equilateral_triangle_third_vertex 
  (t : EquilateralTriangle)
  (h1 : t.v1 = ⟨0, 3⟩)
  (h2 : t.v2 = ⟨6, 3⟩)
  (h3 : isInFirstQuadrant t.v3)
  (h4 : isEquilateral t) :
  t.v3 = ⟨6, 3 + 3 * Real.sqrt 3⟩ := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_l2712_271214


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2712_271207

theorem arithmetic_sequence_sum (a₁ aₙ d : ℕ) (n : ℕ) (h1 : a₁ = 1) (h2 : aₙ = 31) (h3 : d = 3) (h4 : n = (aₙ - a₁) / d + 1) :
  (n : ℝ) / 2 * (a₁ + aₙ) = 176 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2712_271207


namespace NUMINAMATH_CALUDE_not_equal_necessary_not_sufficient_l2712_271246

-- Define the relationship between α and β
def not_equal (α β : Real) : Prop := α ≠ β

-- Define the relationship between sin α and sin β
def sin_not_equal (α β : Real) : Prop := Real.sin α ≠ Real.sin β

-- Theorem stating that not_equal is a necessary but not sufficient condition for sin_not_equal
theorem not_equal_necessary_not_sufficient :
  (∀ α β : Real, sin_not_equal α β → not_equal α β) ∧
  ¬(∀ α β : Real, not_equal α β → sin_not_equal α β) :=
sorry

end NUMINAMATH_CALUDE_not_equal_necessary_not_sufficient_l2712_271246


namespace NUMINAMATH_CALUDE_certain_number_problem_l2712_271204

theorem certain_number_problem (x : ℝ) : 
  ((x + 10) * 2) / 2 - 2 = 88 / 2 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2712_271204
