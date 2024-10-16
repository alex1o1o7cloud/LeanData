import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l3806_380632

/-- Represents an ellipse in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a line in the Cartesian coordinate system -/
structure Line where
  k : ℝ

/-- Theorem stating the properties of the ellipse and line -/
theorem ellipse_and_line_properties
  (C : Ellipse)
  (h1 : C.a^2 / C.b^2 = 4 / 3)  -- Eccentricity condition
  (h2 : 1 / C.a^2 + (9/4) / C.b^2 = 1)  -- Point (1, 3/2) lies on the ellipse
  (l : Line)
  (h3 : ∀ x y, y = l.k * (x - 1))  -- Line equation
  (h4 : ∃ x1 y1 x2 y2, 
    x1^2 / C.a^2 + y1^2 / C.b^2 = 1 ∧
    x2^2 / C.a^2 + y2^2 / C.b^2 = 1 ∧
    y1 = l.k * (x1 - 1) ∧
    y2 = l.k * (x2 - 1) ∧
    x1 * x2 + y1 * y2 = -2)  -- Intersection points and dot product condition
  : C.a^2 = 4 ∧ C.b^2 = 3 ∧ l.k^2 = 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l3806_380632


namespace NUMINAMATH_CALUDE_paco_initial_cookies_l3806_380657

/-- The number of cookies Paco ate -/
def cookies_eaten : ℕ := 21

/-- The number of cookies Paco had left -/
def cookies_left : ℕ := 7

/-- The initial number of cookies Paco had -/
def initial_cookies : ℕ := cookies_eaten + cookies_left

theorem paco_initial_cookies : initial_cookies = 28 := by sorry

end NUMINAMATH_CALUDE_paco_initial_cookies_l3806_380657


namespace NUMINAMATH_CALUDE_square_area_probability_square_area_probability_proof_l3806_380689

/-- The probability of a randomly chosen point P on a line segment AB of length 10 cm
    resulting in a square with side length AP having an area between 25 cm² and 49 cm² -/
theorem square_area_probability : ℝ :=
  let AB : ℝ := 10
  let lower_bound : ℝ := 25
  let upper_bound : ℝ := 49
  1 / 5

/-- Proof of the theorem -/
theorem square_area_probability_proof :
  square_area_probability = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_square_area_probability_square_area_probability_proof_l3806_380689


namespace NUMINAMATH_CALUDE_sum_fifth_sixth_is_180_l3806_380633

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  sum_first_two : a 1 + a 2 = 20
  sum_third_fourth : a 3 + a 4 = 60

/-- The sum of the fifth and sixth terms of the geometric sequence is 180 -/
theorem sum_fifth_sixth_is_180 (seq : GeometricSequence) : seq.a 5 + seq.a 6 = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_fifth_sixth_is_180_l3806_380633


namespace NUMINAMATH_CALUDE_problem_statement_l3806_380677

theorem problem_statement (a : ℤ) (h1 : 0 < a) (h2 : a < 13) 
  (h3 : ∃ k : ℤ, 53^2016 + a = 13 * k) : a = 12 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3806_380677


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l3806_380673

theorem necessary_and_sufficient_condition : 
  (∀ x : ℝ, x^2 - 2*x + 1 = 0 ↔ x = 1) := by sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l3806_380673


namespace NUMINAMATH_CALUDE_jelly_beans_weight_l3806_380615

theorem jelly_beans_weight (initial_weight : ℝ) : 
  initial_weight > 0 →
  2 * (4 * initial_weight) = 16 →
  initial_weight = 2 := by
sorry

end NUMINAMATH_CALUDE_jelly_beans_weight_l3806_380615


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3806_380684

/-- A primitive third root of unity -/
noncomputable def α : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

/-- The polynomial x^103 + Cx^2 + Dx + E -/
def P (C D E : ℂ) (x : ℂ) : ℂ := x^103 + C*x^2 + D*x + E

theorem polynomial_divisibility (C D E : ℂ) :
  (∀ x, x^2 - x + 1 = 0 → P C D E x = 0) →
  C + D + E = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3806_380684


namespace NUMINAMATH_CALUDE_r_earnings_l3806_380672

/-- Represents the daily earnings of individuals p, q, and r -/
structure Earnings where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The conditions given in the problem -/
def problem_conditions (e : Earnings) : Prop :=
  e.p + e.q + e.r = 1980 / 9 ∧
  e.p + e.r = 600 / 5 ∧
  e.q + e.r = 910 / 7

/-- The theorem stating that under the given conditions, r earns 30 rs per day -/
theorem r_earnings (e : Earnings) : problem_conditions e → e.r = 30 := by
  sorry

end NUMINAMATH_CALUDE_r_earnings_l3806_380672


namespace NUMINAMATH_CALUDE_library_books_end_of_month_l3806_380602

theorem library_books_end_of_month 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) :
  initial_books = 75 →
  loaned_books = 60 →
  return_rate = 65 / 100 →
  initial_books - loaned_books + (return_rate * loaned_books).floor = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_library_books_end_of_month_l3806_380602


namespace NUMINAMATH_CALUDE_investment_comparison_l3806_380653

def initial_AA : ℝ := 200
def initial_BB : ℝ := 150
def initial_CC : ℝ := 100

def year1_AA_change : ℝ := 1.30
def year1_BB_change : ℝ := 0.80
def year1_CC_change : ℝ := 1.10

def year2_AA_change : ℝ := 0.85
def year2_BB_change : ℝ := 1.30
def year2_CC_change : ℝ := 0.95

def final_A : ℝ := initial_AA * year1_AA_change * year2_AA_change
def final_B : ℝ := initial_BB * year1_BB_change * year2_BB_change
def final_C : ℝ := initial_CC * year1_CC_change * year2_CC_change

theorem investment_comparison : final_A > final_B ∧ final_B > final_C := by
  sorry

end NUMINAMATH_CALUDE_investment_comparison_l3806_380653


namespace NUMINAMATH_CALUDE_circle_equations_l3806_380600

-- Define the points
def M : ℝ × ℝ := (-1, 1)
def N : ℝ × ℝ := (0, 2)
def Q : ℝ × ℝ := (2, 0)

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 1/2)^2 + (y - 1/2)^2 = 5/2
def C₂ (x y : ℝ) : Prop := (x + 3/2)^2 + (y - 5/2)^2 = 5/2

-- Define the line MN
def MN (x y : ℝ) : Prop := x - y + 2 = 0

-- Theorem statement
theorem circle_equations :
  (C₁ M.1 M.2 ∧ C₁ N.1 N.2 ∧ C₁ Q.1 Q.2) ∧
  (∀ x y x' y', C₁ x y ∧ C₂ x' y' → 
    MN ((x + x')/2) ((y + y')/2) ∧
    (x' - x)^2 + (y' - y)^2 = 4 * ((x - 1/2)^2 + (y - 1/2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equations_l3806_380600


namespace NUMINAMATH_CALUDE_revenue_decrease_l3806_380665

theorem revenue_decrease (last_year_revenue : ℝ) : 
  let projected_revenue := 1.25 * last_year_revenue
  let actual_revenue := 0.6 * projected_revenue
  let decrease := projected_revenue - actual_revenue
  let percentage_decrease := (decrease / projected_revenue) * 100
  percentage_decrease = 40 := by
sorry

end NUMINAMATH_CALUDE_revenue_decrease_l3806_380665


namespace NUMINAMATH_CALUDE_bike_clamps_promotion_l3806_380698

/-- The number of bike clamps given per bicycle purchase -/
def clamps_per_bike (morning_bikes : ℕ) (afternoon_bikes : ℕ) (total_clamps : ℕ) : ℚ :=
  total_clamps / (morning_bikes + afternoon_bikes)

/-- Theorem stating that the number of bike clamps given per bicycle purchase is 2 -/
theorem bike_clamps_promotion (morning_bikes afternoon_bikes total_clamps : ℕ)
  (h1 : morning_bikes = 19)
  (h2 : afternoon_bikes = 27)
  (h3 : total_clamps = 92) :
  clamps_per_bike morning_bikes afternoon_bikes total_clamps = 2 := by
  sorry

end NUMINAMATH_CALUDE_bike_clamps_promotion_l3806_380698


namespace NUMINAMATH_CALUDE_probability_not_adjacent_l3806_380646

theorem probability_not_adjacent (n : ℕ) : 
  n = 5 → (36 : ℚ) / (120 : ℚ) = (3 : ℚ) / (10 : ℚ) := by sorry

end NUMINAMATH_CALUDE_probability_not_adjacent_l3806_380646


namespace NUMINAMATH_CALUDE_average_age_combined_l3806_380661

theorem average_age_combined (num_students : Nat) (num_parents : Nat) 
  (avg_age_students : ℝ) (avg_age_parents : ℝ) :
  num_students = 45 →
  num_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 40 →
  (num_students * avg_age_students + num_parents * avg_age_parents) / (num_students + num_parents : ℝ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l3806_380661


namespace NUMINAMATH_CALUDE_x_cubed_greater_y_squared_l3806_380687

theorem x_cubed_greater_y_squared (x y : ℝ) 
  (h1 : x^5 > y^4) (h2 : y^5 > x^4) : x^3 > y^2 := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_greater_y_squared_l3806_380687


namespace NUMINAMATH_CALUDE_largest_x_value_l3806_380644

theorem largest_x_value : ∃ (x_max : ℝ), 
  (∀ x : ℝ, (15 * x^2 - 40 * x + 18) / (4 * x - 3) + 7 * x = 8 * x - 2 → x ≤ x_max) ∧
  ((15 * x_max^2 - 40 * x_max + 18) / (4 * x_max - 3) + 7 * x_max = 8 * x_max - 2) ∧
  x_max = 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l3806_380644


namespace NUMINAMATH_CALUDE_negative_sum_l3806_380611

theorem negative_sum (u v w : ℝ) 
  (hu : -1 < u ∧ u < 0) 
  (hv : 0 < v ∧ v < 1) 
  (hw : -2 < w ∧ w < -1) : 
  v + w < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_sum_l3806_380611


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l3806_380682

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l3806_380682


namespace NUMINAMATH_CALUDE_cats_given_by_mr_sheridan_l3806_380621

/-- The number of cats Mrs. Sheridan initially had -/
def initial_cats : ℕ := 17

/-- The total number of cats Mrs. Sheridan has now -/
def total_cats : ℕ := 31

/-- The number of cats Mr. Sheridan gave to Mrs. Sheridan -/
def given_cats : ℕ := total_cats - initial_cats

theorem cats_given_by_mr_sheridan : given_cats = 14 := by sorry

end NUMINAMATH_CALUDE_cats_given_by_mr_sheridan_l3806_380621


namespace NUMINAMATH_CALUDE_inequality_range_l3806_380656

/-- Given that m < (e^x) / (x*e^x - x + 1) has exactly two integer solutions, 
    prove that the range of m is [e^2 / (2e^2 - 1), 1) -/
theorem inequality_range (m : ℝ) : 
  (∃! (a b : ℤ), ∀ (x : ℤ), m < (Real.exp x) / (x * Real.exp x - x + 1) ↔ x = a ∨ x = b) →
  m ∈ Set.Ici (Real.exp 2 / (2 * Real.exp 2 - 1)) ∩ Set.Iio 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l3806_380656


namespace NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l3806_380659

theorem condition_neither_sufficient_nor_necessary
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ)
  (ha₁ : a₁ ≠ 0) (hb₁ : b₁ ≠ 0) (hc₁ : c₁ ≠ 0)
  (ha₂ : a₂ ≠ 0) (hb₂ : b₂ ≠ 0) (hc₂ : c₂ ≠ 0) :
  ¬(((a₁ / a₂ = b₁ / b₂) ∧ (b₁ / b₂ = c₁ / c₂)) ↔
    (∀ x, a₁ * x^2 + b₁ * x + c₁ > 0 ↔ a₂ * x^2 + b₂ * x + c₂ > 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l3806_380659


namespace NUMINAMATH_CALUDE_hannah_ran_9km_on_monday_l3806_380695

/-- The distance Hannah ran on Wednesday in meters -/
def wednesday_distance : ℕ := 4816

/-- The distance Hannah ran on Friday in meters -/
def friday_distance : ℕ := 2095

/-- The additional distance Hannah ran on Monday compared to Wednesday and Friday combined, in meters -/
def monday_additional_distance : ℕ := 2089

/-- The number of meters in a kilometer -/
def meters_per_kilometer : ℕ := 1000

/-- Theorem stating that Hannah ran 9 kilometers on Monday -/
theorem hannah_ran_9km_on_monday : 
  (wednesday_distance + friday_distance + monday_additional_distance) / meters_per_kilometer = 9 := by
  sorry

end NUMINAMATH_CALUDE_hannah_ran_9km_on_monday_l3806_380695


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l3806_380686

theorem jelly_bean_probability (p_red p_orange p_yellow p_green : ℝ) :
  p_red = 0.25 →
  p_orange = 0.35 →
  p_red + p_orange + p_yellow + p_green = 1 →
  p_red ≥ 0 ∧ p_orange ≥ 0 ∧ p_yellow ≥ 0 ∧ p_green ≥ 0 →
  p_yellow = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l3806_380686


namespace NUMINAMATH_CALUDE_initial_pink_hats_l3806_380605

/-- The number of pink hard hats initially in the truck -/
def initial_pink : ℕ := sorry

/-- The number of green hard hats initially in the truck -/
def initial_green : ℕ := 15

/-- The number of yellow hard hats initially in the truck -/
def initial_yellow : ℕ := 24

/-- The number of pink hard hats Carl takes away -/
def carl_pink : ℕ := 4

/-- The number of pink hard hats John takes away -/
def john_pink : ℕ := 6

/-- The number of green hard hats John takes away -/
def john_green : ℕ := 2 * john_pink

/-- The total number of hard hats remaining in the truck after Carl and John take some away -/
def remaining_hats : ℕ := 43

theorem initial_pink_hats : initial_pink = 26 := by sorry

end NUMINAMATH_CALUDE_initial_pink_hats_l3806_380605


namespace NUMINAMATH_CALUDE_average_student_height_l3806_380683

/-- The average height of all students given specific conditions -/
theorem average_student_height 
  (avg_female_height : ℝ) 
  (avg_male_height : ℝ) 
  (male_to_female_ratio : ℝ) 
  (h1 : avg_female_height = 170) 
  (h2 : avg_male_height = 182) 
  (h3 : male_to_female_ratio = 5) : 
  (male_to_female_ratio * avg_male_height + avg_female_height) / (male_to_female_ratio + 1) = 180 := by
  sorry

end NUMINAMATH_CALUDE_average_student_height_l3806_380683


namespace NUMINAMATH_CALUDE_point_m_location_l3806_380607

theorem point_m_location (L P M : ℚ) : 
  L = 1/6 → P = 1/12 → M - L = (P - L) / 3 → M = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_point_m_location_l3806_380607


namespace NUMINAMATH_CALUDE_cube_volume_in_pyramid_l3806_380663

/-- A regular pyramid with a rectangular base and isosceles triangular lateral faces -/
structure RegularPyramid where
  base_length : ℝ
  base_width : ℝ
  lateral_faces_isosceles : Bool

/-- A cube placed inside the pyramid -/
structure InsideCube where
  side_length : ℝ

/-- The theorem stating the volume of the cube inside the pyramid -/
theorem cube_volume_in_pyramid (pyramid : RegularPyramid) (cube : InsideCube) : 
  pyramid.base_length = 2 →
  pyramid.base_width = 3 →
  pyramid.lateral_faces_isosceles = true →
  (cube.side_length * Real.sqrt 3 = Real.sqrt 13) →
  cube.side_length^3 = (39 * Real.sqrt 39) / 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_in_pyramid_l3806_380663


namespace NUMINAMATH_CALUDE_probability_theorem_l3806_380650

def total_balls : ℕ := 9
def red_balls : ℕ := 2
def black_balls : ℕ := 3
def white_balls : ℕ := 4

def prob_black_then_white : ℚ := 1/6

def prob_red_within_three : ℚ := 7/12

theorem probability_theorem :
  (black_balls / total_balls * white_balls / (total_balls - 1) = prob_black_then_white) ∧
  (red_balls / total_balls + 
   (total_balls - red_balls) / total_balls * red_balls / (total_balls - 1) + 
   (total_balls - red_balls) / total_balls * (total_balls - red_balls - 1) / (total_balls - 1) * red_balls / (total_balls - 2) = prob_red_within_three) :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l3806_380650


namespace NUMINAMATH_CALUDE_equidifference_ratio_sequence_properties_l3806_380606

/-- Definition of an equidifference ratio sequence -/
def IsEquidifferenceRatioSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ n : ℕ+, (a (n + 2) - a (n + 1)) / (a (n + 1) - a n) = k

theorem equidifference_ratio_sequence_properties
  (a : ℕ+ → ℝ) (h : IsEquidifferenceRatioSequence a) :
  (∃ k : ℝ, k ≠ 0 ∧ ∀ n : ℕ+, (a (n + 2) - a (n + 1)) / (a (n + 1) - a n) = k) ∧
  (∃ b : ℕ+ → ℝ, IsEquidifferenceRatioSequence b ∧ Set.Infinite {n : ℕ+ | b n = 0}) :=
by sorry

end NUMINAMATH_CALUDE_equidifference_ratio_sequence_properties_l3806_380606


namespace NUMINAMATH_CALUDE_function_max_min_difference_l3806_380678

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a^x else -x + a

-- State the theorem
theorem function_max_min_difference (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 0 2, f a x ≤ max) ∧
    (∀ x ∈ Set.Icc 0 2, f a x ≥ min) ∧
    (max - min = 5/2)) →
  (a = 1/2 ∨ a = 7/2) :=
by sorry

end NUMINAMATH_CALUDE_function_max_min_difference_l3806_380678


namespace NUMINAMATH_CALUDE_constant_remainder_implies_b_value_l3806_380643

/-- The dividend polynomial -/
def dividend (b x : ℚ) : ℚ := 12 * x^4 - 14 * x^3 + b * x^2 + 7 * x + 9

/-- The divisor polynomial -/
def divisor (x : ℚ) : ℚ := 3 * x^2 - 4 * x + 2

/-- The remainder polynomial -/
def remainder (b x : ℚ) : ℚ := dividend b x - divisor x * (4 * x^2 + 2/3 * x)

theorem constant_remainder_implies_b_value :
  (∃ (r : ℚ), ∀ (x : ℚ), remainder b x = r) ↔ b = 16/3 := by sorry

end NUMINAMATH_CALUDE_constant_remainder_implies_b_value_l3806_380643


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l3806_380609

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def M : Finset Nat := {1, 2, 4}

theorem complement_of_M_in_U :
  (U \ M) = {3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l3806_380609


namespace NUMINAMATH_CALUDE_not_on_inverse_proportion_graph_l3806_380634

def inverse_proportion (x y : ℝ) : Prop := x * y = 6

def point_on_graph (p : ℝ × ℝ) : Prop :=
  inverse_proportion p.1 p.2

theorem not_on_inverse_proportion_graph :
  point_on_graph (-2, -3) ∧
  point_on_graph (-3, -2) ∧
  ¬point_on_graph (1, 5) ∧
  point_on_graph (4, 1.5) :=
sorry

end NUMINAMATH_CALUDE_not_on_inverse_proportion_graph_l3806_380634


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l3806_380667

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) :
  x^4 * y^3 * z^2 ≥ 1/9^9 :=
by sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 1/x + 1/y + 1/z = 9 ∧
  x^4 * y^3 * z^2 < 1/9^9 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l3806_380667


namespace NUMINAMATH_CALUDE_admission_probability_l3806_380671

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of universities -/
def num_universities : ℕ := 3

/-- The total number of possible admission arrangements -/
def total_arrangements : ℕ := num_universities ^ num_students

/-- The number of arrangements where each university admits at least one student -/
def favorable_arrangements : ℕ := 36

/-- The probability that each university admits at least one student -/
def probability : ℚ := favorable_arrangements / total_arrangements

theorem admission_probability : probability = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_admission_probability_l3806_380671


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3806_380629

-- Define the logarithm with base 0.5
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log 0.5

-- State the theorem
theorem inequality_equivalence (x : ℝ) (h : x > 0) :
  2 * (log_half x)^2 + 9 * log_half x + 9 ≤ 0 ↔ 2 * Real.sqrt 2 ≤ x ∧ x ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3806_380629


namespace NUMINAMATH_CALUDE_wire_length_l3806_380693

/-- Represents the lengths of five wire pieces in a specific ratio --/
structure WirePieces where
  ratio : Fin 5 → ℕ
  shortest : ℝ
  total : ℝ

/-- The wire pieces satisfy the given conditions --/
def satisfies_conditions (w : WirePieces) : Prop :=
  w.ratio 0 = 4 ∧
  w.ratio 1 = 5 ∧
  w.ratio 2 = 7 ∧
  w.ratio 3 = 3 ∧
  w.ratio 4 = 2 ∧
  w.shortest = 16

/-- Theorem stating the total length of the wire --/
theorem wire_length (w : WirePieces) (h : satisfies_conditions w) : w.total = 84 := by
  sorry


end NUMINAMATH_CALUDE_wire_length_l3806_380693


namespace NUMINAMATH_CALUDE_two_numbers_sum_product_difference_l3806_380637

theorem two_numbers_sum_product_difference (n : ℕ) (hn : n = 38) :
  ∃ x y : ℕ,
    1 ≤ x ∧ x < y ∧ y ≤ n ∧
    (n * (n + 1)) / 2 - x - y = x * y ∧
    y - x = 39 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_product_difference_l3806_380637


namespace NUMINAMATH_CALUDE_vacation_payment_difference_l3806_380614

/-- Represents the vacation expenses and payments for four people. -/
structure VacationExpenses where
  tom_paid : ℕ
  dorothy_paid : ℕ
  sammy_paid : ℕ
  nancy_paid : ℕ
  total_cost : ℕ
  equal_share : ℕ

/-- The given vacation expenses. -/
def given_expenses : VacationExpenses := {
  tom_paid := 150,
  dorothy_paid := 190,
  sammy_paid := 240,
  nancy_paid := 320,
  total_cost := 900,
  equal_share := 225
}

/-- Theorem stating the difference between Tom's and Dorothy's additional payments. -/
theorem vacation_payment_difference (e : VacationExpenses) 
  (h1 : e.total_cost = e.tom_paid + e.dorothy_paid + e.sammy_paid + e.nancy_paid)
  (h2 : e.equal_share = e.total_cost / 4)
  (h3 : e = given_expenses) :
  (e.equal_share - e.tom_paid) - (e.equal_share - e.dorothy_paid) = 40 := by
  sorry

end NUMINAMATH_CALUDE_vacation_payment_difference_l3806_380614


namespace NUMINAMATH_CALUDE_angle_representation_l3806_380624

theorem angle_representation (given_angle : ℝ) : 
  given_angle = -1485 → 
  ∃ (α k : ℝ), 
    given_angle = α + k * 360 ∧ 
    0 ≤ α ∧ α < 360 ∧ 
    k = -5 ∧
    α = 315 := by
  sorry

end NUMINAMATH_CALUDE_angle_representation_l3806_380624


namespace NUMINAMATH_CALUDE_vector_representation_l3806_380640

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, -2)
def a : ℝ × ℝ := (-4, 0)

theorem vector_representation :
  a = (-1 : ℝ) • e₁ + (-1 : ℝ) • e₂ := by sorry

end NUMINAMATH_CALUDE_vector_representation_l3806_380640


namespace NUMINAMATH_CALUDE_octagon_area_given_equal_perimeter_and_square_area_l3806_380630

/-- Given a square and a regular octagon with equal perimeters, 
    if the square's area is 16, then the area of the octagon is 8(1+√2) -/
theorem octagon_area_given_equal_perimeter_and_square_area (a b : ℝ) : 
  a > 0 → b > 0 →
  (4 * a = 8 * b) →  -- Equal perimeters
  (a ^ 2 = 16) →     -- Square's area is 16
  (2 * (1 + Real.sqrt 2) * b ^ 2 = 8 * (1 + Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_octagon_area_given_equal_perimeter_and_square_area_l3806_380630


namespace NUMINAMATH_CALUDE_perfectSquareFactors450_l3806_380639

/-- The number of perfect square factors of 450 -/
def perfectSquareFactorsOf450 : ℕ :=
  4

/-- 450 as a natural number -/
def n : ℕ := 450

/-- A function that returns the number of perfect square factors of a natural number -/
def numPerfectSquareFactors (m : ℕ) : ℕ := sorry

theorem perfectSquareFactors450 : numPerfectSquareFactors n = perfectSquareFactorsOf450 := by
  sorry

end NUMINAMATH_CALUDE_perfectSquareFactors450_l3806_380639


namespace NUMINAMATH_CALUDE_train_length_l3806_380662

/-- The length of a train given its speed and time to cross a bridge -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 45 * (5 / 18) → 
  crossing_time = 30 →
  bridge_length = 240 →
  (train_speed * crossing_time) - bridge_length = 135 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3806_380662


namespace NUMINAMATH_CALUDE_projection_theorem_l3806_380625

def vector1 : ℝ × ℝ := (-4, 2)
def vector2 : ℝ × ℝ := (3, 5)

theorem projection_theorem (v : ℝ × ℝ) :
  ∃ (p : ℝ × ℝ), 
    (∃ (k1 : ℝ), p = Prod.mk (k1 * v.1) (k1 * v.2) ∧ 
      (p.1 - vector1.1) * v.1 + (p.2 - vector1.2) * v.2 = 0) ∧
    (∃ (k2 : ℝ), p = Prod.mk (k2 * v.1) (k2 * v.2) ∧ 
      (p.1 - vector2.1) * v.1 + (p.2 - vector2.2) * v.2 = 0) →
    p = (-39/29, 91/29) := by
  sorry

end NUMINAMATH_CALUDE_projection_theorem_l3806_380625


namespace NUMINAMATH_CALUDE_sin_difference_product_l3806_380612

theorem sin_difference_product (a b : ℝ) : 
  Real.sin (2 * a + b) - Real.sin (2 * a - b) = 2 * Real.cos (2 * a) * Real.sin b := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_product_l3806_380612


namespace NUMINAMATH_CALUDE_domain_of_shifted_f_l3806_380635

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc 1 2

-- Define the property that f is defined only on its domain
axiom f_defined_on_domain : ∀ x, x ∈ domain_f → f x ≠ 0

-- State the theorem
theorem domain_of_shifted_f :
  {x | f (x + 1) ≠ 0} = Set.Icc 0 1 :=
sorry

end NUMINAMATH_CALUDE_domain_of_shifted_f_l3806_380635


namespace NUMINAMATH_CALUDE_complex_sum_to_polar_l3806_380617

theorem complex_sum_to_polar : 
  5 * Complex.exp (Complex.I * (3 * Real.pi / 8)) + 
  5 * Complex.exp (Complex.I * (17 * Real.pi / 16)) = 
  10 * Real.cos (5 * Real.pi / 32) * Complex.exp (Complex.I * (23 * Real.pi / 32)) := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_to_polar_l3806_380617


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3806_380604

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) :
  InverselyProportional x y →
  (x + y = 30 → x = 3 * y) →
  x = -12 →
  y = -14.0625 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3806_380604


namespace NUMINAMATH_CALUDE_button_probability_l3806_380674

/-- Represents the number of buttons of each color in a jar -/
structure JarContents where
  red : ℕ
  blue : ℕ

/-- Represents the action of removing buttons from one jar to another -/
structure ButtonRemoval where
  removed : ℕ

theorem button_probability (initial_jar_a : JarContents) 
  (removal : ButtonRemoval) (final_jar_a : JarContents) :
  initial_jar_a.red = 4 →
  initial_jar_a.blue = 8 →
  removal.removed + removal.removed = initial_jar_a.red + initial_jar_a.blue - (final_jar_a.red + final_jar_a.blue) →
  3 * (final_jar_a.red + final_jar_a.blue) = 2 * (initial_jar_a.red + initial_jar_a.blue) →
  (final_jar_a.red / (final_jar_a.red + final_jar_a.blue : ℚ)) * 
  (removal.removed / ((initial_jar_a.red + initial_jar_a.blue - (final_jar_a.red + final_jar_a.blue)) : ℚ)) = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_button_probability_l3806_380674


namespace NUMINAMATH_CALUDE_silver_beads_count_l3806_380668

/-- Represents the number of beads in a necklace. -/
structure BeadCount where
  total : Nat
  blue : Nat
  red : Nat
  white : Nat
  silver : Nat

/-- Conditions for Michelle's necklace. -/
def michellesNecklace : BeadCount where
  total := 40
  blue := 5
  red := 2 * 5
  white := 5 + (2 * 5)
  silver := 40 - (5 + (2 * 5) + (5 + (2 * 5)))

/-- Theorem stating that the number of silver beads in Michelle's necklace is 10. -/
theorem silver_beads_count : michellesNecklace.silver = 10 := by
  sorry

#eval michellesNecklace.silver

end NUMINAMATH_CALUDE_silver_beads_count_l3806_380668


namespace NUMINAMATH_CALUDE_box_length_l3806_380692

/-- Given a box with width 16 units and height 13 units, which can contain 3120 unit cubes (1 x 1 x 1), prove that the length of the box is 15 units. -/
theorem box_length (width : ℕ) (height : ℕ) (volume : ℕ) (length : ℕ) : 
  width = 16 → height = 13 → volume = 3120 → volume = length * width * height → length = 15 := by
  sorry

end NUMINAMATH_CALUDE_box_length_l3806_380692


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3806_380622

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7*I
  let z₂ : ℂ := 4 - 7*I
  (z₁ / z₂) + (z₂ / z₁) = -66/65 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3806_380622


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l3806_380613

/-- Represents the population sizes for each age group -/
structure PopulationSizes where
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Represents the sample sizes for each age group -/
structure SampleSizes where
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Calculates the stratified sample sizes given population sizes and total sample size -/
def stratifiedSampleSizes (pop : PopulationSizes) (totalSample : ℕ) : SampleSizes :=
  { young := (pop.young * totalSample) / (pop.young + pop.middleAged + pop.elderly),
    middleAged := (pop.middleAged * totalSample) / (pop.young + pop.middleAged + pop.elderly),
    elderly := (pop.elderly * totalSample) / (pop.young + pop.middleAged + pop.elderly) }

theorem correct_stratified_sample (pop : PopulationSizes) (totalSample : ℕ) :
  pop.young = 45 ∧ pop.middleAged = 25 ∧ pop.elderly = 30 ∧ totalSample = 20 →
  let sample := stratifiedSampleSizes pop totalSample
  sample.young = 9 ∧ sample.middleAged = 5 ∧ sample.elderly = 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l3806_380613


namespace NUMINAMATH_CALUDE_tyler_meal_choices_l3806_380675

def meat_options : ℕ := 4
def vegetable_options : ℕ := 5
def dessert_options : ℕ := 3

def meat_choices : ℕ := 2
def vegetable_choices : ℕ := 3
def dessert_choices : ℕ := 1

theorem tyler_meal_choices :
  (Nat.choose meat_options meat_choices) *
  (Nat.choose vegetable_options vegetable_choices) *
  (Nat.choose dessert_options dessert_choices) = 180 := by
  sorry

end NUMINAMATH_CALUDE_tyler_meal_choices_l3806_380675


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3806_380690

/-- Calculate the profit percentage given the cost price and selling price -/
theorem profit_percentage_calculation (cost_price selling_price : ℚ) :
  cost_price = 60 →
  selling_price = 78 →
  (selling_price - cost_price) / cost_price * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l3806_380690


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3806_380648

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²) / a -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (a^2 + b^2) / a
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) → e = Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3806_380648


namespace NUMINAMATH_CALUDE_square_difference_of_quadratic_solutions_l3806_380670

theorem square_difference_of_quadratic_solutions : ∃ α β : ℝ,
  (α ≠ β) ∧ (α^2 = 2*α + 1) ∧ (β^2 = 2*β + 1) ∧ ((α - β)^2 = 8) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_quadratic_solutions_l3806_380670


namespace NUMINAMATH_CALUDE_range_of_f_l3806_380647

def f (x : ℕ) : ℕ := 2 * x + 1

def domain : Set ℕ := {1, 2, 3}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3806_380647


namespace NUMINAMATH_CALUDE_sean_played_14_days_l3806_380697

/-- The number of days Sean played cricket -/
def sean_days (sean_minutes_per_day : ℕ) (total_minutes : ℕ) (indira_minutes : ℕ) : ℕ :=
  (total_minutes - indira_minutes) / sean_minutes_per_day

/-- Proof that Sean played cricket for 14 days -/
theorem sean_played_14_days :
  sean_days 50 1512 812 = 14 := by
  sorry

end NUMINAMATH_CALUDE_sean_played_14_days_l3806_380697


namespace NUMINAMATH_CALUDE_existence_of_special_number_l3806_380616

def small_number (n : ℕ) : Prop := n ≤ 150

theorem existence_of_special_number :
  ∃ (N : ℕ) (a b : ℕ), 
    small_number a ∧ 
    small_number b ∧ 
    b = a + 1 ∧
    ¬(N % a = 0) ∧
    ¬(N % b = 0) ∧
    (∀ (k : ℕ), small_number k → k ≠ a → k ≠ b → N % k = 0) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l3806_380616


namespace NUMINAMATH_CALUDE_terrys_spending_l3806_380642

/-- Terry's spending problem -/
theorem terrys_spending (monday : ℕ) : 
  monday = 6 →
  let tuesday := 2 * monday
  let wednesday := 2 * (monday + tuesday)
  monday + tuesday + wednesday = 54 := by
  sorry

end NUMINAMATH_CALUDE_terrys_spending_l3806_380642


namespace NUMINAMATH_CALUDE_probability_king_hearts_then_ace_in_standard_deck_l3806_380651

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of King of Hearts in a standard deck -/
def KingOfHearts : ℕ := 1

/-- Number of Aces in a standard deck -/
def Aces : ℕ := 4

/-- Probability of drawing King of Hearts first and any Ace second -/
def probability_king_hearts_then_ace (deck : ℕ) (king_of_hearts : ℕ) (aces : ℕ) : ℚ :=
  (king_of_hearts : ℚ) / deck * (aces : ℚ) / (deck - 1)

theorem probability_king_hearts_then_ace_in_standard_deck :
  probability_king_hearts_then_ace StandardDeck KingOfHearts Aces = 1 / 663 := by
  sorry

end NUMINAMATH_CALUDE_probability_king_hearts_then_ace_in_standard_deck_l3806_380651


namespace NUMINAMATH_CALUDE_conic_is_parabola_l3806_380636

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  |x - 4| = Real.sqrt ((y + 3)^2 + x^2)

-- Theorem stating that the equation describes a parabola
theorem conic_is_parabola :
  ∃ (a b c d : ℝ), a ≠ 0 ∧
  ∀ (x y : ℝ), conic_equation x y ↔ y = a * x^2 + b * x + c * y + d :=
sorry

end NUMINAMATH_CALUDE_conic_is_parabola_l3806_380636


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3806_380658

theorem quadratic_inequality (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ ≠ x₂ →
  y₁ = -x₁^2 →
  y₂ = -x₂^2 →
  x₁ * x₂ > x₂^2 →
  y₁ < y₂ := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3806_380658


namespace NUMINAMATH_CALUDE_total_count_equals_115248_l3806_380601

/-- The number of digits that can be used (excluding 3, 6, and 9) -/
def available_digits : ℕ := 7

/-- The number of non-zero digits that can be used as the first digit -/
def first_digit_choices : ℕ := 6

/-- Calculates the number of n-digit numbers without 3, 6, or 9 -/
def count_numbers (n : ℕ) : ℕ :=
  first_digit_choices * available_digits^(n - 1)

/-- The total number of 5 and 6-digit numbers without 3, 6, or 9 -/
def total_count : ℕ := count_numbers 5 + count_numbers 6

theorem total_count_equals_115248 : total_count = 115248 := by
  sorry

end NUMINAMATH_CALUDE_total_count_equals_115248_l3806_380601


namespace NUMINAMATH_CALUDE_correct_calculation_l3806_380652

theorem correct_calculation : 
  (-2 + 3 = 1) ∧ 
  (-2 - 3 ≠ 1) ∧ 
  (-2 / (-1/2) ≠ 1) ∧ 
  ((-2)^3 ≠ -6) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3806_380652


namespace NUMINAMATH_CALUDE_units_digit_of_power_difference_l3806_380603

theorem units_digit_of_power_difference : ∃ n : ℕ, (25^2010 - 3^2012) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_difference_l3806_380603


namespace NUMINAMATH_CALUDE_triangle_ratio_l3806_380610

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  2 * b * Real.sin (2 * A) = 3 * a * Real.sin B →
  c = 2 * b →
  a / b = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3806_380610


namespace NUMINAMATH_CALUDE_R_is_converse_negation_of_P_l3806_380696

-- Define the proposition P
def P : Prop := ∀ x y : ℝ, x + y = 0 → (x = -y ∧ y = -x)

-- Define the negation of P (Q)
def Q : Prop := ¬P

-- Define the inverse of Q (R)
def R : Prop := ∀ x y : ℝ, ¬(x = -y ∧ y = -x) → x + y ≠ 0

-- Theorem stating that R is the converse negation of P
theorem R_is_converse_negation_of_P : R = (∀ x y : ℝ, ¬(x = -y ∧ y = -x) → x + y ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_R_is_converse_negation_of_P_l3806_380696


namespace NUMINAMATH_CALUDE_wire_length_ratio_l3806_380685

/-- Represents the length of a single piece of wire used by Bonnie -/
def bonnie_wire_length : ℝ := 4

/-- Represents the number of wire pieces used by Bonnie to construct her cube -/
def bonnie_wire_count : ℕ := 12

/-- Represents the length of a single piece of wire used by Roark -/
def roark_wire_length : ℝ := 1

/-- Represents the side length of Bonnie's cube -/
def bonnie_cube_side : ℝ := bonnie_wire_length

/-- Represents the volume of a single unit cube constructed by Roark -/
def unit_cube_volume : ℝ := 1

/-- Theorem stating that the ratio of Bonnie's total wire length to Roark's total wire length is 1/16 -/
theorem wire_length_ratio :
  (bonnie_wire_length * bonnie_wire_count) / 
  (roark_wire_length * (12 * (bonnie_cube_side ^ 3))) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l3806_380685


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l3806_380628

/-- The number of ways to choose k elements from a set of n elements --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players on the team --/
def totalPlayers : ℕ := 15

/-- The number of All-Star players --/
def allStars : ℕ := 3

/-- The size of the starting lineup --/
def lineupSize : ℕ := 5

theorem starting_lineup_combinations :
  choose (totalPlayers - allStars) (lineupSize - allStars) = 66 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l3806_380628


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l3806_380631

/-- A square with two vertices on a parabola and one side on a line -/
structure SquareOnParabola where
  /-- The side length of the square -/
  side : ℝ
  /-- The y-intercept of the line parallel to y = 2x - 22 that contains two vertices of the square -/
  b : ℝ
  /-- Two vertices of the square lie on the parabola y = x^2 -/
  vertices_on_parabola : ∃ (x₁ x₂ : ℝ), x₁^2 = (2 * x₁ + b) ∧ x₂^2 = (2 * x₂ + b) ∧ (x₁ - x₂)^2 + (x₁^2 - x₂^2)^2 = side^2
  /-- One side of the square lies on the line y = 2x - 22 -/
  side_on_line : side = |b + 22| / Real.sqrt 5

/-- The theorem stating the possible areas of the square -/
theorem square_area_on_parabola (s : SquareOnParabola) :
  s.side^2 = 115.2 ∨ s.side^2 = 156.8 := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l3806_380631


namespace NUMINAMATH_CALUDE_graph_translation_symmetry_l3806_380676

theorem graph_translation_symmetry (m : Real) : m > 0 →
  (∀ x, 2 * Real.sin (x + m - π / 3) = 2 * Real.sin (-x + m - π / 3)) →
  m = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_graph_translation_symmetry_l3806_380676


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3806_380681

theorem condition_necessary_not_sufficient :
  (∀ a b : ℝ, a^2 + b^2 = 2*a*b → a^2 = b^2) ∧
  (∃ a b : ℝ, a^2 = b^2 ∧ a^2 + b^2 ≠ 2*a*b) := by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3806_380681


namespace NUMINAMATH_CALUDE_ice_cube_ratio_l3806_380666

def ice_cubes_in_glass : ℕ := 8
def number_of_trays : ℕ := 2
def spaces_per_tray : ℕ := 12

def total_ice_cubes : ℕ := number_of_trays * spaces_per_tray
def ice_cubes_in_pitcher : ℕ := total_ice_cubes - ice_cubes_in_glass

theorem ice_cube_ratio :
  (ice_cubes_in_pitcher : ℚ) / ice_cubes_in_glass = 2 / 1 := by sorry

end NUMINAMATH_CALUDE_ice_cube_ratio_l3806_380666


namespace NUMINAMATH_CALUDE_pages_difference_l3806_380660

/-- The number of pages Person A reads per day -/
def pages_per_day_A : ℕ := 8

/-- The number of pages Person B reads per day -/
def pages_per_day_B : ℕ := 13

/-- The number of days we're considering -/
def days : ℕ := 7

/-- The total number of pages Person A reads in the given number of days -/
def total_pages_A : ℕ := pages_per_day_A * days

/-- The total number of pages Person B reads in the given number of days -/
def total_pages_B : ℕ := pages_per_day_B * days

theorem pages_difference : total_pages_B - total_pages_A = 9 := by
  sorry

end NUMINAMATH_CALUDE_pages_difference_l3806_380660


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3806_380655

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3806_380655


namespace NUMINAMATH_CALUDE_sum_equality_l3806_380679

theorem sum_equality (a b : Fin 2016 → ℝ) 
  (h1 : ∀ n ∈ Finset.range 2015, a (n + 1) = (1 / 65) * Real.sqrt (2 * (n + 1) + 2) + a n)
  (h2 : ∀ n ∈ Finset.range 2015, b (n + 1) = (1 / 1009) * Real.sqrt (2 * (n + 1) + 2) - b n)
  (h3 : a 0 = b 2015)
  (h4 : b 0 = a 2015) :
  (Finset.range 2015).sum (λ k => a (k + 1) * b k - a k * b (k + 1)) = 62 := by
sorry

end NUMINAMATH_CALUDE_sum_equality_l3806_380679


namespace NUMINAMATH_CALUDE_photo_arrangements_count_l3806_380626

/-- The number of ways to rearrange a group photo with the given conditions -/
def photo_arrangements : ℕ :=
  Nat.choose 8 2 * (5 * 4)

/-- Theorem stating that the number of photo arrangements is 560 -/
theorem photo_arrangements_count : photo_arrangements = 560 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_count_l3806_380626


namespace NUMINAMATH_CALUDE_correct_statements_l3806_380664

-- Define the statements
inductive Statement
| Synthesis1
| Synthesis2
| Analysis1
| Analysis2
| Contradiction

-- Define a function to check if a statement is correct
def is_correct (s : Statement) : Prop :=
  match s with
  | Statement.Synthesis1 => True  -- Synthesis is a method of cause and effect
  | Statement.Synthesis2 => True  -- Synthesis is a forward reasoning method
  | Statement.Analysis1 => True   -- Analysis is a method of seeking cause from effect
  | Statement.Analysis2 => False  -- Analysis is NOT an indirect proof method
  | Statement.Contradiction => False  -- Contradiction is NOT a backward reasoning method

-- Theorem to prove
theorem correct_statements :
  (is_correct Statement.Synthesis1) ∧
  (is_correct Statement.Synthesis2) ∧
  (is_correct Statement.Analysis1) ∧
  ¬(is_correct Statement.Analysis2) ∧
  ¬(is_correct Statement.Contradiction) :=
by sorry

#check correct_statements

end NUMINAMATH_CALUDE_correct_statements_l3806_380664


namespace NUMINAMATH_CALUDE_fraction_simplification_l3806_380694

theorem fraction_simplification (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3806_380694


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3806_380649

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := (1 + i) * z = 2 * i

-- Theorem statement
theorem complex_equation_solution :
  ∀ z : ℂ, equation z → z = 1 + i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3806_380649


namespace NUMINAMATH_CALUDE_seventeen_in_sample_l3806_380654

/-- Systematic sampling function -/
def systematicSample (populationSize sampleSize : ℕ) (first : ℕ) : List ℕ :=
  let interval := populationSize / sampleSize
  List.range sampleSize |>.map (fun i => first + i * interval)

/-- Theorem: In a systematic sample of size 4 from a population of 56, 
    if 3 is the first sampled number, then 17 will also be in the sample -/
theorem seventeen_in_sample :
  let sample := systematicSample 56 4 3
  17 ∈ sample := by
  sorry

end NUMINAMATH_CALUDE_seventeen_in_sample_l3806_380654


namespace NUMINAMATH_CALUDE_waiter_earnings_theorem_l3806_380619

/-- Calculates the total earnings for the first four nights of a five-day work week,
    given the target average per night and the required earnings for the last night. -/
def earnings_first_four_nights (days_per_week : ℕ) (target_average : ℚ) (last_night_earnings : ℚ) : ℚ :=
  days_per_week * target_average - last_night_earnings

theorem waiter_earnings_theorem :
  earnings_first_four_nights 5 50 115 = 135 := by
  sorry

end NUMINAMATH_CALUDE_waiter_earnings_theorem_l3806_380619


namespace NUMINAMATH_CALUDE_relative_complement_of_T_in_S_l3806_380669

open Set

def A₁ : Set ℕ := {0, 1}
def A₂ : Set ℕ := {1, 2}
def S : Set ℕ := A₁ ∪ A₂
def T : Set ℕ := A₁ ∩ A₂

theorem relative_complement_of_T_in_S :
  S \ T = {0, 2} := by sorry

end NUMINAMATH_CALUDE_relative_complement_of_T_in_S_l3806_380669


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l3806_380608

/-- The area of a square with one side on y = 10 and endpoints on y = x^2 + 4x + 3 is 44 -/
theorem square_area_on_parabola : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁^2 + 4*x₁ + 3 = 10) ∧ 
    (x₂^2 + 4*x₂ + 3 = 10) ∧ 
    (x₁ ≠ x₂) ∧
    ((x₂ - x₁)^2 = 44) := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l3806_380608


namespace NUMINAMATH_CALUDE_smallest_n_mod_30_l3806_380623

theorem smallest_n_mod_30 : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬(435 * m ≡ 867 * m [ZMOD 30])) ∧ 
  (435 * n ≡ 867 * n [ZMOD 30]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_mod_30_l3806_380623


namespace NUMINAMATH_CALUDE_tan_neg_five_pi_fourth_l3806_380618

theorem tan_neg_five_pi_fourth : Real.tan (-5 * Real.pi / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_five_pi_fourth_l3806_380618


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l3806_380620

theorem sufficient_condition_range (m : ℝ) : 
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 3) → x ≤ m) → m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l3806_380620


namespace NUMINAMATH_CALUDE_ball_max_height_l3806_380645

def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem ball_max_height :
  ∃ (max : ℝ), max = 161 ∧ ∀ (t : ℝ), h t ≤ max :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l3806_380645


namespace NUMINAMATH_CALUDE_dealer_pricing_theorem_l3806_380699

/-- A dealer's pricing strategy -/
structure DealerPricing where
  cash_discount : ℝ
  profit_percentage : ℝ
  articles_sold : ℕ
  articles_cost_price : ℕ

/-- Calculate the listing percentage above cost price -/
def listing_percentage (d : DealerPricing) : ℝ :=
  -- Define the calculation here
  sorry

/-- Theorem: Under specific conditions, the listing percentage is 60% -/
theorem dealer_pricing_theorem (d : DealerPricing) 
  (h1 : d.cash_discount = 0.15)
  (h2 : d.profit_percentage = 0.36)
  (h3 : d.articles_sold = 25)
  (h4 : d.articles_cost_price = 20) :
  listing_percentage d = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_dealer_pricing_theorem_l3806_380699


namespace NUMINAMATH_CALUDE_not_q_necessary_not_sufficient_for_not_p_l3806_380691

-- Define the propositions p and q
def p (x : ℝ) : Prop := x < 3
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Theorem stating that ¬q is a necessary but not sufficient condition for ¬p
theorem not_q_necessary_not_sufficient_for_not_p :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  (∃ x, ¬(q x) ∧ p x) :=
sorry

end NUMINAMATH_CALUDE_not_q_necessary_not_sufficient_for_not_p_l3806_380691


namespace NUMINAMATH_CALUDE_lynne_spent_75_l3806_380688

/-- The total amount Lynne spent on books and magazines -/
def total_spent (cat_books solar_books magazines book_price magazine_price : ℕ) : ℕ :=
  cat_books * book_price + solar_books * book_price + magazines * magazine_price

/-- Theorem stating that Lynne spent $75 in total -/
theorem lynne_spent_75 :
  total_spent 7 2 3 7 4 = 75 := by
  sorry

end NUMINAMATH_CALUDE_lynne_spent_75_l3806_380688


namespace NUMINAMATH_CALUDE_total_oranges_picked_l3806_380641

/-- The total number of oranges picked by Joan and Sara -/
def total_oranges (joan_oranges sara_oranges : ℕ) : ℕ :=
  joan_oranges + sara_oranges

/-- Theorem: Given that Joan picked 37 oranges and Sara picked 10 oranges,
    the total number of oranges picked is 47 -/
theorem total_oranges_picked :
  total_oranges 37 10 = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_picked_l3806_380641


namespace NUMINAMATH_CALUDE_mikaela_savings_l3806_380627

-- Define the hourly rate
def hourly_rate : ℕ := 10

-- Define the hours worked in the first month
def first_month_hours : ℕ := 35

-- Define the additional hours worked in the second month
def additional_hours : ℕ := 5

-- Define the fraction of earnings spent on personal needs
def spent_fraction : ℚ := 4/5

-- Function to calculate total earnings
def total_earnings (rate : ℕ) (hours1 : ℕ) (hours2 : ℕ) : ℕ :=
  rate * (hours1 + hours2)

-- Function to calculate savings
def savings (total : ℕ) (spent_frac : ℚ) : ℚ :=
  (1 - spent_frac) * total

-- Theorem statement
theorem mikaela_savings :
  savings (total_earnings hourly_rate first_month_hours (first_month_hours + additional_hours)) spent_fraction = 150 := by
  sorry


end NUMINAMATH_CALUDE_mikaela_savings_l3806_380627


namespace NUMINAMATH_CALUDE_first_three_digits_after_decimal_l3806_380638

theorem first_three_digits_after_decimal (n : ℕ) (x : ℝ) :
  n = 1200 →
  x = (10^n + 1)^(5/3) →
  ∃ (k : ℕ), x = k + 0.333 + r ∧ r < 0.001 :=
sorry

end NUMINAMATH_CALUDE_first_three_digits_after_decimal_l3806_380638


namespace NUMINAMATH_CALUDE_munchausen_polygon_exists_l3806_380680

/-- A polygon in a 2D plane --/
structure Polygon :=
  (vertices : Set (ℝ × ℝ))

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- A line in a 2D plane --/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Predicate to check if a point is inside a polygon --/
def IsInside (p : Point) (poly : Polygon) : Prop := sorry

/-- Predicate to check if a line passes through a point --/
def PassesThrough (l : Line) (p : Point) : Prop := sorry

/-- Predicate to check if a line divides a polygon into three parts --/
def DividesIntoThree (l : Line) (poly : Polygon) : Prop := sorry

/-- The main theorem --/
theorem munchausen_polygon_exists :
  ∃ (poly : Polygon) (p : Point),
    IsInside p poly ∧
    ∀ (l : Line), PassesThrough l p → DividesIntoThree l poly := by
  sorry

end NUMINAMATH_CALUDE_munchausen_polygon_exists_l3806_380680
