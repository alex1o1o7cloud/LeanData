import Mathlib

namespace NUMINAMATH_CALUDE_distance_between_cities_l3170_317027

/-- The distance between City A and City B -/
def distance : ℝ := 427.5

/-- The time for the first trip in hours -/
def time_first_trip : ℝ := 6

/-- The time for the return trip in hours -/
def time_return_trip : ℝ := 4.5

/-- The time saved on each trip in hours -/
def time_saved_per_trip : ℝ := 0.5

/-- The speed of the round trip if time was saved, in miles per hour -/
def speed_with_time_saved : ℝ := 90

theorem distance_between_cities :
  distance = 427.5 ∧
  (2 * distance) / (time_first_trip + time_return_trip - 2 * time_saved_per_trip) = speed_with_time_saved :=
by sorry

end NUMINAMATH_CALUDE_distance_between_cities_l3170_317027


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l3170_317075

open Real

theorem negation_of_existence_proposition :
  (¬ ∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) ↔
  (∀ a : ℝ, ∀ x : ℝ, a * x^2 + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l3170_317075


namespace NUMINAMATH_CALUDE_abs_6y_minus_8_not_positive_l3170_317041

theorem abs_6y_minus_8_not_positive (y : ℚ) : ¬(0 < |6 * y - 8|) ↔ y = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_abs_6y_minus_8_not_positive_l3170_317041


namespace NUMINAMATH_CALUDE_palindrome_count_l3170_317040

/-- A multiset representing the available digits -/
def availableDigits : Multiset ℕ := {1, 1, 2, 2, 2, 4, 4, 5, 5}

/-- The length of the palindrome -/
def palindromeLength : ℕ := 9

/-- Function to count valid 9-digit palindromes -/
def countPalindromes (digits : Multiset ℕ) (length : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of valid palindromes -/
theorem palindrome_count :
  countPalindromes availableDigits palindromeLength = 36 :=
sorry

end NUMINAMATH_CALUDE_palindrome_count_l3170_317040


namespace NUMINAMATH_CALUDE_division_problem_l3170_317023

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3)
  (h2 : b / c = 1 / 2)
  : c / a = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3170_317023


namespace NUMINAMATH_CALUDE_area_of_DBCE_l3170_317069

/-- Represents a triangle in the diagram -/
structure Triangle where
  area : ℝ

/-- Represents the trapezoid DBCE in the diagram -/
structure Trapezoid where
  area : ℝ

/-- The isosceles triangle ABC -/
def ABC : Triangle := { area := 96 }

/-- One of the smallest triangles in the diagram -/
def smallTriangle : Triangle := { area := 2 }

/-- The number of smallest triangles in the diagram -/
def numSmallTriangles : ℕ := 12

/-- The triangle ADF formed by 8 smallest triangles -/
def ADF : Triangle := { area := 8 * smallTriangle.area }

/-- The trapezoid DBCE -/
def DBCE : Trapezoid := { area := ABC.area - ADF.area }

theorem area_of_DBCE : DBCE.area = 80 := by
  sorry

end NUMINAMATH_CALUDE_area_of_DBCE_l3170_317069


namespace NUMINAMATH_CALUDE_pant_cost_l3170_317047

theorem pant_cost (num_shirts : ℕ) (shirt_cost : ℕ) (total_cost : ℕ) : 
  num_shirts = 10 →
  shirt_cost = 6 →
  total_cost = 100 →
  ∃ (pant_cost : ℕ), 
    pant_cost = 8 ∧ 
    num_shirts * shirt_cost + (num_shirts / 2) * pant_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_pant_cost_l3170_317047


namespace NUMINAMATH_CALUDE_fraction_equalities_l3170_317098

theorem fraction_equalities (a b : ℚ) (h : a / b = 5 / 6) : 
  ((a + 2 * b) / b = 17 / 6) ∧
  (b / (2 * a - b) = 3 / 2) ∧
  ((a + 3 * b) / (2 * a) = 23 / 10) ∧
  (a / (3 * b) = 5 / 18) ∧
  ((a - 2 * b) / b = -7 / 6) := by
sorry

end NUMINAMATH_CALUDE_fraction_equalities_l3170_317098


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3170_317087

theorem polynomial_divisibility (F : Int → Int) (A : Finset Int) :
  (∀ (x : Int), ∃ (a : Int), a ∈ A ∧ (∃ (k : Int), F x = a * k)) →
  (∀ (n : Int), ∃ (coeff : Int), F n = F (n + coeff) - F n) →
  ∃ (B : Finset Int), B ⊆ A ∧ B.card = 2 ∧
    ∀ (n : Int), ∃ (b : Int), b ∈ B ∧ (∃ (k : Int), F n = b * k) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3170_317087


namespace NUMINAMATH_CALUDE_average_marks_l3170_317017

theorem average_marks (num_subjects : ℕ) (avg_five : ℝ) (sixth_mark : ℝ) :
  num_subjects = 6 →
  avg_five = 74 →
  sixth_mark = 98 →
  ((avg_five * 5 + sixth_mark) / num_subjects : ℝ) = 78 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_l3170_317017


namespace NUMINAMATH_CALUDE_largest_of_seven_consecutive_integers_l3170_317062

theorem largest_of_seven_consecutive_integers (a : ℕ) : 
  (∃ (x : ℕ), x > 0 ∧ 
    (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) = 3020) ∧
    (∀ (y : ℕ), y > 0 → 
      (y + (y + 1) + (y + 2) + (y + 3) + (y + 4) + (y + 5) + (y + 6) = 3020) → 
      y = x)) →
  a = 434 ∧
  (∃ (x : ℕ), x > 0 ∧ 
    (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + a = 3020) ∧
    (∀ (y : ℕ), y > 0 → 
      (y + (y + 1) + (y + 2) + (y + 3) + (y + 4) + (y + 5) + a = 3020) → 
      y = x)) :=
by sorry

end NUMINAMATH_CALUDE_largest_of_seven_consecutive_integers_l3170_317062


namespace NUMINAMATH_CALUDE_f_of_one_equals_one_l3170_317011

theorem f_of_one_equals_one (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = Real.cos x) : f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_of_one_equals_one_l3170_317011


namespace NUMINAMATH_CALUDE_gcd_lcm_problem_l3170_317064

theorem gcd_lcm_problem (a b : ℕ+) 
  (h1 : Nat.gcd a b = 24)
  (h2 : Nat.lcm a b = 432)
  (h3 : a = 144) :
  b = 72 := by
sorry

end NUMINAMATH_CALUDE_gcd_lcm_problem_l3170_317064


namespace NUMINAMATH_CALUDE_count_negative_numbers_l3170_317013

def number_list : List ℚ := [-2 - 2/3, 9/14, -3, 5/2, 0, -48/10, 5, -1]

theorem count_negative_numbers : 
  (number_list.filter (λ x => x < 0)).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l3170_317013


namespace NUMINAMATH_CALUDE_min_draws_for_even_product_l3170_317086

theorem min_draws_for_even_product (n : ℕ) (h : n = 14) :
  let S := Finset.range n
  let even_count := (S.filter (λ x => x % 2 = 0)).card
  let odd_count := (S.filter (λ x => x % 2 ≠ 0)).card
  odd_count + 1 = 8 ∧ odd_count = even_count :=
by sorry

end NUMINAMATH_CALUDE_min_draws_for_even_product_l3170_317086


namespace NUMINAMATH_CALUDE_equal_areas_in_circle_configuration_l3170_317002

/-- Given a circle with radius R and four smaller circles with radius r = R/2 drawn through its center and touching it, 
    the area of the region not covered by the smaller circles (black region) 
    is equal to the sum of the areas of the overlapping regions of the smaller circles (gray regions). -/
theorem equal_areas_in_circle_configuration (R : ℝ) (h : R > 0) : 
  ∃ (black_area gray_area : ℝ),
    black_area = R^2 * π - 4 * (R/2)^2 * π ∧
    gray_area = 4 * ((R/2)^2 * π - (R/2)^2 * π / 3) ∧
    black_area = gray_area :=
by sorry

end NUMINAMATH_CALUDE_equal_areas_in_circle_configuration_l3170_317002


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3170_317077

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are equal. -/
def symmetric_wrt_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = y₂

/-- Given point A(2, -3) is symmetric to point A'(a, b) with respect to the y-axis, prove that a + b = -5. -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_y_axis 2 (-3) a b) : a + b = -5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3170_317077


namespace NUMINAMATH_CALUDE_triangle_ratio_l3170_317076

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →
  b = 1 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 :=
sorry


end NUMINAMATH_CALUDE_triangle_ratio_l3170_317076


namespace NUMINAMATH_CALUDE_store_B_cheapest_l3170_317045

/-- Represents a store with its pricing strategy -/
structure Store :=
  (name : String)
  (basePrice : ℕ)
  (discountStrategy : ℕ → ℕ)

/-- Calculates the cost of buying balls from a store -/
def cost (s : Store) (balls : ℕ) : ℕ :=
  s.discountStrategy balls

/-- Store A's discount strategy -/
def storeAStrategy (balls : ℕ) : ℕ :=
  let freeBalls := (balls / 10) * 2
  (balls - freeBalls) * 25

/-- Store B's discount strategy -/
def storeBStrategy (balls : ℕ) : ℕ :=
  balls * (25 - 5)

/-- Store C's discount strategy -/
def storeCStrategy (balls : ℕ) : ℕ :=
  let totalSpent := balls * 25
  let cashback := (totalSpent / 200) * 30
  totalSpent - cashback

/-- The three stores -/
def storeA : Store := ⟨"A", 25, storeAStrategy⟩
def storeB : Store := ⟨"B", 25, storeBStrategy⟩
def storeC : Store := ⟨"C", 25, storeCStrategy⟩

/-- The theorem to prove -/
theorem store_B_cheapest : 
  cost storeB 60 < cost storeA 60 ∧ cost storeB 60 < cost storeC 60 := by
  sorry

end NUMINAMATH_CALUDE_store_B_cheapest_l3170_317045


namespace NUMINAMATH_CALUDE_num_non_congruent_triangles_l3170_317009

/-- Represents a point on a 2D grid --/
structure GridPoint where
  x : ℚ
  y : ℚ

/-- The set of points on the grid --/
def gridPoints : Finset GridPoint := sorry

/-- Predicate to check if three points form a triangle --/
def isTriangle (p q r : GridPoint) : Prop := sorry

/-- Predicate to check if two triangles are congruent --/
def areCongruent (t1 t2 : GridPoint × GridPoint × GridPoint) : Prop := sorry

/-- The set of all possible triangles formed by the grid points --/
def allTriangles : Finset (GridPoint × GridPoint × GridPoint) := sorry

/-- The set of non-congruent triangles --/
def nonCongruentTriangles : Finset (GridPoint × GridPoint × GridPoint) := sorry

theorem num_non_congruent_triangles :
  Finset.card nonCongruentTriangles = 4 := by sorry

end NUMINAMATH_CALUDE_num_non_congruent_triangles_l3170_317009


namespace NUMINAMATH_CALUDE_a_range_l3170_317030

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3-2*a)^x < (3-2*a)^y

def range_a (a : ℝ) : Prop := a ≤ -2 ∨ (1 ≤ a ∧ a < 2)

theorem a_range (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_a a := by sorry

end NUMINAMATH_CALUDE_a_range_l3170_317030


namespace NUMINAMATH_CALUDE_probability_second_yellow_ball_l3170_317067

def initial_white_balls : ℕ := 5
def initial_yellow_balls : ℕ := 3

def remaining_white_balls : ℕ := initial_white_balls
def remaining_yellow_balls : ℕ := initial_yellow_balls - 1

def total_remaining_balls : ℕ := remaining_white_balls + remaining_yellow_balls

theorem probability_second_yellow_ball :
  (remaining_yellow_balls : ℚ) / total_remaining_balls = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_yellow_ball_l3170_317067


namespace NUMINAMATH_CALUDE_sin_2theta_value_l3170_317007

theorem sin_2theta_value (θ : Real) 
  (h1 : Real.cos (π/4 - θ) * Real.cos (π/4 + θ) = Real.sqrt 2 / 6)
  (h2 : 0 < θ) (h3 : θ < π/2) : 
  Real.sin (2*θ) = Real.sqrt 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l3170_317007


namespace NUMINAMATH_CALUDE_system_solution_l3170_317074

theorem system_solution (x y m : ℝ) : 
  (2 * x + y = 5) → 
  (x - 2 * y = m) → 
  (2 * x - 3 * y = 1) → 
  (m = 0) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3170_317074


namespace NUMINAMATH_CALUDE_no_four_consecutive_integers_product_perfect_square_l3170_317089

theorem no_four_consecutive_integers_product_perfect_square :
  ∀ x : ℕ+, ¬∃ y : ℕ, x * (x + 1) * (x + 2) * (x + 3) = y^2 := by
sorry

end NUMINAMATH_CALUDE_no_four_consecutive_integers_product_perfect_square_l3170_317089


namespace NUMINAMATH_CALUDE_divisible_by_three_l3170_317022

theorem divisible_by_three (n : ℕ) : 3 ∣ (5^n - 2^n) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_l3170_317022


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3170_317048

theorem rectangle_dimensions (x : ℝ) : 
  (x - 3) * (3 * x + 4) = 12 * x - 9 → x = (17 + 5 * Real.sqrt 13) / 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3170_317048


namespace NUMINAMATH_CALUDE_lukes_weekly_spending_l3170_317084

/-- Luke's weekly spending given his earnings and duration --/
theorem lukes_weekly_spending (mowing_earnings weed_eating_earnings : ℕ) (weeks : ℕ) :
  mowing_earnings = 9 →
  weed_eating_earnings = 18 →
  weeks = 9 →
  (mowing_earnings + weed_eating_earnings) / weeks = 3 := by
  sorry

end NUMINAMATH_CALUDE_lukes_weekly_spending_l3170_317084


namespace NUMINAMATH_CALUDE_base12_addition_correct_l3170_317049

/-- Converts a base 12 number represented as a list of digits to its decimal (base 10) equivalent -/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 12 + d) 0

/-- Converts a decimal (base 10) number to its base 12 representation as a list of digits -/
def decimalToBase12 (n : Nat) : List Nat :=
  if n < 12 then [n]
  else (n % 12) :: decimalToBase12 (n / 12)

/-- Represents a number in base 12 -/
structure Base12 where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 12

/-- Addition of two Base12 numbers -/
def add (a b : Base12) : Base12 :=
  let sum := base12ToDecimal a.digits + base12ToDecimal b.digits
  ⟨decimalToBase12 sum, sorry⟩

theorem base12_addition_correct :
  let a : Base12 := ⟨[3, 12, 5, 10], sorry⟩  -- 3C5A₁₂
  let b : Base12 := ⟨[4, 10, 3, 11], sorry⟩  -- 4A3B₁₂
  let result : Base12 := ⟨[8, 10, 9, 8], sorry⟩  -- 8A98₁₂
  add a b = result :=
sorry

end NUMINAMATH_CALUDE_base12_addition_correct_l3170_317049


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3170_317029

theorem sum_of_fractions : (3 : ℚ) / 5 + 5 / 11 + 1 / 3 = 229 / 165 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3170_317029


namespace NUMINAMATH_CALUDE_sugar_solution_concentration_increases_l3170_317059

theorem sugar_solution_concentration_increases 
  (a b m : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : m > 0) : 
  (b + m) / (a + m) > b / a := by
  sorry

end NUMINAMATH_CALUDE_sugar_solution_concentration_increases_l3170_317059


namespace NUMINAMATH_CALUDE_complement_of_angle_alpha_l3170_317034

/-- Represents an angle in degrees, minutes, and seconds -/
structure AngleDMS where
  degrees : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Calculates the complement of an angle in DMS format -/
def angleComplement (α : AngleDMS) : AngleDMS :=
  sorry

/-- The given angle α -/
def α : AngleDMS := ⟨36, 14, 25⟩

/-- Theorem: The complement of angle α is 53°45'35" -/
theorem complement_of_angle_alpha :
  angleComplement α = ⟨53, 45, 35⟩ := by sorry

end NUMINAMATH_CALUDE_complement_of_angle_alpha_l3170_317034


namespace NUMINAMATH_CALUDE_mark_initial_punch_l3170_317099

/-- The amount of punch in gallons that Mark initially added to the bowl -/
def initial_punch : ℝ := 4

/-- The capacity of the punch bowl in gallons -/
def bowl_capacity : ℝ := 16

/-- The amount of punch Mark adds after his cousin drinks -/
def second_addition : ℝ := 4

/-- The amount of punch Sally drinks -/
def sally_drinks : ℝ := 2

/-- The amount of punch Mark adds to fill the bowl completely -/
def final_addition : ℝ := 12

theorem mark_initial_punch :
  initial_punch / 2 + second_addition - sally_drinks + final_addition = bowl_capacity :=
by sorry

end NUMINAMATH_CALUDE_mark_initial_punch_l3170_317099


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3170_317012

/-- The repeating decimal 5.8̄ -/
def repeating_decimal : ℚ := 5 + 8/9

/-- The fraction 53/9 -/
def target_fraction : ℚ := 53/9

/-- Theorem stating that the repeating decimal 5.8̄ is equal to the fraction 53/9 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3170_317012


namespace NUMINAMATH_CALUDE_difference_of_x_and_y_l3170_317097

theorem difference_of_x_and_y (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (diff_squares : x^2 - y^2 = 24) : 
  x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_x_and_y_l3170_317097


namespace NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l3170_317054

def U : Finset ℕ := {1,2,3,4,5,6,7}
def A : Finset ℕ := {1,3,5}
def B : Finset ℕ := {2,3,6}

theorem union_of_A_and_complement_of_B :
  A ∪ (U \ B) = {1,3,4,5,7} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l3170_317054


namespace NUMINAMATH_CALUDE_some_number_value_l3170_317091

theorem some_number_value (x y : ℝ) (hx : x = 12) 
  (heq : ((17.28 / x) / (3.6 * y)) = 2) : y = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3170_317091


namespace NUMINAMATH_CALUDE_alberts_expression_l3170_317006

theorem alberts_expression (p q r s t u : ℚ) : 
  p = 2 ∧ q = 3 ∧ r = 4 ∧ s = 5 ∧ t = 6 →
  p - (q - (r - (s * (t + u)))) = p - q - r - s * t + u →
  u = 4/3 := by sorry

end NUMINAMATH_CALUDE_alberts_expression_l3170_317006


namespace NUMINAMATH_CALUDE_trig_expression_equality_l3170_317063

theorem trig_expression_equality : 
  (Real.sin (24 * π / 180) * Real.cos (16 * π / 180) + Real.cos (156 * π / 180) * Real.sin (66 * π / 180)) / 
  (Real.sin (28 * π / 180) * Real.cos (12 * π / 180) + Real.cos (152 * π / 180) * Real.sin (72 * π / 180)) = 
  1 / Real.sin (80 * π / 180) := by sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l3170_317063


namespace NUMINAMATH_CALUDE_perpendicular_to_same_plane_implies_parallel_perpendicular_to_two_planes_implies_parallel_l3170_317090

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem 1: If two lines are perpendicular to the same plane, then they are parallel
theorem perpendicular_to_same_plane_implies_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel_lines m n :=
sorry

-- Theorem 2: If a line is perpendicular to two planes, then those planes are parallel
theorem perpendicular_to_two_planes_implies_parallel 
  (n : Line) (α β : Plane) :
  perpendicular n α → perpendicular n β → parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_same_plane_implies_parallel_perpendicular_to_two_planes_implies_parallel_l3170_317090


namespace NUMINAMATH_CALUDE_paving_stones_required_l3170_317088

/-- The minimum number of paving stones required to cover a rectangular courtyard -/
theorem paving_stones_required (courtyard_length courtyard_width stone_length stone_width : ℝ) 
  (courtyard_length_pos : 0 < courtyard_length)
  (courtyard_width_pos : 0 < courtyard_width)
  (stone_length_pos : 0 < stone_length)
  (stone_width_pos : 0 < stone_width)
  (h_courtyard_length : courtyard_length = 120)
  (h_courtyard_width : courtyard_width = 25.5)
  (h_stone_length : stone_length = 3.5)
  (h_stone_width : stone_width = 3) : 
  ⌈(courtyard_length * courtyard_width) / (stone_length * stone_width)⌉ = 292 := by
  sorry

end NUMINAMATH_CALUDE_paving_stones_required_l3170_317088


namespace NUMINAMATH_CALUDE_percentage_problem_l3170_317093

theorem percentage_problem (x : ℝ) (P : ℝ) : 
  x = 780 ∧ 
  (P / 100) * x = 0.15 * 1500 - 30 → 
  P = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3170_317093


namespace NUMINAMATH_CALUDE_circle_center_distance_to_line_l3170_317021

/-- A circle passing through (1, 2) and tangent to both coordinate axes has its center
    at distance 2√5/5 from the line 2x - y - 3 = 0 -/
theorem circle_center_distance_to_line :
  ∀ (a : ℝ), 
    (∃ (x y : ℝ), (x - a)^2 + (y - a)^2 = a^2 ∧ x = 1 ∧ y = 2) →  -- Circle passes through (1, 2)
    (∃ (x : ℝ), (x - a)^2 + a^2 = a^2) →                          -- Circle is tangent to x-axis
    (∃ (y : ℝ), a^2 + (y - a)^2 = a^2) →                          -- Circle is tangent to y-axis
    (|a - 3| / Real.sqrt 5 : ℝ) = 2 * Real.sqrt 5 / 5 :=
by sorry


end NUMINAMATH_CALUDE_circle_center_distance_to_line_l3170_317021


namespace NUMINAMATH_CALUDE_garden_transformation_cost_and_area_increase_l3170_317003

/-- Represents a rectangular garden with its dimensions and fence cost -/
structure RectGarden where
  length : ℝ
  width : ℝ
  fence_cost : ℝ

/-- Represents a square garden with its side length and fence cost -/
structure SquareGarden where
  side : ℝ
  fence_cost : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def rect_perimeter (g : RectGarden) : ℝ :=
  2 * (g.length + g.width)

/-- Calculates the area of a rectangular garden -/
def rect_area (g : RectGarden) : ℝ :=
  g.length * g.width

/-- Calculates the total fencing cost of a rectangular garden -/
def rect_fence_cost (g : RectGarden) : ℝ :=
  rect_perimeter g * g.fence_cost

/-- Calculates the area of a square garden -/
def square_area (g : SquareGarden) : ℝ :=
  g.side * g.side

/-- Calculates the total fencing cost of a square garden -/
def square_fence_cost (g : SquareGarden) : ℝ :=
  4 * g.side * g.fence_cost

/-- The main theorem to prove -/
theorem garden_transformation_cost_and_area_increase :
  let rect := RectGarden.mk 60 20 15
  let square := SquareGarden.mk (rect_perimeter rect / 4) 20
  square_fence_cost square - rect_fence_cost rect = 800 ∧
  square_area square - rect_area rect = 400 := by
  sorry


end NUMINAMATH_CALUDE_garden_transformation_cost_and_area_increase_l3170_317003


namespace NUMINAMATH_CALUDE_glen_village_count_l3170_317068

theorem glen_village_count (p h s c d : ℕ) : 
  p = 2 * h →  -- 2 people for each horse
  s = 5 * c →  -- 5 sheep for each cow
  d = 4 * p →  -- 4 ducks for each person
  p + h + s + c + d ≠ 47 :=
by sorry

end NUMINAMATH_CALUDE_glen_village_count_l3170_317068


namespace NUMINAMATH_CALUDE_polynomial_nonnegative_iff_equal_roots_l3170_317094

theorem polynomial_nonnegative_iff_equal_roots (a b c : ℝ) :
  (∀ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) ≥ 0) ↔ 
  (a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_nonnegative_iff_equal_roots_l3170_317094


namespace NUMINAMATH_CALUDE_max_inradius_value_l3170_317001

/-- A parabola with equation y^2 = 4x and focus at (1,0) -/
structure Parabola where
  equation : ℝ → ℝ → Prop := fun x y ↦ y^2 = 4*x
  focus : ℝ × ℝ := (1, 0)

/-- The inradius of a triangle formed by a point on the parabola, the focus, and the origin -/
def inradius (p : Parabola) (P : ℝ × ℝ) : ℝ :=
  sorry

/-- The maximum inradius of triangle OPF -/
def max_inradius (p : Parabola) : ℝ :=
  sorry

theorem max_inradius_value (p : Parabola) :
  max_inradius p = 2 * Real.sqrt 3 / 9 :=
sorry

end NUMINAMATH_CALUDE_max_inradius_value_l3170_317001


namespace NUMINAMATH_CALUDE_algebraic_identities_l3170_317043

theorem algebraic_identities (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hab : a ≠ b) :
  (a / (a - b) + b / (b - a) = 1) ∧
  (a^2 / (b^2 * c) * (-b * c^2 / (2 * a)) / (a / b) = -c) :=
sorry

end NUMINAMATH_CALUDE_algebraic_identities_l3170_317043


namespace NUMINAMATH_CALUDE_apple_boxes_l3170_317033

theorem apple_boxes (class_5A class_5B class_5C : ℕ) 
  (h1 : class_5A = 560)
  (h2 : class_5B = 595)
  (h3 : class_5C = 735) :
  let box_weight := Nat.gcd class_5A (Nat.gcd class_5B class_5C)
  (class_5A / box_weight, class_5B / box_weight, class_5C / box_weight) = (16, 17, 21) := by
  sorry

#check apple_boxes

end NUMINAMATH_CALUDE_apple_boxes_l3170_317033


namespace NUMINAMATH_CALUDE_original_price_after_discounts_l3170_317031

/-- Given an article sold at $144 after two successive discounts of 10% and 20%, 
    prove that its original price was $200. -/
theorem original_price_after_discounts (final_price : ℝ) 
  (h1 : final_price = 144)
  (discount1 : ℝ) (h2 : discount1 = 0.1)
  (discount2 : ℝ) (h3 : discount2 = 0.2) :
  ∃ (original_price : ℝ), 
    original_price = 200 ∧
    final_price = original_price * (1 - discount1) * (1 - discount2) :=
by
  sorry


end NUMINAMATH_CALUDE_original_price_after_discounts_l3170_317031


namespace NUMINAMATH_CALUDE_max_amount_is_7550_l3170_317020

-- Define the total value of chips bought
def total_value : ℕ := 10000

-- Define the chip denominations
def chip_50_value : ℕ := 50
def chip_200_value : ℕ := 200

-- Define the total number of chips lost
def total_chips_lost : ℕ := 30

-- Define the relationship between lost chips
axiom lost_chips_relation : ∃ (x y : ℕ), x = 3 * y ∧ x + y = total_chips_lost

-- Define the function to calculate the maximum amount received back
def max_amount_received : ℕ := 
  total_value - (7 * chip_200_value + 21 * chip_50_value)

-- Theorem to prove
theorem max_amount_is_7550 : max_amount_received = 7550 := by
  sorry

end NUMINAMATH_CALUDE_max_amount_is_7550_l3170_317020


namespace NUMINAMATH_CALUDE_total_carrots_is_twenty_l3170_317036

/-- The number of carrots grown by Sally -/
def sally_carrots : ℕ := 6

/-- The number of carrots grown by Fred -/
def fred_carrots : ℕ := 4

/-- The number of carrots grown by Mary -/
def mary_carrots : ℕ := 10

/-- The total number of carrots grown by Sally, Fred, and Mary -/
def total_carrots : ℕ := sally_carrots + fred_carrots + mary_carrots

theorem total_carrots_is_twenty : total_carrots = 20 := by sorry

end NUMINAMATH_CALUDE_total_carrots_is_twenty_l3170_317036


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_eighty_l3170_317028

theorem thirty_percent_less_than_eighty (x : ℝ) : x + (1/4) * x = 80 - (30/100) * 80 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_eighty_l3170_317028


namespace NUMINAMATH_CALUDE_gcd_count_for_product_252_l3170_317073

theorem gcd_count_for_product_252 (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 252) :
  ∃! (s : Finset ℕ+), s.card = 8 ∧ ∀ d, d ∈ s ↔ ∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 252 :=
sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_252_l3170_317073


namespace NUMINAMATH_CALUDE_number_of_possible_lists_l3170_317056

def number_of_balls : ℕ := 15
def list_length : ℕ := 4

theorem number_of_possible_lists :
  (number_of_balls ^ list_length : ℕ) = 50625 := by
sorry

end NUMINAMATH_CALUDE_number_of_possible_lists_l3170_317056


namespace NUMINAMATH_CALUDE_tan_negative_seven_pi_fourths_l3170_317071

theorem tan_negative_seven_pi_fourths : Real.tan (-7 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_seven_pi_fourths_l3170_317071


namespace NUMINAMATH_CALUDE_quadratic_root_cube_l3170_317052

theorem quadratic_root_cube (A B C : ℝ) (r s : ℝ) (h1 : A ≠ 0) :
  (A * r^2 + B * r + C = 0) →
  (A * s^2 + B * s + C = 0) →
  (r + s = -B / A) →
  (r * s = C / A) →
  let p := (B^3 - 3*A*B*C) / A^3
  ∃ q, (r^3)^2 + p*(r^3) + q = 0 ∧ (s^3)^2 + p*(s^3) + q = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_cube_l3170_317052


namespace NUMINAMATH_CALUDE_arrange_five_books_two_identical_l3170_317057

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (identical : ℕ) : ℕ :=
  (Nat.factorial total) / (Nat.factorial identical)

/-- Theorem: The number of ways to arrange 5 books, where 2 are identical, is 60 -/
theorem arrange_five_books_two_identical :
  arrange_books 5 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arrange_five_books_two_identical_l3170_317057


namespace NUMINAMATH_CALUDE_irrational_sqrt_three_rational_others_l3170_317024

theorem irrational_sqrt_three_rational_others : 
  (Irrational (Real.sqrt 3)) ∧ 
  (¬ Irrational (-8 : ℝ)) ∧ 
  (¬ Irrational (0.3070809 : ℝ)) ∧ 
  (¬ Irrational (22 / 7 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_irrational_sqrt_three_rational_others_l3170_317024


namespace NUMINAMATH_CALUDE_zlatoust_miass_distance_l3170_317050

theorem zlatoust_miass_distance :
  ∀ (g m k : ℝ), g > 0 → m > 0 → k > 0 →
  ∃ (x : ℝ), x > 0 ∧
  (x + 18) / k = (x - 18) / m ∧
  (x + 25) / k = (x - 25) / g ∧
  (x + 8) / m = (x - 8) / g ∧
  x = 60 :=
by sorry

end NUMINAMATH_CALUDE_zlatoust_miass_distance_l3170_317050


namespace NUMINAMATH_CALUDE_star_expression_l3170_317010

/-- The star operation on real numbers -/
def star (a b : ℝ) : ℝ := a^2 + b^2 - a*b

/-- Theorem stating the result of (x+2y) ⋆ (y+3x) -/
theorem star_expression (x y : ℝ) : star (x + 2*y) (y + 3*x) = 7*x^2 + 3*y^2 + 3*x*y := by
  sorry

end NUMINAMATH_CALUDE_star_expression_l3170_317010


namespace NUMINAMATH_CALUDE_complex_product_real_l3170_317008

theorem complex_product_real (x : ℝ) : 
  let z₁ : ℂ := 1 + I
  let z₂ : ℂ := x - I
  (z₁ * z₂).im = 0 → x = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_l3170_317008


namespace NUMINAMATH_CALUDE_girls_walking_time_l3170_317079

/-- The time taken for two girls walking in opposite directions to be 120 km apart -/
theorem girls_walking_time (speed1 speed2 distance : ℝ) (h1 : speed1 = 7)
  (h2 : speed2 = 3) (h3 : distance = 120) : 
  distance / (speed1 + speed2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_girls_walking_time_l3170_317079


namespace NUMINAMATH_CALUDE_sum_at_13th_position_l3170_317066

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℕ
  is_permutation : Function.Bijective vertices

/-- The sum of numbers in a specific position across all rotations of a regular polygon -/
def sum_at_position (p : RegularPolygon 100) (pos : ℕ) : ℕ :=
  (Finset.range 100).sum (λ i => p.vertices ((i + pos - 1) % 100 : Fin 100))

/-- The main theorem -/
theorem sum_at_13th_position (p : RegularPolygon 100) 
  (h_vertices : ∀ i : Fin 100, p.vertices i = i.val + 1) : 
  sum_at_position p 13 = 10100 := by
  sorry

end NUMINAMATH_CALUDE_sum_at_13th_position_l3170_317066


namespace NUMINAMATH_CALUDE_complement_of_beta_l3170_317081

/-- Given two angles α and β that are complementary and α > β, 
    the complement of β is (α - β)/2 -/
theorem complement_of_beta (α β : ℝ) 
  (h1 : α + β = 90) -- α and β are complementary
  (h2 : α > β) : 
  90 - β = (α - β) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_beta_l3170_317081


namespace NUMINAMATH_CALUDE_set_A_elements_l3170_317018

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | x^2 + 2*x + a = 0}

-- State the theorem
theorem set_A_elements (a : ℝ) (h : 1 ∈ A a) : A a = {-3, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_A_elements_l3170_317018


namespace NUMINAMATH_CALUDE_remaining_payment_l3170_317080

theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (h1 : deposit = 150) (h2 : deposit_percentage = 0.1) : 
  (deposit / deposit_percentage) - deposit = 1350 := by
  sorry

end NUMINAMATH_CALUDE_remaining_payment_l3170_317080


namespace NUMINAMATH_CALUDE_waiter_customers_l3170_317055

/-- The initial number of customers -/
def initial_customers : ℕ := 47

/-- The number of customers who left -/
def customers_left : ℕ := 41

/-- The number of new customers who arrived -/
def new_customers : ℕ := 20

/-- The final number of customers -/
def final_customers : ℕ := 26

theorem waiter_customers : 
  initial_customers - customers_left + new_customers = final_customers :=
by sorry

end NUMINAMATH_CALUDE_waiter_customers_l3170_317055


namespace NUMINAMATH_CALUDE_product_divisible_by_twelve_l3170_317014

theorem product_divisible_by_twelve (a b c d : ℤ) 
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) : 
  12 ∣ ((a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d)) := by
sorry

end NUMINAMATH_CALUDE_product_divisible_by_twelve_l3170_317014


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_find_m_value_l3170_317085

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4*x - 5 ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x - m < 0}

-- Part 1
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 3) = {x : ℝ | x = -1 ∨ (3 ≤ x ∧ x ≤ 5)} := by sorry

-- Part 2
theorem find_m_value (h : A ∩ B m = {x : ℝ | -1 ≤ x ∧ x < 4}) : m = 8 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_find_m_value_l3170_317085


namespace NUMINAMATH_CALUDE_perimeter_of_arranged_rectangles_l3170_317096

theorem perimeter_of_arranged_rectangles :
  let small_length : ℕ := 9
  let small_width : ℕ := 3
  let horizontal_count : ℕ := 8
  let vertical_count : ℕ := 4
  let additional_edges : ℕ := 2 * 3
  let large_length : ℕ := small_length * horizontal_count
  let large_width : ℕ := small_width * vertical_count
  let perimeter : ℕ := 2 * (large_length + large_width) + additional_edges
  perimeter = 180 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_arranged_rectangles_l3170_317096


namespace NUMINAMATH_CALUDE_option1_better_than_option2_l3170_317058

def initial_amount : ℝ := 12000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_discounts : List ℝ := [0.15, 0.25, 0.10]
def option2_discounts : List ℝ := [0.25, 0.10, 0.10]

def apply_successive_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

theorem option1_better_than_option2 :
  apply_successive_discounts initial_amount option1_discounts <
  apply_successive_discounts initial_amount option2_discounts :=
sorry

end NUMINAMATH_CALUDE_option1_better_than_option2_l3170_317058


namespace NUMINAMATH_CALUDE_square_pyramid_sum_l3170_317070

/-- A square pyramid is a three-dimensional geometric shape with a square base and four triangular faces -/
structure SquarePyramid where
  base : Square
  apex : Point

/-- The number of faces in a square pyramid -/
def num_faces (sp : SquarePyramid) : ℕ := 5

/-- The number of edges in a square pyramid -/
def num_edges (sp : SquarePyramid) : ℕ := 8

/-- The number of vertices in a square pyramid -/
def num_vertices (sp : SquarePyramid) : ℕ := 5

/-- The sum of faces, edges, and vertices of a square pyramid is 18 -/
theorem square_pyramid_sum (sp : SquarePyramid) : 
  num_faces sp + num_edges sp + num_vertices sp = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_sum_l3170_317070


namespace NUMINAMATH_CALUDE_area_relationship_l3170_317044

/-- Triangle with sides 13, 14, and 15 inscribed in a circle -/
structure InscribedTriangle where
  -- Define the sides of the triangle
  a : ℝ := 13
  b : ℝ := 14
  c : ℝ := 15
  -- Define the areas of non-triangular regions
  A : ℝ
  B : ℝ
  C : ℝ
  -- C is the largest area
  hC_largest : C ≥ A ∧ C ≥ B

/-- The relationship between areas A, B, C, and the triangle area -/
theorem area_relationship (t : InscribedTriangle) : t.A + t.B + 84 = t.C := by
  sorry

end NUMINAMATH_CALUDE_area_relationship_l3170_317044


namespace NUMINAMATH_CALUDE_incorrect_inequality_l3170_317060

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ¬(-a + 2 > -b + 2) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l3170_317060


namespace NUMINAMATH_CALUDE_calculation_proof_l3170_317025

theorem calculation_proof :
  (- 2^2 * (1/4) + 4 / (4/9) + (-1)^2023 = 7) ∧
  (- 1^4 + |2 - (-3)^2| + (1/2) / (-(3/2)) = 17/3) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l3170_317025


namespace NUMINAMATH_CALUDE_min_value_not_five_max_value_half_x_gt_y_iff_x_over_c_gt_y_over_c_min_value_eight_l3170_317019

-- Statement 1
theorem min_value_not_five : 
  ¬ (∀ x : ℝ, x + 4 / (x - 1) ≥ 5) :=
sorry

-- Statement 2
theorem max_value_half : 
  (∀ x : ℝ, x * Real.sqrt (1 - x^2) ≤ 1/2) ∧ 
  (∃ x : ℝ, x * Real.sqrt (1 - x^2) = 1/2) :=
sorry

-- Statement 3
theorem x_gt_y_iff_x_over_c_gt_y_over_c :
  ∀ x y c : ℝ, c ≠ 0 → (x > y ↔ x / c^2 > y / c^2) :=
sorry

-- Statement 4
theorem min_value_eight :
  ∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 1 →
  (∀ a b : ℝ, a > 0 → b > 0 → a + 2*b = 1 → 2/a + 1/b ≥ 2/x + 1/y) →
  2/x + 1/y = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_not_five_max_value_half_x_gt_y_iff_x_over_c_gt_y_over_c_min_value_eight_l3170_317019


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3170_317078

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + x + 1

-- Define the point through which the tangent line passes
def point : ℝ × ℝ := (-1, 1)

-- Theorem statement
theorem tangent_line_equation :
  let (x₀, y₀) := point
  let m := (2 * x₀ + 1)  -- Slope of the tangent line
  (∀ x y, y - y₀ = m * (x - x₀)) ↔ (∀ x y, x + y = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3170_317078


namespace NUMINAMATH_CALUDE_usual_time_to_catch_bus_l3170_317061

/-- Given a person who misses the bus by 4 minutes when walking at 4/5 of their usual speed,
    their usual time to catch the bus is 16 minutes. -/
theorem usual_time_to_catch_bus (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (4/5 * usual_speed) * (usual_time + 4) = usual_speed * usual_time → usual_time = 16 := by
  sorry

#check usual_time_to_catch_bus

end NUMINAMATH_CALUDE_usual_time_to_catch_bus_l3170_317061


namespace NUMINAMATH_CALUDE_symmetric_line_l3170_317035

/-- Given a line L1 with equation x - 2y + 1 = 0 and a line of symmetry x = 1,
    the symmetric line L2 has the equation x + 2y - 3 = 0 -/
theorem symmetric_line (x y : ℝ) :
  (x - 2*y + 1 = 0) →  -- equation of L1
  (x = 1) →            -- line of symmetry
  (x + 2*y - 3 = 0)    -- equation of L2
:= by sorry

end NUMINAMATH_CALUDE_symmetric_line_l3170_317035


namespace NUMINAMATH_CALUDE_negative_one_squared_equals_negative_one_l3170_317065

theorem negative_one_squared_equals_negative_one : -1^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_squared_equals_negative_one_l3170_317065


namespace NUMINAMATH_CALUDE_equation_solution_l3170_317042

theorem equation_solution : 
  ∃ x : ℝ, 3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * x)) = 2800.0000000000005 ∧ x = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3170_317042


namespace NUMINAMATH_CALUDE_parallelogram_exclusive_properties_l3170_317072

structure Parallelogram where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ
  diagonals : Fin 2 → ℝ
  vertex_midpoint_segments : Fin 4 → ℝ
  has_symmetry_axes : Bool
  is_circumscribable : Bool
  is_inscribable : Bool

def all_sides_equal (p : Parallelogram) : Prop :=
  ∀ i j : Fin 4, p.sides i = p.sides j

def all_angles_equal (p : Parallelogram) : Prop :=
  ∀ i j : Fin 4, p.angles i = p.angles j

def all_diagonals_equal (p : Parallelogram) : Prop :=
  p.diagonals 0 = p.diagonals 1

def all_vertex_midpoint_segments_equal (p : Parallelogram) : Prop :=
  ∀ i j : Fin 4, p.vertex_midpoint_segments i = p.vertex_midpoint_segments j

def vertex_midpoint_segments_perpendicular (p : Parallelogram) : Prop :=
  sorry -- This would require more complex geometry definitions

def vertex_midpoint_segments_intersect (p : Parallelogram) : Prop :=
  sorry -- This would require more complex geometry definitions

theorem parallelogram_exclusive_properties (p : Parallelogram) : 
  ¬(all_sides_equal p ∧ all_angles_equal p) ∧
  ¬(all_sides_equal p ∧ all_diagonals_equal p) ∧
  ¬(all_sides_equal p ∧ all_vertex_midpoint_segments_equal p) ∧
  ¬(all_sides_equal p ∧ vertex_midpoint_segments_perpendicular p) ∧
  ¬(all_sides_equal p ∧ p.has_symmetry_axes) ∧
  ¬(all_sides_equal p ∧ p.is_circumscribable) ∧
  ¬(all_angles_equal p ∧ all_diagonals_equal p) ∧
  ¬(all_angles_equal p ∧ all_vertex_midpoint_segments_equal p) ∧
  ¬(all_angles_equal p ∧ vertex_midpoint_segments_perpendicular p) ∧
  ¬(all_angles_equal p ∧ p.has_symmetry_axes) ∧
  ¬(all_angles_equal p ∧ p.is_inscribable) ∧
  ¬(all_diagonals_equal p ∧ vertex_midpoint_segments_perpendicular p) ∧
  ¬(all_diagonals_equal p ∧ p.is_inscribable) ∧
  ¬(all_vertex_midpoint_segments_equal p ∧ vertex_midpoint_segments_perpendicular p) ∧
  ¬(all_vertex_midpoint_segments_equal p ∧ p.is_inscribable) ∧
  ¬(vertex_midpoint_segments_perpendicular p ∧ p.is_circumscribable) := by
  sorry

#check parallelogram_exclusive_properties

end NUMINAMATH_CALUDE_parallelogram_exclusive_properties_l3170_317072


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3170_317038

/-- Given a right-angled triangle PQR with the right angle at R, PR = 8, PQ = 15, and QR = 6,
    prove that the perimeter of the triangle is 24. -/
theorem triangle_perimeter (P Q R : ℝ × ℝ) : 
  (R.2 - P.2) * (Q.1 - P.1) = (Q.2 - P.2) * (R.1 - P.1) →  -- Right angle at R
  dist P R = 8 →
  dist P Q = 15 →
  dist Q R = 6 →
  dist P R + dist P Q + dist Q R = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3170_317038


namespace NUMINAMATH_CALUDE_triangular_pyramid_no_circular_cross_section_l3170_317051

-- Define the types of solids
inductive Solid
  | Cone
  | Cylinder
  | Sphere
  | TriangularPyramid

-- Define a predicate for having a circular cross-section
def has_circular_cross_section (s : Solid) : Prop :=
  match s with
  | Solid.Cone => True
  | Solid.Cylinder => True
  | Solid.Sphere => True
  | Solid.TriangularPyramid => False

-- Theorem statement
theorem triangular_pyramid_no_circular_cross_section :
  ∀ s : Solid, ¬(has_circular_cross_section s) ↔ s = Solid.TriangularPyramid :=
by sorry


end NUMINAMATH_CALUDE_triangular_pyramid_no_circular_cross_section_l3170_317051


namespace NUMINAMATH_CALUDE_intersection_of_B_and_complement_of_A_l3170_317000

def U : Set Int := {-1, 0, 1, 2, 3, 4, 5}
def A : Set Int := {1, 2, 5}
def B : Set Int := {0, 1, 2, 3}

theorem intersection_of_B_and_complement_of_A : B ∩ (U \ A) = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_B_and_complement_of_A_l3170_317000


namespace NUMINAMATH_CALUDE_polynomial_roots_imply_a_ge_5_l3170_317015

theorem polynomial_roots_imply_a_ge_5 (a b c : ℤ) (ha : a > 0) 
  (h_roots : ∃ x y : ℝ, x ≠ y ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ 
    a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) : 
  a ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_imply_a_ge_5_l3170_317015


namespace NUMINAMATH_CALUDE_third_row_sum_l3170_317004

def is_valid_grid (grid : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 9 ∧
  ∀ i' j', (i ≠ i' ∨ j ≠ j') → grid i j ≠ grid i' j'

theorem third_row_sum (grid : Matrix (Fin 3) (Fin 3) ℕ) 
  (h_valid : is_valid_grid grid)
  (h_row1 : (grid 0 0) * (grid 0 1) * (grid 0 2) = 60)
  (h_row2 : (grid 1 0) * (grid 1 1) * (grid 1 2) = 96) :
  (grid 2 0) + (grid 2 1) + (grid 2 2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_third_row_sum_l3170_317004


namespace NUMINAMATH_CALUDE_parallel_lines_j_value_l3170_317092

/-- Given that a line through (2, -9) and (j, 17) is parallel to 2x + 3y = 21, prove that j = -37 -/
theorem parallel_lines_j_value (j : ℝ) : 
  (∃ (m b : ℝ), ∀ x y, y = m * x + b → 
    (y = -9 ∧ x = 2 ∨ y = 17 ∧ x = j) ∧ 
    m = -2/3) → 
  j = -37 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_j_value_l3170_317092


namespace NUMINAMATH_CALUDE_team_selection_probability_l3170_317037

/-- The probability of randomly selecting a team that includes three specific players -/
theorem team_selection_probability 
  (total_players : ℕ) 
  (team_size : ℕ) 
  (specific_players : ℕ) 
  (h1 : total_players = 12) 
  (h2 : team_size = 6) 
  (h3 : specific_players = 3) :
  (Nat.choose (total_players - specific_players) (team_size - specific_players)) / 
  (Nat.choose total_players team_size) = 1 / 11 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_probability_l3170_317037


namespace NUMINAMATH_CALUDE_ashley_pies_eaten_l3170_317082

theorem ashley_pies_eaten (pies_per_day : ℕ) (days : ℕ) (remaining_pies : ℕ) :
  pies_per_day = 7 → days = 12 → remaining_pies = 34 →
  pies_per_day * days - remaining_pies = 50 := by
  sorry

end NUMINAMATH_CALUDE_ashley_pies_eaten_l3170_317082


namespace NUMINAMATH_CALUDE_anns_age_l3170_317039

theorem anns_age (ann barbara : ℕ) : 
  ann + barbara = 62 →  -- Sum of their ages is 62
  ann = 2 * (barbara - (ann - barbara)) →  -- Ann's age relation
  ann = 50 :=
by sorry

end NUMINAMATH_CALUDE_anns_age_l3170_317039


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3170_317032

-- Problem 1
theorem problem_1 : Real.sqrt 12 + (-1/3)⁻¹ + (-2)^2 = 2 * Real.sqrt 3 + 1 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (2*a / (a^2 - 4)) / (1 + (a - 2) / (a + 2)) = 1 / (a - 2) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3170_317032


namespace NUMINAMATH_CALUDE_sum_of_roots_is_one_l3170_317026

-- Define a quadratic polynomial Q(x) = ax^2 + bx + c
def Q (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem sum_of_roots_is_one 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, Q a b c (x^4 + x^2) ≥ Q a b c (x^3 + 1)) : 
  (- b) / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_one_l3170_317026


namespace NUMINAMATH_CALUDE_square_difference_formula_l3170_317016

theorem square_difference_formula (a b : ℚ) 
  (sum_eq : a + b = 3/4)
  (diff_eq : a - b = 1/8) : 
  a^2 - b^2 = 3/32 := by
sorry

end NUMINAMATH_CALUDE_square_difference_formula_l3170_317016


namespace NUMINAMATH_CALUDE_candy_distribution_l3170_317053

def is_valid_student_count (n : ℕ) : Prop :=
  n > 1 ∧ 129 % n = 0

theorem candy_distribution (total_candies : ℕ) (h_total : total_candies = 130) :
  ∀ n : ℕ, is_valid_student_count n ↔ (n = 3 ∨ n = 43 ∨ n = 129) :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3170_317053


namespace NUMINAMATH_CALUDE_highest_power_of_two_dividing_difference_of_sixth_powers_l3170_317005

theorem highest_power_of_two_dividing_difference_of_sixth_powers :
  ∃ k : ℕ, 2^k = (Nat.gcd (15^6 - 9^6) (2^64)) ∧ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_two_dividing_difference_of_sixth_powers_l3170_317005


namespace NUMINAMATH_CALUDE_no_solutions_to_radical_equation_l3170_317046

theorem no_solutions_to_radical_equation :
  ∀ x : ℝ, x ≥ 2 →
    ¬ (Real.sqrt (x + 7 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 2)) = 2) :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_to_radical_equation_l3170_317046


namespace NUMINAMATH_CALUDE_jerry_earnings_l3170_317095

/-- Calculates the total earnings for an independent contractor over a week -/
def total_earnings (pay_per_task : ℕ) (hours_per_task : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  (pay_per_task * hours_per_day * days_per_week) / hours_per_task

/-- Proves that Jerry's total earnings for the week equal $1400 -/
theorem jerry_earnings :
  total_earnings 40 2 10 7 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_jerry_earnings_l3170_317095


namespace NUMINAMATH_CALUDE_max_y_over_x_on_circle_l3170_317083

theorem max_y_over_x_on_circle (z : ℂ) (x y : ℝ) :
  z = x + y * I →
  x ≠ 0 →
  Complex.abs (z - 2) = Real.sqrt 3 →
  ∃ (k : ℝ), ∀ (w : ℂ) (u v : ℝ),
    w = u + v * I →
    u ≠ 0 →
    Complex.abs (w - 2) = Real.sqrt 3 →
    |v / u| ≤ k ∧
    k = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_y_over_x_on_circle_l3170_317083
