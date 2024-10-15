import Mathlib

namespace NUMINAMATH_CALUDE_cos_225_degrees_l930_93099

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l930_93099


namespace NUMINAMATH_CALUDE_triangle_point_inequalities_l930_93078

/-- Given a triangle ABC and a point P, prove two inequalities involving side lengths and distances --/
theorem triangle_point_inequalities 
  (A B C P : ℝ × ℝ) -- Points in 2D plane
  (a b c : ℝ) -- Side lengths of triangle ABC
  (α β γ : ℝ) -- Distances from P to A, B, C respectively
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) -- Triangle inequality
  (h_a : a = Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)) -- Definition of side length a
  (h_b : b = Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)) -- Definition of side length b
  (h_c : c = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) -- Definition of side length c
  (h_α : α = Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)) -- Definition of distance α
  (h_β : β = Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)) -- Definition of distance β
  (h_γ : γ = Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)) -- Definition of distance γ
  : (a * β * γ + b * γ * α + c * α * β ≥ a * b * c) ∧ 
    (α * b * c + β * c * a + γ * a * b ≥ Real.sqrt 3 * a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_point_inequalities_l930_93078


namespace NUMINAMATH_CALUDE_line_through_points_l930_93061

/-- Given two intersecting lines and their intersection point, prove the equation of the line passing through specific points. -/
theorem line_through_points (A₁ B₁ A₂ B₂ : ℝ) :
  (2 * A₁ + 3 * B₁ = 1) →  -- l₁ passes through P(2, 3)
  (2 * A₂ + 3 * B₂ = 1) →  -- l₂ passes through P(2, 3)
  (∀ x y : ℝ, A₁ * x + B₁ * y = 1 → 2 * x + 3 * y = 1) →  -- l₁ equation
  (∀ x y : ℝ, A₂ * x + B₂ * y = 1 → 2 * x + 3 * y = 1) →  -- l₂ equation
  ∀ x y : ℝ, (y - B₁) * (A₂ - A₁) = (x - A₁) * (B₂ - B₁) → 2 * x + 3 * y = 1 :=
by sorry


end NUMINAMATH_CALUDE_line_through_points_l930_93061


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l930_93085

theorem imaginary_part_of_complex_product : Complex.im ((5 + Complex.I) * (1 - Complex.I)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l930_93085


namespace NUMINAMATH_CALUDE_div_ratio_problem_l930_93029

theorem div_ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 3)
  (h2 : b / c = 3 / 4)
  (h3 : c / d = 2 / 3) :
  d / a = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_div_ratio_problem_l930_93029


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l930_93040

theorem sum_of_three_squares (p : ℕ) (hp : Nat.Prime p) (hp_neq_3 : p ≠ 3) :
  ∃ a b c : ℕ, 4 * p^2 + 1 = a^2 + b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l930_93040


namespace NUMINAMATH_CALUDE_closest_integer_to_expression_l930_93077

theorem closest_integer_to_expression : 
  let expr := (8^1500 + 8^1502) / (8^1501 + 8^1501)
  expr = 65/16 ∧ 
  ∀ n : ℤ, |expr - 4| ≤ |expr - n| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_expression_l930_93077


namespace NUMINAMATH_CALUDE_total_miles_equals_484_l930_93066

/-- The number of ladies in the walking group -/
def num_ladies : ℕ := 5

/-- The number of miles walked together by the group per day -/
def group_miles_per_day : ℕ := 3

/-- The number of days per week the group walks together -/
def group_days_per_week : ℕ := 6

/-- Jamie's additional miles walked per day -/
def jamie_additional_miles : ℕ := 2

/-- Sue's additional miles walked per day (half of Jamie's) -/
def sue_additional_miles : ℕ := jamie_additional_miles / 2

/-- Laura's additional miles walked every two days -/
def laura_additional_miles : ℕ := 1

/-- Melissa's additional miles walked every three days -/
def melissa_additional_miles : ℕ := 2

/-- Katie's additional miles walked per day -/
def katie_additional_miles : ℕ := 1

/-- The number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Calculate the total miles walked by all ladies in the group during a month -/
def total_miles_per_month : ℕ :=
  let jamie_miles := (group_miles_per_day * group_days_per_week + jamie_additional_miles * group_days_per_week) * weeks_per_month
  let sue_miles := (group_miles_per_day * group_days_per_week + sue_additional_miles * group_days_per_week) * weeks_per_month
  let laura_miles := (group_miles_per_day * group_days_per_week + laura_additional_miles * 3) * weeks_per_month
  let melissa_miles := (group_miles_per_day * group_days_per_week + melissa_additional_miles * 2) * weeks_per_month
  let katie_miles := (group_miles_per_day * group_days_per_week + katie_additional_miles * group_days_per_week) * weeks_per_month
  jamie_miles + sue_miles + laura_miles + melissa_miles + katie_miles

theorem total_miles_equals_484 : total_miles_per_month = 484 := by
  sorry

end NUMINAMATH_CALUDE_total_miles_equals_484_l930_93066


namespace NUMINAMATH_CALUDE_pizza_sales_total_l930_93069

theorem pizza_sales_total (pepperoni bacon cheese : ℕ) 
  (h1 : pepperoni = 2) 
  (h2 : bacon = 6) 
  (h3 : cheese = 6) : 
  pepperoni + bacon + cheese = 14 := by
  sorry

end NUMINAMATH_CALUDE_pizza_sales_total_l930_93069


namespace NUMINAMATH_CALUDE_arun_weight_estimation_l930_93049

/-- Arun's weight estimation problem -/
theorem arun_weight_estimation (x : ℝ) 
  (h1 : 65 < x)  -- Arun's lower bound
  (h2 : 60 < x ∧ x < 70)  -- Brother's estimation
  (h3 : x ≤ 68)  -- Mother's estimation
  (h4 : (65 + x) / 2 = 67)  -- Average of probable weights
  : x = 68 := by
  sorry

end NUMINAMATH_CALUDE_arun_weight_estimation_l930_93049


namespace NUMINAMATH_CALUDE_special_number_fraction_l930_93084

theorem special_number_fraction (numbers : List ℝ) (n : ℝ) :
  numbers.length = 21 ∧
  n ∈ numbers ∧
  n = 4 * ((numbers.sum - n) / 20) →
  n = (1 / 6) * numbers.sum :=
by sorry

end NUMINAMATH_CALUDE_special_number_fraction_l930_93084


namespace NUMINAMATH_CALUDE_charles_picked_50_pears_l930_93094

/-- The number of pears Charles picked -/
def pears_picked : ℕ := sorry

/-- The number of bananas Charles cooked -/
def bananas_cooked : ℕ := sorry

/-- The number of dishes Sandrine washed -/
def dishes_washed : ℕ := 160

theorem charles_picked_50_pears :
  (dishes_washed = bananas_cooked + 10) ∧
  (bananas_cooked = 3 * pears_picked) →
  pears_picked = 50 := by sorry

end NUMINAMATH_CALUDE_charles_picked_50_pears_l930_93094


namespace NUMINAMATH_CALUDE_rhombus_area_l930_93096

/-- A rhombus with specific properties. -/
structure Rhombus where
  /-- The side length of the rhombus. -/
  side_length : ℝ
  /-- The length of half of the shorter diagonal. -/
  half_shorter_diagonal : ℝ
  /-- The difference between the diagonals. -/
  diagonal_difference : ℝ
  /-- The side length is √109. -/
  side_length_eq : side_length = Real.sqrt 109
  /-- The diagonal difference is 12. -/
  diagonal_difference_eq : diagonal_difference = 12
  /-- The Pythagorean theorem holds for the right triangle formed by half of each diagonal and the side. -/
  pythagorean_theorem : half_shorter_diagonal ^ 2 + (half_shorter_diagonal + diagonal_difference / 2) ^ 2 = side_length ^ 2

/-- The area of a rhombus with the given properties is 364 square units. -/
theorem rhombus_area (r : Rhombus) : r.half_shorter_diagonal * (r.half_shorter_diagonal + r.diagonal_difference / 2) * 2 = 364 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l930_93096


namespace NUMINAMATH_CALUDE_largest_square_4digits_base7_l930_93033

/-- The largest integer whose square has exactly 4 digits in base 7 -/
def M : ℕ := 48

/-- Conversion of a natural number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem largest_square_4digits_base7 :
  (M^2 ≥ 7^3) ∧ (M^2 < 7^4) ∧ (∀ n : ℕ, n > M → n^2 ≥ 7^4) ∧ (toBase7 M = [6, 6]) := by
  sorry

end NUMINAMATH_CALUDE_largest_square_4digits_base7_l930_93033


namespace NUMINAMATH_CALUDE_gcd_1113_1897_l930_93095

theorem gcd_1113_1897 : Nat.gcd 1113 1897 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1113_1897_l930_93095


namespace NUMINAMATH_CALUDE_tan_product_eighths_pi_l930_93032

theorem tan_product_eighths_pi : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_eighths_pi_l930_93032


namespace NUMINAMATH_CALUDE_prob_good_or_excellent_grade_l930_93073

/-- Represents the types of students in the group -/
inductive StudentType
| Excellent
| Good
| Poor

/-- Represents the possible grades a student can receive -/
inductive Grade
| Excellent
| Good
| Satisfactory
| Unsatisfactory

/-- The total number of students -/
def totalStudents : ℕ := 21

/-- The number of excellent students -/
def excellentCount : ℕ := 5

/-- The number of good students -/
def goodCount : ℕ := 10

/-- The number of poorly performing students -/
def poorCount : ℕ := 6

/-- The probability of selecting an excellent student -/
def probExcellent : ℚ := excellentCount / totalStudents

/-- The probability of selecting a good student -/
def probGood : ℚ := goodCount / totalStudents

/-- The probability of selecting a poor student -/
def probPoor : ℚ := poorCount / totalStudents

/-- The probability of an excellent student receiving an excellent grade -/
def probExcellentGrade (s : StudentType) : ℚ :=
  match s with
  | StudentType.Excellent => 1
  | _ => 0

/-- The probability of a good student receiving a good or excellent grade -/
def probGoodOrExcellentGrade (s : StudentType) : ℚ :=
  match s with
  | StudentType.Excellent => 1
  | StudentType.Good => 1
  | StudentType.Poor => 1/3

/-- The probability of a randomly selected student receiving a good or excellent grade -/
theorem prob_good_or_excellent_grade :
  probExcellent * probExcellentGrade StudentType.Excellent +
  probGood * probGoodOrExcellentGrade StudentType.Good +
  probPoor * probGoodOrExcellentGrade StudentType.Poor = 17/21 := by
  sorry


end NUMINAMATH_CALUDE_prob_good_or_excellent_grade_l930_93073


namespace NUMINAMATH_CALUDE_sphere_volume_of_inscribed_parallelepiped_l930_93056

/-- The volume of a sphere circumscribing a rectangular parallelepiped with edge lengths 1, √2, and 3 -/
theorem sphere_volume_of_inscribed_parallelepiped : ∃ (V : ℝ),
  let a : ℝ := 1
  let b : ℝ := Real.sqrt 2
  let c : ℝ := 3
  let r : ℝ := Real.sqrt ((a^2 + b^2 + c^2) / 4)
  V = (4 / 3) * π * r^3 ∧ V = 4 * Real.sqrt 3 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_of_inscribed_parallelepiped_l930_93056


namespace NUMINAMATH_CALUDE_solution_set_l930_93036

theorem solution_set (x : ℝ) :
  x > 9 →
  Real.sqrt (x - 4 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 4 * Real.sqrt (x - 9)) - 3 →
  x ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l930_93036


namespace NUMINAMATH_CALUDE_min_distance_points_l930_93055

theorem min_distance_points (a b : ℝ) : 
  a = 2 → 
  (∃ (min_val : ℝ), min_val = 7 ∧ 
    ∀ (x : ℝ), |x - a| + |x - b| ≥ min_val) → 
  (b = -5 ∨ b = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_points_l930_93055


namespace NUMINAMATH_CALUDE_price_decrease_fifty_percent_l930_93043

/-- Calculates the percentage decrease in price given the original and new prices. -/
def percentage_decrease (original_price new_price : ℚ) : ℚ :=
  (original_price - new_price) / original_price * 100

/-- Theorem stating that the percentage decrease is 50% given the specific prices. -/
theorem price_decrease_fifty_percent (original_price new_price : ℚ) 
  (h1 : original_price = 1240)
  (h2 : new_price = 620) : 
  percentage_decrease original_price new_price = 50 := by
  sorry

#eval percentage_decrease 1240 620

end NUMINAMATH_CALUDE_price_decrease_fifty_percent_l930_93043


namespace NUMINAMATH_CALUDE_quadratic_factorization_l930_93035

theorem quadratic_factorization (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l930_93035


namespace NUMINAMATH_CALUDE_trig_special_angles_sum_l930_93089

theorem trig_special_angles_sum : 
  4 * Real.sin (30 * π / 180) - Real.sqrt 2 * Real.cos (45 * π / 180) - 
  Real.sqrt 3 * Real.tan (30 * π / 180) + 2 * Real.sin (60 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_special_angles_sum_l930_93089


namespace NUMINAMATH_CALUDE_chocolate_bar_breaks_chocolate_bar_40_pieces_l930_93019

/-- The minimum number of breaks required to separate a chocolate bar into individual pieces -/
def min_breaks (n : ℕ) : ℕ := n - 1

/-- Theorem stating that the minimum number of breaks for a chocolate bar with n pieces is n - 1 -/
theorem chocolate_bar_breaks (n : ℕ) (h : n > 0) : 
  min_breaks n = n - 1 := by
  sorry

/-- Corollary for the specific case of 40 pieces -/
theorem chocolate_bar_40_pieces : 
  min_breaks 40 = 39 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_breaks_chocolate_bar_40_pieces_l930_93019


namespace NUMINAMATH_CALUDE_evaluate_expression_l930_93063

-- Define the $ operation
def dollar (a b : ℝ) : ℝ := (a - b)^2

-- State the theorem
theorem evaluate_expression (x y : ℝ) : 
  dollar ((x + y)^2) ((y - x)^2) = 16 * x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l930_93063


namespace NUMINAMATH_CALUDE_investment_return_correct_l930_93074

def investment_return (n : ℕ+) : ℚ :=
  2^(n.val - 2)

theorem investment_return_correct :
  ∀ (n : ℕ+),
  (n = 1 → investment_return n = (1/2)) ∧
  (∀ (k : ℕ+), investment_return (k + 1) = 2 * investment_return k) :=
by sorry

end NUMINAMATH_CALUDE_investment_return_correct_l930_93074


namespace NUMINAMATH_CALUDE_Y_two_five_l930_93057

def Y (a b : ℝ) : ℝ := a^2 - 3*a*b + b^2 + 3

theorem Y_two_five : Y 2 5 = 2 := by sorry

end NUMINAMATH_CALUDE_Y_two_five_l930_93057


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l930_93070

/-- Given vectors a and b in ℝ³, if a is perpendicular to b, then x = -2 -/
theorem perpendicular_vectors (a b : ℝ × ℝ × ℝ) (h : a.1 = -1 ∧ a.2.1 = 2 ∧ a.2.2 = 1/2) 
  (k : b.1 = -3 ∧ b.2.2 = 2) (perp : a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = 0) :
  b.2.1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l930_93070


namespace NUMINAMATH_CALUDE_initial_kids_on_field_l930_93017

theorem initial_kids_on_field (initial : ℕ) (joined : ℕ) (total : ℕ) : 
  joined = 22 → total = 36 → total = initial + joined → initial = 14 := by
sorry

end NUMINAMATH_CALUDE_initial_kids_on_field_l930_93017


namespace NUMINAMATH_CALUDE_additional_men_needed_l930_93038

/-- Proves that given a work that can be finished by 12 men in 11 days,
    if the work is completed in 8 days (3 days earlier),
    then the number of additional men needed is 5. -/
theorem additional_men_needed
  (original_days : ℕ)
  (original_men : ℕ)
  (actual_days : ℕ)
  (h1 : original_days = 11)
  (h2 : original_men = 12)
  (h3 : actual_days = original_days - 3)
  : ∃ (additional_men : ℕ), 
    (original_men * original_days = (original_men + additional_men) * actual_days) ∧
    additional_men = 5 := by
  sorry

end NUMINAMATH_CALUDE_additional_men_needed_l930_93038


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l930_93044

def f (x : ℝ) := x^3 - 12*x + 12

theorem extreme_values_of_f :
  (∃ x, f x = -4 ∧ x = 2) ∧
  (∃ x, f x = 28) →
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ 28) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = -4) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = 28) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l930_93044


namespace NUMINAMATH_CALUDE_simplify_fraction_l930_93021

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l930_93021


namespace NUMINAMATH_CALUDE_estimate_value_l930_93015

theorem estimate_value : 5 < (3 * Real.sqrt 15 - Real.sqrt 3) * Real.sqrt (1/3) ∧
                         (3 * Real.sqrt 15 - Real.sqrt 3) * Real.sqrt (1/3) < 6 := by
  sorry

end NUMINAMATH_CALUDE_estimate_value_l930_93015


namespace NUMINAMATH_CALUDE_least_coins_in_purse_l930_93052

theorem least_coins_in_purse (n : ℕ) : 
  (n % 7 = 3 ∧ n % 5 = 4) → n ≥ 24 :=
sorry

end NUMINAMATH_CALUDE_least_coins_in_purse_l930_93052


namespace NUMINAMATH_CALUDE_mildred_oranges_l930_93010

/-- The number of oranges Mildred's father ate -/
def fatherAte : ℕ := 2

/-- The number of oranges Mildred has now -/
def currentOranges : ℕ := 75

/-- The initial number of oranges Mildred collected -/
def initialOranges : ℕ := currentOranges + fatherAte

theorem mildred_oranges : initialOranges = 77 := by
  sorry

end NUMINAMATH_CALUDE_mildred_oranges_l930_93010


namespace NUMINAMATH_CALUDE_class_test_result_l930_93031

theorem class_test_result (boys : ℕ) (grade5 : ℕ) : ∃ (low_grade : ℕ), low_grade ≤ 2 ∧ low_grade > 0 := by
  -- Define the number of girls
  let girls : ℕ := boys + 3
  
  -- Define the number of grade 4s
  let grade4 : ℕ := grade5 + 6
  
  -- Define the number of grade 3s
  let grade3 : ℕ := 2 * grade4
  
  -- Define the total number of students
  let total_students : ℕ := boys + girls
  
  -- Define the total number of positive grades (3, 4, 5)
  let total_positive_grades : ℕ := grade3 + grade4 + grade5
  
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_class_test_result_l930_93031


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l930_93023

theorem complex_magnitude_example : Complex.abs (Complex.mk (7/8) 3) = 25/8 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l930_93023


namespace NUMINAMATH_CALUDE_cone_volume_from_triangle_rotation_l930_93001

/-- The volume of a cone formed by rotating a right triangle -/
def cone_volume (S L : ℝ) : ℝ :=
  S * L

/-- Theorem: The volume of a cone formed by rotating a right triangle with area S
    around one of its legs is equal to SL, where L is the length of the circumference
    described by the intersection point of the medians during rotation -/
theorem cone_volume_from_triangle_rotation (S L : ℝ) (h1 : S > 0) (h2 : L > 0) :
  cone_volume S L = S * L :=
by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_triangle_rotation_l930_93001


namespace NUMINAMATH_CALUDE_sum_of_roots_is_36_l930_93020

def f (x : ℝ) : ℝ := (11 - x)^3 + (13 - x)^3 - (24 - 2*x)^3

theorem sum_of_roots_is_36 :
  ∃ (x₁ x₂ x₃ : ℝ), 
    (∀ x, f x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    x₁ + x₂ + x₃ = 36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_36_l930_93020


namespace NUMINAMATH_CALUDE_system_solution_l930_93013

theorem system_solution (u v : ℚ) 
  (eq1 : 3 * u - 4 * v = -14)
  (eq2 : 6 * u + 5 * v = 7) :
  2 * u - v = -63/13 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l930_93013


namespace NUMINAMATH_CALUDE_circle_passes_through_points_circle_uniqueness_l930_93026

/-- A circle is defined by the equation x^2 + y^2 + Dx + Ey + F = 0 --/
def Circle (D E F : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ x^2 + y^2 + D*x + E*y + F = 0

/-- The specific circle we're interested in --/
def SpecificCircle : ℝ × ℝ → Prop :=
  Circle (-4) (-6) 0

theorem circle_passes_through_points :
  SpecificCircle (0, 0) ∧
  SpecificCircle (4, 0) ∧
  SpecificCircle (-1, 1) :=
by sorry

/-- Uniqueness of the circle --/
theorem circle_uniqueness (D E F : ℝ) :
  Circle D E F (0, 0) →
  Circle D E F (4, 0) →
  Circle D E F (-1, 1) →
  ∀ (x y : ℝ), Circle D E F (x, y) ↔ SpecificCircle (x, y) :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_circle_uniqueness_l930_93026


namespace NUMINAMATH_CALUDE_marble_distribution_l930_93000

theorem marble_distribution (total_marbles : ℕ) (people : ℕ) : 
  total_marbles = 180 →
  (total_marbles / people : ℚ) - (total_marbles / (people + 2) : ℚ) = 1 →
  people = 18 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l930_93000


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l930_93006

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  a = 3 →
  b = 2 * Real.sqrt 6 →
  B = 2 * A →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b / Real.sin B = c / Real.sin C →
  (Real.cos A = Real.sqrt 6 / 3) ∧ (c = 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l930_93006


namespace NUMINAMATH_CALUDE_even_function_range_l930_93058

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_range (f : ℝ → ℝ) (h_even : IsEven f)
  (h_cond : ∀ x₁ x₂, x₁ ∈ Set.Ici 0 ∧ x₂ ∈ Set.Ici 0 ∧ x₁ ≠ x₂ → 
    (x₁ - x₂) * (f x₁ - f x₂) > 0)
  (m : ℝ) (h_ineq : f (m + 1) ≥ f 2) :
  m ∈ Set.Iic (-3) ∪ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_range_l930_93058


namespace NUMINAMATH_CALUDE_equal_integers_in_table_l930_93088

theorem equal_integers_in_table (t : Fin 10 → Fin 10 → ℤ) 
  (h : ∀ i j i' j', (i = i' ∧ |j - j'| = 1) ∨ (j = j' ∧ |i - i'| = 1) → |t i j - t i' j'| ≤ 5) :
  ∃ i j i' j', (i, j) ≠ (i', j') ∧ t i j = t i' j' :=
sorry

end NUMINAMATH_CALUDE_equal_integers_in_table_l930_93088


namespace NUMINAMATH_CALUDE_student_arrangement_count_l930_93087

/-- The number of ways to arrange students from three grades in a row --/
def arrange_students (grade1 : ℕ) (grade2 : ℕ) (grade3 : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial grade2) * (Nat.factorial grade3)

/-- Theorem stating the number of arrangements for the specific case --/
theorem student_arrangement_count :
  arrange_students 1 2 3 = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l930_93087


namespace NUMINAMATH_CALUDE_place_face_value_difference_l930_93053

def numeral : ℕ := 856973

def digit_of_interest : ℕ := 7

def place_value (n : ℕ) (d : ℕ) : ℕ :=
  (n / 10) % 10 * 10

def face_value (d : ℕ) : ℕ := d

theorem place_face_value_difference :
  place_value numeral digit_of_interest - face_value digit_of_interest = 63 := by
  sorry

end NUMINAMATH_CALUDE_place_face_value_difference_l930_93053


namespace NUMINAMATH_CALUDE_first_shirt_costs_15_l930_93065

/-- The cost of the first shirt given the conditions of the problem -/
def first_shirt_cost (second_shirt_cost : ℝ) : ℝ :=
  second_shirt_cost + 6

/-- The total cost of both shirts -/
def total_cost (second_shirt_cost : ℝ) : ℝ :=
  second_shirt_cost + first_shirt_cost second_shirt_cost

theorem first_shirt_costs_15 :
  ∃ (second_shirt_cost : ℝ),
    first_shirt_cost second_shirt_cost = 15 ∧
    total_cost second_shirt_cost = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_first_shirt_costs_15_l930_93065


namespace NUMINAMATH_CALUDE_exterior_angle_square_octagon_exterior_angle_is_135_l930_93083

/-- The exterior angle of a square and a regular octagon sharing a common side is 135°. -/
theorem exterior_angle_square_octagon : ℝ → Prop :=
  fun angle =>
    let square_angle := 90
    let octagon_interior_angle := 135
    let exterior_angle := 360 - square_angle - octagon_interior_angle
    exterior_angle = angle

/-- The theorem statement -/
theorem exterior_angle_is_135 : exterior_angle_square_octagon 135 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_square_octagon_exterior_angle_is_135_l930_93083


namespace NUMINAMATH_CALUDE_white_line_length_l930_93048

theorem white_line_length : 
  let blue_line_length : Float := 3.3333333333333335
  let difference : Float := 4.333333333333333
  let white_line_length : Float := blue_line_length + difference
  white_line_length = 7.666666666666667 := by
sorry

end NUMINAMATH_CALUDE_white_line_length_l930_93048


namespace NUMINAMATH_CALUDE_interest_calculation_l930_93007

/-- Calculates the compound interest earned over a period of time -/
def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- Proves that the compound interest earned on $2000 at 5% for 5 years is approximately $552.56 -/
theorem interest_calculation :
  let principal := 2000
  let rate := 0.05
  let years := 5
  abs (compoundInterest principal rate years - 552.56) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_interest_calculation_l930_93007


namespace NUMINAMATH_CALUDE_percentage_change_difference_l930_93011

theorem percentage_change_difference (initial_yes initial_no final_yes final_no : ℚ) :
  initial_yes = 60 / 100 →
  initial_no = 40 / 100 →
  final_yes = 80 / 100 →
  final_no = 20 / 100 →
  ∃ (min_change max_change : ℚ),
    min_change ≥ 0 ∧
    max_change ≥ 0 ∧
    min_change ≤ max_change ∧
    max_change - min_change = 20 / 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_change_difference_l930_93011


namespace NUMINAMATH_CALUDE_race_catch_up_time_l930_93008

/-- Given a 10-mile race with two runners, where the first runner's pace is 8 minutes per mile
    and the second runner's pace is 7 minutes per mile, prove that if the second runner stops
    after 56 minutes, they can remain stopped for 8 minutes before the first runner catches up. -/
theorem race_catch_up_time (race_length : ℝ) (pace1 pace2 stop_time : ℝ) :
  race_length = 10 →
  pace1 = 8 →
  pace2 = 7 →
  stop_time = 56 →
  let distance1 := stop_time / pace1
  let distance2 := stop_time / pace2
  let distance_diff := distance2 - distance1
  distance_diff * pace1 = 8 := by sorry

end NUMINAMATH_CALUDE_race_catch_up_time_l930_93008


namespace NUMINAMATH_CALUDE_solve_inequality_find_a_range_l930_93050

-- Define the function f
def f (x : ℝ) := |x + 2|

-- Part 1: Solve the inequality
theorem solve_inequality :
  {x : ℝ | 2 * f x < 4 - |x - 1|} = {x : ℝ | -7/3 < x ∧ x < -1} := by sorry

-- Part 2: Find the range of a
theorem find_a_range (m n : ℝ) (h1 : m + n = 1) (h2 : m > 0) (h3 : n > 0) :
  (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) ↔ -6 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_find_a_range_l930_93050


namespace NUMINAMATH_CALUDE_blocks_shared_l930_93030

theorem blocks_shared (start_blocks end_blocks : ℝ) (h1 : start_blocks = 86.0) (h2 : end_blocks = 127) : 
  end_blocks - start_blocks = 41 := by
sorry

end NUMINAMATH_CALUDE_blocks_shared_l930_93030


namespace NUMINAMATH_CALUDE_travel_ratio_l930_93054

theorem travel_ratio (total : ℕ) (europe : ℕ) (south_america : ℕ) (asia : ℕ)
  (h1 : total = 42)
  (h2 : europe = 20)
  (h3 : south_america = 10)
  (h4 : asia = 6)
  (h5 : europe + south_america + asia ≤ total) :
  asia * 2 = total - europe - south_america :=
by sorry

end NUMINAMATH_CALUDE_travel_ratio_l930_93054


namespace NUMINAMATH_CALUDE_additional_people_calculation_l930_93060

/-- Represents Carl's open house scenario -/
structure OpenHouse where
  confirmed_attendees : ℕ
  extravagant_bags : ℕ
  initial_average_bags : ℕ
  additional_bags_needed : ℕ

/-- Calculates the number of additional people Carl hopes will show up -/
def additional_people (oh : OpenHouse) : ℕ :=
  (oh.extravagant_bags + oh.initial_average_bags + oh.additional_bags_needed) - oh.confirmed_attendees

/-- Theorem stating that the number of additional people Carl hopes will show up
    is equal to the total number of gift bags minus the number of confirmed attendees -/
theorem additional_people_calculation (oh : OpenHouse) :
  additional_people oh = (oh.extravagant_bags + oh.initial_average_bags + oh.additional_bags_needed) - oh.confirmed_attendees :=
by
  sorry

#eval additional_people {
  confirmed_attendees := 50,
  extravagant_bags := 10,
  initial_average_bags := 20,
  additional_bags_needed := 60
}

end NUMINAMATH_CALUDE_additional_people_calculation_l930_93060


namespace NUMINAMATH_CALUDE_multiple_problem_l930_93067

theorem multiple_problem (n : ℝ) (m : ℝ) (h1 : n = 25.0) (h2 : 2 * n = m * n - 25) : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiple_problem_l930_93067


namespace NUMINAMATH_CALUDE_independence_day_bananas_l930_93003

theorem independence_day_bananas (total_children : ℕ) 
  (present_children : ℕ) (absent_children : ℕ) (bananas : ℕ) : 
  total_children = 260 →
  bananas = 4 * present_children →
  bananas = 2 * total_children →
  present_children + absent_children = total_children →
  absent_children = 130 := by
sorry

end NUMINAMATH_CALUDE_independence_day_bananas_l930_93003


namespace NUMINAMATH_CALUDE_james_black_spools_l930_93009

/-- Represents the number of spools of yarn needed to make one beret -/
def spools_per_beret : ℕ := 3

/-- Represents the number of spools of red yarn James has -/
def red_spools : ℕ := 12

/-- Represents the number of spools of blue yarn James has -/
def blue_spools : ℕ := 6

/-- Represents the number of berets James can make -/
def total_berets : ℕ := 11

/-- Calculates the number of black yarn spools James has -/
def black_spools : ℕ := 
  spools_per_beret * total_berets - (red_spools + blue_spools)

theorem james_black_spools : black_spools = 15 := by
  sorry

end NUMINAMATH_CALUDE_james_black_spools_l930_93009


namespace NUMINAMATH_CALUDE_mans_swimming_speed_l930_93045

/-- The swimming speed of a man in still water, given that it takes him twice as long to swim upstream
    than downstream in a stream with a speed of 2.5 km/h. -/
theorem mans_swimming_speed (v : ℝ) (s : ℝ) (h1 : s = 2.5) 
    (h2 : ∃ t : ℝ, t > 0 ∧ (v + s) * t = (v - s) * (2 * t)) : v = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_mans_swimming_speed_l930_93045


namespace NUMINAMATH_CALUDE_amount_distributed_l930_93062

theorem amount_distributed (A : ℝ) : 
  (A / 20 = A / 25 + 100) → A = 10000 := by
  sorry

end NUMINAMATH_CALUDE_amount_distributed_l930_93062


namespace NUMINAMATH_CALUDE_tile_border_ratio_l930_93014

theorem tile_border_ratio (t w : ℝ) (h : t > 0) (h' : w > 0) : 
  (900 * t^2) / ((30 * t + 30 * w)^2) = 81/100 → w/t = 1/9 := by
sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l930_93014


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l930_93071

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first five terms of the sequence is 20. -/
def SumOfFirstFiveIs20 (a : ℕ → ℚ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 = 20

theorem arithmetic_sequence_third_term
    (a : ℕ → ℚ)
    (h_arithmetic : IsArithmeticSequence a)
    (h_sum : SumOfFirstFiveIs20 a) :
    a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l930_93071


namespace NUMINAMATH_CALUDE_point_order_on_parabola_l930_93016

-- Define the parabola function
def parabola (x : ℝ) : ℝ := -(x - 1)^2 + 2

-- Define the theorem
theorem point_order_on_parabola (a b c : ℝ) :
  parabola a = -2 →
  parabola b = -2 →
  parabola c = -7 →
  a < b →
  c > 2 →
  a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_point_order_on_parabola_l930_93016


namespace NUMINAMATH_CALUDE_max_regions_circle_rectangle_triangle_l930_93075

/-- Represents a shape in the plane -/
inductive Shape
  | Circle
  | Rectangle
  | Triangle

/-- The number of regions created by intersecting shapes in the plane -/
def num_regions (shapes : List Shape) : ℕ :=
  sorry

/-- The maximum number of regions created by intersecting a circle, rectangle, and triangle -/
theorem max_regions_circle_rectangle_triangle :
  num_regions [Shape.Circle, Shape.Rectangle, Shape.Triangle] = 21 :=
by sorry

end NUMINAMATH_CALUDE_max_regions_circle_rectangle_triangle_l930_93075


namespace NUMINAMATH_CALUDE_a1_range_for_three_greater_terms_l930_93051

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

theorem a1_range_for_three_greater_terms
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : geometric_sequence a (1/2))
  (b : ℕ → ℝ)
  (h_b : ∀ n, b n = n / 2)
  (h_three : ∃! (s : Finset ℕ),
    s.card = 3 ∧ (∀ n ∈ s, a n > b n) ∧ (∀ n ∉ s, a n ≤ b n)) :
  6 < a 1 ∧ a 1 ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_a1_range_for_three_greater_terms_l930_93051


namespace NUMINAMATH_CALUDE_max_sum_of_seventh_powers_l930_93005

theorem max_sum_of_seventh_powers (a b c d : ℝ) 
  (h : a^6 + b^6 + c^6 + d^6 = 64) : 
  ∃ (M : ℝ), M = 128 ∧ a^7 + b^7 + c^7 + d^7 ≤ M ∧ 
  ∃ (a' b' c' d' : ℝ), a'^6 + b'^6 + c'^6 + d'^6 = 64 ∧ 
                        a'^7 + b'^7 + c'^7 + d'^7 = M :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_seventh_powers_l930_93005


namespace NUMINAMATH_CALUDE_age_difference_proof_l930_93076

-- Define variables for ages
variable (A B C : ℕ)

-- Define the condition that C is 10 years younger than A
def age_difference : Prop := C = A - 10

-- Define the difference in total ages
def total_age_difference : ℕ := (A + B) - (B + C)

-- Theorem to prove
theorem age_difference_proof (h : age_difference A C) : total_age_difference A B C = 10 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l930_93076


namespace NUMINAMATH_CALUDE_complex_conversion_l930_93022

theorem complex_conversion :
  3 * Real.sqrt 2 * Complex.exp ((-5 * π * Complex.I) / 4) = -3 - 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_conversion_l930_93022


namespace NUMINAMATH_CALUDE_cut_piece_weight_for_equal_copper_percent_l930_93090

/-- Represents an alloy with a given weight and copper percentage -/
structure Alloy where
  weight : ℝ
  copper_percent : ℝ

/-- Theorem stating the weight of the cut piece that equalizes copper percentages -/
theorem cut_piece_weight_for_equal_copper_percent 
  (alloy1 alloy2 : Alloy) 
  (h1 : alloy1.weight = 10)
  (h2 : alloy2.weight = 15)
  (h3 : alloy1.copper_percent ≠ alloy2.copper_percent) :
  ∃ x : ℝ, 
    x > 0 ∧ 
    x < min alloy1.weight alloy2.weight ∧
    ((alloy1.weight - x) * alloy1.copper_percent + x * alloy2.copper_percent) / alloy1.weight = 
    ((alloy2.weight - x) * alloy2.copper_percent + x * alloy1.copper_percent) / alloy2.weight → 
    x = 6 := by
  sorry

#check cut_piece_weight_for_equal_copper_percent

end NUMINAMATH_CALUDE_cut_piece_weight_for_equal_copper_percent_l930_93090


namespace NUMINAMATH_CALUDE_tangent_line_constant_l930_93098

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2

/-- The tangent line function -/
def tangent_line (x b : ℝ) : ℝ := -3*x + b

/-- Theorem stating that if the line y = -3x + b is tangent to the curve y = x^3 - 3x^2, then b = 1 -/
theorem tangent_line_constant (b : ℝ) : 
  (∃ x : ℝ, f x = tangent_line x b ∧ 
    (∀ y : ℝ, y ≠ x → f y ≠ tangent_line y b)) → 
  b = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_constant_l930_93098


namespace NUMINAMATH_CALUDE_no_valid_equation_l930_93064

/-- Represents a letter in the equation -/
structure Letter where
  value : Nat
  property : value < 10

/-- Represents a two-digit number as a pair of letters -/
structure TwoDigitNumber where
  tens : Letter
  ones : Letter
  different : tens ≠ ones

/-- Represents the equation АБ×ВГ = ДДЕЕ -/
structure Equation where
  ab : TwoDigitNumber
  vg : TwoDigitNumber
  d : Letter
  e : Letter
  different_letters : ab.tens ≠ ab.ones ∧ ab.tens ≠ vg.tens ∧ ab.tens ≠ vg.ones ∧
                      ab.ones ≠ vg.tens ∧ ab.ones ≠ vg.ones ∧ vg.tens ≠ vg.ones ∧
                      d ≠ e
  valid_multiplication : ab.tens.value * 10 + ab.ones.value *
                         (vg.tens.value * 10 + vg.ones.value) =
                         d.value * 1000 + d.value * 100 + e.value * 10 + e.value

theorem no_valid_equation : ¬ ∃ (eq : Equation), True := by
  sorry

end NUMINAMATH_CALUDE_no_valid_equation_l930_93064


namespace NUMINAMATH_CALUDE_linear_transformation_mapping_l930_93059

theorem linear_transformation_mapping (x : ℝ) :
  0 ≤ x ∧ x ≤ 1 → -1 ≤ 4 * x - 1 ∧ 4 * x - 1 ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_transformation_mapping_l930_93059


namespace NUMINAMATH_CALUDE_line_intercept_form_l930_93082

/-- A line passing through a point with a given direction vector has a specific intercept form -/
theorem line_intercept_form (P : ℝ × ℝ) (v : ℝ × ℝ) :
  P = (2, 3) →
  v = (2, -6) →
  ∃ (f : ℝ × ℝ → ℝ), f = (λ (x, y) => x / 3 + y / 9) ∧
    (∀ (Q : ℝ × ℝ), (∃ t : ℝ, Q = (P.1 + t * v.1, P.2 + t * v.2)) ↔ f Q = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_intercept_form_l930_93082


namespace NUMINAMATH_CALUDE_cheap_module_count_l930_93024

/-- Represents the stock of modules -/
structure ModuleStock where
  expensive_count : ℕ
  cheap_count : ℕ

/-- The cost of an expensive module -/
def expensive_cost : ℚ := 10

/-- The cost of a cheap module -/
def cheap_cost : ℚ := 3.5

/-- The total value of the stock -/
def total_value (stock : ModuleStock) : ℚ :=
  (stock.expensive_count : ℚ) * expensive_cost + (stock.cheap_count : ℚ) * cheap_cost

/-- The total count of modules in the stock -/
def total_count (stock : ModuleStock) : ℕ :=
  stock.expensive_count + stock.cheap_count

theorem cheap_module_count (stock : ModuleStock) :
  total_value stock = 45 ∧ total_count stock = 11 → stock.cheap_count = 10 :=
by sorry

end NUMINAMATH_CALUDE_cheap_module_count_l930_93024


namespace NUMINAMATH_CALUDE_x_value_when_y_is_two_l930_93047

theorem x_value_when_y_is_two (x y : ℝ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_two_l930_93047


namespace NUMINAMATH_CALUDE_convex_polygon_equal_division_l930_93091

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields and conditions for a convex polygon
  is_convex : Bool

/-- A straight line that divides a polygon -/
structure DividingLine where
  -- Add necessary fields for a dividing line

/-- A smaller polygon resulting from division -/
structure SmallerPolygon where
  perimeter : ℝ
  longest_side : ℝ

/-- Function to divide a convex polygon with a dividing line -/
def divide_polygon (p : ConvexPolygon) (l : DividingLine) : (SmallerPolygon × SmallerPolygon) :=
  sorry

/-- Theorem stating that any convex polygon can be divided into two smaller polygons
    with equal perimeters and equal longest sides -/
theorem convex_polygon_equal_division (p : ConvexPolygon) :
  ∃ (l : DividingLine),
    let (p1, p2) := divide_polygon p l
    p1.perimeter = p2.perimeter ∧ p1.longest_side = p2.longest_side :=
  sorry

end NUMINAMATH_CALUDE_convex_polygon_equal_division_l930_93091


namespace NUMINAMATH_CALUDE_salary_grade_increase_amount_l930_93079

/-- Represents the salary grade of an employee -/
def SalaryGrade := {s : ℝ // 1 ≤ s ∧ s ≤ 5}

/-- Calculates the hourly wage based on salary grade and base increase -/
def hourlyWage (s : SalaryGrade) (x : ℝ) : ℝ :=
  7.50 + x * (s.val - 1)

/-- States that the difference in hourly wage between grade 5 and grade 1 is $1.25 -/
def wageDifference (x : ℝ) : Prop :=
  hourlyWage ⟨5, by norm_num⟩ x - hourlyWage ⟨1, by norm_num⟩ x = 1.25

theorem salary_grade_increase_amount :
  ∃ x : ℝ, wageDifference x ∧ x = 0.3125 := by sorry

end NUMINAMATH_CALUDE_salary_grade_increase_amount_l930_93079


namespace NUMINAMATH_CALUDE_intersection_M_N_l930_93002

-- Define the sets M and N
def M : Set ℝ := {x | 2 - x > 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Define the interval [1, 2)
def interval_1_2 : Set ℝ := {x | 1 ≤ x ∧ x < 2}

-- State the theorem
theorem intersection_M_N : M ∩ N = interval_1_2 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l930_93002


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l930_93037

def i : ℂ := Complex.I

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_condition (a : ℝ) :
  let Z : ℂ := (a + i) / (1 + i)
  is_pure_imaginary Z → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l930_93037


namespace NUMINAMATH_CALUDE_t_formula_correct_t_2022_last_digit_l930_93039

/-- The number of unordered triples of non-empty and pairwise disjoint subsets of a set with n elements -/
def t (n : ℕ+) : ℚ :=
  (4^n.val - 3 * 3^n.val + 3 * 2^n.val - 1) / 6

/-- The closed form formula for t_n is correct -/
theorem t_formula_correct (n : ℕ+) :
  t n = (4^n.val - 3 * 3^n.val + 3 * 2^n.val - 1) / 6 := by sorry

/-- The last digit of t_2022 is 1 -/
theorem t_2022_last_digit :
  t 2022 % 1 = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_t_formula_correct_t_2022_last_digit_l930_93039


namespace NUMINAMATH_CALUDE_election_outcomes_count_l930_93081

def boys : ℕ := 28
def girls : ℕ := 22
def total_students : ℕ := boys + girls
def committee_size : ℕ := 5

theorem election_outcomes_count :
  (Nat.descFactorial total_students committee_size) -
  (Nat.descFactorial boys committee_size) -
  (Nat.descFactorial girls committee_size) = 239297520 :=
by sorry

end NUMINAMATH_CALUDE_election_outcomes_count_l930_93081


namespace NUMINAMATH_CALUDE_quadratic_root_equation_l930_93012

theorem quadratic_root_equation (x : ℝ) : 
  (∃ r : ℝ, x = (2 + r * Real.sqrt (4 - 4 * 3 * (-1))) / (2 * 3) ∧ r^2 = 1) →
  (3 * x^2 - 2 * x - 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_equation_l930_93012


namespace NUMINAMATH_CALUDE_trees_died_in_typhoon_l930_93027

theorem trees_died_in_typhoon (initial_trees : ℕ) (remaining_trees : ℕ) : 
  initial_trees = 20 → remaining_trees = 4 → initial_trees - remaining_trees = 16 := by
  sorry

end NUMINAMATH_CALUDE_trees_died_in_typhoon_l930_93027


namespace NUMINAMATH_CALUDE_range_of_a_l930_93086

open Real

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the theorem
theorem range_of_a (a : ℝ) (h : (¬p a) ∧ q a) : a > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l930_93086


namespace NUMINAMATH_CALUDE_whitney_cant_afford_all_items_l930_93068

def poster_cost : ℕ := 7
def notebook_cost : ℕ := 5
def bookmark_cost : ℕ := 3
def pencil_cost : ℕ := 1

def poster_quantity : ℕ := 3
def notebook_quantity : ℕ := 4
def bookmark_quantity : ℕ := 5
def pencil_quantity : ℕ := 2

def available_funds : ℕ := 2 * 20

theorem whitney_cant_afford_all_items :
  poster_cost * poster_quantity +
  notebook_cost * notebook_quantity +
  bookmark_cost * bookmark_quantity +
  pencil_cost * pencil_quantity > available_funds :=
by sorry

end NUMINAMATH_CALUDE_whitney_cant_afford_all_items_l930_93068


namespace NUMINAMATH_CALUDE_factor_theorem_l930_93097

theorem factor_theorem (h k : ℝ) : 
  (∃ c : ℝ, 3 * x^3 - h * x + k = c * (x + 3) * (x - 2)) →
  |3 * h - 2 * k| = 27 := by
sorry

end NUMINAMATH_CALUDE_factor_theorem_l930_93097


namespace NUMINAMATH_CALUDE_quadratic_roots_l930_93042

-- Define the quadratic equations
def eq1 (x : ℝ) : Prop := x^2 - x + 1 = 0
def eq2 (x : ℝ) : Prop := x * (x - 1) = 0
def eq3 (x : ℝ) : Prop := x^2 + 12*x = 0
def eq4 (x : ℝ) : Prop := x^2 + x = 1

-- Theorem stating that eq1 has no real roots while others have
theorem quadratic_roots :
  (¬ ∃ x : ℝ, eq1 x) ∧
  (∃ x : ℝ, eq2 x) ∧
  (∃ x : ℝ, eq3 x) ∧
  (∃ x : ℝ, eq4 x) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l930_93042


namespace NUMINAMATH_CALUDE_flower_count_l930_93041

theorem flower_count : 
  ∀ (flowers bees : ℕ), 
    bees = 3 → 
    bees = flowers - 2 → 
    flowers = 5 := by sorry

end NUMINAMATH_CALUDE_flower_count_l930_93041


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l930_93018

theorem dining_bill_calculation (total : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (food_price : ℝ) : 
  total = 158.40 ∧ 
  tax_rate = 0.10 ∧ 
  tip_rate = 0.20 ∧
  total = food_price * (1 + tax_rate) * (1 + tip_rate) →
  food_price = 120 := by
sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l930_93018


namespace NUMINAMATH_CALUDE_function_characterization_l930_93080

theorem function_characterization
  (f : ℕ → ℕ)
  (h : ∀ a b c d : ℕ, 2 * a * b = c^2 + d^2 →
       f (a + b) = f a + f b + f c + f d) :
  ∀ n : ℕ, f n = n^2 * f 1 :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l930_93080


namespace NUMINAMATH_CALUDE_race_finish_difference_l930_93028

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance : ℝ

/-- Represents the race with three runners -/
structure Race where
  runner1 : Runner
  runner2 : Runner
  runner3 : Runner
  constant_speed : Prop

/-- The difference in distance between two runners at the finish line -/
def distance_difference (r1 r2 : Runner) : ℝ :=
  r1.distance - r2.distance

/-- The theorem statement -/
theorem race_finish_difference (race : Race) 
  (h1 : distance_difference race.runner1 race.runner2 = 2)
  (h2 : distance_difference race.runner1 race.runner3 = 4)
  (h3 : race.constant_speed) :
  distance_difference race.runner2 race.runner3 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_difference_l930_93028


namespace NUMINAMATH_CALUDE_hoseok_calculation_l930_93092

theorem hoseok_calculation (x : ℝ) (h : 6 * x = 72) : x + 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_calculation_l930_93092


namespace NUMINAMATH_CALUDE_koschei_stopped_month_l930_93072

/-- The number of children Baba Yaga helps per month -/
def baba_yaga_rate : ℕ := 77

/-- The number of children Koschei helps per month -/
def koschei_rate : ℕ := 12

/-- The number of months between the start and end of the competition -/
def competition_duration : ℕ := 120

/-- The ratio of Baba Yaga's total good deeds to Koschei's at the end -/
def final_ratio : ℕ := 5

/-- Theorem stating when Koschei stopped doing good deeds -/
theorem koschei_stopped_month :
  ∃ (m : ℕ), m * koschei_rate * final_ratio = competition_duration * baba_yaga_rate ∧ m = 154 := by
  sorry

end NUMINAMATH_CALUDE_koschei_stopped_month_l930_93072


namespace NUMINAMATH_CALUDE_max_x_value_l930_93004

theorem max_x_value (x y z : ℝ) 
  (sum_eq : x + y + z = 9) 
  (prod_sum_eq : x*y + x*z + y*z = 20) : 
  x ≤ (18 + Real.sqrt 312) / 6 := by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l930_93004


namespace NUMINAMATH_CALUDE_max_abcd_is_one_l930_93034

theorem max_abcd_is_one (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : (1 + a) * (1 + b) * (1 + c) * (1 + d) = 16) :
  abcd ≤ 1 ∧ ∃ (a' b' c' d' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧
    (1 + a') * (1 + b') * (1 + c') * (1 + d') = 16 ∧ a' * b' * c' * d' = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_abcd_is_one_l930_93034


namespace NUMINAMATH_CALUDE_fraction_simplification_l930_93025

theorem fraction_simplification (a x : ℝ) :
  (Real.sqrt (a^2 + x^2) - (x^2 + a^2) / Real.sqrt (a^2 + x^2)) / (a^2 + x^2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l930_93025


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l930_93093

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

theorem minimum_value_implies_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x ≤ Real.exp 1 → f a x ≥ 3) ∧ 
  (∃ x : ℝ, 0 < x ∧ x ≤ Real.exp 1 ∧ f a x = 3) → 
  a = Real.exp 2 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l930_93093


namespace NUMINAMATH_CALUDE_solve_equation_l930_93046

theorem solve_equation (x : ℝ) : 3 * x + 15 = (1/3) * (6 * x + 45) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l930_93046
