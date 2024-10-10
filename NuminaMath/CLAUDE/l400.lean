import Mathlib

namespace average_of_21_multiples_of_17_l400_40049

/-- The average of the first n multiples of a number -/
def average_of_multiples (n : ℕ) (x : ℕ) : ℚ :=
  (n * x * (n + 1)) / (2 * n)

/-- Theorem: The average of the first 21 multiples of 17 is 187 -/
theorem average_of_21_multiples_of_17 : 
  average_of_multiples 21 17 = 187 := by
  sorry

end average_of_21_multiples_of_17_l400_40049


namespace expression_perfect_square_iff_l400_40013

def factorial (n : ℕ) : ℕ := Nat.factorial n

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

def expression (n : ℕ) : ℕ := 
  (List.range (2*n + 1)).foldl (λ acc i => acc * factorial i) 1 / factorial (n + 1)

theorem expression_perfect_square_iff (n : ℕ) : 
  is_perfect_square (expression n) ↔ 
  (∃ k : ℕ, n = 4 * k * (k + 1)) ∨ (∃ k : ℕ, n = 2 * k * k - 1) :=
sorry

end expression_perfect_square_iff_l400_40013


namespace volume_between_concentric_spheres_l400_40009

theorem volume_between_concentric_spheres :
  let r₁ : ℝ := 3  -- radius of smaller sphere
  let r₂ : ℝ := 6  -- radius of larger sphere
  let V₁ := (4 / 3) * π * r₁^3  -- volume of smaller sphere
  let V₂ := (4 / 3) * π * r₂^3  -- volume of larger sphere
  V₂ - V₁ = 252 * π := by sorry

end volume_between_concentric_spheres_l400_40009


namespace min_value_quadratic_l400_40035

/-- The function f(x) = 3x^2 - 18x + 7 attains its minimum value when x = 3. -/
theorem min_value_quadratic (x : ℝ) : 
  ∃ (min : ℝ), ∀ y : ℝ, 3 * x^2 - 18 * x + 7 ≥ 3 * min^2 - 18 * min + 7 ∧ 
  (3 * min^2 - 18 * min + 7 = 3 * 3^2 - 18 * 3 + 7) := by
  sorry

end min_value_quadratic_l400_40035


namespace parabola_symmetry_l400_40001

/-- Represents a parabola in the form y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shift a parabola horizontally and vertically -/
def shift (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h - dx, k := p.k + dy }

/-- Reflect a parabola about the x-axis -/
def reflect_x (p : Parabola) : Parabola :=
  { a := -p.a, h := p.h, k := -p.k }

theorem parabola_symmetry (A B C : Parabola) :
  A = reflect_x B →
  C = shift B 2 1 →
  C.a = 2 ∧ C.h = -1 ∧ C.k = -1 →
  A.a = -2 ∧ A.h = 1 ∧ A.k = 2 := by sorry

end parabola_symmetry_l400_40001


namespace real_solutions_quadratic_l400_40024

theorem real_solutions_quadratic (x : ℝ) : 
  (∃ y : ℝ, 3 * y^2 + 5 * x * y - 2 * x + 8 = 0) ↔ (x ≤ -12/5 ∨ x ≥ 8/5) :=
sorry

end real_solutions_quadratic_l400_40024


namespace meeting_distance_l400_40014

-- Define the speeds and distance
def xiaoBinSpeed : ℝ := 15
def xiaoMingSpeed : ℝ := 5
def distanceToSchool : ℝ := 30

-- Define the theorem
theorem meeting_distance :
  let totalDistance : ℝ := 2 * distanceToSchool
  let meetingTime : ℝ := totalDistance / (xiaoBinSpeed + xiaoMingSpeed)
  let xiaoMingDistance : ℝ := meetingTime * xiaoMingSpeed
  xiaoMingDistance = 15 := by sorry

end meeting_distance_l400_40014


namespace equal_roots_quadratic_l400_40062

/-- 
Given a quadratic equation x^2 - 2x + k = 0, 
if it has two equal real roots, then k = 1.
-/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y + k = 0 → y = x) → 
  k = 1 := by
sorry

end equal_roots_quadratic_l400_40062


namespace shifted_increasing_interval_l400_40095

-- Define a function f
variable (f : ℝ → ℝ)

-- Define what it means for f to be increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem shifted_increasing_interval :
  IncreasingOn f (-2) 3 → IncreasingOn (fun x ↦ f (x + 4)) (-6) (-1) := by
  sorry

end shifted_increasing_interval_l400_40095


namespace f_has_three_zeros_l400_40070

/-- The cubic function f(x) = x^3 + ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- Theorem stating that f has exactly 3 zeros if and only if a is in the interval (-∞, -3) -/
theorem f_has_three_zeros (a : ℝ) :
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔
  a < -3 :=
sorry

end f_has_three_zeros_l400_40070


namespace proportional_segments_l400_40045

theorem proportional_segments (a b c d : ℝ) :
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a / b = c / d) →
  a = 6 → b = 9 → c = 12 →
  d = 18 := by
sorry

end proportional_segments_l400_40045


namespace difference_of_squares_153_147_l400_40053

theorem difference_of_squares_153_147 : 153^2 - 147^2 = 1800 := by
  sorry

end difference_of_squares_153_147_l400_40053


namespace mean_of_data_l400_40012

def data : List ℕ := [7, 5, 3, 5, 10]

theorem mean_of_data : (data.sum : ℚ) / data.length = 6 := by sorry

end mean_of_data_l400_40012


namespace anne_heavier_than_douglas_l400_40086

/-- Anne's weight in pounds -/
def anne_weight : ℕ := 67

/-- Douglas's weight in pounds -/
def douglas_weight : ℕ := 52

/-- The difference in weight between Anne and Douglas -/
def weight_difference : ℕ := anne_weight - douglas_weight

/-- Theorem stating that Anne is 15 pounds heavier than Douglas -/
theorem anne_heavier_than_douglas : weight_difference = 15 := by
  sorry

end anne_heavier_than_douglas_l400_40086


namespace integer_bounds_l400_40083

theorem integer_bounds (x : ℤ) 
  (h1 : 5 < x ∧ x < 21)
  (h2 : x < 18)
  (h3 : 13 > x ∧ x > 2)
  (h4 : 12 > x ∧ x > 9)
  (h5 : x + 1 < 13) :
  ∀ y : ℤ, x > y → y ≤ 9 :=
by sorry

end integer_bounds_l400_40083


namespace infinite_sum_equals_four_l400_40060

open BigOperators

theorem infinite_sum_equals_four : 
  ∑' (n : ℕ), (3 * n + 2) / (n * (n + 1) * (n + 3)) = 4 := by sorry

end infinite_sum_equals_four_l400_40060


namespace salary_increase_after_three_years_l400_40088

-- Define the annual raise rate
def annual_raise : ℝ := 1.15

-- Define the number of years
def years : ℕ := 3

-- Theorem statement
theorem salary_increase_after_three_years :
  (annual_raise ^ years - 1) * 100 = 52.0875 := by
  sorry

end salary_increase_after_three_years_l400_40088


namespace expression_factorization_l400_40002

theorem expression_factorization (x : ℝ) :
  (16 * x^7 + 49 * x^5 - 9) - (4 * x^7 - 7 * x^5 - 9) = 4 * x^5 * (3 * x^2 + 14) := by
  sorry

end expression_factorization_l400_40002


namespace one_intersection_iff_a_in_set_l400_40087

-- Define the function
def f (a x : ℝ) : ℝ := a * x^2 - a * x + 3 * x + 1

-- Define the condition for exactly one intersection point
def has_one_intersection (a : ℝ) : Prop :=
  ∃! x, f a x = 0

-- Theorem statement
theorem one_intersection_iff_a_in_set :
  ∀ a : ℝ, has_one_intersection a ↔ a ∈ ({0, 1, 9} : Set ℝ) := by sorry

end one_intersection_iff_a_in_set_l400_40087


namespace optimal_line_and_minimum_value_l400_40047

/-- A line passing through the origin with positive slope -/
structure PositiveSlopeLine where
  slope : ℝ
  positive : slope > 0

/-- A circle in the first quadrant -/
structure FirstQuadrantCircle where
  center : ℝ × ℝ
  radius : ℝ
  in_first_quadrant : center.1 ≥ 0 ∧ center.2 ≥ 0

/-- Predicate for two circles touching a line at the same point -/
def circles_touch_line_at_same_point (C1 C2 : FirstQuadrantCircle) (l : PositiveSlopeLine) : Prop :=
  ∃ (x y : ℝ), (y = l.slope * x) ∧
    ((x - C1.center.1)^2 + (y - C1.center.2)^2 = C1.radius^2) ∧
    ((x - C2.center.1)^2 + (y - C2.center.2)^2 = C2.radius^2)

/-- Predicate for a circle touching the x-axis at (1, 0) -/
def circle_touches_x_axis_at_one (C : FirstQuadrantCircle) : Prop :=
  C.center.1 = 1 ∧ C.center.2 = C.radius

/-- Predicate for a circle touching the y-axis -/
def circle_touches_y_axis (C : FirstQuadrantCircle) : Prop :=
  C.center.1 = C.radius

/-- Main theorem -/
theorem optimal_line_and_minimum_value
  (C1 C2 : FirstQuadrantCircle)
  (h1 : circle_touches_x_axis_at_one C1)
  (h2 : circle_touches_y_axis C2)
  (h3 : ∀ l : PositiveSlopeLine, circles_touch_line_at_same_point C1 C2 l) :
  ∃ (l : PositiveSlopeLine),
    l.slope = 4/3 ∧
    (∀ l' : PositiveSlopeLine, 8 * C1.radius + 9 * C2.radius ≤ 8 * C1.radius + 9 * C2.radius) ∧
    8 * C1.radius + 9 * C2.radius = 7 :=
  sorry

end optimal_line_and_minimum_value_l400_40047


namespace f_max_value_l400_40078

-- Define the function
def f (t : ℝ) : ℝ := -6 * t^2 + 36 * t - 18

-- State the theorem
theorem f_max_value :
  (∃ (t_max : ℝ), ∀ (t : ℝ), f t ≤ f t_max) ∧
  (∃ (t_max : ℝ), f t_max = 36) ∧
  (f 3 = 36) :=
sorry

end f_max_value_l400_40078


namespace probability_not_snow_l400_40007

theorem probability_not_snow (p_snow : ℚ) (h : p_snow = 2 / 5) : 1 - p_snow = 3 / 5 := by
  sorry

end probability_not_snow_l400_40007


namespace sharon_has_13_plums_l400_40029

/-- The number of plums Sharon has -/
def sharons_plums : ℕ := 13

/-- The number of plums Allan has -/
def allans_plums : ℕ := 10

/-- The difference between Sharon's plums and Allan's plums -/
def plum_difference : ℕ := 3

/-- Theorem: Given the conditions, Sharon has 13 plums -/
theorem sharon_has_13_plums : 
  sharons_plums = allans_plums + plum_difference :=
by sorry

end sharon_has_13_plums_l400_40029


namespace parentheses_placement_l400_40080

theorem parentheses_placement : 90 - 72 / (6 + 3) = 82 := by sorry

end parentheses_placement_l400_40080


namespace set_operations_proof_l400_40052

def A : Set ℕ := {1, 2}
def B : Set ℕ := {0, 2}
def C : Set ℕ := {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}

theorem set_operations_proof :
  (A ∩ C ≠ ∅) ∧
  (A ∪ C ≠ C) ∧
  (B ∩ C = B) ∧
  (A ∪ B ≠ C) := by
  sorry

end set_operations_proof_l400_40052


namespace absolute_value_ratio_l400_40026

theorem absolute_value_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 8*a*b) :
  |((a+b)/(a-b))| = Real.sqrt 15 / 3 := by
  sorry

end absolute_value_ratio_l400_40026


namespace count_nonzero_monomials_l400_40021

/-- The number of monomials with nonzero coefficients in the expansion of (x+y+z)^2030 + (x-y-z)^2030 -/
def nonzero_monomials_count : ℕ := 1032256

/-- The exponent used in the expression -/
def exponent : ℕ := 2030

theorem count_nonzero_monomials :
  (∃ (x y z : ℝ), (x + y + z)^exponent + (x - y - z)^exponent ≠ 0) →
  nonzero_monomials_count = (exponent / 2 + 1)^2 :=
sorry

end count_nonzero_monomials_l400_40021


namespace fraction_product_l400_40015

theorem fraction_product : (2 : ℚ) / 3 * 3 / 5 * 4 / 7 * 5 / 9 = 8 / 63 := by
  sorry

end fraction_product_l400_40015


namespace train_meeting_time_l400_40094

/-- The time when the trains meet -/
def meeting_time : ℝ := 11

/-- The time when train A starts -/
def start_time_A : ℝ := 7

/-- The distance between stations A and B in km -/
def total_distance : ℝ := 155

/-- The speed of train A in km/h -/
def speed_A : ℝ := 20

/-- The speed of train B in km/h -/
def speed_B : ℝ := 25

/-- The start time of train B -/
def start_time_B : ℝ := 8

theorem train_meeting_time :
  start_time_B = 8 :=
by sorry

end train_meeting_time_l400_40094


namespace pencil_cost_difference_l400_40059

def joy_pencils : ℕ := 30
def colleen_pencils : ℕ := 50
def pencil_cost : ℕ := 4

theorem pencil_cost_difference : 
  colleen_pencils * pencil_cost - joy_pencils * pencil_cost = 80 := by
  sorry

end pencil_cost_difference_l400_40059


namespace granger_cisco_equal_spots_sum_equal_total_granger_cisco_ratio_one_to_one_l400_40075

/-- The number of spots Granger has -/
def granger_spots : ℕ := 54

/-- The number of spots Cisco has -/
def cisco_spots : ℕ := 54

/-- The total number of spots Granger and Cisco have combined -/
def total_spots : ℕ := 108

/-- Theorem stating that Granger and Cisco have the same number of spots -/
theorem granger_cisco_equal : granger_spots = cisco_spots := by sorry

/-- Theorem stating that the sum of Granger's and Cisco's spots equals the total -/
theorem spots_sum_equal_total : granger_spots + cisco_spots = total_spots := by sorry

/-- Theorem proving that the ratio of Granger's spots to Cisco's spots is 1:1 -/
theorem granger_cisco_ratio_one_to_one : 
  granger_spots / cisco_spots = 1 := by sorry

end granger_cisco_equal_spots_sum_equal_total_granger_cisco_ratio_one_to_one_l400_40075


namespace total_items_given_out_l400_40068

/-- The number of groups in Miss Davis's class -/
def num_groups : ℕ := 10

/-- The number of popsicle sticks given to each group -/
def popsicle_sticks_per_group : ℕ := 15

/-- The number of straws given to each group -/
def straws_per_group : ℕ := 20

/-- Theorem stating the total number of items given out by Miss Davis -/
theorem total_items_given_out :
  (num_groups * (popsicle_sticks_per_group + straws_per_group)) = 350 := by
  sorry

end total_items_given_out_l400_40068


namespace square_prism_volume_is_two_l400_40050

/-- A square prism with all vertices on a sphere -/
structure SquarePrismOnSphere where
  /-- The side length of the square base -/
  side_length : ℝ
  /-- The height of the prism -/
  height : ℝ
  /-- The radius of the sphere -/
  sphere_radius : ℝ
  /-- All vertices of the prism lie on the sphere -/
  vertices_on_sphere : side_length ^ 2 * 2 + height ^ 2 = (2 * sphere_radius) ^ 2
  /-- The height of the prism is 2 -/
  height_is_two : height = 2
  /-- The surface area of the sphere is 6π -/
  sphere_surface_area : 4 * Real.pi * sphere_radius ^ 2 = 6 * Real.pi

/-- The volume of a square prism -/
def prism_volume (p : SquarePrismOnSphere) : ℝ := p.side_length ^ 2 * p.height

/-- Theorem: The volume of the square prism on the sphere is 2 -/
theorem square_prism_volume_is_two (p : SquarePrismOnSphere) : prism_volume p = 2 := by
  sorry

end square_prism_volume_is_two_l400_40050


namespace dore_change_correct_l400_40097

/-- Calculate the change given to a customer after a purchase. -/
def calculate_change (pants_cost shirt_cost tie_cost amount_paid : ℕ) : ℕ :=
  amount_paid - (pants_cost + shirt_cost + tie_cost)

/-- Theorem stating that the change is correctly calculated for Mr. Doré's purchase. -/
theorem dore_change_correct :
  calculate_change 140 43 15 200 = 2 := by
  sorry

end dore_change_correct_l400_40097


namespace joes_fruit_spending_l400_40056

theorem joes_fruit_spending (total_money : ℚ) (chocolate_fraction : ℚ) (money_left : ℚ) : 
  total_money = 450 →
  chocolate_fraction = 1/9 →
  money_left = 220 →
  (total_money - chocolate_fraction * total_money - money_left) / total_money = 2/5 := by
sorry

end joes_fruit_spending_l400_40056


namespace unsold_books_count_l400_40006

/-- Proves that the number of unsold books is 36 given the sale conditions --/
theorem unsold_books_count (total_books : ℕ) : 
  (2 : ℚ) / 3 * total_books * (7 : ℚ) / 2 = 252 → 
  (1 : ℚ) / 3 * total_books = 36 := by
  sorry

end unsold_books_count_l400_40006


namespace f_inequality_implies_m_bound_l400_40003

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Theorem statement
theorem f_inequality_implies_m_bound (m : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f m x < -m + 4) →
  m < 1/7 := by
  sorry

end f_inequality_implies_m_bound_l400_40003


namespace dance_troupe_max_members_l400_40069

theorem dance_troupe_max_members :
  ∀ m : ℤ,
  (∃ k : ℤ, 25 * m = 31 * k + 7) →
  25 * m < 1300 →
  25 * m ≤ 875 :=
by sorry

end dance_troupe_max_members_l400_40069


namespace right_triangle_sets_l400_40073

theorem right_triangle_sets : ∃! (a b c : ℝ), (a = 6 ∧ b = 8 ∧ c = 13) ∧
  ¬(a^2 + b^2 = c^2) ∧
  (0.3^2 + 0.4^2 = 0.5^2) ∧
  (1^2 + 1^2 = (Real.sqrt 2)^2) ∧
  (8^2 + 15^2 = 17^2) :=
by
  sorry

#check right_triangle_sets

end right_triangle_sets_l400_40073


namespace age_difference_mandy_sarah_l400_40046

/-- Given the ages and relationships of Mandy's siblings, prove the age difference between Mandy and Sarah. -/
theorem age_difference_mandy_sarah :
  let mandy_age : ℕ := 3
  let tom_age : ℕ := 5 * mandy_age
  let julia_age : ℕ := tom_age - 3
  let max_age : ℕ := 2 * julia_age + 2
  let sarah_age : ℕ := max_age + 4
  sarah_age - mandy_age = 27 := by sorry

end age_difference_mandy_sarah_l400_40046


namespace smallest_max_sum_l400_40039

theorem smallest_max_sum (a b c d e : ℕ+) 
  (sum_eq : a + b + c + d + e = 2020) :
  1010 ≤ max (a + b) (max (b + c) (max (c + d) (d + e))) :=
sorry

end smallest_max_sum_l400_40039


namespace perimeter_pentagon_l400_40019

/-- Given a square PQRS and a triangle PZS, this theorem proves the perimeter of pentagon PQRSZ -/
theorem perimeter_pentagon (x : ℝ) : 
  let square_perimeter : ℝ := 120
  let triangle_perimeter : ℝ := 2 * x
  let pentagon_perimeter : ℝ := square_perimeter / 2 + triangle_perimeter - square_perimeter / 4
  pentagon_perimeter = 60 + 2 * x :=
by sorry

end perimeter_pentagon_l400_40019


namespace ellipse_and_line_problem_l400_40037

/-- An ellipse with center at origin and foci on x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_c_pos : 0 < c
  h_a_b : b < a
  h_c_eq : c^2 = a^2 - b^2

/-- A line intersecting the ellipse -/
structure IntersectingLine where
  m : ℝ  -- slope
  k : ℝ  -- y-intercept

/-- The problem statement -/
theorem ellipse_and_line_problem 
  (E : Ellipse) 
  (l : IntersectingLine) 
  (h_arithmetic : E.c + (E.a^2 / E.c + E.c) = 4 * E.c) 
  (h_midpoint : -2 = (l.m * -2 + l.k + -2) / 2 ∧ 1 = (l.m * -2 + l.k + 1) / 2) 
  (h_length : (4 * Real.sqrt 3)^2 = 2 * ((l.m * -2 + l.k + 2)^2 + 9)) :
  l.m = 1 ∧ l.k = 3 ∧ E.a^2 = 24 ∧ E.b^2 = 12 := by sorry

end ellipse_and_line_problem_l400_40037


namespace triangle_inequality_l400_40093

theorem triangle_inequality (A B C a b c : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧ 
  A < π/2 ∧ B < π/2 ∧ C < π/2 ∧
  a = 2 * Real.sin A ∧
  b = 2 * Real.sin B ∧
  c = 2 * Real.sin C →
  (a / (1 - Real.sin A)) + (b / (1 - Real.sin B)) + (c / (1 - Real.sin C)) ≥ 18 + 12 * Real.sqrt 3 :=
by sorry

end triangle_inequality_l400_40093


namespace parabola_tangent_intersections_l400_40010

/-- Given a parabola y = x^2 + bx + c, prove that the polynomial of least degree
    passing through the intersections of consecutive tangents is x^2 + bx + c - 1/4 -/
theorem parabola_tangent_intersections
  (b c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + b*x + c
  let y : ℕ → ℝ := λ j ↦ f j
  let tangent : ℕ → ℝ → ℝ := λ j x ↦ (2*j + b)*x - j^2 + c
  let I : ℕ → ℝ × ℝ := λ j ↦ (j + 1/2, (j + 1/2)^2 + b*(j + 1/2) + j + b/2 + c)
  ∀ j : ℕ, j ∈ Finset.range 9 →
    (λ x ↦ x^2 + b*x + c - 1/4) ((I j).1) = (I j).2 :=
by sorry

end parabola_tangent_intersections_l400_40010


namespace largest_three_digit_square_ending_identical_l400_40063

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ends_with_three_identical_nonzero_digits (n : ℕ) : Prop :=
  ∃ d : ℕ, d ≠ 0 ∧ n % 1000 = d * 100 + d * 10 + d

theorem largest_three_digit_square_ending_identical : 
  (is_three_digit 376) ∧ 
  (ends_with_three_identical_nonzero_digits (376^2)) ∧ 
  (∀ n : ℕ, is_three_digit n → ends_with_three_identical_nonzero_digits (n^2) → n ≤ 376) :=
sorry

end largest_three_digit_square_ending_identical_l400_40063


namespace unique_arrangement_l400_40028

-- Define the Letter type
inductive Letter
| A
| B

-- Define a function to represent whether a letter tells the truth
def tellsTruth (l : Letter) : Bool :=
  match l with
  | Letter.A => true
  | Letter.B => false

-- Define the statements made by each letter
def statement1 (l1 l2 l3 : Letter) : Prop :=
  (l1 = l2 ∧ l1 ≠ l3) ∨ (l1 = l3 ∧ l1 ≠ l2)

def statement2 (l1 l2 l3 : Letter) : Prop :=
  (l1 = Letter.A → l2 ≠ Letter.A) ∧ (l3 = Letter.A → l2 ≠ Letter.A)

def statement3 (l1 l2 l3 : Letter) : Prop :=
  (l1 = Letter.B ∧ l2 ≠ Letter.B ∧ l3 ≠ Letter.B) ∨
  (l1 ≠ Letter.B ∧ l2 = Letter.B ∧ l3 ≠ Letter.B) ∨
  (l1 ≠ Letter.B ∧ l2 ≠ Letter.B ∧ l3 = Letter.B)

-- Define the main theorem
theorem unique_arrangement :
  ∃! (l1 l2 l3 : Letter),
    (tellsTruth l1 → statement1 l1 l2 l3) ∧
    (¬tellsTruth l1 → ¬statement1 l1 l2 l3) ∧
    (tellsTruth l2 → statement2 l1 l2 l3) ∧
    (¬tellsTruth l2 → ¬statement2 l1 l2 l3) ∧
    (tellsTruth l3 → statement3 l1 l2 l3) ∧
    (¬tellsTruth l3 → ¬statement3 l1 l2 l3) ∧
    l1 = Letter.B ∧ l2 = Letter.A ∧ l3 = Letter.A :=
  by sorry

end unique_arrangement_l400_40028


namespace intersection_of_M_and_N_l400_40027

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 4, 6, 8, 10}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by
  sorry

end intersection_of_M_and_N_l400_40027


namespace cakes_served_yesterday_proof_l400_40098

def cakes_served_yesterday (lunch_today dinner_today total : ℕ) : ℕ :=
  total - (lunch_today + dinner_today)

theorem cakes_served_yesterday_proof :
  cakes_served_yesterday 5 6 14 = 3 := by
  sorry

end cakes_served_yesterday_proof_l400_40098


namespace milburg_population_l400_40032

theorem milburg_population : 
  let grown_ups : ℕ := 5256
  let children : ℕ := 2987
  grown_ups + children = 8243 := by sorry

end milburg_population_l400_40032


namespace square_triangle_equal_area_l400_40071

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) :
  square_perimeter = 64 →
  triangle_height = 32 →
  (square_perimeter / 4)^2 = 1/2 * x * triangle_height →
  x = 16 := by
  sorry

end square_triangle_equal_area_l400_40071


namespace statement_a_statement_c_l400_40008

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop :=
  (a^2 + a + 1) * x - y + 1 = 0

-- Define perpendicularity of two lines
def perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Theorem for statement A
theorem statement_a :
  perpendicular (((-1)^2 + (-1) + 1) / (-1)) (-1) :=
by sorry

-- Theorem for statement C
theorem statement_c (a : ℝ) :
  line_l a 0 1 :=
by sorry

end statement_a_statement_c_l400_40008


namespace q_satisfies_conditions_l400_40096

/-- A quadratic polynomial that satisfies specific conditions -/
def q (x : ℚ) : ℚ := 5/4 * x^2 + 3/4 * x + 1

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-3) = 10 ∧ q 0 = 1 ∧ q 1 = 3 := by
  sorry

end q_satisfies_conditions_l400_40096


namespace skittles_indeterminate_l400_40044

/-- Given information about pencils and children, prove that the number of skittles per child cannot be determined. -/
theorem skittles_indeterminate (num_children : ℕ) (pencils_per_child : ℕ) (total_pencils : ℕ) 
  (h1 : num_children = 9)
  (h2 : pencils_per_child = 2)
  (h3 : total_pencils = 18)
  (h4 : num_children * pencils_per_child = total_pencils) :
  ∀ (skittles_per_child : ℕ), ∃ (other_skittles_per_child : ℕ), 
    other_skittles_per_child ≠ skittles_per_child ∧ 
    (∀ (total_skittles : ℕ), total_skittles = num_children * skittles_per_child → 
      total_skittles = num_children * other_skittles_per_child) :=
by
  sorry

end skittles_indeterminate_l400_40044


namespace rectangle_perimeter_l400_40092

/-- Given a rectangle with length thrice its breadth and area 588 square meters,
    prove that its perimeter is 112 meters. -/
theorem rectangle_perimeter (breadth : ℝ) (length : ℝ) (area : ℝ) (perimeter : ℝ) : 
  length = 3 * breadth →
  area = 588 →
  area = length * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 112 := by
  sorry

end rectangle_perimeter_l400_40092


namespace inverse_as_linear_combination_l400_40017

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, 1; 0, 4]

theorem inverse_as_linear_combination :
  ∃ (c d : ℚ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) ∧ c = -1/12 ∧ d = 7/12 := by
  sorry

end inverse_as_linear_combination_l400_40017


namespace complex_fraction_equality_l400_40057

theorem complex_fraction_equality (a b : ℂ) 
  (h : (a + b) / (a - b) + (a - b) / (a + b) = 2) :
  (a^4 + b^4) / (a^4 - b^4) + (a^4 - b^4) / (a^4 + b^4) = 2 :=
by sorry

end complex_fraction_equality_l400_40057


namespace tan_alpha_same_terminal_side_l400_40005

-- Define the angle α
def α : Real := sorry

-- Define the condition that the terminal side of α lies on y = -√3x
axiom terminal_side : ∀ (x y : Real), y = -Real.sqrt 3 * x → (∃ (r : Real), r > 0 ∧ x = r * Real.cos α ∧ y = r * Real.sin α)

-- Theorem 1: tan α = -√3
theorem tan_alpha : Real.tan α = -Real.sqrt 3 := by sorry

-- Define the set S of angles with the same terminal side as α
def S : Set Real := {θ | ∃ (k : ℤ), θ = k * Real.pi + 2 * Real.pi / 3}

-- Theorem 2: S is the set of all angles with the same terminal side as α
theorem same_terminal_side (θ : Real) : 
  (∀ (x y : Real), y = -Real.sqrt 3 * x → (∃ (r : Real), r > 0 ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ)) ↔ θ ∈ S := by sorry

end tan_alpha_same_terminal_side_l400_40005


namespace regular_triangular_prism_volume_l400_40066

/-- 
Given a regular triangular prism where:
1. The lateral edge is equal to the height of the base
2. The area of the cross-section passing through the lateral edge and height of the base is Q

Prove that the volume of the prism is Q * sqrt(Q/3)
-/
theorem regular_triangular_prism_volume (Q : ℝ) (Q_pos : 0 < Q) : 
  ∃ (volume : ℝ), volume = Q * Real.sqrt (Q / 3) ∧ 
  ∃ (lateral_edge base_height : ℝ), 
    lateral_edge = base_height ∧
    Q = lateral_edge * base_height ∧
    volume = (Real.sqrt 3 / 4 * lateral_edge^2) * lateral_edge :=
by sorry

end regular_triangular_prism_volume_l400_40066


namespace total_cats_l400_40077

theorem total_cats (white : ℕ) (black_percentage : ℚ) (grey : ℕ) :
  white = 2 →
  black_percentage = 1/4 →
  grey = 10 →
  ∃ (total : ℕ), 
    total = white + (black_percentage * total).floor + grey ∧
    total = 16 :=
by sorry

end total_cats_l400_40077


namespace power_product_l400_40025

theorem power_product (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_product_l400_40025


namespace average_marks_bcd_e_is_48_l400_40067

def average_marks_bcd_e (a b c d e : ℕ) : Prop :=
  -- The average marks of a, b, c is 48
  (a + b + c) / 3 = 48 ∧
  -- When d joins, the average becomes 47
  (a + b + c + d) / 4 = 47 ∧
  -- E has 3 more marks than d
  e = d + 3 ∧
  -- The marks of a is 43
  a = 43 →
  -- The average marks of b, c, d, e is 48
  (b + c + d + e) / 4 = 48

theorem average_marks_bcd_e_is_48 : 
  ∀ (a b c d e : ℕ), average_marks_bcd_e a b c d e :=
sorry

end average_marks_bcd_e_is_48_l400_40067


namespace z_magnitude_l400_40091

open Complex

/-- Euler's formula -/
axiom euler_formula (θ : ℝ) : exp (I * θ) = cos θ + I * sin θ

/-- The complex number z satisfies the given equation -/
def z : ℂ := by sorry

/-- The equation that z satisfies -/
axiom z_equation : (exp (I * Real.pi) - I) * z = 1

/-- The magnitude of z is √2/2 -/
theorem z_magnitude : abs z = Real.sqrt 2 / 2 := by sorry

end z_magnitude_l400_40091


namespace weight_moved_three_triples_l400_40031

/-- Calculates the total weight moved in three triples given the initial back squat, 
    back squat increase, front squat percentage, and triple percentage. -/
def total_weight_three_triples (initial_back_squat : ℝ) (back_squat_increase : ℝ) 
                               (front_squat_percentage : ℝ) (triple_percentage : ℝ) : ℝ :=
  let new_back_squat := initial_back_squat + back_squat_increase
  let front_squat := front_squat_percentage * new_back_squat
  let triple_weight := triple_percentage * front_squat
  3 * triple_weight

/-- Theorem stating that given the specific conditions, 
    the total weight moved in three triples is 540 kg. -/
theorem weight_moved_three_triples :
  total_weight_three_triples 200 50 0.8 0.9 = 540 := by
  sorry

end weight_moved_three_triples_l400_40031


namespace two_tangent_lines_l400_40079

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line passing through a point with a given slope
def line_through_point (p : Point) (slope : ℝ) (x y : ℝ) : Prop :=
  y - p.y = slope * (x - p.x)

-- Define the condition for a line to intersect the parabola at a single point
def single_intersection (p : Point) (slope : ℝ) : Prop :=
  ∃! q : Point, parabola q.x q.y ∧ line_through_point p slope q.x q.y

-- Theorem statement
theorem two_tangent_lines (p : Point) (h : parabola p.x p.y) :
  ∃! s : Finset ℝ, s.card = 2 ∧ ∀ k ∈ s, single_intersection p k :=
sorry

end two_tangent_lines_l400_40079


namespace smallest_floor_x_l400_40072

-- Define a tetrahedron type
structure Tetrahedron :=
  (a b c d e x : ℝ)

-- Define the conditions for a valid tetrahedron
def is_valid_tetrahedron (t : Tetrahedron) : Prop :=
  t.a = 4 ∧ t.b = 7 ∧ t.c = 20 ∧ t.d = 22 ∧ t.e = 28 ∧
  t.x > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b ∧
  t.a + t.d > t.e ∧ t.d + t.e > t.a ∧ t.e + t.a > t.d ∧
  t.b + t.d > t.x ∧ t.d + t.x > t.b ∧ t.x + t.b > t.d ∧
  t.b + t.e > t.c ∧ t.e + t.c > t.b ∧ t.c + t.b > t.e ∧
  t.c + t.d > t.x ∧ t.d + t.x > t.c ∧ t.x + t.c > t.d ∧
  t.c + t.e > t.x ∧ t.e + t.x > t.c ∧ t.x + t.c > t.e ∧
  t.d + t.e > t.x ∧ t.e + t.x > t.d ∧ t.x + t.d > t.e

-- Theorem statement
theorem smallest_floor_x (t : Tetrahedron) (h : is_valid_tetrahedron t) :
  ∀ (y : ℝ), (is_valid_tetrahedron {a := t.a, b := t.b, c := t.c, d := t.d, e := t.e, x := y} →
  ⌊t.x⌋ ≥ 8) ∧ (∃ (z : ℝ), is_valid_tetrahedron {a := t.a, b := t.b, c := t.c, d := t.d, e := t.e, x := z} ∧ ⌊z⌋ = 8) :=
sorry

end smallest_floor_x_l400_40072


namespace ratio_equality_l400_40042

theorem ratio_equality (x y z : ℝ) (h : x / 4 = y / 3 ∧ y / 3 = z / 2) :
  (x - y + 3 * z) / x = 7 / 4 := by
  sorry

end ratio_equality_l400_40042


namespace polynomial_real_root_l400_40054

theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, x^4 + a*x^3 - x^2 + a*x + 1 = 0) ↔ 
  (a ≤ -1/2 ∨ a ≥ 1/2) := by sorry

end polynomial_real_root_l400_40054


namespace multiple_of_72_l400_40020

theorem multiple_of_72 (a b : Nat) :
  (a ≤ 9) →
  (b ≤ 9) →
  (a * 10000 + 6790 + b) % 72 = 0 ↔ a = 3 ∧ b = 2 := by
  sorry

end multiple_of_72_l400_40020


namespace simplify_expression_l400_40099

theorem simplify_expression (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x^2 + 4*x + 4) = |x - 2| + |x + 2| := by
sorry

end simplify_expression_l400_40099


namespace visitors_scientific_notation_l400_40043

-- Define the number of visitors
def visitors : ℝ := 256000

-- Define the scientific notation representation
def scientific_notation : ℝ := 2.56 * (10 ^ 5)

-- Theorem to prove that the number of visitors is equal to its scientific notation representation
theorem visitors_scientific_notation : visitors = scientific_notation := by
  sorry

end visitors_scientific_notation_l400_40043


namespace min_h_for_circle_in_halfplane_l400_40036

/-- The minimum value of h for a circle (x-h)^2 + (y-1)^2 = 1 located within the plane region x + y + 1 ≥ 0 is √2 - 2. -/
theorem min_h_for_circle_in_halfplane :
  let C : ℝ → Set (ℝ × ℝ) := fun h => {p : ℝ × ℝ | (p.1 - h)^2 + (p.2 - 1)^2 = 1}
  let halfplane : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 + 1 ≥ 0}
  ∃ (h_min : ℝ), h_min = Real.sqrt 2 - 2 ∧
    (∀ h, (C h ⊆ halfplane) → h ≥ h_min) ∧
    (C h_min ⊆ halfplane) :=
by sorry

end min_h_for_circle_in_halfplane_l400_40036


namespace abc_product_l400_40090

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 24) (hac : a * c = 40) (hbc : b * c = 60) : a * b * c = 240 := by
  sorry

end abc_product_l400_40090


namespace bella_apples_per_day_l400_40030

/-- The number of apples Grace has left after 6 weeks -/
def apples_left : ℕ := 504

/-- The number of weeks -/
def weeks : ℕ := 6

/-- The fraction of apples Bella consumes from what Grace picks -/
def bella_fraction : ℚ := 1 / 3

/-- The number of apples Bella eats per day -/
def bella_daily_apples : ℕ := 6

theorem bella_apples_per_day :
  ∃ (grace_total : ℕ),
    grace_total - (bella_fraction * grace_total).num = apples_left ∧
    bella_daily_apples * (7 * weeks) = (bella_fraction * grace_total).num :=
by sorry

end bella_apples_per_day_l400_40030


namespace linear_diophantine_equation_solutions_l400_40082

theorem linear_diophantine_equation_solutions
  (a b c x₀ y₀ : ℤ)
  (h_coprime : Nat.gcd a.natAbs b.natAbs = 1)
  (h_solution : a * x₀ + b * y₀ = c) :
  ∀ x y : ℤ, a * x + b * y = c ↔ ∃ t : ℤ, x = x₀ + b * t ∧ y = y₀ - a * t :=
by sorry

end linear_diophantine_equation_solutions_l400_40082


namespace difference_in_dimes_l400_40051

/-- The number of quarters Susan has -/
def susan_quarters (p : ℚ) : ℚ := 7 * p + 3

/-- The number of quarters George has -/
def george_quarters (p : ℚ) : ℚ := 2 * p + 9

/-- The conversion rate from quarters to dimes -/
def quarter_to_dime : ℚ := 2.5

theorem difference_in_dimes (p : ℚ) :
  (susan_quarters p - george_quarters p) * quarter_to_dime = 12.5 * p - 15 := by
  sorry

end difference_in_dimes_l400_40051


namespace sum_of_solutions_l400_40034

theorem sum_of_solutions (x : ℕ) : 
  (∃ (s : Finset ℕ), 
    (∀ n ∈ s, 0 < n ∧ n ≤ 25 ∧ (7*(5*n - 3) : ℤ) ≡ 35 [ZMOD 9]) ∧
    (∀ m : ℕ, 0 < m ∧ m ≤ 25 ∧ (7*(5*m - 3) : ℤ) ≡ 35 [ZMOD 9] → m ∈ s) ∧
    s.sum id = 48) :=
by sorry

end sum_of_solutions_l400_40034


namespace matt_weight_matt_weight_is_80kg_l400_40016

/-- Given Matt's protein intake and requirements, calculate his weight. -/
theorem matt_weight (protein_percentage : ℝ) (protein_per_kg : ℝ) (powder_per_week : ℝ) : ℝ :=
  let protein_per_day := (powder_per_week / 7) * protein_percentage
  protein_per_day / protein_per_kg

/-- Prove that Matt weighs 80 kilograms given his protein intake and requirements. -/
theorem matt_weight_is_80kg : 
  matt_weight 0.80 2 1400 = 80 := by
  sorry

end matt_weight_matt_weight_is_80kg_l400_40016


namespace percent_of_m_equal_to_l_l400_40074

theorem percent_of_m_equal_to_l (j k l m x : ℝ) 
  (h1 : 1.25 * j = 0.25 * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : 1.75 * l = (x / 100) * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 75 := by
sorry

end percent_of_m_equal_to_l_l400_40074


namespace geometric_sequence_property_l400_40089

/-- A positive geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n > 0 ∧ ∃ r : ℝ, r > 0 ∧ b (n + 1) = r * b n

theorem geometric_sequence_property (b : ℕ → ℝ) (h : GeometricSequence b) :
  (b 1 * b 2 * b 3 * b 4 * b 5 * b 6) ^ (1/6 : ℝ) = (b 3 * b 4) ^ (1/2 : ℝ) := by
  sorry

end geometric_sequence_property_l400_40089


namespace inequality_exp_l400_40018

theorem inequality_exp (m n : ℝ) (h1 : m > n) (h2 : n > 0) : m * Real.exp m + n < n * Real.exp m + m := by
  sorry

end inequality_exp_l400_40018


namespace storks_joined_l400_40038

def initial_birds : ℕ := 3
def additional_birds : ℕ := 2

def total_birds : ℕ := initial_birds + additional_birds

def storks : ℕ := total_birds + 1

theorem storks_joined : storks = 6 := by sorry

end storks_joined_l400_40038


namespace decimal_to_fraction_l400_40058

theorem decimal_to_fraction : (2.36 : ℚ) = 59 / 25 := by
  sorry

end decimal_to_fraction_l400_40058


namespace min_a5_in_geometric_sequence_l400_40076

/-- A geometric sequence of 6 terms -/
def GeometricSequence (a : Fin 6 → ℝ) : Prop :=
  ∃ q : ℝ, ∀ i : Fin 5, a (i + 1) = a i * q

theorem min_a5_in_geometric_sequence 
  (a : Fin 6 → ℝ) 
  (h_geometric : GeometricSequence a) 
  (h_terms : ∃ i j : Fin 6, a i = 1 ∧ a j = 9) :
  ∃ a5_min : ℝ, a5_min = -27 ∧ ∀ a' : Fin 6 → ℝ, 
    GeometricSequence a' → 
    (∃ i j : Fin 6, a' i = 1 ∧ a' j = 9) → 
    a' 5 ≥ a5_min :=
sorry

end min_a5_in_geometric_sequence_l400_40076


namespace total_trees_is_433_l400_40084

/-- The total number of trees in the park after planting -/
def total_trees_after_planting (current_walnut current_oak current_maple new_walnut new_oak new_maple : ℕ) : ℕ :=
  current_walnut + current_oak + current_maple + new_walnut + new_oak + new_maple

/-- Theorem: The total number of trees after planting is 433 -/
theorem total_trees_is_433 :
  total_trees_after_planting 107 65 32 104 79 46 = 433 := by
  sorry


end total_trees_is_433_l400_40084


namespace diamond_equation_solution_l400_40081

-- Define the binary operation ◇ on nonzero real numbers
noncomputable def diamond (a b : ℝ) : ℝ := a / b

-- State the theorem
theorem diamond_equation_solution :
  (∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 →
    diamond a (diamond b c) = (diamond a b) * c) →
  (∀ (a : ℝ), a ≠ 0 → diamond a a = 1) →
  (∃! (y : ℝ), diamond 2024 (diamond 8 y) = 200 ∧ y = 200 / 253) :=
by sorry

end diamond_equation_solution_l400_40081


namespace honda_sales_calculation_l400_40085

/-- Given a car dealer's sales data, calculate the number of Hondas sold. -/
theorem honda_sales_calculation (total_cars : ℕ) 
  (audi_percent toyota_percent acura_percent : ℚ) : 
  total_cars = 200 →
  audi_percent = 15/100 →
  toyota_percent = 22/100 →
  acura_percent = 28/100 →
  (total_cars : ℚ) * (1 - (audi_percent + toyota_percent + acura_percent)) = 70 :=
by sorry

end honda_sales_calculation_l400_40085


namespace stratified_sampling_theorem_l400_40004

/-- Represents a school with its student count -/
structure School where
  students : ℕ

/-- Represents the sampling result for a school -/
structure SamplingResult where
  school : School
  sampleSize : ℕ

def totalStudents (schools : List School) : ℕ :=
  schools.foldl (fun acc school => acc + school.students) 0

def calculateSampleSize (school : School) (totalStudents : ℕ) (totalSampleSize : ℕ) : ℕ :=
  (school.students * totalSampleSize) / totalStudents

theorem stratified_sampling_theorem (schoolA schoolB schoolC : School)
    (h1 : schoolA.students = 3600)
    (h2 : schoolB.students = 5400)
    (h3 : schoolC.students = 1800)
    (totalSampleSize : ℕ)
    (h4 : totalSampleSize = 90) :
  let schools := [schoolA, schoolB, schoolC]
  let total := totalStudents schools
  let samplingResults := schools.map (fun school => 
    SamplingResult.mk school (calculateSampleSize school total totalSampleSize))
  samplingResults.map (fun result => result.sampleSize) = [30, 45, 15] := by
  sorry

end stratified_sampling_theorem_l400_40004


namespace binomial_expansion_coefficient_l400_40048

/-- 
Given a natural number n, such that in the expansion of (x³ + 1/x²)^n,
the coefficient of the fourth term is the largest,
prove that the coefficient of the term with x³ is 20.
-/
theorem binomial_expansion_coefficient (n : ℕ) : 
  (∃ k : ℕ, k = 3 ∧ 
    ∀ m : ℕ, m ≠ k → 
      Nat.choose n k ≥ Nat.choose n m) → 
  (∃ r : ℕ, Nat.choose n r * (3 * n - 5 * r) = 3 ∧ 
    Nat.choose n r = 20) := by
sorry

end binomial_expansion_coefficient_l400_40048


namespace tangent_line_and_inequality_l400_40011

noncomputable section

def f (x : ℝ) : ℝ := Real.log x + (x + 1) / x

theorem tangent_line_and_inequality (x : ℝ) (hx : x > 0) (hx1 : x ≠ 1) :
  (∃ (m b : ℝ), m * 1 + b = f 1 ∧ m = 2 ∧ ∀ t, f t = m * t + b) ∧
  f x > ((x + 1) * Real.log x) / (x - 1) :=
sorry

end tangent_line_and_inequality_l400_40011


namespace health_codes_survey_is_comprehensive_l400_40041

/-- Represents a survey option -/
inductive SurveyOption
  | MovieViewing
  | SeedGermination
  | RiverWaterQuality
  | StudentHealthCodes

/-- Characteristics of a survey that make it suitable for a comprehensive survey (census) -/
def isSuitableForComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.StudentHealthCodes => true
  | _ => false

/-- Theorem stating that the survey on health codes of students during an epidemic
    is suitable for a comprehensive survey (census) -/
theorem health_codes_survey_is_comprehensive :
  isSuitableForComprehensiveSurvey SurveyOption.StudentHealthCodes :=
by sorry

end health_codes_survey_is_comprehensive_l400_40041


namespace equation_solution_l400_40040

theorem equation_solution : ∃! x : ℝ, x > 0 ∧ (x - 3) / 12 = 5 / (x - 12) ∧ x = (15 + Real.sqrt 321) / 2 := by
  sorry

end equation_solution_l400_40040


namespace sarah_marriage_age_l400_40033

/-- Predicts the marriage age based on name length, current age, and birth month --/
def predictMarriageAge (nameLength : ℕ) (age : ℕ) (birthMonth : ℕ) : ℕ :=
  (nameLength + 2 * age) * birthMonth

theorem sarah_marriage_age :
  let sarahNameLength : ℕ := 5
  let sarahAge : ℕ := 9
  let sarahBirthMonth : ℕ := 7
  predictMarriageAge sarahNameLength sarahAge sarahBirthMonth = 161 := by
  sorry

end sarah_marriage_age_l400_40033


namespace point_B_coordinates_l400_40022

-- Define the vector a
def a : ℝ × ℝ := (1, 1)

-- Define point A
def A : ℝ × ℝ := (-3, -1)

-- Define the line y = 2x
def line (x : ℝ) : ℝ × ℝ := (x, 2 * x)

-- Define vector parallelism
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem point_B_coordinates :
  ∃ (x : ℝ), 
    let B := line x
    parallel (a) (B.1 - A.1, B.2 - A.2) →
    B = (2, 4) := by sorry

end point_B_coordinates_l400_40022


namespace inheritance_theorem_l400_40000

/-- Represents the inheritance distribution system -/
structure InheritanceSystem where
  total : ℕ  -- Total inheritance
  sons : ℕ   -- Number of sons
  share : ℕ  -- Share per son

/-- Calculates the share of the nth son -/
def nthSonShare (n : ℕ) (total : ℕ) : ℕ :=
  100 * n + (total - 100 * n) / 10

/-- Checks if all sons receive equal shares -/
def allSharesEqual (system : InheritanceSystem) : Prop :=
  ∀ i j, i ≤ system.sons → j ≤ system.sons →
    nthSonShare i system.total = nthSonShare j system.total

/-- The main theorem about the inheritance system -/
theorem inheritance_theorem (system : InheritanceSystem) :
  allSharesEqual system →
  system.total = 8100 ∧ system.sons = 9 ∧ system.share = 900 :=
sorry

end inheritance_theorem_l400_40000


namespace first_free_friday_after_college_start_l400_40065

/-- Represents a date with year, month, and day -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

/-- Returns true if the given year is a leap year -/
def isLeapYear (year : ℕ) : Bool := sorry

/-- Returns the number of days in a given month of a given year -/
def daysInMonth (year : ℕ) (month : ℕ) : ℕ := sorry

/-- Returns the day of the week for a given date -/
def dayOfWeek (d : Date) : DayOfWeek := sorry

/-- Returns the next date after the given date -/
def nextDate (d : Date) : Date := sorry

/-- Returns true if the given date is a Friday -/
def isFriday (d : Date) : Bool := sorry

/-- Returns true if the given date is a Free Friday -/
def isFreeFriday (d : Date) : Bool := sorry

/-- Finds the first Free Friday after the given start date -/
def firstFreeFriday (startDate : Date) : Date := sorry

theorem first_free_friday_after_college_start :
  let collegeStart := Date.mk 2023 2 1
  firstFreeFriday collegeStart = Date.mk 2023 3 31 := by sorry

end first_free_friday_after_college_start_l400_40065


namespace matt_jellybean_count_l400_40064

/-- The number of jellybeans each person has -/
structure JellybeanCount where
  steve : ℕ
  matt : ℕ
  matilda : ℕ

/-- The conditions of the jellybean problem -/
def jellybean_conditions (j : JellybeanCount) : Prop :=
  j.steve = 84 ∧
  j.matilda = 420 ∧
  j.matilda * 2 = j.matt ∧
  ∃ k : ℕ, j.matt = k * j.steve

theorem matt_jellybean_count (j : JellybeanCount) 
  (h : jellybean_conditions j) : j.matt = 840 := by
  sorry

end matt_jellybean_count_l400_40064


namespace beach_problem_l400_40055

/-- The number of people originally in the second row of the beach -/
def original_second_row : ℕ := 20

theorem beach_problem :
  let first_row : ℕ := 24
  let first_row_left : ℕ := 3
  let second_row_left : ℕ := 5
  let third_row : ℕ := 18
  let total_remaining : ℕ := 54
  (first_row - first_row_left) + (original_second_row - second_row_left) + third_row = total_remaining :=
by sorry

end beach_problem_l400_40055


namespace total_books_on_shelf_l400_40061

/-- Given a shelf with history books, geography books, and math books,
    prove that the total number of books is 100. -/
theorem total_books_on_shelf (history_books : ℕ) (geography_books : ℕ) (math_books : ℕ)
    (h1 : history_books = 32)
    (h2 : geography_books = 25)
    (h3 : math_books = 43) :
    history_books + geography_books + math_books = 100 := by
  sorry

end total_books_on_shelf_l400_40061


namespace sheep_grass_consumption_l400_40023

theorem sheep_grass_consumption 
  (num_sheep : ℕ) 
  (num_bags : ℕ) 
  (num_days : ℕ) 
  (h1 : num_sheep = 40) 
  (h2 : num_bags = 40) 
  (h3 : num_days = 40) :
  num_days = 40 := by
  sorry

end sheep_grass_consumption_l400_40023
