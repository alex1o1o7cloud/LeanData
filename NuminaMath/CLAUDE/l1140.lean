import Mathlib

namespace softball_team_size_l1140_114085

theorem softball_team_size :
  ∀ (men women : ℕ),
  women = men + 4 →
  (men : ℚ) / (women : ℚ) = 7/11 →
  men + women = 18 :=
by
  sorry

end softball_team_size_l1140_114085


namespace line_through_points_l1140_114006

/-- Given three points on a line, find the y-coordinate of a fourth point on the same line -/
theorem line_through_points (x1 y1 x2 y2 x3 y3 x4 : ℝ) (h1 : y2 - y1 = (x2 - x1) * ((y3 - y1) / (x3 - x1))) 
  (h2 : y3 - y2 = (x3 - x2) * ((y3 - y1) / (x3 - x1))) : 
  let t := y1 + (x4 - x1) * ((y3 - y1) / (x3 - x1))
  (x1 = 2 ∧ y1 = 6 ∧ x2 = 5 ∧ y2 = 12 ∧ x3 = 8 ∧ y3 = 18 ∧ x4 = 20) → t = 42 := by
  sorry


end line_through_points_l1140_114006


namespace race_average_time_per_km_l1140_114044

theorem race_average_time_per_km (race_distance : ℝ) (first_half_time second_half_time : ℝ) :
  race_distance = 10 →
  first_half_time = 20 →
  second_half_time = 30 →
  (first_half_time + second_half_time) / race_distance = 5 := by
  sorry

end race_average_time_per_km_l1140_114044


namespace min_layoff_rounds_is_four_l1140_114064

def initial_employees : ℕ := 1000
def layoff_rate : ℝ := 0.1
def total_layoffs : ℕ := 271

def remaining_employees (n : ℕ) : ℝ :=
  initial_employees * (1 - layoff_rate) ^ n

def layoffs_after_rounds (n : ℕ) : ℝ :=
  initial_employees - remaining_employees n

theorem min_layoff_rounds_is_four :
  (∀ k < 4, layoffs_after_rounds k < total_layoffs) ∧
  layoffs_after_rounds 4 ≥ total_layoffs := by sorry

end min_layoff_rounds_is_four_l1140_114064


namespace isosceles_triangle_base_l1140_114077

/-- Proves that the base of an isosceles triangle is 10, given specific conditions about its perimeter and relationship to an equilateral triangle. -/
theorem isosceles_triangle_base : 
  ∀ (s b : ℝ),
  -- Equilateral triangle perimeter condition
  3 * s = 45 →
  -- Isosceles triangle perimeter condition
  2 * s + b = 40 →
  -- Base of isosceles triangle is 10
  b = 10 := by
sorry

end isosceles_triangle_base_l1140_114077


namespace function_is_identity_l1140_114058

def is_positive (n : ℕ) : Prop := n > 0

def satisfies_functional_equation (f : ℕ → ℕ) : Prop :=
  ∀ m n, is_positive m → is_positive n → f (f m + f n) = m + n

theorem function_is_identity 
  (f : ℕ → ℕ) 
  (h : satisfies_functional_equation f) :
  ∀ x, is_positive x → f x = x :=
sorry

end function_is_identity_l1140_114058


namespace vector_sum_equality_l1140_114063

theorem vector_sum_equality (a b : ℝ × ℝ) :
  a = (2, 1) →
  b = (-3, 4) →
  (3 : ℝ) • a + (4 : ℝ) • b = (-6, 19) := by
  sorry

end vector_sum_equality_l1140_114063


namespace rhombus_diagonal_l1140_114003

/-- Given a rhombus with area 64/5 square centimeters and one diagonal 64/9 centimeters,
    prove that the other diagonal is 18/5 centimeters. -/
theorem rhombus_diagonal (area : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ) : 
  area = 64/5 → 
  diagonal1 = 64/9 → 
  area = (diagonal1 * diagonal2) / 2 → 
  diagonal2 = 18/5 := by
  sorry

end rhombus_diagonal_l1140_114003


namespace factorization_problem_multiplication_problem_l1140_114066

variable (x y : ℝ)

theorem factorization_problem : x^5 - x^3 * y^2 = x^3 * (x - y) * (x + y) := by sorry

theorem multiplication_problem : (-2 * x^3 * y^2) * (3 * x^2 * y) = -6 * x^5 * y^3 := by sorry

end factorization_problem_multiplication_problem_l1140_114066


namespace distinct_integer_parts_l1140_114045

theorem distinct_integer_parts (N : ℕ) (h : N > 1) :
  {α : ℝ | (∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ N → ⌊i * α⌋ ≠ ⌊j * α⌋) ∧
           (∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ N → ⌊i / α⌋ ≠ ⌊j / α⌋)} =
  {α : ℝ | (N - 1) / N ≤ α ∧ α ≤ N / (N - 1)} :=
sorry

end distinct_integer_parts_l1140_114045


namespace factorial_ratio_simplification_l1140_114037

theorem factorial_ratio_simplification :
  (11 * Nat.factorial 10 * Nat.factorial 7 * Nat.factorial 3) / 
  (Nat.factorial 10 * Nat.factorial 8) = 11 / 56 := by
  sorry

end factorial_ratio_simplification_l1140_114037


namespace opposite_numbers_not_on_hyperbola_l1140_114018

theorem opposite_numbers_not_on_hyperbola (x y : ℝ) : 
  y = 1 / x → x ≠ -y := by
  sorry

end opposite_numbers_not_on_hyperbola_l1140_114018


namespace chess_team_arrangements_l1140_114071

/-- Represents the number of boys on the chess team -/
def num_boys : ℕ := 3

/-- Represents the number of girls on the chess team -/
def num_girls : ℕ := 2

/-- Calculates the number of possible arrangements for the chess team photo -/
def num_arrangements : ℕ := num_girls.factorial * num_boys.factorial

/-- Theorem stating that the number of possible arrangements is 12 -/
theorem chess_team_arrangements :
  num_arrangements = 12 := by sorry

end chess_team_arrangements_l1140_114071


namespace fraction_equals_zero_l1140_114033

theorem fraction_equals_zero (x : ℝ) : (x - 5) / (5 * x - 15) = 0 ↔ x = 5 :=
by sorry

end fraction_equals_zero_l1140_114033


namespace valentines_day_theorem_l1140_114020

/-- The number of valentines given on Valentine's Day -/
def valentines_given (male_students female_students : ℕ) : ℕ :=
  male_students * female_students

/-- The total number of students -/
def total_students (male_students female_students : ℕ) : ℕ :=
  male_students + female_students

/-- Theorem stating the number of valentines given -/
theorem valentines_day_theorem (male_students female_students : ℕ) :
  valentines_given male_students female_students = 
  total_students male_students female_students + 22 →
  valentines_given male_students female_students = 48 :=
by
  sorry

end valentines_day_theorem_l1140_114020


namespace gcd_840_1764_l1140_114072

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by sorry

end gcd_840_1764_l1140_114072


namespace rhombus_area_in_square_l1140_114092

/-- The area of a rhombus formed by connecting the midpoints of a square -/
theorem rhombus_area_in_square (side_length : ℝ) (h : side_length = 10) :
  let square_diagonal := side_length * Real.sqrt 2
  let rhombus_side := square_diagonal / 2
  let rhombus_area := (rhombus_side * rhombus_side) / 2
  rhombus_area = 25 := by sorry

end rhombus_area_in_square_l1140_114092


namespace total_cost_is_3200_cents_l1140_114010

/-- Represents the number of shirt boxes that can be wrapped with one roll of paper -/
def shirt_boxes_per_roll : ℕ := 5

/-- Represents the number of XL boxes that can be wrapped with one roll of paper -/
def xl_boxes_per_roll : ℕ := 3

/-- Represents the number of shirt boxes Harold needs to wrap -/
def total_shirt_boxes : ℕ := 20

/-- Represents the number of XL boxes Harold needs to wrap -/
def total_xl_boxes : ℕ := 12

/-- Represents the cost of one roll of wrapping paper in cents -/
def cost_per_roll : ℕ := 400

/-- Theorem stating that the total cost for Harold to wrap all boxes is $32.00 -/
theorem total_cost_is_3200_cents : 
  (((total_shirt_boxes + shirt_boxes_per_roll - 1) / shirt_boxes_per_roll) + 
   ((total_xl_boxes + xl_boxes_per_roll - 1) / xl_boxes_per_roll)) * 
  cost_per_roll = 3200 := by
  sorry

end total_cost_is_3200_cents_l1140_114010


namespace abs_one_fifth_set_l1140_114039

theorem abs_one_fifth_set : 
  {x : ℝ | |x| = (1 : ℝ) / 5} = {-(1 : ℝ) / 5, (1 : ℝ) / 5} := by
  sorry

end abs_one_fifth_set_l1140_114039


namespace arithmetic_sequence_sum_l1140_114035

/-- The sum of an arithmetic sequence with first term 1, common difference 2, and 20 terms -/
def arithmetic_sum : ℕ → ℕ
  | 0 => 0
  | n + 1 => (n + 1) + arithmetic_sum n

/-- The first term of the sequence -/
def a₁ : ℕ := 1

/-- The common difference of the sequence -/
def d : ℕ := 2

/-- The number of terms in the sequence -/
def n : ℕ := 20

/-- The n-th term of the sequence -/
def aₙ (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_sum :
  arithmetic_sum n = n * (a₁ + aₙ n) / 2 ∧ arithmetic_sum n = 400 :=
sorry

end arithmetic_sequence_sum_l1140_114035


namespace solution_x_l1140_114080

theorem solution_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 5) (h2 : y + 1 / x = 7 / 4) :
  x = 4 / 7 ∨ x = 5 := by
sorry

end solution_x_l1140_114080


namespace car_mileage_l1140_114052

theorem car_mileage (highway_miles_per_tank : ℕ) (city_mpg : ℕ) (mpg_difference : ℕ) :
  highway_miles_per_tank = 462 →
  city_mpg = 24 →
  mpg_difference = 9 →
  (highway_miles_per_tank / (city_mpg + mpg_difference)) * city_mpg = 336 :=
by sorry

end car_mileage_l1140_114052


namespace line_through_points_l1140_114027

/-- The general form equation of a line passing through two points -/
def general_form_equation (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ (a b c : ℝ), a * x + b * y + c = 0 ∧
    a * x₁ + b * y₁ + c = 0 ∧
    a * x₂ + b * y₂ + c = 0 ∧
    (a ≠ 0 ∨ b ≠ 0)}

/-- Theorem: The general form equation of the line passing through (1, 1) and (-2, 4) is x + y - 2 = 0 -/
theorem line_through_points : 
  general_form_equation 1 1 (-2) 4 = {(x, y) | x + y - 2 = 0} := by
  sorry

end line_through_points_l1140_114027


namespace julia_tag_kids_l1140_114017

/-- The number of kids Julia played tag with on Monday -/
def monday_kids : ℕ := 7

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_kids : ℕ := 13

/-- The total number of kids Julia played tag with -/
def total_kids : ℕ := monday_kids + tuesday_kids

theorem julia_tag_kids : total_kids = 20 := by
  sorry

end julia_tag_kids_l1140_114017


namespace symmetric_circle_equation_l1140_114065

/-- Given a circle with equation (x-3)^2+(y+4)^2=2, 
    prove that its symmetric circle with respect to y=0 
    has the equation (x-3)^2+(y-4)^2=2 -/
theorem symmetric_circle_equation : 
  ∀ (x y : ℝ), 
  (∃ (x₀ y₀ : ℝ), (x - x₀)^2 + (y - y₀)^2 = 2 ∧ x₀ = 3 ∧ y₀ = -4) →
  (∃ (x₁ y₁ : ℝ), (x - x₁)^2 + (y - y₁)^2 = 2 ∧ x₁ = 3 ∧ y₁ = 4) :=
by sorry

end symmetric_circle_equation_l1140_114065


namespace quadratic_equation_roots_quadratic_equation_other_root_l1140_114005

/-- The quadratic equation x^2 - 2x + m - 1 = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 - 2*x + m - 1 = 0

theorem quadratic_equation_roots (m : ℝ) :
  (∃ x : ℝ, ∀ y : ℝ, quadratic_equation y m ↔ y = x) →
  m = 2 :=
sorry

theorem quadratic_equation_other_root (m : ℝ) :
  (quadratic_equation 5 m) →
  (∃ x : ℝ, x ≠ 5 ∧ quadratic_equation x m ∧ x = -3) :=
sorry

end quadratic_equation_roots_quadratic_equation_other_root_l1140_114005


namespace common_tangent_bisection_l1140_114008

-- Define the basic geometric objects
variable (Circle₁ Circle₂ : Type) [MetricSpace Circle₁] [MetricSpace Circle₂]
variable (A B : ℝ × ℝ)  -- Intersection points of the circles
variable (M N : ℝ × ℝ)  -- Points of tangency on the common tangent

-- Define the property of being a point on a circle
def OnCircle (p : ℝ × ℝ) (circle : Type) [MetricSpace circle] : Prop := sorry

-- Define the property of being a tangent line to a circle
def IsTangent (p q : ℝ × ℝ) (circle : Type) [MetricSpace circle] : Prop := sorry

-- Define the property of a line bisecting another line segment
def Bisects (p q r s : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem common_tangent_bisection 
  (hA₁ : OnCircle A Circle₁) (hA₂ : OnCircle A Circle₂)
  (hB₁ : OnCircle B Circle₁) (hB₂ : OnCircle B Circle₂)
  (hM₁ : OnCircle M Circle₁) (hN₂ : OnCircle N Circle₂)
  (hMN₁ : IsTangent M N Circle₁) (hMN₂ : IsTangent M N Circle₂) :
  Bisects A B M N := by sorry

end common_tangent_bisection_l1140_114008


namespace matching_probability_abe_bob_l1140_114022

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ := jb.green + jb.red + jb.yellow

/-- Represents the jelly beans held by Abe -/
def abe : JellyBeans := { green := 1, red := 1, yellow := 0 }

/-- Represents the jelly beans held by Bob -/
def bob : JellyBeans := { green := 1, red := 2, yellow := 1 }

/-- Calculates the probability of two people showing the same color jelly bean -/
def matchingProbability (person1 person2 : JellyBeans) : ℚ :=
  let greenProb := (person1.green : ℚ) / person1.total * (person2.green : ℚ) / person2.total
  let redProb := (person1.red : ℚ) / person1.total * (person2.red : ℚ) / person2.total
  greenProb + redProb

theorem matching_probability_abe_bob :
  matchingProbability abe bob = 3 / 8 := by
  sorry

end matching_probability_abe_bob_l1140_114022


namespace part_1_part_2_l1140_114062

-- Define the inequality
def inequality (a x : ℝ) : Prop := a * x^2 - 2*a + 1 < (1 - a) * x

-- Define the solution set for part (1)
def solution_set_1 (x : ℝ) : Prop := x < -4 ∨ x > 1

-- Define the condition for part (2)
def condition_2 (a : ℝ) : Prop := a > 0

-- Define the property of having exactly 7 prime elements in the solution set
def has_seven_primes (a : ℝ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 p6 p7 : ℕ),
    Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ Prime p4 ∧ Prime p5 ∧ Prime p6 ∧ Prime p7 ∧
    p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ p4 < p5 ∧ p5 < p6 ∧ p6 < p7 ∧
    (∀ x : ℝ, inequality a x ↔ (x < p1 ∨ x > p7))

-- Theorem for part (1)
theorem part_1 : 
  (∀ x : ℝ, inequality a x ↔ solution_set_1 x) → a = -1/2 :=
sorry

-- Theorem for part (2)
theorem part_2 :
  condition_2 a → has_seven_primes a → 1/21 ≤ a ∧ a < 1/19 :=
sorry

end part_1_part_2_l1140_114062


namespace square_perimeter_32cm_l1140_114096

theorem square_perimeter_32cm (side_length : ℝ) (h : side_length = 8) : 
  4 * side_length = 32 := by
  sorry

end square_perimeter_32cm_l1140_114096


namespace inscribing_square_area_l1140_114021

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x - 6*y + 24 = 0

-- Define the square that inscribes the circle
structure InscribingSquare :=
  (side_length : ℝ)
  (parallel_to_x_axis : Prop)
  (inscribes_circle : Prop)

-- Theorem statement
theorem inscribing_square_area
  (square : InscribingSquare)
  (h_circle : ∀ x y, circle_equation x y ↔ (x - 4)^2 + (y - 3)^2 = 1) :
  square.side_length^2 = 4 :=
sorry

end inscribing_square_area_l1140_114021


namespace sum_of_divisors_450_has_three_prime_factors_l1140_114069

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_450_has_three_prime_factors : 
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by sorry

end sum_of_divisors_450_has_three_prime_factors_l1140_114069


namespace equilateral_perimeter_is_60_l1140_114043

/-- An equilateral triangle with a side shared with an isosceles triangle -/
structure TrianglePair where
  equilateral_side : ℝ
  isosceles_base : ℝ
  isosceles_perimeter : ℝ
  equilateral_side_positive : 0 < equilateral_side
  isosceles_base_positive : 0 < isosceles_base
  isosceles_perimeter_positive : 0 < isosceles_perimeter

/-- The perimeter of the equilateral triangle in the TrianglePair -/
def equilateral_perimeter (tp : TrianglePair) : ℝ := 3 * tp.equilateral_side

/-- Theorem: The perimeter of the equilateral triangle is 60 -/
theorem equilateral_perimeter_is_60 (tp : TrianglePair)
  (h1 : tp.isosceles_base = 15)
  (h2 : tp.isosceles_perimeter = 55) :
  equilateral_perimeter tp = 60 := by
  sorry

#check equilateral_perimeter_is_60

end equilateral_perimeter_is_60_l1140_114043


namespace parallel_line_to_plane_parallel_lines_in_intersecting_planes_l1140_114029

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (intersects : Plane → Plane → Line → Prop)

-- Theorem 1
theorem parallel_line_to_plane 
  (α β : Plane) (m : Line) 
  (h1 : parallel_plane α β) 
  (h2 : contains α m) : 
  parallel_plane_line β m :=
sorry

-- Theorem 2
theorem parallel_lines_in_intersecting_planes 
  (α β : Plane) (m n : Line)
  (h1 : parallel_plane_line β m)
  (h2 : contains α m)
  (h3 : intersects α β n) :
  parallel m n :=
sorry

end parallel_line_to_plane_parallel_lines_in_intersecting_planes_l1140_114029


namespace count_pairs_eq_50_l1140_114015

/-- The number of pairs of positive integers (m,n) satisfying m^2 + mn < 30 -/
def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + p.1 * p.2 < 30) (Finset.product (Finset.range 30) (Finset.range 30))).card

/-- Theorem stating that the count of pairs satisfying the condition is 50 -/
theorem count_pairs_eq_50 : count_pairs = 50 := by
  sorry

end count_pairs_eq_50_l1140_114015


namespace mrs_hilt_candy_distribution_l1140_114049

/-- Mrs. Hilt's candy distribution problem -/
theorem mrs_hilt_candy_distribution 
  (chocolate_per_student : ℕ) 
  (chocolate_students : ℕ) 
  (hard_candy_per_student : ℕ) 
  (hard_candy_students : ℕ) 
  (gummy_per_student : ℕ) 
  (gummy_students : ℕ) 
  (h1 : chocolate_per_student = 2) 
  (h2 : chocolate_students = 3) 
  (h3 : hard_candy_per_student = 4) 
  (h4 : hard_candy_students = 2) 
  (h5 : gummy_per_student = 6) 
  (h6 : gummy_students = 4) : 
  chocolate_per_student * chocolate_students + 
  hard_candy_per_student * hard_candy_students + 
  gummy_per_student * gummy_students = 38 := by
sorry

end mrs_hilt_candy_distribution_l1140_114049


namespace equation_solutions_l1140_114001

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (2 * x₁^2 = 5 * x₁ ∧ x₁ = 0) ∧ (2 * x₂^2 = 5 * x₂ ∧ x₂ = 5/2)) ∧
  (∃ y₁ y₂ : ℝ, (y₁^2 + 3*y₁ = 3 ∧ y₁ = (-3 + Real.sqrt 21) / 2) ∧
               (y₂^2 + 3*y₂ = 3 ∧ y₂ = (-3 - Real.sqrt 21) / 2)) :=
by sorry

end equation_solutions_l1140_114001


namespace max_y_coordinate_polar_curve_l1140_114091

theorem max_y_coordinate_polar_curve (θ : Real) :
  let r := λ θ : Real => Real.cos (2 * θ)
  let x := λ θ : Real => (r θ) * Real.cos θ
  let y := λ θ : Real => (r θ) * Real.sin θ
  (∀ θ', |y θ'| ≤ |y θ|) → y θ = Real.sqrt (30 * Real.sqrt 6) / 9 :=
by sorry

end max_y_coordinate_polar_curve_l1140_114091


namespace probability_at_least_two_same_l1140_114048

def num_dice : ℕ := 8
def num_sides : ℕ := 8

theorem probability_at_least_two_same :
  let total_outcomes := num_sides ^ num_dice
  let all_different := Nat.factorial num_sides
  (1 - (all_different : ℚ) / total_outcomes) = 2043 / 2048 := by
  sorry

end probability_at_least_two_same_l1140_114048


namespace skittles_division_l1140_114013

theorem skittles_division (total_skittles : Nat) (num_groups : Nat) (group_size : Nat) :
  total_skittles = 5929 →
  num_groups = 77 →
  total_skittles = num_groups * group_size →
  group_size = 77 := by
sorry

end skittles_division_l1140_114013


namespace hcf_from_lcm_and_product_l1140_114009

theorem hcf_from_lcm_and_product (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 750)
  (h_product : a * b = 18750) : 
  Nat.gcd a b = 25 := by
  sorry

end hcf_from_lcm_and_product_l1140_114009


namespace tuesday_temperature_l1140_114038

theorem tuesday_temperature
  (temp_tue wed thu fri : ℝ)
  (h1 : (temp_tue + wed + thu) / 3 = 45)
  (h2 : (wed + thu + fri) / 3 = 50)
  (h3 : fri = 53) :
  temp_tue = 38 := by
sorry

end tuesday_temperature_l1140_114038


namespace chess_tournament_theorem_l1140_114011

/-- Represents a participant in the chess tournament -/
structure Participant :=
  (id : Nat)

/-- Represents the results of the chess tournament -/
structure TournamentResult :=
  (participants : Finset Participant)
  (white_wins : Participant → Nat)
  (black_wins : Participant → Nat)

/-- Defines the "no weaker than" relation between two participants -/
def no_weaker_than (result : TournamentResult) (a b : Participant) : Prop :=
  result.white_wins a ≥ result.white_wins b ∧ result.black_wins a ≥ result.black_wins b

theorem chess_tournament_theorem :
  ∀ (result : TournamentResult),
    result.participants.card = 20 →
    (∀ p q : Participant, p ∈ result.participants → q ∈ result.participants → p ≠ q →
      result.white_wins p + result.white_wins q = 1 ∧
      result.black_wins p + result.black_wins q = 1) →
    ∃ a b : Participant, a ∈ result.participants ∧ b ∈ result.participants ∧ a ≠ b ∧
      no_weaker_than result a b := by
  sorry


end chess_tournament_theorem_l1140_114011


namespace complement_A_in_U_intersection_A_B_l1140_114050

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 4}

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | -3 < x ∧ x ≤ 1}

-- Theorem for the complement of A in U
theorem complement_A_in_U : 
  (U \ A) = {x | x ≤ -2 ∨ (3 ≤ x ∧ x ≤ 4)} := by sorry

-- Theorem for the intersection of A and B
theorem intersection_A_B : 
  (A ∩ B) = {x | -2 < x ∧ x ≤ 1} := by sorry

end complement_A_in_U_intersection_A_B_l1140_114050


namespace hyperbola_asymptotes_l1140_114016

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

-- Theorem stating that the given asymptote equation is correct for the hyperbola
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola x y → (∃ x' y' : ℝ, x' ≠ x ∧ y' ≠ y ∧ hyperbola x' y' ∧ asymptote x' y') :=
sorry

end hyperbola_asymptotes_l1140_114016


namespace science_club_teams_l1140_114032

theorem science_club_teams (girls : ℕ) (boys : ℕ) :
  girls = 4 → boys = 7 → (girls.choose 3) * (boys.choose 2) = 84 := by
  sorry

end science_club_teams_l1140_114032


namespace sum_of_exponents_eight_l1140_114082

/-- Sum of geometric series from 1 to x^n -/
def geometricSum (x n : ℕ) : ℕ := (x^(n+1) - 1) / (x - 1)

/-- Sum of divisors of 2^i * 3^j * 5^k -/
def sumDivisors (i j k : ℕ) : ℕ :=
  (geometricSum 2 i) * (geometricSum 3 j) * (geometricSum 5 k)

/-- Theorem: If the sum of divisors of 2^i * 3^j * 5^k is 1800, then i + j + k = 8 -/
theorem sum_of_exponents_eight (i j k : ℕ) :
  sumDivisors i j k = 1800 → i + j + k = 8 := by
  sorry

end sum_of_exponents_eight_l1140_114082


namespace square_value_l1140_114097

theorem square_value : ∃ (square : ℚ), 
  16.2 * ((4 + 1/7 - square * 700) / (1 + 2/7)) = 8.1 ∧ square = 0.005 := by sorry

end square_value_l1140_114097


namespace catering_cost_comparison_l1140_114053

def cost_caterer1 (x : ℕ) : ℚ := 150 + 18 * x
def cost_caterer2 (x : ℕ) : ℚ := 250 + 15 * x

theorem catering_cost_comparison :
  (∀ x : ℕ, x < 34 → cost_caterer1 x ≤ cost_caterer2 x) ∧
  (∀ x : ℕ, x ≥ 34 → cost_caterer1 x > cost_caterer2 x) :=
by sorry

end catering_cost_comparison_l1140_114053


namespace joker_selection_ways_l1140_114075

def total_cards : ℕ := 54
def jokers : ℕ := 2
def standard_cards : ℕ := 52

def ways_to_pick_joker_first (cards : ℕ) (jokers : ℕ) : ℕ :=
  jokers * (cards - 1)

def ways_to_pick_joker_second (cards : ℕ) (standard_cards : ℕ) (jokers : ℕ) : ℕ :=
  standard_cards * jokers

theorem joker_selection_ways :
  ways_to_pick_joker_first total_cards jokers +
  ways_to_pick_joker_second total_cards standard_cards jokers = 210 :=
by sorry

end joker_selection_ways_l1140_114075


namespace product_of_three_numbers_l1140_114019

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_eq_one : a + b + c = 1)
  (sum_sq_eq_one : a^2 + b^2 + c^2 = 1)
  (sum_cube_eq_one : a^3 + b^3 + c^3 = 1) : 
  a * b * c = 0 := by
  sorry

end product_of_three_numbers_l1140_114019


namespace intersection_and_union_of_sets_l1140_114014

def A (p : ℝ) : Set ℝ := {x | 2 * x^2 + 3 * p * x + 2 = 0}
def B (q : ℝ) : Set ℝ := {x | 2 * x^2 + x + q = 0}

theorem intersection_and_union_of_sets (p q : ℝ) :
  A p ∩ B q = {1/2} →
  p = -5/3 ∧ q = -1 ∧ A p ∪ B q = {-1, 1/2, 2} := by
  sorry

end intersection_and_union_of_sets_l1140_114014


namespace min_equal_number_example_min_equal_number_is_minimum_l1140_114090

/-- Given three initial numbers on a blackboard, this function represents
    the minimum number to which all three can be made equal by repeatedly
    selecting two numbers and adding 1 to each. -/
def min_equal_number (a b c : ℕ) : ℕ :=
  (a + b + c + 2 * ((a + b + c) % 3)) / 3

/-- Theorem stating that 747 is the minimum number to which 20, 201, and 2016
    can be made equal using the described operation. -/
theorem min_equal_number_example : min_equal_number 20 201 2016 = 747 := by
  sorry

/-- Theorem stating that the result of min_equal_number is indeed the minimum
    possible number to which the initial numbers can be made equal. -/
theorem min_equal_number_is_minimum (a b c : ℕ) :
  ∀ n : ℕ, (∃ k : ℕ, a + k ≤ n ∧ b + k ≤ n ∧ c + k ≤ n) →
  min_equal_number a b c ≤ n := by
  sorry

end min_equal_number_example_min_equal_number_is_minimum_l1140_114090


namespace product_94_106_l1140_114095

theorem product_94_106 : 94 * 106 = 9964 := by
  sorry

end product_94_106_l1140_114095


namespace parabola_equation_l1140_114073

theorem parabola_equation (x : ℝ) :
  let f := fun x => -3 * x^2 + 12 * x - 8
  let vertex := (2, 4)
  let point := (1, 1)
  (∀ h, f (vertex.1 + h) = f (vertex.1 - h)) ∧  -- Vertical axis of symmetry
  (f vertex.1 = vertex.2) ∧                     -- Passes through vertex
  (f point.1 = point.2) ∧                       -- Contains the given point
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) -- In quadratic form
  :=
by sorry

end parabola_equation_l1140_114073


namespace complex_number_quadrant_l1140_114030

theorem complex_number_quadrant (z : ℂ) : z - Complex.I = Complex.abs (1 + 2 * Complex.I) → 
  z.re > 0 ∧ z.im > 0 := by
  sorry

end complex_number_quadrant_l1140_114030


namespace trajectory_eq_sufficient_not_necessary_l1140_114026

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The distance from a point to the x-axis -/
def distToXAxis (p : Point2D) : ℝ := |p.y|

/-- The distance from a point to the y-axis -/
def distToYAxis (p : Point2D) : ℝ := |p.x|

/-- A point has equal distance to both axes -/
def equalDistToAxes (p : Point2D) : Prop :=
  distToXAxis p = distToYAxis p

/-- The trajectory equation y = |x| -/
def trajectoryEq (p : Point2D) : Prop :=
  p.y = |p.x|

/-- Theorem: y = |x| is a sufficient but not necessary condition for equal distance to both axes -/
theorem trajectory_eq_sufficient_not_necessary :
  (∀ p : Point2D, trajectoryEq p → equalDistToAxes p) ∧
  (∃ p : Point2D, equalDistToAxes p ∧ ¬trajectoryEq p) :=
sorry

end trajectory_eq_sufficient_not_necessary_l1140_114026


namespace home_electronics_budget_allocation_l1140_114054

theorem home_electronics_budget_allocation 
  (total_budget : ℝ)
  (microphotonics : ℝ)
  (food_additives : ℝ)
  (genetically_modified_microorganisms : ℝ)
  (industrial_lubricants : ℝ)
  (basic_astrophysics_degrees : ℝ)
  (h1 : total_budget = 100)
  (h2 : microphotonics = 14)
  (h3 : food_additives = 20)
  (h4 : genetically_modified_microorganisms = 29)
  (h5 : industrial_lubricants = 8)
  (h6 : basic_astrophysics_degrees = 18)
  (h7 : (basic_astrophysics_degrees / 360) * 100 + microphotonics + food_additives + genetically_modified_microorganisms + industrial_lubricants + home_electronics = total_budget) :
  home_electronics = 24 := by
  sorry

end home_electronics_budget_allocation_l1140_114054


namespace work_completion_time_l1140_114042

/-- The number of days it takes for a and b together to complete the work -/
def combined_time : ℝ := 6

/-- The number of days it takes for b alone to complete the work -/
def b_time : ℝ := 11.142857142857144

/-- The number of days it takes for a alone to complete the work -/
def a_time : ℝ := 13

/-- The theorem stating that given the combined time and b's time, a's time is 13 days -/
theorem work_completion_time : 
  (1 / combined_time) = (1 / a_time) + (1 / b_time) :=
sorry

end work_completion_time_l1140_114042


namespace inequality_solution_set_l1140_114047

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - 2*x - 3)*(x^2 + 1) < 0 ↔ -1 < x ∧ x < 3 :=
by sorry

end inequality_solution_set_l1140_114047


namespace subtract_negatives_l1140_114025

theorem subtract_negatives : (-1) - (-4) = 3 := by
  sorry

end subtract_negatives_l1140_114025


namespace balloon_distribution_l1140_114028

theorem balloon_distribution (total_balloons : ℕ) (friends : ℕ) 
  (h1 : total_balloons = 235) (h2 : friends = 10) : 
  total_balloons % friends = 5 := by
  sorry

end balloon_distribution_l1140_114028


namespace equation_solution_l1140_114055

theorem equation_solution :
  let a : ℝ := 9
  let b : ℝ := 4
  let c : ℝ := 3
  ∃ x : ℝ, (x^2 + c + b^2 = (a - x)^2 + c) ∧ (x = 65 / 18) := by
  sorry

end equation_solution_l1140_114055


namespace no_real_solutions_l1140_114060

theorem no_real_solutions : ¬ ∃ x : ℝ, Real.sqrt ((x^2 - 2*x + 1) + 1) = -x := by
  sorry

end no_real_solutions_l1140_114060


namespace circle_fixed_point_l1140_114098

/-- A circle with center (a, b) on the parabola y^2 = 4x and tangent to x = -1 passes through (1, 0) -/
theorem circle_fixed_point (a b : ℝ) : 
  b^2 = 4*a →  -- Center (a, b) lies on the parabola y^2 = 4x
  (a + 1)^2 = (1 - a)^2 + b^2 -- Circle is tangent to x = -1
  → (1 - a)^2 + 0^2 = (a + 1)^2 -- Point (1, 0) lies on the circle
  := by sorry

end circle_fixed_point_l1140_114098


namespace lending_interest_rate_l1140_114056

/-- Calculates the simple interest --/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem lending_interest_rate 
  (principal : ℝ)
  (b_to_c_rate : ℝ)
  (time : ℝ)
  (b_gain : ℝ)
  (h1 : principal = 3200)
  (h2 : b_to_c_rate = 0.145)
  (h3 : time = 5)
  (h4 : b_gain = 400)
  : ∃ (a_to_b_rate : ℝ), 
    simpleInterest principal a_to_b_rate time = 
    simpleInterest principal b_to_c_rate time - b_gain ∧ 
    a_to_b_rate = 0.12 := by
  sorry

end lending_interest_rate_l1140_114056


namespace blood_drops_per_liter_l1140_114084

/-- The number of drops of blood sucked by one mosquito in a single feeding. -/
def drops_per_mosquito : ℕ := 20

/-- The number of liters of blood loss that is fatal. -/
def fatal_blood_loss : ℕ := 3

/-- The number of mosquitoes that would cause a fatal blood loss if they all fed. -/
def fatal_mosquito_count : ℕ := 750

/-- The number of drops of blood in one liter. -/
def drops_per_liter : ℕ := 5000

theorem blood_drops_per_liter :
  drops_per_liter = (drops_per_mosquito * fatal_mosquito_count) / fatal_blood_loss := by
  sorry

end blood_drops_per_liter_l1140_114084


namespace special_square_smallest_area_l1140_114086

/-- A square with specific properties -/
structure SpecialSquare where
  /-- Two vertices lie on the line y = 2x + 3 -/
  vertices_on_line : ℝ → ℝ → Prop
  /-- Two vertices lie on the parabola y = -x^2 + 4x + 5 -/
  vertices_on_parabola : ℝ → ℝ → Prop
  /-- One vertex lies on the origin (0, 0) -/
  vertex_on_origin : Prop

/-- The smallest possible area of a SpecialSquare -/
def smallest_area (s : SpecialSquare) : ℝ := 580

/-- Theorem stating the smallest possible area of a SpecialSquare -/
theorem special_square_smallest_area (s : SpecialSquare) :
  smallest_area s = 580 := by sorry

end special_square_smallest_area_l1140_114086


namespace bowling_ball_weight_l1140_114057

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (9 * bowling_ball_weight = 6 * canoe_weight) →
    (4 * canoe_weight = 120) →
    bowling_ball_weight = 20 :=
by
  sorry

end bowling_ball_weight_l1140_114057


namespace sqrt_s6_plus_s3_l1140_114046

theorem sqrt_s6_plus_s3 (s : ℝ) : Real.sqrt (s^6 + s^3) = |s| * Real.sqrt (s * (s^3 + 1)) := by
  sorry

end sqrt_s6_plus_s3_l1140_114046


namespace solve_for_x_l1140_114070

theorem solve_for_x (x y : ℝ) (h1 : x - y = 15) (h2 : x + y = 9) : x = 12 := by
  sorry

end solve_for_x_l1140_114070


namespace range_of_a_range_is_nonnegative_reals_l1140_114067

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 = a}

-- State the theorem
theorem range_of_a (h : ∃ x, x ∈ A a) : a ≥ 0 := by
  sorry

-- Prove that this covers the entire range [0, +∞)
theorem range_is_nonnegative_reals : 
  ∀ a ≥ 0, ∃ x, x ∈ A a := by
  sorry

end range_of_a_range_is_nonnegative_reals_l1140_114067


namespace percentage_increase_l1140_114004

theorem percentage_increase (original : ℝ) (difference : ℝ) (increase : ℝ) : 
  original = 80 →
  original + (increase / 100) * original - (original - 25 / 100 * original) = difference →
  difference = 30 →
  increase = 12.5 := by
sorry

end percentage_increase_l1140_114004


namespace quadratic_inequality_range_l1140_114076

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + m > 0) → m ∈ Set.Ioo 0 4 := by
  sorry

end quadratic_inequality_range_l1140_114076


namespace quadratic_inequality_solution_set_l1140_114031

theorem quadratic_inequality_solution_set 
  (α β a b c : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < β) 
  (h3 : ∀ x, a * x^2 + b * x + c > 0 ↔ α < x ∧ x < β) :
  ∀ x, (a + c - b) * x^2 + (b - 2*a) * x + a > 0 ↔ 1 / (1 + β) < x ∧ x < 1 / (1 + α) :=
by sorry

end quadratic_inequality_solution_set_l1140_114031


namespace football_team_size_l1140_114002

/-- Represents the number of players on a football team -/
def total_players : ℕ := 70

/-- Represents the number of throwers on the team -/
def throwers : ℕ := 52

/-- Represents the total number of right-handed players -/
def right_handed_players : ℕ := 64

/-- States that one third of non-throwers are left-handed -/
axiom one_third_non_throwers_left_handed :
  (total_players - throwers) / 3 = (total_players - throwers - (right_handed_players - throwers))

/-- All throwers are right-handed -/
axiom all_throwers_right_handed :
  throwers ≤ right_handed_players

/-- Theorem stating that the total number of players is 70 -/
theorem football_team_size :
  total_players = 70 :=
sorry

end football_team_size_l1140_114002


namespace catherine_friends_l1140_114079

/-- The number of friends Catherine gave pens and pencils to -/
def num_friends : ℕ := sorry

/-- The initial number of pens Catherine had -/
def initial_pens : ℕ := 60

/-- The number of pens given to each friend -/
def pens_per_friend : ℕ := 8

/-- The number of pencils given to each friend -/
def pencils_per_friend : ℕ := 6

/-- The total number of pens and pencils left after giving away -/
def items_left : ℕ := 22

theorem catherine_friends :
  (initial_pens * 2 - items_left) / (pens_per_friend + pencils_per_friend) = num_friends :=
sorry

end catherine_friends_l1140_114079


namespace simon_blueberries_l1140_114088

def blueberry_problem (own_bushes nearby_bushes pies_made blueberries_per_pie : ℕ) : Prop :=
  own_bushes + nearby_bushes = pies_made * blueberries_per_pie

theorem simon_blueberries : 
  ∃ (own_bushes : ℕ), 
    blueberry_problem own_bushes 200 3 100 ∧ 
    own_bushes = 100 := by sorry

end simon_blueberries_l1140_114088


namespace isabel_pop_albums_l1140_114012

/-- The number of country albums Isabel bought -/
def country_albums : ℕ := 6

/-- The number of songs per album -/
def songs_per_album : ℕ := 9

/-- The total number of songs Isabel bought -/
def total_songs : ℕ := 72

/-- The number of pop albums Isabel bought -/
def pop_albums : ℕ := (total_songs - country_albums * songs_per_album) / songs_per_album

theorem isabel_pop_albums : pop_albums = 2 := by
  sorry

end isabel_pop_albums_l1140_114012


namespace power_sum_value_l1140_114051

theorem power_sum_value (a : ℝ) (x y : ℝ) (h1 : a^x = 4) (h2 : a^y = 9) : a^(x+y) = 36 := by
  sorry

end power_sum_value_l1140_114051


namespace people_got_on_second_stop_is_two_l1140_114007

/-- The number of people who got on at the second stop of a bus journey -/
def people_got_on_second_stop : ℕ :=
  let initial_people : ℕ := 50
  let first_stop_off : ℕ := 15
  let second_stop_off : ℕ := 8
  let third_stop_off : ℕ := 4
  let third_stop_on : ℕ := 3
  let final_people : ℕ := 28
  initial_people - first_stop_off - second_stop_off + 
    (final_people - (initial_people - first_stop_off - second_stop_off - third_stop_off + third_stop_on))

theorem people_got_on_second_stop_is_two : 
  people_got_on_second_stop = 2 := by sorry

end people_got_on_second_stop_is_two_l1140_114007


namespace perpendicular_line_modulus_l1140_114093

/-- Given a line ax + y + 5 = 0 and points P and Q, prove the modulus of z = a + 4i -/
theorem perpendicular_line_modulus (a : ℝ) : 
  let P : ℝ × ℝ := (2, 4)
  let Q : ℝ × ℝ := (4, 3)
  let line (x y : ℝ) := a * x + y + 5 = 0
  let perpendicular (P Q : ℝ × ℝ) (line : ℝ → ℝ → Prop) := 
    (Q.2 - P.2) * a = -(Q.1 - P.1)  -- Perpendicular condition
  let z : ℂ := a + 4 * Complex.I
  perpendicular P Q line → Complex.abs z = 2 * Real.sqrt 5 :=
by sorry

end perpendicular_line_modulus_l1140_114093


namespace same_color_sock_pairs_l1140_114078

def white_socks : ℕ := 5
def brown_socks : ℕ := 5
def blue_socks : ℕ := 4
def black_socks : ℕ := 2

def total_socks : ℕ := white_socks + brown_socks + blue_socks + black_socks

def choose_pair (n : ℕ) : ℕ := n.choose 2

theorem same_color_sock_pairs :
  choose_pair white_socks + choose_pair brown_socks + choose_pair blue_socks + choose_pair black_socks = 27 := by
  sorry

end same_color_sock_pairs_l1140_114078


namespace train_speed_l1140_114024

/-- The speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) (h1 : train_length = 240) 
  (h2 : bridge_length = 130) (h3 : time = 26.64) : 
  ∃ (speed : ℝ), abs (speed - 50.004) < 0.001 ∧ 
  speed = (train_length + bridge_length) / time * 3.6 := by
  sorry

end train_speed_l1140_114024


namespace abc_inequality_l1140_114068

theorem abc_inequality (a b c : ℝ) 
  (ha : a = 2 * Real.sqrt 7)
  (hb : b = 3 * Real.sqrt 5)
  (hc : c = 5 * Real.sqrt 2) : 
  c > b ∧ b > a :=
sorry

end abc_inequality_l1140_114068


namespace seven_b_equals_ten_l1140_114034

theorem seven_b_equals_ten (a b : ℚ) (h1 : 5 * a + 2 * b = 0) (h2 : b - 2 = a) : 7 * b = 10 := by
  sorry

end seven_b_equals_ten_l1140_114034


namespace time_to_make_one_toy_l1140_114099

/-- Given that a worker makes 40 toys in 80 hours, prove that it takes 2 hours to make one toy. -/
theorem time_to_make_one_toy (total_hours : ℝ) (total_toys : ℝ) 
  (h1 : total_hours = 80) (h2 : total_toys = 40) : 
  total_hours / total_toys = 2 := by
sorry

end time_to_make_one_toy_l1140_114099


namespace percentage_equality_l1140_114023

theorem percentage_equality : (0.75 * 40 : ℝ) = (4/5 : ℝ) * 25 + 10 := by
  sorry

end percentage_equality_l1140_114023


namespace parabola_transformation_l1140_114061

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := 2 * (x - 3)^2 + 2

-- Define the transformation
def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 3) - 2

-- Define the resulting parabola
def result_parabola (x : ℝ) : ℝ := 2 * x^2

-- Theorem statement
theorem parabola_transformation :
  ∀ x : ℝ, transform original_parabola x = result_parabola x :=
by
  sorry

end parabola_transformation_l1140_114061


namespace last_four_digits_of_5_pow_2018_l1140_114087

def last_four_digits (n : ℕ) : ℕ := n % 10000

def cycle : List ℕ := [3125, 5625, 8125, 0625]

theorem last_four_digits_of_5_pow_2018 :
  last_four_digits (5^2018) = 5625 := by
  sorry

end last_four_digits_of_5_pow_2018_l1140_114087


namespace gcd_lcm_sum_120_3507_l1140_114000

theorem gcd_lcm_sum_120_3507 : 
  Nat.gcd 120 3507 + Nat.lcm 120 3507 = 140283 := by
  sorry

end gcd_lcm_sum_120_3507_l1140_114000


namespace complement_of_union_l1140_114081

def U : Set ℕ := {x | x ∈ Finset.range 6 \ {0}}
def A : Set ℕ := {2, 4}
def B : Set ℕ := {2, 3}

theorem complement_of_union : (U \ (A ∪ B)) = {1, 5} := by sorry

end complement_of_union_l1140_114081


namespace l_shaped_area_l1140_114041

/-- The area of an L-shaped region formed by subtracting two smaller squares
    from a larger square -/
theorem l_shaped_area (side_large : ℝ) (side_small1 : ℝ) (side_small2 : ℝ)
    (h1 : side_large = side_small1 + side_small2)
    (h2 : side_small1 = 4)
    (h3 : side_small2 = 2) :
    side_large^2 - (side_small1^2 + side_small2^2) = 16 := by
  sorry

end l_shaped_area_l1140_114041


namespace fifteen_ones_sum_multiple_of_30_l1140_114059

theorem fifteen_ones_sum_multiple_of_30 : 
  (Nat.choose 14 9 : ℕ) = 2002 := by sorry

end fifteen_ones_sum_multiple_of_30_l1140_114059


namespace discriminant_of_5x2_minus_9x_plus_1_l1140_114083

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 - 9x + 1 = 0 -/
def a : ℝ := 5
def b : ℝ := -9
def c : ℝ := 1

theorem discriminant_of_5x2_minus_9x_plus_1 :
  discriminant a b c = 61 := by sorry

end discriminant_of_5x2_minus_9x_plus_1_l1140_114083


namespace construct_remaining_vertices_l1140_114074

/-- Represents a point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → Point2D

/-- Represents a parallel projection of a regular hexagon onto a plane -/
structure ParallelProjection where
  original : RegularHexagon
  projected : Fin 6 → Point2D

/-- Given three consecutive projected vertices of a regular hexagon, 
    the remaining three vertices can be uniquely determined -/
theorem construct_remaining_vertices 
  (p : ParallelProjection) 
  (h : ∃ (i : Fin 6), 
       (p.projected i).x ≠ (p.projected (i + 1)).x ∨ 
       (p.projected i).y ≠ (p.projected (i + 1)).y) :
  ∃! (q : ParallelProjection), 
    (∃ (i : Fin 6), 
      q.projected i = p.projected i ∧ 
      q.projected (i + 1) = p.projected (i + 1) ∧ 
      q.projected (i + 2) = p.projected (i + 2)) ∧
    (∀ (j : Fin 6), q.projected j = p.projected j) :=
  sorry

end construct_remaining_vertices_l1140_114074


namespace ellipse_eccentricity_l1140_114036

/-- Prove that for an ellipse with the given properties, its eccentricity is 2/3 -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}
  let F := (Real.sqrt (a^2 - b^2), 0)
  let l := {(x, y) : ℝ × ℝ | y = Real.sqrt 3 * (x - F.1)}
  ∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ C ∧ A ∈ l ∧ B ∈ l ∧
    (A.2 < 0 ∧ B.2 > 0) ∧ 
    (-A.2 = 2 * B.2) →
  (Real.sqrt (a^2 - b^2)) / a = 2 / 3 :=
sorry

end ellipse_eccentricity_l1140_114036


namespace number_difference_l1140_114040

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 21780)
  (a_div_5 : a % 5 = 0)
  (b_relation : b * 10 + 5 = a) :
  a - b = 17825 := by
sorry

end number_difference_l1140_114040


namespace edward_candy_purchase_l1140_114094

def whack_a_mole_tickets : ℕ := 3
def skee_ball_tickets : ℕ := 5
def candy_cost : ℕ := 4

theorem edward_candy_purchase :
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost = 2 := by
  sorry

end edward_candy_purchase_l1140_114094


namespace candle_flower_groupings_l1140_114089

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

theorem candle_flower_groupings :
  (choose 4 2) * (choose 9 8) = 54 := by
  sorry

end candle_flower_groupings_l1140_114089
