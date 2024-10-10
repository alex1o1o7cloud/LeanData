import Mathlib

namespace man_birth_year_l2894_289458

-- Define the birth year function
def birthYear (x : ℕ) : ℕ := x^2 - x - 2

-- State the theorem
theorem man_birth_year :
  ∃ x : ℕ, 
    (birthYear x > 1900) ∧ 
    (birthYear x < 1950) ∧ 
    (birthYear x = 1890) := by
  sorry

end man_birth_year_l2894_289458


namespace iron_rod_weight_l2894_289488

/-- The weight of an iron rod given its length, cross-sectional area, and specific gravity -/
theorem iron_rod_weight 
  (length : Real) 
  (cross_sectional_area : Real) 
  (specific_gravity : Real) 
  (h1 : length = 1) -- 1 m
  (h2 : cross_sectional_area = 188) -- 188 cm²
  (h3 : specific_gravity = 7.8) -- 7.8 kp/dm³
  : Real :=
  let weight := 0.78 * cross_sectional_area
  have weight_eq : weight = 146.64 := by sorry
  weight

#check iron_rod_weight

end iron_rod_weight_l2894_289488


namespace f_equals_neg_tan_f_at_eight_pi_thirds_l2894_289406

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin (Real.pi + x) * Real.cos (Real.pi - x) * Real.sin (2 * Real.pi - x)) /
  (Real.sin (Real.pi / 2 + x) * Real.cos (x - Real.pi / 2) * Real.cos (-x))

/-- Theorem stating that f(x) = -tan(x) for all x -/
theorem f_equals_neg_tan (x : ℝ) : f x = -Real.tan x := by sorry

/-- Theorem stating that f(8π/3) = -√3 -/
theorem f_at_eight_pi_thirds : f (8 * Real.pi / 3) = -Real.sqrt 3 := by sorry

end f_equals_neg_tan_f_at_eight_pi_thirds_l2894_289406


namespace bread_roll_combinations_eq_21_l2894_289408

/-- The number of ways to distribute n identical items into k distinct groups -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of combinations of bread rolls Tom could purchase -/
def breadRollCombinations : ℕ := starsAndBars 5 3

theorem bread_roll_combinations_eq_21 : breadRollCombinations = 21 := by
  sorry

end bread_roll_combinations_eq_21_l2894_289408


namespace arithmetic_mean_problem_l2894_289461

theorem arithmetic_mean_problem (x : ℝ) : 
  ((x + 10) + 20 + 3*x + 15 + (3*x + 6)) / 5 = 30 → x = 99 / 7 := by
  sorry

end arithmetic_mean_problem_l2894_289461


namespace evaluate_expression_l2894_289465

theorem evaluate_expression : (2^3001 * 3^3003) / 6^3002 = 3/2 := by sorry

end evaluate_expression_l2894_289465


namespace equation_solution_l2894_289474

theorem equation_solution : ∃ x : ℕ, 9^12 + 9^12 + 9^12 = 3^x ∧ x = 25 := by
  sorry

end equation_solution_l2894_289474


namespace min_value_of_f_l2894_289455

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 3| + Real.exp x

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 6 - Real.log 4 := by
  sorry

end min_value_of_f_l2894_289455


namespace line_y_coordinate_at_15_l2894_289498

/-- A line passing through three given points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem line_y_coordinate_at_15 (l : Line) 
    (h1 : l.point1 = (4, 5))
    (h2 : l.point2 = (8, 17))
    (h3 : l.point3 = (12, 29))
    (h4 : collinear l.point1 l.point2 l.point3) :
    ∃ t : ℝ, collinear l.point1 l.point2 (15, t) ∧ t = 38 := by
  sorry

end line_y_coordinate_at_15_l2894_289498


namespace balloon_radius_increase_l2894_289492

/-- Proves that when a circular object's circumference increases from 24 inches to 30 inches, 
    its radius increases by 3/π inches. -/
theorem balloon_radius_increase (r₁ r₂ : ℝ) : 
  2 * π * r₁ = 24 → 2 * π * r₂ = 30 → r₂ - r₁ = 3 / π := by sorry

end balloon_radius_increase_l2894_289492


namespace roses_given_to_friends_l2894_289433

def total_money : ℕ := 300
def rose_price : ℕ := 2
def jenna_fraction : ℚ := 1/3
def imma_fraction : ℚ := 1/2

theorem roses_given_to_friends :
  let total_roses := total_money / rose_price
  let jenna_roses := (jenna_fraction * total_roses).floor
  let imma_roses := (imma_fraction * total_roses).floor
  jenna_roses + imma_roses = 125 := by sorry

end roses_given_to_friends_l2894_289433


namespace color_preference_theorem_l2894_289499

theorem color_preference_theorem (total_students : ℕ) 
  (blue_percentage : ℚ) (red_percentage : ℚ) :
  total_students = 200 →
  blue_percentage = 30 / 100 →
  red_percentage = 40 / 100 →
  ∃ (blue_students red_students yellow_students : ℕ),
    blue_students = (blue_percentage * total_students).floor ∧
    red_students = (red_percentage * (total_students - blue_students)).floor ∧
    yellow_students = total_students - blue_students - red_students ∧
    blue_students + yellow_students = 144 :=
by sorry

end color_preference_theorem_l2894_289499


namespace cos_negative_300_degrees_l2894_289441

theorem cos_negative_300_degrees : Real.cos (-(300 * π / 180)) = 1 / 2 := by
  sorry

end cos_negative_300_degrees_l2894_289441


namespace triangle_point_distance_inequality_l2894_289497

/-- Given a triangle ABC and a point P in its plane, this theorem proves that
    the sum of the ratios of distances from P to each vertex divided by the opposite side
    is greater than or equal to the square root of 3. -/
theorem triangle_point_distance_inequality 
  (A B C P : ℝ × ℝ) -- Points in 2D plane
  (a b c : ℝ) -- Side lengths of triangle ABC
  (u v ω : ℝ) -- Distances from P to vertices
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) -- Positive side lengths
  (h_triangle : dist B C = a ∧ dist C A = b ∧ dist A B = c) -- Triangle side lengths
  (h_distances : dist P A = u ∧ dist P B = v ∧ dist P C = ω) -- Distances from P to vertices
  : u / a + v / b + ω / c ≥ Real.sqrt 3 := by
  sorry

#check triangle_point_distance_inequality

end triangle_point_distance_inequality_l2894_289497


namespace investment_problem_l2894_289419

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem investment_problem : 
  let principal : ℝ := 4000
  let rate : ℝ := 0.1
  let time : ℕ := 2
  compound_interest principal rate time = 4840.000000000001 := by
sorry

end investment_problem_l2894_289419


namespace point_one_and_ten_are_reciprocals_l2894_289462

/-- Two numbers are reciprocals if their product is 1 -/
def are_reciprocals (a b : ℝ) : Prop := a * b = 1

/-- 0.1 and 10 are reciprocals of each other -/
theorem point_one_and_ten_are_reciprocals : are_reciprocals 0.1 10 := by
  sorry

end point_one_and_ten_are_reciprocals_l2894_289462


namespace line_properties_l2894_289448

/-- Two lines in the plane, parameterized by a -/
def Line1 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => a * x - y + 1 = 0

def Line2 (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => x + a * y + 1 = 0

/-- The theorem stating the properties of the two lines -/
theorem line_properties :
  ∀ a : ℝ,
    (∀ x y : ℝ, Line1 a x y → Line2 a x y → (a * 1 - 1 * a = 0)) ∧ 
    (Line1 a 0 1) ∧
    (Line2 a (-1) 0) ∧
    (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → Line1 a x y → Line2 a x y → x^2 + x + y^2 - y = 0) :=
by sorry

end line_properties_l2894_289448


namespace normal_distribution_symmetry_l2894_289481

-- Define a random variable following normal distribution
def normal_distribution (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def probability (ξ : normal_distribution 1 σ) (a b : ℝ) : ℝ := sorry

-- State the theorem
theorem normal_distribution_symmetry 
  (σ : ℝ) 
  (ξ : normal_distribution 1 σ) 
  (h : probability ξ 0 1 = 0.4) : 
  probability ξ 0 2 = 0.8 := by sorry

end normal_distribution_symmetry_l2894_289481


namespace mark_fruit_consumption_l2894_289437

/-- Given the total number of fruit pieces, the number kept for next week,
    and the number brought to school on Friday, calculate the number of
    pieces eaten in the first four days. -/
def fruitEatenInFourDays (total : ℕ) (keptForNextWeek : ℕ) (broughtFriday : ℕ) : ℕ :=
  total - keptForNextWeek - broughtFriday

/-- Theorem stating that given 10 pieces of fruit, if 2 are kept for next week
    and 3 are brought to school on Friday, then 5 pieces were eaten in the first four days. -/
theorem mark_fruit_consumption :
  fruitEatenInFourDays 10 2 3 = 5 := by
  sorry

#eval fruitEatenInFourDays 10 2 3

end mark_fruit_consumption_l2894_289437


namespace even_product_probability_l2894_289402

-- Define the spinners
def spinner1 : List ℕ := [0, 2]
def spinner2 : List ℕ := [1, 3, 5]

-- Define a function to check if a number is even
def isEven (n : ℕ) : Bool := n % 2 = 0

-- Define a function to calculate the probability of an even product
def probEvenProduct (s1 s2 : List ℕ) : ℚ :=
  let totalOutcomes := s1.length * s2.length
  let evenOutcomes := (s1.filter isEven).length * s2.length
  evenOutcomes / totalOutcomes

-- Theorem statement
theorem even_product_probability :
  probEvenProduct spinner1 spinner2 = 1 := by sorry

end even_product_probability_l2894_289402


namespace min_value_sum_reciprocals_l2894_289487

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_4 : x + y + z = 4) : 
  (∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 4 → 
    9/x + 1/y + 25/z ≤ 9/a + 1/b + 25/c) ∧ 
  9/x + 1/y + 25/z = 20.25 := by
sorry

end min_value_sum_reciprocals_l2894_289487


namespace polynomial_division_quotient_l2894_289475

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 10 * X^3 + 20 * X^2 - 9 * X + 3
  let divisor : Polynomial ℚ := 5 * X + 3
  let quotient : Polynomial ℚ := 2 * X^2 - X
  (dividend).div divisor = quotient := by sorry

end polynomial_division_quotient_l2894_289475


namespace intersection_implies_a_value_l2894_289420

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem intersection_implies_a_value :
  ∀ a : ℝ, (A a) ∩ (B a) = {9} → a = -3 := by sorry

end intersection_implies_a_value_l2894_289420


namespace parallelogram_area_specific_parallelogram_area_l2894_289454

/-- The area of a parallelogram with given base, side length, and included angle --/
theorem parallelogram_area (base : ℝ) (side : ℝ) (angle : ℝ) : 
  base > 0 → side > 0 → 0 < angle ∧ angle < π →
  abs (base * side * Real.sin angle - 498.465) < 0.001 := by
  sorry

/-- Specific instance of the parallelogram area theorem --/
theorem specific_parallelogram_area : 
  abs (22 * 25 * Real.sin (65 * π / 180) - 498.465) < 0.001 := by
  sorry

end parallelogram_area_specific_parallelogram_area_l2894_289454


namespace sickness_temp_increase_l2894_289446

def normal_temp : ℝ := 95
def fever_threshold : ℝ := 100
def above_threshold : ℝ := 5

theorem sickness_temp_increase : 
  let current_temp := fever_threshold + above_threshold
  current_temp - normal_temp = 10 := by sorry

end sickness_temp_increase_l2894_289446


namespace remainder_sum_mod_seven_l2894_289443

theorem remainder_sum_mod_seven : (9^7 + 6^9 + 5^11) % 7 = 4 := by
  sorry

end remainder_sum_mod_seven_l2894_289443


namespace inequality_solution_l2894_289469

theorem inequality_solution (x : ℝ) : 3 * x - 6 > 5 * (x - 2) → x < 2 := by
  sorry

end inequality_solution_l2894_289469


namespace probability_equals_three_elevenths_l2894_289416

/-- A quadruple of non-negative integers satisfying 2p + q + r + s = 4 -/
def ValidQuadruple : Type := 
  { quad : Fin 4 → ℕ // 2 * quad 0 + quad 1 + quad 2 + quad 3 = 4 }

/-- The set of all valid quadruples -/
def AllQuadruples : Finset ValidQuadruple := sorry

/-- The set of quadruples satisfying p + q + r + s = 3 -/
def SatisfyingQuadruples : Finset ValidQuadruple :=
  AllQuadruples.filter (fun quad => quad.val 0 + quad.val 1 + quad.val 2 + quad.val 3 = 3)

theorem probability_equals_three_elevenths :
  Nat.card SatisfyingQuadruples / Nat.card AllQuadruples = 3 / 11 := by sorry

end probability_equals_three_elevenths_l2894_289416


namespace number_pair_uniqueness_l2894_289414

theorem number_pair_uniqueness (S P : ℝ) (h : S^2 ≥ 4*P) :
  let x₁ := (S + Real.sqrt (S^2 - 4*P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4*P)) / 2
  let y₁ := S - x₁
  let y₂ := S - x₂
  ∀ x y : ℝ, (x + y = S ∧ x * y = P) ↔ ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by
  sorry

end number_pair_uniqueness_l2894_289414


namespace sugar_amount_l2894_289484

/-- Represents the quantities of ingredients in a bakery storage room. -/
structure BakeryStorage where
  sugar : ℝ
  flour : ℝ
  bakingSoda : ℝ
  eggs : ℝ
  chocolateChips : ℝ

/-- Represents the ratios between ingredients in the bakery storage room. -/
def BakeryRatios (s : BakeryStorage) : Prop :=
  s.sugar / s.flour = 5 / 2 ∧
  s.flour / s.bakingSoda = 10 / 1 ∧
  s.eggs / s.sugar = 3 / 4 ∧
  s.chocolateChips / s.flour = 3 / 5

/-- Represents the new ratios after adding more baking soda and chocolate chips. -/
def NewRatios (s : BakeryStorage) : Prop :=
  s.flour / (s.bakingSoda + 60) = 8 / 1 ∧
  s.eggs / s.sugar = 5 / 6

/-- Theorem stating that given the conditions, the amount of sugar is 6000 pounds. -/
theorem sugar_amount (s : BakeryStorage) 
  (h1 : BakeryRatios s) (h2 : NewRatios s) : s.sugar = 6000 := by
  sorry


end sugar_amount_l2894_289484


namespace triangle_area_is_86_div_7_l2894_289449

/-- The slope of the first line -/
def m1 : ℚ := 3/4

/-- The slope of the second line -/
def m2 : ℚ := -2

/-- The x-coordinate of the intersection point of the first two lines -/
def x0 : ℚ := 1

/-- The y-coordinate of the intersection point of the first two lines -/
def y0 : ℚ := 3

/-- The equation of the third line: x + y = 8 -/
def line3 (x y : ℚ) : Prop := x + y = 8

/-- The area of the triangle formed by the three lines -/
def triangle_area : ℚ := 86/7

/-- Theorem stating that the area of the triangle is 86/7 -/
theorem triangle_area_is_86_div_7 : triangle_area = 86/7 := by
  sorry

end triangle_area_is_86_div_7_l2894_289449


namespace pam_has_ten_bags_l2894_289467

/-- Represents the number of apples in one of Gerald's bags -/
def geralds_bag_count : ℕ := 40

/-- Represents the total number of apples Pam has -/
def pams_total_apples : ℕ := 1200

/-- Represents the number of Gerald's bags equivalent to one of Pam's bags -/
def bags_ratio : ℕ := 3

/-- Calculates the number of bags Pam has -/
def pams_bag_count : ℕ := pams_total_apples / (bags_ratio * geralds_bag_count)

/-- Theorem stating that Pam has 10 bags of apples -/
theorem pam_has_ten_bags : pams_bag_count = 10 := by
  sorry

end pam_has_ten_bags_l2894_289467


namespace darnels_scooping_rate_l2894_289428

/-- Proves Darrel's scooping rate given the problem conditions -/
theorem darnels_scooping_rate 
  (steven_rate : ℝ) 
  (total_time : ℝ) 
  (total_load : ℝ) 
  (h1 : steven_rate = 75)
  (h2 : total_time = 30)
  (h3 : total_load = 2550) :
  ∃ (darrel_rate : ℝ), 
    (steven_rate + darrel_rate) * total_time = total_load ∧ 
    darrel_rate = 10 := by
  sorry

end darnels_scooping_rate_l2894_289428


namespace molecular_weight_calculation_l2894_289439

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The atomic weight of Deuterium in g/mol -/
def atomic_weight_D : ℝ := 2.01

/-- The number of Barium atoms in the compound -/
def num_Ba : ℕ := 2

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 3

/-- The number of regular Hydrogen atoms in the compound -/
def num_H : ℕ := 4

/-- The number of Deuterium atoms in the compound -/
def num_D : ℕ := 1

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ :=
  (num_Ba : ℝ) * atomic_weight_Ba +
  (num_O : ℝ) * atomic_weight_O +
  (num_H : ℝ) * atomic_weight_H +
  (num_D : ℝ) * atomic_weight_D

theorem molecular_weight_calculation :
  molecular_weight = 328.71 := by sorry

end molecular_weight_calculation_l2894_289439


namespace remainder_problem_l2894_289453

theorem remainder_problem (n : ℤ) (h : n % 9 = 4) : (4 * n - 11) % 9 = 5 := by
  sorry

end remainder_problem_l2894_289453


namespace poly_arrangement_l2894_289460

/-- The original polynomial -/
def original_poly (x y : ℝ) : ℝ := 3*x*y^3 - x^2*y^3 - 9*y + x^3

/-- The polynomial arranged in ascending order of x -/
def arranged_poly (x y : ℝ) : ℝ := -9*y + 3*x*y^3 - x^2*y^3 + x^3

/-- Theorem stating that the arranged polynomial is equivalent to the original polynomial -/
theorem poly_arrangement (x y : ℝ) : original_poly x y = arranged_poly x y := by
  sorry

end poly_arrangement_l2894_289460


namespace inverse_difference_evaluation_l2894_289407

theorem inverse_difference_evaluation (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hxy : 5*x - 3*y ≠ 0) : 
  (5*x - 3*y)⁻¹ * ((5*x)⁻¹ - (3*y)⁻¹) = -1 / (15*x*y) := by
  sorry

end inverse_difference_evaluation_l2894_289407


namespace changsha_gdp_scientific_notation_l2894_289436

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- The GDP value of Changsha city in 2022 -/
def changsha_gdp : ℕ := 1400000000000

/-- Converts a natural number to its scientific notation representation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem changsha_gdp_scientific_notation :
  to_scientific_notation changsha_gdp =
    ScientificNotation.mk 1.4 12 (by norm_num) (by norm_num) :=
  sorry

end changsha_gdp_scientific_notation_l2894_289436


namespace hundreds_digit_of_expression_l2894_289464

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to get the hundreds digit
def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

-- State the theorem
theorem hundreds_digit_of_expression :
  hundreds_digit ((factorial 17 / 5) - (factorial 10 / 2)) = 8 := by
  sorry

end hundreds_digit_of_expression_l2894_289464


namespace bank_deposit_l2894_289403

theorem bank_deposit (n : ℕ) (x y : ℕ) (h1 : n = 100 * x + y) (h2 : 0 ≤ y ∧ y ≤ 99) 
  (h3 : (x : ℝ) + y = 0.02 * n) : n = 4950 := by
  sorry

end bank_deposit_l2894_289403


namespace trig_simplification_l2894_289456

theorem trig_simplification :
  (Real.sin (20 * π / 180) + Real.sin (40 * π / 180) + Real.sin (60 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.sin (30 * π / 180)) =
  8 * Real.cos (40 * π / 180) := by
  sorry

end trig_simplification_l2894_289456


namespace quadratic_necessary_not_sufficient_l2894_289421

theorem quadratic_necessary_not_sufficient :
  (∀ x : ℝ, x > 2 → x^2 + 5*x - 6 > 0) ∧
  (∃ x : ℝ, x^2 + 5*x - 6 > 0 ∧ x ≤ 2) :=
by sorry

end quadratic_necessary_not_sufficient_l2894_289421


namespace parabola_intersects_x_axis_l2894_289430

/-- Given a parabola y = x^2 - 3mx + m + n, prove that for the parabola to intersect
    the x-axis for all real numbers m, n must satisfy n ≤ -1/9 -/
theorem parabola_intersects_x_axis (n : ℝ) :
  (∀ m : ℝ, ∃ x : ℝ, x^2 - 3*m*x + m + n = 0) ↔ n ≤ -1/9 := by sorry

end parabola_intersects_x_axis_l2894_289430


namespace problem_solution_l2894_289401

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sqrt 3 * Real.sin x + Real.cos x) - 1

theorem problem_solution :
  (∀ x ∈ Set.Icc 0 (π/4), f x ≤ 2) ∧
  (∀ x ∈ Set.Icc 0 (π/4), f x ≥ 1) ∧
  (∀ x₀ ∈ Set.Icc (π/4) (π/2), f x₀ = 6/5 → Real.cos (2*x₀) = (3 - 4*Real.sqrt 3)/10) ∧
  (∀ ω > 0, (∀ x ∈ Set.Ioo (π/3) (2*π/3), StrictMono (λ x => f (ω*x))) → ω ≤ 1/4) :=
by sorry

end problem_solution_l2894_289401


namespace walnut_trees_cut_down_count_l2894_289405

/-- The number of walnut trees cut down in the park --/
def walnut_trees_cut_down (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that 13 walnut trees were cut down --/
theorem walnut_trees_cut_down_count : 
  walnut_trees_cut_down 42 29 = 13 := by
  sorry

end walnut_trees_cut_down_count_l2894_289405


namespace sum_reciprocals_inequality_l2894_289445

theorem sum_reciprocals_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 := by
  sorry

end sum_reciprocals_inequality_l2894_289445


namespace tram_length_l2894_289471

/-- The length of a tram given its passing time and tunnel transit time -/
theorem tram_length (passing_time tunnel_time tunnel_length : ℝ) 
  (h1 : passing_time = 4)
  (h2 : tunnel_time = 12)
  (h3 : tunnel_length = 64)
  (h4 : passing_time > 0)
  (h5 : tunnel_time > 0)
  (h6 : tunnel_length > 0) :
  (tunnel_length * passing_time) / (tunnel_time - passing_time) = 32 := by
  sorry

end tram_length_l2894_289471


namespace prob_sum_greater_than_8_l2894_289486

/-- Represents a bag of cards -/
def Bag := Finset Nat

/-- Creates a bag with cards numbered from 0 to 5 -/
def createBag : Bag := Finset.range 6

/-- Calculates the probability of selecting two cards with sum > 8 -/
def probSumGreaterThan8 (bag1 bag2 : Bag) : ℚ :=
  let allPairs := bag1.product bag2
  let favorablePairs := allPairs.filter (fun p => p.1 + p.2 > 8)
  favorablePairs.card / allPairs.card

/-- Main theorem: probability of sum > 8 is 1/12 -/
theorem prob_sum_greater_than_8 :
  probSumGreaterThan8 createBag createBag = 1 / 12 := by
  sorry

end prob_sum_greater_than_8_l2894_289486


namespace pool_filling_time_l2894_289410

/-- Proves the time required to fill a pool given the pool capacity, bucket size, and time per trip -/
theorem pool_filling_time 
  (pool_capacity : ℕ) 
  (bucket_size : ℕ) 
  (seconds_per_trip : ℕ) 
  (h1 : pool_capacity = 84)
  (h2 : bucket_size = 2)
  (h3 : seconds_per_trip = 20) :
  (pool_capacity / bucket_size) * seconds_per_trip / 60 = 14 := by
  sorry

#check pool_filling_time

end pool_filling_time_l2894_289410


namespace fruit_cost_calculation_l2894_289493

/-- The cost of a water bottle in dollars -/
def water_cost : ℚ := 0.5

/-- The cost of a snack in dollars -/
def snack_cost : ℚ := 1

/-- The number of water bottles in a bundle -/
def water_count : ℕ := 1

/-- The number of snacks in a bundle -/
def snack_count : ℕ := 3

/-- The number of fruits in a bundle -/
def fruit_count : ℕ := 2

/-- The selling price of a bundle in dollars -/
def bundle_price : ℚ := 4.6

/-- The cost of each fruit in dollars -/
def fruit_cost : ℚ := 0.55

theorem fruit_cost_calculation :
  water_cost * water_count + snack_cost * snack_count + fruit_cost * fruit_count = bundle_price :=
by sorry

end fruit_cost_calculation_l2894_289493


namespace minimum_value_of_polynomial_l2894_289478

theorem minimum_value_of_polynomial (a b : ℝ) : 
  2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a + 4 * b + 1999 ≥ 1947 ∧ 
  ∃ (a b : ℝ), 2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a + 4 * b + 1999 = 1947 := by
  sorry

end minimum_value_of_polynomial_l2894_289478


namespace infinitely_many_silesian_infinitely_many_non_silesian_l2894_289413

/-- An integer n is Silesian if there exist positive integers a, b, c such that
    n = (a² + b² + c²) / (ab + bc + ca) -/
def is_silesian (n : ℤ) : Prop :=
  ∃ (a b c : ℕ+), n = (a.val^2 + b.val^2 + c.val^2) / (a.val * b.val + b.val * c.val + c.val * a.val)

/-- There are infinitely many Silesian integers -/
theorem infinitely_many_silesian : ∀ N : ℕ, ∃ n : ℤ, n > N ∧ is_silesian n :=
sorry

/-- There are infinitely many positive integers that are not Silesian -/
theorem infinitely_many_non_silesian : ∀ N : ℕ, ∃ k : ℕ, k > N ∧ ¬is_silesian (3 * k) :=
sorry

end infinitely_many_silesian_infinitely_many_non_silesian_l2894_289413


namespace carrie_highlighters_l2894_289434

/-- The total number of highlighters in Carrie's desk drawer -/
def total_highlighters (y p b o g : ℕ) : ℕ := y + p + b + o + g

/-- Theorem stating the total number of highlighters in Carrie's desk drawer -/
theorem carrie_highlighters : ∃ (y p b o g : ℕ),
  y = 7 ∧
  p = y + 7 ∧
  b = p + 5 ∧
  o + g = 21 ∧
  o * 7 = g * 3 ∧
  total_highlighters y p b o g = 61 :=
by sorry

end carrie_highlighters_l2894_289434


namespace problem_solution_l2894_289417

def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

theorem problem_solution :
  (∃ a : ℝ, 
    (A ∩ B a = {x : ℝ | 1/2 ≤ x ∧ x < 2} ∧
     A ∪ B a = {x : ℝ | -2 < x ∧ x ≤ 3})) ∧
  (∀ a : ℝ, (Aᶜ ∩ B a = B a) ↔ a ≥ -1/4) := by
  sorry

end problem_solution_l2894_289417


namespace cubic_roots_properties_l2894_289466

theorem cubic_roots_properties (x₁ x₂ x₃ : ℝ) :
  (x₁^3 - 17*x₁ - 18 = 0) →
  (x₂^3 - 17*x₂ - 18 = 0) →
  (x₃^3 - 17*x₃ - 18 = 0) →
  (-4 < x₁) → (x₁ < -3) →
  (4 < x₃) → (x₃ < 5) →
  (⌊x₂⌋ = -2) ∧ (Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = -π/4) := by
  sorry

end cubic_roots_properties_l2894_289466


namespace factors_180_l2894_289426

/-- The number of positive factors of 180 -/
def num_factors_180 : ℕ :=
  (Finset.filter (· ∣ 180) (Finset.range 181)).card

/-- Theorem stating that the number of positive factors of 180 is 18 -/
theorem factors_180 : num_factors_180 = 18 := by
  sorry

end factors_180_l2894_289426


namespace sixtysecond_term_is_seven_five_l2894_289431

/-- Represents an integer pair in the sequence -/
structure IntegerPair :=
  (first : ℕ)
  (second : ℕ)

/-- Generates the nth term of the sequence -/
def sequenceTerm (n : ℕ) : IntegerPair :=
  sorry

/-- The main theorem stating that the 62nd term is (7,5) -/
theorem sixtysecond_term_is_seven_five :
  sequenceTerm 62 = IntegerPair.mk 7 5 := by
  sorry

end sixtysecond_term_is_seven_five_l2894_289431


namespace optimal_solution_l2894_289447

/-- Represents the prices and quantities of agricultural products A and B --/
structure AgriProducts where
  price_A : ℝ
  price_B : ℝ
  quantity_A : ℝ
  quantity_B : ℝ

/-- Defines the conditions given in the problem --/
def satisfies_conditions (p : AgriProducts) : Prop :=
  2 * p.price_A + 3 * p.price_B = 690 ∧
  p.price_A + 4 * p.price_B = 720 ∧
  p.quantity_A + p.quantity_B = 40 ∧
  p.price_A * p.quantity_A + p.price_B * p.quantity_B ≤ 5400 ∧
  p.quantity_A ≤ 3 * p.quantity_B

/-- Calculates the profit given the prices and quantities --/
def profit (p : AgriProducts) : ℝ :=
  (160 - p.price_A) * p.quantity_A + (200 - p.price_B) * p.quantity_B

/-- Theorem stating the optimal solution --/
theorem optimal_solution :
  ∃ (p : AgriProducts),
    satisfies_conditions p ∧
    p.price_A = 120 ∧
    p.price_B = 150 ∧
    p.quantity_A = 20 ∧
    p.quantity_B = 20 ∧
    ∀ (q : AgriProducts), satisfies_conditions q → profit q ≤ profit p :=
  sorry


end optimal_solution_l2894_289447


namespace perfect_square_property_l2894_289429

theorem perfect_square_property : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 8 * n + 1 = k * k) ∧ 
  (∃ (m : ℕ), n = 2 * m) ∧
  (∃ (p : ℕ), n = p * p) := by
  sorry

end perfect_square_property_l2894_289429


namespace suit_price_calculation_suit_price_theorem_l2894_289490

theorem suit_price_calculation (original_price : ℝ) 
  (increase_rate : ℝ) (reduction_rate : ℝ) : ℝ :=
  let increased_price := original_price * (1 + increase_rate)
  let final_price := increased_price * (1 - reduction_rate)
  final_price

theorem suit_price_theorem : 
  suit_price_calculation 300 0.2 0.1 = 324 := by
  sorry

end suit_price_calculation_suit_price_theorem_l2894_289490


namespace total_lives_theorem_l2894_289483

def cat_lives : ℕ := 9

def dog_lives : ℕ := cat_lives - 3

def mouse_lives : ℕ := dog_lives + 7

def elephant_lives : ℕ := 2 * cat_lives - 5

def fish_lives : ℕ := min (dog_lives + mouse_lives) (elephant_lives / 2)

theorem total_lives_theorem :
  cat_lives + dog_lives + mouse_lives + elephant_lives + fish_lives = 47 := by
  sorry

end total_lives_theorem_l2894_289483


namespace deck_total_cost_l2894_289473

def deck_length : ℝ := 30
def deck_width : ℝ := 40
def base_cost_per_sqft : ℝ := 3
def sealant_cost_per_sqft : ℝ := 1

theorem deck_total_cost :
  deck_length * deck_width * (base_cost_per_sqft + sealant_cost_per_sqft) = 4800 := by
  sorry

end deck_total_cost_l2894_289473


namespace race_time_l2894_289423

/-- The time A takes to complete a 1 kilometer race, given that A can give B a start of 50 meters or 10 seconds. -/
theorem race_time : ℝ := by
  -- Define the race distance
  let race_distance : ℝ := 1000

  -- Define the head start distance
  let head_start_distance : ℝ := 50

  -- Define the head start time
  let head_start_time : ℝ := 10

  -- Define A's time to complete the race
  let time_A : ℝ := 200

  -- Prove that A's time is 200 seconds
  have h1 : race_distance / time_A * (time_A - head_start_time) = race_distance - head_start_distance := by sorry
  
  -- The final statement that proves the theorem
  exact time_A


end race_time_l2894_289423


namespace rachel_tips_l2894_289418

theorem rachel_tips (hourly_wage : ℚ) (people_served : ℕ) (total_made : ℚ) 
  (hw : hourly_wage = 12)
  (ps : people_served = 20)
  (tm : total_made = 37) :
  (total_made - hourly_wage) / people_served = 25 / 20 := by
  sorry

#eval (37 : ℚ) - 12
#eval (25 : ℚ) / 20

end rachel_tips_l2894_289418


namespace sum_of_union_elements_l2894_289450

def A : Finset ℕ := {2, 0, 1, 9}

def B : Finset ℕ := Finset.image (· * 2) A

theorem sum_of_union_elements : Finset.sum (A ∪ B) id = 34 := by
  sorry

end sum_of_union_elements_l2894_289450


namespace problems_completed_is_120_l2894_289411

/-- The number of problems completed given the conditions in the problem -/
def problems_completed (p t : ℕ) : ℕ := p * t

/-- The conditions of the problem -/
def problem_conditions (p t : ℕ) : Prop :=
  p > 15 ∧ t > 0 ∧ p * t = (3 * p - 6) * (t - 3)

/-- The theorem stating that under the given conditions, 120 problems are completed -/
theorem problems_completed_is_120 :
  ∃ p t : ℕ, problem_conditions p t ∧ problems_completed p t = 120 :=
sorry

end problems_completed_is_120_l2894_289411


namespace x_fourth_minus_reciprocal_l2894_289468

theorem x_fourth_minus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 := by
  sorry

end x_fourth_minus_reciprocal_l2894_289468


namespace circle_center_radius_sum_l2894_289438

/-- Given a circle D with equation x^2 + 4y - 16 = -y^2 + 12x + 16,
    prove that its center (c,d) and radius s satisfy c + d + s = 4 + 6√2 -/
theorem circle_center_radius_sum (x y c d s : ℝ) : 
  (∀ x y, x^2 + 4*y - 16 = -y^2 + 12*x + 16) → 
  ((x - c)^2 + (y - d)^2 = s^2) → 
  c + d + s = 4 + 6 * Real.sqrt 2 := by
  sorry

end circle_center_radius_sum_l2894_289438


namespace chocolate_difference_is_fifteen_l2894_289422

/-- The number of chocolates Nick has -/
def nick_chocolates : ℕ := 10

/-- The number of chocolates Alix initially had -/
def alix_initial_chocolates : ℕ := 3 * nick_chocolates

/-- The number of chocolates taken from Alix -/
def chocolates_taken : ℕ := 5

/-- The number of chocolates Alix has after some were taken -/
def alix_remaining_chocolates : ℕ := alix_initial_chocolates - chocolates_taken

/-- The difference in chocolates between Alix and Nick -/
def chocolate_difference : ℕ := alix_remaining_chocolates - nick_chocolates

theorem chocolate_difference_is_fifteen : chocolate_difference = 15 := by
  sorry

end chocolate_difference_is_fifteen_l2894_289422


namespace sequence_properties_l2894_289452

def a (n : ℕ) : ℚ := (2 * n) / (3 * n + 2)

theorem sequence_properties : 
  (a 3 = 6 / 11) ∧ 
  (∀ n : ℕ, a (n - 1) = (2 * n - 2) / (3 * n - 1)) ∧ 
  (a 8 = 8 / 13) := by
  sorry

end sequence_properties_l2894_289452


namespace orange_juice_orders_l2894_289440

/-- Proves that the number of members who ordered orange juice is 12 --/
theorem orange_juice_orders (total_members : ℕ) 
  (h1 : total_members = 30)
  (h2 : ∃ lemon_orders : ℕ, lemon_orders = (2 : ℕ) * total_members / (5 : ℕ))
  (h3 : ∃ remaining : ℕ, remaining = total_members - (2 : ℕ) * total_members / (5 : ℕ))
  (h4 : ∃ mango_orders : ℕ, mango_orders = remaining / (3 : ℕ))
  (h5 : ∃ orange_orders : ℕ, orange_orders = total_members - ((2 : ℕ) * total_members / (5 : ℕ) + remaining / (3 : ℕ))) :
  orange_orders = 12 := by
  sorry

end orange_juice_orders_l2894_289440


namespace second_number_form_l2894_289489

theorem second_number_form (G S : ℕ) (h1 : G = 4) 
  (h2 : ∃ k : ℕ, 1642 = k * G + 6) 
  (h3 : ∃ l : ℕ, S = l * G + 4) : 
  ∃ m : ℕ, S = 4 * m + 4 := by
sorry

end second_number_form_l2894_289489


namespace no_valid_arrangement_l2894_289424

/-- Represents a circular arrangement of numbers from 1 to 60 -/
def CircularArrangement := Fin 60 → ℕ

/-- Checks if the sum of two numbers with k numbers between them is divisible by n -/
def SatisfiesDivisibilityCondition (arr : CircularArrangement) (k n : ℕ) : Prop :=
  ∀ i : Fin 60, (arr i + arr ((i + k + 1) % 60)) % n = 0

/-- Checks if the arrangement satisfies all given conditions -/
def SatisfiesAllConditions (arr : CircularArrangement) : Prop :=
  (∀ i : Fin 60, arr i ∈ Finset.range 60) ∧ 
  (Finset.card (Finset.image arr Finset.univ) = 60) ∧
  SatisfiesDivisibilityCondition arr 1 2 ∧
  SatisfiesDivisibilityCondition arr 2 3 ∧
  SatisfiesDivisibilityCondition arr 6 7

theorem no_valid_arrangement : ¬ ∃ arr : CircularArrangement, SatisfiesAllConditions arr := by
  sorry

end no_valid_arrangement_l2894_289424


namespace angle_trig_values_l2894_289494

theorem angle_trig_values (α : Real) (m : Real) 
  (h1 : m ≠ 0)
  (h2 : Real.sin α = (Real.sqrt 2 / 4) * m)
  (h3 : -Real.sqrt 3 = Real.cos α * Real.sqrt (3 + m^2))
  (h4 : m = Real.sin α * Real.sqrt (3 + m^2)) :
  Real.cos α = -Real.sqrt 6 / 4 ∧ 
  (Real.tan α = Real.sqrt 15 / 3 ∨ Real.tan α = -Real.sqrt 15 / 3) := by
sorry

end angle_trig_values_l2894_289494


namespace beidou_chip_scientific_notation_correct_l2894_289404

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The value of the "Fourth Generation Beidou Chip" size in meters -/
def beidou_chip_size : ℝ := 0.000000022

/-- The scientific notation representation of the Beidou chip size -/
def beidou_chip_scientific : ScientificNotation :=
  { coefficient := 2.2
    exponent := -8
    is_valid := by sorry }

theorem beidou_chip_scientific_notation_correct :
  beidou_chip_size = beidou_chip_scientific.coefficient * (10 : ℝ) ^ beidou_chip_scientific.exponent :=
by sorry

end beidou_chip_scientific_notation_correct_l2894_289404


namespace recurrence_sequence_a8_l2894_289476

/-- A strictly increasing sequence of positive integers satisfying the given recurrence relation. -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, n ≥ 1 → a (n + 2) = a n + a (n + 1))

theorem recurrence_sequence_a8 (a : ℕ → ℕ) (h : RecurrenceSequence a) (h7 : a 7 = 120) : 
  a 8 = 194 := by
  sorry

end recurrence_sequence_a8_l2894_289476


namespace minimum_advantageous_discount_l2894_289412

theorem minimum_advantageous_discount (n : ℕ) : n = 29 ↔ 
  (∀ m : ℕ, m < n → 
    ((1 - m / 100 : ℝ) ≥ (1 - 0.12)^2 ∨
     (1 - m / 100 : ℝ) ≥ (1 - 0.08)^2 * (1 - 0.09) ∨
     (1 - m / 100 : ℝ) ≥ (1 - 0.20) * (1 - 0.10))) ∧
  ((1 - n / 100 : ℝ) < (1 - 0.12)^2 ∧
   (1 - n / 100 : ℝ) < (1 - 0.08)^2 * (1 - 0.09) ∧
   (1 - n / 100 : ℝ) < (1 - 0.20) * (1 - 0.10)) :=
by sorry

end minimum_advantageous_discount_l2894_289412


namespace calculation_proofs_l2894_289432

theorem calculation_proofs :
  (4.5 * 0.9 + 5.5 * 0.9 = 9) ∧
  (1.6 * (2.25 + 10.5 / 1.5) = 14.8) ∧
  (0.36 / ((6.1 - 4.6) * 0.8) = 0.3) := by
  sorry

end calculation_proofs_l2894_289432


namespace circle_area_solution_l2894_289457

theorem circle_area_solution :
  ∃! (x y z : ℕ), 6 * x + 15 * y + 83 * z = 220 ∧ x = 4 ∧ y = 2 ∧ z = 2 := by
sorry

end circle_area_solution_l2894_289457


namespace acme_cheaper_at_min_shirts_l2894_289470

/-- Acme T-Shirt Company's pricing function -/
def acme_cost (x : ℕ) : ℝ := 75 + 12 * x

/-- Gamma T-Shirt Company's pricing function -/
def gamma_cost (x : ℕ) : ℝ := 18 * x

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_for_acme_cheaper : ℕ := 13

theorem acme_cheaper_at_min_shirts :
  acme_cost min_shirts_for_acme_cheaper < gamma_cost min_shirts_for_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_for_acme_cheaper → 
    acme_cost n ≥ gamma_cost n :=
by sorry

end acme_cheaper_at_min_shirts_l2894_289470


namespace max_value_4x_3y_l2894_289435

theorem max_value_4x_3y (x y : ℝ) : 
  x^2 + y^2 = 16*x + 8*y + 10 → 
  (4*x + 3*y ≤ (82.47 : ℝ) / 18) ∧ 
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 16*x₀ + 8*y₀ + 10 ∧ 4*x₀ + 3*y₀ = (82.47 : ℝ) / 18 :=
by sorry

end max_value_4x_3y_l2894_289435


namespace f_1000_value_l2894_289479

def is_multiplicative_to_additive (f : ℕ+ → ℕ) : Prop :=
  ∀ x y : ℕ+, f (x * y) = f x + f y

theorem f_1000_value
  (f : ℕ+ → ℕ)
  (h_mult_add : is_multiplicative_to_additive f)
  (h_10 : f 10 = 16)
  (h_40 : f 40 = 22) :
  f 1000 = 48 :=
sorry

end f_1000_value_l2894_289479


namespace division_subtraction_problem_l2894_289415

theorem division_subtraction_problem (x : ℝ) : 
  (848 / x) - 100 = 6 → x = 8 := by
  sorry

end division_subtraction_problem_l2894_289415


namespace evaluate_expression_l2894_289442

theorem evaluate_expression : 
  let sixteen : ℝ := 2^4
  let eight : ℝ := 2^3
  ∀ ε > 0, |Real.sqrt ((sixteen^15 + eight^20) / (sixteen^7 + eight^21)) - (1/2)| < ε :=
by
  sorry

end evaluate_expression_l2894_289442


namespace max_expression_value_l2894_289444

def expression (x y z w : ℕ) : ℕ := x * y^z - w

theorem max_expression_value :
  ∃ (x y z w : ℕ),
    x ∈ ({0, 1, 2, 3} : Set ℕ) ∧
    y ∈ ({0, 1, 2, 3} : Set ℕ) ∧
    z ∈ ({0, 1, 2, 3} : Set ℕ) ∧
    w ∈ ({0, 1, 2, 3} : Set ℕ) ∧
    x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
    expression x y z w = 24 ∧
    ∀ (a b c d : ℕ),
      a ∈ ({0, 1, 2, 3} : Set ℕ) →
      b ∈ ({0, 1, 2, 3} : Set ℕ) →
      c ∈ ({0, 1, 2, 3} : Set ℕ) →
      d ∈ ({0, 1, 2, 3} : Set ℕ) →
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
      expression a b c d ≤ 24 :=
by sorry

end max_expression_value_l2894_289444


namespace D_180_l2894_289451

/-- 
D(n) represents the number of ways to express a positive integer n 
as a product of integers greater than 1, where the order matters.
-/
def D (n : ℕ+) : ℕ := sorry

/-- The prime factorization of 180 -/
def prime_factorization_180 : List ℕ+ := [2, 2, 3, 3, 5]

/-- Theorem stating that D(180) = 43 -/
theorem D_180 : D 180 = 43 := by sorry

end D_180_l2894_289451


namespace train_braking_problem_l2894_289491

/-- The distance function for the train's motion during braking -/
def S (t : ℝ) : ℝ := 27 * t - 0.45 * t^2

/-- The time when the train stops -/
def stop_time : ℝ := 30

/-- The distance traveled during the braking period -/
def total_distance : ℝ := 405

theorem train_braking_problem :
  (∀ t, t > stop_time → S t < S stop_time) ∧
  S stop_time = total_distance := by
  sorry

end train_braking_problem_l2894_289491


namespace inverse_proportion_y_relation_l2894_289459

theorem inverse_proportion_y_relation (k : ℝ) (y₁ y₂ : ℝ) 
  (h1 : k < 0) 
  (h2 : y₁ = k / (-4)) 
  (h3 : y₂ = k / (-1)) : 
  y₁ < y₂ := by
  sorry

end inverse_proportion_y_relation_l2894_289459


namespace charity_donation_division_l2894_289496

theorem charity_donation_division (total : ℕ) (people : ℕ) (share : ℕ) : 
  total = 1800 → people = 10 → share = total / people → share = 180 := by
  sorry

end charity_donation_division_l2894_289496


namespace line_through_123_quadrants_l2894_289485

-- Define a line in 2D space
structure Line where
  k : ℝ
  b : ℝ

-- Define the property of a line passing through the first, second, and third quadrants
def passesThrough123Quadrants (l : Line) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁ > 0 ∧ y₁ > 0) ∧  -- First quadrant
    (x₂ < 0 ∧ y₂ > 0) ∧  -- Second quadrant
    (x₃ < 0 ∧ y₃ < 0) ∧  -- Third quadrant
    (y₁ = l.k * x₁ + l.b) ∧
    (y₂ = l.k * x₂ + l.b) ∧
    (y₃ = l.k * x₃ + l.b)

-- Theorem statement
theorem line_through_123_quadrants (l : Line) :
  passesThrough123Quadrants l → l.k > 0 ∧ l.b > 0 := by
  sorry

end line_through_123_quadrants_l2894_289485


namespace apple_orange_ratio_l2894_289427

theorem apple_orange_ratio (num_oranges : ℕ) : 
  (15 : ℚ) + num_oranges = 50 * (3/2) → 
  (15 : ℚ) / num_oranges = 1 / 4 := by
sorry

end apple_orange_ratio_l2894_289427


namespace sqrt_equation_solution_l2894_289409

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt ((2 + Real.sqrt 3) ^ x) + Real.sqrt ((2 - Real.sqrt 3) ^ x) = 6 ∧ x = -2 := by
  sorry

end sqrt_equation_solution_l2894_289409


namespace solution_set_of_inequality_l2894_289495

theorem solution_set_of_inequality (x : ℝ) :
  (x - 3) / (x + 2) < 0 ↔ -2 < x ∧ x < 3 := by
sorry

end solution_set_of_inequality_l2894_289495


namespace candy_probability_l2894_289480

def yellow_candies : ℕ := 2
def red_candies : ℕ := 4

def total_candies : ℕ := yellow_candies + red_candies

def favorable_arrangements : ℕ := 1

def total_arrangements : ℕ := Nat.choose total_candies yellow_candies

def probability : ℚ := favorable_arrangements / total_arrangements

theorem candy_probability : probability = 1 / 15 := by sorry

end candy_probability_l2894_289480


namespace sandy_shirt_cost_l2894_289425

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℝ := 13.99

/-- The amount Sandy received for returning a jacket -/
def jacket_return : ℝ := 7.43

/-- The net amount Sandy spent on clothes -/
def net_spend : ℝ := 18.7

/-- The amount Sandy spent on the shirt -/
def shirt_cost : ℝ := net_spend + jacket_return - shorts_cost

theorem sandy_shirt_cost : shirt_cost = 12.14 := by sorry

end sandy_shirt_cost_l2894_289425


namespace executive_board_count_l2894_289482

/-- The number of ways to choose an executive board from a club -/
def choose_executive_board (total_members : ℕ) (board_size : ℕ) (specific_roles : ℕ) : ℕ :=
  Nat.choose total_members board_size * (board_size * (board_size - 1))

/-- Theorem stating the number of ways to choose the executive board -/
theorem executive_board_count :
  choose_executive_board 40 6 2 = 115151400 := by
  sorry

end executive_board_count_l2894_289482


namespace problem_1_problem_2_problem_3_problem_4_l2894_289463

-- Problem 1
theorem problem_1 : 42.67 - (12.95 - 7.33) = 37.05 := by sorry

-- Problem 2
theorem problem_2 : (8.4 - 8.4 * (3.12 - 3.7)) / 0.42 = 31.6 := by sorry

-- Problem 3
theorem problem_3 : 5.13 * 0.23 + 8.7 * 0.513 - 5.13 = 0.513 := by sorry

-- Problem 4
theorem problem_4 : 6.66 * 222 + 3.33 * 556 = 3330 := by sorry

end problem_1_problem_2_problem_3_problem_4_l2894_289463


namespace multiply_sum_problem_l2894_289472

theorem multiply_sum_problem (x : ℝ) (h : x = 62.5) :
  ∃! y : ℝ, ((x + 5) * y / 5) - 5 = 22 := by
  sorry

end multiply_sum_problem_l2894_289472


namespace trajectory_and_intersection_l2894_289477

-- Define the points A and B
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the condition for point C
def condition (C : ℝ × ℝ) : Prop :=
  let (x, y) := C
  (x - 3) * (x + 1) + y * y = 5

-- Define the line l
def line_l (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x - y + 3 = 0

-- State the theorem
theorem trajectory_and_intersection :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ C, condition C ↔ (C.1 - center.1)^2 + (C.2 - center.2)^2 = radius^2) ∧
    (center = (1, 0) ∧ radius = 3) ∧
    (∃ M N : ℝ × ℝ,
      M ≠ N ∧
      condition M ∧ condition N ∧
      line_l M ∧ line_l N ∧
      ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 4) :=
sorry

end trajectory_and_intersection_l2894_289477


namespace jessica_money_l2894_289400

theorem jessica_money (rodney ian jessica : ℕ) 
  (h1 : rodney = ian + 35)
  (h2 : ian = jessica / 2)
  (h3 : jessica = rodney + 15) : 
  jessica = 100 := by
sorry

end jessica_money_l2894_289400
