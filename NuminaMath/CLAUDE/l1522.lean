import Mathlib

namespace NUMINAMATH_CALUDE_range_of_c_l1522_152213

-- Define the sets corresponding to propositions p and q
def p (c : ℝ) : Set ℝ := {x | 1 - c < x ∧ x < 1 + c ∧ c > 0}
def q : Set ℝ := {x | (x - 3)^2 < 16}

-- Define the property that p is a sufficient but not necessary condition for q
def sufficient_not_necessary (c : ℝ) : Prop :=
  p c ⊂ q ∧ p c ≠ q

-- State the theorem
theorem range_of_c :
  ∀ c : ℝ, sufficient_not_necessary c ↔ 0 < c ∧ c ≤ 6 := by sorry

end NUMINAMATH_CALUDE_range_of_c_l1522_152213


namespace NUMINAMATH_CALUDE_meaningful_square_root_range_l1522_152215

theorem meaningful_square_root_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (2 - x)) ↔ x < 2 := by
sorry

end NUMINAMATH_CALUDE_meaningful_square_root_range_l1522_152215


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1522_152250

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x| ≤ 2}
def B : Set ℝ := {x : ℝ | 3*x - 2 ≥ 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1522_152250


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1522_152201

theorem quadratic_roots_relation (p q : ℝ) : 
  (∃ r s : ℝ, (2 * r^2 - 4 * r - 5 = 0) ∧ 
               (2 * s^2 - 4 * s - 5 = 0) ∧ 
               ((r + 3)^2 + p * (r + 3) + q = 0) ∧ 
               ((s + 3)^2 + p * (s + 3) + q = 0)) →
  q = 25/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1522_152201


namespace NUMINAMATH_CALUDE_max_profit_theorem_l1522_152221

/-- Represents the sales data for a week -/
structure WeekData where
  modelA : ℕ
  modelB : ℕ
  revenue : ℕ

/-- Represents the appliance models -/
inductive Model
| A
| B

def purchase_price (m : Model) : ℕ :=
  match m with
  | Model.A => 180
  | Model.B => 160

def selling_price (m : Model) : ℕ :=
  match m with
  | Model.A => 240
  | Model.B => 200

def profit (m : Model) : ℕ :=
  selling_price m - purchase_price m

def total_units : ℕ := 35

def max_budget : ℕ := 6000

def profit_goal : ℕ := 1750

def week1_data : WeekData := ⟨3, 2, 1120⟩

def week2_data : WeekData := ⟨4, 3, 1560⟩

/-- Calculates the total profit for a given number of units of each model -/
def total_profit (units_A units_B : ℕ) : ℕ :=
  units_A * profit Model.A + units_B * profit Model.B

/-- Calculates the total purchase cost for a given number of units of each model -/
def total_cost (units_A units_B : ℕ) : ℕ :=
  units_A * purchase_price Model.A + units_B * purchase_price Model.B

theorem max_profit_theorem :
  ∃ (units_A units_B : ℕ),
    units_A + units_B = total_units ∧
    total_cost units_A units_B ≤ max_budget ∧
    total_profit units_A units_B > profit_goal ∧
    total_profit units_A units_B = 1800 ∧
    ∀ (x y : ℕ), x + y = total_units → total_cost x y ≤ max_budget →
      total_profit x y ≤ total_profit units_A units_B :=
by sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l1522_152221


namespace NUMINAMATH_CALUDE_hamburgers_for_lunch_l1522_152254

theorem hamburgers_for_lunch (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 9 → additional = 3 → total = initial + additional → total = 12 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_for_lunch_l1522_152254


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1522_152242

/-- A geometric sequence with positive common ratio -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_first_term
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_eq : a 1 * a 9 = 2 * a 52)
  (h_a2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1522_152242


namespace NUMINAMATH_CALUDE_pens_kept_each_l1522_152297

/-- Calculates the number of pens Kendra and Tony keep each after giving some away to friends. -/
theorem pens_kept_each (kendra_packs tony_packs pens_per_pack friends_given : ℕ) :
  kendra_packs = 4 →
  tony_packs = 2 →
  pens_per_pack = 3 →
  friends_given = 14 →
  let total_pens := (kendra_packs + tony_packs) * pens_per_pack
  let remaining_pens := total_pens - friends_given
  remaining_pens / 2 = 2 := by
sorry

end NUMINAMATH_CALUDE_pens_kept_each_l1522_152297


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l1522_152211

theorem sum_of_four_numbers : 1234 + 2341 + 3412 + 4123 = 11110 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l1522_152211


namespace NUMINAMATH_CALUDE_unique_difference_of_squares_1979_l1522_152217

theorem unique_difference_of_squares_1979 : 
  ∃! (x y : ℕ), 1979 = x^2 - y^2 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_difference_of_squares_1979_l1522_152217


namespace NUMINAMATH_CALUDE_shooting_probability_theorem_l1522_152271

def shooting_probability (accuracy_A accuracy_B : ℝ) : ℝ × ℝ :=
  let prob_both_two := accuracy_A * accuracy_A * accuracy_B * accuracy_B
  let prob_at_least_three := prob_both_two + 
    accuracy_A * accuracy_A * accuracy_B * (1 - accuracy_B) +
    accuracy_A * (1 - accuracy_A) * accuracy_B * accuracy_B
  (prob_both_two, prob_at_least_three)

theorem shooting_probability_theorem :
  shooting_probability 0.4 0.6 = (0.0576, 0.1824) := by
  sorry

end NUMINAMATH_CALUDE_shooting_probability_theorem_l1522_152271


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l1522_152205

-- Define the ellipse parameters
def a : ℝ := 3
def b_squared : ℝ := 8

-- Define the focal length
def focal_length : ℝ := 2

-- Theorem statement
theorem ellipse_focal_length :
  focal_length = 2 * Real.sqrt (a^2 - b_squared) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l1522_152205


namespace NUMINAMATH_CALUDE_at_least_two_positive_l1522_152233

theorem at_least_two_positive (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c > 0) (h_prod : a * b + b * c + c * a > 0) :
  (a > 0 ∧ b > 0) ∨ (b > 0 ∧ c > 0) ∨ (c > 0 ∧ a > 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_two_positive_l1522_152233


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt7_l1522_152209

theorem sqrt_sum_equals_2sqrt7 :
  Real.sqrt (10 - 2 * Real.sqrt 21) + Real.sqrt (10 + 2 * Real.sqrt 21) = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt7_l1522_152209


namespace NUMINAMATH_CALUDE_rain_problem_l1522_152268

theorem rain_problem (first_hour : ℝ) (second_hour : ℝ) : 
  (second_hour = 2 * first_hour + 7) → 
  (first_hour + second_hour = 22) → 
  (first_hour = 5) := by
sorry

end NUMINAMATH_CALUDE_rain_problem_l1522_152268


namespace NUMINAMATH_CALUDE_smallest_k_with_remainders_l1522_152200

theorem smallest_k_with_remainders : ∃! k : ℕ, 
  k > 1 ∧ 
  k % 19 = 1 ∧ 
  k % 7 = 1 ∧ 
  k % 3 = 1 ∧
  ∀ m : ℕ, (m > 1 ∧ m % 19 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1) → k ≤ m :=
by
  use 400
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainders_l1522_152200


namespace NUMINAMATH_CALUDE_stating_selling_price_is_43_l1522_152265

/-- Represents the selling price of an article when the loss is equal to the profit. -/
def selling_price_equal_loss_profit (cost_price : ℕ) (profit_price : ℕ) : ℕ :=
  cost_price * 2 - profit_price

/-- 
Theorem stating that the selling price of an article is 43 when the loss is equal to the profit,
given that the cost price is 50 and the profit obtained by selling for 57 is the same as the loss
obtained by selling for the unknown price.
-/
theorem selling_price_is_43 :
  selling_price_equal_loss_profit 50 57 = 43 := by
  sorry

#eval selling_price_equal_loss_profit 50 57

end NUMINAMATH_CALUDE_stating_selling_price_is_43_l1522_152265


namespace NUMINAMATH_CALUDE_fifth_term_is_thirteen_l1522_152230

/-- An arithmetic sequence with the first term 1 and common difference 3 -/
def arithmeticSeq : ℕ → ℤ
  | 0 => 1
  | n+1 => arithmeticSeq n + 3

/-- The theorem stating that the 5th term of the sequence is 13 -/
theorem fifth_term_is_thirteen : arithmeticSeq 4 = 13 := by
  sorry

#eval arithmeticSeq 4  -- This will evaluate to 13

end NUMINAMATH_CALUDE_fifth_term_is_thirteen_l1522_152230


namespace NUMINAMATH_CALUDE_three_number_difference_l1522_152216

theorem three_number_difference (x y : ℝ) (h : (23 + x + y) / 3 = 31) :
  max (max 23 x) y - min (min 23 x) y ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_three_number_difference_l1522_152216


namespace NUMINAMATH_CALUDE_girls_more_likely_separated_l1522_152253

/-- The probability of two girls being separated when randomly seated among three boys on a 5-seat bench is greater than the probability of them sitting together. -/
theorem girls_more_likely_separated (n : ℕ) (h : n = 5) :
  let total_arrangements := Nat.choose n 2
  let adjacent_arrangements := n - 1
  (total_arrangements - adjacent_arrangements : ℚ) / total_arrangements > adjacent_arrangements / total_arrangements :=
by
  sorry

end NUMINAMATH_CALUDE_girls_more_likely_separated_l1522_152253


namespace NUMINAMATH_CALUDE_chamber_boundary_area_l1522_152206

/-- The area of the boundary of a chamber formed by three intersecting pipes -/
theorem chamber_boundary_area (pipe_circumference : ℝ) (h1 : pipe_circumference = 4) :
  let pipe_diameter := pipe_circumference / Real.pi
  let cross_section_area := Real.pi * (pipe_diameter / 2) ^ 2
  let chamber_boundary_area := 2 * (1 / 4) * Real.pi * pipe_diameter ^ 2
  chamber_boundary_area = 8 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_chamber_boundary_area_l1522_152206


namespace NUMINAMATH_CALUDE_first_group_size_l1522_152239

/-- The number of persons in the first group that can repair a road -/
def first_group : ℕ :=
  let days : ℕ := 12
  let hours_per_day_first : ℕ := 5
  let second_group : ℕ := 30
  let hours_per_day_second : ℕ := 6
  (second_group * hours_per_day_second) / hours_per_day_first

theorem first_group_size :
  first_group = 36 :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_l1522_152239


namespace NUMINAMATH_CALUDE_binomial_square_constant_l1522_152294

theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + c = (a * x + b)^2) → c = 25 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l1522_152294


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l1522_152225

theorem smallest_positive_solution (x : ℝ) :
  (x > 0 ∧ x / 7 + 2 / (7 * x) = 1) → x = (7 - Real.sqrt 41) / 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l1522_152225


namespace NUMINAMATH_CALUDE_prize_interval_l1522_152249

/-- Proves that the interval between prizes is 400 given the conditions of the tournament. -/
theorem prize_interval (total_prize : ℕ) (first_prize : ℕ) (interval : ℕ) : 
  total_prize = 4800 → 
  first_prize = 2000 → 
  total_prize = first_prize + (first_prize - interval) + (first_prize - 2 * interval) →
  interval = 400 := by
  sorry

#check prize_interval

end NUMINAMATH_CALUDE_prize_interval_l1522_152249


namespace NUMINAMATH_CALUDE_line_y_intercept_l1522_152231

/-- Given a line passing through the points (2, -3) and (6, 5), its y-intercept is -7 -/
theorem line_y_intercept : 
  ∀ (f : ℝ → ℝ), 
  (f 2 = -3) → 
  (f 6 = 5) → 
  (∀ x y, f x = y ↔ ∃ m b, y = m * x + b) →
  (∃ b, f 0 = b) →
  f 0 = -7 := by
sorry

end NUMINAMATH_CALUDE_line_y_intercept_l1522_152231


namespace NUMINAMATH_CALUDE_base7_to_base10_65432_l1522_152293

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number --/
def base7Number : List Nat := [2, 3, 4, 5, 6]

/-- Theorem stating that the base 10 equivalent of 65432 in base 7 is 16340 --/
theorem base7_to_base10_65432 :
  base7ToBase10 base7Number = 16340 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_65432_l1522_152293


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1522_152280

/-- A line y = 3x + d is tangent to the parabola y² = 12x if and only if d = 1 -/
theorem line_tangent_to_parabola (d : ℝ) : 
  (∃ x y : ℝ, y = 3*x + d ∧ y^2 = 12*x ∧ 
    ∀ x' y' : ℝ, y' = 3*x' + d → y'^2 = 12*x' → (x', y') = (x, y)) ↔ 
  d = 1 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1522_152280


namespace NUMINAMATH_CALUDE_well_digging_rate_l1522_152238

/-- Calculates the rate per cubic meter for digging a cylindrical well -/
theorem well_digging_rate (depth : ℝ) (diameter : ℝ) (total_cost : ℝ) : 
  depth = 14 →
  diameter = 3 →
  total_cost = 1583.3626974092558 →
  ∃ (rate : ℝ), abs (rate - 15.993) < 0.001 ∧ 
    rate = total_cost / (Real.pi * (diameter / 2)^2 * depth) := by
  sorry

end NUMINAMATH_CALUDE_well_digging_rate_l1522_152238


namespace NUMINAMATH_CALUDE_sqrt_neg_four_squared_l1522_152223

theorem sqrt_neg_four_squared : Real.sqrt ((-4)^2) = 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_neg_four_squared_l1522_152223


namespace NUMINAMATH_CALUDE_sum_of_digits_power_of_six_l1522_152279

def last_two_digits (n : ℕ) : ℕ := n % 100

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_power_of_six :
  tens_digit (last_two_digits ((4 + 2)^21)) + ones_digit (last_two_digits ((4 + 2)^21)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_of_six_l1522_152279


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l1522_152295

/-- A line passing through a point with equal intercepts on both axes -/
structure EqualInterceptLine where
  -- The point the line passes through
  point : ℝ × ℝ
  -- The equation of the line in the form ax + by = c
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through the given point
  point_on_line : a * point.1 + b * point.2 = c
  -- The line has equal intercepts on both axes
  equal_intercepts : c / a = c / b

/-- The theorem stating the equation of the line -/
theorem equal_intercept_line_equation :
  ∀ (l : EqualInterceptLine),
  l.point = (3, 2) →
  (l.a = 2 ∧ l.b = -3 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = 5) :=
by sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l1522_152295


namespace NUMINAMATH_CALUDE_sum_m_n_equals_negative_one_l1522_152285

theorem sum_m_n_equals_negative_one (m n : ℝ) 
  (h : Real.sqrt (m - 2) + (n + 3)^2 = 0) : m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_negative_one_l1522_152285


namespace NUMINAMATH_CALUDE_inverse_variation_cube_l1522_152278

/-- Given positive real numbers x and y that vary inversely with respect to x^3,
    prove that if y = 8 when x = 2, then x = 0.4 when y = 1000. -/
theorem inverse_variation_cube (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h_inverse : ∃ k : ℝ, ∀ x y, x^3 * y = k) 
    (h_initial : 2^3 * 8 = (x^3 * y)) :
  y = 1000 → x = 0.4 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_l1522_152278


namespace NUMINAMATH_CALUDE_range_of_a_l1522_152220

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- State the theorem
theorem range_of_a :
  (∀ x ∈ Set.Ioo 0 2, f a x ≥ 0) ↔ a ∈ Set.Iic 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1522_152220


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1522_152227

-- Problem 1
theorem problem_1 : -20 - (-8) + (-4) = -16 := by sorry

-- Problem 2
theorem problem_2 : -1^3 * (-2)^2 / (4/3) + |5-8| = 0 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1522_152227


namespace NUMINAMATH_CALUDE_card_sum_problem_l1522_152245

theorem card_sum_problem (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
  sorry

end NUMINAMATH_CALUDE_card_sum_problem_l1522_152245


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_cylinder_l1522_152284

/-- The surface area of a cube with volume equal to a cylinder of radius 4 and height 12 -/
theorem cube_surface_area_equal_volume_cylinder (π : ℝ) :
  let cylinder_volume := π * 4^2 * 12
  let cube_edge := (cylinder_volume)^(1/3)
  let cube_surface_area := 6 * cube_edge^2
  cube_surface_area = 6 * (192 * π)^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_cylinder_l1522_152284


namespace NUMINAMATH_CALUDE_cos_18_deg_root_l1522_152292

theorem cos_18_deg_root : ∃ (p : ℝ → ℝ), (∀ x, p x = 16 * x^4 - 20 * x^2 + 5) ∧ p (Real.cos (18 * Real.pi / 180)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_18_deg_root_l1522_152292


namespace NUMINAMATH_CALUDE_line_point_sum_l1522_152255

/-- The line equation y = -1/2x + 8 --/
def line_equation (x y : ℝ) : Prop := y = -1/2 * x + 8

/-- Point P is where the line crosses the x-axis --/
def P : ℝ × ℝ := (16, 0)

/-- Point Q is where the line crosses the y-axis --/
def Q : ℝ × ℝ := (0, 8)

/-- Point T is on line segment PQ --/
def T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- Area of triangle POQ is twice the area of triangle TOP --/
def area_condition (r s : ℝ) : Prop :=
  abs ((P.1 - 0) * (Q.2 - 0) - (Q.1 - 0) * (P.2 - 0)) / 2 =
  2 * abs ((P.1 - 0) * (s - 0) - (r - 0) * (P.2 - 0)) / 2

theorem line_point_sum :
  ∀ r s : ℝ,
  line_equation r s →
  T_on_PQ r s →
  area_condition r s →
  r + s = 12 := by sorry

end NUMINAMATH_CALUDE_line_point_sum_l1522_152255


namespace NUMINAMATH_CALUDE_expected_total_score_l1522_152212

/-- The number of students participating in the contest -/
def num_students : ℕ := 10

/-- The number of shooting opportunities for each student -/
def shots_per_student : ℕ := 2

/-- The probability of scoring a goal -/
def goal_probability : ℝ := 0.6

/-- The scoring system -/
def score (goals : ℕ) : ℝ :=
  match goals with
  | 0 => 0
  | 1 => 5
  | _ => 10

/-- The expected score for a single student -/
def expected_score_per_student : ℝ :=
  (score 0) * (1 - goal_probability)^2 +
  (score 1) * 2 * goal_probability * (1 - goal_probability) +
  (score 2) * goal_probability^2

/-- Theorem: The expected total score for all students is 60 -/
theorem expected_total_score :
  num_students * expected_score_per_student = 60 := by
  sorry

end NUMINAMATH_CALUDE_expected_total_score_l1522_152212


namespace NUMINAMATH_CALUDE_exists_n_satisfying_conditions_l1522_152214

/-- The number of distinct prime factors of n -/
def omega (n : ℕ) : ℕ := sorry

/-- The sum of the exponents in the prime factorization of n -/
def Omega (n : ℕ) : ℕ := sorry

/-- For any fixed positive integer k and positive reals α and β,
    there exists a positive integer n > 1 satisfying the given conditions -/
theorem exists_n_satisfying_conditions (k : ℕ) (α β : ℝ) 
    (hk : k > 0) (hα : α > 0) (hβ : β > 0) :
  ∃ n : ℕ, n > 1 ∧ 
    (omega (n + k) : ℝ) / (omega n) > α ∧
    (Omega (n + k) : ℝ) / (Omega n) < β := by
  sorry

end NUMINAMATH_CALUDE_exists_n_satisfying_conditions_l1522_152214


namespace NUMINAMATH_CALUDE_complex_equality_modulus_l1522_152202

theorem complex_equality_modulus (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (1 + a * i) * i = 2 - b * i →
  Complex.abs (a + b * i) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_modulus_l1522_152202


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l1522_152276

theorem price_reduction_percentage (original_price current_price : ℝ) 
  (h1 : original_price = 3000)
  (h2 : current_price = 2400) :
  (original_price - current_price) / original_price = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l1522_152276


namespace NUMINAMATH_CALUDE_maria_total_money_l1522_152207

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The number of dimes Maria has initially -/
def initial_dimes : ℕ := 4

/-- The number of quarters Maria has initially -/
def initial_quarters : ℕ := 4

/-- The number of nickels Maria has initially -/
def initial_nickels : ℕ := 7

/-- The number of quarters Maria's mom gives her -/
def additional_quarters : ℕ := 5

/-- The total amount of money Maria has after receiving the additional quarters -/
theorem maria_total_money :
  (initial_dimes * dime_value +
   initial_quarters * quarter_value +
   initial_nickels * nickel_value +
   additional_quarters * quarter_value) = 3 :=
by sorry

end NUMINAMATH_CALUDE_maria_total_money_l1522_152207


namespace NUMINAMATH_CALUDE_spelling_bee_points_ratio_l1522_152259

/-- Represents the spelling bee problem and proves the ratio of Val's points to Max and Dulce's combined points --/
theorem spelling_bee_points_ratio :
  -- Define the given points
  let max_points : ℕ := 5
  let dulce_points : ℕ := 3
  let opponents_points : ℕ := 40
  let points_behind : ℕ := 16

  -- Define Val's points as a multiple of Max and Dulce's combined points
  let val_points : ℕ → ℕ := λ k ↦ k * (max_points + dulce_points)

  -- Define the total points of Max, Dulce, and Val's team
  let team_total_points : ℕ → ℕ := λ k ↦ max_points + dulce_points + val_points k

  -- State the condition that their team's total points plus the points they're behind equals the opponents' points
  ∀ k : ℕ, team_total_points k + points_behind = opponents_points →

  -- Prove that the ratio of Val's points to Max and Dulce's combined points is 2:1
  ∃ k : ℕ, val_points k = 2 * (max_points + dulce_points) := by
  sorry


end NUMINAMATH_CALUDE_spelling_bee_points_ratio_l1522_152259


namespace NUMINAMATH_CALUDE_copy_pages_theorem_l1522_152235

/-- Given a cost per page in cents and a budget in dollars, 
    calculate the maximum number of pages that can be copied. -/
def max_pages_copied (cost_per_page : ℕ) (budget_dollars : ℕ) : ℕ :=
  (budget_dollars * 100) / cost_per_page

/-- Theorem: With a cost of 3 cents per page and a budget of $15, 
    the maximum number of pages that can be copied is 500. -/
theorem copy_pages_theorem : max_pages_copied 3 15 = 500 := by
  sorry

end NUMINAMATH_CALUDE_copy_pages_theorem_l1522_152235


namespace NUMINAMATH_CALUDE_linear_function_properties_l1522_152244

-- Define the linear function
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem linear_function_properties (k b : ℝ) (hk : k < 0) (hb : b > 0) :
  -- 1. The graph passes through the first, second, and fourth quadrants
  (∃ x y, x > 0 ∧ y > 0 ∧ y = linear_function k b x) ∧
  (∃ x y, x < 0 ∧ y > 0 ∧ y = linear_function k b x) ∧
  (∃ x y, x > 0 ∧ y < 0 ∧ y = linear_function k b x) ∧
  -- 2. y decreases as x increases
  (∀ x₁ x₂, x₁ < x₂ → linear_function k b x₁ > linear_function k b x₂) ∧
  -- 3. The graph intersects the y-axis at the point (0, b)
  (linear_function k b 0 = b) ∧
  -- 4. When x > -b/k, y < 0
  (∀ x, x > -b/k → linear_function k b x < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l1522_152244


namespace NUMINAMATH_CALUDE_monster_family_kids_l1522_152290

/-- The number of kids in the monster family -/
def num_kids : ℕ := 3

/-- The number of eyes the mom has -/
def mom_eyes : ℕ := 1

/-- The number of eyes the dad has -/
def dad_eyes : ℕ := 3

/-- The number of eyes each kid has -/
def kid_eyes : ℕ := 4

/-- The total number of eyes in the family -/
def total_eyes : ℕ := 16

theorem monster_family_kids :
  mom_eyes + dad_eyes + num_kids * kid_eyes = total_eyes :=
by sorry

end NUMINAMATH_CALUDE_monster_family_kids_l1522_152290


namespace NUMINAMATH_CALUDE_y_percent_of_x_in_terms_of_z_l1522_152222

theorem y_percent_of_x_in_terms_of_z (x y z : ℝ) 
  (h1 : 0.7 * (x - y) = 0.3 * (x + y))
  (h2 : 0.6 * (x + z) = 0.4 * (y - z)) :
  y = 0.4 * x := by
  sorry

end NUMINAMATH_CALUDE_y_percent_of_x_in_terms_of_z_l1522_152222


namespace NUMINAMATH_CALUDE_problem_solution_l1522_152252

theorem problem_solution : 
  (Real.sqrt 27 + Real.sqrt 3 - 2 * Real.sqrt 12 = 0) ∧
  ((3 + 2 * Real.sqrt 2) * (3 - 2 * Real.sqrt 2) - Real.sqrt 54 / Real.sqrt 6 = -2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1522_152252


namespace NUMINAMATH_CALUDE_perpendicular_skew_lines_iff_plane_exists_l1522_152282

/-- Two lines in 3D space are skew if they are not parallel and do not intersect. -/
def are_skew_lines (a b : Line3D) : Prop := sorry

/-- A line is perpendicular to another line if their direction vectors are orthogonal. -/
def line_perpendicular (a b : Line3D) : Prop := sorry

/-- A plane passes through a line if the line is contained in the plane. -/
def plane_passes_through_line (p : Plane3D) (l : Line3D) : Prop := sorry

/-- A plane is perpendicular to a line if the normal vector of the plane is parallel to the direction vector of the line. -/
def plane_perpendicular_to_line (p : Plane3D) (l : Line3D) : Prop := sorry

/-- Main theorem: For two skew lines, one line is perpendicular to the other if and only if
    there exists a plane passing through the first line and perpendicular to the second line. -/
theorem perpendicular_skew_lines_iff_plane_exists (a b : Line3D) 
  (h : are_skew_lines a b) : 
  line_perpendicular a b ↔ 
  ∃ (p : Plane3D), plane_passes_through_line p a ∧ plane_perpendicular_to_line p b := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_skew_lines_iff_plane_exists_l1522_152282


namespace NUMINAMATH_CALUDE_population_ratio_l1522_152261

-- Define populations as real numbers
variable (P_A P_B P_C P_D P_E P_F : ℝ)

-- Define the relationships between city populations
def population_relations : Prop :=
  (P_A = 8 * P_B) ∧
  (P_B = 5 * P_C) ∧
  (P_D = 3 * P_C) ∧
  (P_D = P_E / 2) ∧
  (P_F = P_A / 4)

-- Theorem to prove
theorem population_ratio (h : population_relations P_A P_B P_C P_D P_E P_F) :
  P_E / P_B = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_population_ratio_l1522_152261


namespace NUMINAMATH_CALUDE_sum_is_composite_l1522_152264

theorem sum_is_composite (a b c d : ℕ+) 
  (h : a^2 - a*b + b^2 = c^2 - c*d + d^2) : 
  ∃ (k m : ℕ+), k > 1 ∧ m > 1 ∧ a + b + c + d = k * m :=
sorry

end NUMINAMATH_CALUDE_sum_is_composite_l1522_152264


namespace NUMINAMATH_CALUDE_g_five_equals_one_l1522_152299

theorem g_five_equals_one (g : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
  (h2 : ∀ x : ℝ, g x ≠ 0) :
  g 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_g_five_equals_one_l1522_152299


namespace NUMINAMATH_CALUDE_average_age_combined_l1522_152286

theorem average_age_combined (num_students : ℕ) (num_guardians : ℕ) 
  (avg_age_students : ℚ) (avg_age_guardians : ℚ) :
  num_students = 40 →
  num_guardians = 60 →
  avg_age_students = 10 →
  avg_age_guardians = 35 →
  (num_students * avg_age_students + num_guardians * avg_age_guardians) / (num_students + num_guardians) = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l1522_152286


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1522_152256

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem about the sum of a_3 and a_5 in a specific geometric sequence. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  IsGeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1522_152256


namespace NUMINAMATH_CALUDE_ratio_sum_in_triangle_l1522_152210

/-- Given a triangle ABC with the following properties:
  - B is the midpoint of AC
  - D divides BC such that BD:DC = 2:1
  - E divides AB such that AE:EB = 1:3
  This theorem proves that the sum of the ratios EF/FC + AF/FD equals 13/4 -/
theorem ratio_sum_in_triangle (A B C D E F : ℝ × ℝ) : 
  let midpoint (P Q : ℝ × ℝ) := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let divide_segment (P Q : ℝ × ℝ) (r s : ℝ) := 
    ((r * Q.1 + s * P.1) / (r + s), (r * Q.2 + s * P.2) / (r + s))
  B = midpoint A C ∧
  D = divide_segment B C 2 1 ∧
  E = divide_segment A B 1 3 →
  let EF := ‖E - F‖
  let FC := ‖F - C‖
  let AF := ‖A - F‖
  let FD := ‖F - D‖
  EF / FC + AF / FD = 13 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_in_triangle_l1522_152210


namespace NUMINAMATH_CALUDE_coordinate_conditions_l1522_152218

theorem coordinate_conditions (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁ = 4 * π / 5 ∧ y₁ = -π / 5)
  (h₂ : x₂ = 12 * π / 5 ∧ y₂ = -3 * π / 5)
  (h₃ : x₃ = 4 * π / 3 ∧ y₃ = -π / 3) :
  (x₁ + 4 * y₁ = 0 ∧ x₁ + 3 * y₁ < π ∧ π - x₁ - 3 * y₁ ≠ 1 ∧ 3 * x₁ + 5 * y₁ > 0) ∧
  (x₂ + 4 * y₂ = 0 ∧ x₂ + 3 * y₂ < π ∧ π - x₂ - 3 * y₂ ≠ 1 ∧ 3 * x₂ + 5 * y₂ > 0) ∧
  (x₃ + 4 * y₃ = 0 ∧ x₃ + 3 * y₃ < π ∧ π - x₃ - 3 * y₃ ≠ 1 ∧ 3 * x₃ + 5 * y₃ > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_coordinate_conditions_l1522_152218


namespace NUMINAMATH_CALUDE_expression_equality_l1522_152234

theorem expression_equality : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1522_152234


namespace NUMINAMATH_CALUDE_problem_solution_l1522_152246

theorem problem_solution :
  (2017^2 - 2016 * 2018 = 1) ∧
  (∀ a b : ℤ, a + b = 7 → a * b = -1 → 
    ((a + b)^2 = 49) ∧ (a^2 - 3*a*b + b^2 = 54)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1522_152246


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_line_slope_intercept_sum_proof_l1522_152203

/-- Given two points A(1, 4) and B(5, 16) on a line, 
    the sum of the line's slope and y-intercept is 4. -/
theorem line_slope_intercept_sum : ℝ → ℝ → Prop :=
  fun (slope : ℝ) (y_intercept : ℝ) =>
    (slope * 1 + y_intercept = 4) ∧  -- Point A satisfies the line equation
    (slope * 5 + y_intercept = 16) ∧ -- Point B satisfies the line equation
    (slope + y_intercept = 4)        -- Sum of slope and y-intercept is 4

/-- Proof of the theorem -/
theorem line_slope_intercept_sum_proof : ∃ (slope : ℝ) (y_intercept : ℝ), 
  line_slope_intercept_sum slope y_intercept := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_line_slope_intercept_sum_proof_l1522_152203


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l1522_152289

def num_boys : ℕ := 1500
def num_girls : ℕ := 1200

theorem boys_to_girls_ratio :
  (num_boys : ℚ) / (num_girls : ℚ) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l1522_152289


namespace NUMINAMATH_CALUDE_unique_solution_iff_k_eq_six_l1522_152236

/-- The equation (x+5)(x+2) = k + 3x has exactly one real solution if and only if k = 6 -/
theorem unique_solution_iff_k_eq_six (k : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x) ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_iff_k_eq_six_l1522_152236


namespace NUMINAMATH_CALUDE_miraflores_can_win_l1522_152298

/-- Represents a voting system with multiple tiers --/
structure VotingSystem :=
  (total_voters : ℕ)
  (supporter_percentage : ℚ)
  (min_group_size : ℕ)
  (max_group_size : ℕ)

/-- Checks if a candidate can win in the given voting system --/
def can_win (vs : VotingSystem) : Prop :=
  ∃ (grouping : ℕ → ℕ),
    (∀ n, vs.min_group_size ≤ grouping n ∧ grouping n ≤ vs.max_group_size) ∧
    (∃ (final_group : ℕ), 
      final_group > 1 ∧
      final_group ≤ vs.total_voters ∧
      (vs.total_voters * vs.supporter_percentage).num * 2 > 
        (vs.total_voters * vs.supporter_percentage).den * final_group)

/-- The main theorem --/
theorem miraflores_can_win :
  ∃ (vs : VotingSystem), 
    vs.total_voters = 20000000 ∧
    vs.supporter_percentage = 1/100 ∧
    vs.min_group_size = 2 ∧
    vs.max_group_size = 5 ∧
    can_win vs :=
  sorry

end NUMINAMATH_CALUDE_miraflores_can_win_l1522_152298


namespace NUMINAMATH_CALUDE_exponent_calculation_l1522_152228

theorem exponent_calculation (a n m k : ℝ) 
  (h1 : a^n = 2) 
  (h2 : a^m = 3) 
  (h3 : a^k = 4) : 
  a^(2*n + m - 2*k) = 3/4 := by
sorry

end NUMINAMATH_CALUDE_exponent_calculation_l1522_152228


namespace NUMINAMATH_CALUDE_multiply_72514_9999_l1522_152270

theorem multiply_72514_9999 : 72514 * 9999 = 725067486 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72514_9999_l1522_152270


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l1522_152224

/-- The function f(x) = x * e^(-x) is increasing on (-∞, 1) -/
theorem f_increasing_on_interval (x : ℝ) : x < 1 → Monotone (fun x => x * Real.exp (-x)) := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l1522_152224


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l1522_152240

theorem arithmetic_sequence_nth_term (x : ℝ) (n : ℕ) : 
  (2*x - 3 = (5*x - 11) - (3*x - 8)) → 
  (5*x - 11 = (3*x + 1) - (3*x - 8)) → 
  (1 + 4*n = 2009) → 
  n = 502 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l1522_152240


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1522_152204

theorem rectangular_field_area (m : ℝ) : ∃ m : ℝ, (3*m + 8)*(m - 3) = 100 ∧ m > 0 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1522_152204


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l1522_152283

/-- The probability of having at least one boy and one girl in a family of four children,
    given that the birth of a boy or a girl is equally likely. -/
theorem prob_at_least_one_boy_one_girl : 
  let p_boy : ℚ := 1/2  -- Probability of having a boy
  let p_girl : ℚ := 1/2  -- Probability of having a girl
  let n : ℕ := 4  -- Number of children
  1 - (p_boy ^ n + p_girl ^ n) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l1522_152283


namespace NUMINAMATH_CALUDE_joes_trip_expenses_l1522_152251

/-- Joe's trip expenses problem -/
theorem joes_trip_expenses (initial_savings : ℕ) (flight_cost : ℕ) (hotel_cost : ℕ) (remaining : ℕ) 
  (h1 : initial_savings = 6000)
  (h2 : flight_cost = 1200)
  (h3 : hotel_cost = 800)
  (h4 : remaining = 1000) :
  initial_savings - flight_cost - hotel_cost - remaining = 3000 := by
  sorry

end NUMINAMATH_CALUDE_joes_trip_expenses_l1522_152251


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1522_152273

-- Define sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1522_152273


namespace NUMINAMATH_CALUDE_binary_1101_is_13_l1522_152237

def binary_to_decimal (b : List Bool) : ℕ :=
  List.foldl (fun acc d => 2 * acc + if d then 1 else 0) 0 b

theorem binary_1101_is_13 :
  binary_to_decimal [true, false, true, true] = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_is_13_l1522_152237


namespace NUMINAMATH_CALUDE_solve_for_time_l1522_152258

-- Define the exponential growth formula
def exponential_growth (P₀ A r t : ℝ) : Prop :=
  A = P₀ * Real.exp (r * t)

-- Theorem statement
theorem solve_for_time (P₀ A r t : ℝ) (h_pos : P₀ > 0) (h_r_nonzero : r ≠ 0) :
  exponential_growth P₀ A r t ↔ t = Real.log (A / P₀) / r :=
sorry

end NUMINAMATH_CALUDE_solve_for_time_l1522_152258


namespace NUMINAMATH_CALUDE_largest_four_digit_mod_5_3_l1522_152272

theorem largest_four_digit_mod_5_3 : 
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 5 = 3 → n ≤ 9998 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_mod_5_3_l1522_152272


namespace NUMINAMATH_CALUDE_always_two_distinct_roots_find_m_value_l1522_152257

/-- The quadratic equation x^2 - (2m + 1)x - 2 = 0 -/
def quadratic (m : ℝ) (x : ℝ) : Prop :=
  x^2 - (2*m + 1)*x - 2 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  (2*m + 1)^2 + 8

theorem always_two_distinct_roots (m : ℝ) :
  discriminant m > 0 :=
sorry

theorem find_m_value (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : quadratic m x₁) 
  (h₂ : quadratic m x₂) 
  (h₃ : x₁ + x₂ + x₁*x₂ = 1) :
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_always_two_distinct_roots_find_m_value_l1522_152257


namespace NUMINAMATH_CALUDE_fraction_subtraction_and_division_l1522_152229

theorem fraction_subtraction_and_division :
  (5/6 - 1/3) / (2/9) = 9/4 := by
sorry

end NUMINAMATH_CALUDE_fraction_subtraction_and_division_l1522_152229


namespace NUMINAMATH_CALUDE_total_nails_eq_252_l1522_152281

/-- The number of nails/claws/toes to be cut -/
def total_nails : ℕ :=
  let dogs := 4
  let parrots := 8
  let cats := 2
  let rabbits := 6
  let dog_nails := dogs * 4 * 4
  let parrot_claws := (parrots - 1) * 2 * 3 + 1 * 2 * 4
  let cat_toes := cats * (2 * 5 + 2 * 4)
  let rabbit_nails := rabbits * (2 * 5 + 3 + 4)
  dog_nails + parrot_claws + cat_toes + rabbit_nails

/-- Theorem stating that the total number of nails/claws/toes to be cut is 252 -/
theorem total_nails_eq_252 : total_nails = 252 := by
  sorry

end NUMINAMATH_CALUDE_total_nails_eq_252_l1522_152281


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l1522_152275

/-- The number of trees in the coconut grove that yield 60 nuts per year -/
def trees_60 (x : ℝ) : ℝ := x + 3

/-- The number of trees in the coconut grove that yield 120 nuts per year -/
def trees_120 (x : ℝ) : ℝ := x

/-- The number of trees in the coconut grove that yield 180 nuts per year -/
def trees_180 (x : ℝ) : ℝ := x - 3

/-- The total number of trees in the coconut grove -/
def total_trees (x : ℝ) : ℝ := trees_60 x + trees_120 x + trees_180 x

/-- The total number of nuts produced by all trees in the coconut grove -/
def total_nuts (x : ℝ) : ℝ := 60 * trees_60 x + 120 * trees_120 x + 180 * trees_180 x

/-- The average yield per tree per year -/
def average_yield : ℝ := 100

theorem coconut_grove_problem :
  ∃ x : ℝ, total_nuts x = average_yield * total_trees x ∧ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l1522_152275


namespace NUMINAMATH_CALUDE_isosceles_triangle_properties_l1522_152267

-- Define the triangle and its properties
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (D : ℝ × ℝ), 
    -- AB = AC (isosceles)
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
    -- Line AB: 2x + y - 4 = 0
    2 * A.1 + A.2 - 4 = 0 ∧ 2 * B.1 + B.2 - 4 = 0 ∧
    -- Median AD: x - y + 1 = 0
    D.1 - D.2 + 1 = 0 ∧
    -- D is midpoint of BC
    D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ∧
    -- Point D: (4, 5)
    D = (4, 5)

-- Theorem statement
theorem isosceles_triangle_properties (A B C : ℝ × ℝ) 
  (h : Triangle A B C) : 
  -- Line BC: x + y - 9 = 0
  B.1 + B.2 - 9 = 0 ∧ C.1 + C.2 - 9 = 0 ∧
  -- Point B: (-5, 14)
  B = (-5, 14) ∧
  -- Point C: (13, -4)
  C = (13, -4) ∧
  -- Line AC: x + 2y - 5 = 0
  A.1 + 2 * A.2 - 5 = 0 ∧ C.1 + 2 * C.2 - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_properties_l1522_152267


namespace NUMINAMATH_CALUDE_front_view_of_given_stack_map_l1522_152219

/-- Represents a stack map as a list of lists of natural numbers -/
def StackMap := List (List Nat)

/-- Calculates the front view of a stack map -/
def frontView (sm : StackMap) : List Nat :=
  let columns := sm.map List.length
  List.map (fun col => List.foldl Nat.max 0 (List.map (fun row => row.getD col 0) sm)) (List.range (columns.foldl Nat.max 0))

/-- The given stack map -/
def givenStackMap : StackMap := [[4, 1], [1, 2, 4], [3, 1]]

theorem front_view_of_given_stack_map :
  frontView givenStackMap = [4, 2, 4] := by sorry

end NUMINAMATH_CALUDE_front_view_of_given_stack_map_l1522_152219


namespace NUMINAMATH_CALUDE_additional_oil_needed_l1522_152260

/-- Calculates the additional oil needed for a car engine -/
theorem additional_oil_needed
  (oil_per_cylinder : ℕ)
  (num_cylinders : ℕ)
  (oil_already_added : ℕ)
  (h1 : oil_per_cylinder = 8)
  (h2 : num_cylinders = 6)
  (h3 : oil_already_added = 16) :
  oil_per_cylinder * num_cylinders - oil_already_added = 32 :=
by sorry

end NUMINAMATH_CALUDE_additional_oil_needed_l1522_152260


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_vertices_l1522_152208

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Add necessary fields

/-- A pyramid with a regular polygon base -/
structure Pyramid (n : ℕ) where
  base : RegularPolygon n

/-- The number of vertices in a pyramid -/
def Pyramid.numVertices (p : Pyramid n) : ℕ := sorry

/-- Theorem: A pyramid with a base that is a regular polygon with six equal angles has 7 vertices -/
theorem hexagonal_pyramid_vertices :
  ∀ (p : Pyramid 6), p.numVertices = 7 := by sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_vertices_l1522_152208


namespace NUMINAMATH_CALUDE_container_capacity_solution_l1522_152296

def container_capacity (replace_volume : ℝ) (num_replacements : ℕ) 
  (final_ratio_original : ℝ) (final_ratio_new : ℝ) : ℝ → Prop :=
  λ C => (C - replace_volume)^num_replacements / C^(num_replacements - 1) = 
    (final_ratio_original / (final_ratio_original + final_ratio_new)) * C

theorem container_capacity_solution :
  ∃ C : ℝ, C > 0 ∧ container_capacity 15 4 81 256 C ∧ 
    C = 15 / (1 - 3 / (337 : ℝ)^(1/4)) :=
by sorry

end NUMINAMATH_CALUDE_container_capacity_solution_l1522_152296


namespace NUMINAMATH_CALUDE_ciphertext_solution_l1522_152241

theorem ciphertext_solution :
  ∃! (x₁ x₂ x₃ x₄ : ℕ),
    x₁ ≤ 25 ∧ x₂ ≤ 25 ∧ x₃ ≤ 25 ∧ x₄ ≤ 25 ∧
    (x₁ + 2*x₂) % 26 = 9 ∧
    (3*x₂) % 26 = 16 ∧
    (x₃ + 2*x₄) % 26 = 23 ∧
    (3*x₄) % 26 = 12 ∧
    x₁ = 7 ∧ x₂ = 14 ∧ x₃ = 15 ∧ x₄ = 4 :=
by sorry

end NUMINAMATH_CALUDE_ciphertext_solution_l1522_152241


namespace NUMINAMATH_CALUDE_sqrt_representation_l1522_152247

theorem sqrt_representation (n : ℕ+) :
  (∃ (x : ℝ), x > 0 ∧ x^2 = n ∧ x = Real.sqrt (Real.sqrt n)) ↔ n = 1 ∧
  (∀ (x : ℝ), x > 0 ∧ x^2 = n → ∃ (m k : ℕ+), x = (k : ℝ) ^ (1 / m : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_representation_l1522_152247


namespace NUMINAMATH_CALUDE_gcd_98_63_l1522_152243

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_63_l1522_152243


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1522_152291

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 4 ∧
  ∀ (y : ℝ), y * |y| = 3 * y + 4 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1522_152291


namespace NUMINAMATH_CALUDE_unique_solution_system_l1522_152266

theorem unique_solution_system (x y : ℝ) :
  (y = (x + 2)^2 ∧ x * y + y = 2) ↔ (x = 2^(1/3) - 2 ∧ y = 2^(2/3)) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1522_152266


namespace NUMINAMATH_CALUDE_extremum_condition_l1522_152248

/-- The function f(x) = ax^3 + x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- The function f has an extremum -/
def has_extremum (a : ℝ) : Prop :=
  ∃ x : ℝ, ∀ y : ℝ, f a x ≤ f a y ∨ f a x ≥ f a y

/-- The necessary and sufficient condition for f to have an extremum is a < 0 -/
theorem extremum_condition (a : ℝ) :
  has_extremum a ↔ a < 0 :=
sorry

end NUMINAMATH_CALUDE_extremum_condition_l1522_152248


namespace NUMINAMATH_CALUDE_domain_of_f_is_all_reals_l1522_152269

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x - 3) ^ (1/3) + (5 - 2 * x) ^ (1/3)

-- Theorem stating that the domain of f is all real numbers
theorem domain_of_f_is_all_reals :
  ∀ x : ℝ, ∃ y : ℝ, f x = y :=
by
  sorry

end NUMINAMATH_CALUDE_domain_of_f_is_all_reals_l1522_152269


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_22_l1522_152274

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

theorem smallest_prime_with_digit_sum_22 :
  ∃ (p : ℕ), is_prime p ∧ digit_sum p = 22 ∧
  ∀ (q : ℕ), is_prime q → digit_sum q = 22 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_22_l1522_152274


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2017_l1522_152277

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f is increasing if f(x) < f(y) whenever x < y -/
def IsIncreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

/-- An arithmetic sequence with common difference d -/
def ArithmeticSequence (x : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, x (n + 1) = x n + d

theorem arithmetic_sequence_2017 (f : ℝ → ℝ) (x : ℕ → ℝ) :
  IsOdd f →
  IsIncreasing f →
  ArithmeticSequence x 2 →
  f (x 7) + f (x 8) = 0 →
  x 2017 = 4019 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2017_l1522_152277


namespace NUMINAMATH_CALUDE_range_of_f_l1522_152232

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 else 2^x

theorem range_of_f :
  Set.range f = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1522_152232


namespace NUMINAMATH_CALUDE_number_minus_six_l1522_152287

theorem number_minus_six (x : ℚ) : x / 5 = 2 → x - 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_minus_six_l1522_152287


namespace NUMINAMATH_CALUDE_misread_weight_correction_l1522_152262

theorem misread_weight_correction (n : ℕ) (incorrect_avg correct_avg misread_weight : ℝ) :
  n = 20 ∧ 
  incorrect_avg = 58.4 ∧ 
  correct_avg = 58.7 ∧ 
  misread_weight = 56 →
  ∃ correct_weight : ℝ,
    correct_weight = 62 ∧
    n * correct_avg = (n - 1) * incorrect_avg + correct_weight ∧
    n * incorrect_avg = (n - 1) * incorrect_avg + misread_weight :=
by sorry

end NUMINAMATH_CALUDE_misread_weight_correction_l1522_152262


namespace NUMINAMATH_CALUDE_cos_two_theta_value_l1522_152226

theorem cos_two_theta_value (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 4) : 
  Real.cos (2 * θ) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_theta_value_l1522_152226


namespace NUMINAMATH_CALUDE_leaves_first_hour_is_seven_l1522_152288

/-- The number of leaves that fell in the first hour -/
def leaves_first_hour : ℕ := 7

/-- The total number of hours -/
def total_hours : ℕ := 3

/-- The rate of leaves falling per hour in the second and third hour -/
def rate_later_hours : ℕ := 4

/-- The average number of leaves that fell per hour over the entire period -/
def average_leaves_per_hour : ℕ := 5

/-- Theorem stating that the number of leaves that fell in the first hour is 7 -/
theorem leaves_first_hour_is_seven :
  leaves_first_hour = 
    total_hours * average_leaves_per_hour - rate_later_hours * (total_hours - 1) :=
by sorry

end NUMINAMATH_CALUDE_leaves_first_hour_is_seven_l1522_152288


namespace NUMINAMATH_CALUDE_max_a_value_l1522_152263

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, 1 + a * Real.cos x ≥ 2/3 * Real.sin (π/2 + 2*x)) → 
  a ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1522_152263
