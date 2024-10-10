import Mathlib

namespace folded_area_ratio_l2530_253082

/-- Represents a rectangular paper with specific folding properties -/
structure FoldedPaper where
  width : ℝ
  length : ℝ
  area : ℝ
  foldedArea : ℝ
  lengthWidthRatio : length = Real.sqrt 3 * width
  areaDefinition : area = length * width
  foldedAreaDefinition : foldedArea = area - (Real.sqrt 3 * width^2) / 6

/-- The ratio of the folded area to the original area is 5/6 -/
theorem folded_area_ratio (paper : FoldedPaper) : 
  paper.foldedArea / paper.area = 5 / 6 := by
  sorry


end folded_area_ratio_l2530_253082


namespace area_of_specific_quadrilateral_l2530_253090

/-- Represents a convex quadrilateral ABCD with specific side lengths and a right angle -/
structure ConvexQuadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  angle_CDA : ℝ
  convex : AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0
  right_angle : angle_CDA = 90

/-- The area of the specific convex quadrilateral ABCD is 62 -/
theorem area_of_specific_quadrilateral (ABCD : ConvexQuadrilateral)
    (h1 : ABCD.AB = 8)
    (h2 : ABCD.BC = 4)
    (h3 : ABCD.CD = 10)
    (h4 : ABCD.DA = 10) :
    Real.sqrt 0 + 62 * Real.sqrt 1 = 62 := by
  sorry

end area_of_specific_quadrilateral_l2530_253090


namespace course_selection_schemes_l2530_253005

theorem course_selection_schemes :
  let total_courses : ℕ := 4
  let student_a_choices : ℕ := 2
  let student_b_choices : ℕ := 3
  let student_c_choices : ℕ := 3
  
  (Nat.choose total_courses student_a_choices) *
  (Nat.choose total_courses student_b_choices) *
  (Nat.choose total_courses student_c_choices) = 96 :=
by
  sorry

end course_selection_schemes_l2530_253005


namespace polynomial_factorization_l2530_253079

theorem polynomial_factorization :
  (∀ x : ℝ, 3 * x^2 - 7 * x - 6 = (x - 3) * (3 * x + 2)) ∧
  (∀ x : ℝ, 6 * x^2 - 7 * x - 5 = (2 * x + 1) * (3 * x - 5)) := by
  sorry

end polynomial_factorization_l2530_253079


namespace number_of_subsets_l2530_253036

universe u

def card {α : Type u} (s : Set α) : ℕ := sorry

theorem number_of_subsets (M A B : Set ℕ) : 
  card M = 10 →
  A ⊆ M →
  B ⊆ M →
  A ∩ B = ∅ →
  card A = 2 →
  card B = 3 →
  card {X : Set ℕ | A ⊆ X ∧ X ⊆ M} = 256 := by sorry

end number_of_subsets_l2530_253036


namespace floor_of_4_7_l2530_253013

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end floor_of_4_7_l2530_253013


namespace picture_area_calculation_l2530_253012

/-- Given a sheet of paper with width, length, and margin, calculates the area of the picture. -/
def picture_area (paper_width paper_length margin : ℝ) : ℝ :=
  (paper_width - 2 * margin) * (paper_length - 2 * margin)

/-- Theorem stating that for a paper of 8.5 by 10 inches with a 1.5-inch margin, 
    the picture area is 38.5 square inches. -/
theorem picture_area_calculation :
  picture_area 8.5 10 1.5 = 38.5 := by
  sorry

#eval picture_area 8.5 10 1.5

end picture_area_calculation_l2530_253012


namespace sum_of_coefficients_l2530_253000

def g (p q r s : ℝ) (x : ℂ) : ℂ :=
  x^4 + p*x^3 + q*x^2 + r*x + s

theorem sum_of_coefficients 
  (p q r s : ℝ) 
  (h1 : g p q r s (3*I) = 0)
  (h2 : g p q r s (1 + 2*I) = 0) : 
  p + q + r + s = -41 := by
  sorry

end sum_of_coefficients_l2530_253000


namespace prime_divisibility_l2530_253092

theorem prime_divisibility (p m n : ℕ) : 
  Prime p → 
  p > 2 → 
  m > 1 → 
  n > 0 → 
  Prime ((m^(p*n) - 1) / (m^n - 1)) → 
  (p * n) ∣ ((p - 1)^n + 1) :=
by sorry

end prime_divisibility_l2530_253092


namespace man_speed_against_current_l2530_253020

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem stating that given the specific conditions,
    the man's speed against the current is 18 kmph. -/
theorem man_speed_against_current :
  speed_against_current 20 1 = 18 := by
  sorry

#eval speed_against_current 20 1

end man_speed_against_current_l2530_253020


namespace ratio_equality_l2530_253026

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc : a^2 + b^2 + c^2 = 25)
  (sum_xyz : x^2 + y^2 + z^2 = 36)
  (sum_axbycz : a*x + b*y + c*z = 30) :
  (a + b + c) / (x + y + z) = 5/6 := by
sorry

end ratio_equality_l2530_253026


namespace swimmers_passing_count_l2530_253054

/-- Represents the swimming scenario -/
structure SwimmingScenario where
  poolLength : ℝ
  speedA : ℝ
  speedB : ℝ
  totalTime : ℝ
  turnDelay : ℝ

/-- Calculates the number of times swimmers pass each other -/
def passingCount (s : SwimmingScenario) : ℕ :=
  sorry

/-- The main theorem stating the number of times swimmers pass each other -/
theorem swimmers_passing_count :
  let s : SwimmingScenario := {
    poolLength := 100,
    speedA := 4,
    speedB := 3,
    totalTime := 900,  -- 15 minutes in seconds
    turnDelay := 5
  }
  passingCount s = 63 := by sorry

end swimmers_passing_count_l2530_253054


namespace greatest_non_sum_of_composites_l2530_253083

def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬ Nat.Prime n

def isSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, isComposite a ∧ isComposite b ∧ a + b = n

theorem greatest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → isSumOfTwoComposites n) ∧
  ¬ isSumOfTwoComposites 11 := by sorry

end greatest_non_sum_of_composites_l2530_253083


namespace age_difference_l2530_253018

/-- The difference in years between individuals a and c -/
def R (a c : ℕ) : ℕ := a - c

/-- The age of an individual after 5 years -/
def L (x : ℕ) : ℕ := x + 5

theorem age_difference (a b c d : ℕ) :
  (L a + L b = L b + L c + 10) →
  (c + d = a + d - 12) →
  R a c = 12 := by
  sorry

end age_difference_l2530_253018


namespace problem_solution_l2530_253031

def S : Set ℝ := {x | (x + 2) / (x - 5) < 0}
def P (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2*a + 15}

theorem problem_solution :
  (S = {x : ℝ | -2 < x ∧ x < 5}) ∧
  (∀ a : ℝ, S ⊆ P a ↔ -5 ≤ a ∧ a ≤ -3) := by sorry

end problem_solution_l2530_253031


namespace waiter_income_fraction_l2530_253041

/-- Given a waiter's salary and tips, where the tips are 7/4 of the salary,
    prove that the fraction of total income from tips is 7/11. -/
theorem waiter_income_fraction (salary : ℚ) (tips : ℚ) (h : tips = (7 / 4) * salary) :
  tips / (salary + tips) = 7 / 11 := by
  sorry

end waiter_income_fraction_l2530_253041


namespace number_divided_by_24_is_19_l2530_253040

theorem number_divided_by_24_is_19 (x : ℤ) : (x / 24 = 19) → x = 456 := by
  sorry

end number_divided_by_24_is_19_l2530_253040


namespace f_inequality_solution_set_f_inequality_a_range_l2530_253098

def f (x : ℝ) : ℝ := |2*x - 2| - |x + 1|

theorem f_inequality_solution_set :
  {x : ℝ | f x ≤ 3} = {x : ℝ | -2/3 ≤ x ∧ x ≤ 6} := by sorry

theorem f_inequality_a_range :
  {a : ℝ | ∀ x, f x ≤ |x + 1| + a^2} = {a : ℝ | a ≤ -2 ∨ 2 ≤ a} := by sorry

end f_inequality_solution_set_f_inequality_a_range_l2530_253098


namespace spanish_test_average_score_l2530_253057

theorem spanish_test_average_score (marco_score margaret_score average_score : ℝ) : 
  marco_score = 0.9 * average_score →
  margaret_score = marco_score + 5 →
  margaret_score = 86 →
  average_score = 90 := by
sorry

end spanish_test_average_score_l2530_253057


namespace simplify_polynomial_l2530_253044

theorem simplify_polynomial (x : ℝ) : 
  4 * x^3 + 5 * x + 6 * x^2 + 10 - (3 - 6 * x^2 - 4 * x^3 + 2 * x) = 
  8 * x^3 + 12 * x^2 + 3 * x + 7 := by sorry

end simplify_polynomial_l2530_253044


namespace dress_shop_inventory_l2530_253033

/-- Proves that given a total space of 200 dresses and 83 red dresses,
    the number of additional blue dresses compared to red dresses is 34. -/
theorem dress_shop_inventory (total_space : Nat) (red_dresses : Nat)
    (h1 : total_space = 200)
    (h2 : red_dresses = 83) :
    total_space - red_dresses - red_dresses = 34 := by
  sorry

end dress_shop_inventory_l2530_253033


namespace equal_expressions_l2530_253045

theorem equal_expressions : 2007 * 2011 - 2008 * 2010 = 2008 * 2012 - 2009 * 2011 := by
  sorry

end equal_expressions_l2530_253045


namespace cube_of_negative_half_x_y_squared_l2530_253093

theorem cube_of_negative_half_x_y_squared (x y : ℝ) :
  (-1/2 * x * y^2)^3 = -1/8 * x^3 * y^6 := by sorry

end cube_of_negative_half_x_y_squared_l2530_253093


namespace division_zero_implies_divisor_greater_l2530_253024

theorem division_zero_implies_divisor_greater (d : ℕ) :
  2016 / d = 0 → d > 2016 := by
sorry

end division_zero_implies_divisor_greater_l2530_253024


namespace solution_product_log_l2530_253097

-- Define the system of equations
def system_of_equations (x y : ℝ) : Prop :=
  (Real.log x / Real.log 225 + Real.log y / Real.log 64 = 4) ∧
  (Real.log 225 / Real.log x - Real.log 64 / Real.log y = 1)

-- State the theorem
theorem solution_product_log (x₁ y₁ x₂ y₂ : ℝ) :
  system_of_equations x₁ y₁ ∧ system_of_equations x₂ y₂ →
  Real.log (x₁ * y₁ * x₂ * y₂) / Real.log 30 = 12 :=
sorry

end solution_product_log_l2530_253097


namespace largest_non_factor_product_of_100_l2530_253059

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem largest_non_factor_product_of_100 :
  ∀ x y : ℕ,
    x ≠ y →
    x > 0 →
    y > 0 →
    is_factor x 100 →
    is_factor y 100 →
    ¬ is_factor (x * y) 100 →
    x * y ≤ 40 :=
by sorry

end largest_non_factor_product_of_100_l2530_253059


namespace max_parts_is_ten_l2530_253070

/-- Represents a Viennese pretzel lying on a table -/
structure ViennesePretzel where
  loops : ℕ
  intersections : ℕ

/-- Represents a straight cut through the pretzel -/
structure StraightCut where
  intersectionsCut : ℕ

/-- The number of parts resulting from a straight cut -/
def numParts (p : ViennesePretzel) (c : StraightCut) : ℕ :=
  c.intersectionsCut + 1

/-- The maximum number of intersections that can be cut by a single straight line -/
def maxIntersectionsCut (p : ViennesePretzel) : ℕ := 9

/-- Theorem stating that the maximum number of parts is 10 -/
theorem max_parts_is_ten (p : ViennesePretzel) :
  ∃ c : StraightCut, numParts p c = 10 ∧
  ∀ c' : StraightCut, numParts p c' ≤ 10 :=
sorry

end max_parts_is_ten_l2530_253070


namespace investment_principal_is_200_l2530_253002

/-- Represents the simple interest investment scenario -/
structure SimpleInterestInvestment where
  principal : ℝ
  rate : ℝ
  amount_after_2_years : ℝ
  amount_after_5_years : ℝ

/-- The simple interest investment satisfies the given conditions -/
def satisfies_conditions (investment : SimpleInterestInvestment) : Prop :=
  investment.amount_after_2_years = investment.principal * (1 + 2 * investment.rate) ∧
  investment.amount_after_5_years = investment.principal * (1 + 5 * investment.rate) ∧
  investment.amount_after_2_years = 260 ∧
  investment.amount_after_5_years = 350

/-- Theorem stating that the investment with the given conditions has a principal of $200 -/
theorem investment_principal_is_200 :
  ∃ (investment : SimpleInterestInvestment), 
    satisfies_conditions investment ∧ investment.principal = 200 := by
  sorry

end investment_principal_is_200_l2530_253002


namespace sector_area_l2530_253038

theorem sector_area (n : Real) (r : Real) (h1 : n = 120) (h2 : r = 3) :
  (n * π * r^2) / 360 = 3 * π := by
  sorry

end sector_area_l2530_253038


namespace diamond_eight_three_l2530_253006

-- Define the diamond operation
def diamond (x y : ℤ) : ℤ :=
  sorry

-- State the theorem
theorem diamond_eight_three : diamond 8 3 = 39 := by
  sorry

-- Define the properties of the diamond operation
axiom diamond_zero (x : ℤ) : diamond x 0 = x

axiom diamond_comm (x y : ℤ) : diamond x y = diamond y x

axiom diamond_recursive (x y : ℤ) : diamond (x + 2) y = diamond x y + 2 * y + 3

end diamond_eight_three_l2530_253006


namespace f_value_for_specific_inputs_l2530_253003

-- Define the function f
def f (m n k p : ℕ) : ℤ := (n^2 - m) * (n^k - m^p)

-- Theorem statement
theorem f_value_for_specific_inputs :
  f 5 3 2 3 = -464 :=
by sorry

end f_value_for_specific_inputs_l2530_253003


namespace range_of_a_l2530_253080

theorem range_of_a (a : ℝ) : 
  (¬ ∃ t : ℝ, t^2 - 2*t - a < 0) → 
  (∀ x : ℝ, x ≤ a → x ≤ -1) ∧ a ≤ -1 :=
sorry

end range_of_a_l2530_253080


namespace cheese_cookies_per_box_l2530_253035

/-- The number of boxes in a carton -/
def boxes_per_carton : ℕ := 12

/-- The price of a pack of cheese cookies in dollars -/
def price_per_pack : ℕ := 1

/-- The cost of a dozen cartons in dollars -/
def cost_dozen_cartons : ℕ := 1440

/-- The number of packs of cheese cookies in each box -/
def packs_per_box : ℕ := 10

theorem cheese_cookies_per_box :
  packs_per_box = 10 := by sorry

end cheese_cookies_per_box_l2530_253035


namespace no_linear_term_in_product_l2530_253028

theorem no_linear_term_in_product (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (2 - x) = a * x^2 + b) → m = 2 := by
  sorry

end no_linear_term_in_product_l2530_253028


namespace money_sharing_calculation_l2530_253063

/-- Proves that given a money sharing scenario with a specific ratio and known amount,
    the total shared amount can be calculated. -/
theorem money_sharing_calculation (mark_ratio nina_ratio oliver_ratio : ℕ)
                                  (nina_amount : ℕ) :
  mark_ratio = 2 →
  nina_ratio = 3 →
  oliver_ratio = 9 →
  nina_amount = 60 →
  ∃ (total : ℕ), total = 280 ∧ 
    nina_amount * (mark_ratio + nina_ratio + oliver_ratio) = total * nina_ratio :=
by
  sorry

end money_sharing_calculation_l2530_253063


namespace largest_number_given_hcf_and_lcm_factors_l2530_253088

theorem largest_number_given_hcf_and_lcm_factors (a b : ℕ+) : 
  (Nat.gcd a b = 40) → 
  (∃ (k : ℕ+), Nat.lcm a b = 40 * 11 * 12 * k) → 
  (max a b = 480) := by
sorry

end largest_number_given_hcf_and_lcm_factors_l2530_253088


namespace jack_weight_l2530_253051

theorem jack_weight (total_weight sam_weight jack_weight : ℕ) : 
  total_weight = 96 →
  jack_weight = sam_weight + 8 →
  total_weight = sam_weight + jack_weight →
  jack_weight = 52 := by
sorry

end jack_weight_l2530_253051


namespace condition_relationship_l2530_253084

theorem condition_relationship (x : ℝ) :
  (∀ x, (x + 2) * (x - 1) < 0 → x < 1) ∧
  (∃ x, x < 1 ∧ ¬((x + 2) * (x - 1) < 0)) :=
by sorry

end condition_relationship_l2530_253084


namespace mans_rowing_speed_l2530_253072

/-- Given a man's speed with and against a stream, calculate his rowing speed in still water. -/
theorem mans_rowing_speed (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 26)
  (h2 : speed_against_stream = 4) :
  (speed_with_stream + speed_against_stream) / 2 = 15 := by
  sorry

#check mans_rowing_speed

end mans_rowing_speed_l2530_253072


namespace correct_probabilities_l2530_253047

def ball_probabilities (total_balls : ℕ) (p_red p_black_or_yellow p_yellow_or_green : ℚ) : Prop :=
  let p_black := 1/4
  let p_yellow := 1/6
  let p_green := 1/4
  total_balls = 12 ∧
  p_red = 1/3 ∧
  p_black_or_yellow = 5/12 ∧
  p_yellow_or_green = 5/12 ∧
  p_red + p_black + p_yellow + p_green = 1 ∧
  p_black_or_yellow = p_black + p_yellow ∧
  p_yellow_or_green = p_yellow + p_green

theorem correct_probabilities : 
  ∀ (total_balls : ℕ) (p_red p_black_or_yellow p_yellow_or_green : ℚ),
  ball_probabilities total_balls p_red p_black_or_yellow p_yellow_or_green := by
  sorry

end correct_probabilities_l2530_253047


namespace unfair_coin_probability_l2530_253065

theorem unfair_coin_probability (p : ℝ) : 
  (0 ≤ p ∧ p ≤ 1) →
  (35 * p^4 * (1-p)^3 = 343/3125) →
  p = 0.7 := by
sorry

end unfair_coin_probability_l2530_253065


namespace intersection_count_l2530_253096

/-- The maximum number of intersection points formed by line segments connecting 
    points on the x-axis to points on the y-axis -/
def max_intersections (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating that for 8 points on the x-axis and 6 points on the y-axis, 
    the maximum number of intersections is 420 -/
theorem intersection_count : max_intersections 8 6 = 420 := by
  sorry

end intersection_count_l2530_253096


namespace quadratic_no_roots_implies_line_not_in_third_quadrant_l2530_253089

theorem quadratic_no_roots_implies_line_not_in_third_quadrant 
  (m : ℝ) (h : ∀ x : ℝ, m * x^2 - 2*x - 1 ≠ 0) :
  ∀ x y : ℝ, y = m*x - m → ¬(x < 0 ∧ y < 0) :=
sorry

end quadratic_no_roots_implies_line_not_in_third_quadrant_l2530_253089


namespace sum_of_coefficients_l2530_253061

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a*x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅) →
  a₁ + a₃ + a₅ = -121 := by
sorry

end sum_of_coefficients_l2530_253061


namespace equation_solutions_l2530_253052

theorem equation_solutions :
  (∃ (s₁ s₂ : Set ℝ),
    s₁ = {x : ℝ | (5 - 2*x)^2 - 16 = 0} ∧
    s₂ = {x : ℝ | 2*(x - 3) = x^2 - 9} ∧
    s₁ = {1/2, 9/2} ∧
    s₂ = {3, -1}) :=
by sorry

end equation_solutions_l2530_253052


namespace total_troll_count_l2530_253074

/-- The number of trolls Erin counted in different locations -/
structure TrollCount where
  forest : ℕ
  bridge : ℕ
  plains : ℕ

/-- The conditions given in the problem -/
def troll_conditions (t : TrollCount) : Prop :=
  t.forest = 6 ∧
  t.bridge = 4 * t.forest - 6 ∧
  t.plains = t.bridge / 2

/-- The theorem stating the total number of trolls Erin counted -/
theorem total_troll_count (t : TrollCount) (h : troll_conditions t) : 
  t.forest + t.bridge + t.plains = 33 := by
  sorry


end total_troll_count_l2530_253074


namespace polygon_sides_from_angle_sum_l2530_253017

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 180 * (n - 2) → angle_sum = 1080 → n = 8 := by
  sorry

end polygon_sides_from_angle_sum_l2530_253017


namespace bricks_used_l2530_253043

/-- Calculates the total number of bricks used in a construction project --/
theorem bricks_used (courses_per_wall : ℕ) (bricks_per_course : ℕ) (total_walls : ℕ) 
  (h1 : courses_per_wall = 15)
  (h2 : bricks_per_course = 25)
  (h3 : total_walls = 8) : 
  (total_walls - 1) * courses_per_wall * bricks_per_course + 
  (courses_per_wall - 1) * bricks_per_course = 2975 := by
  sorry

end bricks_used_l2530_253043


namespace shot_radius_l2530_253046

/-- Given a sphere of radius 4 cm from which 64 equal-sized spherical shots can be made,
    the radius of each shot is 1 cm. -/
theorem shot_radius (R : ℝ) (N : ℕ) (r : ℝ) : R = 4 → N = 64 → (R / r)^3 = N → r = 1 := by
  sorry

end shot_radius_l2530_253046


namespace roots_properties_l2530_253076

-- Define the coefficients of the quadratic equation
def a : ℝ := 24
def b : ℝ := 60
def c : ℝ := -600

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Theorem statement
theorem roots_properties :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation x ∧ quadratic_equation y) →
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation x ∧ quadratic_equation y ∧ x * y = -25 ∧ x + y = -2.5) :=
sorry

end roots_properties_l2530_253076


namespace greatest_integer_less_than_negative_nineteen_fifths_l2530_253077

theorem greatest_integer_less_than_negative_nineteen_fifths :
  Int.floor (-19 / 5 : ℚ) = -4 := by sorry

end greatest_integer_less_than_negative_nineteen_fifths_l2530_253077


namespace x_range_l2530_253011

theorem x_range (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : x + y + z = 1) (h4 : x^2 + y^2 + z^2 = 3) :
  x ∈ Set.Icc 1 (5/3) :=
sorry

end x_range_l2530_253011


namespace complex_fraction_sum_l2530_253027

theorem complex_fraction_sum (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 + Complex.I) / (1 + Complex.I) = Complex.mk a b →
  a + b = 1 := by
sorry

end complex_fraction_sum_l2530_253027


namespace mobile_bus_uses_gps_and_gis_mobile_bus_not_uses_remote_sensing_l2530_253087

/-- Represents the technologies used in the "Mobile Bus" app --/
inductive MobileBusTechnology
  | GPS        : MobileBusTechnology
  | GIS        : MobileBusTechnology
  | RemoteSensing : MobileBusTechnology
  | DigitalEarth  : MobileBusTechnology

/-- The set of technologies used in the "Mobile Bus" app --/
def mobileBusTechnologies : Set MobileBusTechnology :=
  {MobileBusTechnology.GPS, MobileBusTechnology.GIS}

/-- Theorem stating that the "Mobile Bus" app uses GPS and GIS --/
theorem mobile_bus_uses_gps_and_gis :
  MobileBusTechnology.GPS ∈ mobileBusTechnologies ∧
  MobileBusTechnology.GIS ∈ mobileBusTechnologies :=
by sorry

/-- Theorem stating that the "Mobile Bus" app does not use Remote Sensing --/
theorem mobile_bus_not_uses_remote_sensing :
  MobileBusTechnology.RemoteSensing ∉ mobileBusTechnologies :=
by sorry

end mobile_bus_uses_gps_and_gis_mobile_bus_not_uses_remote_sensing_l2530_253087


namespace roots_sum_of_squares_l2530_253029

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 7*a + 7 = 0) → (b^2 - 7*b + 7 = 0) → a^2 + b^2 = 35 := by
  sorry

end roots_sum_of_squares_l2530_253029


namespace largest_n_binomial_sum_equals_binomial_l2530_253009

theorem largest_n_binomial_sum_equals_binomial (n : ℕ) : 
  (Nat.choose 9 4 + Nat.choose 9 5 = Nat.choose 10 n) → n ≤ 5 :=
by sorry

end largest_n_binomial_sum_equals_binomial_l2530_253009


namespace least_perimeter_triangle_l2530_253067

theorem least_perimeter_triangle (a b x : ℕ) (ha : a = 24) (hb : b = 37) : 
  (a + b > x ∧ a + x > b ∧ b + x > a) → (∀ y : ℕ, (a + b > y ∧ a + y > b ∧ b + y > a) → x ≤ y) →
  a + b + x = 75 :=
sorry

end least_perimeter_triangle_l2530_253067


namespace cube_fraction_product_l2530_253050

theorem cube_fraction_product : 
  (((7^3 - 1) / (7^3 + 1)) * 
   ((8^3 - 1) / (8^3 + 1)) * 
   ((9^3 - 1) / (9^3 + 1)) * 
   ((10^3 - 1) / (10^3 + 1)) * 
   ((11^3 - 1) / (11^3 + 1))) = 931 / 946 := by
  sorry

end cube_fraction_product_l2530_253050


namespace events_mutually_exclusive_not_opposite_l2530_253004

structure Bag where
  red : Nat
  white : Nat
  black : Nat

def draw_two_balls (b : Bag) : Nat := b.red + b.white + b.black - 2

def exactly_one_white (b : Bag) : Prop := 
  ∃ (x : Nat), x = 1 ∧ x ≤ b.white ∧ x ≤ draw_two_balls b

def exactly_two_white (b : Bag) : Prop := 
  ∃ (x : Nat), x = 2 ∧ x ≤ b.white ∧ x ≤ draw_two_balls b

def mutually_exclusive (p q : Prop) : Prop :=
  ¬(p ∧ q)

def opposite (p q : Prop) : Prop :=
  (p ↔ ¬q) ∧ (q ↔ ¬p)

theorem events_mutually_exclusive_not_opposite 
  (b : Bag) (h1 : b.red = 3) (h2 : b.white = 2) (h3 : b.black = 1) : 
  mutually_exclusive (exactly_one_white b) (exactly_two_white b) ∧ 
  ¬(opposite (exactly_one_white b) (exactly_two_white b)) := by
  sorry

end events_mutually_exclusive_not_opposite_l2530_253004


namespace total_seashells_l2530_253064

def sam_seashells : ℕ := 18
def mary_seashells : ℕ := 47
def john_seashells : ℕ := 32
def emily_seashells : ℕ := 26

theorem total_seashells : 
  sam_seashells + mary_seashells + john_seashells + emily_seashells = 123 := by
  sorry

end total_seashells_l2530_253064


namespace polynomial_roots_l2530_253030

theorem polynomial_roots : 
  ∀ z : ℂ, z^4 - 6*z^2 + z + 8 = 0 ↔ z = -2 ∨ z = 1 ∨ z = Complex.I * Real.sqrt 7 ∨ z = -Complex.I * Real.sqrt 7 := by
  sorry

end polynomial_roots_l2530_253030


namespace circle_equation_from_parabola_focus_l2530_253015

/-- The equation of a circle with its center at the focus of the parabola y² = 4x
    and passing through the origin is x² + y² - 2x = 0. -/
theorem circle_equation_from_parabola_focus (x y : ℝ) : 
  (∃ (h : ℝ), y^2 = 4*x ∧ h = 1) →  -- Focus of parabola y² = 4x is at (1, 0)
  (0^2 + 0^2 = (x - 1)^2 + y^2) →  -- Circle passes through origin (0, 0)
  (x^2 + y^2 - 2*x = 0) :=
by sorry

end circle_equation_from_parabola_focus_l2530_253015


namespace chicken_cost_problem_l2530_253086

/-- A problem about calculating the cost of chickens given various expenses --/
theorem chicken_cost_problem (land_acres : ℕ) (land_cost_per_acre : ℕ) 
  (house_cost : ℕ) (cow_count : ℕ) (cow_cost : ℕ) (chicken_count : ℕ) 
  (solar_install_hours : ℕ) (solar_install_rate : ℕ) (solar_equipment_cost : ℕ) 
  (total_cost : ℕ) : 
  land_acres = 30 →
  land_cost_per_acre = 20 →
  house_cost = 120000 →
  cow_count = 20 →
  cow_cost = 1000 →
  chicken_count = 100 →
  solar_install_hours = 6 →
  solar_install_rate = 100 →
  solar_equipment_cost = 6000 →
  total_cost = 147700 →
  (total_cost - (land_acres * land_cost_per_acre + house_cost + cow_count * cow_cost + 
    solar_install_hours * solar_install_rate + solar_equipment_cost)) / chicken_count = 5 :=
by sorry

end chicken_cost_problem_l2530_253086


namespace polynomial_evaluation_l2530_253068

theorem polynomial_evaluation : 
  let a : ℝ := 2
  (3 * a^3 - 7 * a^2 + a - 5) * (4 * a - 6) = -14 := by
sorry

end polynomial_evaluation_l2530_253068


namespace subtract_percentage_equivalent_to_multiply_l2530_253007

theorem subtract_percentage_equivalent_to_multiply (a : ℝ) : 
  a - (0.04 * a) = 0.96 * a := by sorry

end subtract_percentage_equivalent_to_multiply_l2530_253007


namespace cos_225_degrees_l2530_253023

theorem cos_225_degrees : 
  Real.cos (225 * π / 180) = -1 / Real.sqrt 2 := by
  have cos_addition : ∀ θ, Real.cos (π + θ) = -Real.cos θ := sorry
  have cos_45_degrees : Real.cos (45 * π / 180) = 1 / Real.sqrt 2 := sorry
  sorry

end cos_225_degrees_l2530_253023


namespace z_in_first_quadrant_l2530_253049

theorem z_in_first_quadrant (z : ℂ) (h : z * (1 - 3*I) = 5 - 5*I) : 
  0 < z.re ∧ 0 < z.im :=
sorry

end z_in_first_quadrant_l2530_253049


namespace percentage_difference_l2530_253085

theorem percentage_difference (a b : ℝ) (h : b ≠ 0) :
  (a - b) / b * 100 = 25 → a = 100 ∧ b = 80 := by
  sorry

end percentage_difference_l2530_253085


namespace lap_time_improvement_l2530_253055

-- Define the initial swimming scenario
def initial_total_time : ℚ := 29
def initial_break_time : ℚ := 3
def initial_laps : ℚ := 14

-- Define the current swimming scenario
def current_total_time : ℚ := 28
def current_laps : ℚ := 16

-- Define the lap time calculation function
def lap_time (total_time : ℚ) (break_time : ℚ) (laps : ℚ) : ℚ :=
  (total_time - break_time) / laps

-- State the theorem
theorem lap_time_improvement :
  lap_time initial_total_time initial_break_time initial_laps -
  lap_time current_total_time 0 current_laps = 3 / 28 := by
  sorry

end lap_time_improvement_l2530_253055


namespace initial_games_count_l2530_253058

theorem initial_games_count (initial remaining given : ℕ) : 
  remaining = initial - given → 
  given = 7 → 
  remaining = 91 → 
  initial = 98 := by sorry

end initial_games_count_l2530_253058


namespace inverse_f_at_10_l2530_253056

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the domain of f
def f_domain (x : ℝ) : Prop := x ≥ 1

-- State the theorem
theorem inverse_f_at_10 (f_inv : ℝ → ℝ) 
  (h1 : ∀ x, f_domain x → f_inv (f x) = x) 
  (h2 : ∀ y, y ≥ 1 → f (f_inv y) = y) : 
  f_inv 10 = 3 := by sorry

end inverse_f_at_10_l2530_253056


namespace total_trees_after_planting_l2530_253069

/-- The number of walnut trees initially in the park -/
def initial_trees : ℕ := 4

/-- The number of new walnut trees to be planted -/
def new_trees : ℕ := 6

/-- Theorem: The total number of walnut trees after planting is 10 -/
theorem total_trees_after_planting : 
  initial_trees + new_trees = 10 := by sorry

end total_trees_after_planting_l2530_253069


namespace same_color_eyes_percentage_l2530_253081

/-- Represents the proportion of students with a specific eye color combination -/
structure EyeColorProportion where
  eggCream : ℝ    -- proportion of students with eggshell and cream eyes
  eggCorn : ℝ     -- proportion of students with eggshell and cornsilk eyes
  eggEgg : ℝ      -- proportion of students with both eggshell eyes
  creamCorn : ℝ   -- proportion of students with cream and cornsilk eyes
  creamCream : ℝ  -- proportion of students with both cream eyes
  cornCorn : ℝ    -- proportion of students with both cornsilk eyes

/-- The conditions given in the problem -/
def eyeColorConditions (p : EyeColorProportion) : Prop :=
  p.eggCream + p.eggCorn + p.eggEgg = 0.3 ∧
  p.eggCream + p.creamCorn + p.creamCream = 0.4 ∧
  p.eggCorn + p.creamCorn + p.cornCorn = 0.5 ∧
  p.eggCream + p.eggCorn + p.eggEgg + p.creamCorn + p.creamCream + p.cornCorn = 1

/-- The theorem to be proved -/
theorem same_color_eyes_percentage (p : EyeColorProportion) 
  (h : eyeColorConditions p) : 
  p.eggEgg + p.creamCream + p.cornCorn = 0.8 := by
  sorry


end same_color_eyes_percentage_l2530_253081


namespace probability_even_8_sided_die_l2530_253039

/-- A fair 8-sided die -/
def fair_8_sided_die : Finset ℕ := Finset.range 8

/-- The set of even outcomes on the die -/
def even_outcomes : Finset ℕ := Finset.filter (λ x => x % 2 = 0) fair_8_sided_die

/-- The probability of an event occurring when rolling the die -/
def probability (event : Finset ℕ) : ℚ :=
  (event.card : ℚ) / (fair_8_sided_die.card : ℚ)

theorem probability_even_8_sided_die :
  probability even_outcomes = 1/2 := by sorry

end probability_even_8_sided_die_l2530_253039


namespace base_eight_23456_equals_10030_l2530_253016

def base_eight_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_eight_23456_equals_10030 :
  base_eight_to_ten [6, 5, 4, 3, 2] = 10030 := by
  sorry

end base_eight_23456_equals_10030_l2530_253016


namespace bakery_sugar_amount_l2530_253062

/-- Given the ratios of ingredients in a bakery storage room, prove the amount of sugar. -/
theorem bakery_sugar_amount (sugar flour baking_soda : ℚ) 
  (h1 : sugar / flour = 5 / 2)
  (h2 : flour / baking_soda = 10 / 1)
  (h3 : flour / (baking_soda + 60) = 8 / 1) :
  sugar = 6000 := by
  sorry

end bakery_sugar_amount_l2530_253062


namespace trajectory_and_PQ_length_l2530_253025

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the point A on C₁
def A_on_C₁ (x₀ y₀ : ℝ) : Prop := C₁ x₀ y₀

-- Define the perpendicular condition for AN
def AN_perp_x (x₀ y₀ : ℝ) : Prop := ∃ (N : ℝ × ℝ), N.1 = x₀ ∧ N.2 = 0

-- Define the condition for point M
def M_condition (x y x₀ y₀ : ℝ) : Prop :=
  ∃ (N : ℝ × ℝ), N.1 = x₀ ∧ N.2 = 0 ∧
  (x, y) + 2 * (x - x₀, y - y₀) = (2 * Real.sqrt 2 - 2) • (x₀, 0)

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the intersection of line l with curve C
def l_intersects_C (P Q : ℝ × ℝ) : Prop :=
  C P.1 P.2 ∧ C Q.1 Q.2 ∧ P ≠ Q

-- Define the condition for circle PQ passing through O
def circle_PQ_through_O (P Q : ℝ × ℝ) : Prop :=
  P.1 * Q.1 + P.2 * Q.2 = 0

theorem trajectory_and_PQ_length :
  ∀ (x y x₀ y₀ : ℝ) (P Q : ℝ × ℝ),
  A_on_C₁ x₀ y₀ →
  AN_perp_x x₀ y₀ →
  M_condition x y x₀ y₀ →
  l_intersects_C P Q →
  circle_PQ_through_O P Q →
  (C x y ∧ 
   (4 * Real.sqrt 6 / 3)^2 ≤ ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ∧
   ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ (2 * Real.sqrt 3)^2) :=
by sorry

end trajectory_and_PQ_length_l2530_253025


namespace percentage_problem_l2530_253010

theorem percentage_problem (x : ℝ) (h : 24 = (75 / 100) * x) : x = 32 := by
  sorry

end percentage_problem_l2530_253010


namespace min_value_expression_lower_bound_achievable_l2530_253048

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_ineq : 21 * a * b + 2 * b * c + 8 * c * a ≤ 12) :
  1 / a + 2 / b + 3 / c ≥ 15 / 2 := by
  sorry

theorem lower_bound_achievable :
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧
  21 * a * b + 2 * b * c + 8 * c * a ≤ 12 ∧
  1 / a + 2 / b + 3 / c = 15 / 2 := by
  sorry

end min_value_expression_lower_bound_achievable_l2530_253048


namespace no_valid_b_exists_l2530_253022

/-- Given a point P and its symmetric point Q about the origin, 
    prove that there is no real value of b for which both points 
    satisfy the inequality 2x - by + 1 ≤ 0. -/
theorem no_valid_b_exists (P : ℝ × ℝ) (Q : ℝ × ℝ) : 
  P = (1, -2) → 
  Q.1 = -P.1 → 
  Q.2 = -P.2 → 
  ¬∃ b : ℝ, (2 * P.1 - b * P.2 + 1 ≤ 0) ∧ (2 * Q.1 - b * Q.2 + 1 ≤ 0) :=
by sorry

end no_valid_b_exists_l2530_253022


namespace exam_candidates_l2530_253094

theorem exam_candidates (average_marks : ℝ) (total_marks : ℝ) (h1 : average_marks = 35) (h2 : total_marks = 4200) :
  total_marks / average_marks = 120 := by
  sorry

end exam_candidates_l2530_253094


namespace mary_marbles_l2530_253034

def dan_marbles : ℕ := 5
def mary_multiplier : ℕ := 2

theorem mary_marbles : 
  dan_marbles * mary_multiplier = 10 := by sorry

end mary_marbles_l2530_253034


namespace pe_class_size_l2530_253008

theorem pe_class_size (fourth_grade_classes : ℕ) (students_per_class : ℕ) (total_cupcakes : ℕ) :
  fourth_grade_classes = 3 →
  students_per_class = 30 →
  total_cupcakes = 140 →
  total_cupcakes - (fourth_grade_classes * students_per_class) = 50 :=
by
  sorry

end pe_class_size_l2530_253008


namespace car_fuel_efficiency_l2530_253032

/-- Given a car that travels 140 kilometers using 3.5 gallons of gasoline,
    prove that the car's fuel efficiency is 40 kilometers per gallon. -/
theorem car_fuel_efficiency :
  let distance : ℝ := 140  -- Total distance in kilometers
  let fuel : ℝ := 3.5      -- Fuel used in gallons
  let efficiency : ℝ := distance / fuel  -- Fuel efficiency in km/gallon
  efficiency = 40 := by sorry

end car_fuel_efficiency_l2530_253032


namespace unique_prime_factors_count_l2530_253042

def product : ℕ := 102 * 103 * 105 * 107

theorem unique_prime_factors_count :
  (Nat.factors product).toFinset.card = 7 := by sorry

end unique_prime_factors_count_l2530_253042


namespace unique_solution_is_one_l2530_253095

noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.log x + x - 1

theorem unique_solution_is_one :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

end unique_solution_is_one_l2530_253095


namespace third_degree_polynomial_property_l2530_253014

/-- A third-degree polynomial with real coefficients. -/
def ThirdDegreePolynomial := ℝ → ℝ

/-- The property that |f(1)| = |f(2)| = |f(4)| = 10 -/
def SatisfiesCondition (f : ThirdDegreePolynomial) : Prop :=
  |f 1| = 10 ∧ |f 2| = 10 ∧ |f 4| = 10

theorem third_degree_polynomial_property (f : ThirdDegreePolynomial) 
  (h : SatisfiesCondition f) : |f 0| = 34/3 := by
  sorry

end third_degree_polynomial_property_l2530_253014


namespace chord_line_equation_l2530_253019

/-- The equation of a line passing through a chord of an ellipse -/
theorem chord_line_equation (x₁ y₁ x₂ y₂ : ℝ) :
  (x₁^2 / 36 + y₁^2 / 9 = 1) →
  (x₂^2 / 36 + y₂^2 / 9 = 1) →
  ((x₁ + x₂) / 2 = 4) →
  ((y₁ + y₂) / 2 = 2) →
  (∀ x y : ℝ, y - 2 = -(1/2) * (x - 4) ↔ x + 2*y - 8 = 0) :=
by sorry

end chord_line_equation_l2530_253019


namespace root_property_l2530_253099

theorem root_property (a : ℝ) (h : a^2 + a - 2009 = 0) : a^2 + a - 1 = 2008 := by
  sorry

end root_property_l2530_253099


namespace change_amount_l2530_253071

-- Define the given conditions
def pants_price : ℚ := 60
def shirt_price : ℚ := 45
def tie_price : ℚ := 20
def discount_rate : ℚ := 0.1
def tax_rate : ℚ := 0.075
def paid_amount : ℚ := 500

-- Define the calculation steps
def pants_total : ℚ := 3 * pants_price
def shirts_total : ℚ := 2 * shirt_price
def discount_amount : ℚ := discount_rate * shirt_price
def discounted_shirts_total : ℚ := shirts_total - discount_amount
def subtotal : ℚ := pants_total + discounted_shirts_total + tie_price
def tax_amount : ℚ := tax_rate * subtotal
def total_purchase : ℚ := subtotal + tax_amount
def change : ℚ := paid_amount - total_purchase

-- Theorem to prove
theorem change_amount : change = 193.09 := by
  sorry

end change_amount_l2530_253071


namespace min_red_beads_l2530_253001

/-- Represents a necklace with blue and red beads. -/
structure Necklace where
  blue_count : ℕ
  red_count : ℕ
  cyclic : Bool
  segment_condition : Bool

/-- Checks if a necklace satisfies the given conditions. -/
def is_valid_necklace (n : Necklace) : Prop :=
  n.blue_count = 50 ∧
  n.cyclic ∧
  n.segment_condition

/-- Theorem stating the minimum number of red beads required. -/
theorem min_red_beads (n : Necklace) :
  is_valid_necklace n → n.red_count ≥ 29 := by
  sorry

#check min_red_beads

end min_red_beads_l2530_253001


namespace sixth_graders_count_l2530_253066

theorem sixth_graders_count (seventh_graders : ℕ) (seventh_percent : ℚ) (sixth_percent : ℚ) 
  (h1 : seventh_graders = 64)
  (h2 : seventh_percent = 32 / 100)
  (h3 : sixth_percent = 38 / 100)
  (h4 : seventh_graders = (seventh_percent * (seventh_graders / seventh_percent)).floor) :
  (sixth_percent * (seventh_graders / seventh_percent)).floor = 76 := by
  sorry

end sixth_graders_count_l2530_253066


namespace willowton_vampires_l2530_253073

def vampire_growth (initial_population : ℕ) (initial_vampires : ℕ) (turns_per_night : ℕ) (nights : ℕ) : ℕ :=
  sorry

theorem willowton_vampires :
  vampire_growth 300 2 5 2 = 72 :=
sorry

end willowton_vampires_l2530_253073


namespace p_and_not_q_is_true_l2530_253060

-- Define proposition p
def p : Prop := ∃ x : ℝ, x - 2 > 0

-- Define proposition q
def q : Prop := ∀ x : ℝ, Real.sqrt x > x

-- Theorem to prove
theorem p_and_not_q_is_true : p ∧ ¬q := by
  sorry

end p_and_not_q_is_true_l2530_253060


namespace one_fifths_in_one_fourth_l2530_253037

theorem one_fifths_in_one_fourth : (1 : ℚ) / 4 / ((1 : ℚ) / 5) = 5 / 4 := by
  sorry

end one_fifths_in_one_fourth_l2530_253037


namespace parentheses_placement_l2530_253078

theorem parentheses_placement :
  let original := 0.5 + 0.5 / 0.5 + 0.5 / 0.5
  let with_parentheses := ((0.5 + 0.5) / 0.5 + 0.5) / 0.5
  with_parentheses = 5 ∧ with_parentheses ≠ original :=
by sorry

end parentheses_placement_l2530_253078


namespace no_integer_triangle_with_integer_altitudes_and_perimeter_1995_l2530_253053

theorem no_integer_triangle_with_integer_altitudes_and_perimeter_1995 :
  ¬ ∃ (a b c h_a h_b h_c : ℕ), 
    (a + b + c = 1995) ∧ 
    (h_a^2 * (4*a^2) = 2*a^2*b^2 + 2*a^2*c^2 + 2*c^2*b^2 - a^4 - b^4 - c^4) ∧
    (h_b^2 * (4*b^2) = 2*a^2*b^2 + 2*b^2*c^2 + 2*c^2*a^2 - a^4 - b^4 - c^4) ∧
    (h_c^2 * (4*c^2) = 2*a^2*c^2 + 2*b^2*c^2 + 2*a^2*b^2 - a^4 - b^4 - c^4) :=
by sorry


end no_integer_triangle_with_integer_altitudes_and_perimeter_1995_l2530_253053


namespace odd_heads_probability_not_simple_closed_form_l2530_253091

/-- Represents the probability of getting heads on the i-th flip -/
def p (i : ℕ) : ℚ := 3/4 - i/200

/-- Represents the probability of having an odd number of heads after n flips -/
noncomputable def P : ℕ → ℚ
  | 0 => 0
  | n + 1 => (1 - 2 * p n) * P n + p n

/-- The statement that the probability of odd number of heads after 100 flips
    cannot be expressed in a simple closed form -/
theorem odd_heads_probability_not_simple_closed_form :
  ∃ (f : ℚ → Prop), f (P 100) ∧ ∀ (x : ℚ), f x → x = P 100 :=
sorry

end odd_heads_probability_not_simple_closed_form_l2530_253091


namespace remainder_of_3_pow_2000_mod_13_l2530_253075

theorem remainder_of_3_pow_2000_mod_13 : (3^2000 : ℕ) % 13 = 9 := by sorry

end remainder_of_3_pow_2000_mod_13_l2530_253075


namespace line_equation_through_point_with_inclination_l2530_253021

/-- The equation of a line passing through (1, 2) with a 45° inclination angle -/
theorem line_equation_through_point_with_inclination (x y : ℝ) : 
  (x - y + 1 = 0) ↔ 
  (∃ (t : ℝ), x = 1 + t ∧ y = 2 + t) ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), x₁ - x₂ ≠ 0 → (y₁ - y₂) / (x₁ - x₂) = 1) :=
by sorry

end line_equation_through_point_with_inclination_l2530_253021
