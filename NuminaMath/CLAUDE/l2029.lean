import Mathlib

namespace NUMINAMATH_CALUDE_routes_in_grid_l2029_202923

/-- The number of routes in a 3x3 grid from top-left to bottom-right -/
def num_routes : ℕ := Nat.choose 6 3

/-- The dimensions of the grid -/
def grid_size : ℕ := 3

/-- The total number of moves required -/
def total_moves : ℕ := 2 * grid_size

/-- The number of moves in each direction -/
def moves_per_direction : ℕ := grid_size

theorem routes_in_grid :
  num_routes = Nat.choose total_moves moves_per_direction :=
sorry

end NUMINAMATH_CALUDE_routes_in_grid_l2029_202923


namespace NUMINAMATH_CALUDE_tv_cash_price_l2029_202934

def installment_plan_cost (down_payment : ℕ) (monthly_payment : ℕ) (num_months : ℕ) : ℕ :=
  down_payment + monthly_payment * num_months

def cash_price (total_installment_cost : ℕ) (savings : ℕ) : ℕ :=
  total_installment_cost - savings

theorem tv_cash_price :
  let down_payment : ℕ := 120
  let monthly_payment : ℕ := 30
  let num_months : ℕ := 12
  let savings : ℕ := 80
  let total_installment_cost : ℕ := installment_plan_cost down_payment monthly_payment num_months
  cash_price total_installment_cost savings = 400 := by
  sorry

end NUMINAMATH_CALUDE_tv_cash_price_l2029_202934


namespace NUMINAMATH_CALUDE_correct_calculation_l2029_202924

theorem correct_calculation (x : ℤ) (h : x + 238 = 637) : x - 382 = 17 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2029_202924


namespace NUMINAMATH_CALUDE_complex_equation_result_l2029_202937

theorem complex_equation_result (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : a + 2 * i = i * (b - i)) : 
  a - b = -3 := by sorry

end NUMINAMATH_CALUDE_complex_equation_result_l2029_202937


namespace NUMINAMATH_CALUDE_first_angle_is_55_l2029_202976

-- Define the triangle with the given conditions
def triangle (x : ℝ) : Prop :=
  let angle1 := x
  let angle2 := 2 * x
  let angle3 := x - 40
  (angle1 + angle2 + angle3 = 180) ∧ (angle1 > 0) ∧ (angle2 > 0) ∧ (angle3 > 0)

-- Theorem stating that the first angle is 55 degrees
theorem first_angle_is_55 : ∃ x, triangle x ∧ x = 55 := by
  sorry

end NUMINAMATH_CALUDE_first_angle_is_55_l2029_202976


namespace NUMINAMATH_CALUDE_sam_distance_l2029_202979

/-- Given that Harvey runs 8 miles more than Sam and their total distance is 32 miles,
    prove that Sam runs 12 miles. -/
theorem sam_distance (sam : ℝ) (harvey : ℝ) : 
  harvey = sam + 8 → sam + harvey = 32 → sam = 12 := by
  sorry

end NUMINAMATH_CALUDE_sam_distance_l2029_202979


namespace NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l2029_202945

/-- Proves that given a train with a speed of 90 kmph including stoppages
    and stopping for 15 minutes per hour, the speed of the train excluding
    stoppages is 120 kmph. -/
theorem train_speed_excluding_stoppages
  (speed_with_stoppages : ℝ)
  (stopping_time : ℝ)
  (h1 : speed_with_stoppages = 90)
  (h2 : stopping_time = 15/60) :
  let running_time := 1 - stopping_time
  speed_with_stoppages * 1 = speed_with_stoppages * running_time →
  speed_with_stoppages / running_time = 120 :=
by
  sorry

#check train_speed_excluding_stoppages

end NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l2029_202945


namespace NUMINAMATH_CALUDE_king_queen_ages_l2029_202902

theorem king_queen_ages : ∃ (K Q : ℕ),
  -- The king is twice as old as the queen was when the king was as old as the queen is now
  K = 2 * (Q - (K - Q)) ∧
  -- When the queen is as old as the king is now, their combined ages will be 63 years
  Q + (K - (K - Q)) + K = 63 ∧
  -- The king's age is 28 and the queen's age is 21
  K = 28 ∧ Q = 21 := by
sorry

end NUMINAMATH_CALUDE_king_queen_ages_l2029_202902


namespace NUMINAMATH_CALUDE_no_nontrivial_integer_solution_l2029_202903

theorem no_nontrivial_integer_solution :
  ∀ (a b c d : ℤ), a^2 - b = c^2 ∧ b^2 - a = d^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nontrivial_integer_solution_l2029_202903


namespace NUMINAMATH_CALUDE_total_water_consumption_in_week_l2029_202914

/-- Represents the water consumption of a sibling -/
structure WaterConsumption where
  weekday : ℕ
  weekend : ℕ

/-- Calculates the total water consumption for a sibling in a week -/
def weeklyConsumption (wc : WaterConsumption) : ℕ :=
  wc.weekday * 5 + wc.weekend * 2

/-- Theorem: Total water consumption of siblings in a week -/
theorem total_water_consumption_in_week (theo mason roxy zara lily : WaterConsumption)
  (h_theo : theo = { weekday := 8, weekend := 10 })
  (h_mason : mason = { weekday := 7, weekend := 8 })
  (h_roxy : roxy = { weekday := 9, weekend := 11 })
  (h_zara : zara = { weekday := 10, weekend := 12 })
  (h_lily : lily = { weekday := 6, weekend := 7 }) :
  weeklyConsumption theo + weeklyConsumption mason + weeklyConsumption roxy +
  weeklyConsumption zara + weeklyConsumption lily = 296 := by
  sorry


end NUMINAMATH_CALUDE_total_water_consumption_in_week_l2029_202914


namespace NUMINAMATH_CALUDE_lower_variance_more_stable_student_B_more_stable_l2029_202928

/-- Represents a student's throwing performance -/
structure StudentPerformance where
  name : String
  variance : ℝ

/-- Defines the concept of stability in performance -/
def moreStable (a b : StudentPerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two students' performances, the one with lower variance is more stable -/
theorem lower_variance_more_stable (a b : StudentPerformance) :
    moreStable a b ↔ a.variance < b.variance :=
  by sorry

/-- The specific problem instance -/
def studentA : StudentPerformance :=
  { name := "A", variance := 0.2 }

def studentB : StudentPerformance :=
  { name := "B", variance := 0.09 }

/-- Theorem: Student B has more stable performance than Student A -/
theorem student_B_more_stable : moreStable studentB studentA :=
  by sorry

end NUMINAMATH_CALUDE_lower_variance_more_stable_student_B_more_stable_l2029_202928


namespace NUMINAMATH_CALUDE_prob_2012_higher_than_2011_l2029_202967

/-- The probability of guessing the correct answer to a single question -/
def p : ℝ := 0.25

/-- The probability of guessing incorrectly -/
def q : ℝ := 1 - p

/-- Calculate the probability of passing the exam given the total number of questions and the minimum required correct answers -/
def prob_pass (n : ℕ) (k : ℕ) : ℝ :=
  1 - (Finset.sum (Finset.range k) (λ i => Nat.choose n i * p^i * q^(n - i)))

/-- The probability of passing the exam in 2011 -/
def prob_2011 : ℝ := prob_pass 20 3

/-- The probability of passing the exam in 2012 -/
def prob_2012 : ℝ := prob_pass 40 6

/-- Theorem stating that the probability of passing in 2012 is higher than in 2011 -/
theorem prob_2012_higher_than_2011 : prob_2012 > prob_2011 := by
  sorry

end NUMINAMATH_CALUDE_prob_2012_higher_than_2011_l2029_202967


namespace NUMINAMATH_CALUDE_triangle_problem_l2029_202948

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
prove that if 2b*sin(B) = (2a+c)*sin(A) + (2c+a)*sin(C), b = √3, and A = π/4,
then B = 2π/3 and the area of the triangle is (3 - √3)/4.
-/
theorem triangle_problem (a b c A B C : ℝ) : 
  2 * b * Real.sin B = (2 * a + c) * Real.sin A + (2 * c + a) * Real.sin C →
  b = Real.sqrt 3 →
  A = π / 4 →
  B = 2 * π / 3 ∧ 
  (1 / 2 : ℝ) * b * c * Real.sin A = (3 - Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2029_202948


namespace NUMINAMATH_CALUDE_tangent_point_segment_difference_l2029_202994

/-- A cyclic quadrilateral with an inscribed circle -/
structure CyclicQuadrilateral where
  /-- The lengths of the four sides of the quadrilateral -/
  sides : Fin 4 → ℝ
  /-- The radius of the inscribed circle -/
  inradius : ℝ
  /-- The semiperimeter of the quadrilateral -/
  semiperimeter : ℝ
  /-- The area of the quadrilateral -/
  area : ℝ

/-- The theorem about the difference of segments created by the point of tangency -/
theorem tangent_point_segment_difference
  (Q : CyclicQuadrilateral)
  (h1 : Q.sides 0 = 80)
  (h2 : Q.sides 1 = 100)
  (h3 : Q.sides 2 = 140)
  (h4 : Q.sides 3 = 120)
  (h5 : Q.semiperimeter = (Q.sides 0 + Q.sides 1 + Q.sides 2 + Q.sides 3) / 2)
  (h6 : Q.area = Real.sqrt ((Q.semiperimeter - Q.sides 0) *
                            (Q.semiperimeter - Q.sides 1) *
                            (Q.semiperimeter - Q.sides 2) *
                            (Q.semiperimeter - Q.sides 3)))
  (h7 : Q.inradius * Q.semiperimeter = Q.area) :
  ∃ (x y : ℝ), x + y = 140 ∧ |x - y| = 5 := by
  sorry


end NUMINAMATH_CALUDE_tangent_point_segment_difference_l2029_202994


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2029_202974

def geometric_sequence (a : ℕ → ℤ) (q : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℤ) (q : ℤ) :
  geometric_sequence a q ∧ a 1 = 1 ∧ q = -2 →
  a 1 + |a 2| + a 3 + |a 4| = 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2029_202974


namespace NUMINAMATH_CALUDE_root_modulus_preservation_l2029_202952

theorem root_modulus_preservation (a b c : ℂ) :
  (∀ z : ℂ, z^3 + a*z^2 + b*z + c = 0 → Complex.abs z = 1) →
  (∀ z : ℂ, z^3 + Complex.abs a*z^2 + Complex.abs b*z + Complex.abs c = 0 → Complex.abs z = 1) :=
by sorry

end NUMINAMATH_CALUDE_root_modulus_preservation_l2029_202952


namespace NUMINAMATH_CALUDE_cube_root_product_equals_48_l2029_202942

theorem cube_root_product_equals_48 : 
  (64 : ℝ) ^ (1/3) * (27 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 48 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_product_equals_48_l2029_202942


namespace NUMINAMATH_CALUDE_men_in_first_group_l2029_202999

/-- Represents the daily work done by a boy -/
def boy_work : ℝ := 1

/-- Represents the daily work done by a man -/
def man_work : ℝ := 2 * boy_work

/-- The number of days taken by the first group to complete the work -/
def days_group1 : ℕ := 5

/-- The number of days taken by the second group to complete the work -/
def days_group2 : ℕ := 4

/-- The number of boys in the first group -/
def boys_group1 : ℕ := 16

/-- The number of men in the second group -/
def men_group2 : ℕ := 13

/-- The number of boys in the second group -/
def boys_group2 : ℕ := 24

/-- The theorem stating that the number of men in the first group is 12 -/
theorem men_in_first_group :
  ∃ (m : ℕ), 
    (days_group1 : ℝ) * (m * man_work + boys_group1 * boy_work) = 
    (days_group2 : ℝ) * (men_group2 * man_work + boys_group2 * boy_work) ∧
    m = 12 := by
  sorry

end NUMINAMATH_CALUDE_men_in_first_group_l2029_202999


namespace NUMINAMATH_CALUDE_expand_polynomial_l2029_202950

theorem expand_polynomial (x : ℝ) : (4 * x + 3) * (2 * x - 7) + x = 8 * x^2 - 21 * x - 21 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l2029_202950


namespace NUMINAMATH_CALUDE_rectangle_area_l2029_202919

/-- Given a rectangle with diagonal length x and length three times its width, 
    prove that its area is (3/10)x^2 -/
theorem rectangle_area (x : ℝ) (h : x > 0) : ∃ w : ℝ, 
  w > 0 ∧ 
  x^2 = (3*w)^2 + w^2 ∧ 
  (3*w) * w = (3/10) * x^2 :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2029_202919


namespace NUMINAMATH_CALUDE_unique_f_3_l2029_202922

/-- A function satisfying the given functional equation and initial condition -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 2 * (f x) * y) ∧ f 1 = 1

/-- The theorem stating that for any function satisfying the conditions, f(3) must equal 9 -/
theorem unique_f_3 (f : ℝ → ℝ) (hf : special_function f) : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_f_3_l2029_202922


namespace NUMINAMATH_CALUDE_great_eighteen_league_games_l2029_202975

/-- Calculates the number of games in a soccer league with specified structure -/
def soccer_league_games (divisions : Nat) (teams_per_division : Nat) 
  (intra_division_games : Nat) (inter_division_games : Nat) : Nat :=
  let intra_games := divisions * (teams_per_division.choose 2) * intra_division_games
  let inter_games := divisions.choose 2 * teams_per_division^2 * inter_division_games
  intra_games + inter_games

/-- The Great Eighteen Soccer League game count theorem -/
theorem great_eighteen_league_games : 
  soccer_league_games 3 6 3 2 = 351 := by
  sorry

end NUMINAMATH_CALUDE_great_eighteen_league_games_l2029_202975


namespace NUMINAMATH_CALUDE_james_stickers_l2029_202965

theorem james_stickers (initial_stickers new_stickers total_stickers : ℕ) 
  (h1 : new_stickers = 22)
  (h2 : total_stickers = 61)
  (h3 : total_stickers = initial_stickers + new_stickers) : 
  initial_stickers = 39 := by
  sorry

end NUMINAMATH_CALUDE_james_stickers_l2029_202965


namespace NUMINAMATH_CALUDE_frank_uniform_number_l2029_202917

def is_two_digit_prime (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem frank_uniform_number 
  (d e f : ℕ) 
  (h1 : is_two_digit_prime d) 
  (h2 : is_two_digit_prime e) 
  (h3 : is_two_digit_prime f) 
  (h4 : d + f = 28) 
  (h5 : d + e = 24) 
  (h6 : e + f = 30) : 
  f = 17 := by
sorry

end NUMINAMATH_CALUDE_frank_uniform_number_l2029_202917


namespace NUMINAMATH_CALUDE_sum_of_x_intercepts_is_14_l2029_202990

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := (x - 3)^2 + 4

-- Define the transformed parabola
def transformed_parabola (x : ℝ) : ℝ := -(x - 7)^2 + 1

-- Theorem statement
theorem sum_of_x_intercepts_is_14 :
  ∃ a b : ℝ, 
    transformed_parabola a = 0 ∧ 
    transformed_parabola b = 0 ∧ 
    a + b = 14 :=
sorry

end NUMINAMATH_CALUDE_sum_of_x_intercepts_is_14_l2029_202990


namespace NUMINAMATH_CALUDE_symmetry_preserves_circle_l2029_202930

/-- A circle in R^2 -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in R^2 of the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- The given circle (x-1)^2 + (y-1)^2 = 1 -/
def given_circle : Circle := { center := (1, 1), radius := 1 }

/-- The given line y = 5x - 4 -/
def given_line : Line := { m := 5, b := -4 }

/-- Predicate to check if a point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop :=
  p.2 = l.m * p.1 + l.b

/-- The symmetrical circle with respect to a line -/
def symmetrical_circle (c : Circle) (l : Line) : Circle :=
  sorry -- Definition of symmetrical circle

theorem symmetry_preserves_circle (c : Circle) (l : Line) :
  point_on_line c.center l →
  symmetrical_circle c l = c := by
  sorry

end NUMINAMATH_CALUDE_symmetry_preserves_circle_l2029_202930


namespace NUMINAMATH_CALUDE_gcd_of_large_numbers_l2029_202918

theorem gcd_of_large_numbers : Nat.gcd 1000000000 1000000005 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_large_numbers_l2029_202918


namespace NUMINAMATH_CALUDE_inverse_function_property_l2029_202933

-- Define a function f with an inverse
def f_has_inverse (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- State the theorem
theorem inverse_function_property
  (f : ℝ → ℝ)
  (h_inverse : f_has_inverse f)
  (h_point : f 2 = -1) :
  ∃ f_inv : ℝ → ℝ, f_inv (-1) = 2 ∧ (∀ x, f_inv (f x) = x) ∧ (∀ y, f (f_inv y) = y) :=
sorry

end NUMINAMATH_CALUDE_inverse_function_property_l2029_202933


namespace NUMINAMATH_CALUDE_jan_claims_l2029_202907

/-- The number of claims each agent can handle --/
structure AgentClaims where
  missy : ℕ
  john : ℕ
  jan : ℕ

/-- Conditions for the insurance claims problem --/
def insurance_claims_conditions (claims : AgentClaims) : Prop :=
  claims.missy = 41 ∧
  claims.missy = claims.john + 15 ∧
  claims.john = claims.jan + (claims.jan / 10) * 3

/-- Theorem stating that under the given conditions, Jan can handle 20 claims --/
theorem jan_claims (claims : AgentClaims) 
  (h : insurance_claims_conditions claims) : claims.jan = 20 := by
  sorry


end NUMINAMATH_CALUDE_jan_claims_l2029_202907


namespace NUMINAMATH_CALUDE_interest_calculation_l2029_202954

def deposit : ℝ := 30000
def term : ℝ := 3
def interest_rate : ℝ := 0.047
def tax_rate : ℝ := 0.2

def pre_tax_interest : ℝ := deposit * interest_rate * term
def after_tax_interest : ℝ := pre_tax_interest * (1 - tax_rate)
def total_withdrawal : ℝ := deposit + after_tax_interest

theorem interest_calculation :
  after_tax_interest = 3372 ∧ total_withdrawal = 33372 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l2029_202954


namespace NUMINAMATH_CALUDE_ellipse_properties_l2029_202941

-- Define the ellipse C
def Ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the condition for points A and B
def SlopeCondition (xA yA xB yB : ℝ) : Prop :=
  xA ≠ 0 ∧ xB ≠ 0 ∧ (yA / xA) * (yB / xB) = -3/2

theorem ellipse_properties :
  -- Given conditions
  let D := (0, 2)
  let F := (c, 0)
  let E := (4*c/3, -2/3)
  -- The ellipse passes through D and E
  Ellipse D.1 D.2 ∧ Ellipse E.1 E.2 →
  -- |DF| = 3|EF|
  (D.1 - F.1)^2 + (D.2 - F.2)^2 = 9 * ((E.1 - F.1)^2 + (E.2 - F.2)^2) →
  -- Theorem statements
  (∀ x y, Ellipse x y ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  (∀ xA yA xB yB,
    Ellipse xA yA → Ellipse xB yB → SlopeCondition xA yA xB yB →
    -1 ≤ (xA * xB + yA * yB) ∧ (xA * xB + yA * yB) ≤ 1 ∧
    (xA * xB + yA * yB) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2029_202941


namespace NUMINAMATH_CALUDE_uncommon_roots_product_l2029_202940

def P (x : ℝ) : ℝ := x^4 + 2*x^3 - 8*x^2 - 6*x + 15
def Q (x : ℝ) : ℝ := x^3 + 4*x^2 - x - 10

theorem uncommon_roots_product : 
  ∃ (r₁ r₂ : ℝ), 
    P r₁ = 0 ∧ 
    Q r₂ = 0 ∧ 
    r₁ ≠ r₂ ∧
    (∀ x : ℝ, (P x = 0 ∧ Q x ≠ 0) ∨ (Q x = 0 ∧ P x ≠ 0) → x = r₁ ∨ x = r₂) ∧
    r₁ * r₂ = -2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_uncommon_roots_product_l2029_202940


namespace NUMINAMATH_CALUDE_sum_odd_integers_7_to_35_l2029_202908

/-- The sum of odd integers from 7 to 35 (inclusive) is 315 -/
theorem sum_odd_integers_7_to_35 : 
  (Finset.range 15).sum (fun i => 2 * i + 7) = 315 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_integers_7_to_35_l2029_202908


namespace NUMINAMATH_CALUDE_product_square_theorem_l2029_202989

theorem product_square_theorem : (10 * 0.2 * 3 * 0.1)^2 = 9/25 := by
  sorry

end NUMINAMATH_CALUDE_product_square_theorem_l2029_202989


namespace NUMINAMATH_CALUDE_quadratic_real_root_l2029_202982

theorem quadratic_real_root (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l2029_202982


namespace NUMINAMATH_CALUDE_coefficient_corresponds_to_20th_term_l2029_202911

/-- The general term of the arithmetic sequence -/
def a (n : ℕ) : ℤ := 3 * n - 5

/-- The coefficient of x^4 in the expansion of (1+x)^k -/
def coeff (k : ℕ) : ℕ := Nat.choose k 4

/-- The theorem stating that the 20th term of the sequence corresponds to
    the coefficient of x^4 in the given expansion -/
theorem coefficient_corresponds_to_20th_term :
  a 20 = (coeff 5 + coeff 6 + coeff 7) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_corresponds_to_20th_term_l2029_202911


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2029_202993

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  1 / a + 2 / b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2 * b₀ = 1 ∧ 1 / a₀ + 2 / b₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2029_202993


namespace NUMINAMATH_CALUDE_max_p_value_l2029_202992

-- Define the equation
def equation (x p : ℝ) : Prop :=
  2 * Real.cos (2 * Real.pi - Real.pi * x^2 / 6) * Real.cos (Real.pi / 3 * Real.sqrt (9 - x^2)) - 3 =
  p - 2 * Real.sin (-Real.pi * x^2 / 6) * Real.cos (Real.pi / 3 * Real.sqrt (9 - x^2))

-- Define the theorem
theorem max_p_value :
  ∃ (p_max : ℝ), p_max = -2 ∧
  (∀ p : ℝ, (∃ x : ℝ, equation x p) → p ≤ p_max) ∧
  (∃ x : ℝ, equation x p_max) :=
sorry

end NUMINAMATH_CALUDE_max_p_value_l2029_202992


namespace NUMINAMATH_CALUDE_sum_of_possible_m_values_l2029_202984

theorem sum_of_possible_m_values (p q r m : ℂ) : 
  p ≠ q ∧ q ≠ r ∧ r ≠ p →
  p / (1 - q) = m ∧ q / (1 - r) = m ∧ r / (1 - p) = m →
  ∃ (m₁ m₂ m₃ : ℂ), 
    (m₁ = 0 ∨ m₁ = (1 + Complex.I * Real.sqrt 3) / 2 ∨ m₁ = (1 - Complex.I * Real.sqrt 3) / 2) ∧
    (m₂ = 0 ∨ m₂ = (1 + Complex.I * Real.sqrt 3) / 2 ∨ m₂ = (1 - Complex.I * Real.sqrt 3) / 2) ∧
    (m₃ = 0 ∨ m₃ = (1 + Complex.I * Real.sqrt 3) / 2 ∨ m₃ = (1 - Complex.I * Real.sqrt 3) / 2) ∧
    m₁ + m₂ + m₃ = 1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_possible_m_values_l2029_202984


namespace NUMINAMATH_CALUDE_intersection_is_sinusoid_l2029_202943

/-- Represents a cylinder with radius R and height H -/
structure Cylinder where
  R : ℝ
  H : ℝ

/-- Represents the inclined plane intersecting the cylinder -/
structure InclinedPlane where
  α : ℝ  -- Angle of inclination

/-- Represents a point on the unfolded lateral surface of the cylinder -/
structure UnfoldedPoint where
  x : ℝ  -- Horizontal distance along unwrapped cylinder
  z : ℝ  -- Vertical distance

/-- The equation of the intersection line on the unfolded surface -/
def intersectionLine (c : Cylinder) (p : InclinedPlane) (point : UnfoldedPoint) : Prop :=
  ∃ (A z₀ : ℝ), point.z = A * Real.sin (point.x / c.R) + z₀

/-- Theorem stating that the intersection line is sinusoidal -/
theorem intersection_is_sinusoid (c : Cylinder) (p : InclinedPlane) :
  ∀ point : UnfoldedPoint, intersectionLine c p point := by
  sorry

end NUMINAMATH_CALUDE_intersection_is_sinusoid_l2029_202943


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2029_202909

theorem expand_and_simplify (x : ℝ) : (7 * x - 3) * 3 * x^2 = 21 * x^3 - 9 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2029_202909


namespace NUMINAMATH_CALUDE_game_result_l2029_202995

theorem game_result (a : ℝ) : ((2 * a + 6) / 2) - a = 3 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l2029_202995


namespace NUMINAMATH_CALUDE_sons_age_l2029_202983

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l2029_202983


namespace NUMINAMATH_CALUDE_divisibility_implies_one_or_seven_l2029_202959

theorem divisibility_implies_one_or_seven (a n : ℤ) 
  (ha : a ≥ 1) 
  (h1 : a ∣ n + 2) 
  (h2 : a ∣ n^2 + n + 5) : 
  a = 1 ∨ a = 7 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_one_or_seven_l2029_202959


namespace NUMINAMATH_CALUDE_A_intersect_B_empty_l2029_202997

-- Define set A
def A : Set ℝ := {0, 1, 2}

-- Define set B
def B : Set ℝ := {x : ℝ | (x + 1) * (x + 2) < 0}

-- Theorem statement
theorem A_intersect_B_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_empty_l2029_202997


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l2029_202966

-- 1
theorem problem_1 : 4 - (-28) + (-2) = 30 := by sorry

-- 2
theorem problem_2 : (-3) * ((-2/5) / (-1/4)) = -24/5 := by sorry

-- 3
theorem problem_3 : (-42) / (-7) - (-6) * 4 = 30 := by sorry

-- 4
theorem problem_4 : -3^2 / (-3)^2 + 3 * (-2) + |(-4)| = -3 := by sorry

-- 5
theorem problem_5 : (-24) * (3/4 - 5/6 + 7/12) = -12 := by sorry

-- 6
theorem problem_6 : -1^4 - (1 - 0.5) / (5/2) * (1/5) = -26/25 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l2029_202966


namespace NUMINAMATH_CALUDE_gp_ratio_is_four_l2029_202960

theorem gp_ratio_is_four (x : ℝ) :
  (∀ r : ℝ, (40 + x) = (10 + x) * r ∧ (160 + x) = (40 + x) * r) →
  r = 4 :=
by sorry

end NUMINAMATH_CALUDE_gp_ratio_is_four_l2029_202960


namespace NUMINAMATH_CALUDE_forest_coverage_growth_rate_l2029_202926

theorem forest_coverage_growth_rate (x : ℝ) : 
  (0.63 * (1 + x)^2 = 0.68) ↔ 
  (∃ (rate : ℝ → ℝ), 
    rate 0 = 0.63 ∧ 
    rate 2 = 0.68 ∧ 
    ∀ t, 0 ≤ t → t ≤ 2 → rate t = 0.63 * (1 + x)^t) :=
sorry

end NUMINAMATH_CALUDE_forest_coverage_growth_rate_l2029_202926


namespace NUMINAMATH_CALUDE_jason_potato_eating_time_l2029_202961

/-- Given that Jason eats 3 potatoes in 20 minutes, prove that it takes him 3 hours to eat 27 potatoes. -/
theorem jason_potato_eating_time :
  let potatoes_per_20_min : ℚ := 3
  let total_potatoes : ℚ := 27
  let minutes_per_session : ℚ := 20
  let hours_to_eat_all : ℚ := (total_potatoes / potatoes_per_20_min) * (minutes_per_session / 60)
  hours_to_eat_all = 3 := by
sorry

end NUMINAMATH_CALUDE_jason_potato_eating_time_l2029_202961


namespace NUMINAMATH_CALUDE_intersection_points_product_l2029_202912

-- Define the curve C in Cartesian coordinates
def curve_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line l
def line_l (x y m : ℝ) : Prop := x - Real.sqrt 3 * y - m = 0

-- Define the intersection condition
def intersects (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    line_l x₁ y₁ m ∧ line_l x₂ y₂ m ∧
    (x₁ - m)^2 + y₁^2 * (x₂ - m)^2 + y₂^2 = 1

-- Theorem statement
theorem intersection_points_product (m : ℝ) :
  intersects m ↔ m = 1 ∨ m = 1 + Real.sqrt 2 ∨ m = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_product_l2029_202912


namespace NUMINAMATH_CALUDE_max_diff_slightly_unlucky_l2029_202956

/-- A natural number is slightly unlucky if the sum of its digits in decimal system is divisible by 13. -/
def SlightlyUnlucky (n : ℕ) : Prop :=
  (n.digits 10).sum % 13 = 0

/-- For any non-negative integer k, the intervals [100(k+1), 100(k+1)+39], [100k+60, 100k+99], and [100k+20, 100k+59] each contain at least one slightly unlucky number. -/
axiom slightly_unlucky_intervals (k : ℕ) :
  (∃ n : ℕ, SlightlyUnlucky n ∧ 100*(k+1) ≤ n ∧ n ≤ 100*(k+1)+39) ∧
  (∃ n : ℕ, SlightlyUnlucky n ∧ 100*k+60 ≤ n ∧ n ≤ 100*k+99) ∧
  (∃ n : ℕ, SlightlyUnlucky n ∧ 100*k+20 ≤ n ∧ n ≤ 100*k+59)

/-- The maximum difference between consecutive slightly unlucky numbers is 79. -/
theorem max_diff_slightly_unlucky :
  ∀ m n : ℕ, SlightlyUnlucky m → SlightlyUnlucky n → m < n →
  (∀ k : ℕ, SlightlyUnlucky k → m < k → k < n → False) →
  n - m ≤ 79 :=
sorry

end NUMINAMATH_CALUDE_max_diff_slightly_unlucky_l2029_202956


namespace NUMINAMATH_CALUDE_range_of_values_l2029_202958

theorem range_of_values (x y : ℝ) 
  (hx : 30 < x ∧ x < 42) 
  (hy : 16 < y ∧ y < 24) : 
  (46 < x + y ∧ x + y < 66) ∧ 
  (-18 < x - 2*y ∧ x - 2*y < 10) ∧ 
  (5/4 < x/y ∧ x/y < 21/8) := by
sorry

end NUMINAMATH_CALUDE_range_of_values_l2029_202958


namespace NUMINAMATH_CALUDE_wrong_height_calculation_l2029_202973

theorem wrong_height_calculation (n : ℕ) (initial_avg : ℝ) (actual_height : ℝ) (actual_avg : ℝ) :
  n = 35 ∧ initial_avg = 184 ∧ actual_height = 106 ∧ actual_avg = 182 →
  ∃ wrong_height : ℝ, wrong_height = 176 ∧
    n * actual_avg = n * initial_avg - wrong_height + actual_height :=
by sorry

end NUMINAMATH_CALUDE_wrong_height_calculation_l2029_202973


namespace NUMINAMATH_CALUDE_max_value_xy_x_minus_y_l2029_202936

theorem max_value_xy_x_minus_y (x y : ℝ) (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) :
  x * y * (x - y) ≤ (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xy_x_minus_y_l2029_202936


namespace NUMINAMATH_CALUDE_distinct_sums_count_l2029_202905

def bag_A : Finset Nat := {1, 3, 5, 7}
def bag_B : Finset Nat := {2, 4, 6, 8}

def possible_sums : Finset Nat :=
  Finset.image (λ (pair : Nat × Nat) => pair.1 + pair.2) (bag_A.product bag_B)

theorem distinct_sums_count : Finset.card possible_sums = 7 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sums_count_l2029_202905


namespace NUMINAMATH_CALUDE_three_std_dev_below_mean_undetermined_l2029_202906

/-- Represents a non-normal probability distribution --/
structure NonNormalDistribution where
  mean : ℝ
  std_dev : ℝ
  skewness : ℝ
  kurtosis : ℝ
  is_non_normal : Bool

/-- The value that is exactly 3 standard deviations less than the mean --/
def three_std_dev_below_mean (d : NonNormalDistribution) : ℝ := sorry

/-- Theorem stating that the value 3 standard deviations below the mean cannot be determined
    for a non-normal distribution without additional information --/
theorem three_std_dev_below_mean_undetermined
  (d : NonNormalDistribution)
  (h_mean : d.mean = 15)
  (h_std_dev : d.std_dev = 1.5)
  (h_skewness : d.skewness = 0.5)
  (h_kurtosis : d.kurtosis = 0.6)
  (h_non_normal : d.is_non_normal = true) :
  ¬ ∃ (x : ℝ), three_std_dev_below_mean d = x :=
sorry

end NUMINAMATH_CALUDE_three_std_dev_below_mean_undetermined_l2029_202906


namespace NUMINAMATH_CALUDE_contrapositive_correct_l2029_202978

-- Define the proposition p
def p (passing_score : ℝ) (A_passes B_passes C_passes : Prop) : Prop :=
  (passing_score < 70) → (¬A_passes ∧ ¬B_passes ∧ ¬C_passes)

-- Define the contrapositive of p
def contrapositive_p (passing_score : ℝ) (A_passes B_passes C_passes : Prop) : Prop :=
  (A_passes ∨ B_passes ∨ C_passes) → (passing_score ≥ 70)

-- Theorem stating that contrapositive_p is indeed the contrapositive of p
theorem contrapositive_correct (passing_score : ℝ) (A_passes B_passes C_passes : Prop) :
  contrapositive_p passing_score A_passes B_passes C_passes ↔
  (¬p passing_score A_passes B_passes C_passes → False) → False :=
sorry

end NUMINAMATH_CALUDE_contrapositive_correct_l2029_202978


namespace NUMINAMATH_CALUDE_acid_dilution_l2029_202985

theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (final_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 50 →
  initial_concentration = 0.4 →
  final_concentration = 0.25 →
  water_added = 30 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

end NUMINAMATH_CALUDE_acid_dilution_l2029_202985


namespace NUMINAMATH_CALUDE_subtract_from_twenty_l2029_202972

theorem subtract_from_twenty (x : ℤ) (h : x + 40 = 52) : 20 - x = 8 := by
  sorry

end NUMINAMATH_CALUDE_subtract_from_twenty_l2029_202972


namespace NUMINAMATH_CALUDE_sam_remaining_money_l2029_202949

/-- Calculates the remaining money in cents after Sam's purchases -/
def remaining_money (initial_dimes : Nat) (initial_quarters : Nat) 
  (candy_bars : Nat) (candy_bar_cost : Nat) (lollipop_cost : Nat) : Nat :=
  let remaining_dimes := initial_dimes - candy_bars * candy_bar_cost
  let remaining_quarters := initial_quarters - 1
  remaining_dimes * 10 + remaining_quarters * 25

/-- Proves that Sam has 195 cents left after her purchases -/
theorem sam_remaining_money :
  remaining_money 19 6 4 3 1 = 195 := by
  sorry

end NUMINAMATH_CALUDE_sam_remaining_money_l2029_202949


namespace NUMINAMATH_CALUDE_move_down_two_units_l2029_202968

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moving a point down in a Cartesian coordinate system -/
def moveDown (p : Point) (distance : ℝ) : Point :=
  ⟨p.x, p.y - distance⟩

/-- Theorem: Moving a point (a,b) down 2 units results in (a,b-2) -/
theorem move_down_two_units (a b : ℝ) :
  moveDown ⟨a, b⟩ 2 = ⟨a, b - 2⟩ := by
  sorry

end NUMINAMATH_CALUDE_move_down_two_units_l2029_202968


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_897_l2029_202916

theorem largest_prime_factor_of_897 : ∃ (p : ℕ), Prime p ∧ p ∣ 897 ∧ ∀ (q : ℕ), Prime q → q ∣ 897 → q ≤ p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_897_l2029_202916


namespace NUMINAMATH_CALUDE_range_of_a_l2029_202931

-- Define the propositions p and q
def p (m a : ℝ) : Prop := m^2 - 7*m*a + 12*a^2 < 0

def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m - 1) + y^2 / (2 - m) = 1 ∧ 
  ∃ (c : ℝ), c > 0 ∧ x^2 / (m - 1) + y^2 / (2 - m - c) = 1

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (a > 0) → 
  (∀ m : ℝ, (¬(p m a) → ¬(q m)) ∧ ∃ m : ℝ, ¬(p m a) ∧ q m) →
  a ∈ Set.Icc (1/3 : ℝ) (3/8 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2029_202931


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l2029_202998

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) : 
  train_length = 160 → 
  train_speed_kmh = 45 → 
  bridge_length = 215 → 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l2029_202998


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2029_202969

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x y : ℝ), (x - 2)^2 + y^2 = 3 ∧ 
    (∃ (k : ℝ), b * x + a * y = k ∨ b * x - a * y = k)) →
  (a^2 = 1 ∧ b^2 = 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2029_202969


namespace NUMINAMATH_CALUDE_stripe_area_on_cylinder_l2029_202964

/-- The area of a stripe wrapped around a cylinder -/
theorem stripe_area_on_cylinder 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℝ) 
  (h1 : diameter = 20) 
  (h2 : stripe_width = 2) 
  (h3 : revolutions = 3) : 
  stripe_width * revolutions * (π * diameter) = 240 * π := by
  sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylinder_l2029_202964


namespace NUMINAMATH_CALUDE_ellipse_equation_l2029_202957

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C : Set (ℝ × ℝ) := {(x, y) | x^2 / a^2 + y^2 / b^2 = 1}
  let e : ℝ := 1/3
  let A₁ : ℝ × ℝ := (-a, 0)
  let A₂ : ℝ × ℝ := (a, 0)
  let B : ℝ × ℝ := (0, b)
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 / a^2 + y^2 / b^2 = 1) →
  e = Real.sqrt (1 - b^2 / a^2) →
  ((B.1 - A₁.1) * (B.1 - A₂.1) + (B.2 - A₁.2) * (B.2 - A₂.2) = -1) →
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 / 9 + y^2 / 8 = 1) := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2029_202957


namespace NUMINAMATH_CALUDE_flowers_left_l2029_202944

theorem flowers_left (alissa_flowers melissa_flowers given_flowers : ℕ) :
  alissa_flowers = 16 →
  melissa_flowers = 16 →
  given_flowers = 18 →
  alissa_flowers + melissa_flowers - given_flowers = 14 := by
sorry

end NUMINAMATH_CALUDE_flowers_left_l2029_202944


namespace NUMINAMATH_CALUDE_k_range_k_trapezoid_l2029_202946

-- Define the circles and lines
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 2
def circle_N (x y : ℝ) : Prop := x^2 + (y - 8)^2 = 40

def line_l1 (k x y : ℝ) : Prop := y = k * x
def line_l2 (k x y : ℝ) : Prop := y = -1/k * x

-- Define the intersection points
def point_A (k : ℝ) : ℝ × ℝ := sorry
def point_B (k : ℝ) : ℝ × ℝ := sorry
def point_C (k : ℝ) : ℝ × ℝ := sorry
def point_D (k : ℝ) : ℝ × ℝ := sorry

-- Define the conditions
def conditions (k : ℝ) : Prop :=
  ∃ (A B C D : ℝ × ℝ),
    A ≠ B ∧ C ≠ D ∧
    circle_M A.1 A.2 ∧ circle_M B.1 B.2 ∧
    circle_N C.1 C.2 ∧ circle_N D.1 D.2 ∧
    line_l1 k A.1 A.2 ∧ line_l1 k B.1 B.2 ∧
    line_l2 k C.1 C.2 ∧ line_l2 k D.1 D.2

-- Define the trapezoid condition
def is_trapezoid (A B C D : ℝ × ℝ) : Prop := sorry

-- Theorem for the range of k
theorem k_range (k : ℝ) (h : conditions k) : 
  2 - Real.sqrt 3 < k ∧ k < Real.sqrt 15 / 3 := by sorry

-- Theorem for k when ABCD is a trapezoid
theorem k_trapezoid (k : ℝ) (h : conditions k) :
  (∃ (A B C D : ℝ × ℝ), is_trapezoid A B C D) → k = 1 := by sorry

end NUMINAMATH_CALUDE_k_range_k_trapezoid_l2029_202946


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2029_202996

/-- A quadratic function satisfying certain conditions -/
def f (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The function g defined in terms of f and m -/
def g (a b c m : ℝ) : ℝ → ℝ := fun x ↦ f a b c x + 2 * (1 - m) * x

/-- The theorem statement -/
theorem quadratic_function_theorem (a b c : ℝ) :
  (∀ x, f a b c x ≥ 0) →
  f a b c 0 = 1 →
  f a b c 1 = 0 →
  (∃ m, (∀ x ∈ Set.Icc (-2 : ℝ) 5, g a b c m x ≤ 13) ∧
        (∃ x ∈ Set.Icc (-2 : ℝ) 5, g a b c m x = 13)) →
  ((a = 1 ∧ b = -2 ∧ c = 1) ∧ (m = 13/10 ∨ m = 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2029_202996


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_is_two_l2029_202970

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- An asymptote of a hyperbola -/
def asymptote (h : Hyperbola a b) : ℝ → ℝ := sorry

/-- The symmetric point of a point with respect to a line -/
def symmetric_point (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ × ℝ := sorry

/-- Theorem: The eccentricity of a hyperbola is 2 given the specified conditions -/
theorem hyperbola_eccentricity_is_two (a b : ℝ) (h : Hyperbola a b) :
  let l₁ := asymptote h
  let l₂ := fun x => -l₁ x
  let f := right_focus h
  let s := symmetric_point f l₁
  s.2 = l₂ s.1 →
  eccentricity h = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_is_two_l2029_202970


namespace NUMINAMATH_CALUDE_taxi_problem_l2029_202925

/-- Represents the direction of travel -/
inductive Direction
| East
| West

/-- Represents a single trip -/
structure Trip where
  distance : ℝ
  direction : Direction

def trips : List Trip := [
  ⟨8, Direction.East⟩, ⟨6, Direction.West⟩, ⟨3, Direction.East⟩,
  ⟨7, Direction.West⟩, ⟨8, Direction.East⟩, ⟨4, Direction.East⟩,
  ⟨9, Direction.West⟩, ⟨4, Direction.West⟩, ⟨3, Direction.East⟩,
  ⟨3, Direction.East⟩
]

def totalTime : ℝ := 1.25 -- in hours

def startingFare : ℝ := 8
def additionalFarePerKm : ℝ := 2
def freeDistance : ℝ := 3

theorem taxi_problem (trips : List Trip) (totalTime : ℝ) 
    (startingFare additionalFarePerKm freeDistance : ℝ) :
  -- 1. Final position is 3 km east
  (trips.foldl (fun acc trip => 
    match trip.direction with
    | Direction.East => acc + trip.distance
    | Direction.West => acc - trip.distance
  ) 0 = 3) ∧
  -- 2. Average speed is 44 km/h
  ((trips.foldl (fun acc trip => acc + trip.distance) 0) / totalTime = 44) ∧
  -- 3. Total earnings are 130 yuan
  (trips.length * startingFare + 
   (trips.foldl (fun acc trip => acc + max (trip.distance - freeDistance) 0) 0) * additionalFarePerKm = 130) := by
  sorry

end NUMINAMATH_CALUDE_taxi_problem_l2029_202925


namespace NUMINAMATH_CALUDE_ten_percent_of_x_l2029_202900

theorem ten_percent_of_x (x c : ℝ) : 
  3 - (1/4)*2 - (1/3)*3 - (1/7)*x = c → 
  (10/100) * x = 0.7 * (1.5 - c) := by
sorry

end NUMINAMATH_CALUDE_ten_percent_of_x_l2029_202900


namespace NUMINAMATH_CALUDE_equation_solution_l2029_202910

theorem equation_solution : 
  ∃! x : ℚ, (x - 17) / 3 = (3 * x + 8) / 6 :=
by
  use (-42 : ℚ)
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2029_202910


namespace NUMINAMATH_CALUDE_expression_equality_l2029_202920

theorem expression_equality (x y : ℝ) (h : x^2 + y^2 = 1) :
  2*x^4 + 3*x^2*y^2 + y^4 + y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2029_202920


namespace NUMINAMATH_CALUDE_f_2005_of_2006_eq_145_l2029_202929

/-- Sum of squares of digits of a positive integer -/
def f (n : ℕ+) : ℕ := sorry

/-- Recursive application of f, k times -/
def f_k (k : ℕ) (n : ℕ+) : ℕ :=
  match k with
  | 0 => n.val
  | k + 1 => f (⟨f_k k n, sorry⟩)

/-- The main theorem to prove -/
theorem f_2005_of_2006_eq_145 : f_k 2005 ⟨2006, sorry⟩ = 145 := by sorry

end NUMINAMATH_CALUDE_f_2005_of_2006_eq_145_l2029_202929


namespace NUMINAMATH_CALUDE_leg_head_difference_l2029_202915

/-- Represents the number of ducks in the group -/
def num_ducks : ℕ := sorry

/-- Represents the number of cows in the group -/
def num_cows : ℕ := 13

/-- Calculates the total number of legs in the group -/
def total_legs : ℕ := 2 * num_ducks + 4 * num_cows

/-- Calculates the total number of heads in the group -/
def total_heads : ℕ := num_ducks + num_cows

/-- States that the difference between total legs and thrice the total heads is 13 -/
theorem leg_head_difference : total_legs - 3 * total_heads = 13 := by sorry

end NUMINAMATH_CALUDE_leg_head_difference_l2029_202915


namespace NUMINAMATH_CALUDE_smallest_n_with_conditions_l2029_202980

theorem smallest_n_with_conditions : ∃ (m a : ℕ),
  145^2 = m^3 - (m-1)^3 + 5 ∧
  2*145 + 117 = a^2 ∧
  ∀ (n : ℕ), n > 0 → n < 145 →
    (∀ (m' a' : ℕ), n^2 ≠ m'^3 - (m'-1)^3 + 5 ∨ 2*n + 117 ≠ a'^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_conditions_l2029_202980


namespace NUMINAMATH_CALUDE_cost_of_nuts_l2029_202927

/-- Given Alyssa's purchases and refund, calculate the cost of the pack of nuts -/
theorem cost_of_nuts (grapes_cost refund_cherries total_spent : ℚ) 
  (h1 : grapes_cost = 12.08)
  (h2 : refund_cherries = 9.85)
  (h3 : total_spent = 26.35) :
  grapes_cost - refund_cherries + (total_spent - (grapes_cost - refund_cherries)) = 24.12 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_nuts_l2029_202927


namespace NUMINAMATH_CALUDE_root_conditions_l2029_202955

theorem root_conditions (a b c : ℝ) : 
  (∀ x : ℝ, x^5 + 2*x^4 + a*x^2 + b*x = c ↔ x = -1 ∨ x = 1) ↔ 
  (a = -6 ∧ b = -1 ∧ c = -4) :=
sorry

end NUMINAMATH_CALUDE_root_conditions_l2029_202955


namespace NUMINAMATH_CALUDE_parabola_coeff_sum_l2029_202904

/-- A parabola with equation y = ax^2 + bx + c, vertex at (2, 3), and passing through (5, 6) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 2
  vertex_y : ℝ := 3
  point_x : ℝ := 5
  point_y : ℝ := 6
  eq_at_vertex : 3 = a * 2^2 + b * 2 + c
  eq_at_point : 6 = a * 5^2 + b * 5 + c

/-- The sum of coefficients a, b, and c equals 4 -/
theorem parabola_coeff_sum (p : Parabola) : p.a + p.b + p.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coeff_sum_l2029_202904


namespace NUMINAMATH_CALUDE_locus_of_point_C_l2029_202932

/-- The locus of point C in a triangle ABC with given conditions forms an ellipse -/
theorem locus_of_point_C (A B C : ℝ × ℝ) : 
  (A = (-6, 0) ∧ B = (6, 0)) →  -- Coordinates of A and B
  (dist A B + dist B C + dist C A = 26) →  -- Perimeter condition
  (C.1 ≠ 7 ∧ C.1 ≠ -7) →  -- Exclude points where x = ±7
  (C.1^2 / 49 + C.2^2 / 13 = 1) :=  -- Equation of the ellipse
by sorry

end NUMINAMATH_CALUDE_locus_of_point_C_l2029_202932


namespace NUMINAMATH_CALUDE_product_mod_seven_l2029_202913

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l2029_202913


namespace NUMINAMATH_CALUDE_right_triangle_sets_l2029_202971

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  ¬(is_right_triangle 5 7 10) ∧
  (is_right_triangle 3 4 5) ∧
  (is_right_triangle 6 8 10) ∧
  (is_right_triangle 1 2 (Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l2029_202971


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2029_202977

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 + a 8 = 10 → 3 * a 5 + a 7 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2029_202977


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2029_202938

-- Define a geometric sequence with common ratio 2
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

-- Theorem statement
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_positive : ∀ n, a n > 0)
  (h_product : a 3 * a 11 = 16) :
  a 5 = 1 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l2029_202938


namespace NUMINAMATH_CALUDE_part_one_part_two_l2029_202951

-- Part 1
theorem part_one (x y : ℝ) (hx : x = Real.sqrt 2 - 1) (hy : y = Real.sqrt 2 + 1) :
  y / x + x / y = 6 := by sorry

-- Part 2
theorem part_two :
  (Real.sqrt 3 + Real.sqrt 2 - 2) * (Real.sqrt 3 - Real.sqrt 2 + 2) = 4 * Real.sqrt 2 - 3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2029_202951


namespace NUMINAMATH_CALUDE_floor_expression_l2029_202987

theorem floor_expression (n : ℕ) (hn : n = 101) : 
  ⌊(8 * (n^2 + 1) : ℚ) / (n^2 - 1)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_l2029_202987


namespace NUMINAMATH_CALUDE_special_polynomial_form_l2029_202947

/-- A polynomial satisfying the given functional equation. -/
structure SpecialPolynomial where
  P : ℝ → ℝ
  equation_holds : ∀ (a b c : ℝ),
    P (a + b - 2*c) + P (b + c - 2*a) + P (c + a - 2*b) =
    3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)

/-- The theorem stating the form of polynomials satisfying the functional equation. -/
theorem special_polynomial_form (p : SpecialPolynomial) :
  ∃ (a b : ℝ), (∀ x, p.P x = a * x + b) ∨ (∀ x, p.P x = a * x^2 + b * x) := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_form_l2029_202947


namespace NUMINAMATH_CALUDE_conditions_necessary_not_sufficient_l2029_202988

theorem conditions_necessary_not_sufficient :
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → |x| ≤ 1 ∧ |y| ≤ 1) ∧
  ¬(∀ x y : ℝ, |x| ≤ 1 ∧ |y| ≤ 1 → x^2 + y^2 ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_conditions_necessary_not_sufficient_l2029_202988


namespace NUMINAMATH_CALUDE_white_then_red_prob_is_one_thirtieth_l2029_202901

/-- A type representing the colors of balls in the bag -/
inductive Color
| Red | Blue | Green | Yellow | Purple | White

/-- The total number of balls in the bag -/
def total_balls : ℕ := 6

/-- The number of colored balls (excluding white) in the bag -/
def colored_balls : ℕ := 5

/-- The probability of drawing a specific ball from the bag -/
def draw_probability (n : ℕ) : ℚ := 1 / n

/-- The probability of drawing the white ball first and the red ball second -/
def white_then_red_probability : ℚ :=
  draw_probability total_balls * draw_probability colored_balls

/-- Theorem stating that the probability of drawing the white ball first
    and the red ball second is 1/30 -/
theorem white_then_red_prob_is_one_thirtieth :
  white_then_red_probability = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_white_then_red_prob_is_one_thirtieth_l2029_202901


namespace NUMINAMATH_CALUDE_darius_bucket_count_l2029_202962

/-- Represents the number of ounces in each of Darius's water buckets -/
def water_buckets : List ℕ := [11, 13, 12, 16, 10]

/-- The total amount of water in the first large bucket -/
def first_large_bucket : ℕ := 23

/-- The total amount of water in the second large bucket -/
def second_large_bucket : ℕ := 39

theorem darius_bucket_count :
  ∃ (bucket : ℕ) (remaining : List ℕ),
    bucket ∈ water_buckets ∧
    remaining = water_buckets.filter (λ x => x ≠ bucket ∧ x ≠ 10) ∧
    bucket + 10 = first_large_bucket ∧
    remaining.sum = second_large_bucket ∧
    water_buckets.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_darius_bucket_count_l2029_202962


namespace NUMINAMATH_CALUDE_triangle_height_calculation_l2029_202963

/-- Given a triangle with area 615 m² and one side of 123 meters, 
    the length of the perpendicular dropped on this side from the opposite vertex is 10 meters. -/
theorem triangle_height_calculation (A : ℝ) (b h : ℝ) 
    (h_area : A = 615)
    (h_base : b = 123)
    (h_triangle_area : A = (b * h) / 2) : h = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_calculation_l2029_202963


namespace NUMINAMATH_CALUDE_lending_scenario_l2029_202935

/-- Proves that given the conditions of the lending scenario, the principal amount is 3500 Rs. -/
theorem lending_scenario (P : ℝ) 
  (h1 : P + (P * 0.1 * 3) = 1.3 * P)  -- B owes A after 3 years
  (h2 : P + (P * 0.12 * 3) = 1.36 * P)  -- C owes B after 3 years
  (h3 : 1.36 * P - 1.3 * P = 210)  -- B's gain over 3 years
  : P = 3500 := by
  sorry

#check lending_scenario

end NUMINAMATH_CALUDE_lending_scenario_l2029_202935


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2029_202986

theorem polynomial_simplification (x : ℝ) :
  (15 * x^13 + 10 * x^12 + 7 * x^11) + (3 * x^15 + 2 * x^13 + x^11 + 4 * x^9 + 2 * x^5 + 6) =
  3 * x^15 + 17 * x^13 + 10 * x^12 + 8 * x^11 + 4 * x^9 + 2 * x^5 + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2029_202986


namespace NUMINAMATH_CALUDE_lea_binders_purchase_l2029_202953

/-- The number of binders Léa bought -/
def num_binders : ℕ := 3

/-- The cost of one book in dollars -/
def book_cost : ℕ := 16

/-- The cost of one binder in dollars -/
def binder_cost : ℕ := 2

/-- The cost of one notebook in dollars -/
def notebook_cost : ℕ := 1

/-- The number of notebooks Léa bought -/
def num_notebooks : ℕ := 6

/-- The total cost of Léa's purchases in dollars -/
def total_cost : ℕ := 28

theorem lea_binders_purchase :
  book_cost + binder_cost * num_binders + notebook_cost * num_notebooks = total_cost :=
by sorry

end NUMINAMATH_CALUDE_lea_binders_purchase_l2029_202953


namespace NUMINAMATH_CALUDE_expected_rolls_in_non_leap_year_l2029_202921

/-- Represents the outcomes of rolling an eight-sided die -/
inductive DieOutcome
| Composite
| Prime
| RollAgain

/-- The probability of each outcome when rolling the die -/
def outcomeProbability (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.Composite => 2/8
  | DieOutcome.Prime => 4/8
  | DieOutcome.RollAgain => 2/8

/-- The expected number of rolls on a single day -/
def expectedRollsPerDay : ℚ :=
  4/3

/-- The number of days in a non-leap year -/
def daysInNonLeapYear : ℕ := 365

/-- The expected number of rolls in a non-leap year -/
def expectedRollsInYear : ℚ :=
  expectedRollsPerDay * daysInNonLeapYear

theorem expected_rolls_in_non_leap_year :
  expectedRollsInYear = 486 + 2/3 :=
sorry

end NUMINAMATH_CALUDE_expected_rolls_in_non_leap_year_l2029_202921


namespace NUMINAMATH_CALUDE_calculation_proof_l2029_202991

theorem calculation_proof : (-1/2)⁻¹ - 4 * Real.cos (30 * π / 180) - (π + 2013)^0 + Real.sqrt 12 = -3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2029_202991


namespace NUMINAMATH_CALUDE_trig_identity_l2029_202981

theorem trig_identity (α : Real) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2029_202981


namespace NUMINAMATH_CALUDE_total_age_is_23_l2029_202939

/-- Proves that the total combined age of Ryanne, Hezekiah, and Jamison is 23 years -/
theorem total_age_is_23 (hezekiah_age : ℕ) 
  (ryanne_older : hezekiah_age + 7 = ryanne_age)
  (sum_ryanne_hezekiah : ryanne_age + hezekiah_age = 15)
  (jamison_twice : jamison_age = 2 * hezekiah_age) : 
  ryanne_age + hezekiah_age + jamison_age = 23 :=
by
  sorry

#check total_age_is_23

end NUMINAMATH_CALUDE_total_age_is_23_l2029_202939
