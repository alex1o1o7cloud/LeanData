import Mathlib

namespace NUMINAMATH_CALUDE_area_of_triangle_PQR_prove_area_of_triangle_PQR_l1748_174850

/-- Given two lines intersecting at point P(2,8), where one line has a slope of 3
    and the other has a slope of -1, and Q and R are the x-intercepts of these lines respectively,
    the area of triangle PQR is 128/3. -/
theorem area_of_triangle_PQR : ℝ → Prop :=
  fun area =>
    let P : ℝ × ℝ := (2, 8)
    let slope1 : ℝ := 3
    let slope2 : ℝ := -1
    let line1 := fun x => slope1 * (x - P.1) + P.2
    let line2 := fun x => slope2 * (x - P.1) + P.2
    let Q : ℝ × ℝ := (-(line1 0) / slope1, 0)
    let R : ℝ × ℝ := (-(line2 0) / slope2, 0)
    area = 128 / 3 ∧
    area = (1 / 2) * (R.1 - Q.1) * P.2

/-- Proof of the theorem -/
theorem prove_area_of_triangle_PQR : area_of_triangle_PQR (128 / 3) := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_PQR_prove_area_of_triangle_PQR_l1748_174850


namespace NUMINAMATH_CALUDE_winter_mows_calculation_winter_mows_value_l1748_174870

/-- The number of times Kale mowed his lawn in winter -/
def winter_mows : ℕ := sorry

/-- The number of times Kale mowed his lawn in spring -/
def spring_mows : ℕ := 8

/-- The number of times Kale mowed his lawn in summer -/
def summer_mows : ℕ := 5

/-- The number of times Kale mowed his lawn in fall -/
def fall_mows : ℕ := 12

/-- The average number of times Kale mowed his lawn per season -/
def average_mows_per_season : ℕ := 7

/-- The number of seasons in a year -/
def seasons_per_year : ℕ := 4

theorem winter_mows_calculation :
  winter_mows = average_mows_per_season * seasons_per_year - (spring_mows + summer_mows + fall_mows) :=
by sorry

theorem winter_mows_value : winter_mows = 3 :=
by sorry

end NUMINAMATH_CALUDE_winter_mows_calculation_winter_mows_value_l1748_174870


namespace NUMINAMATH_CALUDE_fraction_simplification_l1748_174826

theorem fraction_simplification (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (1 / y) / (1 / x) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1748_174826


namespace NUMINAMATH_CALUDE_t_in_possible_values_l1748_174857

/-- The set of possible values for t given the conditions -/
def possible_t_values : Set ℝ :=
  {t | 3 < t ∧ t < 4}

/-- The point (1, t) is above the line 2x - y + 1 = 0 -/
def above_line (t : ℝ) : Prop :=
  2 * 1 - t + 1 < 0

/-- The inequality x^2 + (2t-4)x + 4 > 0 always holds -/
def inequality_holds (t : ℝ) : Prop :=
  ∀ x, x^2 + (2*t-4)*x + 4 > 0

/-- Given the conditions, prove that t is in the set of possible values -/
theorem t_in_possible_values (t : ℝ) 
  (h1 : above_line t) 
  (h2 : inequality_holds t) : 
  t ∈ possible_t_values :=
sorry

end NUMINAMATH_CALUDE_t_in_possible_values_l1748_174857


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l1748_174847

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k * (x - 1)

theorem ellipse_and_line_intersection :
  ∀ (a b : ℝ), a > b ∧ b > 0 →
  ellipse_C (Real.sqrt 2) 1 a b →
  (∃ (x : ℝ), x > 0 ∧ ellipse_C x 0 a b ∧ x^2 = 2) →
  (∀ (k : ℝ), k > 0 →
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      ellipse_C x₁ y₁ a b ∧
      ellipse_C x₂ y₂ a b ∧
      line_l x₁ y₁ k ∧
      line_l x₂ y₂ k ∧
      x₂ - 1 = -x₁ ∧
      y₂ = -k - y₁) →
    k = Real.sqrt 2 / 2 ∧
    ∃ (x₁ y₁ x₂ y₂ : ℝ),
      ellipse_C x₁ y₁ a b ∧
      ellipse_C x₂ y₂ a b ∧
      line_l x₁ y₁ k ∧
      line_l x₂ y₂ k ∧
      Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = Real.sqrt 42 / 2) →
  a^2 = 4 ∧ b^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l1748_174847


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1748_174802

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r

theorem geometric_sequence_product (x y z : ℝ) :
  is_geometric_sequence (-1) x y z (-2) →
  x * y * z = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1748_174802


namespace NUMINAMATH_CALUDE_max_missed_problems_correct_l1748_174801

/-- The number of problems in the test -/
def total_problems : ℕ := 50

/-- The minimum percentage required to pass the test -/
def pass_percentage : ℚ := 85 / 100

/-- The maximum number of problems a student can miss and still pass the test -/
def max_missed_problems : ℕ := 7

theorem max_missed_problems_correct :
  (max_missed_problems ≤ total_problems) ∧
  ((total_problems - max_missed_problems : ℚ) / total_problems ≥ pass_percentage) ∧
  ∀ n : ℕ, n > max_missed_problems →
    ((total_problems - n : ℚ) / total_problems < pass_percentage) :=
by sorry

end NUMINAMATH_CALUDE_max_missed_problems_correct_l1748_174801


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1748_174827

/-- Given a hyperbola (x^2 / a^2) - (y^2 / 81) = 1 with a > 0, if one of its asymptotes is y = 3x, then a = 3 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 81 = 1 → ∃ k : ℝ, y = k * x ∧ |k| = 9 / a) →
  (∃ x y : ℝ, x^2 / a^2 - y^2 / 81 = 1 ∧ y = 3 * x) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1748_174827


namespace NUMINAMATH_CALUDE_production_time_calculation_l1748_174828

/-- Given that 5 machines can produce 20 units in 10 hours, 
    prove that 10 machines will take 25 hours to produce 100 units. -/
theorem production_time_calculation 
  (machines_initial : ℕ) 
  (units_initial : ℕ) 
  (hours_initial : ℕ) 
  (machines_final : ℕ) 
  (units_final : ℕ) 
  (h1 : machines_initial = 5) 
  (h2 : units_initial = 20) 
  (h3 : hours_initial = 10) 
  (h4 : machines_final = 10) 
  (h5 : units_final = 100) : 
  (units_final : ℚ) * machines_initial * hours_initial / 
  (units_initial * machines_final) = 25 := by
  sorry


end NUMINAMATH_CALUDE_production_time_calculation_l1748_174828


namespace NUMINAMATH_CALUDE_sector_central_angle_l1748_174895

-- Define the sector
structure Sector where
  circumference : ℝ
  area : ℝ

-- Define the given sector
def given_sector : Sector := { circumference := 6, area := 2 }

-- Define the possible central angles
def possible_angles : Set ℝ := {1, 4}

-- Theorem statement
theorem sector_central_angle (s : Sector) (h1 : s = given_sector) :
  ∃ θ ∈ possible_angles, 
    ∃ r l : ℝ, 
      r > 0 ∧ 
      l > 0 ∧ 
      2 * r + l = s.circumference ∧ 
      1 / 2 * r * l = s.area ∧ 
      θ = l / r :=
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1748_174895


namespace NUMINAMATH_CALUDE_nested_subtraction_simplification_l1748_174859

theorem nested_subtraction_simplification (y : ℝ) : 1 - (2 - (3 - (4 - (5 - y)))) = 3 - y := by
  sorry

end NUMINAMATH_CALUDE_nested_subtraction_simplification_l1748_174859


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l1748_174858

/-- p is the condition a^2 + a ≠ 0 -/
def p (a : ℝ) : Prop := a^2 + a ≠ 0

/-- q is the condition a ≠ 0 -/
def q (a : ℝ) : Prop := a ≠ 0

/-- p is a sufficient but not necessary condition for q -/
theorem p_sufficient_not_necessary : 
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬p a) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l1748_174858


namespace NUMINAMATH_CALUDE_john_popcorn_profit_l1748_174854

/-- Calculates the profit from selling popcorn bags -/
def popcorn_profit (cost_price selling_price number_of_bags : ℕ) : ℕ :=
  (selling_price - cost_price) * number_of_bags

/-- Theorem: John's profit from selling 30 bags of popcorn is $120 -/
theorem john_popcorn_profit :
  popcorn_profit 4 8 30 = 120 := by
  sorry

end NUMINAMATH_CALUDE_john_popcorn_profit_l1748_174854


namespace NUMINAMATH_CALUDE_g_has_four_roots_l1748_174835

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then |2^x - 1| else 3 / (x - 1)

-- Define the composition function g
noncomputable def g (x : ℝ) : ℝ := f (f x) - 2

-- Theorem statement
theorem g_has_four_roots :
  ∃ (a b c d : ℝ), (∀ x : ℝ, g x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :=
sorry

end NUMINAMATH_CALUDE_g_has_four_roots_l1748_174835


namespace NUMINAMATH_CALUDE_sophia_next_test_score_l1748_174899

def current_scores : List ℕ := [95, 85, 75, 65, 95]
def desired_increase : ℕ := 5

def minimum_required_score (scores : List ℕ) (increase : ℕ) : ℕ :=
  let current_sum := scores.sum
  let current_count := scores.length
  let current_average := current_sum / current_count
  let target_average := current_average + increase
  let total_count := current_count + 1
  target_average * total_count - current_sum

theorem sophia_next_test_score :
  minimum_required_score current_scores desired_increase = 113 := by
  sorry

end NUMINAMATH_CALUDE_sophia_next_test_score_l1748_174899


namespace NUMINAMATH_CALUDE_climbing_time_problem_l1748_174800

/-- The time it takes for Jason to be 42 feet higher than Matt, given their climbing rates. -/
theorem climbing_time_problem (matt_rate jason_rate : ℝ) (height_difference : ℝ)
  (h_matt : matt_rate = 6)
  (h_jason : jason_rate = 12)
  (h_diff : height_difference = 42) :
  (height_difference / (jason_rate - matt_rate)) = 7 :=
by sorry

end NUMINAMATH_CALUDE_climbing_time_problem_l1748_174800


namespace NUMINAMATH_CALUDE_triangle_area_l1748_174874

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  -- Right triangle condition
  (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0 ∧
  -- Angle Q = 60°
  (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = (P.1 - Q.1)^2 + (P.2 - Q.2)^2 ∧
  -- Angle R = 30°
  4 * ((P.1 - R.1)^2 + (P.2 - R.2)^2) = (Q.1 - R.1)^2 + (Q.2 - R.2)^2 ∧
  -- QR = 12
  (Q.1 - R.1)^2 + (Q.2 - R.2)^2 = 144

-- Theorem statement
theorem triangle_area (P Q R : ℝ × ℝ) (h : Triangle P Q R) :
  let area := abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2)) / 2
  area = 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1748_174874


namespace NUMINAMATH_CALUDE_distance_to_grandma_is_100_l1748_174846

/-- Represents the efficiency of a car in miles per gallon -/
def car_efficiency : ℝ := 20

/-- Represents the amount of gas needed to reach Grandma's house in gallons -/
def gas_needed : ℝ := 5

/-- Calculates the distance to Grandma's house based on car efficiency and gas needed -/
def distance_to_grandma : ℝ := car_efficiency * gas_needed

/-- Theorem stating that the distance to Grandma's house is 100 miles -/
theorem distance_to_grandma_is_100 : distance_to_grandma = 100 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_grandma_is_100_l1748_174846


namespace NUMINAMATH_CALUDE_math_team_combinations_l1748_174814

theorem math_team_combinations (girls : ℕ) (boys : ℕ) : 
  girls = 5 → boys = 8 → (girls.choose 1) * ((girls - 1).choose 2) * (boys.choose 2) = 840 := by
  sorry

end NUMINAMATH_CALUDE_math_team_combinations_l1748_174814


namespace NUMINAMATH_CALUDE_min_operations_to_240_l1748_174833

def transform (n : ℕ) : ℕ → ℕ
| 0 => n
| (k + 1) => min (transform (n + 1) k) (transform (n * 2) k)

theorem min_operations_to_240 : transform 1 10 = 240 ∧ ∀ k < 10, transform 1 k < 240 := by sorry

end NUMINAMATH_CALUDE_min_operations_to_240_l1748_174833


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1748_174812

theorem sum_of_x_and_y (x y : ℝ) (h1 : y - x = 1) (h2 : y^2 = x^2 + 6) : x + y = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1748_174812


namespace NUMINAMATH_CALUDE_semicircle_radius_is_24_over_5_l1748_174830

/-- A right triangle with a semicircle inscribed -/
structure RightTriangleWithSemicircle where
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of side QR -/
  qr : ℝ
  /-- Radius of the inscribed semicircle -/
  r : ℝ
  /-- PQ is positive -/
  pq_pos : 0 < pq
  /-- QR is positive -/
  qr_pos : 0 < qr
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagorean : pq^2 = qr^2 + (pq - qr)^2
  /-- The radius satisfies the relation with sides -/
  radius_relation : r = (pq * qr) / (pq + qr + (pq - qr))

/-- The main theorem: For a right triangle with PQ = 15 and QR = 8, 
    the radius of the inscribed semicircle is 24/5 -/
theorem semicircle_radius_is_24_over_5 :
  ∃ (t : RightTriangleWithSemicircle), t.pq = 15 ∧ t.qr = 8 ∧ t.r = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_is_24_over_5_l1748_174830


namespace NUMINAMATH_CALUDE_set_inclusion_l1748_174824

-- Define the sets A, B, and C
def A : Set ℝ := {x | ∃ k : ℤ, x = k / 6 + 1}
def B : Set ℝ := {x | ∃ k : ℤ, x = k / 3 + 1 / 2}
def C : Set ℝ := {x | ∃ k : ℤ, x = 2 * k / 3 + 1 / 2}

-- State the theorem
theorem set_inclusion : C ⊆ B ∧ B ⊆ A := by sorry

end NUMINAMATH_CALUDE_set_inclusion_l1748_174824


namespace NUMINAMATH_CALUDE_divisibility_in_sequence_l1748_174831

theorem divisibility_in_sequence (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ k : ℕ, k ∈ Finset.range (n - 1) ∧ (n ∣ 2^(k + 1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_in_sequence_l1748_174831


namespace NUMINAMATH_CALUDE_correct_number_of_students_l1748_174822

/-- The number of students in a class preparing for a field trip --/
def number_of_students : ℕ := 30

/-- The amount each student contributes per Friday in dollars --/
def contribution_per_friday : ℕ := 2

/-- The number of Fridays in the collection period --/
def number_of_fridays : ℕ := 8

/-- The total amount collected for the trip in dollars --/
def total_amount : ℕ := 480

/-- Theorem stating that the number of students is correct given the conditions --/
theorem correct_number_of_students :
  number_of_students * contribution_per_friday * number_of_fridays = total_amount :=
sorry

end NUMINAMATH_CALUDE_correct_number_of_students_l1748_174822


namespace NUMINAMATH_CALUDE_triangle_squares_area_l1748_174888

theorem triangle_squares_area (y : ℝ) : 
  y > 0 →
  (3 * y)^2 + (6 * y)^2 + (1/2 * (3 * y) * (6 * y)) = 1000 →
  y = 10 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_squares_area_l1748_174888


namespace NUMINAMATH_CALUDE_kathleen_remaining_money_l1748_174813

def kathleen_problem (june_savings july_savings august_savings : ℕ)
                     (school_supplies_cost clothes_cost : ℕ)
                     (aunt_bonus_threshold aunt_bonus : ℕ) : ℕ :=
  let total_savings := june_savings + july_savings + august_savings
  let total_expenses := school_supplies_cost + clothes_cost
  let bonus := if total_savings > aunt_bonus_threshold then aunt_bonus else 0
  total_savings + bonus - total_expenses

theorem kathleen_remaining_money :
  kathleen_problem 21 46 45 12 54 125 25 = 46 := by sorry

end NUMINAMATH_CALUDE_kathleen_remaining_money_l1748_174813


namespace NUMINAMATH_CALUDE_monotonicity_nonpositive_a_monotonicity_positive_a_f_geq_f_neg_l1748_174807

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

-- Theorem for monotonicity when a ≤ 0
theorem monotonicity_nonpositive_a (a : ℝ) (h : a ≤ 0) :
  StrictMono (f a) := by sorry

-- Theorem for monotonicity when a > 0
theorem monotonicity_positive_a (a : ℝ) (h : a > 0) :
  ∀ x y, x < y → (
    (x < Real.log a ∧ y < Real.log a → f a y < f a x) ∧
    (Real.log a < x ∧ Real.log a < y → f a x < f a y)
  ) := by sorry

-- Theorem for f(x) ≥ f(-x) when a = 1 and x ≥ 0
theorem f_geq_f_neg (x : ℝ) (h : x ≥ 0) :
  f 1 x ≥ f 1 (-x) := by sorry

end

end NUMINAMATH_CALUDE_monotonicity_nonpositive_a_monotonicity_positive_a_f_geq_f_neg_l1748_174807


namespace NUMINAMATH_CALUDE_tan_four_expression_equals_21_68_l1748_174834

theorem tan_four_expression_equals_21_68 (θ : ℝ) (h : Real.tan θ = 4) :
  (Real.sin θ + Real.cos θ) / (17 * Real.sin θ) + Real.sin θ^2 / 4 = 21/68 := by
  sorry

end NUMINAMATH_CALUDE_tan_four_expression_equals_21_68_l1748_174834


namespace NUMINAMATH_CALUDE_factorial_difference_l1748_174883

theorem factorial_difference : Nat.factorial 8 - Nat.factorial 7 = 35280 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l1748_174883


namespace NUMINAMATH_CALUDE_magic_8_ball_three_out_of_six_l1748_174863

def magic_8_ball_probability (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem magic_8_ball_three_out_of_six :
  magic_8_ball_probability 6 3 (1/4) = 135/1024 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_three_out_of_six_l1748_174863


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l1748_174893

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 1 → (1 / a) < 1) ∧ ¬((1 / a) < 1 → a > 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l1748_174893


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1748_174890

/-- Given that the solution set of ax² + bx + c > 0 is {x | -4 < x < 7},
    prove that the solution set of cx² - bx + a > 0 is {x | x < -1/7 or x > 1/4} -/
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, (a * x^2 + b * x + c > 0) ↔ (-4 < x ∧ x < 7)) :
  ∀ x : ℝ, (c * x^2 - b * x + a > 0) ↔ (x < -1/7 ∨ x > 1/4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1748_174890


namespace NUMINAMATH_CALUDE_students_without_A_l1748_174862

theorem students_without_A (total : ℕ) (lit_A : ℕ) (sci_A : ℕ) (both_A : ℕ) : 
  total = 35 → lit_A = 10 → sci_A = 15 → both_A = 5 → 
  total - (lit_A + sci_A - both_A) = 15 := by
  sorry

end NUMINAMATH_CALUDE_students_without_A_l1748_174862


namespace NUMINAMATH_CALUDE_triangle_property_l1748_174855

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  -- The angle at C is a right angle
  sorry

def altitude_meets_AB (t : Triangle) (D : ℝ × ℝ) : Prop :=
  -- The altitude from C meets AB at D
  sorry

def integer_sides (t : Triangle) : Prop :=
  -- The lengths of the sides of triangle ABC are integers
  sorry

def BD_length (t : Triangle) (D : ℝ × ℝ) : Prop :=
  -- BD = 29³
  sorry

def cos_B_fraction (t : Triangle) (m n : ℕ) : Prop :=
  -- cos B = m/n, where m and n are relatively prime positive integers
  sorry

theorem triangle_property (t : Triangle) (D : ℝ × ℝ) (m n : ℕ) :
  is_right_triangle t →
  altitude_meets_AB t D →
  integer_sides t →
  BD_length t D →
  cos_B_fraction t m n →
  m + n = 450 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l1748_174855


namespace NUMINAMATH_CALUDE_inequality_solution_l1748_174896

theorem inequality_solution (x : ℝ) : 3 * x^2 - x > 4 ∧ x < 3 → 1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1748_174896


namespace NUMINAMATH_CALUDE_unique_twelve_times_digit_sum_l1748_174876

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem unique_twelve_times_digit_sum :
  ∀ n : ℕ, n > 0 → (n = 12 * sum_of_digits n ↔ n = 108) := by
  sorry

end NUMINAMATH_CALUDE_unique_twelve_times_digit_sum_l1748_174876


namespace NUMINAMATH_CALUDE_g_five_l1748_174878

/-- A function satisfying g(x+y) = g(x) + g(y) for all real x and y, and g(1) = 2 -/
def g : ℝ → ℝ :=
  fun x => sorry

/-- The functional equation property of g -/
axiom g_add (x y : ℝ) : g (x + y) = g x + g y

/-- The value of g at 1 -/
axiom g_one : g 1 = 2

/-- The theorem stating that g(5) = 10 -/
theorem g_five : g 5 = 10 := by sorry

end NUMINAMATH_CALUDE_g_five_l1748_174878


namespace NUMINAMATH_CALUDE_intersection_equals_A_l1748_174840

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x < 0}

-- Define set B
def B : Set ℝ := {x : ℝ | |x| < 2}

-- Theorem statement
theorem intersection_equals_A : A ∩ B = A := by sorry

end NUMINAMATH_CALUDE_intersection_equals_A_l1748_174840


namespace NUMINAMATH_CALUDE_distinct_laptop_choices_l1748_174881

/-- The number of ways to choose 3 distinct items from a set of 15 items -/
def choose_distinct (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else (n - k + 1).factorial / (n - k).factorial

theorem distinct_laptop_choices :
  choose_distinct 15 3 = 2730 := by
sorry

end NUMINAMATH_CALUDE_distinct_laptop_choices_l1748_174881


namespace NUMINAMATH_CALUDE_equation_proof_l1748_174845

theorem equation_proof : Real.sqrt ((5568 / 87) ^ (1/3) + Real.sqrt (72 * 2)) = Real.sqrt 256 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1748_174845


namespace NUMINAMATH_CALUDE_rectangle_diagonal_shortcut_l1748_174875

theorem rectangle_diagonal_shortcut (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≤ y) :
  x + y - Real.sqrt (x^2 + y^2) = (1/3) * y → x/y = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_shortcut_l1748_174875


namespace NUMINAMATH_CALUDE_hilt_money_left_l1748_174871

theorem hilt_money_left (initial : ℕ) (pencil_cost : ℕ) (candy_cost : ℕ) : 
  initial = 43 → pencil_cost = 20 → candy_cost = 5 → 
  initial - (pencil_cost + candy_cost) = 18 := by
sorry

end NUMINAMATH_CALUDE_hilt_money_left_l1748_174871


namespace NUMINAMATH_CALUDE_coins_percentage_of_dollar_l1748_174866

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of pennies in Samantha's purse -/
def num_pennies : ℕ := 2

/-- The number of nickels in Samantha's purse -/
def num_nickels : ℕ := 1

/-- The number of dimes in Samantha's purse -/
def num_dimes : ℕ := 3

/-- The number of quarters in Samantha's purse -/
def num_quarters : ℕ := 2

/-- The total value of coins in Samantha's purse as a percentage of a dollar -/
theorem coins_percentage_of_dollar :
  (num_pennies * penny_value + num_nickels * nickel_value +
   num_dimes * dime_value + num_quarters * quarter_value) * 100 / 100 = 87 := by
  sorry

end NUMINAMATH_CALUDE_coins_percentage_of_dollar_l1748_174866


namespace NUMINAMATH_CALUDE_travel_time_ratio_l1748_174886

def time_NY_to_SF : ℝ := 24
def layover_time : ℝ := 16
def total_time : ℝ := 58

def time_NO_to_NY : ℝ := total_time - layover_time - time_NY_to_SF

theorem travel_time_ratio : time_NO_to_NY / time_NY_to_SF = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_ratio_l1748_174886


namespace NUMINAMATH_CALUDE_equation_solution_l1748_174856

theorem equation_solution :
  ∀ x : ℚ, (x ≠ 4 ∧ x ≠ -6) →
  ((x + 8) / (x - 4) = (x - 3) / (x + 6) ↔ x = -12 / 7) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1748_174856


namespace NUMINAMATH_CALUDE_total_highlighters_l1748_174851

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ)
  (h1 : pink = 3)
  (h2 : yellow = 7)
  (h3 : blue = 5) :
  pink + yellow + blue = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_l1748_174851


namespace NUMINAMATH_CALUDE_complex_power_sum_l1748_174811

open Complex

theorem complex_power_sum (z : ℂ) (h : z + 1 / z = 2 * Real.cos (5 * π / 180)) :
  z^1000 + 1 / z^1000 = 2 * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1748_174811


namespace NUMINAMATH_CALUDE_service_fee_calculation_l1748_174891

/-- Calculate the service fee for ticket purchase --/
theorem service_fee_calculation (num_tickets : ℕ) (ticket_price total_paid : ℚ) :
  num_tickets = 3 →
  ticket_price = 44 →
  total_paid = 150 →
  total_paid - (num_tickets : ℚ) * ticket_price = 18 := by
  sorry

end NUMINAMATH_CALUDE_service_fee_calculation_l1748_174891


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l1748_174865

/-- Proves that the number of adult tickets sold is 122, given the total number of tickets
    and the relationship between student and adult tickets. -/
theorem adult_tickets_sold (total_tickets : ℕ) (adult_tickets : ℕ) (student_tickets : ℕ)
    (h1 : total_tickets = 366)
    (h2 : student_tickets = 2 * adult_tickets)
    (h3 : total_tickets = adult_tickets + student_tickets) :
    adult_tickets = 122 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l1748_174865


namespace NUMINAMATH_CALUDE_choose_four_from_ten_l1748_174825

theorem choose_four_from_ten : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_ten_l1748_174825


namespace NUMINAMATH_CALUDE_students_with_dogs_l1748_174853

theorem students_with_dogs (total_students : ℕ) (girls_percentage : ℚ) (boys_percentage : ℚ)
  (girls_with_dogs_percentage : ℚ) (boys_with_dogs_percentage : ℚ)
  (h1 : total_students = 100)
  (h2 : girls_percentage = 1/2)
  (h3 : boys_percentage = 1/2)
  (h4 : girls_with_dogs_percentage = 1/5)
  (h5 : boys_with_dogs_percentage = 1/10) :
  (total_students : ℚ) * girls_percentage * girls_with_dogs_percentage +
  (total_students : ℚ) * boys_percentage * boys_with_dogs_percentage = 15 :=
by sorry

end NUMINAMATH_CALUDE_students_with_dogs_l1748_174853


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1748_174897

theorem rationalize_denominator :
  ∃ (A B C D E F : ℤ),
    (1 : ℝ) / (Real.sqrt 5 + Real.sqrt 2 + Real.sqrt 3) =
    (A * Real.sqrt 2 + B * Real.sqrt 3 + C * Real.sqrt 5 + D * Real.sqrt E) / F ∧
    F > 0 ∧
    A = -3 ∧
    B = -2 ∧
    C = 0 ∧
    D = 1 ∧
    E = 30 ∧
    F = 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1748_174897


namespace NUMINAMATH_CALUDE_negation_equivalence_l1748_174872

theorem negation_equivalence : 
  (¬∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1748_174872


namespace NUMINAMATH_CALUDE_ones_digit_largest_power_of_3_dividing_27_factorial_l1748_174829

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largestPowerOf3DividingFactorial (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (x + 1) / 3) 0

def onesDigit (n : ℕ) : ℕ := n % 10

theorem ones_digit_largest_power_of_3_dividing_27_factorial :
  onesDigit (3^(largestPowerOf3DividingFactorial 27)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_largest_power_of_3_dividing_27_factorial_l1748_174829


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1748_174810

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  (a^2 - 5*a + 6 = 0) → 
  (b^2 - 5*b + 6 = 0) → 
  (a ≠ b) →
  (c^2 = a^2 + b^2) →
  c = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1748_174810


namespace NUMINAMATH_CALUDE_john_thrice_tom_age_l1748_174842

/-- Proves that John was thrice as old as Tom 6 years ago, given the conditions -/
theorem john_thrice_tom_age (tom_current_age john_current_age x : ℕ) : 
  tom_current_age = 16 →
  john_current_age + 4 = 2 * (tom_current_age + 4) →
  john_current_age - x = 3 * (tom_current_age - x) →
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_john_thrice_tom_age_l1748_174842


namespace NUMINAMATH_CALUDE_savings_increase_l1748_174806

theorem savings_increase (income expenditure savings new_income new_expenditure new_savings : ℝ)
  (h1 : expenditure = 0.75 * income)
  (h2 : savings = income - expenditure)
  (h3 : new_income = 1.2 * income)
  (h4 : new_expenditure = 1.1 * expenditure)
  (h5 : new_savings = new_income - new_expenditure) :
  (new_savings - savings) / savings * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_savings_increase_l1748_174806


namespace NUMINAMATH_CALUDE_base_five_representation_l1748_174841

theorem base_five_representation (b : ℕ) : 
  (b^3 ≤ 329 ∧ 329 < b^4 ∧ 329 % b % 2 = 0) ↔ b = 5 :=
by sorry

end NUMINAMATH_CALUDE_base_five_representation_l1748_174841


namespace NUMINAMATH_CALUDE_polynomial_uniqueness_l1748_174864

/-- Given a polynomial Q(x) = Q(0) + Q(1)x + Q(2)x^2 with Q(-1) = 3, prove Q(x) = 3(1 + x + x^2) -/
theorem polynomial_uniqueness (Q : ℝ → ℝ) (h1 : ∀ x, Q x = Q 0 + Q 1 * x + Q 2 * x^2) 
  (h2 : Q (-1) = 3) : ∀ x, Q x = 3 * (1 + x + x^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_uniqueness_l1748_174864


namespace NUMINAMATH_CALUDE_no_99_cents_combination_l1748_174823

/-- Represents the types of coins available -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents -/
def coinValue (c : Coin) : Nat :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Represents a combination of five coins -/
def CoinCombination := Vector Coin 5

/-- Calculates the total value of a coin combination in cents -/
def totalValue (combo : CoinCombination) : Nat :=
  combo.toList.map coinValue |>.sum

/-- Theorem: It's impossible to make 99 cents with exactly five coins -/
theorem no_99_cents_combination :
  ¬∃ (combo : CoinCombination), totalValue combo = 99 := by
  sorry


end NUMINAMATH_CALUDE_no_99_cents_combination_l1748_174823


namespace NUMINAMATH_CALUDE_sum_reciprocals_simplification_l1748_174882

theorem sum_reciprocals_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^3 + b^3 = 3*(a + b)) : 
  a/b + b/a + 1/(a*b) = 4/(a*b) + 1 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_simplification_l1748_174882


namespace NUMINAMATH_CALUDE_sandy_marbles_multiple_l1748_174898

def melanie_marbles : ℕ := 84
def sandy_dozens : ℕ := 56

def marbles_in_dozen : ℕ := 12

theorem sandy_marbles_multiple : 
  (sandy_dozens * marbles_in_dozen) / melanie_marbles = 8 := by
  sorry

end NUMINAMATH_CALUDE_sandy_marbles_multiple_l1748_174898


namespace NUMINAMATH_CALUDE_base_for_six_digit_thousand_l1748_174832

theorem base_for_six_digit_thousand : ∃! (b : ℕ), b > 1 ∧ b^5 ≤ 1000 ∧ 1000 < b^6 := by sorry

end NUMINAMATH_CALUDE_base_for_six_digit_thousand_l1748_174832


namespace NUMINAMATH_CALUDE_decreasing_function_range_l1748_174868

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then (a - 5) * x - 2
  else x^2 - 2 * (a + 1) * x + 3 * a

-- Define the condition for the function to be decreasing
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0

-- Theorem statement
theorem decreasing_function_range (a : ℝ) :
  is_decreasing (f a) ↔ a ∈ Set.Icc 1 4 :=
sorry

end NUMINAMATH_CALUDE_decreasing_function_range_l1748_174868


namespace NUMINAMATH_CALUDE_alexis_skirt_time_l1748_174809

/-- The time it takes Alexis to sew a skirt -/
def skirt_time : ℝ := 2

/-- The time it takes Alexis to sew a coat -/
def coat_time : ℝ := 7

/-- The total time it takes Alexis to sew 6 skirts and 4 coats -/
def total_time : ℝ := 40

theorem alexis_skirt_time :
  skirt_time = 2 ∧
  coat_time = 7 ∧
  total_time = 40 ∧
  6 * skirt_time + 4 * coat_time = total_time :=
by sorry

end NUMINAMATH_CALUDE_alexis_skirt_time_l1748_174809


namespace NUMINAMATH_CALUDE_jakes_score_l1748_174821

theorem jakes_score (total_students : Nat) (avg_18 : ℚ) (avg_19 : ℚ) (avg_20 : ℚ) 
  (h1 : total_students = 20)
  (h2 : avg_18 = 75)
  (h3 : avg_19 = 76)
  (h4 : avg_20 = 77) :
  (total_students * avg_20 - (total_students - 1) * avg_19 : ℚ) = 96 := by
  sorry

end NUMINAMATH_CALUDE_jakes_score_l1748_174821


namespace NUMINAMATH_CALUDE_fabian_walking_speed_l1748_174873

theorem fabian_walking_speed (initial_hours : ℕ) (additional_hours : ℕ) (total_distance : ℕ) :
  initial_hours = 3 →
  additional_hours = 3 →
  total_distance = 30 →
  (initial_hours + additional_hours) * (total_distance / (initial_hours + additional_hours)) = total_distance :=
by sorry

end NUMINAMATH_CALUDE_fabian_walking_speed_l1748_174873


namespace NUMINAMATH_CALUDE_fraction_value_l1748_174837

theorem fraction_value (x : ℝ) : (3 * x^2 + 9 * x + 15) / (3 * x^2 + 9 * x + 5) = 41 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1748_174837


namespace NUMINAMATH_CALUDE_distinct_integer_roots_l1748_174818

theorem distinct_integer_roots (a : ℝ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 + 2*a*x = 8*a ∧ y^2 + 2*a*y = 8*a) ↔ 
  a ∈ ({4.5, 1, -12.5, -9} : Set ℝ) :=
by sorry

end NUMINAMATH_CALUDE_distinct_integer_roots_l1748_174818


namespace NUMINAMATH_CALUDE_incenter_bisects_orthocenter_circumcenter_angle_l1748_174894

-- Define the types for points and triangles
variable (Point : Type)
variable (Triangle : Type)

-- Define the properties of the triangle
variable (is_acute : Triangle → Prop)
variable (orthocenter : Triangle → Point)
variable (circumcenter : Triangle → Point)
variable (incenter : Triangle → Point)

-- Define the angle bisector property
variable (bisects_angle : Point → Point → Point → Point → Prop)

theorem incenter_bisects_orthocenter_circumcenter_angle 
  (ABC : Triangle) (H O I : Point) :
  is_acute ABC →
  H = orthocenter ABC →
  O = circumcenter ABC →
  I = incenter ABC →
  bisects_angle I A H O :=
sorry

end NUMINAMATH_CALUDE_incenter_bisects_orthocenter_circumcenter_angle_l1748_174894


namespace NUMINAMATH_CALUDE_min_road_length_on_grid_min_road_length_specific_points_l1748_174860

/-- Represents a point on a grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Calculates the Manhattan distance between two grid points -/
def manhattan_distance (p1 p2 : GridPoint) : ℕ :=
  (Int.natAbs (p1.x - p2.x)) + (Int.natAbs (p1.y - p2.y))

/-- Theorem: Minimum road length on a grid -/
theorem min_road_length_on_grid (square_side_length : ℕ) 
  (A B C : GridPoint) (h : square_side_length = 100) :
  let total_distance := 
    (manhattan_distance A B + manhattan_distance B C + manhattan_distance A C) / 2
  total_distance * square_side_length = 1000 :=
by
  sorry

/-- Main theorem application -/
theorem min_road_length_specific_points :
  let A : GridPoint := ⟨0, 0⟩
  let B : GridPoint := ⟨3, 2⟩
  let C : GridPoint := ⟨4, 3⟩
  let square_side_length : ℕ := 100
  let total_distance := 
    (manhattan_distance A B + manhattan_distance B C + manhattan_distance A C) / 2
  total_distance * square_side_length = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_min_road_length_on_grid_min_road_length_specific_points_l1748_174860


namespace NUMINAMATH_CALUDE_count_numbers_l1748_174819

/-- The number of digits available for creating numbers -/
def num_digits : ℕ := 5

/-- The set of digits available for creating numbers -/
def digit_set : Finset ℕ := {0, 1, 2, 3, 4}

/-- The number of digits required for each number -/
def num_places : ℕ := 4

/-- Function to calculate the number of four-digit numbers -/
def four_digit_numbers : ℕ := sorry

/-- Function to calculate the number of four-digit even numbers -/
def four_digit_even_numbers : ℕ := sorry

/-- Function to calculate the number of four-digit numbers without repeating digits -/
def four_digit_no_repeat : ℕ := sorry

/-- Function to calculate the number of four-digit even numbers without repeating digits -/
def four_digit_even_no_repeat : ℕ := sorry

theorem count_numbers :
  four_digit_numbers = 500 ∧
  four_digit_even_numbers = 300 ∧
  four_digit_no_repeat = 96 ∧
  four_digit_even_no_repeat = 60 := by sorry

end NUMINAMATH_CALUDE_count_numbers_l1748_174819


namespace NUMINAMATH_CALUDE_quadratic_inequality_implication_l1748_174879

theorem quadratic_inequality_implication (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + a > 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implication_l1748_174879


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1748_174836

theorem intersection_of_sets : 
  let A : Set ℝ := {x | x + 2 = 0}
  let B : Set ℝ := {x | x^2 - 4 = 0}
  A ∩ B = {-2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1748_174836


namespace NUMINAMATH_CALUDE_external_tangent_y_intercept_l1748_174848

-- Define the circles
def circle1_center : ℝ × ℝ := (1, 3)
def circle1_radius : ℝ := 3
def circle2_center : ℝ × ℝ := (10, 6)
def circle2_radius : ℝ := 7

-- Define the tangent line equation
def tangent_line (m b : ℝ) (x : ℝ) : ℝ := m * x + b

-- State the theorem
theorem external_tangent_y_intercept :
  ∃ (m b : ℝ), m > 0 ∧
  (∀ (x : ℝ), tangent_line m b x = m * x + b) ∧
  b = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_external_tangent_y_intercept_l1748_174848


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1748_174843

theorem trigonometric_identity 
  (α β γ : Real) 
  (a b c : Real) 
  (h1 : 0 < α) (h2 : α < π)
  (h3 : 0 < β) (h4 : β < π)
  (h5 : 0 < γ) (h6 : γ < π)
  (h7 : a > 0) (h8 : b > 0) (h9 : c > 0)
  (h10 : b = c * (Real.cos α + Real.cos β * Real.cos γ) / (Real.sin γ)^2)
  (h11 : a = c * (Real.cos β + Real.cos α * Real.cos γ) / (Real.sin γ)^2) :
  1 - (Real.cos α)^2 - (Real.cos β)^2 - (Real.cos γ)^2 - 2 * Real.cos α * Real.cos β * Real.cos γ = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1748_174843


namespace NUMINAMATH_CALUDE_min_value_expression_l1748_174889

theorem min_value_expression (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) :
  (6 * w) / (3 * u + 2 * v) + (6 * u) / (2 * v + 3 * w) + (2 * v) / (u + w) ≥ 2.5 + 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1748_174889


namespace NUMINAMATH_CALUDE_weight_of_b_l1748_174884

theorem weight_of_b (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 60 →
  (a + b + c) / 3 = 55 →
  (b + c + d) / 3 = 58 →
  (c + d + e) / 3 = 62 →
  b = 114 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l1748_174884


namespace NUMINAMATH_CALUDE_number_equation_solution_l1748_174815

theorem number_equation_solution : 
  ∃ x : ℝ, x - (1002 / 20.04) = 3500 ∧ x = 3550 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1748_174815


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_algebraic_expression_equality_l1748_174852

-- Part 1
theorem logarithm_expression_equality : 
  Real.log 5 * Real.log 20 - Real.log 2 * Real.log 50 - Real.log 25 = -1 := by sorry

-- Part 2
theorem algebraic_expression_equality (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  (2 * a^(2/3) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = 4 * a := by sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_algebraic_expression_equality_l1748_174852


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1748_174808

/-- The quadratic equation qx^2 - 8x + 2 = 0 has only one solution when q = 8 -/
theorem unique_solution_quadratic :
  ∃! (q : ℝ), q ≠ 0 ∧ (∃! x : ℝ, q * x^2 - 8 * x + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1748_174808


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l1748_174803

theorem right_triangle_perimeter 
  (area : ℝ) 
  (leg : ℝ) 
  (h1 : area = 150) 
  (h2 : leg = 30) : 
  ∃ (other_leg hypotenuse : ℝ),
    area = (1/2) * leg * other_leg ∧
    hypotenuse^2 = leg^2 + other_leg^2 ∧
    leg + other_leg + hypotenuse = 40 + 10 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l1748_174803


namespace NUMINAMATH_CALUDE_johnny_distance_when_met_l1748_174849

/-- The distance between Q and Y in km -/
def total_distance : ℝ := 45

/-- Matthew's walking rate in km/hour -/
def matthew_rate : ℝ := 3

/-- Johnny's walking rate in km/hour -/
def johnny_rate : ℝ := 4

/-- The time difference between Matthew's and Johnny's start in hours -/
def time_difference : ℝ := 1

/-- The distance Johnny walked when they met -/
def johnny_distance : ℝ := 24

theorem johnny_distance_when_met :
  let t := (total_distance - matthew_rate * time_difference) / (matthew_rate + johnny_rate)
  johnny_distance = johnny_rate * t :=
by sorry

end NUMINAMATH_CALUDE_johnny_distance_when_met_l1748_174849


namespace NUMINAMATH_CALUDE_original_number_l1748_174816

theorem original_number : ∃ x : ℝ, 3 * (2 * x + 9) = 51 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1748_174816


namespace NUMINAMATH_CALUDE_sunway_performance_equivalence_l1748_174892

/-- The peak performance of the Sunway TaihuLight supercomputer in calculations per second -/
def peak_performance : ℝ := 12.5 * 1e12

/-- The scientific notation representation of the peak performance -/
def scientific_notation : ℝ := 1.25 * 1e13

theorem sunway_performance_equivalence :
  peak_performance = scientific_notation := by sorry

end NUMINAMATH_CALUDE_sunway_performance_equivalence_l1748_174892


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_f_extrema_sum_max_l1748_174804

noncomputable section

def f (a x : ℝ) := x^2 / 2 - 4*a*x + a * Real.log x + 3*a^2 + 2*a

def f_deriv (a x : ℝ) := x - 4*a + a/x

theorem f_monotonicity_and_extrema (a : ℝ) (ha : a > 0) :
  (∀ x > 0, f_deriv a x ≥ 0) ∨
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0) :=
sorry

theorem f_extrema_sum_max (a : ℝ) (ha : a > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0 →
  ∀ y₁ y₂ : ℝ, y₁ ≠ y₂ → f_deriv a y₁ = 0 → f_deriv a y₂ = 0 →
  f a x₁ + f a x₂ ≥ f a y₁ + f a y₂ ∧
  f a x₁ + f a x₂ ≤ 1 :=
sorry

end

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_f_extrema_sum_max_l1748_174804


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l1748_174844

-- Define the edge lengths
def cube1_edge_inches : ℝ := 4
def cube2_edge_feet : ℝ := 2

-- Define the conversion factor from feet to inches
def feet_to_inches : ℝ := 12

-- Theorem statement
theorem cube_volume_ratio :
  (cube1_edge_inches ^ 3) / ((cube2_edge_feet * feet_to_inches) ^ 3) = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l1748_174844


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l1748_174869

theorem intersection_line_of_circles (x y : ℝ) : 
  (x^2 + y^2 - 4*x = 0 ∧ x^2 + y^2 + 4*y = 0) → y = -x := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l1748_174869


namespace NUMINAMATH_CALUDE_heroes_on_front_l1748_174861

theorem heroes_on_front (total : ℕ) (on_back : ℕ) (on_front : ℕ) : 
  total = 15 → on_back = 9 → on_front = total - on_back → on_front = 6 := by
  sorry

end NUMINAMATH_CALUDE_heroes_on_front_l1748_174861


namespace NUMINAMATH_CALUDE_goldfish_equality_month_l1748_174885

theorem goldfish_equality_month : ∃ (n : ℕ), n > 0 ∧ 3 * 5^n = 243 * 3^n ∧ ∀ (m : ℕ), m > 0 → m < n → 3 * 5^m ≠ 243 * 3^m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_goldfish_equality_month_l1748_174885


namespace NUMINAMATH_CALUDE_sheep_in_wilderness_l1748_174880

/-- Given that 90% of sheep are in a pen and there are 81 sheep in the pen,
    prove that there are 9 sheep in the wilderness. -/
theorem sheep_in_wilderness (total : ℕ) (in_pen : ℕ) (h1 : in_pen = 81) 
    (h2 : in_pen = (90 : ℕ) * total / 100) : total - in_pen = 9 := by
  sorry

end NUMINAMATH_CALUDE_sheep_in_wilderness_l1748_174880


namespace NUMINAMATH_CALUDE_unique_square_difference_l1748_174839

theorem unique_square_difference (n : ℕ) : 
  (∃ k m : ℕ, n + 30 = k^2 ∧ n - 17 = m^2) ↔ n = 546 := by
sorry

end NUMINAMATH_CALUDE_unique_square_difference_l1748_174839


namespace NUMINAMATH_CALUDE_rectangle_area_l1748_174805

theorem rectangle_area (breadth : ℝ) (h1 : breadth > 0) : 
  let length := 3 * breadth
  let perimeter := 2 * (length + breadth)
  perimeter = 48 → breadth * length = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1748_174805


namespace NUMINAMATH_CALUDE_expression_value_at_three_l1748_174817

theorem expression_value_at_three :
  let x : ℕ := 3
  x + x * (x ^ x) + x ^ 3 = 111 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l1748_174817


namespace NUMINAMATH_CALUDE_tax_rate_percentage_l1748_174867

/-- A tax rate of $65 per $100.00 is equivalent to 65% -/
theorem tax_rate_percentage : 
  let tax_amount : ℚ := 65
  let base_amount : ℚ := 100
  (tax_amount / base_amount) * 100 = 65 := by sorry

end NUMINAMATH_CALUDE_tax_rate_percentage_l1748_174867


namespace NUMINAMATH_CALUDE_sum_positive_given_difference_abs_l1748_174877

theorem sum_positive_given_difference_abs (a b : ℝ) : a - |b| > 0 → b + a > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_given_difference_abs_l1748_174877


namespace NUMINAMATH_CALUDE_factor_expression_l1748_174887

theorem factor_expression (b : ℝ) : 180 * b^2 + 36 * b = 36 * b * (5 * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1748_174887


namespace NUMINAMATH_CALUDE_infinitely_many_satisfying_inequality_l1748_174820

/-- Given a sequence of positive real numbers, prove that there are infinitely many natural numbers
    satisfying the inequality involving the first term, the nth term, and the (n+1)th term. -/
theorem infinitely_many_satisfying_inequality (a : ℕ → ℝ) (h : ∀ n, a n > 0) :
  Set.Infinite {n : ℕ | (a 1 + a (n + 1)) / a n > 1 + 1 / n} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_satisfying_inequality_l1748_174820


namespace NUMINAMATH_CALUDE_min_value_of_a_l1748_174838

theorem min_value_of_a (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 9) ↔ a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l1748_174838
