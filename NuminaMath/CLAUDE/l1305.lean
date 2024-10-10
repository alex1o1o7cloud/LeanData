import Mathlib

namespace katie_has_more_games_l1305_130513

/-- The number of games Katie has -/
def katie_games : ℕ := 81

/-- The number of games Katie's friends have -/
def friends_games : ℕ := 59

/-- The difference in games between Katie and her friends -/
def game_difference : ℕ := katie_games - friends_games

theorem katie_has_more_games : game_difference = 22 := by
  sorry

end katie_has_more_games_l1305_130513


namespace compound_ratio_example_l1305_130555

-- Define the compound ratio function
def compound_ratio (a b c d e f g h : ℚ) : ℚ := (a * c * e * g) / (b * d * f * h)

-- State the theorem
theorem compound_ratio_example : compound_ratio 2 3 6 7 1 3 3 8 = 1 / 14 := by
  sorry

end compound_ratio_example_l1305_130555


namespace largest_quantity_l1305_130596

theorem largest_quantity (a b c d : ℝ) (h : a + 1 = b - 3 ∧ a + 1 = c + 4 ∧ a + 1 = d - 2) :
  b ≥ a ∧ b ≥ c ∧ b ≥ d :=
by sorry

end largest_quantity_l1305_130596


namespace b_bounded_a_value_l1305_130500

/-- A quadratic function f(x) = ax^2 + bx + c with certain properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  bounded : ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a * x^2 + b * x + c| ≤ 1

/-- The coefficient b of a QuadraticFunction is bounded by 1 -/
theorem b_bounded (f : QuadraticFunction) : |f.b| ≤ 1 := by sorry

/-- If f(0) = -1 and f(1) = 1, then a = 2 -/
theorem a_value (f : QuadraticFunction) 
  (h0 : f.c = -1) 
  (h1 : f.a + f.b + f.c = 1) : 
  f.a = 2 := by sorry

end b_bounded_a_value_l1305_130500


namespace equation_solution_l1305_130564

theorem equation_solution : ∃! x : ℚ, (3*x^2 + 2*x + 1) / (x - 1) = 3*x + 1 ∧ x = -1/2 := by
  sorry

end equation_solution_l1305_130564


namespace ellipse_circle_intersection_l1305_130542

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  25 * x^2 + 36 * y^2 = 900

noncomputable def point_M : ℝ × ℝ := (4.8, Real.sqrt (900 / 36 - 25 * 4.8^2 / 36))

noncomputable def tangent_line (x y : ℝ) : Prop :=
  25 * 4.8 * x + 36 * point_M.2 * y = 900

noncomputable def point_N : ℝ × ℝ := (0, 900 / (36 * point_M.2))

noncomputable def circle_equation (x y : ℝ) : Prop :=
  x^2 + (y - (263 / 75))^2 = (362 / 75)^2

theorem ellipse_circle_intersection :
  ∀ x y : ℝ,
  ellipse_equation x y ∧ circle_equation x y ∧ y = 0 →
  x = Real.sqrt 11 ∨ x = -Real.sqrt 11 :=
sorry

end ellipse_circle_intersection_l1305_130542


namespace f_g_f_1_equals_102_l1305_130504

def f (x : ℝ) : ℝ := 5 * x + 2

def g (x : ℝ) : ℝ := 3 * x - 1

theorem f_g_f_1_equals_102 : f (g (f 1)) = 102 := by
  sorry

end f_g_f_1_equals_102_l1305_130504


namespace average_running_time_l1305_130562

theorem average_running_time (f : ℕ) : 
  let third_graders := 9 * f
  let fourth_graders := 3 * f
  let fifth_graders := f
  let total_students := third_graders + fourth_graders + fifth_graders
  let total_minutes := 10 * third_graders + 18 * fourth_graders + 12 * fifth_graders
  (total_minutes : ℚ) / total_students = 12 := by
sorry

end average_running_time_l1305_130562


namespace g_at_zero_l1305_130552

-- Define polynomials f, g, and h
variable (f g h : ℝ[X])

-- Define the relationship between h, f, and g
axiom h_eq_f_mul_g : h = f * g

-- Define the constant term of f
axiom f_constant_term : f.coeff 0 = 5

-- Define the constant term of h
axiom h_constant_term : h.coeff 0 = -10

-- Theorem to prove
theorem g_at_zero : g.eval 0 = -2 := by sorry

end g_at_zero_l1305_130552


namespace perimeter_ABCDE_l1305_130593

-- Define the points as 2D vectors
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom AB_eq_BC : dist A B = dist B C
axiom AB_eq_4 : dist A B = 4
axiom AE_eq_5 : dist A E = 5
axiom ED_eq_8 : dist E D = 8
axiom right_angle_AEB : (B.1 - E.1) * (A.2 - E.2) = (A.1 - E.1) * (B.2 - E.2)
axiom right_angle_BAE : (E.1 - A.1) * (B.2 - A.2) = (B.1 - A.1) * (E.2 - A.2)
axiom right_angle_ABC : (C.1 - B.1) * (A.2 - B.2) = (A.1 - B.1) * (C.2 - B.2)

-- Define the perimeter function
def perimeter (A B C D E : ℝ × ℝ) : ℝ :=
  dist A B + dist B C + dist C D + dist D E + dist E A

-- State the theorem
theorem perimeter_ABCDE :
  perimeter A B C D E = 21 + Real.sqrt 17 := by sorry

end perimeter_ABCDE_l1305_130593


namespace remaining_time_is_three_and_half_l1305_130536

/-- The time taken for Cameron and Sandra to complete the remaining task -/
def remaining_time (cameron_rate : ℚ) (combined_rate : ℚ) (cameron_solo_days : ℚ) : ℚ :=
  (1 - cameron_rate * cameron_solo_days) / combined_rate

/-- Theorem stating the remaining time is 3.5 days -/
theorem remaining_time_is_three_and_half :
  remaining_time (1/18) (1/7) 9 = 7/2 := by
  sorry

end remaining_time_is_three_and_half_l1305_130536


namespace function_value_at_pi_over_four_l1305_130565

/-- Given a function f where f(x) = f'(π/4) * cos(x) + sin(x), prove that f(π/4) = 1 -/
theorem function_value_at_pi_over_four (f : ℝ → ℝ) 
  (h : ∀ x, f x = (deriv f (π/4)) * Real.cos x + Real.sin x) : 
  f (π/4) = 1 := by
  sorry

end function_value_at_pi_over_four_l1305_130565


namespace polynomial_divisibility_l1305_130583

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℤ, (x - 2) * (x + 1) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x - 8)) →
  p = -1 ∧ q = -10 := by
sorry

end polynomial_divisibility_l1305_130583


namespace rational_expression_equality_algebraic_expression_equality_l1305_130574

/-- Prove the equality of the given rational expression -/
theorem rational_expression_equality (m : ℝ) (hm1 : m ≠ -4) (hm2 : m ≠ -2) : 
  (m^2 - 16) / (m^2 + 8*m + 16) / ((m - 4) / (2*m + 8)) * ((m - 2) / (m + 2)) = 2*(m - 2) / (m + 2) := by
  sorry

/-- Prove the equality of the given algebraic expression -/
theorem algebraic_expression_equality (x : ℝ) (hx1 : x ≠ -2) (hx2 : x ≠ 2) : 
  3 / (x + 2) + 1 / (2 - x) - (2*x) / (4 - x^2) = 4 / (x + 2) := by
  sorry

end rational_expression_equality_algebraic_expression_equality_l1305_130574


namespace danny_bottle_caps_indeterminate_l1305_130569

/-- Represents Danny's collection of bottle caps and wrappers --/
structure Collection where
  bottle_caps : ℕ
  wrappers : ℕ

/-- The problem statement --/
theorem danny_bottle_caps_indeterminate 
  (initial : Collection) 
  (park_found : Collection)
  (final_wrappers : ℕ) :
  park_found.bottle_caps = 22 →
  park_found.wrappers = 30 →
  final_wrappers = initial.wrappers + park_found.wrappers →
  final_wrappers = 57 →
  ¬∃ (n : ℕ), ∀ (x : ℕ), initial.bottle_caps = n ∨ initial.bottle_caps ≠ x :=
by sorry

end danny_bottle_caps_indeterminate_l1305_130569


namespace hyperbola_triangle_area_l1305_130544

/-- Given a right triangle OAB with O at the origin, this structure represents the hyperbola
    y = k/x passing through the midpoint of OB and intersecting AB at C. -/
structure Hyperbola_Triangle :=
  (a b : ℝ)  -- Coordinates of point B (a, b)
  (k : ℝ)    -- Parameter of the hyperbola y = k/x
  (h_k_pos : k > 0)  -- k is positive
  (h_right_triangle : a * b = 2 * 3)  -- Area of OAB is 3, so a * b / 2 = 3
  (h_midpoint : k / (a/2) = b/2)  -- Hyperbola passes through midpoint of OB
  (c : ℝ)    -- x-coordinate of point C
  (h_c_on_ab : 0 < c ∧ c < a)  -- C is between O and B on AB
  (h_c_on_hyperbola : k / c = b * (1 - c/a))  -- C is on the hyperbola

/-- The main theorem: if the area of OBC is 3, then k = 2 -/
theorem hyperbola_triangle_area (ht : Hyperbola_Triangle) 
  (h_area_obc : ht.a * ht.b * (1 - ht.c/ht.a) / 2 = 3) : ht.k = 2 := by
  sorry

end hyperbola_triangle_area_l1305_130544


namespace tutor_schedule_lcm_l1305_130526

theorem tutor_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 9 (Nat.lcm 8 10))) = 360 := by
  sorry

end tutor_schedule_lcm_l1305_130526


namespace chalkboard_area_l1305_130531

/-- The area of a rectangle with width 3.5 feet and length 2.3 times its width is 28.175 square feet. -/
theorem chalkboard_area : 
  let width : ℝ := 3.5
  let length : ℝ := 2.3 * width
  width * length = 28.175 := by sorry

end chalkboard_area_l1305_130531


namespace problem_statement_l1305_130543

theorem problem_statement (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) :
  let expr1 := (1 + a*b)/(a - b) * (1 + b*c)/(b - c) + 
               (1 + b*c)/(b - c) * (1 + c*a)/(c - a) + 
               (1 + c*a)/(c - a) * (1 + a*b)/(a - b)
  let expr2 := (1 - a*b)/(a - b) * (1 - b*c)/(b - c) + 
               (1 - b*c)/(b - c) * (1 - c*a)/(c - a) + 
               (1 - c*a)/(c - a) * (1 - a*b)/(a - b)
  let expr3 := (1 + a^2*b^2)/(a - b)^2 + (1 + b^2*c^2)/(b - c)^2 + (1 + c^2*a^2)/(c - a)^2
  (expr1 = 1) ∧ 
  (expr2 = -1) ∧ 
  (expr3 ≥ (3/2)) ∧ 
  (expr3 = (3/2) ↔ a = b ∨ b = c ∨ c = a) := by
  sorry

end problem_statement_l1305_130543


namespace sisters_portions_l1305_130576

/-- Represents the types of granola bars --/
inductive BarType
  | ChocolateChip
  | OatAndHoney
  | PeanutButter

/-- Represents the number of bars of each type --/
structure BarCounts where
  chocolateChip : ℕ
  oatAndHoney : ℕ
  peanutButter : ℕ

/-- Represents the initial distribution of bars --/
def initialDistribution : BarCounts :=
  { chocolateChip := 8, oatAndHoney := 6, peanutButter := 6 }

/-- Represents the bars set aside daily --/
def dailySetAside : BarCounts :=
  { chocolateChip := 3, oatAndHoney := 2, peanutButter := 2 }

/-- Represents the bars left after setting aside --/
def afterSetAside : BarCounts :=
  { chocolateChip := 5, oatAndHoney := 4, peanutButter := 4 }

/-- Represents the bars traded --/
def traded : BarCounts :=
  { chocolateChip := 2, oatAndHoney := 4, peanutButter := 0 }

/-- Represents the bars left after trading --/
def afterTrading : BarCounts :=
  { chocolateChip := 3, oatAndHoney := 0, peanutButter := 3 }

/-- Represents the whole bars given to each sister --/
def wholeBarsGiven (type : BarType) : ℕ := 2

/-- Theorem: Each sister receives 2.5 portions of their preferred granola bar type --/
theorem sisters_portions (type : BarType) : 
  (wholeBarsGiven type : ℚ) + (1 : ℚ) / 2 = (5 : ℚ) / 2 := by
  sorry

end sisters_portions_l1305_130576


namespace max_product_of_digits_divisible_by_25_l1305_130589

theorem max_product_of_digits_divisible_by_25 (a b : Nat) : 
  a ≤ 9 →
  b ≤ 9 →
  (10 * a + b) % 25 = 0 →
  b * a ≤ 35 := by
  sorry

end max_product_of_digits_divisible_by_25_l1305_130589


namespace geometric_sequence_sum_condition_l1305_130541

/-- Theorem: For an infinite geometric sequence with first term a₁ and common ratio q,
    if the sum of the sequence is 1/2, then 2a₁ + q = 1. -/
theorem geometric_sequence_sum_condition (a₁ q : ℝ) (h : |q| < 1) :
  (∑' n, a₁ * q ^ (n - 1) = 1/2) → 2 * a₁ + q = 1 := by
  sorry

end geometric_sequence_sum_condition_l1305_130541


namespace largest_hexagon_angle_l1305_130534

/-- Represents the angles of a hexagon -/
structure HexagonAngles where
  a₁ : ℝ
  a₂ : ℝ
  a₃ : ℝ
  a₄ : ℝ
  a₅ : ℝ
  a₆ : ℝ

/-- The sum of angles in a hexagon is 720 degrees -/
axiom hexagon_angle_sum (h : HexagonAngles) : h.a₁ + h.a₂ + h.a₃ + h.a₄ + h.a₅ + h.a₆ = 720

/-- The angles of the hexagon are in the ratio 3:3:3:4:5:6 -/
def hexagon_angle_ratio (h : HexagonAngles) : Prop :=
  ∃ x : ℝ, h.a₁ = 3*x ∧ h.a₂ = 3*x ∧ h.a₃ = 3*x ∧ h.a₄ = 4*x ∧ h.a₅ = 5*x ∧ h.a₆ = 6*x

/-- The largest angle in the hexagon is 180 degrees -/
theorem largest_hexagon_angle (h : HexagonAngles) 
  (ratio : hexagon_angle_ratio h) : h.a₆ = 180 := by
  sorry

end largest_hexagon_angle_l1305_130534


namespace tenths_place_of_five_twelfths_l1305_130563

theorem tenths_place_of_five_twelfths (ε : ℚ) : 
  ε = 5 / 12 → 
  ∃ (n : ℕ) (r : ℚ), ε = (4 : ℚ) / 10 + n / 100 + r ∧ 0 ≤ r ∧ r < 1 / 100 :=
sorry

end tenths_place_of_five_twelfths_l1305_130563


namespace inspection_probability_l1305_130566

theorem inspection_probability (pass_rate1 pass_rate2 : ℝ) 
  (h1 : pass_rate1 = 0.90)
  (h2 : pass_rate2 = 0.95) :
  let fail_rate1 := 1 - pass_rate1
  let fail_rate2 := 1 - pass_rate2
  pass_rate1 * fail_rate2 + fail_rate1 * pass_rate2 = 0.14 :=
by sorry

end inspection_probability_l1305_130566


namespace line_intersects_circle_l1305_130530

/-- Given a point M(a, b) outside the unit circle, prove that the line ax + by = 1 intersects the circle -/
theorem line_intersects_circle (a b : ℝ) (h : a^2 + b^2 > 1) :
  ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a * x + b * y = 1 := by
  sorry

end line_intersects_circle_l1305_130530


namespace three_brothers_selection_probability_l1305_130591

theorem three_brothers_selection_probability 
  (p_ram : ℚ) (p_ravi : ℚ) (p_raj : ℚ) 
  (h_ram : p_ram = 2/7)
  (h_ravi : p_ravi = 1/5)
  (h_raj : p_raj = 3/8) :
  p_ram * p_ravi * p_raj = 3/140 := by
sorry

end three_brothers_selection_probability_l1305_130591


namespace coefficient_x3_equals_neg16_l1305_130580

/-- The coefficient of x^3 in the expansion of (1-ax)^2(1+x)^6 -/
def coefficient_x3 (a : ℝ) : ℝ :=
  20 - 30*a + 6*a^2

theorem coefficient_x3_equals_neg16 (a : ℝ) :
  coefficient_x3 a = -16 → a = 2 ∨ a = 3 := by
  sorry

end coefficient_x3_equals_neg16_l1305_130580


namespace seventeen_integer_chords_l1305_130556

/-- Represents a circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distanceToP : ℝ

/-- Counts the number of integer-length chords containing P in the given circle -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem seventeen_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 13) 
  (h2 : c.distanceToP = 12) : 
  countIntegerChords c = 17 :=
sorry

end seventeen_integer_chords_l1305_130556


namespace rectangle_square_overlap_ratio_l1305_130592

/-- Rectangle represented by its side lengths -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Square represented by its side length -/
structure Square where
  side : ℝ

/-- The overlapping area between a rectangle and a square -/
def overlap_area (r : Rectangle) (s : Square) : ℝ := sorry

theorem rectangle_square_overlap_ratio :
  ∀ (r : Rectangle) (s : Square),
    overlap_area r s = 0.4 * r.length * r.width →
    overlap_area r s = 0.25 * s.side * s.side →
    r.length / r.width = 2 / 5 := by
  sorry

end rectangle_square_overlap_ratio_l1305_130592


namespace independence_test_most_appropriate_l1305_130571

/-- Represents the survey data --/
structure SurveyData where
  male_total : Nat
  male_opposing : Nat
  female_total : Nat
  female_opposing : Nat

/-- Represents different statistical methods --/
inductive StatisticalMethod
  | MeanAndVariance
  | RegressionLine
  | IndependenceTest
  | Probability

/-- Determines the most appropriate method for analyzing the relationship between gender and judgment --/
def most_appropriate_method (data : SurveyData) : StatisticalMethod :=
  StatisticalMethod.IndependenceTest

/-- Theorem stating that the Independence test is the most appropriate method for the given survey data --/
theorem independence_test_most_appropriate (data : SurveyData) :
  most_appropriate_method data = StatisticalMethod.IndependenceTest :=
sorry

end independence_test_most_appropriate_l1305_130571


namespace conic_is_hyperbola_l1305_130508

/-- The equation of a conic section -/
def conicEquation (x y : ℝ) : Prop :=
  (x - 3)^2 = 4 * (y + 2)^2 + 25

/-- Definition of a hyperbola -/
def isHyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
    ∀ x y, f x y ↔ a * x^2 + b * y^2 + c * x + d * y + e = 0

/-- Theorem: The given conic equation represents a hyperbola -/
theorem conic_is_hyperbola : isHyperbola conicEquation :=
sorry

end conic_is_hyperbola_l1305_130508


namespace y_derivative_l1305_130568

noncomputable def y (x : ℝ) : ℝ :=
  (Real.cos x) / (3 * (2 + Real.sin x)) + (4 / (3 * Real.sqrt 3)) * Real.arctan ((2 * Real.tan (x / 2) + 1) / Real.sqrt 3)

theorem y_derivative (x : ℝ) :
  deriv y x = (2 * Real.sin x + 7) / (3 * (2 + Real.sin x)^2) :=
sorry

end y_derivative_l1305_130568


namespace room_length_proof_l1305_130560

/-- Proves that a room with given width, paving cost per area, and total paving cost has a specific length -/
theorem room_length_proof (width : ℝ) (cost_per_area : ℝ) (total_cost : ℝ) :
  width = 3.75 →
  cost_per_area = 800 →
  total_cost = 16500 →
  (total_cost / cost_per_area) / width = 5.5 := by
  sorry

#check room_length_proof

end room_length_proof_l1305_130560


namespace simple_interest_years_l1305_130549

/-- Given a principal amount and the additional interest earned from a 1% rate increase,
    calculate the number of years the sum was put at simple interest. -/
theorem simple_interest_years (principal : ℝ) (additional_interest : ℝ) : 
  principal = 2400 →
  additional_interest = 72 →
  (principal * 0.01 * (3 : ℝ)) = additional_interest :=
by sorry

end simple_interest_years_l1305_130549


namespace F_minimum_at_negative_one_F_monotonic_intervals_t_ge_3_F_monotonic_intervals_t_between_F_monotonic_intervals_t_le_neg_one_l1305_130561

-- Define the function F(x, t)
def F (x t : ℝ) : ℝ := |2*x + t| + x^2 + x + 1

-- Theorem for the minimum value when t = -1
theorem F_minimum_at_negative_one :
  ∃ (x_min : ℝ), F x_min (-1) = 7/4 ∧ ∀ (x : ℝ), F x (-1) ≥ 7/4 :=
sorry

-- Theorems for monotonic intervals
theorem F_monotonic_intervals_t_ge_3 (t : ℝ) (h : t ≥ 3) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ -3/2 → F x t ≥ F y t) ∧
  (∀ x y : ℝ, -3/2 ≤ x ∧ x ≤ y → F x t ≤ F y t) :=
sorry

theorem F_monotonic_intervals_t_between (t : ℝ) (h1 : -1 < t) (h2 : t < 3) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ -t/2 → F x t ≥ F y t) ∧
  (∀ x y : ℝ, -t/2 ≤ x ∧ x ≤ y → F x t ≤ F y t) :=
sorry

theorem F_monotonic_intervals_t_le_neg_one (t : ℝ) (h : t ≤ -1) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 1/2 → F x t ≥ F y t) ∧
  (∀ x y : ℝ, 1/2 ≤ x ∧ x ≤ y → F x t ≤ F y t) :=
sorry

end F_minimum_at_negative_one_F_monotonic_intervals_t_ge_3_F_monotonic_intervals_t_between_F_monotonic_intervals_t_le_neg_one_l1305_130561


namespace sum_of_coefficients_l1305_130506

/-- Given a polynomial equation, prove the sum of specific coefficients -/
theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = 
    a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + 
    a₉*(x+1)^9 + a₁₀*(x+1)^10 + a₁₁*(x+1)^11) →
  a₁ + a₂ + a₁₁ = 781 := by
sorry

end sum_of_coefficients_l1305_130506


namespace cube_cross_sections_l1305_130523

/-- A regular polygon obtained by cutting a cube with a plane. -/
inductive CubeCrossSection
  | Triangle
  | Square
  | Hexagon

/-- The set of all possible regular polygons obtained by cutting a cube with a plane. -/
def ValidCubeCrossSections : Set CubeCrossSection :=
  {CubeCrossSection.Triangle, CubeCrossSection.Square, CubeCrossSection.Hexagon}

/-- Theorem: The only regular polygons that can be obtained by cutting a cube with a plane
    are triangles, squares, and hexagons. -/
theorem cube_cross_sections (cs : CubeCrossSection) :
  cs ∈ ValidCubeCrossSections := by sorry

end cube_cross_sections_l1305_130523


namespace unique_valid_n_l1305_130585

def is_valid_n (n : ℕ+) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ : ℕ+),
    d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧
    (∀ (d : ℕ+), d ∣ n → d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄ ∨ d₄ < d) ∧
    n = d₁^2 + d₂^2 + d₃^2 + d₄^2

theorem unique_valid_n :
  ∃! (n : ℕ+), is_valid_n n ∧ n = 130 := by sorry

end unique_valid_n_l1305_130585


namespace probability_at_least_six_fives_value_l1305_130539

/-- The probability of rolling at least a five on a fair die -/
def p_at_least_five : ℚ := 1/3

/-- The probability of not rolling at least a five on a fair die -/
def p_not_at_least_five : ℚ := 2/3

/-- The number of rolls -/
def num_rolls : ℕ := 8

/-- The minimum number of successful rolls (at least a five) -/
def min_success : ℕ := 6

/-- Calculates the probability of exactly k successes in n trials -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1-p)^(n-k)

/-- The probability of rolling at least a five at least six times in eight rolls -/
def probability_at_least_six_fives : ℚ :=
  binomial_probability num_rolls min_success p_at_least_five +
  binomial_probability num_rolls (min_success + 1) p_at_least_five +
  binomial_probability num_rolls (min_success + 2) p_at_least_five

theorem probability_at_least_six_fives_value :
  probability_at_least_six_fives = 129/6561 := by sorry

end probability_at_least_six_fives_value_l1305_130539


namespace production_exceeds_target_in_seventh_year_l1305_130575

-- Define the initial production and growth rate
def initial_production : ℝ := 40000
def growth_rate : ℝ := 1.2

-- Define the target production
def target_production : ℝ := 120000

-- Define the function to calculate production after n years
def production (n : ℕ) : ℝ := initial_production * growth_rate ^ n

-- Theorem statement
theorem production_exceeds_target_in_seventh_year :
  ∀ n : ℕ, n < 7 → production n ≤ target_production ∧
  production 7 > target_production :=
sorry

end production_exceeds_target_in_seventh_year_l1305_130575


namespace total_snow_volume_l1305_130597

/-- Calculates the total volume of snow on two sidewalk sections -/
theorem total_snow_volume 
  (length1 width1 depth1 : ℝ)
  (length2 width2 depth2 : ℝ)
  (h1 : length1 = 30)
  (h2 : width1 = 3)
  (h3 : depth1 = 1)
  (h4 : length2 = 15)
  (h5 : width2 = 2)
  (h6 : depth2 = 1/2) :
  length1 * width1 * depth1 + length2 * width2 * depth2 = 105 := by
sorry

end total_snow_volume_l1305_130597


namespace pencils_added_l1305_130511

theorem pencils_added (initial_pencils final_pencils : ℕ) (h1 : initial_pencils = 41) (h2 : final_pencils = 71) :
  final_pencils - initial_pencils = 30 := by
  sorry

end pencils_added_l1305_130511


namespace shadow_problem_l1305_130598

/-- Given a cube with edge length 2 cm and a light source y cm above an upper vertex
    casting a shadow with area 175 sq cm (excluding the area beneath the cube),
    prove that the greatest integer less than or equal to 100y is 333. -/
theorem shadow_problem (y : ℝ) : 
  (2 : ℝ) > 0 ∧ 
  y > 0 ∧ 
  175 = (Real.sqrt 179 - 2)^2 →
  ⌊100 * y⌋ = 333 := by
  sorry

end shadow_problem_l1305_130598


namespace smallest_cookie_count_l1305_130520

theorem smallest_cookie_count : ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → 4*m - 4 = (m^2)/2 → m ≥ n) ∧ 4*n - 4 = (n^2)/2 ∧ n^2 = 49 := by
  sorry

end smallest_cookie_count_l1305_130520


namespace modular_congruence_l1305_130540

theorem modular_congruence (n : ℕ) : 
  0 ≤ n ∧ n < 37 ∧ (5 * n) % 37 = 1 → 
  (((2^n)^3 - 2) % 37 = 1) := by
  sorry

end modular_congruence_l1305_130540


namespace favorite_movies_total_duration_l1305_130512

/-- Given the durations of four people's favorite movies with specific relationships,
    prove that the total duration of all movies is 76 hours. -/
theorem favorite_movies_total_duration
  (michael_duration : ℝ)
  (joyce_duration : ℝ)
  (nikki_duration : ℝ)
  (ryn_duration : ℝ)
  (h1 : joyce_duration = michael_duration + 2)
  (h2 : nikki_duration = 3 * michael_duration)
  (h3 : ryn_duration = 4/5 * nikki_duration)
  (h4 : nikki_duration = 30) :
  joyce_duration + michael_duration + nikki_duration + ryn_duration = 76 := by
sorry


end favorite_movies_total_duration_l1305_130512


namespace remaining_sugar_is_one_l1305_130578

/-- Represents the recipe and Mary's baking process -/
structure Recipe where
  sugar_total : ℕ
  salt_total : ℕ
  flour_added : ℕ
  sugar_salt_diff : ℕ

/-- Calculates the remaining sugar to be added based on the recipe and current state -/
def remaining_sugar (r : Recipe) : ℕ :=
  r.sugar_total - (r.salt_total + r.sugar_salt_diff)

/-- Theorem stating that the remaining sugar to be added is 1 cup -/
theorem remaining_sugar_is_one (r : Recipe) 
  (h1 : r.sugar_total = 8)
  (h2 : r.salt_total = 7)
  (h3 : r.flour_added = 5)
  (h4 : r.sugar_salt_diff = 1) : 
  remaining_sugar r = 1 := by
  sorry

#check remaining_sugar_is_one

end remaining_sugar_is_one_l1305_130578


namespace weight_of_a_l1305_130551

theorem weight_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 3 →
  (b + c + d + e) / 4 = 79 →
  a = 75 := by
sorry

end weight_of_a_l1305_130551


namespace rectangle_square_overlap_ratio_l1305_130582

/-- Given a rectangle ABCD and a square EFGH, if 30% of the rectangle's area
    overlaps with the square, and 25% of the square's area overlaps with the rectangle,
    then the ratio of the rectangle's length to its width is 7.5. -/
theorem rectangle_square_overlap_ratio :
  ∀ (l w s : ℝ),
    l > 0 → w > 0 → s > 0 →
    (0.3 * l * w = 0.25 * s^2) →
    (l / w = 7.5) :=
by sorry

end rectangle_square_overlap_ratio_l1305_130582


namespace orthogonal_vectors_l1305_130577

theorem orthogonal_vectors (y : ℝ) : 
  (2 * -3 + -4 * y + 5 * 2 = 0) ↔ (y = 1) :=
by sorry

end orthogonal_vectors_l1305_130577


namespace tablet_cash_savings_l1305_130595

/-- Represents the cost and payment structure for a tablet purchase -/
structure TabletPurchase where
  cash_price : ℕ
  down_payment : ℕ
  first_four_months_payment : ℕ
  middle_four_months_payment : ℕ
  last_four_months_payment : ℕ

/-- Calculates the total amount paid through installments -/
def total_installment_cost (tp : TabletPurchase) : ℕ :=
  tp.down_payment + 4 * tp.first_four_months_payment + 4 * tp.middle_four_months_payment + 4 * tp.last_four_months_payment

/-- Calculates the savings when buying in cash versus installments -/
def cash_savings (tp : TabletPurchase) : ℕ :=
  total_installment_cost tp - tp.cash_price

/-- Theorem stating the savings when buying the tablet in cash -/
theorem tablet_cash_savings :
  ∃ (tp : TabletPurchase),
    tp.cash_price = 450 ∧
    tp.down_payment = 100 ∧
    tp.first_four_months_payment = 40 ∧
    tp.middle_four_months_payment = 35 ∧
    tp.last_four_months_payment = 30 ∧
    cash_savings tp = 70 := by
  sorry

end tablet_cash_savings_l1305_130595


namespace not_prime_qt_plus_q_plus_t_l1305_130518

theorem not_prime_qt_plus_q_plus_t (q t : ℕ+) (h : q > 1 ∨ t > 1) : 
  ¬ Nat.Prime (q * t + q + t) := by
sorry

end not_prime_qt_plus_q_plus_t_l1305_130518


namespace rotated_angle_measure_l1305_130527

/-- 
Given an initial angle of 40 degrees that is rotated 480 degrees clockwise,
the resulting acute angle measures 80 degrees.
-/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 40 →
  rotation = 480 →
  (rotation % 360 - initial_angle) % 180 = 80 :=
by sorry

end rotated_angle_measure_l1305_130527


namespace beaker_volume_difference_l1305_130537

theorem beaker_volume_difference (total_volume : ℝ) (beaker_one_volume : ℝ) 
  (h1 : total_volume = 9.28)
  (h2 : beaker_one_volume = 2.95) : 
  abs (beaker_one_volume - (total_volume - beaker_one_volume)) = 3.38 := by
  sorry

end beaker_volume_difference_l1305_130537


namespace old_lamp_height_is_one_foot_l1305_130514

-- Define the height of the new lamp
def new_lamp_height : ℝ := 2.3333333333333335

-- Define the difference in height between the new and old lamps
def height_difference : ℝ := 1.3333333333333333

-- Theorem to prove
theorem old_lamp_height_is_one_foot :
  new_lamp_height - height_difference = 1 := by sorry

end old_lamp_height_is_one_foot_l1305_130514


namespace variable_value_proof_l1305_130559

theorem variable_value_proof : ∃ x : ℝ, 3 * x + 36 = 48 ∧ x = 4 := by
  sorry

end variable_value_proof_l1305_130559


namespace baseball_team_selection_l1305_130594

theorem baseball_team_selection (total_players : Nat) (selected_players : Nat) (twins : Nat) :
  total_players = 16 →
  selected_players = 9 →
  twins = 2 →
  Nat.choose (total_players - twins) (selected_players - twins) = 3432 := by
sorry

end baseball_team_selection_l1305_130594


namespace difference_of_squares_l1305_130584

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 8) : x^2 - y^2 = 80 := by
  sorry

end difference_of_squares_l1305_130584


namespace sum_s_r_x_is_negative_fifteen_l1305_130599

def r (x : ℝ) : ℝ := |x| - 3
def s (x : ℝ) : ℝ := -|x|

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_s_r_x_is_negative_fifteen :
  (x_values.map (λ x => s (r x))).sum = -15 := by sorry

end sum_s_r_x_is_negative_fifteen_l1305_130599


namespace intersection_point_y_axis_l1305_130538

/-- The intersection point of a line with the y-axis -/
def y_axis_intersection (m a : ℝ) : ℝ × ℝ := (0, a)

/-- The line equation y = mx + b -/
def line_equation (m b : ℝ) (x : ℝ) : ℝ := m * x + b

theorem intersection_point_y_axis :
  let m : ℝ := 2
  let b : ℝ := 2
  y_axis_intersection m b = (0, line_equation m b 0) :=
by sorry

end intersection_point_y_axis_l1305_130538


namespace arithmetic_sequence_sum_l1305_130550

theorem arithmetic_sequence_sum (n : ℕ) (sum : ℕ) (d : ℕ) : 
  n = 4020 →
  d = 2 →
  sum = 10614 →
  ∃ (a : ℕ), 
    (a + (n - 1) * d / 2) * n = sum ∧
    a + (n / 2 - 1) * (2 * d) = 3297 :=
by sorry

end arithmetic_sequence_sum_l1305_130550


namespace production_cost_decrease_rate_l1305_130501

theorem production_cost_decrease_rate : ∃ x : ℝ, 
  (400 * (1 - x)^2 = 361) ∧ (x = 0.05) := by sorry

end production_cost_decrease_rate_l1305_130501


namespace pairings_equal_twenty_l1305_130535

/-- The number of items in the first set -/
def set1_size : ℕ := 5

/-- The number of items in the second set -/
def set2_size : ℕ := 4

/-- The total number of possible pairings -/
def total_pairings : ℕ := set1_size * set2_size

/-- Theorem: The total number of possible pairings is 20 -/
theorem pairings_equal_twenty : total_pairings = 20 := by
  sorry

end pairings_equal_twenty_l1305_130535


namespace factorial_difference_l1305_130588

theorem factorial_difference : Nat.factorial 9 - Nat.factorial 8 = 322560 := by
  sorry

end factorial_difference_l1305_130588


namespace different_color_probability_l1305_130519

/-- The set of colors for shorts -/
def shorts_colors : Finset String := {"black", "gold", "silver"}

/-- The set of colors for jerseys -/
def jersey_colors : Finset String := {"black", "white", "gold"}

/-- The probability of selecting different colors for shorts and jerseys -/
theorem different_color_probability : 
  (shorts_colors.card * jersey_colors.card - (shorts_colors ∩ jersey_colors).card) / 
  (shorts_colors.card * jersey_colors.card : ℚ) = 7/9 := by
sorry

end different_color_probability_l1305_130519


namespace gas_cans_volume_l1305_130567

/-- The volume of gas needed to fill a given number of gas cans with a specified capacity. -/
def total_gas_volume (num_cans : ℕ) (can_capacity : ℝ) : ℝ :=
  num_cans * can_capacity

/-- Theorem: The total volume of gas needed to fill 4 gas cans, each with a capacity of 5.0 gallons, is equal to 20.0 gallons. -/
theorem gas_cans_volume :
  total_gas_volume 4 5.0 = 20.0 := by
  sorry

end gas_cans_volume_l1305_130567


namespace investment_ratio_l1305_130524

/-- Represents the business investment scenario -/
structure Investment where
  nandan_amount : ℝ
  nandan_time : ℝ
  krishan_amount : ℝ
  krishan_time : ℝ
  total_gain : ℝ
  nandan_gain : ℝ

/-- The theorem representing the investment problem -/
theorem investment_ratio (i : Investment) 
  (h1 : i.krishan_amount = 4 * i.nandan_amount)
  (h2 : i.total_gain = 26000)
  (h3 : i.nandan_gain = 2000)
  (h4 : ∃ (k : ℝ), i.nandan_gain / i.total_gain = 
       (i.nandan_amount * i.nandan_time) / 
       (i.nandan_amount * i.nandan_time + i.krishan_amount * i.krishan_time)) :
  i.krishan_time / i.nandan_time = 3 := by
  sorry


end investment_ratio_l1305_130524


namespace inequality_property_l1305_130510

theorem inequality_property (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end inequality_property_l1305_130510


namespace two_part_journey_average_speed_l1305_130525

/-- Calculates the average speed of a two-part journey -/
theorem two_part_journey_average_speed 
  (t1 : ℝ) (v1 : ℝ) (t2 : ℝ) (v2 : ℝ) 
  (h1 : t1 = 5) 
  (h2 : v1 = 40) 
  (h3 : t2 = 3) 
  (h4 : v2 = 80) : 
  (t1 * v1 + t2 * v2) / (t1 + t2) = 55 := by
  sorry

#check two_part_journey_average_speed

end two_part_journey_average_speed_l1305_130525


namespace positive_intervals_l1305_130521

theorem positive_intervals (x : ℝ) : (x + 2) * (x - 3) > 0 ↔ x < -2 ∨ x > 3 := by
  sorry

end positive_intervals_l1305_130521


namespace clips_ratio_april_to_may_l1305_130590

def clips_sold_april : ℕ := 48
def total_clips_sold : ℕ := 72

theorem clips_ratio_april_to_may :
  (clips_sold_april : ℚ) / (total_clips_sold - clips_sold_april : ℚ) = 2 / 1 := by
  sorry

end clips_ratio_april_to_may_l1305_130590


namespace cards_rick_keeps_rick_keeps_fifteen_cards_l1305_130529

/-- The number of cards Rick keeps for himself given the initial number of cards and the distribution to others. -/
theorem cards_rick_keeps (initial_cards : ℕ) (cards_to_miguel : ℕ) (num_friends : ℕ) (cards_per_friend : ℕ) (num_sisters : ℕ) (cards_per_sister : ℕ) : ℕ :=
  initial_cards - cards_to_miguel - (num_friends * cards_per_friend) - (num_sisters * cards_per_sister)

/-- Proof that Rick keeps 15 cards for himself -/
theorem rick_keeps_fifteen_cards :
  cards_rick_keeps 130 13 8 12 2 3 = 15 := by
  sorry

end cards_rick_keeps_rick_keeps_fifteen_cards_l1305_130529


namespace exists_claw_count_for_total_time_specific_grooming_problem_l1305_130581

/-- Represents the grooming time for a cat -/
structure GroomingTime where
  clipTime : ℕ -- Time to clip one nail in seconds
  earCleanTime : ℕ -- Time to clean one ear in seconds
  shampooTime : ℕ -- Time to shampoo in seconds

/-- Theorem stating that there exists a number of claws that results in the given total grooming time -/
theorem exists_claw_count_for_total_time 
  (g : GroomingTime) 
  (totalTime : ℕ) : 
  ∃ (clawCount : ℕ), 
    g.clipTime * clawCount + g.earCleanTime * 2 + g.shampooTime * 60 = totalTime :=
by
  sorry

/-- Application of the theorem to the specific problem -/
theorem specific_grooming_problem : 
  ∃ (clawCount : ℕ), 
    10 * clawCount + 90 * 2 + 5 * 60 = 640 :=
by
  sorry

end exists_claw_count_for_total_time_specific_grooming_problem_l1305_130581


namespace prime_has_property_P_infinitely_many_composite_with_property_P_l1305_130528

-- Define property P
def has_property_P (n : ℕ) : Prop :=
  ∀ a : ℕ, a > 0 → (n ∣ a^n - 1) → (n^2 ∣ a^n - 1)

-- Theorem 1: Every prime number has property P
theorem prime_has_property_P (p : ℕ) (hp : Prime p) : has_property_P p := by
  sorry

-- Theorem 2: There are infinitely many composite numbers with property P
theorem infinitely_many_composite_with_property_P :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ ¬Prime m ∧ has_property_P m := by
  sorry

end prime_has_property_P_infinitely_many_composite_with_property_P_l1305_130528


namespace equation_solution_l1305_130505

theorem equation_solution (x : ℚ) : 
  5 * x - 6 = 15 * x + 21 → 3 * (x + 5)^2 = 2523 / 100 := by
  sorry

end equation_solution_l1305_130505


namespace B_necessary_not_sufficient_l1305_130517

def A (x : ℝ) : Prop := 0 < x ∧ x < 5

def B (x : ℝ) : Prop := |x - 2| < 3

theorem B_necessary_not_sufficient :
  (∀ x, A x → B x) ∧ (∃ x, B x ∧ ¬A x) := by
  sorry

end B_necessary_not_sufficient_l1305_130517


namespace total_registration_methods_l1305_130545

-- Define the number of students and clubs
def num_students : Nat := 5
def num_clubs : Nat := 3

-- Define the students with restrictions
structure RestrictedStudent where
  name : String
  restricted_club : Nat

-- Define the list of restricted students
def restricted_students : List RestrictedStudent := [
  { name := "Xiao Bin", restricted_club := 1 },  -- 1 represents chess club
  { name := "Xiao Cong", restricted_club := 0 }, -- 0 represents basketball club
  { name := "Xiao Hao", restricted_club := 2 }   -- 2 represents environmental club
]

-- Define the theorem
theorem total_registration_methods :
  (restricted_students.length * 2 + (num_students - restricted_students.length) * num_clubs) ^ num_students = 72 := by
  sorry

end total_registration_methods_l1305_130545


namespace proposition_equivalence_l1305_130557

theorem proposition_equivalence (m : ℝ) :
  (∃ x : ℝ, -x^2 - 2*m*x + 2*m - 3 ≥ 0) ↔ (m ≤ -3 ∨ m ≥ 1) := by
  sorry

end proposition_equivalence_l1305_130557


namespace coin_array_problem_l1305_130554

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem coin_array_problem :
  ∃ (n : ℕ), triangular_sum n = 3003 ∧ n = 77 ∧ sum_of_digits n = 14 :=
sorry

end coin_array_problem_l1305_130554


namespace parallel_tangents_imply_m_value_monotonicity_intervals_l1305_130558

noncomputable section

variable (m : ℝ)

def f (x : ℝ) : ℝ := (1/2) * m * x^2 + 1

def g (x : ℝ) : ℝ := 2 * Real.log x - (2*m + 1) * x - 1

def h (x : ℝ) : ℝ := f m x + g m x

def h_derivative (x : ℝ) : ℝ := m * x - (2*m + 1) + 2 / x

theorem parallel_tangents_imply_m_value :
  (h_derivative m 1 = h_derivative m 3) → m = 2/3 := by sorry

theorem monotonicity_intervals (x : ℝ) (hx : x > 0) :
  (m ≤ 0 → 
    (x < 2 → h_derivative m x > 0) ∧ 
    (x > 2 → h_derivative m x < 0)) ∧
  (0 < m ∧ m < 1/2 → 
    ((x < 2 ∨ x > 1/m) → h_derivative m x > 0) ∧ 
    (2 < x ∧ x < 1/m → h_derivative m x < 0)) ∧
  (m = 1/2 → h_derivative m x > 0) ∧
  (m > 1/2 → 
    ((x < 1/m ∨ x > 2) → h_derivative m x > 0) ∧ 
    (1/m < x ∧ x < 2 → h_derivative m x < 0)) := by sorry

end

end parallel_tangents_imply_m_value_monotonicity_intervals_l1305_130558


namespace english_score_l1305_130502

theorem english_score (korean math : ℕ) (h1 : (korean + math) / 2 = 88) 
  (h2 : (korean + math + 94) / 3 = 90) : 94 = 94 := by
  sorry

end english_score_l1305_130502


namespace valid_three_digit_count_correct_l1305_130509

/-- The count of valid three-digit numbers -/
def valid_three_digit_count : ℕ := 819

/-- The total count of three-digit numbers -/
def total_three_digit_count : ℕ := 900

/-- The count of invalid three-digit numbers where the hundreds and units digits
    are the same but the tens digit is different -/
def invalid_three_digit_count : ℕ := 81

/-- Theorem stating that the count of valid three-digit numbers is correct -/
theorem valid_three_digit_count_correct :
  valid_three_digit_count = total_three_digit_count - invalid_three_digit_count :=
by sorry

end valid_three_digit_count_correct_l1305_130509


namespace set_operations_l1305_130553

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 + 2*x - 3 > 0}

-- Define the theorem
theorem set_operations :
  (Set.compl (A ∪ B) = {x | -3 ≤ x ∧ x ≤ 0}) ∧
  ((Set.compl A) ∩ B = {x | x > 1 ∨ x < -3}) := by
  sorry

end set_operations_l1305_130553


namespace G_is_leftmost_l1305_130547

/-- Represents a square with four integer labels -/
structure Square where
  name : Char
  w : Int
  x : Int
  y : Int
  z : Int

/-- The set of all squares -/
def squares : Finset Square := sorry

/-- Predicate to check if a square is leftmost -/
def is_leftmost (s : Square) : Prop := sorry

/-- The squares are arranged in a row without rotating or reflecting -/
axiom squares_in_row : sorry

/-- All squares are distinct -/
axiom squares_distinct : sorry

/-- The specific squares given in the problem -/
def F : Square := ⟨'F', 5, 1, 7, 9⟩
def G : Square := ⟨'G', 1, 0, 4, 6⟩
def H : Square := ⟨'H', 4, 8, 6, 2⟩
def I : Square := ⟨'I', 8, 5, 3, 7⟩
def J : Square := ⟨'J', 9, 2, 8, 0⟩

/-- All given squares are in the set of squares -/
axiom all_squares_in_set : F ∈ squares ∧ G ∈ squares ∧ H ∈ squares ∧ I ∈ squares ∧ J ∈ squares

/-- Theorem: Square G is the leftmost square -/
theorem G_is_leftmost : is_leftmost G := by sorry

end G_is_leftmost_l1305_130547


namespace sum_is_34_l1305_130586

/-- Represents a 4x4 grid filled with integers from 1 to 16 -/
def Grid : Type := Fin 4 → Fin 4 → Fin 16

/-- Fills the grid sequentially from 1 to 16 -/
def fillGrid : Grid :=
  fun i j => ⟨i.val * 4 + j.val + 1, by sorry⟩

/-- Represents a selection of 4 positions in the grid, each from a different row and column -/
structure Selection :=
  (pos : Fin 4 → Fin 4 × Fin 4)
  (different_rows : ∀ i j, i ≠ j → (pos i).1 ≠ (pos j).1)
  (different_cols : ∀ i j, i ≠ j → (pos i).2 ≠ (pos j).2)

/-- The main theorem to be proved -/
theorem sum_is_34 (s : Selection) : 
  (Finset.univ.sum fun i => (fillGrid (s.pos i).1 (s.pos i).2).val) = 34 := by
  sorry

end sum_is_34_l1305_130586


namespace cube_edge_sum_exists_l1305_130516

/-- Represents the edges of a cube --/
def CubeEdges := Fin 12

/-- Represents the faces of a cube --/
def CubeFaces := Fin 6

/-- A function that assigns numbers to the edges of a cube --/
def EdgeAssignment := CubeEdges → Fin 12

/-- A function that returns the edges that make up a face --/
def FaceEdges : CubeFaces → Finset CubeEdges := sorry

/-- The sum of numbers on a face given an edge assignment --/
def FaceSum (assignment : EdgeAssignment) (face : CubeFaces) : ℕ :=
  (FaceEdges face).sum (fun edge => (assignment edge).val + 1)

/-- Theorem stating that there exists an assignment of numbers from 1 to 12
    to the edges of a cube such that the sum of numbers on each face is equal --/
theorem cube_edge_sum_exists : 
  ∃ (assignment : EdgeAssignment), 
    (∀ (face1 face2 : CubeFaces), FaceSum assignment face1 = FaceSum assignment face2) ∧ 
    (∀ (edge1 edge2 : CubeEdges), edge1 ≠ edge2 → assignment edge1 ≠ assignment edge2) := by
  sorry

end cube_edge_sum_exists_l1305_130516


namespace increasing_absolute_value_function_l1305_130572

-- Define the function f(x) = |x - a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem increasing_absolute_value_function (a : ℝ) :
  (∀ x y, 1 ≤ x → x < y → f a x < f a y) → a ≤ 1 := by
  sorry

end increasing_absolute_value_function_l1305_130572


namespace complex_arithmetic_equation_l1305_130573

theorem complex_arithmetic_equation : 
  ((4501 * 2350) - (7125 / 9)) + (3250 ^ 2) * 4167 = 44045164058.33 := by
  sorry

end complex_arithmetic_equation_l1305_130573


namespace square_perimeter_from_rectangle_perimeter_l1305_130579

/-- Given a square divided into four congruent rectangles, 
    if the perimeter of each rectangle is 30 inches, 
    then the perimeter of the square is 48 inches. -/
theorem square_perimeter_from_rectangle_perimeter :
  ∀ s : ℝ,
  s > 0 →
  (2 * s + 2 * (s / 4) = 30) →
  (4 * s = 48) :=
by
  sorry

end square_perimeter_from_rectangle_perimeter_l1305_130579


namespace quadratic_always_positive_a_squared_plus_a_zero_not_equivalent_a_zero_a_plus_b_greater_than_two_ab_greater_than_one_not_equivalent_a_greater_than_four_iff_positive_roots_l1305_130532

-- Proposition A
theorem quadratic_always_positive : ∀ x : ℝ, x^2 - x + 1 > 0 := by sorry

-- Proposition B
theorem a_squared_plus_a_zero_not_equivalent_a_zero : ∃ a : ℝ, a^2 + a = 0 ∧ a ≠ 0 := by sorry

-- Proposition C
theorem a_plus_b_greater_than_two_ab_greater_than_one_not_equivalent :
  ∃ a b : ℝ, a + b > 2 ∧ a * b > 1 ∧ ¬(a > 1 ∧ b > 1) := by sorry

-- Proposition D
theorem a_greater_than_four_iff_positive_roots (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a = 0 → x > 0) ↔ a > 4 := by sorry

end quadratic_always_positive_a_squared_plus_a_zero_not_equivalent_a_zero_a_plus_b_greater_than_two_ab_greater_than_one_not_equivalent_a_greater_than_four_iff_positive_roots_l1305_130532


namespace not_prime_special_polynomial_l1305_130548

theorem not_prime_special_polynomial (n : ℕ+) : 
  ¬ Nat.Prime (n.val^2 - 2^2014 * 2014 * n.val + 4^2013 * (2014^2 - 1)) := by
  sorry

end not_prime_special_polynomial_l1305_130548


namespace last_three_digits_of_7_pow_215_l1305_130507

theorem last_three_digits_of_7_pow_215 : 7^215 ≡ 447 [ZMOD 1000] := by
  sorry

end last_three_digits_of_7_pow_215_l1305_130507


namespace system_inequality_solution_range_l1305_130515

theorem system_inequality_solution_range (x y m : ℝ) : 
  x - 2*y = 1 → 
  2*x + y = 4*m → 
  x + 3*y < 6 → 
  m < 7/4 := by
sorry

end system_inequality_solution_range_l1305_130515


namespace system_of_equations_solution_l1305_130587

theorem system_of_equations_solution :
  ∃! (x y : ℝ), x + 2*y = 4 ∧ x + 3*y = 5 :=
by
  sorry

end system_of_equations_solution_l1305_130587


namespace chi_square_test_error_probability_l1305_130503

/-- Represents the chi-square statistic -/
def chi_square : ℝ := 15.02

/-- Represents the critical value -/
def critical_value : ℝ := 6.635

/-- Represents the p-value -/
def p_value : ℝ := 0.01

/-- Represents the sample size -/
def sample_size : ℕ := 1000

/-- Represents the probability of error in rejecting the null hypothesis -/
def error_probability : ℝ := p_value

theorem chi_square_test_error_probability :
  error_probability = p_value :=
sorry

end chi_square_test_error_probability_l1305_130503


namespace expression_evaluation_l1305_130522

theorem expression_evaluation :
  let x : ℚ := 6
  let y : ℚ := -1/6
  let expr := 7 * x^2 * y - (3*x*y - 2*(x*y - 7/2*x^2*y + 1) + 1/2*x*y)
  expr = 7/2 := by sorry

end expression_evaluation_l1305_130522


namespace cell_count_after_12_days_first_six_days_growth_next_six_days_growth_l1305_130570

/-- Represents the cell growth model over 12 days -/
def CellGrowth : Nat → Nat
| 0 => 5  -- Initial cell count
| n + 1 =>
  if n < 6 then
    CellGrowth n * 3  -- Growth rate for first 6 days
  else if n < 12 then
    CellGrowth n * 2  -- Growth rate for next 6 days
  else
    CellGrowth n      -- No growth after 12 days

/-- Theorem stating the number of cells after 12 days -/
theorem cell_count_after_12_days :
  CellGrowth 12 = 180 := by
  sorry

/-- Verifies the growth pattern for the first 6 days -/
theorem first_six_days_growth (n : Nat) (h : n < 6) :
  CellGrowth (n + 1) = CellGrowth n * 3 := by
  sorry

/-- Verifies the growth pattern for days 7 to 12 -/
theorem next_six_days_growth (n : Nat) (h1 : 6 ≤ n) (h2 : n < 12) :
  CellGrowth (n + 1) = CellGrowth n * 2 := by
  sorry

end cell_count_after_12_days_first_six_days_growth_next_six_days_growth_l1305_130570


namespace amanda_summer_work_hours_l1305_130533

/-- Calculates the required weekly work hours for Amanda during summer -/
theorem amanda_summer_work_hours 
  (winter_weekly_hours : ℝ) 
  (winter_weeks : ℝ) 
  (winter_earnings : ℝ) 
  (summer_weeks : ℝ) 
  (summer_earnings : ℝ) 
  (h1 : winter_weekly_hours = 45) 
  (h2 : winter_weeks = 8) 
  (h3 : winter_earnings = 3600) 
  (h4 : summer_weeks = 20) 
  (h5 : summer_earnings = 4500) :
  (summer_earnings / (winter_earnings / (winter_weekly_hours * winter_weeks))) / summer_weeks = 22.5 := by
  sorry

#check amanda_summer_work_hours

end amanda_summer_work_hours_l1305_130533


namespace frequency_distribution_purpose_l1305_130546

/-- A frequency distribution table showing sample data sizes in groups -/
structure FrequencyDistributionTable where
  groups : Set (ℕ → ℕ)  -- Each function represents a group mapping sample size to frequency

/-- The proportion of data in each group -/
def proportion (t : FrequencyDistributionTable) : Set (ℕ → ℝ) :=
  sorry

/-- The overall corresponding situation being estimated -/
def overallSituation (t : FrequencyDistributionTable) : Type :=
  sorry

/-- Theorem stating the equivalence between the frequency distribution table
    and understanding proportions and estimating the overall situation -/
theorem frequency_distribution_purpose (t : FrequencyDistributionTable) :
  (∃ p : Set (ℕ → ℝ), p = proportion t) ∧
  (∃ s : Type, s = overallSituation t) :=
sorry

end frequency_distribution_purpose_l1305_130546
