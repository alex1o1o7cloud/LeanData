import Mathlib

namespace remainder_sum_factorials_l515_51592

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem remainder_sum_factorials (n : ℕ) (h : n ≥ 50) :
  sum_factorials n % 24 = (factorial 1 + factorial 2 + factorial 3 + factorial 4) % 24 :=
sorry

end remainder_sum_factorials_l515_51592


namespace arithmetic_simplification_l515_51504

theorem arithmetic_simplification :
  (0.25 * 4 - (5/6 + 1/12) * 6/5 = 1/10) ∧
  ((5/12 - 5/16) * 4/5 + 2/3 - 3/4 = 0) := by sorry

end arithmetic_simplification_l515_51504


namespace irreducible_fraction_l515_51520

theorem irreducible_fraction (n : ℕ+) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end irreducible_fraction_l515_51520


namespace polynomial_division_remainder_l515_51556

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^2 - 20 * X + 62 = (X - 6) * q + 50 := by
  sorry

end polynomial_division_remainder_l515_51556


namespace amusement_park_admission_l515_51562

/-- The number of children admitted to an amusement park -/
def num_children : ℕ := 180

/-- The number of adults admitted to an amusement park -/
def num_adults : ℕ := 315 - num_children

/-- The admission fee for children in dollars -/
def child_fee : ℚ := 3/2

/-- The admission fee for adults in dollars -/
def adult_fee : ℚ := 4

/-- The total number of people admitted to the park -/
def total_people : ℕ := 315

/-- The total admission fees collected in dollars -/
def total_fees : ℚ := 810

theorem amusement_park_admission :
  (num_children : ℚ) * child_fee + (num_adults : ℚ) * adult_fee = total_fees ∧
  num_children + num_adults = total_people :=
by sorry

end amusement_park_admission_l515_51562


namespace equation_solution_l515_51569

theorem equation_solution :
  ∀ x : ℚ, (Real.sqrt (4 * x + 9) / Real.sqrt (8 * x + 9) = Real.sqrt 3 / 2) → x = 9/8 := by
sorry

end equation_solution_l515_51569


namespace sum_of_roots_equation_l515_51529

theorem sum_of_roots_equation (x : ℝ) : 
  (3 = (x^3 - 3*x^2 - 12*x) / (x + 3)) → 
  (∃ y : ℝ, (3 = (y^3 - 3*y^2 - 12*y) / (y + 3)) ∧ (x + y = 6)) :=
by sorry

end sum_of_roots_equation_l515_51529


namespace pirate_treasure_distribution_l515_51535

def coin_distribution (x : ℕ) : ℕ × ℕ := 
  (x * (x + 1) / 2, x)

theorem pirate_treasure_distribution :
  ∃ x : ℕ, 
    let (bob_coins, sam_coins) := coin_distribution x
    bob_coins = 3 * sam_coins ∧ 
    bob_coins + sam_coins = 20 := by
  sorry

end pirate_treasure_distribution_l515_51535


namespace min_product_of_three_l515_51588

def S : Finset Int := {-9, -5, -1, 1, 3, 5, 8}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → a * b * c ≤ x * y * z) ∧ 
  (∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = -360) :=
by sorry

end min_product_of_three_l515_51588


namespace circle_E_and_tangents_l515_51546

-- Define the circle E
def circle_E (x y : ℝ) : Prop := (x - 5)^2 + (y - 1)^2 = 25

-- Define points A, B, C, and P
def point_A : ℝ × ℝ := (0, 1)
def point_B : ℝ × ℝ := (1, 4)
def point_C : ℝ × ℝ := (10, 1)
def point_P : ℝ × ℝ := (10, 11)

-- Define lines l1 and l2
def line_l1 (x y : ℝ) : Prop := x - 5*y - 5 = 0
def line_l2 (x y : ℝ) : Prop := x - 2*y - 8 = 0

-- Define tangent lines
def tangent_line1 (x : ℝ) : Prop := x = 10
def tangent_line2 (x y : ℝ) : Prop := 3*x - 4*y + 14 = 0

theorem circle_E_and_tangents :
  (∀ x y : ℝ, line_l1 x y ∧ line_l2 x y → (x, y) = point_C) →
  circle_E point_A.1 point_A.2 →
  circle_E point_B.1 point_B.2 →
  circle_E point_C.1 point_C.2 →
  (∀ x y : ℝ, circle_E x y ∧ (tangent_line1 x ∨ tangent_line2 x y) →
    ((x - point_P.1)^2 + (y - point_P.2)^2) * 25 = ((x - 5)^2 + (y - 1)^2) * ((point_P.1 - 5)^2 + (point_P.2 - 1)^2)) :=
by sorry

end circle_E_and_tangents_l515_51546


namespace sqrt_x_plus_y_equals_two_l515_51571

theorem sqrt_x_plus_y_equals_two (x y : ℝ) (h : Real.sqrt (3 - x) + Real.sqrt (x - 3) + 1 = y) :
  Real.sqrt (x + y) = 2 := by
  sorry

end sqrt_x_plus_y_equals_two_l515_51571


namespace unique_intersection_l515_51575

/-- The value of k for which the graphs of y = kx^2 - 5x + 4 and y = 2x - 6 intersect at exactly one point -/
def intersection_k : ℚ := 49/40

/-- First equation: y = kx^2 - 5x + 4 -/
def equation1 (k : ℚ) (x : ℚ) : ℚ := k * x^2 - 5*x + 4

/-- Second equation: y = 2x - 6 -/
def equation2 (x : ℚ) : ℚ := 2*x - 6

/-- Theorem stating that the graphs intersect at exactly one point if and only if k = 49/40 -/
theorem unique_intersection :
  ∀ k : ℚ, (∃! x : ℚ, equation1 k x = equation2 x) ↔ k = intersection_k :=
sorry

end unique_intersection_l515_51575


namespace inscribed_cube_diagonal_l515_51517

/-- The diagonal length of a cube inscribed in a sphere of radius R is 2R -/
theorem inscribed_cube_diagonal (R : ℝ) (R_pos : R > 0) :
  ∃ (cube : Set (Fin 3 → ℝ)), 
    (∀ p ∈ cube, ‖p‖ = R) ∧ 
    (∃ (d : Fin 3 → ℝ), d ∈ cube ∧ ‖d‖ = 2*R) :=
sorry

end inscribed_cube_diagonal_l515_51517


namespace trigonometric_identity_l515_51508

theorem trigonometric_identity (a b c : Real) :
  (Real.sin (a - b)) / (Real.sin a * Real.sin b) +
  (Real.sin (b - c)) / (Real.sin b * Real.sin c) +
  (Real.sin (c - a)) / (Real.sin c * Real.sin a) = 0 := by
  sorry

end trigonometric_identity_l515_51508


namespace chocolate_bar_breaks_l515_51547

/-- Represents a chocolate bar with grooves -/
structure ChocolateBar where
  longitudinal_grooves : Nat
  transverse_grooves : Nat

/-- Calculates the minimum number of breaks required to separate the chocolate bar into pieces with no grooves -/
def min_breaks (bar : ChocolateBar) : Nat :=
  4

/-- Theorem stating that a chocolate bar with 2 longitudinal grooves and 3 transverse grooves requires 4 breaks -/
theorem chocolate_bar_breaks (bar : ChocolateBar) 
  (h1 : bar.longitudinal_grooves = 2) 
  (h2 : bar.transverse_grooves = 3) : 
  min_breaks bar = 4 := by
  sorry

end chocolate_bar_breaks_l515_51547


namespace batsman_highest_score_l515_51565

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (overall_average : ℚ) 
  (score_difference : ℕ) 
  (average_excluding_extremes : ℚ) 
  (h : total_innings = 46)
  (h1 : overall_average = 61)
  (h2 : score_difference = 150)
  (h3 : average_excluding_extremes = 58) :
  ∃ (highest_score lowest_score : ℕ),
    highest_score - lowest_score = score_difference ∧
    (highest_score + lowest_score : ℚ) = 
      total_innings * overall_average - (total_innings - 2) * average_excluding_extremes ∧
    highest_score = 202 :=
by sorry

end batsman_highest_score_l515_51565


namespace bus_children_count_l515_51573

theorem bus_children_count (initial : ℕ) (joined : ℕ) (total : ℕ) : 
  initial = 64 → joined = 14 → total = initial + joined → total = 78 := by
  sorry

end bus_children_count_l515_51573


namespace isosceles_right_triangle_area_l515_51506

/-- Given an isosceles right triangle with hypotenuse 6√2, prove its area is 18 -/
theorem isosceles_right_triangle_area (h : ℝ) (is_hypotenuse : h = 6 * Real.sqrt 2) :
  let a : ℝ := h / Real.sqrt 2
  let area : ℝ := (1 / 2) * a ^ 2
  area = 18 := by
  sorry

end isosceles_right_triangle_area_l515_51506


namespace gcd_of_98_and_63_l515_51586

theorem gcd_of_98_and_63 : Nat.gcd 98 63 = 7 := by
  sorry

end gcd_of_98_and_63_l515_51586


namespace tan_double_special_angle_l515_51591

/-- An angle with vertex at the origin, initial side on positive x-axis, and terminal side on y = 2x -/
structure SpecialAngle where
  θ : Real
  terminal_side : (x : Real) → y = 2 * x

theorem tan_double_special_angle (α : SpecialAngle) : Real.tan (2 * α.θ) = -4/3 := by
  sorry

end tan_double_special_angle_l515_51591


namespace photo_arrangement_count_l515_51595

/-- The number of arrangements of 5 people where 3 specific people maintain their relative order but are not adjacent -/
def photo_arrangements : ℕ := 20

/-- The number of ways to choose 2 positions from 5 available positions -/
def choose_two_from_five : ℕ := 20

theorem photo_arrangement_count :
  photo_arrangements = choose_two_from_five :=
by sorry

end photo_arrangement_count_l515_51595


namespace unique_number_with_properties_l515_51576

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def all_divisors_even (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → Even d

def count_prime_divisors (n : ℕ) : ℕ := (n.divisors.filter Nat.Prime).card

def count_composite_divisors (n : ℕ) : ℕ := (n.divisors.filter (λ d => ¬Nat.Prime d ∧ d ≠ 1)).card

theorem unique_number_with_properties : 
  ∃! n : ℕ, is_four_digit n ∧ 
            all_divisors_even n ∧ 
            count_prime_divisors n = 3 ∧ 
            count_composite_divisors n = 39 ∧
            n = 6336 := by sorry

end unique_number_with_properties_l515_51576


namespace coffee_mug_price_l515_51507

/-- The cost of a personalized coffee mug -/
def coffee_mug_cost : ℕ := sorry

/-- The price of a bracelet -/
def bracelet_price : ℕ := 15

/-- The price of a gold heart necklace -/
def necklace_price : ℕ := 10

/-- The number of bracelets bought -/
def num_bracelets : ℕ := 3

/-- The number of gold heart necklaces bought -/
def num_necklaces : ℕ := 2

/-- The amount paid with -/
def amount_paid : ℕ := 100

/-- The change received -/
def change_received : ℕ := 15

theorem coffee_mug_price : coffee_mug_cost = 20 := by
  sorry

end coffee_mug_price_l515_51507


namespace max_value_of_f_one_l515_51531

/-- Given a function f(x) = x^2 + abx + a + 2b where f(0) = 4, 
    the maximum value of f(1) is 7. -/
theorem max_value_of_f_one (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + a*b*x + a + 2*b
  (f 0 = 4) → (∀ y : ℝ, f 1 ≤ 7) ∧ (∃ y : ℝ, f 1 = 7) := by
  sorry

end max_value_of_f_one_l515_51531


namespace mary_speed_calculation_l515_51534

/-- Mary's running speed in miles per hour -/
def mary_speed : ℝ := sorry

/-- Jimmy's running speed in miles per hour -/
def jimmy_speed : ℝ := 4

/-- Time elapsed in hours -/
def time : ℝ := 1

/-- Distance between Mary and Jimmy after 1 hour in miles -/
def distance : ℝ := 9

theorem mary_speed_calculation :
  mary_speed = 5 :=
by
  sorry

end mary_speed_calculation_l515_51534


namespace sin_15_cos_15_l515_51574

theorem sin_15_cos_15 : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end sin_15_cos_15_l515_51574


namespace triangle_side_length_l515_51561

theorem triangle_side_length (A B C D : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BD := Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  AB = 30 →
  (B.1 - D.1) * (A.1 - D.1) + (B.2 - D.2) * (A.2 - D.2) = 0 →
  (A.2 - D.2) / AB = 4/5 →
  BD / BC = 1/5 →
  C.2 = D.2 →
  C.1 > D.1 →
  CD = 24 * Real.sqrt 23 := by
sorry

end triangle_side_length_l515_51561


namespace wage_payment_problem_l515_51539

/-- Given a sum of money that can pay A's wages for 20 days and both A and B's wages for 12 days,
    prove that it can pay B's wages for 30 days. -/
theorem wage_payment_problem (total_sum : ℝ) (wage_A wage_B : ℝ) 
  (h1 : total_sum = 20 * wage_A)
  (h2 : total_sum = 12 * (wage_A + wage_B)) :
  total_sum = 30 * wage_B :=
by sorry

end wage_payment_problem_l515_51539


namespace evaluate_expression_l515_51525

theorem evaluate_expression : (0.5^4 - 0.25^2) / (0.1^2) = 0 := by
  sorry

end evaluate_expression_l515_51525


namespace max_value_m_l515_51502

theorem max_value_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, (3/a + 1/b ≥ m/(a + 3*b))) → m ≤ 12 :=
by sorry

end max_value_m_l515_51502


namespace triangle_side_length_l515_51501

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 5 →
  c = 2 →
  Real.cos A = 2/3 →
  b = 3 :=
by sorry

end triangle_side_length_l515_51501


namespace monotonic_quadratic_function_condition_l515_51543

/-- A function f(x) = x^2 + 2(a - 1)x + 2 is monotonic on the interval [-1, 2] if and only if a ∈ (-∞, -1] ∪ [2, +∞) -/
theorem monotonic_quadratic_function_condition (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, Monotone (fun x => x^2 + 2*(a - 1)*x + 2)) ↔
  a ∈ Set.Iic (-1 : ℝ) ∪ Set.Ici 2 :=
sorry

end monotonic_quadratic_function_condition_l515_51543


namespace polygon_area_is_400_l515_51579

/-- The area of a right triangle -/
def rightTriangleArea (base height : ℝ) : ℝ := 0.5 * base * height

/-- The area of a trapezoid -/
def trapezoidArea (shortBase longBase height : ℝ) : ℝ := 0.5 * (shortBase + longBase) * height

/-- The total area of the polygon -/
def polygonArea (triangleBase triangleHeight trapezoidShortBase trapezoidLongBase trapezoidHeight : ℝ) : ℝ :=
  2 * rightTriangleArea triangleBase triangleHeight + 
  2 * trapezoidArea trapezoidShortBase trapezoidLongBase trapezoidHeight

theorem polygon_area_is_400 :
  polygonArea 10 10 10 20 10 = 400 := by
  sorry

end polygon_area_is_400_l515_51579


namespace time_after_1875_minutes_l515_51518

/-- Represents time of day in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

theorem time_after_1875_minutes : 
  let start_time := Time.mk 15 15  -- 3:15 p.m.
  let end_time := Time.mk 10 30    -- 10:30 a.m.
  addMinutes start_time 1875 = end_time :=
by sorry

end time_after_1875_minutes_l515_51518


namespace number_fraction_problem_l515_51503

theorem number_fraction_problem (x : ℝ) : (1/3 : ℝ) * (1/4 : ℝ) * x = 15 → (3/10 : ℝ) * x = 54 := by
  sorry

end number_fraction_problem_l515_51503


namespace quadratic_discriminant_l515_51583

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_discriminant :
  discriminant 5 (-6) 1 = 16 := by sorry

end quadratic_discriminant_l515_51583


namespace point_movement_on_number_line_l515_51581

/-- 
Given a point on a number line that:
1. Starts at position -2
2. Moves 8 units to the right
3. Moves 4 units to the left
This theorem proves that the final position of the point is 2.
-/
theorem point_movement_on_number_line : 
  let start_position : ℤ := -2
  let right_movement : ℤ := 8
  let left_movement : ℤ := 4
  let final_position := start_position + right_movement - left_movement
  final_position = 2 := by sorry

end point_movement_on_number_line_l515_51581


namespace consecutive_integers_product_l515_51560

theorem consecutive_integers_product (n : ℤ) : 
  n * (n + 1) * (n + 2) = (n + 1)^3 - (n + 1) := by
  sorry

end consecutive_integers_product_l515_51560


namespace expression_evaluation_l515_51549

theorem expression_evaluation (x y : ℤ) (hx : x = -2) (hy : y = -3) :
  (x - 3*y)^2 + (x - 2*y)*(x + 2*y) - x*(2*x - 5*y) - y = 42 := by
  sorry

end expression_evaluation_l515_51549


namespace triangle_area_l515_51528

theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 3 * Real.sqrt 2) (h2 : b = 2 * Real.sqrt 3) (h3 : Real.cos C = 1/3) :
  (1/2) * a * b * Real.sin C = 4 * Real.sqrt 3 := by
sorry

end triangle_area_l515_51528


namespace triangle_construction_from_feet_l515_51598

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The foot of an altitude in a triangle -/
def altitude_foot (T : Triangle) (v : Point) : Point :=
  sorry

/-- The foot of an angle bisector in a triangle -/
def angle_bisector_foot (T : Triangle) (v : Point) : Point :=
  sorry

/-- Theorem: A unique triangle exists given the feet of two altitudes and one angle bisector -/
theorem triangle_construction_from_feet 
  (A₁ B₁ B₂ : Point) : 
  ∃! (T : Triangle), 
    altitude_foot T T.A = A₁ ∧ 
    altitude_foot T T.B = B₁ ∧ 
    angle_bisector_foot T T.B = B₂ :=
  sorry

end triangle_construction_from_feet_l515_51598


namespace ab_is_perfect_cube_l515_51545

theorem ab_is_perfect_cube (a b : ℕ+) (h1 : b < a) 
  (h2 : ∃ k : ℕ, k * (a * b * (a - b)) = a^3 + b^3 + a*b) : 
  ∃ n : ℕ, (a * b : ℕ) = n^3 := by
sorry

end ab_is_perfect_cube_l515_51545


namespace reciprocal_power_l515_51566

theorem reciprocal_power (a : ℝ) (h : a⁻¹ = -1) : a^2023 = -1 := by
  sorry

end reciprocal_power_l515_51566


namespace square_difference_l515_51536

/-- A configuration of four squares with specific side length differences -/
structure SquareConfiguration where
  small : ℝ
  third : ℝ
  second : ℝ
  largest : ℝ
  third_diff : third = small + 13
  second_diff : second = third + 5
  largest_diff : largest = second + 11

/-- The theorem stating that the difference between the largest and smallest square's side lengths is 29 -/
theorem square_difference (config : SquareConfiguration) : config.largest - config.small = 29 :=
  sorry

end square_difference_l515_51536


namespace smallest_zero_floor_is_three_l515_51554

noncomputable def g (x : ℝ) : ℝ := Real.cos x - Real.sin x + 4 * Real.tan x

theorem smallest_zero_floor_is_three :
  ∃ (s : ℝ), s > 0 ∧ g s = 0 ∧ (∀ x, x > 0 ∧ g x = 0 → x ≥ s) ∧ ⌊s⌋ = 3 :=
sorry

end smallest_zero_floor_is_three_l515_51554


namespace arithmetic_sequence_middle_term_l515_51589

/-- Given an arithmetic sequence with first term 3² and last term 3⁴, 
    the middle term y is equal to 45. -/
theorem arithmetic_sequence_middle_term : 
  ∀ (a : ℕ → ℤ), 
    (a 0 = 3^2) → 
    (a 2 = 3^4) → 
    (∀ n : ℕ, n < 2 → a (n + 1) - a n = a 1 - a 0) → 
    a 1 = 45 := by
  sorry

end arithmetic_sequence_middle_term_l515_51589


namespace complex_fraction_simplification_l515_51570

theorem complex_fraction_simplification :
  (3 / 7 - 2 / 5) / (5 / 12 + 1 / 4) = 3 / 70 := by
sorry

end complex_fraction_simplification_l515_51570


namespace wind_speed_calculation_l515_51524

/-- Wind speed calculation for a helicopter flight --/
theorem wind_speed_calculation 
  (s v : ℝ) 
  (h_positive_s : 0 < s)
  (h_positive_v : 0 < v)
  (h_v_greater_s : s < v) :
  ∃ (x y vB : ℝ),
    x + y = 2 ∧                 -- Total flight time is 2 hours
    v + vB = s / x ∧            -- Speed from A to B (with wind)
    v - vB = s / y ∧            -- Speed from B to A (against wind)
    vB = Real.sqrt (v * (v - s)) -- Wind speed formula
  := by sorry

end wind_speed_calculation_l515_51524


namespace max_value_of_exponential_difference_l515_51594

theorem max_value_of_exponential_difference (x : ℝ) : 5^x - 25^x ≤ (1/4 : ℝ) := by
  sorry

end max_value_of_exponential_difference_l515_51594


namespace fifth_term_value_l515_51593

/-- An arithmetic sequence satisfying the given recursive relation -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = -a n + n

/-- The fifth term of the sequence is 9/4 -/
theorem fifth_term_value (a : ℕ → ℚ) (h : ArithmeticSequence a) : a 5 = 9/4 := by
  sorry

end fifth_term_value_l515_51593


namespace tangent_line_to_unit_circle_l515_51563

/-- The equation of the tangent line to the unit circle at point (a, b) -/
theorem tangent_line_to_unit_circle (a b : ℝ) (h : a^2 + b^2 = 1) :
  ∀ x y : ℝ, (x - a)^2 + (y - b)^2 = (a*x + b*y - 1)^2 / (a^2 + b^2) :=
by sorry

end tangent_line_to_unit_circle_l515_51563


namespace finleys_age_l515_51522

/-- Proves that Finley's age is 55 years old given the conditions in the problem --/
theorem finleys_age (jill roger finley : ℕ) : 
  jill = 20 → 
  roger = 2 * jill + 5 → 
  (roger + 15) - (jill + 15) = finley - 30 → 
  finley = 55 := by sorry

end finleys_age_l515_51522


namespace points_bound_l515_51544

/-- A structure representing a set of points on a line with colored circles -/
structure ColoredCircles where
  k : ℕ  -- Number of points
  n : ℕ  -- Number of colors
  points : Fin k → ℝ  -- Function mapping point indices to their positions on the line
  circle_color : Fin k → Fin k → Fin n  -- Function assigning a color to each circle

/-- Predicate to check if two circles are mutually tangent -/
def mutually_tangent (cc : ColoredCircles) (i j m l : Fin cc.k) : Prop :=
  (cc.points i < cc.points m) ∧ (cc.points m < cc.points j) ∧ (cc.points j < cc.points l)

/-- Axiom: Mutually tangent circles have different colors -/
axiom different_colors (cc : ColoredCircles) :
  ∀ (i j m l : Fin cc.k), mutually_tangent cc i j m l →
    cc.circle_color i j ≠ cc.circle_color m l

/-- Theorem: The number of points is at most 2^n -/
theorem points_bound (cc : ColoredCircles) : cc.k ≤ 2^cc.n := by
  sorry

end points_bound_l515_51544


namespace vector_sum_magnitude_l515_51513

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem vector_sum_magnitude (a b : E) 
  (ha : ‖a‖ = 1) 
  (hb : ‖b‖ = 1) 
  (hab : ‖a - b‖ = 1) : 
  ‖a + b‖ = Real.sqrt 3 := by
  sorry

end vector_sum_magnitude_l515_51513


namespace escalator_travel_time_l515_51512

-- Define the escalator properties and person's walking speed
def escalator_speed : ℝ := 11
def escalator_length : ℝ := 126
def person_speed : ℝ := 3

-- Theorem statement
theorem escalator_travel_time :
  let combined_speed := escalator_speed + person_speed
  let time := escalator_length / combined_speed
  time = 9 := by sorry

end escalator_travel_time_l515_51512


namespace inequality_system_solution_l515_51587

theorem inequality_system_solution (x : ℝ) : 
  (2 + x > 7 - 4 * x ∧ x < (4 + x) / 2) ↔ (1 < x ∧ x < 4) := by
sorry

end inequality_system_solution_l515_51587


namespace round_robin_28_games_8_teams_l515_51590

/-- The number of games in a single round-robin tournament with n teams -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A single round-robin tournament with 28 games requires 8 teams -/
theorem round_robin_28_games_8_teams :
  ∃ (n : ℕ), n > 0 ∧ num_games n = 28 ∧ n = 8 := by sorry

end round_robin_28_games_8_teams_l515_51590


namespace sum_of_squares_of_roots_l515_51553

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (10 * x₁^2 + 16 * x₁ - 18 = 0) → 
  (10 * x₂^2 + 16 * x₂ - 18 = 0) → 
  x₁^2 + x₂^2 = 244 / 25 := by
  sorry

end sum_of_squares_of_roots_l515_51553


namespace smallest_product_l515_51580

def digits : List ℕ := [5, 6, 7, 8]

def is_valid_arrangement (a b c d : ℕ) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : ℕ) : ℕ := (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : ℕ, is_valid_arrangement a b c d →
  product a b c d ≥ 3876 :=
by sorry

end smallest_product_l515_51580


namespace minimally_intersecting_triples_count_l515_51568

def Universe : Finset Nat := Finset.range 8

structure MinimallyIntersectingTriple (A B C : Finset Nat) : Prop where
  subset_universe : A ⊆ Universe ∧ B ⊆ Universe ∧ C ⊆ Universe
  intersection_size : (A ∩ B).card = 1 ∧ (B ∩ C).card = 1 ∧ (C ∩ A).card = 1
  empty_triple_intersection : (A ∩ B ∩ C).card = 0

def M : Nat := (Finset.powerset Universe).card

theorem minimally_intersecting_triples_count : M % 1000 = 344 := by
  sorry

end minimally_intersecting_triples_count_l515_51568


namespace sugar_packs_theorem_l515_51532

/-- Given the total amount of sugar, weight per pack, and leftover sugar, 
    calculate the number of packs. -/
def calculate_packs (total_sugar : ℕ) (weight_per_pack : ℕ) (leftover_sugar : ℕ) : ℕ :=
  (total_sugar - leftover_sugar) / weight_per_pack

/-- Theorem stating that given the specific conditions, 
    the number of packs is 12. -/
theorem sugar_packs_theorem (total_sugar weight_per_pack leftover_sugar : ℕ) 
  (h1 : total_sugar = 3020)
  (h2 : weight_per_pack = 250)
  (h3 : leftover_sugar = 20) :
  calculate_packs total_sugar weight_per_pack leftover_sugar = 12 := by
  sorry

#eval calculate_packs 3020 250 20

end sugar_packs_theorem_l515_51532


namespace max_value_of_f_l515_51509

def f (m : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + m

theorem max_value_of_f (m : ℝ) :
  (∀ x ∈ Set.Icc (-2) 2, f m x ≥ 1) →
  (∃ x ∈ Set.Icc (-2) 2, f m x = 1) →
  (∃ x ∈ Set.Icc (-2) 2, f m x = 21) ∧
  (∀ x ∈ Set.Icc (-2) 2, f m x ≤ 21) := by
sorry

end max_value_of_f_l515_51509


namespace spherical_coordinate_conversion_l515_51523

def standardSphericalCoordinates (ρ θ φ : Real) : Prop :=
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi

theorem spherical_coordinate_conversion :
  ∃ (ρ θ φ : Real),
    standardSphericalCoordinates ρ θ φ ∧
    ρ = 4 ∧
    θ = 5 * Real.pi / 4 ∧
    φ = Real.pi / 5 ∧
    (ρ, θ, φ) = (4, Real.pi / 4, 9 * Real.pi / 5) :=
sorry

end spherical_coordinate_conversion_l515_51523


namespace student_congress_sample_size_l515_51541

/-- Given a school with classes and students, prove the sample size for a "Student Congress" -/
theorem student_congress_sample_size 
  (num_classes : ℕ) 
  (students_per_class : ℕ) 
  (selected_students : ℕ) 
  (h1 : num_classes = 40)
  (h2 : students_per_class = 50)
  (h3 : selected_students = 150) :
  selected_students = 150 := by
  sorry

end student_congress_sample_size_l515_51541


namespace normal_distribution_symmetry_l515_51514

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  hσ : σ > 0

/-- Probability function for a normal random variable -/
noncomputable def P (ξ : NormalRV) (x : ℝ) : ℝ := sorry

theorem normal_distribution_symmetry 
  (ξ : NormalRV) 
  (h1 : ξ.μ = 2) 
  (h2 : P ξ 4 = 0.84) : 
  P ξ 0 = 0.16 := by sorry

end normal_distribution_symmetry_l515_51514


namespace determine_a_l515_51578

-- Define positive integers a, b, c, and d
variable (a b c d : ℕ+)

-- Define the main theorem
theorem determine_a :
  (18^a.val * 9^(4*a.val - 1) * 27^c.val = 2^6 * 3^b.val * 7^d.val) →
  (a.val * c.val : ℚ) = 4 / (2*b.val + d.val) →
  b.val^2 - 4*a.val*c.val = d.val →
  a = 6 := by
  sorry


end determine_a_l515_51578


namespace train_length_calculation_train_length_proof_l515_51511

/-- Calculates the length of a train given its speed, the time it takes to cross a bridge, and the length of the bridge. -/
theorem train_length_calculation (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 240 →
  (train_speed * crossing_time) - bridge_length = 135 :=
by
  sorry

/-- Proves that a train traveling at 45 km/hr that crosses a 240-meter bridge in 30 seconds has a length of 135 meters. -/
theorem train_length_proof : 
  ∃ (train_speed crossing_time bridge_length : ℝ),
    train_speed = 45 * (1000 / 3600) ∧
    crossing_time = 30 ∧
    bridge_length = 240 ∧
    (train_speed * crossing_time) - bridge_length = 135 :=
by
  sorry

end train_length_calculation_train_length_proof_l515_51511


namespace south_american_countries_visited_l515_51527

/-- Proves the number of South American countries visited given the conditions --/
theorem south_american_countries_visited
  (total : ℕ)
  (europe : ℕ)
  (asia : ℕ)
  (h1 : total = 42)
  (h2 : europe = 20)
  (h3 : asia = 6)
  (h4 : 2 * asia = total - europe - asia) :
  total - europe - asia = 8 :=
by sorry

end south_american_countries_visited_l515_51527


namespace crocus_bulb_cost_l515_51516

theorem crocus_bulb_cost (total_space : ℕ) (daffodil_cost : ℚ) (total_budget : ℚ) (crocus_count : ℕ) :
  total_space = 55 →
  daffodil_cost = 65/100 →
  total_budget = 2915/100 →
  crocus_count = 22 →
  ∃ (crocus_cost : ℚ), crocus_cost = 35/100 ∧
    crocus_count * crocus_cost + (total_space - crocus_count) * daffodil_cost = total_budget :=
by sorry

end crocus_bulb_cost_l515_51516


namespace train_speed_l515_51597

/-- The speed of a train crossing a platform -/
theorem train_speed (train_length platform_length : Real) (crossing_time : Real) 
  (h1 : train_length = 120)
  (h2 : platform_length = 130.02)
  (h3 : crossing_time = 15) : 
  ∃ (speed : Real), abs (speed - 60.0048) < 0.0001 := by
  sorry

end train_speed_l515_51597


namespace new_mean_after_adding_specific_problem_l515_51582

theorem new_mean_after_adding (n : ℕ) (original_mean add_value : ℝ) :
  n > 0 →
  let new_mean := (n * original_mean + n * add_value) / n
  new_mean = original_mean + add_value :=
by sorry

theorem specific_problem :
  let n : ℕ := 15
  let original_mean : ℝ := 40
  let add_value : ℝ := 13
  (n * original_mean + n * add_value) / n = 53 :=
by sorry

end new_mean_after_adding_specific_problem_l515_51582


namespace smallest_X_proof_l515_51537

/-- A function that checks if a positive integer only contains 0s and 1s as digits -/
def onlyZerosAndOnes (n : ℕ) : Prop := sorry

/-- The smallest positive integer X such that there exists a T satisfying the conditions -/
def smallestX : ℕ := 74

theorem smallest_X_proof :
  ∀ T : ℕ,
  T > 0 →
  onlyZerosAndOnes T →
  (∃ X : ℕ, T = 15 * X) →
  ∃ X : ℕ, T = 15 * X ∧ X ≥ smallestX :=
sorry

end smallest_X_proof_l515_51537


namespace length_of_side_b_area_of_triangle_ABC_l515_51542

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Add conditions for a valid triangle if necessary
  true

-- Given conditions
axiom side_a : ℝ
axiom side_a_value : side_a = 3 * Real.sqrt 3

axiom side_c : ℝ
axiom side_c_value : side_c = 2

axiom angle_B : ℝ
axiom angle_B_value : angle_B = 150 * Real.pi / 180

-- Theorem for the length of side b
theorem length_of_side_b (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C)
  (ha : a = side_a) (hc : c = side_c) (hB : B = angle_B) :
  b = 7 := by sorry

-- Theorem for the area of triangle ABC
theorem area_of_triangle_ABC (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C)
  (ha : a = side_a) (hc : c = side_c) (hB : B = angle_B) :
  (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2 := by sorry

end length_of_side_b_area_of_triangle_ABC_l515_51542


namespace decimal_to_fraction_l515_51505

theorem decimal_to_fraction :
  (2.35 : ℚ) = 47 / 20 := by sorry

end decimal_to_fraction_l515_51505


namespace square_has_most_symmetry_l515_51500

-- Define the types of figures
inductive Figure
  | EquilateralTriangle
  | NonSquareRhombus
  | NonSquareRectangle
  | IsoscelesTrapezoid
  | Square

-- Function to get the number of lines of symmetry for each figure
def linesOfSymmetry (f : Figure) : ℕ :=
  match f with
  | Figure.EquilateralTriangle => 3
  | Figure.NonSquareRhombus => 2
  | Figure.NonSquareRectangle => 2
  | Figure.IsoscelesTrapezoid => 1
  | Figure.Square => 4

-- Theorem stating that the square has the greatest number of lines of symmetry
theorem square_has_most_symmetry :
  ∀ f : Figure, linesOfSymmetry Figure.Square ≥ linesOfSymmetry f :=
by
  sorry


end square_has_most_symmetry_l515_51500


namespace football_players_l515_51596

theorem football_players (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 35)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 6) :
  total - tennis + both - neither = 26 := by
  sorry

end football_players_l515_51596


namespace line_relationship_l515_51526

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (are_skew : Line → Line → Prop)
variable (are_parallel : Line → Line → Prop)
variable (are_intersecting : Line → Line → Prop)

-- Theorem statement
theorem line_relationship (a b c : Line)
  (h1 : are_skew a b)
  (h2 : are_parallel a c) :
  are_skew b c ∨ are_intersecting b c :=
sorry

end line_relationship_l515_51526


namespace reciprocal_determinant_solution_ratios_l515_51585

/-- Given a 2x2 matrix with determinant D ≠ 0, prove that the determinant of its adjugate divided by D is equal to 1/D -/
theorem reciprocal_determinant (a b c d : ℝ) (h : a * d - b * c ≠ 0) :
  let D := a * d - b * c
  (d / D) * (a / D) - (-c / D) * (-b / D) = 1 / D := by sorry

/-- For a system of two linear equations in three variables,
    prove that the ratios of the solutions are given by specific 2x2 determinants -/
theorem solution_ratios (a b c d e f : ℝ) 
  (h1 : ∀ x y z : ℝ, a * x + b * y + c * z = 0 → d * x + e * y + f * z = 0) :
  ∃ (k : ℝ), k ≠ 0 ∧
    (b * f - c * e) * k = (c * d - a * f) * k ∧
    (c * d - a * f) * k = (a * e - b * d) * k := by sorry

end reciprocal_determinant_solution_ratios_l515_51585


namespace odd_function_values_and_monotonicity_and_inequality_l515_51548

noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x ↦ (2^x + a) / (2^x + b)

theorem odd_function_values_and_monotonicity_and_inequality
  (h_odd : ∀ x, f a b (-x) = -(f a b x)) :
  (a = -1 ∧ b = 1) ∧
  (∀ x y, x < y → f a b x < f a b y) ∧
  (∀ x, f a b x + f a b (6 - x^2) ≤ 0 ↔ x ≤ -2 ∨ x ≥ 3) :=
sorry

end odd_function_values_and_monotonicity_and_inequality_l515_51548


namespace no_division_into_non_convex_quadrilaterals_l515_51557

/-- A polygon is a set of points in the plane -/
def Polygon : Type := Set (ℝ × ℝ)

/-- A convex polygon is a polygon where any line segment between two points in the polygon lies entirely within the polygon -/
def ConvexPolygon (P : Polygon) : Prop := sorry

/-- A quadrilateral is a polygon with exactly four vertices -/
def Quadrilateral (Q : Polygon) : Prop := sorry

/-- A non-convex quadrilateral is a quadrilateral that is not convex -/
def NonConvexQuadrilateral (Q : Polygon) : Prop := Quadrilateral Q ∧ ¬ConvexPolygon Q

/-- A division of a polygon into quadrilaterals is a finite set of quadrilaterals that cover the polygon without overlap -/
def DivisionIntoQuadrilaterals (P : Polygon) (Qs : Finset Polygon) : Prop := sorry

/-- Theorem: It's impossible to divide a convex polygon into a finite number of non-convex quadrilaterals -/
theorem no_division_into_non_convex_quadrilaterals (P : Polygon) (Qs : Finset Polygon) :
  ConvexPolygon P → DivisionIntoQuadrilaterals P Qs → ¬(∀ Q ∈ Qs, NonConvexQuadrilateral Q) := by
  sorry

end no_division_into_non_convex_quadrilaterals_l515_51557


namespace parallelogram_point_C_l515_51599

structure Point where
  x : ℝ
  y : ℝ

def Parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x = C.x - D.x) ∧ (B.y - A.y = C.y - D.y) ∧
  (D.x - A.x = C.x - B.x) ∧ (D.y - A.y = C.y - B.y)

def InFirstQuadrant (P : Point) : Prop :=
  P.x > 0 ∧ P.y > 0

theorem parallelogram_point_C : 
  ∀ (A B C D : Point),
    Parallelogram A B C D →
    InFirstQuadrant A →
    InFirstQuadrant B →
    InFirstQuadrant C →
    InFirstQuadrant D →
    A.x = 2 ∧ A.y = 3 →
    B.x = 7 ∧ B.y = 3 →
    D.x = 3 ∧ D.y = 7 →
    C.x = 8 ∧ C.y = 7 :=
by sorry

end parallelogram_point_C_l515_51599


namespace segment_length_l515_51551

/-- Given three points A, B, and C on a line, with AB = 4 and BC = 3,
    the length of AC is either 7 or 1. -/
theorem segment_length (A B C : ℝ) : 
  (B - A = 4) → (C - B = 3 ∨ B - C = 3) → (C - A = 7 ∨ C - A = 1) := by
  sorry

end segment_length_l515_51551


namespace smallest_n_for_sqrt_difference_l515_51533

theorem smallest_n_for_sqrt_difference : ∃ n : ℕ+, 
  (n = 626 ∧ 
   ∀ m : ℕ+, m < n → Real.sqrt m.val - Real.sqrt (m.val - 1) ≥ 0.02) ∧
  Real.sqrt n.val - Real.sqrt (n.val - 1) < 0.02 := by
  sorry

end smallest_n_for_sqrt_difference_l515_51533


namespace largest_partition_size_l515_51540

/-- A partition of the positive integers into k subsets -/
def Partition (k : ℕ) := Fin k → Set ℕ

/-- The property that every integer ≥ 15 can be represented as a sum of two distinct elements from each subset -/
def HasPropertyForAll (P : Partition k) : Prop :=
  ∀ (n : ℕ) (i : Fin k), n ≥ 15 → ∃ (x y : ℕ), x ≠ y ∧ x ∈ P i ∧ y ∈ P i ∧ x + y = n

/-- The main theorem statement -/
theorem largest_partition_size :
  ∃ (k : ℕ), k > 0 ∧ 
    (∃ (P : Partition k), HasPropertyForAll P) ∧ 
    (∀ (m : ℕ), m > k → ¬∃ (Q : Partition m), HasPropertyForAll Q) ∧
    k = 3 := by
  sorry

end largest_partition_size_l515_51540


namespace min_integer_value_is_seven_l515_51552

def expression (parentheses : List (Nat × Nat)) : ℚ :=
  let nums := [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
  -- Define a function to evaluate the expression based on parentheses placement
  sorry

def is_valid_parentheses (parentheses : List (Nat × Nat)) : Prop :=
  -- Define a predicate to check if the parentheses placement is valid
  sorry

theorem min_integer_value_is_seven :
  ∃ (parentheses : List (Nat × Nat)),
    is_valid_parentheses parentheses ∧
    (expression parentheses).num = 7 ∧
    (expression parentheses).den = 1 ∧
    (∀ (other_parentheses : List (Nat × Nat)),
      is_valid_parentheses other_parentheses →
      (expression other_parentheses).num ≥ 7 ∨ (expression other_parentheses).den ≠ 1) :=
by
  sorry

end min_integer_value_is_seven_l515_51552


namespace red_pens_count_l515_51577

/-- Proves the number of red pens initially in a jar --/
theorem red_pens_count (initial_blue : ℕ) (initial_black : ℕ) (removed_blue : ℕ) (removed_black : ℕ) (remaining_total : ℕ) : 
  initial_blue = 9 →
  initial_black = 21 →
  removed_blue = 4 →
  removed_black = 7 →
  remaining_total = 25 →
  ∃ (initial_red : ℕ), 
    initial_red = 6 ∧
    initial_blue + initial_black + initial_red = 
    remaining_total + removed_blue + removed_black :=
by sorry

end red_pens_count_l515_51577


namespace absolute_value_non_positive_l515_51538

theorem absolute_value_non_positive (y : ℚ) : 
  |4 * y - 6| ≤ 0 ↔ y = 3/2 := by sorry

end absolute_value_non_positive_l515_51538


namespace negation_equivalence_l515_51510

theorem negation_equivalence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m < 0) ↔ (∀ x : ℤ, x^2 + 2*x + m ≥ 0) :=
by sorry

end negation_equivalence_l515_51510


namespace possible_values_of_a_l515_51530

theorem possible_values_of_a : 
  ∀ (a b c : ℤ), 
  (∀ x : ℝ, (x - a) * (x - 8) + 1 = (x + b) * (x + c)) → 
  (a = 6 ∨ a = 10) := by
  sorry

end possible_values_of_a_l515_51530


namespace train_length_l515_51519

/-- The length of a train given its speed and time to cross a fixed point. -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 72 * (5 / 18) → time = 16 → speed * time = 320 := by
  sorry

#check train_length

end train_length_l515_51519


namespace function_value_at_three_l515_51558

/-- Given a function f(x) = ax^4 + b cos(x) - x where f(-3) = 7, prove that f(3) = 1 -/
theorem function_value_at_three (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^4 + b * Real.cos x - x)
  (h2 : f (-3) = 7) : 
  f 3 = 1 := by sorry

end function_value_at_three_l515_51558


namespace coefficient_of_a_is_one_l515_51564

-- Define a monomial type
def Monomial := ℚ → ℕ → ℚ

-- Define the coefficient of a monomial
def coefficient (m : Monomial) : ℚ := m 1 0

-- Define the monomial 'a'
def a : Monomial := fun c n => if n = 1 then 1 else 0

-- Theorem statement
theorem coefficient_of_a_is_one : coefficient a = 1 := by
  sorry

end coefficient_of_a_is_one_l515_51564


namespace distance_vertical_line_l515_51521

/-- The distance between two points on a vertical line with y-coordinates differing by 2 is 2. -/
theorem distance_vertical_line (a : ℝ) : 
  Real.sqrt (((-3) - (-3))^2 + ((2 - a) - (-a))^2) = 2 := by
  sorry

end distance_vertical_line_l515_51521


namespace simplify_expression_1_simplify_expression_2_simplify_expression_3_l515_51550

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : (1 : ℝ) * (2 * x^2)^3 - x^2 * x^4 = 7 * x^6 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) : (a + b)^2 - b * (2 * a + b) = a^2 := by sorry

-- Problem 3
theorem simplify_expression_3 (x : ℝ) : (x + 1) * (x - 1) - x^2 = -1 := by sorry

end simplify_expression_1_simplify_expression_2_simplify_expression_3_l515_51550


namespace arithmetic_calculations_l515_51559

theorem arithmetic_calculations :
  (12 - (-18) + (-7) - 20 = 3) ∧ (-4 / (1/2) * 8 = -64) := by
  sorry

end arithmetic_calculations_l515_51559


namespace average_grade_year_before_l515_51567

/-- Calculates the average grade for the year before last, given the following conditions:
  * The student took 6 courses last year with an average grade of 100 points
  * The student took 5 courses the year before
  * The average grade for the entire two-year period was 72 points
-/
theorem average_grade_year_before (courses_last_year : Nat) (avg_grade_last_year : ℝ)
  (courses_year_before : Nat) (avg_grade_two_years : ℝ) :
  courses_last_year = 6 →
  avg_grade_last_year = 100 →
  courses_year_before = 5 →
  avg_grade_two_years = 72 →
  (courses_year_before * avg_grade_year_before + courses_last_year * avg_grade_last_year) /
    (courses_year_before + courses_last_year) = avg_grade_two_years →
  avg_grade_year_before = 38.4 :=
by
  sorry

#check average_grade_year_before

end average_grade_year_before_l515_51567


namespace set_A_equals_interval_rep_l515_51515

-- Define the set A
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 5 ∨ x > 10}

-- Define the interval representation
def intervalRep : Set ℝ := Set.Ici 0 ∩ Set.Iio 5 ∪ Set.Ioi 10

-- Theorem statement
theorem set_A_equals_interval_rep : A = intervalRep := by sorry

end set_A_equals_interval_rep_l515_51515


namespace sequence_matches_formula_l515_51555

-- Define the sequence
def a (n : ℕ) : ℚ := (-1)^(n+1) * (2*n + 1) / 2^n

-- State the theorem
theorem sequence_matches_formula : 
  a 1 = 3/2 ∧ a 2 = -5/4 ∧ a 3 = 7/8 ∧ a 4 = -9/16 := by
  sorry

end sequence_matches_formula_l515_51555


namespace correct_calculation_l515_51584

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y + 2 * y * x^2 = 5 * x^2 * y := by
  sorry

end correct_calculation_l515_51584


namespace kendras_cookies_l515_51572

/-- Kendra's cookie problem -/
theorem kendras_cookies (cookies_per_batch : ℕ) (family_members : ℕ) (batches : ℕ) (chips_per_cookie : ℕ)
  (h1 : cookies_per_batch = 12)
  (h2 : family_members = 4)
  (h3 : batches = 3)
  (h4 : chips_per_cookie = 2) :
  (batches * cookies_per_batch / family_members) * chips_per_cookie = 18 := by
  sorry

end kendras_cookies_l515_51572
