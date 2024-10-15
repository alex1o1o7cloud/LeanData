import Mathlib

namespace NUMINAMATH_CALUDE_fraction_unchanged_l2520_252058

theorem fraction_unchanged (x y : ℝ) : 
  x / (3 * x + y) = (3 * x) / (3 * (3 * x + y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l2520_252058


namespace NUMINAMATH_CALUDE_total_road_cost_l2520_252014

/-- Represents the dimensions of a rectangular lawn -/
structure LawnDimensions where
  length : ℕ
  width : ℕ

/-- Represents a road segment with its length and cost per square meter -/
structure RoadSegment where
  length : ℕ
  cost_per_sqm : ℕ

/-- Calculates the total cost of a road given its segments and width -/
def road_cost (segments : List RoadSegment) (width : ℕ) : ℕ :=
  segments.foldl (fun acc segment => acc + segment.length * segment.cost_per_sqm * width) 0

/-- The main theorem stating the total cost of traveling the two roads -/
theorem total_road_cost (lawn : LawnDimensions)
  (length_road : List RoadSegment) (breadth_road : List RoadSegment) (road_width : ℕ) :
  lawn.length = 100 ∧ lawn.width = 60 ∧
  road_width = 10 ∧
  length_road = [⟨30, 4⟩, ⟨40, 5⟩, ⟨30, 6⟩] ∧
  breadth_road = [⟨20, 3⟩, ⟨40, 2⟩] →
  road_cost length_road road_width + road_cost breadth_road road_width = 6400 := by
  sorry

end NUMINAMATH_CALUDE_total_road_cost_l2520_252014


namespace NUMINAMATH_CALUDE_candy_distribution_l2520_252084

theorem candy_distribution (tabitha julie carlos stan : ℕ) : 
  tabitha = 22 →
  julie = tabitha / 2 →
  carlos = 2 * stan →
  tabitha + julie + carlos + stan = 72 →
  stan = 13 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l2520_252084


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l2520_252015

/-- A line with slope -3 passing through (3,0) has y-intercept (0,9) -/
theorem y_intercept_of_line (m : ℝ) (x₀ y₀ : ℝ) :
  m = -3 →
  x₀ = 3 →
  y₀ = 0 →
  ∃ (b : ℝ), ∀ (x y : ℝ), y = m * (x - x₀) + y₀ → y = m * x + b →
  b = 9 ∧ 9 = m * 0 + b :=
by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l2520_252015


namespace NUMINAMATH_CALUDE_quaternary_1010_equals_68_l2520_252069

/-- Converts a quaternary (base 4) digit to its decimal value --/
def quaternaryToDecimal (digit : Nat) : Nat :=
  if digit < 4 then digit else 0

/-- Calculates the decimal value of a quaternary number represented as a list of digits --/
def quaternaryListToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + quaternaryToDecimal d * (4 ^ (digits.length - 1 - i))) 0

/-- The quaternary representation of the number to be converted --/
def quaternaryNumber : List Nat := [1, 0, 1, 0]

/-- Statement: The quaternary number 1010₍₄₎ is equal to the decimal number 68 --/
theorem quaternary_1010_equals_68 : 
  quaternaryListToDecimal quaternaryNumber = 68 := by
  sorry

end NUMINAMATH_CALUDE_quaternary_1010_equals_68_l2520_252069


namespace NUMINAMATH_CALUDE_team_selection_count_l2520_252005

def num_boys : ℕ := 7
def num_girls : ℕ := 10
def team_size : ℕ := 5
def min_girls : ℕ := 2

theorem team_selection_count :
  (Finset.sum (Finset.range (team_size - min_girls + 1))
    (λ k => Nat.choose num_girls (min_girls + k) * Nat.choose num_boys (team_size - (min_girls + k)))) = 5817 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_count_l2520_252005


namespace NUMINAMATH_CALUDE_point_not_in_fourth_quadrant_l2520_252073

theorem point_not_in_fourth_quadrant :
  ¬ ∃ a : ℝ, (a - 3 > 0 ∧ a + 3 < 0) := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_fourth_quadrant_l2520_252073


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2520_252076

theorem expression_simplification_and_evaluation :
  ∀ x : ℤ, -3 < x → x < 3 → x ≠ -1 → x ≠ 1 → x ≠ 0 →
  (((x^2 - 2*x + 1) / (x^2 - 1)) / ((x - 1) / (x + 1) - x + 1) = -1 / x) ∧
  (x = -2 → -1 / x = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2520_252076


namespace NUMINAMATH_CALUDE_kindergarten_tissues_l2520_252007

/-- The number of tissues brought by kindergartner groups -/
def total_tissues (group1 group2 group3 tissues_per_box : ℕ) : ℕ :=
  (group1 + group2 + group3) * tissues_per_box

/-- Theorem: The total number of tissues brought by the kindergartner groups is 1200 -/
theorem kindergarten_tissues :
  total_tissues 9 10 11 40 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_tissues_l2520_252007


namespace NUMINAMATH_CALUDE_tangent_product_simplification_l2520_252088

theorem tangent_product_simplification :
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_product_simplification_l2520_252088


namespace NUMINAMATH_CALUDE_wire_resistance_theorem_l2520_252045

/-- The resistance of a wire loop -/
def wire_loop_resistance (R : ℝ) : ℝ := R

/-- The distance between points A and B -/
def distance_AB : ℝ := 2

/-- The resistance of one meter of wire -/
def wire_resistance_per_meter (R : ℝ) : ℝ := R

/-- Theorem: The resistance of one meter of wire is equal to the total resistance of the wire loop -/
theorem wire_resistance_theorem (R : ℝ) :
  wire_loop_resistance R = wire_resistance_per_meter R :=
by sorry

end NUMINAMATH_CALUDE_wire_resistance_theorem_l2520_252045


namespace NUMINAMATH_CALUDE_sum_lower_bound_l2520_252033

theorem sum_lower_bound (x : ℕ → ℝ) (h_incr : ∀ n, x n ≤ x (n + 1)) (h_x0 : x 0 = 1) :
  (∑' n, x (n + 1) / (x n)^3) ≥ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l2520_252033


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l2520_252047

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, Prime n ∧ p = 2^n - 1 ∧ Prime p

theorem largest_mersenne_prime_under_500 :
  (∀ m : ℕ, is_mersenne_prime m ∧ m < 500 → m ≤ 127) ∧
  is_mersenne_prime 127 ∧
  127 < 500 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l2520_252047


namespace NUMINAMATH_CALUDE_sin_double_angle_special_case_l2520_252063

theorem sin_double_angle_special_case (θ : ℝ) (h : Real.tan θ + (Real.tan θ)⁻¹ = Real.sqrt 5) : 
  Real.sin (2 * θ) = (2 * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_special_case_l2520_252063


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_range_of_a_for_empty_intersection_l2520_252021

-- Define set A
def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3*a) < 0}

-- Part 1: Intersection when a = 1
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x | 2 < x ∧ x < 3} := by sorry

-- Part 2: Range of a when intersection is empty
theorem range_of_a_for_empty_intersection :
  ∀ a, A ∩ B a = ∅ ↔ a ≤ 2/3 ∨ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_range_of_a_for_empty_intersection_l2520_252021


namespace NUMINAMATH_CALUDE_green_face_probability_l2520_252023

/-- A structure representing a die with colored faces -/
structure ColoredDie where
  sides : ℕ
  green_faces : ℕ
  red_faces : ℕ
  blue_faces : ℕ
  yellow_faces : ℕ
  total_eq_sum : sides = green_faces + red_faces + blue_faces + yellow_faces

/-- The probability of rolling a specific color on a colored die -/
def roll_probability (d : ColoredDie) (color_faces : ℕ) : ℚ :=
  color_faces / d.sides

/-- Theorem: The probability of rolling a green face on our specific 12-sided die is 1/12 -/
theorem green_face_probability :
  let d : ColoredDie := {
    sides := 12,
    green_faces := 1,
    red_faces := 5,
    blue_faces := 4,
    yellow_faces := 2,
    total_eq_sum := by simp
  }
  roll_probability d d.green_faces = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_green_face_probability_l2520_252023


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l2520_252019

/-- Given an ellipse and a hyperbola with equations as specified,
    if they have the same foci, then m = ±1 -/
theorem ellipse_hyperbola_same_foci (m : ℝ) :
  (∀ x y : ℝ, x^2 / 4 + y^2 / m^2 = 1 → x^2 / m^2 - y^2 / 2 = 1 → 
    (∃ c : ℝ, c^2 = 4 - m^2 ∧ c^2 = m^2 + 2)) →
  m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l2520_252019


namespace NUMINAMATH_CALUDE_odd_function_value_l2520_252062

-- Define an odd function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem odd_function_value :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  (∀ x < 0, f x = x^3 + x + 1) →  -- f(x) = x^3 + x + 1 for x < 0
  f 2 = 9 := by sorry

end NUMINAMATH_CALUDE_odd_function_value_l2520_252062


namespace NUMINAMATH_CALUDE_five_numbers_satisfy_conditions_l2520_252043

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- Checks if a natural number is even -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Calculates the sum of digits of a two-digit number -/
def sumOfDigits (n : TwoDigitNumber) : ℕ :=
  (n.val / 10) + (n.val % 10)

/-- Performs the operation described in the problem -/
def operation (n : TwoDigitNumber) : ℕ :=
  n.val - sumOfDigits n

/-- Checks if the units digit of a number is 4 -/
def hasUnitsDigit4 (n : ℕ) : Prop :=
  n % 10 = 4

/-- The main theorem stating that exactly 5 two-digit numbers satisfy the conditions -/
theorem five_numbers_satisfy_conditions :
  ∃! (s : Finset TwoDigitNumber),
    (∀ n ∈ s, isEven n.val ∧ hasUnitsDigit4 (operation n)) ∧
    s.card = 5 :=
sorry

end NUMINAMATH_CALUDE_five_numbers_satisfy_conditions_l2520_252043


namespace NUMINAMATH_CALUDE_shorts_folded_l2520_252029

/-- Given the following:
  * There are 20 shirts and 8 pairs of shorts in total
  * 12 shirts are folded
  * 11 pieces of clothing remain to be folded
  Prove that 5 pairs of shorts were folded -/
theorem shorts_folded (total_shirts : ℕ) (total_shorts : ℕ) (folded_shirts : ℕ) (remaining_to_fold : ℕ) : ℕ :=
  by
  have h1 : total_shirts = 20 := by sorry
  have h2 : total_shorts = 8 := by sorry
  have h3 : folded_shirts = 12 := by sorry
  have h4 : remaining_to_fold = 11 := by sorry
  exact 5

end NUMINAMATH_CALUDE_shorts_folded_l2520_252029


namespace NUMINAMATH_CALUDE_beatrice_auction_tvs_l2520_252071

/-- The number of TVs Beatrice looked at on the auction site -/
def auction_tvs (in_person : ℕ) (online_multiplier : ℕ) (total : ℕ) : ℕ :=
  total - (in_person + online_multiplier * in_person)

/-- Proof that Beatrice looked at 10 TVs on the auction site -/
theorem beatrice_auction_tvs :
  auction_tvs 8 3 42 = 10 := by
  sorry

end NUMINAMATH_CALUDE_beatrice_auction_tvs_l2520_252071


namespace NUMINAMATH_CALUDE_restaurant_menu_fraction_l2520_252061

/-- Given a restaurant menu with vegan dishes and dietary restrictions, 
    calculate the fraction of suitable dishes. -/
theorem restaurant_menu_fraction (total_dishes : ℕ) 
  (vegan_dishes : ℕ) (restricted_vegan_dishes : ℕ) : 
  vegan_dishes = (3 : ℕ) * total_dishes / 10 →
  vegan_dishes = 9 →
  restricted_vegan_dishes = 7 →
  (vegan_dishes - restricted_vegan_dishes : ℚ) / total_dishes = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_menu_fraction_l2520_252061


namespace NUMINAMATH_CALUDE_triangle_cos_C_l2520_252025

theorem triangle_cos_C (A B C : Real) (h1 : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h2 : A + B + C = Real.pi) (h3 : Real.sin A = 4/5) (h4 : Real.cos B = 3/5) : 
  Real.cos C = 7/25 := by
sorry

end NUMINAMATH_CALUDE_triangle_cos_C_l2520_252025


namespace NUMINAMATH_CALUDE_complex_square_root_l2520_252051

theorem complex_square_root (z : ℂ) : z^2 = -5 - 12*I → z = 2 - 3*I ∨ z = -2 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_l2520_252051


namespace NUMINAMATH_CALUDE_inequality_sign_change_l2520_252022

theorem inequality_sign_change (a b : ℝ) (c : ℝ) (h1 : c < 0) (h2 : a < b) : c * b < c * a := by
  sorry

end NUMINAMATH_CALUDE_inequality_sign_change_l2520_252022


namespace NUMINAMATH_CALUDE_circle_line_intersection_theorem_l2520_252004

/-- Circle C with equation x^2 + (y-4)^2 = 4 -/
def C (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4

/-- Line l with equation y = kx -/
def l (k x y : ℝ) : Prop := y = k * x

/-- Point Q(m, n) is on segment MN -/
def Q_on_MN (m n x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ m = t * x₁ + (1 - t) * x₂ ∧ n = t * y₁ + (1 - t) * y₂

/-- The condition 2/|OQ|^2 = 1/|OM|^2 + 1/|ON|^2 -/
def harmonic_condition (m n x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  2 / (m^2 + n^2) = 1 / (x₁^2 + y₁^2) + 1 / (x₂^2 + y₂^2)

theorem circle_line_intersection_theorem
  (k m n x₁ y₁ x₂ y₂ : ℝ)
  (hC₁ : C x₁ y₁)
  (hC₂ : C x₂ y₂)
  (hl₁ : l k x₁ y₁)
  (hl₂ : l k x₂ y₂)
  (hQ : Q_on_MN m n x₁ y₁ x₂ y₂)
  (hHarmonic : harmonic_condition m n x₁ y₁ x₂ y₂)
  (hm : m ∈ Set.Ioo (-Real.sqrt 3) 0 ∪ Set.Ioo 0 (Real.sqrt 3)) :
  n = Real.sqrt (15 * m^2 + 180) / 5 :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_theorem_l2520_252004


namespace NUMINAMATH_CALUDE_f_increasing_l2520_252077

-- Define the function f(x) = x^3 + x
def f (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem f_increasing : ∀ (a b : ℝ), a < b → f a < f b := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_l2520_252077


namespace NUMINAMATH_CALUDE_ones_digit_of_triple_4567_l2520_252013

theorem ones_digit_of_triple_4567 : (3 * 4567) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_triple_4567_l2520_252013


namespace NUMINAMATH_CALUDE_value_of_y_l2520_252095

theorem value_of_y (x y : ℝ) (h1 : 1.5 * x = 0.25 * y) (h2 : x = 24) : y = 144 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l2520_252095


namespace NUMINAMATH_CALUDE_turnover_equation_l2520_252065

-- Define the monthly average growth rate
variable (x : ℝ)

-- Define the initial turnover in January (in units of 10,000 yuan)
def initial_turnover : ℝ := 200

-- Define the total turnover in the first quarter (in units of 10,000 yuan)
def total_turnover : ℝ := 1000

-- Theorem statement
theorem turnover_equation :
  initial_turnover + initial_turnover * (1 + x) + initial_turnover * (1 + x)^2 = total_turnover := by
  sorry

end NUMINAMATH_CALUDE_turnover_equation_l2520_252065


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2520_252048

/-- The eccentricity of a hyperbola with asymptotic lines y = ±(3/2)x is either √13/2 or √13/3 -/
theorem hyperbola_eccentricity (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  (b / a = 3 / 2 ∨ a / b = 3 / 2) →
  c^2 = a^2 + b^2 →
  (c / a = Real.sqrt 13 / 2 ∨ c / a = Real.sqrt 13 / 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2520_252048


namespace NUMINAMATH_CALUDE_certain_number_problem_l2520_252008

theorem certain_number_problem (x : ℝ) : ((7 * (x + 10)) / 5) - 5 = 44 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2520_252008


namespace NUMINAMATH_CALUDE_marching_band_members_l2520_252094

theorem marching_band_members : ∃! n : ℕ, 
  150 < n ∧ n < 250 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 4 ∧ 
  n % 7 = 2 ∧ 
  n = 163 := by
sorry

end NUMINAMATH_CALUDE_marching_band_members_l2520_252094


namespace NUMINAMATH_CALUDE_triple_composition_fixed_point_implies_fixed_point_l2520_252049

theorem triple_composition_fixed_point_implies_fixed_point
  (f : ℝ → ℝ) (hf : Continuous f)
  (h : ∃ x, f (f (f x)) = x) :
  ∃ x₀, f x₀ = x₀ := by
  sorry

end NUMINAMATH_CALUDE_triple_composition_fixed_point_implies_fixed_point_l2520_252049


namespace NUMINAMATH_CALUDE_unique_function_solution_l2520_252009

theorem unique_function_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x - f y) = f (f y) + x * f y + f x - 1) ↔ 
  (∀ x : ℝ, f x = 1 - x^2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_function_solution_l2520_252009


namespace NUMINAMATH_CALUDE_simplify_expression_l2520_252020

theorem simplify_expression (x : ℝ) : (3 * x)^4 - (4 * x) * (x^3) = 77 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2520_252020


namespace NUMINAMATH_CALUDE_bus_children_difference_l2520_252002

theorem bus_children_difference (initial : ℕ) (got_off : ℕ) (final : ℕ) : 
  initial = 5 → got_off = 63 → final = 14 → 
  ∃ (got_on : ℕ), got_on - got_off = 9 ∧ initial - got_off + got_on = final :=
sorry

end NUMINAMATH_CALUDE_bus_children_difference_l2520_252002


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l2520_252024

/-- A point P with coordinates (m, 4+2m) is in the third quadrant if and only if m < -2 -/
theorem point_in_third_quadrant (m : ℝ) : 
  (m < 0 ∧ 4 + 2*m < 0) ↔ m < -2 :=
sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l2520_252024


namespace NUMINAMATH_CALUDE_ratio_calculation_l2520_252039

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (2 * A + 3 * B) / (5 * C - 2 * A) = 12 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l2520_252039


namespace NUMINAMATH_CALUDE_value_of_x_l2520_252053

theorem value_of_x : ∃ x : ℝ, 
  x * 0.48 * 2.50 / (0.12 * 0.09 * 0.5) = 800.0000000000001 ∧ 
  abs (x - 3.6) < 0.0000000000001 :=
by sorry

end NUMINAMATH_CALUDE_value_of_x_l2520_252053


namespace NUMINAMATH_CALUDE_carla_teaches_23_students_l2520_252086

/-- The number of students Carla teaches -/
def total_students : ℕ :=
  let students_in_restroom : ℕ := 2
  let absent_students : ℕ := 3 * students_in_restroom - 1
  let total_desks : ℕ := 4 * 6
  let occupied_desks : ℕ := (2 * total_desks) / 3
  occupied_desks + students_in_restroom + absent_students

/-- Theorem stating that Carla teaches 23 students -/
theorem carla_teaches_23_students : total_students = 23 := by
  sorry

end NUMINAMATH_CALUDE_carla_teaches_23_students_l2520_252086


namespace NUMINAMATH_CALUDE_child_ticket_cost_l2520_252046

theorem child_ticket_cost 
  (adult_price : ℕ) 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (adult_attendance : ℕ) 
  (h1 : adult_price = 9)
  (h2 : total_tickets = 225)
  (h3 : total_revenue = 1875)
  (h4 : adult_attendance = 175) :
  ∃ (child_price : ℕ), 
    child_price * (total_tickets - adult_attendance) + 
    adult_price * adult_attendance = total_revenue ∧ 
    child_price = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l2520_252046


namespace NUMINAMATH_CALUDE_alex_jane_pen_difference_l2520_252035

/-- Calculates the number of pens Alex has after a given number of weeks -/
def alex_pens (initial_pens : ℕ) (weeks : ℕ) : ℕ :=
  initial_pens * (2 ^ weeks)

/-- The number of pens Jane has after a month -/
def jane_pens : ℕ := 16

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := 4

/-- The initial number of pens Alex has -/
def alex_initial_pens : ℕ := 4

theorem alex_jane_pen_difference :
  alex_pens alex_initial_pens weeks_in_month - jane_pens = 16 := by
  sorry


end NUMINAMATH_CALUDE_alex_jane_pen_difference_l2520_252035


namespace NUMINAMATH_CALUDE_homework_time_decrease_l2520_252006

theorem homework_time_decrease (initial_time final_time : ℝ) (x : ℝ) 
  (h_initial : initial_time = 100)
  (h_final : final_time = 70)
  (h_positive : 0 < x ∧ x < 1) :
  initial_time * (1 - x)^2 = final_time := by
  sorry

end NUMINAMATH_CALUDE_homework_time_decrease_l2520_252006


namespace NUMINAMATH_CALUDE_large_box_length_l2520_252052

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Theorem: If a large box with dimensions L × 14 × 16 can fit exactly 64 small boxes
    with dimensions 3 × 7 × 2, then L must be 12. -/
theorem large_box_length (L : ℝ) : 
  let largeBox : BoxDimensions := ⟨L, 14, 16⟩
  let smallBox : BoxDimensions := ⟨3, 7, 2⟩
  (boxVolume largeBox) / (boxVolume smallBox) = 64 → L = 12 := by
sorry

end NUMINAMATH_CALUDE_large_box_length_l2520_252052


namespace NUMINAMATH_CALUDE_count_valid_markings_l2520_252070

/-- Represents a valid marking of an 8x8 chessboard -/
def ValidMarking : Type := 
  { marking : Fin 8 → Fin 8 // 
    (∀ i j, i ≠ j → marking i ≠ marking j) ∧ 
    (∀ i, marking i ≠ 0 ∧ marking i ≠ 7) ∧
    (marking 0 ≠ 0 ∧ marking 0 ≠ 7) ∧
    (marking 7 ≠ 0 ∧ marking 7 ≠ 7) }

/-- The number of valid markings on an 8x8 chessboard -/
def numValidMarkings : ℕ := sorry

/-- The theorem stating the number of valid markings -/
theorem count_valid_markings : numValidMarkings = 21600 := by sorry

end NUMINAMATH_CALUDE_count_valid_markings_l2520_252070


namespace NUMINAMATH_CALUDE_building_occupancy_l2520_252042

/-- Given a building with a certain number of stories, apartments per floor, and people per apartment,
    calculate the total number of people housed in the building. -/
def total_people (stories : ℕ) (apartments_per_floor : ℕ) (people_per_apartment : ℕ) : ℕ :=
  stories * apartments_per_floor * people_per_apartment

/-- Theorem stating that a 25-story building with 4 apartments per floor and 2 people per apartment
    houses 200 people in total. -/
theorem building_occupancy :
  total_people 25 4 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_building_occupancy_l2520_252042


namespace NUMINAMATH_CALUDE_find_divisor_l2520_252085

theorem find_divisor (n : ℕ) (k : ℕ) (h1 : n + k = 8261966) (h2 : k = 11) :
  11 ∣ n + k :=
sorry

end NUMINAMATH_CALUDE_find_divisor_l2520_252085


namespace NUMINAMATH_CALUDE_square_roots_theorem_l2520_252068

theorem square_roots_theorem (x : ℝ) (m : ℝ) : 
  x > 0 → (2*m - 1)^2 = x → (2 - m)^2 = x → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_theorem_l2520_252068


namespace NUMINAMATH_CALUDE_polygon_has_13_sides_l2520_252041

/-- A polygon has n sides. The number of diagonals is equal to 5 times the number of sides. -/
def polygon_diagonals (n : ℕ) : Prop :=
  n * (n - 3) = 5 * n

/-- The polygon satisfying the given condition has 13 sides. -/
theorem polygon_has_13_sides : 
  ∃ (n : ℕ), polygon_diagonals n ∧ n = 13 :=
sorry

end NUMINAMATH_CALUDE_polygon_has_13_sides_l2520_252041


namespace NUMINAMATH_CALUDE_fraction_simplification_l2520_252096

theorem fraction_simplification (x : ℝ) (h : x ≠ 0) : (4 * x) / (x + 2 * x) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2520_252096


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2520_252026

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 7 + a 13 = 20 →
  a 9 + a 10 + a 11 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2520_252026


namespace NUMINAMATH_CALUDE_work_completion_time_l2520_252099

/-- Given that A can do a work in 6 days and B can do the same work in 12 days,
    prove that A and B working together can finish the work in 4 days. -/
theorem work_completion_time (work : ℝ) (days_A : ℝ) (days_B : ℝ)
    (h_work : work > 0)
    (h_days_A : days_A = 6)
    (h_days_B : days_B = 12) :
    work / (work / days_A + work / days_B) = 4 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2520_252099


namespace NUMINAMATH_CALUDE_trapezoid_vector_range_l2520_252080

/-- Right trapezoid ABCD with moving point P -/
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  P : ℝ × ℝ
  h : A.2 = D.2  -- AB ⟂ AD
  i : D.1 - A.1 = 1  -- AD = 1
  j : C.1 - D.1 = 1  -- DC = 1
  k : B.2 - A.2 = 3  -- AB = 3
  l : (P.1 - C.1)^2 + (P.2 - C.2)^2 ≤ 1  -- P is within or on the circle centered at C with radius 1

def vector_decomposition (t : Trapezoid) (α β : ℝ) : Prop :=
  t.P.1 - t.A.1 = α * (t.D.1 - t.A.1) + β * (t.B.1 - t.A.1) ∧
  t.P.2 - t.A.2 = α * (t.D.2 - t.A.2) + β * (t.B.2 - t.A.2)

theorem trapezoid_vector_range (t : Trapezoid) :
  ∃ (α β : ℝ), vector_decomposition t α β ∧ 
  (∀ (γ δ : ℝ), vector_decomposition t γ δ → 1 < γ + δ ∧ γ + δ < 5/3) :=
sorry

end NUMINAMATH_CALUDE_trapezoid_vector_range_l2520_252080


namespace NUMINAMATH_CALUDE_identity_condition_l2520_252075

theorem identity_condition (a b c : ℝ) : 
  (∀ x y z : ℝ, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) → 
  ((a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = 0 ∧ b = 0 ∧ c = -1)) := by
  sorry

end NUMINAMATH_CALUDE_identity_condition_l2520_252075


namespace NUMINAMATH_CALUDE_prime_factors_of_30_factorial_l2520_252003

theorem prime_factors_of_30_factorial (n : ℕ) : n = 30 →
  (Finset.filter (Nat.Prime) (Finset.range (n + 1))).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_of_30_factorial_l2520_252003


namespace NUMINAMATH_CALUDE_power_of_two_greater_than_square_l2520_252072

theorem power_of_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_greater_than_square_l2520_252072


namespace NUMINAMATH_CALUDE_linear_coefficient_of_example_quadratic_l2520_252066

/-- The coefficient of the linear term in a quadratic equation ax^2 + bx + c = 0 -/
def linearCoefficient (a b c : ℝ) : ℝ := b

theorem linear_coefficient_of_example_quadratic :
  linearCoefficient 1 (-5) (-2) = -5 := by sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_example_quadratic_l2520_252066


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2520_252091

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2520_252091


namespace NUMINAMATH_CALUDE_count_magic_numbers_l2520_252057

def is_magic_number (N : ℕ) : Prop :=
  ∀ m : ℕ, m > 0 → ∃ k : ℕ, k > 0 ∧ (m * 10^k + N) % N = 0

theorem count_magic_numbers :
  (∃! (L : List ℕ), 
    (∀ N ∈ L, N < 600 ∧ is_magic_number N) ∧
    (∀ N < 600, is_magic_number N → N ∈ L) ∧
    L.length = 13) :=
sorry

end NUMINAMATH_CALUDE_count_magic_numbers_l2520_252057


namespace NUMINAMATH_CALUDE_remainder_theorem_l2520_252038

def dividend (k : ℤ) (x : ℝ) : ℝ := 3 * x^3 + k * x^2 + 8 * x - 24

def divisor (x : ℝ) : ℝ := 3 * x + 4

theorem remainder_theorem (k : ℤ) :
  (∃ q : ℝ → ℝ, ∀ x, dividend k x = (divisor x) * (q x) + 5) ↔ k = 29 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2520_252038


namespace NUMINAMATH_CALUDE_total_cost_is_246_l2520_252092

/-- Represents a person's balloon collection --/
structure BalloonCollection where
  yellowCount : Nat
  yellowPrice : Nat
  redCount : Nat
  redPrice : Nat

/-- Calculates the total cost of a balloon collection --/
def totalCost (bc : BalloonCollection) : Nat :=
  bc.yellowCount * bc.yellowPrice + bc.redCount * bc.redPrice

/-- The balloon collections for each person --/
def fred : BalloonCollection := ⟨5, 3, 3, 4⟩
def sam : BalloonCollection := ⟨6, 4, 4, 5⟩
def mary : BalloonCollection := ⟨7, 5, 5, 6⟩
def susan : BalloonCollection := ⟨4, 6, 6, 7⟩
def tom : BalloonCollection := ⟨10, 2, 8, 3⟩

/-- Theorem: The total cost of all balloon collections is $246 --/
theorem total_cost_is_246 :
  totalCost fred + totalCost sam + totalCost mary + totalCost susan + totalCost tom = 246 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_246_l2520_252092


namespace NUMINAMATH_CALUDE_probability_of_correct_number_l2520_252027

def first_three_options : ℕ := 3

def last_five_digits : ℕ := 5
def repeating_digits : ℕ := 2

def total_combinations : ℕ := first_three_options * (Nat.factorial last_five_digits / Nat.factorial repeating_digits)

theorem probability_of_correct_number :
  (1 : ℚ) / total_combinations = 1 / 180 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_correct_number_l2520_252027


namespace NUMINAMATH_CALUDE_cube_sum_magnitude_l2520_252011

theorem cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2) 
  (h2 : Complex.abs (w^2 + z^2) = 8) : 
  Complex.abs (w^3 + z^3) = 20 := by sorry

end NUMINAMATH_CALUDE_cube_sum_magnitude_l2520_252011


namespace NUMINAMATH_CALUDE_sunflower_height_feet_l2520_252059

def feet_to_inches (feet : ℕ) : ℕ := feet * 12

def sister_height_inches : ℕ := feet_to_inches 4 + 3

def sunflower_height_inches : ℕ := sister_height_inches + 21

def inches_to_feet (inches : ℕ) : ℕ := inches / 12

theorem sunflower_height_feet :
  inches_to_feet sunflower_height_inches = 6 :=
sorry

end NUMINAMATH_CALUDE_sunflower_height_feet_l2520_252059


namespace NUMINAMATH_CALUDE_profit_percentage_l2520_252078

theorem profit_percentage (C S : ℝ) (h : 72 * C = 60 * S) : 
  (S - C) / C * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l2520_252078


namespace NUMINAMATH_CALUDE_expected_score_is_one_l2520_252097

/-- The number of black balls in the bag -/
def num_black : ℕ := 3

/-- The number of red balls in the bag -/
def num_red : ℕ := 1

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_black + num_red

/-- The score for drawing a black ball -/
def black_score : ℝ := 0

/-- The score for drawing a red ball -/
def red_score : ℝ := 2

/-- The expected value of the score when drawing two balls -/
def expected_score : ℝ := 1

/-- Theorem stating that the expected score when drawing two balls is 1 -/
theorem expected_score_is_one :
  let prob_two_black : ℝ := (num_black / total_balls) * ((num_black - 1) / (total_balls - 1))
  let prob_one_each : ℝ := (num_black / total_balls) * (num_red / (total_balls - 1)) +
                           (num_red / total_balls) * (num_black / (total_balls - 1))
  prob_two_black * (2 * black_score) + prob_one_each * (black_score + red_score) = expected_score :=
by sorry

end NUMINAMATH_CALUDE_expected_score_is_one_l2520_252097


namespace NUMINAMATH_CALUDE_thirty_percent_less_problem_l2520_252050

theorem thirty_percent_less_problem (x : ℝ) : 
  (63 = 90 - 0.3 * 90) → (x + 0.25 * x = 63) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_problem_l2520_252050


namespace NUMINAMATH_CALUDE_junior_score_l2520_252067

theorem junior_score (n : ℝ) (junior_score : ℝ) :
  n > 0 →
  0.1 * n * junior_score + 0.9 * n * 83 = n * 84 →
  junior_score = 93 := by
sorry

end NUMINAMATH_CALUDE_junior_score_l2520_252067


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2520_252034

/-- Given an arithmetic sequence {a_n} where a_5 + a_6 + a_7 = 15, 
    prove that a_3 + a_4 + ... + a_9 equals 35. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 5 + a 6 + a 7 = 15 →                                -- given condition
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=        -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2520_252034


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_is_gcd_l2520_252017

/-- The value of a pig in dollars -/
def pig_value : ℕ := 300

/-- The value of a goat in dollars -/
def goat_value : ℕ := 210

/-- The smallest positive debt that can be resolved using pigs and goats -/
def smallest_resolvable_debt : ℕ := 30

/-- Theorem stating that the smallest_resolvable_debt is the smallest positive integer
    that can be expressed as a linear combination of pig_value and goat_value -/
theorem smallest_resolvable_debt_is_gcd :
  smallest_resolvable_debt = Nat.gcd pig_value goat_value ∧
  ∀ d : ℕ, d > 0 → (∃ a b : ℤ, d = a * pig_value + b * goat_value) →
    d ≥ smallest_resolvable_debt :=
by sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_is_gcd_l2520_252017


namespace NUMINAMATH_CALUDE_stratified_sample_male_count_l2520_252081

theorem stratified_sample_male_count :
  let total_male : ℕ := 560
  let total_female : ℕ := 420
  let sample_size : ℕ := 280
  let total_students : ℕ := total_male + total_female
  let male_ratio : ℚ := total_male / total_students
  male_ratio * sample_size = 160 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_male_count_l2520_252081


namespace NUMINAMATH_CALUDE_largest_number_l2520_252030

theorem largest_number : ∀ (a b c d : ℝ), 
  a = -1 → b = 0 → c = 2 → d = Real.sqrt 3 →
  a < b ∧ b < d ∧ d < c :=
fun a b c d ha hb hc hd => by
  sorry

end NUMINAMATH_CALUDE_largest_number_l2520_252030


namespace NUMINAMATH_CALUDE_ben_egg_count_l2520_252037

/-- The number of trays Ben was given -/
def num_trays : ℕ := 7

/-- The number of eggs in each tray -/
def eggs_per_tray : ℕ := 10

/-- The total number of eggs Ben examined -/
def total_eggs : ℕ := num_trays * eggs_per_tray

theorem ben_egg_count : total_eggs = 70 := by
  sorry

end NUMINAMATH_CALUDE_ben_egg_count_l2520_252037


namespace NUMINAMATH_CALUDE_max_alpha_squared_l2520_252010

theorem max_alpha_squared (a b x y : ℝ) : 
  a > 0 → b > 0 → a = 2 * b →
  0 ≤ x → x < a → 0 ≤ y → y < b →
  a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a - x)^2 + (b - y)^2 →
  (∀ α : ℝ, α = a / b → α^2 ≤ 4) ∧ (∃ α : ℝ, α = a / b ∧ α^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_max_alpha_squared_l2520_252010


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2520_252055

theorem algebraic_expression_value :
  let a : ℝ := 1 + Real.sqrt 2
  let b : ℝ := Real.sqrt 3
  a^2 + b^2 - 2*a + 1 = 5 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2520_252055


namespace NUMINAMATH_CALUDE_product_digit_permutation_l2520_252079

theorem product_digit_permutation :
  ∃ (x : ℕ) (A B C D : ℕ),
    x * (x + 1) = 1000 * A + 100 * B + 10 * C + D ∧
    (x - 3) * (x - 2) = 1000 * C + 100 * A + 10 * B + D ∧
    (x - 30) * (x - 29) = 1000 * B + 100 * C + 10 * A + D ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
    x = 91 ∧ A = 8 ∧ B = 3 ∧ C = 7 ∧ D = 2 :=
by sorry

end NUMINAMATH_CALUDE_product_digit_permutation_l2520_252079


namespace NUMINAMATH_CALUDE_solve_flour_problem_l2520_252000

def flour_problem (total_flour sugar flour_to_add flour_already_in : ℕ) : Prop :=
  total_flour = 10 ∧
  sugar = 2 ∧
  flour_to_add = sugar + 1 ∧
  flour_already_in + flour_to_add = total_flour

theorem solve_flour_problem :
  ∃ (flour_already_in : ℕ), flour_problem 10 2 3 flour_already_in ∧ flour_already_in = 7 :=
by sorry

end NUMINAMATH_CALUDE_solve_flour_problem_l2520_252000


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l2520_252090

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l2520_252090


namespace NUMINAMATH_CALUDE_range_of_a_l2520_252083

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*a*x + a + 2 = 0}

-- State the theorem
theorem range_of_a (a : ℝ) : B a ⊆ A → a ∈ Set.Ioo (-1 : ℝ) (18/7 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2520_252083


namespace NUMINAMATH_CALUDE_min_difference_triangle_sides_l2520_252001

theorem min_difference_triangle_sides (PQ QR PR : ℕ) : 
  PQ + QR + PR = 2010 →
  PQ < QR →
  QR < PR →
  (∀ PQ' QR' PR' : ℕ, 
    PQ' + QR' + PR' = 2010 →
    PQ' < QR' →
    QR' < PR' →
    QR - PQ ≤ QR' - PQ') →
  QR - PQ = 1 := by
sorry

end NUMINAMATH_CALUDE_min_difference_triangle_sides_l2520_252001


namespace NUMINAMATH_CALUDE_wire_cut_square_circle_ratio_l2520_252032

theorem wire_cut_square_circle_ratio (x y : ℝ) (h : x > 0) (k : y > 0) : 
  (x^2 / 16 = y^2 / (4 * Real.pi)) → x / y = 2 / Real.sqrt Real.pi := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_square_circle_ratio_l2520_252032


namespace NUMINAMATH_CALUDE_vector_properties_l2520_252089

def a : ℝ × ℝ := (3, 0)
def b : ℝ × ℝ := (-5, 5)
def c (k : ℝ) : ℝ × ℝ := (2, k)

theorem vector_properties :
  (∃ θ : ℝ, θ = Real.pi * 3 / 4 ∧ 
    Real.cos θ = (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) ∧
  (∃ k : ℝ, b.1 / (c k).1 = b.2 / (c k).2 → k = -2) ∧
  (∃ k : ℝ, b.1 * (a.1 + (c k).1) + b.2 * (a.2 + (c k).2) = 0 → k = 5) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l2520_252089


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2520_252036

/-- Given a geometric sequence {a_n} with positive terms where a_1, (1/2)a_3, and 2a_2 form an arithmetic sequence, 
    the ratio a_10 / a_8 is equal to 3 + 2√2. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
    (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
    (h_arithmetic : 2 * ((1/2) * a 3) = a 1 + 2 * a 2) :
    a 10 / a 8 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2520_252036


namespace NUMINAMATH_CALUDE_card_collection_difference_l2520_252044

/-- Represents the number of cards each person has -/
structure CardCollection where
  heike : ℕ
  anton : ℕ
  ann : ℕ
  bertrand : ℕ
  carla : ℕ
  desmond : ℕ

/-- The conditions of the card collection problem -/
def card_collection_conditions (c : CardCollection) : Prop :=
  c.anton = 3 * c.heike ∧
  c.ann = 6 * c.heike ∧
  c.bertrand = 2 * c.heike ∧
  c.carla = 4 * c.heike ∧
  c.desmond = 8 * c.heike ∧
  c.ann = 60

/-- The theorem stating the difference between the highest and lowest number of cards -/
theorem card_collection_difference (c : CardCollection) 
  (h : card_collection_conditions c) : 
  max c.anton (max c.ann (max c.bertrand (max c.carla c.desmond))) - 
  min c.heike (min c.anton (min c.ann (min c.bertrand (min c.carla c.desmond)))) = 70 := by
  sorry

end NUMINAMATH_CALUDE_card_collection_difference_l2520_252044


namespace NUMINAMATH_CALUDE_smallest_angle_theorem_l2520_252056

/-- The smallest positive angle y in degrees that satisfies the equation 
    9 sin(y) cos³(y) - 9 sin³(y) cos(y) = 3√2 is 22.5° -/
theorem smallest_angle_theorem : 
  ∃ y : ℝ, y > 0 ∧ y < 360 ∧ 
  (9 * Real.sin y * (Real.cos y)^3 - 9 * (Real.sin y)^3 * Real.cos y = 3 * Real.sqrt 2) ∧
  (∀ z : ℝ, z > 0 ∧ z < y → 
    9 * Real.sin z * (Real.cos z)^3 - 9 * (Real.sin z)^3 * Real.cos z ≠ 3 * Real.sqrt 2) ∧
  y = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_theorem_l2520_252056


namespace NUMINAMATH_CALUDE_quadratic_function_value_l2520_252093

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_value (a b c : ℝ) :
  f a b c 1 = 7 → f a b c 3 = 19 → f a b c 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l2520_252093


namespace NUMINAMATH_CALUDE_trigonometric_sum_l2520_252018

theorem trigonometric_sum (x : ℝ) : 
  (Real.cos x + Real.cos (x + 2 * Real.pi / 3) + Real.cos (x + 4 * Real.pi / 3) = 0) ∧
  (Real.sin x + Real.sin (x + 2 * Real.pi / 3) + Real.sin (x + 4 * Real.pi / 3) = 0) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_l2520_252018


namespace NUMINAMATH_CALUDE_inequalities_for_M_l2520_252098

-- Define the set M
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- State the theorem
theorem inequalities_for_M (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2*|a - b|) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_for_M_l2520_252098


namespace NUMINAMATH_CALUDE_ring_area_between_circles_l2520_252064

theorem ring_area_between_circles (π : ℝ) (h : π > 0) :
  let r₁ : ℝ := 12
  let r₂ : ℝ := 7
  let area_larger := π * r₁^2
  let area_smaller := π * r₂^2
  area_larger - area_smaller = 95 * π :=
by sorry

end NUMINAMATH_CALUDE_ring_area_between_circles_l2520_252064


namespace NUMINAMATH_CALUDE_expected_heads_theorem_l2520_252031

/-- The number of coins -/
def num_coins : ℕ := 100

/-- The number of maximum flips -/
def max_flips : ℕ := 4

/-- The probability of a coin showing heads on a single flip -/
def prob_heads : ℚ := 1/2

/-- The probability of a coin showing heads after all flips -/
def prob_heads_after_flips : ℚ := 15/16

/-- The expected number of coins showing heads after all flips -/
def expected_heads : ℚ := num_coins * prob_heads_after_flips

theorem expected_heads_theorem :
  ⌊expected_heads⌋ = 94 :=
sorry

end NUMINAMATH_CALUDE_expected_heads_theorem_l2520_252031


namespace NUMINAMATH_CALUDE_pages_multiple_l2520_252082

theorem pages_multiple (beatrix_pages cristobal_extra_pages : ℕ) 
  (h1 : beatrix_pages = 704)
  (h2 : cristobal_extra_pages = 1423)
  (h3 : ∃ x : ℕ, x * beatrix_pages + 15 = cristobal_extra_pages) :
  ∃ x : ℕ, x * beatrix_pages + 15 = cristobal_extra_pages ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_pages_multiple_l2520_252082


namespace NUMINAMATH_CALUDE_inequality_proof_l2520_252016

theorem inequality_proof (a b : ℝ) (ha : |a| ≤ Real.sqrt 3) (hb : |b| ≤ Real.sqrt 3) :
  Real.sqrt 3 * |a + b| ≤ |a * b + 3| := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2520_252016


namespace NUMINAMATH_CALUDE_total_jellybeans_l2520_252060

def dozen : ℕ := 12

def caleb_jellybeans : ℕ := 3 * dozen

def sophie_jellybeans : ℕ := caleb_jellybeans / 2

theorem total_jellybeans : caleb_jellybeans + sophie_jellybeans = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_jellybeans_l2520_252060


namespace NUMINAMATH_CALUDE_drill_bits_purchase_l2520_252040

theorem drill_bits_purchase (cost_per_set : ℝ) (tax_rate : ℝ) (total_paid : ℝ) 
  (h1 : cost_per_set = 6)
  (h2 : tax_rate = 0.1)
  (h3 : total_paid = 33) :
  ∃ (num_sets : ℕ), (cost_per_set * (num_sets : ℝ)) * (1 + tax_rate) = total_paid ∧ num_sets = 5 := by
  sorry

end NUMINAMATH_CALUDE_drill_bits_purchase_l2520_252040


namespace NUMINAMATH_CALUDE_square_product_sequence_max_l2520_252054

/-- A sequence of natural numbers where each pair of consecutive numbers has a perfect square product -/
def SquareProductSequence (a : ℕ → ℕ) : Prop :=
  ∀ n, ∃ k, (a n) * (a (n + 1)) = k^2

theorem square_product_sequence_max (a : ℕ → ℕ) :
  (∀ i j, i ≠ j → a i ≠ a j) →  -- All numbers are different
  (SquareProductSequence a) →   -- Product of consecutive pairs is a perfect square
  (a 0 = 42) →                  -- First number is 42
  (∃ n, n < 20 ∧ a n ≥ 16800) :=  -- At least one of the first 20 numbers is ≥ 16800
by sorry

end NUMINAMATH_CALUDE_square_product_sequence_max_l2520_252054


namespace NUMINAMATH_CALUDE_rectangle_area_l2520_252087

/-- The area of a rectangle with dimensions 0.5 meters and 0.36 meters is 1800 square centimeters. -/
theorem rectangle_area : 
  let length_m : ℝ := 0.5
  let width_m : ℝ := 0.36
  let cm_per_m : ℝ := 100
  let length_cm := length_m * cm_per_m
  let width_cm := width_m * cm_per_m
  length_cm * width_cm = 1800 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2520_252087


namespace NUMINAMATH_CALUDE_equation_solutions_l2520_252012

theorem equation_solutions :
  (∃ x : ℚ, (3 : ℚ) / 5 - (5 : ℚ) / 8 * x = (2 : ℚ) / 5 ∧ x = (8 : ℚ) / 25) ∧
  (∃ x : ℚ, 7 * (x - 2) = 8 * (x - 4) ∧ x = 18) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2520_252012


namespace NUMINAMATH_CALUDE_inverse_g_at_negative_one_l2520_252028

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^3 - 5

-- State the theorem
theorem inverse_g_at_negative_one :
  Function.invFun g (-1) = 1 :=
sorry

end NUMINAMATH_CALUDE_inverse_g_at_negative_one_l2520_252028


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l2520_252074

-- Define the product
def product : ℕ := 45 * 320 * 60

-- Define a function to count trailing zeros
def count_trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 10 = 0 then 1 + count_trailing_zeros (n / 10)
  else 0

-- Theorem statement
theorem product_trailing_zeros :
  count_trailing_zeros product = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l2520_252074
