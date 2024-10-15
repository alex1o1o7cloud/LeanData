import Mathlib

namespace NUMINAMATH_CALUDE_aqua_opposite_red_l2140_214049

-- Define the set of colors
inductive Color : Type
  | Red | White | Green | Brown | Aqua | Purple

-- Define a cube as a function from face positions to colors
def Cube := Fin 6 → Color

-- Define face positions
def top : Fin 6 := 0
def bottom : Fin 6 := 1
def front : Fin 6 := 2
def back : Fin 6 := 3
def right : Fin 6 := 4
def left : Fin 6 := 5

-- Define the conditions of the problem
def cube_conditions (c : Cube) : Prop :=
  (c top = Color.Brown) ∧
  (c right = Color.Green) ∧
  (c front = Color.Red ∨ c front = Color.White ∨ c front = Color.Purple) ∧
  (c back = Color.Aqua)

-- State the theorem
theorem aqua_opposite_red (c : Cube) :
  cube_conditions c → c front = Color.Red :=
by sorry

end NUMINAMATH_CALUDE_aqua_opposite_red_l2140_214049


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2140_214052

theorem negation_of_universal_proposition :
  ¬(∀ n : ℤ, n % 5 = 0 → Odd n) ↔ ∃ n : ℤ, n % 5 = 0 ∧ ¬(Odd n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2140_214052


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_l2140_214067

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x + 2| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -6 ∨ x > 2/3} := by sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (7/2)*t} = {t : ℝ | 3/2 ≤ t ∧ t ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_l2140_214067


namespace NUMINAMATH_CALUDE_triangle_angle_R_l2140_214071

theorem triangle_angle_R (P Q R : Real) (h1 : 2 * Real.sin P + 5 * Real.cos Q = 4) 
  (h2 : 5 * Real.sin Q + 2 * Real.cos P = 3) 
  (h3 : P + Q + R = Real.pi) : R = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_R_l2140_214071


namespace NUMINAMATH_CALUDE_three_points_in_circle_l2140_214096

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in a 2D plane -/
structure Square where
  side : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point is inside a square -/
def Point.inSquare (p : Point) (s : Square) : Prop :=
  0 ≤ p.x ∧ p.x ≤ s.side ∧ 0 ≤ p.y ∧ p.y ≤ s.side

/-- Check if a point is inside a circle -/
def Point.inCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

/-- The main theorem -/
theorem three_points_in_circle (points : Finset Point) (s : Square) :
  s.side = 1 →
  points.card = 51 →
  ∀ p ∈ points, p.inSquare s →
  ∃ (c : Circle) (p1 p2 p3 : Point),
    c.radius = 1/7 ∧
    p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    p1.inCircle c ∧ p2.inCircle c ∧ p3.inCircle c :=
sorry


end NUMINAMATH_CALUDE_three_points_in_circle_l2140_214096


namespace NUMINAMATH_CALUDE_largest_y_value_l2140_214069

theorem largest_y_value (x y : ℝ) 
  (eq1 : x^2 + 3*x*y - y^2 = 27)
  (eq2 : 3*x^2 - x*y + y^2 = 27) :
  ∃ (y_max : ℝ), y_max = 3 ∧ 
  (∀ (y' : ℝ), (∃ (x' : ℝ), x'^2 + 3*x'*y' - y'^2 = 27 ∧ 
                             3*x'^2 - x'*y' + y'^2 = 27) → 
                y' ≤ y_max) :=
sorry

end NUMINAMATH_CALUDE_largest_y_value_l2140_214069


namespace NUMINAMATH_CALUDE_min_diff_same_last_two_digits_l2140_214042

/-- Given positive integers m and n where m > n, if the last two digits of 9^m and 9^n are the same, 
    then the minimum value of m - n is 10. -/
theorem min_diff_same_last_two_digits (m n : ℕ) : 
  m > n → 
  (∃ k : ℕ, 9^m ≡ k [ZMOD 100] ∧ 9^n ≡ k [ZMOD 100]) → 
  (∀ p q : ℕ, p > q → (∃ j : ℕ, 9^p ≡ j [ZMOD 100] ∧ 9^q ≡ j [ZMOD 100]) → m - n ≤ p - q) → 
  m - n = 10 := by
sorry

end NUMINAMATH_CALUDE_min_diff_same_last_two_digits_l2140_214042


namespace NUMINAMATH_CALUDE_science_club_problem_l2140_214083

theorem science_club_problem (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ)
  (h1 : total = 75)
  (h2 : biology = 42)
  (h3 : chemistry = 38)
  (h4 : both = 25) :
  total - (biology + chemistry - both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_science_club_problem_l2140_214083


namespace NUMINAMATH_CALUDE_line_properties_l2140_214011

/-- The slope of the line sqrt(3)x - y - 1 = 0 is sqrt(3) and its inclination angle is 60° --/
theorem line_properties :
  let line := fun (x y : ℝ) => Real.sqrt 3 * x - y - 1 = 0
  ∃ (m θ : ℝ),
    (∀ x y, line x y → y = m * x - 1) ∧ 
    m = Real.sqrt 3 ∧
    θ = 60 * π / 180 ∧
    Real.tan θ = m :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l2140_214011


namespace NUMINAMATH_CALUDE_bicycle_trip_average_speed_l2140_214039

/-- Calculates the average speed of a bicycle trip with varying conditions -/
theorem bicycle_trip_average_speed :
  let total_distance : ℝ := 500
  let flat_road_distance : ℝ := 100
  let flat_road_speed : ℝ := 20
  let uphill_distance : ℝ := 50
  let uphill_speed : ℝ := 10
  let flat_terrain_distance : ℝ := 200
  let flat_terrain_speed : ℝ := 15
  let headwind_distance : ℝ := 150
  let headwind_speed : ℝ := 12
  let rest_time_1 : ℝ := 0.5  -- 30 minutes in hours
  let rest_time_2 : ℝ := 1/3  -- 20 minutes in hours
  let rest_time_3 : ℝ := 2/3  -- 40 minutes in hours
  
  let total_time : ℝ := 
    flat_road_distance / flat_road_speed +
    uphill_distance / uphill_speed +
    flat_terrain_distance / flat_terrain_speed +
    headwind_distance / headwind_speed +
    rest_time_1 + rest_time_2 + rest_time_3
  
  let average_speed : ℝ := total_distance / total_time
  
  ∃ ε > 0, |average_speed - 13.4| < ε :=
by sorry

end NUMINAMATH_CALUDE_bicycle_trip_average_speed_l2140_214039


namespace NUMINAMATH_CALUDE_linear_function_constraint_l2140_214065

/-- A linear function y = kx + b -/
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

/-- Predicate to check if a point (x, y) is in the third quadrant -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- Predicate to check if a point (x, y) is the origin -/
def is_origin (x y : ℝ) : Prop := x = 0 ∧ y = 0

/-- Theorem stating that if a linear function doesn't pass through the third quadrant or origin, 
    then k < 0 and b > 0 -/
theorem linear_function_constraint (k b : ℝ) :
  (∀ x : ℝ, ¬(in_third_quadrant x (linear_function k b x))) ∧
  (∀ x : ℝ, ¬(is_origin x (linear_function k b x))) →
  k < 0 ∧ b > 0 :=
sorry

end NUMINAMATH_CALUDE_linear_function_constraint_l2140_214065


namespace NUMINAMATH_CALUDE_sophomore_selection_l2140_214076

/-- Calculates the number of sophomores selected for a study tour using proportional allocation -/
theorem sophomore_selection (freshmen sophomore junior total_spots : ℕ) : 
  freshmen = 240 →
  sophomore = 260 →
  junior = 300 →
  total_spots = 40 →
  (sophomore * total_spots) / (freshmen + sophomore + junior) = 26 := by
  sorry

end NUMINAMATH_CALUDE_sophomore_selection_l2140_214076


namespace NUMINAMATH_CALUDE_tan_difference_l2140_214057

theorem tan_difference (α β : Real) (h1 : Real.tan α = 3) (h2 : Real.tan β = 4/3) :
  Real.tan (α - β) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_l2140_214057


namespace NUMINAMATH_CALUDE_inequality_minimum_a_l2140_214033

theorem inequality_minimum_a : 
  (∀ a : ℝ, (∀ x : ℝ, x > a → (2*x^2 - 2*a*x + 2) / (x - a) ≥ 5) → a ≥ 1/2) ∧
  (∃ a : ℝ, a = 1/2 ∧ ∀ x : ℝ, x > a → (2*x^2 - 2*a*x + 2) / (x - a) ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_minimum_a_l2140_214033


namespace NUMINAMATH_CALUDE_min_value_sum_of_fractions_l2140_214094

theorem min_value_sum_of_fractions (a b : ℤ) (h : a ≠ b) :
  (a^2 + b^2 : ℚ) / (a^2 - b^2) + (a^2 - b^2 : ℚ) / (a^2 + b^2) ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_fractions_l2140_214094


namespace NUMINAMATH_CALUDE_system_solution_l2140_214089

theorem system_solution : ∃ (x y : ℚ), 
  (7 * x = -10 - 3 * y) ∧ 
  (4 * x = 5 * y - 35) ∧ 
  (x = -155 / 47) ∧ 
  (y = 205 / 47) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2140_214089


namespace NUMINAMATH_CALUDE_division_value_problem_l2140_214082

theorem division_value_problem (x : ℝ) (h : (5 / x) * 12 = 10) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_value_problem_l2140_214082


namespace NUMINAMATH_CALUDE_circle_center_correct_l2140_214045

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 6*y = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, -3)

/-- Theorem: The center of the circle defined by circle_equation is circle_center -/
theorem circle_center_correct :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l2140_214045


namespace NUMINAMATH_CALUDE_zero_subset_M_l2140_214009

def M : Set ℝ := {x | x > -2}

theorem zero_subset_M : {0} ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_zero_subset_M_l2140_214009


namespace NUMINAMATH_CALUDE_system_negative_solution_l2140_214001

theorem system_negative_solution (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧
    a * x + b * y = c ∧
    b * x + c * y = a ∧
    c * x + a * y = b) ↔
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_system_negative_solution_l2140_214001


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l2140_214050

-- Define a function to convert from binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert from ternary to decimal
def ternary_to_decimal (ternary : List ℕ) : ℕ :=
  ternary.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

-- Define the binary and ternary numbers
def binary_num : List Bool := [true, true, false, true]
def ternary_num : List ℕ := [2, 2, 1]

-- State the theorem
theorem product_of_binary_and_ternary :
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 187 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l2140_214050


namespace NUMINAMATH_CALUDE_melissa_points_per_game_l2140_214075

-- Define the number of games
def num_games : ℕ := 3

-- Define the total points scored
def total_points : ℕ := 81

-- Define the points per game as a function
def points_per_game : ℕ := total_points / num_games

-- Theorem to prove
theorem melissa_points_per_game : points_per_game = 27 := by
  sorry

end NUMINAMATH_CALUDE_melissa_points_per_game_l2140_214075


namespace NUMINAMATH_CALUDE_outfit_combinations_l2140_214063

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (restricted_combinations : ℕ) :
  shirts = 5 →
  pants = 4 →
  restricted_combinations = 1 →
  shirts * pants - restricted_combinations = 19 :=
by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l2140_214063


namespace NUMINAMATH_CALUDE_min_value_expression_l2140_214046

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (2 * a / b) + (3 * b / c) + (4 * c / a) ≥ 9 ∧
  ((2 * a / b) + (3 * b / c) + (4 * c / a) = 9 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2140_214046


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2140_214031

theorem diophantine_equation_solutions : 
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 4 * p.2 = 806 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 807) (Finset.range 807))).card = 67 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2140_214031


namespace NUMINAMATH_CALUDE_complex_sum_equality_l2140_214018

/-- Given complex numbers a and b, prove that 2a + 3b = 1 + i -/
theorem complex_sum_equality (a b : ℂ) (ha : a = 2 - I) (hb : b = -1 + I) :
  2 * a + 3 * b = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l2140_214018


namespace NUMINAMATH_CALUDE_min_value_of_f_inequality_condition_l2140_214077

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 4| + |x + 2|

-- Statement 1: The minimum value of f(x) is 4
theorem min_value_of_f : ∃ (x : ℝ), f x = 4 ∧ ∀ (y : ℝ), f y ≥ 4 :=
sorry

-- Statement 2: f(x) ≥ |a+4| - |a-3| for all x if and only if a ≤ 3/2
theorem inequality_condition (a : ℝ) : 
  (∀ (x : ℝ), f x ≥ |a + 4| - |a - 3|) ↔ a ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_inequality_condition_l2140_214077


namespace NUMINAMATH_CALUDE_james_cycling_distance_l2140_214025

theorem james_cycling_distance (speed : ℝ) (morning_time : ℝ) (afternoon_time : ℝ) 
  (h1 : speed = 8)
  (h2 : morning_time = 2.5)
  (h3 : afternoon_time = 1.5) :
  speed * morning_time + speed * afternoon_time = 32 :=
by sorry

end NUMINAMATH_CALUDE_james_cycling_distance_l2140_214025


namespace NUMINAMATH_CALUDE_pig_teeth_count_l2140_214015

/-- The number of teeth a dog has -/
def dog_teeth : ℕ := 42

/-- The number of teeth a cat has -/
def cat_teeth : ℕ := 30

/-- The number of dogs Vann will clean -/
def num_dogs : ℕ := 5

/-- The number of cats Vann will clean -/
def num_cats : ℕ := 10

/-- The number of pigs Vann will clean -/
def num_pigs : ℕ := 7

/-- The total number of teeth Vann will clean -/
def total_teeth : ℕ := 706

/-- Theorem stating that pigs have 28 teeth each -/
theorem pig_teeth_count : 
  (total_teeth - (num_dogs * dog_teeth + num_cats * cat_teeth)) / num_pigs = 28 := by
  sorry

end NUMINAMATH_CALUDE_pig_teeth_count_l2140_214015


namespace NUMINAMATH_CALUDE_chocolate_difference_l2140_214027

theorem chocolate_difference (robert_chocolates nickel_chocolates : ℕ) 
  (h1 : robert_chocolates = 7)
  (h2 : nickel_chocolates = 3) : 
  robert_chocolates - nickel_chocolates = 4 := by
sorry

end NUMINAMATH_CALUDE_chocolate_difference_l2140_214027


namespace NUMINAMATH_CALUDE_expression_evaluation_l2140_214097

theorem expression_evaluation : 
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) + Real.sqrt 12 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2140_214097


namespace NUMINAMATH_CALUDE_binary_channel_properties_l2140_214002

/-- A binary channel with error probabilities α and β -/
structure BinaryChannel where
  α : ℝ
  β : ℝ
  α_pos : 0 < α
  α_lt_one : α < 1
  β_pos : 0 < β
  β_lt_one : β < 1

/-- Probability of receiving 1,0,1 when sending 1,0,1 in single transmission -/
def prob_single_101 (bc : BinaryChannel) : ℝ := (1 - bc.α) * (1 - bc.β)^2

/-- Probability of receiving 1,0,1 when sending 1 in triple transmission -/
def prob_triple_101 (bc : BinaryChannel) : ℝ := bc.β * (1 - bc.β)^2

/-- Probability of decoding 1 when sending 1 in triple transmission -/
def prob_triple_decode_1 (bc : BinaryChannel) : ℝ := 3 * bc.β * (1 - bc.β)^2 + (1 - bc.β)^3

/-- Probability of decoding 0 when sending 0 in single transmission -/
def prob_single_decode_0 (bc : BinaryChannel) : ℝ := 1 - bc.α

/-- Probability of decoding 0 when sending 0 in triple transmission -/
def prob_triple_decode_0 (bc : BinaryChannel) : ℝ := 3 * bc.α * (1 - bc.α)^2 + (1 - bc.α)^3

theorem binary_channel_properties (bc : BinaryChannel) :
  prob_single_101 bc = (1 - bc.α) * (1 - bc.β)^2 ∧
  prob_triple_101 bc = bc.β * (1 - bc.β)^2 ∧
  prob_triple_decode_1 bc = 3 * bc.β * (1 - bc.β)^2 + (1 - bc.β)^3 ∧
  (bc.α < 0.5 → prob_triple_decode_0 bc > prob_single_decode_0 bc) :=
by sorry

end NUMINAMATH_CALUDE_binary_channel_properties_l2140_214002


namespace NUMINAMATH_CALUDE_a_range_l2140_214062

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 2 * a * x - 8

-- State the theorem
theorem a_range (a : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, ¬Monotone (f a)) →
  a ∈ Set.Ioo 3 6 :=
by
  sorry

end NUMINAMATH_CALUDE_a_range_l2140_214062


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l2140_214053

theorem tan_double_angle_special_case (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l2140_214053


namespace NUMINAMATH_CALUDE_remainder_after_adding_2024_l2140_214006

theorem remainder_after_adding_2024 (n : ℤ) (h : n % 8 = 3) : (n + 2024) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2024_l2140_214006


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l2140_214005

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + y + z = 1) :
  2 * x^2 + 3 * y^2 + z^2 ≥ 6/11 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l2140_214005


namespace NUMINAMATH_CALUDE_calculation_proof_l2140_214091

theorem calculation_proof (h1 : 9 + 3/4 = 9.75) (h2 : 975/100 = 9.75) (h3 : 0.142857 = 1/7) :
  4/7 * (9 + 3/4) + 9.75 * 2/7 + 0.142857 * 975/100 = 9.75 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2140_214091


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2140_214037

theorem line_segment_endpoint (x : ℝ) :
  x > 0 →
  (((x - 2)^2 + (6 - 2)^2).sqrt = 7) →
  x = 2 + Real.sqrt 33 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2140_214037


namespace NUMINAMATH_CALUDE_first_cut_ratio_l2140_214043

/-- Proves the ratio of the first cut rope to the initial rope length is 1/2 -/
theorem first_cut_ratio (initial_length : ℝ) (final_piece_length : ℝ) : 
  initial_length = 100 → 
  final_piece_length = 5 → 
  (initial_length / 2) / initial_length = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_first_cut_ratio_l2140_214043


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l2140_214036

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l2140_214036


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l2140_214085

theorem multiplication_addition_equality : 45 * 72 + 28 * 45 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l2140_214085


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2140_214003

theorem cubic_roots_sum (p q r : ℝ) : 
  (p^3 - 5*p^2 + 9*p - 7 = 0) →
  (q^3 - 5*q^2 + 9*q - 7 = 0) →
  (r^3 - 5*r^2 + 9*r - 7 = 0) →
  ∃ (u v : ℝ), ((p+q)^3 + u*(p+q)^2 + v*(p+q) + (-13) = 0) ∧
               ((q+r)^3 + u*(q+r)^2 + v*(q+r) + (-13) = 0) ∧
               ((r+p)^3 + u*(r+p)^2 + v*(r+p) + (-13) = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2140_214003


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2140_214080

theorem polynomial_factorization (m x y : ℝ) : 4*m*x^2 - m*y^2 = m*(2*x+y)*(2*x-y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2140_214080


namespace NUMINAMATH_CALUDE_pyramid_volume_l2140_214020

theorem pyramid_volume (total_area : ℝ) (base_area : ℝ) (triangular_face_area : ℝ) :
  total_area = 648 ∧
  triangular_face_area = (1/3) * base_area ∧
  total_area = base_area + 4 * triangular_face_area →
  ∃ (s h : ℝ),
    s > 0 ∧
    h > 0 ∧
    base_area = s^2 ∧
    (1/3) * s^2 * h = 486 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l2140_214020


namespace NUMINAMATH_CALUDE_water_amount_for_scaled_solution_l2140_214017

theorem water_amount_for_scaled_solution 
  (chemical_a : Real) 
  (water : Real) 
  (total : Real) 
  (new_total : Real) 
  (h1 : chemical_a + water = total)
  (h2 : chemical_a = 0.07)
  (h3 : water = 0.03)
  (h4 : total = 0.1)
  (h5 : new_total = 0.6) : 
  (water / total) * new_total = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_water_amount_for_scaled_solution_l2140_214017


namespace NUMINAMATH_CALUDE_sqrt_62_plus_24_sqrt_11_l2140_214028

theorem sqrt_62_plus_24_sqrt_11 :
  ∃ (a b c : ℤ), 
    (∀ (n : ℕ), n > 1 → ¬(∃ (k : ℕ), c = n^2 * k)) →
    Real.sqrt (62 + 24 * Real.sqrt 11) = a + b * Real.sqrt c ∧
    a = 6 ∧ b = 2 ∧ c = 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_62_plus_24_sqrt_11_l2140_214028


namespace NUMINAMATH_CALUDE_lizzy_money_theorem_l2140_214087

/-- Calculates the final amount Lizzy has after lending money and receiving it back with interest -/
def final_amount (initial : ℝ) (loan : ℝ) (interest_rate : ℝ) : ℝ :=
  initial - loan + loan * (1 + interest_rate)

/-- Theorem stating that given the specific conditions, Lizzy will have $33 -/
theorem lizzy_money_theorem :
  let initial := 30
  let loan := 15
  let interest_rate := 0.2
  final_amount initial loan interest_rate = 33 := by
  sorry

end NUMINAMATH_CALUDE_lizzy_money_theorem_l2140_214087


namespace NUMINAMATH_CALUDE_log_inequality_range_l2140_214093

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_inequality_range (a : ℝ) :
  log a (2/5) < 1 ↔ (0 < a ∧ a < 2/5) ∨ a > 1 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_range_l2140_214093


namespace NUMINAMATH_CALUDE_jeremy_pill_count_l2140_214055

/-- Calculates the total number of pills taken over a period of time given dosage information --/
def total_pills (dose_mg : ℕ) (dose_interval_hours : ℕ) (pill_mg : ℕ) (duration_weeks : ℕ) : ℕ :=
  let pills_per_dose := dose_mg / pill_mg
  let doses_per_day := 24 / dose_interval_hours
  let pills_per_day := pills_per_dose * doses_per_day
  let days := duration_weeks * 7
  pills_per_day * days

/-- Proves that Jeremy takes 112 pills in total during his 2-week treatment --/
theorem jeremy_pill_count : total_pills 1000 6 500 2 = 112 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_pill_count_l2140_214055


namespace NUMINAMATH_CALUDE_ryan_age_problem_l2140_214056

/-- Ryan's age problem -/
theorem ryan_age_problem : ∃ x : ℕ, 
  (∃ n : ℕ, x - 2 = n^3) ∧ 
  (∃ m : ℕ, x + 3 = m^2) ∧ 
  x = 2195 :=
sorry

end NUMINAMATH_CALUDE_ryan_age_problem_l2140_214056


namespace NUMINAMATH_CALUDE_coincide_points_l2140_214051

/-- A point on the coordinate plane with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- A vector between two integer points -/
def vector (a b : IntPoint) : IntPoint :=
  ⟨b.x - a.x, b.y - a.y⟩

/-- Move a point by a vector -/
def movePoint (p v : IntPoint) : IntPoint :=
  ⟨p.x + v.x, p.y + v.y⟩

/-- The main theorem stating that any two points can be made to coincide -/
theorem coincide_points (a b c d : IntPoint) :
  ∃ (moves : List (IntPoint → IntPoint)),
    ∃ (p q : IntPoint),
      (p ∈ [a, b, c, d]) ∧
      (q ∈ [a, b, c, d]) ∧
      (p ≠ q) ∧
      (moves.foldl (λ acc f => f acc) p = moves.foldl (λ acc f => f acc) q) :=
by sorry

end NUMINAMATH_CALUDE_coincide_points_l2140_214051


namespace NUMINAMATH_CALUDE_quadratic_one_root_l2140_214058

/-- Given a quadratic equation x^2 + (6+4m)x + (9-m) = 0 where m is a real number,
    prove that it has exactly one real root if and only if m = 0 and m ≥ 0 -/
theorem quadratic_one_root (m : ℝ) : 
  (∃! x : ℝ, x^2 + (6+4*m)*x + (9-m) = 0) ↔ (m = 0 ∧ m ≥ 0) := by
  sorry

#check quadratic_one_root

end NUMINAMATH_CALUDE_quadratic_one_root_l2140_214058


namespace NUMINAMATH_CALUDE_car_distance_l2140_214013

theorem car_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (time : ℝ) :
  train_speed = 100 →
  car_speed_ratio = 5 / 8 →
  time = 45 / 60 →
  car_speed_ratio * train_speed * time = 46.875 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_l2140_214013


namespace NUMINAMATH_CALUDE_cos_2x_value_l2140_214068

theorem cos_2x_value (x : Real) (h : 2 * Real.sin (Real.pi - x) + 1 = 0) : 
  Real.cos (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_value_l2140_214068


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_four_exists_unique_greatest_solution_is_148_l2140_214010

theorem greatest_integer_with_gcf_four (n : ℕ) : n < 150 ∧ Nat.gcd n 24 = 4 → n ≤ 148 := by
  sorry

theorem exists_unique_greatest : ∃! n : ℕ, n < 150 ∧ Nat.gcd n 24 = 4 ∧ ∀ m : ℕ, m < 150 ∧ Nat.gcd m 24 = 4 → m ≤ n := by
  sorry

theorem solution_is_148 : ∃! n : ℕ, n < 150 ∧ Nat.gcd n 24 = 4 ∧ ∀ m : ℕ, m < 150 ∧ Nat.gcd m 24 = 4 → m ≤ n ∧ n = 148 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_four_exists_unique_greatest_solution_is_148_l2140_214010


namespace NUMINAMATH_CALUDE_turnover_growth_equation_l2140_214064

/-- Represents the turnover growth of a supermarket from January to March -/
structure SupermarketGrowth where
  january_turnover : ℝ
  march_turnover : ℝ
  monthly_growth_rate : ℝ

/-- The equation correctly represents the relationship between turnovers and growth rate -/
theorem turnover_growth_equation (sg : SupermarketGrowth) 
  (h1 : sg.january_turnover = 36)
  (h2 : sg.march_turnover = 48) :
  sg.january_turnover * (1 + sg.monthly_growth_rate)^2 = sg.march_turnover :=
sorry

end NUMINAMATH_CALUDE_turnover_growth_equation_l2140_214064


namespace NUMINAMATH_CALUDE_eliot_votes_l2140_214047

/-- Given the vote distribution in a school election, prove that Eliot got 160 votes. -/
theorem eliot_votes (randy_votes shaun_votes eliot_votes : ℕ) : 
  randy_votes = 16 → 
  shaun_votes = 5 * randy_votes → 
  eliot_votes = 2 * shaun_votes → 
  eliot_votes = 160 := by
sorry


end NUMINAMATH_CALUDE_eliot_votes_l2140_214047


namespace NUMINAMATH_CALUDE_quadratic_with_prime_roots_l2140_214099

theorem quadratic_with_prime_roots (m : ℕ) : 
  (∃ x y : ℕ, x.Prime ∧ y.Prime ∧ x ≠ y ∧ x^2 - 1999*x + m = 0 ∧ y^2 - 1999*y + m = 0) → 
  m = 3994 := by
sorry

end NUMINAMATH_CALUDE_quadratic_with_prime_roots_l2140_214099


namespace NUMINAMATH_CALUDE_onion_weight_problem_l2140_214014

theorem onion_weight_problem (total_weight : Real) (avg_weight_35 : Real) :
  total_weight = 7.68 ∧ avg_weight_35 = 0.190 →
  (total_weight * 1000 - 35 * avg_weight_35 * 1000) / 5 = 206 := by
  sorry

end NUMINAMATH_CALUDE_onion_weight_problem_l2140_214014


namespace NUMINAMATH_CALUDE_mouse_cost_l2140_214044

theorem mouse_cost (mouse_cost keyboard_cost total_cost : ℝ) : 
  keyboard_cost = 3 * mouse_cost →
  total_cost = mouse_cost + keyboard_cost →
  total_cost = 64 →
  mouse_cost = 16 := by
sorry

end NUMINAMATH_CALUDE_mouse_cost_l2140_214044


namespace NUMINAMATH_CALUDE_range_of_a_l2140_214026

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2140_214026


namespace NUMINAMATH_CALUDE_line_slope_is_two_l2140_214012

-- Define the polar equation of the line
def polar_equation (ρ θ : ℝ) : Prop := ρ * Real.sin θ - 2 * ρ * Real.cos θ + 3 = 0

-- Theorem: The slope of the line defined by the polar equation is 2
theorem line_slope_is_two :
  ∃ (m : ℝ), m = 2 ∧
  ∀ (x y : ℝ), (∃ (ρ θ : ℝ), polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  y = m * x - 3 :=
sorry

end NUMINAMATH_CALUDE_line_slope_is_two_l2140_214012


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_including_13_l2140_214032

theorem unique_number_with_three_prime_divisors_including_13 :
  ∀ x n : ℕ,
  x = 9^n - 1 →
  (∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
  13 ∣ x →
  x = 728 := by sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_including_13_l2140_214032


namespace NUMINAMATH_CALUDE_gwen_homework_problems_l2140_214088

/-- The number of math problems Gwen had -/
def math_problems : ℕ := 18

/-- The number of science problems Gwen had -/
def science_problems : ℕ := 11

/-- The number of problems Gwen finished at school -/
def finished_at_school : ℕ := 24

/-- The number of problems Gwen had to do for homework -/
def homework_problems : ℕ := math_problems + science_problems - finished_at_school

theorem gwen_homework_problems :
  homework_problems = 5 := by sorry

end NUMINAMATH_CALUDE_gwen_homework_problems_l2140_214088


namespace NUMINAMATH_CALUDE_derek_dogs_now_l2140_214024

-- Define the number of dogs Derek had at age 7
def dogs_at_7 : ℕ := 120

-- Define the number of cars Derek had at age 7
def cars_at_7 : ℕ := dogs_at_7 / 4

-- Define the number of cars Derek bought
def cars_bought : ℕ := 350

-- Define the total number of cars Derek has now
def cars_now : ℕ := cars_at_7 + cars_bought

-- Define the number of dogs Derek has now
def dogs_now : ℕ := cars_now / 3

-- Theorem to prove
theorem derek_dogs_now : dogs_now = 126 := by
  sorry

end NUMINAMATH_CALUDE_derek_dogs_now_l2140_214024


namespace NUMINAMATH_CALUDE_ellipse_max_value_l2140_214070

theorem ellipse_max_value (x y : ℝ) :
  (x^2 / 6 + y^2 / 4 = 1) →
  (∃ (max : ℝ), ∀ (a b : ℝ), a^2 / 6 + b^2 / 4 = 1 → x + 2*y ≤ max ∧ max = Real.sqrt 22) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_max_value_l2140_214070


namespace NUMINAMATH_CALUDE_jack_initial_marbles_l2140_214079

/-- The number of marbles Jack shared with Rebecca -/
def shared_marbles : ℕ := 33

/-- The number of marbles Jack had after sharing -/
def remaining_marbles : ℕ := 29

/-- The initial number of marbles Jack had -/
def initial_marbles : ℕ := shared_marbles + remaining_marbles

theorem jack_initial_marbles : initial_marbles = 62 := by
  sorry

end NUMINAMATH_CALUDE_jack_initial_marbles_l2140_214079


namespace NUMINAMATH_CALUDE_smallest_n_for_radio_profit_l2140_214023

theorem smallest_n_for_radio_profit (n d : ℕ) (h1 : d > 0) : 
  (∃ (m : ℕ), m ≥ n ∧ 
    d - (3 * d) / (2 * m) + 10 * m - 30 = d + 100 ∧
    (∀ k : ℕ, k < m → d - (3 * d) / (2 * k) + 10 * k - 30 ≠ d + 100)) →
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_for_radio_profit_l2140_214023


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l2140_214038

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l2140_214038


namespace NUMINAMATH_CALUDE_sachin_gain_is_487_50_l2140_214040

/-- Calculates Sachin's gain in one year based on given borrowing and lending conditions. -/
def sachinsGain (X R1 R2 R3 : ℚ) : ℚ :=
  let interestFromRahul := X * R2 / 100
  let interestFromRavi := X * R3 / 100
  let interestPaid := X * R1 / 100
  interestFromRahul + interestFromRavi - interestPaid

/-- Theorem stating that Sachin's gain in one year is 487.50 rupees. -/
theorem sachin_gain_is_487_50 :
  sachinsGain 5000 4 (25/4) (15/2) = 487.5 := by
  sorry

#eval sachinsGain 5000 4 (25/4) (15/2)

end NUMINAMATH_CALUDE_sachin_gain_is_487_50_l2140_214040


namespace NUMINAMATH_CALUDE_total_raisins_added_l2140_214061

theorem total_raisins_added (yellow_raisins : ℝ) (black_raisins : ℝ)
  (h1 : yellow_raisins = 0.3)
  (h2 : black_raisins = 0.4) :
  yellow_raisins + black_raisins = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_total_raisins_added_l2140_214061


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l2140_214073

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 4
  f 1 = 5 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l2140_214073


namespace NUMINAMATH_CALUDE_sevens_to_hundred_l2140_214098

theorem sevens_to_hundred : ∃ (expr : ℕ), 
  (expr = 100) ∧ 
  (∃ (a b c d e f g h i : ℕ), 
    (a ≤ 7 ∧ b ≤ 7 ∧ c ≤ 7 ∧ d ≤ 7 ∧ e ≤ 7 ∧ f ≤ 7 ∧ g ≤ 7 ∧ h ≤ 7 ∧ i ≤ 7) ∧
    (expr = a * b - c * d + e * f + g + h + i) ∧
    (a + b + c + d + e + f + g + h + i < 10 * 7)) :=
by sorry

end NUMINAMATH_CALUDE_sevens_to_hundred_l2140_214098


namespace NUMINAMATH_CALUDE_chord_line_equation_l2140_214000

/-- The equation of a line containing a chord of an ellipse --/
theorem chord_line_equation (x y : ℝ) :
  let ellipse := fun (x y : ℝ) ↦ x^2 / 16 + y^2 / 9 = 1
  let midpoint := (2, (3 : ℝ) / 2)
  let chord_line := fun (x y : ℝ) ↦ 3 * x + 4 * y - 12 = 0
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧
    ellipse x₂ y₂ ∧
    ((x₁ + x₂) / 2, (y₁ + y₂) / 2) = midpoint ∧
    (∀ x y, chord_line x y ↔ ∃ t, x = (1 - t) * x₁ + t * x₂ ∧ y = (1 - t) * y₁ + t * y₂) :=
by
  sorry

end NUMINAMATH_CALUDE_chord_line_equation_l2140_214000


namespace NUMINAMATH_CALUDE_a_card_is_one_three_l2140_214016

structure Card where
  n1 : Nat
  n2 : Nat
  deriving Repr

structure Person where
  name : String
  card : Card
  deriving Repr

def validCards : List Card := [⟨1, 2⟩, ⟨1, 3⟩, ⟨2, 3⟩]

def commonNumber (c1 c2 : Card) : Nat :=
  if c1.n1 = c2.n1 ∨ c1.n1 = c2.n2 then c1.n1
  else if c1.n2 = c2.n1 ∨ c1.n2 = c2.n2 then c1.n2
  else 0

theorem a_card_is_one_three 
  (a b c : Person)
  (h1 : a.card ∈ validCards ∧ b.card ∈ validCards ∧ c.card ∈ validCards)
  (h2 : a.card ≠ b.card ∧ b.card ≠ c.card ∧ a.card ≠ c.card)
  (h3 : commonNumber a.card b.card ≠ 2)
  (h4 : commonNumber b.card c.card ≠ 1)
  (h5 : c.card.n1 + c.card.n2 ≠ 5) :
  a.card = ⟨1, 3⟩ := by
sorry

end NUMINAMATH_CALUDE_a_card_is_one_three_l2140_214016


namespace NUMINAMATH_CALUDE_smallest_possible_n_l2140_214007

theorem smallest_possible_n (a b c n : ℕ) : 
  a < b → b < c → c < n → 
  a + b + c + n = 100 → 
  (∀ m : ℕ, m < n → ¬∃ x y z : ℕ, x < y ∧ y < z ∧ z < m ∧ x + y + z + m = 100) →
  n = 27 := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_n_l2140_214007


namespace NUMINAMATH_CALUDE_special_circle_properties_l2140_214030

/-- A circle passing through two points with its center on a given line -/
structure SpecialCircle where
  -- The circle passes through these two points
  pointA : ℝ × ℝ := (1, 4)
  pointB : ℝ × ℝ := (3, 2)
  -- The center lies on this line
  centerLine : ℝ → ℝ := fun x => 3 - x

/-- The equation of the circle -/
def circleEquation (c : SpecialCircle) (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 4

/-- A point on the circle -/
def pointOnCircle (c : SpecialCircle) (p : ℝ × ℝ) : Prop :=
  circleEquation c p.1 p.2

theorem special_circle_properties (c : SpecialCircle) :
  -- The circle equation is correct
  (∀ x y, circleEquation c x y ↔ pointOnCircle c (x, y)) ∧
  -- The maximum value of x+y for points on the circle
  (∃ max : ℝ, max = 3 + 2 * Real.sqrt 2 ∧
    ∀ p, pointOnCircle c p → p.1 + p.2 ≤ max) := by
  sorry

end NUMINAMATH_CALUDE_special_circle_properties_l2140_214030


namespace NUMINAMATH_CALUDE_difference_of_squares_1027_l2140_214004

theorem difference_of_squares_1027 : (1027 : ℤ) * 1027 - 1026 * 1028 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_1027_l2140_214004


namespace NUMINAMATH_CALUDE_jeong_hyeok_is_nine_l2140_214048

/-- Jeong-hyeok's age -/
def jeong_hyeok_age : ℕ := sorry

/-- Jeong-hyeok's uncle's age -/
def uncle_age : ℕ := sorry

/-- Condition 1: Jeong-hyeok's age is 1 year less than 1/4 of his uncle's age -/
axiom condition1 : jeong_hyeok_age = uncle_age / 4 - 1

/-- Condition 2: His uncle's age is 5 years less than 5 times Jeong-hyeok's age -/
axiom condition2 : uncle_age = 5 * jeong_hyeok_age - 5

/-- Theorem: Jeong-hyeok is 9 years old -/
theorem jeong_hyeok_is_nine : jeong_hyeok_age = 9 := by sorry

end NUMINAMATH_CALUDE_jeong_hyeok_is_nine_l2140_214048


namespace NUMINAMATH_CALUDE_power_division_rule_l2140_214078

theorem power_division_rule (a : ℝ) : a^8 / a^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l2140_214078


namespace NUMINAMATH_CALUDE_sum_of_specific_T_l2140_214090

def T (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    (3 * n) / 2
  else
    (3 * n - 1) / 2

theorem sum_of_specific_T : T 18 + T 34 + T 51 = 154 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_T_l2140_214090


namespace NUMINAMATH_CALUDE_tent_max_profit_l2140_214092

/-- Represents the purchase and sales information for tents --/
structure TentInfo where
  regular_purchase_price : ℝ
  sunshade_purchase_price : ℝ
  regular_selling_price : ℝ
  sunshade_selling_price : ℝ
  total_budget : ℝ

/-- Represents the constraints on tent purchases --/
structure TentConstraints where
  min_regular_tents : ℕ
  regular_not_exceeding_sunshade : Bool

/-- Calculates the maximum profit given tent information and constraints --/
def max_profit (info : TentInfo) (constraints : TentConstraints) : ℝ :=
  sorry

/-- Theorem stating the maximum profit for the given scenario --/
theorem tent_max_profit :
  let info : TentInfo := {
    regular_purchase_price := 150,
    sunshade_purchase_price := 300,
    regular_selling_price := 180,
    sunshade_selling_price := 380,
    total_budget := 9000
  }
  let constraints : TentConstraints := {
    min_regular_tents := 12,
    regular_not_exceeding_sunshade := true
  }
  max_profit info constraints = 2280 := by sorry

end NUMINAMATH_CALUDE_tent_max_profit_l2140_214092


namespace NUMINAMATH_CALUDE_prize_guesses_count_l2140_214034

def digit_partitions : List (Nat × Nat × Nat) :=
  [(1,1,6), (1,2,5), (1,3,4), (1,4,3), (1,5,2), (1,6,1),
   (2,1,5), (2,2,4), (2,3,3), (2,4,2), (2,5,1),
   (3,1,4), (3,2,3), (3,3,2), (3,4,1),
   (4,1,3), (4,2,2), (4,3,1)]

def digit_arrangements : Nat := 70

theorem prize_guesses_count : 
  (List.length digit_partitions) * digit_arrangements = 1260 := by
  sorry

end NUMINAMATH_CALUDE_prize_guesses_count_l2140_214034


namespace NUMINAMATH_CALUDE_functional_inequality_implies_zero_function_l2140_214041

theorem functional_inequality_implies_zero_function 
  (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x * y) ≤ y * f x + f y) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_functional_inequality_implies_zero_function_l2140_214041


namespace NUMINAMATH_CALUDE_parabola_properties_l2140_214029

/-- A parabola with the given properties -/
def Parabola : Set (ℝ × ℝ) :=
  {(x, y) | y^2 = 8*x}

theorem parabola_properties :
  -- The parabola is symmetric about the x-axis
  (∀ x y, (x, y) ∈ Parabola ↔ (x, -y) ∈ Parabola) ∧
  -- The vertex of the parabola is at the origin
  (0, 0) ∈ Parabola ∧
  -- The parabola passes through point (2, 4)
  (2, 4) ∈ Parabola :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2140_214029


namespace NUMINAMATH_CALUDE_sales_difference_is_25_l2140_214086

-- Define the prices and quantities for each company
def company_a_price : ℝ := 4
def company_b_price : ℝ := 3.5
def company_a_quantity : ℕ := 300
def company_b_quantity : ℕ := 350

-- Define the sales difference function
def sales_difference : ℝ :=
  (company_b_price * company_b_quantity) - (company_a_price * company_a_quantity)

-- Theorem statement
theorem sales_difference_is_25 : sales_difference = 25 := by
  sorry

end NUMINAMATH_CALUDE_sales_difference_is_25_l2140_214086


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_negative_sixteen_l2140_214074

theorem sqrt_difference_equals_negative_sixteen :
  Real.sqrt (16 - 8 * Real.sqrt 2) - Real.sqrt (16 + 8 * Real.sqrt 2) = -16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_negative_sixteen_l2140_214074


namespace NUMINAMATH_CALUDE_inequality_proof_l2140_214022

theorem inequality_proof (x : ℝ) : x > 0 ∧ |4*x - 5| < 8 → 0 < x ∧ x < 13/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2140_214022


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l2140_214060

theorem complex_modulus_sqrt_two (x y : ℝ) (h : (1 + Complex.I) * x = 1 + y * Complex.I) :
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l2140_214060


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2140_214081

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 4th term is 23 and the 6th term is 47,
    the 8th term is 71. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_4th : a 4 = 23) 
    (h_6th : a 6 = 47) : 
  a 8 = 71 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2140_214081


namespace NUMINAMATH_CALUDE_estimate_larger_than_original_l2140_214084

theorem estimate_larger_than_original (x y ε : ℝ) 
  (h1 : x > y) (h2 : y > 0) (h3 : ε > 0) : 
  (x + ε) - (y - ε) > x - y := by
  sorry

end NUMINAMATH_CALUDE_estimate_larger_than_original_l2140_214084


namespace NUMINAMATH_CALUDE_vincent_songs_l2140_214019

/-- The number of songs Vincent knows after summer camp -/
def total_songs (initial_songs : ℕ) (new_songs : ℕ) : ℕ :=
  initial_songs + new_songs

/-- Theorem stating that Vincent knows 74 songs after summer camp -/
theorem vincent_songs : total_songs 56 18 = 74 := by
  sorry

end NUMINAMATH_CALUDE_vincent_songs_l2140_214019


namespace NUMINAMATH_CALUDE_second_year_interest_rate_l2140_214008

/-- Calculates the interest rate for the second year given the initial amount,
    first-year interest rate, and final amount after two years. -/
theorem second_year_interest_rate
  (initial_amount : ℝ)
  (first_year_rate : ℝ)
  (final_amount : ℝ)
  (h1 : initial_amount = 9000)
  (h2 : first_year_rate = 0.04)
  (h3 : final_amount = 9828) :
  ∃ (second_year_rate : ℝ),
    second_year_rate = 0.05 ∧
    final_amount = initial_amount * (1 + first_year_rate) * (1 + second_year_rate) :=
by sorry


end NUMINAMATH_CALUDE_second_year_interest_rate_l2140_214008


namespace NUMINAMATH_CALUDE_partner_investment_time_l2140_214059

/-- Given two partners p and q with investment ratio 7:5 and profit ratio 7:10,
    where q invested for 40 months, prove that p invested for 28 months. -/
theorem partner_investment_time
  (investment_ratio : ℚ) -- Ratio of p's investment to q's investment
  (profit_ratio : ℚ) -- Ratio of p's profit to q's profit
  (q_time : ℕ) -- Time q invested in months
  (h1 : investment_ratio = 7 / 5)
  (h2 : profit_ratio = 7 / 10)
  (h3 : q_time = 40) :
  ∃ (p_time : ℕ), p_time = 28 := by
  sorry

end NUMINAMATH_CALUDE_partner_investment_time_l2140_214059


namespace NUMINAMATH_CALUDE_triangle_problem_l2140_214066

/-- Given a triangle ABC with area 3√15, b - c = 2, and cos A = -1/4, prove the following: -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (1/2 * b * c * Real.sqrt (1 - (-1/4)^2) = 3 * Real.sqrt 15) →
  (b - c = 2) →
  (Real.cos A = -1/4) →
  (a^2 = b^2 + c^2 - 2*b*c*(-1/4)) →
  (a / Real.sqrt (1 - (-1/4)^2) = c / Real.sin C) →
  (a = 8 ∧ 
   Real.sin C = Real.sqrt 15 / 8 ∧ 
   Real.cos (2*A + π/6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2140_214066


namespace NUMINAMATH_CALUDE_smallest_base_for_90_l2140_214021

theorem smallest_base_for_90 : 
  ∃ (b : ℕ), b = 5 ∧ 
  (∀ (x : ℕ), x < b → ¬(∃ (d₁ d₂ d₃ : ℕ), d₁ < x ∧ d₂ < x ∧ d₃ < x ∧ 
    90 = d₁ * x^2 + d₂ * x + d₃)) ∧
  (∃ (d₁ d₂ d₃ : ℕ), d₁ < b ∧ d₂ < b ∧ d₃ < b ∧ 
    90 = d₁ * b^2 + d₂ * b + d₃) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_90_l2140_214021


namespace NUMINAMATH_CALUDE_andrews_friends_pizza_slices_l2140_214072

/-- The total number of pizza slices brought by Andrew's friends -/
def total_pizza_slices (num_friends : ℕ) (slices_per_friend : ℕ) : ℕ :=
  num_friends * slices_per_friend

/-- Theorem stating that the total number of pizza slices is 16 -/
theorem andrews_friends_pizza_slices :
  total_pizza_slices 4 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_andrews_friends_pizza_slices_l2140_214072


namespace NUMINAMATH_CALUDE_cube_side_ratio_l2140_214054

theorem cube_side_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (6 * a^2) / (6 * b^2) = 25 → a / b = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l2140_214054


namespace NUMINAMATH_CALUDE_largest_number_problem_l2140_214035

theorem largest_number_problem (a b c : ℝ) : 
  a < b → b < c → 
  a + b + c = 67 → 
  c - b = 7 → 
  b - a = 3 → 
  c = 28 := by
sorry

end NUMINAMATH_CALUDE_largest_number_problem_l2140_214035


namespace NUMINAMATH_CALUDE_notebook_cost_l2140_214095

/-- Proves that the cost of each notebook before discount is $1.48 -/
theorem notebook_cost 
  (total_spent : ℚ)
  (num_backpacks : ℕ)
  (num_pen_packs : ℕ)
  (num_pencil_packs : ℕ)
  (num_notebooks : ℕ)
  (num_calculators : ℕ)
  (discount_rate : ℚ)
  (backpack_price : ℚ)
  (pen_pack_price : ℚ)
  (pencil_pack_price : ℚ)
  (calculator_price : ℚ)
  (h1 : total_spent = 56)
  (h2 : num_backpacks = 1)
  (h3 : num_pen_packs = 3)
  (h4 : num_pencil_packs = 2)
  (h5 : num_notebooks = 5)
  (h6 : num_calculators = 1)
  (h7 : discount_rate = 1/10)
  (h8 : backpack_price = 30)
  (h9 : pen_pack_price = 2)
  (h10 : pencil_pack_price = 3/2)
  (h11 : calculator_price = 15) :
  let other_items_cost := backpack_price * num_backpacks + 
                          pen_pack_price * num_pen_packs + 
                          pencil_pack_price * num_pencil_packs + 
                          calculator_price * num_calculators
  let discounted_other_items_cost := other_items_cost * (1 - discount_rate)
  let notebooks_cost := total_spent - discounted_other_items_cost
  notebooks_cost / num_notebooks = 37/25 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l2140_214095
