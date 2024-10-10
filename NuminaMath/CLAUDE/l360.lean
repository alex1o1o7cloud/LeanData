import Mathlib

namespace complex_equation_solution_l360_36022

theorem complex_equation_solution (Z : ℂ) : (3 + Z) * Complex.I = 1 → Z = -3 - Complex.I := by
  sorry

end complex_equation_solution_l360_36022


namespace existence_of_h_l360_36057

theorem existence_of_h : ∃ h : ℝ, ∀ n : ℕ, 
  ¬(⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋) := by sorry

end existence_of_h_l360_36057


namespace new_alcohol_concentration_l360_36034

/-- Calculates the new alcohol concentration after adding water to an alcohol solution -/
theorem new_alcohol_concentration
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 3)
  (h2 : initial_concentration = 0.33)
  (h3 : added_water = 1)
  : (initial_volume * initial_concentration) / (initial_volume + added_water) = 0.2475 := by
  sorry

end new_alcohol_concentration_l360_36034


namespace odd_function_m_value_l360_36055

/-- Given a > 0 and a ≠ 1, if f(x) = 1/(a^x + 1) - m is an odd function, then m = 1/2 -/
theorem odd_function_m_value (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => 1 / (a^x + 1) - m
  (∀ x, f x + f (-x) = 0) → m = 1/2 := by
  sorry

end odd_function_m_value_l360_36055


namespace intersection_of_A_and_B_l360_36062

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l360_36062


namespace evaluate_expression_l360_36095

theorem evaluate_expression : 
  Real.sqrt 8 * 2^(3/2) + 18 / 3 * 3 - 6^(5/2) = 26 - 36 * Real.sqrt 6 := by sorry

end evaluate_expression_l360_36095


namespace no_fractional_solution_l360_36082

theorem no_fractional_solution (x y : ℝ) : 
  (∃ m n : ℤ, 13 * x + 4 * y = m ∧ 10 * x + 3 * y = n) → 
  (∃ a b : ℤ, x = a ∧ y = b) :=
by sorry

end no_fractional_solution_l360_36082


namespace field_trip_cost_calculation_l360_36026

def field_trip_cost (num_students : ℕ) (num_teachers : ℕ) (student_ticket_price : ℚ) 
  (adult_ticket_price : ℚ) (discount_rate : ℚ) (min_tickets_for_discount : ℕ) 
  (transportation_cost : ℚ) (meal_cost_per_person : ℚ) : ℚ :=
  sorry

theorem field_trip_cost_calculation : 
  field_trip_cost 25 6 1 3 0.2 20 100 7.5 = 366.9 := by
  sorry

end field_trip_cost_calculation_l360_36026


namespace solve_for_y_l360_36044

theorem solve_for_y (x y : ℤ) (h1 : x^2 = y - 2) (h2 : x = -6) : y = 38 := by
  sorry

end solve_for_y_l360_36044


namespace expand_polynomial_l360_36088

theorem expand_polynomial (a b : ℝ) : (a - b) * (a + b) * (a^2 - b^2) = a^4 - 2*a^2*b^2 + b^4 := by
  sorry

end expand_polynomial_l360_36088


namespace grandfather_pension_increase_l360_36023

/-- Represents the percentage increase in family income when a member's income is doubled -/
structure IncomeIncrease where
  masha : ℝ
  mother : ℝ
  father : ℝ

/-- Calculates the percentage increase in family income when grandfather's pension is doubled -/
def grandfather_increase (i : IncomeIncrease) : ℝ :=
  100 - (i.masha + i.mother + i.father)

/-- Theorem stating that given the specified income increases for Masha, mother, and father,
    doubling grandfather's pension will increase the family income by 55% -/
theorem grandfather_pension_increase (i : IncomeIncrease) 
  (h1 : i.masha = 5)
  (h2 : i.mother = 15)
  (h3 : i.father = 25) :
  grandfather_increase i = 55 := by
  sorry

#eval grandfather_increase { masha := 5, mother := 15, father := 25 }

end grandfather_pension_increase_l360_36023


namespace equation_solution_l360_36018

theorem equation_solution : ∃! x : ℤ, 45 - (28 - (x - (15 - 19))) = 58 ∧ x = 37 := by
  sorry

end equation_solution_l360_36018


namespace inequality_system_solutions_l360_36006

theorem inequality_system_solutions :
  let S := {x : ℤ | x ≥ 0 ∧ x - 3 * (x - 1) ≥ 1 ∧ (1 + 3 * x) / 2 > x - 1}
  S = {0, 1} := by sorry

end inequality_system_solutions_l360_36006


namespace era_burger_division_l360_36086

/-- The number of slices each of the third and fourth friends receive when Era divides her burgers. -/
def slices_per_friend (total_burgers : ℕ) (first_friend_slices second_friend_slices era_slices : ℕ) : ℕ :=
  let total_slices := total_burgers * 2
  let remaining_slices := total_slices - (first_friend_slices + second_friend_slices + era_slices)
  remaining_slices / 2

/-- Theorem stating that under the given conditions, each of the third and fourth friends receives 3 slices. -/
theorem era_burger_division :
  slices_per_friend 5 1 2 1 = 3 := by
  sorry

end era_burger_division_l360_36086


namespace tangent_lines_count_l360_36069

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Counts the number of lines tangent to two circles -/
def count_tangent_lines (c1 c2 : Circle) : ℕ :=
  sorry

theorem tangent_lines_count 
  (A B : ℝ × ℝ)
  (C_A : Circle)
  (C_B : Circle)
  (h_distance : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 7)
  (h_C_A_center : C_A.center = A)
  (h_C_B_center : C_B.center = B)
  (h_C_A_radius : C_A.radius = 3)
  (h_C_B_radius : C_B.radius = 4) :
  count_tangent_lines C_A C_B = 3 :=
sorry

end tangent_lines_count_l360_36069


namespace base_seven_digits_of_1234_l360_36024

theorem base_seven_digits_of_1234 : ∃ n : ℕ, (7^n ≤ 1234 ∧ ∀ m : ℕ, 7^m ≤ 1234 → m ≤ n) ∧ n + 1 = 4 := by
  sorry

end base_seven_digits_of_1234_l360_36024


namespace whale_population_prediction_l360_36090

theorem whale_population_prediction (whales_last_year whales_this_year whales_next_year predicted_increase : ℕ) : 
  whales_last_year = 4000 →
  whales_this_year = 2 * whales_last_year →
  predicted_increase = 800 →
  whales_next_year = whales_this_year + predicted_increase →
  whales_next_year = 8800 := by
sorry

end whale_population_prediction_l360_36090


namespace triangle_property_l360_36092

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h : 2 * Real.sin t.B * Real.sin t.C * Real.cos t.A = 1 - Real.cos (2 * t.A)) :
  (t.b^2 + t.c^2) / t.a^2 = 3 ∧ 
  (∀ (t' : Triangle), Real.sin t'.A ≤ Real.sqrt 5 / 3) := by
  sorry

end triangle_property_l360_36092


namespace product_mod_five_l360_36014

theorem product_mod_five : (2023 * 2024 * 2025 * 2026) % 5 = 0 := by
  sorry

end product_mod_five_l360_36014


namespace unbroken_matches_count_l360_36051

def dozen : ℕ := 12
def boxes_count : ℕ := 5 * dozen
def matches_per_box : ℕ := 20
def broken_matches_per_box : ℕ := 3

theorem unbroken_matches_count :
  boxes_count * (matches_per_box - broken_matches_per_box) = 1020 :=
by sorry

end unbroken_matches_count_l360_36051


namespace order_of_logarithms_and_fraction_l360_36000

theorem order_of_logarithms_and_fraction :
  let a := Real.log 5 / Real.log 8
  let b := Real.log 3 / Real.log 4
  let c := 2 / 3
  c < a ∧ a < b := by sorry

end order_of_logarithms_and_fraction_l360_36000


namespace sum_of_roots_equation_l360_36002

theorem sum_of_roots_equation (x : ℝ) : (x - 1) * (x + 4) = 18 → ∃ y : ℝ, (y - 1) * (y + 4) = 18 ∧ x + y = -3 := by
  sorry

end sum_of_roots_equation_l360_36002


namespace probability_of_sum_25_l360_36050

/-- Represents a die with numbered faces and a blank face -/
structure Die where
  faces : ℕ
  numbers : List ℕ
  blank_faces : ℕ

/-- Calculates the probability of a specific sum when rolling two dice -/
def probability_of_sum (die1 die2 : Die) (target_sum : ℕ) : ℚ :=
  sorry

/-- The first die with 19 faces numbered 1 through 18 and one blank face -/
def first_die : Die :=
  { faces := 20,
    numbers := List.range 18,
    blank_faces := 1 }

/-- The second die with 19 faces numbered 2 through 9 and 11 through 21 and one blank face -/
def second_die : Die :=
  { faces := 20,
    numbers := (List.range 8).map (· + 2) ++ (List.range 11).map (· + 11),
    blank_faces := 1 }

/-- Theorem stating the probability of rolling a sum of 25 with the given dice -/
theorem probability_of_sum_25 :
  probability_of_sum first_die second_die 25 = 3 / 80 := by
  sorry

end probability_of_sum_25_l360_36050


namespace rain_duration_l360_36093

theorem rain_duration (total_hours : ℕ) (no_rain_hours : ℕ) 
  (h1 : total_hours = 8) (h2 : no_rain_hours = 6) : 
  total_hours - no_rain_hours = 2 := by
  sorry

end rain_duration_l360_36093


namespace matrix_multiplication_result_l360_36096

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 2; 2, 1]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-1, 2; 3, -4]
def C : Matrix (Fin 2) (Fin 2) ℤ := !![5, -6; 1, 0]

theorem matrix_multiplication_result : A * B = C := by sorry

end matrix_multiplication_result_l360_36096


namespace direct_proportion_increases_l360_36042

theorem direct_proportion_increases (x₁ x₂ : ℝ) (h : x₁ < x₂) : 2 * x₁ < 2 * x₂ := by
  sorry

end direct_proportion_increases_l360_36042


namespace volleyball_team_selection_count_l360_36074

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 6 starters from a team of 15 players,
    including 4 quadruplets, with at least two quadruplets in the starting lineup -/
def volleyball_team_selection : ℕ :=
  choose 4 2 * choose 11 4 +
  choose 4 3 * choose 11 3 +
  choose 11 2

theorem volleyball_team_selection_count :
  volleyball_team_selection = 2695 := by sorry

end volleyball_team_selection_count_l360_36074


namespace salary_before_raise_l360_36037

theorem salary_before_raise (new_salary : ℝ) (increase_percentage : ℝ) (old_salary : ℝ) :
  new_salary = 70 →
  increase_percentage = 16.666666666666664 →
  old_salary * (1 + increase_percentage / 100) = new_salary →
  old_salary = 60 := by
sorry

end salary_before_raise_l360_36037


namespace equilateral_triangle_side_length_squared_l360_36041

/-- The square of the side length of an equilateral triangle inscribed in a specific circle -/
theorem equilateral_triangle_side_length_squared (x y : ℝ) : 
  x^2 + y^2 = 16 →  -- Circle equation
  (0 : ℝ)^2 + 4^2 = 16 →  -- One vertex at (0, 4)
  (∃ a b : ℝ, a^2 + b^2 = 16 ∧ (0 - a)^2 + (4 - b)^2 = a^2 + (4 - b)^2) →  -- Triangle inscribed in circle
  (∃ c : ℝ, c^2 + (-3)^2 = 16) →  -- Altitude on y-axis (implied by y = -3 for other vertices)
  (0 : ℝ)^2 + 7^2 = 49 :=  -- Square of side length is 49
by sorry

end equilateral_triangle_side_length_squared_l360_36041


namespace adult_ticket_cost_l360_36027

theorem adult_ticket_cost (num_adults : ℕ) (child_ticket_cost : ℚ) (total_receipts : ℚ) :
  num_adults = 152 →
  child_ticket_cost = 5/2 →
  total_receipts = 1026 →
  ∃ A : ℚ, A * num_adults + child_ticket_cost * (num_adults / 2) = total_receipts ∧ A = 11/2 := by
  sorry

end adult_ticket_cost_l360_36027


namespace isosceles_triangle_side_length_l360_36058

structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  isIsosceles : side > 0

def medianDividesDifference (t : IsoscelesTriangle) (diff : ℝ) : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = t.base + 2 * t.side ∧ |x - y| = diff

theorem isosceles_triangle_side_length 
  (t : IsoscelesTriangle) 
  (h_base : t.base = 7) 
  (h_median : medianDividesDifference t 3) : 
  t.side = 4 ∨ t.side = 10 := by
  sorry

end isosceles_triangle_side_length_l360_36058


namespace matrix_power_four_l360_36003

/-- Given a 2x2 matrix A, prove that A^4 equals the given result. -/
theorem matrix_power_four (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A = !![3 * Real.sqrt 2, -3; 3, 3 * Real.sqrt 2] →
  A ^ 4 = !![-81, 0; 0, -81] := by
  sorry

end matrix_power_four_l360_36003


namespace perfect_square_trinomial_l360_36030

/-- For a constant m, x^2 + 2x + m is a perfect square trinomial if and only if m = 1 -/
theorem perfect_square_trinomial (m : ℝ) :
  (∀ x : ℝ, ∃ a : ℝ, x^2 + 2*x + m = (x + a)^2) ↔ m = 1 := by
  sorry

end perfect_square_trinomial_l360_36030


namespace songs_per_album_l360_36060

/-- The number of country albums bought -/
def country_albums : ℕ := 4

/-- The number of pop albums bought -/
def pop_albums : ℕ := 5

/-- The total number of songs bought -/
def total_songs : ℕ := 72

/-- Proves that if all albums have the same number of songs, then each album contains 8 songs -/
theorem songs_per_album :
  ∀ (songs_per_album : ℕ),
  country_albums * songs_per_album + pop_albums * songs_per_album = total_songs →
  songs_per_album = 8 := by
sorry

end songs_per_album_l360_36060


namespace distance_a_beats_b_proof_l360_36017

/-- The distance A can beat B when running 4.5 km -/
def distance_a_beats_b (a_speed : ℝ) (time_diff : ℝ) : ℝ :=
  a_speed * time_diff

/-- Theorem stating that the distance A beats B is equal to A's speed multiplied by the time difference -/
theorem distance_a_beats_b_proof (a_speed : ℝ) (time_diff : ℝ) (a_time : ℝ) (b_time : ℝ) 
    (h1 : a_speed = 4.5 / a_time)
    (h2 : time_diff = b_time - a_time)
    (h3 : a_time = 90)
    (h4 : b_time = 180) :
  distance_a_beats_b a_speed time_diff = 4.5 := by
  sorry

#check distance_a_beats_b_proof

end distance_a_beats_b_proof_l360_36017


namespace melissa_pencils_count_l360_36012

/-- The number of pencils Melissa wants to buy -/
def melissa_pencils : ℕ := 2

/-- The price of one pencil in cents -/
def pencil_price : ℕ := 20

/-- The number of pencils Tolu wants to buy -/
def tolu_pencils : ℕ := 3

/-- The number of pencils Robert wants to buy -/
def robert_pencils : ℕ := 5

/-- The total amount spent by all students in cents -/
def total_spent : ℕ := 200

theorem melissa_pencils_count :
  melissa_pencils * pencil_price + tolu_pencils * pencil_price + robert_pencils * pencil_price = total_spent :=
by sorry

end melissa_pencils_count_l360_36012


namespace ratio_difference_l360_36028

theorem ratio_difference (a b : ℕ) (ha : a > 5) (hb : b > 5) : 
  (a : ℚ) / b = 6 / 5 → (a - 5 : ℚ) / (b - 5) = 5 / 4 → a - b = 5 := by
sorry

end ratio_difference_l360_36028


namespace vector_operation_l360_36087

def a : Fin 2 → ℝ := ![2, 4]
def b : Fin 2 → ℝ := ![-1, 1]

theorem vector_operation : 
  (2 • a - b) = ![5, 7] := by sorry

end vector_operation_l360_36087


namespace triangle_area_l360_36084

def a : ℝ × ℝ := (7, 3)
def b : ℝ × ℝ := (-1, 5)

theorem triangle_area : 
  let det := a.1 * b.2 - a.2 * b.1
  (1/2 : ℝ) * |det| = 19 := by sorry

end triangle_area_l360_36084


namespace arithmetic_evaluation_l360_36029

theorem arithmetic_evaluation : 3 + 2 * (8 - 3) = 13 := by
  sorry

end arithmetic_evaluation_l360_36029


namespace recycling_points_l360_36032

/-- The number of pounds needed to recycle to earn one point -/
def poundsPerPoint (gwenPounds friendsPounds totalPoints : ℕ) : ℚ :=
  (gwenPounds + friendsPounds : ℚ) / totalPoints

theorem recycling_points (gwenPounds friendsPounds totalPoints : ℕ) 
  (h1 : gwenPounds = 5)
  (h2 : friendsPounds = 13)
  (h3 : totalPoints = 6) :
  poundsPerPoint gwenPounds friendsPounds totalPoints = 3 := by
  sorry

end recycling_points_l360_36032


namespace fraction_ordering_l360_36061

theorem fraction_ordering : 
  (20 : ℚ) / 16 < (18 : ℚ) / 14 ∧ (18 : ℚ) / 14 < (16 : ℚ) / 12 := by
  sorry

end fraction_ordering_l360_36061


namespace line_through_points_l360_36038

-- Define the line equation
def line_equation (a b x : ℝ) : ℝ := a * x + b

-- Define the condition that the line passes through two points
def passes_through (a b : ℝ) : Prop :=
  line_equation a b 3 = 4 ∧ line_equation a b 9 = 22

-- Theorem statement
theorem line_through_points :
  ∀ a b : ℝ, passes_through a b → a - b = 8 := by
  sorry

end line_through_points_l360_36038


namespace fraction_sum_equals_one_l360_36047

theorem fraction_sum_equals_one (x y : ℝ) (h : x + y ≠ 0) :
  x / (x + y) + y / (x + y) = 1 := by
  sorry

end fraction_sum_equals_one_l360_36047


namespace smallest_integer_square_triple_plus_100_l360_36036

theorem smallest_integer_square_triple_plus_100 : 
  ∃ (x : ℤ), x^2 = 3*x + 100 ∧ ∀ (y : ℤ), y^2 = 3*y + 100 → x ≤ y :=
by sorry

end smallest_integer_square_triple_plus_100_l360_36036


namespace garden_path_width_l360_36078

/-- Given two concentric circles with a difference in circumference of 20π meters,
    the width of the path between them is 10 meters. -/
theorem garden_path_width (R r : ℝ) (h : 2 * Real.pi * R - 2 * Real.pi * r = 20 * Real.pi) :
  R - r = 10 := by
  sorry

end garden_path_width_l360_36078


namespace cubic_root_of_unity_expression_l360_36011

theorem cubic_root_of_unity_expression : 
  ∀ ω : ℂ, ω ^ 3 = 1 → ω ≠ (1 : ℂ) → (2 - ω + ω^2)^3 + (2 + ω - ω^2)^3 = 28 := by
  sorry

end cubic_root_of_unity_expression_l360_36011


namespace f_at_5_l360_36099

def f (x : ℝ) : ℝ := 2*x^5 - 5*x^4 - 4*x^3 + 3*x^2 - 524

theorem f_at_5 : f 5 = 2176 := by
  sorry

end f_at_5_l360_36099


namespace zach_score_l360_36085

/-- Given that Ben scored 21 points in a football game and Zach scored 21 more points than Ben,
    prove that Zach scored 42 points. -/
theorem zach_score (ben_score : ℕ) (zach_ben_diff : ℕ) 
  (h1 : ben_score = 21)
  (h2 : zach_ben_diff = 21) :
  ben_score + zach_ben_diff = 42 := by
  sorry

end zach_score_l360_36085


namespace rectangle_ratio_theorem_l360_36070

theorem rectangle_ratio_theorem (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a ≤ b) :
  let d := Real.sqrt (a^2 + b^2)
  let k := a / b
  (a / b = (a + 2*b) / d) → (k^4 - 3*k^2 - 4*k - 4 = 0) :=
by sorry

end rectangle_ratio_theorem_l360_36070


namespace token_game_ends_in_37_rounds_l360_36043

/-- Represents a player in the token game -/
inductive Player : Type
| A
| B
| C

/-- The state of the game at any given round -/
structure GameState :=
  (tokens : Player → ℕ)

/-- The initial state of the game -/
def initial_state : GameState :=
  { tokens := fun p => match p with
    | Player.A => 15
    | Player.B => 14
    | Player.C => 13 }

/-- Simulates one round of the game -/
def play_round (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended -/
def game_ended (state : GameState) : Bool :=
  sorry

/-- Counts the number of rounds until the game ends -/
def count_rounds (state : GameState) : ℕ :=
  sorry

theorem token_game_ends_in_37_rounds :
  count_rounds initial_state = 37 :=
sorry

end token_game_ends_in_37_rounds_l360_36043


namespace square_side_length_l360_36059

theorem square_side_length (width height : ℝ) (h1 : width = 3320) (h2 : height = 2025) : ∃ (r s : ℝ),
  2 * r + s = height ∧
  2 * r + 3 * s = width ∧
  s = 647.5 := by
sorry

end square_side_length_l360_36059


namespace expression_simplification_l360_36063

theorem expression_simplification (x : ℝ) :
  2*x*(4*x^2 - 3) - 4*(x^2 - 3*x + 8) = 8*x^3 - 4*x^2 + 6*x - 32 := by
  sorry

end expression_simplification_l360_36063


namespace modulus_of_z_l360_36052

theorem modulus_of_z (z : ℂ) (h : z * (1 + Complex.I) = 1 + 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end modulus_of_z_l360_36052


namespace trajectory_of_equidistant_complex_l360_36094

theorem trajectory_of_equidistant_complex (z : ℂ) :
  Complex.abs (z + 1 - Complex.I) = Complex.abs (z - 1 + Complex.I) →
  z.re = z.im :=
by sorry

end trajectory_of_equidistant_complex_l360_36094


namespace xyz_equals_ten_l360_36089

theorem xyz_equals_ten (a b c x y z : ℂ) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h_a : a = (b + c) / (x - 3))
  (h_b : b = (a + c) / (y - 3))
  (h_c : c = (a + b) / (z - 3))
  (h_sum_prod : x * y + x * z + y * z = 7)
  (h_sum : x + y + z = 4) :
  x * y * z = 10 := by
sorry

end xyz_equals_ten_l360_36089


namespace shirt_discount_problem_l360_36046

theorem shirt_discount_problem (list_price : ℝ) (final_price : ℝ) (second_discount : ℝ) (first_discount : ℝ) : 
  list_price = 150 →
  final_price = 105 →
  second_discount = 12.5 →
  final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) →
  first_discount = 20 := by
sorry

end shirt_discount_problem_l360_36046


namespace expression_evaluation_l360_36098

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  ((x * y + 2) * (x * y - 2) - 2 * x^2 * y^2 + 4) / (x * y) = -6 :=
by sorry

end expression_evaluation_l360_36098


namespace parabolas_intersection_l360_36016

-- Define the two parabola functions
def f (x : ℝ) : ℝ := 2 * x^2 + 5 * x - 3
def g (x : ℝ) : ℝ := x^2 + 2

-- Theorem statement
theorem parabolas_intersection :
  (∃ (x y : ℝ), f x = g x ∧ y = f x) ↔
  (∃ (x y : ℝ), (x = -5 ∧ y = 27) ∨ (x = 1 ∧ y = 3)) :=
by sorry

end parabolas_intersection_l360_36016


namespace negative_1651_mod_9_l360_36010

theorem negative_1651_mod_9 : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1651 ≡ n [ZMOD 9] ∧ n = 5 := by
  sorry

end negative_1651_mod_9_l360_36010


namespace highest_score_percentage_l360_36049

/-- The percentage of correct answers on an exam with a given number of questions -/
def examPercentage (correctAnswers : ℕ) (totalQuestions : ℕ) : ℚ :=
  (correctAnswers : ℚ) / (totalQuestions : ℚ) * 100

theorem highest_score_percentage
  (totalQuestions : ℕ)
  (hannahsTarget : ℕ)
  (otherStudentWrong : ℕ)
  (hTotal : totalQuestions = 40)
  (hHannah : hannahsTarget = 39)
  (hOther : otherStudentWrong = 3)
  : examPercentage (totalQuestions - otherStudentWrong - 1) totalQuestions = 95 := by
  sorry

end highest_score_percentage_l360_36049


namespace min_distance_to_line_l360_36031

/-- The minimum distance from the origin to the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : 
  let line := {p : ℝ × ℝ | p.1 + p.2 = 4}
  ∀ p ∈ line, Real.sqrt ((0 - p.1)^2 + (0 - p.2)^2) ≥ 2 * Real.sqrt 2 ∧
  ∃ q ∈ line, Real.sqrt ((0 - q.1)^2 + (0 - q.2)^2) = 2 * Real.sqrt 2 :=
by sorry

end min_distance_to_line_l360_36031


namespace razorback_tshirt_sales_l360_36020

/-- The number of t-shirts sold by the Razorback t-shirt shop during a game -/
def num_tshirts_sold (original_price discount total_revenue : ℕ) : ℕ :=
  total_revenue / (original_price - discount)

/-- Theorem stating that 130 t-shirts were sold given the problem conditions -/
theorem razorback_tshirt_sales : num_tshirts_sold 51 8 5590 = 130 := by
  sorry

end razorback_tshirt_sales_l360_36020


namespace key_dimension_in_polygon_division_l360_36033

/-- Represents a polygon with a key dimension --/
structure Polygon where
  keyDimension : ℝ

/-- Represents a rectangle --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square --/
structure Square where
  side : ℝ

/-- Function to check if two polygons are congruent --/
def areCongruent (p1 p2 : Polygon) : Prop := sorry

/-- Function to check if polygons can form a square --/
def canFormSquare (p1 p2 : Polygon) (s : Square) : Prop := sorry

/-- Theorem stating the existence of a key dimension x = 4 in the polygons --/
theorem key_dimension_in_polygon_division (r : Rectangle) 
  (h1 : r.width = 12 ∧ r.height = 12) 
  (p1 p2 : Polygon) (s : Square)
  (h2 : areCongruent p1 p2)
  (h3 : canFormSquare p1 p2 s)
  (h4 : s.side^2 = r.width * r.height) :
  ∃ x : ℝ, x = 4 ∧ (p1.keyDimension = x ∨ p2.keyDimension = x) :=
sorry

end key_dimension_in_polygon_division_l360_36033


namespace complex_power_result_l360_36053

theorem complex_power_result : ∃ (i : ℂ), i^2 = -1 ∧ ((1 + i) / i)^2014 = 2^1007 * i := by sorry

end complex_power_result_l360_36053


namespace rachel_age_when_father_is_60_l360_36072

/-- Rachel's age when the problem is stated -/
def rachel_initial_age : ℕ := 12

/-- Rachel's grandfather's age is 7 times Rachel's age -/
def grandfather_age (rachel_age : ℕ) : ℕ := 7 * rachel_age

/-- Rachel's mother's age is half her grandfather's age -/
def mother_age (grandfather_age : ℕ) : ℕ := grandfather_age / 2

/-- Rachel's father's age is 5 years older than her mother -/
def father_age (mother_age : ℕ) : ℕ := mother_age + 5

/-- The age difference between Rachel and her father -/
def age_difference : ℕ := father_age (mother_age (grandfather_age rachel_initial_age)) - rachel_initial_age

theorem rachel_age_when_father_is_60 :
  rachel_initial_age + age_difference = 25 ∧ father_age (mother_age (grandfather_age (rachel_initial_age + age_difference))) = 60 := by
  sorry


end rachel_age_when_father_is_60_l360_36072


namespace min_sheets_for_boats_l360_36008

theorem min_sheets_for_boats (boats_per_sheet : ℕ) (planes_per_sheet : ℕ) (total_toys : ℕ) :
  boats_per_sheet = 8 →
  planes_per_sheet = 6 →
  total_toys = 80 →
  ∃ (sheets : ℕ), 
    sheets * boats_per_sheet = total_toys ∧
    sheets = 10 ∧
    (∀ (s : ℕ), s * boats_per_sheet = total_toys → s ≥ sheets) :=
by sorry

end min_sheets_for_boats_l360_36008


namespace function_property_l360_36013

/-- Piecewise function f(x) as described in the problem -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then a * x + b else 7 - 2 * x

/-- The main theorem to prove -/
theorem function_property (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) → a + b = 4 := by
  sorry

end function_property_l360_36013


namespace composition_central_symmetries_is_translation_translation_then_symmetry_is_symmetry_symmetry_then_translation_is_symmetry_l360_36068

-- Define central symmetry
def central_symmetry (O : ℝ × ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  (2 * O.1 - P.1, 2 * O.2 - P.2)

-- Define parallel translation
def parallel_translation (a : ℝ × ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 + a.1, P.2 + a.2)

-- Theorem 1: Composition of two central symmetries is a parallel translation
theorem composition_central_symmetries_is_translation 
  (O₁ O₂ : ℝ × ℝ) (P : ℝ × ℝ) :
  ∃ a : ℝ × ℝ, central_symmetry O₂ (central_symmetry O₁ P) = parallel_translation a P :=
sorry

-- Theorem 2a: Composition of translation and central symmetry is a central symmetry
theorem translation_then_symmetry_is_symmetry 
  (a : ℝ × ℝ) (O : ℝ × ℝ) (P : ℝ × ℝ) :
  ∃ O' : ℝ × ℝ, central_symmetry O (parallel_translation a P) = central_symmetry O' P :=
sorry

-- Theorem 2b: Composition of central symmetry and translation is a central symmetry
theorem symmetry_then_translation_is_symmetry 
  (O : ℝ × ℝ) (a : ℝ × ℝ) (P : ℝ × ℝ) :
  ∃ O' : ℝ × ℝ, parallel_translation a (central_symmetry O P) = central_symmetry O' P :=
sorry

end composition_central_symmetries_is_translation_translation_then_symmetry_is_symmetry_symmetry_then_translation_is_symmetry_l360_36068


namespace collinear_points_determine_a_l360_36066

/-- Given three points A(1,-1), B(a,3), and C(4,5) that are collinear,
    prove that a = 3. -/
theorem collinear_points_determine_a (a : ℝ) :
  let A : ℝ × ℝ := (1, -1)
  let B : ℝ × ℝ := (a, 3)
  let C : ℝ × ℝ := (4, 5)
  (∃ (t : ℝ), B.1 = A.1 + t * (C.1 - A.1) ∧ B.2 = A.2 + t * (C.2 - A.2)) →
  a = 3 :=
by
  sorry


end collinear_points_determine_a_l360_36066


namespace root_sum_cubes_l360_36021

theorem root_sum_cubes (a b c : ℝ) : 
  (a^3 + 14*a^2 + 49*a + 36 = 0) → 
  (b^3 + 14*b^2 + 49*b + 36 = 0) → 
  (c^3 + 14*c^2 + 49*c + 36 = 0) → 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 686 := by
  sorry

end root_sum_cubes_l360_36021


namespace division_result_l360_36065

theorem division_result : (0.0204 : ℝ) / 17 = 0.0012 := by sorry

end division_result_l360_36065


namespace sandy_has_144_marbles_l360_36039

def dozen : ℕ := 12

def jessica_marbles : ℕ := 3 * dozen

def sandy_marbles : ℕ := 4 * jessica_marbles

theorem sandy_has_144_marbles : sandy_marbles = 144 := by
  sorry

end sandy_has_144_marbles_l360_36039


namespace sum_of_distinct_prime_factors_l360_36009

def expression : ℕ := 7^7 - 7^3

theorem sum_of_distinct_prime_factors : 
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (expression + 1)))
    (λ p => if p ∣ expression then p else 0)) = 17 := by sorry

end sum_of_distinct_prime_factors_l360_36009


namespace cube_angle_range_l360_36048

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Calculates the angle between two vectors -/
def angle (v1 v2 : Point3D) : ℝ := sorry

/-- Theorem: The angle between A₁M and C₁N is in the range (π/3, π/2) -/
theorem cube_angle_range (cube : Cube) (M N : Point3D) 
  (h_M : M.x > cube.A.x ∧ M.x < cube.B.x ∧ M.y = cube.A.y ∧ M.z = cube.A.z)
  (h_N : N.x = cube.B.x ∧ N.y > cube.B.y ∧ N.y < cube.B₁.y ∧ N.z = cube.B.z)
  (h_AM_eq_B₁N : (M.x - cube.A.x)^2 = (cube.B₁.y - N.y)^2) :
  let θ := angle (Point3D.mk (cube.A₁.x - M.x) (cube.A₁.y - M.y) (cube.A₁.z - M.z))
              (Point3D.mk (cube.C₁.x - N.x) (cube.C₁.y - N.y) (cube.C₁.z - N.z))
  π/3 < θ ∧ θ < π/2 := by
  sorry

end cube_angle_range_l360_36048


namespace incorrect_statement_l360_36045

-- Define propositions P and Q
def P : Prop := 2 + 2 = 5
def Q : Prop := 3 > 2

-- Theorem stating that the incorrect statement is "'P and Q' is false, 'not P' is false"
theorem incorrect_statement :
  ¬((P ∧ Q → False) ∧ (¬P → False)) :=
by
  sorry


end incorrect_statement_l360_36045


namespace arithmetic_sequence_properties_l360_36015

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h1 : a 5 = -1)
  (h2 : a 8 = 2)
  (m n : ℕ+)
  (h3 : m ≠ n)
  (h4 : a m = n)
  (h5 : a n = m) :
  (a 1 = -5 ∧ ∃ d : ℤ, d = 1 ∧ ∀ k : ℕ, a (k + 1) = a k + d) ∧
  a (m + n) = 0 := by
sorry

end arithmetic_sequence_properties_l360_36015


namespace parallelogram_network_l360_36007

theorem parallelogram_network (first_set : ℕ) (total_parallelograms : ℕ) (second_set : ℕ) : 
  first_set = 7 → 
  total_parallelograms = 588 → 
  total_parallelograms = (first_set - 1) * (second_set - 1) → 
  second_set = 99 :=
by sorry

end parallelogram_network_l360_36007


namespace taxi_fare_calculation_l360_36080

/-- Taxi fare function -/
def fare (k : ℝ) (d : ℝ) : ℝ := 20 + k * d

/-- Theorem: If the fare for 60 miles is $140, then the fare for 85 miles is $190 -/
theorem taxi_fare_calculation (k : ℝ) :
  fare k 60 = 140 → fare k 85 = 190 := by
  sorry

end taxi_fare_calculation_l360_36080


namespace girl_scouts_expenses_l360_36025

def total_earnings : ℝ := 30

def pool_entry_cost : ℝ :=
  5 * 3.5 + 3 * 2 + 2 * 1

def transportation_cost : ℝ :=
  6 * 1.5 + 4 * 0.75

def snack_cost : ℝ :=
  3 * 3 + 4 * 2.5 + 3 * 2

def total_expenses : ℝ :=
  pool_entry_cost + transportation_cost + snack_cost

theorem girl_scouts_expenses (h : total_expenses > total_earnings) :
  total_expenses - total_earnings = 32.5 := by
  sorry

end girl_scouts_expenses_l360_36025


namespace duck_count_proof_l360_36067

/-- The number of mallard ducks initially at the park -/
def initial_ducks : ℕ := sorry

/-- The number of geese initially at the park -/
def initial_geese : ℕ := 2 * initial_ducks - 10

/-- The number of ducks after the small flock arrives -/
def ducks_after_arrival : ℕ := initial_ducks + 4

/-- The number of geese after some leave -/
def geese_after_leaving : ℕ := initial_geese - 10

theorem duck_count_proof : 
  initial_ducks = 25 ∧ 
  geese_after_leaving = ducks_after_arrival + 1 :=
sorry

end duck_count_proof_l360_36067


namespace vertex_on_x_axis_l360_36071

/-- The vertex of the parabola y = x^2 - 6x + c lies on the x-axis if and only if c = 9 -/
theorem vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + c = 0 ∧ ∀ y : ℝ, y^2 - 6*y + c ≥ x^2 - 6*x + c) ↔ c = 9 := by
  sorry

end vertex_on_x_axis_l360_36071


namespace alcohol_solution_volume_l360_36001

theorem alcohol_solution_volume 
  (V : ℝ) 
  (h1 : 0.30 * V + 2.4 = 0.50 * (V + 2.4)) : 
  V = 6 := by
sorry

end alcohol_solution_volume_l360_36001


namespace mam_mgm_difference_bound_l360_36054

theorem mam_mgm_difference_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a < b) :
  let mam := (a^(1/3) + b^(1/3)) / 2
  let mgm := (a * b)^(1/6)
  mam - mgm < (b - a) / (2 * b) := by
  sorry

end mam_mgm_difference_bound_l360_36054


namespace circle_area_right_triangle_circle_area_right_triangle_value_l360_36005

/-- The area of a circle passing through the vertices of a right triangle with legs of lengths 4 and 3 -/
theorem circle_area_right_triangle (π : ℝ) : ℝ :=
  let a : ℝ := 3  -- Length of one leg
  let b : ℝ := 4  -- Length of the other leg
  let c : ℝ := Real.sqrt (a^2 + b^2)  -- Length of the hypotenuse
  let r : ℝ := c / 2  -- Radius of the circle
  π * r^2

/-- The area of the circle is equal to 25π/4 -/
theorem circle_area_right_triangle_value (π : ℝ) :
  circle_area_right_triangle π = 25 / 4 * π := by
  sorry

end circle_area_right_triangle_circle_area_right_triangle_value_l360_36005


namespace doll_collection_increase_l360_36079

theorem doll_collection_increase (original_count : ℕ) (increase : ℕ) (final_count : ℕ) :
  original_count + increase = final_count →
  final_count = 10 →
  increase = 2 →
  (increase : ℚ) / (original_count : ℚ) * 100 = 25 := by
  sorry

end doll_collection_increase_l360_36079


namespace binomial_coeff_equality_l360_36076

def binomial_coeff (n m : ℕ) : ℕ := Nat.choose n m

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem binomial_coeff_equality (n m : ℕ) :
  binomial_coeff n (m - 1) = binomial_coeff (n - 1) m ↔
  ∃ k : ℕ, n = fibonacci (2 * k) * fibonacci (2 * k + 1) ∧
            m = fibonacci (2 * k) * fibonacci (2 * k - 1) :=
sorry

end binomial_coeff_equality_l360_36076


namespace no_solution_exists_l360_36064

-- Define the system of equations
def system (a b c d : ℝ) : Prop :=
  a^3 + c^3 = 2 ∧
  a^2*b + c^2*d = 0 ∧
  b^3 + d^3 = 1 ∧
  a*b^2 + c*d^2 = -6

-- Theorem stating that no solution exists
theorem no_solution_exists : ¬∃ (a b c d : ℝ), system a b c d := by
  sorry


end no_solution_exists_l360_36064


namespace triangle_inequality_range_l360_36081

/-- The triangle operation on real numbers -/
def triangle (x y : ℝ) : ℝ := x * (2 - y)

/-- Theorem stating the range of m for which (x + m) △ x < 1 holds for all real x -/
theorem triangle_inequality_range (m : ℝ) :
  (∀ x : ℝ, triangle (x + m) x < 1) ↔ m ∈ Set.Ioo (-4 : ℝ) 0 := by sorry

end triangle_inequality_range_l360_36081


namespace candy_distribution_l360_36019

theorem candy_distribution (total_children : ℕ) (absent_children : ℕ) (extra_candies : ℕ) :
  total_children = 300 →
  absent_children = 150 →
  extra_candies = 24 →
  (total_children - absent_children) * (total_children / (total_children - absent_children) + extra_candies) = 
    total_children * (48 : ℕ) :=
by sorry

end candy_distribution_l360_36019


namespace tangent_range_l360_36075

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- The equation for the tangent line passing through (1, m) and touching the curve at x₀ --/
def tangent_equation (x₀ m : ℝ) : Prop :=
  (x₀^3 - 3*x₀ - m) / (x₀ - 1) = 3*x₀^2 - 3

/-- The condition for exactly three tangent lines --/
def three_tangents (m : ℝ) : Prop :=
  ∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    tangent_equation x₁ m ∧ tangent_equation x₂ m ∧ tangent_equation x₃ m

/-- The main theorem --/
theorem tangent_range :
  ∀ m : ℝ, m ≠ -2 → three_tangents m → -3 < m ∧ m < -2 :=
by sorry

end tangent_range_l360_36075


namespace isosceles_right_triangle_area_l360_36040

/-- The area of an isosceles right triangle with hypotenuse 6√2 is 18 -/
theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) : 
  h = 6 * Real.sqrt 2 →  -- hypotenuse is 6√2
  A = (h^2) / 4 →        -- area formula for isosceles right triangle
  A = 18 := by
sorry

end isosceles_right_triangle_area_l360_36040


namespace students_playing_both_sports_l360_36091

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (cricket : ℕ) (neither : ℕ) : 
  total = 420 → football = 325 → cricket = 175 → neither = 50 →
  football + cricket - (total - neither) = 130 := by
sorry

end students_playing_both_sports_l360_36091


namespace smallest_value_l360_36056

theorem smallest_value (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  x^3 < x ∧ x^3 < 3*x ∧ x^3 < x^(1/3) ∧ x^3 < 1/(x+1) := by
  sorry

end smallest_value_l360_36056


namespace expression_equivalence_l360_36035

theorem expression_equivalence (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 1) / x) * ((y^2 + 1) / y) + ((x^2 - 1) / y) * ((y^2 - 1) / x) = 2 * x * y + 2 / (x * y) :=
by sorry

end expression_equivalence_l360_36035


namespace x_plus_y_value_l360_36077

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.sin y = 2023)
  (eq2 : x + 2023 * Real.cos y = 2021)
  (y_range : π/4 ≤ y ∧ y ≤ 3*π/4) :
  x + y = 2023 - Real.sqrt 2 / 2 + 3*π/4 := by
  sorry

end x_plus_y_value_l360_36077


namespace order_of_numbers_l360_36097

theorem order_of_numbers : 
  20.3 > 1 → 
  0 < 0.32 ∧ 0.32 < 1 → 
  Real.log 0.32 < 0 → 
  Real.log 0.32 < 0.32 ∧ 0.32 < 20.3 := by
sorry

end order_of_numbers_l360_36097


namespace gcd_204_85_l360_36083

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l360_36083


namespace printer_depreciation_l360_36073

def initial_price : ℝ := 625000
def first_year_depreciation : ℝ := 0.20
def subsequent_depreciation : ℝ := 0.08
def target_value : ℝ := 400000

def resale_value (n : ℕ) : ℝ :=
  if n = 0 then initial_price
  else if n = 1 then initial_price * (1 - first_year_depreciation)
  else (resale_value (n - 1)) * (1 - subsequent_depreciation)

theorem printer_depreciation :
  resale_value 4 < target_value ∧
  ∀ k : ℕ, k < 4 → resale_value k ≥ target_value :=
sorry

end printer_depreciation_l360_36073


namespace oldest_child_age_l360_36004

theorem oldest_child_age (ages : Fin 4 → ℕ) 
  (h_average : (ages 0 + ages 1 + ages 2 + ages 3) / 4 = 9)
  (h_younger : ages 0 = 6 ∧ ages 1 = 8 ∧ ages 2 = 10) :
  ages 3 = 12 := by
sorry

end oldest_child_age_l360_36004
