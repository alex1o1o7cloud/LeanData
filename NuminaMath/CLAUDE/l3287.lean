import Mathlib

namespace gcd_204_85_l3287_328740

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l3287_328740


namespace circle_radius_l3287_328742

/-- The radius of a circle given its area and a modified area formula -/
theorem circle_radius (k : ℝ) (A : ℝ) (h1 : k = 4) (h2 : A = 225 * Real.pi) :
  ∃ (r : ℝ), k * Real.pi * r^2 = A ∧ r = 7.5 := by
  sorry

end circle_radius_l3287_328742


namespace rectangle_shorter_side_l3287_328725

theorem rectangle_shorter_side (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Positive dimensions
  2 * (a + b) = 52 ∧  -- Perimeter condition
  a * b = 168 ∧  -- Area condition
  a ≥ b  -- a is the longer side
  → b = 12 := by
sorry

end rectangle_shorter_side_l3287_328725


namespace total_washing_time_l3287_328747

/-- The time William spends washing a normal car -/
def normal_car_time : ℕ := 4 + 7 + 4 + 9

/-- The number of normal cars William washed -/
def normal_cars : ℕ := 2

/-- The number of SUVs William washed -/
def suvs : ℕ := 1

/-- The time multiplier for washing an SUV compared to a normal car -/
def suv_time_multiplier : ℕ := 2

/-- Theorem: William spent 96 minutes washing all vehicles -/
theorem total_washing_time : 
  normal_car_time * normal_cars + normal_car_time * suv_time_multiplier * suvs = 96 := by
  sorry

end total_washing_time_l3287_328747


namespace insect_count_in_lab_l3287_328743

/-- Given a total number of insect legs and the number of legs per insect, 
    calculates the number of insects. -/
def count_insects (total_legs : ℕ) (legs_per_insect : ℕ) : ℕ :=
  total_legs / legs_per_insect

/-- Theorem stating that given 30 total insect legs and 6 legs per insect, 
    there are 5 insects in the laboratory. -/
theorem insect_count_in_lab : count_insects 30 6 = 5 := by
  sorry

end insect_count_in_lab_l3287_328743


namespace tan_value_implies_cosine_sine_ratio_l3287_328796

theorem tan_value_implies_cosine_sine_ratio 
  (α : Real) 
  (h : Real.tan α = 1/3) : 
  (Real.cos α)^2 - 2*(Real.sin α)^2 = 7/9 * (Real.cos α)^2 := by
  sorry

end tan_value_implies_cosine_sine_ratio_l3287_328796


namespace cube_of_negative_double_l3287_328700

theorem cube_of_negative_double (a : ℝ) : (-2 * a)^3 = -8 * a^3 := by
  sorry

end cube_of_negative_double_l3287_328700


namespace pastries_sold_l3287_328755

def initial_pastries : ℕ := 148
def remaining_pastries : ℕ := 45

theorem pastries_sold : initial_pastries - remaining_pastries = 103 := by
  sorry

end pastries_sold_l3287_328755


namespace velocity_dividing_trapezoid_area_l3287_328766

/-- 
Given a trapezoidal velocity-time graph with bases V and U, 
this theorem proves that the velocity W that divides the area 
under the curve in the ratio 1:k is given by W = √((V^2 + kU^2) / (k + 1)).
-/
theorem velocity_dividing_trapezoid_area 
  (V U : ℝ) (k : ℝ) (hk : k > 0) :
  let W := Real.sqrt ((V^2 + k * U^2) / (k + 1))
  ∃ (h : ℝ), 
    h * (V - W) = (1 / (k + 1)) * ((1 / 2) * h * (V + U)) ∧
    h * (W - U) = (k / (k + 1)) * ((1 / 2) * h * (V + U)) :=
by sorry

end velocity_dividing_trapezoid_area_l3287_328766


namespace cos_sin_identity_l3287_328746

theorem cos_sin_identity (α β : Real) :
  (Real.cos (α * π / 180) * Real.cos ((180 - α) * π / 180) + 
   Real.sin (α * π / 180) * Real.sin ((α / 2) * π / 180)) = -1/2 :=
by sorry

end cos_sin_identity_l3287_328746


namespace yoga_studio_women_count_l3287_328722

theorem yoga_studio_women_count :
  let num_men : ℕ := 8
  let avg_weight_men : ℚ := 190
  let avg_weight_women : ℚ := 120
  let total_people : ℕ := 14
  let avg_weight_all : ℚ := 160
  let num_women : ℕ := total_people - num_men
  (num_men : ℚ) * avg_weight_men + (num_women : ℚ) * avg_weight_women = (total_people : ℚ) * avg_weight_all →
  num_women = 6 :=
by
  sorry

end yoga_studio_women_count_l3287_328722


namespace expression_equals_24_l3287_328791

/-- An arithmetic expression using integers and basic operators -/
inductive Expr where
  | const : Int → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an arithmetic expression -/
def eval : Expr → Int
  | Expr.const n => n
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Check if an expression uses each of the given numbers exactly once -/
def usesNumbers (e : Expr) (nums : List Int) : Bool := sorry

/-- There exists an arithmetic expression using 1, 4, 7, and 7 that evaluates to 24 -/
theorem expression_equals_24 : ∃ e : Expr, 
  usesNumbers e [1, 4, 7, 7] ∧ eval e = 24 := by sorry

end expression_equals_24_l3287_328791


namespace find_divisor_l3287_328712

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 16698) (h2 : quotient = 89) (h3 : remainder = 14) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 187 :=
by sorry

end find_divisor_l3287_328712


namespace unique_solution_l3287_328709

theorem unique_solution : ∃! n : ℝ, 7 * n - 15 = 2 * n + 10 := by
  sorry

end unique_solution_l3287_328709


namespace sue_dogs_walked_l3287_328759

def perfume_cost : ℕ := 50
def christian_initial_savings : ℕ := 5
def sue_initial_savings : ℕ := 7
def yards_mowed : ℕ := 4
def yard_mowing_rate : ℕ := 5
def dog_walking_rate : ℕ := 2
def additional_needed : ℕ := 6

theorem sue_dogs_walked :
  ∃ (dogs_walked : ℕ),
    perfume_cost =
      christian_initial_savings + sue_initial_savings +
      yards_mowed * yard_mowing_rate +
      dogs_walked * dog_walking_rate +
      additional_needed ∧
    dogs_walked = 6 := by
  sorry

end sue_dogs_walked_l3287_328759


namespace quadratic_equation_solution_l3287_328719

theorem quadratic_equation_solution : ∃ y : ℝ, y^2 + 6*y + 8 = -(y + 4)*(y + 6) ∧ y = -4 := by
  sorry

end quadratic_equation_solution_l3287_328719


namespace derivative_sin_squared_minus_cos_squared_l3287_328778

theorem derivative_sin_squared_minus_cos_squared (x : ℝ) :
  (deriv (fun x => Real.sin x ^ 2 - Real.cos x ^ 2)) x = 2 * Real.sin (2 * x) :=
by sorry

end derivative_sin_squared_minus_cos_squared_l3287_328778


namespace divisible_by_512_l3287_328758

theorem divisible_by_512 (n : ℤ) (h : Odd n) :
  ∃ k : ℤ, n^12 - n^8 - n^4 + 1 = 512 * k := by
sorry

end divisible_by_512_l3287_328758


namespace linear_function_properties_l3287_328794

theorem linear_function_properties (m k b : ℝ) (h1 : m > 1) 
  (h2 : k * m + b = 1) (h3 : -k + b = m) : k < 0 ∧ b > 0 := by
  sorry

end linear_function_properties_l3287_328794


namespace cupcake_cookie_price_ratio_l3287_328775

theorem cupcake_cookie_price_ratio :
  ∀ (cookie_price cupcake_price : ℚ),
    cookie_price > 0 →
    cupcake_price > 0 →
    5 * cookie_price + 3 * cupcake_price = 23 →
    4 * cookie_price + 4 * cupcake_price = 21 →
    cupcake_price / cookie_price = 13 / 29 := by
  sorry

end cupcake_cookie_price_ratio_l3287_328775


namespace percentage_increase_proof_l3287_328757

def lowest_price : ℝ := 10
def highest_price : ℝ := 17

theorem percentage_increase_proof :
  (highest_price - lowest_price) / lowest_price * 100 = 70 := by sorry

end percentage_increase_proof_l3287_328757


namespace olivia_baseball_cards_l3287_328765

/-- The number of decks of baseball cards Olivia bought -/
def baseball_decks : ℕ :=
  let basketball_packs : ℕ := 2
  let basketball_price : ℕ := 3
  let baseball_price : ℕ := 4
  let initial_money : ℕ := 50
  let change : ℕ := 24
  let total_spent : ℕ := initial_money - change
  let basketball_cost : ℕ := basketball_packs * basketball_price
  let baseball_cost : ℕ := total_spent - basketball_cost
  baseball_cost / baseball_price

theorem olivia_baseball_cards : baseball_decks = 5 := by
  sorry

end olivia_baseball_cards_l3287_328765


namespace solve_equation_l3287_328706

theorem solve_equation (x : ℝ) : 5 * x + 3 = 10 * x - 17 → x = 4 := by
  sorry

end solve_equation_l3287_328706


namespace num_friends_is_four_l3287_328793

/-- The number of friends who volunteered with James to plant flowers -/
def num_friends : ℕ :=
  let total_flowers : ℕ := 200
  let days : ℕ := 2
  let james_flowers_per_day : ℕ := 20
  (total_flowers - james_flowers_per_day * days) / (james_flowers_per_day * days)

theorem num_friends_is_four : num_friends = 4 := by
  sorry

end num_friends_is_four_l3287_328793


namespace arithmetic_sequence_common_difference_l3287_328721

/-- Given an arithmetic sequence {aₙ}, prove that its common difference is 2 -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Definition of arithmetic sequence
  (h_sum : a 1 + a 5 = 10)  -- Given condition
  (h_S4 : (a 1 + a 2 + a 3 + a 4) = 16)  -- Given condition for S₄
  : a 2 - a 1 = 2 :=
sorry

end arithmetic_sequence_common_difference_l3287_328721


namespace bet_winnings_ratio_l3287_328768

def initial_amount : ℕ := 400
def final_amount : ℕ := 1200

def amount_won : ℕ := final_amount - initial_amount

theorem bet_winnings_ratio :
  (amount_won : ℚ) / initial_amount = 2 := by sorry

end bet_winnings_ratio_l3287_328768


namespace seating_theorem_l3287_328763

/-- The number of ways to seat n people around a round table. -/
def circular_permutations (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to seat 6 people around a round table,
    with two specific people always sitting next to each other. -/
def seating_arrangements : ℕ :=
  2 * circular_permutations 5

theorem seating_theorem : seating_arrangements = 48 := by
  sorry

end seating_theorem_l3287_328763


namespace polynomial_roots_l3287_328730

theorem polynomial_roots : ∃ (p : ℝ → ℝ), 
  (∀ x, p x = 6 * x^4 + 19 * x^3 - 51 * x^2 + 20 * x) ∧ 
  (p 0 = 0) ∧ 
  (p (1/2) = 0) ∧ 
  (p (4/3) = 0) ∧ 
  (p (-5) = 0) := by
sorry

end polynomial_roots_l3287_328730


namespace race_finish_order_l3287_328723

def race_order : List Nat := [1, 7, 9, 10, 8, 11, 2, 5, 3, 4, 6, 12]

theorem race_finish_order :
  ∀ (finish : Nat → Nat),
  (∀ n, n ∈ race_order → finish n ∈ Finset.range 13) →
  (∀ n, n ∈ race_order → ∃ k, n * (finish n) = 13 * k + 1) →
  (∀ n m, n ≠ m → n ∈ race_order → m ∈ race_order → finish n ≠ finish m) →
  (∀ n, n ∈ race_order → finish n = (List.indexOf n race_order).succ) :=
by sorry

#check race_finish_order

end race_finish_order_l3287_328723


namespace tens_digit_of_13_pow_1997_l3287_328760

theorem tens_digit_of_13_pow_1997 :
  13^1997 % 100 = 53 := by sorry

end tens_digit_of_13_pow_1997_l3287_328760


namespace date_statistics_order_l3287_328738

def date_counts : List (Nat × Nat) := 
  (List.range 30).map (fun n => (n + 1, 12)) ++ [(31, 7)]

def total_count : Nat := date_counts.foldl (fun acc (_, count) => acc + count) 0

def sum_of_values : Nat := date_counts.foldl (fun acc (date, count) => acc + date * count) 0

def mean : ℚ := sum_of_values / total_count

def median : Nat := 16

def median_of_modes : ℚ := 15.5

theorem date_statistics_order : median_of_modes < mean ∧ mean < median := by sorry

end date_statistics_order_l3287_328738


namespace exact_selection_probability_l3287_328720

def num_forks : ℕ := 8
def num_spoons : ℕ := 8
def num_knives : ℕ := 8
def total_pieces : ℕ := num_forks + num_spoons + num_knives
def selected_pieces : ℕ := 6

def probability_exact_selection : ℚ :=
  (Nat.choose num_forks 2 * Nat.choose num_spoons 2 * Nat.choose num_knives 2) /
  Nat.choose total_pieces selected_pieces

theorem exact_selection_probability :
  probability_exact_selection = 2744 / 16825 := by
  sorry

#eval probability_exact_selection

end exact_selection_probability_l3287_328720


namespace intersection_of_A_and_B_l3287_328717

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - x)}
def B : Set ℝ := {1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {1, 2, 3} := by sorry

end intersection_of_A_and_B_l3287_328717


namespace ant_count_approximation_l3287_328772

/-- Calculates the approximate number of ants in a rectangular field -/
def approximate_ant_count (width_feet : ℝ) (length_feet : ℝ) (ants_per_sq_inch : ℝ) : ℝ :=
  let width_inches := width_feet * 12
  let length_inches := length_feet * 12
  let area_sq_inches := width_inches * length_inches
  area_sq_inches * ants_per_sq_inch

/-- Theorem stating that the number of ants in the given field is approximately 59 million -/
theorem ant_count_approximation :
  let field_width := 250
  let field_length := 330
  let ants_density := 5
  let calculated_count := approximate_ant_count field_width field_length ants_density
  abs (calculated_count - 59000000) / 59000000 < 0.01 := by
  sorry

end ant_count_approximation_l3287_328772


namespace sqrt_product_simplification_l3287_328702

theorem sqrt_product_simplification (x : ℝ) (hx : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (8 * x) = 30 * x * Real.sqrt (2 * x) :=
by sorry

end sqrt_product_simplification_l3287_328702


namespace sixth_student_matches_l3287_328731

/-- Represents the number of matches played by each student -/
structure MatchCounts where
  student1 : ℕ
  student2 : ℕ
  student3 : ℕ
  student4 : ℕ
  student5 : ℕ
  student6 : ℕ

/-- The total number of matches in a complete tournament with 6 players -/
def totalMatches : ℕ := 15

/-- Theorem stating that if 5 students have played 5, 4, 3, 2, and 1 matches respectively,
    then the 6th student must have played 3 matches -/
theorem sixth_student_matches (mc : MatchCounts) : 
  mc.student1 = 5 ∧ 
  mc.student2 = 4 ∧ 
  mc.student3 = 3 ∧ 
  mc.student4 = 2 ∧ 
  mc.student5 = 1 ∧
  (mc.student1 + mc.student2 + mc.student3 + mc.student4 + mc.student5 + mc.student6 = 2 * totalMatches) →
  mc.student6 = 3 := by
  sorry

end sixth_student_matches_l3287_328731


namespace squares_in_100th_ring_l3287_328799

/-- The number of squares in the nth ring of a diamond pattern -/
def ring_squares (n : ℕ) : ℕ :=
  4 + 8 * (n - 1)

/-- Theorem stating the number of squares in the 100th ring -/
theorem squares_in_100th_ring :
  ring_squares 100 = 796 := by
  sorry

end squares_in_100th_ring_l3287_328799


namespace sequence_general_term_l3287_328728

def S (n : ℕ+) : ℚ := 2 * n.val ^ 2 + n.val

def a (n : ℕ+) : ℚ := 4 * n.val - 1

theorem sequence_general_term (n : ℕ+) : 
  (∀ k : ℕ+, S k - S (k - 1) = a k) ∧ S 1 = a 1 := by sorry

end sequence_general_term_l3287_328728


namespace red_pencils_count_l3287_328774

theorem red_pencils_count (total_packs : ℕ) (normal_red_per_pack : ℕ) (special_packs : ℕ) (extra_red_per_special : ℕ) : 
  total_packs = 15 → 
  normal_red_per_pack = 1 → 
  special_packs = 3 → 
  extra_red_per_special = 2 → 
  total_packs * normal_red_per_pack + special_packs * extra_red_per_special = 21 := by
sorry

end red_pencils_count_l3287_328774


namespace im_z_squared_gt_two_iff_xy_gt_one_l3287_328756

/-- For a complex number z, Im(z^2) > 2 if and only if the product of its real and imaginary parts is greater than 1 -/
theorem im_z_squared_gt_two_iff_xy_gt_one (z : ℂ) :
  Complex.im (z^2) > 2 ↔ Complex.re z * Complex.im z > 1 := by
sorry

end im_z_squared_gt_two_iff_xy_gt_one_l3287_328756


namespace hyperbola_curve_is_hyperbola_l3287_328733

/-- A curve defined by x = cos^2 u and y = sin^4 u for real u -/
def HyperbolaCurve : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ u : ℝ, p.1 = Real.cos u ^ 2 ∧ p.2 = Real.sin u ^ 4}

/-- The curve defined by HyperbolaCurve is a hyperbola -/
theorem hyperbola_curve_is_hyperbola : 
  ∃ a b c d e f : ℝ, a ≠ 0 ∧ (a * b > 0 ∨ a * b < 0) ∧
  ∀ p : ℝ × ℝ, p ∈ HyperbolaCurve ↔ 
    a * p.1^2 + b * p.2^2 + c * p.1 * p.2 + d * p.1 + e * p.2 + f = 0 :=
sorry

end hyperbola_curve_is_hyperbola_l3287_328733


namespace equation_solution_l3287_328751

theorem equation_solution (a c : ℝ) :
  let x := (c^2 - a^3) / (3*a^2 - 1)
  x^2 + c^2 = (a - x)^3 := by
  sorry

end equation_solution_l3287_328751


namespace triangle_equilateral_condition_l3287_328795

/-- Triangle ABC with angles A, B, C and sides a, b, c -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- A triangle is equilateral if all its sides are equal -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- The theorem stating the conditions and conclusion about the triangle -/
theorem triangle_equilateral_condition (t : Triangle)
    (h1 : t.B = (t.A + t.C) / 2)  -- B is arithmetic mean of A and C
    (h2 : t.b ^ 2 = t.a * t.c)    -- b is geometric mean of a and c
    : t.isEquilateral := by
  sorry

end triangle_equilateral_condition_l3287_328795


namespace square_and_fourth_power_mod_eight_l3287_328753

theorem square_and_fourth_power_mod_eight (n : ℤ) :
  (Even n → n ^ 2 % 8 = 0 ∨ n ^ 2 % 8 = 4) ∧
  (Odd n → n ^ 2 % 8 = 1) ∧
  (Odd n → n ^ 4 % 8 = 1) := by
  sorry

end square_and_fourth_power_mod_eight_l3287_328753


namespace definite_integral_x_plus_two_cubed_ln_squared_l3287_328710

open Real MeasureTheory

theorem definite_integral_x_plus_two_cubed_ln_squared :
  ∫ x in (-1)..(0), (x + 2)^3 * (log (x + 2))^2 = 4 * (log 2)^2 - 2 * log 2 + 15/32 := by
  sorry

end definite_integral_x_plus_two_cubed_ln_squared_l3287_328710


namespace triangle_inequality_l3287_328726

theorem triangle_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
by sorry

end triangle_inequality_l3287_328726


namespace bees_flew_in_l3287_328767

/-- Given an initial number of bees in a hive and a total number of bees after more flew in,
    this theorem proves that the number of bees that flew in is equal to the difference
    between the total and initial number of bees. -/
theorem bees_flew_in (initial_bees total_bees : ℕ) 
    (h1 : initial_bees = 16) 
    (h2 : total_bees = 26) : 
  total_bees - initial_bees = 10 := by
  sorry

#check bees_flew_in

end bees_flew_in_l3287_328767


namespace factorization_3mx_minus_9my_l3287_328752

theorem factorization_3mx_minus_9my (m x y : ℝ) :
  3 * m * x - 9 * m * y = 3 * m * (x - 3 * y) := by
  sorry

end factorization_3mx_minus_9my_l3287_328752


namespace recurring_decimal_fraction_sum_l3287_328707

theorem recurring_decimal_fraction_sum (a b : ℕ+) :
  (a : ℚ) / (b : ℚ) = 36 / 99 →
  Nat.gcd a b = 1 →
  a + b = 15 := by
  sorry

end recurring_decimal_fraction_sum_l3287_328707


namespace chantel_final_bracelet_count_l3287_328789

/-- The number of bracelets Chantel has at the end -/
def final_bracelet_count : ℕ :=
  let first_week_production := 7 * 4
  let after_first_giveaway := first_week_production - 8
  let second_period_production := 10 * 5
  let before_second_giveaway := after_first_giveaway + second_period_production
  before_second_giveaway - 12

/-- Theorem stating that Chantel ends up with 58 bracelets -/
theorem chantel_final_bracelet_count : final_bracelet_count = 58 := by
  sorry

end chantel_final_bracelet_count_l3287_328789


namespace alice_bob_games_l3287_328737

/-- The number of players in the league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- The number of games two specific players play together -/
def games_together : ℕ := 210

/-- The total number of possible game combinations -/
def total_combinations : ℕ := Nat.choose total_players players_per_game

theorem alice_bob_games :
  games_together = Nat.choose (total_players - 2) (players_per_game - 2) :=
by sorry

#check alice_bob_games

end alice_bob_games_l3287_328737


namespace product_xyz_is_zero_l3287_328708

theorem product_xyz_is_zero 
  (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 1) 
  (h3 : y ≠ 0) 
  (h4 : z ≠ 0) : 
  x * y * z = 0 := by
sorry

end product_xyz_is_zero_l3287_328708


namespace bears_per_shelf_l3287_328748

theorem bears_per_shelf (initial_stock : ℕ) (new_shipment : ℕ) (num_shelves : ℕ) :
  initial_stock = 4 →
  new_shipment = 10 →
  num_shelves = 2 →
  (initial_stock + new_shipment) / num_shelves = 7 :=
by sorry

end bears_per_shelf_l3287_328748


namespace sqrt_two_plus_one_squared_l3287_328732

theorem sqrt_two_plus_one_squared : (Real.sqrt 2 + 1)^2 = 3 + 2 * Real.sqrt 2 := by
  sorry

end sqrt_two_plus_one_squared_l3287_328732


namespace other_number_in_product_l3287_328779

theorem other_number_in_product (P w n : ℕ) : 
  P % 2^4 = 0 →
  P % 3^3 = 0 →
  P % 13^3 = 0 →
  P = n * w →
  w > 0 →
  w = 468 →
  (∀ w' : ℕ, w' > 0 ∧ w' < w → ¬(P % w' = 0)) →
  n = 2028 := by
sorry

end other_number_in_product_l3287_328779


namespace two_invariant_lines_l3287_328745

/-- Given a transformation from (x,y) to (x',y'), prove the existence of exactly two lines
    that both (x,y) and (x',y') lie on. -/
theorem two_invariant_lines 
  (x y x' y' : ℝ) 
  (h1 : x' = 3 * x + 2 * y + 1) 
  (h2 : y' = x + 4 * y - 3) :
  ∃! (L1 L2 : ℝ → ℝ → ℝ),
    (∀ x y, L1 x y = 0 ↔ L2 x y = 0 → L1 = L2) ∧
    (∀ x y, L1 x y = 0 → L1 x' y' = 0) ∧
    (∀ x y, L2 x y = 0 → L2 x' y' = 0) ∧
    L1 x y = x - y + 4 ∧
    L2 x y = 4 * x - 8 * y - 5 :=
by sorry

end two_invariant_lines_l3287_328745


namespace xyz_product_l3287_328761

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 3 * y = -9)
  (eq2 : y * z + 3 * z = -9)
  (eq3 : z * x + 3 * x = -9) :
  x * y * z = 27 := by sorry

end xyz_product_l3287_328761


namespace temperature_difference_l3287_328754

/-- The difference between the highest and lowest temperatures of the day -/
theorem temperature_difference (highest lowest : ℤ) (h1 : highest = 1) (h2 : lowest = -9) :
  highest - lowest = 10 := by
  sorry

end temperature_difference_l3287_328754


namespace classroom_pencils_l3287_328792

/-- The number of pencils a teacher needs to give out to a classroom of students -/
def pencils_to_give_out (num_students : ℕ) (dozens_per_student : ℕ) : ℕ :=
  num_students * (dozens_per_student * 12)

/-- Theorem: Given 46 children in a classroom, with each child receiving 4 dozen pencils,
    the total number of pencils the teacher needs to give out is 2208 -/
theorem classroom_pencils : pencils_to_give_out 46 4 = 2208 := by
  sorry

end classroom_pencils_l3287_328792


namespace sufficient_not_necessary_l3287_328741

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, 1 < a ∧ a < 2 → a^2 - 3*a ≤ 0) ∧
  (∃ a, a^2 - 3*a ≤ 0 ∧ ¬(1 < a ∧ a < 2)) :=
by sorry

end sufficient_not_necessary_l3287_328741


namespace obtuse_angle_range_l3287_328718

def vector_AB (x : ℝ) : ℝ × ℝ := (x, 2*x)
def vector_AC (x : ℝ) : ℝ × ℝ := (-3*x, 2)

def is_obtuse_angle (x : ℝ) : Prop :=
  let dot_product := (vector_AB x).1 * (vector_AC x).1 + (vector_AB x).2 * (vector_AC x).2
  dot_product < 0 ∧ x ≠ -1/3

def range_of_x : Set ℝ :=
  {x | x < -1/3 ∨ (-1/3 < x ∧ x < 0) ∨ x > 4/3}

theorem obtuse_angle_range :
  ∀ x, is_obtuse_angle x ↔ x ∈ range_of_x :=
sorry

end obtuse_angle_range_l3287_328718


namespace three_true_propositions_l3287_328781

-- Define reciprocals
def reciprocals (x y : ℝ) : Prop := x * y = 1

-- Define triangle congruence and area
def triangle_congruent (t1 t2 : Set ℝ × Set ℝ) : Prop := sorry
def triangle_area (t : Set ℝ × Set ℝ) : ℝ := sorry

-- Define the quadratic equation
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + m = 0

theorem three_true_propositions :
  (∀ x y : ℝ, reciprocals x y → x * y = 1) ∧
  (∃ t1 t2 : Set ℝ × Set ℝ, triangle_area t1 = triangle_area t2 ∧ ¬ triangle_congruent t1 t2) ∧
  (∀ m : ℝ, ¬ has_real_roots m → m > 1) :=
by sorry

end three_true_propositions_l3287_328781


namespace specific_tetrahedron_volume_l3287_328797

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem: The volume of the specific tetrahedron PQRS is 1715/(144√2) -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 3,
    PS := 5,
    QR := 5,
    QS := 4,
    RS := 15 / 4 * Real.sqrt 2
  }
  tetrahedronVolume t = 1715 / (144 * Real.sqrt 2) := by
  sorry

end specific_tetrahedron_volume_l3287_328797


namespace certain_number_problem_l3287_328785

theorem certain_number_problem (x : ℝ) : 
  (0.8 * 40 = (4/5) * x + 16) → x = 20 := by
sorry

end certain_number_problem_l3287_328785


namespace product_equality_implies_sum_l3287_328780

theorem product_equality_implies_sum (m n : ℝ) : 
  (m^2 + 4*m + 5) * (n^2 - 2*n + 6) = 5 → 2*m + 3*n = -1 := by
  sorry

end product_equality_implies_sum_l3287_328780


namespace circle_radius_is_three_l3287_328770

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 10*y + 32 = 0

/-- The radius of a circle given by its equation -/
def CircleRadius (eq : (ℝ → ℝ → Prop)) : ℝ :=
  sorry

theorem circle_radius_is_three :
  CircleRadius CircleEquation = 3 := by
  sorry

end circle_radius_is_three_l3287_328770


namespace factorization_equality_minimum_value_expression_minimum_value_at_one_l3287_328790

-- Problem 1
theorem factorization_equality (x y : ℝ) :
  1 - 2 * (x - y) + (x - y)^2 = (1 - x + y)^2 := by sorry

-- Problem 2
theorem minimum_value_expression (n : ℝ) :
  (n^2 - 2*n - 3) * (n^2 - 2*n + 5) + 17 ≥ 1 := by sorry

theorem minimum_value_at_one :
  (1^2 - 2*1 - 3) * (1^2 - 2*1 + 5) + 17 = 1 := by sorry

end factorization_equality_minimum_value_expression_minimum_value_at_one_l3287_328790


namespace yellow_to_red_ratio_l3287_328783

/-- Represents the number of chairs of each color in Susan's house. -/
structure ChairCounts where
  red : ℕ
  yellow : ℕ
  blue : ℕ

/-- The conditions of the chair problem in Susan's house. -/
def susansChairs : ChairCounts → Prop := fun c =>
  c.red = 5 ∧
  c.blue = c.yellow - 2 ∧
  c.red + c.yellow + c.blue = 43

/-- The theorem stating the ratio of yellow to red chairs. -/
theorem yellow_to_red_ratio (c : ChairCounts) (h : susansChairs c) :
  c.yellow / c.red = 4 := by
  sorry

#check yellow_to_red_ratio

end yellow_to_red_ratio_l3287_328783


namespace complex_modulus_equation_l3287_328715

theorem complex_modulus_equation (n : ℝ) (hn : 0 < n) :
  Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 13 → n = 10 * Real.sqrt 3 := by
  sorry

end complex_modulus_equation_l3287_328715


namespace prime_divides_mn_minus_one_l3287_328764

theorem prime_divides_mn_minus_one (m n p : ℕ) 
  (h_prime : Nat.Prime p)
  (h_order : m < n ∧ n < p)
  (h_div_m : p ∣ m^2 + 1)
  (h_div_n : p ∣ n^2 + 1) :
  p ∣ m * n - 1 := by
  sorry

end prime_divides_mn_minus_one_l3287_328764


namespace square_of_sum_product_l3287_328713

theorem square_of_sum_product (a b c d A : ℤ) 
  (h1 : a^2 + A = b^2) (h2 : c^2 + A = d^2) : 
  ∃ n : ℕ, 2 * (a + b) * (c + d) * (a * c + b * d - A) = n^2 := by
  sorry

end square_of_sum_product_l3287_328713


namespace smallest_positive_omega_l3287_328782

theorem smallest_positive_omega : ∃ ω : ℝ, ω > 0 ∧
  (∀ x : ℝ, Real.sin (ω * x - Real.pi / 4) = Real.cos (ω * (x - Real.pi / 2))) ∧
  (∀ ω' : ℝ, ω' > 0 → 
    (∀ x : ℝ, Real.sin (ω' * x - Real.pi / 4) = Real.cos (ω' * (x - Real.pi / 2))) → 
    ω ≤ ω') ∧
  ω = 3 / 2 := by
sorry

end smallest_positive_omega_l3287_328782


namespace calculate_income_person_income_l3287_328744

/-- Calculates a person's total income based on given distributions --/
theorem calculate_income (children_percentage : ℝ) (wife_percentage : ℝ) (orphan_percentage : ℝ) (remaining_amount : ℝ) : ℝ :=
  let total_children_percentage := 3 * children_percentage
  let remaining_percentage := 1 - (total_children_percentage + wife_percentage)
  let orphan_amount := orphan_percentage * remaining_percentage
  let final_percentage := remaining_percentage - orphan_amount
  remaining_amount / final_percentage

/-- Proves that the person's total income is approximately $168,421.05 --/
theorem person_income : 
  let income := calculate_income 0.15 0.3 0.05 40000
  ∃ ε > 0, |income - 168421.05| < ε :=
sorry

end calculate_income_person_income_l3287_328744


namespace sqrt_product_equality_l3287_328735

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l3287_328735


namespace heath_carrot_planting_rate_l3287_328716

/-- Proves that Heath planted an average of 3000 carrots per hour over the weekend --/
theorem heath_carrot_planting_rate :
  let total_rows : ℕ := 400
  let first_half_rows : ℕ := 200
  let second_half_rows : ℕ := 200
  let plants_per_row_first_half : ℕ := 275
  let plants_per_row_second_half : ℕ := 325
  let hours_first_half : ℕ := 15
  let hours_second_half : ℕ := 25

  let total_plants : ℕ := first_half_rows * plants_per_row_first_half + 
                          second_half_rows * plants_per_row_second_half
  let total_hours : ℕ := hours_first_half + hours_second_half

  (total_plants : ℚ) / (total_hours : ℚ) = 3000 := by
  sorry

end heath_carrot_planting_rate_l3287_328716


namespace space_diagonal_probability_l3287_328798

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The number of space diagonals in a cube -/
def space_diagonals : ℕ := 4

/-- The probability of selecting two vertices that are endpoints of a space diagonal -/
def probability : ℚ := 1 / 7

/-- Theorem: The probability of randomly selecting two vertices of a cube that are endpoints
    of a space diagonal is 1/7, given that a cube has 8 vertices and 4 space diagonals. -/
theorem space_diagonal_probability :
  (space_diagonals * 2 : ℚ) / (cube_vertices.choose 2) = probability := by
  sorry


end space_diagonal_probability_l3287_328798


namespace haley_marbles_l3287_328777

/-- The number of marbles Haley had, given the number of boys and marbles per boy -/
def total_marbles (num_boys : ℕ) (marbles_per_boy : ℕ) : ℕ :=
  num_boys * marbles_per_boy

/-- Theorem stating that Haley had 99 marbles -/
theorem haley_marbles : total_marbles 11 9 = 99 := by
  sorry

end haley_marbles_l3287_328777


namespace blocks_used_in_tower_l3287_328701

/-- Given that Randy initially had 59 blocks and now has 23 blocks left,
    prove that he used 36 blocks to build the tower. -/
theorem blocks_used_in_tower (initial_blocks : ℕ) (remaining_blocks : ℕ) 
  (h1 : initial_blocks = 59)
  (h2 : remaining_blocks = 23) : 
  initial_blocks - remaining_blocks = 36 := by
  sorry

end blocks_used_in_tower_l3287_328701


namespace functional_equation_solution_l3287_328714

open Real

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ w x y z : ℝ, w > 0 → x > 0 → y > 0 → z > 0 → w * x = y * z →
    (f w)^2 + (f x)^2 / (f (y^2) + f (z^2)) = (w^2 + x^2) / (y^2 + z^2)

/-- The main theorem stating the form of functions satisfying the equation -/
theorem functional_equation_solution (f : ℝ → ℝ) 
    (h1 : ∀ x, x > 0 → f x > 0) 
    (h2 : SatisfiesEquation f) : 
    ∀ x, x > 0 → (f x = x ∨ f x = 1 / x) := by
  sorry

end functional_equation_solution_l3287_328714


namespace corner_sum_implies_bottom_right_l3287_328786

/-- Represents a 24 by 24 grid containing numbers 1 to 576 -/
def Grid := Fin 24 → Fin 24 → Nat

/-- Checks if a given number is in the grid -/
def in_grid (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 576

/-- Defines a valid 24 by 24 grid -/
def is_valid_grid (g : Grid) : Prop :=
  ∀ i j, in_grid (g i j) ∧ g i j = i * 24 + j + 1

/-- Represents an 8 by 8 square within the grid -/
structure Square (g : Grid) where
  top_left : Fin 24 × Fin 24
  h_valid : top_left.1 + 7 < 24 ∧ top_left.2 + 7 < 24

/-- Gets the corner values of an 8 by 8 square -/
def corner_values (g : Grid) (s : Square g) : Fin 4 → Nat
| 0 => g s.top_left.1 s.top_left.2
| 1 => g s.top_left.1 (s.top_left.2 + 7)
| 2 => g (s.top_left.1 + 7) s.top_left.2
| 3 => g (s.top_left.1 + 7) (s.top_left.2 + 7)
| _ => 0

/-- The main theorem -/
theorem corner_sum_implies_bottom_right (g : Grid) (s : Square g) :
  is_valid_grid g →
  (corner_values g s 0 + corner_values g s 1 + corner_values g s 2 + corner_values g s 3 = 1646) →
  corner_values g s 3 = 499 := by
  sorry

end corner_sum_implies_bottom_right_l3287_328786


namespace circle_equation_l3287_328739

-- Define the line L1: x + y + 2 = 0
def L1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 + 2 = 0}

-- Define the circle C1: x² + y² = 4
def C1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define the line L2: 2x - y - 3 = 0
def L2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2*p.1 - p.2 - 3 = 0}

-- Define the circle C we're looking for
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 - 6*p.2 - 16 = 0}

theorem circle_equation :
  (∀ p ∈ L1 ∩ C1, p ∈ C) ∧
  (∃ center ∈ L2, ∀ p ∈ C, (p.1 - center.1)^2 + (p.2 - center.2)^2 = (6^2 + 6^2) / 4) :=
sorry

end circle_equation_l3287_328739


namespace arithmetic_mean_of_fractions_l3287_328727

theorem arithmetic_mean_of_fractions (x b c : ℝ) (hx : x ≠ 0) (hc : c ≠ 0) :
  ((x + b) / (c * x) + (x - b) / (c * x)) / 2 = 1 / c :=
by sorry

end arithmetic_mean_of_fractions_l3287_328727


namespace eight_possible_rankings_l3287_328705

/-- Represents a player in the tournament -/
inductive Player : Type
| X : Player
| Y : Player
| Z : Player
| W : Player

/-- Represents a match between two players -/
structure Match :=
  (player1 : Player)
  (player2 : Player)

/-- Represents the tournament structure -/
structure Tournament :=
  (day1_match1 : Match)
  (day1_match2 : Match)
  (no_draws : Bool)

/-- Represents a final ranking of players -/
def Ranking := List Player

/-- Function to generate all possible rankings given a tournament structure -/
def generateRankings (t : Tournament) : List Ranking :=
  sorry

/-- Theorem stating that there are exactly 8 possible ranking sequences -/
theorem eight_possible_rankings (t : Tournament) 
  (h1 : t.day1_match1 = ⟨Player.X, Player.Y⟩)
  (h2 : t.day1_match2 = ⟨Player.Z, Player.W⟩)
  (h3 : t.no_draws = true)
  (h4 : (generateRankings t).length > 0)
  (h5 : [Player.X, Player.Z, Player.Y, Player.W] ∈ generateRankings t) :
  (generateRankings t).length = 8 :=
sorry

end eight_possible_rankings_l3287_328705


namespace carpet_width_l3287_328704

/-- Calculates the width of a carpet given room dimensions and carpeting costs -/
theorem carpet_width
  (room_length : ℝ)
  (room_breadth : ℝ)
  (carpet_cost_paisa : ℝ)
  (total_cost_rupees : ℝ)
  (h1 : room_length = 15)
  (h2 : room_breadth = 6)
  (h3 : carpet_cost_paisa = 30)
  (h4 : total_cost_rupees = 36)
  : ∃ (carpet_width : ℝ), carpet_width = 75 := by
  sorry

end carpet_width_l3287_328704


namespace power_function_through_point_l3287_328776

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = 8) : 
  f 3 = 27 := by
sorry

end power_function_through_point_l3287_328776


namespace shirt_price_proof_l3287_328762

theorem shirt_price_proof (shirt_price pants_price : ℝ) 
  (h1 : shirt_price ≠ pants_price)
  (h2 : 2 * shirt_price + 3 * pants_price = 120)
  (h3 : 3 * pants_price = 0.25 * 120) : 
  shirt_price = 45 := by
sorry

end shirt_price_proof_l3287_328762


namespace bennys_books_l3287_328736

/-- Given the number of books Sandy, Tim, and the total, find Benny's books --/
theorem bennys_books (sandy_books tim_books total_books : ℕ) 
  (h1 : sandy_books = 10)
  (h2 : tim_books = 33)
  (h3 : total_books = 67)
  (h4 : total_books = sandy_books + tim_books + benny_books) :
  benny_books = 24 := by
  sorry

end bennys_books_l3287_328736


namespace percent_to_decimal_l3287_328703

theorem percent_to_decimal (p : ℚ) : p / 100 = p / 100 := by sorry

end percent_to_decimal_l3287_328703


namespace mod_equivalence_l3287_328729

theorem mod_equivalence (n : ℕ) : 
  185 * 944 ≡ n [ZMOD 60] → 0 ≤ n → n < 60 → n = 40 := by
sorry

end mod_equivalence_l3287_328729


namespace archer_probability_l3287_328724

theorem archer_probability (p_a p_b : ℝ) (h_p_a : p_a = 1/3) (h_p_b : p_b = 1/2) :
  1 - p_a * p_b = 5/6 := by
  sorry

end archer_probability_l3287_328724


namespace purely_imaginary_z_l3287_328788

theorem purely_imaginary_z (z : ℂ) : 
  (∃ b : ℝ, z = b * I) → 
  (∃ c : ℝ, (z + 2)^2 - 8*I = c * I) → 
  z = -2*I :=
sorry

end purely_imaginary_z_l3287_328788


namespace last_two_digits_1976_power_100_l3287_328734

theorem last_two_digits_1976_power_100 : 
  1976^100 % 100 = 76 := by sorry

end last_two_digits_1976_power_100_l3287_328734


namespace inequality_system_solution_l3287_328787

theorem inequality_system_solution :
  {x : ℝ | 2 + x > 7 - 4*x ∧ x < (4 + x) / 2} = {x : ℝ | 1 < x ∧ x < 4} := by
  sorry

end inequality_system_solution_l3287_328787


namespace box_side_face_area_l3287_328769

/-- Represents a rectangular box with length, width, and height -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box -/
def volume (b : Box) : ℝ := b.length * b.width * b.height

/-- Calculates the area of the top face of a box -/
def topFaceArea (b : Box) : ℝ := b.length * b.width

/-- Calculates the area of the front face of a box -/
def frontFaceArea (b : Box) : ℝ := b.width * b.height

/-- Calculates the area of the side face of a box -/
def sideFaceArea (b : Box) : ℝ := b.length * b.height

theorem box_side_face_area (b : Box) 
  (h1 : volume b = 192)
  (h2 : frontFaceArea b = (1/2) * topFaceArea b)
  (h3 : topFaceArea b = (3/2) * sideFaceArea b) :
  sideFaceArea b = 32 := by
  sorry

end box_side_face_area_l3287_328769


namespace correct_division_result_l3287_328711

theorem correct_division_result (x : ℚ) (h : 9 - x = 3) : 96 / x = 16 := by
  sorry

end correct_division_result_l3287_328711


namespace complex_fraction_simplification_l3287_328773

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 - 11 * i) / (3 - 4 * i) = 2 - i :=
by sorry

end complex_fraction_simplification_l3287_328773


namespace x_value_proof_l3287_328784

theorem x_value_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h1 : x^2 / y = 2)
  (h2 : y^2 / z = 5)
  (h3 : z^2 / x = 7) : 
  x = (2800 : ℝ)^(1/7) := by
sorry

end x_value_proof_l3287_328784


namespace smallest_number_divisible_l3287_328749

theorem smallest_number_divisible (n : ℕ) : n = 257 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ : ℕ, 
    m + 7 = 8 * k₁ ∧ 
    m + 7 = 11 * k₂ ∧ 
    m + 7 = 24 * k₃)) ∧ 
  (∃ k₁ k₂ k₃ : ℕ, 
    n + 7 = 8 * k₁ ∧ 
    n + 7 = 11 * k₂ ∧ 
    n + 7 = 24 * k₃) := by
  sorry

end smallest_number_divisible_l3287_328749


namespace quadratic_roots_transformation_l3287_328771

theorem quadratic_roots_transformation (a b : ℝ) (r₁ r₂ : ℝ) : 
  r₁^2 + a*r₁ + b = 0 → 
  r₂^2 + a*r₂ + b = 0 → 
  ∃ t : ℝ, (r₁^2 + 2*r₁*r₂ + r₂^2)^2 + (ab - a^2)*(r₁^2 + 2*r₁*r₂ + r₂^2) + t = 0 ∧ 
           (r₁*r₂*(r₁ + r₂))^2 + (ab - a^2)*(r₁*r₂*(r₁ + r₂)) + t = 0 :=
by sorry

end quadratic_roots_transformation_l3287_328771


namespace display_configurations_l3287_328750

/-- The number of holes in the row -/
def num_holes : ℕ := 8

/-- The number of holes that can display at a time -/
def num_display : ℕ := 3

/-- The number of possible states for each displaying hole -/
def num_states : ℕ := 2

/-- A function that calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of possible configurations -/
def total_configurations : ℕ := sorry

theorem display_configurations :
  total_configurations = choose (num_holes - num_display + 1) num_display * num_states ^ num_display :=
sorry

end display_configurations_l3287_328750
