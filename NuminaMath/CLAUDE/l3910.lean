import Mathlib

namespace duration_is_twelve_hours_l3910_391015

/-- Calculates the duration of a population change period given birth rate, death rate, and total net increase -/
def calculate_duration (birth_rate : ℚ) (death_rate : ℚ) (net_increase : ℕ) : ℚ :=
  let net_rate_per_second := (birth_rate - death_rate) / 2
  let duration_seconds := net_increase / net_rate_per_second
  duration_seconds / 3600

/-- Theorem stating that given the specified birth rate, death rate, and net increase, the duration is 12 hours -/
theorem duration_is_twelve_hours :
  calculate_duration (7 : ℚ) (3 : ℚ) 172800 = 12 := by
  sorry

#eval calculate_duration (7 : ℚ) (3 : ℚ) 172800

end duration_is_twelve_hours_l3910_391015


namespace school_age_ratio_l3910_391089

theorem school_age_ratio :
  ∀ (total below_eight eight above_eight : ℕ),
    total = 125 →
    below_eight = total / 5 →
    eight = 60 →
    total = below_eight + eight + above_eight →
    (above_eight : ℚ) / (eight : ℚ) = 2 / 3 := by
  sorry

end school_age_ratio_l3910_391089


namespace perception_permutations_l3910_391013

def word_length : ℕ := 10
def p_count : ℕ := 2
def e_count : ℕ := 2

theorem perception_permutations :
  (word_length.factorial) / (p_count.factorial * e_count.factorial) = 907200 := by
  sorry

end perception_permutations_l3910_391013


namespace vector_addition_subtraction_l3910_391028

theorem vector_addition_subtraction :
  let v1 : Fin 3 → ℝ := ![4, -3, 7]
  let v2 : Fin 3 → ℝ := ![-1, 5, 2]
  let v3 : Fin 3 → ℝ := ![2, -4, 9]
  v1 + v2 - v3 = ![1, 6, 0] := by sorry

end vector_addition_subtraction_l3910_391028


namespace vertex_on_x_axis_l3910_391077

/-- A parabola with equation y = x^2 - 12x + c -/
def parabola (c : ℝ) (x : ℝ) : ℝ := x^2 - 12*x + c

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 6

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y (c : ℝ) : ℝ := parabola c vertex_x

/-- The vertex lies on the x-axis if and only if c = 36 -/
theorem vertex_on_x_axis (c : ℝ) : vertex_y c = 0 ↔ c = 36 := by
  sorry

end vertex_on_x_axis_l3910_391077


namespace dividend_calculation_l3910_391067

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 8) : 
  divisor * quotient + remainder = 161 := by
  sorry

end dividend_calculation_l3910_391067


namespace max_product_constraint_l3910_391041

theorem max_product_constraint (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 30) :
  x^5 * y^3 ≤ (18.75 : ℝ)^5 * (11.25 : ℝ)^3 := by
  sorry

end max_product_constraint_l3910_391041


namespace investment_problem_l3910_391010

theorem investment_problem (total_investment rate1 rate2 rate3 : ℚ) 
  (h1 : total_investment = 6000)
  (h2 : rate1 = 7/100)
  (h3 : rate2 = 9/100)
  (h4 : rate3 = 11/100)
  (h5 : ∃ (a b c : ℚ), a + b + c = total_investment ∧ a / b = 2/3 ∧ b / c = 3) :
  ∃ (a b c : ℚ), 
    a + b + c = total_investment ∧ 
    a / b = 2/3 ∧ 
    b / c = 3 ∧
    a * rate1 + b * rate2 + c * rate3 = 520/100 := by
  sorry

end investment_problem_l3910_391010


namespace odd_function_sum_l3910_391074

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f is periodic with period p if f(x + p) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_sum (f : ℝ → ℝ) (h_odd : IsOdd f) (h_period : ∀ x, f (x + 2) = -f x) (h_f1 : f 1 = 8) :
  f 2012 + f 2013 + f 2014 = 8 := by
  sorry

end odd_function_sum_l3910_391074


namespace remainder_problem_l3910_391090

theorem remainder_problem (n : ℤ) : n % 5 = 3 → (4 * n + 6) % 5 = 3 := by
  sorry

end remainder_problem_l3910_391090


namespace geometric_sequence_min_a5_l3910_391056

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_min_a5 (a : ℕ → ℝ) (h1 : GeometricSequence a) 
    (h2 : ∀ n, a n > 0) (h3 : a 3 - a 1 = 2) :
  ∃ m : ℝ, m = 8 ∧ ∀ x : ℝ, (∃ q : ℝ, q > 0 ∧ x = (2 * q^4) / (q^2 - 1)) → x ≥ m :=
sorry

end geometric_sequence_min_a5_l3910_391056


namespace rook_placements_on_chessboard_l3910_391079

/-- The number of ways to place n rooks on an n×n chessboard so that no two rooks 
    are in the same row or column -/
def valid_rook_placements (n : ℕ) : ℕ := Nat.factorial n

/-- The size of the chessboard -/
def board_size : ℕ := 8

theorem rook_placements_on_chessboard : 
  valid_rook_placements board_size = 40320 := by
  sorry

#eval valid_rook_placements board_size

end rook_placements_on_chessboard_l3910_391079


namespace lunch_cost_l3910_391026

def weekly_savings : ℝ := 50
def total_weeks : ℕ := 5
def final_savings : ℝ := 135

theorem lunch_cost (lunch_frequency : ℕ) (lunch_cost : ℝ) : 
  lunch_frequency = 2 ∧ 
  lunch_cost * (total_weeks / lunch_frequency) = weekly_savings * total_weeks - final_savings →
  lunch_cost = 57.5 := by
sorry

end lunch_cost_l3910_391026


namespace negation_of_existential_l3910_391052

theorem negation_of_existential (p : Prop) :
  (¬ ∃ (x : ℝ), x^2 > 1) ↔ (∀ (x : ℝ), x^2 ≤ 1) := by
  sorry

end negation_of_existential_l3910_391052


namespace man_walking_speed_percentage_l3910_391038

/-- Proves that if a man's usual time to cover a distance is 72.00000000000001 minutes,
    and he takes 24 minutes more when walking at a slower speed,
    then he is walking at 75% of his usual speed. -/
theorem man_walking_speed_percentage : 
  let usual_time : ℝ := 72.00000000000001
  let additional_time : ℝ := 24
  let new_time : ℝ := usual_time + additional_time
  let speed_ratio : ℝ := usual_time / new_time
  speed_ratio = 0.75 := by sorry

end man_walking_speed_percentage_l3910_391038


namespace initial_fee_is_correct_l3910_391051

/-- The initial fee for a taxi trip -/
def initial_fee : ℝ := 2.25

/-- The charge per 2/5 of a mile -/
def charge_per_two_fifths_mile : ℝ := 0.15

/-- The length of the trip in miles -/
def trip_length : ℝ := 3.6

/-- The total charge for the trip -/
def total_charge : ℝ := 3.60

/-- Theorem stating that the initial fee is correct given the conditions -/
theorem initial_fee_is_correct :
  initial_fee + (trip_length * (charge_per_two_fifths_mile * 5 / 2)) = total_charge :=
by sorry

end initial_fee_is_correct_l3910_391051


namespace fraction_subtraction_l3910_391098

theorem fraction_subtraction : (8 : ℚ) / 19 - (5 : ℚ) / 57 = (1 : ℚ) / 3 := by
  sorry

end fraction_subtraction_l3910_391098


namespace teacher_selection_l3910_391065

theorem teacher_selection (n m f k : ℕ) (h1 : n = m + f) (h2 : n = 9) (h3 : m = 3) (h4 : f = 6) (h5 : k = 5) :
  (Nat.choose n k) - (Nat.choose f k) = 120 :=
sorry

end teacher_selection_l3910_391065


namespace jose_remaining_caps_l3910_391097

-- Define the initial number of bottle caps Jose has
def initial_caps : ℝ := 143.6

-- Define the number of bottle caps given to Rebecca
def given_to_rebecca : ℝ := 89.2

-- Define the number of bottle caps given to Michael
def given_to_michael : ℝ := 16.7

-- Theorem to prove the number of bottle caps Jose has left
theorem jose_remaining_caps :
  initial_caps - (given_to_rebecca + given_to_michael) = 37.7 := by
  sorry

end jose_remaining_caps_l3910_391097


namespace imaginary_part_of_z_l3910_391061

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2*I) / z = I) : z.im = -1 := by
  sorry

end imaginary_part_of_z_l3910_391061


namespace fixed_points_of_f_l3910_391025

def f (x : ℝ) := x^2 - 2*x - 10

theorem fixed_points_of_f :
  ∀ x : ℝ, f x = x ↔ x = -2 ∨ x = 5 := by sorry

end fixed_points_of_f_l3910_391025


namespace tangent_line_at_origin_l3910_391096

/-- Given a real number a and a function f with the specified properties,
    prove that the tangent line to f at the origin has the equation 3x + y = 0 -/
theorem tangent_line_at_origin (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + (a-3)*x
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 + 2*a*x + (a-3)
  (∀ x, f' x = f' (-x)) →  -- f' is an even function
  (∃ k : ℝ, ∀ x y, y = f x → (y - f 0) = k * (x - 0) → 3*x + y = 0) := by
sorry

end tangent_line_at_origin_l3910_391096


namespace circle_radius_from_chords_l3910_391002

/-- Given a circle with two chords AB and AC, where AB = a, AC = b, and the length of arc AC is twice the length of arc AB, the radius of the circle is a²/√(4a² - b²). -/
theorem circle_radius_from_chords (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : b < 2*a) : 
  ∃ (R : ℝ), R > 0 ∧ R = a^2 / Real.sqrt (4*a^2 - b^2) := by
  sorry


end circle_radius_from_chords_l3910_391002


namespace fraction_equality_l3910_391046

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 5) :
  m / q = 1 := by
  sorry

end fraction_equality_l3910_391046


namespace max_consecutive_integers_sum_l3910_391023

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 44 ↔ n * (n + 1) ≤ 2000 := by sorry

#check max_consecutive_integers_sum

end max_consecutive_integers_sum_l3910_391023


namespace max_value_of_f_l3910_391063

def f (x : ℝ) : ℝ := x - 5

theorem max_value_of_f :
  ∃ (max : ℝ), max = 8 ∧
  ∀ x : ℝ, -5 ≤ x ∧ x ≤ 13 → f x ≤ max :=
by
  sorry

end max_value_of_f_l3910_391063


namespace albert_large_pizzas_l3910_391037

/-- The number of large pizzas Albert bought -/
def large_pizzas : ℕ := 2

/-- The number of small pizzas Albert bought -/
def small_pizzas : ℕ := 2

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 16

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := 8

/-- The total number of slices Albert ate -/
def total_slices : ℕ := 48

/-- Theorem stating that Albert bought 2 large pizzas -/
theorem albert_large_pizzas : 
  large_pizzas * large_pizza_slices + small_pizzas * small_pizza_slices = total_slices :=
by sorry

end albert_large_pizzas_l3910_391037


namespace special_points_divide_plane_into_four_regions_l3910_391029

-- Define the set of points where one coordinate is four times the other
def special_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 4 * p.2 ∨ p.2 = 4 * p.1}

-- Define a function that counts the number of regions
def count_regions (S : Set (ℝ × ℝ)) : ℕ := sorry

-- Theorem statement
theorem special_points_divide_plane_into_four_regions :
  count_regions special_points = 4 := by sorry

end special_points_divide_plane_into_four_regions_l3910_391029


namespace max_value_sqrt_x_10_minus_x_l3910_391081

theorem max_value_sqrt_x_10_minus_x :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 10 → Real.sqrt (x * (10 - x)) ≤ 5) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 10 ∧ Real.sqrt (x * (10 - x)) = 5) :=
by sorry

end max_value_sqrt_x_10_minus_x_l3910_391081


namespace marble_ratio_l3910_391020

/-- The number of marbles Wolfgang bought -/
def wolfgang_marbles : ℕ := 16

/-- The number of marbles Ludo bought -/
def ludo_marbles : ℕ := 20

/-- The number of marbles Michael bought -/
def michael_marbles : ℕ := (2 * (wolfgang_marbles + ludo_marbles)) / 3

/-- The total number of marbles -/
def total_marbles : ℕ := wolfgang_marbles + ludo_marbles + michael_marbles

/-- Each person's share when divided equally -/
def equal_share : ℕ := 20

theorem marble_ratio :
  (ludo_marbles : ℚ) / wolfgang_marbles = 5 / 4 ∧
  total_marbles = 3 * equal_share :=
sorry

end marble_ratio_l3910_391020


namespace baker_cheesecake_problem_l3910_391042

/-- Calculates the total number of cheesecakes left to be sold given the initial quantities and sales. -/
def cheesecakes_left_to_sell (display_initial : ℕ) (fridge_initial : ℕ) (sold_from_display : ℕ) : ℕ :=
  (display_initial - sold_from_display) + fridge_initial

/-- Theorem stating that given the specific initial quantities and sales, 18 cheesecakes are left to be sold. -/
theorem baker_cheesecake_problem :
  cheesecakes_left_to_sell 10 15 7 = 18 := by
  sorry

end baker_cheesecake_problem_l3910_391042


namespace fraction_equivalence_l3910_391019

theorem fraction_equivalence : ∃ n : ℚ, (2 + n) / (7 + n) = 3 / 5 :=
by
  use 11 / 2
  sorry

end fraction_equivalence_l3910_391019


namespace complex_sum_problem_l3910_391036

theorem complex_sum_problem (a b c d e f : ℝ) :
  d = 2 →
  e = -a - c →
  (Complex.mk a b) + (Complex.mk c d) + (Complex.mk e f) = Complex.I * 2 →
  b + f = 0 := by
  sorry

end complex_sum_problem_l3910_391036


namespace mod_eight_equivalence_l3910_391016

theorem mod_eight_equivalence : ∃ n : ℤ, 0 ≤ n ∧ n < 8 ∧ -2222 ≡ n [ZMOD 8] ∧ n = 2 := by
  sorry

end mod_eight_equivalence_l3910_391016


namespace store_A_cheaper_store_A_cheaper_30_l3910_391018

/-- The cost of buying pens at Store A -/
def costA (x : ℝ) : ℝ := 0.9 * x + 6

/-- The cost of buying pens at Store B -/
def costB (x : ℝ) : ℝ := 1.2 * x

/-- Theorem: Store A is cheaper than Store B for 20 or more pens -/
theorem store_A_cheaper (x : ℝ) (h : x ≥ 20) : costA x ≤ costB x := by
  sorry

/-- Corollary: Store A is cheaper for exactly 30 pens -/
theorem store_A_cheaper_30 : costA 30 < costB 30 := by
  sorry

end store_A_cheaper_store_A_cheaper_30_l3910_391018


namespace problem_solution_l3910_391030

theorem problem_solution :
  (∀ x : ℝ, (1 : ℝ) > 0 ∧ x^2 - 4*1*x + 3*1^2 < 0 ∧ (x-3)/(x-2) ≤ 0 → 2 < x ∧ x < 3) ∧
  (∀ a : ℝ, a > 0 ∧ 
    (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 ≥ 0 → (x-3)/(x-2) > 0) ∧
    (∃ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0 ∧ (x-3)/(x-2) > 0) →
    1 < a ∧ a ≤ 2) :=
by sorry

end problem_solution_l3910_391030


namespace amoeba_count_after_week_l3910_391040

/-- Calculates the number of amoebas on a given day -/
def amoeba_count (day : ℕ) : ℕ :=
  if day = 0 then 1
  else if day % 2 = 1 then 3 * amoeba_count (day - 1)
  else (3 * amoeba_count (day - 1)) / 2

/-- The number of amoebas after 7 days is 243 -/
theorem amoeba_count_after_week : amoeba_count 7 = 243 := by
  sorry

end amoeba_count_after_week_l3910_391040


namespace tangent_line_to_parabola_l3910_391084

/-- The value of d for which the line y = 4x + d is tangent to the parabola y^2 = 16x -/
theorem tangent_line_to_parabola : 
  ∃ (d : ℝ), (∀ x y : ℝ, y = 4*x + d ∧ y^2 = 16*x → (∃! x', y' = 4*x' + d ∧ y'^2 = 16*x')) → d = 1 :=
by sorry

end tangent_line_to_parabola_l3910_391084


namespace smallest_sum_of_squares_l3910_391057

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 231 → 
  ∀ a b : ℕ, a^2 - b^2 = 231 → x^2 + y^2 ≤ a^2 + b^2 → 
  x^2 + y^2 = 281 :=
by sorry

end smallest_sum_of_squares_l3910_391057


namespace circle_center_line_max_ab_l3910_391007

theorem circle_center_line_max_ab (a b : ℝ) :
  let circle := (fun (x y : ℝ) => x^2 + y^2 + 2*x - 4*y + 1 = 0)
  let center_line := (fun (x y : ℝ) => a*x - b*y + 1 = 0)
  let center := (-1, 2)
  (∀ x y, circle x y ↔ (x + 1)^2 + (y - 2)^2 = 4) →
  center_line (-1) 2 →
  (∀ k, k * a * b ≤ 1/8) :=
by sorry

end circle_center_line_max_ab_l3910_391007


namespace family_weight_problem_l3910_391048

/-- The total weight of a grandmother, her daughter, and her grandchild -/
def total_weight (mother_weight daughter_weight child_weight : ℝ) : ℝ :=
  mother_weight + daughter_weight + child_weight

/-- Theorem: Given the conditions, the total weight is 160 kg -/
theorem family_weight_problem :
  ∀ (mother_weight daughter_weight child_weight : ℝ),
    daughter_weight + child_weight = 60 →
    child_weight = (1 / 5) * mother_weight →
    daughter_weight = 40 →
    total_weight mother_weight daughter_weight child_weight = 160 :=
by
  sorry

end family_weight_problem_l3910_391048


namespace probability_black_then_white_l3910_391078

/-- The probability of drawing a black ball first and then a white ball from a bag. -/
theorem probability_black_then_white (white_balls black_balls : ℕ) : 
  white_balls = 7 → black_balls = 3 → 
  (black_balls : ℚ) / (white_balls + black_balls) * 
  (white_balls : ℚ) / (white_balls + black_balls - 1) = 7 / 30 := by
  sorry

#check probability_black_then_white

end probability_black_then_white_l3910_391078


namespace range_of_x_plus_inverse_x_l3910_391003

theorem range_of_x_plus_inverse_x (x : ℝ) (h : x < 0) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ -2 ∧ ∀ (z : ℝ), (∃ (w : ℝ), w < 0 ∧ z = w + 1/w) → z ≤ y :=
sorry

end range_of_x_plus_inverse_x_l3910_391003


namespace candidate_a_democratic_vote_percentage_l3910_391021

theorem candidate_a_democratic_vote_percentage
  (total_voters : ℝ)
  (democrat_percentage : ℝ)
  (republican_percentage : ℝ)
  (republican_vote_for_a : ℝ)
  (total_vote_for_a : ℝ)
  (h1 : democrat_percentage = 0.70)
  (h2 : republican_percentage = 1 - democrat_percentage)
  (h3 : republican_vote_for_a = 0.30)
  (h4 : total_vote_for_a = 0.65) :
  ∃ (democrat_vote_for_a : ℝ),
    democrat_vote_for_a * democrat_percentage +
    republican_vote_for_a * republican_percentage = total_vote_for_a ∧
    democrat_vote_for_a = 0.80 :=
sorry

end candidate_a_democratic_vote_percentage_l3910_391021


namespace matrix_commutation_l3910_391033

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]

def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]

theorem matrix_commutation (x y z w : ℝ) :
  A * B x y z w = B x y z w * A →
  4 * z ≠ y →
  (x - w) / (y - 4 * z) = -3/13 := by sorry

end matrix_commutation_l3910_391033


namespace opposite_of_negative_two_l3910_391001

theorem opposite_of_negative_two : 
  ∀ x : ℤ, x + (-2) = 0 → x = 2 := by
sorry

end opposite_of_negative_two_l3910_391001


namespace no_real_solutions_l3910_391053

theorem no_real_solutions :
  ¬ ∃ x : ℝ, Real.sqrt (3 * x - 2) + 8 / Real.sqrt (3 * x - 2) = 4 := by
sorry

end no_real_solutions_l3910_391053


namespace raine_change_calculation_l3910_391022

/-- Calculate the change Raine receives after purchasing items from a gift shop -/
theorem raine_change_calculation (bracelet_price gold_necklace_price mug_price : ℕ)
  (bracelet_quantity gold_necklace_quantity mug_quantity : ℕ)
  (paid_amount : ℕ) :
  bracelet_price = 15 →
  gold_necklace_price = 10 →
  mug_price = 20 →
  bracelet_quantity = 3 →
  gold_necklace_quantity = 2 →
  mug_quantity = 1 →
  paid_amount = 100 →
  paid_amount - (bracelet_price * bracelet_quantity + 
                 gold_necklace_price * gold_necklace_quantity + 
                 mug_price * mug_quantity) = 15 := by
  sorry

end raine_change_calculation_l3910_391022


namespace impossible_same_color_l3910_391017

/-- Represents the number of chips of each color -/
structure ChipState :=
  (blue : Nat)
  (red : Nat)
  (yellow : Nat)

/-- Represents a single recoloring step -/
inductive RecolorStep
  | BlueRedToYellow
  | RedYellowToBlue
  | BlueYellowToRed

/-- The initial state of chips -/
def initialState : ChipState :=
  { blue := 2008, red := 2009, yellow := 2010 }

/-- Applies a recoloring step to a given state -/
def applyStep (state : ChipState) (step : RecolorStep) : ChipState :=
  match step with
  | RecolorStep.BlueRedToYellow => 
      { blue := state.blue - 1, red := state.red - 1, yellow := state.yellow + 2 }
  | RecolorStep.RedYellowToBlue => 
      { blue := state.blue + 2, red := state.red - 1, yellow := state.yellow - 1 }
  | RecolorStep.BlueYellowToRed => 
      { blue := state.blue - 1, red := state.red + 2, yellow := state.yellow - 1 }

/-- Represents a sequence of recoloring steps -/
def RecolorSequence := List RecolorStep

/-- Applies a sequence of recoloring steps to the initial state -/
def applySequence (seq : RecolorSequence) : ChipState :=
  seq.foldl applyStep initialState

/-- Checks if all chips are of the same color -/
def allSameColor (state : ChipState) : Bool :=
  (state.blue = 0 && state.red = 0) ||
  (state.blue = 0 && state.yellow = 0) ||
  (state.red = 0 && state.yellow = 0)

/-- The main theorem: It's impossible to make all chips the same color -/
theorem impossible_same_color : ∀ (seq : RecolorSequence), ¬(allSameColor (applySequence seq)) := by
  sorry

end impossible_same_color_l3910_391017


namespace officer_selection_ways_l3910_391044

/-- The number of ways to select distinct officers from a group -/
def selectOfficers (n m : ℕ) : ℕ :=
  (n - 0) * (n - 1) * (n - 2) * (n - 3) * (n - 4)

/-- Theorem: Selecting 5 distinct officers from 12 people results in 95,040 ways -/
theorem officer_selection_ways :
  selectOfficers 12 5 = 95040 := by
  sorry

end officer_selection_ways_l3910_391044


namespace circle_radius_proof_l3910_391095

/-- Representation of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Representation of a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Given geometric configuration -/
theorem circle_radius_proof (C1 C2 : Circle) (X Y Z : Point) :
  -- C1's center O is on C2
  C2.center.x^2 + C2.center.y^2 = C2.radius^2 →
  -- X and Y are intersection points of C1 and C2
  (X.x - C1.center.x)^2 + (X.y - C1.center.y)^2 = C1.radius^2 →
  (X.x - C2.center.x)^2 + (X.y - C2.center.y)^2 = C2.radius^2 →
  (Y.x - C1.center.x)^2 + (Y.y - C1.center.y)^2 = C1.radius^2 →
  (Y.x - C2.center.x)^2 + (Y.y - C2.center.y)^2 = C2.radius^2 →
  -- Z is on C2 but outside C1
  (Z.x - C2.center.x)^2 + (Z.y - C2.center.y)^2 = C2.radius^2 →
  (Z.x - C1.center.x)^2 + (Z.y - C1.center.y)^2 > C1.radius^2 →
  -- Given distances
  (X.x - Z.x)^2 + (X.y - Z.y)^2 = 15^2 →
  (C1.center.x - Z.x)^2 + (C1.center.y - Z.y)^2 = 13^2 →
  (Y.x - Z.x)^2 + (Y.y - Z.y)^2 = 8^2 →
  -- Conclusion
  C1.radius = Real.sqrt 394 := by
    sorry


end circle_radius_proof_l3910_391095


namespace circle_equation_from_diameter_l3910_391066

/-- Given two points as the endpoints of a diameter of a circle, 
    prove that the equation of the circle is as stated. -/
theorem circle_equation_from_diameter (p1 p2 : ℝ × ℝ) 
  (h : p1 = (-1, 3) ∧ p2 = (5, -5)) : 
  ∃ (x y : ℝ), x^2 + y^2 - 4*x + 2*y - 20 = 0 := by
  sorry

end circle_equation_from_diameter_l3910_391066


namespace angle_hda_measure_l3910_391064

-- Define the points
variable (A B C D E F G H I : Point)

-- Define the shapes
def is_square (A B C D : Point) : Prop := sorry
def is_equilateral_triangle (C D E : Point) : Prop := sorry
def is_regular_hexagon (D E F G H I : Point) : Prop := sorry
def is_isosceles_triangle (G H I : Point) : Prop := sorry

-- Define the angle measure function
def angle_measure (A B C : Point) : ℝ := sorry

-- Theorem statement
theorem angle_hda_measure 
  (h1 : is_square A B C D)
  (h2 : is_equilateral_triangle C D E)
  (h3 : is_regular_hexagon D E F G H I)
  (h4 : is_isosceles_triangle G H I) :
  angle_measure H D A = 270 := by sorry

end angle_hda_measure_l3910_391064


namespace triangle_theorem_l3910_391005

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : (t.a^2 + t.b^2 - t.c^2) * Real.tan t.C = Real.sqrt 2 * t.a * t.b)
  (h2 : t.c = 2)
  (h3 : t.b = 2 * Real.sqrt 2) :
  t.C = π/4 ∧ t.a = 2 ∧ (1/2 * t.a * t.c = 2) := by
  sorry

end triangle_theorem_l3910_391005


namespace arithmetic_equality_l3910_391082

theorem arithmetic_equality : (3652 * 2487) + (979 - 45 * 13) = 9085008 := by
  sorry

end arithmetic_equality_l3910_391082


namespace max_k_on_unit_circle_l3910_391043

theorem max_k_on_unit_circle (k : ℤ) : 
  (0 ≤ k ∧ k ≤ 2019) →
  (∀ m : ℤ, 0 ≤ m ∧ m ≤ 2019 → 
    Complex.abs (Complex.exp (2 * Real.pi * Complex.I * (↑k / 2019)) - 1) ≥ 
    Complex.abs (Complex.exp (2 * Real.pi * Complex.I * (↑m / 2019)) - 1)) →
  k = 1010 := by
  sorry

end max_k_on_unit_circle_l3910_391043


namespace smallest_value_of_3a_plus_2_l3910_391060

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 7 * a + 6 = 5) :
  ∃ (min : ℝ), min = 1/2 ∧ ∀ x, (∃ y, 8 * y^2 + 7 * y + 6 = 5 ∧ 3 * y + 2 = x) → min ≤ x :=
by sorry

end smallest_value_of_3a_plus_2_l3910_391060


namespace carmen_cats_given_up_l3910_391049

/-- The number of cats Carmen gave up for adoption -/
def cats_given_up : ℕ := 3

/-- The initial number of cats Carmen had -/
def initial_cats : ℕ := 28

/-- The number of dogs Carmen has -/
def dogs : ℕ := 18

theorem carmen_cats_given_up :
  initial_cats - cats_given_up = dogs + 7 :=
by sorry

end carmen_cats_given_up_l3910_391049


namespace average_daily_net_income_is_366_l3910_391083

/-- Represents the financial data for a single day -/
structure DailyData where
  income : ℕ
  tips : ℕ
  expenses : ℕ

/-- Calculates the net income for a single day -/
def netIncome (data : DailyData) : ℕ :=
  data.income + data.tips - data.expenses

/-- The financial data for 5 days -/
def fiveDaysData : Vector DailyData 5 :=
  ⟨[
    { income := 300, tips := 50, expenses := 80 },
    { income := 150, tips := 20, expenses := 40 },
    { income := 750, tips := 100, expenses := 150 },
    { income := 200, tips := 30, expenses := 50 },
    { income := 600, tips := 70, expenses := 120 }
  ], rfl⟩

/-- Calculates the average daily net income -/
def averageDailyNetIncome (data : Vector DailyData 5) : ℚ :=
  (data.toList.map netIncome).sum / 5

/-- Theorem stating that the average daily net income is $366 -/
theorem average_daily_net_income_is_366 :
  averageDailyNetIncome fiveDaysData = 366 := by
  sorry

end average_daily_net_income_is_366_l3910_391083


namespace circle_tangent_to_x_axis_max_area_triangle_AOB_l3910_391086

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = 4*x
def line (x y b : ℝ) : Prop := x + 2*y - 2*b = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) (b : ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ line A.1 A.2 b ∧ line B.1 B.2 b

-- Define the circle with diameter AB
def circle_equation (x y x0 y0 r : ℝ) : Prop := (x - x0)^2 + (y - y0)^2 = r^2

-- Part I: Circle tangent to x-axis
theorem circle_tangent_to_x_axis 
  (A B : ℝ × ℝ) (b : ℝ) 
  (h_intersect : intersection_points A B b) 
  (h_tangent : ∃ (x0 y0 r : ℝ), circle_equation A.1 A.2 x0 y0 r ∧ 
                                circle_equation B.1 B.2 x0 y0 r ∧ 
                                y0 = r) : 
  ∃ (x0 y0 : ℝ), circle_equation x y x0 y0 4 ∧ x0 = 24/5 ∧ y0 = -4 :=
sorry

-- Part II: Maximum area of triangle AOB
theorem max_area_triangle_AOB 
  (A B : ℝ × ℝ) (b : ℝ) 
  (h_intersect : intersection_points A B b) 
  (h_negative_y : b < 0) : 
  ∃ (max_area : ℝ), max_area = 32 * Real.sqrt 3 / 9 ∧ 
    ∀ (area : ℝ), area ≤ max_area :=
sorry

end circle_tangent_to_x_axis_max_area_triangle_AOB_l3910_391086


namespace reflect_A_across_x_axis_l3910_391091

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem reflect_A_across_x_axis :
  let A : ℝ × ℝ := (-4, 3)
  reflect_x A = (-4, -3) := by
sorry

end reflect_A_across_x_axis_l3910_391091


namespace fraction_sum_equals_decimal_l3910_391062

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 20 + (5 : ℚ) / 200 + (7 : ℚ) / 2000 = (1785 : ℚ) / 10000 := by
  sorry

end fraction_sum_equals_decimal_l3910_391062


namespace cube_root_equation_l3910_391047

theorem cube_root_equation (x : ℝ) : (1 + Real.sqrt x)^(1/3 : ℝ) = 2 → x = 49 := by
  sorry

end cube_root_equation_l3910_391047


namespace sams_effective_speed_l3910_391076

/-- Represents the problem of calculating Sam's effective average speed -/
theorem sams_effective_speed (total_distance total_time first_speed second_speed stop_time : ℝ) :
  total_distance = 120 →
  total_time = 120 →
  first_speed = 50 →
  second_speed = 55 →
  stop_time = 5 →
  let first_segment_time : ℝ := 40
  let second_segment_time : ℝ := 40
  let last_segment_time : ℝ := total_time - first_segment_time - second_segment_time
  let first_segment_distance : ℝ := first_speed * (first_segment_time / 60)
  let second_segment_distance : ℝ := second_speed * (second_segment_time / 60)
  let last_segment_distance : ℝ := total_distance - first_segment_distance - second_segment_distance
  let effective_driving_time : ℝ := last_segment_time - stop_time
  let effective_average_speed : ℝ := (last_segment_distance / effective_driving_time) * 60
  effective_average_speed = 85 := by
  sorry

end sams_effective_speed_l3910_391076


namespace sqrt_2x_plus_4_real_range_l3910_391080

theorem sqrt_2x_plus_4_real_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = 2 * x + 4) ↔ x ≥ -2 := by
  sorry

end sqrt_2x_plus_4_real_range_l3910_391080


namespace sin_5pi_minus_alpha_l3910_391031

theorem sin_5pi_minus_alpha (α : ℝ) (h : Real.sin (π + α) = -(1/2)) :
  Real.sin (5*π - α) = 1/2 := by sorry

end sin_5pi_minus_alpha_l3910_391031


namespace circle_equation_from_diameter_l3910_391039

/-- The equation of a circle given two points on its diameter -/
theorem circle_equation_from_diameter (A B : ℝ × ℝ) :
  A = (0, 3) →
  B = (-4, 0) →
  ∀ (x y : ℝ),
    (x - (-2))^2 + (y - (3/2))^2 = (5/2)^2 ↔ x^2 + y^2 + 4*x - 3*y = 0 :=
by sorry

end circle_equation_from_diameter_l3910_391039


namespace product_digit_sum_l3910_391073

/-- Represents a 99-digit number with a repeating 3-digit pattern -/
def RepeatingNumber (pattern : Nat) : Nat :=
  -- Implementation details omitted for brevity
  sorry

theorem product_digit_sum :
  let a := RepeatingNumber 909
  let b := RepeatingNumber 707
  let product := a * b
  (product % 10) + ((product / 1000) % 10) = 8 := by
  sorry

end product_digit_sum_l3910_391073


namespace octagon_diagonals_l3910_391094

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in a regular octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end octagon_diagonals_l3910_391094


namespace solution_pairs_count_l3910_391071

theorem solution_pairs_count : 
  (∃ n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    7 * p.1 + 4 * p.2 = 800 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 801) (Finset.range 801))).card ∧ n = 29) := by
  sorry

end solution_pairs_count_l3910_391071


namespace delegate_grouping_exists_l3910_391088

/-- Represents a delegate with their country and seating position. -/
structure Delegate where
  country : Fin 50
  position : Fin 100

/-- Represents a seating arrangement of delegates around a circular table. -/
def SeatingArrangement := Fin 100 → Delegate

/-- Represents a grouping of delegates. -/
def Grouping := Delegate → Bool

/-- Checks if a delegate has at most one neighbor in the same group. -/
def atMostOneNeighborInGroup (s : SeatingArrangement) (g : Grouping) (d : Delegate) : Prop :=
  let leftNeighbor := s ((d.position - 1 + 100) % 100)
  let rightNeighbor := s ((d.position + 1) % 100)
  ¬(g leftNeighbor ∧ g rightNeighbor ∧ g d = g leftNeighbor ∧ g d = g rightNeighbor)

/-- Main theorem statement -/
theorem delegate_grouping_exists (s : SeatingArrangement) :
  ∃ g : Grouping,
    (∀ c : Fin 50, ∃! d : Delegate, g d = true ∧ d.country = c) ∧
    (∀ c : Fin 50, ∃! d : Delegate, g d = false ∧ d.country = c) ∧
    (∀ d : Delegate, atMostOneNeighborInGroup s g d) :=
  sorry

end delegate_grouping_exists_l3910_391088


namespace count_valid_selections_l3910_391032

/-- Represents a grid with subgrids -/
structure Grid (n : ℕ) where
  size : ℕ := n * n
  subgrid_size : ℕ := n
  num_subgrids : ℕ := n * n

/-- Represents a valid selection of cells from the grid -/
structure ValidSelection (n : ℕ) where
  grid : Grid n
  num_selected : ℕ := n * n
  one_per_subgrid : Bool
  one_per_row : Bool
  one_per_column : Bool

/-- The number of valid selections for a given grid size -/
def num_valid_selections (n : ℕ) : ℕ := (n.factorial ^ (n * n)) * ((n * n).factorial)

/-- Theorem stating the number of valid selections -/
theorem count_valid_selections (n : ℕ) :
  ∀ (selection : ValidSelection n),
    selection.one_per_subgrid ∧
    selection.one_per_row ∧
    selection.one_per_column →
    num_valid_selections n = (n.factorial ^ (n * n)) * ((n * n).factorial) :=
by sorry


end count_valid_selections_l3910_391032


namespace unique_square_property_l3910_391024

theorem unique_square_property (A : ℕ+) 
  (h1 : 100 ≤ A.val^2 ∧ A.val^2 < 1000)
  (h2 : ∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ A.val^2 = 100*x + 10*y + z)
  (h3 : ∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ x*100 + y*10 + z = A.val - 1) :
  A.val = 19 ∧ A.val^2 = 361 := by
sorry

end unique_square_property_l3910_391024


namespace ratio_simplification_l3910_391070

theorem ratio_simplification : 
  (10^2001 + 10^2003) / (10^2002 + 10^2002) = 101 / 20 := by sorry

end ratio_simplification_l3910_391070


namespace least_prime_factor_of_5_5_minus_5_4_l3910_391085

theorem least_prime_factor_of_5_5_minus_5_4 :
  Nat.minFac (5^5 - 5^4) = 2 := by
  sorry

end least_prime_factor_of_5_5_minus_5_4_l3910_391085


namespace unique_linear_equation_solution_l3910_391050

/-- A linear equation of the form y = kx + b -/
structure LinearEquation where
  k : ℝ
  b : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point satisfies the linear equation -/
def satisfiesEquation (p : Point) (eq : LinearEquation) : Prop :=
  p.y = eq.k * p.x + eq.b

theorem unique_linear_equation_solution (eq : LinearEquation) :
  satisfiesEquation ⟨1, 1⟩ eq → satisfiesEquation ⟨2, 3⟩ eq → eq.k = 2 ∧ eq.b = -1 := by
  sorry

#check unique_linear_equation_solution

end unique_linear_equation_solution_l3910_391050


namespace f_greater_than_g_l3910_391058

/-- Given two quadratic functions f and g, prove that f(x) > g(x) for all real x. -/
theorem f_greater_than_g : ∀ x : ℝ, (3 * x^2 - x + 1) > (2 * x^2 + x - 1) := by
  sorry

end f_greater_than_g_l3910_391058


namespace reconstruct_triangle_from_altitude_feet_l3910_391059

-- Define the basic types
def Point : Type := ℝ × ℝ

-- Define the triangle
structure Triangle :=
  (A B C : Point)

-- Define the orthocenter
def orthocenter (t : Triangle) : Point := sorry

-- Define the feet of altitudes
def altitude_foot (t : Triangle) (v : Point) : Point := sorry

-- Define an acute-angled triangle
def is_acute_angled (t : Triangle) : Prop := sorry

-- Define compass and straightedge constructibility
def constructible (p : Point) : Prop := sorry

-- Main theorem
theorem reconstruct_triangle_from_altitude_feet 
  (t : Triangle) 
  (h_acute : is_acute_angled t) 
  (A1 : Point) 
  (B1 : Point) 
  (C1 : Point) 
  (h_A1 : A1 = altitude_foot t t.A) 
  (h_B1 : B1 = altitude_foot t t.B) 
  (h_C1 : C1 = altitude_foot t t.C) :
  constructible t.A ∧ constructible t.B ∧ constructible t.C :=
sorry

end reconstruct_triangle_from_altitude_feet_l3910_391059


namespace constant_function_l3910_391075

theorem constant_function (t : ℝ) (f : ℝ → ℝ) 
  (h1 : f 0 = (1 : ℝ) / 2)
  (h2 : ∀ x y : ℝ, f (x + y) = f x * f (t - y) + f y * f (t - x)) :
  ∀ x : ℝ, f x = (1 : ℝ) / 2 :=
by sorry

end constant_function_l3910_391075


namespace cubic_root_relation_l3910_391004

theorem cubic_root_relation (a b c d : ℝ) (h : a ≠ 0) :
  (∃ u v w : ℝ, a * u^3 + b * u^2 + c * u + d = 0 ∧
               a * v^3 + b * v^2 + c * v + d = 0 ∧
               a * w^3 + b * w^2 + c * w + d = 0 ∧
               u + v = u * v) →
  (c + d) * (b + c + d) = a * d :=
by sorry

end cubic_root_relation_l3910_391004


namespace population_growth_theorem_l3910_391000

/-- The annual population growth rate due to natural growth -/
def natural_growth_rate : ℝ := 0.06

/-- The overall population growth rate over 3 years -/
def total_growth_rate : ℝ := 0.157625

/-- The annual population decrease rate due to migration -/
def migration_decrease_rate : ℝ := 0.009434

theorem population_growth_theorem :
  ∃ (x : ℝ),
    (((1 + natural_growth_rate) * (1 - x))^3 = 1 + total_growth_rate) ∧
    (abs (x - migration_decrease_rate) < 0.00001) := by
  sorry

end population_growth_theorem_l3910_391000


namespace investment_amounts_l3910_391034

/-- Represents the investment and profit ratios for three investors over two years -/
structure InvestmentData where
  p_investment_year1 : ℚ
  p_investment_year2 : ℚ
  profit_ratio_year1 : Fin 3 → ℚ
  profit_ratio_year2 : Fin 3 → ℚ

/-- Calculates the investments of q and r based on the given data -/
def calculate_investments (data : InvestmentData) : ℚ × ℚ :=
  let q_investment := (data.profit_ratio_year1 1 / data.profit_ratio_year1 0) * data.p_investment_year1
  let r_investment := (data.profit_ratio_year1 2 / data.profit_ratio_year1 0) * data.p_investment_year1
  (q_investment, r_investment)

/-- The main theorem stating the investment amounts for q and r -/
theorem investment_amounts (data : InvestmentData)
  (h1 : data.p_investment_year1 = 52000)
  (h2 : data.p_investment_year2 = 62400)
  (h3 : data.profit_ratio_year1 = ![4, 5, 6])
  (h4 : data.profit_ratio_year2 = ![3, 4, 5])
  (h5 : data.p_investment_year2 = data.p_investment_year1 * (1 + 1/5)) :
  calculate_investments data = (65000, 78000) := by
  sorry

end investment_amounts_l3910_391034


namespace quadratic_function_range_l3910_391054

/-- A quadratic function f(x) = x^2 - 2bx + b^2 + b - 5 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + b^2 + b - 5

/-- The derivative of f with respect to x -/
def f_derivative (b : ℝ) (x : ℝ) : ℝ := 2*x - 2*b

theorem quadratic_function_range (b : ℝ) :
  (∃ x, f b x = 0) ∧ (∀ x < (3.5 : ℝ), f_derivative b x < 0) →
  3.5 ≤ b ∧ b ≤ 5 :=
sorry

end quadratic_function_range_l3910_391054


namespace least_possible_bc_length_l3910_391072

theorem least_possible_bc_length (AB AC DC BD BC : ℝ) : 
  AB = 7 → AC = 15 → DC = 10 → BD = 24 → 
  BC > AC - AB → BC > BD - DC → 
  (∃ (n : ℕ), BC = n ∧ ∀ (m : ℕ), BC ≥ m → n ≤ m) → 
  BC ≥ 14 := by sorry

end least_possible_bc_length_l3910_391072


namespace square_difference_72_24_l3910_391009

theorem square_difference_72_24 : 72^2 - 24^2 = 4608 := by
  sorry

end square_difference_72_24_l3910_391009


namespace unique_positive_solution_l3910_391099

theorem unique_positive_solution (y : ℝ) (h_pos : y > 0) 
  (h_eq : Real.sqrt (12 * y) * Real.sqrt (6 * y) * Real.sqrt (18 * y) * Real.sqrt (9 * y) = 27) : 
  y = (1 : ℝ) / 2 := by
sorry

end unique_positive_solution_l3910_391099


namespace tan_sum_product_equals_one_l3910_391035

theorem tan_sum_product_equals_one :
  ∀ (a b : ℝ),
  a + b = Real.pi / 4 →
  Real.tan (Real.pi / 4) = 1 →
  Real.tan a + Real.tan b + Real.tan a * Real.tan b = 1 := by
  sorry

end tan_sum_product_equals_one_l3910_391035


namespace intersection_point_sum_l3910_391006

theorem intersection_point_sum (a b : ℝ) :
  (2 = (1/3) * 1 + a) ∧ (1 = (1/3) * 2 + b) → a + b = 2 := by
  sorry

end intersection_point_sum_l3910_391006


namespace excel_in_both_subjects_l3910_391068

theorem excel_in_both_subjects
  (total_students : ℕ)
  (excel_chinese : ℕ)
  (excel_math : ℕ)
  (h1 : total_students = 45)
  (h2 : excel_chinese = 34)
  (h3 : excel_math = 39)
  (h4 : ∀ s, s ≤ total_students → (s ≤ excel_chinese ∨ s ≤ excel_math)) :
  excel_chinese + excel_math - total_students = 28 := by
  sorry

end excel_in_both_subjects_l3910_391068


namespace product_of_repeating_decimals_l3910_391087

-- Define the repeating decimals
def repeating_038 : ℚ := 38 / 999
def repeating_4 : ℚ := 4 / 9

-- State the theorem
theorem product_of_repeating_decimals :
  repeating_038 * repeating_4 = 152 / 8991 := by
  sorry

end product_of_repeating_decimals_l3910_391087


namespace action_figure_pricing_theorem_l3910_391069

theorem action_figure_pricing_theorem :
  ∃ (x y z w : ℝ),
    -- Total money from selling action figures equals required amount
    12 * x + 8 * y + 5 * z + 10 * w = 220 ∧
    -- Price ratios
    x / 4 = y / 3 ∧
    x / 4 = z / 2 ∧
    x / 4 = w / 1 ∧
    -- Prices are positive
    x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 :=
by sorry

end action_figure_pricing_theorem_l3910_391069


namespace mean_height_correct_l3910_391014

/-- The number of players on the basketball team -/
def num_players : ℕ := 16

/-- The total height of all players in inches -/
def total_height : ℕ := 965

/-- The mean height of the players -/
def mean_height : ℚ := 60.31

/-- Theorem stating that the mean height is correct given the number of players and total height -/
theorem mean_height_correct : 
  (total_height : ℚ) / (num_players : ℚ) = mean_height := by sorry

end mean_height_correct_l3910_391014


namespace no_nonzero_integer_solutions_l3910_391055

theorem no_nonzero_integer_solutions :
  ∀ x y z : ℤ, x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 :=
by sorry

end no_nonzero_integer_solutions_l3910_391055


namespace equation_solution_l3910_391008

theorem equation_solution (x y : ℝ) : 
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ (y = -x - 2 ∨ y = -2 * x + 1) := by
sorry

end equation_solution_l3910_391008


namespace cookie_count_l3910_391011

/-- The number of edible cookies at the end of Alice and Bob's baking session -/
def total_edible_cookies (alice_initial : ℕ) (bob_initial : ℕ) (thrown_away : ℕ) (alice_additional : ℕ) (bob_additional : ℕ) : ℕ :=
  alice_initial + bob_initial - thrown_away + alice_additional + bob_additional

/-- Theorem stating the total number of edible cookies at the end -/
theorem cookie_count : total_edible_cookies 74 7 29 5 36 = 93 := by
  sorry

end cookie_count_l3910_391011


namespace sufficient_not_necessary_l3910_391012

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

-- Define the condition p
def condition_p (m : ℝ) : Prop :=
  ∀ x₁ x₂, 1/2 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂

-- Define the condition q
def condition_q (m : ℝ) : Prop := m ≥ -4/3

-- Theorem statement
theorem sufficient_not_necessary :
  (∃ m, condition_p m ∧ ¬condition_q m) ∧
  (∀ m, condition_p m → condition_q m) :=
by sorry

end sufficient_not_necessary_l3910_391012


namespace barry_pretzels_l3910_391045

/-- Given the following conditions about pretzel purchases:
  - Angie bought three times as many pretzels as Shelly
  - Shelly bought half as many pretzels as Barry
  - Angie bought 18 pretzels
Prove that Barry bought 12 pretzels. -/
theorem barry_pretzels (angie shelly barry : ℕ) 
  (h1 : angie = 3 * shelly) 
  (h2 : shelly = barry / 2) 
  (h3 : angie = 18) : 
  barry = 12 := by
  sorry

end barry_pretzels_l3910_391045


namespace min_value_x_plus_2y_l3910_391093

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + 2*x*y = 3) :
  x + 2*y ≥ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ + 2*x₀*y₀ = 3 ∧ x₀ + 2*y₀ = 2 :=
sorry

end min_value_x_plus_2y_l3910_391093


namespace f_derivative_negative_solution_set_l3910_391092

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

-- State the theorem
theorem f_derivative_negative_solution_set :
  {x : ℝ | (deriv f) x < 0} = Set.Ioo (-1) 3 := by sorry

end f_derivative_negative_solution_set_l3910_391092


namespace jan_miles_difference_l3910_391027

-- Define variables
variable (t s : ℝ)

-- Define distances
def ian_distance : ℝ := s * t
def han_distance : ℝ := (s + 10) * (t + 2)
def jan_distance : ℝ := (s + 15) * (t + 3)

-- State the theorem
theorem jan_miles_difference :
  han_distance t s = ian_distance t s + 120 →
  jan_distance t s = ian_distance t s + 195 := by
  sorry

end jan_miles_difference_l3910_391027
