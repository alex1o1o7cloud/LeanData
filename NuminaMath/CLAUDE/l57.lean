import Mathlib

namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l57_5720

theorem fixed_point_of_exponential_function (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 4
  f 1 = 5 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l57_5720


namespace NUMINAMATH_CALUDE_successfully_served_pizzas_l57_5751

def pizzas_served : ℕ := 9
def pizzas_returned : ℕ := 6

theorem successfully_served_pizzas : 
  pizzas_served - pizzas_returned = 3 := by sorry

end NUMINAMATH_CALUDE_successfully_served_pizzas_l57_5751


namespace NUMINAMATH_CALUDE_vertex_on_x_axis_l57_5706

/-- The parabola function -/
def f (d : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + d

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 3

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y (d : ℝ) : ℝ := f d vertex_x

/-- The theorem stating that the vertex lies on the x-axis iff d = 9 -/
theorem vertex_on_x_axis (d : ℝ) : vertex_y d = 0 ↔ d = 9 := by sorry

end NUMINAMATH_CALUDE_vertex_on_x_axis_l57_5706


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l57_5711

theorem quadratic_root_sum (p q : ℝ) : 
  (∃ x : ℂ, x^2 + p*x + q = 0 ∧ x = 1 + Complex.I) → p + q = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l57_5711


namespace NUMINAMATH_CALUDE_max_bottles_from_C_and_D_l57_5789

/-- Represents the shops selling recyclable bottles -/
inductive Shop
| A
| B
| C
| D

/-- The price of a bottle at each shop -/
def price (s : Shop) : ℕ :=
  match s with
  | Shop.A => 1
  | Shop.B => 2
  | Shop.C => 3
  | Shop.D => 5

/-- Don's initial budget -/
def initial_budget : ℕ := 600

/-- Number of bottles Don buys from Shop A -/
def bottles_from_A : ℕ := 150

/-- Number of bottles Don buys from Shop B -/
def bottles_from_B : ℕ := 180

/-- The remaining budget after buying from shops A and B -/
def remaining_budget : ℕ := 
  initial_budget - (bottles_from_A * price Shop.A + bottles_from_B * price Shop.B)

/-- The theorem stating the maximum number of bottles Don can buy from shops C and D combined -/
theorem max_bottles_from_C_and_D : 
  (remaining_budget / price Shop.C) = 30 := by sorry

end NUMINAMATH_CALUDE_max_bottles_from_C_and_D_l57_5789


namespace NUMINAMATH_CALUDE_hemisphere_with_spire_surface_area_l57_5797

/-- The total surface area of a hemisphere with a conical spire -/
theorem hemisphere_with_spire_surface_area :
  let r : ℝ := 8  -- radius of hemisphere
  let h : ℝ := 10 -- height of conical spire
  let l : ℝ := Real.sqrt (r^2 + h^2)  -- slant height of cone
  let area_base : ℝ := π * r^2  -- area of circular base
  let area_hemisphere : ℝ := 2 * π * r^2  -- surface area of hemisphere
  let area_cone : ℝ := π * r * l  -- lateral surface area of cone
  area_base + area_hemisphere + area_cone = 192 * π + 8 * π * Real.sqrt 164 :=
by sorry


end NUMINAMATH_CALUDE_hemisphere_with_spire_surface_area_l57_5797


namespace NUMINAMATH_CALUDE_razorback_shop_profit_l57_5734

/-- The amount the shop makes off each jersey in dollars. -/
def jersey_profit : ℕ := 34

/-- The additional cost of a t-shirt compared to a jersey in dollars. -/
def tshirt_additional_cost : ℕ := 158

/-- The amount the shop makes off each t-shirt in dollars. -/
def tshirt_profit : ℕ := jersey_profit + tshirt_additional_cost

theorem razorback_shop_profit : tshirt_profit = 192 := by
  sorry

end NUMINAMATH_CALUDE_razorback_shop_profit_l57_5734


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l57_5798

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2023 + 3) :
  (3 / (x + 3) - 1) / (x / (x^2 - 9)) = -Real.sqrt 2023 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l57_5798


namespace NUMINAMATH_CALUDE_probability_selection_l57_5787

def research_team (total : ℝ) : Prop :=
  let women := 0.75 * total
  let men := 0.25 * total
  let women_lawyers := 0.60 * women
  let women_engineers := 0.25 * women
  let women_doctors := 0.15 * women
  let men_lawyers := 0.40 * men
  let men_engineers := 0.35 * men
  let men_doctors := 0.25 * men
  (women + men = total) ∧
  (women_lawyers + women_engineers + women_doctors = women) ∧
  (men_lawyers + men_engineers + men_doctors = men)

theorem probability_selection (total : ℝ) (h : research_team total) :
  (0.75 * 0.60 * total + 0.75 * 0.25 * total + 0.25 * 0.25 * total) / total = 0.70 :=
by sorry

end NUMINAMATH_CALUDE_probability_selection_l57_5787


namespace NUMINAMATH_CALUDE_range_of_P_l57_5702

theorem range_of_P (x y : ℝ) (h : x^2/3 + y^2 = 1) :
  2 ≤ |2*x + y - 4| + |4 - x - 2*y| ∧ |2*x + y - 4| + |4 - x - 2*y| ≤ 14 := by
  sorry

end NUMINAMATH_CALUDE_range_of_P_l57_5702


namespace NUMINAMATH_CALUDE_equation_roots_l57_5792

theorem equation_roots : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 4 ∧ x₂ = -2.5) ∧ 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 → (18 / (x^2 - 4) - 3 / (x - 2) = 2 ↔ (x = x₁ ∨ x = x₂))) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l57_5792


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l57_5745

theorem polynomial_division_remainder (x : ℝ) : 
  ∃ (Q : ℝ → ℝ), x^150 = (x^2 - 4*x + 3) * Q x + ((3^150 - 1)*x + (4 - 3^150)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l57_5745


namespace NUMINAMATH_CALUDE_line_with_45_degree_slope_l57_5757

/-- Given a line passing through points (1, -2) and (a, 3) with a slope angle of 45°, 
    the value of a is 6. -/
theorem line_with_45_degree_slope (a : ℝ) : 
  (((3 - (-2)) / (a - 1) = Real.tan (π / 4)) → a = 6) :=
by sorry

end NUMINAMATH_CALUDE_line_with_45_degree_slope_l57_5757


namespace NUMINAMATH_CALUDE_perimeter_of_cut_square_perimeter_of_specific_cut_square_l57_5781

/-- The perimeter of a figure formed by cutting a square into two equal rectangles and placing them side by side -/
theorem perimeter_of_cut_square (side_length : ℝ) : 
  side_length > 0 → 
  (3 * side_length + 4 * (side_length / 2)) = 5 * side_length := by
  sorry

/-- The perimeter of a figure formed by cutting a square with side length 100 into two equal rectangles and placing them side by side is 500 -/
theorem perimeter_of_specific_cut_square : 
  (3 * 100 + 4 * (100 / 2)) = 500 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_cut_square_perimeter_of_specific_cut_square_l57_5781


namespace NUMINAMATH_CALUDE_point_P_y_coordinate_l57_5749

theorem point_P_y_coordinate :
  ∀ (x y : ℝ),
  (|y| = (1/2) * |x|) →  -- Distance from x-axis is half the distance from y-axis
  (|x| = 18) →           -- Point P is 18 units from y-axis
  y = 9 := by            -- The y-coordinate of point P is 9
sorry

end NUMINAMATH_CALUDE_point_P_y_coordinate_l57_5749


namespace NUMINAMATH_CALUDE_four_number_sequence_l57_5741

theorem four_number_sequence (a b c d : ℝ) 
  (h1 : b^2 = a*c)
  (h2 : a*b*c = 216)
  (h3 : 2*c = b + d)
  (h4 : b + c + d = 12) :
  a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2 := by
sorry

end NUMINAMATH_CALUDE_four_number_sequence_l57_5741


namespace NUMINAMATH_CALUDE_log_equation_solution_l57_5726

theorem log_equation_solution (y : ℝ) (h : y > 0) :
  Real.log y / Real.log 3 + Real.log y / Real.log 9 = 5 → y = 3^(10/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l57_5726


namespace NUMINAMATH_CALUDE_base5_conversion_and_modulo_l57_5744

/-- Converts a base 5 number to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Computes the modulo of a number -/
def modulo (n m : Nat) : Nat :=
  n % m

theorem base5_conversion_and_modulo :
  let base5Num : List Nat := [4, 1, 0, 1, 2]  -- 21014 in base 5, least significant digit first
  let base10Num : Nat := base5ToBase10 base5Num
  base10Num = 1384 ∧ modulo base10Num 7 = 6 := by
  sorry

#eval base5ToBase10 [4, 1, 0, 1, 2]  -- Should output 1384
#eval modulo 1384 7  -- Should output 6

end NUMINAMATH_CALUDE_base5_conversion_and_modulo_l57_5744


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_over_four_l57_5786

theorem tan_alpha_plus_pi_over_four (α : Real) (h : Real.tan α = 2) :
  Real.tan (α + π / 4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_over_four_l57_5786


namespace NUMINAMATH_CALUDE_complex_power_equality_l57_5730

theorem complex_power_equality : (1 - Complex.I) ^ (2 * Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_equality_l57_5730


namespace NUMINAMATH_CALUDE_complement_of_B_in_A_l57_5709

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 3, 5}

theorem complement_of_B_in_A : (A \ B) = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_B_in_A_l57_5709


namespace NUMINAMATH_CALUDE_linear_function_proof_l57_5771

def f (x : ℝ) := -3 * x + 5

theorem linear_function_proof :
  (∀ x y : ℝ, f y - f x = -3 * (y - x)) ∧ 
  (∃ y : ℝ, f 0 = 3 * 0 + 5 ∧ f 0 = y) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_proof_l57_5771


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l57_5776

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

/-- The x-intercept of the parabola -/
def x_intercept : ℝ := parabola 0

/-- The y-intercepts of the parabola -/
noncomputable def y_intercepts : Set ℝ := {y | parabola y = 0}

theorem parabola_intercepts_sum :
  ∃ (b c : ℝ), b ∈ y_intercepts ∧ c ∈ y_intercepts ∧ b ≠ c ∧ x_intercept + b + c = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l57_5776


namespace NUMINAMATH_CALUDE_log_equation_solution_l57_5735

theorem log_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.log x + Real.log (x + 4) = Real.log (2 * x + 8) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l57_5735


namespace NUMINAMATH_CALUDE_system_solution_equation_solution_l57_5780

-- Part 1: System of equations
theorem system_solution :
  ∃! (x y : ℝ), (2 * x - y = 3) ∧ (x + y = -12) ∧ (x = -3) ∧ (y = -9) := by
  sorry

-- Part 2: Single equation
theorem equation_solution :
  ∃! x : ℝ, (2 / (1 - x) + 1 = x / (1 + x)) ∧ (x = -3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_equation_solution_l57_5780


namespace NUMINAMATH_CALUDE_jose_peanuts_l57_5740

theorem jose_peanuts (kenya_peanuts : ℕ) (difference : ℕ) (h1 : kenya_peanuts = 133) (h2 : difference = 48) :
  kenya_peanuts - difference = 85 := by
  sorry

end NUMINAMATH_CALUDE_jose_peanuts_l57_5740


namespace NUMINAMATH_CALUDE_order_of_expressions_l57_5742

theorem order_of_expressions (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  let a := Real.sqrt ((x^2 + y^2) / 2) - (x + y) / 2
  let b := (x + y) / 2 - Real.sqrt (x * y)
  let c := Real.sqrt (x * y) - 2 / (1 / x + 1 / y)
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_order_of_expressions_l57_5742


namespace NUMINAMATH_CALUDE_second_red_ball_most_likely_l57_5743

/-- The total number of balls in the urn -/
def total_balls : ℕ := 101

/-- The number of red balls in the urn -/
def red_balls : ℕ := 3

/-- The probability of drawing the second red ball on the kth draw -/
def prob_second_red (k : ℕ) : ℚ :=
  if 1 < k ∧ k < total_balls
  then (k - 1 : ℚ) * (total_balls - k : ℚ) / (total_balls.choose red_balls : ℚ)
  else 0

/-- The draw number that maximizes the probability of drawing the second red ball -/
def max_prob_draw : ℕ := 51

theorem second_red_ball_most_likely :
  ∀ k, prob_second_red max_prob_draw ≥ prob_second_red k :=
sorry

end NUMINAMATH_CALUDE_second_red_ball_most_likely_l57_5743


namespace NUMINAMATH_CALUDE_greatest_possible_median_l57_5783

theorem greatest_possible_median (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 18 →
  k < m → m < r → r < s → s < t →
  t = 40 →
  r ≤ 23 ∧ ∃ (k' m' r' s' : ℕ), 
    k' > 0 ∧ m' > 0 ∧ r' > 0 ∧ s' > 0 ∧
    (k' + m' + r' + s' + 40) / 5 = 18 ∧
    k' < m' ∧ m' < r' ∧ r' < s' ∧ s' < 40 ∧
    r' = 23 :=
by sorry

end NUMINAMATH_CALUDE_greatest_possible_median_l57_5783


namespace NUMINAMATH_CALUDE_incorrect_permutations_hello_l57_5723

def word := "hello"

theorem incorrect_permutations_hello :
  let total_letters := word.length
  let duplicate_letters := 2  -- number of 'l's
  let total_permutations := Nat.factorial total_letters
  let unique_permutations := total_permutations / (Nat.factorial duplicate_letters)
  unique_permutations - 1 = 59 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_permutations_hello_l57_5723


namespace NUMINAMATH_CALUDE_circle_center_in_second_quadrant_l57_5758

/-- A line passing through the second, third, and fourth quadrants -/
structure Line where
  a : ℝ
  b : ℝ
  second_quadrant : a < 0 ∧ 0 < a * 0 - b
  third_quadrant : a * (-1) - b < 0
  fourth_quadrant : 0 < a * 1 - b

/-- The center of a circle (x-a)^2 + (y-b)^2 = 1 -/
def circle_center (l : Line) : ℝ × ℝ := (l.a, l.b)

/-- A point is in the second quadrant if its x-coordinate is negative and y-coordinate is positive -/
def in_second_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ 0 < p.2

theorem circle_center_in_second_quadrant (l : Line) :
  in_second_quadrant (circle_center l) := by sorry

end NUMINAMATH_CALUDE_circle_center_in_second_quadrant_l57_5758


namespace NUMINAMATH_CALUDE_inequality_proof_l57_5772

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≥ 1) :
  (a + 2 * b + 2 / (a + 1)) * (b + 2 * a + 2 / (b + 1)) ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l57_5772


namespace NUMINAMATH_CALUDE_absolute_value_condition_l57_5705

theorem absolute_value_condition (x : ℝ) :
  (∀ x, x < -2 → |x| > 2) ∧ 
  (∃ x, |x| > 2 ∧ x ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_condition_l57_5705


namespace NUMINAMATH_CALUDE_function_inequality_l57_5738

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h1 : ∀ x, (x - 3) * deriv f x ≤ 0) : 
  f 0 + f 6 ≤ 2 * f 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l57_5738


namespace NUMINAMATH_CALUDE_emily_coloring_books_l57_5739

theorem emily_coloring_books (x : ℕ) : 
  x - 2 + 14 = 19 → x = 7 := by
sorry

end NUMINAMATH_CALUDE_emily_coloring_books_l57_5739


namespace NUMINAMATH_CALUDE_book_price_percentage_l57_5755

/-- The percentage of the suggested retail price that Bob paid for a book -/
theorem book_price_percentage (suggested_retail_price : ℝ) : 
  suggested_retail_price > 0 →
  let marked_price := 0.6 * suggested_retail_price
  let bob_paid := 0.6 * marked_price
  bob_paid / suggested_retail_price = 0.36 :=
by sorry

end NUMINAMATH_CALUDE_book_price_percentage_l57_5755


namespace NUMINAMATH_CALUDE_min_value_at_three_l57_5795

/-- The function f(y) = 3y^2 - 18y + 7 -/
def f (y : ℝ) : ℝ := 3 * y^2 - 18 * y + 7

/-- Theorem stating that the minimum value of f occurs when y = 3 -/
theorem min_value_at_three :
  ∃ (y_min : ℝ), ∀ (y : ℝ), f y ≥ f y_min ∧ y_min = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_at_three_l57_5795


namespace NUMINAMATH_CALUDE_dallas_pears_count_l57_5770

/-- The number of bags of pears Dallas picked -/
def dallas_pears : ℕ := 9

/-- The number of bags of apples Dallas picked -/
def dallas_apples : ℕ := 14

/-- The total number of bags Austin picked -/
def austin_total : ℕ := 24

theorem dallas_pears_count :
  dallas_pears = 9 ∧
  dallas_apples = 14 ∧
  austin_total = 24 ∧
  austin_total = (dallas_apples + 6) + (dallas_pears - 5) :=
by sorry

end NUMINAMATH_CALUDE_dallas_pears_count_l57_5770


namespace NUMINAMATH_CALUDE_min_games_for_equal_play_l57_5774

/-- Represents a bridge game between 2 guys and 2 girls -/
structure BridgeGame where
  guys : Fin 2 → Fin 5
  girls : Fin 2 → Fin 5

/-- The minimum number of games required for the given conditions -/
def minGames : Nat := 25

/-- Checks if a set of games satisfies the equal play condition -/
def satisfiesEqualPlay (games : List BridgeGame) : Prop :=
  ∀ (guy : Fin 5) (girl : Fin 5),
    (games.filter (λ g => g.guys 0 = guy ∨ g.guys 1 = guy)).length =
    (games.filter (λ g => g.girls 0 = girl ∨ g.girls 1 = girl)).length

theorem min_games_for_equal_play :
  ∀ (games : List BridgeGame),
    satisfiesEqualPlay games →
    games.length ≥ minGames :=
  sorry

end NUMINAMATH_CALUDE_min_games_for_equal_play_l57_5774


namespace NUMINAMATH_CALUDE_pauls_lost_crayons_l57_5718

/-- Given Paul's crayon situation, prove the number of lost crayons --/
theorem pauls_lost_crayons
  (initial_crayons : ℕ)
  (given_to_friends : ℕ)
  (total_lost_or_given : ℕ)
  (h1 : initial_crayons = 65)
  (h2 : given_to_friends = 213)
  (h3 : total_lost_or_given = 229)
  : total_lost_or_given - given_to_friends = 16 := by
  sorry

end NUMINAMATH_CALUDE_pauls_lost_crayons_l57_5718


namespace NUMINAMATH_CALUDE_open_box_volume_l57_5707

/-- The volume of an open box formed by cutting squares from a rectangular sheet -/
theorem open_box_volume (sheet_length sheet_width cut_size : ℝ)
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_size = 7)
  (h4 : sheet_length > 2 * cut_size)
  (h5 : sheet_width > 2 * cut_size) :
  (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size = 5244 :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l57_5707


namespace NUMINAMATH_CALUDE_alia_markers_l57_5704

theorem alia_markers (steve_markers : ℕ) (austin_markers : ℕ) (alia_markers : ℕ)
  (h1 : steve_markers = 60)
  (h2 : austin_markers = steve_markers / 3)
  (h3 : alia_markers = 2 * austin_markers) :
  alia_markers = 40 := by
sorry

end NUMINAMATH_CALUDE_alia_markers_l57_5704


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l57_5750

/-- A hexagon with vertices N, U, M, B, E, S -/
structure Hexagon :=
  (N U M B E S : ℝ)

/-- The property that three angles are congruent -/
def three_angles_congruent (h : Hexagon) : Prop :=
  h.N = h.M ∧ h.M = h.B

/-- The property that two angles are supplementary -/
def supplementary (a b : ℝ) : Prop :=
  a + b = 180

/-- The theorem stating that in a hexagon NUMBERS where ∠N ≅ ∠M ≅ ∠B 
    and ∠U is supplementary to ∠S, the measure of ∠B is 135° -/
theorem hexagon_angle_measure (h : Hexagon) 
  (h_congruent : three_angles_congruent h)
  (h_supplementary : supplementary h.U h.S) :
  h.B = 135 :=
sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l57_5750


namespace NUMINAMATH_CALUDE_gcd_98_63_l57_5754

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_63_l57_5754


namespace NUMINAMATH_CALUDE_different_genre_pairs_count_l57_5748

/-- Represents the number of books in each genre -/
structure BookCollection where
  mystery : Nat
  fantasy : Nat
  biography : Nat

/-- Calculates the number of possible pairs of books from different genres -/
def differentGenrePairs (books : BookCollection) : Nat :=
  books.mystery * books.fantasy +
  books.mystery * books.biography +
  books.fantasy * books.biography

/-- Theorem: Given 4 mystery novels, 3 fantasy novels, and 2 biographies,
    the number of possible pairs of books from different genres is 26 -/
theorem different_genre_pairs_count :
  differentGenrePairs ⟨4, 3, 2⟩ = 26 := by
  sorry

end NUMINAMATH_CALUDE_different_genre_pairs_count_l57_5748


namespace NUMINAMATH_CALUDE_tangent_lines_theorem_intersection_theorem_l57_5778

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Define the point M
def point_M : ℝ × ℝ := (3, 1)

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = 3
def tangent_line_2 (x y : ℝ) : Prop := 3*x - 4*y - 5 = 0

-- Define the family of lines
def line_family (a x y : ℝ) : Prop := a*x - y + 3 = 0

-- Theorem for tangent lines
theorem tangent_lines_theorem :
  (∀ x y : ℝ, tangent_line_1 x → circle_C x y → x = 3 ∧ y = 1 ∨ x = 3 ∧ y = 3) ∧
  (∀ x y : ℝ, tangent_line_2 x y → circle_C x y → x = 3 ∧ y = 1 ∨ x = 0 ∧ y = 5/4) :=
sorry

-- Theorem for intersection
theorem intersection_theorem :
  ∀ a : ℝ, ∃ x y : ℝ, line_family a x y ∧ circle_C x y :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_theorem_intersection_theorem_l57_5778


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l57_5766

/-- A quadratic equation with parameter m -/
def quadratic_equation (m : ℤ) (x : ℤ) : Prop :=
  m * x^2 - (m + 1) * x + 1 = 0

/-- The property that the equation has two distinct integer roots -/
def has_two_distinct_integer_roots (m : ℤ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y

theorem quadratic_equation_solution :
  ∀ m : ℤ, has_two_distinct_integer_roots m → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l57_5766


namespace NUMINAMATH_CALUDE_infinite_special_numbers_l57_5719

theorem infinite_special_numbers :
  ∃ (seq : ℕ → ℕ), 
    (∀ i, ∃ n, seq i = n) ∧
    (∀ i j, i < j → seq i < seq j) ∧
    (∀ i, ∀ p : ℕ, Prime p → p ∣ (seq i)^2 + 3 →
      ∃ k : ℕ, k^2 < seq i ∧ p ∣ k^2 + 3) :=
by sorry

end NUMINAMATH_CALUDE_infinite_special_numbers_l57_5719


namespace NUMINAMATH_CALUDE_vectors_coplanar_iff_x_eq_five_l57_5724

/-- Given vectors a, b, and c in ℝ³, prove that they are coplanar if and only if x = 5 -/
theorem vectors_coplanar_iff_x_eq_five (a b c : ℝ × ℝ × ℝ) :
  a = (1, -1, 3) →
  b = (-1, 4, -2) →
  c = (1, 5, x) →
  (∃ (m n : ℝ), c = m • a + n • b) ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_vectors_coplanar_iff_x_eq_five_l57_5724


namespace NUMINAMATH_CALUDE_marble_fraction_l57_5799

theorem marble_fraction (total : ℝ) (h : total > 0) : 
  let initial_blue := (2/3) * total
  let initial_red := total - initial_blue
  let new_blue := 2 * initial_blue
  let new_red := 3 * initial_red
  let new_total := new_blue + new_red
  new_red / new_total = 3/7 := by sorry

end NUMINAMATH_CALUDE_marble_fraction_l57_5799


namespace NUMINAMATH_CALUDE_number_count_l57_5722

theorem number_count (total_average : ℝ) (first_six_average : ℝ) (last_six_average : ℝ) (middle_number : ℝ) :
  total_average = 9.9 →
  first_six_average = 10.5 →
  last_six_average = 11.4 →
  middle_number = 22.5 →
  ∃ (n : ℕ), n = 11 ∧ n % 2 = 1 ∧
  n * total_average = 6 * first_six_average + 6 * last_six_average - middle_number :=
by sorry


end NUMINAMATH_CALUDE_number_count_l57_5722


namespace NUMINAMATH_CALUDE_joe_weight_lifting_l57_5760

theorem joe_weight_lifting (first_lift second_lift : ℕ) 
  (h1 : first_lift + second_lift = 1500)
  (h2 : 2 * first_lift = second_lift + 300) :
  first_lift = 600 := by
sorry

end NUMINAMATH_CALUDE_joe_weight_lifting_l57_5760


namespace NUMINAMATH_CALUDE_apple_basket_count_apple_basket_theorem_l57_5737

theorem apple_basket_count : ℕ → Prop :=
  fun total_apples =>
    (total_apples : ℚ) * (12 : ℚ) / 100 + 66 = total_apples ∧ total_apples = 75

-- Proof
theorem apple_basket_theorem : ∃ n : ℕ, apple_basket_count n := by
  sorry

end NUMINAMATH_CALUDE_apple_basket_count_apple_basket_theorem_l57_5737


namespace NUMINAMATH_CALUDE_some_number_less_than_two_l57_5700

theorem some_number_less_than_two (x y : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : -2 < x ∧ x < 9)
  (h4 : 0 < x ∧ x < 8)
  (h5 : x + y < 9)
  (h6 : x = 7) : 
  y < 2 := by
sorry

end NUMINAMATH_CALUDE_some_number_less_than_two_l57_5700


namespace NUMINAMATH_CALUDE_equation1_solutions_equation2_solutions_l57_5714

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * x * (x + 3) = 2 * (x + 3)
def equation2 (x : ℝ) : Prop := x^2 - 4*x - 5 = 0

-- Theorem for equation 1
theorem equation1_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ = -3 ∧ x₂ = 2/3 ∧ equation1 x₁ ∧ equation1 x₂ ∧
  ∀ (x : ℝ), equation1 x → x = x₁ ∨ x = x₂ := by sorry

-- Theorem for equation 2
theorem equation2_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ = 5 ∧ x₂ = -1 ∧ equation2 x₁ ∧ equation2 x₂ ∧
  ∀ (x : ℝ), equation2 x → x = x₁ ∨ x = x₂ := by sorry

end NUMINAMATH_CALUDE_equation1_solutions_equation2_solutions_l57_5714


namespace NUMINAMATH_CALUDE_five_sixteenths_decimal_l57_5731

theorem five_sixteenths_decimal : (5 : ℚ) / 16 = 0.3125 := by
  sorry

end NUMINAMATH_CALUDE_five_sixteenths_decimal_l57_5731


namespace NUMINAMATH_CALUDE_find_number_l57_5716

theorem find_number : ∃ x : ℕ, x * 99999 = 65818408915 ∧ x = 658185 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l57_5716


namespace NUMINAMATH_CALUDE_fib_100_mod_5_l57_5710

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_100_mod_5 : fib 100 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_100_mod_5_l57_5710


namespace NUMINAMATH_CALUDE_min_squares_15_step_staircase_l57_5773

/-- Represents a staircase with a given number of steps -/
structure Staircase :=
  (steps : ℕ)

/-- The minimum number of squares required to cover a staircase -/
def min_squares_to_cover (s : Staircase) : ℕ := s.steps

/-- Theorem: The minimum number of squares required to cover a 15-step staircase is 15 -/
theorem min_squares_15_step_staircase :
  ∀ (s : Staircase), s.steps = 15 → min_squares_to_cover s = 15 := by
  sorry

/-- Lemma: Cutting can only be done along the boundaries of the cells -/
lemma cut_along_boundaries (s : Staircase) : True := by
  sorry

/-- Lemma: Each step in the staircase forms a unit square -/
lemma step_is_unit_square (s : Staircase) : True := by
  sorry

end NUMINAMATH_CALUDE_min_squares_15_step_staircase_l57_5773


namespace NUMINAMATH_CALUDE_fraction_multiplication_l57_5713

theorem fraction_multiplication : (2 : ℚ) / 5 * 5 / 7 * 7 / 3 * 3 / 8 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l57_5713


namespace NUMINAMATH_CALUDE_total_stones_in_five_piles_l57_5756

/-- Given five piles of stones with the following properties:
    1. The number of stones in the fifth pile is six times the number of stones in the third pile
    2. The number of stones in the second pile is twice the total number of stones in the third and fifth piles combined
    3. The number of stones in the first pile is three times less than the number in the fifth pile and 10 less than the number in the fourth pile
    4. The number of stones in the fourth pile is half the number in the second pile
    Prove that the total number of stones in all five piles is 60. -/
theorem total_stones_in_five_piles (p1 p2 p3 p4 p5 : ℕ) 
  (h1 : p5 = 6 * p3)
  (h2 : p2 = 2 * (p3 + p5))
  (h3 : p1 = p5 / 3 ∧ p1 = p4 - 10)
  (h4 : p4 = p2 / 2) :
  p1 + p2 + p3 + p4 + p5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_stones_in_five_piles_l57_5756


namespace NUMINAMATH_CALUDE_yellow_balls_count_l57_5788

theorem yellow_balls_count (purple_count blue_count : ℕ) 
  (min_tries : ℕ) (yellow_count : ℕ) : 
  purple_count = 7 → 
  blue_count = 5 → 
  min_tries = 19 →
  yellow_count = min_tries - (purple_count + blue_count + 1) →
  yellow_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l57_5788


namespace NUMINAMATH_CALUDE_ratio_of_segments_on_line_l57_5768

/-- Given four points P, Q, R, S on a line in that order, with given distances between them,
    prove that the ratio of PR to QS is 7/12. -/
theorem ratio_of_segments_on_line (P Q R S : ℝ) (h_order : P < Q ∧ Q < R ∧ R < S)
    (h_PQ : Q - P = 4) (h_QR : R - Q = 10) (h_PS : S - P = 28) :
    (R - P) / (S - Q) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_segments_on_line_l57_5768


namespace NUMINAMATH_CALUDE_extreme_values_and_tangent_line_l57_5779

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8

/-- The derivative of f(x) with respect to x -/
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_and_tangent_line (a b : ℝ) :
  (f' a b 1 = 0 ∧ f' a b 2 = 0) →
  (a = -3 ∧ b = 4) ∧
  (∃ k m : ℝ, k * 0 + m = f (-3) 4 0 ∧ k = f' (-3) 4 0 ∧ k = 12 ∧ m = 8) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_tangent_line_l57_5779


namespace NUMINAMATH_CALUDE_number_ordering_l57_5761

theorem number_ordering : (2 : ℕ)^30 < (6 : ℕ)^10 ∧ (6 : ℕ)^10 < (3 : ℕ)^20 := by sorry

end NUMINAMATH_CALUDE_number_ordering_l57_5761


namespace NUMINAMATH_CALUDE_complex_cube_root_unity_l57_5764

theorem complex_cube_root_unity (i : ℂ) (y : ℂ) :
  i^2 = -1 →
  y = (1 + i * Real.sqrt 3) / 2 →
  1 / (y^3 - y) = -1/2 + (i * Real.sqrt 3) / 6 := by sorry

end NUMINAMATH_CALUDE_complex_cube_root_unity_l57_5764


namespace NUMINAMATH_CALUDE_sum_of_exponents_product_divisors_360000_l57_5746

/-- The product of all positive integer divisors of a natural number n -/
def product_of_divisors (n : ℕ) : ℕ := sorry

/-- The sum of exponents in the prime factorization of a natural number n -/
def sum_of_exponents (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of exponents in the prime factorization of the product of all positive integer divisors of 360000 is 630 -/
theorem sum_of_exponents_product_divisors_360000 :
  sum_of_exponents (product_of_divisors 360000) = 630 := by sorry

end NUMINAMATH_CALUDE_sum_of_exponents_product_divisors_360000_l57_5746


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l57_5733

/-- The asymptotes of the hyperbola x²/9 - y²/16 = 1 are given by y = ±(4/3)x -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → ℝ := λ x y => x^2 / 9 - y^2 / 16 - 1
  ∃ (k : ℝ), k > 0 ∧ ∀ (x y : ℝ), h x y = 0 → y = k * x ∨ y = -k * x :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l57_5733


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l57_5736

def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Finset ℕ := {2, 4, 5}

theorem complement_of_A_in_U :
  (U \ A) = {1, 3, 6, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l57_5736


namespace NUMINAMATH_CALUDE_total_subscription_is_50000_l57_5753

/-- Represents the subscription amounts and profit distribution for a business venture. -/
structure BusinessSubscription where
  /-- Subscription amount of person C -/
  c_amount : ℕ
  /-- Total profit of the business -/
  total_profit : ℕ
  /-- Profit received by person C -/
  c_profit : ℕ

/-- Calculates the total subscription amount based on the given conditions -/
def total_subscription (bs : BusinessSubscription) : ℕ :=
  3 * bs.c_amount + 14000

/-- Theorem stating that the total subscription amount is 50,000 given the problem conditions -/
theorem total_subscription_is_50000 (bs : BusinessSubscription) 
  (h1 : bs.total_profit = 35000)
  (h2 : bs.c_profit = 8400)
  (h3 : bs.c_profit * (total_subscription bs) = bs.total_profit * bs.c_amount) :
  total_subscription bs = 50000 := by
  sorry

#eval total_subscription { c_amount := 12000, total_profit := 35000, c_profit := 8400 }

end NUMINAMATH_CALUDE_total_subscription_is_50000_l57_5753


namespace NUMINAMATH_CALUDE_log_equation_solution_l57_5715

theorem log_equation_solution :
  ∃ y : ℝ, (2 * Real.log y + 3 * Real.log 2 = 1) ∧ (y = Real.sqrt 5 / 2) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l57_5715


namespace NUMINAMATH_CALUDE_intersection_length_l57_5793

/-- The length of segment AB is 8 when a line y = kx - k intersects 
    the parabola y² = 4x at points A and B, and the distance from 
    the midpoint of segment AB to the y-axis is 3 -/
theorem intersection_length (k : ℝ) (A B : ℝ × ℝ) : 
  (∃ (x y : ℝ), y = k * x - k ∧ y^2 = 4 * x) →  -- line intersects parabola
  (A.1 + B.1) / 2 = 3 →                         -- midpoint x-coordinate is 3
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    A = (x₁, y₁) ∧ 
    B = (x₂, y₂) ∧ 
    y₁ = k * x₁ - k ∧ 
    y₁^2 = 4 * x₁ ∧ 
    y₂ = k * x₂ - k ∧ 
    y₂^2 = 4 * x₂ ∧
    ((x₂ - x₁)^2 + (y₂ - y₁)^2).sqrt = 8 :=
by sorry

end NUMINAMATH_CALUDE_intersection_length_l57_5793


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l57_5796

theorem square_garden_perimeter (side : ℝ) (area perimeter : ℝ) : 
  area = side^2 → 
  perimeter = 4 * side → 
  area = 100 → 
  area = 2 * perimeter + 20 → 
  perimeter = 40 := by sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l57_5796


namespace NUMINAMATH_CALUDE_range_of_a_minus_abs_b_l57_5712

theorem range_of_a_minus_abs_b (a b : ℝ) (ha : 1 < a ∧ a < 3) (hb : -4 < b ∧ b < 2) :
  -3 < a - |b| ∧ a - |b| < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_minus_abs_b_l57_5712


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l57_5784

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : a 1 = 4)
  (h3 : arithmetic_sequence a d)
  (h4 : geometric_sequence (a 1) (a 3) (a 4)) :
  (∀ n : ℕ, a n = 5 - n) ∧
  (∃ max_sum : ℝ, max_sum = 10 ∧
    ∀ n : ℕ, (n * (2 * a 1 + (n - 1) * d)) / 2 ≤ max_sum) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l57_5784


namespace NUMINAMATH_CALUDE_expression_evaluation_l57_5732

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  let z : ℝ := 3
  2 * x^2 + y^2 - z^2 + 3 * x * y = -6 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l57_5732


namespace NUMINAMATH_CALUDE_alex_has_48_shells_l57_5717

/-- The number of seashells in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of seashells Mimi picked up -/
def mimi_dozens : ℕ := 2

/-- The number of seashells Mimi picked up -/
def mimi_shells : ℕ := mimi_dozens * dozen

/-- The number of seashells Kyle found -/
def kyle_shells : ℕ := 2 * mimi_shells

/-- The number of seashells Leigh grabbed -/
def leigh_shells : ℕ := kyle_shells / 3

/-- The number of seashells Alex unearthed -/
def alex_shells : ℕ := 3 * leigh_shells

/-- Theorem stating that Alex had 48 seashells -/
theorem alex_has_48_shells : alex_shells = 48 := by
  sorry

end NUMINAMATH_CALUDE_alex_has_48_shells_l57_5717


namespace NUMINAMATH_CALUDE_same_color_inevitable_l57_5775

/-- A type representing the colors of the balls -/
inductive Color
| Red
| Yellow

/-- The total number of balls in the bag -/
def total_balls : Nat := 6

/-- The number of red balls in the bag -/
def red_balls : Nat := 3

/-- The number of yellow balls in the bag -/
def yellow_balls : Nat := 3

/-- The number of balls drawn from the bag -/
def drawn_balls : Nat := 3

/-- A function that determines if a drawing of balls inevitably results in at least two balls of the same color -/
def inevitable_same_color (total : Nat) (red : Nat) (yellow : Nat) (drawn : Nat) : Prop :=
  ∀ (selection : Finset Color), selection.card = drawn → selection.card ≥ 2

/-- Theorem stating that drawing 3 balls from the bag inevitably results in at least two balls of the same color -/
theorem same_color_inevitable :
  inevitable_same_color total_balls red_balls yellow_balls drawn_balls :=
by sorry

end NUMINAMATH_CALUDE_same_color_inevitable_l57_5775


namespace NUMINAMATH_CALUDE_exists_surjective_function_with_property_l57_5759

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x else x - 1

-- State the theorem
theorem exists_surjective_function_with_property :
  ∃ (f : ℝ → ℝ), Function.Surjective f ∧
  (∀ x y : ℝ, (f (x + y) - f x - f y) ∈ ({0, 1} : Set ℝ)) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_exists_surjective_function_with_property_l57_5759


namespace NUMINAMATH_CALUDE_unique_solution_xy_equation_l57_5725

theorem unique_solution_xy_equation :
  ∃! (x y : ℕ), x < y ∧ x^y = y^x :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_xy_equation_l57_5725


namespace NUMINAMATH_CALUDE_perpendicular_lines_k_values_l57_5790

/-- Given two lines l₁ and l₂ defined by their equations, 
    this theorem states that if they are perpendicular, 
    then k must be either 0 or 3. -/
theorem perpendicular_lines_k_values 
  (k : ℝ) 
  (l₁ : ℝ → ℝ → Prop) 
  (l₂ : ℝ → ℝ → Prop) 
  (h₁ : ∀ x y, l₁ x y ↔ x + k * y - 2 * k = 0) 
  (h₂ : ∀ x y, l₂ x y ↔ k * x - (k - 2) * y + 1 = 0) 
  (h_perp : (∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ → l₂ x₂ y₂ → (x₂ - x₁) * (y₂ - y₁) = 0)) : 
  k = 0 ∨ k = 3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_k_values_l57_5790


namespace NUMINAMATH_CALUDE_distribute_five_to_three_l57_5747

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    with each box containing at least one object. -/
def distributeObjects (n k : ℕ) : ℕ :=
  sorry

theorem distribute_five_to_three :
  distributeObjects 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_to_three_l57_5747


namespace NUMINAMATH_CALUDE_total_markers_l57_5785

theorem total_markers (red_markers blue_markers : ℕ) : 
  red_markers = 2315 → blue_markers = 1028 → red_markers + blue_markers = 3343 := by
  sorry

end NUMINAMATH_CALUDE_total_markers_l57_5785


namespace NUMINAMATH_CALUDE_odd_prime_properties_l57_5763

theorem odd_prime_properties (p n : ℕ) (hp : Nat.Prime p) (hodd : Odd p) (hform : p = 4 * n + 1) :
  (∃ (x : ℕ), x ^ 2 % p = n % p) ∧ (n ^ n % p = 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_properties_l57_5763


namespace NUMINAMATH_CALUDE_min_stamps_for_47_cents_l57_5729

/-- Represents the number of stamps of each denomination -/
structure StampCombination where
  threes : ℕ
  fours : ℕ
  fives : ℕ

/-- Calculates the total value of stamps in cents -/
def total_value (sc : StampCombination) : ℕ :=
  3 * sc.threes + 4 * sc.fours + 5 * sc.fives

/-- Calculates the total number of stamps -/
def total_stamps (sc : StampCombination) : ℕ :=
  sc.threes + sc.fours + sc.fives

/-- Checks if at least two types of stamps are used -/
def uses_at_least_two_types (sc : StampCombination) : Prop :=
  (sc.threes > 0 && sc.fours > 0) || (sc.threes > 0 && sc.fives > 0) || (sc.fours > 0 && sc.fives > 0)

/-- States that 10 is the minimum number of stamps needed to make 47 cents -/
theorem min_stamps_for_47_cents :
  ∃ (sc : StampCombination),
    total_value sc = 47 ∧
    uses_at_least_two_types sc ∧
    total_stamps sc = 10 ∧
    (∀ (sc' : StampCombination),
      total_value sc' = 47 →
      uses_at_least_two_types sc' →
      total_stamps sc' ≥ 10) :=
  sorry

end NUMINAMATH_CALUDE_min_stamps_for_47_cents_l57_5729


namespace NUMINAMATH_CALUDE_bucket3_most_efficient_bucket3_count_verification_l57_5782

-- Define the tank capacities
def tank1_capacity : ℕ := 20000
def tank2_capacity : ℕ := 25000
def tank3_capacity : ℕ := 30000

-- Define the bucket capacities
def bucket1_capacity : ℕ := 13
def bucket2_capacity : ℕ := 28
def bucket3_capacity : ℕ := 36

-- Function to calculate the number of buckets needed
def buckets_needed (tank_capacity bucket_capacity : ℕ) : ℕ :=
  (tank_capacity + bucket_capacity - 1) / bucket_capacity

-- Theorem stating that the 36-litre bucket is most efficient for all tanks
theorem bucket3_most_efficient :
  (buckets_needed tank1_capacity bucket3_capacity ≤ buckets_needed tank1_capacity bucket1_capacity) ∧
  (buckets_needed tank1_capacity bucket3_capacity ≤ buckets_needed tank1_capacity bucket2_capacity) ∧
  (buckets_needed tank2_capacity bucket3_capacity ≤ buckets_needed tank2_capacity bucket1_capacity) ∧
  (buckets_needed tank2_capacity bucket3_capacity ≤ buckets_needed tank2_capacity bucket2_capacity) ∧
  (buckets_needed tank3_capacity bucket3_capacity ≤ buckets_needed tank3_capacity bucket1_capacity) ∧
  (buckets_needed tank3_capacity bucket3_capacity ≤ buckets_needed tank3_capacity bucket2_capacity) :=
by sorry

-- Verify the exact number of 36-litre buckets needed for each tank
theorem bucket3_count_verification :
  (buckets_needed tank1_capacity bucket3_capacity = 556) ∧
  (buckets_needed tank2_capacity bucket3_capacity = 695) ∧
  (buckets_needed tank3_capacity bucket3_capacity = 834) :=
by sorry

end NUMINAMATH_CALUDE_bucket3_most_efficient_bucket3_count_verification_l57_5782


namespace NUMINAMATH_CALUDE_batsman_new_average_l57_5721

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  innings : Nat
  totalRuns : Nat
  latestScore : Nat
  averageIncrease : Nat

/-- Calculates the average score after the latest innings -/
def calculateNewAverage (stats : BatsmanStats) : Nat :=
  (stats.totalRuns + stats.latestScore) / stats.innings

/-- Theorem: Given the conditions, the batsman's new average is 43 -/
theorem batsman_new_average (stats : BatsmanStats) 
  (h1 : stats.innings = 12)
  (h2 : stats.latestScore = 65)
  (h3 : stats.averageIncrease = 2)
  (h4 : calculateNewAverage stats = (calculateNewAverage stats - stats.averageIncrease) + stats.averageIncrease) :
  calculateNewAverage stats = 43 := by
  sorry

#eval calculateNewAverage { innings := 12, totalRuns := 451, latestScore := 65, averageIncrease := 2 }

end NUMINAMATH_CALUDE_batsman_new_average_l57_5721


namespace NUMINAMATH_CALUDE_cube_surface_area_l57_5727

/-- Given a cube with volume 1728 cubic centimeters, its surface area is 864 square centimeters. -/
theorem cube_surface_area (volume : ℝ) (side : ℝ) :
  volume = 1728 →
  volume = side^3 →
  6 * side^2 = 864 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l57_5727


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l57_5791

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (8 * bowling_ball_weight = 4 * canoe_weight) →
    (3 * canoe_weight = 108) →
    bowling_ball_weight = 18 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l57_5791


namespace NUMINAMATH_CALUDE_sin_cos_value_l57_5769

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem sin_cos_value (θ : ℝ) :
  determinant (Real.sin θ) 2 (Real.cos θ) 3 = 0 →
  2 * (Real.sin θ)^2 + (Real.sin θ) * (Real.cos θ) = 14/13 :=
by
  sorry

end NUMINAMATH_CALUDE_sin_cos_value_l57_5769


namespace NUMINAMATH_CALUDE_total_spending_is_correct_l57_5703

-- Define the structure for a week's theater visit
structure WeekVisit where
  duration : Float
  price_per_hour : Float
  discount_rate : Float
  visit_count : Nat

-- Define the list of visits for 6 weeks
def theater_visits : List WeekVisit := [
  { duration := 3, price_per_hour := 5, discount_rate := 0.2, visit_count := 1 },
  { duration := 2.5, price_per_hour := 6, discount_rate := 0.1, visit_count := 1 },
  { duration := 4, price_per_hour := 4, discount_rate := 0, visit_count := 1 },
  { duration := 3, price_per_hour := 5, discount_rate := 0.2, visit_count := 1 },
  { duration := 3.5, price_per_hour := 6, discount_rate := 0.1, visit_count := 2 },
  { duration := 2, price_per_hour := 7, discount_rate := 0, visit_count := 1 }
]

-- Define the transportation cost per visit
def transportation_cost : Float := 3

-- Calculate the total cost for a single visit
def visit_cost (visit : WeekVisit) : Float :=
  let performance_cost := visit.duration * visit.price_per_hour
  let discount := performance_cost * visit.discount_rate
  let discounted_cost := performance_cost - discount
  discounted_cost + transportation_cost

-- Calculate the total spending for all visits
def total_spending : Float :=
  theater_visits.map (fun visit => visit_cost visit * visit.visit_count.toFloat) |>.sum

-- Theorem statement
theorem total_spending_is_correct : total_spending = 126.3 := by
  sorry

end NUMINAMATH_CALUDE_total_spending_is_correct_l57_5703


namespace NUMINAMATH_CALUDE_solution_set_characterization_l57_5728

def is_solution_set (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x, x ∈ S ↔ 2^(1 + f x) + 2^(1 - f x) + 2 * f (x^2) ≤ 7

theorem solution_set_characterization
  (f : ℝ → ℝ)
  (h1 : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂)
  (h2 : ∀ x, x > 0 → f x > 0)
  (h3 : f 1 = 1) :
  is_solution_set f (Set.Icc (-1) 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l57_5728


namespace NUMINAMATH_CALUDE_prob_higher_roll_and_sum_l57_5762

/-- The number of sides on a standard die -/
def die_sides : ℕ := 6

/-- The probability of rolling a higher number on one die compared to another -/
def prob_higher_roll : ℚ :=
  (die_sides * (die_sides - 1) / 2) / (die_sides^2 : ℚ)

/-- The sum of the numerator and denominator of the probability fraction in lowest terms -/
def sum_num_denom : ℕ := 17

theorem prob_higher_roll_and_sum :
  prob_higher_roll = 5/12 ∧ sum_num_denom = 17 := by sorry

end NUMINAMATH_CALUDE_prob_higher_roll_and_sum_l57_5762


namespace NUMINAMATH_CALUDE_percentage_commutation_l57_5701

theorem percentage_commutation (n : ℝ) (h : 0.20 * 0.10 * n = 12) : 0.10 * 0.20 * n = 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_commutation_l57_5701


namespace NUMINAMATH_CALUDE_equation_proof_l57_5708

theorem equation_proof : 484 + 2 * 22 * 3 + 9 = 625 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l57_5708


namespace NUMINAMATH_CALUDE_line_slope_is_pi_over_three_l57_5752

theorem line_slope_is_pi_over_three (x y : ℝ) :
  2 * Real.sqrt 3 * x - 2 * y - 1 = 0 →
  ∃ (m : ℝ), (∀ x y, y = m * x - 1 / 2) ∧ m = Real.tan (π / 3) :=
sorry

end NUMINAMATH_CALUDE_line_slope_is_pi_over_three_l57_5752


namespace NUMINAMATH_CALUDE_patio_rearrangement_l57_5767

theorem patio_rearrangement (r c : ℕ) : 
  r * c = 48 ∧ 
  (r + 4) * (c - 2) = 48 ∧ 
  c > 2 →
  r = 6 :=
by sorry

end NUMINAMATH_CALUDE_patio_rearrangement_l57_5767


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l57_5794

theorem base_conversion_theorem (n : ℕ) (C D : ℕ) : 
  n > 0 ∧ 
  C < 8 ∧ 
  D < 5 ∧ 
  n = 8 * C + D ∧ 
  n = 5 * D + C → 
  n = 0 := by sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l57_5794


namespace NUMINAMATH_CALUDE_min_value_range_lower_bound_value_l57_5777

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + a * |x - 2|

-- Theorem for part I
theorem min_value_range (a : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m) → -1 ≤ a ∧ a ≤ 1 :=
by sorry

-- Theorem for part II
theorem lower_bound_value (a : ℝ) :
  (∀ (x : ℝ), f a x ≥ 1/2) → a = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_range_lower_bound_value_l57_5777


namespace NUMINAMATH_CALUDE_power_of_four_exponent_l57_5765

theorem power_of_four_exponent (n : ℕ) (x : ℕ) 
  (h : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^x) 
  (hn : n = 25) : x = 26 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_exponent_l57_5765
