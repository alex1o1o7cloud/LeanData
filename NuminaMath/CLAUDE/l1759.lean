import Mathlib

namespace tangent_circles_count_l1759_175992

-- Define a circle in a plane
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the property of two circles being tangent
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Define the property of a circle being tangent to two other circles
def is_tangent_to_both (c : Circle) (c1 c2 : Circle) : Prop :=
  are_tangent c c1 ∧ are_tangent c c2

-- State the theorem
theorem tangent_circles_count 
  (c1 c2 : Circle) 
  (h1 : c1.radius = 2) 
  (h2 : c2.radius = 2) 
  (h3 : are_tangent c1 c2) :
  ∃! (s : Finset Circle), 
    (∀ c ∈ s, c.radius = 4 ∧ is_tangent_to_both c c1 c2) ∧ 
    s.card = 6 :=
sorry

end tangent_circles_count_l1759_175992


namespace cheryl_eggs_count_l1759_175901

/-- The number of eggs found by Kevin -/
def kevin_eggs : ℕ := 5

/-- The number of eggs found by Bonnie -/
def bonnie_eggs : ℕ := 13

/-- The number of eggs found by George -/
def george_eggs : ℕ := 9

/-- The number of additional eggs Cheryl found compared to the others -/
def cheryl_additional_eggs : ℕ := 29

/-- Theorem stating that Cheryl found 56 eggs -/
theorem cheryl_eggs_count : 
  kevin_eggs + bonnie_eggs + george_eggs + cheryl_additional_eggs = 56 := by
  sorry

end cheryl_eggs_count_l1759_175901


namespace zorg_game_threshold_l1759_175966

theorem zorg_game_threshold : ∃ (n : ℕ), n = 40 ∧ ∀ (m : ℕ), m < n → (m * (m + 1)) / 2 ≤ 20 * m :=
by sorry

end zorg_game_threshold_l1759_175966


namespace remainder_2_pow_13_mod_3_l1759_175952

theorem remainder_2_pow_13_mod_3 : 2^13 ≡ 2 [ZMOD 3] := by sorry

end remainder_2_pow_13_mod_3_l1759_175952


namespace points_always_odd_l1759_175983

/-- Represents the number of points on the line after a certain number of operations -/
def num_points (initial : ℕ) (operations : ℕ) : ℕ :=
  if operations = 0 then
    initial
  else
    2 * num_points initial (operations - 1) - 1

/-- Theorem stating that the number of points is always odd after any number of operations -/
theorem points_always_odd (initial : ℕ) (operations : ℕ) :
  Odd (num_points initial operations) :=
by
  sorry


end points_always_odd_l1759_175983


namespace larger_number_proof_l1759_175935

theorem larger_number_proof (x y : ℝ) (h1 : x + y = 28) (h2 : x - y = 4) : 
  max x y = 16 := by
sorry

end larger_number_proof_l1759_175935


namespace vincents_earnings_l1759_175900

def fantasy_book_price : ℚ := 4
def literature_book_price : ℚ := fantasy_book_price / 2
def fantasy_books_sold_per_day : ℕ := 5
def literature_books_sold_per_day : ℕ := 8
def days : ℕ := 5

theorem vincents_earnings :
  (fantasy_book_price * fantasy_books_sold_per_day +
   literature_book_price * literature_books_sold_per_day) * days = 180 := by
  sorry

end vincents_earnings_l1759_175900


namespace ninth_minus_eighth_rectangle_tiles_l1759_175923

/-- The number of tiles in the nth rectangle of the sequence -/
def tiles (n : ℕ) : ℕ := 2 * n * n

/-- The difference in tiles between the 9th and 8th rectangles -/
def tile_difference : ℕ := tiles 9 - tiles 8

theorem ninth_minus_eighth_rectangle_tiles : tile_difference = 34 := by
  sorry

end ninth_minus_eighth_rectangle_tiles_l1759_175923


namespace janice_earnings_this_week_l1759_175967

/-- Calculates Janice's weekly earnings based on her work schedule and wages -/
def janice_weekly_earnings (regular_days : ℕ) (regular_wage : ℕ) (overtime_shifts : ℕ) (overtime_bonus : ℕ) : ℕ :=
  regular_days * regular_wage + overtime_shifts * overtime_bonus

/-- Proves that Janice's weekly earnings are $195 given her work schedule -/
theorem janice_earnings_this_week :
  janice_weekly_earnings 5 30 3 15 = 195 := by
  sorry

#eval janice_weekly_earnings 5 30 3 15

end janice_earnings_this_week_l1759_175967


namespace binomial_expansion_coefficient_ratio_l1759_175912

theorem binomial_expansion_coefficient_ratio (n : ℕ) : 
  4 * (n.choose 2) = 7 * (2 * n) → n = 8 := by
  sorry

end binomial_expansion_coefficient_ratio_l1759_175912


namespace fractional_exponent_simplification_l1759_175919

theorem fractional_exponent_simplification :
  (2^2024 + 2^2020) / (2^2024 - 2^2020) = 17 / 15 := by
  sorry

end fractional_exponent_simplification_l1759_175919


namespace polynomial_factor_implies_coefficients_l1759_175978

theorem polynomial_factor_implies_coefficients 
  (a b : ℚ) 
  (h : ∃ (c d : ℚ), ax^4 + bx^3 + 40*x^2 - 20*x + 10 = (5*x^2 - 3*x + 2)*(c*x^2 + d*x + 5)) :
  a = 25/4 ∧ b = -65/4 := by
sorry

end polynomial_factor_implies_coefficients_l1759_175978


namespace complex_function_property_l1759_175931

/-- A complex function f(z) = (a+bi)z with certain properties -/
def f (a b : ℝ) (z : ℂ) : ℂ := (Complex.mk a b) * z

/-- The theorem statement -/
theorem complex_function_property (a b c : ℝ) :
  (a > 0) →
  (b > 0) →
  (c > 0) →
  (∀ z : ℂ, Complex.abs (f a b z - z) = Complex.abs (f a b z - Complex.I * c)) →
  (Complex.abs (Complex.mk a b) = 9) →
  (b^2 = 323/4) := by
  sorry

end complex_function_property_l1759_175931


namespace orangeade_price_day2_l1759_175971

/-- Represents the price and volume of orangeade on two consecutive days -/
structure Orangeade where
  orange_juice : ℝ  -- Amount of orange juice (same for both days)
  water_day1 : ℝ    -- Amount of water on day 1
  water_day2 : ℝ    -- Amount of water on day 2
  price_day1 : ℝ    -- Price per glass on day 1
  price_day2 : ℝ    -- Price per glass on day 2
  revenue : ℝ        -- Revenue (same for both days)

/-- The price per glass on the second day is $0.20 given the conditions -/
theorem orangeade_price_day2 (o : Orangeade)
    (h1 : o.orange_juice = o.water_day1)
    (h2 : o.water_day2 = 2 * o.water_day1)
    (h3 : o.price_day1 = 0.30)
    (h4 : o.revenue = (o.orange_juice + o.water_day1) * o.price_day1)
    (h5 : o.revenue = (o.orange_juice + o.water_day2) * o.price_day2) :
  o.price_day2 = 0.20 := by
  sorry

end orangeade_price_day2_l1759_175971


namespace sqrt_product_equality_l1759_175925

theorem sqrt_product_equality : (Real.sqrt 8 + Real.sqrt 3) * Real.sqrt 6 = 4 * Real.sqrt 3 + 3 * Real.sqrt 2 := by
  sorry

end sqrt_product_equality_l1759_175925


namespace emma_has_eight_l1759_175990

/-- The amount of money each person has -/
structure Money where
  emma : ℝ
  daya : ℝ
  jeff : ℝ
  brenda : ℝ

/-- The conditions of the problem -/
def money_conditions (m : Money) : Prop :=
  m.daya = 1.25 * m.emma ∧
  m.jeff = 0.4 * m.daya ∧
  m.brenda = m.jeff + 4 ∧
  m.brenda = 8

/-- The theorem stating Emma has $8 -/
theorem emma_has_eight (m : Money) (h : money_conditions m) : m.emma = 8 := by
  sorry

end emma_has_eight_l1759_175990


namespace geometric_series_first_term_l1759_175928

theorem geometric_series_first_term
  (r : ℝ)
  (hr : |r| < 1)
  (h_sum : (∑' n, r^n) * a = 15)
  (h_sum_squares : (∑' n, (r^n)^2) * a^2 = 45) :
  a = 5 :=
sorry

end geometric_series_first_term_l1759_175928


namespace eulers_criterion_l1759_175945

theorem eulers_criterion (p : Nat) (a : Nat) (h_prime : Nat.Prime p) (h_p : p > 2) (h_a : 1 ≤ a ∧ a ≤ p - 1) :
  (∃ x : Nat, x ^ 2 % p = a % p) ↔ a ^ ((p - 1) / 2) % p = 1 := by
  sorry

end eulers_criterion_l1759_175945


namespace total_birds_in_marsh_l1759_175954

theorem total_birds_in_marsh (geese ducks swans : ℕ) 
  (h1 : geese = 58) 
  (h2 : ducks = 37) 
  (h3 : swans = 42) : 
  geese + ducks + swans = 137 := by
  sorry

end total_birds_in_marsh_l1759_175954


namespace range_of_m_l1759_175937

/-- The curve equation -/
def curve (x y m : ℝ) : Prop := x^2 + y^2 + y + m = 0

/-- The symmetry line equation -/
def symmetry_line (x y : ℝ) : Prop := x + 2*y - 1 = 0

/-- Predicate for having four common tangents -/
def has_four_common_tangents (m : ℝ) : Prop := sorry

/-- Theorem stating the range of m -/
theorem range_of_m : 
  ∀ m : ℝ, (∀ x y : ℝ, curve x y m → ∃ x' y' : ℝ, symmetry_line x' y' ∧ has_four_common_tangents m) 
  ↔ -11/20 < m ∧ m < 1/4 := by sorry

end range_of_m_l1759_175937


namespace right_triangle_set_l1759_175934

theorem right_triangle_set : ∃! (a b c : ℝ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∨ 
   (a = 2 ∧ b = 3 ∧ c = 4) ∨ 
   (a = 6 ∧ b = 8 ∧ c = 12) ∨ 
   (a = Real.sqrt 3 ∧ b = Real.sqrt 4 ∧ c = Real.sqrt 5)) ∧
  a^2 + b^2 = c^2 := by
sorry

end right_triangle_set_l1759_175934


namespace angle_increase_in_equilateral_triangle_l1759_175946

/-- 
Given an equilateral triangle where each angle initially measures 60 degrees,
if one angle is increased by 40 degrees, the resulting measure of that angle is 100 degrees.
-/
theorem angle_increase_in_equilateral_triangle :
  ∀ (A B C : ℝ),
  A = 60 ∧ B = 60 ∧ C = 60 →  -- Initially equilateral triangle
  (C + 40 : ℝ) = 100 :=
by sorry

end angle_increase_in_equilateral_triangle_l1759_175946


namespace fraction_to_decimal_l1759_175997

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 := by
  sorry

end fraction_to_decimal_l1759_175997


namespace prize_distribution_l1759_175994

theorem prize_distribution (total_winners : ℕ) (min_award : ℚ) (max_award : ℚ) :
  total_winners = 15 →
  min_award = 15 →
  max_award = 285 →
  ∃ (total_prize : ℚ),
    (2 / 5 : ℚ) * total_prize = max_award * ((3 / 5 : ℚ) * total_winners) ∧
    total_prize = 6502.5 :=
by sorry

end prize_distribution_l1759_175994


namespace product_sequence_value_l1759_175961

theorem product_sequence_value : 
  (1 / 3) * (9 / 1) * (1 / 27) * (81 / 1) * (1 / 243) * (729 / 1) * (1 / 729) * (2187 / 1) = 729 := by
  sorry

end product_sequence_value_l1759_175961


namespace fraction_to_decimal_l1759_175993

theorem fraction_to_decimal : (21 : ℚ) / 160 = 0.13125 := by
  sorry

end fraction_to_decimal_l1759_175993


namespace g_of_7_eq_92_l1759_175914

def g (n : ℕ) : ℕ := n^2 + 2*n + 29

theorem g_of_7_eq_92 : g 7 = 92 := by sorry

end g_of_7_eq_92_l1759_175914


namespace subtraction_sum_l1759_175998

/-- Given a subtraction problem with digits K, L, M, and N, prove that their sum is 20 -/
theorem subtraction_sum (K L M N : Nat) : 
  (K < 10) → (L < 10) → (M < 10) → (N < 10) →
  (5000 + 100 * K + 30 + L) - (1000 * M + 400 + 10 * N + 1) = 4451 →
  K + L + M + N = 20 := by
sorry

end subtraction_sum_l1759_175998


namespace smallest_n_with_conditions_n_satisfies_conditions_l1759_175973

def has_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + d + 10 * m

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

theorem smallest_n_with_conditions :
  ∀ n : ℕ, n > 0 →
    (is_terminating_decimal n ∧ has_digit n 9 ∧ has_digit n 2) →
    n ≥ 524288 :=
by sorry

theorem n_satisfies_conditions :
  is_terminating_decimal 524288 ∧ has_digit 524288 9 ∧ has_digit 524288 2 :=
by sorry

end smallest_n_with_conditions_n_satisfies_conditions_l1759_175973


namespace bryan_total_books_l1759_175975

/-- The number of books in each of Bryan's bookshelves -/
def books_per_shelf : ℕ := 27

/-- The number of bookshelves Bryan has -/
def number_of_shelves : ℕ := 23

/-- The total number of books Bryan has -/
def total_books : ℕ := books_per_shelf * number_of_shelves

theorem bryan_total_books : total_books = 621 := by
  sorry

end bryan_total_books_l1759_175975


namespace max_valid_arrangement_l1759_175907

/-- A type representing the cards with numbers 1 to 9 -/
inductive Card : Type
  | one | two | three | four | five | six | seven | eight | nine

/-- A function that returns the numerical value of a card -/
def cardValue : Card → Nat
  | Card.one => 1
  | Card.two => 2
  | Card.three => 3
  | Card.four => 4
  | Card.five => 5
  | Card.six => 6
  | Card.seven => 7
  | Card.eight => 8
  | Card.nine => 9

/-- A predicate that checks if two cards satisfy the adjacency condition -/
def validAdjacent (c1 c2 : Card) : Prop :=
  (cardValue c1 ∣ cardValue c2) ∨ (cardValue c2 ∣ cardValue c1)

/-- A type representing a valid arrangement of cards -/
def ValidArrangement := List Card

/-- A predicate that checks if an arrangement is valid -/
def isValidArrangement : ValidArrangement → Prop
  | [] => True
  | [_] => True
  | (c1 :: c2 :: rest) => validAdjacent c1 c2 ∧ isValidArrangement (c2 :: rest)

/-- The main theorem stating that the maximum number of cards in a valid arrangement is 8 -/
theorem max_valid_arrangement :
  (∃ (arr : ValidArrangement), isValidArrangement arr ∧ arr.length = 8) ∧
  (∀ (arr : ValidArrangement), isValidArrangement arr → arr.length ≤ 8) :=
sorry

end max_valid_arrangement_l1759_175907


namespace square_sum_equals_73_l1759_175930

theorem square_sum_equals_73 (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 24) : 
  a^2 + b^2 = 73 := by
sorry

end square_sum_equals_73_l1759_175930


namespace shortest_distance_circle_to_line_l1759_175958

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 7

-- Define the line C₂
def C₂ (x y : ℝ) : Prop := x + y = 4

-- State the theorem
theorem shortest_distance_circle_to_line :
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 - Real.sqrt 7 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    C₁ x₁ y₁ → C₂ x₂ y₂ →
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≥ d :=
sorry

end shortest_distance_circle_to_line_l1759_175958


namespace percentage_problem_l1759_175995

theorem percentage_problem (x : ℝ) : 
  (0.15 * 25) + (x / 100 * 45) = 9.15 ↔ x = 12 := by
  sorry

end percentage_problem_l1759_175995


namespace cricket_captain_age_l1759_175964

theorem cricket_captain_age (team_size : ℕ) (captain_age wicket_keeper_age : ℕ) 
  (team_average : ℚ) (remaining_average : ℚ) :
  team_size = 11 →
  wicket_keeper_age = captain_age + 3 →
  team_average = 25 →
  remaining_average = team_average - 1 →
  (team_size : ℚ) * team_average = 
    (team_size - 2 : ℚ) * remaining_average + captain_age + wicket_keeper_age →
  captain_age = 28 := by
  sorry

end cricket_captain_age_l1759_175964


namespace specific_pyramid_base_edge_length_l1759_175913

/-- A square pyramid with a sphere inside --/
structure PyramidWithSphere where
  pyramid_height : ℝ
  sphere_radius : ℝ
  sphere_tangent_to_faces : Bool
  sphere_contacts_base : Bool

/-- Calculates the edge length of the pyramid's base --/
def base_edge_length (p : PyramidWithSphere) : ℝ :=
  sorry

/-- Theorem stating the base edge length of the specific pyramid --/
theorem specific_pyramid_base_edge_length :
  let p : PyramidWithSphere := {
    pyramid_height := 9,
    sphere_radius := 3,
    sphere_tangent_to_faces := true,
    sphere_contacts_base := true
  }
  base_edge_length p = 4.5 := by
  sorry

end specific_pyramid_base_edge_length_l1759_175913


namespace truncated_pyramid_edge_count_l1759_175918

/-- A square-based pyramid with truncated vertices -/
structure TruncatedPyramid where
  /-- The number of vertices in the original square-based pyramid -/
  original_vertices : Nat
  /-- The number of edges in the original square-based pyramid -/
  original_edges : Nat
  /-- The number of new edges created by each truncation -/
  new_edges_per_truncation : Nat
  /-- Assertion that the original shape is a square-based pyramid -/
  is_square_based_pyramid : original_vertices = 5 ∧ original_edges = 8
  /-- Assertion that each truncation creates a triangular face -/
  truncation_creates_triangle : new_edges_per_truncation = 3

/-- Theorem stating that a truncated square-based pyramid has 23 edges -/
theorem truncated_pyramid_edge_count (p : TruncatedPyramid) :
  p.original_edges + p.original_vertices * p.new_edges_per_truncation = 23 :=
by sorry

end truncated_pyramid_edge_count_l1759_175918


namespace percentage_of_360_l1759_175940

theorem percentage_of_360 : (42 : ℝ) / 100 * 360 = 151.2 := by
  sorry

end percentage_of_360_l1759_175940


namespace largest_four_digit_sum_16_l1759_175922

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  let rec aux (m : ℕ) (acc : ℕ) : ℕ :=
    if m = 0 then acc
    else aux (m / 10) (acc + m % 10)
  aux n 0

-- Define the property for a number to be a four-digit number
def isFourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

-- Theorem statement
theorem largest_four_digit_sum_16 :
  ∀ n : ℕ, isFourDigitNumber n → sumOfDigits n = 16 → n ≤ 9700 :=
sorry

end largest_four_digit_sum_16_l1759_175922


namespace jakes_birdhouse_width_l1759_175985

/-- Sara's birdhouse dimensions in feet -/
def sara_width : ℝ := 1
def sara_height : ℝ := 2
def sara_depth : ℝ := 2

/-- Jake's birdhouse dimensions in inches -/
def jake_height : ℝ := 20
def jake_depth : ℝ := 18

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

/-- Volume difference between Jake's and Sara's birdhouses in cubic inches -/
def volume_difference : ℝ := 1152

/-- Theorem stating that Jake's birdhouse width is 22.4 inches -/
theorem jakes_birdhouse_width :
  ∃ (jake_width : ℝ),
    jake_width * jake_height * jake_depth -
    (sara_width * sara_height * sara_depth * feet_to_inches^3) =
    volume_difference ∧
    jake_width = 22.4 := by
  sorry

end jakes_birdhouse_width_l1759_175985


namespace correct_substitution_l1759_175941

theorem correct_substitution (x y : ℝ) : 
  y = 1 - x ∧ x - 2*y = 4 → x - 2 + 2*x = 4 := by
  sorry

end correct_substitution_l1759_175941


namespace pie_chart_most_suitable_for_gas_mixture_l1759_175960

/-- Represents different types of statistical charts -/
inductive StatChart
  | PieChart
  | LineChart
  | BarChart
  deriving Repr

/-- Represents a mixture of gases -/
structure GasMixture where
  components : List String
  proportions : List Float
  sum_to_one : proportions.sum = 1

/-- Determines if a chart type is suitable for representing a gas mixture -/
def is_suitable_chart (chart : StatChart) (mixture : GasMixture) : Prop :=
  match chart with
  | StatChart.PieChart => 
      mixture.components.length > 1 ∧ 
      mixture.proportions.all (λ p => p ≥ 0 ∧ p ≤ 1)
  | _ => False

/-- Theorem stating that a pie chart is the most suitable for representing a gas mixture -/
theorem pie_chart_most_suitable_for_gas_mixture (mixture : GasMixture) :
  ∀ (chart : StatChart), is_suitable_chart chart mixture → chart = StatChart.PieChart :=
by sorry

end pie_chart_most_suitable_for_gas_mixture_l1759_175960


namespace total_minutes_worked_l1759_175902

/-- Calculates the total minutes worked by three people given specific conditions -/
theorem total_minutes_worked (bianca_hours : ℝ) : 
  bianca_hours = 12.5 → 
  (3 * bianca_hours + bianca_hours - 8.5) * 60 = 3240 := by
  sorry

#check total_minutes_worked

end total_minutes_worked_l1759_175902


namespace gcd_840_1764_l1759_175904

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l1759_175904


namespace system_A_is_valid_other_systems_not_valid_l1759_175988

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants. -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A system of two linear equations. -/
structure LinearSystem where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- The system of equations from option A. -/
def systemA : LinearSystem := {
  eq1 := { a := 1, b := 0, c := 2 }
  eq2 := { a := 0, b := 1, c := 7 }
}

/-- Predicate to check if a given system is a valid system of two linear equations. -/
def isValidLinearSystem (s : LinearSystem) : Prop :=
  -- Additional conditions can be added here if needed
  True

theorem system_A_is_valid : isValidLinearSystem systemA := by
  sorry

/-- The other systems (B, C, D) are not valid systems of two linear equations. -/
theorem other_systems_not_valid :
  ∃ (systemB systemC systemD : LinearSystem),
    ¬ isValidLinearSystem systemB ∧
    ¬ isValidLinearSystem systemC ∧
    ¬ isValidLinearSystem systemD := by
  sorry

end system_A_is_valid_other_systems_not_valid_l1759_175988


namespace common_chord_length_l1759_175996

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 9
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 3 = 0

-- Define the common chord line
def common_chord_line (x y : ℝ) : Prop := 2*x - y - 3 = 0

-- Theorem statement
theorem common_chord_length :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧
    C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧
    common_chord_line A.1 A.2 ∧ common_chord_line B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 12 * Real.sqrt 5 / 5 :=
by sorry

end common_chord_length_l1759_175996


namespace expression_evaluation_l1759_175962

theorem expression_evaluation (a b : ℝ) 
  (h : |a + 2| + (b - 1)^2 = 0) : 
  (a + 3*b) * (2*a - b) - 2*(a - b)^2 = -23 := by
  sorry

end expression_evaluation_l1759_175962


namespace wood_sawing_problem_l1759_175957

theorem wood_sawing_problem (original_length final_length : ℝ) 
  (h1 : original_length = 8.9)
  (h2 : final_length = 6.6) :
  original_length - final_length = 2.3 := by
  sorry

end wood_sawing_problem_l1759_175957


namespace cat_food_sale_theorem_l1759_175989

/-- Calculates the total number of cat food cases sold during a sale. -/
def total_cases_sold (first_group : Nat) (second_group : Nat) (third_group : Nat)
  (first_group_cases : Nat) (second_group_cases : Nat) (third_group_cases : Nat) : Nat :=
  first_group * first_group_cases + second_group * second_group_cases + third_group * third_group_cases

/-- Proves that the total number of cat food cases sold is 40 given the specified customer groups and their purchases. -/
theorem cat_food_sale_theorem :
  total_cases_sold 8 4 8 3 2 1 = 40 := by
  sorry

end cat_food_sale_theorem_l1759_175989


namespace intersection_circle_line_l1759_175917

/-- Given a line and a circle that intersect at two points, prove that the radius of the circle
    has a specific value when the line from the origin to one intersection point is perpendicular
    to the line from the origin to the other intersection point. -/
theorem intersection_circle_line (r : ℝ) (A B : ℝ × ℝ) : r > 0 →
  (3 * A.1 - 4 * A.2 - 1 = 0) →
  (3 * B.1 - 4 * B.2 - 1 = 0) →
  (A.1^2 + A.2^2 = r^2) →
  (B.1^2 + B.2^2 = r^2) →
  (A.1 * B.1 + A.2 * B.2 = 0) →
  r = Real.sqrt 2 / 5 := by
  sorry

#check intersection_circle_line

end intersection_circle_line_l1759_175917


namespace all_good_numbers_less_than_1000_l1759_175979

def isGood (n : ℕ) : Prop :=
  ∀ k p : ℕ, (10^p * k + n) % n = 0

def goodNumbersLessThan1000 : List ℕ := [1, 2, 5, 10, 20, 25, 50, 100, 125, 200]

theorem all_good_numbers_less_than_1000 :
  ∀ n ∈ goodNumbersLessThan1000, isGood n ∧ n < 1000 := by
  sorry

#check all_good_numbers_less_than_1000

end all_good_numbers_less_than_1000_l1759_175979


namespace complex_symmetry_ratio_imag_part_l1759_175906

theorem complex_symmetry_ratio_imag_part (z₁ z₂ : ℂ) :
  z₁ = 1 - 2*I →
  (z₂.re = -z₁.re ∧ z₂.im = z₁.im) →
  (z₂ / z₁).im = -4/5 := by
  sorry

end complex_symmetry_ratio_imag_part_l1759_175906


namespace min_value_of_expression_l1759_175977

theorem min_value_of_expression (c d : ℤ) (h : c^2 > d^2) :
  (((c^2 + d^2) / (c^2 - d^2)) + ((c^2 - d^2) / (c^2 + d^2)) : ℚ) ≥ 2 ∧
  ∃ (c d : ℤ), c^2 > d^2 ∧ ((c^2 + d^2) / (c^2 - d^2)) + ((c^2 - d^2) / (c^2 + d^2)) = 2 :=
by sorry

end min_value_of_expression_l1759_175977


namespace binomial_coefficient_x3y5_in_x_plus_y_8_l1759_175950

theorem binomial_coefficient_x3y5_in_x_plus_y_8 :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k) * (1 : ℕ)^k * (1 : ℕ)^(8 - k)) = 256 ∧
  (Nat.choose 8 3) = 56 :=
sorry

end binomial_coefficient_x3y5_in_x_plus_y_8_l1759_175950


namespace profit_difference_A_C_l1759_175959

-- Define the profit-sharing ratios
def ratio_A : ℕ := 3
def ratio_B : ℕ := 5
def ratio_C : ℕ := 6
def ratio_D : ℕ := 7

-- Define B's profit share
def profit_B : ℕ := 2000

-- Theorem statement
theorem profit_difference_A_C : 
  let part_value : ℚ := profit_B / ratio_B
  let profit_A : ℚ := part_value * ratio_A
  let profit_C : ℚ := part_value * ratio_C
  profit_C - profit_A = 1200 := by sorry

end profit_difference_A_C_l1759_175959


namespace complement_of_A_l1759_175929

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x ≥ 1} ∪ {x | x ≤ 0}

theorem complement_of_A (x : ℝ) : x ∈ Aᶜ ↔ 0 < x ∧ x < 1 := by sorry

end complement_of_A_l1759_175929


namespace glove_at_midpoint_l1759_175974

/-- Represents the escalator system and Semyon's movement -/
structure EscalatorSystem where
  /-- The speed of both escalators -/
  escalator_speed : ℝ
  /-- Semyon's walking speed -/
  semyon_speed : ℝ
  /-- The total height of the escalators -/
  total_height : ℝ

/-- Theorem stating that the glove will be at the midpoint when Semyon reaches the top -/
theorem glove_at_midpoint (system : EscalatorSystem)
  (h1 : system.escalator_speed > 0)
  (h2 : system.semyon_speed = system.escalator_speed)
  (h3 : system.total_height > 0) :
  let time_to_top := system.total_height / (2 * system.escalator_speed)
  let glove_position := system.escalator_speed * time_to_top
  glove_position = system.total_height / 2 := by
  sorry


end glove_at_midpoint_l1759_175974


namespace jack_grassy_time_is_six_l1759_175910

/-- Represents the race up the hill -/
structure HillRace where
  jackSandyTime : ℝ
  jackSpeedIncrease : ℝ
  jillTotalTime : ℝ
  jillFinishDifference : ℝ

/-- Calculates Jack's time on the grassy second half of the hill -/
def jackGrassyTime (race : HillRace) : ℝ :=
  race.jillTotalTime - race.jillFinishDifference - race.jackSandyTime

/-- Theorem stating that Jack's time on the grassy second half is 6 seconds -/
theorem jack_grassy_time_is_six (race : HillRace) 
  (h1 : race.jackSandyTime = 19)
  (h2 : race.jackSpeedIncrease = 0.25)
  (h3 : race.jillTotalTime = 32)
  (h4 : race.jillFinishDifference = 7) :
  jackGrassyTime race = 6 := by
  sorry

#check jack_grassy_time_is_six

end jack_grassy_time_is_six_l1759_175910


namespace number_satisfying_condition_l1759_175948

theorem number_satisfying_condition (x : ℝ) : x = 40 ↔ 0.65 * x = 0.05 * 60 + 23 := by
  sorry

end number_satisfying_condition_l1759_175948


namespace increasing_magnitude_l1759_175982

theorem increasing_magnitude (x : ℝ) (h : 1 < x ∧ x < 1.1) : x < x^x ∧ x^x < x^(x^x) := by
  sorry

end increasing_magnitude_l1759_175982


namespace range_of_a_l1759_175980

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / (a * Real.log (x + 1))

/-- The function g(x) defined in the problem -/
def g (x : ℝ) : ℝ := x^2 * (x - 1)^2

/-- The helper function h(x) used in the proof -/
noncomputable def h (x : ℝ) : ℝ := x^2 / Real.log x

theorem range_of_a :
  ∃ (a : ℝ), ∀ (x₁ x₂ : ℝ),
    (Real.exp (1/4) - 1 < x₁ ∧ x₁ < Real.exp 1 - 1) →
    x₂ < 0 →
    x₂ = -x₁ →
    (f a x₁) * (-x₁) + (g x₂) * x₁ = 0 →
    2 * Real.exp 1 ≤ a ∧ a < Real.exp 2 :=
sorry

end range_of_a_l1759_175980


namespace quadratic_sum_of_coefficients_l1759_175970

/-- A quadratic function f(x) = ax^2 + bx + c with roots at -2 and 4, and maximum value 54 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum_of_coefficients 
  (a b c : ℝ) 
  (h1 : QuadraticFunction a b c (-2) = 0)
  (h2 : QuadraticFunction a b c 4 = 0)
  (h3 : ∀ x, QuadraticFunction a b c x ≤ 54)
  (h4 : ∃ x, QuadraticFunction a b c x = 54) :
  a + b + c = 54 := by
  sorry

end quadratic_sum_of_coefficients_l1759_175970


namespace equation_solution_denominator_never_zero_l1759_175965

theorem equation_solution (x : ℝ) : 
  (x + 5) / (x^2 + 4*x + 10) = 0 ↔ x = -5 :=
by sorry

theorem denominator_never_zero (x : ℝ) : 
  x^2 + 4*x + 10 ≠ 0 :=
by sorry

end equation_solution_denominator_never_zero_l1759_175965


namespace monotone_increasing_implies_a_geq_one_third_l1759_175936

/-- A cubic function f(x) = ax^3 - x^2 + x - 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x - 5

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 2 * x + 1

/-- f is monotonically increasing if its derivative is non-negative for all x -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x : ℝ, f_deriv a x ≥ 0

theorem monotone_increasing_implies_a_geq_one_third :
  ∀ a : ℝ, is_monotone_increasing a → a ≥ 1/3 :=
sorry

end monotone_increasing_implies_a_geq_one_third_l1759_175936


namespace rabbit_count_l1759_175976

theorem rabbit_count (total_legs : ℕ) (rabbit_chicken_diff : ℕ) : 
  total_legs = 250 → rabbit_chicken_diff = 53 → 
  ∃ (rabbits : ℕ), 
    rabbits + rabbit_chicken_diff = total_legs / 2 ∧
    4 * rabbits + 2 * (rabbits + rabbit_chicken_diff) = total_legs ∧
    rabbits = 24 := by
  sorry

end rabbit_count_l1759_175976


namespace smallest_difference_of_valid_units_digits_l1759_175920

def is_multiple_of_five (n : ℕ) : Prop := ∃ k, n = 5 * k

def valid_units_digit (x : ℕ) : Prop :=
  x < 10 ∧ is_multiple_of_five (520 + x)

theorem smallest_difference_of_valid_units_digits :
  ∃ (a b : ℕ), valid_units_digit a ∧ valid_units_digit b ∧
  (∀ (c d : ℕ), valid_units_digit c → valid_units_digit d →
    a - b ≤ c - d ∨ b - a ≤ c - d) ∧
  a - b = 5 ∨ b - a = 5 :=
sorry

end smallest_difference_of_valid_units_digits_l1759_175920


namespace peter_class_size_l1759_175955

/-- The number of students in Peter's class with 2 hands each -/
def students_with_two_hands : ℕ := 10

/-- The number of students in Peter's class with 1 hand each -/
def students_with_one_hand : ℕ := 3

/-- The number of students in Peter's class with 3 hands each -/
def students_with_three_hands : ℕ := 1

/-- The total number of hands in the class excluding Peter's -/
def total_hands_excluding_peter : ℕ := 20

/-- The number of hands Peter has (assumed to be typical) -/
def peter_hands : ℕ := 2

/-- The total number of students in Peter's class, including Peter -/
def total_students : ℕ := 14

theorem peter_class_size :
  (students_with_two_hands * 2 + 
   students_with_one_hand * 1 + 
   students_with_three_hands * 3 + 
   peter_hands) / 2 = total_students := by sorry

end peter_class_size_l1759_175955


namespace nth_term_is_3012_l1759_175927

def arithmetic_sequence (a₁ a₂ a₃ : ℚ) : ℕ → ℚ :=
  λ n => a₁ + (n - 1) * (a₂ - a₁)

theorem nth_term_is_3012 (x : ℚ) :
  let a₁ := 3 * x - 4
  let a₂ := 7 * x - 14
  let a₃ := 4 * x + 6
  (∃ n : ℕ, arithmetic_sequence a₁ a₂ a₃ n = 3012) →
  (∃ n : ℕ, n = 392 ∧ arithmetic_sequence a₁ a₂ a₃ n = 3012) :=
by
  sorry

#check nth_term_is_3012

end nth_term_is_3012_l1759_175927


namespace initial_number_solution_l1759_175987

theorem initial_number_solution : ∃ x : ℤ, x - 12 * 3 * 2 = 9938 ∧ x = 10010 := by
  sorry

end initial_number_solution_l1759_175987


namespace sqrt_sum_fraction_simplification_l1759_175968

theorem sqrt_sum_fraction_simplification :
  Real.sqrt ((9 : ℝ) / 16 + 16 / 81) = Real.sqrt 985 / 36 := by
  sorry

end sqrt_sum_fraction_simplification_l1759_175968


namespace langsley_commute_time_l1759_175986

def pickup_time : Nat := 6 * 60  -- 6:00 a.m. in minutes since midnight
def first_station_travel_time : Nat := 40  -- 40 minutes
def work_arrival_time : Nat := 9 * 60  -- 9:00 a.m. in minutes since midnight

theorem langsley_commute_time :
  work_arrival_time - (pickup_time + first_station_travel_time) = 140 := by
  sorry

end langsley_commute_time_l1759_175986


namespace smallest_cube_ending_432_l1759_175949

theorem smallest_cube_ending_432 : 
  ∀ n : ℕ+, n.val^3 % 1000 = 432 → n.val ≥ 138 := by sorry

end smallest_cube_ending_432_l1759_175949


namespace symmetry_of_functions_l1759_175932

theorem symmetry_of_functions (f : ℝ → ℝ) : 
  ∀ x y : ℝ, f (1 - x) = y ↔ f (x - 1) = y :=
by sorry

end symmetry_of_functions_l1759_175932


namespace angle_theorem_l1759_175905

theorem angle_theorem (α β θ : Real) 
  (h1 : 0 < α ∧ α < 60)
  (h2 : 0 < β ∧ β < 60)
  (h3 : 0 < θ ∧ θ < 60)
  (h4 : α + β = 2 * θ)
  (h5 : Real.sin α * Real.sin β * Real.sin θ = 
        Real.sin (60 - α) * Real.sin (60 - β) * Real.sin (60 - θ)) :
  θ = 30 := by sorry

end angle_theorem_l1759_175905


namespace imaginary_part_of_complex_expression_l1759_175984

theorem imaginary_part_of_complex_expression (i : ℂ) (h : i^2 = -1) :
  Complex.im ((3 + i) / i^2 * i) = 1 := by sorry

end imaginary_part_of_complex_expression_l1759_175984


namespace unique_cube_difference_l1759_175933

theorem unique_cube_difference (n : ℕ+) : 
  (∃ x y : ℕ+, (837 + n : ℕ) = y^3 ∧ (837 - n : ℕ) = x^3) ↔ n = 494 := by
  sorry

end unique_cube_difference_l1759_175933


namespace ernesto_age_proof_l1759_175981

/-- Ernesto's current age -/
def ernesto_age : ℕ := 11

/-- Jayden's current age -/
def jayden_age : ℕ := 4

/-- The number of years in the future when the age comparison is made -/
def years_future : ℕ := 3

theorem ernesto_age_proof :
  ernesto_age = 11 ∧
  jayden_age = 4 ∧
  jayden_age + years_future = (ernesto_age + years_future) / 2 :=
by sorry

end ernesto_age_proof_l1759_175981


namespace solve_equation_l1759_175921

theorem solve_equation : ∃ x : ℝ, 2*x + 3*x = 600 - (4*x + 6*x) ∧ x = 40 := by
  sorry

end solve_equation_l1759_175921


namespace difference_of_squares_l1759_175915

theorem difference_of_squares (x y : ℝ) 
  (sum_eq : x + y = 24) 
  (diff_eq : x - y = 8) : 
  x^2 - y^2 = 192 := by
sorry

end difference_of_squares_l1759_175915


namespace petya_vasya_meet_at_64_l1759_175972

/-- The number of lampposts along the alley -/
def num_lampposts : ℕ := 100

/-- The lamppost number where Petya is observed -/
def petya_observed : ℕ := 22

/-- The lamppost number where Vasya is observed -/
def vasya_observed : ℕ := 88

/-- The function to calculate the meeting point of Petya and Vasya -/
def meeting_point : ℕ := sorry

/-- Theorem stating that Petya and Vasya meet at lamppost 64 -/
theorem petya_vasya_meet_at_64 : meeting_point = 64 := by sorry

end petya_vasya_meet_at_64_l1759_175972


namespace percentage_calculation_l1759_175942

theorem percentage_calculation (P : ℝ) : 
  0.15 * 0.30 * P * 5600 = 126 → P = 0.5 := by
  sorry

end percentage_calculation_l1759_175942


namespace three_boxes_of_five_balls_l1759_175953

/-- Calculates the total number of balls given the number of boxes and balls per box -/
def totalBalls (numBoxes : ℕ) (ballsPerBox : ℕ) : ℕ :=
  numBoxes * ballsPerBox

/-- Proves that the total number of balls is 15 when there are 3 boxes with 5 balls each -/
theorem three_boxes_of_five_balls :
  totalBalls 3 5 = 15 := by
  sorry

end three_boxes_of_five_balls_l1759_175953


namespace sin_cos_sum_zero_l1759_175908

theorem sin_cos_sum_zero : 
  Real.sin (35 * π / 6) + Real.cos (-11 * π / 3) = 0 := by
  sorry

end sin_cos_sum_zero_l1759_175908


namespace average_weight_of_boys_l1759_175939

theorem average_weight_of_boys (group1_count : ℕ) (group1_avg : ℚ) 
  (group2_count : ℕ) (group2_avg : ℚ) : 
  group1_count = 16 → 
  group1_avg = 50.25 → 
  group2_count = 8 → 
  group2_avg = 45.15 → 
  let total_count := group1_count + group2_count
  let total_weight := group1_count * group1_avg + group2_count * group2_avg
  total_weight / total_count = 48.55 := by
sorry

end average_weight_of_boys_l1759_175939


namespace prob_three_two_digit_out_of_five_l1759_175924

/-- A 12-sided die with numbers from 1 to 12 -/
def TwelveSidedDie : Type := Fin 12

/-- The probability of rolling a two-digit number on a single 12-sided die -/
def prob_two_digit : ℚ := 1 / 4

/-- The probability of rolling a one-digit number on a single 12-sided die -/
def prob_one_digit : ℚ := 3 / 4

/-- The number of 12-sided dice rolled -/
def num_dice : ℕ := 5

/-- The number of dice required to show a two-digit number -/
def required_two_digit : ℕ := 3

/-- Theorem stating the probability of exactly 3 out of 5 12-sided dice showing a two-digit number -/
theorem prob_three_two_digit_out_of_five :
  (Nat.choose num_dice required_two_digit : ℚ) *
  (prob_two_digit ^ required_two_digit) *
  (prob_one_digit ^ (num_dice - required_two_digit)) = 45 / 512 := by
  sorry

end prob_three_two_digit_out_of_five_l1759_175924


namespace employee_y_pay_l1759_175999

/-- Represents the weekly pay of employees x, y, and z -/
structure EmployeePay where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the total pay for all employees -/
def totalPay (pay : EmployeePay) : ℝ :=
  pay.x + pay.y + pay.z

/-- Theorem: Given the conditions, employee y's pay is 478.125 -/
theorem employee_y_pay :
  ∀ (pay : EmployeePay),
    totalPay pay = 1550 →
    pay.x = 1.2 * pay.y →
    pay.z = pay.y - 30 + 50 →
    pay.y = 478.125 := by
  sorry


end employee_y_pay_l1759_175999


namespace parallelepiped_count_l1759_175991

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A type representing a set of four points in 3D space -/
structure FourPoints where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D
  p4 : Point3D

/-- Predicate to check if four points are coplanar -/
def areCoplanar (points : FourPoints) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∧
    a * points.p1.x + b * points.p1.y + c * points.p1.z + d = 0 ∧
    a * points.p2.x + b * points.p2.y + c * points.p2.z + d = 0 ∧
    a * points.p3.x + b * points.p3.y + c * points.p3.z + d = 0 ∧
    a * points.p4.x + b * points.p4.y + c * points.p4.z + d = 0

/-- Function to count the number of distinct parallelepipeds -/
def countParallelepipeds (points : FourPoints) : ℕ :=
  sorry -- Implementation details omitted

/-- Theorem stating that the number of distinct parallelepipeds is 29 -/
theorem parallelepiped_count (points : FourPoints) 
  (h : ¬ areCoplanar points) : countParallelepipeds points = 29 := by
  sorry

end parallelepiped_count_l1759_175991


namespace area_of_triangle_MEF_l1759_175943

-- Define the circle P
def circle_P : Real := 10

-- Define the chord EF
def chord_EF : Real := 12

-- Define the segment MQ
def segment_MQ : Real := 20

-- Define the perpendicular distance from P to EF
def perpendicular_distance : Real := 8

-- Theorem statement
theorem area_of_triangle_MEF :
  let radius : Real := circle_P
  let chord_length : Real := chord_EF
  let height : Real := perpendicular_distance
  (1/2 : Real) * chord_length * height = 48 := by sorry

end area_of_triangle_MEF_l1759_175943


namespace modulus_of_complex_number_l1759_175926

theorem modulus_of_complex_number (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end modulus_of_complex_number_l1759_175926


namespace perpendicular_parallel_theorem_l1759_175944

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_theorem 
  (m n l : Line) (α β γ : Plane) 
  (h1 : m ≠ n ∧ m ≠ l ∧ n ≠ l) 
  (h2 : α ≠ β ∧ α ≠ γ ∧ β ≠ γ) :
  perpendicularLP m α → parallelLP n β → parallelPP α β → perpendicular m n :=
by sorry

end perpendicular_parallel_theorem_l1759_175944


namespace total_students_is_880_l1759_175947

/-- The total number of students at the college -/
def total_students : ℕ := 880

/-- The fraction of students enrolled in biology classes -/
def biology_enrollment_rate : ℚ := 35 / 100

/-- The number of students not enrolled in a biology class -/
def students_not_in_biology : ℕ := 572

/-- Theorem stating that the total number of students is 880 -/
theorem total_students_is_880 :
  (1 - biology_enrollment_rate) * total_students = students_not_in_biology :=
sorry

end total_students_is_880_l1759_175947


namespace intersection_M_N_l1759_175911

def M : Set ℤ := {-1, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = x^2}

theorem intersection_M_N : M ∩ N = {1} := by
  sorry

end intersection_M_N_l1759_175911


namespace tax_deduction_proof_l1759_175951

/-- Represents the hourly wage in dollars -/
def hourly_wage : ℝ := 25

/-- Represents the local tax rate as a decimal -/
def tax_rate : ℝ := 0.025

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℝ := 100

/-- Calculates the tax deduction in cents -/
def tax_deduction_cents : ℝ := hourly_wage * cents_per_dollar * tax_rate

theorem tax_deduction_proof : tax_deduction_cents = 62.5 := by
  sorry

end tax_deduction_proof_l1759_175951


namespace intersection_equality_implies_a_in_range_l1759_175956

open Set

def A (a : ℝ) : Set ℝ := {x | |x - a| < 2}
def B : Set ℝ := {x | (2*x - 1) / (x + 2) < 1}

theorem intersection_equality_implies_a_in_range (a : ℝ) :
  A a ∩ B = A a → a ∈ Set.Icc 0 1 := by
  sorry

end intersection_equality_implies_a_in_range_l1759_175956


namespace quadratic_trinomial_factorization_l1759_175938

theorem quadratic_trinomial_factorization (p q : ℝ) (x : ℝ) :
  x^2 + (p + q)*x + p*q = (x + p)*(x + q) := by sorry

end quadratic_trinomial_factorization_l1759_175938


namespace fraction_equation_solution_l1759_175909

/-- Represents a four-digit number in the form 28a3 where a is a digit -/
def fourDigitNumber (a : ℕ) : ℕ := 2803 + 100 * a

/-- The denominator of the original fraction -/
def denominator : ℕ := 7276

/-- Theorem stating that 641 is the solution to the fraction equation -/
theorem fraction_equation_solution :
  ∃ (a : ℕ), a < 10 ∧ 
  (fourDigitNumber a - 641) * 7 = 2 * (denominator + 641) := by
  sorry

end fraction_equation_solution_l1759_175909


namespace power_between_n_and_2n_smallest_m_s_for_2_and_3_l1759_175969

theorem power_between_n_and_2n (s : ℕ) (hs : s > 1) :
  ∃ (m_s : ℕ), ∀ (n : ℕ), n ≥ m_s → ∃ (k : ℕ), n < k^s ∧ k^s < 2*n :=
by sorry

theorem smallest_m_s_for_2_and_3 :
  (∃ (m_2 : ℕ), ∀ (n : ℕ), n ≥ m_2 → ∃ (k : ℕ), n < k^2 ∧ k^2 < 2*n) ∧
  (∃ (m_3 : ℕ), ∀ (n : ℕ), n ≥ m_3 → ∃ (k : ℕ), n < k^3 ∧ k^3 < 2*n) ∧
  (∀ (m_2' : ℕ), m_2' < 5 → ∃ (n : ℕ), n ≥ m_2' ∧ ∀ (k : ℕ), n ≥ k^2 ∨ k^2 ≥ 2*n) ∧
  (∀ (m_3' : ℕ), m_3' < 33 → ∃ (n : ℕ), n ≥ m_3' ∧ ∀ (k : ℕ), n ≥ k^3 ∨ k^3 ≥ 2*n) :=
by sorry

end power_between_n_and_2n_smallest_m_s_for_2_and_3_l1759_175969


namespace complex_modulus_problem_l1759_175903

theorem complex_modulus_problem (z : ℂ) (h : (8 + 6*I)*z = 5 + 12*I) : 
  Complex.abs z = 13/10 := by sorry

end complex_modulus_problem_l1759_175903


namespace min_value_expression_min_value_achievable_l1759_175963

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 9) (h_rel : y = 2 * x) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 6 :=
by sorry

theorem min_value_achievable (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 9) (h_rel : y = 2 * x) :
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ + y₀ + z₀ = 9 ∧ y₀ = 2 * x₀ ∧
    (x₀^2 + y₀^2) / (x₀ + y₀) + (x₀^2 + z₀^2) / (x₀ + z₀) + (y₀^2 + z₀^2) / (y₀ + z₀) = 6 :=
by sorry

end min_value_expression_min_value_achievable_l1759_175963


namespace gear_q_revolutions_per_minute_l1759_175916

/-- The number of revolutions per minute for gear p -/
def p_rev_per_min : ℚ := 10

/-- The number of seconds in the given time interval -/
def time_interval : ℚ := 10

/-- The additional revolutions gear q makes compared to gear p in the given time interval -/
def additional_rev : ℚ := 5

/-- The number of seconds in a minute -/
def seconds_per_minute : ℚ := 60

theorem gear_q_revolutions_per_minute :
  let p_rev_in_interval := p_rev_per_min * time_interval / seconds_per_minute
  let q_rev_in_interval := p_rev_in_interval + additional_rev
  let q_rev_per_min := q_rev_in_interval * seconds_per_minute / time_interval
  q_rev_per_min = 40 := by
  sorry

end gear_q_revolutions_per_minute_l1759_175916
