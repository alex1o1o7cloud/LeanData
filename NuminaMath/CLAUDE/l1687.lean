import Mathlib

namespace chess_tournament_schedules_l1687_168794

/-- Represents the number of players from each school -/
def num_players : ℕ := 4

/-- Represents the number of rounds in the tournament -/
def num_rounds : ℕ := 4

/-- Represents the number of games per round -/
def games_per_round : ℕ := 4

/-- Calculates the total number of games in the tournament -/
def total_games : ℕ := num_players * num_players

/-- Theorem stating the number of ways to schedule the chess tournament -/
theorem chess_tournament_schedules : 
  (num_rounds.factorial * (games_per_round.factorial ^ num_rounds)) = 7962624 := by
  sorry

end chess_tournament_schedules_l1687_168794


namespace lizard_eye_difference_l1687_168728

theorem lizard_eye_difference : ∀ (E W S : ℕ),
  E = 3 →
  W = 3 * E →
  S = 7 * W →
  S + W - E = 69 :=
by
  sorry

end lizard_eye_difference_l1687_168728


namespace equal_area_division_l1687_168772

/-- Represents a shape on a grid -/
structure GridShape where
  area : ℝ
  mk_area_pos : area > 0

/-- Represents a line on a grid -/
structure GridLine where
  distance_from_origin : ℝ

/-- Represents the division of a shape by a line -/
def divides_equally (s : GridShape) (l : GridLine) : Prop :=
  ∃ (area1 area2 : ℝ), 
    area1 > 0 ∧ 
    area2 > 0 ∧ 
    area1 = area2 ∧ 
    area1 + area2 = s.area

/-- The main theorem -/
theorem equal_area_division 
  (gray_shape : GridShape) 
  (h_area : gray_shape.area = 10) 
  (mo : GridLine) 
  (parallel_line : GridLine) 
  (h_distance : parallel_line.distance_from_origin = mo.distance_from_origin + 2.6) :
  divides_equally gray_shape parallel_line := by
  sorry

end equal_area_division_l1687_168772


namespace simplify_expression_l1687_168762

theorem simplify_expression (x : ℝ) (h : x^2 ≠ 1) :
  Real.sqrt (1 + ((x^4 + 1) / (2 * x^2))^2) = (Real.sqrt (x^8 + 6 * x^4 + 1)) / (2 * x^2) :=
by sorry

end simplify_expression_l1687_168762


namespace x_value_l1687_168716

theorem x_value (x : ℚ) 
  (eq1 : 9 * x^2 + 8 * x - 1 = 0) 
  (eq2 : 27 * x^2 + 65 * x - 8 = 0) : 
  x = 1/9 := by
sorry

end x_value_l1687_168716


namespace total_pizza_weight_l1687_168722

/-- Represents the weight of a pizza with toppings -/
structure Pizza where
  base : Nat
  toppings : List Nat

/-- Calculates the total weight of a pizza -/
def totalWeight (p : Pizza) : Nat :=
  p.base + p.toppings.sum

/-- Rachel's pizza -/
def rachelPizza : Pizza :=
  { base := 400
  , toppings := [100, 50, 60] }

/-- Bella's pizza -/
def bellaPizza : Pizza :=
  { base := 350
  , toppings := [75, 55, 35] }

/-- Theorem: The total weight of Rachel's and Bella's pizzas is 1125 grams -/
theorem total_pizza_weight :
  totalWeight rachelPizza + totalWeight bellaPizza = 1125 := by
  sorry

end total_pizza_weight_l1687_168722


namespace gloria_money_calculation_l1687_168724

def combined_quarters_and_dimes (total_quarters : ℕ) (total_dimes : ℕ) : ℕ :=
  let quarters_put_aside := (2 * total_quarters) / 5
  let remaining_quarters := total_quarters - quarters_put_aside
  remaining_quarters + total_dimes

theorem gloria_money_calculation :
  ∀ (total_quarters : ℕ) (total_dimes : ℕ),
    total_dimes = 5 * total_quarters →
    total_dimes = 350 →
    combined_quarters_and_dimes total_quarters total_dimes = 392 :=
by
  sorry

end gloria_money_calculation_l1687_168724


namespace second_child_birth_year_l1687_168715

/-- 
Given a couple married in 1980 with two children, one born in 1982 and the other
in an unknown year, if their combined ages equal the years of marriage in 1986,
then the second child was born in 1992.
-/
theorem second_child_birth_year 
  (marriage_year : Nat) 
  (first_child_birth_year : Nat) 
  (second_child_birth_year : Nat) 
  (h1 : marriage_year = 1980)
  (h2 : first_child_birth_year = 1982)
  (h3 : (1986 - first_child_birth_year) + (1986 - second_child_birth_year) + (1986 - marriage_year) = 1986) :
  second_child_birth_year = 1992 := by
sorry

end second_child_birth_year_l1687_168715


namespace fraction_simplification_l1687_168799

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) :
  a / (a - b) - b / (b - a) = (a + b) / (a - b) := by
  sorry

end fraction_simplification_l1687_168799


namespace isosceles_right_triangle_distance_l1687_168737

theorem isosceles_right_triangle_distance (a : ℝ) (h : a = 8) :
  Real.sqrt (a^2 + a^2) = a * Real.sqrt 2 :=
by sorry

end isosceles_right_triangle_distance_l1687_168737


namespace product_of_fractions_l1687_168700

theorem product_of_fractions : 
  (8 / 4) * (14 / 7) * (20 / 10) * (25 / 50) * (9 / 18) * (12 / 6) * (21 / 42) * (16 / 8) = 8 := by
  sorry

end product_of_fractions_l1687_168700


namespace greatest_n_with_divisibility_conditions_l1687_168775

theorem greatest_n_with_divisibility_conditions :
  ∃ (n : ℕ), n < 1000 ∧
  (Int.floor (Real.sqrt n) - 2 : ℤ) ∣ (n - 4 : ℤ) ∧
  (Int.floor (Real.sqrt n) + 2 : ℤ) ∣ (n + 4 : ℤ) ∧
  (∀ (m : ℕ), m < 1000 →
    (Int.floor (Real.sqrt m) - 2 : ℤ) ∣ (m - 4 : ℤ) →
    (Int.floor (Real.sqrt m) + 2 : ℤ) ∣ (m + 4 : ℤ) →
    m ≤ n) ∧
  n = 956 :=
sorry

end greatest_n_with_divisibility_conditions_l1687_168775


namespace find_T_l1687_168767

theorem find_T (S : ℚ) (T : ℚ) (h1 : (1/4) * (1/6) * T = (1/2) * (1/8) * S) (h2 : S = 64) :
  T = 96 := by
  sorry

end find_T_l1687_168767


namespace triangle_area_is_one_l1687_168771

-- Define the complex number z on the unit circle
def z : ℂ :=
  sorry

-- Define the condition |z| = 1
axiom z_on_unit_circle : Complex.abs z = 1

-- Define the vertices of the triangle
def vertex1 : ℂ := z
def vertex2 : ℂ := z^2
def vertex3 : ℂ := z + z^2

-- Define the function to calculate the area of a triangle given three complex points
def triangle_area (a b c : ℂ) : ℝ :=
  sorry

-- State the theorem
theorem triangle_area_is_one :
  triangle_area vertex1 vertex2 vertex3 = 1 :=
sorry

end triangle_area_is_one_l1687_168771


namespace mikes_investment_interest_l1687_168781

/-- Calculates the total interest earned from a two-part investment --/
def total_interest (total_investment : ℚ) (amount_at_lower_rate : ℚ) (lower_rate : ℚ) (higher_rate : ℚ) : ℚ :=
  let amount_at_higher_rate := total_investment - amount_at_lower_rate
  let interest_lower := amount_at_lower_rate * lower_rate
  let interest_higher := amount_at_higher_rate * higher_rate
  interest_lower + interest_higher

/-- Theorem stating that Mike's investment yields $624 in interest --/
theorem mikes_investment_interest :
  total_interest 6000 1800 (9/100) (11/100) = 624 := by
  sorry

end mikes_investment_interest_l1687_168781


namespace equation_infinite_solutions_l1687_168798

theorem equation_infinite_solutions (c : ℝ) : 
  (∀ y : ℝ, 3 * (5 + 2 * c * y) = 18 * y + 15) ↔ c = 3 := by
  sorry

end equation_infinite_solutions_l1687_168798


namespace parabola_transformation_l1687_168777

/-- A parabola is defined by its coefficient and horizontal shift -/
structure Parabola where
  a : ℝ
  h : ℝ

/-- The equation of a parabola y = a(x-h)^2 -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h)^2

/-- The transformation that shifts a parabola horizontally -/
def horizontal_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, h := p.h + shift }

theorem parabola_transformation (p1 p2 : Parabola) :
  p1.a = 2 ∧ p1.h = 0 ∧ p2.a = 2 ∧ p2.h = 3 →
  ∃ (shift : ℝ), shift = 3 ∧ horizontal_shift p1 shift = p2 :=
sorry

end parabola_transformation_l1687_168777


namespace imaginary_part_of_z_l1687_168746

/-- The imaginary part of (1+2i) / (1-i)² is 1/2 -/
theorem imaginary_part_of_z : Complex.im ((1 + 2*Complex.I) / (1 - Complex.I)^2) = 1/2 := by
  sorry

end imaginary_part_of_z_l1687_168746


namespace product_equals_fraction_l1687_168770

/-- The repeating decimal 0.1357̄ as a rational number -/
def s : ℚ := 1357 / 9999

/-- The product of 0.1357̄ and 7 -/
def product : ℚ := 7 * s

theorem product_equals_fraction : product = 9499 / 9999 := by
  sorry

end product_equals_fraction_l1687_168770


namespace three_pumps_fill_time_l1687_168745

-- Define the pumps and tank
variable (T : ℝ) -- Volume of the tank
variable (X Y Z : ℝ) -- Rates at which pumps X, Y, and Z fill the tank

-- Define the conditions
axiom cond1 : T = 3 * (X + Y)
axiom cond2 : T = 6 * (X + Z)
axiom cond3 : T = 4.5 * (Y + Z)

-- Define the theorem
theorem three_pumps_fill_time : 
  T / (X + Y + Z) = 36 / 13 := by sorry

end three_pumps_fill_time_l1687_168745


namespace sin_squared_alpha_plus_5pi_12_l1687_168721

theorem sin_squared_alpha_plus_5pi_12 (α : ℝ) 
  (h : Real.sin (2 * Real.pi / 3 - 2 * α) = 3 / 5) : 
  Real.sin (α + 5 * Real.pi / 12) ^ 2 = 4 / 5 := by
  sorry

end sin_squared_alpha_plus_5pi_12_l1687_168721


namespace append_two_to_three_digit_number_l1687_168713

/-- Given a three-digit number with digits h, t, and u, appending 2 results in 1000h + 100t + 10u + 2 -/
theorem append_two_to_three_digit_number (h t u : ℕ) :
  let original := 100 * h + 10 * t + u
  let appended := original * 10 + 2
  appended = 1000 * h + 100 * t + 10 * u + 2 := by
sorry

end append_two_to_three_digit_number_l1687_168713


namespace triangle_abc_properties_l1687_168790

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Conditions
  C = π / 3 →
  b = 8 →
  (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3 →
  -- Definitions
  a > 0 →
  b > 0 →
  c > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  -- Triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Theorem statements
  c = 7 ∧ Real.cos (B - C) = 13/14 := by
  sorry

end triangle_abc_properties_l1687_168790


namespace real_axis_length_l1687_168753

/-- Hyperbola C with center at origin and foci on x-axis -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop
  center_origin : equation 0 0
  foci_on_x_axis : ∀ y, ¬(∃ x ≠ 0, equation x y ∧ equation (-x) y)

/-- Parabola with equation y² = 16x -/
def Parabola : ℝ → ℝ → Prop :=
  λ x y => y^2 = 16 * x

/-- Directrix of the parabola y² = 16x -/
def Directrix : ℝ → Prop :=
  λ x => x = -4

/-- Points A and B where hyperbola C intersects the directrix -/
structure IntersectionPoints (C : Hyperbola) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  on_directrix : Directrix A.1 ∧ Directrix B.1
  on_hyperbola : C.equation A.1 A.2 ∧ C.equation B.1 B.2
  distance : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 3

/-- The theorem to be proved -/
theorem real_axis_length (C : Hyperbola) (AB : IntersectionPoints C) :
  ∃ a : ℝ, a = 4 ∧ ∀ x y, C.equation x y ↔ x^2 / a^2 - y^2 / (a^2 - 4) = 1 :=
sorry

end real_axis_length_l1687_168753


namespace expression_evaluation_l1687_168750

theorem expression_evaluation (x y z : ℝ) :
  (x + (y - z)) - ((x + z) - y) = 2*y - 2*z := by sorry

end expression_evaluation_l1687_168750


namespace pass_probability_theorem_l1687_168703

/-- A highway with n intersecting roads. -/
structure Highway where
  n : ℕ
  k : ℕ
  h1 : 0 < n
  h2 : k ≤ n

/-- The probability of a car passing through the k-th intersection on a highway. -/
def pass_probability (h : Highway) : ℚ :=
  (2 * h.k * h.n - 2 * h.k^2 + 2 * h.k - 1) / (h.n^2 : ℚ)

/-- Theorem stating the probability of a car passing through the k-th intersection. -/
theorem pass_probability_theorem (h : Highway) :
  pass_probability h = (2 * h.k * h.n - 2 * h.k^2 + 2 * h.k - 1) / (h.n^2 : ℚ) := by
  sorry

end pass_probability_theorem_l1687_168703


namespace function_problem_l1687_168732

theorem function_problem (f : ℝ → ℝ) (a b c : ℝ) 
  (h_inv : Function.Injective f)
  (h1 : f a = b)
  (h2 : f b = 5)
  (h3 : f c = 3)
  (h4 : c = a + 1) :
  a - b = -2 := by
  sorry

end function_problem_l1687_168732


namespace investment_calculation_correct_l1687_168763

/-- Calculates the total investment given share details and dividend income -/
def calculate_investment (face_value : ℚ) (quoted_price : ℚ) (dividend_rate : ℚ) (annual_income : ℚ) : ℚ :=
  let dividend_per_share := (dividend_rate / 100) * face_value
  let num_shares := annual_income / dividend_per_share
  num_shares * quoted_price

/-- Theorem stating that the investment calculation is correct for the given problem -/
theorem investment_calculation_correct :
  calculate_investment 10 8.25 12 648 = 4455 := by
  sorry

end investment_calculation_correct_l1687_168763


namespace inequality_proof_l1687_168796

theorem inequality_proof (a x : ℝ) (h1 : 0 < x) (h2 : x < a) :
  (a - x)^6 - 3*a*(a - x)^5 + 5/2 * a^2 * (a - x)^4 - 1/2 * a^4 * (a - x)^2 < 0 := by
  sorry

end inequality_proof_l1687_168796


namespace exists_circle_with_n_grid_points_l1687_168729

/-- A grid point is a point with integer coordinates -/
def GridPoint : Type := ℤ × ℤ

/-- A circle is defined by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Count the number of grid points within a circle -/
def countGridPointsInCircle (c : Circle) : ℕ :=
  sorry

/-- Main theorem: For any positive integer n, there exists a circle with exactly n grid points -/
theorem exists_circle_with_n_grid_points (n : ℕ) (hn : n > 0) :
  ∃ (c : Circle), countGridPointsInCircle c = n :=
sorry

end exists_circle_with_n_grid_points_l1687_168729


namespace relationship_between_3a_3b_4a_l1687_168780

theorem relationship_between_3a_3b_4a (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  3 * b < 3 * a ∧ 3 * a < 4 * a := by
  sorry

end relationship_between_3a_3b_4a_l1687_168780


namespace tan_half_sum_of_angles_l1687_168795

theorem tan_half_sum_of_angles (a b : Real) 
  (h1 : Real.cos a + Real.cos b = 1)
  (h2 : Real.sin a + Real.sin b = 1/2)
  (h3 : Real.tan (a - b) = 1) :
  Real.tan ((a + b) / 2) = 1/2 := by
  sorry

end tan_half_sum_of_angles_l1687_168795


namespace power_of_81_l1687_168765

theorem power_of_81 : (81 : ℝ) ^ (5/4 : ℝ) = 243 := by sorry

end power_of_81_l1687_168765


namespace map_scale_conversion_l1687_168768

/-- Given a map where 15 cm represents 90 km, a 20 cm length represents 120,000 meters -/
theorem map_scale_conversion (map_scale : ℝ) (h : map_scale * 15 = 90) : 
  map_scale * 20 * 1000 = 120000 := by
  sorry

end map_scale_conversion_l1687_168768


namespace car_washing_time_l1687_168701

theorem car_washing_time (x : ℝ) : 
  x > 0 → 
  x + (1/4) * x = 100 → 
  x = 80 :=
by sorry

end car_washing_time_l1687_168701


namespace product_of_slopes_l1687_168743

theorem product_of_slopes (m n : ℝ) : 
  (∃ θ₁ θ₂ : ℝ, θ₁ = 3 * θ₂ ∧ m = Real.tan θ₁ ∧ n = Real.tan θ₂) →  -- L1 makes three times the angle with horizontal as L2
  m = 3 * n →                                                      -- L1 has 3 times the slope of L2
  m ≠ 0 →                                                          -- L1 is not vertical
  m * n = 0 :=                                                     -- Conclusion: mn = 0
by sorry

end product_of_slopes_l1687_168743


namespace smallest_divisible_by_one_to_ten_l1687_168774

theorem smallest_divisible_by_one_to_ten : 
  ∃ n : ℕ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ 
    (∀ m : ℕ, m < n → ∃ j : ℕ, 1 ≤ j ∧ j ≤ 10 ∧ ¬(j ∣ m)) ∧ 
    n = 2520 :=
by sorry

end smallest_divisible_by_one_to_ten_l1687_168774


namespace abc_equality_l1687_168723

theorem abc_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 4 * a * b * c * (a + b + c) = (a + b)^2 * (a + c)^2) :
  a * (a + b + c) = b * c := by
sorry

end abc_equality_l1687_168723


namespace roots_have_unit_modulus_l1687_168766

theorem roots_have_unit_modulus (z : ℂ) : 
  11 * z^10 + 10 * Complex.I * z^9 + 10 * Complex.I * z - 11 = 0 → Complex.abs z = 1 := by
  sorry

end roots_have_unit_modulus_l1687_168766


namespace fraction_change_l1687_168764

theorem fraction_change (original_fraction : ℚ) 
  (numerator_increase : ℚ) (denominator_decrease : ℚ) (new_fraction : ℚ) : 
  original_fraction = 3/4 →
  numerator_increase = 12/100 →
  new_fraction = 6/7 →
  (1 + numerator_increase) * original_fraction / (1 - denominator_decrease/100) = new_fraction →
  denominator_decrease = 6 := by
  sorry

end fraction_change_l1687_168764


namespace minimum_orchestra_size_l1687_168730

theorem minimum_orchestra_size : ∃ n : ℕ, n > 0 ∧ 
  n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧
  ∀ m : ℕ, m > 0 → m % 9 = 0 → m % 10 = 0 → m % 11 = 0 → m ≥ n :=
by
  -- Proof goes here
  sorry

end minimum_orchestra_size_l1687_168730


namespace japanese_students_fraction_l1687_168748

theorem japanese_students_fraction (J : ℚ) (h1 : J > 0) : 
  let S := 3 * J
  let seniors_studying := (1/3) * S
  let juniors_studying := (3/4) * J
  let total_students := S + J
  (seniors_studying + juniors_studying) / total_students = 7/16 := by
sorry

end japanese_students_fraction_l1687_168748


namespace problem_statement_l1687_168718

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : Real.exp a * (1 - Real.log b) = 1) :
  (1 < b ∧ b < Real.exp 1) ∧
  (b - a > 1) ∧
  (a > Real.log b) ∧
  (Real.exp a - Real.log b > 1) := by
  sorry

end problem_statement_l1687_168718


namespace quadratic_roots_average_l1687_168787

theorem quadratic_roots_average (c : ℝ) : 
  ∃ (x₁ x₂ : ℝ), (2 * x₁^2 - 4 * x₁ + c = 0) ∧ 
                 (2 * x₂^2 - 4 * x₂ + c = 0) ∧ 
                 (x₁ ≠ x₂) → 
                 (x₁ + x₂) / 2 = 1 := by
sorry

end quadratic_roots_average_l1687_168787


namespace sanitizer_sales_theorem_l1687_168720

/-- Represents the hand sanitizer sales problem -/
structure SanitizerSales where
  cost : ℝ  -- Cost per bottle in yuan
  initial_price : ℝ  -- Initial selling price per bottle in yuan
  initial_volume : ℝ  -- Initial daily sales volume
  price_sensitivity : ℝ  -- Decrease in sales for every 1 yuan increase in price
  x : ℝ  -- Increase in selling price

/-- Calculates the daily sales volume given the price increase -/
def daily_volume (s : SanitizerSales) : ℝ :=
  s.initial_volume - s.price_sensitivity * s.x

/-- Calculates the profit per bottle given the price increase -/
def profit_per_bottle (s : SanitizerSales) : ℝ :=
  (s.initial_price - s.cost) + s.x

/-- Calculates the daily profit given the price increase -/
def daily_profit (s : SanitizerSales) : ℝ :=
  (daily_volume s) * (profit_per_bottle s)

/-- The main theorem about the sanitizer sales problem -/
theorem sanitizer_sales_theorem (s : SanitizerSales) 
  (h1 : s.cost = 16)
  (h2 : s.initial_price = 20)
  (h3 : s.initial_volume = 60)
  (h4 : s.price_sensitivity = 5) :
  (daily_volume s = 60 - 5 * s.x) ∧
  (profit_per_bottle s = 4 + s.x) ∧
  (daily_profit s = 300 → s.x = 2 ∨ s.x = 6) ∧
  (∃ (max_profit : ℝ), max_profit = 320 ∧ 
    ∀ (y : ℝ), y = daily_profit s → y ≤ max_profit ∧
    (y = max_profit ↔ s.x = 4)) := by
  sorry


end sanitizer_sales_theorem_l1687_168720


namespace inequality_and_equality_condition_l1687_168769

theorem inequality_and_equality_condition (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 ∧
  ((a^2 + b^2) * (c^2 + d^2) = (a*c + b*d)^2 ↔ a*d = b*c) := by
  sorry

end inequality_and_equality_condition_l1687_168769


namespace number_subtraction_problem_l1687_168749

theorem number_subtraction_problem (x : ℝ) : 0.60 * x - 40 = 50 ↔ x = 150 := by
  sorry

end number_subtraction_problem_l1687_168749


namespace cow_count_is_18_l1687_168778

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs for a given AnimalCount -/
def totalLegs (count : AnimalCount) : ℕ :=
  2 * count.ducks + 4 * count.cows

/-- Calculates the total number of heads for a given AnimalCount -/
def totalHeads (count : AnimalCount) : ℕ :=
  count.ducks + count.cows

/-- Theorem stating that the number of cows is 18 given the problem conditions -/
theorem cow_count_is_18 (count : AnimalCount) :
  totalLegs count = 2 * totalHeads count + 36 →
  count.cows = 18 := by
  sorry

#check cow_count_is_18

end cow_count_is_18_l1687_168778


namespace problem_statement_l1687_168710

theorem problem_statement : ∃ x : ℝ, x * (1/2)^2 = 2^3 ∧ x = 32 := by sorry

end problem_statement_l1687_168710


namespace acute_angle_range_l1687_168705

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def is_acute_angle (v w : ℝ × ℝ) : Prop := dot_product v w > 0

def not_same_direction (v w : ℝ × ℝ) : Prop := v.1 / v.2 ≠ w.1 / w.2

theorem acute_angle_range (x : ℝ) :
  is_acute_angle a (b x) ∧ not_same_direction a (b x) ↔ x ∈ Set.Ioo (-8) 2 ∪ Set.Ioi 2 :=
sorry

end acute_angle_range_l1687_168705


namespace wrong_quotient_problem_l1687_168756

theorem wrong_quotient_problem (dividend : ℕ) (correct_divisor wrong_divisor correct_quotient : ℕ) 
  (h1 : dividend % correct_divisor = 0)
  (h2 : correct_divisor = 21)
  (h3 : wrong_divisor = 12)
  (h4 : correct_quotient = 24)
  (h5 : dividend = correct_divisor * correct_quotient) :
  dividend / wrong_divisor = 42 := by
  sorry

end wrong_quotient_problem_l1687_168756


namespace airline_services_overlap_l1687_168757

theorem airline_services_overlap (wireless_percent : Real) (snacks_percent : Real) 
  (wireless_percent_hyp : wireless_percent = 35) 
  (snacks_percent_hyp : snacks_percent = 70) :
  (max_overlap : Real) → max_overlap ≤ 35 ∧ 
  ∃ (overlap : Real), overlap ≤ max_overlap ∧ 
                      overlap ≤ wireless_percent ∧ 
                      overlap ≤ snacks_percent :=
by sorry

end airline_services_overlap_l1687_168757


namespace chess_game_probability_l1687_168714

theorem chess_game_probability (draw_prob win_prob lose_prob : ℚ) : 
  draw_prob = 1/2 →
  win_prob = 1/3 →
  draw_prob + win_prob + lose_prob = 1 →
  lose_prob = 1/6 :=
by
  sorry

end chess_game_probability_l1687_168714


namespace inequality_range_l1687_168785

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x > 2*a*x + a) ↔ a ∈ Set.Ioo (-4 : ℝ) (-1 : ℝ) :=
sorry

end inequality_range_l1687_168785


namespace election_result_l1687_168761

theorem election_result (total_votes : ℕ) (majority : ℕ) (winner_percentage : ℝ) :
  total_votes = 440 →
  majority = 176 →
  winner_percentage * (total_votes : ℝ) / 100 - (100 - winner_percentage) * (total_votes : ℝ) / 100 = majority →
  winner_percentage = 70 := by
sorry

end election_result_l1687_168761


namespace harmonic_mean_of_2_3_6_l1687_168726

theorem harmonic_mean_of_2_3_6 : 
  3 = 3 / (1 / 2 + 1 / 3 + 1 / 6) := by sorry

end harmonic_mean_of_2_3_6_l1687_168726


namespace solution_replacement_fraction_l1687_168782

theorem solution_replacement_fraction (initial_conc : ℚ) (replacement_conc : ℚ) (final_conc : ℚ)
  (h_initial : initial_conc = 60 / 100)
  (h_replacement : replacement_conc = 25 / 100)
  (h_final : final_conc = 35 / 100) :
  let replaced_fraction := (initial_conc - final_conc) / (initial_conc - replacement_conc)
  replaced_fraction = 5 / 7 := by
sorry

end solution_replacement_fraction_l1687_168782


namespace cars_lifted_is_six_l1687_168735

/-- The number of people needed to lift a car -/
def people_per_car : ℕ := 5

/-- The number of people needed to lift a truck -/
def people_per_truck : ℕ := 2 * people_per_car

/-- The number of cars being lifted -/
def cars_lifted : ℕ := 6

/-- The number of trucks being lifted -/
def trucks_lifted : ℕ := 3

/-- Theorem stating that the number of cars being lifted is 6 -/
theorem cars_lifted_is_six : cars_lifted = 6 := by sorry

end cars_lifted_is_six_l1687_168735


namespace michael_matchsticks_l1687_168793

/-- The number of matchstick houses Michael creates -/
def num_houses : ℕ := 30

/-- The number of matchsticks used per house -/
def matchsticks_per_house : ℕ := 10

/-- The total number of matchsticks Michael used -/
def total_matchsticks_used : ℕ := num_houses * matchsticks_per_house

/-- Michael's original number of matchsticks -/
def original_matchsticks : ℕ := 2 * total_matchsticks_used

theorem michael_matchsticks : original_matchsticks = 600 := by
  sorry

end michael_matchsticks_l1687_168793


namespace subset_condition_l1687_168747

def P : Set ℝ := {x | x^2 - 2*x - 3 = 0}
def S (a : ℝ) : Set ℝ := {x | a*x + 2 = 0}

theorem subset_condition (a : ℝ) : S a ⊆ P ↔ a = 0 ∨ a = 2 ∨ a = -2/3 := by
  sorry

end subset_condition_l1687_168747


namespace prob_two_red_cards_standard_deck_l1687_168791

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (red_suits : Nat)
  (black_suits : Nat)

/-- Definition of a standard deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    suits := 4,
    cards_per_suit := 13,
    red_suits := 2,
    black_suits := 2 }

/-- Probability of drawing two red cards in succession -/
def prob_two_red_cards (d : Deck) : Rat :=
  let red_cards := d.red_suits * d.cards_per_suit
  let first_draw := red_cards / d.total_cards
  let second_draw := (red_cards - 1) / (d.total_cards - 1)
  first_draw * second_draw

/-- Theorem: The probability of drawing two red cards in succession from a standard deck is 25/102 -/
theorem prob_two_red_cards_standard_deck :
  prob_two_red_cards standard_deck = 25 / 102 := by
  sorry

end prob_two_red_cards_standard_deck_l1687_168791


namespace quadratic_root_sum_l1687_168760

theorem quadratic_root_sum (p q : ℚ) : 
  (1 - Real.sqrt 3) / 2 = -p / 2 - Real.sqrt ((p / 2) ^ 2 - q) →
  |p| + 2 * |q| = 2 := by
  sorry

end quadratic_root_sum_l1687_168760


namespace initial_machines_count_l1687_168736

/-- The number of bottles produced per minute by the initial number of machines -/
def initial_production_rate : ℕ := 270

/-- The number of machines used in the second scenario -/
def second_scenario_machines : ℕ := 20

/-- The number of bottles produced in the second scenario -/
def second_scenario_production : ℕ := 3600

/-- The time in minutes for the second scenario -/
def second_scenario_time : ℕ := 4

/-- The number of machines running initially -/
def initial_machines : ℕ := 6

theorem initial_machines_count :
  initial_machines * initial_production_rate = second_scenario_machines * (second_scenario_production / second_scenario_time) :=
by sorry

end initial_machines_count_l1687_168736


namespace ark5_ensures_metabolic_energy_needs_l1687_168708

-- Define the Ark5 enzyme
def Ark5 : Type := Unit

-- Define cancer cells
def CancerCell : Type := Unit

-- Define the function that represents the ability to balance energy
def balanceEnergy (a : Ark5) (c : CancerCell) : Prop := sorry

-- Define the function that represents the ability to proliferate without limit
def proliferateWithoutLimit (c : CancerCell) : Prop := sorry

-- Define the function that represents the state of energy scarcity
def energyScarcity : Prop := sorry

-- Define the function that represents cell death due to lack of energy
def dieFromLackOfEnergy (c : CancerCell) : Prop := sorry

-- Define the function that represents ensuring metabolic energy needs
def ensureMetabolicEnergyNeeds (a : Ark5) (c : CancerCell) : Prop := sorry

-- Theorem statement
theorem ark5_ensures_metabolic_energy_needs :
  ∀ (a : Ark5) (c : CancerCell),
    (¬balanceEnergy a c → (energyScarcity → proliferateWithoutLimit c)) ∧
    (¬balanceEnergy a c → (energyScarcity → dieFromLackOfEnergy c)) →
    ensureMetabolicEnergyNeeds a c :=
by
  sorry

end ark5_ensures_metabolic_energy_needs_l1687_168708


namespace inverse_variation_problem_l1687_168719

-- Define the relationship between x and y
def inverse_relation (x y : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ y * x^2 = k

-- State the theorem
theorem inverse_variation_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : inverse_relation x₁ y₁)
  (h₂ : x₁ = 3)
  (h₃ : y₁ = 2)
  (h₄ : y₂ = 18)
  (h₅ : inverse_relation x₂ y₂) :
  x₂ = 1 := by
sorry

end inverse_variation_problem_l1687_168719


namespace summer_camp_probability_l1687_168789

/-- Given a summer camp with 30 kids total, 22 in coding, and 19 in robotics,
    the probability of selecting two kids from different workshops is 32/39. -/
theorem summer_camp_probability (total : ℕ) (coding : ℕ) (robotics : ℕ) 
  (h_total : total = 30)
  (h_coding : coding = 22)
  (h_robotics : robotics = 19) :
  (total.choose 2 - (coding - (coding + robotics - total)).choose 2 - (robotics - (coding + robotics - total)).choose 2) / total.choose 2 = 32 / 39 :=
by sorry

end summer_camp_probability_l1687_168789


namespace binomial_square_special_case_l1687_168779

theorem binomial_square_special_case (a b : ℝ) : (2*a - 3*b)^2 = 4*a^2 - 12*a*b + 9*b^2 := by
  sorry

end binomial_square_special_case_l1687_168779


namespace sin_cos_value_f_minus_cos_value_l1687_168727

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.tan x) * Real.cos x / (1 + Real.cos (-x))

-- Theorem 1
theorem sin_cos_value (θ : ℝ) (h : f θ * Real.sin (π/6) - Real.cos θ = 0) :
  Real.sin θ * Real.cos θ = 2/5 := by sorry

-- Theorem 2
theorem f_minus_cos_value (θ : ℝ) (h1 : f θ * Real.cos θ = 1/8) (h2 : π/4 < θ ∧ θ < 3*π/4) :
  f (2019*π - θ) - Real.cos (2018*π - θ) = Real.sqrt 3 / 2 := by sorry

end sin_cos_value_f_minus_cos_value_l1687_168727


namespace length_of_segment_AB_l1687_168755

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y + 6 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem length_of_segment_AB (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 * Real.sqrt 5 / 5 :=
sorry

end length_of_segment_AB_l1687_168755


namespace karens_class_size_l1687_168776

def total_cookies : ℕ := 50
def kept_cookies : ℕ := 10
def grandparents_cookies : ℕ := 8
def cookies_per_classmate : ℕ := 2

theorem karens_class_size :
  (total_cookies - kept_cookies - grandparents_cookies) / cookies_per_classmate = 16 := by
  sorry

end karens_class_size_l1687_168776


namespace fraction_simplification_l1687_168784

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 * b^2) / (b/a)^2 = a^4 := by
  sorry

end fraction_simplification_l1687_168784


namespace solve_equation_l1687_168734

theorem solve_equation (x : ℝ) (h : Real.sqrt (3 / x + 1) = 5 / 3) : x = 27 / 16 := by
  sorry

end solve_equation_l1687_168734


namespace best_fit_model_l1687_168712

/-- Represents a regression model with its coefficient of determination -/
structure RegressionModel where
  r_squared : ℝ
  h_r_squared_nonneg : 0 ≤ r_squared
  h_r_squared_le_one : r_squared ≤ 1

/-- Determines if a model has better fit than another based on R² values -/
def better_fit (model1 model2 : RegressionModel) : Prop :=
  model1.r_squared > model2.r_squared

theorem best_fit_model (model1 model2 model3 model4 : RegressionModel)
  (h1 : model1.r_squared = 0.05)
  (h2 : model2.r_squared = 0.49)
  (h3 : model3.r_squared = 0.89)
  (h4 : model4.r_squared = 0.98) :
  better_fit model4 model1 ∧ better_fit model4 model2 ∧ better_fit model4 model3 :=
sorry

end best_fit_model_l1687_168712


namespace pistachio_stairs_l1687_168738

/-- The number of steps between each floor -/
def steps_per_floor : ℕ := 20

/-- The floor where Pistachio lives -/
def target_floor : ℕ := 11

/-- The starting floor -/
def start_floor : ℕ := 1

/-- The total number of steps to reach the target floor -/
def total_steps : ℕ := (target_floor - start_floor) * steps_per_floor

theorem pistachio_stairs : total_steps = 200 := by
  sorry

end pistachio_stairs_l1687_168738


namespace proposition_not_hold_for_2_l1687_168754

theorem proposition_not_hold_for_2 (P : ℕ → Prop)
  (h1 : ¬ P 3)
  (h2 : ∀ k : ℕ, k > 0 → P k → P (k + 1)) :
  ¬ P 2 := by
  sorry

end proposition_not_hold_for_2_l1687_168754


namespace chores_repayment_l1687_168731

/-- Calculates the amount earned for a given hour in the chore cycle -/
def hourly_rate (hour : ℕ) : ℕ :=
  match hour % 3 with
  | 1 => 2
  | 2 => 4
  | 0 => 6
  | _ => 0 -- This case should never occur, but Lean requires it for completeness

/-- Calculates the total amount earned for a given number of hours -/
def total_earned (hours : ℕ) : ℕ :=
  (List.range hours).map hourly_rate |>.sum

/-- The main theorem stating that 45 hours of chores results in $180 earned -/
theorem chores_repayment : total_earned 45 = 180 := by
  sorry

end chores_repayment_l1687_168731


namespace algebraic_expression_value_l1687_168733

theorem algebraic_expression_value (a b : ℝ) 
  (h1 : a * b = 2) 
  (h2 : a - b = 3) : 
  2 * a^3 * b - 4 * a^2 * b^2 + 2 * a * b^3 = 36 := by
  sorry

end algebraic_expression_value_l1687_168733


namespace base5_2202_equals_base10_302_l1687_168773

/-- Converts a base 5 digit to its base 10 equivalent --/
def base5ToBase10 (digit : Nat) (position : Nat) : Nat :=
  digit * (5 ^ position)

/-- Theorem: The base 5 number 2202₅ is equal to the base 10 number 302 --/
theorem base5_2202_equals_base10_302 :
  base5ToBase10 2 3 + base5ToBase10 2 2 + base5ToBase10 0 1 + base5ToBase10 2 0 = 302 := by
  sorry

end base5_2202_equals_base10_302_l1687_168773


namespace simplify_fraction_l1687_168741

theorem simplify_fraction : 
  (5^1004)^2 - (5^1002)^2 / (5^1003)^2 - (5^1001)^2 = 25 := by
sorry

end simplify_fraction_l1687_168741


namespace base_five_product_131_21_l1687_168783

/-- Represents a number in base 5 --/
def BaseFive : Type := List Nat

/-- Converts a base 5 number to its decimal representation --/
def to_decimal (n : BaseFive) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to its base 5 representation --/
def to_base_five (n : Nat) : BaseFive :=
  sorry

/-- Multiplies two base 5 numbers --/
def base_five_mul (a b : BaseFive) : BaseFive :=
  to_base_five (to_decimal a * to_decimal b)

theorem base_five_product_131_21 :
  base_five_mul [1, 3, 1] [1, 2] = [1, 5, 2, 3] :=
sorry

end base_five_product_131_21_l1687_168783


namespace union_equals_reals_implies_a_is_negative_one_l1687_168717

-- Define the sets S and P
def S : Set ℝ := {x : ℝ | x ≤ -1 ∨ x ≥ 2}
def P (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 3}

-- State the theorem
theorem union_equals_reals_implies_a_is_negative_one (a : ℝ) :
  S ∪ P a = Set.univ → a = -1 := by
  sorry

end union_equals_reals_implies_a_is_negative_one_l1687_168717


namespace phil_initial_money_l1687_168744

/-- The amount of money Phil started with, given his purchases and remaining quarters. -/
theorem phil_initial_money (pizza_cost soda_cost jeans_cost : ℚ)
  (quarters_left : ℕ) (quarter_value : ℚ) :
  pizza_cost = 2.75 →
  soda_cost = 1.50 →
  jeans_cost = 11.50 →
  quarters_left = 97 →
  quarter_value = 0.25 →
  pizza_cost + soda_cost + jeans_cost + (quarters_left : ℚ) * quarter_value = 40 :=
by sorry

end phil_initial_money_l1687_168744


namespace num_triangles_on_circle_l1687_168786

/-- The number of ways to choose k items from n items. -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of points on the circle. -/
def numPoints : ℕ := 10

/-- The number of points needed to form a triangle. -/
def pointsPerTriangle : ℕ := 3

/-- Theorem: The number of triangles that can be formed from 10 points on a circle is 120. -/
theorem num_triangles_on_circle :
  binomial numPoints pointsPerTriangle = 120 := by
  sorry

end num_triangles_on_circle_l1687_168786


namespace smallest_prime_twelve_less_square_l1687_168742

theorem smallest_prime_twelve_less_square : 
  ∃ (n : ℕ), 
    (n > 0) ∧ 
    (Nat.Prime n) ∧ 
    (∃ (m : ℕ), n = m^2 - 12) ∧
    (∀ (k : ℕ), k > 0 → Nat.Prime k → (∃ (l : ℕ), k = l^2 - 12) → k ≥ n) ∧
    n = 13 :=
by sorry

end smallest_prime_twelve_less_square_l1687_168742


namespace complex_equation_solutions_l1687_168706

theorem complex_equation_solutions :
  ∃! (S : Finset ℂ), 
    (∀ z ∈ S, Complex.abs z < 15 ∧ Complex.exp (2 * z) = (z - 2) / (z + 2)) ∧
    Finset.card S = 9 := by
  sorry

end complex_equation_solutions_l1687_168706


namespace negation_existential_statement_l1687_168707

theorem negation_existential_statement :
  ¬(∃ (x : ℝ), x^2 - x + 2 > 0) ≠ (∀ (x : ℝ), x^2 - x + 2 ≤ 0) := by sorry

end negation_existential_statement_l1687_168707


namespace smallest_addend_for_divisibility_l1687_168751

theorem smallest_addend_for_divisibility (a b : ℕ) (ha : a = 87908235) (hb : b = 12587) :
  let x := (b - (a % b)) % b
  (a + x) % b = 0 ∧ ∀ y : ℕ, y < x → (a + y) % b ≠ 0 := by
  sorry

end smallest_addend_for_divisibility_l1687_168751


namespace remainder_of_product_l1687_168797

theorem remainder_of_product (n : ℕ) (h : n = 67545) : (n * 11) % 13 = 11 := by
  sorry

end remainder_of_product_l1687_168797


namespace trip_duration_is_101_l1687_168739

/-- Calculates the total trip duration for Jill's journey to the library --/
def total_trip_duration (first_bus_wait : ℕ) (first_bus_ride : ℕ) (first_bus_delay : ℕ)
                        (walk_time : ℕ) (train_wait : ℕ) (train_ride : ℕ) (train_delay : ℕ)
                        (second_bus_wait_A : ℕ) (second_bus_ride_A : ℕ)
                        (second_bus_wait_B : ℕ) (second_bus_ride_B : ℕ) : ℕ :=
  let first_bus_total := first_bus_wait + first_bus_ride + first_bus_delay
  let train_total := walk_time + train_wait + train_ride + train_delay
  let second_bus_total := if second_bus_ride_A < second_bus_ride_B
                          then (second_bus_wait_A + second_bus_ride_A) / 2
                          else (second_bus_wait_B + second_bus_ride_B) / 2
  first_bus_total + train_total + second_bus_total

/-- Theorem stating that the total trip duration is 101 minutes --/
theorem trip_duration_is_101 :
  total_trip_duration 12 30 5 10 8 20 3 15 10 20 6 = 101 := by
  sorry

end trip_duration_is_101_l1687_168739


namespace andy_final_position_l1687_168740

/-- Represents the position of Andy the Ant -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents the direction Andy is facing -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents Andy's state at any given moment -/
structure AntState :=
  (pos : Position)
  (dir : Direction)
  (moveCount : Nat)

/-- The movement function for Andy -/
def move (state : AntState) : AntState :=
  sorry

/-- The main theorem stating Andy's final position -/
theorem andy_final_position :
  let initialState : AntState :=
    { pos := { x := 30, y := -30 }
    , dir := Direction.North
    , moveCount := 0
    }
  let finalState := (move^[3030]) initialState
  finalState.pos = { x := 4573, y := -1546 } :=
sorry

end andy_final_position_l1687_168740


namespace special_function_is_odd_and_even_l1687_168788

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x + y ≠ 0 → f (x * y) = (f x + f y) / (x + y)

/-- A function is both odd and even -/
def odd_and_even (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f (-x) = f x)

/-- The main theorem -/
theorem special_function_is_odd_and_even (f : ℝ → ℝ) (h : special_function f) :
  odd_and_even f :=
sorry

end special_function_is_odd_and_even_l1687_168788


namespace log_sum_equals_five_l1687_168702

-- Define the logarithm functions
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10
noncomputable def log_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem log_sum_equals_five : lg 25 + log_3 27 + lg 4 = 5 := by
  sorry

end log_sum_equals_five_l1687_168702


namespace consecutive_squares_sum_l1687_168711

theorem consecutive_squares_sum (n : ℕ) (h : 2 * n + 1 = 144169^2) :
  ∃ (a : ℕ), a^2 + (a + 1)^2 = n + 1 :=
sorry

end consecutive_squares_sum_l1687_168711


namespace geometric_sequence_problem_l1687_168758

/-- Given a geometric sequence {a_n}, if a_4 and a_8 are the roots of x^2 - 8x + 9 = 0, then a_6 = 3 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  (a 4 + a 8 = 8) →                    -- sum of roots
  (a 4 * a 8 = 9) →                    -- product of roots
  a 6 = 3 := by sorry

end geometric_sequence_problem_l1687_168758


namespace consecutive_integers_product_plus_one_is_square_l1687_168725

theorem consecutive_integers_product_plus_one_is_square (n : ℤ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end consecutive_integers_product_plus_one_is_square_l1687_168725


namespace combined_miles_per_gallon_l1687_168792

/-- The combined miles per gallon of two cars given their individual efficiencies and distance ratio -/
theorem combined_miles_per_gallon
  (sam_mpg : ℝ)
  (alex_mpg : ℝ)
  (distance_ratio : ℚ)
  (h_sam_mpg : sam_mpg = 50)
  (h_alex_mpg : alex_mpg = 20)
  (h_distance_ratio : distance_ratio = 2 / 3) :
  (2 * distance_ratio + 3) / (2 * distance_ratio / sam_mpg + 3 / alex_mpg) = 500 / 19 := by
  sorry

end combined_miles_per_gallon_l1687_168792


namespace sqrt_decimal_expansion_unique_l1687_168752

theorem sqrt_decimal_expansion_unique 
  (p n : ℚ) 
  (hp : 0 < p) 
  (hn : 0 < n) 
  (hp_not_square : ¬ ∃ (m : ℚ), p = m ^ 2) 
  (hn_not_square : ¬ ∃ (m : ℚ), n = m ^ 2) : 
  ¬ ∃ (k : ℤ), Real.sqrt p - Real.sqrt n = k := by
  sorry

end sqrt_decimal_expansion_unique_l1687_168752


namespace franklin_valentines_l1687_168704

/-- The number of Valentines Mrs. Franklin initially had -/
def initial_valentines : ℕ := 58

/-- The number of Valentines Mrs. Franklin gave to her students -/
def given_valentines : ℕ := 42

/-- The number of Valentines Mrs. Franklin has now -/
def remaining_valentines : ℕ := initial_valentines - given_valentines

/-- Theorem stating that Mrs. Franklin now has 16 Valentines -/
theorem franklin_valentines : remaining_valentines = 16 := by
  sorry

end franklin_valentines_l1687_168704


namespace inequality_solution_l1687_168759

theorem inequality_solution (x : ℝ) : (x + 3) * (x - 2) < 0 ↔ -3 < x ∧ x < 2 := by
  sorry

end inequality_solution_l1687_168759


namespace ages_sum_l1687_168709

theorem ages_sum (a b c : ℕ) : 
  a = b + c + 18 ∧ 
  a^2 = (b + c)^2 + 2016 → 
  a + b + c = 112 :=
by
  sorry

end ages_sum_l1687_168709
