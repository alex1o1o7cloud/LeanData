import Mathlib

namespace NUMINAMATH_CALUDE_grass_field_length_l4061_406160

/-- Represents a rectangular grass field with a surrounding path. -/
structure GrassField where
  length : ℝ
  width : ℝ
  pathWidth : ℝ

/-- Calculates the area of the path surrounding the grass field. -/
def pathArea (field : GrassField) : ℝ :=
  (field.length + 2 * field.pathWidth) * (field.width + 2 * field.pathWidth) - field.length * field.width

/-- Theorem stating the length of the grass field given specific conditions. -/
theorem grass_field_length : 
  ∀ (field : GrassField),
  field.width = 55 →
  field.pathWidth = 2.5 →
  pathArea field = 1250 →
  field.length = 190 := by
sorry

end NUMINAMATH_CALUDE_grass_field_length_l4061_406160


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l4061_406106

theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (remaining_age_diff : ℝ),
    team_size = 15 →
    captain_age = 32 →
    wicket_keeper_age_diff = 5 →
    remaining_age_diff = 2 →
    ∃ (team_avg_age : ℝ),
      team_avg_age * team_size =
        captain_age + (captain_age + wicket_keeper_age_diff) +
        (team_size - 2) * (team_avg_age - remaining_age_diff) ∧
      team_avg_age = 21.5 := by
sorry


end NUMINAMATH_CALUDE_cricket_team_average_age_l4061_406106


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l4061_406174

theorem arithmetic_to_geometric_sequence (a₁ a₂ a₃ a₄ d : ℝ) : 
  d ≠ 0 →
  a₂ = a₁ + d →
  a₃ = a₁ + 2*d →
  a₄ = a₁ + 3*d →
  ((a₂^2 = a₁ * a₃) ∨ (a₂^2 = a₁ * a₄) ∨ (a₃^2 = a₁ * a₄) ∨ (a₃^2 = a₂ * a₄)) →
  (a₁ / d = -4 ∨ a₁ / d = 1) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l4061_406174


namespace NUMINAMATH_CALUDE_pencils_given_to_dorothy_l4061_406107

theorem pencils_given_to_dorothy (initial_pencils : ℕ) (remaining_pencils : ℕ) 
  (h1 : initial_pencils = 142) 
  (h2 : remaining_pencils = 111) : 
  initial_pencils - remaining_pencils = 31 := by
  sorry

end NUMINAMATH_CALUDE_pencils_given_to_dorothy_l4061_406107


namespace NUMINAMATH_CALUDE_hawthorn_box_maximum_l4061_406117

theorem hawthorn_box_maximum (N : ℕ) : 
  N > 100 ∧
  N % 3 = 1 ∧
  N % 4 = 2 ∧
  N % 5 = 3 ∧
  N % 6 = 4 →
  N ≤ 178 ∧ ∃ (M : ℕ), M = 178 ∧ 
    M > 100 ∧
    M % 3 = 1 ∧
    M % 4 = 2 ∧
    M % 5 = 3 ∧
    M % 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hawthorn_box_maximum_l4061_406117


namespace NUMINAMATH_CALUDE_extremum_of_f_l4061_406100

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x

-- Define the derivative of f(x)
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_of_f (a b : ℝ) :
  (f' a b 2 = 0) →  -- Extremum at x=2
  (f' a b 1 = -3) →  -- Tangent line at x=1 parallel to y=-3x-2
  (∃ x, f a b x = -4 ∧ ∀ y, f a b y ≥ f a b x) :=
by
  sorry

end NUMINAMATH_CALUDE_extremum_of_f_l4061_406100


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l4061_406176

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a + b + c = 6 →
  A = π/3 ∧ a = 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l4061_406176


namespace NUMINAMATH_CALUDE_sean_final_houses_l4061_406119

/-- Calculates the final number of houses Sean has after a series of transactions in Monopoly. -/
def final_houses (initial : ℕ) (traded_for_money : ℕ) (bought : ℕ) (traded_for_marvin : ℕ) (sold_for_atlantic : ℕ) (traded_for_hotels : ℕ) : ℕ :=
  initial - traded_for_money + bought - traded_for_marvin - sold_for_atlantic - traded_for_hotels

/-- Theorem stating that Sean ends up with 20 houses after the given transactions. -/
theorem sean_final_houses : 
  final_houses 45 15 18 5 7 16 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sean_final_houses_l4061_406119


namespace NUMINAMATH_CALUDE_smallest_cube_ending_632_l4061_406167

theorem smallest_cube_ending_632 :
  ∃ n : ℕ+, (n : ℤ)^3 ≡ 632 [ZMOD 1000] ∧
  ∀ m : ℕ+, (m : ℤ)^3 ≡ 632 [ZMOD 1000] → n ≤ m ∧ n = 192 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_632_l4061_406167


namespace NUMINAMATH_CALUDE_fraction_equals_seven_l4061_406152

theorem fraction_equals_seven : (2^2016 + 3 * 2^2014) / (2^2016 - 3 * 2^2014) = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_seven_l4061_406152


namespace NUMINAMATH_CALUDE_max_equal_ending_digits_of_squares_max_equal_ending_digits_is_tight_l4061_406115

/-- The maximum number of equal non-zero digits that can appear at the end of a perfect square in base 10 -/
def max_equal_ending_digits : ℕ := 3

/-- A function that returns the number of equal non-zero digits at the end of a number in base 10 -/
def count_equal_ending_digits (n : ℕ) : ℕ := sorry

theorem max_equal_ending_digits_of_squares :
  ∀ n : ℕ, count_equal_ending_digits (n^2) ≤ max_equal_ending_digits :=
by sorry

theorem max_equal_ending_digits_is_tight :
  ∃ n : ℕ, count_equal_ending_digits (n^2) = max_equal_ending_digits :=
by sorry

end NUMINAMATH_CALUDE_max_equal_ending_digits_of_squares_max_equal_ending_digits_is_tight_l4061_406115


namespace NUMINAMATH_CALUDE_sum_of_fractions_l4061_406162

theorem sum_of_fractions : (1 : ℚ) / 6 + (5 : ℚ) / 12 = (7 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l4061_406162


namespace NUMINAMATH_CALUDE_second_discount_is_fifteen_percent_l4061_406149

/-- Calculates the final price of a car after three successive discounts -/
def finalPrice (initialPrice : ℝ) (discount1 discount2 discount3 : ℝ) : ℝ :=
  let price1 := initialPrice * (1 - discount1)
  let price2 := price1 * (1 - discount2)
  price2 * (1 - discount3)

/-- Theorem stating that given the initial price and three discounts, 
    the second discount is 15% when the final price is $7,752 -/
theorem second_discount_is_fifteen_percent 
  (initialPrice : ℝ) 
  (discount1 : ℝ) 
  (discount3 : ℝ) :
  initialPrice = 12000 →
  discount1 = 0.20 →
  discount3 = 0.05 →
  finalPrice initialPrice discount1 0.15 discount3 = 7752 :=
by
  sorry

#eval finalPrice 12000 0.20 0.15 0.05

end NUMINAMATH_CALUDE_second_discount_is_fifteen_percent_l4061_406149


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l4061_406113

theorem z_in_fourth_quadrant (z : ℂ) (h : z * Complex.I = 2 + Complex.I) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l4061_406113


namespace NUMINAMATH_CALUDE_problem_solution_l4061_406186

theorem problem_solution (a b : ℝ) (h : |a - 1| + |b + 2| = 0) : 
  (a + b)^2013 + |b| = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4061_406186


namespace NUMINAMATH_CALUDE_haris_contribution_haris_contribution_is_9720_l4061_406168

/-- Calculates Hari's contribution to the capital given the investment conditions --/
theorem haris_contribution (praveen_investment : ℕ) (praveen_months : ℕ) (hari_months : ℕ) 
  (profit_ratio_praveen : ℕ) (profit_ratio_hari : ℕ) : ℕ :=
  let total_months := praveen_months
  let hari_contribution := (praveen_investment * praveen_months * profit_ratio_hari) / 
                           (hari_months * profit_ratio_praveen)
  hari_contribution

/-- Proves that Hari's contribution is 9720 given the specific conditions --/
theorem haris_contribution_is_9720 : 
  haris_contribution 3780 12 7 2 3 = 9720 := by
  sorry

end NUMINAMATH_CALUDE_haris_contribution_haris_contribution_is_9720_l4061_406168


namespace NUMINAMATH_CALUDE_number_multiplied_by_48_l4061_406123

theorem number_multiplied_by_48 : ∃ x : ℤ, x * 48 = 173 * 240 ∧ x = 865 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_48_l4061_406123


namespace NUMINAMATH_CALUDE_housing_boom_construction_l4061_406199

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before : ℕ := 1426

/-- The number of houses in Lawrence County after the housing boom -/
def houses_after : ℕ := 2000

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := houses_after - houses_before

theorem housing_boom_construction : houses_built = 574 := by
  sorry

end NUMINAMATH_CALUDE_housing_boom_construction_l4061_406199


namespace NUMINAMATH_CALUDE_inequality_proof_l4061_406169

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 3) :
  Real.sqrt x + Real.sqrt y + Real.sqrt z ≥ x * y + y * z + z * x := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4061_406169


namespace NUMINAMATH_CALUDE_total_onion_weight_is_10_9_l4061_406132

/-- The total weight of onions grown by Sara, Sally, Fred, and Jack -/
def total_onion_weight : ℝ :=
  let sara_onions := 4
  let sara_weight := 0.5
  let sally_onions := 5
  let sally_weight := 0.4
  let fred_onions := 9
  let fred_weight := 0.3
  let jack_onions := 7
  let jack_weight := 0.6
  sara_onions * sara_weight +
  sally_onions * sally_weight +
  fred_onions * fred_weight +
  jack_onions * jack_weight

/-- Proof that the total weight of onions is 10.9 pounds -/
theorem total_onion_weight_is_10_9 :
  total_onion_weight = 10.9 := by sorry

end NUMINAMATH_CALUDE_total_onion_weight_is_10_9_l4061_406132


namespace NUMINAMATH_CALUDE_price_decrease_l4061_406138

/-- Given an article with an original price of 700 rupees and a price decrease of 24%,
    the new price after the decrease is 532 rupees. -/
theorem price_decrease (original_price : ℝ) (decrease_percentage : ℝ) (new_price : ℝ) :
  original_price = 700 →
  decrease_percentage = 24 →
  new_price = original_price * (1 - decrease_percentage / 100) →
  new_price = 532 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_l4061_406138


namespace NUMINAMATH_CALUDE_tv_selection_problem_l4061_406195

theorem tv_selection_problem (type_a : ℕ) (type_b : ℕ) (total_selection : ℕ) :
  type_a = 4 →
  type_b = 5 →
  total_selection = 3 →
  (Nat.choose type_a 2 * Nat.choose type_b 1) + (Nat.choose type_a 1 * Nat.choose type_b 2) = 70 :=
by sorry

end NUMINAMATH_CALUDE_tv_selection_problem_l4061_406195


namespace NUMINAMATH_CALUDE_square_root_of_two_squared_equals_two_l4061_406185

theorem square_root_of_two_squared_equals_two :
  (Real.sqrt 2) ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_two_squared_equals_two_l4061_406185


namespace NUMINAMATH_CALUDE_range_of_m_l4061_406184

theorem range_of_m (x m : ℝ) : 
  (∀ x, (|x - 1| < 2 → -1 < x ∧ x < m + 1) ∧ 
   ∃ x, (-1 < x ∧ x < m + 1 ∧ ¬(|x - 1| < 2))) →
  m > 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l4061_406184


namespace NUMINAMATH_CALUDE_movie_choice_l4061_406147

-- Define the set of all movies
def Movies : Set Char := {'A', 'B', 'C', 'D', 'E'}

-- Define the acceptable movies for each person
def Zhao : Set Char := Movies \ {'B'}
def Zhang : Set Char := {'B', 'C', 'D', 'E'}
def Li : Set Char := Movies \ {'C'}
def Liu : Set Char := Movies \ {'E'}

-- Theorem statement
theorem movie_choice : Zhao ∩ Zhang ∩ Li ∩ Liu = {'D'} := by
  sorry

end NUMINAMATH_CALUDE_movie_choice_l4061_406147


namespace NUMINAMATH_CALUDE_call_duration_l4061_406127

def calls_per_year : ℕ := 52
def cost_per_minute : ℚ := 5 / 100
def total_cost_per_year : ℚ := 78

theorem call_duration :
  (total_cost_per_year / cost_per_minute) / calls_per_year = 30 := by
  sorry

end NUMINAMATH_CALUDE_call_duration_l4061_406127


namespace NUMINAMATH_CALUDE_lens_price_proof_l4061_406166

theorem lens_price_proof (price_no_discount : ℝ) (discount_rate : ℝ) (cheaper_lens_price : ℝ) :
  price_no_discount = 300 ∧
  discount_rate = 0.2 ∧
  cheaper_lens_price = 220 ∧
  price_no_discount * (1 - discount_rate) = cheaper_lens_price + 20 :=
by sorry

end NUMINAMATH_CALUDE_lens_price_proof_l4061_406166


namespace NUMINAMATH_CALUDE_closest_to_500_div_025_l4061_406177

def options : List ℝ := [1000, 1500, 2000, 2500, 3000]

theorem closest_to_500_div_025 :
  ∃ (x : ℝ), x ∈ options ∧ 
  ∀ (y : ℝ), y ∈ options → |x - 500/0.25| ≤ |y - 500/0.25| ∧
  x = 2000 :=
by sorry

end NUMINAMATH_CALUDE_closest_to_500_div_025_l4061_406177


namespace NUMINAMATH_CALUDE_spade_calculation_l4061_406144

-- Define the ⋄ operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_calculation : spade (spade 1 2) (spade 9 (spade 5 4)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l4061_406144


namespace NUMINAMATH_CALUDE_complex_multiplication_l4061_406193

theorem complex_multiplication (i : ℂ) (h : i * i = -1) : 
  (-1 + i) * (2 - i) = -1 + 3*i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l4061_406193


namespace NUMINAMATH_CALUDE_boxes_per_case_l4061_406122

theorem boxes_per_case (total_boxes : ℕ) (total_cases : ℕ) (boxes_per_case : ℕ) : 
  total_boxes = 20 → total_cases = 5 → total_boxes = total_cases * boxes_per_case → boxes_per_case = 4 := by
  sorry

end NUMINAMATH_CALUDE_boxes_per_case_l4061_406122


namespace NUMINAMATH_CALUDE_min_side_length_triangle_l4061_406151

theorem min_side_length_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  a + b = 2 →
  C = 2 * π / 3 →
  c = Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos C)) →
  c ≥ Real.sqrt 3 ∧ (c = Real.sqrt 3 ↔ a = b) := by
sorry

end NUMINAMATH_CALUDE_min_side_length_triangle_l4061_406151


namespace NUMINAMATH_CALUDE_probability_black_or_white_ball_l4061_406102

theorem probability_black_or_white_ball 
  (p_red : ℝ) 
  (p_white : ℝ) 
  (h1 : p_red = 0.45) 
  (h2 : p_white = 0.25) 
  (h3 : 0 ≤ p_red ∧ p_red ≤ 1) 
  (h4 : 0 ≤ p_white ∧ p_white ≤ 1) : 
  p_red + p_white + (1 - p_red - p_white) = 1 ∧ 1 - p_red = 0.55 := by
sorry

end NUMINAMATH_CALUDE_probability_black_or_white_ball_l4061_406102


namespace NUMINAMATH_CALUDE_polynomial_characterization_l4061_406130

-- Define S(k) as the sum of digits of k in decimal representation
def S (k : ℕ) : ℕ := sorry

-- Define the property that P(x) must satisfy
def satisfies_property (P : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 2016 → (S (P n) = P (S n) ∧ P n > 0)

-- Define the set of valid polynomials
def valid_polynomial (P : ℕ → ℕ) : Prop :=
  (∃ c : ℕ, c ≥ 1 ∧ c ≤ 9 ∧ (∀ x : ℕ, P x = c)) ∨
  (∀ x : ℕ, P x = x)

-- Theorem statement
theorem polynomial_characterization :
  ∀ P : ℕ → ℕ, satisfies_property P → valid_polynomial P :=
sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l4061_406130


namespace NUMINAMATH_CALUDE_tangent_line_equation_l4061_406116

/-- A line passing through (b, 0) and tangent to a circle of radius r centered at (0, 0),
    forming a triangle in the first quadrant with area S, has the equation rx - bry - rb = 0 --/
theorem tangent_line_equation (b r S : ℝ) (hb : b > 0) (hr : r > 0) (hS : S > 0) :
  ∃ (x y : ℝ → ℝ), 
    (∀ t, x t = b ∧ y t = 0) →  -- Line passes through (b, 0)
    (∃ t, (x t)^2 + (y t)^2 = r^2) →  -- Line touches the circle
    (∃ h, S = (1/2) * b * h) →  -- Triangle area
    (∀ t, r * (x t) - b * r * (y t) - r * b = 0) :=  -- Equation of the line
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l4061_406116


namespace NUMINAMATH_CALUDE_regular_nonagon_diagonal_sum_l4061_406191

/-- A regular nonagon is a 9-sided polygon with all sides equal and all angles equal. -/
structure RegularNonagon where
  side_length : ℝ
  shortest_diagonal : ℝ
  longest_diagonal : ℝ

/-- 
In a regular nonagon, the longest diagonal is equal to the sum of 
the side length and the shortest diagonal.
-/
theorem regular_nonagon_diagonal_sum (n : RegularNonagon) : 
  n.longest_diagonal = n.side_length + n.shortest_diagonal := by
  sorry

end NUMINAMATH_CALUDE_regular_nonagon_diagonal_sum_l4061_406191


namespace NUMINAMATH_CALUDE_intersection_theorem_l4061_406120

def P : Set ℝ := {x | (x - 1)^2 > 16}
def Q (a : ℝ) : Set ℝ := {x | x^2 + (a - 8) * x - 8 * a ≤ 0}

theorem intersection_theorem (a : ℝ) :
  (∃ a, a = 3 → P ∩ Q a = {x | 5 < x ∧ x ≤ 8}) ∧
  (P ∩ Q a = {x | 5 < x ∧ x ≤ 8} ↔ a ∈ Set.Icc (-5) 3) ∧
  (∀ a,
    (a > 3 → P ∩ Q a = {x | -a ≤ x ∧ x < -3 ∨ 5 < x ∧ x ≤ 8}) ∧
    (-5 ≤ a ∧ a ≤ 3 → P ∩ Q a = {x | 5 < x ∧ x ≤ 8}) ∧
    (-8 ≤ a ∧ a < -5 → P ∩ Q a = {x | -a ≤ x ∧ x ≤ 8}) ∧
    (a < -8 → P ∩ Q a = {x | 8 ≤ x ∧ x ≤ -a})) :=
sorry

end NUMINAMATH_CALUDE_intersection_theorem_l4061_406120


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l4061_406110

-- Problem 1
theorem problem_1 (m n : ℝ) : 2 * m * n^2 * (1/4 * m * n) = 1/2 * m^2 * n^3 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : (2 * a^3 * b^2 + a^2 * b) / (a * b) = 2 * a^2 * b + a := by sorry

-- Problem 3
theorem problem_3 (x : ℝ) : (2 * x + 3) * (x - 1) = 2 * x^2 + x - 3 := by sorry

-- Problem 4
theorem problem_4 (x y : ℝ) : (x + y)^2 - 2 * y * (x - y) = x^2 + 3 * y^2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l4061_406110


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l4061_406158

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_in_fourth_quadrant : 
  let x : ℝ := 1
  let y : ℝ := -5
  fourth_quadrant x y :=
by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l4061_406158


namespace NUMINAMATH_CALUDE_intersection_M_N_l4061_406128

def M : Set ℝ := {x | ∃ t : ℝ, x = Real.exp (-t * Real.log 2)}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4061_406128


namespace NUMINAMATH_CALUDE_line_y_axis_intersection_l4061_406126

def line (x y : ℝ) : Prop := y = 2 * x + 1

def y_axis (x : ℝ) : Prop := x = 0

def intersection_point : Set (ℝ × ℝ) := {(0, 1)}

theorem line_y_axis_intersection :
  {p : ℝ × ℝ | line p.1 p.2 ∧ y_axis p.1} = intersection_point := by
sorry

end NUMINAMATH_CALUDE_line_y_axis_intersection_l4061_406126


namespace NUMINAMATH_CALUDE_watermelon_price_per_pound_l4061_406175

/-- The price per pound of watermelons sold by Farmer Kent -/
def price_per_pound (watermelon_weight : ℕ) (num_watermelons : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (watermelon_weight * num_watermelons)

/-- Theorem stating that the price per pound of Farmer Kent's watermelons is $2 -/
theorem watermelon_price_per_pound :
  price_per_pound 23 18 828 = 2 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_price_per_pound_l4061_406175


namespace NUMINAMATH_CALUDE_water_balloon_count_l4061_406112

-- Define the number of water balloons for each person
def sarah_balloons : ℕ := 5
def janice_balloons : ℕ := 6

-- Define the relationships between the number of water balloons
theorem water_balloon_count :
  ∀ (tim_balloons randy_balloons cynthia_balloons : ℕ),
  (tim_balloons = 2 * sarah_balloons) →
  (tim_balloons + 3 = janice_balloons) →
  (2 * randy_balloons = janice_balloons) →
  (cynthia_balloons = 4 * randy_balloons) →
  cynthia_balloons = 12 := by
  sorry

end NUMINAMATH_CALUDE_water_balloon_count_l4061_406112


namespace NUMINAMATH_CALUDE_square_diameter_double_area_l4061_406157

theorem square_diameter_double_area (d₁ : ℝ) (d₂ : ℝ) : 
  d₁ = 4 * Real.sqrt 2 →
  (d₂ / 2)^2 = 2 * (d₁ / 2)^2 →
  d₂ = 8 :=
by sorry

end NUMINAMATH_CALUDE_square_diameter_double_area_l4061_406157


namespace NUMINAMATH_CALUDE_count_pairs_eq_nine_l4061_406183

/-- The number of distinct ordered pairs of positive integers (x, y) such that 1/x + 1/y = 1/6 -/
def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    p.1 > 0 ∧ p.2 > 0 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = (1 : ℚ) / 6)
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card

theorem count_pairs_eq_nine : count_pairs = 9 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_nine_l4061_406183


namespace NUMINAMATH_CALUDE_number_of_students_l4061_406187

def candy_bar_cost : ℚ := 2
def chips_cost : ℚ := 1/2

def student_purchase_cost : ℚ := candy_bar_cost + 2 * chips_cost

def total_cost : ℚ := 15

theorem number_of_students : 
  (total_cost / student_purchase_cost : ℚ) = 5 := by sorry

end NUMINAMATH_CALUDE_number_of_students_l4061_406187


namespace NUMINAMATH_CALUDE_hannahs_running_distance_l4061_406148

/-- Hannah's running distances problem -/
theorem hannahs_running_distance :
  -- Define the distances
  let monday_distance : ℕ := 9000
  let friday_distance : ℕ := 2095
  let additional_distance : ℕ := 2089

  -- Define the relation between distances
  ∀ wednesday_distance : ℕ,
    monday_distance = wednesday_distance + friday_distance + additional_distance →
    wednesday_distance = 4816 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_running_distance_l4061_406148


namespace NUMINAMATH_CALUDE_pizza_toppings_l4061_406194

theorem pizza_toppings (total_slices : ℕ) (cheese_slices : ℕ) (bacon_slices : ℕ) 
  (h1 : total_slices = 15)
  (h2 : cheese_slices = 8)
  (h3 : bacon_slices = 13)
  (h4 : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range cheese_slices ∨ slice ∈ Finset.range bacon_slices)) :
  ∃ both_toppings : ℕ, both_toppings = 6 ∧ 
    cheese_slices + bacon_slices - both_toppings = total_slices :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l4061_406194


namespace NUMINAMATH_CALUDE_ellipse_fixed_point_theorem_l4061_406189

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-1, 0)
def right_focus : ℝ × ℝ := (1, 0)

-- Define the upper vertex
noncomputable def upper_vertex : ℝ × ℝ := (0, 1)

-- Define the slope condition
def slope_condition (M N : ℝ × ℝ) : Prop :=
  let Q := upper_vertex
  let k_QM := (M.2 - Q.2) / (M.1 - Q.1)
  let k_QN := (N.2 - Q.2) / (N.1 - Q.1)
  k_QM + k_QN = 1

-- Define the fixed point
def fixed_point : ℝ × ℝ := (-2, -1)

-- Theorem statement
theorem ellipse_fixed_point_theorem (M N : ℝ × ℝ) :
  ellipse M.1 M.2 →
  ellipse N.1 N.2 →
  M ≠ upper_vertex →
  N ≠ upper_vertex →
  slope_condition M N →
  ∃ (k t : ℝ), M.2 = k * M.1 + t ∧ N.2 = k * N.1 + t ∧ fixed_point.2 = k * fixed_point.1 + t :=
sorry

end NUMINAMATH_CALUDE_ellipse_fixed_point_theorem_l4061_406189


namespace NUMINAMATH_CALUDE_annual_population_increase_rate_l4061_406156

theorem annual_population_increase_rate (initial_population final_population : ℕ) 
  (h : initial_population = 14000 ∧ final_population = 16940) : 
  ∃ r : ℝ, initial_population * (1 + r)^2 = final_population := by
  sorry

end NUMINAMATH_CALUDE_annual_population_increase_rate_l4061_406156


namespace NUMINAMATH_CALUDE_no_sin_4x_function_of_sin_x_l4061_406124

open Real

theorem no_sin_4x_function_of_sin_x : ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, sin (4 * x) = f (sin x) := by
  sorry

end NUMINAMATH_CALUDE_no_sin_4x_function_of_sin_x_l4061_406124


namespace NUMINAMATH_CALUDE_chord_arithmetic_sequence_l4061_406173

theorem chord_arithmetic_sequence (n : ℕ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 5*x}
  let point := (5/2, 3/2)
  let shortest_chord := 4
  let longest_chord := 5
  ∀ d : ℝ, 1/6 < d ∧ d ≤ 1/3 →
    (n > 0 ∧ 
     shortest_chord + (n - 1) * d = longest_chord ∧
     point ∈ circle) →
    n ∈ ({4, 5, 6} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_chord_arithmetic_sequence_l4061_406173


namespace NUMINAMATH_CALUDE_matrix_power_2023_l4061_406153

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 0; 2, 1]

theorem matrix_power_2023 :
  A^2023 = !![1, 0; 4046, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l4061_406153


namespace NUMINAMATH_CALUDE_order_of_numbers_l4061_406196

theorem order_of_numbers : Real.log 0.76 < 0.76 ∧ 0.76 < 60.7 := by sorry

end NUMINAMATH_CALUDE_order_of_numbers_l4061_406196


namespace NUMINAMATH_CALUDE_polynomial_ratio_l4061_406171

-- Define the polynomial function
def p (x a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : ℝ :=
  a₀ + a₁ * (2 - x) + a₂ * (2 - x)^2 + a₃ * (2 - x)^3 + a₄ * (2 - x)^4 + a₅ * (2 - x)^5

-- State the theorem
theorem polynomial_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 = p x a₀ a₁ a₂ a₃ a₄ a₅) →
  (a₀ + a₂ + a₄) / (a₁ + a₃) = -61/60 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_ratio_l4061_406171


namespace NUMINAMATH_CALUDE_find_number_l4061_406140

theorem find_number : ∃! x : ℕ, 220080 = (x + 445) * (2 * (x - 445)) + 80 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l4061_406140


namespace NUMINAMATH_CALUDE_prime_square_mod_twelve_l4061_406179

theorem prime_square_mod_twelve (p : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) :
  p ^ 2 % 12 = 1 := by
sorry

end NUMINAMATH_CALUDE_prime_square_mod_twelve_l4061_406179


namespace NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l4061_406163

theorem sin_15_cos_15_eq_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l4061_406163


namespace NUMINAMATH_CALUDE_exponent_division_simplification_l4061_406145

theorem exponent_division_simplification (a b : ℝ) :
  (-a * b)^5 / (-a * b)^3 = a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_simplification_l4061_406145


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l4061_406125

def numbers : List ℕ := [1871, 1997, 2020, 2028, 2113, 2125, 2140, 2222, 2300]

theorem mean_of_remaining_numbers :
  (∃ (subset : List ℕ), subset.length = 7 ∧ subset.sum / 7 = 2100 ∧ subset.toFinset ⊆ numbers.toFinset) →
  (numbers.sum - (2100 * 7)) / 2 = 1158 := by
sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l4061_406125


namespace NUMINAMATH_CALUDE_strawberry_rows_l4061_406190

/-- Given that each row of strawberry plants produces 268 kg of fruit
    and the total harvest is 1876 kg, prove that there are 7 rows of strawberry plants. -/
theorem strawberry_rows (yield_per_row : ℕ) (total_harvest : ℕ) 
  (h1 : yield_per_row = 268)
  (h2 : total_harvest = 1876) :
  total_harvest / yield_per_row = 7 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_rows_l4061_406190


namespace NUMINAMATH_CALUDE_syllogism_validity_l4061_406133

theorem syllogism_validity (a b c : Prop) : 
  ((b → c) ∧ (a → b)) → (a → c) := by sorry

end NUMINAMATH_CALUDE_syllogism_validity_l4061_406133


namespace NUMINAMATH_CALUDE_angle_bisector_median_inequality_l4061_406172

variable (a b c : ℝ)
variable (s : ℝ)
variable (f₁ f₂ s₃ : ℝ)

/-- Given a triangle with sides a, b, c, semiperimeter s, 
    angle bisectors f₁ and f₂, and median s₃, 
    prove that f₁ + f₂ + s₃ ≤ √3 * s -/
theorem angle_bisector_median_inequality 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_semiperimeter : s = (a + b + c) / 2)
  (h_f₁ : f₁^2 = (b * c * ((b + c)^2 - a^2)) / (b + c)^2)
  (h_f₂ : f₂^2 = (a * b * ((a + b)^2 - c^2)) / (a + b)^2)
  (h_s₃ : (2 * s₃)^2 = 2 * a^2 + 2 * c^2 - b^2) :
  f₁ + f₂ + s₃ ≤ Real.sqrt 3 * s :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_median_inequality_l4061_406172


namespace NUMINAMATH_CALUDE_complex_calculation_l4061_406131

theorem complex_calculation : 
  let z : ℂ := 1 - Complex.I
  2 / z + z^2 = 1 - Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_calculation_l4061_406131


namespace NUMINAMATH_CALUDE_jose_profit_share_l4061_406146

/-- Calculates the share of profit for an investor given their investment amount, duration, and the total profit and investment-months. -/
def shareOfProfit (investment : ℕ) (duration : ℕ) (totalProfit : ℕ) (totalInvestmentMonths : ℕ) : ℕ :=
  (investment * duration * totalProfit) / totalInvestmentMonths

theorem jose_profit_share (tomInvestment jose_investment : ℕ) (tomDuration joseDuration : ℕ) (totalProfit : ℕ)
    (h1 : tomInvestment = 30000)
    (h2 : jose_investment = 45000)
    (h3 : tomDuration = 12)
    (h4 : joseDuration = 10)
    (h5 : totalProfit = 72000) :
  shareOfProfit jose_investment joseDuration totalProfit (tomInvestment * tomDuration + jose_investment * joseDuration) = 40000 := by
  sorry

#eval shareOfProfit 45000 10 72000 (30000 * 12 + 45000 * 10)

end NUMINAMATH_CALUDE_jose_profit_share_l4061_406146


namespace NUMINAMATH_CALUDE_systematic_sampling_fourth_student_l4061_406136

/-- Systematic sampling function that returns the nth sample given a starting point and interval -/
def systematic_sample (start : Nat) (interval : Nat) (n : Nat) : Nat :=
  start + (n - 1) * interval

theorem systematic_sampling_fourth_student 
  (total_students : Nat) 
  (sample_size : Nat) 
  (sample1 sample2 sample3 : Nat) :
  total_students = 36 →
  sample_size = 4 →
  sample1 = 6 →
  sample2 = 24 →
  sample3 = 33 →
  ∃ (start interval : Nat),
    systematic_sample start interval 1 = sample1 ∧
    systematic_sample start interval 2 = sample2 ∧
    systematic_sample start interval 3 = sample3 ∧
    systematic_sample start interval 4 = 15 :=
by
  sorry

#check systematic_sampling_fourth_student

end NUMINAMATH_CALUDE_systematic_sampling_fourth_student_l4061_406136


namespace NUMINAMATH_CALUDE_min_value_expression_l4061_406161

theorem min_value_expression (x y z k : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hk : k > 0) :
  (k * 4 * z) / (2 * x + y) + (k * 4 * x) / (y + 2 * z) + (k * y) / (x + z) ≥ 3 * k :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l4061_406161


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4061_406137

theorem sufficient_not_necessary (p q : Prop) :
  (¬(p ∨ q) → ¬p) ∧ ¬(¬p → ¬(p ∨ q)) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4061_406137


namespace NUMINAMATH_CALUDE_number_with_special_divisor_property_l4061_406165

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def proper_divisor (d n : ℕ) : Prop := d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def divisor_difference_property (n : ℕ) : Prop :=
  ∀ d₁ d₂ : ℕ, proper_divisor d₁ n → proper_divisor d₂ n → d₁ ≠ d₂ → (d₁ - d₂) ∣ n

theorem number_with_special_divisor_property (n : ℕ) :
  n ≥ 2 →
  (∃ (d₁ d₂ d₃ d₄ : ℕ), d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₄ ∣ n ∧
    ∀ d : ℕ, d ∣ n → d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄) →
  divisor_difference_property n →
  n = 4 ∨ is_prime n :=
sorry

end NUMINAMATH_CALUDE_number_with_special_divisor_property_l4061_406165


namespace NUMINAMATH_CALUDE_lines_perpendicular_imply_parallel_l4061_406105

-- Define a type for lines in 3D space
structure Line3D where
  -- You might want to add more properties here, but for this problem, we only need the line itself
  line : Type

-- Define perpendicularity and parallelism for lines
def perpendicular (l1 l2 : Line3D) : Prop := sorry

def parallel (l1 l2 : Line3D) : Prop := sorry

-- State the theorem
theorem lines_perpendicular_imply_parallel (a b c d : Line3D) 
  (h1 : perpendicular a c)
  (h2 : perpendicular b c)
  (h3 : perpendicular a d)
  (h4 : perpendicular b d) :
  parallel a b ∨ parallel c d := by
  sorry

end NUMINAMATH_CALUDE_lines_perpendicular_imply_parallel_l4061_406105


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l4061_406192

-- Define a right-angled triangle with one side of length 11 and the other two sides being natural numbers
def RightTriangle (a b c : ℕ) : Prop :=
  a = 11 ∧ a^2 + b^2 = c^2

-- Define the perimeter of the triangle
def Perimeter (a b c : ℕ) : ℕ := a + b + c

-- Theorem statement
theorem right_triangle_perimeter :
  ∃ (a b c : ℕ), RightTriangle a b c ∧ Perimeter a b c = 132 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l4061_406192


namespace NUMINAMATH_CALUDE_initial_girls_on_team_l4061_406154

theorem initial_girls_on_team (initial_boys : ℕ) (girls_joined : ℕ) (boys_quit : ℕ) (final_total : ℕ) :
  initial_boys = 15 →
  girls_joined = 7 →
  boys_quit = 4 →
  final_total = 36 →
  ∃ initial_girls : ℕ, initial_girls + initial_boys = final_total - girls_joined + boys_quit :=
by
  sorry

end NUMINAMATH_CALUDE_initial_girls_on_team_l4061_406154


namespace NUMINAMATH_CALUDE_largest_n_binomial_equality_l4061_406142

theorem largest_n_binomial_equality : ∃ (n : ℕ), (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) ∧ 
  (∀ (m : ℕ), Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_n_binomial_equality_l4061_406142


namespace NUMINAMATH_CALUDE_investment_problem_l4061_406114

/-- Proves that given the investment conditions, the amount invested at Speedy Growth Bank is $300 --/
theorem investment_problem (total_investment : ℝ) (speedy_rate : ℝ) (safe_rate : ℝ) (final_amount : ℝ)
  (h1 : total_investment = 1500)
  (h2 : speedy_rate = 0.04)
  (h3 : safe_rate = 0.06)
  (h4 : final_amount = 1584)
  (h5 : ∀ x : ℝ, x * (1 + speedy_rate) + (total_investment - x) * (1 + safe_rate) = final_amount) :
  ∃ x : ℝ, x = 300 ∧ x * (1 + speedy_rate) + (total_investment - x) * (1 + safe_rate) = final_amount :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l4061_406114


namespace NUMINAMATH_CALUDE_circle_center_is_three_halves_thirty_seven_fourths_l4061_406108

/-- A circle passes through (0, 9) and is tangent to y = x^2 at (3, 9) -/
def CircleTangentToParabola (center : ℝ × ℝ) : Prop :=
  let (a, b) := center
  -- Circle passes through (0, 9)
  (a^2 + (b - 9)^2 = a^2 + (b - 9)^2) ∧
  -- Circle is tangent to y = x^2 at (3, 9)
  ((a - 3)^2 + (b - 9)^2 = (a - 0)^2 + (b - 9)^2) ∧
  -- Tangent line to parabola at (3, 9) is perpendicular to line from (3, 9) to center
  ((b - 9) / (a - 3) = -1 / (2 * 3))

/-- The center of the circle is (3/2, 37/4) -/
theorem circle_center_is_three_halves_thirty_seven_fourths :
  CircleTangentToParabola (3/2, 37/4) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_is_three_halves_thirty_seven_fourths_l4061_406108


namespace NUMINAMATH_CALUDE_hotel_nights_calculation_l4061_406118

theorem hotel_nights_calculation (total_value car_value hotel_cost_per_night : ℕ) 
  (h1 : total_value = 158000)
  (h2 : car_value = 30000)
  (h3 : hotel_cost_per_night = 4000) :
  (total_value - (car_value + 4 * car_value)) / hotel_cost_per_night = 2 := by
  sorry

end NUMINAMATH_CALUDE_hotel_nights_calculation_l4061_406118


namespace NUMINAMATH_CALUDE_oil_leak_before_fixing_l4061_406104

theorem oil_leak_before_fixing (total_leak : ℕ) (leak_during_work : ℕ) 
  (h1 : total_leak = 11687)
  (h2 : leak_during_work = 5165) :
  total_leak - leak_during_work = 6522 := by
sorry

end NUMINAMATH_CALUDE_oil_leak_before_fixing_l4061_406104


namespace NUMINAMATH_CALUDE_missing_term_in_geometric_sequence_l4061_406155

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem missing_term_in_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_first : a 1 = 2)
  (h_second : a 2 = 6)
  (h_third : a 3 = 18)
  (h_fourth : a 4 = 54)
  (h_sixth : a 6 = 486) :
  a 5 = 162 := by
sorry


end NUMINAMATH_CALUDE_missing_term_in_geometric_sequence_l4061_406155


namespace NUMINAMATH_CALUDE_min_value_sum_l4061_406103

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  ∃ (m : ℝ), m = 10 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x * y = 1 → 
    (x + 1/x) + (y + 1/y) ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l4061_406103


namespace NUMINAMATH_CALUDE_real_part_expression1_real_part_expression2_real_part_expression3_l4061_406164

open Complex

-- Define the function f that returns the real part of a complex number
def f (z : ℂ) : ℝ := z.re

-- Theorem 1
theorem real_part_expression1 : f ((1 + 2*I)^2 + 3*(1 - I)) / (2 + I) = 1/5 := by sorry

-- Theorem 2
theorem real_part_expression2 : f (1 + (1 - I) / (1 + I)^2 + (1 + I) / (1 - I)^2) = -1 := by sorry

-- Theorem 3
theorem real_part_expression3 : f (1 + (1 - Complex.I * Real.sqrt 3) / (Real.sqrt 3 + I)^2) = 3/4 := by sorry

end NUMINAMATH_CALUDE_real_part_expression1_real_part_expression2_real_part_expression3_l4061_406164


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l4061_406141

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((3*a*b - 6*b + a*(1 - a))^2 + (9*b^2 + 2*a + 3*b*(1 - a))^2) / (a^2 + 9*b^2) ≥ 4 :=
by sorry

theorem min_value_achievable :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ((3*a*b - 6*b + a*(1 - a))^2 + (9*b^2 + 2*a + 3*b*(1 - a))^2) / (a^2 + 9*b^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l4061_406141


namespace NUMINAMATH_CALUDE_twins_age_problem_l4061_406178

theorem twins_age_problem (x : ℕ) : 
  (x + 1) * (x + 1) = x * x + 11 → x = 5 := by
sorry

end NUMINAMATH_CALUDE_twins_age_problem_l4061_406178


namespace NUMINAMATH_CALUDE_tabitha_honey_nights_l4061_406129

/-- Calculates the number of nights Tabitha can enjoy honey in her tea before running out. -/
def honey_nights (servings_per_cup : ℕ) (cups_per_night : ℕ) (container_size : ℕ) (servings_per_ounce : ℕ) : ℕ :=
  let total_servings := container_size * servings_per_ounce
  let servings_per_night := servings_per_cup * cups_per_night
  total_servings / servings_per_night

/-- Proves that Tabitha can enjoy honey in her tea for 48 nights before running out. -/
theorem tabitha_honey_nights : 
  honey_nights 1 2 16 6 = 48 := by
  sorry

end NUMINAMATH_CALUDE_tabitha_honey_nights_l4061_406129


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4061_406159

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4061_406159


namespace NUMINAMATH_CALUDE_garden_area_increase_l4061_406188

/-- Given a rectangular garden with length 60 feet and width 20 feet,
    prove that changing it to a square garden with the same perimeter
    increases the area by 400 square feet. -/
theorem garden_area_increase :
  let rectangular_length : ℝ := 60
  let rectangular_width : ℝ := 20
  let perimeter : ℝ := 2 * (rectangular_length + rectangular_width)
  let square_side : ℝ := perimeter / 4
  let rectangular_area : ℝ := rectangular_length * rectangular_width
  let square_area : ℝ := square_side * square_side
  square_area - rectangular_area = 400 :=
by sorry


end NUMINAMATH_CALUDE_garden_area_increase_l4061_406188


namespace NUMINAMATH_CALUDE_tour_budget_l4061_406109

/-- Given a tour scenario, proves that the total budget for the original tour is 360 units -/
theorem tour_budget (original_days : ℕ) (extension_days : ℕ) (expense_reduction : ℕ) : 
  original_days = 20 → 
  extension_days = 4 → 
  expense_reduction = 3 →
  (original_days * (original_days + extension_days)) / extension_days = 360 :=
by
  sorry

#check tour_budget

end NUMINAMATH_CALUDE_tour_budget_l4061_406109


namespace NUMINAMATH_CALUDE_ribbon_leftover_l4061_406143

theorem ribbon_leftover (total_ribbon : ℕ) (num_gifts : ℕ) (ribbon_per_gift : ℕ) :
  total_ribbon = 18 ∧ num_gifts = 6 ∧ ribbon_per_gift = 2 →
  total_ribbon - (num_gifts * ribbon_per_gift) = 6 := by
sorry

end NUMINAMATH_CALUDE_ribbon_leftover_l4061_406143


namespace NUMINAMATH_CALUDE_tree_height_after_three_years_l4061_406170

/-- Tree growth function -/
def tree_height (initial_height : ℝ) (growth_factor : ℝ) (years : ℕ) : ℝ :=
  initial_height * growth_factor ^ years

theorem tree_height_after_three_years
  (initial_height : ℝ)
  (growth_factor : ℝ)
  (h1 : initial_height = 1)
  (h2 : growth_factor = 3)
  (h3 : tree_height initial_height growth_factor 5 = 243) :
  tree_height initial_height growth_factor 3 = 27 := by
sorry

end NUMINAMATH_CALUDE_tree_height_after_three_years_l4061_406170


namespace NUMINAMATH_CALUDE_initial_boys_count_l4061_406181

/-- The number of boys who went down the slide initially -/
def initial_boys : ℕ := sorry

/-- The number of additional boys who went down the slide -/
def additional_boys : ℕ := 13

/-- The total number of boys who went down the slide -/
def total_boys : ℕ := 35

/-- Theorem stating that the initial number of boys is 22 -/
theorem initial_boys_count : initial_boys = 22 := by
  have h : initial_boys + additional_boys = total_boys := sorry
  sorry

end NUMINAMATH_CALUDE_initial_boys_count_l4061_406181


namespace NUMINAMATH_CALUDE_least_number_of_cookies_l4061_406198

theorem least_number_of_cookies (a : ℕ) : 
  a > 0 ∧ 
  a % 4 = 3 ∧ 
  a % 5 = 2 ∧ 
  a % 7 = 4 ∧ 
  (∀ b : ℕ, b > 0 ∧ b % 4 = 3 ∧ b % 5 = 2 ∧ b % 7 = 4 → a ≤ b) → 
  a = 67 := by
sorry

end NUMINAMATH_CALUDE_least_number_of_cookies_l4061_406198


namespace NUMINAMATH_CALUDE_integer_equation_solution_l4061_406139

theorem integer_equation_solution (x y z : ℤ) :
  x^2 * (y - z) + y^2 * (z - x) + z^2 * (x - y) = 2 ↔ 
  ∃ k : ℤ, x = k + 1 ∧ y = k ∧ z = k - 1 :=
by sorry

end NUMINAMATH_CALUDE_integer_equation_solution_l4061_406139


namespace NUMINAMATH_CALUDE_expression_evaluation_l4061_406121

theorem expression_evaluation : (2^10 * 3^3) / (6 * 2^5) = 144 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4061_406121


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l4061_406111

/-- Two concentric circles with center Q -/
structure ConcentricCircles where
  center : Point
  radius₁ : ℝ
  radius₂ : ℝ
  h : radius₁ < radius₂

/-- The length of an arc given its central angle and the circle's radius -/
def arcLength (angle : ℝ) (radius : ℝ) : ℝ := angle * radius

theorem concentric_circles_area_ratio 
  (circles : ConcentricCircles) 
  (h : arcLength (π/3) circles.radius₁ = arcLength (π/6) circles.radius₂) : 
  (circles.radius₁^2) / (circles.radius₂^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l4061_406111


namespace NUMINAMATH_CALUDE_gcd_78_182_l4061_406101

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := by
  sorry

end NUMINAMATH_CALUDE_gcd_78_182_l4061_406101


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l4061_406150

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℚ)
  (h_arithmetic : ArithmeticSequence a)
  (h_fourth : a 4 = 23)
  (h_sixth : a 6 = 47) :
  a 8 = 71 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l4061_406150


namespace NUMINAMATH_CALUDE_arrow_symmetry_axis_l4061_406182

/-- Represents a point on a 2D grid --/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a geometric figure on a grid --/
structure GeometricFigure where
  points : Set GridPoint

/-- Represents a line on a grid --/
inductive GridLine
  | Vertical : Int → GridLine
  | Horizontal : Int → GridLine
  | DiagonalTopLeftToBottomRight : GridLine
  | DiagonalBottomLeftToTopRight : GridLine

/-- Predicate to check if a figure is arrow-shaped --/
def isArrowShaped (figure : GeometricFigure) : Prop := sorry

/-- Predicate to check if a line is an axis of symmetry for a figure --/
def isAxisOfSymmetry (line : GridLine) (figure : GeometricFigure) : Prop := sorry

/-- Theorem: An arrow-shaped figure with only one axis of symmetry has a vertical line through the center as its axis of symmetry --/
theorem arrow_symmetry_axis (figure : GeometricFigure) (h1 : isArrowShaped figure) 
    (h2 : ∃! (line : GridLine), isAxisOfSymmetry line figure) : 
    ∃ (x : Int), isAxisOfSymmetry (GridLine.Vertical x) figure := by
  sorry

end NUMINAMATH_CALUDE_arrow_symmetry_axis_l4061_406182


namespace NUMINAMATH_CALUDE_jesse_room_area_l4061_406134

/-- The area of a rectangular room -/
def room_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of Jesse's room is 96 square feet -/
theorem jesse_room_area : room_area 12 8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_jesse_room_area_l4061_406134


namespace NUMINAMATH_CALUDE_original_savings_calculation_l4061_406180

theorem original_savings_calculation (savings : ℚ) : 
  (3 / 4 : ℚ) * savings + (1 / 4 : ℚ) * savings = savings ∧ 
  (1 / 4 : ℚ) * savings = 240 → 
  savings = 960 := by
  sorry

end NUMINAMATH_CALUDE_original_savings_calculation_l4061_406180


namespace NUMINAMATH_CALUDE_friends_assignment_l4061_406197

-- Define the types for names, surnames, and grades
inductive Name : Type
  | Petya | Kolya | Alyosha | Misha | Dima | Borya | Vasya

inductive Surname : Type
  | Ivanov | Petrov | Krylov | Orlov

inductive Grade : Type
  | First | Second | Third | Fourth

-- Define a function to represent the assignment of names, surnames, and grades
def Assignment := Name → Surname × Grade

-- Define the conditions
def not_first_grader (a : Assignment) (n : Name) : Prop :=
  (a n).2 ≠ Grade.First

def different_streets (a : Assignment) (n1 n2 : Name) : Prop :=
  (a n1).1 ≠ (a n2).1

def one_year_older (a : Assignment) (n1 n2 : Name) : Prop :=
  match (a n1).2, (a n2).2 with
  | Grade.Second, Grade.First => True
  | Grade.Third, Grade.Second => True
  | Grade.Fourth, Grade.Third => True
  | _, _ => False

def neighbors (a : Assignment) (n1 n2 : Name) : Prop :=
  (a n1).1 = (a n2).1

def met_year_ago_first_grade (a : Assignment) (n : Name) : Prop :=
  (a n).2 = Grade.Second

def gave_last_year_textbook (a : Assignment) (n1 n2 : Name) : Prop :=
  match (a n1).2, (a n2).2 with
  | Grade.Second, Grade.First => True
  | Grade.Third, Grade.Second => True
  | Grade.Fourth, Grade.Third => True
  | _, _ => False

-- Define the theorem
theorem friends_assignment (a : Assignment) :
  not_first_grader a Name.Borya
  ∧ different_streets a Name.Vasya Name.Dima
  ∧ one_year_older a Name.Misha Name.Dima
  ∧ neighbors a Name.Borya Name.Vasya
  ∧ met_year_ago_first_grade a Name.Misha
  ∧ gave_last_year_textbook a Name.Vasya Name.Borya
  → a Name.Dima = (Surname.Ivanov, Grade.First)
  ∧ a Name.Misha = (Surname.Krylov, Grade.Second)
  ∧ a Name.Borya = (Surname.Petrov, Grade.Third)
  ∧ a Name.Vasya = (Surname.Orlov, Grade.Fourth) :=
by
  sorry

end NUMINAMATH_CALUDE_friends_assignment_l4061_406197


namespace NUMINAMATH_CALUDE_bike_speed_l4061_406135

/-- Proves that a bike moving at constant speed, covering 32 meters in 8 seconds, has a speed of 4 meters per second -/
theorem bike_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 32 → time = 8 → speed = distance / time → speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_bike_speed_l4061_406135
