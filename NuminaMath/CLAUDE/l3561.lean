import Mathlib

namespace solution_set_f_less_than_8_range_of_a_for_nonempty_solution_l3561_356103

-- Define the function f(x)
def f (x : ℝ) : ℝ := |4*x - 1| - |x + 2|

-- Theorem for the solution set of f(x) < 8
theorem solution_set_f_less_than_8 :
  {x : ℝ | f x < 8} = {x : ℝ | -9/5 < x ∧ x < 11/3} :=
sorry

-- Theorem for the range of a
theorem range_of_a_for_nonempty_solution (a : ℝ) :
  (∃ x : ℝ, f x + 5*|x + 2| < a^2 - 8*a) ↔ (a < -1 ∨ a > 9) :=
sorry

end solution_set_f_less_than_8_range_of_a_for_nonempty_solution_l3561_356103


namespace bowl_game_score_l3561_356188

/-- Given the scores of Noa, Phillip, and Lucy in a bowl game, prove their total score. -/
theorem bowl_game_score (noa_score : ℕ) (phillip_score : ℕ) (lucy_score : ℕ) 
  (h1 : noa_score = 30)
  (h2 : phillip_score = 2 * noa_score)
  (h3 : lucy_score = (3 : ℕ) / 2 * phillip_score) :
  noa_score + phillip_score + lucy_score = 180 := by
  sorry

end bowl_game_score_l3561_356188


namespace junior_boy_girl_ratio_l3561_356162

/-- Represents the number of participants in each category -/
structure Participants where
  juniorBoys : ℕ
  seniorBoys : ℕ
  juniorGirls : ℕ
  seniorGirls : ℕ

/-- The ratio of boys to total participants is 55% -/
def boyRatio (p : Participants) : Prop :=
  (p.juniorBoys + p.seniorBoys : ℚ) / (p.juniorBoys + p.seniorBoys + p.juniorGirls + p.seniorGirls) = 55 / 100

/-- The ratio of junior boys to senior boys equals the ratio of all juniors to all seniors -/
def juniorSeniorRatio (p : Participants) : Prop :=
  (p.juniorBoys : ℚ) / p.seniorBoys = (p.juniorBoys + p.juniorGirls : ℚ) / (p.seniorBoys + p.seniorGirls)

/-- The main theorem: given the conditions, prove that the ratio of junior boys to junior girls is 11:9 -/
theorem junior_boy_girl_ratio (p : Participants) 
  (hBoyRatio : boyRatio p) (hJuniorSeniorRatio : juniorSeniorRatio p) : 
  (p.juniorBoys : ℚ) / p.juniorGirls = 11 / 9 := by
  sorry

end junior_boy_girl_ratio_l3561_356162


namespace quadratic_points_relation_l3561_356187

-- Define the quadratic function
def f (x m : ℝ) : ℝ := x^2 - 2*x + m

-- Define the points A, B, and C
def A (m : ℝ) : ℝ × ℝ := (-4, f (-4) m)
def B (m : ℝ) : ℝ × ℝ := (0, f 0 m)
def C (m : ℝ) : ℝ × ℝ := (3, f 3 m)

-- State the theorem
theorem quadratic_points_relation (m : ℝ) :
  (B m).2 < (C m).2 ∧ (C m).2 < (A m).2 := by
  sorry

end quadratic_points_relation_l3561_356187


namespace problem_statement_l3561_356110

theorem problem_statement (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h : (x * Real.sin (π/5) + y * Real.cos (π/5)) / (x * Real.cos (π/5) - y * Real.sin (π/5)) = Real.tan (9*π/20)) :
  (y / x = 1) ∧
  (∃ (A B C : ℝ), Real.tan C = y / x ∧ 
    (∀ A' B' C' : ℝ, Real.tan C' = y / x → 
      Real.sin (2*A') + 2 * Real.cos B' ≤ Real.sin (2*A) + 2 * Real.cos B) ∧
    Real.sin (2*A) + 2 * Real.cos B = 3/2) := by
  sorry

end problem_statement_l3561_356110


namespace binomial_12_choose_10_l3561_356150

theorem binomial_12_choose_10 : Nat.choose 12 10 = 66 := by
  sorry

end binomial_12_choose_10_l3561_356150


namespace triangle_is_acute_l3561_356197

theorem triangle_is_acute (a b c : ℝ) (ha : a = 4) (hb : b = 5) (hc : c = 6) :
  a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2 := by
  sorry

#check triangle_is_acute

end triangle_is_acute_l3561_356197


namespace specific_figure_perimeter_l3561_356104

/-- Calculates the perimeter of a figure composed of a central square and four smaller squares attached to its sides. -/
def figure_perimeter (central_side_length : ℝ) (small_side_length : ℝ) : ℝ :=
  4 * central_side_length + 4 * (3 * small_side_length)

/-- Theorem stating that the perimeter of the specific figure is 140 -/
theorem specific_figure_perimeter :
  figure_perimeter 20 5 = 140 := by
  sorry

#eval figure_perimeter 20 5

end specific_figure_perimeter_l3561_356104


namespace symmetrical_point_l3561_356154

/-- Given a point (m, m+1) and a line of symmetry x=3, 
    the symmetrical point is (6-m, m+1) --/
theorem symmetrical_point (m : ℝ) : 
  let original_point := (m, m+1)
  let line_of_symmetry := 3
  let symmetrical_point := (6-m, m+1)
  symmetrical_point = 
    (2 * line_of_symmetry - original_point.1, original_point.2) := by
  sorry

end symmetrical_point_l3561_356154


namespace equal_area_rectangles_width_l3561_356101

/-- Proves that given two rectangles with equal area, where one rectangle has dimensions 12 inches 
    by W inches, and the other rectangle has dimensions 6 inches by 30 inches, the value of W is 15 inches. -/
theorem equal_area_rectangles_width (W : ℝ) : 
  (12 * W = 6 * 30) → W = 15 := by
  sorry

end equal_area_rectangles_width_l3561_356101


namespace dog_turn_point_sum_cat_dog_problem_l3561_356149

/-- The point where the dog starts moving away from the cat -/
def dog_turn_point (cat_x cat_y : ℚ) (dog_line_slope dog_line_intercept : ℚ) : ℚ × ℚ :=
  sorry

/-- Theorem stating that the sum of coordinates of the turn point is 243/68 -/
theorem dog_turn_point_sum (cat_x cat_y : ℚ) (dog_line_slope dog_line_intercept : ℚ) :
  let (c, d) := dog_turn_point cat_x cat_y dog_line_slope dog_line_intercept
  c + d = 243/68 := by sorry

/-- Main theorem proving the specific case in the problem -/
theorem cat_dog_problem :
  let (c, d) := dog_turn_point 15 12 (-4) 15
  c + d = 243/68 := by sorry

end dog_turn_point_sum_cat_dog_problem_l3561_356149


namespace sum_C_D_equals_negative_ten_l3561_356125

variable (x : ℝ)
variable (C D : ℝ)

theorem sum_C_D_equals_negative_ten :
  (∀ x ≠ 3, C / (x - 3) + D * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) →
  C + D = -10 := by
sorry

end sum_C_D_equals_negative_ten_l3561_356125


namespace money_division_l3561_356183

/-- Proves that the total amount of money divided among A, B, and C is 980,
    given the specified conditions. -/
theorem money_division (a b c : ℕ) : 
  b = 290 →            -- B's share is 290
  a = b + 40 →         -- A has 40 more than B
  c = a + 30 →         -- C has 30 more than A
  a + b + c = 980 :=   -- Total amount is 980
by
  sorry


end money_division_l3561_356183


namespace number_of_girls_l3561_356108

theorem number_of_girls (total_pupils : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_pupils = 929 → boys = 387 → girls = total_pupils - boys → girls = 542 := by
sorry

end number_of_girls_l3561_356108


namespace bill_face_value_l3561_356119

/-- Calculates the face value of a bill given true discount, time, and interest rate. -/
def face_value (true_discount : ℚ) (time_months : ℚ) (interest_rate : ℚ) : ℚ :=
  (true_discount * 100) / (interest_rate * (time_months / 12))

/-- Theorem stating that given the specified conditions, the face value of the bill is 1575. -/
theorem bill_face_value :
  let true_discount : ℚ := 189
  let time_months : ℚ := 9
  let interest_rate : ℚ := 16
  face_value true_discount time_months interest_rate = 1575 := by
  sorry


end bill_face_value_l3561_356119


namespace problem_statement_l3561_356173

/-- The set D of positive real pairs (x₁, x₂) that sum to k -/
def D (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 = k}

theorem problem_statement (k : ℝ) (hk : k > 0) :
  (∀ p ∈ D k, 0 < p.1 * p.2 ∧ p.1 * p.2 ≤ k^2 / 4) ∧
  (k ≥ 1 → ∀ p ∈ D k, (1 / p.1 - p.1) * (1 / p.2 - p.2) ≤ (k / 2 - 2 / k)^2) ∧
  (∀ p ∈ D k, (1 / p.1 - p.1) * (1 / p.2 - p.2) ≥ (k / 2 - 2 / k)^2 ↔ 0 < k^2 ∧ k^2 ≤ 4 * Real.sqrt 5 - 8) :=
by sorry

end problem_statement_l3561_356173


namespace loan_repayment_months_l3561_356122

/-- Represents the monthly income in ten thousands of yuan -/
def monthlyIncome : ℕ → ℚ
  | 0 => 20  -- First month's income
  | n + 1 => if n < 5 then monthlyIncome n * 1.2 else monthlyIncome n + 2

/-- Calculates the cumulative income up to month n -/
def cumulativeIncome (n : ℕ) : ℚ :=
  (List.range n).map monthlyIncome |>.sum

/-- The loan amount in ten thousands of yuan -/
def loanAmount : ℚ := 400

theorem loan_repayment_months :
  (∀ k < 10, cumulativeIncome k < loanAmount) ∧
  cumulativeIncome 10 ≥ loanAmount := by
  sorry

#eval cumulativeIncome 10  -- For verification

end loan_repayment_months_l3561_356122


namespace p_properties_l3561_356198

-- Define the polynomial
def p (x y : ℝ) : ℝ := x^2 * y^3 - 3 * x * y^3 - 2

-- Define the degree of a monomial
def monomial_degree (m : ℕ × ℕ) : ℕ := m.1 + m.2

-- Define the degree of a polynomial
def polynomial_degree (p : ℝ → ℝ → ℝ) : ℕ :=
  -- This definition is a placeholder and needs to be implemented
  sorry

-- Define the number of terms in a polynomial
def number_of_terms (p : ℝ → ℝ → ℝ) : ℕ :=
  -- This definition is a placeholder and needs to be implemented
  sorry

-- Theorem stating the properties of the polynomial p
theorem p_properties :
  polynomial_degree p = 5 ∧ number_of_terms p = 3 := by
  sorry

end p_properties_l3561_356198


namespace melted_prism_to_cube_l3561_356164

-- Define the prism's properties
def prism_base_area : Real := 16
def prism_height : Real := 4

-- Define the volume of the prism
def prism_volume : Real := prism_base_area * prism_height

-- Define the edge length of the resulting cube
def cube_edge_length : Real := 4

-- Theorem statement
theorem melted_prism_to_cube :
  prism_volume = cube_edge_length ^ 3 :=
by
  sorry

#check melted_prism_to_cube

end melted_prism_to_cube_l3561_356164


namespace letterbox_strip_height_calculation_l3561_356167

/-- Represents a screen with width, height, and diagonal measurements -/
structure Screen where
  width : ℝ
  height : ℝ
  diagonal : ℝ

/-- Represents an aspect ratio as a pair of numbers -/
structure AspectRatio where
  horizontal : ℝ
  vertical : ℝ

/-- Calculates the height of letterbox strips when showing a movie on a TV -/
def letterboxStripHeight (tv : Screen) (movieRatio : AspectRatio) : ℝ :=
  sorry

theorem letterbox_strip_height_calculation 
  (tv : Screen)
  (movieRatio : AspectRatio)
  (h1 : tv.diagonal = 27)
  (h2 : tv.width / tv.height = 4 / 3)
  (h3 : movieRatio.horizontal / movieRatio.vertical = 2 / 1)
  (h4 : tv.width^2 + tv.height^2 = tv.diagonal^2) :
  letterboxStripHeight tv movieRatio = 2.7 := by
  sorry

end letterbox_strip_height_calculation_l3561_356167


namespace function_properties_l3561_356132

open Real

noncomputable def f (a b x : ℝ) : ℝ := a * log x - b * x^2 + 1

theorem function_properties (a b : ℝ) :
  (∀ x > 0, deriv (f a b) x = 3 → f a b 1 = 1/2) →
  (a = 4 ∧ b = 1/2) ∧
  (∀ x ∈ Set.Icc (1/ℯ) (ℯ^2), f 4 (1/2) x ≤ 4 * log 2 - 1) ∧
  (∃ x ∈ Set.Icc (1/ℯ) (ℯ^2), f 4 (1/2) x = 4 * log 2 - 1) :=
by sorry

end function_properties_l3561_356132


namespace police_emergency_number_prime_divisor_l3561_356182

/-- A police emergency number is a positive integer that ends in 133 in decimal representation. -/
def PoliceEmergencyNumber (n : ℕ) : Prop :=
  n > 0 ∧ n % 1000 = 133

/-- Every police emergency number has a prime divisor greater than 7. -/
theorem police_emergency_number_prime_divisor (n : ℕ) (h : PoliceEmergencyNumber n) :
  ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n := by
  sorry

end police_emergency_number_prime_divisor_l3561_356182


namespace min_value_fraction_l3561_356145

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 2) :
  (2 * x + y) / (x * y) ≥ (7 + 2 * Real.sqrt 6) / 2 :=
sorry

end min_value_fraction_l3561_356145


namespace jewelry_price_increase_is_10_l3561_356124

/-- Represents the increase in price of jewelry -/
def jewelry_price_increase : ℝ := sorry

/-- Original price of jewelry -/
def original_jewelry_price : ℝ := 30

/-- Original price of paintings -/
def original_painting_price : ℝ := 100

/-- New price of paintings after 20% increase -/
def new_painting_price : ℝ := original_painting_price * 1.2

/-- Total cost for 2 pieces of jewelry and 5 paintings -/
def total_cost : ℝ := 680

theorem jewelry_price_increase_is_10 :
  2 * (original_jewelry_price + jewelry_price_increase) + 5 * new_painting_price = total_cost ∧
  jewelry_price_increase = 10 := by sorry

end jewelry_price_increase_is_10_l3561_356124


namespace perimeter_ABCDEHG_l3561_356107

-- Define the points
variable (A B C D E F G H : ℝ × ℝ)

-- Define the conditions
def is_equilateral (X Y Z : ℝ × ℝ) : Prop := sorry
def is_midpoint (M X Y : ℝ × ℝ) : Prop := sorry
def distance (X Y : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem perimeter_ABCDEHG :
  is_equilateral A B C →
  is_equilateral A D E →
  is_equilateral E F G →
  is_midpoint D A C →
  is_midpoint H A E →
  distance A B = 6 →
  distance A B + distance B C + distance C D + distance D E +
  distance E F + distance F G + distance G H + distance H A = 22.5 := by
  sorry

end perimeter_ABCDEHG_l3561_356107


namespace special_polygon_sum_angles_l3561_356170

/-- A polygon where 3 diagonals can be drawn from one vertex -/
structure SpecialPolygon where
  /-- The number of diagonals that can be drawn from one vertex -/
  diagonals_from_vertex : ℕ
  /-- The condition that 3 diagonals can be drawn from one vertex -/
  diag_condition : diagonals_from_vertex = 3

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- Theorem: The sum of interior angles of a SpecialPolygon is 720° -/
theorem special_polygon_sum_angles (p : SpecialPolygon) : 
  sum_interior_angles (p.diagonals_from_vertex + 3) = 720 := by
  sorry

end special_polygon_sum_angles_l3561_356170


namespace exists_piecewise_linear_involution_l3561_356117

/-- A piecewise-linear function is a function whose graph is a union of a finite number of points and line segments. -/
def PiecewiseLinear (f : ℝ → ℝ) : Prop := sorry

theorem exists_piecewise_linear_involution :
  ∃ (f : ℝ → ℝ), PiecewiseLinear f ∧ 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f x ∈ Set.Icc (-1) 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f (f x) = -x) :=
sorry

end exists_piecewise_linear_involution_l3561_356117


namespace platyfish_white_balls_l3561_356174

theorem platyfish_white_balls :
  let total_balls : ℕ := 80
  let num_goldfish : ℕ := 3
  let red_balls_per_goldfish : ℕ := 10
  let num_platyfish : ℕ := 10
  let total_red_balls : ℕ := num_goldfish * red_balls_per_goldfish
  let total_white_balls : ℕ := total_balls - total_red_balls
  let white_balls_per_platyfish : ℕ := total_white_balls / num_platyfish
  white_balls_per_platyfish = 5 :=
by
  sorry

end platyfish_white_balls_l3561_356174


namespace power_fraction_equality_l3561_356134

theorem power_fraction_equality : (2^2014 + 2^2012) / (2^2014 - 2^2012) = 5/3 := by
  sorry

end power_fraction_equality_l3561_356134


namespace arithmetic_square_root_of_four_l3561_356116

theorem arithmetic_square_root_of_four : ∃ x : ℝ, x ≥ 0 ∧ x^2 = 4 ∧ ∀ y : ℝ, y ≥ 0 ∧ y^2 = 4 → y = x :=
by sorry

end arithmetic_square_root_of_four_l3561_356116


namespace integer_root_values_l3561_356178

def polynomial (a x : ℤ) : ℤ := x^3 - 2*x^2 + a*x + 8

def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, polynomial a x = 0

theorem integer_root_values : 
  {a : ℤ | has_integer_root a} = {-49, -47, -22, -10, -7, 4, 9, 16} := by sorry

end integer_root_values_l3561_356178


namespace softball_team_ratio_l3561_356176

theorem softball_team_ratio :
  ∀ (men women : ℕ),
  women = men + 4 →
  men + women = 14 →
  (men : ℚ) / women = 5 / 9 :=
by
  sorry

end softball_team_ratio_l3561_356176


namespace smallest_divisor_power_l3561_356193

def polynomial (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_divisor_power : ∃! k : ℕ+, 
  (∀ z : ℂ, polynomial z ∣ (z^k.val - 1)) ∧ 
  (∀ m : ℕ+, m < k → ∃ z : ℂ, ¬(polynomial z ∣ (z^m.val - 1))) ∧
  k = 84 := by sorry

end smallest_divisor_power_l3561_356193


namespace mode_identifies_favorite_dish_l3561_356189

/-- A statistical measure for a dataset -/
inductive StatisticalMeasure
  | Mean
  | Median
  | Mode
  | Variance

/-- A dataset representing student preferences for dishes at a food festival -/
structure FoodFestivalData where
  preferences : List String

/-- Definition: The mode of a dataset is the most frequently occurring value -/
def mode (data : FoodFestivalData) : String :=
  sorry

/-- The statistical measure that identifies the favorite dish at a food festival -/
def favoriteDishMeasure : StatisticalMeasure :=
  sorry

/-- Theorem: The mode is the appropriate measure for identifying the favorite dish -/
theorem mode_identifies_favorite_dish :
  favoriteDishMeasure = StatisticalMeasure.Mode :=
  sorry

end mode_identifies_favorite_dish_l3561_356189


namespace restaurant_customer_prediction_l3561_356192

theorem restaurant_customer_prediction 
  (breakfast_customers : ℕ) 
  (lunch_customers : ℕ) 
  (dinner_customers : ℕ) 
  (h1 : breakfast_customers = 73)
  (h2 : lunch_customers = 127)
  (h3 : dinner_customers = 87) :
  2 * (breakfast_customers + lunch_customers + dinner_customers) = 574 :=
by sorry

end restaurant_customer_prediction_l3561_356192


namespace crayons_in_box_l3561_356161

def initial_crayons : ℕ := 7
def added_crayons : ℕ := 6

theorem crayons_in_box : initial_crayons + added_crayons = 13 := by
  sorry

end crayons_in_box_l3561_356161


namespace expression_simplification_l3561_356115

theorem expression_simplification (m : ℝ) (h : m^2 + 3*m - 2 = 0) :
  (m - 3) / (3 * m^2 - 6*m) / (m + 2 - 5 / (m - 2)) = 1/6 := by sorry

end expression_simplification_l3561_356115


namespace correct_calculation_l3561_356184

theorem correct_calculation (x : ℝ) : 3 * x = 135 → x / 3 = 15 := by
  sorry

end correct_calculation_l3561_356184


namespace josh_initial_money_l3561_356105

/-- The cost of the candy bar in dollars -/
def candy_cost : ℚ := 45 / 100

/-- The change Josh received in dollars -/
def change_received : ℚ := 135 / 100

/-- The initial amount of money Josh had -/
def initial_money : ℚ := candy_cost + change_received

theorem josh_initial_money : initial_money = 180 / 100 := by
  sorry

end josh_initial_money_l3561_356105


namespace cents_left_over_l3561_356166

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the number of pennies in the jar -/
def num_pennies : ℕ := 123

/-- Represents the number of nickels in the jar -/
def num_nickels : ℕ := 85

/-- Represents the number of dimes in the jar -/
def num_dimes : ℕ := 35

/-- Represents the number of quarters in the jar -/
def num_quarters : ℕ := 26

/-- Represents the cost of a double scoop in dollars -/
def double_scoop_cost : ℕ := 3

/-- Represents the number of family members -/
def num_family_members : ℕ := 5

/-- Theorem stating that the number of cents left over after the trip is 48 -/
theorem cents_left_over : 
  (num_pennies * penny_value + 
   num_nickels * nickel_value + 
   num_dimes * dime_value + 
   num_quarters * quarter_value) - 
  (double_scoop_cost * num_family_members * cents_per_dollar) = 48 := by
  sorry


end cents_left_over_l3561_356166


namespace sum_of_two_and_four_l3561_356141

theorem sum_of_two_and_four : 2 + 4 = 6 := by
  sorry

end sum_of_two_and_four_l3561_356141


namespace shopkeeper_profit_percentage_l3561_356175

theorem shopkeeper_profit_percentage 
  (cost_price : ℝ) 
  (cost_price_positive : cost_price > 0) :
  let markup_percentage : ℝ := 30
  let discount_percentage : ℝ := 18.461538461538467
  let marked_price : ℝ := cost_price * (1 + markup_percentage / 100)
  let selling_price : ℝ := marked_price * (1 - discount_percentage / 100)
  let profit : ℝ := selling_price - cost_price
  let profit_percentage : ℝ := (profit / cost_price) * 100
  profit_percentage = 6 := by sorry

end shopkeeper_profit_percentage_l3561_356175


namespace inequality_solution_condition_sum_a_b_l3561_356199

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|
def g (a x : ℝ) : ℝ := a - |x - 2|

-- Theorem for part 1
theorem inequality_solution_condition (a : ℝ) :
  (∃ x, f x < g a x) ↔ a > 4 :=
sorry

-- Theorem for part 2
theorem sum_a_b (a b : ℝ) :
  (∀ x, f x < g a x ↔ b < x ∧ x < 7/2) →
  a + b = 6 :=
sorry

end inequality_solution_condition_sum_a_b_l3561_356199


namespace figure_area_is_61_l3561_356157

/-- Calculates the area of a figure composed of three rectangles -/
def figure_area (rect1_height rect1_width rect2_height rect2_width rect3_height rect3_width : ℕ) : ℕ :=
  rect1_height * rect1_width + rect2_height * rect2_width + rect3_height * rect3_width

/-- The area of the given figure is 61 square units -/
theorem figure_area_is_61 : figure_area 7 6 3 3 2 5 = 61 := by
  sorry

end figure_area_is_61_l3561_356157


namespace weighted_sum_square_inequality_l3561_356137

theorem weighted_sum_square_inequality (a b x y : ℝ) 
  (h1 : a + b = 1) (h2 : a ≥ 0) (h3 : b ≥ 0) : 
  a * x^2 + b * y^2 - (a * x + b * y)^2 ≥ 0 :=
by sorry

end weighted_sum_square_inequality_l3561_356137


namespace stockholm_uppsala_distance_l3561_356163

/-- Represents the distance on a map in centimeters -/
def map_distance : ℝ := 65

/-- Represents the scale factor of the map (km per cm) -/
def scale_factor : ℝ := 20

/-- Calculates the actual distance in kilometers given the map distance and scale factor -/
def actual_distance (map_dist : ℝ) (scale : ℝ) : ℝ := map_dist * scale

theorem stockholm_uppsala_distance :
  actual_distance map_distance scale_factor = 1300 := by
  sorry

end stockholm_uppsala_distance_l3561_356163


namespace club_juniors_count_l3561_356171

theorem club_juniors_count :
  ∀ (j s x y : ℕ),
  -- Total students in the club
  j + s = 36 →
  -- Juniors on science team
  x = (40 * j) / 100 →
  -- Seniors on science team
  y = (25 * s) / 100 →
  -- Twice as many juniors as seniors on science team
  x = 2 * y →
  -- Conclusion: number of juniors is 20
  j = 20 :=
by
  sorry

end club_juniors_count_l3561_356171


namespace f_odd_and_decreasing_l3561_356196

-- Define the function f(x) = -x|x|
def f (x : ℝ) : ℝ := -x * abs x

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f y < f x) :=
by sorry


end f_odd_and_decreasing_l3561_356196


namespace min_trees_chopped_is_270_l3561_356118

def min_trees_chopped (axe_resharpen_interval : ℕ) (saw_regrind_interval : ℕ)
  (axe_sharpen_cost : ℕ) (saw_regrind_cost : ℕ)
  (total_axe_sharpen_cost : ℕ) (total_saw_regrind_cost : ℕ) : ℕ :=
  let axe_sharpenings := (total_axe_sharpen_cost + axe_sharpen_cost - 1) / axe_sharpen_cost
  let saw_regrindings := total_saw_regrind_cost / saw_regrind_cost
  axe_sharpenings * axe_resharpen_interval + saw_regrindings * saw_regrind_interval

theorem min_trees_chopped_is_270 :
  min_trees_chopped 25 20 8 10 46 60 = 270 := by
  sorry

end min_trees_chopped_is_270_l3561_356118


namespace square_room_tiles_l3561_356139

theorem square_room_tiles (n : ℕ) : 
  n > 0 →  -- Ensure the room has a positive size
  (2 * n = 62) →  -- Total tiles on both diagonals
  n * n = 961  -- Total tiles in the room
  := by sorry

end square_room_tiles_l3561_356139


namespace smallest_solution_correct_l3561_356109

noncomputable def smallest_solution : ℝ := 4 - Real.sqrt 15 / 3

theorem smallest_solution_correct :
  let x := smallest_solution
  (1 / (x - 3) + 1 / (x - 5) = 5 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 5 / (y - 4)) → y ≥ x :=
by sorry

end smallest_solution_correct_l3561_356109


namespace simplify_trig_expression_1_simplify_trig_expression_2_l3561_356131

-- Part 1
theorem simplify_trig_expression_1 :
  (Real.sin (35 * π / 180))^2 - (1/2) = -Real.cos (10 * π / 180) * Real.cos (80 * π / 180) := by
  sorry

-- Part 2
theorem simplify_trig_expression_2 (α : ℝ) :
  (1 / Real.tan (α/2) - Real.tan (α/2)) * ((1 - Real.cos (2*α)) / Real.sin (2*α)) = 2 := by
  sorry

end simplify_trig_expression_1_simplify_trig_expression_2_l3561_356131


namespace workbook_selection_cases_l3561_356100

/-- The number of cases to choose either a Korean workbook or a math workbook -/
def total_cases (korean_books : ℕ) (math_books : ℕ) : ℕ :=
  korean_books + math_books

/-- Theorem: Given 2 types of Korean workbooks and 4 types of math workbooks,
    the total number of cases to choose either a Korean workbook or a math workbook is 6 -/
theorem workbook_selection_cases : total_cases 2 4 = 6 := by
  sorry

end workbook_selection_cases_l3561_356100


namespace quadratic_real_roots_l3561_356140

-- Define the quadratic equation
def quadratic_equation (x k : ℝ) : ℝ := x^2 + 2*k*x + k

-- Theorem statement
theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, quadratic_equation x k = 0) ↔ (k ≤ 0 ∨ k ≥ 1) :=
by sorry

end quadratic_real_roots_l3561_356140


namespace find_number_l3561_356123

theorem find_number (k : ℝ) (x : ℝ) (h1 : x / k = 4) (h2 : k = 16) : x = 64 := by
  sorry

end find_number_l3561_356123


namespace animal_arrangement_count_l3561_356152

def num_chickens : ℕ := 3
def num_dogs : ℕ := 3
def num_cats : ℕ := 4
def num_rabbits : ℕ := 2
def total_animals : ℕ := num_chickens + num_dogs + num_cats + num_rabbits

def arrangement_count : ℕ := 41472

theorem animal_arrangement_count :
  (Nat.factorial 4) * 
  (Nat.factorial num_chickens) * 
  (Nat.factorial num_dogs) * 
  (Nat.factorial num_cats) * 
  (Nat.factorial num_rabbits) = arrangement_count :=
by sorry

end animal_arrangement_count_l3561_356152


namespace phone_number_fraction_l3561_356112

def is_valid_phone_number (n : ℕ) : Prop :=
  1000000 ≤ n ∧ n < 10000000 ∧ n / 1000000 ≥ 3

def starts_with_3_to_9_ends_even (n : ℕ) : Prop :=
  is_valid_phone_number n ∧ n % 2 = 0

def count_valid_numbers : ℕ := 7 * 10^6

def count_start_3_to_9_end_even : ℕ := 7 * 10^5 * 5

theorem phone_number_fraction :
  (count_start_3_to_9_end_even : ℚ) / count_valid_numbers = 1 / 2 :=
sorry

end phone_number_fraction_l3561_356112


namespace consecutive_integers_product_sum_l3561_356143

theorem consecutive_integers_product_sum (a b c d : ℕ) : 
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ a * b * c * d = 5040 → a + b + c + d = 34 := by
  sorry

end consecutive_integers_product_sum_l3561_356143


namespace complex_equation_solution_l3561_356148

theorem complex_equation_solution (x y : ℂ) (hx : x ≠ 0) (hxy : x + 2*y ≠ 0) :
  (x + 2*y) / x = 2*y / (x + 2*y) →
  (x = -y + Complex.I * Real.sqrt 3 * y) ∨ (x = -y - Complex.I * Real.sqrt 3 * y) :=
by sorry

end complex_equation_solution_l3561_356148


namespace insufficient_funds_for_all_l3561_356186

theorem insufficient_funds_for_all 
  (num_workers : ℕ) 
  (total_salary : ℕ) 
  (item_cost : ℕ) 
  (h1 : num_workers = 5) 
  (h2 : total_salary = 1500) 
  (h3 : item_cost = 320) : 
  ∃ (worker : ℕ), worker ≤ num_workers ∧ total_salary < num_workers * item_cost :=
sorry

end insufficient_funds_for_all_l3561_356186


namespace checkerboard_inner_probability_l3561_356168

/-- The size of one side of the square checkerboard -/
def boardSize : ℕ := 10

/-- The total number of squares on the checkerboard -/
def totalSquares : ℕ := boardSize * boardSize

/-- The number of squares on the perimeter of the checkerboard -/
def perimeterSquares : ℕ := 4 * boardSize - 4

/-- The number of squares not touching the outer edge -/
def innerSquares : ℕ := totalSquares - perimeterSquares

/-- The probability of choosing a square not touching the outer edge -/
def innerProbability : ℚ := innerSquares / totalSquares

theorem checkerboard_inner_probability :
  innerProbability = 16 / 25 := by sorry

end checkerboard_inner_probability_l3561_356168


namespace derivative_y_wrt_x_l3561_356120

noncomputable def x (t : ℝ) : ℝ := Real.log (1 / Real.tan t)

noncomputable def y (t : ℝ) : ℝ := 1 / (Real.cos t)^2

theorem derivative_y_wrt_x (t : ℝ) (h : Real.cos t ≠ 0) (h' : Real.sin t ≠ 0) :
  deriv y t / deriv x t = -2 * (Real.tan t)^2 :=
sorry

end derivative_y_wrt_x_l3561_356120


namespace cone_volume_from_lateral_surface_l3561_356185

/-- Given a cone whose lateral surface, when unfolded, forms a semicircle with area 8π,
    the volume of the cone is (8√3π)/3 -/
theorem cone_volume_from_lateral_surface (r h : ℝ) : 
  r > 0 → h > 0 → 
  (π * r * (r^2 + h^2).sqrt / 2 = 8 * π) → 
  (1/3 * π * r^2 * h = 8 * Real.sqrt 3 * π / 3) := by
  sorry

end cone_volume_from_lateral_surface_l3561_356185


namespace product_remainder_l3561_356144

def product : ℕ := 3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93

theorem product_remainder (n : ℕ) (h : n = product) : n % 7 = 4 := by
  sorry

end product_remainder_l3561_356144


namespace m_equals_one_iff_z_purely_imaginary_l3561_356165

-- Define a complex number
def z (m : ℝ) : ℂ := m^2 * (1 + Complex.I) + m * (Complex.I - 1)

-- Define what it means for a complex number to be purely imaginary
def isPurelyImaginary (c : ℂ) : Prop := c.re = 0 ∧ c.im ≠ 0

-- State the theorem
theorem m_equals_one_iff_z_purely_imaginary :
  ∀ m : ℝ, m = 1 ↔ isPurelyImaginary (z m) := by sorry

end m_equals_one_iff_z_purely_imaginary_l3561_356165


namespace jordan_oreos_l3561_356190

theorem jordan_oreos (total : ℕ) (h1 : total = 36) : ∃ (jordan : ℕ), 
  jordan + (2 * jordan + 3) = total ∧ jordan = 11 := by
  sorry

end jordan_oreos_l3561_356190


namespace set_M_properties_l3561_356113

-- Define the set M
variable (M : Set ℝ)

-- Define the properties of M
variable (h_nonempty : M.Nonempty)
variable (h_two : 2 ∈ M)
variable (h_diff : ∀ x y, x ∈ M → y ∈ M → x - y ∈ M)

-- Theorem statement
theorem set_M_properties :
  (0 ∈ M) ∧
  (∀ x y, x ∈ M → y ∈ M → x + y ∈ M) ∧
  (∀ x, x ∈ M → x ≠ 0 → x ≠ 1 → (1 / (x * (x - 1))) ∈ M) :=
sorry

end set_M_properties_l3561_356113


namespace caleb_ice_cream_purchase_l3561_356126

/-- The number of cartons of ice cream Caleb bought -/
def ice_cream_cartons : ℕ := sorry

/-- The number of cartons of frozen yoghurt Caleb bought -/
def frozen_yoghurt_cartons : ℕ := 4

/-- The cost of one carton of ice cream in dollars -/
def ice_cream_cost : ℕ := 4

/-- The cost of one carton of frozen yoghurt in dollars -/
def frozen_yoghurt_cost : ℕ := 1

/-- The difference in dollars between the total cost of ice cream and frozen yoghurt -/
def cost_difference : ℕ := 36

theorem caleb_ice_cream_purchase : 
  ice_cream_cartons = 10 ∧
  ice_cream_cartons * ice_cream_cost = 
    frozen_yoghurt_cartons * frozen_yoghurt_cost + cost_difference := by
  sorry

end caleb_ice_cream_purchase_l3561_356126


namespace set_operations_l3561_356158

def U : Set Int := {x | |x| < 3}
def A : Set Int := {0, 1, 2}
def B : Set Int := {1, 2}

theorem set_operations :
  (A ∪ B = {0, 1, 2}) ∧
  ((U \ A) ∩ (U \ B) = {-2, -1}) := by
  sorry

end set_operations_l3561_356158


namespace grid_exists_l3561_356151

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_primes (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ ∀ r, is_prime r → r ≤ p ∨ r ≥ q

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

theorem grid_exists : ∃ (a b c d e f g h : ℕ),
  (∀ x ∈ [a, b, c, d, e, f, g, h], x > 0 ∧ x < 10) ∧
  a ≠ 0 ∧ e ≠ 0 ∧
  (∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ 1000 * a + 100 * b + 10 * c + d = p^q) ∧
  (∃ (p q : ℕ), consecutive_primes p q ∧ 1000 * e + 100 * f + 10 * c + d = p * q) ∧
  is_perfect_square (1000 * e + 100 * e + 10 * g + g) ∧
  is_multiple_of (1000 * e + 100 * h + 10 * g + g) 37 ∧
  1000 * a + 100 * b + 10 * c + d = 2187 ∧
  1000 * e + 100 * f + 10 * c + d = 7387 ∧
  1000 * e + 100 * e + 10 * g + g = 7744 ∧
  1000 * e + 100 * h + 10 * g + g = 7744 :=
by
  sorry


end grid_exists_l3561_356151


namespace matrix_power_500_l3561_356135

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; -1, 1]

theorem matrix_power_500 : A ^ 500 = !![1, 0; -500, 1] := by sorry

end matrix_power_500_l3561_356135


namespace intersection_when_a_is_3_empty_intersection_iff_a_in_range_l3561_356180

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | x ≤ 1 ∨ 4 ≤ x}

-- Theorem 1
theorem intersection_when_a_is_3 :
  A 3 ∩ B = {x | -1 ≤ x ∧ x ≤ 1 ∨ 4 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem empty_intersection_iff_a_in_range (a : ℝ) :
  (a > 0) → (A a ∩ B = ∅ ↔ 0 < a ∧ a < 1) := by sorry

end intersection_when_a_is_3_empty_intersection_iff_a_in_range_l3561_356180


namespace boys_on_playground_l3561_356142

/-- The number of girls on the playground -/
def num_girls : ℕ := 28

/-- The difference between the number of boys and girls -/
def difference : ℕ := 7

/-- The number of boys on the playground -/
def num_boys : ℕ := num_girls + difference

theorem boys_on_playground : num_boys = 35 := by
  sorry

end boys_on_playground_l3561_356142


namespace polar_to_rectangular_conversion_l3561_356114

theorem polar_to_rectangular_conversion :
  let r : ℝ := 4 * Real.sqrt 2
  let θ : ℝ := π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 2 * Real.sqrt 2) ∧ (y = 2 * Real.sqrt 6) := by
sorry

end polar_to_rectangular_conversion_l3561_356114


namespace polynomial_shift_representation_l3561_356177

theorem polynomial_shift_representation (f : Polynomial ℝ) (x₀ : ℝ) :
  ∃! g : Polynomial ℝ, ∀ x, f.eval x = g.eval (x - x₀) := by
  sorry

end polynomial_shift_representation_l3561_356177


namespace largest_positive_integer_binary_op_l3561_356191

def binary_op (n : ℤ) : ℤ := n - (n * 5)

theorem largest_positive_integer_binary_op :
  ∀ m : ℕ+, m > 4 → binary_op m ≥ 18 ∧ binary_op 4 < 18 :=
by sorry

end largest_positive_integer_binary_op_l3561_356191


namespace number_calculation_l3561_356181

theorem number_calculation (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 25 → (40/100 : ℝ) * N = 300 := by
  sorry

end number_calculation_l3561_356181


namespace equation_solutions_l3561_356106

theorem equation_solutions : 
  (∃ (S₁ S₂ : Set ℝ), 
    S₁ = {x : ℝ | x * (x + 4) = -5 * (x + 4)} ∧ 
    S₂ = {x : ℝ | (x + 2)^2 = (2*x - 1)^2} ∧
    S₁ = {-4, -5} ∧
    S₂ = {3, -1/3}) := by
  sorry

end equation_solutions_l3561_356106


namespace rahul_to_deepak_age_ratio_l3561_356160

def rahul_age_after_6_years : ℕ := 26
def deepak_current_age : ℕ := 15
def years_to_add : ℕ := 6

theorem rahul_to_deepak_age_ratio :
  (rahul_age_after_6_years - years_to_add) / deepak_current_age = 4 / 3 := by
  sorry

end rahul_to_deepak_age_ratio_l3561_356160


namespace unique_identical_lines_l3561_356159

theorem unique_identical_lines : 
  ∃! (a d : ℝ), ∀ (x y : ℝ), (2 * x + a * y + 4 = 0 ↔ d * x - 3 * y + 9 = 0) := by
  sorry

end unique_identical_lines_l3561_356159


namespace alex_zhu_same_section_probability_l3561_356155

def total_students : ℕ := 100
def selected_students : ℕ := 60
def num_sections : ℕ := 3
def students_per_section : ℕ := 20

theorem alex_zhu_same_section_probability :
  (3 : ℚ) * (Nat.choose 58 18) / (Nat.choose 60 20) = 19 / 165 := by
  sorry

end alex_zhu_same_section_probability_l3561_356155


namespace fruit_lovers_count_l3561_356195

/-- The number of people who like apple -/
def apple_lovers : ℕ := 40

/-- The number of people who like orange and mango but dislike apple -/
def orange_mango_lovers : ℕ := 7

/-- The number of people who like mango and apple and dislike orange -/
def apple_mango_lovers : ℕ := 10

/-- The total number of people who like apple -/
def total_apple_lovers : ℕ := 47

/-- The number of people who like all three fruits (apple, orange, and mango) -/
def all_fruit_lovers : ℕ := 3

theorem fruit_lovers_count : 
  apple_lovers + (apple_mango_lovers - all_fruit_lovers) + all_fruit_lovers = total_apple_lovers :=
by sorry

end fruit_lovers_count_l3561_356195


namespace triangle_abc_properties_l3561_356127

theorem triangle_abc_properties (A B C : ℝ) (AB : ℝ) :
  2 * Real.sin (2 * C) * Real.cos C - Real.sin (3 * C) = Real.sqrt 3 * (1 - Real.cos C) →
  AB = 2 →
  Real.sin C + Real.sin (B - A) = 2 * Real.sin (2 * A) →
  C = π / 3 ∧ (1 / 2) * AB * Real.sin C * Real.sqrt ((4 - AB^2) / (4 * Real.sin C^2)) = (2 * Real.sqrt 3) / 3 := by
  sorry


end triangle_abc_properties_l3561_356127


namespace sheet_length_is_48_l3561_356136

/-- Represents the dimensions of a rectangular sheet and the resulting box after cutting squares from corners. -/
structure SheetDimensions where
  length : ℝ
  width : ℝ
  cutSize : ℝ
  boxVolume : ℝ

/-- Theorem stating that given specific dimensions and volume, the length of the sheet must be 48 meters. -/
theorem sheet_length_is_48 (d : SheetDimensions)
  (h1 : d.width = 36)
  (h2 : d.cutSize = 4)
  (h3 : d.boxVolume = 4480)
  (h4 : d.boxVolume = (d.length - 2 * d.cutSize) * (d.width - 2 * d.cutSize) * d.cutSize) :
  d.length = 48 := by
  sorry


end sheet_length_is_48_l3561_356136


namespace polynomial_divisibility_l3561_356133

theorem polynomial_divisibility (p q : ℚ) : 
  (∀ x : ℚ, (x + 1) * (x - 2) ∣ (x^5 - x^4 + 2*x^3 - p*x^2 + q*x - 5)) → 
  p = 3/2 ∧ q = -21/2 := by
  sorry

end polynomial_divisibility_l3561_356133


namespace range_of_c_l3561_356153

theorem range_of_c (a c : ℝ) : 
  (∀ x > 0, 2*x + a/x ≥ c) → 
  (a ≥ 1/8 → ∀ x > 0, 2*x + a/x ≥ c) → 
  (∃ a < 1/8, ∀ x > 0, 2*x + a/x ≥ c) → 
  c ≤ 1 := by sorry

end range_of_c_l3561_356153


namespace vector_magnitude_proof_l3561_356169

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![3, -2]

theorem vector_magnitude_proof :
  Real.sqrt ((2 * a 0 - b 0)^2 + (2 * a 1 - b 1)^2) = Real.sqrt 17 := by
  sorry

end vector_magnitude_proof_l3561_356169


namespace store_profit_percentage_l3561_356111

/-- Given a store that purchases an item and the conditions of a potential price change,
    this theorem proves that the original profit percentage was 35%. -/
theorem store_profit_percentage
  (original_cost : ℝ)
  (cost_decrease_percentage : ℝ)
  (profit_increase_percentage : ℝ)
  (h1 : original_cost = 200)
  (h2 : cost_decrease_percentage = 10)
  (h3 : profit_increase_percentage = 15)
  (h4 : ∃ (sale_price : ℝ) (original_profit_percentage : ℝ),
        sale_price = original_cost * (1 + original_profit_percentage / 100) ∧
        sale_price = (original_cost * (1 - cost_decrease_percentage / 100)) *
                     (1 + (original_profit_percentage + profit_increase_percentage) / 100)) :
  ∃ (original_profit_percentage : ℝ), original_profit_percentage = 35 :=
sorry

end store_profit_percentage_l3561_356111


namespace min_height_box_l3561_356156

/-- Represents a rectangular box with a square base -/
structure Box where
  base : ℝ
  height : ℝ

/-- Calculates the surface area of a box -/
def surfaceArea (b : Box) : ℝ :=
  2 * b.base^2 + 4 * b.base * b.height

/-- Theorem stating the minimum height of the box under given conditions -/
theorem min_height_box :
  ∀ (b : Box),
    b.height = b.base + 4 →
    surfaceArea b ≥ 150 →
    ∀ (b' : Box),
      b'.height = b'.base + 4 →
      surfaceArea b' ≥ 150 →
      b.height ≤ b'.height →
      b.height = 9 :=
by sorry

end min_height_box_l3561_356156


namespace sum_of_even_coefficients_l3561_356129

def polynomial_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) : ℕ → ℤ
  | 0 => a₀
  | 1 => a₁
  | 2 => a₂
  | 3 => a₃
  | 4 => a₄
  | 5 => a₅
  | 6 => a₆
  | 7 => a₇
  | _ => 0

theorem sum_of_even_coefficients 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) :
  (∀ (x : ℤ), (3*x - 1)^7 = 
    a₀*x^7 + a₁*x^6 + a₂*x^5 + a₃*x^4 + a₄*x^3 + a₅*x^2 + a₆*x + a₇) →
  a₀ + a₂ + a₄ + a₆ = 4128 :=
by sorry

end sum_of_even_coefficients_l3561_356129


namespace arithmetic_sequence_2010th_term_l3561_356138

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  p : ℚ
  q : ℚ
  first_term : ℚ := p
  second_term : ℚ := 7
  third_term : ℚ := 3*p - q
  fourth_term : ℚ := 5*p + q
  is_arithmetic : ∃ d : ℚ, second_term = first_term + d ∧ 
                           third_term = second_term + d ∧ 
                           fourth_term = third_term + d

/-- The 2010th term of the arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.first_term + (n - 1) * ((seq.fourth_term - seq.first_term) / 3)

/-- Theorem stating that the 2010th term is 6253 -/
theorem arithmetic_sequence_2010th_term (seq : ArithmeticSequence) :
  nth_term seq 2010 = 6253 := by
  sorry

end arithmetic_sequence_2010th_term_l3561_356138


namespace ludwig_weekly_earnings_l3561_356121

/-- Calculates Ludwig's total earnings for a week --/
def ludwig_earnings (weekday_rate : ℚ) (weekend_rate : ℚ) (total_hours : ℕ) : ℚ :=
  let weekday_earnings := 4 * weekday_rate
  let weekend_earnings := 3 * weekend_rate / 2
  let regular_earnings := weekday_earnings + weekend_earnings
  let overtime_hours := max (total_hours - 48) 0
  let overtime_rate := weekend_rate * 3 / 8
  let overtime_earnings := overtime_hours * overtime_rate * 3 / 2
  regular_earnings + overtime_earnings

/-- Theorem stating Ludwig's earnings for the given week --/
theorem ludwig_weekly_earnings :
  ludwig_earnings 12 15 52 = 115.5 := by
  sorry


end ludwig_weekly_earnings_l3561_356121


namespace trajectory_is_line_segment_l3561_356146

/-- Given two fixed points in a metric space, the set of points whose sum of distances to these fixed points equals the distance between the fixed points is equal to the set containing only the fixed points. -/
theorem trajectory_is_line_segment {α : Type*} [MetricSpace α] (F₁ F₂ : α) (h : dist F₁ F₂ = 8) :
  {M : α | dist M F₁ + dist M F₂ = 8} = {F₁, F₂} := by sorry

end trajectory_is_line_segment_l3561_356146


namespace penetrated_cubes_count_stating_penetrated_cubes_calculation_correct_l3561_356102

/-- 
Given a rectangular solid with dimensions 120 × 260 × 300,
the number of unit cubes penetrated by an internal diagonal is 520.
-/
theorem penetrated_cubes_count : ℕ → ℕ → ℕ → ℕ
  | 120, 260, 300 => 520
  | _, _, _ => 0

/-- Function to calculate the number of penetrated cubes -/
def calculate_penetrated_cubes (a b c : ℕ) : ℕ :=
  a + b + c - (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) + Nat.gcd a (Nat.gcd b c)

/-- 
Theorem stating that the calculate_penetrated_cubes function 
correctly calculates the number of penetrated cubes for the given dimensions
-/
theorem penetrated_cubes_calculation_correct :
  calculate_penetrated_cubes 120 260 300 = penetrated_cubes_count 120 260 300 := by
  sorry

#eval calculate_penetrated_cubes 120 260 300

end penetrated_cubes_count_stating_penetrated_cubes_calculation_correct_l3561_356102


namespace max_discount_rate_l3561_356128

theorem max_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) : 
  cost_price = 4 →
  selling_price = 5 →
  min_profit_margin = 0.1 →
  ∃ (max_discount : ℝ),
    max_discount = 12 ∧
    ∀ (discount : ℝ),
      0 ≤ discount →
      discount ≤ max_discount →
      selling_price * (1 - discount / 100) - cost_price ≥ min_profit_margin * cost_price ∧
      ∀ (other_discount : ℝ),
        other_discount > max_discount →
        selling_price * (1 - other_discount / 100) - cost_price < min_profit_margin * cost_price :=
by sorry

end max_discount_rate_l3561_356128


namespace quadratic_equation_roots_specific_root_condition_l3561_356172

theorem quadratic_equation_roots (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + 4*x + m - 1
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) → m ≤ 5 :=
by sorry

theorem specific_root_condition (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + 4*x + m - 1
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 2*(x₁ + x₂) + x₁*x₂ + 10 = 0) → m = -1 :=
by sorry

end quadratic_equation_roots_specific_root_condition_l3561_356172


namespace garden_potato_yield_l3561_356194

def garden_length_steps : ℕ := 18
def garden_width_steps : ℕ := 25
def feet_per_step : ℕ := 3
def potato_yield_per_sqft : ℚ := 3/4

def garden_length_feet : ℕ := garden_length_steps * feet_per_step
def garden_width_feet : ℕ := garden_width_steps * feet_per_step
def garden_area_sqft : ℕ := garden_length_feet * garden_width_feet

def expected_potato_yield : ℚ := garden_area_sqft * potato_yield_per_sqft

theorem garden_potato_yield :
  expected_potato_yield = 3037.5 := by sorry

end garden_potato_yield_l3561_356194


namespace paths_from_C_to_D_l3561_356130

/-- The number of paths on a grid from (0,0) to (m,n) where only right and up moves are allowed -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- The dimensions of the grid -/
def gridWidth : ℕ := 7
def gridHeight : ℕ := 9

/-- The theorem stating the number of paths from C to D -/
theorem paths_from_C_to_D : gridPaths gridWidth gridHeight = 11440 := by
  sorry

end paths_from_C_to_D_l3561_356130


namespace trapezium_other_side_length_l3561_356179

-- Define the trapezium properties
def trapezium_side1 : ℝ := 20
def trapezium_height : ℝ := 15
def trapezium_area : ℝ := 285

-- Define the theorem
theorem trapezium_other_side_length :
  ∃ (side2 : ℝ), 
    (1/2 : ℝ) * (trapezium_side1 + side2) * trapezium_height = trapezium_area ∧
    side2 = 18 := by
  sorry

end trapezium_other_side_length_l3561_356179


namespace divisibility_theorem_l3561_356147

theorem divisibility_theorem (k n : ℕ) (hk : k > 0) (hn : n > 0) :
  ∃ m : ℤ, (n^4 - 1) * (n^3 - n^2 + n - 1)^k + (n + 1) * n^(4*k - 1) = m * (n^5 + 1) := by
  sorry

end divisibility_theorem_l3561_356147
