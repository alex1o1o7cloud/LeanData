import Mathlib

namespace hexagon_sequence_theorem_l1802_180245

/-- Represents the number of dots in the nth hexagon of the sequence -/
def hexagon_dots (n : ℕ) : ℕ :=
  if n = 0 then 0
  else 1 + 3 * n * (n - 1)

/-- The theorem stating the number of dots in the first four hexagons -/
theorem hexagon_sequence_theorem :
  hexagon_dots 1 = 1 ∧
  hexagon_dots 2 = 7 ∧
  hexagon_dots 3 = 19 ∧
  hexagon_dots 4 = 37 := by
  sorry

end hexagon_sequence_theorem_l1802_180245


namespace product_pqr_equals_864_l1802_180260

theorem product_pqr_equals_864 (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 36)
  (h3 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 540 / (p * q * r) = 1) :
  p * q * r = 864 := by
  sorry

end product_pqr_equals_864_l1802_180260


namespace least_seven_binary_digits_l1802_180200

/-- The number of binary digits required to represent a positive integer -/
def binary_digits (n : ℕ+) : ℕ :=
  (Nat.log 2 n.val).succ

/-- Predicate to check if a number has exactly 7 binary digits -/
def has_seven_binary_digits (n : ℕ+) : Prop :=
  binary_digits n = 7

theorem least_seven_binary_digits :
  ∃ (n : ℕ+), has_seven_binary_digits n ∧
    ∀ (m : ℕ+), has_seven_binary_digits m → n ≤ m ∧
    n = 64 := by sorry

end least_seven_binary_digits_l1802_180200


namespace men_work_hours_per_day_l1802_180285

-- Define the number of men, women, and days
def num_men : ℕ := 15
def num_women : ℕ := 21
def days_men : ℕ := 21
def days_women : ℕ := 20
def hours_women : ℕ := 9

-- Define the ratio of work done by women to men
def women_to_men_ratio : ℚ := 2 / 3

-- Define the function to calculate total work hours
def total_work_hours (num_workers : ℕ) (num_days : ℕ) (hours_per_day : ℕ) : ℕ :=
  num_workers * num_days * hours_per_day

-- Theorem statement
theorem men_work_hours_per_day :
  ∃ (hours_men : ℕ),
    (total_work_hours num_men days_men hours_men : ℚ) * women_to_men_ratio =
    (total_work_hours num_women days_women hours_women : ℚ) ∧
    hours_men = 8 := by
  sorry

end men_work_hours_per_day_l1802_180285


namespace michael_twice_jacob_age_l1802_180240

theorem michael_twice_jacob_age (jacob_current_age : ℕ) (michael_current_age : ℕ) : 
  jacob_current_age = 11 - 4 →
  michael_current_age = jacob_current_age + 12 →
  ∃ x : ℕ, michael_current_age + x = 2 * (jacob_current_age + x) ∧ x = 5 :=
by sorry

end michael_twice_jacob_age_l1802_180240


namespace triangle_area_l1802_180239

/-- Given a triangle with perimeter 48 cm and inradius 2.5 cm, its area is 60 cm² -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 48 → inradius = 2.5 → area = perimeter / 2 * inradius → area = 60 := by
sorry

end triangle_area_l1802_180239


namespace goose_eggs_count_l1802_180218

theorem goose_eggs_count (total_eggs : ℕ) : 
  (2 : ℚ) / 3 * (3 : ℚ) / 4 * (2 : ℚ) / 5 * total_eggs = 180 →
  total_eggs = 2700 := by
sorry

end goose_eggs_count_l1802_180218


namespace grisha_has_winning_strategy_l1802_180287

/-- Represents the state of the game board -/
def GameBoard := List Nat

/-- Represents a player's move -/
inductive Move
| Square : Nat → Move  -- Square the number at a given index
| Increment : Nat → Move  -- Increment the number at a given index

/-- Represents a player -/
inductive Player
| Grisha
| Gleb

/-- Applies a move to the game board -/
def applyMove (board : GameBoard) (move : Move) : GameBoard :=
  match move with
  | Move.Square i => sorry
  | Move.Increment i => sorry

/-- Checks if any number on the board is divisible by 2023 -/
def hasDivisibleBy2023 (board : GameBoard) : Bool :=
  sorry

/-- Represents a game strategy -/
def Strategy := GameBoard → Move

/-- Checks if a strategy is winning for a player -/
def isWinningStrategy (player : Player) (strategy : Strategy) : Prop :=
  sorry

/-- The main theorem stating that Grisha has a winning strategy -/
theorem grisha_has_winning_strategy :
  ∃ (strategy : Strategy), isWinningStrategy Player.Grisha strategy :=
sorry

end grisha_has_winning_strategy_l1802_180287


namespace point_transformation_l1802_180279

def initial_point : ℝ × ℝ × ℝ := (2, 3, -1)

def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  reflect_yz (reflect_xz (rotate_z_90 p))

theorem point_transformation :
  transform initial_point = (3, -2, -1) := by
  sorry

end point_transformation_l1802_180279


namespace ellipse_parameter_inequality_l1802_180254

/-- An ellipse with equation ax^2 + by^2 = 1 and foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  is_ellipse : a > 0 ∧ b > 0
  foci_on_x_axis : a ≠ b

theorem ellipse_parameter_inequality (e : Ellipse) : 0 < e.a ∧ e.a < e.b := by
  sorry

end ellipse_parameter_inequality_l1802_180254


namespace farmers_market_total_sales_l1802_180205

/-- Calculates the total sales from a farmers' market given specific conditions --/
theorem farmers_market_total_sales :
  let broccoli_sales : ℕ := 57
  let carrot_sales : ℕ := 2 * broccoli_sales
  let spinach_sales : ℕ := carrot_sales / 2 + 16
  let cauliflower_sales : ℕ := 136
  broccoli_sales + carrot_sales + spinach_sales + cauliflower_sales = 380 :=
by
  sorry


end farmers_market_total_sales_l1802_180205


namespace complete_square_sum_l1802_180298

/-- 
Given a quadratic equation x^2 - 6x + 5 = 0, when rewritten in the form (x + d)^2 = e 
where d and e are integers, prove that d + e = 1
-/
theorem complete_square_sum (d e : ℤ) : 
  (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x + d)^2 = e) → d + e = 1 := by
  sorry

end complete_square_sum_l1802_180298


namespace complex_root_product_simplification_l1802_180203

theorem complex_root_product_simplification :
  2 * Real.sqrt 3 * 6 * (12 ^ (1/6 : ℝ)) * 3 * ((3/2) ^ (1/3 : ℝ)) = 6 := by
  sorry

end complex_root_product_simplification_l1802_180203


namespace remainder_2468135792_mod_101_l1802_180262

theorem remainder_2468135792_mod_101 : 2468135792 % 101 = 52 := by
  sorry

end remainder_2468135792_mod_101_l1802_180262


namespace equation_solution_l1802_180212

theorem equation_solution : ∃! (x : ℝ), x ≠ 0 ∧ (5*x)^20 = (20*x)^10 ∧ x = 4/5 := by
  sorry

end equation_solution_l1802_180212


namespace range_of_t_l1802_180217

theorem range_of_t (a b t : ℝ) (h1 : a^2 + a*b + b^2 = 1) (h2 : t = a*b - a^2 - b^2) :
  -3 ≤ t ∧ t ≤ -1/3 :=
by sorry

end range_of_t_l1802_180217


namespace expected_games_specific_scenario_l1802_180225

/-- Represents a table tennis game between two players -/
structure TableTennisGame where
  probAWins : ℝ
  aheadBy : ℕ

/-- Calculates the expected number of games in a table tennis match -/
def expectedGames (game : TableTennisGame) : ℝ :=
  sorry

/-- Theorem stating that the expected number of games in the specific scenario is 18/5 -/
theorem expected_games_specific_scenario :
  let game : TableTennisGame := ⟨2/3, 2⟩
  expectedGames game = 18/5 := by
  sorry

end expected_games_specific_scenario_l1802_180225


namespace min_sixth_graders_l1802_180229

theorem min_sixth_graders (x : ℕ) (hx : x > 0) : 
  let girls := x / 3
  let boys := x - girls
  let sixth_grade_girls := girls / 2
  let non_sixth_grade_boys := (boys * 5) / 7
  let sixth_grade_boys := boys - non_sixth_grade_boys
  let total_sixth_graders := sixth_grade_girls + sixth_grade_boys
  x % 3 = 0 ∧ girls % 2 = 0 ∧ boys % 7 = 0 →
  ∀ y : ℕ, y > 0 ∧ y < x ∧ 
    (let girls_y := y / 3
     let boys_y := y - girls_y
     let sixth_grade_girls_y := girls_y / 2
     let non_sixth_grade_boys_y := (boys_y * 5) / 7
     let sixth_grade_boys_y := boys_y - non_sixth_grade_boys_y
     let total_sixth_graders_y := sixth_grade_girls_y + sixth_grade_boys_y
     y % 3 = 0 ∧ girls_y % 2 = 0 ∧ boys_y % 7 = 0) →
    total_sixth_graders_y < total_sixth_graders →
  total_sixth_graders = 15 := by
sorry

end min_sixth_graders_l1802_180229


namespace green_knights_magical_fraction_l1802_180282

/-- Represents the fraction of knights of a certain color who are magical -/
structure MagicalFraction where
  numerator : ℕ
  denominator : ℕ
  denominator_pos : denominator > 0

/-- Represents the distribution of knights in the kingdom -/
structure KnightDistribution where
  green_fraction : Rat
  yellow_fraction : Rat
  magical_fraction : Rat
  green_magical : MagicalFraction
  yellow_magical : MagicalFraction
  green_fraction_valid : green_fraction = 3 / 8
  yellow_fraction_valid : yellow_fraction = 5 / 8
  fractions_sum_to_one : green_fraction + yellow_fraction = 1
  magical_fraction_valid : magical_fraction = 1 / 5
  green_thrice_yellow : green_magical.numerator * yellow_magical.denominator = 
                        3 * yellow_magical.numerator * green_magical.denominator

theorem green_knights_magical_fraction 
  (k : KnightDistribution) : k.green_magical = MagicalFraction.mk 12 35 (by norm_num) := by
  sorry

end green_knights_magical_fraction_l1802_180282


namespace combined_average_score_l1802_180216

theorem combined_average_score (score_u score_b score_c : ℝ)
  (ratio_u ratio_b ratio_c : ℕ) :
  score_u = 65 →
  score_b = 80 →
  score_c = 77 →
  ratio_u = 4 →
  ratio_b = 6 →
  ratio_c = 5 →
  (score_u * ratio_u + score_b * ratio_b + score_c * ratio_c) / (ratio_u + ratio_b + ratio_c) = 75 :=
by
  sorry

end combined_average_score_l1802_180216


namespace remainder_3_pow_20_mod_5_l1802_180232

theorem remainder_3_pow_20_mod_5 : 3^20 % 5 = 1 := by
  sorry

end remainder_3_pow_20_mod_5_l1802_180232


namespace root_transformation_l1802_180206

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 3*r₁^2 + 9 = 0) ∧ 
  (r₂^3 - 3*r₂^2 + 9 = 0) ∧ 
  (r₃^3 - 3*r₃^2 + 9 = 0) →
  ((3*r₁)^3 - 9*(3*r₁)^2 + 243 = 0) ∧
  ((3*r₂)^3 - 9*(3*r₂)^2 + 243 = 0) ∧
  ((3*r₃)^3 - 9*(3*r₃)^2 + 243 = 0) :=
by sorry

end root_transformation_l1802_180206


namespace equal_diagonal_polygon_is_quadrilateral_or_pentagon_l1802_180271

/-- A convex polygon with n sides and all diagonals equal -/
structure EqualDiagonalPolygon where
  n : ℕ
  sides : n ≥ 4
  convex : Bool
  all_diagonals_equal : Bool

/-- The set of quadrilaterals -/
def Quadrilaterals : Set EqualDiagonalPolygon :=
  {p : EqualDiagonalPolygon | p.n = 4}

/-- The set of pentagons -/
def Pentagons : Set EqualDiagonalPolygon :=
  {p : EqualDiagonalPolygon | p.n = 5}

theorem equal_diagonal_polygon_is_quadrilateral_or_pentagon 
  (F : EqualDiagonalPolygon) (h_convex : F.convex = true) 
  (h_diag : F.all_diagonals_equal = true) :
  F ∈ Quadrilaterals ∪ Pentagons :=
sorry

end equal_diagonal_polygon_is_quadrilateral_or_pentagon_l1802_180271


namespace modular_congruence_solution_l1802_180209

theorem modular_congruence_solution :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -5203 [ZMOD 13] := by
  sorry

end modular_congruence_solution_l1802_180209


namespace four_intersection_points_l1802_180213

/-- The polynomial function representing the curve -/
def f (c : ℝ) (x : ℝ) : ℝ := x^4 + 9*x^3 + c*x^2 + 9*x + 4

/-- Theorem stating the condition for the existence of a line intersecting the curve in four distinct points -/
theorem four_intersection_points (c : ℝ) :
  (∃ (m n : ℝ), ∀ (x : ℝ), (f c x = m*x + n) → (∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f c x₁ = m*x₁ + n ∧ f c x₂ = m*x₂ + n ∧ f c x₃ = m*x₃ + n ∧ f c x₄ = m*x₄ + n)) ↔
  c ≤ 243/8 :=
sorry

end four_intersection_points_l1802_180213


namespace suit_price_increase_l1802_180223

/-- Proves that the percentage increase in the price of a suit was 25% --/
theorem suit_price_increase (original_price : ℝ) (final_price : ℝ) : 
  original_price = 200 →
  final_price = 187.5 →
  ∃ (increase_percentage : ℝ),
    increase_percentage = 25 ∧
    final_price = (original_price + original_price * (increase_percentage / 100)) * 0.75 :=
by sorry

end suit_price_increase_l1802_180223


namespace sales_equation_solution_l1802_180231

/-- Given the sales equation and conditions, prove the value of p. -/
theorem sales_equation_solution (f w p : ℂ) (h1 : f * p - w = 15000) 
  (h2 : f = 10) (h3 : w = 10 + 250 * Complex.I) : p = 1501 + 25 * Complex.I := by
  sorry

end sales_equation_solution_l1802_180231


namespace rectangle_longer_side_l1802_180265

-- Define the circle radius
def circle_radius : ℝ := 6

-- Define the relationship between rectangle and circle areas
def area_ratio : ℝ := 3

-- Theorem statement
theorem rectangle_longer_side (circle_radius : ℝ) (area_ratio : ℝ) :
  circle_radius = 6 →
  area_ratio = 3 →
  let circle_area := π * circle_radius^2
  let rectangle_area := area_ratio * circle_area
  let shorter_side := 2 * circle_radius
  rectangle_area / shorter_side = 9 * π :=
by sorry

end rectangle_longer_side_l1802_180265


namespace melted_ice_cream_height_l1802_180247

/-- The height of a cylindrical region formed by a melted spherical ice cream scoop -/
theorem melted_ice_cream_height (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 3)
    (h_cylinder : r_cylinder = 9) : 
    (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_cylinder^2 * (4 / 9) := by
  sorry

#check melted_ice_cream_height

end melted_ice_cream_height_l1802_180247


namespace factorization_equality_l1802_180259

theorem factorization_equality (a b : ℝ) : 3 * a^2 * b - 12 * b = 3 * b * (a + 2) * (a - 2) := by
  sorry

end factorization_equality_l1802_180259


namespace special_number_value_l1802_180288

/-- Represents a positive integer with specific properties in different bases -/
def SpecialNumber (n : ℕ+) : Prop :=
  ∃ (X Y : ℕ),
    X < 8 ∧ Y < 9 ∧
    n = 8 * X + Y ∧
    n = 9 * Y + X

/-- The unique value of the special number in base 10 -/
theorem special_number_value :
  ∀ n : ℕ+, SpecialNumber n → n = 71 := by
  sorry

end special_number_value_l1802_180288


namespace geometric_sequence_sum_l1802_180215

theorem geometric_sequence_sum (a : ℝ) : 
  (a + 2*a + 4*a + 8*a = 1) →  -- Sum of first 4 terms equals 1
  (a + 2*a + 4*a + 8*a + 16*a + 32*a + 64*a + 128*a = 17) := by
sorry

end geometric_sequence_sum_l1802_180215


namespace min_value_of_f_for_shangmei_numbers_l1802_180220

/-- Definition of a Shangmei number -/
def isShangmeiNumber (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a + c = 11 ∧ b + d = 11

/-- Definition of function f -/
def f (n : ℕ) : ℚ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (b - d : ℚ) / (a - c)

/-- Definition of function G -/
def G (n : ℕ) : ℤ :=
  let ab := n / 100
  let cd := n % 100
  (ab : ℤ) - cd

/-- Main theorem -/
theorem min_value_of_f_for_shangmei_numbers :
  ∀ M : ℕ,
    isShangmeiNumber M →
    (M / 1000 < (M / 100) % 10) →
    (G M) % 7 = 0 →
    f M ≥ -3 ∧ ∃ M₀, isShangmeiNumber M₀ ∧ (M₀ / 1000 < (M₀ / 100) % 10) ∧ (G M₀) % 7 = 0 ∧ f M₀ = -3 :=
by sorry

end min_value_of_f_for_shangmei_numbers_l1802_180220


namespace ship_passengers_l1802_180290

theorem ship_passengers : ∀ (P : ℕ),
  (P / 12 : ℚ) + (P / 8 : ℚ) + (P / 3 : ℚ) + (P / 6 : ℚ) + 35 = P →
  P = 120 := by
  sorry

end ship_passengers_l1802_180290


namespace problem_solution_l1802_180238

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x + 1) / (2^(x+1) - a)

theorem problem_solution :
  ∃ (a : ℝ),
    (∀ (x : ℝ), f a x = (2^x + 1) / (2^(x+1) - a)) ∧
    (a = 2) ∧
    (∀ (x y : ℝ), 0 < x → 0 < y → x < y → f a x > f a y) ∧
    (∀ (k : ℝ), (∃ (x : ℝ), 0 < x ∧ x ≤ 1 ∧ k * f a x = 2) → 0 < k ∧ k ≤ 4/3) :=
by sorry

end problem_solution_l1802_180238


namespace sqrt_288_simplification_l1802_180204

theorem sqrt_288_simplification : Real.sqrt 288 = 12 * Real.sqrt 2 := by
  sorry

end sqrt_288_simplification_l1802_180204


namespace prob_one_defective_in_two_l1802_180263

/-- Given a set of 4 items with 3 genuine and 1 defective, the probability
of selecting exactly one defective item when randomly choosing 2 items is 1/2. -/
theorem prob_one_defective_in_two (n : ℕ) (k : ℕ) (d : ℕ) :
  n = 4 →
  k = 2 →
  d = 1 →
  (n.choose k) = 6 →
  (d * (n - d).choose (k - 1)) = 3 →
  (d * (n - d).choose (k - 1)) / (n.choose k) = 1 / 2 :=
by sorry

end prob_one_defective_in_two_l1802_180263


namespace cubic_greater_than_quadratic_l1802_180243

theorem cubic_greater_than_quadratic (x : ℝ) (h : x > 1) : x^3 > x^2 - x + 1 := by
  sorry

end cubic_greater_than_quadratic_l1802_180243


namespace gecko_hatched_eggs_l1802_180234

/-- Theorem: Number of hatched eggs for a gecko --/
theorem gecko_hatched_eggs (total_eggs : ℕ) (infertile_rate : ℚ) (calcification_rate : ℚ)
  (h_total : total_eggs = 30)
  (h_infertile : infertile_rate = 1/5)
  (h_calcification : calcification_rate = 1/3) :
  (total_eggs : ℚ) * (1 - infertile_rate) * (1 - calcification_rate) = 16 := by
  sorry

end gecko_hatched_eggs_l1802_180234


namespace polynomial_factor_implies_d_value_l1802_180237

theorem polynomial_factor_implies_d_value (c d : ℤ) : 
  (∃ k : ℤ, (X^3 - 2*X^2 - X + 2) * (c*X + k) = c*X^4 + d*X^3 - 2*X^2 + 2) → 
  d = -1 :=
by sorry

end polynomial_factor_implies_d_value_l1802_180237


namespace sum_of_coefficients_after_shift_l1802_180272

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a quadratic function horizontally -/
def shift_quadratic (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a,
    b := 2 * f.a * shift + f.b,
    c := f.a * shift^2 + f.b * shift + f.c }

/-- The original quadratic function y = 3x^2 - 2x + 6 -/
def original_function : QuadraticFunction :=
  { a := 3, b := -2, c := 6 }

/-- The amount of left shift -/
def left_shift : ℝ := 5

theorem sum_of_coefficients_after_shift :
  let shifted := shift_quadratic original_function left_shift
  shifted.a + shifted.b + shifted.c = 102 := by
  sorry

end sum_of_coefficients_after_shift_l1802_180272


namespace quadratic_function_min_value_l1802_180257

theorem quadratic_function_min_value 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x^2 + x + a) 
  (h2 : ∃ x ∈ Set.Icc (-1 : ℝ) 1, ∀ y ∈ Set.Icc (-1 : ℝ) 1, f y ≤ f x) 
  (h3 : ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = 2) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, ∀ y ∈ Set.Icc (-1 : ℝ) 1, f x ≤ f y ∧ f x = -1/4 := by
  sorry

end quadratic_function_min_value_l1802_180257


namespace inequality_solution_range_l1802_180249

-- Define the function f(x) = -x^2 + 2x + 1
def f (x : ℝ) : ℝ := -x^2 + 2*x + 1

-- Define the inequality
def has_solution (m : ℝ) : Prop :=
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ x^2 - 2*x - 1 + m ≤ 0

-- Theorem statement
theorem inequality_solution_range :
  ∀ m : ℝ, has_solution m ↔ m ≤ 2 :=
sorry

end inequality_solution_range_l1802_180249


namespace sandy_molly_age_difference_l1802_180256

theorem sandy_molly_age_difference :
  ∀ (sandy_age molly_age : ℕ),
    sandy_age = 70 →
    sandy_age * 9 = molly_age * 7 →
    molly_age - sandy_age = 20 :=
by
  sorry

end sandy_molly_age_difference_l1802_180256


namespace polar_to_cartesian_l1802_180281

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ = 8 * Real.cos θ - 6 * Real.sin θ

-- Define the Cartesian equation
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = 0

-- Theorem statement
theorem polar_to_cartesian :
  ∀ (x y ρ θ : ℝ),
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →  -- Conversion between polar and Cartesian coordinates
  polar_equation ρ θ ↔ cartesian_equation x y :=
by sorry

end polar_to_cartesian_l1802_180281


namespace gmat_test_probabilities_l1802_180291

theorem gmat_test_probabilities
  (p_first : ℝ)
  (p_second : ℝ)
  (p_both : ℝ)
  (h1 : p_first = 0.85)
  (h2 : p_second = 0.80)
  (h3 : p_both = 0.70)
  : 1 - (p_first + p_second - p_both) = 0.05 := by
  sorry

end gmat_test_probabilities_l1802_180291


namespace number_ratio_l1802_180226

theorem number_ratio (x : ℝ) (h : 3 * (2 * x + 9) = 69) : x / (2 * x) = 1 / 2 := by
  sorry

end number_ratio_l1802_180226


namespace sector_radius_l1802_180283

/-- Given a sector with arc length and area, calculate its radius -/
theorem sector_radius (arc_length : ℝ) (area : ℝ) (radius : ℝ) : 
  arc_length = 2 → area = 2 → (1/2) * arc_length * radius = area → radius = 2 := by
  sorry

#check sector_radius

end sector_radius_l1802_180283


namespace square_triangle_equal_area_l1802_180241

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) : 
  square_perimeter = 64 →
  triangle_height = 32 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_height * x →
  x = 16 := by
sorry

end square_triangle_equal_area_l1802_180241


namespace return_speed_calculation_l1802_180230

/-- Proves that given a round trip with specified conditions, the return speed is 160 km/h -/
theorem return_speed_calculation (total_time : ℝ) (outbound_time_minutes : ℝ) (outbound_speed : ℝ) :
  total_time = 5 →
  outbound_time_minutes = 192 →
  outbound_speed = 90 →
  let outbound_time_hours : ℝ := outbound_time_minutes / 60
  let distance : ℝ := outbound_speed * outbound_time_hours
  let return_time : ℝ := total_time - outbound_time_hours
  let return_speed : ℝ := distance / return_time
  return_speed = 160 := by
  sorry

#check return_speed_calculation

end return_speed_calculation_l1802_180230


namespace nap_time_calculation_l1802_180222

/-- Calculates the remaining time for a nap given flight duration and time spent on activities --/
def time_for_nap (flight_duration : ℕ) (reading : ℕ) (movies : ℕ) (dinner : ℕ) (radio : ℕ) (games : ℕ) : ℕ :=
  flight_duration - (reading + movies + dinner + radio + games)

/-- Converts hours and minutes to minutes --/
def to_minutes (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * 60 + minutes

theorem nap_time_calculation :
  let flight_duration := to_minutes 11 20
  let reading := to_minutes 2 0
  let movies := to_minutes 4 0
  let dinner := 30
  let radio := 40
  let games := to_minutes 1 10
  let nap_time := time_for_nap flight_duration reading movies dinner radio games
  nap_time = to_minutes 3 0 := by sorry

end nap_time_calculation_l1802_180222


namespace largest_number_with_conditions_l1802_180202

def is_valid_digit (d : Nat) : Prop := d = 1 ∨ d = 2 ∨ d = 3

def digits_sum_to_13 (n : Nat) : Prop :=
  (Nat.digits 10 n).sum = 13

def all_digits_valid (n : Nat) : Prop :=
  ∀ d ∈ Nat.digits 10 n, is_valid_digit d

def is_largest_with_conditions (n : Nat) : Prop :=
  digits_sum_to_13 n ∧ all_digits_valid n ∧
  ∀ m : Nat, digits_sum_to_13 m → all_digits_valid m → m ≤ n

theorem largest_number_with_conditions :
  is_largest_with_conditions 322222 :=
sorry

end largest_number_with_conditions_l1802_180202


namespace resistor_value_l1802_180289

/-- The resistance of a single resistor in a circuit where three identical resistors are initially 
    in series, and then connected in parallel, such that the change in total resistance is 10 Ω. -/
theorem resistor_value (R : ℝ) : 
  (3 * R - R / 3 = 10) → R = 3.75 := by
  sorry

end resistor_value_l1802_180289


namespace equal_roots_quadratic_l1802_180224

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y + k = 0 → y = x) → 
  k = 1 := by
sorry

end equal_roots_quadratic_l1802_180224


namespace second_number_proof_l1802_180228

theorem second_number_proof (x y z : ℚ) 
  (sum_eq : x + y + z = 125)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 7 / 6) :
  y = 3500 / 73 := by
sorry

end second_number_proof_l1802_180228


namespace millet_exceeds_half_on_thursday_l1802_180277

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : ℕ
  millet : ℚ
  other : ℚ

/-- Calculates the next day's feeder state -/
def nextDay (state : FeederState) : FeederState :=
  { day := state.day + 1,
    millet := state.millet / 2 + 2 / 5,
    other := 3 / 5 }

/-- Checks if millet proportion exceeds 50% -/
def milletExceedsHalf (state : FeederState) : Prop :=
  state.millet > (state.millet + state.other) / 2

/-- Initial state of the feeder on Monday -/
def initialState : FeederState :=
  { day := 1, millet := 2 / 5, other := 3 / 5 }

theorem millet_exceeds_half_on_thursday :
  let thursday := nextDay (nextDay (nextDay initialState))
  milletExceedsHalf thursday ∧
  ∀ (prevDay : FeederState), prevDay.day < thursday.day →
    ¬ milletExceedsHalf prevDay := by
  sorry

end millet_exceeds_half_on_thursday_l1802_180277


namespace two_cars_in_garage_l1802_180299

/-- Represents the number of wheels on various vehicles --/
structure VehicleWheels where
  lawnmower : Nat
  bicycle : Nat
  tricycle : Nat
  unicycle : Nat
  car : Nat

/-- Calculates the total number of wheels for non-car vehicles --/
def nonCarWheels (v : VehicleWheels) (numBicycles : Nat) : Nat :=
  v.lawnmower + numBicycles * v.bicycle + v.tricycle + v.unicycle

/-- Theorem stating that given the conditions in the problem, there are 2 cars in the garage --/
theorem two_cars_in_garage (totalWheels : Nat) (v : VehicleWheels) (numBicycles : Nat) :
  totalWheels = 22 →
  v.lawnmower = 4 →
  v.bicycle = 2 →
  v.tricycle = 3 →
  v.unicycle = 1 →
  v.car = 4 →
  numBicycles = 3 →
  (totalWheels - nonCarWheels v numBicycles) / v.car = 2 :=
by sorry

end two_cars_in_garage_l1802_180299


namespace tomato_plants_count_l1802_180208

def strawberry_plants : ℕ := 5
def strawberries_per_plant : ℕ := 14
def tomatoes_per_plant : ℕ := 16
def fruits_per_basket : ℕ := 7
def strawberry_basket_price : ℕ := 9
def tomato_basket_price : ℕ := 6
def total_revenue : ℕ := 186

theorem tomato_plants_count (tomato_plants : ℕ) : 
  strawberry_plants * strawberries_per_plant / fruits_per_basket * strawberry_basket_price + 
  tomato_plants * tomatoes_per_plant / fruits_per_basket * tomato_basket_price = total_revenue → 
  tomato_plants = 7 := by
  sorry

end tomato_plants_count_l1802_180208


namespace problem_solution_l1802_180273

theorem problem_solution (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x / 3 = y ^ 2) 
  (h3 : x / 6 = 3 * y) : 
  x = 108 := by
  sorry

end problem_solution_l1802_180273


namespace adult_tickets_sold_l1802_180274

theorem adult_tickets_sold (children_tickets : ℕ) (children_price : ℕ) (adult_price : ℕ) (total_earnings : ℕ) : 
  children_tickets = 210 →
  children_price = 25 →
  adult_price = 50 →
  total_earnings = 5950 →
  (total_earnings - children_tickets * children_price) / adult_price = 14 :=
by sorry

end adult_tickets_sold_l1802_180274


namespace irrational_sqrt_6_l1802_180275

-- Define rationality
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define irrationality
def IsIrrational (x : ℝ) : Prop := ¬ IsRational x

-- Theorem statement
theorem irrational_sqrt_6 :
  IsIrrational (Real.sqrt 6) ∧ 
  IsRational 3.14 ∧
  IsRational (-1/3) ∧
  IsRational (22/7) :=
sorry

end irrational_sqrt_6_l1802_180275


namespace specific_pentagon_area_l1802_180236

/-- A pentagon with specific side lengths that can be divided into a right triangle and a trapezoid -/
structure SpecificPentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  triangle_base : ℝ
  triangle_height : ℝ
  trapezoid_base1 : ℝ
  trapezoid_base2 : ℝ
  trapezoid_height : ℝ
  side1_eq : side1 = 15
  side2_eq : side2 = 20
  side3_eq : side3 = 27
  side4_eq : side4 = 24
  side5_eq : side5 = 20
  triangle_base_eq : triangle_base = 15
  triangle_height_eq : triangle_height = 20
  trapezoid_base1_eq : trapezoid_base1 = 20
  trapezoid_base2_eq : trapezoid_base2 = 27
  trapezoid_height_eq : trapezoid_height = 24

/-- The area of the specific pentagon is 714 square units -/
theorem specific_pentagon_area (p : SpecificPentagon) : 
  (1/2 * p.triangle_base * p.triangle_height) + 
  (1/2 * (p.trapezoid_base1 + p.trapezoid_base2) * p.trapezoid_height) = 714 := by
  sorry

end specific_pentagon_area_l1802_180236


namespace octal_perfect_square_b_is_one_l1802_180286

/-- Represents a digit in base 8 -/
def OctalDigit := { n : Nat // n < 8 }

/-- Converts a number from base 8 to decimal -/
def octalToDecimal (a b c : OctalDigit) : Nat :=
  512 * a.val + 192 + 8 * b.val + c.val

/-- Represents a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, n = m * m

theorem octal_perfect_square_b_is_one
  (a : OctalDigit)
  (h_a : a.val ≠ 0)
  (b : OctalDigit)
  (c : OctalDigit) :
  isPerfectSquare (octalToDecimal a b c) → b.val = 1 := by
  sorry

end octal_perfect_square_b_is_one_l1802_180286


namespace seed_germination_percentage_l1802_180235

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) : 
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate1 = 1/5 →
  germination_rate2 = 7/20 →
  (((seeds_plot1 : ℚ) * germination_rate1 + (seeds_plot2 : ℚ) * germination_rate2) / 
   ((seeds_plot1 : ℚ) + (seeds_plot2 : ℚ))) = 13/50 := by
  sorry

end seed_germination_percentage_l1802_180235


namespace power_simplification_l1802_180295

theorem power_simplification : 
  ((12^15 / 12^7)^3 * 8^3) / 2^9 = 12^24 := by sorry

end power_simplification_l1802_180295


namespace find_largest_base_l1802_180210

theorem find_largest_base (x : ℤ) (base : ℕ) :
  (x ≤ 3) →
  (2.134 * (base : ℝ) ^ (x : ℝ) < 21000) →
  (∀ y : ℤ, y ≤ 3 → 2.134 * (base : ℝ) ^ (y : ℝ) < 21000) →
  base ≤ 21 :=
sorry

end find_largest_base_l1802_180210


namespace bird_percentage_l1802_180248

/-- The percentage of birds that are not hawks, paddyfield-warblers, kingfishers, or blackbirds in Goshawk-Eurasian Nature Reserve -/
theorem bird_percentage (total : ℝ) (hawks paddyfield_warblers kingfishers blackbirds : ℝ)
  (h1 : hawks = 0.3 * total)
  (h2 : paddyfield_warblers = 0.4 * (total - hawks))
  (h3 : kingfishers = 0.25 * paddyfield_warblers)
  (h4 : blackbirds = 0.15 * (hawks + paddyfield_warblers))
  (h5 : total > 0) :
  (total - (hawks + paddyfield_warblers + kingfishers + blackbirds)) / total = 0.26 := by
  sorry

end bird_percentage_l1802_180248


namespace abs_diff_properties_l1802_180242

-- Define the binary operation ⊕
def abs_diff (x y : ℝ) : ℝ := |x - y|

-- Main theorem
theorem abs_diff_properties :
  -- 1. ⊕ is commutative
  (∀ x y : ℝ, abs_diff x y = abs_diff y x) ∧
  -- 2. Addition distributes over ⊕
  (∀ a b c : ℝ, a + abs_diff b c = abs_diff (a + b) (a + c)) ∧
  -- 3. ⊕ is not associative
  (∃ x y z : ℝ, abs_diff x (abs_diff y z) ≠ abs_diff (abs_diff x y) z) ∧
  -- 4. ⊕ does not have an identity element
  (∀ e : ℝ, ∃ x : ℝ, abs_diff x e ≠ x) ∧
  -- 5. ⊕ does not distribute over addition
  (∃ x y z : ℝ, abs_diff x (y + z) ≠ abs_diff x y + abs_diff x z) :=
by sorry

end abs_diff_properties_l1802_180242


namespace three_numbers_problem_l1802_180292

theorem three_numbers_problem (x y z : ℝ) : 
  x = 0.8 * y ∧ 
  y / z = 0.5 / (9/20) ∧ 
  x + z = y + 70 →
  x = 80 ∧ y = 100 ∧ z = 90 := by
sorry

end three_numbers_problem_l1802_180292


namespace volume_cylinder_from_square_rotation_l1802_180269

/-- The volume of a cylinder formed by rotating a square around one of its sides. -/
theorem volume_cylinder_from_square_rotation (side_length : Real) (volume : Real) : 
  side_length = 2 → volume = 8 * Real.pi := by
  sorry

end volume_cylinder_from_square_rotation_l1802_180269


namespace age_of_B_l1802_180264

/-- Given the ages of four people A, B, C, and D, prove that B's age is 27 years. -/
theorem age_of_B (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 28 →
  (a + c) / 2 = 29 →
  (2 * b + 3 * d) / 5 = 27 →
  b = 27 := by
  sorry

end age_of_B_l1802_180264


namespace sum_of_squares_of_roots_l1802_180270

theorem sum_of_squares_of_roots (r₁ r₂ r₃ r₄ : ℂ) : 
  (r₁^4 + 6*r₁^3 + 11*r₁^2 + 6*r₁ + 1 = 0) →
  (r₂^4 + 6*r₂^3 + 11*r₂^2 + 6*r₂ + 1 = 0) →
  (r₃^4 + 6*r₃^3 + 11*r₃^2 + 6*r₃ + 1 = 0) →
  (r₄^4 + 6*r₄^3 + 11*r₄^2 + 6*r₄ + 1 = 0) →
  r₁^2 + r₂^2 + r₃^2 + r₄^2 = 14 :=
by sorry

end sum_of_squares_of_roots_l1802_180270


namespace abc_product_l1802_180280

theorem abc_product (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_sum : a + b + c = 30) 
  (h_eq : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (420 : ℚ) / (a * b * c) = 1) : 
  a * b * c = 450 := by
  sorry

end abc_product_l1802_180280


namespace total_interest_calculation_l1802_180201

/-- Calculate simple interest -/
def simple_interest (principal : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  principal * rate * time / 100

theorem total_interest_calculation (rate : ℕ) : 
  rate = 10 → 
  simple_interest 5000 rate 2 + simple_interest 3000 rate 4 = 2200 := by
  sorry

#check total_interest_calculation

end total_interest_calculation_l1802_180201


namespace coupon1_best_for_given_prices_coupon1_not_best_for_lower_prices_l1802_180258

-- Define the discount functions for each coupon
def coupon1_discount (price : ℝ) : ℝ := 0.12 * price

def coupon2_discount (price : ℝ) : ℝ := 30

def coupon3_discount (price : ℝ) : ℝ := 0.15 * (price - 150)

def coupon4_discount (price : ℝ) : ℝ := 25 + 0.05 * (price - 25)

-- Define a function to check if Coupon 1 gives the best discount
def coupon1_is_best (price : ℝ) : Prop :=
  coupon1_discount price > coupon2_discount price ∧
  coupon1_discount price > coupon3_discount price ∧
  coupon1_discount price > coupon4_discount price

-- Theorem stating that Coupon 1 is best for $300, $350, and $400
theorem coupon1_best_for_given_prices :
  coupon1_is_best 300 ∧ coupon1_is_best 350 ∧ coupon1_is_best 400 :=
by sorry

-- Additional theorem to show Coupon 1 is not best for $200 and $250
theorem coupon1_not_best_for_lower_prices :
  ¬(coupon1_is_best 200) ∧ ¬(coupon1_is_best 250) :=
by sorry

end coupon1_best_for_given_prices_coupon1_not_best_for_lower_prices_l1802_180258


namespace sequence_ratio_l1802_180233

/-- Given two sequences where (-1, a₁, a₂, 8) form an arithmetic sequence
and (-1, b₁, b₂, b₃, -4) form a geometric sequence,
prove that (a₁ * a₂) / b₂ = -5 -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  ((-1 : ℝ) - a₁ = a₁ - a₂) → 
  (a₂ - a₁ = 8 - a₂) → 
  (b₁ / (-1 : ℝ) = b₂ / b₁) → 
  (b₂ / b₁ = b₃ / b₂) → 
  (b₃ / b₂ = (-4 : ℝ) / b₃) → 
  (a₁ * a₂) / b₂ = -5 := by
sorry

end sequence_ratio_l1802_180233


namespace exponential_sum_rule_l1802_180284

theorem exponential_sum_rule (a : ℝ) (x₁ x₂ : ℝ) (ha : 0 < a) :
  a^(x₁ + x₂) = a^x₁ * a^x₂ := by
  sorry

end exponential_sum_rule_l1802_180284


namespace pure_imaginary_complex_number_l1802_180211

theorem pure_imaginary_complex_number (x : ℝ) :
  (Complex.I * (x + 1) = (x^2 - 1) + Complex.I * (x + 1)) → (x = 1 ∨ x = -1) := by
  sorry

end pure_imaginary_complex_number_l1802_180211


namespace integral_sin_plus_sqrt_one_minus_x_squared_l1802_180207

open Real MeasureTheory

theorem integral_sin_plus_sqrt_one_minus_x_squared (f g : ℝ → ℝ) :
  (∫ x in (-1)..1, f x) = 0 →
  (∫ x in (-1)..1, g x) = π / 2 →
  (∫ x in (-1)..1, f x + g x) = π / 2 :=
by sorry

end integral_sin_plus_sqrt_one_minus_x_squared_l1802_180207


namespace farmer_bean_seedlings_per_row_l1802_180267

/-- Represents the farmer's planting scenario -/
structure FarmPlanting where
  bean_seedlings : ℕ
  pumpkin_seeds : ℕ
  pumpkin_per_row : ℕ
  radishes : ℕ
  radishes_per_row : ℕ
  rows_per_bed : ℕ
  plant_beds : ℕ

/-- Calculates the number of bean seedlings per row -/
def bean_seedlings_per_row (fp : FarmPlanting) : ℕ :=
  fp.bean_seedlings / (fp.plant_beds * fp.rows_per_bed - 
    (fp.pumpkin_seeds / fp.pumpkin_per_row + fp.radishes / fp.radishes_per_row))

/-- Theorem stating that given the farmer's planting scenario, 
    the number of bean seedlings per row is 8 -/
theorem farmer_bean_seedlings_per_row :
  let fp : FarmPlanting := {
    bean_seedlings := 64,
    pumpkin_seeds := 84,
    pumpkin_per_row := 7,
    radishes := 48,
    radishes_per_row := 6,
    rows_per_bed := 2,
    plant_beds := 14
  }
  bean_seedlings_per_row fp = 8 := by
  sorry

end farmer_bean_seedlings_per_row_l1802_180267


namespace board_cut_theorem_l1802_180266

theorem board_cut_theorem (total_length : ℝ) (shorter_piece : ℝ) : 
  total_length = 69 →
  total_length = shorter_piece + 2 * shorter_piece →
  shorter_piece = 23 := by
  sorry

end board_cut_theorem_l1802_180266


namespace floor_of_4_7_l1802_180253

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end floor_of_4_7_l1802_180253


namespace rectangular_prism_volume_l1802_180278

theorem rectangular_prism_volume
  (side_area front_area bottom_area : ℝ)
  (h₁ : side_area = 12)
  (h₂ : front_area = 8)
  (h₃ : bottom_area = 6)
  : ∃ (length width height : ℝ),
    length * width = front_area ∧
    width * height = side_area ∧
    length * height = bottom_area ∧
    length * width * height = 24 :=
by sorry

end rectangular_prism_volume_l1802_180278


namespace workers_wage_increase_l1802_180227

/-- If a worker's daily wage is increased by 40% resulting in a new wage of $35 per day, 
    then the original daily wage was $25. -/
theorem workers_wage_increase (original_wage : ℝ) 
  (h1 : original_wage * 1.4 = 35) : original_wage = 25 := by
  sorry

end workers_wage_increase_l1802_180227


namespace min_dot_product_l1802_180251

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the center O and left focus F
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-3, 0)

-- Define a point P on the right branch of the hyperbola
def P : ℝ × ℝ → Prop := λ p => hyperbola p.1 p.2 ∧ p.1 ≥ 2

-- Define the dot product of OP and FP
def dot_product (p : ℝ × ℝ) : ℝ := p.1 * (p.1 + 3) + p.2 * p.2

-- Theorem statement
theorem min_dot_product :
  ∀ p : ℝ × ℝ, P p → ∀ q : ℝ × ℝ, P q → dot_product p ≥ 10 :=
sorry

end min_dot_product_l1802_180251


namespace no_valid_numbers_with_19x_relation_l1802_180246

/-- Checks if a natural number is composed only of digits 2, 3, 4, and 9 -/
def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [2, 3, 4, 9]

/-- The main theorem stating the impossibility of finding two numbers
    with the given properties -/
theorem no_valid_numbers_with_19x_relation :
  ¬∃ (a b : ℕ), is_valid_number a ∧ is_valid_number b ∧ b = 19 * a :=
sorry

end no_valid_numbers_with_19x_relation_l1802_180246


namespace K_factorization_l1802_180250

theorem K_factorization (x y z : ℝ) : 
  (x + 2*y + 3*z) * (2*x - y - z) * (y + 2*z + 3*x) +
  (y + 2*z + 3*x) * (2*y - z - x) * (z + 2*x + 3*y) +
  (z + 2*x + 3*y) * (2*z - x - y) * (x + 2*y + 3*z) =
  (y + z - 2*x) * (z + x - 2*y) * (x + y - 2*z) := by
sorry

end K_factorization_l1802_180250


namespace clock_cost_price_l1802_180252

theorem clock_cost_price (total_clocks : ℕ) (clocks_10_percent : ℕ) (clocks_20_percent : ℕ)
  (price_difference : ℝ) :
  total_clocks = 90 →
  clocks_10_percent = 40 →
  clocks_20_percent = 50 →
  price_difference = 40 →
  ∃ (cost_price : ℝ),
    cost_price = 80 ∧
    (clocks_10_percent : ℝ) * cost_price * 1.1 +
    (clocks_20_percent : ℝ) * cost_price * 1.2 -
    (total_clocks : ℝ) * cost_price * 1.15 = price_difference :=
by sorry

end clock_cost_price_l1802_180252


namespace jane_exercise_goal_l1802_180219

/-- Jane's exercise routine -/
structure ExerciseRoutine where
  daily_hours : ℕ
  days_per_week : ℕ
  total_hours : ℕ

/-- Calculate the number of weeks Jane hit her goal -/
def weeks_goal_met (routine : ExerciseRoutine) : ℕ :=
  routine.total_hours / (routine.daily_hours * routine.days_per_week)

/-- Theorem: Jane hit her goal for 8 weeks -/
theorem jane_exercise_goal (routine : ExerciseRoutine) 
  (h1 : routine.daily_hours = 1)
  (h2 : routine.days_per_week = 5)
  (h3 : routine.total_hours = 40) : 
  weeks_goal_met routine = 8 := by
  sorry

end jane_exercise_goal_l1802_180219


namespace systematic_sampling_size_l1802_180293

/-- Proves that the sample size for systematic sampling is 6 given the conditions of the problem -/
theorem systematic_sampling_size (total_population : Nat) (n : Nat) 
  (h1 : total_population = 36)
  (h2 : total_population % n = 0)
  (h3 : (total_population - 1) % (n + 1) = 0) : 
  n = 6 := by
  sorry

end systematic_sampling_size_l1802_180293


namespace three_painters_three_rooms_l1802_180255

/-- Represents the time taken for painters to complete rooms -/
def time_to_complete (painters : ℕ) (rooms : ℕ) : ℝ := sorry

/-- The work rate is proportional to the number of painters -/
axiom work_rate_proportional (p1 p2 r1 r2 : ℕ) (t : ℝ) :
  time_to_complete p1 r1 = t → time_to_complete p2 r2 = t * (r2 * p1 : ℝ) / (r1 * p2 : ℝ)

theorem three_painters_three_rooms : 
  time_to_complete 9 27 = 9 → time_to_complete 3 3 = 3 :=
by sorry

end three_painters_three_rooms_l1802_180255


namespace power_of_two_l1802_180268

theorem power_of_two (n : ℕ) : 32 * (1/2)^2 = 2^n → n = 7 := by
  sorry

end power_of_two_l1802_180268


namespace max_value_on_ellipse_l1802_180296

theorem max_value_on_ellipse :
  ∀ x y : ℝ, (x^2 / 6 + y^2 / 4 = 1) →
  ∃ (max : ℝ), (∀ x' y' : ℝ, (x'^2 / 6 + y'^2 / 4 = 1) → x' + 2*y' ≤ max) ∧
  max = Real.sqrt 22 := by
  sorry

end max_value_on_ellipse_l1802_180296


namespace circle_symmetry_l1802_180276

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 - x + 2*y = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x + 2)^2 + (y - 3/2)^2 = 5/4

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y x' y' : ℝ),
  original_circle x y →
  symmetry_line ((x + x') / 2) ((y + y') / 2) →
  symmetric_circle x' y' :=
sorry

end circle_symmetry_l1802_180276


namespace hot_dogs_remainder_l1802_180244

theorem hot_dogs_remainder : 25197643 % 6 = 3 := by
  sorry

end hot_dogs_remainder_l1802_180244


namespace gcd_9155_4892_l1802_180221

theorem gcd_9155_4892 : Nat.gcd 9155 4892 = 1 := by
  sorry

end gcd_9155_4892_l1802_180221


namespace semicircle_perimeter_l1802_180261

/-- The perimeter of a semi-circle with radius 14 cm is 14π + 28 cm -/
theorem semicircle_perimeter :
  let r : ℝ := 14
  let diameter : ℝ := 2 * r
  let half_circumference : ℝ := π * r
  let perimeter : ℝ := half_circumference + diameter
  perimeter = 14 * π + 28 := by sorry

end semicircle_perimeter_l1802_180261


namespace curve_is_ellipse_l1802_180294

open Real

/-- Given that θ is an internal angle of an oblique triangle and 
    F: x²sin²θcos²θ + y²sin²θ = cos²θ is the equation of a curve,
    prove that F represents an ellipse with foci on the x-axis and eccentricity sin θ. -/
theorem curve_is_ellipse (θ : ℝ) (h1 : 0 < θ ∧ θ < π) 
  (h2 : ∀ (x y : ℝ), x^2 * (sin θ)^2 * (cos θ)^2 + y^2 * (sin θ)^2 = (cos θ)^2 → 
    ∃ (a b : ℝ), 0 < b ∧ b < a ∧ x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    (a^2 - b^2) / a^2 = (sin θ)^2) : 
  ∃ (a b : ℝ), 0 < b ∧ b < a ∧ 
    (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ 
      x^2 * (sin θ)^2 * (cos θ)^2 + y^2 * (sin θ)^2 = (cos θ)^2) ∧
    (a^2 - b^2) / a^2 = (sin θ)^2 := by
  sorry

end curve_is_ellipse_l1802_180294


namespace overlapping_area_is_64_l1802_180297

/-- Represents a square sheet of paper -/
structure Sheet :=
  (side_length : ℝ)

/-- Represents the rotation of a sheet -/
inductive Rotation
  | NoRotation
  | Rotate45
  | Rotate90

/-- Represents the configuration of three sheets -/
structure SheetConfiguration :=
  (bottom : Sheet)
  (middle : Sheet)
  (top : Sheet)
  (middle_rotation : Rotation)
  (top_rotation : Rotation)

/-- Calculates the area of the overlapping polygon -/
def overlapping_area (config : SheetConfiguration) : ℝ :=
  sorry

/-- Theorem stating that the overlapping area is 64 for the given configuration -/
theorem overlapping_area_is_64 :
  ∀ (config : SheetConfiguration),
    config.bottom.side_length = 8 ∧
    config.middle.side_length = 8 ∧
    config.top.side_length = 8 ∧
    config.middle_rotation = Rotation.Rotate45 ∧
    config.top_rotation = Rotation.Rotate90 →
    overlapping_area config = 64 :=
  sorry

end overlapping_area_is_64_l1802_180297


namespace pies_sold_is_fifteen_l1802_180214

/-- Represents the number of slices in an apple pie -/
def apple_slices : ℕ := 8

/-- Represents the number of slices in a peach pie -/
def peach_slices : ℕ := 6

/-- Represents the number of apple pie slices ordered -/
def apple_orders : ℕ := 56

/-- Represents the number of peach pie slices ordered -/
def peach_orders : ℕ := 48

/-- Calculates the total number of pies sold based on the given conditions -/
def total_pies_sold : ℕ := apple_orders / apple_slices + peach_orders / peach_slices

/-- Theorem stating that the total number of pies sold is 15 -/
theorem pies_sold_is_fifteen : total_pies_sold = 15 := by
  sorry

end pies_sold_is_fifteen_l1802_180214
