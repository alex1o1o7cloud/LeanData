import Mathlib

namespace cubic_equation_roots_l2517_251784

theorem cubic_equation_roots (p : ℝ) : 
  (p = 6 ∨ p = -6) → 
  ∃ x y : ℝ, x ≠ y ∧ y - x = 1 ∧ 
  x^3 - 7*x + p = 0 ∧ 
  y^3 - 7*y + p = 0 := by
sorry

end cubic_equation_roots_l2517_251784


namespace like_terms_sum_l2517_251783

theorem like_terms_sum (a b : ℤ) : 
  (a + 1 = 2) ∧ (b - 2 = 3) → a + b = 6 := by
  sorry

end like_terms_sum_l2517_251783


namespace age_ratio_is_two_to_one_l2517_251746

def B_current_age : ℕ := 34
def A_current_age : ℕ := B_current_age + 4

def A_future_age : ℕ := A_current_age + 10
def B_past_age : ℕ := B_current_age - 10

theorem age_ratio_is_two_to_one :
  A_future_age / B_past_age = 2 ∧ A_future_age % B_past_age = 0 := by
  sorry

end age_ratio_is_two_to_one_l2517_251746


namespace repeating_decimal_proof_l2517_251787

def repeating_decimal : ℚ := 78 / 99

theorem repeating_decimal_proof :
  repeating_decimal = 26 / 33 ∧
  26 + 33 = 59 := by
  sorry

#eval (Nat.gcd 78 99)  -- Expected output: 3
#eval (78 / 3)         -- Expected output: 26
#eval (99 / 3)         -- Expected output: 33

end repeating_decimal_proof_l2517_251787


namespace sport_water_amount_l2517_251745

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (cornSyrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standardRatio : DrinkRatio :=
  { flavoring := 1,
    cornSyrup := 12,
    water := 30 }

/-- The sport formulation ratio -/
def sportRatio : DrinkRatio :=
  { flavoring := standardRatio.flavoring,
    cornSyrup := standardRatio.cornSyrup / 3,
    water := standardRatio.water * 2 }

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def sportCornSyrup : ℚ := 6

/-- Theorem: The amount of water in the sport formulation is 90 ounces -/
theorem sport_water_amount : 
  (sportRatio.water / sportRatio.cornSyrup) * sportCornSyrup = 90 := by
  sorry

end sport_water_amount_l2517_251745


namespace equal_population_after_17_years_l2517_251728

/-- The number of years needed for two villages' populations to become equal -/
def years_until_equal_population (x_initial : ℕ) (x_decrease : ℕ) (y_initial : ℕ) (y_increase : ℕ) : ℕ :=
  (x_initial - y_initial) / (x_decrease + y_increase)

/-- Theorem stating that the populations of Village X and Village Y will be equal after 17 years -/
theorem equal_population_after_17_years :
  years_until_equal_population 76000 1200 42000 800 = 17 := by
  sorry

end equal_population_after_17_years_l2517_251728


namespace bea_earned_more_than_dawn_l2517_251709

/-- The price of Bea's lemonade in cents -/
def bea_price : ℕ := 25

/-- The price of Dawn's lemonade in cents -/
def dawn_price : ℕ := 28

/-- The number of glasses Bea sold -/
def bea_glasses : ℕ := 10

/-- The number of glasses Dawn sold -/
def dawn_glasses : ℕ := 8

/-- Theorem: Bea earned 26 cents more than Dawn -/
theorem bea_earned_more_than_dawn : 
  bea_price * bea_glasses - dawn_price * dawn_glasses = 26 := by
  sorry

end bea_earned_more_than_dawn_l2517_251709


namespace complex_expression_simplification_l2517_251717

theorem complex_expression_simplification :
  (7 - 3*Complex.I) - 3*(2 + 4*Complex.I) + 4*(1 - Complex.I) = 5 - 19*Complex.I :=
by sorry

end complex_expression_simplification_l2517_251717


namespace initial_puppies_count_l2517_251775

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℕ := 7

/-- The number of puppies Alyssa has left -/
def puppies_left : ℕ := 5

/-- The initial number of puppies Alyssa had -/
def initial_puppies : ℕ := puppies_given_away + puppies_left

theorem initial_puppies_count : initial_puppies = 12 := by
  sorry

end initial_puppies_count_l2517_251775


namespace special_triangle_properties_l2517_251791

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Properties of the specific triangle in the problem -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a + t.b + t.c = Real.sqrt 2 + 1 ∧
  Real.sin t.A + Real.sin t.B = Real.sqrt 2 * Real.sin t.C ∧
  (1/2) * t.a * t.b * Real.sin t.C = (1/5) * Real.sin t.C

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.c = 1 ∧ Real.cos t.C = 1/4 := by
  sorry


end special_triangle_properties_l2517_251791


namespace union_of_sets_l2517_251758

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 4}
  let B : Set ℕ := {2, 4, 6}
  A ∪ B = {1, 2, 4, 6} := by sorry

end union_of_sets_l2517_251758


namespace jack_position_change_l2517_251741

-- Define the constants from the problem
def flights_up : ℕ := 3
def flights_down : ℕ := 6
def steps_per_flight : ℕ := 12
def inches_per_step : ℕ := 8
def inches_per_foot : ℕ := 12

-- Define the function to calculate the net change in position
def net_position_change : ℚ :=
  (flights_down - flights_up) * steps_per_flight * inches_per_step / inches_per_foot

-- Theorem statement
theorem jack_position_change :
  net_position_change = 24 := by sorry

end jack_position_change_l2517_251741


namespace drums_hit_calculation_l2517_251723

/-- Represents the drumming contest scenario --/
structure DrummingContest where
  entryFee : ℝ
  costPerDrum : ℝ
  earningsStartDrum : ℕ
  earningsPerDrum : ℝ
  bonusRoundDrum : ℕ
  totalLoss : ℝ

/-- Calculates the number of drums hit in the contest --/
def drumsHit (contest : DrummingContest) : ℕ :=
  sorry

/-- Theorem stating the number of drums hit in the given scenario --/
theorem drums_hit_calculation (contest : DrummingContest) 
  (h1 : contest.entryFee = 10)
  (h2 : contest.costPerDrum = 0.02)
  (h3 : contest.earningsStartDrum = 200)
  (h4 : contest.earningsPerDrum = 0.025)
  (h5 : contest.bonusRoundDrum = 250)
  (h6 : contest.totalLoss = 7.5) :
  drumsHit contest = 4500 :=
sorry

end drums_hit_calculation_l2517_251723


namespace hyperbola_to_ellipse_l2517_251765

/-- Given a hyperbola with equation x²/4 - y²/12 = -1, 
    the ellipse with foci at the vertices of this hyperbola 
    has the equation x²/4 + y²/16 = 1 -/
theorem hyperbola_to_ellipse : 
  ∃ (h : Set (ℝ × ℝ)) (e : Set (ℝ × ℝ)),
    (h = {(x, y) | x^2/4 - y^2/12 = -1}) →
    (e = {(x, y) | x^2/4 + y^2/16 = 1}) →
    (∀ (fx fy : ℝ), (fx, fy) ∈ {v | v ∈ h ∧ (∀ (x y : ℝ), (x, y) ∈ h → x^2 + y^2 ≤ fx^2 + fy^2)} →
      (fx, fy) ∈ {f | f ∈ e ∧ (∀ (x y : ℝ), (x, y) ∈ e → (x - fx)^2 + (y - fy)^2 ≥ 
        (x + fx)^2 + (y + fy)^2)}) :=
by sorry

end hyperbola_to_ellipse_l2517_251765


namespace solve_equation_l2517_251797

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 2 / 3 → x = -27 / 23 := by
  sorry

end solve_equation_l2517_251797


namespace point_on_line_l2517_251769

/-- A point (x, y) lies on a line passing through (x1, y1) and (x2, y2) if and only if
    (y - y1) / (x - x1) = (y2 - y1) / (x2 - x1) -/
def on_line (x y x1 y1 x2 y2 : ℚ) : Prop :=
  (y - y1) * (x2 - x1) = (y2 - y1) * (x - x1)

theorem point_on_line :
  on_line (-2/3) (-1) 2 1 10 7 := by sorry

end point_on_line_l2517_251769


namespace fourth_term_max_coefficient_l2517_251753

def has_max_fourth_term (n : ℕ) : Prop :=
  ∀ k, 0 ≤ k ∧ k ≤ n → Nat.choose n 3 ≥ Nat.choose n k

theorem fourth_term_max_coefficient (n : ℕ) :
  has_max_fourth_term n ↔ n = 5 ∨ n = 6 ∨ n = 7 := by sorry

end fourth_term_max_coefficient_l2517_251753


namespace pure_imaginary_square_root_l2517_251706

theorem pure_imaginary_square_root (a : ℝ) :
  (∃ (b : ℝ), (a - Complex.I) ^ 2 = Complex.I * b) → (a = 1 ∨ a = -1) := by
  sorry

end pure_imaginary_square_root_l2517_251706


namespace estimate_fish_population_l2517_251778

/-- Estimates the total number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population (initially_marked : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) :
  initially_marked = 120 →
  second_catch = 100 →
  marked_in_second = 10 →
  (initially_marked * second_catch) / marked_in_second = 1200 :=
by
  sorry

#check estimate_fish_population

end estimate_fish_population_l2517_251778


namespace profit_difference_l2517_251770

def business_problem (a b c : ℕ) (b_profit : ℕ) : Prop :=
  let total_capital := a + b + c
  let a_ratio := a * b_profit * 3 / b
  let c_ratio := c * b_profit * 3 / b
  c_ratio - a_ratio = 760

theorem profit_difference :
  business_problem 8000 10000 12000 1900 :=
sorry

end profit_difference_l2517_251770


namespace interest_calculation_l2517_251742

def total_investment : ℝ := 33000
def rate1 : ℝ := 0.04
def rate2 : ℝ := 0.0225
def partial_investment : ℝ := 13000

theorem interest_calculation :
  ∃ (investment1 investment2 : ℝ),
    investment1 + investment2 = total_investment ∧
    (investment1 = partial_investment ∨ investment2 = partial_investment) ∧
    investment1 * rate1 + investment2 * rate2 = 970 :=
by sorry

end interest_calculation_l2517_251742


namespace range_of_m_l2517_251772

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -Real.sqrt (4 - p.2^2)}

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 6}

-- Define the point A
def A (m : ℝ) : ℝ × ℝ := (m, 0)

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (∃ P ∈ C, ∃ Q ∈ l, A m + P - (A m + Q) = (0, 0)) →
  m ∈ Set.Icc 2 3 :=
sorry

end range_of_m_l2517_251772


namespace mike_catches_l2517_251759

/-- The number of times Joe caught the ball -/
def J : ℕ := 23

/-- The number of times Derek caught the ball -/
def D : ℕ := 2 * J - 4

/-- The number of times Tammy caught the ball -/
def T : ℕ := (D / 3) + 16

/-- The number of times Mike caught the ball -/
def M : ℕ := (2 * T * 120) / 100

theorem mike_catches : M = 72 := by
  sorry

end mike_catches_l2517_251759


namespace linear_function_characterization_l2517_251732

theorem linear_function_characterization (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) → 
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
by sorry

end linear_function_characterization_l2517_251732


namespace b_work_days_l2517_251716

/-- The number of days it takes for two workers to complete a job together -/
def combined_days : ℕ := 16

/-- The number of days it takes for worker 'a' to complete the job alone -/
def a_days : ℕ := 24

/-- The work rate of a worker is the fraction of the job they complete in one day -/
def work_rate (days : ℕ) : ℚ := 1 / days

/-- The number of days it takes for worker 'b' to complete the job alone -/
def b_days : ℕ := 48

theorem b_work_days : 
  work_rate combined_days = work_rate a_days + work_rate b_days :=
sorry

end b_work_days_l2517_251716


namespace conference_games_count_l2517_251793

/-- Calculates the number of games in a sports conference season. -/
def conference_games (total_teams : ℕ) (division_size : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let teams_per_division := total_teams / 2
  let intra_games := teams_per_division * (division_size - 1) * intra_division_games
  let inter_games := total_teams * division_size * inter_division_games
  (intra_games + inter_games) / 2

/-- Theorem stating the number of games in the specific conference setup. -/
theorem conference_games_count : 
  conference_games 16 8 3 2 = 296 := by
  sorry

end conference_games_count_l2517_251793


namespace power_of_power_l2517_251761

theorem power_of_power (a : ℝ) : (a^5)^3 = a^15 := by
  sorry

end power_of_power_l2517_251761


namespace sandy_book_purchase_l2517_251749

/-- The number of books Sandy bought from the first shop -/
def books_first_shop : ℕ := 65

/-- The amount Sandy spent at the first shop -/
def amount_first_shop : ℕ := 1380

/-- The number of books Sandy bought from the second shop -/
def books_second_shop : ℕ := 55

/-- The amount Sandy spent at the second shop -/
def amount_second_shop : ℕ := 900

/-- The average price per book -/
def average_price : ℕ := 19

theorem sandy_book_purchase :
  books_first_shop = 65 ∧
  (amount_first_shop + amount_second_shop : ℚ) / (books_first_shop + books_second_shop) = average_price := by
  sorry

end sandy_book_purchase_l2517_251749


namespace eventually_linear_closed_under_addition_l2517_251743

theorem eventually_linear_closed_under_addition (S : Set ℕ) 
  (h_closed : ∀ a b : ℕ, a ∈ S → b ∈ S → (a + b) ∈ S) :
  ∃ k N : ℕ, ∀ n : ℕ, n > N → (n ∈ S ↔ k ∣ n) := by
  sorry

end eventually_linear_closed_under_addition_l2517_251743


namespace expansion_terms_count_l2517_251751

/-- The number of terms in the expansion of a product of two sums -/
def num_terms_in_expansion (m n : ℕ) : ℕ := m * n

/-- Theorem: The expansion of (a+b+c+d)(e+f+g+h+i) has 20 terms -/
theorem expansion_terms_count : num_terms_in_expansion 4 5 = 20 := by
  sorry

end expansion_terms_count_l2517_251751


namespace regular_polygon_sides_is_ten_l2517_251748

/-- The number of sides of a regular polygon with an interior angle of 144 degrees -/
def regular_polygon_sides : ℕ := by
  -- Define the interior angle
  let interior_angle : ℝ := 144

  -- Define the function for the sum of interior angles of an n-sided polygon
  let sum_of_angles (n : ℕ) : ℝ := 180 * (n - 2)

  -- Define the equation: sum of angles equals n times the interior angle
  let sides_equation (n : ℕ) : Prop := sum_of_angles n = n * interior_angle

  -- The number of sides is the solution to this equation
  exact sorry

theorem regular_polygon_sides_is_ten : regular_polygon_sides = 10 := by sorry

end regular_polygon_sides_is_ten_l2517_251748


namespace mango_rate_is_59_l2517_251722

/-- Calculates the rate per kg for mangoes given the total amount paid, grape price, grape weight, and mango weight. -/
def mango_rate (total_paid : ℕ) (grape_price : ℕ) (grape_weight : ℕ) (mango_weight : ℕ) : ℕ :=
  (total_paid - grape_price * grape_weight) / mango_weight

/-- Theorem stating that under the given conditions, the mango rate is 59 -/
theorem mango_rate_is_59 :
  mango_rate 975 74 6 9 = 59 := by
  sorry

#eval mango_rate 975 74 6 9

end mango_rate_is_59_l2517_251722


namespace smallest_c_for_inequality_l2517_251779

theorem smallest_c_for_inequality : ∃ c : ℕ, c = 9 ∧ (∀ k : ℕ, 27 ^ k > 3 ^ 24 → k ≥ c) := by
  sorry

end smallest_c_for_inequality_l2517_251779


namespace rectangle_longer_side_l2517_251756

def circle_radius : ℝ := 6

theorem rectangle_longer_side (rectangle_area rectangle_shorter_side rectangle_longer_side : ℝ) : 
  rectangle_area = 3 * (π * circle_radius^2) →
  rectangle_shorter_side = 2 * circle_radius →
  rectangle_area = rectangle_shorter_side * rectangle_longer_side →
  rectangle_longer_side = 9 * π := by
sorry

end rectangle_longer_side_l2517_251756


namespace triangle_division_l2517_251731

/-- The number of triangles formed by n points inside a triangle -/
def numTriangles (n : ℕ) : ℕ := 2 * n + 1

/-- Theorem stating that 1997 points inside a triangle creates 3995 smaller triangles -/
theorem triangle_division (n : ℕ) (h : n = 1997) : numTriangles n = 3995 := by
  sorry

end triangle_division_l2517_251731


namespace tiling_iff_div_four_l2517_251767

/-- A T-tetromino is a shape that covers exactly 4 squares. -/
def TTetromino : Type := Unit

/-- A tiling of an n×n board with T-tetrominos. -/
def Tiling (n : ℕ) : Type := 
  {arrangement : Fin n → Fin n → Option TTetromino // 
    ∀ (i j : Fin n), ∃ (t : TTetromino), arrangement i j = some t}

/-- The main theorem: An n×n board can be tiled with T-tetrominos iff n is divisible by 4. -/
theorem tiling_iff_div_four (n : ℕ) : 
  (∃ (t : Tiling n), True) ↔ 4 ∣ n := by sorry

end tiling_iff_div_four_l2517_251767


namespace quadrilateral_area_not_integer_l2517_251786

theorem quadrilateral_area_not_integer (n : ℕ) : 
  ¬ (∃ (m : ℕ), m^2 = n * (n + 1) * (n + 2) * (n + 3)) :=
by sorry

end quadrilateral_area_not_integer_l2517_251786


namespace no_prime_solution_l2517_251768

/-- Represents a number in base p notation -/
def BaseP (coeffs : List Nat) (p : Nat) : Nat :=
  coeffs.enum.foldl (fun acc (i, a) => acc + a * p^i) 0

theorem no_prime_solution :
  ¬∃ (p : Nat), 
    Nat.Prime p ∧ 
    (BaseP [7, 1, 0, 2] p + BaseP [2, 0, 4] p + BaseP [4, 1, 1] p + 
     BaseP [0, 3, 2] p + BaseP [7] p = 
     BaseP [1, 0, 3] p + BaseP [2, 7, 4] p + BaseP [3, 1, 5] p) :=
by sorry

#eval BaseP [7, 1, 0, 2] 10  -- Should output 2017
#eval BaseP [2, 0, 4] 10     -- Should output 402
#eval BaseP [4, 1, 1] 10     -- Should output 114
#eval BaseP [0, 3, 2] 10     -- Should output 230
#eval BaseP [7] 10           -- Should output 7
#eval BaseP [1, 0, 3] 10     -- Should output 301
#eval BaseP [2, 7, 4] 10     -- Should output 472
#eval BaseP [3, 1, 5] 10     -- Should output 503

end no_prime_solution_l2517_251768


namespace product_sum_coefficients_l2517_251781

theorem product_sum_coefficients :
  ∀ (A B C D : ℝ), 
  (∀ x : ℝ, (2 * x^2 - 3 * x + 5) * (5 - 3 * x) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 8 :=
by sorry

end product_sum_coefficients_l2517_251781


namespace solution_of_linear_equation_l2517_251708

theorem solution_of_linear_equation (x y a : ℝ) : 
  x = 1 → y = 3 → a * x - 2 * y = 4 → a = 10 := by sorry

end solution_of_linear_equation_l2517_251708


namespace line_equation_slope_intercept_l2517_251725

/-- Given a line equation, prove its slope and y-intercept -/
theorem line_equation_slope_intercept :
  let line_eq : ℝ × ℝ → ℝ := λ p => 3 * (p.1 + 2) + (-7) * (p.2 - 4)
  ∃ m b : ℝ, m = 3 / 7 ∧ b = -34 / 7 ∧
    ∀ x y : ℝ, line_eq (x, y) = 0 ↔ y = m * x + b := by
  sorry

end line_equation_slope_intercept_l2517_251725


namespace regression_lines_common_point_l2517_251757

/-- Represents a regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Represents a dataset with means -/
structure Dataset where
  x_mean : ℝ
  y_mean : ℝ

/-- Checks if a point is on a regression line -/
def point_on_line (line : RegressionLine) (x y : ℝ) : Prop :=
  y = line.slope * x + line.intercept

/-- Theorem: Two regression lines with the same dataset means have a common point -/
theorem regression_lines_common_point 
  (line1 line2 : RegressionLine) (data : Dataset) :
  point_on_line line1 data.x_mean data.y_mean →
  point_on_line line2 data.x_mean data.y_mean →
  ∃ (x y : ℝ), point_on_line line1 x y ∧ point_on_line line2 x y :=
sorry

end regression_lines_common_point_l2517_251757


namespace abs_z_eq_sqrt_10_div_2_l2517_251780

theorem abs_z_eq_sqrt_10_div_2 (z : ℂ) (h : (1 - Complex.I) * z = 1 + 2 * Complex.I) :
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end abs_z_eq_sqrt_10_div_2_l2517_251780


namespace cubic_sum_identity_l2517_251777

theorem cubic_sum_identity (y : ℝ) (h : y^3 + 1/y^3 = 110) : y + 1/y = 5 := by
  sorry

end cubic_sum_identity_l2517_251777


namespace apple_picking_problem_l2517_251752

theorem apple_picking_problem (maggie_apples layla_apples average_apples : ℕ) 
  (h1 : maggie_apples = 40)
  (h2 : layla_apples = 22)
  (h3 : average_apples = 30)
  (h4 : (maggie_apples + layla_apples + kelsey_apples) / 3 = average_apples) :
  kelsey_apples = 28 := by
  sorry

end apple_picking_problem_l2517_251752


namespace nine_valid_numbers_l2517_251792

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≥ 0 ∧ units ≤ 9

/-- Reverses the digits of a two-digit number -/
def reverse (n : TwoDigitNumber) : TwoDigitNumber :=
  ⟨n.units, n.tens, by sorry⟩

/-- Converts a TwoDigitNumber to a natural number -/
def to_nat (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- Checks if a natural number is a positive perfect square -/
def is_positive_perfect_square (n : Nat) : Prop :=
  ∃ m : Nat, m > 0 ∧ m * m = n

/-- The main theorem to prove -/
theorem nine_valid_numbers :
  ∃ (S : Finset TwoDigitNumber),
    S.card = 9 ∧
    (∀ n : TwoDigitNumber, n ∈ S ↔
      is_positive_perfect_square (to_nat n - to_nat (reverse n))) ∧
    (∀ n : TwoDigitNumber,
      is_positive_perfect_square (to_nat n - to_nat (reverse n)) →
      n ∈ S) :=
by sorry

end nine_valid_numbers_l2517_251792


namespace expression_evaluation_l2517_251737

theorem expression_evaluation (a b c : ℝ) 
  (ha : a = 15) (hb : b = 19) (hc : c = 13) : 
  (a^2 * (1/c - 1/b) + b^2 * (1/a - 1/c) + c^2 * (1/b - 1/a)) / 
  (a * (1/c - 1/b) + b * (1/a - 1/c) + c * (1/b - 1/a)) = a + b + c := by
  sorry

end expression_evaluation_l2517_251737


namespace sector_central_angle_l2517_251789

/-- The central angle of a circular sector, given its radius and area -/
theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 2) (h2 : area = 4) :
  (2 * area) / (r ^ 2) = 2 := by
  sorry

end sector_central_angle_l2517_251789


namespace circle_intersection_range_l2517_251774

theorem circle_intersection_range (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = m ∧ x^2 + y^2 + 6*x - 8*y - 11 = 0) →
  1 ≤ m ∧ m ≤ 121 := by
sorry

end circle_intersection_range_l2517_251774


namespace arithmetic_sequence_24th_term_l2517_251721

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the 3rd term is 7 and the 10th term is 27,
    the 24th term is 67. -/
theorem arithmetic_sequence_24th_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_3rd : a 3 = 7)
  (h_10th : a 10 = 27) :
  a 24 = 67 := by
sorry


end arithmetic_sequence_24th_term_l2517_251721


namespace base_is_ten_l2517_251701

-- Define a function to convert a number from base h to decimal
def to_decimal (digits : List Nat) (h : Nat) : Nat :=
  digits.foldr (fun d acc => d + h * acc) 0

-- Define a function to check if the equation holds in base h
def equation_holds (h : Nat) : Prop :=
  to_decimal [5, 7, 3, 4] h + to_decimal [6, 4, 2, 1] h = to_decimal [1, 4, 1, 5, 5] h

-- Theorem statement
theorem base_is_ten : ∃ h, h = 10 ∧ equation_holds h := by
  sorry

end base_is_ten_l2517_251701


namespace rationalize_denominator_l2517_251776

theorem rationalize_denominator : 
  (Real.sqrt 18 - Real.sqrt 8) / (Real.sqrt 8 + Real.sqrt 2) = (1 + Real.sqrt 2) / 3 := by
sorry

end rationalize_denominator_l2517_251776


namespace division_fraction_equality_l2517_251713

theorem division_fraction_equality : (2 / 7) / (1 / 14) = 4 := by sorry

end division_fraction_equality_l2517_251713


namespace planet_combinations_count_l2517_251755

/-- Represents the number of Earth-like planets -/
def earth_like_planets : ℕ := 5

/-- Represents the number of Mars-like planets -/
def mars_like_planets : ℕ := 6

/-- Represents the colonization units required for an Earth-like planet -/
def earth_like_units : ℕ := 2

/-- Represents the colonization units required for a Mars-like planet -/
def mars_like_units : ℕ := 1

/-- Represents the total available colonization units -/
def total_units : ℕ := 12

/-- Calculates the number of ways to choose planets given the constraints -/
def count_planet_combinations : ℕ :=
  (Nat.choose earth_like_planets 3 * Nat.choose mars_like_planets 6) +
  (Nat.choose earth_like_planets 4 * Nat.choose mars_like_planets 4) +
  (Nat.choose earth_like_planets 5 * Nat.choose mars_like_planets 2)

/-- Theorem stating that the number of planet combinations is 100 -/
theorem planet_combinations_count :
  count_planet_combinations = 100 := by sorry

end planet_combinations_count_l2517_251755


namespace volume_of_cut_cube_l2517_251790

/-- Represents a three-dimensional solid --/
structure Solid :=
  (volume : ℝ)

/-- Represents a cube --/
def Cube (edge_length : ℝ) : Solid :=
  { volume := edge_length ^ 3 }

/-- Represents the result of cutting parts off a cube --/
def CutCube (c : Solid) (cut_volume : ℝ) : Solid :=
  { volume := c.volume - cut_volume }

/-- Theorem stating that the volume of the resulting solid is 9 --/
theorem volume_of_cut_cube : 
  ∃ (cut_volume : ℝ), 
    (CutCube (Cube 3) cut_volume).volume = 9 :=
sorry

end volume_of_cut_cube_l2517_251790


namespace short_trees_planted_calculation_park_short_trees_planted_l2517_251799

/-- The number of short trees planted in a park -/
def short_trees_planted (initial_short_trees final_short_trees : ℕ) : ℕ :=
  final_short_trees - initial_short_trees

/-- Theorem stating that the number of short trees planted is the difference between the final and initial number of short trees -/
theorem short_trees_planted_calculation (initial_short_trees final_short_trees : ℕ) 
  (h : final_short_trees ≥ initial_short_trees) :
  short_trees_planted initial_short_trees final_short_trees = final_short_trees - initial_short_trees :=
by
  sorry

/-- The specific case for the park problem -/
theorem park_short_trees_planted :
  short_trees_planted 41 98 = 57 :=
by
  sorry

end short_trees_planted_calculation_park_short_trees_planted_l2517_251799


namespace unique_n_for_equation_l2517_251750

theorem unique_n_for_equation : ∃! (n : ℕ+), 
  ∃ (x y : ℕ+), y^2 + x*y + 3*x = n*(x^2 + x*y + 3*y) := by
  sorry

end unique_n_for_equation_l2517_251750


namespace trigonometric_identities_l2517_251715

theorem trigonometric_identities (θ : ℝ) (h : Real.sin (θ - π/3) = 1/3) :
  (Real.sin (θ + 2*π/3) = -1/3) ∧ (Real.cos (θ - 5*π/6) = 1/3) := by
  sorry

end trigonometric_identities_l2517_251715


namespace gcd_lcm_product_24_60_l2517_251727

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end gcd_lcm_product_24_60_l2517_251727


namespace exists_subset_with_constant_gcd_l2517_251738

/-- A function that checks if a natural number is the product of at most 2000 distinct primes -/
def is_product_of_limited_primes (n : ℕ) : Prop :=
  ∃ (primes : Finset ℕ), (∀ p ∈ primes, Nat.Prime p) ∧ primes.card ≤ 2000 ∧ n = primes.prod id

/-- The main theorem -/
theorem exists_subset_with_constant_gcd 
  (A : Set ℕ) 
  (h_infinite : Set.Infinite A) 
  (h_limited_primes : ∀ a ∈ A, is_product_of_limited_primes a) :
  ∃ (B : Set ℕ) (k : ℕ), Set.Infinite B ∧ B ⊆ A ∧ 
    ∀ (b1 b2 : ℕ), b1 ∈ B → b2 ∈ B → b1 ≠ b2 → Nat.gcd b1 b2 = k :=
sorry

end exists_subset_with_constant_gcd_l2517_251738


namespace ratio_problem_l2517_251703

theorem ratio_problem (a b c P : ℝ) 
  (h1 : b / (a + c) = 1 / 2)
  (h2 : a / (b + c) = 1 / P) :
  (a + b + c) / a = 4 := by
sorry

end ratio_problem_l2517_251703


namespace characterize_valid_common_differences_l2517_251796

/-- A number is interesting if 2018 divides its number of positive divisors -/
def IsInteresting (n : ℕ) : Prop :=
  2018 ∣ (Nat.divisors n).card

/-- An arithmetic progression with first term a and common difference k -/
def ArithmeticProgression (a k : ℕ) : ℕ → ℕ :=
  fun i => a + i * k

/-- The property of k being a valid common difference for an infinite
    arithmetic progression of interesting numbers -/
def IsValidCommonDifference (k : ℕ) : Prop :=
  ∃ a : ℕ, ∀ i : ℕ, IsInteresting (ArithmeticProgression a k i)

/-- The main theorem characterizing valid common differences -/
theorem characterize_valid_common_differences :
  ∀ k : ℕ, k > 0 →
  (IsValidCommonDifference k ↔
    (∃ (m : ℕ) (p : ℕ), m > 0 ∧ Nat.Prime p ∧ k = m * p^1009) ∧
    k ≠ 2^2009) :=
  sorry

end characterize_valid_common_differences_l2517_251796


namespace bryan_tshirt_count_l2517_251795

def total_cost : ℕ := 1500
def tshirt_cost : ℕ := 100
def pants_cost : ℕ := 250
def pants_count : ℕ := 4

theorem bryan_tshirt_count :
  (total_cost - pants_count * pants_cost) / tshirt_cost = 5 := by
  sorry

end bryan_tshirt_count_l2517_251795


namespace brick_height_calculation_l2517_251718

theorem brick_height_calculation (brick_length : ℝ) (brick_width : ℝ)
  (wall_length : ℝ) (wall_height : ℝ) (wall_width : ℝ)
  (num_bricks : ℕ) :
  brick_length = 25 →
  brick_width = 11.25 →
  wall_length = 800 →
  wall_height = 600 →
  wall_width = 22.5 →
  num_bricks = 6400 →
  ∃ brick_height : ℝ,
    brick_height = 6 ∧
    wall_length * wall_height * wall_width =
    num_bricks * (brick_length * brick_width * brick_height) :=
by
  sorry

#check brick_height_calculation

end brick_height_calculation_l2517_251718


namespace cube_root_sum_zero_implies_opposite_l2517_251714

theorem cube_root_sum_zero_implies_opposite (x y : ℝ) : 
  (x^(1/3 : ℝ) + y^(1/3 : ℝ) = 0) → (x = -y) := by
  sorry

end cube_root_sum_zero_implies_opposite_l2517_251714


namespace equilateral_triangle_perimeter_l2517_251710

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) :
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_perimeter_l2517_251710


namespace largest_angle_in_triangle_l2517_251744

theorem largest_angle_in_triangle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = 45 →           -- One angle is 45°
  5 * b = 4 * c →    -- The other two angles are in the ratio 4:5
  max a (max b c) = 75 -- The largest angle is 75°
  := by sorry

end largest_angle_in_triangle_l2517_251744


namespace sqrt_meaningful_range_l2517_251734

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end sqrt_meaningful_range_l2517_251734


namespace tiles_difference_l2517_251764

/-- The side length of the nth square in the sequence -/
def side_length (n : ℕ) : ℕ := 2 * n - 1

/-- The number of tiles in the nth square -/
def tiles_in_square (n : ℕ) : ℕ := (side_length n) ^ 2

/-- The theorem stating the difference in tiles between the 10th and 9th squares -/
theorem tiles_difference : tiles_in_square 10 - tiles_in_square 9 = 72 := by
  sorry

end tiles_difference_l2517_251764


namespace henrys_money_l2517_251707

/-- Henry's money calculation -/
theorem henrys_money (initial_amount : ℕ) (birthday_gift : ℕ) (spent_amount : ℕ) : 
  initial_amount = 11 → birthday_gift = 18 → spent_amount = 10 → 
  initial_amount + birthday_gift - spent_amount = 19 := by
  sorry

#check henrys_money

end henrys_money_l2517_251707


namespace f_shifted_positive_set_l2517_251704

/-- An odd function f defined on ℝ satisfying f(x) = 2^x - 4 for x > 0 -/
noncomputable def f : ℝ → ℝ :=
  fun x => if x > 0 then 2^x - 4 else -(2^(-x) - 4)

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f(x) = 2^x - 4 for x > 0 -/
axiom f_pos : ∀ x, x > 0 → f x = 2^x - 4

theorem f_shifted_positive_set :
  {x : ℝ | f (x - 1) > 0} = {x : ℝ | -1 < x ∧ x < 1 ∨ x > 3} :=
sorry

end f_shifted_positive_set_l2517_251704


namespace remainder_theorem_l2517_251740

theorem remainder_theorem (n : ℤ) : 
  (∃ k : ℤ, 2 * n = 10 * k + 2) → 
  (∃ m : ℤ, n = 20 * m + 1) := by
  sorry

end remainder_theorem_l2517_251740


namespace inscribed_triangle_angle_60_l2517_251712

/-- Represents a triangle inscribed in a circle -/
structure InscribedTriangle where
  /-- Measure of arc PQ -/
  arc_pq : ℝ
  /-- Measure of arc QR -/
  arc_qr : ℝ
  /-- Measure of arc RP -/
  arc_rp : ℝ
  /-- The sum of all arcs is 360° -/
  sum_arcs : arc_pq + arc_qr + arc_rp = 360

/-- Theorem: If a triangle is inscribed in a circle with the given arc measures,
    then one of its interior angles is 60° -/
theorem inscribed_triangle_angle_60 (t : InscribedTriangle)
  (h1 : ∃ x : ℝ, t.arc_pq = x + 80 ∧ t.arc_qr = 3*x - 30 ∧ t.arc_rp = 2*x + 10) :
  ∃ θ : ℝ, θ = 60 ∧ (θ = t.arc_qr / 2 ∨ θ = t.arc_rp / 2 ∨ θ = t.arc_pq / 2) := by
  sorry

end inscribed_triangle_angle_60_l2517_251712


namespace annie_diorama_building_time_l2517_251705

/-- The time Annie spent building her diorama -/
def building_time (planning_time : ℕ) : ℕ := 3 * planning_time - 5

/-- The total time Annie spent on her diorama project -/
def total_time (planning_time : ℕ) : ℕ := building_time planning_time + planning_time

theorem annie_diorama_building_time :
  ∃ (planning_time : ℕ), total_time planning_time = 67 ∧ building_time planning_time = 49 := by
sorry

end annie_diorama_building_time_l2517_251705


namespace product_reciprocals_equals_one_l2517_251735

theorem product_reciprocals_equals_one :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end product_reciprocals_equals_one_l2517_251735


namespace sqrt_five_addition_l2517_251726

theorem sqrt_five_addition : 2 * Real.sqrt 5 + 3 * Real.sqrt 5 = 5 * Real.sqrt 5 := by
  sorry

end sqrt_five_addition_l2517_251726


namespace sin_identity_l2517_251773

theorem sin_identity (α : Real) (h : Real.sin (α - π/4) = 1/2) :
  Real.sin (5*π/4 - α) = 1/2 := by
  sorry

end sin_identity_l2517_251773


namespace tenth_minus_ninth_square_diff_l2517_251762

/-- The number of tiles in the nth square of the sequence -/
def tiles_in_square (n : ℕ) : ℕ := n^2

/-- The theorem stating the difference in tiles between the 10th and 9th squares -/
theorem tenth_minus_ninth_square_diff : tiles_in_square 10 - tiles_in_square 9 = 19 := by
  sorry

end tenth_minus_ninth_square_diff_l2517_251762


namespace blue_hat_cost_is_6_l2517_251711

-- Define the total number of hats
def total_hats : ℕ := 85

-- Define the cost of each green hat
def green_hat_cost : ℕ := 7

-- Define the total price
def total_price : ℕ := 540

-- Define the number of green hats
def green_hats : ℕ := 30

-- Define the number of blue hats
def blue_hats : ℕ := total_hats - green_hats

-- Define the cost of green hats
def green_hats_cost : ℕ := green_hats * green_hat_cost

-- Define the cost of blue hats
def blue_hats_cost : ℕ := total_price - green_hats_cost

-- Theorem: The cost of each blue hat is $6
theorem blue_hat_cost_is_6 : blue_hats_cost / blue_hats = 6 := by
  sorry

end blue_hat_cost_is_6_l2517_251711


namespace p_half_q_age_years_ago_l2517_251771

/-- The number of years ago when p was half of q in age -/
def years_ago : ℕ := 12

/-- The present age of p -/
def p_age : ℕ := 18

/-- The present age of q -/
def q_age : ℕ := 24

/-- Theorem: Given the conditions, prove that p was half of q in age 12 years ago -/
theorem p_half_q_age_years_ago :
  (p_age : ℚ) / (q_age : ℚ) = 3 / 4 ∧
  p_age + q_age = 42 ∧
  (p_age - years_ago : ℚ) = (q_age - years_ago : ℚ) / 2 := by
  sorry

end p_half_q_age_years_ago_l2517_251771


namespace sixth_term_value_l2517_251760

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the properties of a₄ and a₈
def roots_property (a : ℕ → ℝ) : Prop :=
  a 4 ^ 2 - 34 * a 4 + 64 = 0 ∧ a 8 ^ 2 - 34 * a 8 + 64 = 0

-- Theorem statement
theorem sixth_term_value (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : roots_property a) : 
  a 6 = 8 := by sorry

end sixth_term_value_l2517_251760


namespace sally_grew_five_onions_l2517_251763

/-- The number of onions grown by Sara -/
def sara_onions : ℕ := 4

/-- The number of onions grown by Fred -/
def fred_onions : ℕ := 9

/-- The total number of onions grown -/
def total_onions : ℕ := 18

/-- The number of onions grown by Sally -/
def sally_onions : ℕ := total_onions - (sara_onions + fred_onions)

theorem sally_grew_five_onions : sally_onions = 5 := by
  sorry

end sally_grew_five_onions_l2517_251763


namespace f_minus_g_equals_one_l2517_251785

-- Define f and g as functions from ℝ to ℝ
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- State the theorem
theorem f_minus_g_equals_one 
  (h_even : is_even f) 
  (h_odd : is_odd g) 
  (h_sum : ∀ x, f x + g x = x^3 + x^2 + 1) : 
  f 1 - g 1 = 1 := by
sorry

end f_minus_g_equals_one_l2517_251785


namespace quadrilateral_bd_value_l2517_251724

/-- Represents a quadrilateral ABCD with given side lengths and diagonal BD --/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  BD : ℤ

/-- The quadrilateral satisfies the triangle inequality --/
def satisfies_triangle_inequality (q : Quadrilateral) : Prop :=
  q.AB + q.BD > q.DA ∧
  q.BC + q.CD > q.BD ∧
  q.DA + q.BD > q.AB ∧
  q.BD + q.CD > q.BC

/-- The theorem to be proved --/
theorem quadrilateral_bd_value (q : Quadrilateral) 
  (h1 : q.AB = 6)
  (h2 : q.BC = 19)
  (h3 : q.CD = 6)
  (h4 : q.DA = 10)
  (h5 : satisfies_triangle_inequality q) :
  q.BD = 15 := by
  sorry

end quadrilateral_bd_value_l2517_251724


namespace jessy_reading_plan_l2517_251739

/-- The number of pages Jessy initially plans to read each time -/
def pages_per_reading : ℕ := sorry

/-- The total number of pages in the book -/
def total_pages : ℕ := 140

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of times Jessy reads per day -/
def readings_per_day : ℕ := 3

/-- The additional pages Jessy needs to read per day to achieve her goal -/
def additional_pages_per_day : ℕ := 2

theorem jessy_reading_plan :
  pages_per_reading = 6 ∧
  days_in_week * readings_per_day * pages_per_reading + 
  days_in_week * additional_pages_per_day = total_pages := by
  sorry

end jessy_reading_plan_l2517_251739


namespace sheridan_cats_l2517_251766

/-- The number of cats Mrs. Sheridan gave away -/
def cats_given_away : ℝ := 14.0

/-- The number of cats Mrs. Sheridan has left -/
def cats_left : ℕ := 3

/-- The initial number of cats Mrs. Sheridan had -/
def initial_cats : ℕ := 17

theorem sheridan_cats : ↑initial_cats = cats_given_away + cats_left := by sorry

end sheridan_cats_l2517_251766


namespace least_number_divisible_by_multiple_l2517_251747

theorem least_number_divisible_by_multiple (n : ℕ) : n = 856 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ k₄ : ℕ, 
    (m + 8 = 24 * k₁) ∧ 
    (m + 8 = 32 * k₂) ∧ 
    (m + 8 = 36 * k₃) ∧ 
    (m + 8 = 54 * k₄))) ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℕ, 
    (n + 8 = 24 * k₁) ∧ 
    (n + 8 = 32 * k₂) ∧ 
    (n + 8 = 36 * k₃) ∧ 
    (n + 8 = 54 * k₄)) := by
  sorry

end least_number_divisible_by_multiple_l2517_251747


namespace geometric_sequence_product_l2517_251730

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 5 * a 14 = 5 →
  a 8 * a 9 * a 10 * a 11 = 25 := by
sorry

end geometric_sequence_product_l2517_251730


namespace correct_fill_in_l2517_251733

def sentence (phrase : String) : String :=
  s!"It will not be {phrase} we meet again."

def correctPhrase : String := "long before"

theorem correct_fill_in :
  sentence correctPhrase = "It will not be long before we meet again." :=
by sorry

end correct_fill_in_l2517_251733


namespace al_original_amount_l2517_251798

/-- Represents the investment scenario with Al, Betty, and Clare --/
structure Investment where
  al : ℝ
  betty : ℝ
  clare : ℝ

/-- The conditions of the investment problem --/
def validInvestment (inv : Investment) : Prop :=
  inv.al + inv.betty + inv.clare = 1200 ∧
  (inv.al - 200) + (3 * inv.betty) + (4 * inv.clare) = 1800

/-- The theorem stating Al's original investment amount --/
theorem al_original_amount :
  ∀ inv : Investment, validInvestment inv → inv.al = 860 := by
  sorry

end al_original_amount_l2517_251798


namespace move_right_example_l2517_251794

/-- Represents a point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point horizontally by a given distance -/
def moveRight (p : Point) (distance : ℝ) : Point :=
  { x := p.x + distance, y := p.y }

/-- The theorem stating that moving (-1, 3) by 5 units right results in (4, 3) -/
theorem move_right_example :
  let initial := Point.mk (-1) 3
  let final := moveRight initial 5
  final = Point.mk 4 3 := by sorry

end move_right_example_l2517_251794


namespace sum_of_solutions_eq_five_l2517_251720

theorem sum_of_solutions_eq_five :
  let f : ℝ → ℝ := λ M => M * (M - 5) + 9
  ∃ M₁ M₂ : ℝ, (f M₁ = 0 ∧ f M₂ = 0 ∧ M₁ ≠ M₂) ∧ M₁ + M₂ = 5 :=
by sorry

end sum_of_solutions_eq_five_l2517_251720


namespace number_of_preferred_shares_l2517_251719

/-- Represents the number of preferred shares -/
def preferred_shares : ℕ := sorry

/-- Represents the number of common shares -/
def common_shares : ℕ := 3000

/-- Represents the par value of each share in rupees -/
def par_value : ℚ := 50

/-- Represents the annual dividend rate for preferred shares -/
def preferred_dividend_rate : ℚ := 1 / 10

/-- Represents the annual dividend rate for common shares -/
def common_dividend_rate : ℚ := 7 / 100

/-- Represents the total annual dividend received in rupees -/
def total_annual_dividend : ℚ := 16500

/-- Theorem stating that the number of preferred shares is 1200 -/
theorem number_of_preferred_shares : 
  preferred_shares = 1200 :=
by sorry

end number_of_preferred_shares_l2517_251719


namespace max_square_plots_l2517_251754

/-- Represents the field dimensions and available fencing -/
structure FieldData where
  width : ℝ
  length : ℝ
  fence : ℝ

/-- Calculates the number of square plots given the number of plots along the width -/
def numPlots (n : ℕ) : ℕ := n * (2 * n)

/-- Calculates the length of fence used given the number of plots along the width -/
def fenceUsed (n : ℕ) : ℝ := 120 * n - 90

/-- The main theorem stating the maximum number of square plots -/
theorem max_square_plots (field : FieldData) 
    (h_width : field.width = 30)
    (h_length : field.length = 60)
    (h_fence : field.fence = 2268) : 
  (∃ (n : ℕ), numPlots n = 722 ∧ 
              fenceUsed n ≤ field.fence ∧ 
              ∀ (m : ℕ), m > n → fenceUsed m > field.fence) :=
sorry

end max_square_plots_l2517_251754


namespace mitch_weekend_hours_l2517_251788

/-- Represents Mitch's work schedule and earnings --/
structure MitchWork where
  weekdayHours : ℕ  -- Hours worked per weekday
  weekdayRate : ℕ   -- Hourly rate for weekdays in dollars
  totalWeeklyEarnings : ℕ  -- Total weekly earnings in dollars
  weekendRate : ℕ   -- Hourly rate for weekends in dollars

/-- Calculates the number of weekend hours Mitch works --/
def weekendHours (m : MitchWork) : ℕ :=
  let weekdayEarnings := m.weekdayHours * 5 * m.weekdayRate
  let weekendEarnings := m.totalWeeklyEarnings - weekdayEarnings
  weekendEarnings / m.weekendRate

/-- Theorem stating that Mitch works 6 hours on weekends --/
theorem mitch_weekend_hours :
  ∀ (m : MitchWork),
  m.weekdayHours = 5 ∧
  m.weekdayRate = 3 ∧
  m.totalWeeklyEarnings = 111 ∧
  m.weekendRate = 6 →
  weekendHours m = 6 :=
by
  sorry

end mitch_weekend_hours_l2517_251788


namespace range_of_a_l2517_251702

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 0 1 ∧ 2^x₀ * (3 * x₀ + a) < 1) → a < 1 := by
  sorry

end range_of_a_l2517_251702


namespace min_value_theorem_l2517_251736

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_eq : 2*m + n = 2) :
  (2/m) + (1/n) ≥ 9/2 ∧ ∃ m n, m > 0 ∧ n > 0 ∧ 2*m + n = 2 ∧ (2/m) + (1/n) = 9/2 :=
sorry

end min_value_theorem_l2517_251736


namespace quadratic_real_solutions_range_l2517_251782

theorem quadratic_real_solutions_range (m : ℝ) :
  (∃ x : ℝ, m * x^2 + 2 * x + 1 = 0) ∧ (m ≠ 0) ↔ m ≤ 1 ∧ m ≠ 0 := by
  sorry

end quadratic_real_solutions_range_l2517_251782


namespace total_animal_eyes_pond_animal_eyes_l2517_251700

theorem total_animal_eyes (num_frogs num_crocodiles : ℕ) 
  (eyes_per_frog eyes_per_crocodile : ℕ) : ℕ :=
  num_frogs * eyes_per_frog + num_crocodiles * eyes_per_crocodile

theorem pond_animal_eyes : total_animal_eyes 20 6 2 2 = 52 := by
  sorry

end total_animal_eyes_pond_animal_eyes_l2517_251700


namespace mila_coin_collection_value_l2517_251729

/-- The total value of Mila's coin collection -/
def total_value (gold_coins silver_coins : ℕ) (gold_value silver_value : ℚ) : ℚ :=
  gold_coins * gold_value + silver_coins * silver_value

/-- Theorem stating the total value of Mila's coin collection -/
theorem mila_coin_collection_value :
  let gold_coins : ℕ := 20
  let silver_coins : ℕ := 15
  let gold_value : ℚ := 10 / 4
  let silver_value : ℚ := 15 / 5
  total_value gold_coins silver_coins gold_value silver_value = 95
  := by sorry

end mila_coin_collection_value_l2517_251729
