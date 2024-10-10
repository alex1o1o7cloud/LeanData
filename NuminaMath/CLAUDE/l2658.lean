import Mathlib

namespace parallel_non_coincident_lines_l2658_265854

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- Two lines are distinct if and only if their y-intercepts are different -/
axiom distinct_lines_different_intercepts {m b1 b2 : ℝ} :
  (∃ x y : ℝ, y = m * x + b1 ∧ y ≠ m * x + b2) ↔ b1 ≠ b2

theorem parallel_non_coincident_lines (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0 ↔ y = -a/2 * x - 3) ∧
  (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0 ↔ y = -1/(a-1) * x - (a^2-1)/(a-1)) ∧
  (∀ x y : ℝ, y = -a/2 * x - 3 ↔ y = -1/(a-1) * x - (a^2-1)/(a-1)) ∧
  (∃ x y : ℝ, y = -a/2 * x - 3 ∧ y ≠ -1/(a-1) * x - (a^2-1)/(a-1)) →
  a = -1 := by sorry

end parallel_non_coincident_lines_l2658_265854


namespace y_satisfies_equation_l2658_265876

noncomputable def y (a x : ℝ) : ℝ := a * Real.tan (Real.sqrt (a / x - 1))

theorem y_satisfies_equation (a x : ℝ) (h1 : x ≠ 0) (h2 : a / x - 1 ≥ 0) :
  a^2 + (y a x)^2 + 2 * x * Real.sqrt (a * x - x^2) * (deriv (y a) x) = 0 := by
  sorry

end y_satisfies_equation_l2658_265876


namespace smallest_e_value_l2658_265863

theorem smallest_e_value (a b c d e : ℤ) : 
  (∃ (x : ℝ), a * x^4 + b * x^3 + c * x^2 + d * x + e = 0) →
  (a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0) →
  (a * 4^4 + b * 4^3 + c * 4^2 + d * 4 + e = 0) →
  (a * 8^4 + b * 8^3 + c * 8^2 + d * 8 + e = 0) →
  (a * (-1/4)^4 + b * (-1/4)^3 + c * (-1/4)^2 + d * (-1/4) + e = 0) →
  e > 0 →
  e ≥ 96 := by
sorry

end smallest_e_value_l2658_265863


namespace min_value_of_function_l2658_265848

theorem min_value_of_function (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt (x^2 - 3*x + 3) + Real.sqrt (y^2 - 3*y + 3) + Real.sqrt (x^2 - Real.sqrt 3 * x * y + y^2) ≥ Real.sqrt 6 := by
  sorry

end min_value_of_function_l2658_265848


namespace pen_notebook_ratio_l2658_265835

theorem pen_notebook_ratio (num_notebooks : ℕ) : 
  num_notebooks = 40 → 
  (5 : ℚ) / 4 * num_notebooks = 50 := by
  sorry

end pen_notebook_ratio_l2658_265835


namespace candidate_votes_l2658_265862

theorem candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_percent : ℚ) :
  total_votes = 560000 →
  invalid_percent = 15 / 100 →
  candidate_percent = 75 / 100 →
  ↑⌊(1 - invalid_percent) * candidate_percent * total_votes⌋ = 357000 := by
  sorry

end candidate_votes_l2658_265862


namespace sqrt_product_quotient_l2658_265842

theorem sqrt_product_quotient :
  3 * Real.sqrt 5 * (2 * Real.sqrt 15) / Real.sqrt 3 = 30 := by
  sorry

end sqrt_product_quotient_l2658_265842


namespace symmetric_points_m_value_l2658_265865

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry with respect to the y-axis
def symmetricAboutYAxis (p1 p2 : Point2D) : Prop :=
  p1.x = -p2.x ∧ p1.y = p2.y

-- Theorem statement
theorem symmetric_points_m_value :
  let A : Point2D := ⟨-3, 4⟩
  let B : Point2D := ⟨3, m⟩
  symmetricAboutYAxis A B → m = 4 := by
  sorry

end symmetric_points_m_value_l2658_265865


namespace cylinder_circumference_l2658_265858

/-- Given two right circular cylinders C and B, prove that the circumference of C is 8√5 meters -/
theorem cylinder_circumference (h_C h_B r_B : ℝ) (vol_C vol_B : ℝ) : 
  h_C = 10 →
  h_B = 8 →
  2 * Real.pi * r_B = 10 →
  vol_C = Real.pi * (h_C * r_C^2) →
  vol_B = Real.pi * (h_B * r_B^2) →
  vol_C = 0.8 * vol_B →
  2 * Real.pi * r_C = 8 * Real.sqrt 5 :=
by sorry

end cylinder_circumference_l2658_265858


namespace total_money_found_l2658_265826

def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10
def nickel_value : ℚ := 0.05
def penny_value : ℚ := 0.01

def num_quarters : ℕ := 10
def num_dimes : ℕ := 3
def num_nickels : ℕ := 3
def num_pennies : ℕ := 5

theorem total_money_found :
  (num_quarters : ℚ) * quarter_value +
  (num_dimes : ℚ) * dime_value +
  (num_nickels : ℚ) * nickel_value +
  (num_pennies : ℚ) * penny_value = 3 := by sorry

end total_money_found_l2658_265826


namespace circle_in_circle_theorem_l2658_265898

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a point
def Point := ℝ × ℝ

-- Define what it means for a point to be inside a circle
def is_inside (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 < c.radius^2

-- Define what it means for a point to be on a circle
def is_on (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define what it means for one circle to be contained in another
def is_contained (c1 c2 : Circle) : Prop :=
  ∀ (p : Point), is_on p c1 → is_inside p c2

-- State the theorem
theorem circle_in_circle_theorem (ω : Circle) (A B : Point) 
  (h1 : is_inside A ω) (h2 : is_inside B ω) : 
  ∃ (ω' : Circle), is_on A ω' ∧ is_on B ω' ∧ is_contained ω' ω := by
  sorry

end circle_in_circle_theorem_l2658_265898


namespace table_tennis_match_results_l2658_265871

/-- Represents a "best-of-3" table tennis match -/
structure TableTennisMatch where
  prob_a_win : ℝ
  prob_b_win : ℝ

/-- The probability of player A winning a single game -/
def prob_a_win (m : TableTennisMatch) : ℝ := m.prob_a_win

/-- The probability of player B winning a single game -/
def prob_b_win (m : TableTennisMatch) : ℝ := m.prob_b_win

/-- The probability of player A winning the entire match -/
def prob_a_win_match (m : TableTennisMatch) : ℝ :=
  (m.prob_a_win)^2 + 2 * m.prob_b_win * (m.prob_a_win)^2

/-- The expected number of games won by player A -/
def expected_games_won_a (m : TableTennisMatch) : ℝ :=
  1 * (2 * m.prob_a_win * (m.prob_b_win)^2) + 2 * ((m.prob_a_win)^2 + 2 * m.prob_b_win * (m.prob_a_win)^2)

/-- The variance of the number of games won by player A -/
def variance_games_won_a (m : TableTennisMatch) : ℝ :=
  (m.prob_b_win)^2 * (0 - expected_games_won_a m)^2 +
  (2 * m.prob_a_win * (m.prob_b_win)^2) * (1 - expected_games_won_a m)^2 +
  ((m.prob_a_win)^2 + 2 * m.prob_b_win * (m.prob_a_win)^2) * (2 - expected_games_won_a m)^2

theorem table_tennis_match_results (m : TableTennisMatch) 
  (h1 : m.prob_a_win = 0.6) 
  (h2 : m.prob_b_win = 0.4) : 
  prob_a_win_match m = 0.648 ∧ 
  expected_games_won_a m = 1.5 ∧ 
  variance_games_won_a m = 0.57 := by
  sorry

end table_tennis_match_results_l2658_265871


namespace complement_of_A_l2658_265816

def U : Set Int := Set.univ

def A : Set Int := {x : Int | x^2 - x - 2 ≥ 0}

theorem complement_of_A : Set.compl A = {0, 1} := by
  sorry

end complement_of_A_l2658_265816


namespace license_plate_count_l2658_265893

/-- The number of letters in the alphabet -/
def number_of_letters : ℕ := 26

/-- The number of digits (0-9) -/
def number_of_digits : ℕ := 10

/-- The number of even (or odd) digits -/
def number_of_even_digits : ℕ := 5

/-- The total number of license plates with 2 letters followed by 2 digits,
    where one digit is odd and the other is even -/
def total_license_plates : ℕ := number_of_letters^2 * number_of_digits * number_of_even_digits

theorem license_plate_count : total_license_plates = 33800 := by
  sorry

end license_plate_count_l2658_265893


namespace larger_integer_value_l2658_265818

theorem larger_integer_value (a b : ℕ+) 
  (h1 : (a : ℝ) / (b : ℝ) = 7 / 3) 
  (h2 : (a : ℕ) * b = 294) : 
  max a b = ⌈7 * Real.sqrt 14⌉ := by
  sorry

end larger_integer_value_l2658_265818


namespace smallest_integer_larger_than_root_sum_fourth_power_l2658_265877

theorem smallest_integer_larger_than_root_sum_fourth_power :
  ∃ n : ℕ, n = 248 ∧ (∀ m : ℕ, m < n → (m : ℝ) ≤ (Real.sqrt 5 + Real.sqrt 3)^4) ∧
  (n : ℝ) > (Real.sqrt 5 + Real.sqrt 3)^4 := by
  sorry

end smallest_integer_larger_than_root_sum_fourth_power_l2658_265877


namespace athlete_weights_problem_l2658_265849

theorem athlete_weights_problem (a b c : ℝ) (k₁ k₂ k₃ : ℤ) : 
  (a + b) / 2 = 40 →
  (b + c) / 2 = 43 →
  (a + c) / 2 = 44 →
  a + b = 5 * k₁ →
  b + c = 5 * k₂ →
  a + c = 5 * k₃ →
  b = 40 := by
  sorry

end athlete_weights_problem_l2658_265849


namespace gcd_459_357_l2658_265825

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l2658_265825


namespace lauren_mail_total_l2658_265806

/-- The number of pieces of mail Lauren sent on Monday -/
def monday : ℕ := 65

/-- The number of pieces of mail Lauren sent on Tuesday -/
def tuesday : ℕ := monday + 10

/-- The number of pieces of mail Lauren sent on Wednesday -/
def wednesday : ℕ := tuesday - 5

/-- The number of pieces of mail Lauren sent on Thursday -/
def thursday : ℕ := wednesday + 15

/-- The total number of pieces of mail Lauren sent over four days -/
def total : ℕ := monday + tuesday + wednesday + thursday

theorem lauren_mail_total : total = 295 := by sorry

end lauren_mail_total_l2658_265806


namespace board_division_theorem_l2658_265880

/-- Represents a cell on the board -/
structure Cell :=
  (x : Nat) (y : Nat) (shaded : Bool)

/-- Represents the board -/
def Board := List Cell

/-- Represents a rectangle on the board -/
structure Rectangle :=
  (topLeft : Cell) (width : Nat) (height : Nat)

/-- The initial board configuration -/
def initialBoard : Board := sorry

/-- Check if a cell is within a rectangle -/
def isInRectangle (cell : Cell) (rect : Rectangle) : Bool := sorry

/-- Count shaded cells in a rectangle -/
def countShadedCells (board : Board) (rect : Rectangle) : Nat := sorry

/-- Check if two rectangles are identical -/
def areIdenticalRectangles (rect1 rect2 : Rectangle) : Bool := sorry

/-- Main theorem -/
theorem board_division_theorem (board : Board) :
  ∃ (rect1 rect2 rect3 rect4 : Rectangle),
    (rect1.width = 4 ∧ rect1.height = 2) ∧
    (rect2.width = 4 ∧ rect2.height = 2) ∧
    (rect3.width = 4 ∧ rect3.height = 2) ∧
    (rect4.width = 4 ∧ rect4.height = 2) ∧
    areIdenticalRectangles rect1 rect2 ∧
    areIdenticalRectangles rect1 rect3 ∧
    areIdenticalRectangles rect1 rect4 ∧
    countShadedCells board rect1 = 3 ∧
    countShadedCells board rect2 = 3 ∧
    countShadedCells board rect3 = 3 ∧
    countShadedCells board rect4 = 3 :=
  sorry

end board_division_theorem_l2658_265880


namespace button_problem_l2658_265866

/-- Proof of the button-making problem --/
theorem button_problem (mari_buttons sue_buttons : ℕ) 
  (h_mari : mari_buttons = 8)
  (h_sue : sue_buttons = 22)
  (h_sue_half_kendra : sue_buttons * 2 = mari_buttons * x + 4)
  : x = 5 := by
  sorry

#check button_problem

end button_problem_l2658_265866


namespace weekend_to_weekday_ratio_is_two_to_one_l2658_265879

/-- Represents the earnings of an Italian restaurant over a month. -/
structure RestaurantEarnings where
  weekday_earnings : ℕ  -- Daily earnings on weekdays
  total_earnings : ℕ    -- Total earnings for the month
  weeks_per_month : ℕ   -- Number of weeks in the month

/-- Calculates the ratio of weekend to weekday earnings. -/
def weekend_to_weekday_ratio (r : RestaurantEarnings) : ℚ :=
  let weekday_total := r.weekday_earnings * 5 * r.weeks_per_month
  let weekend_total := r.total_earnings - weekday_total
  let weekend_daily := weekend_total / (2 * r.weeks_per_month)
  weekend_daily / r.weekday_earnings

/-- Theorem stating that the ratio of weekend to weekday earnings is 2:1. -/
theorem weekend_to_weekday_ratio_is_two_to_one 
  (r : RestaurantEarnings) 
  (h1 : r.weekday_earnings = 600)
  (h2 : r.total_earnings = 21600)
  (h3 : r.weeks_per_month = 4) : 
  weekend_to_weekday_ratio r = 2 := by
  sorry

end weekend_to_weekday_ratio_is_two_to_one_l2658_265879


namespace total_movies_l2658_265823

-- Define the number of movies Timothy watched in 2009
def timothy_2009 : ℕ := 24

-- Define the number of movies Timothy watched in 2010
def timothy_2010 : ℕ := timothy_2009 + 7

-- Define the number of movies Theresa watched in 2009
def theresa_2009 : ℕ := timothy_2009 / 2

-- Define the number of movies Theresa watched in 2010
def theresa_2010 : ℕ := timothy_2010 * 2

-- Theorem to prove
theorem total_movies : timothy_2009 + timothy_2010 + theresa_2009 + theresa_2010 = 129 := by
  sorry

end total_movies_l2658_265823


namespace ratio_problem_l2658_265878

theorem ratio_problem (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) :
  (2 * a + b) / (b + 2 * c) = 5 / 27 := by
  sorry

end ratio_problem_l2658_265878


namespace imaginary_part_of_z_is_zero_l2658_265841

theorem imaginary_part_of_z_is_zero (z : ℂ) (h : z * (Complex.I + 1) = 2 / (Complex.I - 1)) :
  z.im = 0 := by
  sorry

end imaginary_part_of_z_is_zero_l2658_265841


namespace prob_at_most_sixes_equals_sum_l2658_265855

def numDice : ℕ := 10
def maxSixes : ℕ := 3

def probExactlySixes (n : ℕ) : ℚ :=
  (Nat.choose numDice n) * (1/6)^n * (5/6)^(numDice - n)

def probAtMostSixes : ℚ :=
  (Finset.range (maxSixes + 1)).sum probExactlySixes

theorem prob_at_most_sixes_equals_sum :
  probAtMostSixes = 
    probExactlySixes 0 + probExactlySixes 1 + 
    probExactlySixes 2 + probExactlySixes 3 := by
  sorry

end prob_at_most_sixes_equals_sum_l2658_265855


namespace gum_cost_l2658_265888

/-- The cost of gum in cents -/
def cost_per_piece : ℕ := 2

/-- The number of pieces of gum -/
def num_pieces : ℕ := 500

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Theorem: The cost of 500 pieces of gum is 1000 cents and 10 dollars -/
theorem gum_cost :
  (num_pieces * cost_per_piece = 1000) ∧
  (num_pieces * cost_per_piece / cents_per_dollar = 10) :=
by sorry

end gum_cost_l2658_265888


namespace alphabet_value_proof_l2658_265839

/-- Given the alphabet values where H = 8, prove that A = 25 when PACK = 50, PECK = 54, and CAKE = 40 -/
theorem alphabet_value_proof (P A C K E : ℤ) (h1 : P + A + C + K = 50) (h2 : P + E + C + K = 54) (h3 : C + A + K + E = 40) : A = 25 := by
  sorry

end alphabet_value_proof_l2658_265839


namespace jessica_coins_value_l2658_265805

/-- Represents the value of a coin in cents -/
def coin_value (is_dime : Bool) : ℕ :=
  if is_dime then 10 else 5

/-- Calculates the total value of coins in cents -/
def total_value (num_nickels num_dimes : ℕ) : ℕ :=
  coin_value false * num_nickels + coin_value true * num_dimes

theorem jessica_coins_value :
  ∀ (num_nickels num_dimes : ℕ),
    num_nickels + num_dimes = 30 →
    total_value num_dimes num_nickels - total_value num_nickels num_dimes = 120 →
    total_value num_nickels num_dimes = 165 := by
  sorry

end jessica_coins_value_l2658_265805


namespace locus_of_p_forms_two_circles_l2658_265821

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the circle
structure PointOnCircle (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

-- Define the projection of a point onto a line
def project_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ × ℝ :=
  sorry

-- Define the point P on OQ such that OP = QQ'
def point_p (c : Circle) (q : PointOnCircle c) (diameter : ℝ × ℝ → ℝ) : ℝ × ℝ :=
  sorry

-- Theorem stating that the locus of P forms two circles
theorem locus_of_p_forms_two_circles (c : Circle) (diameter : ℝ × ℝ → ℝ) :
  ∃ (c1 c2 : Circle),
    ∀ (q : PointOnCircle c),
      let p := point_p c q diameter
      (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∨
      (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2 :=
sorry

end locus_of_p_forms_two_circles_l2658_265821


namespace only_statement_one_true_l2658_265811

variable (b x y : ℝ)

theorem only_statement_one_true :
  (∀ x y, b * (x + y) = b * x + b * y) ∧
  (∃ x y, b^(x + y) ≠ b^x + b^y) ∧
  (∃ x y, Real.log (x + y) ≠ Real.log x + Real.log y) ∧
  (∃ x y, Real.log x / Real.log y ≠ Real.log (x * y)) ∧
  (∃ x y, b * (x / y) ≠ (b * x) / (b * y)) :=
by sorry

end only_statement_one_true_l2658_265811


namespace tegwens_family_size_l2658_265872

theorem tegwens_family_size :
  ∀ (g b : ℕ),
  g > 0 →  -- At least one girl (Tegwen)
  b = g - 1 →  -- Tegwen has same number of brothers as sisters
  g = (3 * (b - 1)) / 2 →  -- Each brother has 50% more sisters than brothers
  g + b = 11 :=
by
  sorry

end tegwens_family_size_l2658_265872


namespace marble_probability_l2658_265824

theorem marble_probability (total : ℕ) (p_white p_green p_yellow p_orange : ℚ) :
  total = 500 →
  p_white = 1/4 →
  p_green = 1/5 →
  p_yellow = 1/6 →
  p_orange = 1/10 →
  let p_red_blue := 1 - (p_white + p_green + p_yellow + p_orange)
  p_red_blue = 71/250 := by
  sorry

end marble_probability_l2658_265824


namespace pizza_slice_cost_l2658_265896

theorem pizza_slice_cost (num_pizzas : ℕ) (slices_per_pizza : ℕ) (total_cost : ℚ) :
  num_pizzas = 3 →
  slices_per_pizza = 12 →
  total_cost = 72 →
  (5 : ℚ) * (total_cost / (↑num_pizzas * ↑slices_per_pizza)) = 10 := by
  sorry

end pizza_slice_cost_l2658_265896


namespace corrected_mean_l2658_265814

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 40 ∧ original_mean = 36 ∧ incorrect_value = 20 ∧ correct_value = 34 →
  (n * original_mean + (correct_value - incorrect_value)) / n = 36.35 := by
  sorry

end corrected_mean_l2658_265814


namespace incorrect_accuracy_statement_l2658_265875

def accurate_to_nearest_hundred (x : ℝ) : Prop :=
  ∃ n : ℤ, x = (n : ℝ) * 100 ∧ |x - (n : ℝ) * 100| ≤ 50

theorem incorrect_accuracy_statement :
  ¬(accurate_to_nearest_hundred 2130) :=
sorry

end incorrect_accuracy_statement_l2658_265875


namespace uki_biscuits_per_day_l2658_265807

/-- Represents the daily production and pricing of bakery items -/
structure BakeryData where
  cupcake_price : ℝ
  cookie_price : ℝ
  biscuit_price : ℝ
  cupcakes_per_day : ℕ
  cookie_packets_per_day : ℕ
  total_earnings_five_days : ℝ

/-- Calculates the number of biscuit packets that can be baked daily -/
def biscuits_per_day (data : BakeryData) : ℕ :=
  20

/-- Theorem stating that given the bakery data, Uki can bake 20 packets of biscuits per day -/
theorem uki_biscuits_per_day (data : BakeryData)
    (h1 : data.cupcake_price = 1.5)
    (h2 : data.cookie_price = 2)
    (h3 : data.biscuit_price = 1)
    (h4 : data.cupcakes_per_day = 20)
    (h5 : data.cookie_packets_per_day = 10)
    (h6 : data.total_earnings_five_days = 350) :
    biscuits_per_day data = 20 := by
  sorry

end uki_biscuits_per_day_l2658_265807


namespace fixed_point_of_f_l2658_265840

/-- The logarithm function with base a, where a > 0 and a ≠ 1 -/
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

/-- The function f(x) = log_a(x+1) - 2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 1) - 2

/-- Theorem: For any a > 0 and a ≠ 1, f(x) passes through the point (0, -2) -/
theorem fixed_point_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 0 = -2 := by
  sorry

end fixed_point_of_f_l2658_265840


namespace first_divisor_problem_l2658_265889

theorem first_divisor_problem (x : ℕ) : x = 31 ↔ 
  x > 9 ∧ 
  x < 282 ∧
  282 % x = 3 ∧
  282 % 9 = 3 ∧
  279 % x = 0 ∧
  ∀ y : ℕ, y > 9 ∧ y < x → (282 % y ≠ 3 ∨ 279 % y ≠ 0) :=
by sorry

end first_divisor_problem_l2658_265889


namespace cadastral_value_calculation_l2658_265852

/-- Calculates the cadastral value of a land plot given the tax amount and tax rate -/
theorem cadastral_value_calculation (tax_amount : ℝ) (tax_rate : ℝ) :
  tax_amount = 4500 →
  tax_rate = 0.003 →
  tax_amount = tax_rate * 1500000 := by
  sorry

#check cadastral_value_calculation

end cadastral_value_calculation_l2658_265852


namespace exists_arrangement_for_23_l2658_265808

/-- Fibonacci-like sequence with a specific recurrence relation -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of a sequence satisfying the required property for P = 23 -/
theorem exists_arrangement_for_23 : ∃ (F : ℕ → ℤ), F 12 ≡ 0 [ZMOD 23] := by
  sorry

end exists_arrangement_for_23_l2658_265808


namespace cricket_average_l2658_265820

theorem cricket_average (initial_average : ℝ) (innings : ℕ) (new_score : ℝ) (average_increase : ℝ) :
  innings = 16 →
  new_score = 92 →
  average_increase = 4 →
  (innings * initial_average + new_score) / (innings + 1) = initial_average + average_increase →
  initial_average + average_increase = 28 := by
  sorry

end cricket_average_l2658_265820


namespace max_value_theorem_l2658_265822

theorem max_value_theorem (a b c : ℝ) (h : a * b * c + a + c - b = 0) :
  ∃ (max : ℝ), max = 5/4 ∧ 
  ∀ (x y z : ℝ), x * y * z + x + z - y = 0 →
  (1 / (1 + x^2) - 1 / (1 + y^2) + 1 / (1 + z^2)) ≤ max :=
by sorry

end max_value_theorem_l2658_265822


namespace park_perimeter_calculation_l2658_265846

/-- The perimeter of a rectangular park with given length and breadth. -/
def park_perimeter (length breadth : ℝ) : ℝ :=
  2 * (length + breadth)

/-- Theorem stating that the perimeter of a rectangular park with length 300 m and breadth 200 m is 1000 m. -/
theorem park_perimeter_calculation :
  park_perimeter 300 200 = 1000 := by
  sorry

end park_perimeter_calculation_l2658_265846


namespace solution_count_of_system_l2658_265810

theorem solution_count_of_system (x y : ℂ) : 
  (y = (x + 1)^2 ∧ x * y + y = 1) → 
  (∃! (xr yr : ℝ), yr = (xr + 1)^2 ∧ xr * yr + yr = 1) ∧
  (∃ (xc1 yc1 xc2 yc2 : ℂ), 
    (xc1 ≠ xc2) ∧
    (yc1 = (xc1 + 1)^2 ∧ xc1 * yc1 + yc1 = 1) ∧
    (yc2 = (xc2 + 1)^2 ∧ xc2 * yc2 + yc2 = 1) ∧
    (xc1.im ≠ 0 ∧ xc2.im ≠ 0)) :=
by sorry

end solution_count_of_system_l2658_265810


namespace quadratic_coefficient_l2658_265828

theorem quadratic_coefficient (b : ℝ) (n : ℝ) : 
  (∀ x, x^2 + b*x + 56 = (x + n)^2 + 12) → 
  b > 0 → 
  b = 4 * Real.sqrt 11 := by
sorry

end quadratic_coefficient_l2658_265828


namespace binary_to_decimal_1010101_l2658_265860

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number -/
def binary_num : List Nat := [1, 0, 1, 0, 1, 0, 1]

theorem binary_to_decimal_1010101 :
  binary_to_decimal (binary_num.reverse) = 85 := by
  sorry

end binary_to_decimal_1010101_l2658_265860


namespace van_rental_equation_l2658_265836

theorem van_rental_equation (x : ℕ) (h : x > 0) :
  (180 : ℝ) / x - 180 / (x + 2) = 3 ↔
  (∃ (y : ℝ), y > 0 ∧ 180 / x = y ∧ 180 / (x + 2) = y - 3) :=
by sorry

end van_rental_equation_l2658_265836


namespace prime_quadruples_sum_882_l2658_265809

theorem prime_quadruples_sum_882 :
  ∀ p₁ p₂ p₃ p₄ : ℕ,
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ →
    p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ →
    p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882 →
    ((p₁ = 2 ∧ p₂ = 5 ∧ p₃ = 19 ∧ p₄ = 37) ∨
     (p₁ = 2 ∧ p₂ = 11 ∧ p₃ = 19 ∧ p₄ = 31) ∨
     (p₁ = 2 ∧ p₂ = 13 ∧ p₃ = 19 ∧ p₄ = 29)) :=
by sorry

end prime_quadruples_sum_882_l2658_265809


namespace interior_triangle_area_l2658_265850

theorem interior_triangle_area (a b c : ℝ) (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100) :
  (1/2) * a * b = 24 := by
  sorry

end interior_triangle_area_l2658_265850


namespace vertical_line_not_conic_section_l2658_265829

/-- The equation |y-3| = √((x+4)² + (y-3)²) describes a vertical line x = -4 -/
theorem vertical_line_not_conic_section :
  ∀ x y : ℝ, |y - 3| = Real.sqrt ((x + 4)^2 + (y - 3)^2) ↔ x = -4 :=
by sorry

end vertical_line_not_conic_section_l2658_265829


namespace total_different_books_l2658_265890

/-- The number of different books read by three people given their individual book counts and shared book information. -/
def differentBooksRead (tonyBooks deanBooks breannaBooks tonyDeanShared allShared : ℕ) : ℕ :=
  tonyBooks + deanBooks + breannaBooks - tonyDeanShared - 2 * allShared

/-- Theorem stating that Tony, Dean, and Breanna read 47 different books in total. -/
theorem total_different_books : 
  differentBooksRead 23 12 17 3 1 = 47 := by
  sorry

end total_different_books_l2658_265890


namespace hemisphere_to_spheres_l2658_265868

/-- The radius of a sphere when a hemisphere is divided into equal parts -/
theorem hemisphere_to_spheres (r : Real) (n : Nat) (r_small : Real) : 
  r = 2 → n = 18 → (2/3 * π * r^3) = (n * (4/3 * π * r_small^3)) → r_small = (2/3)^(1/3) := by
  sorry

end hemisphere_to_spheres_l2658_265868


namespace abs_four_implies_plus_minus_four_l2658_265885

theorem abs_four_implies_plus_minus_four (x : ℝ) : |x| = 4 → x = 4 ∨ x = -4 := by
  sorry

end abs_four_implies_plus_minus_four_l2658_265885


namespace rectangular_solid_surface_area_l2658_265864

/-- A rectangular solid with prime edge lengths and volume 231 has surface area 262 -/
theorem rectangular_solid_surface_area : ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 231 →
  2 * (a * b + b * c + a * c) = 262 := by
sorry

end rectangular_solid_surface_area_l2658_265864


namespace this_year_cabbage_production_l2658_265886

/-- Represents a square garden where cabbages are grown -/
structure CabbageGarden where
  side : ℕ -- Side length of the square garden

/-- Calculates the number of cabbages in a square garden -/
def cabbageCount (garden : CabbageGarden) : ℕ := garden.side ^ 2

/-- Theorem stating the number of cabbages produced this year -/
theorem this_year_cabbage_production 
  (last_year : CabbageGarden) 
  (this_year : CabbageGarden) 
  (h1 : cabbageCount this_year - cabbageCount last_year = 211) :
  cabbageCount this_year = 11236 := by
  sorry


end this_year_cabbage_production_l2658_265886


namespace no_real_solutions_l2658_265856

theorem no_real_solutions : ∀ s : ℝ, s ≠ 2 → (s^2 - 5*s - 10) / (s - 2) ≠ 3*s + 6 := by
  sorry

end no_real_solutions_l2658_265856


namespace congruent_triangles_side_lengths_isosceles_triangle_side_lengths_l2658_265884

-- Part 1
theorem congruent_triangles_side_lengths (m n : ℝ) :
  (6 :: 8 :: 10 :: []).toFinset = (6 :: (2*m-2) :: (n+1) :: []).toFinset →
  ((m = 5 ∧ n = 9) ∨ (m = 6 ∧ n = 7)) := by sorry

-- Part 2
theorem isosceles_triangle_side_lengths (a b : ℝ) :
  (a = b ∧ a + a + 5 = 16) →
  ((a = 5 ∧ b = 6) ∨ (a = 5.5 ∧ b = 5)) := by sorry

end congruent_triangles_side_lengths_isosceles_triangle_side_lengths_l2658_265884


namespace problem_solution_l2658_265882

-- Define the proposition for the first statement
def converse_square_sum_zero (x y : ℝ) : Prop :=
  x = 0 ∧ y = 0 → x^2 + y^2 = 0

-- Define the proposition for the second statement
def intersection_subset (A B : Set α) : Prop :=
  A ∩ B = A → A ⊆ B

-- Theorem combining both propositions
theorem problem_solution :
  (∀ x y : ℝ, converse_square_sum_zero x y) ∧
  (∀ A B : Set α, intersection_subset A B) :=
by
  sorry


end problem_solution_l2658_265882


namespace sum_ratio_equality_l2658_265802

theorem sum_ratio_equality (a b c x y z : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 25)
  (h2 : x^2 + y^2 + z^2 = 36)
  (h3 : a*x + b*y + c*z = 30) :
  (a + b + c) / (x + y + z) = 5/6 := by
sorry

end sum_ratio_equality_l2658_265802


namespace derivative_sin_cos_l2658_265819

theorem derivative_sin_cos (x : ℝ) :
  deriv (λ x => 3 * Real.sin x - 4 * Real.cos x) x = 3 * Real.cos x + 4 * Real.sin x := by
  sorry

end derivative_sin_cos_l2658_265819


namespace extremum_values_l2658_265887

/-- The function f(x) = x^3 - ax^2 - bx + a^2 has an extremum of 10 at x = 1 -/
def has_extremum (a b : ℝ) : Prop :=
  let f := fun x => x^3 - a*x^2 - b*x + a^2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1) ∨
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1)

/-- The main theorem -/
theorem extremum_values (a b : ℝ) :
  has_extremum a b ∧ (1^3 - a*1^2 - b*1 + a^2 = 10) →
  (a = 3 ∧ b = -3) ∨ (a = -4 ∧ b = 11) := by sorry


end extremum_values_l2658_265887


namespace highway_length_l2658_265897

/-- The length of a highway where two cars meet --/
theorem highway_length (v1 v2 t : ℝ) (h1 : v1 = 25) (h2 : v2 = 45) (h3 : t = 2.5) :
  (v1 + v2) * t = 175 := by
  sorry

end highway_length_l2658_265897


namespace max_correct_answers_l2658_265861

/-- Represents the scoring system and results of a math contest. -/
structure MathContest where
  total_questions : ℕ
  correct_points : ℤ
  blank_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Theorem stating the maximum number of correct answers for the given contest conditions. -/
theorem max_correct_answers (contest : MathContest)
  (h1 : contest.total_questions = 60)
  (h2 : contest.correct_points = 5)
  (h3 : contest.blank_points = 0)
  (h4 : contest.incorrect_points = -2)
  (h5 : contest.total_score = 139) :
  ∃ (max_correct : ℕ), max_correct = 37 ∧
  ∀ (correct : ℕ), correct ≤ contest.total_questions →
    (∃ (blank incorrect : ℕ),
      correct + blank + incorrect = contest.total_questions ∧
      contest.correct_points * correct + contest.blank_points * blank + contest.incorrect_points * incorrect = contest.total_score) →
    correct ≤ max_correct :=
sorry

end max_correct_answers_l2658_265861


namespace sqrt_division_equality_l2658_265832

theorem sqrt_division_equality : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end sqrt_division_equality_l2658_265832


namespace roots_properties_l2658_265830

theorem roots_properties (a b : ℝ) 
  (h1 : a^2 - 6*a + 4 = 0) 
  (h2 : b^2 - 6*b + 4 = 0) 
  (h3 : a > b) : 
  (a > 0 ∧ b > 0) ∧ 
  (((Real.sqrt a - Real.sqrt b) / (Real.sqrt a + Real.sqrt b)) = Real.sqrt 5 / 5) := by
  sorry

end roots_properties_l2658_265830


namespace sum_of_coefficients_l2658_265869

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, 1 + x^5 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end sum_of_coefficients_l2658_265869


namespace unequal_weight_l2658_265844

-- Define the shapes as variables
variable (square circle big_circle triangle big_triangle : ℕ)

-- Define the balance conditions
def balance1 : Prop := 4 * square = big_circle + circle
def balance2 : Prop := 2 * circle + big_circle = 2 * triangle

-- Define the weight of the original combination
def original_weight : ℕ := triangle + big_circle + square

-- Define the weight of the option to be proven unequal
def option_d_weight : ℕ := 2 * big_triangle + square

-- Theorem statement
theorem unequal_weight 
  (h1 : balance1 square circle big_circle)
  (h2 : balance2 circle big_circle triangle)
  (h3 : big_triangle = triangle) :
  option_d_weight square big_triangle ≠ original_weight triangle big_circle square :=
sorry

end unequal_weight_l2658_265844


namespace davids_english_marks_l2658_265803

def marks_math : ℕ := 65
def marks_physics : ℕ := 82
def marks_chemistry : ℕ := 67
def marks_biology : ℕ := 85
def average_marks : ℕ := 72
def num_subjects : ℕ := 5

theorem davids_english_marks :
  ∃ (marks_english : ℕ),
    (marks_english + marks_math + marks_physics + marks_chemistry + marks_biology) / num_subjects = average_marks ∧
    marks_english = 61 := by
  sorry

end davids_english_marks_l2658_265803


namespace particular_number_problem_l2658_265827

theorem particular_number_problem (x : ℝ) :
  4 * (x - 220) = 320 → (5 * x) / 3 = 500 := by
  sorry

end particular_number_problem_l2658_265827


namespace main_tire_mileage_approx_l2658_265817

/-- Represents the mileage distribution of a car's tires -/
structure CarTires where
  total_miles : ℕ
  num_main_tires : ℕ
  num_spare_tires : ℕ
  spare_multiplier : ℕ

/-- Calculates the mileage for each main tire -/
def main_tire_mileage (c : CarTires) : ℚ :=
  c.total_miles / (c.num_main_tires + c.spare_multiplier * c.num_spare_tires)

/-- Theorem stating the main tire mileage for the given conditions -/
theorem main_tire_mileage_approx :
  let c : CarTires := {
    total_miles := 40000,
    num_main_tires := 4,
    num_spare_tires := 1,
    spare_multiplier := 2
  }
  ∃ ε > 0, |main_tire_mileage c - 6667| < ε :=
sorry

end main_tire_mileage_approx_l2658_265817


namespace hyperbola_equation_l2658_265804

/-- Represents a hyperbola with equation x^2/a^2 - y^2/b^2 = 1 --/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive_a : a > 0
  h_positive_b : b > 0

/-- Theorem: Given a hyperbola with an asymptote through (2, √3) and a focus at (-√7, 0),
    prove that a = 2 and b = √3 --/
theorem hyperbola_equation (h : Hyperbola)
  (h_asymptote : 2 * h.b = Real.sqrt 3 * h.a)
  (h_focus : h.a ^ 2 - h.b ^ 2 = 7) :
  h.a = 2 ∧ h.b = Real.sqrt 3 := by
  sorry

end hyperbola_equation_l2658_265804


namespace exists_int_between_sqrt2_and_sqrt17_l2658_265881

theorem exists_int_between_sqrt2_and_sqrt17 : ∃ n : ℤ, Real.sqrt 2 < n ∧ n < Real.sqrt 17 := by
  sorry

end exists_int_between_sqrt2_and_sqrt17_l2658_265881


namespace inequality_solution_positive_for_all_x_zeros_greater_than_5_2_l2658_265834

-- Define the function f
def f (k x : ℝ) : ℝ := x^2 - k*x + (2*k - 3)

-- Statement 1
theorem inequality_solution (x : ℝ) :
  f (3/2) x > 0 ↔ x < 0 ∨ x > 3/2 := by sorry

-- Statement 2
theorem positive_for_all_x (k : ℝ) :
  (∀ x, f k x > 0) ↔ 2 < k ∧ k < 6 := by sorry

-- Statement 3
theorem zeros_greater_than_5_2 (k : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 ∧ x₁ > 5/2 ∧ x₂ > 5/2) ↔
  6 < k ∧ k < 13/2 := by sorry

end inequality_solution_positive_for_all_x_zeros_greater_than_5_2_l2658_265834


namespace smallest_possible_b_l2658_265891

def is_valid_polynomial (Q : ℤ → ℤ) (b : ℕ) : Prop :=
  b > 0 ∧
  Q 0 = b ∧ Q 4 = b ∧ Q 6 = b ∧ Q 10 = b ∧
  Q 1 = -b ∧ Q 5 = -b ∧ Q 7 = -b ∧ Q 11 = -b

theorem smallest_possible_b :
  ∀ Q : ℤ → ℤ, ∀ b : ℕ,
  is_valid_polynomial Q b →
  (∀ b' : ℕ, is_valid_polynomial Q b' → b ≤ b') →
  b = 1350 :=
sorry

end smallest_possible_b_l2658_265891


namespace expression_evaluation_l2658_265837

theorem expression_evaluation (a b c : ℝ) : 
  let d := a + b + c
  2 * (a^2 * b^2 + a^2 * c^2 + a^2 * d^2 + b^2 * c^2 + b^2 * d^2 + c^2 * d^2) - 
  (a^4 + b^4 + c^4 + d^4) + 8 * a * b * c * d = 0 := by
  sorry

end expression_evaluation_l2658_265837


namespace border_area_is_198_l2658_265843

-- Define the dimensions of the photograph
def photo_height : ℕ := 12
def photo_width : ℕ := 15

-- Define the width of the border
def border_width : ℕ := 3

-- Define the area of the border
def border_area : ℕ := 
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width

-- Theorem statement
theorem border_area_is_198 : border_area = 198 := by
  sorry

end border_area_is_198_l2658_265843


namespace derivative_f_at_zero_l2658_265845

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- State the theorem
theorem derivative_f_at_zero :
  (deriv f) 0 = 2 := by sorry

end derivative_f_at_zero_l2658_265845


namespace factorization_m_squared_minus_3m_l2658_265838

theorem factorization_m_squared_minus_3m (m : ℝ) : m^2 - 3*m = m*(m - 3) := by
  sorry

end factorization_m_squared_minus_3m_l2658_265838


namespace prob_is_one_fourth_l2658_265812

/-- The number of cards -/
def n : ℕ := 72

/-- The set of card numbers -/
def S : Finset ℕ := Finset.range n

/-- The set of multiples of 6 in S -/
def A : Finset ℕ := S.filter (fun x => x % 6 = 0)

/-- The set of multiples of 8 in S -/
def B : Finset ℕ := S.filter (fun x => x % 8 = 0)

/-- The probability of selecting a card that is a multiple of 6 or 8 or both -/
def prob : ℚ := (A ∪ B).card / S.card

theorem prob_is_one_fourth : prob = 1/4 := by
  sorry

end prob_is_one_fourth_l2658_265812


namespace movie_theater_sections_l2658_265833

theorem movie_theater_sections (total_seats : ℕ) (seats_per_section : ℕ) (h1 : total_seats = 270) (h2 : seats_per_section = 30) :
  total_seats / seats_per_section = 9 := by
  sorry

end movie_theater_sections_l2658_265833


namespace system_solution_l2658_265892

theorem system_solution : ∃! (x y : ℝ), 
  (2 * Real.sqrt (2 * x + 3 * y) + Real.sqrt (5 - x - y) = 7) ∧ 
  (3 * Real.sqrt (5 - x - y) - Real.sqrt (2 * x + y - 3) = 1) ∧ 
  x = 3 ∧ y = 1 := by
sorry

end system_solution_l2658_265892


namespace inequality_proof_l2658_265801

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a * b + b * c + c * a = 1) : 
  (((1 / a + 6 * b) ^ (1/3 : ℝ)) + 
   ((1 / b + 6 * c) ^ (1/3 : ℝ)) + 
   ((1 / c + 6 * a) ^ (1/3 : ℝ))) ≤ 1 / (a * b * c) :=
by sorry

end inequality_proof_l2658_265801


namespace min_black_edges_on_border_l2658_265851

/-- Represents a small square in the grid -/
structure SmallSquare where
  blackTriangles : Fin 4
  blackEdges : Fin 4

/-- Represents the 5x5 grid -/
def Grid := Matrix (Fin 5) (Fin 5) SmallSquare

/-- Checks if two adjacent small squares have consistent edge colors -/
def consistentEdges (s1 s2 : SmallSquare) : Prop :=
  s1.blackEdges = s2.blackEdges

/-- Counts the number of black edges on the border of the grid -/
def countBorderBlackEdges (g : Grid) : ℕ :=
  sorry

/-- The main theorem stating the minimum number of black edges on the border -/
theorem min_black_edges_on_border (g : Grid) 
  (h1 : ∀ (i j : Fin 5), (g i j).blackTriangles = 3)
  (h2 : ∀ (i j k l : Fin 5), (j = k + 1 ∨ i = l + 1) → consistentEdges (g i j) (g k l)) :
  countBorderBlackEdges g ≥ 16 :=
sorry

end min_black_edges_on_border_l2658_265851


namespace largest_two_digit_prime_factor_of_150_choose_75_l2658_265831

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem largest_two_digit_prime_factor_of_150_choose_75 :
  ∃ (p : ℕ), p = 47 ∧ 
    Prime p ∧ 
    10 ≤ p ∧ p < 100 ∧
    p ∣ binomial 150 75 ∧
    ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ binomial 150 75 → q ≤ p :=
by sorry

end largest_two_digit_prime_factor_of_150_choose_75_l2658_265831


namespace focus_of_standard_parabola_l2658_265853

/-- The focus of the parabola y = x^2 is at the point (0, 1/4). -/
theorem focus_of_standard_parabola :
  let f : ℝ × ℝ := (0, 1/4)
  let parabola := {(x, y) : ℝ × ℝ | y = x^2}
  f ∈ parabola ∧ ∀ p ∈ parabola, dist p f = dist p (0, -1/4) :=
by sorry

end focus_of_standard_parabola_l2658_265853


namespace selection_ways_l2658_265873

theorem selection_ways (male_count female_count : ℕ) 
  (h1 : male_count = 5)
  (h2 : female_count = 4) :
  male_count + female_count = 9 := by
  sorry

end selection_ways_l2658_265873


namespace factorial_simplification_l2658_265895

theorem factorial_simplification (N : ℕ) :
  (Nat.factorial (N + 1)) / ((Nat.factorial (N - 1)) * (N + 2)) = N * (N + 1) / (N + 2) := by
  sorry

end factorial_simplification_l2658_265895


namespace spherical_rotation_l2658_265899

/-- Given a point with rectangular coordinates (-3, 2, 5) and corresponding 
    spherical coordinates (r, θ, φ), the point with spherical coordinates 
    (r, θ, φ-π/2) has rectangular coordinates (-3, 2, -5). -/
theorem spherical_rotation (r θ φ : Real) : 
  r * Real.sin φ * Real.cos θ = -3 ∧ 
  r * Real.sin φ * Real.sin θ = 2 ∧ 
  r * Real.cos φ = 5 → 
  r * Real.sin (φ - π/2) * Real.cos θ = -3 ∧
  r * Real.sin (φ - π/2) * Real.sin θ = 2 ∧
  r * Real.cos (φ - π/2) = -5 := by
sorry

end spherical_rotation_l2658_265899


namespace problem_statement_l2658_265859

theorem problem_statement (a b : ℝ) (h : 5 * a - 3 * b + 2 = 0) : 
  10 * a - 6 * b - 3 = -7 := by
sorry

end problem_statement_l2658_265859


namespace max_revenue_is_70_l2658_265883

/-- Represents the advertising problem for a company --/
structure AdvertisingProblem where
  maxTime : ℝ
  maxCost : ℝ
  rateA : ℝ
  rateB : ℝ
  revenueA : ℝ
  revenueB : ℝ

/-- Calculates the maximum revenue for the given advertising problem --/
def maxRevenue (p : AdvertisingProblem) : ℝ :=
  let x := 100  -- Time for TV Station A
  let y := 200  -- Time for TV Station B
  p.revenueA * x + p.revenueB * y

/-- Theorem stating that the maximum revenue is 70 million yuan --/
theorem max_revenue_is_70 (p : AdvertisingProblem) 
    (h1 : p.maxTime = 300)
    (h2 : p.maxCost = 900000)
    (h3 : p.rateA = 500)
    (h4 : p.rateB = 200)
    (h5 : p.revenueA = 0.3)
    (h6 : p.revenueB = 0.2) :
  maxRevenue p = 70 := by
  sorry

#eval maxRevenue { maxTime := 300, maxCost := 900000, rateA := 500, rateB := 200, revenueA := 0.3, revenueB := 0.2 }

end max_revenue_is_70_l2658_265883


namespace silver_coins_removed_l2658_265847

theorem silver_coins_removed (total_coins : ℕ) (initial_gold_percent : ℚ) (final_gold_percent : ℚ) :
  total_coins = 200 →
  initial_gold_percent = 2 / 100 →
  final_gold_percent = 20 / 100 →
  (total_coins : ℚ) * initial_gold_percent = (total_coins - (total_coins : ℚ) * initial_gold_percent * (1 / final_gold_percent - 1)) * final_gold_percent →
  ⌊total_coins - (total_coins : ℚ) * initial_gold_percent * (1 / final_gold_percent)⌋ = 180 :=
by sorry

end silver_coins_removed_l2658_265847


namespace result_2011th_operation_l2658_265815

/-- Represents the sequence of operations starting with 25 -/
def operationSequence : ℕ → ℕ
| 0 => 25
| 1 => 133
| 2 => 55
| 3 => 250
| (n + 4) => operationSequence n

/-- The result of the nth operation in the sequence -/
def nthOperationResult (n : ℕ) : ℕ := operationSequence (n % 4)

theorem result_2011th_operation :
  nthOperationResult 2011 = 133 := by sorry

end result_2011th_operation_l2658_265815


namespace museum_clock_position_l2658_265800

/-- A special clock with the given properties -/
structure SpecialClock where
  positions : ℕ
  jump_interval : ℕ
  jump_distance : ℕ

/-- Calculate the position of the clock hand after a given number of minutes -/
def clock_position (clock : SpecialClock) (initial_position : ℕ) (minutes : ℕ) : ℕ :=
  (initial_position + (minutes / clock.jump_interval) * clock.jump_distance) % clock.positions

theorem museum_clock_position : 
  let clock := SpecialClock.mk 20 7 9
  let minutes_between_8pm_and_8am := 12 * 60
  clock_position clock 9 minutes_between_8pm_and_8am = 2 := by
  sorry

end museum_clock_position_l2658_265800


namespace brendans_dad_fish_count_l2658_265894

theorem brendans_dad_fish_count :
  ∀ (morning afternoon thrown_back total dad_catch : ℕ),
    morning = 8 →
    afternoon = 5 →
    thrown_back = 3 →
    total = 23 →
    dad_catch = total - (morning + afternoon - thrown_back) →
    dad_catch = 13 := by
  sorry

end brendans_dad_fish_count_l2658_265894


namespace glass_volume_l2658_265874

theorem glass_volume (V : ℝ) 
  (h1 : 0.4 * V = V - 0.6 * V) -- pessimist's glass is 60% empty
  (h2 : 0.6 * V - 0.4 * V = 46) -- difference in water volume is 46 ml
  : V = 230 := by
  sorry

end glass_volume_l2658_265874


namespace intersection_range_l2658_265857

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 + 1 ≥ Real.sqrt (2 * (p.1^2 + p.2^2))}
def N (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | |p.1 - a| + |p.2 - 1| ≤ 1}

-- State the theorem
theorem intersection_range (a : ℝ) :
  (M ∩ N a).Nonempty ↔ a ∈ Set.Icc (1 - Real.sqrt 6) (3 + Real.sqrt 10) :=
sorry

end intersection_range_l2658_265857


namespace power_sum_equals_w_minus_one_l2658_265867

theorem power_sum_equals_w_minus_one (w : ℂ) (hw : w^2 - w + 1 = 0) : 
  w^98 + w^99 + w^100 + w^101 + w^102 = w - 1 := by
  sorry

end power_sum_equals_w_minus_one_l2658_265867


namespace consecutive_20_divisibility_l2658_265870

theorem consecutive_20_divisibility (n : ℤ) : 
  (∃ k ∈ Finset.range 20, (n + k) % 9 = 0) ∧ 
  (∃ k ∈ Finset.range 20, (n + k) % 9 ≠ 0) := by
  sorry

end consecutive_20_divisibility_l2658_265870


namespace sin_2alpha_value_l2658_265813

theorem sin_2alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.tan (π / 4 - α) = 1 / 3) : 
  Real.sin (2 * α) = 4 / 5 := by
  sorry

end sin_2alpha_value_l2658_265813
