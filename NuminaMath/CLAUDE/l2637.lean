import Mathlib

namespace NUMINAMATH_CALUDE_cube_edge_length_l2637_263790

-- Define the cube
structure Cube where
  edge_length : ℝ
  sum_of_edges : ℝ

-- State the theorem
theorem cube_edge_length (c : Cube) (h : c.sum_of_edges = 108) : c.edge_length = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l2637_263790


namespace NUMINAMATH_CALUDE_bee_speed_difference_l2637_263771

/-- Proves the difference in bee's speed between two flight segments -/
theorem bee_speed_difference (time_daisy_rose time_rose_poppy : ℝ)
  (distance_difference : ℝ) (speed_daisy_rose : ℝ)
  (h1 : time_daisy_rose = 10)
  (h2 : time_rose_poppy = 6)
  (h3 : distance_difference = 8)
  (h4 : speed_daisy_rose = 2.6) :
  speed_daisy_rose * time_daisy_rose - distance_difference = 
  (speed_daisy_rose + 0.4) * time_rose_poppy := by
  sorry

end NUMINAMATH_CALUDE_bee_speed_difference_l2637_263771


namespace NUMINAMATH_CALUDE_prob_independent_of_trials_l2637_263730

/-- A random event. -/
structure RandomEvent where
  /-- The probability of the event occurring in a single trial. -/
  probability : ℝ
  /-- Assumption that the probability is between 0 and 1. -/
  prob_nonneg : 0 ≤ probability
  prob_le_one : probability ≤ 1

/-- The probability of the event not occurring in n trials. -/
def prob_not_occur (E : RandomEvent) (n : ℕ) : ℝ :=
  (1 - E.probability) ^ n

/-- The probability of the event occurring at least once in n trials. -/
def prob_occur_at_least_once (E : RandomEvent) (n : ℕ) : ℝ :=
  1 - prob_not_occur E n

/-- Theorem stating that the probability of a random event occurring
    is independent of the number of trials. -/
theorem prob_independent_of_trials (E : RandomEvent) :
  ∀ n : ℕ, prob_occur_at_least_once E (n + 1) - prob_occur_at_least_once E n = E.probability * (prob_not_occur E n) :=
sorry


end NUMINAMATH_CALUDE_prob_independent_of_trials_l2637_263730


namespace NUMINAMATH_CALUDE_trajectory_is_line_segment_l2637_263743

/-- The trajectory of a point P, where the sum of its distances to two fixed points A(-1, 0) and B(1, 0) is constant 2, is the line segment AB. -/
theorem trajectory_is_line_segment (P : ℝ × ℝ) : 
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let dist (X Y : ℝ × ℝ) := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  (dist P A + dist P B = 2) → 
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (2*t - 1, 0) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_is_line_segment_l2637_263743


namespace NUMINAMATH_CALUDE_complex_magnitude_l2637_263757

theorem complex_magnitude (z : ℂ) (h : z * Complex.I = 1 - 2 * Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2637_263757


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l2637_263785

/-- The standard equation of an ellipse given its parametric form -/
theorem ellipse_standard_equation (x y α : ℝ) :
  (x = 5 * Real.cos α) ∧ (y = 3 * Real.sin α) →
  (x^2 / 25 + y^2 / 9 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l2637_263785


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l2637_263751

/-- Represents the number of employees to be sampled from each category -/
structure StratifiedSample where
  business : ℕ
  management : ℕ
  logistics : ℕ

/-- Calculates the stratified sample given total employees and sample size -/
def calculateStratifiedSample (totalEmployees business management logistics sampleSize : ℕ) : StratifiedSample :=
  { business := (business * sampleSize) / totalEmployees,
    management := (management * sampleSize) / totalEmployees,
    logistics := (logistics * sampleSize) / totalEmployees }

theorem correct_stratified_sample :
  let totalEmployees : ℕ := 160
  let business : ℕ := 120
  let management : ℕ := 16
  let logistics : ℕ := 24
  let sampleSize : ℕ := 20
  let sample := calculateStratifiedSample totalEmployees business management logistics sampleSize
  sample.business = 15 ∧ sample.management = 2 ∧ sample.logistics = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l2637_263751


namespace NUMINAMATH_CALUDE_two_bishops_placement_l2637_263796

/-- Represents a chessboard with 8 rows and 8 columns -/
structure Chessboard :=
  (rows : Nat)
  (columns : Nat)
  (total_squares : Nat)
  (white_squares : Nat)
  (black_squares : Nat)

/-- Represents the number of ways to place two bishops on a chessboard -/
def bishop_placements (board : Chessboard) : Nat :=
  board.white_squares * (board.black_squares - board.rows)

/-- Theorem stating the number of ways to place two bishops on a chessboard -/
theorem two_bishops_placement (board : Chessboard) 
  (h1 : board.rows = 8)
  (h2 : board.columns = 8)
  (h3 : board.total_squares = board.rows * board.columns)
  (h4 : board.white_squares = board.total_squares / 2)
  (h5 : board.black_squares = board.total_squares / 2) :
  bishop_placements board = 768 := by
  sorry

#eval bishop_placements {rows := 8, columns := 8, total_squares := 64, white_squares := 32, black_squares := 32}

end NUMINAMATH_CALUDE_two_bishops_placement_l2637_263796


namespace NUMINAMATH_CALUDE_roots_of_equation_l2637_263780

theorem roots_of_equation (x : ℝ) : 
  (x + 2)^2 = 8 ↔ x = 2 * Real.sqrt 2 - 2 ∨ x = -2 * Real.sqrt 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2637_263780


namespace NUMINAMATH_CALUDE_stretches_per_meter_l2637_263772

/-- Given the following conversions between paces, stretches, leaps, and meters:
    p paces equals q stretches,
    r leaps equals s stretches,
    t leaps equals u meters,
    prove that the number of stretches in one meter is ts/ur. -/
theorem stretches_per_meter
  (p q r s t u : ℝ)
  (h1 : p * q⁻¹ = 1)  -- p paces equals q stretches
  (h2 : r * s⁻¹ = 1)  -- r leaps equals s stretches
  (h3 : t * u⁻¹ = 1)  -- t leaps equals u meters
  : 1 = t * s * (u * r)⁻¹ :=
sorry

end NUMINAMATH_CALUDE_stretches_per_meter_l2637_263772


namespace NUMINAMATH_CALUDE_min_value_problem_l2637_263766

/-- Given positive real numbers a, b, c, and a function f with minimum value 4, 
    prove the sum of a, b, c is 4 and find the minimum value of a quadratic expression. -/
theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hf : ∀ x, |x + a| + |x - b| + c ≥ 4) : 
  (a + b + c = 4) ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 4 → 
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l2637_263766


namespace NUMINAMATH_CALUDE_exists_cheaper_bulk_purchase_l2637_263776

/-- The original price of a notebook --/
def original_price : ℝ := 8

/-- The discounted price of a notebook when buying more than 100 --/
def discounted_price : ℝ := original_price - 2

/-- The cost of buying n books under Plan 1 (n ≤ 100) --/
def cost_plan1 (n : ℝ) : ℝ := original_price * n

/-- The cost of buying n books under Plan 2 (n > 100) --/
def cost_plan2 (n : ℝ) : ℝ := discounted_price * n

/-- Theorem stating that there exists a scenario where buying n books (n > 100) 
    costs less than buying 80 books under Plan 1 --/
theorem exists_cheaper_bulk_purchase :
  ∃ n : ℝ, n > 100 ∧ cost_plan2 n < cost_plan1 80 := by
  sorry

end NUMINAMATH_CALUDE_exists_cheaper_bulk_purchase_l2637_263776


namespace NUMINAMATH_CALUDE_a1_range_for_three_greater_terms_l2637_263723

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

theorem a1_range_for_three_greater_terms
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : geometric_sequence a (1/2))
  (b : ℕ → ℝ)
  (h_b : ∀ n, b n = n / 2)
  (h_three : ∃! (s : Finset ℕ),
    s.card = 3 ∧ (∀ n ∈ s, a n > b n) ∧ (∀ n ∉ s, a n ≤ b n)) :
  6 < a 1 ∧ a 1 ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_a1_range_for_three_greater_terms_l2637_263723


namespace NUMINAMATH_CALUDE_square_root_equation_solutions_cube_root_equation_solution_l2637_263763

theorem square_root_equation_solutions (x : ℝ) :
  (x - 1)^2 = 4 ↔ x = 3 ∨ x = -1 := by sorry

theorem cube_root_equation_solution (x : ℝ) :
  (x - 2)^3 = -125 ↔ x = -3 := by sorry

end NUMINAMATH_CALUDE_square_root_equation_solutions_cube_root_equation_solution_l2637_263763


namespace NUMINAMATH_CALUDE_stella_unpaid_leave_l2637_263747

/-- Calculates the number of months of unpaid leave taken by an employee given their monthly income and actual annual income. -/
def unpaid_leave_months (monthly_income : ℕ) (actual_annual_income : ℕ) : ℕ :=
  12 - actual_annual_income / monthly_income

/-- Proves that given Stella's monthly income of 4919 dollars and her actual annual income of 49190 dollars, the number of months of unpaid leave she took is 2. -/
theorem stella_unpaid_leave :
  unpaid_leave_months 4919 49190 = 2 := by
  sorry

end NUMINAMATH_CALUDE_stella_unpaid_leave_l2637_263747


namespace NUMINAMATH_CALUDE_jerry_vote_difference_l2637_263728

def total_votes : ℕ := 196554
def jerry_votes : ℕ := 108375

theorem jerry_vote_difference : 
  jerry_votes - (total_votes - jerry_votes) = 20196 := by
  sorry

end NUMINAMATH_CALUDE_jerry_vote_difference_l2637_263728


namespace NUMINAMATH_CALUDE_binary_rep_156_ones_minus_zeros_eq_zero_l2637_263781

/-- Converts a natural number to its binary representation as a list of booleans -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Counts the number of true values in a list of booleans -/
def countOnes (l : List Bool) : ℕ :=
  l.filter id |>.length

/-- Counts the number of false values in a list of booleans -/
def countZeros (l : List Bool) : ℕ :=
  l.filter not |>.length

theorem binary_rep_156_ones_minus_zeros_eq_zero :
  let binary := toBinary 156
  let y := countOnes binary
  let x := countZeros binary
  y - x = 0 := by sorry

end NUMINAMATH_CALUDE_binary_rep_156_ones_minus_zeros_eq_zero_l2637_263781


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l2637_263753

theorem sum_of_reciprocal_roots (γ δ : ℝ) : 
  (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ 
   6 * c^2 + 5 * c + 7 = 0 ∧ 
   6 * d^2 + 5 * d + 7 = 0 ∧ 
   γ = 1 / c ∧ 
   δ = 1 / d) → 
  γ + δ = -5 / 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l2637_263753


namespace NUMINAMATH_CALUDE_solution_set_equals_plus_minus_one_l2637_263787

def solution_set : Set ℝ := {x | x^2 - 1 = 0}

theorem solution_set_equals_plus_minus_one : solution_set = {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equals_plus_minus_one_l2637_263787


namespace NUMINAMATH_CALUDE_count_negative_numbers_l2637_263729

def number_list : List ℝ := [0, -2, 3, -0.1, -(-5)]

theorem count_negative_numbers : 
  (number_list.filter (λ x => x < 0)).length = 2 := by sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l2637_263729


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l2637_263734

theorem trigonometric_expression_equality (x : ℝ) (h : Real.tan x = 2) :
  (2 * Real.sin (Real.pi + x) * Real.cos (Real.pi - x) - Real.cos (Real.pi + x)) /
  (1 + Real.sin x ^ 2 + Real.sin (Real.pi - x) - Real.cos (Real.pi - x) ^ 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l2637_263734


namespace NUMINAMATH_CALUDE_sequence_strictly_decreasing_l2637_263739

/-- Given real numbers a and b with b > a > 1, prove that the sequence x_n is strictly monotonically decreasing -/
theorem sequence_strictly_decreasing (a b : ℝ) (h1 : a > 1) (h2 : b > a) : 
  ∀ n : ℕ, (2^n * (b^(1/2^n) - a^(1/2^n))) > (2^(n+1) * (b^(1/2^(n+1)) - a^(1/2^(n+1)))) := by
  sorry

#check sequence_strictly_decreasing

end NUMINAMATH_CALUDE_sequence_strictly_decreasing_l2637_263739


namespace NUMINAMATH_CALUDE_constant_distance_l2637_263750

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope 1/2 and y-intercept m -/
structure Line where
  m : ℝ

/-- Theorem stating the constant distance between points B and N -/
theorem constant_distance (E : Ellipse) (l : Line) : 
  E.a^2 * (1 / E.b^2 - 1 / E.a^2) = 3 / 4 →  -- eccentricity condition
  E.b = 1 →  -- passes through (0, 1)
  ∃ (A C : ℝ × ℝ), 
    (A.1^2 / E.a^2 + A.2^2 / E.b^2 = 1) ∧  -- A is on the ellipse
    (C.1^2 / E.a^2 + C.2^2 / E.b^2 = 1) ∧  -- C is on the ellipse
    (A.2 = A.1 / 2 + l.m) ∧  -- A is on the line
    (C.2 = C.1 / 2 + l.m) ∧  -- C is on the line
    ∃ (B D : ℝ × ℝ), 
      (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 ∧  -- ABCD is a square
      (D.1 - A.1)^2 + (D.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 ∧
      (B.1 - C.1)^2 + (B.2 - C.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 ∧
      (B.1 - 2 * l.m)^2 + B.2^2 = 5 / 2  -- distance between B and N is √(5/2)
  := by sorry

end NUMINAMATH_CALUDE_constant_distance_l2637_263750


namespace NUMINAMATH_CALUDE_two_numbers_with_specific_means_l2637_263754

theorem two_numbers_with_specific_means : ∃ a b : ℝ, 
  a > 0 ∧ b > 0 ∧ 
  Real.sqrt (a * b) = Real.sqrt 5 ∧
  2 / (1 / a + 1 / b) = 2 ∧
  a = (5 + Real.sqrt 5) / 2 ∧ 
  b = (5 - Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_with_specific_means_l2637_263754


namespace NUMINAMATH_CALUDE_fruit_basket_total_cost_l2637_263759

def fruit_basket_cost (banana_count : ℕ) (apple_count : ℕ) (strawberry_count : ℕ) (avocado_count : ℕ) (grape_bunch_count : ℕ) : ℕ :=
  let banana_price := 1
  let apple_price := 2
  let strawberry_price_per_12 := 4
  let avocado_price := 3
  let grape_half_bunch_price := 2

  banana_count * banana_price +
  apple_count * apple_price +
  (strawberry_count / 12) * strawberry_price_per_12 +
  avocado_count * avocado_price +
  grape_bunch_count * (2 * grape_half_bunch_price)

theorem fruit_basket_total_cost :
  fruit_basket_cost 4 3 24 2 1 = 28 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_total_cost_l2637_263759


namespace NUMINAMATH_CALUDE_polynomial_inequality_l2637_263727

theorem polynomial_inequality (x : ℝ) : x * (x + 1) * (x + 2) * (x + 3) ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l2637_263727


namespace NUMINAMATH_CALUDE_polynomial_degree_condition_l2637_263702

theorem polynomial_degree_condition (k m : ℝ) : 
  (∀ x, ∃ a b, (k - 1) * x^2 + 4 * x - m = a * x + b) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_condition_l2637_263702


namespace NUMINAMATH_CALUDE_dan_onions_l2637_263700

/-- The number of onions grown by Nancy, Dan, and Mike -/
structure OnionGrowth where
  nancy : ℕ
  dan : ℕ
  mike : ℕ

/-- The total number of onions grown -/
def total_onions (g : OnionGrowth) : ℕ :=
  g.nancy + g.dan + g.mike

/-- Theorem: Dan grew 9 onions -/
theorem dan_onions :
  ∀ g : OnionGrowth,
    g.nancy = 2 →
    g.mike = 4 →
    total_onions g = 15 →
    g.dan = 9 := by
  sorry

end NUMINAMATH_CALUDE_dan_onions_l2637_263700


namespace NUMINAMATH_CALUDE_chess_group_players_l2637_263783

theorem chess_group_players (n : ℕ) : n * (n - 1) / 2 = 1225 → n = 50 := by
  sorry

end NUMINAMATH_CALUDE_chess_group_players_l2637_263783


namespace NUMINAMATH_CALUDE_tangent_line_at_one_a_lower_bound_local_max_inequality_l2637_263755

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - 2 * a * x

def g (a : ℝ) (x : ℝ) : ℝ := f a x + (1/2) * x^2

def hasTangentLine (f : ℝ → ℝ) (x₀ y₀ m : ℝ) : Prop :=
  ∀ x, f x = m * (x - x₀) + y₀

def hasLocalMax (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x ≤ f x₀

theorem tangent_line_at_one (a : ℝ) (h : a = 2) :
  hasTangentLine (f a) 1 (-4) (-3) :=
sorry

theorem a_lower_bound (a : ℝ) (h : ∀ x > 0, f a x ≤ 2) :
  a ≥ 1 / (2 * Real.exp 3) :=
sorry

theorem local_max_inequality (a : ℝ) (x₀ : ℝ) (h : hasLocalMax (g a) x₀) :
  x₀ * f a x₀ + 1 + a * x₀^2 > 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_a_lower_bound_local_max_inequality_l2637_263755


namespace NUMINAMATH_CALUDE_least_coins_in_purse_l2637_263724

theorem least_coins_in_purse (n : ℕ) : 
  (n % 7 = 3 ∧ n % 5 = 4) → n ≥ 24 :=
sorry

end NUMINAMATH_CALUDE_least_coins_in_purse_l2637_263724


namespace NUMINAMATH_CALUDE_full_house_count_l2637_263740

theorem full_house_count :
  let n_values : ℕ := 13
  let cards_per_value : ℕ := 4
  let full_house_count := n_values * (n_values - 1) * (cards_per_value.choose 3) * (cards_per_value.choose 2)
  full_house_count = 3744 :=
by sorry

end NUMINAMATH_CALUDE_full_house_count_l2637_263740


namespace NUMINAMATH_CALUDE_sum_of_mobile_keypad_numbers_l2637_263717

def mobile_keypad : List Nat := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem sum_of_mobile_keypad_numbers : 
  mobile_keypad.sum = 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_mobile_keypad_numbers_l2637_263717


namespace NUMINAMATH_CALUDE_grain_warehouse_analysis_l2637_263778

def grain_records : List Int := [26, -32, -25, 34, -38, 10]
def current_stock : Int := 480

theorem grain_warehouse_analysis :
  (List.sum grain_records < 0) ∧
  (current_stock - List.sum grain_records = 505) := by
  sorry

end NUMINAMATH_CALUDE_grain_warehouse_analysis_l2637_263778


namespace NUMINAMATH_CALUDE_contest_probability_l2637_263715

theorem contest_probability (n : ℕ) : n = 4 ↔ n = Nat.succ (Nat.floor (Real.log 10 / Real.log 2)) := by sorry

end NUMINAMATH_CALUDE_contest_probability_l2637_263715


namespace NUMINAMATH_CALUDE_F_less_than_G_l2637_263765

theorem F_less_than_G : ∀ x : ℝ, (2 * x^2 - 3 * x - 2) < (3 * x^2 - 7 * x + 5) := by
  sorry

end NUMINAMATH_CALUDE_F_less_than_G_l2637_263765


namespace NUMINAMATH_CALUDE_simplify_sqrt_fraction_l2637_263738

theorem simplify_sqrt_fraction : 
  Real.sqrt ((25 : ℝ) / 36 + 16 / 9) = Real.sqrt 89 / 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_fraction_l2637_263738


namespace NUMINAMATH_CALUDE_kate_red_bouncy_balls_l2637_263768

/-- The number of packs of yellow bouncy balls Kate bought -/
def yellow_packs : ℕ := 6

/-- The number of bouncy balls in each pack -/
def balls_per_pack : ℕ := 18

/-- The difference in the number of red and yellow bouncy balls -/
def red_yellow_diff : ℕ := 18

/-- The number of packs of red bouncy balls Kate bought -/
def red_packs : ℕ := 7

theorem kate_red_bouncy_balls :
  red_packs * balls_per_pack = yellow_packs * balls_per_pack + red_yellow_diff :=
by sorry

end NUMINAMATH_CALUDE_kate_red_bouncy_balls_l2637_263768


namespace NUMINAMATH_CALUDE_ab_value_l2637_263774

theorem ab_value (a b : ℝ) 
  (h1 : (a + b)^2 + |b + 5| = b + 5)
  (h2 : 2 * a - b + 1 = 0) : 
  a * b = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2637_263774


namespace NUMINAMATH_CALUDE_sum_remainder_mod_17_l2637_263791

theorem sum_remainder_mod_17 : ∃ k : ℕ, (78 + 79 + 80 + 81 + 82 + 83 + 84 + 85) = 17 * k + 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_17_l2637_263791


namespace NUMINAMATH_CALUDE_place_face_value_difference_l2637_263725

def numeral : ℕ := 856973

def digit_of_interest : ℕ := 7

def place_value (n : ℕ) (d : ℕ) : ℕ :=
  (n / 10) % 10 * 10

def face_value (d : ℕ) : ℕ := d

theorem place_face_value_difference :
  place_value numeral digit_of_interest - face_value digit_of_interest = 63 := by
  sorry

end NUMINAMATH_CALUDE_place_face_value_difference_l2637_263725


namespace NUMINAMATH_CALUDE_pepper_spray_ratio_l2637_263786

theorem pepper_spray_ratio (total animals : ℕ) (raccoons : ℕ) : 
  total = 84 → raccoons = 12 → (total - raccoons) / raccoons = 6 := by
  sorry

end NUMINAMATH_CALUDE_pepper_spray_ratio_l2637_263786


namespace NUMINAMATH_CALUDE_largest_c_for_no_integer_in_interval_l2637_263795

theorem largest_c_for_no_integer_in_interval :
  ∃ (c : ℝ), c = 6 - 4 * Real.sqrt 2 ∧
  (∀ (n : ℕ), ∀ (k : ℤ),
    (n : ℝ) * Real.sqrt 2 - c / (n : ℝ) < (k : ℝ) →
    (k : ℝ) < (n : ℝ) * Real.sqrt 2 + c / (n : ℝ)) ∧
  (∀ (c' : ℝ), c' > c →
    ∃ (n : ℕ), ∃ (k : ℤ),
      (n : ℝ) * Real.sqrt 2 - c' / (n : ℝ) ≤ (k : ℝ) ∧
      (k : ℝ) ≤ (n : ℝ) * Real.sqrt 2 + c' / (n : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_no_integer_in_interval_l2637_263795


namespace NUMINAMATH_CALUDE_base_conversion_2014_to_base_9_l2637_263769

theorem base_conversion_2014_to_base_9 :
  2014 = 2 * (9^3) + 6 * (9^2) + 7 * (9^1) + 7 * (9^0) :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_2014_to_base_9_l2637_263769


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_is_22_l2637_263767

/-- The first polynomial -/
def p1 (x : ℝ) : ℝ := x^4 - 2*x^3 + 3*x^2 - 4*x + 5

/-- The second polynomial -/
def p2 (x : ℝ) : ℝ := 3*x^3 - 4*x^2 + x + 6

/-- The product of the two polynomials -/
def product (x : ℝ) : ℝ := p1 x * p2 x

/-- Theorem stating that the coefficient of x^3 in the product is 22 -/
theorem coefficient_x_cubed_is_22 : 
  ∃ (a b c d e : ℝ), product = fun x ↦ 22 * x^3 + a * x^4 + b * x^2 + c * x + d * x^5 + e :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_is_22_l2637_263767


namespace NUMINAMATH_CALUDE_expression_simplification_l2637_263716

theorem expression_simplification (a b : ℝ) (ha : a = -1) (hb : b = 2) :
  (a + b)^2 + (a^2 * b - 2 * a * b^2 - b^3) / b - (a - b) * (a + b) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2637_263716


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l2637_263741

theorem no_prime_roots_for_quadratic : ¬∃ k : ℤ, ∃ p q : ℕ,
  Prime p ∧ Prime q ∧ p + q = 95 ∧ p * q = k :=
sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l2637_263741


namespace NUMINAMATH_CALUDE_jessica_quarters_l2637_263736

theorem jessica_quarters (initial borrowed current : ℕ) : 
  borrowed = 3 → current = 5 → initial = current + borrowed :=
by sorry

end NUMINAMATH_CALUDE_jessica_quarters_l2637_263736


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2637_263731

theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  Real.sqrt ((x - 2)^2 + (7 - 1)^2) = 8 → 
  x = 2 + 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2637_263731


namespace NUMINAMATH_CALUDE_not_divisible_by_power_of_two_l2637_263719

theorem not_divisible_by_power_of_two (p : ℕ) (hp : p > 1) :
  ¬(2^p ∣ 3^p + 1) := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_power_of_two_l2637_263719


namespace NUMINAMATH_CALUDE_cone_radius_is_one_l2637_263762

/-- Given a cone whose surface area is 3π and whose lateral surface unfolds into a semicircle,
    prove that the radius of its base is 1. -/
theorem cone_radius_is_one (r : ℝ) (l : ℝ) : 
  r > 0 → l > 0 → 
  π * l = 2 * π * r →  -- lateral surface unfolds into a semicircle
  π * r^2 + π * r * l = 3 * π →  -- surface area is 3π
  r = 1 := by
  sorry

end NUMINAMATH_CALUDE_cone_radius_is_one_l2637_263762


namespace NUMINAMATH_CALUDE_shiela_neighbors_l2637_263705

theorem shiela_neighbors (total_drawings : ℕ) (drawings_per_neighbor : ℕ) 
  (h1 : total_drawings = 54)
  (h2 : drawings_per_neighbor = 9)
  : total_drawings / drawings_per_neighbor = 6 := by
  sorry

end NUMINAMATH_CALUDE_shiela_neighbors_l2637_263705


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2637_263710

def num_math_books : ℕ := 4
def num_history_books : ℕ := 6

def alternating_arrangement (m h : ℕ) : Prop :=
  m > 1 ∧ h > 0 ∧ m = h + 1

theorem book_arrangement_count :
  alternating_arrangement num_math_books num_history_books →
  (num_math_books * (num_math_books - 1) * (num_history_books.factorial / (num_history_books - (num_math_books - 1)).factorial)) = 2880 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2637_263710


namespace NUMINAMATH_CALUDE_parallelepiped_net_theorem_l2637_263749

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

/-- Represents a net formed from a parallelepiped -/
structure Net :=
  (squares : ℕ)

/-- Unfolds a parallelepiped into a net -/
def unfold (p : Parallelepiped) : Net :=
  { squares := 2 * (p.length * p.width + p.length * p.height + p.width * p.height) }

/-- Removes one square from a net -/
def remove_square (n : Net) : Net :=
  { squares := n.squares - 1 }

theorem parallelepiped_net_theorem (p : Parallelepiped) 
  (h1 : p.length = 2)
  (h2 : p.width = 1)
  (h3 : p.height = 1) :
  (remove_square (unfold p)).squares = 9 :=
sorry

end NUMINAMATH_CALUDE_parallelepiped_net_theorem_l2637_263749


namespace NUMINAMATH_CALUDE_emilys_skirt_cost_l2637_263706

theorem emilys_skirt_cost (art_supplies_cost total_cost : ℕ) (num_skirts : ℕ) 
  (h1 : art_supplies_cost = 20)
  (h2 : num_skirts = 2)
  (h3 : total_cost = 50) :
  ∃ (skirt_cost : ℕ), skirt_cost * num_skirts + art_supplies_cost = total_cost ∧ skirt_cost = 15 :=
by
  sorry

#check emilys_skirt_cost

end NUMINAMATH_CALUDE_emilys_skirt_cost_l2637_263706


namespace NUMINAMATH_CALUDE_expand_and_equate_l2637_263788

theorem expand_and_equate : 
  (∀ x : ℝ, (x - 5) * (x + 2) = x^2 + p * x + q) → p = -3 ∧ q = -10 := by
sorry

end NUMINAMATH_CALUDE_expand_and_equate_l2637_263788


namespace NUMINAMATH_CALUDE_remainder_of_M_mod_50_l2637_263704

def M : ℕ := sorry -- Definition of M as concatenation of numbers from 1 to 49

theorem remainder_of_M_mod_50 : M % 50 = 49 := by sorry

end NUMINAMATH_CALUDE_remainder_of_M_mod_50_l2637_263704


namespace NUMINAMATH_CALUDE_two_prime_roots_equation_l2637_263775

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

theorem two_prime_roots_equation (n : ℕ) (h_pos : n > 0) :
  ∃ (x₁ x₂ : ℕ), 
    is_prime x₁ ∧ 
    is_prime x₂ ∧ 
    x₁ ≠ x₂ ∧
    2 * x₁^2 - 8*n*x₁ + 10*x₁ - n^2 + 35*n - 76 = 0 ∧
    2 * x₂^2 - 8*n*x₂ + 10*x₂ - n^2 + 35*n - 76 = 0 →
  n = 3 ∧ x₁ = 2 ∧ x₂ = 5 :=
sorry

end NUMINAMATH_CALUDE_two_prime_roots_equation_l2637_263775


namespace NUMINAMATH_CALUDE_sqrt_12_div_sqrt_3_equals_2_l2637_263722

theorem sqrt_12_div_sqrt_3_equals_2 : Real.sqrt 12 / Real.sqrt 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_div_sqrt_3_equals_2_l2637_263722


namespace NUMINAMATH_CALUDE_hyperbola_b_value_l2637_263748

/-- The value of b for a hyperbola with given equation and asymptote -/
theorem hyperbola_b_value (b : ℝ) (h1 : b > 0) : 
  (∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, 3*x - 2*y = 0 ∧ x^2 / 4 - y^2 / b^2 = 1) →
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_b_value_l2637_263748


namespace NUMINAMATH_CALUDE_club_juniors_count_l2637_263756

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

end NUMINAMATH_CALUDE_club_juniors_count_l2637_263756


namespace NUMINAMATH_CALUDE_square_ratio_side_length_l2637_263760

theorem square_ratio_side_length (area_ratio : ℚ) : 
  area_ratio = 250 / 98 →
  ∃ (a b c : ℕ), 
    (a^2 * b : ℚ) / c^2 = area_ratio ∧
    a = 5 ∧ b = 5 ∧ c = 7 ∧
    a + b + c = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_l2637_263760


namespace NUMINAMATH_CALUDE_lcm_sum_triplet_l2637_263799

theorem lcm_sum_triplet (a b c : ℕ+) :
  a + b + c = Nat.lcm (Nat.lcm a.val b.val) c.val ↔ b = 2 * a ∧ c = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_lcm_sum_triplet_l2637_263799


namespace NUMINAMATH_CALUDE_cube_can_be_threaded_tetrahedron_can_be_threaded_l2637_263777

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a 2D point
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a frame (cube or tetrahedron)
structure Frame where
  vertices : List Point3D
  edges : List (Point3D × Point3D)

-- Define a hole in the plane
structure Hole where
  boundary : Point2D → Bool

-- Function to check if a hole is valid (closed and non-self-intersecting)
def isValidHole (h : Hole) : Prop :=
  sorry

-- Function to check if a frame can be threaded through a hole
def canThreadThrough (f : Frame) (h : Hole) : Prop :=
  sorry

-- Theorem for cube
theorem cube_can_be_threaded :
  ∃ (cubef : Frame) (h : Hole), isValidHole h ∧ canThreadThrough cubef h :=
sorry

-- Theorem for tetrahedron
theorem tetrahedron_can_be_threaded :
  ∃ (tetf : Frame) (h : Hole), isValidHole h ∧ canThreadThrough tetf h :=
sorry

end NUMINAMATH_CALUDE_cube_can_be_threaded_tetrahedron_can_be_threaded_l2637_263777


namespace NUMINAMATH_CALUDE_course_size_l2637_263707

theorem course_size (a b c d : ℕ) (h1 : a + b + c + d = 800) 
  (h2 : a = 800 / 5) (h3 : b = 800 / 4) (h4 : c = 800 / 2) (h5 : d = 40) : 
  800 = 800 := by sorry

end NUMINAMATH_CALUDE_course_size_l2637_263707


namespace NUMINAMATH_CALUDE_transform_F_coordinates_l2637_263744

/-- Reflection over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflection over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflection over the line y = x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

/-- The initial coordinates of point F -/
def F : ℝ × ℝ := (1, 0)

theorem transform_F_coordinates :
  (reflect_y_eq_x ∘ reflect_y ∘ reflect_x) F = (0, -1) := by
  sorry

end NUMINAMATH_CALUDE_transform_F_coordinates_l2637_263744


namespace NUMINAMATH_CALUDE_quadratic_root_arithmetic_sequence_l2637_263792

/-- Given real numbers a, b, c forming an arithmetic sequence with c ≥ b ≥ a ≥ 0,
    the single root of the quadratic cx^2 + bx + a = 0 is -1 - (√3)/3 -/
theorem quadratic_root_arithmetic_sequence (a b c : ℝ) : 
  c ≥ b ∧ b ≥ a ∧ a ≥ 0 →
  (∃ d : ℝ, b = a + d ∧ c = a + 2*d) →
  (∃! x : ℝ, c*x^2 + b*x + a = 0) →
  (∃ x : ℝ, c*x^2 + b*x + a = 0 ∧ x = -1 - Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_arithmetic_sequence_l2637_263792


namespace NUMINAMATH_CALUDE_decagon_adjacent_probability_l2637_263712

/-- A decagon is a polygon with 10 sides and vertices -/
def Decagon : Type := Unit

/-- Two vertices in a polygon are adjacent if they share an edge -/
def adjacent (v1 v2 : ℕ) (p : Decagon) : Prop := sorry

/-- The probability of an event is the number of favorable outcomes divided by the total number of possible outcomes -/
def probability (event total : ℕ) : ℚ := event / total

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem decagon_adjacent_probability :
  ∀ (d : Decagon),
  probability 
    (10 : ℕ)  -- Number of adjacent vertex pairs
    (choose_two 10)  -- Total number of vertex pairs
  = 2 / 9 := by sorry

end NUMINAMATH_CALUDE_decagon_adjacent_probability_l2637_263712


namespace NUMINAMATH_CALUDE_angle_calculation_l2637_263732

-- Define a structure for angles in degrees, minutes, and seconds
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)
  (seconds : ℕ)

-- Define multiplication of an angle by a natural number
def multiply_angle (a : Angle) (n : ℕ) : Angle :=
  sorry

-- Define division of an angle by a natural number
def divide_angle (a : Angle) (n : ℕ) : Angle :=
  sorry

-- Define addition of two angles
def add_angles (a b : Angle) : Angle :=
  sorry

-- Theorem statement
theorem angle_calculation :
  let a1 := Angle.mk 50 24 0
  let a2 := Angle.mk 98 12 25
  add_angles (multiply_angle a1 3) (divide_angle a2 5) = Angle.mk 170 50 29 :=
sorry

end NUMINAMATH_CALUDE_angle_calculation_l2637_263732


namespace NUMINAMATH_CALUDE_power_function_theorem_l2637_263733

theorem power_function_theorem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x^a) →  -- f is a power function with exponent a
  f 2 = 1/4 →         -- f passes through the point (2, 1/4)
  f (-2) = 1/4 :=     -- prove that f(-2) = 1/4
by
  sorry

end NUMINAMATH_CALUDE_power_function_theorem_l2637_263733


namespace NUMINAMATH_CALUDE_ball_probabilities_l2637_263708

/-- Represents the color of a ball -/
inductive BallColor
| Black
| White

/-- Represents the bag of balls -/
structure Bag :=
  (black : ℕ)
  (white : ℕ)

/-- The probability of drawing a white ball on the third draw with replacement -/
def prob_white_third_with_replacement (bag : Bag) : ℚ :=
  bag.white / (bag.black + bag.white)

/-- The probability of drawing a white ball only on the third draw with replacement -/
def prob_white_only_third_with_replacement (bag : Bag) : ℚ :=
  (bag.black / (bag.black + bag.white))^2 * (bag.white / (bag.black + bag.white))

/-- The probability of drawing a white ball on the third draw without replacement -/
def prob_white_third_without_replacement (bag : Bag) : ℚ :=
  (bag.white * (bag.black * (bag.black - 1) + 2 * bag.black * bag.white + bag.white * (bag.white - 1))) /
  ((bag.black + bag.white) * (bag.black + bag.white - 1) * (bag.black + bag.white - 2))

/-- The probability of drawing a white ball only on the third draw without replacement -/
def prob_white_only_third_without_replacement (bag : Bag) : ℚ :=
  (bag.black * (bag.black - 1) * bag.white) /
  ((bag.black + bag.white) * (bag.black + bag.white - 1) * (bag.black + bag.white - 2))

theorem ball_probabilities (bag : Bag) (h : bag.black = 3 ∧ bag.white = 2) :
  prob_white_third_with_replacement bag = 2/5 ∧
  prob_white_only_third_with_replacement bag = 18/125 ∧
  prob_white_third_without_replacement bag = 2/5 ∧
  prob_white_only_third_without_replacement bag = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l2637_263708


namespace NUMINAMATH_CALUDE_square_field_area_l2637_263797

/-- The area of a square field with side length 15 m is 225 square meters. -/
theorem square_field_area : 
  ∀ (side_length area : ℝ), 
  side_length = 15 → 
  area = side_length * side_length → 
  area = 225 := by
sorry

end NUMINAMATH_CALUDE_square_field_area_l2637_263797


namespace NUMINAMATH_CALUDE_total_deduction_in_cents_l2637_263764

/-- Elena's hourly wage in dollars -/
def hourly_wage : ℝ := 25

/-- Local tax rate as a decimal -/
def tax_rate : ℝ := 0.02

/-- Health benefit rate as a decimal -/
def health_rate : ℝ := 0.015

/-- Conversion rate from dollars to cents -/
def dollars_to_cents : ℝ := 100

/-- Theorem stating the total deduction in cents -/
theorem total_deduction_in_cents : 
  hourly_wage * dollars_to_cents * (tax_rate + health_rate) = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_total_deduction_in_cents_l2637_263764


namespace NUMINAMATH_CALUDE_problem_solution_l2637_263703

theorem problem_solution : 
  (Real.sqrt 4 + abs (-3) + (2 - Real.pi) ^ 0 = 6) ∧ 
  (Real.sqrt 18 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt ((-5)^2) = 5) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2637_263703


namespace NUMINAMATH_CALUDE_straight_line_properties_l2637_263782

-- Define a straight line in a Cartesian coordinate system
structure StraightLine where
  -- We don't define the line using a specific equation to keep it general
  slope_angle : Real
  has_defined_slope : Bool

-- Theorem statement
theorem straight_line_properties (l : StraightLine) : 
  (0 ≤ l.slope_angle ∧ l.slope_angle < π) ∧ 
  (l.has_defined_slope = false → l.slope_angle = π/2) :=
by sorry

end NUMINAMATH_CALUDE_straight_line_properties_l2637_263782


namespace NUMINAMATH_CALUDE_servant_cash_compensation_l2637_263770

/-- Calculates the cash compensation for a servant given the annual salary, work period, and value of a non-cash item received. -/
def servant_compensation (annual_salary : ℚ) (work_months : ℕ) (item_value : ℚ) : ℚ :=
  annual_salary * (work_months / 12 : ℚ) - item_value

/-- Proves that the cash compensation for the servant is 57.5 given the problem conditions. -/
theorem servant_cash_compensation : 
  servant_compensation 90 9 10 = 57.5 := by
  sorry

end NUMINAMATH_CALUDE_servant_cash_compensation_l2637_263770


namespace NUMINAMATH_CALUDE_total_tables_proof_l2637_263721

/-- Represents the number of table styles -/
def num_styles : ℕ := 10

/-- Represents the sum of x values for all styles -/
def sum_x : ℕ := 100

/-- Calculates the total number of tables made in both months -/
def total_tables (num_styles : ℕ) (sum_x : ℕ) : ℕ :=
  num_styles * (2 * (sum_x / num_styles) - 3)

theorem total_tables_proof :
  total_tables num_styles sum_x = 170 :=
by sorry

end NUMINAMATH_CALUDE_total_tables_proof_l2637_263721


namespace NUMINAMATH_CALUDE_seed_testing_methods_eq_18_l2637_263784

/-- The number of ways to select and arrange seeds for testing -/
def seed_testing_methods : ℕ :=
  (Nat.choose 3 2) * (Nat.factorial 3)

/-- Theorem stating that the number of seed testing methods is 18 -/
theorem seed_testing_methods_eq_18 : seed_testing_methods = 18 := by
  sorry

end NUMINAMATH_CALUDE_seed_testing_methods_eq_18_l2637_263784


namespace NUMINAMATH_CALUDE_f_odd_a_range_l2637_263798

variable (f : ℝ → ℝ)

/-- f is an increasing function -/
axiom f_increasing : ∀ x y, x < y → f x < f y

/-- f satisfies the functional equation f(x+y) = f(x) + f(y) for all x, y ∈ ℝ -/
axiom f_add : ∀ x y, f (x + y) = f x + f y

/-- f is an odd function -/
theorem f_odd : ∀ x, f (-x) = -f x := by sorry

/-- The range of a for which f(x²) - 2f(x) < f(ax) - 2f(a) has exactly 3 positive integer solutions -/
theorem a_range : 
  {a : ℝ | 5 < a ∧ a ≤ 6} = 
  {a : ℝ | ∃! (x₁ x₂ x₃ : ℕ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    (∀ x : ℝ, x > 0 → (f (x^2) - 2*f x < f (a*x) - 2*f a ↔ x = x₁ ∨ x = x₂ ∨ x = x₃))} := by sorry

end NUMINAMATH_CALUDE_f_odd_a_range_l2637_263798


namespace NUMINAMATH_CALUDE_square_binomial_constant_l2637_263779

theorem square_binomial_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 100*x + c = (x + a)^2) → c = 2500 :=
by sorry

end NUMINAMATH_CALUDE_square_binomial_constant_l2637_263779


namespace NUMINAMATH_CALUDE_roundness_of_24300000_l2637_263737

/-- Roundness of a positive integer is the sum of the exponents of its prime factors. -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The number we're analyzing -/
def number : ℕ+ := 24300000

theorem roundness_of_24300000 : roundness number = 15 := by sorry

end NUMINAMATH_CALUDE_roundness_of_24300000_l2637_263737


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l2637_263745

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l2637_263745


namespace NUMINAMATH_CALUDE_intersection_points_l2637_263720

-- Define the two functions
def f (x : ℝ) : ℝ := x^2 + 3*x + 2
def g (x : ℝ) : ℝ := 4*x^2 + 6*x + 2

-- Theorem stating the intersection points
theorem intersection_points :
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = 0 ∧ f x₁ = g x₁ ∧ f x₁ = 2) ∧
    (x₂ = -1 ∧ f x₂ = g x₂ ∧ f x₂ = 0) ∧
    (∀ x : ℝ, f x = g x → x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_l2637_263720


namespace NUMINAMATH_CALUDE_rotated_square_height_l2637_263713

/-- The distance of point B from the original line when a square is rotated -/
theorem rotated_square_height (side_length : ℝ) (rotation_angle : ℝ) : 
  side_length = 4 →
  rotation_angle = 30 * π / 180 →
  let diagonal := side_length * Real.sqrt 2
  let height := (diagonal / 2) * Real.sin rotation_angle
  height = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rotated_square_height_l2637_263713


namespace NUMINAMATH_CALUDE_f_properties_l2637_263752

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ a then (1/a) * x
  else if a < x ∧ x ≤ 1 then (1/(1-a)) * (1-x)
  else 0

def is_turning_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f (f x) = x ∧ f x ≠ x

theorem f_properties (a : ℝ) (h : 0 < a ∧ a < 1) :
  (f (1/2) (f (1/2) (4/5)) = 4/5 ∧ is_turning_point (f (1/2)) (4/5)) ∧
  (∀ x : ℝ, a < x → x ≤ 1 →
    f a (f a x) = if x < a^2 - a + 1
                  then 1/(1-a) * (1 - 1/(1-a) * (1-x))
                  else 1/(a*(1-a)) * (1-x)) ∧
  (is_turning_point (f a) (1/(2-a)) ∧ is_turning_point (f a) (1/(1+a-a^2))) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l2637_263752


namespace NUMINAMATH_CALUDE_unreachable_target_l2637_263718

/-- A permutation of the first 100 natural numbers -/
def Permutation := Fin 100 → Fin 100

/-- The initial sequence 1, 2, 3, ..., 99, 100 -/
def initial : Permutation := fun i => i + 1

/-- The target sequence 100, 99, 98, ..., 2, 1 -/
def target : Permutation := fun i => 100 - i

/-- A valid swap in the sequence -/
def validSwap (p : Permutation) (i j : Fin 100) : Prop :=
  ∃ k, i < k ∧ k < j ∧ j = i + 2 ∧
    (∀ m, m ≠ i ∧ m ≠ j → p m = p m) ∧
    p i = p j ∧ p j = p i

/-- A sequence that can be obtained from the initial sequence using valid swaps -/
inductive reachable : Permutation → Prop
  | init : reachable initial
  | swap : ∀ {p q : Permutation}, reachable p → validSwap p i j → q = p ∘ (Equiv.swap i j) → reachable q

theorem unreachable_target : ¬ reachable target := by sorry

end NUMINAMATH_CALUDE_unreachable_target_l2637_263718


namespace NUMINAMATH_CALUDE_special_arrangements_count_l2637_263789

/-- The number of ways to arrange 3 boys and 3 girls in a row, 
    where one specific boy is not adjacent to the other two boys -/
def special_arrangements : ℕ :=
  let n_boys := 3
  let n_girls := 3
  let arrangements_with_boys_separated := n_girls.factorial * (n_girls + 1).factorial
  let arrangements_with_two_boys_adjacent := 2 * (n_girls + 1).factorial * n_girls.factorial
  arrangements_with_boys_separated + arrangements_with_two_boys_adjacent

theorem special_arrangements_count : special_arrangements = 288 := by
  sorry

end NUMINAMATH_CALUDE_special_arrangements_count_l2637_263789


namespace NUMINAMATH_CALUDE_inequality_group_solution_set_l2637_263746

theorem inequality_group_solution_set :
  let S := {x : ℝ | 2 * x + 3 ≥ -1 ∧ 7 - 3 * x > 1}
  S = {x : ℝ | -2 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_inequality_group_solution_set_l2637_263746


namespace NUMINAMATH_CALUDE_range_of_f_range_of_m_l2637_263709

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

-- Theorem for the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-3) 3 :=
sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∀ x, f x + 2 * m - 1 ≥ 0) ↔ m ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_range_of_m_l2637_263709


namespace NUMINAMATH_CALUDE_min_y_value_l2637_263793

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16*x + 50*y + 64) : y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_min_y_value_l2637_263793


namespace NUMINAMATH_CALUDE_anya_initial_seat_l2637_263735

def Friend := Fin 5

structure SeatingArrangement where
  seats : Friend → Fin 5
  bijective : Function.Bijective seats

def move_right (n : Nat) (s : Fin 5) : Fin 5 :=
  ⟨(s.val + n) % 5, by sorry⟩

def move_left (n : Nat) (s : Fin 5) : Fin 5 :=
  ⟨(s.val + 5 - n % 5) % 5, by sorry⟩

def swap (s1 s2 : Fin 5) (s : Fin 5) : Fin 5 :=
  if s = s1 then s2
  else if s = s2 then s1
  else s

theorem anya_initial_seat (initial final : SeatingArrangement) 
  (anya varya galya diana ellya : Friend) :
  initial.seats anya ≠ 1 →
  initial.seats anya ≠ 5 →
  final.seats anya = 1 ∨ final.seats anya = 5 →
  final.seats varya = move_right 1 (initial.seats varya) →
  final.seats galya = move_left 3 (initial.seats galya) →
  final.seats diana = initial.seats ellya →
  final.seats ellya = initial.seats diana →
  initial.seats anya = 3 := by sorry

end NUMINAMATH_CALUDE_anya_initial_seat_l2637_263735


namespace NUMINAMATH_CALUDE_white_line_length_l2637_263726

theorem white_line_length : 
  let blue_line_length : Float := 3.3333333333333335
  let difference : Float := 4.333333333333333
  let white_line_length : Float := blue_line_length + difference
  white_line_length = 7.666666666666667 := by
sorry

end NUMINAMATH_CALUDE_white_line_length_l2637_263726


namespace NUMINAMATH_CALUDE_sequence_sum_theorem_l2637_263711

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n : ℕ, b (n + 1) = b n * r

def increasing_sequence (s : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, s (n + 1) > s n

theorem sequence_sum_theorem (a b : ℕ → ℕ) (k : ℕ) :
  a 1 = 1 →
  b 1 = 1 →
  arithmetic_sequence a →
  geometric_sequence b →
  increasing_sequence a →
  increasing_sequence b →
  (∃ k : ℕ, a (k - 1) + b (k - 1) = 250 ∧ a (k + 1) + b (k + 1) = 1250) →
  a k + b k = 502 :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_theorem_l2637_263711


namespace NUMINAMATH_CALUDE_power_of_32_equals_power_of_2_l2637_263714

theorem power_of_32_equals_power_of_2 : ∀ q : ℕ, 32^5 = 2^q → q = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_of_32_equals_power_of_2_l2637_263714


namespace NUMINAMATH_CALUDE_lcm_n_n_plus_3_l2637_263794

theorem lcm_n_n_plus_3 (n : ℕ) :
  lcm n (n + 3) = if n % 3 = 0 then n * (n + 3) / 3 else n * (n + 3) := by
  sorry

end NUMINAMATH_CALUDE_lcm_n_n_plus_3_l2637_263794


namespace NUMINAMATH_CALUDE_doll_collection_increase_l2637_263773

theorem doll_collection_increase (original_count : ℕ) (increase_percentage : ℚ) (final_count : ℕ) 
  (h1 : increase_percentage = 25 / 100)
  (h2 : final_count = 10)
  (h3 : final_count = original_count + (increase_percentage * original_count).floor) :
  final_count - original_count = 2 := by
  sorry

end NUMINAMATH_CALUDE_doll_collection_increase_l2637_263773


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l2637_263742

/-- Proves that mixing 200 mL of 10% alcohol solution with 600 mL of 30% alcohol solution results in a 25% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 200
  let x_concentration : ℝ := 0.10
  let y_volume : ℝ := 600
  let y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.25
  let total_volume := x_volume + y_volume
  let total_alcohol := x_volume * x_concentration + y_volume * y_concentration
  (total_alcohol / total_volume) = target_concentration :=
by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l2637_263742


namespace NUMINAMATH_CALUDE_multiple_of_960_l2637_263761

theorem multiple_of_960 (a : ℤ) 
  (h1 : ∃ k : ℤ, a = 10 * k + 4) 
  (h2 : ¬ (∃ m : ℤ, a = 4 * m)) : 
  ∃ n : ℤ, a * (a^2 - 1) * (a^2 - 4) = 960 * n :=
sorry

end NUMINAMATH_CALUDE_multiple_of_960_l2637_263761


namespace NUMINAMATH_CALUDE_cos_96_cos_24_minus_sin_96_cos_66_l2637_263701

theorem cos_96_cos_24_minus_sin_96_cos_66 : 
  Real.cos (96 * π / 180) * Real.cos (24 * π / 180) - 
  Real.sin (96 * π / 180) * Real.cos (66 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_96_cos_24_minus_sin_96_cos_66_l2637_263701


namespace NUMINAMATH_CALUDE_total_pets_is_19_l2637_263758

/-- Represents the number of pets Frankie has -/
structure PetCounts where
  cats : ℕ
  snakes : ℕ
  parrots : ℕ
  dogs : ℕ

/-- Conditions for Frankie's pets -/
def validPetCounts (p : PetCounts) : Prop :=
  p.snakes = p.cats + 6 ∧
  p.parrots = p.cats - 1 ∧
  p.cats + p.dogs = 6

/-- Theorem stating that the total number of pets is 19 -/
theorem total_pets_is_19 (p : PetCounts) (h : validPetCounts p) :
  p.cats + p.snakes + p.parrots + p.dogs = 19 := by
  sorry


end NUMINAMATH_CALUDE_total_pets_is_19_l2637_263758
