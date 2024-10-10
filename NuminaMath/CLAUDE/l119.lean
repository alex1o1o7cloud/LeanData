import Mathlib

namespace brenda_bracelets_l119_11972

theorem brenda_bracelets (total_stones : ℕ) (stones_per_bracelet : ℕ) (h1 : total_stones = 36) (h2 : stones_per_bracelet = 12) :
  total_stones / stones_per_bracelet = 3 := by
  sorry

end brenda_bracelets_l119_11972


namespace least_three_digit_7_shifty_l119_11966

def is_7_shifty (n : ℕ) : Prop := n % 7 > 2

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_7_shifty : 
  (∀ m : ℕ, is_three_digit m → is_7_shifty m → 101 ≤ m) ∧ 
  is_three_digit 101 ∧ 
  is_7_shifty 101 :=
sorry

end least_three_digit_7_shifty_l119_11966


namespace imaginary_part_of_complex_expression_l119_11963

theorem imaginary_part_of_complex_expression :
  let z : ℂ := (1 + I) / (1 - I) + (1 - I)^2
  Complex.im z = -1 := by sorry

end imaginary_part_of_complex_expression_l119_11963


namespace distinct_roots_of_quadratic_l119_11975

theorem distinct_roots_of_quadratic (a : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - a*x₁ - 2 = 0 ∧ x₂^2 - a*x₂ - 2 = 0 := by
  sorry

end distinct_roots_of_quadratic_l119_11975


namespace supermarket_growth_l119_11992

/-- Represents the growth of a supermarket's turnover from January to March -/
theorem supermarket_growth (x : ℝ) : 
  (36 : ℝ) * (1 + x)^2 = 48 ↔ 
  (∃ (jan mar : ℝ), 
    jan = 36 ∧ 
    mar = 48 ∧ 
    mar = jan * (1 + x)^2 ∧ 
    x ≥ 0) :=
sorry

end supermarket_growth_l119_11992


namespace regular_ngon_diagonal_difference_l119_11936

/-- The difference between the longest and shortest diagonals of a regular n-gon equals its side length if and only if n = 9 -/
theorem regular_ngon_diagonal_difference (n : ℕ) (h : n ≥ 3) :
  let R : ℝ := 1  -- Assume unit circle for simplicity
  let side_length := 2 * Real.sin (Real.pi / n)
  let shortest_diagonal := 2 * Real.sin (2 * Real.pi / n)
  let longest_diagonal := if n % 2 = 0 then 2 else 2 * Real.cos (Real.pi / (2 * n))
  longest_diagonal - shortest_diagonal = side_length ↔ n = 9 := by
sorry


end regular_ngon_diagonal_difference_l119_11936


namespace min_value_reciprocal_sum_l119_11924

theorem min_value_reciprocal_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : Real.log (a + b) = 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → Real.log (x + y) = 0 → 1/x + 4/y ≥ 1/a + 4/b) ∧ 
  1/a + 4/b = 9 := by
  sorry

end min_value_reciprocal_sum_l119_11924


namespace system_solution_l119_11988

theorem system_solution : ∃! (x y : ℚ), 3 * x + 4 * y = 20 ∧ 9 * x - 8 * y = 36 ∧ x = 76 / 15 ∧ y = 18 / 15 := by
  sorry

end system_solution_l119_11988


namespace initial_candies_equals_sum_of_given_and_left_l119_11970

/-- Given the number of candies given away and the number of candies left,
    prove that the initial number of candies is their sum. -/
theorem initial_candies_equals_sum_of_given_and_left (given away : ℕ) (left : ℕ) :
  given + left = given + left := by sorry

end initial_candies_equals_sum_of_given_and_left_l119_11970


namespace sequence_properties_l119_11914

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def S (b : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => S b n + b (n + 1)

def T (c : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => T c n + c (n + 1)

theorem sequence_properties (a b c : ℕ → ℝ) :
  arithmetic_sequence a
  ∧ a 5 = 14
  ∧ a 7 = 20
  ∧ b 1 = 2/3
  ∧ (∀ n : ℕ, n ≥ 2 → 3 * S b n = S b (n-1) + 2)
  ∧ (∀ n : ℕ, c n = a n * b n)
  →
  (∀ n : ℕ, a n = 3*n - 1)
  ∧ (∀ n : ℕ, b n = 2 * (1/3)^n)
  ∧ (∀ n : ℕ, n ≥ 1 → T c n < 7/2)
  ∧ (∀ m : ℝ, (∀ n : ℕ, n ≥ 1 → T c n < m) → m ≥ 7/2) :=
by sorry

end sequence_properties_l119_11914


namespace power_equality_l119_11912

theorem power_equality (p : ℕ) : 81^6 = 3^p → p = 24 := by
  sorry

end power_equality_l119_11912


namespace linear_equation_solution_l119_11953

theorem linear_equation_solution (x y : ℝ) :
  2 * x + y - 5 = 0 → x = (5 - y) / 2 := by
  sorry

end linear_equation_solution_l119_11953


namespace ratio_section_area_l119_11945

/-- Regular quadrilateral prism -/
structure RegularQuadPrism where
  base : Real
  height : Real

/-- Cross-section passing through midpoints -/
def midpoint_section (p : RegularQuadPrism) : Real :=
  12

/-- Cross-section dividing axis in ratio 1:3 -/
def ratio_section (p : RegularQuadPrism) : Real :=
  9

/-- Theorem statement -/
theorem ratio_section_area (p : RegularQuadPrism) :
  midpoint_section p = 12 → ratio_section p = 9 := by
  sorry

end ratio_section_area_l119_11945


namespace cuboidal_block_dimension_l119_11944

/-- Given a cuboidal block with dimensions x cm × 9 cm × 12 cm that can be cut into at least 24 equal cubes,
    prove that the length of the first dimension (x) must be 6 cm. -/
theorem cuboidal_block_dimension (x : ℕ) : 
  (∃ (n : ℕ), n ≥ 24 ∧ x * 9 * 12 = n * (gcd x (gcd 9 12))^3) → x = 6 := by
  sorry

end cuboidal_block_dimension_l119_11944


namespace polynomial_expansion_equality_l119_11931

theorem polynomial_expansion_equality (x : ℝ) : 
  (3*x^2 + 4*x + 8)*(x - 1) - (x - 1)*(x^2 + 5*x - 72) + (4*x - 15)*(x - 1)*(x + 6) = 
  6*x^3 + 2*x^2 - 18*x + 10 := by
sorry

end polynomial_expansion_equality_l119_11931


namespace decimal_sum_as_fraction_l119_11998

theorem decimal_sum_as_fraction : 
  (0.2 + 0.03 + 0.004 + 0.0006 + 0.00007 + 0.000008 + 0.0000009 : ℚ) = 2340087/10000000 := by
  sorry

end decimal_sum_as_fraction_l119_11998


namespace no_rational_roots_l119_11915

def f (x : ℚ) : ℚ := 3 * x^4 - 2 * x^3 - 8 * x^2 + 3 * x + 1

theorem no_rational_roots : ∀ (x : ℚ), f x ≠ 0 := by
  sorry

end no_rational_roots_l119_11915


namespace age_equality_l119_11960

theorem age_equality (joe_current_age : ℕ) (james_current_age : ℕ) (years_until_equality : ℕ) : 
  joe_current_age = 22 →
  james_current_age = joe_current_age - 10 →
  2 * (joe_current_age + years_until_equality) = 3 * (james_current_age + years_until_equality) →
  years_until_equality = 8 := by
sorry

end age_equality_l119_11960


namespace logical_reasoning_classification_l119_11918

-- Define the types of reasoning
inductive ReasoningType
  | Sphere
  | Triangle
  | Chair
  | Polygon

-- Define a predicate for logical reasoning
def is_logical (r : ReasoningType) : Prop :=
  match r with
  | ReasoningType.Sphere => true   -- Analogy reasoning
  | ReasoningType.Triangle => true -- Inductive reasoning
  | ReasoningType.Chair => false   -- Not logical
  | ReasoningType.Polygon => true  -- Inductive reasoning

-- Theorem statement
theorem logical_reasoning_classification :
  (is_logical ReasoningType.Sphere) ∧
  (is_logical ReasoningType.Triangle) ∧
  (¬is_logical ReasoningType.Chair) ∧
  (is_logical ReasoningType.Polygon) :=
sorry

end logical_reasoning_classification_l119_11918


namespace wickets_before_last_match_is_55_l119_11935

/-- Represents a bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  lastMatchWickets : ℕ
  lastMatchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the number of wickets taken before the last match -/
def wicketsBeforeLastMatch (stats : BowlerStats) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the number of wickets before the last match is 55 -/
theorem wickets_before_last_match_is_55 (stats : BowlerStats) 
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.lastMatchWickets = 4)
  (h3 : stats.lastMatchRuns = 26)
  (h4 : stats.averageDecrease = 0.4) :
  wicketsBeforeLastMatch stats = 55 := by
  sorry

end wickets_before_last_match_is_55_l119_11935


namespace even_sum_difference_l119_11938

def sum_even_range (a b : ℕ) : ℕ := 
  (b - a + 2) / 2 * (a + b) / 2

theorem even_sum_difference : 
  sum_even_range 62 110 - sum_even_range 22 70 = 1000 := by
  sorry

end even_sum_difference_l119_11938


namespace half_abs_diff_squares_l119_11961

theorem half_abs_diff_squares : (1/2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end half_abs_diff_squares_l119_11961


namespace value_of_a_l119_11910

theorem value_of_a (A B : Set ℕ) (a : ℕ) :
  A = {a, 2} →
  B = {1, 2} →
  A ∪ B = {1, 2, 3} →
  a = 3 := by
sorry

end value_of_a_l119_11910


namespace prob_non_black_ball_l119_11940

/-- Given a bag of balls where the odds of drawing a black ball are 5:3,
    the probability of drawing a non-black ball is 3/8 -/
theorem prob_non_black_ball (total : ℕ) (black : ℕ) (non_black : ℕ)
  (h_total : total = black + non_black)
  (h_odds : (black : ℚ) / non_black = 5 / 3) :
  (non_black : ℚ) / total = 3 / 8 := by
  sorry

end prob_non_black_ball_l119_11940


namespace square_of_negative_triple_l119_11943

theorem square_of_negative_triple (a : ℝ) : (-3 * a)^2 = 9 * a^2 := by
  sorry

end square_of_negative_triple_l119_11943


namespace nurses_who_quit_l119_11903

theorem nurses_who_quit (initial_doctors initial_nurses doctors_quit total_remaining : ℕ) :
  initial_doctors = 11 →
  initial_nurses = 18 →
  doctors_quit = 5 →
  total_remaining = 22 →
  initial_doctors + initial_nurses - doctors_quit - total_remaining = 2 := by
  sorry

end nurses_who_quit_l119_11903


namespace ratio_w_to_y_l119_11962

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 5 / 2)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 4) :
  w / y = 20 / 3 := by
sorry

end ratio_w_to_y_l119_11962


namespace at_equals_rc_l119_11921

-- Define the points
variable (A B C D M P R Q S T : Point)

-- Define the cyclic quadrilateral ABCD
def is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Define M as midpoint of CD
def is_midpoint (M C D : Point) : Prop := sorry

-- Define P as intersection of diagonals AC and BD
def is_diagonal_intersection (P A B C D : Point) : Prop := sorry

-- Define circle through P touching CD at M and meeting AC at R and BD at Q
def circle_touches_and_meets (P M C D R Q : Point) : Prop := sorry

-- Define S on BD such that BS = DQ
def point_on_line_with_equal_distance (S B D Q : Point) : Prop := sorry

-- Define line through S parallel to AB meeting AC at T
def parallel_line_intersection (S T A B C : Point) : Prop := sorry

-- Theorem statement
theorem at_equals_rc 
  (h1 : is_cyclic_quadrilateral A B C D)
  (h2 : is_midpoint M C D)
  (h3 : is_diagonal_intersection P A B C D)
  (h4 : circle_touches_and_meets P M C D R Q)
  (h5 : point_on_line_with_equal_distance S B D Q)
  (h6 : parallel_line_intersection S T A B C) :
  AT = RC := by sorry

end at_equals_rc_l119_11921


namespace games_given_away_l119_11990

def initial_games : ℕ := 183
def remaining_games : ℕ := 92

theorem games_given_away : initial_games - remaining_games = 91 := by
  sorry

end games_given_away_l119_11990


namespace conclusion_1_conclusion_2_conclusion_3_l119_11926

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + 3

-- Theorem 1
theorem conclusion_1 (b : ℝ) :
  (∀ m : ℝ, m*(m - 2*b) ≥ 1 - 2*b) → b = 1 := by sorry

-- Theorem 2
theorem conclusion_2 (b : ℝ) :
  ∃ h k : ℝ, (∀ x : ℝ, f b x ≥ f b h) ∧ k = f b h ∧ k = -h^2 + 3 := by sorry

-- Theorem 3
theorem conclusion_3 (b : ℝ) :
  (∀ x : ℝ, -1 ≤ x → x ≤ 5 → f b x ≤ f b (-1)) →
  (∃ m₁ m₂ p : ℝ, m₁ ≠ m₂ ∧ f b m₁ = p ∧ f b m₂ = p) →
  ∃ m₁ m₂ : ℝ, m₁ + m₂ > 4 := by sorry

end conclusion_1_conclusion_2_conclusion_3_l119_11926


namespace movie_admission_problem_l119_11905

theorem movie_admission_problem (total_admitted : ℕ) 
  (west_side_total : ℕ) (west_side_denied_percent : ℚ)
  (mountaintop_total : ℕ) (mountaintop_denied_percent : ℚ)
  (first_school_denied_percent : ℚ) :
  total_admitted = 148 →
  west_side_total = 90 →
  west_side_denied_percent = 70/100 →
  mountaintop_total = 50 →
  mountaintop_denied_percent = 1/2 →
  first_school_denied_percent = 20/100 →
  ∃ (first_school_total : ℕ),
    first_school_total = 120 ∧
    total_admitted = 
      (first_school_total * (1 - first_school_denied_percent)).floor +
      (west_side_total * (1 - west_side_denied_percent)).floor +
      (mountaintop_total * (1 - mountaintop_denied_percent)).floor :=
by sorry

end movie_admission_problem_l119_11905


namespace store_earnings_calculation_l119_11965

/-- Represents the earnings calculation for a store selling bottled drinks -/
theorem store_earnings_calculation (cola_price juice_price water_price sports_price : ℚ)
                                   (cola_sold juice_sold water_sold sports_sold : ℕ) :
  cola_price = 3 →
  juice_price = 3/2 →
  water_price = 1 →
  sports_price = 5/2 →
  cola_sold = 18 →
  juice_sold = 15 →
  water_sold = 30 →
  sports_sold = 22 →
  cola_price * cola_sold + juice_price * juice_sold + 
  water_price * water_sold + sports_price * sports_sold = 161.5 := by
sorry

end store_earnings_calculation_l119_11965


namespace n_equals_fourteen_l119_11919

def first_seven_multiples_of_seven : List ℕ := [7, 14, 21, 28, 35, 42, 49]

def a : ℚ := (first_seven_multiples_of_seven.sum : ℚ) / 7

def first_three_multiples (n : ℕ) : List ℕ := [n, 2*n, 3*n]

def b (n : ℕ) : ℕ := (first_three_multiples n).nthLe 1 sorry

theorem n_equals_fourteen (n : ℕ) (h : a^2 - (b n : ℚ)^2 = 0) : n = 14 := by
  sorry

end n_equals_fourteen_l119_11919


namespace arrow_pointing_theorem_l119_11991

/-- Represents the direction of an arrow -/
inductive Direction
| Left
| Right

/-- Represents an arrangement of arrows -/
def ArrowArrangement (n : ℕ) := Fin n → Direction

/-- The number of arrows pointing to the i-th arrow -/
def pointingTo (arr : ArrowArrangement n) (i : Fin n) : ℕ := sorry

/-- The number of arrows that the i-th arrow is pointing to -/
def pointingFrom (arr : ArrowArrangement n) (i : Fin n) : ℕ := sorry

theorem arrow_pointing_theorem (n : ℕ) (h : Odd n) (h1 : n ≥ 1) (arr : ArrowArrangement n) :
  ∃ i : Fin n, pointingTo arr i = pointingFrom arr i := by sorry

end arrow_pointing_theorem_l119_11991


namespace ellipse_min_value_l119_11969

/-- For an ellipse with semi-major axis a, semi-minor axis b, and eccentricity e,
    prove that the minimum value of (a² + 1) / b is 4√3 / 3 when e = 1/2. -/
theorem ellipse_min_value (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (a^2 - b^2) / a^2 = 1/4) :
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
  (a^2 + 1) / b ≥ 4 * Real.sqrt 3 / 3 :=
sorry

end ellipse_min_value_l119_11969


namespace sin_negative_150_degrees_l119_11920

theorem sin_negative_150_degrees : Real.sin (-(150 * π / 180)) = -(1 / 2) := by
  sorry

end sin_negative_150_degrees_l119_11920


namespace elysses_carrying_capacity_l119_11942

/-- The number of bags Elysse can carry in one trip -/
def elysses_bags : ℕ := 3

/-- The number of trips Elysse and her brother take -/
def num_trips : ℕ := 5

/-- The total number of bags they carry -/
def total_bags : ℕ := 30

theorem elysses_carrying_capacity :
  (elysses_bags * 2 * num_trips = total_bags) ∧ 
  (elysses_bags > 0) ∧ 
  (num_trips > 0) ∧ 
  (total_bags > 0) := by
  sorry

end elysses_carrying_capacity_l119_11942


namespace initial_amount_A_l119_11967

theorem initial_amount_A (a b c : ℝ) : 
  b = 28 → 
  c = 20 → 
  (a - b - c) + 2 * (a - b - c) + 4 * (a - b - c) = 24 →
  (b + b) - (2 * (a - b - c) + 2 * c) + 2 * ((b + b) - (2 * (a - b - c) + 2 * c)) = 24 →
  (c + c) - (4 * (a - b - c) + 2 * ((b + b) - (2 * (a - b - c) + 2 * c))) = 24 →
  a = 54 := by
  sorry

#check initial_amount_A

end initial_amount_A_l119_11967


namespace fraction_equality_l119_11982

theorem fraction_equality (a b : ℝ) (h : a / b = 2) : a / (a - b) = 2 := by
  sorry

end fraction_equality_l119_11982


namespace right_triangle_sides_l119_11911

theorem right_triangle_sides : 
  (7^2 + 24^2 = 25^2) ∧ 
  (1.5^2 + 2^2 = 2.5^2) ∧ 
  (8^2 + 15^2 = 17^2) ∧ 
  (Real.sqrt 3)^2 + (Real.sqrt 4)^2 ≠ (Real.sqrt 5)^2 :=
by sorry

end right_triangle_sides_l119_11911


namespace fraction_simplification_l119_11954

theorem fraction_simplification : 
  (1 / 3 + 1 / 4) / (2 / 5 - 1 / 6) = 5 / 2 := by
  sorry

end fraction_simplification_l119_11954


namespace broken_bowls_l119_11955

theorem broken_bowls (total_bowls : ℕ) (lost_bowls : ℕ) (fee : ℕ) (safe_payment : ℕ) (penalty : ℕ) (total_payment : ℕ) :
  total_bowls = 638 →
  lost_bowls = 12 →
  fee = 100 →
  safe_payment = 3 →
  penalty = 4 →
  total_payment = 1825 →
  ∃ (broken_bowls : ℕ),
    fee + safe_payment * (total_bowls - lost_bowls - broken_bowls) - 
    (penalty * lost_bowls + penalty * broken_bowls) = total_payment ∧
    broken_bowls = 29 :=
sorry

end broken_bowls_l119_11955


namespace triangle_squares_area_l119_11947

theorem triangle_squares_area (x : ℝ) : 
  let small_square_area := (3 * x)^2
  let large_square_area := (6 * x)^2
  let triangle_area := (1/2) * (3 * x) * (6 * x)
  small_square_area + large_square_area + triangle_area = 1000 →
  x = (10 * Real.sqrt 3) / 3 :=
by sorry

end triangle_squares_area_l119_11947


namespace train_journey_time_l119_11900

theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_speed > 0 →
  usual_time > 0 →
  (6/7 * usual_speed) * (usual_time + 15/60) = usual_speed * usual_time →
  usual_time = 3/2 := by
  sorry

end train_journey_time_l119_11900


namespace max_n_given_average_l119_11952

theorem max_n_given_average (m n : ℕ+) : 
  (m + n : ℚ) / 2 = 5 → n ≤ 9 :=
by sorry

end max_n_given_average_l119_11952


namespace smaller_number_problem_l119_11997

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 24) (h2 : x - y = 16) : 
  min x y = 4 := by
  sorry

end smaller_number_problem_l119_11997


namespace age_difference_l119_11958

/-- Given three people A, B, and C, where C is 13 years younger than A,
    prove that the sum of ages of A and B is 13 years more than the sum of ages of B and C. -/
theorem age_difference (A B C : ℕ) (h : C = A - 13) :
  A + B - (B + C) = 13 := by
  sorry

end age_difference_l119_11958


namespace cylinder_volume_ratio_l119_11928

/-- Given two cylinders A and B with base areas S₁ and S₂, volumes V₁ and V₂,
    if S₁/S₂ = 9/4 and their lateral surface areas are equal, then V₁/V₂ = 3/2 -/
theorem cylinder_volume_ratio (S₁ S₂ V₁ V₂ R r H h : ℝ) 
    (h_base_ratio : S₁ / S₂ = 9 / 4)
    (h_S₁ : S₁ = π * R^2)
    (h_S₂ : S₂ = π * r^2)
    (h_V₁ : V₁ = S₁ * H)
    (h_V₂ : V₂ = S₂ * h)
    (h_lateral_area : 2 * π * R * H = 2 * π * r * h) : 
  V₁ / V₂ = 3 / 2 := by
  sorry

end cylinder_volume_ratio_l119_11928


namespace math_books_arrangement_l119_11968

theorem math_books_arrangement (num_math_books num_english_books : ℕ) : 
  num_math_books = 2 → num_english_books = 2 → 
  (num_math_books.factorial * (num_math_books + num_english_books).factorial) = 12 := by
  sorry

end math_books_arrangement_l119_11968


namespace part_one_part_two_l119_11941

-- Define the propositions
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x-3)/(x+2) < 0

-- Part 1
theorem part_one (x : ℝ) (h1 : p x 1) (h2 : q x) : 1 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h : a > 0) 
  (h_suff : ∀ x, ¬(q x) → ¬(p x a))
  (h_not_nec : ∃ x, q x ∧ ¬(p x a)) : 
  0 < a ∧ a ≤ 1 := by sorry

end part_one_part_two_l119_11941


namespace functional_equation_solution_l119_11904

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 - x + 1) / (5 * (x - 1))

theorem functional_equation_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  x * f x + 2 * f ((x - 1) / (x + 1)) = 1 := by
  sorry

end functional_equation_solution_l119_11904


namespace vector_collinearity_l119_11996

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-2, 1)
def c : ℝ × ℝ := (3, 2)

def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, u.1 * v.2 = t * u.2 * v.1

theorem vector_collinearity (k : ℝ) :
  collinear c ((k * a.1 + b.1, k * a.2 + b.2)) → k = -1 := by
  sorry

end vector_collinearity_l119_11996


namespace intersection_is_empty_l119_11950

def set_A : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 = 0}
def set_B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 2 * p.1 - 3}

theorem intersection_is_empty : set_A ∩ set_B = ∅ := by
  sorry

end intersection_is_empty_l119_11950


namespace pizza_toppings_combinations_l119_11927

theorem pizza_toppings_combinations (n : ℕ) (h : n = 8) : 
  n + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end pizza_toppings_combinations_l119_11927


namespace largest_integer_m_l119_11946

theorem largest_integer_m (m : ℤ) : (∀ k : ℤ, k > 6 → k^2 - 11*k + 28 ≥ 0) ∧ 6^2 - 11*6 + 28 < 0 := by
  sorry

end largest_integer_m_l119_11946


namespace smallest_k_value_l119_11976

theorem smallest_k_value (x y : ℤ) (h1 : x = -2) (h2 : y = 5) : 
  ∃ k : ℤ, (∀ m : ℤ, k * x + 2 * y ≤ 4 → m * x + 2 * y ≤ 4 → k ≤ m) ∧ k = 3 :=
by sorry

end smallest_k_value_l119_11976


namespace arithmetic_geometric_mean_sum_of_squares_l119_11956

theorem arithmetic_geometric_mean_sum_of_squares 
  (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = 15) : 
  x^2 + y^2 = 1150 := by sorry

end arithmetic_geometric_mean_sum_of_squares_l119_11956


namespace mike_ride_distance_l119_11971

/-- Represents the taxi fare structure and ride details -/
structure TaxiRide where
  start_fee : ℝ
  per_mile_fee : ℝ
  toll_fee : ℝ
  distance : ℝ

/-- Calculates the total fare for a taxi ride -/
def total_fare (ride : TaxiRide) : ℝ :=
  ride.start_fee + ride.toll_fee + ride.per_mile_fee * ride.distance

/-- Proves that Mike's ride was 34 miles long given the conditions -/
theorem mike_ride_distance :
  let mike : TaxiRide := { start_fee := 2.5, per_mile_fee := 0.25, toll_fee := 0, distance := 34 }
  let annie : TaxiRide := { start_fee := 2.5, per_mile_fee := 0.25, toll_fee := 5, distance := 14 }
  total_fare mike = total_fare annie := by
  sorry

#check mike_ride_distance

end mike_ride_distance_l119_11971


namespace equation_solutions_l119_11916

theorem equation_solutions :
  (∃ x : ℝ, 3 * x^3 - 15 = 9 ∧ x = 2) ∧
  (∃ x₁ x₂ : ℝ, 2 * (x₁ - 1)^2 = 72 ∧ 2 * (x₂ - 1)^2 = 72 ∧ x₁ = 7 ∧ x₂ = -5) := by
  sorry

end equation_solutions_l119_11916


namespace quadratic_equation_solution_l119_11901

theorem quadratic_equation_solution (m : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 - (m + 3) * x₁ + m + 2 = 0) →
  (x₂^2 - (m + 3) * x₂ + m + 2 = 0) →
  (x₁ / (x₁ + 1) + x₂ / (x₂ + 1) = 13 / 10) →
  m = 2 := by
sorry

end quadratic_equation_solution_l119_11901


namespace solution_product_l119_11951

theorem solution_product (a b : ℝ) : 
  (a - 3) * (2 * a + 7) = a^2 - 11 * a + 28 →
  (b - 3) * (2 * b + 7) = b^2 - 11 * b + 28 →
  a ≠ b →
  (a + 2) * (b + 2) = -66 := by
sorry

end solution_product_l119_11951


namespace division_remainder_l119_11986

theorem division_remainder : ∀ (dividend divisor quotient : ℕ),
  dividend = 166 →
  divisor = 18 →
  quotient = 9 →
  dividend = divisor * quotient + 4 :=
by sorry

end division_remainder_l119_11986


namespace smallest_cube_ending_2016_l119_11985

theorem smallest_cube_ending_2016 :
  ∃ (n : ℕ), n^3 % 10000 = 2016 ∧ ∀ (m : ℕ), m < n → m^3 % 10000 ≠ 2016 :=
by
  use 856
  sorry

end smallest_cube_ending_2016_l119_11985


namespace smaller_number_proof_l119_11939

theorem smaller_number_proof (x y : ℝ) (h1 : x + y = 56) (h2 : y = x + 12) : x = 22 := by
  sorry

end smaller_number_proof_l119_11939


namespace smallest_non_odd_units_digit_l119_11930

def isOddUnitsDigit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def isSingleDigit (d : ℕ) : Prop := d < 10

theorem smallest_non_odd_units_digit :
  ∀ d : ℕ, isSingleDigit d → (d < 0 ∨ isOddUnitsDigit d) → 0 ≤ d :=
by sorry

end smallest_non_odd_units_digit_l119_11930


namespace tickets_theorem_l119_11932

/-- Calculates the total number of tickets Tate and Peyton have together -/
def totalTickets (tateInitial : ℕ) (tateAdditional : ℕ) : ℕ :=
  let tateFinal := tateInitial + tateAdditional
  let peyton := tateFinal / 2
  tateFinal + peyton

/-- Theorem stating that given the initial conditions, the total number of tickets is 51 -/
theorem tickets_theorem :
  totalTickets 32 2 = 51 := by
  sorry

end tickets_theorem_l119_11932


namespace sallys_initial_cards_l119_11979

/-- Proves that Sally's initial number of cards was 27 given the problem conditions -/
theorem sallys_initial_cards : 
  ∀ x : ℕ, 
  (x + 41 + 20 = 88) → 
  x = 27 := by
  sorry

end sallys_initial_cards_l119_11979


namespace percentage_spent_on_hats_l119_11913

def total_money : ℕ := 90
def scarf_count : ℕ := 18
def scarf_price : ℕ := 2
def hat_to_scarf_ratio : ℕ := 2

theorem percentage_spent_on_hats :
  let money_spent_on_scarves := scarf_count * scarf_price
  let money_spent_on_hats := total_money - money_spent_on_scarves
  let percentage_on_hats := (money_spent_on_hats : ℚ) / total_money * 100
  percentage_on_hats = 60 := by
  sorry

end percentage_spent_on_hats_l119_11913


namespace candy_bar_profit_l119_11949

/-- Calculates the profit from selling candy bars --/
theorem candy_bar_profit : 
  let total_bars : ℕ := 1500
  let purchase_price : ℚ := 3 / 4
  let sold_first : ℕ := 1200
  let price_first : ℚ := 2 / 3
  let sold_second : ℕ := 300
  let price_second : ℚ := 8 / 10
  let total_cost : ℚ := total_bars * purchase_price
  let revenue_first : ℚ := sold_first * price_first
  let revenue_second : ℚ := sold_second * price_second
  let total_revenue : ℚ := revenue_first + revenue_second
  let profit : ℚ := total_revenue - total_cost
  profit = -85 :=
by sorry

end candy_bar_profit_l119_11949


namespace historical_fiction_new_releases_fraction_l119_11977

/-- Represents the inventory of a bookstore -/
structure BookInventory where
  total : ℕ
  historicalFiction : ℕ
  historicalFictionNewReleases : ℕ
  otherNewReleases : ℕ

/-- Conditions for Joel's bookstore inventory -/
def joelsBookstore (inventory : BookInventory) : Prop :=
  inventory.historicalFiction = (30 * inventory.total) / 100 ∧
  inventory.historicalFictionNewReleases = (30 * inventory.historicalFiction) / 100 ∧
  inventory.otherNewReleases = (40 * (inventory.total - inventory.historicalFiction)) / 100

/-- Theorem: The fraction of all new releases that are historical fiction is 9/37 -/
theorem historical_fiction_new_releases_fraction 
  (inventory : BookInventory) (h : joelsBookstore inventory) :
  (inventory.historicalFictionNewReleases : ℚ) / 
  (inventory.historicalFictionNewReleases + inventory.otherNewReleases) = 9 / 37 := by
  sorry

end historical_fiction_new_releases_fraction_l119_11977


namespace exists_increasing_omega_sequence_l119_11933

/-- The number of distinct prime factors of a natural number -/
def omega (n : ℕ) : ℕ := sorry

/-- For any k, there exists an n > k satisfying the omega inequality -/
theorem exists_increasing_omega_sequence (k : ℕ) :
  ∃ n : ℕ, n > k ∧ omega n < omega (n + 1) ∧ omega (n + 1) < omega (n + 2) :=
sorry

end exists_increasing_omega_sequence_l119_11933


namespace kaleb_candy_count_l119_11923

/-- The number of candies Kaleb can buy with his arcade tickets -/
def candies_kaleb_can_buy (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) : ℕ :=
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost

/-- Proof that Kaleb can buy 3 candies with his arcade tickets -/
theorem kaleb_candy_count : candies_kaleb_can_buy 8 7 5 = 3 := by
  sorry

end kaleb_candy_count_l119_11923


namespace square_sum_and_reciprocal_l119_11987

theorem square_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end square_sum_and_reciprocal_l119_11987


namespace infinitely_many_even_floor_alpha_n_squared_l119_11929

theorem infinitely_many_even_floor_alpha_n_squared (α : ℝ) (hα : α > 0) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, Even ⌊α * n^2⌋ := by sorry

end infinitely_many_even_floor_alpha_n_squared_l119_11929


namespace complex_square_simplification_l119_11909

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (5 - 3 * i)^2 = 16 - 30 * i := by sorry

end complex_square_simplification_l119_11909


namespace triangle_abc_problem_l119_11917

theorem triangle_abc_problem (A B C : ℝ) (a b c : ℝ) :
  b * Real.sin A = 3 * c * Real.sin B →
  a = 3 →
  Real.cos B = 2/3 →
  b = Real.sqrt 6 ∧ 
  Real.sin (2*B - π/3) = (4*Real.sqrt 5 + Real.sqrt 3) / 18 :=
by sorry

end triangle_abc_problem_l119_11917


namespace colin_skipping_speed_l119_11925

theorem colin_skipping_speed (bruce_speed tony_speed brandon_speed colin_speed : ℝ) :
  bruce_speed = 1 →
  tony_speed = 2 * bruce_speed →
  brandon_speed = (1/3) * tony_speed →
  colin_speed = 6 * brandon_speed →
  colin_speed = 4 := by
sorry

end colin_skipping_speed_l119_11925


namespace dot_product_of_vectors_l119_11959

theorem dot_product_of_vectors (a b : ℝ × ℝ) 
  (h1 : a + b = (1, -3))
  (h2 : a - b = (3, 7)) :
  a • b = -12 := by
  sorry

end dot_product_of_vectors_l119_11959


namespace minimum_value_theorem_l119_11922

-- Define the line equation
def line_equation (m n x y : ℝ) : Prop := m * x + n * y + 2 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 3)^2 + (y + 1)^2 = 1

-- Define the chord length condition
def chord_length_condition (m n : ℝ) : Prop := 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line_equation m n x₁ y₁ ∧ 
    line_equation m n x₂ y₂ ∧ 
    circle_equation x₁ y₁ ∧ 
    circle_equation x₂ y₂ ∧ 
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4

theorem minimum_value_theorem (m n : ℝ) 
  (hm : m > 0) (hn : n > 0) 
  (h_chord : chord_length_condition m n) : 
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → chord_length_condition m' n' → 1/m' + 3/n' ≥ 1/m + 3/n) → 
  1/m + 3/n = 6 := by
sorry

end minimum_value_theorem_l119_11922


namespace shaded_area_is_65_l119_11907

/-- Represents a trapezoid with a line segment dividing it into two parts -/
structure DividedTrapezoid where
  total_area : ℝ
  dividing_segment_length : ℝ
  inner_segment_length : ℝ

/-- Calculates the area of the shaded region in the divided trapezoid -/
def shaded_area (t : DividedTrapezoid) : ℝ :=
  t.total_area - (t.dividing_segment_length * t.inner_segment_length)

/-- Theorem stating that for the given trapezoid, the shaded area is 65 -/
theorem shaded_area_is_65 (t : DividedTrapezoid) 
  (h1 : t.total_area = 117)
  (h2 : t.dividing_segment_length = 13)
  (h3 : t.inner_segment_length = 4) :
  shaded_area t = 65 := by
  sorry

#eval shaded_area { total_area := 117, dividing_segment_length := 13, inner_segment_length := 4 }

end shaded_area_is_65_l119_11907


namespace population_growth_percentage_l119_11995

theorem population_growth_percentage (initial_population final_population : ℕ) 
  (h1 : initial_population = 684)
  (h2 : final_population = 513) :
  ∃ (P : ℝ), 
    (P > 0) ∧ 
    (initial_population : ℝ) * (1 + P / 100) * (1 - 40 / 100) = final_population ∧
    P = 25 := by
  sorry

end population_growth_percentage_l119_11995


namespace concentric_circles_ratio_l119_11957

theorem concentric_circles_ratio (r R : ℝ) (h : r > 0) (H : R > r) :
  π * R^2 - π * r^2 = 4 * (π * r^2) → r / R = 1 / Real.sqrt 5 := by
  sorry

end concentric_circles_ratio_l119_11957


namespace difference_of_squares_example_l119_11906

theorem difference_of_squares_example : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end difference_of_squares_example_l119_11906


namespace manufacturer_buyers_count_l119_11993

theorem manufacturer_buyers_count :
  ∀ (N : ℕ) 
    (cake_buyers muffin_buyers both_buyers : ℕ)
    (prob_neither : ℚ),
  cake_buyers = 50 →
  muffin_buyers = 40 →
  both_buyers = 15 →
  prob_neither = 1/4 →
  (N : ℚ) * prob_neither = N - (cake_buyers + muffin_buyers - both_buyers) →
  N = 100 := by
sorry

end manufacturer_buyers_count_l119_11993


namespace polynomial_symmetry_l119_11937

def P : ℕ → ℝ → ℝ → ℝ → ℝ
  | 0, x, y, z => 1
  | m + 1, x, y, z => (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

theorem polynomial_symmetry (m : ℕ) (x y z : ℝ) :
  P m x y z = P m y x z ∧
  P m x y z = P m x z y ∧
  P m x y z = P m y z x ∧
  P m x y z = P m z x y ∧
  P m x y z = P m z y x :=
by sorry

end polynomial_symmetry_l119_11937


namespace inequality_proof_l119_11964

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) :
  (x^2 + y*z) / Real.sqrt (2*x^2*(y+z)) + 
  (y^2 + z*x) / Real.sqrt (2*y^2*(z+x)) + 
  (z^2 + x*y) / Real.sqrt (2*z^2*(x+y)) ≥ 1 := by
sorry

end inequality_proof_l119_11964


namespace mary_final_card_count_l119_11983

/-- The number of baseball cards Mary has after repairing some torn cards, receiving gifts, and buying new ones. -/
def final_card_count (initial_cards torn_cards repaired_percentage gift_cards bought_cards : ℕ) : ℕ :=
  let repaired_cards := (torn_cards * repaired_percentage) / 100
  let cards_after_repair := initial_cards - torn_cards + repaired_cards
  cards_after_repair + gift_cards + bought_cards

/-- Theorem stating that Mary ends up with 82 baseball cards given the initial conditions. -/
theorem mary_final_card_count :
  final_card_count 18 8 75 26 40 = 82 := by
  sorry

end mary_final_card_count_l119_11983


namespace wand_price_l119_11948

theorem wand_price (price : ℝ) (original : ℝ) : 
  price = 8 ∧ price = (1/8) * original → original = 64 := by
  sorry

end wand_price_l119_11948


namespace female_democrats_count_l119_11989

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 720 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = total / 3 →
  female / 2 = 120 :=
by sorry

end female_democrats_count_l119_11989


namespace geometric_sequence_common_ratio_l119_11973

/-- Given a geometric sequence {a_n} with a₁ = 3, if 4a₁, 2a₂, a₃ form an arithmetic sequence,
    then the common ratio of the geometric sequence is 2. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 3 →                     -- first term condition
  4 * a 1 - 2 * a 2 = 2 * a 2 - a 3 →  -- arithmetic sequence condition
  q = 2 := by
sorry


end geometric_sequence_common_ratio_l119_11973


namespace min_value_x_plus_2y_l119_11902

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  x + 2*y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2/x₀ + 1/y₀ = 1 ∧ x₀ + 2*y₀ = 8 := by
sorry

end min_value_x_plus_2y_l119_11902


namespace function_properties_l119_11974

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 1

-- State the theorem
theorem function_properties (a : ℝ) (h : a > 0) :
  (∃ m : ℝ, m = -1 ∧ ∀ x : ℝ, f a x ≥ m) ∧
  ((∀ x : ℝ, f a x > 0) → a > 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ → f a x₁ < f a x₂) :=
by sorry

end function_properties_l119_11974


namespace least_valid_integer_l119_11934

def is_valid (n : ℕ) : Prop :=
  ∃ (d : ℕ) (m : ℕ), 
    n = 10 * d + m ∧ 
    d ≠ 0 ∧ 
    d < 10 ∧ 
    19 * m = n

theorem least_valid_integer : 
  (is_valid 95) ∧ (∀ n : ℕ, n < 95 → ¬(is_valid n)) :=
sorry

end least_valid_integer_l119_11934


namespace water_added_to_container_l119_11978

/-- Proves that the amount of water added to a container with a capacity of 40 liters,
    initially 40% full, to make it 3/4 full, is 14 liters. -/
theorem water_added_to_container (capacity : ℝ) (initial_percentage : ℝ) (final_fraction : ℝ) :
  capacity = 40 →
  initial_percentage = 0.4 →
  final_fraction = 3/4 →
  (final_fraction * capacity) - (initial_percentage * capacity) = 14 := by
  sorry

end water_added_to_container_l119_11978


namespace odd_function_theorem_l119_11980

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def satisfies_equation (f : ℝ → ℝ) : Prop := ∀ x, f x + f (2 - x) = 4

-- State the theorem
theorem odd_function_theorem (h1 : is_odd f) (h2 : satisfies_equation f) : f 3 = 6 := by
  sorry

end odd_function_theorem_l119_11980


namespace min_bounces_to_height_ball_bounce_problem_l119_11908

def bounce_height (initial_height : ℝ) (bounce_ratio : ℝ) (n : ℕ) : ℝ :=
  initial_height * (bounce_ratio ^ n)

theorem min_bounces_to_height (initial_height bounce_ratio target_height : ℝ) :
  ∃ (n : ℕ), 
    (∀ (k : ℕ), k < n → bounce_height initial_height bounce_ratio k ≥ target_height) ∧
    bounce_height initial_height bounce_ratio n < target_height :=
  sorry

theorem ball_bounce_problem :
  let initial_height := 243
  let bounce_ratio := 2/3
  let target_height := 30
  ∃ (n : ℕ), n = 6 ∧
    (∀ (k : ℕ), k < n → bounce_height initial_height bounce_ratio k ≥ target_height) ∧
    bounce_height initial_height bounce_ratio n < target_height :=
  sorry

end min_bounces_to_height_ball_bounce_problem_l119_11908


namespace sarahs_mean_score_l119_11984

def scores : List ℕ := [78, 80, 85, 87, 90, 95, 100]

theorem sarahs_mean_score 
  (john_score_count : ℕ) 
  (sarah_score_count : ℕ)
  (total_score_count : ℕ)
  (john_mean : ℚ)
  (h1 : john_score_count = 4)
  (h2 : sarah_score_count = 3)
  (h3 : total_score_count = john_score_count + sarah_score_count)
  (h4 : john_mean = 86)
  (h5 : (scores.sum : ℚ) = john_mean * john_score_count + sarah_score_count * sarah_mean) :
  sarah_mean = 90 + 1/3 := by
    sorry

#check sarahs_mean_score

end sarahs_mean_score_l119_11984


namespace stock_value_decrease_l119_11981

theorem stock_value_decrease (F : ℝ) (h1 : F > 0) : 
  let J := 0.9 * F
  let M := J / 1.2
  (F - M) / F * 100 = 28 := by
sorry

end stock_value_decrease_l119_11981


namespace perfect_square_condition_l119_11999

theorem perfect_square_condition (n : ℕ) : 
  (∃ m : ℕ, 2^n + 65 = m^2) ↔ (n = 10 ∨ n = 4) := by
  sorry

end perfect_square_condition_l119_11999


namespace relationship_between_x_and_y_l119_11994

-- Define variables x and y
variable (x y : ℝ)

-- Define the conditions
def condition1 (x y : ℝ) : Prop := 2 * x - 3 * y < x - 1
def condition2 (x y : ℝ) : Prop := 3 * x + 4 * y > 2 * y + 5

-- State the theorem
theorem relationship_between_x_and_y 
  (h1 : condition1 x y) (h2 : condition2 x y) : 
  x < 3 * y - 1 ∧ y > (5 - 3 * x) / 2 := by
  sorry

end relationship_between_x_and_y_l119_11994
