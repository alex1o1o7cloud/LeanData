import Mathlib

namespace seed_germination_percentage_l1047_104720

/-- Calculates the percentage of total seeds that germinated given the number of seeds and germination rates for two plots. -/
theorem seed_germination_percentage 
  (seeds_plot1 : ℕ) 
  (seeds_plot2 : ℕ) 
  (germination_rate_plot1 : ℚ) 
  (germination_rate_plot2 : ℚ) 
  (h1 : seeds_plot1 = 300)
  (h2 : seeds_plot2 = 200)
  (h3 : germination_rate_plot1 = 25 / 100)
  (h4 : germination_rate_plot2 = 40 / 100) :
  (((seeds_plot1 : ℚ) * germination_rate_plot1 + (seeds_plot2 : ℚ) * germination_rate_plot2) / 
   ((seeds_plot1 : ℚ) + (seeds_plot2 : ℚ))) * 100 = 31 := by
  sorry

end seed_germination_percentage_l1047_104720


namespace unique_root_condition_l1047_104706

theorem unique_root_condition (k : ℝ) : 
  (∃! x : ℝ, (1/2) * Real.log (k * x) = Real.log (x + 1)) ↔ (k = 4 ∨ k < 0) :=
by sorry

end unique_root_condition_l1047_104706


namespace inequality_proof_l1047_104747

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) (h3 : d > 0) :
  d / c < (d + 4) / (c + 4) := by
  sorry

end inequality_proof_l1047_104747


namespace percentage_increase_l1047_104781

theorem percentage_increase (x y : ℝ) (h : x > y) :
  (x - y) / y * 100 = 50 → x = 132 ∧ y = 88 := by
  sorry

end percentage_increase_l1047_104781


namespace rectangle_side_ratio_l1047_104737

/-- Given two rectangles A and B, prove the ratio of their sides -/
theorem rectangle_side_ratio 
  (a b c d : ℝ) 
  (h1 : a * b / (c * d) = 0.16) 
  (h2 : a / c = 2 / 5) : 
  b / d = 0.4 := by
  sorry

end rectangle_side_ratio_l1047_104737


namespace parabola_through_point_l1047_104798

/-- The value of 'a' for a parabola y = ax^2 passing through (-1, 2) -/
theorem parabola_through_point (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) → 2 = a * (-1)^2 → a = 2 := by
  sorry

end parabola_through_point_l1047_104798


namespace midpoint_specific_segment_l1047_104718

/-- The midpoint of a line segment in polar coordinates -/
def polar_midpoint (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

theorem midpoint_specific_segment :
  let (r, θ) := polar_midpoint 10 (π/4) 10 (3*π/4)
  r = 5 * Real.sqrt 2 ∧ θ = π/2 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*π := by
  sorry

end midpoint_specific_segment_l1047_104718


namespace basketball_game_score_l1047_104794

/-- Represents the score of a team in a basketball game -/
structure Score :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if a sequence is geometric with common ratio r -/
def isGeometric (s : Score) (r : ℚ) : Prop :=
  s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- Checks if a sequence is arithmetic with common difference d -/
def isArithmetic (s : Score) (d : ℕ) : Prop :=
  s.q2 = s.q1 + d ∧ s.q3 = s.q2 + d ∧ s.q4 = s.q3 + d

/-- The main theorem -/
theorem basketball_game_score 
  (sharks lions : Score) 
  (r : ℚ) 
  (d : ℕ) : 
  sharks.q1 = lions.q1 →  -- Tied at first quarter
  isGeometric sharks r →  -- Sharks scored in geometric sequence
  isArithmetic lions d →  -- Lions scored in arithmetic sequence
  (sharks.q1 + sharks.q2 + sharks.q3 + sharks.q4) = 
    (lions.q1 + lions.q2 + lions.q3 + lions.q4 + 2) →  -- Sharks won by 2 points
  sharks.q1 + sharks.q2 + sharks.q3 + sharks.q4 ≤ 120 →  -- Sharks' total ≤ 120
  lions.q1 + lions.q2 + lions.q3 + lions.q4 ≤ 120 →  -- Lions' total ≤ 120
  sharks.q1 + sharks.q2 + lions.q1 + lions.q2 = 45  -- First half total is 45
  := by sorry

end basketball_game_score_l1047_104794


namespace yeast_growth_20_minutes_l1047_104757

/-- Represents the population growth of yeast cells over time -/
def yeast_population (initial_population : ℕ) (growth_factor : ℕ) (intervals : ℕ) : ℕ :=
  initial_population * growth_factor ^ intervals

theorem yeast_growth_20_minutes :
  let initial_population := 30
  let growth_factor := 3
  let intervals := 5
  yeast_population initial_population growth_factor intervals = 7290 := by sorry

end yeast_growth_20_minutes_l1047_104757


namespace Q_satisfies_conditions_l1047_104770

/-- A polynomial Q(x) with the given properties -/
def Q (x : ℝ) : ℝ := 4 - x + x^2

/-- The theorem stating that Q(x) satisfies the given conditions -/
theorem Q_satisfies_conditions :
  (Q (-2) = 2) ∧
  (∀ x, Q x = Q 0 + Q 1 * x + Q 2 * x^2 + Q 3 * x^3) :=
by sorry

end Q_satisfies_conditions_l1047_104770


namespace remainder_theorem_l1047_104728

theorem remainder_theorem : (9 * 10^20 + 1^20) % 11 = 10 := by
  sorry

end remainder_theorem_l1047_104728


namespace function_properties_l1047_104779

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2 + b * x

-- Define the function g
def g (a b : ℝ) (x : ℝ) : ℝ := f a b x + (3 * a * x^2 + 2 * x + b)

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x, g a b x = -g a b (-x)) →  -- g is an odd function
  (∃ C, ∀ x, f a b x = -1/3 * x^3 + x^2 + C) ∧ 
  (∀ x ∈ Set.Icc 1 2, g (-1/3) 0 x ≤ 4 * Real.sqrt 2 / 3) ∧
  (∀ x ∈ Set.Icc 1 2, g (-1/3) 0 x ≥ 4 / 3) ∧
  (g (-1/3) 0 (Real.sqrt 2) = 4 * Real.sqrt 2 / 3) ∧
  (g (-1/3) 0 2 = 4 / 3) := by
  sorry

end function_properties_l1047_104779


namespace divisible_by_25_l1047_104744

theorem divisible_by_25 (n : ℕ) : ∃ k : ℤ, (2^(n+2) * 3^n + 5*n - 4 : ℤ) = 25 * k := by
  sorry

end divisible_by_25_l1047_104744


namespace complement_intersection_A_B_l1047_104756

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {1, 4, 5} := by sorry

end complement_intersection_A_B_l1047_104756


namespace beidou_usage_scientific_notation_l1047_104748

/-- Expresses a number in scientific notation -/
def scientific_notation (n : ℕ) : ℝ × ℤ :=
  sorry

theorem beidou_usage_scientific_notation :
  scientific_notation 360000000000 = (3.6, 11) :=
sorry

end beidou_usage_scientific_notation_l1047_104748


namespace product_of_successive_numbers_l1047_104782

theorem product_of_successive_numbers : 
  let x : ℝ := 97.49871794028884
  let y : ℝ := x + 1
  abs (x * y - 9603) < 0.001 := by
sorry

end product_of_successive_numbers_l1047_104782


namespace arithmetic_sequence_fourth_term_l1047_104754

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a1 : a 1 = 2)
  (h_a2 : a 2 = 4) :
  a 4 = 8 := by
sorry

end arithmetic_sequence_fourth_term_l1047_104754


namespace expression_values_l1047_104713

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / abs a + b / abs b + c / abs c + d / abs d + (a * b * c * d) / abs (a * b * c * d)
  expr = 5 ∨ expr = 1 ∨ expr = -3 := by
  sorry

end expression_values_l1047_104713


namespace laborer_wage_calculation_l1047_104786

/-- Proves that the daily wage for a laborer is 2.00 rupees given the problem conditions --/
theorem laborer_wage_calculation (total_days : ℕ) (absent_days : ℕ) (fine_per_day : ℚ) (total_received : ℚ) :
  total_days = 25 →
  absent_days = 5 →
  fine_per_day = 1/2 →
  total_received = 75/2 →
  ∃ (daily_wage : ℚ), 
    daily_wage * (total_days - absent_days : ℚ) - (fine_per_day * absent_days) = total_received ∧
    daily_wage = 2 := by
  sorry

#eval (2 : ℚ)

end laborer_wage_calculation_l1047_104786


namespace quadratic_solution_difference_squared_l1047_104709

theorem quadratic_solution_difference_squared :
  ∀ p q : ℝ,
  (2 * p^2 + 11 * p - 21 = 0) →
  (2 * q^2 + 11 * q - 21 = 0) →
  p ≠ q →
  (p - q)^2 = 289/4 := by
sorry

end quadratic_solution_difference_squared_l1047_104709


namespace is_projection_matrix_l1047_104776

def projection_matrix (A : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  A * A = A

theorem is_projection_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![11/12, 12/25; 12/25, 13/25]
  projection_matrix A := by
  sorry

end is_projection_matrix_l1047_104776


namespace jens_son_age_l1047_104773

theorem jens_son_age :
  ∀ (sons_age : ℕ),
  (41 : ℕ) = 25 + sons_age →  -- Jen was 25 when her son was born, and she's 41 now
  (41 : ℕ) = 3 * sons_age - 7 →  -- Jen's age is 7 less than 3 times her son's age
  sons_age = 16 :=
by
  sorry

end jens_son_age_l1047_104773


namespace bearded_male_percentage_is_40_percent_l1047_104727

/-- Represents the data for Scrabble champions over a period of years -/
structure ScrabbleChampionData where
  total_years : ℕ
  women_percentage : ℚ
  champions_per_year : ℕ
  bearded_men : ℕ

/-- Calculates the percentage of male Scrabble champions with beards -/
def bearded_male_percentage (data : ScrabbleChampionData) : ℚ :=
  sorry

/-- Theorem stating that given the specific conditions, 
    the percentage of male Scrabble champions with beards is 40% -/
theorem bearded_male_percentage_is_40_percent 
  (data : ScrabbleChampionData)
  (h1 : data.total_years = 25)
  (h2 : data.women_percentage = 60 / 100)
  (h3 : data.champions_per_year = 1)
  (h4 : data.bearded_men = 4) :
  bearded_male_percentage data = 40 / 100 :=
sorry

end bearded_male_percentage_is_40_percent_l1047_104727


namespace books_unchanged_l1047_104767

/-- Represents the number of items before and after a garage sale. -/
structure GarageSale where
  initial_books : ℕ
  initial_pens : ℕ
  sold_pens : ℕ
  final_pens : ℕ

/-- Theorem stating that the number of books remains unchanged after the garage sale. -/
theorem books_unchanged (sale : GarageSale) 
  (h1 : sale.initial_books = 51)
  (h2 : sale.initial_pens = 106)
  (h3 : sale.sold_pens = 92)
  (h4 : sale.final_pens = 14)
  (h5 : sale.initial_pens - sale.sold_pens = sale.final_pens) :
  sale.initial_books = 51 := by
  sorry

end books_unchanged_l1047_104767


namespace sophias_book_length_l1047_104722

theorem sophias_book_length (P : ℕ) : 
  (2 : ℚ) / 3 * P = (1 : ℚ) / 3 * P + 90 → P = 270 := by
  sorry

end sophias_book_length_l1047_104722


namespace discount_is_twenty_percent_l1047_104797

/-- Calculates the discount percentage given the original price, quantity, tax rate, and final price --/
def calculate_discount_percentage (original_price quantity : ℕ) (tax_rate final_price : ℚ) : ℚ :=
  let discounted_price := final_price / (1 + tax_rate) / quantity
  let discount_amount := original_price - discounted_price
  (discount_amount / original_price) * 100

/-- The discount percentage is 20% given the problem conditions --/
theorem discount_is_twenty_percent :
  calculate_discount_percentage 45 10 (1/10) 396 = 20 := by
  sorry

end discount_is_twenty_percent_l1047_104797


namespace divisible_by_five_l1047_104702

theorem divisible_by_five (n : ℕ) : ∃ k : ℤ, (n^5 : ℤ) + 4*n = 5*k := by
  sorry

end divisible_by_five_l1047_104702


namespace greg_total_distance_l1047_104724

/-- The total distance Greg travels given his individual trip distances -/
theorem greg_total_distance (d1 d2 d3 : ℝ) 
  (h1 : d1 = 30) -- Distance from workplace to farmer's market
  (h2 : d2 = 20) -- Distance from farmer's market to friend's house
  (h3 : d3 = 25) -- Distance from friend's house to home
  : d1 + d2 + d3 = 75 := by sorry

end greg_total_distance_l1047_104724


namespace intersection_of_A_and_B_l1047_104717

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 3}
def B : Set ℝ := {x | x - 2 > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l1047_104717


namespace polynomial_problem_l1047_104788

/-- Given polynomial P = 2(ax-3) - 3(bx+5) -/
def P (a b x : ℝ) : ℝ := 2*(a*x - 3) - 3*(b*x + 5)

theorem polynomial_problem (a b : ℝ) (h1 : P a b 2 = -31) (h2 : a + b = 0) :
  (a = -1 ∧ b = 1) ∧ 
  (∀ x : ℤ, P a b x > 0 → x ≤ -5) ∧
  (P a b (-5 : ℝ) > 0) :=
sorry

end polynomial_problem_l1047_104788


namespace complex_number_equality_l1047_104739

theorem complex_number_equality (z : ℂ) : 
  (Complex.abs (z - 2) = Complex.abs (z + 4) ∧ 
   Complex.abs (z - 2) = Complex.abs (z - 2*I)) ↔ 
  z = -1 - I :=
sorry

end complex_number_equality_l1047_104739


namespace maintenance_team_journey_l1047_104762

def walking_records : List Int := [15, -2, 5, -1, 10, -3, -2, 12, 4, -5, 6]
def fuel_consumption_rate : ℝ := 3
def initial_fuel : ℝ := 180

theorem maintenance_team_journey :
  let net_distance : Int := walking_records.sum
  let total_distance : ℕ := walking_records.map (Int.natAbs) |>.sum
  let total_fuel_consumption : ℝ := (total_distance : ℝ) * fuel_consumption_rate
  let fuel_needed : ℝ := total_fuel_consumption - initial_fuel
  (net_distance = 39) ∧ 
  (total_distance = 65) ∧ 
  (total_fuel_consumption = 195) ∧ 
  (fuel_needed = 15) := by
sorry

end maintenance_team_journey_l1047_104762


namespace three_greater_than_negative_five_l1047_104783

theorem three_greater_than_negative_five :
  3 > -5 :=
by
  -- Proof goes here
  sorry

end three_greater_than_negative_five_l1047_104783


namespace oranges_thrown_away_l1047_104758

theorem oranges_thrown_away (initial_oranges : ℕ) (new_oranges : ℕ) (final_oranges : ℕ)
  (h1 : initial_oranges = 50)
  (h2 : new_oranges = 24)
  (h3 : final_oranges = 34)
  : initial_oranges - (initial_oranges - new_oranges + final_oranges) = 40 := by
  sorry

end oranges_thrown_away_l1047_104758


namespace function_inequality_l1047_104712

open Real

theorem function_inequality (f : ℝ → ℝ) (h : ∀ x > 0, Real.sqrt x * (deriv f x) < (1 / 2)) :
  f 9 - 1 < f 4 ∧ f 4 < f 1 + 1 := by
  sorry

end function_inequality_l1047_104712


namespace equation_solution_l1047_104764

theorem equation_solution : 
  ∃! x : ℚ, (x - 27) / 3 = (3 * x + 6) / 8 ∧ x = -234 :=
by sorry

end equation_solution_l1047_104764


namespace min_abs_z_on_circle_l1047_104791

theorem min_abs_z_on_circle (z : ℂ) (h : Complex.abs (z - (1 + Complex.I)) = 1) :
  ∃ (w : ℂ), Complex.abs (w - (1 + Complex.I)) = 1 ∧
             Complex.abs w = Real.sqrt 2 - 1 ∧
             ∀ (v : ℂ), Complex.abs (v - (1 + Complex.I)) = 1 → Complex.abs w ≤ Complex.abs v :=
by sorry

end min_abs_z_on_circle_l1047_104791


namespace board_coverage_problem_boards_l1047_104769

/-- Represents a rectangular board --/
structure Board where
  rows : ℕ
  cols : ℕ

/-- Checks if a board can be completely covered by dominoes --/
def canCoverWithDominoes (b : Board) : Prop :=
  (b.rows * b.cols) % 2 = 0

/-- Theorem stating that a board can be covered iff its area is even --/
theorem board_coverage (b : Board) :
  canCoverWithDominoes b ↔ (b.rows * b.cols) % 2 = 0 := by sorry

/-- Function to check if a board can be covered --/
def checkBoard (b : Board) : Bool :=
  (b.rows * b.cols) % 2 = 0

/-- Theorem for the specific boards in the problem --/
theorem problem_boards :
  (¬ checkBoard ⟨5, 5⟩) ∧
  (checkBoard ⟨4, 6⟩) ∧
  (¬ checkBoard ⟨3, 7⟩) ∧
  (checkBoard ⟨5, 6⟩) ∧
  (checkBoard ⟨3, 8⟩) := by sorry

end board_coverage_problem_boards_l1047_104769


namespace reciprocal_product_theorem_l1047_104795

theorem reciprocal_product_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 6 * x * y) : (1 / x) * (1 / y) = 1 / 6 := by
  sorry

end reciprocal_product_theorem_l1047_104795


namespace octahedron_triangle_count_l1047_104765

/-- A regular octahedron -/
structure RegularOctahedron where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 6 × Fin 6)
  vertex_count : vertices.card = 6
  edge_count : edges.card = 12
  edge_validity : ∀ e ∈ edges, e.1 ≠ e.2 ∧ e.1 ∈ vertices ∧ e.2 ∈ vertices

/-- A triangle on the octahedron -/
structure OctahedronTriangle (O : RegularOctahedron) where
  vertices : Finset (Fin 6)
  vertex_count : vertices.card = 3
  vertex_validity : vertices ⊆ O.vertices
  edge_shared : ∃ e ∈ O.edges, (e.1 ∈ vertices ∧ e.2 ∈ vertices)

/-- The set of all valid triangles on the octahedron -/
def validTriangles (O : RegularOctahedron) : Set (OctahedronTriangle O) :=
  {t | t.vertices ⊆ O.vertices ∧ ∃ e ∈ O.edges, (e.1 ∈ t.vertices ∧ e.2 ∈ t.vertices)}

theorem octahedron_triangle_count (O : RegularOctahedron) :
  (validTriangles O).ncard = 12 := by
  sorry

end octahedron_triangle_count_l1047_104765


namespace factor_polynomial_l1047_104774

theorem factor_polynomial (x : ℝ) : 66 * x^6 - 231 * x^12 = 33 * x^6 * (2 - 7 * x^6) := by
  sorry

end factor_polynomial_l1047_104774


namespace power_multiplication_l1047_104711

theorem power_multiplication (x : ℝ) : x^5 * x^6 = x^11 := by
  sorry

end power_multiplication_l1047_104711


namespace frog_distribution_l1047_104784

/-- Represents the three lakes in the problem -/
inductive Lake
| Crystal
| Lassie
| Emerald

/-- Represents the three frog species in the problem -/
inductive Species
| A
| B
| C

/-- The number of frogs of a given species in a given lake -/
def frog_count (l : Lake) (s : Species) : ℕ :=
  match l, s with
  | Lake.Lassie, Species.A => 45
  | Lake.Lassie, Species.B => 35
  | Lake.Lassie, Species.C => 25
  | Lake.Crystal, Species.A => 36
  | Lake.Crystal, Species.B => 39
  | Lake.Crystal, Species.C => 25
  | Lake.Emerald, Species.A => 59
  | Lake.Emerald, Species.B => 70
  | Lake.Emerald, Species.C => 38

/-- The total number of frogs of a given species across all lakes -/
def total_frogs (s : Species) : ℕ :=
  (frog_count Lake.Crystal s) + (frog_count Lake.Lassie s) + (frog_count Lake.Emerald s)

theorem frog_distribution :
  (total_frogs Species.A = 140) ∧
  (total_frogs Species.B = 144) ∧
  (total_frogs Species.C = 88) :=
by sorry


end frog_distribution_l1047_104784


namespace f_sum_over_sum_positive_l1047_104775

noncomputable def f (x : ℝ) : ℝ := x^3 - Real.log (Real.sqrt (x^2 + 1) - x)

theorem f_sum_over_sum_positive (a b : ℝ) (h : a + b ≠ 0) :
  (f a + f b) / (a + b) > 0 := by
  sorry

end f_sum_over_sum_positive_l1047_104775


namespace units_digit_of_fraction_l1047_104746

def numerator : ℕ := 25 * 26 * 27 * 28 * 29 * 30
def denominator : ℕ := 1250

theorem units_digit_of_fraction : (numerator / denominator) % 10 = 2 := by sorry

end units_digit_of_fraction_l1047_104746


namespace calendar_box_sum_divisible_by_four_l1047_104725

/-- Represents a box of four numbers in a 7-column calendar --/
structure CalendarBox where
  top_right : ℕ
  top_left : ℕ
  bottom_left : ℕ
  bottom_right : ℕ

/-- Creates a calendar box given the top right number --/
def make_calendar_box (a : ℕ) : CalendarBox :=
  { top_right := a
  , top_left := a - 1
  , bottom_left := a + 6
  , bottom_right := a + 7 }

/-- The sum of numbers in a calendar box --/
def box_sum (box : CalendarBox) : ℕ :=
  box.top_right + box.top_left + box.bottom_left + box.bottom_right

/-- Theorem: The sum of numbers in any calendar box is divisible by 4 --/
theorem calendar_box_sum_divisible_by_four (a : ℕ) :
  4 ∣ box_sum (make_calendar_box a) := by
  sorry

end calendar_box_sum_divisible_by_four_l1047_104725


namespace min_sum_of_product_l1047_104760

theorem min_sum_of_product (a b c : ℕ+) (h : a * b * c = 3960) :
  ∃ (x y z : ℕ+), x * y * z = 3960 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 150 :=
sorry

end min_sum_of_product_l1047_104760


namespace solution_satisfies_system_l1047_104700

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop := x^2 + 25*y + 19*z = -471
def equation2 (x y z : ℝ) : Prop := y^2 + 23*x + 21*z = -397
def equation3 (x y z : ℝ) : Prop := z^2 + 21*x + 21*y = -545

-- Theorem statement
theorem solution_satisfies_system :
  equation1 (-22) (-23) (-20) ∧
  equation2 (-22) (-23) (-20) ∧
  equation3 (-22) (-23) (-20) := by
  sorry

end solution_satisfies_system_l1047_104700


namespace largest_divisor_n_plus_10_divisibility_condition_l1047_104752

theorem largest_divisor_n_plus_10 :
  ∀ n : ℕ, n > 0 → (n + 10) ∣ (n^3 + 2011) → n ≤ 1001 :=
by sorry

theorem divisibility_condition :
  (1001 + 10) ∣ (1001^3 + 2011) :=
by sorry

end largest_divisor_n_plus_10_divisibility_condition_l1047_104752


namespace expenditure_ratio_l1047_104792

-- Define the incomes and expenditures
def uma_income : ℚ := 20000
def bala_income : ℚ := 15000
def uma_expenditure : ℚ := 15000
def bala_expenditure : ℚ := 10000
def savings : ℚ := 5000

-- Define the theorem
theorem expenditure_ratio :
  (uma_income / bala_income = 4 / 3) →
  (uma_income = 20000) →
  (uma_income - uma_expenditure = savings) →
  (bala_income - bala_expenditure = savings) →
  (uma_expenditure / bala_expenditure = 3 / 2) := by
  sorry

end expenditure_ratio_l1047_104792


namespace sphere_water_volume_l1047_104729

def hemisphere_volume : ℝ := 4
def num_hemispheres : ℕ := 2749

theorem sphere_water_volume :
  let total_volume := (num_hemispheres : ℝ) * hemisphere_volume
  total_volume = 10996 := by sorry

end sphere_water_volume_l1047_104729


namespace barbed_wire_rate_l1047_104743

/-- The rate of drawing barbed wire per meter given a square field's area, gate widths, and total cost --/
theorem barbed_wire_rate (field_area : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ) : 
  field_area = 3136 →
  gate_width = 1 →
  num_gates = 2 →
  total_cost = 2331 →
  (total_cost / (4 * Real.sqrt field_area - num_gates * gate_width)) = 10.5 := by
  sorry

#check barbed_wire_rate

end barbed_wire_rate_l1047_104743


namespace tom_trout_count_l1047_104761

/-- Given that Melanie catches 8 trout and Tom catches 2 times as many trout as Melanie,
    prove that Tom catches 16 trout. -/
theorem tom_trout_count (melanie_trout : ℕ) (tom_multiplier : ℕ) 
    (h1 : melanie_trout = 8)
    (h2 : tom_multiplier = 2) : 
  tom_multiplier * melanie_trout = 16 := by
  sorry

end tom_trout_count_l1047_104761


namespace arrangements_eq_36_l1047_104778

/-- The number of students in the row -/
def n : ℕ := 5

/-- A function that calculates the number of arrangements given the conditions -/
def arrangements (n : ℕ) : ℕ :=
  let positions := n - 1  -- Possible positions for A (excluding ends)
  let pairs := 2  -- A and B can be arranged in 2 ways next to each other
  let others := n - 2  -- Remaining students to arrange
  positions * pairs * (others.factorial)

/-- The theorem stating that the number of arrangements is 36 -/
theorem arrangements_eq_36 : arrangements n = 36 := by
  sorry

end arrangements_eq_36_l1047_104778


namespace marks_deposit_is_88_l1047_104750

-- Define Mark's deposit
def mark_deposit : ℕ := 88

-- Define Bryan's deposit in terms of Mark's
def bryan_deposit : ℕ := 5 * mark_deposit - 40

-- Theorem to prove
theorem marks_deposit_is_88 : mark_deposit = 88 := by
  sorry

end marks_deposit_is_88_l1047_104750


namespace sample_size_definition_l1047_104723

/-- Represents a population of students' exam scores -/
structure Population where
  scores : Set ℝ

/-- Represents a sample drawn from a population -/
structure Sample where
  elements : Finset ℝ

/-- Simple random sampling function -/
def simpleRandomSampling (pop : Population) (n : ℕ) : Sample :=
  sorry

theorem sample_size_definition 
  (pop : Population) 
  (sample : Sample) 
  (n : ℕ) 
  (h1 : sample = simpleRandomSampling pop n) 
  (h2 : n = 100) : 
  n = Finset.card sample.elements :=
sorry

end sample_size_definition_l1047_104723


namespace lines_skew_when_one_parallel_to_plane_other_in_plane_l1047_104707

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line in 3D space
  -- (We'll leave this abstract for now)

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane in 3D space
  -- (We'll leave this abstract for now)

/-- Proposition that a line is parallel to a plane -/
def is_parallel_to_plane (l : Line3D) (p : Plane3D) : Prop :=
  -- Define what it means for a line to be parallel to a plane
  sorry

/-- Proposition that a line is contained within a plane -/
def is_contained_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  -- Define what it means for a line to be contained in a plane
  sorry

/-- Proposition that two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Define what it means for two lines to be skew
  sorry

/-- Theorem statement -/
theorem lines_skew_when_one_parallel_to_plane_other_in_plane 
  (a b : Line3D) (α : Plane3D) 
  (h1 : is_parallel_to_plane a α) 
  (h2 : is_contained_in_plane b α) : 
  are_skew a b :=
sorry

end lines_skew_when_one_parallel_to_plane_other_in_plane_l1047_104707


namespace probability_two_hits_l1047_104753

def probability_at_least_one_hit : ℚ := 65/81

def number_of_shots : ℕ := 4

def probability_single_hit : ℚ := 1/3

theorem probability_two_hits :
  (1 - probability_at_least_one_hit) = (1 - probability_single_hit) ^ number_of_shots →
  Nat.choose number_of_shots 2 * probability_single_hit^2 * (1 - probability_single_hit)^2 = 8/27 := by
sorry

end probability_two_hits_l1047_104753


namespace kendra_shirts_theorem_l1047_104749

/-- Represents the number of shirts Kendra needs for various activities --/
structure ShirtRequirements where
  weekdaySchool : Nat
  afterSchoolClub : Nat
  spiritDay : Nat
  saturday : Nat
  sunday : Nat
  familyReunion : Nat

/-- Calculates the total number of shirts needed for a given number of weeks --/
def totalShirtsNeeded (req : ShirtRequirements) (weeks : Nat) : Nat :=
  (req.weekdaySchool + req.afterSchoolClub + req.spiritDay + req.saturday + req.sunday) * weeks + req.familyReunion

/-- Theorem stating that Kendra needs 61 shirts for 4 weeks --/
theorem kendra_shirts_theorem (req : ShirtRequirements) 
    (h1 : req.weekdaySchool = 5)
    (h2 : req.afterSchoolClub = 3)
    (h3 : req.spiritDay = 1)
    (h4 : req.saturday = 3)
    (h5 : req.sunday = 3)
    (h6 : req.familyReunion = 1) :
  totalShirtsNeeded req 4 = 61 := by
  sorry

#eval totalShirtsNeeded ⟨5, 3, 1, 3, 3, 1⟩ 4

end kendra_shirts_theorem_l1047_104749


namespace smallest_two_digit_with_digit_product_12_l1047_104703

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 12 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 12 → n ≤ m :=
by sorry

end smallest_two_digit_with_digit_product_12_l1047_104703


namespace factory_production_equation_l1047_104705

/-- Represents the production equation for a factory with monthly growth rate --/
theorem factory_production_equation (april_production : ℝ) (quarter_production : ℝ) (x : ℝ) :
  april_production = 500000 →
  quarter_production = 1820000 →
  50 + 50 * (1 + x) + 50 * (1 + x)^2 = 182 :=
by sorry

end factory_production_equation_l1047_104705


namespace two_car_garage_count_l1047_104731

theorem two_car_garage_count (total : ℕ) (pool : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 90 → pool = 40 → both = 35 → neither = 35 → 
  ∃ (garage : ℕ), garage = 50 ∧ garage + pool - both = total - neither :=
by
  sorry

end two_car_garage_count_l1047_104731


namespace triangle_inequality_l1047_104796

/-- Given a triangle ABC with point P inside, prove the inequality involving
    sides and distances from P to the sides. -/
theorem triangle_inequality (a b c d₁ d₂ d₃ S_ABC : ℝ) 
    (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
    (h₄ : d₁ > 0) (h₅ : d₂ > 0) (h₆ : d₃ > 0)
    (h₇ : S_ABC > 0)
    (h₈ : S_ABC = (1/2) * (a * d₁ + b * d₂ + c * d₃)) :
  (a / d₁) + (b / d₂) + (c / d₃) ≥ (a + b + c)^2 / (2 * S_ABC) := by
  sorry

end triangle_inequality_l1047_104796


namespace kebul_family_children_l1047_104730

/-- Represents a family with children -/
structure Family where
  total_members : ℕ
  father_age : ℕ
  average_age : ℚ
  average_age_without_father : ℚ

/-- Calculates the number of children in a family -/
def number_of_children (f : Family) : ℕ :=
  f.total_members - 2

/-- Theorem stating the number of children in the Kebul family -/
theorem kebul_family_children (f : Family) 
  (h1 : f.average_age = 18)
  (h2 : f.father_age = 38)
  (h3 : f.average_age_without_father = 14) :
  number_of_children f = 4 := by
  sorry

#eval number_of_children { total_members := 6, father_age := 38, average_age := 18, average_age_without_father := 14 }

end kebul_family_children_l1047_104730


namespace line_equidistant_points_l1047_104715

/-- Given a line passing through (4, 4) with slope 0.5, equidistant from (0, 2) and (A, 8), prove A = -3 -/
theorem line_equidistant_points (A : ℝ) : 
  let line_point : ℝ × ℝ := (4, 4)
  let line_slope : ℝ := 0.5
  let P : ℝ × ℝ := (0, 2)
  let Q : ℝ × ℝ := (A, 8)
  let midpoint : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let perpendicular_slope : ℝ := -1 / ((Q.2 - P.2) / (Q.1 - P.1))
  (line_slope = perpendicular_slope) ∧ 
  (midpoint.2 - line_point.2 = line_slope * (midpoint.1 - line_point.1)) →
  A = -3 := by
sorry

end line_equidistant_points_l1047_104715


namespace net_profit_calculation_l1047_104785

def calculate_net_profit (basil_seed_cost mint_seed_cost zinnia_seed_cost : ℚ)
  (potting_soil_cost packaging_cost : ℚ)
  (sellers_fee_rate sales_tax_rate : ℚ)
  (basil_yield mint_yield zinnia_yield : ℕ)
  (basil_germination mint_germination zinnia_germination : ℚ)
  (healthy_basil_price healthy_mint_price healthy_zinnia_price : ℚ)
  (small_basil_price small_mint_price small_zinnia_price : ℚ)
  (healthy_basil_sold small_basil_sold : ℕ)
  (healthy_mint_sold small_mint_sold : ℕ)
  (healthy_zinnia_sold small_zinnia_sold : ℕ) : ℚ :=
  let total_revenue := 
    healthy_basil_price * healthy_basil_sold + small_basil_price * small_basil_sold +
    healthy_mint_price * healthy_mint_sold + small_mint_price * small_mint_sold +
    healthy_zinnia_price * healthy_zinnia_sold + small_zinnia_price * small_zinnia_sold
  let total_expenses := 
    basil_seed_cost + mint_seed_cost + zinnia_seed_cost + potting_soil_cost + packaging_cost
  let sellers_fee := sellers_fee_rate * total_revenue
  let sales_tax := sales_tax_rate * total_revenue
  total_revenue - total_expenses - sellers_fee - sales_tax

theorem net_profit_calculation : 
  calculate_net_profit 2 3 7 15 5 (1/10) (1/20)
    20 15 10 (4/5) (3/4) (7/10)
    5 6 10 3 4 7
    12 8 10 4 5 2 = 158.4 := by sorry

end net_profit_calculation_l1047_104785


namespace becky_eddie_age_ratio_l1047_104763

/-- Given the ages of Eddie, Irene, and the relationship between Irene and Becky's ages,
    prove that the ratio of Becky's age to Eddie's age is 1:4. -/
theorem becky_eddie_age_ratio 
  (eddie_age : ℕ) 
  (irene_age : ℕ) 
  (becky_age : ℕ) 
  (h1 : eddie_age = 92) 
  (h2 : irene_age = 46) 
  (h3 : irene_age = 2 * becky_age) : 
  becky_age * 4 = eddie_age := by
  sorry

#check becky_eddie_age_ratio

end becky_eddie_age_ratio_l1047_104763


namespace chicken_rabbit_problem_l1047_104726

/-- The number of chickens and rabbits in the cage satisfying the given conditions -/
theorem chicken_rabbit_problem :
  ∃ (chickens rabbits : ℕ),
    chickens + rabbits = 35 ∧
    2 * chickens + 4 * rabbits = 94 ∧
    chickens = 23 ∧
    rabbits = 12 := by
  sorry

end chicken_rabbit_problem_l1047_104726


namespace product_factor_proof_l1047_104738

theorem product_factor_proof (w : ℕ+) (h1 : 2^5 ∣ (936 * w)) (h2 : 3^3 ∣ (936 * w)) (h3 : w ≥ 144) :
  ∃ x : ℕ, 12^x ∣ (936 * w) ∧ ∀ y : ℕ, 12^y ∣ (936 * w) → y ≤ 2 :=
sorry

end product_factor_proof_l1047_104738


namespace probability_qualified_bulb_factory_A_l1047_104768

/-- The probability of purchasing a qualified light bulb produced by Factory A from the market -/
theorem probability_qualified_bulb_factory_A
  (factory_A_production_rate : ℝ)
  (factory_B_production_rate : ℝ)
  (factory_A_pass_rate : ℝ)
  (factory_B_pass_rate : ℝ)
  (h1 : factory_A_production_rate = 0.7)
  (h2 : factory_B_production_rate = 0.3)
  (h3 : factory_A_pass_rate = 0.95)
  (h4 : factory_B_pass_rate = 0.8)
  (h5 : factory_A_production_rate + factory_B_production_rate = 1) :
  factory_A_production_rate * factory_A_pass_rate = 0.665 := by
  sorry


end probability_qualified_bulb_factory_A_l1047_104768


namespace number_divided_by_five_l1047_104759

theorem number_divided_by_five (x : ℝ) : x - 5 = 35 → x / 5 = 8 := by
  sorry

end number_divided_by_five_l1047_104759


namespace box_balls_problem_l1047_104732

theorem box_balls_problem (balls : ℕ) (x : ℕ) : 
  balls = 57 → 
  (balls - x = 70 - balls) →
  x = 44 := by
  sorry

end box_balls_problem_l1047_104732


namespace girls_with_rulers_l1047_104741

theorem girls_with_rulers (total_students : ℕ) (students_with_rulers : ℕ) (boys_with_set_squares : ℕ) (total_girls : ℕ) :
  total_students = 50 →
  students_with_rulers = 28 →
  boys_with_set_squares = 14 →
  total_girls = 31 →
  (total_students - students_with_rulers) = boys_with_set_squares + (total_girls - (total_students - students_with_rulers - boys_with_set_squares)) →
  total_girls - (total_students - students_with_rulers - boys_with_set_squares) = 23 :=
by sorry

end girls_with_rulers_l1047_104741


namespace henrysFriendMoney_l1047_104793

/-- Calculates the amount of money Henry's friend has -/
def friendsMoney (henryInitial : ℕ) (henryEarned : ℕ) (totalCombined : ℕ) : ℕ :=
  totalCombined - (henryInitial + henryEarned)

/-- Theorem: Henry's friend has 13 dollars -/
theorem henrysFriendMoney : friendsMoney 5 2 20 = 13 := by
  sorry

end henrysFriendMoney_l1047_104793


namespace last_two_digits_sum_factorials_50_l1047_104721

/-- The last two digits of a natural number -/
def lastTwoDigits (n : ℕ) : ℕ := n % 100

/-- The sum of factorials from 1 to n -/
def sumFactorials (n : ℕ) : ℕ := (List.range n).map Nat.factorial |>.sum

/-- The last two digits of the sum of factorials from 1 to 50 are 13 -/
theorem last_two_digits_sum_factorials_50 :
  lastTwoDigits (sumFactorials 50) = 13 := by
  sorry

end last_two_digits_sum_factorials_50_l1047_104721


namespace intersection_of_A_and_B_l1047_104719

def A : Set ℝ := {x | x^2 - x = 0}
def B : Set ℝ := {y | y^2 + y = 0}

theorem intersection_of_A_and_B : A ∩ B = {0} := by sorry

end intersection_of_A_and_B_l1047_104719


namespace perfect_square_condition_l1047_104708

theorem perfect_square_condition (p : ℕ) : 
  Nat.Prime p → (∃ (x : ℕ), 7^p - p - 16 = x^2) ↔ p = 3 :=
by sorry

end perfect_square_condition_l1047_104708


namespace marbles_given_proof_l1047_104740

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := sorry

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := 143

/-- The number of marbles Connie has left -/
def remaining_marbles : ℕ := 70

/-- Theorem stating that the number of marbles given is equal to the difference
between the initial number of marbles and the remaining marbles -/
theorem marbles_given_proof : 
  marbles_given = initial_marbles - remaining_marbles :=
by sorry

end marbles_given_proof_l1047_104740


namespace geometric_sequence_example_l1047_104734

def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

theorem geometric_sequence_example :
  is_geometric_sequence 3 (-3 * Real.sqrt 3) 9 := by
  sorry

end geometric_sequence_example_l1047_104734


namespace largest_c_value_l1047_104799

theorem largest_c_value (c d e : ℤ) 
  (eq : 5 * c + (d - 12)^2 + e^3 = 235)
  (c_lt_d : c < d) : 
  c ≤ 22 ∧ ∃ (c' d' e' : ℤ), c' = 22 ∧ c' < d' ∧ 5 * c' + (d' - 12)^2 + e'^3 = 235 :=
sorry

end largest_c_value_l1047_104799


namespace chord_count_for_concentric_circles_l1047_104701

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle formed by two adjacent chords is 60°, then exactly 3 such chords are needed to complete a full circle. -/
theorem chord_count_for_concentric_circles (angle : ℝ) (n : ℕ) : 
  angle = 60 → n * angle = 360 → n = 3 := by sorry

end chord_count_for_concentric_circles_l1047_104701


namespace cyclic_inequality_l1047_104777

theorem cyclic_inequality (x y z m n : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hm : m > 0) (hn : n > 0)
  (hmn : m + n ≥ 2) : 
  x * Real.sqrt (y * z * (x + m * y) * (x + n * z)) + 
  y * Real.sqrt (x * z * (y + m * x) * (y + n * z)) + 
  z * Real.sqrt (x * y * (z + m * x) * (z + n * y)) ≤ 
  (3 * (m + n) / 8) * (x + y) * (y + z) * (z + x) := by
  sorry

end cyclic_inequality_l1047_104777


namespace largest_multiple_of_nine_less_than_hundred_l1047_104771

theorem largest_multiple_of_nine_less_than_hundred : 
  ∃ (n : ℕ), n * 9 = 99 ∧ 
  ∀ (m : ℕ), m * 9 < 100 → m * 9 ≤ 99 := by
  sorry

end largest_multiple_of_nine_less_than_hundred_l1047_104771


namespace distribution_plans_count_l1047_104716

-- Define the number of awards and schools
def total_awards : ℕ := 7
def num_schools : ℕ := 5
def min_awards_per_special_school : ℕ := 2
def num_special_schools : ℕ := 2

-- Define the function to calculate the number of distribution plans
def num_distribution_plans : ℕ :=
  Nat.choose (total_awards - min_awards_per_special_school * num_special_schools + num_schools - 1) (num_schools - 1)

-- Theorem statement
theorem distribution_plans_count :
  num_distribution_plans = 35 :=
sorry

end distribution_plans_count_l1047_104716


namespace constant_term_implies_n_12_l1047_104745

/-- The general term formula for the expansion of (√x - 2/x)^n -/
def generalTerm (n : ℕ) (r : ℕ) : ℚ → ℚ := 
  λ x => (n.choose r) * (-2)^r * x^((n - 3*r) / 2)

/-- The condition that the 5th term (r = 4) is the constant term -/
def fifthTermIsConstant (n : ℕ) : Prop :=
  (n - 3*4) / 2 = 0

theorem constant_term_implies_n_12 : 
  ∀ n : ℕ, fifthTermIsConstant n → n = 12 := by
  sorry

end constant_term_implies_n_12_l1047_104745


namespace tan_pi_twelve_l1047_104710

theorem tan_pi_twelve : Real.tan (π / 12) = 2 - Real.sqrt 3 := by
  sorry

end tan_pi_twelve_l1047_104710


namespace scientific_notation_conversion_l1047_104736

theorem scientific_notation_conversion :
  (380180000000 : ℝ) = 3.8018 * (10 : ℝ)^11 :=
by sorry

end scientific_notation_conversion_l1047_104736


namespace B_power_101_l1047_104780

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, -1, 0],
    ![1,  0, 0],
    ![0,  0, 0]]

theorem B_power_101 : B ^ 101 = B := by sorry

end B_power_101_l1047_104780


namespace f_negative_five_halves_l1047_104714

def f (x : ℝ) : ℝ := sorry

theorem f_negative_five_halves :
  (∀ x, f (-x) = -f x) →                     -- f is odd
  (∀ x, f (x + 2) = f x) →                   -- f has period 2
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2*x*(1 - x)) → -- f(x) = 2x(1-x) for 0 ≤ x ≤ 1
  f (-5/2) = -1/2 := by sorry

end f_negative_five_halves_l1047_104714


namespace inequality_proof_l1047_104742

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b) / (a * b + a^5 + b^5) + (b * c) / (b * c + b^5 + c^5) + (c * a) / (c * a + c^5 + a^5) ≤ 1 := by
sorry

end inequality_proof_l1047_104742


namespace prob_at_least_one_black_is_five_sixths_l1047_104751

/-- The number of white balls in the pouch -/
def num_white_balls : ℕ := 2

/-- The number of black balls in the pouch -/
def num_black_balls : ℕ := 2

/-- The total number of balls in the pouch -/
def total_balls : ℕ := num_white_balls + num_black_balls

/-- The number of balls drawn from the pouch -/
def drawn_balls : ℕ := 2

/-- The probability of drawing at least one black ball -/
def prob_at_least_one_black : ℚ := 5/6

theorem prob_at_least_one_black_is_five_sixths :
  prob_at_least_one_black = 1 - (num_white_balls.choose drawn_balls : ℚ) / (total_balls.choose drawn_balls : ℚ) :=
by sorry

end prob_at_least_one_black_is_five_sixths_l1047_104751


namespace assignment_count_assignment_count_proof_l1047_104733

theorem assignment_count : ℕ → Prop :=
  fun total_assignments =>
    ∃ (initial_hours : ℕ),
      -- Initial plan: 6 assignments per hour for initial_hours
      6 * initial_hours = total_assignments ∧
      -- New plan: 2 hours at 6 per hour, then 8 per hour for (initial_hours - 5) hours
      2 * 6 + 8 * (initial_hours - 5) = total_assignments ∧
      -- Total assignments is 84
      total_assignments = 84

-- The proof of this theorem would show that the conditions are satisfied
-- and the total number of assignments is indeed 84
theorem assignment_count_proof : assignment_count 84 := by
  sorry

#check assignment_count_proof

end assignment_count_assignment_count_proof_l1047_104733


namespace rectangle_area_and_range_l1047_104755

/-- Represents the area of a rectangle formed by a rope of length 10cm -/
def area (x : ℝ) : ℝ := -x^2 + 5*x

/-- The length of the rope forming the rectangle -/
def ropeLength : ℝ := 10

theorem rectangle_area_and_range :
  ∀ x : ℝ, 0 < x ∧ x < 5 →
  (2 * (x + (ropeLength / 2 - x)) = ropeLength) ∧
  (area x = x * (ropeLength / 2 - x)) :=
sorry

end rectangle_area_and_range_l1047_104755


namespace carlos_initial_blocks_l1047_104789

/-- The number of blocks Carlos gave to Rachel -/
def blocks_given : ℕ := 21

/-- The number of blocks Carlos had left -/
def blocks_left : ℕ := 37

/-- The initial number of blocks Carlos had -/
def initial_blocks : ℕ := blocks_given + blocks_left

theorem carlos_initial_blocks : initial_blocks = 58 := by sorry

end carlos_initial_blocks_l1047_104789


namespace empty_solution_set_l1047_104766

theorem empty_solution_set : ∀ x : ℝ, ¬(2 * x - x^2 > 5) := by
  sorry

end empty_solution_set_l1047_104766


namespace max_table_height_l1047_104704

/-- Given a triangle DEF with side lengths 26, 28, and 34, prove that the maximum height k
    of a table formed by making right angle folds parallel to each side is 96√55/54. -/
theorem max_table_height (DE EF FD : ℝ) (h_DE : DE = 26) (h_EF : EF = 28) (h_FD : FD = 34) :
  let s := (DE + EF + FD) / 2
  let A := Real.sqrt (s * (s - DE) * (s - EF) * (s - FD))
  let h_e := 2 * A / EF
  let h_f := 2 * A / FD
  let k := h_e * h_f / (h_e + h_f)
  k = 96 * Real.sqrt 55 / 54 :=
by sorry

end max_table_height_l1047_104704


namespace inscribed_circle_radius_l1047_104772

/-- The radius of the inscribed circle in a triangle with sides 26, 15, and 17 is √6 -/
theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 26) (hb : b = 15) (hc : c = 17) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = Real.sqrt 6 := by sorry

end inscribed_circle_radius_l1047_104772


namespace total_distance_equals_expected_l1047_104787

/-- The initial travel distance per year in kilometers -/
def initial_distance : ℝ := 983400000000

/-- The factor by which the speed increases every 50 years -/
def speed_increase_factor : ℝ := 2

/-- The number of years for each speed increase -/
def years_per_increase : ℕ := 50

/-- The total number of years of travel -/
def total_years : ℕ := 150

/-- The function to calculate the total distance traveled -/
def total_distance : ℝ := 
  initial_distance * years_per_increase * (1 + speed_increase_factor + speed_increase_factor^2)

theorem total_distance_equals_expected : 
  total_distance = 3.4718e14 := by sorry

end total_distance_equals_expected_l1047_104787


namespace monotone_increasing_condition_l1047_104790

theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo (π / 6) (π / 3), 
    Monotone (fun x => (a - Real.sin x) / Real.cos x)) →
  a ≥ 2 := by
  sorry

end monotone_increasing_condition_l1047_104790


namespace arithmetic_sequence_count_l1047_104735

theorem arithmetic_sequence_count (a l d : ℤ) (h1 : a = -58) (h2 : l = 78) (h3 : d = 7) :
  ∃ n : ℕ, n > 0 ∧ l = a + (n - 1) * d ∧ n = 20 := by
  sorry

end arithmetic_sequence_count_l1047_104735
