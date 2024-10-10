import Mathlib

namespace intersection_nonempty_condition_l3920_392027

theorem intersection_nonempty_condition (m n : ℝ) :
  let A := {x : ℝ | m - 1 < x ∧ x < m + 1}
  let B := {x : ℝ | 3 - n < x ∧ x < 4 - n}
  (∃ x, x ∈ A ∩ B) ↔ 2 < m + n ∧ m + n < 5 :=
by sorry

end intersection_nonempty_condition_l3920_392027


namespace expression_evaluation_l3920_392015

theorem expression_evaluation (x y : ℤ) (hx : x = -1) (hy : y = 2) :
  x^2 - 2*(3*y^2 - x*y) + (y^2 - 2*x*y) = -19 := by
  sorry

end expression_evaluation_l3920_392015


namespace geometric_sequence_sum_l3920_392084

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∃ r : ℝ, r > 0 ∧ ∀ n, a (n + 1) = r * a n) →
  a 1 + a 2 + a 3 = 2 →
  a 3 + a 4 + a 5 = 8 →
  a 4 + a 5 + a 6 = 32 := by
sorry

end geometric_sequence_sum_l3920_392084


namespace intersection_points_theorem_l3920_392029

theorem intersection_points_theorem : 
  ∃ (k₁ k₂ k₃ k₄ : ℕ+), 
    k₁ + k₂ + k₃ + k₄ = 100 ∧ 
    k₁^2 + k₂^2 + k₃^2 + k₄^2 = 5996 ∧
    k₁ * k₂ + k₁ * k₃ + k₁ * k₄ + k₂ * k₃ + k₂ * k₄ + k₃ * k₄ = 2002 :=
by sorry

end intersection_points_theorem_l3920_392029


namespace number_equation_solution_l3920_392007

theorem number_equation_solution :
  ∃ x : ℝ, x + 5 * 12 / (180 / 3) = 51 ∧ x = 50 := by
sorry

end number_equation_solution_l3920_392007


namespace marbles_probability_l3920_392081

def total_marbles : ℕ := 13
def black_marbles : ℕ := 4
def red_marbles : ℕ := 3
def green_marbles : ℕ := 6
def drawn_marbles : ℕ := 2

def prob_same_color : ℚ := 
  (black_marbles * (black_marbles - 1) + 
   red_marbles * (red_marbles - 1) + 
   green_marbles * (green_marbles - 1)) / 
  (total_marbles * (total_marbles - 1))

theorem marbles_probability : 
  prob_same_color = 4 / 13 :=
sorry

end marbles_probability_l3920_392081


namespace intersection_A_complement_B_range_of_a_when_B_equals_A_l3920_392098

-- Define sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 5) < 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}

-- Part 1
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 2) = {x | -2 < x ∧ x ≤ 1 ∨ 3 ≤ x ∧ x < 5} := by sorry

-- Part 2
theorem range_of_a_when_B_equals_A :
  (∀ x, x ∈ B a ↔ x ∈ A) → -1 ≤ a ∧ a ≤ 4 := by sorry

end intersection_A_complement_B_range_of_a_when_B_equals_A_l3920_392098


namespace total_speech_time_l3920_392047

def speech_time (outline_time writing_time rewrite_time practice_time break1_time break2_time : ℝ) : ℝ :=
  outline_time + writing_time + rewrite_time + practice_time + break1_time + break2_time

theorem total_speech_time :
  let outline_time : ℝ := 30
  let break1_time : ℝ := 10
  let writing_time : ℝ := outline_time + 28
  let rewrite_time : ℝ := 15
  let break2_time : ℝ := 5
  let practice_time : ℝ := (writing_time + rewrite_time) / 2
  speech_time outline_time writing_time rewrite_time practice_time break1_time break2_time = 154.5 := by
  sorry

end total_speech_time_l3920_392047


namespace union_of_M_and_N_l3920_392028

def M : Set ℝ := {x | x^2 - x = 0}
def N : Set ℝ := {y | y^2 + y = 0}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1} := by sorry

end union_of_M_and_N_l3920_392028


namespace duanes_initial_pages_l3920_392010

theorem duanes_initial_pages (lana_initial : ℕ) (lana_final : ℕ) (duane_initial : ℕ) : 
  lana_initial = 8 → 
  lana_final = 29 → 
  lana_final = lana_initial + duane_initial / 2 →
  duane_initial = 42 := by
sorry

end duanes_initial_pages_l3920_392010


namespace percentage_seats_sold_l3920_392045

def stadium_capacity : ℕ := 60000
def fans_stayed_home : ℕ := 5000
def fans_attended : ℕ := 40000

theorem percentage_seats_sold :
  (fans_attended + fans_stayed_home) / stadium_capacity * 100 = 75 := by
  sorry

end percentage_seats_sold_l3920_392045


namespace digit_sum_reduction_count_l3920_392004

def digitSumReduction (n : ℕ) : ℕ :=
  if n % 9 = 0 then 9 else n % 9

def countDigits (d : ℕ) : ℕ := 
  (999999999 / 9 : ℕ) + (if d = 1 then 1 else 0)

theorem digit_sum_reduction_count :
  countDigits 1 = countDigits 2 + 1 :=
sorry

end digit_sum_reduction_count_l3920_392004


namespace football_progress_l3920_392099

/-- Calculates the overall progress in meters for a football team given their yard changes and the yard-to-meter conversion rate. -/
theorem football_progress (yard_to_meter : ℝ) (play1 play2 penalty play3 play4 : ℝ) :
  yard_to_meter = 0.9144 →
  play1 = -15 →
  play2 = 20 →
  penalty = -10 →
  play3 = 25 →
  play4 = -5 →
  (play1 + play2 + penalty + play3 + play4) * yard_to_meter = 13.716 := by
  sorry

end football_progress_l3920_392099


namespace basketball_team_selection_l3920_392071

/-- Represents the number of players selected from a class using stratified sampling -/
def stratified_sample (total_players : ℕ) (class_players : ℕ) (sample_size : ℕ) : ℕ :=
  (class_players * sample_size) / total_players

theorem basketball_team_selection (class5_players class16_players class33_players : ℕ) 
  (h1 : class5_players = 6)
  (h2 : class16_players = 8)
  (h3 : class33_players = 10)
  (h4 : class5_players + class16_players + class33_players = 24)
  (sample_size : ℕ)
  (h5 : sample_size = 12) :
  stratified_sample (class5_players + class16_players + class33_players) class5_players sample_size = 3 ∧
  stratified_sample (class5_players + class16_players + class33_players) class16_players sample_size = 4 := by
  sorry

end basketball_team_selection_l3920_392071


namespace point_5_neg1_in_fourth_quadrant_l3920_392088

def point_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_5_neg1_in_fourth_quadrant :
  point_in_fourth_quadrant 5 (-1) := by
  sorry

end point_5_neg1_in_fourth_quadrant_l3920_392088


namespace cos_neg_135_degrees_l3920_392087

theorem cos_neg_135_degrees :
  Real.cos ((-135 : ℝ) * π / 180) = -Real.sqrt 2 / 2 := by sorry

end cos_neg_135_degrees_l3920_392087


namespace budget_allocation_l3920_392020

theorem budget_allocation (microphotonics : ℝ) (home_electronics : ℝ) (gm_microorganisms : ℝ) (industrial_lubricants : ℝ) (astrophysics_degrees : ℝ) :
  microphotonics = 12 ∧
  home_electronics = 24 ∧
  gm_microorganisms = 29 ∧
  industrial_lubricants = 8 ∧
  astrophysics_degrees = 43.2 →
  100 - (microphotonics + home_electronics + gm_microorganisms + industrial_lubricants + (astrophysics_degrees / 360 * 100)) = 15 := by
  sorry

end budget_allocation_l3920_392020


namespace consecutive_integers_product_sum_l3920_392055

theorem consecutive_integers_product_sum (n : ℤ) : 
  n * (n + 1) * (n + 2) * (n + 3) = 3024 → n + (n + 1) + (n + 2) + (n + 3) = 30 := by
  sorry

end consecutive_integers_product_sum_l3920_392055


namespace circle_equation_l3920_392017

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line_l1 (x y : ℝ) : Prop := x - 3 * y = 0
def line_l2 (x y : ℝ) : Prop := x - y = 0

-- Define the conditions
def tangent_to_y_axis (c : Circle) : Prop :=
  c.center.1 = c.radius

def center_on_l1 (c : Circle) : Prop :=
  line_l1 c.center.1 c.center.2

def intersects_l2_with_chord (c : Circle) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_l2 x₁ y₁ ∧ line_l2 x₂ y₂ ∧
    (x₁ - c.center.1)^2 + (y₁ - c.center.2)^2 = c.radius^2 ∧
    (x₂ - c.center.1)^2 + (y₂ - c.center.2)^2 = c.radius^2 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8

-- Define the theorem
theorem circle_equation (c : Circle) :
  tangent_to_y_axis c →
  center_on_l1 c →
  intersects_l2_with_chord c →
  ((∀ x y, (x - 6 * Real.sqrt 2)^2 + (y - 2 * Real.sqrt 2)^2 = 72) ∨
   (∀ x y, (x + 6 * Real.sqrt 2)^2 + (y + 2 * Real.sqrt 2)^2 = 72)) :=
by sorry

end circle_equation_l3920_392017


namespace grape_juice_percentage_l3920_392080

/-- Calculates the percentage of grape juice in a mixture after adding pure grape juice --/
theorem grape_juice_percentage
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_grape_juice : ℝ)
  (h1 : initial_volume = 30)
  (h2 : initial_concentration = 0.1)
  (h3 : added_grape_juice = 10) :
  let final_volume := initial_volume + added_grape_juice
  let initial_grape_juice := initial_volume * initial_concentration
  let final_grape_juice := initial_grape_juice + added_grape_juice
  final_grape_juice / final_volume = 0.325 := by
  sorry

end grape_juice_percentage_l3920_392080


namespace cost_price_calculation_l3920_392058

/-- Proves that the cost price of an article is 40 given specific conditions --/
theorem cost_price_calculation (C M : ℝ) 
  (h1 : 0.95 * M = 1.25 * C)  -- Selling price after 5% discount equals 25% profit on cost
  (h2 : 0.95 * M = 50)        -- Selling price is 50
  : C = 40 := by
  sorry

end cost_price_calculation_l3920_392058


namespace rooster_earnings_l3920_392052

/-- Calculates the total earnings from selling roosters -/
def total_earnings (price_per_kg : ℝ) (weight1 : ℝ) (weight2 : ℝ) : ℝ :=
  price_per_kg * (weight1 + weight2)

/-- Theorem: The total earnings from selling two roosters weighing 30 kg and 40 kg at $0.50 per kg is $35 -/
theorem rooster_earnings : total_earnings 0.5 30 40 = 35 := by
  sorry

end rooster_earnings_l3920_392052


namespace radical_simplification_l3920_392091

theorem radical_simplification :
  Real.sqrt (4 - 2 * Real.sqrt 3) - Real.sqrt (4 + 2 * Real.sqrt 3) = -2 := by
  sorry

end radical_simplification_l3920_392091


namespace remainder_of_n_squared_plus_2n_plus_3_l3920_392067

theorem remainder_of_n_squared_plus_2n_plus_3 (n : ℤ) (k : ℤ) (h : n = 100 * k - 1) :
  (n^2 + 2*n + 3) % 100 = 2 := by
sorry

end remainder_of_n_squared_plus_2n_plus_3_l3920_392067


namespace ellipse_and_line_properties_l3920_392018

-- Define the ellipse
structure Ellipse where
  center : ℝ × ℝ
  vertex : ℝ × ℝ
  focus : ℝ × ℝ
  b_point : ℝ × ℝ

-- Define the line
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define the problem conditions
def ellipse_conditions (e : Ellipse) : Prop :=
  e.center = (0, 0) ∧
  e.vertex = (0, 2) ∧
  e.b_point = (Real.sqrt 2, Real.sqrt 2) ∧
  Real.sqrt ((e.focus.1 - Real.sqrt 2)^2 + (e.focus.2 - Real.sqrt 2)^2) = 2

-- Define the theorem
theorem ellipse_and_line_properties (e : Ellipse) (l : Line) :
  ellipse_conditions e →
  (∀ x y, y = l.slope * x + l.y_intercept → x^2 / 12 + y^2 / 4 = 1) →
  (0, -3) ∈ {(x, y) | y = l.slope * x + l.y_intercept} →
  (∃ m n : ℝ × ℝ, m ≠ n ∧
    m ∈ {(x, y) | x^2 / 12 + y^2 / 4 = 1} ∧
    n ∈ {(x, y) | x^2 / 12 + y^2 / 4 = 1} ∧
    m ∈ {(x, y) | y = l.slope * x + l.y_intercept} ∧
    n ∈ {(x, y) | y = l.slope * x + l.y_intercept} ∧
    (m.1 - 0)^2 + (m.2 - 2)^2 = (n.1 - 0)^2 + (n.2 - 2)^2) →
  (x^2 / 12 + y^2 / 4 = 1 ∧ (l.slope = Real.sqrt 6 / 3 ∨ l.slope = -Real.sqrt 6 / 3) ∧ l.y_intercept = -3) :=
by sorry

end ellipse_and_line_properties_l3920_392018


namespace parallelogram_inscribed_circles_l3920_392035

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a parallelogram ABCD -/
structure Parallelogram :=
  (A B C D : Point)

/-- Checks if a circle is inscribed in a triangle -/
def is_inscribed (c : Circle) (p1 p2 p3 : Point) : Prop := sorry

/-- Checks if a point lies on a line segment -/
def on_segment (p : Point) (p1 p2 : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Main theorem -/
theorem parallelogram_inscribed_circles 
  (ABCD : Parallelogram) 
  (P : Point) 
  (c_ABC : Circle) 
  (c_DAP : Circle) 
  (c_DCP : Circle) :
  on_segment P ABCD.A ABCD.C →
  is_inscribed c_ABC ABCD.A ABCD.B ABCD.C →
  is_inscribed c_DAP ABCD.D ABCD.A P →
  is_inscribed c_DCP ABCD.D ABCD.C P →
  distance ABCD.D ABCD.A + distance ABCD.D ABCD.C = 3 * distance ABCD.A ABCD.C →
  distance ABCD.D ABCD.A = distance ABCD.D P →
  (distance ABCD.D ABCD.A + distance ABCD.A P = distance ABCD.D ABCD.C + distance ABCD.C P) ∧
  (c_DAP.radius / c_DCP.radius = distance ABCD.A P / distance P ABCD.C) ∧
  (c_DAP.radius / c_DCP.radius = 4 / 3) := by
  sorry

end parallelogram_inscribed_circles_l3920_392035


namespace money_sharing_l3920_392001

theorem money_sharing (amanda_share : ℕ) (total : ℕ) : 
  amanda_share = 30 →
  3 * total = 16 * amanda_share →
  total = 160 := by sorry

end money_sharing_l3920_392001


namespace parabola_through_point_l3920_392079

/-- A parabola passing through point (4, -2) has a standard equation of either x² = -8y or y² = x -/
theorem parabola_through_point (P : ℝ × ℝ) (h : P = (4, -2)) :
  (∃ (x y : ℝ), x^2 = -8*y ∧ P.1 = x ∧ P.2 = y) ∨
  (∃ (x y : ℝ), y^2 = x ∧ P.1 = x ∧ P.2 = y) := by
sorry

end parabola_through_point_l3920_392079


namespace basketball_prices_and_discounts_l3920_392014

/-- Represents the prices and quantities of basketballs -/
structure BasketballPrices where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℝ
  quantity_b : ℝ

/-- Represents the discount options -/
inductive DiscountOption
  | Percent10
  | Buy3Get1Free

/-- The main theorem about basketball prices and discount options -/
theorem basketball_prices_and_discounts 
  (prices : BasketballPrices)
  (x : ℝ) :
  prices.price_a = prices.price_b + 40 →
  1200 / prices.price_a = 600 / prices.price_b →
  x ≥ 5 →
  (prices.price_a = 80 ∧ prices.price_b = 40) ∧
  (∀ y : ℝ, 
    y > 20 → (0.9 * (80 * 15 + 40 * y) < 80 * 15 + 40 * (y - 15 / 3)) ∧
    y = 20 → (0.9 * (80 * 15 + 40 * y) = 80 * 15 + 40 * (y - 15 / 3)) ∧
    y < 20 → (0.9 * (80 * 15 + 40 * y) > 80 * 15 + 40 * (y - 15 / 3))) := by
  sorry

#check basketball_prices_and_discounts

end basketball_prices_and_discounts_l3920_392014


namespace complex_number_proof_l3920_392070

theorem complex_number_proof (a : ℝ) (h_a : a > 0) (z : ℂ) (h_z : z = a - Complex.I) 
  (h_real : (z + 2 / z).im = 0) :
  z = 1 - Complex.I ∧ 
  ∀ m : ℝ, (((m : ℂ) - z)^2).re < 0 ∧ (((m : ℂ) - z)^2).im > 0 ↔ 1 < m ∧ m < 2 := by
  sorry

end complex_number_proof_l3920_392070


namespace min_distance_MN_l3920_392057

def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

def point_on_line (x y : ℝ) : Prop := ∃ (m b : ℝ), y = m*x + b ∧ 1 = m*3 + b

def intersect_points (M N : ℝ × ℝ) : Prop :=
  point_on_line M.1 M.2 ∧ point_on_line N.1 N.2 ∧
  circle_equation M.1 M.2 ∧ circle_equation N.1 N.2

theorem min_distance_MN :
  ∀ (M N : ℝ × ℝ), intersect_points M N →
  ∃ (MN : ℝ × ℝ → ℝ × ℝ → ℝ),
    (∀ (A B : ℝ × ℝ), MN A B ≥ 0) ∧
    (MN M N ≥ 4) ∧
    (∃ (M' N' : ℝ × ℝ), intersect_points M' N' ∧ MN M' N' = 4) :=
by sorry

end min_distance_MN_l3920_392057


namespace cube_labeling_theorem_l3920_392005

/-- A labeling of a cube's edges -/
def CubeLabeling := Fin 12 → Fin 13

/-- The sum of labels at a vertex given a labeling -/
def vertexSum (l : CubeLabeling) (v : Fin 8) : ℕ := sorry

/-- Predicate for a valid labeling using numbers 1 to 12 -/
def validLabeling12 (l : CubeLabeling) : Prop :=
  (∀ i : Fin 12, l i < 13) ∧ (∀ i j : Fin 12, i ≠ j → l i ≠ l j)

/-- Predicate for a valid labeling using numbers 1 to 13 with one unused -/
def validLabeling13 (l : CubeLabeling) : Prop :=
  (∀ i : Fin 12, l i > 0) ∧ (∀ i j : Fin 12, i ≠ j → l i ≠ l j)

theorem cube_labeling_theorem :
  (∀ l : CubeLabeling, validLabeling12 l →
    ∃ v1 v2 : Fin 8, v1 ≠ v2 ∧ vertexSum l v1 ≠ vertexSum l v2) ∧
  (∃ l : CubeLabeling, validLabeling13 l ∧
    ∀ v1 v2 : Fin 8, vertexSum l v1 = vertexSum l v2) :=
by sorry

end cube_labeling_theorem_l3920_392005


namespace range_of_m_l3920_392068

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h_eq : 2/x + 1/y = 1) 
  (h_ineq : ∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 1/y = 1 → x + 2*y > m^2 + 2*m) : 
  -4 < m ∧ m < 2 := by
sorry

end range_of_m_l3920_392068


namespace smallest_n_with_19_odd_digit_squares_l3920_392065

/-- A function that returns true if a number has an odd number of digits, false otherwise -/
def has_odd_digits (n : ℕ) : Bool :=
  sorry

/-- A function that counts how many numbers from 1 to n have squares with an odd number of digits -/
def count_odd_digit_squares (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that 44 is the smallest natural number N such that
    among the squares of integers from 1 to N, exactly 19 of them have an odd number of digits -/
theorem smallest_n_with_19_odd_digit_squares :
  ∀ n : ℕ, n < 44 → count_odd_digit_squares n < 19 ∧ count_odd_digit_squares 44 = 19 :=
sorry

end smallest_n_with_19_odd_digit_squares_l3920_392065


namespace seeds_in_fourth_pot_is_one_l3920_392097

/-- Given a total number of seeds, number of pots, and number of seeds per pot for the first three pots,
    calculate the number of seeds that will be planted in the fourth pot. -/
def seeds_in_fourth_pot (total_seeds : ℕ) (num_pots : ℕ) (seeds_per_pot : ℕ) : ℕ :=
  total_seeds - (seeds_per_pot * (num_pots - 1))

/-- Theorem stating that for the given problem, the number of seeds in the fourth pot is 1. -/
theorem seeds_in_fourth_pot_is_one :
  seeds_in_fourth_pot 10 4 3 = 1 := by
  sorry

#eval seeds_in_fourth_pot 10 4 3

end seeds_in_fourth_pot_is_one_l3920_392097


namespace area_of_rectangle_with_three_squares_l3920_392090

/-- Given three non-overlapping squares where one square has twice the side length of the other two,
    and the larger square has an area of 4 square inches, the area of the rectangle encompassing
    all three squares is 6 square inches. -/
theorem area_of_rectangle_with_three_squares (s : ℝ) : 
  s > 0 → (2 * s)^2 = 4 → 3 * s * 2 * s = 6 := by
  sorry

end area_of_rectangle_with_three_squares_l3920_392090


namespace product_is_three_digit_l3920_392000

def smallest_three_digit_number : ℕ := 100
def largest_single_digit_number : ℕ := 9

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem product_is_three_digit : 
  is_three_digit (smallest_three_digit_number * largest_single_digit_number) := by
  sorry

end product_is_three_digit_l3920_392000


namespace disinfectant_purchase_problem_l3920_392048

/-- The price difference between outdoor and indoor disinfectant -/
def price_difference : ℕ := 30

/-- The cost of 2 indoor and 3 outdoor disinfectant barrels -/
def sample_cost : ℕ := 340

/-- The total number of barrels to be purchased -/
def total_barrels : ℕ := 200

/-- The maximum total cost allowed -/
def max_cost : ℕ := 14000

/-- The price of indoor disinfectant -/
def indoor_price : ℕ := 50

/-- The price of outdoor disinfectant -/
def outdoor_price : ℕ := 80

/-- The minimum number of indoor disinfectant barrels to be purchased -/
def min_indoor_barrels : ℕ := 67

theorem disinfectant_purchase_problem :
  (outdoor_price = indoor_price + price_difference) ∧
  (2 * indoor_price + 3 * outdoor_price = sample_cost) ∧
  (∀ m : ℕ, m ≤ total_barrels →
    indoor_price * m + outdoor_price * (total_barrels - m) ≤ max_cost →
    m ≥ min_indoor_barrels) :=
by sorry

end disinfectant_purchase_problem_l3920_392048


namespace difference_after_five_iterations_l3920_392093

def initial_sequence : List ℕ := [2, 0, 1, 9, 0]

def next_sequence (seq : List ℕ) : List ℕ :=
  let pairs := seq.zip (seq.rotateRight 1)
  pairs.map (fun (a, b) => a + b)

def iterate_sequence (seq : List ℕ) (n : ℕ) : List ℕ :=
  match n with
  | 0 => seq
  | n + 1 => iterate_sequence (next_sequence seq) n

def sum_between_zeros (seq : List ℕ) : ℕ :=
  let rotated := seq.dropWhile (· ≠ 0)
  (rotated.takeWhile (· ≠ 0)).sum

def sum_not_between_zeros (seq : List ℕ) : ℕ :=
  seq.sum - sum_between_zeros seq

theorem difference_after_five_iterations :
  let final_seq := iterate_sequence initial_sequence 5
  sum_not_between_zeros final_seq - sum_between_zeros final_seq = 1944 := by
  sorry

end difference_after_five_iterations_l3920_392093


namespace square_minus_product_l3920_392021

theorem square_minus_product : (422 + 404)^2 - (4 * 422 * 404) = 324 := by
  sorry

end square_minus_product_l3920_392021


namespace correlation_analysis_l3920_392089

-- Define the types of variables
def TaxiFare : Type := ℝ
def Distance : Type := ℝ
def HouseSize : Type := ℝ
def HousePrice : Type := ℝ
def Height : Type := ℝ
def Weight : Type := ℝ
def IronBlockSize : Type := ℝ
def IronBlockMass : Type := ℝ

-- Define the relationship between variables
def functionalRelationship (α β : Type) : Prop := ∃ f : α → β, ∀ x : α, ∃! y : β, f x = y

-- Define correlation
def correlated (α β : Type) : Prop := 
  ¬(functionalRelationship α β) ∧ ¬(functionalRelationship β α) ∧ 
  ∃ f : α → β, ∀ x y : α, x ≠ y → f x ≠ f y

-- State the theorem
theorem correlation_analysis :
  functionalRelationship TaxiFare Distance ∧
  functionalRelationship HouseSize HousePrice ∧
  correlated Height Weight ∧
  functionalRelationship IronBlockSize IronBlockMass :=
by sorry

end correlation_analysis_l3920_392089


namespace sophie_joe_marbles_l3920_392037

theorem sophie_joe_marbles (sophie_initial : ℕ) (joe_initial : ℕ) (marbles_given : ℕ) :
  sophie_initial = 120 →
  joe_initial = 19 →
  marbles_given = 16 →
  sophie_initial - marbles_given = 3 * (joe_initial + marbles_given) :=
by
  sorry

end sophie_joe_marbles_l3920_392037


namespace brownie_pieces_l3920_392040

theorem brownie_pieces (pan_length pan_width piece_length piece_width : ℕ) 
  (h1 : pan_length = 24)
  (h2 : pan_width = 30)
  (h3 : piece_length = 3)
  (h4 : piece_width = 4) :
  (pan_length * pan_width) / (piece_length * piece_width) = 60 := by
  sorry

#check brownie_pieces

end brownie_pieces_l3920_392040


namespace stratified_sampling_high_group_l3920_392039

/-- Represents the number of students in each height group -/
structure HeightGroups where
  low : ℕ  -- [120, 130)
  mid : ℕ  -- [130, 140)
  high : ℕ -- [140, 150]

/-- Calculates the number of students to be selected from a group in stratified sampling -/
def stratifiedSample (totalPopulation : ℕ) (groupSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (groupSize * sampleSize + totalPopulation - 1) / totalPopulation

/-- Proves that the number of students to be selected from the [140, 150] group is 3 -/
theorem stratified_sampling_high_group 
  (groups : HeightGroups)
  (h1 : groups.low + groups.mid + groups.high = 100)
  (h2 : groups.low = 20)
  (h3 : groups.mid = 50)
  (h4 : groups.high = 30)
  (totalSample : ℕ)
  (h5 : totalSample = 18) :
  stratifiedSample 100 groups.high totalSample = 3 := by
sorry

#eval stratifiedSample 100 30 18

end stratified_sampling_high_group_l3920_392039


namespace inequality_range_l3920_392032

theorem inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1) 2, x^2 - a*x - 3 < 0) → 
  a ∈ Set.Ioo (1/2) 2 := by
  sorry

end inequality_range_l3920_392032


namespace alternate_interior_angles_parallel_l3920_392096

-- Define a structure for lines in a plane
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a structure for angles
structure Angle :=
  (measure : ℝ)

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define a function to represent alternate interior angles
def alternate_interior_angles (l1 l2 : Line) (t : Line) : (Angle × Angle) :=
  sorry

-- Theorem statement
theorem alternate_interior_angles_parallel (l1 l2 t : Line) :
  let (angle1, angle2) := alternate_interior_angles l1 l2 t
  (angle1.measure = angle2.measure) → are_parallel l1 l2 :=
sorry

end alternate_interior_angles_parallel_l3920_392096


namespace rational_sqrt_one_minus_xy_l3920_392049

theorem rational_sqrt_one_minus_xy (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ (q : ℚ), q^2 = 1 - x*y := by
  sorry

end rational_sqrt_one_minus_xy_l3920_392049


namespace sticker_distribution_l3920_392053

/-- The number of ways to distribute n identical objects among k distinct containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of stickers to be distributed -/
def num_stickers : ℕ := 11

/-- The number of sheets of paper -/
def num_sheets : ℕ := 5

theorem sticker_distribution :
  distribute num_stickers num_sheets = 1365 := by sorry

end sticker_distribution_l3920_392053


namespace repeating_decimal_division_l3920_392012

/-- Represents a repeating decimal with a whole number part and a repeating fractional part. -/
structure RepeatingDecimal where
  whole : ℕ
  repeating : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def repeatingDecimalToRational (d : RepeatingDecimal) : ℚ :=
  d.whole + d.repeating / (999 : ℚ)

/-- The main theorem: proving that 0.714714... divided by 2.857857... equals 119/476 -/
theorem repeating_decimal_division :
  let x : RepeatingDecimal := ⟨0, 714⟩
  let y : RepeatingDecimal := ⟨2, 857⟩
  (repeatingDecimalToRational x) / (repeatingDecimalToRational y) = 119 / 476 := by
  sorry


end repeating_decimal_division_l3920_392012


namespace factorial_500_trailing_zeroes_l3920_392023

def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_500_trailing_zeroes :
  trailingZeroes 500 = 124 := by
  sorry

end factorial_500_trailing_zeroes_l3920_392023


namespace irrational_partner_is_one_l3920_392083

theorem irrational_partner_is_one (a b : ℝ) : 
  (∃ (q : ℚ), a ≠ (q : ℝ)) → -- a is irrational
  (a * b - a - b + 1 = 0) →  -- given equation
  b = 1 :=                   -- conclusion: b equals 1
by sorry

end irrational_partner_is_one_l3920_392083


namespace student_wrestling_match_l3920_392030

theorem student_wrestling_match (n : ℕ) : n * (n - 1) / 2 = 91 → n = 14 := by
  sorry

end student_wrestling_match_l3920_392030


namespace magic_king_episodes_l3920_392025

/-- Calculates the total number of episodes for a TV show with the given parameters -/
def total_episodes (total_seasons : ℕ) (episodes_first_half : ℕ) (episodes_second_half : ℕ) : ℕ :=
  let half_seasons := total_seasons / 2
  half_seasons * episodes_first_half + half_seasons * episodes_second_half

/-- Proves that the TV show Magic King has 225 episodes in total -/
theorem magic_king_episodes : 
  total_episodes 10 20 25 = 225 := by
  sorry

end magic_king_episodes_l3920_392025


namespace rugs_bought_is_twenty_l3920_392066

/-- Calculates the number of rugs bought given buying price, selling price, and total profit -/
def rugs_bought (buying_price selling_price total_profit : ℚ) : ℚ :=
  total_profit / (selling_price - buying_price)

/-- Theorem stating that the number of rugs bought is 20 -/
theorem rugs_bought_is_twenty :
  rugs_bought 40 60 400 = 20 := by
  sorry

end rugs_bought_is_twenty_l3920_392066


namespace cubic_expression_factorization_l3920_392026

theorem cubic_expression_factorization (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) =
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end cubic_expression_factorization_l3920_392026


namespace num_ways_to_form_triangles_l3920_392011

/-- The number of distinguishable balls -/
def num_balls : ℕ := 6

/-- The number of distinguishable sticks -/
def num_sticks : ℕ := 6

/-- The number of balls required to form a triangle -/
def balls_per_triangle : ℕ := 3

/-- The number of sticks required to form a triangle -/
def sticks_per_triangle : ℕ := 3

/-- The number of triangles to be formed -/
def num_triangles : ℕ := 2

/-- The number of symmetries for each triangle (rotations and reflections) -/
def symmetries_per_triangle : ℕ := 6

/-- Theorem stating the number of ways to form two disjoint non-interlocking triangles -/
theorem num_ways_to_form_triangles : 
  (Nat.choose num_balls balls_per_triangle * Nat.factorial num_sticks) / 
  (Nat.factorial num_triangles * symmetries_per_triangle ^ num_triangles) = 200 :=
sorry

end num_ways_to_form_triangles_l3920_392011


namespace camping_trip_percentage_l3920_392059

theorem camping_trip_percentage :
  ∀ (total_percentage : ℝ) 
    (more_than_100 : ℝ) 
    (not_more_than_100 : ℝ),
  more_than_100 = 18 →
  total_percentage = more_than_100 + not_more_than_100 →
  total_percentage = 72 :=
by sorry

end camping_trip_percentage_l3920_392059


namespace video_game_pricing_l3920_392041

theorem video_game_pricing (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ) :
  total_games = 15 →
  non_working_games = 9 →
  total_earnings = 30 →
  (total_earnings : ℚ) / (total_games - non_working_games : ℚ) = 5 := by
  sorry

end video_game_pricing_l3920_392041


namespace subtraction_decimal_l3920_392072

theorem subtraction_decimal : 7.42 - 2.09 = 5.33 := by sorry

end subtraction_decimal_l3920_392072


namespace sum_of_coefficients_equals_one_l3920_392085

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the theorem
theorem sum_of_coefficients_equals_one (a b : ℝ) : 
  (i^2 + a * i + b = 0) → (a + b = 1) := by
  sorry

end sum_of_coefficients_equals_one_l3920_392085


namespace log_equation_solution_l3920_392075

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 2 + Real.log x / Real.log 4 = 6 → x = 16 := by
  sorry

end log_equation_solution_l3920_392075


namespace ratio_problem_l3920_392095

/-- Given ratios A : B : C where A = 4x, B = 6x, C = 9x, and A = 50, prove the values of B and C and their average -/
theorem ratio_problem (x : ℚ) (A B C : ℚ) (h1 : A = 4 * x) (h2 : B = 6 * x) (h3 : C = 9 * x) (h4 : A = 50) :
  B = 75 ∧ C = 112.5 ∧ (B + C) / 2 = 93.75 := by
  sorry


end ratio_problem_l3920_392095


namespace line_passes_through_fixed_point_l3920_392082

/-- A line that always passes through a fixed point regardless of the parameter m -/
def always_passes_through (m : ℝ) : Prop :=
  (m - 1) * (-3) + (2 * m - 3) * 1 + m = 0

/-- The theorem stating that the line passes through (-3, 1) for all real m -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, always_passes_through m :=
sorry

end line_passes_through_fixed_point_l3920_392082


namespace sales_volume_formula_max_profit_price_max_profit_value_profit_range_l3920_392077

/-- Represents the weekly sales volume as a function of price --/
def sales_volume (x : ℝ) : ℝ := -30 * x + 2100

/-- Represents the weekly profit as a function of price --/
def profit (x : ℝ) : ℝ := (x - 40) * (sales_volume x)

/-- The initial price in yuan --/
def initial_price : ℝ := 60

/-- The initial weekly sales in pieces --/
def initial_sales : ℝ := 300

/-- The cost price per piece in yuan --/
def cost_price : ℝ := 40

theorem sales_volume_formula (x : ℝ) : 
  sales_volume x = -30 * x + 2100 := by sorry

theorem max_profit_price : 
  ∃ (x : ℝ), ∀ (y : ℝ), profit x ≥ profit y ∧ x = 55 := by sorry

theorem max_profit_value : 
  profit 55 = 6750 := by sorry

theorem profit_range (x : ℝ) : 
  profit x ≥ 6480 ↔ 52 ≤ x ∧ x ≤ 58 := by sorry

end sales_volume_formula_max_profit_price_max_profit_value_profit_range_l3920_392077


namespace x_plus_reciprocal_two_implies_x_twelve_one_l3920_392069

theorem x_plus_reciprocal_two_implies_x_twelve_one (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 := by
  sorry

end x_plus_reciprocal_two_implies_x_twelve_one_l3920_392069


namespace same_solution_implies_b_value_l3920_392024

theorem same_solution_implies_b_value (x b : ℚ) : 
  (3 * x + 5 = 1) → 
  (b * x + 6 = 0) → 
  b = 9/2 := by
sorry

end same_solution_implies_b_value_l3920_392024


namespace einstein_fundraising_goal_l3920_392078

def pizza_price : ℚ := 12
def fries_price : ℚ := 0.3
def soda_price : ℚ := 2

def pizza_sold : ℕ := 15
def fries_sold : ℕ := 40
def soda_sold : ℕ := 25

def additional_needed : ℚ := 258

def total_raised : ℚ := pizza_price * pizza_sold + fries_price * fries_sold + soda_price * soda_sold

theorem einstein_fundraising_goal :
  total_raised + additional_needed = 500 := by sorry

end einstein_fundraising_goal_l3920_392078


namespace exists_x0_abs_fx0_plus_a_nonneg_l3920_392061

theorem exists_x0_abs_fx0_plus_a_nonneg (a b : ℝ) :
  ∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, |((x₀^2 : ℝ) + a * x₀ + b) + a| ≥ 0 := by
  sorry

end exists_x0_abs_fx0_plus_a_nonneg_l3920_392061


namespace f_properties_l3920_392009

/-- An odd function f(x) with a parameter a ≠ 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((a * x) / (1 - x) + 1)

/-- The function f is odd -/
axiom f_odd (a : ℝ) (h : a ≠ 0) : ∀ x, f a (-x) = -(f a x)

/-- Theorem stating the properties of the function f -/
theorem f_properties :
  ∃ (a : ℝ), a ≠ 0 ∧ 
  (a = 2) ∧ 
  (∀ x, f a x ≠ 0 ↔ -1 < x ∧ x < 1) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f a x < f a y) :=
sorry

end f_properties_l3920_392009


namespace negative_fraction_comparison_l3920_392002

theorem negative_fraction_comparison :
  -3/4 > -4/5 :=
by
  sorry

end negative_fraction_comparison_l3920_392002


namespace jon_points_l3920_392092

theorem jon_points (jon jack tom : ℕ) : 
  (jack = jon + 5) →
  (tom = jon + jack - 4) →
  (jon + jack + tom = 18) →
  (jon = 3) := by
sorry

end jon_points_l3920_392092


namespace log_8_4096_sum_bounds_l3920_392033

theorem log_8_4096_sum_bounds : ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) ≤ Real.log 4096 / Real.log 8 ∧ Real.log 4096 / Real.log 8 < (b : ℝ) ∧ a + b = 9 := by
  sorry

end log_8_4096_sum_bounds_l3920_392033


namespace four_numbers_in_interval_l3920_392086

theorem four_numbers_in_interval (a b c d : Real) : 
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < π / 2 →
  ∃ x y, (x = a ∨ x = b ∨ x = c ∨ x = d) ∧
         (y = a ∨ y = b ∨ y = c ∨ y = d) ∧
         x ≠ y ∧
         |x - y| < π / 6 :=
by sorry

end four_numbers_in_interval_l3920_392086


namespace parabola_vertex_l3920_392042

/-- The parabola is defined by the equation y = (x+2)^2 - 1 -/
def parabola (x y : ℝ) : Prop := y = (x + 2)^2 - 1

/-- The vertex of a parabola is the point where it reaches its maximum or minimum -/
def is_vertex (x y : ℝ) : Prop := 
  parabola x y ∧ ∀ x' y', parabola x' y' → y ≤ y'

/-- Theorem: The vertex of the parabola y = (x+2)^2 - 1 has coordinates (-2, -1) -/
theorem parabola_vertex : is_vertex (-2) (-1) := by sorry

end parabola_vertex_l3920_392042


namespace national_day_2020_l3920_392074

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

-- Theorem statement
theorem national_day_2020 (national_day_2019 : DayOfWeek) 
  (h1 : national_day_2019 = DayOfWeek.Tuesday) 
  (h2 : advanceDay national_day_2019 2 = DayOfWeek.Thursday) : 
  advanceDay national_day_2019 2 = DayOfWeek.Thursday := by
  sorry

#check national_day_2020

end national_day_2020_l3920_392074


namespace card_sending_probability_l3920_392036

def num_senders : ℕ := 3
def num_recipients : ℕ := 2

theorem card_sending_probability :
  let total_outcomes := num_recipients ^ num_senders
  let favorable_outcomes := num_recipients
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by
sorry

end card_sending_probability_l3920_392036


namespace unique_root_condition_l3920_392043

/-- The equation x + 1 = √(px) has exactly one real root if and only if p = 4 or p ≤ 0. -/
theorem unique_root_condition (p : ℝ) : 
  (∃! x : ℝ, x + 1 = Real.sqrt (p * x)) ↔ p = 4 ∨ p ≤ 0 := by
  sorry

end unique_root_condition_l3920_392043


namespace vector_representation_l3920_392016

def a : Fin 2 → ℝ := ![3, -1]
def e1B : Fin 2 → ℝ := ![-1, 2]
def e2B : Fin 2 → ℝ := ![3, 2]
def e1A : Fin 2 → ℝ := ![0, 0]
def e2A : Fin 2 → ℝ := ![3, 2]
def e1C : Fin 2 → ℝ := ![3, 5]
def e2C : Fin 2 → ℝ := ![6, 10]
def e1D : Fin 2 → ℝ := ![-3, 5]
def e2D : Fin 2 → ℝ := ![3, -5]

theorem vector_representation :
  (∃ α β : ℝ, a = α • e1B + β • e2B) ∧
  (∀ α β : ℝ, a ≠ α • e1A + β • e2A) ∧
  (∀ α β : ℝ, a ≠ α • e1C + β • e2C) ∧
  (∀ α β : ℝ, a ≠ α • e1D + β • e2D) :=
by sorry

end vector_representation_l3920_392016


namespace cos_4theta_from_exp_l3920_392062

theorem cos_4theta_from_exp (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (1 - Complex.I * Real.sqrt 3) / 2) :
  Real.cos (4 * θ) = -1/2 := by sorry

end cos_4theta_from_exp_l3920_392062


namespace basketball_probability_l3920_392008

theorem basketball_probability (p_no_make : ℝ) (num_tries : ℕ) : 
  p_no_make = 1/3 → num_tries = 3 → 
  let p_make := 1 - p_no_make
  (num_tries.choose 1) * p_make * p_no_make^2 = 2/9 := by
sorry

end basketball_probability_l3920_392008


namespace unique_a_divisibility_l3920_392056

theorem unique_a_divisibility (a : ℤ) (h1 : 0 < a) (h2 : a < 13) 
  (h3 : (13 : ℤ) ∣ (53^2017 + a)) : a = 12 := by
  sorry

end unique_a_divisibility_l3920_392056


namespace percentage_problem_l3920_392051

theorem percentage_problem (x : ℝ) (P : ℝ) : 
  x = 680 →
  (P / 100) * x = 0.20 * 1000 - 30 →
  P = 25 := by
sorry

end percentage_problem_l3920_392051


namespace complement_union_equal_l3920_392054

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 3, 4}
def B : Set Nat := {1, 3}

theorem complement_union_equal : (U \ A) ∪ B = {1, 2, 3} := by sorry

end complement_union_equal_l3920_392054


namespace proposition_implications_l3920_392038

theorem proposition_implications (p q : Prop) :
  ¬(¬p ∨ ¬q) → (p ∧ q) ∧ (p ∨ q) :=
by sorry

end proposition_implications_l3920_392038


namespace triangle_problem_l3920_392019

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Conditions
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a * Real.sin B + b * Real.cos A = c) →
  (a = Real.sqrt 2 * c) →
  (b = 2) →
  -- Conclusions
  (B = π / 4 ∧ c = 2) :=
by sorry


end triangle_problem_l3920_392019


namespace min_value_a_plus_4b_l3920_392076

theorem min_value_a_plus_4b (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : 1 / (a - 1) + 1 / (b - 1) = 1) : 
  ∀ x y, x > 1 → y > 1 → 1 / (x - 1) + 1 / (y - 1) = 1 → a + 4 * b ≤ x + 4 * y ∧ 
  ∃ a₀ b₀, a₀ > 1 ∧ b₀ > 1 ∧ 1 / (a₀ - 1) + 1 / (b₀ - 1) = 1 ∧ a₀ + 4 * b₀ = 14 :=
by sorry

end min_value_a_plus_4b_l3920_392076


namespace lottery_probability_l3920_392094

theorem lottery_probability (winning_rate : ℚ) (num_tickets : ℕ) : 
  winning_rate = 1/3 → num_tickets = 3 → 
  (1 - (1 - winning_rate) ^ num_tickets) = 19/27 := by
  sorry

end lottery_probability_l3920_392094


namespace original_price_of_discounted_shoes_l3920_392044

theorem original_price_of_discounted_shoes 
  (purchase_price : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : purchase_price = 51)
  (h2 : discount_percentage = 75) : 
  purchase_price / (1 - discount_percentage / 100) = 204 := by
  sorry

end original_price_of_discounted_shoes_l3920_392044


namespace least_multiple_of_first_four_primes_two_ten_divisible_by_first_four_primes_least_multiple_is_two_ten_l3920_392034

theorem least_multiple_of_first_four_primes : 
  ∀ n : ℕ, n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n → n ≥ 210 :=
by sorry

theorem two_ten_divisible_by_first_four_primes : 
  2 ∣ 210 ∧ 3 ∣ 210 ∧ 5 ∣ 210 ∧ 7 ∣ 210 :=
by sorry

theorem least_multiple_is_two_ten : 
  ∃! n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m → n ≤ m) ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ n = 210 :=
by sorry

end least_multiple_of_first_four_primes_two_ten_divisible_by_first_four_primes_least_multiple_is_two_ten_l3920_392034


namespace original_equals_scientific_l3920_392006

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 1030000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 1.03
  , exponent := 9
  , is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end original_equals_scientific_l3920_392006


namespace decade_cost_l3920_392046

/-- Vivian's annual car insurance cost in dollars -/
def annual_cost : ℕ := 2000

/-- Number of years in a decade -/
def decade : ℕ := 10

/-- Theorem: Vivian's total car insurance cost over a decade -/
theorem decade_cost : annual_cost * decade = 20000 := by
  sorry

end decade_cost_l3920_392046


namespace sum_1984_consecutive_not_square_l3920_392060

theorem sum_1984_consecutive_not_square (n : ℕ) : 
  ¬ ∃ k : ℕ, (992 * (2 * n + 1985) : ℕ) = k^2 := by
  sorry

end sum_1984_consecutive_not_square_l3920_392060


namespace distinct_prime_factors_of_75_l3920_392050

theorem distinct_prime_factors_of_75 : Nat.card (Nat.factors 75).toFinset = 2 := by
  sorry

end distinct_prime_factors_of_75_l3920_392050


namespace product_from_lcm_gcf_l3920_392063

theorem product_from_lcm_gcf (a b c : ℕ+) 
  (h1 : Nat.lcm (Nat.lcm a.val b.val) c.val = 2310)
  (h2 : Nat.gcd (Nat.gcd a.val b.val) c.val = 30) :
  a * b * c = 69300 := by
  sorry

end product_from_lcm_gcf_l3920_392063


namespace fixed_point_on_linear_function_l3920_392013

theorem fixed_point_on_linear_function (m : ℝ) (h : m ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ m * x - (3 * m + 2)
  f 3 = -2 := by
  sorry

end fixed_point_on_linear_function_l3920_392013


namespace max_prism_volume_in_hexagonal_pyramid_l3920_392073

/-- Represents a regular hexagonal pyramid -/
structure HexagonalPyramid where
  base_side_length : ℝ
  lateral_face_leg_length : ℝ

/-- Represents a right square prism -/
structure SquarePrism where
  side_length : ℝ

/-- Calculates the volume of a right square prism -/
def prism_volume (p : SquarePrism) : ℝ := p.side_length ^ 3

/-- Theorem stating the maximum volume of the square prism within the hexagonal pyramid -/
theorem max_prism_volume_in_hexagonal_pyramid 
  (pyramid : HexagonalPyramid) 
  (prism : SquarePrism) 
  (h1 : pyramid.base_side_length = 2) 
  (h2 : prism.side_length ≤ pyramid.base_side_length) 
  (h3 : prism.side_length > 0) :
  prism_volume prism ≤ 8 :=
sorry

end max_prism_volume_in_hexagonal_pyramid_l3920_392073


namespace extremum_implies_a_eq_one_f_less_than_c_squared_implies_c_range_l3920_392031

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 - 9*x + 5

-- Theorem 1: If f has an extremum at x = 1, then a = 1
theorem extremum_implies_a_eq_one (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1 ∨ f a x ≥ f a 1) →
  a = 1 :=
sorry

-- Theorem 2: If f(x) < c² for all x in [-4, 4], then c is in (-∞, -9) ∪ (9, +∞)
theorem f_less_than_c_squared_implies_c_range :
  (∀ x ∈ Set.Icc (-4) 4, f 1 x < c^2) →
  c ∈ Set.Iio (-9) ∪ Set.Ioi 9 :=
sorry

end extremum_implies_a_eq_one_f_less_than_c_squared_implies_c_range_l3920_392031


namespace cost_of_pencils_l3920_392022

/-- The cost of a single pencil in cents -/
def pencil_cost : ℕ := 3

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of pencils to calculate the cost for -/
def num_pencils : ℕ := 500

/-- Theorem: The cost of 500 pencils in dollars is 15.00 -/
theorem cost_of_pencils : 
  (num_pencils * pencil_cost) / cents_per_dollar = 15 := by
  sorry

end cost_of_pencils_l3920_392022


namespace industrial_lubricants_percentage_l3920_392003

theorem industrial_lubricants_percentage 
  (microphotonics : ℝ) 
  (home_electronics : ℝ) 
  (food_additives : ℝ) 
  (genetically_modified_microorganisms : ℝ) 
  (basic_astrophysics_degrees : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 20 →
  genetically_modified_microorganisms = 29 →
  basic_astrophysics_degrees = 18 →
  let basic_astrophysics := (basic_astrophysics_degrees / 360) * 100
  let total_known := microphotonics + home_electronics + food_additives + 
                     genetically_modified_microorganisms + basic_astrophysics
  let industrial_lubricants := 100 - total_known
  industrial_lubricants = 8 :=
by sorry

end industrial_lubricants_percentage_l3920_392003


namespace power_of_square_l3920_392064

theorem power_of_square (x : ℝ) : (x^2)^3 = x^6 := by
  sorry

end power_of_square_l3920_392064
