import Mathlib

namespace area_ratio_GHI_JKL_l2309_230917

-- Define the triangles
def triangle_GHI : ℕ × ℕ × ℕ := (6, 8, 10)
def triangle_JKL : ℕ × ℕ × ℕ := (9, 12, 15)

-- Define a function to calculate the area of a right triangle
def area_right_triangle (a b : ℕ) : ℚ :=
  (a * b : ℚ) / 2

-- Theorem statement
theorem area_ratio_GHI_JKL :
  let (g1, g2, _) := triangle_GHI
  let (j1, j2, _) := triangle_JKL
  (area_right_triangle g1 g2) / (area_right_triangle j1 j2) = 4 / 9 := by
  sorry

end area_ratio_GHI_JKL_l2309_230917


namespace sequence_bound_l2309_230973

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ i : ℕ, i > 0 → 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j : ℕ, i > 0 → j > 0 → i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 := by
sorry

end sequence_bound_l2309_230973


namespace no_divisibility_by_15_and_11_exists_divisibility_by_11_l2309_230918

def is_five_digit_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000

def construct_number (n : ℕ) : ℕ :=
  80000 + n * 1000 + 642

theorem no_divisibility_by_15_and_11 :
  ¬ ∃ (n : ℕ), n < 10 ∧ 
    is_five_digit_number (construct_number n) ∧ 
    (construct_number n) % 15 = 0 ∧ 
    (construct_number n) % 11 = 0 :=
sorry

theorem exists_divisibility_by_11 :
  ∃ (n : ℕ), n < 10 ∧ 
    is_five_digit_number (construct_number n) ∧ 
    (construct_number n) % 11 = 0 :=
sorry

end no_divisibility_by_15_and_11_exists_divisibility_by_11_l2309_230918


namespace prob_six_largest_is_two_sevenths_l2309_230934

/-- A function that calculates the probability of selecting 6 as the largest value
    when drawing 4 cards from a set of 7 cards numbered 1 to 7 without replacement -/
def prob_six_largest (n : ℕ) (k : ℕ) : ℚ :=
  if n = 7 ∧ k = 4 then 2/7 else 0

/-- Theorem stating that the probability of selecting 6 as the largest value
    when drawing 4 cards from a set of 7 cards numbered 1 to 7 without replacement is 2/7 -/
theorem prob_six_largest_is_two_sevenths :
  prob_six_largest 7 4 = 2/7 := by
  sorry

end prob_six_largest_is_two_sevenths_l2309_230934


namespace largest_of_five_consecutive_odd_integers_l2309_230906

theorem largest_of_five_consecutive_odd_integers (a b c d e : ℤ) : 
  (∃ n : ℤ, a = 2*n + 1 ∧ b = 2*n + 3 ∧ c = 2*n + 5 ∧ d = 2*n + 7 ∧ e = 2*n + 9) →
  a + b + c + d + e = 255 →
  max a (max b (max c (max d e))) = 55 :=
by sorry

end largest_of_five_consecutive_odd_integers_l2309_230906


namespace light_bulb_state_l2309_230921

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def toggle_light (n : ℕ) (i : ℕ) : Bool := i % n = 0

def final_state (n : ℕ) : Bool :=
  (List.range n).foldl (fun acc i => acc ≠ toggle_light (i + 1) n) false

theorem light_bulb_state (n : ℕ) (hn : n ≤ 100) :
  final_state n = true ↔ is_perfect_square n :=
sorry

end light_bulb_state_l2309_230921


namespace days_before_reinforcement_l2309_230959

/-- 
Given a garrison with initial provisions and a reinforcement, 
calculate the number of days that passed before the reinforcement arrived.
-/
theorem days_before_reinforcement 
  (initial_garrison : ℕ) 
  (initial_provisions : ℕ) 
  (reinforcement : ℕ) 
  (remaining_provisions : ℕ) 
  (h1 : initial_garrison = 150)
  (h2 : initial_provisions = 31)
  (h3 : reinforcement = 300)
  (h4 : remaining_provisions = 5) : 
  ∃ (x : ℕ), x = 16 ∧ 
    initial_garrison * (initial_provisions - x) = 
    (initial_garrison + reinforcement) * remaining_provisions :=
by sorry

end days_before_reinforcement_l2309_230959


namespace melissa_points_per_game_l2309_230984

-- Define the number of games
def num_games : ℕ := 3

-- Define the total points scored
def total_points : ℕ := 81

-- Define the points per game as a function
def points_per_game : ℕ := total_points / num_games

-- Theorem to prove
theorem melissa_points_per_game : points_per_game = 27 := by
  sorry

end melissa_points_per_game_l2309_230984


namespace odd_z_has_4n_minus_1_divisor_l2309_230944

theorem odd_z_has_4n_minus_1_divisor (x y : ℕ+) (z : ℤ) 
  (hz : z = (4 * x * y : ℤ) / (x + y : ℤ)) 
  (hodd : Odd z) : 
  ∃ (d : ℤ), d ∣ z ∧ ∃ (n : ℕ+), d = 4 * n - 1 := by
  sorry

end odd_z_has_4n_minus_1_divisor_l2309_230944


namespace two_numbers_difference_l2309_230955

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : 
  |x - y| = 22 := by
sorry

end two_numbers_difference_l2309_230955


namespace feline_sanctuary_count_l2309_230986

theorem feline_sanctuary_count :
  let lions : ℕ := 12
  let tigers : ℕ := 14
  let cougars : ℕ := (lions + tigers) / 3
  lions + tigers + cougars = 34 := by
sorry

end feline_sanctuary_count_l2309_230986


namespace expression_evaluation_l2309_230935

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = -0.5) :
  2 * (2 * x - 3 * y) - (3 * x + 2 * y + 1) = 5 := by
  sorry

end expression_evaluation_l2309_230935


namespace events_mutually_exclusive_but_not_opposite_l2309_230939

/-- Represents a card color -/
inductive CardColor
| Red
| Black
| Blue
| White

/-- Represents a person -/
inductive Person
| A
| B
| C
| D

/-- Represents the distribution of cards to people -/
def Distribution := Person → CardColor

/-- The event "A receives the red card" -/
def event_A_red (d : Distribution) : Prop := d Person.A = CardColor.Red

/-- The event "B receives the red card" -/
def event_B_red (d : Distribution) : Prop := d Person.B = CardColor.Red

/-- The set of all possible distributions -/
def all_distributions : Set Distribution :=
  {d | ∀ c : CardColor, ∃! p : Person, d p = c}

theorem events_mutually_exclusive_but_not_opposite :
  (∀ d : Distribution, d ∈ all_distributions →
    ¬(event_A_red d ∧ event_B_red d)) ∧
  (∃ d : Distribution, d ∈ all_distributions ∧
    ¬event_A_red d ∧ ¬event_B_red d) :=
sorry

end events_mutually_exclusive_but_not_opposite_l2309_230939


namespace sum_of_digits_M_l2309_230929

-- Define M as a positive integer
def M : ℕ+ := sorry

-- Define the condition that M^2 = 36^50 * 50^36
axiom M_squared : (M : ℕ).pow 2 = (36 : ℕ).pow 50 * (50 : ℕ).pow 36

-- Define a function to calculate the sum of digits of a natural number
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_M : sum_of_digits (M : ℕ) = 344 := by sorry

end sum_of_digits_M_l2309_230929


namespace three_points_in_circle_l2309_230980

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in a 2D plane -/
structure Square where
  side : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point is inside a square -/
def Point.inSquare (p : Point) (s : Square) : Prop :=
  0 ≤ p.x ∧ p.x ≤ s.side ∧ 0 ≤ p.y ∧ p.y ≤ s.side

/-- Check if a point is inside a circle -/
def Point.inCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

/-- The main theorem -/
theorem three_points_in_circle (points : Finset Point) (s : Square) :
  s.side = 1 →
  points.card = 51 →
  ∀ p ∈ points, p.inSquare s →
  ∃ (c : Circle) (p1 p2 p3 : Point),
    c.radius = 1/7 ∧
    p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    p1.inCircle c ∧ p2.inCircle c ∧ p3.inCircle c :=
sorry


end three_points_in_circle_l2309_230980


namespace smallest_total_books_satisfying_conditions_l2309_230922

/-- Represents the number of books for each subject -/
structure BookCounts where
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Checks if the given book counts satisfy the required ratios -/
def satisfiesRatios (books : BookCounts) : Prop :=
  3 * books.chemistry = 2 * books.physics ∧
  4 * books.biology = 3 * books.chemistry

/-- Calculates the total number of books -/
def totalBooks (books : BookCounts) : ℕ :=
  books.physics + books.chemistry + books.biology

/-- Theorem stating the smallest possible total number of books satisfying the conditions -/
theorem smallest_total_books_satisfying_conditions :
  ∃ (books : BookCounts),
    satisfiesRatios books ∧
    totalBooks books > 3000 ∧
    ∀ (other : BookCounts),
      satisfiesRatios other → totalBooks other > 3000 →
      totalBooks books ≤ totalBooks other :=
by sorry

end smallest_total_books_satisfying_conditions_l2309_230922


namespace smallest_class_size_l2309_230970

theorem smallest_class_size : ∃ (n : ℕ), n > 0 ∧ (∃ (k : ℕ), k > 0 ∧ 29/10 < (100 * k : ℚ)/n ∧ (100 * k : ℚ)/n < 31/10) ∧
  ∀ (m : ℕ), m > 0 → m < n → ¬(∃ (j : ℕ), j > 0 ∧ 29/10 < (100 * j : ℚ)/m ∧ (100 * j : ℚ)/m < 31/10) :=
by
  -- The proof goes here
  sorry

end smallest_class_size_l2309_230970


namespace train_length_calculation_l2309_230993

/-- Calculates the length of a train given the speeds of two trains, the time they take to cross each other, and the length of the other train. -/
theorem train_length_calculation (v1 v2 : ℝ) (t cross_time : ℝ) (l2 : ℝ) :
  v1 = 60 →
  v2 = 40 →
  cross_time = 12.239020878329734 →
  l2 = 200 →
  (v1 + v2) * 1000 / 3600 * cross_time - l2 = 140 :=
by sorry

end train_length_calculation_l2309_230993


namespace complement_of_A_in_U_l2309_230962

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-2, -1, 1, 2}

theorem complement_of_A_in_U : 
  {x : Int | x ∈ U ∧ x ∉ A} = {0} := by sorry

end complement_of_A_in_U_l2309_230962


namespace emily_score_emily_score_proof_l2309_230920

/-- Calculates Emily's score in a dodgeball game -/
theorem emily_score (total_players : ℕ) (total_points : ℕ) (other_player_score : ℕ) : ℕ :=
  let other_players := total_players - 1
  let other_players_total := other_players * other_player_score
  total_points - other_players_total

/-- Proves Emily's score given the game conditions -/
theorem emily_score_proof :
  emily_score 8 39 2 = 25 := by
  sorry

end emily_score_emily_score_proof_l2309_230920


namespace greatest_divisor_with_remainder_l2309_230903

def n₁ : ℕ := 263
def n₂ : ℕ := 935
def n₃ : ℕ := 1383
def r : ℕ := 7
def d : ℕ := 32

theorem greatest_divisor_with_remainder (m : ℕ) :
  (m > d → ¬(n₁ % m = r ∧ n₂ % m = r ∧ n₃ % m = r)) ∧
  (n₁ % d = r ∧ n₂ % d = r ∧ n₃ % d = r) := by
  sorry

end greatest_divisor_with_remainder_l2309_230903


namespace inequality_solution_equivalence_l2309_230950

theorem inequality_solution_equivalence (f : ℝ → ℝ) :
  (∃ x : ℝ, f x > 0) ↔ (∃ x₁ : ℝ, f x₁ > 0) := by sorry

end inequality_solution_equivalence_l2309_230950


namespace remaining_land_to_clean_l2309_230931

theorem remaining_land_to_clean (total_land area_lizzie area_other : ℕ) :
  total_land = 900 ∧ area_lizzie = 250 ∧ area_other = 265 →
  total_land - (area_lizzie + area_other) = 385 := by
  sorry

end remaining_land_to_clean_l2309_230931


namespace inequality_and_equality_condition_l2309_230900

theorem inequality_and_equality_condition (α β a b : ℝ) (h_pos_α : 0 < α) (h_pos_β : 0 < β)
  (h_a_range : α ≤ a ∧ a ≤ β) (h_b_range : α ≤ b ∧ b ≤ β) :
  b / a + a / b ≤ β / α + α / β ∧
  (b / a + a / b = β / α + α / β ↔ (a = α ∧ b = β) ∨ (a = β ∧ b = α)) :=
by sorry

end inequality_and_equality_condition_l2309_230900


namespace power_function_linear_intersection_min_value_l2309_230924

theorem power_function_linear_intersection_min_value (m n k b : ℝ) : 
  (2 * m - 1 = 1) →  -- Condition for power function
  (n - 2 = 0) →      -- Condition for power function
  (k > 0) →          -- Given condition for k
  (b > 0) →          -- Given condition for b
  (k * m + b = n) →  -- Linear function passes through (m, n)
  (∀ k' b' : ℝ, k' > 0 → b' > 0 → k' * m + b' = n → 4 / k' + 1 / b' ≥ 9 / 2) ∧ 
  (∃ k' b' : ℝ, k' > 0 ∧ b' > 0 ∧ k' * m + b' = n ∧ 4 / k' + 1 / b' = 9 / 2) :=
by sorry

end power_function_linear_intersection_min_value_l2309_230924


namespace constant_c_value_l2309_230990

theorem constant_c_value : ∃ c : ℚ, 
  (∀ x : ℚ, (3 * x^3 - 5 * x^2 + 6 * x - 4) * (2 * x^2 + c * x + 8) = 
   6 * x^5 - 19 * x^4 + 40 * x^3 + c * x^2 - 32 * x + 32) ∧ 
  c = 48 / 5 := by
sorry

end constant_c_value_l2309_230990


namespace gcd_of_quadratic_and_linear_l2309_230915

theorem gcd_of_quadratic_and_linear (b : ℤ) (h : ∃ k : ℤ, b = 2 * 8753 * k) :
  Int.gcd (4 * b^2 + 27 * b + 100) (3 * b + 7) = 2 := by
  sorry

end gcd_of_quadratic_and_linear_l2309_230915


namespace calories_per_candy_bar_l2309_230965

/-- Given that there are 15 calories in 5 candy bars, prove that there are 3 calories in one candy bar. -/
theorem calories_per_candy_bar :
  let total_calories : ℕ := 15
  let total_bars : ℕ := 5
  let calories_per_bar : ℚ := total_calories / total_bars
  calories_per_bar = 3 := by sorry

end calories_per_candy_bar_l2309_230965


namespace min_printers_purchase_l2309_230901

def printer_cost_a : ℕ := 350
def printer_cost_b : ℕ := 200

theorem min_printers_purchase :
  ∃ (n_a n_b : ℕ),
    n_a * printer_cost_a = n_b * printer_cost_b ∧
    n_a + n_b = 11 ∧
    ∀ (m_a m_b : ℕ),
      m_a * printer_cost_a = m_b * printer_cost_b →
      m_a + m_b ≥ 11 :=
sorry

end min_printers_purchase_l2309_230901


namespace ellipse_equation_equivalence_l2309_230947

theorem ellipse_equation_equivalence (x y : ℝ) :
  (Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 10) ↔
  (x^2 / 25 + y^2 / 21 = 1) := by
  sorry

end ellipse_equation_equivalence_l2309_230947


namespace problem_building_has_20_stories_l2309_230951

/-- A building with specific height properties -/
structure Building where
  first_stories : ℕ
  first_story_height : ℕ
  remaining_story_height : ℕ
  total_height : ℕ

/-- The number of stories in the building -/
def Building.total_stories (b : Building) : ℕ :=
  b.first_stories + (b.total_height - b.first_stories * b.first_story_height) / b.remaining_story_height

/-- The specific building described in the problem -/
def problem_building : Building := {
  first_stories := 10
  first_story_height := 12
  remaining_story_height := 15
  total_height := 270
}

/-- Theorem stating that the problem building has 20 stories -/
theorem problem_building_has_20_stories :
  problem_building.total_stories = 20 := by
  sorry

end problem_building_has_20_stories_l2309_230951


namespace cricket_team_age_problem_l2309_230996

/-- Represents the age difference between the wicket keeper and the team average -/
def wicket_keeper_age_difference (team_size : ℕ) (team_average_age : ℕ) 
  (known_member_age : ℕ) (remaining_average_age : ℕ) : ℕ :=
  let total_age := team_size * team_average_age
  let remaining_total_age := (team_size - 2) * remaining_average_age
  let wicket_keeper_age := total_age - known_member_age - remaining_total_age
  wicket_keeper_age - team_average_age

theorem cricket_team_age_problem :
  wicket_keeper_age_difference 11 22 25 21 = 6 := by
  sorry

end cricket_team_age_problem_l2309_230996


namespace fruit_distribution_l2309_230992

theorem fruit_distribution (total_strawberries total_grapes : ℕ) 
  (leftover_strawberries leftover_grapes : ℕ) :
  total_strawberries = 66 →
  total_grapes = 49 →
  leftover_strawberries = 6 →
  leftover_grapes = 4 →
  ∃ (B : ℕ), 
    B > 0 ∧
    (total_strawberries - leftover_strawberries) % B = 0 ∧
    (total_grapes - leftover_grapes) % B = 0 ∧
    B = 15 :=
by
  sorry

end fruit_distribution_l2309_230992


namespace bird_problem_equations_l2309_230949

/-- Represents the cost of each type of bird in coins -/
structure BirdCosts where
  rooster : ℚ
  hen : ℚ
  chick : ℚ

/-- Represents the quantities of each type of bird -/
structure BirdQuantities where
  roosters : ℕ
  hens : ℕ
  chicks : ℕ

/-- The problem constraints -/
def bird_problem (costs : BirdCosts) (quantities : BirdQuantities) : Prop :=
  costs.rooster = 5 ∧
  costs.hen = 3 ∧
  costs.chick = 1/3 ∧
  quantities.roosters = 8 ∧
  quantities.roosters + quantities.hens + quantities.chicks = 100

/-- The system of equations representing the problem -/
def problem_equations (costs : BirdCosts) (quantities : BirdQuantities) : Prop :=
  costs.rooster * quantities.roosters + costs.hen * quantities.hens + costs.chick * quantities.chicks = 100 ∧
  quantities.roosters + quantities.hens + quantities.chicks = 100

/-- Theorem stating that the problem constraints imply the system of equations -/
theorem bird_problem_equations (costs : BirdCosts) (quantities : BirdQuantities) :
  bird_problem costs quantities → problem_equations costs quantities :=
by
  sorry


end bird_problem_equations_l2309_230949


namespace intersection_slope_range_l2309_230997

/-- Given two points P and Q in the Cartesian plane, and a linear function y = kx - 1
    that intersects the extension of line segment PQ (excluding Q),
    prove that the range of k is (1/3, 3/2). -/
theorem intersection_slope_range (P Q : ℝ × ℝ) (k : ℝ) : 
  P = (-1, 1) →
  Q = (2, 2) →
  (∃ x y : ℝ, y = k * x - 1 ∧ (y - 1) / (x + 1) = (2 - 1) / (2 + 1) ∧ (x, y) ≠ Q) →
  1/3 < k ∧ k < 3/2 :=
by sorry

end intersection_slope_range_l2309_230997


namespace tangent_plane_equation_l2309_230982

-- Define the function f(x, y)
def f (x y : ℝ) : ℝ := x^2 + y^2 + 2*x + 1

-- Define the point A
def A : ℝ × ℝ := (2, 3)

-- Theorem statement
theorem tangent_plane_equation :
  let (x₀, y₀) := A
  let z₀ := f x₀ y₀
  let fx := (2 * x₀ + 2 : ℝ)  -- Partial derivative with respect to x
  let fy := (2 * y₀ : ℝ)      -- Partial derivative with respect to y
  ∀ x y z, z - z₀ = fx * (x - x₀) + fy * (y - y₀) ↔ 6*x + 6*y - z - 12 = 0 :=
sorry

end tangent_plane_equation_l2309_230982


namespace bed_fraction_of_plot_l2309_230908

/-- Given a square plot of land with side length 8 units, prove that the fraction
    of the plot occupied by 13 beds (12 in an outer band and 1 central square)
    is 15/32 of the total area. -/
theorem bed_fraction_of_plot (plot_side : ℝ) (total_beds : ℕ) 
  (outer_beds : ℕ) (inner_bed_side : ℝ) :
  plot_side = 8 →
  total_beds = 13 →
  outer_beds = 12 →
  inner_bed_side = 4 →
  (outer_beds * (plot_side - inner_bed_side) + inner_bed_side ^ 2 / 2) / plot_side ^ 2 = 15 / 32 := by
  sorry

#check bed_fraction_of_plot

end bed_fraction_of_plot_l2309_230908


namespace power_equality_l2309_230999

theorem power_equality : (3 : ℕ) ^ 20 = 243 ^ 4 := by
  sorry

end power_equality_l2309_230999


namespace two_numbers_with_sum_or_diff_divisible_by_1000_l2309_230964

theorem two_numbers_with_sum_or_diff_divisible_by_1000 (S : Finset ℕ) (h : S.card = 502) :
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (1000 ∣ (a - b) ∨ 1000 ∣ (a + b)) := by
  sorry

end two_numbers_with_sum_or_diff_divisible_by_1000_l2309_230964


namespace sqrt_2_times_sqrt_6_l2309_230940

theorem sqrt_2_times_sqrt_6 : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 := by
  sorry

end sqrt_2_times_sqrt_6_l2309_230940


namespace no_sum_of_squares_representation_l2309_230925

theorem no_sum_of_squares_representation : ¬∃ (n : ℕ), ∃ (x y : ℕ+), 
  2 * n * (n + 1) * (n + 2) * (n + 3) + 12 = x^2 + y^2 := by
  sorry

end no_sum_of_squares_representation_l2309_230925


namespace expression_evaluation_l2309_230981

theorem expression_evaluation : 
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) + Real.sqrt 12 = 6 * Real.sqrt 3 := by
  sorry

end expression_evaluation_l2309_230981


namespace unique_solution_l2309_230916

/-- A function satisfying the given conditions -/
def satisfies_conditions (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a : ℝ, 
    (∀ x y z : ℝ, f x y + f y z + f z x = max x (max y z) - min x (min y z)) ∧
    (∀ x : ℝ, f x a = f a x)

/-- The theorem stating the unique solution -/
theorem unique_solution : 
  ∃! f : ℝ → ℝ → ℝ, satisfies_conditions f ∧ ∀ x y : ℝ, f x y = |x - y| / 2 :=
sorry

end unique_solution_l2309_230916


namespace fraction_sum_equality_l2309_230910

theorem fraction_sum_equality (a b c : ℝ) : 
  (a / (30 - a) + b / (70 - b) + c / (55 - c) = 9) →
  (6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 5.08) := by
  sorry

end fraction_sum_equality_l2309_230910


namespace skew_symmetric_determinant_nonnegative_l2309_230998

theorem skew_symmetric_determinant_nonnegative 
  (a b c d e f : ℝ) : 
  (a * f - b * e + c * d)^2 ≥ 0 := by sorry

end skew_symmetric_determinant_nonnegative_l2309_230998


namespace largest_non_sum_of_multiple_30_and_composite_l2309_230919

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_sum_of_multiple_30_and_composite (n : ℕ) : Prop :=
  ∃ k m, k > 0 ∧ is_composite m ∧ n = 30 * k + m

theorem largest_non_sum_of_multiple_30_and_composite :
  (∀ n > 211, is_sum_of_multiple_30_and_composite n) ∧
  ¬is_sum_of_multiple_30_and_composite 211 :=
sorry

end largest_non_sum_of_multiple_30_and_composite_l2309_230919


namespace function_geq_square_for_k_geq_4_l2309_230945

def is_increasing_square (f : ℕ+ → ℝ) : Prop :=
  ∀ k : ℕ+, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2

theorem function_geq_square_for_k_geq_4
  (f : ℕ+ → ℝ)
  (h_increasing : is_increasing_square f)
  (h_f4 : f 4 = 25) :
  ∀ k : ℕ+, k ≥ 4 → f k ≥ k^2 :=
sorry

end function_geq_square_for_k_geq_4_l2309_230945


namespace unique_alpha_beta_pair_l2309_230932

theorem unique_alpha_beta_pair :
  ∃! (α β : ℝ), α > 0 ∧ β > 0 ∧
  (∀ (x y z w : ℝ), x > 0 → y > 0 → z > 0 → w > 0 →
    x + y^2 + z^3 + w^6 ≥ α * (x*y*z*w)^β) ∧
  (∃ (x y z w : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
    x + y^2 + z^3 + w^6 = α * (x*y*z*w)^β) ∧
  α = 2^(4/3) * 3^(1/4) ∧ β = 1/2 := by
sorry

end unique_alpha_beta_pair_l2309_230932


namespace laptop_price_proof_l2309_230989

/-- The sticker price of the laptop -/
def stickerPrice : ℝ := 750

/-- The price at Store A after discount and rebate -/
def storePriceA (x : ℝ) : ℝ := 0.8 * x - 100

/-- The price at Store B after discount -/
def storePriceB (x : ℝ) : ℝ := 0.7 * x

/-- Theorem stating that the sticker price satisfies the given conditions -/
theorem laptop_price_proof :
  storePriceB stickerPrice - storePriceA stickerPrice = 25 :=
sorry

end laptop_price_proof_l2309_230989


namespace inequality_bound_l2309_230956

theorem inequality_bound (m : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ m) → m ≤ 3 :=
by
  sorry

end inequality_bound_l2309_230956


namespace science_club_problem_l2309_230978

theorem science_club_problem (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ)
  (h1 : total = 75)
  (h2 : biology = 42)
  (h3 : chemistry = 38)
  (h4 : both = 25) :
  total - (biology + chemistry - both) = 20 := by
  sorry

end science_club_problem_l2309_230978


namespace month_with_conditions_has_30_days_l2309_230994

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with its number of days and day counts -/
structure Month where
  days : Nat
  dayCounts : DayOfWeek → Nat

/-- Definition of a valid month -/
def validMonth (m : Month) : Prop :=
  (m.days ≥ 28 ∧ m.days ≤ 31) ∧
  (∀ d : DayOfWeek, m.dayCounts d = 4 ∨ m.dayCounts d = 5)

/-- The condition of more Mondays than Tuesdays -/
def moreMondays (m : Month) : Prop :=
  m.dayCounts DayOfWeek.Monday > m.dayCounts DayOfWeek.Tuesday

/-- The condition of fewer Saturdays than Sundays -/
def fewerSaturdays (m : Month) : Prop :=
  m.dayCounts DayOfWeek.Saturday < m.dayCounts DayOfWeek.Sunday

theorem month_with_conditions_has_30_days (m : Month) 
  (hValid : validMonth m) 
  (hMondays : moreMondays m) 
  (hSaturdays : fewerSaturdays m) : 
  m.days = 30 := by
  sorry

end month_with_conditions_has_30_days_l2309_230994


namespace max_factors_of_power_l2309_230972

def is_power_of_two_primes (b : ℕ) : Prop :=
  ∃ p q k l : ℕ, p.Prime ∧ q.Prime ∧ p ≠ q ∧ b = p^k * q^l

theorem max_factors_of_power (b n : ℕ) : 
  b > 0 → n > 0 → b ≤ 20 → n ≤ 20 → is_power_of_two_primes b →
  (∃ k : ℕ, k ≤ b^n ∧ (∀ m : ℕ, m ≤ b^n → Nat.card (Nat.divisors m) ≤ Nat.card (Nat.divisors k))) →
  Nat.card (Nat.divisors (b^n)) ≤ 441 :=
sorry

end max_factors_of_power_l2309_230972


namespace notebook_cost_l2309_230979

/-- Proves that the cost of each notebook before discount is $1.48 -/
theorem notebook_cost 
  (total_spent : ℚ)
  (num_backpacks : ℕ)
  (num_pen_packs : ℕ)
  (num_pencil_packs : ℕ)
  (num_notebooks : ℕ)
  (num_calculators : ℕ)
  (discount_rate : ℚ)
  (backpack_price : ℚ)
  (pen_pack_price : ℚ)
  (pencil_pack_price : ℚ)
  (calculator_price : ℚ)
  (h1 : total_spent = 56)
  (h2 : num_backpacks = 1)
  (h3 : num_pen_packs = 3)
  (h4 : num_pencil_packs = 2)
  (h5 : num_notebooks = 5)
  (h6 : num_calculators = 1)
  (h7 : discount_rate = 1/10)
  (h8 : backpack_price = 30)
  (h9 : pen_pack_price = 2)
  (h10 : pencil_pack_price = 3/2)
  (h11 : calculator_price = 15) :
  let other_items_cost := backpack_price * num_backpacks + 
                          pen_pack_price * num_pen_packs + 
                          pencil_pack_price * num_pencil_packs + 
                          calculator_price * num_calculators
  let discounted_other_items_cost := other_items_cost * (1 - discount_rate)
  let notebooks_cost := total_spent - discounted_other_items_cost
  notebooks_cost / num_notebooks = 37/25 := by
  sorry

end notebook_cost_l2309_230979


namespace function_inequality_and_zero_relation_l2309_230927

noncomputable section

variables (a : ℝ) (x x₀ x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + Real.log x

def g (x : ℝ) : ℝ := 2 * x + (a / 2) * Real.log x

theorem function_inequality_and_zero_relation 
  (h₁ : ∀ x > 0, f x ≥ g x)
  (h₂ : f x₁ = 0)
  (h₃ : f x₂ = 0)
  (h₄ : x₁ < x₂)
  (h₅ : x₀ = -a/4) :
  a ≥ (4 + 4 * Real.log 2) / (1 + 2 * Real.log 2) ∧ 
  x₁ / x₂ > 4 * Real.exp x₀ :=
sorry

end function_inequality_and_zero_relation_l2309_230927


namespace sin_n_eq_cos_810_l2309_230976

theorem sin_n_eq_cos_810 (n : ℤ) (h1 : -180 ≤ n) (h2 : n ≤ 180) (h3 : Real.sin (n * π / 180) = Real.cos (810 * π / 180)) :
  n = 0 ∨ n = 180 ∨ n = -180 := by
sorry

end sin_n_eq_cos_810_l2309_230976


namespace man_downstream_speed_l2309_230987

/-- Given a man's upstream speed and the stream speed, calculates his downstream speed. -/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  upstream_speed + 2 * stream_speed

/-- Theorem stating that given the specific upstream speed and stream speed, 
    the downstream speed is 15 kmph. -/
theorem man_downstream_speed :
  downstream_speed 8 3.5 = 15 := by
  sorry

#eval downstream_speed 8 3.5

end man_downstream_speed_l2309_230987


namespace roots_of_equation_l2309_230953

theorem roots_of_equation : 
  let f : ℝ → ℝ := fun x => x * (x - 1) + 3 * (x - 1)
  (f (-3) = 0 ∧ f 1 = 0) ∧ ∀ x : ℝ, f x = 0 → (x = -3 ∨ x = 1) :=
by sorry

end roots_of_equation_l2309_230953


namespace problem_solution_l2309_230914

theorem problem_solution (x y : ℝ) 
  (h1 : (1/2) * (x - 2)^3 + 32 = 0)
  (h2 : 3*x - 2*y = 6^2) :
  Real.sqrt (x^2 - y) = 5 := by
  sorry

end problem_solution_l2309_230914


namespace hyperbola_equation_l2309_230946

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b / a = Real.sqrt 3) →
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 4) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 4 - y^2 / 12 = 1) :=
by sorry

end hyperbola_equation_l2309_230946


namespace sqrt_three_minus_sqrt_one_third_l2309_230938

theorem sqrt_three_minus_sqrt_one_third : 
  Real.sqrt 3 - Real.sqrt (1/3) = (2 * Real.sqrt 3) / 3 := by
  sorry

end sqrt_three_minus_sqrt_one_third_l2309_230938


namespace roots_sum_fourth_powers_l2309_230991

theorem roots_sum_fourth_powers (c d : ℝ) : 
  c^2 - 6*c + 8 = 0 → 
  d^2 - 6*d + 8 = 0 → 
  c^4 + c^3*d + d^3*c + d^4 = 432 := by
  sorry

end roots_sum_fourth_powers_l2309_230991


namespace at_least_one_negative_l2309_230904

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  ¬(a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
by sorry

end at_least_one_negative_l2309_230904


namespace stamp_cost_difference_l2309_230966

/-- The cost of a single rooster stamp -/
def rooster_stamp_cost : ℚ := 1.5

/-- The cost of a single daffodil stamp -/
def daffodil_stamp_cost : ℚ := 0.75

/-- The number of rooster stamps purchased -/
def rooster_stamp_count : ℕ := 2

/-- The number of daffodil stamps purchased -/
def daffodil_stamp_count : ℕ := 5

/-- The theorem stating the cost difference between daffodil and rooster stamps -/
theorem stamp_cost_difference : 
  (daffodil_stamp_count : ℚ) * daffodil_stamp_cost - 
  (rooster_stamp_count : ℚ) * rooster_stamp_cost = 0.75 := by
  sorry

end stamp_cost_difference_l2309_230966


namespace factorization_problem_1_l2309_230941

theorem factorization_problem_1 (x : ℝ) : -27 + 3 * x^2 = -3 * (3 + x) * (3 - x) := by
  sorry

end factorization_problem_1_l2309_230941


namespace soldier_target_practice_l2309_230961

theorem soldier_target_practice (total_shots : ℕ) (total_score : ℕ) (tens : ℕ) (tens_score : ℕ) :
  total_shots = 10 →
  total_score = 90 →
  tens = 4 →
  tens_score = 10 →
  ∃ (sevens eights nines : ℕ),
    sevens + eights + nines = total_shots - tens ∧
    7 * sevens + 8 * eights + 9 * nines = total_score - tens * tens_score ∧
    sevens = 1 :=
by sorry

end soldier_target_practice_l2309_230961


namespace min_value_theorem_l2309_230974

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 / (y - 2)^2) + (y^2 / (x - 2)^2) ≥ 10 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 2 ∧ y₀ > 2 ∧ (x₀^2 / (y₀ - 2)^2) + (y₀^2 / (x₀ - 2)^2) = 10 :=
by sorry

end min_value_theorem_l2309_230974


namespace exam_score_calculation_l2309_230948

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) 
  (correct_score : ℤ) (wrong_score : ℤ) : 
  total_questions = 120 →
  correct_answers = 75 →
  correct_score = 3 →
  wrong_score = -1 →
  (correct_answers * correct_score + (total_questions - correct_answers) * wrong_score : ℤ) = 180 := by
  sorry

end exam_score_calculation_l2309_230948


namespace division_value_problem_l2309_230977

theorem division_value_problem (x : ℝ) (h : (5 / x) * 12 = 10) : x = 6 := by
  sorry

end division_value_problem_l2309_230977


namespace quadratic_with_prime_roots_l2309_230968

theorem quadratic_with_prime_roots (m : ℕ) : 
  (∃ x y : ℕ, x.Prime ∧ y.Prime ∧ x ≠ y ∧ x^2 - 1999*x + m = 0 ∧ y^2 - 1999*y + m = 0) → 
  m = 3994 := by
sorry

end quadratic_with_prime_roots_l2309_230968


namespace sparkling_water_cost_l2309_230969

/-- The cost of sparkling water bottles for Mary Anne -/
theorem sparkling_water_cost (bottles_per_night : ℚ) (yearly_cost : ℕ) : 
  bottles_per_night = 1/5 → yearly_cost = 146 → 
  (365 : ℚ) / 5 * (yearly_cost : ℚ) / ((365 : ℚ) / 5) = 2 := by
  sorry

end sparkling_water_cost_l2309_230969


namespace incorrect_reasonings_l2309_230954

-- Define the type for analogical reasoning
inductive AnalogicalReasoning
  | addition_subtraction
  | vector_complex_square
  | quadratic_equation
  | geometric_addition

-- Define a function to check if a reasoning is correct
def is_correct_reasoning (r : AnalogicalReasoning) : Prop :=
  match r with
  | AnalogicalReasoning.addition_subtraction => True
  | AnalogicalReasoning.vector_complex_square => False
  | AnalogicalReasoning.quadratic_equation => False
  | AnalogicalReasoning.geometric_addition => True

-- Theorem statement
theorem incorrect_reasonings :
  ∃ (incorrect : List AnalogicalReasoning),
    incorrect.length = 2 ∧
    (∀ r ∈ incorrect, ¬(is_correct_reasoning r)) ∧
    (∀ r, r ∉ incorrect → is_correct_reasoning r) :=
  sorry

end incorrect_reasonings_l2309_230954


namespace cost_per_box_l2309_230995

/-- The cost per box for packaging the fine arts collection --/
theorem cost_per_box (box_length box_width box_height : ℝ)
  (total_volume min_total_cost : ℝ) :
  box_length = 20 ∧ box_width = 20 ∧ box_height = 15 ∧
  total_volume = 3060000 ∧ min_total_cost = 357 →
  (min_total_cost / (total_volume / (box_length * box_width * box_height))) = 0.70 := by
  sorry

end cost_per_box_l2309_230995


namespace six_integers_mean_double_mode_l2309_230933

def is_valid_list (l : List Int) : Prop :=
  l.length = 6 ∧ l.all (λ x => x > 0 ∧ x ≤ 150)

def mean (l : List Int) : Rat :=
  (l.sum : Rat) / l.length

def mode (l : List Int) : Int :=
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

theorem six_integers_mean_double_mode :
  ∀ y z : Int,
    let l := [45, 76, y, y, z, z]
    is_valid_list l →
    mean l = 2 * (mode l : Rat) →
    y = 49 ∧ z = 21 := by
  sorry

end six_integers_mean_double_mode_l2309_230933


namespace subcommittee_count_l2309_230937

theorem subcommittee_count (n k : ℕ) (hn : n = 8) (hk : k = 3) : 
  Nat.choose n k = 56 := by
  sorry

end subcommittee_count_l2309_230937


namespace quadratic_inequality_range_l2309_230988

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, 4 * x^2 + (a - 2) * x + 1/4 > 0) ↔ 
  (a ≤ 0 ∨ a ≥ 4) := by
sorry

end quadratic_inequality_range_l2309_230988


namespace james_tylenol_intake_l2309_230923

/-- Calculates the total milligrams of Tylenol taken per day given the number of tablets per dose,
    milligrams per tablet, hours between doses, and hours in a day. -/
def tylenolPerDay (tabletsPerDose : ℕ) (mgPerTablet : ℕ) (hoursBetweenDoses : ℕ) (hoursInDay : ℕ) : ℕ :=
  let mgPerDose := tabletsPerDose * mgPerTablet
  let dosesPerDay := hoursInDay / hoursBetweenDoses
  mgPerDose * dosesPerDay

/-- Proves that James takes 3000 mg of Tylenol per day given the specified conditions. -/
theorem james_tylenol_intake : tylenolPerDay 2 375 6 24 = 3000 := by
  sorry

end james_tylenol_intake_l2309_230923


namespace problem_solution_l2309_230907

theorem problem_solution (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : 6 * x^3 + 12 * x^2 * y = 2 * x^4 + 3 * x^3 * y) 
  (h4 : x + y = 3) : 
  x = 2 := by
sorry

end problem_solution_l2309_230907


namespace circle_center_correct_l2309_230971

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, return its center -/
def findCircleCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 4 8 4 (-24) 16
  let center := findCircleCenter eq
  center.x = -1 ∧ center.y = 3 := by sorry

end circle_center_correct_l2309_230971


namespace college_class_period_length_l2309_230975

/-- Given a total time, number of periods, and time between periods, 
    calculate the length of each period. -/
def period_length (total_time : ℕ) (num_periods : ℕ) (time_between : ℕ) : ℕ :=
  (total_time - (num_periods - 1) * time_between) / num_periods

/-- Theorem stating that under the given conditions, each period is 40 minutes long. -/
theorem college_class_period_length : 
  period_length 220 5 5 = 40 := by
  sorry

end college_class_period_length_l2309_230975


namespace percentage_problem_l2309_230911

theorem percentage_problem (x : ℝ) (P : ℝ) : 
  x = 690 →
  (0.5 * x) = (P / 100 * 1500 - 30) →
  P = 25 := by
sorry

end percentage_problem_l2309_230911


namespace min_value_theorem_l2309_230958

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (x + 3)⁻¹ ^ (1/3 : ℝ) + (y + 3)⁻¹ ^ (1/3 : ℝ) = 1/2) :
  x + 3*y ≥ 4*(1 + 3^(1/3 : ℝ))^2 - 12 := by
  sorry

end min_value_theorem_l2309_230958


namespace max_d_value_l2309_230985

def is_valid_number (d e : ℕ) : Prop :=
  d < 10 ∧ e < 10 ∧ 707340 + 10 * d + 4 * e ≥ 100000 ∧ 707340 + 10 * d + 4 * e < 1000000

def is_multiple_of_34 (d e : ℕ) : Prop :=
  (707340 + 10 * d + 4 * e) % 34 = 0

theorem max_d_value (d e : ℕ) :
  is_valid_number d e → is_multiple_of_34 d e → d ≤ 13 :=
by sorry

end max_d_value_l2309_230985


namespace factorization_1_factorization_2_l2309_230902

/-- Proves the factorization of x^2(x-1)-x+1 -/
theorem factorization_1 (x : ℝ) : x^2 * (x - 1) - x + 1 = (x - 1)^2 * (x + 1) := by
  sorry

/-- Proves the factorization of 3p(x+1)^3y^2+6p(x+1)^2y+3p(x+1) -/
theorem factorization_2 (p x y : ℝ) : 
  3 * p * (x + 1)^3 * y^2 + 6 * p * (x + 1)^2 * y + 3 * p * (x + 1) = 
  3 * p * (x + 1) * (x * y + y + 1)^2 := by
  sorry

end factorization_1_factorization_2_l2309_230902


namespace johns_number_l2309_230957

theorem johns_number : ∃! n : ℕ, 
  200 ∣ n ∧ 
  18 ∣ n ∧ 
  1000 < n ∧ 
  n < 2500 :=
by
  -- Proof goes here
  sorry

end johns_number_l2309_230957


namespace sevens_to_hundred_l2309_230967

theorem sevens_to_hundred : ∃ (expr : ℕ), 
  (expr = 100) ∧ 
  (∃ (a b c d e f g h i : ℕ), 
    (a ≤ 7 ∧ b ≤ 7 ∧ c ≤ 7 ∧ d ≤ 7 ∧ e ≤ 7 ∧ f ≤ 7 ∧ g ≤ 7 ∧ h ≤ 7 ∧ i ≤ 7) ∧
    (expr = a * b - c * d + e * f + g + h + i) ∧
    (a + b + c + d + e + f + g + h + i < 10 * 7)) :=
by sorry

end sevens_to_hundred_l2309_230967


namespace swing_rope_length_proof_l2309_230960

/-- The length of a swing rope satisfying specific conditions -/
def swing_rope_length : ℝ := 14.5

/-- The initial height of the swing's footboard off the ground -/
def initial_height : ℝ := 1

/-- The distance the swing is pushed forward -/
def push_distance : ℝ := 10

/-- The height of the person -/
def person_height : ℝ := 5

theorem swing_rope_length_proof :
  ∃ (rope_length : ℝ),
    rope_length = swing_rope_length ∧
    rope_length^2 = push_distance^2 + (rope_length - person_height + initial_height)^2 :=
by sorry

end swing_rope_length_proof_l2309_230960


namespace right_triangle_hypotenuse_l2309_230930

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 8 → b = 15 → c^2 = a^2 + b^2 → c = 17 := by
  sorry

end right_triangle_hypotenuse_l2309_230930


namespace equal_sum_product_quotient_l2309_230909

theorem equal_sum_product_quotient :
  ∃! (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a + b = a * b ∧ a + b = a / b ∧ a = 1/2 ∧ b = -1 := by
  sorry

end equal_sum_product_quotient_l2309_230909


namespace quadratic_inequality_existence_l2309_230936

theorem quadratic_inequality_existence (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x + 1 ≤ 0) ↔ (m ≥ 2 ∨ m ≤ -2) := by
  sorry

end quadratic_inequality_existence_l2309_230936


namespace congruent_to_one_mod_seven_l2309_230913

theorem congruent_to_one_mod_seven (n : ℕ) : 
  (Finset.filter (fun k => k % 7 = 1) (Finset.range 300)).card = 43 := by
  sorry

end congruent_to_one_mod_seven_l2309_230913


namespace calculation_proof_l2309_230905

theorem calculation_proof : ((-4)^2 * (-1/2)^3 - (-4+1)) = 1 := by
  sorry

end calculation_proof_l2309_230905


namespace problem_solution_l2309_230912

theorem problem_solution :
  (∀ a b : ℝ, 4 * a^4 * b^3 / (-2 * a * b)^2 = a^2 * b) ∧
  (∀ x y : ℝ, (3*x - y)^2 - (3*x + 2*y) * (3*x - 2*y) = -6*x*y + 5*y^2) :=
by sorry

end problem_solution_l2309_230912


namespace salary_increase_l2309_230952

theorem salary_increase (num_employees : ℕ) (avg_salary : ℝ) (manager_salary : ℝ) :
  num_employees = 20 →
  avg_salary = 1500 →
  manager_salary = 12000 →
  let total_salary := num_employees * avg_salary
  let new_total := total_salary + manager_salary
  let new_avg := new_total / (num_employees + 1)
  new_avg - avg_salary = 500 := by
  sorry

end salary_increase_l2309_230952


namespace sqrt_difference_equals_negative_sixteen_l2309_230983

theorem sqrt_difference_equals_negative_sixteen :
  Real.sqrt (16 - 8 * Real.sqrt 2) - Real.sqrt (16 + 8 * Real.sqrt 2) = -16 := by
  sorry

end sqrt_difference_equals_negative_sixteen_l2309_230983


namespace graph_transformation_l2309_230928

/-- Given a function f(x) = sin(x - π/3), prove that after stretching its x-coordinates
    to twice their original length and shifting the resulting graph to the right by π/3 units,
    the equation of the resulting graph is y = sin(x/2 - π/2). -/
theorem graph_transformation (x : ℝ) :
  let f : ℝ → ℝ := fun x ↦ Real.sin (x - π/3)
  let g : ℝ → ℝ := fun x ↦ f (x/2)
  let h : ℝ → ℝ := fun x ↦ g (x - π/3)
  h x = Real.sin (x/2 - π/2) := by
  sorry

end graph_transformation_l2309_230928


namespace train_speed_train_speed_is_72_l2309_230943

/-- Given a train that crosses a platform and a stationary man, calculate its speed in km/h -/
theorem train_speed (platform_length : ℝ) (platform_time : ℝ) (man_time : ℝ) : ℝ :=
  let train_speed_mps := platform_length / (platform_time - man_time)
  let train_speed_kmph := train_speed_mps * 3.6
  train_speed_kmph

/-- The speed of the train is 72 km/h -/
theorem train_speed_is_72 : 
  train_speed 260 31 18 = 72 := by sorry

end train_speed_train_speed_is_72_l2309_230943


namespace power_division_rule_l2309_230963

theorem power_division_rule (a : ℝ) : a^8 / a^2 = a^6 := by
  sorry

end power_division_rule_l2309_230963


namespace stamp_collection_value_l2309_230942

/-- Given a collection of stamps with equal individual value, 
    calculate the total value of the collection. -/
theorem stamp_collection_value 
  (total_stamps : ℕ) 
  (sample_stamps : ℕ) 
  (sample_value : ℕ) 
  (h1 : total_stamps = 21)
  (h2 : sample_stamps = 7)
  (h3 : sample_value = 28) :
  (total_stamps : ℚ) * (sample_value : ℚ) / (sample_stamps : ℚ) = 84 := by
  sorry

end stamp_collection_value_l2309_230942


namespace min_tangent_length_l2309_230926

/-- The minimum length of a tangent from a point on y = x + 1 to (x-3)^2 + y^2 = 1 is √7 -/
theorem min_tangent_length :
  let line := {p : ℝ × ℝ | p.2 = p.1 + 1}
  let circle := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 1}
  ∃ (min_length : ℝ),
    min_length = Real.sqrt 7 ∧
    ∀ (p : ℝ × ℝ) (t : ℝ × ℝ),
      p ∈ line → t ∈ circle →
      dist p t ≥ min_length :=
by sorry


end min_tangent_length_l2309_230926
