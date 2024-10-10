import Mathlib

namespace square_rectangle_area_difference_l2565_256525

theorem square_rectangle_area_difference : 
  let square_side : ℝ := 8
  let rect_length : ℝ := 10
  let rect_width : ℝ := 5
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  square_area - rect_area = 14 := by
  sorry

end square_rectangle_area_difference_l2565_256525


namespace cantaloupes_left_l2565_256509

/-- Represents the number of melons and their prices --/
structure MelonSales where
  cantaloupe_price : ℕ
  honeydew_price : ℕ
  initial_cantaloupes : ℕ
  initial_honeydews : ℕ
  dropped_cantaloupes : ℕ
  rotten_honeydews : ℕ
  remaining_honeydews : ℕ
  total_revenue : ℕ

/-- Theorem stating the number of cantaloupes left at the end of the day --/
theorem cantaloupes_left (s : MelonSales)
    (h1 : s.cantaloupe_price = 2)
    (h2 : s.honeydew_price = 3)
    (h3 : s.initial_cantaloupes = 30)
    (h4 : s.initial_honeydews = 27)
    (h5 : s.dropped_cantaloupes = 2)
    (h6 : s.rotten_honeydews = 3)
    (h7 : s.remaining_honeydews = 9)
    (h8 : s.total_revenue = 85) :
    s.initial_cantaloupes - s.dropped_cantaloupes -
    ((s.total_revenue - (s.honeydew_price * (s.initial_honeydews - s.rotten_honeydews - s.remaining_honeydews))) / s.cantaloupe_price) = 8 := by
  sorry

end cantaloupes_left_l2565_256509


namespace three_heads_probability_l2565_256545

/-- The probability of getting heads on a single flip of a biased coin. -/
def p_heads : ℚ := 1 / 3

/-- The number of consecutive flips we're considering. -/
def n_flips : ℕ := 3

/-- The probability of getting n_flips consecutive heads. -/
def p_all_heads : ℚ := p_heads ^ n_flips

theorem three_heads_probability :
  p_all_heads = 1 / 27 := by sorry

end three_heads_probability_l2565_256545


namespace representation_of_2021_l2565_256532

theorem representation_of_2021 : ∃ (a b c : ℤ), 2021 = a^2 - b^2 + c^2 := by
  -- We need to prove that there exist integers a, b, and c such that
  -- 2021 = a^2 - b^2 + c^2
  sorry

end representation_of_2021_l2565_256532


namespace right_triangle_equality_l2565_256550

theorem right_triangle_equality (a b c p S : ℝ) : 
  a ≤ b → b ≤ c → 
  2 * p = a + b + c → 
  S = (1/2) * a * b → 
  a^2 + b^2 = c^2 → 
  p * (p - c) = (p - a) * (p - b) ∧ p * (p - c) = S := by
  sorry

end right_triangle_equality_l2565_256550


namespace game_choices_l2565_256588

theorem game_choices (p : ℝ) (n : ℕ) 
  (h1 : p = 0.9375) 
  (h2 : p = 1 - 1 / n) : n = 16 := by
  sorry

end game_choices_l2565_256588


namespace unique_number_between_2_and_5_l2565_256546

theorem unique_number_between_2_and_5 (n : ℕ) : 
  2 < n ∧ n < 5 ∧ n < 10 ∧ n < 4 → n = 3 := by
  sorry

end unique_number_between_2_and_5_l2565_256546


namespace problem_solution_l2565_256527

-- Define the propositions
def proposition_A (x : ℝ) : Prop := (x^2 - 4*x + 3 = 0) → (x = 3)
def proposition_B (x : ℝ) : Prop := (x > 1) → (|x| > 0)
def proposition_C (p q : Prop) : Prop := (¬p ∧ ¬q) → (¬p ∧ ¬q)
def proposition_D : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- Define the correctness of each statement
def statement_A_correct : Prop :=
  ∀ x : ℝ, (x ≠ 3 → x^2 - 4*x + 3 ≠ 0) ↔ proposition_A x

def statement_B_correct : Prop :=
  (∀ x : ℝ, x > 1 → |x| > 0) ∧ (∃ x : ℝ, |x| > 0 ∧ x ≤ 1)

def statement_C_incorrect : Prop :=
  ∃ p q : Prop, ¬p ∧ ¬q ∧ ¬(proposition_C p q)

def statement_D_correct : Prop :=
  (¬proposition_D) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0)

-- Main theorem
theorem problem_solution :
  statement_A_correct ∧ statement_B_correct ∧ statement_C_incorrect ∧ statement_D_correct :=
sorry

end problem_solution_l2565_256527


namespace guitar_price_proof_l2565_256547

/-- The price Gerald paid for the guitar -/
def gerald_price : ℝ := 250

/-- The price Hendricks paid for the guitar -/
def hendricks_price : ℝ := 200

/-- The percentage discount Hendricks got compared to Gerald's price -/
def discount_percentage : ℝ := 20

theorem guitar_price_proof :
  hendricks_price = gerald_price * (1 - discount_percentage / 100) →
  gerald_price = 250 := by
  sorry

end guitar_price_proof_l2565_256547


namespace equation_represents_ellipse_and_hyperbola_l2565_256526

-- Define the equation
def equation (x y : ℝ) : Prop := y^4 - 6*x^4 = 3*y^2 - 2

-- Define what constitutes an ellipse in this context
def is_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ (∀ x y : ℝ, f x y ↔ y^2 = a*x^2 + b)

-- Define what constitutes a hyperbola in this context
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, f x y ↔ y^2 = a*x^2 + b)

-- Theorem statement
theorem equation_represents_ellipse_and_hyperbola :
  (is_ellipse equation) ∧ (is_hyperbola equation) :=
sorry

end equation_represents_ellipse_and_hyperbola_l2565_256526


namespace min_neighbors_2005_points_l2565_256561

/-- The number of points on the circumference of the circle -/
def num_points : ℕ := 2005

/-- The maximum angle (in degrees) that a chord can subtend at the center for its endpoints to be considered neighbors -/
def max_neighbor_angle : ℝ := 10

/-- A function that calculates the minimum number of neighbor pairs given the number of points and maximum neighbor angle -/
noncomputable def min_neighbor_pairs (n : ℕ) (max_angle : ℝ) : ℕ := sorry

/-- Theorem stating that the minimum number of neighbor pairs for 2005 points with a 10° maximum angle is 56430 -/
theorem min_neighbors_2005_points :
  min_neighbor_pairs num_points max_neighbor_angle = 56430 := by sorry

end min_neighbors_2005_points_l2565_256561


namespace age_ratio_proof_l2565_256575

/-- Given three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  a + b + c = 32 →  -- The total of the ages of a, b, and c is 32
  b = 12 →  -- b is 12 years old
  b = 2 * c :=  -- The ratio of b's age to c's age is 2:1
by
  sorry

end age_ratio_proof_l2565_256575


namespace remainder_sum_l2565_256530

theorem remainder_sum (x y : ℤ) 
  (hx : x ≡ 47 [ZMOD 60])
  (hy : y ≡ 26 [ZMOD 45]) :
  x + y ≡ 13 [ZMOD 15] := by
sorry

end remainder_sum_l2565_256530


namespace solve_inequality_minimum_a_l2565_256563

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - a * |x - 1|

-- Part 1
theorem solve_inequality (x : ℝ) : 
  f (-2) x > 5 ↔ x < -4/3 ∨ x > 2 := by sorry

-- Part 2
theorem minimum_a : 
  (∃ (a : ℝ), ∀ (x : ℝ), f a x ≤ a * |x + 3|) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), f a x ≤ a * |x + 3|) → a ≥ 1/2) := by sorry

end solve_inequality_minimum_a_l2565_256563


namespace nickel_probability_is_one_fourth_l2565_256599

/-- Represents the types of coins in the jar -/
inductive Coin
  | Dime
  | Nickel
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℕ
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- The total value of each coin type in cents -/
def totalValue : Coin → ℕ
  | Coin.Dime => 1000
  | Coin.Nickel => 500
  | Coin.Penny => 200

/-- The number of coins of each type -/
def coinCount (c : Coin) : ℕ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℕ := coinCount Coin.Dime + coinCount Coin.Nickel + coinCount Coin.Penny

/-- The probability of selecting a nickel -/
def nickelProbability : ℚ := coinCount Coin.Nickel / totalCoins

theorem nickel_probability_is_one_fourth :
  nickelProbability = 1 / 4 := by sorry

end nickel_probability_is_one_fourth_l2565_256599


namespace paving_cost_example_l2565_256507

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with length 5.5 m and width 3.75 m
    at a rate of $600 per square metre is $12,375. -/
theorem paving_cost_example : paving_cost 5.5 3.75 600 = 12375 := by
  sorry

end paving_cost_example_l2565_256507


namespace total_lives_calculation_l2565_256576

/-- Given 6 initial players, 9 additional players, and 5 lives per player,
    the total number of lives is 75. -/
theorem total_lives_calculation (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ)
    (h1 : initial_players = 6)
    (h2 : additional_players = 9)
    (h3 : lives_per_player = 5) :
    (initial_players + additional_players) * lives_per_player = 75 := by
  sorry

end total_lives_calculation_l2565_256576


namespace baker_cakes_problem_l2565_256543

theorem baker_cakes_problem (initial_cakes : ℕ) (sold_cakes : ℕ) (extra_sold : ℕ) :
  initial_cakes = 8 →
  sold_cakes = 145 →
  extra_sold = 6 →
  ∃ (new_cakes : ℕ), 
    new_cakes + initial_cakes = sold_cakes + extra_sold ∧
    new_cakes = 131 :=
by sorry

end baker_cakes_problem_l2565_256543


namespace symmetrical_point_not_in_third_quadrant_l2565_256555

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the symmetrical point with respect to the y-axis -/
def symmetricalPointY (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Checks if a point is in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The main theorem to prove -/
theorem symmetrical_point_not_in_third_quadrant :
  let p := Point.mk (-3) 4
  let symmetricalP := symmetricalPointY p
  ¬(isInThirdQuadrant symmetricalP) := by
  sorry


end symmetrical_point_not_in_third_quadrant_l2565_256555


namespace man_birth_year_proof_l2565_256573

theorem man_birth_year_proof : ∃! x : ℤ,
  x^2 - x = 1640 ∧
  2*(x + 2*x) = 2*x ∧
  x^2 - x < 1825 := by
  sorry

end man_birth_year_proof_l2565_256573


namespace line_through_points_l2565_256544

/-- Given a line y = ax + b passing through points (3, -2) and (7, 14), prove that a + b = -10 -/
theorem line_through_points (a b : ℝ) : 
  (∀ x y : ℝ, y = a * x + b) → 
  (-2 : ℝ) = a * 3 + b → 
  (14 : ℝ) = a * 7 + b → 
  a + b = -10 := by sorry

end line_through_points_l2565_256544


namespace smallest_difference_PR_QR_l2565_256503

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  PQ : ℕ
  QR : ℕ
  PR : ℕ

/-- Checks if the given side lengths form a valid triangle -/
def is_valid_triangle (t : Triangle) : Prop :=
  t.PQ + t.QR > t.PR ∧ t.PQ + t.PR > t.QR ∧ t.QR + t.PR > t.PQ

/-- Represents the conditions of the problem -/
def satisfies_conditions (t : Triangle) : Prop :=
  t.PQ + t.QR + t.PR = 2023 ∧
  t.PQ ≤ t.QR ∧ t.QR < t.PR ∧
  is_valid_triangle t

/-- The main theorem stating the smallest possible difference between PR and QR -/
theorem smallest_difference_PR_QR :
  ∃ (t : Triangle), satisfies_conditions t ∧
  ∀ (t' : Triangle), satisfies_conditions t' → t.PR - t.QR ≤ t'.PR - t'.QR ∧
  t.PR - t.QR = 13 :=
sorry

end smallest_difference_PR_QR_l2565_256503


namespace males_with_college_degree_only_count_l2565_256598

/-- Represents the employee demographics of a company -/
structure CompanyDemographics where
  total_employees : Nat
  total_females : Nat
  employees_with_advanced_degrees : Nat
  females_with_advanced_degrees : Nat

/-- Calculates the number of males with a college degree only -/
def males_with_college_degree_only (demo : CompanyDemographics) : Nat :=
  let total_males := demo.total_employees - demo.total_females
  let males_with_advanced_degrees := demo.employees_with_advanced_degrees - demo.females_with_advanced_degrees
  total_males - males_with_advanced_degrees

/-- Theorem stating the number of males with a college degree only -/
theorem males_with_college_degree_only_count 
  (demo : CompanyDemographics) 
  (h1 : demo.total_employees = 180)
  (h2 : demo.total_females = 110)
  (h3 : demo.employees_with_advanced_degrees = 90)
  (h4 : demo.females_with_advanced_degrees = 55) :
  males_with_college_degree_only demo = 35 := by
  sorry

#eval males_with_college_degree_only { 
  total_employees := 180, 
  total_females := 110, 
  employees_with_advanced_degrees := 90, 
  females_with_advanced_degrees := 55 
}

end males_with_college_degree_only_count_l2565_256598


namespace tangent_line_at_point_one_l2565_256559

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem tangent_line_at_point_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  let tangent_line (x : ℝ) : ℝ := m * (x - x₀) + y₀
  (∀ x, tangent_line x = -3 * x + 2) ∧ y₀ = -1 := by sorry

end tangent_line_at_point_one_l2565_256559


namespace permutation_and_combination_problem_l2565_256520

-- Define the permutation function
def A (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Define the combination function
def C (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem permutation_and_combination_problem :
  ∃ (x : ℕ), x > 0 ∧ 7 * A 6 x = 20 * A 7 (x - 1) ∧ 
  x = 3 ∧
  C 20 (20 - x) + C (17 + x) (x - 1) = 1330 :=
sorry

end permutation_and_combination_problem_l2565_256520


namespace f_composition_neg_two_l2565_256511

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

-- State the theorem
theorem f_composition_neg_two : f (f (-2)) = 1/2 := by sorry

end f_composition_neg_two_l2565_256511


namespace sisters_height_l2565_256505

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Converts inches to feet and remaining inches -/
def inches_to_feet_and_inches (inches : ℕ) : ℕ × ℕ :=
  (inches / 12, inches % 12)

/-- Represents a height in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ
  h_valid : inches < 12

theorem sisters_height 
  (sunflower_height_feet : ℕ)
  (height_difference_inches : ℕ)
  (h_sunflower : sunflower_height_feet = 6)
  (h_difference : height_difference_inches = 21) :
  let sunflower_height_inches := feet_to_inches sunflower_height_feet
  let sister_height_inches := sunflower_height_inches - height_difference_inches
  let (sister_feet, sister_inches) := inches_to_feet_and_inches sister_height_inches
  Height.mk sister_feet sister_inches (by sorry) = Height.mk 4 3 (by sorry) :=
by sorry

end sisters_height_l2565_256505


namespace sum_of_last_two_digits_of_modified_fibonacci_factorial_series_l2565_256564

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 55]

def last_two_digits (n : ℕ) : ℕ := n % 100

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem sum_of_last_two_digits_of_modified_fibonacci_factorial_series :
  (fibonacci_factorial_series.map (λ n => last_two_digits (factorial n))).sum % 100 = 5 := by
  sorry

end sum_of_last_two_digits_of_modified_fibonacci_factorial_series_l2565_256564


namespace not_enough_money_l2565_256572

/-- The cost of a single storybook in yuan -/
def storybook_cost : ℕ := 18

/-- The number of storybooks to be purchased -/
def num_books : ℕ := 12

/-- The available money in yuan -/
def available_money : ℕ := 200

/-- Theorem stating that the available money is not enough to buy the desired number of storybooks -/
theorem not_enough_money : storybook_cost * num_books > available_money := by
  sorry

end not_enough_money_l2565_256572


namespace valentines_day_equality_l2565_256523

theorem valentines_day_equality (m d : ℕ) : 
  (∃ k : ℕ, 
    5 * m = 3 * k + 2 * (d - 3) ∧ 
    4 * d = 2 * k + 2 * (m - 2)) → 
  m = d :=
by
  sorry

end valentines_day_equality_l2565_256523


namespace count_valid_m_l2565_256519

theorem count_valid_m : ∃! (s : Finset ℕ), 
  (∀ m ∈ s, m > 0 ∧ (2520 : ℤ) % (m^2 - 2) = 0) ∧
  (∀ m : ℕ, m > 0 ∧ (2520 : ℤ) % (m^2 - 2) = 0 → m ∈ s) ∧
  s.card = 5 := by sorry

end count_valid_m_l2565_256519


namespace arithmetic_sequence_properties_l2565_256591

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  d : ℤ
  h1 : a 4 = -15
  h2 : d = 3
  h3 : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def SumOfTerms (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n * (seq.a 1 + seq.a n)) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 3 * n - 27) ∧
  ∃ min : ℤ, min = -108 ∧ ∀ n : ℕ, SumOfTerms seq n ≥ min :=
sorry

end arithmetic_sequence_properties_l2565_256591


namespace diamond_symmetry_points_l2565_256558

/-- The diamond operation -/
def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

/-- The set of points (x, y) satisfying x ⋄ y = y ⋄ x -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | diamond p.1 p.2 = diamond p.2 p.1}

/-- Two lines in ℝ² -/
def two_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = p.2 ∨ p.1 = -p.2}

theorem diamond_symmetry_points :
  S = two_lines ∪ ({0} : Set ℝ).prod Set.univ ∪ Set.univ.prod ({0} : Set ℝ) :=
sorry

end diamond_symmetry_points_l2565_256558


namespace min_workers_theorem_l2565_256506

/-- Represents the company's profit scenario -/
structure CompanyProfit where
  maintenance_cost : ℕ
  worker_wage : ℕ
  production_rate : ℕ
  gadget_price : ℚ
  workday_length : ℕ

/-- Calculates the minimum number of workers required for profit -/
def min_workers_for_profit (c : CompanyProfit) : ℕ :=
  Nat.succ (Nat.ceil ((c.maintenance_cost : ℚ) / 
    (c.production_rate * c.workday_length * c.gadget_price - c.worker_wage * c.workday_length)))

/-- Theorem stating the minimum number of workers required for profit -/
theorem min_workers_theorem (c : CompanyProfit) 
  (h1 : c.maintenance_cost = 800)
  (h2 : c.worker_wage = 20)
  (h3 : c.production_rate = 6)
  (h4 : c.gadget_price = 9/2)
  (h5 : c.workday_length = 9) :
  min_workers_for_profit c = 13 := by
  sorry

#eval min_workers_for_profit { 
  maintenance_cost := 800, 
  worker_wage := 20, 
  production_rate := 6, 
  gadget_price := 9/2, 
  workday_length := 9 
}

end min_workers_theorem_l2565_256506


namespace sufficient_not_necessary_l2565_256584

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b, b > a ∧ a > 0 → a * (b + 1) > a^2) ∧
  (∃ a b, a * (b + 1) > a^2 ∧ ¬(b > a ∧ a > 0)) :=
sorry

end sufficient_not_necessary_l2565_256584


namespace ages_when_bella_turns_18_l2565_256536

/-- Given the initial ages and birth years, prove the ages when Bella turns 18 -/
theorem ages_when_bella_turns_18 
  (marianne_age_2000 : ℕ)
  (bella_age_2000 : ℕ)
  (carmen_age_2000 : ℕ)
  (elli_birth_year : ℕ)
  (h1 : marianne_age_2000 = 20)
  (h2 : bella_age_2000 = 8)
  (h3 : carmen_age_2000 = 15)
  (h4 : elli_birth_year = 2003) :
  let year_bella_18 := 2000 + (18 - bella_age_2000)
  (year_bella_18 - 2000 + marianne_age_2000 = 30) ∧ 
  (year_bella_18 - 2000 + carmen_age_2000 = 33) ∧
  (year_bella_18 - elli_birth_year = 15) :=
sorry

end ages_when_bella_turns_18_l2565_256536


namespace x_intercept_of_perpendicular_line_l2565_256583

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℚ
  y_intercept : ℚ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℚ :=
  -l.y_intercept / l.slope

/-- The original line 4x + 5y = 10 -/
def original_line : Line :=
  { slope := -4/5, y_intercept := 2 }

/-- The perpendicular line we're interested in -/
def perpendicular_line : Line :=
  { slope := 5/4, y_intercept := -3 }

theorem x_intercept_of_perpendicular_line :
  perpendicular original_line perpendicular_line ∧
  perpendicular_line.y_intercept = -3 →
  x_intercept perpendicular_line = 12/5 := by
  sorry

end x_intercept_of_perpendicular_line_l2565_256583


namespace fraction_value_l2565_256556

theorem fraction_value (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) := by
sorry

end fraction_value_l2565_256556


namespace max_moves_less_than_500000_l2565_256513

/-- Represents the maximum number of moves for a given number of cards. -/
def max_moves (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the maximum number of moves for 1000 cards is less than 500,000. -/
theorem max_moves_less_than_500000 :
  max_moves 1000 < 500000 := by
  sorry

#eval max_moves 1000  -- This will evaluate to 499500

end max_moves_less_than_500000_l2565_256513


namespace parabola_line_intersection_l2565_256528

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line passing through D(4,0)
def line_through_D (x y : ℝ) : Prop := ∃ t : ℝ, x = t*y + 4

-- Define points A and B as intersections of the line and parabola
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  line_through_D A.1 A.2 ∧ line_through_D B.1 B.2 ∧
  A ≠ B

-- State the theorem
theorem parabola_line_intersection 
  (A B : ℝ × ℝ) (h : intersection_points A B) :
  (A.1 * B.1 + A.2 * B.2 = 0) ∧  -- OA ⊥ OB
  (∀ S : ℝ, S = (1/2) * abs (A.1 * B.2 - A.2 * B.1) → S ≥ 16) :=
by sorry

end parabola_line_intersection_l2565_256528


namespace problem_solution_l2565_256553

theorem problem_solution (x y z : ℝ) 
  (hx : x ≠ 0)
  (hz : z ≠ 0)
  (eq1 : x/2 = y^2 + z)
  (eq2 : x/4 = 4*y + 2*z) :
  x = 120 := by
sorry

end problem_solution_l2565_256553


namespace gcd_b_n_b_n_plus_one_is_one_l2565_256517

def b (n : ℕ) : ℚ := (15^n - 1) / 14

theorem gcd_b_n_b_n_plus_one_is_one (n : ℕ) : 
  Nat.gcd (Nat.floor (b n)) (Nat.floor (b (n + 1))) = 1 := by
  sorry

end gcd_b_n_b_n_plus_one_is_one_l2565_256517


namespace correct_factorization_l2565_256522

theorem correct_factorization (a : ℝ) : a^2 - 2*a + 1 = (a - 1)^2 := by
  sorry

end correct_factorization_l2565_256522


namespace vector_magnitude_l2565_256502

theorem vector_magnitude (a b : ℝ × ℝ) : 
  a = (Real.cos (5 * π / 180), Real.sin (5 * π / 180)) →
  b = (Real.cos (65 * π / 180), Real.sin (65 * π / 180)) →
  Real.sqrt ((a.1 + 2 * b.1)^2 + (a.2 + 2 * b.2)^2) = Real.sqrt 7 :=
by sorry

end vector_magnitude_l2565_256502


namespace inscribed_circle_rectangle_area_l2565_256568

/-- Proves that the area of a rectangle with an inscribed circle of radius 7 and a length-to-width ratio of 3:1 is equal to 588 -/
theorem inscribed_circle_rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 7 → ratio = 3 → 2 * r * ratio * 2 * r = 588 := by sorry

end inscribed_circle_rectangle_area_l2565_256568


namespace triangle_third_angle_l2565_256565

theorem triangle_third_angle (a b c : ℝ) (ha : a = 40) (hb : b = 60) 
  (sum : a + b + c = 180) : c = 80 := by
  sorry

end triangle_third_angle_l2565_256565


namespace statue_original_cost_l2565_256552

/-- Proves that if a statue is sold for $540 with a 35% profit, then its original cost was $400. -/
theorem statue_original_cost (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) :
  selling_price = 540 →
  profit_percentage = 0.35 →
  selling_price = original_cost * (1 + profit_percentage) →
  original_cost = 400 := by
sorry

end statue_original_cost_l2565_256552


namespace simplify_expression_l2565_256508

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 9) - (x + 6)*(3*x - 2) = 7*x - 24 := by
  sorry

end simplify_expression_l2565_256508


namespace composition_equation_solution_l2565_256500

theorem composition_equation_solution :
  let δ : ℝ → ℝ := λ x ↦ 4 * x + 9
  let φ : ℝ → ℝ := λ x ↦ 9 * x + 6
  ∃ x : ℝ, δ (φ x) = 3 ∧ x = -5/6 := by
sorry

end composition_equation_solution_l2565_256500


namespace expression_value_l2565_256515

theorem expression_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 - y^2 = x + y) :
  x / y + y / x = 2 + 1 / (y^2 + y) := by
  sorry

end expression_value_l2565_256515


namespace president_and_committee_selection_l2565_256570

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem president_and_committee_selection :
  let total_people : ℕ := 10
  let committee_size : ℕ := 3
  let president_choices : ℕ := total_people
  let committee_choices : ℕ := choose (total_people - 1) committee_size
  president_choices * committee_choices = 840 :=
by sorry

end president_and_committee_selection_l2565_256570


namespace fourth_root_16_times_sixth_root_9_l2565_256539

theorem fourth_root_16_times_sixth_root_9 : 
  (16 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/6) = 2 * (3 : ℝ) ^ (1/3) := by
  sorry

end fourth_root_16_times_sixth_root_9_l2565_256539


namespace bouquet_carnations_fraction_l2565_256582

theorem bouquet_carnations_fraction (total_flowers : ℕ) 
  (blue_flowers red_flowers blue_roses red_roses blue_carnations red_carnations : ℕ) :
  (blue_flowers = red_flowers) →  -- Half of the flowers are blue
  (blue_flowers + red_flowers = total_flowers) →
  (blue_roses = 2 * blue_flowers / 5) →  -- Two-fifths of blue flowers are roses
  (red_carnations = 2 * red_flowers / 3) →  -- Two-thirds of red flowers are carnations
  (blue_carnations = blue_flowers - blue_roses) →
  (red_roses = red_flowers - red_carnations) →
  ((blue_carnations + red_carnations : ℚ) / total_flowers = 19 / 30) := by
sorry

end bouquet_carnations_fraction_l2565_256582


namespace unique_function_solution_l2565_256540

-- Define the property that f should satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y - 1) + f x * f y = 2 * x * y - 1

-- State the theorem
theorem unique_function_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f → (∀ x : ℝ, f x = x) :=
by
  sorry

end unique_function_solution_l2565_256540


namespace quadratic_inequality_range_l2565_256596

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → x^2 + a*x - 3*a < 0) → a > (1/2 : ℝ) := by
  sorry

end quadratic_inequality_range_l2565_256596


namespace theater_ticket_difference_l2565_256597

theorem theater_ticket_difference :
  ∀ (x y : ℕ),
    x + y = 350 →
    12 * x + 8 * y = 3320 →
    y - x = 90 :=
by
  sorry

end theater_ticket_difference_l2565_256597


namespace cube_of_neg_cube_l2565_256518

theorem cube_of_neg_cube (x : ℝ) : (-x^3)^3 = -x^9 := by
  sorry

end cube_of_neg_cube_l2565_256518


namespace inequality_holds_l2565_256535

theorem inequality_holds (a b c d : ℝ) : (a - b) * (b - c) * (c - d) * (d - a) + (a - c)^2 * (b - d)^2 ≥ 0 := by
  sorry

end inequality_holds_l2565_256535


namespace firm_partners_count_l2565_256521

theorem firm_partners_count (partners associates : ℕ) : 
  partners / associates = 2 / 63 →
  partners / (associates + 45) = 1 / 34 →
  partners = 18 := by
sorry

end firm_partners_count_l2565_256521


namespace valid_pairs_count_l2565_256585

/-- A function that counts the number of valid (a,b) pairs -/
def count_valid_pairs : ℕ :=
  (Finset.range 50).sum (fun a => 
    Nat.ceil (((a + 1) : ℕ) / 2))

/-- The main theorem stating that there are exactly 75 valid pairs -/
theorem valid_pairs_count : count_valid_pairs = 75 := by
  sorry

end valid_pairs_count_l2565_256585


namespace three_circles_cover_horizon_two_circles_cannot_cover_horizon_l2565_256586

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_positive : radius > 0

-- Define a point in 2D space
def Point : Type := ℝ × ℝ

-- Define a ray emanating from a point
structure Ray where
  origin : Point
  direction : ℝ × ℝ
  direction_nonzero : direction ≠ (0, 0)

-- Function to check if two circles are non-overlapping and non-touching
def non_overlapping_non_touching (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let distance := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance > c1.radius + c2.radius

-- Function to check if a point is outside a circle
def point_outside_circle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 > c.radius^2

-- Function to check if a ray intersects a circle
def ray_intersects_circle (r : Ray) (c : Circle) : Prop :=
  sorry  -- The actual implementation would go here

-- Theorem for three circles covering the horizon
theorem three_circles_cover_horizon :
  ∃ (c1 c2 c3 : Circle) (p : Point),
    non_overlapping_non_touching c1 c2 ∧
    non_overlapping_non_touching c1 c3 ∧
    non_overlapping_non_touching c2 c3 ∧
    point_outside_circle p c1 ∧
    point_outside_circle p c2 ∧
    point_outside_circle p c3 ∧
    ∀ (r : Ray), r.origin = p →
      ray_intersects_circle r c1 ∨
      ray_intersects_circle r c2 ∨
      ray_intersects_circle r c3 :=
  sorry

-- Theorem for two circles not covering the horizon
theorem two_circles_cannot_cover_horizon :
  ¬ ∃ (c1 c2 : Circle) (p : Point),
    non_overlapping_non_touching c1 c2 ∧
    point_outside_circle p c1 ∧
    point_outside_circle p c2 ∧
    ∀ (r : Ray), r.origin = p →
      ray_intersects_circle r c1 ∨
      ray_intersects_circle r c2 :=
  sorry

end three_circles_cover_horizon_two_circles_cannot_cover_horizon_l2565_256586


namespace die_roll_probability_l2565_256587

/-- The number of sides on the die -/
def n : ℕ := 8

/-- The number of rolls -/
def r : ℕ := 12

/-- The probability of rolling a different number from the previous roll -/
def p : ℚ := (n - 1) / n

/-- The probability of rolling the same number as the previous roll -/
def q : ℚ := 1 / n

theorem die_roll_probability : 
  p^(r - 2) * q = 282475249 / 8589934592 := by sorry

end die_roll_probability_l2565_256587


namespace cubic_roots_sum_l2565_256578

theorem cubic_roots_sum (a b c : ℂ) (r s t : ℝ) : 
  (∀ x, x^3 - 3*x^2 + 5*x + 7 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  (∀ x, x^3 + r*x^2 + s*x + t = 0 ↔ (x = a + b ∨ x = b + c ∨ x = c + a)) →
  t = 8 := by
sorry

end cubic_roots_sum_l2565_256578


namespace point_c_coordinates_l2565_256531

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- The distance between two points -/
def distance (p q : Point) : ℚ :=
  ((p.x - q.x)^2 + (p.y - q.y)^2).sqrt

/-- Check if a point is on a line segment -/
def isOnSegment (p q r : Point) : Prop :=
  distance p r + distance r q = distance p q

theorem point_c_coordinates :
  let a : Point := ⟨-2, 1⟩
  let b : Point := ⟨4, 9⟩
  let c : Point := ⟨22/7, 55/7⟩
  isOnSegment a c b ∧ distance a c = 4 * distance c b → c = ⟨22/7, 55/7⟩ := by
  sorry

end point_c_coordinates_l2565_256531


namespace min_value_of_fraction_sum_l2565_256516

theorem min_value_of_fraction_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 1) :
  (2/x) + (1/y) ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 1 ∧ (2/x₀) + (1/y₀) = 8 :=
by sorry

end min_value_of_fraction_sum_l2565_256516


namespace complex_fraction_eval_l2565_256512

theorem complex_fraction_eval (c d : ℂ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : c^2 + c*d + d^2 = 0) : 
  (c^12 + d^12) / (c^3 + d^3)^4 = 1/8 := by
  sorry

end complex_fraction_eval_l2565_256512


namespace quadratic_inequality_solution_set_l2565_256577

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + 3*x + 10 > 0} = Set.Ioo (-2) 5 := by sorry

end quadratic_inequality_solution_set_l2565_256577


namespace largest_angle_in_consecutive_angle_hexagon_l2565_256524

/-- The largest angle in a convex hexagon with six consecutive integer angles -/
def largest_hexagon_angle : ℝ := 122.5

/-- A convex hexagon with six consecutive integer angles -/
structure ConsecutiveAngleHexagon where
  angles : Fin 6 → ℤ
  is_consecutive : ∀ i : Fin 5, angles i.succ = angles i + 1
  is_convex : ∀ i : Fin 6, 0 < angles i ∧ angles i < 180

theorem largest_angle_in_consecutive_angle_hexagon (h : ConsecutiveAngleHexagon) :
  (h.angles 5 : ℝ) = largest_hexagon_angle :=
sorry

end largest_angle_in_consecutive_angle_hexagon_l2565_256524


namespace no_repeating_stock_price_l2565_256594

theorem no_repeating_stock_price (n : ℕ) : ¬ ∃ (k l : ℕ), k + l > 0 ∧ k + l ≤ 365 ∧ (1 + n / 100 : ℚ)^k * (1 - n / 100 : ℚ)^l = 1 := by
  sorry

end no_repeating_stock_price_l2565_256594


namespace tetrahedron_vertices_tetrahedron_has_four_vertices_l2565_256590

/-- A tetrahedron is a three-dimensional geometric shape with four triangular faces. -/
structure Tetrahedron where
  -- No specific fields needed for this problem

/-- The number of vertices in a tetrahedron is 4. -/
theorem tetrahedron_vertices (t : Tetrahedron) : Nat := 4

/-- Proof that a tetrahedron has 4 vertices. -/
theorem tetrahedron_has_four_vertices (t : Tetrahedron) : tetrahedron_vertices t = 4 := by
  sorry

end tetrahedron_vertices_tetrahedron_has_four_vertices_l2565_256590


namespace trig_roots_theorem_l2565_256562

theorem trig_roots_theorem (θ : ℝ) (m : ℝ) 
  (h1 : (Real.sin θ)^2 - (Real.sqrt 3 - 1) * (Real.sin θ) + m = 0)
  (h2 : (Real.cos θ)^2 - (Real.sqrt 3 - 1) * (Real.cos θ) + m = 0) :
  (m = (3 - 2 * Real.sqrt 3) / 2) ∧
  ((Real.cos θ - Real.sin θ * Real.tan θ) / (1 - Real.tan θ) = Real.sqrt 3 - 1) := by
  sorry

end trig_roots_theorem_l2565_256562


namespace quadratic_function_m_range_l2565_256538

/-- A quadratic function of the form y = (m-2)x^2 + 2x - 3 -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + 2 * x - 3

/-- The range of m for which the function is quadratic -/
theorem quadratic_function_m_range (m : ℝ) :
  (∃ (a : ℝ), a ≠ 0 ∧ ∀ (x : ℝ), quadratic_function m x = a * x^2 + 2 * x - 3) ↔ m ≠ 2 :=
by sorry

end quadratic_function_m_range_l2565_256538


namespace water_requirement_proof_l2565_256510

/-- The number of households in the village -/
def num_households : ℕ := 10

/-- The total amount of water available in litres -/
def total_water : ℕ := 2000

/-- The number of months the water lasts -/
def num_months : ℕ := 10

/-- The number of litres of water required per household per month -/
def water_per_household_per_month : ℚ :=
  total_water / (num_households * num_months)

theorem water_requirement_proof :
  water_per_household_per_month = 20 := by
  sorry

end water_requirement_proof_l2565_256510


namespace green_bows_count_l2565_256551

theorem green_bows_count (total : ℕ) : 
  (3 : ℚ) / 20 * total + 3 / 10 * total + 1 / 5 * total + 1 / 20 * total + 24 = total →
  1 / 5 * total = 16 := by
  sorry

end green_bows_count_l2565_256551


namespace base_sum_theorem_l2565_256504

/-- Represents a repeating decimal in a given base -/
def repeating_decimal (numerator denominator base : ℕ) : ℚ :=
  (numerator : ℚ) / ((base ^ 2 - 1) : ℚ)

theorem base_sum_theorem :
  ∃! (B₁ B₂ : ℕ), 
    B₁ > 1 ∧ B₂ > 1 ∧
    repeating_decimal 45 99 B₁ = repeating_decimal 3 9 B₂ ∧
    repeating_decimal 54 99 B₁ = repeating_decimal 6 9 B₂ ∧
    B₁ + B₂ = 20 := by sorry

end base_sum_theorem_l2565_256504


namespace alex_bike_trip_l2565_256533

/-- Alex's bike trip problem -/
theorem alex_bike_trip (v : ℝ) 
  (h1 : 4.5 * v + 2.5 * 12 + 1.5 * 24 + 8 = 164) : v = 20 := by
  sorry

end alex_bike_trip_l2565_256533


namespace systematic_sampling_example_l2565_256579

def isValidSystematicSample (n : ℕ) (k : ℕ) (sample : List ℕ) : Prop :=
  let interval := n / k
  sample.length = k ∧
  ∀ i, i ∈ sample → i < n ∧
  ∀ i j, i < j → i ∈ sample → j ∈ sample → j - i = interval

theorem systematic_sampling_example :
  isValidSystematicSample 50 5 [1, 11, 21, 31, 41] := by
  sorry

end systematic_sampling_example_l2565_256579


namespace never_exceeds_100_l2565_256566

def repeated_square (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | m + 1 => (repeated_square m) ^ 2

theorem never_exceeds_100 (n : ℕ) : repeated_square n ≤ 100 := by
  sorry

#check never_exceeds_100

end never_exceeds_100_l2565_256566


namespace project_savings_percentage_l2565_256537

theorem project_savings_percentage 
  (actual_investment : ℕ) 
  (savings : ℕ) 
  (h1 : actual_investment = 150000)
  (h2 : savings = 50000) :
  (savings : ℝ) / ((actual_investment : ℝ) + (savings : ℝ)) * 100 = 25 := by
sorry

end project_savings_percentage_l2565_256537


namespace solution_difference_l2565_256529

theorem solution_difference (a b : ℝ) : 
  ((a - 4) * (a + 4) = 28 * a - 112) → 
  ((b - 4) * (b + 4) = 28 * b - 112) → 
  a ≠ b →
  a > b →
  a - b = 20 := by
sorry

end solution_difference_l2565_256529


namespace car_speed_problem_l2565_256541

/-- Proves that if a car traveling at speed v km/h takes 15 seconds longer to travel 1 km
    than it would at 48 km/h, then v = 40 km/h. -/
theorem car_speed_problem (v : ℝ) :
  (v > 0) →  -- Ensure speed is positive
  (3600 / v = 3600 / 48 + 15) →  -- Time difference equation
  v = 40 := by
  sorry

end car_speed_problem_l2565_256541


namespace sqrt_one_hundredth_l2565_256567

theorem sqrt_one_hundredth : Real.sqrt (1 / 100) = 1 / 10 := by
  sorry

end sqrt_one_hundredth_l2565_256567


namespace polynomial_roots_problem_l2565_256574

theorem polynomial_roots_problem (r s t : ℝ) 
  (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (h1 : ∀ x : ℝ, x^2 + r*x + s = 0 ↔ x = s ∨ x = t)
  (h2 : (5 : ℝ)^2 + t*5 + r = 0) : 
  s = 29 := by sorry

end polynomial_roots_problem_l2565_256574


namespace sock_cost_is_three_l2565_256514

/-- The cost of a uniform given the cost of socks -/
def uniform_cost (sock_cost : ℚ) : ℚ :=
  20 + 2 * 20 + (2 * 20) / 5 + sock_cost

/-- The total cost of 5 uniforms given the cost of socks -/
def total_cost (sock_cost : ℚ) : ℚ :=
  5 * uniform_cost sock_cost

theorem sock_cost_is_three :
  ∃ (sock_cost : ℚ), total_cost sock_cost = 355 ∧ sock_cost = 3 := by
  sorry

end sock_cost_is_three_l2565_256514


namespace factorization_of_4x_squared_minus_4_l2565_256589

theorem factorization_of_4x_squared_minus_4 (x : ℝ) : 4 * x^2 - 4 = 4 * (x + 1) * (x - 1) := by
  sorry

end factorization_of_4x_squared_minus_4_l2565_256589


namespace squared_difference_of_quadratic_roots_l2565_256571

theorem squared_difference_of_quadratic_roots :
  ∀ d e : ℝ, (3 * d^2 + 10 * d - 25 = 0) → (3 * e^2 + 10 * e - 25 = 0) →
  (d - e)^2 = 400 / 9 := by
  sorry

end squared_difference_of_quadratic_roots_l2565_256571


namespace twenty_one_billion_scientific_notation_l2565_256581

/-- The scientific notation representation of 21 billion -/
def twenty_one_billion_scientific : ℝ := 2.1 * (10 ^ 9)

/-- The value of 21 billion -/
def twenty_one_billion : ℝ := 21 * (10 ^ 9)

theorem twenty_one_billion_scientific_notation :
  twenty_one_billion = twenty_one_billion_scientific :=
by sorry

end twenty_one_billion_scientific_notation_l2565_256581


namespace card_probability_l2565_256580

def cards : Finset ℕ := Finset.range 11

def group_A : Finset ℕ := cards.filter (λ x => x % 2 = 1)
def group_B : Finset ℕ := cards.filter (λ x => x % 2 = 0)

def average (a b c : ℕ) : ℚ := (a + b + c : ℚ) / 3

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (group_A.product group_B).filter (λ (a, b) => a + b < 6)

theorem card_probability :
  (favorable_outcomes.card : ℚ) / (group_A.card * group_B.card) = 1 / 10 :=
by sorry

end card_probability_l2565_256580


namespace certain_number_proof_l2565_256554

theorem certain_number_proof : ∃ x : ℝ, (3889 + 12.952 - x = 3854.002) ∧ (x = 47.95) := by sorry

end certain_number_proof_l2565_256554


namespace max_triangles_correct_l2565_256560

/-- The number of points on the hypotenuse of the right triangle -/
def num_points : ℕ := 8

/-- The maximum number of triangles that can be formed -/
def max_triangles : ℕ := 28

/-- Theorem stating that the maximum number of triangles is correct -/
theorem max_triangles_correct :
  (num_points.choose 2) = max_triangles := by sorry

end max_triangles_correct_l2565_256560


namespace line_touches_x_axis_twice_l2565_256593

-- Define the function representing the line
def f (x : ℝ) : ℝ := x^2 - x^3

-- Theorem statement
theorem line_touches_x_axis_twice :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ (∀ x, f x = 0 → x = a ∨ x = b) :=
sorry

end line_touches_x_axis_twice_l2565_256593


namespace solution_mixture_problem_l2565_256595

/-- Solution X and Y mixture problem -/
theorem solution_mixture_problem 
  (total : ℝ) (total_pos : 0 < total)
  (x : ℝ) (x_nonneg : 0 ≤ x) (x_le_total : x ≤ total)
  (ha : x * 0.2 + (total - x) * 0.3 = total * 0.22) :
  x / total = 0.8 := by
sorry

end solution_mixture_problem_l2565_256595


namespace stratified_sampling_most_appropriate_l2565_256549

/-- Represents a breeding room with a certain number of mice -/
structure BreedingRoom where
  mice : ℕ

/-- Represents a research institute with multiple breeding rooms -/
structure ResearchInstitute where
  rooms : List BreedingRoom

/-- Different sampling methods -/
inductive SamplingMethod
  | Simple
  | Stratified
  | Cluster
  | Systematic

/-- Determines if a population has significant subgroup differences -/
def hasSignificantSubgroupDifferences (institute : ResearchInstitute) : Prop :=
  ∃ (r1 r2 : BreedingRoom), r1 ∈ institute.rooms ∧ r2 ∈ institute.rooms ∧ r1.mice ≠ r2.mice

/-- Determines the most appropriate sampling method given a research institute and sample size -/
def mostAppropriateSamplingMethod (institute : ResearchInstitute) (sampleSize : ℕ) : SamplingMethod :=
  sorry

/-- Theorem stating that stratified sampling is most appropriate for the given conditions -/
theorem stratified_sampling_most_appropriate
  (institute : ResearchInstitute)
  (sampleSize : ℕ)
  (h1 : institute.rooms.length = 4)
  (h2 : institute.rooms = [⟨18⟩, ⟨24⟩, ⟨54⟩, ⟨48⟩])
  (h3 : sampleSize = 24)
  (h4 : hasSignificantSubgroupDifferences institute) :
  mostAppropriateSamplingMethod institute sampleSize = SamplingMethod.Stratified :=
  sorry

end stratified_sampling_most_appropriate_l2565_256549


namespace u_converges_immediately_l2565_256592

def u : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 3 * u n - 3 * (u n)^2

theorem u_converges_immediately :
  ∀ k : ℕ, |u k - 1/2| ≤ 1/2^20 := by
  sorry

end u_converges_immediately_l2565_256592


namespace initial_meals_proof_l2565_256542

/-- The number of meals initially available for adults -/
def initial_meals : ℕ := 70

/-- The number of children that can be fed with one adult meal -/
def children_per_adult_meal : ℚ := 90 / initial_meals

theorem initial_meals_proof :
  (initial_meals - 21) * children_per_adult_meal = 63 :=
by sorry

end initial_meals_proof_l2565_256542


namespace quadratic_inequality_range_l2565_256501

/-- The range of m for which the quadratic inequality (m-3)x^2 - 2mx - 8 > 0
    has a solution set that is an open interval with length between 1 and 2 -/
theorem quadratic_inequality_range (m : ℝ) : 
  (∃ a b : ℝ, 
    (∀ x : ℝ, (m - 3) * x^2 - 2 * m * x - 8 > 0 ↔ a < x ∧ x < b) ∧ 
    1 ≤ b - a ∧ b - a ≤ 2) ↔ 
  m ≤ -15 ∨ (7/3 ≤ m ∧ m ≤ 33/14) :=
sorry

end quadratic_inequality_range_l2565_256501


namespace quadratic_factorization_l2565_256534

theorem quadratic_factorization (C D : ℤ) :
  (∀ y, 20 * y^2 - 117 * y + 72 = (C * y - 8) * (D * y - 9)) →
  C * D + C = 25 := by
  sorry

end quadratic_factorization_l2565_256534


namespace wrong_mark_value_l2565_256548

/-- Proves that the wrongly entered mark is 85 given the conditions of the problem -/
theorem wrong_mark_value (n : ℕ) (correct_mark : ℕ) (average_increase : ℚ) 
  (h1 : n = 80)
  (h2 : correct_mark = 45)
  (h3 : average_increase = 1/2) : 
  ∃ (wrong_mark : ℕ), wrong_mark = 85 ∧ 
    (wrong_mark - correct_mark : ℚ) = n * average_increase := by
  sorry

end wrong_mark_value_l2565_256548


namespace train_length_calculation_l2565_256569

/-- Calculates the length of a train given its speed and time to cross a point. -/
def trainLength (speed : Real) (time : Real) : Real :=
  speed * time

theorem train_length_calculation (speed : Real) (time : Real) 
  (h1 : speed = 18 * (1000 / 3600)) -- 18 km/h converted to m/s
  (h2 : time = 200) : 
  trainLength speed time = 1000 := by
  sorry

#check train_length_calculation

end train_length_calculation_l2565_256569


namespace polynomial_value_at_five_l2565_256557

/-- Given a polynomial g(x) = ax^7 + bx^6 + cx - 3 where g(-5) = -3, prove that g(5) = 31250b - 3 -/
theorem polynomial_value_at_five (a b c : ℝ) :
  let g : ℝ → ℝ := λ x => a * x^7 + b * x^6 + c * x - 3
  g (-5) = -3 →
  g 5 = 31250 * b - 3 := by sorry

end polynomial_value_at_five_l2565_256557
