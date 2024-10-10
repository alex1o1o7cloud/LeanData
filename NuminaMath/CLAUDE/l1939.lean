import Mathlib

namespace fraction_simplification_l1939_193963

theorem fraction_simplification : (4 * 5) / 10 = 2 := by sorry

end fraction_simplification_l1939_193963


namespace fraction_sum_l1939_193976

theorem fraction_sum : (3 : ℚ) / 8 + (9 : ℚ) / 12 = (9 : ℚ) / 8 := by
  sorry

end fraction_sum_l1939_193976


namespace fence_poles_needed_l1939_193911

theorem fence_poles_needed (length width pole_distance : ℕ) : 
  length = 90 → width = 40 → pole_distance = 5 →
  (2 * (length + width)) / pole_distance = 52 := by
sorry

end fence_poles_needed_l1939_193911


namespace village_population_equality_l1939_193974

/-- Represents the population change in a village over time. -/
structure VillagePopulation where
  initial : ℕ  -- Initial population
  rate : ℤ     -- Annual rate of change (positive for increase, negative for decrease)

/-- Calculates the population after a given number of years. -/
def population_after (v : VillagePopulation) (years : ℕ) : ℤ :=
  v.initial + v.rate * years

theorem village_population_equality (village_x village_y : VillagePopulation) 
  (h1 : village_x.initial = 78000)
  (h2 : village_x.rate = -1200)
  (h3 : village_y.initial = 42000)
  (h4 : population_after village_x 18 = population_after village_y 18) :
  village_y.rate = 800 := by
  sorry

#check village_population_equality

end village_population_equality_l1939_193974


namespace parallelogram_vertices_l1939_193984

/-- A parallelogram with two known vertices and one side parallel to x-axis -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  parallel_to_x_axis : Bool

/-- The other pair of opposite vertices of the parallelogram -/
def other_vertices (p : Parallelogram) : (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

theorem parallelogram_vertices (p : Parallelogram) 
  (h1 : p.v1 = (2, -3)) 
  (h2 : p.v2 = (8, 9)) 
  (h3 : p.parallel_to_x_axis = true) : 
  other_vertices p = ((5, -3), (5, 9)) := by
  sorry

end parallelogram_vertices_l1939_193984


namespace polynomial_equality_main_result_l1939_193958

def f (a₁ a₂ a₃ a₄ : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let b₁ := 0
  let b₂ := -3
  let b₃ := 4
  let b₄ := -1
  (b₁, b₂, b₃, b₄)

theorem polynomial_equality (x : ℝ) (a₁ a₂ a₃ a₄ : ℝ) (b₁ b₂ b₃ b₄ : ℝ) :
  x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄ = (x+1)^4 + b₁*(x+1)^3 + b₂*(x+1)^2 + b₃*(x+1) + b₄ →
  f a₁ a₂ a₃ a₄ = (b₁, b₂, b₃, b₄) :=
by sorry

theorem main_result : f 4 3 2 1 = (0, -3, 4, -1) :=
by sorry

end polynomial_equality_main_result_l1939_193958


namespace waiter_customers_l1939_193910

theorem waiter_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) 
  (h1 : num_tables = 8)
  (h2 : women_per_table = 7)
  (h3 : men_per_table = 4) :
  num_tables * (women_per_table + men_per_table) = 88 := by
sorry

end waiter_customers_l1939_193910


namespace max_figures_in_cube_l1939_193937

/-- The volume of a rectangular cuboid -/
def volume (length width height : ℕ) : ℕ := length * width * height

/-- The dimensions of the cube -/
def cube_dim : ℕ := 3

/-- The dimensions of the figure -/
def figure_dim : Vector ℕ 3 := ⟨[2, 2, 1], by simp⟩

/-- The maximum number of figures that can fit in the cube -/
def max_figures : ℕ := 6

theorem max_figures_in_cube :
  (volume cube_dim cube_dim cube_dim) ≥ max_figures * (volume figure_dim[0] figure_dim[1] figure_dim[2]) ∧
  ∀ n : ℕ, n > max_figures → (volume cube_dim cube_dim cube_dim) < n * (volume figure_dim[0] figure_dim[1] figure_dim[2]) :=
by sorry

end max_figures_in_cube_l1939_193937


namespace election_winner_votes_l1939_193913

theorem election_winner_votes 
  (total_votes : ℕ)
  (winner_percentage : ℚ)
  (vote_difference : ℕ)
  (h1 : winner_percentage = 62 / 100)
  (h2 : vote_difference = 312)
  (h3 : ↑total_votes * winner_percentage - ↑total_votes * (1 - winner_percentage) = vote_difference) :
  ↑total_votes * winner_percentage = 806 :=
by sorry

end election_winner_votes_l1939_193913


namespace rectangle_diagonal_intersections_l1939_193968

theorem rectangle_diagonal_intersections (ℓ b : ℕ) (hℓ : ℓ > 0) (hb : b > 0) : 
  let V := ℓ + b - Nat.gcd ℓ b
  ℓ = 6 → b = 4 → V = 8 := by
  sorry

end rectangle_diagonal_intersections_l1939_193968


namespace quadratic_and_inequality_system_l1939_193947

theorem quadratic_and_inequality_system :
  -- Part 1: Quadratic equation
  (∀ x : ℝ, x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) ∧
  -- Part 2: Inequality system
  (∀ x : ℝ, x - 2*(x-1) ≤ 1 ∧ (1+x)/3 > x-1 ↔ -1 ≤ x ∧ x < 2) :=
by sorry

end quadratic_and_inequality_system_l1939_193947


namespace total_price_two_corgis_is_2507_l1939_193943

/-- Calculates the total price for two Corgi dogs with given conditions -/
def total_price_two_corgis (cost : ℝ) (profit_percent : ℝ) (discount_percent : ℝ) (tax_percent : ℝ) (shipping_fee : ℝ) : ℝ :=
  let selling_price := cost * (1 + profit_percent)
  let total_before_discount := 2 * selling_price
  let discounted_price := total_before_discount * (1 - discount_percent)
  let price_with_tax := discounted_price * (1 + tax_percent)
  price_with_tax + shipping_fee

/-- Theorem stating the total price for two Corgi dogs is $2507 -/
theorem total_price_two_corgis_is_2507 :
  total_price_two_corgis 1000 0.30 0.10 0.05 50 = 2507 := by
  sorry

end total_price_two_corgis_is_2507_l1939_193943


namespace hyperbola_equation_l1939_193920

/-- Given a hyperbola with foci F₁(-√5,0) and F₂(√5,0), and a point P on the hyperbola
    such that PF₁ · PF₂ = 0 and |PF₁| · |PF₂| = 2, the standard equation of the hyperbola
    is x²/4 - y² = 1. -/
theorem hyperbola_equation (F₁ F₂ P : ℝ × ℝ) : 
  F₁ = (-Real.sqrt 5, 0) →
  F₂ = (Real.sqrt 5, 0) →
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0 →
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * 
    Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 2 →
  ∃ (x y : ℝ), x^2 / 4 - y^2 = 1 ∧ 
    (x, y) = P :=
by sorry

end hyperbola_equation_l1939_193920


namespace larger_number_proof_l1939_193912

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1365) 
  (h2 : L = 4 * S + 15) : 
  L = 1815 := by
  sorry

end larger_number_proof_l1939_193912


namespace base4_even_digits_145_l1939_193944

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

theorem base4_even_digits_145 :
  countEvenDigits (toBase4 145) = 2 := by
  sorry

end base4_even_digits_145_l1939_193944


namespace scientists_from_usa_l1939_193970

theorem scientists_from_usa (total : ℕ) (europe : ℕ) (canada : ℕ) (usa : ℕ)
  (h_total : total = 70)
  (h_europe : europe = total / 2)
  (h_canada : canada = total / 5)
  (h_sum : total = europe + canada + usa) :
  usa = 21 := by
  sorry

end scientists_from_usa_l1939_193970


namespace factorial_6_equals_720_l1939_193929

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_6_equals_720 : factorial 6 = 720 := by
  sorry

end factorial_6_equals_720_l1939_193929


namespace perpendicular_parallel_lines_to_plane_l1939_193942

/-- Two lines are parallel -/
def parallel_lines (a b : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

/-- The theorem to be proved -/
theorem perpendicular_parallel_lines_to_plane 
  (a b : Line) (α : Plane) 
  (h1 : a ≠ b) 
  (h2 : parallel_lines a b) 
  (h3 : perpendicular_line_plane a α) : 
  perpendicular_line_plane b α := by sorry

end perpendicular_parallel_lines_to_plane_l1939_193942


namespace functional_inequality_l1939_193924

theorem functional_inequality (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) :
  let f : ℝ → ℝ := λ y => y^2 - y + 1
  2 * f x + x^2 * f (1/x) ≥ (3*x^3 - x^2 + 4*x + 3) / (x + 1) :=
by sorry

end functional_inequality_l1939_193924


namespace piggy_bank_compartments_l1939_193973

/-- Given a piggy bank with an unknown number of compartments, prove that the number of compartments is 12 based on the given conditions. -/
theorem piggy_bank_compartments :
  ∀ (c : ℕ), -- c represents the number of compartments
  (∀ (i : ℕ), i < c → 2 = 2) → -- Each compartment initially has 2 pennies (this is a trivial condition in Lean)
  (∀ (i : ℕ), i < c → 6 = 6) → -- 6 pennies are added to each compartment (also trivial in Lean)
  (c * (2 + 6) = 96) →         -- Total pennies after adding is 96
  c = 12 := by
sorry


end piggy_bank_compartments_l1939_193973


namespace binomial_expression_approx_l1939_193946

/-- Calculates the binomial coefficient for real x and nonnegative integer k -/
def binomial (x : ℝ) (k : ℕ) : ℝ := sorry

/-- The main theorem stating that the given expression is approximately equal to -1.243 -/
theorem binomial_expression_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |((binomial (3/2 : ℝ) 10) * 3^10) / (binomial 20 10) + 1.243| < ε :=
sorry

end binomial_expression_approx_l1939_193946


namespace expand_product_l1939_193982

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12*x + 27 := by
  sorry

end expand_product_l1939_193982


namespace planet_colonization_combinations_l1939_193951

/-- Represents the number of habitable planets discovered -/
def total_planets : ℕ := 13

/-- Represents the number of Earth-like planets -/
def earth_like_planets : ℕ := 5

/-- Represents the number of Mars-like planets -/
def mars_like_planets : ℕ := total_planets - earth_like_planets

/-- Represents the units required to colonize an Earth-like planet -/
def earth_like_units : ℕ := 2

/-- Represents the units required to colonize a Mars-like planet -/
def mars_like_units : ℕ := 1

/-- Represents the total units available for colonization -/
def available_units : ℕ := 15

/-- Calculates the number of unique combinations of planets that can be occupied -/
def count_combinations : ℕ :=
  (Nat.choose earth_like_planets earth_like_planets * Nat.choose mars_like_planets 5) +
  (Nat.choose earth_like_planets 4 * Nat.choose mars_like_planets 7)

theorem planet_colonization_combinations :
  count_combinations = 96 :=
sorry

end planet_colonization_combinations_l1939_193951


namespace smallest_sum_of_roots_l1939_193992

theorem smallest_sum_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + a = 0) :
  a + b ≥ 6.5 := by
sorry

end smallest_sum_of_roots_l1939_193992


namespace correct_evaluation_l1939_193985

/-- Evaluates an expression according to right-to-left rules -/
noncomputable def evaluate (a b c d e : ℝ) : ℝ :=
  a * (b^c - (d + e))

/-- Theorem stating that the evaluation is correct -/
theorem correct_evaluation (a b c d e : ℝ) :
  evaluate a b c d e = a * (b^c - (d + e)) := by sorry

end correct_evaluation_l1939_193985


namespace train_length_calculation_l1939_193965

/-- The length of a train given its speed and time to pass a point -/
def train_length (speed : Real) (time : Real) : Real :=
  speed * time

theorem train_length_calculation (speed : Real) (time : Real) 
  (h1 : speed = 160 * 1000 / 3600) -- Speed in m/s
  (h2 : time = 2.699784017278618) : 
  ∃ (ε : Real), ε > 0 ∧ |train_length speed time - 120| < ε :=
sorry

end train_length_calculation_l1939_193965


namespace linear_function_composition_l1939_193903

theorem linear_function_composition (a b : ℝ) :
  (∀ x : ℝ, (3 * ((a * x + b) : ℝ) - 4 : ℝ) = 4 * x + 5) →
  a + b = 13/3 := by
sorry

end linear_function_composition_l1939_193903


namespace girls_at_game_l1939_193977

theorem girls_at_game (boys girls : ℕ) : 
  (boys : ℚ) / girls = 8 / 5 → 
  boys = girls + 18 → 
  girls = 30 := by
sorry

end girls_at_game_l1939_193977


namespace triangle_inequality_l1939_193918

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  let p := (a + b + c) / 2
  a * Real.sqrt ((p - b) * (p - c) / (b * c)) +
  b * Real.sqrt ((p - c) * (p - a) / (a * c)) +
  c * Real.sqrt ((p - a) * (p - b) / (a * b)) ≥ p := by
  sorry

end triangle_inequality_l1939_193918


namespace power_of_product_cubed_l1939_193950

theorem power_of_product_cubed (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by sorry

end power_of_product_cubed_l1939_193950


namespace exam_score_calculation_l1939_193949

/-- Given an examination with the following conditions:
  * Total number of questions is 120
  * Each correct answer scores 3 marks
  * Each wrong answer loses 1 mark
  * The total score is 180 marks
  This theorem proves that the number of correctly answered questions is 75. -/
theorem exam_score_calculation (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) 
  (h1 : total_questions = 120)
  (h2 : correct_score = 3)
  (h3 : wrong_score = -1)
  (h4 : total_score = 180) :
  ∃ (correct_answers : ℕ), 
    correct_answers * correct_score + (total_questions - correct_answers) * wrong_score = total_score ∧ 
    correct_answers = 75 := by
  sorry

end exam_score_calculation_l1939_193949


namespace sin_max_at_neg_pi_fourth_l1939_193994

/-- The smallest positive constant c such that y = 3 sin(2x + c) reaches a maximum at x = -π/4 is π -/
theorem sin_max_at_neg_pi_fourth (c : ℝ) :
  c > 0 ∧ 
  (∀ x : ℝ, 3 * Real.sin (2 * x + c) ≤ 3 * Real.sin (2 * (-π/4) + c)) →
  c = π :=
sorry

end sin_max_at_neg_pi_fourth_l1939_193994


namespace log_equation_solution_l1939_193915

theorem log_equation_solution :
  ∃! x : ℝ, x > 0 ∧ Real.log x - Real.log 6 = 2 :=
by
  use 3/2
  sorry

end log_equation_solution_l1939_193915


namespace side_c_length_l1939_193956

/-- Given a triangle ABC with side lengths a, b, and c, and angle C opposite side c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  C : ℝ

/-- The Law of Cosines for a triangle -/
def lawOfCosines (t : Triangle) : Prop :=
  t.c^2 = t.a^2 + t.b^2 - 2 * t.a * t.b * Real.cos t.C

theorem side_c_length (t : Triangle) 
  (ha : t.a = 2) 
  (hb : t.b = 1) 
  (hC : t.C = π / 3) -- 60° in radians
  (hlawCosines : lawOfCosines t) :
  t.c = Real.sqrt 3 := by
  sorry

end side_c_length_l1939_193956


namespace M_intersect_N_l1939_193953

def M : Set ℤ := {-1, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = x^2}

theorem M_intersect_N : M ∩ N = {1} := by sorry

end M_intersect_N_l1939_193953


namespace inverse_proportion_order_l1939_193925

theorem inverse_proportion_order : ∀ y₁ y₂ y₃ : ℝ,
  y₁ = -6 / (-3) →
  y₂ = -6 / (-1) →
  y₃ = -6 / 2 →
  y₃ < y₁ ∧ y₁ < y₂ := by
sorry

end inverse_proportion_order_l1939_193925


namespace number_of_teams_in_league_l1939_193986

/-- The number of teams in the league -/
def n : ℕ := 20

/-- The number of games each team plays against every other team -/
def games_per_pair : ℕ := 4

/-- The total number of games played in the season -/
def total_games : ℕ := 760

/-- Theorem stating that n is the correct number of teams in the league -/
theorem number_of_teams_in_league :
  n * (n - 1) * games_per_pair / 2 = total_games :=
sorry

end number_of_teams_in_league_l1939_193986


namespace hexagon_dimension_theorem_l1939_193941

/-- Represents a hexagon that can be part of a rectangle and repositioned to form a square --/
structure Hexagon where
  area : ℝ
  significantDimension : ℝ

/-- Represents a rectangle that can be divided into two congruent hexagons --/
structure Rectangle where
  width : ℝ
  height : ℝ
  hexagons : Fin 2 → Hexagon
  isCongruent : hexagons 0 = hexagons 1

/-- Represents a square formed by repositioning two hexagons --/
structure Square where
  sideLength : ℝ

/-- Theorem stating the relationship between the rectangle, hexagons, and resulting square --/
theorem hexagon_dimension_theorem (rect : Rectangle) (sq : Square) : 
  rect.width = 9 ∧ 
  rect.height = 16 ∧ 
  (rect.width * rect.height = sq.sideLength * sq.sideLength) ∧
  (rect.hexagons 0).significantDimension = 6 := by
  sorry

#check hexagon_dimension_theorem

end hexagon_dimension_theorem_l1939_193941


namespace binomial_expansion_coefficient_l1939_193972

theorem binomial_expansion_coefficient (a b : ℝ) : 
  (∃ x, (1 + a*x)^5 = 1 + 10*x + b*x^2 + a^5*x^5) → b = 40 := by
sorry

end binomial_expansion_coefficient_l1939_193972


namespace miranda_rearrangement_time_l1939_193930

/-- Calculates the time in hours to write all rearrangements of a name -/
def time_to_write_rearrangements (num_letters : ℕ) (rearrangements_per_minute : ℕ) : ℚ :=
  (Nat.factorial num_letters : ℚ) / (rearrangements_per_minute : ℚ) / 60

/-- Proves that writing all rearrangements of a 6-letter name at 15 per minute takes 0.8 hours -/
theorem miranda_rearrangement_time :
  time_to_write_rearrangements 6 15 = 4/5 := by
  sorry

#eval time_to_write_rearrangements 6 15

end miranda_rearrangement_time_l1939_193930


namespace both_chromatids_contain_N15_l1939_193900

/-- Represents a chromatid -/
structure Chromatid where
  hasN15 : Bool

/-- Represents a chromosome with two chromatids -/
structure Chromosome where
  chromatid1 : Chromatid
  chromatid2 : Chromatid

/-- Represents a cell at the tetraploid stage -/
structure TetraploidCell where
  chromosomes : List Chromosome

/-- Represents the initial condition of progenitor cells -/
def initialProgenitorCell : Bool := true

/-- Represents the culture medium containing N -/
def cultureMediumWithN : Bool := true

/-- Theorem stating that both chromatids contain N15 at the tetraploid stage -/
theorem both_chromatids_contain_N15 (cell : TetraploidCell) 
  (h1 : initialProgenitorCell = true) 
  (h2 : cultureMediumWithN = true) : 
  ∀ c ∈ cell.chromosomes, c.chromatid1.hasN15 ∧ c.chromatid2.hasN15 := by
  sorry


end both_chromatids_contain_N15_l1939_193900


namespace jeonghoons_math_score_l1939_193988

theorem jeonghoons_math_score 
  (ethics : ℕ) (korean : ℕ) (science : ℕ) (social : ℕ) (average : ℕ) :
  ethics = 82 →
  korean = 90 →
  science = 88 →
  social = 84 →
  average = 88 →
  (ethics + korean + science + social + (average * 5 - (ethics + korean + science + social))) / 5 = average →
  average * 5 - (ethics + korean + science + social) = 96 :=
by sorry

end jeonghoons_math_score_l1939_193988


namespace greatest_among_five_l1939_193940

theorem greatest_among_five : ∀ (a b c d e : ℕ), 
  a = 5 → b = 8 → c = 4 → d = 3 → e = 2 →
  (b ≥ a ∧ b ≥ c ∧ b ≥ d ∧ b ≥ e) := by
  sorry

end greatest_among_five_l1939_193940


namespace total_birds_caught_l1939_193975

def bird_hunting (day_catch : ℕ) (night_multiplier : ℕ) : ℕ :=
  day_catch + night_multiplier * day_catch

theorem total_birds_caught : bird_hunting 8 2 = 24 := by
  sorry

end total_birds_caught_l1939_193975


namespace distinct_paths_count_l1939_193959

/-- Represents the number of purple arrows from point A -/
def purple_arrows : Nat := 2

/-- Represents the number of gray arrows each purple arrow leads to -/
def gray_arrows_per_purple : Nat := 2

/-- Represents the number of teal arrows each gray arrow leads to -/
def teal_arrows_per_gray : Nat := 3

/-- Represents the number of yellow arrows each teal arrow leads to -/
def yellow_arrows_per_teal : Nat := 2

/-- Represents the number of yellow arrows that lead to point B -/
def yellow_arrows_to_B : Nat := 4

/-- Theorem stating that the number of distinct paths from A to B is 96 -/
theorem distinct_paths_count : 
  purple_arrows * gray_arrows_per_purple * teal_arrows_per_gray * yellow_arrows_per_teal * yellow_arrows_to_B = 96 := by
  sorry

#eval purple_arrows * gray_arrows_per_purple * teal_arrows_per_gray * yellow_arrows_per_teal * yellow_arrows_to_B

end distinct_paths_count_l1939_193959


namespace brother_is_tweedledee_l1939_193969

-- Define the two brothers
inductive Brother
| tweedledee
| tweedledum

-- Define a proposition for "lying today"
def lying_today (b : Brother) : Prop := sorry

-- Define the statement made by the brother
def brother_statement (b : Brother) : Prop :=
  lying_today b ∨ b = Brother.tweedledee

-- Theorem stating that the brother must be Tweedledee
theorem brother_is_tweedledee (b : Brother) : 
  brother_statement b → b = Brother.tweedledee :=
by sorry

end brother_is_tweedledee_l1939_193969


namespace impossible_30_cents_with_5_coins_l1939_193955

def coin_values : List ℕ := [1, 5, 10, 25, 50]

theorem impossible_30_cents_with_5_coins :
  ¬ ∃ (coins : List ℕ), 
    coins.length = 5 ∧ 
    (∀ c ∈ coins, c ∈ coin_values) ∧ 
    coins.sum = 30 :=
by sorry

end impossible_30_cents_with_5_coins_l1939_193955


namespace prime_root_pairs_classification_l1939_193931

/-- A pair of positive primes (p,q) such that 3x^2 - px + q = 0 has two distinct rational roots -/
structure PrimeRootPair where
  p : ℕ
  q : ℕ
  p_prime : Nat.Prime p
  q_prime : Nat.Prime q
  has_distinct_rational_roots : ∃ (x y : ℚ), x ≠ y ∧ 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0

/-- The theorem stating that there are only two pairs of primes satisfying the condition -/
theorem prime_root_pairs_classification : 
  {pair : PrimeRootPair | True} = {⟨5, 2, sorry, sorry, sorry⟩, ⟨7, 2, sorry, sorry, sorry⟩} :=
by sorry

end prime_root_pairs_classification_l1939_193931


namespace inequality_proof_l1939_193995

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) : 
  b / Real.sqrt (a + 2 * c) + c / Real.sqrt (b + 2 * d) + 
  d / Real.sqrt (c + 2 * a) + a / Real.sqrt (d + 2 * b) ≥ 4 * Real.sqrt 3 / 3 := by
  sorry

end inequality_proof_l1939_193995


namespace quadratic_inequality_solution_l1939_193962

theorem quadratic_inequality_solution (c : ℝ) : 
  (c > 0 ∧ ∃ x : ℝ, x^2 - 8*x + c < 0) ↔ (0 < c ∧ c < 16) :=
sorry

end quadratic_inequality_solution_l1939_193962


namespace equal_remainders_implies_m_zero_l1939_193983

-- Define the polynomials
def P₁ (m : ℝ) (y : ℝ) : ℝ := 29 * 42 * y^2 + m * y + 2
def P₂ (m : ℝ) (y : ℝ) : ℝ := y^2 + m * y + 2

-- Define the remainder functions
def R₁ (m : ℝ) : ℝ := P₁ m 1
def R₂ (m : ℝ) : ℝ := P₂ m (-1)

-- Theorem statement
theorem equal_remainders_implies_m_zero :
  ∀ m : ℝ, R₁ m = R₂ m → m = 0 :=
by sorry

end equal_remainders_implies_m_zero_l1939_193983


namespace tea_trader_profit_percentage_l1939_193989

/-- Calculates the profit percentage for a tea trader --/
theorem tea_trader_profit_percentage
  (tea1_weight : ℝ) (tea1_cost : ℝ)
  (tea2_weight : ℝ) (tea2_cost : ℝ)
  (sale_price : ℝ)
  (h1 : tea1_weight = 80)
  (h2 : tea1_cost = 15)
  (h3 : tea2_weight = 20)
  (h4 : tea2_cost = 20)
  (h5 : sale_price = 19.2) :
  let total_cost := tea1_weight * tea1_cost + tea2_weight * tea2_cost
  let total_weight := tea1_weight + tea2_weight
  let cost_per_kg := total_cost / total_weight
  let profit_per_kg := sale_price - cost_per_kg
  let profit_percentage := (profit_per_kg / cost_per_kg) * 100
  profit_percentage = 20 := by
sorry

end tea_trader_profit_percentage_l1939_193989


namespace sport_to_standard_ratio_l1939_193926

/-- Represents the ratio of flavoring to corn syrup to water in the standard formulation -/
def standard_ratio : Fin 3 → ℚ
  | 0 => 1
  | 1 => 12
  | 2 => 30

/-- The amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 7

/-- The amount of water in the sport formulation (in ounces) -/
def sport_water : ℚ := 105

/-- The ratio of flavoring to water in the sport formulation compared to the standard formulation -/
def sport_flavoring_water_ratio : ℚ := 1/2

theorem sport_to_standard_ratio :
  let sport_flavoring := sport_water * (1 / (2 * standard_ratio 2))
  let sport_ratio := sport_flavoring / sport_corn_syrup
  let standard_ratio := (standard_ratio 0) / (standard_ratio 1)
  sport_ratio / standard_ratio = 1/3 := by sorry

end sport_to_standard_ratio_l1939_193926


namespace solve_paint_problem_l1939_193966

def paint_problem (original_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) : Prop :=
  ∃ (cans_per_room : ℚ) (total_cans : ℕ),
    cans_per_room > 0 ∧
    total_cans * cans_per_room = original_rooms ∧
    (total_cans - lost_cans) * cans_per_room = remaining_rooms ∧
    remaining_rooms / cans_per_room = 17

theorem solve_paint_problem :
  paint_problem 42 4 34 := by
  sorry

end solve_paint_problem_l1939_193966


namespace alfred_ranking_bounds_l1939_193938

/-- Represents a participant in the Generic Math Tournament -/
structure Participant where
  algebra_rank : Nat
  combinatorics_rank : Nat
  geometry_rank : Nat

/-- The total number of participants in the tournament -/
def total_participants : Nat := 99

/-- Alfred's rankings in each subject -/
def alfred : Participant :=
  { algebra_rank := 16
  , combinatorics_rank := 30
  , geometry_rank := 23 }

/-- Calculate the total score of a participant -/
def total_score (p : Participant) : Nat :=
  p.algebra_rank + p.combinatorics_rank + p.geometry_rank

/-- The best possible ranking Alfred could achieve -/
def best_ranking : Nat := 1

/-- The worst possible ranking Alfred could achieve -/
def worst_ranking : Nat := 67

theorem alfred_ranking_bounds :
  (∀ p : Participant, p ≠ alfred → total_score p ≠ total_score alfred) →
  (best_ranking = 1 ∧ worst_ranking = 67) :=
by sorry

end alfred_ranking_bounds_l1939_193938


namespace survey_methods_correct_l1939_193952

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a school survey --/
structure SchoolSurvey where
  totalStudents : Nat
  method1 : SamplingMethod
  method2 : SamplingMethod

/-- Defines the specific survey conducted by the school --/
def surveyConducted : SchoolSurvey := {
  totalStudents := 240,
  method1 := SamplingMethod.SimpleRandom,
  method2 := SamplingMethod.Systematic
}

/-- Theorem stating that the survey methods are correctly identified --/
theorem survey_methods_correct : 
  surveyConducted.method1 = SamplingMethod.SimpleRandom ∧
  surveyConducted.method2 = SamplingMethod.Systematic :=
by sorry

end survey_methods_correct_l1939_193952


namespace weekly_payment_problem_l1939_193928

/-- The weekly payment problem -/
theorem weekly_payment_problem (n_pay m_pay total_pay : ℕ) : 
  n_pay = 250 →
  m_pay = (120 * n_pay) / 100 →
  total_pay = m_pay + n_pay →
  total_pay = 550 := by
  sorry

end weekly_payment_problem_l1939_193928


namespace alcohol_concentration_proof_l1939_193998

-- Define the initial solution parameters
def initial_volume : ℝ := 6
def initial_concentration : ℝ := 0.35
def target_concentration : ℝ := 0.50

-- Define the amount of pure alcohol to be added
def added_alcohol : ℝ := 1.8

-- Theorem statement
theorem alcohol_concentration_proof :
  let initial_alcohol := initial_volume * initial_concentration
  let final_volume := initial_volume + added_alcohol
  let final_alcohol := initial_alcohol + added_alcohol
  (final_alcohol / final_volume) = target_concentration := by
  sorry


end alcohol_concentration_proof_l1939_193998


namespace jesses_room_length_l1939_193957

theorem jesses_room_length (width : ℝ) (total_area : ℝ) (h1 : width = 8) (h2 : total_area = 96) :
  total_area / width = 12 := by
sorry

end jesses_room_length_l1939_193957


namespace eight_to_power_divided_by_four_l1939_193916

theorem eight_to_power_divided_by_four (n : ℕ) : 
  n = 8^2022 → n / 4 = 4^3032 := by sorry

end eight_to_power_divided_by_four_l1939_193916


namespace milk_ratio_l1939_193935

def weekday_boxes : ℕ := 3
def saturday_boxes : ℕ := 2 * weekday_boxes
def total_boxes : ℕ := 30

def weekdays : ℕ := 5
def saturdays : ℕ := 1

def sunday_boxes : ℕ := total_boxes - (weekday_boxes * weekdays + saturday_boxes * saturdays)

theorem milk_ratio :
  (sunday_boxes : ℚ) / (weekday_boxes * weekdays : ℚ) = 3 / 5 := by sorry

end milk_ratio_l1939_193935


namespace tan_a_values_l1939_193960

theorem tan_a_values (a : Real) (h : Real.sin (2 * a) = 2 - 2 * Real.cos (2 * a)) :
  Real.tan a = 0 ∨ Real.tan a = 1 / 2 := by
  sorry

end tan_a_values_l1939_193960


namespace baker_cakes_sold_l1939_193979

theorem baker_cakes_sold (pastries_made : ℕ) (cakes_made : ℕ) (pastries_sold : ℕ) (cakes_left : ℕ) 
  (h1 : pastries_made = 61)
  (h2 : cakes_made = 167)
  (h3 : pastries_sold = 44)
  (h4 : cakes_left = 59) :
  cakes_made - cakes_left = 108 :=
by
  sorry

end baker_cakes_sold_l1939_193979


namespace quadratic_inequality_solution_l1939_193964

theorem quadratic_inequality_solution (a c : ℝ) : 
  (∀ x : ℝ, ax^2 + 5*x + c > 0 ↔ 1/3 < x ∧ x < 1/2) → 
  a + c = -7 :=
by sorry

end quadratic_inequality_solution_l1939_193964


namespace min_value_fraction_l1939_193954

theorem min_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) :
  (x + y) / (x^2) ≥ -1/12 := by sorry

end min_value_fraction_l1939_193954


namespace parallel_iff_a_eq_two_l1939_193934

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line ax + 2y = 0 -/
def line1 (a : ℝ) : Line :=
  { a := a, b := 2, c := 0 }

/-- The second line x + y = 1 -/
def line2 : Line :=
  { a := 1, b := 1, c := -1 }

/-- Theorem: a = 2 is necessary and sufficient for the lines to be parallel -/
theorem parallel_iff_a_eq_two :
  ∀ a : ℝ, parallel (line1 a) line2 ↔ a = 2 := by
  sorry

end parallel_iff_a_eq_two_l1939_193934


namespace quadratic_inequality_range_l1939_193902

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3/2 < 0) ↔ k ∈ Set.Ioc (-12) 0 := by
  sorry

end quadratic_inequality_range_l1939_193902


namespace triangle_balls_proof_l1939_193917

/-- The number of balls in an equilateral triangle arrangement -/
def triangle_balls : ℕ := 820

/-- The number of balls added to form a square -/
def added_balls : ℕ := 424

/-- The difference in side length between the triangle and the square -/
def side_difference : ℕ := 8

/-- Formula for the sum of the first n natural numbers -/
def triangle_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The side length of the triangle -/
def triangle_side : ℕ := 40

/-- The side length of the square -/
def square_side : ℕ := triangle_side - side_difference

theorem triangle_balls_proof :
  triangle_balls = triangle_sum triangle_side ∧
  triangle_balls + added_balls = square_side^2 ∧
  triangle_side = square_side + side_difference :=
sorry

end triangle_balls_proof_l1939_193917


namespace problem_statement_l1939_193927

theorem problem_statement (x y : ℝ) (h : Real.sqrt (x - 1) + (y + 2)^2 = 0) :
  (x + y)^2014 = 1 := by
  sorry

end problem_statement_l1939_193927


namespace chessboard_star_property_l1939_193914

/-- Represents a chessboard with stars -/
structure Chessboard (n : ℕ) where
  has_star : Fin n → Fin n → Prop

/-- Represents a set of rows or columns -/
def Subset (n : ℕ) := Fin n → Prop

/-- Checks if a subset is not the entire set -/
def is_proper_subset {n : ℕ} (s : Subset n) : Prop :=
  ∃ i, ¬s i

/-- Checks if a column has exactly one uncrossed star after crossing out rows -/
def column_has_one_star {n : ℕ} (b : Chessboard n) (crossed_rows : Subset n) (j : Fin n) : Prop :=
  ∃! i, ¬crossed_rows i ∧ b.has_star i j

/-- Checks if a row has exactly one uncrossed star after crossing out columns -/
def row_has_one_star {n : ℕ} (b : Chessboard n) (crossed_cols : Subset n) (i : Fin n) : Prop :=
  ∃! j, ¬crossed_cols j ∧ b.has_star i j

/-- The main theorem -/
theorem chessboard_star_property {n : ℕ} (b : Chessboard n) :
  (∀ crossed_rows : Subset n, is_proper_subset crossed_rows →
    ∃ j, column_has_one_star b crossed_rows j) →
  (∀ crossed_cols : Subset n, is_proper_subset crossed_cols →
    ∃ i, row_has_one_star b crossed_cols i) :=
by sorry

end chessboard_star_property_l1939_193914


namespace geometric_sequence_17th_term_l1939_193921

/-- Given a geometric sequence where a₅ = 5 and a₁₁ = 40, prove that a₁₇ = 320 -/
theorem geometric_sequence_17th_term (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) / a n = a 2 / a 1) 
    (h2 : a 5 = 5) (h3 : a 11 = 40) : a 17 = 320 := by
  sorry

end geometric_sequence_17th_term_l1939_193921


namespace original_typing_speed_l1939_193999

theorem original_typing_speed 
  (original_speed : ℕ) 
  (speed_decrease : ℕ) 
  (words_typed : ℕ) 
  (time_taken : ℕ) :
  speed_decrease = 40 →
  words_typed = 3440 →
  time_taken = 20 →
  (original_speed - speed_decrease) * time_taken = words_typed →
  original_speed = 212 := by
sorry

end original_typing_speed_l1939_193999


namespace golus_journey_l1939_193990

theorem golus_journey (a b c : ℝ) (h1 : a = 8) (h2 : c = 10) (h3 : a^2 + b^2 = c^2) : b = 6 := by
  sorry

end golus_journey_l1939_193990


namespace mean_equality_implies_y_l1939_193996

theorem mean_equality_implies_y (y : ℝ) : 
  (7 + 9 + 14 + 23) / 4 = (18 + y) / 2 → y = 8.5 := by
  sorry

end mean_equality_implies_y_l1939_193996


namespace min_value_f_max_value_g_l1939_193909

-- Define the functions
def f (m : ℝ) : ℝ := m^2 + 2*m + 3
def g (m : ℝ) : ℝ := -m^2 + 2*m + 3

-- Theorem for the minimum value of f
theorem min_value_f : ∀ m : ℝ, f m ≥ 2 ∧ ∃ m₀ : ℝ, f m₀ = 2 :=
sorry

-- Theorem for the maximum value of g
theorem max_value_g : ∀ m : ℝ, g m ≤ 4 ∧ ∃ m₀ : ℝ, g m₀ = 4 :=
sorry

end min_value_f_max_value_g_l1939_193909


namespace range_of_m_l1939_193905

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |x - m| ≤ 2) → 
  -1 ≤ m ∧ m ≤ 1 := by
sorry

end range_of_m_l1939_193905


namespace red_highest_probability_l1939_193945

/-- A color of a ball -/
inductive Color
  | Red
  | Yellow
  | White

/-- The number of balls of each color in the bag -/
def ballCount (c : Color) : ℕ :=
  match c with
  | Color.Red => 6
  | Color.Yellow => 4
  | Color.White => 1

/-- The total number of balls in the bag -/
def totalBalls : ℕ := ballCount Color.Red + ballCount Color.Yellow + ballCount Color.White

/-- The probability of drawing a ball of a given color -/
def probability (c : Color) : ℚ :=
  ballCount c / totalBalls

/-- Theorem: The probability of drawing a red ball is the highest -/
theorem red_highest_probability :
  probability Color.Red > probability Color.Yellow ∧
  probability Color.Red > probability Color.White :=
sorry

end red_highest_probability_l1939_193945


namespace steel_rod_length_l1939_193997

/-- Represents the properties of a uniform steel rod -/
structure SteelRod where
  /-- The weight of the rod in kilograms -/
  weight : ℝ
  /-- The length of the rod in meters -/
  length : ℝ
  /-- The rod is uniform, so weight per unit length is constant -/
  uniform : weight / length = 19 / 5

/-- Theorem stating that a steel rod weighing 42.75 kg has a length of 11.25 meters -/
theorem steel_rod_length (rod : SteelRod) (h : rod.weight = 42.75) : rod.length = 11.25 := by
  sorry

end steel_rod_length_l1939_193997


namespace max_value_theorem_l1939_193904

theorem max_value_theorem (a c : ℝ) (ha : 0 < a) (hc : 0 < c) :
  (∀ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + c^2)) ≤ a^2 + c^2) ∧
  (∃ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + c^2)) = a^2 + c^2) :=
by sorry

end max_value_theorem_l1939_193904


namespace min_abs_z_l1939_193993

/-- Given a complex number z satisfying |z - 16| + |z + 3i| = 17, 
    the smallest possible value of |z| is 768/265 -/
theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 16) + Complex.abs (z + 3*I) = 17) :
  ∃ (w : ℂ), Complex.abs (z - 16) + Complex.abs (z + 3*I) = 17 ∧ 
             Complex.abs w ≤ Complex.abs z ∧
             Complex.abs w = 768 / 265 :=
sorry

end min_abs_z_l1939_193993


namespace subcommittee_formation_count_l1939_193991

def senate_committee_ways (total_republicans : ℕ) (total_democrats : ℕ) 
  (subcommittee_republicans : ℕ) (subcommittee_democrats : ℕ) : ℕ :=
  Nat.choose total_republicans subcommittee_republicans * 
  Nat.choose total_democrats subcommittee_democrats

theorem subcommittee_formation_count : 
  senate_committee_ways 10 8 4 3 = 11760 := by
  sorry

end subcommittee_formation_count_l1939_193991


namespace cubic_equation_roots_l1939_193906

theorem cubic_equation_roots (k m : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (∀ x : ℝ, x^3 - 9*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  k + m = 50 := by
sorry

end cubic_equation_roots_l1939_193906


namespace rationalize_denominator_l1939_193923

theorem rationalize_denominator :
  (2 * Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 5 + Real.sqrt 3) = (3 * Real.sqrt 15 - 7) / 2 := by
  sorry

end rationalize_denominator_l1939_193923


namespace complex_equality_l1939_193978

theorem complex_equality (a b : ℂ) : a - b = 0 → a = b := by sorry

end complex_equality_l1939_193978


namespace max_value_abcd_l1939_193967

theorem max_value_abcd (a b c d : ℤ) (hb : b > 0) 
  (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) : 
  (∀ a' b' c' d' : ℤ, b' > 0 → a' + b' = c' → b' + c' = d' → c' + d' = a' → 
    a' - 2*b' + 3*c' - 4*d' ≤ a - 2*b + 3*c - 4*d) ∧ 
  (a - 2*b + 3*c - 4*d = -7) := by
  sorry

end max_value_abcd_l1939_193967


namespace condition_equivalence_l1939_193908

theorem condition_equivalence (p q : Prop) :
  (¬(p ∧ q) ∧ (p ∨ q)) ↔ (p ≠ q) :=
sorry

end condition_equivalence_l1939_193908


namespace constant_term_of_expansion_l1939_193932

/-- The constant term in the expansion of (x^2 - 2/x)^6 -/
def constantTerm : ℤ := 240

/-- The binomial expansion of (x^2 - 2/x)^6 -/
def expansion (x : ℚ) : ℚ := (x^2 - 2/x)^6

theorem constant_term_of_expansion :
  ∃ (f : ℚ → ℚ), (∀ x ≠ 0, f x = expansion x) ∧ 
  (∃ c : ℚ, ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x - c| < ε) ∧
  (c = constantTerm) := by
  sorry

end constant_term_of_expansion_l1939_193932


namespace bart_tree_cutting_l1939_193933

/-- The number of pieces of firewood obtained from one tree -/
def pieces_per_tree : ℕ := 75

/-- The number of pieces of firewood Bart burns daily -/
def daily_burn_rate : ℕ := 5

/-- The number of days from November 1 through February 28 -/
def total_days : ℕ := 120

/-- The number of trees Bart needs to cut down -/
def trees_needed : ℕ := (daily_burn_rate * total_days) / pieces_per_tree

theorem bart_tree_cutting :
  trees_needed = 8 :=
sorry

end bart_tree_cutting_l1939_193933


namespace f_increasing_on_positive_reals_l1939_193939

-- Define the function f(x) = x² + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- Theorem statement
theorem f_increasing_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y :=
by sorry

end f_increasing_on_positive_reals_l1939_193939


namespace perpendicular_lines_main_theorem_l1939_193936

/-- Two lines are perpendicular if their slopes multiply to -1 or if one of them is vertical --/
def perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1 ∨ m1 = 0 ∨ m2 = 0

theorem perpendicular_lines (a : ℝ) :
  perpendicular (-a/2) (-1/(a*(a+1))) → a = -3/2 ∨ a = 0 := by
  sorry

/-- The main theorem stating the conditions for perpendicularity of the given lines --/
theorem main_theorem :
  ∀ a : ℝ, (∃ x y : ℝ, a*x + 2*y + 6 = 0 ∧ x + a*(a+1)*y + (a^2-1) = 0) →
  perpendicular (-a/2) (-1/(a*(a+1))) →
  a = -3/2 ∨ a = 0 := by
  sorry

end perpendicular_lines_main_theorem_l1939_193936


namespace cotton_planting_solution_l1939_193971

/-- Represents the cotton planting problem with given parameters -/
structure CottonPlanting where
  total_area : ℕ
  total_days : ℕ
  first_crew_tractors : ℕ
  first_crew_days : ℕ
  second_crew_tractors : ℕ
  second_crew_days : ℕ

/-- Calculates the required acres per tractor per day -/
def acres_per_tractor_per_day (cp : CottonPlanting) : ℚ :=
  cp.total_area / (cp.first_crew_tractors * cp.first_crew_days + cp.second_crew_tractors * cp.second_crew_days)

/-- Theorem stating that for the given parameters, each tractor needs to plant 68 acres per day -/
theorem cotton_planting_solution (cp : CottonPlanting) 
  (h1 : cp.total_area = 1700)
  (h2 : cp.total_days = 5)
  (h3 : cp.first_crew_tractors = 2)
  (h4 : cp.first_crew_days = 2)
  (h5 : cp.second_crew_tractors = 7)
  (h6 : cp.second_crew_days = 3) :
  acres_per_tractor_per_day cp = 68 := by
  sorry

end cotton_planting_solution_l1939_193971


namespace trigonometric_identity_l1939_193907

theorem trigonometric_identity (α : ℝ) :
  3.404 * (8 * Real.cos α ^ 4 - 4 * Real.cos α ^ 3 - 8 * Real.cos α ^ 2 + 3 * Real.cos α + 1) /
  (8 * Real.cos α ^ 4 + 4 * Real.cos α ^ 3 - 8 * Real.cos α ^ 2 - 3 * Real.cos α + 1) =
  -Real.tan (7 * α / 2) * Real.tan (α / 2) := by
  sorry

end trigonometric_identity_l1939_193907


namespace shaded_area_percentage_l1939_193948

-- Define the square PQRS
def square_side : ℝ := 7

-- Define the shaded areas
def shaded_area_1 : ℝ := 2^2
def shaded_area_2 : ℝ := 5^2 - 3^2
def shaded_area_3 : ℝ := square_side^2 - 6^2

-- Total shaded area
def total_shaded_area : ℝ := shaded_area_1 + shaded_area_2 + shaded_area_3

-- Total area of square PQRS
def total_area : ℝ := square_side^2

-- Theorem statement
theorem shaded_area_percentage :
  total_shaded_area = 33 ∧ (total_shaded_area / total_area) = 33 / 49 := by
  sorry

end shaded_area_percentage_l1939_193948


namespace imaginary_unit_power_2013_l1939_193901

theorem imaginary_unit_power_2013 (i : ℂ) (h : i^2 = -1) : i^2013 = i := by
  sorry

end imaginary_unit_power_2013_l1939_193901


namespace pizza_sales_l1939_193922

theorem pizza_sales (pepperoni cheese total : ℕ) (h1 : pepperoni = 2) (h2 : cheese = 6) (h3 : total = 14) :
  total - (pepperoni + cheese) = 6 := by
  sorry

end pizza_sales_l1939_193922


namespace max_profit_is_900_l1939_193987

/-- Represents the daily sales volume as a function of the selling price. -/
def sales_volume (x : ℕ) : ℤ := -10 * x + 300

/-- Represents the daily profit as a function of the selling price. -/
def profit (x : ℕ) : ℤ := (x - 11) * sales_volume x

/-- The selling price that maximizes profit. -/
def optimal_price : ℕ := 20

theorem max_profit_is_900 :
  ∀ x : ℕ, x > 0 → profit x ≤ 900 ∧ profit optimal_price = 900 := by
  sorry

#eval profit optimal_price

end max_profit_is_900_l1939_193987


namespace log_inequality_implication_l1939_193981

theorem log_inequality_implication (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.log 3 / Real.log a < Real.log 3 / Real.log b) ∧
  (Real.log 3 / Real.log b < Real.log 3 / Real.log c) →
  ¬(a < b ∧ b < c) :=
by sorry

end log_inequality_implication_l1939_193981


namespace expression_equality_l1939_193961

theorem expression_equality : 
  Real.sqrt 3 * Real.tan (30 * π / 180) - (1 / 2)⁻¹ + Real.sqrt 8 - |1 - Real.sqrt 2| = Real.sqrt 2 := by
  sorry

end expression_equality_l1939_193961


namespace rectangle_two_axes_l1939_193919

-- Define the types of shapes
inductive Shape
  | EquilateralTriangle
  | Parallelogram
  | Rectangle
  | Square

-- Define a function to count axes of symmetry
def axesOfSymmetry (s : Shape) : ℕ :=
  match s with
  | Shape.EquilateralTriangle => 3
  | Shape.Parallelogram => 0
  | Shape.Rectangle => 2
  | Shape.Square => 4

-- Theorem statement
theorem rectangle_two_axes :
  ∀ s : Shape, axesOfSymmetry s = 2 ↔ s = Shape.Rectangle :=
by sorry

end rectangle_two_axes_l1939_193919


namespace accounting_majors_count_l1939_193980

theorem accounting_majors_count 
  (p q r s : ℕ+) 
  (h1 : p * q * r * s = 1365)
  (h2 : 1 < p) (h3 : p < q) (h4 : q < r) (h5 : r < s) :
  p = 3 := by
sorry

end accounting_majors_count_l1939_193980
