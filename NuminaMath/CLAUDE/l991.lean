import Mathlib

namespace chess_tournament_games_l991_99193

theorem chess_tournament_games (n : ℕ) (h : n = 14) : 
  (n.choose 2) = 91 := by
  sorry

end chess_tournament_games_l991_99193


namespace max_value_a_l991_99176

theorem max_value_a (x y : ℝ) 
  (h1 : x - y ≤ 0) 
  (h2 : x + y - 5 ≥ 0) 
  (h3 : y - 3 ≤ 0) : 
  (∃ (a : ℝ), a = 25/13 ∧ 
    (∀ (b : ℝ), (∀ (x y : ℝ), 
      x - y ≤ 0 → x + y - 5 ≥ 0 → y - 3 ≤ 0 → 
      b * (x^2 + y^2) ≤ (x + y)^2) → 
    b ≤ a)) :=
by sorry

end max_value_a_l991_99176


namespace isosceles_right_triangle_l991_99139

theorem isosceles_right_triangle (A B C : ℝ) (a b c : ℝ) : 
  (Real.sin (A - B))^2 + (Real.cos C)^2 = 0 → 
  (A = B ∧ C = Real.pi / 2) :=
by sorry

end isosceles_right_triangle_l991_99139


namespace problem_solution_l991_99187

theorem problem_solution (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xyz : x * y * z = 1) (h_x_z : x + 1 / z = 7) (h_y_x : y + 1 / x = 31) :
  z + 1 / y = 5 / 27 := by
sorry

end problem_solution_l991_99187


namespace smallest_winning_number_l991_99155

theorem smallest_winning_number : ∃ N : ℕ, 
  (N = 28) ∧ 
  (0 ≤ N) ∧ (N ≤ 999) ∧
  (36 * N < 2000) ∧
  (72 * N ≥ 2000) ∧
  (∀ M : ℕ, M < N → 
    (M = 0) ∨ (M > 999) ∨ 
    (36 * M ≥ 2000) ∨ 
    (72 * M < 2000)) :=
by sorry

end smallest_winning_number_l991_99155


namespace abs_minus_sqrt_eq_three_l991_99166

theorem abs_minus_sqrt_eq_three (a : ℝ) (h : a < 0) : |a - 3| - Real.sqrt (a^2) = 3 := by
  sorry

end abs_minus_sqrt_eq_three_l991_99166


namespace polygon_sides_l991_99130

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →
  ((n - 2) * 180 + 360 = 1260) →
  n = 7 :=
by sorry

end polygon_sides_l991_99130


namespace remainder_6n_mod_4_l991_99133

theorem remainder_6n_mod_4 (n : ℤ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := by
  sorry

end remainder_6n_mod_4_l991_99133


namespace imaginary_part_of_z_l991_99136

theorem imaginary_part_of_z : Complex.im ((-3 + Complex.I) / Complex.I^3) = -3 := by
  sorry

end imaginary_part_of_z_l991_99136


namespace integer_power_sum_l991_99115

theorem integer_power_sum (a : ℝ) (h : ∃ k : ℤ, a + 1/a = k) :
  ∀ n : ℕ, ∃ m : ℤ, a^n + 1/(a^n) = m :=
sorry

end integer_power_sum_l991_99115


namespace hyperbola_eccentricity_l991_99159

/-- A hyperbola with the given properties has eccentricity √3 -/
theorem hyperbola_eccentricity (a b c : ℝ) (P : ℝ × ℝ) :
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  let e := c / a
  -- Hyperbola equation
  (P.1 / a) ^ 2 - (P.2 / b) ^ 2 = 1 ∧
  -- Line through F₁ at 30° inclination
  (P.2 + c * Real.tan (30 * π / 180)) / (P.1 + c) = Real.tan (30 * π / 180) ∧
  -- Circle with diameter PF₁ passes through F₂
  (P.1 - (-c)) ^ 2 + P.2 ^ 2 = (2 * c) ^ 2 ∧
  -- Standard hyperbola relations
  c ^ 2 = a ^ 2 + b ^ 2 ∧
  P.1 > 0 -- P is on the right branch
  →
  e = Real.sqrt 3 :=
by sorry

end hyperbola_eccentricity_l991_99159


namespace repeating_decimal_sum_l991_99126

/-- Represents a repeating decimal in the form 0.nnn... where n is a single digit -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

theorem repeating_decimal_sum :
  RepeatingDecimal 6 - RepeatingDecimal 4 + RepeatingDecimal 8 = 10 / 9 := by
  sorry

end repeating_decimal_sum_l991_99126


namespace smallest_valid_seating_l991_99104

/-- Represents a circular table with chairs and people seated. -/
structure CircularTable where
  totalChairs : ℕ
  seatedPeople : ℕ

/-- Checks if the seating arrangement is valid. -/
def isValidSeating (table : CircularTable) : Prop :=
  table.seatedPeople > 0 ∧ 
  table.seatedPeople ≤ table.totalChairs ∧
  ∀ n : ℕ, n ≤ table.seatedPeople → ∃ m : ℕ, m < n ∧ (n - m = 1 ∨ m - n = 1 ∨ n = 1)

/-- The theorem to be proved. -/
theorem smallest_valid_seating (table : CircularTable) :
  table.totalChairs = 75 →
  (isValidSeating table ∧ ∀ t : CircularTable, t.totalChairs = 75 → isValidSeating t → t.seatedPeople ≥ table.seatedPeople) →
  table.seatedPeople = 25 := by
  sorry

end smallest_valid_seating_l991_99104


namespace path_length_squares_l991_99123

/-- Given a line PQ of length 24 cm divided into six equal parts, with squares drawn on each part,
    the path following three sides of each square from P to Q is 72 cm long. -/
theorem path_length_squares (PQ : ℝ) (num_parts : ℕ) : 
  PQ = 24 →
  num_parts = 6 →
  (num_parts : ℝ) * (3 * (PQ / num_parts)) = 72 :=
by sorry

end path_length_squares_l991_99123


namespace square_root_equation_implies_y_minus_x_equals_two_l991_99183

theorem square_root_equation_implies_y_minus_x_equals_two (x y : ℝ) :
  Real.sqrt (x + 1) - Real.sqrt (-1 - x) = (x + y)^2 → y - x = 2 := by
  sorry

end square_root_equation_implies_y_minus_x_equals_two_l991_99183


namespace money_sum_l991_99179

theorem money_sum (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 320) (h3 : C = 20) :
  A + B + C = 500 := by
  sorry

end money_sum_l991_99179


namespace language_letters_l991_99151

theorem language_letters (n : ℕ) : 
  (n + n^2) - ((n - 1) + (n - 1)^2) = 129 → n = 65 := by
  sorry

end language_letters_l991_99151


namespace derivative_sqrt_derivative_log2_l991_99121

-- Define the derivative of square root
theorem derivative_sqrt (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.sqrt x) x = 1 / (2 * Real.sqrt x) := by sorry

-- Define the derivative of log base 2
theorem derivative_log2 (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by sorry

end derivative_sqrt_derivative_log2_l991_99121


namespace cafe_tables_l991_99119

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Calculates the number of tables needed given the number of people and people per table -/
def tablesNeeded (people : ℕ) (peoplePerTable : ℕ) : ℕ := 
  (people + peoplePerTable - 1) / peoplePerTable

theorem cafe_tables : 
  let totalPeople : ℕ := base7ToBase10 310
  let peoplePerTable : ℕ := 3
  tablesNeeded totalPeople peoplePerTable = 52 := by sorry

end cafe_tables_l991_99119


namespace percentage_difference_l991_99182

theorem percentage_difference (p j t : ℝ) 
  (hj : j = 0.75 * p) 
  (ht : t = 0.9375 * p) : 
  (t - j) / t = 0.2 := by
sorry

end percentage_difference_l991_99182


namespace yearly_increase_fraction_l991_99116

/-- 
Given an initial amount that increases by a fraction each year, 
this theorem proves that the fraction is 0.125 when the initial amount 
is 3200 and becomes 4050 after two years.
-/
theorem yearly_increase_fraction 
  (initial_amount : ℝ) 
  (final_amount : ℝ) 
  (f : ℝ) 
  (h1 : initial_amount = 3200) 
  (h2 : final_amount = 4050) 
  (h3 : final_amount = initial_amount * (1 + f)^2) : 
  f = 0.125 := by
sorry

end yearly_increase_fraction_l991_99116


namespace kat_strength_training_frequency_l991_99161

/-- Kat's weekly training schedule -/
structure TrainingSchedule where
  strength_duration : ℝ  -- Duration of each strength training session in hours
  strength_frequency : ℝ  -- Number of strength training sessions per week
  boxing_duration : ℝ     -- Duration of each boxing session in hours
  boxing_frequency : ℝ    -- Number of boxing sessions per week
  total_hours : ℝ         -- Total training hours per week

/-- Theorem stating that Kat does strength training 3 times a week -/
theorem kat_strength_training_frequency 
  (schedule : TrainingSchedule) 
  (h1 : schedule.strength_duration = 1)
  (h2 : schedule.boxing_duration = 1.5)
  (h3 : schedule.boxing_frequency = 4)
  (h4 : schedule.total_hours = 9)
  (h5 : schedule.total_hours = schedule.strength_duration * schedule.strength_frequency + 
                               schedule.boxing_duration * schedule.boxing_frequency) :
  schedule.strength_frequency = 3 := by
  sorry

#check kat_strength_training_frequency

end kat_strength_training_frequency_l991_99161


namespace ellipse_angle_bisector_l991_99154

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

/-- Definition of a point being on a chord through F -/
def is_on_chord_through_F (x y : ℝ) : Prop := 
  ∃ (m : ℝ), y = m * (x - 2)

/-- Definition of the angle equality condition -/
def angle_equality (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₁ / (x₁ - p)) = -(y₂ / (x₂ - p))

/-- The main theorem -/
theorem ellipse_angle_bisector :
  ∃! (p : ℝ), p > 0 ∧ 
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    is_on_ellipse x₁ y₁ ∧ is_on_ellipse x₂ y₂ ∧
    is_on_chord_through_F x₁ y₁ ∧ is_on_chord_through_F x₂ y₂ ∧
    x₁ ≠ x₂ →
    angle_equality p x₁ y₁ x₂ y₂) ∧
  p = 2 :=
sorry

end ellipse_angle_bisector_l991_99154


namespace prism_with_five_faces_has_nine_edges_l991_99114

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  faces : ℕ
  edges : ℕ

/-- Theorem: A prism with 5 faces has 9 edges. -/
theorem prism_with_five_faces_has_nine_edges (p : Prism) (h : p.faces = 5) : p.edges = 9 := by
  sorry


end prism_with_five_faces_has_nine_edges_l991_99114


namespace notebook_cost_l991_99132

/-- The cost of a notebook and a pen given two equations -/
theorem notebook_cost (n p : ℚ) 
  (eq1 : 3 * n + 4 * p = 3.75)
  (eq2 : 5 * n + 2 * p = 3.05) :
  n = 0.3357 := by
  sorry

end notebook_cost_l991_99132


namespace range_of_a_l991_99188

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + 1 > 0) ∨ (∃ x : ℝ, a*x^2 + 2 ≤ 0) = False →
  a ∈ Set.Ici 1 := by
sorry

end range_of_a_l991_99188


namespace cubic_linear_inequality_l991_99134

theorem cubic_linear_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^3 + b^3 + a + b ≥ 4 * a * b := by
  sorry

end cubic_linear_inequality_l991_99134


namespace angle_C_measure_l991_99135

-- Define the triangle and its angles
structure Triangle :=
  (A B C : ℝ)

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.B = t.A + 20 ∧ t.C = t.A + 40 ∧ t.A + t.B + t.C = 180

-- Theorem statement
theorem angle_C_measure (t : Triangle) :
  satisfies_conditions t → t.C = 80 := by
  sorry

end angle_C_measure_l991_99135


namespace northern_walks_of_length_6_l991_99142

/-- A northern walk is a path on a grid with the following properties:
  1. It starts at the origin.
  2. Each step is 1 unit north, east, or west.
  3. It never revisits a point.
  4. It has a specified length. -/
def NorthernWalk (length : ℕ) : Type := Unit

/-- Count the number of northern walks of a given length. -/
def countNorthernWalks (length : ℕ) : ℕ := sorry

/-- The main theorem stating that there are 239 northern walks of length 6. -/
theorem northern_walks_of_length_6 : countNorthernWalks 6 = 239 := by sorry

end northern_walks_of_length_6_l991_99142


namespace random_walk_2d_properties_l991_99140

-- Define the random walk on a 2D grid
def RandomWalk2D := ℕ × ℕ → ℝ

-- Probability of reaching a specific x-coordinate
def prob_reach_x (walk : RandomWalk2D) (x : ℕ) : ℝ := sorry

-- Expected y-coordinate when reaching a specific x-coordinate
def expected_y_at_x (walk : RandomWalk2D) (x : ℕ) : ℝ := sorry

-- Theorem statement
theorem random_walk_2d_properties (walk : RandomWalk2D) :
  (∀ x : ℕ, prob_reach_x walk x = 1) ∧
  (∀ n : ℕ, expected_y_at_x walk n = n) := by
  sorry

end random_walk_2d_properties_l991_99140


namespace pair_farm_animals_l991_99129

/-- Represents the number of ways to pair animals of different species -/
def pairAnimals (cows pigs horses : ℕ) : ℕ :=
  let cowPigPairs := cows * pigs
  let remainingPairs := Nat.factorial horses
  cowPigPairs * remainingPairs

/-- Theorem stating the number of ways to pair 5 cows, 4 pigs, and 7 horses -/
theorem pair_farm_animals :
  pairAnimals 5 4 7 = 100800 := by
  sorry

#eval pairAnimals 5 4 7

end pair_farm_animals_l991_99129


namespace min_tiles_for_floor_coverage_l991_99113

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Calculates the number of smaller rectangles needed to cover a larger rectangle -/
def tilesNeeded (region : Dimensions) (tile : Dimensions) : ℕ :=
  (area region + area tile - 1) / area tile

theorem min_tiles_for_floor_coverage :
  let tile := Dimensions.mk 2 6
  let region := Dimensions.mk (feetToInches 3) (feetToInches 4)
  tilesNeeded region tile = 144 := by
    sorry

end min_tiles_for_floor_coverage_l991_99113


namespace developed_countries_modern_pattern_l991_99128

/-- Represents different types of countries --/
inductive CountryType
| Developed
| Developing

/-- Represents different population growth patterns --/
inductive GrowthPattern
| Traditional
| Modern

/-- Represents the growth rate of a country --/
structure GrowthRate where
  rate : ℝ

/-- A country with its properties --/
structure Country where
  type : CountryType
  growthPattern : GrowthPattern
  growthRate : GrowthRate
  hasImplementedFamilyPlanning : Bool

/-- Axiom: Developed countries have slow growth rates --/
axiom developed_country_slow_growth (c : Country) :
  c.type = CountryType.Developed → c.growthRate.rate ≤ 0

/-- Axiom: Developing countries have faster growth rates --/
axiom developing_country_faster_growth (c : Country) :
  c.type = CountryType.Developing → c.growthRate.rate > 0

/-- Axiom: Most developing countries are in the traditional growth pattern --/
axiom most_developing_traditional (c : Country) :
  c.type = CountryType.Developing → c.growthPattern = GrowthPattern.Traditional

/-- Axiom: Countries with family planning are in the modern growth pattern --/
axiom family_planning_modern_pattern (c : Country) :
  c.hasImplementedFamilyPlanning → c.growthPattern = GrowthPattern.Modern

/-- Theorem: Developed countries are in the modern population growth pattern --/
theorem developed_countries_modern_pattern (c : Country) :
  c.type = CountryType.Developed → c.growthPattern = GrowthPattern.Modern := by
  sorry

end developed_countries_modern_pattern_l991_99128


namespace x_eq_two_iff_quadratic_eq_zero_l991_99163

theorem x_eq_two_iff_quadratic_eq_zero : ∀ x : ℝ, x = 2 ↔ x^2 - 4*x + 4 = 0 := by
  sorry

end x_eq_two_iff_quadratic_eq_zero_l991_99163


namespace brownie_pieces_count_l991_99178

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a pan of brownies -/
structure BrowniePan where
  panDimensions : Dimensions
  pieceDimensions : Dimensions

/-- Calculates the number of brownie pieces that can be cut from the pan -/
def numberOfPieces (pan : BrowniePan) : ℕ :=
  area pan.panDimensions / area pan.pieceDimensions

/-- Theorem: A 30-inch by 24-inch pan can be divided into exactly 120 pieces of 3-inch by 2-inch brownies -/
theorem brownie_pieces_count :
  let pan : BrowniePan := {
    panDimensions := { length := 30, width := 24 },
    pieceDimensions := { length := 3, width := 2 }
  }
  numberOfPieces pan = 120 := by
  sorry


end brownie_pieces_count_l991_99178


namespace wall_thickness_calculation_l991_99186

/-- Calculates the thickness of a wall given brick dimensions and wall specifications -/
theorem wall_thickness_calculation (brick_length brick_width brick_height : ℝ)
                                   (wall_length wall_height : ℝ)
                                   (num_bricks : ℕ) :
  brick_length = 50 →
  brick_width = 11.25 →
  brick_height = 6 →
  wall_length = 800 →
  wall_height = 600 →
  num_bricks = 3200 →
  ∃ (wall_thickness : ℝ),
    wall_thickness = 22.5 ∧
    wall_length * wall_height * wall_thickness = num_bricks * brick_length * brick_width * brick_height :=
by
  sorry

#check wall_thickness_calculation

end wall_thickness_calculation_l991_99186


namespace root_property_l991_99190

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 4*x - a

-- State the theorem
theorem root_property (a : ℝ) (x₁ x₂ x₃ : ℝ) 
  (h₁ : 0 < a) (h₂ : a < 2)
  (h₃ : f a x₁ = 0) (h₄ : f a x₂ = 0) (h₅ : f a x₃ = 0)
  (h₆ : x₁ < x₂) (h₇ : x₂ < x₃) :
  x₂ > 0 := by
  sorry

end root_property_l991_99190


namespace exists_triangle_with_different_colors_l991_99194

/-- A color type representing the three possible colors of vertices -/
inductive Color
  | A
  | B
  | C

/-- A graph representing the triangulation -/
structure Graph (α : Type) where
  V : Set α
  E : Set (α × α)

/-- A coloring function that assigns a color to each vertex -/
def Coloring (α : Type) := α → Color

/-- A predicate to check if three vertices form a triangle in the graph -/
def IsTriangle {α : Type} (G : Graph α) (a b c : α) : Prop :=
  a ∈ G.V ∧ b ∈ G.V ∧ c ∈ G.V ∧
  (a, b) ∈ G.E ∧ (b, c) ∈ G.E ∧ (c, a) ∈ G.E

/-- The main theorem statement -/
theorem exists_triangle_with_different_colors {α : Type} (G : Graph α) (f : Coloring α)
  (hA : ∃ a ∈ G.V, f a = Color.A)
  (hB : ∃ b ∈ G.V, f b = Color.B)
  (hC : ∃ c ∈ G.V, f c = Color.C) :
  ∃ x y z : α, IsTriangle G x y z ∧ f x ≠ f y ∧ f y ≠ f z ∧ f z ≠ f x :=
sorry

end exists_triangle_with_different_colors_l991_99194


namespace quadratic_inequality_range_l991_99124

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 < 0) → (a < -2 ∨ a > 2) := by
  sorry

end quadratic_inequality_range_l991_99124


namespace problem_solution_l991_99105

theorem problem_solution : (2023^2 - 2023 - 4^2) / 2023 = 2022 - 16/2023 := by
  sorry

end problem_solution_l991_99105


namespace max_sum_of_squares_l991_99112

/-- Given that x₁ and x₂ are real roots of the equation x² - (k-2)x + (k² + 3k + 5) = 0,
    where k is a real number, prove that the maximum value of x₁² + x₂² is 18. -/
theorem max_sum_of_squares (k : ℝ) (x₁ x₂ : ℝ) 
    (h₁ : x₁^2 - (k-2)*x₁ + (k^2 + 3*k + 5) = 0)
    (h₂ : x₂^2 - (k-2)*x₂ + (k^2 + 3*k + 5) = 0)
    (h₃ : x₁ ≠ x₂) : 
  ∃ (M : ℝ), M = 18 ∧ x₁^2 + x₂^2 ≤ M :=
by sorry

end max_sum_of_squares_l991_99112


namespace total_rats_l991_99174

/-- The number of rats each person has -/
structure RatCounts where
  elodie : ℕ
  hunter : ℕ
  kenia : ℕ
  teagan : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (rc : RatCounts) : Prop :=
  rc.elodie = 30 ∧
  rc.hunter = rc.elodie - 10 ∧
  rc.kenia = 3 * (rc.hunter + rc.elodie) ∧
  rc.teagan = 2 * rc.elodie ∧
  rc.teagan = rc.kenia - 5

/-- The theorem stating that the total number of rats is 260 -/
theorem total_rats (rc : RatCounts) (h : satisfiesConditions rc) :
  rc.elodie + rc.hunter + rc.kenia + rc.teagan = 260 :=
by sorry

end total_rats_l991_99174


namespace negation_equivalence_l991_99171

theorem negation_equivalence :
  (¬ (∃ x : ℝ, x < 2 ∧ x^2 - 2*x < 0)) ↔ (∀ x : ℝ, x < 2 → x^2 - 2*x ≥ 0) :=
by sorry

end negation_equivalence_l991_99171


namespace tree_age_at_height_l991_99118

/-- Represents the growth of a tree over time. -/
def tree_growth (initial_height : ℝ) (growth_rate : ℝ) (initial_age : ℝ) (years : ℝ) : ℝ :=
  initial_height + growth_rate * years

/-- Theorem stating the age of the tree when it reaches a specific height. -/
theorem tree_age_at_height (initial_height : ℝ) (growth_rate : ℝ) (initial_age : ℝ) (final_height : ℝ) :
  initial_height = 5 →
  growth_rate = 3 →
  initial_age = 1 →
  final_height = 23 →
  ∃ (years : ℝ), tree_growth initial_height growth_rate initial_age years = final_height ∧ initial_age + years = 7 :=
by
  sorry


end tree_age_at_height_l991_99118


namespace equation_solutions_l991_99198

-- Define the equation
def equation (x : ℝ) : Prop := x / 50 = Real.cos x

-- State the theorem
theorem equation_solutions :
  ∃! (s : Finset ℝ), s.card = 31 ∧ ∀ x, x ∈ s ↔ equation x :=
sorry

end equation_solutions_l991_99198


namespace hyperbola_iff_product_negative_l991_99175

/-- Definition of a hyperbola equation -/
def is_hyperbola_equation (m n : ℝ) : Prop :=
  ∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / m + (y t)^2 / n = 1 ∧
  (∃ t₁ t₂, (x t₁, y t₁) ≠ (x t₂, y t₂))

/-- The main theorem stating the condition for a hyperbola -/
theorem hyperbola_iff_product_negative (m n : ℝ) :
  is_hyperbola_equation m n ↔ m * n < 0 := by
  sorry

end hyperbola_iff_product_negative_l991_99175


namespace special_triples_count_l991_99192

/-- Represents a graph with a specific number of vertices and edges per vertex -/
structure Graph where
  numVertices : ℕ
  edgesPerVertex : ℕ

/-- Calculates the number of triples in a graph where each pair of vertices is either all connected or all disconnected -/
def countSpecialTriples (g : Graph) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem special_triples_count (g : Graph) (h1 : g.numVertices = 30) (h2 : g.edgesPerVertex = 6) :
  countSpecialTriples g = 1990 := by
  sorry

end special_triples_count_l991_99192


namespace bucket_weight_l991_99145

theorem bucket_weight (p q : ℝ) : ℝ :=
  let one_quarter_full := p
  let three_quarters_full := q
  let full_weight := -1/2 * p + 3/2 * q
  full_weight

#check bucket_weight

end bucket_weight_l991_99145


namespace wire_cut_ratio_l991_99131

theorem wire_cut_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  (4 * (a / 4) = 6 * (b / 6)) → a / b = 1 := by
sorry

end wire_cut_ratio_l991_99131


namespace min_distance_is_3420_div_181_l991_99184

/-- Triangle ABC with right angle at B, side lengths, and intersecting circles --/
structure RightTriangleWithCircles where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  ac : ℝ
  -- Angle condition
  right_angle : ab^2 + bc^2 = ac^2
  -- Side length values
  ab_eq : ab = 19
  bc_eq : bc = 180
  ac_eq : ac = 181
  -- Midpoints
  m : ℝ × ℝ  -- midpoint of AB
  n : ℝ × ℝ  -- midpoint of BC
  -- Intersection points
  d : ℝ × ℝ
  e : ℝ × ℝ
  p : ℝ × ℝ
  -- Conditions for D and E
  d_on_circle_m : (d.1 - m.1)^2 + (d.2 - m.2)^2 = (ac/2)^2
  d_on_circle_n : (d.1 - n.1)^2 + (d.2 - n.2)^2 = (ac/2)^2
  e_on_circle_m : (e.1 - m.1)^2 + (e.2 - m.2)^2 = (ac/2)^2
  e_on_circle_n : (e.1 - n.1)^2 + (e.2 - n.2)^2 = (ac/2)^2
  -- P is on AC
  p_on_ac : p.2 = 0
  -- DE intersects AC at P
  p_on_de : ∃ (t : ℝ), p = (1 - t) • d + t • e

/-- The minimum of DP and EP is 3420/181 --/
theorem min_distance_is_3420_div_181 (triangle : RightTriangleWithCircles) :
  min ((triangle.d.1 - triangle.p.1)^2 + (triangle.d.2 - triangle.p.2)^2)
      ((triangle.e.1 - triangle.p.1)^2 + (triangle.e.2 - triangle.p.2)^2) = (3420/181)^2 := by
  sorry

end min_distance_is_3420_div_181_l991_99184


namespace maria_chairs_l991_99152

/-- The number of chairs Maria bought -/
def num_chairs : ℕ := 2

/-- The number of tables Maria bought -/
def num_tables : ℕ := 2

/-- The time spent on each piece of furniture (in minutes) -/
def time_per_furniture : ℕ := 8

/-- The total time spent (in minutes) -/
def total_time : ℕ := 32

theorem maria_chairs :
  num_chairs * time_per_furniture + num_tables * time_per_furniture = total_time :=
by sorry

end maria_chairs_l991_99152


namespace solve_for_w_l991_99185

theorem solve_for_w (u v w : ℝ) 
  (eq1 : 10 * u + 8 * v + 5 * w = 160)
  (eq2 : v = u + 3)
  (eq3 : w = 2 * v) : 
  w = 13.5714 := by
  sorry

end solve_for_w_l991_99185


namespace min_product_of_three_numbers_l991_99122

theorem min_product_of_three_numbers (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 2 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y → 
  x * y * z ≥ 1/9 := by
sorry

end min_product_of_three_numbers_l991_99122


namespace cone_no_rectangular_front_view_l991_99120

-- Define the types of solids
inductive Solid
  | Cube
  | RegularTriangularPrism
  | Cylinder
  | Cone

-- Define a property for having a rectangular front view
def has_rectangular_front_view (s : Solid) : Prop :=
  match s with
  | Solid.Cube => True
  | Solid.RegularTriangularPrism => True
  | Solid.Cylinder => True
  | Solid.Cone => False

-- Theorem statement
theorem cone_no_rectangular_front_view :
  ∀ s : Solid, ¬(has_rectangular_front_view s) ↔ s = Solid.Cone :=
sorry

end cone_no_rectangular_front_view_l991_99120


namespace g_zero_value_l991_99146

-- Define polynomials f, g, and h
variable (f g h : ℝ[X])

-- Define the relationship between h, f, and g
axiom h_eq_f_mul_g : h = f * g

-- Define the constant term of f
axiom f_const_term : f.coeff 0 = 2

-- Define the constant term of h
axiom h_const_term : h.coeff 0 = -6

-- Theorem to prove
theorem g_zero_value : g.eval 0 = -3 := by sorry

end g_zero_value_l991_99146


namespace exists_valid_coloring_for_all_k_l991_99107

/-- A point on an infinite 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A set of black squares on an infinite white grid -/
def BlackSquares := Set GridPoint

/-- A line on the grid (vertical, horizontal, or diagonal) -/
structure GridLine where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The number of black squares on a given line -/
def blackSquaresOnLine (blacks : BlackSquares) (line : GridLine) : ℕ :=
  sorry

/-- A valid coloring of the grid for a given k -/
def validColoring (k : ℕ) (blacks : BlackSquares) : Prop :=
  (blacks.Nonempty) ∧
  (∀ line : GridLine, blackSquaresOnLine blacks line = k ∨ blackSquaresOnLine blacks line = 0)

/-- The main theorem: for any positive k, there exists a valid coloring -/
theorem exists_valid_coloring_for_all_k :
  ∀ k : ℕ, k > 0 → ∃ blacks : BlackSquares, validColoring k blacks :=
sorry

end exists_valid_coloring_for_all_k_l991_99107


namespace line_passes_through_second_and_fourth_quadrants_l991_99199

/-- A line with equation y = -2x + b (where b is a constant) always passes through the second and fourth quadrants. -/
theorem line_passes_through_second_and_fourth_quadrants (b : ℝ) :
  ∃ (x₁ x₂ y₁ y₂ : ℝ), 
    (x₁ < 0 ∧ y₁ > 0 ∧ y₁ = -2*x₁ + b) ∧ 
    (x₂ > 0 ∧ y₂ < 0 ∧ y₂ = -2*x₂ + b) :=
sorry

end line_passes_through_second_and_fourth_quadrants_l991_99199


namespace unit_circle_complex_bound_l991_99169

theorem unit_circle_complex_bound (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (zmin zmax : ℂ),
    Complex.abs (z^3 - 3*z - 2) ≤ Real.sqrt 27 ∧
    Complex.abs (zmax^3 - 3*zmax - 2) = Real.sqrt 27 ∧
    Complex.abs (zmin^3 - 3*zmin - 2) = 0 ∧
    Complex.abs zmax = 1 ∧
    Complex.abs zmin = 1 :=
by sorry

end unit_circle_complex_bound_l991_99169


namespace fifteenth_shape_black_tiles_l991_99117

/-- The dimension of the nth shape in the sequence -/
def shape_dimension (n : ℕ) : ℕ := 2 * n - 1

/-- The total number of tiles in the nth shape -/
def total_tiles (n : ℕ) : ℕ := (shape_dimension n) ^ 2

/-- The number of black tiles in the nth shape -/
def black_tiles (n : ℕ) : ℕ := (total_tiles n + 1) / 2

theorem fifteenth_shape_black_tiles :
  black_tiles 15 = 421 := by sorry

end fifteenth_shape_black_tiles_l991_99117


namespace vertex_of_quadratic_l991_99144

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * (x - 1)^2 - 3

-- Theorem stating that the vertex of the quadratic function is at (1, -3)
theorem vertex_of_quadratic :
  ∃ (x y : ℝ), x = 1 ∧ y = -3 ∧ ∀ (t : ℝ), f t ≤ f x :=
sorry

end vertex_of_quadratic_l991_99144


namespace sin_cos_identity_l991_99148

theorem sin_cos_identity :
  Real.sin (68 * π / 180) * Real.sin (67 * π / 180) - 
  Real.sin (23 * π / 180) * Real.cos (68 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end sin_cos_identity_l991_99148


namespace range_of_m_l991_99109

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 else -x^2

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Iic 1, f (x + m) ≤ -f x) → m ∈ Set.Ici (-2) := by
  sorry

end range_of_m_l991_99109


namespace b_oxen_count_l991_99150

/-- Represents the number of oxen-months for a person's contribution -/
def oxen_months (oxen : ℕ) (months : ℕ) : ℕ := oxen * months

/-- Represents the total rent of the pasture -/
def total_rent : ℕ := 175

/-- Represents A's contribution in oxen-months -/
def a_contribution : ℕ := oxen_months 10 7

/-- Represents C's contribution in oxen-months -/
def c_contribution : ℕ := oxen_months 15 3

/-- Represents C's share of the rent -/
def c_share : ℕ := 45

/-- Represents the number of months B's oxen grazed -/
def b_months : ℕ := 5

/-- Theorem stating that B put 12 oxen for grazing -/
theorem b_oxen_count : 
  ∃ (b_oxen : ℕ), 
    b_oxen = 12 ∧ 
    (c_share : ℚ) / total_rent = 
      (c_contribution : ℚ) / (a_contribution + oxen_months b_oxen b_months + c_contribution) :=
sorry

end b_oxen_count_l991_99150


namespace two_tangent_circles_l991_99137

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are tangent to each other -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Counts the number of circles with radius 4 that are tangent to both given circles -/
def count_tangent_circles (c1 c2 : Circle) : ℕ :=
  sorry

theorem two_tangent_circles 
  (c1 c2 : Circle) 
  (h1 : c1.radius = 2) 
  (h2 : c2.radius = 2) 
  (h3 : are_tangent c1 c2) :
  count_tangent_circles c1 c2 = 2 :=
sorry

end two_tangent_circles_l991_99137


namespace function_describes_relationship_l991_99125

-- Define the set of x values
def X : Set ℕ := {1, 2, 3, 4, 5}

-- Define the function f
def f (x : ℕ) : ℕ := x^2

-- Define the set of points (x, y)
def points : Set (ℕ × ℕ) := {(1, 1), (2, 4), (3, 9), (4, 16), (5, 25)}

-- Theorem statement
theorem function_describes_relationship :
  ∀ (x : ℕ), x ∈ X → (x, f x) ∈ points := by
  sorry

end function_describes_relationship_l991_99125


namespace f_min_value_l991_99172

/-- The function f(x) = |3-x| + |x-7| -/
def f (x : ℝ) := |3 - x| + |x - 7|

/-- The minimum value of f(x) is 4 -/
theorem f_min_value : ∀ x : ℝ, f x ≥ 4 ∧ ∃ y : ℝ, f y = 4 := by
  sorry

end f_min_value_l991_99172


namespace fruit_store_solution_l991_99189

/-- Represents the purchase quantities and costs of two types of fruits -/
structure FruitPurchase where
  quantityA : ℕ  -- Quantity of fruit A in kg
  quantityB : ℕ  -- Quantity of fruit B in kg
  totalCost : ℕ  -- Total cost in yuan

/-- The fruit store problem -/
def fruitStoreProblem (purchase1 purchase2 : FruitPurchase) : Prop :=
  ∃ (priceA priceB : ℕ),
    -- Conditions from the first purchase
    purchase1.quantityA * priceA + purchase1.quantityB * priceB = purchase1.totalCost ∧
    -- Conditions from the second purchase
    purchase2.quantityA * priceA + purchase2.quantityB * priceB = purchase2.totalCost ∧
    -- Unique solution condition
    ∀ (x y : ℕ),
      (purchase1.quantityA * x + purchase1.quantityB * y = purchase1.totalCost ∧
       purchase2.quantityA * x + purchase2.quantityB * y = purchase2.totalCost) →
      x = priceA ∧ y = priceB

/-- Theorem stating the solution to the fruit store problem -/
theorem fruit_store_solution :
  fruitStoreProblem
    { quantityA := 60, quantityB := 40, totalCost := 1520 }
    { quantityA := 30, quantityB := 50, totalCost := 1360 } →
  ∃ (priceA priceB : ℕ), priceA = 12 ∧ priceB = 20 := by
  sorry

end fruit_store_solution_l991_99189


namespace school_travel_speed_l991_99180

/-- Proves that the speed on the second day is 10 km/hr given the conditions of the problem -/
theorem school_travel_speed 
  (distance : ℝ) 
  (speed_day1 : ℝ) 
  (late_time : ℝ) 
  (early_time : ℝ) 
  (h1 : distance = 2.5) 
  (h2 : speed_day1 = 5) 
  (h3 : late_time = 7 / 60) 
  (h4 : early_time = 8 / 60) : 
  let correct_time := distance / speed_day1
  let actual_time_day2 := correct_time - late_time - early_time
  distance / actual_time_day2 = 10 := by
  sorry

end school_travel_speed_l991_99180


namespace range_of_z_l991_99138

theorem range_of_z (α β z : ℝ) 
  (h1 : -2 < α ∧ α ≤ 3) 
  (h2 : 2 < β ∧ β ≤ 4) 
  (h3 : z = 2*α - (1/2)*β) : 
  -6 < z ∧ z < 5 := by
  sorry

end range_of_z_l991_99138


namespace quadratic_inequality_solution_l991_99197

theorem quadratic_inequality_solution (x : ℝ) :
  (2 * x - 1) * (3 * x + 1) > 0 ↔ x < -1/3 ∨ x > 1/2 := by sorry

end quadratic_inequality_solution_l991_99197


namespace data_center_connections_l991_99156

theorem data_center_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end data_center_connections_l991_99156


namespace flowers_per_pot_l991_99170

theorem flowers_per_pot (num_gardens : ℕ) (pots_per_garden : ℕ) (total_flowers : ℕ) : 
  num_gardens = 10 →
  pots_per_garden = 544 →
  total_flowers = 174080 →
  total_flowers / (num_gardens * pots_per_garden) = 32 := by
  sorry

end flowers_per_pot_l991_99170


namespace butterfat_mixture_proof_l991_99181

/-- Proves that mixing 8 gallons of 35% butterfat milk with 12 gallons of 10% butterfat milk
    results in a mixture that is 20% butterfat. -/
theorem butterfat_mixture_proof :
  let x : ℝ := 8 -- Amount of 35% butterfat milk in gallons
  let y : ℝ := 12 -- Amount of 10% butterfat milk in gallons
  let butterfat_high : ℝ := 0.35 -- Percentage of butterfat in high-fat milk
  let butterfat_low : ℝ := 0.10 -- Percentage of butterfat in low-fat milk
  let butterfat_target : ℝ := 0.20 -- Target percentage of butterfat in mixture
  (butterfat_high * x + butterfat_low * y) / (x + y) = butterfat_target :=
by sorry

end butterfat_mixture_proof_l991_99181


namespace melanie_dimes_count_l991_99102

/-- The total number of dimes Melanie has after receiving dimes from her parents -/
def total_dimes (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Theorem stating that Melanie's total dimes is the sum of her initial dimes and those received from her parents -/
theorem melanie_dimes_count (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) :
  total_dimes initial from_dad from_mom = initial + from_dad + from_mom :=
by
  sorry

#eval total_dimes 19 39 25

end melanie_dimes_count_l991_99102


namespace expression_factorization_l991_99147

theorem expression_factorization (a : ℝ) : 
  (10 * a^4 - 160 * a^3 - 32) - (-2 * a^4 - 16 * a^3 + 32) = 4 * (3 * a^3 * (a - 12) - 16) := by
  sorry

end expression_factorization_l991_99147


namespace circle_center_is_3_0_l991_99108

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in standard form -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- The given circle equation -/
def given_circle_equation (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 9

theorem circle_center_is_3_0 :
  ∃ (c : Circle), (∀ x y : ℝ, circle_equation c x y ↔ given_circle_equation x y) ∧ c.center = (3, 0) := by
  sorry

end circle_center_is_3_0_l991_99108


namespace standing_men_ratio_l991_99167

theorem standing_men_ratio (total_passengers : ℕ) (seated_men : ℕ) : 
  total_passengers = 48 →
  seated_men = 14 →
  (standing_men : ℚ) / (total_men : ℚ) = 1 / 8 :=
by
  intros h_total h_seated
  sorry
where
  women := (2 : ℚ) / 3 * total_passengers
  total_men := total_passengers - women
  standing_men := total_men - seated_men

end standing_men_ratio_l991_99167


namespace complex_magnitude_real_part_l991_99103

theorem complex_magnitude_real_part (t : ℝ) : 
  t > 0 → Complex.abs (9 + t * Complex.I) = 15 → Complex.re (9 + t * Complex.I) = 9 → t = 12 := by
  sorry

end complex_magnitude_real_part_l991_99103


namespace first_fund_profit_percentage_l991_99111

/-- Proves that the profit percentage of the first mutual fund is approximately 2.82% given the specified conditions --/
theorem first_fund_profit_percentage 
  (total_investment : ℝ) 
  (investment_higher_profit : ℝ) 
  (second_fund_profit : ℝ) 
  (total_profit : ℝ) 
  (h1 : total_investment = 1900)
  (h2 : investment_higher_profit = 1700)
  (h3 : second_fund_profit = 0.02)
  (h4 : total_profit = 52)
  : ∃ (first_fund_profit : ℝ), 
    (first_fund_profit * investment_higher_profit + 
     second_fund_profit * (total_investment - investment_higher_profit) = total_profit) ∧
    (abs (first_fund_profit - 0.0282) < 0.0001) :=
by sorry

end first_fund_profit_percentage_l991_99111


namespace complex_equation_solution_l991_99160

theorem complex_equation_solution (z a b : ℂ) : 
  z = (1 - Complex.I)^2 + 1 + 3 * Complex.I →
  z^2 + a * z + b = 1 - Complex.I →
  a.im = 0 →
  b.im = 0 →
  a = -2 ∧ b = 4 := by
  sorry

end complex_equation_solution_l991_99160


namespace archibald_win_percentage_l991_99164

/-- Given that Archibald has won 12 games and his brother has won 18 games,
    prove that the percentage of games Archibald has won is 40%. -/
theorem archibald_win_percentage 
  (archibald_wins : ℕ) 
  (brother_wins : ℕ) 
  (h1 : archibald_wins = 12)
  (h2 : brother_wins = 18) :
  (archibald_wins : ℚ) / (archibald_wins + brother_wins) * 100 = 40 := by
  sorry

end archibald_win_percentage_l991_99164


namespace child_ticket_price_l991_99157

theorem child_ticket_price
  (adult_price : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (child_tickets : ℕ)
  (h1 : adult_price = 7)
  (h2 : total_tickets = 900)
  (h3 : total_revenue = 5100)
  (h4 : child_tickets = 400)
  : ∃ (child_price : ℕ),
    child_price * child_tickets + adult_price * (total_tickets - child_tickets) = total_revenue ∧
    child_price = 4 :=
by sorry

end child_ticket_price_l991_99157


namespace inequality_iff_solution_set_l991_99101

def inequality (x : ℝ) : Prop :=
  2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 20

def solution_set (x : ℝ) : Prop :=
  x < -3 ∨ (-2 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 7) ∨ 8 < x

theorem inequality_iff_solution_set :
  ∀ x : ℝ, inequality x ↔ solution_set x :=
by sorry

end inequality_iff_solution_set_l991_99101


namespace sum_remainder_mod_8_l991_99196

theorem sum_remainder_mod_8 : (7150 + 7151 + 7152 + 7153 + 7154 + 7155) % 8 = 3 := by
  sorry

end sum_remainder_mod_8_l991_99196


namespace lottery_tickets_theorem_lottery_tickets_minimality_l991_99195

/-- The probability of winning with a single lottery ticket -/
def p : ℝ := 0.01

/-- The desired probability of winning at least once -/
def desired_prob : ℝ := 0.95

/-- The number of tickets needed to achieve the desired probability -/
def n : ℕ := 300

/-- Theorem stating that n tickets are sufficient to achieve the desired probability -/
theorem lottery_tickets_theorem :
  1 - (1 - p) ^ n ≥ desired_prob :=
sorry

/-- Theorem stating that n-1 tickets are not sufficient to achieve the desired probability -/
theorem lottery_tickets_minimality :
  1 - (1 - p) ^ (n - 1) < desired_prob :=
sorry

end lottery_tickets_theorem_lottery_tickets_minimality_l991_99195


namespace mixture_weight_l991_99165

/-- Proves that the total weight of a mixture of cashews and peanuts is 29.5 pounds
    given specific prices and constraints. -/
theorem mixture_weight (cashew_price peanut_price total_cost cashew_weight : ℝ) 
  (h1 : cashew_price = 5)
  (h2 : peanut_price = 2)
  (h3 : total_cost = 92)
  (h4 : cashew_weight = 11) : 
  cashew_weight + (total_cost - cashew_price * cashew_weight) / peanut_price = 29.5 := by
  sorry

#check mixture_weight

end mixture_weight_l991_99165


namespace similar_triangles_side_length_l991_99177

/-- Represents a triangle with an area and a side length -/
structure Triangle where
  area : ℕ
  side : ℕ

/-- The theorem statement -/
theorem similar_triangles_side_length 
  (small large : Triangle)
  (h_diff : large.area - small.area = 32)
  (h_ratio : ∃ k : ℕ, large.area = k^2 * small.area)
  (h_small_side : small.side = 4) :
  large.side = 12 := by
  sorry

end similar_triangles_side_length_l991_99177


namespace inequality_proof_l991_99162

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 + (a + b + c)^2 ≤ 4) :
  (a*b + 1) / (a + b)^2 + (b*c + 1) / (b + c)^2 + (c*a + 1) / (c + a)^2 ≥ 3 :=
by sorry

end inequality_proof_l991_99162


namespace tan_678_degrees_equals_138_l991_99127

theorem tan_678_degrees_equals_138 :
  ∃ (n : ℤ), -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (678 * π / 180) ∧ n = 138 := by
  sorry

end tan_678_degrees_equals_138_l991_99127


namespace population_problem_l991_99110

theorem population_problem : ∃ (n : ℕ), 
  (∃ (m k : ℕ), 
    (n^2 + 200 = m^2 + 1) ∧ 
    (n^2 + 500 = k^2) ∧ 
    (21 ∣ n^2) ∧ 
    (n^2 = 9801)) := by
  sorry

end population_problem_l991_99110


namespace meeting_percentage_is_37_5_percent_l991_99168

def total_work_hours : ℕ := 8
def first_meeting_duration : ℕ := 30
def minutes_per_hour : ℕ := 60

def total_work_minutes : ℕ := total_work_hours * minutes_per_hour
def second_meeting_duration : ℕ := 2 * first_meeting_duration
def third_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration
def total_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration

theorem meeting_percentage_is_37_5_percent :
  (total_meeting_duration : ℚ) / (total_work_minutes : ℚ) * 100 = 37.5 := by
  sorry

end meeting_percentage_is_37_5_percent_l991_99168


namespace product_of_square_roots_l991_99149

theorem product_of_square_roots (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p^3) * Real.sqrt (20 * p) * Real.sqrt (8 * p^5) = 20 * p^4 * Real.sqrt (6 * p) := by
  sorry

end product_of_square_roots_l991_99149


namespace iggy_ran_four_miles_on_tuesday_l991_99173

/-- Represents the days of the week Iggy runs --/
inductive RunDay
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Represents Iggy's running data --/
structure RunningData where
  miles : RunDay → ℕ
  pace : ℕ  -- minutes per mile
  totalTime : ℕ  -- total running time in minutes

/-- Theorem stating that Iggy ran 4 miles on Tuesday --/
theorem iggy_ran_four_miles_on_tuesday (data : RunningData) : data.miles RunDay.Tuesday = 4 :=
  by
  have h1 : data.miles RunDay.Monday = 3 := by sorry
  have h2 : data.miles RunDay.Wednesday = 6 := by sorry
  have h3 : data.miles RunDay.Thursday = 8 := by sorry
  have h4 : data.miles RunDay.Friday = 3 := by sorry
  have h5 : data.pace = 10 := by sorry
  have h6 : data.totalTime = 4 * 60 := by sorry
  
  sorry


end iggy_ran_four_miles_on_tuesday_l991_99173


namespace perpendicular_vectors_m_value_l991_99106

/-- Given two vectors a and b in ℝ², prove that if (a - b) is perpendicular
    to (m * a + b), then m = 1/4. -/
theorem perpendicular_vectors_m_value 
  (a b : ℝ × ℝ) 
  (ha : a = (2, 1)) 
  (hb : b = (1, -1)) 
  (h_perp : (a.1 - b.1) * (m * a.1 + b.1) + (a.2 - b.2) * (m * a.2 + b.2) = 0) : 
  m = 1/4 := by
  sorry

end perpendicular_vectors_m_value_l991_99106


namespace complex_roots_count_l991_99143

theorem complex_roots_count (z : ℂ) : 
  let θ := Complex.arg z
  (Complex.abs z = 1) →
  ((z ^ (7 * 6 * 5 * 4 * 3 * 2 * 1) - z ^ (6 * 5 * 4 * 3 * 2 * 1)).im = 0) →
  ((z ^ (6 * 5 * 4 * 3 * 2 * 1) - z ^ (5 * 4 * 3 * 2 * 1)).im = 0) →
  (0 ≤ θ) →
  (θ < 2 * Real.pi) →
  (Real.cos (4320 * θ) = 0 ∨ Real.sin (3360 * θ) = 0) →
  (Real.cos (420 * θ) = 0 ∨ Real.sin (300 * θ) = 0) →
  Nat := by sorry

end complex_roots_count_l991_99143


namespace fraction_equality_l991_99153

theorem fraction_equality (a b c d e : ℚ) 
  (h1 : a / b = 1 / 2)
  (h2 : c / d = 1 / 2)
  (h3 : e / 5 = 1 / 2)
  (h4 : b + d + 5 ≠ 0) :
  (a + c + e) / (b + d + 5) = 1 / 2 := by
sorry

end fraction_equality_l991_99153


namespace exists_multiple_indecomposable_factorizations_l991_99100

/-- The set V_n for a given positive integer n -/
def V_n (n : ℕ) : Set ℕ := {m : ℕ | ∃ k : ℕ+, m = 1 + k * n}

/-- A number is indecomposable in V_n if it cannot be expressed as a product of two numbers in V_n -/
def Indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ p q : ℕ, p ∈ V_n n → q ∈ V_n n → p * q ≠ m

/-- Main theorem: There exists a number in V_n that can be expressed as a product of
    indecomposable numbers in V_n in more than one way -/
theorem exists_multiple_indecomposable_factorizations (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ V_n n ∧
    ∃ (a b c d : ℕ) (ha : Indecomposable n a) (hb : Indecomposable n b)
      (hc : Indecomposable n c) (hd : Indecomposable n d),
    r = a * b ∧ r = c * d ∧ (a ≠ c ∨ b ≠ d) :=
  sorry

end exists_multiple_indecomposable_factorizations_l991_99100


namespace max_product_difference_l991_99191

theorem max_product_difference (x y : ℕ) : 
  x > 0 → y > 0 → x + 2 * y = 2008 → (∀ a b : ℕ, a > 0 → b > 0 → a + 2 * b = 2008 → x * y ≥ a * b) → x - y = 502 := by
sorry

end max_product_difference_l991_99191


namespace same_color_probability_l991_99158

/-- The probability of drawing two balls of the same color from a bag containing 3 white balls
    and 2 black balls when 2 balls are randomly drawn at the same time. -/
theorem same_color_probability (total : ℕ) (white : ℕ) (black : ℕ) :
  total = 5 →
  white = 3 →
  black = 2 →
  (Nat.choose white 2 + Nat.choose black 2) / Nat.choose total 2 = 2 / 5 := by
  sorry

#eval Nat.choose 5 2  -- Total number of ways to draw 2 balls from 5
#eval Nat.choose 3 2  -- Number of ways to draw 2 white balls
#eval Nat.choose 2 2  -- Number of ways to draw 2 black balls

end same_color_probability_l991_99158


namespace square_value_l991_99141

theorem square_value (x : ℚ) : 
  10 + 9 + 8 * 7 / x + 6 - 5 * 4 - 3 * 2 = 1 → x = 28 := by
sorry

end square_value_l991_99141
