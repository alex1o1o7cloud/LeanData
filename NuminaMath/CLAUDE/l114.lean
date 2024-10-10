import Mathlib

namespace sum_of_decimals_l114_11457

theorem sum_of_decimals : (7.46 : ℝ) + 4.29 = 11.75 := by
  sorry

end sum_of_decimals_l114_11457


namespace voter_percentage_for_candidate_A_l114_11467

theorem voter_percentage_for_candidate_A
  (total_voters : ℝ)
  (democrat_percentage : ℝ)
  (democrat_support_A : ℝ)
  (republican_support_A : ℝ)
  (h1 : democrat_percentage = 0.6)
  (h2 : democrat_support_A = 0.7)
  (h3 : republican_support_A = 0.2)
  (h4 : total_voters > 0) :
  let republican_percentage := 1 - democrat_percentage
  let voters_for_A := total_voters * (democrat_percentage * democrat_support_A + republican_percentage * republican_support_A)
  voters_for_A / total_voters = 0.5 := by
sorry

end voter_percentage_for_candidate_A_l114_11467


namespace cookie_cost_l114_11433

theorem cookie_cost (initial_amount : ℚ) (hat_cost : ℚ) (pencil_cost : ℚ) (num_cookies : ℕ) (remaining_amount : ℚ)
  (h1 : initial_amount = 20)
  (h2 : hat_cost = 10)
  (h3 : pencil_cost = 2)
  (h4 : num_cookies = 4)
  (h5 : remaining_amount = 3)
  : (initial_amount - hat_cost - pencil_cost - remaining_amount) / num_cookies = 5/4 := by
  sorry

end cookie_cost_l114_11433


namespace parallelogram_base_proof_l114_11446

/-- The area of a parallelogram -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_base_proof (area height : ℝ) (h1 : area = 96) (h2 : height = 8) :
  parallelogram_area (area / height) height = area → area / height = 12 := by
sorry

end parallelogram_base_proof_l114_11446


namespace factors_of_36_l114_11477

def number : ℕ := 36

-- Sum of positive factors
def sum_of_factors (n : ℕ) : ℕ := sorry

-- Product of prime factors
def product_of_prime_factors (n : ℕ) : ℕ := sorry

theorem factors_of_36 :
  sum_of_factors number = 91 ∧ product_of_prime_factors number = 6 := by sorry

end factors_of_36_l114_11477


namespace quadratic_roots_condition_l114_11472

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - m*x₁ - 1 = 0 ∧ x₂^2 - m*x₂ - 1 = 0 ∧ x₁ > 2 ∧ x₂ < 2) ↔ 
  m > 3/2 :=
sorry

end quadratic_roots_condition_l114_11472


namespace solution_set_implies_a_value_l114_11465

def f (x a : ℝ) : ℝ := |2 * x - a| + a

theorem solution_set_implies_a_value :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ f x 1 ≤ 6) →
  (∀ x : ℝ, f x 1 ≤ 6 → -2 ≤ x ∧ x ≤ 3) →
  (∃ a : ℝ, ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ f x a ≤ 6) →
  (∃! a : ℝ, ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ f x a ≤ 6) :=
by sorry

end solution_set_implies_a_value_l114_11465


namespace simplify_expression1_simplify_expression2_l114_11408

-- Define the expressions
def expression1 (a : ℝ) : ℝ := 5 * a^2 - 7 + 4 * a - 2 * a^2 - 9 * a + 3
def expression2 (x : ℝ) : ℝ := (5 * x^2 - 6 * x) - 3 * (2 * x^2 - 3 * x)

-- State the theorems
theorem simplify_expression1 : ∀ a : ℝ, expression1 a = 3 * a^2 - 5 * a - 4 := by sorry

theorem simplify_expression2 : ∀ x : ℝ, expression2 x = -x^2 + 3 * x := by sorry

end simplify_expression1_simplify_expression2_l114_11408


namespace count_grid_paths_l114_11460

/-- The number of paths from (0,0) to (m, n) on a grid, moving only right or up by one unit at a time -/
def gridPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem stating that the number of paths from (0,0) to (m, n) on a grid,
    moving only right or up by one unit at a time, is equal to (m+n choose m) -/
theorem count_grid_paths (m n : ℕ) : 
  gridPaths m n = Nat.choose (m + n) m := by
  sorry

end count_grid_paths_l114_11460


namespace alpha_range_l114_11418

open Real

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) + 2 * sin x

theorem alpha_range (α : ℝ) :
  α > 0 ∧
  (∀ x, x ∈ Set.Icc 0 α → f x ∈ Set.Icc 1 (3/2)) ∧
  (∃ x₁ x₂, x₁ ∈ Set.Icc 0 α ∧ x₂ ∈ Set.Icc 0 α ∧ f x₁ = 1 ∧ f x₂ = 3/2) →
  α ∈ Set.Icc (π/6) π :=
sorry

end alpha_range_l114_11418


namespace snakes_count_l114_11461

theorem snakes_count (breeding_balls : ℕ) (snakes_per_ball : ℕ) (snake_pairs : ℕ) : 
  breeding_balls * snakes_per_ball + 2 * snake_pairs = 36 :=
by
  sorry

#check snakes_count 3 8 6

end snakes_count_l114_11461


namespace parallelogram_probability_l114_11425

-- Define the vertices of the parallelogram
def P : ℝ × ℝ := (4, 4)
def Q : ℝ × ℝ := (-2, -2)
def R : ℝ × ℝ := (-8, -2)
def S : ℝ × ℝ := (-2, 4)

-- Define the line y = -1
def line (x : ℝ) : ℝ := -1

-- Define the area of a parallelogram given base and height
def parallelogram_area (base height : ℝ) : ℝ := base * height

-- Theorem statement
theorem parallelogram_probability : 
  let total_area := parallelogram_area (P.1 - S.1) (P.2 - Q.2)
  let below_line_area := parallelogram_area (P.1 - S.1) 1
  below_line_area / total_area = 1 / 6 := by sorry

end parallelogram_probability_l114_11425


namespace sum_of_squares_inequality_l114_11470

theorem sum_of_squares_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end sum_of_squares_inequality_l114_11470


namespace max_triangle_area_l114_11498

/-- The maximum area of a triangle with constrained side lengths -/
theorem max_triangle_area (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 1 ≤ b ∧ b ≤ 2) (hc : 2 ≤ c ∧ c ≤ 3)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (S : ℝ), S ≤ 1 ∧ ∀ (S' : ℝ), (∃ (a' b' c' : ℝ),
    0 ≤ a' ∧ a' ≤ 1 ∧
    1 ≤ b' ∧ b' ≤ 2 ∧
    2 ≤ c' ∧ c' ≤ 3 ∧
    a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' ∧
    S' = (a' * b' * Real.sqrt (1 - (a'*a' + b'*b' - c'*c')^2 / (4*a'*a'*b'*b'))) / 2) →
    S' ≤ S :=
by
  sorry

end max_triangle_area_l114_11498


namespace number_equals_scientific_notation_l114_11455

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be represented in scientific notation -/
def number : ℕ := 858000000

/-- The scientific notation representation of the number -/
def scientific_representation : ScientificNotation := {
  coefficient := 8.58
  exponent := 8
  is_valid := by sorry
}

/-- Theorem stating that the number is equal to its scientific notation representation -/
theorem number_equals_scientific_notation : 
  (scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent) = number := by
  sorry

end number_equals_scientific_notation_l114_11455


namespace max_point_difference_is_n_l114_11406

/-- Represents a hockey tournament with n teams -/
structure HockeyTournament where
  n : ℕ  -- number of teams
  n_pos : 0 < n  -- n is positive

/-- The maximum point difference between consecutively ranked teams in a hockey tournament -/
def maxPointDifference (tournament : HockeyTournament) : ℕ :=
  tournament.n

/-- Theorem: The maximum point difference between consecutively ranked teams is n -/
theorem max_point_difference_is_n (tournament : HockeyTournament) :
  maxPointDifference tournament = tournament.n := by
  sorry

end max_point_difference_is_n_l114_11406


namespace reena_loan_interest_l114_11464

/-- Calculate simple interest for a loan where the loan period in years equals the interest rate -/
def simple_interest (principal : ℚ) (rate : ℚ) : ℚ :=
  principal * rate * rate / 100

theorem reena_loan_interest :
  let principal : ℚ := 1200
  let rate : ℚ := 4
  simple_interest principal rate = 192 := by
sorry

end reena_loan_interest_l114_11464


namespace range_of_a_proposition_holds_l114_11468

/-- The proposition that the inequality ax^2 - 2ax - 3 ≥ 0 does not hold for all real x -/
def proposition (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬(a * x^2 - 2 * a * x - 3 ≥ 0)

/-- The theorem stating that if the proposition holds, then a is in the range (-3, 0] -/
theorem range_of_a (a : ℝ) (h : proposition a) : -3 < a ∧ a ≤ 0 := by
  sorry

/-- The theorem stating that if a is in the range (-3, 0], then the proposition holds -/
theorem proposition_holds (a : ℝ) (h : -3 < a ∧ a ≤ 0) : proposition a := by
  sorry

end range_of_a_proposition_holds_l114_11468


namespace cut_cube_volume_l114_11427

/-- A polyhedron formed by cutting off the eight corners of a cube -/
structure CutCube where
  /-- The polyhedron has 6 octagonal faces -/
  octagonal_faces : Nat
  /-- The polyhedron has 8 triangular faces -/
  triangular_faces : Nat
  /-- All edges of the polyhedron have length 2 -/
  edge_length : ℝ

/-- The volume of the CutCube -/
def volume (c : CutCube) : ℝ := sorry

/-- Theorem stating the volume of the CutCube -/
theorem cut_cube_volume (c : CutCube) 
  (h1 : c.octagonal_faces = 6)
  (h2 : c.triangular_faces = 8)
  (h3 : c.edge_length = 2) :
  volume c = 56 + 112 * Real.sqrt 2 / 3 := by sorry

end cut_cube_volume_l114_11427


namespace function_relationship_l114_11495

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h1 : ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y)
variable (h2 : ∀ x, f (x + 2) = f (-x + 2))

-- State the theorem
theorem function_relationship :
  f (7/2) < f 1 ∧ f 1 < f (5/2) := by sorry

end function_relationship_l114_11495


namespace earth_livable_fraction_l114_11478

/-- The fraction of the earth's surface not covered by water -/
def land_fraction : ℚ := 1/3

/-- The fraction of exposed land that is inhabitable -/
def inhabitable_fraction : ℚ := 1/3

/-- The fraction of the earth's surface that humans can live on -/
def livable_fraction : ℚ := land_fraction * inhabitable_fraction

theorem earth_livable_fraction :
  livable_fraction = 1/9 := by sorry

end earth_livable_fraction_l114_11478


namespace problem_solution_l114_11491

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 2|

-- Theorem statement
theorem problem_solution :
  (∃ (M : ℝ), (∀ x, f x ≥ M) ∧ (∃ x, f x = M) ∧ M = 3) ∧
  ({x : ℝ | f x < 3 + |2*x + 2|} = Set.Ioo (-1) 2) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 2*b^2 = 3 → 2*a + b ≤ 3*Real.sqrt 6 / 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + 2*b^2 = 3 ∧ 2*a + b = 3*Real.sqrt 6 / 2) :=
by
  sorry


end problem_solution_l114_11491


namespace inequality_solution_range_l114_11409

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) ↔ (a < 1 ∨ a > 3) :=
sorry

end inequality_solution_range_l114_11409


namespace polygon_problem_l114_11481

theorem polygon_problem :
  ∀ (a b n d : ℤ),
    a > 0 →
    a^2 - 1 = 123 * 125 →
    (x^3 - 16*x^2 - 9*x + a) % (x - 2) = b →
    (n * (n - 3)) / 2 = b + 4 →
    (1 - n) / 2 = (d - 1) / 2 →
    a = 124 ∧ b = 50 ∧ n = 12 ∧ d = -10 := by
  sorry

end polygon_problem_l114_11481


namespace adam_nuts_purchase_l114_11469

theorem adam_nuts_purchase (nuts_price dried_fruits_price dried_fruits_weight total_cost : ℝ) 
  (h1 : nuts_price = 12)
  (h2 : dried_fruits_price = 8)
  (h3 : dried_fruits_weight = 2.5)
  (h4 : total_cost = 56) :
  ∃ (nuts_weight : ℝ), 
    nuts_weight * nuts_price + dried_fruits_weight * dried_fruits_price = total_cost ∧ 
    nuts_weight = 3 := by
sorry


end adam_nuts_purchase_l114_11469


namespace y_intercept_of_parallel_line_l114_11402

/-- A line in the xy-plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Returns true if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Returns true if a point (x, y) is on the given line -/
def on_line (l : Line) (x y : ℝ) : Prop := y = l.slope * x + l.y_intercept

theorem y_intercept_of_parallel_line 
  (line1 : Line) 
  (hline1 : line1.slope = -3 ∧ line1.y_intercept = 6) 
  (line2 : Line)
  (hparallel : parallel line1 line2)
  (hon_line : on_line line2 3 1) : 
  line2.y_intercept = 10 := by
  sorry

end y_intercept_of_parallel_line_l114_11402


namespace average_car_selections_l114_11485

theorem average_car_selections (num_cars : ℕ) (num_clients : ℕ) (selections_per_client : ℕ) : 
  num_cars = 15 → num_clients = 15 → selections_per_client = 3 →
  (num_clients * selections_per_client) / num_cars = 3 := by
  sorry

end average_car_selections_l114_11485


namespace prop_3_prop_4_l114_11490

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (skew : Line → Line → Prop)

-- Define the lines and planes
variable (m n l : Line)
variable (α : Plane)

-- State the theorems
theorem prop_3 (h1 : parallel m n) (h2 : perpendicular_plane m α) :
  perpendicular_plane n α := by sorry

theorem prop_4 (h1 : skew m n) (h2 : parallel_plane m α) (h3 : parallel_plane n α)
  (h4 : perpendicular m l) (h5 : perpendicular n l) :
  perpendicular_plane l α := by sorry

end prop_3_prop_4_l114_11490


namespace five_cubed_sum_equals_five_to_fourth_l114_11432

theorem five_cubed_sum_equals_five_to_fourth : 5^3 + 5^3 + 5^3 + 5^3 + 5^3 = 5^4 := by
  sorry

end five_cubed_sum_equals_five_to_fourth_l114_11432


namespace truncated_pyramid_ratio_l114_11479

/-- Given a right prism with a square base of side length L₁ and height H, 
    and a truncated pyramid extracted from it with square bases of side lengths 
    L₁ (bottom) and L₂ (top) and height H, if the volume of the truncated pyramid 
    is 2/3 of the total volume of the prism, then L₁/L₂ = (1 + √5) / 2. -/
theorem truncated_pyramid_ratio (L₁ L₂ H : ℝ) (h₁ : L₁ > 0) (h₂ : L₂ > 0) (h₃ : H > 0) :
  (H / 3 * (L₁^2 + L₁*L₂ + L₂^2) = 2/3 * H * L₁^2) → L₁ / L₂ = (1 + Real.sqrt 5) / 2 :=
by sorry

end truncated_pyramid_ratio_l114_11479


namespace marching_band_weight_theorem_l114_11473

/-- Represents the weight carried by each instrument player in the marching band --/
structure BandWeights where
  trumpet_clarinet : ℕ
  trombone : ℕ
  tuba : ℕ
  drum : ℕ

/-- Represents the number of players for each instrument in the marching band --/
structure BandComposition where
  trumpets : ℕ
  clarinets : ℕ
  trombones : ℕ
  tubas : ℕ
  drummers : ℕ

/-- Calculates the total weight carried by the marching band --/
def total_weight (weights : BandWeights) (composition : BandComposition) : ℕ :=
  (weights.trumpet_clarinet * (composition.trumpets + composition.clarinets)) +
  (weights.trombone * composition.trombones) +
  (weights.tuba * composition.tubas) +
  (weights.drum * composition.drummers)

theorem marching_band_weight_theorem (weights : BandWeights) (composition : BandComposition) :
  weights.trombone = 10 →
  weights.tuba = 20 →
  weights.drum = 15 →
  composition.trumpets = 6 →
  composition.clarinets = 9 →
  composition.trombones = 8 →
  composition.tubas = 3 →
  composition.drummers = 2 →
  total_weight weights composition = 245 →
  weights.trumpet_clarinet = 5 := by
  sorry

end marching_band_weight_theorem_l114_11473


namespace negation_equivalence_l114_11407

theorem negation_equivalence :
  (¬ (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → x ≠ 0 ∨ y ≠ 0) :=
by sorry

end negation_equivalence_l114_11407


namespace expected_total_rainfall_l114_11442

/-- Represents the weather conditions for a single day --/
structure WeatherCondition where
  sun_prob : ℝ
  light_rain_prob : ℝ
  heavy_rain_prob : ℝ
  light_rain_amount : ℝ
  heavy_rain_amount : ℝ

/-- Calculates the expected rain amount for a single day --/
def expected_rain_per_day (w : WeatherCondition) : ℝ :=
  w.light_rain_prob * w.light_rain_amount + w.heavy_rain_prob * w.heavy_rain_amount

/-- The number of days in the forecast --/
def forecast_days : ℕ := 6

/-- The weather condition for each day in the forecast --/
def daily_weather : WeatherCondition :=
  { sun_prob := 0.3,
    light_rain_prob := 0.3,
    heavy_rain_prob := 0.4,
    light_rain_amount := 5,
    heavy_rain_amount := 12 }

/-- Theorem: The expected total rainfall over the forecast period is 37.8 inches --/
theorem expected_total_rainfall :
  (forecast_days : ℝ) * expected_rain_per_day daily_weather = 37.8 := by
  sorry

end expected_total_rainfall_l114_11442


namespace sum_m_n_equals_three_l114_11459

theorem sum_m_n_equals_three (m n : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : m / (1 + i) = 1 - n * i) : m + n = 3 := by
  sorry

end sum_m_n_equals_three_l114_11459


namespace relatively_prime_powers_l114_11416

theorem relatively_prime_powers (a n m : ℕ) :
  Odd a → n > 0 → m > 0 → n ≠ m →
  Nat.gcd (a^(2^n) + 2^(2^n)) (a^(2^m) + 2^(2^m)) = 1 := by
  sorry

end relatively_prime_powers_l114_11416


namespace jellybean_count_l114_11466

theorem jellybean_count (initial_count : ℕ) : 
  (initial_count : ℝ) * (0.7 ^ 3) = 28 → initial_count = 82 := by
  sorry

end jellybean_count_l114_11466


namespace constant_ratio_problem_l114_11456

theorem constant_ratio_problem (x₁ x₂ : ℝ) (y₁ y₂ : ℝ) (k : ℝ) :
  (3 * x₁ - 4) / (y₁ + 15) = k →
  (3 * x₂ - 4) / (y₂ + 15) = k →
  x₁ = 2 →
  y₁ = 3 →
  y₂ = 12 →
  x₂ = 7 / 3 := by
sorry

end constant_ratio_problem_l114_11456


namespace dessert_preference_l114_11475

theorem dessert_preference (total : ℕ) (apple : ℕ) (chocolate : ℕ) (neither : ℕ) :
  total = 40 →
  apple = 18 →
  chocolate = 15 →
  neither = 12 →
  ∃ (both : ℕ), both = 5 ∧ total = apple + chocolate - both + neither :=
by
  sorry

end dessert_preference_l114_11475


namespace coefficient_x_cubed_in_expansion_l114_11453

theorem coefficient_x_cubed_in_expansion :
  let n : ℕ := 20
  let k : ℕ := 3
  let a : ℤ := 2
  let b : ℤ := -3
  (n.choose k) * a^k * b^(n-k) = -1174898049840 :=
by sorry

end coefficient_x_cubed_in_expansion_l114_11453


namespace square_root_of_nine_l114_11499

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end square_root_of_nine_l114_11499


namespace diagonals_perpendicular_l114_11474

-- Define a cube
structure Cube where
  -- Add necessary properties of a cube
  is_cube : Bool

-- Define the angle between diagonals of adjacent faces
def angle_between_diagonals (c : Cube) : ℝ :=
  sorry

-- Theorem statement
theorem diagonals_perpendicular (c : Cube) :
  angle_between_diagonals c = 90 :=
sorry

end diagonals_perpendicular_l114_11474


namespace girls_examined_l114_11480

theorem girls_examined (boys : ℕ) (girls : ℕ) 
  (h1 : boys = 50)
  (h2 : (25 : ℝ) + 0.6 * girls = 0.5667 * (boys + girls)) :
  girls = 100 := by
  sorry

end girls_examined_l114_11480


namespace problem_statement_l114_11441

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b + 2 * a + b = 16) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y + 2 * x + y = 16 ∧ x * y > a * b) →
    a * b ≤ 8 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y + 2 * x + y = 16 → 2 * x + y ≥ 8) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y + 2 * x + y = 16 → x + y ≥ 6 * Real.sqrt 2 - 3) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x * y + 2 * x + y = 16 → 1 / (x + 1) + 1 / (y + 2) > Real.sqrt 2 / 2) :=
by sorry

end problem_statement_l114_11441


namespace max_clocks_in_workshop_l114_11431

/-- Represents a digital clock with hours and minutes -/
structure DigitalClock where
  hours : Nat
  minutes : Nat

/-- Represents the state of all clocks in the workshop -/
structure ClockWorkshop where
  clocks : List DigitalClock

/-- Checks if all clocks in the workshop show different times -/
def allDifferentTimes (workshop : ClockWorkshop) : Prop :=
  ∀ c1 c2 : DigitalClock, c1 ∈ workshop.clocks → c2 ∈ workshop.clocks → c1 ≠ c2 →
    (c1.hours ≠ c2.hours ∨ c1.minutes ≠ c2.minutes)

/-- Calculates the sum of hours displayed on all clocks -/
def sumHours (workshop : ClockWorkshop) : Nat :=
  workshop.clocks.foldl (fun sum clock => sum + clock.hours) 0

/-- Calculates the sum of minutes displayed on all clocks -/
def sumMinutes (workshop : ClockWorkshop) : Nat :=
  workshop.clocks.foldl (fun sum clock => sum + clock.minutes) 0

/-- Represents the state of the workshop after some time has passed -/
def advanceTime (workshop : ClockWorkshop) : ClockWorkshop := sorry

theorem max_clocks_in_workshop :
  ∀ (workshop : ClockWorkshop),
    workshop.clocks.length > 1 →
    (∀ clock ∈ workshop.clocks, clock.hours ≥ 1 ∧ clock.hours ≤ 12) →
    (∀ clock ∈ workshop.clocks, clock.minutes ≥ 0 ∧ clock.minutes < 60) →
    allDifferentTimes workshop →
    sumHours (advanceTime workshop) + 1 = sumHours workshop →
    sumMinutes (advanceTime workshop) + 1 = sumMinutes workshop →
    workshop.clocks.length ≤ 11 :=
by sorry

end max_clocks_in_workshop_l114_11431


namespace complex_fraction_simplification_l114_11434

theorem complex_fraction_simplification :
  (2 + 4 * Complex.I) / ((1 + Complex.I)^2) = 2 - Complex.I :=
by sorry

end complex_fraction_simplification_l114_11434


namespace sufficient_not_necessary_l114_11435

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 1 → a^2 > 1) ∧
  (∃ a, a ≤ 1 ∧ a^2 > 1) :=
sorry

end sufficient_not_necessary_l114_11435


namespace quadratic_function_properties_l114_11476

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties :
  ∀ (a b c : ℝ),
  (∀ x, f a b c (x + 1) - f a b c x = 2 * x) →
  f a b c 0 = 1 →
  (∃ m : ℝ, m = -1 ∧
    (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f a b c x ≥ 2 * x + m) ∧
    (∀ m' : ℝ, m' > m →
      ∃ x, x ∈ Set.Icc (-1 : ℝ) 1 ∧ f a b c x < 2 * x + m')) →
  (∀ x, f a b c x = x^2 - x + 1) :=
by sorry


end quadratic_function_properties_l114_11476


namespace hypercoplanar_iff_b_eq_plusminus_one_over_sqrt_two_l114_11436

/-- A point in 4D space -/
def Point4D := Fin 4 → ℝ

/-- The determinant of a 4x4 matrix -/
def det4 (m : Fin 4 → Fin 4 → ℝ) : ℝ := sorry

/-- Check if five points in 4D space are hypercoplanar -/
def are_hypercoplanar (p1 p2 p3 p4 p5 : Point4D) : Prop :=
  det4 (λ i j => match i, j with
    | 0, _ => p2 j - p1 j
    | 1, _ => p3 j - p1 j
    | 2, _ => p4 j - p1 j
    | 3, _ => p5 j - p1 j) = 0

/-- The given points in 4D space -/
def p1 : Point4D := λ _ => 0
def p2 (b : ℝ) : Point4D := λ i => match i with | 0 => 1 | 1 => b | _ => 0
def p3 (b : ℝ) : Point4D := λ i => match i with | 1 => 1 | 2 => b | _ => 0
def p4 (b : ℝ) : Point4D := λ i => match i with | 0 => b | 2 => 1 | _ => 0
def p5 (b : ℝ) : Point4D := λ i => match i with | 1 => b | 3 => 1 | _ => 0

theorem hypercoplanar_iff_b_eq_plusminus_one_over_sqrt_two :
  ∀ b : ℝ, are_hypercoplanar (p1) (p2 b) (p3 b) (p4 b) (p5 b) ↔ b = 1 / Real.sqrt 2 ∨ b = -1 / Real.sqrt 2 :=
by sorry

end hypercoplanar_iff_b_eq_plusminus_one_over_sqrt_two_l114_11436


namespace geese_in_marsh_l114_11493

theorem geese_in_marsh (total_birds ducks : ℕ) (h1 : total_birds = 95) (h2 : ducks = 37) :
  total_birds - ducks = 58 := by
  sorry

end geese_in_marsh_l114_11493


namespace circle_and_five_lines_max_regions_circle_divides_plane_two_parts_circle_and_one_line_max_four_parts_circle_and_two_lines_max_eight_parts_l114_11421

/-- The maximum number of regions into which n lines can divide a plane -/
def max_regions_lines (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- The maximum number of additional regions created when k lines intersect a circle -/
def max_additional_regions (k : ℕ) : ℕ := k * 2

/-- The maximum number of regions into which a plane can be divided by 1 circle and n lines -/
def max_regions_circle_and_lines (n : ℕ) : ℕ :=
  max_regions_lines n + max_additional_regions n

theorem circle_and_five_lines_max_regions :
  max_regions_circle_and_lines 5 = 26 :=
by sorry

theorem circle_divides_plane_two_parts :
  max_regions_circle_and_lines 0 = 2 :=
by sorry

theorem circle_and_one_line_max_four_parts :
  max_regions_circle_and_lines 1 = 4 :=
by sorry

theorem circle_and_two_lines_max_eight_parts :
  max_regions_circle_and_lines 2 = 8 :=
by sorry

end circle_and_five_lines_max_regions_circle_divides_plane_two_parts_circle_and_one_line_max_four_parts_circle_and_two_lines_max_eight_parts_l114_11421


namespace initial_kittens_l114_11415

theorem initial_kittens (kittens_to_jessica kittens_to_sara kittens_left : ℕ) :
  kittens_to_jessica = 3 →
  kittens_to_sara = 6 →
  kittens_left = 9 →
  kittens_to_jessica + kittens_to_sara + kittens_left = 18 :=
by sorry

end initial_kittens_l114_11415


namespace divisor_problem_l114_11449

theorem divisor_problem (n d k q : ℤ) : 
  n = 25 * k + 4 →
  n + 15 = d * q + 4 →
  d > 0 →
  d = 19 := by
  sorry

end divisor_problem_l114_11449


namespace variance_linear_transform_l114_11458

-- Define a random variable X
variable (X : ℝ → ℝ)

-- Define the variance function D
noncomputable def D (Y : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem variance_linear_transform (h : D X = 2) : D (fun ω => 3 * X ω + 2) = 18 := by
  sorry

end variance_linear_transform_l114_11458


namespace meal_distribution_theorem_l114_11417

/-- The number of ways to derange 8 items -/
def derangement_8 : ℕ := 14833

/-- The number of ways to choose 2 items from 10 -/
def choose_2_from_10 : ℕ := 45

/-- The number of ways to distribute 10 meals of 4 types to 10 people
    such that exactly 2 people receive the correct meal type -/
def distribute_meals (d₈ : ℕ) (c₁₀₂ : ℕ) : ℕ := d₈ * c₁₀₂

theorem meal_distribution_theorem :
  distribute_meals derangement_8 choose_2_from_10 = 666885 := by
  sorry

end meal_distribution_theorem_l114_11417


namespace train_length_calculation_l114_11483

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length_calculation (train_speed : Real) (crossing_time : Real) (bridge_length : Real) :
  let speed_ms : Real := train_speed * (1000 / 3600)
  let total_distance : Real := speed_ms * crossing_time
  let train_length : Real := total_distance - bridge_length
  train_speed = 45 ∧ crossing_time = 30 ∧ bridge_length = 230 →
  train_length = 145 := by
  sorry

end train_length_calculation_l114_11483


namespace boat_stream_speed_ratio_l114_11462

/-- Proves that if rowing against the stream takes twice as long as rowing with the stream,
    then the ratio of boat speed to stream speed is 3:1 -/
theorem boat_stream_speed_ratio
  (D : ℝ) -- Distance rowed
  (B : ℝ) -- Speed of the boat in still water
  (S : ℝ) -- Speed of the stream
  (hD : D > 0) -- Distance is positive
  (hB : B > 0) -- Boat speed is positive
  (hS : S > 0) -- Stream speed is positive
  (hBS : B > S) -- Boat is faster than the stream
  (h_time : D / (B - S) = 2 * (D / (B + S))) -- Time against stream is twice time with stream
  : B / S = 3 := by
  sorry

end boat_stream_speed_ratio_l114_11462


namespace batsman_average_l114_11400

/-- Represents a batsman's cricket performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat
  lastInningsScore : Nat

/-- Calculates the average score of a batsman after their last innings -/
def calculateAverage (b : Batsman) : Nat :=
  (b.totalRuns + b.lastInningsScore) / b.innings

/-- Theorem: Given the conditions, prove that the batsman's average after the 12th innings is 82 runs -/
theorem batsman_average (b : Batsman)
  (h1 : b.innings = 12)
  (h2 : b.lastInningsScore = 115)
  (h3 : b.averageIncrease = 3)
  (h4 : calculateAverage b = calculateAverage { b with innings := b.innings - 1 } + b.averageIncrease) :
  calculateAverage b = 82 := by
  sorry

#check batsman_average

end batsman_average_l114_11400


namespace total_skips_eq_2450_l114_11439

/-- Represents the number of skips completed by a person given their skipping rate and duration. -/
def skips_completed (rate : ℚ) (duration : ℚ) : ℚ := rate * duration

/-- Calculates the total number of skips completed by Roberto, Valerie, and Lucas. -/
def total_skips : ℚ :=
  let roberto_rate : ℚ := 4200 / 60  -- skips per minute
  let valerie_rate : ℚ := 80         -- skips per minute
  let lucas_rate : ℚ := 150 / 5      -- skips per minute
  let roberto_duration : ℚ := 15     -- minutes
  let valerie_duration : ℚ := 10     -- minutes
  let lucas_duration : ℚ := 20       -- minutes
  skips_completed roberto_rate roberto_duration +
  skips_completed valerie_rate valerie_duration +
  skips_completed lucas_rate lucas_duration

theorem total_skips_eq_2450 : total_skips = 2450 := by
  sorry

end total_skips_eq_2450_l114_11439


namespace red_flesh_probability_l114_11450

/-- Represents the probability of a tomato having yellow skin -/
def yellow_skin_prob : ℚ := 3/8

/-- Represents the probability of a tomato having red flesh given it has yellow skin -/
def red_flesh_given_yellow_skin_prob : ℚ := 8/15

/-- Represents the probability of a tomato having yellow skin given it doesn't have red flesh -/
def yellow_skin_given_not_red_flesh_prob : ℚ := 7/30

/-- Theorem stating that the probability of red flesh is 1/4 given the conditions -/
theorem red_flesh_probability :
  let yellow_and_not_red : ℚ := yellow_skin_prob * (1 - red_flesh_given_yellow_skin_prob)
  let not_red_flesh_prob : ℚ := yellow_and_not_red / yellow_skin_given_not_red_flesh_prob
  let red_flesh_prob : ℚ := 1 - not_red_flesh_prob
  red_flesh_prob = 1/4 := by sorry

end red_flesh_probability_l114_11450


namespace quadratic_equation_solution_difference_l114_11428

theorem quadratic_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 5*x₁ + 15 = x₁ + 55) ∧
  (x₂^2 - 5*x₂ + 15 = x₂ + 55) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 14 :=
by
  sorry

end quadratic_equation_solution_difference_l114_11428


namespace binary_eight_ones_decimal_l114_11448

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryNumber := List Nat

/-- Converts a binary number to its decimal representation -/
def binaryToDecimal (b : BinaryNumber) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + bit * 2^i) 0

/-- The binary number (11111111)₂ -/
def binaryEightOnes : BinaryNumber := [1,1,1,1,1,1,1,1]

theorem binary_eight_ones_decimal :
  binaryToDecimal binaryEightOnes = 2^8 - 1 := by
  sorry

end binary_eight_ones_decimal_l114_11448


namespace joan_has_77_balloons_l114_11492

/-- The number of balloons Joan has after giving some away and receiving more -/
def joans_balloons (initial_blue initial_red mark_blue mark_red sarah_blue additional_red : ℕ) : ℕ :=
  (initial_blue - mark_blue - sarah_blue) + (initial_red - mark_red + additional_red)

/-- Theorem stating that Joan has 77 balloons given the problem conditions -/
theorem joan_has_77_balloons :
  joans_balloons 72 48 15 10 24 6 = 77 := by
  sorry

#eval joans_balloons 72 48 15 10 24 6

end joan_has_77_balloons_l114_11492


namespace meaningful_sqrt_range_l114_11410

theorem meaningful_sqrt_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = 2 * x - 1) ↔ x ≥ 1 / 2 := by
  sorry

end meaningful_sqrt_range_l114_11410


namespace inequality_always_true_range_l114_11484

theorem inequality_always_true_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := by
sorry

end inequality_always_true_range_l114_11484


namespace distance_between_last_two_points_l114_11424

def cube_vertices : List (Fin 3 → ℝ) := [
  ![0, 0, 0], ![0, 0, 6], ![0, 6, 0], ![0, 6, 6],
  ![6, 0, 0], ![6, 0, 6], ![6, 6, 0], ![6, 6, 6]
]

def plane_intersections : List (Fin 3 → ℝ) := [
  ![0, 3, 0], ![2, 0, 0], ![2, 6, 6], ![4, 0, 6], ![0, 6, 6]
]

theorem distance_between_last_two_points :
  let S := plane_intersections[3]
  let T := plane_intersections[4]
  Real.sqrt ((S 0 - T 0)^2 + (S 1 - T 1)^2 + (S 2 - T 2)^2) = 2 * Real.sqrt 13 := by
  sorry

end distance_between_last_two_points_l114_11424


namespace meeting_participants_l114_11429

theorem meeting_participants :
  ∀ (F M : ℕ),
  F > 0 ∧ M > 0 →
  F / 2 + M / 4 = (F + M) / 3 →
  F / 2 = 110 →
  F + M = 330 :=
by
  sorry

end meeting_participants_l114_11429


namespace street_lights_configuration_l114_11489

theorem street_lights_configuration (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 12 →
  k = 4 →
  m = n - k - 1 →
  Nat.choose m k = 35 :=
by
  sorry

end street_lights_configuration_l114_11489


namespace journey_time_ratio_l114_11419

/-- Proves that the ratio of return journey time to initial journey time is 3:2 
    given specific speed conditions -/
theorem journey_time_ratio 
  (initial_speed : ℝ) 
  (average_speed : ℝ) 
  (h1 : initial_speed = 51)
  (h2 : average_speed = 34) :
  (1 / average_speed) / (1 / initial_speed) = 3 / 2 := by
  sorry

end journey_time_ratio_l114_11419


namespace power_multiplication_l114_11494

theorem power_multiplication (m : ℝ) : m^2 * m^3 = m^5 := by
  sorry

end power_multiplication_l114_11494


namespace special_divisors_count_l114_11451

/-- The base number -/
def base : ℕ := 540

/-- The exponent of the base number -/
def exponent : ℕ := 540

/-- The number of divisors we're looking for -/
def target_divisors : ℕ := 108

/-- A function that counts the number of positive integer divisors of base^exponent 
    that are divisible by exactly target_divisors positive integers -/
def count_special_divisors (base exponent target_divisors : ℕ) : ℕ := sorry

/-- The main theorem stating that the count of special divisors is 6 -/
theorem special_divisors_count : 
  count_special_divisors base exponent target_divisors = 6 := by sorry

end special_divisors_count_l114_11451


namespace geometric_sequence_sum_l114_11443

/-- Given a geometric sequence {aₙ} where all terms are positive, 
    with a₁ = 3 and a₁ + a₂ + a₃ = 21, prove that a₃ + a₄ + a₅ = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  a 1 = 3 →  -- first term is 3
  a 1 + a 2 + a 3 = 21 →  -- sum of first three terms is 21
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence property
  a 3 + a 4 + a 5 = 84 := by
sorry

end geometric_sequence_sum_l114_11443


namespace equivalent_operation_l114_11488

theorem equivalent_operation (x : ℝ) : (x / (5/4)) * (4/3) = x * (16/15) := by
  sorry

end equivalent_operation_l114_11488


namespace max_c_value_l114_11452

theorem max_c_value (a b c x y z : ℝ) : 
  a ≥ 1 → b ≥ 1 → c ≥ 1 → x > 0 → y > 0 → z > 0 →
  a^x + b^y + c^z = 4 →
  x * a^x + y * b^y + z * c^z = 6 →
  x^2 * a^x + y^2 * b^y + z^2 * c^z = 9 →
  c ≤ Real.rpow 4 (1/3) :=
sorry

end max_c_value_l114_11452


namespace M_is_real_l114_11411

-- Define the set M
def M : Set ℂ := {z : ℂ | (z - 1)^2 = Complex.abs (z - 1)^2}

-- Theorem stating that M is equal to the set of real numbers
theorem M_is_real : M = {z : ℂ | z.im = 0} := by sorry

end M_is_real_l114_11411


namespace system_solution_l114_11423

theorem system_solution : ∃ (s t : ℝ), 
  (7 * s + 3 * t = 102) ∧ 
  (s = (t - 3)^2) ∧ 
  (abs (t - 6.44) < 0.01) ∧ 
  (abs (s - 11.83) < 0.01) := by
sorry

end system_solution_l114_11423


namespace standard_deviation_from_age_range_job_applicants_standard_deviation_l114_11413

/-- Given an average age and a number of distinct integer ages within one standard deviation,
    calculate the standard deviation. -/
theorem standard_deviation_from_age_range (average_age : ℕ) (distinct_ages : ℕ) : ℕ :=
  let standard_deviation := (distinct_ages - 1) / 2
  standard_deviation

/-- Prove that for an average age of 20 and 17 distinct integer ages within one standard deviation,
    the standard deviation is 8. -/
theorem job_applicants_standard_deviation : 
  standard_deviation_from_age_range 20 17 = 8 := by
  sorry

end standard_deviation_from_age_range_job_applicants_standard_deviation_l114_11413


namespace edward_money_left_l114_11426

def initial_money : ℕ := 41
def books_cost : ℕ := 6
def pens_cost : ℕ := 16

theorem edward_money_left : initial_money - (books_cost + pens_cost) = 19 := by
  sorry

end edward_money_left_l114_11426


namespace ratio_of_divisor_sums_l114_11496

def M : ℕ := 36 * 36 * 65 * 272

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors M) * 510 = sum_even_divisors M := by sorry

end ratio_of_divisor_sums_l114_11496


namespace firefighters_total_fires_l114_11454

/-- The number of fires put out by three firefighters -/
def total_fires (doug_fires : ℕ) (kai_multiplier : ℕ) (eli_divisor : ℕ) : ℕ :=
  doug_fires + (doug_fires * kai_multiplier) + (doug_fires * kai_multiplier / eli_divisor)

/-- Theorem stating the total number of fires put out by Doug, Kai, and Eli -/
theorem firefighters_total_fires :
  total_fires 20 3 2 = 110 := by
  sorry

#eval total_fires 20 3 2

end firefighters_total_fires_l114_11454


namespace tangent_line_equation_l114_11401

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := -1
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ 3*x + y - 2 = 0 :=
by sorry

end tangent_line_equation_l114_11401


namespace lcm_gcd_product_36_60_l114_11420

theorem lcm_gcd_product_36_60 : Nat.lcm 36 60 * Nat.gcd 36 60 = 36 * 60 := by
  sorry

end lcm_gcd_product_36_60_l114_11420


namespace square_from_l_pieces_l114_11486

/-- Represents a three-cell L-shaped piece -/
structure LPiece :=
  (cells : Fin 3 → Fin 2 → Fin 2)

/-- Represents a square grid -/
structure Square (n : ℕ) :=
  (grid : Fin n → Fin n → Bool)

/-- Checks if a given square is filled completely -/
def is_filled (s : Square n) : Prop :=
  ∀ i j, s.grid i j = true

/-- Defines the ability to place L-pieces on a square grid -/
def can_place_pieces (n : ℕ) (pieces : List LPiece) (s : Square n) : Prop :=
  sorry

/-- The main theorem stating that it's possible to form a square using L-pieces -/
theorem square_from_l_pieces :
  ∃ (n : ℕ) (pieces : List LPiece) (s : Square n),
    can_place_pieces n pieces s ∧ is_filled s :=
  sorry

end square_from_l_pieces_l114_11486


namespace probability_two_blue_jellybeans_l114_11440

-- Define the total number of jellybeans and the number of each color
def total_jellybeans : ℕ := 12
def red_jellybeans : ℕ := 3
def blue_jellybeans : ℕ := 4
def white_jellybeans : ℕ := 5

-- Define the number of jellybeans to be picked
def picked_jellybeans : ℕ := 3

-- Define the probability of picking exactly two blue jellybeans
def prob_two_blue : ℚ := 12 / 55

-- Theorem statement
theorem probability_two_blue_jellybeans : 
  prob_two_blue = (Nat.choose blue_jellybeans 2 * Nat.choose (total_jellybeans - blue_jellybeans) 1) / 
                  Nat.choose total_jellybeans picked_jellybeans :=
by sorry

end probability_two_blue_jellybeans_l114_11440


namespace shortest_side_of_right_triangle_l114_11430

theorem shortest_side_of_right_triangle (a b c : ℝ) : 
  a = 5 → b = 12 → c^2 = a^2 + b^2 → min a (min b c) = 5 := by
  sorry

end shortest_side_of_right_triangle_l114_11430


namespace x_squared_mod_20_l114_11404

theorem x_squared_mod_20 (x : ℕ) (h1 : 5 * x ≡ 10 [ZMOD 20]) (h2 : 2 * x ≡ 14 [ZMOD 20]) :
  x^2 ≡ 16 [ZMOD 20] := by
  sorry

end x_squared_mod_20_l114_11404


namespace tuesday_temperature_l114_11403

-- Define temperatures for each day
def tuesday_temp : ℝ := sorry
def wednesday_temp : ℝ := sorry
def thursday_temp : ℝ := sorry
def friday_temp : ℝ := 53

-- Define the conditions
axiom avg_tue_wed_thu : (tuesday_temp + wednesday_temp + thursday_temp) / 3 = 52
axiom avg_wed_thu_fri : (wednesday_temp + thursday_temp + friday_temp) / 3 = 54

-- Theorem to prove
theorem tuesday_temperature : tuesday_temp = 47 := by
  sorry

end tuesday_temperature_l114_11403


namespace tan_alpha_values_l114_11444

theorem tan_alpha_values (α : Real) (h : 2 * Real.sin (2 * α) = 1 - Real.cos (2 * α)) :
  Real.tan α = 2 ∨ Real.tan α = 0 := by sorry

end tan_alpha_values_l114_11444


namespace deal_or_no_deal_probability_l114_11482

def box_values : List ℝ := [0.01, 1, 5, 10, 25, 50, 100, 200, 300, 400, 500, 750, 1000, 5000, 10000, 25000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 750000, 1000000]

def total_boxes : ℕ := 30

def threshold : ℝ := 200000

theorem deal_or_no_deal_probability (boxes_to_eliminate : ℕ) :
  boxes_to_eliminate = 16 ↔
    (((box_values.filter (λ x => x ≥ threshold)).length : ℝ) / (total_boxes - boxes_to_eliminate : ℝ) = 1/2 ∧
     boxes_to_eliminate < total_boxes ∧
     ∀ n : ℕ, n < boxes_to_eliminate →
       ((box_values.filter (λ x => x ≥ threshold)).length : ℝ) / (total_boxes - n : ℝ) < 1/2) :=
by sorry

end deal_or_no_deal_probability_l114_11482


namespace sequence_of_primes_l114_11447

theorem sequence_of_primes (a p : ℕ → ℕ) 
  (h_increasing : ∀ n m, n < m → a n < a m)
  (h_positive : ∀ n, 0 < a n)
  (h_prime : ∀ n, Nat.Prime (p n))
  (h_distinct : ∀ n m, n ≠ m → p n ≠ p m)
  (h_divides : ∀ n, p n ∣ a n)
  (h_difference : ∀ n k, a n - a k = p n - p k) :
  ∀ n, a n = p n :=
sorry

end sequence_of_primes_l114_11447


namespace subset_coloring_existence_l114_11405

/-- The coloring function type -/
def ColoringFunction (α : Type*) := Set α → Bool

/-- Theorem statement -/
theorem subset_coloring_existence
  (S : Type*)
  [Fintype S]
  (h_card : Fintype.card S = 2002)
  (N : ℕ)
  (h_N : N ≤ 2^2002) :
  ∃ (f : ColoringFunction S),
    (∀ A B : Set S, f A ∧ f B → f (A ∪ B)) ∧
    (∀ A B : Set S, ¬f A ∧ ¬f B → ¬f (A ∪ B)) ∧
    (Fintype.card {A : Set S | f A} = N) :=
by sorry

end subset_coloring_existence_l114_11405


namespace A_inverse_proof_l114_11463

def A : Matrix (Fin 3) (Fin 3) ℚ := !![2, 5, 6; 1, 2, 5; 1, 2, 3]

def A_inv : Matrix (Fin 3) (Fin 3) ℚ := !![-2, 3/2, 13/2; 1, 0, 2; 0, -1/2, -1/2]

theorem A_inverse_proof : A⁻¹ = A_inv := by sorry

end A_inverse_proof_l114_11463


namespace eighth_odd_multiple_of_5_l114_11445

/-- The nth positive integer that is both odd and a multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ :=
  10 * n - 5

theorem eighth_odd_multiple_of_5 : nthOddMultipleOf5 8 = 75 := by
  sorry

end eighth_odd_multiple_of_5_l114_11445


namespace min_value_expression_l114_11422

theorem min_value_expression (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 4 * (5^(1/4) - 5/4)^2 := by
  sorry

end min_value_expression_l114_11422


namespace concrete_mixture_theorem_l114_11412

/-- The amount of 80% cement mixture used in tons -/
def amount_80_percent : ℝ := 7.0

/-- The percentage of cement in the final mixture -/
def final_cement_percentage : ℝ := 0.62

/-- The percentage of cement in the first mixture -/
def first_mixture_percentage : ℝ := 0.20

/-- The percentage of cement in the second mixture -/
def second_mixture_percentage : ℝ := 0.80

/-- The total amount of concrete made in tons -/
def total_concrete : ℝ := 10.0

theorem concrete_mixture_theorem :
  ∃ (x : ℝ),
    x ≥ 0 ∧
    x * first_mixture_percentage + amount_80_percent * second_mixture_percentage =
      final_cement_percentage * (x + amount_80_percent) ∧
    x + amount_80_percent = total_concrete :=
by sorry

end concrete_mixture_theorem_l114_11412


namespace quarterback_sacks_l114_11487

theorem quarterback_sacks (total_attempts : ℕ) (no_throw_percentage : ℚ) (sack_ratio : ℚ) : 
  total_attempts = 80 → 
  no_throw_percentage = 30 / 100 → 
  sack_ratio = 1 / 2 → 
  ⌊(total_attempts : ℚ) * no_throw_percentage * sack_ratio⌋ = 12 := by
  sorry

end quarterback_sacks_l114_11487


namespace product_absolute_value_one_l114_11497

theorem product_absolute_value_one 
  (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d)
  (h1 : a + 1/b = b + 1/c)
  (h2 : b + 1/c = c + 1/d)
  (h3 : c + 1/d = d + 1/a) :
  |a * b * c * d| = 1 := by
sorry

end product_absolute_value_one_l114_11497


namespace least_trees_for_rows_trees_168_divisible_least_trees_is_168_l114_11471

theorem least_trees_for_rows (n : ℕ) : n > 0 ∧ 6 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n → n ≥ 168 := by
  sorry

theorem trees_168_divisible : 6 ∣ 168 ∧ 7 ∣ 168 ∧ 8 ∣ 168 := by
  sorry

theorem least_trees_is_168 : ∃ (n : ℕ), n > 0 ∧ 6 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n ∧ ∀ (m : ℕ), (m > 0 ∧ 6 ∣ m ∧ 7 ∣ m ∧ 8 ∣ m) → m ≥ n := by
  sorry

end least_trees_for_rows_trees_168_divisible_least_trees_is_168_l114_11471


namespace proportion_solution_l114_11437

theorem proportion_solution (x : ℝ) : (0.60 / x = 6 / 4) → x = 0.4 := by
  sorry

end proportion_solution_l114_11437


namespace square_side_length_l114_11438

theorem square_side_length (perimeter : ℝ) (h1 : perimeter = 16) : 
  perimeter / 4 = 4 := by
  sorry

#check square_side_length

end square_side_length_l114_11438


namespace closest_to_M_div_N_l114_11414

-- Define the state space complexity of Go
def M : ℝ := 3^361

-- Define the number of atoms in the observable universe
def N : ℝ := 10^80

-- Define the options
def options : List ℝ := [10^33, 10^53, 10^73, 10^93]

-- Theorem statement
theorem closest_to_M_div_N :
  let ratio := M / N
  (∀ x ∈ options, |ratio - 10^93| ≤ |ratio - x|) ∧ (10^93 ∈ options) :=
by sorry

end closest_to_M_div_N_l114_11414
