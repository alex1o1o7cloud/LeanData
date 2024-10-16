import Mathlib

namespace NUMINAMATH_CALUDE_farmer_plan_proof_l2712_271206

/-- The number of cows a farmer plans to add to their farm -/
def planned_cows : ℕ := 3

/-- The initial number of animals on the farm -/
def initial_animals : ℕ := 11

/-- The number of pigs and goats the farmer plans to add -/
def planned_pigs_and_goats : ℕ := 7

/-- The total number of animals after all additions -/
def total_animals : ℕ := 21

theorem farmer_plan_proof :
  initial_animals + planned_pigs_and_goats + planned_cows = total_animals :=
by sorry

end NUMINAMATH_CALUDE_farmer_plan_proof_l2712_271206


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_l2712_271287

theorem quadratic_equation_general_form :
  ∀ x : ℝ, (1 + 3 * x) * (x - 3) = 2 * x^2 + 1 ↔ x^2 - 8 * x - 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_l2712_271287


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l2712_271220

theorem quadratic_equation_problem (k : ℝ) (α β : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + 3 - k = 0 ∧ y^2 + 2*y + 3 - k = 0) →
  (k > 2 ∧ 
   (k^2 = α*β + 3*k ∧ α^2 + 2*α + 3 - k = 0 ∧ β^2 + 2*β + 3 - k = 0) → k = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l2712_271220


namespace NUMINAMATH_CALUDE_ellipse_points_equiv_target_set_l2712_271278

/-- An ellipse passing through (2,1) with the given conditions -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : 4 / a^2 + 1 / b^2 = 1

/-- The set of points on the ellipse satisfying |y| > 1 -/
def ellipse_points (e : Ellipse) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1 ∧ |p.2| > 1}

/-- The target set -/
def target_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 < 5 ∧ |p.2| > 1}

/-- The main theorem -/
theorem ellipse_points_equiv_target_set (e : Ellipse) :
  ellipse_points e = target_set := by sorry

end NUMINAMATH_CALUDE_ellipse_points_equiv_target_set_l2712_271278


namespace NUMINAMATH_CALUDE_negation_of_exists_negation_of_quadratic_equation_l2712_271207

theorem negation_of_exists (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 - 3*x + 2 = 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_negation_of_quadratic_equation_l2712_271207


namespace NUMINAMATH_CALUDE_largest_prime_factor_l2712_271248

def numbers : List Nat := [85, 57, 119, 143, 169]

def has_largest_prime_factor (n : Nat) (ns : List Nat) : Prop :=
  ∀ m ∈ ns, ∀ p : Nat, p.Prime → p ∣ m → ∃ q : Nat, q.Prime ∧ q ∣ n ∧ q ≥ p

theorem largest_prime_factor :
  has_largest_prime_factor 57 numbers := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l2712_271248


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2712_271274

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 8) * (Real.sqrt 6 / Real.sqrt 9) = Real.sqrt 35 / 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2712_271274


namespace NUMINAMATH_CALUDE_probability_of_sum_17_l2712_271271

/-- The number of faces on each die -/
def numFaces : ℕ := 8

/-- The target sum we're aiming for -/
def targetSum : ℕ := 17

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The probability of rolling a specific number on a single die -/
def singleDieProbability : ℚ := 1 / numFaces

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (ways to get a sum of 17) -/
def favorableOutcomes : ℕ := 27

/-- The theorem stating the probability of rolling a sum of 17 with three 8-faced dice -/
theorem probability_of_sum_17 : 
  (favorableOutcomes : ℚ) / totalOutcomes = 27 / 512 :=
sorry

end NUMINAMATH_CALUDE_probability_of_sum_17_l2712_271271


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2712_271223

theorem cubic_equation_solution :
  {x : ℝ | x^3 + 6*x^2 + 11*x + 6 = 12} = {-3, -2, -1} := by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2712_271223


namespace NUMINAMATH_CALUDE_solution_inequality1_solution_inequality_system_l2712_271282

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := 2 * x + 3 ≤ 5 * x
def inequality2 (x : ℝ) : Prop := 5 * x - 1 ≤ 3 * (x + 1)
def inequality3 (x : ℝ) : Prop := (2 * x - 1) / 2 - (5 * x - 1) / 4 < 1

-- Theorem for the first inequality
theorem solution_inequality1 :
  {x : ℝ | inequality1 x} = {x : ℝ | x ≥ 1} :=
sorry

-- Theorem for the system of inequalities
theorem solution_inequality_system :
  {x : ℝ | inequality2 x ∧ inequality3 x} = {x : ℝ | -5 < x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_CALUDE_solution_inequality1_solution_inequality_system_l2712_271282


namespace NUMINAMATH_CALUDE_six_by_six_tiling_impossible_l2712_271255

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a tile -/
structure Tile :=
  (length : Nat)
  (width : Nat)

/-- Represents a tiling configuration -/
structure TilingConfig :=
  (board : Chessboard)
  (tile : Tile)
  (num_tiles : Nat)

/-- Predicate to check if a tiling configuration is valid -/
def is_valid_tiling (config : TilingConfig) : Prop :=
  config.board.rows * config.board.cols = config.tile.length * config.tile.width * config.num_tiles

/-- Theorem stating that a 6x6 chessboard cannot be tiled with nine 1x4 tiles -/
theorem six_by_six_tiling_impossible :
  ¬ is_valid_tiling { board := { rows := 6, cols := 6 },
                      tile := { length := 1, width := 4 },
                      num_tiles := 9 } :=
by sorry

end NUMINAMATH_CALUDE_six_by_six_tiling_impossible_l2712_271255


namespace NUMINAMATH_CALUDE_white_bread_cost_l2712_271270

/-- Represents the cost of bread items in dollars -/
structure BreadCosts where
  white : ℝ
  baguette : ℝ := 1.50
  sourdough : ℝ := 4.50
  croissant : ℝ := 2.00

/-- Represents the weekly purchase of bread items -/
structure WeeklyPurchase where
  white : ℕ := 2
  baguette : ℕ := 1
  sourdough : ℕ := 2
  croissant : ℕ := 1

def total_spent_over_4_weeks : ℝ := 78

/-- Calculates the weekly cost of non-white bread items -/
def weekly_non_white_cost (costs : BreadCosts) (purchase : WeeklyPurchase) : ℝ :=
  costs.baguette * purchase.baguette + 
  costs.sourdough * purchase.sourdough + 
  costs.croissant * purchase.croissant

/-- Theorem stating that the cost of each loaf of white bread is $3.50 -/
theorem white_bread_cost (costs : BreadCosts) (purchase : WeeklyPurchase) :
  costs.white = 3.50 ↔ 
  total_spent_over_4_weeks = 
    4 * (weekly_non_white_cost costs purchase + costs.white * purchase.white) :=
sorry

end NUMINAMATH_CALUDE_white_bread_cost_l2712_271270


namespace NUMINAMATH_CALUDE_dog_food_theorem_l2712_271296

/-- The number of cups of dog food in a bag that lasts 16 days -/
def cups_in_bag (morning_cups : ℕ) (evening_cups : ℕ) (days : ℕ) : ℕ :=
  (morning_cups + evening_cups) * days

/-- Theorem stating that a bag lasting 16 days contains 32 cups of dog food -/
theorem dog_food_theorem :
  cups_in_bag 1 1 16 = 32 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_theorem_l2712_271296


namespace NUMINAMATH_CALUDE_hexagon_congruent_angles_l2712_271261

/-- In a hexagon with three congruent angles and two pairs of supplementary angles,
    each of the congruent angles measures 120 degrees. -/
theorem hexagon_congruent_angles (F I G U R E : Real) : 
  F = I ∧ I = U ∧  -- Three angles are congruent
  G + E = 180 ∧    -- One pair of supplementary angles
  R + U = 180 ∧    -- Another pair of supplementary angles
  F + I + G + U + R + E = 720  -- Sum of angles in a hexagon
  → U = 120 := by sorry

end NUMINAMATH_CALUDE_hexagon_congruent_angles_l2712_271261


namespace NUMINAMATH_CALUDE_circle_chord_length_l2712_271283

theorem circle_chord_length (AB CD : ℝ) (h1 : AB = 13) (h2 : CD = 6) :
  let AD := (x : ℝ)
  (x = 4 ∨ x = 9) ↔ x^2 - AB*x + CD^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_circle_chord_length_l2712_271283


namespace NUMINAMATH_CALUDE_total_basketball_cost_l2712_271225

/-- Represents a basketball team -/
structure Team where
  players : Nat
  basketballs_per_player : Nat
  price_per_basketball : Nat

/-- Calculates the total cost of basketballs for a team -/
def team_cost (t : Team) : Nat :=
  t.players * t.basketballs_per_player * t.price_per_basketball

/-- The Spurs basketball team -/
def spurs : Team :=
  { players := 22
    basketballs_per_player := 11
    price_per_basketball := 15 }

/-- The Dynamos basketball team -/
def dynamos : Team :=
  { players := 18
    basketballs_per_player := 9
    price_per_basketball := 20 }

/-- The Lions basketball team -/
def lions : Team :=
  { players := 26
    basketballs_per_player := 7
    price_per_basketball := 12 }

/-- Theorem stating the total cost of basketballs for all three teams -/
theorem total_basketball_cost :
  team_cost spurs + team_cost dynamos + team_cost lions = 9054 := by
  sorry

end NUMINAMATH_CALUDE_total_basketball_cost_l2712_271225


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2712_271257

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I) = 2) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2712_271257


namespace NUMINAMATH_CALUDE_gcd_228_1995_l2712_271292

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l2712_271292


namespace NUMINAMATH_CALUDE_solution_characterization_l2712_271288

/-- A function satisfying the given differential equation for all real x and positive integers n -/
def SatisfiesDiffEq (f : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ) (n : ℕ), n > 0 → Differentiable ℝ f ∧ 
    deriv f x = (f (x + n) - f x) / n

/-- The main theorem stating that any function satisfying the differential equation
    is of the form f(x) = ax + b for some real constants a and b -/
theorem solution_characterization (f : ℝ → ℝ) :
  SatisfiesDiffEq f → ∃ (a b : ℝ), ∀ x, f x = a * x + b :=
sorry

end NUMINAMATH_CALUDE_solution_characterization_l2712_271288


namespace NUMINAMATH_CALUDE_volume_range_l2712_271253

/-- A rectangular prism with given surface area and sum of edge lengths -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  surface_area_eq : 2 * (a * b + b * c + a * c) = 48
  edge_sum_eq : 4 * (a + b + c) = 36

/-- The volume of a rectangular prism -/
def volume (p : RectangularPrism) : ℝ := p.a * p.b * p.c

/-- Theorem stating the range of possible volumes for the given rectangular prism -/
theorem volume_range (p : RectangularPrism) : 
  16 ≤ volume p ∧ volume p ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_volume_range_l2712_271253


namespace NUMINAMATH_CALUDE_susan_drinks_eight_l2712_271247

/-- The number of juice bottles Paul drinks per day -/
def paul_bottles : ℚ := 2

/-- The number of juice bottles Donald drinks per day -/
def donald_bottles : ℚ := 2 * paul_bottles + 3

/-- The number of juice bottles Susan drinks per day -/
def susan_bottles : ℚ := 1.5 * donald_bottles - 2.5

/-- Theorem stating that Susan drinks 8 bottles of juice per day -/
theorem susan_drinks_eight : susan_bottles = 8 := by
  sorry

end NUMINAMATH_CALUDE_susan_drinks_eight_l2712_271247


namespace NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l2712_271212

theorem smallest_x_satisfying_equation : 
  ∃ (x : ℝ), x > 0 ∧ 
  (⌊x^2⌋ : ℤ) - x * (⌊x⌋ : ℤ) = 7 ∧ 
  ∀ (y : ℝ), y > 0 → (⌊y^2⌋ : ℤ) - y * (⌊y⌋ : ℤ) = 7 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l2712_271212


namespace NUMINAMATH_CALUDE_elena_garden_petals_l2712_271279

/-- The number of lilies in Elena's garden -/
def num_lilies : ℕ := 8

/-- The number of tulips in Elena's garden -/
def num_tulips : ℕ := 5

/-- The number of petals each lily has -/
def petals_per_lily : ℕ := 6

/-- The number of petals each tulip has -/
def petals_per_tulip : ℕ := 3

/-- The total number of petals in Elena's garden -/
def total_petals : ℕ := num_lilies * petals_per_lily + num_tulips * petals_per_tulip

theorem elena_garden_petals : total_petals = 63 := by
  sorry

end NUMINAMATH_CALUDE_elena_garden_petals_l2712_271279


namespace NUMINAMATH_CALUDE_eduardo_flour_amount_l2712_271268

/-- Represents the number of cookies in the original recipe -/
def original_cookies : ℕ := 30

/-- Represents the amount of flour (in cups) needed for the original recipe -/
def original_flour : ℕ := 2

/-- Represents the number of cookies Eduardo wants to bake -/
def eduardo_cookies : ℕ := 90

/-- Calculates the amount of flour needed for a given number of cookies -/
def flour_needed (cookies : ℕ) : ℕ :=
  (cookies * original_flour) / original_cookies

theorem eduardo_flour_amount : flour_needed eduardo_cookies = 6 := by
  sorry

end NUMINAMATH_CALUDE_eduardo_flour_amount_l2712_271268


namespace NUMINAMATH_CALUDE_western_rattlesnake_segments_l2712_271231

/-- The number of segments in Eastern rattlesnakes' tails -/
def eastern_segments : ℕ := 6

/-- The percentage difference in tail size as a fraction -/
def percentage_difference : ℚ := 1/4

/-- The number of segments in Western rattlesnakes' tails -/
def western_segments : ℕ := 8

/-- Theorem stating that the number of segments in Western rattlesnakes' tails is 8,
    given the conditions from the problem -/
theorem western_rattlesnake_segments :
  (western_segments : ℚ) - eastern_segments = percentage_difference * western_segments :=
sorry

end NUMINAMATH_CALUDE_western_rattlesnake_segments_l2712_271231


namespace NUMINAMATH_CALUDE_triangular_area_l2712_271211

/-- The area of the triangular part of a piece of land -/
theorem triangular_area (total_length total_width rect_length rect_width : ℝ) 
  (h1 : total_length = 20)
  (h2 : total_width = 6)
  (h3 : rect_length = 15)
  (h4 : rect_width = 6) :
  total_length * total_width - rect_length * rect_width = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangular_area_l2712_271211


namespace NUMINAMATH_CALUDE_maze_exit_probabilities_l2712_271217

/-- Represents the three passages in the maze -/
inductive Passage
| one
| two
| three

/-- Time taken to exit each passage -/
def exit_time (p : Passage) : ℕ :=
  match p with
  | Passage.one => 1
  | Passage.two => 2
  | Passage.three => 3

/-- The probability of selecting a passage when n passages are available -/
def select_prob (n : ℕ) : ℚ :=
  1 / n

theorem maze_exit_probabilities :
  let p_one_hour := select_prob 3
  let p_more_than_three_hours := 
    select_prob 3 * select_prob 2 + 
    select_prob 3 * select_prob 2 + 
    select_prob 3 * select_prob 2
  (p_one_hour = 1/3) ∧ 
  (p_more_than_three_hours = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_maze_exit_probabilities_l2712_271217


namespace NUMINAMATH_CALUDE_cube_divided_by_self_l2712_271233

theorem cube_divided_by_self (a : ℝ) (h : a ≠ 0) : a^3 / a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_divided_by_self_l2712_271233


namespace NUMINAMATH_CALUDE_abs_diff_roots_quadratic_l2712_271222

theorem abs_diff_roots_quadratic : 
  let a : ℝ := 1
  let b : ℝ := -7
  let c : ℝ := 10
  let r₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  |r₁ - r₂| = 3 := by
sorry


end NUMINAMATH_CALUDE_abs_diff_roots_quadratic_l2712_271222


namespace NUMINAMATH_CALUDE_third_number_in_list_l2712_271201

theorem third_number_in_list (a b c d e : ℕ) : 
  a = 60 → 
  e = 300 → 
  a * b * c = 810000 → 
  b * c * d = 2430000 → 
  c * d * e = 8100000 → 
  c = 150 := by
sorry

end NUMINAMATH_CALUDE_third_number_in_list_l2712_271201


namespace NUMINAMATH_CALUDE_current_wax_amount_l2712_271299

theorem current_wax_amount (total_required : ℕ) (additional_needed : ℕ) 
  (h1 : total_required = 492)
  (h2 : additional_needed = 481) :
  total_required - additional_needed = 11 := by
  sorry

end NUMINAMATH_CALUDE_current_wax_amount_l2712_271299


namespace NUMINAMATH_CALUDE_division_problem_l2712_271238

theorem division_problem (d : ℕ) (h : 23 = d * 4 + 3) : d = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2712_271238


namespace NUMINAMATH_CALUDE_segments_form_quadrilateral_l2712_271200

/-- A function that checks if three line segments can form a quadrilateral with a fourth segment -/
def can_form_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c > d

/-- Theorem stating that line segments of length 2, 2, 2 can form a quadrilateral with a segment of length 5 -/
theorem segments_form_quadrilateral :
  can_form_quadrilateral 2 2 2 5 := by
  sorry

end NUMINAMATH_CALUDE_segments_form_quadrilateral_l2712_271200


namespace NUMINAMATH_CALUDE_tuesday_appointment_duration_l2712_271243

/-- Amanda's hourly rate in dollars -/
def hourly_rate : ℚ := 20

/-- Duration of Monday appointments in hours -/
def monday_hours : ℚ := 5 * (3/2)

/-- Duration of Thursday appointments in hours -/
def thursday_hours : ℚ := 2 * 2

/-- Duration of Saturday appointment in hours -/
def saturday_hours : ℚ := 6

/-- Total earnings for the week in dollars -/
def total_earnings : ℚ := 410

/-- Duration of Tuesday appointment in hours -/
def tuesday_hours : ℚ := 3

theorem tuesday_appointment_duration :
  hourly_rate * (monday_hours + thursday_hours + saturday_hours + tuesday_hours) = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_tuesday_appointment_duration_l2712_271243


namespace NUMINAMATH_CALUDE_square_root_of_49_l2712_271218

theorem square_root_of_49 : (Real.sqrt 49)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_49_l2712_271218


namespace NUMINAMATH_CALUDE_cube_preserves_order_for_negative_numbers_l2712_271295

theorem cube_preserves_order_for_negative_numbers (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^3 < b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_for_negative_numbers_l2712_271295


namespace NUMINAMATH_CALUDE_lines_parallel_iff_l2712_271230

/-- Two lines in R² defined by parametric equations -/
structure ParallelLines where
  k : ℝ
  line1 : ℝ → ℝ × ℝ := λ t => (1 + 5*t, 3 - 3*t)
  line2 : ℝ → ℝ × ℝ := λ s => (4 - 2*s, 1 + k*s)

/-- The lines are parallel (do not intersect) if and only if k = 6/5 -/
theorem lines_parallel_iff (pl : ParallelLines) : 
  (∀ t s, pl.line1 t ≠ pl.line2 s) ↔ pl.k = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_l2712_271230


namespace NUMINAMATH_CALUDE_compound_interest_rate_l2712_271272

theorem compound_interest_rate (ci_2 ci_3 : ℚ) 
  (h1 : ci_2 = 1200)
  (h2 : ci_3 = 1260) : 
  ∃ r : ℚ, r = 0.05 ∧ r * ci_2 = ci_3 - ci_2 :=
sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l2712_271272


namespace NUMINAMATH_CALUDE_max_magic_triangle_sum_l2712_271284

def MagicTriangle : Type := Fin 6 → Nat

def isValidTriangle (t : MagicTriangle) : Prop :=
  (∀ i : Fin 6, t i ≥ 4 ∧ t i ≤ 9) ∧
  (∀ i j : Fin 6, i ≠ j → t i ≠ t j)

def sumS (t : MagicTriangle) : Nat :=
  3 * t 0 + 2 * t 1 + 2 * t 2 + t 3 + t 4

def isBalanced (t : MagicTriangle) : Prop :=
  sumS t = 2 * t 2 + t 3 + 2 * t 4 ∧
  sumS t = 2 * t 4 + t 5 + 2 * t 1

theorem max_magic_triangle_sum :
  ∀ t : MagicTriangle, isValidTriangle t → isBalanced t →
  sumS t ≤ 40 :=
sorry

end NUMINAMATH_CALUDE_max_magic_triangle_sum_l2712_271284


namespace NUMINAMATH_CALUDE_cot_thirty_degrees_l2712_271202

theorem cot_thirty_degrees : Real.cos (π / 6) / Real.sin (π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_thirty_degrees_l2712_271202


namespace NUMINAMATH_CALUDE_austin_hourly_rate_l2712_271265

def hours_per_week : ℕ := 6
def weeks_worked : ℕ := 6
def bicycle_cost : ℕ := 180

theorem austin_hourly_rate :
  ∃ (rate : ℚ), rate * (hours_per_week * weeks_worked : ℚ) = bicycle_cost ∧ rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_austin_hourly_rate_l2712_271265


namespace NUMINAMATH_CALUDE_waiter_income_fraction_l2712_271285

theorem waiter_income_fraction (salary tips income : ℚ) : 
  income = salary + tips → 
  tips = (5 : ℚ) / 4 * salary → 
  tips / income = (5 : ℚ) / 9 := by
  sorry

end NUMINAMATH_CALUDE_waiter_income_fraction_l2712_271285


namespace NUMINAMATH_CALUDE_pet_walking_problem_l2712_271291

def smallest_common_multiple (a b : ℕ) : ℕ := Nat.lcm a b

theorem pet_walking_problem (gabe_group_size steven_group_size : ℕ) 
  (h1 : gabe_group_size = 2) 
  (h2 : steven_group_size = 10) : 
  smallest_common_multiple gabe_group_size steven_group_size = 20 := by
  sorry

#check pet_walking_problem

end NUMINAMATH_CALUDE_pet_walking_problem_l2712_271291


namespace NUMINAMATH_CALUDE_percentage_difference_l2712_271286

theorem percentage_difference : (0.6 * 50) - (0.42 * 30) = 17.4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2712_271286


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_product_l2712_271280

theorem square_difference_of_sum_and_product (x y : ℕ+) 
  (sum_eq : x + y = 26) 
  (product_eq : x * y = 168) : 
  x^2 - y^2 = 52 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_product_l2712_271280


namespace NUMINAMATH_CALUDE_donuts_left_for_coworkers_l2712_271250

def total_donuts : ℕ := 30
def gluten_free_donuts : ℕ := 12
def regular_donuts : ℕ := 18
def chocolate_gluten_free : ℕ := 6
def plain_gluten_free : ℕ := 6
def chocolate_regular : ℕ := 11
def plain_regular : ℕ := 7

def eaten_while_driving_gluten_free : ℕ := 1
def eaten_while_driving_regular : ℕ := 1

def afternoon_snack_regular : ℕ := 3
def afternoon_snack_gluten_free : ℕ := 3

theorem donuts_left_for_coworkers :
  total_donuts - 
  (eaten_while_driving_gluten_free + eaten_while_driving_regular + 
   afternoon_snack_regular + afternoon_snack_gluten_free) = 23 := by
  sorry

end NUMINAMATH_CALUDE_donuts_left_for_coworkers_l2712_271250


namespace NUMINAMATH_CALUDE_min_turns_to_win_l2712_271264

/-- Represents the state of the game -/
structure GameState :=
  (a₁ a₂ a₃ a₄ a₅ : ℕ)

/-- Defines a valid move in the game -/
def validMove (i : ℕ) : Prop :=
  2 ≤ i ∧ i ≤ 5

/-- Applies a move to the game state -/
def applyMove (state : GameState) (i : ℕ) : GameState :=
  match i with
  | 2 => GameState.mk state.a₁ (state.a₁ + state.a₂) state.a₃ state.a₄ state.a₅
  | 3 => GameState.mk state.a₁ state.a₂ (state.a₂ + state.a₃) state.a₄ state.a₅
  | 4 => GameState.mk state.a₁ state.a₂ state.a₃ (state.a₃ + state.a₄) state.a₅
  | 5 => GameState.mk state.a₁ state.a₂ state.a₃ state.a₄ (state.a₄ + state.a₅)
  | _ => state

/-- The initial state of the game -/
def initialState : GameState :=
  GameState.mk 1 0 0 0 0

/-- Predicate to check if the game is won -/
def isWinningState (state : GameState) : Prop :=
  state.a₅ > 1000000

/-- Theorem: The minimum number of turns to win the game is 127 -/
theorem min_turns_to_win :
  ∃ (moves : List ℕ),
    (∀ m ∈ moves, validMove m) ∧
    isWinningState (moves.foldl applyMove initialState) ∧
    moves.length = 127 ∧
    (∀ (other_moves : List ℕ),
      (∀ m ∈ other_moves, validMove m) →
      isWinningState (other_moves.foldl applyMove initialState) →
      other_moves.length ≥ 127) :=
by sorry


end NUMINAMATH_CALUDE_min_turns_to_win_l2712_271264


namespace NUMINAMATH_CALUDE_min_distance_point_triangle_l2712_271237

/-- Given a triangle ABC with vertices (x₁, y₁), (x₂, y₂), and (x₃, y₃), 
    this theorem states that the point P which minimizes the sum of squared distances 
    to the vertices of triangle ABC has coordinates ((x₁ + x₂ + x₃)/3, (y₁ + y₂ + y₃)/3). -/
theorem min_distance_point_triangle (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) :
  let vertices := [(x₁, y₁), (x₂, y₂), (x₃, y₃)]
  let sum_squared_distances (px py : ℝ) := 
    (vertices.map (fun (x, y) => (px - x)^2 + (py - y)^2)).sum
  let p := ((x₁ + x₂ + x₃)/3, (y₁ + y₂ + y₃)/3)
  ∀ q : ℝ × ℝ, sum_squared_distances p.1 p.2 ≤ sum_squared_distances q.1 q.2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_point_triangle_l2712_271237


namespace NUMINAMATH_CALUDE_circle_center_distance_l2712_271205

/-- Given a circle with equation x^2 + y^2 - 6x + 8y + 4 = 0 and a point (19, 11),
    the distance between the center of the circle and the point is √481. -/
theorem circle_center_distance (x y : ℝ) : 
  (x^2 + y^2 - 6*x + 8*y + 4 = 0) → 
  Real.sqrt ((19 - x)^2 + (11 - y)^2) = Real.sqrt 481 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_distance_l2712_271205


namespace NUMINAMATH_CALUDE_nonagon_side_length_l2712_271297

/-- The length of one side of a regular nonagon with circumference 171 cm is 19 cm. -/
theorem nonagon_side_length : 
  ∀ (circumference side_length : ℝ),
  circumference = 171 →
  side_length * 9 = circumference →
  side_length = 19 := by
sorry

end NUMINAMATH_CALUDE_nonagon_side_length_l2712_271297


namespace NUMINAMATH_CALUDE_num_hexagons_l2712_271228

/-- The number of triangle-shaped cookie cutters -/
def num_triangles : ℕ := 6

/-- The number of square-shaped cookie cutters -/
def num_squares : ℕ := 4

/-- The total number of sides on all cookie cutters -/
def total_sides : ℕ := 46

/-- The number of sides on a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides on a square -/
def square_sides : ℕ := 4

/-- The number of sides on a hexagon -/
def hexagon_sides : ℕ := 6

/-- Theorem stating that there are 2 hexagon-shaped cookie cutters -/
theorem num_hexagons : ℕ := by
  sorry

end NUMINAMATH_CALUDE_num_hexagons_l2712_271228


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l2712_271294

/-- 
Given an isosceles triangle ABC where:
- Angle A is congruent to angle C
- The measure of angle B is 40 degrees less than twice the measure of angle A
Prove that the measure of angle B is 70 degrees
-/
theorem isosceles_triangle_angle_measure (A B C : ℝ) : 
  A = C →  -- Angle A is congruent to angle C
  B = 2 * A - 40 →  -- Measure of angle B is 40 degrees less than twice the measure of angle A
  A + B + C = 180 →  -- Sum of angles in a triangle is 180 degrees
  B = 70 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l2712_271294


namespace NUMINAMATH_CALUDE_power_sum_and_division_l2712_271240

theorem power_sum_and_division (a b c : ℕ) : 3^456 + 9^5 / 9^3 = 82 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_and_division_l2712_271240


namespace NUMINAMATH_CALUDE_seven_thirteenths_of_3940_percent_of_25000_l2712_271227

theorem seven_thirteenths_of_3940_percent_of_25000 : 
  (7 / 13 * 3940) / 25000 * 100 = 8.484 := by
  sorry

end NUMINAMATH_CALUDE_seven_thirteenths_of_3940_percent_of_25000_l2712_271227


namespace NUMINAMATH_CALUDE_xy_plus_inverse_min_value_l2712_271241

theorem xy_plus_inverse_min_value (x y : ℝ) 
  (hx : x < 0) (hy : y < 0) (hsum : x + y = -1) :
  ∀ z, z = x * y + 1 / (x * y) → z ≥ 17 / 4 :=
sorry

end NUMINAMATH_CALUDE_xy_plus_inverse_min_value_l2712_271241


namespace NUMINAMATH_CALUDE_oxford_high_school_principals_l2712_271281

/-- Oxford High School Problem -/
theorem oxford_high_school_principals 
  (total_people : ℕ) 
  (teachers : ℕ) 
  (classes : ℕ) 
  (students_per_class : ℕ) 
  (h1 : total_people = 349) 
  (h2 : teachers = 48) 
  (h3 : classes = 15) 
  (h4 : students_per_class = 20) :
  total_people - (teachers + classes * students_per_class) = 1 :=
by sorry

end NUMINAMATH_CALUDE_oxford_high_school_principals_l2712_271281


namespace NUMINAMATH_CALUDE_parabola_translation_l2712_271209

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := 2 * p.a * h + p.b
  , c := p.a * h^2 + p.b * h + p.c - v }

theorem parabola_translation (p : Parabola) :
  p.a = 1/2 ∧ p.b = 0 ∧ p.c = 1 →
  let p' := translate p 1 3
  p'.a = 1/2 ∧ p'.b = 1 ∧ p'.c = -3/2 := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2712_271209


namespace NUMINAMATH_CALUDE_total_migration_l2712_271221

/-- The number of bird families that flew away for the winter -/
def total_migrated : ℕ := 118

/-- The number of bird families that flew to Africa -/
def africa_migrated : ℕ := 38

/-- The number of bird families that flew to Asia -/
def asia_migrated : ℕ := 80

/-- Theorem: The total number of bird families that migrated is equal to the sum of those that flew to Africa and Asia -/
theorem total_migration :
  total_migrated = africa_migrated + asia_migrated := by
sorry

end NUMINAMATH_CALUDE_total_migration_l2712_271221


namespace NUMINAMATH_CALUDE_repeating_decimal_difference_l2712_271242

theorem repeating_decimal_difference : 
  (6 : ℚ) / 11 - 54 / 100 = 6 / 1100 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_difference_l2712_271242


namespace NUMINAMATH_CALUDE_binomial_expectation_variance_relation_l2712_271215

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: If 3E(X) = 10D(X) for a binomial random variable X, then p = 0.7 -/
theorem binomial_expectation_variance_relation (X : BinomialRV) 
  (h : 3 * expectation X = 10 * variance X) : X.p = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expectation_variance_relation_l2712_271215


namespace NUMINAMATH_CALUDE_inserted_sequence_theorem_l2712_271254

/-- Given a sequence, insert_between inserts n elements between each pair of adjacent elements -/
def insert_between (seq : ℕ → α) (n : ℕ) : ℕ → α :=
  λ k => if k % (n + 1) = 0 then seq (k / (n + 1) + 1) else seq (k / (n + 1) + 1)

theorem inserted_sequence_theorem (original_seq : ℕ → α) :
  (insert_between original_seq 3) 69 = original_seq 18 := by
  sorry

end NUMINAMATH_CALUDE_inserted_sequence_theorem_l2712_271254


namespace NUMINAMATH_CALUDE_or_and_not_implies_false_and_true_l2712_271235

theorem or_and_not_implies_false_and_true (p q : Prop) :
  (p ∨ q) → (¬p) → (¬p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_or_and_not_implies_false_and_true_l2712_271235


namespace NUMINAMATH_CALUDE_prime_sum_implies_prime_exponent_l2712_271269

theorem prime_sum_implies_prime_exponent (p d : ℕ) : 
  Prime p → p = (10^d - 1) / 9 → Prime d := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_implies_prime_exponent_l2712_271269


namespace NUMINAMATH_CALUDE_shaded_triangle_area_l2712_271226

/-- Given a rectangle with sides 12 units long and a square with sides 4 units long
    placed in one corner of the rectangle, the area of the triangle formed by
    the diagonal of the rectangle and two sides of the rectangle is 54 square units. -/
theorem shaded_triangle_area (rectangle_side : ℝ) (square_side : ℝ) : 
  rectangle_side = 12 →
  square_side = 4 →
  let triangle_base := rectangle_side - (rectangle_side - square_side) * (square_side / rectangle_side)
  let triangle_height := rectangle_side
  (1/2) * triangle_base * triangle_height = 54 := by
sorry

end NUMINAMATH_CALUDE_shaded_triangle_area_l2712_271226


namespace NUMINAMATH_CALUDE_initial_men_count_l2712_271277

/-- Represents the work scenario with given parameters -/
structure WorkScenario where
  men : ℕ
  hoursPerDay : ℕ
  depth : ℕ

/-- Calculates the work done in a given scenario -/
def workDone (scenario : WorkScenario) : ℕ :=
  scenario.men * scenario.hoursPerDay * scenario.depth

theorem initial_men_count : ∃ (initialMen : ℕ),
  let scenario1 := WorkScenario.mk initialMen 8 30
  let scenario2 := WorkScenario.mk (initialMen + 55) 6 50
  workDone scenario1 = workDone scenario2 ∧ initialMen = 275 := by
  sorry

#check initial_men_count

end NUMINAMATH_CALUDE_initial_men_count_l2712_271277


namespace NUMINAMATH_CALUDE_min_cost_is_84_l2712_271246

/-- Represents a salon with prices for haircut, facial cleaning, and nails --/
structure Salon where
  haircut : ℕ
  facial : ℕ
  nails : ℕ

/-- Calculates the total cost for a salon --/
def totalCost (s : Salon) : ℕ := s.haircut + s.facial + s.nails

/-- The three salons with their respective prices --/
def gustranSalon : Salon := ⟨45, 22, 30⟩
def barbarasShop : Salon := ⟨30, 28, 40⟩
def fancySalon : Salon := ⟨34, 30, 20⟩

/-- Theorem stating that the minimum total cost among the three salons is 84 --/
theorem min_cost_is_84 : 
  min (totalCost gustranSalon) (min (totalCost barbarasShop) (totalCost fancySalon)) = 84 := by
  sorry


end NUMINAMATH_CALUDE_min_cost_is_84_l2712_271246


namespace NUMINAMATH_CALUDE_last_number_proof_l2712_271262

theorem last_number_proof (A B C D : ℝ) 
  (h1 : (A + B + C) / 3 = 6)
  (h2 : (B + C + D) / 3 = 3)
  (h3 : A + D = 13) :
  D = 2 := by
sorry

end NUMINAMATH_CALUDE_last_number_proof_l2712_271262


namespace NUMINAMATH_CALUDE_polynomial_form_l2712_271289

/-- A polynomial that satisfies the given condition -/
noncomputable def satisfying_polynomial (P : ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), P (a + b - 2*c) + P (b + c - 2*a) + P (c + a - 2*b) = 
                  3*P (a - b) + 3*P (b - c) + 3*P (c - a)

/-- The theorem stating the form of polynomials satisfying the condition -/
theorem polynomial_form (P : ℝ → ℝ) (hP : satisfying_polynomial P) :
  ∃ (a b : ℝ), ∀ x, P x = a * x^2 + b * x :=
sorry

end NUMINAMATH_CALUDE_polynomial_form_l2712_271289


namespace NUMINAMATH_CALUDE_sector_angle_l2712_271219

/-- Given a circular sector with circumference 8 and area 4, 
    prove that the central angle in radians is 2. -/
theorem sector_angle (r : ℝ) (α : ℝ) 
  (h_circumference : α * r + 2 * r = 8) 
  (h_area : (1 / 2) * α * r^2 = 4) : 
  α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l2712_271219


namespace NUMINAMATH_CALUDE_range_of_f_is_real_l2712_271260

-- Define the function f
def f (x : ℝ) : ℝ := -4 * x + 5

-- Theorem stating that the range of f is ℝ
theorem range_of_f_is_real : Set.range f = Set.univ :=
sorry

end NUMINAMATH_CALUDE_range_of_f_is_real_l2712_271260


namespace NUMINAMATH_CALUDE_orthocenter_locus_l2712_271256

noncomputable section

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The orthocenter of a triangle --/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Check if a point is inside a circle --/
def is_inside (p : ℝ × ℝ) (c : Circle) : Prop := sorry

/-- Check if a triangle is inscribed in a circle --/
def is_inscribed (t : Triangle) (c : Circle) : Prop := sorry

/-- The theorem stating the locus of orthocenters --/
theorem orthocenter_locus (c : Circle) :
  ∀ t : Triangle, is_inscribed t c →
    is_inside (orthocenter t) { center := c.center, radius := 3 * c.radius } :=
sorry

end NUMINAMATH_CALUDE_orthocenter_locus_l2712_271256


namespace NUMINAMATH_CALUDE_sine_of_pi_thirds_minus_two_theta_l2712_271234

theorem sine_of_pi_thirds_minus_two_theta (θ : ℝ) 
  (h : Real.tan (θ + π / 12) = 2) : 
  Real.sin (π / 3 - 2 * θ) = -3 / 5 := by
sorry

end NUMINAMATH_CALUDE_sine_of_pi_thirds_minus_two_theta_l2712_271234


namespace NUMINAMATH_CALUDE_prime_iff_divides_factorial_plus_one_power_of_n_iff_two_or_three_l2712_271267

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def is_power_of (a n : ℕ) : Prop := ∃ k : ℕ, a = n^k

theorem prime_iff_divides_factorial_plus_one (n : ℕ) (h : n ≥ 2) :
  is_prime n ↔ n ∣ (factorial (n - 1) + 1) :=
sorry

theorem power_of_n_iff_two_or_three (n : ℕ) (h : n ≥ 2) :
  is_power_of (factorial (n - 1) + 1) n ↔ n = 2 ∨ n = 3 :=
sorry

end NUMINAMATH_CALUDE_prime_iff_divides_factorial_plus_one_power_of_n_iff_two_or_three_l2712_271267


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2712_271210

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℤ, ∃ k : ℤ, x^3 + p*x + q = 3*k) ↔ 
  (∃ m : ℤ, p = 3*m + 2) ∧ (∃ n : ℤ, q = 3*n) := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2712_271210


namespace NUMINAMATH_CALUDE_first_term_of_geometric_series_l2712_271251

/-- The first term of an infinite geometric series with common ratio 1/4 and sum 80 is 60. -/
theorem first_term_of_geometric_series : ∀ (a : ℝ),
  (∑' n, a * (1/4)^n = 80) → a = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_series_l2712_271251


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l2712_271266

/-- The volume of a tetrahedron with skew edges -/
theorem tetrahedron_volume (a b d θ : ℝ) (ha : a > 0) (hb : b > 0) (hd : d > 0) (hθ : 0 < θ ∧ θ < π) :
  ∃ (V : ℝ), V = (1/6) * a * b * d * Real.sin θ ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l2712_271266


namespace NUMINAMATH_CALUDE_concert_revenue_proof_l2712_271214

/-- Calculates the total revenue of a concert given ticket prices and attendance numbers. -/
def concertRevenue (adultPrice : ℕ) (adultAttendance : ℕ) (childAttendance : ℕ) : ℕ :=
  adultPrice * adultAttendance + (adultPrice / 2) * childAttendance

/-- Proves that the total revenue of the concert is $5122 given the specified conditions. -/
theorem concert_revenue_proof :
  concertRevenue 26 183 28 = 5122 := by
  sorry

#eval concertRevenue 26 183 28

end NUMINAMATH_CALUDE_concert_revenue_proof_l2712_271214


namespace NUMINAMATH_CALUDE_terminating_decimal_count_l2712_271249

theorem terminating_decimal_count : 
  let n_range := Finset.range 449
  let divisible_by_nine := n_range.filter (λ n => (n + 1) % 9 = 0)
  divisible_by_nine.card = 49 := by
  sorry

end NUMINAMATH_CALUDE_terminating_decimal_count_l2712_271249


namespace NUMINAMATH_CALUDE_sum_increases_l2712_271216

/-- A set of 30 distinct positive real numbers -/
def M : Finset ℝ :=
  sorry

/-- The sum of the first n elements of M -/
def A (n : ℕ) : ℝ :=
  sorry

theorem sum_increases (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 29) : A (n + 1) > A n := by
  sorry

end NUMINAMATH_CALUDE_sum_increases_l2712_271216


namespace NUMINAMATH_CALUDE_base_7_representation_of_864_base_7_correctness_l2712_271259

/-- Converts a natural number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 7 to a natural number -/
def fromBase7 (digits : List ℕ) : ℕ :=
  sorry

theorem base_7_representation_of_864 :
  toBase7 864 = [2, 3, 4, 3] :=
sorry

theorem base_7_correctness :
  fromBase7 [2, 3, 4, 3] = 864 :=
sorry

end NUMINAMATH_CALUDE_base_7_representation_of_864_base_7_correctness_l2712_271259


namespace NUMINAMATH_CALUDE_certain_number_value_l2712_271245

theorem certain_number_value (a x : ℝ) : 
  (-6 * a^2 = x * (4 * a + 2)) → ((-6 : ℝ) = x * 6) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l2712_271245


namespace NUMINAMATH_CALUDE_max_product_with_sum_2016_l2712_271204

theorem max_product_with_sum_2016 :
  ∀ x y : ℤ, x + y = 2016 → x * y ≤ 1016064 := by
  sorry

end NUMINAMATH_CALUDE_max_product_with_sum_2016_l2712_271204


namespace NUMINAMATH_CALUDE_complement_of_M_l2712_271293

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 - 2*x ≤ 0}

-- State the theorem
theorem complement_of_M (x : ℝ) : x ∈ (Set.univ \ M) ↔ x < 0 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_l2712_271293


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_5_l2712_271244

open Complex

theorem complex_modulus_sqrt_5 (a b : ℝ) (i : ℂ) (h : i * i = -1) :
  (a - 2 * i) * i = b - i →
  Complex.abs (a + b * i) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_5_l2712_271244


namespace NUMINAMATH_CALUDE_bus_capacity_is_180_l2712_271276

/-- Represents the seating capacity of a double-decker bus -/
def double_decker_bus_capacity : ℕ :=
  let lower_left := 15 * 3
  let lower_right := 12 * 3
  let lower_back := 9
  let upper_left := 20 * 2
  let upper_right := 20 * 2
  let jump_seats := 4 * 1
  let emergency := 6
  lower_left + lower_right + lower_back + upper_left + upper_right + jump_seats + emergency

/-- Theorem stating the total seating capacity of the double-decker bus -/
theorem bus_capacity_is_180 : double_decker_bus_capacity = 180 := by
  sorry

#eval double_decker_bus_capacity

end NUMINAMATH_CALUDE_bus_capacity_is_180_l2712_271276


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l2712_271229

-- Problem 1
theorem factorization_problem_1 (a : ℝ) : a^3 - 16*a = a*(a+4)*(a-4) := by sorry

-- Problem 2
theorem factorization_problem_2 (x : ℝ) : (x-2)*(x-4)+1 = (x-3)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l2712_271229


namespace NUMINAMATH_CALUDE_sqrt_product_equals_27_l2712_271224

theorem sqrt_product_equals_27 (x : ℝ) (h1 : x > 0) 
  (h2 : Real.sqrt (12 * x) * Real.sqrt (18 * x) * Real.sqrt (6 * x) * Real.sqrt (9 * x) = 27) : 
  x = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_27_l2712_271224


namespace NUMINAMATH_CALUDE_fish_price_calculation_l2712_271275

theorem fish_price_calculation (discount_rate : ℝ) (discounted_price : ℝ) (package_weight : ℝ) : 
  discount_rate = 0.6 →
  discounted_price = 4.5 →
  package_weight = 0.75 →
  (discounted_price / package_weight) / (1 - discount_rate) = 15 := by
sorry

end NUMINAMATH_CALUDE_fish_price_calculation_l2712_271275


namespace NUMINAMATH_CALUDE_area_implies_m_value_existence_implies_a_range_l2712_271208

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Theorem for part (1)
theorem area_implies_m_value (m : ℝ) (h1 : m > 3) 
  (h2 : (1/2) * ((m - 1)/2 - (-(m + 1)/2) + 3) * (m - 3) = 7/2) : 
  m = 4 := by sorry

-- Theorem for part (2)
theorem existence_implies_a_range (a : ℝ) 
  (h : ∃ x ∈ Set.Icc 0 2, f x ≥ |a - 3|) :
  -2 ≤ a ∧ a ≤ 8 := by sorry

end NUMINAMATH_CALUDE_area_implies_m_value_existence_implies_a_range_l2712_271208


namespace NUMINAMATH_CALUDE_yogurt_combinations_l2712_271273

theorem yogurt_combinations (flavors : Nat) (toppings : Nat) : 
  flavors = 5 → toppings = 7 → 
  flavors * (1 + toppings.choose 1 + toppings.choose 2) = 145 := by
sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l2712_271273


namespace NUMINAMATH_CALUDE_yule_log_surface_area_increase_l2712_271236

/-- Proves that cutting a cylindrical Yule log into 9 slices increases its surface area by 100π -/
theorem yule_log_surface_area_increase :
  let h : ℝ := 10  -- height of the log
  let d : ℝ := 5   -- diameter of the log
  let n : ℕ := 9   -- number of slices
  let r : ℝ := d / 2  -- radius of the log
  let original_surface_area : ℝ := 2 * π * r * h + 2 * π * r^2
  let slice_height : ℝ := h / n
  let slice_surface_area : ℝ := 2 * π * r * slice_height + 2 * π * r^2
  let total_sliced_surface_area : ℝ := n * slice_surface_area
  let surface_area_increase : ℝ := total_sliced_surface_area - original_surface_area
  surface_area_increase = 100 * π := by
  sorry

end NUMINAMATH_CALUDE_yule_log_surface_area_increase_l2712_271236


namespace NUMINAMATH_CALUDE_planted_fraction_for_specific_plot_l2712_271252

/-- Represents a right triangle plot with an unplanted square at the right angle --/
structure PlotWithUnplantedSquare where
  leg1 : ℝ
  leg2 : ℝ
  unplanted_square_side : ℝ
  shortest_distance_to_hypotenuse : ℝ

/-- Calculates the fraction of the plot that is planted --/
def planted_fraction (plot : PlotWithUnplantedSquare) : ℝ := by sorry

/-- Theorem stating the planted fraction for the given plot dimensions --/
theorem planted_fraction_for_specific_plot :
  let plot : PlotWithUnplantedSquare := {
    leg1 := 5,
    leg2 := 12,
    unplanted_square_side := 3 * 7 / 5,
    shortest_distance_to_hypotenuse := 3
  }
  planted_fraction plot = 412 / 1000 := by sorry

end NUMINAMATH_CALUDE_planted_fraction_for_specific_plot_l2712_271252


namespace NUMINAMATH_CALUDE_square_of_product_l2712_271258

theorem square_of_product (x : ℝ) : (3 * x)^2 = 9 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_product_l2712_271258


namespace NUMINAMATH_CALUDE_max_tickets_jane_can_buy_l2712_271239

theorem max_tickets_jane_can_buy (ticket_cost : ℕ) (service_charge : ℕ) (budget : ℕ) :
  ticket_cost = 15 →
  service_charge = 10 →
  budget = 120 →
  ∃ (n : ℕ), n = 7 ∧ 
    n * ticket_cost + service_charge ≤ budget ∧
    ∀ (m : ℕ), m * ticket_cost + service_charge ≤ budget → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_max_tickets_jane_can_buy_l2712_271239


namespace NUMINAMATH_CALUDE_similar_triangle_coordinates_l2712_271213

/-- Given two points A and B in a Cartesian coordinate system, with O as the origin and center of
    similarity, and triangle A'B'O similar to triangle ABO with a similarity ratio of 1:2,
    prove that the coordinates of B' are (-3, -2). -/
theorem similar_triangle_coordinates (A B : ℝ × ℝ) (h_A : A = (-4, 2)) (h_B : B = (-6, -4)) :
  let O : ℝ × ℝ := (0, 0)
  let similarity_ratio : ℝ := 1 / 2
  let B' : ℝ × ℝ := (similarity_ratio * B.1, similarity_ratio * B.2)
  B' = (-3, -2) := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_coordinates_l2712_271213


namespace NUMINAMATH_CALUDE_ratio_equation_solution_product_l2712_271290

theorem ratio_equation_solution_product (x : ℝ) : 
  (((x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → x = 0) ∧ 
  (∃ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_product_l2712_271290


namespace NUMINAMATH_CALUDE_sum_of_max_pairs_nonnegative_l2712_271263

theorem sum_of_max_pairs_nonnegative 
  (a b c d : ℝ) 
  (h : a + b + c + d = 0) : 
  max a b + max a c + max a d + max b c + max b d + max c d ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_pairs_nonnegative_l2712_271263


namespace NUMINAMATH_CALUDE_unique_three_digit_number_divisible_by_nine_l2712_271232

theorem unique_three_digit_number_divisible_by_nine :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
  n % 10 = 5 ∧ 
  (n / 100) % 10 = 3 ∧ 
  n % 9 = 0 ∧
  n = 315 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_divisible_by_nine_l2712_271232


namespace NUMINAMATH_CALUDE_classroom_gpa_l2712_271203

theorem classroom_gpa (total_students : ℕ) (gpa1 gpa2 gpa3 : ℚ) 
  (h1 : total_students = 60)
  (h2 : gpa1 = 54)
  (h3 : gpa2 = 48)
  (h4 : gpa3 = 45)
  (h5 : (total_students : ℚ) / 3 * gpa1 + (total_students : ℚ) / 4 * gpa2 + 
        (total_students - (total_students / 3 + total_students / 4) : ℚ) * gpa3 = 
        total_students * 48.75) : 
  (((total_students : ℚ) / 3 * gpa1 + (total_students : ℚ) / 4 * gpa2 + 
    (total_students - (total_students / 3 + total_students / 4) : ℚ) * gpa3) / total_students) = 48.75 :=
by sorry

end NUMINAMATH_CALUDE_classroom_gpa_l2712_271203


namespace NUMINAMATH_CALUDE_bacteria_count_after_six_hours_l2712_271298

/-- The number of bacteria at time n -/
def bacteria : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => bacteria (n + 1) + bacteria n

/-- The time in half-hour units after which we want to count bacteria -/
def target_time : ℕ := 12

theorem bacteria_count_after_six_hours :
  bacteria target_time = 233 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_count_after_six_hours_l2712_271298
