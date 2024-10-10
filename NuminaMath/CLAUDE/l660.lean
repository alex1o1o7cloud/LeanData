import Mathlib

namespace max_value_on_circle_l660_66094

theorem max_value_on_circle (x y : ℝ) : 
  (x - 1)^2 + y^2 = 4 → 
  ∃ b : ℝ, (∀ x' y' : ℝ, (x' - 1)^2 + y'^2 = 4 → 2*x' + y'^2 ≤ b) ∧ 
           (∃ x'' y'' : ℝ, (x'' - 1)^2 + y''^2 = 4 ∧ 2*x'' + y''^2 = b) ∧
           b = 7 :=
by sorry

end max_value_on_circle_l660_66094


namespace initial_oranges_count_l660_66021

/-- Proves that the initial number of oranges in a bin was 50, given the described changes and final count. -/
theorem initial_oranges_count (initial thrown_away added final : ℕ) 
  (h1 : thrown_away = 40)
  (h2 : added = 24)
  (h3 : final = 34)
  (h4 : initial - thrown_away + added = final) : initial = 50 := by
  sorry

end initial_oranges_count_l660_66021


namespace darrel_nickels_l660_66033

def quarters : ℕ := 76
def dimes : ℕ := 85
def pennies : ℕ := 150
def fee_percentage : ℚ := 10 / 100
def amount_after_fee : ℚ := 27

def quarter_value : ℚ := 25 / 100
def dime_value : ℚ := 10 / 100
def nickel_value : ℚ := 5 / 100
def penny_value : ℚ := 1 / 100

theorem darrel_nickels :
  let total_before_fee := amount_after_fee / (1 - fee_percentage)
  let known_coins_value := quarters * quarter_value + dimes * dime_value + pennies * penny_value
  let nickel_value_sum := total_before_fee - known_coins_value
  (nickel_value_sum / nickel_value : ℚ) = 20 := by sorry

end darrel_nickels_l660_66033


namespace complex_magnitude_l660_66011

theorem complex_magnitude (w : ℂ) (h : w^2 = 48 - 14*I) : Complex.abs w = 5 * Real.sqrt 2 := by
  sorry

end complex_magnitude_l660_66011


namespace bingley_has_four_bracelets_l660_66068

/-- The number of bracelets Bingley has remaining after the exchanges -/
def bingley_remaining_bracelets : ℕ :=
  let bingley_initial : ℕ := 5
  let kelly_initial : ℕ := 16
  let kelly_gives : ℕ := kelly_initial / 4 / 3
  let bingley_after_receiving : ℕ := bingley_initial + kelly_gives
  let bingley_gives : ℕ := bingley_after_receiving / 3
  bingley_after_receiving - bingley_gives

/-- Theorem stating that Bingley has 4 bracelets remaining -/
theorem bingley_has_four_bracelets : bingley_remaining_bracelets = 4 := by
  sorry

end bingley_has_four_bracelets_l660_66068


namespace bridge_length_calculation_bridge_length_proof_l660_66017

theorem bridge_length_calculation (train_length : Real) (train_speed_kmh : Real) (time_to_pass : Real) : Real :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * time_to_pass
  let bridge_length := total_distance - train_length
  bridge_length

theorem bridge_length_proof :
  bridge_length_calculation 250 35 41.142857142857146 = 150 := by
  sorry

end bridge_length_calculation_bridge_length_proof_l660_66017


namespace system_solution_l660_66057

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x^2 - 5*x*y + 6*y^2 = 0
def equation2 (x y : ℝ) : Prop := x^2 + y^2 + x - 11*y - 2 = 0

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {(-2/5, -1/5), (4, 2), (-3/5, -1/5), (3, 1)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solution_set :=
sorry

end system_solution_l660_66057


namespace largest_number_l660_66025

theorem largest_number (a b c d e : ℚ) 
  (sum1 : a + b + c + d = 210)
  (sum2 : a + b + c + e = 230)
  (sum3 : a + b + d + e = 250)
  (sum4 : a + c + d + e = 270)
  (sum5 : b + c + d + e = 290) :
  max a (max b (max c (max d e))) = 102.5 := by
sorry

end largest_number_l660_66025


namespace concatenated_numbers_problem_l660_66013

theorem concatenated_numbers_problem : 
  ∃! (x y : ℕ), 
    100 ≤ x ∧ x < 1000 ∧ 
    100 ≤ y ∧ y < 1000 ∧ 
    1000 * x + y = 7 * x * y ∧
    x = 143 ∧ y = 143 := by
  sorry

end concatenated_numbers_problem_l660_66013


namespace cat_bird_hunting_l660_66024

theorem cat_bird_hunting (day_catch : ℕ) (night_catch : ℕ) : 
  day_catch = 8 → night_catch = 2 * day_catch → day_catch + night_catch = 24 := by
  sorry

end cat_bird_hunting_l660_66024


namespace fractional_expression_transformation_l660_66030

theorem fractional_expression_transformation (x : ℝ) :
  let A : ℝ → ℝ := λ x => x^2 - 2*x
  x / (x + 2) = A x / (x^2 - 4) :=
by sorry

end fractional_expression_transformation_l660_66030


namespace quadratic_factorization_l660_66018

theorem quadratic_factorization (a b : ℤ) :
  (∀ x : ℝ, 25 * x^2 - 155 * x - 150 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = -66 := by
sorry

end quadratic_factorization_l660_66018


namespace lunch_cost_proof_l660_66014

/-- Proves that under given conditions, one person's lunch cost is $45 --/
theorem lunch_cost_proof (cost_A cost_R cost_J : ℚ) : 
  cost_A = (2/3) * cost_R →
  cost_R = cost_J →
  cost_A + cost_R + cost_J = 120 →
  cost_J = 45 := by
sorry

end lunch_cost_proof_l660_66014


namespace solve_percentage_equation_l660_66055

theorem solve_percentage_equation : ∃ x : ℝ, 0.65 * x = 0.20 * 487.50 ∧ x = 150 := by
  sorry

end solve_percentage_equation_l660_66055


namespace three_from_eight_committee_l660_66067

/-- The number of ways to select k items from n items without replacement and where order doesn't matter. -/
def combinations (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

/-- Theorem: There are 56 ways to select 3 people from a group of 8 people where order doesn't matter. -/
theorem three_from_eight_committee : combinations 8 3 = 56 := by
  sorry

end three_from_eight_committee_l660_66067


namespace min_red_chips_l660_66049

/-- Represents the number of chips of each color -/
structure ChipCount where
  red : Nat
  blue : Nat

/-- Checks if a number is prime -/
def isPrime (n : Nat) : Prop := sorry

theorem min_red_chips :
  ∀ (chips : ChipCount),
  chips.red + chips.blue = 70 →
  isPrime (chips.red + 2 * chips.blue) →
  chips.red ≥ 69 :=
by sorry

end min_red_chips_l660_66049


namespace sum_of_999_and_999_l660_66089

theorem sum_of_999_and_999 : 999 + 999 = 1998 := by sorry

end sum_of_999_and_999_l660_66089


namespace brendan_total_wins_l660_66095

/-- Represents the number of matches won in each round of the kickboxing competition -/
structure KickboxingResults where
  round1_wins : Nat
  round2_wins : Nat
  round3_wins : Nat
  round4_wins : Nat

/-- Calculates the total number of matches won across all rounds -/
def total_wins (results : KickboxingResults) : Nat :=
  results.round1_wins + results.round2_wins + results.round3_wins + results.round4_wins

/-- Theorem stating that Brendan's total wins in the kickboxing competition is 18 -/
theorem brendan_total_wins :
  ∃ (results : KickboxingResults),
    results.round1_wins = 6 ∧
    results.round2_wins = 4 ∧
    results.round3_wins = 3 ∧
    results.round4_wins = 5 ∧
    total_wins results = 18 := by
  sorry

end brendan_total_wins_l660_66095


namespace liar_knight_difference_district_A_l660_66005

/-- Represents the number of residents in the city -/
def total_residents : ℕ := 50

/-- Represents the number of questions asked -/
def num_questions : ℕ := 4

/-- Represents the number of affirmative answers given by a knight -/
def knight_affirmative : ℕ := 1

/-- Represents the number of affirmative answers given by a liar -/
def liar_affirmative : ℕ := 3

/-- Represents the total number of affirmative answers given -/
def total_affirmative : ℕ := 290

/-- Theorem stating the difference between liars and knights in District A -/
theorem liar_knight_difference_district_A :
  ∃ (knights_A liars_A : ℕ),
    knights_A + liars_A ≤ total_residents ∧
    knights_A * knight_affirmative * num_questions +
    liars_A * liar_affirmative * num_questions ≤ total_affirmative ∧
    liars_A = knights_A + 3 := by
  sorry

end liar_knight_difference_district_A_l660_66005


namespace arithmetic_puzzle_2016_l660_66009

/-- Represents a basic arithmetic operation --/
inductive Operation
  | Add
  | Subtract
  | Multiply
  | Divide

/-- Represents an arithmetic expression --/
inductive Expr
  | Num (n : ℕ)
  | Op (op : Operation) (e1 e2 : Expr)
  | Paren (e : Expr)

/-- Evaluates an arithmetic expression --/
def eval : Expr → ℚ
  | Expr.Num n => n
  | Expr.Op Operation.Add e1 e2 => eval e1 + eval e2
  | Expr.Op Operation.Subtract e1 e2 => eval e1 - eval e2
  | Expr.Op Operation.Multiply e1 e2 => eval e1 * eval e2
  | Expr.Op Operation.Divide e1 e2 => eval e1 / eval e2
  | Expr.Paren e => eval e

/-- Checks if an expression uses digits 1 through 9 in sequence --/
def usesDigitsInSequence : Expr → Bool
  | _ => sorry  -- Implementation omitted for brevity

theorem arithmetic_puzzle_2016 :
  ∃ (e : Expr), usesDigitsInSequence e ∧ eval e = 2016 := by
  sorry


end arithmetic_puzzle_2016_l660_66009


namespace triangle_angles_from_bisector_ratio_l660_66059

theorem triangle_angles_from_bisector_ratio :
  ∀ (α β γ : ℝ),
  (α > 0) → (β > 0) → (γ > 0) →
  (α + β + γ = 180) →
  (∃ (k : ℝ), k > 0 ∧
    (α/2 + β/2 = 37*k) ∧
    (β/2 + γ/2 = 41*k) ∧
    (γ/2 + α/2 = 42*k)) →
  (α = 72 ∧ β = 66 ∧ γ = 42) :=
by sorry

end triangle_angles_from_bisector_ratio_l660_66059


namespace mary_picked_12kg_l660_66088

/-- Given three people picking chestnuts, prove that one person picked 12 kg. -/
theorem mary_picked_12kg (peter lucy mary : ℕ) : 
  mary = 2 * peter →  -- Mary picked twice as much as Peter
  lucy = peter + 2 →  -- Lucy picked 2 kg more than Peter
  peter + mary + lucy = 26 →  -- Total amount picked is 26 kg
  mary = 12 := by
sorry

end mary_picked_12kg_l660_66088


namespace integer_less_than_sqrt_23_l660_66041

theorem integer_less_than_sqrt_23 : ∃ n : ℤ, (n : ℝ) < Real.sqrt 23 := by
  sorry

end integer_less_than_sqrt_23_l660_66041


namespace sum_of_circle_areas_l660_66061

/-- Represents a right triangle with side lengths 6, 8, and 10 -/
structure Triangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (is_right : a^2 + b^2 = c^2)
  (side_lengths : a = 6 ∧ b = 8 ∧ c = 10)

/-- Represents three mutually externally tangent circles -/
structure TangentCircles :=
  (r₁ : ℝ) (r₂ : ℝ) (r₃ : ℝ)
  (tangent_condition : r₁ + r₂ = 6 ∧ r₁ + r₃ = 8 ∧ r₂ + r₃ = 10)

/-- The sum of the areas of three mutually externally tangent circles
    centered at the vertices of a 6-8-10 right triangle is 56π -/
theorem sum_of_circle_areas (t : Triangle) (c : TangentCircles) :
  π * (c.r₁^2 + c.r₂^2 + c.r₃^2) = 56 * π :=
sorry

end sum_of_circle_areas_l660_66061


namespace jersey_tshirt_cost_difference_l660_66051

/-- The cost of a jersey in dollars -/
def jersey_cost : ℕ := 115

/-- The cost of a t-shirt in dollars -/
def tshirt_cost : ℕ := 25

/-- The number of t-shirts sold during the game -/
def tshirts_sold : ℕ := 113

/-- The number of jerseys sold during the game -/
def jerseys_sold : ℕ := 78

theorem jersey_tshirt_cost_difference : jersey_cost - tshirt_cost = 90 := by
  sorry

end jersey_tshirt_cost_difference_l660_66051


namespace banana_sharing_l660_66016

theorem banana_sharing (jefferson_bananas : ℕ) (walter_bananas : ℕ) : 
  jefferson_bananas = 56 →
  walter_bananas = jefferson_bananas - (jefferson_bananas / 4) →
  (jefferson_bananas + walter_bananas) / 2 = 49 :=
by
  sorry

end banana_sharing_l660_66016


namespace condition_necessary_not_sufficient_l660_66031

theorem condition_necessary_not_sufficient (x y : ℝ) :
  (x + y > 3 → (x > 1 ∨ y > 2)) ∧
  ¬((x > 1 ∨ y > 2) → x + y > 3) :=
by sorry

end condition_necessary_not_sufficient_l660_66031


namespace probability_specific_pair_from_six_l660_66023

/-- The probability of selecting a specific pair when choosing 2 from 6 -/
theorem probability_specific_pair_from_six (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 2) :
  (1 : ℚ) / (n.choose k) = 1 / 15 := by
  sorry

end probability_specific_pair_from_six_l660_66023


namespace units_digit_of_2_to_2010_l660_66002

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the cycle of units digits for powers of 2
def powerOfTwoCycle : List ℕ := [2, 4, 8, 6]

-- Theorem statement
theorem units_digit_of_2_to_2010 :
  unitsDigit (2^2010) = 4 := by
  sorry

end units_digit_of_2_to_2010_l660_66002


namespace inequality_range_l660_66022

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (3 : ℝ)^(x^2 - 2*a*x) > (1/3 : ℝ)^(x + 1)) → 
  -1/2 < a ∧ a < 3/2 := by
sorry

end inequality_range_l660_66022


namespace line_intersection_theorem_l660_66081

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the properties and relations
variable (skew : Line → Line → Prop)
variable (lies_on : Line → Plane → Prop)
variable (intersection : Plane → Plane → Line)
variable (intersects : Line → Line → Prop)

-- State the theorem
theorem line_intersection_theorem 
  (l₁ l₂ l : Line) (α β : Plane)
  (h1 : skew l₁ l₂)
  (h2 : lies_on l₁ α)
  (h3 : lies_on l₂ β)
  (h4 : l = intersection α β) :
  intersects l l₁ ∨ intersects l l₂ :=
sorry

end line_intersection_theorem_l660_66081


namespace sum_of_reciprocal_roots_l660_66097

theorem sum_of_reciprocal_roots (a b : ℝ) : 
  a ≠ b → 
  a^2 - 3*a - 1 = 0 → 
  b^2 - 3*b - 1 = 0 → 
  b/a + a/b = -11 := by
sorry

end sum_of_reciprocal_roots_l660_66097


namespace james_training_hours_l660_66069

/-- James' Olympic training schedule and yearly hours --/
theorem james_training_hours :
  (sessions_per_day : ℕ) →
  (hours_per_session : ℕ) →
  (training_days_per_week : ℕ) →
  (weeks_per_year : ℕ) →
  sessions_per_day = 2 →
  hours_per_session = 4 →
  training_days_per_week = 5 →
  weeks_per_year = 52 →
  sessions_per_day * hours_per_session * training_days_per_week * weeks_per_year = 2080 :=
by sorry

end james_training_hours_l660_66069


namespace rhombus_area_l660_66075

/-- A rhombus with diagonals of lengths 10 and 30 has an area of 150 -/
theorem rhombus_area (d₁ d₂ area : ℝ) (h₁ : d₁ = 10) (h₂ : d₂ = 30) 
    (h₃ : area = (d₁ * d₂) / 2) : area = 150 := by
  sorry

end rhombus_area_l660_66075


namespace total_edges_theorem_l660_66038

/-- A graph with the properties described in the problem -/
structure WonderGraph where
  n : ℕ  -- number of cities
  a : ℕ  -- number of roads
  connected : Bool  -- graph is connected
  at_most_one_edge : Bool  -- at most one edge between any two vertices
  indirect_path : Bool  -- indirect path exists between directly connected vertices

/-- The number of subgraphs with even degree vertices -/
def num_even_subgraphs (G : WonderGraph) : ℕ := sorry

/-- The total number of edges in all subgraphs with even degree vertices -/
def total_edges_in_even_subgraphs (G : WonderGraph) : ℕ := sorry

/-- Main theorem: The total number of edges in all subgraphs with even degree vertices is ar/2 -/
theorem total_edges_theorem (G : WonderGraph) :
  total_edges_in_even_subgraphs G = G.a * (num_even_subgraphs G) / 2 :=
sorry

end total_edges_theorem_l660_66038


namespace valid_points_characterization_l660_66073

def is_in_second_quadrant (x y : ℤ) : Prop := x < 0 ∧ y > 0

def satisfies_inequality (x y : ℤ) : Prop := y ≤ x + 4

def is_valid_point (x y : ℤ) : Prop :=
  is_in_second_quadrant x y ∧ satisfies_inequality x y

def valid_points : Set (ℤ × ℤ) :=
  {(-1, 1), (-1, 2), (-1, 3), (-2, 1), (-2, 2), (-3, 1)}

theorem valid_points_characterization :
  ∀ x y : ℤ, is_valid_point x y ↔ (x, y) ∈ valid_points := by sorry

end valid_points_characterization_l660_66073


namespace sin_cos_identity_l660_66050

theorem sin_cos_identity : 
  Real.sin (50 * π / 180) * Real.cos (20 * π / 180) - 
  Real.sin (40 * π / 180) * Real.cos (70 * π / 180) = 1/2 := by
  sorry

end sin_cos_identity_l660_66050


namespace student_arrangements_l660_66084

/-- The number of students in the row -/
def n : ℕ := 7

/-- The number of arrangements where A and B must stand together -/
def arrangements_together : ℕ := 1440

/-- The number of arrangements where A is not at the head and B is not at the end -/
def arrangements_not_head_end : ℕ := 3720

/-- The number of arrangements where there is exactly one person between A and B -/
def arrangements_one_between : ℕ := 1200

/-- Theorem stating the correct number of arrangements for each situation -/
theorem student_arrangements :
  (arrangements_together = 1440) ∧
  (arrangements_not_head_end = 3720) ∧
  (arrangements_one_between = 1200) := by sorry

end student_arrangements_l660_66084


namespace expand_product_l660_66079

theorem expand_product (x : ℝ) : (x + 4) * (2 * x - 9) = 2 * x^2 - x - 36 := by
  sorry

end expand_product_l660_66079


namespace factor_implies_b_value_l660_66045

theorem factor_implies_b_value (a b : ℤ) : 
  (∃ (c : ℤ), (X^2 - 2*X - 1) * (a*X - c) = a*X^3 + b*X^2 + 2) → b = -6 :=
by sorry

end factor_implies_b_value_l660_66045


namespace problem_statement_l660_66092

theorem problem_statement (x y : ℝ) : (x + 1)^2 + |y - 2| = 0 → 2*x + 3*y = 4 := by
  sorry

end problem_statement_l660_66092


namespace traffic_class_multiple_l660_66012

theorem traffic_class_multiple (drunk_drivers : ℕ) (total_students : ℕ) (M : ℕ) : 
  drunk_drivers = 6 →
  total_students = 45 →
  total_students = drunk_drivers + (M * drunk_drivers - 3) →
  M = 7 := by
sorry

end traffic_class_multiple_l660_66012


namespace binomial_10_choose_3_l660_66007

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_choose_3_l660_66007


namespace sum_pqrs_equals_32_1_l660_66074

theorem sum_pqrs_equals_32_1 
  (p q r s : ℝ)
  (hp : p = 2)
  (hpq : p * q = 20)
  (hpqr : p * q * r = 202)
  (hpqrs : p * q * r * s = 2020) :
  p + q + r + s = 32.1 := by
sorry

end sum_pqrs_equals_32_1_l660_66074


namespace complex_equation_sum_l660_66077

theorem complex_equation_sum (a b : ℝ) :
  (a + 4 * Complex.I) * Complex.I = b + Complex.I →
  a + b = -3 := by
sorry

end complex_equation_sum_l660_66077


namespace solution_satisfies_system_l660_66028

theorem solution_satisfies_system :
  let eq1 (x y : ℝ) := y + Real.sqrt (y - 3 * x) + 3 * x = 12
  let eq2 (x y : ℝ) := y^2 + y - 3 * x - 9 * x^2 = 144
  (eq1 (-24) 72 ∧ eq2 (-24) 72) ∧
  (eq1 (-4/3) 12 ∧ eq2 (-4/3) 12) := by
  sorry

#check solution_satisfies_system

end solution_satisfies_system_l660_66028


namespace cylindrical_block_volume_l660_66085

/-- Represents a cylindrical iron block -/
structure CylindricalBlock where
  height : ℝ
  volume : ℝ

/-- Represents a frustum-shaped iron block -/
structure FrustumBlock where
  height : ℝ
  base_radius : ℝ

/-- Represents a container with a cylindrical and a frustum-shaped block -/
structure Container where
  cylindrical_block : CylindricalBlock
  frustum_block : FrustumBlock

/-- Theorem stating the volume of the cylindrical block in the container -/
theorem cylindrical_block_volume (container : Container) 
  (h1 : container.cylindrical_block.height = 3)
  (h2 : container.frustum_block.height = 3)
  (h3 : container.frustum_block.base_radius = container.frustum_block.base_radius) :
  container.cylindrical_block.volume = 15.42 := by
  sorry

end cylindrical_block_volume_l660_66085


namespace impossibleTransformation_l660_66066

/-- Represents the three possible colors of the sides of the 99-gon -/
inductive Color
| Red
| Blue
| Yellow

/-- Represents the coloring of the 99-gon -/
def Coloring := Fin 99 → Color

/-- The initial coloring of the 99-gon -/
def initialColoring : Coloring :=
  fun i => match i.val % 3 with
    | 0 => Color.Red
    | 1 => Color.Blue
    | _ => Color.Yellow

/-- The target coloring of the 99-gon -/
def targetColoring : Coloring :=
  fun i => match i.val % 3 with
    | 0 => Color.Blue
    | 1 => Color.Red
    | _ => if i.val == 98 then Color.Blue else Color.Yellow

/-- Checks if a coloring is valid (no adjacent sides have the same color) -/
def isValidColoring (c : Coloring) : Prop :=
  ∀ i : Fin 98, c i ≠ c (i.succ)

/-- Represents a single color change operation -/
def colorChange (c : Coloring) (i : Fin 99) (newColor : Color) : Coloring :=
  fun j => if j = i then newColor else c j

/-- Theorem stating the impossibility of transforming the initial coloring to the target coloring -/
theorem impossibleTransformation :
  ¬∃ (steps : List (Fin 99 × Color)),
    (steps.foldl (fun acc (i, col) => colorChange acc i col) initialColoring = targetColoring) ∧
    (∀ step ∈ steps, isValidColoring (colorChange (steps.foldl (fun acc (i, col) => colorChange acc i col) initialColoring) step.fst step.snd)) :=
sorry


end impossibleTransformation_l660_66066


namespace convex_polygon_with_equal_diagonals_l660_66048

/-- A convex polygon with n sides and all diagonals equal -/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 4
  all_diagonals_equal : Bool

/-- Theorem: If a convex n-gon (n ≥ 4) has all diagonals equal, then n is either 4 or 5 -/
theorem convex_polygon_with_equal_diagonals 
  {n : ℕ} (F : ConvexPolygon n) (h : F.all_diagonals_equal = true) : 
  n = 4 ∨ n = 5 := by
  sorry

end convex_polygon_with_equal_diagonals_l660_66048


namespace average_not_equal_given_l660_66096

def numbers : List ℝ := [1200, 1300, 1400, 1510, 1530, 1200]
def given_average : ℝ := 1380

theorem average_not_equal_given : (numbers.sum / numbers.length) ≠ given_average := by
  sorry

end average_not_equal_given_l660_66096


namespace max_sphere_radius_squared_l660_66043

/-- The maximum squared radius of a sphere fitting within two congruent right circular cones -/
theorem max_sphere_radius_squared (base_radius height intersection_distance : ℝ) 
  (hr : base_radius = 4)
  (hh : height = 10)
  (hi : intersection_distance = 4) : 
  ∃ (r : ℝ), r^2 = (528 - 32 * Real.sqrt 116) / 29 ∧ 
  ∀ (s : ℝ), s^2 ≤ (528 - 32 * Real.sqrt 116) / 29 := by
  sorry

#check max_sphere_radius_squared

end max_sphere_radius_squared_l660_66043


namespace circle_center_coordinate_sum_l660_66058

theorem circle_center_coordinate_sum :
  ∀ (x y : ℝ), x^2 + y^2 = 10*x - 4*y + 6 →
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 6 - 10*h + 4*k) ∧ h + k = 3 :=
by sorry

end circle_center_coordinate_sum_l660_66058


namespace series_sum_equals_eleven_twentieths_l660_66062

theorem series_sum_equals_eleven_twentieths : 
  (1 / 3 : ℚ) + (1 / 5 : ℚ) + (1 / 7 : ℚ) + (1 / 9 : ℚ) = 11 / 20 := by
  sorry

end series_sum_equals_eleven_twentieths_l660_66062


namespace part_a_part_b_part_c_l660_66035

/-- Rachel's jump length in cm -/
def rachel_jump : ℕ := 168

/-- Joel's jump length in cm -/
def joel_jump : ℕ := 120

/-- Mark's jump length in cm -/
def mark_jump : ℕ := 72

/-- Theorem for part (a) -/
theorem part_a (n : ℕ) : 
  n > 0 → 5 * rachel_jump = n * joel_jump → n = 7 := by sorry

/-- Theorem for part (b) -/
theorem part_b (r t : ℕ) : 
  r > 0 → t > 0 → 11 ≤ t → t ≤ 19 → r * joel_jump = t * mark_jump → r = 9 ∧ t = 15 := by sorry

/-- Theorem for part (c) -/
theorem part_c (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 → 
  a * rachel_jump = b * joel_jump → 
  b * joel_jump = c * mark_jump → 
  (∀ c' : ℕ, c' > 0 → c' * mark_jump = a * rachel_jump → c ≤ c') → 
  c = 35 := by sorry

end part_a_part_b_part_c_l660_66035


namespace arithmetic_sequence_ratio_l660_66070

/-- Given an arithmetic sequence with sum S_n = a n^2, prove a_5/d = 9/2 -/
theorem arithmetic_sequence_ratio (a : ℝ) (d : ℝ) (S : ℕ → ℝ) (a_seq : ℕ → ℝ) :
  d ≠ 0 →
  (∀ n : ℕ, S n = a * n^2) →
  (∀ n : ℕ, a_seq (n + 1) - a_seq n = d) →
  (∀ n : ℕ, S n = (n * (a_seq 1 + a_seq n)) / 2) →
  a_seq 5 / d = 9 / 2 :=
by sorry

end arithmetic_sequence_ratio_l660_66070


namespace largest_ball_on_specific_torus_l660_66004

/-- The radius of the largest spherical ball that can be placed atop a torus -/
def largest_ball_radius (torus_center : ℝ × ℝ × ℝ) (torus_radius : ℝ) : ℝ :=
  let (x, y, z) := torus_center
  2

/-- Theorem: The radius of the largest spherical ball on a specific torus is 2 -/
theorem largest_ball_on_specific_torus :
  largest_ball_radius (4, 0, 2) 2 = 2 := by
  sorry

end largest_ball_on_specific_torus_l660_66004


namespace ferris_wheel_capacity_l660_66072

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := 4

/-- The total number of people that can ride the wheel at the same time -/
def total_people : ℕ := 20

/-- The number of people each seat can hold -/
def people_per_seat : ℕ := total_people / num_seats

theorem ferris_wheel_capacity : people_per_seat = 5 := by
  sorry

end ferris_wheel_capacity_l660_66072


namespace correct_calculation_l660_66047

theorem correct_calculation (a b : ℝ) (h : a ≠ 0) : (a^2 + a*b) / a = a + b := by
  sorry

end correct_calculation_l660_66047


namespace tan_thirty_degrees_l660_66056

theorem tan_thirty_degrees : Real.tan (π / 6) = Real.sqrt 3 / 3 := by
  sorry

end tan_thirty_degrees_l660_66056


namespace reflect_P_across_x_axis_l660_66042

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point P -/
def P : ℝ × ℝ := (-2, 3)

theorem reflect_P_across_x_axis : 
  reflect_x P = (-2, -3) := by
  sorry

end reflect_P_across_x_axis_l660_66042


namespace paths_from_A_to_D_l660_66052

/-- The number of paths between two adjacent points -/
def paths_between_adjacent : ℕ := 2

/-- The number of direct paths from A to D -/
def direct_paths : ℕ := 1

/-- The total number of paths from A to D -/
def total_paths : ℕ := paths_between_adjacent^3 + direct_paths

/-- Theorem stating that the total number of paths from A to D is 9 -/
theorem paths_from_A_to_D : total_paths = 9 := by sorry

end paths_from_A_to_D_l660_66052


namespace terminal_side_in_first_or_third_quadrant_l660_66032

-- Define the angle α as a function of k
def α (k : ℤ) : Real := k * 180 + 45

-- Define a function to determine the quadrant of an angle
def inFirstOrThirdQuadrant (angle : Real) : Prop :=
  (0 < angle % 360 ∧ angle % 360 < 90) ∨ 
  (180 < angle % 360 ∧ angle % 360 < 270)

-- Theorem statement
theorem terminal_side_in_first_or_third_quadrant (k : ℤ) :
  inFirstOrThirdQuadrant (α k) := by sorry

end terminal_side_in_first_or_third_quadrant_l660_66032


namespace sum_of_factors_72_l660_66008

/-- Sum of positive factors of a natural number n -/
def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

/-- The theorem stating that the sum of positive factors of 72 is 195 -/
theorem sum_of_factors_72 : sum_of_factors 72 = 195 := by
  sorry

end sum_of_factors_72_l660_66008


namespace mike_owes_jennifer_l660_66054

theorem mike_owes_jennifer (rate : ℚ) (rooms : ℚ) (amount_owed : ℚ) : 
  rate = 13 / 3 → rooms = 8 / 5 → amount_owed = rate * rooms → amount_owed = 104 / 15 := by
  sorry

end mike_owes_jennifer_l660_66054


namespace vector_b_determination_l660_66040

def vector_a : ℝ × ℝ := (4, 3)

theorem vector_b_determination (b : ℝ × ℝ) 
  (h1 : (b.1 * vector_a.1 + b.2 * vector_a.2) / Real.sqrt (vector_a.1^2 + vector_a.2^2) = 4)
  (h2 : b.1 = 2) :
  b = (2, 4) := by
  sorry

end vector_b_determination_l660_66040


namespace subtract_point_six_from_forty_five_point_nine_l660_66060

theorem subtract_point_six_from_forty_five_point_nine : 45.9 - 0.6 = 45.3 := by
  sorry

end subtract_point_six_from_forty_five_point_nine_l660_66060


namespace inverse_proportion_problem_l660_66053

/-- Given that x and y are inversely proportional, prove that y = -49 when x = -8,
    given the conditions that x + y = 42 and x = 2y for some values of x and y. -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) 
    (h1 : x * y = k)  -- x and y are inversely proportional
    (h2 : ∃ (a b : ℝ), a + b = 42 ∧ a = 2 * b ∧ a * b = k) : 
  (-8 : ℝ) * y = k → y = -49 := by
sorry

end inverse_proportion_problem_l660_66053


namespace parallel_lines_perpendicular_lines_l660_66006

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 4 * y + 6 = 0
def l₂ (a x y : ℝ) : Prop := ((3/4) * a + 1) * x + a * y - 3/2 = 0

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l₁ a x y ↔ l₂ a x y) ↔ a = 4 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → x * x + y * y = 0) ↔ (a = 0 ∨ a = -20/3) :=
sorry

end parallel_lines_perpendicular_lines_l660_66006


namespace range_of_a_l660_66093

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) ↔ a ∈ Set.Ioo (-1 : ℝ) 3 :=
sorry

end range_of_a_l660_66093


namespace two_correct_probability_l660_66065

/-- The number of packages and houses -/
def n : ℕ := 5

/-- The probability of exactly 2 out of n packages being delivered correctly -/
def prob_two_correct (n : ℕ) : ℚ :=
  if n ≥ 2 then
    (n.choose 2 : ℚ) / n.factorial
  else 0

theorem two_correct_probability :
  prob_two_correct n = 1 / 12 :=
sorry

end two_correct_probability_l660_66065


namespace total_seedlings_l660_66099

/-- Given that each packet contains 7 seeds and there are 60 packets,
    prove that the total number of seedlings is 420. -/
theorem total_seedlings (seeds_per_packet : ℕ) (num_packets : ℕ) 
  (h1 : seeds_per_packet = 7) 
  (h2 : num_packets = 60) : 
  seeds_per_packet * num_packets = 420 := by
sorry

end total_seedlings_l660_66099


namespace negation_existence_proposition_l660_66086

theorem negation_existence_proposition :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry

end negation_existence_proposition_l660_66086


namespace circus_acrobats_l660_66029

/-- Represents the number of acrobats in the circus show -/
def acrobats : ℕ := 11

/-- Represents the number of elephants in the circus show -/
def elephants : ℕ := 4

/-- Represents the number of clowns in the circus show -/
def clowns : ℕ := 10

/-- The total number of legs in the circus show -/
def total_legs : ℕ := 58

/-- The total number of heads in the circus show -/
def total_heads : ℕ := 25

/-- Theorem stating that the number of acrobats is 11 given the conditions of the circus show -/
theorem circus_acrobats :
  (2 * acrobats + 4 * elephants + 2 * clowns = total_legs) ∧
  (acrobats + elephants + clowns = total_heads) ∧
  (acrobats = 11) := by
  sorry

end circus_acrobats_l660_66029


namespace factor_and_multiple_of_thirteen_l660_66010

theorem factor_and_multiple_of_thirteen (n : ℕ) : 
  (∃ k : ℕ, 13 = n * k) ∧ (∃ m : ℕ, n = 13 * m) → n = 13 := by
  sorry

end factor_and_multiple_of_thirteen_l660_66010


namespace johns_age_satisfies_condition_l660_66087

/-- Represents John's current age in years -/
def johnsCurrentAge : ℕ := 18

/-- Represents the condition that five years ago, John's age was half of what it will be in 8 years -/
def ageCondition (age : ℕ) : Prop :=
  age - 5 = (age + 8) / 2

/-- Theorem stating that John's current age satisfies the given condition -/
theorem johns_age_satisfies_condition : ageCondition johnsCurrentAge := by
  sorry

#check johns_age_satisfies_condition

end johns_age_satisfies_condition_l660_66087


namespace probability_marked_vertex_half_l660_66078

/-- Represents a shape with triangles and a marked vertex -/
structure TriangleShape where
  totalTriangles : ℕ
  trianglesWithMarkedVertex : ℕ
  hasProp : trianglesWithMarkedVertex ≤ totalTriangles

/-- The probability of selecting a triangle with the marked vertex -/
def probabilityMarkedVertex (shape : TriangleShape) : ℚ :=
  shape.trianglesWithMarkedVertex / shape.totalTriangles

theorem probability_marked_vertex_half (shape : TriangleShape) 
  (h1 : shape.totalTriangles = 6)
  (h2 : shape.trianglesWithMarkedVertex = 3) :
  probabilityMarkedVertex shape = 1/2 := by
  sorry

#check probability_marked_vertex_half

end probability_marked_vertex_half_l660_66078


namespace solution_set_m_zero_solution_set_real_l660_66034

-- Define the inequality
def inequality (m : ℝ) (x : ℝ) : Prop :=
  (m - 1) * x^2 + (m - 1) * x + 2 > 0

-- Part 1: Solution set when m = 0
theorem solution_set_m_zero :
  {x : ℝ | inequality 0 x} = Set.Ioo (-2) 1 := by sorry

-- Part 2: Range of m for solution set = ℝ
theorem solution_set_real :
  ∀ m : ℝ, (∀ x : ℝ, inequality m x) ↔ 1 ≤ m ∧ m < 9 := by sorry

end solution_set_m_zero_solution_set_real_l660_66034


namespace pants_cost_rita_pants_cost_l660_66037

/-- Calculates the cost of each pair of pants given Rita's shopping information -/
theorem pants_cost (initial_money : ℕ) (remaining_money : ℕ) (num_dresses : ℕ) (dress_cost : ℕ) 
  (num_pants : ℕ) (num_jackets : ℕ) (jacket_cost : ℕ) (transportation_cost : ℕ) : ℕ :=
  let total_spent := initial_money - remaining_money
  let dress_total := num_dresses * dress_cost
  let jacket_total := num_jackets * jacket_cost
  let pants_total := total_spent - dress_total - jacket_total - transportation_cost
  pants_total / num_pants

/-- Proves that each pair of pants costs $12 given Rita's shopping information -/
theorem rita_pants_cost : pants_cost 400 139 5 20 3 4 30 5 = 12 := by
  sorry

end pants_cost_rita_pants_cost_l660_66037


namespace expected_heads_3000_tosses_l660_66063

/-- A coin toss experiment with a fair coin -/
structure CoinTossExperiment where
  numTosses : ℕ
  probHeads : ℝ
  probHeads_eq : probHeads = 0.5

/-- The expected frequency of heads in a coin toss experiment -/
def expectedHeads (e : CoinTossExperiment) : ℝ :=
  e.numTosses * e.probHeads

/-- Theorem: The expected frequency of heads for 3000 tosses of a fair coin is 1500 -/
theorem expected_heads_3000_tosses (e : CoinTossExperiment) 
    (h : e.numTosses = 3000) : expectedHeads e = 1500 := by
  sorry

end expected_heads_3000_tosses_l660_66063


namespace survey_analysis_l660_66000

structure SurveyData where
  total : Nat
  aged_50_below_not_return : Nat
  aged_50_above_return : Nat
  aged_50_above_total : Nat

def chi_square (a b c d : Nat) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem survey_analysis (data : SurveyData) 
  (h1 : data.total = 100)
  (h2 : data.aged_50_below_not_return = 55)
  (h3 : data.aged_50_above_return = 15)
  (h4 : data.aged_50_above_total = 40) :
  let a := data.total - data.aged_50_above_total - data.aged_50_below_not_return
  let b := data.aged_50_below_not_return
  let c := data.aged_50_above_return
  let d := data.aged_50_above_total - data.aged_50_above_return
  (c : ℚ) / data.aged_50_above_total = 3 / 8 ∧ 
  chi_square a b c d > 10828 / 1000 := by
  sorry

end survey_analysis_l660_66000


namespace nancy_crayons_l660_66082

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 15

/-- The number of packs Nancy bought -/
def packs_bought : ℕ := 41

/-- The total number of crayons Nancy bought -/
def total_crayons : ℕ := crayons_per_pack * packs_bought

theorem nancy_crayons : total_crayons = 615 := by
  sorry

end nancy_crayons_l660_66082


namespace infinitely_many_prime_divisors_l660_66090

theorem infinitely_many_prime_divisors 
  (a b c d : ℕ+) 
  (ha : a ≠ b ∧ a ≠ c ∧ a ≠ d) 
  (hb : b ≠ c ∧ b ≠ d) 
  (hc : c ≠ d) : 
  ∃ (s : Set ℕ), Set.Infinite s ∧ 
  (∀ p ∈ s, Prime p ∧ ∃ n : ℕ, p ∣ (a * c^n + b * d^n)) :=
sorry

end infinitely_many_prime_divisors_l660_66090


namespace sausage_cost_per_pound_l660_66026

theorem sausage_cost_per_pound : 
  let packages : ℕ := 3
  let pounds_per_package : ℕ := 2
  let total_cost : ℕ := 24
  let total_pounds := packages * pounds_per_package
  let cost_per_pound := total_cost / total_pounds
  cost_per_pound = 4 := by sorry

end sausage_cost_per_pound_l660_66026


namespace marbles_given_correct_l660_66091

/-- The number of marbles Tyrone gave to Eric -/
def marbles_given : ℕ := sorry

/-- Tyrone's initial number of marbles -/
def tyrone_initial : ℕ := 150

/-- Eric's initial number of marbles -/
def eric_initial : ℕ := 18

/-- Theorem stating the number of marbles Tyrone gave to Eric -/
theorem marbles_given_correct : 
  marbles_given = 24 ∧
  tyrone_initial - marbles_given = 3 * (eric_initial + marbles_given) :=
by sorry

end marbles_given_correct_l660_66091


namespace smallest_multiple_of_three_l660_66098

def cards : List ℕ := [1, 2, 6]

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ ∃ (a b : ℕ), a ∈ cards ∧ b ∈ cards ∧ a ≠ b ∧ n = 10 * a + b

def is_multiple_of_three (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 3 * k

theorem smallest_multiple_of_three :
  ∃ (n : ℕ), is_valid_number n ∧ is_multiple_of_three n ∧
  ∀ (m : ℕ), is_valid_number m → is_multiple_of_three m → n ≤ m :=
by sorry

end smallest_multiple_of_three_l660_66098


namespace quadratic_roots_difference_squared_l660_66020

theorem quadratic_roots_difference_squared : 
  ∀ Θ θ : ℝ, 
  (Θ^2 - 3*Θ + 1 = 0) → 
  (θ^2 - 3*θ + 1 = 0) → 
  (Θ ≠ θ) → 
  (Θ - θ)^2 = 5 := by
sorry

end quadratic_roots_difference_squared_l660_66020


namespace unique_solution_system_l660_66071

/-- Given positive real numbers a, b, c, prove that the unique solution to the system of equations:
    1. x + y + z = a + b + c
    2. 4xyz - (a²x + b²y + c²z) = abc
    is x = (b+c)/2, y = (c+a)/2, z = (a+b)/2, where x, y, z are positive real numbers. -/
theorem unique_solution_system (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃! (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    x + y + z = a + b + c ∧
    4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c ∧
    x = (b + c) / 2 ∧ y = (c + a) / 2 ∧ z = (a + b) / 2 :=
by sorry


end unique_solution_system_l660_66071


namespace larger_number_problem_l660_66076

theorem larger_number_problem (x y : ℤ) (h1 : y = x + 10) (h2 : x + y = 34) : y = 22 := by
  sorry

end larger_number_problem_l660_66076


namespace midpoint_sum_x_invariant_l660_66080

/-- Represents a polygon in the Cartesian plane -/
structure Polygon :=
  (vertices : List (ℝ × ℝ))

/-- Creates a new polygon from the midpoints of the sides of the given polygon -/
def midpointPolygon (p : Polygon) : Polygon :=
  sorry

/-- Computes the sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

/-- Theorem: The sum of x-coordinates remains constant through midpoint constructions -/
theorem midpoint_sum_x_invariant (Q₁ : Polygon) :
  sumXCoordinates Q₁ = 120 →
  let Q₂ := midpointPolygon Q₁
  let Q₃ := midpointPolygon Q₂
  let Q₄ := midpointPolygon Q₃
  sumXCoordinates Q₄ = 120 :=
by
  sorry

end midpoint_sum_x_invariant_l660_66080


namespace amy_biking_distance_l660_66083

theorem amy_biking_distance (yesterday_distance today_distance : ℝ) : 
  yesterday_distance = 12 →
  yesterday_distance + today_distance = 33 →
  today_distance < 2 * yesterday_distance →
  2 * yesterday_distance - today_distance = 3 :=
by sorry

end amy_biking_distance_l660_66083


namespace sphere_volume_from_surface_area_l660_66019

theorem sphere_volume_from_surface_area (S : ℝ) (h : S = 36 * Real.pi) :
  (4 / 3 : ℝ) * Real.pi * ((S / (4 * Real.pi)) ^ (3 / 2 : ℝ)) = 36 * Real.pi := by
  sorry

end sphere_volume_from_surface_area_l660_66019


namespace handshake_arrangements_l660_66044

/-- The number of ways to arrange 10 people into two rings of 5, where each person in a ring is connected to 3 others -/
def M : ℕ := sorry

/-- The number of ways to select 5 people from 10 -/
def choose_five_from_ten : ℕ := sorry

/-- The number of arrangements within a ring of 5 -/
def ring_arrangements : ℕ := sorry

theorem handshake_arrangements :
  M = choose_five_from_ten * ring_arrangements * ring_arrangements ∧
  M % 1000 = 288 := by sorry

end handshake_arrangements_l660_66044


namespace consecutive_odd_integers_problem_l660_66003

theorem consecutive_odd_integers_problem (n : ℕ) : 
  n ≥ 3 ∧ n ≤ 9 ∧ n % 2 = 1 →
  (n - 2) + n + (n + 2) = ((n - 2) * n * (n + 2)) / 9 →
  n = 5 := by sorry

end consecutive_odd_integers_problem_l660_66003


namespace floor_fraction_theorem_l660_66039

theorem floor_fraction_theorem (d : ℝ) : 
  (∃ x : ℤ, x = ⌊d⌋ ∧ 3 * (x : ℝ)^2 + 10 * (x : ℝ) - 40 = 0) ∧ 
  (∃ y : ℝ, y = d - ⌊d⌋ ∧ 4 * y^2 - 20 * y + 19 = 0) →
  d = -9/2 := by
sorry

end floor_fraction_theorem_l660_66039


namespace special_polynomial_at_seven_l660_66001

/-- A monic polynomial of degree 7 satisfying specific conditions -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∀ x, ∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆, p x = x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) ∧
  p 0 = 0 ∧ p 1 = 1 ∧ p 2 = 2 ∧ p 3 = 3 ∧ p 4 = 4 ∧ p 5 = 5 ∧ p 6 = 6

/-- The theorem stating that any polynomial satisfying the special conditions will have p(7) = 5047 -/
theorem special_polynomial_at_seven (p : ℝ → ℝ) (h : special_polynomial p) : p 7 = 5047 := by
  sorry

end special_polynomial_at_seven_l660_66001


namespace stratified_random_most_appropriate_l660_66036

/-- Represents a laboratory with a certain number of mice -/
structure Laboratory where
  mice : ℕ

/-- Represents a sampling method -/
inductive SamplingMethod
  | EqualFromEach
  | FullyRandom
  | ArbitraryStratified
  | StratifiedRandom

/-- The problem setup -/
def biochemistryLabs : List Laboratory := [
  { mice := 18 },
  { mice := 24 },
  { mice := 54 },
  { mice := 48 }
]

/-- The total number of mice to be selected -/
def selectionSize : ℕ := 24

/-- Function to determine the most appropriate sampling method -/
def mostAppropriateSamplingMethod (labs : List Laboratory) (selectionSize : ℕ) : SamplingMethod :=
  SamplingMethod.StratifiedRandom

/-- Theorem stating that StratifiedRandom is the most appropriate method -/
theorem stratified_random_most_appropriate :
  mostAppropriateSamplingMethod biochemistryLabs selectionSize = SamplingMethod.StratifiedRandom := by
  sorry


end stratified_random_most_appropriate_l660_66036


namespace arithmetic_geometric_sequence_l660_66015

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) :
  (∀ n, a (n + 1) - a n = 2) →  -- arithmetic sequence with common difference 2
  (a 4)^2 = a 2 * a 5 →  -- a_2, a_4, a_5 form a geometric sequence
  a 2 = -8 :=
by sorry

end arithmetic_geometric_sequence_l660_66015


namespace percent_relation_l660_66027

theorem percent_relation (x y z : ℝ) (p : ℝ) 
  (h1 : y = 0.75 * x) 
  (h2 : z = 2 * x) 
  (h3 : p / 100 * z = 1.2 * y) : 
  p = 45 := by sorry

end percent_relation_l660_66027


namespace item_sale_ratio_l660_66046

theorem item_sale_ratio (c x y : ℝ) (hx : x = 0.85 * c) (hy : y = 1.15 * c) :
  y / x = 23 / 17 := by
  sorry

end item_sale_ratio_l660_66046


namespace quadratic_sum_has_root_l660_66064

/-- A quadratic polynomial with a positive leading coefficient -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a > 0

/-- The value of a quadratic polynomial at a given point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Two polynomials have a common root -/
def has_common_root (p q : QuadraticPolynomial) : Prop :=
  ∃ x, p.eval x = 0 ∧ q.eval x = 0

theorem quadratic_sum_has_root (p₁ p₂ p₃ : QuadraticPolynomial)
  (h₁₂ : has_common_root p₁ p₂)
  (h₂₃ : has_common_root p₂ p₃)
  (h₃₁ : has_common_root p₃ p₁) :
  ∃ x, (p₁.eval x + p₂.eval x + p₃.eval x = 0) :=
sorry

end quadratic_sum_has_root_l660_66064
