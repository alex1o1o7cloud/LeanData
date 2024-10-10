import Mathlib

namespace triangle_side_length_l964_96406

/-- Proves that in a triangle ABC with given conditions, the length of side c is 20 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) : 
  a = 4 → B = π / 3 → S = 20 * Real.sqrt 3 → c = 20 := by
  sorry

end triangle_side_length_l964_96406


namespace min_slope_tangent_line_l964_96498

variable (x y : ℝ)

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 3*x - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 3

-- Theorem statement
theorem min_slope_tangent_line :
  ∃ (x₀ : ℝ), 
    (∀ x, f' x ≥ f' x₀) ∧ 
    (3*x - y + 1 = 0 ↔ y = f' x₀ * (x - x₀) + f x₀) :=
sorry

end min_slope_tangent_line_l964_96498


namespace allens_blocks_combinations_l964_96449

/-- Given conditions for Allen's blocks problem -/
structure BlocksProblem where
  total_blocks : ℕ
  num_shapes : ℕ
  blocks_per_color : ℕ

/-- Calculate the number of color and shape combinations -/
def calculate_combinations (problem : BlocksProblem) : ℕ :=
  let num_colors := problem.total_blocks / problem.blocks_per_color
  problem.num_shapes * num_colors

/-- Theorem: The number of color and shape combinations is 80 -/
theorem allens_blocks_combinations (problem : BlocksProblem) 
  (h1 : problem.total_blocks = 100)
  (h2 : problem.num_shapes = 4)
  (h3 : problem.blocks_per_color = 5) :
  calculate_combinations problem = 80 := by
  sorry

#eval calculate_combinations ⟨100, 4, 5⟩

end allens_blocks_combinations_l964_96449


namespace fraction_existence_and_nonexistence_l964_96475

theorem fraction_existence_and_nonexistence :
  (∀ n : ℕ+, ∃ a b : ℤ, 0 < b ∧ b ≤ Real.sqrt n + 1 ∧ Real.sqrt n ≤ a / b ∧ a / b ≤ Real.sqrt (n + 1)) ∧
  (∃ f : ℕ → ℕ+, StrictMono f ∧ ∀ n : ℕ, ¬∃ a b : ℤ, 0 < b ∧ b ≤ Real.sqrt (f n) ∧ Real.sqrt (f n) ≤ a / b ∧ a / b ≤ Real.sqrt (f n + 1)) :=
by sorry

end fraction_existence_and_nonexistence_l964_96475


namespace small_tile_position_l964_96429

/-- Represents a tile on the grid -/
inductive Tile
| Small : Tile  -- 1x1 tile
| Large : Tile  -- 1x3 tile

/-- Represents a position on the 7x7 grid -/
structure Position where
  row : Fin 7
  col : Fin 7

/-- Represents the state of the grid -/
structure GridState where
  smallTilePos : Position
  largeTiles : Finset (Position × Position × Position)

/-- Checks if a position is at the center or adjacent to the border -/
def isCenterOrBorder (pos : Position) : Prop :=
  pos.row = 0 ∨ pos.row = 3 ∨ pos.row = 6 ∨
  pos.col = 0 ∨ pos.col = 3 ∨ pos.col = 6

/-- Main theorem -/
theorem small_tile_position (grid : GridState) 
  (h1 : grid.largeTiles.card = 16) : 
  isCenterOrBorder grid.smallTilePos :=
sorry

end small_tile_position_l964_96429


namespace perpendicular_vectors_x_value_l964_96468

/-- Given two vectors a and b in ℝ², where a = (1,2) and b = (3,x), 
    if (a + b) is perpendicular to a, then x = -4. -/
theorem perpendicular_vectors_x_value : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![3, x]
  (∀ i : Fin 2, (a + b) i * a i = 0) → x = -4 := by
  sorry

end perpendicular_vectors_x_value_l964_96468


namespace cubic_factorization_l964_96422

theorem cubic_factorization (x : ℝ) : x^3 - 6*x^2 + 9*x = x*(x-3)^2 := by
  sorry

end cubic_factorization_l964_96422


namespace fixed_points_for_specific_values_two_distinct_fixed_points_iff_l964_96473

/-- Definition of a fixed point for a function f -/
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

/-- The quadratic function f(x) = ax² + (b+1)x + b - 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

theorem fixed_points_for_specific_values (a b : ℝ) (h1 : a = 1) (h2 : b = 5) :
  is_fixed_point (f a b) (-4) ∧ is_fixed_point (f a b) (-1) :=
sorry

theorem two_distinct_fixed_points_iff (a : ℝ) (h : a ≠ 0) :
  (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ is_fixed_point (f a b) x ∧ is_fixed_point (f a b) y) ↔
  (0 < a ∧ a < 1) :=
sorry

end fixed_points_for_specific_values_two_distinct_fixed_points_iff_l964_96473


namespace sufficient_not_necessary_l964_96454

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x^2 + 1 > 2*x ∧ x ≤ 1) ∧
  (∀ x : ℝ, x > 1 → x^2 + 1 > 2*x) := by
  sorry

end sufficient_not_necessary_l964_96454


namespace multiple_z_values_l964_96411

/-- Given two four-digit integers x and y where y is the reverse of x, 
    z = |x - y| can take multiple distinct values. -/
theorem multiple_z_values (x y z : ℕ) : 
  (1000 ≤ x ∧ x ≤ 9999) →
  (1000 ≤ y ∧ y ≤ 9999) →
  (y = (x % 10) * 1000 + ((x / 10) % 10) * 100 + ((x / 100) % 10) * 10 + (x / 1000)) →
  (z = Int.natAbs (x - y)) →
  ∃ (z₁ z₂ : ℕ), z₁ ≠ z₂ ∧ 
    ∃ (x₁ y₁ x₂ y₂ : ℕ), 
      (1000 ≤ x₁ ∧ x₁ ≤ 9999) ∧
      (1000 ≤ y₁ ∧ y₁ ≤ 9999) ∧
      (y₁ = (x₁ % 10) * 1000 + ((x₁ / 10) % 10) * 100 + ((x₁ / 100) % 10) * 10 + (x₁ / 1000)) ∧
      (z₁ = Int.natAbs (x₁ - y₁)) ∧
      (1000 ≤ x₂ ∧ x₂ ≤ 9999) ∧
      (1000 ≤ y₂ ∧ y₂ ≤ 9999) ∧
      (y₂ = (x₂ % 10) * 1000 + ((x₂ / 10) % 10) * 100 + ((x₂ / 100) % 10) * 10 + (x₂ / 1000)) ∧
      (z₂ = Int.natAbs (x₂ - y₂)) :=
by
  sorry


end multiple_z_values_l964_96411


namespace distinct_collections_eq_110_l964_96413

def vowels : ℕ := 5
def consonants : ℕ := 4
def indistinguishable_consonants : ℕ := 2
def vowels_to_select : ℕ := 3
def consonants_to_select : ℕ := 4

def distinct_collections : ℕ :=
  (Nat.choose vowels vowels_to_select) *
  (Nat.choose consonants consonants_to_select +
   Nat.choose consonants (consonants_to_select - 1) +
   Nat.choose consonants (consonants_to_select - 2))

theorem distinct_collections_eq_110 :
  distinct_collections = 110 :=
sorry

end distinct_collections_eq_110_l964_96413


namespace perpendicular_equivalence_l964_96425

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the theorem
theorem perpendicular_equivalence 
  (α β : Plane) (m n : Line) 
  (h_diff_planes : α ≠ β) 
  (h_diff_lines : m ≠ n) 
  (h_n_perp_α : perp n α) 
  (h_n_perp_β : perp n β) :
  perp m α ↔ perp m β :=
sorry

end perpendicular_equivalence_l964_96425


namespace infinitely_many_integers_l964_96432

theorem infinitely_many_integers (k : ℕ) (hk : k > 1) :
  ∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a - 1) / b + (b - 1) / c + (c - 1) / a = k + 1 :=
sorry

end infinitely_many_integers_l964_96432


namespace linear_equation_integer_solutions_l964_96492

theorem linear_equation_integer_solutions :
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0) ∧ 
    (∀ m ∈ S, ∃ x : ℕ, x > 0 ∧ m * x + 2 * x - 12 = 0) ∧
    (∀ m : ℕ, m > 0 → (∃ x : ℕ, x > 0 ∧ m * x + 2 * x - 12 = 0) → m ∈ S) ∧
    S.card = 4 :=
by sorry

end linear_equation_integer_solutions_l964_96492


namespace triangle_area_lower_bound_no_triangle_large_altitudes_small_area_l964_96447

/-- A triangle with two altitudes greater than 100 has an area greater than 1 -/
theorem triangle_area_lower_bound (a b c h1 h2 : ℝ) (hpos : a > 0 ∧ b > 0 ∧ c > 0) 
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) (halt : h1 > 100 ∧ h2 > 100) : 
  (1/2) * a * h1 > 1 := by
  sorry

/-- There does not exist a triangle with two altitudes greater than 100 and area less than 1 -/
theorem no_triangle_large_altitudes_small_area : 
  ¬ ∃ (a b c h1 h2 : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧ 
  h1 > 100 ∧ h2 > 100 ∧ 
  (1/2) * a * h1 < 1 := by
  sorry

end triangle_area_lower_bound_no_triangle_large_altitudes_small_area_l964_96447


namespace price_reduction_achieves_target_profit_l964_96420

/-- Represents the price reduction in yuan -/
def price_reduction : ℕ := 10

/-- Cost to purchase each piece of clothing -/
def purchase_cost : ℕ := 45

/-- Original selling price of each piece of clothing -/
def original_price : ℕ := 65

/-- Original daily sales quantity -/
def original_sales : ℕ := 30

/-- Additional sales for each yuan of price reduction -/
def sales_increase_rate : ℕ := 5

/-- Target daily profit -/
def target_profit : ℕ := 800

/-- Theorem stating that the given price reduction achieves the target profit -/
theorem price_reduction_achieves_target_profit :
  (original_price - price_reduction - purchase_cost) *
  (original_sales + sales_increase_rate * price_reduction) = target_profit :=
sorry

end price_reduction_achieves_target_profit_l964_96420


namespace intersection_length_l964_96457

-- Define the line l passing through A(0,1) with slope k
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the circle C: (x-2)^2 + (y-3)^2 = 1
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define the condition that line l intersects circle C at points M and N
def intersects (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧ 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 12

-- Main theorem
theorem intersection_length (k : ℝ) :
  intersects k →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧ 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    dot_product_condition x₁ y₁ x₂ y₂) →
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧ 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4 :=
by sorry

end intersection_length_l964_96457


namespace expression_simplification_l964_96452

theorem expression_simplification (x : ℝ) : (36 + 12*x)^2 - (12^2*x^2 + 36^2) = 864*x := by
  sorry

end expression_simplification_l964_96452


namespace abc_product_l964_96453

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + 1/b = 5) (h2 : b + 1/c = 2) (h3 : (c + 1/a)^2 = 4) :
  a * b * c = (11 + Real.sqrt 117) / 2 := by
sorry

end abc_product_l964_96453


namespace triangle_perimeter_not_55_l964_96465

theorem triangle_perimeter_not_55 (a b x : ℝ) : 
  a = 18 → b = 10 → 
  (a + b > x ∧ a + x > b ∧ b + x > a) → 
  a + b + x ≠ 55 :=
by
  sorry

end triangle_perimeter_not_55_l964_96465


namespace perimeter_after_adding_tiles_l964_96437

/-- Represents a configuration of square tiles --/
structure TileConfiguration where
  numTiles : ℕ
  perimeter : ℕ

/-- Adds tiles to a configuration --/
def addTiles (config : TileConfiguration) (newTiles : ℕ) : TileConfiguration :=
  { numTiles := config.numTiles + newTiles
  , perimeter := config.perimeter + 2 * newTiles }

theorem perimeter_after_adding_tiles 
  (initialConfig : TileConfiguration) 
  (tilesAdded : ℕ) :
  initialConfig.numTiles = 9 →
  initialConfig.perimeter = 16 →
  tilesAdded = 3 →
  (addTiles initialConfig tilesAdded).perimeter = 22 := by
  sorry

#check perimeter_after_adding_tiles

end perimeter_after_adding_tiles_l964_96437


namespace quadratic_two_roots_l964_96494

/-- A quadratic function f(x) = ax^2 + bx + c with a ≠ 0 and satisfying 5a + b + 2c = 0 has two distinct real roots. -/
theorem quadratic_two_roots (a b c : ℝ) (ha : a ≠ 0) (h_cond : 5 * a + b + 2 * c = 0) :
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by sorry

end quadratic_two_roots_l964_96494


namespace devin_teaching_years_l964_96488

/-- Represents the number of years Devin taught each subject -/
structure TeachingYears where
  calculus : ℕ
  algebra : ℕ
  statistics : ℕ
  geometry : ℕ
  discrete_math : ℕ

/-- Calculates the total number of years taught -/
def total_years (years : TeachingYears) : ℕ :=
  years.calculus + years.algebra + years.statistics + years.geometry + years.discrete_math

/-- Theorem stating the total number of years Devin taught -/
theorem devin_teaching_years :
  ∃ (years : TeachingYears),
    years.calculus = 4 ∧
    years.algebra = 2 * years.calculus ∧
    years.statistics = 5 * years.algebra ∧
    years.geometry = 3 * years.statistics ∧
    years.discrete_math = years.geometry / 2 ∧
    total_years years = 232 :=
by sorry

end devin_teaching_years_l964_96488


namespace jane_age_problem_l964_96499

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem jane_age_problem :
  ∃ j : ℕ, j > 0 ∧ is_perfect_square (j - 2) ∧ is_perfect_cube (j + 2) ∧
  ∀ k : ℕ, k > 0 → is_perfect_square (k - 2) → is_perfect_cube (k + 2) → j ≤ k :=
sorry

end jane_age_problem_l964_96499


namespace triangle_problem_l964_96404

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) 
  (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π) 
  (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) 
  (h5 : a * Real.sin B = Real.sqrt 3 * b * Real.cos A) 
  (h6 : b = 3) 
  (h7 : c = 2) : 
  A = π / 3 ∧ a = Real.sqrt 7 := by
sorry


end triangle_problem_l964_96404


namespace power_sum_of_i_l964_96430

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i :
  i^15 + i^20 + i^25 + i^30 + i^35 = -i :=
by
  sorry

end power_sum_of_i_l964_96430


namespace fundraiser_proof_l964_96482

def fundraiser (total_promised : ℕ) (amount_received : ℕ) (amy_owes : ℕ) : Prop :=
  let total_owed : ℕ := total_promised - amount_received
  let derek_owes : ℕ := amy_owes / 2
  let sally_carl_owe : ℕ := total_owed - (amy_owes + derek_owes)
  sally_carl_owe / 2 = 35

theorem fundraiser_proof : fundraiser 400 285 30 := by
  sorry

end fundraiser_proof_l964_96482


namespace math_competition_problem_l964_96462

theorem math_competition_problem :
  ∀ (total students_only_A students_A_and_others students_only_B students_only_C students_B_and_C : ℕ),
    total = 25 →
    total = students_only_A + students_A_and_others + students_only_B + students_only_C + students_B_and_C →
    students_only_B + students_B_and_C = 2 * (students_only_C + students_B_and_C) →
    students_only_A = students_A_and_others + 1 →
    2 * (students_only_B + students_only_C) = students_only_A →
    students_only_B = 6 := by
  sorry

end math_competition_problem_l964_96462


namespace min_value_expression_l964_96490

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hsum : x + y = 1/2) (horder : x ≤ y ∧ y ≤ z) :
  (x + z) / (x * y * z) ≥ 48 := by
  sorry

end min_value_expression_l964_96490


namespace circus_ticket_cost_l964_96439

theorem circus_ticket_cost (total_spent : ℕ) (num_tickets : ℕ) (cost_per_ticket : ℕ) : 
  total_spent = 308 → num_tickets = 7 → cost_per_ticket = total_spent / num_tickets → cost_per_ticket = 44 := by
  sorry

end circus_ticket_cost_l964_96439


namespace divisors_of_sum_of_primes_l964_96495

-- Define a prime number p ≥ 5
def p : ℕ := sorry

-- Define q as the smallest prime number greater than p
def q : ℕ := sorry

-- Define n as the number of positive divisors of p + q
def n : ℕ := sorry

-- Axioms based on the problem conditions
axiom p_prime : Nat.Prime p
axiom p_ge_5 : p ≥ 5
axiom q_prime : Nat.Prime q
axiom q_gt_p : q > p
axiom q_smallest : ∀ r, Nat.Prime r → r > p → r ≥ q

-- Theorem to prove
theorem divisors_of_sum_of_primes :
  n ≥ 4 ∧ (∀ m, m ≥ 6 → n ≤ m) := by sorry

end divisors_of_sum_of_primes_l964_96495


namespace vacation_cost_l964_96493

theorem vacation_cost (cost : ℝ) : 
  (cost / 3 - cost / 4 = 60) → cost = 720 := by
  sorry

end vacation_cost_l964_96493


namespace quiz_score_difference_l964_96414

def quiz_scores : List (Float × Float) := [
  (0.05, 65),
  (0.25, 75),
  (0.40, 85),
  (0.20, 95),
  (0.10, 105)
]

def mean (scores : List (Float × Float)) : Float :=
  (scores.map (λ (p, s) => p * s)).sum

def median (scores : List (Float × Float)) : Float :=
  if (scores.map (λ (p, _) => p)).sum ≥ 0.5 then
    scores.filter (λ (_, s) => s ≥ 85)
      |> List.head!
      |> (λ (_, s) => s)
  else 85

theorem quiz_score_difference :
  median quiz_scores - mean quiz_scores = -0.5 := by
  sorry

end quiz_score_difference_l964_96414


namespace initially_calculated_average_weight_l964_96487

theorem initially_calculated_average_weight
  (n : ℕ)
  (correct_avg : ℝ)
  (misread_weight : ℝ)
  (correct_weight : ℝ)
  (h1 : n = 20)
  (h2 : correct_avg = 58.9)
  (h3 : misread_weight = 56)
  (h4 : correct_weight = 66)
  : ∃ (initial_avg : ℝ), initial_avg = 58.4 :=
by
  sorry

end initially_calculated_average_weight_l964_96487


namespace olivers_shirts_l964_96467

theorem olivers_shirts (short_sleeve : ℕ) (washed : ℕ) (unwashed : ℕ) :
  short_sleeve = 39 →
  washed = 20 →
  unwashed = 66 →
  ∃ (long_sleeve : ℕ), long_sleeve = 7 ∧ short_sleeve + long_sleeve = washed + unwashed :=
by sorry

end olivers_shirts_l964_96467


namespace smallest_base_perfect_square_l964_96409

theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 4 ∧ ∃ (n : ℕ), 4 * b + 5 = n^2 ∧ 
  ∀ (x : ℕ), x > 4 ∧ x < b → ¬∃ (m : ℕ), 4 * x + 5 = m^2 :=
by
  -- Proof goes here
  sorry

end smallest_base_perfect_square_l964_96409


namespace sum_of_integers_l964_96476

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val - y.val = 14) 
  (h2 : x.val * y.val = 180) : 
  x.val + y.val = 2 * Real.sqrt 229 := by
sorry

end sum_of_integers_l964_96476


namespace three_factor_numbers_product_l964_96421

theorem three_factor_numbers_product (x y z : ℕ) : 
  x ≠ y ∧ y ≠ z ∧ x ≠ z →
  (∃ p₁ : ℕ, Prime p₁ ∧ x = p₁^2) →
  (∃ p₂ : ℕ, Prime p₂ ∧ y = p₂^2) →
  (∃ p₃ : ℕ, Prime p₃ ∧ z = p₃^2) →
  (Nat.card {d : ℕ | d ∣ x} = 3) →
  (Nat.card {d : ℕ | d ∣ y} = 3) →
  (Nat.card {d : ℕ | d ∣ z} = 3) →
  Nat.card {d : ℕ | d ∣ (x^2 * y^3 * z^4)} = 315 := by
sorry

end three_factor_numbers_product_l964_96421


namespace profit_increase_l964_96481

theorem profit_increase (initial_profit : ℝ) (march_to_april : ℝ) : 
  (initial_profit * (1 + march_to_april / 100) * 0.8 * 1.5 = initial_profit * 1.5600000000000001) →
  march_to_april = 30 :=
by sorry

end profit_increase_l964_96481


namespace inequality_and_equality_condition_l964_96463

theorem inequality_and_equality_condition (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) + 
  Real.sqrt ((a * b + b * c + c * a) / (a^2 + b^2 + c^2)) ≥ (5 : ℝ) / 2 ∧
  ((a / (b + c)) + (b / (c + a)) + (c / (a + b)) + 
   Real.sqrt ((a * b + b * c + c * a) / (a^2 + b^2 + c^2)) = (5 : ℝ) / 2 ↔ 
   a = b ∧ b = c) :=
by sorry

end inequality_and_equality_condition_l964_96463


namespace not_always_congruent_l964_96433

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (α β γ : ℝ)

-- Define the property of having two equal sides and three equal angles
def hasTwoEqualSidesThreeEqualAngles (t1 t2 : Triangle) : Prop :=
  ((t1.a = t2.a ∧ t1.b = t2.b) ∨ (t1.a = t2.a ∧ t1.c = t2.c) ∨ (t1.b = t2.b ∧ t1.c = t2.c)) ∧
  (t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ)

-- Define triangle congruence
def isCongruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c ∧
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- Theorem statement
theorem not_always_congruent :
  ∃ (t1 t2 : Triangle), hasTwoEqualSidesThreeEqualAngles t1 t2 ∧ ¬isCongruent t1 t2 :=
sorry

end not_always_congruent_l964_96433


namespace expression_value_l964_96403

theorem expression_value (a b x y c : ℝ) 
  (h1 : a = -b) 
  (h2 : x * y = 1) 
  (h3 : c = 2 ∨ c = -2) : 
  (a + b) / 2 + x * y - (1 / 4) * c = 1 / 2 ∨ (a + b) / 2 + x * y - (1 / 4) * c = 3 / 2 := by
sorry

end expression_value_l964_96403


namespace jenna_peeled_potatoes_l964_96470

/-- The number of potatoes Jenna peeled -/
def jenna_potatoes : ℕ := 24

/-- The total number of potatoes -/
def total_potatoes : ℕ := 60

/-- Homer's peeling rate in potatoes per minute -/
def homer_rate : ℕ := 4

/-- Jenna's peeling rate in potatoes per minute -/
def jenna_rate : ℕ := 6

/-- The time Homer peeled alone in minutes -/
def homer_alone_time : ℕ := 6

/-- The combined peeling rate of Homer and Jenna in potatoes per minute -/
def combined_rate : ℕ := homer_rate + jenna_rate

theorem jenna_peeled_potatoes :
  jenna_potatoes = total_potatoes - (homer_rate * homer_alone_time) :=
by sorry

#check jenna_peeled_potatoes

end jenna_peeled_potatoes_l964_96470


namespace quadratic_inequality_solution_set_l964_96466

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - x - 2 < 0 ↔ -1 < x ∧ x < 2 :=
by sorry

end quadratic_inequality_solution_set_l964_96466


namespace calculation_result_l964_96486

theorem calculation_result : (786 * 74) / 30 = 1938.8 := by
  sorry

end calculation_result_l964_96486


namespace ribbon_length_difference_l964_96446

/-- The theorem about ribbon lengths difference after cutting and giving -/
theorem ribbon_length_difference 
  (initial_difference : Real) 
  (cut_length : Real) : 
  initial_difference = 8.8 → 
  cut_length = 4.3 → 
  initial_difference + 2 * cut_length = 17.4 := by
  sorry

end ribbon_length_difference_l964_96446


namespace fuel_cost_calculation_l964_96480

/-- Calculates the new fuel cost after a price increase and capacity increase -/
def new_fuel_cost (original_cost : ℚ) (price_increase_percent : ℚ) (capacity_multiplier : ℚ) : ℚ :=
  original_cost * (1 + price_increase_percent / 100) * capacity_multiplier

/-- Proves that the new fuel cost is $480 given the specified conditions -/
theorem fuel_cost_calculation :
  new_fuel_cost 200 20 2 = 480 := by
  sorry

end fuel_cost_calculation_l964_96480


namespace power_mod_eleven_l964_96472

theorem power_mod_eleven : 5^2023 % 11 = 4 := by
  sorry

end power_mod_eleven_l964_96472


namespace floor_sum_example_l964_96443

theorem floor_sum_example : ⌊(2.7 : ℝ) + 1.5⌋ = 4 := by
  sorry

end floor_sum_example_l964_96443


namespace smallest_prime_after_eight_nonprimes_l964_96450

def is_first_prime_after_eight_nonprimes (p : ℕ) : Prop :=
  Nat.Prime p ∧
  ∃ n : ℕ, n > 0 ∧
    (∀ k : ℕ, n ≤ k ∧ k < n + 8 → ¬Nat.Prime k) ∧
    (∀ q : ℕ, Nat.Prime q → q < p → q ≤ n - 1 ∨ q ≥ n + 8)

theorem smallest_prime_after_eight_nonprimes :
  is_first_prime_after_eight_nonprimes 59 :=
sorry

end smallest_prime_after_eight_nonprimes_l964_96450


namespace pyramid_top_value_l964_96410

/-- Represents a pyramid structure where each number is the product of the two numbers above it -/
structure Pyramid where
  bottom_row : Fin 3 → ℕ
  x : ℕ
  y : ℕ

/-- The conditions of the pyramid problem -/
def pyramid_conditions (p : Pyramid) : Prop :=
  p.bottom_row 0 = 240 ∧
  p.bottom_row 1 = 720 ∧
  p.bottom_row 2 = 1440 ∧
  p.x * 6 = 720

/-- The theorem stating that given the conditions, y must be 120 -/
theorem pyramid_top_value (p : Pyramid) (h : pyramid_conditions p) : p.y = 120 := by
  sorry

#check pyramid_top_value

end pyramid_top_value_l964_96410


namespace max_pieces_8x8_grid_l964_96451

/-- Represents a square grid -/
structure Grid :=
  (size : Nat)

/-- Represents the number of pieces after cutting -/
def num_pieces (g : Grid) (num_cuts : Nat) : Nat :=
  sorry

/-- The maximum number of pieces that can be obtained from an 8x8 grid -/
theorem max_pieces_8x8_grid :
  ∃ (max_pieces : Nat), 
    (∀ (g : Grid) (num_cuts : Nat), 
      g.size = 8 → num_pieces g num_cuts ≤ max_pieces) ∧ 
    (∃ (g : Grid) (num_cuts : Nat), 
      g.size = 8 ∧ num_pieces g num_cuts = max_pieces) ∧
    max_pieces = 16 := by
  sorry

end max_pieces_8x8_grid_l964_96451


namespace sum_first_15_odd_from_5_l964_96412

/-- The sum of the first n odd positive integers starting from a given odd number -/
def sum_odd_integers (start : ℕ) (n : ℕ) : ℕ :=
  let last := start + 2 * (n - 1)
  n * (start + last) / 2

/-- Theorem: The sum of the first 15 odd positive integers starting from 5 is 255 -/
theorem sum_first_15_odd_from_5 : sum_odd_integers 5 15 = 255 := by
  sorry

end sum_first_15_odd_from_5_l964_96412


namespace equidifference_ratio_sequence_properties_l964_96442

/-- Definition of an equidifference ratio sequence -/
def IsEquidifferenceRatioSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ n : ℕ+, (a (n + 2) - a (n + 1)) / (a (n + 1) - a n) = k

theorem equidifference_ratio_sequence_properties
  (a : ℕ+ → ℝ) (h : IsEquidifferenceRatioSequence a) :
  (∃ k : ℝ, k ≠ 0 ∧ ∀ n : ℕ+, (a (n + 2) - a (n + 1)) / (a (n + 1) - a n) = k) ∧
  (∃ b : ℕ+ → ℝ, IsEquidifferenceRatioSequence b ∧ Set.Infinite {n : ℕ+ | b n = 0}) :=
by sorry

end equidifference_ratio_sequence_properties_l964_96442


namespace base_b_is_five_l964_96464

/-- The base in which 200 (base 10) is represented with exactly 4 digits -/
def base_b : ℕ := 5

/-- 200 in base 10 -/
def number : ℕ := 200

theorem base_b_is_five :
  ∃! b : ℕ, b > 1 ∧ 
  (b ^ 3 ≤ number) ∧ 
  (number < b ^ 4) ∧
  (∀ d : ℕ, d < b → number ≥ d * b ^ 3) :=
sorry

end base_b_is_five_l964_96464


namespace max_sequence_length_l964_96419

/-- Represents a quadratic equation in the sequence -/
structure QuadraticEquation where
  p : ℝ
  q : ℝ
  h : p < q

/-- Constructs the next quadratic equation in the sequence -/
def nextEquation (eq : QuadraticEquation) : QuadraticEquation :=
  { p := eq.q, q := -eq.p - eq.q, h := sorry }

/-- The sequence of quadratic equations -/
def quadraticSequence (initial : QuadraticEquation) : ℕ → QuadraticEquation
  | 0 => initial
  | n + 1 => nextEquation (quadraticSequence initial n)

/-- The main theorem: the maximum length of the sequence is 5 -/
theorem max_sequence_length (initial : QuadraticEquation) :
  ∃ n : ℕ, n ≤ 5 ∧ ∀ m : ℕ, m > n → ¬ (quadraticSequence initial m).p < (quadraticSequence initial m).q :=
sorry

end max_sequence_length_l964_96419


namespace monotonicity_condition_l964_96436

/-- A function f is monotonically increasing on ℝ -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The cubic function f(x) = x³ - 2x² - mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 - m*x + 1

theorem monotonicity_condition (m : ℝ) :
  (m > 4/3 → MonotonicallyIncreasing (f m)) ∧
  (∃ m : ℝ, m ≤ 4/3 ∧ MonotonicallyIncreasing (f m)) :=
by sorry

end monotonicity_condition_l964_96436


namespace point_movement_specific_point_movement_l964_96461

/-- Given a point P in a 2D Cartesian coordinate system, moving it right and up results in a new point P'. -/
theorem point_movement (x y dx dy : ℝ) :
  let P : ℝ × ℝ := (x, y)
  let P' : ℝ × ℝ := (x + dx, y + dy)
  P' = (x + dx, y + dy) :=
by sorry

/-- The specific case of moving point P(2, -3) right by 2 units and up by 4 units results in P'(4, 1). -/
theorem specific_point_movement :
  let P : ℝ × ℝ := (2, -3)
  let P' : ℝ × ℝ := (2 + 2, -3 + 4)
  P' = (4, 1) :=
by sorry

end point_movement_specific_point_movement_l964_96461


namespace product_nine_sum_zero_l964_96485

theorem product_nine_sum_zero (a b c d : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 9 →
  a + b + c + d = 0 := by
sorry

end product_nine_sum_zero_l964_96485


namespace odd_function_inequality_l964_96469

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_inequality (f : ℝ → ℝ) 
  (h_odd : OddFunction f)
  (h_ineq : ∀ x₁ x₂, x₁ < 0 → x₂ < 0 → x₁ ≠ x₂ → 
    (x₂ * f x₁ - x₁ * f x₂) / (x₁ - x₂) > 0) :
  3 * f (1/3) > -5/2 * f (-2/5) ∧ -5/2 * f (-2/5) > f 1 := by
  sorry

end odd_function_inequality_l964_96469


namespace min_value_fraction_sum_l964_96405

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 1) :
  (3 / a + 2 / b) ≥ 25 := by sorry

end min_value_fraction_sum_l964_96405


namespace rectangular_prism_surface_area_bound_l964_96448

/-- Given a quadrilateral with sides a, b, c, and d, the surface area of a rectangular prism
    with edges a, b, and c meeting at a vertex is at most (a+b+c)^2 - d^2/3 -/
theorem rectangular_prism_surface_area_bound 
  (a b c d : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (quad : a + b + c > d ∧ b + c + d > a ∧ c + d + a > b ∧ d + a + b > c) :
  2 * (a * b + b * c + c * a) ≤ (a + b + c)^2 - d^2 / 3 := by
  sorry

end rectangular_prism_surface_area_bound_l964_96448


namespace quadratic_minimum_l964_96434

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*x + 6

-- Theorem statement
theorem quadratic_minimum :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min) ∧
  (∀ (x : ℝ), f x ≥ 5) ∧
  (f 1 = 5) :=
sorry

end quadratic_minimum_l964_96434


namespace sum_of_ages_is_105_l964_96400

/-- Calculates the sum of Riza's and her son's ages given their initial conditions -/
def sumOfAges (rizaAgeAtBirth : ℕ) (sonCurrentAge : ℕ) : ℕ :=
  rizaAgeAtBirth + 2 * sonCurrentAge

/-- Proves that the sum of Riza's and her son's ages is 105 years -/
theorem sum_of_ages_is_105 :
  sumOfAges 25 40 = 105 := by
  sorry

end sum_of_ages_is_105_l964_96400


namespace multiplication_of_squares_l964_96471

theorem multiplication_of_squares (a b : ℝ) : 2 * a^2 * 3 * b^2 = 6 * a^2 * b^2 := by
  sorry

end multiplication_of_squares_l964_96471


namespace sum_xy_equals_three_l964_96484

theorem sum_xy_equals_three (x y : ℝ) (h : Real.sqrt (1 - x) + abs (2 - y) = 0) : x + y = 3 := by
  sorry

end sum_xy_equals_three_l964_96484


namespace cube_probability_l964_96423

-- Define the type for cube faces
inductive CubeFace
| Face1 | Face2 | Face3 | Face4 | Face5 | Face6

-- Define the type for numbers
inductive Number
| One | Two | Three | Four | Five | Six | Seven | Eight | Nine

-- Define a function to check if two numbers are consecutive
def isConsecutive (n1 n2 : Number) : Prop := sorry

-- Define a function to check if two faces share an edge
def sharesEdge (f1 f2 : CubeFace) : Prop := sorry

-- Define the type for cube configuration
def CubeConfig := CubeFace → Option Number

-- Define a valid cube configuration
def isValidConfig (config : CubeConfig) : Prop :=
  (∀ f1 f2 : CubeFace, f1 ≠ f2 → config f1 ≠ config f2) ∧
  (∀ f1 f2 : CubeFace, sharesEdge f1 f2 →
    ∀ n1 n2 : Number, config f1 = some n1 → config f2 = some n2 →
      ¬isConsecutive n1 n2)

-- Define the total number of possible configurations
def totalConfigs : ℕ := sorry

-- Define the number of valid configurations
def validConfigs : ℕ := sorry

-- The main theorem
theorem cube_probability :
  (validConfigs : ℚ) / totalConfigs = 1 / 672 := by sorry

end cube_probability_l964_96423


namespace pages_torn_off_l964_96431

def total_pages : ℕ := 100
def sum_remaining_pages : ℕ := 4949

theorem pages_torn_off : 
  ∃ (torn_pages : Finset ℕ), 
    torn_pages.card = 3 ∧ 
    (Finset.range total_pages.succ).sum id - torn_pages.sum id = sum_remaining_pages :=
sorry

end pages_torn_off_l964_96431


namespace midpoint_of_translated_segment_l964_96456

/-- Given a segment s₁ with endpoints (2, -3) and (10, 7), and segment s₂ obtained by
    translating s₁ by 3 units to the left and 2 units down, prove that the midpoint
    of s₂ is (3, 0). -/
theorem midpoint_of_translated_segment :
  let s₁_start : ℝ × ℝ := (2, -3)
  let s₁_end : ℝ × ℝ := (10, 7)
  let translation : ℝ × ℝ := (-3, -2)
  let s₂_start : ℝ × ℝ := (s₁_start.1 + translation.1, s₁_start.2 + translation.2)
  let s₂_end : ℝ × ℝ := (s₁_end.1 + translation.1, s₁_end.2 + translation.2)
  let s₂_midpoint : ℝ × ℝ := ((s₂_start.1 + s₂_end.1) / 2, (s₂_start.2 + s₂_end.2) / 2)
  s₂_midpoint = (3, 0) := by
  sorry

end midpoint_of_translated_segment_l964_96456


namespace arccos_cos_nine_l964_96417

theorem arccos_cos_nine : Real.arccos (Real.cos 9) = 9 - 2 * Real.pi := by
  sorry

end arccos_cos_nine_l964_96417


namespace tangent_perpendicular_to_line_l964_96407

theorem tangent_perpendicular_to_line (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.cos x
  let f' : ℝ → ℝ := λ x ↦ -Real.sin x
  let tangent_slope : ℝ := f' (π/6)
  tangent_slope * a = -1 → a = 2 := by
  sorry

end tangent_perpendicular_to_line_l964_96407


namespace sales_tax_percentage_l964_96483

theorem sales_tax_percentage 
  (total_bill : ℝ)
  (food_price : ℝ)
  (tip_percentage : ℝ)
  (h1 : total_bill = 158.40)
  (h2 : food_price = 120)
  (h3 : tip_percentage = 0.20)
  : ∃ (tax_percentage : ℝ), 
    tax_percentage = 0.10 ∧ 
    total_bill = (food_price * (1 + tax_percentage) * (1 + tip_percentage)) :=
by sorry

end sales_tax_percentage_l964_96483


namespace vectors_opposite_direction_l964_96416

/-- Given non-zero vectors a and b satisfying a + 4b = 0, prove that the directions of a and b are opposite. -/
theorem vectors_opposite_direction {n : Type*} [NormedAddCommGroup n] [NormedSpace ℝ n] 
  (a b : n) (ha : a ≠ 0) (hb : b ≠ 0) (h : a + 4 • b = 0) : 
  ∃ (k : ℝ), k < 0 ∧ a = k • b :=
sorry

end vectors_opposite_direction_l964_96416


namespace max_value_sqrt_sum_l964_96489

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5) :
  Real.sqrt (a + 1) + Real.sqrt (b + 3) ≤ 3 * Real.sqrt 2 := by
  sorry

end max_value_sqrt_sum_l964_96489


namespace sanchez_rope_theorem_l964_96478

def rope_problem (rope_last_week : ℕ) (rope_difference : ℕ) (inches_per_foot : ℕ) : Prop :=
  let rope_this_week : ℕ := rope_last_week - rope_difference
  let total_rope_feet : ℕ := rope_last_week + rope_this_week
  let total_rope_inches : ℕ := total_rope_feet * inches_per_foot
  total_rope_inches = 96

theorem sanchez_rope_theorem : rope_problem 6 4 12 := by
  sorry

end sanchez_rope_theorem_l964_96478


namespace quadratic_function_properties_l964_96479

def f (a x : ℝ) : ℝ := -x^2 + 2*a*x + 2*a

theorem quadratic_function_properties (a : ℝ) :
  (f a (-1) = -1 → a = 0) ∧
  (f a 3 = -1 → (∀ x ∈ Set.Icc (-2) 3, f a x ≤ 3) ∧ (∃ x ∈ Set.Icc (-2) 3, f a x = -6)) ∧
  (∃ x y : ℝ, x ≠ y ∧ f a x = -1 ∧ f a y = -1 ∧ |x - y| = |2*a + 2|) ∧
  (∃! x : ℝ, x ∈ Set.Icc (a - 1) (2*a + 3) ∧ f a x = -1 ↔ a ≥ 0 ∨ a = -1 ∨ a = -2) :=
sorry

end quadratic_function_properties_l964_96479


namespace sqrt_sum_difference_l964_96418

theorem sqrt_sum_difference : Real.sqrt 18 - Real.sqrt 8 + Real.sqrt 2 = 2 * Real.sqrt 2 := by
  sorry

end sqrt_sum_difference_l964_96418


namespace system_solution_range_l964_96497

/-- Given a system of equations 2x + y = 1 + 4a and x + 2y = 2 - a,
    if x + y > 0, then a > -1 -/
theorem system_solution_range (x y a : ℝ) 
  (eq1 : 2 * x + y = 1 + 4 * a)
  (eq2 : x + 2 * y = 2 - a)
  (h : x + y > 0) : 
  a > -1 := by
  sorry

end system_solution_range_l964_96497


namespace flour_for_two_loaves_l964_96496

/-- The amount of flour needed for one loaf of bread in cups -/
def flour_per_loaf : ℝ := 2.5

/-- The number of loaves of bread to be baked -/
def num_loaves : ℕ := 2

/-- Theorem: The amount of flour needed for two loaves of bread is 5 cups -/
theorem flour_for_two_loaves : flour_per_loaf * num_loaves = 5 := by
  sorry

end flour_for_two_loaves_l964_96496


namespace stream_speed_l964_96459

/-- The speed of the stream given rowing distances and times -/
theorem stream_speed (downstream_distance upstream_distance : ℝ) 
  (downstream_time upstream_time : ℝ) (h1 : downstream_distance = 78) 
  (h2 : upstream_distance = 50) (h3 : downstream_time = 2) (h4 : upstream_time = 2) : 
  ∃ (boat_speed stream_speed : ℝ), 
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧ 
    upstream_distance = (boat_speed - stream_speed) * upstream_time ∧ 
    stream_speed = 7 :=
by sorry

end stream_speed_l964_96459


namespace students_in_both_clubs_l964_96477

theorem students_in_both_clubs 
  (total_students : ℕ) 
  (drama_club : ℕ) 
  (science_club : ℕ) 
  (either_club : ℕ) 
  (h1 : total_students = 300)
  (h2 : drama_club = 120)
  (h3 : science_club = 180)
  (h4 : either_club = 250) :
  drama_club + science_club - either_club = 50 := by
  sorry

#check students_in_both_clubs

end students_in_both_clubs_l964_96477


namespace inequality_proof_l964_96435

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : z ≥ 1)
  (h4 : 1 / (x^2 - 1) + 1 / (y^2 - 1) + 1 / (z^2 - 1) = 1) :
  1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 1 := by
  sorry

end inequality_proof_l964_96435


namespace cube_edge_length_from_sphere_volume_l964_96474

theorem cube_edge_length_from_sphere_volume (V : ℝ) (h : V = 4 * Real.pi / 3) :
  ∃ (a : ℝ), a > 0 ∧ a = 2 * Real.sqrt 3 / 3 ∧
  V = 4 * Real.pi * (3 * a^2 / 4) / 3 :=
sorry

end cube_edge_length_from_sphere_volume_l964_96474


namespace bailey_rawhide_bones_l964_96455

theorem bailey_rawhide_bones (dog_treats chew_toys credit_cards items_per_charge : ℕ) 
  (h1 : dog_treats = 8)
  (h2 : chew_toys = 2)
  (h3 : credit_cards = 4)
  (h4 : items_per_charge = 5) :
  credit_cards * items_per_charge - (dog_treats + chew_toys) = 10 := by
  sorry

end bailey_rawhide_bones_l964_96455


namespace existence_of_cube_sum_equal_100_power_100_l964_96408

theorem existence_of_cube_sum_equal_100_power_100 : 
  ∃ (a b c d : ℕ), a^3 + b^3 + c^3 + d^3 = 100^100 :=
by sorry

end existence_of_cube_sum_equal_100_power_100_l964_96408


namespace intersection_area_is_zero_l964_96427

/-- The first curve: x^2 + y^2 = 16 -/
def curve1 (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- The second curve: (x-3)^2 + y^2 = 9 -/
def curve2 (x y : ℝ) : Prop := (x-3)^2 + y^2 = 9

/-- The set of intersection points of the two curves -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | curve1 p.1 p.2 ∧ curve2 p.1 p.2}

/-- The polygon formed by the intersection points -/
def intersection_polygon : Set (ℝ × ℝ) :=
  intersection_points

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The area of the intersection polygon is 0 -/
theorem intersection_area_is_zero :
  area intersection_polygon = 0 := by sorry

end intersection_area_is_zero_l964_96427


namespace max_value_ln_minus_x_l964_96458

open Real

theorem max_value_ln_minus_x :
  ∃ (x : ℝ), 0 < x ∧ x ≤ exp 1 ∧
  (∀ (y : ℝ), 0 < y ∧ y ≤ exp 1 → log y - y ≤ log x - x) ∧
  log x - x = -1 ∧ x = 1 := by
  sorry

end max_value_ln_minus_x_l964_96458


namespace inequality_solution_l964_96424

theorem inequality_solution (x : ℝ) (hx : x ≠ 0) (hx2 : x^2 ≠ 6) :
  |4 * x^2 - 32 / x| + |x^2 + 5 / (x^2 - 6)| ≤ |3 * x^2 - 5 / (x^2 - 6) - 32 / x| ↔
  (x > -Real.sqrt 6 ∧ x ≤ -Real.sqrt 5) ∨
  (x ≥ -1 ∧ x < 0) ∨
  (x ≥ 1 ∧ x ≤ 2) ∨
  (x ≥ Real.sqrt 5 ∧ x < Real.sqrt 6) :=
by sorry

end inequality_solution_l964_96424


namespace sum_of_digits_of_1962_digit_number_div_by_9_l964_96438

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 1962 digits -/
def has1962Digits (n : ℕ) : Prop := sorry

theorem sum_of_digits_of_1962_digit_number_div_by_9 (n : ℕ) 
  (h1 : has1962Digits n) 
  (h2 : n % 9 = 0) : 
  let a := sumOfDigits n
  let b := sumOfDigits a
  let c := sumOfDigits b
  c = 9 := by sorry

end sum_of_digits_of_1962_digit_number_div_by_9_l964_96438


namespace angle_sum_equals_arctangent_of_ratio_l964_96415

theorem angle_sum_equals_arctangent_of_ratio
  (θ φ : ℝ)
  (θ_acute : 0 < θ ∧ θ < π / 2)
  (φ_acute : 0 < φ ∧ φ < π / 2)
  (tan_θ : Real.tan θ = 2 / 9)
  (sin_φ : Real.sin φ = 3 / 5) :
  θ + 2 * φ = Real.arctan (230 / 15) :=
sorry

end angle_sum_equals_arctangent_of_ratio_l964_96415


namespace triangle_problem_l964_96445

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement for the given triangle problem -/
theorem triangle_problem (t : Triangle) 
  (h_area : (1/2) * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 15)
  (h_diff : t.b - t.c = 2)
  (h_cosA : Real.cos t.A = -1/4) : 
  t.a = 8 ∧ Real.sin t.C = Real.sqrt 15 / 8 ∧ 
  Real.cos (2 * t.A + π/6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16 := by
  sorry


end triangle_problem_l964_96445


namespace sin_difference_product_l964_96401

theorem sin_difference_product (a b : ℝ) : 
  Real.sin (2 * a + b) - Real.sin (2 * a - b) = 2 * Real.cos (2 * a) * Real.sin b := by
  sorry

end sin_difference_product_l964_96401


namespace quadratic_polynomial_sufficiency_necessity_l964_96440

/-- A second-degree polynomial with distinct roots -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  x₁ : ℝ
  x₂ : ℝ
  distinct_roots : x₁ ≠ x₂
  is_root_x₁ : a * x₁^2 + b * x₁ + c = 0
  is_root_x₂ : a * x₂^2 + b * x₂ + c = 0

/-- The value of the polynomial at a given x -/
def QuadraticPolynomial.value (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem quadratic_polynomial_sufficiency_necessity 
    (p : QuadraticPolynomial) : 
    (p.a^2 + 3*p.a*p.c - p.b^2 = 0 → p.value (p.x₁^3) = p.value (p.x₂^3)) ∧
    (∃ p : QuadraticPolynomial, p.value (p.x₁^3) = p.value (p.x₂^3) ∧ p.a^2 + 3*p.a*p.c - p.b^2 ≠ 0) :=
  sorry

end quadratic_polynomial_sufficiency_necessity_l964_96440


namespace count_m_with_integer_roots_l964_96460

def quadratic_equation (m : ℤ) (x : ℤ) : Prop :=
  x^2 - m*x + m + 2006 = 0

def has_integer_roots (m : ℤ) : Prop :=
  ∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ quadratic_equation m a ∧ quadratic_equation m b

theorem count_m_with_integer_roots :
  ∃! (S : Finset ℤ), (∀ m : ℤ, m ∈ S ↔ has_integer_roots m) ∧ S.card = 5 :=
sorry

end count_m_with_integer_roots_l964_96460


namespace max_value_inequality_l964_96491

theorem max_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (M : ℝ), M = 1/2 ∧ 
  (∀ (N : ℝ), (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0),
    x^3 + y^3 + z^3 - 3*x*y*z ≥ N*(|x-y|^3 + |x-z|^3 + |z-y|^3)) → N ≤ M) ∧
  (a^3 + b^3 + c^3 - 3*a*b*c ≥ M*(|a-b|^3 + |a-c|^3 + |c-b|^3)) := by
sorry

end max_value_inequality_l964_96491


namespace linear_function_shift_l964_96441

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Shifts a linear function horizontally -/
def shiftHorizontal (f : LinearFunction) (units : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept + f.slope * units }

/-- Shifts a linear function vertically -/
def shiftVertical (f : LinearFunction) (units : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept - units }

/-- The theorem to be proved -/
theorem linear_function_shift :
  let f := LinearFunction.mk 3 2
  let f_shifted_left := shiftHorizontal f 3
  let f_final := shiftVertical f_shifted_left 1
  f_final = LinearFunction.mk 3 10 := by sorry

end linear_function_shift_l964_96441


namespace f_value_at_one_l964_96444

/-- A quadratic function f(x) with a specific behavior -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- The function is increasing on [-2, +∞) -/
def increasing_on_right (m : ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y → f m x < f m y

/-- The function is decreasing on (-∞, -2] -/
def decreasing_on_left (m : ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ -2 → f m x > f m y

theorem f_value_at_one (m : ℝ) 
  (h1 : increasing_on_right m) 
  (h2 : decreasing_on_left m) : 
  f m 1 = 25 := by sorry

end f_value_at_one_l964_96444


namespace circle_area_around_equilateral_triangle_l964_96428

theorem circle_area_around_equilateral_triangle :
  let side_length : ℝ := 12
  let circumradius : ℝ := side_length / Real.sqrt 3
  let circle_area : ℝ := Real.pi * circumradius^2
  circle_area = 48 * Real.pi :=
by sorry

end circle_area_around_equilateral_triangle_l964_96428


namespace heather_starting_blocks_l964_96426

/-- The number of blocks Heather shared with Jose -/
def shared_blocks : ℕ := 41

/-- The number of blocks Heather ended up with -/
def remaining_blocks : ℕ := 45

/-- The total number of blocks Heather started with -/
def starting_blocks : ℕ := shared_blocks + remaining_blocks

theorem heather_starting_blocks : starting_blocks = 86 := by
  sorry

end heather_starting_blocks_l964_96426


namespace right_triangle_sets_l964_96402

def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

theorem right_triangle_sets :
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle (Real.sqrt 7) 3 5 ∧
  is_right_triangle 6 8 10 ∧
  ¬ is_right_triangle 5 12 12 :=
by sorry

end right_triangle_sets_l964_96402
