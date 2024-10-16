import Mathlib

namespace NUMINAMATH_CALUDE_lucky_coin_steps_l1054_105498

/-- Represents the state of a coin on the number line -/
inductive CoinState
| HeadsUp
| TailsUp
| NoCoin

/-- Represents the direction Lucky is facing -/
inductive Direction
| Positive
| Negative

/-- Represents Lucky's position and the state of the number line -/
structure GameState where
  position : Int
  direction : Direction
  coins : Int → CoinState

/-- Represents the procedure Lucky follows -/
def step (state : GameState) : GameState :=
  sorry

/-- Counts the number of tails-up coins -/
def countTailsUp (coins : Int → CoinState) : Nat :=
  sorry

/-- Theorem stating that the process stops after 6098 steps -/
theorem lucky_coin_steps :
  ∀ (initial : GameState),
    initial.position = 0 ∧
    initial.direction = Direction.Positive ∧
    (∀ n : Int, initial.coins n = CoinState.HeadsUp) →
    ∃ (final : GameState) (steps : Nat),
      steps = 6098 ∧
      countTailsUp final.coins = 20 ∧
      (∀ k : Nat, k < steps → countTailsUp (step^[k] initial).coins < 20) :=
  sorry

end NUMINAMATH_CALUDE_lucky_coin_steps_l1054_105498


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1054_105425

/-- 
For a quadratic equation qx^2 - 20x + 9 = 0 to have exactly one solution,
q must equal 100/9.
-/
theorem unique_solution_quadratic : 
  ∃! q : ℚ, q ≠ 0 ∧ (∃! x : ℝ, q * x^2 - 20 * x + 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1054_105425


namespace NUMINAMATH_CALUDE_standard_colony_requirements_l1054_105446

/-- Represents the type of culture medium -/
inductive CultureMedium
| Liquid
| Solid

/-- Represents the number of initial bacteria -/
inductive InitialBacteria
| One
| Many

/-- Defines a bacterial colony -/
structure BacterialColony where
  origin : InitialBacteria
  medium : CultureMedium
  visible_mass : Bool
  single_mother_cell : Bool
  significant_for_identification : Bool

/-- Defines a standard bacterial colony -/
def is_standard_colony (colony : BacterialColony) : Prop :=
  colony.origin = InitialBacteria.One ∧
  colony.medium = CultureMedium.Solid ∧
  colony.visible_mass ∧
  colony.single_mother_cell ∧
  colony.significant_for_identification

theorem standard_colony_requirements :
  ∀ (colony : BacterialColony),
    colony.visible_mass ∧
    colony.single_mother_cell ∧
    colony.significant_for_identification →
    is_standard_colony colony ↔
      colony.origin = InitialBacteria.One ∧
      colony.medium = CultureMedium.Solid :=
by sorry

end NUMINAMATH_CALUDE_standard_colony_requirements_l1054_105446


namespace NUMINAMATH_CALUDE_table_covering_l1054_105485

/-- Represents a cell in the table -/
inductive Cell
| Zero
| One

/-- Represents the 1000x1000 table -/
def Table := Fin 1000 → Fin 1000 → Cell

/-- Checks if a set of rows covers all columns with at least one 1 -/
def coversColumnsWithOnes (t : Table) (rows : Finset (Fin 1000)) : Prop :=
  ∀ j : Fin 1000, ∃ i ∈ rows, t i j = Cell.One

/-- Checks if a set of columns covers all rows with at least one 0 -/
def coversRowsWithZeros (t : Table) (cols : Finset (Fin 1000)) : Prop :=
  ∀ i : Fin 1000, ∃ j ∈ cols, t i j = Cell.Zero

/-- The main theorem -/
theorem table_covering (t : Table) :
  (∃ rows : Finset (Fin 1000), rows.card = 10 ∧ coversColumnsWithOnes t rows) ∨
  (∃ cols : Finset (Fin 1000), cols.card = 10 ∧ coversRowsWithZeros t cols) :=
sorry

end NUMINAMATH_CALUDE_table_covering_l1054_105485


namespace NUMINAMATH_CALUDE_map_length_l1054_105426

theorem map_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 10 → area = 20 → area = width * length → length = 2 := by
sorry

end NUMINAMATH_CALUDE_map_length_l1054_105426


namespace NUMINAMATH_CALUDE_rectangle_tiling_l1054_105480

/-- A rectangle can be perfectly tiled by unit-width strips if and only if
    at least one of its dimensions is an integer. -/
theorem rectangle_tiling (a b : ℝ) :
  (∃ (n : ℕ), a * b = n) →
  (∃ (k : ℕ), a = k ∨ b = k) := by sorry

end NUMINAMATH_CALUDE_rectangle_tiling_l1054_105480


namespace NUMINAMATH_CALUDE_first_part_days_count_l1054_105466

/-- Proves that the number of days in the first part of the week is 3, given the expenditure conditions. -/
theorem first_part_days_count (x : ℕ) : 
  (350 * x + 420 * 4 = 390 * (x + 4)) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_part_days_count_l1054_105466


namespace NUMINAMATH_CALUDE_gcd_4370_13824_l1054_105420

theorem gcd_4370_13824 : Nat.gcd 4370 13824 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_4370_13824_l1054_105420


namespace NUMINAMATH_CALUDE_water_depth_at_points_l1054_105458

/-- The depth function that calculates water depth based on Ron's height -/
def depth (x : ℝ) : ℝ := 16 * x

/-- Ron's height at point A -/
def ronHeightA : ℝ := 13

/-- Ron's height at point B -/
def ronHeightB : ℝ := ronHeightA + 4

/-- Theorem: The depth of water at points A and B -/
theorem water_depth_at_points : 
  depth ronHeightA = 208 ∧ depth ronHeightB = 272 := by
  sorry

/-- Dean's height relative to Ron -/
def deanHeight (ronHeight : ℝ) : ℝ := ronHeight + 9

/-- Alex's height relative to Dean -/
def alexHeight (deanHeight : ℝ) : ℝ := deanHeight - 5

end NUMINAMATH_CALUDE_water_depth_at_points_l1054_105458


namespace NUMINAMATH_CALUDE_man_speed_l1054_105474

/-- The speed of a man running opposite to a bullet train --/
theorem man_speed (train_length : ℝ) (train_speed : ℝ) (passing_time : ℝ) :
  train_length = 200 →
  train_speed = 69 →
  passing_time = 10 →
  ∃ (man_speed : ℝ), 
    (man_speed ≥ 2.9 ∧ man_speed ≤ 3.1) ∧
    (train_length / passing_time = train_speed * (1000 / 3600) + man_speed) :=
by sorry

end NUMINAMATH_CALUDE_man_speed_l1054_105474


namespace NUMINAMATH_CALUDE_distinct_c_values_l1054_105402

theorem distinct_c_values (r s t u : ℂ) (h_distinct : r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ s ≠ t ∧ s ≠ u ∧ t ≠ u) 
  (h_eq : ∀ z : ℂ, (z - r) * (z - s) * (z - t) * (z - u) = 
    (z - r * c) * (z - s * c) * (z - t * c) * (z - u * c)) : 
  ∃! (values : Finset ℂ), values.card = 4 ∧ ∀ c : ℂ, c ∈ values ↔ 
    (∀ z : ℂ, (z - r) * (z - s) * (z - t) * (z - u) = 
      (z - r * c) * (z - s * c) * (z - t * c) * (z - u * c)) :=
by sorry

end NUMINAMATH_CALUDE_distinct_c_values_l1054_105402


namespace NUMINAMATH_CALUDE_negation_equivalence_l1054_105441

-- Define the universe of switches and lights
variable (Switch Light : Type)

-- Define the state of switches and lights
variable (is_off : Switch → Prop)
variable (is_on : Light → Prop)

-- Define the main switch
variable (main_switch : Switch)

-- Define the conditions
variable (h1 : ∀ s : Switch, is_off s → ∀ l : Light, ¬(is_on l))
variable (h2 : is_off main_switch → ∀ s : Switch, is_off s)

-- The theorem to prove
theorem negation_equivalence :
  ¬(is_off main_switch → ∀ l : Light, ¬(is_on l)) ↔
  (is_off main_switch ∧ ∃ l : Light, is_on l) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1054_105441


namespace NUMINAMATH_CALUDE_max_consecutive_even_sum_l1054_105487

/-- The sum of k consecutive even integers starting from 2n is 156 -/
def ConsecutiveEvenSum (n k : ℕ) : Prop :=
  2 * k * n + k * (k - 1) = 156

/-- The proposition that 4 is the maximum number of consecutive even integers summing to 156 -/
theorem max_consecutive_even_sum :
  (∃ n : ℕ, ConsecutiveEvenSum n 4) ∧
  (∀ k : ℕ, k > 4 → ¬∃ n : ℕ, ConsecutiveEvenSum n k) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_even_sum_l1054_105487


namespace NUMINAMATH_CALUDE_sum_of_roots_of_equation_l1054_105401

theorem sum_of_roots_of_equation (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁ - 3)^2 = 16 ∧ (x₂ - 3)^2 = 16 ∧ x₁ + x₂ = 6) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_of_equation_l1054_105401


namespace NUMINAMATH_CALUDE_complex_root_modulus_one_l1054_105473

theorem complex_root_modulus_one (n : ℕ) :
  (∃ z : ℂ, z^(n+1) - z^n - 1 = 0 ∧ Complex.abs z = 1) ↔ 6 ∣ (n + 2) :=
sorry

end NUMINAMATH_CALUDE_complex_root_modulus_one_l1054_105473


namespace NUMINAMATH_CALUDE_inequality_proof_l1054_105424

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ 0) :
  (x^2 * y) / z + (y^2 * z) / x + (z^2 * x) / y ≥ x^2 + y^2 + z^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1054_105424


namespace NUMINAMATH_CALUDE_opposite_numbers_cube_inequality_l1054_105494

theorem opposite_numbers_cube_inequality (a b : ℝ) (h1 : a = -b) (h2 : a ≠ 0) : a^3 ≠ b^3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_cube_inequality_l1054_105494


namespace NUMINAMATH_CALUDE_fourth_boy_payment_l1054_105407

theorem fourth_boy_payment (a b c d : ℝ) : 
  a + b + c + d = 80 →
  a = (1/2) * (b + c + d) →
  b = (1/4) * (a + c + d) →
  c = (1/3) * (a + b + d) →
  d + 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_fourth_boy_payment_l1054_105407


namespace NUMINAMATH_CALUDE_triangle_angle_B_l1054_105455

theorem triangle_angle_B (a b : ℝ) (A : ℝ) (h1 : a = 4) (h2 : b = 4 * Real.sqrt 3) (h3 : A = 30 * π / 180) :
  let B := Real.arcsin ((b * Real.sin A) / a)
  B = 60 * π / 180 ∨ B = 120 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l1054_105455


namespace NUMINAMATH_CALUDE_equation_solution_set_l1054_105409

def solution_set : Set Int := {0, -6, -12, -14, -18, -20, -21, -24, -26, -27, -28, -30, -32, -33, -34, -35, -36, -38, -39, -40, -41, -44, -45, -46, -47, -49, -50, -51, -52, -53, -55, -57, -58, -59, -61, -64, -65, -67, -71, -73, -79, -85}

theorem equation_solution_set :
  {x : Int | Int.floor (x / 2) + Int.floor (x / 3) + Int.floor (x / 7) = x} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_set_l1054_105409


namespace NUMINAMATH_CALUDE_planes_perpendicular_l1054_105411

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (a b c : Line) 
  (α β γ : Plane) 
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h2 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) 
  (h3 : parallel_line_plane a α) 
  (h4 : contained_in b β) 
  (h5 : parallel_lines a b) : 
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l1054_105411


namespace NUMINAMATH_CALUDE_brads_running_speed_l1054_105454

/-- Calculates Brad's running speed given the problem conditions -/
theorem brads_running_speed 
  (maxwell_speed : ℝ) 
  (total_distance : ℝ) 
  (maxwell_time : ℝ) 
  (brad_delay : ℝ) 
  (h1 : maxwell_speed = 4)
  (h2 : total_distance = 14)
  (h3 : maxwell_time = 2)
  (h4 : brad_delay = 1) : 
  (total_distance - maxwell_speed * maxwell_time) / (maxwell_time - brad_delay) = 6 := by
  sorry

#check brads_running_speed

end NUMINAMATH_CALUDE_brads_running_speed_l1054_105454


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_distance_to_8_15_l1054_105423

theorem distance_from_origin_to_point : ℝ → ℝ → ℝ
  | x, y => Real.sqrt (x^2 + y^2)

theorem distance_to_8_15 :
  distance_from_origin_to_point 8 15 = 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_distance_to_8_15_l1054_105423


namespace NUMINAMATH_CALUDE_line_slope_through_point_with_x_intercept_l1054_105469

/-- Given a line passing through the point (3, 4) with an x-intercept of 1, 
    its slope is 2. -/
theorem line_slope_through_point_with_x_intercept : 
  ∀ (f : ℝ → ℝ), 
    (∃ m b : ℝ, ∀ x, f x = m * x + b) →  -- f is a linear function
    f 3 = 4 →                           -- f passes through (3, 4)
    f 1 = 0 →                           -- x-intercept is 1
    ∃ m : ℝ, (∀ x, f x = m * x + b) ∧ m = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_line_slope_through_point_with_x_intercept_l1054_105469


namespace NUMINAMATH_CALUDE_pencil_sales_theorem_l1054_105460

/-- The number of pencils initially sold for a rupee when losing 30% --/
def initial_pencils : ℝ := 20

/-- The number of pencils sold for a rupee when gaining 30% --/
def gain_pencils : ℝ := 10.77

/-- The percentage of cost price when losing 30% --/
def loss_percentage : ℝ := 0.7

/-- The percentage of cost price when gaining 30% --/
def gain_percentage : ℝ := 1.3

theorem pencil_sales_theorem :
  initial_pencils * loss_percentage = gain_pencils * gain_percentage := by
  sorry

#check pencil_sales_theorem

end NUMINAMATH_CALUDE_pencil_sales_theorem_l1054_105460


namespace NUMINAMATH_CALUDE_pet_store_ratio_l1054_105413

theorem pet_store_ratio (num_cats : ℕ) (num_dogs : ℕ) : 
  (num_cats : ℚ) / num_dogs = 3 / 4 →
  num_cats = 18 →
  num_dogs = 24 := by
sorry

end NUMINAMATH_CALUDE_pet_store_ratio_l1054_105413


namespace NUMINAMATH_CALUDE_alison_bought_six_small_tubs_l1054_105421

/-- Represents the number of small tubs Alison bought -/
def num_small_tubs : ℕ := sorry

/-- Represents the number of large tubs Alison bought -/
def num_large_tubs : ℕ := 3

/-- Represents the cost of a large tub in dollars -/
def cost_large_tub : ℕ := 6

/-- Represents the cost of a small tub in dollars -/
def cost_small_tub : ℕ := 5

/-- Represents the total cost of all tubs in dollars -/
def total_cost : ℕ := 48

/-- Theorem stating that Alison bought 6 small tubs -/
theorem alison_bought_six_small_tubs :
  num_large_tubs * cost_large_tub + num_small_tubs * cost_small_tub = total_cost →
  num_small_tubs = 6 := by
  sorry

end NUMINAMATH_CALUDE_alison_bought_six_small_tubs_l1054_105421


namespace NUMINAMATH_CALUDE_units_digit_of_square_l1054_105417

theorem units_digit_of_square (n : ℕ) : (n ^ 2) % 10 ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_square_l1054_105417


namespace NUMINAMATH_CALUDE_equal_coffee_and_milk_consumed_l1054_105436

/-- Represents the contents of a cup --/
structure CupContents where
  coffee : ℚ
  milk : ℚ

/-- Represents the drinking and refilling process --/
def drinkAndRefill (contents : CupContents) (amount : ℚ) : CupContents :=
  let remainingCoffee := contents.coffee * (1 - amount)
  let remainingMilk := contents.milk * (1 - amount)
  { coffee := remainingCoffee,
    milk := 1 - remainingCoffee }

/-- The main theorem stating that equal amounts of coffee and milk are consumed --/
theorem equal_coffee_and_milk_consumed :
  let initial := { coffee := 1, milk := 0 }
  let step1 := drinkAndRefill initial (1/6)
  let step2 := drinkAndRefill step1 (1/3)
  let step3 := drinkAndRefill step2 (1/2)
  let finalDrink := 1 - step3.coffee - step3.milk
  1 - initial.coffee + finalDrink = 1 := by sorry

end NUMINAMATH_CALUDE_equal_coffee_and_milk_consumed_l1054_105436


namespace NUMINAMATH_CALUDE_rectangle_area_l1054_105439

/-- Rectangle ABCD with point E on CD and point F on AC -/
structure Rectangle :=
  (A B C D E F : ℝ × ℝ)

/-- The area of a triangle given its vertices -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

theorem rectangle_area (rect : Rectangle) : 
  -- Point E is located one-third of the way along side CD
  rect.E.1 = rect.D.1 + (rect.C.1 - rect.D.1) / 3 ∧
  rect.E.2 = rect.D.2 →
  -- AB is twice the length of BC
  rect.A.1 - rect.B.1 = 2 * (rect.B.2 - rect.C.2) →
  -- Line BE intersects diagonal AC at point F
  ∃ t : ℝ, rect.F = (1 - t) • rect.A + t • rect.C ∧
          ∃ s : ℝ, rect.F = (1 - s) • rect.B + s • rect.E →
  -- The area of triangle BFE is 18
  triangleArea rect.B rect.F rect.E = 18 →
  -- The area of rectangle ABCD is 108
  (rect.A.1 - rect.D.1) * (rect.A.2 - rect.D.2) = 108 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1054_105439


namespace NUMINAMATH_CALUDE_golden_ratio_properties_l1054_105452

theorem golden_ratio_properties : ∃ a : ℝ, 
  (a = (Real.sqrt 5 - 1) / 2) ∧ 
  (a^2 + a - 1 = 0) ∧ 
  (a^3 - 2*a + 2015 = 2014) := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_properties_l1054_105452


namespace NUMINAMATH_CALUDE_classroom_notebooks_l1054_105430

theorem classroom_notebooks (total_students : ℕ) 
  (notebooks_group1 : ℕ) (notebooks_group2 : ℕ) : 
  total_students = 28 → 
  notebooks_group1 = 5 → 
  notebooks_group2 = 3 → 
  (total_students / 2) * notebooks_group1 + (total_students / 2) * notebooks_group2 = 112 := by
  sorry

end NUMINAMATH_CALUDE_classroom_notebooks_l1054_105430


namespace NUMINAMATH_CALUDE_division_scaling_l1054_105415

theorem division_scaling (a b c : ℝ) (h : a / b = c) :
  (a / 10) / (b / 10) = c := by
  sorry

end NUMINAMATH_CALUDE_division_scaling_l1054_105415


namespace NUMINAMATH_CALUDE_seven_sum_problem_l1054_105493

theorem seven_sum_problem :
  ∃ (S : Finset ℕ), (Finset.card S = 108) ∧ 
  (∀ n : ℕ, n ∈ S ↔ 
    ∃ a b c : ℕ, (7 * a + 77 * b + 777 * c = 7000) ∧ 
                 (a + 2 * b + 3 * c = n)) :=
sorry

end NUMINAMATH_CALUDE_seven_sum_problem_l1054_105493


namespace NUMINAMATH_CALUDE_basketball_team_cutoff_l1054_105472

theorem basketball_team_cutoff (girls boys callback : ℕ) 
  (h1 : girls = 17) 
  (h2 : boys = 32) 
  (h3 : callback = 10) : 
  girls + boys - callback = 39 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_cutoff_l1054_105472


namespace NUMINAMATH_CALUDE_min_degree_g_l1054_105431

variables (x : ℝ) (f g h : ℝ → ℝ)

def is_polynomial (p : ℝ → ℝ) : Prop := sorry

def degree (p : ℝ → ℝ) : ℕ := sorry

theorem min_degree_g 
  (eq : ∀ x, 5 * f x + 7 * g x = h x)
  (f_poly : is_polynomial f)
  (g_poly : is_polynomial g)
  (h_poly : is_polynomial h)
  (f_deg : degree f = 10)
  (h_deg : degree h = 13) :
  degree g ≥ 13 ∧ ∃ g', is_polynomial g' ∧ degree g' = 13 ∧ ∀ x, 5 * f x + 7 * g' x = h x :=
sorry

end NUMINAMATH_CALUDE_min_degree_g_l1054_105431


namespace NUMINAMATH_CALUDE_bridge_length_l1054_105491

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 → 
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 225 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l1054_105491


namespace NUMINAMATH_CALUDE_digit_150_is_5_l1054_105496

/-- The decimal representation of 5/13 as a list of digits -/
def decimal_rep_5_13 : List Nat := [3, 8, 4, 6, 1, 5]

/-- The length of the repeating sequence in the decimal representation of 5/13 -/
def repeat_length : Nat := 6

/-- The 150th digit after the decimal point in the decimal representation of 5/13 -/
def digit_150 : Nat :=
  decimal_rep_5_13[(150 - 1) % repeat_length]

theorem digit_150_is_5 : digit_150 = 5 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_5_l1054_105496


namespace NUMINAMATH_CALUDE_min_value_sqrt_expression_l1054_105449

theorem min_value_sqrt_expression (x : ℝ) (hx : x > 0) :
  4 * Real.sqrt x + 2 / Real.sqrt x ≥ 6 ∧
  ∃ y : ℝ, y > 0 ∧ 4 * Real.sqrt y + 2 / Real.sqrt y = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_expression_l1054_105449


namespace NUMINAMATH_CALUDE_team_a_games_l1054_105478

theorem team_a_games (a : ℕ) : 
  (2 : ℚ) / 3 * a + (1 : ℚ) / 3 * a = a → -- Team A's wins + losses = total games
  (3 : ℚ) / 5 * (a + 12) = (2 : ℚ) / 3 * a + 6 → -- Team B's wins = Team A's wins + 6
  (2 : ℚ) / 5 * (a + 12) = (1 : ℚ) / 3 * a + 6 → -- Team B's losses = Team A's losses + 6
  a = 18 := by
sorry

end NUMINAMATH_CALUDE_team_a_games_l1054_105478


namespace NUMINAMATH_CALUDE_sphere_volume_from_cylinder_volume_l1054_105483

/-- The volume of a sphere with the same radius as a cylinder of volume 72π -/
theorem sphere_volume_from_cylinder_volume (r : ℝ) (h : ℝ) :
  (π * r^2 * h = 72 * π) →
  ((4 / 3) * π * r^3 = 48 * π) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_from_cylinder_volume_l1054_105483


namespace NUMINAMATH_CALUDE_trapezoid_area_l1054_105438

-- Define a trapezoid
structure Trapezoid :=
  (smaller_base : ℝ)
  (adjacent_angle : ℝ)
  (diagonal_angle : ℝ)

-- Define the area function for a trapezoid
def area (t : Trapezoid) : ℝ := sorry

-- Theorem statement
theorem trapezoid_area (t : Trapezoid) :
  t.smaller_base = 2 ∧
  t.adjacent_angle = 135 ∧
  t.diagonal_angle = 150 →
  area t = 2 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1054_105438


namespace NUMINAMATH_CALUDE_limit_of_f_l1054_105477

open Real

noncomputable def f (x : ℝ) : ℝ :=
  tan (cos x + sin ((x - 1) / (x + 1)) * cos ((x + 1) / (x - 1)))

theorem limit_of_f :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → |f x - tan (cos 1)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_f_l1054_105477


namespace NUMINAMATH_CALUDE_all_points_fit_l1054_105475

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the square
def square : Set Point :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ max (|x| + |y|) (|x - y|) ≤ 2}

-- Define a translation
def translate (p : Point) (v : ℝ × ℝ) : Point :=
  (p.1 + v.1, p.2 + v.2)

-- Define the property that any three points can be translated to fit in the square
def any_three_fit (S : Set Point) :=
  ∀ (p1 p2 p3 : Point), p1 ∈ S → p2 ∈ S → p3 ∈ S →
    ∃ (v : ℝ × ℝ), {translate p1 v, translate p2 v, translate p3 v} ⊆ square

-- Theorem statement
theorem all_points_fit (S : Set Point) (h : any_three_fit S) :
  ∃ (v : ℝ × ℝ), ∀ (p : Point), p ∈ S → translate p v ∈ square :=
sorry

end NUMINAMATH_CALUDE_all_points_fit_l1054_105475


namespace NUMINAMATH_CALUDE_factors_of_M_l1054_105408

/-- The number of natural-number factors of M, where M = 2^5 · 3^4 · 5^3 · 7^3 · 11^2 -/
def num_factors (M : ℕ) : ℕ :=
  (5 + 1) * (4 + 1) * (3 + 1) * (3 + 1) * (2 + 1)

/-- Theorem stating that the number of natural-number factors of M is 1440 -/
theorem factors_of_M :
  let M : ℕ := 2^5 * 3^4 * 5^3 * 7^3 * 11^2
  num_factors M = 1440 := by sorry

end NUMINAMATH_CALUDE_factors_of_M_l1054_105408


namespace NUMINAMATH_CALUDE_total_travel_time_is_58_hours_l1054_105445

/-- Represents the travel times between cities -/
structure TravelTimes where
  newOrleansToNewYork : ℝ
  newYorkToSanFrancisco : ℝ
  layoverInNewYork : ℝ

/-- The total travel time from New Orleans to San Francisco -/
def totalTravelTime (t : TravelTimes) : ℝ :=
  t.newOrleansToNewYork + t.layoverInNewYork + t.newYorkToSanFrancisco

/-- Theorem stating the total travel time is 58 hours -/
theorem total_travel_time_is_58_hours (t : TravelTimes) 
  (h1 : t.newOrleansToNewYork = 3/4 * t.newYorkToSanFrancisco)
  (h2 : t.newYorkToSanFrancisco = 24)
  (h3 : t.layoverInNewYork = 16) : 
  totalTravelTime t = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_time_is_58_hours_l1054_105445


namespace NUMINAMATH_CALUDE_soccer_camp_ratio_l1054_105488

theorem soccer_camp_ratio :
  let total_kids : ℕ := 2000
  let soccer_kids : ℕ := total_kids / 2
  let afternoon_soccer_kids : ℕ := 750
  let morning_soccer_kids : ℕ := soccer_kids - afternoon_soccer_kids
  (morning_soccer_kids : ℚ) / (soccer_kids : ℚ) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_soccer_camp_ratio_l1054_105488


namespace NUMINAMATH_CALUDE_smallest_k_for_distinct_roots_l1054_105471

theorem smallest_k_for_distinct_roots (k : ℤ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 3 * x - 9/4 = 0 ∧ k * y^2 - 3 * y - 9/4 = 0) →
  k ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_distinct_roots_l1054_105471


namespace NUMINAMATH_CALUDE_lees_friend_money_l1054_105406

/-- 
Given:
- Lee had $10
- The total cost of the meal was $15 (including tax)
- They received $3 in change
- The total amount paid was $18

Prove that Lee's friend had $8 initially.
-/
theorem lees_friend_money (lee_money : ℕ) (meal_cost : ℕ) (change : ℕ) (total_paid : ℕ)
  (h1 : lee_money = 10)
  (h2 : meal_cost = 15)
  (h3 : change = 3)
  (h4 : total_paid = 18)
  : total_paid - lee_money = 8 := by
  sorry

end NUMINAMATH_CALUDE_lees_friend_money_l1054_105406


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l1054_105456

theorem abs_sum_inequality (x : ℝ) : |x - 1| + |x - 3| < 8 ↔ -2 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l1054_105456


namespace NUMINAMATH_CALUDE_four_digit_integer_problem_l1054_105453

theorem four_digit_integer_problem (a b c d : ℕ) : 
  a * 1000 + b * 100 + c * 10 + d > 0 →
  a + b + c + d = 16 →
  b + c = 8 →
  a - d = 2 →
  (a * 1000 + b * 100 + c * 10 + d) % 9 = 0 →
  a * 1000 + b * 100 + c * 10 + d = 5533 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_integer_problem_l1054_105453


namespace NUMINAMATH_CALUDE_factorization_proof_l1054_105429

theorem factorization_proof (x : ℝ) : 4*x*(x-5) + 7*(x-5) + 12*(x-5) = (4*x + 19)*(x-5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1054_105429


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l1054_105457

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: No positive integer n satisfies n + S(n) + S(S(n)) = 2099 -/
theorem no_solution_for_equation : ¬ ∃ (n : ℕ), n > 0 ∧ n + S n + S (S n) = 2099 := by sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l1054_105457


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l1054_105468

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l1054_105468


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1054_105428

/-- An isosceles triangle with sides of length 4 and 8 has a perimeter of 20 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive side lengths
  (a = 4 ∧ b = 8 ∧ c = 8) ∨ (a = 8 ∧ b = 4 ∧ c = 8) →  -- possible configurations
  a + b > c ∧ b + c > a ∧ a + c > b →  -- triangle inequality
  a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1054_105428


namespace NUMINAMATH_CALUDE_arithmetic_mean_negative_seven_to_six_l1054_105476

def arithmetic_mean (a b : Int) : ℚ :=
  let n := b - a + 1
  let sum := (a + b) * n / 2
  sum / n

theorem arithmetic_mean_negative_seven_to_six :
  arithmetic_mean (-7) 6 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_negative_seven_to_six_l1054_105476


namespace NUMINAMATH_CALUDE_eventual_bounded_groups_l1054_105486

/-- Represents a group distribution in the society -/
def GroupDistribution := List Nat

/-- The redistribution process for a week -/
def redistribute : GroupDistribution → GroupDistribution := sorry

/-- Checks if all groups in a distribution have size at most 1 + √(2n) -/
def all_groups_bounded (n : Nat) (dist : GroupDistribution) : Prop := sorry

/-- Theorem: Eventually, all groups will be bounded by 1 + √(2n) -/
theorem eventual_bounded_groups (n : Nat) :
  ∃ (k : Nat), all_groups_bounded n ((redistribute^[k]) [n]) := by
  sorry

end NUMINAMATH_CALUDE_eventual_bounded_groups_l1054_105486


namespace NUMINAMATH_CALUDE_clients_using_radio_l1054_105490

/-- The number of clients using radio in an advertising agency with given client distribution. -/
theorem clients_using_radio (total : ℕ) (tv : ℕ) (mag : ℕ) (tv_mag : ℕ) (tv_radio : ℕ) (radio_mag : ℕ) (all_three : ℕ) :
  total = 180 →
  tv = 115 →
  mag = 130 →
  tv_mag = 85 →
  tv_radio = 75 →
  radio_mag = 95 →
  all_three = 80 →
  ∃ radio : ℕ, radio = 30 ∧ 
    total = tv + radio + mag - tv_mag - tv_radio - radio_mag + all_three :=
by
  sorry


end NUMINAMATH_CALUDE_clients_using_radio_l1054_105490


namespace NUMINAMATH_CALUDE_woman_work_days_l1054_105442

/-- A woman's work and pay scenario -/
theorem woman_work_days (total_days : ℕ) (pay_per_day : ℕ) (forfeit_per_day : ℕ) (net_earnings : ℕ) 
    (h1 : total_days = 25)
    (h2 : pay_per_day = 20)
    (h3 : forfeit_per_day = 5)
    (h4 : net_earnings = 450) :
  ∃ (work_days : ℕ), 
    work_days ≤ total_days ∧ 
    (pay_per_day * work_days - forfeit_per_day * (total_days - work_days) = net_earnings) ∧
    work_days = 23 := by
  sorry

end NUMINAMATH_CALUDE_woman_work_days_l1054_105442


namespace NUMINAMATH_CALUDE_largest_ratio_in_arithmetic_sequence_l1054_105400

/-- Given an arithmetic sequence {a_n} with S_n as the sum of its first n terms,
    if S_15 > 0 and S_16 < 0, then S_8/a_8 is the largest among S_1/a_1, S_2/a_2, ..., S_15/a_15 -/
theorem largest_ratio_in_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2) 
  (h_S15 : S 15 > 0) 
  (h_S16 : S 16 < 0) : 
  ∀ k ∈ Finset.range 15, S 8 / a 8 ≥ S (k + 1) / a (k + 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_ratio_in_arithmetic_sequence_l1054_105400


namespace NUMINAMATH_CALUDE_num_successful_sequences_l1054_105412

/-- Represents the number of cards in the game -/
def num_cards : ℕ := 13

/-- Represents the number of cards that need to be flipped for success -/
def cards_to_flip : ℕ := 12

/-- Represents the number of choices for each flip after the first -/
def choices_per_flip : ℕ := 2

/-- Represents the rules of the card flipping game -/
structure CardGame where
  cards : Fin num_cards → Bool
  is_valid_flip : Fin num_cards → Bool

/-- Theorem stating the number of successful flip sequences -/
theorem num_successful_sequences (game : CardGame) :
  (num_cards : ℕ) * (choices_per_flip ^ (cards_to_flip - 1) : ℕ) = 26624 := by
  sorry

end NUMINAMATH_CALUDE_num_successful_sequences_l1054_105412


namespace NUMINAMATH_CALUDE_one_dollar_bills_count_l1054_105497

/-- Represents the number of bills of each denomination -/
structure WalletContent where
  ones : ℕ
  twos : ℕ
  fives : ℕ

/-- Calculates the total number of bills -/
def total_bills (w : WalletContent) : ℕ :=
  w.ones + w.twos + w.fives

/-- Calculates the total amount of money -/
def total_money (w : WalletContent) : ℕ :=
  w.ones + 2 * w.twos + 5 * w.fives

/-- Theorem stating that given the conditions, the number of one dollar bills is 20 -/
theorem one_dollar_bills_count (w : WalletContent) 
  (h1 : total_bills w = 60) 
  (h2 : total_money w = 120) : 
  w.ones = 20 := by
  sorry

end NUMINAMATH_CALUDE_one_dollar_bills_count_l1054_105497


namespace NUMINAMATH_CALUDE_race_permutations_l1054_105440

theorem race_permutations (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_permutations_l1054_105440


namespace NUMINAMATH_CALUDE_license_plate_difference_l1054_105481

/-- The number of letters in the alphabet --/
def num_letters : ℕ := 26

/-- The number of digits available --/
def num_digits : ℕ := 10

/-- The number of possible Florida license plates --/
def florida_plates : ℕ := num_letters^4 * num_digits^2

/-- The number of possible North Dakota license plates --/
def north_dakota_plates : ℕ := num_letters^3 * num_digits^3

/-- The difference in the number of possible license plates between Florida and North Dakota --/
def plate_difference : ℕ := florida_plates - north_dakota_plates

theorem license_plate_difference : plate_difference = 28121600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l1054_105481


namespace NUMINAMATH_CALUDE_equation_containing_2012_l1054_105450

theorem equation_containing_2012 (n : ℕ) : 
  (n^2 ≤ 2012 ∧ ∀ m : ℕ, m > n → m^2 > 2012) → 
  (n = 44 ∧ 2012 ∈ Finset.range (2*n^2 - n^2 + 1) \ Finset.range (n^2)) := by
  sorry

end NUMINAMATH_CALUDE_equation_containing_2012_l1054_105450


namespace NUMINAMATH_CALUDE_direct_proportion_l1054_105433

theorem direct_proportion (x y : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y, y = k * x) ↔ (∃ k : ℝ, k ≠ 0 ∧ y = k * x) :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_l1054_105433


namespace NUMINAMATH_CALUDE_B_subset_A_l1054_105495

-- Define set A
def A : Set ℝ := {x : ℝ | |2*x - 3| > 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 + x - 6 > 0}

-- Theorem to prove
theorem B_subset_A : B ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_B_subset_A_l1054_105495


namespace NUMINAMATH_CALUDE_right_triangle_with_constraint_l1054_105403

-- Define the triangle sides
def side1 (p q : ℝ) : ℝ := p
def side2 (p q : ℝ) : ℝ := p + q
def side3 (p q : ℝ) : ℝ := p + 2*q

-- Define the conditions
def is_right_triangle (p q : ℝ) : Prop :=
  (side3 p q)^2 = (side1 p q)^2 + (side2 p q)^2

def longest_side_constraint (p q : ℝ) : Prop :=
  side3 p q ≤ 12

-- Theorem statement
theorem right_triangle_with_constraint :
  ∃ (p q : ℝ),
    is_right_triangle p q ∧
    longest_side_constraint p q ∧
    p = (1 + Real.sqrt 7) / 2 ∧
    q = 1 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_constraint_l1054_105403


namespace NUMINAMATH_CALUDE_tank_fraction_problem_l1054_105447

/-- The problem of determining the fraction of the first tank's capacity that is filled. -/
theorem tank_fraction_problem (tank1_capacity tank2_capacity tank3_capacity : ℚ)
  (tank2_fill_fraction tank3_fill_fraction : ℚ)
  (total_water : ℚ) :
  tank1_capacity = 7000 →
  tank2_capacity = 5000 →
  tank3_capacity = 3000 →
  tank2_fill_fraction = 4/5 →
  tank3_fill_fraction = 1/2 →
  total_water = 10850 →
  total_water = tank1_capacity * (107/140) + tank2_capacity * tank2_fill_fraction + tank3_capacity * tank3_fill_fraction :=
by sorry

end NUMINAMATH_CALUDE_tank_fraction_problem_l1054_105447


namespace NUMINAMATH_CALUDE_utilities_percentage_l1054_105434

def budget_circle_graph (transportation research_development equipment supplies salaries utilities : ℝ) : Prop :=
  transportation = 20 ∧
  research_development = 9 ∧
  equipment = 4 ∧
  supplies = 2 ∧
  salaries = 60 ∧
  transportation + research_development + equipment + supplies + salaries + utilities = 100

theorem utilities_percentage 
  (transportation research_development equipment supplies salaries utilities : ℝ)
  (h : budget_circle_graph transportation research_development equipment supplies salaries utilities)
  (h_salaries : salaries * 360 / 100 = 216) : utilities = 5 := by
  sorry

end NUMINAMATH_CALUDE_utilities_percentage_l1054_105434


namespace NUMINAMATH_CALUDE_AtLeastOneSolution_l1054_105422

-- Define the property that the function f should satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f (y^2)) = x^2 + y

-- Theorem statement
theorem AtLeastOneSolution : ∃ f : ℝ → ℝ, SatisfiesProperty f := by
  sorry

end NUMINAMATH_CALUDE_AtLeastOneSolution_l1054_105422


namespace NUMINAMATH_CALUDE_polynomial_equation_l1054_105437

variable (x : ℝ)

def f (x : ℝ) : ℝ := x^4 - 3*x^2 + 1
def g (x : ℝ) : ℝ := -x^4 + 5*x^2 - 4

theorem polynomial_equation :
  (∀ x, f x + g x = 2*x^2 - 3) →
  (∀ x, g x = -x^4 + 5*x^2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equation_l1054_105437


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l1054_105435

/-- Given a triangle DEF with side lengths d, e, and f satisfying certain conditions,
    prove that its largest angle is 120°. -/
theorem largest_angle_in_triangle (d e f : ℝ) (h1 : d + 2*e + 2*f = d^2) (h2 : d + 2*e - 2*f = -9) :
  ∃ (D E F : ℝ), D + E + F = 180 ∧ max D (max E F) = 120 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l1054_105435


namespace NUMINAMATH_CALUDE_range_of_m_l1054_105405

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1/3)^(-x) - 2 else 2 * Real.log (-x) / Real.log 3

theorem range_of_m (m : ℝ) (h : f m > 1) :
  m ∈ Set.Ioi 1 ∪ Set.Iic (-Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1054_105405


namespace NUMINAMATH_CALUDE_unique_satisfying_polynomial_l1054_105410

/-- A polynomial satisfying the given conditions -/
def satisfying_polynomial (P : ℝ → ℝ) : Prop :=
  (P 2017 = 2016) ∧ 
  (∀ x : ℝ, (P x + 1)^2 = P (x^2 + 1))

/-- The theorem stating that the only polynomial satisfying the conditions is x - 1 -/
theorem unique_satisfying_polynomial : 
  ∀ P : ℝ → ℝ, satisfying_polynomial P → (∀ x : ℝ, P x = x - 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_satisfying_polynomial_l1054_105410


namespace NUMINAMATH_CALUDE_three_300deg_arcs_must_intersect_l1054_105464

/-- A great circle arc on a sphere -/
structure GreatCircleArc where
  /-- The angle of the arc in degrees -/
  angle : ℝ

/-- A configuration of three great circle arcs on a sphere -/
structure ThreeArcsConfiguration where
  arc1 : GreatCircleArc
  arc2 : GreatCircleArc
  arc3 : GreatCircleArc

/-- Predicate to check if two arcs intersect -/
def intersect (a b : GreatCircleArc) : Prop := sorry

/-- Theorem: It's impossible to place 3 great circle arcs of 300° each on a sphere with no common points -/
theorem three_300deg_arcs_must_intersect (config : ThreeArcsConfiguration) :
  config.arc1.angle = 300 ∧ config.arc2.angle = 300 ∧ config.arc3.angle = 300 →
  intersect config.arc1 config.arc2 ∨ intersect config.arc2 config.arc3 ∨ intersect config.arc3 config.arc1 :=
by sorry

end NUMINAMATH_CALUDE_three_300deg_arcs_must_intersect_l1054_105464


namespace NUMINAMATH_CALUDE_oil_purchase_amount_l1054_105444

/-- Represents the price and quantity of oil before and after a price reduction --/
structure OilPurchase where
  original_price : ℝ
  reduced_price : ℝ
  original_quantity : ℝ
  additional_quantity : ℝ
  price_reduction_percent : ℝ

/-- Calculates the total amount spent on oil after the price reduction --/
def total_spent (purchase : OilPurchase) : ℝ :=
  purchase.reduced_price * (purchase.original_quantity + purchase.additional_quantity)

/-- Theorem stating the total amount spent on oil after the price reduction --/
theorem oil_purchase_amount (purchase : OilPurchase) 
  (h1 : purchase.price_reduction_percent = 25)
  (h2 : purchase.additional_quantity = 5)
  (h3 : purchase.reduced_price = 60)
  (h4 : purchase.reduced_price = purchase.original_price * (1 - purchase.price_reduction_percent / 100)) :
  total_spent purchase = 1200 := by
  sorry

#eval total_spent { original_price := 80, reduced_price := 60, original_quantity := 15, additional_quantity := 5, price_reduction_percent := 25 }

end NUMINAMATH_CALUDE_oil_purchase_amount_l1054_105444


namespace NUMINAMATH_CALUDE_max_ratio_OB_OA_l1054_105404

-- Define the curves C₁ and C₂ in Cartesian coordinates
def C₁ (x y : ℝ) : Prop := x + y = 1

def C₂ (x y : ℝ) : Prop := ∃ φ : ℝ, 0 ≤ φ ∧ φ < 2 * Real.pi ∧ x = 2 + 2 * Real.cos φ ∧ y = 2 * Real.sin φ

-- Define the polar equations of C₁ and C₂
def C₁_polar (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + Real.sin θ) = 1

def C₂_polar (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

-- Define the ray l
def ray_l (ρ θ α : ℝ) : Prop := θ = α ∧ ρ ≥ 0

-- Define points A and B
def point_A (ρ α : ℝ) : Prop := C₁_polar ρ α ∧ ray_l ρ α α

def point_B (ρ α : ℝ) : Prop := C₂_polar ρ α ∧ ray_l ρ α α

-- Theorem statement
theorem max_ratio_OB_OA :
  ∃ max_ratio : ℝ, max_ratio = 2 + 2 * Real.sqrt 2 ∧
  (∀ α : ℝ, 0 ≤ α ∧ α ≤ Real.pi / 2 →
    ∀ ρA ρB : ℝ, point_A ρA α → point_B ρB α →
      ρB / ρA ≤ max_ratio) ∧
  (∃ α : ℝ, 0 ≤ α ∧ α ≤ Real.pi / 2 ∧
    ∃ ρA ρB : ℝ, point_A ρA α ∧ point_B ρB α ∧
      ρB / ρA = max_ratio) :=
sorry

end NUMINAMATH_CALUDE_max_ratio_OB_OA_l1054_105404


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1054_105459

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1054_105459


namespace NUMINAMATH_CALUDE_product_closure_infinite_pairs_l1054_105419

/-- The set M of integers of the form a^2 + 13b^2, where a and b are nonzero integers -/
def M : Set ℤ := {n : ℤ | ∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ n = a^2 + 13*b^2}

/-- The product of any two elements of M is an element of M -/
theorem product_closure (m1 m2 : ℤ) (h1 : m1 ∈ M) (h2 : m2 ∈ M) : m1 * m2 ∈ M := by
  sorry

/-- Definition of the sequence xk -/
def x (k : ℕ) : ℤ := (2^13 + 1) * ((4*k)^2 + 13*(4*k + 1)^2)

/-- Definition of the sequence yk -/
def y (k : ℕ) : ℤ := 2 * x k

/-- There are infinitely many pairs (x, y) such that x + y ∉ M but x^13 + y^13 ∈ M -/
theorem infinite_pairs : ∀ k : ℕ, (x k + y k ∉ M) ∧ ((x k)^13 + (y k)^13 ∈ M) := by
  sorry

end NUMINAMATH_CALUDE_product_closure_infinite_pairs_l1054_105419


namespace NUMINAMATH_CALUDE_triangle_sine_problem_l1054_105427

theorem triangle_sine_problem (a b c : ℝ) (A B C : ℝ) :
  a = 1 →
  b = Real.sqrt 2 →
  Real.sin A = 1/3 →
  (a / Real.sin A = b / Real.sin B) →
  Real.sin B = Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_problem_l1054_105427


namespace NUMINAMATH_CALUDE_total_length_of_items_l1054_105461

/-- Given a pencil, pen, and rubber with specific length relationships, 
    prove their total length is 29 cm. -/
theorem total_length_of_items (pencil_length pen_length rubber_length : ℕ) : 
  pencil_length = 12 →
  pen_length = pencil_length - 2 →
  rubber_length = pen_length - 3 →
  pencil_length + pen_length + rubber_length = 29 := by
  sorry

#check total_length_of_items

end NUMINAMATH_CALUDE_total_length_of_items_l1054_105461


namespace NUMINAMATH_CALUDE_ratio_of_45_to_9_l1054_105451

theorem ratio_of_45_to_9 (certain_number : ℕ) (h : certain_number = 45) : 
  certain_number / 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_45_to_9_l1054_105451


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l1054_105416

/-- A function f with two extremum points on ℝ -/
def has_two_extrema (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ y : ℝ, (f y ≤ f x₁ ∨ f y ≥ f x₁) ∧ (f y ≤ f x₂ ∨ f y ≥ f x₂))

/-- The main theorem -/
theorem cubic_function_extrema (a : ℝ) :
  has_two_extrema (λ x : ℝ => x^3 + a*x) → a < 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l1054_105416


namespace NUMINAMATH_CALUDE_range_of_a_and_m_l1054_105492

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

theorem range_of_a_and_m :
  ∀ (a m : ℝ), A ∪ B a = A → A ∩ C m = C m →
  (a = 2 ∨ a = 3) ∧ (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_and_m_l1054_105492


namespace NUMINAMATH_CALUDE_stadium_length_l1054_105499

theorem stadium_length (w h p : ℝ) (hw : w = 18) (hh : h = 16) (hp : p = 34) :
  ∃ l : ℝ, l = 24 ∧ p^2 = l^2 + w^2 + h^2 :=
by sorry

end NUMINAMATH_CALUDE_stadium_length_l1054_105499


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1054_105414

theorem partial_fraction_decomposition (x A B C : ℚ) :
  x ≠ 2 → x ≠ 4 → x ≠ 5 →
  ((x^2 - 9) / ((x - 2) * (x - 4) * (x - 5)) = A / (x - 2) + B / (x - 4) + C / (x - 5)) ↔
  (A = 5/3 ∧ B = -7/2 ∧ C = 8/3) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1054_105414


namespace NUMINAMATH_CALUDE_change_calculation_l1054_105432

def shirt_price : ℕ := 5
def sandal_price : ℕ := 3
def num_shirts : ℕ := 10
def num_sandals : ℕ := 3
def payment : ℕ := 100

def total_cost : ℕ := shirt_price * num_shirts + sandal_price * num_sandals

theorem change_calculation : payment - total_cost = 41 := by
  sorry

end NUMINAMATH_CALUDE_change_calculation_l1054_105432


namespace NUMINAMATH_CALUDE_knicks_knacks_knocks_conversion_l1054_105484

/-- Given the conversions between knicks, knacks, and knocks, 
    prove that 80 knocks is equal to 192 knicks. -/
theorem knicks_knacks_knocks_conversion : 
  ∀ (knicks knacks knocks : ℚ),
    (9 * knicks = 3 * knacks) →
    (4 * knacks = 5 * knocks) →
    (80 * knocks = 192 * knicks) :=
by
  sorry

end NUMINAMATH_CALUDE_knicks_knacks_knocks_conversion_l1054_105484


namespace NUMINAMATH_CALUDE_percentage_saved_approximately_11_percent_l1054_105489

def original_price : ℝ := 49.50
def spent_amount : ℝ := 44.00
def saved_amount : ℝ := original_price - spent_amount

theorem percentage_saved_approximately_11_percent :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  (saved_amount / original_price) * 100 ∈ Set.Icc (11 - ε) (11 + ε) := by
sorry

end NUMINAMATH_CALUDE_percentage_saved_approximately_11_percent_l1054_105489


namespace NUMINAMATH_CALUDE_inequality_solution_l1054_105418

theorem inequality_solution (a x : ℝ) :
  (a * x^2 - (a + 3) * x + 3 ≤ 0) ↔
    (a < 0 ∧ (x ≤ 3/a ∨ x ≥ 1)) ∨
    (a = 0 ∧ x ≥ 1) ∨
    (0 < a ∧ a < 3 ∧ 1 ≤ x ∧ x ≤ 3/a) ∨
    (a = 3 ∧ x = 1) ∨
    (a > 3 ∧ 3/a ≤ x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1054_105418


namespace NUMINAMATH_CALUDE_exists_interest_rate_unique_interest_rate_l1054_105443

/-- The interest rate that satisfies the given conditions --/
def interest_rate_equation (r : ℝ) : Prop :=
  1200 * ((1 + r/2)^2 - 1 - r) = 3

/-- Theorem stating that there exists an interest rate satisfying the equation --/
theorem exists_interest_rate : ∃ r : ℝ, interest_rate_equation r ∧ r > 0 ∧ r < 1 := by
  sorry

/-- Theorem stating that the interest rate solution is unique --/
theorem unique_interest_rate : ∃! r : ℝ, interest_rate_equation r ∧ r > 0 ∧ r < 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_interest_rate_unique_interest_rate_l1054_105443


namespace NUMINAMATH_CALUDE_five_equal_angles_l1054_105463

theorem five_equal_angles (rays : ℕ) (angle : ℝ) : 
  rays = 5 → 
  rays * angle = 360 → 
  angle = 72 := by
sorry

end NUMINAMATH_CALUDE_five_equal_angles_l1054_105463


namespace NUMINAMATH_CALUDE_complex_power_four_l1054_105465

theorem complex_power_four (i : ℂ) : i * i = -1 → (1 - i)^4 = -4 := by sorry

end NUMINAMATH_CALUDE_complex_power_four_l1054_105465


namespace NUMINAMATH_CALUDE_first_term_value_l1054_105467

/-- A geometric sequence with five terms -/
structure GeometricSequence :=
  (a b c d e : ℝ)
  (is_geometric : ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r)

/-- Theorem: In a geometric sequence where the fourth term is 81 and the fifth term is 243, the first term is 3 -/
theorem first_term_value (seq : GeometricSequence) 
  (h1 : seq.d = 81)
  (h2 : seq.e = 243) : 
  seq.a = 3 := by
sorry

end NUMINAMATH_CALUDE_first_term_value_l1054_105467


namespace NUMINAMATH_CALUDE_car_speed_first_hour_l1054_105470

/-- Given a car's speed over two hours, prove its speed in the first hour -/
theorem car_speed_first_hour (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 30 →
  average_speed = 65 →
  ∃ speed_first_hour : ℝ,
    speed_first_hour = 100 ∧
    average_speed = (speed_first_hour + speed_second_hour) / 2 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_first_hour_l1054_105470


namespace NUMINAMATH_CALUDE_trains_passing_time_l1054_105462

/-- The time taken for two trains to pass each other completely -/
theorem trains_passing_time (length : ℝ) (speed1 speed2 : ℝ) : 
  length = 150 →
  speed1 = 95 * (1000 / 3600) →
  speed2 = 85 * (1000 / 3600) →
  (2 * length) / (speed1 + speed2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_trains_passing_time_l1054_105462


namespace NUMINAMATH_CALUDE_unique_solution_cube_difference_square_l1054_105448

theorem unique_solution_cube_difference_square (x y z : ℕ+) : 
  (x.val : ℤ)^3 - (y.val : ℤ)^3 = (z.val : ℤ)^2 →
  Nat.Prime y.val →
  ¬(3 ∣ z.val) →
  ¬(y.val ∣ z.val) →
  x = 8 ∧ y = 7 ∧ z = 13 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_cube_difference_square_l1054_105448


namespace NUMINAMATH_CALUDE_prob_two_boys_one_girl_l1054_105479

/-- A hobby group with boys and girls -/
structure HobbyGroup where
  boys : Nat
  girls : Nat

/-- The probability of selecting exactly one boy and one girl -/
def prob_one_boy_one_girl (group : HobbyGroup) : Rat :=
  if group.boys ≥ 1 ∧ group.girls ≥ 1 then
    (group.boys * group.girls : Rat) / (group.boys + group.girls).choose 2
  else
    0

/-- Theorem: The probability of selecting exactly one boy and one girl
    from a group of 2 boys and 1 girl is 2/3 -/
theorem prob_two_boys_one_girl :
  prob_one_boy_one_girl ⟨2, 1⟩ = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_prob_two_boys_one_girl_l1054_105479


namespace NUMINAMATH_CALUDE_sin_beta_value_l1054_105482

theorem sin_beta_value (α β : Real)
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.cos (α - β) = -5 / 13)
  (h4 : Real.sin α = 4 / 5) :
  Real.sin β = -56 / 65 := by
sorry

end NUMINAMATH_CALUDE_sin_beta_value_l1054_105482
