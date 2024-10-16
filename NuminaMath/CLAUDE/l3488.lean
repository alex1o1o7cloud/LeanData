import Mathlib

namespace NUMINAMATH_CALUDE_water_pouring_theorem_l3488_348847

/-- Represents a state of water distribution among three containers -/
structure WaterState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a pouring action from one container to another -/
inductive PourAction
  | AtoB
  | AtoC
  | BtoA
  | BtoC
  | CtoA
  | CtoB

/-- Applies a pouring action to a water state -/
def applyPour (state : WaterState) (action : PourAction) : WaterState :=
  match action with
  | PourAction.AtoB => { a := state.a - state.b, b := state.b * 2, c := state.c }
  | PourAction.AtoC => { a := state.a - state.c, b := state.b, c := state.c * 2 }
  | PourAction.BtoA => { a := state.a * 2, b := state.b - state.a, c := state.c }
  | PourAction.BtoC => { a := state.a, b := state.b - state.c, c := state.c * 2 }
  | PourAction.CtoA => { a := state.a * 2, b := state.b, c := state.c - state.a }
  | PourAction.CtoB => { a := state.a, b := state.b * 2, c := state.c - state.b }

/-- Predicate to check if a container is empty -/
def isEmptyContainer (state : WaterState) : Prop :=
  state.a = 0 ∨ state.b = 0 ∨ state.c = 0

/-- The main theorem to be proved -/
theorem water_pouring_theorem (initialState : WaterState) :
  ∃ (actions : List PourAction), isEmptyContainer (actions.foldl applyPour initialState) :=
sorry


end NUMINAMATH_CALUDE_water_pouring_theorem_l3488_348847


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3488_348873

def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

theorem intersection_of_M_and_N : M ∩ N = Set.Ioo 1 2 ∪ {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3488_348873


namespace NUMINAMATH_CALUDE_rectangle_longest_side_l3488_348893

/-- A rectangle with perimeter 240 feet and area eight times its perimeter has its longest side equal to 101 feet. -/
theorem rectangle_longest_side : ∀ l w : ℝ,
  l > 0 → w > 0 →
  2 * (l + w) = 240 →
  l * w = 8 * 240 →
  max l w = 101 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longest_side_l3488_348893


namespace NUMINAMATH_CALUDE_two_true_propositions_l3488_348833

theorem two_true_propositions :
  let original := ∀ a : ℝ, a > -3 → a > 0
  let converse := ∀ a : ℝ, a > 0 → a > -3
  let inverse := ∀ a : ℝ, a ≤ -3 → a ≤ 0
  let contrapositive := ∀ a : ℝ, a ≤ 0 → a ≤ -3
  (¬original ∧ converse ∧ inverse ∧ ¬contrapositive) :=
by sorry

end NUMINAMATH_CALUDE_two_true_propositions_l3488_348833


namespace NUMINAMATH_CALUDE_pauls_crayons_l3488_348897

theorem pauls_crayons (erasers_birthday : ℕ) (crayons_left : ℕ) (eraser_crayon_diff : ℕ) 
  (h1 : erasers_birthday = 406)
  (h2 : crayons_left = 336)
  (h3 : eraser_crayon_diff = 70)
  (h4 : erasers_birthday = crayons_left + eraser_crayon_diff) :
  crayons_left + eraser_crayon_diff = 406 := by
  sorry

end NUMINAMATH_CALUDE_pauls_crayons_l3488_348897


namespace NUMINAMATH_CALUDE_min_operations_to_500_l3488_348876

/-- Represents the available operations on the calculator --/
inductive Operation
  | addOne
  | subOne
  | mulTwo

/-- Applies an operation to a number --/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.addOne => n + 1
  | Operation.subOne => if n > 0 then n - 1 else 0
  | Operation.mulTwo => n * 2

/-- Checks if a sequence of operations contains all three operation types --/
def containsAllOperations (ops : List Operation) : Prop :=
  Operation.addOne ∈ ops ∧ Operation.subOne ∈ ops ∧ Operation.mulTwo ∈ ops

/-- Applies a sequence of operations to a starting number --/
def applyOperations (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation start

/-- Theorem: The minimum number of operations to reach 500 from 1 is 13 --/
theorem min_operations_to_500 :
  (∃ (ops : List Operation),
    applyOperations 1 ops = 500 ∧
    containsAllOperations ops ∧
    ops.length = 13) ∧
  (∀ (ops : List Operation),
    applyOperations 1 ops = 500 →
    containsAllOperations ops →
    ops.length ≥ 13) :=
  sorry

end NUMINAMATH_CALUDE_min_operations_to_500_l3488_348876


namespace NUMINAMATH_CALUDE_statue_of_liberty_model_height_l3488_348865

/-- The scale ratio of the model to the actual size -/
def scale_ratio : ℚ := 1 / 30

/-- The actual height of the Statue of Liberty in feet -/
def actual_height : ℕ := 305

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem statue_of_liberty_model_height :
  round_to_nearest (actual_height / scale_ratio) = 10 := by
  sorry

end NUMINAMATH_CALUDE_statue_of_liberty_model_height_l3488_348865


namespace NUMINAMATH_CALUDE_profit_formula_l3488_348868

/-- Represents the cost and pricing structure of a shop selling bundles -/
structure ShopBundle where
  water_cost : ℚ  -- Cost of water bottle in dollars
  fruit_cost : ℚ  -- Cost of fruit in dollars
  snack_cost : ℚ  -- Cost of snack in dollars (unknown)
  regular_price : ℚ  -- Regular selling price of a bundle
  fifth_bundle_price : ℚ  -- Price of every 5th bundle
  water_per_bundle : ℕ  -- Number of water bottles per bundle
  fruit_per_bundle : ℕ  -- Number of fruits per bundle
  snack_per_bundle : ℕ  -- Number of snacks per regular bundle
  extra_snack : ℕ  -- Extra snacks given in 5th bundle

/-- Calculates the total profit for 5 bundles given the shop's pricing structure -/
def total_profit_five_bundles (shop : ShopBundle) : ℚ :=
  let regular_cost := shop.water_cost * shop.water_per_bundle +
                      shop.fruit_cost * shop.fruit_per_bundle +
                      shop.snack_cost * shop.snack_per_bundle
  let fifth_bundle_cost := shop.water_cost * shop.water_per_bundle +
                           shop.fruit_cost * shop.fruit_per_bundle +
                           shop.snack_cost * (shop.snack_per_bundle + shop.extra_snack)
  let regular_profit := shop.regular_price - regular_cost
  let fifth_bundle_profit := shop.fifth_bundle_price - fifth_bundle_cost
  4 * regular_profit + fifth_bundle_profit

/-- Theorem stating that the total profit for 5 bundles can be expressed as 15.40 - 16S -/
theorem profit_formula (shop : ShopBundle)
  (h1 : shop.water_cost = 0.5)
  (h2 : shop.fruit_cost = 0.25)
  (h3 : shop.regular_price = 4.6)
  (h4 : shop.fifth_bundle_price = 2)
  (h5 : shop.water_per_bundle = 1)
  (h6 : shop.fruit_per_bundle = 2)
  (h7 : shop.snack_per_bundle = 3)
  (h8 : shop.extra_snack = 1) :
  total_profit_five_bundles shop = 15.4 - 16 * shop.snack_cost := by
  sorry

end NUMINAMATH_CALUDE_profit_formula_l3488_348868


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3488_348815

theorem cubic_equation_solution (a w : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * w) : w = 49 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3488_348815


namespace NUMINAMATH_CALUDE_diagonals_in_30_sided_polygon_l3488_348807

theorem diagonals_in_30_sided_polygon : ∀ (n : ℕ), n = 30 → (n * (n - 3)) / 2 = 405 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_in_30_sided_polygon_l3488_348807


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3488_348866

/-- Three real numbers form an arithmetic sequence if the middle term is the arithmetic mean of the other two terms -/
def is_arithmetic_sequence (a b c : ℝ) : Prop := b = (a + c) / 2

theorem arithmetic_sequence_middle_term :
  ∀ m : ℝ, is_arithmetic_sequence 2 m 6 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3488_348866


namespace NUMINAMATH_CALUDE_parallel_planes_l3488_348814

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the necessary relations
variable (lies_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)

-- State the theorem
theorem parallel_planes
  (α β : Plane) (a b : Line) (A : Point) :
  lies_in a α →
  lies_in b α →
  intersect a b →
  ¬ line_parallel a β →
  ¬ line_parallel b β →
  parallel α β :=
by sorry

end NUMINAMATH_CALUDE_parallel_planes_l3488_348814


namespace NUMINAMATH_CALUDE_spade_calculation_l3488_348862

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_calculation : spade 5 (spade 6 7 + 2) = -96 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l3488_348862


namespace NUMINAMATH_CALUDE_tank_fill_problem_l3488_348844

theorem tank_fill_problem (tank_capacity : ℚ) (added_amount : ℚ) (final_fraction : ℚ) :
  tank_capacity = 54 →
  added_amount = 9 →
  final_fraction = 9/10 →
  (tank_capacity * final_fraction - added_amount) / tank_capacity = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_problem_l3488_348844


namespace NUMINAMATH_CALUDE_benny_lunch_payment_l3488_348838

/-- The cost of a lunch special -/
def lunch_special_cost : ℕ := 8

/-- The number of people having lunch -/
def number_of_people : ℕ := 3

/-- The total cost Benny will pay for lunch -/
def total_cost : ℕ := number_of_people * lunch_special_cost

theorem benny_lunch_payment : total_cost = 24 := by
  sorry

end NUMINAMATH_CALUDE_benny_lunch_payment_l3488_348838


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3488_348872

theorem trigonometric_identity (α : Real) : 
  0 < α ∧ α < π/2 →
  Real.sin (5*π/12 + 2*α) = -3/5 →
  Real.sin (π/12 + α) * Real.sin (5*π/12 - α) = Real.sqrt 2 / 20 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3488_348872


namespace NUMINAMATH_CALUDE_penny_shark_species_l3488_348831

/-- Given the number of species Penny identified in an aquarium, prove the number of shark species. -/
theorem penny_shark_species (total : ℕ) (eels : ℕ) (whales : ℕ) (sharks : ℕ)
  (h1 : total = 55)
  (h2 : eels = 15)
  (h3 : whales = 5)
  (h4 : total = sharks + eels + whales) :
  sharks = 35 := by
  sorry

end NUMINAMATH_CALUDE_penny_shark_species_l3488_348831


namespace NUMINAMATH_CALUDE_four_numbers_with_one_sixth_property_l3488_348843

/-- A four-digit number -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

/-- The three-digit number obtained by removing the leftmost digit of a four-digit number -/
def RemoveLeftmostDigit (n : ℕ) : ℕ := n % 1000

/-- The property that the three-digit number obtained by removing the leftmost digit is one sixth of the original number -/
def HasOneSixthProperty (n : ℕ) : Prop :=
  FourDigitNumber n ∧ RemoveLeftmostDigit n = n / 6

/-- The theorem stating that there are exactly 4 numbers satisfying the property -/
theorem four_numbers_with_one_sixth_property :
  ∃! (s : Finset ℕ), s.card = 4 ∧ ∀ n, n ∈ s ↔ HasOneSixthProperty n :=
sorry

end NUMINAMATH_CALUDE_four_numbers_with_one_sixth_property_l3488_348843


namespace NUMINAMATH_CALUDE_karina_to_brother_age_ratio_l3488_348870

-- Define the given information
def karina_birth_year : ℕ := 1970
def karina_current_age : ℕ := 40
def brother_birth_year : ℕ := 1990

-- Define the current year based on Karina's age
def current_year : ℕ := karina_birth_year + karina_current_age

-- Calculate brother's age
def brother_current_age : ℕ := current_year - brother_birth_year

-- Theorem to prove
theorem karina_to_brother_age_ratio :
  (karina_current_age : ℚ) / (brother_current_age : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_karina_to_brother_age_ratio_l3488_348870


namespace NUMINAMATH_CALUDE_max_side_length_of_special_triangle_l3488_348880

theorem max_side_length_of_special_triangle :
  ∀ a b c : ℕ,
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a + b + c = 30 →
  a < b + c ∧ b < a + c ∧ c < a + b →
  ∀ x : ℕ, x ≤ a ∧ x ≤ b ∧ x ≤ c →
  x ≤ 14 :=
by sorry

end NUMINAMATH_CALUDE_max_side_length_of_special_triangle_l3488_348880


namespace NUMINAMATH_CALUDE_chairs_subset_count_l3488_348894

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- The minimum number of adjacent chairs required in a subset -/
def k : ℕ := 4

/-- The number of subsets of n chairs arranged in a circle that contain at least k adjacent chairs -/
def subsets_with_adjacent_chairs (n k : ℕ) : ℕ := sorry

theorem chairs_subset_count : subsets_with_adjacent_chairs n k = 1610 := by sorry

end NUMINAMATH_CALUDE_chairs_subset_count_l3488_348894


namespace NUMINAMATH_CALUDE_planter_pots_cost_l3488_348890

/-- Calculates the total cost of filling planter pots with plants, including sales tax. -/
def total_cost (num_pots : ℕ) (palm_fern_cost creeping_jenny_cost geranium_cost elephant_ear_cost purple_grass_cost : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let plant_cost_per_pot := palm_fern_cost + 4 * creeping_jenny_cost + 4 * geranium_cost + 2 * elephant_ear_cost + 3 * purple_grass_cost
  let total_plant_cost := num_pots * plant_cost_per_pot
  let sales_tax := sales_tax_rate * total_plant_cost
  total_plant_cost + sales_tax

/-- Theorem stating that the total cost to fill 6 planter pots with the given plants and 7% sales tax is $494.34. -/
theorem planter_pots_cost : total_cost 6 15 4 3.5 7 6 (7/100) = 494.34 := by
  sorry

end NUMINAMATH_CALUDE_planter_pots_cost_l3488_348890


namespace NUMINAMATH_CALUDE_exists_probability_outside_range_l3488_348889

/-- Represents a packet of candies -/
structure Packet :=
  (total : ℕ)
  (blue : ℕ)
  (h : blue ≤ total)

/-- Represents a box containing two packets of candies -/
structure Box :=
  (packet1 : Packet)
  (packet2 : Packet)

/-- Calculates the probability of drawing a blue candy from the box -/
def blueProbability (box : Box) : ℚ :=
  (box.packet1.blue + box.packet2.blue : ℚ) / (box.packet1.total + box.packet2.total)

/-- Theorem stating that there exists a box configuration where the probability
    of drawing a blue candy is not between 3/8 and 2/5 -/
theorem exists_probability_outside_range :
  ∃ (box : Box), ¬(3/8 < blueProbability box ∧ blueProbability box < 2/5) :=
sorry

end NUMINAMATH_CALUDE_exists_probability_outside_range_l3488_348889


namespace NUMINAMATH_CALUDE_factorization_equality_l3488_348820

theorem factorization_equality (x : ℝ) : 4 * x - x^2 - 4 = -(x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3488_348820


namespace NUMINAMATH_CALUDE_absolute_value_equation_roots_l3488_348888

theorem absolute_value_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ |x| = a * x - a) ∧ 
  (∀ x : ℝ, x < 0 → |x| ≠ a * x - a) → 
  a > 1 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_roots_l3488_348888


namespace NUMINAMATH_CALUDE_box_bottle_count_l3488_348896

def dozen : ℕ := 12

def water_bottles : ℕ := 2 * dozen

def apple_bottles : ℕ := water_bottles + (dozen / 2)

def total_bottles : ℕ := water_bottles + apple_bottles

theorem box_bottle_count : total_bottles = 54 := by
  sorry

end NUMINAMATH_CALUDE_box_bottle_count_l3488_348896


namespace NUMINAMATH_CALUDE_birds_joined_fence_l3488_348856

/-- Given initial numbers of storks and birds on a fence, and the fact that after some birds
    joined there are 2 more birds than storks, prove that 4 birds joined the fence. -/
theorem birds_joined_fence (initial_storks initial_birds : ℕ) 
  (h1 : initial_storks = 5)
  (h2 : initial_birds = 3)
  (h3 : ∃ (joined : ℕ), initial_birds + joined = initial_storks + 2) :
  ∃ (joined : ℕ), joined = 4 ∧ initial_birds + joined = initial_storks + 2 :=
by sorry

end NUMINAMATH_CALUDE_birds_joined_fence_l3488_348856


namespace NUMINAMATH_CALUDE_euro_op_calculation_l3488_348839

-- Define the € operation
def euro_op (x y z : ℕ) : ℕ := 3 * x * y * z

-- State the theorem
theorem euro_op_calculation : 
  euro_op 3 (euro_op 4 5 6) 1 = 3240 := by
  sorry

end NUMINAMATH_CALUDE_euro_op_calculation_l3488_348839


namespace NUMINAMATH_CALUDE_min_a_for_polynomial_with_two_zeros_in_unit_interval_l3488_348817

theorem min_a_for_polynomial_with_two_zeros_in_unit_interval : 
  ∃ (a b c : ℤ), 
    (∀ (a' : ℤ), a' > 0 ∧ a' < a →
      ¬∃ (b' c' : ℤ), ∃ (x y : ℝ), 
        0 < x ∧ x < y ∧ y < 1 ∧
        a' * x^2 - b' * x + c' = 0 ∧
        a' * y^2 - b' * y + c' = 0) ∧
    (∃ (x y : ℝ), 
      0 < x ∧ x < y ∧ y < 1 ∧
      a * x^2 - b * x + c = 0 ∧
      a * y^2 - b * y + c = 0) ∧
    a = 5 := by sorry

end NUMINAMATH_CALUDE_min_a_for_polynomial_with_two_zeros_in_unit_interval_l3488_348817


namespace NUMINAMATH_CALUDE_min_wins_for_playoffs_l3488_348886

/-- 
Given a basketball league with the following conditions:
- Each game must have a winner
- A team earns 3 points for a win and loses 1 point for a loss
- The season consists of 32 games
- A team needs at least 48 points to have a chance at the playoffs

This theorem proves that a team must win at least 20 games to have a chance of advancing to the playoffs.
-/
theorem min_wins_for_playoffs (total_games : ℕ) (win_points loss_points : ℤ) (min_points : ℕ) : 
  total_games = 32 → win_points = 3 → loss_points = -1 → min_points = 48 → 
  ∃ (min_wins : ℕ), min_wins = 20 ∧ 
    ∀ (wins : ℕ), wins ≥ min_wins → 
      wins * win_points + (total_games - wins) * loss_points ≥ min_points :=
by sorry

end NUMINAMATH_CALUDE_min_wins_for_playoffs_l3488_348886


namespace NUMINAMATH_CALUDE_sine_function_max_min_l3488_348803

theorem sine_function_max_min (a b : ℝ) (h1 : a < 0) :
  (∀ x, b + a * Real.sin x ≤ -1) ∧
  (∃ x, b + a * Real.sin x = -1) ∧
  (∀ x, b + a * Real.sin x ≥ -5) ∧
  (∃ x, b + a * Real.sin x = -5) →
  a = -2 ∧ b = -3 := by
sorry

end NUMINAMATH_CALUDE_sine_function_max_min_l3488_348803


namespace NUMINAMATH_CALUDE_tan_315_degrees_l3488_348858

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l3488_348858


namespace NUMINAMATH_CALUDE_estate_distribution_theorem_l3488_348860

/-- Represents the estate distribution problem --/
structure EstateDistribution where
  total : ℚ
  daughter_share : ℚ
  son_share : ℚ
  wife_share : ℚ
  nephew_share : ℚ
  gardener_share : ℚ

/-- Theorem stating the conditions and the result to be proved --/
theorem estate_distribution_theorem (e : EstateDistribution) : 
  e.daughter_share + e.son_share = (2 : ℚ) / (3 : ℚ) * e.total ∧ 
  e.daughter_share = (5 : ℚ) / (9 : ℚ) * ((2 : ℚ) / (3 : ℚ) * e.total) ∧
  e.son_share = (4 : ℚ) / (9 : ℚ) * ((2 : ℚ) / (3 : ℚ) * e.total) ∧
  e.wife_share = 3 * e.son_share ∧
  e.nephew_share = 1000 ∧
  e.gardener_share = 600 ∧
  e.total = e.daughter_share + e.son_share + e.wife_share + e.nephew_share + e.gardener_share
  →
  e.total = 2880 := by
  sorry


end NUMINAMATH_CALUDE_estate_distribution_theorem_l3488_348860


namespace NUMINAMATH_CALUDE_triangle_problem_l3488_348895

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  (a * Real.sin C) / (1 - Real.cos A) = Real.sqrt 3 * c →
  b + c = 10 →
  (1 / 2) * b * c * Real.sin A = 4 * Real.sqrt 3 →
  (A = π / 3) ∧ (a = 2 * Real.sqrt 13) := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l3488_348895


namespace NUMINAMATH_CALUDE_original_expenditure_problem_l3488_348810

/-- The original expenditure problem -/
theorem original_expenditure_problem :
  ∀ (E A : ℕ),
  -- Initial conditions
  E = 35 * A →
  -- After first admission
  E + 84 = 42 * (A - 1) →
  -- After second change
  E + 124 = 37 * (A + 1) →
  -- Conclusion
  E = 630 := by
  sorry

end NUMINAMATH_CALUDE_original_expenditure_problem_l3488_348810


namespace NUMINAMATH_CALUDE_heptagon_foldable_to_quadrilateral_l3488_348819

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A polygon represented by its vertices -/
structure Polygon where
  vertices : List Point2D

/-- A function to check if a polygon is convex -/
def isConvex (p : Polygon) : Prop := sorry

/-- A function to check if a polygon can be folded into a two-layered quadrilateral -/
def canFoldToTwoLayeredQuadrilateral (p : Polygon) : Prop := sorry

/-- Theorem: There exists a convex heptagon that can be folded into a two-layered quadrilateral -/
theorem heptagon_foldable_to_quadrilateral :
  ∃ (h : Polygon), h.vertices.length = 7 ∧ isConvex h ∧ canFoldToTwoLayeredQuadrilateral h := by
  sorry

end NUMINAMATH_CALUDE_heptagon_foldable_to_quadrilateral_l3488_348819


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l3488_348859

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.25 * P) 
  (hN : N = 0.6 * P) : 
  M / N = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l3488_348859


namespace NUMINAMATH_CALUDE_carrot_calories_l3488_348891

/-- The number of calories in a pound of carrots -/
def calories_per_pound_carrots : ℕ := 51

/-- The number of pounds of carrots Tom eats -/
def pounds_carrots : ℕ := 1

/-- The number of pounds of broccoli Tom eats -/
def pounds_broccoli : ℕ := 2

/-- The ratio of calories in broccoli compared to carrots -/
def broccoli_calorie_ratio : ℚ := 1/3

/-- The total number of calories Tom ate -/
def total_calories : ℕ := 85

theorem carrot_calories :
  calories_per_pound_carrots * pounds_carrots +
  (calories_per_pound_carrots : ℚ) * broccoli_calorie_ratio * pounds_broccoli = total_calories := by
  sorry

end NUMINAMATH_CALUDE_carrot_calories_l3488_348891


namespace NUMINAMATH_CALUDE_cost_increase_is_six_percent_l3488_348853

/-- Represents the cost components of manufacturing a car -/
structure CarCost where
  rawMaterial : ℝ
  labor : ℝ
  overheads : ℝ

/-- Calculates the total cost of manufacturing a car -/
def totalCost (cost : CarCost) : ℝ :=
  cost.rawMaterial + cost.labor + cost.overheads

/-- Represents the cost ratio in the first year -/
def initialRatio : CarCost :=
  { rawMaterial := 4
    labor := 3
    overheads := 2 }

/-- Calculates the new cost after applying percentage changes -/
def newCost (cost : CarCost) : CarCost :=
  { rawMaterial := cost.rawMaterial * 1.1
    labor := cost.labor * 1.08
    overheads := cost.overheads * 0.95 }

/-- Theorem stating that the total cost increase is 6% -/
theorem cost_increase_is_six_percent :
  (totalCost (newCost initialRatio) - totalCost initialRatio) / totalCost initialRatio * 100 = 6 := by
  sorry


end NUMINAMATH_CALUDE_cost_increase_is_six_percent_l3488_348853


namespace NUMINAMATH_CALUDE_quadratic_points_relationship_l3488_348883

/-- A quadratic function f(x) = -x² + 2x + 3 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

/-- Point P₁ on the graph of f -/
def P₁ : ℝ × ℝ := (-1, f (-1))

/-- Point P₂ on the graph of f -/
def P₂ : ℝ × ℝ := (3, f 3)

/-- Point P₃ on the graph of f -/
def P₃ : ℝ × ℝ := (5, f 5)

theorem quadratic_points_relationship :
  P₁.2 = P₂.2 ∧ P₂.2 > P₃.2 := by sorry

end NUMINAMATH_CALUDE_quadratic_points_relationship_l3488_348883


namespace NUMINAMATH_CALUDE_star_problem_l3488_348818

-- Define the star operation
def star (a b : ℕ) : ℕ := 3 + b^(a + 1)

-- State the theorem
theorem star_problem : star (star 2 3) 2 = 3 + 2^31 := by
  sorry

end NUMINAMATH_CALUDE_star_problem_l3488_348818


namespace NUMINAMATH_CALUDE_f_value_at_100_l3488_348829

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem f_value_at_100 (f : ℝ → ℝ) 
  (h1 : ∀ x, f x * f (x + 3) = 12)
  (h2 : f 1 = 4) :
  f 100 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_100_l3488_348829


namespace NUMINAMATH_CALUDE_set_inclusion_implies_m_range_l3488_348863

theorem set_inclusion_implies_m_range (m : ℝ) :
  let P : Set ℝ := {x | -2 ≤ x ∧ x ≤ 10}
  let S : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}
  (S.Nonempty) → (S ⊆ P) → (0 ≤ m ∧ m ≤ 3) :=
by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_m_range_l3488_348863


namespace NUMINAMATH_CALUDE_greatest_possible_average_speed_l3488_348805

/-- Represents a palindromic number -/
def IsPalindrome (n : ℕ) : Prop := sorry

/-- The initial odometer reading -/
def initial_reading : ℕ := 12321

/-- The duration of the drive in hours -/
def drive_duration : ℝ := 4

/-- The speed limit in miles per hour -/
def speed_limit : ℝ := 65

/-- The greatest possible average speed in miles per hour -/
def max_average_speed : ℝ := 50

/-- The theorem stating the greatest possible average speed -/
theorem greatest_possible_average_speed :
  ∀ (final_reading : ℕ),
    IsPalindrome initial_reading →
    IsPalindrome final_reading →
    final_reading > initial_reading →
    (final_reading - initial_reading : ℝ) ≤ speed_limit * drive_duration →
    (∀ (speed : ℝ), speed ≤ speed_limit → 
      (final_reading - initial_reading : ℝ) / drive_duration ≤ speed) →
    (final_reading - initial_reading : ℝ) / drive_duration = max_average_speed :=
sorry

end NUMINAMATH_CALUDE_greatest_possible_average_speed_l3488_348805


namespace NUMINAMATH_CALUDE_liar_proportion_is_half_l3488_348842

/-- Represents the proportion of liars in a village -/
def proportion_of_liars : ℝ := sorry

/-- The proportion of liars is between 0 and 1 -/
axiom proportion_bounds : 0 ≤ proportion_of_liars ∧ proportion_of_liars ≤ 1

/-- The proportion of liars is indistinguishable from the proportion of truth-tellers when roles are reversed -/
axiom indistinguishable_proportion : proportion_of_liars = 1 - proportion_of_liars

theorem liar_proportion_is_half : proportion_of_liars = 1/2 := by sorry

end NUMINAMATH_CALUDE_liar_proportion_is_half_l3488_348842


namespace NUMINAMATH_CALUDE_correct_mark_l3488_348828

theorem correct_mark (wrong_mark : ℝ) (class_size : ℕ) (average_increase : ℝ) : 
  wrong_mark = 85 → 
  class_size = 80 → 
  average_increase = 0.5 →
  (wrong_mark - (wrong_mark - class_size * average_increase)) = 45 := by
  sorry

end NUMINAMATH_CALUDE_correct_mark_l3488_348828


namespace NUMINAMATH_CALUDE_angle_Q_is_90_degrees_l3488_348885

/-- A regular dodecagon with vertices ABCDEFGHIJKL -/
structure RegularDodecagon where
  vertices : Fin 12 → Point

/-- The point Q where extended sides AL and FG meet -/
def Q (d : RegularDodecagon) : Point := sorry

/-- The angle at point Q formed by the extended sides AL and FG -/
def angle_Q (d : RegularDodecagon) : AngularMeasure := sorry

/-- The theorem stating that the measure of angle Q is 90 degrees -/
theorem angle_Q_is_90_degrees (d : RegularDodecagon) : 
  angle_Q d = 90 := by sorry

end NUMINAMATH_CALUDE_angle_Q_is_90_degrees_l3488_348885


namespace NUMINAMATH_CALUDE_expression_simplification_l3488_348804

theorem expression_simplification (a b : ℝ) 
  (h1 : b ≠ 0) (h2 : b ≠ -3*a) (h3 : b ≠ a) (h4 : b ≠ -a) (h5 : a ≠ 0) :
  (2*b + a - (4*a^2 - b^2)/a) / (b^3 + 2*a*b^2 - 3*a^2*b) * 
  (a^3*b - 2*a^2*b^2 + a*b^3) / (a^2 - b^2) = (a - b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3488_348804


namespace NUMINAMATH_CALUDE_squares_in_figure_150_l3488_348855

/-- The number of squares in figure n -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The sequence of squares for the first four figures -/
def initial_sequence : List ℕ := [1, 7, 19, 37]

theorem squares_in_figure_150 :
  f 150 = 67951 ∧
  (∀ n : Fin 4, f n.val = initial_sequence.get n) :=
sorry

end NUMINAMATH_CALUDE_squares_in_figure_150_l3488_348855


namespace NUMINAMATH_CALUDE_binary_253_ones_minus_zeros_l3488_348811

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Count the number of true values in a list of booleans -/
def countOnes (l : List Bool) : ℕ :=
  l.filter id |>.length

/-- Count the number of false values in a list of booleans -/
def countZeros (l : List Bool) : ℕ :=
  l.filter not |>.length

theorem binary_253_ones_minus_zeros :
  let binary := toBinary 253
  let y := countOnes binary
  let x := countZeros binary
  y - x = 6 := by sorry

end NUMINAMATH_CALUDE_binary_253_ones_minus_zeros_l3488_348811


namespace NUMINAMATH_CALUDE_continuity_at_negative_one_l3488_348808

noncomputable def f (x : ℝ) : ℝ := (x^3 - x^2 + x + 1) / (x^2 - 1)

theorem continuity_at_negative_one :
  Filter.Tendsto f (nhds (-1)) (nhds (-2)) :=
sorry

end NUMINAMATH_CALUDE_continuity_at_negative_one_l3488_348808


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3488_348879

theorem complex_number_in_first_quadrant :
  let z : ℂ := 1 / (1 - Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3488_348879


namespace NUMINAMATH_CALUDE_tan_alpha_2_implies_expression_zero_l3488_348892

theorem tan_alpha_2_implies_expression_zero (α : Real) (h : Real.tan α = 2) :
  2 * (Real.sin α)^2 - 3 * (Real.sin α) * (Real.cos α) - 2 * (Real.cos α)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implies_expression_zero_l3488_348892


namespace NUMINAMATH_CALUDE_balloon_distribution_l3488_348834

theorem balloon_distribution (yellow_balloons : ℕ) (black_balloons_diff : ℕ) (balloons_per_school : ℕ) : 
  yellow_balloons = 3414 →
  black_balloons_diff = 1762 →
  balloons_per_school = 859 →
  (yellow_balloons + (yellow_balloons + black_balloons_diff)) / balloons_per_school = 10 := by
  sorry

end NUMINAMATH_CALUDE_balloon_distribution_l3488_348834


namespace NUMINAMATH_CALUDE_triangle_pieces_count_l3488_348867

/-- Sum of an arithmetic sequence -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Number of rods in the triangle -/
def totalRods : ℕ := arithmeticSum 5 5 10

/-- Number of connectors in the triangle -/
def totalConnectors : ℕ := arithmeticSum 3 3 11

/-- Total number of pieces in the triangle -/
def totalPieces : ℕ := totalRods + totalConnectors

theorem triangle_pieces_count : totalPieces = 473 := by
  sorry

end NUMINAMATH_CALUDE_triangle_pieces_count_l3488_348867


namespace NUMINAMATH_CALUDE_johns_hats_cost_l3488_348830

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks John can wear a different hat each day -/
def weeks_of_different_hats : ℕ := 2

/-- The cost of each hat in dollars -/
def cost_per_hat : ℕ := 50

/-- The total cost of John's hats -/
def total_cost : ℕ := weeks_of_different_hats * days_per_week * cost_per_hat

theorem johns_hats_cost :
  total_cost = 700 :=
by sorry

end NUMINAMATH_CALUDE_johns_hats_cost_l3488_348830


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l3488_348832

def U : Set ℤ := {1, 2, 3, 4, 5, 6, 7}

def M : Set ℤ := {x | x^2 - 6*x + 5 ≤ 0 ∧ x ∈ U}

theorem complement_of_M_in_U :
  U \ M = {6, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l3488_348832


namespace NUMINAMATH_CALUDE_square_side_length_l3488_348861

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 1 / 9) (h2 : side * side = area) : 
  side = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l3488_348861


namespace NUMINAMATH_CALUDE_minjoo_walked_distance_l3488_348851

-- Define the distances walked by Yongchan and Min-joo
def yongchan_distance : ℝ := 1.05
def difference : ℝ := 0.46

-- Define Min-joo's distance as a function of Yongchan's distance and the difference
def minjoo_distance : ℝ := yongchan_distance - difference

-- Theorem statement
theorem minjoo_walked_distance : minjoo_distance = 0.59 := by
  sorry

end NUMINAMATH_CALUDE_minjoo_walked_distance_l3488_348851


namespace NUMINAMATH_CALUDE_ellipse_sum_bound_l3488_348881

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Theorem statement
theorem ellipse_sum_bound :
  ∀ x y : ℝ, ellipse x y → -4 ≤ x + 2*y ∧ x + 2*y ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_sum_bound_l3488_348881


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3488_348874

/-- A quadratic function y = px^2 + qx + r with specific properties -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  r : ℝ
  is_parabola : True
  vertex_x : p ≠ 0 → -q / (2 * p) = -3
  vertex_y : p ≠ 0 → p * (-3)^2 + q * (-3) + r = 4
  passes_through_origin : p * 0^2 + q * 0 + r = -2
  vertical_symmetry : True

/-- The sum of coefficients p, q, and r equals -20/3 -/
theorem sum_of_coefficients (f : QuadraticFunction) : f.p + f.q + f.r = -20/3 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_coefficients_l3488_348874


namespace NUMINAMATH_CALUDE_probability_point_near_vertex_l3488_348836

/-- The probability of a randomly selected point from a square being within a certain distance from a vertex -/
theorem probability_point_near_vertex (side_length : ℝ) (distance : ℝ) : 
  side_length > 0 → distance > 0 → distance ≤ side_length →
  (π * distance^2) / (4 * side_length^2) = π / 16 ↔ side_length = 4 ∧ distance = 2 :=
by sorry

end NUMINAMATH_CALUDE_probability_point_near_vertex_l3488_348836


namespace NUMINAMATH_CALUDE_second_group_count_l3488_348809

theorem second_group_count (total : ℕ) (avg : ℚ) (sum_three : ℕ) (avg_others : ℚ) :
  total = 5 ∧ 
  avg = 20 ∧ 
  sum_three = 48 ∧ 
  avg_others = 26 →
  (total - 3 : ℕ) = 2 :=
by sorry

end NUMINAMATH_CALUDE_second_group_count_l3488_348809


namespace NUMINAMATH_CALUDE_largest_initial_number_l3488_348801

theorem largest_initial_number : ∃ (a b c d e : ℕ), 
  189 + a + b + c + d + e = 200 ∧ 
  a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ d ≥ 2 ∧ e ≥ 2 ∧
  189 % a ≠ 0 ∧ 189 % b ≠ 0 ∧ 189 % c ≠ 0 ∧ 189 % d ≠ 0 ∧ 189 % e ≠ 0 ∧
  ∀ n > 189, ¬(∃ (x y z w v : ℕ), 
    n + x + y + z + w + v = 200 ∧ 
    x ≥ 2 ∧ y ≥ 2 ∧ z ≥ 2 ∧ w ≥ 2 ∧ v ≥ 2 ∧
    n % x ≠ 0 ∧ n % y ≠ 0 ∧ n % z ≠ 0 ∧ n % w ≠ 0 ∧ n % v ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_initial_number_l3488_348801


namespace NUMINAMATH_CALUDE_briana_investment_proof_l3488_348864

def emma_investment : ℝ := 300
def emma_yield_rate : ℝ := 0.15
def briana_yield_rate : ℝ := 0.10
def years : ℕ := 2
def return_difference : ℝ := 10

def briana_investment : ℝ := 400

theorem briana_investment_proof :
  (years : ℝ) * emma_yield_rate * emma_investment - 
  (years : ℝ) * briana_yield_rate * briana_investment = return_difference :=
by sorry

end NUMINAMATH_CALUDE_briana_investment_proof_l3488_348864


namespace NUMINAMATH_CALUDE_line_and_chord_problem_l3488_348852

-- Define the circle M
def circle_M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

-- Define the line l
def line_l : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}

-- Define the midpoint P
def point_P : ℝ × ℝ := (1, 1)

-- Define the intersection points A and B
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry

theorem line_and_chord_problem :
  point_P = ((point_A.1 + point_B.1) / 2, (point_A.2 + point_B.2) / 2) ∧
  point_A ∈ circle_M ∧ point_B ∈ circle_M ∧
  point_A ∈ line_l ∧ point_B ∈ line_l →
  (∀ p : ℝ × ℝ, p ∈ line_l ↔ p.1 + p.2 = 2) ∧
  Real.sqrt ((point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2) = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_line_and_chord_problem_l3488_348852


namespace NUMINAMATH_CALUDE_min_value_on_interval_l3488_348849

def f (x : ℝ) := -x^2 + 4*x - 2

theorem min_value_on_interval :
  ∀ x ∈ Set.Icc 1 4, f x ≥ -2 ∧ ∃ y ∈ Set.Icc 1 4, f y = -2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l3488_348849


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_geq_neg_eight_l3488_348877

theorem intersection_nonempty_implies_a_geq_neg_eight (a : ℝ) : 
  (∃ x : ℝ, x ∈ {x | 1 ≤ x ∧ x ≤ 2} ∩ {x | x^2 + 2*x + a ≥ 0}) → a ≥ -8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_geq_neg_eight_l3488_348877


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3488_348813

theorem cube_root_equation_solution (x : ℝ) :
  (x * (x^2)^(1/2))^(1/3) = 2 → x = 2 * (2^(1/2)) ∨ x = -2 * (2^(1/2)) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3488_348813


namespace NUMINAMATH_CALUDE_complex_equation_implies_sum_l3488_348826

theorem complex_equation_implies_sum (a t : ℝ) :
  (a + Complex.I) / (1 + 2 * Complex.I) = t * Complex.I →
  t + a = -1 := by sorry

end NUMINAMATH_CALUDE_complex_equation_implies_sum_l3488_348826


namespace NUMINAMATH_CALUDE_correct_metal_ratio_l3488_348854

/-- Represents the ratio of two metals in an alloy -/
structure MetalRatio where
  a : ℚ
  b : ℚ

/-- Calculates the cost of an alloy given the ratio of metals and their individual costs -/
def alloyCost (ratio : MetalRatio) (costA costB : ℚ) : ℚ :=
  (ratio.a * costA + ratio.b * costB) / (ratio.a + ratio.b)

/-- Theorem stating the correct ratio of metals to achieve the desired alloy cost -/
theorem correct_metal_ratio :
  let desiredRatio : MetalRatio := ⟨3, 1⟩
  let costA : ℚ := 68
  let costB : ℚ := 96
  let desiredCost : ℚ := 75
  alloyCost desiredRatio costA costB = desiredCost := by sorry

end NUMINAMATH_CALUDE_correct_metal_ratio_l3488_348854


namespace NUMINAMATH_CALUDE_color_drawing_cost_theorem_l3488_348824

/-- The cost of a color drawing given the cost of a black and white drawing and the additional percentage for color. -/
def color_drawing_cost (bw_cost : ℝ) (color_percentage : ℝ) : ℝ :=
  bw_cost * (1 + color_percentage)

/-- Theorem: The cost of a color drawing is $240 when a black and white drawing costs $160 and color is 50% more expensive. -/
theorem color_drawing_cost_theorem :
  color_drawing_cost 160 0.5 = 240 := by
  sorry

end NUMINAMATH_CALUDE_color_drawing_cost_theorem_l3488_348824


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l3488_348841

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l3488_348841


namespace NUMINAMATH_CALUDE_draw_specific_nondefective_draw_at_least_one_defective_l3488_348871

/-- Represents the total number of products -/
def total_products : ℕ := 10

/-- Represents the number of defective products -/
def defective_products : ℕ := 2

/-- Represents the number of products drawn for inspection -/
def drawn_products : ℕ := 3

/-- Theorem for the number of ways to draw a specific non-defective product -/
theorem draw_specific_nondefective :
  (total_products - defective_products).choose (drawn_products - 1) = 36 := by sorry

/-- Theorem for the number of ways to draw at least one defective product -/
theorem draw_at_least_one_defective :
  (defective_products.choose 1 * (total_products - defective_products).choose (drawn_products - 1)) +
  (defective_products.choose 2 * (total_products - defective_products).choose (drawn_products - 2)) = 64 := by sorry

end NUMINAMATH_CALUDE_draw_specific_nondefective_draw_at_least_one_defective_l3488_348871


namespace NUMINAMATH_CALUDE_sum_equals_1529_l3488_348887

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The value of C in base 14 -/
def C : Nat := 12

/-- The value of D in base 14 -/
def D : Nat := 13

/-- Theorem stating that 345₁₃ + 4CD₁₄ = 1529 in base 10 -/
theorem sum_equals_1529 : 
  toBase10 [5, 4, 3] 13 + toBase10 [D, C, 4] 14 = 1529 := by sorry

end NUMINAMATH_CALUDE_sum_equals_1529_l3488_348887


namespace NUMINAMATH_CALUDE_triangle_side_sum_l3488_348845

theorem triangle_side_sum (a b c : ℝ) (h_angles : a = 30 ∧ b = 45 ∧ c = 105) 
  (h_sum : a + b + c = 180) (h_side : ∃ side : ℝ, side = 6 * Real.sqrt 2) : 
  ∃ (x y : ℝ), x + y = 18 + 6 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l3488_348845


namespace NUMINAMATH_CALUDE_height_matching_problem_sixteenth_answer_l3488_348835

/-- Represents a group of people with their height matches -/
structure HeightGroup :=
  (total : ℕ)
  (one_match : ℕ)
  (two_matches : ℕ)
  (three_matches : ℕ)
  (h_total : total = 16)
  (h_one : one_match = 6)
  (h_two : two_matches = 6)
  (h_three : three_matches = 3)

/-- The number of people accounted for by each match type -/
def accounted_for (g : HeightGroup) : ℕ :=
  g.one_match * 2 + g.two_matches * 3 + g.three_matches * 4

theorem height_matching_problem (g : HeightGroup) :
  accounted_for g = g.total + 3 :=
sorry

theorem sixteenth_answer (g : HeightGroup) :
  g.total - (g.one_match + g.two_matches + g.three_matches) = 1 ∧
  accounted_for g = g.total + 3 →
  3 = g.total - (accounted_for g - 3) :=
sorry

end NUMINAMATH_CALUDE_height_matching_problem_sixteenth_answer_l3488_348835


namespace NUMINAMATH_CALUDE_distinct_positive_factors_of_81_l3488_348857

theorem distinct_positive_factors_of_81 : 
  Finset.card (Nat.divisors 81) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distinct_positive_factors_of_81_l3488_348857


namespace NUMINAMATH_CALUDE_solution_count_l3488_348800

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem solution_count :
  let gcd_value := factorial 20
  let lcm_value := factorial 30
  (Finset.filter (fun p : ℕ × ℕ => 
    Nat.gcd p.1 p.2 = gcd_value ∧ 
    Nat.lcm p.1 p.2 = lcm_value
  ) (Finset.product (Finset.range (lcm_value + 1)) (Finset.range (lcm_value + 1)))).card = 1024 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_l3488_348800


namespace NUMINAMATH_CALUDE_remainder_of_x_divided_by_82_l3488_348875

theorem remainder_of_x_divided_by_82 (x : ℤ) (k m R : ℤ) 
  (h1 : x = 82 * k + R)
  (h2 : 0 ≤ R ∧ R < 82)
  (h3 : x + 7 = 41 * m + 12) :
  R = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_x_divided_by_82_l3488_348875


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_l3488_348878

/-- Checks if a natural number is a palindrome in the given base. -/
def is_palindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def to_base (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome :
  ∀ n : ℕ, n > 20 →
    (is_palindrome n 2 ∧ is_palindrome n 4) →
    n ≥ 21 :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_l3488_348878


namespace NUMINAMATH_CALUDE_product_of_solutions_absolute_value_equation_l3488_348806

theorem product_of_solutions_absolute_value_equation :
  ∃ (x₁ x₂ : ℝ), 
    (∀ x : ℝ, |x| = 3 * (|x| - 2) ↔ x = x₁ ∨ x = x₂) ∧
    x₁ * x₂ = -9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_absolute_value_equation_l3488_348806


namespace NUMINAMATH_CALUDE_distance_traveled_l3488_348825

-- Define the speed in km/hr
def speed : ℝ := 40

-- Define the time in hours
def time : ℝ := 6

-- Define the distance formula
def distance : ℝ := speed * time

-- Theorem to prove
theorem distance_traveled : distance = 240 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l3488_348825


namespace NUMINAMATH_CALUDE_opponent_score_proof_l3488_348869

def championship_game (total_points : ℕ) (num_games : ℕ) (point_difference : ℕ) : Prop :=
  let avg_points : ℚ := (total_points : ℚ) / num_games
  let uf_championship_score : ℚ := avg_points / 2 - 2
  let opponent_score : ℚ := uf_championship_score + point_difference
  opponent_score = 15

theorem opponent_score_proof :
  championship_game 720 24 2 := by
  sorry

end NUMINAMATH_CALUDE_opponent_score_proof_l3488_348869


namespace NUMINAMATH_CALUDE_probability_science_second_given_arts_first_l3488_348850

-- Define the total number of questions
def total_questions : ℕ := 5

-- Define the number of science questions
def science_questions : ℕ := 3

-- Define the number of arts questions
def arts_questions : ℕ := 2

-- Define the probability of drawing an arts question in the first draw
def prob_arts_first : ℚ := arts_questions / total_questions

-- Define the probability of drawing a science question in the second draw given an arts question was drawn first
def prob_science_second_given_arts_first : ℚ := science_questions / (total_questions - 1)

-- Theorem statement
theorem probability_science_second_given_arts_first :
  prob_science_second_given_arts_first = 3/4 :=
sorry

end NUMINAMATH_CALUDE_probability_science_second_given_arts_first_l3488_348850


namespace NUMINAMATH_CALUDE_cube_volume_from_face_diagonal_l3488_348837

theorem cube_volume_from_face_diagonal (d : ℝ) (h : d = 5 * Real.sqrt 2) :
  ∃ (s : ℝ), s > 0 ∧ s * Real.sqrt 2 = d ∧ s^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_diagonal_l3488_348837


namespace NUMINAMATH_CALUDE_min_inequality_solution_l3488_348821

theorem min_inequality_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y * z ≤ min (4 * (x - 1 / y)) (min (4 * (y - 1 / z)) (4 * (z - 1 / x)))) :
  x = Real.sqrt 2 ∧ y = Real.sqrt 2 ∧ z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_inequality_solution_l3488_348821


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3488_348899

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_abc_properties (t : Triangle) 
  (h1 : t.a = t.b * Real.cos t.C + (Real.sqrt 3 / 3) * t.c * Real.sin t.B)
  (h2 : t.a + t.c = 6)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 2) : 
  t.B = π/3 ∧ t.b = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3488_348899


namespace NUMINAMATH_CALUDE_rectangular_to_square_formation_l3488_348816

theorem rectangular_to_square_formation :
  ∃ n : ℕ,
    (∃ a : ℕ, 8 * n + 120 = a * a) ∧
    (∃ b : ℕ, 8 * n - 120 = b * b) →
    n = 17 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_square_formation_l3488_348816


namespace NUMINAMATH_CALUDE_school_students_count_l3488_348802

/-- The number of students who play football -/
def football : ℕ := 325

/-- The number of students who play cricket -/
def cricket : ℕ := 175

/-- The number of students who play neither football nor cricket -/
def neither : ℕ := 50

/-- The number of students who play both football and cricket -/
def both : ℕ := 140

/-- The total number of students in the school -/
def total_students : ℕ := football + cricket - both + neither

theorem school_students_count :
  total_students = 410 := by sorry

end NUMINAMATH_CALUDE_school_students_count_l3488_348802


namespace NUMINAMATH_CALUDE_bee_count_l3488_348812

theorem bee_count (initial_bees : ℕ) (h1 : initial_bees = 144) : 
  ⌊(3 * initial_bees : ℚ) * (1 - 0.2)⌋ = 346 := by
  sorry

end NUMINAMATH_CALUDE_bee_count_l3488_348812


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l3488_348848

theorem integer_solutions_of_equation :
  let S : Set (ℤ × ℤ) := {(x, y) | x * y - 2 * x - 2 * y + 7 = 0}
  S = {(5, 1), (-1, 3), (3, -1), (1, 5)} := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l3488_348848


namespace NUMINAMATH_CALUDE_simplify_fraction_l3488_348840

theorem simplify_fraction : (2^5 + 2^3) / (2^4 - 2^2) = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3488_348840


namespace NUMINAMATH_CALUDE_smallest_digit_change_l3488_348898

def original_sum : ℕ := 2457
def correct_sum : ℕ := 2547
def discrepancy : ℕ := correct_sum - original_sum

def num1 : ℕ := 731
def num2 : ℕ := 964
def num3 : ℕ := 852

def is_smallest_change (d : ℕ) : Prop :=
  d ≤ 9 ∧ 
  (num1 - d * 100 + num2 + num3 = correct_sum) ∧
  ∀ (d' : ℕ), d' < d → (num1 - d' * 100 + num2 + num3 ≠ correct_sum ∧
                        num1 + num2 - d' * 100 + num3 ≠ correct_sum ∧
                        num1 + num2 + num3 - d' * 100 ≠ correct_sum)

theorem smallest_digit_change :
  is_smallest_change 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_digit_change_l3488_348898


namespace NUMINAMATH_CALUDE_inequality_proof_l3488_348822

def M : Set ℝ := {x : ℝ | |x + 1| + |x - 1| ≤ 2}

theorem inequality_proof (x y z : ℝ) (hx : x ∈ M) (hy : |y| ≤ 1/6) (hz : |z| ≤ 1/9) :
  |x + 2*y - 3*z| ≤ 5/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3488_348822


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l3488_348827

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (base_is_square : base_side > 0)
  (lateral_faces_equilateral : True)

/-- A cube inscribed in a pyramid -/
structure InscribedCube (p : Pyramid) :=
  (side_length : ℝ)
  (base_on_pyramid_base : True)
  (top_face_on_lateral_faces : True)

/-- The volume of the inscribed cube -/
def cube_volume (p : Pyramid) (c : InscribedCube p) : ℝ :=
  c.side_length ^ 3

/-- The theorem stating the volume of the inscribed cube -/
theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube p)
  (h : p.base_side = 1) :
  cube_volume p c = 5 * Real.sqrt 2 - 7 :=
sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l3488_348827


namespace NUMINAMATH_CALUDE_claire_photos_l3488_348882

theorem claire_photos (lisa robert claire : ℕ) 
  (h1 : lisa = robert)
  (h2 : lisa = 3 * claire)
  (h3 : robert = claire + 20) :
  claire = 10 := by
  sorry

end NUMINAMATH_CALUDE_claire_photos_l3488_348882


namespace NUMINAMATH_CALUDE_first_prize_winners_l3488_348823

theorem first_prize_winners (n : ℕ) : 
  (30 ≤ n ∧ n ≤ 55) ∧ 
  (n % 3 = 2) ∧ 
  (n % 5 = 4) ∧ 
  (n % 7 = 2) → 
  n = 44 := by
sorry

end NUMINAMATH_CALUDE_first_prize_winners_l3488_348823


namespace NUMINAMATH_CALUDE_matrix_product_equality_l3488_348846

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 4]

theorem matrix_product_equality :
  A * B = !![17, -5; 16, -20] := by sorry

end NUMINAMATH_CALUDE_matrix_product_equality_l3488_348846


namespace NUMINAMATH_CALUDE_stratified_sample_science_students_l3488_348884

theorem stratified_sample_science_students 
  (total_students : ℕ) 
  (science_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 140) 
  (h2 : science_students = 100) 
  (h3 : sample_size = 14) :
  (sample_size : ℚ) / total_students * science_students = 10 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_science_students_l3488_348884
