import Mathlib

namespace NUMINAMATH_CALUDE_diamond_calculation_l1053_105309

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := Int.natAbs (a + b - 10)

-- Theorem statement
theorem diamond_calculation : diamond 5 (diamond 3 8) = 4 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l1053_105309


namespace NUMINAMATH_CALUDE_previous_painting_price_l1053_105390

/-- 
Given a painter whose most recent painting sold for $44,000, and this price is $1000 less than 
five times more than his previous painting, prove that the price of the previous painting was $9,000.
-/
theorem previous_painting_price (recent_price previous_price : ℕ) : 
  recent_price = 44000 ∧ 
  recent_price = 5 * previous_price - 1000 →
  previous_price = 9000 := by
sorry

end NUMINAMATH_CALUDE_previous_painting_price_l1053_105390


namespace NUMINAMATH_CALUDE_set_A_nonempty_iff_l1053_105396

def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*x - a = 0}

theorem set_A_nonempty_iff (a : ℝ) : Set.Nonempty (A a) ↔ a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_set_A_nonempty_iff_l1053_105396


namespace NUMINAMATH_CALUDE_cyrus_shot_percentage_l1053_105377

def total_shots : ℕ := 20
def missed_shots : ℕ := 4

def shots_made : ℕ := total_shots - missed_shots

def percentage_made : ℚ := (shots_made : ℚ) / (total_shots : ℚ) * 100

theorem cyrus_shot_percentage :
  percentage_made = 80 := by
  sorry

end NUMINAMATH_CALUDE_cyrus_shot_percentage_l1053_105377


namespace NUMINAMATH_CALUDE_two_never_appears_l1053_105329

/-- Represents a move in the game -/
def Move (s : List Int) : List Int :=
  -- Implementation details omitted
  sorry

/-- Represents the state of the board after any number of moves -/
inductive BoardState
| initial (n : Nat) : BoardState
| after_move (prev : BoardState) : BoardState

/-- The sequence of numbers on the board -/
def board_sequence (state : BoardState) : List Int :=
  match state with
  | BoardState.initial n => List.range (2*n) -- Simplified representation
  | BoardState.after_move prev => Move (board_sequence prev)

/-- Theorem stating that 2 never appears after any number of moves -/
theorem two_never_appears (n : Nat) (state : BoardState) : 
  2 ∉ board_sequence state :=
sorry

end NUMINAMATH_CALUDE_two_never_appears_l1053_105329


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l1053_105313

theorem subcommittee_formation_count :
  let total_republicans : ℕ := 8
  let total_democrats : ℕ := 10
  let subcommittee_republicans : ℕ := 3
  let subcommittee_democrats : ℕ := 2
  let ways_to_choose_republicans : ℕ := Nat.choose total_republicans subcommittee_republicans
  let ways_to_choose_chair : ℕ := total_democrats
  let ways_to_choose_other_democrat : ℕ := Nat.choose (total_democrats - 1) (subcommittee_democrats - 1)
  ways_to_choose_republicans * ways_to_choose_chair * ways_to_choose_other_democrat = 5040 :=
by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l1053_105313


namespace NUMINAMATH_CALUDE_quadratic_sum_zero_l1053_105384

/-- A quadratic function f(x) = ax^2 + bx + c with specified properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum_zero
  (a b c : ℝ)
  (h_min : ∃ x, ∀ y, QuadraticFunction a b c y ≥ QuadraticFunction a b c x ∧ QuadraticFunction a b c x = 36)
  (h_root1 : QuadraticFunction a b c 1 = 0)
  (h_root5 : QuadraticFunction a b c 5 = 0) :
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_zero_l1053_105384


namespace NUMINAMATH_CALUDE_lighter_box_weight_l1053_105319

/-- Proves that the weight of lighter boxes is 12 pounds given the conditions of the shipment. -/
theorem lighter_box_weight (total_boxes : Nat) (heavier_box_weight : Nat) (initial_average : Nat) 
  (final_average : Nat) (removed_boxes : Nat) :
  total_boxes = 20 →
  heavier_box_weight = 20 →
  initial_average = 18 →
  final_average = 12 →
  removed_boxes = 15 →
  ∃ (lighter_box_weight : Nat), 
    lighter_box_weight = 12 ∧
    lighter_box_weight * (total_boxes - removed_boxes) = 
      final_average * (total_boxes - removed_boxes) :=
by sorry

end NUMINAMATH_CALUDE_lighter_box_weight_l1053_105319


namespace NUMINAMATH_CALUDE_residue_system_product_condition_l1053_105327

/-- A function that generates a complete residue system modulo n -/
def completeResidueSystem (n : ℕ) : Fin n → ℕ :=
  fun i => i.val

/-- Predicate to check if a list of natural numbers forms a complete residue system modulo n -/
def isCompleteResidueSystem (n : ℕ) (l : List ℕ) : Prop :=
  l.length = n ∧ ∀ k, 0 ≤ k ∧ k < n → ∃ x ∈ l, x % n = k

theorem residue_system_product_condition (n : ℕ) : 
  (∃ (a b : Fin n → ℕ), 
    isCompleteResidueSystem n (List.ofFn a) ∧
    isCompleteResidueSystem n (List.ofFn b) ∧
    isCompleteResidueSystem n (List.ofFn (fun i => (a i * b i) % n))) ↔ 
  n = 1 ∨ n = 2 :=
sorry

end NUMINAMATH_CALUDE_residue_system_product_condition_l1053_105327


namespace NUMINAMATH_CALUDE_clubsuit_calculation_l1053_105326

/-- Custom operation ⊗ for real numbers -/
def clubsuit (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

/-- Theorem stating that 5 ⊗ (7 ⊗ 8) = 4480 -/
theorem clubsuit_calculation : clubsuit 5 (clubsuit 7 8) = 4480 := by
  sorry

end NUMINAMATH_CALUDE_clubsuit_calculation_l1053_105326


namespace NUMINAMATH_CALUDE_system_of_equations_l1053_105388

theorem system_of_equations (x y z k : ℝ) : 
  (2 * x - y + 3 * z = 9) → 
  (x + 2 * y - z = k) → 
  (-x + y + 4 * z = 6) → 
  (y = -1) → 
  (k = -3) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l1053_105388


namespace NUMINAMATH_CALUDE_product_of_specific_primes_l1053_105387

def largest_one_digit_prime : ℕ := 7

def largest_two_digit_primes : List ℕ := [97, 89]

theorem product_of_specific_primes : 
  (largest_one_digit_prime * (largest_two_digit_primes.prod)) = 60431 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_primes_l1053_105387


namespace NUMINAMATH_CALUDE_real_part_of_z_is_zero_l1053_105331

theorem real_part_of_z_is_zero :
  let i : ℂ := Complex.I
  let z : ℂ := (2 + i) / (-2*i + 1)
  Complex.re z = 0 := by
sorry

end NUMINAMATH_CALUDE_real_part_of_z_is_zero_l1053_105331


namespace NUMINAMATH_CALUDE_smallest_class_is_four_l1053_105302

/-- Represents a systematic sampling of classes. -/
structure SystematicSampling where
  total_classes : ℕ
  selected_classes : ℕ
  sum_of_selected : ℕ

/-- The smallest class number in a systematic sampling. -/
def smallest_class (s : SystematicSampling) : ℕ :=
  (s.sum_of_selected - (s.selected_classes * (s.selected_classes - 1) * s.total_classes / s.selected_classes / 2)) / s.selected_classes

/-- Theorem stating that for the given conditions, the smallest class number is 4. -/
theorem smallest_class_is_four (s : SystematicSampling) 
  (h1 : s.total_classes = 24)
  (h2 : s.selected_classes = 4)
  (h3 : s.sum_of_selected = 52) :
  smallest_class s = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_class_is_four_l1053_105302


namespace NUMINAMATH_CALUDE_balls_in_boxes_l1053_105338

def num_balls : ℕ := 6
def num_boxes : ℕ := 3

theorem balls_in_boxes :
  (num_boxes ^ num_balls : ℕ) = 729 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l1053_105338


namespace NUMINAMATH_CALUDE_white_balls_estimate_l1053_105359

theorem white_balls_estimate (total_balls : ℕ) (total_draws : ℕ) (white_draws : ℕ) 
  (h_total_balls : total_balls = 20)
  (h_total_draws : total_draws = 100)
  (h_white_draws : white_draws = 40) :
  (white_draws : ℚ) / total_draws * total_balls = 8 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_estimate_l1053_105359


namespace NUMINAMATH_CALUDE_calculate_expression_l1053_105310

theorem calculate_expression : ((28 / (5 + 3 - 6)) * 7) = 98 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1053_105310


namespace NUMINAMATH_CALUDE_smallest_m_for_equal_notebooks_and_pencils_l1053_105385

theorem smallest_m_for_equal_notebooks_and_pencils :
  ∃ (M : ℕ+), (M = 5) ∧
  (∀ (k : ℕ+), k < M → ¬∃ (n : ℕ+), 3 * k = 5 * n) ∧
  (∃ (n : ℕ+), 3 * M = 5 * n) := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_for_equal_notebooks_and_pencils_l1053_105385


namespace NUMINAMATH_CALUDE_probability_ten_nine_eight_sequence_l1053_105305

theorem probability_ten_nine_eight_sequence (deck : Nat) (tens : Nat) (nines : Nat) (eights : Nat) :
  deck = 52 →
  tens = 4 →
  nines = 4 →
  eights = 4 →
  (tens / deck) * (nines / (deck - 1)) * (eights / (deck - 2)) = 4 / 33150 := by
  sorry

end NUMINAMATH_CALUDE_probability_ten_nine_eight_sequence_l1053_105305


namespace NUMINAMATH_CALUDE_leila_bought_two_armchairs_l1053_105339

/-- Represents the living room set purchase --/
structure LivingRoomSet where
  sofaCost : ℕ
  armchairCost : ℕ
  coffeeTableCost : ℕ
  totalCost : ℕ

/-- Calculates the number of armchairs in the living room set --/
def numberOfArmchairs (set : LivingRoomSet) : ℕ :=
  (set.totalCost - set.sofaCost - set.coffeeTableCost) / set.armchairCost

/-- Theorem stating that Leila bought 2 armchairs --/
theorem leila_bought_two_armchairs (set : LivingRoomSet)
    (h1 : set.sofaCost = 1250)
    (h2 : set.armchairCost = 425)
    (h3 : set.coffeeTableCost = 330)
    (h4 : set.totalCost = 2430) :
    numberOfArmchairs set = 2 := by
  sorry

#eval numberOfArmchairs {
  sofaCost := 1250,
  armchairCost := 425,
  coffeeTableCost := 330,
  totalCost := 2430
}

end NUMINAMATH_CALUDE_leila_bought_two_armchairs_l1053_105339


namespace NUMINAMATH_CALUDE_temperature_below_freezing_is_negative_three_l1053_105349

/-- The freezing point of water in degrees Celsius -/
def freezing_point : ℝ := 0

/-- The temperature difference in degrees Celsius -/
def temperature_difference : ℝ := 3

/-- The temperature below freezing point -/
def temperature_below_freezing : ℝ := freezing_point - temperature_difference

theorem temperature_below_freezing_is_negative_three :
  temperature_below_freezing = -3 := by sorry

end NUMINAMATH_CALUDE_temperature_below_freezing_is_negative_three_l1053_105349


namespace NUMINAMATH_CALUDE_divisibility_by_290304_l1053_105303

theorem divisibility_by_290304 (a b : Nat) (ha : Nat.Prime a) (hb : Nat.Prime b) 
  (ga : a > 7) (gb : b > 7) : 
  290304 ∣ (a^2 - 1) * (b^2 - 1) * (a^6 - b^6) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_290304_l1053_105303


namespace NUMINAMATH_CALUDE_church_capacity_l1053_105335

/-- Calculates the number of usable chairs in a church with three sections -/
def total_usable_chairs : ℕ :=
  let section1_rows : ℕ := 15
  let section1_chairs_per_row : ℕ := 8
  let section1_unusable_per_row : ℕ := 3
  let section2_rows : ℕ := 20
  let section2_chairs_per_row : ℕ := 6
  let section2_unavailable_rows : ℕ := 2
  let section3_rows : ℕ := 25
  let section3_chairs_per_row : ℕ := 10
  let section3_unusable_every_second : ℕ := 5

  let section1_usable := section1_rows * (section1_chairs_per_row - section1_unusable_per_row)
  let section2_usable := (section2_rows - section2_unavailable_rows) * section2_chairs_per_row
  let section3_usable := (section3_rows / 2) * section3_chairs_per_row + 
                         (section3_rows - section3_rows / 2) * (section3_chairs_per_row - section3_unusable_every_second)

  section1_usable + section2_usable + section3_usable

theorem church_capacity : total_usable_chairs = 373 := by
  sorry

end NUMINAMATH_CALUDE_church_capacity_l1053_105335


namespace NUMINAMATH_CALUDE_problems_per_worksheet_l1053_105351

theorem problems_per_worksheet
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (remaining_problems : ℕ)
  (h1 : total_worksheets = 15)
  (h2 : graded_worksheets = 7)
  (h3 : remaining_problems = 24)
  : (remaining_problems / (total_worksheets - graded_worksheets) : ℚ) = 3 :=
by sorry

end NUMINAMATH_CALUDE_problems_per_worksheet_l1053_105351


namespace NUMINAMATH_CALUDE_correct_calculation_l1053_105311

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * x^2 * y = x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1053_105311


namespace NUMINAMATH_CALUDE_min_blocks_for_garden_wall_l1053_105323

/-- Represents the configuration of a garden wall --/
structure WallConfig where
  length : ℕ
  height : ℕ
  blockHeight : ℕ
  shortBlockLength : ℕ
  longBlockLength : ℕ

/-- Calculates the minimum number of blocks required for the wall --/
def minBlocksRequired (config : WallConfig) : ℕ :=
  sorry

/-- The specific wall configuration from the problem --/
def gardenWall : WallConfig :=
  { length := 90
  , height := 8
  , blockHeight := 1
  , shortBlockLength := 2
  , longBlockLength := 3 }

/-- Theorem stating that the minimum number of blocks required is 244 --/
theorem min_blocks_for_garden_wall :
  minBlocksRequired gardenWall = 244 :=
sorry

end NUMINAMATH_CALUDE_min_blocks_for_garden_wall_l1053_105323


namespace NUMINAMATH_CALUDE_rational_equation_solution_l1053_105355

theorem rational_equation_solution (x : ℝ) : 
  (1 / (x^2 + 12*x - 9) + 1 / (x^2 + 3*x - 9) + 1 / (x^2 - 14*x - 9) = 0) ↔ 
  (x = 3 ∨ x = 1 ∨ x = -3 ∨ x = -9) := by
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l1053_105355


namespace NUMINAMATH_CALUDE_dragon_defeat_probability_is_one_l1053_105344

/-- Represents the state of the dragon's heads -/
structure DragonState where
  heads : ℕ

/-- Represents the possible outcomes after chopping off a head -/
inductive ChopOutcome
  | TwoHeadsGrow
  | OneHeadGrows
  | NoHeadGrows

/-- The probability distribution of chop outcomes -/
def chopProbability : ChopOutcome → ℚ
  | ChopOutcome.TwoHeadsGrow => 1/4
  | ChopOutcome.OneHeadGrows => 1/3
  | ChopOutcome.NoHeadGrows => 5/12

/-- The transition function for the dragon state after a chop -/
def transition (state : DragonState) (outcome : ChopOutcome) : DragonState :=
  match outcome with
  | ChopOutcome.TwoHeadsGrow => ⟨state.heads + 1⟩
  | ChopOutcome.OneHeadGrows => state
  | ChopOutcome.NoHeadGrows => ⟨state.heads - 1⟩

/-- The probability of eventually defeating the dragon -/
noncomputable def defeatProbability (initialState : DragonState) : ℝ :=
  sorry

/-- Theorem stating that the probability of defeating the dragon is 1 -/
theorem dragon_defeat_probability_is_one :
  defeatProbability ⟨3⟩ = 1 := by sorry

end NUMINAMATH_CALUDE_dragon_defeat_probability_is_one_l1053_105344


namespace NUMINAMATH_CALUDE_custom_op_result_l1053_105312

-- Define the custom operation
def custom_op (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem custom_op_result : custom_op (custom_op 12 8) 2 = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l1053_105312


namespace NUMINAMATH_CALUDE_tsunami_area_theorem_l1053_105345

/-- Regular tetrahedron with edge length 900 km -/
structure Tetrahedron where
  edge_length : ℝ
  regular : edge_length = 900

/-- Tsunami propagation properties -/
structure Tsunami where
  speed : ℝ
  time : ℝ
  speed_is_300 : speed = 300
  time_is_2 : time = 2

/-- Epicenter location -/
inductive EpicenterLocation
  | FaceCenter
  | EdgeMidpoint

/-- Area covered by tsunami -/
noncomputable def tsunami_area (t : Tetrahedron) (w : Tsunami) (loc : EpicenterLocation) : ℝ :=
  match loc with
  | EpicenterLocation.FaceCenter => 180000 * Real.pi + 270000 * Real.sqrt 3
  | EpicenterLocation.EdgeMidpoint => 720000 * Real.arccos (3/4) + 135000 * Real.sqrt 7

/-- Main theorem -/
theorem tsunami_area_theorem (t : Tetrahedron) (w : Tsunami) :
  (tsunami_area t w EpicenterLocation.FaceCenter = 180000 * Real.pi + 270000 * Real.sqrt 3) ∧
  (tsunami_area t w EpicenterLocation.EdgeMidpoint = 720000 * Real.arccos (3/4) + 135000 * Real.sqrt 7) := by
  sorry


end NUMINAMATH_CALUDE_tsunami_area_theorem_l1053_105345


namespace NUMINAMATH_CALUDE_A_intersect_Z_l1053_105353

def A : Set ℝ := {x : ℝ | |x - 1| < 2}

theorem A_intersect_Z : A ∩ Set.range (Int.cast : ℤ → ℝ) = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_Z_l1053_105353


namespace NUMINAMATH_CALUDE_square_root_equal_self_l1053_105347

theorem square_root_equal_self (a : ℝ) : 
  (Real.sqrt a = a) → (a^2 + 1 = 1 ∨ a^2 + 1 = 2) := by
  sorry

end NUMINAMATH_CALUDE_square_root_equal_self_l1053_105347


namespace NUMINAMATH_CALUDE_stable_performance_lower_variance_athlete_a_more_stable_l1053_105348

-- Define the structure for an athlete's performance
structure AthletePerformance where
  average_score : ℝ
  variance : ℝ
  variance_positive : variance > 0

-- Define the notion of stability
def more_stable (a b : AthletePerformance) : Prop :=
  a.variance < b.variance

-- Theorem statement
theorem stable_performance_lower_variance 
  (a b : AthletePerformance) 
  (h_equal_avg : a.average_score = b.average_score) :
  more_stable a b ↔ a.variance < b.variance :=
sorry

-- Specific instance for the given problem
def athlete_a : AthletePerformance := {
  average_score := 9
  variance := 1.2
  variance_positive := by norm_num
}

def athlete_b : AthletePerformance := {
  average_score := 9
  variance := 2.4
  variance_positive := by norm_num
}

-- Theorem application to the specific instance
theorem athlete_a_more_stable : more_stable athlete_a athlete_b :=
sorry

end NUMINAMATH_CALUDE_stable_performance_lower_variance_athlete_a_more_stable_l1053_105348


namespace NUMINAMATH_CALUDE_expression_value_l1053_105380

theorem expression_value : 4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 8000 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1053_105380


namespace NUMINAMATH_CALUDE_binary_to_base5_equivalence_l1053_105308

-- Define the binary number
def binary_num : ℕ := 168  -- 10101000 in binary is 168 in decimal

-- Define the base-5 number
def base5_num : List ℕ := [1, 1, 3, 3]  -- 1133 in base-5

-- Theorem to prove the equivalence
theorem binary_to_base5_equivalence :
  (binary_num : ℕ) = (List.foldl (λ acc d => acc * 5 + d) 0 base5_num) :=
sorry

end NUMINAMATH_CALUDE_binary_to_base5_equivalence_l1053_105308


namespace NUMINAMATH_CALUDE_find_a_l1053_105354

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {2, 4, 3 - a^2}

-- Define set P
def P (a : ℝ) : Set ℝ := {2, a^2 - a + 2}

-- Define the complement of P with respect to U
def complementP (a : ℝ) : Set ℝ := {-1}

-- Theorem statement
theorem find_a : ∃ a : ℝ, 
  (U a = P a ∪ complementP a) ∧ 
  (U a = {2, 4, 3 - a^2}) ∧ 
  (P a = {2, a^2 - a + 2}) ∧ 
  (complementP a = {-1}) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_find_a_l1053_105354


namespace NUMINAMATH_CALUDE_circle_equation_and_tangent_lines_l1053_105389

/-- Circle C with center (a, b) and radius 5 -/
structure CircleC where
  a : ℝ
  b : ℝ
  center_on_line : a + b + 1 = 0
  passes_through_p : ((-2) - a)^2 + (0 - b)^2 = 25
  passes_through_q : (5 - a)^2 + (1 - b)^2 = 25

/-- Tangent line to circle C passing through point A(-3, 0) -/
structure TangentLine where
  k : ℝ

theorem circle_equation_and_tangent_lines (c : CircleC) :
  ((c.a = 2 ∧ c.b = -3) ∧
   (∀ x y : ℝ, (x - 2)^2 + (y + 3)^2 = 25 ↔ (x - c.a)^2 + (y - c.b)^2 = 25)) ∧
  (∃ t : TangentLine,
    (t.k = 0 ∧ ∀ x y : ℝ, y = t.k * (x + 3) ↔ x = -3) ∨
    (t.k = 8/15 ∧ ∀ x y : ℝ, y = t.k * (x + 3) ↔ y = (8/15) * (x + 3))) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_and_tangent_lines_l1053_105389


namespace NUMINAMATH_CALUDE_equation_solution_l1053_105343

theorem equation_solution : 
  ∀ x y z : ℕ+, 
  (x : ℚ) / 21 * (y : ℚ) / 189 + (z : ℚ) = 1 → 
  x = 21 ∧ y = 567 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1053_105343


namespace NUMINAMATH_CALUDE_division_addition_equality_l1053_105314

theorem division_addition_equality : (-180) / (-45) + (-9) = -5 := by
  sorry

end NUMINAMATH_CALUDE_division_addition_equality_l1053_105314


namespace NUMINAMATH_CALUDE_number_with_fraction_difference_l1053_105358

theorem number_with_fraction_difference (x : ℤ) : x - (7 : ℤ) * x / (13 : ℤ) = 110 ↔ x = 237 := by
  sorry

end NUMINAMATH_CALUDE_number_with_fraction_difference_l1053_105358


namespace NUMINAMATH_CALUDE_sequence_sum_equals_eight_l1053_105307

/-- Given a geometric sequence and an arithmetic sequence with specific properties, 
    prove that the sum of two terms in the arithmetic sequence equals 8. -/
theorem sequence_sum_equals_eight 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h_geometric : ∀ n m : ℕ, a (n + m) = a n * (a 1) ^ m) 
  (h_arithmetic : ∀ n m : ℕ, b (n + m) = b n + m * (b 1 - b 0)) 
  (h_relation : a 3 * a 11 = 4 * a 7) 
  (h_equal : b 7 = a 7) : 
  b 5 + b 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_eight_l1053_105307


namespace NUMINAMATH_CALUDE_car_production_increase_l1053_105300

/-- Proves that adding 50 cars to the monthly production of 100 cars
    will result in an annual production of 1800 cars. -/
theorem car_production_increase (current_monthly : ℕ) (target_yearly : ℕ) (increase : ℕ) :
  current_monthly = 100 →
  target_yearly = 1800 →
  increase = 50 →
  (current_monthly + increase) * 12 = target_yearly := by
  sorry

#check car_production_increase

end NUMINAMATH_CALUDE_car_production_increase_l1053_105300


namespace NUMINAMATH_CALUDE_archer_probability_l1053_105361

theorem archer_probability (p10 p9 p8 : ℝ) (h1 : p10 = 0.2) (h2 : p9 = 0.3) (h3 : p8 = 0.3) :
  1 - p10 - p9 - p8 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_archer_probability_l1053_105361


namespace NUMINAMATH_CALUDE_ratio_problem_l1053_105374

theorem ratio_problem (A B C : ℝ) (h1 : A + B + C = 98) (h2 : A / B = 2 / 3) (h3 : B = 30) :
  B / C = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1053_105374


namespace NUMINAMATH_CALUDE_exists_special_sequence_l1053_105356

def sequence_condition (a : ℕ → ℕ) : Prop :=
  ∀ i j p q r, i ≠ j ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r →
    Nat.gcd (a i + a j) (a p + a q + a r) = 1

theorem exists_special_sequence :
  ∃ a : ℕ → ℕ, sequence_condition a ∧ (∀ n, a n < a (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_exists_special_sequence_l1053_105356


namespace NUMINAMATH_CALUDE_arithmetic_problem_l1053_105386

theorem arithmetic_problem : 4 * (8 - 3) / 2 - 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l1053_105386


namespace NUMINAMATH_CALUDE_consecutive_integers_with_unique_prime_factors_l1053_105373

theorem consecutive_integers_with_unique_prime_factors (n : ℕ) (hn : n > 0) :
  ∃ x : ℤ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n →
    ∃ (p : ℕ) (k : ℕ), Prime p ∧ (x + i : ℤ) = p * k ∧ ¬(p ∣ k) :=
sorry

end NUMINAMATH_CALUDE_consecutive_integers_with_unique_prime_factors_l1053_105373


namespace NUMINAMATH_CALUDE_unique_eight_times_digit_sum_l1053_105328

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem unique_eight_times_digit_sum :
  ∃! n : ℕ, n < 500 ∧ n > 0 ∧ n = 8 * sum_of_digits n := by sorry

end NUMINAMATH_CALUDE_unique_eight_times_digit_sum_l1053_105328


namespace NUMINAMATH_CALUDE_convex_quadrilateral_probability_l1053_105394

/-- The number of points on the circle -/
def n : ℕ := 7

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords with n points -/
def total_chords : ℕ := n.choose 2

/-- The probability of four randomly selected chords from n points on a circle forming a convex quadrilateral -/
theorem convex_quadrilateral_probability :
  (n.choose k : ℚ) / (total_chords.choose k : ℚ) = 1 / 171 :=
sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_probability_l1053_105394


namespace NUMINAMATH_CALUDE_germination_probability_l1053_105368

/-- The probability of exactly k successes in n independent Bernoulli trials -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The germination rate of seeds -/
def germination_rate : ℝ := 0.9

/-- The number of seeds sown -/
def total_seeds : ℕ := 7

/-- The number of seeds expected to germinate -/
def germinated_seeds : ℕ := 5

theorem germination_probability :
  binomial_probability total_seeds germinated_seeds germination_rate =
  21 * (germination_rate^5) * ((1 - germination_rate)^2) :=
by sorry

end NUMINAMATH_CALUDE_germination_probability_l1053_105368


namespace NUMINAMATH_CALUDE_longest_lifetime_l1053_105393

/-- A binary string is a list of booleans, where true represents 1 and false represents 0. -/
def BinaryString := List Bool

/-- The transformation function f as described in the problem. -/
def f (s : BinaryString) : BinaryString :=
  sorry

/-- The lifetime of a binary string is the number of times f can be applied until no falses remain. -/
def lifetime (s : BinaryString) : Nat :=
  sorry

/-- Generate a binary string of length n with repeated 110 pattern. -/
def repeated110 (n : Nat) : BinaryString :=
  sorry

/-- Theorem: For any n ≥ 2, the binary string with repeated 110 pattern has the longest lifetime. -/
theorem longest_lifetime (n : Nat) (h : n ≥ 2) :
  ∀ s : BinaryString, s.length = n → lifetime (repeated110 n) ≥ lifetime s :=
  sorry

end NUMINAMATH_CALUDE_longest_lifetime_l1053_105393


namespace NUMINAMATH_CALUDE_a_10_value_l1053_105391

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_10_value
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_7 : a 7 = 9)
  (h_13 : a 13 = -3) :
  a 10 = 3 :=
sorry

end NUMINAMATH_CALUDE_a_10_value_l1053_105391


namespace NUMINAMATH_CALUDE_pizza_meat_calculation_l1053_105350

/-- Represents the number of pieces of each type of meat on a pizza --/
structure PizzaToppings where
  pepperoni : ℕ
  ham : ℕ
  sausage : ℕ

/-- Calculates the total number of pieces of meat on each slice of pizza --/
def meat_per_slice (toppings : PizzaToppings) (slices : ℕ) : ℚ :=
  (toppings.pepperoni + toppings.ham + toppings.sausage : ℚ) / slices

theorem pizza_meat_calculation :
  let toppings : PizzaToppings := {
    pepperoni := 30,
    ham := 30 * 2,
    sausage := 30 + 12
  }
  let slices : ℕ := 6
  meat_per_slice toppings slices = 22 := by
  sorry

#eval meat_per_slice { pepperoni := 30, ham := 30 * 2, sausage := 30 + 12 } 6

end NUMINAMATH_CALUDE_pizza_meat_calculation_l1053_105350


namespace NUMINAMATH_CALUDE_binomial_product_l1053_105332

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l1053_105332


namespace NUMINAMATH_CALUDE_cube_surface_area_l1053_105375

theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 20) :
  6 * edge_length^2 = 2400 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1053_105375


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1053_105324

theorem quadratic_factorization (y : ℝ) (A B : ℤ) 
  (h : ∀ y, 12 * y^2 - 65 * y + 42 = (A * y - 14) * (B * y - 3)) : 
  A * B + A = 15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1053_105324


namespace NUMINAMATH_CALUDE_inequality_solution_l1053_105370

theorem inequality_solution (x : ℝ) :
  (2 - 1 / (3 * x + 4) < 5) ↔ (x < -4/3 ∨ x > -13/9) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1053_105370


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1053_105382

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ = 2 + Real.sqrt 6 ∧ x₁^2 - 4*x₁ = 2) ∧ 
              (x₂ = 2 - Real.sqrt 6 ∧ x₂^2 - 4*x₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1053_105382


namespace NUMINAMATH_CALUDE_bridge_length_l1053_105316

/-- The length of a bridge given a train crossing it -/
theorem bridge_length (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  train_length = 100 →
  crossing_time = 60 →
  train_speed = 5 →
  train_speed * crossing_time - train_length = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l1053_105316


namespace NUMINAMATH_CALUDE_rotated_square_height_l1053_105367

theorem rotated_square_height :
  let square_side : ℝ := 1
  let rotation_angle : ℝ := 60 * (π / 180)  -- 60 degrees in radians
  let diagonal : ℝ := square_side * Real.sqrt 2
  let height_above_center : ℝ := (diagonal / 2) * Real.sin rotation_angle
  let original_center_height : ℝ := square_side / 2
  let total_height : ℝ := original_center_height + height_above_center
  total_height = (2 + Real.sqrt 6) / 4 := by sorry

end NUMINAMATH_CALUDE_rotated_square_height_l1053_105367


namespace NUMINAMATH_CALUDE_largest_temperature_time_l1053_105317

theorem largest_temperature_time (t : ℝ) : 
  (-t^2 + 10*t + 40 = 60) → t ≤ 5 + Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_temperature_time_l1053_105317


namespace NUMINAMATH_CALUDE_trig_identity_l1053_105336

theorem trig_identity (a b : ℝ) (θ : ℝ) (h : (Real.sin θ)^6 / a + (Real.cos θ)^6 / b = 1 / (a + b)) :
  (Real.sin θ)^12 / a^2 + (Real.cos θ)^12 / b^2 = (a^4 + b^4) / (a + b)^6 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1053_105336


namespace NUMINAMATH_CALUDE_correct_comparison_l1053_105392

theorem correct_comparison :
  (-5/6 : ℚ) < -4/5 ∧
  ¬(-(-21) < -21) ∧
  ¬(-(abs (-21/2)) > 26/3) ∧
  ¬(-(abs (-23/3)) > -(-23/3)) :=
by sorry

end NUMINAMATH_CALUDE_correct_comparison_l1053_105392


namespace NUMINAMATH_CALUDE_truncated_trigonal_pyramid_theorem_l1053_105357

/-- A truncated trigonal pyramid circumscribed around a sphere -/
structure TruncatedTrigonalPyramid where
  /-- The altitude of the pyramid -/
  h : ℝ
  /-- The circumradius of the lower base -/
  R₁ : ℝ
  /-- The circumradius of the upper base -/
  R₂ : ℝ
  /-- The distance from the circumcenter of the lower base to the point where the sphere touches it -/
  O₁T₁ : ℝ
  /-- The distance from the circumcenter of the upper base to the point where the sphere touches it -/
  O₂T₂ : ℝ
  /-- The sphere touches both bases of the pyramid -/
  touches_bases : True

/-- The main theorem about the relationship between the measurements of a truncated trigonal pyramid -/
theorem truncated_trigonal_pyramid_theorem (p : TruncatedTrigonalPyramid) :
  p.R₁ * p.R₂ * p.h^2 = (p.R₁^2 - p.O₁T₁^2) * (p.R₂^2 - p.O₂T₂^2) := by
  sorry

end NUMINAMATH_CALUDE_truncated_trigonal_pyramid_theorem_l1053_105357


namespace NUMINAMATH_CALUDE_ferris_break_length_is_correct_l1053_105306

/-- Represents the job completion scenario with Audrey and Ferris --/
structure JobCompletion where
  audrey_solo_time : ℝ
  ferris_solo_time : ℝ
  collaboration_time : ℝ
  ferris_break_count : ℕ

/-- Calculates the length of each of Ferris' breaks in minutes --/
def ferris_break_length (job : JobCompletion) : ℝ :=
  2.5

/-- Theorem stating that Ferris' break length is 2.5 minutes under the given conditions --/
theorem ferris_break_length_is_correct (job : JobCompletion) 
  (h1 : job.audrey_solo_time = 4)
  (h2 : job.ferris_solo_time = 3)
  (h3 : job.collaboration_time = 2)
  (h4 : job.ferris_break_count = 6) :
  ferris_break_length job = 2.5 := by
  sorry

#eval ferris_break_length { audrey_solo_time := 4, ferris_solo_time := 3, collaboration_time := 2, ferris_break_count := 6 }

end NUMINAMATH_CALUDE_ferris_break_length_is_correct_l1053_105306


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1053_105352

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 1 / a^2 → a^2 > 1 / a) ∧
  (∃ a, a^2 > 1 / a ∧ a ≤ 1 / a^2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1053_105352


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1053_105325

theorem triangle_perimeter (a b c : ℝ) (ha : a = Real.sqrt 8) (hb : b = Real.sqrt 18) (hc : c = Real.sqrt 32) :
  a + b + c = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1053_105325


namespace NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l1053_105340

/-- A circle with center (a, b) and radius r -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- The sum of y-coordinates of intersection points between a circle and the y-axis -/
def sumYIntersections (c : Circle) : ℝ :=
  2 * c.b

/-- Theorem: For a circle with center (-3, -4) and radius 7, 
    the sum of y-coordinates of its intersection points with the y-axis is -8 -/
theorem circle_y_axis_intersection_sum :
  ∃ (c : Circle), c.a = -3 ∧ c.b = -4 ∧ c.r = 7 ∧ sumYIntersections c = -8 := by
  sorry


end NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l1053_105340


namespace NUMINAMATH_CALUDE_parallel_sides_in_pentagon_l1053_105364

-- Define the pentagon
structure Pentagon (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D E : V)

-- Define the parallelism relation
def Parallel (V : Type*) [AddCommGroup V] [Module ℝ V] (v w : V) : Prop :=
  ∃ (t : ℝ), v = t • w

-- State the theorem
theorem parallel_sides_in_pentagon
  (V : Type*) [AddCommGroup V] [Module ℝ V] (p : Pentagon V)
  (h1 : Parallel V (p.B - p.C) (p.A - p.D))
  (h2 : Parallel V (p.C - p.D) (p.B - p.E))
  (h3 : Parallel V (p.D - p.E) (p.A - p.C))
  (h4 : Parallel V (p.A - p.E) (p.B - p.D)) :
  Parallel V (p.A - p.B) (p.C - p.E) :=
sorry

end NUMINAMATH_CALUDE_parallel_sides_in_pentagon_l1053_105364


namespace NUMINAMATH_CALUDE_quadratic_points_order_l1053_105322

/-- The quadratic function f(x) = x² - 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

/-- Theorem: Given points A(-1, y₁), B(1, y₂), C(4, y₃) on the graph of f(x) = x² - 6x + c,
    prove that y₁ > y₂ > y₃ -/
theorem quadratic_points_order (c y₁ y₂ y₃ : ℝ) 
  (h₁ : f c (-1) = y₁)
  (h₂ : f c 1 = y₂)
  (h₃ : f c 4 = y₃) :
  y₁ > y₂ ∧ y₂ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_points_order_l1053_105322


namespace NUMINAMATH_CALUDE_building_cost_theorem_l1053_105330

/-- Calculates the total cost of all units in a building -/
def total_cost (total_units : ℕ) (cost_1bed : ℕ) (cost_2bed : ℕ) (num_2bed : ℕ) : ℕ :=
  let num_1bed := total_units - num_2bed
  num_1bed * cost_1bed + num_2bed * cost_2bed

/-- Theorem stating the total cost of all units in the given building configuration -/
theorem building_cost_theorem : 
  total_cost 12 360 450 7 = 4950 := by
  sorry

#eval total_cost 12 360 450 7

end NUMINAMATH_CALUDE_building_cost_theorem_l1053_105330


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_b_equals_one_l1053_105341

/-- 
Given a hyperbola with equation x²/4 - y²/b² = 1 where b > 0,
if the asymptotes are y = ±(1/2)x, then b = 1.
-/
theorem hyperbola_asymptote_implies_b_equals_one (b : ℝ) : 
  b > 0 → 
  (∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1 → 
    (y = (1/2) * x ∨ y = -(1/2) * x)) → 
  b = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_b_equals_one_l1053_105341


namespace NUMINAMATH_CALUDE_custom_operation_theorem_l1053_105365

/-- Custom operation ⊗ defined for real numbers -/
def otimes (a b : ℝ) : ℝ := 2 * a + b

/-- Theorem stating that if x ⊗ (-y) = 2 and (2y) ⊗ x = 1, then x + y = 1 -/
theorem custom_operation_theorem (x y : ℝ) 
  (h1 : otimes x (-y) = 2) 
  (h2 : otimes (2 * y) x = 1) : 
  x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_theorem_l1053_105365


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l1053_105318

theorem largest_prime_factor_of_expression : 
  let expression := 18^4 + 3 * 18^2 + 1 - 17^4
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ expression ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ expression → q ≤ p ∧ p = 307 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l1053_105318


namespace NUMINAMATH_CALUDE_water_boiling_time_l1053_105378

/-- Time for water to boil away given initial conditions -/
theorem water_boiling_time 
  (T₀ : ℝ) (Tₘ : ℝ) (t : ℝ) (c : ℝ) (L : ℝ)
  (h₁ : T₀ = 20)
  (h₂ : Tₘ = 100)
  (h₃ : t = 10)
  (h₄ : c = 4200)
  (h₅ : L = 2.3e6) :
  ∃ t₁ : ℝ, t₁ ≥ 67.5 ∧ t₁ < 68.5 :=
by sorry

end NUMINAMATH_CALUDE_water_boiling_time_l1053_105378


namespace NUMINAMATH_CALUDE_trip_distance_is_3_6_miles_l1053_105397

/-- Calculates the trip distance given the initial fee, charge per segment, and total charge -/
def calculate_trip_distance (initial_fee : ℚ) (charge_per_segment : ℚ) (segment_length : ℚ) (total_charge : ℚ) : ℚ :=
  let distance_charge := total_charge - initial_fee
  let num_segments := distance_charge / charge_per_segment
  num_segments * segment_length

/-- Proves that the trip distance is 3.6 miles given the specified conditions -/
theorem trip_distance_is_3_6_miles :
  let initial_fee : ℚ := 5/2
  let charge_per_segment : ℚ := 7/20
  let segment_length : ℚ := 2/5
  let total_charge : ℚ := 113/20
  calculate_trip_distance initial_fee charge_per_segment segment_length total_charge = 18/5 := by
  sorry

#eval (18 : ℚ) / 5

end NUMINAMATH_CALUDE_trip_distance_is_3_6_miles_l1053_105397


namespace NUMINAMATH_CALUDE_rectangle_similarity_symmetry_l1053_105376

/-- A rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Two rectangles are similar if their aspect ratios are equal -/
def similar (r1 r2 : Rectangle) : Prop :=
  r1.width / r1.height = r2.width / r2.height

/-- A rectangle can be formed from congruent copies of another rectangle -/
def can_form (r1 r2 : Rectangle) : Prop :=
  ∃ (m n p q : ℕ), m * r1.width + n * r1.height = r2.width ∧
                   p * r1.width + q * r1.height = r2.height

theorem rectangle_similarity_symmetry (A B : Rectangle) :
  (∃ C : Rectangle, similar C B ∧ can_form C A) →
  (∃ D : Rectangle, similar D A ∧ can_form D B) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_similarity_symmetry_l1053_105376


namespace NUMINAMATH_CALUDE_triangle_right_angled_l1053_105304

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if b*cos(B) + c*cos(C) = a*cos(A), then the triangle is right-angled. -/
theorem triangle_right_angled (A B C a b c : ℝ) : 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  b * Real.cos B + c * Real.cos C = a * Real.cos A →
  B = π/2 ∨ C = π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_right_angled_l1053_105304


namespace NUMINAMATH_CALUDE_x_value_l1053_105381

theorem x_value (x y : ℝ) : 
  (x = y * 0.9) → (y = 125 * 1.1) → x = 123.75 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1053_105381


namespace NUMINAMATH_CALUDE_no_roots_composition_l1053_105346

/-- A quadratic polynomial f(x) = ax^2 + bx + c -/
structure QuadraticPolynomial (α : Type*) [Ring α] where
  a : α
  b : α
  c : α

/-- The function represented by a quadratic polynomial -/
def evalQuadratic {α : Type*} [Ring α] (f : QuadraticPolynomial α) (x : α) : α :=
  f.a * x * x + f.b * x + f.c

theorem no_roots_composition {α : Type*} [LinearOrderedField α] (f : QuadraticPolynomial α) :
  (∀ x : α, evalQuadratic f x ≠ x) →
  (∀ x : α, evalQuadratic f (evalQuadratic f x) ≠ x) := by
  sorry

end NUMINAMATH_CALUDE_no_roots_composition_l1053_105346


namespace NUMINAMATH_CALUDE_min_distance_point_to_line_l1053_105363

theorem min_distance_point_to_line (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m * n - m - n = 3) :
  let d := |m + n| / Real.sqrt 2
  d ≥ 3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_distance_point_to_line_l1053_105363


namespace NUMINAMATH_CALUDE_remainder_of_1279_divided_by_89_l1053_105371

theorem remainder_of_1279_divided_by_89 : Nat.mod 1279 89 = 33 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1279_divided_by_89_l1053_105371


namespace NUMINAMATH_CALUDE_soccer_challenge_kicks_l1053_105398

/-- The number of penalty kicks needed for a soccer team challenge --/
def penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  (total_players - 1) * goalies

theorem soccer_challenge_kicks :
  penalty_kicks 25 5 = 120 :=
by sorry

end NUMINAMATH_CALUDE_soccer_challenge_kicks_l1053_105398


namespace NUMINAMATH_CALUDE_angle_between_hexagon_and_square_diagonal_l1053_105321

/-- A configuration with a square inside a regular hexagon sharing a common vertex. -/
structure SquareInHexagon where
  /-- The measure of an interior angle of the regular hexagon -/
  hexagon_angle : ℝ
  /-- The measure of an interior angle of the square -/
  square_angle : ℝ
  /-- The hexagon is regular -/
  hexagon_regular : hexagon_angle = 120
  /-- The square has right angles -/
  square_right : square_angle = 90

/-- The theorem stating that the angle between the hexagon side and square diagonal is 75° -/
theorem angle_between_hexagon_and_square_diagonal (config : SquareInHexagon) :
  config.hexagon_angle - (config.square_angle / 2) = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_hexagon_and_square_diagonal_l1053_105321


namespace NUMINAMATH_CALUDE_pencil_distribution_l1053_105395

theorem pencil_distribution (num_pens : ℕ) (num_pencils : ℕ) (num_students : ℕ) :
  num_pens = 1048 →
  num_students = 4 →
  num_pens % num_students = 0 →
  num_pencils % num_students = 0 →
  num_students = Nat.gcd num_pens num_pencils →
  num_pencils % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1053_105395


namespace NUMINAMATH_CALUDE_min_value_complex_sum_l1053_105360

theorem min_value_complex_sum (z : ℂ) (h : Complex.abs (z - (3 - 3*I)) = 4) :
  Complex.abs (z + (2 - I))^2 + Complex.abs (z - (6 - 5*I))^2 ≥ 76 :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_sum_l1053_105360


namespace NUMINAMATH_CALUDE_power_neg_square_cube_l1053_105362

theorem power_neg_square_cube (b : ℝ) : ((-b)^2)^3 = b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_neg_square_cube_l1053_105362


namespace NUMINAMATH_CALUDE_greatest_b_not_in_range_l1053_105334

/-- The quadratic function f(x) = x^2 + bx + 20 -/
def f (b : ℤ) (x : ℝ) : ℝ := x^2 + b*x + 20

/-- Predicate that checks if -9 is not in the range of f for a given b -/
def not_in_range (b : ℤ) : Prop := ∀ x : ℝ, f b x ≠ -9

/-- The theorem stating that 10 is the greatest integer b such that -9 is not in the range of f -/
theorem greatest_b_not_in_range : 
  (not_in_range 10 ∧ ∀ b : ℤ, b > 10 → ¬(not_in_range b)) := by sorry

end NUMINAMATH_CALUDE_greatest_b_not_in_range_l1053_105334


namespace NUMINAMATH_CALUDE_fraction_equality_l1053_105337

theorem fraction_equality (a b : ℝ) (h : 1/a - 1/b = 4) :
  (a - 2*a*b - b) / (2*a + 7*a*b - 2*b) = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1053_105337


namespace NUMINAMATH_CALUDE_right_triangle_in_segment_sets_l1053_105383

/-- Check if three line segments can form a right-angled triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The given sets of line segments -/
def segment_sets : List (ℝ × ℝ × ℝ) :=
  [(1, 2, 4), (3, 4, 5), (4, 6, 8), (5, 7, 11)]

theorem right_triangle_in_segment_sets :
  ∃! (a b c : ℝ), (a, b, c) ∈ segment_sets ∧ is_right_triangle a b c :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_in_segment_sets_l1053_105383


namespace NUMINAMATH_CALUDE_total_turkey_cost_l1053_105315

def turkey_cost (weight : ℕ) (price_per_kg : ℕ) : ℕ := weight * price_per_kg

theorem total_turkey_cost : 
  let first_turkey := 6
  let second_turkey := 9
  let third_turkey := 2 * second_turkey
  let price_per_kg := 2
  turkey_cost first_turkey price_per_kg + 
  turkey_cost second_turkey price_per_kg + 
  turkey_cost third_turkey price_per_kg = 66 := by
sorry

end NUMINAMATH_CALUDE_total_turkey_cost_l1053_105315


namespace NUMINAMATH_CALUDE_range_of_a_l1053_105333

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 4*x + 3 ≤ 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x < x - a}

-- Theorem statement
theorem range_of_a :
  (∀ x : ℝ, x ∈ B a → x ∈ A) ∧ 
  (∃ x : ℝ, x ∈ A ∧ x ∉ B a) →
  a ∈ Set.Icc 1 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1053_105333


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_triangle_l1053_105379

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers can form a triangle -/
def canFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- A function that checks if three numbers are consecutive odd primes -/
def areConsecutiveOddPrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧
  b = a + 2 ∧ c = b + 2 ∧
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1

/-- The main theorem stating that the smallest perimeter of a scalene triangle
    with consecutive odd prime side lengths and a prime perimeter is 23 -/
theorem smallest_prime_perimeter_triangle :
  ∀ a b c : ℕ,
  areConsecutiveOddPrimes a b c →
  canFormTriangle a b c →
  isPrime (a + b + c) →
  a + b + c ≥ 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_triangle_l1053_105379


namespace NUMINAMATH_CALUDE_hammer_weight_exceeds_ton_on_10th_day_l1053_105342

def hammer_weight (day : ℕ) : ℝ :=
  7 * (2 ^ (day - 1))

theorem hammer_weight_exceeds_ton_on_10th_day :
  (∀ d : ℕ, d < 10 → hammer_weight d ≤ 2000) ∧
  hammer_weight 10 > 2000 :=
by sorry

end NUMINAMATH_CALUDE_hammer_weight_exceeds_ton_on_10th_day_l1053_105342


namespace NUMINAMATH_CALUDE_room_population_theorem_l1053_105366

theorem room_population_theorem (total : ℕ) (under_21 : ℕ) (over_65 : ℕ) :
  (3 : ℚ) / 7 * total = under_21 →
  50 < total →
  total < 100 →
  under_21 = 24 →
  total = 56 ∧ (over_65 : ℚ) / total = over_65 / 56 := by
  sorry

end NUMINAMATH_CALUDE_room_population_theorem_l1053_105366


namespace NUMINAMATH_CALUDE_trigonometric_expressions_l1053_105320

theorem trigonometric_expressions (α : Real) (h : Real.tan α = 2) : 
  ((Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1/6) ∧ 
  (Real.sin α ^ 2 + Real.sin (2 * α) = 8/5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expressions_l1053_105320


namespace NUMINAMATH_CALUDE_math_evening_students_l1053_105301

theorem math_evening_students (total_rows : ℕ) 
  (h1 : 70 < total_rows * total_rows ∧ total_rows * total_rows < 90)
  (h2 : total_rows = (total_rows - 3) + 3)
  (h3 : 3 * total_rows < 90 ∧ 3 * total_rows > 70) :
  (total_rows * (total_rows - 3) = 54) ∧ (total_rows * 3 = 27) := by
  sorry

end NUMINAMATH_CALUDE_math_evening_students_l1053_105301


namespace NUMINAMATH_CALUDE_successive_projections_l1053_105399

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Projection of a point onto the xOy plane -/
def proj_xOy (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := 0 }

/-- Projection of a point onto the yOz plane -/
def proj_yOz (p : Point3D) : Point3D :=
  { x := 0, y := p.y, z := p.z }

/-- Projection of a point onto the xOz plane -/
def proj_xOz (p : Point3D) : Point3D :=
  { x := p.x, y := 0, z := p.z }

/-- The origin (0, 0, 0) -/
def origin : Point3D :=
  { x := 0, y := 0, z := 0 }

theorem successive_projections (M : Point3D) :
  proj_xOz (proj_yOz (proj_xOy M)) = origin := by
  sorry

end NUMINAMATH_CALUDE_successive_projections_l1053_105399


namespace NUMINAMATH_CALUDE_queen_mary_legs_l1053_105372

/-- The total number of legs on a ship with cats and humans -/
def total_legs (total_heads : ℕ) (cat_count : ℕ) (one_legged_human_count : ℕ) : ℕ :=
  let human_count := total_heads - cat_count
  let cat_legs := cat_count * 4
  let human_legs := (human_count - one_legged_human_count) * 2 + one_legged_human_count
  cat_legs + human_legs

/-- Theorem stating the total number of legs on the Queen Mary II -/
theorem queen_mary_legs : total_legs 16 5 1 = 41 := by
  sorry

end NUMINAMATH_CALUDE_queen_mary_legs_l1053_105372


namespace NUMINAMATH_CALUDE_number_of_routes_l1053_105369

/-- Recursive function representing the number of possible routes after n minutes -/
def M : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => M (n + 1) + M n

/-- The racing duration in minutes -/
def racingDuration : ℕ := 10

/-- Theorem stating that the number of possible routes after 10 minutes is 34 -/
theorem number_of_routes : M racingDuration = 34 := by
  sorry

end NUMINAMATH_CALUDE_number_of_routes_l1053_105369
