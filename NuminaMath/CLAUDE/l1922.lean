import Mathlib

namespace NUMINAMATH_CALUDE_square_of_binomial_with_sqrt_l1922_192234

theorem square_of_binomial_with_sqrt : 36^2 + 2 * 36 * Real.sqrt 49 + (Real.sqrt 49)^2 = 1849 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_with_sqrt_l1922_192234


namespace NUMINAMATH_CALUDE_unique_triplet_solution_l1922_192290

theorem unique_triplet_solution :
  ∀ x y p : ℕ+,
  p.Prime →
  (x.val * y.val^3 : ℚ) / (x.val + y.val) = p.val →
  x = 14 ∧ y = 2 ∧ p = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_triplet_solution_l1922_192290


namespace NUMINAMATH_CALUDE_spinner_probability_l1922_192204

theorem spinner_probability (p_A p_B p_C p_D p_E : ℚ) : 
  p_A = 1/5 →
  p_B = 1/10 →
  p_D = p_E →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_D = 7/20 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l1922_192204


namespace NUMINAMATH_CALUDE_spencer_total_distance_l1922_192285

/-- The total distance Spencer walked on Saturday -/
def total_distance (house_to_library library_to_post_office post_office_to_house : ℝ) : ℝ :=
  house_to_library + library_to_post_office + post_office_to_house

/-- Theorem stating that Spencer walked 0.8 mile in total -/
theorem spencer_total_distance :
  total_distance 0.3 0.1 0.4 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_spencer_total_distance_l1922_192285


namespace NUMINAMATH_CALUDE_certain_number_is_eleven_l1922_192269

theorem certain_number_is_eleven : ∃ x : ℕ, 
  x + (3 * 13 + 3 * 14 + 3 * 17) = 143 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_eleven_l1922_192269


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1922_192243

theorem min_value_of_expression : 
  ∃ (min : ℝ), min = Real.sqrt 2 * Real.sqrt 5 ∧ 
  ∀ (x : ℝ), Real.sqrt (x^2 + (1 + 2*x)^2) + Real.sqrt ((x - 1)^2 + (x - 1)^2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1922_192243


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l1922_192202

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l1922_192202


namespace NUMINAMATH_CALUDE_circular_table_theorem_l1922_192274

/-- Represents the setup of people sitting around a circular table -/
structure CircularTable where
  num_men : ℕ
  num_women : ℕ

/-- Defines when a man is considered satisfied -/
def is_satisfied (t : CircularTable) : Prop :=
  ∃ (i j : ℕ), i ≠ j ∧ i < t.num_men + t.num_women ∧ j < t.num_men + t.num_women

/-- The probability of a specific man being satisfied -/
def prob_man_satisfied (t : CircularTable) : ℚ :=
  25 / 33

/-- The expected number of satisfied men -/
def expected_satisfied_men (t : CircularTable) : ℚ :=
  1250 / 33

/-- Main theorem statement -/
theorem circular_table_theorem (t : CircularTable) 
  (h1 : t.num_men = 50) (h2 : t.num_women = 50) : 
  prob_man_satisfied t = 25 / 33 ∧ 
  expected_satisfied_men t = 1250 / 33 := by
  sorry

#check circular_table_theorem

end NUMINAMATH_CALUDE_circular_table_theorem_l1922_192274


namespace NUMINAMATH_CALUDE_smallest_n_with_triple_sum_l1922_192267

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Proposition: The smallest positive integer N whose sum of digits is three times 
    the sum of digits of N+1 has a sum of digits equal to 12 -/
theorem smallest_n_with_triple_sum : 
  ∃ (N : ℕ), N > 0 ∧ 
  sum_of_digits N = 3 * sum_of_digits (N + 1) ∧
  sum_of_digits N = 12 ∧
  ∀ (M : ℕ), M > 0 → sum_of_digits M = 3 * sum_of_digits (M + 1) → 
    sum_of_digits M ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_triple_sum_l1922_192267


namespace NUMINAMATH_CALUDE_reappearance_is_lcm_reappearance_is_twenty_l1922_192233

/-- The cycle length of the letter sequence -/
def letter_cycle_length : ℕ := 5

/-- The cycle length of the digit sequence -/
def digit_cycle_length : ℕ := 4

/-- The line number where the original sequences reappear together -/
def reappearance_line : ℕ := 20

/-- Theorem stating that the reappearance line is the LCM of the cycle lengths -/
theorem reappearance_is_lcm :
  reappearance_line = Nat.lcm letter_cycle_length digit_cycle_length := by
  sorry

/-- Theorem stating that the reappearance line is 20 -/
theorem reappearance_is_twenty : reappearance_line = 20 := by
  sorry

end NUMINAMATH_CALUDE_reappearance_is_lcm_reappearance_is_twenty_l1922_192233


namespace NUMINAMATH_CALUDE_deepak_investment_l1922_192214

/-- Proves that Deepak's investment is 15000 given the conditions of the business problem -/
theorem deepak_investment (total_profit : ℝ) (anand_investment : ℝ) (deepak_profit : ℝ) 
  (h1 : total_profit = 13800)
  (h2 : anand_investment = 22500)
  (h3 : deepak_profit = 5400) :
  ∃ deepak_investment : ℝ, 
    deepak_investment = 15000 ∧ 
    deepak_profit / total_profit = deepak_investment / (anand_investment + deepak_investment) :=
by
  sorry


end NUMINAMATH_CALUDE_deepak_investment_l1922_192214


namespace NUMINAMATH_CALUDE_oak_trees_after_planting_l1922_192241

/-- The number of oak trees in the park after planting -/
def total_oak_trees (initial : ℕ) (new : ℕ) : ℕ :=
  initial + new

/-- Theorem: The total number of oak trees after planting is 11 -/
theorem oak_trees_after_planting :
  total_oak_trees 9 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_after_planting_l1922_192241


namespace NUMINAMATH_CALUDE_second_player_can_always_win_l1922_192235

/-- Represents a square on the game board -/
inductive Square
| Empty : Square
| S : Square
| O : Square

/-- Represents the game board -/
def Board := Vector Square 2000

/-- Represents a player in the game -/
inductive Player
| First : Player
| Second : Player

/-- Checks if the game is over (SOS pattern found) -/
def is_game_over (board : Board) : Prop := sorry

/-- Represents a valid move in the game -/
structure Move where
  position : Fin 2000
  symbol : Square

/-- Applies a move to the board -/
def apply_move (board : Board) (move : Move) : Board := sorry

/-- Represents the game state -/
structure GameState where
  board : Board
  current_player : Player

/-- Represents a strategy for a player -/
def Strategy := GameState → Move

/-- Checks if a strategy is winning for a player -/
def is_winning_strategy (player : Player) (strategy : Strategy) : Prop := sorry

/-- The main theorem to prove -/
theorem second_player_can_always_win :
  ∃ (strategy : Strategy), is_winning_strategy Player.Second strategy := sorry

end NUMINAMATH_CALUDE_second_player_can_always_win_l1922_192235


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l1922_192282

-- Define the ellipse equation
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (m - 1) + y^2 / m = 1

-- Theorem statement
theorem ellipse_focal_length (m : ℝ) (h1 : m > m - 1) 
  (h2 : ∀ x y : ℝ, ellipse_equation x y m) : 
  ∃ (a b c : ℝ), a^2 = m ∧ b^2 = m - 1 ∧ c^2 = 1 ∧ 2 * c = 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l1922_192282


namespace NUMINAMATH_CALUDE_distance_per_interval_l1922_192297

-- Define the total distance walked
def total_distance : ℝ := 3

-- Define the total time taken
def total_time : ℝ := 45

-- Define the interval time
def interval_time : ℝ := 15

-- Theorem to prove
theorem distance_per_interval : 
  (total_distance / (total_time / interval_time)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_per_interval_l1922_192297


namespace NUMINAMATH_CALUDE_sum_of_possible_x_minus_y_values_l1922_192236

theorem sum_of_possible_x_minus_y_values (x y : ℝ) 
  (eq1 : x^2 - x*y + x = 2018)
  (eq2 : y^2 - x*y - y = 52) : 
  ∃ (z₁ z₂ : ℝ), (z₁ = x - y ∨ z₂ = x - y) ∧ z₁ + z₂ = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_possible_x_minus_y_values_l1922_192236


namespace NUMINAMATH_CALUDE_original_solution_concentration_l1922_192248

/-- Represents a chemical solution with a certain concentration --/
structure ChemicalSolution :=
  (concentration : ℝ)

/-- Represents a mixture of two chemical solutions --/
def mix (s1 s2 : ChemicalSolution) (ratio : ℝ) : ChemicalSolution :=
  { concentration := ratio * s1.concentration + (1 - ratio) * s2.concentration }

/-- Theorem: If half of an original solution is replaced with a 60% solution,
    resulting in a 55% solution, then the original solution was 50% --/
theorem original_solution_concentration
  (original replacement result : ChemicalSolution)
  (h1 : replacement.concentration = 0.6)
  (h2 : result = mix original replacement 0.5)
  (h3 : result.concentration = 0.55) :
  original.concentration = 0.5 := by
  sorry

#check original_solution_concentration

end NUMINAMATH_CALUDE_original_solution_concentration_l1922_192248


namespace NUMINAMATH_CALUDE_shipment_total_correct_l1922_192296

/-- The total number of novels in the shipment -/
def total_novels : ℕ := 300

/-- The percentage of novels displayed in the window -/
def display_percentage : ℚ := 30 / 100

/-- The number of novels left in the stockroom -/
def stockroom_novels : ℕ := 210

/-- Theorem stating that the total number of novels is correct given the conditions -/
theorem shipment_total_correct :
  (1 - display_percentage) * total_novels = stockroom_novels := by
  sorry

end NUMINAMATH_CALUDE_shipment_total_correct_l1922_192296


namespace NUMINAMATH_CALUDE_integer_solution_divisibility_l1922_192286

theorem integer_solution_divisibility (a b c : ℕ) : 
  1 < a ∧ a < b ∧ b < c ∧ ((a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1) →
  ((a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solution_divisibility_l1922_192286


namespace NUMINAMATH_CALUDE_holly_chocolate_milk_l1922_192284

/-- Holly's chocolate milk consumption throughout the day -/
def chocolate_milk_problem (initial_consumption breakfast_consumption lunch_consumption dinner_consumption new_container_size : ℕ) : Prop :=
  let remaining_milk := new_container_size - (lunch_consumption + dinner_consumption)
  remaining_milk = 48

/-- Theorem stating Holly ends the day with 48 ounces of chocolate milk -/
theorem holly_chocolate_milk :
  chocolate_milk_problem 8 8 8 8 64 := by
  sorry

end NUMINAMATH_CALUDE_holly_chocolate_milk_l1922_192284


namespace NUMINAMATH_CALUDE_puppies_sold_l1922_192258

/-- Given a pet store scenario, prove the number of puppies sold. -/
theorem puppies_sold (initial_puppies : ℕ) (puppies_per_cage : ℕ) (cages_used : ℕ) :
  initial_puppies ≥ puppies_per_cage * cages_used →
  initial_puppies - (puppies_per_cage * cages_used) =
    initial_puppies - puppies_per_cage * cages_used :=
by
  sorry

#check puppies_sold 102 9 9

end NUMINAMATH_CALUDE_puppies_sold_l1922_192258


namespace NUMINAMATH_CALUDE_consecutive_digit_sums_l1922_192219

def S (n : ℕ) : ℕ := sorry  -- Sum of digits function

theorem consecutive_digit_sums 
  (N : ℕ) 
  (h1 : S N + S (N + 1) = 200)
  (h2 : S (N + 2) + S (N + 3) = 105) :
  S (N + 1) + S (N + 2) = 202 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_digit_sums_l1922_192219


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1922_192291

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Theorem statement
theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 4)^2 - 34 * (a 4) + 64 = 0 →
  (a 8)^2 - 34 * (a 8) + 64 = 0 →
  a 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1922_192291


namespace NUMINAMATH_CALUDE_janes_trail_mix_nuts_percentage_l1922_192218

theorem janes_trail_mix_nuts_percentage :
  -- Sue's trail mix composition
  let sue_nuts_percent : ℚ := 30
  let sue_fruit_percent : ℚ := 70
  
  -- Jane's trail mix composition
  let jane_choc_percent : ℚ := 40
  
  -- Combined mixture composition
  let combined_nuts_percent : ℚ := 45
  let combined_fruit_percent : ℚ := 35
  
  -- Equal contribution assumption
  let equal_contribution : Prop := True
  
  -- Jane's nuts percentage to be determined
  ∃ jane_nuts_percent : ℚ,
    -- Jane's trail mix percentages sum to 100%
    jane_nuts_percent + jane_choc_percent + (60 - jane_nuts_percent) = 100 ∧
    
    -- Combined nuts percentage is average of Sue's and Jane's
    (sue_nuts_percent + jane_nuts_percent) / 2 = combined_nuts_percent ∧
    
    -- Combined fruit percentage is average of Sue's and Jane's
    (sue_fruit_percent + (60 - jane_nuts_percent)) / 2 = combined_fruit_percent ∧
    
    -- Jane's nuts percentage is 60%
    jane_nuts_percent = 60 := by
  sorry

end NUMINAMATH_CALUDE_janes_trail_mix_nuts_percentage_l1922_192218


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1922_192230

def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1922_192230


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l1922_192226

-- Define the number in base 5
def base_5_number : Nat := 200220220

-- Define the function to convert from base 5 to base 10
def base_5_to_10 (n : Nat) : Nat :=
  let digits := n.digits 5
  (List.range digits.length).foldl (fun acc i => acc + digits[i]! * (5^i)) 0

-- Define the number in base 10
def number : Nat := base_5_to_10 base_5_number

-- Statement to prove
theorem largest_prime_divisor :
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ number ∧ ∀ (q : Nat), Nat.Prime q → q ∣ number → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l1922_192226


namespace NUMINAMATH_CALUDE_triangle_side_length_l1922_192222

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a = 3 →
  C = 2 * π / 3 →  -- 120° in radians
  S = (15 * Real.sqrt 3) / 4 →
  S = (1 / 2) * a * b * Real.sin C →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l1922_192222


namespace NUMINAMATH_CALUDE_even_multiple_six_sum_properties_l1922_192256

theorem even_multiple_six_sum_properties (a b : ℤ) 
  (h_a_even : Even a) (h_b_multiple_six : ∃ k, b = 6 * k) :
  Even (a + b) ∧ 
  (∃ m, a + b = 3 * m) ∧ 
  ¬(∀ (a b : ℤ), Even a → (∃ k, b = 6 * k) → ∃ n, a + b = 6 * n) ∧
  ∃ (a b : ℤ), Even a ∧ (∃ k, b = 6 * k) ∧ (∃ n, a + b = 6 * n) :=
by sorry

end NUMINAMATH_CALUDE_even_multiple_six_sum_properties_l1922_192256


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1922_192252

-- Theorem 1
theorem simplify_expression_1 (a : ℝ) : a^2 - 2*a - 3*a^2 + 4*a = -2*a^2 + 2*a := by
  sorry

-- Theorem 2
theorem simplify_expression_2 (x : ℝ) : 4*(x^2 - 2) - 2*(2*x^2 + 3*x + 3) + 7*x = x - 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1922_192252


namespace NUMINAMATH_CALUDE_min_shift_for_odd_cosine_l1922_192253

/-- Given a function f(x) = cos(2x + π/6) that is shifted right by φ units,
    prove that the minimum positive φ that makes the resulting function odd is π/3. -/
theorem min_shift_for_odd_cosine :
  let f (x : ℝ) := Real.cos (2 * x + π / 6)
  let g (φ : ℝ) (x : ℝ) := f (x - φ)
  ∀ φ : ℝ, φ > 0 →
    (∀ x : ℝ, g φ (-x) = -(g φ x)) →
    φ ≥ π / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_shift_for_odd_cosine_l1922_192253


namespace NUMINAMATH_CALUDE_blister_slowdown_proof_l1922_192262

/-- Represents the speed reduction caused by each blister -/
def blister_slowdown : ℝ := 10

theorem blister_slowdown_proof :
  let old_speed : ℝ := 6
  let new_speed : ℝ := 11
  let hike_duration : ℝ := 4
  let blister_interval : ℝ := 2
  let num_blisters : ℝ := hike_duration / blister_interval
  old_speed * hike_duration = 
    new_speed * blister_interval + 
    (new_speed - num_blisters * blister_slowdown) * blister_interval →
  blister_slowdown = 10 := by
sorry

end NUMINAMATH_CALUDE_blister_slowdown_proof_l1922_192262


namespace NUMINAMATH_CALUDE_base_6_addition_l1922_192207

/-- Converts a base-6 number to base-10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- Converts a base-10 number to base-6 --/
def to_base_6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

theorem base_6_addition :
  to_base_6 (to_base_10 [4, 2, 5, 3] + to_base_10 [2, 4, 4, 2]) = [0, 1, 4, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_6_addition_l1922_192207


namespace NUMINAMATH_CALUDE_remainder_problem_l1922_192206

theorem remainder_problem (N : ℤ) : 
  N % 19 = 7 → N % 20 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1922_192206


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l1922_192266

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 3) * circle_radius
  rectangle_length * rectangle_breadth

theorem rectangle_area_proof (h1 : square_area = 4761) (h2 : rectangle_breadth = 13) :
  rectangle_area square_area rectangle_breadth = 598 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l1922_192266


namespace NUMINAMATH_CALUDE_system_solution_l1922_192257

/-- The system of differential equations -/
def system (t x y : ℝ) : Prop :=
  ∃ (dt dx dy : ℝ), dt / (4*y - 5*x) = dx / (5*t - 3*y) ∧ dx / (5*t - 3*y) = dy / (3*x - 4*t)

/-- The general solution of the system -/
def solution (t x y : ℝ) : Prop :=
  ∃ (C₁ C₂ : ℝ), 3*t + 4*x + 5*y = C₁ ∧ t^2 + x^2 + y^2 = C₂

/-- Theorem stating that the solution satisfies the system -/
theorem system_solution :
  ∀ (t x y : ℝ), system t x y → solution t x y :=
sorry

end NUMINAMATH_CALUDE_system_solution_l1922_192257


namespace NUMINAMATH_CALUDE_greg_additional_rotations_l1922_192213

/-- Represents the number of wheel rotations per block on flat ground. -/
def flatRotations : ℕ := 200

/-- Represents the number of wheel rotations per block uphill. -/
def uphillRotations : ℕ := 250

/-- Represents the number of blocks Greg has already ridden on flat ground. -/
def flatBlocksRidden : ℕ := 2

/-- Represents the number of blocks Greg has already ridden uphill. -/
def uphillBlocksRidden : ℕ := 1

/-- Represents the total number of wheel rotations Greg has already completed. -/
def rotationsCompleted : ℕ := 600

/-- Represents the number of additional uphill blocks Greg plans to ride. -/
def additionalUphillBlocks : ℕ := 3

/-- Represents the number of additional flat blocks Greg plans to ride. -/
def additionalFlatBlocks : ℕ := 2

/-- Represents the minimum total number of blocks Greg wants to ride. -/
def minTotalBlocks : ℕ := 8

/-- Theorem stating that Greg needs 550 more wheel rotations to reach his goal. -/
theorem greg_additional_rotations :
  let totalPlannedBlocks := flatBlocksRidden + uphillBlocksRidden + additionalFlatBlocks + additionalUphillBlocks
  let totalPlannedRotations := flatBlocksRidden * flatRotations + uphillBlocksRidden * uphillRotations +
                               additionalFlatBlocks * flatRotations + additionalUphillBlocks * uphillRotations
  totalPlannedBlocks ≥ minTotalBlocks ∧
  totalPlannedRotations - rotationsCompleted = 550 := by
  sorry


end NUMINAMATH_CALUDE_greg_additional_rotations_l1922_192213


namespace NUMINAMATH_CALUDE_square_difference_equality_l1922_192263

theorem square_difference_equality : (36 + 12)^2 - (12^2 + 36^2 + 24) = 840 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1922_192263


namespace NUMINAMATH_CALUDE_admission_score_calculation_l1922_192264

theorem admission_score_calculation (total_applicants : ℕ) 
  (admitted_ratio : ℚ) 
  (admitted_avg_diff : ℝ) 
  (not_admitted_avg_diff : ℝ) 
  (total_avg_score : ℝ) 
  (h1 : admitted_ratio = 1 / 4)
  (h2 : admitted_avg_diff = 10)
  (h3 : not_admitted_avg_diff = -26)
  (h4 : total_avg_score = 70) :
  ∃ (admission_score : ℝ),
    admission_score = 87 ∧
    (admitted_ratio * (admission_score + admitted_avg_diff) + 
     (1 - admitted_ratio) * (admission_score + not_admitted_avg_diff) = total_avg_score) := by
  sorry

end NUMINAMATH_CALUDE_admission_score_calculation_l1922_192264


namespace NUMINAMATH_CALUDE_liam_chocolate_consumption_l1922_192295

/-- Given that Liam ate a total of 150 chocolates in five days, and each day after
    the first day he ate 8 more chocolates than the previous day, prove that
    he ate 38 chocolates on the fourth day. -/
theorem liam_chocolate_consumption :
  ∀ (x : ℕ),
  (x + (x + 8) + (x + 16) + (x + 24) + (x + 32) = 150) →
  (x + 24 = 38) :=
by sorry

end NUMINAMATH_CALUDE_liam_chocolate_consumption_l1922_192295


namespace NUMINAMATH_CALUDE_calculate_expression_l1922_192276

theorem calculate_expression : (3^1006 + 7^1007)^2 - (3^1006 - 7^1007)^2 = 42 * 10^1007 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1922_192276


namespace NUMINAMATH_CALUDE_points_four_units_from_negative_two_l1922_192259

theorem points_four_units_from_negative_two (x : ℝ) : 
  (|x - (-2)| = 4) ↔ (x = 2 ∨ x = -6) := by sorry

end NUMINAMATH_CALUDE_points_four_units_from_negative_two_l1922_192259


namespace NUMINAMATH_CALUDE_area_YPW_is_8_l1922_192201

/-- Represents a rectangle XYZW with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a point P that divides the diagonal XW of a rectangle -/
structure DiagonalPoint where
  ratio_XP : ℝ
  ratio_PW : ℝ

/-- Calculates the area of triangle YPW in the given rectangle with the given diagonal point -/
def area_YPW (rect : Rectangle) (p : DiagonalPoint) : ℝ :=
  sorry

/-- Theorem stating that for a rectangle with length 8 and width 6, 
    if P divides XW in ratio 2:1, then area of YPW is 8 -/
theorem area_YPW_is_8 (rect : Rectangle) (p : DiagonalPoint) :
  rect.length = 8 →
  rect.width = 6 →
  p.ratio_XP = 2 →
  p.ratio_PW = 1 →
  area_YPW rect p = 8 := by
  sorry

end NUMINAMATH_CALUDE_area_YPW_is_8_l1922_192201


namespace NUMINAMATH_CALUDE_jar_balls_count_l1922_192225

theorem jar_balls_count (initial_blue : ℕ) (removed : ℕ) (prob : ℚ) :
  initial_blue = 6 →
  removed = 3 →
  prob = 1/5 →
  (initial_blue - removed : ℚ) / ((initial_blue - removed : ℚ) + (18 - initial_blue : ℚ)) = prob →
  18 = initial_blue + (18 - initial_blue) :=
by sorry

end NUMINAMATH_CALUDE_jar_balls_count_l1922_192225


namespace NUMINAMATH_CALUDE_total_students_at_concert_l1922_192261

/-- The number of buses used for the concert. -/
def num_buses : ℕ := 8

/-- The number of students each bus can carry. -/
def students_per_bus : ℕ := 45

/-- Theorem: The total number of students who went to the concert is 360. -/
theorem total_students_at_concert : num_buses * students_per_bus = 360 := by
  sorry

end NUMINAMATH_CALUDE_total_students_at_concert_l1922_192261


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1922_192270

theorem complex_magnitude_problem (m : ℝ) : 
  (Complex.I * ((1 + m * Complex.I) * (3 + Complex.I))).re = 0 →
  Complex.abs ((m + 3 * Complex.I) / (1 - Complex.I)) = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1922_192270


namespace NUMINAMATH_CALUDE_class_size_proof_l1922_192228

theorem class_size_proof (boys_ratio : Nat) (girls_ratio : Nat) (num_girls : Nat) :
  boys_ratio = 5 →
  girls_ratio = 8 →
  num_girls = 160 →
  (boys_ratio + girls_ratio : Rat) * (num_girls / girls_ratio : Rat) = 260 :=
by sorry

end NUMINAMATH_CALUDE_class_size_proof_l1922_192228


namespace NUMINAMATH_CALUDE_simplify_expressions_l1922_192260

variable (x y : ℝ)

theorem simplify_expressions :
  (3 * x^2 - 2*x*y + y^2 - 3*x^2 + 3*x*y = x*y + y^2) ∧
  ((7*x^2 - 3*x*y) - 6*(x^2 - 1/3*x*y) = x^2 - x*y) := by sorry

end NUMINAMATH_CALUDE_simplify_expressions_l1922_192260


namespace NUMINAMATH_CALUDE_lunks_needed_for_twenty_apples_l1922_192217

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks (l : ℚ) : ℚ := (4/7) * l

/-- Exchange rate between kunks and apples -/
def kunks_to_apples (k : ℚ) : ℚ := (5/3) * k

/-- The number of lunks needed to purchase a given number of apples -/
def lunks_for_apples (a : ℚ) : ℚ := 
  let kunks := (3/5) * a
  (7/4) * kunks

theorem lunks_needed_for_twenty_apples : 
  lunks_for_apples 20 = 21 := by sorry

end NUMINAMATH_CALUDE_lunks_needed_for_twenty_apples_l1922_192217


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1922_192244

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = 2 → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1922_192244


namespace NUMINAMATH_CALUDE_circle_area_squared_gt_ngon_areas_product_l1922_192203

/-- Given a circle and two regular n-gons (one inscribed, one circumscribed),
    prove that the square of the circle's area is greater than
    the product of the areas of the inscribed and circumscribed n-gons. -/
theorem circle_area_squared_gt_ngon_areas_product
  (n : ℕ) (S S₁ S₂ : ℝ) (hn : n ≥ 3) (hS : S > 0) (hS₁ : S₁ > 0) (hS₂ : S₂ > 0)
  (h_inscribed : S₁ = n / 2 * S * Real.sin (2 * Real.pi / n))
  (h_circumscribed : S₂ = n / 2 * S * Real.tan (Real.pi / n)) :
  S^2 > S₁ * S₂ := by
sorry

end NUMINAMATH_CALUDE_circle_area_squared_gt_ngon_areas_product_l1922_192203


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1922_192231

/-- Given a line L1 with equation 2x - 5y + 3 = 0 and a point P(2, -1),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation 5x + 2y - 8 = 0 -/
theorem perpendicular_line_equation 
  (L1 : Set (ℝ × ℝ)) 
  (P : ℝ × ℝ) :
  (L1 = {(x, y) | 2 * x - 5 * y + 3 = 0}) →
  (P = (2, -1)) →
  (∃ L2 : Set (ℝ × ℝ), 
    (P ∈ L2) ∧ 
    (∀ (Q R : ℝ × ℝ), Q ∈ L1 → R ∈ L1 → Q ≠ R → 
      ∀ (S T : ℝ × ℝ), S ∈ L2 → T ∈ L2 → S ≠ T →
        ((Q.1 - R.1) * (S.1 - T.1) + (Q.2 - R.2) * (S.2 - T.2) = 0)) ∧
    (L2 = {(x, y) | 5 * x + 2 * y - 8 = 0})) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1922_192231


namespace NUMINAMATH_CALUDE_course_selection_theorem_l1922_192232

def physical_education_courses : ℕ := 4
def art_courses : ℕ := 4
def total_courses : ℕ := physical_education_courses + art_courses

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def two_course_selections : ℕ := (choose physical_education_courses 1) * (choose art_courses 1)

def three_course_selections : ℕ := 
  (choose physical_education_courses 2) * (choose art_courses 1) +
  (choose physical_education_courses 1) * (choose art_courses 2)

def total_selections : ℕ := two_course_selections + three_course_selections

theorem course_selection_theorem : total_selections = 64 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l1922_192232


namespace NUMINAMATH_CALUDE_vitamin_boxes_count_l1922_192221

/-- Given the total number of medicine boxes and the number of supplement boxes,
    prove that the number of vitamin boxes is 472. -/
theorem vitamin_boxes_count (total_medicine : ℕ) (supplements : ℕ) 
    (h1 : total_medicine = 760)
    (h2 : supplements = 288)
    (h3 : ∃ vitamins : ℕ, total_medicine = vitamins + supplements) :
  ∃ vitamins : ℕ, vitamins = 472 ∧ total_medicine = vitamins + supplements :=
by
  sorry

end NUMINAMATH_CALUDE_vitamin_boxes_count_l1922_192221


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1922_192251

theorem quadratic_roots_condition (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) →
  c < (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1922_192251


namespace NUMINAMATH_CALUDE_max_true_statements_l1922_192255

theorem max_true_statements (a b : ℝ) : 
  ¬(∃ (s1 s2 s3 s4 : Bool), 
    (s1 → 1/a > 1/b) ∧
    (s2 → abs a > abs b) ∧
    (s3 → a > b) ∧
    (s4 → a < 0) ∧
    (¬s1 ∨ ¬s2 ∨ ¬s3 ∨ ¬s4 → b > 0) ∧
    s1 ∧ s2 ∧ s3 ∧ s4) :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l1922_192255


namespace NUMINAMATH_CALUDE_square_area_with_two_edge_representations_l1922_192209

theorem square_area_with_two_edge_representations (x : ℝ) :
  (3 * x - 12 = 18 - 2 * x) →
  (3 * x - 12)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_two_edge_representations_l1922_192209


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l1922_192200

theorem factorization_of_quadratic (a : ℝ) : a^2 + 2*a = a*(a + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l1922_192200


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l1922_192247

/-- A right trapezoid with specific properties -/
structure RightTrapezoid where
  /-- The length of one lateral side -/
  side1 : ℝ
  /-- The length of the other lateral side -/
  side2 : ℝ
  /-- The diagonal bisects the acute angle -/
  diagonal_bisects_acute_angle : Bool

/-- The area of the right trapezoid -/
def area (t : RightTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific right trapezoid is 104 -/
theorem specific_trapezoid_area :
  ∀ (t : RightTrapezoid),
    t.side1 = 10 ∧
    t.side2 = 8 ∧
    t.diagonal_bisects_acute_angle = true →
    area t = 104 :=
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l1922_192247


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_correct_l1922_192215

/-- The coefficient of x^2 in the binomial expansion of (x^2/2 - 1/√x)^6 -/
def coefficient_x_squared : ℚ :=
  let expression := (fun x => x^2/2 - 1/Real.sqrt x)^6
  -- We don't actually compute the coefficient here, just define it
  15/4

/-- Theorem stating that the coefficient of x^2 in the binomial expansion of (x^2/2 - 1/√x)^6 is 15/4 -/
theorem coefficient_x_squared_is_correct :
  coefficient_x_squared = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_is_correct_l1922_192215


namespace NUMINAMATH_CALUDE_coke_cost_l1922_192205

def cheeseburger_cost : ℚ := 3.65
def milkshake_cost : ℚ := 2
def fries_cost : ℚ := 4
def cookie_cost : ℚ := 0.5
def tax : ℚ := 0.2
def toby_initial : ℚ := 15
def toby_change : ℚ := 7

theorem coke_cost (coke_price : ℚ) : coke_price = 1 := by
  sorry

end NUMINAMATH_CALUDE_coke_cost_l1922_192205


namespace NUMINAMATH_CALUDE_fraction_simplification_l1922_192280

theorem fraction_simplification : 
  let numerator := (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400)
  let denominator := (6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400)
  ∀ x : ℕ, x^4 + 400 = (x^2 - 10*x + 20) * (x^2 + 10*x + 20) →
  numerator / denominator = 995 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1922_192280


namespace NUMINAMATH_CALUDE_pi_power_zero_plus_two_power_neg_two_l1922_192216

theorem pi_power_zero_plus_two_power_neg_two :
  (-Real.pi)^(0 : ℤ) + 2^(-2 : ℤ) = 5/4 := by sorry

end NUMINAMATH_CALUDE_pi_power_zero_plus_two_power_neg_two_l1922_192216


namespace NUMINAMATH_CALUDE_danny_bottle_caps_wrappers_l1922_192242

theorem danny_bottle_caps_wrappers : 
  let bottle_caps_found : ℕ := 50
  let wrappers_found : ℕ := 46
  bottle_caps_found - wrappers_found = 4 := by
  sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_wrappers_l1922_192242


namespace NUMINAMATH_CALUDE_sixtieth_term_of_arithmetic_sequence_l1922_192279

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence as a function from ℕ to ℚ
  d : ℚ      -- The common difference
  h : ∀ n, a (n + 1) = a n + d

/-- Given an arithmetic sequence with a₁ = 7 and a₁₅ = 37,
    prove that a₆₀ = 134.5 -/
theorem sixtieth_term_of_arithmetic_sequence
  (seq : ArithmeticSequence)
  (h1 : seq.a 1 = 7)
  (h15 : seq.a 15 = 37) :
  seq.a 60 = 134.5 := by
  sorry

end NUMINAMATH_CALUDE_sixtieth_term_of_arithmetic_sequence_l1922_192279


namespace NUMINAMATH_CALUDE_solution_set_f_geq_12_range_of_a_l1922_192278

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 5| + |x + 4|

-- Theorem for the solution set of f(x) ≥ 12
theorem solution_set_f_geq_12 :
  {x : ℝ | f x ≥ 12} = {x : ℝ | x ≥ 13/2 ∨ x ≤ -11/2} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x - 2^(1 - 3*a) - 1 ≥ 0} = {a : ℝ | a ≥ -2/3} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_12_range_of_a_l1922_192278


namespace NUMINAMATH_CALUDE_vector_operation_result_l1922_192298

def a : ℝ × ℝ × ℝ := (2, 0, 1)
def b : ℝ × ℝ × ℝ := (-3, 1, -1)
def c : ℝ × ℝ × ℝ := (1, 1, 0)

theorem vector_operation_result :
  a + 2 • b - 3 • c = (-7, -1, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_result_l1922_192298


namespace NUMINAMATH_CALUDE_total_staff_is_250_l1922_192283

/-- Represents a hospital with doctors and nurses -/
structure Hospital where
  doctors : ℕ
  nurses : ℕ

/-- The total number of staff (doctors and nurses) in a hospital -/
def Hospital.total (h : Hospital) : ℕ := h.doctors + h.nurses

/-- A hospital satisfying the given conditions -/
def special_hospital : Hospital :=
  { doctors := 100,  -- This is derived from the ratio, not given directly
    nurses := 150 }

theorem total_staff_is_250 :
  (special_hospital.doctors : ℚ) / special_hospital.nurses = 2 / 3 ∧
  special_hospital.nurses = 150 →
  special_hospital.total = 250 := by
  sorry

end NUMINAMATH_CALUDE_total_staff_is_250_l1922_192283


namespace NUMINAMATH_CALUDE_min_socks_for_pairs_l1922_192210

/-- Represents the number of colors of socks in the drawer -/
def num_colors : ℕ := 4

/-- Represents the number of pairs we want to guarantee -/
def required_pairs : ℕ := 10

/-- Theorem: The minimum number of socks to guarantee the required pairs -/
theorem min_socks_for_pairs :
  ∀ (sock_counts : Fin num_colors → ℕ),
  (∀ i, sock_counts i > 0) →
  ∃ (n : ℕ),
    n = num_colors + 2 * required_pairs ∧
    ∀ (m : ℕ), m ≥ n →
      ∀ (selection : Fin m → Fin num_colors),
      ∃ (pairs : Fin required_pairs → Fin m × Fin m),
        ∀ i, 
          (pairs i).1 < (pairs i).2 ∧
          selection (pairs i).1 = selection (pairs i).2 ∧
          ∀ j, i ≠ j → 
            ({(pairs i).1, (pairs i).2} : Set (Fin m)) ∩ {(pairs j).1, (pairs j).2} = ∅ :=
by
  sorry

end NUMINAMATH_CALUDE_min_socks_for_pairs_l1922_192210


namespace NUMINAMATH_CALUDE_subtracted_number_l1922_192272

theorem subtracted_number (m n x : ℕ) : 
  m > 0 → n > 0 → m = 15 * n - x → m % 5 = 4 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l1922_192272


namespace NUMINAMATH_CALUDE_combination_simplification_l1922_192254

theorem combination_simplification (n : ℕ) : 
  (n.choose (n - 2)) + (n.choose 3) + ((n + 1).choose 2) = ((n + 2).choose 3) := by
  sorry

end NUMINAMATH_CALUDE_combination_simplification_l1922_192254


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l1922_192268

theorem purely_imaginary_condition (a : ℝ) : 
  (Complex.I : ℂ).im ≠ 0 → 
  (Complex.re ((1 - a * Complex.I) * (3 + 2 * Complex.I)) = 0 ∧
   Complex.im ((1 - a * Complex.I) * (3 + 2 * Complex.I)) ≠ 0) → 
  a = -3/2 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l1922_192268


namespace NUMINAMATH_CALUDE_count_valid_pairs_l1922_192223

def is_valid_pair (x y : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 23 ∧ 1 ≤ y ∧ y ≤ 23 ∧ (x^2 + y^2 + x + y) % 6 = 0

theorem count_valid_pairs :
  ∃ (S : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ S ↔ is_valid_pair p.1 p.2) ∧ S.card = 225 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l1922_192223


namespace NUMINAMATH_CALUDE_cos_five_pi_sixths_l1922_192237

theorem cos_five_pi_sixths : Real.cos (5 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_five_pi_sixths_l1922_192237


namespace NUMINAMATH_CALUDE_nth_prime_upper_bound_l1922_192249

def nth_prime (n : ℕ) : ℕ := sorry

theorem nth_prime_upper_bound (n : ℕ) : nth_prime n ≤ 2^(2^(n-1)) := by sorry

end NUMINAMATH_CALUDE_nth_prime_upper_bound_l1922_192249


namespace NUMINAMATH_CALUDE_second_sum_is_1720_l1922_192246

/-- Given a total sum of 2795 rupees divided into two parts, where the interest on the first part
    for 8 years at 3% per annum equals the interest on the second part for 3 years at 5% per annum,
    prove that the second part is equal to 1720 rupees. -/
theorem second_sum_is_1720 (total : ℝ) (first_part second_part : ℝ) : 
  total = 2795 →
  first_part + second_part = total →
  (first_part * 3 * 8) / 100 = (second_part * 5 * 3) / 100 →
  second_part = 1720 := by
  sorry

end NUMINAMATH_CALUDE_second_sum_is_1720_l1922_192246


namespace NUMINAMATH_CALUDE_parabola_slope_l1922_192227

/-- The slope of line MF for a parabola y² = 2px with point M(3, m) at distance 4 from focus -/
theorem parabola_slope (p m : ℝ) : p > 0 → m > 0 → m^2 = 6*p → (3 + p/2)^2 + m^2 = 16 → 
  (m / (3 - p/2) : ℝ) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_slope_l1922_192227


namespace NUMINAMATH_CALUDE_tank_fill_time_l1922_192289

/-- The time it takes to fill a tank with two pipes and a leak -/
theorem tank_fill_time (pipe1_time pipe2_time : ℝ) (leak_fraction : ℝ) : 
  pipe1_time = 20 →
  pipe2_time = 30 →
  leak_fraction = 1/3 →
  (1 / ((1 / pipe1_time + 1 / pipe2_time) * (1 - leak_fraction))) = 18 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l1922_192289


namespace NUMINAMATH_CALUDE_system_1_solution_system_2_solution_l1922_192294

-- System 1
theorem system_1_solution :
  ∃ (x y : ℝ), x - y = 3 ∧ x = 3 * y - 1 ∧ x = 5 ∧ y = 2 := by
  sorry

-- System 2
theorem system_2_solution :
  ∃ (x y : ℝ), 2 * x + 3 * y = -1 ∧ 3 * x - 2 * y = 18 ∧ x = 4 ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_system_1_solution_system_2_solution_l1922_192294


namespace NUMINAMATH_CALUDE_logarithm_calculation_l1922_192220

theorem logarithm_calculation : 
  (Real.log 3 / Real.log (1/9) - (-8)^(2/3)) * (0.125^(1/3)) = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_calculation_l1922_192220


namespace NUMINAMATH_CALUDE_min_squared_distance_to_origin_l1922_192224

/-- The minimum squared distance from a point on the line 2x + y + 5 = 0 to the origin is 5 -/
theorem min_squared_distance_to_origin : 
  ∀ x y : ℝ, 2 * x + y + 5 = 0 → x^2 + y^2 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_squared_distance_to_origin_l1922_192224


namespace NUMINAMATH_CALUDE_y_gets_20_percent_more_than_z_l1922_192288

/-- The problem setup with given conditions -/
def problem_setup (x y z : ℝ) : Prop :=
  x = y * 1.25 ∧  -- x gets 25% more than y
  740 = x + y + z ∧  -- total amount is 740
  z = 200  -- z's share is 200

/-- The theorem to prove -/
theorem y_gets_20_percent_more_than_z 
  (x y z : ℝ) (h : problem_setup x y z) : y = z * 1.2 := by
  sorry


end NUMINAMATH_CALUDE_y_gets_20_percent_more_than_z_l1922_192288


namespace NUMINAMATH_CALUDE_tan_120_degrees_l1922_192299

theorem tan_120_degrees : Real.tan (2 * Real.pi / 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_120_degrees_l1922_192299


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1922_192277

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |1 - x| > 1} = Set.Ioi 2 ∪ Set.Iio 0 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1922_192277


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1922_192281

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  1/a + 2/b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1922_192281


namespace NUMINAMATH_CALUDE_furniture_dealer_profit_l1922_192287

/-- Calculates the gross profit for a furniture dealer selling a desk -/
theorem furniture_dealer_profit
  (purchase_price : ℝ)
  (markup_percentage : ℝ)
  (discount_percentage : ℝ)
  (h1 : purchase_price = 150)
  (h2 : markup_percentage = 0.5)
  (h3 : discount_percentage = 0.2) :
  let selling_price := purchase_price / (1 - markup_percentage)
  let discounted_price := selling_price * (1 - discount_percentage)
  let gross_profit := discounted_price - purchase_price
  gross_profit = 90 := by sorry

end NUMINAMATH_CALUDE_furniture_dealer_profit_l1922_192287


namespace NUMINAMATH_CALUDE_system_solution_l1922_192245

theorem system_solution (m n : ℝ) : 
  (m * 2 + n * 4 = 8 ∧ 2 * m * 2 - 3 * n * 4 = -4) → m = 2 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1922_192245


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1922_192240

theorem quadratic_equal_roots (a b : ℝ) (h : b^2 = 4*a) :
  (a * b^2) / (a^2 - 4*a + b^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1922_192240


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l1922_192238

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - 2*x + 5

/-- The axis of symmetry -/
def axis_of_symmetry : ℝ := 1

/-- Theorem: The axis of symmetry of the parabola y = x^2 - 2x + 5 is the line x = 1 -/
theorem parabola_axis_of_symmetry :
  ∀ x : ℝ, parabola (axis_of_symmetry + x) = parabola (axis_of_symmetry - x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l1922_192238


namespace NUMINAMATH_CALUDE_a_101_mod_49_l1922_192212

/-- Definition of the sequence a_n -/
def a (n : ℕ) : ℕ := 5^n + 9^n

/-- Theorem stating that a_101 is congruent to 0 modulo 49 -/
theorem a_101_mod_49 : a 101 ≡ 0 [ZMOD 49] := by
  sorry

end NUMINAMATH_CALUDE_a_101_mod_49_l1922_192212


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1922_192273

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → a ∈ Set.Icc (-1) 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1922_192273


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l1922_192211

/-- Given a line segment with midpoint (3, -3) and one endpoint (7, 4),
    prove that the other endpoint is (-1, -10). -/
theorem line_segment_endpoint
  (midpoint : ℝ × ℝ)
  (endpoint1 : ℝ × ℝ)
  (h_midpoint : midpoint = (3, -3))
  (h_endpoint1 : endpoint1 = (7, 4)) :
  ∃ (endpoint2 : ℝ × ℝ),
    endpoint2 = (-1, -10) ∧
    midpoint = (
      (endpoint1.1 + endpoint2.1) / 2,
      (endpoint1.2 + endpoint2.2) / 2
    ) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l1922_192211


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l1922_192229

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 3) : 
  3 * π * r^2 = 2 * π * r^2 + π * r^2 := by
sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l1922_192229


namespace NUMINAMATH_CALUDE_house_construction_fraction_l1922_192293

theorem house_construction_fraction (total : ℕ) (additional : ℕ) (remaining : ℕ) 
  (h_total : total = 2000)
  (h_additional : additional = 300)
  (h_remaining : remaining = 500) :
  (total - additional - remaining : ℚ) / total = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_house_construction_fraction_l1922_192293


namespace NUMINAMATH_CALUDE_unique_number_l1922_192275

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ 
  n / 100000 = 1 ∧
  (n % 100000) * 10 + 1 = 3 * n

theorem unique_number : ∃! n : ℕ, is_valid_number n :=
  sorry

end NUMINAMATH_CALUDE_unique_number_l1922_192275


namespace NUMINAMATH_CALUDE_sandwiches_sold_out_l1922_192265

theorem sandwiches_sold_out (original : ℕ) (available : ℕ) (h1 : original = 9) (h2 : available = 4) :
  original - available = 5 := by
sorry

end NUMINAMATH_CALUDE_sandwiches_sold_out_l1922_192265


namespace NUMINAMATH_CALUDE_twins_age_product_difference_l1922_192250

theorem twins_age_product_difference (current_age : ℕ) : 
  current_age = 2 → (current_age + 1)^2 - current_age^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_twins_age_product_difference_l1922_192250


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l1922_192239

/-- The line equation kx - y - 2k = 0 -/
def line (k : ℝ) (x y : ℝ) : Prop := k * x - y - 2 * k = 0

/-- The hyperbola equation x^2 - y^2 = 2 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

/-- The asymptotes of the hyperbola -/
def asymptotes (x y : ℝ) : Prop := y = x ∨ y = -x

/-- The theorem stating that if the line and hyperbola have only one common point, then k = 1 or k = -1 -/
theorem line_hyperbola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, line k p.1 p.2 ∧ hyperbola p.1 p.2) → 
  k = 1 ∨ k = -1 :=
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l1922_192239


namespace NUMINAMATH_CALUDE_jessica_attended_two_games_l1922_192292

/-- The number of soccer games Jessica attended -/
def games_attended (total_games missed_games : ℕ) : ℕ :=
  total_games - missed_games

/-- Proof that Jessica attended 2 games -/
theorem jessica_attended_two_games :
  games_attended 6 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_attended_two_games_l1922_192292


namespace NUMINAMATH_CALUDE_max_value_sum_of_reciprocals_l1922_192208

theorem max_value_sum_of_reciprocals (a b : ℝ) (h : a + b = 2) :
  (∃ (x : ℝ), ∀ (y : ℝ), (1 / (a^2 + 1) + 1 / (b^2 + 1)) ≤ y) ∧
  (∀ (ε : ℝ), ε > 0 → ∃ (a' b' : ℝ), a' + b' = 2 ∧ 
    (1 / (a'^2 + 1) + 1 / (b'^2 + 1)) > (Real.sqrt 2 + 1) / 2 - ε) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_reciprocals_l1922_192208


namespace NUMINAMATH_CALUDE_tan_sum_of_quadratic_roots_l1922_192271

theorem tan_sum_of_quadratic_roots (α β : Real) (h : ∀ x, x^2 + 6*x + 7 = 0 ↔ x = Real.tan α ∨ x = Real.tan β) :
  Real.tan (α + β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_of_quadratic_roots_l1922_192271
