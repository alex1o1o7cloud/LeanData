import Mathlib

namespace NUMINAMATH_CALUDE_number_manipulation_l486_48654

theorem number_manipulation (x : ℚ) (h : (7/8) * x = 28) : (x + 16) * (5/16) = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_manipulation_l486_48654


namespace NUMINAMATH_CALUDE_negation_of_existential_l486_48691

theorem negation_of_existential (p : Prop) :
  (¬ ∃ (x : ℝ), x^2 > 1) ↔ (∀ (x : ℝ), x^2 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_l486_48691


namespace NUMINAMATH_CALUDE_egg_roll_count_l486_48662

/-- The number of egg rolls Omar rolled -/
def omar_rolls : ℕ := 219

/-- The number of egg rolls Karen rolled -/
def karen_rolls : ℕ := 229

/-- The total number of egg rolls rolled by Omar and Karen -/
def total_rolls : ℕ := omar_rolls + karen_rolls

theorem egg_roll_count : total_rolls = 448 := by sorry

end NUMINAMATH_CALUDE_egg_roll_count_l486_48662


namespace NUMINAMATH_CALUDE_undefined_fraction_l486_48631

theorem undefined_fraction (x : ℝ) : x = 1 → ¬∃y : ℝ, y = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_undefined_fraction_l486_48631


namespace NUMINAMATH_CALUDE_product_sum_equality_l486_48609

theorem product_sum_equality (x y z : ℕ) 
  (h1 : 2014 + y = 2015 + x)
  (h2 : 2015 + x = 2016 + z)
  (h3 : y * x * z = 504) :
  y * x + x * z = 128 := by
sorry

end NUMINAMATH_CALUDE_product_sum_equality_l486_48609


namespace NUMINAMATH_CALUDE_jean_trips_l486_48615

theorem jean_trips (total : ℕ) (extra : ℕ) (h1 : total = 40) (h2 : extra = 6) :
  ∃ (bill : ℕ) (jean : ℕ), bill + jean = total ∧ jean = bill + extra ∧ jean = 23 :=
by sorry

end NUMINAMATH_CALUDE_jean_trips_l486_48615


namespace NUMINAMATH_CALUDE_infinitely_many_amiable_squares_l486_48642

/-- A number is amiable if the set {1,2,...,N} can be partitioned into pairs
    of elements, each pair having the sum of its elements a perfect square. -/
def IsAmiable (N : ℕ) : Prop :=
  ∃ (partition : List (ℕ × ℕ)),
    (∀ (pair : ℕ × ℕ), pair ∈ partition → pair.1 ≤ N ∧ pair.2 ≤ N) ∧
    (∀ n : ℕ, n ≤ N → ∃ (pair : ℕ × ℕ), pair ∈ partition ∧ (n = pair.1 ∨ n = pair.2)) ∧
    (∀ (pair : ℕ × ℕ), pair ∈ partition → ∃ m : ℕ, pair.1 + pair.2 = m^2)

/-- There exist infinitely many amiable numbers which are themselves perfect squares. -/
theorem infinitely_many_amiable_squares :
  ∀ k : ℕ, ∃ N : ℕ, N > k ∧ ∃ m : ℕ, N = m^2 ∧ IsAmiable N :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_amiable_squares_l486_48642


namespace NUMINAMATH_CALUDE_rice_distribution_l486_48650

theorem rice_distribution (total_pounds : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_pounds = 35 / 2 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_pounds * ounces_per_pound) / num_containers = 70 := by
  sorry

end NUMINAMATH_CALUDE_rice_distribution_l486_48650


namespace NUMINAMATH_CALUDE_expression_evaluation_l486_48606

theorem expression_evaluation : 
  let x : ℝ := -3
  (5 + x * (4 + x) - 4^2 + (x^2 - 3*x + 2)) / (x^2 - 4 + x - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l486_48606


namespace NUMINAMATH_CALUDE_undefined_fraction_l486_48600

theorem undefined_fraction (a : ℝ) : 
  ¬ (∃ x : ℝ, x = (a + 3) / (a^2 - 9)) ↔ a = -3 ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_undefined_fraction_l486_48600


namespace NUMINAMATH_CALUDE_game_lives_per_player_l486_48671

theorem game_lives_per_player (initial_players : ℕ) (new_players : ℕ) (total_lives : ℕ) :
  initial_players = 7 →
  new_players = 2 →
  total_lives = 63 →
  (total_lives / (initial_players + new_players) : ℚ) = 7 :=
by sorry

end NUMINAMATH_CALUDE_game_lives_per_player_l486_48671


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l486_48623

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2)*x - k + 4 > 0) ↔ k > -2*Real.sqrt 3 ∧ k < 2*Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l486_48623


namespace NUMINAMATH_CALUDE_fixed_point_on_quadratic_graph_l486_48616

/-- The fixed point on the graph of y = 9x^2 + mx + 3m for all real m -/
theorem fixed_point_on_quadratic_graph :
  ∀ (m : ℝ), 9 * (-3)^2 + m * (-3) + 3 * m = 81 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_quadratic_graph_l486_48616


namespace NUMINAMATH_CALUDE_complex_absolute_value_product_l486_48655

theorem complex_absolute_value_product : 
  Complex.abs ((7 - 4 * Complex.I) * (5 + 12 * Complex.I)) = 13 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_product_l486_48655


namespace NUMINAMATH_CALUDE_tangential_quadrilateral_l486_48667

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A circle in the plane -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- A quadrilateral in the plane -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Predicate to check if a quadrilateral is cyclic -/
def is_cyclic (q : Quadrilateral) : Prop := sorry

/-- Predicate to check if a quadrilateral is tangential -/
def is_tangential (q : Quadrilateral) : Prop := sorry

/-- Get the incenter of a triangle -/
def incenter (A B C : Point) : Point := sorry

/-- Check if four points are concyclic -/
def are_concyclic (A B C D : Point) : Prop := sorry

/-- Main theorem -/
theorem tangential_quadrilateral 
  (q : Quadrilateral) 
  (h1 : is_cyclic q) 
  (h2 : let I := incenter q.A q.B q.C
        let J := incenter q.A q.D q.C
        are_concyclic q.B I J q.D) : 
  is_tangential q :=
sorry

end NUMINAMATH_CALUDE_tangential_quadrilateral_l486_48667


namespace NUMINAMATH_CALUDE_binary_multiplication_division_l486_48601

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryNumber := List Nat

/-- Converts a decimal number to its binary representation -/
def toBinary (n : Nat) : BinaryNumber :=
  sorry

/-- Converts a binary number to its decimal representation -/
def toDecimal (b : BinaryNumber) : Nat :=
  sorry

/-- Multiplies two binary numbers -/
def binaryMultiply (a b : BinaryNumber) : BinaryNumber :=
  sorry

/-- Divides a binary number by 2 -/
def binaryDivideByTwo (b : BinaryNumber) : BinaryNumber :=
  sorry

theorem binary_multiplication_division :
  let a := [1, 0, 1, 1, 0]  -- 10110₂
  let b := [1, 0, 1, 0, 0]  -- 10100₂
  let result := [1, 1, 0, 1, 1, 1, 0, 0]  -- 11011100₂
  binaryDivideByTwo (binaryMultiply a b) = result :=
sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_l486_48601


namespace NUMINAMATH_CALUDE_rally_ticket_cost_l486_48617

/-- The cost of tickets bought at the door at a rally --/
def ticket_cost_at_door (total_attendance : ℕ) (pre_rally_ticket_cost : ℚ) 
  (total_receipts : ℚ) (pre_rally_tickets : ℕ) : ℚ :=
  (total_receipts - pre_rally_ticket_cost * pre_rally_tickets) / (total_attendance - pre_rally_tickets)

/-- Theorem stating the cost of tickets bought at the door --/
theorem rally_ticket_cost : 
  ticket_cost_at_door 750 2 (1706.25) 475 = (2.75 : ℚ) := by sorry

end NUMINAMATH_CALUDE_rally_ticket_cost_l486_48617


namespace NUMINAMATH_CALUDE_szilveszter_age_l486_48661

def birth_year (a b : ℕ) := 1900 + 10 * a + b

def grandfather_birth_year (a b : ℕ) := 1910 + a + b

def current_year := 1999

theorem szilveszter_age (a b : ℕ) 
  (h1 : a < 10 ∧ b < 10) 
  (h2 : 1 + 9 + a + b = current_year - grandfather_birth_year a b) 
  (h3 : 10 * a + b = current_year - grandfather_birth_year a b) :
  current_year - birth_year a b = 23 := by
sorry

end NUMINAMATH_CALUDE_szilveszter_age_l486_48661


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_l486_48633

theorem absolute_value_sum_zero (x y : ℝ) :
  |x - 2| + |4 - y| = 0 → x = 2 ∧ y = 4 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_l486_48633


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l486_48621

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l486_48621


namespace NUMINAMATH_CALUDE_rook_placements_on_chessboard_l486_48696

/-- The number of ways to place n rooks on an n×n chessboard so that no two rooks 
    are in the same row or column -/
def valid_rook_placements (n : ℕ) : ℕ := Nat.factorial n

/-- The size of the chessboard -/
def board_size : ℕ := 8

theorem rook_placements_on_chessboard : 
  valid_rook_placements board_size = 40320 := by
  sorry

#eval valid_rook_placements board_size

end NUMINAMATH_CALUDE_rook_placements_on_chessboard_l486_48696


namespace NUMINAMATH_CALUDE_parabola_transformation_l486_48663

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := x^2 - 1

/-- The transformed parabola function -/
def transformed_parabola (x : ℝ) : ℝ := (x - 1)^2 + 1

/-- Theorem stating that the transformation of the original parabola
    by shifting 2 units up and 1 unit right results in the transformed parabola -/
theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 1) + 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l486_48663


namespace NUMINAMATH_CALUDE_group_b_sample_size_l486_48676

/-- Represents the number of cities in each group and the total sample size -/
structure CityGroups where
  total : Nat
  groupA : Nat
  groupB : Nat
  groupC : Nat
  sampleSize : Nat

/-- Calculates the number of cities to be selected from a specific group in stratified sampling -/
def stratifiedSampleSize (cg : CityGroups) (groupSize : Nat) : Nat :=
  (groupSize * cg.sampleSize) / cg.total

/-- Theorem stating that for the given city groups, the stratified sample size for Group B is 3 -/
theorem group_b_sample_size (cg : CityGroups) 
  (h1 : cg.total = 24)
  (h2 : cg.groupA = 4)
  (h3 : cg.groupB = 12)
  (h4 : cg.groupC = 8)
  (h5 : cg.sampleSize = 6)
  : stratifiedSampleSize cg cg.groupB = 3 := by
  sorry

end NUMINAMATH_CALUDE_group_b_sample_size_l486_48676


namespace NUMINAMATH_CALUDE_probability_consonant_initials_l486_48649

/-- The probability of selecting a student with consonant initials -/
theorem probability_consonant_initials :
  let total_letters : ℕ := 26
  let vowels : Finset Char := {'A', 'E', 'I', 'O', 'U', 'Y'}
  let consonants : ℕ := total_letters - vowels.card
  consonants / total_letters = 10 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_consonant_initials_l486_48649


namespace NUMINAMATH_CALUDE_distribute_six_tasks_three_people_l486_48637

/-- The number of ways to distribute tasks among people -/
def distribute_tasks (num_tasks : ℕ) (num_people : ℕ) : ℕ :=
  num_people^num_tasks - num_people * (num_people - 1)^num_tasks + num_people

/-- Theorem stating the correct number of ways to distribute 6 tasks among 3 people -/
theorem distribute_six_tasks_three_people :
  distribute_tasks 6 3 = 540 := by
  sorry


end NUMINAMATH_CALUDE_distribute_six_tasks_three_people_l486_48637


namespace NUMINAMATH_CALUDE_rotation_90_degrees_l486_48685

theorem rotation_90_degrees (z : ℂ) : z = -4 - I → z * I = 1 - 4*I := by
  sorry

end NUMINAMATH_CALUDE_rotation_90_degrees_l486_48685


namespace NUMINAMATH_CALUDE_fixed_points_of_f_l486_48690

def f (x : ℝ) := x^2 - 2*x - 10

theorem fixed_points_of_f :
  ∀ x : ℝ, f x = x ↔ x = -2 ∨ x = 5 := by sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_l486_48690


namespace NUMINAMATH_CALUDE_nickel_probability_l486_48641

/-- Represents the types of coins in the jar -/
inductive Coin
| Dime
| Nickel
| Penny

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
| Coin.Dime => 10
| Coin.Nickel => 5
| Coin.Penny => 1

/-- The total value of each coin type in cents -/
def total_value : Coin → ℕ
| Coin.Dime => 500
| Coin.Nickel => 300
| Coin.Penny => 200

/-- The number of coins of each type -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the jar -/
def total_coins : ℕ := coin_count Coin.Dime + coin_count Coin.Nickel + coin_count Coin.Penny

/-- The probability of randomly choosing a nickel from the jar -/
theorem nickel_probability : 
  (coin_count Coin.Nickel : ℚ) / total_coins = 6 / 31 := by sorry

end NUMINAMATH_CALUDE_nickel_probability_l486_48641


namespace NUMINAMATH_CALUDE_third_circle_radius_l486_48625

theorem third_circle_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 21) (h₂ : r₂ = 35) (h₃ : r₃ = 28) :
  π * r₃^2 = π * r₂^2 - π * r₁^2 := by
  sorry

#check third_circle_radius

end NUMINAMATH_CALUDE_third_circle_radius_l486_48625


namespace NUMINAMATH_CALUDE_M_intersect_N_l486_48651

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem M_intersect_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_l486_48651


namespace NUMINAMATH_CALUDE_million_to_scientific_notation_two_point_684_million_scientific_notation_l486_48645

theorem million_to_scientific_notation (n : ℝ) : 
  n * 1000000 = n * (10 : ℝ) ^ 6 := by sorry

-- Define 2.684 million
def two_point_684_million : ℝ := 2.684 * 1000000

-- Theorem to prove
theorem two_point_684_million_scientific_notation : 
  two_point_684_million = 2.684 * (10 : ℝ) ^ 6 := by sorry

end NUMINAMATH_CALUDE_million_to_scientific_notation_two_point_684_million_scientific_notation_l486_48645


namespace NUMINAMATH_CALUDE_megan_candy_from_sister_l486_48673

/-- Calculates the number of candy pieces Megan received from her older sister. -/
def candy_from_sister (candy_from_neighbors : ℝ) (candy_eaten_per_day : ℝ) (days_lasted : ℝ) : ℝ :=
  candy_eaten_per_day * days_lasted - candy_from_neighbors

/-- Proves that Megan received 5.0 pieces of candy from her older sister. -/
theorem megan_candy_from_sister :
  candy_from_sister 11.0 8.0 2.0 = 5.0 := by
  sorry

end NUMINAMATH_CALUDE_megan_candy_from_sister_l486_48673


namespace NUMINAMATH_CALUDE_range_of_a_l486_48620

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, a^x > 1 ↔ x < 0

def q (a : ℝ) : Prop := ∀ x : ℝ, ∃ y : ℝ, y = Real.log (a*x^2 - x + 2)

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, 
    (a > 0) → 
    (a ≠ 1) → 
    ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → 
    ((0 < a ∧ a ≤ 1/8) ∨ (a ≥ 1)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l486_48620


namespace NUMINAMATH_CALUDE_motion_analysis_l486_48627

-- Define the motion law
def s (t : ℝ) : ℝ := 4 * t + t^3

-- Define velocity as the derivative of s
noncomputable def v (t : ℝ) : ℝ := deriv s t

-- Define acceleration as the derivative of v
noncomputable def a (t : ℝ) : ℝ := deriv v t

-- Theorem statement
theorem motion_analysis :
  (∀ t, v t = 4 + 3 * t^2) ∧
  (∀ t, a t = 6 * t) ∧
  (v 0 = 4 ∧ a 0 = 0) ∧
  (v 1 = 7 ∧ a 1 = 6) ∧
  (v 2 = 16 ∧ a 2 = 12) := by sorry

end NUMINAMATH_CALUDE_motion_analysis_l486_48627


namespace NUMINAMATH_CALUDE_exist_non_congruent_equal_perimeter_area_cyclic_quadrilaterals_l486_48677

/-- A cyclic quadrilateral is a quadrilateral that can be inscribed in a circle. -/
structure CyclicQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_cyclic : ∃ (center : ℝ × ℝ) (radius : ℝ), ∀ i, dist center (vertices i) = radius

/-- The perimeter of a quadrilateral. -/
def perimeter (q : CyclicQuadrilateral) : ℝ :=
  (dist (q.vertices 0) (q.vertices 1)) +
  (dist (q.vertices 1) (q.vertices 2)) +
  (dist (q.vertices 2) (q.vertices 3)) +
  (dist (q.vertices 3) (q.vertices 0))

/-- The area of a quadrilateral. -/
def area (q : CyclicQuadrilateral) : ℝ := sorry

/-- Two quadrilaterals are congruent if there exists a rigid transformation that maps one to the other. -/
def congruent (q1 q2 : CyclicQuadrilateral) : Prop := sorry

theorem exist_non_congruent_equal_perimeter_area_cyclic_quadrilaterals :
  ∃ (q1 q2 : CyclicQuadrilateral),
    perimeter q1 = perimeter q2 ∧
    area q1 = area q2 ∧
    ¬congruent q1 q2 := by
  sorry

end NUMINAMATH_CALUDE_exist_non_congruent_equal_perimeter_area_cyclic_quadrilaterals_l486_48677


namespace NUMINAMATH_CALUDE_expected_twos_l486_48643

/-- The probability of rolling a 2 on a standard die -/
def prob_two : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 2's when rolling three standard dice -/
theorem expected_twos : 
  num_dice * prob_two = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_expected_twos_l486_48643


namespace NUMINAMATH_CALUDE_canoe_upstream_speed_l486_48648

/-- The speed of a canoe rowing upstream, given its downstream speed and the stream speed -/
theorem canoe_upstream_speed (downstream_speed stream_speed : ℝ) :
  downstream_speed = 12 →
  stream_speed = 4 →
  downstream_speed - 2 * stream_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_canoe_upstream_speed_l486_48648


namespace NUMINAMATH_CALUDE_joan_egg_count_l486_48603

-- Define the number of dozens Joan bought
def dozen_count : ℕ := 6

-- Define the number of eggs in a dozen
def eggs_per_dozen : ℕ := 12

-- Theorem to prove
theorem joan_egg_count : dozen_count * eggs_per_dozen = 72 := by
  sorry

end NUMINAMATH_CALUDE_joan_egg_count_l486_48603


namespace NUMINAMATH_CALUDE_robins_hair_length_l486_48656

/-- Given Robin's initial hair length and the length he cut off, calculate his final hair length -/
theorem robins_hair_length (initial_length cut_length : ℕ) 
  (h1 : initial_length = 14)
  (h2 : cut_length = 13) :
  initial_length - cut_length = 1 := by
  sorry

end NUMINAMATH_CALUDE_robins_hair_length_l486_48656


namespace NUMINAMATH_CALUDE_min_cost_verification_l486_48678

/-- Represents a set of weights -/
def WeightSet := List Nat

/-- Cost of using a weight once -/
def weighing_cost : Nat := 100

/-- The range of possible diamond masses -/
def diamond_range : Set Nat := Finset.range 15

/-- Checks if a set of weights can measure all masses in the given range -/
def can_measure_all (weights : WeightSet) (range : Set Nat) : Prop :=
  ∀ n ∈ range, ∃ subset : List Nat, subset.toFinset ⊆ weights.toFinset ∧ subset.sum = n

/-- Calculates the minimum number of weighings needed for a given set of weights -/
def min_weighings (weights : WeightSet) : Nat :=
  weights.length + 1

/-- Calculates the total cost for a given number of weighings -/
def total_cost (num_weighings : Nat) : Nat :=
  num_weighings * weighing_cost

/-- The optimal set of weights for measuring masses from 1 to 15 -/
def optimal_weights : WeightSet := [1, 2, 4, 8]

theorem min_cost_verification :
  can_measure_all optimal_weights diamond_range →
  total_cost (min_weighings optimal_weights) = 800 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_verification_l486_48678


namespace NUMINAMATH_CALUDE_number_satisfies_equation_l486_48636

theorem number_satisfies_equation : ∃ x : ℝ, (45 - 3 * x^2 = 12) ∧ (x = Real.sqrt 11 ∨ x = -Real.sqrt 11) := by
  sorry

end NUMINAMATH_CALUDE_number_satisfies_equation_l486_48636


namespace NUMINAMATH_CALUDE_pet_store_cages_l486_48635

def bird_cages (total_birds : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ) : ℕ :=
  total_birds / (parrots_per_cage + parakeets_per_cage)

theorem pet_store_cages :
  bird_cages 36 2 2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cages_l486_48635


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l486_48669

/-- Proves that given a loan of 1000 for 5 years, where the interest amount is 750 less than the sum lent, the interest rate per annum is 5% -/
theorem interest_rate_calculation (sum_lent : ℝ) (time_period : ℝ) (interest_amount : ℝ) 
  (h1 : sum_lent = 1000)
  (h2 : time_period = 5)
  (h3 : interest_amount = sum_lent - 750) :
  (interest_amount * 100) / (sum_lent * time_period) = 5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l486_48669


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_l486_48607

def A (m : ℝ) : Set ℝ := {m - 1, -3}
def B (m : ℝ) : Set ℝ := {2*m - 1, m - 3}

theorem intersection_implies_m_value :
  ∀ m : ℝ, A m ∩ B m = {-3} → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_l486_48607


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l486_48698

/-- Given a real number a and a function f with the specified properties,
    prove that the tangent line to f at the origin has the equation 3x + y = 0 -/
theorem tangent_line_at_origin (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + (a-3)*x
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 + 2*a*x + (a-3)
  (∀ x, f' x = f' (-x)) →  -- f' is an even function
  (∃ k : ℝ, ∀ x y, y = f x → (y - f 0) = k * (x - 0) → 3*x + y = 0) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l486_48698


namespace NUMINAMATH_CALUDE_no_integer_solutions_l486_48618

/-- The system of equations has no integer solutions -/
theorem no_integer_solutions : ¬ ∃ (x y z : ℤ), 
  (x^2 - 2*x*y + y^2 - z^2 = 17) ∧ 
  (-x^2 + 3*y*z + 3*z^2 = 27) ∧ 
  (x^2 - x*y + 5*z^2 = 50) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l486_48618


namespace NUMINAMATH_CALUDE_greatest_integer_with_prime_absolute_value_l486_48665

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_integer_with_prime_absolute_value :
  ∀ x : ℤ, (is_prime (Int.natAbs (8 * x^2 - 66 * x + 21))) →
    x ≤ 2 ∧ is_prime (Int.natAbs (8 * 2^2 - 66 * 2 + 21)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_prime_absolute_value_l486_48665


namespace NUMINAMATH_CALUDE_race_speed_ratio_l486_48640

theorem race_speed_ratio (va vb : ℝ) (h : va > 0 ∧ vb > 0) :
  (1 / va = (1 - 0.09523809523809523) / vb) → (va / vb = 21 / 19) := by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l486_48640


namespace NUMINAMATH_CALUDE_circle_center_line_max_ab_l486_48686

theorem circle_center_line_max_ab (a b : ℝ) :
  let circle := (fun (x y : ℝ) => x^2 + y^2 + 2*x - 4*y + 1 = 0)
  let center_line := (fun (x y : ℝ) => a*x - b*y + 1 = 0)
  let center := (-1, 2)
  (∀ x y, circle x y ↔ (x + 1)^2 + (y - 2)^2 = 4) →
  center_line (-1) 2 →
  (∀ k, k * a * b ≤ 1/8) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_line_max_ab_l486_48686


namespace NUMINAMATH_CALUDE_cookies_left_in_scenario_l486_48683

/-- Represents the cookie-making scenario -/
structure CookieScenario where
  cookies_per_batch : ℕ
  flour_per_batch : ℕ
  flour_bags : ℕ
  flour_per_bag : ℕ
  cookies_eaten : ℕ

/-- Calculates the number of cookies left after baking and eating -/
def cookies_left (scenario : CookieScenario) : ℕ :=
  let total_flour := scenario.flour_bags * scenario.flour_per_bag
  let total_cookies := (total_flour / scenario.flour_per_batch) * scenario.cookies_per_batch
  total_cookies - scenario.cookies_eaten

/-- Theorem stating the number of cookies left in the given scenario -/
theorem cookies_left_in_scenario : 
  let scenario : CookieScenario := {
    cookies_per_batch := 12,
    flour_per_batch := 2,
    flour_bags := 4,
    flour_per_bag := 5,
    cookies_eaten := 15
  }
  cookies_left scenario = 105 := by
  sorry


end NUMINAMATH_CALUDE_cookies_left_in_scenario_l486_48683


namespace NUMINAMATH_CALUDE_inequality_proof_l486_48682

theorem inequality_proof (a b c : ℝ) :
  a = 31/32 →
  b = Real.cos (1/4) →
  c = 4 * Real.sin (1/4) →
  c > b ∧ b > a := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l486_48682


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l486_48693

-- Define the repeating decimals
def repeating_038 : ℚ := 38 / 999
def repeating_4 : ℚ := 4 / 9

-- State the theorem
theorem product_of_repeating_decimals :
  repeating_038 * repeating_4 = 152 / 8991 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l486_48693


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l486_48647

theorem quadratic_rewrite (g h j : ℤ) :
  (∀ x : ℝ, 4 * x^2 - 16 * x - 21 = (g * x + h)^2 + j) →
  g * h = -8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l486_48647


namespace NUMINAMATH_CALUDE_floor_equation_iff_solution_set_l486_48622

def floor_equation (x : ℝ) : Prop :=
  ⌊x^2 - 2*x⌋ + 2*⌊x⌋ = ⌊x⌋^2

def solution_set (x : ℝ) : Prop :=
  (∃ (n : ℤ), n < 0 ∧ x = n) ∨
  x = 0 ∨
  (∃ (n : ℕ), n ≥ 1 ∧ n ≤ x ∧ x < Real.sqrt (n^2 - 2*n + 2) + 1)

theorem floor_equation_iff_solution_set :
  ∀ x : ℝ, floor_equation x ↔ solution_set x :=
sorry

end NUMINAMATH_CALUDE_floor_equation_iff_solution_set_l486_48622


namespace NUMINAMATH_CALUDE_student_age_problem_l486_48658

theorem student_age_problem (total_students : Nat) 
  (avg_age_all : Nat) (num_group1 : Nat) (avg_age_group1 : Nat) 
  (num_group2 : Nat) (avg_age_group2 : Nat) :
  total_students = 17 →
  avg_age_all = 17 →
  num_group1 = 5 →
  avg_age_group1 = 14 →
  num_group2 = 9 →
  avg_age_group2 = 16 →
  (total_students * avg_age_all) - (num_group1 * avg_age_group1) - (num_group2 * avg_age_group2) = 75 := by
  sorry

end NUMINAMATH_CALUDE_student_age_problem_l486_48658


namespace NUMINAMATH_CALUDE_solution_count_l486_48608

/-- The number of distinct divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The number of positive integer solutions (x, y) to the equation 1/n = 1/x + 1/y where x ≠ y -/
def num_solutions (n : ℕ+) : ℕ := sorry

theorem solution_count (n : ℕ+) : num_solutions n = num_divisors (n^2) - 1 := by sorry

end NUMINAMATH_CALUDE_solution_count_l486_48608


namespace NUMINAMATH_CALUDE_ones_digit_of_13_power_power_cycle_of_3_main_theorem_l486_48679

theorem ones_digit_of_13_power (n : ℕ) : n > 0 → (13^n) % 10 = (3^n) % 10 := by sorry

theorem power_cycle_of_3 (n : ℕ) : (3^n) % 10 = (3^(n % 4)) % 10 := by sorry

theorem main_theorem : (13^(13 * (12^12))) % 10 = 9 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_13_power_power_cycle_of_3_main_theorem_l486_48679


namespace NUMINAMATH_CALUDE_dan_added_sixteen_pencils_l486_48681

/-- The number of pencils Dan placed on the desk -/
def pencils_added (drawer : ℕ) (desk : ℕ) (total : ℕ) : ℕ :=
  total - (drawer + desk)

/-- Proof that Dan placed 16 pencils on the desk -/
theorem dan_added_sixteen_pencils :
  pencils_added 43 19 78 = 16 := by
  sorry

end NUMINAMATH_CALUDE_dan_added_sixteen_pencils_l486_48681


namespace NUMINAMATH_CALUDE_problem_solution_l486_48614

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | (x - m + 2)*(x - m - 2) ≤ 0}

theorem problem_solution :
  (∀ m : ℝ, (A ∩ B m = {x | 0 ≤ x ∧ x ≤ 3}) → m = 2) ∧
  (∀ m : ℝ, (A ⊆ (Set.univ \ B m)) → (m < -3 ∨ m > 5)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l486_48614


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l486_48634

theorem complex_equation_solutions :
  ∃ (S : Set ℂ), S = {z : ℂ | z^6 + 6*I = 0} ∧
  S = {I, -I} ∪ {z : ℂ | ∃ k : ℕ, 0 ≤ k ∧ k < 4 ∧ z = (-6*I)^(1/6) * (Complex.exp (2*π*I*(k:ℝ)/4))} :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l486_48634


namespace NUMINAMATH_CALUDE_sqrt_2x_plus_4_real_range_l486_48697

theorem sqrt_2x_plus_4_real_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = 2 * x + 4) ↔ x ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2x_plus_4_real_range_l486_48697


namespace NUMINAMATH_CALUDE_average_daily_net_income_is_366_l486_48689

/-- Represents the financial data for a single day -/
structure DailyData where
  income : ℕ
  tips : ℕ
  expenses : ℕ

/-- Calculates the net income for a single day -/
def netIncome (data : DailyData) : ℕ :=
  data.income + data.tips - data.expenses

/-- The financial data for 5 days -/
def fiveDaysData : Vector DailyData 5 :=
  ⟨[
    { income := 300, tips := 50, expenses := 80 },
    { income := 150, tips := 20, expenses := 40 },
    { income := 750, tips := 100, expenses := 150 },
    { income := 200, tips := 30, expenses := 50 },
    { income := 600, tips := 70, expenses := 120 }
  ], rfl⟩

/-- Calculates the average daily net income -/
def averageDailyNetIncome (data : Vector DailyData 5) : ℚ :=
  (data.toList.map netIncome).sum / 5

/-- Theorem stating that the average daily net income is $366 -/
theorem average_daily_net_income_is_366 :
  averageDailyNetIncome fiveDaysData = 366 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_net_income_is_366_l486_48689


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l486_48629

theorem shopkeeper_profit (discount : ℝ) (profit_with_discount : ℝ) :
  discount = 0.04 →
  profit_with_discount = 0.26 →
  let cost_price := 100
  let selling_price := cost_price * (1 + profit_with_discount)
  let marked_price := selling_price / (1 - discount)
  let profit_without_discount := (marked_price - cost_price) / cost_price
  profit_without_discount = 0.3125 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l486_48629


namespace NUMINAMATH_CALUDE_trajectory_of_M_equation_of_l_area_of_POM_smallest_circle_l486_48611

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*y = 0

-- Define point P
def P : ℝ × ℝ := (2, 2)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the midpoint M of chord AB
def M (x y : ℝ) : Prop := ∃ (a b : ℝ × ℝ), 
  circle_C a.1 a.2 ∧ circle_C b.1 b.2 ∧ 
  x = (a.1 + b.1) / 2 ∧ y = (a.2 + b.2) / 2

-- Theorem 1: Trajectory of M
theorem trajectory_of_M : 
  ∀ x y : ℝ, M x y → (x - 1)^2 + (y - 3)^2 = 2 :=
sorry

-- Theorem 2a: Equation of line l when |OP| = |OM|
theorem equation_of_l : 
  ∃ x y : ℝ, M x y ∧ (x^2 + y^2 = P.1^2 + P.2^2) → 
  ∀ x y : ℝ, y = -1/3 * x + 8/3 :=
sorry

-- Theorem 2b: Area of triangle POM when |OP| = |OM|
theorem area_of_POM : 
  ∃ x y : ℝ, M x y ∧ (x^2 + y^2 = P.1^2 + P.2^2) → 
  (1/2) * |P.1 * y - P.2 * x| = 16/5 :=
sorry

-- Theorem 3: Equation of smallest circle through intersection of C and l
theorem smallest_circle : 
  ∃ x y : ℝ, circle_C x y ∧ y = -1/3 * x + 8/3 → 
  ∀ x y : ℝ, (x + 2/5)^2 + (y - 14/5)^2 = 72/5 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_M_equation_of_l_area_of_POM_smallest_circle_l486_48611


namespace NUMINAMATH_CALUDE_largest_inexpressible_number_l486_48657

/-- A function that checks if a natural number can be expressed as a non-negative linear combination of 5 and 6 -/
def isExpressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

/-- The theorem stating that 19 is the largest natural number not exceeding 50 that cannot be expressed as a non-negative linear combination of 5 and 6 -/
theorem largest_inexpressible_number :
  (∀ (k : ℕ), k > 19 ∧ k ≤ 50 → isExpressible k) ∧
  ¬isExpressible 19 :=
sorry

end NUMINAMATH_CALUDE_largest_inexpressible_number_l486_48657


namespace NUMINAMATH_CALUDE_equation_solution_l486_48687

theorem equation_solution (x y : ℝ) : 
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ (y = -x - 2 ∨ y = -2 * x + 1) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l486_48687


namespace NUMINAMATH_CALUDE_arithmetic_equality_l486_48688

theorem arithmetic_equality : (3652 * 2487) + (979 - 45 * 13) = 9085008 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l486_48688


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_nonagon_diagonal_intersection_probability_proof_l486_48684

/-- The probability of two randomly chosen diagonals intersecting inside a regular nonagon -/
theorem nonagon_diagonal_intersection_probability : ℚ :=
  14 / 39

/-- The number of sides in a nonagon -/
def nonagon_sides : ℕ := 9

/-- The number of diagonals in a nonagon -/
def nonagon_diagonals : ℕ := (nonagon_sides.choose 2) - nonagon_sides

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def diagonal_pairs : ℕ := nonagon_diagonals.choose 2

/-- The number of ways to choose 4 points from the nonagon vertices -/
def intersecting_diagonal_sets : ℕ := nonagon_sides.choose 4

theorem nonagon_diagonal_intersection_probability_proof :
  (intersecting_diagonal_sets : ℚ) / diagonal_pairs = nonagon_diagonal_intersection_probability :=
sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_nonagon_diagonal_intersection_probability_proof_l486_48684


namespace NUMINAMATH_CALUDE_delegate_grouping_exists_l486_48694

/-- Represents a delegate with their country and seating position. -/
structure Delegate where
  country : Fin 50
  position : Fin 100

/-- Represents a seating arrangement of delegates around a circular table. -/
def SeatingArrangement := Fin 100 → Delegate

/-- Represents a grouping of delegates. -/
def Grouping := Delegate → Bool

/-- Checks if a delegate has at most one neighbor in the same group. -/
def atMostOneNeighborInGroup (s : SeatingArrangement) (g : Grouping) (d : Delegate) : Prop :=
  let leftNeighbor := s ((d.position - 1 + 100) % 100)
  let rightNeighbor := s ((d.position + 1) % 100)
  ¬(g leftNeighbor ∧ g rightNeighbor ∧ g d = g leftNeighbor ∧ g d = g rightNeighbor)

/-- Main theorem statement -/
theorem delegate_grouping_exists (s : SeatingArrangement) :
  ∃ g : Grouping,
    (∀ c : Fin 50, ∃! d : Delegate, g d = true ∧ d.country = c) ∧
    (∀ c : Fin 50, ∃! d : Delegate, g d = false ∧ d.country = c) ∧
    (∀ d : Delegate, atMostOneNeighborInGroup s g d) :=
  sorry

end NUMINAMATH_CALUDE_delegate_grouping_exists_l486_48694


namespace NUMINAMATH_CALUDE_monogram_count_is_300_l486_48653

/-- The number of letters in the alphabet before 'A' --/
def n : ℕ := 25

/-- The number of initials to choose (first and middle) --/
def k : ℕ := 2

/-- The number of ways to choose k distinct letters from n letters in alphabetical order --/
def monogram_count : ℕ := Nat.choose n k

/-- Theorem stating that the number of possible monograms is 300 --/
theorem monogram_count_is_300 : monogram_count = 300 := by
  sorry

end NUMINAMATH_CALUDE_monogram_count_is_300_l486_48653


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l486_48674

def A (a : ℝ) : Set ℝ := {2, a}
def B (a : ℝ) : Set ℝ := {-1, a^2 - 2}

theorem intersection_implies_a_value (a : ℝ) :
  (A a ∩ B a).Nonempty → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l486_48674


namespace NUMINAMATH_CALUDE_job_completion_time_l486_48670

theorem job_completion_time (x : ℝ) : 
  x > 0 →  -- A's completion time is positive
  8 * (1 / x + 1 / 20) = 1 - 0.06666666666666665 →  -- Condition after 8 days of working together
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_job_completion_time_l486_48670


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l486_48666

def A : Set ℚ := {x | ∃ k : ℕ, x = 3 * k + 1}
def B : Set ℚ := {x | x ≤ 7}

theorem intersection_of_A_and_B : A ∩ B = {1, 4, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l486_48666


namespace NUMINAMATH_CALUDE_race_distance_proof_l486_48612

/-- The distance John was behind Steve when he began his final push -/
def initial_distance : ℝ := 16

/-- John's speed in meters per second -/
def john_speed : ℝ := 4.2

/-- Steve's speed in meters per second -/
def steve_speed : ℝ := 3.7

/-- Duration of the final push in seconds -/
def final_push_duration : ℝ := 36

/-- The distance John finishes ahead of Steve -/
def final_distance_ahead : ℝ := 2

theorem race_distance_proof :
  john_speed * final_push_duration = 
  steve_speed * final_push_duration + initial_distance + final_distance_ahead :=
by sorry

end NUMINAMATH_CALUDE_race_distance_proof_l486_48612


namespace NUMINAMATH_CALUDE_max_cross_sum_l486_48613

def CrossNumbers : Finset ℕ := {2, 5, 8, 11, 14}

theorem max_cross_sum :
  ∃ (a b c d e : ℕ),
    a ∈ CrossNumbers ∧ b ∈ CrossNumbers ∧ c ∈ CrossNumbers ∧ d ∈ CrossNumbers ∧ e ∈ CrossNumbers ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    a + b + e = b + d + e ∧
    a + c + e = a + b + e ∧
    a + b + e = 36 ∧
    ∀ (x y z : ℕ),
      x ∈ CrossNumbers → y ∈ CrossNumbers → z ∈ CrossNumbers →
      x + y + z ≤ 36 :=
by sorry

end NUMINAMATH_CALUDE_max_cross_sum_l486_48613


namespace NUMINAMATH_CALUDE_jose_remaining_caps_l486_48699

-- Define the initial number of bottle caps Jose has
def initial_caps : ℝ := 143.6

-- Define the number of bottle caps given to Rebecca
def given_to_rebecca : ℝ := 89.2

-- Define the number of bottle caps given to Michael
def given_to_michael : ℝ := 16.7

-- Theorem to prove the number of bottle caps Jose has left
theorem jose_remaining_caps :
  initial_caps - (given_to_rebecca + given_to_michael) = 37.7 := by
  sorry

end NUMINAMATH_CALUDE_jose_remaining_caps_l486_48699


namespace NUMINAMATH_CALUDE_valeria_apartment_number_l486_48660

def is_not_multiple_of_5 (n : ℕ) : Prop := n % 5 ≠ 0

def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_digits_less_than_8 (n : ℕ) : Prop :=
  (n / 10 + n % 10) < 8

def units_digit_is_6 (n : ℕ) : Prop := n % 10 = 6

theorem valeria_apartment_number (n : ℕ) :
  n ≥ 10 ∧ n < 100 →
  (is_not_multiple_of_5 n ∧ is_odd n ∧ units_digit_is_6 n) ∨
  (is_not_multiple_of_5 n ∧ is_odd n ∧ sum_of_digits_less_than_8 n) ∨
  (is_not_multiple_of_5 n ∧ sum_of_digits_less_than_8 n ∧ units_digit_is_6 n) ∨
  (is_odd n ∧ sum_of_digits_less_than_8 n ∧ units_digit_is_6 n) →
  units_digit_is_6 n :=
by sorry

end NUMINAMATH_CALUDE_valeria_apartment_number_l486_48660


namespace NUMINAMATH_CALUDE_jerrys_age_l486_48630

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 2 * jerry_age - 4 →
  mickey_age = 22 →
  jerry_age = 13 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l486_48630


namespace NUMINAMATH_CALUDE_checkers_placement_divisibility_l486_48619

/-- Given a prime p ≥ 5, r(p) is the number of ways to place p identical checkers 
    on a p × p checkerboard such that not all checkers are in the same row. -/
def r (p : ℕ) : ℕ := sorry

theorem checkers_placement_divisibility (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) : 
  p^5 ∣ r p := by
  sorry

end NUMINAMATH_CALUDE_checkers_placement_divisibility_l486_48619


namespace NUMINAMATH_CALUDE_ratio_simplification_l486_48695

theorem ratio_simplification : 
  (10^2001 + 10^2003) / (10^2002 + 10^2002) = 101 / 20 := by sorry

end NUMINAMATH_CALUDE_ratio_simplification_l486_48695


namespace NUMINAMATH_CALUDE_triangle_area_and_side_l486_48659

theorem triangle_area_and_side (a b c : ℝ) (A B C : ℝ) :
  b = 2 →
  c = Real.sqrt 3 →
  A = π / 6 →
  (1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) ∧
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) ∧ (a = 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_and_side_l486_48659


namespace NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l486_48646

theorem polar_to_rectangular_equivalence (ρ θ x y : ℝ) :
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (3 * ρ * Real.cos θ + 4 * ρ * Real.sin θ = 2) →
  (3 * x + 4 * y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l486_48646


namespace NUMINAMATH_CALUDE_franks_age_l486_48638

/-- Represents the ages of Dave, Ella, and Frank -/
structure Ages where
  dave : ℕ
  ella : ℕ
  frank : ℕ

/-- The conditions from the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- The average of their ages is 10
  (ages.dave + ages.ella + ages.frank) / 3 = 10 ∧
  -- Five years ago, Frank was the same age as Dave is now
  ages.frank - 5 = ages.dave ∧
  -- In 2 years, Ella's age will be 3/4 of Dave's age at that time
  ages.ella + 2 = (3 * (ages.dave + 2)) / 4

/-- The theorem to prove -/
theorem franks_age (ages : Ages) (h : satisfies_conditions ages) : ages.frank = 14 := by
  sorry


end NUMINAMATH_CALUDE_franks_age_l486_48638


namespace NUMINAMATH_CALUDE_lawn_mowing_payment_l486_48675

theorem lawn_mowing_payment (rate : ℚ) (lawns_mowed : ℚ) : 
  rate = 15 / 4 → lawns_mowed = 5 / 2 → rate * lawns_mowed = 75 / 8 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_payment_l486_48675


namespace NUMINAMATH_CALUDE_power_sum_equality_l486_48605

theorem power_sum_equality (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  (a^5 + b^5 = c^5 + d^5) ∧ 
  ∃ (a b c d : ℝ), (a + b = c + d) ∧ (a^3 + b^3 = c^3 + d^3) ∧ (a^4 + b^4 ≠ c^4 + d^4) :=
by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l486_48605


namespace NUMINAMATH_CALUDE_smallest_valid_amount_l486_48628

/-- Represents the number of bags -/
def num_bags : List Nat := [8, 7, 6]

/-- Represents the types of currency -/
inductive Currency
| Dollar
| HalfDollar
| QuarterDollar

/-- Checks if a given amount can be equally distributed into the specified number of bags for all currency types -/
def is_valid_distribution (amount : Nat) (bags : Nat) : Prop :=
  ∀ c : Currency, ∃ n : Nat, n * bags = amount

/-- The main theorem stating the smallest valid amount -/
theorem smallest_valid_amount :
  (∀ bags ∈ num_bags, is_valid_distribution 294 bags) ∧
  (∀ amount < 294, ¬(∀ bags ∈ num_bags, is_valid_distribution amount bags)) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_amount_l486_48628


namespace NUMINAMATH_CALUDE_triangle_inequality_l486_48639

theorem triangle_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^3 + b^3 + c^3 + 4*a*b*c ≤ 9/32 * (a + b + c)^3 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l486_48639


namespace NUMINAMATH_CALUDE_man_and_son_work_time_l486_48664

/-- Given a task that takes a man 4 days and his son 12 days to complete individually, 
    prove that they can complete the task together in 3 days. -/
theorem man_and_son_work_time (task : ℝ) (man_rate son_rate combined_rate : ℝ) : 
  task > 0 ∧ 
  man_rate = task / 4 ∧ 
  son_rate = task / 12 ∧ 
  combined_rate = man_rate + son_rate → 
  task / combined_rate = 3 := by
sorry

end NUMINAMATH_CALUDE_man_and_son_work_time_l486_48664


namespace NUMINAMATH_CALUDE_no_real_solutions_l486_48692

theorem no_real_solutions :
  ¬ ∃ x : ℝ, Real.sqrt (3 * x - 2) + 8 / Real.sqrt (3 * x - 2) = 4 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l486_48692


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l486_48632

/-- Represents the number of seats in each row -/
def seats : Fin 2 → Nat
  | 0 => 6  -- front row
  | 1 => 7  -- back row

/-- Calculates the number of ways to arrange 2 people in two rows of seats
    such that they are not sitting next to each other -/
def seating_arrangements : Nat :=
  let different_rows := seats 0 * seats 1 * 2
  let front_row := 2 * 4 + 4 * 3
  let back_row := 2 * 5 + 5 * 4
  different_rows + front_row + back_row

/-- Theorem stating that the number of seating arrangements is 134 -/
theorem seating_arrangements_count : seating_arrangements = 134 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l486_48632


namespace NUMINAMATH_CALUDE_min_value_of_fraction_sum_l486_48668

/-- Given a quadratic function f(x) = ax^2 - 4x + c with range [0,+∞),
    prove that the minimum value of 1/c + 9/a is 3 -/
theorem min_value_of_fraction_sum (a c : ℝ) (h₁ : a > 0) (h₂ : c > 0)
    (h₃ : ∀ x, ax^2 - 4*x + c ≥ 0) : 
    ∃ (m : ℝ), m = 3 ∧ ∀ a c, a > 0 → c > 0 → (∀ x, ax^2 - 4*x + c ≥ 0) → 1/c + 9/a ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_sum_l486_48668


namespace NUMINAMATH_CALUDE_suyeong_run_distance_l486_48610

/-- The circumference of the playground in meters -/
def playground_circumference : ℝ := 242.7

/-- The number of laps Suyeong ran -/
def laps_run : ℕ := 5

/-- The total distance Suyeong ran in meters -/
def total_distance : ℝ := playground_circumference * (laps_run : ℝ)

theorem suyeong_run_distance : total_distance = 1213.5 := by
  sorry

end NUMINAMATH_CALUDE_suyeong_run_distance_l486_48610


namespace NUMINAMATH_CALUDE_tan_roots_sum_angles_l486_48672

theorem tan_roots_sum_angles (α β : Real) : 
  (∃ (x y : Real), x^2 + Real.sqrt 3 * x - 2 = 0 ∧ y^2 + Real.sqrt 3 * y - 2 = 0 ∧ 
   x = Real.tan α ∧ y = Real.tan β) →
  -π/2 < α ∧ α < π/2 →
  -π/2 < β ∧ β < π/2 →
  α + β = π/6 ∨ α + β = -5*π/6 :=
by sorry

end NUMINAMATH_CALUDE_tan_roots_sum_angles_l486_48672


namespace NUMINAMATH_CALUDE_frog_grasshopper_difference_l486_48604

/-- Represents the jumping distances in the contest -/
structure JumpDistances where
  grasshopper : ℕ
  frog : ℕ
  mouse : ℕ

/-- The conditions of the jumping contest -/
def contest_conditions (j : JumpDistances) : Prop :=
  j.grasshopper = 19 ∧
  j.frog > j.grasshopper ∧
  j.mouse = j.frog + 20 ∧
  j.mouse = j.grasshopper + 30

/-- The theorem stating the difference between the frog's and grasshopper's jump distances -/
theorem frog_grasshopper_difference (j : JumpDistances) 
  (h : contest_conditions j) : j.frog - j.grasshopper = 10 := by
  sorry


end NUMINAMATH_CALUDE_frog_grasshopper_difference_l486_48604


namespace NUMINAMATH_CALUDE_total_waiting_after_changes_l486_48652

/-- Represents the number of people waiting at each entrance of SFL -/
structure EntranceCount where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- Calculates the total number of people waiting at all entrances -/
def total_waiting (count : EntranceCount) : ℕ :=
  count.a + count.b + count.c + count.d + count.e

/-- Initial count of people waiting at each entrance -/
def initial_count : EntranceCount :=
  { a := 283, b := 356, c := 412, d := 179, e := 389 }

/-- Final count of people waiting at each entrance after changes -/
def final_count : EntranceCount :=
  { a := initial_count.a - 15,
    b := initial_count.b,
    c := initial_count.c + 10,
    d := initial_count.d,
    e := initial_count.e - 20 }

/-- Theorem stating that the total number of people waiting after changes is 1594 -/
theorem total_waiting_after_changes :
  total_waiting final_count = 1594 := by sorry

end NUMINAMATH_CALUDE_total_waiting_after_changes_l486_48652


namespace NUMINAMATH_CALUDE_bianca_drawing_time_l486_48644

/-- The number of minutes Bianca spent drawing at school -/
def minutes_at_school : ℕ := sorry

/-- The number of minutes Bianca spent drawing at home -/
def minutes_at_home : ℕ := 19

/-- The total number of minutes Bianca spent drawing -/
def total_minutes : ℕ := 41

/-- Theorem stating that Bianca spent 22 minutes drawing at school -/
theorem bianca_drawing_time : minutes_at_school = 22 := by
  sorry

end NUMINAMATH_CALUDE_bianca_drawing_time_l486_48644


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l486_48602

/-- An arithmetic sequence {a_n} satisfying the given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 60) : 
  a 7 - (1/3) * a 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l486_48602


namespace NUMINAMATH_CALUDE_volume_ratio_cylinder_cone_sphere_l486_48626

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h

theorem volume_ratio_cylinder_cone_sphere (r : ℝ) (h_pos : r > 0) :
  ∃ (k : ℝ), k > 0 ∧ 
    cylinder_volume r (2 * r) = 3 * k ∧
    cone_volume r (2 * r) = k ∧
    sphere_volume r = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_cylinder_cone_sphere_l486_48626


namespace NUMINAMATH_CALUDE_share_price_increase_l486_48624

theorem share_price_increase (P : ℝ) (X : ℝ) : 
  X > 0 →
  (P * (1 + X / 100)) * (1 + 1 / 3) = P * 1.6 →
  X = 20 := by
sorry

end NUMINAMATH_CALUDE_share_price_increase_l486_48624


namespace NUMINAMATH_CALUDE_parking_probability_l486_48680

/-- Represents a parking lot configuration -/
structure ParkingLot :=
  (total_spaces : ℕ)
  (occupied_spaces : ℕ)

/-- Calculates the probability of finding two adjacent empty spaces in a parking lot -/
def probability_of_two_adjacent_empty_spaces (p : ParkingLot) : ℚ :=
  1 - (Nat.choose (p.total_spaces - p.occupied_spaces + 1) 5 : ℚ) / (Nat.choose p.total_spaces (p.total_spaces - p.occupied_spaces) : ℚ)

/-- Theorem stating the probability of finding two adjacent empty spaces in the given scenario -/
theorem parking_probability (p : ParkingLot) 
  (h1 : p.total_spaces = 20) 
  (h2 : p.occupied_spaces = 15) : 
  probability_of_two_adjacent_empty_spaces p = 232 / 323 := by
  sorry

end NUMINAMATH_CALUDE_parking_probability_l486_48680
