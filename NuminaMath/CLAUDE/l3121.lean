import Mathlib

namespace NUMINAMATH_CALUDE_matthew_age_difference_l3121_312163

/-- Given three children whose ages sum to 35, with Matthew 2 years older than Rebecca
    and Freddy being 15, prove that Matthew is 4 years younger than Freddy. -/
theorem matthew_age_difference (matthew rebecca freddy : ℕ) : 
  matthew + rebecca + freddy = 35 →
  matthew = rebecca + 2 →
  freddy = 15 →
  freddy - matthew = 4 := by
sorry

end NUMINAMATH_CALUDE_matthew_age_difference_l3121_312163


namespace NUMINAMATH_CALUDE_shopping_theorem_l3121_312172

def shopping_problem (total_amount : ℝ) : Prop :=
  let clothing_percent : ℝ := 0.40
  let food_percent : ℝ := 0.20
  let electronics_percent : ℝ := 0.10
  let cosmetics_percent : ℝ := 0.20
  let household_percent : ℝ := 0.10

  let clothing_discount : ℝ := 0.10
  let food_discount : ℝ := 0.05
  let electronics_discount : ℝ := 0.15
  let cosmetics_discount : ℝ := 0
  let household_discount : ℝ := 0

  let clothing_tax : ℝ := 0.06
  let food_tax : ℝ := 0
  let electronics_tax : ℝ := 0.10
  let cosmetics_tax : ℝ := 0.08
  let household_tax : ℝ := 0.04

  let clothing_amount := total_amount * clothing_percent
  let food_amount := total_amount * food_percent
  let electronics_amount := total_amount * electronics_percent
  let cosmetics_amount := total_amount * cosmetics_percent
  let household_amount := total_amount * household_percent

  let clothing_tax_paid := clothing_amount * (1 - clothing_discount) * clothing_tax
  let food_tax_paid := food_amount * (1 - food_discount) * food_tax
  let electronics_tax_paid := electronics_amount * (1 - electronics_discount) * electronics_tax
  let cosmetics_tax_paid := cosmetics_amount * (1 - cosmetics_discount) * cosmetics_tax
  let household_tax_paid := household_amount * (1 - household_discount) * household_tax

  let total_tax_paid := clothing_tax_paid + food_tax_paid + electronics_tax_paid + cosmetics_tax_paid + household_tax_paid
  let total_tax_percentage := (total_tax_paid / total_amount) * 100

  total_tax_percentage = 5.01

theorem shopping_theorem : ∀ (total_amount : ℝ), total_amount > 0 → shopping_problem total_amount := by
  sorry

end NUMINAMATH_CALUDE_shopping_theorem_l3121_312172


namespace NUMINAMATH_CALUDE_smallest_assembly_size_l3121_312109

theorem smallest_assembly_size : ∃ n : ℕ, n > 50 ∧ 
  (∃ x : ℕ, n = 4 * x + (x + 2)) ∧ 
  (∀ m : ℕ, m > 50 → (∃ y : ℕ, m = 4 * y + (y + 2)) → m ≥ n) ∧
  n = 52 :=
by sorry

end NUMINAMATH_CALUDE_smallest_assembly_size_l3121_312109


namespace NUMINAMATH_CALUDE_henry_actual_earnings_l3121_312188

/-- Represents Henry's summer job earnings --/
def HenryEarnings : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  fun (lawn_rate pile_rate driveway_rate lawns_mowed piles_raked driveways_shoveled : ℕ) =>
    lawn_rate * lawns_mowed + pile_rate * piles_raked + driveway_rate * driveways_shoveled

/-- Theorem stating Henry's actual earnings --/
theorem henry_actual_earnings :
  HenryEarnings 5 10 15 5 3 2 = 85 := by
  sorry

end NUMINAMATH_CALUDE_henry_actual_earnings_l3121_312188


namespace NUMINAMATH_CALUDE_min_sum_squares_l3121_312183

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  ∃ (min : ℝ), min = t^2 / 3 ∧ 
  (∀ (x y z : ℝ), x + y + z = t → x^2 + y^2 + z^2 ≥ min) ∧
  (∃ (x y z : ℝ), x + y + z = t ∧ x^2 + y^2 + z^2 = min) := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3121_312183


namespace NUMINAMATH_CALUDE_go_kart_tickets_value_l3121_312145

/-- The number of tickets required for a go-kart ride -/
def go_kart_tickets : ℕ := sorry

/-- The number of times Paula rides the go-karts -/
def go_kart_rides : ℕ := 1

/-- The number of times Paula rides the bumper cars -/
def bumper_car_rides : ℕ := 4

/-- The number of tickets required for a bumper car ride -/
def bumper_car_tickets : ℕ := 5

/-- The total number of tickets Paula needs -/
def total_tickets : ℕ := 24

theorem go_kart_tickets_value : 
  go_kart_tickets = 4 :=
by sorry

end NUMINAMATH_CALUDE_go_kart_tickets_value_l3121_312145


namespace NUMINAMATH_CALUDE_highway_time_greater_than_swamp_time_l3121_312128

/-- Represents the hunter's journey with different terrains and speeds -/
structure HunterJourney where
  swamp_speed : ℝ
  forest_speed : ℝ
  highway_speed : ℝ
  total_time : ℝ
  total_distance : ℝ

/-- Theorem stating that the time spent on the highway is greater than the time spent in the swamp -/
theorem highway_time_greater_than_swamp_time (journey : HunterJourney) 
  (h1 : journey.swamp_speed = 2)
  (h2 : journey.forest_speed = 4)
  (h3 : journey.highway_speed = 6)
  (h4 : journey.total_time = 4)
  (h5 : journey.total_distance = 17) : 
  ∃ (swamp_time highway_time : ℝ), 
    swamp_time * journey.swamp_speed + 
    (journey.total_time - swamp_time - highway_time) * journey.forest_speed + 
    highway_time * journey.highway_speed = journey.total_distance ∧
    highway_time > swamp_time :=
by sorry

end NUMINAMATH_CALUDE_highway_time_greater_than_swamp_time_l3121_312128


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3121_312151

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0}

-- State the theorem
theorem intersection_implies_a_value :
  ∃ a : ℝ, A ∩ B a = {x : ℝ | -2 ≤ x ∧ x ≤ 1} → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3121_312151


namespace NUMINAMATH_CALUDE_amp_four_neg_three_l3121_312191

-- Define the & operation
def amp (x y : ℤ) : ℤ := x * (y + 2) + x * y

-- Theorem statement
theorem amp_four_neg_three : amp 4 (-3) = -16 := by
  sorry

end NUMINAMATH_CALUDE_amp_four_neg_three_l3121_312191


namespace NUMINAMATH_CALUDE_seulgi_kicks_to_win_l3121_312182

theorem seulgi_kicks_to_win (hohyeon_first hohyeon_second hyunjeong_first hyunjeong_second seulgi_first : ℕ) 
  (h1 : hohyeon_first = 23)
  (h2 : hohyeon_second = 28)
  (h3 : hyunjeong_first = 32)
  (h4 : hyunjeong_second = 17)
  (h5 : seulgi_first = 27) :
  ∃ seulgi_second : ℕ, 
    seulgi_second ≥ 25 ∧ 
    seulgi_first + seulgi_second > hohyeon_first + hohyeon_second ∧ 
    seulgi_first + seulgi_second > hyunjeong_first + hyunjeong_second :=
by sorry

end NUMINAMATH_CALUDE_seulgi_kicks_to_win_l3121_312182


namespace NUMINAMATH_CALUDE_current_speed_current_speed_calculation_l3121_312125

/-- The speed of the current in a river given the man's rowing speed in still water,
    the time taken to cover a distance downstream, and the distance covered. -/
theorem current_speed (rowing_speed : ℝ) (time : ℝ) (distance : ℝ) : ℝ :=
  let downstream_speed := distance / (time / 3600)
  downstream_speed - rowing_speed

/-- Proof that the speed of the current is approximately 3.00048 kmph given the specified conditions. -/
theorem current_speed_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ 
  |current_speed 9 17.998560115190788 0.06 - 3.00048| < ε :=
sorry

end NUMINAMATH_CALUDE_current_speed_current_speed_calculation_l3121_312125


namespace NUMINAMATH_CALUDE_fabric_problem_solution_l3121_312198

/-- Represents the fabric and flag problem -/
structure FabricProblem where
  initial_fabric : Float
  square_side : Float
  square_count : Nat
  wide_rect_length : Float
  wide_rect_width : Float
  wide_rect_count : Nat
  tall_rect_length : Float
  tall_rect_width : Float
  tall_rect_count : Nat
  triangle_base : Float
  triangle_height : Float
  triangle_count : Nat
  hexagon_side : Float
  hexagon_apothem : Float
  hexagon_count : Nat

/-- Calculates the remaining fabric after making flags -/
def remaining_fabric (p : FabricProblem) : Float :=
  p.initial_fabric -
  (p.square_side * p.square_side * p.square_count.toFloat +
   p.wide_rect_length * p.wide_rect_width * p.wide_rect_count.toFloat +
   p.tall_rect_length * p.tall_rect_width * p.tall_rect_count.toFloat +
   (p.triangle_base * p.triangle_height / 2) * p.triangle_count.toFloat +
   (6 * p.hexagon_side * p.hexagon_apothem / 2) * p.hexagon_count.toFloat)

/-- The theorem stating the remaining fabric for the given problem -/
theorem fabric_problem_solution (p : FabricProblem) :
  p.initial_fabric = 1500 ∧
  p.square_side = 4 ∧
  p.square_count = 22 ∧
  p.wide_rect_length = 5 ∧
  p.wide_rect_width = 3 ∧
  p.wide_rect_count = 28 ∧
  p.tall_rect_length = 3 ∧
  p.tall_rect_width = 5 ∧
  p.tall_rect_count = 14 ∧
  p.triangle_base = 6 ∧
  p.triangle_height = 4 ∧
  p.triangle_count = 18 ∧
  p.hexagon_side = 3 ∧
  p.hexagon_apothem = 2.6 ∧
  p.hexagon_count = 24 →
  remaining_fabric p = -259.6 := by
  sorry

end NUMINAMATH_CALUDE_fabric_problem_solution_l3121_312198


namespace NUMINAMATH_CALUDE_total_guesses_l3121_312105

def digits : List ℕ := [1, 1, 1, 1, 2, 2, 2, 2]

def valid_partition (p : List ℕ) : Prop :=
  p.length = 4 ∧ p.sum = 8 ∧ ∀ x ∈ p, 1 ≤ x ∧ x ≤ 3

def num_arrangements : ℕ := Nat.choose 8 4

def num_partitions : ℕ := 35

theorem total_guesses :
  num_arrangements * num_partitions = 2450 :=
sorry

end NUMINAMATH_CALUDE_total_guesses_l3121_312105


namespace NUMINAMATH_CALUDE_labeled_cube_probabilities_l3121_312159

/-- A cube with 6 faces, where 1 face is labeled with 1, 2 faces are labeled with 2, and 3 faces are labeled with 3 -/
structure LabeledCube where
  total_faces : ℕ
  faces_with_1 : ℕ
  faces_with_2 : ℕ
  faces_with_3 : ℕ
  face_sum : total_faces = faces_with_1 + faces_with_2 + faces_with_3
  face_distribution : faces_with_1 = 1 ∧ faces_with_2 = 2 ∧ faces_with_3 = 3

/-- The probability of an event occurring when rolling the cube -/
def probability (cube : LabeledCube) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / cube.total_faces

theorem labeled_cube_probabilities (cube : LabeledCube) :
  (probability cube cube.faces_with_2 = 1/3) ∧
  (∀ n, probability cube (cube.faces_with_1) ≤ probability cube n ∧
        probability cube (cube.faces_with_2) ≤ probability cube n →
        n = cube.faces_with_3) ∧
  (probability cube (cube.faces_with_1 + cube.faces_with_2) =
   probability cube cube.faces_with_3) :=
by sorry

end NUMINAMATH_CALUDE_labeled_cube_probabilities_l3121_312159


namespace NUMINAMATH_CALUDE_samuel_fraction_l3121_312133

theorem samuel_fraction (total : ℝ) (spent : ℝ) (left : ℝ) :
  total = 240 →
  spent = (1 / 5) * total →
  left = 132 →
  (left + spent) / total = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_samuel_fraction_l3121_312133


namespace NUMINAMATH_CALUDE_avg_children_with_children_example_l3121_312170

/-- The average number of children in families with children, given:
  - total_families: The total number of families
  - avg_children: The average number of children per family (including all families)
  - childless_families: The number of childless families
-/
def avg_children_with_children (total_families : ℕ) (avg_children : ℚ) (childless_families : ℕ) : ℚ :=
  (total_families : ℚ) * avg_children / ((total_families : ℚ) - (childless_families : ℚ))

/-- Theorem stating that given 15 families with an average of 3 children per family,
    and exactly 3 childless families, the average number of children in the families
    with children is 3.75 -/
theorem avg_children_with_children_example :
  avg_children_with_children 15 3 3 = 45 / 12 :=
sorry

end NUMINAMATH_CALUDE_avg_children_with_children_example_l3121_312170


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3121_312181

/-- Prove that for an ellipse with the given conditions, its eccentricity is √2/2 -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) :
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / b^2 = 1
  let P := (-c, b^2 / a)
  let A := (a, 0)
  let B := (0, b)
  let O := (0, 0)
  ellipse (-c) (b^2 / a) ∧ 
  (B.2 - A.2) / (B.1 - A.1) = (P.2 - O.2) / (P.1 - O.1) →
  c / a = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3121_312181


namespace NUMINAMATH_CALUDE_different_suit_probability_l3121_312192

theorem different_suit_probability (total_cards : ℕ) (num_suits : ℕ) 
  (h1 : total_cards = 65) 
  (h2 : num_suits = 5) 
  (h3 : total_cards % num_suits = 0) :
  let cards_per_suit := total_cards / num_suits
  let remaining_cards := total_cards - 1
  let different_suit_cards := total_cards - cards_per_suit
  (different_suit_cards : ℚ) / remaining_cards = 13 / 16 := by
sorry


end NUMINAMATH_CALUDE_different_suit_probability_l3121_312192


namespace NUMINAMATH_CALUDE_solution_difference_l3121_312102

-- Define the equation
def equation (x : ℝ) : Prop := (4 - x^2 / 3)^(1/3) = -2

-- Define the set of solutions
def solutions : Set ℝ := {x : ℝ | equation x}

-- Theorem statement
theorem solution_difference : 
  ∃ (x y : ℝ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x - y| = 12 :=
sorry

end NUMINAMATH_CALUDE_solution_difference_l3121_312102


namespace NUMINAMATH_CALUDE_smallest_n_for_probability_threshold_l3121_312116

/-- The probability of drawing n-1 white marbles followed by a red marble -/
def P (n : ℕ) : ℚ := 1 / (n * (n^2 + 1))

/-- The number of boxes -/
def num_boxes : ℕ := 3000

theorem smallest_n_for_probability_threshold :
  ∀ n : ℕ, n > 0 → n < 15 → P n ≥ 1 / num_boxes ∧
  P 15 < 1 / num_boxes :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_probability_threshold_l3121_312116


namespace NUMINAMATH_CALUDE_tournament_size_l3121_312148

/-- Represents a tournament with the given conditions -/
structure Tournament where
  n : ℕ  -- Number of players not in the weakest 15
  total_players : ℕ := n + 15
  total_games : ℕ := (total_players * (total_players - 1)) / 2
  weak_player_games : ℕ := 15 * 14 / 2
  strong_player_games : ℕ := n * (n - 1) / 2
  cross_games : ℕ := 15 * n

/-- The theorem stating that the tournament must have 36 players -/
theorem tournament_size (t : Tournament) : t.total_players = 36 := by
  sorry

end NUMINAMATH_CALUDE_tournament_size_l3121_312148


namespace NUMINAMATH_CALUDE_correct_factorization_l3121_312121

/-- Given a quadratic expression x^2 - ax + b, if (x+6)(x-1) and (x-2)(x+1) are incorrect
    factorizations due to misreading a and b respectively, then the correct factorization
    is (x+2)(x-3). -/
theorem correct_factorization (a b : ℤ) : 
  (∃ a', (x^2 - a'*x + b = (x+6)*(x-1)) ∧ (a' ≠ a)) →
  (∃ b', (x^2 - a*x + b' = (x-2)*(x+1)) ∧ (b' ≠ b)) →
  (x^2 - a*x + b = (x+2)*(x-3)) :=
sorry

end NUMINAMATH_CALUDE_correct_factorization_l3121_312121


namespace NUMINAMATH_CALUDE_fraction_irreducible_l3121_312195

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l3121_312195


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3121_312174

theorem sum_of_cubes (a b c : ℝ) 
  (sum_condition : a + b + c = 3)
  (sum_of_products : a * b + a * c + b * c = 5)
  (product : a * b * c = -6) :
  a^3 + b^3 + c^3 = -36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3121_312174


namespace NUMINAMATH_CALUDE_parallel_lines_point_l3121_312152

def line (m : ℝ) (b : ℝ) := {p : ℝ × ℝ | p.2 = m * p.1 + b}

def parallel (l₁ l₂ : Set (ℝ × ℝ)) : Prop :=
  ∃ m b₁ b₂, l₁ = line m b₁ ∧ l₂ = line m b₂

def angle_of_inclination (l : Set (ℝ × ℝ)) (θ : ℝ) : Prop :=
  ∃ m b, l = line m b ∧ m = Real.tan θ

theorem parallel_lines_point (a : ℝ) : 
  let l₁ : Set (ℝ × ℝ) := line 1 0
  let l₂ : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 = -2 ∧ p.2 = -1) ∨ (p.1 = 3 ∧ p.2 = a)}
  angle_of_inclination l₁ (π/4) → parallel l₁ l₂ → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_point_l3121_312152


namespace NUMINAMATH_CALUDE_arithmetic_sequence_angles_eq_solution_angles_l3121_312101

open Real

-- Define the set of angles that satisfy the condition
def ArithmeticSequenceAngles : Set ℝ :=
  {a | 0 < a ∧ a < 2 * π ∧ 2 * sin (2 * a) = sin a + sin (3 * a)}

-- Define the set of solution angles in radians
def SolutionAngles : Set ℝ :=
  {π/6, 5*π/6, 7*π/6, 11*π/6}

-- Theorem statement
theorem arithmetic_sequence_angles_eq_solution_angles :
  ArithmeticSequenceAngles = SolutionAngles := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_angles_eq_solution_angles_l3121_312101


namespace NUMINAMATH_CALUDE_log_intersects_x_axis_l3121_312160

theorem log_intersects_x_axis : ∃ x : ℝ, x > 0 ∧ Real.log x = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_intersects_x_axis_l3121_312160


namespace NUMINAMATH_CALUDE_ice_cream_bill_calculation_l3121_312127

/-- Calculate the final bill amount for an ice cream outing --/
theorem ice_cream_bill_calculation
  (alicia_total brant_total josh_total yvette_total : ℝ)
  (discount_rate tax_rate tip_rate : ℝ)
  (h_alicia : alicia_total = 16.50)
  (h_brant : brant_total = 20.50)
  (h_josh : josh_total = 16.00)
  (h_yvette : yvette_total = 19.50)
  (h_discount : discount_rate = 0.10)
  (h_tax : tax_rate = 0.08)
  (h_tip : tip_rate = 0.20) :
  let subtotal := alicia_total + brant_total + josh_total + yvette_total
  let discounted_subtotal := subtotal * (1 - discount_rate)
  let tax_amount := discounted_subtotal * tax_rate
  let tip_amount := subtotal * tip_rate
  discounted_subtotal + tax_amount + tip_amount = 84.97 := by
  sorry


end NUMINAMATH_CALUDE_ice_cream_bill_calculation_l3121_312127


namespace NUMINAMATH_CALUDE_complex_ratio_pure_imaginary_l3121_312141

theorem complex_ratio_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 3*I
  let z₂ : ℂ := 3 + 4*I
  (∃ (b : ℝ), z₁ / z₂ = b*I) → a = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_ratio_pure_imaginary_l3121_312141


namespace NUMINAMATH_CALUDE_roadwork_pitch_barrels_l3121_312187

/-- Roadwork problem -/
theorem roadwork_pitch_barrels (total_length day1_paving : ℕ)
  (gravel_per_truckload : ℕ) (gravel_pitch_ratio : ℕ) (truckloads_per_mile : ℕ) :
  total_length = 16 →
  day1_paving = 4 →
  gravel_per_truckload = 2 →
  gravel_pitch_ratio = 5 →
  truckloads_per_mile = 3 →
  (total_length - (day1_paving + (2 * day1_paving - 1))) * truckloads_per_mile * 
    (gravel_per_truckload / gravel_pitch_ratio : ℚ) = 6 :=
by sorry

end NUMINAMATH_CALUDE_roadwork_pitch_barrels_l3121_312187


namespace NUMINAMATH_CALUDE_problem_statement_l3121_312104

theorem problem_statement (a b : ℝ) (h1 : a * b = -3) (h2 : a + b = 2) :
  a^2 * b + a * b^2 = -6 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3121_312104


namespace NUMINAMATH_CALUDE_juans_number_problem_l3121_312146

theorem juans_number_problem (x : ℝ) : 
  (((x + 3) * 3 - 3) / 3 = 10) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_juans_number_problem_l3121_312146


namespace NUMINAMATH_CALUDE_algebraic_expression_range_l3121_312115

theorem algebraic_expression_range (a : ℝ) : (2 * a - 8) / 3 < 0 → a < 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_range_l3121_312115


namespace NUMINAMATH_CALUDE_propositions_p_and_q_l3121_312185

theorem propositions_p_and_q :
  (¬ ∀ x : ℝ, x^2 ≥ x) ∧ (∃ x : ℝ, x^2 ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_propositions_p_and_q_l3121_312185


namespace NUMINAMATH_CALUDE_thirdYearSelected_l3121_312110

/-- Represents the number of students in each year -/
structure StudentPopulation where
  firstYear : ℕ
  secondYear : ℕ
  thirdYear : ℕ

/-- Calculates the total number of students -/
def totalStudents (pop : StudentPopulation) : ℕ :=
  pop.firstYear + pop.secondYear + pop.thirdYear

/-- Calculates the number of students selected from a specific year -/
def selectedFromYear (pop : StudentPopulation) (year : ℕ) (sampleSize : ℕ) : ℕ :=
  (year * sampleSize) / totalStudents pop

/-- Theorem: The number of third-year students selected in the stratified sampling -/
theorem thirdYearSelected (pop : StudentPopulation) (sampleSize : ℕ) :
  pop.firstYear = 150 →
  pop.secondYear = 120 →
  pop.thirdYear = 180 →
  sampleSize = 50 →
  selectedFromYear pop pop.thirdYear sampleSize = 20 := by
  sorry


end NUMINAMATH_CALUDE_thirdYearSelected_l3121_312110


namespace NUMINAMATH_CALUDE_simplify_and_square_l3121_312197

theorem simplify_and_square : (8 * (15 / 9) * (-45 / 50))^2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_square_l3121_312197


namespace NUMINAMATH_CALUDE_scientists_communication_l3121_312144

/-- A coloring of edges in a complete graph -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A triangle in a graph -/
def Triangle (n : ℕ) := { t : Fin n × Fin n × Fin n // t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2 ∧ t.2.1 ≠ t.2.2 }

/-- A monochromatic triangle under a given coloring -/
def MonochromaticTriangle (n : ℕ) (c : Coloring n) (t : Triangle n) :=
  c t.val.1 t.val.2.1 = c t.val.1 t.val.2.2 ∧
  c t.val.1 t.val.2.1 = c t.val.2.1 t.val.2.2

theorem scientists_communication :
  ∀ c : Coloring 17, ∃ t : Triangle 17, MonochromaticTriangle 17 c t :=
sorry

end NUMINAMATH_CALUDE_scientists_communication_l3121_312144


namespace NUMINAMATH_CALUDE_simplify_fraction_l3121_312126

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3121_312126


namespace NUMINAMATH_CALUDE_electronic_cat_run_time_l3121_312150

/-- Proves that an electronic cat running on a circular track takes 36 seconds to run the last 120 meters -/
theorem electronic_cat_run_time (track_length : ℝ) (speed1 speed2 : ℝ) :
  track_length = 240 →
  speed1 = 5 →
  speed2 = 3 →
  let avg_speed := (speed1 + speed2) / 2
  let total_time := track_length / avg_speed
  let half_time := total_time / 2
  let first_half_distance := speed1 * half_time
  let second_half_distance := track_length - first_half_distance
  let time_at_speed1 := (first_half_distance - track_length / 2) / speed1
  let time_at_speed2 := (track_length / 2 - (first_half_distance - track_length / 2)) / speed2
  (time_at_speed1 + time_at_speed2 : ℝ) = 36 :=
by sorry

end NUMINAMATH_CALUDE_electronic_cat_run_time_l3121_312150


namespace NUMINAMATH_CALUDE_joan_apples_total_l3121_312106

/-- The number of apples Joan has now, given her initial pick and Melanie's gift -/
def total_apples (initial_pick : ℕ) (melanie_gift : ℕ) : ℕ :=
  initial_pick + melanie_gift

/-- Theorem stating that Joan has 70 apples in total -/
theorem joan_apples_total :
  total_apples 43 27 = 70 := by
  sorry

end NUMINAMATH_CALUDE_joan_apples_total_l3121_312106


namespace NUMINAMATH_CALUDE_four_three_eight_nine_has_two_prime_products_l3121_312165

-- Define set C
def C : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 4 * k + 1}

-- Define prime with respect to C
def isPrimeWrtC (k : ℕ) : Prop :=
  k ∈ C ∧ ∀ a b : ℕ, a ∈ C → b ∈ C → k ≠ a * b

-- Define the property of being expressible as product of two primes wrt C in two ways
def hasTwoPrimeProductsInC (n : ℕ) : Prop :=
  ∃ a b c d : ℕ,
    a ≠ c ∧ b ≠ d ∧
    n = a * b ∧ n = c * d ∧
    isPrimeWrtC a ∧ isPrimeWrtC b ∧ isPrimeWrtC c ∧ isPrimeWrtC d

-- Theorem statement
theorem four_three_eight_nine_has_two_prime_products :
  4389 ∈ C ∧ hasTwoPrimeProductsInC 4389 :=
sorry

end NUMINAMATH_CALUDE_four_three_eight_nine_has_two_prime_products_l3121_312165


namespace NUMINAMATH_CALUDE_chairs_per_round_table_l3121_312190

/-- Proves that each round table has 6 chairs in the office canteen -/
theorem chairs_per_round_table : 
  ∀ (x : ℕ),
  (2 * x + 2 * 7 = 26) →
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_chairs_per_round_table_l3121_312190


namespace NUMINAMATH_CALUDE_survey_sampling_suitability_mainland_survey_suitable_l3121_312194

/-- Represents a survey with its characteristics -/
structure Survey where
  population_size : ℕ
  requires_comprehensive : Bool
  is_safety_critical : Bool

/-- Defines when a survey is suitable for sampling -/
def suitable_for_sampling (s : Survey) : Prop :=
  s.population_size > 1000 ∧ ¬s.requires_comprehensive ∧ ¬s.is_safety_critical

/-- Theorem stating the condition for a survey to be suitable for sampling -/
theorem survey_sampling_suitability (s : Survey) :
  suitable_for_sampling s ↔
    s.population_size > 1000 ∧ ¬s.requires_comprehensive ∧ ¬s.is_safety_critical := by
  sorry

/-- The mainland population survey (Option C) -/
def mainland_survey : Survey :=
  { population_size := 1000000,  -- Large population
    requires_comprehensive := false,
    is_safety_critical := false }

/-- Theorem stating that the mainland survey is suitable for sampling -/
theorem mainland_survey_suitable :
  suitable_for_sampling mainland_survey := by
  sorry

end NUMINAMATH_CALUDE_survey_sampling_suitability_mainland_survey_suitable_l3121_312194


namespace NUMINAMATH_CALUDE_circle_radius_order_l3121_312154

theorem circle_radius_order :
  let rA : ℝ := Real.sqrt 50
  let aB : ℝ := 16 * Real.pi
  let cC : ℝ := 10 * Real.pi
  let rB : ℝ := Real.sqrt (aB / Real.pi)
  let rC : ℝ := cC / (2 * Real.pi)
  rB < rC ∧ rC < rA := by sorry

end NUMINAMATH_CALUDE_circle_radius_order_l3121_312154


namespace NUMINAMATH_CALUDE_problem_solution_l3121_312136

/-- The function f defined on real numbers --/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + 2)

/-- f is an odd function --/
axiom f_odd (b : ℝ) : ∀ x, f b (-x) = -(f b x)

theorem problem_solution :
  ∃ b : ℝ,
  (∀ x, f b (-x) = -(f b x)) ∧  -- f is odd
  (b = 1) ∧  -- part 1
  (∀ x y, x < y → f b x > f b y) ∧  -- part 2: f is decreasing
  (∀ k, (∀ t, f b (t^2 - 2*t) + f b (2*t^2 - k) < 0) → k < -1/3)  -- part 3
  := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3121_312136


namespace NUMINAMATH_CALUDE_count_numbers_theorem_l3121_312134

/-- A function that checks if a natural number contains the digit 4 in its decimal representation -/
def contains_four (n : ℕ) : Prop := sorry

/-- The count of numbers from 1 to 1000 that are divisible by 4 and do not contain the digit 4 -/
def count_numbers : ℕ := sorry

/-- Theorem stating that the count of numbers from 1 to 1000 that are divisible by 4 
    and do not contain the digit 4 is equal to 162 -/
theorem count_numbers_theorem : count_numbers = 162 := by sorry

end NUMINAMATH_CALUDE_count_numbers_theorem_l3121_312134


namespace NUMINAMATH_CALUDE_math_course_scheduling_l3121_312157

theorem math_course_scheduling (n : ℕ) (k : ℕ) (courses : ℕ) : 
  n = 6 → k = 3 → courses = 3 →
  (Nat.choose (n - k + 1) k) * (Nat.factorial courses) = 24 := by
sorry

end NUMINAMATH_CALUDE_math_course_scheduling_l3121_312157


namespace NUMINAMATH_CALUDE_purchasing_power_decrease_l3121_312124

def initial_amount : ℝ := 100
def monthly_price_index_increase : ℝ := 0.00465
def months : ℕ := 12

theorem purchasing_power_decrease :
  let final_value := initial_amount * (1 - monthly_price_index_increase) ^ months
  initial_amount - final_value = 4.55 := by
  sorry

end NUMINAMATH_CALUDE_purchasing_power_decrease_l3121_312124


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3121_312177

theorem largest_constant_inequality (C : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ↔ C ≤ 2 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3121_312177


namespace NUMINAMATH_CALUDE_jennifer_gave_away_six_fruits_l3121_312120

/-- Represents the number of fruits Jennifer gave to her sister -/
def fruits_given_away (initial_pears initial_oranges initial_apples fruits_left : ℕ) : ℕ :=
  initial_pears + initial_oranges + initial_apples - fruits_left

/-- Proves that Jennifer gave away 6 fruits -/
theorem jennifer_gave_away_six_fruits :
  ∀ (initial_pears initial_oranges initial_apples fruits_left : ℕ),
    initial_pears = 10 →
    initial_oranges = 20 →
    initial_apples = 2 * initial_pears →
    fruits_left = 44 →
    fruits_given_away initial_pears initial_oranges initial_apples fruits_left = 6 :=
by
  sorry

#check jennifer_gave_away_six_fruits

end NUMINAMATH_CALUDE_jennifer_gave_away_six_fruits_l3121_312120


namespace NUMINAMATH_CALUDE_negation_of_existential_real_equation_l3121_312113

theorem negation_of_existential_real_equation :
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ (∀ x : ℝ, x^3 - 2*x + 1 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_real_equation_l3121_312113


namespace NUMINAMATH_CALUDE_expression_equals_one_l3121_312169

theorem expression_equals_one : 
  (2009 * 2029 + 100) * (1999 * 2039 + 400) / (2019^4 : ℝ) = 1 := by sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3121_312169


namespace NUMINAMATH_CALUDE_jungkook_apples_l3121_312132

theorem jungkook_apples (initial_apples given_apples : ℕ) : 
  initial_apples = 8 → given_apples = 7 → initial_apples + given_apples = 15 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_apples_l3121_312132


namespace NUMINAMATH_CALUDE_colored_paper_problem_l3121_312166

/-- The initial number of colored paper pieces Yuna had -/
def initial_yuna : ℕ := 100

/-- The initial number of colored paper pieces Yoojung had -/
def initial_yoojung : ℕ := 210

/-- The number of pieces Yoojung gave to Yuna -/
def transferred : ℕ := 30

/-- The difference in pieces after the transfer -/
def difference : ℕ := 50

theorem colored_paper_problem :
  initial_yuna = 100 ∧
  initial_yoojung = 210 ∧
  transferred = 30 ∧
  difference = 50 ∧
  initial_yoojung - transferred = initial_yuna + transferred + difference :=
by sorry

end NUMINAMATH_CALUDE_colored_paper_problem_l3121_312166


namespace NUMINAMATH_CALUDE_sum_of_squares_with_hcf_lcm_constraint_l3121_312117

theorem sum_of_squares_with_hcf_lcm_constraint 
  (a b c : ℕ+) 
  (sum_of_squares : a^2 + b^2 + c^2 = 2011)
  (x : ℕ) 
  (hx : x = Nat.gcd a (Nat.gcd b c))
  (y : ℕ) 
  (hy : y = Nat.lcm a (Nat.lcm b c))
  (hxy : x + y = 388) : 
  a + b + c = 61 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_with_hcf_lcm_constraint_l3121_312117


namespace NUMINAMATH_CALUDE_smallest_deletion_for_order_l3121_312139

theorem smallest_deletion_for_order (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℕ, k = n - Int.ceil (Real.sqrt n) ∧
    (∀ perm : List ℕ, perm.length = n → perm.toFinset = Finset.range n →
      ∃ subseq : List ℕ, subseq.length = n - k ∧ 
        (subseq.Sorted (·<·) ∨ subseq.Sorted (·>·)) ∧
        subseq.toFinset ⊆ perm.toFinset) ∧
    (∀ k' : ℕ, k' < k →
      ∃ perm : List ℕ, perm.length = n ∧ perm.toFinset = Finset.range n ∧
        ∀ subseq : List ℕ, subseq.length > n - k' →
          subseq.toFinset ⊆ perm.toFinset →
            ¬(subseq.Sorted (·<·) ∨ subseq.Sorted (·>·))) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_deletion_for_order_l3121_312139


namespace NUMINAMATH_CALUDE_polynomial_expansion_theorem_l3121_312103

/-- Given (2x-1)^5 = ax^5 + bx^4 + cx^3 + dx^2 + ex + f, prove the following statements -/
theorem polynomial_expansion_theorem (a b c d e f : ℝ) :
  (∀ x, (2*x - 1)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  (a + b + c + d + e + f = 1) ∧
  (b + c + d + e = -30) ∧
  (a + c + e = 122) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_theorem_l3121_312103


namespace NUMINAMATH_CALUDE_distance_to_village_l3121_312137

theorem distance_to_village (d : ℝ) : 
  (¬(d ≥ 8) ∧ ¬(d ≤ 7) ∧ ¬(d ≤ 6) ∧ ¬(d ≥ 10)) → 
  (d > 7 ∧ d < 8) :=
sorry

end NUMINAMATH_CALUDE_distance_to_village_l3121_312137


namespace NUMINAMATH_CALUDE_sum_of_four_integers_l3121_312173

theorem sum_of_four_integers (k l m n : ℕ+) 
  (h1 : k + l + m + n = k * m) 
  (h2 : k + l + m + n = l * n) : 
  k + l + m + n = 16 ∨ k + l + m + n = 18 ∨ k + l + m + n = 24 ∨ k + l + m + n = 30 :=
sorry

end NUMINAMATH_CALUDE_sum_of_four_integers_l3121_312173


namespace NUMINAMATH_CALUDE_f_properties_l3121_312162

noncomputable def f (x : ℝ) : ℝ := x / (1 - |x|)

noncomputable def g (x : ℝ) : ℝ := f x + x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (Set.range f = Set.univ) ∧
  (∃! (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ g a = 0 ∧ g b = 0 ∧ g c = 0) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3121_312162


namespace NUMINAMATH_CALUDE_distance_between_points_l3121_312112

theorem distance_between_points (A B C : EuclideanSpace ℝ (Fin 2)) 
  (angle_BAC : Real) (dist_AB : Real) (dist_AC : Real) :
  angle_BAC = 120 * π / 180 →
  dist_AB = 2 →
  dist_AC = 3 →
  ‖B - C‖ = Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3121_312112


namespace NUMINAMATH_CALUDE_games_left_l3121_312158

def initial_games : ℕ := 50
def games_given_away : ℕ := 15

theorem games_left : initial_games - games_given_away = 35 := by
  sorry

end NUMINAMATH_CALUDE_games_left_l3121_312158


namespace NUMINAMATH_CALUDE_min_marbles_for_ten_of_one_color_l3121_312119

/-- Represents the number of marbles of each color in the container -/
structure MarbleContainer :=
  (red : ℕ)
  (green : ℕ)
  (yellow : ℕ)
  (blue : ℕ)
  (white : ℕ)
  (black : ℕ)

/-- Defines the specific container from the problem -/
def problemContainer : MarbleContainer :=
  { red := 30
  , green := 25
  , yellow := 23
  , blue := 15
  , white := 10
  , black := 7 }

/-- 
  Theorem: The minimum number of marbles that must be drawn from the container
  without replacement to ensure that at least 10 marbles of a single color are drawn is 53.
-/
theorem min_marbles_for_ten_of_one_color (container : MarbleContainer := problemContainer) :
  (∃ (n : ℕ), n = 53 ∧
    (∀ (m : ℕ), m < n →
      ∃ (r g y b w bl : ℕ),
        r + g + y + b + w + bl = m ∧
        r ≤ container.red ∧
        g ≤ container.green ∧
        y ≤ container.yellow ∧
        b ≤ container.blue ∧
        w ≤ container.white ∧
        bl ≤ container.black ∧
        r < 10 ∧ g < 10 ∧ y < 10 ∧ b < 10 ∧ w < 10 ∧ bl < 10) ∧
    (∀ (r g y b w bl : ℕ),
      r + g + y + b + w + bl = n →
      r ≤ container.red →
      g ≤ container.green →
      y ≤ container.yellow →
      b ≤ container.blue →
      w ≤ container.white →
      bl ≤ container.black →
      (r ≥ 10 ∨ g ≥ 10 ∨ y ≥ 10 ∨ b ≥ 10 ∨ w ≥ 10 ∨ bl ≥ 10))) :=
by sorry


end NUMINAMATH_CALUDE_min_marbles_for_ten_of_one_color_l3121_312119


namespace NUMINAMATH_CALUDE_division_problem_l3121_312129

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 265 →
  divisor = 22 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 12 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3121_312129


namespace NUMINAMATH_CALUDE_toothpicks_150th_stage_l3121_312135

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ := 4 + 4 * (n - 1)

/-- Theorem: The 150th stage of the pattern contains 600 toothpicks -/
theorem toothpicks_150th_stage : toothpicks 150 = 600 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_150th_stage_l3121_312135


namespace NUMINAMATH_CALUDE_tan_value_from_log_equation_l3121_312168

theorem tan_value_from_log_equation (x : Real) 
  (h1 : x ∈ Set.Ioo 0 (π/2))
  (h2 : Real.log (Real.sin (2*x)) - Real.log (Real.sin x) = Real.log (1/2)) :
  Real.tan x = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_log_equation_l3121_312168


namespace NUMINAMATH_CALUDE_debby_jogging_distance_l3121_312123

/-- Represents the number of kilometers Debby jogged on each day -/
structure JoggingDistance where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ

/-- Theorem stating that given the conditions, Debby jogged 5 kilometers on Tuesday -/
theorem debby_jogging_distance (d : JoggingDistance) 
  (h1 : d.monday = 2)
  (h2 : d.wednesday = 9)
  (h3 : d.monday + d.tuesday + d.wednesday = 16) :
  d.tuesday = 5 := by
  sorry

end NUMINAMATH_CALUDE_debby_jogging_distance_l3121_312123


namespace NUMINAMATH_CALUDE_bottling_probability_l3121_312184

def chocolate_prob : ℚ := 3/4
def vanilla_prob : ℚ := 1/2
def total_days : ℕ := 6
def chocolate_days : ℕ := 4
def vanilla_days : ℕ := 3

theorem bottling_probability : 
  (Nat.choose total_days chocolate_days * chocolate_prob ^ chocolate_days * (1 - chocolate_prob) ^ (total_days - chocolate_days)) *
  (1 - (Nat.choose total_days 0 * vanilla_prob ^ 0 * (1 - vanilla_prob) ^ total_days +
        Nat.choose total_days 1 * vanilla_prob ^ 1 * (1 - vanilla_prob) ^ (total_days - 1) +
        Nat.choose total_days 2 * vanilla_prob ^ 2 * (1 - vanilla_prob) ^ (total_days - 2))) =
  25515/131072 := by
sorry

end NUMINAMATH_CALUDE_bottling_probability_l3121_312184


namespace NUMINAMATH_CALUDE_angle_B_is_80_l3121_312171

-- Define the quadrilateral and its properties
structure Quadrilateral where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  x : ℝ
  BEC : ℝ

-- Define the conditions
def quadrilateral_conditions (q : Quadrilateral) : Prop :=
  q.A = 60 ∧
  q.B = 2 * q.C ∧
  q.D = 2 * q.C - q.x ∧
  q.x > 0 ∧
  q.A + q.B + q.C + q.D = 360 ∧
  q.BEC = 20

-- Theorem statement
theorem angle_B_is_80 (q : Quadrilateral) 
  (h : quadrilateral_conditions q) : q.B = 80 :=
sorry

end NUMINAMATH_CALUDE_angle_B_is_80_l3121_312171


namespace NUMINAMATH_CALUDE_A_symmetric_to_B_about_x_axis_l3121_312178

/-- Two points are symmetric about the x-axis if they have the same x-coordinate
    and their y-coordinates are negatives of each other. -/
def symmetric_about_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- Point A with coordinates (3, 2) -/
def A : ℝ × ℝ := (3, 2)

/-- Point B with coordinates (3, -2) -/
def B : ℝ × ℝ := (3, -2)

/-- Theorem stating that point A is symmetric to point B about the x-axis -/
theorem A_symmetric_to_B_about_x_axis : symmetric_about_x_axis A B := by
  sorry

end NUMINAMATH_CALUDE_A_symmetric_to_B_about_x_axis_l3121_312178


namespace NUMINAMATH_CALUDE_pizza_dough_milk_calculation_l3121_312118

/-- Given a ratio of milk to flour for pizza dough, calculate the amount of milk needed for a given amount of flour -/
theorem pizza_dough_milk_calculation 
  (milk_per_portion : ℝ) 
  (flour_per_portion : ℝ) 
  (total_flour : ℝ) 
  (h1 : milk_per_portion = 50) 
  (h2 : flour_per_portion = 250) 
  (h3 : total_flour = 750) :
  (total_flour / flour_per_portion) * milk_per_portion = 150 :=
by sorry

end NUMINAMATH_CALUDE_pizza_dough_milk_calculation_l3121_312118


namespace NUMINAMATH_CALUDE_tangent_semiperimeter_median_inequality_l3121_312142

variable (a b c : ℝ)
variable (s : ℝ)
variable (ta tb tc : ℝ)
variable (ma mb mc : ℝ)

/-- Triangle inequality --/
axiom triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Semi-perimeter definition --/
axiom semi_perimeter : s = (a + b + c) / 2

/-- Tangent line definitions --/
axiom tangent_a : ta = (2 / (b + c)) * Real.sqrt (b * c * s * (s - a))
axiom tangent_b : tb = (2 / (a + c)) * Real.sqrt (a * c * s * (s - b))
axiom tangent_c : tc = (2 / (a + b)) * Real.sqrt (a * b * s * (s - c))

/-- Median line definitions --/
axiom median_a : ma^2 = (2 * b^2 + 2 * c^2 - a^2) / 4
axiom median_b : mb^2 = (2 * a^2 + 2 * c^2 - b^2) / 4
axiom median_c : mc^2 = (2 * a^2 + 2 * b^2 - c^2) / 4

/-- Theorem: Tangent-Semiperimeter-Median Inequality --/
theorem tangent_semiperimeter_median_inequality :
  ta^2 + tb^2 + tc^2 ≤ s^2 ∧ s^2 ≤ ma^2 + mb^2 + mc^2 :=
sorry

end NUMINAMATH_CALUDE_tangent_semiperimeter_median_inequality_l3121_312142


namespace NUMINAMATH_CALUDE_child_wage_is_eight_l3121_312147

/-- Represents the daily wage structure and worker composition of a building contractor. -/
structure ContractorData where
  male_workers : ℕ
  female_workers : ℕ
  child_workers : ℕ
  male_wage : ℕ
  female_wage : ℕ
  average_wage : ℕ

/-- Calculates the daily wage of a child worker given the contractor's data. -/
def child_worker_wage (data : ContractorData) : ℕ :=
  ((data.average_wage * (data.male_workers + data.female_workers + data.child_workers)) -
   (data.male_wage * data.male_workers + data.female_wage * data.female_workers)) / data.child_workers

/-- Theorem stating that given the specific conditions, the child worker's daily wage is 8 rupees. -/
theorem child_wage_is_eight (data : ContractorData)
  (h1 : data.male_workers = 20)
  (h2 : data.female_workers = 15)
  (h3 : data.child_workers = 5)
  (h4 : data.male_wage = 25)
  (h5 : data.female_wage = 20)
  (h6 : data.average_wage = 21) :
  child_worker_wage data = 8 := by
  sorry


end NUMINAMATH_CALUDE_child_wage_is_eight_l3121_312147


namespace NUMINAMATH_CALUDE_stating_clock_confusion_times_l3121_312108

/-- Represents the number of degrees the hour hand moves per minute -/
def hourHandSpeed : ℝ := 0.5

/-- Represents the number of degrees the minute hand moves per minute -/
def minuteHandSpeed : ℝ := 6

/-- Represents the total number of degrees in a circle -/
def totalDegrees : ℕ := 360

/-- Represents the number of hours in the given time period -/
def timePeriod : ℕ := 12

/-- Represents the number of times the hands overlap in the given time period -/
def overlapTimes : ℕ := 11

/-- 
  Theorem stating that there are 132 times in a 12-hour period when the clock hands
  can be mistaken for each other, excluding overlaps.
-/
theorem clock_confusion_times : 
  ∃ (confusionTimes : ℕ), 
    confusionTimes = timePeriod * (totalDegrees / (minuteHandSpeed - hourHandSpeed) - 1) - overlapTimes := by
  sorry

end NUMINAMATH_CALUDE_stating_clock_confusion_times_l3121_312108


namespace NUMINAMATH_CALUDE_X_mod_100_l3121_312138

/-- The number of sequences satisfying the given conditions -/
def X : ℕ := sorry

/-- Condition: Each aᵢ is either 0 or a power of 2 -/
def is_valid_element (a : ℕ) : Prop :=
  a = 0 ∨ ∃ k : ℕ, a = 2^k

/-- Condition: aᵢ = a₂ᵢ + a₂ᵢ₊₁ for 1 ≤ i ≤ 1023 -/
def satisfies_sum_condition (a : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 1023 → a i = a (2*i) + a (2*i + 1)

/-- All conditions for the sequence -/
def valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 2047 → is_valid_element (a i)) ∧
  satisfies_sum_condition a ∧
  a 1 = 1024

theorem X_mod_100 : X % 100 = 15 := by sorry

end NUMINAMATH_CALUDE_X_mod_100_l3121_312138


namespace NUMINAMATH_CALUDE_zero_in_interval_implies_a_range_l3121_312161

/-- The function f(x) = -x^2 + 4x + a has a zero in the interval [-3, 3] -/
def has_zero_in_interval (f : ℝ → ℝ) : Prop :=
  ∃ x ∈ Set.Icc (-3) 3, f x = 0

/-- The main theorem: if f(x) = -x^2 + 4x + a has a zero in [-3, 3], then a ∈ [-3, 21] -/
theorem zero_in_interval_implies_a_range (a : ℝ) :
  has_zero_in_interval (fun x => -x^2 + 4*x + a) → a ∈ Set.Icc (-3) 21 := by
  sorry


end NUMINAMATH_CALUDE_zero_in_interval_implies_a_range_l3121_312161


namespace NUMINAMATH_CALUDE_tangent_line_equation_no_collinear_intersection_l3121_312164

-- Define the line l
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y = k * x + 2}

-- Define the circle Q
def circle_Q : Set (ℝ × ℝ) :=
  {(x, y) | x^2 + y^2 - 12*x + 32 = 0}

-- Define the point P
def point_P : ℝ × ℝ := (0, 2)

-- Define the center of the circle Q
def center_Q : ℝ × ℝ := (6, 0)

-- Define the tangency condition
def is_tangent (k : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) ∈ line_l k ∩ circle_Q ∧
  ∀ (x' y' : ℝ), (x', y') ∈ line_l k ∩ circle_Q → (x', y') = (x, y)

-- Define the intersection condition
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ line_l k ∩ circle_Q ∧
  (x₂, y₂) ∈ line_l k ∩ circle_Q ∧ (x₁, y₁) ≠ (x₂, y₂)

-- Define collinearity condition
def are_collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), B - A = t • (C - A)

theorem tangent_line_equation :
  ∀ k : ℝ, is_tangent k →
  (∀ x y : ℝ, (x, y) ∈ line_l k ↔ y = 2 ∨ 3*x + 4*y = 8) :=
sorry

theorem no_collinear_intersection :
  ¬∃ k : ℝ, intersects_at_two_points k ∧
  (∀ A B : ℝ × ℝ, A ∈ circle_Q → B ∈ circle_Q → A ≠ B →
   are_collinear (0, 0) (A.1 + B.1, A.2 + B.2) (6, -2)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_no_collinear_intersection_l3121_312164


namespace NUMINAMATH_CALUDE_correct_outfits_l3121_312140

-- Define the colors
inductive Color
| Red
| Blue

-- Define the clothing types
inductive ClothingType
| Tshirt
| Shorts

-- Define a structure for a child's outfit
structure Outfit :=
  (tshirt : Color)
  (shorts : Color)

-- Define the children
inductive Child
| Alyna
| Bohdan
| Vika
| Grysha

def outfit : Child → Outfit
| Child.Alyna => ⟨Color.Red, Color.Red⟩
| Child.Bohdan => ⟨Color.Red, Color.Blue⟩
| Child.Vika => ⟨Color.Blue, Color.Blue⟩
| Child.Grysha => ⟨Color.Red, Color.Blue⟩

theorem correct_outfits :
  (outfit Child.Alyna).tshirt = Color.Red ∧
  (outfit Child.Bohdan).tshirt = Color.Red ∧
  (outfit Child.Alyna).shorts ≠ (outfit Child.Bohdan).shorts ∧
  (outfit Child.Vika).tshirt ≠ (outfit Child.Grysha).tshirt ∧
  (outfit Child.Vika).shorts = Color.Blue ∧
  (outfit Child.Grysha).shorts = Color.Blue ∧
  (outfit Child.Alyna).tshirt ≠ (outfit Child.Vika).tshirt ∧
  (outfit Child.Alyna).shorts ≠ (outfit Child.Vika).shorts ∧
  (∀ c : Child, (outfit c).tshirt = Color.Red ∨ (outfit c).tshirt = Color.Blue) ∧
  (∀ c : Child, (outfit c).shorts = Color.Red ∨ (outfit c).shorts = Color.Blue) :=
by sorry

#check correct_outfits

end NUMINAMATH_CALUDE_correct_outfits_l3121_312140


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3121_312179

theorem complex_equation_solution (z : ℂ) : z * (1 - Complex.I) = 2 → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3121_312179


namespace NUMINAMATH_CALUDE_point_D_coordinates_l3121_312111

def vector_AB : ℝ × ℝ := (5, -3)
def point_C : ℝ × ℝ := (-1, 3)

theorem point_D_coordinates :
  ∀ (D : ℝ × ℝ),
  (D.1 - point_C.1, D.2 - point_C.2) = (2 * vector_AB.1, 2 * vector_AB.2) →
  D = (9, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_D_coordinates_l3121_312111


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3121_312130

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + x - 2 ≥ 0)) ↔ (∃ x : ℝ, x^2 + x - 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3121_312130


namespace NUMINAMATH_CALUDE_money_sharing_l3121_312189

theorem money_sharing (john jose binoy total : ℕ) : 
  john + jose + binoy = total →
  2 * jose = 4 * john →
  3 * binoy = 6 * john →
  john = 1440 →
  total = 8640 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l3121_312189


namespace NUMINAMATH_CALUDE_function_values_l3121_312107

def is_prime (n : ℕ) : Prop := sorry

def coprime (a b : ℕ) : Prop := sorry

def number_theory_function (f : ℕ → ℕ) : Prop :=
  (∀ a b, coprime a b → f (a * b) = f a * f b) ∧
  (∀ p q, is_prime p → is_prime q → f (p + q) = f p + f q)

theorem function_values (f : ℕ → ℕ) (h : number_theory_function f) :
  f 2 = 2 ∧ f 3 = 3 ∧ f 1999 = 1999 := by sorry

end NUMINAMATH_CALUDE_function_values_l3121_312107


namespace NUMINAMATH_CALUDE_total_cost_is_24_l3121_312176

/-- The cost of a burger meal and soda order for two people -/
def total_cost (burger_price : ℚ) : ℚ :=
  let soda_price := burger_price / 3
  let paulo_total := burger_price + soda_price
  let jeremy_total := 2 * paulo_total
  paulo_total + jeremy_total

/-- Theorem: The total cost of Paulo and Jeremy's orders is $24 when a burger meal costs $6 -/
theorem total_cost_is_24 : total_cost 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_24_l3121_312176


namespace NUMINAMATH_CALUDE_janinas_pancakes_l3121_312167

/-- Calculates the number of pancakes Janina must sell daily to cover her expenses -/
theorem janinas_pancakes (daily_rent : ℕ) (daily_supplies : ℕ) (price_per_pancake : ℕ) :
  daily_rent = 30 →
  daily_supplies = 12 →
  price_per_pancake = 2 →
  (daily_rent + daily_supplies) / price_per_pancake = 21 := by
  sorry

#check janinas_pancakes

end NUMINAMATH_CALUDE_janinas_pancakes_l3121_312167


namespace NUMINAMATH_CALUDE_parabola_properties_l3121_312199

def parabola (x : ℝ) : ℝ := x^2 - 4*x - 4

theorem parabola_properties :
  (∀ x : ℝ, parabola x ≥ parabola 2) ∧
  (∀ x : ℝ, parabola x = parabola (4 - x)) ∧
  (parabola 2 = -8) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l3121_312199


namespace NUMINAMATH_CALUDE_star_property_l3121_312196

/-- Operation ⋆ for positive real numbers -/
noncomputable def star (k : ℝ) (x y : ℝ) : ℝ := x^y * k

/-- Theorem stating the properties of the ⋆ operation and the result to be proved -/
theorem star_property (k : ℝ) :
  (k > 0) →
  (∀ x y, x > 0 → y > 0 → (star k (x^y) y = x * star k y y)) →
  (∀ x, x > 0 → star k (star k x 1) x = star k x 1) →
  (star k 1 1 = k) →
  (star k 2 3 = 8 * k) := by
  sorry

end NUMINAMATH_CALUDE_star_property_l3121_312196


namespace NUMINAMATH_CALUDE_five_digit_number_product_l3121_312149

theorem five_digit_number_product (a b c d e : Nat) : 
  a ≠ 0 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
  (10 * a + b + 10 * b + c) * 
  (10 * b + c + 10 * c + d) * 
  (10 * c + d + 10 * d + e) = 157605 →
  (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 ∧ e = 5) ∨
  (a = 2 ∧ b = 1 ∧ c = 4 ∧ d = 3 ∧ e = 6) := by
  sorry

end NUMINAMATH_CALUDE_five_digit_number_product_l3121_312149


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3121_312193

theorem absolute_value_equation_solution :
  ∃! y : ℝ, abs (y - 4) + 3 * y = 15 :=
by
  -- The unique solution is y = 4.75
  use 4.75
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3121_312193


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3121_312122

theorem negation_of_proposition (p : Prop) : 
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 3*x + 2 > 0)) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 3*x + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3121_312122


namespace NUMINAMATH_CALUDE_price_per_bracelet_l3121_312180

/-- Represents the problem of determining the price per bracelet --/
def bracelet_problem (total_cost : ℕ) (selling_period_weeks : ℕ) (avg_daily_sales : ℕ) : Prop :=
  let total_days : ℕ := selling_period_weeks * 7
  let total_bracelets : ℕ := total_days * avg_daily_sales
  total_bracelets = total_cost ∧ (total_cost : ℚ) / total_bracelets = 1

/-- Proves that the price per bracelet is $1 given the problem conditions --/
theorem price_per_bracelet :
  bracelet_problem 112 2 8 :=
by sorry

end NUMINAMATH_CALUDE_price_per_bracelet_l3121_312180


namespace NUMINAMATH_CALUDE_opposite_numbers_l3121_312143

theorem opposite_numbers : -5^2 = -((-5)^2) := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_l3121_312143


namespace NUMINAMATH_CALUDE_no_real_sqrt_negative_four_l3121_312114

theorem no_real_sqrt_negative_four :
  ¬ ∃ (x : ℝ), x^2 = -4 := by
sorry

end NUMINAMATH_CALUDE_no_real_sqrt_negative_four_l3121_312114


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_l3121_312100

/-- Given a point A with coordinates (-3, 2), its symmetric point
    with respect to the y-axis has coordinates (3, 2). -/
theorem symmetric_point_y_axis :
  let A : ℝ × ℝ := (-3, 2)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
  symmetric_point A = (3, 2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_l3121_312100


namespace NUMINAMATH_CALUDE_beth_crayon_packs_l3121_312153

theorem beth_crayon_packs :
  ∀ (total_crayons : ℕ) (crayons_per_pack : ℕ) (extra_crayons : ℕ),
    total_crayons = 40 →
    crayons_per_pack = 10 →
    extra_crayons = 6 →
    (total_crayons - extra_crayons) / crayons_per_pack = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_beth_crayon_packs_l3121_312153


namespace NUMINAMATH_CALUDE_equation_one_root_range_l3121_312175

-- Define the equation
def equation (k : ℝ) (x : ℝ) : Prop :=
  Real.log (k * x) = 2 * Real.log (x + 1)

-- Define the condition for having only one real root
def has_one_real_root (k : ℝ) : Prop :=
  ∃! x : ℝ, equation k x

-- Define the range of k
def k_range : Set ℝ :=
  Set.Iio 0 ∪ {4}

-- Theorem statement
theorem equation_one_root_range :
  {k : ℝ | has_one_real_root k} = k_range :=
sorry

end NUMINAMATH_CALUDE_equation_one_root_range_l3121_312175


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3121_312156

/-- A circle M with center (a, 2) and radius 2 -/
def circle_M (a x y : ℝ) : Prop := (x - a)^2 + (y - 2)^2 = 4

/-- A line l with equation x - y + 3 = 0 -/
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

/-- The chord intercepted by line l on circle M has a length of 4 -/
def chord_length_4 (a : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ,
  circle_M a x₁ y₁ ∧ circle_M a x₂ y₂ ∧
  line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16

theorem circle_line_intersection (a : ℝ) :
  circle_M a a 2 ∧ line_l a 2 ∧ chord_length_4 a → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3121_312156


namespace NUMINAMATH_CALUDE_angle_on_diagonal_line_l3121_312186

/-- If the terminal side of angle α lies on the line y = x, then α = kπ + π/4 for some integer k. -/
theorem angle_on_diagonal_line (α : ℝ) :
  (∃ (x y : ℝ), x = y ∧ x = Real.cos α ∧ y = Real.sin α) →
  ∃ (k : ℤ), α = k * π + π / 4 := by sorry

end NUMINAMATH_CALUDE_angle_on_diagonal_line_l3121_312186


namespace NUMINAMATH_CALUDE_function_properties_l3121_312131

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem function_properties (a : ℝ) :
  (∀ x : ℝ, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) →
  (a = 2 ∧ 
   ∀ x : ℝ, f 2 (3*x) + f 2 (x+3) ≥ 5/3 ∧
   ∃ x : ℝ, f 2 (3*x) + f 2 (x+3) = 5/3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3121_312131


namespace NUMINAMATH_CALUDE_fraction_subtraction_simplification_l3121_312155

theorem fraction_subtraction_simplification :
  (7 : ℚ) / 17 - (4 : ℚ) / 51 = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_simplification_l3121_312155
