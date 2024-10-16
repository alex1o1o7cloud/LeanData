import Mathlib

namespace NUMINAMATH_CALUDE_length_to_width_ratio_l1589_158965

def field_perimeter : ℝ := 384
def field_width : ℝ := 80

theorem length_to_width_ratio :
  let field_length := (field_perimeter - 2 * field_width) / 2
  field_length / field_width = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_length_to_width_ratio_l1589_158965


namespace NUMINAMATH_CALUDE_total_earning_calculation_l1589_158908

theorem total_earning_calculation (days_a days_b days_c : ℕ) 
  (wage_ratio_a wage_ratio_b wage_ratio_c : ℕ) (wage_c : ℕ) :
  days_a = 6 →
  days_b = 9 →
  days_c = 4 →
  wage_ratio_a = 3 →
  wage_ratio_b = 4 →
  wage_ratio_c = 5 →
  wage_c = 110 →
  (days_a * (wage_c * wage_ratio_a / wage_ratio_c) +
   days_b * (wage_c * wage_ratio_b / wage_ratio_c) +
   days_c * wage_c) = 1628 :=
by sorry

end NUMINAMATH_CALUDE_total_earning_calculation_l1589_158908


namespace NUMINAMATH_CALUDE_triangular_front_view_solids_l1589_158983

/-- Enumeration of possible solids --/
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

/-- Definition of a solid having a triangular front view --/
def hasTriangularFrontView (s : Solid) : Prop :=
  match s with
  | Solid.TriangularPyramid => True
  | Solid.SquarePyramid => True
  | Solid.TriangularPrism => True  -- Assuming it can be laid on its side
  | Solid.Cone => True
  | _ => False

/-- Theorem stating that a solid with a triangular front view must be one of the specified solids --/
theorem triangular_front_view_solids (s : Solid) :
  hasTriangularFrontView s →
  s = Solid.TriangularPyramid ∨
  s = Solid.SquarePyramid ∨
  s = Solid.TriangularPrism ∨
  s = Solid.Cone :=
by
  sorry

end NUMINAMATH_CALUDE_triangular_front_view_solids_l1589_158983


namespace NUMINAMATH_CALUDE_original_number_is_429_l1589_158922

/-- Given a three-digit number abc, this function returns the sum of all its permutations -/
def sum_of_permutations (a b c : Nat) : Nat :=
  100 * a + 10 * b + c +
  100 * a + 10 * c + b +
  100 * b + 10 * a + c +
  100 * b + 10 * c + a +
  100 * c + 10 * a + b +
  100 * c + 10 * b + a

/-- The sum of all permutations of the three-digit number we're looking for -/
def S : Nat := 4239

/-- Theorem stating that the original three-digit number is 429 -/
theorem original_number_is_429 :
  ∃ (a b c : Nat), a < 10 ∧ b < 10 ∧ c < 10 ∧ a ≠ 0 ∧ sum_of_permutations a b c = S ∧ a = 4 ∧ b = 2 ∧ c = 9 := by
  sorry


end NUMINAMATH_CALUDE_original_number_is_429_l1589_158922


namespace NUMINAMATH_CALUDE_uncovered_area_is_eight_l1589_158989

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ

/-- Represents the side length of a square block -/
def BlockSide : ℝ := 4

/-- Calculates the area of a rectangular box -/
def boxArea (box : BoxDimensions) : ℝ :=
  box.length * box.width

/-- Calculates the area of a square block -/
def blockArea (side : ℝ) : ℝ :=
  side * side

/-- Theorem: The uncovered area in the box is 8 square inches -/
theorem uncovered_area_is_eight (box : BoxDimensions)
    (h1 : box.length = 6)
    (h2 : box.width = 4) :
    boxArea box - blockArea BlockSide = 8 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_area_is_eight_l1589_158989


namespace NUMINAMATH_CALUDE_max_production_in_seven_days_l1589_158959

/-- Represents the daily production capacity of a group -/
structure ProductionCapacity where
  shirts : ℕ
  trousers : ℕ

/-- Represents the production assignment for a group -/
structure ProductionAssignment where
  shirtDays : ℕ
  trouserDays : ℕ

/-- Calculates the total production of a group given its capacity and assignment -/
def totalProduction (capacity : ProductionCapacity) (assignment : ProductionAssignment) : ℕ × ℕ :=
  (capacity.shirts * assignment.shirtDays, capacity.trousers * assignment.trouserDays)

/-- Theorem: Maximum production of matching sets in 7 days -/
theorem max_production_in_seven_days 
  (groupA groupB groupC groupD : ProductionCapacity)
  (assignmentA assignmentB assignmentC assignmentD : ProductionAssignment)
  (h1 : assignmentA.shirtDays + assignmentA.trouserDays = 7)
  (h2 : assignmentB.shirtDays + assignmentB.trouserDays = 7)
  (h3 : assignmentC.shirtDays + assignmentC.trouserDays = 7)
  (h4 : assignmentD.shirtDays + assignmentD.trouserDays = 7)
  (h5 : groupA.shirts = 8 ∧ groupA.trousers = 10)
  (h6 : groupB.shirts = 9 ∧ groupB.trousers = 12)
  (h7 : groupC.shirts = 7 ∧ groupC.trousers = 11)
  (h8 : groupD.shirts = 6 ∧ groupD.trousers = 7) :
  (∃ (assignmentA assignmentB assignmentC assignmentD : ProductionAssignment),
    let (shirtsTotalA, trousersTotalA) := totalProduction groupA assignmentA
    let (shirtsTotalB, trousersTotalB) := totalProduction groupB assignmentB
    let (shirtsTotalC, trousersTotalC) := totalProduction groupC assignmentC
    let (shirtsTotalD, trousersTotalD) := totalProduction groupD assignmentD
    let shirtsTotal := shirtsTotalA + shirtsTotalB + shirtsTotalC + shirtsTotalD
    let trousersTotal := trousersTotalA + trousersTotalB + trousersTotalC + trousersTotalD
    min shirtsTotal trousersTotal = 125) :=
by sorry

end NUMINAMATH_CALUDE_max_production_in_seven_days_l1589_158959


namespace NUMINAMATH_CALUDE_sum_of_smallest_multiples_l1589_158938

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def smallest_two_digit_multiple_of_5 (a : ℕ) : Prop :=
  is_two_digit a ∧ 5 ∣ a ∧ ∀ m : ℕ, is_two_digit m → 5 ∣ m → a ≤ m

def smallest_three_digit_multiple_of_7 (b : ℕ) : Prop :=
  is_three_digit b ∧ 7 ∣ b ∧ ∀ m : ℕ, is_three_digit m → 7 ∣ m → b ≤ m

theorem sum_of_smallest_multiples (a b : ℕ) :
  smallest_two_digit_multiple_of_5 a →
  smallest_three_digit_multiple_of_7 b →
  a + b = 115 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_smallest_multiples_l1589_158938


namespace NUMINAMATH_CALUDE_article_price_l1589_158968

theorem article_price (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  final_price = 126 ∧ discount1 = 0.1 ∧ discount2 = 0.2 →
  ∃ (original_price : ℝ), original_price = 175 ∧ 
    final_price = original_price * (1 - discount1) * (1 - discount2) :=
by
  sorry

end NUMINAMATH_CALUDE_article_price_l1589_158968


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l1589_158928

def n : ℕ := 20  -- Total number of knights
def k : ℕ := 4   -- Number of knights chosen

-- Probability that at least two of the four chosen knights were sitting next to each other
def adjacent_probability : ℚ :=
  1 - (Nat.choose (n - k) (k - 1) : ℚ) / (Nat.choose n k : ℚ)

theorem adjacent_knights_probability :
  adjacent_probability = 66 / 75 :=
sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l1589_158928


namespace NUMINAMATH_CALUDE_initial_water_amount_l1589_158934

theorem initial_water_amount (initial_amount : ℝ) : 
  initial_amount + 6.8 = 9.8 → initial_amount = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_amount_l1589_158934


namespace NUMINAMATH_CALUDE_smallest_n_exceeding_1500_l1589_158978

def exterior_sum (n : ℕ) : ℕ := 8 + 24 * (n - 2) + 12 * (n - 2)^2

theorem smallest_n_exceeding_1500 :
  ∀ n : ℕ, n ≥ 13 ↔ exterior_sum n > 1500 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_exceeding_1500_l1589_158978


namespace NUMINAMATH_CALUDE_weight_of_water_moles_l1589_158982

/-- The atomic weight of hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.008

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of hydrogen atoms in a water molecule -/
def H_count : ℕ := 2

/-- The number of oxygen atoms in a water molecule -/
def O_count : ℕ := 1

/-- The number of moles of water -/
def moles_of_water : ℝ := 4

/-- The molecular weight of water (H2O) in g/mol -/
def molecular_weight_H2O : ℝ := H_count * atomic_weight_H + O_count * atomic_weight_O

theorem weight_of_water_moles : 
  moles_of_water * molecular_weight_H2O = 72.064 := by sorry

end NUMINAMATH_CALUDE_weight_of_water_moles_l1589_158982


namespace NUMINAMATH_CALUDE_cube_cutting_surface_area_l1589_158900

/-- Calculates the total surface area of pieces after cutting a cube -/
def total_surface_area_after_cutting (edge_length : ℝ) (horizontal_cuts : ℕ) (vertical_cuts : ℕ) : ℝ :=
  let original_surface_area := 6 * edge_length^2
  let new_horizontal_faces := 2 * edge_length^2 * (2 * horizontal_cuts : ℝ)
  let new_vertical_faces := 2 * edge_length^2 * (2 * vertical_cuts : ℝ)
  original_surface_area + new_horizontal_faces + new_vertical_faces

/-- Theorem: The total surface area of pieces after cutting a 2-decimeter cube 4 times horizontally and 5 times vertically is 96 square decimeters -/
theorem cube_cutting_surface_area :
  total_surface_area_after_cutting 2 4 5 = 96 := by
  sorry

end NUMINAMATH_CALUDE_cube_cutting_surface_area_l1589_158900


namespace NUMINAMATH_CALUDE_symmetric_points_on_circumcircle_l1589_158920

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the orthocenter of a triangle
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a point symmetric to another point with respect to a line
def symmetric_point (p : ℝ × ℝ) (l : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Define the circumcircle of a triangle
def circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

-- Main theorem
theorem symmetric_points_on_circumcircle (t : Triangle) :
  let H := orthocenter t
  let A1 := symmetric_point H (t.B, t.C)
  let B1 := symmetric_point H (t.C, t.A)
  let C1 := symmetric_point H (t.A, t.B)
  A1 ∈ circumcircle t ∧ B1 ∈ circumcircle t ∧ C1 ∈ circumcircle t := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_on_circumcircle_l1589_158920


namespace NUMINAMATH_CALUDE_point_distance_from_origin_l1589_158970

theorem point_distance_from_origin (A : ℝ) : 
  (|A - 0| = 4) → (A = 4 ∨ A = -4) := by
  sorry

end NUMINAMATH_CALUDE_point_distance_from_origin_l1589_158970


namespace NUMINAMATH_CALUDE_average_of_numbers_l1589_158949

def numbers : List ℝ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1, 1252140]

theorem average_of_numbers :
  (numbers.sum / numbers.length : ℝ) = 125397.5 ∧
  (numbers.sum / numbers.length : ℝ) ≠ 858.5454545454545 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1589_158949


namespace NUMINAMATH_CALUDE_merchant_articles_l1589_158913

/-- The number of articles a merchant has, given profit percentage and price relationship -/
theorem merchant_articles (N : ℕ) (profit_percentage : ℚ) : 
  profit_percentage = 25 / 400 →
  (N : ℚ) * (1 : ℚ) = 16 * (1 + profit_percentage) →
  N = 17 := by
  sorry

#check merchant_articles

end NUMINAMATH_CALUDE_merchant_articles_l1589_158913


namespace NUMINAMATH_CALUDE_reciprocal_of_complex_l1589_158966

/-- Given a complex number z = -1 + √3i, prove that its reciprocal is -1/4 - (√3/4)i -/
theorem reciprocal_of_complex (z : ℂ) : 
  z = -1 + Complex.I * Real.sqrt 3 → 
  z⁻¹ = -(1/4 : ℂ) - Complex.I * ((Real.sqrt 3)/4) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_complex_l1589_158966


namespace NUMINAMATH_CALUDE_factorization_of_2a_5_minus_8a_l1589_158907

theorem factorization_of_2a_5_minus_8a (a : ℝ) : 
  2 * a^5 - 8 * a = 2 * a * (a^2 + 2) * (a + Real.sqrt 2) * (a - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2a_5_minus_8a_l1589_158907


namespace NUMINAMATH_CALUDE_sallys_payment_l1589_158909

/-- Proves that the amount Sally paid with is $20, given that she bought 3 frames at $3 each and received $11 in change. -/
theorem sallys_payment (num_frames : ℕ) (frame_cost : ℕ) (change : ℕ) : 
  num_frames = 3 → frame_cost = 3 → change = 11 → 
  num_frames * frame_cost + change = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_sallys_payment_l1589_158909


namespace NUMINAMATH_CALUDE_concatenation_product_relation_l1589_158999

theorem concatenation_product_relation :
  ∃! (x y : ℕ), 10 ≤ x ∧ x ≤ 99 ∧ 100 ≤ y ∧ y ≤ 999 ∧ 
  1000 * x + y = 11 * x * y ∧ x + y = 110 := by
sorry

end NUMINAMATH_CALUDE_concatenation_product_relation_l1589_158999


namespace NUMINAMATH_CALUDE_roots_square_sum_l1589_158925

-- Define the quadratic equation
def quadratic (x : ℝ) : ℝ := x^2 - 3*x + 1

-- Define the roots
theorem roots_square_sum : 
  ∀ r s : ℝ, quadratic r = 0 → quadratic s = 0 → r^2 + s^2 = 7 :=
by
  sorry

#check roots_square_sum

end NUMINAMATH_CALUDE_roots_square_sum_l1589_158925


namespace NUMINAMATH_CALUDE_fly_path_length_l1589_158916

theorem fly_path_length (r : ℝ) (path_end : ℝ) (h1 : r = 100) (h2 : path_end = 120) : 
  let diameter := 2 * r
  let chord := Real.sqrt (diameter^2 - path_end^2)
  diameter + chord + path_end = 480 := by sorry

end NUMINAMATH_CALUDE_fly_path_length_l1589_158916


namespace NUMINAMATH_CALUDE_wall_building_time_l1589_158902

/-- Given that 60 workers can build a wall in 3 days, prove that 30 workers 
    will take 6 days to build the same wall, assuming consistent work rate and conditions. -/
theorem wall_building_time 
  (workers_initial : ℕ) 
  (days_initial : ℕ) 
  (workers_new : ℕ) 
  (h1 : workers_initial = 60) 
  (h2 : days_initial = 3) 
  (h3 : workers_new = 30) :
  (workers_initial * days_initial) / workers_new = 6 := by
sorry

end NUMINAMATH_CALUDE_wall_building_time_l1589_158902


namespace NUMINAMATH_CALUDE_bee_colony_fraction_l1589_158931

theorem bee_colony_fraction (initial_bees : ℕ) (daily_loss : ℕ) (days : ℕ) :
  initial_bees = 80000 →
  daily_loss = 1200 →
  days = 50 →
  (initial_bees - daily_loss * days) / initial_bees = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_bee_colony_fraction_l1589_158931


namespace NUMINAMATH_CALUDE_larger_number_problem_l1589_158951

theorem larger_number_problem (L S : ℕ) (hL : L > S) : 
  L - S = 2415 → L = 21 * S + 15 → L = 2535 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1589_158951


namespace NUMINAMATH_CALUDE_corvette_trip_speed_l1589_158939

theorem corvette_trip_speed (total_distance : ℝ) (average_speed : ℝ) 
  (h1 : total_distance = 640)
  (h2 : average_speed = 40) : ℝ :=
  let first_half_distance := total_distance / 2
  let second_half_time_ratio := 3
  let first_half_speed := 
    (2 * total_distance * average_speed) / (total_distance + 2 * first_half_distance)
  have h3 : first_half_speed = 80 := by sorry
  first_half_speed

#check corvette_trip_speed

end NUMINAMATH_CALUDE_corvette_trip_speed_l1589_158939


namespace NUMINAMATH_CALUDE_determinant_of_2x2_matrix_l1589_158932

theorem determinant_of_2x2_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![7, -2; 3, 5]
  Matrix.det A = 41 := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_2x2_matrix_l1589_158932


namespace NUMINAMATH_CALUDE_joker_spade_probability_l1589_158980

/-- Custom deck of cards -/
structure CustomDeck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (jokers : ℕ)
  (cards_per_suit : ℕ)

/-- Properties of the custom deck -/
def custom_deck_properties (d : CustomDeck) : Prop :=
  d.total_cards = 60 ∧
  d.ranks = 15 ∧
  d.suits = 4 ∧
  d.jokers = 4 ∧
  d.cards_per_suit = 15

/-- Probability of drawing a Joker first and any spade second -/
def joker_spade_prob (d : CustomDeck) : ℚ :=
  224 / 885

/-- Theorem stating the probability of drawing a Joker first and any spade second -/
theorem joker_spade_probability (d : CustomDeck) 
  (h : custom_deck_properties d) : 
  joker_spade_prob d = 224 / 885 := by
  sorry

end NUMINAMATH_CALUDE_joker_spade_probability_l1589_158980


namespace NUMINAMATH_CALUDE_base_conversion_problem_l1589_158946

/-- Convert a number from base 6 to base 10 -/
def base6To10 (n : Nat) : Nat :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Check if a number is a valid base-10 digit -/
def isBase10Digit (n : Nat) : Prop := n < 10

theorem base_conversion_problem :
  ∀ c d : Nat,
  isBase10Digit c →
  isBase10Digit d →
  base6To10 524 = 2 * (10 * c + d) →
  (c * d : ℚ) / 12 = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l1589_158946


namespace NUMINAMATH_CALUDE_equation_solution_l1589_158954

theorem equation_solution : ∃! x : ℚ, 
  (1 : ℚ) / ((x + 12)^2) + (1 : ℚ) / ((x + 8)^2) = 
  (1 : ℚ) / ((x + 13)^2) + (1 : ℚ) / ((x + 7)^2) ∧ 
  x = -15/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1589_158954


namespace NUMINAMATH_CALUDE_purple_valley_skirts_l1589_158940

def azure_skirts : ℕ := 60

def seafoam_skirts (azure : ℕ) : ℕ := (2 * azure) / 3

def purple_skirts (seafoam : ℕ) : ℕ := seafoam / 4

theorem purple_valley_skirts : 
  purple_skirts (seafoam_skirts azure_skirts) = 10 := by
  sorry

end NUMINAMATH_CALUDE_purple_valley_skirts_l1589_158940


namespace NUMINAMATH_CALUDE_sin_fourth_powers_sum_l1589_158929

theorem sin_fourth_powers_sum : 
  Real.sin (π / 8) ^ 4 + Real.sin (3 * π / 8) ^ 4 + 
  Real.sin (5 * π / 8) ^ 4 + Real.sin (7 * π / 8) ^ 4 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_fourth_powers_sum_l1589_158929


namespace NUMINAMATH_CALUDE_gas_volume_ranking_l1589_158974

/-- Gas volume per capita for a region -/
structure GasVolume where
  region : String
  volume : Float

/-- Theorem: Russia has the highest gas volume per capita, followed by Non-West, then West -/
theorem gas_volume_ranking (west non_west russia : GasVolume) 
  (h_west : west.region = "West" ∧ west.volume = 21428)
  (h_non_west : non_west.region = "Non-West" ∧ non_west.volume = 26848.55)
  (h_russia : russia.region = "Russia" ∧ russia.volume = 302790.13) :
  russia.volume > non_west.volume ∧ non_west.volume > west.volume :=
by sorry

end NUMINAMATH_CALUDE_gas_volume_ranking_l1589_158974


namespace NUMINAMATH_CALUDE_good_games_count_l1589_158904

def games_from_friend : ℕ := 11
def games_from_garage_sale : ℕ := 22
def non_working_games : ℕ := 19

theorem good_games_count :
  games_from_friend + games_from_garage_sale - non_working_games = 14 := by
  sorry

end NUMINAMATH_CALUDE_good_games_count_l1589_158904


namespace NUMINAMATH_CALUDE_fifth_term_sum_l1589_158921

def sequence_a (n : ℕ) : ℕ := 2 * n - 1

def sequence_b (n : ℕ) : ℕ := 2^(n-1)

def sequence_c (n : ℕ) : ℕ := sequence_a n * sequence_b n

theorem fifth_term_sum :
  sequence_a 5 + sequence_b 5 + sequence_c 5 = 169 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_sum_l1589_158921


namespace NUMINAMATH_CALUDE_highest_water_level_in_narrow_neck_vase_l1589_158956

/-- Represents a vase with a specific shape --/
inductive VaseShape
  | NarrowNeck
  | Symmetrical
  | WideTop

/-- Represents a vase with its properties --/
structure Vase where
  shape : VaseShape
  height : ℝ
  volume : ℝ

/-- Calculates the water level in a vase given the amount of water --/
noncomputable def waterLevel (v : Vase) (waterAmount : ℝ) : ℝ :=
  sorry

theorem highest_water_level_in_narrow_neck_vase 
  (vases : Fin 5 → Vase)
  (h_same_height : ∀ i j, (vases i).height = (vases j).height)
  (h_same_volume : ∀ i, (vases i).volume = 1)
  (h_water_amount : ∀ i, waterLevel (vases i) 0.5 > 0)
  (h_vase_a_narrow : (vases 0).shape = VaseShape.NarrowNeck)
  (h_other_shapes : ∀ i, i ≠ 0 → (vases i).shape ≠ VaseShape.NarrowNeck) :
  ∀ i, i ≠ 0 → waterLevel (vases 0) 0.5 > waterLevel (vases i) 0.5 :=
sorry

end NUMINAMATH_CALUDE_highest_water_level_in_narrow_neck_vase_l1589_158956


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_9_l1589_158918

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

theorem four_digit_divisible_by_9 (B : ℕ) : 
  B ≤ 9 → is_divisible_by_9 (5000 + 100 * B + 10 * B + 3) → B = 5 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_9_l1589_158918


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l1589_158963

theorem complex_expression_evaluation : 
  let expr := (((32400 * 4^3) / (3 * Real.sqrt 343)) / 18 / (7^3 * 10)) / 
              ((2 * Real.sqrt ((49^2 * 11)^4)) / 25^3)
  ∃ ε > 0, abs (expr - 0.00005366) < ε := by
sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l1589_158963


namespace NUMINAMATH_CALUDE_dot_product_special_vectors_l1589_158996

/-- The dot product of vectors a = (sin 55°, sin 35°) and b = (sin 25°, sin 65°) is equal to √3/2 -/
theorem dot_product_special_vectors :
  let a : ℝ × ℝ := (Real.sin (55 * π / 180), Real.sin (35 * π / 180))
  let b : ℝ × ℝ := (Real.sin (25 * π / 180), Real.sin (65 * π / 180))
  (a.1 * b.1 + a.2 * b.2) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_special_vectors_l1589_158996


namespace NUMINAMATH_CALUDE_max_cities_is_107_l1589_158952

/-- The maximum number of cities that can be visited in a specific sequence -/
def max_cities : ℕ := 107

/-- The total number of cities in the country -/
def total_cities : ℕ := 110

/-- A function representing the number of roads for each city in the sequence -/
def roads_for_city (k : ℕ) : ℕ := k

/-- Theorem stating that the maximum number of cities that can be visited in the specific sequence is 107 -/
theorem max_cities_is_107 :
  ∀ N : ℕ, 
  (∀ k : ℕ, 2 ≤ k → k ≤ N → roads_for_city k = k) →
  N ≤ total_cities →
  N ≤ max_cities :=
sorry

end NUMINAMATH_CALUDE_max_cities_is_107_l1589_158952


namespace NUMINAMATH_CALUDE_equiangular_iff_rectangle_l1589_158961

-- Define a quadrilateral
class Quadrilateral :=
(angles : Fin 4 → ℝ)

-- Define an equiangular quadrilateral
def is_equiangular (q : Quadrilateral) : Prop :=
∀ i j : Fin 4, q.angles i = q.angles j

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
∀ i : Fin 4, q.angles i = 90

-- Theorem statement
theorem equiangular_iff_rectangle (q : Quadrilateral) : 
  is_equiangular q ↔ is_rectangle q :=
sorry

end NUMINAMATH_CALUDE_equiangular_iff_rectangle_l1589_158961


namespace NUMINAMATH_CALUDE_f_properties_l1589_158962

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) ^ (a * x^2 - 4*x + 3)

theorem f_properties :
  (∀ x > 2, ∀ y > x, f 1 y < f 1 x) ∧
  (∃ x, f 1 x = 2 → 1 = 1) ∧
  (∀ a, (∀ x < 2, ∀ y < x, f a y < f a x) → 0 ≤ a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1589_158962


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_56_l1589_158987

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 ∧ m < n → n % m ≠ 0 ∨ m = 1

theorem no_primes_divisible_by_56 :
  ∀ p : ℕ, is_prime p → p % 56 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_56_l1589_158987


namespace NUMINAMATH_CALUDE_pizza_piece_volume_l1589_158997

/-- The volume of a piece of pizza -/
theorem pizza_piece_volume (thickness : ℝ) (diameter : ℝ) (num_pieces : ℕ) : 
  thickness = 1/3 →
  diameter = 12 →
  num_pieces = 12 →
  (π * (diameter/2)^2 * thickness) / num_pieces = π := by
  sorry

#check pizza_piece_volume

end NUMINAMATH_CALUDE_pizza_piece_volume_l1589_158997


namespace NUMINAMATH_CALUDE_count_integers_with_7_or_8_eq_386_l1589_158988

/-- The number of digits in base 9 that do not include 7 or 8 -/
def base7_digits : ℕ := 7

/-- The number of digits we consider in base 9 -/
def num_digits : ℕ := 3

/-- The total number of integers we consider -/
def total_integers : ℕ := 729

/-- The function that calculates the number of integers in base 9 
    from 1 to 729 that contain at least one digit 7 or 8 -/
def count_integers_with_7_or_8 : ℕ := total_integers - base7_digits ^ num_digits

theorem count_integers_with_7_or_8_eq_386 : 
  count_integers_with_7_or_8 = 386 := by sorry

end NUMINAMATH_CALUDE_count_integers_with_7_or_8_eq_386_l1589_158988


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1589_158955

theorem gcd_of_specific_numbers : Nat.gcd 33333 666666 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1589_158955


namespace NUMINAMATH_CALUDE_urn_probability_l1589_158964

/-- Represents the state of the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents one operation on the urn -/
inductive Operation
  | Red
  | Blue

/-- Calculates the probability of a specific sequence of operations -/
def sequenceProbability (ops : List Operation) : ℚ :=
  sorry

/-- Calculates the number of sequences with 3 red and 2 blue operations -/
def validSequences : ℕ :=
  sorry

/-- The main theorem stating the probability of having 4 balls of each color -/
theorem urn_probability : 
  let initialState : UrnState := ⟨1, 1⟩
  let finalState : UrnState := ⟨4, 4⟩
  let totalOperations : ℕ := 5
  let probability : ℚ := (validSequences : ℚ) * sequenceProbability (List.replicate 3 Operation.Red ++ List.replicate 2 Operation.Blue)
  probability = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_urn_probability_l1589_158964


namespace NUMINAMATH_CALUDE_bijection_between_sets_l1589_158994

def N (n : ℕ) : ℕ := n^9 % 10000

def set_greater (m : ℕ) : Set ℕ :=
  {n : ℕ | n < m ∧ n % 2 = 1 ∧ N n > n}

def set_lesser (m : ℕ) : Set ℕ :=
  {n : ℕ | n < m ∧ n % 2 = 1 ∧ N n < n}

theorem bijection_between_sets :
  ∃ (f : set_greater 10000 → set_lesser 10000),
    Function.Bijective f :=
  sorry

end NUMINAMATH_CALUDE_bijection_between_sets_l1589_158994


namespace NUMINAMATH_CALUDE_continuous_iff_a_eq_one_l1589_158977

-- Define the piecewise function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then Real.exp (3 * x) else a + 5 * x

-- State the theorem
theorem continuous_iff_a_eq_one (a : ℝ) :
  Continuous (f a) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_continuous_iff_a_eq_one_l1589_158977


namespace NUMINAMATH_CALUDE_square_sum_representation_l1589_158981

theorem square_sum_representation : ∃ (a b c : ℕ), 
  15129 = a^2 + b^2 + c^2 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a ≠ 27 ∨ b ≠ 72 ∨ c ≠ 96) ∧
  ∃ (d e f g h i : ℕ), 
    378225 = d^2 + e^2 + f^2 + g^2 + h^2 + i^2 ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
    g ≠ h ∧ g ≠ i ∧
    h ≠ i := by
  sorry

end NUMINAMATH_CALUDE_square_sum_representation_l1589_158981


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l1589_158948

/-- Represents a right circular cone -/
structure Cone where
  diameter : ℝ
  altitude : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ

/-- Represents the configuration of a cylinder inscribed in a cone -/
structure InscribedCylinder where
  cone : Cone
  cylinder : Cylinder
  height_radius_ratio : ℝ
  axes_coincide : Bool

/-- Theorem stating the radius of the inscribed cylinder -/
theorem inscribed_cylinder_radius 
  (ic : InscribedCylinder) 
  (h1 : ic.cone.diameter = 12) 
  (h2 : ic.cone.altitude = 15) 
  (h3 : ic.height_radius_ratio = 3) 
  (h4 : ic.axes_coincide = true) : 
  ic.cylinder.radius = 30 / 11 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l1589_158948


namespace NUMINAMATH_CALUDE_painting_price_change_l1589_158958

/-- The percentage increase in the first year given the conditions of the problem -/
def first_year_increase : ℝ := 30

/-- The percentage decrease in the second year -/
def second_year_decrease : ℝ := 15

/-- The final price as a percentage of the original price -/
def final_price_percentage : ℝ := 110.5

theorem painting_price_change : 
  (100 + first_year_increase) * (100 - second_year_decrease) / 100 = final_price_percentage := by
  sorry

#check painting_price_change

end NUMINAMATH_CALUDE_painting_price_change_l1589_158958


namespace NUMINAMATH_CALUDE_expression_value_l1589_158901

theorem expression_value (x : ℝ) (h : x = 5) : 3 * x + 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1589_158901


namespace NUMINAMATH_CALUDE_distance_between_circle_centers_l1589_158912

/-- Given two circles, one inside the other, with radii R and r respectively,
    and the shortest distance d between points on these circles,
    the distance between their centers is calculated as √((R-r+d)² - R²). -/
theorem distance_between_circle_centers
  (R r d : ℝ)
  (h_R : R = 28)
  (h_r : r = 12)
  (h_d : d = 10)
  (h_R_pos : R > 0)
  (h_r_pos : r > 0)
  (h_d_pos : d > 0)
  (h_R_gt_r : R > r)
  (h_inside : R - r > d) :
  Real.sqrt ((R - r + d)^2 - R^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_circle_centers_l1589_158912


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1589_158927

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  q > 1 →  -- common ratio > 1
  4 * (a 2010)^2 - 8 * (a 2010) + 3 = 0 →  -- a_2010 is a root
  4 * (a 2011)^2 - 8 * (a 2011) + 3 = 0 →  -- a_2011 is a root
  a 2012 + a 2013 = 18 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1589_158927


namespace NUMINAMATH_CALUDE_club_leadership_selection_l1589_158998

def total_members : ℕ := 24
def num_boys : ℕ := 14
def num_girls : ℕ := 10

theorem club_leadership_selection :
  (num_boys * (num_boys - 1) + num_girls * (num_girls - 1) : ℕ) = 272 :=
by sorry

end NUMINAMATH_CALUDE_club_leadership_selection_l1589_158998


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_condition_l1589_158915

/-- Represents the condition for a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  (m + 3) * (2*m + 1) < 0

/-- Represents the condition for an ellipse with foci on y-axis -/
def is_ellipse_y_foci (m : ℝ) : Prop :=
  -(2*m - 1) > m + 2 ∧ m + 2 > 0

/-- The necessary but not sufficient condition -/
def necessary_condition (m : ℝ) : Prop :=
  -2 < m ∧ m < -1/3

theorem hyperbola_ellipse_condition :
  (∀ m, is_hyperbola m ∧ is_ellipse_y_foci m → necessary_condition m) ∧
  ¬(∀ m, necessary_condition m → is_hyperbola m ∧ is_ellipse_y_foci m) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_condition_l1589_158915


namespace NUMINAMATH_CALUDE_simultaneous_hit_probability_l1589_158930

theorem simultaneous_hit_probability 
  (prob_A_hit : ℝ) 
  (prob_B_miss : ℝ) 
  (h1 : prob_A_hit = 0.8) 
  (h2 : prob_B_miss = 0.3) 
  (h3 : 0 ≤ prob_A_hit ∧ prob_A_hit ≤ 1) 
  (h4 : 0 ≤ prob_B_miss ∧ prob_B_miss ≤ 1) :
  prob_A_hit * (1 - prob_B_miss) = 14/25 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_hit_probability_l1589_158930


namespace NUMINAMATH_CALUDE_sequence_sum_l1589_158947

def geometric_sequence (a : ℕ → ℚ) := ∀ n, a (n + 1) = 2 * a n

def arithmetic_sequence (b : ℕ → ℚ) := ∃ d, ∀ n, b (n + 1) = b n + d

theorem sequence_sum (a : ℕ → ℚ) (b : ℕ → ℚ) :
  geometric_sequence a →
  a 2 * a 3 * a 4 = 27 / 64 →
  arithmetic_sequence b →
  b 7 = a 5 →
  b 3 + b 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l1589_158947


namespace NUMINAMATH_CALUDE_min_value_theorem_l1589_158919

theorem min_value_theorem (x y : ℝ) 
  (h1 : x > -1) 
  (h2 : y > 0) 
  (h3 : x + y = 1) : 
  (∀ x' y' : ℝ, x' > -1 → y' > 0 → x' + y' = 1 → 
    1 / (x' + 1) + 4 / y' ≥ 1 / (x + 1) + 4 / y) ∧ 
  1 / (x + 1) + 4 / y = 9/2 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1589_158919


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1589_158935

theorem necessary_but_not_sufficient (x : ℝ) :
  (∀ x, (1/2)^x > 1 → 1/x < 1) ∧
  ¬(∀ x, 1/x < 1 → (1/2)^x > 1) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1589_158935


namespace NUMINAMATH_CALUDE_equation_describes_ellipse_l1589_158950

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y - 4)^2) + Real.sqrt ((x - 8)^2 + (y + 1)^2) = 15

-- Define what it means for a point to be on the conic
def point_on_conic (x y : ℝ) : Prop := conic_equation x y

-- Define the foci of the ellipse
def focus1 : ℝ × ℝ := (2, 4)
def focus2 : ℝ × ℝ := (8, -1)

-- Theorem stating that the equation describes an ellipse
theorem equation_describes_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), point_on_conic x y ↔
    (x - (focus1.1 + focus2.1) / 2)^2 / a^2 +
    (y - (focus1.2 + focus2.2) / 2)^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_equation_describes_ellipse_l1589_158950


namespace NUMINAMATH_CALUDE_prove_present_age_of_B_l1589_158986

/-- The present age of person B given the conditions:
    1. In 10 years, A will be twice as old as B was 10 years ago
    2. A is now 9 years older than B -/
def present_age_of_B (a b : ℕ) : Prop :=
  (a + 10 = 2 * (b - 10)) ∧ (a = b + 9) → b = 39

theorem prove_present_age_of_B :
  ∀ (a b : ℕ), present_age_of_B a b :=
by
  sorry

end NUMINAMATH_CALUDE_prove_present_age_of_B_l1589_158986


namespace NUMINAMATH_CALUDE_angle_x_is_60_l1589_158917

/-- Given a geometric configuration where:
  1. y + 140° forms a straight angle
  2. There's a triangle with angles 40°, 80°, and z°
  3. x is an angle opposite to z
Prove that x = 60° -/
theorem angle_x_is_60 (y z x : ℝ) : 
  y + 140 = 180 →  -- Straight angle property
  40 + 80 + z = 180 →  -- Triangle angle sum property
  x = z →  -- Opposite angles are equal
  x = 60 := by sorry

end NUMINAMATH_CALUDE_angle_x_is_60_l1589_158917


namespace NUMINAMATH_CALUDE_absolute_value_equality_l1589_158967

theorem absolute_value_equality (a b : ℝ) : 
  (∀ x y : ℝ, |a * x + b * y| + |b * x + a * y| = |x| + |y|) ↔ 
  ((a = 1 ∧ b = 0) ∨ (a = 0 ∧ b = 1) ∨ (a = 0 ∧ b = -1) ∨ (a = -1 ∧ b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l1589_158967


namespace NUMINAMATH_CALUDE_yellow_to_red_ratio_l1589_158933

/-- Represents the number of chairs of each color in Rodrigo's classroom --/
structure ChairCounts where
  red : ℕ
  yellow : ℕ
  blue : ℕ

/-- Represents the state of chairs in Rodrigo's classroom --/
def classroom_state : ChairCounts → Prop
  | ⟨red, yellow, blue⟩ => 
    red = 4 ∧ 
    blue = yellow - 2 ∧ 
    red + yellow + blue = 18 ∧ 
    red + yellow + blue - 3 = 15

/-- The theorem stating the ratio of yellow to red chairs --/
theorem yellow_to_red_ratio (chairs : ChairCounts) :
  classroom_state chairs → chairs.yellow / chairs.red = 2 := by
  sorry

#check yellow_to_red_ratio

end NUMINAMATH_CALUDE_yellow_to_red_ratio_l1589_158933


namespace NUMINAMATH_CALUDE_no_solution_for_inequality_l1589_158926

theorem no_solution_for_inequality :
  ¬ ∃ x : ℝ, 3 * x^2 + 9 * x ≤ -12 := by sorry

end NUMINAMATH_CALUDE_no_solution_for_inequality_l1589_158926


namespace NUMINAMATH_CALUDE_profit_maximizing_prices_l1589_158972

/-- Represents the selling price in yuan -/
def selling_price : ℝ → ℝ := id

/-- Represents the daily sales quantity as a function of selling price -/
def daily_sales (x : ℝ) : ℝ := 200 - (x - 20) * 20

/-- Represents the daily profit as a function of selling price -/
def daily_profit (x : ℝ) : ℝ := (x - 12) * (daily_sales x)

/-- The theorem states that 19 and 23 are the only selling prices that achieve a daily profit of 1540 yuan -/
theorem profit_maximizing_prices :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  daily_profit x₁ = 1540 ∧ 
  daily_profit x₂ = 1540 ∧
  (∀ x : ℝ, daily_profit x = 1540 → (x = x₁ ∨ x = x₂)) :=
sorry

end NUMINAMATH_CALUDE_profit_maximizing_prices_l1589_158972


namespace NUMINAMATH_CALUDE_cos_is_even_l1589_158910

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- State the theorem
theorem cos_is_even : IsEven Real.cos := by
  sorry

end NUMINAMATH_CALUDE_cos_is_even_l1589_158910


namespace NUMINAMATH_CALUDE_rectangle_area_l1589_158906

/-- The area of the rectangle formed by the intersections of x^4 + y^4 = 100 and xy = 4 -/
theorem rectangle_area : ∃ (a b : ℝ), 
  (a^4 + b^4 = 100) ∧ 
  (a * b = 4) ∧ 
  (2 * (a^2 - b^2) = 4 * Real.sqrt 17) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1589_158906


namespace NUMINAMATH_CALUDE_problem_solution_l1589_158973

theorem problem_solution : ∀ A B : ℕ, 
  A = 55 * 100 + 19 * 10 → 
  B = 173 + 5 * 224 → 
  A - B = 4397 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1589_158973


namespace NUMINAMATH_CALUDE_eggs_laid_per_chicken_l1589_158923

theorem eggs_laid_per_chicken 
  (initial_eggs : ℕ) 
  (used_eggs : ℕ) 
  (num_chickens : ℕ) 
  (final_eggs : ℕ) 
  (h1 : initial_eggs = 10)
  (h2 : used_eggs = 5)
  (h3 : num_chickens = 2)
  (h4 : final_eggs = 11)
  : (final_eggs - (initial_eggs - used_eggs)) / num_chickens = 3 := by
  sorry

end NUMINAMATH_CALUDE_eggs_laid_per_chicken_l1589_158923


namespace NUMINAMATH_CALUDE_celeste_song_probability_l1589_158905

/-- Represents the collection of songs on Celeste's o-Pod -/
structure SongCollection where
  total_songs : Nat
  shortest_song : Nat
  song_increment : Nat
  favorite_song_length : Nat
  time_limit : Nat

/-- Calculates the probability of not hearing the entire favorite song 
    within the time limit for a given song collection -/
def probability_not_hearing_favorite (sc : SongCollection) : Rat :=
  1 - (Nat.factorial (sc.total_songs - 1) + Nat.factorial (sc.total_songs - 2)) / 
      Nat.factorial sc.total_songs

/-- The main theorem stating the probability for Celeste's specific case -/
theorem celeste_song_probability : 
  let sc : SongCollection := {
    total_songs := 12,
    shortest_song := 45,
    song_increment := 15,
    favorite_song_length := 240,
    time_limit := 300
  }
  probability_not_hearing_favorite sc = 10 / 11 := by
  sorry


end NUMINAMATH_CALUDE_celeste_song_probability_l1589_158905


namespace NUMINAMATH_CALUDE_count_special_numbers_l1589_158941

def is_valid_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def reverse_number (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

theorem count_special_numbers :
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, is_valid_three_digit_number n ∧ is_valid_three_digit_number (n - 297) ∧ 
               n - 297 = reverse_number n) ∧
    S.card = 60 :=
sorry

end NUMINAMATH_CALUDE_count_special_numbers_l1589_158941


namespace NUMINAMATH_CALUDE_probability_all_even_is_one_eightyfourth_l1589_158944

/-- Represents a tile with a number from 1 to 10 -/
def Tile := Fin 10

/-- Represents a player's selection of 3 tiles -/
def Selection := Finset Tile

/-- The set of all possible selections -/
def AllSelections : Finset Selection :=
  sorry

/-- Predicate to check if a selection sum is even -/
def hasEvenSum (s : Selection) : Prop :=
  sorry

/-- The probability of all three players getting an even sum -/
def probabilityAllEven : ℚ :=
  sorry

theorem probability_all_even_is_one_eightyfourth :
  probabilityAllEven = 1 / 84 :=
sorry

end NUMINAMATH_CALUDE_probability_all_even_is_one_eightyfourth_l1589_158944


namespace NUMINAMATH_CALUDE_range_of_a_when_a_minus_3_in_M_range_of_a_when_interval_subset_M_l1589_158903

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x + a|
def g (x : ℝ) : ℝ := |x + 3| - x

-- Define the set M
def M (a : ℝ) : Set ℝ := {x | f a x < g x}

-- Theorem 1
theorem range_of_a_when_a_minus_3_in_M (a : ℝ) :
  (a - 3) ∈ M a → 0 < a ∧ a < 3 := by sorry

-- Theorem 2
theorem range_of_a_when_interval_subset_M (a : ℝ) :
  Set.Icc (-1) 1 ⊆ M a → -2 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_when_a_minus_3_in_M_range_of_a_when_interval_subset_M_l1589_158903


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1589_158985

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1589_158985


namespace NUMINAMATH_CALUDE_largest_number_is_482_l1589_158953

/-- Represents a systematic sample from a range of products -/
structure SystematicSample where
  total_products : Nat
  first_number : Nat
  second_number : Nat

/-- Calculates the largest number in a systematic sample -/
def largest_number (s : SystematicSample) : Nat :=
  let interval := s.second_number - s.first_number
  let sample_size := s.total_products / interval
  s.first_number + interval * (sample_size - 1)

/-- Theorem stating that for the given systematic sample, the largest number is 482 -/
theorem largest_number_is_482 :
  let s : SystematicSample := ⟨500, 7, 32⟩
  largest_number s = 482 := by sorry

end NUMINAMATH_CALUDE_largest_number_is_482_l1589_158953


namespace NUMINAMATH_CALUDE_pentagonal_cross_section_exists_regular_pentagonal_cross_section_impossible_l1589_158957

/-- A cube in 3D space -/
structure Cube :=
  (side : ℝ)
  (side_pos : side > 0)

/-- A plane in 3D space -/
structure Plane :=
  (normal : ℝ × ℝ × ℝ)
  (point : ℝ × ℝ × ℝ)

/-- A pentagon in 2D space -/
structure Pentagon :=
  (vertices : Finset (ℝ × ℝ))
  (is_pentagon : vertices.card = 5)

/-- A regular pentagon in 2D space -/
structure RegularPentagon extends Pentagon :=
  (is_regular : ∀ (v1 v2 : ℝ × ℝ), v1 ∈ vertices → v2 ∈ vertices → v1 ≠ v2 → 
    ∃ (rotation : ℝ × ℝ → ℝ × ℝ), rotation v1 = v2 ∧ rotation '' vertices = vertices)

/-- The cross-section formed by intersecting a cube with a plane -/
def crossSection (c : Cube) (p : Plane) : Set (ℝ × ℝ) :=
  sorry

theorem pentagonal_cross_section_exists (c : Cube) : 
  ∃ (p : Plane), ∃ (pent : Pentagon), crossSection c p = ↑pent.vertices :=
sorry

theorem regular_pentagonal_cross_section_impossible (c : Cube) : 
  ¬∃ (p : Plane), ∃ (reg_pent : RegularPentagon), crossSection c p = ↑reg_pent.vertices :=
sorry

end NUMINAMATH_CALUDE_pentagonal_cross_section_exists_regular_pentagonal_cross_section_impossible_l1589_158957


namespace NUMINAMATH_CALUDE_cone_height_l1589_158992

/-- A cone with volume 8192π cubic inches and a vertical cross-section vertex angle of 90 degrees has a height equal to the cube root of 24576 inches. -/
theorem cone_height (V : ℝ) (θ : ℝ) (h : ℝ) :
  V = 8192 * Real.pi ∧ θ = 90 → h = (24576 : ℝ) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_cone_height_l1589_158992


namespace NUMINAMATH_CALUDE_repeating_decimal_difference_l1589_158971

/-- Represents a repeating decimal with a single digit repeating part -/
def SingleDigitRepeatingDecimal (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with a two-digit repeating part -/
def TwoDigitRepeatingDecimal (n : ℕ) : ℚ := n / 99

theorem repeating_decimal_difference (h1 : 0 < 99) (h2 : 0 < 9) :
  99 * (TwoDigitRepeatingDecimal 49 - SingleDigitRepeatingDecimal 4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_difference_l1589_158971


namespace NUMINAMATH_CALUDE_darcys_commute_l1589_158975

/-- Darcy's commute problem -/
theorem darcys_commute (distance_to_work : ℝ) (walking_speed : ℝ) (train_speed : ℝ) 
  (extra_time : ℝ) :
  distance_to_work = 1.5 →
  walking_speed = 3 →
  train_speed = 20 →
  (distance_to_work / walking_speed) * 60 = 
    (distance_to_work / train_speed) * 60 + extra_time + 2 →
  extra_time = 25.5 := by
sorry

end NUMINAMATH_CALUDE_darcys_commute_l1589_158975


namespace NUMINAMATH_CALUDE_theater_group_arrangement_l1589_158960

theorem theater_group_arrangement (n : ℕ) : n ≥ 1981 ∧ 
  n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧ n % 12 = 1 → n = 1981 :=
by sorry

end NUMINAMATH_CALUDE_theater_group_arrangement_l1589_158960


namespace NUMINAMATH_CALUDE_chips_for_dinner_l1589_158914

theorem chips_for_dinner (dinner : ℕ) (after : ℕ) : 
  dinner > 0 → 
  after > 0 → 
  dinner + after = 3 → 
  dinner = 2 := by
sorry

end NUMINAMATH_CALUDE_chips_for_dinner_l1589_158914


namespace NUMINAMATH_CALUDE_f_3_range_l1589_158990

theorem f_3_range (a c : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = a * x^2 - c)
  (h_1 : -4 ≤ f 1 ∧ f 1 ≤ -1)
  (h_2 : -1 ≤ f 2 ∧ f 2 ≤ 5) :
  -1 ≤ f 3 ∧ f 3 ≤ 20 := by
sorry

end NUMINAMATH_CALUDE_f_3_range_l1589_158990


namespace NUMINAMATH_CALUDE_power_function_difference_l1589_158995

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_difference (f : ℝ → ℝ) :
  isPowerFunction f → f 9 = 3 → f 2 - f 1 = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_difference_l1589_158995


namespace NUMINAMATH_CALUDE_betty_afternoon_catch_l1589_158976

/-- The number of flies a frog eats per day -/
def flies_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of flies Betty caught in the morning -/
def morning_catch : ℕ := 5

/-- The number of additional flies Betty needs -/
def additional_flies_needed : ℕ := 4

/-- The number of flies that escaped -/
def escaped_flies : ℕ := 1

theorem betty_afternoon_catch :
  ∃ (afternoon_catch : ℕ),
    afternoon_catch = 6 ∧
    flies_per_day * days_in_week =
      morning_catch + (afternoon_catch - escaped_flies) + additional_flies_needed :=
by sorry

end NUMINAMATH_CALUDE_betty_afternoon_catch_l1589_158976


namespace NUMINAMATH_CALUDE_potato_rows_l1589_158942

theorem potato_rows (seeds_per_row : ℕ) (total_potatoes : ℕ) (h1 : seeds_per_row = 9) (h2 : total_potatoes = 54) :
  total_potatoes / seeds_per_row = 6 := by
  sorry

end NUMINAMATH_CALUDE_potato_rows_l1589_158942


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1589_158943

theorem system_solution_ratio (x y c d : ℝ) : 
  x ≠ 0 → y ≠ 0 → d ≠ 0 →
  8 * x - 6 * y = c →
  10 * y - 15 * x = d →
  c / d = -8 / 15 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1589_158943


namespace NUMINAMATH_CALUDE_line_segment_coefficient_sum_squares_l1589_158911

/-- Given a line segment connecting points (1, -3) and (4, 5), parameterized by x = at + b and y = ct + d
    where 0 ≤ t ≤ 1 and t = 0 corresponds to point (1, -3), 
    the sum of squares of the coefficients a^2 + b^2 + c^2 + d^2 equals 83. -/
theorem line_segment_coefficient_sum_squares :
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (b = 1 ∧ d = -3) →
  (a + b = 4 ∧ c + d = 5) →
  a^2 + b^2 + c^2 + d^2 = 83 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_coefficient_sum_squares_l1589_158911


namespace NUMINAMATH_CALUDE_locus_and_fixed_points_l1589_158936

-- Define the points and vectors
variable (P Q R M A S B D E F : ℝ × ℝ)
variable (a b : ℝ)

-- Define the conditions
axiom P_on_x_axis : P.2 = 0
axiom Q_on_y_axis : Q.1 = 0
axiom R_coord : R = (0, -3)
axiom S_coord : S = (0, 2)
axiom PR_dot_PM : (R.1 - P.1) * (M.1 - P.1) + (R.2 - P.2) * (M.2 - P.2) = 0
axiom PQ_half_QM : (Q.1 - P.1, Q.2 - P.2) = (1/2 : ℝ) • (M.1 - Q.1, M.2 - Q.2)
axiom A_coord : A = (a, b)
axiom A_outside_C : a ≠ 0 ∧ b ≠ 2
axiom AB_AD_tangent : True  -- This condition is implied but not directly stated
axiom E_on_line : E.2 = -2
axiom F_on_line : F.2 = -2

-- Define the locus C
def C (x y : ℝ) : Prop := x^2 = 4*y

-- State the theorem
theorem locus_and_fixed_points :
  (∀ x y, C x y ↔ x^2 = 4*y) ∧
  (∃ r : ℝ, r > 0 ∧ 
    ((E.1 + F.1)/2 - 0)^2 + ((E.2 + F.2)/2 - (-2 + 2*Real.sqrt 2))^2 = r^2 ∧
    ((E.1 + F.1)/2 - 0)^2 + ((E.2 + F.2)/2 - (-2 - 2*Real.sqrt 2))^2 = r^2) :=
sorry

end NUMINAMATH_CALUDE_locus_and_fixed_points_l1589_158936


namespace NUMINAMATH_CALUDE_solution_l1589_158991

def problem (m n : ℕ) : Prop :=
  m + n = 80 ∧ 
  Nat.gcd m n = 6 ∧ 
  Nat.lcm m n = 210

theorem solution (m n : ℕ) (h : problem m n) : 
  (1 : ℚ) / m + (1 : ℚ) / n = 1 / 15.75 := by
  sorry

end NUMINAMATH_CALUDE_solution_l1589_158991


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l1589_158984

/-- Given a triangle with angles 45°, 3x°, and x°, prove that x = 33.75° -/
theorem triangle_angle_proof (x : ℝ) : 
  45 + 3*x + x = 180 → x = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l1589_158984


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1589_158924

theorem arithmetic_sequence_first_term 
  (a : ℚ) -- First term of the sequence
  (d : ℚ) -- Common difference of the sequence
  (h1 : (30 : ℚ) / 2 * (2 * a + 29 * d) = 450) -- Sum of first 30 terms
  (h2 : (30 : ℚ) / 2 * (2 * (a + 30 * d) + 29 * d) = 1950) -- Sum of next 30 terms
  : a = -55 / 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1589_158924


namespace NUMINAMATH_CALUDE_table_free_sides_length_l1589_158969

theorem table_free_sides_length (length width : ℝ) : 
  length > 0 → 
  width > 0 → 
  length = 2 * width → 
  length * width = 128 → 
  length + 2 * width = 32 := by
sorry

end NUMINAMATH_CALUDE_table_free_sides_length_l1589_158969


namespace NUMINAMATH_CALUDE_cross_section_distance_theorem_l1589_158993

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  -- Add any necessary fields here
  mk ::

/-- Represents a cross-section of the pyramid -/
structure CrossSection where
  area : ℝ
  distance_from_apex : ℝ

/-- Theorem about the distance of cross-sections in a right hexagonal pyramid -/
theorem cross_section_distance_theorem 
  (pyramid : RightHexagonalPyramid)
  (cs1 cs2 : CrossSection)
  (h : cs1.distance_from_apex < cs2.distance_from_apex)
  (area_h : cs1.area < cs2.area)
  (d : ℝ)
  (h_d : d = cs2.distance_from_apex - cs1.distance_from_apex) :
  cs2.distance_from_apex = d / (1 - Real.sqrt (cs1.area / cs2.area)) :=
by sorry

end NUMINAMATH_CALUDE_cross_section_distance_theorem_l1589_158993


namespace NUMINAMATH_CALUDE_positive_increase_l1589_158945

/-- Represents the change in water level -/
def WaterLevelChange := ℝ

/-- Interpretation function for water level change -/
def interpret (change : WaterLevelChange) : ℝ := change

/-- Axiom: Negative numbers represent a decrease in water level -/
axiom negative_decrease (x : ℝ) (h : x < 0) : 
  interpret x = -x

/-- Theorem: Positive numbers represent an increase in water level -/
theorem positive_increase (x : ℝ) (h : x > 0) : 
  interpret x = x := by sorry

end NUMINAMATH_CALUDE_positive_increase_l1589_158945


namespace NUMINAMATH_CALUDE_range_of_c_l1589_158937

open Real

theorem range_of_c (c : ℝ) : 
  (∀ x y : ℝ, x < y → c^x > c^y) →  -- y = c^x is decreasing
  (∃ x : ℝ, x^2 - Real.sqrt 2 * x + c ≤ 0) →  -- negation of q
  (0 < c ∧ c < 1) →  -- derived from decreasing function condition
  0 < c ∧ c ≤ (1/2) := by
sorry

end NUMINAMATH_CALUDE_range_of_c_l1589_158937


namespace NUMINAMATH_CALUDE_placement_count_l1589_158979

/-- Represents a painting with width and height -/
structure Painting :=
  (width : Nat)
  (height : Nat)

/-- Represents a wall with width and height -/
structure Wall :=
  (width : Nat)
  (height : Nat)

/-- Represents the collection of paintings -/
def paintings : List Painting := [
  ⟨2, 1⟩,
  ⟨1, 1⟩, ⟨1, 1⟩,
  ⟨1, 2⟩, ⟨1, 2⟩,
  ⟨2, 2⟩, ⟨2, 2⟩,
  ⟨4, 3⟩, ⟨4, 3⟩,
  ⟨4, 4⟩, ⟨4, 4⟩
]

/-- The wall on which paintings are to be placed -/
def wall : Wall := ⟨12, 6⟩

/-- Function to calculate the number of ways to place paintings on the wall -/
def numberOfPlacements (w : Wall) (p : List Painting) : Nat :=
  sorry

/-- Theorem stating that the number of placements is 16896 -/
theorem placement_count : numberOfPlacements wall paintings = 16896 := by
  sorry

end NUMINAMATH_CALUDE_placement_count_l1589_158979
