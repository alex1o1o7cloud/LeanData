import Mathlib

namespace NUMINAMATH_CALUDE_units_digit_sum_base8_l2447_244762

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10_to_base8 (n : ℕ) : ℕ := sorry

/-- Returns the units digit of a base-8 number --/
def units_digit_base8 (n : ℕ) : ℕ := sorry

theorem units_digit_sum_base8 : 
  units_digit_base8 (base10_to_base8 (base8_to_base10 53 + base8_to_base10 64)) = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_base8_l2447_244762


namespace NUMINAMATH_CALUDE_regular_star_points_l2447_244712

/-- A p-pointed regular star with specific angle properties -/
structure RegularStar where
  p : ℕ
  angle_d : ℝ
  angle_c : ℝ
  angle_c_minus_d : angle_c = angle_d + 15
  sum_of_angles : p * angle_c + p * angle_d = 360

/-- The number of points in a regular star with given properties is 24 -/
theorem regular_star_points (star : RegularStar) : star.p = 24 := by
  sorry

end NUMINAMATH_CALUDE_regular_star_points_l2447_244712


namespace NUMINAMATH_CALUDE_fish_weight_l2447_244701

/-- Given a barrel of fish with the following properties:
    - The initial weight of the barrel with all fish is 54 kg
    - The weight of the barrel with half of the fish removed is 29 kg
    This theorem proves that the total weight of the fish is 50 kg. -/
theorem fish_weight (initial_weight : ℝ) (half_removed_weight : ℝ) 
  (h1 : initial_weight = 54)
  (h2 : half_removed_weight = 29) :
  ∃ (barrel_weight fish_weight : ℝ),
    barrel_weight + fish_weight = initial_weight ∧
    barrel_weight + fish_weight / 2 = half_removed_weight ∧
    fish_weight = 50 := by
  sorry

end NUMINAMATH_CALUDE_fish_weight_l2447_244701


namespace NUMINAMATH_CALUDE_tan_sum_greater_than_three_l2447_244753

theorem tan_sum_greater_than_three :
  Real.tan (40 * π / 180) + Real.tan (45 * π / 180) + Real.tan (50 * π / 180) > 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_greater_than_three_l2447_244753


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2447_244789

theorem quadratic_inequality_solution (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 < a ∧ a < 1 → (a * x^2 - (a + 1) * x + 1 < 0 ↔ 1 < x ∧ x < 1/a)) ∧
  (∀ x : ℝ, a > 1 → (a * x^2 - (a + 1) * x + 1 < 0 ↔ 1/a < x ∧ x < 1)) ∧
  (a = 1 → ¬∃ x : ℝ, a * x^2 - (a + 1) * x + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2447_244789


namespace NUMINAMATH_CALUDE_water_in_pool_l2447_244703

-- Define the parameters
def initial_bucket : ℝ := 1
def additional_buckets : ℝ := 8.8
def liters_per_bucket : ℝ := 10
def evaporation_rate : ℝ := 0.2
def splashing_rate : ℝ := 0.5
def time_taken : ℝ := 20

-- Define the theorem
theorem water_in_pool : 
  let total_buckets := initial_bucket + additional_buckets
  let total_water := total_buckets * liters_per_bucket
  let evaporation_loss := evaporation_rate * time_taken
  let splashing_loss := splashing_rate * time_taken
  let total_loss := evaporation_loss + splashing_loss
  let net_water := total_water - total_loss
  net_water = 84 := by
  sorry


end NUMINAMATH_CALUDE_water_in_pool_l2447_244703


namespace NUMINAMATH_CALUDE_product_multiple_of_five_probability_l2447_244760

def N : ℕ := 2020

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

def count_multiples_of_five : ℕ := N / 5

def prob_not_multiple_of_five : ℚ := (N - count_multiples_of_five) / N

theorem product_multiple_of_five_probability :
  let p := 1 - (prob_not_multiple_of_five * (prob_not_multiple_of_five - 1 / N) * (prob_not_multiple_of_five - 2 / N))
  ∃ ε > 0, |p - 0.485| < ε :=
sorry

end NUMINAMATH_CALUDE_product_multiple_of_five_probability_l2447_244760


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2447_244761

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms

/-- Given an arithmetic sequence with S_10 = 20 and S_20 = 15, prove S_30 = -15 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) 
  (h1 : a.S 10 = 20) 
  (h2 : a.S 20 = 15) : 
  a.S 30 = -15 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2447_244761


namespace NUMINAMATH_CALUDE_friend_spent_ten_l2447_244749

def lunch_problem (total : ℝ) (difference : ℝ) : Prop :=
  ∃ (your_cost friend_cost : ℝ),
    your_cost + friend_cost = total ∧
    friend_cost = your_cost + difference ∧
    friend_cost = 10

theorem friend_spent_ten :
  lunch_problem 17 3 :=
sorry

end NUMINAMATH_CALUDE_friend_spent_ten_l2447_244749


namespace NUMINAMATH_CALUDE_boat_distance_proof_l2447_244783

theorem boat_distance_proof (boat_speed : ℝ) (stream_speed : ℝ) (time_difference : ℝ) :
  boat_speed = 10 →
  stream_speed = 2 →
  time_difference = 1.5 →
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  ∃ distance : ℝ,
    distance / upstream_speed = distance / downstream_speed + time_difference ∧
    distance = 36 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_proof_l2447_244783


namespace NUMINAMATH_CALUDE_largest_sales_increase_2011_l2447_244737

-- Define the sales data for each year
def sales : Fin 8 → ℕ
  | 0 => 20
  | 1 => 24
  | 2 => 27
  | 3 => 26
  | 4 => 28
  | 5 => 33
  | 6 => 32
  | 7 => 35

-- Define the function to calculate the sales increase between two consecutive years
def salesIncrease (i : Fin 7) : ℤ :=
  (sales (i.succ : Fin 8) : ℤ) - (sales i : ℤ)

-- Define the theorem to prove
theorem largest_sales_increase_2011 :
  ∃ i : Fin 7, salesIncrease i = 5 ∧
  ∀ j : Fin 7, salesIncrease j ≤ 5 ∧
  (i : ℕ) + 2006 = 2011 :=
by sorry

end NUMINAMATH_CALUDE_largest_sales_increase_2011_l2447_244737


namespace NUMINAMATH_CALUDE_unpainted_cubes_4x4x4_l2447_244777

/-- Represents a cube with side length n --/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted strip on a cube face --/
structure PaintedStrip where
  width : ℕ
  height : ℕ

/-- Calculates the number of unpainted unit cubes in a cube with painted strips --/
def unpainted_cubes (c : Cube 4) (strip : PaintedStrip) : ℕ :=
  sorry

theorem unpainted_cubes_4x4x4 :
  ∀ (c : Cube 4) (strip : PaintedStrip),
    strip.width = 2 ∧ strip.height = c.side_length →
    unpainted_cubes c strip = 40 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_4x4x4_l2447_244777


namespace NUMINAMATH_CALUDE_total_run_time_l2447_244772

def emma_time : ℝ := 20

theorem total_run_time (fernando_time : ℝ) 
  (h1 : fernando_time = 2 * emma_time) : 
  emma_time + fernando_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_run_time_l2447_244772


namespace NUMINAMATH_CALUDE_axisymmetric_shapes_l2447_244707

-- Define the basic shapes
inductive Shape
  | Triangle
  | Parallelogram
  | Rectangle
  | Circle

-- Define the property of being axisymmetric
def is_axisymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => true
  | Shape.Circle => true
  | _ => false

-- Theorem statement
theorem axisymmetric_shapes :
  ∀ s : Shape, is_axisymmetric s ↔ (s = Shape.Rectangle ∨ s = Shape.Circle) :=
by sorry

end NUMINAMATH_CALUDE_axisymmetric_shapes_l2447_244707


namespace NUMINAMATH_CALUDE_mary_green_crayons_mary_green_crayons_correct_l2447_244752

theorem mary_green_crayons 
  (initial_blue : ℕ) 
  (green_given : ℕ) 
  (blue_given : ℕ) 
  (remaining : ℕ) : ℕ :=
  
  let initial_total := remaining + green_given + blue_given
  let initial_green := initial_total - initial_blue

  initial_green

theorem mary_green_crayons_correct 
  (initial_blue : ℕ) 
  (green_given : ℕ) 
  (blue_given : ℕ) 
  (remaining : ℕ) : 
  mary_green_crayons initial_blue green_given blue_given remaining = 5 :=
by
  -- Given conditions
  have h1 : initial_blue = 8 := by sorry
  have h2 : green_given = 3 := by sorry
  have h3 : blue_given = 1 := by sorry
  have h4 : remaining = 9 := by sorry

  -- Proof
  sorry

end NUMINAMATH_CALUDE_mary_green_crayons_mary_green_crayons_correct_l2447_244752


namespace NUMINAMATH_CALUDE_turnip_bag_weights_l2447_244785

def bag_weights : List Nat := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (t : Nat) : Prop :=
  t ∈ bag_weights ∧
  ∃ (o c : Nat),
    o + c = (bag_weights.sum - t) ∧
    c = 2 * o

theorem turnip_bag_weights :
  ∀ t, is_valid_turnip_weight t ↔ t = 13 ∨ t = 16 :=
by sorry

end NUMINAMATH_CALUDE_turnip_bag_weights_l2447_244785


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2447_244723

theorem solve_linear_equation (x : ℝ) : 5 * x - 3 = 17 ↔ x = 4 := by sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2447_244723


namespace NUMINAMATH_CALUDE_point_on_line_iff_vector_sum_l2447_244750

/-- 
Given distinct points A and B, and a point O not on line AB,
a point C is on line AB if and only if there exist real numbers x and y 
such that OC = x * OA + y * OB and x + y = 1.
-/
theorem point_on_line_iff_vector_sum (O A B C : EuclideanSpace ℝ (Fin 3)) 
  (hAB : A ≠ B) (hO : O ∉ affineSpan ℝ {A, B}) : 
  C ∈ affineSpan ℝ {A, B} ↔ 
  ∃ (x y : ℝ), C - O = x • (A - O) + y • (B - O) ∧ x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_iff_vector_sum_l2447_244750


namespace NUMINAMATH_CALUDE_problem_solution_l2447_244721

theorem problem_solution : ∃ N : ℕ, 
  (N / (555 + 445) = 2 * (555 - 445)) ∧ 
  (N % (555 + 445) = 50) ∧ 
  N = 220050 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2447_244721


namespace NUMINAMATH_CALUDE_odd_function_extension_l2447_244751

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_extension
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_neg : ∀ x < 0, f x = 1 - Real.exp (-x + 1)) :
  ∀ x > 0, f x = Real.exp (x + 1) - 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_extension_l2447_244751


namespace NUMINAMATH_CALUDE_f_max_at_neg_two_l2447_244724

/-- The function f(x) = -x^2 - 4x + 16 -/
def f (x : ℝ) : ℝ := -x^2 - 4*x + 16

/-- The statement that f(x) attains its maximum value when x = -2 -/
theorem f_max_at_neg_two :
  ∀ x : ℝ, f x ≤ f (-2) :=
sorry

end NUMINAMATH_CALUDE_f_max_at_neg_two_l2447_244724


namespace NUMINAMATH_CALUDE_prob_draw_king_l2447_244798

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the ranks in a standard deck -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Represents the suits in a standard deck -/
inductive Suit
  | Spades | Hearts | Diamonds | Clubs

/-- A card in the deck -/
structure Card :=
  (rank : Rank)
  (suit : Suit)

/-- The number of ranks in a standard deck -/
def rank_count : Nat := 13

/-- The number of suits in a standard deck -/
def suit_count : Nat := 4

/-- The total number of cards in a standard deck -/
def total_cards : Nat := rank_count * suit_count

/-- The number of Kings in a standard deck -/
def king_count : Nat := suit_count

/-- Theorem: The probability of drawing a King from a standard 52-card deck is 1/13 -/
theorem prob_draw_king (d : Deck) : 
  (king_count : ℚ) / total_cards = 1 / 13 := by sorry

end NUMINAMATH_CALUDE_prob_draw_king_l2447_244798


namespace NUMINAMATH_CALUDE_set_A_properties_l2447_244717

def A : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 2^k}

theorem set_A_properties :
  (∀ a ∈ A, ∀ b : ℕ, b ≠ 0 → b < 2*a - 1 → ¬(2*a ∣ b*(b+1))) ∧
  (∀ a ∉ A, a ≠ 1 → ∃ b : ℕ, b ≠ 0 ∧ b < 2*a - 1 ∧ (2*a ∣ b*(b+1))) :=
by sorry

end NUMINAMATH_CALUDE_set_A_properties_l2447_244717


namespace NUMINAMATH_CALUDE_water_fraction_in_first_container_l2447_244708

/-- Represents the amount of liquid in a container -/
structure Container where
  juice : ℚ
  water : ℚ

/-- The problem setup and operations -/
def liquidTransfer : Prop :=
  ∃ (initial1 initial2 after_first_transfer after_second_transfer final1 : Container),
    -- Initial setup
    initial1.juice = 5 ∧ initial1.water = 0 ∧
    initial2.juice = 0 ∧ initial2.water = 5 ∧
    
    -- First transfer (1/3 of juice from container 1 to 2)
    after_first_transfer.juice = initial1.juice - (initial1.juice / 3) ∧
    after_first_transfer.water = initial1.water ∧
    
    -- Second transfer (1/4 of mixture from container 2 back to 1)
    final1.juice = after_first_transfer.juice + 
      ((initial2.water + (initial1.juice / 3)) / 4) * ((initial1.juice / 3) / (initial2.water + (initial1.juice / 3))) ∧
    final1.water = ((initial2.water + (initial1.juice / 3)) / 4) * (initial2.water / (initial2.water + (initial1.juice / 3))) ∧
    
    -- Final result
    final1.water / (final1.juice + final1.water) = 3 / 13

/-- The main theorem to prove -/
theorem water_fraction_in_first_container : liquidTransfer := by
  sorry

end NUMINAMATH_CALUDE_water_fraction_in_first_container_l2447_244708


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2447_244767

/-- Represents a three-digit number in base 10 -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundreds_range : hundreds ≥ 1 ∧ hundreds ≤ 9
  tens_range : tens ≥ 0 ∧ tens ≤ 9
  ones_range : ones ≥ 0 ∧ ones ≤ 9
  hundreds_odd : Odd hundreds

/-- Converts a ThreeDigitNumber to its decimal representation -/
def toDecimal (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Sums all permutations of a ThreeDigitNumber -/
def sumPermutations (n : ThreeDigitNumber) : Nat :=
  toDecimal n +
  (100 * n.hundreds + n.tens + 10 * n.ones) +
  (100 * n.tens + 10 * n.ones + n.hundreds) +
  (100 * n.tens + n.hundreds + 10 * n.ones) +
  (100 * n.ones + 10 * n.hundreds + n.tens) +
  (100 * n.ones + 10 * n.tens + n.hundreds)

theorem unique_three_digit_number :
  ∀ n : ThreeDigitNumber, sumPermutations n = 3300 → toDecimal n = 192 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2447_244767


namespace NUMINAMATH_CALUDE_ferry_tourists_l2447_244797

theorem ferry_tourists (a₁ : ℕ) (d : ℕ) (n : ℕ) (h1 : a₁ = 85) (h2 : d = 3) (h3 : n = 5) :
  n * (2 * a₁ + (n - 1) * d) / 2 = 455 := by
  sorry

end NUMINAMATH_CALUDE_ferry_tourists_l2447_244797


namespace NUMINAMATH_CALUDE_rectangle_area_l2447_244773

theorem rectangle_area (side1 side2 : ℚ) (h1 : side1 = 2/3) (h2 : side2 = 3/5) :
  side1 * side2 = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2447_244773


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2447_244702

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 3 * a 3 - 6 * a 3 + 8 = 0) →
  (a 15 * a 15 - 6 * a 15 + 8 = 0) →
  (a 1 * a 17) / a 9 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2447_244702


namespace NUMINAMATH_CALUDE_marcy_serves_36_people_l2447_244786

/-- Represents the makeup supplies and application rates --/
structure MakeupSupplies where
  lip_gloss_per_tube : ℕ
  mascara_per_tube : ℕ
  lip_gloss_tubs : ℕ
  lip_gloss_tubes_per_tub : ℕ
  mascara_tubs : ℕ
  mascara_tubes_per_tub : ℕ

/-- Calculates the number of people that can be served with the given makeup supplies --/
def people_served (supplies : MakeupSupplies) : ℕ :=
  min
    (supplies.lip_gloss_tubs * supplies.lip_gloss_tubes_per_tub * supplies.lip_gloss_per_tube)
    (supplies.mascara_tubs * supplies.mascara_tubes_per_tub * supplies.mascara_per_tube)

/-- Theorem stating that Marcy can serve exactly 36 people with her makeup supplies --/
theorem marcy_serves_36_people :
  let supplies := MakeupSupplies.mk 3 5 6 2 4 3
  people_served supplies = 36 := by
  sorry

#eval people_served (MakeupSupplies.mk 3 5 6 2 4 3)

end NUMINAMATH_CALUDE_marcy_serves_36_people_l2447_244786


namespace NUMINAMATH_CALUDE_factor_expression_l2447_244726

theorem factor_expression (x : ℝ) : 72 * x^4 - 252 * x^9 = 36 * x^4 * (2 - 7 * x^5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2447_244726


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_for_nonnegative_f_l2447_244700

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + 2 / (x + 1) + a * x - 2

theorem min_value_when_a_is_one :
  ∃ (x : ℝ), ∀ (y : ℝ), f 1 x ≤ f 1 y ∧ f 1 x = 0 :=
sorry

theorem range_of_a_for_nonnegative_f :
  ∀ (a : ℝ), (∀ (x : ℝ), x ∈ Set.Icc 0 2 → f a x ≥ 0) ↔ a ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_for_nonnegative_f_l2447_244700


namespace NUMINAMATH_CALUDE_stone_slab_length_l2447_244746

/-- Given a floor area of 67.5 square meters covered by 30 equal-sized square stone slabs,
    prove that the length of each slab is 150 centimeters. -/
theorem stone_slab_length (floor_area : ℝ) (num_slabs : ℕ) (slab_length : ℝ) : 
  floor_area = 67.5 ∧ 
  num_slabs = 30 ∧ 
  floor_area = (num_slabs : ℝ) * slab_length * slab_length / 10000 →
  slab_length = 150 := by
  sorry

end NUMINAMATH_CALUDE_stone_slab_length_l2447_244746


namespace NUMINAMATH_CALUDE_closest_to_fraction_l2447_244754

def options : List ℝ := [0.2, 2, 20, 200, 2000]

theorem closest_to_fraction (x : ℝ) (h : x ∈ options) :
  |403 / 0.21 - 2000| ≤ |403 / 0.21 - x| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_fraction_l2447_244754


namespace NUMINAMATH_CALUDE_solution_difference_l2447_244756

theorem solution_difference (p q : ℝ) : 
  ((p - 4) * (p + 4) = 24 * p - 96) →
  ((q - 4) * (q + 4) = 24 * q - 96) →
  p ≠ q →
  p > q →
  p - q = 16 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l2447_244756


namespace NUMINAMATH_CALUDE_coin_collection_problem_l2447_244728

theorem coin_collection_problem :
  ∀ (n d q : ℕ),
    n + d + q = 30 →
    d = n + 4 →
    5 * n + 10 * d + 25 * q = 410 →
    q = n + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_collection_problem_l2447_244728


namespace NUMINAMATH_CALUDE_candy_distribution_l2447_244734

theorem candy_distribution (tabitha julie carlos stan : ℕ) : 
  tabitha = 22 →
  julie = tabitha / 2 →
  carlos = 2 * stan →
  tabitha + julie + carlos + stan = 72 →
  stan = 13 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l2447_244734


namespace NUMINAMATH_CALUDE_sqrt_23_squared_minus_one_l2447_244776

theorem sqrt_23_squared_minus_one : (Real.sqrt 23 - 1) * (Real.sqrt 23 + 1) = 22 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_23_squared_minus_one_l2447_244776


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2447_244731

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | |x| ≤ 3}

theorem intersection_P_Q : P ∩ Q = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2447_244731


namespace NUMINAMATH_CALUDE_max_games_purchasable_l2447_244710

def initial_amount : ℕ := 35
def spent_amount : ℕ := 7
def game_cost : ℕ := 4

theorem max_games_purchasable :
  (initial_amount - spent_amount) / game_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_games_purchasable_l2447_244710


namespace NUMINAMATH_CALUDE_four_digit_integer_transformation_l2447_244795

theorem four_digit_integer_transformation (A : ℕ) (n : ℕ) :
  (A ≥ 1000 ∧ A < 10000) →
  (∃ a b c d : ℕ,
    A = 1000 * a + 100 * b + 10 * c + d ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    1000 * (a + n) + 100 * (b - n) + 10 * (c + n) + (d - n) = n * A) →
  A = 1818 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_integer_transformation_l2447_244795


namespace NUMINAMATH_CALUDE_min_value_x_plus_reciprocal_l2447_244781

theorem min_value_x_plus_reciprocal (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_reciprocal_l2447_244781


namespace NUMINAMATH_CALUDE_blue_pencil_length_l2447_244787

theorem blue_pencil_length (total : ℝ) (purple : ℝ) (black : ℝ) (blue : ℝ)
  (h_total : total = 4)
  (h_purple : purple = 1.5)
  (h_black : black = 0.5)
  (h_sum : total = purple + black + blue) :
  blue = 2 := by
sorry

end NUMINAMATH_CALUDE_blue_pencil_length_l2447_244787


namespace NUMINAMATH_CALUDE_rotation_result_l2447_244713

/-- Applies a 270° counter-clockwise rotation to a complex number -/
def rotate270 (z : ℂ) : ℂ := -Complex.I * z

/-- The initial complex number -/
def initial : ℂ := 4 - 2 * Complex.I

/-- The result of rotating the initial complex number by 270° counter-clockwise -/
def rotated : ℂ := rotate270 initial

/-- Theorem stating that rotating 4 - 2i by 270° counter-clockwise results in -4i - 2 -/
theorem rotation_result : rotated = -4 * Complex.I - 2 := by sorry

end NUMINAMATH_CALUDE_rotation_result_l2447_244713


namespace NUMINAMATH_CALUDE_apple_cost_price_l2447_244764

theorem apple_cost_price (selling_price : ℚ) (loss_fraction : ℚ) : 
  selling_price = 20 → loss_fraction = 1/6 → 
  ∃ cost_price : ℚ, 
    selling_price = cost_price - (loss_fraction * cost_price) ∧ 
    cost_price = 24 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_price_l2447_244764


namespace NUMINAMATH_CALUDE_full_time_employees_count_l2447_244779

/-- A corporation with part-time and full-time employees -/
structure Corporation where
  total_employees : ℕ
  part_time_employees : ℕ

/-- The number of full-time employees in a corporation -/
def full_time_employees (c : Corporation) : ℕ :=
  c.total_employees - c.part_time_employees

/-- Theorem stating the number of full-time employees in a specific corporation -/
theorem full_time_employees_count (c : Corporation) 
  (h1 : c.total_employees = 65134)
  (h2 : c.part_time_employees = 2041) :
  full_time_employees c = 63093 := by
  sorry

end NUMINAMATH_CALUDE_full_time_employees_count_l2447_244779


namespace NUMINAMATH_CALUDE_same_perimeter_l2447_244725

-- Define the rectangle
def rectangle_length : ℝ := 10
def rectangle_width : ℝ := 8

-- Define the square side length
def square_side : ℝ := 9

-- Define perimeter functions
def rectangle_perimeter (l w : ℝ) : ℝ := 2 * (l + w)
def square_perimeter (s : ℝ) : ℝ := 4 * s

-- Theorem statement
theorem same_perimeter :
  rectangle_perimeter rectangle_length rectangle_width = square_perimeter square_side :=
by sorry

end NUMINAMATH_CALUDE_same_perimeter_l2447_244725


namespace NUMINAMATH_CALUDE_horse_value_is_240_l2447_244709

/-- Represents the payment terms and actual service of a soldier --/
structure SoldierPayment where
  total_payment : ℕ  -- Total payment promised for full service in florins
  service_period : ℕ  -- Full service period in months
  actual_service : ℕ  -- Actual service period in months
  cash_payment : ℕ   -- Cash payment given at the end of actual service

/-- Calculates the value of a horse given to a soldier as part of payment --/
def horse_value (p : SoldierPayment) : ℕ :=
  p.total_payment - (p.total_payment / p.service_period * p.actual_service + p.cash_payment)

/-- Theorem stating the value of the horse in the given problem --/
theorem horse_value_is_240 (p : SoldierPayment) 
  (h1 : p.total_payment = 300)
  (h2 : p.service_period = 36)
  (h3 : p.actual_service = 17)
  (h4 : p.cash_payment = 15) :
  horse_value p = 240 := by
  sorry

end NUMINAMATH_CALUDE_horse_value_is_240_l2447_244709


namespace NUMINAMATH_CALUDE_log_relation_l2447_244741

theorem log_relation (y : ℝ) (k : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → 
  (Real.log 625 / Real.log 2 = k * y) → 
  k = 12 := by sorry

end NUMINAMATH_CALUDE_log_relation_l2447_244741


namespace NUMINAMATH_CALUDE_meena_baked_five_dozens_l2447_244738

/-- The number of cookies in a dozen -/
def cookies_per_dozen : ℕ := 12

/-- The number of dozens of cookies sold to Mr. Stone -/
def dozens_sold_to_stone : ℕ := 2

/-- The number of cookies bought by Brock -/
def cookies_bought_by_brock : ℕ := 7

/-- The number of cookies Meena has left -/
def cookies_left : ℕ := 15

/-- Theorem: Meena baked 5 dozens of cookies initially -/
theorem meena_baked_five_dozens :
  let cookies_sold_to_stone := dozens_sold_to_stone * cookies_per_dozen
  let cookies_bought_by_katy := 2 * cookies_bought_by_brock
  let total_cookies_sold := cookies_sold_to_stone + cookies_bought_by_brock + cookies_bought_by_katy
  let total_cookies := total_cookies_sold + cookies_left
  total_cookies / cookies_per_dozen = 5 := by
  sorry

end NUMINAMATH_CALUDE_meena_baked_five_dozens_l2447_244738


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2447_244706

def solution_set (x : ℝ) : Prop := x ∈ Set.Ici 0 ∩ Set.Iio 2

theorem inequality_solution_set :
  ∀ x : ℝ, x ≠ 2 → (x / (x - 2) ≤ 0 ↔ solution_set x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2447_244706


namespace NUMINAMATH_CALUDE_f_f_neg_two_equals_one_l2447_244766

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -2 then x + 2
  else if x < 3 then 2^x
  else Real.log x

-- State the theorem
theorem f_f_neg_two_equals_one : f (f (-2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_f_neg_two_equals_one_l2447_244766


namespace NUMINAMATH_CALUDE_feuerbach_circle_equation_l2447_244780

/-- The Feuerbach circle (nine-point circle) of a triangle -/
def feuerbach_circle (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * c * (p.1^2 + p.2^2) - (a + b) * c * p.1 + (a * b - c^2) * p.2 = 0}

/-- The vertices of the triangle -/
def triangle_vertices (a b c : ℝ) : Set (ℝ × ℝ) :=
  {(a, 0), (b, 0), (0, c)}

theorem feuerbach_circle_equation (a b c : ℝ) (h : c ≠ 0) :
  ∃ (circle : Set (ℝ × ℝ)), circle = feuerbach_circle a b c ∧
  (∀ (p : ℝ × ℝ), p ∈ circle ↔ 2 * c * (p.1^2 + p.2^2) - (a + b) * c * p.1 + (a * b - c^2) * p.2 = 0) :=
sorry

end NUMINAMATH_CALUDE_feuerbach_circle_equation_l2447_244780


namespace NUMINAMATH_CALUDE_volume_increase_rectangular_prism_l2447_244719

/-- Theorem: Increase in volume of a rectangular prism -/
theorem volume_increase_rectangular_prism 
  (L B H : ℝ) 
  (h_positive : L > 0 ∧ B > 0 ∧ H > 0) :
  let V_original := L * B * H
  let V_new := (L * 1.15) * (B * 1.30) * (H * 1.20)
  (V_new - V_original) / V_original = 0.794 := by
  sorry

end NUMINAMATH_CALUDE_volume_increase_rectangular_prism_l2447_244719


namespace NUMINAMATH_CALUDE_gcd_360_504_l2447_244739

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by sorry

end NUMINAMATH_CALUDE_gcd_360_504_l2447_244739


namespace NUMINAMATH_CALUDE_product_remainder_l2447_244730

theorem product_remainder (a b : ℕ) (ha : a % 3 = 2) (hb : b % 3 = 2) : (a * b) % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l2447_244730


namespace NUMINAMATH_CALUDE_fiftieth_term_is_198_l2447_244765

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

/-- The 50th term of the specific arithmetic sequence -/
theorem fiftieth_term_is_198 :
  arithmeticSequenceTerm 2 4 50 = 198 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_198_l2447_244765


namespace NUMINAMATH_CALUDE_x_value_l2447_244791

theorem x_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2447_244791


namespace NUMINAMATH_CALUDE_rational_representation_l2447_244747

theorem rational_representation (q : ℚ) (hq : q > 0) :
  ∃ (a b c d : ℕ+), q = (a^2021 + b^2023) / (c^2022 + d^2024) :=
by sorry

end NUMINAMATH_CALUDE_rational_representation_l2447_244747


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l2447_244757

/-- The number of distinct diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

/-- Theorem: The number of distinct diagonals in a convex dodecagon is 54 -/
theorem dodecagon_diagonals : num_diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l2447_244757


namespace NUMINAMATH_CALUDE_max_sum_square_roots_equality_condition_l2447_244758

theorem max_sum_square_roots (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 25) :
  Real.sqrt (x + 16) + Real.sqrt (25 - x) + 2 * Real.sqrt x ≤ 11 + 4 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x : ℝ) (h : x = 16) :
  Real.sqrt (x + 16) + Real.sqrt (25 - x) + 2 * Real.sqrt x = 11 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_square_roots_equality_condition_l2447_244758


namespace NUMINAMATH_CALUDE_pau_total_cost_l2447_244705

/-- Represents the cost of fried chicken orders for three people -/
def fried_chicken_cost 
  (kobe_pieces : ℕ) 
  (kobe_price : ℚ) 
  (pau_multiplier : ℕ) 
  (pau_extra : ℚ) 
  (pau_price : ℚ) 
  (shaq_multiplier : ℚ) 
  (discount : ℚ) : ℚ :=
  let pau_pieces := pau_multiplier * kobe_pieces + pau_extra
  let pau_initial := pau_pieces * pau_price
  pau_initial + pau_initial * (1 - discount)

/-- Theorem stating the total cost of Pau's fried chicken orders -/
theorem pau_total_cost : 
  fried_chicken_cost 5 (175/100) 2 (5/2) (3/2) (3/2) (15/100) = 346875/10000 := by
  sorry

end NUMINAMATH_CALUDE_pau_total_cost_l2447_244705


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2447_244782

/-- Proves the range of m for which x^2 - 3x - m = 0 has two unequal real roots -/
theorem quadratic_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 3*x - m = 0 ∧ y^2 - 3*y - m = 0) ↔ m > -9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2447_244782


namespace NUMINAMATH_CALUDE_certain_number_proof_l2447_244743

theorem certain_number_proof : ∃ x : ℝ, (0.80 * x = 0.50 * 960) ∧ (x = 600) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2447_244743


namespace NUMINAMATH_CALUDE_simplest_common_denominator_l2447_244733

variable (x : ℝ)

theorem simplest_common_denominator :
  ∃ (d : ℝ), d = x * (x + 1) * (x - 1) ∧
  (∃ (a b : ℝ), a / (x^2 - 1) + b / (x^2 + x) = (a * (x^2 + x) + b * (x^2 - 1)) / d) ∧
  (∀ (d' : ℝ), (∃ (a' b' : ℝ), a' / (x^2 - 1) + b' / (x^2 + x) = (a' * (x^2 + x) + b' * (x^2 - 1)) / d') →
    d ∣ d') :=
sorry

end NUMINAMATH_CALUDE_simplest_common_denominator_l2447_244733


namespace NUMINAMATH_CALUDE_minimum_bailing_rate_for_steve_and_leroy_l2447_244755

/-- Represents the fishing scenario with Steve and LeRoy --/
structure FishingScenario where
  distance_to_shore : ℝ
  water_intake_rate : ℝ
  boat_capacity : ℝ
  rowing_speed : ℝ

/-- Calculates the minimum bailing rate required to reach shore without sinking --/
def minimum_bailing_rate (scenario : FishingScenario) : ℝ :=
  sorry

/-- Theorem stating the minimum bailing rate for the given scenario --/
theorem minimum_bailing_rate_for_steve_and_leroy :
  let scenario : FishingScenario := {
    distance_to_shore := 2,
    water_intake_rate := 12,
    boat_capacity := 40,
    rowing_speed := 3
  }
  minimum_bailing_rate scenario = 11 := by sorry

end NUMINAMATH_CALUDE_minimum_bailing_rate_for_steve_and_leroy_l2447_244755


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l2447_244704

/-- A function that generates all three-digit numbers using the digits 5, 6, and 7 only once -/
def threeDigitNumbers : List Nat := sorry

/-- The smallest three-digit number formed using 5, 6, and 7 only once -/
def smallestNumber : Nat := sorry

/-- The largest three-digit number formed using 5, 6, and 7 only once -/
def largestNumber : Nat := sorry

/-- Theorem stating that the sum of the largest and smallest three-digit numbers
    formed using 5, 6, and 7 only once is 1332 -/
theorem sum_of_largest_and_smallest : smallestNumber + largestNumber = 1332 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l2447_244704


namespace NUMINAMATH_CALUDE_sum_fraction_bounds_l2447_244774

theorem sum_fraction_bounds (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let S := (a / (a + b + d)) + (b / (b + c + a)) + (c / (c + d + b)) + (d / (d + a + c))
  1 < S ∧ S < 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_fraction_bounds_l2447_244774


namespace NUMINAMATH_CALUDE_willies_stickers_l2447_244775

theorem willies_stickers (starting_stickers : Real) (received_stickers : Real) :
  starting_stickers = 36.0 →
  received_stickers = 7.0 →
  starting_stickers + received_stickers = 43.0 := by
  sorry

end NUMINAMATH_CALUDE_willies_stickers_l2447_244775


namespace NUMINAMATH_CALUDE_inequality_system_sum_l2447_244740

theorem inequality_system_sum (a b : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ (x + 2*a > 4 ∧ 2*x < b)) → 
  a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_sum_l2447_244740


namespace NUMINAMATH_CALUDE_johns_weight_l2447_244790

/-- Given that Roy weighs 4 pounds and John is 77 pounds heavier than Roy,
    prove that John weighs 81 pounds. -/
theorem johns_weight (roy_weight : ℕ) (weight_difference : ℕ) :
  roy_weight = 4 →
  weight_difference = 77 →
  roy_weight + weight_difference = 81 :=
by sorry

end NUMINAMATH_CALUDE_johns_weight_l2447_244790


namespace NUMINAMATH_CALUDE_z_sixth_power_l2447_244715

theorem z_sixth_power (z : ℂ) : z = (-Real.sqrt 3 - Complex.I) / 2 → z^6 = -1 := by
  sorry

end NUMINAMATH_CALUDE_z_sixth_power_l2447_244715


namespace NUMINAMATH_CALUDE_determinant_trigonometric_matrix_l2447_244718

open Matrix Real

theorem determinant_trigonometric_matrix (θ φ : ℝ) : 
  det ![![0, cos θ, sin θ],
       ![-cos θ, 0, cos (θ + φ)],
       ![-sin θ, -cos (θ + φ), 0]] = -sin θ * cos θ * cos (θ + φ) := by
  sorry

end NUMINAMATH_CALUDE_determinant_trigonometric_matrix_l2447_244718


namespace NUMINAMATH_CALUDE_remainder_101_35_mod_100_l2447_244736

theorem remainder_101_35_mod_100 : 101^35 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_35_mod_100_l2447_244736


namespace NUMINAMATH_CALUDE_baseball_stats_l2447_244770

-- Define the total number of hits and the number of each type of hit
def total_hits : ℕ := 45
def home_runs : ℕ := 2
def triples : ℕ := 3
def doubles : ℕ := 6

-- Define the number of singles
def singles : ℕ := total_hits - (home_runs + triples + doubles)

-- Define the percentage of singles
def singles_percentage : ℚ := (singles : ℚ) / (total_hits : ℚ) * 100

-- Theorem to prove
theorem baseball_stats :
  singles = 34 ∧ singles_percentage = 75.56 := by
  sorry

end NUMINAMATH_CALUDE_baseball_stats_l2447_244770


namespace NUMINAMATH_CALUDE_triangle_area_l2447_244794

/-- Given a triangle MNP where:
  * MN is the side opposite to the 60° angle
  * MP is the hypotenuse with length 40
  * Angle N is 90°
  Prove that the area of triangle MNP is 200√3 -/
theorem triangle_area (M N P : ℝ × ℝ) : 
  let MN := Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)
  let MP := Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2)
  let NP := Real.sqrt ((N.1 - P.1)^2 + (N.2 - P.2)^2)
  (∃ θ : Real, θ = π/3 ∧ MN = MP * Real.sin θ) →  -- MN is opposite to 60° angle
  MP = 40 →  -- MP is the hypotenuse with length 40
  (N.1 - M.1) * (P.1 - M.1) + (N.2 - M.2) * (P.2 - M.2) = 0 →  -- Angle N is 90°
  (1/2) * MN * NP = 200 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2447_244794


namespace NUMINAMATH_CALUDE_base_five_digits_of_1250_l2447_244729

theorem base_five_digits_of_1250 : ∃ n : ℕ, n > 0 ∧ 5^(n-1) ≤ 1250 ∧ 1250 < 5^n ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_five_digits_of_1250_l2447_244729


namespace NUMINAMATH_CALUDE_equal_perimeter_parallel_sections_l2447_244771

/-- A tetrahedron with vertices A, B, C, and D -/
structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

/-- A plane that intersects a tetrahedron -/
structure IntersectingPlane where
  plane : Plane
  tetrahedron : Tetrahedron

/-- The perimeter of the intersection between a plane and a tetrahedron -/
def intersectionPerimeter (p : IntersectingPlane) : ℝ := sorry

/-- Two edges of a tetrahedron are disjoint -/
def disjointEdges (t : Tetrahedron) (e1 e2 : Segment) : Prop := sorry

/-- A plane is parallel to two edges of a tetrahedron -/
def parallelToEdges (p : IntersectingPlane) (e1 e2 : Segment) : Prop := sorry

/-- The length of a segment -/
def length (s : Segment) : ℝ := sorry

theorem equal_perimeter_parallel_sections (t : Tetrahedron) 
  (e1 e2 : Segment) (p1 p2 : IntersectingPlane) :
  disjointEdges t e1 e2 →
  length e1 = length e2 →
  parallelToEdges p1 e1 e2 →
  parallelToEdges p2 e1 e2 →
  intersectionPerimeter p1 = intersectionPerimeter p2 := by
  sorry

end NUMINAMATH_CALUDE_equal_perimeter_parallel_sections_l2447_244771


namespace NUMINAMATH_CALUDE_wire_length_ratio_l2447_244742

/-- The ratio of wire lengths in cube frame construction -/
theorem wire_length_ratio : 
  ∀ (bonnie_wire_length roark_wire_length : ℕ) 
    (bonnie_cube_volume roark_total_volume : ℕ),
  bonnie_wire_length = 12 * 8 →
  bonnie_cube_volume = 8^3 →
  roark_total_volume = bonnie_cube_volume →
  (∃ (num_small_cubes : ℕ), 
    roark_total_volume = num_small_cubes * 2^3 ∧
    roark_wire_length = num_small_cubes * 12 * 2) →
  (bonnie_wire_length : ℚ) / roark_wire_length = 1 / 16 := by
sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l2447_244742


namespace NUMINAMATH_CALUDE_tangent_equation_solution_l2447_244720

open Real

theorem tangent_equation_solution (x : ℝ) :
  tan x + tan (50 * π / 180) + tan (70 * π / 180) = tan x * tan (50 * π / 180) * tan (70 * π / 180) →
  ∃ n : ℤ, x = (60 + 180 * n) * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_tangent_equation_solution_l2447_244720


namespace NUMINAMATH_CALUDE_tan_half_sum_l2447_244793

theorem tan_half_sum (a b : Real) 
  (h1 : Real.cos a + Real.cos b = 3/5)
  (h2 : Real.sin a + Real.sin b = 1/5) :
  Real.tan ((a + b) / 2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_tan_half_sum_l2447_244793


namespace NUMINAMATH_CALUDE_power_2021_representation_l2447_244792

theorem power_2021_representation (n : ℕ+) :
  (∃ (x y : ℤ), (2021 : ℤ)^(n : ℕ) = x^4 - 4*y^4) ↔ 4 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_power_2021_representation_l2447_244792


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2447_244732

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 18 →
  a * b + c + d = 83 →
  a * d + b * c = 174 →
  c * d = 105 →
  a^2 + b^2 + c^2 + d^2 ≤ 702 :=
by
  sorry


end NUMINAMATH_CALUDE_max_sum_of_squares_l2447_244732


namespace NUMINAMATH_CALUDE_time_for_order_l2447_244744

/-- Represents the time it takes to make one shirt -/
def shirt_time : ℝ := 1

/-- Represents the time it takes to make one pair of pants -/
def pants_time : ℝ := 2 * shirt_time

/-- Represents the time it takes to make one jacket -/
def jacket_time : ℝ := 3 * shirt_time

/-- The total time to make 2 shirts, 3 pairs of pants, and 4 jackets is 10 hours -/
axiom total_time_10 : 2 * shirt_time + 3 * pants_time + 4 * jacket_time = 10

/-- Theorem: It takes 20 working hours to make 14 shirts, 10 pairs of pants, and 2 jackets -/
theorem time_for_order : 14 * shirt_time + 10 * pants_time + 2 * jacket_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_time_for_order_l2447_244744


namespace NUMINAMATH_CALUDE_garden_expenses_l2447_244796

/-- Calculate the total expenses for flowers in a garden --/
theorem garden_expenses (tulips carnations roses : ℕ) (price : ℚ) : 
  tulips = 250 → 
  carnations = 375 → 
  roses = 320 → 
  price = 2 → 
  (tulips + carnations + roses : ℚ) * price = 1890 := by
sorry

end NUMINAMATH_CALUDE_garden_expenses_l2447_244796


namespace NUMINAMATH_CALUDE_square_equals_product_sum_solutions_l2447_244759

theorem square_equals_product_sum_solutions :
  ∀ (a b : ℤ), a ≥ 0 → b ≥ 0 → a^2 = b * (b + 7) → (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end NUMINAMATH_CALUDE_square_equals_product_sum_solutions_l2447_244759


namespace NUMINAMATH_CALUDE_sector_perimeter_example_l2447_244769

/-- The perimeter of a sector with given radius and central angle -/
def sector_perimeter (r : ℝ) (θ : ℝ) : ℝ := r * θ + 2 * r

/-- Theorem: The perimeter of a sector with radius 1.5 and central angle 2 radians is 6 -/
theorem sector_perimeter_example : sector_perimeter 1.5 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_example_l2447_244769


namespace NUMINAMATH_CALUDE_determinant_transformation_l2447_244722

theorem determinant_transformation (x y z w : ℝ) :
  (x * w - y * z = 3) →
  (x * (5 * z + 4 * w) - z * (5 * x + 4 * y) = 12) :=
by sorry

end NUMINAMATH_CALUDE_determinant_transformation_l2447_244722


namespace NUMINAMATH_CALUDE_ratio_equality_l2447_244714

theorem ratio_equality (p q r u v w : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l2447_244714


namespace NUMINAMATH_CALUDE_power_two_33_mod_9_l2447_244727

theorem power_two_33_mod_9 : 2^33 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_two_33_mod_9_l2447_244727


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2447_244784

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (x > -a ∧ x > -b) ↔ x > -b) → a ≥ b := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2447_244784


namespace NUMINAMATH_CALUDE_correlation_coefficient_properties_l2447_244711

-- Define the correlation coefficient
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

-- Define the concept of increasing
def increasing (f : ℝ → ℝ) : Prop := 
  ∀ a b, a < b → f a < f b

-- Define the concept of linear correlation strength
def linear_correlation_strength (r : ℝ) : ℝ := sorry

-- Define the concept of functional relationship
def functional_relationship (x y : ℝ → ℝ) : Prop := sorry

theorem correlation_coefficient_properties 
  (x y : ℝ → ℝ) (r : ℝ) (h : r = correlation_coefficient x y) :
  (r > 0 → increasing y) ∧ 
  (∀ s : ℝ, abs s < abs r → linear_correlation_strength s < linear_correlation_strength r) ∧
  ((r = 1 ∨ r = -1) → functional_relationship x y) := by
  sorry

end NUMINAMATH_CALUDE_correlation_coefficient_properties_l2447_244711


namespace NUMINAMATH_CALUDE_tea_cups_problem_l2447_244763

theorem tea_cups_problem (total_tea : ℕ) (cup_capacity : ℕ) (h1 : total_tea = 1050) (h2 : cup_capacity = 65) :
  (total_tea / cup_capacity : ℕ) = 16 :=
by sorry

end NUMINAMATH_CALUDE_tea_cups_problem_l2447_244763


namespace NUMINAMATH_CALUDE_find_divisor_l2447_244735

theorem find_divisor (n : ℕ) (k : ℕ) (h1 : n + k = 8261966) (h2 : k = 11) :
  11 ∣ n + k :=
sorry

end NUMINAMATH_CALUDE_find_divisor_l2447_244735


namespace NUMINAMATH_CALUDE_lost_card_number_l2447_244716

theorem lost_card_number (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ∈ Set.range (λ i => i : Fin n → ℕ)) : 
  ∃ (k : Fin n), k.val + 1 = 4 ∧ (n * (n + 1)) / 2 - (k.val + 1) = 101 :=
by sorry

end NUMINAMATH_CALUDE_lost_card_number_l2447_244716


namespace NUMINAMATH_CALUDE_number_exceeding_16_percent_l2447_244799

theorem number_exceeding_16_percent : ∃ x : ℝ, x = 0.16 * x + 21 ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_16_percent_l2447_244799


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l2447_244748

theorem quadratic_always_nonnegative_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (1 - a)*x + 1 ≥ 0) → a ∈ Set.Icc (-1 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l2447_244748


namespace NUMINAMATH_CALUDE_problem_statement_l2447_244768

theorem problem_statement (s x y : ℝ) 
  (h1 : s > 0) 
  (h2 : s ≠ 1) 
  (h3 : x * y ≠ 0) 
  (h4 : s * x > y) : 
  s < y / x := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2447_244768


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2447_244778

theorem min_value_of_expression (x y : ℝ) : (2*x*y - 3)^2 + (x - y)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2447_244778


namespace NUMINAMATH_CALUDE_cost_of_sneakers_l2447_244788

/-- Given the costs of items John bought, prove the cost of sneakers. -/
theorem cost_of_sneakers
  (total_cost : ℕ)
  (racket_cost : ℕ)
  (outfit_cost : ℕ)
  (h1 : total_cost = 750)
  (h2 : racket_cost = 300)
  (h3 : outfit_cost = 250) :
  total_cost - racket_cost - outfit_cost = 200 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_sneakers_l2447_244788


namespace NUMINAMATH_CALUDE_largest_k_value_l2447_244745

theorem largest_k_value (x y k : ℝ) : 
  (2 * x + y = k) →
  (3 * x + y = 3) →
  (x - 2 * y ≥ 1) →
  (∀ m : ℤ, m > k → ¬(∃ x' y' : ℝ, 2 * x' + y' = m ∧ 3 * x' + y' = 3 ∧ x' - 2 * y' ≥ 1)) →
  k ≤ 2 ∧ (∃ x' y' : ℝ, 2 * x' + y' = 2 ∧ 3 * x' + y' = 3 ∧ x' - 2 * y' ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_k_value_l2447_244745
