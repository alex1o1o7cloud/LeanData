import Mathlib

namespace NUMINAMATH_CALUDE_mason_hotdog_weight_l1734_173455

/-- Represents the weight of different food items in ounces -/
def food_weight : (String → Nat)
| "hotdog" => 2
| "burger" => 5
| "pie" => 10
| _ => 0

/-- The number of burgers Noah ate -/
def noah_burgers : Nat := 8

/-- Calculates the number of pies Jacob ate -/
def jacob_pies : Nat := noah_burgers - 3

/-- Calculates the number of hotdogs Mason ate -/
def mason_hotdogs : Nat := 3 * jacob_pies

/-- Theorem stating that the total weight of hotdogs Mason ate is 30 ounces -/
theorem mason_hotdog_weight :
  mason_hotdogs * food_weight "hotdog" = 30 := by
  sorry

end NUMINAMATH_CALUDE_mason_hotdog_weight_l1734_173455


namespace NUMINAMATH_CALUDE_apple_price_is_two_l1734_173488

/-- The cost of items in Fabian's shopping basket -/
def shopping_cost (apple_price : ℝ) : ℝ :=
  5 * apple_price +  -- 5 kg of apples
  3 * (apple_price - 1) +  -- 3 packs of sugar
  0.5 * 6  -- 500g of walnuts

/-- Theorem: The price of apples is $2 per kg -/
theorem apple_price_is_two :
  ∃ (apple_price : ℝ), apple_price = 2 ∧ shopping_cost apple_price = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_price_is_two_l1734_173488


namespace NUMINAMATH_CALUDE_rational_function_inequality_l1734_173415

theorem rational_function_inequality (f : ℚ → ℤ) :
  ∃ a b : ℚ, (f a + f b : ℚ) / 2 ≤ f ((a + b) / 2) := by
  sorry

end NUMINAMATH_CALUDE_rational_function_inequality_l1734_173415


namespace NUMINAMATH_CALUDE_nell_card_difference_l1734_173430

/-- Represents the number of cards Nell has -/
structure CardCount where
  baseball : ℕ
  ace : ℕ

/-- The difference between baseball and ace cards -/
def cardDifference (cards : CardCount) : ℤ :=
  cards.baseball - cards.ace

theorem nell_card_difference (initial final : CardCount) 
  (h1 : initial.baseball = 438)
  (h2 : initial.ace = 18)
  (h3 : final.baseball = 178)
  (h4 : final.ace = 55) :
  cardDifference final = 123 := by
  sorry

end NUMINAMATH_CALUDE_nell_card_difference_l1734_173430


namespace NUMINAMATH_CALUDE_direction_vector_b_value_l1734_173448

/-- Given a line passing through points (-6, 0) and (-3, 3) with direction vector (3, b), prove b = 3 -/
theorem direction_vector_b_value (b : ℝ) : b = 3 :=
  by
  -- Define the two points on the line
  let p1 : Fin 2 → ℝ := ![- 6, 0]
  let p2 : Fin 2 → ℝ := ![- 3, 3]
  
  -- Define the direction vector of the line
  let dir : Fin 2 → ℝ := ![3, b]
  
  -- Assert that the direction vector is parallel to the vector between the two points
  have h : ∃ (k : ℝ), k ≠ 0 ∧ (λ i => p2 i - p1 i) = (λ i => k * dir i) := by sorry
  
  sorry

end NUMINAMATH_CALUDE_direction_vector_b_value_l1734_173448


namespace NUMINAMATH_CALUDE_trace_bag_weight_proof_l1734_173483

/-- The weight of one of Trace's shopping bags -/
def trace_bag_weight (
  trace_bags : ℕ)
  (gordon_bags : ℕ)
  (gordon_bag1_weight : ℕ)
  (gordon_bag2_weight : ℕ)
  (lola_bags : ℕ)
  (lola_total_weight : ℕ)
  : ℕ :=
2

theorem trace_bag_weight_proof 
  (trace_bags : ℕ)
  (gordon_bags : ℕ)
  (gordon_bag1_weight : ℕ)
  (gordon_bag2_weight : ℕ)
  (lola_bags : ℕ)
  (lola_total_weight : ℕ)
  (h1 : trace_bags = 5)
  (h2 : gordon_bags = 2)
  (h3 : gordon_bag1_weight = 3)
  (h4 : gordon_bag2_weight = 7)
  (h5 : lola_bags = 4)
  (h6 : trace_bags * trace_bag_weight trace_bags gordon_bags gordon_bag1_weight gordon_bag2_weight lola_bags lola_total_weight = gordon_bag1_weight + gordon_bag2_weight)
  (h7 : lola_total_weight = gordon_bag1_weight + gordon_bag2_weight - 2)
  : trace_bag_weight trace_bags gordon_bags gordon_bag1_weight gordon_bag2_weight lola_bags lola_total_weight = 2 := by
  sorry

#check trace_bag_weight_proof

end NUMINAMATH_CALUDE_trace_bag_weight_proof_l1734_173483


namespace NUMINAMATH_CALUDE_manuscript_cost_theorem_l1734_173406

/-- Represents the cost calculation for manuscript printing and binding --/
def manuscript_cost (
  num_copies : ℕ
  ) (
  total_pages : ℕ
  ) (
  color_pages : ℕ
  ) (
  bw_cost : ℚ
  ) (
  color_cost : ℚ
  ) (
  binding_cost : ℚ
  ) (
  index_cost : ℚ
  ) (
  rush_copies : ℕ
  ) (
  rush_cost : ℚ
  ) (
  binding_discount_rate : ℚ
  ) (
  bundle_discount : ℚ
  ) : ℚ :=
  let bw_pages := total_pages - color_pages
  let print_cost := (bw_pages : ℚ) * bw_cost + (color_pages : ℚ) * color_cost
  let additional_cost := binding_cost + index_cost - bundle_discount
  let copy_cost := print_cost + additional_cost
  let total_before_discount := (num_copies : ℚ) * copy_cost
  let binding_discount := (num_copies : ℚ) * binding_cost * binding_discount_rate
  let rush_fee := (rush_copies : ℚ) * rush_cost
  total_before_discount - binding_discount + rush_fee

/-- Theorem stating the total cost for the manuscript printing and binding --/
theorem manuscript_cost_theorem :
  manuscript_cost 10 400 50 (5/100) (1/10) 5 2 5 3 (1/10) (1/2) = 300 :=
by sorry

end NUMINAMATH_CALUDE_manuscript_cost_theorem_l1734_173406


namespace NUMINAMATH_CALUDE_arnel_pencil_boxes_l1734_173465

/-- The number of boxes of pencils Arnel had -/
def number_of_boxes : ℕ := sorry

/-- The number of pencils in each box -/
def pencils_per_box : ℕ := 5

/-- The number of pencils Arnel kept for himself -/
def pencils_kept : ℕ := 10

/-- The number of Arnel's friends -/
def number_of_friends : ℕ := 5

/-- The number of pencils each friend received -/
def pencils_per_friend : ℕ := 8

theorem arnel_pencil_boxes :
  number_of_boxes = 10 ∧
  number_of_boxes * pencils_per_box = 
    pencils_kept + number_of_friends * pencils_per_friend :=
by sorry

end NUMINAMATH_CALUDE_arnel_pencil_boxes_l1734_173465


namespace NUMINAMATH_CALUDE_makeup_fraction_of_savings_l1734_173468

/-- Given Leila's original savings and the cost of a sweater, prove the fraction spent on make-up -/
theorem makeup_fraction_of_savings (original_savings : ℚ) (sweater_cost : ℚ) 
  (h1 : original_savings = 80)
  (h2 : sweater_cost = 20) :
  (original_savings - sweater_cost) / original_savings = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_makeup_fraction_of_savings_l1734_173468


namespace NUMINAMATH_CALUDE_sum_of_squares_bound_l1734_173437

theorem sum_of_squares_bound (x y z t : ℝ) 
  (h1 : |x + y + z - t| ≤ 1)
  (h2 : |y + z + t - x| ≤ 1)
  (h3 : |z + t + x - y| ≤ 1)
  (h4 : |t + x + y - z| ≤ 1) :
  x^2 + y^2 + z^2 + t^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_bound_l1734_173437


namespace NUMINAMATH_CALUDE_polynomial_equality_l1734_173489

theorem polynomial_equality (p : ℝ → ℝ) :
  (∀ x, p x + (x^5 + 3*x^3 + 9*x) = 7*x^3 + 24*x^2 + 25*x + 1) →
  (∀ x, p x = -x^5 + 4*x^3 + 24*x^2 + 16*x + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1734_173489


namespace NUMINAMATH_CALUDE_pure_imaginary_iff_real_zero_imag_nonzero_l1734_173434

/-- A complex number is pure imaginary if and only if its real part is zero and its imaginary part is non-zero -/
theorem pure_imaginary_iff_real_zero_imag_nonzero (z : ℂ) :
  (∃ b : ℝ, b ≠ 0 ∧ z = Complex.I * b) ↔ (z.re = 0 ∧ z.im ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_iff_real_zero_imag_nonzero_l1734_173434


namespace NUMINAMATH_CALUDE_james_bills_denomination_l1734_173413

/-- Proves that the denomination of each bill James found is $20 -/
theorem james_bills_denomination (initial_amount : ℕ) (final_amount : ℕ) (num_bills : ℕ) :
  initial_amount = 75 →
  final_amount = 135 →
  num_bills = 3 →
  (final_amount - initial_amount) / num_bills = 20 :=
by sorry

end NUMINAMATH_CALUDE_james_bills_denomination_l1734_173413


namespace NUMINAMATH_CALUDE_a_outside_interval_l1734_173486

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def decreasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≥ f y

-- State the theorem
theorem a_outside_interval (f : ℝ → ℝ) (a : ℝ) 
  (h_even : is_even f) 
  (h_decreasing : decreasing_on_nonpositive f) 
  (h_inequality : f a > f 2) : 
  a < -2 ∨ a > 2 :=
sorry

end NUMINAMATH_CALUDE_a_outside_interval_l1734_173486


namespace NUMINAMATH_CALUDE_inequality_solution_interval_l1734_173435

theorem inequality_solution_interval (x : ℝ) : 
  (1 / (x^2 + 1) > 3 / x + 13 / 10) ↔ -2 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_interval_l1734_173435


namespace NUMINAMATH_CALUDE_parallelogram_area_parallelogram_area_proof_l1734_173419

/-- The area of a parallelogram with base 12 cm and height 10 cm is 120 cm². -/
theorem parallelogram_area : ℝ → ℝ → ℝ → Prop :=
  fun base height area =>
    base = 12 ∧ height = 10 → area = base * height → area = 120

#check parallelogram_area

-- Proof
theorem parallelogram_area_proof : parallelogram_area 12 10 120 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_parallelogram_area_proof_l1734_173419


namespace NUMINAMATH_CALUDE_least_tiles_required_l1734_173487

def room_length : ℕ := 544
def room_width : ℕ := 374

theorem least_tiles_required (length width : ℕ) (h1 : length = room_length) (h2 : width = room_width) :
  ∃ (tile_size : ℕ), 
    tile_size > 0 ∧
    length % tile_size = 0 ∧
    width % tile_size = 0 ∧
    (length / tile_size) * (width / tile_size) = 176 :=
sorry

end NUMINAMATH_CALUDE_least_tiles_required_l1734_173487


namespace NUMINAMATH_CALUDE_sequence_sum_l1734_173469

def S (n : ℕ) : ℤ :=
  if n % 2 = 1 then (n + 1) / 2 else -(n / 2)

theorem sequence_sum : S 19 * S 31 + S 48 = 136 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l1734_173469


namespace NUMINAMATH_CALUDE_cyclic_iff_perpendicular_diagonals_l1734_173491

-- Define the basic geometric objects
variable (A B C D P Q R S : Point)

-- Define the quadrilateral ABCD
def is_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the incircle and its tangency points
def has_incircle_with_tangent_points (A B C D P Q R S : Point) : Prop := sorry

-- Define cyclic quadrilateral
def is_cyclic (A B C D : Point) : Prop := sorry

-- Define perpendicularity
def perpendicular (P Q R S : Point) : Prop := sorry

-- The main theorem
theorem cyclic_iff_perpendicular_diagonals 
  (h_quad : is_quadrilateral A B C D)
  (h_incircle : has_incircle_with_tangent_points A B C D P Q R S) :
  is_cyclic A B C D ↔ perpendicular P R Q S := by sorry

end NUMINAMATH_CALUDE_cyclic_iff_perpendicular_diagonals_l1734_173491


namespace NUMINAMATH_CALUDE_common_tangents_count_l1734_173443

/-- Circle C₁ with equation x² + y² + 2x + 8y + 16 = 0 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + 8*p.2 + 16 = 0}

/-- Circle C₂ with equation x² + y² - 4x - 4y - 1 = 0 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 4*p.2 - 1 = 0}

/-- The number of common tangents to circles C₁ and C₂ -/
def numCommonTangents : ℕ := 4

/-- Theorem stating that the number of common tangents to C₁ and C₂ is 4 -/
theorem common_tangents_count :
  numCommonTangents = 4 :=
sorry

end NUMINAMATH_CALUDE_common_tangents_count_l1734_173443


namespace NUMINAMATH_CALUDE_value_of_b_l1734_173454

theorem value_of_b (b : ℚ) (h : b + 2 * b / 5 = 22 / 5) : b = 22 / 7 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l1734_173454


namespace NUMINAMATH_CALUDE_fireflies_joining_l1734_173414

theorem fireflies_joining (initial : ℕ) (joined : ℕ) (left : ℕ) (remaining : ℕ) : 
  initial = 3 → left = 2 → remaining = 9 → initial + joined - left = remaining → joined = 8 := by
  sorry

end NUMINAMATH_CALUDE_fireflies_joining_l1734_173414


namespace NUMINAMATH_CALUDE_pauls_money_duration_l1734_173457

/-- Given Paul's earnings and weekly spending, prove how long his money will last. -/
theorem pauls_money_duration (lawn_mowing : ℕ) (weed_eating : ℕ) (weekly_spending : ℕ) :
  lawn_mowing = 68 →
  weed_eating = 13 →
  weekly_spending = 9 →
  (lawn_mowing + weed_eating) / weekly_spending = 9 := by
  sorry

#check pauls_money_duration

end NUMINAMATH_CALUDE_pauls_money_duration_l1734_173457


namespace NUMINAMATH_CALUDE_not_proportional_D_l1734_173440

/-- Represents a relation between x and y --/
inductive Relation
  | DirectlyProportional
  | InverselyProportional
  | Neither

/-- Determines the type of relation between x and y given an equation --/
def determineRelation (equation : ℝ → ℝ → Prop) : Relation :=
  sorry

/-- The equation x + y = 0 --/
def equationA (x y : ℝ) : Prop := x + y = 0

/-- The equation 3xy = 10 --/
def equationB (x y : ℝ) : Prop := 3 * x * y = 10

/-- The equation x = 5y --/
def equationC (x y : ℝ) : Prop := x = 5 * y

/-- The equation 3x + y = 10 --/
def equationD (x y : ℝ) : Prop := 3 * x + y = 10

/-- The equation x/y = √3 --/
def equationE (x y : ℝ) : Prop := x / y = Real.sqrt 3

theorem not_proportional_D :
  determineRelation equationD = Relation.Neither ∧
  determineRelation equationA ≠ Relation.Neither ∧
  determineRelation equationB ≠ Relation.Neither ∧
  determineRelation equationC ≠ Relation.Neither ∧
  determineRelation equationE ≠ Relation.Neither :=
  sorry

end NUMINAMATH_CALUDE_not_proportional_D_l1734_173440


namespace NUMINAMATH_CALUDE_cuboidal_box_volume_l1734_173424

/-- A cuboidal box with given adjacent face areas has a specific volume -/
theorem cuboidal_box_volume (l w h : ℝ) (h1 : l * w = 120) (h2 : w * h = 72) (h3 : h * l = 60) :
  l * w * h = 4320 := by
  sorry

end NUMINAMATH_CALUDE_cuboidal_box_volume_l1734_173424


namespace NUMINAMATH_CALUDE_geometric_series_problem_l1734_173470

/-- Given two infinite geometric series with the specified properties, prove that n = 195 --/
theorem geometric_series_problem (n : ℝ) : 
  let first_series_a1 : ℝ := 15
  let first_series_a2 : ℝ := 5
  let second_series_a1 : ℝ := 15
  let second_series_a2 : ℝ := 5 + n
  let first_series_sum := first_series_a1 / (1 - (first_series_a2 / first_series_a1))
  let second_series_sum := second_series_a1 / (1 - (second_series_a2 / second_series_a1))
  second_series_sum = 5 * first_series_sum →
  n = 195 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l1734_173470


namespace NUMINAMATH_CALUDE_divisor_congruence_l1734_173459

theorem divisor_congruence (d : ℕ) (x y : ℤ) : 
  d > 0 ∧ d ∣ (5 + 1998^1998) →
  (d = 2*x^2 + 2*x*y + 3*y^2 ↔ d % 20 = 3 ∨ d % 20 = 7) :=
by sorry

end NUMINAMATH_CALUDE_divisor_congruence_l1734_173459


namespace NUMINAMATH_CALUDE_kates_bill_l1734_173439

theorem kates_bill (bob_bill : ℝ) (bob_discount : ℝ) (kate_discount : ℝ) (total_after_discount : ℝ) :
  bob_bill = 30 →
  bob_discount = 0.05 →
  kate_discount = 0.02 →
  total_after_discount = 53 →
  ∃ kate_bill : ℝ,
    kate_bill = 25 ∧
    bob_bill * (1 - bob_discount) + kate_bill * (1 - kate_discount) = total_after_discount :=
by sorry

end NUMINAMATH_CALUDE_kates_bill_l1734_173439


namespace NUMINAMATH_CALUDE_cards_bought_equals_difference_l1734_173412

/-- The number of baseball cards Sam bought is equal to the difference between
    Mike's initial number of cards and his current number of cards. -/
theorem cards_bought_equals_difference (initial_cards current_cards cards_bought : ℕ) :
  initial_cards = 87 →
  current_cards = 74 →
  cards_bought = initial_cards - current_cards →
  cards_bought = 13 := by
  sorry

end NUMINAMATH_CALUDE_cards_bought_equals_difference_l1734_173412


namespace NUMINAMATH_CALUDE_norbs_age_l1734_173433

def guesses : List Nat := [25, 29, 31, 33, 37, 39, 42, 45, 48, 50]

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

def count_low_guesses (age : Nat) : Nat :=
  (guesses.filter (· < age)).length

def count_off_by_one (age : Nat) : Nat :=
  (guesses.filter (λ g => g = age - 1 ∨ g = age + 1)).length

theorem norbs_age :
  ∃ (age : Nat),
    age ∈ guesses ∧
    is_prime age ∧
    count_low_guesses age < (2 * guesses.length) / 3 ∧
    count_off_by_one age = 2 ∧
    age = 29 :=
  sorry

end NUMINAMATH_CALUDE_norbs_age_l1734_173433


namespace NUMINAMATH_CALUDE_robin_final_candy_l1734_173493

def initial_candy : ℕ := 23

def eaten_fraction : ℚ := 2/3

def sister_bonus_fraction : ℚ := 1/2

theorem robin_final_candy : 
  ∃ (eaten : ℕ) (leftover : ℕ) (bonus : ℕ),
    eaten = ⌊(eaten_fraction : ℚ) * initial_candy⌋ ∧
    leftover = initial_candy - eaten ∧
    bonus = ⌊(sister_bonus_fraction : ℚ) * initial_candy⌋ ∧
    leftover + bonus = 19 :=
by sorry

end NUMINAMATH_CALUDE_robin_final_candy_l1734_173493


namespace NUMINAMATH_CALUDE_sin_15_30_75_product_l1734_173441

theorem sin_15_30_75_product : Real.sin (15 * π / 180) * Real.sin (30 * π / 180) * Real.sin (75 * π / 180) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_30_75_product_l1734_173441


namespace NUMINAMATH_CALUDE_simple_interest_growth_factor_l1734_173479

/-- The growth factor for simple interest -/
def growth_factor (rate : ℝ) (time : ℝ) : ℝ :=
  1 + rate * time

/-- Theorem: The growth factor for a 5% simple interest rate over 20 years is 2 -/
theorem simple_interest_growth_factor : 
  growth_factor (5 / 100) 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_growth_factor_l1734_173479


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l1734_173411

theorem solution_set_abs_inequality :
  {x : ℝ | |2*x - 1| < 1} = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l1734_173411


namespace NUMINAMATH_CALUDE_power_of_power_l1734_173466

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1734_173466


namespace NUMINAMATH_CALUDE_four_equidistant_lines_l1734_173471

-- Define the distance between two points
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Define a line in 2D space
def Line : Type := sorry

-- Define the distance from a point to a line
def point_to_line_distance (p : ℝ × ℝ) (l : Line) : ℝ := sorry

theorem four_equidistant_lines 
  (A B : ℝ × ℝ) 
  (h_distance : distance A B = 8) :
  ∃ (lines : Finset Line), 
    lines.card = 4 ∧ 
    (∀ l ∈ lines, point_to_line_distance A l = 3 ∧ point_to_line_distance B l = 4) ∧
    (∀ l : Line, point_to_line_distance A l = 3 ∧ point_to_line_distance B l = 4 → l ∈ lines) :=
sorry

end NUMINAMATH_CALUDE_four_equidistant_lines_l1734_173471


namespace NUMINAMATH_CALUDE_no_solution_exists_l1734_173492

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the iterated sum of digits function
def iteratedSumOfDigits (n : ℕ) (k : ℕ) : ℕ := sorry

-- Define the sum of iterated sum of digits
def sumOfIteratedSumOfDigits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_solution_exists :
  ∀ n : ℕ, sumOfIteratedSumOfDigits n ≠ 2000000 := by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1734_173492


namespace NUMINAMATH_CALUDE_exists_non_intersecting_circle_l1734_173445

-- Define the circular billiard table
def CircularBilliardTable := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 1}

-- Define a trajectory of the ball
def Trajectory := Set (ℝ × ℝ)

-- Define the property of a trajectory following the laws of reflection
def FollowsReflectionLaws (t : Trajectory) : Prop := sorry

-- Define a circle inside the table
def InsideCircle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2 ∧ p ∈ CircularBilliardTable}

-- The main theorem
theorem exists_non_intersecting_circle :
  ∀ (start : ℝ × ℝ) (t : Trajectory),
    start ∈ CircularBilliardTable →
    FollowsReflectionLaws t →
    ∃ (center : ℝ × ℝ) (radius : ℝ),
      InsideCircle center radius ⊆ CircularBilliardTable ∧
      (InsideCircle center radius ∩ t = ∅) :=
by
  sorry

end NUMINAMATH_CALUDE_exists_non_intersecting_circle_l1734_173445


namespace NUMINAMATH_CALUDE_fraction_equality_l1734_173452

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 5) : 
  (4 * m * r - 5 * n * t) / (3 * n * t - 2 * m * r) = -9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1734_173452


namespace NUMINAMATH_CALUDE_squirrel_acorns_l1734_173490

/-- Given 5 squirrels collecting 575 acorns in total, and each squirrel needing 130 acorns for winter,
    the number of additional acorns each squirrel needs to collect is 15. -/
theorem squirrel_acorns (num_squirrels : ℕ) (total_acorns : ℕ) (acorns_needed : ℕ) :
  num_squirrels = 5 →
  total_acorns = 575 →
  acorns_needed = 130 →
  acorns_needed - (total_acorns / num_squirrels) = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_squirrel_acorns_l1734_173490


namespace NUMINAMATH_CALUDE_reciprocal_difference_product_relation_l1734_173423

theorem reciprocal_difference_product_relation :
  ∃ (a b : ℕ), a > b ∧ (1 : ℚ) / (a - b) = 3 * (1 : ℚ) / (a * b) :=
by
  use 6, 2
  sorry

end NUMINAMATH_CALUDE_reciprocal_difference_product_relation_l1734_173423


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l1734_173436

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a with a₂ = 2 and a₃ = 4,
    prove that the 10th term a₁₀ = 18. -/
theorem arithmetic_sequence_10th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_2 : a 2 = 2)
  (h_3 : a 3 = 4) :
  a 10 = 18 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l1734_173436


namespace NUMINAMATH_CALUDE_other_people_in_house_l1734_173409

-- Define the given conditions
def cups_per_person_per_day : ℕ := 2
def ounces_per_cup : ℚ := 1/2
def price_per_ounce : ℚ := 5/4
def weekly_spend : ℚ := 35

-- Define the theorem
theorem other_people_in_house :
  let total_ounces : ℚ := weekly_spend / price_per_ounce
  let ounces_per_person_per_week : ℚ := 7 * cups_per_person_per_day * ounces_per_cup
  let total_people : ℕ := Nat.floor (total_ounces / ounces_per_person_per_week)
  total_people - 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_other_people_in_house_l1734_173409


namespace NUMINAMATH_CALUDE_slices_per_banana_l1734_173407

/-- Given information about yogurt preparation and banana usage, 
    calculate the number of slices per banana. -/
theorem slices_per_banana 
  (slices_per_yogurt : ℕ) 
  (yogurts_to_make : ℕ) 
  (bananas_needed : ℕ) 
  (h1 : slices_per_yogurt = 8) 
  (h2 : yogurts_to_make = 5) 
  (h3 : bananas_needed = 4) : 
  (slices_per_yogurt * yogurts_to_make) / bananas_needed = 10 := by
sorry

end NUMINAMATH_CALUDE_slices_per_banana_l1734_173407


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l1734_173400

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : a + b ≤ c * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l1734_173400


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1734_173461

theorem quadratic_transformation (x : ℝ) :
  ∃ (d e : ℝ), (∀ x, x^2 - 24*x + 50 = (x + d)^2 + e) ∧ d + e = -106 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1734_173461


namespace NUMINAMATH_CALUDE_train_speed_l1734_173431

/-- Proves that a train with given length and time to cross a pole has a specific speed in km/h -/
theorem train_speed (length : Real) (time : Real) (speed_kmh : Real) : 
  length = 140 ∧ time = 7 → speed_kmh = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1734_173431


namespace NUMINAMATH_CALUDE_simplify_expression_l1734_173472

theorem simplify_expression : (256 : ℝ) ^ (1/4) * (343 : ℝ) ^ (1/3) = 28 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1734_173472


namespace NUMINAMATH_CALUDE_net_amount_theorem_l1734_173494

def net_amount_spent (shorts_cost shirt_cost jacket_return : ℚ) : ℚ :=
  shorts_cost + shirt_cost - jacket_return

theorem net_amount_theorem (shorts_cost shirt_cost jacket_return : ℚ) :
  net_amount_spent shorts_cost shirt_cost jacket_return =
  shorts_cost + shirt_cost - jacket_return := by
  sorry

#eval net_amount_spent (13.99 : ℚ) (12.14 : ℚ) (7.43 : ℚ)

end NUMINAMATH_CALUDE_net_amount_theorem_l1734_173494


namespace NUMINAMATH_CALUDE_problem_solution_l1734_173404

theorem problem_solution : 
  ((5 * Real.sqrt 3 + 2 * Real.sqrt 5) ^ 2 = 95 + 20 * Real.sqrt 15) ∧ 
  ((1/2) * (Real.sqrt 2 + Real.sqrt 3) - (3/4) * (Real.sqrt 2 + Real.sqrt 27) = 
   -(1/4) * Real.sqrt 2 - (7/4) * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1734_173404


namespace NUMINAMATH_CALUDE_l_shape_tiling_l1734_173473

/-- Number of ways to tile an L-shaped region with dominos -/
def tiling_count (m n : ℕ) : ℕ :=
  sorry

/-- The L-shaped region is formed by attaching two 2 by 5 rectangles to adjacent sides of a 2 by 2 square -/
theorem l_shape_tiling :
  tiling_count 5 5 = 208 :=
sorry

end NUMINAMATH_CALUDE_l_shape_tiling_l1734_173473


namespace NUMINAMATH_CALUDE_part_one_part_two_l1734_173449

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := |x - c|

-- Part I: Prove that f(x) + f(-1/x) ≥ 2 for any real x and c
theorem part_one (c : ℝ) (x : ℝ) : f c x + f c (-1/x) ≥ 2 :=
sorry

-- Part II: Prove that for c = 4, the solution set of |f(1/2x+c) - 1/2f(x)| ≤ 1 is {x | 1 ≤ x ≤ 3}
theorem part_two :
  let c : ℝ := 4
  ∀ x : ℝ, |f c (1/2 * x + c) - 1/2 * f c x| ≤ 1 ↔ 1 ≤ x ∧ x ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1734_173449


namespace NUMINAMATH_CALUDE_bottle_count_theorem_l1734_173429

/-- Represents the number of bottles for each team and the total filled -/
structure BottleCount where
  total : Nat
  football : Nat
  soccer : Nat
  lacrosse : Nat
  rugby : Nat
  unaccounted : Nat

/-- The given conditions and the statement to prove -/
theorem bottle_count_theorem (bc : BottleCount) : 
  bc.total = 254 ∧ 
  bc.football = 11 * 6 ∧ 
  bc.soccer = 53 ∧ 
  bc.lacrosse = bc.football + 12 ∧ 
  bc.rugby = 49 → 
  bc.total = bc.football + bc.soccer + bc.lacrosse + bc.rugby + bc.unaccounted :=
by sorry

end NUMINAMATH_CALUDE_bottle_count_theorem_l1734_173429


namespace NUMINAMATH_CALUDE_short_story_section_pages_l1734_173460

/-- The number of pages in each booklet -/
def pages_per_booklet : ℕ := 9

/-- The number of booklets in the short story section -/
def number_of_booklets : ℕ := 49

/-- The total number of pages in the short story section -/
def total_pages : ℕ := pages_per_booklet * number_of_booklets

theorem short_story_section_pages :
  total_pages = 441 :=
by sorry

end NUMINAMATH_CALUDE_short_story_section_pages_l1734_173460


namespace NUMINAMATH_CALUDE_reduce_tiles_to_less_than_five_l1734_173402

/-- Represents the operation of removing prime-numbered tiles and renumbering --/
def remove_primes_and_renumber (n : ℕ) : ℕ := sorry

/-- Counts the number of operations needed to reduce the set to fewer than 5 tiles --/
def count_operations (initial_count : ℕ) : ℕ := sorry

/-- Theorem stating that 5 operations are needed to reduce 50 tiles to fewer than 5 --/
theorem reduce_tiles_to_less_than_five :
  count_operations 50 = 5 ∧ remove_primes_and_renumber (remove_primes_and_renumber (remove_primes_and_renumber (remove_primes_and_renumber (remove_primes_and_renumber 50)))) < 5 := by
  sorry

end NUMINAMATH_CALUDE_reduce_tiles_to_less_than_five_l1734_173402


namespace NUMINAMATH_CALUDE_sandwiches_available_l1734_173444

def initial_sandwiches : ℕ := 23
def sold_out_sandwiches : ℕ := 14

theorem sandwiches_available : initial_sandwiches - sold_out_sandwiches = 9 := by
  sorry

end NUMINAMATH_CALUDE_sandwiches_available_l1734_173444


namespace NUMINAMATH_CALUDE_car_speed_calculation_l1734_173418

theorem car_speed_calculation (D : ℝ) (h_D_pos : D > 0) : ∃ v : ℝ,
  (D / ((0.8 * D / 80) + (0.2 * D / v)) = 50) → v = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_calculation_l1734_173418


namespace NUMINAMATH_CALUDE_line_bisects_circle_implies_b_eq_neg_two_l1734_173425

/-- The line l is defined by parametric equations x = 2t and y = 1 + bt -/
def line_l (b t : ℝ) : ℝ × ℝ := (2 * t, 1 + b * t)

/-- The circle C is defined by the equation (x - 1)^2 + y^2 = 1 -/
def circle_C (p : ℝ × ℝ) : Prop :=
  (p.1 - 1)^2 + p.2^2 = 1

/-- A line bisects the area of a circle if it passes through the center of the circle -/
def bisects_circle_area (l : ℝ → ℝ × ℝ) (c : ℝ × ℝ → Prop) : Prop :=
  ∃ t, l t = (1, 0)

/-- Main theorem: If line l bisects the area of circle C, then b = -2 -/
theorem line_bisects_circle_implies_b_eq_neg_two (b : ℝ) :
  bisects_circle_area (line_l b) circle_C → b = -2 :=
sorry

end NUMINAMATH_CALUDE_line_bisects_circle_implies_b_eq_neg_two_l1734_173425


namespace NUMINAMATH_CALUDE_ac_cube_l1734_173421

theorem ac_cube (a b c : ℝ) (h1 : a * b = 1) (h2 : b + c = 0) : (a * c)^3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ac_cube_l1734_173421


namespace NUMINAMATH_CALUDE_cans_for_final_rooms_l1734_173456

-- Define the initial and final number of rooms that can be painted
def initial_rooms : ℕ := 50
def final_rooms : ℕ := 42

-- Define the number of cans lost
def cans_lost : ℕ := 4

-- Define the function to calculate the number of cans needed for a given number of rooms
def cans_needed (rooms : ℕ) : ℕ :=
  rooms / ((initial_rooms - final_rooms) / cans_lost)

-- Theorem statement
theorem cans_for_final_rooms :
  cans_needed final_rooms = 21 :=
sorry

end NUMINAMATH_CALUDE_cans_for_final_rooms_l1734_173456


namespace NUMINAMATH_CALUDE_problem_solution_l1734_173422

def problem (m : ℝ) : Prop :=
  let a : Fin 2 → ℝ := ![m + 2, 1]
  let b : Fin 2 → ℝ := ![1, -2*m]
  (a 0 * b 0 + a 1 * b 1 = 0) →  -- a ⊥ b condition
  ‖(a 0 + b 0, a 1 + b 1)‖ = Real.sqrt 34

theorem problem_solution :
  ∃ m : ℝ, problem m := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1734_173422


namespace NUMINAMATH_CALUDE_unique_digit_arrangement_l1734_173477

-- Define the type for positions
inductive Position : Type
  | A | B | C | D | E | F

-- Define the function type for digit assignments
def DigitAssignment := Position → Fin 6

-- Define the condition that all digits are used exactly once
def allDigitsUsedOnce (assignment : DigitAssignment) : Prop :=
  ∀ d : Fin 6, ∃! p : Position, assignment p = d

-- Define the sum conditions
def sumConditions (assignment : DigitAssignment) : Prop :=
  (assignment Position.A).val + (assignment Position.D).val + (assignment Position.E).val = 15 ∧
  7 + (assignment Position.C).val + (assignment Position.E).val = 15 ∧
  9 + (assignment Position.C).val + (assignment Position.A).val = 15 ∧
  (assignment Position.A).val + 8 + (assignment Position.F).val = 15 ∧
  7 + (assignment Position.D).val + (assignment Position.F).val = 15 ∧
  9 + (assignment Position.D).val + (assignment Position.B).val = 15 ∧
  (assignment Position.B).val + (assignment Position.C).val + (assignment Position.F).val = 15

-- Define the correct assignment
def correctAssignment : DigitAssignment :=
  λ p => match p with
  | Position.A => 3  -- 4 - 1 (Fin 6 is 0-based)
  | Position.B => 0  -- 1 - 1
  | Position.C => 1  -- 2 - 1
  | Position.D => 4  -- 5 - 1
  | Position.E => 5  -- 6 - 1
  | Position.F => 2  -- 3 - 1

-- Theorem statement
theorem unique_digit_arrangement :
  ∀ assignment : DigitAssignment,
    allDigitsUsedOnce assignment ∧ sumConditions assignment →
    assignment = correctAssignment :=
sorry

end NUMINAMATH_CALUDE_unique_digit_arrangement_l1734_173477


namespace NUMINAMATH_CALUDE_greatest_y_value_l1734_173495

theorem greatest_y_value (y : ℝ) : 3 * y^2 + 5 * y + 3 = 3 → y ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_greatest_y_value_l1734_173495


namespace NUMINAMATH_CALUDE_inequality_solution_l1734_173499

theorem inequality_solution : ∃! x : ℝ, 
  (Real.sqrt (x^3 + x - 90) + 7) * |x^3 - 10*x^2 + 31*x - 28| ≤ 0 ∧
  x = 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1734_173499


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l1734_173416

theorem ratio_of_numbers (x y : ℝ) (h1 : x + y = 14) (h2 : y = 3.5) (h3 : x > y) :
  x / y = 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l1734_173416


namespace NUMINAMATH_CALUDE_nested_expression_sum_l1734_173497

theorem nested_expression_sum : 
  4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4)))))))) = 1398100 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_sum_l1734_173497


namespace NUMINAMATH_CALUDE_wayne_blocks_problem_l1734_173447

theorem wayne_blocks_problem (initial_blocks : ℕ) (father_blocks : ℕ) : 
  initial_blocks = 9 →
  father_blocks = 6 →
  (3 * (initial_blocks + father_blocks)) - (initial_blocks + father_blocks) = 30 :=
by sorry

end NUMINAMATH_CALUDE_wayne_blocks_problem_l1734_173447


namespace NUMINAMATH_CALUDE_supremum_of_function_l1734_173427

theorem supremum_of_function (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ M : ℝ, (∀ x : ℝ, -1/(2*x) - 2/((1-x)) ≤ M) ∧ 
  (∀ ε > 0, ∃ y : ℝ, -1/(2*y) - 2/((1-y)) > M - ε) ∧
  M = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_supremum_of_function_l1734_173427


namespace NUMINAMATH_CALUDE_cubic_equation_root_l1734_173496

theorem cubic_equation_root (a b : ℚ) : 
  ((-2 - 3 * Real.sqrt 3) ^ 3 + a * (-2 - 3 * Real.sqrt 3) ^ 2 + b * (-2 - 3 * Real.sqrt 3) + 49 = 0) → 
  a = -3/23 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l1734_173496


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a3_value_l1734_173450

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a3_value
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a2 : a 2 = 2 * a 3 + 1)
  (h_a4 : a 4 = 2 * a 3 + 7) :
  a 3 = -4 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a3_value_l1734_173450


namespace NUMINAMATH_CALUDE_election_votes_difference_l1734_173484

theorem election_votes_difference (total_votes : ℕ) (winner_votes : ℕ) (second_votes : ℕ) (third_votes : ℕ) (fourth_votes : ℕ) 
  (h_total : total_votes = 979)
  (h_candidates : winner_votes + second_votes + third_votes + fourth_votes = total_votes)
  (h_winner_second : winner_votes = second_votes + 53)
  (h_winner_fourth : winner_votes = fourth_votes + 105)
  (h_fourth : fourth_votes = 199) :
  winner_votes - third_votes = 79 := by
sorry

end NUMINAMATH_CALUDE_election_votes_difference_l1734_173484


namespace NUMINAMATH_CALUDE_system_solution_exists_no_solution_when_m_eq_one_l1734_173481

theorem system_solution_exists (m : ℝ) (h : m ≠ 1) :
  ∃ (x y : ℝ), y = m * x + 4 ∧ y = (3 * m - 2) * x + 5 := by
  sorry

theorem no_solution_when_m_eq_one :
  ¬ ∃ (x y : ℝ), y = 1 * x + 4 ∧ y = (3 * 1 - 2) * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_no_solution_when_m_eq_one_l1734_173481


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1734_173464

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 1}

theorem intersection_of_P_and_Q : P ∩ Q = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1734_173464


namespace NUMINAMATH_CALUDE_max_quotient_is_21996_l1734_173451

def is_valid_divisor (d : ℕ) : Prop :=
  d ≥ 10 ∧ d < 100

def quotient_hundreds_condition (dividend : ℕ) (divisor : ℕ) : Prop :=
  let q := dividend / divisor
  (q / 100) * divisor ≥ 200 ∧ (q / 100) * divisor < 300

def max_quotient_dividend (dividends : List ℕ) : ℕ := sorry

theorem max_quotient_is_21996 :
  let dividends := [21944, 21996, 24054, 24111]
  ∃ d : ℕ, is_valid_divisor d ∧ 
           quotient_hundreds_condition (max_quotient_dividend dividends) d ∧
           max_quotient_dividend dividends = 21996 := by sorry

end NUMINAMATH_CALUDE_max_quotient_is_21996_l1734_173451


namespace NUMINAMATH_CALUDE_pyramid_height_l1734_173420

theorem pyramid_height (base_perimeter : ℝ) (apex_to_vertex : ℝ) (h_perimeter : base_perimeter = 40) (h_apex_dist : apex_to_vertex = 12) :
  let side_length := base_perimeter / 4
  let diagonal := side_length * Real.sqrt 2
  let half_diagonal := diagonal / 2
  Real.sqrt (apex_to_vertex ^ 2 - half_diagonal ^ 2) = Real.sqrt 94 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_height_l1734_173420


namespace NUMINAMATH_CALUDE_bridge_length_l1734_173474

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 225 :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l1734_173474


namespace NUMINAMATH_CALUDE_bus_stop_walk_time_l1734_173446

/-- The time taken to walk to the bus stop at the usual speed, in minutes -/
def usual_time : ℝ := 30

/-- The time taken to walk to the bus stop at 4/5 of the usual speed, in minutes -/
def slower_time : ℝ := usual_time + 6

theorem bus_stop_walk_time : usual_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_walk_time_l1734_173446


namespace NUMINAMATH_CALUDE_point_four_units_from_one_l1734_173428

theorem point_four_units_from_one (x : ℝ) : 
  (x = 1 + 4 ∨ x = 1 - 4) ↔ (x = 5 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_point_four_units_from_one_l1734_173428


namespace NUMINAMATH_CALUDE_train_length_calculation_l1734_173432

/-- The length of each train in meters -/
def train_length : ℝ := 79.92

/-- The speed of the faster train in km/hr -/
def faster_speed : ℝ := 52

/-- The speed of the slower train in km/hr -/
def slower_speed : ℝ := 36

/-- The time it takes for the faster train to pass the slower train in seconds -/
def passing_time : ℝ := 36

theorem train_length_calculation :
  let relative_speed := (faster_speed - slower_speed) * 1000 / 3600
  2 * train_length = relative_speed * passing_time := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1734_173432


namespace NUMINAMATH_CALUDE_car_A_time_is_5_hours_l1734_173476

/-- Represents the properties of a car's journey -/
structure CarJourney where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem setup -/
def problem : Prop :=
  ∃ (carA carB : CarJourney),
    carA.speed = 80 ∧
    carB.speed = 100 ∧
    carB.time = 2 ∧
    carA.distance = 2 * carB.distance ∧
    carA.distance = carA.speed * carA.time ∧
    carB.distance = carB.speed * carB.time

/-- The theorem to prove -/
theorem car_A_time_is_5_hours (h : problem) : 
  ∃ (carA : CarJourney), carA.time = 5 := by
  sorry


end NUMINAMATH_CALUDE_car_A_time_is_5_hours_l1734_173476


namespace NUMINAMATH_CALUDE_triangle_power_equality_l1734_173405

theorem triangle_power_equality (a b c : ℝ) 
  (h : ∀ n : ℕ, (a^n + b^n > c^n) ∧ (b^n + c^n > a^n) ∧ (c^n + a^n > b^n)) :
  (a = b) ∨ (b = c) ∨ (c = a) := by
sorry

end NUMINAMATH_CALUDE_triangle_power_equality_l1734_173405


namespace NUMINAMATH_CALUDE_five_dozen_apple_cost_l1734_173467

/-- The cost of apples given the number of dozens and the price -/
def apple_cost (dozens : ℚ) (price : ℚ) : ℚ := dozens * (price / 4)

/-- Theorem: If 4 dozen apples cost $31.20, then 5 dozen apples at the same rate will cost $39.00 -/
theorem five_dozen_apple_cost :
  apple_cost 5 31.20 = 39.00 :=
sorry

end NUMINAMATH_CALUDE_five_dozen_apple_cost_l1734_173467


namespace NUMINAMATH_CALUDE_cuboid_sum_of_edges_l1734_173482

/-- Represents the dimensions of a rectangular cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Checks if the dimensions form a geometric progression -/
def isGeometricProgression (d : CuboidDimensions) : Prop :=
  ∃ q : ℝ, q > 0 ∧ d.length = q * d.width ∧ d.width = q * d.height

/-- Calculates the volume of a rectangular cuboid -/
def volume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the surface area of a rectangular cuboid -/
def surfaceArea (d : CuboidDimensions) : ℝ :=
  2 * (d.length * d.width + d.width * d.height + d.height * d.length)

/-- Calculates the sum of all edges of a rectangular cuboid -/
def sumOfEdges (d : CuboidDimensions) : ℝ :=
  4 * (d.length + d.width + d.height)

/-- Theorem: For a rectangular cuboid with volume 8, surface area 32, and dimensions 
    forming a geometric progression, the sum of all edges is 32 -/
theorem cuboid_sum_of_edges : 
  ∀ d : CuboidDimensions, 
    volume d = 8 → 
    surfaceArea d = 32 → 
    isGeometricProgression d → 
    sumOfEdges d = 32 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_sum_of_edges_l1734_173482


namespace NUMINAMATH_CALUDE_conic_parametric_to_cartesian_l1734_173410

theorem conic_parametric_to_cartesian (t : ℝ) (x y : ℝ) :
  x = t^2 + 1/t^2 - 2 ∧ y = t - 1/t → y^2 = x :=
by sorry

end NUMINAMATH_CALUDE_conic_parametric_to_cartesian_l1734_173410


namespace NUMINAMATH_CALUDE_trigonometric_equation_l1734_173438

theorem trigonometric_equation (x : ℝ) (h : |Real.cos (2 * x)| ≠ 1) :
  8.451 * ((1 - Real.cos (2 * x)) / (1 + Real.cos (2 * x))) = (1/3) * Real.tan x ^ 4 ↔
  ∃ k : ℤ, x = π/3 * (3 * k + 1) ∨ x = π/3 * (3 * k - 1) :=
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_l1734_173438


namespace NUMINAMATH_CALUDE_local_minimum_implies_c_equals_2_l1734_173426

/-- The function f(x) = x(x-c)^2 has a local minimum at x=2 -/
def has_local_minimum_at_2 (c : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - 2| < δ → x * (x - c)^2 ≥ 2 * (2 - c)^2

theorem local_minimum_implies_c_equals_2 :
  ∀ c : ℝ, has_local_minimum_at_2 c → c = 2 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_implies_c_equals_2_l1734_173426


namespace NUMINAMATH_CALUDE_increasing_shift_l1734_173401

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- Theorem statement
theorem increasing_shift (h : IncreasingOn f (-2) 3) :
  IncreasingOn (fun x => f (x + 5)) (-7) (-2) :=
sorry

end NUMINAMATH_CALUDE_increasing_shift_l1734_173401


namespace NUMINAMATH_CALUDE_jasmine_solution_problem_l1734_173442

theorem jasmine_solution_problem (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_jasmine : ℝ) (final_concentration : ℝ) (x : ℝ) : 
  initial_volume = 90 →
  initial_concentration = 0.05 →
  added_jasmine = 8 →
  final_concentration = 0.125 →
  initial_volume * initial_concentration + added_jasmine = 
    (initial_volume + added_jasmine + x) * final_concentration →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_jasmine_solution_problem_l1734_173442


namespace NUMINAMATH_CALUDE_pond_length_l1734_173403

/-- Given a rectangular field and a square pond, prove the length of the pond's side -/
theorem pond_length (field_width field_length pond_area : ℝ) : 
  field_length = 2 * field_width →
  field_length = 36 →
  pond_area = (1/8) * (field_length * field_width) →
  Real.sqrt pond_area = 9 := by
  sorry

end NUMINAMATH_CALUDE_pond_length_l1734_173403


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_37_l1734_173498

theorem modular_inverse_of_5_mod_37 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 36 ∧ (5 * x) % 37 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_37_l1734_173498


namespace NUMINAMATH_CALUDE_congcong_carbon_emissions_l1734_173478

/-- Carbon dioxide emissions calculation for household tap water -/
def carbon_emissions (water_usage : ℝ) : ℝ := water_usage * 0.91

/-- Congcong's water usage in a certain month (in tons) -/
def congcong_water_usage : ℝ := 6

/-- Theorem stating the carbon dioxide emissions from Congcong's tap water for a certain month -/
theorem congcong_carbon_emissions :
  carbon_emissions congcong_water_usage = 5.46 := by
  sorry

end NUMINAMATH_CALUDE_congcong_carbon_emissions_l1734_173478


namespace NUMINAMATH_CALUDE_complex_quadrant_range_l1734_173485

theorem complex_quadrant_range (z : ℂ) (a : ℝ) :
  z * (a + Complex.I) = 2 + 3 * Complex.I →
  (z.re * z.im < 0 ↔ -3/2 < a ∧ a < 2/3) :=
by sorry

end NUMINAMATH_CALUDE_complex_quadrant_range_l1734_173485


namespace NUMINAMATH_CALUDE_positive_number_inequality_l1734_173475

theorem positive_number_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x y, (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ x ≠ y ∧
  0 < (x - y) / (1 + x * y) ∧ (x - y) / (1 + x * y) ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_positive_number_inequality_l1734_173475


namespace NUMINAMATH_CALUDE_binary_1010_to_decimal_l1734_173463

/-- Converts a list of binary digits to its decimal representation. -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1010₂ -/
def binary_1010 : List Bool := [false, true, false, true]

/-- Theorem stating that the decimal representation of 1010₂ is 10 -/
theorem binary_1010_to_decimal :
  binary_to_decimal binary_1010 = 10 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010_to_decimal_l1734_173463


namespace NUMINAMATH_CALUDE_inequality_proof_l1734_173453

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (1 + x + 2*x^2) * (2 + 3*y + y^2) * (4 + z + z^2) ≥ 60*x*y*z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1734_173453


namespace NUMINAMATH_CALUDE_min_distance_tan_intersection_l1734_173417

theorem min_distance_tan_intersection (a : ℝ) : 
  let f (x : ℝ) := Real.tan (2 * x - π / 3)
  let g (x : ℝ) := -a
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
    f x₁ = g x₁ ∧ 
    f x₂ = g x₂ ∧
    ∀ (y : ℝ), x₁ < y ∧ y < x₂ → f y ≠ g y ∧
    x₂ - x₁ = π / 2 ∧
    ∀ (z₁ z₂ : ℝ), (f z₁ = g z₁ ∧ f z₂ = g z₂ ∧ z₁ < z₂) → z₂ - z₁ ≥ π / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_tan_intersection_l1734_173417


namespace NUMINAMATH_CALUDE_total_popsicles_l1734_173480

/-- The number of grape popsicles in the freezer -/
def grape_popsicles : ℕ := 2

/-- The number of cherry popsicles in the freezer -/
def cherry_popsicles : ℕ := 13

/-- The number of banana popsicles in the freezer -/
def banana_popsicles : ℕ := 2

/-- Theorem stating the total number of popsicles in the freezer -/
theorem total_popsicles : grape_popsicles + cherry_popsicles + banana_popsicles = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_popsicles_l1734_173480


namespace NUMINAMATH_CALUDE_trinomial_product_degree_15_l1734_173408

def trinomial (p q : ℕ) (a : ℝ) (x : ℝ) : ℝ := x^p + a * x^q + 1

theorem trinomial_product_degree_15 :
  ∀ (p q r s : ℕ) (a b : ℝ),
    q < p → s < r → p + r = 15 →
    (∃ (t : ℕ) (c : ℝ), 
      trinomial p q a * trinomial r s b = trinomial 15 t c) ↔
    ((p = 5 ∧ q = 0 ∧ r = 10 ∧ s = 5 ∧ a = 1 ∧ b = -1) ∨
     (p = 9 ∧ q = 3 ∧ r = 6 ∧ s = 3 ∧ a = -1 ∧ b = 1) ∨
     (p = 9 ∧ q = 6 ∧ r = 6 ∧ s = 3 ∧ a = -1 ∧ b = 1)) :=
by sorry

end NUMINAMATH_CALUDE_trinomial_product_degree_15_l1734_173408


namespace NUMINAMATH_CALUDE_all_propositions_false_l1734_173458

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (subset_line_plane : Line → Plane → Prop)

-- Define the theorem
theorem all_propositions_false 
  (a b : Line) (α β γ : Plane) : 
  ¬(∀ a b α, (parallel_line_plane a α ∧ parallel_line_plane b α) → parallel_line_line a b) ∧
  ¬(∀ α β γ, (perpendicular_plane α β ∧ perpendicular_plane β γ) → parallel_plane_plane α γ) ∧
  ¬(∀ a α β, (parallel_line_plane a α ∧ parallel_line_plane a β) → parallel_plane_plane α β) ∧
  ¬(∀ a b α, (parallel_line_line a b ∧ subset_line_plane b α) → parallel_line_plane a α) :=
sorry

end NUMINAMATH_CALUDE_all_propositions_false_l1734_173458


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l1734_173462

def f (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem coefficient_of_x_squared (x : ℝ) : 
  ∃ (a b c d : ℝ), (f x)^3 = a*x^6 + b*x^5 + c*x^4 + (-9)*x^2 + d := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l1734_173462
