import Mathlib

namespace NUMINAMATH_CALUDE_total_crayons_l3475_347597

/-- Represents the number of crayons in a box of type 1 -/
def box_type1 : ℕ := 8 + 4 + 5

/-- Represents the number of crayons in a box of type 2 -/
def box_type2 : ℕ := 7 + 6 + 3

/-- Represents the number of crayons in a box of type 3 -/
def box_type3 : ℕ := 11 + 5 + 2

/-- Represents the number of crayons in the unique box -/
def unique_box : ℕ := 9 + 2 + 7

/-- Represents the total number of boxes -/
def total_boxes : ℕ := 3 + 4 + 2 + 1

theorem total_crayons : 
  3 * box_type1 + 4 * box_type2 + 2 * box_type3 + unique_box = 169 :=
sorry

end NUMINAMATH_CALUDE_total_crayons_l3475_347597


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l3475_347507

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_proof :
  rectangle_area 2500 10 = 200 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l3475_347507


namespace NUMINAMATH_CALUDE_union_set_problem_l3475_347500

theorem union_set_problem (A B : Set ℕ) (m : ℕ) :
  A = {1, 2, m} →
  B = {2, 4} →
  A ∪ B = {1, 2, 3, 4} →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_union_set_problem_l3475_347500


namespace NUMINAMATH_CALUDE_distance_point_to_line_bounded_l3475_347573

/-- The distance from a point to a line in 2D space is bounded. -/
theorem distance_point_to_line_bounded (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let P : ℝ × ℝ := (-2, 2)
  let l := {(x, y) : ℝ × ℝ | a * (x - 1) + b * (y + 2) = 0}
  let d := Real.sqrt ((a * (-2 - 1) + b * (2 + 2))^2 / (a^2 + b^2))
  0 ≤ d ∧ d ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_distance_point_to_line_bounded_l3475_347573


namespace NUMINAMATH_CALUDE_find_a_l3475_347556

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := (a * x) / (x - 1) < 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | x < 1 ∨ x > 2}

-- Theorem statement
theorem find_a : ∃ a : ℝ, (∀ x : ℝ, inequality a x ↔ x ∈ solution_set a) ∧ a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l3475_347556


namespace NUMINAMATH_CALUDE_segment_ratio_l3475_347591

/-- Represents a point on a line segment --/
structure Point :=
  (x : ℝ)

/-- Represents a line segment between two points --/
structure Segment (A B : Point) :=
  (length : ℝ)

/-- The main theorem --/
theorem segment_ratio 
  (A B C D : Point)
  (h1 : Segment A D)
  (h2 : Segment A B)
  (h3 : Segment B D)
  (h4 : Segment A C)
  (h5 : Segment C D)
  (h6 : Segment B C)
  (cond1 : B.x < D.x ∧ D.x < C.x)
  (cond2 : h2.length = 3 * h3.length)
  (cond3 : h4.length = 4 * h5.length)
  (cond4 : h1.length = h2.length + h3.length + h5.length) :
  h6.length / h1.length = 5 / 6 :=
sorry

end NUMINAMATH_CALUDE_segment_ratio_l3475_347591


namespace NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l3475_347598

/-- 
For an infinite geometric series with first term a and sum S,
prove that if a = 500 and S = 3500, then the common ratio r is 6/7.
-/
theorem infinite_geometric_series_ratio 
  (a S : ℝ) 
  (h_a : a = 500) 
  (h_S : S = 3500) 
  (h_sum : S = a / (1 - r)) 
  (h_conv : |r| < 1) : 
  r = 6/7 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l3475_347598


namespace NUMINAMATH_CALUDE_base12_remainder_theorem_l3475_347543

/-- Converts a base-12 number to decimal --/
def base12ToDecimal (a b c d : ℕ) : ℕ :=
  a * 12^3 + b * 12^2 + c * 12^1 + d * 12^0

/-- The base-12 number 2563₁₂ --/
def base12Number : ℕ := base12ToDecimal 2 5 6 3

/-- The theorem stating that the remainder of 2563₁₂ divided by 17 is 1 --/
theorem base12_remainder_theorem : base12Number % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_base12_remainder_theorem_l3475_347543


namespace NUMINAMATH_CALUDE_circular_ring_area_l3475_347571

/-- The area of a circular ring enclosed between two concentric circles -/
theorem circular_ring_area (C₁ C₂ : ℝ) (h : C₁ > C₂) :
  let S := (C₁^2 - C₂^2) / (4 * Real.pi)
  ∃ (R₁ R₂ : ℝ), R₁ > R₂ ∧ 
    C₁ = 2 * Real.pi * R₁ ∧ 
    C₂ = 2 * Real.pi * R₂ ∧
    S = Real.pi * R₁^2 - Real.pi * R₂^2 :=
by sorry

end NUMINAMATH_CALUDE_circular_ring_area_l3475_347571


namespace NUMINAMATH_CALUDE_newspapers_sold_l3475_347502

theorem newspapers_sold (total : ℝ) (magazines : ℕ) 
  (h1 : total = 425.0) (h2 : magazines = 150) : 
  total - magazines = 275 := by
  sorry

end NUMINAMATH_CALUDE_newspapers_sold_l3475_347502


namespace NUMINAMATH_CALUDE_target_scientific_notation_l3475_347521

/-- Represents one billion in decimal notation -/
def billion : ℕ := 100000000

/-- The number we want to express in scientific notation -/
def target : ℕ := 1360000000

/-- Scientific notation for the target number -/
def scientific_notation (n : ℕ) : ℚ := 1.36 * (10 : ℚ) ^ n

theorem target_scientific_notation :
  ∃ n : ℕ, scientific_notation n = target ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_target_scientific_notation_l3475_347521


namespace NUMINAMATH_CALUDE_min_value_theorem_l3475_347509

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  ∃ (min_val : ℝ), min_val = 8 + 4 * Real.sqrt 3 ∧
  ∀ z, z = (x + 1) * (y + 1) / (x * y) → z ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3475_347509


namespace NUMINAMATH_CALUDE_pirate_game_l3475_347583

/-- Represents the number of coins each pirate has -/
structure PirateCoins where
  first : ℕ
  second : ℕ

/-- Simulates one round of the game where the first pirate loses half their coins -/
def firstLosesHalf (coins : PirateCoins) : PirateCoins :=
  { first := coins.first / 2,
    second := coins.second + coins.first / 2 }

/-- Simulates one round of the game where the second pirate loses half their coins -/
def secondLosesHalf (coins : PirateCoins) : PirateCoins :=
  { first := coins.first + coins.second / 2,
    second := coins.second / 2 }

/-- The main theorem to prove -/
theorem pirate_game (initial : ℕ) :
  (firstLosesHalf (secondLosesHalf (firstLosesHalf { first := initial, second := 0 })))
  = { first := 15, second := 33 } →
  initial = 24 := by
  sorry

end NUMINAMATH_CALUDE_pirate_game_l3475_347583


namespace NUMINAMATH_CALUDE_optimal_solution_satisfies_criteria_l3475_347510

/-- Represents the optimal solution for the medicine problem -/
def optimal_solution : ℕ × ℕ := (6, 3)

/-- Vitamin contents of the first medicine -/
def medicine1_contents : Fin 4 → ℕ
| 0 => 3  -- Vitamin A
| 1 => 1  -- Vitamin B
| 2 => 1  -- Vitamin C
| 3 => 0  -- Vitamin D

/-- Vitamin contents of the second medicine -/
def medicine2_contents : Fin 4 → ℕ
| 0 => 0  -- Vitamin A
| 1 => 1  -- Vitamin B
| 2 => 3  -- Vitamin C
| 3 => 2  -- Vitamin D

/-- Daily vitamin requirements -/
def daily_requirements : Fin 4 → ℕ
| 0 => 3  -- Vitamin A
| 1 => 9  -- Vitamin B
| 2 => 15 -- Vitamin C
| 3 => 2  -- Vitamin D

/-- Cost of medicines in fillér -/
def medicine_costs : Fin 2 → ℕ
| 0 => 20  -- Cost of medicine 1
| 1 => 60  -- Cost of medicine 2

/-- Theorem stating that the optimal solution satisfies all criteria -/
theorem optimal_solution_satisfies_criteria :
  let (x, y) := optimal_solution
  (x + y = 9) ∧ 
  (medicine_costs 0 * x + medicine_costs 1 * y = 300) ∧
  (x + 2 * y = 12) ∧
  (∀ i : Fin 4, medicine1_contents i * x + medicine2_contents i * y ≥ daily_requirements i) :=
by sorry

end NUMINAMATH_CALUDE_optimal_solution_satisfies_criteria_l3475_347510


namespace NUMINAMATH_CALUDE_problem_solution_l3475_347525

theorem problem_solution : ∃ (S L x : ℕ), 
  S = 18 ∧ 
  S + L = 51 ∧ 
  L = 2 * S - x ∧ 
  x > 0 ∧ 
  x = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3475_347525


namespace NUMINAMATH_CALUDE_douglas_vote_percentage_l3475_347563

theorem douglas_vote_percentage (total_percentage : ℝ) (x_percentage : ℝ) (x_ratio : ℝ) (y_ratio : ℝ) :
  total_percentage = 0.54 →
  x_percentage = 0.62 →
  x_ratio = 3 →
  y_ratio = 2 →
  let total_ratio := x_ratio + y_ratio
  let y_percentage := (total_percentage * total_ratio - x_percentage * x_ratio) / y_ratio
  y_percentage = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_douglas_vote_percentage_l3475_347563


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3475_347535

theorem simplify_fraction_product : 
  (256 : ℚ) / 20 * (10 : ℚ) / 160 * ((16 : ℚ) / 6)^2 = (256 : ℚ) / 45 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3475_347535


namespace NUMINAMATH_CALUDE_smallest_ten_digit_divisible_by_first_five_primes_l3475_347546

/-- The product of the first five prime numbers -/
def first_five_primes_product : ℕ := 2 * 3 * 5 * 7 * 11

/-- A number is a 10-digit number if it's between 1000000000 and 9999999999 -/
def is_ten_digit (n : ℕ) : Prop := 1000000000 ≤ n ∧ n ≤ 9999999999

theorem smallest_ten_digit_divisible_by_first_five_primes :
  ∀ n : ℕ, is_ten_digit n ∧ n % first_five_primes_product = 0 → n ≥ 1000000310 :=
by sorry

end NUMINAMATH_CALUDE_smallest_ten_digit_divisible_by_first_five_primes_l3475_347546


namespace NUMINAMATH_CALUDE_winnie_balloon_distribution_l3475_347555

/-- The number of balloons Winnie keeps for herself when distributing balloons to friends -/
def balloons_kept (total : ℕ) (friends : ℕ) : ℕ :=
  total % friends

theorem winnie_balloon_distribution :
  let total_balloons : ℕ := 20 + 40 + 70 + 90
  let num_friends : ℕ := 9
  balloons_kept total_balloons num_friends = 4 := by
  sorry

end NUMINAMATH_CALUDE_winnie_balloon_distribution_l3475_347555


namespace NUMINAMATH_CALUDE_exists_number_with_properties_l3475_347577

/-- A function that counts the occurrences of a digit in a natural number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ := sorry

/-- A function that checks if a natural number contains only 7s and 5s -/
def containsOnly7sAnd5s (n : ℕ) : Prop := sorry

/-- Theorem stating the existence of a number with the required properties -/
theorem exists_number_with_properties : ∃ n : ℕ, 
  containsOnly7sAnd5s n ∧ 
  countDigit n 7 = countDigit n 5 ∧ 
  n % 7 = 0 ∧ 
  n % 5 = 0 := by sorry

end NUMINAMATH_CALUDE_exists_number_with_properties_l3475_347577


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3475_347538

theorem absolute_value_inequality (a b : ℝ) (ha : a ≠ 0) :
  ∃ (m : ℝ), m = 2 ∧ (∀ (a b : ℝ), a ≠ 0 → |a + b| + |a - b| ≥ m * |a|) ∧
  ∀ (m' : ℝ), (∀ (a b : ℝ), a ≠ 0 → |a + b| + |a - b| ≥ m' * |a|) → m' ≤ m :=
sorry

theorem solution_set (m : ℝ) (hm : m = 2) :
  {x : ℝ | |x - 1| + |x - 2| ≤ m} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 5/2} :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3475_347538


namespace NUMINAMATH_CALUDE_power_plus_one_prime_l3475_347586

theorem power_plus_one_prime (a n : ℕ) (ha : a > 1) (hprime : Nat.Prime (a^n + 1)) :
  Even a ∧ ∃ k : ℕ, n = 2^k :=
sorry

end NUMINAMATH_CALUDE_power_plus_one_prime_l3475_347586


namespace NUMINAMATH_CALUDE_aluminum_foil_thickness_thickness_satisfies_density_l3475_347511

/-- The thickness of a rectangular piece of aluminum foil -/
noncomputable def thickness (d m l w : ℝ) : ℝ := m / (d * l * w)

/-- The volume of a rectangular piece of aluminum foil -/
noncomputable def volume (l w t : ℝ) : ℝ := l * w * t

/-- Theorem: The thickness of a rectangular piece of aluminum foil
    is equal to its mass divided by the product of its density, length, and width -/
theorem aluminum_foil_thickness (d m l w : ℝ) (hd : d > 0) (hl : l > 0) (hw : w > 0) :
  thickness d m l w = m / (d * l * w) :=
by sorry

/-- Theorem: The thickness formula satisfies the density definition -/
theorem thickness_satisfies_density (d m l w : ℝ) (hd : d > 0) (hl : l > 0) (hw : w > 0) :
  d = m / volume l w (thickness d m l w) :=
by sorry

end NUMINAMATH_CALUDE_aluminum_foil_thickness_thickness_satisfies_density_l3475_347511


namespace NUMINAMATH_CALUDE_marias_water_bottles_l3475_347561

/-- Calculates the final number of water bottles Maria has -/
def final_bottle_count (initial : ℕ) (drunk : ℕ) (bought : ℕ) : ℕ :=
  initial - drunk + bought

/-- Proves that Maria's final bottle count is correct -/
theorem marias_water_bottles (initial : ℕ) (drunk : ℕ) (bought : ℕ) 
  (h1 : initial ≥ drunk) : 
  final_bottle_count initial drunk bought = initial - drunk + bought :=
by
  sorry

#eval final_bottle_count 14 8 45

end NUMINAMATH_CALUDE_marias_water_bottles_l3475_347561


namespace NUMINAMATH_CALUDE_school_supplies_cost_l3475_347584

/-- The total cost of school supplies given the number of cartons, boxes per carton, and cost per unit -/
def total_cost (pencil_cartons : ℕ) (pencil_boxes_per_carton : ℕ) (pencil_cost_per_box : ℕ)
                (marker_cartons : ℕ) (marker_boxes_per_carton : ℕ) (marker_cost_per_carton : ℕ) : ℕ :=
  (pencil_cartons * pencil_boxes_per_carton * pencil_cost_per_box) +
  (marker_cartons * marker_cost_per_carton)

/-- Theorem stating that the total cost for the school's purchase is $440 -/
theorem school_supplies_cost :
  total_cost 20 10 2 10 5 4 = 440 := by
  sorry

end NUMINAMATH_CALUDE_school_supplies_cost_l3475_347584


namespace NUMINAMATH_CALUDE_incircle_radius_of_specific_triangle_l3475_347564

theorem incircle_radius_of_specific_triangle : 
  ∀ (a b c h : ℝ) (r : ℝ),
  a = 5 ∧ b = 12 ∧ c = 13 ∧ h = 10 →
  (a^2 + b^2 = c^2) →  -- Pythagorean theorem to ensure right-angled triangle
  r = (b * h / 2) / ((a + b + c) / 2) →
  r = 4 := by sorry

end NUMINAMATH_CALUDE_incircle_radius_of_specific_triangle_l3475_347564


namespace NUMINAMATH_CALUDE_existence_of_four_numbers_l3475_347528

theorem existence_of_four_numbers : ∃ (a b c d : ℕ+), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (3 ∣ a) ∧ (3 ∣ b) ∧ (3 ∣ c) ∧ (3 ∣ d) ∧
  (d ∣ (a + b + c)) ∧ (c ∣ (a + b + d)) ∧ 
  (b ∣ (a + c + d)) ∧ (a ∣ (b + c + d)) :=
sorry

end NUMINAMATH_CALUDE_existence_of_four_numbers_l3475_347528


namespace NUMINAMATH_CALUDE_rectangle_parallel_to_diagonals_l3475_347516

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define a point on the side of a square
structure PointOnSquareSide where
  square : Square
  x : ℝ
  y : ℝ
  on_side : (x = 0 ∧ 0 ≤ y ∧ y ≤ square.side) ∨
            (y = 0 ∧ 0 ≤ x ∧ x ≤ square.side) ∨
            (x = square.side ∧ 0 ≤ y ∧ y ≤ square.side) ∨
            (y = square.side ∧ 0 ≤ x ∧ x ≤ square.side)

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ
  width_positive : width > 0
  height_positive : height > 0

-- Theorem statement
theorem rectangle_parallel_to_diagonals
  (s : Square) (p : PointOnSquareSide) (h : p.square = s) :
  ∃ (r : Rectangle), 
    -- One vertex of the rectangle is at point p
    (r.width = p.x ∧ r.height = p.y) ∨
    (r.width = s.side - p.x ∧ r.height = p.y) ∨
    (r.width = p.x ∧ r.height = s.side - p.y) ∨
    (r.width = s.side - p.x ∧ r.height = s.side - p.y) ∧
    -- Sides of the rectangle are parallel to the diagonals of the square
    (r.width / r.height = 1 ∨ r.width / r.height = -1) :=
sorry

end NUMINAMATH_CALUDE_rectangle_parallel_to_diagonals_l3475_347516


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l3475_347547

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1)
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l3475_347547


namespace NUMINAMATH_CALUDE_barbed_wire_rate_problem_solution_l3475_347548

/-- Calculates the rate of drawing barbed wire per meter given the conditions of the problem. -/
theorem barbed_wire_rate (field_area : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ) : ℝ :=
  let side_length := Real.sqrt field_area
  let perimeter := 4 * side_length
  let wire_length := perimeter - (↑num_gates * gate_width)
  let rate_per_meter := total_cost / wire_length
  by
    -- Proof goes here
    sorry

/-- The rate of drawing barbed wire per meter for the given problem is 4.5 Rs/m. -/
theorem problem_solution : 
  barbed_wire_rate 3136 1 2 999 = 4.5 := by sorry

end NUMINAMATH_CALUDE_barbed_wire_rate_problem_solution_l3475_347548


namespace NUMINAMATH_CALUDE_jerry_shelf_difference_l3475_347518

/-- Calculates the difference between action figures and books on Jerry's shelf -/
def action_figure_book_difference (
  initial_books : ℕ
  ) (initial_action_figures : ℕ)
  (added_action_figures : ℕ) : ℕ :=
  (initial_action_figures + added_action_figures) - initial_books

/-- Proves that the difference between action figures and books is 3 -/
theorem jerry_shelf_difference :
  action_figure_book_difference 3 4 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jerry_shelf_difference_l3475_347518


namespace NUMINAMATH_CALUDE_shopping_discount_l3475_347508

theorem shopping_discount (shoe_price : ℝ) (dress_price : ℝ) 
  (shoe_discount : ℝ) (dress_discount : ℝ) (num_shoes : ℕ) :
  shoe_price = 50 →
  dress_price = 100 →
  shoe_discount = 0.4 →
  dress_discount = 0.2 →
  num_shoes = 2 →
  (num_shoes : ℝ) * shoe_price * (1 - shoe_discount) + 
    dress_price * (1 - dress_discount) = 140 := by
  sorry

end NUMINAMATH_CALUDE_shopping_discount_l3475_347508


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_closed_open_interval_l3475_347519

def A : Set ℝ := {x : ℝ | |x| ≥ 2}
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

theorem A_intersect_B_equals_closed_open_interval :
  A ∩ B = Set.Icc 2 3 \ {3} :=
by sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_closed_open_interval_l3475_347519


namespace NUMINAMATH_CALUDE_coin_value_difference_l3475_347589

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value in cents for a given coin count -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- Represents the constraint that there are 2500 coins in total -/
def totalCoins (coins : CoinCount) : Prop :=
  coins.pennies + coins.nickels + coins.dimes = 2500

/-- Represents the constraint that there is at least one of each type of coin -/
def atLeastOne (coins : CoinCount) : Prop :=
  coins.pennies ≥ 1 ∧ coins.nickels ≥ 1 ∧ coins.dimes ≥ 1

theorem coin_value_difference :
  ∃ (maxCoins minCoins : CoinCount),
    totalCoins maxCoins ∧
    totalCoins minCoins ∧
    atLeastOne maxCoins ∧
    atLeastOne minCoins ∧
    (∀ (coins : CoinCount), totalCoins coins → atLeastOne coins →
      totalValue coins ≤ totalValue maxCoins) ∧
    (∀ (coins : CoinCount), totalCoins coins → atLeastOne coins →
      totalValue coins ≥ totalValue minCoins) ∧
    totalValue maxCoins - totalValue minCoins = 22473 :=
by sorry

end NUMINAMATH_CALUDE_coin_value_difference_l3475_347589


namespace NUMINAMATH_CALUDE_parabola_intersecting_line_slope_l3475_347568

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point M
def point_M : ℝ × ℝ := (-1, 1)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through a point with a given slope
def line_through_point (p : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop :=
  y - p.2 = k * (x - p.1)

-- Define the condition for a right angle
def is_right_angle (a b c : ℝ × ℝ) : Prop :=
  (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0

-- Main theorem
theorem parabola_intersecting_line_slope :
  ∀ (k : ℝ) (a b : ℝ × ℝ),
    (∀ x y, parabola x y ↔ (x, y) = a ∨ (x, y) = b) →
    (∀ x y, line_through_point focus k x y ↔ (x, y) = a ∨ (x, y) = b) →
    is_right_angle point_M a b →
    k = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_intersecting_line_slope_l3475_347568


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l3475_347590

theorem right_triangle_ratio (a b c x y : ℝ) : 
  a > 0 → b > 0 → c > 0 → x > 0 → y > 0 →
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  x * y = a^2 →     -- Geometric mean theorem for x
  x * y = b^2 →     -- Geometric mean theorem for y
  x + y = c →       -- x and y form the hypotenuse
  a / b = 2 / 5 →   -- Given ratio
  x / y = 4 / 25 := by sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l3475_347590


namespace NUMINAMATH_CALUDE_reciprocal_sum_pairs_l3475_347531

theorem reciprocal_sum_pairs : 
  ∃! (count : ℕ), ∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ m > 0 ∧ n > 0 ∧ 1 / m + 1 / n = 1 / 5) ∧
    pairs.card = count ∧
    count = 3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_pairs_l3475_347531


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3475_347557

theorem division_remainder_problem (a b : ℕ) 
  (h1 : a - b = 1390)
  (h2 : a = 1650)
  (h3 : a / b = 6) :
  a % b = 90 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3475_347557


namespace NUMINAMATH_CALUDE_range_of_function_l3475_347506

theorem range_of_function (x : ℝ) (h : x^2 ≥ 1) :
  x^2 + Real.sqrt (x^2 - 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l3475_347506


namespace NUMINAMATH_CALUDE_greatest_prime_divisor_digit_sum_l3475_347572

def number : Nat := 18191

theorem greatest_prime_divisor_digit_sum (p : Nat) : 
  Nat.Prime p ∧ 
  p ∣ number ∧ 
  (∀ q : Nat, Nat.Prime q → q ∣ number → q ≤ p) →
  (p / 10 + p % 10) = 16 := by
sorry

end NUMINAMATH_CALUDE_greatest_prime_divisor_digit_sum_l3475_347572


namespace NUMINAMATH_CALUDE_baseton_transaction_baseton_base_equation_baseton_base_value_l3475_347550

/-- The base of the number system in Baseton -/
def r : ℕ := sorry

/-- The cost of the laptop in base r -/
def laptop_cost : ℕ := 534

/-- The amount paid in base r -/
def amount_paid : ℕ := 1000

/-- The change received in base r -/
def change_received : ℕ := 366

/-- Conversion from base r to base 10 -/
def to_base_10 (n : ℕ) : ℕ := 
  (n / 100) * r^2 + ((n / 10) % 10) * r + (n % 10)

theorem baseton_transaction :
  to_base_10 laptop_cost + to_base_10 change_received = to_base_10 amount_paid :=
by sorry

theorem baseton_base_equation :
  r^3 - 8*r^2 - 9*r - 10 = 0 :=
by sorry

theorem baseton_base_value : r = 10 :=
by sorry

end NUMINAMATH_CALUDE_baseton_transaction_baseton_base_equation_baseton_base_value_l3475_347550


namespace NUMINAMATH_CALUDE_max_true_statements_l3475_347579

theorem max_true_statements (y : ℝ) : 
  let statements := [
    (0 < y^3 ∧ y^3 < 2),
    (y^3 > 2),
    (-2 < y ∧ y < 0),
    (0 < y ∧ y < 2),
    (0 < y - y^3 ∧ y - y^3 < 2)
  ]
  ∀ (s : Finset (Fin 5)), (∀ i ∈ s, statements[i]) → s.card ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l3475_347579


namespace NUMINAMATH_CALUDE_rooster_count_l3475_347593

theorem rooster_count (total_birds : ℕ) (rooster_ratio hen_ratio chick_ratio duck_ratio goose_ratio : ℕ) 
  (h1 : total_birds = 9000)
  (h2 : rooster_ratio = 4)
  (h3 : hen_ratio = 2)
  (h4 : chick_ratio = 6)
  (h5 : duck_ratio = 3)
  (h6 : goose_ratio = 1) :
  (total_birds * rooster_ratio) / (rooster_ratio + hen_ratio + chick_ratio + duck_ratio + goose_ratio) = 2250 := by
  sorry

end NUMINAMATH_CALUDE_rooster_count_l3475_347593


namespace NUMINAMATH_CALUDE_beetle_average_speed_l3475_347527

/-- Represents the terrain types --/
inductive Terrain
  | Flat
  | Sandy
  | Gravel

/-- Represents an insect (ant or beetle) --/
structure Insect where
  flatSpeed : ℝ  -- Speed on flat terrain in meters per minute
  sandySpeedFactor : ℝ  -- Factor to multiply flat speed for sandy terrain
  gravelSpeedFactor : ℝ  -- Factor to multiply flat speed for gravel terrain

/-- Calculates the distance traveled by an insect on a given terrain for a given time --/
def distanceTraveled (insect : Insect) (terrain : Terrain) (time : ℝ) : ℝ :=
  match terrain with
  | Terrain.Flat => insect.flatSpeed * time
  | Terrain.Sandy => insect.flatSpeed * insect.sandySpeedFactor * time
  | Terrain.Gravel => insect.flatSpeed * insect.gravelSpeedFactor * time

/-- The main theorem to prove --/
theorem beetle_average_speed :
  let ant : Insect := {
    flatSpeed := 50,  -- 600 meters / 12 minutes
    sandySpeedFactor := 0.9,  -- 10% decrease
    gravelSpeedFactor := 0.8  -- 20% decrease
  }
  let beetle : Insect := {
    flatSpeed := ant.flatSpeed * 0.85,  -- 15% less than ant
    sandySpeedFactor := 0.95,  -- 5% decrease
    gravelSpeedFactor := 0.75  -- 25% decrease
  }
  let totalDistance := 
    distanceTraveled beetle Terrain.Flat 4 +
    distanceTraveled beetle Terrain.Sandy 3 +
    distanceTraveled beetle Terrain.Gravel 5
  let totalTime := 12
  let averageSpeed := totalDistance / totalTime
  averageSpeed * (60 / 1000) = 2.2525 := by
  sorry

end NUMINAMATH_CALUDE_beetle_average_speed_l3475_347527


namespace NUMINAMATH_CALUDE_power_equality_l3475_347530

theorem power_equality : 5^29 * 4^15 = 2 * 10^29 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3475_347530


namespace NUMINAMATH_CALUDE_lcm_36_105_l3475_347578

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_105_l3475_347578


namespace NUMINAMATH_CALUDE_mikes_apples_l3475_347594

theorem mikes_apples (nancy_apples keith_ate_apples apples_left : ℝ) 
  (h1 : nancy_apples = 3.0)
  (h2 : keith_ate_apples = 6.0)
  (h3 : apples_left = 4.0) :
  ∃ mike_apples : ℝ, mike_apples = 7.0 ∧ mike_apples + nancy_apples - keith_ate_apples = apples_left :=
by sorry

end NUMINAMATH_CALUDE_mikes_apples_l3475_347594


namespace NUMINAMATH_CALUDE_perpendicular_parallel_lines_l3475_347514

/-- Given a line l with inclination 45°, line l₁ passing through A(3,2) and B(a,-1) perpendicular to l,
    and line l₂: 2x+by+1=0 parallel to l₁, prove that a + b = 8 -/
theorem perpendicular_parallel_lines (a b : ℝ) : 
  (∃ (l l₁ l₂ : Set (ℝ × ℝ)),
    -- l has inclination 45°
    (∀ (x y : ℝ), (x, y) ∈ l ↔ y = x) ∧
    -- l₁ passes through A(3,2) and B(a,-1)
    ((3, 2) ∈ l₁ ∧ (a, -1) ∈ l₁) ∧
    -- l₁ is perpendicular to l
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l ∧ (x₂, y₂) ∈ l₁ → (x₂ - x₁) * (y₂ - y₁) = -1) ∧
    -- l₂: 2x+by+1=0
    (∀ (x y : ℝ), (x, y) ∈ l₂ ↔ 2*x + b*y + 1 = 0) ∧
    -- l₂ is parallel to l₁
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l₁ ∧ (x₂, y₂) ∈ l₂ → (x₂ - x₁) * (y₂ - y₁) = 0))
  → a + b = 8 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_parallel_lines_l3475_347514


namespace NUMINAMATH_CALUDE_start_page_second_day_l3475_347549

/-- Given a book with 200 pages, and 20% read on the first day,
    prove that the page number to start reading on the second day is 41. -/
theorem start_page_second_day (total_pages : ℕ) (percent_read : ℚ) : 
  total_pages = 200 → percent_read = 1/5 → 
  (total_pages : ℚ) * percent_read + 1 = 41 := by
  sorry

end NUMINAMATH_CALUDE_start_page_second_day_l3475_347549


namespace NUMINAMATH_CALUDE_cubic_minus_xy_squared_factorization_l3475_347596

theorem cubic_minus_xy_squared_factorization (x y : ℝ) :
  x^3 - x*y^2 = x*(x+y)*(x-y) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_xy_squared_factorization_l3475_347596


namespace NUMINAMATH_CALUDE_toddler_count_l3475_347552

theorem toddler_count (bill_count : ℕ) (double_counted : ℕ) (missed : ℕ) : 
  bill_count = 26 → double_counted = 8 → missed = 3 → 
  bill_count - double_counted + missed = 21 := by
sorry

end NUMINAMATH_CALUDE_toddler_count_l3475_347552


namespace NUMINAMATH_CALUDE_winning_team_fourth_quarter_points_l3475_347513

/-- The points scored by the winning team in the fourth quarter of a basketball game. -/
def fourth_quarter_points (first_quarter_losing : ℕ) 
                          (second_quarter_increase : ℕ) 
                          (third_quarter_increase : ℕ) 
                          (total_points : ℕ) : ℕ :=
  let first_quarter_winning := 2 * first_quarter_losing
  let second_quarter_winning := first_quarter_winning + second_quarter_increase
  let third_quarter_winning := second_quarter_winning + third_quarter_increase
  total_points - third_quarter_winning

/-- Theorem stating that the winning team scored 30 points in the fourth quarter. -/
theorem winning_team_fourth_quarter_points : 
  fourth_quarter_points 10 10 20 80 = 30 := by
  sorry

end NUMINAMATH_CALUDE_winning_team_fourth_quarter_points_l3475_347513


namespace NUMINAMATH_CALUDE_factorial_difference_is_perfect_square_l3475_347599

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_difference_is_perfect_square (q : ℕ) (r : ℕ) 
  (h : factorial (q + 2) - factorial (q + 1) = factorial q * r) :
  r = (q + 1) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_is_perfect_square_l3475_347599


namespace NUMINAMATH_CALUDE_equidistant_line_slope_l3475_347529

-- Define the points P and Q
def P : ℝ × ℝ := (4, 6)
def Q : ℝ × ℝ := (6, 2)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem equidistant_line_slope :
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), y = m * x → 
      (x - P.1)^2 + (y - P.2)^2 = (x - Q.1)^2 + (y - Q.2)^2) ∧
    m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_line_slope_l3475_347529


namespace NUMINAMATH_CALUDE_m_range_l3475_347575

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 > m

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (2 - m) * x + 1 < (2 - m) * y + 1

-- Define the theorem
theorem m_range (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : m ∈ Set.Icc 1 2 ∧ m ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l3475_347575


namespace NUMINAMATH_CALUDE_factors_of_8_to_15_l3475_347570

/-- The number of positive factors of 8^15 is 46 -/
theorem factors_of_8_to_15 : Nat.card (Nat.divisors (8^15)) = 46 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_8_to_15_l3475_347570


namespace NUMINAMATH_CALUDE_integer_roots_iff_m_in_M_l3475_347553

/-- The set of values for m where the equation has only integer roots -/
def M : Set ℝ := {3, 7, 15, 6, 9}

/-- The quadratic equation in x parameterized by m -/
def equation (m : ℝ) (x : ℝ) : ℝ :=
  (m - 6) * (m - 9) * x^2 + (15 * m - 117) * x + 54

/-- A predicate to check if a real number is an integer -/
def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- The main theorem stating that the equation has only integer roots iff m ∈ M -/
theorem integer_roots_iff_m_in_M (m : ℝ) : 
  (∀ x : ℝ, equation m x = 0 → is_integer x) ↔ m ∈ M := by sorry

end NUMINAMATH_CALUDE_integer_roots_iff_m_in_M_l3475_347553


namespace NUMINAMATH_CALUDE_minimum_m_in_range_l3475_347567

/-- Represents a sequence of five consecutive integers -/
structure FiveConsecutiveIntegers where
  m : ℕ  -- The middle integer
  h1 : m > 2  -- Ensures all integers are positive

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- Checks if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k * k

/-- The main theorem -/
theorem minimum_m_in_range (seq : FiveConsecutiveIntegers) :
  (isPerfectSquare (3 * seq.m)) →
  (isPerfectCube (5 * seq.m)) →
  (∃ min_m : ℕ, 
    (∀ m : ℕ, m < min_m → ¬(isPerfectSquare (3 * m) ∧ isPerfectCube (5 * m))) ∧
    600 < min_m ∧
    min_m ≤ 800) :=
sorry

end NUMINAMATH_CALUDE_minimum_m_in_range_l3475_347567


namespace NUMINAMATH_CALUDE_longest_path_old_town_l3475_347558

structure OldTown where
  intersections : Nat
  start_color : Bool
  end_color : Bool

def longest_path (town : OldTown) : Nat :=
  sorry

theorem longest_path_old_town (town : OldTown) :
  town.intersections = 36 →
  town.start_color = town.end_color →
  longest_path town = 34 := by
  sorry

end NUMINAMATH_CALUDE_longest_path_old_town_l3475_347558


namespace NUMINAMATH_CALUDE_sam_non_black_cows_l3475_347581

/-- Given a herd of cows, calculate the number of non-black cows. -/
def non_black_cows (total : ℕ) (black : ℕ) : ℕ :=
  total - black

theorem sam_non_black_cows :
  let total := 18
  let black := (total / 2) + 5
  non_black_cows total black = 4 := by
sorry

end NUMINAMATH_CALUDE_sam_non_black_cows_l3475_347581


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l3475_347539

theorem sum_of_reciprocals_squared (a b : ℝ) (h : a > 0) (k : b > 0) (hab : a * b = 1) :
  1 / (1 + a^2) + 1 / (1 + b^2) = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l3475_347539


namespace NUMINAMATH_CALUDE_largest_root_l3475_347524

theorem largest_root (p q r : ℝ) 
  (sum_eq : p + q + r = 1)
  (sum_prod_eq : p * q + p * r + q * r = -8)
  (prod_eq : p * q * r = 15) :
  max p (max q r) = 3 := by sorry

end NUMINAMATH_CALUDE_largest_root_l3475_347524


namespace NUMINAMATH_CALUDE_unattainable_y_l3475_347566

theorem unattainable_y (x : ℝ) :
  (2 * x^2 + 3 * x + 4 ≠ 0) →
  ∃ y : ℝ, y = (1 - x) / (2 * x^2 + 3 * x + 4) ∧ y ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_unattainable_y_l3475_347566


namespace NUMINAMATH_CALUDE_pipe_length_l3475_347512

theorem pipe_length : ∀ (shorter_piece longer_piece total_length : ℕ),
  shorter_piece = 28 →
  longer_piece = shorter_piece + 12 →
  total_length = shorter_piece + longer_piece →
  total_length = 68 :=
by
  sorry

end NUMINAMATH_CALUDE_pipe_length_l3475_347512


namespace NUMINAMATH_CALUDE_tangent_segment_length_l3475_347560

/-- Given a square and a circle with the following properties:
    - The circle has a radius of 10
    - The circle is tangent to two adjacent sides of the square
    - The circle intersects the other two sides of the square, cutting off segments of 4 and 2 from the vertices
    This theorem proves that the length of the segment cut off from the vertex at the point of tangency is 8. -/
theorem tangent_segment_length (square_side : ℝ) (circle_radius : ℝ) (cut_segment1 : ℝ) (cut_segment2 : ℝ) :
  circle_radius = 10 →
  cut_segment1 = 4 →
  cut_segment2 = 2 →
  square_side = circle_radius + (square_side - cut_segment1 - cut_segment2) / 2 →
  square_side - circle_radius = 8 :=
by sorry

end NUMINAMATH_CALUDE_tangent_segment_length_l3475_347560


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3475_347533

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 10) : x^2 + 1/x^2 = 98 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3475_347533


namespace NUMINAMATH_CALUDE_real_roots_quadratic_equation_l3475_347580

theorem real_roots_quadratic_equation (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 4 * x - 1 = 0) ↔ k ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_equation_l3475_347580


namespace NUMINAMATH_CALUDE_odd_function_when_c_zero_increasing_when_b_zero_central_symmetry_more_than_two_roots_possible_l3475_347551

-- Define the function f
def f (b c x : ℝ) : ℝ := x * |x| + b * x + c

-- Statement 1
theorem odd_function_when_c_zero (b : ℝ) :
  (∀ x : ℝ, f b 0 (-x) = -(f b 0 x)) := by sorry

-- Statement 2
theorem increasing_when_b_zero (c : ℝ) :
  Monotone (f 0 c) := by sorry

-- Statement 3
theorem central_symmetry (b c : ℝ) :
  ∀ x : ℝ, f b c (-x) + f b c x = 2 * c := by sorry

-- Statement 4 (negation of the original statement)
theorem more_than_two_roots_possible :
  ∃ b c : ℝ, ∃ x₁ x₂ x₃ : ℝ, (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
  (f b c x₁ = 0 ∧ f b c x₂ = 0 ∧ f b c x₃ = 0) := by sorry

end NUMINAMATH_CALUDE_odd_function_when_c_zero_increasing_when_b_zero_central_symmetry_more_than_two_roots_possible_l3475_347551


namespace NUMINAMATH_CALUDE_total_protest_days_l3475_347526

/-- Given a first protest lasting 4 days and a second protest lasting 25% longer,
    prove that the total number of days spent protesting is 9. -/
theorem total_protest_days : 
  let first_protest_days : ℕ := 4
  let second_protest_days : ℕ := first_protest_days + first_protest_days / 4
  first_protest_days + second_protest_days = 9 := by sorry

end NUMINAMATH_CALUDE_total_protest_days_l3475_347526


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l3475_347595

theorem right_triangle_acute_angles (α β : ℝ) : 
  α > 0 ∧ β > 0 ∧  -- Acute angles are positive
  α + β = 90 ∧     -- Sum of acute angles in a right triangle is 90°
  α = 4 * β →      -- Ratio of acute angles is 4:1
  (min α β = 18 ∧ max α β = 72) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l3475_347595


namespace NUMINAMATH_CALUDE_dolphin_count_theorem_l3475_347532

/-- Given an initial number of dolphins in the ocean and a factor for additional dolphins joining,
    calculate the total number of dolphins after joining. -/
def total_dolphins (initial : ℕ) (joining_factor : ℕ) : ℕ :=
  initial + joining_factor * initial

/-- Theorem stating that with 65 initial dolphins and 3 times that number joining,
    the total number of dolphins is 260. -/
theorem dolphin_count_theorem :
  total_dolphins 65 3 = 260 := by
  sorry

end NUMINAMATH_CALUDE_dolphin_count_theorem_l3475_347532


namespace NUMINAMATH_CALUDE_bear_color_theorem_l3475_347501

/-- Represents the Earth's surface --/
structure EarthSurface where
  latitude : ℝ
  longitude : ℝ

/-- Represents a bear --/
inductive Bear
| Polar
| Other

/-- Represents the hunter's position and orientation --/
structure HunterState where
  position : EarthSurface
  facing : EarthSurface

/-- Function to determine if a point is at the North Pole --/
def isNorthPole (p : EarthSurface) : Prop :=
  p.latitude = 90 -- Assuming 90 degrees latitude is the North Pole

/-- Function to move a point on the Earth's surface --/
def move (start : EarthSurface) (direction : String) (distance : ℝ) : EarthSurface :=
  sorry -- Implementation details omitted

/-- Function to determine the type of bear based on location --/
def bearType (location : EarthSurface) : Bear :=
  sorry -- Implementation details omitted

/-- The main theorem --/
theorem bear_color_theorem 
  (bear_position : EarthSurface)
  (initial_hunter_position : EarthSurface)
  (h1 : initial_hunter_position = move bear_position "south" 100)
  (h2 : let east_position := move initial_hunter_position "east" 100
        east_position.latitude = initial_hunter_position.latitude)
  (h3 : let final_hunter_state := HunterState.mk (move initial_hunter_position "east" 100) bear_position
        final_hunter_state.facing = bear_position)
  : bearType bear_position = Bear.Polar :=
sorry


end NUMINAMATH_CALUDE_bear_color_theorem_l3475_347501


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3475_347562

theorem quadratic_inequality (x : ℝ) :
  x^2 - 50*x + 575 ≤ 25 ↔ 25 - 5*Real.sqrt 3 ≤ x ∧ x ≤ 25 + 5*Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3475_347562


namespace NUMINAMATH_CALUDE_water_container_problem_l3475_347541

/-- Given a container with capacity 120 liters, if adding 48 liters makes it 3/4 full,
    then the initial percentage of water in the container was 35%. -/
theorem water_container_problem :
  let capacity : ℝ := 120
  let added_water : ℝ := 48
  let final_fraction : ℝ := 3/4
  let initial_percentage : ℝ := (final_fraction * capacity - added_water) / capacity * 100
  initial_percentage = 35 := by sorry

end NUMINAMATH_CALUDE_water_container_problem_l3475_347541


namespace NUMINAMATH_CALUDE_optimal_revenue_model_depends_on_factors_l3475_347565

/-- Represents the revenue model for a movie --/
inductive RevenueModel
  | Forever
  | Rental

/-- Represents various economic factors --/
structure EconomicFactors where
  immediateRevenue : ℝ
  longTermRevenuePotential : ℝ
  customerPriceSensitivity : ℝ
  administrativeCosts : ℝ
  piracyRisks : ℝ

/-- Calculates the overall economic value of a revenue model --/
def economicValue (model : RevenueModel) (factors : EconomicFactors) : ℝ :=
  sorry

/-- The theorem stating that the optimal revenue model depends on economic factors --/
theorem optimal_revenue_model_depends_on_factors
  (factors : EconomicFactors) :
  ∃ (model : RevenueModel),
    ∀ (other : RevenueModel),
      economicValue model factors ≥ economicValue other factors :=
  sorry

end NUMINAMATH_CALUDE_optimal_revenue_model_depends_on_factors_l3475_347565


namespace NUMINAMATH_CALUDE_fifteen_subcommittees_l3475_347582

/-- The number of ways to form a two-person sub-committee from a larger committee,
    where one member must be from a designated group. -/
def subcommittee_count (total : ℕ) (designated : ℕ) : ℕ :=
  designated * (total - designated)

/-- Theorem stating that for a committee of 8 people with a designated group of 3,
    there are 15 possible two-person sub-committees. -/
theorem fifteen_subcommittees :
  subcommittee_count 8 3 = 15 := by
  sorry

#eval subcommittee_count 8 3

end NUMINAMATH_CALUDE_fifteen_subcommittees_l3475_347582


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3475_347517

theorem fractional_equation_solution :
  ∃ (x : ℝ), x ≠ 3 ∧ (2 - x) / (x - 3) + 1 / (3 - x) = 1 ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3475_347517


namespace NUMINAMATH_CALUDE_min_value_theorem_l3475_347574

theorem min_value_theorem (x y : ℝ) : (x^2*y - 1)^2 + (x + y - 1)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3475_347574


namespace NUMINAMATH_CALUDE_horner_method_f_2_l3475_347576

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem horner_method_f_2 :
  f 2 = horner_eval [1, 0, 2, 3, 1, 1] 2 ∧ horner_eval [1, 0, 2, 3, 1, 1] 2 = 41 := by
  sorry

#eval f 2
#eval horner_eval [1, 0, 2, 3, 1, 1] 2

end NUMINAMATH_CALUDE_horner_method_f_2_l3475_347576


namespace NUMINAMATH_CALUDE_complex_number_location_l3475_347588

theorem complex_number_location :
  let z : ℂ := 2 / (1 - Complex.I)
  z = 1 + Complex.I ∧ z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l3475_347588


namespace NUMINAMATH_CALUDE_sin_negative_1740_degrees_l3475_347505

theorem sin_negative_1740_degrees : 
  Real.sin ((-1740 : ℝ) * π / 180) = (Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_sin_negative_1740_degrees_l3475_347505


namespace NUMINAMATH_CALUDE_complex_value_calculation_l3475_347537

theorem complex_value_calculation : Complex.I - (1 / Complex.I) = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_value_calculation_l3475_347537


namespace NUMINAMATH_CALUDE_geese_flock_count_l3475_347545

theorem geese_flock_count : ∃ x : ℕ, 
  (x + x + x / 2 + x / 4 + 1 = 100) ∧ (x = 36) := by
  sorry

end NUMINAMATH_CALUDE_geese_flock_count_l3475_347545


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l3475_347540

/-- The standard equation of a hyperbola with foci on the y-axis -/
def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop :=
  y^2 / a^2 - x^2 / b^2 = 1

/-- Theorem: The standard equation of a hyperbola with foci on the y-axis,
    semi-minor axis length of 4, and semi-focal distance of 6 -/
theorem hyperbola_standard_equation :
  let b : ℝ := 4  -- semi-minor axis length
  let c : ℝ := 6  -- semi-focal distance
  let a : ℝ := (c^2 - b^2).sqrt  -- semi-major axis length
  ∀ x y : ℝ, hyperbola_equation a b x y ↔ y^2 / 20 - x^2 / 16 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l3475_347540


namespace NUMINAMATH_CALUDE_joes_lift_l3475_347536

theorem joes_lift (total : ℕ) (diff : ℕ) (first_lift : ℕ) (second_lift : ℕ) 
  (h1 : total = 600)
  (h2 : first_lift + second_lift = total)
  (h3 : 2 * first_lift = second_lift + diff)
  (h4 : diff = 300) : 
  first_lift = 300 := by
sorry

end NUMINAMATH_CALUDE_joes_lift_l3475_347536


namespace NUMINAMATH_CALUDE_polynomial_sum_property_l3475_347504

/-- Generate all words of length n using letters A and B -/
def generateWords (n : ℕ) : List String :=
  sorry

/-- Convert a word to a polynomial by replacing A with x and B with (1-x) -/
def wordToPolynomial (word : String) : ℝ → ℝ :=
  sorry

/-- Sum the first k polynomials -/
def sumPolynomials (n : ℕ) (k : ℕ) : ℝ → ℝ :=
  sorry

/-- A function is increasing on [0,1] -/
def isIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y

/-- A function is constant on [0,1] -/
def isConstant (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → f x = f y

theorem polynomial_sum_property (n : ℕ) (k : ℕ) (h : 1 ≤ k ∧ k ≤ 2^n) :
  let f := sumPolynomials n k
  isConstant f ∨ isIncreasing f :=
sorry

end NUMINAMATH_CALUDE_polynomial_sum_property_l3475_347504


namespace NUMINAMATH_CALUDE_n_sided_polygon_interior_angles_l3475_347503

theorem n_sided_polygon_interior_angles (n : ℕ) : 
  (n - 2) * 180 = 720 → n = 6 := by sorry

end NUMINAMATH_CALUDE_n_sided_polygon_interior_angles_l3475_347503


namespace NUMINAMATH_CALUDE_money_transfer_game_probability_l3475_347523

/-- Represents the state of the game as a triple of integers -/
def GameState := ℕ × ℕ × ℕ

/-- The initial state of the game -/
def initialState : GameState := (3, 3, 3)

/-- Represents a single transfer in the game -/
def Transfer := GameState → GameState

/-- The set of all possible transfers in the game -/
def allTransfers : Set Transfer := sorry

/-- The transition matrix for the Markov chain representing the game -/
def transitionMatrix : GameState → GameState → ℝ := sorry

/-- The steady state distribution of the Markov chain -/
def steadyStateDistribution : GameState → ℝ := sorry

theorem money_transfer_game_probability :
  steadyStateDistribution initialState = 8 / 13 := by sorry

end NUMINAMATH_CALUDE_money_transfer_game_probability_l3475_347523


namespace NUMINAMATH_CALUDE_circle_equation_l3475_347520

/-- The equation of a circle in its general form -/
def is_circle (h x y a : ℝ) : Prop :=
  ∃ (c_x c_y r : ℝ), (x - c_x)^2 + (y - c_y)^2 = r^2 ∧ r > 0

/-- The given equation -/
def given_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 5*a = 0

theorem circle_equation (x y : ℝ) :
  is_circle 0 x y 1 ↔ given_equation x y 1 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3475_347520


namespace NUMINAMATH_CALUDE_compute_expression_l3475_347544

theorem compute_expression : 20 * (144 / 3 + 36 / 6 + 16 / 32 + 2) = 1130 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3475_347544


namespace NUMINAMATH_CALUDE_machine_C_time_l3475_347592

/-- Time for Machine A to finish the job -/
def timeA : ℝ := 4

/-- Time for Machine B to finish the job -/
def timeB : ℝ := 12

/-- Time for all machines together to finish the job -/
def timeAll : ℝ := 2

/-- Time for Machine C to finish the job -/
def timeC : ℝ := 6

/-- Theorem stating that given the conditions, Machine C takes 6 hours to finish the job alone -/
theorem machine_C_time : 
  1 / timeA + 1 / timeB + 1 / timeC = 1 / timeAll := by sorry

end NUMINAMATH_CALUDE_machine_C_time_l3475_347592


namespace NUMINAMATH_CALUDE_new_person_age_l3475_347559

theorem new_person_age (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) : 
  n = 9 → initial_avg = 15 → new_avg = 17 → 
  ∃ (new_person_age : ℝ), 
    (n * initial_avg + new_person_age) / (n + 1) = new_avg ∧ 
    new_person_age = 35 := by
  sorry

end NUMINAMATH_CALUDE_new_person_age_l3475_347559


namespace NUMINAMATH_CALUDE_fraction_equality_existence_l3475_347554

theorem fraction_equality_existence :
  (∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / c + (1 : ℚ) / d) ∧
  (∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
    (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / c + (1 : ℚ) / d + (1 : ℚ) / e) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_existence_l3475_347554


namespace NUMINAMATH_CALUDE_range_of_f_l3475_347587

open Real

theorem range_of_f (x : ℝ) (h : x ∈ Set.Icc 0 (π / 2)) :
  let f := λ x : ℝ => 3 * sin (2 * x - π / 6)
  ∃ y, y ∈ Set.Icc (-3/2) 3 ∧ ∃ x, x ∈ Set.Icc 0 (π / 2) ∧ f x = y :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l3475_347587


namespace NUMINAMATH_CALUDE_math_olympiad_properties_l3475_347542

/-- Represents the math Olympiad team composition and assessment rules -/
structure MathOlympiadTeam where
  total_students : Nat
  grade_11_students : Nat
  grade_12_students : Nat
  grade_13_students : Nat
  selected_students : Nat
  prob_correct_easy : Rat
  prob_correct_hard : Rat
  points_easy : Nat
  points_hard : Nat
  excellent_threshold : Nat

/-- Calculates the probability of selecting exactly 2 students from Grade 11 -/
def prob_two_from_grade_11 (team : MathOlympiadTeam) : Rat :=
  sorry

/-- Calculates the mathematical expectation of Zhang's score -/
def expected_score (team : MathOlympiadTeam) : Rat :=
  sorry

/-- Calculates the probability of Zhang being an excellent student -/
def prob_excellent_student (team : MathOlympiadTeam) : Rat :=
  sorry

/-- The main theorem proving the three required properties -/
theorem math_olympiad_properties (team : MathOlympiadTeam)
  (h1 : team.total_students = 20)
  (h2 : team.grade_11_students = 8)
  (h3 : team.grade_12_students = 7)
  (h4 : team.grade_13_students = 5)
  (h5 : team.selected_students = 3)
  (h6 : team.prob_correct_easy = 2/3)
  (h7 : team.prob_correct_hard = 1/2)
  (h8 : team.points_easy = 1)
  (h9 : team.points_hard = 2)
  (h10 : team.excellent_threshold = 5) :
  prob_two_from_grade_11 team = 28/95 ∧
  expected_score team = 10/3 ∧
  prob_excellent_student team = 2/9 :=
by
  sorry

end NUMINAMATH_CALUDE_math_olympiad_properties_l3475_347542


namespace NUMINAMATH_CALUDE_smaller_to_larger_volume_ratio_l3475_347534

/-- Represents a regular octahedron -/
structure RegularOctahedron where
  -- Add necessary fields if needed

/-- Represents the smaller octahedron formed by face centers -/
def smaller_octahedron (o : RegularOctahedron) : RegularOctahedron :=
  sorry

/-- Calculates the volume of an octahedron -/
def volume (o : RegularOctahedron) : ℝ :=
  sorry

/-- Theorem stating the volume ratio of smaller to larger octahedron -/
theorem smaller_to_larger_volume_ratio (o : RegularOctahedron) :
  volume (smaller_octahedron o) / volume o = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_smaller_to_larger_volume_ratio_l3475_347534


namespace NUMINAMATH_CALUDE_trick_deck_cost_is_nine_l3475_347515

/-- The cost of a single trick deck, given that 8 decks cost 72 dollars -/
def trick_deck_cost : ℚ :=
  72 / 8

/-- Theorem stating that the cost of each trick deck is 9 dollars -/
theorem trick_deck_cost_is_nine : trick_deck_cost = 9 := by
  sorry

end NUMINAMATH_CALUDE_trick_deck_cost_is_nine_l3475_347515


namespace NUMINAMATH_CALUDE_ababab_divisible_by_13_l3475_347569

theorem ababab_divisible_by_13 (a b : Nat) (h1 : a < 10) (h2 : b < 10) :
  ∃ k : Nat, 100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b = 13 * k := by
  sorry

end NUMINAMATH_CALUDE_ababab_divisible_by_13_l3475_347569


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l3475_347522

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l3475_347522


namespace NUMINAMATH_CALUDE_car_y_time_is_one_third_correct_graph_is_c_l3475_347585

/-- Represents a car's travel characteristics -/
structure Car where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The scenario of two cars traveling the same distance -/
def TwoCarScenario (x y : Car) : Prop :=
  x.distance = y.distance ∧ y.speed = 3 * x.speed

/-- Theorem: In the given scenario, Car Y's time is one-third of Car X's time -/
theorem car_y_time_is_one_third (x y : Car) 
  (h : TwoCarScenario x y) : y.time = x.time / 3 := by
  sorry

/-- Theorem: The correct graph representation matches option C -/
theorem correct_graph_is_c (x y : Car) 
  (h : TwoCarScenario x y) : 
  (x.speed = y.speed / 3 ∧ x.time = y.time * 3) := by
  sorry

end NUMINAMATH_CALUDE_car_y_time_is_one_third_correct_graph_is_c_l3475_347585
