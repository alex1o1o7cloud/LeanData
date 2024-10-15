import Mathlib

namespace NUMINAMATH_CALUDE_daisy_percentage_in_bouquet_l985_98529

theorem daisy_percentage_in_bouquet : 
  ∀ (total_flowers : ℕ) (white_flowers yellow_flowers white_tulips yellow_tulips white_daisies yellow_daisies : ℕ),
  total_flowers > 0 →
  white_flowers + yellow_flowers = total_flowers →
  white_tulips + white_daisies = white_flowers →
  yellow_tulips + yellow_daisies = yellow_flowers →
  white_tulips = white_flowers / 2 →
  yellow_daisies = (2 * yellow_flowers) / 3 →
  white_flowers = (7 * total_flowers) / 10 →
  (white_daisies + yellow_daisies) * 100 = 55 * total_flowers :=
by sorry

end NUMINAMATH_CALUDE_daisy_percentage_in_bouquet_l985_98529


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l985_98514

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The incident ray passes through these points -/
def M : Point := { x := 3, y := -2 }
def P : Point := { x := 0, y := 1 }

/-- P is on the y-axis -/
axiom P_on_y_axis : P.x = 0

/-- Function to check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.A * p.x + l.B * p.y + l.C = 0

/-- The reflected ray -/
def reflected_ray : Line := { A := 1, B := -1, C := 1 }

/-- Theorem stating that the reflected ray has the equation x - y + 1 = 0 -/
theorem reflected_ray_equation :
  point_on_line P reflected_ray ∧
  point_on_line { x := -M.x, y := M.y } reflected_ray :=
sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l985_98514


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l985_98504

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l985_98504


namespace NUMINAMATH_CALUDE_power_product_equality_l985_98564

theorem power_product_equality (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l985_98564


namespace NUMINAMATH_CALUDE_handshake_frames_remaining_l985_98593

theorem handshake_frames_remaining (d₁ d₂ : ℕ) 
  (h₁ : d₁ % 9 = 4) 
  (h₂ : d₂ % 9 = 6) : 
  (d₁ * d₂) % 9 = 6 := by
sorry

end NUMINAMATH_CALUDE_handshake_frames_remaining_l985_98593


namespace NUMINAMATH_CALUDE_interior_angles_sum_l985_98538

/-- Given a convex polygon where the sum of interior angles is 3600 degrees,
    prove that the sum of interior angles of a polygon with 3 more sides is 4140 degrees. -/
theorem interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 3600) → (180 * ((n + 3) - 2) = 4140) := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l985_98538


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l985_98575

/-- A geometric sequence with positive terms, a₁ = 1, and a₁ + a₂ + a₃ = 7 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (a 1 = 1) ∧
  (a 1 + a 2 + a 3 = 7) ∧
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n)

/-- The general formula for the geometric sequence -/
theorem geometric_sequence_formula (a : ℕ → ℝ) (h : GeometricSequence a) :
  ∀ n : ℕ, a n = 2^(n - 1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l985_98575


namespace NUMINAMATH_CALUDE_coin_denomination_l985_98586

/-- Given a total bill of 285 pesos, paid with 11 20-peso bills and 11 coins of unknown denomination,
    prove that the denomination of the coins must be 5 pesos. -/
theorem coin_denomination (total_bill : ℕ) (bill_value : ℕ) (num_bills : ℕ) (num_coins : ℕ) 
  (h1 : total_bill = 285)
  (h2 : bill_value = 20)
  (h3 : num_bills = 11)
  (h4 : num_coins = 11) :
  ∃ (coin_value : ℕ), coin_value = 5 ∧ total_bill = num_bills * bill_value + num_coins * coin_value :=
by
  sorry

#check coin_denomination

end NUMINAMATH_CALUDE_coin_denomination_l985_98586


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l985_98516

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  1 / (2*a + b) + 1 / (2*b + c) + 1 / (2*c + a) ≥ 3 ∧
  (1 / (2*a + b) + 1 / (2*b + c) + 1 / (2*c + a) = 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l985_98516


namespace NUMINAMATH_CALUDE_sams_mystery_books_l985_98588

theorem sams_mystery_books (total_books : ℝ) (used_adventure_books : ℝ) (new_crime_books : ℝ)
  (h1 : total_books = 45)
  (h2 : used_adventure_books = 13)
  (h3 : new_crime_books = 15) :
  total_books - (used_adventure_books + new_crime_books) = 17 :=
by sorry

end NUMINAMATH_CALUDE_sams_mystery_books_l985_98588


namespace NUMINAMATH_CALUDE_division_result_l985_98534

theorem division_result : 75 / 0.05 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l985_98534


namespace NUMINAMATH_CALUDE_m_range_l985_98599

-- Define the plane region
def plane_region (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |2 * p.1 + p.2 + m| < 3}

-- Define the theorem
theorem m_range (m : ℝ) :
  ((0, 0) ∈ plane_region m) ∧ ((-1, 1) ∈ plane_region m) →
  -2 < m ∧ m < 3 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l985_98599


namespace NUMINAMATH_CALUDE_hyperbola_properties_l985_98522

-- Define the hyperbola
def Hyperbola (a b h k : ℝ) (x y : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

-- Define the asymptotes
def Asymptote1 (x y : ℝ) : Prop := y = 3 * x + 6
def Asymptote2 (x y : ℝ) : Prop := y = -3 * x - 2

theorem hyperbola_properties :
  ∃ (a b h k : ℝ),
    (∀ x y, Asymptote1 x y ∨ Asymptote2 x y → Hyperbola a b h k x y) ∧
    Hyperbola a b h k 1 9 ∧
    a + h = (21 * Real.sqrt 6 - 8) / 6 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l985_98522


namespace NUMINAMATH_CALUDE_robe_cost_is_two_l985_98558

/-- Calculates the cost per robe given the total number of singers, existing robes, and total cost for new robes. -/
def cost_per_robe (total_singers : ℕ) (existing_robes : ℕ) (total_cost : ℕ) : ℚ :=
  total_cost / (total_singers - existing_robes)

/-- Proves that the cost per robe is $2 given the specific conditions of the problem. -/
theorem robe_cost_is_two :
  cost_per_robe 30 12 36 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robe_cost_is_two_l985_98558


namespace NUMINAMATH_CALUDE_wood_length_after_sawing_l985_98584

/-- The new length of a piece of wood after sawing off a portion. -/
def new_wood_length (original_length saw_off_length : ℝ) : ℝ :=
  original_length - saw_off_length

/-- Theorem stating that the new length of the wood is 6.6 cm. -/
theorem wood_length_after_sawing :
  new_wood_length 8.9 2.3 = 6.6 := by
  sorry

end NUMINAMATH_CALUDE_wood_length_after_sawing_l985_98584


namespace NUMINAMATH_CALUDE_not_perfect_square_l985_98587

theorem not_perfect_square (m : ℕ) : ¬ ∃ (n : ℕ), ((4 * 10^(2*m+1) + 5) / 9 : ℚ) = n^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l985_98587


namespace NUMINAMATH_CALUDE_kennel_dogs_l985_98523

theorem kennel_dogs (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 3 / 4 →
  cats = dogs - 8 →
  dogs = 32 := by
sorry

end NUMINAMATH_CALUDE_kennel_dogs_l985_98523


namespace NUMINAMATH_CALUDE_f_properties_l985_98582

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x / Real.exp x) + (1/2) * x^2 - x

def monotonic_intervals (a : ℝ) : Prop :=
  (a ≤ 0 → (∀ x y, x < y → x < 1 → f a x > f a y) ∧ 
            (∀ x y, x < y → 1 < x → f a x < f a y)) ∧
  (a = Real.exp 1 → (∀ x y, x < y → f a x < f a y)) ∧
  (0 < a ∧ a < Real.exp 1 → 
    (∀ x y, x < y → y < Real.log a → f a x < f a y) ∧
    (∀ x y, x < y → Real.log a < x ∧ y < 1 → f a x > f a y) ∧
    (∀ x y, x < y → 1 < x → f a x < f a y)) ∧
  (Real.exp 1 < a → 
    (∀ x y, x < y → y < 1 → f a x < f a y) ∧
    (∀ x y, x < y → 1 < x ∧ y < Real.log a → f a x > f a y) ∧
    (∀ x y, x < y → Real.log a < x → f a x < f a y))

def number_of_zeros (a : ℝ) : Prop :=
  (Real.exp 1 / 2 < a → ∃! x, f a x = 0) ∧
  ((a = 1 ∨ a = Real.exp 1 / 2) → ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ 
    (∀ z, f a z = 0 → z = x ∨ z = y)) ∧
  (1 < a ∧ a < Real.exp 1 / 2 → ∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f a x = 0 ∧ f a y = 0 ∧ f a z = 0 ∧ 
    (∀ w, f a w = 0 → w = x ∨ w = y ∨ w = z))

theorem f_properties (a : ℝ) (h : 1 ≤ a) : 
  monotonic_intervals a ∧ number_of_zeros a := by sorry

end NUMINAMATH_CALUDE_f_properties_l985_98582


namespace NUMINAMATH_CALUDE_max_toys_frank_can_buy_l985_98559

/-- Represents the types of toys available --/
inductive Toy
| SmallCar
| Puzzle
| LegoSet

/-- Returns the price of a toy --/
def toyPrice (t : Toy) : ℕ :=
  match t with
  | Toy.SmallCar => 8
  | Toy.Puzzle => 12
  | Toy.LegoSet => 20

/-- Represents a shopping cart with toys --/
structure Cart :=
  (smallCars : ℕ)
  (puzzles : ℕ)
  (legoSets : ℕ)

/-- Calculates the total cost of a cart, considering the promotion --/
def cartCost (c : Cart) : ℕ :=
  (c.smallCars / 3 * 2 + c.smallCars % 3) * toyPrice Toy.SmallCar +
  (c.puzzles / 3 * 2 + c.puzzles % 3) * toyPrice Toy.Puzzle +
  (c.legoSets / 3 * 2 + c.legoSets % 3) * toyPrice Toy.LegoSet

/-- Calculates the total number of toys in a cart --/
def cartSize (c : Cart) : ℕ :=
  c.smallCars + c.puzzles + c.legoSets

/-- Theorem: The maximum number of toys Frank can buy with $40 is 6 --/
theorem max_toys_frank_can_buy :
  ∀ c : Cart, cartCost c ≤ 40 → cartSize c ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_toys_frank_can_buy_l985_98559


namespace NUMINAMATH_CALUDE_defective_pens_l985_98528

theorem defective_pens (total : ℕ) (prob : ℚ) (defective : ℕ) : 
  total = 8 →
  prob = 15/28 →
  (total - defective : ℚ) / total * ((total - defective - 1) : ℚ) / (total - 1) = prob →
  defective = 2 :=
sorry

end NUMINAMATH_CALUDE_defective_pens_l985_98528


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_9_4_l985_98565

theorem smallest_four_digit_mod_9_4 : ∃ (n : ℕ), 
  (n ≥ 1000) ∧ 
  (n % 9 = 4) ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m % 9 = 4 → m ≥ n) ∧ 
  (n = 1003) := by
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_9_4_l985_98565


namespace NUMINAMATH_CALUDE_dice_product_divisible_by_8_l985_98539

/-- The number of dice rolled simultaneously -/
def num_dice : ℕ := 8

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The probability that a single die roll is divisible by 2 -/
def prob_divisible_by_2 : ℚ := 1/2

/-- The probability that the product of dice rolls is divisible by 8 -/
def prob_product_divisible_by_8 : ℚ := 247/256

/-- Theorem: The probability that the product of 8 standard 6-sided dice rolls is divisible by 8 is 247/256 -/
theorem dice_product_divisible_by_8 :
  prob_product_divisible_by_8 = 247/256 :=
sorry

end NUMINAMATH_CALUDE_dice_product_divisible_by_8_l985_98539


namespace NUMINAMATH_CALUDE_price_change_l985_98577

theorem price_change (P : ℝ) (h : P > 0) :
  let price_2012 := P * 1.25
  let price_2013 := price_2012 * 0.88
  (price_2013 - P) / P * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_price_change_l985_98577


namespace NUMINAMATH_CALUDE_binary_multiplication_and_shift_l985_98557

theorem binary_multiplication_and_shift :
  let a : Nat := 109  -- 1101101₂ in decimal
  let b : Nat := 15   -- 1111₂ in decimal
  let product : Nat := a * b
  let shifted : Rat := (product : Rat) / 4  -- Shifting 2 places right is equivalent to dividing by 4
  shifted = 1010011111.25 := by sorry

end NUMINAMATH_CALUDE_binary_multiplication_and_shift_l985_98557


namespace NUMINAMATH_CALUDE_jogs_five_miles_per_day_l985_98537

/-- Represents the number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- Represents the number of weeks -/
def num_weeks : ℕ := 3

/-- Represents the total miles run over the given weeks -/
def total_miles : ℕ := 75

/-- Calculates the number of miles jogged per day -/
def miles_per_day : ℚ :=
  total_miles / (weekdays_per_week * num_weeks)

/-- Theorem stating that the person jogs 5 miles per day -/
theorem jogs_five_miles_per_day : miles_per_day = 5 := by
  sorry

end NUMINAMATH_CALUDE_jogs_five_miles_per_day_l985_98537


namespace NUMINAMATH_CALUDE_tornado_distance_l985_98570

/-- Given a tornado that transported objects as follows:
  * A car was transported 200 feet
  * A lawn chair was blown twice as far as the car
  * A birdhouse flew three times farther than the lawn chair
  This theorem proves that the birdhouse flew 1200 feet. -/
theorem tornado_distance (car_distance : ℕ) (lawn_chair_multiplier : ℕ) (birdhouse_multiplier : ℕ)
  (h1 : car_distance = 200)
  (h2 : lawn_chair_multiplier = 2)
  (h3 : birdhouse_multiplier = 3) :
  birdhouse_multiplier * (lawn_chair_multiplier * car_distance) = 1200 := by
  sorry

#check tornado_distance

end NUMINAMATH_CALUDE_tornado_distance_l985_98570


namespace NUMINAMATH_CALUDE_min_value_theorem_l985_98553

theorem min_value_theorem (x y : ℝ) 
  (h : ∀ (n : ℕ), n > 0 → n * x + (1 / n) * y ≥ 1) :
  (∀ (a b : ℝ), (∀ (n : ℕ), n > 0 → n * a + (1 / n) * b ≥ 1) → 41 * x + 2 * y ≤ 41 * a + 2 * b) ∧ 
  (∃ (x₀ y₀ : ℝ), (∀ (n : ℕ), n > 0 → n * x₀ + (1 / n) * y₀ ≥ 1) ∧ 41 * x₀ + 2 * y₀ = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l985_98553


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l985_98590

theorem matrix_equation_proof : 
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![46/7, -58/7; -39/14, 51/14]
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 9, 3]
  N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l985_98590


namespace NUMINAMATH_CALUDE_lemonade_syrup_parts_l985_98566

/-- Given a solution with water and lemonade syrup, prove the original amount of syrup --/
theorem lemonade_syrup_parts (x : ℝ) : 
  x > 0 → -- Ensure x is positive
  x / (x + 8) ≠ 1/5 → -- Ensure the original solution is not already 20% syrup
  x / (x + 8 - 2.1428571428571423 + 2.1428571428571423) = 1/5 → -- After replacement, solution is 20% syrup
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_syrup_parts_l985_98566


namespace NUMINAMATH_CALUDE_max_points_world_cup_group_l985_98563

/-- The maximum sum of points for all teams in a World Cup group stage -/
theorem max_points_world_cup_group (n : ℕ) (win_points tie_points : ℕ) : 
  n = 4 → win_points = 3 → tie_points = 1 → 
  (n.choose 2) * win_points = 18 :=
by sorry

end NUMINAMATH_CALUDE_max_points_world_cup_group_l985_98563


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l985_98519

/-- Given two arithmetic sequences, this theorem proves that if the ratio of their sums
    follows a specific pattern, then the ratio of their 7th terms is 13/20. -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Sum formula for arithmetic sequence a
  (∀ n, T n = (n * (b 1 + b n)) / 2) →  -- Sum formula for arithmetic sequence b
  (∀ n, S n / T n = n / (n + 7)) →      -- Given condition
  a 7 / b 7 = 13 / 20 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l985_98519


namespace NUMINAMATH_CALUDE_shelter_blocks_count_l985_98512

/-- Calculates the number of blocks needed for a rectangular shelter --/
def shelter_blocks (length width height : ℕ) : ℕ :=
  let exterior_volume := length * width * height
  let interior_length := length - 2
  let interior_width := width - 2
  let interior_height := height - 2
  let interior_volume := interior_length * interior_width * interior_height
  exterior_volume - interior_volume

/-- Proves that the number of blocks for a shelter with given dimensions is 528 --/
theorem shelter_blocks_count :
  shelter_blocks 14 12 6 = 528 := by
  sorry

end NUMINAMATH_CALUDE_shelter_blocks_count_l985_98512


namespace NUMINAMATH_CALUDE_simplify_fraction_l985_98591

theorem simplify_fraction : (5^4 + 5^2 + 5) / (5^3 - 2 * 5) = 27 + 14 / 23 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l985_98591


namespace NUMINAMATH_CALUDE_similar_triangle_longest_side_l985_98596

/-- Given a triangle with sides 5, 12, and 13, and a similar triangle with perimeter 150,
    the longest side of the similar triangle is 65. -/
theorem similar_triangle_longest_side : ∀ (a b c : ℝ) (x : ℝ),
  a = 5 → b = 12 → c = 13 →
  a * x + b * x + c * x = 150 →
  max (a * x) (max (b * x) (c * x)) = 65 := by
sorry

end NUMINAMATH_CALUDE_similar_triangle_longest_side_l985_98596


namespace NUMINAMATH_CALUDE_baseball_card_price_l985_98526

/-- Given the following conditions:
  - 2 packs of basketball cards were bought at $3 each
  - 5 decks of baseball cards were bought
  - A $50 bill was used for payment
  - $24 was received in change
  Prove that the price of each baseball card deck is $4 -/
theorem baseball_card_price 
  (basketball_packs : ℕ)
  (basketball_price : ℕ)
  (baseball_decks : ℕ)
  (total_paid : ℕ)
  (change_received : ℕ)
  (h1 : basketball_packs = 2)
  (h2 : basketball_price = 3)
  (h3 : baseball_decks = 5)
  (h4 : total_paid = 50)
  (h5 : change_received = 24) :
  (total_paid - change_received - basketball_packs * basketball_price) / baseball_decks = 4 :=
by sorry

end NUMINAMATH_CALUDE_baseball_card_price_l985_98526


namespace NUMINAMATH_CALUDE_dore_change_l985_98502

/-- The amount of change Mr. Doré receives after his purchase -/
def change (pants_cost shirt_cost tie_cost payment : ℕ) : ℕ :=
  payment - (pants_cost + shirt_cost + tie_cost)

/-- Theorem stating that Mr. Doré receives $2 in change -/
theorem dore_change : change 140 43 15 200 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dore_change_l985_98502


namespace NUMINAMATH_CALUDE_largest_n_for_triangle_inequality_l985_98569

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, 
    such that ∠A + ∠C = 2∠B, the largest positive integer n 
    for which a^n + c^n ≤ 2b^n holds is 4. -/
theorem largest_n_for_triangle_inequality (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  A > 0 → B > 0 → C > 0 → 
  A + B + C = π → 
  A + C = 2 * B → 
  ∃ (n : ℕ), n > 0 ∧ a^n + c^n ≤ 2*b^n ∧ 
  ∀ (m : ℕ), m > n → ¬(a^m + c^m ≤ 2*b^m) → 
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_largest_n_for_triangle_inequality_l985_98569


namespace NUMINAMATH_CALUDE_min_value_theorem_l985_98550

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 5*x*y) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + 3*b = 5*a*b → 3*x + 4*y ≤ 3*a + 4*b ∧ 
  ∃ c d : ℝ, c > 0 ∧ d > 0 ∧ c + 3*d = 5*c*d ∧ 3*c + 4*d = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l985_98550


namespace NUMINAMATH_CALUDE_trig_simplification_l985_98589

theorem trig_simplification (α : Real) (h : π < α ∧ α < 3*π/2) :
  Real.sqrt ((1 + Real.cos α) / (1 - Real.cos α)) + Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = -2 / Real.sin α :=
by sorry

end NUMINAMATH_CALUDE_trig_simplification_l985_98589


namespace NUMINAMATH_CALUDE_james_pays_37_50_l985_98531

/-- Calculates the amount James pays for singing lessons given the specified conditions. -/
def james_payment (total_lessons : ℕ) (lesson_cost : ℚ) (free_lessons : ℕ) (fully_paid_lessons : ℕ) (uncle_contribution : ℚ) : ℚ :=
  let remaining_lessons := total_lessons - free_lessons - fully_paid_lessons
  let partially_paid_lessons := (remaining_lessons + 1) / 2
  let total_paid_lessons := fully_paid_lessons + partially_paid_lessons
  let total_cost := total_paid_lessons * lesson_cost
  total_cost * (1 - uncle_contribution)

/-- Theorem stating that James pays $37.50 for his singing lessons. -/
theorem james_pays_37_50 :
  james_payment 20 5 1 10 (1/2) = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_james_pays_37_50_l985_98531


namespace NUMINAMATH_CALUDE_equation_solution_l985_98576

theorem equation_solution : 
  ∃! x : ℝ, (Real.sqrt (x + 20) - 4 / Real.sqrt (x + 20) = 7) ∧ 
  (x = (114 + 14 * Real.sqrt 65) / 4 - 20) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l985_98576


namespace NUMINAMATH_CALUDE_max_drumming_bunnies_l985_98547

/-- Represents a drum with a specific size -/
structure Drum where
  size : ℕ

/-- Represents a pair of drumsticks with a specific length -/
structure Drumsticks where
  length : ℕ

/-- Represents a bunny with its assigned drum and drumsticks -/
structure Bunny where
  drum : Drum
  sticks : Drumsticks

/-- Determines if a bunny can drum based on its drum and sticks compared to another bunny -/
def canDrum (b1 b2 : Bunny) : Prop :=
  b1.drum.size > b2.drum.size ∧ b1.sticks.length > b2.sticks.length

theorem max_drumming_bunnies 
  (bunnies : Fin 7 → Bunny)
  (h_diff_drums : ∀ i j, i ≠ j → (bunnies i).drum.size ≠ (bunnies j).drum.size)
  (h_diff_sticks : ∀ i j, i ≠ j → (bunnies i).sticks.length ≠ (bunnies j).sticks.length) :
  ∃ (drummers : Finset (Fin 7)),
    drummers.card = 6 ∧
    ∀ i ∈ drummers, ∃ j, canDrum (bunnies i) (bunnies j) :=
by
  sorry

end NUMINAMATH_CALUDE_max_drumming_bunnies_l985_98547


namespace NUMINAMATH_CALUDE_odd_power_sum_divisible_l985_98597

/-- A number is odd if it can be expressed as 2k + 1 for some integer k -/
def IsOdd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

/-- A number is positive if it's greater than zero -/
def IsPositive (n : ℕ) : Prop := n > 0

theorem odd_power_sum_divisible (x y : ℤ) :
  ∀ n : ℕ, IsPositive n → IsOdd n →
  ∃ k : ℤ, x^n + y^n = (x + y) * k :=
sorry

end NUMINAMATH_CALUDE_odd_power_sum_divisible_l985_98597


namespace NUMINAMATH_CALUDE_alternating_sequence_solution_l985_98501

theorem alternating_sequence_solution (n : ℕ) (h : n ≥ 4) :
  ∃! (a : ℕ → ℝ), (∀ i, 1 ≤ i ∧ i ≤ 2*n → a i > 0) ∧
    (∀ k, 0 ≤ k ∧ k < n →
      a (2*k+1) = 1/(a (2*n)) + 1/(a (2*k+2)) ∧
      a (2*k+2) = a (2*k+1) + a (2*k+3)) ∧
    (a (2*n) = a (2*n-1) + a 1) →
  ∀ k, 0 ≤ k ∧ k < n → a (2*k+1) = 1 ∧ a (2*k+2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_alternating_sequence_solution_l985_98501


namespace NUMINAMATH_CALUDE_stating_count_initial_sets_eq_720_l985_98574

/-- The number of letters available (A through J) -/
def num_letters : ℕ := 10

/-- The length of each set of initials -/
def set_length : ℕ := 3

/-- 
Calculates the number of different three-letter sets of initials 
using letters A through J, where no letter can be used more than once in each set.
-/
def count_initial_sets : ℕ :=
  (num_letters) * (num_letters - 1) * (num_letters - 2)

/-- 
Theorem stating that the number of different three-letter sets of initials 
using letters A through J, where no letter can be used more than once in each set, 
is equal to 720.
-/
theorem count_initial_sets_eq_720 : count_initial_sets = 720 := by
  sorry

end NUMINAMATH_CALUDE_stating_count_initial_sets_eq_720_l985_98574


namespace NUMINAMATH_CALUDE_sqrt_product_equals_product_l985_98542

theorem sqrt_product_equals_product : Real.sqrt (9 * 16) = 3 * 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_product_l985_98542


namespace NUMINAMATH_CALUDE_box_volume_problem_l985_98556

theorem box_volume_problem :
  ∃! (x : ℕ), 
    x > 3 ∧ 
    (x + 3) * (x - 3) * (x^2 + 9) < 500 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_problem_l985_98556


namespace NUMINAMATH_CALUDE_sum_of_solutions_prove_sum_of_solutions_l985_98592

theorem sum_of_solutions : ℕ → Prop :=
  fun s => ∃ (S : Finset ℕ), 
    (∀ x ∈ S, (5 * x + 2 > 3 * (x - 1)) ∧ ((1/2) * x - 1 ≤ 7 - (3/2) * x)) ∧
    (∀ x : ℕ, (5 * x + 2 > 3 * (x - 1)) ∧ ((1/2) * x - 1 ≤ 7 - (3/2) * x) → x ∈ S) ∧
    (Finset.sum S id = s) ∧
    s = 10

theorem prove_sum_of_solutions : sum_of_solutions 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_prove_sum_of_solutions_l985_98592


namespace NUMINAMATH_CALUDE_polygon_triangulation_l985_98511

theorem polygon_triangulation (n : ℕ) :
  (n ≥ 3) →  -- Ensure the polygon has at least 3 sides
  (n - 2 = 7) →  -- Number of triangles formed is n - 2, which equals 7
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_polygon_triangulation_l985_98511


namespace NUMINAMATH_CALUDE_exactly_three_sets_sum_to_30_l985_98545

/-- A set of consecutive positive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  length_ge_two : length ≥ 2

/-- The sum of a ConsecutiveSet -/
def sum_consecutive_set (s : ConsecutiveSet) : ℕ :=
  (s.length * (2 * s.start + s.length - 1)) / 2

/-- Predicate for a ConsecutiveSet summing to 30 -/
def sums_to_30 (s : ConsecutiveSet) : Prop :=
  sum_consecutive_set s = 30

theorem exactly_three_sets_sum_to_30 :
  ∃! (sets : Finset ConsecutiveSet), 
    Finset.card sets = 3 ∧ 
    (∀ s ∈ sets, sums_to_30 s) ∧
    (∀ s : ConsecutiveSet, sums_to_30 s → s ∈ sets) :=
sorry

end NUMINAMATH_CALUDE_exactly_three_sets_sum_to_30_l985_98545


namespace NUMINAMATH_CALUDE_constant_c_value_l985_98572

theorem constant_c_value (b c : ℝ) :
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) →
  c = 7 := by
sorry

end NUMINAMATH_CALUDE_constant_c_value_l985_98572


namespace NUMINAMATH_CALUDE_min_four_dollar_frisbees_l985_98540

theorem min_four_dollar_frisbees (total_frisbees : ℕ) (total_receipts : ℕ) : 
  total_frisbees = 64 →
  total_receipts = 196 →
  ∃ (three_dollar : ℕ) (four_dollar : ℕ),
    three_dollar + four_dollar = total_frisbees ∧
    3 * three_dollar + 4 * four_dollar = total_receipts ∧
    ∀ (other_four_dollar : ℕ),
      (∃ (other_three_dollar : ℕ),
        other_three_dollar + other_four_dollar = total_frisbees ∧
        3 * other_three_dollar + 4 * other_four_dollar = total_receipts) →
      four_dollar ≤ other_four_dollar ∧
      four_dollar = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_four_dollar_frisbees_l985_98540


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l985_98541

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l985_98541


namespace NUMINAMATH_CALUDE_right_triangle_sin_c_l985_98544

theorem right_triangle_sin_c (A B C : Real) (h1 : A + B + C = Real.pi)
  (h2 : B = Real.pi / 2) (h3 : Real.sin A = 7 / 25) :
  Real.sin C = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_c_l985_98544


namespace NUMINAMATH_CALUDE_zoo_visitors_l985_98503

theorem zoo_visitors (num_cars : ℝ) (people_per_car : ℝ) :
  num_cars = 3.0 → people_per_car = 63.0 → num_cars * people_per_car = 189.0 := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_l985_98503


namespace NUMINAMATH_CALUDE_chess_team_boys_count_l985_98507

theorem chess_team_boys_count 
  (total_members : ℕ) 
  (meeting_attendance : ℕ) 
  (h1 : total_members = 26)
  (h2 : meeting_attendance = 16)
  : ∃ (boys girls : ℕ),
    boys + girls = total_members ∧
    boys + girls / 2 = meeting_attendance ∧
    boys = 6 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_boys_count_l985_98507


namespace NUMINAMATH_CALUDE_certain_number_l985_98509

theorem certain_number : ∃ x : ℤ, x - 9 = 5 ∧ x = 14 := by sorry

end NUMINAMATH_CALUDE_certain_number_l985_98509


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l985_98568

theorem min_value_sum_of_reciprocals (a b c d e f g : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) 
  (pos_e : 0 < e) (pos_f : 0 < f) (pos_g : 0 < g)
  (sum_eq_8 : a + b + c + d + e + f + g = 8) :
  1/a + 4/b + 9/c + 16/d + 25/e + 36/f + 49/g ≥ 98 ∧ 
  ∃ (a' b' c' d' e' f' g' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 0 < e' ∧ 0 < f' ∧ 0 < g' ∧
    a' + b' + c' + d' + e' + f' + g' = 8 ∧
    1/a' + 4/b' + 9/c' + 16/d' + 25/e' + 36/f' + 49/g' = 98 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l985_98568


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l985_98543

/-- An arithmetic sequence of integers -/
def ArithSeq (a₁ d : ℤ) : ℕ → ℤ
  | 0 => a₁
  | n + 1 => ArithSeq a₁ d n + d

/-- Sum of first n terms of an arithmetic sequence -/
def ArithSeqSum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_first_term (a₁ : ℤ) :
  (∃ d : ℤ, d > 0 ∧
    let S := ArithSeqSum a₁ d 9
    (ArithSeq a₁ d 4) * (ArithSeq a₁ d 17) > S - 4 ∧
    (ArithSeq a₁ d 12) * (ArithSeq a₁ d 9) < S + 60) →
  a₁ ∈ ({-10, -9, -8, -7, -5, -4, -3, -2} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l985_98543


namespace NUMINAMATH_CALUDE_cuboid_volume_l985_98546

theorem cuboid_volume (a b c : ℝ) (h1 : a * b = 2) (h2 : b * c = 6) (h3 : a * c = 9) :
  a * b * c = 6 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_l985_98546


namespace NUMINAMATH_CALUDE_calculate_expression_l985_98551

theorem calculate_expression : -1^2023 + 8 / (-2)^2 - |-4| * 5 = -19 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l985_98551


namespace NUMINAMATH_CALUDE_frank_and_friends_count_l985_98562

/-- The number of people, including Frank, who can eat brownies -/
def num_people (columns rows brownies_per_person : ℕ) : ℕ :=
  (columns * rows) / brownies_per_person

theorem frank_and_friends_count :
  num_people 6 3 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_frank_and_friends_count_l985_98562


namespace NUMINAMATH_CALUDE_may_savings_l985_98513

def savings (month : ℕ) : ℕ :=
  match month with
  | 0 => 20  -- January
  | 1 => 3 * 20  -- February
  | n + 2 => 3 * savings (n + 1) + 50  -- March onwards

theorem may_savings : savings 4 = 2270 := by
  sorry

end NUMINAMATH_CALUDE_may_savings_l985_98513


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l985_98560

-- Define the solution set type
def SolutionSet := Set ℝ

-- Define the given inequality
def givenInequality (k a b c x : ℝ) : Prop :=
  (k / (x + a) + (x + b) / (x + c)) < 0

-- Define the target inequality
def targetInequality (k a b c x : ℝ) : Prop :=
  (k * x / (a * x + 1) + (b * x + 1) / (c * x + 1)) < 0

-- State the theorem
theorem solution_set_equivalence 
  (k a b c : ℝ) 
  (h : SolutionSet = {x | x ∈ (Set.Ioo (-1) (-1/3) ∪ Set.Ioo (1/2) 1) ∧ givenInequality k a b c x}) :
  SolutionSet = {x | x ∈ (Set.Ioo (-3) (-1) ∪ Set.Ioo 1 2) ∧ targetInequality k a b c x} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l985_98560


namespace NUMINAMATH_CALUDE_nearest_integer_to_a_fifth_l985_98580

theorem nearest_integer_to_a_fifth (a b c : ℝ) 
  (h_order : a ≥ b ∧ b ≥ c)
  (h_eq1 : a^2 * b * c + a * b^2 * c + a * b * c^2 + 8 = a + b + c)
  (h_eq2 : a^2 * b + a^2 * c + b^2 * c + b^2 * a + c^2 * a + c^2 * b + 3 * a * b * c = -4)
  (h_eq3 : a^2 * b^2 * c + a * b^2 * c^2 + a^2 * b * c^2 = 2 + a * b + b * c + c * a)
  (h_sum_pos : a + b + c > 0) :
  ∃ (n : ℤ), |n - a^5| < 1/2 ∧ n = 1279 := by
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_a_fifth_l985_98580


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l985_98508

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + y) * f (x - y) = (f x - f y)^2 - 4 * x^2 * f y :=
by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l985_98508


namespace NUMINAMATH_CALUDE_rectangle_in_circle_l985_98552

theorem rectangle_in_circle (d p : ℝ) (h_d_pos : d > 0) (h_p_pos : p > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≥ y ∧
  (2 * x + 2 * y = p) ∧  -- perimeter condition
  (x^2 + y^2 = d^2) ∧    -- inscribed in circle condition
  (x - y = d) :=
sorry

end NUMINAMATH_CALUDE_rectangle_in_circle_l985_98552


namespace NUMINAMATH_CALUDE_citric_acid_molecular_weight_l985_98579

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in a Citric acid molecule -/
def carbon_count : ℕ := 6

/-- The number of Hydrogen atoms in a Citric acid molecule -/
def hydrogen_count : ℕ := 8

/-- The number of Oxygen atoms in a Citric acid molecule -/
def oxygen_count : ℕ := 7

/-- The molecular weight of Citric acid in g/mol -/
def citric_acid_weight : ℝ := 192.124

theorem citric_acid_molecular_weight :
  (carbon_count : ℝ) * carbon_weight +
  (hydrogen_count : ℝ) * hydrogen_weight +
  (oxygen_count : ℝ) * oxygen_weight =
  citric_acid_weight := by sorry

end NUMINAMATH_CALUDE_citric_acid_molecular_weight_l985_98579


namespace NUMINAMATH_CALUDE_original_denominator_proof_l985_98571

theorem original_denominator_proof (d : ℤ) : 
  (2 : ℚ) / d ≠ 0 →
  (5 : ℚ) / (d + 3) = 1 / 3 →
  d = 12 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l985_98571


namespace NUMINAMATH_CALUDE_birds_on_fence_l985_98521

theorem birds_on_fence (initial_storks : ℕ) (additional_storks : ℕ) (total_after : ℕ) 
  (h1 : initial_storks = 4)
  (h2 : additional_storks = 6)
  (h3 : total_after = 13) :
  ∃ initial_birds : ℕ, initial_birds + initial_storks + additional_storks = total_after ∧ initial_birds = 3 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l985_98521


namespace NUMINAMATH_CALUDE_school_ball_purchase_l985_98517

-- Define the unit prices
def soccer_price : ℝ := 40
def basketball_price : ℝ := 60

-- Define the total number of balls and max cost
def total_balls : ℕ := 200
def max_cost : ℝ := 9600

-- Theorem statement
theorem school_ball_purchase :
  -- Condition 1: Basketball price is 20 more than soccer price
  (basketball_price = soccer_price + 20) →
  -- Condition 2: Cost ratio of basketballs to soccer balls
  (6000 / basketball_price = 1.25 * (3200 / soccer_price)) →
  -- Condition 3 and 4 are implicitly used in the conclusion
  -- Conclusion: Correct prices and minimum number of soccer balls
  (soccer_price = 40 ∧ 
   basketball_price = 60 ∧ 
   ∀ m : ℕ, (m : ℝ) * soccer_price + (total_balls - m : ℝ) * basketball_price ≤ max_cost → m ≥ 120) :=
by sorry

end NUMINAMATH_CALUDE_school_ball_purchase_l985_98517


namespace NUMINAMATH_CALUDE_tangent_lines_equal_implies_a_equals_one_l985_98585

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2
def g (a : ℝ) (x : ℝ) : ℝ := 3 * Real.log x - a * x

-- Define the derivatives of f and g
def f' (x : ℝ) : ℝ := 2 * x
def g' (a : ℝ) (x : ℝ) : ℝ := 3 / x - a

-- Theorem statement
theorem tangent_lines_equal_implies_a_equals_one :
  ∃ (x : ℝ), x > 0 ∧ f x = g 1 x ∧ f' x = g' 1 x :=
sorry

end

end NUMINAMATH_CALUDE_tangent_lines_equal_implies_a_equals_one_l985_98585


namespace NUMINAMATH_CALUDE_triangle_side_sum_unbounded_l985_98520

theorem triangle_side_sum_unbounded (b c : ℝ) :
  ∀ ε > 0, ∃ b' c' : ℝ,
    b' > 0 ∧ c' > 0 ∧
    b'^2 + c'^2 + b' * c' = 25 ∧
    b' + c' > ε :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_unbounded_l985_98520


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l985_98500

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup : (Circle × Circle × Circle) :=
  let small_circle : Circle := { center := (0, 0), radius := 2 }
  let large_circle1 : Circle := { center := (-2, 0), radius := 3 }
  let large_circle2 : Circle := { center := (2, 0), radius := 3 }
  (small_circle, large_circle1, large_circle2)

-- Define the shaded area function
noncomputable def shaded_area (setup : Circle × Circle × Circle) : ℝ :=
  2 * Real.pi - 4 * Real.sqrt 5

-- Theorem statement
theorem shaded_area_calculation :
  shaded_area problem_setup = 2 * Real.pi - 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l985_98500


namespace NUMINAMATH_CALUDE_symmetric_points_ab_value_l985_98595

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry across y-axis
def symmetricAcrossYAxis (p q : Point2D) : Prop :=
  p.x = -q.x ∧ p.y = q.y

-- Theorem statement
theorem symmetric_points_ab_value :
  ∀ (a b : ℝ),
  let p : Point2D := ⟨3, -1⟩
  let q : Point2D := ⟨a, 1 - b⟩
  symmetricAcrossYAxis p q →
  a^b = 9 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_ab_value_l985_98595


namespace NUMINAMATH_CALUDE_roundness_of_hundred_billion_l985_98594

/-- Roundness of a positive integer is the sum of exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 100,000,000,000 is 22 -/
theorem roundness_of_hundred_billion : roundness 100000000000 = 22 := by sorry

end NUMINAMATH_CALUDE_roundness_of_hundred_billion_l985_98594


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_zero_l985_98527

theorem quadratic_inequality_implies_zero (a b x y : ℤ) 
  (h1 : a > b^2) 
  (h2 : a^2 * x^2 + 2*a*b * x*y + (b^2 + 1) * y^2 < b^2 + 1) : 
  x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_zero_l985_98527


namespace NUMINAMATH_CALUDE_greatest_product_three_digit_l985_98518

def Digits : Finset Nat := {3, 5, 7, 8, 9}

def is_valid_pair (a b c d e : Nat) : Prop :=
  a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧ e ∈ Digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def three_digit (a b c : Nat) : Nat := 100 * a + 10 * b + c
def two_digit (d e : Nat) : Nat := 10 * d + e

def one_odd_one_even (x y : Nat) : Prop :=
  (x % 2 = 1 ∧ y % 2 = 0) ∨ (x % 2 = 0 ∧ y % 2 = 1)

theorem greatest_product_three_digit :
  ∀ a b c d e,
    is_valid_pair a b c d e →
    one_odd_one_even (three_digit a b c) (two_digit d e) →
    three_digit a b c * two_digit d e ≤ 972 * 85 :=
  sorry

end NUMINAMATH_CALUDE_greatest_product_three_digit_l985_98518


namespace NUMINAMATH_CALUDE_convex_polygon_in_rectangle_l985_98581

/-- A convex polygon in the plane -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  convex : Convex ℝ (convexHull ℝ vertices)
  finite : Finite vertices

/-- The area of a polygon -/
def area (p : ConvexPolygon) : ℝ := sorry

/-- A rectangle in the plane -/
structure Rectangle where
  lower_left : ℝ × ℝ
  upper_right : ℝ × ℝ
  valid : lower_left.1 < upper_right.1 ∧ lower_left.2 < upper_right.2

/-- The area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  (r.upper_right.1 - r.lower_left.1) * (r.upper_right.2 - r.lower_left.2)

/-- A polygon is contained in a rectangle -/
def contained (p : ConvexPolygon) (r : Rectangle) : Prop := sorry

theorem convex_polygon_in_rectangle :
  ∀ (p : ConvexPolygon), area p = 1 →
  ∃ (r : Rectangle), contained p r ∧ rectangleArea r ≤ 2 := by sorry

end NUMINAMATH_CALUDE_convex_polygon_in_rectangle_l985_98581


namespace NUMINAMATH_CALUDE_dhoni_leftover_earnings_l985_98533

/-- Calculates the percentage of earnings left over after Dhoni's expenses --/
theorem dhoni_leftover_earnings (rent : ℝ) (utilities : ℝ) (groceries : ℝ) (transportation : ℝ)
  (h_rent : rent = 25)
  (h_utilities : utilities = 15)
  (h_groceries : groceries = 20)
  (h_transportation : transportation = 12) :
  100 - (rent + (rent - rent * 0.1) + utilities + groceries + transportation) = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_dhoni_leftover_earnings_l985_98533


namespace NUMINAMATH_CALUDE_class_composition_l985_98549

/-- Represents a pair of numbers reported by a student -/
structure ReportedPair :=
  (classmates : ℕ)
  (female_classmates : ℕ)

/-- Checks if a reported pair is valid given the actual numbers of boys and girls -/
def is_valid_report (report : ReportedPair) (boys girls : ℕ) : Prop :=
  (report.classmates = boys + girls - 1 ∧ (report.female_classmates = girls ∨ report.female_classmates = girls + 2 ∨ report.female_classmates = girls - 2)) ∨
  (report.female_classmates = girls ∧ (report.classmates = boys + girls - 1 + 2 ∨ report.classmates = boys + girls - 1 - 2))

theorem class_composition 
  (reports : List ReportedPair)
  (h1 : (12, 18) ∈ reports.map (λ r => (r.classmates, r.female_classmates)))
  (h2 : (15, 15) ∈ reports.map (λ r => (r.classmates, r.female_classmates)))
  (h3 : (11, 15) ∈ reports.map (λ r => (r.classmates, r.female_classmates)))
  (h4 : ∀ r ∈ reports, is_valid_report r 13 16) :
  ∃ (boys girls : ℕ), boys = 13 ∧ girls = 16 ∧ 
    (∀ r ∈ reports, is_valid_report r boys girls) :=
by
  sorry

end NUMINAMATH_CALUDE_class_composition_l985_98549


namespace NUMINAMATH_CALUDE_polly_cooking_time_l985_98567

/-- The number of minutes Polly spends cooking breakfast each day -/
def breakfast_time : ℕ := 20

/-- The number of minutes Polly spends cooking lunch each day -/
def lunch_time : ℕ := 5

/-- The number of minutes Polly spends cooking dinner on 4 days of the week -/
def dinner_time_short : ℕ := 10

/-- The number of minutes Polly spends cooking dinner on the other 3 days of the week -/
def dinner_time_long : ℕ := 30

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of days Polly spends less time cooking dinner -/
def short_dinner_days : ℕ := 4

/-- The number of days Polly spends more time cooking dinner -/
def long_dinner_days : ℕ := days_in_week - short_dinner_days

/-- The total time Polly spends cooking in a week -/
def total_cooking_time : ℕ :=
  breakfast_time * days_in_week +
  lunch_time * days_in_week +
  dinner_time_short * short_dinner_days +
  dinner_time_long * long_dinner_days

/-- Theorem stating that Polly spends 305 minutes cooking in a week -/
theorem polly_cooking_time : total_cooking_time = 305 := by
  sorry

end NUMINAMATH_CALUDE_polly_cooking_time_l985_98567


namespace NUMINAMATH_CALUDE_area_of_rectangle_PQRS_l985_98555

-- Define the Point type
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define the Rectangle type
structure Rectangle :=
  (p : Point) (q : Point) (r : Point) (s : Point)

-- Define the area function for a rectangle
def rectangleArea (rect : Rectangle) : ℝ :=
  let width := abs (rect.q.x - rect.p.x)
  let height := abs (rect.p.y - rect.s.y)
  width * height

-- Theorem statement
theorem area_of_rectangle_PQRS :
  let p := Point.mk (-4) 2
  let q := Point.mk 4 2
  let r := Point.mk 4 (-2)
  let s := Point.mk (-4) (-2)
  let rect := Rectangle.mk p q r s
  rectangleArea rect = 32 := by
  sorry

end NUMINAMATH_CALUDE_area_of_rectangle_PQRS_l985_98555


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l985_98598

theorem perpendicular_lines_slope (a : ℝ) : 
  (∃ (x y : ℝ), y = a * x - 2) ∧ 
  (∃ (x y : ℝ), y = (a + 2) * x + 1) ∧ 
  (a * (a + 2) = -1) → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l985_98598


namespace NUMINAMATH_CALUDE_usable_area_formula_l985_98532

/-- The usable area of a rectangular field with flooded region -/
def usableArea (x : ℝ) : ℝ :=
  (x + 9) * (x + 7) - (2 * x - 2) * (x - 1)

/-- Theorem stating the usable area of the field -/
theorem usable_area_formula (x : ℝ) : 
  usableArea x = -x^2 + 20*x + 61 := by
  sorry

end NUMINAMATH_CALUDE_usable_area_formula_l985_98532


namespace NUMINAMATH_CALUDE_cordelia_bleaching_time_l985_98554

/-- Represents the time in hours for a hair coloring process -/
structure HairColoringTime where
  bleaching : ℝ
  dyeing : ℝ

/-- The properties of Cordelia's hair coloring process -/
def cordelias_hair_coloring (t : HairColoringTime) : Prop :=
  t.bleaching + t.dyeing = 9 ∧ t.dyeing = 2 * t.bleaching

theorem cordelia_bleaching_time :
  ∀ t : HairColoringTime, cordelias_hair_coloring t → t.bleaching = 3 := by
  sorry

end NUMINAMATH_CALUDE_cordelia_bleaching_time_l985_98554


namespace NUMINAMATH_CALUDE_circle_ratio_l985_98505

theorem circle_ratio (r R a c : Real) (hr : r > 0) (hR : R > r) (ha : a > c) (hc : c > 0) :
  π * R^2 = (a - c) * (π * R^2 - π * r^2) →
  R / r = Real.sqrt ((a - c) / (c + 1 - a)) := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l985_98505


namespace NUMINAMATH_CALUDE_composite_quotient_l985_98524

def first_eight_composites : List ℕ := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites : List ℕ := [16, 18, 20, 21, 22, 24, 25, 26]

def product_list (l : List ℕ) : ℕ := l.foldl (·*·) 1

theorem composite_quotient :
  (product_list first_eight_composites) / (product_list next_eight_composites) = 1 / 1430 := by
  sorry

end NUMINAMATH_CALUDE_composite_quotient_l985_98524


namespace NUMINAMATH_CALUDE_larger_number_proof_l985_98530

/-- Given two positive integers with HCF 23 and LCM factors 13 and 16, prove the larger number is 368 -/
theorem larger_number_proof (a b : ℕ) : 
  a > 0 → b > 0 → Nat.gcd a b = 23 → Nat.lcm a b = 23 * 13 * 16 → max a b = 368 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l985_98530


namespace NUMINAMATH_CALUDE_max_sum_at_vertex_l985_98525

/-- Represents a face of the cube -/
structure Face :=
  (number : ℕ)

/-- Represents a cube with six numbered faces -/
structure Cube :=
  (faces : Fin 6 → Face)
  (opposite_sum : ∀ i : Fin 3, (faces i).number + (faces (i + 3)).number = 10)

/-- Represents a vertex of the cube -/
structure Vertex :=
  (face1 : Face)
  (face2 : Face)
  (face3 : Face)

/-- The theorem stating the maximum sum at a vertex -/
theorem max_sum_at_vertex (c : Cube) : 
  (∃ v : Vertex, v.face1 ∈ Set.range c.faces ∧ 
                 v.face2 ∈ Set.range c.faces ∧ 
                 v.face3 ∈ Set.range c.faces ∧ 
                 v.face1 ≠ v.face2 ∧ v.face2 ≠ v.face3 ∧ v.face1 ≠ v.face3) →
  (∀ v : Vertex, v.face1 ∈ Set.range c.faces ∧ 
                v.face2 ∈ Set.range c.faces ∧ 
                v.face3 ∈ Set.range c.faces ∧ 
                v.face1 ≠ v.face2 ∧ v.face2 ≠ v.face3 ∧ v.face1 ≠ v.face3 →
                v.face1.number + v.face2.number + v.face3.number ≤ 22) :=
sorry

end NUMINAMATH_CALUDE_max_sum_at_vertex_l985_98525


namespace NUMINAMATH_CALUDE_males_in_band_only_l985_98583

/-- Represents the number of students in various musical groups and their intersections --/
structure MusicGroups where
  band_male : ℕ
  band_female : ℕ
  orchestra_male : ℕ
  orchestra_female : ℕ
  choir_male : ℕ
  choir_female : ℕ
  band_orchestra_male : ℕ
  band_orchestra_female : ℕ
  band_choir_male : ℕ
  band_choir_female : ℕ
  orchestra_choir_male : ℕ
  orchestra_choir_female : ℕ
  total_students : ℕ

/-- Theorem stating the number of males in the band who are not in the orchestra or choir --/
theorem males_in_band_only (g : MusicGroups)
  (h1 : g.band_male = 120)
  (h2 : g.band_female = 100)
  (h3 : g.orchestra_male = 90)
  (h4 : g.orchestra_female = 130)
  (h5 : g.choir_male = 40)
  (h6 : g.choir_female = 60)
  (h7 : g.band_orchestra_male = 50)
  (h8 : g.band_orchestra_female = 70)
  (h9 : g.band_choir_male = 30)
  (h10 : g.band_choir_female = 40)
  (h11 : g.orchestra_choir_male = 20)
  (h12 : g.orchestra_choir_female = 30)
  (h13 : g.total_students = 260) :
  g.band_male - (g.band_orchestra_male + g.band_choir_male - 20) = 60 := by
  sorry

end NUMINAMATH_CALUDE_males_in_band_only_l985_98583


namespace NUMINAMATH_CALUDE_tree_height_after_four_months_l985_98536

/-- Calculates the height of a tree after a given number of months -/
def tree_height (initial_height : ℕ) (growth_rate : ℕ) (growth_period : ℕ) (months : ℕ) : ℕ :=
  initial_height * 100 + (months * 4 / growth_period) * growth_rate

/-- Theorem stating that a tree with given growth parameters reaches 600 cm after 4 months -/
theorem tree_height_after_four_months :
  tree_height 2 50 2 4 = 600 := by
  sorry

#eval tree_height 2 50 2 4

end NUMINAMATH_CALUDE_tree_height_after_four_months_l985_98536


namespace NUMINAMATH_CALUDE_smallest_repeating_block_7_11_l985_98561

/-- The length of the smallest repeating block in the decimal expansion of 7/11 -/
def repeating_block_length_7_11 : ℕ := 2

/-- The fraction we're considering -/
def fraction : ℚ := 7 / 11

theorem smallest_repeating_block_7_11 :
  repeating_block_length_7_11 = 2 ∧
  ∃ (a b : ℕ), fraction = (a : ℚ) / (10^repeating_block_length_7_11 - 1 : ℚ) +
                (b : ℚ) / (10^repeating_block_length_7_11 : ℚ) ∧
                0 ≤ a ∧ a < 10^repeating_block_length_7_11 - 1 ∧
                0 ≤ b ∧ b < 10^repeating_block_length_7_11 ∧
                ∀ (n : ℕ), n < repeating_block_length_7_11 →
                  ¬∃ (c d : ℕ), fraction = (c : ℚ) / (10^n - 1 : ℚ) +
                                (d : ℚ) / (10^n : ℚ) ∧
                                0 ≤ c ∧ c < 10^n - 1 ∧
                                0 ≤ d ∧ d < 10^n := by
  sorry

#eval repeating_block_length_7_11

end NUMINAMATH_CALUDE_smallest_repeating_block_7_11_l985_98561


namespace NUMINAMATH_CALUDE_possible_values_l985_98510

def Rectangle := Fin 3 → Fin 4 → ℕ

def valid_rectangle (r : Rectangle) : Prop :=
  (∀ i j, r i j ∈ Finset.range 13) ∧
  (∀ i j k, i ≠ j → r i k ≠ r j k) ∧
  (∀ k, r 0 k + r 1 k = 2 * r 2 k) ∧
  (r 0 0 = 6 ∧ r 1 0 = 4 ∧ r 2 1 = 8 ∧ r 2 2 = 11)

theorem possible_values (r : Rectangle) (h : valid_rectangle r) :
  r 2 3 = 2 ∨ r 2 3 = 11 :=
sorry

end NUMINAMATH_CALUDE_possible_values_l985_98510


namespace NUMINAMATH_CALUDE_square_of_difference_l985_98535

theorem square_of_difference (y : ℝ) (h : y^2 ≥ 49) :
  (7 - Real.sqrt (y^2 - 49))^2 = y^2 - 14 * Real.sqrt (y^2 - 49) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l985_98535


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l985_98515

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = Complex.abs (1 - Complex.I) + Complex.I) : 
  z.im = (1 - Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l985_98515


namespace NUMINAMATH_CALUDE_line_l_is_correct_l985_98506

-- Define the given line
def given_line (x y : ℝ) : Prop := 3 * x + 4 * y - 3 = 0

-- Define the point A
def point_A : ℝ × ℝ := (-2, -3)

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := 4 * x - 3 * y - 1 = 0

-- Theorem statement
theorem line_l_is_correct :
  (∀ x y : ℝ, line_l x y → (x, y) = point_A ∨ (x, y) ≠ point_A) ∧
  (∀ x y : ℝ, line_l x y → given_line x y → False) ∧
  line_l point_A.1 point_A.2 :=
sorry

end NUMINAMATH_CALUDE_line_l_is_correct_l985_98506


namespace NUMINAMATH_CALUDE_number_of_selection_schemes_l985_98548

/-- The number of male teachers -/
def num_male : ℕ := 5

/-- The number of female teachers -/
def num_female : ℕ := 4

/-- The total number of teachers -/
def total_teachers : ℕ := num_male + num_female

/-- The number of teachers to be selected -/
def teachers_to_select : ℕ := 3

/-- Calculates the number of permutations of k elements from n elements -/
def permutations (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / Nat.factorial (n - k)

/-- The theorem stating the number of valid selection schemes -/
theorem number_of_selection_schemes : 
  permutations total_teachers teachers_to_select - 
  (permutations num_male teachers_to_select + 
   permutations num_female teachers_to_select) = 420 := by
  sorry

end NUMINAMATH_CALUDE_number_of_selection_schemes_l985_98548


namespace NUMINAMATH_CALUDE_square_area_from_rectangle_l985_98578

theorem square_area_from_rectangle (s r l b : ℝ) : 
  r = s →                  -- radius of circle equals side of square
  l = (2 / 5) * r →        -- length of rectangle is two-fifths of radius
  b = 10 →                 -- breadth of rectangle is 10 units
  l * b = 120 →            -- area of rectangle is 120 sq. units
  s^2 = 900 :=             -- area of square is 900 sq. units
by sorry

end NUMINAMATH_CALUDE_square_area_from_rectangle_l985_98578


namespace NUMINAMATH_CALUDE_tangent_slope_minimum_tangent_slope_minimum_achieved_l985_98573

theorem tangent_slope_minimum (b : ℝ) (h : b > 0) : 
  (2 / b + b) ≥ 2 * Real.sqrt 2 :=
by sorry

theorem tangent_slope_minimum_achieved (b : ℝ) (h : b > 0) : 
  (2 / b + b = 2 * Real.sqrt 2) ↔ (2 / b = b) :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_minimum_tangent_slope_minimum_achieved_l985_98573
