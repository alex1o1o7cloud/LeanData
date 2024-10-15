import Mathlib

namespace NUMINAMATH_CALUDE_car_purchase_cost_difference_l1245_124533

/-- Calculates the difference in individual cost for buying a car when the group size changes --/
theorem car_purchase_cost_difference 
  (base_cost : ℕ) 
  (discount_per_person : ℕ) 
  (car_wash_earnings : ℕ) 
  (original_group_size : ℕ) 
  (new_group_size : ℕ) : 
  base_cost = 1700 →
  discount_per_person = 50 →
  car_wash_earnings = 500 →
  original_group_size = 6 →
  new_group_size = 5 →
  (base_cost - new_group_size * discount_per_person - car_wash_earnings) / new_group_size -
  (base_cost - original_group_size * discount_per_person - car_wash_earnings) / original_group_size = 40 := by
  sorry


end NUMINAMATH_CALUDE_car_purchase_cost_difference_l1245_124533


namespace NUMINAMATH_CALUDE_largest_cosine_in_geometric_triangle_l1245_124578

/-- Given a triangle ABC where its sides form a geometric sequence with common ratio √2,
    the largest cosine value of its angles is -√2/4 -/
theorem largest_cosine_in_geometric_triangle :
  ∀ (a b c : ℝ),
  a > 0 →
  b = a * Real.sqrt 2 →
  c = b * Real.sqrt 2 →
  let cosA := (b^2 + c^2 - a^2) / (2*b*c)
  let cosB := (a^2 + c^2 - b^2) / (2*a*c)
  let cosC := (a^2 + b^2 - c^2) / (2*a*b)
  max cosA (max cosB cosC) = -(Real.sqrt 2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_cosine_in_geometric_triangle_l1245_124578


namespace NUMINAMATH_CALUDE_prob_odd_sum_is_two_thirds_l1245_124574

/-- A type representing the cards labeled 0, 1, and 2 -/
inductive Card : Type
  | zero : Card
  | one : Card
  | two : Card

/-- A function to convert a Card to its numerical value -/
def cardValue : Card → ℕ
  | Card.zero => 0
  | Card.one => 1
  | Card.two => 2

/-- A predicate to check if the sum of two cards is odd -/
def isSumOdd (c1 c2 : Card) : Prop :=
  Odd (cardValue c1 + cardValue c2)

/-- The set of all possible card combinations -/
def allCombinations : Finset (Card × Card) :=
  sorry

/-- The set of card combinations with odd sum -/
def oddSumCombinations : Finset (Card × Card) :=
  sorry

/-- Theorem stating the probability of drawing two cards with odd sum is 2/3 -/
theorem prob_odd_sum_is_two_thirds :
    (Finset.card oddSumCombinations : ℚ) / (Finset.card allCombinations : ℚ) = 2 / 3 :=
  sorry

end NUMINAMATH_CALUDE_prob_odd_sum_is_two_thirds_l1245_124574


namespace NUMINAMATH_CALUDE_tree_height_proof_l1245_124547

/-- Proves that a tree with a current height of 180 inches, which is 50% taller than its original height, had an original height of 10 feet. -/
theorem tree_height_proof (current_height : ℝ) (growth_factor : ℝ) (inches_per_foot : ℝ) :
  current_height = 180 ∧
  growth_factor = 1.5 ∧
  inches_per_foot = 12 →
  current_height / growth_factor / inches_per_foot = 10 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_proof_l1245_124547


namespace NUMINAMATH_CALUDE_total_study_hours_is_three_l1245_124595

-- Define the time spent on each subject in minutes
def science_time : ℕ := 60
def math_time : ℕ := 80
def literature_time : ℕ := 40

-- Define the total study time in minutes
def total_study_time : ℕ := science_time + math_time + literature_time

-- Define the conversion factor from minutes to hours
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem total_study_hours_is_three :
  (total_study_time : ℚ) / minutes_per_hour = 3 := by sorry

end NUMINAMATH_CALUDE_total_study_hours_is_three_l1245_124595


namespace NUMINAMATH_CALUDE_power_two_greater_than_square_l1245_124540

theorem power_two_greater_than_square (n : ℕ) (h : n > 5) : 2^n > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_square_l1245_124540


namespace NUMINAMATH_CALUDE_quadratic_root_sum_product_l1245_124508

theorem quadratic_root_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 8 ∧ x * y = 12) → 
  p + q = 60 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_product_l1245_124508


namespace NUMINAMATH_CALUDE_elevator_descent_time_l1245_124599

/-- Represents the elevator descent problem -/
def elevator_descent (total_floors : ℕ) 
  (first_half_time : ℕ) 
  (mid_floor_time : ℕ) 
  (final_floor_time : ℕ) : Prop :=
  let first_half := total_floors / 2
  let mid_section := 5
  let final_section := 5
  let total_time := first_half_time + mid_floor_time * mid_section + final_floor_time * final_section
  total_floors = 20 ∧ 
  first_half_time = 15 ∧ 
  mid_floor_time = 5 ∧ 
  final_floor_time = 16 ∧ 
  total_time / 60 = 2

/-- Theorem stating that the elevator descent takes 2 hours -/
theorem elevator_descent_time : 
  elevator_descent 20 15 5 16 := by sorry

end NUMINAMATH_CALUDE_elevator_descent_time_l1245_124599


namespace NUMINAMATH_CALUDE_negative_cube_inequality_l1245_124546

theorem negative_cube_inequality (a : ℝ) (h : a < 0) : a^3 ≠ (-a)^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_inequality_l1245_124546


namespace NUMINAMATH_CALUDE_no_primes_satisfying_congruence_l1245_124589

theorem no_primes_satisfying_congruence : 
  ¬ ∃ (p : ℕ) (hp : Nat.Prime p) (r s : ℤ),
    (∀ (x : ℤ), (x^3 - x + 2) % p = ((x - r)^2 * (x - s)) % p) ∧
    (∀ (r' s' : ℤ), (∀ (x : ℤ), (x^3 - x + 2) % p = ((x - r')^2 * (x - s')) % p) → r' = r ∧ s' = s) :=
by sorry


end NUMINAMATH_CALUDE_no_primes_satisfying_congruence_l1245_124589


namespace NUMINAMATH_CALUDE_frosting_calculation_l1245_124586

/-- Calculates the total number of frosting cans needed for a bakery order -/
theorem frosting_calculation (layer_cake_frosting : ℝ) (single_item_frosting : ℝ) 
  (tiered_cake_frosting : ℝ) (mini_cupcake_pair_frosting : ℝ) 
  (layer_cakes : ℕ) (tiered_cakes : ℕ) (cupcake_dozens : ℕ) 
  (mini_cupcakes : ℕ) (single_cakes : ℕ) (brownie_pans : ℕ) :
  layer_cake_frosting = 1 →
  single_item_frosting = 0.5 →
  tiered_cake_frosting = 1.5 →
  mini_cupcake_pair_frosting = 0.25 →
  layer_cakes = 4 →
  tiered_cakes = 8 →
  cupcake_dozens = 10 →
  mini_cupcakes = 30 →
  single_cakes = 15 →
  brownie_pans = 24 →
  layer_cakes * layer_cake_frosting +
  tiered_cakes * tiered_cake_frosting +
  cupcake_dozens * single_item_frosting +
  (mini_cupcakes / 2) * mini_cupcake_pair_frosting +
  single_cakes * single_item_frosting +
  brownie_pans * single_item_frosting = 44.25 := by
sorry

end NUMINAMATH_CALUDE_frosting_calculation_l1245_124586


namespace NUMINAMATH_CALUDE_max_value_of_C_l1245_124554

theorem max_value_of_C (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let a := 1 / y
  let b := y + 1 / x
  let C := min x (min a b)
  ∀ ε > 0, C ≤ Real.sqrt 2 + ε ∧ ∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧
    let a' := 1 / y'
    let b' := y' + 1 / x'
    let C' := min x' (min a' b')
    C' > Real.sqrt 2 - ε :=
sorry

end NUMINAMATH_CALUDE_max_value_of_C_l1245_124554


namespace NUMINAMATH_CALUDE_plane_line_relations_l1245_124565

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (lineparallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Axioms
axiom parallel_trans {α β γ : Plane} : parallel α β → parallel β γ → parallel α γ
axiom perpendicular_trans {l m : Line} {α β : Plane} : 
  perpendicular l α → perpendicular l β → perpendicular m α → perpendicular m β

-- Theorem
theorem plane_line_relations 
  (α β : Plane) (m n : Line) 
  (h_diff_planes : α ≠ β) 
  (h_diff_lines : m ≠ n) :
  (parallel α β ∧ contains α m → lineparallel m β) ∧
  (perpendicular n α ∧ perpendicular n β ∧ perpendicular m α → perpendicular m β) :=
by sorry

end NUMINAMATH_CALUDE_plane_line_relations_l1245_124565


namespace NUMINAMATH_CALUDE_stadium_area_calculation_l1245_124557

/-- Calculates the total surface area of a rectangular stadium in square feet,
    given its dimensions in yards. -/
def stadium_surface_area (length_yd width_yd height_yd : ℕ) : ℕ :=
  let length := length_yd * 3
  let width := width_yd * 3
  let height := height_yd * 3
  2 * (length * width + length * height + width * height)

/-- Theorem stating that the surface area of a stadium with given dimensions is 110,968 sq ft. -/
theorem stadium_area_calculation :
  stadium_surface_area 62 48 30 = 110968 := by
  sorry

#eval stadium_surface_area 62 48 30

end NUMINAMATH_CALUDE_stadium_area_calculation_l1245_124557


namespace NUMINAMATH_CALUDE_remainder_theorem_l1245_124541

theorem remainder_theorem (P D Q R D' Q' R' : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = 2 * D' * Q' + R') : 
  P % (2 * D * D') = D * R' + R := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1245_124541


namespace NUMINAMATH_CALUDE_prob_three_spades_l1245_124524

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- A standard deck of 52 cards with 4 suits and 13 cards per suit -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    h_total := rfl }

/-- The probability of drawing three spades in a row from a standard deck -/
theorem prob_three_spades (d : Deck) (h : d = standard_deck) :
  (d.cards_per_suit : ℚ) / d.total_cards *
  (d.cards_per_suit - 1) / (d.total_cards - 1) *
  (d.cards_per_suit - 2) / (d.total_cards - 2) = 33 / 2550 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_spades_l1245_124524


namespace NUMINAMATH_CALUDE_two_digit_divisible_by_72_l1245_124518

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def six_digit_number (x y : ℕ) : ℕ := 640000 + x * 1000 + 720 + y

theorem two_digit_divisible_by_72 :
  ∀ x y : ℕ, is_two_digit (10 * x + y) →
  (six_digit_number x y ∣ 72) →
  (x = 8 ∧ y = 0) ∨ (x = 9 ∧ y = 8) :=
sorry

end NUMINAMATH_CALUDE_two_digit_divisible_by_72_l1245_124518


namespace NUMINAMATH_CALUDE_reduced_price_is_30_l1245_124597

/-- Represents the price reduction of oil as a percentage -/
def price_reduction : ℝ := 0.20

/-- Represents the additional amount of oil that can be purchased after the price reduction -/
def additional_oil : ℝ := 10

/-- Represents the total cost in Rupees -/
def total_cost : ℝ := 1500

/-- Calculates the reduced price per kg of oil -/
def reduced_price_per_kg (original_price : ℝ) : ℝ :=
  original_price * (1 - price_reduction)

/-- Theorem stating that the reduced price per kg of oil is 30 Rupees -/
theorem reduced_price_is_30 :
  ∃ (original_price : ℝ) (original_quantity : ℝ),
    original_quantity > 0 ∧
    original_price > 0 ∧
    original_quantity * original_price = total_cost ∧
    (original_quantity + additional_oil) * reduced_price_per_kg original_price = total_cost ∧
    reduced_price_per_kg original_price = 30 :=
  sorry

end NUMINAMATH_CALUDE_reduced_price_is_30_l1245_124597


namespace NUMINAMATH_CALUDE_remainder_theorem_l1245_124558

theorem remainder_theorem (n : ℤ) (h : ∃ k : ℤ, n = 50 * k - 1) :
  (n^2 + 2*n + 3) % 50 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1245_124558


namespace NUMINAMATH_CALUDE_passion_fruit_crates_l1245_124572

theorem passion_fruit_crates (total grapes mangoes : ℕ) 
  (h1 : total = 50)
  (h2 : grapes = 13)
  (h3 : mangoes = 20) :
  total - (grapes + mangoes) = 17 := by
  sorry

end NUMINAMATH_CALUDE_passion_fruit_crates_l1245_124572


namespace NUMINAMATH_CALUDE_water_depth_of_specific_tower_l1245_124515

/-- Represents a conical tower -/
structure ConicalTower where
  height : ℝ
  volumeAboveWater : ℝ

/-- Calculates the depth of water at the base of a conical tower -/
def waterDepth (tower : ConicalTower) : ℝ :=
  tower.height * (1 - (tower.volumeAboveWater)^(1/3))

/-- The theorem stating the depth of water for a specific conical tower -/
theorem water_depth_of_specific_tower :
  let tower : ConicalTower := ⟨10000, 1/4⟩
  waterDepth tower = 905 := by sorry

end NUMINAMATH_CALUDE_water_depth_of_specific_tower_l1245_124515


namespace NUMINAMATH_CALUDE_min_distance_sum_l1245_124561

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 9

-- Define the theorem
theorem min_distance_sum (M N P : ℝ × ℝ) :
  C₁ M.1 M.2 →
  C₂ N.1 N.2 →
  P.2 = 0 →
  ∃ (M' N' P' : ℝ × ℝ),
    C₁ M'.1 M'.2 ∧
    C₂ N'.1 N'.2 ∧
    P'.2 = 0 ∧
    Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2) +
    Real.sqrt ((N.1 - P.1)^2 + (N.2 - P.2)^2) ≥
    Real.sqrt ((M'.1 - P'.1)^2 + (M'.2 - P'.2)^2) +
    Real.sqrt ((N'.1 - P'.1)^2 + (N'.2 - P'.2)^2) ∧
    Real.sqrt ((M'.1 - P'.1)^2 + (M'.2 - P'.2)^2) +
    Real.sqrt ((N'.1 - P'.1)^2 + (N'.2 - P'.2)^2) =
    5 * Real.sqrt 2 - 4 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_l1245_124561


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1245_124538

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1245_124538


namespace NUMINAMATH_CALUDE_add_12345_seconds_to_5_15_00_l1245_124504

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

theorem add_12345_seconds_to_5_15_00 :
  addSeconds (Time.mk 5 15 0) 12345 = Time.mk 9 0 45 := by
  sorry

end NUMINAMATH_CALUDE_add_12345_seconds_to_5_15_00_l1245_124504


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l1245_124596

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number is the start of six consecutive non-primes
def isSixConsecutiveNonPrimes (n : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ n → k < n + 6 → ¬(isPrime k)

-- Theorem statement
theorem smallest_prime_after_six_nonprimes :
  ∃ p : ℕ, isPrime p ∧ 
    (∃ n : ℕ, isSixConsecutiveNonPrimes n ∧ p = n + 6) ∧
    (∀ q : ℕ, q < p → ¬(∃ m : ℕ, isSixConsecutiveNonPrimes m ∧ isPrime (m + 6) ∧ q = m + 6)) :=
  sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l1245_124596


namespace NUMINAMATH_CALUDE_larger_cylinder_height_l1245_124590

/-- The height of a larger cylinder given specific conditions -/
theorem larger_cylinder_height : 
  ∀ (d_large h_small r_small : ℝ) (n : ℕ),
  d_large = 6 →
  h_small = 5 →
  r_small = 2 →
  n = 3 →
  ∃ (h_large : ℝ),
    h_large = 20 / 3 ∧
    π * (d_large / 2)^2 * h_large = n * π * r_small^2 * h_small :=
by sorry

end NUMINAMATH_CALUDE_larger_cylinder_height_l1245_124590


namespace NUMINAMATH_CALUDE_lines_intersection_l1245_124549

def line1 (t : ℝ) : ℝ × ℝ := (1 + 2*t, 2 - 3*t)
def line2 (u : ℝ) : ℝ × ℝ := (-1 + 3*u, 4 + u)

theorem lines_intersection :
  ∃! p : ℝ × ℝ, (∃ t : ℝ, line1 t = p) ∧ (∃ u : ℝ, line2 u = p) :=
  by
    use (-5/11, 46/11)
    sorry

#check lines_intersection

end NUMINAMATH_CALUDE_lines_intersection_l1245_124549


namespace NUMINAMATH_CALUDE_equal_chord_circle_exists_l1245_124522

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The incenter of a triangle --/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The length of a chord formed by the intersection of a circle and a line segment --/
def chordLength (c : Circle) (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: For any triangle, there exists a circle that cuts chords of equal length from its sides --/
theorem equal_chord_circle_exists (t : Triangle) : 
  ∃ (c : Circle), 
    chordLength c t.A t.B = chordLength c t.B t.C ∧ 
    chordLength c t.B t.C = chordLength c t.C t.A := by
  sorry

end NUMINAMATH_CALUDE_equal_chord_circle_exists_l1245_124522


namespace NUMINAMATH_CALUDE_sum_remainder_mod_11_l1245_124516

theorem sum_remainder_mod_11 : (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_11_l1245_124516


namespace NUMINAMATH_CALUDE_count_false_propositions_l1245_124511

-- Define the original proposition
def original_prop (a : ℝ) : Prop := a > 1 → a > 2

-- Define the inverse proposition
def inverse_prop (a : ℝ) : Prop := ¬(a > 1) → ¬(a > 2)

-- Define the negation proposition
def negation_prop (a : ℝ) : Prop := ¬(a > 1 → a > 2)

-- Define the converse proposition
def converse_prop (a : ℝ) : Prop := a > 2 → a > 1

-- Count the number of false propositions
def count_false_props : ℕ := 2

-- Theorem statement
theorem count_false_propositions :
  count_false_props = 2 :=
sorry

end NUMINAMATH_CALUDE_count_false_propositions_l1245_124511


namespace NUMINAMATH_CALUDE_at_least_four_2x2_squares_sum_greater_than_100_l1245_124532

/-- Represents a square on the 8x8 board -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the board configuration -/
def Board := Square → Fin 64

/-- Checks if a given 2x2 square has a sum greater than 100 -/
def is_sum_greater_than_100 (board : Board) (top_left : Square) : Prop :=
  let sum := (board top_left).val + 
              (board ⟨top_left.row, top_left.col.succ⟩).val + 
              (board ⟨top_left.row.succ, top_left.col⟩).val + 
              (board ⟨top_left.row.succ, top_left.col.succ⟩).val
  sum > 100

/-- The main theorem to be proved -/
theorem at_least_four_2x2_squares_sum_greater_than_100 (board : Board) 
  (h_unique : ∀ (s1 s2 : Square), board s1 = board s2 → s1 = s2) :
  ∃ (s1 s2 s3 s4 : Square), 
    s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4 ∧
    is_sum_greater_than_100 board s1 ∧
    is_sum_greater_than_100 board s2 ∧
    is_sum_greater_than_100 board s3 ∧
    is_sum_greater_than_100 board s4 :=
  sorry

end NUMINAMATH_CALUDE_at_least_four_2x2_squares_sum_greater_than_100_l1245_124532


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1245_124571

theorem negation_of_universal_proposition :
  (¬ (∀ a : ℝ, a > 0 → Real.exp a ≥ 1)) ↔ (∃ a : ℝ, a > 0 ∧ Real.exp a < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1245_124571


namespace NUMINAMATH_CALUDE_one_two_five_th_number_l1245_124525

def digit_sum (n : ℕ) : ℕ := sorry

def nth_number_with_digit_sum_5 (n : ℕ) : ℕ := sorry

theorem one_two_five_th_number : nth_number_with_digit_sum_5 125 = 41000 := by sorry

end NUMINAMATH_CALUDE_one_two_five_th_number_l1245_124525


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1245_124531

theorem contrapositive_equivalence (a b m : ℝ) :
  (¬(a > b → a * (m^2 + 1) > b * (m^2 + 1))) ↔ (a * (m^2 + 1) ≤ b * (m^2 + 1) → a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1245_124531


namespace NUMINAMATH_CALUDE_tax_rate_on_other_items_l1245_124521

-- Define the total amount spent (excluding taxes)
def total_amount : ℝ := 100

-- Define the percentages spent on each category
def clothing_percent : ℝ := 0.5
def food_percent : ℝ := 0.25
def other_percent : ℝ := 0.25

-- Define the tax rates
def clothing_tax_rate : ℝ := 0.1
def food_tax_rate : ℝ := 0
def total_tax_rate : ℝ := 0.1

-- Define the amounts spent on each category
def clothing_amount : ℝ := total_amount * clothing_percent
def food_amount : ℝ := total_amount * food_percent
def other_amount : ℝ := total_amount * other_percent

-- Define the tax paid on clothing
def clothing_tax : ℝ := clothing_amount * clothing_tax_rate

-- Define the total tax paid
def total_tax : ℝ := total_amount * total_tax_rate

-- Define the tax paid on other items
def other_tax : ℝ := total_tax - clothing_tax

-- Theorem to prove
theorem tax_rate_on_other_items :
  other_tax / other_amount = 0.2 := by sorry

end NUMINAMATH_CALUDE_tax_rate_on_other_items_l1245_124521


namespace NUMINAMATH_CALUDE_number_calculations_l1245_124566

/-- The number that is 17 more than 5 times X -/
def number_more_than_5x (x : ℝ) : ℝ := 5 * x + 17

/-- The number that is less than 5 times 22 by Y -/
def number_less_than_5_times_22 (y : ℝ) : ℝ := 22 * 5 - y

theorem number_calculations (x y : ℝ) : 
  (number_more_than_5x x = 5 * x + 17) ∧ 
  (number_less_than_5_times_22 y = 22 * 5 - y) :=
by sorry

end NUMINAMATH_CALUDE_number_calculations_l1245_124566


namespace NUMINAMATH_CALUDE_xyz_value_l1245_124536

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (eq4 : x * y + x * z + y * z = 9)
  (eq5 : x + y + z = 6) :
  x * y * z = 14 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1245_124536


namespace NUMINAMATH_CALUDE_target_annual_revenue_l1245_124543

/-- Calculates the target annual revenue for a shoe company given their current monthly sales and required monthly increase. -/
theorem target_annual_revenue
  (current_monthly_sales : ℕ)
  (required_monthly_increase : ℕ)
  (months_per_year : ℕ)
  (h1 : current_monthly_sales = 4000)
  (h2 : required_monthly_increase = 1000)
  (h3 : months_per_year = 12) :
  (current_monthly_sales + required_monthly_increase) * months_per_year = 60000 :=
by
  sorry

#check target_annual_revenue

end NUMINAMATH_CALUDE_target_annual_revenue_l1245_124543


namespace NUMINAMATH_CALUDE_competition_max_robot_weight_l1245_124552

/-- The weight of the standard robot in pounds -/
def standard_robot_weight : ℝ := 100

/-- The weight of the battery in pounds -/
def battery_weight : ℝ := 20

/-- The minimum additional weight above the standard robot in pounds -/
def min_additional_weight : ℝ := 5

/-- The maximum weight multiplier -/
def max_weight_multiplier : ℝ := 2

/-- The maximum weight of a robot in the competition, including the battery -/
def max_robot_weight : ℝ := 250

theorem competition_max_robot_weight :
  max_robot_weight = 
    max_weight_multiplier * (standard_robot_weight + min_additional_weight + battery_weight) :=
by sorry

end NUMINAMATH_CALUDE_competition_max_robot_weight_l1245_124552


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1245_124593

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 56 * x + 49 = (4 * x - 7)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1245_124593


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1245_124505

theorem imaginary_part_of_complex_fraction : Complex.im (2 * Complex.I / (2 + Complex.I ^ 3)) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1245_124505


namespace NUMINAMATH_CALUDE_tenth_term_is_one_over_120_l1245_124555

def a (n : ℕ+) : ℚ := 1 / (n * (n + 2))

theorem tenth_term_is_one_over_120 : a 10 = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_one_over_120_l1245_124555


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1245_124500

theorem imaginary_part_of_z : Complex.im (((1 : ℂ) - Complex.I) / (2 * Complex.I)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1245_124500


namespace NUMINAMATH_CALUDE_orchids_cut_count_l1245_124579

/-- Represents the number of flowers in the vase -/
structure FlowerCount where
  roses : ℕ
  orchids : ℕ

/-- Represents the ratio of cut flowers -/
structure CutRatio where
  roses : ℕ
  orchids : ℕ

def initial_count : FlowerCount := { roses := 16, orchids := 3 }
def final_count : FlowerCount := { roses := 31, orchids := 7 }
def cut_ratio : CutRatio := { roses := 5, orchids := 3 }

theorem orchids_cut_count (initial : FlowerCount) (final : FlowerCount) (ratio : CutRatio) :
  final.orchids - initial.orchids = 4 :=
sorry

end NUMINAMATH_CALUDE_orchids_cut_count_l1245_124579


namespace NUMINAMATH_CALUDE_acute_triangle_contains_grid_point_l1245_124542

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a triangle on a 2D grid -/
structure GridTriangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint

/-- Checks if a triangle is acute -/
def isAcute (t : GridTriangle) : Prop := sorry

/-- Checks if a point is inside or on the sides of a triangle -/
def isInsideOrOnSides (p : GridPoint) (t : GridTriangle) : Prop := sorry

/-- Main theorem: If a triangle on a grid is acute, there exists a grid point 
    (other than its vertices) inside or on its sides -/
theorem acute_triangle_contains_grid_point (t : GridTriangle) :
  isAcute t → ∃ p : GridPoint, p ≠ t.A ∧ p ≠ t.B ∧ p ≠ t.C ∧ isInsideOrOnSides p t := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_contains_grid_point_l1245_124542


namespace NUMINAMATH_CALUDE_third_root_of_cubic_l1245_124517

theorem third_root_of_cubic (a b : ℚ) : 
  (∀ x : ℚ, a * x^3 + (a + 3*b) * x^2 + (b - 4*a) * x + (10 - a) = 0 ↔ x = -1 ∨ x = 4 ∨ x = 11/5) :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_cubic_l1245_124517


namespace NUMINAMATH_CALUDE_square_even_implies_even_l1245_124562

theorem square_even_implies_even (a : ℤ) (h : Even (a ^ 2)) : Even a := by
  sorry

end NUMINAMATH_CALUDE_square_even_implies_even_l1245_124562


namespace NUMINAMATH_CALUDE_f_neg_five_eq_one_l1245_124509

/-- A polynomial function of degree 5 with a constant term of 5 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 5

/-- Theorem stating that if f(5) = 9, then f(-5) = 1 -/
theorem f_neg_five_eq_one (a b c : ℝ) (h : f a b c 5 = 9) : f a b c (-5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_five_eq_one_l1245_124509


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1245_124560

theorem quadratic_solution_sum (a b : ℚ) : 
  (∃ x : ℂ, x = a + b * I ∧ 5 * x^2 - 2 * x + 17 = 0) →
  a + b^2 = 89/25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1245_124560


namespace NUMINAMATH_CALUDE_inequality_proof_l1245_124507

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1) : 
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1245_124507


namespace NUMINAMATH_CALUDE_wayne_blocks_l1245_124539

/-- The number of blocks Wayne's father gave him -/
def blocks_given (initial final : ℕ) : ℕ := final - initial

/-- Proof that Wayne's father gave him 6 blocks -/
theorem wayne_blocks : blocks_given 9 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_wayne_blocks_l1245_124539


namespace NUMINAMATH_CALUDE_log_inequality_l1245_124534

theorem log_inequality (a b : ℝ) : Real.log a > Real.log b → a > b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1245_124534


namespace NUMINAMATH_CALUDE_multiply_subtract_distribute_computation_result_l1245_124535

theorem multiply_subtract_distribute (a b c : ℕ) :
  a * c - b * c = (a - b) * c :=
by sorry

theorem computation_result : 72 * 1313 - 32 * 1313 = 52520 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_multiply_subtract_distribute_computation_result_l1245_124535


namespace NUMINAMATH_CALUDE_odd_sum_even_product_l1245_124544

theorem odd_sum_even_product (a b : ℤ) : 
  Odd (a + b) → Even (a * b) := by sorry

end NUMINAMATH_CALUDE_odd_sum_even_product_l1245_124544


namespace NUMINAMATH_CALUDE_regular_quad_pyramid_theorem_l1245_124570

/-- A regular quadrilateral pyramid with a plane drawn through the diagonal of the base and the height -/
structure RegularQuadPyramid where
  /-- The ratio of the area of the cross-section to the lateral surface -/
  k : ℝ
  /-- The ratio k is positive -/
  k_pos : k > 0

/-- The cosine of the angle between slant heights of opposite lateral faces -/
def slant_height_angle_cos (p : RegularQuadPyramid) : ℝ := 16 * p.k^2 - 1

/-- The theorem stating the cosine of the angle between slant heights and the permissible values of k -/
theorem regular_quad_pyramid_theorem (p : RegularQuadPyramid) :
  slant_height_angle_cos p = 16 * p.k^2 - 1 ∧ p.k < 0.25 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_regular_quad_pyramid_theorem_l1245_124570


namespace NUMINAMATH_CALUDE_range_of_a_for_monotonic_f_l1245_124592

/-- The piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 2 * x + a else x + 1

/-- The theorem stating the range of a for which f is monotonic -/
theorem range_of_a_for_monotonic_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_monotonic_f_l1245_124592


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_l1245_124598

theorem sqrt_two_irrational : Irrational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_l1245_124598


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l1245_124568

theorem consecutive_odd_integers_sum (a : ℤ) : 
  (a % 2 = 1) →                 -- a is odd
  (a + (a + 4) = 150) →         -- sum of first and third is 150
  (a + (a + 2) + (a + 4) = 225) -- sum of all three is 225
  := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l1245_124568


namespace NUMINAMATH_CALUDE_alan_cd_purchase_cost_l1245_124520

/-- The price of a CD by "AVN" in dollars -/
def avnPrice : ℝ := 12

/-- The price of a CD by "The Dark" in dollars -/
def darkPrice : ℝ := 2 * avnPrice

/-- The cost of CDs by "The Dark" and "AVN" in dollars -/
def mainCost : ℝ := 2 * darkPrice + avnPrice

/-- The cost of 90s music CDs in dollars -/
def mixCost : ℝ := 0.4 * mainCost

/-- The total cost of Alan's purchase in dollars -/
def totalCost : ℝ := mainCost + mixCost

theorem alan_cd_purchase_cost :
  totalCost = 84 := by sorry

end NUMINAMATH_CALUDE_alan_cd_purchase_cost_l1245_124520


namespace NUMINAMATH_CALUDE_rent_increase_problem_l1245_124580

theorem rent_increase_problem (num_friends : ℕ) (original_rent : ℝ) (increase_percentage : ℝ) (new_mean : ℝ) : 
  num_friends = 4 →
  original_rent = 1400 →
  increase_percentage = 0.20 →
  new_mean = 870 →
  (num_friends * new_mean - original_rent * increase_percentage) / num_friends = 800 := by
sorry

end NUMINAMATH_CALUDE_rent_increase_problem_l1245_124580


namespace NUMINAMATH_CALUDE_true_compound_props_l1245_124512

def p₁ : Prop := True
def p₂ : Prop := False
def p₃ : Prop := False
def p₄ : Prop := True

def compound_prop_1 : Prop := p₁ ∧ p₄
def compound_prop_2 : Prop := p₁ ∧ p₂
def compound_prop_3 : Prop := ¬p₂ ∨ p₃
def compound_prop_4 : Prop := ¬p₃ ∨ ¬p₄

theorem true_compound_props :
  {compound_prop_1, compound_prop_3, compound_prop_4} = 
  {p : Prop | p = compound_prop_1 ∨ p = compound_prop_2 ∨ p = compound_prop_3 ∨ p = compound_prop_4 ∧ p} :=
by sorry

end NUMINAMATH_CALUDE_true_compound_props_l1245_124512


namespace NUMINAMATH_CALUDE_expected_value_of_twelve_sided_die_l1245_124527

/-- A fair 12-sided die with faces numbered from 1 to 12 -/
def twelve_sided_die : Finset ℕ := Finset.range 12

/-- The expected value of rolling the 12-sided die -/
def expected_value : ℚ := (Finset.sum twelve_sided_die (fun i => i + 1)) / 12

/-- Theorem: The expected value of rolling a fair 12-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem expected_value_of_twelve_sided_die : expected_value = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_twelve_sided_die_l1245_124527


namespace NUMINAMATH_CALUDE_households_with_bike_only_l1245_124523

theorem households_with_bike_only 
  (total : ℕ) 
  (neither : ℕ) 
  (both : ℕ) 
  (with_car : ℕ) 
  (h1 : total = 90)
  (h2 : neither = 11)
  (h3 : both = 20)
  (h4 : with_car = 44) :
  total - neither - with_car + both = 35 := by
  sorry

end NUMINAMATH_CALUDE_households_with_bike_only_l1245_124523


namespace NUMINAMATH_CALUDE_glass_count_l1245_124585

/-- Given glasses with a capacity of 6 ounces that are 4/5 full, 
    prove that if 12 ounces of water are needed to fill all glasses, 
    there are 10 glasses. -/
theorem glass_count (glass_capacity : ℚ) (initial_fill : ℚ) (total_water_needed : ℚ) :
  glass_capacity = 6 →
  initial_fill = 4 / 5 →
  total_water_needed = 12 →
  (total_water_needed / (glass_capacity * (1 - initial_fill))) = 10 := by
  sorry


end NUMINAMATH_CALUDE_glass_count_l1245_124585


namespace NUMINAMATH_CALUDE_crate_height_is_16_feet_l1245_124564

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical gas tank -/
structure GasTank where
  radius : ℝ

/-- Theorem stating that the height of the crate is 16 feet -/
theorem crate_height_is_16_feet (crate : CrateDimensions) (tank : GasTank) :
  crate.length = 12 ∧ crate.width = 16 ∧ tank.radius = 8 →
  crate.height = 16 := by
  sorry

#check crate_height_is_16_feet

end NUMINAMATH_CALUDE_crate_height_is_16_feet_l1245_124564


namespace NUMINAMATH_CALUDE_binomial_remainder_l1245_124581

theorem binomial_remainder (x : ℕ) : x = 2000 → (1 - x)^1999 % 2000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_remainder_l1245_124581


namespace NUMINAMATH_CALUDE_inverse_sum_mod_17_l1245_124577

theorem inverse_sum_mod_17 : 
  (∃ x y : ℤ, (7 * x) % 17 = 1 ∧ (7 * y) % 17 = x % 17 ∧ (x + y) % 17 = 13) := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_17_l1245_124577


namespace NUMINAMATH_CALUDE_parabola_tangent_line_l1245_124553

theorem parabola_tangent_line (b c : ℝ) : 
  (∃ x y : ℝ, y = -2 * x^2 + b * x + c ∧ 
              y = x - 3 ∧ 
              x = 2 ∧ 
              y = -1) → 
  b + c = -2 :=
sorry


end NUMINAMATH_CALUDE_parabola_tangent_line_l1245_124553


namespace NUMINAMATH_CALUDE_defective_units_shipped_percentage_l1245_124567

theorem defective_units_shipped_percentage
  (total_units : ℝ)
  (defective_percentage : ℝ)
  (defective_shipped_percentage : ℝ)
  (h1 : defective_percentage = 5)
  (h2 : defective_shipped_percentage = 0.2)
  : (defective_shipped_percentage * total_units) / (defective_percentage * total_units) * 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_shipped_percentage_l1245_124567


namespace NUMINAMATH_CALUDE_jacob_and_nathan_letters_l1245_124573

/-- The number of letters Nathan can write in one hour -/
def nathan_letters_per_hour : ℕ := 25

/-- Jacob's writing speed relative to Nathan's -/
def jacob_speed_multiplier : ℕ := 2

/-- The number of hours Jacob and Nathan work together -/
def total_hours : ℕ := 10

/-- Theorem: Jacob and Nathan can write 750 letters in 10 hours together -/
theorem jacob_and_nathan_letters : 
  (nathan_letters_per_hour + jacob_speed_multiplier * nathan_letters_per_hour) * total_hours = 750 := by
  sorry

end NUMINAMATH_CALUDE_jacob_and_nathan_letters_l1245_124573


namespace NUMINAMATH_CALUDE_victors_friend_wins_checkers_game_wins_l1245_124513

theorem victors_friend_wins (victor_wins : ℕ) (ratio_victor : ℕ) (ratio_friend : ℕ) : ℕ :=
  let friend_wins := (victor_wins * ratio_friend) / ratio_victor
  friend_wins

theorem checkers_game_wins : victors_friend_wins 36 9 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_victors_friend_wins_checkers_game_wins_l1245_124513


namespace NUMINAMATH_CALUDE_max_distance_between_functions_l1245_124528

open Real

theorem max_distance_between_functions :
  let f (x : ℝ) := 2 * (sin (π / 4 + x))^2
  let g (x : ℝ) := Real.sqrt 3 * cos (2 * x)
  ∀ a : ℝ, |f a - g a| ≤ 3 ∧ ∃ b : ℝ, |f b - g b| = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_functions_l1245_124528


namespace NUMINAMATH_CALUDE_middle_number_proof_l1245_124588

theorem middle_number_proof (x y z : ℝ) 
  (h_distinct : x < y ∧ y < z)
  (h_sum1 : x + y = 15)
  (h_sum2 : x + z = 18)
  (h_sum3 : y + z = 21) :
  y = 9 := by
sorry

end NUMINAMATH_CALUDE_middle_number_proof_l1245_124588


namespace NUMINAMATH_CALUDE_ch4_moles_formed_l1245_124526

/-- Represents the balanced chemical equation: Be2C + 4 H2O → 2 Be(OH)2 + 3 CH4 -/
structure ChemicalEquation where
  be2c_coeff : ℚ
  h2o_coeff : ℚ
  beoh2_coeff : ℚ
  ch4_coeff : ℚ

/-- Represents the available moles of reactants -/
structure AvailableReactants where
  be2c_moles : ℚ
  h2o_moles : ℚ

/-- Calculates the moles of CH4 formed based on the chemical equation and available reactants -/
def moles_ch4_formed (equation : ChemicalEquation) (reactants : AvailableReactants) : ℚ :=
  min
    (reactants.be2c_moles * equation.ch4_coeff / equation.be2c_coeff)
    (reactants.h2o_moles * equation.ch4_coeff / equation.h2o_coeff)

theorem ch4_moles_formed
  (equation : ChemicalEquation)
  (reactants : AvailableReactants)
  (h_equation : equation = ⟨1, 4, 2, 3⟩)
  (h_reactants : reactants = ⟨3, 12⟩) :
  moles_ch4_formed equation reactants = 9 := by
  sorry

end NUMINAMATH_CALUDE_ch4_moles_formed_l1245_124526


namespace NUMINAMATH_CALUDE_ratio_equality_l1245_124594

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : y / (x - z) = (x + y) / z ∧ (x + y) / z = x / y) : 
  x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1245_124594


namespace NUMINAMATH_CALUDE_two_pi_irrational_l1245_124569

theorem two_pi_irrational : Irrational (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_two_pi_irrational_l1245_124569


namespace NUMINAMATH_CALUDE_interval_length_implies_difference_l1245_124502

/-- Given an inequality a ≤ 3x + 6 ≤ b, if the length of the interval of solutions is 15, then b - a = 45 -/
theorem interval_length_implies_difference (a b : ℝ) : 
  (∃ (l : ℝ), l = 15 ∧ l = (b - 6) / 3 - (a - 6) / 3) → b - a = 45 := by
  sorry

end NUMINAMATH_CALUDE_interval_length_implies_difference_l1245_124502


namespace NUMINAMATH_CALUDE_fraction_equality_l1245_124576

theorem fraction_equality : (1/4 - 1/6) / (1/3 - 1/4) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1245_124576


namespace NUMINAMATH_CALUDE_shortened_area_l1245_124503

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The original rectangle --/
def original : Rectangle := { length := 5, width := 7 }

/-- The rectangle after shortening one side --/
def shortened : Rectangle := { length := 3, width := 7 }

/-- Theorem stating the relationship between the original rectangle and the shortened rectangle --/
theorem shortened_area (h : area shortened = 21) :
  ∃ (r : Rectangle), r.length = original.length ∧ r.width = original.width - 2 ∧ area r = 25 := by
  sorry


end NUMINAMATH_CALUDE_shortened_area_l1245_124503


namespace NUMINAMATH_CALUDE_negative_two_exponent_division_l1245_124506

theorem negative_two_exponent_division : 
  (-2: ℤ) ^ 2014 / (-2 : ℤ) ^ 2013 = -2 := by sorry

end NUMINAMATH_CALUDE_negative_two_exponent_division_l1245_124506


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1245_124556

def set_A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def set_B : Set ℝ := {x | x - 1 < 0}

theorem union_of_A_and_B : set_A ∪ set_B = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1245_124556


namespace NUMINAMATH_CALUDE_ninas_homework_is_40_l1245_124582

/-- The amount of Nina's total homework given Ruby's homework and the ratios --/
def ninas_total_homework (rubys_math_homework : ℕ) (rubys_reading_homework : ℕ) 
  (math_ratio : ℕ) (reading_ratio : ℕ) : ℕ :=
  math_ratio * rubys_math_homework + reading_ratio * rubys_reading_homework

/-- Theorem stating that Nina's total homework is 40 given the problem conditions --/
theorem ninas_homework_is_40 :
  ninas_total_homework 6 2 4 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ninas_homework_is_40_l1245_124582


namespace NUMINAMATH_CALUDE_chipmunk_increase_l1245_124548

/-- Proves that the number of chipmunks increased by 50 given the initial counts, doubling of beavers, and total animal count. -/
theorem chipmunk_increase (initial_beavers initial_chipmunks total_animals : ℕ) 
  (h1 : initial_beavers = 20)
  (h2 : initial_chipmunks = 40)
  (h3 : total_animals = 130) :
  (total_animals - 2 * initial_beavers) - initial_chipmunks = 50 := by
  sorry

#check chipmunk_increase

end NUMINAMATH_CALUDE_chipmunk_increase_l1245_124548


namespace NUMINAMATH_CALUDE_sandy_earnings_l1245_124563

/-- Calculates Sandy's earnings for a given day -/
def daily_earnings (hours : ℝ) (hourly_rate : ℝ) (with_best_friend : Bool) (longer_than_12_hours : Bool) : ℝ :=
  let base_wage := hours * hourly_rate
  let commission := if with_best_friend then base_wage * 0.1 else 0
  let bonus := if longer_than_12_hours then base_wage * 0.05 else 0
  let total_before_tax := base_wage + commission + bonus
  total_before_tax * 0.93  -- Apply 7% tax deduction

/-- Sandy's total earnings for Friday, Saturday, and Sunday -/
def total_earnings : ℝ :=
  let hourly_rate := 15
  let friday := daily_earnings 10 hourly_rate true false
  let saturday := daily_earnings 6 hourly_rate false false
  let sunday := daily_earnings 14 hourly_rate false true
  friday + saturday + sunday

/-- Theorem stating Sandy's total earnings -/
theorem sandy_earnings : total_earnings = 442.215 := by
  sorry

end NUMINAMATH_CALUDE_sandy_earnings_l1245_124563


namespace NUMINAMATH_CALUDE_max_area_APBQ_l1245_124584

noncomputable section

-- Define the Cartesian coordinate system
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define the distance ratio condition
def distance_ratio (P : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 + 1)^2 + P.2^2) / |P.1 + 2|

-- Define the trajectory C
def C : Set (ℝ × ℝ) :=
  {P | distance_ratio P = Real.sqrt 2 / 2}

-- Define the circle C₁
def C₁ : Set (ℝ × ℝ) :=
  {P | (P.1 - 4)^2 + P.2^2 = 32}

-- Define a chord AB of C passing through F
def chord_AB (m : ℝ) : Set (ℝ × ℝ) :=
  {P | P ∈ C ∧ P.1 = m * P.2 - 1}

-- Define the midpoint M of AB
def M (m : ℝ) : ℝ × ℝ :=
  (-2 / (m^2 + 2), m / (m^2 + 2))

-- Define the line OM
def line_OM (m : ℝ) : Set (ℝ × ℝ) :=
  {P | P.2 = (m / (m^2 + 2)) * P.1}

-- Define the intersection points P and Q
def P_Q (m : ℝ) : Set (ℝ × ℝ) :=
  {P | P ∈ C₁ ∧ P ∈ line_OM m}

-- Define the area of quadrilateral APBQ
def area_APBQ (m : ℝ) : ℝ :=
  8 * Real.sqrt 2 * Real.sqrt ((m^2 + 8) * (m^2 + 1) / (m^2 + 4)^2)

-- Theorem statement
theorem max_area_APBQ :
  ∃ m : ℝ, ∀ n : ℝ, area_APBQ m ≥ area_APBQ n ∧ area_APBQ m = 14 * Real.sqrt 6 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_area_APBQ_l1245_124584


namespace NUMINAMATH_CALUDE_largest_term_binomial_expansion_l1245_124591

theorem largest_term_binomial_expansion (n : ℕ) (x : ℝ) (h : n = 500 ∧ x = 0.1) :
  ∃ k : ℕ, k = 45 ∧
  ∀ j : ℕ, j ≤ n → (n.choose k) * x^k ≥ (n.choose j) * x^j :=
sorry

end NUMINAMATH_CALUDE_largest_term_binomial_expansion_l1245_124591


namespace NUMINAMATH_CALUDE_polynomial_linear_if_all_powers_l1245_124510

/-- A sequence defined by a polynomial recurrence -/
def PolynomialSequence (P : ℕ → ℕ) (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => P (PolynomialSequence P n k)

/-- Predicate to check if a number is a perfect power greater than 1 -/
def IsPerfectPower (m : ℕ) : Prop :=
  ∃ (b : ℕ) (k : ℕ), k > 1 ∧ m = k^b

theorem polynomial_linear_if_all_powers (P : ℕ → ℕ) (n : ℕ) :
  (∀ (x y : ℕ), ∃ (a b c : ℤ), P x - P y = a * (x - y) + b * x + c) →
  (∀ (b : ℕ), ∃ (k : ℕ), IsPerfectPower (PolynomialSequence P n k)) →
  ∃ (m q : ℤ), ∀ (x : ℕ), P x = m * x + q :=
sorry

end NUMINAMATH_CALUDE_polynomial_linear_if_all_powers_l1245_124510


namespace NUMINAMATH_CALUDE_ruby_pizza_order_cost_l1245_124530

/-- Represents the cost of a pizza order --/
structure PizzaOrder where
  basePizzaCost : ℝ
  toppingCost : ℝ
  tipAmount : ℝ
  pepperoniToppings : ℕ
  sausageToppings : ℕ
  blackOliveMushroomToppings : ℕ
  numberOfPizzas : ℕ

/-- Calculates the total cost of a pizza order --/
def totalCost (order : PizzaOrder) : ℝ :=
  order.basePizzaCost * order.numberOfPizzas +
  order.toppingCost * (order.pepperoniToppings + order.sausageToppings + order.blackOliveMushroomToppings) +
  order.tipAmount

/-- Theorem stating that the total cost of Ruby's pizza order is $39.00 --/
theorem ruby_pizza_order_cost :
  let order : PizzaOrder := {
    basePizzaCost := 10,
    toppingCost := 1,
    tipAmount := 5,
    pepperoniToppings := 1,
    sausageToppings := 1,
    blackOliveMushroomToppings := 2,
    numberOfPizzas := 3
  }
  totalCost order = 39 := by
  sorry

end NUMINAMATH_CALUDE_ruby_pizza_order_cost_l1245_124530


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1245_124545

def A : Set ℤ := {1, 2, 3}

def B : Set ℤ := {x | (x + 1) * (x - 2) < 0}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1245_124545


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1245_124514

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1245_124514


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1245_124519

theorem intersection_of_sets : 
  let A : Set ℤ := {0, 1, 2, 4}
  let B : Set ℤ := {-1, 0, 1, 3}
  A ∩ B = {0, 1} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1245_124519


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l1245_124575

theorem quadratic_perfect_square (m : ℝ) :
  (∃ (a b : ℝ), ∀ x, (6*x^2 + 16*x + 3*m) / 6 = (a*x + b)^2) →
  m = 32/9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l1245_124575


namespace NUMINAMATH_CALUDE_solve_y_l1245_124559

theorem solve_y (x y : ℝ) (h1 : x - y = 20) (h2 : x + y = 14) : y = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_y_l1245_124559


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l1245_124537

theorem max_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0)
  (hsum1 : x/y + y/z + z/x = 3) (hsum2 : x + y + z = 6) :
  ∃ (M : ℝ), ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    a/b + b/c + c/a = 3 → a + b + c = 6 →
    y/x + z/y + x/z ≤ M ∧ M = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l1245_124537


namespace NUMINAMATH_CALUDE_right_triangle_area_l1245_124550

/-- The area of a right triangle with a leg of 28 inches and a hypotenuse of 30 inches is 28√29 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 28) (h2 : c = 30) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 28 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1245_124550


namespace NUMINAMATH_CALUDE_delta_donuts_calculation_l1245_124587

def total_donuts : ℕ := 40
def gamma_donuts : ℕ := 8
def beta_donuts : ℕ := 3 * gamma_donuts

theorem delta_donuts_calculation :
  total_donuts - (beta_donuts + gamma_donuts) = 8 :=
by sorry

end NUMINAMATH_CALUDE_delta_donuts_calculation_l1245_124587


namespace NUMINAMATH_CALUDE_system_unique_solution_l1245_124583

/-- The system of equations has a unique solution -/
theorem system_unique_solution :
  ∃! (x₁ x₂ x₃ : ℝ),
    3 * x₁ + 4 * x₂ + 3 * x₃ = 0 ∧
    x₁ - x₂ + x₃ = 0 ∧
    x₁ + 3 * x₂ - x₃ = -2 ∧
    x₁ + 2 * x₂ + 3 * x₃ = 2 ∧
    x₁ = 1 ∧ x₂ = 0 ∧ x₃ = 1 := by
  sorry


end NUMINAMATH_CALUDE_system_unique_solution_l1245_124583


namespace NUMINAMATH_CALUDE_train_length_proof_l1245_124529

def train_problem (speed1 speed2 shorter_length clearing_time : ℝ) : Prop :=
  let relative_speed := (speed1 + speed2) * 1000 / 3600
  let total_distance := relative_speed * clearing_time
  let longer_length := total_distance - shorter_length
  longer_length = 164.9771230827526

theorem train_length_proof :
  train_problem 80 55 121 7.626056582140095 := by sorry

end NUMINAMATH_CALUDE_train_length_proof_l1245_124529


namespace NUMINAMATH_CALUDE_inverse_f_at_4_l1245_124551

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

def HasInverse (f : ℝ → ℝ) (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

theorem inverse_f_at_4 (f_inv : ℝ → ℝ) (h : HasInverse f f_inv) : f_inv 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_4_l1245_124551


namespace NUMINAMATH_CALUDE_count_not_divisible_1999_l1245_124501

def count_not_divisible (n : ℕ) : ℕ :=
  n - (n / 4 + n / 6 - n / 12)

theorem count_not_divisible_1999 :
  count_not_divisible 1999 = 1333 := by
  sorry

end NUMINAMATH_CALUDE_count_not_divisible_1999_l1245_124501
