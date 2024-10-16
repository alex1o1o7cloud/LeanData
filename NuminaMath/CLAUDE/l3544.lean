import Mathlib

namespace NUMINAMATH_CALUDE_product_inequality_l3544_354480

theorem product_inequality (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1) (h₄ : a₄ > 1) : 
  8 * (a₁ * a₂ * a₃ * a₄ + 1) ≥ (1 + a₁) * (1 + a₂) * (1 + a₃) * (1 + a₄) := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3544_354480


namespace NUMINAMATH_CALUDE_nonnegative_difference_of_roots_l3544_354441

theorem nonnegative_difference_of_roots (x : ℝ) : 
  let roots := {r : ℝ | r^2 + 6*r + 8 = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ |r₁ - r₂| = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_nonnegative_difference_of_roots_l3544_354441


namespace NUMINAMATH_CALUDE_correct_2star_reviews_l3544_354454

/-- The number of 2-star reviews for Indigo Restaurant --/
def num_2star_reviews : ℕ := 
  let total_reviews : ℕ := 18
  let num_5star : ℕ := 6
  let num_4star : ℕ := 7
  let num_3star : ℕ := 4
  let avg_rating : ℚ := 4
  1

/-- Theorem stating that the number of 2-star reviews is correct --/
theorem correct_2star_reviews : 
  let total_reviews : ℕ := 18
  let num_5star : ℕ := 6
  let num_4star : ℕ := 7
  let num_3star : ℕ := 4
  let avg_rating : ℚ := 4
  num_2star_reviews = 1 ∧ 
  (5 * num_5star + 4 * num_4star + 3 * num_3star + 2 * num_2star_reviews : ℚ) / total_reviews = avg_rating :=
by sorry

end NUMINAMATH_CALUDE_correct_2star_reviews_l3544_354454


namespace NUMINAMATH_CALUDE_partner_A_share_l3544_354410

/-- Represents a partner's investment in a partnership --/
structure Investment where
  capital_ratio : ℚ
  time_ratio : ℚ

/-- Calculates the share of profit for a given investment --/
def calculate_share (inv : Investment) (total_capital_time : ℚ) (total_profit : ℚ) : ℚ :=
  (inv.capital_ratio * inv.time_ratio) / total_capital_time * total_profit

/-- Theorem stating that partner A's share of the profit is 100 --/
theorem partner_A_share :
  let a := Investment.mk (1/6) (1/6)
  let b := Investment.mk (1/3) (1/3)
  let c := Investment.mk (1/2) 1
  let total_capital_time := (1/6 * 1/6) + (1/3 * 1/3) + (1/2 * 1)
  let total_profit := 2300
  calculate_share a total_capital_time total_profit = 100 := by
  sorry

end NUMINAMATH_CALUDE_partner_A_share_l3544_354410


namespace NUMINAMATH_CALUDE_part1_part2_l3544_354472

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |x - 2*a + 3|

-- Part 1: When a = 2
theorem part1 : {x : ℝ | f x 2 ≤ 9} = {x : ℝ | -2 ≤ x ∧ x ≤ 4} := by sorry

-- Part 2: When a ≠ 2
theorem part2 : ∀ a : ℝ, a ≠ 2 → 
  ((∀ x : ℝ, f x a ≥ 4) ↔ (a ≤ -2/3 ∨ a ≥ 14/3)) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3544_354472


namespace NUMINAMATH_CALUDE_display_board_sides_l3544_354475

/-- A polygonal display board with given perimeter and side ribbon length has a specific number of sides. -/
theorem display_board_sides (perimeter : ℝ) (side_ribbon_length : ℝ) (num_sides : ℕ) : 
  perimeter = 42 → side_ribbon_length = 7 → num_sides * side_ribbon_length = perimeter → num_sides = 6 := by
  sorry

end NUMINAMATH_CALUDE_display_board_sides_l3544_354475


namespace NUMINAMATH_CALUDE_block_rotation_contacts_19_squares_l3544_354497

/-- Represents a rectangular block with three dimensions -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a square board -/
structure Board where
  size : ℕ

/-- Represents a face of the block -/
inductive Face
  | X
  | Y
  | Z

/-- Calculates the area of a given face of the block -/
def faceArea (b : Block) (f : Face) : ℕ :=
  match f with
  | Face.X => b.length * b.width
  | Face.Y => b.length * b.height
  | Face.Z => b.width * b.height

/-- Sequence of faces touched during rotation -/
def rotationSequence : List Face :=
  [Face.X, Face.Y, Face.Z, Face.X, Face.Y, Face.Z]

/-- Calculates the number of unique squares contacted after rotations -/
def uniqueSquaresContacted (block : Block) (board : Board) : ℕ :=
  sorry  -- Proof goes here

/-- Main theorem: The block contacts exactly 19 unique squares -/
theorem block_rotation_contacts_19_squares :
  let block : Block := { length := 1, width := 2, height := 3 }
  let board : Board := { size := 8 }
  uniqueSquaresContacted block board = 19 :=
by sorry

end NUMINAMATH_CALUDE_block_rotation_contacts_19_squares_l3544_354497


namespace NUMINAMATH_CALUDE_combined_girls_average_l3544_354419

/-- Represents a high school with given average scores -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Represents the combined data for two high schools -/
structure CombinedSchools where
  cedar : School
  delta : School
  boys_combined_avg : ℝ

/-- Theorem stating that the combined girls' average is 86 -/
theorem combined_girls_average (schools : CombinedSchools) 
  (h1 : schools.cedar.boys_avg = 85)
  (h2 : schools.cedar.girls_avg = 80)
  (h3 : schools.cedar.combined_avg = 83)
  (h4 : schools.delta.boys_avg = 76)
  (h5 : schools.delta.girls_avg = 95)
  (h6 : schools.delta.combined_avg = 87)
  (h7 : schools.boys_combined_avg = 73) :
  ∃ (cedar_boys cedar_girls delta_boys delta_girls : ℝ),
    cedar_boys > 0 ∧ cedar_girls > 0 ∧ delta_boys > 0 ∧ delta_girls > 0 ∧
    (cedar_boys * 85 + cedar_girls * 80) / (cedar_boys + cedar_girls) = 83 ∧
    (delta_boys * 76 + delta_girls * 95) / (delta_boys + delta_girls) = 87 ∧
    (cedar_boys * 85 + delta_boys * 76) / (cedar_boys + delta_boys) = 73 ∧
    (cedar_girls * 80 + delta_girls * 95) / (cedar_girls + delta_girls) = 86 :=
by sorry


end NUMINAMATH_CALUDE_combined_girls_average_l3544_354419


namespace NUMINAMATH_CALUDE_age_difference_john_aunt_l3544_354446

/-- Represents the ages of family members --/
structure FamilyAges where
  john : ℕ
  father : ℕ
  mother : ℕ
  grandmother : ℕ
  aunt : ℕ

/-- Defines the relationships between family members' ages --/
def valid_family_ages (ages : FamilyAges) : Prop :=
  ages.john * 2 = ages.father ∧
  ages.father = ages.mother + 4 ∧
  ages.grandmother = ages.john * 3 ∧
  ages.aunt = ages.mother * 2 - 5 ∧
  ages.father = 40

/-- Theorem stating the age difference between John and his aunt --/
theorem age_difference_john_aunt (ages : FamilyAges) 
  (h : valid_family_ages ages) : ages.aunt - ages.john = 47 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_john_aunt_l3544_354446


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l3544_354496

/-- Given a line segment with one endpoint at (6, 1) and midpoint at (5, 7),
    the sum of the coordinates of the other endpoint is 17. -/
theorem endpoint_coordinate_sum : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop :=
  fun endpoint1 midpoint endpoint2 =>
    endpoint1 = (6, 1) ∧
    midpoint = (5, 7) ∧
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
    endpoint2.1 + endpoint2.2 = 17

/-- Proof of the theorem -/
theorem endpoint_coordinate_sum_proof : ∃ (endpoint2 : ℝ × ℝ),
  endpoint_coordinate_sum (6, 1) (5, 7) endpoint2 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l3544_354496


namespace NUMINAMATH_CALUDE_arccos_one_eq_zero_l3544_354443

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_eq_zero_l3544_354443


namespace NUMINAMATH_CALUDE_extremum_values_l3544_354465

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_values (a b : ℝ) :
  f a b 1 = 10 ∧ (deriv (f a b)) 1 = 0 → a = 4 ∧ b = -11 := by
  sorry

end NUMINAMATH_CALUDE_extremum_values_l3544_354465


namespace NUMINAMATH_CALUDE_hotel_room_charges_l3544_354436

theorem hotel_room_charges (G : ℝ) (h1 : G > 0) : 
  let R := G * (1 + 0.19999999999999986)
  let P := R * (1 - 0.25)
  P = G * (1 - 0.1) :=
by sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l3544_354436


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3544_354415

/-- For a parabola with equation y^2 = 2x, the distance from its focus to its directrix is 1 -/
theorem parabola_focus_directrix_distance :
  ∀ (y x : ℝ), y^2 = 2*x → (∃ (focus_x directrix_x : ℝ),
    (∀ (point_x point_y : ℝ), point_y^2 = 2*point_x ↔ 
      (point_x - focus_x)^2 + point_y^2 = (point_x - directrix_x)^2) ∧
    focus_x - directrix_x = 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3544_354415


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3544_354423

theorem arithmetic_calculation : 2 + 3 * 4 - 5 + 6 / 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3544_354423


namespace NUMINAMATH_CALUDE_soil_bags_needed_l3544_354438

/-- Calculates the number of soil bags needed for raised beds -/
theorem soil_bags_needed
  (num_beds : ℕ)
  (length width height : ℝ)
  (soil_per_bag : ℝ)
  (h_num_beds : num_beds = 2)
  (h_length : length = 8)
  (h_width : width = 4)
  (h_height : height = 1)
  (h_soil_per_bag : soil_per_bag = 4) :
  ⌈(num_beds * length * width * height) / soil_per_bag⌉ = 16 := by
  sorry

end NUMINAMATH_CALUDE_soil_bags_needed_l3544_354438


namespace NUMINAMATH_CALUDE_smallest_bob_number_l3544_354473

def alice_number : Nat := 30

def has_all_prime_factors_of (n m : Nat) : Prop :=
  ∀ p : Nat, Nat.Prime p → p ∣ n → p ∣ m

def has_additional_prime_factor (n m : Nat) : Prop :=
  ∃ p : Nat, Nat.Prime p ∧ p ∣ m ∧ ¬(p ∣ n)

theorem smallest_bob_number :
  ∃ bob_number : Nat,
    has_all_prime_factors_of alice_number bob_number ∧
    has_additional_prime_factor alice_number bob_number ∧
    (∀ m : Nat, m < bob_number →
      ¬(has_all_prime_factors_of alice_number m ∧
        has_additional_prime_factor alice_number m)) ∧
    bob_number = 210 := by
  sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l3544_354473


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l3544_354434

theorem geometric_series_first_term (a r : ℝ) (h1 : r ≠ 1) (h2 : |r| < 1) :
  (a / (1 - r) = 12) → (a^2 / (1 - r^2) = 36) → a = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l3544_354434


namespace NUMINAMATH_CALUDE_quadratic_common_roots_l3544_354455

theorem quadratic_common_roots : 
  ∀ (p : ℚ) (x : ℚ),
  (9 * x^2 - 3 * (p + 6) * x + 6 * p + 5 = 0 ∧
   6 * x^2 - 3 * (p + 4) * x + 6 * p + 14 = 0) ↔
  ((p = -32/9 ∧ x = -1) ∨ (p = 32/3 ∧ x = 3)) := by sorry

end NUMINAMATH_CALUDE_quadratic_common_roots_l3544_354455


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3544_354495

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 3*x₁ - 4 = 0) → (x₂^2 - 3*x₂ - 4 = 0) → x₁ + x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3544_354495


namespace NUMINAMATH_CALUDE_election_votes_l3544_354458

theorem election_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (60 : ℚ) / 100 * total_votes - (40 : ℚ) / 100 * total_votes = 288) : 
  (60 : ℚ) / 100 * total_votes = 864 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l3544_354458


namespace NUMINAMATH_CALUDE_chess_tournament_schedules_l3544_354406

/-- Represents the number of players from each school -/
def num_players : ℕ := 4

/-- Represents the number of games each player plays against each opponent -/
def games_per_opponent : ℕ := 3

/-- Represents the number of games played simultaneously in each round -/
def games_per_round : ℕ := 3

/-- Calculates the total number of games in the tournament -/
def total_games : ℕ := num_players * num_players * games_per_opponent

/-- Calculates the number of rounds in the tournament -/
def num_rounds : ℕ := total_games / games_per_round

/-- Theorem stating the number of distinct ways to schedule the tournament -/
theorem chess_tournament_schedules :
  (Nat.factorial num_rounds) / (Nat.factorial games_per_round) =
  (Nat.factorial 16) / (Nat.factorial 3) :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_schedules_l3544_354406


namespace NUMINAMATH_CALUDE_part1_part2_l3544_354477

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |2*x + a|

-- Part 1: Prove that for a=1, f(x) + |x-1| ≥ 3 for all x
theorem part1 : ∀ x : ℝ, f 1 x + |x - 1| ≥ 3 := by sorry

-- Part 2: Prove that the minimum value of f(x) is 2 if and only if a = 2 or a = -6
theorem part2 : (∃ x : ℝ, f a x = 2) ∧ (∀ y : ℝ, f a y ≥ 2) ↔ a = 2 ∨ a = -6 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3544_354477


namespace NUMINAMATH_CALUDE_good_permutations_congruence_l3544_354493

/-- Given a prime number p > 3, count_good_permutations p returns the number of permutations
    (a₁, a₂, ..., aₚ₋₁) of (1, 2, ..., p-1) such that p divides the sum of consecutive products. -/
def count_good_permutations (p : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the number of good permutations is congruent to p-1 modulo p(p-1). -/
theorem good_permutations_congruence (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  count_good_permutations p ≡ p - 1 [MOD p * (p - 1)] :=
sorry

end NUMINAMATH_CALUDE_good_permutations_congruence_l3544_354493


namespace NUMINAMATH_CALUDE_donny_remaining_money_l3544_354456

def initial_amount : ℕ := 78
def kite_cost : ℕ := 8
def frisbee_cost : ℕ := 9

theorem donny_remaining_money :
  initial_amount - (kite_cost + frisbee_cost) = 61 := by
  sorry

end NUMINAMATH_CALUDE_donny_remaining_money_l3544_354456


namespace NUMINAMATH_CALUDE_trigonometric_equality_l3544_354427

theorem trigonometric_equality (x y z a : ℝ) 
  (h1 : (Real.sin x + Real.sin y + Real.sin z) / Real.sin (x + y + z) = a)
  (h2 : (Real.cos x + Real.cos y + Real.cos z) / Real.cos (x + y + z) = a) :
  Real.cos (x + y) + Real.cos (y + z) + Real.cos (z + x) = a := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l3544_354427


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3544_354403

theorem polynomial_division_remainder 
  (dividend : Polynomial ℚ) 
  (divisor : Polynomial ℚ) 
  (quotient : Polynomial ℚ) 
  (remainder : Polynomial ℚ) :
  dividend = 3 * X^4 + 7 * X^3 - 28 * X^2 - 32 * X + 53 →
  divisor = X^2 + 5 * X + 3 →
  dividend = divisor * quotient + remainder →
  Polynomial.degree remainder < Polynomial.degree divisor →
  remainder = 97 * X + 116 := by
    sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3544_354403


namespace NUMINAMATH_CALUDE_gcd_2134_155_in_ternary_is_100_l3544_354462

def gcd_2134_155_in_ternary : List Nat :=
  let m := Nat.gcd 2134 155
  Nat.digits 3 m

theorem gcd_2134_155_in_ternary_is_100 : 
  gcd_2134_155_in_ternary = [1, 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_gcd_2134_155_in_ternary_is_100_l3544_354462


namespace NUMINAMATH_CALUDE_cos_420_degrees_l3544_354468

theorem cos_420_degrees : Real.cos (420 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_420_degrees_l3544_354468


namespace NUMINAMATH_CALUDE_opposite_of_two_l3544_354448

-- Define the concept of opposite
def opposite (x : ℝ) : ℝ := -x

-- Theorem statement
theorem opposite_of_two : opposite 2 = -2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_two_l3544_354448


namespace NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l3544_354467

theorem cubic_factorization_sum_of_squares (p q r s t u : ℤ) :
  (∀ x, 512 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 5472 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l3544_354467


namespace NUMINAMATH_CALUDE_probability_square_or_circle_l3544_354485

/- Define the total number of figures -/
def total_figures : ℕ := 10

/- Define the number of squares -/
def num_squares : ℕ := 4

/- Define the number of circles -/
def num_circles : ℕ := 3

/- Theorem statement -/
theorem probability_square_or_circle :
  (num_squares + num_circles : ℚ) / total_figures = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_square_or_circle_l3544_354485


namespace NUMINAMATH_CALUDE_tileIV_in_rectangleD_l3544_354474

-- Define the structure for a tile
structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the tiles
def tileI : Tile := ⟨3, 1, 4, 2⟩
def tileII : Tile := ⟨2, 3, 1, 5⟩
def tileIII : Tile := ⟨4, 0, 3, 1⟩
def tileIV : Tile := ⟨5, 4, 2, 0⟩

-- Define the set of all tiles
def allTiles : Set Tile := {tileI, tileII, tileIII, tileIV}

-- Define a function to check if two tiles can be adjacent
def canBeAdjacent (t1 t2 : Tile) : Bool :=
  (t1.right = t2.left) ∨ (t1.left = t2.right) ∨ (t1.top = t2.bottom) ∨ (t1.bottom = t2.top)

-- Theorem: Tile IV must be placed in Rectangle D
theorem tileIV_in_rectangleD :
  ∀ (t : Tile), t ∈ allTiles → t ≠ tileIV →
    ∃ (t' : Tile), t' ∈ allTiles ∧ t' ≠ t ∧ t' ≠ tileIV ∧ canBeAdjacent t t' = true →
      ¬∃ (t'' : Tile), t'' ∈ allTiles ∧ t'' ≠ tileIV ∧ canBeAdjacent tileIV t'' = true :=
sorry

end NUMINAMATH_CALUDE_tileIV_in_rectangleD_l3544_354474


namespace NUMINAMATH_CALUDE_number_of_workers_l3544_354469

/-- Given the wages for two groups of workers, prove the number of workers in the first group -/
theorem number_of_workers (W : ℕ) : 
  (6 * W * (9975 / (5 * 19)) = 9450) →
  (W = 15) := by
sorry

end NUMINAMATH_CALUDE_number_of_workers_l3544_354469


namespace NUMINAMATH_CALUDE_sphere_to_cone_height_l3544_354424

/-- Given a sphere with diameter 6 cm and a cone with base diameter 12 cm,
    if their volumes are equal, then the height of the cone is 3 cm. -/
theorem sphere_to_cone_height (sphere_diameter : ℝ) (cone_base_diameter : ℝ) (cone_height : ℝ) :
  sphere_diameter = 6 →
  cone_base_diameter = 12 →
  (4 / 3) * Real.pi * (sphere_diameter / 2) ^ 3 = (1 / 3) * Real.pi * (cone_base_diameter / 2) ^ 2 * cone_height →
  cone_height = 3 := by
  sorry

#check sphere_to_cone_height

end NUMINAMATH_CALUDE_sphere_to_cone_height_l3544_354424


namespace NUMINAMATH_CALUDE_product_of_roots_cubic_equation_l3544_354479

theorem product_of_roots_cubic_equation : 
  let f : ℝ → ℝ := λ x => 3 * x^3 - 4 * x^2 + x - 5
  ∃ a b c : ℝ, (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧ a * b * c = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_cubic_equation_l3544_354479


namespace NUMINAMATH_CALUDE_miriam_initial_marbles_l3544_354490

/-- The number of marbles Miriam initially had -/
def initial_marbles : ℕ := sorry

/-- The number of marbles Miriam gave to her brother -/
def marbles_to_brother : ℕ := 60

/-- The number of marbles Miriam gave to her sister -/
def marbles_to_sister : ℕ := 2 * marbles_to_brother

/-- The number of marbles Miriam currently has -/
def current_marbles : ℕ := 300

/-- The number of marbles Miriam gave to her friend Savanna -/
def marbles_to_savanna : ℕ := 3 * current_marbles

theorem miriam_initial_marbles :
  initial_marbles = marbles_to_brother + marbles_to_sister + marbles_to_savanna + current_marbles :=
by sorry

end NUMINAMATH_CALUDE_miriam_initial_marbles_l3544_354490


namespace NUMINAMATH_CALUDE_solve_for_k_l3544_354453

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 6

def g (k x : ℝ) : ℝ := x^2 - k * x - 8

theorem solve_for_k : ∃ k : ℝ, f 5 - g k 5 = 20 ∧ k = -10.8 := by sorry

end NUMINAMATH_CALUDE_solve_for_k_l3544_354453


namespace NUMINAMATH_CALUDE_power_of_power_three_cubed_squared_l3544_354449

theorem power_of_power_three_cubed_squared : (3^3)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_cubed_squared_l3544_354449


namespace NUMINAMATH_CALUDE_completing_square_l3544_354437

theorem completing_square (x : ℝ) : x^2 - 4*x - 8 = 0 ↔ (x - 2)^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l3544_354437


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3544_354426

/-- Given a rectangular solid with length l, width w, and height h, 
    prove that if it satisfies certain volume change conditions, 
    its surface area is 290 square cm. -/
theorem rectangular_solid_surface_area 
  (l w h : ℝ) 
  (h1 : (l - 2) * w * h = l * w * h - 48)
  (h2 : l * (w + 3) * h = l * w * h + 99)
  (h3 : l * w * (h + 4) = l * w * h + 352)
  : 2 * (l * w + l * h + w * h) = 290 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3544_354426


namespace NUMINAMATH_CALUDE_cube_root_of_eight_l3544_354489

theorem cube_root_of_eight (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_eight_l3544_354489


namespace NUMINAMATH_CALUDE_circle_m_range_l3544_354400

/-- A circle equation with parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 2*x + m = 0

/-- Condition for the equation to represent a circle -/
def is_circle (m : ℝ) : Prop :=
  ∃ (x₀ y₀ r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y m ↔ (x - x₀)^2 + (y - y₀)^2 = r^2

/-- The range of m for which the equation represents a circle -/
theorem circle_m_range :
  ∀ m : ℝ, is_circle m ↔ m < 1 :=
sorry

end NUMINAMATH_CALUDE_circle_m_range_l3544_354400


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l3544_354412

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l3544_354412


namespace NUMINAMATH_CALUDE_table_covering_l3544_354421

/-- A tile type used to cover the table -/
inductive Tile
  | Square  -- 2×2 square tile
  | LShaped -- L-shaped tile with 5 cells

/-- Represents a covering of the table -/
def Covering (m n : ℕ) := List (ℕ × ℕ × Tile)

/-- Checks if a covering is valid for the given table dimensions -/
def IsValidCovering (m n : ℕ) (c : Covering m n) : Prop := sorry

/-- The main theorem stating the condition for possible covering -/
theorem table_covering (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  (∃ c : Covering m n, IsValidCovering m n c) ↔ (6 ∣ m ∨ 6 ∣ n) :=
sorry

end NUMINAMATH_CALUDE_table_covering_l3544_354421


namespace NUMINAMATH_CALUDE_manufacturing_sector_degrees_l3544_354411

/-- Represents the number of degrees in a full circle -/
def full_circle : ℝ := 360

/-- Represents the percentage of employees in manufacturing as a decimal -/
def manufacturing_percentage : ℝ := 0.40

/-- Calculates the number of degrees occupied by a sector in a circle graph
    given the percentage it represents -/
def sector_degrees (percentage : ℝ) : ℝ := full_circle * percentage

theorem manufacturing_sector_degrees :
  sector_degrees manufacturing_percentage = 144 := by sorry

end NUMINAMATH_CALUDE_manufacturing_sector_degrees_l3544_354411


namespace NUMINAMATH_CALUDE_complex_exponential_product_l3544_354483

theorem complex_exponential_product (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = -1/3 + 4/5 * Complex.I →
  (Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β)) *
  (Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β)) = 169/225 := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_product_l3544_354483


namespace NUMINAMATH_CALUDE_power_of_power_three_l3544_354405

theorem power_of_power_three : (3^3)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l3544_354405


namespace NUMINAMATH_CALUDE_triangle_areas_l3544_354431

theorem triangle_areas (BD DC : ℝ) (area_ABD : ℝ) :
  BD / DC = 2 / 5 →
  area_ABD = 28 →
  ∃ (area_ADC area_ABC : ℝ),
    area_ADC = 70 ∧
    area_ABC = 98 :=
by sorry

end NUMINAMATH_CALUDE_triangle_areas_l3544_354431


namespace NUMINAMATH_CALUDE_complex_number_location_l3544_354451

theorem complex_number_location (z : ℂ) (h : z * Complex.I = 2 - Complex.I) :
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l3544_354451


namespace NUMINAMATH_CALUDE_one_of_each_color_probability_l3544_354435

-- Define the number of marbles of each color
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 3
def green_marbles : ℕ := 3
def yellow_marbles : ℕ := 3

-- Define the total number of marbles
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles + yellow_marbles

-- Define the number of marbles to be selected
def selected_marbles : ℕ := 4

-- Define the probability of selecting one marble of each color
def probability_one_of_each : ℚ := 9 / 55

-- Theorem statement
theorem one_of_each_color_probability :
  (red_marbles * blue_marbles * green_marbles * yellow_marbles : ℚ) /
  (Nat.choose total_marbles selected_marbles) = probability_one_of_each :=
by sorry

end NUMINAMATH_CALUDE_one_of_each_color_probability_l3544_354435


namespace NUMINAMATH_CALUDE_random_number_table_sampling_sequence_l3544_354487

/-- Represents the steps in the sampling process -/
inductive SamplingStep
  | AssignNumbers
  | ObtainSamples
  | SelectStartingNumber

/-- Represents a sequence of sampling steps -/
def SamplingSequence := List SamplingStep

/-- The correct sampling sequence -/
def correctSequence : SamplingSequence :=
  [SamplingStep.AssignNumbers, SamplingStep.SelectStartingNumber, SamplingStep.ObtainSamples]

/-- Checks if a given sequence is valid for random number table sampling -/
def isValidSequence (seq : SamplingSequence) : Prop :=
  seq = correctSequence

theorem random_number_table_sampling_sequence :
  isValidSequence correctSequence :=
sorry

end NUMINAMATH_CALUDE_random_number_table_sampling_sequence_l3544_354487


namespace NUMINAMATH_CALUDE_problem_statement_l3544_354428

theorem problem_statement (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49/((x - 3)^2) = 23 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3544_354428


namespace NUMINAMATH_CALUDE_quadratic_decreasing_condition_l3544_354464

/-- A quadratic function f(x) = -2x^2 + mx - 3 is decreasing on the interval [-1, +∞) if and only if m ≤ -4 -/
theorem quadratic_decreasing_condition (m : ℝ) :
  (∀ x : ℝ, x ≥ -1 → (∀ y : ℝ, y > x → -2*y^2 + m*y - 3 < -2*x^2 + m*x - 3)) ↔ m ≤ -4 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_decreasing_condition_l3544_354464


namespace NUMINAMATH_CALUDE_plate_distance_to_bottom_l3544_354418

/-- Given a square table with a round plate, if the distances from the plate to the top, left, and right edges
    of the table are 10, 63, and 20 units respectively, then the distance from the plate to the bottom edge
    of the table is 73 units. -/
theorem plate_distance_to_bottom (d : ℝ) :
  let top_distance : ℝ := 10
  let left_distance : ℝ := 63
  let right_distance : ℝ := 20
  let bottom_distance : ℝ := left_distance + right_distance - top_distance
  bottom_distance = 73 := by
  sorry


end NUMINAMATH_CALUDE_plate_distance_to_bottom_l3544_354418


namespace NUMINAMATH_CALUDE_complement_of_union_l3544_354463

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Set Nat := {2, 3, 5}

-- Define set N
def N : Set Nat := {4, 5}

-- Theorem statement
theorem complement_of_union (h : Set Nat → Set Nat → Set Nat) :
  h M N = {1, 6} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3544_354463


namespace NUMINAMATH_CALUDE_age_equation_solution_l3544_354492

/-- Given a person's current age of 50, prove that the equation
    5 * (A + 5) - 5 * (A - X) = A is satisfied when X = 5. -/
theorem age_equation_solution :
  let A : ℕ := 50
  let X : ℕ := 5
  5 * (A + 5) - 5 * (A - X) = A :=
by sorry

end NUMINAMATH_CALUDE_age_equation_solution_l3544_354492


namespace NUMINAMATH_CALUDE_new_person_weight_l3544_354482

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 3.5 →
  replaced_weight = 62 →
  initial_count * weight_increase + replaced_weight = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l3544_354482


namespace NUMINAMATH_CALUDE_product_change_l3544_354444

theorem product_change (a b : ℝ) (h : (a - 3) * (b + 3) - a * b = 900) : 
  a * b - (a + 3) * (b - 3) = 918 := by
sorry

end NUMINAMATH_CALUDE_product_change_l3544_354444


namespace NUMINAMATH_CALUDE_expansion_coefficient_l3544_354417

theorem expansion_coefficient (n : ℕ) : 
  ((-2)^n : ℤ) + ((-2)^(n-1) : ℤ) * n = -128 ↔ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l3544_354417


namespace NUMINAMATH_CALUDE_computer_contract_probability_l3544_354408

theorem computer_contract_probability (p_hardware : ℚ) (p_not_software : ℚ) (p_at_least_one : ℚ)
  (h1 : p_hardware = 3 / 4)
  (h2 : p_not_software = 3 / 5)
  (h3 : p_at_least_one = 5 / 6) :
  p_hardware + (1 - p_not_software) - p_at_least_one = 19 / 60 :=
by sorry

end NUMINAMATH_CALUDE_computer_contract_probability_l3544_354408


namespace NUMINAMATH_CALUDE_probability_no_distinct_roots_l3544_354459

def is_valid_pair (b c : ℤ) : Prop :=
  b.natAbs ≤ 4 ∧ c.natAbs ≤ 4 ∧ c ≥ 0

def has_distinct_real_roots (b c : ℤ) : Prop :=
  b^2 - 4*c > 0

def total_valid_pairs : ℕ := 45

def pairs_without_distinct_roots : ℕ := 27

theorem probability_no_distinct_roots :
  (pairs_without_distinct_roots : ℚ) / total_valid_pairs = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_probability_no_distinct_roots_l3544_354459


namespace NUMINAMATH_CALUDE_complement_B_intersect_A_l3544_354484

open Set

universe u

def U : Set ℝ := univ
def A : Set ℝ := {x | |x| < 1}
def B : Set ℝ := {x | x > -1/2}

theorem complement_B_intersect_A :
  (U \ B) ∩ A = {x : ℝ | -1 < x ∧ x ≤ -1/2} := by sorry

end NUMINAMATH_CALUDE_complement_B_intersect_A_l3544_354484


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l3544_354430

theorem arithmetic_progression_sum (n : ℕ) : 
  (n ≥ 3 ∧ n ≤ 14) ↔ 
  (n : ℝ) / 2 * (2 * 25 + (n - 1) * (-3)) ≥ 66 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l3544_354430


namespace NUMINAMATH_CALUDE_root_implies_k_value_l3544_354486

theorem root_implies_k_value (k : ℝ) : 
  (2^2 - 3*2 + k = 0) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_k_value_l3544_354486


namespace NUMINAMATH_CALUDE_no_solution_in_naturals_l3544_354420

theorem no_solution_in_naturals :
  ∀ (x y z t : ℕ), (15^x + 29^y + 43^z) % 7 ≠ (t^2) % 7 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_in_naturals_l3544_354420


namespace NUMINAMATH_CALUDE_find_t_l3544_354478

-- Define a decreasing function f on ℝ
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

-- Define the property that f passes through (0, 5) and (3, -1)
def passes_through_points (f : ℝ → ℝ) : Prop :=
  f 0 = 5 ∧ f 3 = -1

-- Define the solution set of |f(x+t)-2|<3
def solution_set (f : ℝ → ℝ) (t : ℝ) : Set ℝ :=
  {x : ℝ | |f (x + t) - 2| < 3}

-- State the theorem
theorem find_t (f : ℝ → ℝ) (t : ℝ) :
  is_decreasing f →
  passes_through_points f →
  solution_set f t = Set.Ioo (-1) 2 →
  t = 1 := by sorry

end NUMINAMATH_CALUDE_find_t_l3544_354478


namespace NUMINAMATH_CALUDE_window_width_is_24_inches_l3544_354460

/-- Represents the dimensions of a glass pane -/
structure GlassPane where
  width : ℝ
  height : ℝ

/-- Represents the configuration of a window -/
structure Window where
  pane : GlassPane
  num_columns : ℕ
  num_rows : ℕ
  border_width : ℝ

/-- Calculates the total width of the window -/
def total_width (w : Window) : ℝ :=
  w.num_columns * w.pane.width + (w.num_columns + 1) * w.border_width

/-- Theorem stating that the total width of the window is 24 inches -/
theorem window_width_is_24_inches (w : Window) 
  (h1 : w.pane.height / w.pane.width = 3 / 4)
  (h2 : w.border_width = 3)
  (h3 : w.num_columns = 3)
  (h4 : w.num_rows = 2) :
  total_width w = 24 := by
  sorry


end NUMINAMATH_CALUDE_window_width_is_24_inches_l3544_354460


namespace NUMINAMATH_CALUDE_stating_italian_regular_clock_coincidences_l3544_354404

/-- Represents a clock with specified rotations for hour and minute hands per day. -/
structure Clock :=
  (hour_rotations : ℕ)
  (minute_rotations : ℕ)

/-- The Italian clock with 1 hour hand rotation and 24 minute hand rotations per day. -/
def italian_clock : Clock :=
  { hour_rotations := 1, minute_rotations := 24 }

/-- The regular clock with 2 hour hand rotations and 24 minute hand rotations per day. -/
def regular_clock : Clock :=
  { hour_rotations := 2, minute_rotations := 24 }

/-- 
  The number of times the hands of an Italian clock coincide with 
  the hands of a regular clock in a 24-hour period.
-/
def coincidence_count (ic : Clock) (rc : Clock) : ℕ := sorry

/-- 
  Theorem stating that the number of hand coincidences between 
  the Italian clock and regular clock in a 24-hour period is 12.
-/
theorem italian_regular_clock_coincidences : 
  coincidence_count italian_clock regular_clock = 12 := by sorry

end NUMINAMATH_CALUDE_stating_italian_regular_clock_coincidences_l3544_354404


namespace NUMINAMATH_CALUDE_milk_for_12_cookies_l3544_354481

/-- The number of cookies that can be baked with 5 liters of milk -/
def cookies_per_5_liters : ℕ := 30

/-- The number of cups in a liter -/
def cups_per_liter : ℕ := 4

/-- The number of cookies we want to bake -/
def target_cookies : ℕ := 12

/-- The function that calculates the number of cups of milk needed for a given number of cookies -/
def milk_needed (cookies : ℕ) : ℚ :=
  (cookies * cups_per_liter * 5 : ℚ) / cookies_per_5_liters

theorem milk_for_12_cookies :
  milk_needed target_cookies = 8 := by sorry

end NUMINAMATH_CALUDE_milk_for_12_cookies_l3544_354481


namespace NUMINAMATH_CALUDE_fern_fronds_l3544_354432

theorem fern_fronds (total_ferns : ℕ) (total_leaves : ℕ) (leaves_per_frond : ℕ) 
  (h1 : total_ferns = 6)
  (h2 : total_leaves = 1260)
  (h3 : leaves_per_frond = 30) :
  (total_leaves / leaves_per_frond) / total_ferns = 7 := by
sorry

end NUMINAMATH_CALUDE_fern_fronds_l3544_354432


namespace NUMINAMATH_CALUDE_range_of_a_l3544_354476

/-- The function f(x) = ax^2 - 2x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2*x + 1

/-- f is decreasing on [1, +∞) -/
def f_decreasing (a : ℝ) : Prop := 
  ∀ x y, 1 ≤ x → x < y → f a y < f a x

/-- The range of a is (-∞, 0] -/
theorem range_of_a : 
  (∃ a, f_decreasing a) ↔ (∀ a, f_decreasing a → a ≤ 0) ∧ (∃ a ≤ 0, f_decreasing a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3544_354476


namespace NUMINAMATH_CALUDE_gcd_values_count_l3544_354470

theorem gcd_values_count (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (s : Finset ℕ), s.card = 6 ∧ ∀ (x : ℕ), x ∈ s ↔ ∃ (c d : ℕ+), Nat.gcd c d = x ∧ Nat.gcd c d * Nat.lcm c d = 360) :=
by sorry

end NUMINAMATH_CALUDE_gcd_values_count_l3544_354470


namespace NUMINAMATH_CALUDE_solar_eclipse_viewers_scientific_notation_l3544_354447

theorem solar_eclipse_viewers_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 2580000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.58 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_solar_eclipse_viewers_scientific_notation_l3544_354447


namespace NUMINAMATH_CALUDE_mean_home_runs_l3544_354440

def players_5 : ℕ := 4
def players_6 : ℕ := 3
def players_7 : ℕ := 2
def players_9 : ℕ := 1
def players_11 : ℕ := 1

def total_players : ℕ := players_5 + players_6 + players_7 + players_9 + players_11

def total_home_runs : ℕ := 5 * players_5 + 6 * players_6 + 7 * players_7 + 9 * players_9 + 11 * players_11

theorem mean_home_runs : 
  (total_home_runs : ℚ) / (total_players : ℚ) = 6.545454545 := by
  sorry

end NUMINAMATH_CALUDE_mean_home_runs_l3544_354440


namespace NUMINAMATH_CALUDE_exists_m_divides_f_100_l3544_354425

def f (x : ℤ) : ℤ := 3 * x + 2

theorem exists_m_divides_f_100 :
  ∃ m : ℕ+, 19881 ∣ (3^100 * (m.val + 1) - 1) :=
sorry

end NUMINAMATH_CALUDE_exists_m_divides_f_100_l3544_354425


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_104_l3544_354499

/-- A monic quadratic polynomial -/
structure MonicQuadratic where
  b : ℝ
  c : ℝ

/-- The sum of the roots of a polynomial -/
def sumOfRoots (roots : List ℝ) : ℝ :=
  roots.sum

theorem sum_of_coefficients_equals_104 
  (P Q : MonicQuadratic) 
  (h1 : sumOfRoots [-19, -13, -11, -7] = -50) 
  (h2 : sumOfRoots [-47, -43, -37, -31] = -158) : 
  P.b + Q.b = 104 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_104_l3544_354499


namespace NUMINAMATH_CALUDE_inequality_range_l3544_354429

theorem inequality_range (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ m : ℝ, (3 * x) / (2 * x + y) + (3 * y) / (x + 2 * y) ≤ m^2 + m) ↔ 
  (m ≤ -2 ∨ m ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l3544_354429


namespace NUMINAMATH_CALUDE_solution_exists_l3544_354466

theorem solution_exists : ∃ x : ℚ, 
  (10 / (Real.sqrt (x - 5) - 10) + 
   2 / (Real.sqrt (x - 5) - 5) + 
   9 / (Real.sqrt (x - 5) + 5) + 
   18 / (Real.sqrt (x - 5) + 10) = 0) ∧ 
  (x = 1230 / 121) := by
  sorry


end NUMINAMATH_CALUDE_solution_exists_l3544_354466


namespace NUMINAMATH_CALUDE_parallel_vectors_sin_cos_product_l3544_354442

/-- 
Given two vectors in the plane, a = (4, 3) and b = (sin α, cos α),
prove that if a is parallel to b, then sin α * cos α = 12/25.
-/
theorem parallel_vectors_sin_cos_product (α : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ (4, 3) = k • (Real.sin α, Real.cos α)) → 
  Real.sin α * Real.cos α = 12/25 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sin_cos_product_l3544_354442


namespace NUMINAMATH_CALUDE_rectangle_shading_l3544_354401

theorem rectangle_shading (length width : ℕ) (initial_shaded_fraction final_shaded_fraction : ℚ) :
  length = 15 →
  width = 20 →
  initial_shaded_fraction = 1 / 4 →
  final_shaded_fraction = 1 / 5 →
  (initial_shaded_fraction * final_shaded_fraction : ℚ) = 1 / 20 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_shading_l3544_354401


namespace NUMINAMATH_CALUDE_inequality_implication_l3544_354457

theorem inequality_implication (m n : ℝ) (h : m > n) : -3*m < -3*n := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l3544_354457


namespace NUMINAMATH_CALUDE_dog_grouping_theorem_l3544_354452

/-- The number of ways to divide 12 dogs into specified groups -/
def dog_grouping_ways : ℕ :=
  let total_dogs : ℕ := 12
  let group1_size : ℕ := 4  -- Fluffy's group
  let group2_size : ℕ := 5  -- Nipper's group
  let group3_size : ℕ := 3
  let remaining_dogs : ℕ := total_dogs - 2  -- Excluding Fluffy and Nipper
  let ways_to_fill_group1 : ℕ := Nat.choose remaining_dogs (group1_size - 1)
  let ways_to_fill_group2 : ℕ := Nat.choose (remaining_dogs - (group1_size - 1)) (group2_size - 1)
  ways_to_fill_group1 * ways_to_fill_group2

/-- Theorem stating the number of ways to divide the dogs into groups -/
theorem dog_grouping_theorem : dog_grouping_ways = 4200 := by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_theorem_l3544_354452


namespace NUMINAMATH_CALUDE_jose_bottle_caps_l3544_354445

/-- The number of bottle caps Jose ends up with after receiving more -/
def total_bottle_caps (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Jose ends up with 9 bottle caps -/
theorem jose_bottle_caps : total_bottle_caps 7 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jose_bottle_caps_l3544_354445


namespace NUMINAMATH_CALUDE_determine_M_value_l3544_354402

theorem determine_M_value (a b c d : ℤ) (M : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
  a + b + c + d = 0 →
  M = (b * c - a * d) * (a * c - b * d) * (a * b - c * d) →
  96100 < M ∧ M < 98000 →
  M = 97344 := by
  sorry

end NUMINAMATH_CALUDE_determine_M_value_l3544_354402


namespace NUMINAMATH_CALUDE_train_length_l3544_354422

theorem train_length (tree_time platform_time platform_length : ℝ) :
  tree_time = 120 →
  platform_time = 230 →
  platform_length = 1100 →
  ∃ (train_length : ℝ),
    train_length / tree_time = (train_length + platform_length) / platform_time ∧
    train_length = 1200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3544_354422


namespace NUMINAMATH_CALUDE_binary_sum_equals_result_l3544_354491

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binaryToNat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents the binary number 11011₂ -/
def num1 : List Bool := [true, true, false, true, true]

/-- Represents the binary number 1010₂ -/
def num2 : List Bool := [true, false, true, false]

/-- Represents the binary number 11100₂ -/
def num3 : List Bool := [true, true, true, false, false]

/-- Represents the binary number 1001₂ -/
def num4 : List Bool := [true, false, false, true]

/-- Represents the binary number 100010₂ (the expected result) -/
def result : List Bool := [true, false, false, false, true, false]

/-- The main theorem stating that the sum of the binary numbers equals the expected result -/
theorem binary_sum_equals_result :
  binaryToNat num1 + binaryToNat num2 - binaryToNat num3 + binaryToNat num4 = binaryToNat result :=
by sorry

end NUMINAMATH_CALUDE_binary_sum_equals_result_l3544_354491


namespace NUMINAMATH_CALUDE_final_result_proof_l3544_354498

def chosen_number : ℕ := 63
def multiplier : ℕ := 4
def subtrahend : ℕ := 142

theorem final_result_proof :
  chosen_number * multiplier - subtrahend = 110 := by
  sorry

end NUMINAMATH_CALUDE_final_result_proof_l3544_354498


namespace NUMINAMATH_CALUDE_f_at_two_l3544_354450

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom monotonic_increasing : Monotone f
axiom functional_equation : ∀ x : ℝ, f (f x - Real.exp x) = Real.exp 1 + 1

-- State the theorem
theorem f_at_two (f : ℝ → ℝ) (h1 : Monotone f) (h2 : ∀ x : ℝ, f (f x - Real.exp x) = Real.exp 1 + 1) :
  f 2 = Real.exp 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_at_two_l3544_354450


namespace NUMINAMATH_CALUDE_sum_remainder_l3544_354494

theorem sum_remainder (a b c d e : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9)
  (he : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l3544_354494


namespace NUMINAMATH_CALUDE_inequality_proof_l3544_354407

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_prod : (a + b) * (b + c) * (c + a) = 1) : 
  a^2 / (1 + Real.sqrt (b * c)) + 
  b^2 / (1 + Real.sqrt (c * a)) + 
  c^2 / (1 + Real.sqrt (a * b)) ≥ 1/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3544_354407


namespace NUMINAMATH_CALUDE_prob_even_and_divisible_by_three_on_two_dice_l3544_354461

/-- The probability of rolling an even number on a six-sided die -/
def prob_even_on_six_sided_die : ℚ := 1/2

/-- The probability of rolling a number divisible by three on a six-sided die -/
def prob_divisible_by_three_on_six_sided_die : ℚ := 1/3

/-- The probability of rolling an even number on one six-sided die
    and a number divisible by three on another six-sided die -/
theorem prob_even_and_divisible_by_three_on_two_dice :
  prob_even_on_six_sided_die * prob_divisible_by_three_on_six_sided_die = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_and_divisible_by_three_on_two_dice_l3544_354461


namespace NUMINAMATH_CALUDE_sum_of_three_smallest_primes_l3544_354414

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def primes_between_1_and_50 : Set ℕ := {n : ℕ | 1 < n ∧ n ≤ 50 ∧ is_prime n}

theorem sum_of_three_smallest_primes :
  ∃ (a b c : ℕ), a ∈ primes_between_1_and_50 ∧
                 b ∈ primes_between_1_and_50 ∧
                 c ∈ primes_between_1_and_50 ∧
                 a < b ∧ b < c ∧
                 (∀ p ∈ primes_between_1_and_50, p ≥ c ∨ p = a ∨ p = b) ∧
                 a + b + c = 10 :=
sorry

end NUMINAMATH_CALUDE_sum_of_three_smallest_primes_l3544_354414


namespace NUMINAMATH_CALUDE_marked_elements_not_unique_l3544_354413

/-- Represents the table with 4 rows and 10 columns --/
def Table := Fin 4 → Fin 10 → Fin 10

/-- The table where each row is shifted by one position --/
def shiftedTable : Table :=
  λ i j => (j + i) % 10

/-- A marking of 10 elements in the table --/
def Marking := Fin 10 → Fin 4 × Fin 10

/-- Predicate to check if a marking is valid (one per row and column) --/
def isValidMarking (m : Marking) : Prop :=
  (∀ i : Fin 4, ∃! j : Fin 10, (i, j) ∈ Set.range m) ∧
  (∀ j : Fin 10, ∃! i : Fin 4, (i, j) ∈ Set.range m)

theorem marked_elements_not_unique (t : Table) (m : Marking) 
  (h : isValidMarking m) : 
  ∃ i j : Fin 10, i ≠ j ∧ t (m i).1 (m i).2 = t (m j).1 (m j).2 :=
sorry

end NUMINAMATH_CALUDE_marked_elements_not_unique_l3544_354413


namespace NUMINAMATH_CALUDE_base_8_4532_equals_2394_l3544_354488

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_4532_equals_2394 :
  base_8_to_10 [2, 3, 5, 4] = 2394 := by
  sorry

end NUMINAMATH_CALUDE_base_8_4532_equals_2394_l3544_354488


namespace NUMINAMATH_CALUDE_megatech_budget_allocation_l3544_354409

theorem megatech_budget_allocation :
  let total_budget : ℝ := 100
  let microphotonics : ℝ := 10
  let home_electronics : ℝ := 24
  let food_additives : ℝ := 15
  let industrial_lubricants : ℝ := 8
  let basic_astrophysics_degrees : ℝ := 50.4
  let total_degrees : ℝ := 360
  let basic_astrophysics : ℝ := (basic_astrophysics_degrees / total_degrees) * total_budget
  let genetically_modified_microorganisms : ℝ := total_budget - (microphotonics + home_electronics + food_additives + industrial_lubricants + basic_astrophysics)
  genetically_modified_microorganisms = 29 := by
sorry

end NUMINAMATH_CALUDE_megatech_budget_allocation_l3544_354409


namespace NUMINAMATH_CALUDE_compound_molar_mass_l3544_354439

/-- Given that 8 moles of a compound weigh 1600 grams, prove that its molar mass is 200 grams/mole -/
theorem compound_molar_mass (mass : ℝ) (moles : ℝ) (h1 : mass = 1600) (h2 : moles = 8) :
  mass / moles = 200 := by
  sorry

end NUMINAMATH_CALUDE_compound_molar_mass_l3544_354439


namespace NUMINAMATH_CALUDE_area_ABC_is_72_l3544_354416

def X : ℝ × ℝ := (6, 0)
def Y : ℝ × ℝ := (8, 4)
def Z : ℝ × ℝ := (10, 0)

def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem area_ABC_is_72 :
  let area_XYZ := area_triangle X Y Z
  let area_ABC := area_XYZ / 0.1111111111111111
  area_ABC = 72 := by sorry

end NUMINAMATH_CALUDE_area_ABC_is_72_l3544_354416


namespace NUMINAMATH_CALUDE_absolute_value_equation_l3544_354471

theorem absolute_value_equation (a b c : ℝ) : 
  (∀ x y z : ℝ, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) ↔ 
  ((a = 1 ∧ b = 0 ∧ c = 0) ∨ 
   (a = -1 ∧ b = 0 ∧ c = 0) ∨ 
   (a = 0 ∧ b = 1 ∧ c = 0) ∨ 
   (a = 0 ∧ b = -1 ∧ c = 0) ∨ 
   (a = 0 ∧ b = 0 ∧ c = 1) ∨ 
   (a = 0 ∧ b = 0 ∧ c = -1)) := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l3544_354471


namespace NUMINAMATH_CALUDE_cloth_sales_worth_l3544_354433

-- Define the commission rate as a percentage
def commission_rate : ℚ := 2.5

-- Define the commission earned on a particular day
def commission_earned : ℚ := 15

-- Define the function to calculate the total sales
def total_sales (rate : ℚ) (commission : ℚ) : ℚ :=
  commission / (rate / 100)

-- Theorem statement
theorem cloth_sales_worth :
  total_sales commission_rate commission_earned = 600 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sales_worth_l3544_354433
