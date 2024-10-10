import Mathlib

namespace exists_complete_list_l3591_359156

/-- Represents a tournament where each competitor meets every other competitor exactly once with no draws -/
structure Tournament (α : Type*) :=
  (competitors : Set α)
  (beats : α → α → Prop)
  (all_play_once : ∀ x y : α, x ≠ y → (beats x y ∨ beats y x) ∧ ¬(beats x y ∧ beats y x))

/-- The list of players beaten by a given player and those beaten by the players they've beaten -/
def extended_wins (T : Tournament α) (x : α) : Set α :=
  {y | T.beats x y ∨ ∃ z, T.beats x z ∧ T.beats z y}

/-- There exists a player whose extended wins list includes all other players -/
theorem exists_complete_list (T : Tournament α) :
  ∃ x : α, ∀ y : α, y ≠ x → y ∈ extended_wins T x := by
  sorry


end exists_complete_list_l3591_359156


namespace final_book_count_is_1160_l3591_359131

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSequenceSum (a1 n d : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

/-- Represents Tracy's book store -/
structure BookStore where
  initialBooks : ℕ
  donators : ℕ
  firstDonation : ℕ
  donationIncrement : ℕ
  borrowedBooks : ℕ
  returnedBooks : ℕ

/-- Calculates the final number of books in the store -/
def finalBookCount (store : BookStore) : ℕ :=
  store.initialBooks +
  arithmeticSequenceSum store.firstDonation store.donators store.donationIncrement -
  store.borrowedBooks +
  store.returnedBooks

/-- Theorem stating that the final book count is 1160 -/
theorem final_book_count_is_1160 (store : BookStore)
  (h1 : store.initialBooks = 1000)
  (h2 : store.donators = 15)
  (h3 : store.firstDonation = 2)
  (h4 : store.donationIncrement = 2)
  (h5 : store.borrowedBooks = 350)
  (h6 : store.returnedBooks = 270) :
  finalBookCount store = 1160 := by
  sorry


end final_book_count_is_1160_l3591_359131


namespace symmetric_circle_correct_l3591_359190

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y - 11 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-2, 1)

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 5)^2 + y^2 = 16

-- Theorem statement
theorem symmetric_circle_correct :
  ∀ (x y : ℝ),
  symmetric_circle x y ↔
  ∃ (x₀ y₀ : ℝ),
    original_circle x₀ y₀ ∧
    x = 2 * point_P.1 - x₀ ∧
    y = 2 * point_P.2 - y₀ :=
by sorry

end symmetric_circle_correct_l3591_359190


namespace convex_prism_right_iff_not_four_l3591_359152

/-- A convex n-sided prism with congruent lateral faces -/
structure ConvexPrism (n : ℕ) where
  /-- The prism is convex -/
  convex : Bool
  /-- The prism has n sides -/
  sides : Fin n
  /-- All lateral faces are congruent -/
  congruentLateralFaces : Bool

/-- A prism is right if all its lateral edges are perpendicular to its bases -/
def isRight (p : ConvexPrism n) : Prop := sorry

/-- Main theorem: A convex n-sided prism with congruent lateral faces is necessarily right if and only if n ≠ 4 -/
theorem convex_prism_right_iff_not_four (n : ℕ) (p : ConvexPrism n) :
  p.convex ∧ p.congruentLateralFaces → (isRight p ↔ n ≠ 4) := by sorry

end convex_prism_right_iff_not_four_l3591_359152


namespace perpendicular_line_through_point_l3591_359166

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem perpendicular_line_through_point :
  let l1 : Line := { a := 2, b := 3, c := -4 }
  let l2 : Line := { a := 3, b := -2, c := 2 }
  perpendicular l1 l2 ∧ point_on_line 0 1 l2 :=
by sorry

end perpendicular_line_through_point_l3591_359166


namespace range_of_m_l3591_359178

/-- The function f(x) defined as 1 / √(mx² + mx + 1) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 1 / Real.sqrt (m * x^2 + m * x + 1)

/-- The set of real numbers m for which f(x) has domain ℝ -/
def valid_m : Set ℝ := {m : ℝ | ∀ x : ℝ, m * x^2 + m * x + 1 > 0}

theorem range_of_m : valid_m = Set.Ici 0 ∩ Set.Iio 4 := by sorry

end range_of_m_l3591_359178


namespace sum_of_four_sequential_terms_l3591_359187

theorem sum_of_four_sequential_terms (n : ℝ) : 
  n + (n + 1) + (n + 2) + (n + 3) = 20 → n = 3.5 := by
  sorry

end sum_of_four_sequential_terms_l3591_359187


namespace ice_cream_sundae_combinations_l3591_359153

/-- The number of unique two-scoop sundae combinations given a total number of flavors and vanilla as a required flavor. -/
def sundae_combinations (total_flavors : ℕ) (vanilla_required : Bool) : ℕ :=
  if vanilla_required then total_flavors - 1 else 0

/-- Theorem: Given 8 ice cream flavors with vanilla required, the number of unique two-scoop sundae combinations is 7. -/
theorem ice_cream_sundae_combinations :
  sundae_combinations 8 true = 7 := by
  sorry

end ice_cream_sundae_combinations_l3591_359153


namespace expand_product_l3591_359147

theorem expand_product (y : ℝ) : (y + 3) * (y + 7) = y^2 + 10*y + 21 := by
  sorry

end expand_product_l3591_359147


namespace lcm_of_36_and_105_l3591_359130

theorem lcm_of_36_and_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_of_36_and_105_l3591_359130


namespace probability_divisible_by_8_l3591_359160

def is_valid_digit (d : ℕ) : Prop := d ∈ ({3, 58} : Set ℕ)

def form_number (x y : ℕ) : ℕ := 460000 + x * 1000 + y * 100 + 12

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

theorem probability_divisible_by_8 :
  ∀ x y : ℕ, is_valid_digit x → is_valid_digit y →
  (∃! y', is_valid_digit y' ∧ is_divisible_by_8 (form_number x y')) :=
sorry

end probability_divisible_by_8_l3591_359160


namespace joan_football_games_l3591_359181

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := sorry

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := 9

/-- The total number of football games Joan went to -/
def total_games : ℕ := 13

theorem joan_football_games : games_this_year = 4 := by sorry

end joan_football_games_l3591_359181


namespace sequence_properties_l3591_359167

/-- The sum of the first n terms of sequence a_n -/
def S (n : ℕ) : ℕ := 2^(n+1) - 2

/-- The n-th term of sequence a_n -/
def a (n : ℕ) : ℕ := 2^n

/-- The n-th term of sequence b_n -/
def b (n : ℕ) : ℕ := n * a n

/-- The sum of the first n terms of sequence b_n -/
def T (n : ℕ) : ℕ := (n-1) * 2^(n+1) + 2

theorem sequence_properties (n : ℕ) :
  (∀ k, S k = 2^(k+1) - 2) →
  (∀ k, a k = 2^k) ∧
  T n = (n-1) * 2^(n+1) + 2 :=
by sorry

end sequence_properties_l3591_359167


namespace least_square_tiles_l3591_359101

/-- Given a rectangular room with length 624 cm and width 432 cm, 
    the least number of square tiles of equal size required to cover the entire floor is 117. -/
theorem least_square_tiles (length width : ℕ) (h1 : length = 624) (h2 : width = 432) : 
  (length / (Nat.gcd length width)) * (width / (Nat.gcd length width)) = 117 := by
  sorry

end least_square_tiles_l3591_359101


namespace visual_range_increase_l3591_359151

theorem visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 100)
  (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 50 := by
  sorry

end visual_range_increase_l3591_359151


namespace total_blocks_is_2250_l3591_359125

/-- Represents the size of a dog -/
inductive DogSize
  | Small
  | Medium
  | Large

/-- Represents the walking speed of a dog in blocks per 10 minutes -/
def walkingSpeed (size : DogSize) : ℕ :=
  match size with
  | .Small => 3
  | .Medium => 4
  | .Large => 2

/-- Represents the number of dogs of each size -/
def dogCounts : DogSize → ℕ
  | .Small => 10
  | .Medium => 8
  | .Large => 7

/-- The total vacation cost in dollars -/
def vacationCost : ℕ := 1200

/-- The number of family members -/
def familyMembers : ℕ := 5

/-- The total available time in minutes -/
def totalAvailableTime : ℕ := 8 * 60

/-- The break time in minutes -/
def breakTime : ℕ := 30

/-- Calculates the total number of blocks Jules has to walk -/
def totalBlocks : ℕ :=
  let availableTime := totalAvailableTime - breakTime
  let slowestSpeed := walkingSpeed DogSize.Large
  let blocksPerDog := (availableTime / 10) * slowestSpeed
  (dogCounts DogSize.Small + dogCounts DogSize.Medium + dogCounts DogSize.Large) * blocksPerDog

theorem total_blocks_is_2250 : totalBlocks = 2250 := by
  sorry

#eval totalBlocks

end total_blocks_is_2250_l3591_359125


namespace gcd_count_for_product_504_l3591_359186

theorem gcd_count_for_product_504 (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 504) :
  ∃! (s : Finset ℕ), s.card = 9 ∧ ∀ d, d ∈ s ↔ ∃ (a' b' : ℕ+), Nat.gcd a' b' * Nat.lcm a' b' = 504 ∧ Nat.gcd a' b' = d :=
sorry

end gcd_count_for_product_504_l3591_359186


namespace range_of_a_l3591_359157

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}

-- State the theorem
theorem range_of_a (a : ℝ) (h : A ⊆ B a) : 0 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l3591_359157


namespace road_trip_speed_l3591_359176

/-- Road trip problem -/
theorem road_trip_speed (total_distance : ℝ) (jenna_distance : ℝ) (friend_distance : ℝ)
  (jenna_speed : ℝ) (total_time : ℝ) (num_breaks : ℕ) (break_duration : ℝ) :
  total_distance = jenna_distance + friend_distance →
  jenna_distance = 200 →
  friend_distance = 100 →
  jenna_speed = 50 →
  total_time = 10 →
  num_breaks = 2 →
  break_duration = 0.5 →
  ∃ (friend_speed : ℝ), friend_speed = 20 ∧ 
    total_time = jenna_distance / jenna_speed + friend_distance / friend_speed + num_breaks * break_duration :=
by sorry


end road_trip_speed_l3591_359176


namespace warehouse_construction_l3591_359137

/-- Warehouse construction problem -/
theorem warehouse_construction (investment : ℝ) (front_cost side_cost top_cost : ℝ) 
  (h_investment : investment = 3200)
  (h_front_cost : front_cost = 40)
  (h_side_cost : side_cost = 45)
  (h_top_cost : top_cost = 20) :
  ∃ (x y : ℝ),
    0 < x ∧ x < 80 ∧
    y = (320 - 4*x) / (9 + 2*x) ∧
    x * y ≤ 100 ∧
    (∀ x' y' : ℝ, 0 < x' ∧ x' < 80 ∧ y' = (320 - 4*x') / (9 + 2*x') → x' * y' ≤ x * y) ∧
    x = 15 ∧ y = 20/3 := by
  sorry

end warehouse_construction_l3591_359137


namespace max_utility_problem_l3591_359165

theorem max_utility_problem (s : ℝ) : 
  s ≥ 0 ∧ s * (10 - s) = (3 - s) * (s + 4) → s = 0 := by
  sorry

end max_utility_problem_l3591_359165


namespace athletes_count_is_ten_l3591_359124

-- Define the types for our counts
def TotalLegs : ℕ := 108
def TotalHeads : ℕ := 32

-- Define a structure to represent the counts of each animal type
structure AnimalCounts where
  athletes : ℕ
  elephants : ℕ
  monkeys : ℕ

-- Define the property that the counts satisfy the given conditions
def satisfiesConditions (counts : AnimalCounts) : Prop :=
  2 * counts.athletes + 4 * counts.elephants + 2 * counts.monkeys = TotalLegs ∧
  counts.athletes + counts.elephants + counts.monkeys = TotalHeads

-- The theorem to prove
theorem athletes_count_is_ten :
  ∃ (counts : AnimalCounts), satisfiesConditions counts ∧ counts.athletes = 10 :=
by sorry

end athletes_count_is_ten_l3591_359124


namespace wong_valentines_l3591_359104

/-- Mrs. Wong's Valentine problem -/
theorem wong_valentines (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  initial = 30 → given_away = 8 → remaining = initial - given_away → remaining = 22 := by
  sorry

end wong_valentines_l3591_359104


namespace roses_picked_second_correct_l3591_359106

-- Define the problem parameters
def initial_roses : ℝ := 37.0
def first_picking : ℝ := 16.0
def final_total : ℕ := 72

-- Define the function to calculate roses picked in the second picking
def roses_picked_second (initial : ℝ) (first : ℝ) (total : ℕ) : ℝ :=
  (total : ℝ) - (initial + first)

-- Theorem statement
theorem roses_picked_second_correct :
  roses_picked_second initial_roses first_picking final_total = 19.0 := by
  sorry

end roses_picked_second_correct_l3591_359106


namespace hispanic_west_percentage_l3591_359119

/-- Represents the population data for a specific ethnic group across regions -/
structure PopulationData :=
  (ne : ℕ) (mw : ℕ) (south : ℕ) (west : ℕ)

/-- Calculates the total population across all regions -/
def total_population (data : PopulationData) : ℕ :=
  data.ne + data.mw + data.south + data.west

/-- Calculates the percentage of population in the West, rounded to the nearest percent -/
def west_percentage (data : PopulationData) : ℕ :=
  (data.west * 100 + (total_population data) / 2) / (total_population data)

/-- The given Hispanic population data for 1990 in millions -/
def hispanic_data : PopulationData :=
  { ne := 4, mw := 5, south := 12, west := 20 }

theorem hispanic_west_percentage :
  west_percentage hispanic_data = 49 := by sorry

end hispanic_west_percentage_l3591_359119


namespace paper_covers_cube_l3591_359180

theorem paper_covers_cube (cube_edge : ℝ) (paper_side : ℝ) 
  (h1 : cube_edge = 1) (h2 : paper_side = 2.5) : 
  paper_side ^ 2 ≥ 6 * cube_edge ^ 2 := by
  sorry

end paper_covers_cube_l3591_359180


namespace polynomial_divisibility_l3591_359168

theorem polynomial_divisibility (m n p : ℕ) : 
  ∃ q : Polynomial ℤ, (X^2 + X + 1) * q = X^(3*m) + X^(n+1) + X^(p+2) := by
  sorry

end polynomial_divisibility_l3591_359168


namespace winter_clothing_count_l3591_359127

/-- The number of boxes of clothing -/
def num_boxes : ℕ := 6

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 5

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 5

/-- The total number of pieces of winter clothing -/
def total_clothing : ℕ := num_boxes * (scarves_per_box + mittens_per_box)

theorem winter_clothing_count : total_clothing = 60 := by
  sorry

end winter_clothing_count_l3591_359127


namespace reflex_angle_at_H_l3591_359196

-- Define the points
variable (C D F M H : Point)

-- Define the angles
def angle_CDH : ℝ := 150
def angle_HFM : ℝ := 95

-- Define the properties
def collinear (C D F M : Point) : Prop := sorry
def angle (A B C : Point) : ℝ := sorry
def reflex_angle (A : Point) : ℝ := sorry

-- State the theorem
theorem reflex_angle_at_H (h_collinear : collinear C D F M) 
  (h_CDH : angle C D H = angle_CDH)
  (h_HFM : angle H F M = angle_HFM) : 
  reflex_angle H = 180 := by sorry

end reflex_angle_at_H_l3591_359196


namespace potatoes_left_l3591_359191

theorem potatoes_left (initial : ℕ) (salad : ℕ) (mashed : ℕ) (h1 : initial = 52) (h2 : salad = 15) (h3 : mashed = 24) : initial - (salad + mashed) = 13 := by
  sorry

end potatoes_left_l3591_359191


namespace new_shipment_bears_l3591_359171

/-- Calculates the number of bears in a new shipment given the initial stock,
    bears per shelf, and number of shelves used. -/
theorem new_shipment_bears (initial_stock : ℕ) (bears_per_shelf : ℕ) (shelves_used : ℕ) :
  (bears_per_shelf * shelves_used) - initial_stock =
  (bears_per_shelf * shelves_used) - initial_stock :=
by sorry

end new_shipment_bears_l3591_359171


namespace justin_tim_games_l3591_359185

theorem justin_tim_games (n : ℕ) (h : n = 12) :
  let total_combinations := Nat.choose n 6
  let games_with_both := Nat.choose (n - 2) 4
  games_with_both = 210 ∧ 
  2 * games_with_both = total_combinations := by
  sorry

end justin_tim_games_l3591_359185


namespace quadratic_two_distinct_roots_l3591_359194

/-- The quadratic equation x^2 + 4x - 4 = 0 has two distinct real roots -/
theorem quadratic_two_distinct_roots : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ - 4 = 0 ∧ x₂^2 + 4*x₂ - 4 = 0 := by
  sorry

end quadratic_two_distinct_roots_l3591_359194


namespace diagonals_bisect_implies_parallelogram_parallelogram_right_angle_implies_rectangle_l3591_359107

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop := sorry

-- Define the property of diagonals bisecting each other
def diagonals_bisect (q : Quadrilateral) : Prop := sorry

-- Define the property of having a right angle
def has_right_angle (q : Quadrilateral) : Prop := sorry

-- Theorem 1: If a quadrilateral has diagonals that bisect each other, then it is a parallelogram
theorem diagonals_bisect_implies_parallelogram (q : Quadrilateral) :
  diagonals_bisect q → is_parallelogram q :=
sorry

-- Theorem 2: If a parallelogram has one right angle, then it is a rectangle
theorem parallelogram_right_angle_implies_rectangle (q : Quadrilateral) :
  is_parallelogram q → has_right_angle q → is_rectangle q :=
sorry

end diagonals_bisect_implies_parallelogram_parallelogram_right_angle_implies_rectangle_l3591_359107


namespace perpendicular_line_equation_l3591_359199

/-- Given a line L₁: Ax + By + C = 0 and a point P₀(x₀, y₀), 
    the line L₂ passing through P₀ and perpendicular to L₁ 
    has the equation Bx - Ay - Bx₀ + Ay₀ = 0 -/
theorem perpendicular_line_equation (A B C x₀ y₀ : ℝ) :
  let L₁ := fun (x y : ℝ) ↦ A * x + B * y + C = 0
  let P₀ := (x₀, y₀)
  let L₂ := fun (x y : ℝ) ↦ B * x - A * y - B * x₀ + A * y₀ = 0
  (∀ x y, L₂ x y ↔ (x - x₀) * B = (y - y₀) * A) ∧
  (∀ x₁ y₁ x₂ y₂, L₁ x₁ y₁ ∧ L₁ x₂ y₂ → (x₂ - x₁) * B = -(y₂ - y₁) * A) ∧
  L₂ x₀ y₀ :=
by
  sorry

end perpendicular_line_equation_l3591_359199


namespace angle_of_inclination_range_l3591_359133

theorem angle_of_inclination_range (θ : ℝ) :
  let α := Real.arctan (1 / (Real.sin θ))
  (∃ x y, x - y * Real.sin θ + 1 = 0) →
  π / 4 ≤ α ∧ α ≤ 3 * π / 4 :=
by sorry

end angle_of_inclination_range_l3591_359133


namespace m_range_l3591_359172

/-- Proposition p: For all x ∈ ℝ, |x| + x ≥ 0 -/
def prop_p : Prop := ∀ x : ℝ, |x| + x ≥ 0

/-- Proposition q: The equation x² + mx + 1 = 0 has real roots -/
def prop_q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m*x + 1 = 0

/-- The composite proposition "p ∧ q" is false -/
axiom p_and_q_false : ∀ m : ℝ, ¬(prop_p ∧ prop_q m)

/-- The main theorem: Given the conditions above, prove that -2 < m < 2 -/
theorem m_range : ∀ m : ℝ, (¬(prop_p ∧ prop_q m)) → -2 < m ∧ m < 2 := by
  sorry

end m_range_l3591_359172


namespace sarahs_coin_box_l3591_359109

/-- The number of pennies in Sarah's box --/
def num_coins : ℕ := 36

/-- The total value of coins in cents --/
def total_value : ℕ := 2000

/-- Theorem stating that the number of each type of coin in Sarah's box is 36,
    given that the total value is $20 (2000 cents) and there are equal numbers
    of pennies, nickels, and half-dollars. --/
theorem sarahs_coin_box :
  (num_coins : ℚ) * (1 + 5 + 50) = total_value :=
sorry

end sarahs_coin_box_l3591_359109


namespace brick_height_is_6cm_l3591_359173

/-- Proves that the height of each brick is 6 cm given the wall dimensions,
    brick base dimensions, and the number of bricks needed. -/
theorem brick_height_is_6cm 
  (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ)
  (brick_length : ℝ) (brick_width : ℝ) (num_bricks : ℕ) :
  wall_length = 750 →
  wall_width = 600 →
  wall_height = 22.5 →
  brick_length = 25 →
  brick_width = 11.25 →
  num_bricks = 6000 →
  ∃ (brick_height : ℝ), 
    wall_length * wall_width * wall_height = 
    (brick_length * brick_width * brick_height) * num_bricks ∧
    brick_height = 6 :=
by sorry

end brick_height_is_6cm_l3591_359173


namespace thirtieth_sum_l3591_359102

/-- Represents the sum of elements in the nth set of a sequence where each set starts one more than
    the last element of the preceding set and has one more element than the one before it. -/
def T (n : ℕ) : ℕ :=
  let first := 1 + (n * (n - 1)) / 2
  let last := first + n - 1
  n * (first + last) / 2

/-- The 30th sum in the sequence equals 13515. -/
theorem thirtieth_sum : T 30 = 13515 := by
  sorry

end thirtieth_sum_l3591_359102


namespace trajectory_equation_l3591_359126

/-- Given m ∈ ℝ, vector a = (mx, y+1), vector b = (x, y-1), and a ⊥ b,
    the equation of trajectory E for moving point M(x,y) is mx² + y² = 1 -/
theorem trajectory_equation (m : ℝ) (x y : ℝ) 
    (h : (m * x) * x + (y + 1) * (y - 1) = 0) : 
  m * x^2 + y^2 = 1 := by
sorry

end trajectory_equation_l3591_359126


namespace fraction_squared_equality_l3591_359118

theorem fraction_squared_equality : ((-123456789 : ℤ) / 246913578)^2 = (15241578750190521 : ℚ) / 60995928316126584 := by
  sorry

end fraction_squared_equality_l3591_359118


namespace barbed_wire_rate_l3591_359192

/-- The rate of drawing barbed wire per meter given the conditions of the problem -/
theorem barbed_wire_rate (field_area : ℝ) (wire_extension : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ)
  (h1 : field_area = 3136)
  (h2 : wire_extension = 3)
  (h3 : gate_width = 1)
  (h4 : num_gates = 2)
  (h5 : total_cost = 732.6) :
  (total_cost / (4 * Real.sqrt field_area + wire_extension - num_gates * gate_width)) = 3.256 := by
  sorry

end barbed_wire_rate_l3591_359192


namespace min_draw_for_eight_same_color_l3591_359158

/-- Represents the number of balls of each color in the bag -/
structure BallCounts where
  red : Nat
  green : Nat
  blue : Nat
  yellow : Nat
  white : Nat

/-- The minimum number of balls to draw to ensure at least n of the same color -/
def minDrawToEnsure (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The theorem to prove -/
theorem min_draw_for_eight_same_color (counts : BallCounts)
    (h_red : counts.red = 15)
    (h_green : counts.green = 12)
    (h_blue : counts.blue = 10)
    (h_yellow : counts.yellow = 7)
    (h_white : counts.white = 6)
    (h_total : counts.red + counts.green + counts.blue + counts.yellow + counts.white = 50) :
    minDrawToEnsure counts 8 = 35 := by
  sorry

end min_draw_for_eight_same_color_l3591_359158


namespace ellipse_eccentricity_l3591_359123

theorem ellipse_eccentricity (C2 : Set (ℝ × ℝ)) : 
  (∀ x y, (x = Real.sqrt 5 ∧ y = 0) ∨ (x = 0 ∧ y = 3) → (x, y) ∈ C2) →
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    (∀ x y, (x, y) ∈ C2 ↔ x^2/a^2 + y^2/b^2 = 1) ∧
    c^2 = a^2 - b^2 ∧
    c/a = 2/3) :=
by sorry

end ellipse_eccentricity_l3591_359123


namespace initial_milk_amount_l3591_359177

/-- Proves that the initial amount of milk is 10 liters given the conditions of the problem -/
theorem initial_milk_amount (initial_water_content : Real) 
                            (final_water_content : Real)
                            (pure_milk_added : Real) :
  initial_water_content = 0.05 →
  final_water_content = 0.02 →
  pure_milk_added = 15 →
  ∃ (initial_milk : Real),
    initial_milk = 10 ∧
    initial_water_content * initial_milk = 
    final_water_content * (initial_milk + pure_milk_added) :=
by
  sorry


end initial_milk_amount_l3591_359177


namespace fraction_equality_l3591_359174

theorem fraction_equality : (1 - 1/4) / (1 - 1/5) = 15/16 := by
  sorry

end fraction_equality_l3591_359174


namespace savings_account_balance_l3591_359183

theorem savings_account_balance (initial_amount : ℝ) 
  (increase_percentage : ℝ) (decrease_percentage : ℝ) : 
  initial_amount = 125 ∧ 
  increase_percentage = 0.25 ∧ 
  decrease_percentage = 0.20 →
  initial_amount = initial_amount * (1 + increase_percentage) * (1 - decrease_percentage) :=
by sorry

end savings_account_balance_l3591_359183


namespace tan_100_degrees_l3591_359103

theorem tan_100_degrees (k : ℝ) (h : Real.sin (-(80 * π / 180)) = k) :
  Real.tan ((100 * π / 180)) = k / Real.sqrt (1 - k^2) := by
  sorry

end tan_100_degrees_l3591_359103


namespace sum_range_l3591_359129

theorem sum_range (x y z : ℝ) 
  (eq1 : x + 2*y + 3*z = 1) 
  (eq2 : y*z + z*x + x*y = -1) : 
  (3 - 3*Real.sqrt 3) / 4 ≤ x + y + z ∧ x + y + z ≤ (3 + 3*Real.sqrt 3) / 4 :=
sorry

end sum_range_l3591_359129


namespace jacket_dimes_count_l3591_359138

/-- The value of a single dime in dollars -/
def dime_value : ℚ := 1 / 10

/-- The total amount of money found in dollars -/
def total_money : ℚ := 19 / 10

/-- The number of dimes found in the shorts -/
def dimes_in_shorts : ℕ := 4

/-- The number of dimes found in the jacket -/
def dimes_in_jacket : ℕ := 15

theorem jacket_dimes_count :
  dimes_in_jacket * dime_value + dimes_in_shorts * dime_value = total_money :=
by sorry

end jacket_dimes_count_l3591_359138


namespace tan_ratio_sum_l3591_359120

theorem tan_ratio_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 1/2 := by
  sorry

end tan_ratio_sum_l3591_359120


namespace solution_set_x_abs_x_less_x_l3591_359162

theorem solution_set_x_abs_x_less_x :
  {x : ℝ | x * |x| < x} = {x : ℝ | 0 < x ∧ x < 1} ∪ {x : ℝ | x < -1} := by
  sorry

end solution_set_x_abs_x_less_x_l3591_359162


namespace iains_pennies_l3591_359154

theorem iains_pennies (initial_pennies : ℕ) (older_pennies : ℕ) (discard_percentage : ℚ) : 
  initial_pennies = 200 →
  older_pennies = 30 →
  discard_percentage = 1/5 →
  initial_pennies - older_pennies - (initial_pennies - older_pennies) * discard_percentage = 136 := by
  sorry

end iains_pennies_l3591_359154


namespace triangle_property_l3591_359140

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ 
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) ∧
  ((c - 2*a) * Real.cos B + b * Real.cos C = 0) →
  (B = π/3) ∧
  (a + b + c = 6 ∧ b = 2 → 
    1/2 * a * c * Real.sin B = Real.sqrt 3) :=
by sorry

end triangle_property_l3591_359140


namespace expression_evaluation_l3591_359142

theorem expression_evaluation : 
  (2015^3 - 2 * 2015^2 * 2016 + 3 * 2015 * 2016^2 - 2016^3 + 1) / (2015 * 2016) = 0 := by
  sorry

end expression_evaluation_l3591_359142


namespace f_derivative_l3591_359195

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (x + 1)^2 - (x + 1)

-- State the theorem
theorem f_derivative : 
  ∀ x : ℝ, deriv f x = 4 * x + 3 := by sorry

end f_derivative_l3591_359195


namespace jasmine_needs_seven_cans_l3591_359150

/-- Represents the paint coverage problem for Jasmine --/
def paint_coverage_problem (initial_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) (rooms_per_new_can : ℕ) (total_rooms : ℕ) : Prop :=
  ∃ (additional_cans : ℕ),
    remaining_rooms + additional_cans * rooms_per_new_can = total_rooms

/-- Theorem stating that 7 additional cans are needed to cover all rooms --/
theorem jasmine_needs_seven_cans :
  paint_coverage_problem 50 4 36 2 50 →
  ∃ (additional_cans : ℕ), additional_cans = 7 ∧ 36 + additional_cans * 2 = 50 := by
  sorry

#check jasmine_needs_seven_cans

end jasmine_needs_seven_cans_l3591_359150


namespace nedy_crackers_total_l3591_359159

/-- The number of packs of crackers Nedy eats per day from Monday to Thursday -/
def daily_crackers : ℕ := 8

/-- The number of days from Monday to Thursday -/
def weekdays : ℕ := 4

/-- The factor by which Nedy increases his cracker consumption on Friday -/
def friday_factor : ℕ := 2

/-- Theorem: Given Nedy eats 8 packs of crackers per day from Monday to Thursday
    and twice that amount on Friday, the total number of crackers Nedy eats
    from Monday to Friday is 48 packs. -/
theorem nedy_crackers_total :
  daily_crackers * weekdays + daily_crackers * friday_factor = 48 := by
  sorry

end nedy_crackers_total_l3591_359159


namespace statue_weight_calculation_l3591_359141

/-- The weight of a marble statue after three weeks of carving --/
def final_statue_weight (initial_weight : ℝ) (cut_week1 : ℝ) (cut_week2 : ℝ) (cut_week3 : ℝ) : ℝ :=
  initial_weight * (1 - cut_week1) * (1 - cut_week2) * (1 - cut_week3)

/-- Theorem stating the final weight of the statue --/
theorem statue_weight_calculation :
  let initial_weight : ℝ := 180
  let cut_week1 : ℝ := 0.28
  let cut_week2 : ℝ := 0.18
  let cut_week3 : ℝ := 0.20
  final_statue_weight initial_weight cut_week1 cut_week2 cut_week3 = 85.0176 := by
  sorry

end statue_weight_calculation_l3591_359141


namespace winning_bet_amount_l3591_359148

def initial_amount : ℕ := 400

def bet_multiplier : ℕ := 2

theorem winning_bet_amount (initial : ℕ) (multiplier : ℕ) :
  initial = initial_amount →
  multiplier = bet_multiplier →
  initial + (multiplier * initial) = 1200 := by
  sorry

end winning_bet_amount_l3591_359148


namespace anna_remaining_money_l3591_359189

-- Define the given values
def initial_money : ℝ := 50
def gum_price : ℝ := 1.50
def gum_quantity : ℕ := 4
def chocolate_price : ℝ := 2.25
def chocolate_quantity : ℕ := 7
def candy_cane_price : ℝ := 0.75
def candy_cane_quantity : ℕ := 3
def jelly_beans_original_price : ℝ := 3.00
def jelly_beans_discount : ℝ := 0.20
def sales_tax_rate : ℝ := 0.075

-- Calculate the total cost and remaining money
def calculate_remaining_money : ℝ :=
  let gum_cost := gum_price * gum_quantity
  let chocolate_cost := chocolate_price * chocolate_quantity
  let candy_cane_cost := candy_cane_price * candy_cane_quantity
  let jelly_beans_cost := jelly_beans_original_price * (1 - jelly_beans_discount)
  let total_before_tax := gum_cost + chocolate_cost + candy_cane_cost + jelly_beans_cost
  let total_after_tax := total_before_tax * (1 + sales_tax_rate)
  initial_money - total_after_tax

-- Theorem to prove
theorem anna_remaining_money :
  calculate_remaining_money = 21.62 := by sorry

end anna_remaining_money_l3591_359189


namespace simplify_trig_expression_l3591_359193

theorem simplify_trig_expression (α : ℝ) :
  3 - 4 * Real.cos (4 * α) + Real.cos (8 * α) - 8 * (Real.cos (2 * α))^4 = -8 * Real.cos (4 * α) := by
  sorry

end simplify_trig_expression_l3591_359193


namespace function_value_l3591_359128

theorem function_value (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 1) : f 4 = 8 := by
  sorry

end function_value_l3591_359128


namespace roberts_gre_preparation_time_l3591_359179

/-- Represents the preparation time for each subject in the GRE examination -/
structure GREPreparation where
  vocabulary : Nat
  writing : Nat
  quantitative : Nat

/-- Calculates the total preparation time for the GRE examination -/
def totalPreparationTime (prep : GREPreparation) : Nat :=
  prep.vocabulary + prep.writing + prep.quantitative

/-- Theorem: The total preparation time for Robert's GRE examination is 8 months -/
theorem roberts_gre_preparation_time :
  let robert_prep : GREPreparation := ⟨3, 2, 3⟩
  totalPreparationTime robert_prep = 8 := by
  sorry

#check roberts_gre_preparation_time

end roberts_gre_preparation_time_l3591_359179


namespace fifth_month_sales_l3591_359144

def sales_1 : ℕ := 6435
def sales_2 : ℕ := 6927
def sales_3 : ℕ := 6855
def sales_4 : ℕ := 7230
def sales_6 : ℕ := 4991
def average_sale : ℕ := 6500
def num_months : ℕ := 6

theorem fifth_month_sales :
  ∃ (sales_5 : ℕ),
    sales_5 = average_sale * num_months - (sales_1 + sales_2 + sales_3 + sales_4 + sales_6) ∧
    sales_5 = 6562 :=
by sorry

end fifth_month_sales_l3591_359144


namespace teacher_selection_probability_l3591_359113

/-- Represents a university department -/
structure Department where
  name : String
  teachers : ℕ

/-- Represents a university -/
structure University where
  departments : List Department

/-- Calculates the total number of teachers in a university -/
def totalTeachers (u : University) : ℕ :=
  u.departments.map (·.teachers) |>.sum

/-- Calculates the probability of selecting an individual teacher -/
def selectionProbability (u : University) (numSelected : ℕ) : ℚ :=
  numSelected / (totalTeachers u)

/-- Theorem stating the probability of selecting an individual teacher -/
theorem teacher_selection_probability
  (u : University)
  (hDepartments : u.departments = [
    ⟨"A", 10⟩,
    ⟨"B", 20⟩,
    ⟨"C", 30⟩
  ])
  (hNumSelected : numSelected = 6) :
  selectionProbability u numSelected = 1 / 10 := by
  sorry


end teacher_selection_probability_l3591_359113


namespace sum_of_digits_63_l3591_359198

theorem sum_of_digits_63 : 
  let n : ℕ := 63
  let tens : ℕ := n / 10
  let ones : ℕ := n % 10
  tens - ones = 3 →
  tens + ones = 9 := by
sorry

end sum_of_digits_63_l3591_359198


namespace total_bananas_used_l3591_359155

/-- The number of bananas needed to make one loaf of banana bread -/
def bananas_per_loaf : ℕ := 4

/-- The number of loaves made on Monday -/
def monday_loaves : ℕ := 3

/-- The number of loaves made on Tuesday -/
def tuesday_loaves : ℕ := 2 * monday_loaves

/-- The total number of loaves made on both days -/
def total_loaves : ℕ := monday_loaves + tuesday_loaves

/-- Theorem: The total number of bananas used is 36 -/
theorem total_bananas_used : bananas_per_loaf * total_loaves = 36 := by
  sorry

end total_bananas_used_l3591_359155


namespace polynomial_property_l3591_359184

/-- Polynomial P(x) = 3x^3 + ax^2 + bx + c satisfying given conditions -/
def P (a b c : ℝ) (x : ℝ) : ℝ := 3 * x^3 + a * x^2 + b * x + c

theorem polynomial_property (a b c : ℝ) :
  (∀ x₁ x₂ x₃ : ℝ, x₁ + x₂ + x₃ = 0 → P a b c x₁ = 0 → P a b c x₂ = 0 → P a b c x₃ = 0 →
    ((x₁ + x₂ + x₃) / 3 = x₁ * x₂ * x₃)) →  -- mean of zeros equals product of zeros
  (∀ x₁ x₂ x₃ : ℝ, x₁ + x₂ + x₃ = 0 → P a b c x₁ = 0 → P a b c x₂ = 0 → P a b c x₃ = 0 →
    (x₁ * x₂ * x₃ = 3 + a + b + c)) →  -- product of zeros equals sum of coefficients
  P a b c 0 = 15 →  -- y-intercept is 15
  b = -38 := by
  sorry

end polynomial_property_l3591_359184


namespace solve_for_y_l3591_359117

theorem solve_for_y (x y : ℤ) (h1 : x = 4) (h2 : x + y = 0) : y = -4 := by
  sorry

end solve_for_y_l3591_359117


namespace function_decomposition_l3591_359136

/-- Given a function φ: ℝ³ → ℝ and two functions f, g: ℝ² → ℝ satisfying certain conditions,
    prove the existence of a function h: ℝ → ℝ with a specific property. -/
theorem function_decomposition
  (φ : ℝ → ℝ → ℝ → ℝ)
  (f g : ℝ → ℝ → ℝ)
  (h1 : ∀ x y z, φ x y z = f (x + y) z)
  (h2 : ∀ x y z, φ x y z = g x (y + z)) :
  ∃ h : ℝ → ℝ, ∀ x y z, φ x y z = h (x + y + z) := by
  sorry

end function_decomposition_l3591_359136


namespace collinear_vectors_sum_l3591_359182

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ m : ℝ, b = (m * a.1, m * a.2.1, m * a.2.2)

/-- The problem statement -/
theorem collinear_vectors_sum (x y : ℝ) :
  let a : ℝ × ℝ × ℝ := (x, 2, 2)
  let b : ℝ × ℝ × ℝ := (2, y, 4)
  collinear a b → x + y = 5 := by
sorry

end collinear_vectors_sum_l3591_359182


namespace no_valid_n_l3591_359163

/-- Represents the number of matches won by women -/
def women_wins (n : ℕ) : ℚ := 3 * (n * (4 * n - 1) / 8)

/-- Represents the number of matches won by men -/
def men_wins (n : ℕ) : ℚ := 5 * (n * (4 * n - 1) / 8)

/-- Represents the total number of matches played -/
def total_matches (n : ℕ) : ℕ := n * (4 * n - 1) / 2

theorem no_valid_n : ∀ n : ℕ, n > 0 →
  (women_wins n + men_wins n = total_matches n) →
  (3 * men_wins n = 5 * women_wins n) →
  False :=
sorry

end no_valid_n_l3591_359163


namespace initial_persons_count_l3591_359175

/-- The number of persons initially present -/
def N : ℕ := sorry

/-- The initial average weight -/
def initial_average : ℝ := sorry

/-- The weight increase when the new person replaces one person -/
def weight_increase : ℝ := 4

/-- The weight of the person being replaced -/
def replaced_weight : ℝ := 65

/-- The weight of the new person -/
def new_weight : ℝ := 97

theorem initial_persons_count : N = 8 := by sorry

end initial_persons_count_l3591_359175


namespace remainder_789987_div_8_l3591_359146

theorem remainder_789987_div_8 : 789987 % 8 = 3 := by
  sorry

end remainder_789987_div_8_l3591_359146


namespace hotel_reunions_l3591_359149

theorem hotel_reunions (total_guests : ℕ) (oates_attendees : ℕ) (hall_attendees : ℕ)
  (h1 : total_guests = 100)
  (h2 : oates_attendees = 50)
  (h3 : hall_attendees = 62)
  (h4 : ∀ g, g ≤ total_guests → (g ≤ oates_attendees ∨ g ≤ hall_attendees)) :
  oates_attendees + hall_attendees - total_guests = 12 := by
  sorry

end hotel_reunions_l3591_359149


namespace permutation_inequality_l3591_359115

theorem permutation_inequality (n : ℕ) : 
  (Nat.factorial (n + 1)).choose n ≠ Nat.factorial n := by
  sorry

end permutation_inequality_l3591_359115


namespace midpoint_trajectory_l3591_359169

/-- The trajectory of the midpoint of a line segment with one endpoint fixed and the other moving on a circle -/
theorem midpoint_trajectory (A B M : ℝ × ℝ) : 
  (B = (4, 0)) →  -- B is fixed at (4, 0)
  (∀ t : ℝ, A.1^2 + A.2^2 = 4) →  -- A moves on the circle x^2 + y^2 = 4
  (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) →  -- M is the midpoint of AB
  (M.1 - 2)^2 + M.2^2 = 1 :=  -- The trajectory of M is (x-2)^2 + y^2 = 1
by sorry

end midpoint_trajectory_l3591_359169


namespace residue_625_mod_17_l3591_359108

theorem residue_625_mod_17 : 625 % 17 = 13 := by
  sorry

end residue_625_mod_17_l3591_359108


namespace count_distinct_digit_numbers_l3591_359112

/-- The number of four-digit numbers with distinct digits, including numbers beginning with zero -/
def distinctDigitNumbers : ℕ :=
  10 * 9 * 8 * 7

/-- Theorem stating that the number of four-digit numbers with distinct digits,
    including numbers beginning with zero, is equal to 5040 -/
theorem count_distinct_digit_numbers :
  distinctDigitNumbers = 5040 := by
  sorry

end count_distinct_digit_numbers_l3591_359112


namespace angle_calculation_l3591_359110

theorem angle_calculation (α : ℝ) (h : α = 30) : 
  2 * (90 - α) - (90 - (180 - α)) = 180 := by
  sorry

end angle_calculation_l3591_359110


namespace prime_triplet_equation_l3591_359164

theorem prime_triplet_equation :
  ∀ p q r : ℕ,
  Prime p ∧ Prime q ∧ Prime r →
  p * (p - 7) + q * (q - 7) = r * (r - 7) →
  ((p = 2 ∧ q = 5 ∧ r = 7) ∨
   (p = 2 ∧ q = 5 ∧ r = 7) ∨
   (p = 7 ∧ q = 5 ∧ r = 5) ∨
   (p = 5 ∧ q = 7 ∧ r = 5) ∨
   (p = 5 ∧ q = 7 ∧ r = 2) ∨
   (p = 7 ∧ q = 5 ∧ r = 2) ∨
   (p = 7 ∧ q = 3 ∧ r = 3) ∨
   (p = 3 ∧ q = 7 ∧ r = 3) ∨
   (Prime p ∧ q = 7 ∧ r = p) ∨
   (p = 7 ∧ Prime q ∧ r = q)) :=
by sorry

end prime_triplet_equation_l3591_359164


namespace divisibility_of_polynomial_l3591_359135

theorem divisibility_of_polynomial (n : ℤ) : 
  120 ∣ (n^5 - 5*n^3 + 4*n) := by
  sorry

end divisibility_of_polynomial_l3591_359135


namespace income_increase_percentage_l3591_359161

theorem income_increase_percentage 
  (initial_income : ℝ)
  (initial_expenditure_ratio : ℝ)
  (expenditure_increase_ratio : ℝ)
  (savings_increase_ratio : ℝ)
  (income_increase_ratio : ℝ)
  (h1 : initial_expenditure_ratio = 0.75)
  (h2 : expenditure_increase_ratio = 0.1)
  (h3 : savings_increase_ratio = 0.5)
  (h4 : initial_income > 0) :
  let initial_expenditure := initial_income * initial_expenditure_ratio
  let initial_savings := initial_income - initial_expenditure
  let new_income := initial_income * (1 + income_increase_ratio)
  let new_expenditure := initial_expenditure * (1 + expenditure_increase_ratio)
  let new_savings := new_income - new_expenditure
  (new_savings = initial_savings * (1 + savings_increase_ratio)) →
  (income_increase_ratio = 0.2) :=
by sorry

end income_increase_percentage_l3591_359161


namespace rectangle_area_l3591_359145

theorem rectangle_area (length width : ℝ) (h1 : length = 20) (h2 : length = 4 * width) : length * width = 100 := by
  sorry

end rectangle_area_l3591_359145


namespace circle_radius_through_three_points_l3591_359139

/-- The radius of the circle passing through three given points is 5 -/
theorem circle_radius_through_three_points : ∃ (center : ℝ × ℝ) (r : ℝ),
  r = 5 ∧
  (center.1 - 1)^2 + (center.2 - 3)^2 = r^2 ∧
  (center.1 - 4)^2 + (center.2 - 2)^2 = r^2 ∧
  (center.1 - 1)^2 + (center.2 - (-7))^2 = r^2 :=
by sorry

end circle_radius_through_three_points_l3591_359139


namespace circles_intersect_l3591_359132

theorem circles_intersect (r₁ r₂ d : ℝ) (hr₁ : r₁ = 4) (hr₂ : r₂ = 5) (hd : d = 8) :
  (r₂ - r₁ < d) ∧ (d < r₁ + r₂) := by sorry

end circles_intersect_l3591_359132


namespace sin_angle_through_point_l3591_359134

theorem sin_angle_through_point (α : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 2 ∧ r * Real.sin α = -1) →
  Real.sin α = -Real.sqrt 5 / 5 := by
sorry

end sin_angle_through_point_l3591_359134


namespace longest_collection_pages_l3591_359188

/-- Represents a book collection --/
structure Collection where
  inches_per_page : ℚ
  total_inches : ℚ

/-- Calculates the total number of pages in a collection --/
def total_pages (c : Collection) : ℚ :=
  c.total_inches / c.inches_per_page

theorem longest_collection_pages (miles daphne : Collection)
  (h1 : miles.inches_per_page = 1/5)
  (h2 : daphne.inches_per_page = 1/50)
  (h3 : miles.total_inches = 240)
  (h4 : daphne.total_inches = 25) :
  max (total_pages miles) (total_pages daphne) = 1250 := by
  sorry

end longest_collection_pages_l3591_359188


namespace max_omega_for_monotonic_sin_l3591_359105

/-- The maximum value of ω for which f(x) = sin(ωx) is monotonic on (-π/4, π/4) -/
theorem max_omega_for_monotonic_sin (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.sin (ω * x)) →
  ω > 0 →
  (∀ x y, -π/4 < x ∧ x < y ∧ y < π/4 → (f x < f y ∨ f x > f y)) →
  ω ≤ 2 :=
sorry

end max_omega_for_monotonic_sin_l3591_359105


namespace sphere_surface_volume_relation_l3591_359121

theorem sphere_surface_volume_relation : 
  ∀ (r : ℝ) (S V S' V' : ℝ), 
    r > 0 →
    S = 4 * Real.pi * r^2 →
    V = (4/3) * Real.pi * r^3 →
    S' = 4 * S →
    V' = (4/3) * Real.pi * (2*r)^3 →
    V' = 8 * V := by
  sorry

end sphere_surface_volume_relation_l3591_359121


namespace polynomial_no_ab_term_l3591_359197

theorem polynomial_no_ab_term (m : ℤ) : 
  (∀ a b : ℤ, 2 * (a^2 - 2*a*b - b^2) - (a^2 + m*a*b + 2*b^2) = a^2 - 4*b^2) → m = -4 := by
  sorry

end polynomial_no_ab_term_l3591_359197


namespace donut_selection_count_donut_problem_l3591_359100

theorem donut_selection_count : Nat → Nat → Nat
  | n, k => Nat.choose (n + k - 1) (k - 1)

theorem donut_problem : donut_selection_count 5 4 = 56 := by
  sorry

end donut_selection_count_donut_problem_l3591_359100


namespace triangle_problem_l3591_359122

/-- 
Given an acute triangle ABC where a, b, c are sides opposite to angles A, B, C respectively,
if √3a = 2c sin A, a = 2, and the area of triangle ABC is 3√3/2,
then the measure of angle C is π/3 and c = √7.
-/
theorem triangle_problem (a b c A B C : Real) : 
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 → -- acute triangle
  Real.sqrt 3 * a = 2 * c * Real.sin A →
  a = 2 →
  (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2 →
  C = π/3 ∧ c = Real.sqrt 7 := by
  sorry


end triangle_problem_l3591_359122


namespace circle_trajectory_l3591_359111

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 77 = 0

-- Define the property of being externally tangent
def externally_tangent (P C : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), P x y ∧ C x y ∧ ∀ (x' y' : ℝ), P x' y' → C x' y' → (x = x' ∧ y = y')

-- Define the property of being internally tangent
def internally_tangent (P C : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), P x y ∧ C x y ∧ ∀ (x' y' : ℝ), P x' y' → C x' y' → (x = x' ∧ y = y')

-- Define the trajectory equation
def trajectory (x y : ℝ) : Prop := x^2 / 25 + y^2 / 21 = 1

-- State the theorem
theorem circle_trajectory :
  ∀ (P : ℝ → ℝ → Prop),
  (externally_tangent P C₁ ∧ internally_tangent P C₂) →
  (∀ (x y : ℝ), P x y → trajectory x y) :=
sorry

end circle_trajectory_l3591_359111


namespace sock_pair_combinations_l3591_359116

theorem sock_pair_combinations (black green red : ℕ) 
  (h_black : black = 5) 
  (h_green : green = 3) 
  (h_red : red = 4) : 
  black * green + black * red + green * red = 47 := by
sorry

end sock_pair_combinations_l3591_359116


namespace tomato_price_is_fifty_cents_l3591_359143

/-- Represents the monthly sales data for Village Foods --/
structure VillageFoodsSales where
  customers : ℕ
  lettucePerCustomer : ℕ
  tomatoesPerCustomer : ℕ
  lettucePricePerHead : ℚ
  totalSales : ℚ

/-- Calculates the price of each tomato based on the sales data --/
def tomatoPrice (sales : VillageFoodsSales) : ℚ :=
  let lettuceSales := sales.customers * sales.lettucePerCustomer * sales.lettucePricePerHead
  let tomatoSales := sales.totalSales - lettuceSales
  let totalTomatoes := sales.customers * sales.tomatoesPerCustomer
  tomatoSales / totalTomatoes

/-- Theorem stating that the tomato price is $0.50 given the specific sales data --/
theorem tomato_price_is_fifty_cents 
  (sales : VillageFoodsSales)
  (h1 : sales.customers = 500)
  (h2 : sales.lettucePerCustomer = 2)
  (h3 : sales.tomatoesPerCustomer = 4)
  (h4 : sales.lettucePricePerHead = 1)
  (h5 : sales.totalSales = 2000) :
  tomatoPrice sales = 1/2 := by
  sorry

#eval tomatoPrice {
  customers := 500,
  lettucePerCustomer := 2,
  tomatoesPerCustomer := 4,
  lettucePricePerHead := 1,
  totalSales := 2000
}

end tomato_price_is_fifty_cents_l3591_359143


namespace half_angle_quadrant_l3591_359114

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : Int, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3/2) * Real.pi

def is_in_second_or_fourth_quadrant (α : Real) : Prop :=
  (∃ k : Int, k * Real.pi + Real.pi/2 < α ∧ α < k * Real.pi + Real.pi) ∨
  (∃ k : Int, k * Real.pi + (3/2) * Real.pi < α ∧ α < (k + 1) * Real.pi)

theorem half_angle_quadrant (α : Real) :
  is_in_third_quadrant α → is_in_second_or_fourth_quadrant (α/2) :=
by sorry

end half_angle_quadrant_l3591_359114


namespace select_three_from_boys_and_girls_l3591_359170

theorem select_three_from_boys_and_girls :
  let num_boys : ℕ := 4
  let num_girls : ℕ := 3
  let total_to_select : ℕ := 3
  let ways_to_select : ℕ := 
    (num_boys.choose 2 * num_girls.choose 1) + 
    (num_boys.choose 1 * num_girls.choose 2)
  ways_to_select = 30 := by
sorry

end select_three_from_boys_and_girls_l3591_359170
