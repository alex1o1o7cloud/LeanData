import Mathlib

namespace NUMINAMATH_CALUDE_min_moves_at_least_50_l3386_338646

/-- A 4x4 grid representing the puzzle state -/
def PuzzleState := Fin 4 → Fin 4 → Option (Fin 16)

/-- A move in the puzzle -/
inductive Move
| slide : Fin 4 → Fin 4 → Fin 4 → Fin 4 → Move
| jump  : Fin 4 → Fin 4 → Fin 4 → Fin 4 → Move

/-- Check if a PuzzleState is a valid magic square with sum 30 -/
def isMagicSquare (state : PuzzleState) : Prop := sorry

/-- Check if a move is valid for a given state -/
def isValidMove (state : PuzzleState) (move : Move) : Prop := sorry

/-- Apply a move to a state -/
def applyMove (state : PuzzleState) (move : Move) : PuzzleState := sorry

/-- The minimum number of moves required to solve the puzzle -/
def minMoves (initial : PuzzleState) : ℕ := sorry

/-- The theorem stating that the minimum number of moves is at least 50 -/
theorem min_moves_at_least_50 (initial : PuzzleState) : 
  minMoves initial ≥ 50 := by sorry

end NUMINAMATH_CALUDE_min_moves_at_least_50_l3386_338646


namespace NUMINAMATH_CALUDE_power_calculation_l3386_338665

theorem power_calculation : ((16^10 / 16^8)^3 * 8^3) / 2^9 = 16777216 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l3386_338665


namespace NUMINAMATH_CALUDE_fraction_simplification_l3386_338661

theorem fraction_simplification (a : ℝ) (h : a ≠ 2) :
  (a^2 / (a - 2)) - ((4*a - 4) / (a - 2)) = a - 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3386_338661


namespace NUMINAMATH_CALUDE_binomial_variance_example_l3386_338688

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p
  h2 : p ≤ 1

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: The variance of X ~ B(8, 3/4) is 3/2 -/
theorem binomial_variance_example :
  let X : BinomialRV := ⟨8, 3/4, by norm_num, by norm_num⟩
  variance X = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_example_l3386_338688


namespace NUMINAMATH_CALUDE_line_in_quadrants_implies_ac_bc_negative_l3386_338678

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a line lies in the first, second, and fourth quadrants -/
def liesInQuadrants (l : Line) : Prop :=
  ∃ (x y : ℝ), 
    (l.a * x + l.b * y + l.c = 0) ∧
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))

/-- Theorem stating the relationship between ac and bc for a line in the specified quadrants -/
theorem line_in_quadrants_implies_ac_bc_negative (l : Line) :
  liesInQuadrants l → (l.a * l.c < 0 ∧ l.b * l.c < 0) := by
  sorry

end NUMINAMATH_CALUDE_line_in_quadrants_implies_ac_bc_negative_l3386_338678


namespace NUMINAMATH_CALUDE_toby_first_part_distance_l3386_338676

/-- Represents Toby's journey with a loaded and unloaded sled -/
def toby_journey (x : ℝ) : Prop :=
  let loaded_speed : ℝ := 10
  let unloaded_speed : ℝ := 20
  let second_part : ℝ := 120
  let third_part : ℝ := 80
  let fourth_part : ℝ := 140
  let total_time : ℝ := 39
  (x / loaded_speed) + (second_part / unloaded_speed) + 
  (third_part / loaded_speed) + (fourth_part / unloaded_speed) = total_time

/-- Theorem stating that Toby pulled the loaded sled for 180 miles in the first part of the journey -/
theorem toby_first_part_distance : 
  ∃ (x : ℝ), toby_journey x ∧ x = 180 :=
sorry

end NUMINAMATH_CALUDE_toby_first_part_distance_l3386_338676


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l3386_338685

theorem quadratic_completing_square :
  ∀ x : ℝ, (x^2 - 4*x - 6 = 0) ↔ ((x - 2)^2 = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l3386_338685


namespace NUMINAMATH_CALUDE_megan_bought_42_songs_l3386_338673

/-- The number of songs Megan bought given the initial number of albums,
    the number of albums removed, and the number of songs per album. -/
def total_songs (initial_albums : ℕ) (removed_albums : ℕ) (songs_per_album : ℕ) : ℕ :=
  (initial_albums - removed_albums) * songs_per_album

/-- Theorem stating that Megan bought 42 songs in total. -/
theorem megan_bought_42_songs :
  total_songs 8 2 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_megan_bought_42_songs_l3386_338673


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l3386_338675

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- c is 7/2
  c = 7/2 ∧
  -- Area of triangle ABC is 3√3/2
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 ∧
  -- Relationship between tan A and tan B
  Real.tan A + Real.tan B = Real.sqrt 3 * (Real.tan A * Real.tan B - 1)

-- Theorem statement
theorem triangle_ABC_properties {a b c A B C : ℝ} 
  (h : triangle_ABC a b c A B C) : 
  C = Real.pi / 3 ∧ a + b = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l3386_338675


namespace NUMINAMATH_CALUDE_condition_analysis_l3386_338677

theorem condition_analysis (a : ℕ) : 
  let A : Set ℕ := {1, a}
  let B : Set ℕ := {1, 2, 3}
  (a = 3 → A ⊆ B) ∧ (∃ x ≠ 3, {1, x} ⊆ B) :=
by sorry

end NUMINAMATH_CALUDE_condition_analysis_l3386_338677


namespace NUMINAMATH_CALUDE_registered_number_scientific_notation_l3386_338612

/-- The number of people registered for the national college entrance examination in 2023 -/
def registered_number : ℝ := 12910000

/-- The scientific notation representation of the registered number -/
def scientific_notation : ℝ := 1.291 * (10 ^ 7)

/-- Theorem stating that the registered number is equal to its scientific notation representation -/
theorem registered_number_scientific_notation : registered_number = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_registered_number_scientific_notation_l3386_338612


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3386_338660

theorem polynomial_expansion (w : ℝ) : 
  (3 * w^3 + 4 * w^2 - 7) * (2 * w^3 - 3 * w^2 + 1) = 
  6 * w^6 - 6 * w^5 + 9 * w^3 + 12 * w^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3386_338660


namespace NUMINAMATH_CALUDE_scallop_dinner_cost_l3386_338617

/-- Calculates the cost of scallops for a dinner party. -/
def scallop_cost (people : ℕ) (scallops_per_person : ℕ) (scallops_per_pound : ℕ) (cost_per_pound : ℚ) : ℚ :=
  (people * scallops_per_person : ℚ) / scallops_per_pound * cost_per_pound

/-- Proves that the cost of scallops for 8 people, given 2 scallops per person, 
    is $48.00, when 8 scallops weigh one pound and cost $24.00 per pound. -/
theorem scallop_dinner_cost : 
  scallop_cost 8 2 8 24 = 48 := by
  sorry

end NUMINAMATH_CALUDE_scallop_dinner_cost_l3386_338617


namespace NUMINAMATH_CALUDE_unique_solution_is_six_l3386_338684

theorem unique_solution_is_six :
  ∃! n : ℤ, ⌊(n^2 : ℚ) / 3⌋ - ⌊(n : ℚ) / 2⌋^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_six_l3386_338684


namespace NUMINAMATH_CALUDE_total_peaches_l3386_338663

theorem total_peaches (num_baskets : ℕ) (red_per_basket : ℕ) (green_per_basket : ℕ) :
  num_baskets = 11 →
  red_per_basket = 10 →
  green_per_basket = 18 →
  num_baskets * (red_per_basket + green_per_basket) = 308 :=
by
  sorry

end NUMINAMATH_CALUDE_total_peaches_l3386_338663


namespace NUMINAMATH_CALUDE_malcolm_facebook_followers_l3386_338628

/-- Represents the number of followers on different social media platforms -/
structure Followers where
  instagram : ℕ
  facebook : ℕ
  twitter : ℕ
  tiktok : ℕ
  youtube : ℕ

/-- Calculates the total number of followers across all platforms -/
def totalFollowers (f : Followers) : ℕ :=
  f.instagram + f.facebook + f.twitter + f.tiktok + f.youtube

/-- Theorem stating that given the conditions, Malcolm has 375 followers on Facebook -/
theorem malcolm_facebook_followers :
  ∃ (f : Followers),
    f.instagram = 240 ∧
    f.twitter = (f.instagram + f.facebook) / 2 ∧
    f.tiktok = 3 * f.twitter ∧
    f.youtube = f.tiktok + 510 ∧
    totalFollowers f = 3840 →
    f.facebook = 375 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_facebook_followers_l3386_338628


namespace NUMINAMATH_CALUDE_greatest_integer_radius_eight_is_greatest_l3386_338659

theorem greatest_integer_radius (r : ℕ) : r ^ 2 < 75 → r ≤ 8 := by
  sorry

theorem eight_is_greatest : ∃ (r : ℕ), r ^ 2 < 75 ∧ r = 8 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_eight_is_greatest_l3386_338659


namespace NUMINAMATH_CALUDE_triangle_properties_l3386_338623

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  let m : ℝ × ℝ := (Real.cos B, Real.cos C)
  let n : ℝ × ℝ := (2*a + c, b)
  (m.1 * n.1 + m.2 * n.2 = 0) →  -- m ⟂ n
  (b = Real.sqrt 13) →
  (a + c = 4) →
  (B = 2 * Real.pi / 3) ∧
  (Real.sqrt 3 / 2 < Real.sin (2*A) + Real.sin (2*C)) ∧
  (Real.sin (2*A) + Real.sin (2*C) ≤ Real.sqrt 3) ∧
  (1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3386_338623


namespace NUMINAMATH_CALUDE_only_divisor_square_sum_l3386_338681

theorem only_divisor_square_sum (n : ℕ+) :
  ∀ d : ℕ+, d ∣ (3 * n^2) → ∃ k : ℕ, n^2 + d = k^2 → d = 3 * n^2 :=
sorry

end NUMINAMATH_CALUDE_only_divisor_square_sum_l3386_338681


namespace NUMINAMATH_CALUDE_john_ray_difference_l3386_338666

/-- The number of chickens each person took -/
structure ChickenCount where
  john : ℕ
  mary : ℕ
  ray : ℕ

/-- The conditions of the chicken problem -/
def chicken_problem (c : ChickenCount) : Prop :=
  c.john = c.mary + 5 ∧
  c.mary = c.ray + 6 ∧
  c.ray = 10

/-- The theorem stating the difference between John's and Ray's chickens -/
theorem john_ray_difference (c : ChickenCount) 
  (h : chicken_problem c) : c.john - c.ray = 11 := by
  sorry

end NUMINAMATH_CALUDE_john_ray_difference_l3386_338666


namespace NUMINAMATH_CALUDE_difference_of_squares_simplification_l3386_338608

theorem difference_of_squares_simplification : (164^2 - 148^2) / 16 = 312 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_simplification_l3386_338608


namespace NUMINAMATH_CALUDE_largest_divisor_of_four_consecutive_integers_l3386_338691

theorem largest_divisor_of_four_consecutive_integers (n : ℕ) :
  (∀ m : ℕ, (m * (m + 1) * (m + 2) * (m + 3)) % 24 = 0) ∧
  (∃ k : ℕ, (k * (k + 1) * (k + 2) * (k + 3)) % 25 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_four_consecutive_integers_l3386_338691


namespace NUMINAMATH_CALUDE_max_value_of_S_l3386_338672

def S (a b c d e f g h k : Int) : Int :=
  a*e*k - a*f*h + b*f*g - b*d*k + c*d*h - c*e*g

theorem max_value_of_S :
  ∃ (a b c d e f g h k : Int),
    (a = 1 ∨ a = -1) ∧
    (b = 1 ∨ b = -1) ∧
    (c = 1 ∨ c = -1) ∧
    (d = 1 ∨ d = -1) ∧
    (e = 1 ∨ e = -1) ∧
    (f = 1 ∨ f = -1) ∧
    (g = 1 ∨ g = -1) ∧
    (h = 1 ∨ h = -1) ∧
    (k = 1 ∨ k = -1) ∧
    S a b c d e f g h k = 4 ∧
    ∀ (a' b' c' d' e' f' g' h' k' : Int),
      (a' = 1 ∨ a' = -1) →
      (b' = 1 ∨ b' = -1) →
      (c' = 1 ∨ c' = -1) →
      (d' = 1 ∨ d' = -1) →
      (e' = 1 ∨ e' = -1) →
      (f' = 1 ∨ f' = -1) →
      (g' = 1 ∨ g' = -1) →
      (h' = 1 ∨ h' = -1) →
      (k' = 1 ∨ k' = -1) →
      S a' b' c' d' e' f' g' h' k' ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_S_l3386_338672


namespace NUMINAMATH_CALUDE_parallelogram_area_34_18_l3386_338679

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 34 cm and height 18 cm is 612 square centimeters -/
theorem parallelogram_area_34_18 : parallelogram_area 34 18 = 612 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_34_18_l3386_338679


namespace NUMINAMATH_CALUDE_dot_product_FA_AB_is_zero_l3386_338604

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the focus F
def focus : ℝ × ℝ := (4, 0)

-- Define point A on y-axis with |OA| = |OF|
def point_A : ℝ × ℝ := (0, 4)  -- We choose the positive y-coordinate

-- Define point B as intersection of directrix and x-axis
def point_B : ℝ × ℝ := (-4, 0)

-- Define vector FA
def vector_FA : ℝ × ℝ := (point_A.1 - focus.1, point_A.2 - focus.2)

-- Define vector AB
def vector_AB : ℝ × ℝ := (point_B.1 - point_A.1, point_B.2 - point_A.2)

-- Theorem statement
theorem dot_product_FA_AB_is_zero :
  vector_FA.1 * vector_AB.1 + vector_FA.2 * vector_AB.2 = 0 :=
sorry

end NUMINAMATH_CALUDE_dot_product_FA_AB_is_zero_l3386_338604


namespace NUMINAMATH_CALUDE_nut_problem_l3386_338630

theorem nut_problem (sue_nuts : ℕ) (harry_nuts : ℕ) (bill_nuts : ℕ) 
  (h1 : sue_nuts = 48)
  (h2 : harry_nuts = 2 * sue_nuts)
  (h3 : bill_nuts = 6 * harry_nuts) :
  bill_nuts + harry_nuts = 672 := by
sorry

end NUMINAMATH_CALUDE_nut_problem_l3386_338630


namespace NUMINAMATH_CALUDE_garden_bed_area_l3386_338696

/-- Represents the dimensions of a rectangular garden bed -/
structure GardenBed where
  length : ℝ
  width : ℝ

/-- Calculates the area of a garden bed -/
def area (bed : GardenBed) : ℝ := bed.length * bed.width

/-- Theorem: Given the conditions, prove that the area of each unknown garden bed is 9 sq ft -/
theorem garden_bed_area 
  (known_bed : GardenBed)
  (unknown_bed : GardenBed)
  (h1 : known_bed.length = 4)
  (h2 : known_bed.width = 3)
  (h3 : area known_bed + area known_bed + area unknown_bed + area unknown_bed = 42) :
  area unknown_bed = 9 := by
  sorry

end NUMINAMATH_CALUDE_garden_bed_area_l3386_338696


namespace NUMINAMATH_CALUDE_star_power_equality_l3386_338603

/-- The k-th smallest positive integer not in X -/
def f (X : Finset ℕ) (k : ℕ) : ℕ := sorry

/-- The operation * for finite sets of positive integers -/
def starOp (X Y : Finset ℕ) : Finset ℕ :=
  X ∪ (Y.image (f X))

/-- Repeated application of starOp -/
def starPower (X : Finset ℕ) : ℕ → Finset ℕ
  | 0 => X
  | n + 1 => starOp X (starPower X n)

theorem star_power_equality {A B : Finset ℕ} (ha : A.card > 0) (hb : B.card > 0)
    (h : starOp A B = starOp B A) :
    starPower A B.card = starPower B A.card := by
  sorry

end NUMINAMATH_CALUDE_star_power_equality_l3386_338603


namespace NUMINAMATH_CALUDE_cuboid_volume_l3386_338638

/-- The volume of a cuboid with edges 2, 5, and 8 is 80 -/
theorem cuboid_volume : 
  let edge1 : ℝ := 2
  let edge2 : ℝ := 5
  let edge3 : ℝ := 8
  edge1 * edge2 * edge3 = 80 := by sorry

end NUMINAMATH_CALUDE_cuboid_volume_l3386_338638


namespace NUMINAMATH_CALUDE_bacon_percentage_is_twenty_l3386_338698

/-- Calculates the percentage of calories from bacon in a sandwich -/
def bacon_calorie_percentage (total_calories : ℕ) (bacon_strips : ℕ) (calories_per_strip : ℕ) : ℚ :=
  (bacon_strips * calories_per_strip : ℚ) / total_calories * 100

/-- Theorem stating that the percentage of calories from bacon in the given sandwich is 20% -/
theorem bacon_percentage_is_twenty :
  bacon_calorie_percentage 1250 2 125 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bacon_percentage_is_twenty_l3386_338698


namespace NUMINAMATH_CALUDE_candidates_calculation_l3386_338629

theorem candidates_calculation (total_candidates : ℕ) : 
  (total_candidates * 6 / 100 : ℚ) + 83 = (total_candidates * 7 / 100 : ℚ) → 
  total_candidates = 8300 := by
  sorry

end NUMINAMATH_CALUDE_candidates_calculation_l3386_338629


namespace NUMINAMATH_CALUDE_i_power_2013_l3386_338625

theorem i_power_2013 (i : ℂ) (h : i^2 = -1) : i^2013 = i := by
  sorry

end NUMINAMATH_CALUDE_i_power_2013_l3386_338625


namespace NUMINAMATH_CALUDE_park_visitors_l3386_338654

/-- Represents the charging conditions for the park visit -/
structure ParkVisitConditions where
  base_fee : ℕ  -- Base fee per person
  base_limit : ℕ  -- Number of people for base fee
  discount_per_person : ℕ  -- Discount per additional person
  min_fee : ℕ  -- Minimum fee per person
  total_paid : ℕ  -- Total amount paid

/-- Calculates the fee per person based on the number of visitors -/
def fee_per_person (conditions : ParkVisitConditions) (num_visitors : ℕ) : ℕ :=
  max conditions.min_fee (conditions.base_fee - conditions.discount_per_person * (num_visitors - conditions.base_limit))

/-- Theorem: Given the charging conditions, 30 people visited the park -/
theorem park_visitors (conditions : ParkVisitConditions) 
  (h1 : conditions.base_fee = 100)
  (h2 : conditions.base_limit = 25)
  (h3 : conditions.discount_per_person = 2)
  (h4 : conditions.min_fee = 70)
  (h5 : conditions.total_paid = 2700) :
  ∃ (num_visitors : ℕ), 
    num_visitors = 30 ∧ 
    num_visitors * (fee_per_person conditions num_visitors) = conditions.total_paid :=
sorry

end NUMINAMATH_CALUDE_park_visitors_l3386_338654


namespace NUMINAMATH_CALUDE_delivery_distances_l3386_338626

/-- Represents the direction of travel --/
inductive Direction
  | North
  | South

/-- Represents a location relative to the supermarket --/
structure Location where
  distance : ℝ
  direction : Direction

/-- Calculates the distance between two locations --/
def distanceBetween (a b : Location) : ℝ :=
  match a.direction, b.direction with
  | Direction.North, Direction.North => abs (a.distance - b.distance)
  | Direction.South, Direction.South => abs (a.distance - b.distance)
  | _, _ => a.distance + b.distance

/-- Calculates the round trip distance to a location --/
def roundTripDistance (loc : Location) : ℝ :=
  2 * loc.distance

theorem delivery_distances (unitA unitB unitC : Location) 
  (hA : unitA = { distance := 30, direction := Direction.South })
  (hB : unitB = { distance := 50, direction := Direction.South })
  (hC : unitC = { distance := 15, direction := Direction.North }) :
  distanceBetween unitA unitC = 45 ∧ 
  roundTripDistance unitB + 3 * roundTripDistance unitC = 190 := by
  sorry


end NUMINAMATH_CALUDE_delivery_distances_l3386_338626


namespace NUMINAMATH_CALUDE_paint_remaining_after_three_days_paint_problem_solution_l3386_338634

/-- Represents the amount of paint remaining after a certain number of days -/
def paint_remaining (initial_amount : ℚ) (days : ℕ) : ℚ :=
  initial_amount * (1 / 2) ^ days

/-- Theorem stating that after 3 days of using half the remaining paint each day, 
    1/4 of the original amount remains -/
theorem paint_remaining_after_three_days (initial_amount : ℚ) :
  paint_remaining initial_amount 3 = initial_amount / 4 := by
  sorry

/-- Theorem proving that starting with 2 gallons and using half the remaining paint 
    for three consecutive days leaves 1/4 of the original amount -/
theorem paint_problem_solution :
  paint_remaining 2 3 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_paint_remaining_after_three_days_paint_problem_solution_l3386_338634


namespace NUMINAMATH_CALUDE_inequality_proof_l3386_338621

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3386_338621


namespace NUMINAMATH_CALUDE_not_right_triangle_l3386_338614

-- Define a triangle with angles A, B, and C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the ratio condition
def ratio_condition (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.A = 5 * k ∧ t.B = 12 * k ∧ t.C = 13 * k

-- Theorem statement
theorem not_right_triangle (t : Triangle) (h : ratio_condition t) : 
  t.A ≠ 90 ∧ t.B ≠ 90 ∧ t.C ≠ 90 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_l3386_338614


namespace NUMINAMATH_CALUDE_sphere_volume_inscribed_cylinder_l3386_338618

-- Define the radius of the base of the cylinder
def r : ℝ := 15

-- Define the radius of the sphere
def sphere_radius : ℝ := r + 2

-- Define the height of the cylinder
def cylinder_height : ℝ := r + 1

-- State the theorem
theorem sphere_volume_inscribed_cylinder :
  let volume := (4 / 3) * Real.pi * sphere_radius ^ 3
  (2 * sphere_radius) ^ 2 = (2 * r) ^ 2 + cylinder_height ^ 2 →
  volume = 6550 * (2 / 3) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_sphere_volume_inscribed_cylinder_l3386_338618


namespace NUMINAMATH_CALUDE_x_power_twelve_l3386_338619

theorem x_power_twelve (x : ℝ) (h : x + 1/x = 3) : x^12 + 1/x^12 = 103682 := by
  sorry

end NUMINAMATH_CALUDE_x_power_twelve_l3386_338619


namespace NUMINAMATH_CALUDE_leap_year_hours_l3386_338616

/-- The number of days in a leap year -/
def days_in_leap_year : ℕ := 366

/-- The number of hours in a day -/
def hours_in_day : ℕ := 24

/-- The number of hours in a leap year -/
def hours_in_leap_year : ℕ := days_in_leap_year * hours_in_day

theorem leap_year_hours :
  hours_in_leap_year = 8784 :=
by sorry

end NUMINAMATH_CALUDE_leap_year_hours_l3386_338616


namespace NUMINAMATH_CALUDE_student_language_partition_l3386_338671

/-- Represents a student and the languages they speak -/
structure Student where
  speaksEnglish : Bool
  speaksFrench : Bool
  speaksSpanish : Bool

/-- Represents a group of students -/
def StudentGroup := List Student

/-- Checks if a group satisfies the language requirements -/
def isValidGroup (group : StudentGroup) : Bool :=
  (group.filter (·.speaksEnglish)).length = 10 ∧
  (group.filter (·.speaksFrench)).length = 10 ∧
  (group.filter (·.speaksSpanish)).length = 10

/-- Main theorem -/
theorem student_language_partition 
  (students : List Student)
  (h_english : (students.filter (·.speaksEnglish)).length = 50)
  (h_french : (students.filter (·.speaksFrench)).length = 50)
  (h_spanish : (students.filter (·.speaksSpanish)).length = 50) :
  ∃ (partition : List StudentGroup), 
    partition.length = 5 ∧ 
    (∀ group ∈ partition, isValidGroup group) ∧
    (partition.join = students) :=
  sorry

end NUMINAMATH_CALUDE_student_language_partition_l3386_338671


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3386_338658

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (k + 2) * x^2 - 2 * (k - 1) * x + k + 1 = 0) ↔ (k = -1/5 ∨ k = -2) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3386_338658


namespace NUMINAMATH_CALUDE_quadratic_function_condition_l3386_338622

-- Define the function
def f (m : ℝ) (x : ℝ) : ℝ := (m + 2) * x^2 + m

-- Theorem statement
theorem quadratic_function_condition (m : ℝ) : 
  (∀ x, ∃ a b c, f m x = a * x^2 + b * x + c ∧ a ≠ 0) ↔ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_condition_l3386_338622


namespace NUMINAMATH_CALUDE_arctan_sum_in_triangle_l3386_338644

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (angleC : ℝ)
  (pos_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (pos_angleC : angleC > 0)
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b)

-- State the theorem
theorem arctan_sum_in_triangle (t : Triangle) : 
  Real.arctan (t.a / (t.b + t.c - t.a)) + Real.arctan (t.b / (t.a + t.c - t.b)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_in_triangle_l3386_338644


namespace NUMINAMATH_CALUDE_sixth_term_is_one_sixteenth_l3386_338607

/-- A geometric sequence with specific conditions -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  (∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q) ∧
  a 1 + a 3 = 5/2 ∧
  a 2 + a 4 = 5/4

/-- The sixth term of the geometric sequence is 1/16 -/
theorem sixth_term_is_one_sixteenth (a : ℕ → ℚ) (h : GeometricSequence a) :
  a 6 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_one_sixteenth_l3386_338607


namespace NUMINAMATH_CALUDE_student_number_problem_l3386_338652

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 112 → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l3386_338652


namespace NUMINAMATH_CALUDE_exam_mean_score_l3386_338601

/-- Given an exam where a score of 86 is 7 standard deviations below the mean,
    and a score of 90 is 3 standard deviations above the mean,
    prove that the mean score is 88.8 -/
theorem exam_mean_score (μ σ : ℝ) 
    (h1 : 86 = μ - 7 * σ) 
    (h2 : 90 = μ + 3 * σ) : 
  μ = 88.8 := by
sorry

end NUMINAMATH_CALUDE_exam_mean_score_l3386_338601


namespace NUMINAMATH_CALUDE_power_eight_mod_eleven_l3386_338600

theorem power_eight_mod_eleven : 8^2030 % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_eight_mod_eleven_l3386_338600


namespace NUMINAMATH_CALUDE_triangle_side_value_l3386_338697

/-- Triangle inequality theorem for a triangle with sides 2, 3, and m -/
def triangle_inequality (m : ℝ) : Prop :=
  2 + 3 > m ∧ 2 + m > 3 ∧ 3 + m > 2

/-- The only valid integer value for m is 3 -/
theorem triangle_side_value :
  ∀ m : ℕ, triangle_inequality (m : ℝ) ↔ m = 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_value_l3386_338697


namespace NUMINAMATH_CALUDE_smallest_divisible_number_is_correct_l3386_338611

/-- The smallest six-digit number exactly divisible by 25, 35, 45, and 15 -/
def smallest_divisible_number : ℕ := 100800

/-- Predicate to check if a number is six digits -/
def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

theorem smallest_divisible_number_is_correct :
  is_six_digit smallest_divisible_number ∧
  smallest_divisible_number % 25 = 0 ∧
  smallest_divisible_number % 35 = 0 ∧
  smallest_divisible_number % 45 = 0 ∧
  smallest_divisible_number % 15 = 0 ∧
  ∀ n : ℕ, is_six_digit n →
    n % 25 = 0 → n % 35 = 0 → n % 45 = 0 → n % 15 = 0 →
    n ≥ smallest_divisible_number :=
by sorry

#eval smallest_divisible_number

end NUMINAMATH_CALUDE_smallest_divisible_number_is_correct_l3386_338611


namespace NUMINAMATH_CALUDE_min_chips_for_adjacency_l3386_338695

/-- Represents a color of a chip -/
def Color : Type := Fin 6

/-- Represents a row of chips -/
def ChipRow := List Color

/-- Checks if two colors are adjacent in a row -/
def areAdjacent (c1 c2 : Color) (row : ChipRow) : Prop :=
  ∃ i, (row.get? i = some c1 ∧ row.get? (i+1) = some c2) ∨
       (row.get? i = some c2 ∧ row.get? (i+1) = some c1)

/-- Checks if all pairs of colors are adjacent in a row -/
def allPairsAdjacent (row : ChipRow) : Prop :=
  ∀ c1 c2 : Color, c1 ≠ c2 → areAdjacent c1 c2 row

/-- The main theorem stating the minimum number of chips required -/
theorem min_chips_for_adjacency :
  ∃ (row : ChipRow), allPairsAdjacent row ∧ row.length = 18 ∧
  (∀ (row' : ChipRow), allPairsAdjacent row' → row'.length ≥ 18) :=
sorry

end NUMINAMATH_CALUDE_min_chips_for_adjacency_l3386_338695


namespace NUMINAMATH_CALUDE_single_intersection_l3386_338640

/-- The quadratic function representing the first graph -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + 2 * x + 3

/-- The linear function representing the second graph -/
def g (x : ℝ) : ℝ := 2 * x + 5

/-- The theorem stating the condition for a single intersection point -/
theorem single_intersection (k : ℝ) : 
  (∃! x, f k x = g x) ↔ k = -1/2 := by sorry

end NUMINAMATH_CALUDE_single_intersection_l3386_338640


namespace NUMINAMATH_CALUDE_min_value_equality_l3386_338648

theorem min_value_equality (x y a : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) :
  (∀ x y, x + 2*y = 1 → (3/x + a/y ≥ 6*Real.sqrt 3)) ∧
  (∃ x y, x + 2*y = 1 ∧ 3/x + a/y = 6*Real.sqrt 3) →
  (∀ x y, 1/x + 2/y = 1 → (3*x + a*y ≥ 6*Real.sqrt 3)) ∧
  (∃ x y, 1/x + 2/y = 1 ∧ 3*x + a*y = 6*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_equality_l3386_338648


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3386_338662

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 1 < 0) → a ∈ Set.Iio (-2) ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3386_338662


namespace NUMINAMATH_CALUDE_kitchen_tiling_l3386_338613

def kitchen_length : ℕ := 20
def kitchen_width : ℕ := 15
def border_width : ℕ := 2
def border_tile_length : ℕ := 2
def border_tile_width : ℕ := 1
def inner_tile_size : ℕ := 3

def border_tiles_count : ℕ := 
  2 * (kitchen_length - 2 * border_width) / border_tile_length +
  2 * (kitchen_width - 2 * border_width) / border_tile_length

def inner_area : ℕ := (kitchen_length - 2 * border_width) * (kitchen_width - 2 * border_width)

def inner_tiles_count : ℕ := (inner_area + inner_tile_size^2 - 1) / inner_tile_size^2

def total_tiles : ℕ := border_tiles_count + inner_tiles_count

theorem kitchen_tiling :
  total_tiles = 48 :=
sorry

end NUMINAMATH_CALUDE_kitchen_tiling_l3386_338613


namespace NUMINAMATH_CALUDE_simple_interest_doubling_l3386_338670

/-- The factor by which a sum of money increases under simple interest -/
def simple_interest_factor (rate : ℝ) (time : ℝ) : ℝ :=
  1 + rate * time

/-- Theorem: Given a simple interest rate of 25% per annum over 4 years, 
    the factor by which an initial sum of money increases is 2 -/
theorem simple_interest_doubling : 
  simple_interest_factor 0.25 4 = 2 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_doubling_l3386_338670


namespace NUMINAMATH_CALUDE_y_plus_two_over_y_l3386_338632

theorem y_plus_two_over_y (y : ℝ) (h : 5 = y^2 + 4/y^2) : 
  y + 2/y = 3 ∨ y + 2/y = -3 := by
sorry

end NUMINAMATH_CALUDE_y_plus_two_over_y_l3386_338632


namespace NUMINAMATH_CALUDE_vacation_cost_difference_l3386_338627

theorem vacation_cost_difference (total_cost : ℕ) (initial_people : ℕ) (new_people : ℕ) 
  (h1 : total_cost = 360) 
  (h2 : initial_people = 3) 
  (h3 : new_people = 4) : 
  (total_cost / initial_people) - (total_cost / new_people) = 30 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_difference_l3386_338627


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l3386_338605

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l3386_338605


namespace NUMINAMATH_CALUDE_distance_from_origin_of_complex_fraction_l3386_338645

theorem distance_from_origin_of_complex_fraction : 
  let z : ℂ := 2 / (1 + Complex.I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_distance_from_origin_of_complex_fraction_l3386_338645


namespace NUMINAMATH_CALUDE_team_selection_count_l3386_338657

/-- The number of ways to select a team of 8 members with an equal number of boys and girls
    from a group of 8 boys and 10 girls -/
def select_team (boys girls team_size : ℕ) : ℕ :=
  Nat.choose boys (team_size / 2) * Nat.choose girls (team_size / 2)

/-- Theorem stating the number of ways to select the team -/
theorem team_selection_count :
  select_team 8 10 8 = 14700 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l3386_338657


namespace NUMINAMATH_CALUDE_power_equality_l3386_338650

theorem power_equality (p : ℕ) : 81^6 = 3^p → p = 24 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3386_338650


namespace NUMINAMATH_CALUDE_time_to_see_again_is_75_l3386_338674

/-- The time before Jenny and Kenny can see each other again -/
def time_to_see_again : ℝ → Prop := λ t =>
  let jenny_speed := 2 -- feet per second
  let kenny_speed := 4 -- feet per second
  let path_distance := 300 -- feet
  let building_diameter := 200 -- feet
  let initial_distance := 300 -- feet
  let jenny_position := λ t : ℝ => (-100 + jenny_speed * t, path_distance / 2)
  let kenny_position := λ t : ℝ => (-100 + kenny_speed * t, -path_distance / 2)
  let building_center := (0, 0)
  let building_radius := building_diameter / 2

  -- Line equation connecting Jenny and Kenny
  let line_equation := λ x y : ℝ =>
    y = -(path_distance / t) * x + path_distance - (initial_distance * path_distance / (2 * t))

  -- Circle equation representing the building
  let circle_equation := λ x y : ℝ =>
    x^2 + y^2 = building_radius^2

  -- Tangent condition
  let tangent_condition := λ x y : ℝ =>
    x * t = path_distance / 2 * y

  -- Point of tangency satisfies both line and circle equations
  ∃ x y : ℝ, line_equation x y ∧ circle_equation x y ∧ tangent_condition x y

theorem time_to_see_again_is_75 : time_to_see_again 75 :=
  sorry

end NUMINAMATH_CALUDE_time_to_see_again_is_75_l3386_338674


namespace NUMINAMATH_CALUDE_ideal_solution_range_l3386_338606

theorem ideal_solution_range (m n q : ℝ) : 
  m + 2*n = 6 →
  2*m + n = 3*q →
  m + n > 1 →
  q > -1 := by
sorry

end NUMINAMATH_CALUDE_ideal_solution_range_l3386_338606


namespace NUMINAMATH_CALUDE_circles_intersect_l3386_338642

/-- Circle C₁ with equation x² + y² + 2x + 8y - 8 = 0 -/
def C₁ (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 8*y - 8 = 0

/-- Circle C₂ with equation x² + y² - 4x - 5 = 0 -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 5 = 0

/-- The circles C₁ and C₂ intersect -/
theorem circles_intersect : ∃ (x y : ℝ), C₁ x y ∧ C₂ x y := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l3386_338642


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3386_338637

/-- Given an increasing function f: ℝ → ℝ with f(0) = -1 and f(3) = 1,
    the set {x ∈ ℝ | |f(x)| < 1} is equal to the open interval (0, 3). -/
theorem solution_set_equivalence (f : ℝ → ℝ) 
    (h_increasing : ∀ x y, x < y → f x < f y)
    (h_f_0 : f 0 = -1)
    (h_f_3 : f 3 = 1) :
    {x : ℝ | |f x| < 1} = Set.Ioo 0 3 := by
  sorry


end NUMINAMATH_CALUDE_solution_set_equivalence_l3386_338637


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l3386_338680

theorem complex_fraction_evaluation : 
  2 + (3 / (4 + (5 / 6))) = 76 / 29 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l3386_338680


namespace NUMINAMATH_CALUDE_lo_length_l3386_338667

/-- Represents a parallelogram LMNO with given properties -/
structure Parallelogram where
  -- Length of side MN
  mn_length : ℝ
  -- Altitude from O to MN
  altitude_o_to_mn : ℝ
  -- Altitude from N to LO
  altitude_n_to_lo : ℝ
  -- Condition that LMNO is a parallelogram
  is_parallelogram : True

/-- Theorem stating the length of LO in the parallelogram LMNO -/
theorem lo_length (p : Parallelogram)
  (h1 : p.mn_length = 15)
  (h2 : p.altitude_o_to_mn = 9)
  (h3 : p.altitude_n_to_lo = 7) :
  ∃ (lo_length : ℝ), lo_length = 19 + 2 / 7 ∧ 
  p.mn_length * p.altitude_o_to_mn = lo_length * p.altitude_n_to_lo :=
sorry

end NUMINAMATH_CALUDE_lo_length_l3386_338667


namespace NUMINAMATH_CALUDE_final_alcohol_percentage_l3386_338631

/-- Calculates the final alcohol percentage after adding pure alcohol to a solution -/
theorem final_alcohol_percentage
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (added_alcohol : ℝ)
  (h_initial_volume : initial_volume = 6)
  (h_initial_percentage : initial_percentage = 0.25)
  (h_added_alcohol : added_alcohol = 3) :
  let initial_alcohol := initial_volume * initial_percentage
  let total_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol
  let final_percentage := total_alcohol / final_volume
  final_percentage = 0.5 := by sorry

end NUMINAMATH_CALUDE_final_alcohol_percentage_l3386_338631


namespace NUMINAMATH_CALUDE_prob_both_selected_l3386_338699

/-- The probability of brother X being selected -/
def prob_X : ℚ := 1/5

/-- The probability of brother Y being selected -/
def prob_Y : ℚ := 2/3

/-- Theorem: The probability of both brothers X and Y being selected is 2/15 -/
theorem prob_both_selected : prob_X * prob_Y = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_selected_l3386_338699


namespace NUMINAMATH_CALUDE_sam_football_games_l3386_338690

theorem sam_football_games (games_this_year games_last_year : ℕ) 
  (h1 : games_this_year = 14)
  (h2 : games_last_year = 29) :
  games_this_year + games_last_year = 43 := by
  sorry

end NUMINAMATH_CALUDE_sam_football_games_l3386_338690


namespace NUMINAMATH_CALUDE_divisible_by_nine_l3386_338635

theorem divisible_by_nine (k : ℕ+) : 9 ∣ (3 * (2 + 7^(k : ℕ))) := by sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l3386_338635


namespace NUMINAMATH_CALUDE_ellipse_a_range_l3386_338664

/-- An ellipse with equation (x^2)/(a-5) + (y^2)/2 = 1 and foci on the x-axis -/
structure Ellipse (a : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (a - 5) + y^2 / 2 = 1
  foci_on_x : True  -- This is a placeholder for the condition that foci are on the x-axis

/-- The range of values for a in the given ellipse -/
theorem ellipse_a_range (a : ℝ) (e : Ellipse a) : a > 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_a_range_l3386_338664


namespace NUMINAMATH_CALUDE_f_neg_two_eq_neg_twenty_two_l3386_338633

/-- The function f(x) = x^3 - 3x^2 + x -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + x

/-- Theorem: The value of f(-2) is -22 -/
theorem f_neg_two_eq_neg_twenty_two : f (-2) = -22 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_neg_twenty_two_l3386_338633


namespace NUMINAMATH_CALUDE_remainder_of_first_six_primes_sum_divided_by_seventh_prime_l3386_338620

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  ∃ (q : ℕ), 41 = 17 * q + 7 :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_of_first_six_primes_sum_divided_by_seventh_prime_l3386_338620


namespace NUMINAMATH_CALUDE_select_two_from_nine_l3386_338609

theorem select_two_from_nine (n : ℕ) (k : ℕ) : n = 9 → k = 2 → Nat.choose n k = 36 := by
  sorry

end NUMINAMATH_CALUDE_select_two_from_nine_l3386_338609


namespace NUMINAMATH_CALUDE_base_n_representation_of_b_l3386_338641

/-- Represents a number in base n -/
def BaseNRepr (n : ℕ) (x : ℕ) : Prop :=
  ∃ (d₀ d₁ : ℕ), x = d₁ * n + d₀ ∧ d₁ < n ∧ d₀ < n

theorem base_n_representation_of_b
  (n : ℕ)
  (hn : n > 9)
  (a b : ℕ)
  (heq : n^2 - a*n + b = 0)
  (ha : BaseNRepr n a ∧ a = 19) :
  BaseNRepr n b ∧ b = 90 := by
  sorry

end NUMINAMATH_CALUDE_base_n_representation_of_b_l3386_338641


namespace NUMINAMATH_CALUDE_final_crayons_count_l3386_338682

def initial_crayons : ℝ := 7.5
def mary_took : ℝ := 3.2
def mark_took : ℝ := 0.5
def jane_took : ℝ := 1.3
def mary_returned : ℝ := 0.7
def sarah_added : ℝ := 3.5
def tom_added : ℝ := 2.8
def alice_took : ℝ := 1.5

theorem final_crayons_count :
  initial_crayons - mary_took - mark_took - jane_took + mary_returned + sarah_added + tom_added - alice_took = 8 := by
  sorry

end NUMINAMATH_CALUDE_final_crayons_count_l3386_338682


namespace NUMINAMATH_CALUDE_zinc_copper_mixture_l3386_338687

theorem zinc_copper_mixture (total_weight : ℝ) (zinc_ratio copper_ratio : ℕ) : 
  total_weight = 70 →
  zinc_ratio = 9 →
  copper_ratio = 11 →
  (zinc_ratio : ℝ) / ((zinc_ratio : ℝ) + (copper_ratio : ℝ)) * total_weight = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_zinc_copper_mixture_l3386_338687


namespace NUMINAMATH_CALUDE_restaurant_bill_l3386_338639

theorem restaurant_bill (total_friends : ℕ) (contributing_friends : ℕ) (extra_payment : ℚ) (total_bill : ℚ) :
  total_friends = 10 →
  contributing_friends = 9 →
  extra_payment = 3 →
  total_bill = (contributing_friends * (total_bill / total_friends + extra_payment)) →
  total_bill = 270 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_l3386_338639


namespace NUMINAMATH_CALUDE_circle_properties_l3386_338615

-- Define the circle P
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def intersects_x_axis (P : Circle) : Prop :=
  P.radius^2 - P.center.2^2 = 2

def intersects_y_axis (P : Circle) : Prop :=
  P.radius^2 - P.center.1^2 = 3

def distance_to_y_eq_x (P : Circle) : Prop :=
  |P.center.2 - P.center.1| = 1

-- Define the theorem
theorem circle_properties (P : Circle) 
  (hx : intersects_x_axis P) 
  (hy : intersects_y_axis P) 
  (hd : distance_to_y_eq_x P) : 
  (∃ a b : ℝ, P.center = (a, b) ∧ b^2 - a^2 = 1) ∧ 
  ((P.center = (0, 1) ∧ P.radius = Real.sqrt 3) ∨ 
   (P.center = (0, -1) ∧ P.radius = Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3386_338615


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3386_338649

/-- An ellipse with equation x²/a² + y²/9 = 1, where a > 3 -/
structure Ellipse where
  a : ℝ
  h_a : a > 3

/-- The foci of the ellipse -/
structure Foci (e : Ellipse) where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  h_dist : dist F₁ F₂ = 8

/-- A chord AB passing through F₁ -/
structure Chord (e : Ellipse) (f : Foci e) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_pass : A.1 = f.F₁.1 ∧ A.2 = f.F₁.2

/-- The theorem stating that the perimeter of triangle ABF₂ is 20 -/
theorem triangle_perimeter (e : Ellipse) (f : Foci e) (c : Chord e f) :
  dist c.A c.B + dist c.B f.F₂ + dist c.A f.F₂ = 20 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3386_338649


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l3386_338693

/-- A quadratic function satisfying the given condition -/
noncomputable def f : ℝ → ℝ :=
  fun x => -(1/2) * x^2 - 2*x

/-- Function g defined in terms of f -/
noncomputable def g : ℝ → ℝ :=
  fun x => x * Real.log x + f x

/-- The set of real numbers satisfying the inequality -/
def solution_set : Set ℝ :=
  {x | x ∈ Set.Icc (-2) (-1) ∪ Set.Ioc 0 1}

theorem quadratic_function_theorem :
  (∀ x, f (x + 1) + f x = -x^2 - 5*x - 5/2) ∧
  (f = fun x => -(1/2) * x^2 - 2*x) ∧
  (∀ x, x > 0 → (g (x^2 + x) ≥ g 2 ↔ x ∈ solution_set)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l3386_338693


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l3386_338602

/-- A quadratic polynomial q(x) satisfying specific conditions -/
def q (x : ℚ) : ℚ := (6/7) * x^2 - (2/7) * x + 2

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions : 
  q (-2) = 6 ∧ q 0 = 2 ∧ q 3 = 8 := by
  sorry

#eval q (-2)
#eval q 0
#eval q 3

end NUMINAMATH_CALUDE_q_satisfies_conditions_l3386_338602


namespace NUMINAMATH_CALUDE_jungkook_smallest_number_l3386_338692

-- Define the set of students
inductive Student : Type
| Yoongi : Student
| Jungkook : Student
| Yuna : Student
| Yoojung : Student
| Taehyung : Student

-- Define a function that assigns numbers to students
def studentNumber : Student → ℕ
| Student.Yoongi => 7
| Student.Jungkook => 6
| Student.Yuna => 9
| Student.Yoojung => 8
| Student.Taehyung => 10

-- Theorem: Jungkook has the smallest number
theorem jungkook_smallest_number :
  ∀ s : Student, studentNumber Student.Jungkook ≤ studentNumber s :=
by sorry

end NUMINAMATH_CALUDE_jungkook_smallest_number_l3386_338692


namespace NUMINAMATH_CALUDE_sqrt_twelve_simplification_l3386_338694

theorem sqrt_twelve_simplification : Real.sqrt 12 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_simplification_l3386_338694


namespace NUMINAMATH_CALUDE_max_distance_point_to_line_l3386_338647

/-- The maximum distance from point A(1,1) to the line x*cos(θ) + y*sin(θ) - 2 = 0 -/
theorem max_distance_point_to_line :
  let A : ℝ × ℝ := (1, 1)
  let line (θ : ℝ) (x y : ℝ) := x * Real.cos θ + y * Real.sin θ - 2 = 0
  let distance (θ : ℝ) := |Real.cos θ + Real.sin θ - 2| / Real.sqrt (Real.cos θ ^ 2 + Real.sin θ ^ 2)
  (∀ θ : ℝ, distance θ ≤ 2 + Real.sqrt 2) ∧ (∃ θ : ℝ, distance θ = 2 + Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_max_distance_point_to_line_l3386_338647


namespace NUMINAMATH_CALUDE_candy_count_l3386_338624

/-- The number of bags of candy -/
def num_bags : ℕ := 26

/-- The number of candy pieces in each bag -/
def pieces_per_bag : ℕ := 33

/-- The total number of candy pieces -/
def total_pieces : ℕ := num_bags * pieces_per_bag

theorem candy_count : total_pieces = 858 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_l3386_338624


namespace NUMINAMATH_CALUDE_mo_hot_chocolate_consumption_l3386_338636

/-- Represents the drinking habits of Mo --/
structure MoDrinkingHabits where
  rainyDayHotChocolate : ℚ
  nonRainyDayTea : ℕ
  totalCups : ℕ
  teaMoreThanHotChocolate : ℕ
  rainyDays : ℕ

/-- Theorem stating Mo's hot chocolate consumption on rainy mornings --/
theorem mo_hot_chocolate_consumption (mo : MoDrinkingHabits)
  (h1 : mo.nonRainyDayTea = 3)
  (h2 : mo.totalCups = 20)
  (h3 : mo.teaMoreThanHotChocolate = 10)
  (h4 : mo.rainyDays = 2)
  (h5 : (7 - mo.rainyDays) * mo.nonRainyDayTea + mo.rainyDays * mo.rainyDayHotChocolate = mo.totalCups)
  (h6 : (7 - mo.rainyDays) * mo.nonRainyDayTea = mo.rainyDays * mo.rainyDayHotChocolate + mo.teaMoreThanHotChocolate) :
  mo.rainyDayHotChocolate = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_mo_hot_chocolate_consumption_l3386_338636


namespace NUMINAMATH_CALUDE_no_perfect_squares_l3386_338643

theorem no_perfect_squares (x y z t : ℕ+) : 
  (x * y : ℤ) - (z * t : ℤ) = (x : ℤ) + y ∧ 
  (x : ℤ) + y = (z : ℤ) + t → 
  ¬(∃ (a c : ℕ+), (x * y : ℤ) = (a * a : ℤ) ∧ (z * t : ℤ) = (c * c : ℤ)) :=
by sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l3386_338643


namespace NUMINAMATH_CALUDE_trigonometric_expression_proof_l3386_338656

theorem trigonometric_expression_proof (sin30 cos30 sin60 cos60 : ℝ) 
  (h1 : sin30 = 1/2)
  (h2 : cos30 = Real.sqrt 3 / 2)
  (h3 : sin60 = Real.sqrt 3 / 2)
  (h4 : cos60 = 1/2) :
  (1 - 1/(sin30^2)) * (1 + 1/(cos60^2)) * (1 - 1/(cos30^2)) * (1 + 1/(sin60^2)) = 35/3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_proof_l3386_338656


namespace NUMINAMATH_CALUDE_jerrys_breakfast_calories_l3386_338651

/-- Represents the number of pancakes in Jerry's breakfast -/
def num_pancakes : ℕ := 6

/-- Represents the calories per pancake -/
def calories_per_pancake : ℕ := 120

/-- Represents the number of bacon strips in Jerry's breakfast -/
def num_bacon_strips : ℕ := 2

/-- Represents the calories per bacon strip -/
def calories_per_bacon_strip : ℕ := 100

/-- Represents the calories in the bowl of cereal -/
def cereal_calories : ℕ := 200

/-- Theorem stating that the total calories in Jerry's breakfast is 1120 -/
theorem jerrys_breakfast_calories : 
  num_pancakes * calories_per_pancake + 
  num_bacon_strips * calories_per_bacon_strip + 
  cereal_calories = 1120 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_breakfast_calories_l3386_338651


namespace NUMINAMATH_CALUDE_bill_apples_left_l3386_338669

/-- The number of apples Bill has left after distributing and baking -/
def apples_left (initial_apples : ℕ) (children : ℕ) (apples_per_teacher : ℕ) 
  (teachers_per_child : ℕ) (pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - (children * apples_per_teacher * teachers_per_child) - (pies * apples_per_pie)

/-- Theorem stating that Bill has 18 apples left -/
theorem bill_apples_left : 
  apples_left 50 2 3 2 2 10 = 18 := by sorry

end NUMINAMATH_CALUDE_bill_apples_left_l3386_338669


namespace NUMINAMATH_CALUDE_brendans_morning_catch_l3386_338653

theorem brendans_morning_catch (total : ℕ) (thrown_back : ℕ) (afternoon_catch : ℕ) (dad_catch : ℕ)
  (h1 : total = 23)
  (h2 : thrown_back = 3)
  (h3 : afternoon_catch = 5)
  (h4 : dad_catch = 13) :
  total = (morning_catch - thrown_back + afternoon_catch + dad_catch) →
  morning_catch = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_brendans_morning_catch_l3386_338653


namespace NUMINAMATH_CALUDE_closest_to_99_times_9_l3386_338668

def options : List ℤ := [10000, 100, 100000, 1000, 10]

theorem closest_to_99_times_9 :
  ∀ x ∈ options, |99 * 9 - 1000| ≤ |99 * 9 - x| :=
sorry

end NUMINAMATH_CALUDE_closest_to_99_times_9_l3386_338668


namespace NUMINAMATH_CALUDE_special_triangle_area_squared_l3386_338655

/-- An equilateral triangle with vertices on the hyperbola xy = 4 and centroid at a vertex of the hyperbola -/
structure SpecialTriangle where
  -- The hyperbola equation
  hyperbola : ℝ → ℝ → Prop
  hyperbola_def : hyperbola = fun x y ↦ x * y = 4

  -- The triangle is equilateral
  is_equilateral : Prop

  -- Vertices lie on the hyperbola
  vertices_on_hyperbola : Prop

  -- Centroid is at a vertex of the hyperbola
  centroid_on_hyperbola : Prop

/-- The square of the area of the special triangle is 3888 -/
theorem special_triangle_area_squared (t : SpecialTriangle) : 
  ∃ (area : ℝ), area^2 = 3888 := by sorry

end NUMINAMATH_CALUDE_special_triangle_area_squared_l3386_338655


namespace NUMINAMATH_CALUDE_fraction_geq_one_iff_x_in_range_l3386_338610

theorem fraction_geq_one_iff_x_in_range (x : ℝ) : 2 / x ≥ 1 ↔ 0 < x ∧ x ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_fraction_geq_one_iff_x_in_range_l3386_338610


namespace NUMINAMATH_CALUDE_melon_amount_in_fruit_salad_l3386_338686

/-- Given a fruit salad with melon and berries, prove the amount of melon used. -/
theorem melon_amount_in_fruit_salad
  (total_fruit : ℝ)
  (berries : ℝ)
  (h_total : total_fruit = 0.63)
  (h_berries : berries = 0.38) :
  total_fruit - berries = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_melon_amount_in_fruit_salad_l3386_338686


namespace NUMINAMATH_CALUDE_children_count_l3386_338689

theorem children_count (pencils_per_child : ℕ) (skittles_per_child : ℕ) (total_pencils : ℕ) : 
  pencils_per_child = 2 → 
  skittles_per_child = 13 → 
  total_pencils = 18 → 
  total_pencils / pencils_per_child = 9 := by
sorry

end NUMINAMATH_CALUDE_children_count_l3386_338689


namespace NUMINAMATH_CALUDE_system_solution_l3386_338683

theorem system_solution (k j : ℝ) (h1 : 64 / k = 8) (h2 : k * j = 128) : k = 8 ∧ j = 16 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3386_338683
