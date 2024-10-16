import Mathlib

namespace NUMINAMATH_CALUDE_police_coverage_l2424_242415

-- Define the set of intersections
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

-- Define the streets as sets of intersections
def horizontal1 : Set Intersection := {Intersection.A, Intersection.B, Intersection.C, Intersection.D}
def horizontal2 : Set Intersection := {Intersection.E, Intersection.F, Intersection.G}
def horizontal3 : Set Intersection := {Intersection.H, Intersection.I, Intersection.J, Intersection.K}
def vertical1 : Set Intersection := {Intersection.A, Intersection.E, Intersection.H}
def vertical2 : Set Intersection := {Intersection.B, Intersection.F, Intersection.I}
def vertical3 : Set Intersection := {Intersection.D, Intersection.G, Intersection.J}
def diagonal1 : Set Intersection := {Intersection.H, Intersection.F, Intersection.C}
def diagonal2 : Set Intersection := {Intersection.C, Intersection.G, Intersection.K}

-- Define the set of all streets
def allStreets : Set (Set Intersection) :=
  {horizontal1, horizontal2, horizontal3, vertical1, vertical2, vertical3, diagonal1, diagonal2}

-- Define the set of intersections with police officers
def policeLocations : Set Intersection := {Intersection.B, Intersection.G, Intersection.H}

-- Theorem statement
theorem police_coverage :
  ∀ street ∈ allStreets, ∃ intersection ∈ street, intersection ∈ policeLocations :=
by sorry

end NUMINAMATH_CALUDE_police_coverage_l2424_242415


namespace NUMINAMATH_CALUDE_jerry_color_cartridges_l2424_242467

/-- Represents the cost of a color cartridge in dollars -/
def color_cartridge_cost : ℕ := 32

/-- Represents the cost of a black-and-white cartridge in dollars -/
def bw_cartridge_cost : ℕ := 27

/-- Represents the total amount Jerry pays in dollars -/
def total_cost : ℕ := 123

/-- Represents the number of black-and-white cartridges Jerry needs -/
def bw_cartridges : ℕ := 1

theorem jerry_color_cartridges :
  ∃ (c : ℕ), c * color_cartridge_cost + bw_cartridges * bw_cartridge_cost = total_cost ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_jerry_color_cartridges_l2424_242467


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l2424_242400

/-- Proves that the charge for each additional 1/5 mile is $0.40 --/
theorem taxi_fare_calculation (initial_charge : ℚ) (total_distance : ℚ) (total_charge : ℚ) :
  initial_charge = 280/100 →
  total_distance = 8 →
  total_charge = 1840/100 →
  let additional_distance : ℚ := total_distance - 1/5
  let additional_increments : ℚ := additional_distance / (1/5)
  let charge_per_increment : ℚ := (total_charge - initial_charge) / additional_increments
  charge_per_increment = 40/100 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_calculation_l2424_242400


namespace NUMINAMATH_CALUDE_bianca_points_l2424_242494

/-- Calculates the points earned for recycling cans given the total number of bags, 
    number of bags not recycled, and points per bag. -/
def points_earned (total_bags : ℕ) (bags_not_recycled : ℕ) (points_per_bag : ℕ) : ℕ :=
  (total_bags - bags_not_recycled) * points_per_bag

/-- Proves that Bianca earned 45 points for recycling cans. -/
theorem bianca_points : points_earned 17 8 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_bianca_points_l2424_242494


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l2424_242489

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geo : GeometricSequence a)
  (h_relation : a 7 = a 6 + 2 * a 5)
  (h_product : ∃ (m n : ℕ), m ≠ n ∧ a m * a n = 16 * (a 1)^2) :
  (∃ (m n : ℕ), m ≠ n ∧ 1 / m + 4 / n = 3 / 2) ∧
  (∀ (m n : ℕ), m ≠ n → 1 / m + 4 / n ≥ 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l2424_242489


namespace NUMINAMATH_CALUDE_ball_bearing_bulk_discount_percentage_l2424_242435

/-- Calculates the bulk discount percentage for John's ball bearing purchase --/
theorem ball_bearing_bulk_discount_percentage : 
  let num_machines : ℕ := 10
  let bearings_per_machine : ℕ := 30
  let normal_price : ℚ := 1
  let sale_price : ℚ := 3/4
  let total_savings : ℚ := 120
  let total_bearings := num_machines * bearings_per_machine
  let original_cost := total_bearings * normal_price
  let sale_cost := total_bearings * sale_price
  let discounted_cost := original_cost - total_savings
  let bulk_discount := sale_cost - discounted_cost
  let discount_percentage := (bulk_discount / sale_cost) * 100
  discount_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_ball_bearing_bulk_discount_percentage_l2424_242435


namespace NUMINAMATH_CALUDE_dictionary_page_count_l2424_242458

/-- Count the occurrences of digit 1 in a number -/
def countOnesInNumber (n : ℕ) : ℕ := sorry

/-- Count the occurrences of digit 1 in a range of numbers from 1 to n -/
def countOnesInRange (n : ℕ) : ℕ := sorry

/-- The number of pages in the dictionary -/
def dictionaryPages : ℕ := 3152

/-- The total count of digit 1 appearances -/
def totalOnesCount : ℕ := 1988

theorem dictionary_page_count :
  countOnesInRange dictionaryPages = totalOnesCount ∧
  ∀ m : ℕ, m < dictionaryPages → countOnesInRange m < totalOnesCount :=
sorry

end NUMINAMATH_CALUDE_dictionary_page_count_l2424_242458


namespace NUMINAMATH_CALUDE_unique_nine_digit_number_l2424_242408

/-- A permutation of digits 1 to 9 -/
def Permutation9 := Fin 9 → Fin 9

/-- Checks if a function is a valid permutation of digits 1 to 9 -/
def is_valid_permutation (p : Permutation9) : Prop :=
  Function.Injective p ∧ Function.Surjective p

/-- Converts a permutation to a natural number -/
def permutation_to_nat (p : Permutation9) : ℕ :=
  (List.range 9).foldl (fun acc i => acc * 10 + (p i).val + 1) 0

/-- The property that a permutation decreases by 8 times after rearrangement -/
def decreases_by_8_times (p : Permutation9) : Prop :=
  ∃ q : Permutation9, is_valid_permutation q ∧ permutation_to_nat p = 8 * permutation_to_nat q

theorem unique_nine_digit_number :
  ∃! p : Permutation9, is_valid_permutation p ∧ decreases_by_8_times p ∧ permutation_to_nat p = 123456789 :=
sorry

end NUMINAMATH_CALUDE_unique_nine_digit_number_l2424_242408


namespace NUMINAMATH_CALUDE_ricky_magic_box_friday_l2424_242481

/-- Calculates the number of pennies in the magic money box after a given number of days -/
def pennies_after_days (initial_pennies : ℕ) (days : ℕ) : ℕ :=
  initial_pennies * 2^days

/-- Theorem: Ricky's magic money box contains 48 pennies on Friday -/
theorem ricky_magic_box_friday : pennies_after_days 3 4 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ricky_magic_box_friday_l2424_242481


namespace NUMINAMATH_CALUDE_square_and_cube_roots_l2424_242438

theorem square_and_cube_roots : 
  (∀ x : ℝ, x^2 = 4/9 ↔ x = 2/3 ∨ x = -2/3) ∧ 
  (∀ y : ℝ, y^3 = -64 ↔ y = -4) := by
  sorry

end NUMINAMATH_CALUDE_square_and_cube_roots_l2424_242438


namespace NUMINAMATH_CALUDE_dan_seashells_given_l2424_242499

/-- The number of seashells Dan gave to Jessica -/
def seashells_given (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

theorem dan_seashells_given :
  seashells_given 56 22 = 34 := by
  sorry

end NUMINAMATH_CALUDE_dan_seashells_given_l2424_242499


namespace NUMINAMATH_CALUDE_teds_fruit_purchase_cost_l2424_242443

/-- The total cost of purchasing fruits given their quantities and unit prices -/
def total_cost (banana_qty : ℕ) (orange_qty : ℕ) (apple_qty : ℕ) (grape_qty : ℕ)
                (banana_price : ℚ) (orange_price : ℚ) (apple_price : ℚ) (grape_price : ℚ) : ℚ :=
  banana_qty * banana_price + orange_qty * orange_price + 
  apple_qty * apple_price + grape_qty * grape_price

/-- Theorem stating that the total cost of Ted's fruit purchase is $47 -/
theorem teds_fruit_purchase_cost : 
  total_cost 7 15 6 4 2 1.5 1.25 0.75 = 47 := by
  sorry

end NUMINAMATH_CALUDE_teds_fruit_purchase_cost_l2424_242443


namespace NUMINAMATH_CALUDE_composite_function_evaluation_l2424_242442

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 2
def g (x : ℝ) : ℝ := 5 * x + 2

-- State the theorem
theorem composite_function_evaluation : g (f (g 1)) = 82 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_evaluation_l2424_242442


namespace NUMINAMATH_CALUDE_cats_not_eating_apples_or_chicken_l2424_242453

theorem cats_not_eating_apples_or_chicken
  (total_cats : ℕ)
  (cats_liking_apples : ℕ)
  (cats_liking_chicken : ℕ)
  (cats_liking_both : ℕ)
  (h1 : total_cats = 80)
  (h2 : cats_liking_apples = 15)
  (h3 : cats_liking_chicken = 60)
  (h4 : cats_liking_both = 10) :
  total_cats - (cats_liking_apples + cats_liking_chicken - cats_liking_both) = 15 := by
  sorry

end NUMINAMATH_CALUDE_cats_not_eating_apples_or_chicken_l2424_242453


namespace NUMINAMATH_CALUDE_base4_21012_equals_582_l2424_242416

/-- Converts a base 4 digit to its base 10 equivalent -/
def base4_digit_to_base10 (d : Nat) : Nat :=
  if d < 4 then d else 0

/-- Represents the base 4 number 21012 -/
def base4_number : List Nat := [2, 1, 0, 1, 2]

/-- Converts a list of base 4 digits to a base 10 number -/
def base4_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + base4_digit_to_base10 d * (4 ^ (digits.length - 1 - i))) 0

theorem base4_21012_equals_582 :
  base4_to_base10 base4_number = 582 := by
  sorry

end NUMINAMATH_CALUDE_base4_21012_equals_582_l2424_242416


namespace NUMINAMATH_CALUDE_milk_production_l2424_242432

theorem milk_production (y : ℝ) (h : y > 0) : 
  let initial_production := (y + 2) / (y * (y + 3))
  let new_cows := y + 4
  let new_milk := y + 6
  (new_milk / (new_cows * initial_production)) = (y * (y + 3) * (y + 6)) / ((y + 2) * (y + 4)) :=
by sorry

end NUMINAMATH_CALUDE_milk_production_l2424_242432


namespace NUMINAMATH_CALUDE_optimization_problem_l2424_242409

theorem optimization_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  ((2 / x) + (1 / y) ≥ 9) ∧
  (4 * x^2 + y^2 ≥ 1/2) ∧
  (Real.sqrt (2 * x) + Real.sqrt y ≤ Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_optimization_problem_l2424_242409


namespace NUMINAMATH_CALUDE_inequality_proof_l2424_242471

theorem inequality_proof (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x^4 + y^4 + z^4 = 1) :
  x^3 / (1 - x^8) + y^3 / (1 - y^8) + z^3 / (1 - z^8) ≥ 9/8 * Real.rpow 3 (1/4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2424_242471


namespace NUMINAMATH_CALUDE_depth_difference_is_four_l2424_242486

/-- The depth of Mark's pond in feet -/
def marks_pond_depth : ℕ := 19

/-- The depth of Peter's pond in feet -/
def peters_pond_depth : ℕ := 5

/-- The difference between Mark's pond depth and 3 times Peter's pond depth -/
def depth_difference : ℕ := marks_pond_depth - 3 * peters_pond_depth

theorem depth_difference_is_four :
  depth_difference = 4 :=
by sorry

end NUMINAMATH_CALUDE_depth_difference_is_four_l2424_242486


namespace NUMINAMATH_CALUDE_six_people_handshakes_l2424_242403

/-- The number of unique handshakes between n people, where each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the number of handshakes between 6 people is 15. -/
theorem six_people_handshakes : handshakes 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_six_people_handshakes_l2424_242403


namespace NUMINAMATH_CALUDE_triangle_existence_l2424_242476

theorem triangle_existence (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^4 + b^4 + c^4 + 4*a^2*b^2*c^2 = 2*(a^2*b^2 + a^2*c^2 + b^2*c^2)) :
  ∃ (α β γ : ℝ), α + β + γ = π ∧ Real.sin α = a ∧ Real.sin β = b ∧ Real.sin γ = c := by
sorry

end NUMINAMATH_CALUDE_triangle_existence_l2424_242476


namespace NUMINAMATH_CALUDE_pq_length_is_ten_l2424_242480

/-- A trapezoid PQRS with specific properties -/
structure Trapezoid where
  /-- The length of side RS -/
  rs : ℝ
  /-- The tangent of angle S -/
  tan_s : ℝ
  /-- The tangent of angle Q -/
  tan_q : ℝ
  /-- PQ is parallel to RS -/
  pq_parallel_rs : True
  /-- PR is perpendicular to RS -/
  pr_perpendicular_rs : True
  /-- RS has length 15 -/
  rs_length : rs = 15
  /-- tan S = 2 -/
  tan_s_value : tan_s = 2
  /-- tan Q = 3 -/
  tan_q_value : tan_q = 3

/-- The length of PQ in the trapezoid -/
def pq_length (t : Trapezoid) : ℝ := 10

/-- Theorem stating that the length of PQ is 10 -/
theorem pq_length_is_ten (t : Trapezoid) : pq_length t = 10 := by sorry

end NUMINAMATH_CALUDE_pq_length_is_ten_l2424_242480


namespace NUMINAMATH_CALUDE_bike_lock_rotation_l2424_242421

/-- Rotates a single digit by 180 degrees on a 10-digit wheel. -/
def rotate_digit (d : Nat) : Nat :=
  (d + 5) % 10

/-- The original code of the bike lock. -/
def original_code : List Nat := [6, 3, 4, 8]

/-- The correct code after rotation. -/
def correct_code : List Nat := [1, 8, 9, 3]

/-- Theorem stating that rotating each digit of the original code results in the correct code. -/
theorem bike_lock_rotation :
  original_code.map rotate_digit = correct_code := by
  sorry

#eval original_code.map rotate_digit

end NUMINAMATH_CALUDE_bike_lock_rotation_l2424_242421


namespace NUMINAMATH_CALUDE_bear_weight_gain_ratio_l2424_242464

theorem bear_weight_gain_ratio :
  let total_weight : ℝ := 1000
  let berry_weight : ℝ := total_weight / 5
  let small_animal_weight : ℝ := 200
  let salmon_weight : ℝ := (total_weight - berry_weight - small_animal_weight) / 2
  let acorn_weight : ℝ := total_weight - berry_weight - small_animal_weight - salmon_weight
  acorn_weight / berry_weight = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bear_weight_gain_ratio_l2424_242464


namespace NUMINAMATH_CALUDE_range_of_g_l2424_242423

def f (x : ℝ) : ℝ := 2 * x - 3

def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x ∈ Set.Icc 1 3, -29 ≤ g x ∧ g x ≤ 3 ∧
  ∀ y ∈ Set.Icc (-29) 3, ∃ x ∈ Set.Icc 1 3, g x = y :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l2424_242423


namespace NUMINAMATH_CALUDE_onion_problem_l2424_242466

theorem onion_problem (initial : ℕ) (removed : ℕ) : 
  initial + 4 - removed + 9 = initial + 8 → removed = 5 := by
  sorry

end NUMINAMATH_CALUDE_onion_problem_l2424_242466


namespace NUMINAMATH_CALUDE_total_mass_on_boat_total_mass_is_183_l2424_242478

/-- Calculates the total mass of two individuals on a boat given specific conditions. -/
theorem total_mass_on_boat (boat_length boat_breadth initial_sinking_depth : ℝ) 
  (mass_second_person : ℝ) (water_density : ℝ) : ℝ :=
  let volume_displaced := boat_length * boat_breadth * initial_sinking_depth
  let mass_first_person := volume_displaced * water_density
  mass_first_person + mass_second_person

/-- Proves that the total mass of two individuals on a boat is 183 kg under given conditions. -/
theorem total_mass_is_183 :
  total_mass_on_boat 3 2 0.018 75 1000 = 183 := by
  sorry

end NUMINAMATH_CALUDE_total_mass_on_boat_total_mass_is_183_l2424_242478


namespace NUMINAMATH_CALUDE_median_is_39_l2424_242495

/-- Represents the score distribution of students --/
structure ScoreDistribution where
  scores : List Nat
  counts : List Nat
  total_students : Nat

/-- Calculates the median of a score distribution --/
def median (sd : ScoreDistribution) : Rat :=
  sorry

/-- The specific score distribution from the problem --/
def problem_distribution : ScoreDistribution :=
  { scores := [36, 37, 38, 39, 40],
    counts := [1, 2, 1, 4, 2],
    total_students := 10 }

/-- Theorem stating that the median of the given distribution is 39 --/
theorem median_is_39 : median problem_distribution = 39 := by
  sorry

end NUMINAMATH_CALUDE_median_is_39_l2424_242495


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2424_242422

theorem max_value_of_expression (x y z : ℝ) (h : x + 3 * y + z = 5) :
  ∃ (max : ℝ), max = 125 / 4 ∧ ∀ (a b c : ℝ), a + 3 * b + c = 5 → a * b + a * c + b * c ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2424_242422


namespace NUMINAMATH_CALUDE_class_size_proof_l2424_242430

theorem class_size_proof (total : ℕ) 
  (h1 : 20 < total ∧ total < 30)
  (h2 : ∃ n : ℕ, total = 8 * n + 2)
  (h3 : ∃ M F : ℕ, M = 5 * n ∧ F = 4 * n) 
  (h4 : ∃ n : ℕ, n = (20 * M) / 100 ∧ n = (25 * F) / 100) :
  total = 26 := by
sorry

end NUMINAMATH_CALUDE_class_size_proof_l2424_242430


namespace NUMINAMATH_CALUDE_difference_of_cubes_l2424_242497

theorem difference_of_cubes (a b : ℝ) 
  (h1 : a^3 - b^3 = 4)
  (h2 : a^2 + a*b + b^2 + a - b = 4) : 
  a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_difference_of_cubes_l2424_242497


namespace NUMINAMATH_CALUDE_ashley_interest_earned_l2424_242410

/-- Calculates the total interest earned in one year given the investment conditions --/
def total_interest (contest_winnings : ℝ) (investment1 : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let investment2 := 2 * investment1 - 400
  let interest1 := investment1 * rate1
  let interest2 := investment2 * rate2
  interest1 + interest2

/-- Theorem stating that the total interest earned is $298 --/
theorem ashley_interest_earned :
  total_interest 5000 1800 0.05 0.065 = 298 := by
  sorry

end NUMINAMATH_CALUDE_ashley_interest_earned_l2424_242410


namespace NUMINAMATH_CALUDE_garden_fence_posts_l2424_242449

/-- Calculates the minimum number of fence posts needed for a rectangular garden -/
def min_fence_posts (length width post_spacing : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let wall_side := max length width
  let fenced_perimeter := perimeter - wall_side
  let posts_on_long_side := wall_side / post_spacing + 1
  let posts_on_short_sides := 2 * (fenced_perimeter - wall_side) / post_spacing
  posts_on_long_side + posts_on_short_sides

/-- Proves that for a 50m by 80m garden with 10m post spacing, 17 posts are needed -/
theorem garden_fence_posts :
  min_fence_posts 80 50 10 = 17 := by
  sorry

#eval min_fence_posts 80 50 10

end NUMINAMATH_CALUDE_garden_fence_posts_l2424_242449


namespace NUMINAMATH_CALUDE_sequence_odd_terms_l2424_242454

theorem sequence_odd_terms (a : ℕ → ℤ) 
  (h1 : a 1 = 2)
  (h2 : a 2 = 7)
  (h3 : ∀ n ≥ 2, -1/2 < (a (n+1) : ℚ) - (a n)^2 / (a (n-1))^2 ∧ 
                 (a (n+1) : ℚ) - (a n)^2 / (a (n-1))^2 ≤ 1/2) :
  ∀ n > 1, Odd (a n) := by
sorry

end NUMINAMATH_CALUDE_sequence_odd_terms_l2424_242454


namespace NUMINAMATH_CALUDE_distance_between_centers_l2424_242455

-- Define a circle in the first quadrant tangent to both axes
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_axes : center.1 = center.2
  in_first_quadrant : center.1 > 0 ∧ center.2 > 0

-- Define the property of passing through (4,1)
def passes_through_point (c : Circle) : Prop :=
  (c.center.1 - 4)^2 + (c.center.2 - 1)^2 = c.radius^2

-- Theorem statement
theorem distance_between_centers (c1 c2 : Circle)
  (h1 : passes_through_point c1)
  (h2 : passes_through_point c2)
  (h3 : c1 ≠ c2) :
  Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = 8 :=
sorry

end NUMINAMATH_CALUDE_distance_between_centers_l2424_242455


namespace NUMINAMATH_CALUDE_f_2017_equals_neg_2_l2424_242414

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_2017_equals_neg_2 (f : ℝ → ℝ) 
  (h1 : is_odd_function f)
  (h2 : is_even_function (fun x ↦ f (x + 1)))
  (h3 : f (-1) = 2) : 
  f 2017 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_2017_equals_neg_2_l2424_242414


namespace NUMINAMATH_CALUDE_expression_factorization_l2424_242418

theorem expression_factorization (x : ℝ) :
  (20 * x^3 + 100 * x - 10) - (-5 * x^3 + 5 * x - 10) = 5 * x * (5 * x^2 + 19) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2424_242418


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l2424_242445

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l2424_242445


namespace NUMINAMATH_CALUDE_decreasing_quadratic_function_l2424_242444

theorem decreasing_quadratic_function (a : ℝ) :
  (∀ x < 4, (∀ y < x, x^2 + 2*(a-1)*x + 2 < y^2 + 2*(a-1)*y + 2)) →
  a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_function_l2424_242444


namespace NUMINAMATH_CALUDE_intersection_S_T_l2424_242424

-- Define the sets S and T
def S : Set ℝ := {x | x + 1 > 0}
def T : Set ℝ := {x | 3*x - 6 < 0}

-- State the theorem
theorem intersection_S_T : S ∩ T = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_S_T_l2424_242424


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2424_242472

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  Complex.im (2 / (1 - i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2424_242472


namespace NUMINAMATH_CALUDE_min_m_plus_n_l2424_242473

theorem min_m_plus_n (m n : ℕ+) (h : 108 * m = n ^ 3) : 
  ∀ (m' n' : ℕ+), 108 * m' = n' ^ 3 → m + n ≤ m' + n' := by
  sorry

end NUMINAMATH_CALUDE_min_m_plus_n_l2424_242473


namespace NUMINAMATH_CALUDE_goat_difference_l2424_242407

-- Define the number of goats for each person
def adam_goats : ℕ := 7
def ahmed_goats : ℕ := 13

-- Define Andrew's goats in terms of Adam's
def andrew_goats : ℕ := 2 * adam_goats + 5

-- Theorem statement
theorem goat_difference : andrew_goats - ahmed_goats = 6 := by
  sorry

end NUMINAMATH_CALUDE_goat_difference_l2424_242407


namespace NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_l2424_242484

theorem neither_necessary_nor_sufficient :
  ¬(∀ x : ℝ, -1 < x ∧ x < 2 → |x - 2| < 1) ∧
  ¬(∀ x : ℝ, |x - 2| < 1 → -1 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_l2424_242484


namespace NUMINAMATH_CALUDE_marcus_paintings_l2424_242479

def paintings_per_day (day : Nat) : Nat :=
  match day with
  | 1 => 2
  | 2 => min 8 (2 * paintings_per_day 1)
  | 3 => min 8 (paintings_per_day 2 + (paintings_per_day 2 * 3) / 4)
  | 4 => min 8 (paintings_per_day 3 + (paintings_per_day 3 * 1) / 2)
  | 5 => min 8 (paintings_per_day 4 + (paintings_per_day 4 * 1) / 4)
  | _ => 0

def total_paintings : Nat :=
  paintings_per_day 1 + paintings_per_day 2 + paintings_per_day 3 + paintings_per_day 4 + paintings_per_day 5

theorem marcus_paintings :
  total_paintings = 29 :=
by sorry

end NUMINAMATH_CALUDE_marcus_paintings_l2424_242479


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2424_242436

-- Define the inverse variation relationship
def inverse_variation (a b k : ℝ) : Prop := a * b^3 = k

-- State the theorem
theorem inverse_variation_problem :
  ∀ (a₁ a₂ k : ℝ),
  inverse_variation a₁ 2 k →
  a₁ = 16 →
  inverse_variation a₂ 4 k →
  a₂ = 2 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2424_242436


namespace NUMINAMATH_CALUDE_lcm_18_27_l2424_242482

theorem lcm_18_27 : Nat.lcm 18 27 = 54 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_27_l2424_242482


namespace NUMINAMATH_CALUDE_f_value_at_8pi_over_3_l2424_242459

def f (x : ℝ) : ℝ := sorry

theorem f_value_at_8pi_over_3 
  (h_even : ∀ x, f (-x) = f x)
  (h_periodic : ∀ x, f (x + π) = f x)
  (h_def : ∀ x, 0 ≤ x → x < π/2 → f x = Real.sqrt 3 * Real.tan x - 1) :
  f (8*π/3) = 2 := by sorry

end NUMINAMATH_CALUDE_f_value_at_8pi_over_3_l2424_242459


namespace NUMINAMATH_CALUDE_jason_work_hours_l2424_242493

def after_school_rate : ℝ := 4.00
def saturday_rate : ℝ := 6.00
def total_earnings : ℝ := 88.00
def saturday_hours : ℝ := 8

def total_hours : ℝ := 18

theorem jason_work_hours :
  ∃ (after_school_hours : ℝ),
    after_school_hours * after_school_rate + saturday_hours * saturday_rate = total_earnings ∧
    after_school_hours + saturday_hours = total_hours :=
by sorry

end NUMINAMATH_CALUDE_jason_work_hours_l2424_242493


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l2424_242437

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℝ) (incorrect_value correct_value : ℝ) :
  n = 25 →
  initial_mean = 190 →
  incorrect_value = 130 →
  correct_value = 165 →
  (n * initial_mean - incorrect_value + correct_value) / n = 191.4 :=
by sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l2424_242437


namespace NUMINAMATH_CALUDE_parallel_planes_equidistant_points_l2424_242433

-- Define the concept of a plane
def Plane : Type := sorry

-- Define the concept of a point
def Point : Type := sorry

-- Define what it means for a point to be in a plane
def in_plane (p : Point) (α : Plane) : Prop := sorry

-- Define what it means for three points to be non-collinear
def non_collinear (p q r : Point) : Prop := sorry

-- Define what it means for a point to be equidistant from a plane
def equidistant_from_plane (p : Point) (β : Plane) (d : ℝ) : Prop := sorry

-- Define what it means for two planes to be parallel
def parallel_planes (α β : Plane) : Prop := sorry

-- State the theorem
theorem parallel_planes_equidistant_points (α β : Plane) :
  (∃ (p q r : Point) (d : ℝ), 
    in_plane p α ∧ in_plane q α ∧ in_plane r α ∧
    non_collinear p q r ∧
    equidistant_from_plane p β d ∧
    equidistant_from_plane q β d ∧
    equidistant_from_plane r β d) →
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_equidistant_points_l2424_242433


namespace NUMINAMATH_CALUDE_irrational_plus_five_less_than_five_necessary_for_less_than_three_l2424_242465

-- Define the property of being irrational
def IsIrrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Proposition ②
theorem irrational_plus_five (a : ℝ) : IsIrrational (a + 5) ↔ IsIrrational a := by sorry

-- Proposition ④
theorem less_than_five_necessary_for_less_than_three (a : ℝ) : a < 3 → a < 5 := by sorry

end NUMINAMATH_CALUDE_irrational_plus_five_less_than_five_necessary_for_less_than_three_l2424_242465


namespace NUMINAMATH_CALUDE_age_difference_l2424_242469

/-- The difference in total ages of (A, B) and (B, C) given C is 20 years younger than A -/
theorem age_difference (A B C : ℕ) (h : C = A - 20) : 
  (A + B) - (B + C) = 20 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2424_242469


namespace NUMINAMATH_CALUDE_sara_pumpkins_l2424_242406

/-- The number of pumpkins Sara grew -/
def pumpkins_grown : ℕ := 43

/-- The number of pumpkins eaten by rabbits -/
def pumpkins_eaten : ℕ := 23

/-- The number of pumpkins Sara has left -/
def pumpkins_left : ℕ := pumpkins_grown - pumpkins_eaten

theorem sara_pumpkins : pumpkins_left = 20 := by
  sorry

end NUMINAMATH_CALUDE_sara_pumpkins_l2424_242406


namespace NUMINAMATH_CALUDE_winter_sales_l2424_242461

/-- The number of pizzas sold in millions for each season -/
structure PizzaSales where
  spring : ℝ
  summer : ℝ
  fall : ℝ
  winter : ℝ

/-- The total number of pizzas sold in millions -/
def total_sales (sales : PizzaSales) : ℝ :=
  sales.spring + sales.summer + sales.fall + sales.winter

/-- The given conditions of the problem -/
def pizza_problem (sales : PizzaSales) : Prop :=
  sales.summer = 6 ∧
  sales.spring = 2.5 ∧
  sales.fall = 3.5 ∧
  sales.summer = 0.4 * (total_sales sales)

/-- The theorem to be proved -/
theorem winter_sales (sales : PizzaSales) :
  pizza_problem sales → sales.winter = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_winter_sales_l2424_242461


namespace NUMINAMATH_CALUDE_people_left_of_kolya_l2424_242470

/-- Given a line of people with the following conditions:
  * There are 12 people to the right of Kolya
  * There are 20 people to the left of Sasha
  * There are 8 people to the right of Sasha
  Then there are 16 people to the left of Kolya -/
theorem people_left_of_kolya 
  (total : ℕ) 
  (kolya_right : ℕ) 
  (sasha_left : ℕ) 
  (sasha_right : ℕ) 
  (h1 : kolya_right = 12)
  (h2 : sasha_left = 20)
  (h3 : sasha_right = 8)
  (h4 : total = sasha_left + sasha_right + 1) : 
  total - kolya_right - 1 = 16 := by
sorry

end NUMINAMATH_CALUDE_people_left_of_kolya_l2424_242470


namespace NUMINAMATH_CALUDE_kindergarten_ratio_l2424_242401

theorem kindergarten_ratio (boys girls : ℕ) (h1 : boys = 12) (h2 : 2 * girls = 3 * boys) : girls = 18 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_ratio_l2424_242401


namespace NUMINAMATH_CALUDE_minimum_cuts_for_251_11gons_l2424_242425

/-- Represents a cut on a piece of paper -/
structure Cut where
  -- We don't need to define the structure of a cut for this problem

/-- Represents a polygon on the paper -/
structure Polygon where
  sides : ℕ

/-- The result of applying cuts to a rectangular piece of paper -/
structure CutResult where
  cuts : ℕ
  polygons : List Polygon

/-- Function that applies cuts to a rectangular piece of paper -/
def applyCuts (numCuts : ℕ) : CutResult :=
  sorry

/-- Function that counts the number of polygons with a specific number of sides -/
def countPolygonsWithSides (result : CutResult) (sides : ℕ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem minimum_cuts_for_251_11gons :
  ∃ (n : ℕ), 
    (∀ (m : ℕ), 
      let result := applyCuts m
      countPolygonsWithSides result 11 ≥ 251 → m ≥ n) ∧
    (let result := applyCuts n
     countPolygonsWithSides result 11 = 251) ∧
    n = 2007 :=
  sorry

end NUMINAMATH_CALUDE_minimum_cuts_for_251_11gons_l2424_242425


namespace NUMINAMATH_CALUDE_marcia_project_time_l2424_242456

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of hours Marcia spent on her science project -/
def hours_spent : ℕ := 5

/-- The total number of minutes Marcia spent on her science project -/
def total_minutes : ℕ := hours_spent * minutes_per_hour

theorem marcia_project_time : total_minutes = 300 := by
  sorry

end NUMINAMATH_CALUDE_marcia_project_time_l2424_242456


namespace NUMINAMATH_CALUDE_subtracted_number_l2424_242452

theorem subtracted_number (t k x : ℝ) : 
  t = 5/9 * (k - x) → 
  t = 20 → 
  k = 68 → 
  x = 32 := by
sorry

end NUMINAMATH_CALUDE_subtracted_number_l2424_242452


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l2424_242490

theorem complex_power_magnitude : 
  Complex.abs ((1/2 : ℂ) + (Complex.I * (Real.sqrt 3)/2))^12 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l2424_242490


namespace NUMINAMATH_CALUDE_polygon_angle_theorem_l2424_242491

/-- 
Theorem: For a convex n-sided polygon with one interior angle x° and 
the sum of the remaining interior angles 2180°, x = 160° and n = 15.
-/
theorem polygon_angle_theorem (n : ℕ) (x : ℝ) 
  (h_convex : n ≥ 3)
  (h_sum : x + 2180 = 180 * (n - 2)) :
  x = 160 ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angle_theorem_l2424_242491


namespace NUMINAMATH_CALUDE_existence_of_sum_equality_l2424_242426

theorem existence_of_sum_equality (n : ℕ) (a : Fin (n + 1) → ℤ)
  (h_n : n > 3)
  (h_a : ∀ i j : Fin (n + 1), i < j → a i < a j)
  (h_lower : a 0 ≥ 1)
  (h_upper : a n ≤ 2 * n - 3) :
  ∃ (i j k l m : Fin (n + 1)),
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧
    j ≠ k ∧ j ≠ l ∧ j ≠ m ∧
    k ≠ l ∧ k ≠ m ∧
    l ≠ m ∧
    a i + a j = a k + a l ∧ a k + a l = a m :=
by sorry

end NUMINAMATH_CALUDE_existence_of_sum_equality_l2424_242426


namespace NUMINAMATH_CALUDE_used_car_selections_l2424_242412

/-- Proves that given 16 cars, 24 clients, and each client selecting 2 cars, 
    each car must be selected 3 times. -/
theorem used_car_selections (cars : ℕ) (clients : ℕ) (selections_per_client : ℕ) 
    (h1 : cars = 16) 
    (h2 : clients = 24) 
    (h3 : selections_per_client = 2) : 
  (clients * selections_per_client) / cars = 3 := by
  sorry

#check used_car_selections

end NUMINAMATH_CALUDE_used_car_selections_l2424_242412


namespace NUMINAMATH_CALUDE_largest_of_four_consecutive_evens_l2424_242427

theorem largest_of_four_consecutive_evens (a b c d : ℤ) : 
  (∀ n : ℤ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4 ∧ d = 2*n + 6) →
  a + b + c + d = 140 →
  d = 38 := by
sorry

end NUMINAMATH_CALUDE_largest_of_four_consecutive_evens_l2424_242427


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2424_242485

theorem smallest_prime_divisor_of_sum (p : Nat) 
  (h1 : Prime 7) (h2 : Prime 11) : 
  (p.Prime ∧ p ∣ (7^13 + 11^15) ∧ ∀ q, q.Prime → q ∣ (7^13 + 11^15) → p ≤ q) → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2424_242485


namespace NUMINAMATH_CALUDE_new_person_weight_l2424_242475

/-- Given a group of 8 persons where the average weight increases by 2.5 kg
    when a person weighing 50 kg is replaced, the weight of the new person is 70 kg. -/
theorem new_person_weight (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) :
  n = 8 →
  avg_increase = 2.5 →
  old_weight = 50 →
  n * avg_increase + old_weight = 70 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2424_242475


namespace NUMINAMATH_CALUDE_polynomial_identity_l2424_242429

theorem polynomial_identity (x : ℝ) : 
  (x + 2)^4 + 4*(x + 2)^3 + 6*(x + 2)^2 + 4*(x + 2) + 1 = (x + 3)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2424_242429


namespace NUMINAMATH_CALUDE_original_price_calculation_l2424_242417

theorem original_price_calculation (a b : ℝ) : 
  ∃ x : ℝ, (x - a) * (1 - 0.4) = b ∧ x = a + (5/3) * b := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2424_242417


namespace NUMINAMATH_CALUDE_square_difference_153_147_l2424_242419

theorem square_difference_153_147 : 153^2 - 147^2 = 1800 := by sorry

end NUMINAMATH_CALUDE_square_difference_153_147_l2424_242419


namespace NUMINAMATH_CALUDE_distance_travelled_l2424_242428

theorem distance_travelled (normal_speed : ℝ) (faster_speed : ℝ) (additional_distance : ℝ) :
  normal_speed = 10 →
  faster_speed = 14 →
  additional_distance = 20 →
  (∃ (actual_distance : ℝ), 
    actual_distance / normal_speed = (actual_distance + additional_distance) / faster_speed ∧
    actual_distance = 50) :=
by sorry

end NUMINAMATH_CALUDE_distance_travelled_l2424_242428


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l2424_242431

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 4}

-- Define the line
def line (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = k * p.1 - 1}

-- Define the center of the circle
def center : ℝ × ℝ := (2, 3)

-- Define the property of being tangent to y-axis
def tangent_to_y_axis (C : Set (ℝ × ℝ)) : Prop :=
  ∃ y, (0, y) ∈ C ∧ ∀ x ≠ 0, (x, y) ∉ C

-- Define the perpendicularity condition
def perpendicular (M N : ℝ × ℝ) : Prop :=
  (M.1 - center.1) * (N.1 - center.1) + (M.2 - center.2) * (N.2 - center.2) = 0

theorem circle_and_line_properties :
  tangent_to_y_axis circle_C →
  ∀ k, ∃ M N, M ∈ circle_C ∧ N ∈ circle_C ∧ M ∈ line k ∧ N ∈ line k ∧ perpendicular M N →
  (k = 1 ∨ k = 7) := by
  sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l2424_242431


namespace NUMINAMATH_CALUDE_equation_solution_l2424_242405

theorem equation_solution (x : ℝ) : 
  (Real.sqrt (6 * x^2 + 1)) / (Real.sqrt (3 * x^2 + 4)) = 2 / Real.sqrt 3 ↔ 
  x = Real.sqrt (13/6) ∨ x = -Real.sqrt (13/6) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2424_242405


namespace NUMINAMATH_CALUDE_orchard_trees_l2424_242413

theorem orchard_trees (total : ℕ) (peach : ℕ) (pear : ℕ) 
  (h1 : total = 480) 
  (h2 : pear = 3 * peach) 
  (h3 : total = peach + pear) : 
  peach = 120 ∧ pear = 360 := by
  sorry

end NUMINAMATH_CALUDE_orchard_trees_l2424_242413


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2424_242402

def U : Set ℕ := {0, 2, 4, 6, 8, 10}
def A : Set ℕ := {2, 4, 6}

theorem complement_of_A_in_U :
  U \ A = {0, 8, 10} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2424_242402


namespace NUMINAMATH_CALUDE_complex_equation_solution_complex_inequality_range_l2424_242450

-- Problem 1
theorem complex_equation_solution (z : ℂ) :
  z + Complex.abs z = 2 + 8 * I → z = -15 + 8 * I := by sorry

-- Problem 2
theorem complex_inequality_range (a : ℝ) :
  Complex.abs (3 + a * I) < 4 → -Real.sqrt 7 < a ∧ a < Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_complex_inequality_range_l2424_242450


namespace NUMINAMATH_CALUDE_correct_polynomial_sum_l2424_242440

variable (a : ℝ)
variable (A B : ℝ → ℝ)

theorem correct_polynomial_sum
  (hB : B = λ x => 3 * x^2 - 5 * x - 7)
  (hA_minus_2B : A - 2 * B = λ x => -2 * x^2 + 3 * x + 6) :
  A + 2 * B = λ x => 10 * x^2 - 17 * x - 22 :=
by sorry

end NUMINAMATH_CALUDE_correct_polynomial_sum_l2424_242440


namespace NUMINAMATH_CALUDE_prob_two_females_is_two_fifths_l2424_242483

/-- Represents the survey data for students' preferences on breeding small animal A -/
structure SurveyData where
  male_like : ℕ
  male_dislike : ℕ
  female_like : ℕ
  female_dislike : ℕ

/-- Calculates the probability of selecting two females from a stratified sample -/
def prob_two_females (data : SurveyData) : ℚ :=
  let total_like := data.male_like + data.female_like
  let female_ratio := data.female_like / total_like
  let num_females_selected := 6 * female_ratio
  (num_females_selected * (num_females_selected - 1)) / (6 * 5)

/-- The main theorem to be proved -/
theorem prob_two_females_is_two_fifths (data : SurveyData) 
  (h1 : data.male_like = 20)
  (h2 : data.male_dislike = 30)
  (h3 : data.female_like = 40)
  (h4 : data.female_dislike = 10) :
  prob_two_females data = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_females_is_two_fifths_l2424_242483


namespace NUMINAMATH_CALUDE_frequency_distribution_best_for_proportions_l2424_242496

/-- Represents different statistical measures -/
inductive StatisticalMeasure
  | Mean
  | Variance
  | Mode
  | FrequencyDistribution

/-- Represents a sample of data -/
structure Sample (α : Type) where
  data : List α

/-- Represents a range of values -/
structure Range (α : Type) where
  lower : α
  upper : α

/-- A function that determines if a statistical measure can represent proportions within a range -/
def canRepresentProportionsInRange (measure : StatisticalMeasure) : Prop :=
  ∀ (α : Type) [LinearOrder α] (sample : Sample α) (range : Range α),
    ∃ (proportion : ℝ), 0 ≤ proportion ∧ proportion ≤ 1

/-- Theorem stating that the frequency distribution is the most appropriate measure
    for understanding proportions within a range -/
theorem frequency_distribution_best_for_proportions :
  canRepresentProportionsInRange StatisticalMeasure.FrequencyDistribution ∧
  (∀ m : StatisticalMeasure, m ≠ StatisticalMeasure.FrequencyDistribution →
    ¬(canRepresentProportionsInRange m)) :=
sorry

end NUMINAMATH_CALUDE_frequency_distribution_best_for_proportions_l2424_242496


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l2424_242439

/-- Given a line segment with one endpoint at (3, -5) and midpoint at (7, -15),
    the sum of the coordinates of the other endpoint is -14. -/
theorem endpoint_coordinate_sum :
  ∀ (x y : ℝ),
  (x + 3) / 2 = 7 →
  (y - 5) / 2 = -15 →
  x + y = -14 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l2424_242439


namespace NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l2424_242457

/-- The equation of the parabola -/
def parabola_equation (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 4

/-- A point (x, y) is on the parabola if it satisfies the equation -/
def on_parabola (x y : ℝ) : Prop := x = parabola_equation y

/-- An x-intercept is a point on the parabola where y = 0 -/
def is_x_intercept (x : ℝ) : Prop := on_parabola x 0

/-- The theorem stating that the parabola has exactly one x-intercept -/
theorem parabola_has_one_x_intercept : ∃! x : ℝ, is_x_intercept x := by sorry

end NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l2424_242457


namespace NUMINAMATH_CALUDE_last_student_score_l2424_242463

theorem last_student_score (total_students : ℕ) (average_19 : ℝ) (average_20 : ℝ) :
  total_students = 20 →
  average_19 = 82 →
  average_20 = 84 →
  ∃ (last_score oliver_score : ℝ),
    (19 * average_19 + oliver_score) / total_students = average_20 ∧
    oliver_score = 2 * last_score →
    last_score = 61 := by
  sorry

end NUMINAMATH_CALUDE_last_student_score_l2424_242463


namespace NUMINAMATH_CALUDE_square_overlap_area_difference_l2424_242498

theorem square_overlap_area_difference :
  ∀ (side_large side_small overlap_area : ℝ),
    side_large = 10 →
    side_small = 7 →
    overlap_area = 9 →
    side_large > 0 →
    side_small > 0 →
    overlap_area > 0 →
    (side_large^2 - overlap_area) - (side_small^2 - overlap_area) = 51 :=
by
  sorry

end NUMINAMATH_CALUDE_square_overlap_area_difference_l2424_242498


namespace NUMINAMATH_CALUDE_solve_for_q_l2424_242477

theorem solve_for_q : ∀ (k h q : ℚ),
  (3/4 : ℚ) = k/48 ∧ 
  (3/4 : ℚ) = (h+k)/60 ∧ 
  (3/4 : ℚ) = (q-h)/80 →
  q = 69 := by sorry

end NUMINAMATH_CALUDE_solve_for_q_l2424_242477


namespace NUMINAMATH_CALUDE_pentagon_segment_parallel_and_length_l2424_242487

/-- Given a pentagon ABCDE with points P, Q, R, S on its sides and points M, N on PR and QS respectively,
    satisfying specific ratios, prove that MN is parallel to AE and its length is AE / ((k₁ + 1)(k₂ + 1)). -/
theorem pentagon_segment_parallel_and_length 
  (A B C D E P Q R S M N : ℝ × ℝ) (k₁ k₂ : ℝ) :
  -- Pentagon ABCDE
  -- Points P, Q, R, S on sides AB, BC, CD, DE respectively
  (P.1 - A.1) / (B.1 - P.1) = k₁ ∧ 
  (P.2 - A.2) / (B.2 - P.2) = k₁ ∧
  (Q.1 - B.1) / (C.1 - Q.1) = k₂ ∧
  (Q.2 - B.2) / (C.2 - Q.2) = k₂ ∧
  (R.1 - D.1) / (C.1 - R.1) = k₁ ∧
  (R.2 - D.2) / (C.2 - R.2) = k₁ ∧
  (S.1 - E.1) / (D.1 - S.1) = k₂ ∧
  (S.2 - E.2) / (D.2 - S.2) = k₂ ∧
  -- Points M and N on PR and QS respectively
  (M.1 - P.1) / (R.1 - M.1) = k₂ ∧
  (M.2 - P.2) / (R.2 - M.2) = k₂ ∧
  (N.1 - S.1) / (Q.1 - N.1) = k₁ ∧
  (N.2 - S.2) / (Q.2 - N.2) = k₁ →
  -- MN is parallel to AE
  (N.2 - M.2) / (N.1 - M.1) = (E.2 - A.2) / (E.1 - A.1) ∧
  -- Length of MN
  Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) = 
    Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) / ((k₁ + 1) * (k₂ + 1)) := by
  sorry

end NUMINAMATH_CALUDE_pentagon_segment_parallel_and_length_l2424_242487


namespace NUMINAMATH_CALUDE_class_size_is_50_l2424_242451

def original_average : ℝ := 87.26
def incorrect_score : ℝ := 89
def correct_score : ℝ := 98
def new_average : ℝ := 87.44

theorem class_size_is_50 : 
  ∃ n : ℕ, n > 0 ∧ 
  (n : ℝ) * new_average = (n : ℝ) * original_average + (correct_score - incorrect_score) ∧
  n = 50 :=
sorry

end NUMINAMATH_CALUDE_class_size_is_50_l2424_242451


namespace NUMINAMATH_CALUDE_equilateral_triangle_intersections_l2424_242468

/-- Represents a point on a side of the triangle -/
structure DivisionPoint where
  side : Fin 3
  position : Fin 11

/-- Represents a line segment from a vertex to a division point -/
structure Segment where
  vertex : Fin 3
  endpoint : DivisionPoint

/-- The number of intersection points in the described configuration -/
def intersection_points : ℕ := 301

/-- States that the number of intersection points in the described triangle configuration is 301 -/
theorem equilateral_triangle_intersections :
  ∀ (triangle : Type) (is_equilateral : triangle → Prop) 
    (divide_sides : triangle → Fin 3 → Fin 12 → DivisionPoint)
    (connect_vertices : triangle → Segment → Prop),
  (∃ (t : triangle), is_equilateral t ∧ 
    (∀ (s : Fin 3) (p : Fin 12), ∃ (dp : DivisionPoint), divide_sides t s p = dp) ∧
    (∀ (v : Fin 3) (dp : DivisionPoint), v ≠ dp.side → connect_vertices t ⟨v, dp⟩)) →
  (∃ (intersection_count : ℕ), intersection_count = intersection_points) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_intersections_l2424_242468


namespace NUMINAMATH_CALUDE_football_tournament_yardage_l2424_242462

/-- Represents a football team's yardage progress --/
structure TeamProgress where
  gains : List Int
  losses : List Int
  bonus : Int
  penalty : Int

/-- Calculates the total yardage progress for a team --/
def totalYardage (team : TeamProgress) : Int :=
  (team.gains.sum - team.losses.sum) + team.bonus - team.penalty

/-- The football tournament scenario --/
def footballTournament : Prop :=
  let teamA : TeamProgress := {
    gains := [8, 6],
    losses := [5, 3],
    bonus := 0,
    penalty := 2
  }
  let teamB : TeamProgress := {
    gains := [4, 9],
    losses := [2, 7],
    bonus := 0,
    penalty := 3
  }
  let teamC : TeamProgress := {
    gains := [2, 11],
    losses := [6, 4],
    bonus := 3,
    penalty := 4
  }
  (totalYardage teamA = 4) ∧
  (totalYardage teamB = 1) ∧
  (totalYardage teamC = 2)

theorem football_tournament_yardage : footballTournament := by
  sorry

end NUMINAMATH_CALUDE_football_tournament_yardage_l2424_242462


namespace NUMINAMATH_CALUDE_liz_team_final_deficit_l2424_242488

/-- Calculates the final deficit for Liz's team given the initial deficit and points scored in the final quarter -/
def final_deficit (initial_deficit : ℕ) (liz_free_throws : ℕ) (liz_three_pointers : ℕ) (liz_jump_shots : ℕ) (other_team_points : ℕ) : ℕ :=
  let liz_points := liz_free_throws + 3 * liz_three_pointers + 2 * liz_jump_shots
  let point_difference := liz_points - other_team_points
  initial_deficit - point_difference

theorem liz_team_final_deficit :
  final_deficit 20 5 3 4 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_liz_team_final_deficit_l2424_242488


namespace NUMINAMATH_CALUDE_parallelogram_reflection_theorem_l2424_242492

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the reflection across x-axis
def reflectXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

-- Define the reflection across y=x-1
def reflectYEqXMinus1 (p : Point2D) : Point2D :=
  { x := p.y + 1, y := p.x - 1 }

-- Define the composite transformation
def compositeTransform (p : Point2D) : Point2D :=
  reflectYEqXMinus1 (reflectXAxis p)

-- Theorem statement
theorem parallelogram_reflection_theorem (E F G H : Point2D)
  (hE : E = { x := 3, y := 3 })
  (hF : F = { x := 6, y := 7 })
  (hG : G = { x := 9, y := 3 })
  (hH : H = { x := 6, y := -1 }) :
  compositeTransform H = { x := 2, y := 5 } := by sorry

end NUMINAMATH_CALUDE_parallelogram_reflection_theorem_l2424_242492


namespace NUMINAMATH_CALUDE_certain_number_problem_l2424_242420

theorem certain_number_problem (x y : ℝ) 
  (h1 : 0.25 * x = 0.15 * y - 20) 
  (h2 : x = 820) : 
  y = 1500 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2424_242420


namespace NUMINAMATH_CALUDE_linear_functions_relation_l2424_242441

/-- Given two linear functions f and g, prove that A + B = 2A under certain conditions -/
theorem linear_functions_relation (A B : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = A * x + B + 1)
  (hg : ∀ x, g x = B * x + A - 1)
  (hAB : A ≠ -B)
  (h_comp : ∀ x, f (g x) - g (f x) = A - 2 * B) :
  A + B = 2 * A :=
sorry

end NUMINAMATH_CALUDE_linear_functions_relation_l2424_242441


namespace NUMINAMATH_CALUDE_total_birds_caught_l2424_242434

-- Define the number of birds hunted during the day
def day_hunt : ℕ := 15

-- Define the success rate during the day
def day_success_rate : ℚ := 3/5

-- Define the number of birds hunted at night
def night_hunt : ℕ := 25

-- Define the success rate at night
def night_success_rate : ℚ := 4/5

-- Define the relationship between day and night catches
def night_day_ratio : ℕ := 2

-- Theorem statement
theorem total_birds_caught :
  ⌊(day_hunt : ℚ) * day_success_rate⌋ +
  ⌊(night_hunt : ℚ) * night_success_rate⌋ = 29 := by
  sorry


end NUMINAMATH_CALUDE_total_birds_caught_l2424_242434


namespace NUMINAMATH_CALUDE_abc_product_l2424_242404

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * (b + c) = 154) (h2 : b * (c + a) = 164) (h3 : c * (a + b) = 172) :
  a * b * c = Real.sqrt 538083 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l2424_242404


namespace NUMINAMATH_CALUDE_reciprocal_ratio_sum_inequality_l2424_242460

theorem reciprocal_ratio_sum_inequality (a b : ℝ) (h : a * b < 0) :
  b / a + a / b ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_ratio_sum_inequality_l2424_242460


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2424_242446

theorem container_volume_ratio : 
  ∀ (C D : ℝ), C > 0 → D > 0 → (3/4 * C = 2/3 * D) → C / D = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l2424_242446


namespace NUMINAMATH_CALUDE_cab_driver_income_l2424_242447

/-- The cab driver's income problem -/
theorem cab_driver_income (day1 day2 day3 day5 : ℕ) (average : ℕ) 
  (h1 : day1 = 400)
  (h2 : day2 = 250)
  (h3 : day3 = 650)
  (h5 : day5 = 500)
  (h_avg : average = 440)
  (h_total : day1 + day2 + day3 + day5 + (5 * average - (day1 + day2 + day3 + day5)) = 5 * average) :
  5 * average - (day1 + day2 + day3 + day5) = 400 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_l2424_242447


namespace NUMINAMATH_CALUDE_isosceles_triangle_smallest_base_l2424_242448

theorem isosceles_triangle_smallest_base 
  (α : ℝ) 
  (q : ℝ) 
  (h_α : 0 < α ∧ α < π) 
  (h_q : q > 0) :
  let base (a : ℝ) := 
    Real.sqrt (q^2 * ((1 - Real.cos α) / 2) + 2 * (1 + Real.cos α) * (a - q/2)^2)
  ∀ a, 0 < a ∧ a < q → base (q/2) ≤ base a :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_smallest_base_l2424_242448


namespace NUMINAMATH_CALUDE_sum_of_multiples_l2424_242474

def smallest_two_digit_multiple_of_5 : ℕ := 10

def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem sum_of_multiples : 
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l2424_242474


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2424_242411

theorem polynomial_division_theorem :
  let p (z : ℝ) := 4 * z^3 - 8 * z^2 + 9 * z - 7
  let d (z : ℝ) := 4 * z + 2
  let q (z : ℝ) := z^2 - 2.5 * z + 3.5
  let r : ℝ := -14
  ∀ z : ℝ, p z = d z * q z + r := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2424_242411
