import Mathlib

namespace NUMINAMATH_CALUDE_abby_coins_l2774_277410

theorem abby_coins (total_coins : ℕ) (total_value : ℚ) 
  (h_total_coins : total_coins = 23)
  (h_total_value : total_value = 455/100)
  : ∃ (quarters nickels : ℕ),
    quarters + nickels = total_coins ∧
    (25 * quarters + 5 * nickels : ℚ) / 100 = total_value ∧
    quarters = 17 := by
  sorry

end NUMINAMATH_CALUDE_abby_coins_l2774_277410


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2774_277450

/-- The total surface area of a rectangular solid -/
def totalSurfaceArea (length width depth : ℝ) : ℝ :=
  2 * (length * width + width * depth + length * depth)

/-- Theorem: The total surface area of a rectangular solid with length 10 meters, 
    width 9 meters, and depth 6 meters is 408 square meters -/
theorem rectangular_solid_surface_area :
  totalSurfaceArea 10 9 6 = 408 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2774_277450


namespace NUMINAMATH_CALUDE_validSelectionsCount_l2774_277424

/-- Represents the set of available colors --/
inductive Color
| Red
| Blue
| Yellow
| Green

/-- Represents a ball with a color and number --/
structure Ball where
  color : Color
  number : Fin 6

/-- The set of all balls --/
def allBalls : Finset Ball :=
  sorry

/-- Checks if three numbers are non-consecutive --/
def areNonConsecutive (n1 n2 n3 : Fin 6) : Prop :=
  sorry

/-- Checks if three balls have different colors --/
def haveDifferentColors (b1 b2 b3 : Ball) : Prop :=
  sorry

/-- The set of valid selections of 3 balls --/
def validSelections : Finset (Fin 24 × Fin 24 × Fin 24) :=
  sorry

theorem validSelectionsCount :
  Finset.card validSelections = 96 := by
  sorry

end NUMINAMATH_CALUDE_validSelectionsCount_l2774_277424


namespace NUMINAMATH_CALUDE_distance_to_town_l2774_277490

theorem distance_to_town (d : ℝ) : 
  (∀ x, x ≥ 6 → d < x) →  -- A's statement is false
  (∀ y, y ≤ 5 → d > y) →  -- B's statement is false
  (∀ z, z ≤ 4 → d > z) →  -- C's statement is false
  d ∈ Set.Ioo 5 6 := by
sorry

end NUMINAMATH_CALUDE_distance_to_town_l2774_277490


namespace NUMINAMATH_CALUDE_odd_m_triple_g_65_l2774_277452

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem odd_m_triple_g_65 (m : ℤ) (h_odd : m % 2 = 1) :
  g (g (g m)) = 65 → m = 255 := by sorry

end NUMINAMATH_CALUDE_odd_m_triple_g_65_l2774_277452


namespace NUMINAMATH_CALUDE_jelly_bean_theorem_l2774_277482

def jelly_bean_problem (total_jelly_beans : ℕ) (total_people : ℕ) (last_four_take : ℕ) (remaining : ℕ) : Prop :=
  let last_four_total := 4 * last_four_take
  let taken_by_others := total_jelly_beans - remaining - last_four_total
  let others_take_each := 2 * last_four_take
  let num_others := taken_by_others / others_take_each
  num_others = 6 ∧ 
  num_others + 4 = total_people ∧
  taken_by_others + last_four_total + remaining = total_jelly_beans

theorem jelly_bean_theorem : jelly_bean_problem 8000 10 400 1600 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_theorem_l2774_277482


namespace NUMINAMATH_CALUDE_vector_c_solution_l2774_277480

theorem vector_c_solution (a b c : ℝ × ℝ) : 
  a = (1, 2) → 
  b = (2, -3) → 
  (∃ k : ℝ, c + a = k • b) →
  c • (a + b) = 0 →
  c = (-7/9, -7/3) := by sorry

end NUMINAMATH_CALUDE_vector_c_solution_l2774_277480


namespace NUMINAMATH_CALUDE_store_pricing_theorem_l2774_277438

/-- Represents the cost of pencils and notebooks in a store -/
structure StorePricing where
  pencil_price : ℝ
  notebook_price : ℝ
  h1 : 9 * pencil_price + 5 * notebook_price = 3.45
  h2 : 6 * pencil_price + 4 * notebook_price = 2.40

/-- The cost of 18 pencils and 9 notebooks is $6.75 -/
theorem store_pricing_theorem (sp : StorePricing) :
  18 * sp.pencil_price + 9 * sp.notebook_price = 6.75 := by
  sorry


end NUMINAMATH_CALUDE_store_pricing_theorem_l2774_277438


namespace NUMINAMATH_CALUDE_inequality_theorem_l2774_277454

/-- The function f(x, y) = ax² + 2bxy + cy² -/
def f (a b c x y : ℝ) : ℝ := a * x^2 + 2 * b * x * y + c * y^2

/-- The main theorem -/
theorem inequality_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_nonneg : ∀ (x y : ℝ), 0 ≤ f a b c x y) :
  ∀ (x₁ x₂ y₁ y₂ : ℝ),
    Real.sqrt (f a b c x₁ y₁ * f a b c x₂ y₂) * f a b c (x₁ - x₂) (y₁ - y₂) ≥
    (a * c - b^2) * (x₁ * y₂ - x₂ * y₁)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2774_277454


namespace NUMINAMATH_CALUDE_oxygen_atoms_count_l2774_277441

-- Define the atomic weights
def carbon_weight : ℕ := 12
def hydrogen_weight : ℕ := 1
def oxygen_weight : ℕ := 16

-- Define the number of Carbon and Hydrogen atoms
def carbon_atoms : ℕ := 2
def hydrogen_atoms : ℕ := 4

-- Define the total molecular weight of the compound
def total_weight : ℕ := 60

-- Theorem to prove
theorem oxygen_atoms_count :
  let carbon_hydrogen_weight := carbon_atoms * carbon_weight + hydrogen_atoms * hydrogen_weight
  let oxygen_weight_total := total_weight - carbon_hydrogen_weight
  oxygen_weight_total / oxygen_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_oxygen_atoms_count_l2774_277441


namespace NUMINAMATH_CALUDE_equidistant_points_in_quadrants_I_II_l2774_277404

/-- A point on the line 3x + 5y = 15 that is equidistant from the coordinate axes -/
def equidistant_point (x y : ℝ) : Prop :=
  3 * x + 5 * y = 15 ∧ (x = y ∨ x = -y)

/-- The point is in quadrant I -/
def in_quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- The point is in quadrant II -/
def in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The point is in quadrant III -/
def in_quadrant_III (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- The point is in quadrant IV -/
def in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem equidistant_points_in_quadrants_I_II :
  ∀ x y : ℝ, equidistant_point x y → (in_quadrant_I x y ∨ in_quadrant_II x y) ∧
  ¬(in_quadrant_III x y ∨ in_quadrant_IV x y) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_points_in_quadrants_I_II_l2774_277404


namespace NUMINAMATH_CALUDE_simplify_expression_l2774_277422

theorem simplify_expression (y : ℝ) : 4*y + 5*y + 6*y + 2 = 15*y + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2774_277422


namespace NUMINAMATH_CALUDE_correct_addition_after_digit_change_l2774_277442

theorem correct_addition_after_digit_change :
  let num1 : ℕ := 364765
  let num2 : ℕ := 951872
  let incorrect_sum : ℕ := 1496637
  let d : ℕ := 3
  let e : ℕ := 4
  let new_num1 : ℕ := num1 + 100000 * (e - d)
  let new_num2 : ℕ := num2
  let new_sum : ℕ := incorrect_sum + 100000 * (e - d)
  new_num1 + new_num2 = new_sum ∧ d + e = 7 :=
by sorry

end NUMINAMATH_CALUDE_correct_addition_after_digit_change_l2774_277442


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l2774_277428

theorem stratified_sampling_problem (teachers : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (female_sample : ℕ) (total_sample : ℕ) : 
  teachers = 200 → 
  male_students = 1200 → 
  female_students = 1000 → 
  female_sample = 80 → 
  (female_students : ℚ) / (teachers + male_students + female_students : ℚ) * total_sample = female_sample →
  total_sample = 192 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l2774_277428


namespace NUMINAMATH_CALUDE_traci_road_trip_l2774_277446

/-- Proves that the fraction of the remaining distance traveled between the first and second stops is 1/4 -/
theorem traci_road_trip (total_distance : ℝ) (first_stop_fraction : ℝ) (final_leg : ℝ) : 
  total_distance = 600 →
  first_stop_fraction = 1/3 →
  final_leg = 300 →
  (total_distance - first_stop_fraction * total_distance - final_leg) / (total_distance - first_stop_fraction * total_distance) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_traci_road_trip_l2774_277446


namespace NUMINAMATH_CALUDE_floor_ceiling_difference_l2774_277409

theorem floor_ceiling_difference : ⌊(1.999 : ℝ)⌋ - ⌈(3.001 : ℝ)⌉ = -3 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_difference_l2774_277409


namespace NUMINAMATH_CALUDE_mary_earnings_per_home_l2774_277496

/-- Given that Mary earned $12696 cleaning 276.0 homes, prove that she earns $46 per home. -/
theorem mary_earnings_per_home :
  let total_earnings : ℚ := 12696
  let homes_cleaned : ℚ := 276
  total_earnings / homes_cleaned = 46 := by
  sorry

end NUMINAMATH_CALUDE_mary_earnings_per_home_l2774_277496


namespace NUMINAMATH_CALUDE_trapezoid_area_theorem_l2774_277415

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  diagonal1 : ℝ
  diagonal2 : ℝ
  midpoint_segment : ℝ

/-- The area of a trapezoid with the given properties -/
def trapezoid_area (t : Trapezoid) : ℝ := 6

/-- Theorem: The area of a trapezoid with diagonals 3 and 5, and midpoint segment 2, is 6 -/
theorem trapezoid_area_theorem (t : Trapezoid) 
  (h1 : t.diagonal1 = 3) 
  (h2 : t.diagonal2 = 5) 
  (h3 : t.midpoint_segment = 2) : 
  trapezoid_area t = 6 := by
  sorry

#check trapezoid_area_theorem

end NUMINAMATH_CALUDE_trapezoid_area_theorem_l2774_277415


namespace NUMINAMATH_CALUDE_tree_planting_event_l2774_277419

/-- Calculates 60% of the total number of participants in a tree planting event -/
theorem tree_planting_event (boys : ℕ) (girls : ℕ) : 
  boys = 600 →
  girls - boys = 400 →
  girls > boys →
  (boys + girls) * 60 / 100 = 960 := by
sorry

end NUMINAMATH_CALUDE_tree_planting_event_l2774_277419


namespace NUMINAMATH_CALUDE_total_steps_in_week_l2774_277467

/-- Represents the number of steps taken to school and back for each day -/
structure DailySteps where
  toSchool : ℕ
  fromSchool : ℕ

/-- Calculates the total steps for a given day -/
def totalSteps (day : DailySteps) : ℕ := day.toSchool + day.fromSchool

/-- Represents Raine's walking data for the week -/
structure WeeklyWalk where
  monday : DailySteps
  tuesday : DailySteps
  wednesday : DailySteps
  thursday : DailySteps
  friday : DailySteps

/-- The actual walking data for Raine's week -/
def rainesWeek : WeeklyWalk := {
  monday := { toSchool := 150, fromSchool := 170 }
  tuesday := { toSchool := 140, fromSchool := 170 }  -- 140 + 30 rest stop
  wednesday := { toSchool := 160, fromSchool := 210 }
  thursday := { toSchool := 150, fromSchool := 170 }  -- 140 + 30 rest stop
  friday := { toSchool := 180, fromSchool := 200 }
}

/-- Theorem: The total number of steps Raine takes in five days is 1700 -/
theorem total_steps_in_week (w : WeeklyWalk := rainesWeek) :
  totalSteps w.monday + totalSteps w.tuesday + totalSteps w.wednesday +
  totalSteps w.thursday + totalSteps w.friday = 1700 := by
  sorry

end NUMINAMATH_CALUDE_total_steps_in_week_l2774_277467


namespace NUMINAMATH_CALUDE_base_seven_23456_equals_6068_l2774_277472

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_seven_23456_equals_6068 :
  base_seven_to_ten [6, 5, 4, 3, 2] = 6068 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_23456_equals_6068_l2774_277472


namespace NUMINAMATH_CALUDE_picture_distance_l2774_277445

theorem picture_distance (wall_width picture_width : ℝ) 
  (hw : wall_width = 25)
  (hp : picture_width = 3) :
  (wall_width - picture_width) / 2 = 11 := by
sorry

end NUMINAMATH_CALUDE_picture_distance_l2774_277445


namespace NUMINAMATH_CALUDE_division_equality_may_not_hold_l2774_277449

theorem division_equality_may_not_hold (a b c : ℝ) : 
  a = b → ¬(∀ c, a / c = b / c) :=
by
  sorry

end NUMINAMATH_CALUDE_division_equality_may_not_hold_l2774_277449


namespace NUMINAMATH_CALUDE_clique_of_nine_l2774_277468

/-- Represents the relationship of knowing each other in a group of people -/
def Knows (n : ℕ) := Fin n → Fin n → Prop

/-- States that the 'Knows' relation is symmetric -/
def SymmetricKnows {n : ℕ} (knows : Knows n) :=
  ∀ i j : Fin n, knows i j → knows j i

/-- States that among any 3 people, at least two know each other -/
def AtLeastTwoKnowEachOther {n : ℕ} (knows : Knows n) :=
  ∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    knows i j ∨ knows j k ∨ knows i k

/-- Defines a clique of size 4 where everyone knows each other -/
def HasCliqueFour {n : ℕ} (knows : Knows n) :=
  ∃ i j k l : Fin n, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l ∧
    knows i j ∧ knows i k ∧ knows i l ∧
    knows j k ∧ knows j l ∧
    knows k l

theorem clique_of_nine (knows : Knows 9) 
  (symm : SymmetricKnows knows) 
  (atleast_two : AtLeastTwoKnowEachOther knows) : 
  HasCliqueFour knows := by
  sorry

end NUMINAMATH_CALUDE_clique_of_nine_l2774_277468


namespace NUMINAMATH_CALUDE_f_roots_l2774_277458

-- Define the function f
def f (x : ℝ) : ℝ := 
  let matrix := !![1, 1, 1; x, -1, 1; x^2, 2, 1]
  Matrix.det matrix

-- State the theorem
theorem f_roots : 
  {x : ℝ | f x = 0} = {-3/2, 1} := by sorry

end NUMINAMATH_CALUDE_f_roots_l2774_277458


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a11_l2774_277429

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a11 (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 = 4 → a 5 = 8 → a 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a11_l2774_277429


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2774_277420

theorem complex_magnitude_problem (w z : ℂ) :
  w * z = 24 - 16 * Complex.I ∧ Complex.abs w = Real.sqrt 52 →
  Complex.abs z = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2774_277420


namespace NUMINAMATH_CALUDE_sperner_theorem_l2774_277401

/-- The largest number of subsets of an n-element set such that no subset is contained in any other -/
def largestSperner (n : ℕ) : ℕ :=
  Nat.choose n (n / 2)

/-- Sperner's theorem -/
theorem sperner_theorem (n : ℕ) :
  largestSperner n = Nat.choose n (n / 2) :=
sorry

end NUMINAMATH_CALUDE_sperner_theorem_l2774_277401


namespace NUMINAMATH_CALUDE_quadratic_root_value_l2774_277411

theorem quadratic_root_value (p q : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (3 : ℂ) * (4 + Complex.I) ^ 2 + p * (4 + Complex.I) + q = 0 →
  q = -51 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l2774_277411


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2774_277455

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2774_277455


namespace NUMINAMATH_CALUDE_hikers_distribution_theorem_l2774_277498

/-- The number of ways to distribute 5 people into three rooms -/
def distribution_ways : ℕ := 20

/-- The number of hikers -/
def num_hikers : ℕ := 5

/-- The number of available rooms -/
def num_rooms : ℕ := 3

/-- The capacity of the largest room -/
def large_room_capacity : ℕ := 3

/-- The capacity of each of the smaller rooms -/
def small_room_capacity : ℕ := 2

/-- Theorem stating that the number of ways to distribute the hikers is correct -/
theorem hikers_distribution_theorem :
  distribution_ways = (num_hikers.choose large_room_capacity) * 2 :=
by sorry

end NUMINAMATH_CALUDE_hikers_distribution_theorem_l2774_277498


namespace NUMINAMATH_CALUDE_card_arrangement_count_l2774_277434

/-- The number of ways to arrange 6 cards into 3 envelopes -/
def arrangement_count : ℕ := 18

/-- The number of envelopes -/
def num_envelopes : ℕ := 3

/-- The number of cards -/
def num_cards : ℕ := 6

/-- The number of cards per envelope -/
def cards_per_envelope : ℕ := 2

/-- Cards 1 and 2 are in the same envelope -/
def cards_1_2_together : Prop := True

theorem card_arrangement_count :
  arrangement_count = num_envelopes * (num_cards - cards_per_envelope).choose cards_per_envelope :=
sorry

end NUMINAMATH_CALUDE_card_arrangement_count_l2774_277434


namespace NUMINAMATH_CALUDE_square_of_97_l2774_277493

theorem square_of_97 : 97 * 97 = 9409 := by
  sorry

end NUMINAMATH_CALUDE_square_of_97_l2774_277493


namespace NUMINAMATH_CALUDE_a_101_value_l2774_277427

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, a (n + 1) - a n = 1 / 2

theorem a_101_value (a : ℕ → ℚ) (h : arithmetic_sequence a) : a 101 = 52 := by
  sorry

end NUMINAMATH_CALUDE_a_101_value_l2774_277427


namespace NUMINAMATH_CALUDE_toph_fish_count_l2774_277474

theorem toph_fish_count (total_people : ℕ) (average_fish : ℕ) (aang_fish : ℕ) (sokka_fish : ℕ) :
  total_people = 3 →
  average_fish = 8 →
  aang_fish = 7 →
  sokka_fish = 5 →
  average_fish * total_people - aang_fish - sokka_fish = 12 :=
by sorry

end NUMINAMATH_CALUDE_toph_fish_count_l2774_277474


namespace NUMINAMATH_CALUDE_rich_walk_distance_l2774_277462

/-- Represents Rich's walking route --/
def RichWalk : ℕ → ℕ
| 0 => 20  -- House to sidewalk
| 1 => 200  -- Down the sidewalk
| 2 => 2 * (20 + 200)  -- Left turn, double distance
| 3 => 500  -- Right turn to park
| 4 => 300  -- Inside park
| 5 => 3 * (20 + 200 + 2 * (20 + 200) + 500 + 300)  -- Triple total distance after park
| 6 => (20 + 200 + 2 * (20 + 200) + 500 + 300 + 3 * (20 + 200 + 2 * (20 + 200) + 500 + 300)) / 2  -- Half total distance for last leg
| _ => 0

/-- Calculates the total distance of Rich's walk --/
def TotalDistance : ℕ := 2 * (RichWalk 0 + RichWalk 1 + RichWalk 2 + RichWalk 3 + RichWalk 4 + RichWalk 5 + RichWalk 6)

/-- Theorem stating that Rich's total walking distance is 17520 feet --/
theorem rich_walk_distance : TotalDistance = 17520 := by
  sorry

end NUMINAMATH_CALUDE_rich_walk_distance_l2774_277462


namespace NUMINAMATH_CALUDE_prob_sum_7_twice_l2774_277425

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The set of possible outcomes for a single die roll -/
def outcomes : Set ℕ := {1, 2, 3, 4, 5, 6}

/-- The probability of rolling a sum of 7 with two dice -/
def prob_sum_7 : ℚ := 6 / 36

/-- The probability of rolling a sum of 7 twice in a row with two dice -/
theorem prob_sum_7_twice (h : sides = 6) : prob_sum_7 * prob_sum_7 = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_7_twice_l2774_277425


namespace NUMINAMATH_CALUDE_suna_travel_distance_l2774_277488

theorem suna_travel_distance (D : ℝ) 
  (h1 : (1 - 7/15) * (1 - 5/8) * (1 - 2/3) * D = 2.6) : D = 39 := by
  sorry

end NUMINAMATH_CALUDE_suna_travel_distance_l2774_277488


namespace NUMINAMATH_CALUDE_irrational_among_options_l2774_277439

theorem irrational_among_options : 
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 5 = (a : ℚ) / b) ∧ 
  (∃ (a b : ℤ), b ≠ 0 ∧ (3.14 : ℝ) = (a : ℚ) / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (22 : ℚ) / 7 = (a : ℚ) / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 4 = (a : ℚ) / b) :=
by
  sorry

end NUMINAMATH_CALUDE_irrational_among_options_l2774_277439


namespace NUMINAMATH_CALUDE_fraction_simplification_l2774_277477

theorem fraction_simplification (x : ℝ) (h : 2*x - 2 ≠ 0) :
  (6*x^3 + 13*x^2 + 15*x - 25) / (2*x^3 + 4*x^2 + 4*x - 10) = (6*x - 5) / (2*x - 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2774_277477


namespace NUMINAMATH_CALUDE_arctg_arcctg_comparison_l2774_277464

theorem arctg_arcctg_comparison : (5 * Real.sqrt 7) / 4 > Real.arctan (2 + Real.sqrt 5) + Real.arctan (1 / (2 - Real.sqrt 5)) := by
  sorry

end NUMINAMATH_CALUDE_arctg_arcctg_comparison_l2774_277464


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2774_277473

/-- 
For a quadratic equation (a-1)x^2 - 2x + 1 = 0 to have real roots, 
a must satisfy: a ≤ 2 and a ≠ 1 
-/
theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ (a ≤ 2 ∧ a ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2774_277473


namespace NUMINAMATH_CALUDE_solution_set_f_gt_7_min_m2_plus_n2_l2774_277475

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

-- Theorem for the solution set of f(x) > 7
theorem solution_set_f_gt_7 :
  {x : ℝ | f x > 7} = {x : ℝ | x > 4 ∨ x < -3} := by sorry

-- Theorem for the minimum value of m^2 + n^2 and the values of m and n
theorem min_m2_plus_n2 (m n : ℝ) (hn : n > 0) (h_min : ∀ x, f x ≥ m + n) :
  m^2 + n^2 ≥ 9/2 ∧ (m^2 + n^2 = 9/2 ↔ m = 3/2 ∧ n = 3/2) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_7_min_m2_plus_n2_l2774_277475


namespace NUMINAMATH_CALUDE_angle_A_is_135_l2774_277416

/-- A trapezoid with specific angle relationships -/
structure SpecialTrapezoid where
  /-- The measure of angle A in degrees -/
  A : ℝ
  /-- The measure of angle B in degrees -/
  B : ℝ
  /-- The measure of angle C in degrees -/
  C : ℝ
  /-- The measure of angle D in degrees -/
  D : ℝ
  /-- AB is parallel to CD -/
  parallel : A + D = 180
  /-- Angle A is three times angle D -/
  A_eq_3D : A = 3 * D
  /-- Angle C is four times angle B -/
  C_eq_4B : C = 4 * B

/-- The measure of angle A in a special trapezoid is 135 degrees -/
theorem angle_A_is_135 (t : SpecialTrapezoid) : t.A = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_135_l2774_277416


namespace NUMINAMATH_CALUDE_stockholm_malmo_distance_l2774_277461

/-- The scale factor of the map, representing kilometers per centimeter. -/
def scale : ℝ := 10

/-- The distance between Stockholm and Malmo on the map, in centimeters. -/
def map_distance : ℝ := 112

/-- The actual distance between Stockholm and Malmo, in kilometers. -/
def actual_distance : ℝ := map_distance * scale

theorem stockholm_malmo_distance : actual_distance = 1120 := by
  sorry

end NUMINAMATH_CALUDE_stockholm_malmo_distance_l2774_277461


namespace NUMINAMATH_CALUDE_soccer_team_statistics_l2774_277421

theorem soccer_team_statistics (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) :
  total_players = 25 →
  physics_players = 10 →
  both_subjects = 6 →
  ∃ (statistics_players : ℕ),
    statistics_players = 23 ∧
    statistics_players + physics_players - both_subjects = total_players :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_statistics_l2774_277421


namespace NUMINAMATH_CALUDE_similar_triangles_height_cycle_height_problem_l2774_277489

theorem similar_triangles_height (h₁ : ℝ) (b₁ : ℝ) (b₂ : ℝ) (h₁_pos : h₁ > 0) (b₁_pos : b₁ > 0) (b₂_pos : b₂ > 0) :
  h₁ / b₁ = (h₁ * b₂ / b₁) / b₂ :=
by sorry

theorem cycle_height_problem (h₁ : ℝ) (b₁ : ℝ) (b₂ : ℝ) 
  (h₁_val : h₁ = 2.5) (b₁_val : b₁ = 5) (b₂_val : b₂ = 4) :
  h₁ * b₂ / b₁ = 2 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_height_cycle_height_problem_l2774_277489


namespace NUMINAMATH_CALUDE_office_absenteeism_l2774_277494

theorem office_absenteeism (p : ℕ) (x : ℚ) (h : 0 < p) :
  (1 / ((1 - x) * p) - 1 / p = 1 / (3 * p)) → x = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_office_absenteeism_l2774_277494


namespace NUMINAMATH_CALUDE_sin_1320_degrees_l2774_277400

theorem sin_1320_degrees : Real.sin (1320 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_1320_degrees_l2774_277400


namespace NUMINAMATH_CALUDE_largest_expression_l2774_277491

theorem largest_expression : 
  let expr1 := 2 + (-2)
  let expr2 := 2 - (-2)
  let expr3 := 2 * (-2)
  let expr4 := 2 / (-2)
  expr2 = max expr1 (max expr2 (max expr3 expr4)) := by sorry

end NUMINAMATH_CALUDE_largest_expression_l2774_277491


namespace NUMINAMATH_CALUDE_range_of_a_l2774_277413

/-- Proposition p: The function y=(a-1)x is increasing -/
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (a - 1) * x < (a - 1) * y

/-- Proposition q: The inequality -x^2+2x-2≤a holds true for all real numbers x -/
def q (a : ℝ) : Prop := ∀ x : ℝ, -x^2 + 2*x - 2 ≤ a

/-- The main theorem stating the range of a -/
theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : 
  a ∈ Set.Icc (-1 : ℝ) 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2774_277413


namespace NUMINAMATH_CALUDE_olivia_soda_purchase_l2774_277499

/-- The number of quarters Olivia spent on a soda -/
def quarters_spent (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that Olivia spent 4 quarters on the soda -/
theorem olivia_soda_purchase : quarters_spent 11 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_olivia_soda_purchase_l2774_277499


namespace NUMINAMATH_CALUDE_rational_quadratic_integer_solutions_l2774_277403

theorem rational_quadratic_integer_solutions (r : ℚ) :
  (∃ x : ℤ, r * x^2 + (r + 1) * x + r = 1) ↔ (r = 1 ∨ r = -1/7) := by
  sorry

end NUMINAMATH_CALUDE_rational_quadratic_integer_solutions_l2774_277403


namespace NUMINAMATH_CALUDE_kaeli_problems_per_day_l2774_277405

def marie_pascale_problems_per_day : ℕ := 4
def marie_pascale_total_problems : ℕ := 72
def kaeli_extra_problems : ℕ := 54

def days : ℕ := marie_pascale_total_problems / marie_pascale_problems_per_day

def kaeli_total_problems : ℕ := marie_pascale_total_problems + kaeli_extra_problems

theorem kaeli_problems_per_day : 
  kaeli_total_problems / days = 7 :=
sorry

end NUMINAMATH_CALUDE_kaeli_problems_per_day_l2774_277405


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2774_277486

theorem regular_polygon_sides (D : ℕ) (n : ℕ) : D = 15 → n * (n - 3) / 2 = D → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2774_277486


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_one_range_of_a_given_f_geq_three_halves_l2774_277433

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |a * x + 1| + |x - a|
def g (x : ℝ) : ℝ := x^2 + x

-- Theorem for part (1)
theorem solution_set_when_a_eq_one :
  ∀ x : ℝ, (g x ≥ f 1 x) ↔ (x ≥ 1 ∨ x ≤ -3) :=
sorry

-- Theorem for part (2)
theorem range_of_a_given_f_geq_three_halves :
  (∀ x : ℝ, ∃ a : ℝ, a > 0 ∧ f a x ≥ 3/2) →
  (∀ a : ℝ, a > 0 → f a x ≥ 3/2 → a ≥ Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_one_range_of_a_given_f_geq_three_halves_l2774_277433


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l2774_277466

/-- The equation of the tangent line to y = 2ln(x+1) at (0,0) is y = 2x -/
theorem tangent_line_at_origin (x : ℝ) :
  let f : ℝ → ℝ := λ x => 2 * Real.log (x + 1)
  let f' : ℝ → ℝ := λ x => 2 / (x + 1)
  let tangent_line : ℝ → ℝ := λ x => 2 * x
  (∀ x, HasDerivAt f (f' x) x) →
  HasDerivAt f 2 0 →
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - tangent_line x| ≤ ε * |x|
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l2774_277466


namespace NUMINAMATH_CALUDE_cube_root_of_sqrt_64_l2774_277412

theorem cube_root_of_sqrt_64 : ∃ (x : ℝ), x^3 = Real.sqrt 64 ∧ (x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_sqrt_64_l2774_277412


namespace NUMINAMATH_CALUDE_water_volume_ratio_l2774_277444

/-- Represents a location with rainfall and area data -/
structure Location where
  rainfall : ℝ  -- rainfall in cm
  area : ℝ      -- area in hectares

/-- Calculates the volume of water collected at a location -/
def waterVolume (loc : Location) : ℝ :=
  loc.rainfall * loc.area * 10

/-- Theorem stating the ratio of water volumes collected at locations A, B, and C -/
theorem water_volume_ratio 
  (locA locB locC : Location)
  (hA : locA = { rainfall := 7, area := 2 })
  (hB : locB = { rainfall := 11, area := 3.5 })
  (hC : locC = { rainfall := 15, area := 5 }) :
  ∃ (k : ℝ), k > 0 ∧ 
    waterVolume locA = 140 * k ∧ 
    waterVolume locB = 385 * k ∧ 
    waterVolume locC = 750 * k :=
sorry

end NUMINAMATH_CALUDE_water_volume_ratio_l2774_277444


namespace NUMINAMATH_CALUDE_approximation_problem_l2774_277431

def is_close (x y : ℝ) (ε : ℝ) : Prop := |x - y| ≤ ε

theorem approximation_problem :
  (∀ n : ℕ, 5 ≤ n ∧ n ≤ 9 → is_close (5 * n * 18) 1200 90) ∧
  (∀ m : ℕ, 0 ≤ m ∧ m ≤ 2 → is_close ((3 * 10 + m) * 9 / 5) 60 5) :=
sorry

end NUMINAMATH_CALUDE_approximation_problem_l2774_277431


namespace NUMINAMATH_CALUDE_star_op_identity_l2774_277457

/-- Define the * operation on ordered pairs of real numbers -/
def star_op (a b c d : ℝ) : ℝ × ℝ := (a * c + b * d, a * d + b * c)

/-- Theorem: If (a, b) * (x, y) = (a, b) and a² ≠ b², then (x, y) = (1, 0) -/
theorem star_op_identity {a b x y : ℝ} (h : a^2 ≠ b^2) :
  star_op a b x y = (a, b) → (x, y) = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_star_op_identity_l2774_277457


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_sum_l2774_277479

theorem quadratic_root_ratio_sum (x₁ x₂ : ℝ) : 
  x₁^2 + 2*x₁ - 8 = 0 →
  x₂^2 + 2*x₂ - 8 = 0 →
  x₁ ≠ 0 →
  x₂ ≠ 0 →
  x₂/x₁ + x₁/x₂ = -5/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_sum_l2774_277479


namespace NUMINAMATH_CALUDE_textbooks_on_sale_textbooks_on_sale_is_five_l2774_277408

/-- Proves the number of textbooks bought on sale given the conditions of the problem -/
theorem textbooks_on_sale (sale_price : ℕ) (online_total : ℕ) (bookstore_multiplier : ℕ) (total_spent : ℕ) : ℕ :=
  let sale_count := (total_spent - online_total - (bookstore_multiplier * online_total)) / sale_price
  sale_count

#check textbooks_on_sale 10 40 3 210 = 5

/-- The main theorem that proves the number of textbooks bought on sale is 5 -/
theorem textbooks_on_sale_is_five : textbooks_on_sale 10 40 3 210 = 5 := by
  sorry

end NUMINAMATH_CALUDE_textbooks_on_sale_textbooks_on_sale_is_five_l2774_277408


namespace NUMINAMATH_CALUDE_circle_equation_radius_l2774_277456

theorem circle_equation_radius (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 10*y + c = 0 ↔ (x + 4)^2 + (y + 5)^2 = 25) → 
  c = -16 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_radius_l2774_277456


namespace NUMINAMATH_CALUDE_unique_satisfying_number_l2774_277465

/-- Reverses the digits of a 4-digit number -/
def reverseDigits (n : Nat) : Nat :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  d * 1000 + c * 100 + b * 10 + a

/-- Checks if a number satisfies the given condition -/
def satisfiesCondition (n : Nat) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ n + 8802 > reverseDigits n

theorem unique_satisfying_number : 
  ∀ n : Nat, satisfiesCondition n ↔ n = 1099 :=
by sorry

end NUMINAMATH_CALUDE_unique_satisfying_number_l2774_277465


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l2774_277460

theorem last_two_digits_sum (n : ℕ) : (8^25 + 12^25) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l2774_277460


namespace NUMINAMATH_CALUDE_repeating_decimal_difference_l2774_277437

theorem repeating_decimal_difference : 
  let repeating_decimal := (4 : ℚ) / 11
  let non_repeating_decimal := (36 : ℚ) / 100
  repeating_decimal - non_repeating_decimal = (4 : ℚ) / 1100 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_difference_l2774_277437


namespace NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l2774_277463

theorem tan_ratio_from_sin_sum_diff (α β : Real) 
  (h1 : Real.sin (α + β) = 2/3)
  (h2 : Real.sin (α - β) = 1/3) :
  Real.tan α / Real.tan β = 3 := by
sorry

end NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l2774_277463


namespace NUMINAMATH_CALUDE_glasses_in_larger_box_l2774_277432

theorem glasses_in_larger_box 
  (small_box : ℕ) 
  (total_boxes : ℕ) 
  (average_glasses : ℕ) :
  small_box = 12 → 
  total_boxes = 2 → 
  average_glasses = 15 → 
  ∃ large_box : ℕ, 
    (small_box + large_box) / total_boxes = average_glasses ∧ 
    large_box = 18 := by
sorry

end NUMINAMATH_CALUDE_glasses_in_larger_box_l2774_277432


namespace NUMINAMATH_CALUDE_divisibility_condition_l2774_277402

theorem divisibility_condition (m : ℕ+) :
  (∀ k : ℕ, k ≥ 3 → Odd k → (k^(m : ℕ) - 1) % 2^(m : ℕ) = 0) ↔ m = 1 ∨ m = 2 ∨ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2774_277402


namespace NUMINAMATH_CALUDE_smallest_angle_of_quadrilateral_l2774_277443

theorem smallest_angle_of_quadrilateral (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- All angles are positive
  a + b + c + d = 360 →           -- Sum of angles in a quadrilateral
  b = 4/3 * a →                   -- Ratio condition
  c = 5/3 * a →                   -- Ratio condition
  d = 2 * a →                     -- Ratio condition
  a = 60 ∧ a ≤ b ∧ a ≤ c ∧ a ≤ d  -- a is the smallest angle and equals 60°
  := by sorry

end NUMINAMATH_CALUDE_smallest_angle_of_quadrilateral_l2774_277443


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l2774_277426

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 - 3*x + 5

-- State the theorem
theorem root_exists_in_interval :
  ∃ x₀ ∈ Set.Ioo 1 2, f x₀ = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l2774_277426


namespace NUMINAMATH_CALUDE_pride_and_prejudice_watch_time_l2774_277418

/-- The number of hours spent watching a TV series -/
def watch_time (num_episodes : ℕ) (episode_length : ℕ) : ℚ :=
  (num_episodes * episode_length : ℚ) / 60

/-- Theorem: Watching 6 episodes of 50 minutes each takes 5 hours -/
theorem pride_and_prejudice_watch_time :
  watch_time 6 50 = 5 := by sorry

end NUMINAMATH_CALUDE_pride_and_prejudice_watch_time_l2774_277418


namespace NUMINAMATH_CALUDE_power_division_equality_l2774_277484

theorem power_division_equality (a : ℝ) : a^6 / (-a)^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l2774_277484


namespace NUMINAMATH_CALUDE_f_six_eq_zero_l2774_277436

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_six_eq_zero (f : ℝ → ℝ) (h_odd : is_odd f) (h_period : ∀ x, f (x + 2) = -f x) : f 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_six_eq_zero_l2774_277436


namespace NUMINAMATH_CALUDE_tate_education_years_l2774_277476

/-- The total years Tate spent in high school and college -/
def totalEducationYears (normalHighSchoolYears : ℕ) : ℕ :=
  let tateHighSchoolYears := normalHighSchoolYears - 1
  let tertiaryEducationYears := 3 * tateHighSchoolYears
  tateHighSchoolYears + tertiaryEducationYears

/-- Theorem stating that Tate's total education years is 12 -/
theorem tate_education_years :
  totalEducationYears 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_tate_education_years_l2774_277476


namespace NUMINAMATH_CALUDE_simplify_expression_l2774_277483

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^(2/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / ((1/3) * a^(1/6) * b^(5/6)) = -9 * a :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2774_277483


namespace NUMINAMATH_CALUDE_third_number_in_nth_row_l2774_277407

/-- Represents the function that gives the third number from the left in the nth row
    of a triangular array of positive odd numbers. -/
def thirdNumber (n : ℕ) : ℕ := n^2 - n + 5

/-- Theorem stating that for n ≥ 3, the third number from the left in the nth row
    of a triangular array of positive odd numbers is n^2 - n + 5. -/
theorem third_number_in_nth_row (n : ℕ) (h : n ≥ 3) :
  thirdNumber n = n^2 - n + 5 := by
  sorry

end NUMINAMATH_CALUDE_third_number_in_nth_row_l2774_277407


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l2774_277448

theorem fraction_product_simplification :
  (6 : ℚ) / 3 * (9 : ℚ) / 6 * (12 : ℚ) / 9 * (15 : ℚ) / 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l2774_277448


namespace NUMINAMATH_CALUDE_binomial_coefficient_1000_1000_l2774_277487

theorem binomial_coefficient_1000_1000 : Nat.choose 1000 1000 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1000_1000_l2774_277487


namespace NUMINAMATH_CALUDE_custom_op_example_l2774_277459

/-- Custom binary operation ※ -/
def custom_op (a b : ℕ) : ℕ := a + 5 + b * 15

/-- Theorem stating that 105 ※ 5 = 185 -/
theorem custom_op_example : custom_op 105 5 = 185 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l2774_277459


namespace NUMINAMATH_CALUDE_bottle_caps_remaining_l2774_277481

theorem bottle_caps_remaining (initial_caps : ℕ) (removed_caps : ℕ) :
  initial_caps = 16 → removed_caps = 6 → initial_caps - removed_caps = 10 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_remaining_l2774_277481


namespace NUMINAMATH_CALUDE_cost_graph_two_segments_l2774_277469

/-- The cost function for pencils -/
def cost (n : ℕ) : ℚ :=
  if n ≤ 10 then 10 * n else 8 * n - 40

/-- The graph of the cost function consists of two connected linear segments -/
theorem cost_graph_two_segments :
  ∃ (a b : ℕ) (m₁ m₂ c₁ c₂ : ℚ),
    a < b ∧
    (∀ n, 1 ≤ n ∧ n ≤ a → cost n = m₁ * n + c₁) ∧
    (∀ n, b ≤ n ∧ n ≤ 20 → cost n = m₂ * n + c₂) ∧
    (m₁ * a + c₁ = m₂ * b + c₂) ∧
    m₁ ≠ m₂ :=
sorry

end NUMINAMATH_CALUDE_cost_graph_two_segments_l2774_277469


namespace NUMINAMATH_CALUDE_three_digit_number_times_seven_l2774_277495

theorem three_digit_number_times_seven (n : ℕ) : 
  (100 ≤ n ∧ n < 1000) ∧ (∃ k : ℕ, 7 * n = 1000 * k + 638) ↔ n = 234 :=
sorry

end NUMINAMATH_CALUDE_three_digit_number_times_seven_l2774_277495


namespace NUMINAMATH_CALUDE_equation_solution_l2774_277430

theorem equation_solution (y : ℝ) : ∃ x : ℝ, 19 * (x + y) + 17 = 19 * (-x + y) - 21 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2774_277430


namespace NUMINAMATH_CALUDE_shopping_expense_percentage_l2774_277440

theorem shopping_expense_percentage (T : ℝ) (O : ℝ) : 
  T > 0 →
  0.50 * T + 0.20 * T + O * T / 100 = T →
  0.04 * (0.50 * T) + 0 * (0.20 * T) + 0.08 * (O * T / 100) = 0.044 * T →
  O = 30 := by
sorry

end NUMINAMATH_CALUDE_shopping_expense_percentage_l2774_277440


namespace NUMINAMATH_CALUDE_call_service_comparison_l2774_277417

/-- Represents the cost of a phone call service -/
structure CallService where
  monthly_fee : ℝ
  per_minute_rate : ℝ

/-- Calculates the total cost for a given call duration -/
def total_cost (service : CallService) (duration : ℝ) : ℝ :=
  service.monthly_fee + service.per_minute_rate * duration

/-- Global Call service -/
def global_call : CallService :=
  { monthly_fee := 50, per_minute_rate := 0.4 }

/-- China Mobile service -/
def china_mobile : CallService :=
  { monthly_fee := 0, per_minute_rate := 0.6 }

theorem call_service_comparison :
  ∃ (x : ℝ),
    (∀ (duration : ℝ), total_cost global_call duration = 50 + 0.4 * duration) ∧
    (∀ (duration : ℝ), total_cost china_mobile duration = 0.6 * duration) ∧
    (total_cost global_call x = total_cost china_mobile x ∧ x = 125) ∧
    (∀ (duration : ℝ), duration > 125 → total_cost global_call duration < total_cost china_mobile duration) :=
by sorry

end NUMINAMATH_CALUDE_call_service_comparison_l2774_277417


namespace NUMINAMATH_CALUDE_intersection_A_B_l2774_277451

def A : Set ℝ := {x | ∃ (α β : ℤ), α ≥ 0 ∧ β ≥ 0 ∧ x = 2^α * 3^β}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}

theorem intersection_A_B : A ∩ B = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2774_277451


namespace NUMINAMATH_CALUDE_digit_removal_theorem_l2774_277492

/-- A function that removes digits from a natural number's decimal representation -/
def digit_removal (n : ℕ) : ℕ := sorry

/-- The number formed by 2005 9's -/
def N_2005 : ℕ := 10^2005 - 1

/-- The number formed by 2008 9's -/
def N_2008 : ℕ := 10^2008 - 1

/-- Theorem stating that N_2005^2009 can be obtained by removing digits from N_2008^2009 -/
theorem digit_removal_theorem :
  ∃ (f : ℕ → ℕ), f (N_2008^2009) = N_2005^2009 ∧ 
  (∀ n : ℕ, f n ≤ n) :=
sorry

end NUMINAMATH_CALUDE_digit_removal_theorem_l2774_277492


namespace NUMINAMATH_CALUDE_factoring_expression_l2774_277470

theorem factoring_expression (y : ℝ) : 3*y*(2*y+5) + 4*(2*y+5) = (3*y+4)*(2*y+5) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l2774_277470


namespace NUMINAMATH_CALUDE_bake_sale_ratio_l2774_277453

/-- Given a bake sale where 104 items were sold in total, with 48 cookies sold,
    prove that the ratio of brownies to cookies sold is 7:6. -/
theorem bake_sale_ratio : 
  let total_items : ℕ := 104
  let cookies_sold : ℕ := 48
  let brownies_sold : ℕ := total_items - cookies_sold
  (brownies_sold : ℚ) / (cookies_sold : ℚ) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_ratio_l2774_277453


namespace NUMINAMATH_CALUDE_air_conditioning_price_calculation_air_conditioning_price_proof_l2774_277485

/-- Calculates the final price of an air-conditioning unit after a discount and subsequent increase -/
theorem air_conditioning_price_calculation (initial_price : ℚ) 
  (discount_rate : ℚ) (increase_rate : ℚ) : ℚ :=
  let discounted_price := initial_price * (1 - discount_rate)
  let final_price := discounted_price * (1 + increase_rate)
  final_price

/-- Proves that the final price of the air-conditioning unit is approximately $442.18 -/
theorem air_conditioning_price_proof : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.005 ∧ 
  |air_conditioning_price_calculation 470 (16/100) (12/100) - 442.18| < ε :=
sorry

end NUMINAMATH_CALUDE_air_conditioning_price_calculation_air_conditioning_price_proof_l2774_277485


namespace NUMINAMATH_CALUDE_crayons_per_row_calculation_l2774_277478

/-- Given a total number of crayons and rows, calculate the number of crayons per row. -/
def crayonsPerRow (totalCrayons : ℕ) (numRows : ℕ) : ℕ :=
  totalCrayons / numRows

/-- Prove that given 210 crayons and 7 rows, there are 30 crayons in each row. -/
theorem crayons_per_row_calculation :
  crayonsPerRow 210 7 = 30 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_row_calculation_l2774_277478


namespace NUMINAMATH_CALUDE_equation_solution_l2774_277406

theorem equation_solution : ∃ x : ℝ, (x - 5)^3 = (1/27)⁻¹ ∧ x = 8 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2774_277406


namespace NUMINAMATH_CALUDE_lateral_face_base_angle_l2774_277447

/-- A regular quadrilateral pyramid with specific properties -/
structure RegularQuadPyramid where
  /-- The angle between a lateral edge and the base plane -/
  edge_base_angle : ℝ
  /-- The angle at the apex of the pyramid -/
  apex_angle : ℝ
  /-- The condition that edge_base_angle equals apex_angle -/
  edge_base_eq_apex : edge_base_angle = apex_angle

/-- The theorem stating the angle between the lateral face and the base plane -/
theorem lateral_face_base_angle (p : RegularQuadPyramid) :
  Real.arctan (Real.sqrt (1 + Real.sqrt 5)) =
    Real.arctan (Real.sqrt (1 + Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_lateral_face_base_angle_l2774_277447


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2774_277471

def z : ℂ := Complex.I^3 * (1 + Complex.I) * Complex.I

theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2774_277471


namespace NUMINAMATH_CALUDE_june_score_june_score_correct_l2774_277423

theorem june_score (april_may_avg : ℕ) (april_may_june_avg : ℕ) : ℕ :=
  let april_may_total := april_may_avg * 2
  let april_may_june_total := april_may_june_avg * 3
  april_may_june_total - april_may_total

theorem june_score_correct :
  june_score 89 88 = 86 := by sorry

end NUMINAMATH_CALUDE_june_score_june_score_correct_l2774_277423


namespace NUMINAMATH_CALUDE_mod_eight_difference_l2774_277414

theorem mod_eight_difference (n : ℕ) : (47^n - 23^n) % 8 = 0 :=
sorry

end NUMINAMATH_CALUDE_mod_eight_difference_l2774_277414


namespace NUMINAMATH_CALUDE_irrational_numbers_in_set_l2774_277435

-- Define the set of numbers
def numbers : Set ℝ := {1/3, Real.pi, 0, Real.sqrt 5}

-- Define the property of being irrational
def IsIrrational (x : ℝ) : Prop := ∀ (a b : ℤ), b ≠ 0 → x ≠ a / b

-- Statement to prove
theorem irrational_numbers_in_set :
  {x ∈ numbers | IsIrrational x} = {Real.pi, Real.sqrt 5} := by sorry

end NUMINAMATH_CALUDE_irrational_numbers_in_set_l2774_277435


namespace NUMINAMATH_CALUDE_max_angle_at_3_2_l2774_277497

/-- The line l: x + y - 5 = 0 -/
def line (x y : ℝ) : Prop := x + y - 5 = 0

/-- Point A -/
def A : ℝ × ℝ := (1, 0)

/-- Point B -/
def B : ℝ × ℝ := (3, 0)

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Statement: The point (3,2) maximizes the angle APB on the given line -/
theorem max_angle_at_3_2 :
  line 3 2 ∧
  ∀ x y, line x y → angle A (x, y) B ≤ angle A (3, 2) B :=
by sorry

end NUMINAMATH_CALUDE_max_angle_at_3_2_l2774_277497
