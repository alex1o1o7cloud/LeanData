import Mathlib

namespace NUMINAMATH_CALUDE_percentage_of_percentage_l2883_288321

theorem percentage_of_percentage (amount : ℝ) : (5 / 100) * ((25 / 100) * amount) = 20 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_percentage_of_percentage_l2883_288321


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2883_288338

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) ↔ a ∈ Set.Ioi 3 ∪ Set.Iio 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2883_288338


namespace NUMINAMATH_CALUDE_one_belt_one_road_part1_one_belt_one_road_part2_l2883_288306

/-- Definition of a parabola -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ
  h : ∀ x, eq x = a * x^2 + b * x + c

/-- Definition of a line -/
structure Line where
  m : ℝ
  b : ℝ
  eq : ℝ → ℝ
  h : ∀ x, eq x = m * x + b

/-- Definition of "one belt, one road" relationship -/
def one_belt_one_road (L : Parabola) (l : Line) : Prop :=
  ∃ (P : ℝ × ℝ) (Q : ℝ × ℝ),
    P.1 = 0 ∧ 
    L.eq P.1 = P.2 ∧ 
    l.eq P.1 = P.2 ∧
    L.eq Q.1 = Q.2 ∧ 
    l.eq Q.1 = Q.2 ∧
    Q.1 = -L.b / (2 * L.a) ∧ 
    Q.2 = L.eq Q.1

/-- Part 1 of the theorem -/
theorem one_belt_one_road_part1 (L : Parabola) (l : Line) :
  L.a = 1 ∧ L.b = -2 ∧ l.b = 1 ∧ one_belt_one_road L l →
  l.m = -1 ∧ L.c = 1 :=
sorry

/-- Part 2 of the theorem -/
theorem one_belt_one_road_part2 (L : Parabola) (l : Line) :
  (∃ x, L.eq x = 6 / x) ∧ l.m = 2 ∧ l.b = -4 ∧ one_belt_one_road L l →
  (L.a = 2 ∧ L.b = 4 ∧ L.c = -4) ∨ (L.a = -2/3 ∧ L.b = 4 ∧ L.c = -10/3) :=
sorry

end NUMINAMATH_CALUDE_one_belt_one_road_part1_one_belt_one_road_part2_l2883_288306


namespace NUMINAMATH_CALUDE_line_equation_point_slope_l2883_288389

/-- A line passing through point (-1, 1) with slope 2 has the equation y = 2x + 3 -/
theorem line_equation_point_slope : 
  ∀ (x y : ℝ), y = 2*x + 3 ↔ (y - 1 = 2*(x - (-1)) ∧ (x, y) ≠ (-1, 1)) ∨ (x, y) = (-1, 1) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_point_slope_l2883_288389


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2883_288393

/-- A rectangle with given area and width has a specific perimeter -/
theorem rectangle_perimeter (area width : ℝ) (h_area : area = 200) (h_width : width = 10) :
  2 * (area / width + width) = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2883_288393


namespace NUMINAMATH_CALUDE_dog_turns_four_in_two_years_l2883_288355

/-- The number of years until a dog turns 4, given the owner's current age and age when the dog was born. -/
def years_until_dog_turns_four (owner_current_age : ℕ) (owner_age_when_dog_born : ℕ) : ℕ :=
  4 - (owner_current_age - owner_age_when_dog_born)

/-- Theorem: Given that the dog was born when the owner was 15 and the owner is now 17,
    the dog will turn 4 in 2 years. -/
theorem dog_turns_four_in_two_years :
  years_until_dog_turns_four 17 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dog_turns_four_in_two_years_l2883_288355


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l2883_288325

theorem quadratic_roots_difference_squared : 
  ∀ Φ φ : ℝ, 
  (Φ ^ 2 = Φ + 2) → 
  (φ ^ 2 = φ + 2) → 
  (Φ ≠ φ) → 
  (Φ - φ) ^ 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l2883_288325


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2883_288376

-- Problem 1
theorem problem_1 : Real.sqrt 9 + |-2| - (-3)^2 + (π - 100)^0 = -3 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) : 
  (x^2 + 1 = 5) ↔ (x = 2 ∨ x = -2) := by sorry

-- Problem 3
theorem problem_3 (x : ℝ) :
  (x^2 = (x - 2)^2 + 7) ↔ (x = 11/4) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2883_288376


namespace NUMINAMATH_CALUDE_cookies_distribution_l2883_288356

theorem cookies_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) : 
  total_cookies = 35 → 
  num_people = 5 → 
  total_cookies = num_people * cookies_per_person → 
  cookies_per_person = 7 := by
sorry

end NUMINAMATH_CALUDE_cookies_distribution_l2883_288356


namespace NUMINAMATH_CALUDE_six_pairs_l2883_288304

/-- The number of distinct pairs of integers (x, y) satisfying the conditions -/
def num_pairs : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    0 < p.1 ∧ p.1 < p.2 ∧ p.1 * p.2 = 2025
  ) (Finset.product (Finset.range 2026) (Finset.range 2026))).card

/-- Theorem stating that there are exactly 6 pairs satisfying the conditions -/
theorem six_pairs : num_pairs = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_pairs_l2883_288304


namespace NUMINAMATH_CALUDE_fraction_sum_integer_l2883_288361

theorem fraction_sum_integer (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_sum : ∃ n : ℤ, (a * b) / c + (b * c) / a + (c * a) / b = n) : 
  (∃ n1 : ℤ, (a * b) / c = n1) ∧ (∃ n2 : ℤ, (b * c) / a = n2) ∧ (∃ n3 : ℤ, (c * a) / b = n3) :=
sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_l2883_288361


namespace NUMINAMATH_CALUDE_odd_square_plus_even_product_is_odd_l2883_288324

theorem odd_square_plus_even_product_is_odd (k m : ℤ) : 
  let o : ℤ := 2 * k + 3
  let n : ℤ := 2 * m
  Odd (o^2 + n * o) := by
sorry

end NUMINAMATH_CALUDE_odd_square_plus_even_product_is_odd_l2883_288324


namespace NUMINAMATH_CALUDE_division_problem_l2883_288331

theorem division_problem (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 144 ∧ divisor = 11 ∧ remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 13 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2883_288331


namespace NUMINAMATH_CALUDE_family_strawberry_picking_l2883_288323

/-- The total weight of strawberries picked by a family -/
theorem family_strawberry_picking (marco_weight dad_weight mom_weight sister_weight : ℕ) 
  (h1 : marco_weight = 8)
  (h2 : dad_weight = 32)
  (h3 : mom_weight = 22)
  (h4 : sister_weight = 14) :
  marco_weight + dad_weight + mom_weight + sister_weight = 76 := by
  sorry

#check family_strawberry_picking

end NUMINAMATH_CALUDE_family_strawberry_picking_l2883_288323


namespace NUMINAMATH_CALUDE_atMostTwoInPlaceFive_l2883_288322

/-- The number of ways to arrange n people in n seats. -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of seating arrangements where at most two people
    are in their numbered seats, given n people and n seats. -/
def atMostTwoInPlace (n : ℕ) : ℕ :=
  totalArrangements n - choose n 3 * totalArrangements (n - 3) - 1

theorem atMostTwoInPlaceFive :
  atMostTwoInPlace 5 = 109 := by sorry

end NUMINAMATH_CALUDE_atMostTwoInPlaceFive_l2883_288322


namespace NUMINAMATH_CALUDE_shopping_remaining_amount_l2883_288307

def initial_amount : ℚ := 450

def grocery_fraction : ℚ := 3/5
def household_fraction : ℚ := 1/6
def personal_care_fraction : ℚ := 1/10

def remaining_amount : ℚ := initial_amount - (initial_amount * grocery_fraction + initial_amount * household_fraction + initial_amount * personal_care_fraction)

theorem shopping_remaining_amount : remaining_amount = 60 := by
  sorry

end NUMINAMATH_CALUDE_shopping_remaining_amount_l2883_288307


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2883_288305

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y = x * y) :
  x + y ≥ 9 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 4 * x₀ + y₀ = x₀ * y₀ ∧ x₀ + y₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2883_288305


namespace NUMINAMATH_CALUDE_count_adjacent_arrangements_l2883_288316

/-- The number of distinct arrangements of the letters in "КАРАКАТИЦА" where 'Р' and 'Ц' are adjacent -/
def adjacent_arrangements : ℕ := 15120

/-- The word from which we are forming arrangements -/
def word : String := "КАРАКАТИЦА"

/-- The length of the word -/
def word_length : ℕ := word.length

/-- The number of 'А's in the word -/
def count_A : ℕ := (word.toList.filter (· = 'А')).length

/-- The number of 'К's in the word -/
def count_K : ℕ := (word.toList.filter (· = 'К')).length

/-- Theorem stating that the number of distinct arrangements of the letters in "КАРАКАТИЦА" 
    where 'Р' and 'Ц' are adjacent is equal to adjacent_arrangements -/
theorem count_adjacent_arrangements :
  adjacent_arrangements = 
    2 * (Nat.factorial (word_length - 1)) / 
    (Nat.factorial count_A * Nat.factorial count_K) :=
by sorry

end NUMINAMATH_CALUDE_count_adjacent_arrangements_l2883_288316


namespace NUMINAMATH_CALUDE_sum_of_powers_divisibility_l2883_288380

theorem sum_of_powers_divisibility (n : ℕ+) :
  (((1:ℤ)^n.val + 2^n.val + 3^n.val + 4^n.val) % 5 = 0) ↔ (n.val % 4 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_divisibility_l2883_288380


namespace NUMINAMATH_CALUDE_can_measure_four_liters_l2883_288347

/-- Represents the state of water in the buckets -/
structure BucketState :=
  (small : ℕ)  -- Amount of water in the 3-liter bucket
  (large : ℕ)  -- Amount of water in the 5-liter bucket

/-- Represents the possible operations on the buckets -/
inductive BucketOperation
  | FillSmall
  | FillLarge
  | EmptySmall
  | EmptyLarge
  | PourSmallToLarge
  | PourLargeToSmall

/-- Applies a single operation to a bucket state -/
def applyOperation (state : BucketState) (op : BucketOperation) : BucketState :=
  match op with
  | BucketOperation.FillSmall => { small := 3, large := state.large }
  | BucketOperation.FillLarge => { small := state.small, large := 5 }
  | BucketOperation.EmptySmall => { small := 0, large := state.large }
  | BucketOperation.EmptyLarge => { small := state.small, large := 0 }
  | BucketOperation.PourSmallToLarge =>
      let amount := min state.small (5 - state.large)
      { small := state.small - amount, large := state.large + amount }
  | BucketOperation.PourLargeToSmall =>
      let amount := min state.large (3 - state.small)
      { small := state.small + amount, large := state.large - amount }

/-- Theorem: It is possible to measure exactly 4 liters using buckets of 3 and 5 liters -/
theorem can_measure_four_liters : ∃ (ops : List BucketOperation), 
  let final_state := ops.foldl applyOperation { small := 0, large := 0 }
  final_state.small + final_state.large = 4 := by
  sorry

end NUMINAMATH_CALUDE_can_measure_four_liters_l2883_288347


namespace NUMINAMATH_CALUDE_mikeys_jelly_beans_l2883_288383

theorem mikeys_jelly_beans (napoleon : ℕ) (sedrich : ℕ) (mikey : ℕ) : 
  napoleon = 17 →
  sedrich = napoleon + 4 →
  2 * (napoleon + sedrich) = 4 * mikey →
  mikey = 19 := by
sorry

end NUMINAMATH_CALUDE_mikeys_jelly_beans_l2883_288383


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2883_288379

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- a_n is a geometric sequence with common ratio q
  a 1 + a 2 = 3 →               -- a_1 + a_2 = 3
  a 3 + a 4 = 6 →               -- a_3 + a_4 = 6
  a 7 + a 8 = 24 :=             -- a_7 + a_8 = 24
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2883_288379


namespace NUMINAMATH_CALUDE_cube_and_fifth_power_existence_l2883_288363

theorem cube_and_fifth_power_existence (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ n : ℕ, n ≥ 1 ∧ ∃ k l : ℕ, a * n = k^3 ∧ b * n = l^5 :=
sorry

end NUMINAMATH_CALUDE_cube_and_fifth_power_existence_l2883_288363


namespace NUMINAMATH_CALUDE_city_of_pythagoras_schools_l2883_288396

/-- Represents a student in the math contest -/
structure Student where
  school : Nat
  rank : Nat

/-- The math contest setup -/
structure MathContest where
  numSchools : Nat
  students : Finset Student

theorem city_of_pythagoras_schools (contest : MathContest) : contest.numSchools = 40 :=
  by
  have h1 : ∀ s : Student, s ∈ contest.students → s.rank ≤ 4 * contest.numSchools :=
    sorry
  have h2 : ∀ s1 s2 : Student, s1 ∈ contest.students → s2 ∈ contest.students → s1 ≠ s2 → s1.rank ≠ s2.rank :=
    sorry
  have h3 : ∃ andrea : Student, andrea ∈ contest.students ∧
    andrea.rank = (2 * contest.numSchools) ∨ andrea.rank = (2 * contest.numSchools + 1) :=
    sorry
  have h4 : ∃ beth : Student, beth ∈ contest.students ∧ beth.rank = 41 :=
    sorry
  have h5 : ∃ carla : Student, carla ∈ contest.students ∧ carla.rank = 82 :=
    sorry
  have h6 : ∃ andrea beth carla : Student, 
    andrea ∈ contest.students ∧ beth ∈ contest.students ∧ carla ∈ contest.students ∧
    andrea.school = beth.school ∧ andrea.school = carla.school ∧
    andrea.rank < beth.rank ∧ andrea.rank < carla.rank :=
    sorry
  sorry


end NUMINAMATH_CALUDE_city_of_pythagoras_schools_l2883_288396


namespace NUMINAMATH_CALUDE_ternary_121_equals_16_l2883_288317

/-- Converts a ternary (base-3) number to decimal (base-10) --/
def ternary_to_decimal (a b c : ℕ) : ℕ :=
  a * 3^2 + b * 3^1 + c * 3^0

/-- Proves that the ternary number 121₃ is equal to the decimal number 16 --/
theorem ternary_121_equals_16 : ternary_to_decimal 1 2 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ternary_121_equals_16_l2883_288317


namespace NUMINAMATH_CALUDE_largest_coefficient_expansion_l2883_288378

theorem largest_coefficient_expansion (x : ℝ) (x_nonzero : x ≠ 0) :
  ∃ (terms : List ℝ), 
    (1/x - 1)^5 = terms.sum ∧ 
    (10/x^3 ∈ terms) ∧
    ∀ (term : ℝ), term ∈ terms → |term| ≤ |10/x^3| :=
by sorry

end NUMINAMATH_CALUDE_largest_coefficient_expansion_l2883_288378


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l2883_288394

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides
    and the sum of its interior angles is 3240 degrees. -/
theorem regular_polygon_properties (n : ℕ) (exterior_angle : ℝ) :
  exterior_angle = 18 →
  (360 : ℝ) / exterior_angle = n →
  n = 20 ∧
  180 * (n - 2) = 3240 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l2883_288394


namespace NUMINAMATH_CALUDE_logarithmic_algebraic_equivalence_l2883_288300

theorem logarithmic_algebraic_equivalence : 
  ¬(∀ x : ℝ, (Real.log (x^2 - 4) = Real.log (4*x - 7)) ↔ (x^2 - 4 = 4*x - 7)) :=
by sorry

end NUMINAMATH_CALUDE_logarithmic_algebraic_equivalence_l2883_288300


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2883_288328

theorem quadratic_minimum (x : ℝ) : 
  ∃ (min_x : ℝ), ∀ y : ℝ, 2 * x^2 + 6 * x - 5 ≥ 2 * min_x^2 + 6 * min_x - 5 ∧ min_x = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2883_288328


namespace NUMINAMATH_CALUDE_parabola_and_line_properties_l2883_288385

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Represents a line of the form y = kx + m -/
structure Line where
  k : ℝ
  m : ℝ

/-- The main theorem stating the properties of the parabola and line -/
theorem parabola_and_line_properties
  (p : Parabola)
  (l : Line)
  (passes_through_A : p.a + p.b + p.c = 0)
  (axis_of_symmetry : ∀ x, p.a * (x - 3)^2 + p.b * (x - 3) + p.c = p.a * x^2 + p.b * x + p.c)
  (line_passes_through_A : l.k + l.m = 0)
  (line_passes_through_B : ∃ x, p.a * x^2 + p.b * x + p.c = l.k * x + l.m ∧ x ≠ 1)
  (triangle_area : |l.m| = 4) :
  ((l.k = -4 ∧ l.m = 4 ∧ p.a = 2 ∧ p.b = -12 ∧ p.c = 10) ∨
   (l.k = 4 ∧ l.m = -4 ∧ p.a = -2 ∧ p.b = 12 ∧ p.c = -10)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_and_line_properties_l2883_288385


namespace NUMINAMATH_CALUDE_chord_length_l2883_288333

/-- The length of the chord formed by the intersection of a circle and a line --/
theorem chord_length (x y : ℝ) : 
  let circle := (x - 1)^2 + y^2 = 4
  let line := x + y + 1 = 0
  let chord_length := Real.sqrt (8 : ℝ)
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    ((A.1 - 1)^2 + A.2^2 = 4) ∧ (A.1 + A.2 + 1 = 0) ∧
    ((B.1 - 1)^2 + B.2^2 = 4) ∧ (B.1 + B.2 + 1 = 0)) →
  ∃ A B : ℝ × ℝ, 
    ((A.1 - 1)^2 + A.2^2 = 4) ∧ (A.1 + A.2 + 1 = 0) ∧
    ((B.1 - 1)^2 + B.2^2 = 4) ∧ (B.1 + B.2 + 1 = 0) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = chord_length :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l2883_288333


namespace NUMINAMATH_CALUDE_remaining_red_cards_l2883_288395

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (removed_red_cards : ℕ)

/-- A standard deck with half red cards and 10 red cards removed -/
def standard_deck : Deck :=
  { total_cards := 52,
    red_cards := 52 / 2,
    removed_red_cards := 10 }

/-- Theorem: The number of remaining red cards in the standard deck after removal is 16 -/
theorem remaining_red_cards (d : Deck := standard_deck) :
  d.red_cards - d.removed_red_cards = 16 := by
  sorry

end NUMINAMATH_CALUDE_remaining_red_cards_l2883_288395


namespace NUMINAMATH_CALUDE_abs_neg_x_eq_2023_l2883_288357

theorem abs_neg_x_eq_2023 (x : ℝ) :
  |(-x)| = 2023 → x = 2023 ∨ x = -2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_x_eq_2023_l2883_288357


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l2883_288392

/-- A trapezoid with an inscribed circle -/
structure InscribedTrapezoid where
  -- The length of segment BL
  BL : ℝ
  -- The length of segment CL
  CL : ℝ
  -- The length of side AB
  AB : ℝ
  -- Assumption that BL is positive
  BL_pos : BL > 0
  -- Assumption that CL is positive
  CL_pos : CL > 0
  -- Assumption that AB is positive
  AB_pos : AB > 0

/-- The area of a trapezoid with an inscribed circle -/
def area (t : InscribedTrapezoid) : ℝ :=
  -- Define the area function here
  sorry

/-- Theorem: The area of the specific trapezoid is 6.75 -/
theorem specific_trapezoid_area :
  ∀ t : InscribedTrapezoid,
  t.BL = 4 → t.CL = 1/4 → t.AB = 6 →
  area t = 6.75 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l2883_288392


namespace NUMINAMATH_CALUDE_a_range_l2883_288371

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3-2*a)^x < (3-2*a)^y

def range_a (a : ℝ) : Prop := a ≤ -2 ∨ (1 ≤ a ∧ a < 2)

theorem a_range (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_a a := by sorry

end NUMINAMATH_CALUDE_a_range_l2883_288371


namespace NUMINAMATH_CALUDE_slices_per_pizza_l2883_288365

theorem slices_per_pizza (total_pizzas : ℕ) (total_slices : ℕ) (h1 : total_pizzas = 7) (h2 : total_slices = 14) :
  total_slices / total_pizzas = 2 := by
  sorry

end NUMINAMATH_CALUDE_slices_per_pizza_l2883_288365


namespace NUMINAMATH_CALUDE_abs_difference_inequality_l2883_288375

theorem abs_difference_inequality (x : ℝ) : |x + 1| - |x - 5| < 4 ↔ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_difference_inequality_l2883_288375


namespace NUMINAMATH_CALUDE_dantes_recipe_total_l2883_288367

def dantes_recipe (eggs : ℕ) : ℕ :=
  eggs + eggs / 2

theorem dantes_recipe_total : dantes_recipe 60 = 90 := by
  sorry

end NUMINAMATH_CALUDE_dantes_recipe_total_l2883_288367


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l2883_288346

theorem arithmetic_geometric_sequence_product (a : ℕ → ℝ) (b : ℕ → ℝ) : 
  (∀ n, a n ≠ 0) →  -- a_n is non-zero for all n
  (a 3 - (a 7)^2 / 2 + a 11 = 0) →  -- given condition
  (∃ r, ∀ n, b (n + 1) = r * b n) →  -- b is a geometric sequence
  (b 7 = a 7) →  -- given condition
  (b 1 * b 13 = 16) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l2883_288346


namespace NUMINAMATH_CALUDE_a_investment_is_4410_l2883_288353

/-- Represents the investment and profit scenario of a partnership business --/
structure BusinessPartnership where
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  a_profit_share : ℕ

/-- Calculates A's investment given the business partnership scenario --/
def calculate_a_investment (bp : BusinessPartnership) : ℕ :=
  (bp.a_profit_share * (bp.b_investment + bp.c_investment)) / (bp.total_profit - bp.a_profit_share)

/-- Theorem stating that A's investment is 4410 given the specific scenario --/
theorem a_investment_is_4410 (bp : BusinessPartnership) 
  (h1 : bp.b_investment = 4200)
  (h2 : bp.c_investment = 10500)
  (h3 : bp.total_profit = 13600)
  (h4 : bp.a_profit_share = 4080) :
  calculate_a_investment bp = 4410 := by
  sorry

#eval calculate_a_investment { b_investment := 4200, c_investment := 10500, total_profit := 13600, a_profit_share := 4080 }

end NUMINAMATH_CALUDE_a_investment_is_4410_l2883_288353


namespace NUMINAMATH_CALUDE_tan_eleven_pi_over_four_l2883_288311

theorem tan_eleven_pi_over_four : Real.tan (11 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_over_four_l2883_288311


namespace NUMINAMATH_CALUDE_correct_parking_methods_l2883_288382

/-- Represents the number of consecutive parking spaces -/
def total_spaces : ℕ := 7

/-- Represents the number of cars to be parked -/
def cars_to_park : ℕ := 3

/-- Represents the number of consecutive empty spaces required -/
def required_empty_spaces : ℕ := 4

/-- Calculates the number of different parking methods -/
def parking_methods : ℕ := 24

/-- Theorem stating that the number of parking methods is correct -/
theorem correct_parking_methods :
  ∀ (total : ℕ) (cars : ℕ) (empty : ℕ),
    total = total_spaces →
    cars = cars_to_park →
    empty = required_empty_spaces →
    total - cars = empty →
    parking_methods = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_parking_methods_l2883_288382


namespace NUMINAMATH_CALUDE_impossible_closed_line_1989_sticks_l2883_288359

theorem impossible_closed_line_1989_sticks : ¬ ∃ (a b : ℕ), 2 * (a + b) = 1989 := by
  sorry

end NUMINAMATH_CALUDE_impossible_closed_line_1989_sticks_l2883_288359


namespace NUMINAMATH_CALUDE_square_perimeter_l2883_288343

theorem square_perimeter (s : ℝ) (h1 : s > 0) : 
  (5 / 2 * s = 40) → (4 * s = 64) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2883_288343


namespace NUMINAMATH_CALUDE_tan_equality_periodic_l2883_288399

theorem tan_equality_periodic (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (348 * π / 180) → n = -12 :=
by sorry

end NUMINAMATH_CALUDE_tan_equality_periodic_l2883_288399


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l2883_288340

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → (∀ m : ℕ, m * (m + 1) < 500 → m ≤ n) → n + (n + 1) = 43 :=
by sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l2883_288340


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l2883_288309

theorem greatest_integer_satisfying_inequality :
  ∃ (n : ℤ), n^2 - 11*n + 24 ≤ 0 ∧
  n = 8 ∧
  ∀ (m : ℤ), m^2 - 11*m + 24 ≤ 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l2883_288309


namespace NUMINAMATH_CALUDE_calculate_number_of_bs_l2883_288366

/-- Calculates the number of Bs given the recess rules and report card results -/
theorem calculate_number_of_bs (
  normal_recess : ℕ)
  (extra_time_per_a : ℕ)
  (extra_time_per_b : ℕ)
  (extra_time_per_c : ℕ)
  (less_time_per_d : ℕ)
  (num_as : ℕ)
  (num_cs : ℕ)
  (num_ds : ℕ)
  (total_recess : ℕ)
  (h1 : normal_recess = 20)
  (h2 : extra_time_per_a = 2)
  (h3 : extra_time_per_b = 1)
  (h4 : extra_time_per_c = 0)
  (h5 : less_time_per_d = 1)
  (h6 : num_as = 10)
  (h7 : num_cs = 14)
  (h8 : num_ds = 5)
  (h9 : total_recess = 47) :
  ∃ (num_bs : ℕ), num_bs = 12 ∧
    total_recess = normal_recess + num_as * extra_time_per_a + num_bs * extra_time_per_b + num_cs * extra_time_per_c - num_ds * less_time_per_d :=
by
  sorry


end NUMINAMATH_CALUDE_calculate_number_of_bs_l2883_288366


namespace NUMINAMATH_CALUDE_half_height_of_triangular_prism_l2883_288397

/-- Given a triangular prism with volume 576 cm³ and base area 3 cm², 
    half of its height is 96 cm. -/
theorem half_height_of_triangular_prism (volume : ℝ) (base_area : ℝ) (height : ℝ) :
  volume = 576 ∧ base_area = 3 ∧ volume = base_area * height →
  height / 2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_half_height_of_triangular_prism_l2883_288397


namespace NUMINAMATH_CALUDE_students_playing_cricket_l2883_288337

theorem students_playing_cricket 
  (total_students : ℕ) 
  (football_players : ℕ) 
  (neither_players : ℕ) 
  (both_players : ℕ) 
  (h1 : total_students = 450)
  (h2 : football_players = 325)
  (h3 : neither_players = 50)
  (h4 : both_players = 100)
  : ∃ cricket_players : ℕ, cricket_players = 175 :=
by
  sorry

end NUMINAMATH_CALUDE_students_playing_cricket_l2883_288337


namespace NUMINAMATH_CALUDE_triangle_line_equations_l2883_288308

/-- Triangle with vertices A(0,-5), B(-3,3), and C(2,0) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Given triangle -/
def givenTriangle : Triangle :=
  { A := (0, -5)
  , B := (-3, 3)
  , C := (2, 0) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) satisfies the line equation ax + by + c = 0 -/
def satisfiesLineEquation (p : ℝ × ℝ) (l : LineEquation) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

theorem triangle_line_equations (t : Triangle) :
  (t = givenTriangle) →
  (∃ (lab : LineEquation), lab.a = 8 ∧ lab.b = 3 ∧ lab.c = 15 ∧
    satisfiesLineEquation t.A lab ∧ satisfiesLineEquation t.B lab) ∧
  (∃ (lac : LineEquation), lac.a = 5 ∧ lac.b = -2 ∧ lac.c = -10 ∧
    satisfiesLineEquation t.A lac ∧ satisfiesLineEquation t.C lac) :=
by sorry

end NUMINAMATH_CALUDE_triangle_line_equations_l2883_288308


namespace NUMINAMATH_CALUDE_total_wattage_calculation_l2883_288332

def light_A_initial : ℝ := 60
def light_B_initial : ℝ := 40
def light_C_initial : ℝ := 50

def light_A_increase : ℝ := 0.12
def light_B_increase : ℝ := 0.20
def light_C_increase : ℝ := 0.15

def total_new_wattage : ℝ :=
  light_A_initial * (1 + light_A_increase) +
  light_B_initial * (1 + light_B_increase) +
  light_C_initial * (1 + light_C_increase)

theorem total_wattage_calculation :
  total_new_wattage = 172.7 := by sorry

end NUMINAMATH_CALUDE_total_wattage_calculation_l2883_288332


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2883_288364

theorem line_circle_intersection (k : ℝ) : 
  ∃ x y : ℝ, y = k * (x + 1/2) ∧ x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2883_288364


namespace NUMINAMATH_CALUDE_opposite_number_theorem_l2883_288381

theorem opposite_number_theorem (a b c : ℝ) : 
  -((-a + b - c)) = c - a - b := by sorry

end NUMINAMATH_CALUDE_opposite_number_theorem_l2883_288381


namespace NUMINAMATH_CALUDE_goods_train_speed_l2883_288384

/-- The speed of a goods train passing a woman in an opposite moving train -/
theorem goods_train_speed
  (woman_train_speed : ℝ)
  (passing_time : ℝ)
  (goods_train_length : ℝ)
  (h1 : woman_train_speed = 25)
  (h2 : passing_time = 3)
  (h3 : goods_train_length = 140) :
  ∃ (goods_train_speed : ℝ),
    goods_train_speed = 143 ∧
    (goods_train_length / passing_time) * 3.6 = woman_train_speed + goods_train_speed :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_l2883_288384


namespace NUMINAMATH_CALUDE_metal_rods_for_fence_l2883_288391

/-- Calculates the number of metal rods needed for a fence with given specifications. -/
theorem metal_rods_for_fence (
  sheets_per_panel : ℕ)
  (beams_per_panel : ℕ)
  (panels : ℕ)
  (rods_per_sheet : ℕ)
  (rods_per_beam : ℕ)
  (h1 : sheets_per_panel = 3)
  (h2 : beams_per_panel = 2)
  (h3 : panels = 10)
  (h4 : rods_per_sheet = 10)
  (h5 : rods_per_beam = 4)
  : sheets_per_panel * panels * rods_per_sheet + beams_per_panel * panels * rods_per_beam = 380 := by
  sorry

#check metal_rods_for_fence

end NUMINAMATH_CALUDE_metal_rods_for_fence_l2883_288391


namespace NUMINAMATH_CALUDE_original_price_after_discounts_l2883_288372

/-- Given an article sold at $144 after two successive discounts of 10% and 20%, 
    prove that its original price was $200. -/
theorem original_price_after_discounts (final_price : ℝ) 
  (h1 : final_price = 144)
  (discount1 : ℝ) (h2 : discount1 = 0.1)
  (discount2 : ℝ) (h3 : discount2 = 0.2) :
  ∃ (original_price : ℝ), 
    original_price = 200 ∧
    final_price = original_price * (1 - discount1) * (1 - discount2) :=
by
  sorry


end NUMINAMATH_CALUDE_original_price_after_discounts_l2883_288372


namespace NUMINAMATH_CALUDE_money_bag_madness_l2883_288369

theorem money_bag_madness (total_sacks : ℕ) (target_amount : ℝ) (target_probability : ℝ) 
  (h1 : total_sacks = 30)
  (h2 : target_amount = 65536)
  (h3 : target_probability = 0.4)
  (h4 : ∀ n : ℕ, n < total_sacks → ∃ amount : ℝ, amount = 0.5 * (2 ^ n))
  (h5 : ∃ qualifying_sacks : ℕ, qualifying_sacks = 6) :
  ∃ eliminated_sacks : ℕ, 
    eliminated_sacks = 15 ∧ 
    (qualifying_sacks : ℝ) / (total_sacks - eliminated_sacks : ℝ) ≥ target_probability :=
by sorry

end NUMINAMATH_CALUDE_money_bag_madness_l2883_288369


namespace NUMINAMATH_CALUDE_vector_problem_l2883_288345

def a : Fin 2 → ℝ := ![2, 4]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

theorem vector_problem (x : ℝ) :
  (∀ (i : Fin 2), (a i * b x i) > 0) →
  (x > -2 ∧ x ≠ 1/2) ∧
  ((∀ (i : Fin 2), ((2 * a i - b x i) * a i) = 0) →
   Real.sqrt ((a 0 + b x 0)^2 + (a 1 + b x 1)^2) = 5 * Real.sqrt 17) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l2883_288345


namespace NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_achieved_l2883_288387

theorem quadratic_minimum (x : ℝ) : 7 * x^2 - 28 * x + 2015 ≥ 1987 := by sorry

theorem quadratic_minimum_achieved : ∃ x : ℝ, 7 * x^2 - 28 * x + 2015 = 1987 := by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_achieved_l2883_288387


namespace NUMINAMATH_CALUDE_distance_between_z₁_and_z₂_l2883_288301

noncomputable def z₁ : ℂ := (Complex.I * 2 + 1)⁻¹ * (Complex.I * 3 - 1)

noncomputable def z₂ : ℂ := 1 + (1 + Complex.I)^10

theorem distance_between_z₁_and_z₂ : 
  Complex.abs (z₂ - z₁) = Real.sqrt 231.68 := by sorry

end NUMINAMATH_CALUDE_distance_between_z₁_and_z₂_l2883_288301


namespace NUMINAMATH_CALUDE_union_equals_real_l2883_288336

open Set Real

def A : Set ℝ := {x : ℝ | x^2 + x - 6 > 0}
def B : Set ℝ := {x : ℝ | -π < x ∧ x < Real.exp 1}

theorem union_equals_real : A ∪ B = univ := by
  sorry

end NUMINAMATH_CALUDE_union_equals_real_l2883_288336


namespace NUMINAMATH_CALUDE_discounted_price_calculation_l2883_288374

/-- The original price of the coat -/
def original_price : ℝ := 120

/-- The first discount percentage -/
def first_discount : ℝ := 0.25

/-- The second discount percentage -/
def second_discount : ℝ := 0.20

/-- The final price after both discounts -/
def final_price : ℝ := 72

/-- Theorem stating that applying the two discounts sequentially results in the final price -/
theorem discounted_price_calculation :
  (1 - second_discount) * ((1 - first_discount) * original_price) = final_price :=
by sorry

end NUMINAMATH_CALUDE_discounted_price_calculation_l2883_288374


namespace NUMINAMATH_CALUDE_election_win_margin_l2883_288354

theorem election_win_margin (total_votes : ℕ) (winner_votes : ℕ) :
  winner_votes = 1944 →
  (winner_votes : ℚ) / total_votes = 54 / 100 →
  winner_votes - (total_votes - winner_votes) = 288 :=
by
  sorry

end NUMINAMATH_CALUDE_election_win_margin_l2883_288354


namespace NUMINAMATH_CALUDE_gloria_tickets_count_l2883_288312

/-- Given that Gloria has 9 boxes of tickets and each box contains 5 tickets,
    prove that the total number of tickets is 45. -/
theorem gloria_tickets_count :
  let num_boxes : ℕ := 9
  let tickets_per_box : ℕ := 5
  num_boxes * tickets_per_box = 45 := by
  sorry

end NUMINAMATH_CALUDE_gloria_tickets_count_l2883_288312


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l2883_288313

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ (b : ℝ), (1 - a * Complex.I) / (1 + Complex.I) = b * Complex.I) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l2883_288313


namespace NUMINAMATH_CALUDE_race_finishing_orders_eq_twelve_l2883_288318

/-- Represents the number of possible finishing orders in a race with three participants,
    allowing for a tie only in the first place. -/
def race_finishing_orders : ℕ := 12

/-- Theorem stating that the number of possible finishing orders in a race with three participants,
    allowing for a tie only in the first place, is 12. -/
theorem race_finishing_orders_eq_twelve : race_finishing_orders = 12 := by
  sorry

end NUMINAMATH_CALUDE_race_finishing_orders_eq_twelve_l2883_288318


namespace NUMINAMATH_CALUDE_son_age_l2883_288352

theorem son_age (son_age man_age : ℕ) : 
  man_age = son_age + 24 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_son_age_l2883_288352


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l2883_288390

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a3 : a 3 = 7)
  (h_a5_a2 : a 5 = a 2 + 6) :
  a 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l2883_288390


namespace NUMINAMATH_CALUDE_cosine_equality_l2883_288335

theorem cosine_equality (a : ℝ) (h : Real.sin (π/3 + a) = 5/12) : 
  Real.cos (π/6 - a) = 5/12 := by sorry

end NUMINAMATH_CALUDE_cosine_equality_l2883_288335


namespace NUMINAMATH_CALUDE_diner_menu_problem_l2883_288360

theorem diner_menu_problem (n : ℕ) (h1 : n > 0) : 
  let vegan_dishes : ℕ := 6
  let vegan_fraction : ℚ := 1 / 6
  let nut_containing_vegan : ℕ := 5
  (vegan_dishes : ℚ) / n = vegan_fraction →
  (vegan_dishes - nut_containing_vegan : ℚ) / n = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_diner_menu_problem_l2883_288360


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2883_288358

theorem expression_simplification_and_evaluation (a : ℤ) 
  (h1 : -2 < a) (h2 : a ≤ 2) (h3 : a ≠ 0) (h4 : a ≠ 1) :
  (a - (2 * a - 1) / a) / ((a - 1) / a) = a - 1 ∧
  (a = -1 ∨ a = 2) ∧
  (a = -1 → (a - (2 * a - 1) / a) / ((a - 1) / a) = -2) ∧
  (a = 2 → (a - (2 * a - 1) / a) / ((a - 1) / a) = 1) :=
by sorry

#check expression_simplification_and_evaluation

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2883_288358


namespace NUMINAMATH_CALUDE_farmer_land_usage_l2883_288377

theorem farmer_land_usage (beans wheat corn total : ℕ) : 
  beans + wheat + corn = total →
  5 * wheat = 2 * beans →
  2 * corn = beans →
  corn = 376 →
  total = 1034 := by
sorry

end NUMINAMATH_CALUDE_farmer_land_usage_l2883_288377


namespace NUMINAMATH_CALUDE_gadget_price_proof_l2883_288370

theorem gadget_price_proof (sticker_price : ℝ) : 
  (0.80 * sticker_price - 80) = (0.65 * sticker_price - 20) → sticker_price = 400 := by
  sorry

end NUMINAMATH_CALUDE_gadget_price_proof_l2883_288370


namespace NUMINAMATH_CALUDE_max_transition_BC_l2883_288362

def channel_A_transition : ℕ := 51
def channel_B_transition : ℕ := 63
def channel_C_transition : ℕ := 63

theorem max_transition_BC : 
  max channel_B_transition channel_C_transition = 63 := by
  sorry

end NUMINAMATH_CALUDE_max_transition_BC_l2883_288362


namespace NUMINAMATH_CALUDE_max_n_when_T_less_than_2019_l2883_288339

/-- Define the arithmetic sequence a_n -/
def a (n : ℕ) : ℕ := 2 * n - 1

/-- Define the geometric sequence b_n -/
def b (n : ℕ) : ℕ := 2^(n - 1)

/-- Define the sequence c_n -/
def c (n : ℕ) : ℕ := a (b n)

/-- Define the sum T_n -/
def T (n : ℕ) : ℕ := 2^(n + 1) - n - 2

theorem max_n_when_T_less_than_2019 :
  (∀ n : ℕ, n ≤ 9 → T n < 2019) ∧ T 10 ≥ 2019 := by sorry

end NUMINAMATH_CALUDE_max_n_when_T_less_than_2019_l2883_288339


namespace NUMINAMATH_CALUDE_cos_90_degrees_l2883_288341

theorem cos_90_degrees : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_90_degrees_l2883_288341


namespace NUMINAMATH_CALUDE_class_average_weight_l2883_288303

theorem class_average_weight (students_A students_B : ℕ) (avg_weight_A avg_weight_B : ℝ) :
  students_A = 36 →
  students_B = 44 →
  avg_weight_A = 40 →
  avg_weight_B = 35 →
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B : ℝ) = 37.25 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l2883_288303


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2883_288326

/-- Given that x^2 and ∛y vary inversely, and x = 3 when y = 64, prove that y = 15 * ∛15 when xy = 90 -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) :
  (∀ x y, x^2 * y^(1/3) = k) →  -- x^2 and ∛y vary inversely
  (3^2 * 64^(1/3) = k) →        -- x = 3 when y = 64
  (x * y = 90) →                -- xy = 90
  (y = 15 * 15^(1/5)) :=        -- y = 15 * ∛15
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2883_288326


namespace NUMINAMATH_CALUDE_prime_divisibility_l2883_288319

theorem prime_divisibility (p : ℕ) (hp : Prime p) (hp2 : p > 2) :
  ∃ k : ℤ, (⌊(2 + Real.sqrt 5)^p⌋ : ℤ) - 2^(p + 1) = k * p := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l2883_288319


namespace NUMINAMATH_CALUDE_negative_five_plus_eight_equals_three_l2883_288342

theorem negative_five_plus_eight_equals_three : -5 + 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_negative_five_plus_eight_equals_three_l2883_288342


namespace NUMINAMATH_CALUDE_two_slices_per_pizza_l2883_288314

/-- Given a total number of pizza slices and a number of pizzas,
    calculate the number of slices per pizza. -/
def slices_per_pizza (total_slices : ℕ) (num_pizzas : ℕ) : ℕ :=
  total_slices / num_pizzas

/-- Prove that given 28 total slices and 14 pizzas, each pizza has 2 slices. -/
theorem two_slices_per_pizza :
  slices_per_pizza 28 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_slices_per_pizza_l2883_288314


namespace NUMINAMATH_CALUDE_hyperbola_x_axis_m_range_l2883_288334

/-- Represents the equation of a conic section -/
structure ConicSection where
  m : ℝ
  equation : ℝ → ℝ → Prop := λ x y => x^2 / m + y^2 / (m - 4) = 1

/-- Represents a hyperbola with foci on the x-axis -/
class HyperbolaXAxis extends ConicSection

/-- The range of m for a hyperbola with foci on the x-axis -/
def is_valid_m (m : ℝ) : Prop := 0 < m ∧ m < 4

/-- Theorem stating the condition for m to represent a hyperbola with foci on the x-axis -/
theorem hyperbola_x_axis_m_range (h : HyperbolaXAxis) :
  is_valid_m h.m :=
sorry

end NUMINAMATH_CALUDE_hyperbola_x_axis_m_range_l2883_288334


namespace NUMINAMATH_CALUDE_yoongi_position_l2883_288348

/-- Calculates the number of students behind a runner after passing others. -/
def students_behind (total : ℕ) (initial_position : ℕ) (passed : ℕ) : ℕ :=
  total - (initial_position - passed)

/-- Theorem stating the number of students behind Yoongi after passing others. -/
theorem yoongi_position (total : ℕ) (initial_position : ℕ) (passed : ℕ) 
  (h_total : total = 9)
  (h_initial : initial_position = 7)
  (h_passed : passed = 4) :
  students_behind total initial_position passed = 6 := by
sorry

end NUMINAMATH_CALUDE_yoongi_position_l2883_288348


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l2883_288315

/-- Two circles are tangent if they touch at exactly one point. -/
def are_tangent (c1 c2 : Circle) : Prop := sorry

/-- The distance between the centers of two circles. -/
def center_distance (c1 c2 : Circle) : ℝ := sorry

/-- The radius of a circle. -/
def radius (c : Circle) : ℝ := sorry

theorem tangent_circles_radius (c1 c2 : Circle) :
  are_tangent c1 c2 →
  center_distance c1 c2 = 7 →
  radius c1 = 5 →
  radius c2 = 2 ∨ radius c2 = 12 := by sorry

end NUMINAMATH_CALUDE_tangent_circles_radius_l2883_288315


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l2883_288349

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, (f n + m) ∣ (n^2 + f n * f m)

theorem unique_satisfying_function :
  ∃! f : ℕ → ℕ, satisfies_condition f ∧ ∀ n : ℕ, f n = n :=
by sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l2883_288349


namespace NUMINAMATH_CALUDE_company_workers_count_l2883_288351

theorem company_workers_count (total : ℕ) (men : ℕ) : 
  (total / 3 : ℚ) * (1 / 10 : ℚ) + (2 * total / 3 : ℚ) * (3 / 5 : ℚ) = men →
  men = 120 →
  total - men = 280 :=
by sorry

end NUMINAMATH_CALUDE_company_workers_count_l2883_288351


namespace NUMINAMATH_CALUDE_min_value_fraction_l2883_288329

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem min_value_fraction (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 7 = a 6 + 2 * a 5) →
  (∃ m n : ℕ, a m * a n = 16 * (a 1)^2) →
  (∃ m n : ℕ, ∀ k l : ℕ, 1 / k + 9 / l ≥ 1 / m + 9 / n) →
  (∃ m n : ℕ, 1 / m + 9 / n = 11 / 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2883_288329


namespace NUMINAMATH_CALUDE_range_of_a_l2883_288310

theorem range_of_a (p q : Prop) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 - 2*x - 2 + a ≤ 0) →
  (∃ x : ℝ, x^2 - 2*x - a = 0) →
  a ∈ Set.Icc (-1) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2883_288310


namespace NUMINAMATH_CALUDE_valid_f_forms_l2883_288302

-- Define the function g
def g (x : ℝ) : ℝ := -x^2 - 3

-- Define the properties of function f
def is_valid_f (f : ℝ → ℝ) : Prop :=
  -- f is a quadratic function
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c ∧ a ≠ 0 ∧
  -- The minimum value of f(x) on [-1,2] is 1
  (∀ x ∈ Set.Icc (-1) 2, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1) 2, f x = 1) ∧
  -- f(x) + g(x) is an odd function
  ∀ x, f (-x) + g (-x) = -(f x + g x)

-- Theorem statement
theorem valid_f_forms :
  ∀ f : ℝ → ℝ, is_valid_f f →
    (∀ x, f x = x^2 - 2 * Real.sqrt 2 * x + 3) ∨
    (∀ x, f x = x^2 + 3 * x + 3) :=
sorry

end NUMINAMATH_CALUDE_valid_f_forms_l2883_288302


namespace NUMINAMATH_CALUDE_day_50_of_prev_year_is_tuesday_l2883_288386

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a year -/
structure Year where
  number : ℤ
  isLeapYear : Bool

/-- Returns the day of the week for a given day number in a year -/
def dayOfWeek (y : Year) (dayNumber : ℕ) : DayOfWeek := sorry

/-- Returns the next year -/
def nextYear (y : Year) : Year := sorry

/-- Returns the previous year -/
def prevYear (y : Year) : Year := sorry

theorem day_50_of_prev_year_is_tuesday 
  (N : Year)
  (h1 : dayOfWeek N 250 = DayOfWeek.Friday)
  (h2 : dayOfWeek (nextYear N) 150 = DayOfWeek.Friday)
  (h3 : (nextYear N).isLeapYear = false) :
  dayOfWeek (prevYear N) 50 = DayOfWeek.Tuesday := by sorry

end NUMINAMATH_CALUDE_day_50_of_prev_year_is_tuesday_l2883_288386


namespace NUMINAMATH_CALUDE_initial_machines_count_l2883_288330

/-- Given a number of machines working at a constant rate, this theorem proves
    the number of machines initially working based on their production output. -/
theorem initial_machines_count (x : ℝ) (N : ℕ) : 
  (N : ℝ) * x / 4 = 20 * 3 * x / 6 → N = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_machines_count_l2883_288330


namespace NUMINAMATH_CALUDE_min_value_expression_l2883_288398

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 32) :
  x^2 + 4*x*y + 4*y^2 + 2*z^2 ≥ 96 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 32 ∧ x₀^2 + 4*x₀*y₀ + 4*y₀^2 + 2*z₀^2 = 96 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2883_288398


namespace NUMINAMATH_CALUDE_students_not_playing_sports_l2883_288320

theorem students_not_playing_sports (total : ℕ) (football : ℕ) (volleyball : ℕ) (one_sport : ℕ)
  (h_total : total = 40)
  (h_football : football = 20)
  (h_volleyball : volleyball = 19)
  (h_one_sport : one_sport = 15) :
  total - (football + volleyball - (football + volleyball - one_sport)) = 13 := by
  sorry

end NUMINAMATH_CALUDE_students_not_playing_sports_l2883_288320


namespace NUMINAMATH_CALUDE_chess_tournament_ratio_l2883_288388

theorem chess_tournament_ratio (total_students : ℕ) (tournament_students : ℕ) :
  total_students = 24 →
  tournament_students = 4 →
  (total_students / 3 : ℚ) = (total_students / 3 : ℕ) →
  (tournament_students : ℚ) / (total_students / 3 : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_ratio_l2883_288388


namespace NUMINAMATH_CALUDE_min_value_of_function_l2883_288368

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  (x^2 + 3*x + 1) / x ≥ 5 ∧ ∃ y > 0, (y^2 + 3*y + 1) / y = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2883_288368


namespace NUMINAMATH_CALUDE_trigonometric_equality_l2883_288344

theorem trigonometric_equality : 
  (4.34 : ℝ) = (Real.sqrt 3 * Real.sin (38 * π / 180)) / (4 * Real.sin (2 * π / 180) * Real.sin (28 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l2883_288344


namespace NUMINAMATH_CALUDE_negation_implication_true_l2883_288327

theorem negation_implication_true (a b c : ℝ) : 
  ¬(a > b → a * c^2 > b * c^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_implication_true_l2883_288327


namespace NUMINAMATH_CALUDE_inequality_range_l2883_288373

theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x + 3| > a) → a < 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2883_288373


namespace NUMINAMATH_CALUDE_scatter_plot_always_possible_l2883_288350

/-- Represents statistical data for two variables -/
structure StatisticalData where
  variable1 : List ℝ
  variable2 : List ℝ
  length_eq : variable1.length = variable2.length

/-- Represents a scatter plot -/
structure ScatterPlot where
  points : List (ℝ × ℝ)

/-- Given statistical data for two variables, it is always possible to create a scatter plot -/
theorem scatter_plot_always_possible (data : StatisticalData) : 
  ∃ (plot : ScatterPlot), true := by sorry

end NUMINAMATH_CALUDE_scatter_plot_always_possible_l2883_288350
