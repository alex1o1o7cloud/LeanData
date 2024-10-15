import Mathlib

namespace NUMINAMATH_CALUDE_redwood_percentage_increase_l2101_210128

theorem redwood_percentage_increase (num_pines : ℕ) (total_trees : ℕ) : 
  num_pines = 600 → total_trees = 1320 → 
  (total_trees - num_pines : ℚ) / num_pines * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_redwood_percentage_increase_l2101_210128


namespace NUMINAMATH_CALUDE_simplify_fraction_l2101_210126

theorem simplify_fraction (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 4) :
  (18 * a * b^3 * c^2) / (12 * a^2 * b * c) = 27 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2101_210126


namespace NUMINAMATH_CALUDE_pages_per_chapter_l2101_210130

theorem pages_per_chapter 
  (total_pages : ℕ) 
  (num_chapters : ℕ) 
  (h1 : total_pages = 555) 
  (h2 : num_chapters = 5) 
  (h3 : total_pages % num_chapters = 0) : 
  total_pages / num_chapters = 111 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_chapter_l2101_210130


namespace NUMINAMATH_CALUDE_dress_price_l2101_210189

theorem dress_price (total_revenue : ℕ) (num_dresses : ℕ) (num_shirts : ℕ) (shirt_price : ℕ) (dress_price : ℕ) :
  total_revenue = 69 →
  num_dresses = 7 →
  num_shirts = 4 →
  shirt_price = 5 →
  num_dresses * dress_price + num_shirts * shirt_price = total_revenue →
  dress_price = 7 := by
sorry

end NUMINAMATH_CALUDE_dress_price_l2101_210189


namespace NUMINAMATH_CALUDE_infinite_divisibility_equivalence_l2101_210173

theorem infinite_divisibility_equivalence :
  ∀ (a b c : ℕ+),
  (∃ (S : Set ℕ+), Set.Infinite S ∧ ∀ (n : ℕ+), n ∈ S → (a + n) ∣ (b + c * n!)) ↔
  (∃ (k : ℕ) (t : ℤ), a = 2 * k + 1 ∧ b = t.natAbs ∧ c = (t.natAbs * (2 * k).factorial)) :=
by sorry

end NUMINAMATH_CALUDE_infinite_divisibility_equivalence_l2101_210173


namespace NUMINAMATH_CALUDE_max_sum_of_two_integers_l2101_210119

theorem max_sum_of_two_integers (x y : ℕ+) : 
  y = 2 * x → x + y < 100 → (∀ a b : ℕ+, b = 2 * a → a + b < 100 → a + b ≤ x + y) → x + y = 99 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_two_integers_l2101_210119


namespace NUMINAMATH_CALUDE_envelope_weight_proof_l2101_210166

/-- The weight of the envelope in Jessica's letter mailing scenario -/
def envelope_weight : ℚ := 2/5

/-- The number of pieces of paper used -/
def paper_count : ℕ := 8

/-- The weight of each piece of paper in ounces -/
def paper_weight : ℚ := 1/5

/-- The number of stamps needed -/
def stamps_needed : ℕ := 2

/-- The maximum weight in ounces that can be mailed with the given number of stamps -/
def max_weight (stamps : ℕ) : ℚ := stamps

theorem envelope_weight_proof :
  (paper_count : ℚ) * paper_weight + envelope_weight > (stamps_needed - 1 : ℚ) ∧
  (paper_count : ℚ) * paper_weight + envelope_weight ≤ stamps_needed ∧
  envelope_weight > 0 :=
sorry

end NUMINAMATH_CALUDE_envelope_weight_proof_l2101_210166


namespace NUMINAMATH_CALUDE_prob_adjacent_vertices_decagon_l2101_210194

/-- A decagon is a polygon with 10 sides and 10 vertices. -/
def Decagon : Type := Unit

/-- The number of vertices in a decagon. -/
def num_vertices : ℕ := 10

/-- The number of vertices adjacent to any given vertex in a decagon. -/
def num_adjacent_vertices : ℕ := 2

/-- The probability of selecting two adjacent vertices when choosing 2 distinct vertices at random from a decagon. -/
def prob_adjacent_vertices (d : Decagon) : ℚ :=
  num_adjacent_vertices / (num_vertices - 1)

theorem prob_adjacent_vertices_decagon :
  ∀ d : Decagon, prob_adjacent_vertices d = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_adjacent_vertices_decagon_l2101_210194


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_l2101_210153

def number_of_people : ℕ := 8

-- Define a function to calculate the number of seating arrangements
def seating_arrangements (n : ℕ) (restricted_pair : ℕ) : ℕ :=
  Nat.factorial n - Nat.factorial (n - 1) * restricted_pair

-- Theorem statement
theorem correct_seating_arrangements :
  seating_arrangements number_of_people 2 = 30240 := by
  sorry

end NUMINAMATH_CALUDE_correct_seating_arrangements_l2101_210153


namespace NUMINAMATH_CALUDE_right_triangle_inscribed_circle_inequality_l2101_210132

theorem right_triangle_inscribed_circle_inequality (a b r : ℝ) 
  (ha : a > 0) (hb : b > 0) (hr : r > 0)
  (h_right_triangle : a^2 + b^2 = (a + b)^2 / 2)
  (h_inscribed_circle : r = a * b / (a + b + Real.sqrt (a^2 + b^2))) :
  2 + Real.sqrt 2 ≤ (2 * a * b) / ((a + b) * r) ∧ (2 * a * b) / ((a + b) * r) < 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inscribed_circle_inequality_l2101_210132


namespace NUMINAMATH_CALUDE_no_common_points_l2101_210188

/-- Theorem: If a point (x, y) is inside the parabola y^2 = 4x, 
    then the line yy = 2(x + x) and the parabola have no common points. -/
theorem no_common_points (x y : ℝ) (h : y^2 < 4*x) : 
  ∀ (x' y' : ℝ), y'^2 = 4*x' → y'*y = 2*(x + x') → False :=
by sorry

end NUMINAMATH_CALUDE_no_common_points_l2101_210188


namespace NUMINAMATH_CALUDE_range_of_u_l2101_210140

theorem range_of_u (x y : ℝ) (h : x^2/3 + y^2 = 1) :
  1 ≤ |2*x + y - 4| + |3 - x - 2*y| ∧ |2*x + y - 4| + |3 - x - 2*y| ≤ 13 := by
  sorry

end NUMINAMATH_CALUDE_range_of_u_l2101_210140


namespace NUMINAMATH_CALUDE_lock_probability_l2101_210155

/-- Given a set of keys and a subset that can open a lock, 
    calculate the probability of randomly selecting a key that opens the lock -/
def probability_open_lock (total_keys : ℕ) (opening_keys : ℕ) : ℚ :=
  opening_keys / total_keys

/-- Theorem: The probability of opening a lock with 2 out of 5 keys is 2/5 -/
theorem lock_probability : 
  probability_open_lock 5 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_lock_probability_l2101_210155


namespace NUMINAMATH_CALUDE_pirate_loot_sum_is_correct_l2101_210186

/-- Converts a base 5 number to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5^i)) 0

/-- The sum of the pirate's loot in base 10 -/
def pirateLootSum : Nat :=
  base5ToBase10 [2, 3, 1, 4] + 
  base5ToBase10 [2, 3, 4, 1] + 
  base5ToBase10 [4, 2, 0, 2] + 
  base5ToBase10 [4, 2, 2]

theorem pirate_loot_sum_is_correct : pirateLootSum = 1112 := by
  sorry

end NUMINAMATH_CALUDE_pirate_loot_sum_is_correct_l2101_210186


namespace NUMINAMATH_CALUDE_square_area_ratio_l2101_210152

theorem square_area_ratio : 
  let side_C : ℝ := 24
  let side_D : ℝ := 30
  let area_C := side_C ^ 2
  let area_D := side_D ^ 2
  area_C / area_D = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2101_210152


namespace NUMINAMATH_CALUDE_orange_mango_difference_l2101_210172

/-- Represents the total produce in kilograms for each fruit type -/
structure FruitProduce where
  mangoes : ℕ
  apples : ℕ
  oranges : ℕ

/-- Calculates the total revenue given the price per kg and total produce -/
def totalRevenue (price : ℕ) (produce : FruitProduce) : ℕ :=
  price * (produce.mangoes + produce.apples + produce.oranges)

/-- Theorem stating the difference between orange and mango produce -/
theorem orange_mango_difference (produce : FruitProduce) : 
  produce.mangoes = 400 →
  produce.apples = 2 * produce.mangoes →
  produce.oranges > produce.mangoes →
  totalRevenue 50 produce = 90000 →
  produce.oranges - produce.mangoes = 200 := by
sorry

end NUMINAMATH_CALUDE_orange_mango_difference_l2101_210172


namespace NUMINAMATH_CALUDE_not_equivalent_fraction_l2101_210104

theorem not_equivalent_fraction (x : ℝ) : x = 0.00000325 → x ≠ 1 / 308000000 := by
  sorry

end NUMINAMATH_CALUDE_not_equivalent_fraction_l2101_210104


namespace NUMINAMATH_CALUDE_arrangements_of_opening_rooms_l2101_210141

theorem arrangements_of_opening_rooms (n : ℕ) (hn : n = 6) :
  (Finset.sum (Finset.range 5) (fun k => Nat.choose n (k + 2))) = (2^n - (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_arrangements_of_opening_rooms_l2101_210141


namespace NUMINAMATH_CALUDE_actual_car_body_mass_l2101_210191

/-- Represents the scale factor between the model and the actual car body. -/
def scaleFactor : ℝ := 10

/-- Represents the mass of the model car body in kilograms. -/
def modelMass : ℝ := 1.5

/-- Calculates the mass of the actual car body given the scale factor and model mass. -/
def actualMass (s : ℝ) (m : ℝ) : ℝ := s^3 * m

/-- Theorem stating that the mass of the actual car body is 1500 kg. -/
theorem actual_car_body_mass :
  actualMass scaleFactor modelMass = 1500 := by
  sorry

end NUMINAMATH_CALUDE_actual_car_body_mass_l2101_210191


namespace NUMINAMATH_CALUDE_negative_one_greater_than_negative_sqrt_three_l2101_210143

theorem negative_one_greater_than_negative_sqrt_three : -1 > -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_greater_than_negative_sqrt_three_l2101_210143


namespace NUMINAMATH_CALUDE_unique_number_solution_l2101_210106

def is_valid_number (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  a + b + c = 10 ∧
  b = a + c ∧
  100 * c + 10 * b + a = 100 * a + 10 * b + c + 99

theorem unique_number_solution :
  ∃! (a b c : ℕ), is_valid_number a b c ∧ 100 * a + 10 * b + c = 203 :=
sorry

end NUMINAMATH_CALUDE_unique_number_solution_l2101_210106


namespace NUMINAMATH_CALUDE_haley_concert_spending_l2101_210114

def ticket_price : ℕ := 4
def tickets_for_self_and_friends : ℕ := 3
def extra_tickets : ℕ := 5

theorem haley_concert_spending :
  (tickets_for_self_and_friends + extra_tickets) * ticket_price = 32 := by
  sorry

end NUMINAMATH_CALUDE_haley_concert_spending_l2101_210114


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2101_210160

/-- The repeating decimal 0.4747... expressed as a real number -/
def repeating_decimal : ℚ :=
  (0.47 : ℚ) + (0.0047 : ℚ) / (1 - (0.01 : ℚ))

theorem repeating_decimal_as_fraction :
  repeating_decimal = 47 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l2101_210160


namespace NUMINAMATH_CALUDE_number_in_set_l2101_210178

/-- Represents a 3-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_units : units ≥ 0 ∧ units ≤ 9

/-- The value of a 3-digit number -/
def value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The reversed value of a 3-digit number -/
def reversed_value (n : ThreeDigitNumber) : ℕ :=
  100 * n.units + 10 * n.tens + n.hundreds

/-- The theorem to be proved -/
theorem number_in_set (numbers : List ThreeDigitNumber) (reversed : ThreeDigitNumber) 
  (h_reversed : reversed ∈ numbers)
  (h_diff : reversed.units - reversed.hundreds = 2)
  (h_average_increase : (reversed_value reversed - value reversed : ℚ) / numbers.length = 198/10) :
  numbers.length = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_in_set_l2101_210178


namespace NUMINAMATH_CALUDE_kayla_apples_l2101_210102

theorem kayla_apples (total : ℕ) (kylie : ℕ) (kayla : ℕ) : 
  total = 200 →
  total = kylie + kayla →
  kayla = kylie / 4 →
  kayla = 40 := by
sorry

end NUMINAMATH_CALUDE_kayla_apples_l2101_210102


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2101_210135

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3/8 < 0) ↔ k ∈ Set.Ioo (-3) 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2101_210135


namespace NUMINAMATH_CALUDE_papaya_production_l2101_210177

/-- The number of papaya trees -/
def papaya_trees : ℕ := 2

/-- The number of mango trees -/
def mango_trees : ℕ := 3

/-- The number of mangos each mango tree produces -/
def mangos_per_tree : ℕ := 20

/-- The total number of fruits -/
def total_fruits : ℕ := 80

/-- The number of papayas each papaya tree produces -/
def papayas_per_tree : ℕ := 10

theorem papaya_production :
  papaya_trees * papayas_per_tree + mango_trees * mangos_per_tree = total_fruits :=
by sorry

end NUMINAMATH_CALUDE_papaya_production_l2101_210177


namespace NUMINAMATH_CALUDE_power_comparison_l2101_210123

theorem power_comparison : (2 : ℕ)^16 / (16 : ℕ)^2 = 256 := by sorry

end NUMINAMATH_CALUDE_power_comparison_l2101_210123


namespace NUMINAMATH_CALUDE_transposition_changes_cycles_even_permutation_iff_even_diff_l2101_210181

/-- A permutation of numbers 1 to n -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- Number of cycles in a permutation -/
def numCycles (σ : Permutation n) : ℕ := sorry

/-- Perform a transposition on a permutation -/
def transpose (σ : Permutation n) (i j : Fin n) : Permutation n := sorry

/-- A permutation is even -/
def isEven (σ : Permutation n) : Prop := sorry

theorem transposition_changes_cycles (n : ℕ) (σ : Permutation n) (i j : Fin n) :
  ∃ k : ℤ, k = 1 ∨ k = -1 ∧ numCycles (transpose σ i j) = numCycles σ + k :=
sorry

theorem even_permutation_iff_even_diff (n : ℕ) (σ : Permutation n) :
  isEven σ ↔ Even (n - numCycles σ) :=
sorry

end NUMINAMATH_CALUDE_transposition_changes_cycles_even_permutation_iff_even_diff_l2101_210181


namespace NUMINAMATH_CALUDE_largest_prime_common_divisor_l2101_210127

theorem largest_prime_common_divisor :
  ∃ (n : ℕ), n.Prime ∧ n ∣ 360 ∧ n ∣ 231 ∧
  ∀ (m : ℕ), m.Prime → m ∣ 360 → m ∣ 231 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_prime_common_divisor_l2101_210127


namespace NUMINAMATH_CALUDE_smallest_triangle_side_l2101_210192

theorem smallest_triangle_side : ∃ (t : ℕ), 
  (∀ (s : ℕ), s < t → ¬(7 < s + 13 ∧ 13 < 7 + s ∧ s < 7 + 13)) ∧ 
  (7 < t + 13 ∧ 13 < 7 + t ∧ t < 7 + 13) :=
by sorry

end NUMINAMATH_CALUDE_smallest_triangle_side_l2101_210192


namespace NUMINAMATH_CALUDE_cab_driver_income_l2101_210108

/-- Theorem: Given a cab driver's income for 5 days where 4 days are known and the average income,
    prove that the income for the unknown day is as calculated. -/
theorem cab_driver_income 
  (day1 day2 day3 day5 : ℕ) 
  (average : ℕ) 
  (h1 : day1 = 300)
  (h2 : day2 = 150)
  (h3 : day3 = 750)
  (h5 : day5 = 500)
  (h_avg : average = 420)
  : ∃ day4 : ℕ, 
    day4 = 400 ∧ 
    (day1 + day2 + day3 + day4 + day5) / 5 = average :=
by
  sorry


end NUMINAMATH_CALUDE_cab_driver_income_l2101_210108


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2101_210111

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) : 
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧ 
  ∃ y, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2101_210111


namespace NUMINAMATH_CALUDE_john_biking_distance_l2101_210199

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 7^3 + d₂ * 7^2 + d₁ * 7^1 + d₀ * 7^0

/-- The problem statement --/
theorem john_biking_distance :
  base7ToBase10 3 9 5 6 = 1511 := by
  sorry

end NUMINAMATH_CALUDE_john_biking_distance_l2101_210199


namespace NUMINAMATH_CALUDE_haruto_tomatoes_l2101_210138

def tomato_problem (initial : ℕ) (eaten : ℕ) (remaining : ℕ) (given : ℕ) (left : ℕ) : Prop :=
  (initial - eaten = remaining) ∧
  (remaining / 2 = given) ∧
  (remaining - given = left)

theorem haruto_tomatoes : tomato_problem 127 19 108 54 54 := by
  sorry

end NUMINAMATH_CALUDE_haruto_tomatoes_l2101_210138


namespace NUMINAMATH_CALUDE_percentage_difference_l2101_210182

theorem percentage_difference (x y z n : ℝ) : 
  x = 8 * y ∧ 
  y = 2 * |z - n| ∧ 
  z = 1.1 * n → 
  (x - y) / x * 100 = 87.5 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l2101_210182


namespace NUMINAMATH_CALUDE_smallest_three_digit_number_l2101_210193

def Digits : Set Nat := {0, 3, 5, 6}

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100 ∈ Digits) ∧
  ((n / 10) % 10 ∈ Digits) ∧
  (n % 10 ∈ Digits) ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  (n / 100 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10)

theorem smallest_three_digit_number :
  ∀ n : Nat, is_valid_number n → n ≥ 305 :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_number_l2101_210193


namespace NUMINAMATH_CALUDE_glass_bowl_selling_price_l2101_210198

theorem glass_bowl_selling_price
  (total_bowls : ℕ)
  (cost_per_bowl : ℚ)
  (sold_bowls : ℕ)
  (percentage_gain : ℚ)
  (h1 : total_bowls = 115)
  (h2 : cost_per_bowl = 18)
  (h3 : sold_bowls = 104)
  (h4 : percentage_gain = 0.004830917874396135)
  : ∃ (selling_price : ℚ), selling_price = 20 ∧ 
    selling_price * sold_bowls = cost_per_bowl * total_bowls * (1 + percentage_gain) :=
by sorry

end NUMINAMATH_CALUDE_glass_bowl_selling_price_l2101_210198


namespace NUMINAMATH_CALUDE_batsman_average_increase_l2101_210145

/-- Represents a batsman's performance -/
structure Batsman where
  total_runs_before_16th : ℕ
  runs_in_16th : ℕ
  average_after_16th : ℚ

/-- Calculates the increase in average for a batsman -/
def average_increase (b : Batsman) : ℚ :=
  b.average_after_16th - (b.total_runs_before_16th : ℚ) / 15

/-- Theorem: The increase in average is 3 for a batsman who scores 64 runs in the 16th inning
    and has an average of 19 after the 16th inning -/
theorem batsman_average_increase
  (b : Batsman)
  (h1 : b.runs_in_16th = 64)
  (h2 : b.average_after_16th = 19)
  (h3 : b.total_runs_before_16th + b.runs_in_16th = 16 * b.average_after_16th) :
  average_increase b = 3 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_increase_l2101_210145


namespace NUMINAMATH_CALUDE_power_sum_equals_power_implies_exponent_one_l2101_210148

theorem power_sum_equals_power_implies_exponent_one (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → (2^p + 3^p = a^n) → n = 1 := by sorry

end NUMINAMATH_CALUDE_power_sum_equals_power_implies_exponent_one_l2101_210148


namespace NUMINAMATH_CALUDE_hannah_age_problem_l2101_210196

/-- Hannah's age problem -/
theorem hannah_age_problem :
  let num_brothers : ℕ := 3
  let brother_age : ℕ := 8
  let hannah_age_factor : ℕ := 2
  let hannah_age := hannah_age_factor * (num_brothers * brother_age)
  hannah_age = 48 := by sorry

end NUMINAMATH_CALUDE_hannah_age_problem_l2101_210196


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l2101_210179

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log (2 * x)

theorem min_distance_between_curves : 
  ∃ (d : ℝ), d = Real.sqrt 2 * (1 - Real.log 2) ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → 
  Real.sqrt ((x - y)^2 + (f x - g y)^2) ≥ d :=
sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l2101_210179


namespace NUMINAMATH_CALUDE_equation_one_solution_l2101_210142

theorem equation_one_solution (m : ℝ) : 
  (∃! x : ℝ, (3*x+4)*(x-8) = -50 + m*x) ↔ 
  (m = -20 + 6*Real.sqrt 6 ∨ m = -20 - 6*Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_equation_one_solution_l2101_210142


namespace NUMINAMATH_CALUDE_inverse_variation_solution_l2101_210125

/-- The inverse relationship between 5y and x^2 -/
def inverse_relation (x y : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 5 * y = k / (x ^ 2)

theorem inverse_variation_solution (x₀ y₀ x₁ : ℝ) 
  (h₀ : inverse_relation x₀ y₀)
  (h₁ : x₀ = 2)
  (h₂ : y₀ = 4)
  (h₃ : x₁ = 4) :
  ∃ y₁ : ℝ, inverse_relation x₁ y₁ ∧ y₁ = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_solution_l2101_210125


namespace NUMINAMATH_CALUDE_math_problem_distribution_l2101_210161

theorem math_problem_distribution :
  let num_problems : ℕ := 7
  let num_friends : ℕ := 12
  (num_friends ^ num_problems : ℕ) = 35831808 :=
by sorry

end NUMINAMATH_CALUDE_math_problem_distribution_l2101_210161


namespace NUMINAMATH_CALUDE_water_jars_theorem_l2101_210150

theorem water_jars_theorem (S L : ℚ) (h1 : S > 0) (h2 : L > 0) (h3 : S ≠ L) : 
  (1/3 : ℚ) * S = (1/2 : ℚ) * L → (1/2 : ℚ) * L + (1/3 : ℚ) * S = L := by
  sorry

end NUMINAMATH_CALUDE_water_jars_theorem_l2101_210150


namespace NUMINAMATH_CALUDE_triangle_theorem_l2101_210139

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (2 * t.b - t.a) * Real.cos (t.A + t.B) = -t.c * Real.cos t.A)
  (h2 : t.c = 3)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = (4 * Real.sqrt 3) / 3) :
  t.C = π/3 ∧ t.a + t.b = 5 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2101_210139


namespace NUMINAMATH_CALUDE_proposition_b_is_true_l2101_210136

theorem proposition_b_is_true : ∀ (a b : ℝ), a + b ≠ 6 → a ≠ 3 ∨ b ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_proposition_b_is_true_l2101_210136


namespace NUMINAMATH_CALUDE_equality_of_absolute_value_sums_l2101_210121

theorem equality_of_absolute_value_sums (a b c d : ℝ) 
  (h : ∀ x : ℝ, |2*x + 4| + |a*x + b| = |c*x + d|) : 
  d = 2*c := by sorry

end NUMINAMATH_CALUDE_equality_of_absolute_value_sums_l2101_210121


namespace NUMINAMATH_CALUDE_sequence_appearance_equivalence_l2101_210195

/-- For positive real numbers a and b satisfying 2ab = a - b, 
    any positive integer n appears in the sequence (⌊ak + 1/2⌋)_{k≥1} 
    if and only if it appears at least three times in the sequence (⌊bk + 1/2⌋)_{k≥1} -/
theorem sequence_appearance_equivalence (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a * b = a - b) :
  ∀ n : ℕ, n > 0 → 
    (∃ k : ℕ, k > 0 ∧ |a * k - n| < 1/2) ↔ 
    (∃ m₁ m₂ m₃ : ℕ, m₁ > 0 ∧ m₂ > 0 ∧ m₃ > 0 ∧ m₁ ≠ m₂ ∧ m₁ ≠ m₃ ∧ m₂ ≠ m₃ ∧ 
      |b * m₁ - n| < 1/2 ∧ |b * m₂ - n| < 1/2 ∧ |b * m₃ - n| < 1/2) := by
  sorry


end NUMINAMATH_CALUDE_sequence_appearance_equivalence_l2101_210195


namespace NUMINAMATH_CALUDE_stickers_after_birthday_l2101_210131

def initial_stickers : ℕ := 39
def birthday_stickers : ℕ := 22

theorem stickers_after_birthday :
  initial_stickers + birthday_stickers = 61 := by
  sorry

end NUMINAMATH_CALUDE_stickers_after_birthday_l2101_210131


namespace NUMINAMATH_CALUDE_raj_house_bedrooms_l2101_210176

/-- Represents the floor plan of Raj's house -/
structure RajHouse where
  total_area : ℕ
  bedroom_side : ℕ
  bathroom_length : ℕ
  bathroom_width : ℕ
  num_bathrooms : ℕ
  kitchen_area : ℕ

/-- Calculates the number of bedrooms in Raj's house -/
def num_bedrooms (house : RajHouse) : ℕ :=
  let bathroom_area := house.bathroom_length * house.bathroom_width * house.num_bathrooms
  let kitchen_living_area := 2 * house.kitchen_area
  let non_bedroom_area := bathroom_area + kitchen_living_area
  let bedroom_area := house.total_area - non_bedroom_area
  bedroom_area / (house.bedroom_side * house.bedroom_side)

/-- Theorem stating that Raj's house has 4 bedrooms -/
theorem raj_house_bedrooms :
  let house : RajHouse := {
    total_area := 1110,
    bedroom_side := 11,
    bathroom_length := 6,
    bathroom_width := 8,
    num_bathrooms := 2,
    kitchen_area := 265
  }
  num_bedrooms house = 4 := by
  sorry


end NUMINAMATH_CALUDE_raj_house_bedrooms_l2101_210176


namespace NUMINAMATH_CALUDE_flea_landing_product_l2101_210129

/-- The number of circles in the arrangement -/
def num_circles : ℕ := 12

/-- The number of steps the red flea takes clockwise -/
def red_steps : ℕ := 1991

/-- The number of steps the black flea takes counterclockwise -/
def black_steps : ℕ := 1949

/-- The final position of a flea after taking a number of steps -/
def final_position (steps : ℕ) : ℕ :=
  steps % num_circles

/-- The position of the black flea, adjusted for counterclockwise movement -/
def black_position : ℕ :=
  num_circles - (final_position black_steps)

theorem flea_landing_product :
  final_position red_steps * black_position = 77 := by
  sorry

end NUMINAMATH_CALUDE_flea_landing_product_l2101_210129


namespace NUMINAMATH_CALUDE_rectangle_triangle_configuration_l2101_210157

theorem rectangle_triangle_configuration (AB AD : ℝ) (h1 : AB = 8) (h2 : AD = 10) : ∃ (DE : ℝ),
  let ABCD_area := AB * AD
  let DCE_area := ABCD_area / 2
  let DC := AD
  let CE := 2 * DCE_area / DC
  DE^2 = DC^2 + CE^2 ∧ DE = 2 * Real.sqrt 41 := by sorry

end NUMINAMATH_CALUDE_rectangle_triangle_configuration_l2101_210157


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2101_210175

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : is_geometric_sequence a)
  (h_prod1 : a 1 * a 2 * a 3 = 5)
  (h_prod2 : a 4 * a 8 * a 9 = 10) :
  a 4 * a 5 * a 6 = 5 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2101_210175


namespace NUMINAMATH_CALUDE_min_value_theorem_l2101_210165

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 4 * a + b = 1) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x + y = 1 → 1 / (2 * x) + 2 / y ≥ 1 / (2 * a) + 2 / b) ∧
  1 / (2 * a) + 2 / b = 8 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2101_210165


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l2101_210137

theorem reciprocal_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 0) : 1 / x < 1 / y := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l2101_210137


namespace NUMINAMATH_CALUDE_subtraction_with_division_l2101_210169

theorem subtraction_with_division : 3034 - (1002 / 200.4) = 3029 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_with_division_l2101_210169


namespace NUMINAMATH_CALUDE_area_between_circles_l2101_210170

-- Define the circles
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the centers of the circles
def center_X : ℝ × ℝ := (0, 0)
def center_Y : ℝ × ℝ := (2, 0)
def center_Z : ℝ × ℝ := (0, 2)

-- Define the circles
def X := Circle center_X 1
def Y := Circle center_Y 1
def Z := Circle center_Z 1

-- Define the area function
def area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_between_circles :
  (∀ p ∈ X ∩ Y, p = (1, 0)) →  -- X and Y are tangent
  (∃ p, p ∈ X ∩ Z ∧ p ≠ center_X) →  -- Z is tangent to X
  (∀ p, p ∉ Y ∩ Z) →  -- Z does not intersect Y
  area (Z \ X) = π / 2 := by sorry

end NUMINAMATH_CALUDE_area_between_circles_l2101_210170


namespace NUMINAMATH_CALUDE_hyperbola_triangle_area_l2101_210115

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-5, 0)
def right_focus : ℝ × ℝ := (5, 0)

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop := 
  hyperbola P.1 P.2

-- Define the right angle condition
def right_angle (P : ℝ × ℝ) : Prop :=
  let F₁ := left_focus
  let F₂ := right_focus
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0

-- Theorem statement
theorem hyperbola_triangle_area (P : ℝ × ℝ) :
  point_on_hyperbola P → right_angle P → 
  let F₁ := left_focus
  let F₂ := right_focus
  let area := (1/2) * ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2).sqrt * 
              ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2).sqrt
  area = 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_area_l2101_210115


namespace NUMINAMATH_CALUDE_max_ranked_participants_l2101_210183

/-- The maximum number of participants that can be awarded a rank in a chess tournament -/
theorem max_ranked_participants (n : ℕ) (rank_threshold : ℚ) : 
  n = 30 →
  rank_threshold = 60 / 100 →
  ∃ (max_ranked : ℕ), max_ranked = 23 ∧ 
    (∀ (ranked : ℕ), 
      ranked ≤ n ∧
      (ranked : ℚ) * rank_threshold * (n - 1 : ℚ) ≤ (n * (n - 1) / 2 : ℚ) →
      ranked ≤ max_ranked) :=
by sorry

end NUMINAMATH_CALUDE_max_ranked_participants_l2101_210183


namespace NUMINAMATH_CALUDE_rectangle_similarity_l2101_210174

theorem rectangle_similarity (x y : ℝ) (hxy : 0 < x ∧ x < y) :
  let r := (y, x)
  let r' := (y - x, x)
  let r'' := if y - x < x then ((y - x), (2 * x - y)) else (x, (y - 2 * x))
  ¬ (r'.1 / r'.2 = r.1 / r.2) →
  (r''.1 / r''.2 = r.1 / r.2) →
  y / x = 1 + Real.sqrt 2 ∨ y / x = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_similarity_l2101_210174


namespace NUMINAMATH_CALUDE_star_two_neg_three_l2101_210144

/-- The ⋆ operation defined on real numbers -/
def star (a b : ℝ) : ℝ := a^2 * b^2 + a - 1

/-- Theorem stating that 2 ⋆ (-3) = 37 -/
theorem star_two_neg_three : star 2 (-3) = 37 := by
  sorry

end NUMINAMATH_CALUDE_star_two_neg_three_l2101_210144


namespace NUMINAMATH_CALUDE_wendy_score_l2101_210116

/-- Wendy's video game scoring system -/
structure GameScore where
  points_per_treasure : ℕ
  treasures_level1 : ℕ
  treasures_level2 : ℕ

/-- Calculate the total score for Wendy's game -/
def total_score (game : GameScore) : ℕ :=
  (game.treasures_level1 + game.treasures_level2) * game.points_per_treasure

/-- Theorem: Wendy's total score is 35 points -/
theorem wendy_score : 
  ∀ (game : GameScore), 
  game.points_per_treasure = 5 → 
  game.treasures_level1 = 4 → 
  game.treasures_level2 = 3 → 
  total_score game = 35 := by
  sorry

end NUMINAMATH_CALUDE_wendy_score_l2101_210116


namespace NUMINAMATH_CALUDE_f_properties_l2101_210151

-- Define the function f
def f (x : ℝ) := -x^2 - 4*x + 1

-- Theorem statement
theorem f_properties :
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≤ m ∧ (∃ (x₀ : ℝ), f x₀ = m) ∧ m = 5) ∧
  (∀ (x y : ℝ), x < y ∧ y < -2 → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2101_210151


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_original_solution_l2101_210164

theorem alcohol_percentage_in_original_solution 
  (original_volume : ℝ)
  (added_water : ℝ)
  (new_mixture_percentage : ℝ)
  (h1 : original_volume = 3)
  (h2 : added_water = 1)
  (h3 : new_mixture_percentage = 24.75) :
  let new_volume := original_volume + added_water
  let alcohol_amount := (new_mixture_percentage / 100) * new_volume
  (alcohol_amount / original_volume) * 100 = 33 := by
sorry


end NUMINAMATH_CALUDE_alcohol_percentage_in_original_solution_l2101_210164


namespace NUMINAMATH_CALUDE_joes_total_lift_weight_l2101_210105

/-- The total weight of Joe's two lifts is 1500 pounds, given the conditions of the weight-lifting competition. -/
theorem joes_total_lift_weight :
  let first_lift : ℕ := 600
  let second_lift : ℕ := 2 * first_lift - 300
  first_lift + second_lift = 1500 := by
  sorry

end NUMINAMATH_CALUDE_joes_total_lift_weight_l2101_210105


namespace NUMINAMATH_CALUDE_no_prime_sum_power_four_l2101_210134

theorem no_prime_sum_power_four (n : ℕ+) : ¬ Prime (4^(n : ℕ) + (n : ℕ)^4) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_power_four_l2101_210134


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l2101_210154

/-- Profit as a percentage of revenue in the previous year, given:
  1. In 1999, revenues fell by 30% compared to the previous year.
  2. In 1999, profits were 14% of revenues.
  3. Profits in 1999 were 98% of the profits in the previous year. -/
theorem profit_percentage_previous_year (R : ℝ) (P : ℝ) 
  (h1 : 0.7 * R = R - 0.3 * R)  -- Revenue fell by 30%
  (h2 : 0.14 * (0.7 * R) = 0.098 * R)  -- Profits were 14% of revenues in 1999
  (h3 : 0.98 * P = 0.098 * R)  -- Profits in 1999 were 98% of previous year
  : P / R = 0.1 := by
  sorry

#check profit_percentage_previous_year

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l2101_210154


namespace NUMINAMATH_CALUDE_problem_solution_l2101_210156

-- Define the function f
def f (x m : ℝ) : ℝ := |x - 1| + |x + 3| - m

-- Define the theorem
theorem problem_solution :
  (∃ m : ℝ, ∀ x : ℝ, f x m < 5 ↔ -4 < x ∧ x < 2) →
  (∀ a b c : ℝ, a^2 + b^2/4 + c^2/9 = 1 → a + b + c ≤ Real.sqrt 14) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2101_210156


namespace NUMINAMATH_CALUDE_quarters_remaining_l2101_210171

/-- Calculates the number of quarters remaining after paying for a dress -/
theorem quarters_remaining (initial_quarters : ℕ) (dress_cost : ℚ) (quarter_value : ℚ) : 
  initial_quarters = 160 → 
  dress_cost = 35 → 
  quarter_value = 1/4 → 
  initial_quarters - (dress_cost / quarter_value).floor = 20 := by
sorry

end NUMINAMATH_CALUDE_quarters_remaining_l2101_210171


namespace NUMINAMATH_CALUDE_cos_135_degrees_l2101_210101

theorem cos_135_degrees : Real.cos (135 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l2101_210101


namespace NUMINAMATH_CALUDE_correct_set_representations_l2101_210147

-- Define the sets
def RealNumbers : Type := Real
def NaturalNumbers : Type := Nat
def Integers : Type := Int
def RationalNumbers : Type := Rat

-- State the theorem
theorem correct_set_representations :
  (RealNumbers = ℝ) ∧
  (NaturalNumbers = ℕ) ∧
  (Integers = ℤ) ∧
  (RationalNumbers = ℚ) := by
  sorry

end NUMINAMATH_CALUDE_correct_set_representations_l2101_210147


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l2101_210197

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sum of the first three terms of the binomial expansion -/
def first_three_sum (n : ℕ) : ℕ := binomial n 0 + binomial n 1 + binomial n 2

/-- The constant term in the expansion -/
def constant_term (n : ℕ) : ℤ := binomial n 4 * (-2)^4

/-- The coefficient with the largest absolute value in the expansion -/
def largest_coeff (n : ℕ) : ℤ := binomial n 8 * 2^8

theorem binomial_expansion_properties :
  ∃ n : ℕ, 
    first_three_sum n = 79 ∧ 
    constant_term n = 7920 ∧ 
    largest_coeff n = 126720 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l2101_210197


namespace NUMINAMATH_CALUDE_opposite_of_neg_three_l2101_210162

/-- The opposite number of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- Theorem: The opposite number of -3 is 3 -/
theorem opposite_of_neg_three : opposite (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_three_l2101_210162


namespace NUMINAMATH_CALUDE_fraction_zero_solution_l2101_210180

theorem fraction_zero_solution (x : ℝ) : 
  (x^2 - 4) / (x + 4) = 0 ∧ x ≠ -4 → x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_solution_l2101_210180


namespace NUMINAMATH_CALUDE_limit_to_infinity_l2101_210124

theorem limit_to_infinity (M : ℝ) (h : M > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n > N → (2 * n^2 - 3 * n + 2) / (n + 2) > M := by
  sorry

end NUMINAMATH_CALUDE_limit_to_infinity_l2101_210124


namespace NUMINAMATH_CALUDE_college_application_fee_cost_l2101_210112

/-- Proves that the cost of each college application fee is $25.00 -/
theorem college_application_fee_cost 
  (hourly_rate : ℝ) 
  (num_colleges : ℕ) 
  (hours_worked : ℕ) 
  (h1 : hourly_rate = 10)
  (h2 : num_colleges = 6)
  (h3 : hours_worked = 15) :
  (hourly_rate * hours_worked) / num_colleges = 25 := by
sorry

end NUMINAMATH_CALUDE_college_application_fee_cost_l2101_210112


namespace NUMINAMATH_CALUDE_quadratic_sum_of_b_and_c_l2101_210158

/-- Given a quadratic expression x^2 - 24x + 50, prove that when written in the form (x+b)^2 + c, b + c = -106 -/
theorem quadratic_sum_of_b_and_c : ∃ b c : ℝ, 
  (∀ x : ℝ, x^2 - 24*x + 50 = (x + b)^2 + c) ∧ 
  (b + c = -106) := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_b_and_c_l2101_210158


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2101_210187

theorem complex_equation_solution (x y : ℝ) : 
  (Complex.mk (2 * x - 1) 1 = Complex.mk y (y - 2)) → x = 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2101_210187


namespace NUMINAMATH_CALUDE_theater_rows_count_l2101_210167

/-- Represents a theater seating arrangement -/
structure Theater where
  total_seats : ℕ
  num_rows : ℕ
  first_row_seats : ℕ

/-- Checks if the theater satisfies the given conditions -/
def is_valid_theater (t : Theater) : Prop :=
  t.num_rows > 16 ∧
  t.total_seats = (t.first_row_seats + (t.first_row_seats + t.num_rows - 1)) * t.num_rows / 2

/-- The main theorem stating that a theater with 1000 seats satisfying the conditions has 25 rows -/
theorem theater_rows_count : 
  ∀ t : Theater, t.total_seats = 1000 → is_valid_theater t → t.num_rows = 25 :=
by sorry

end NUMINAMATH_CALUDE_theater_rows_count_l2101_210167


namespace NUMINAMATH_CALUDE_quadratic_factorization_conditions_l2101_210149

theorem quadratic_factorization_conditions (b : ℤ) : 
  ¬ ∀ (m n p q : ℤ), 
    (15 : ℤ) * x^2 + b * x + 75 = (m * x + n) * (p * x + q) → 
    ∃ (r s : ℤ), (15 : ℤ) * x^2 + b * x + 75 = (m * x + n) * (p * x + q) * (r * x + s) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_conditions_l2101_210149


namespace NUMINAMATH_CALUDE_rectangle_area_l2101_210100

theorem rectangle_area (breadth length perimeter area : ℝ) : 
  length = 3 * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 56 →
  area = length * breadth →
  area = 147 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2101_210100


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2101_210120

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 4 + Real.sqrt 15 ∧ x₂ = 4 - Real.sqrt 15 ∧ 
   x₁^2 - 8*x₁ + 1 = 0 ∧ x₂^2 - 8*x₂ + 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 2 ∧ y₂ = 1 ∧
   y₁*(y₁ - 2) - y₁ + 2 = 0 ∧ y₂*(y₂ - 2) - y₂ + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2101_210120


namespace NUMINAMATH_CALUDE_probability_n_less_than_m_plus_one_l2101_210118

/-- The number of balls in the bag -/
def num_balls : ℕ := 4

/-- The set of possible ball numbers -/
def ball_numbers : Finset ℕ := Finset.range num_balls

/-- The sample space of all possible outcomes (m, n) -/
def sample_space : Finset (ℕ × ℕ) :=
  Finset.product ball_numbers ball_numbers

/-- The event where n < m + 1 -/
def event : Finset (ℕ × ℕ) :=
  sample_space.filter (fun p => p.2 < p.1 + 1)

/-- The probability of the event -/
noncomputable def probability : ℚ :=
  (event.card : ℚ) / (sample_space.card : ℚ)

theorem probability_n_less_than_m_plus_one :
  probability = 5/8 := by sorry

end NUMINAMATH_CALUDE_probability_n_less_than_m_plus_one_l2101_210118


namespace NUMINAMATH_CALUDE_expression_simplification_l2101_210163

theorem expression_simplification : 
  ∃ (a b c : ℕ+), 
    (2 * Real.sqrt 3 + 2 / Real.sqrt 3 + 3 * Real.sqrt 2 + 3 / Real.sqrt 2 = (a * Real.sqrt 3 + b * Real.sqrt 2) / c) ∧
    (∀ (a' b' c' : ℕ+), 
      (2 * Real.sqrt 3 + 2 / Real.sqrt 3 + 3 * Real.sqrt 2 + 3 / Real.sqrt 2 = (a' * Real.sqrt 3 + b' * Real.sqrt 2) / c') →
      c ≤ c') ∧
    (a + b + c = 45) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2101_210163


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l2101_210117

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l2101_210117


namespace NUMINAMATH_CALUDE_speed_with_stream_is_ten_l2101_210146

/-- The speed of a man rowing a boat with and against a stream. -/
structure BoatSpeed where
  /-- Speed against the stream in km/h -/
  against_stream : ℝ
  /-- Speed in still water in km/h -/
  still_water : ℝ

/-- Calculate the speed with the stream given speeds against stream and in still water -/
def speed_with_stream (bs : BoatSpeed) : ℝ :=
  2 * bs.still_water - bs.against_stream

/-- Theorem stating that given the specified conditions, the speed with the stream is 10 km/h -/
theorem speed_with_stream_is_ten (bs : BoatSpeed) 
    (h1 : bs.against_stream = 10) 
    (h2 : bs.still_water = 7) : 
    speed_with_stream bs = 10 := by
  sorry

end NUMINAMATH_CALUDE_speed_with_stream_is_ten_l2101_210146


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l2101_210107

/-- Proves that a man's rowing speed in still water is 15 kmph given the conditions of downstream rowing --/
theorem mans_rowing_speed (current_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  current_speed = 3 →
  distance = 70 →
  time = 13.998880089592832 →
  (distance / time - current_speed * 1000 / 3600) * 3.6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_mans_rowing_speed_l2101_210107


namespace NUMINAMATH_CALUDE_buratino_problem_l2101_210122

/-- The sum of a geometric sequence with first term 1 and common ratio 2 -/
def geometricSum (n : ℕ) : ℕ := 2^n - 1

/-- The total payment in kopeks -/
def totalPayment : ℕ := 65535

theorem buratino_problem :
  ∃ n : ℕ, geometricSum n = totalPayment ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_buratino_problem_l2101_210122


namespace NUMINAMATH_CALUDE_trailingZeros_50_factorial_l2101_210159

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 50! is 12 -/
theorem trailingZeros_50_factorial : trailingZeros 50 = 12 := by
  sorry

end NUMINAMATH_CALUDE_trailingZeros_50_factorial_l2101_210159


namespace NUMINAMATH_CALUDE_tax_reduction_percentage_l2101_210190

/-- Given a commodity with tax and consumption, proves that if the tax is reduced by a certain percentage,
    consumption increases by 15%, and revenue decreases by 8%, then the tax reduction percentage is 20%. -/
theorem tax_reduction_percentage
  (T : ℝ) -- Original tax
  (C : ℝ) -- Original consumption
  (X : ℝ) -- Percentage by which tax is diminished
  (h1 : T > 0)
  (h2 : C > 0)
  (h3 : X ≥ 0)
  (h4 : X ≤ 100)
  (h5 : T * (1 - X / 100) * C * 1.15 = 0.92 * T * C) -- Revenue equation
  : X = 20 := by sorry

end NUMINAMATH_CALUDE_tax_reduction_percentage_l2101_210190


namespace NUMINAMATH_CALUDE_range_of_sum_l2101_210184

def f (x : ℝ) := |2 - x^2|

theorem range_of_sum (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  ∃ (y : ℝ), 2 < y ∧ y < 2 * Real.sqrt 2 ∧ y = a + b :=
sorry

end NUMINAMATH_CALUDE_range_of_sum_l2101_210184


namespace NUMINAMATH_CALUDE_median_BD_correct_altitude_CE_correct_l2101_210110

/-- Triangle with vertices A(2,3), B(-1,0), and C(5,-1) -/
structure Triangle where
  A : ℝ × ℝ := (2, 3)
  B : ℝ × ℝ := (-1, 0)
  C : ℝ × ℝ := (5, -1)

/-- Line equation in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Median BD of the triangle -/
def median_BD (t : Triangle) : LineEquation :=
  { a := 2, b := -9, c := 2 }

/-- Altitude CE of the triangle -/
def altitude_CE (t : Triangle) : LineEquation :=
  { a := 1, b := 1, c := -4 }

/-- A point (x, y) lies on a line if it satisfies the line equation -/
def point_on_line (p : ℝ × ℝ) (l : LineEquation) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

theorem median_BD_correct (t : Triangle) : 
  point_on_line t.B (median_BD t) ∧ 
  point_on_line ((t.A.1 + t.C.1) / 2, (t.A.2 + t.C.2) / 2) (median_BD t) :=
sorry

theorem altitude_CE_correct (t : Triangle) : 
  point_on_line t.C (altitude_CE t) ∧ 
  (t.A.2 - t.B.2) * (t.C.1 - t.A.1) = (t.A.1 - t.B.1) * (t.C.2 - t.A.2) :=
sorry

end NUMINAMATH_CALUDE_median_BD_correct_altitude_CE_correct_l2101_210110


namespace NUMINAMATH_CALUDE_no_right_triangle_with_sides_13_17_k_l2101_210113

theorem no_right_triangle_with_sides_13_17_k : 
  ¬ ∃ (k : ℕ), k > 0 ∧ 
  ((13 * 13 + 17 * 17 = k * k) ∨ 
   (13 * 13 + k * k = 17 * 17) ∨ 
   (17 * 17 + k * k = 13 * 13)) := by
sorry

end NUMINAMATH_CALUDE_no_right_triangle_with_sides_13_17_k_l2101_210113


namespace NUMINAMATH_CALUDE_certain_number_proof_l2101_210185

theorem certain_number_proof (p q : ℝ) 
  (h1 : 3 / p = 8) 
  (h2 : p - q = 0.20833333333333334) : 
  3 / q = 18 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2101_210185


namespace NUMINAMATH_CALUDE_gavins_green_shirts_l2101_210103

theorem gavins_green_shirts (total_shirts : ℕ) (blue_shirts : ℕ) (green_shirts : ℕ) 
  (h1 : total_shirts = 23)
  (h2 : blue_shirts = 6)
  (h3 : green_shirts = total_shirts - blue_shirts) :
  green_shirts = 17 :=
by sorry

end NUMINAMATH_CALUDE_gavins_green_shirts_l2101_210103


namespace NUMINAMATH_CALUDE_sequence_formula_l2101_210109

theorem sequence_formula (n : ℕ+) (S : ℕ+ → ℝ) (a : ℕ+ → ℝ) 
  (h : ∀ k : ℕ+, S k = a k - 3) : 
  a n = 2 * 3^(n : ℝ) := by
sorry

end NUMINAMATH_CALUDE_sequence_formula_l2101_210109


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l2101_210133

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l2101_210133


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l2101_210168

/-- 
Given a rectangular prism with edges in the ratio 3:2:1 and 
the sum of all edge lengths equal to 72 cm, its volume is 162 cubic centimeters.
-/
theorem rectangular_prism_volume (l w h : ℝ) : 
  l / w = 3 / 2 → 
  w / h = 2 / 1 → 
  4 * (l + w + h) = 72 → 
  l * w * h = 162 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l2101_210168
