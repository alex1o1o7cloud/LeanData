import Mathlib

namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_9_l2015_201550

theorem smallest_four_digit_mod_9 :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 9 = 8 → 1007 ≤ n) ∧
  1000 ≤ 1007 ∧ 1007 < 10000 ∧ 1007 % 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_9_l2015_201550


namespace NUMINAMATH_CALUDE_square_area_problem_l2015_201526

theorem square_area_problem (s₁ s₂ s₃ s₄ s₅ : ℝ) (h₁ : s₁ = 3) (h₂ : s₂ = 7) (h₃ : s₃ = 22) :
  s₁ + s₂ + s₃ + s₄ + s₅ = s₃ + s₅ → s₄ = 18 :=
by sorry

end NUMINAMATH_CALUDE_square_area_problem_l2015_201526


namespace NUMINAMATH_CALUDE_cupboard_cost_price_l2015_201591

theorem cupboard_cost_price (C : ℝ) : C = 7450 :=
  let selling_price := C * 0.86
  let profitable_price := C * 1.14
  have h1 : profitable_price = selling_price + 2086 := by sorry
  sorry

end NUMINAMATH_CALUDE_cupboard_cost_price_l2015_201591


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_l2015_201512

/-- The quadratic inequality kx^2 - 2x + 6k < 0 -/
def quadratic_inequality (k : ℝ) (x : ℝ) : Prop := k * x^2 - 2 * x + 6 * k < 0

/-- The solution set for case 1: x < -3 or x > -2 -/
def solution_set_1 (x : ℝ) : Prop := x < -3 ∨ x > -2

/-- The solution set for case 2: all real numbers -/
def solution_set_2 (x : ℝ) : Prop := True

/-- The solution set for case 3: empty set -/
def solution_set_3 (x : ℝ) : Prop := False

theorem quadratic_inequality_solutions (k : ℝ) (h : k ≠ 0) :
  (∀ x, quadratic_inequality k x ↔ solution_set_1 x) → k = -2/5 ∧
  (∀ x, quadratic_inequality k x ↔ solution_set_2 x) → k < -Real.sqrt 6 / 6 ∧
  (∀ x, quadratic_inequality k x ↔ solution_set_3 x) → k ≥ Real.sqrt 6 / 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_l2015_201512


namespace NUMINAMATH_CALUDE_probability_through_C_and_D_l2015_201522

/-- Represents the number of eastward and southward moves between two intersections -/
structure Moves where
  east : Nat
  south : Nat

/-- Calculates the number of possible paths given a number of eastward and southward moves -/
def pathCount (m : Moves) : Nat :=
  Nat.choose (m.east + m.south) m.east

/-- The moves from A to C -/
def movesAC : Moves := ⟨3, 2⟩

/-- The moves from C to D -/
def movesCD : Moves := ⟨2, 1⟩

/-- The moves from D to B -/
def movesDB : Moves := ⟨1, 2⟩

/-- The total moves from A to B -/
def movesAB : Moves := ⟨movesAC.east + movesCD.east + movesDB.east, movesAC.south + movesCD.south + movesDB.south⟩

/-- The probability of choosing a specific path at each intersection -/
def pathProbability (m : Moves) : Rat :=
  1 / (2 ^ (m.east + m.south))

theorem probability_through_C_and_D :
  (pathCount movesAC * pathCount movesCD * pathCount movesDB : Rat) /
  (pathCount movesAB : Rat) = 15 / 77 := by sorry

end NUMINAMATH_CALUDE_probability_through_C_and_D_l2015_201522


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_similar_triangle_perimeter_proof_l2015_201564

/-- Given two similar triangles where the smaller triangle has sides 15, 15, and 24,
    and the larger triangle has its longest side measuring 72,
    the perimeter of the larger triangle is 162. -/
theorem similar_triangle_perimeter : ℝ → ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c d p =>
    (a = 15 ∧ b = 15 ∧ c = 24) →  -- Dimensions of smaller triangle
    (d = 72) →                    -- Longest side of larger triangle
    (d / c = b / a) →             -- Triangles are similar
    (p = 3 * a + d) →             -- Perimeter of larger triangle
    p = 162

theorem similar_triangle_perimeter_proof : similar_triangle_perimeter 15 15 24 72 162 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_similar_triangle_perimeter_proof_l2015_201564


namespace NUMINAMATH_CALUDE_alice_baking_cake_l2015_201519

theorem alice_baking_cake (total_flour : ℕ) (cup_capacity : ℕ) (h1 : total_flour = 750) (h2 : cup_capacity = 125) :
  total_flour / cup_capacity = 6 := by
  sorry

end NUMINAMATH_CALUDE_alice_baking_cake_l2015_201519


namespace NUMINAMATH_CALUDE_parallelogram_above_x_axis_ratio_l2015_201594

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four points -/
structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculates the area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ := sorry

/-- Calculates the area of the part of the parallelogram above the x-axis -/
def areaAboveXAxis (p : Parallelogram) : ℝ := sorry

/-- The main theorem to be proved -/
theorem parallelogram_above_x_axis_ratio 
  (p : Parallelogram) 
  (h1 : p.P = ⟨-1, 1⟩) 
  (h2 : p.Q = ⟨3, -5⟩) 
  (h3 : p.R = ⟨1, -3⟩) 
  (h4 : p.S = ⟨-3, 3⟩) : 
  areaAboveXAxis p / parallelogramArea p = 1/4 := by sorry

end NUMINAMATH_CALUDE_parallelogram_above_x_axis_ratio_l2015_201594


namespace NUMINAMATH_CALUDE_product_of_x_values_l2015_201528

theorem product_of_x_values (x : ℝ) : 
  (|15 / x - 2| = 3) → (∃ y : ℝ, (|15 / y - 2| = 3) ∧ x * y = -45) :=
by sorry

end NUMINAMATH_CALUDE_product_of_x_values_l2015_201528


namespace NUMINAMATH_CALUDE_small_circle_radius_l2015_201545

/-- Given two circles where the radius of the larger circle is 80 cm and 4 times
    the radius of the smaller circle, prove that the radius of the smaller circle is 20 cm. -/
theorem small_circle_radius (r : ℝ) : 
  r > 0 → 4 * r = 80 → r = 20 := by
  sorry

end NUMINAMATH_CALUDE_small_circle_radius_l2015_201545


namespace NUMINAMATH_CALUDE_derangement_even_index_odd_l2015_201537

/-- Definition of derangement numbers -/
def D : ℕ → ℕ
  | 0 => 0  -- D₀ is defined as 0 for completeness
  | 1 => 0
  | 2 => 1
  | 3 => 2
  | 4 => 9
  | (n + 5) => (n + 4) * (D (n + 4) + D (n + 3))

/-- Theorem: D₂ₙ is odd for all positive natural numbers n -/
theorem derangement_even_index_odd (n : ℕ+) : Odd (D (2 * n)) := by
  sorry

end NUMINAMATH_CALUDE_derangement_even_index_odd_l2015_201537


namespace NUMINAMATH_CALUDE_solution_set_inequality_empty_solution_set_l2015_201569

-- Part 1
theorem solution_set_inequality (x : ℝ) :
  (x - 3) / (x + 7) < 0 ↔ -7 < x ∧ x < 3 := by sorry

-- Part 2
theorem empty_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 - 4*a*x + 4*a^2 + a > 0) ↔ a > 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_empty_solution_set_l2015_201569


namespace NUMINAMATH_CALUDE_special_numbers_count_l2015_201531

def count_multiples (n : ℕ) (max : ℕ) : ℕ :=
  (max / n : ℕ)

def count_special_numbers (max : ℕ) : ℕ :=
  count_multiples 4 max + count_multiples 5 max - count_multiples 20 max - count_multiples 25 max

theorem special_numbers_count :
  count_special_numbers 3000 = 1080 := by sorry

end NUMINAMATH_CALUDE_special_numbers_count_l2015_201531


namespace NUMINAMATH_CALUDE_monika_total_expense_l2015_201554

def mall_expense : ℝ := 250
def movie_cost : ℝ := 24
def movie_count : ℕ := 3
def bean_bag_cost : ℝ := 1.25
def bean_bag_count : ℕ := 20

theorem monika_total_expense : 
  mall_expense + movie_cost * movie_count + bean_bag_cost * bean_bag_count = 347 := by
  sorry

end NUMINAMATH_CALUDE_monika_total_expense_l2015_201554


namespace NUMINAMATH_CALUDE_prime_pairs_divisibility_l2015_201523

theorem prime_pairs_divisibility (p q : ℕ) : 
  Prime p ∧ Prime q ∧ 
  (p^2 ∣ q^3 + 1) ∧ 
  (q^2 ∣ p^6 - 1) ↔ 
  ((p = 3 ∧ q = 2) ∨ (p = 2 ∧ q = 3)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_divisibility_l2015_201523


namespace NUMINAMATH_CALUDE_fraction_enlargement_l2015_201503

theorem fraction_enlargement (x y : ℝ) (h : 3 * x - y ≠ 0) :
  (2 * (3 * x) * (3 * y)) / (3 * (3 * x) - (3 * y)) = 3 * ((2 * x * y) / (3 * x - y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_enlargement_l2015_201503


namespace NUMINAMATH_CALUDE_league_teams_count_l2015_201556

/-- The number of games in a league where each team plays every other team once -/
def numGames (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league where each team plays every other team exactly once, 
    if the total number of games played is 36, then the number of teams in the league is 9 -/
theorem league_teams_count : ∃ (n : ℕ), n > 0 ∧ numGames n = 36 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_league_teams_count_l2015_201556


namespace NUMINAMATH_CALUDE_system_solution_l2015_201527

theorem system_solution :
  ∃ (x y : ℝ), (3 * x - 5 * y = -1.5) ∧ (7 * x + 2 * y = 4.7) ∧ (x = 0.5) ∧ (y = 0.6) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2015_201527


namespace NUMINAMATH_CALUDE_problem_statement_l2015_201560

theorem problem_statement (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a^b = c^d) (h2 : a / (2 * c) = b / d) (h3 : a / (2 * c) = 2) :
  1 / c = 16 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2015_201560


namespace NUMINAMATH_CALUDE_digit_sum_at_positions_l2015_201553

def sequence_generator (n : ℕ) : ℕ :=
  (n - 1) % 6 + 1

def remove_nth (n : ℕ) (seq : ℕ → ℕ) : ℕ → ℕ :=
  Function.comp seq (λ m => m + m / (n - 1))

def final_sequence : ℕ → ℕ :=
  remove_nth 7 (remove_nth 5 sequence_generator)

theorem digit_sum_at_positions : 
  final_sequence 3031 + final_sequence 3032 + final_sequence 3033 = 9 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_at_positions_l2015_201553


namespace NUMINAMATH_CALUDE_zoo_trip_short_amount_l2015_201511

/-- Represents the zoo trip expenses and budget for two people -/
structure ZooTrip where
  total_budget : ℕ
  zoo_entry_cost : ℕ
  aquarium_entry_cost : ℕ
  animal_show_cost : ℕ
  bus_fare : ℕ
  num_transfers : ℕ
  souvenir_budget : ℕ
  noah_lunch_cost : ℕ
  ava_lunch_cost : ℕ
  beverage_cost : ℕ
  num_people : ℕ

/-- Calculates the amount short for lunch and snacks -/
def amount_short (trip : ZooTrip) : ℕ :=
  let total_entry_cost := (trip.zoo_entry_cost + trip.aquarium_entry_cost + trip.animal_show_cost) * trip.num_people
  let total_bus_fare := trip.bus_fare * trip.num_transfers * trip.num_people
  let total_lunch_cost := trip.noah_lunch_cost + trip.ava_lunch_cost
  let total_beverage_cost := trip.beverage_cost * trip.num_people
  let total_expenses := total_entry_cost + total_bus_fare + trip.souvenir_budget + total_lunch_cost + total_beverage_cost
  total_expenses - trip.total_budget

/-- Theorem stating that the amount short for lunch and snacks is $12 -/
theorem zoo_trip_short_amount (trip : ZooTrip) 
  (h1 : trip.total_budget = 100)
  (h2 : trip.zoo_entry_cost = 5)
  (h3 : trip.aquarium_entry_cost = 7)
  (h4 : trip.animal_show_cost = 4)
  (h5 : trip.bus_fare = 150) -- Using cents for precise integer arithmetic
  (h6 : trip.num_transfers = 4)
  (h7 : trip.souvenir_budget = 20)
  (h8 : trip.noah_lunch_cost = 10)
  (h9 : trip.ava_lunch_cost = 8)
  (h10 : trip.beverage_cost = 3)
  (h11 : trip.num_people = 2) :
  amount_short trip = 12 := by
  sorry


end NUMINAMATH_CALUDE_zoo_trip_short_amount_l2015_201511


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2015_201566

/-- Given two vectors a and b in ℝ², where a = (1,2) and b = (2x,-3),
    if a is parallel to b, then x = -3/4 -/
theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2*x, -3)
  (∃ (k : ℝ), k ≠ 0 ∧ a.1 * k = b.1 ∧ a.2 * k = b.2) →
  x = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2015_201566


namespace NUMINAMATH_CALUDE_tangent_sum_equality_l2015_201583

theorem tangent_sum_equality (α β : Real) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2)
  (h_equality : Real.tan (α - β) = Real.sin (2 * β)) :
  Real.tan α + Real.tan β = 2 * Real.tan (2 * β) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_equality_l2015_201583


namespace NUMINAMATH_CALUDE_wire_length_between_poles_l2015_201546

/-- Given two vertical poles on flat ground with a distance of 20 feet between their bases
    and a height difference of 10 feet, the length of a wire stretched between their tops
    is 10√5 feet. -/
theorem wire_length_between_poles (distance : ℝ) (height_diff : ℝ) :
  distance = 20 → height_diff = 10 → 
  ∃ (wire_length : ℝ), wire_length = 10 * Real.sqrt 5 ∧ 
  wire_length ^ 2 = distance ^ 2 + height_diff ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_wire_length_between_poles_l2015_201546


namespace NUMINAMATH_CALUDE_large_block_length_multiple_l2015_201536

/-- Represents the dimensions of a block of cheese -/
structure CheeseDimensions where
  width : ℝ
  depth : ℝ
  length : ℝ

/-- Calculates the volume of a block of cheese given its dimensions -/
def volume (d : CheeseDimensions) : ℝ :=
  d.width * d.depth * d.length

theorem large_block_length_multiple (normal : CheeseDimensions) (large : CheeseDimensions) :
  volume normal = 3 →
  large.width = 2 * normal.width →
  large.depth = 2 * normal.depth →
  volume large = 36 →
  large.length = 3 * normal.length := by
  sorry

#check large_block_length_multiple

end NUMINAMATH_CALUDE_large_block_length_multiple_l2015_201536


namespace NUMINAMATH_CALUDE_milk_exchange_theorem_l2015_201506

/-- Represents the number of liters of milk obtainable from a given number of empty bottles -/
def milk_obtained (empty_bottles : ℕ) : ℕ :=
  let full_bottles := empty_bottles / 4
  let remaining_empty := empty_bottles % 4
  if full_bottles = 0 then
    0
  else
    full_bottles + milk_obtained (full_bottles + remaining_empty)

/-- Theorem stating that 43 empty bottles can be exchanged for 14 liters of milk -/
theorem milk_exchange_theorem :
  milk_obtained 43 = 14 := by
  sorry

end NUMINAMATH_CALUDE_milk_exchange_theorem_l2015_201506


namespace NUMINAMATH_CALUDE_egg_count_theorem_l2015_201592

/-- Represents a carton of eggs -/
structure EggCarton where
  total_yolks : ℕ
  double_yolk_eggs : ℕ

/-- Calculate the number of eggs in a carton -/
def count_eggs (carton : EggCarton) : ℕ :=
  carton.double_yolk_eggs + (carton.total_yolks - 2 * carton.double_yolk_eggs)

/-- Theorem: A carton with 17 yolks and 5 double-yolk eggs contains 12 eggs -/
theorem egg_count_theorem (carton : EggCarton) 
  (h1 : carton.total_yolks = 17) 
  (h2 : carton.double_yolk_eggs = 5) : 
  count_eggs carton = 12 := by
  sorry

#eval count_eggs { total_yolks := 17, double_yolk_eggs := 5 }

end NUMINAMATH_CALUDE_egg_count_theorem_l2015_201592


namespace NUMINAMATH_CALUDE_average_cost_theorem_l2015_201532

def iPhone_quantity : ℕ := 100
def iPhone_price : ℝ := 1000
def iPhone_tax_rate : ℝ := 0.1

def iPad_quantity : ℕ := 20
def iPad_price : ℝ := 900
def iPad_discount_rate : ℝ := 0.05

def AppleTV_quantity : ℕ := 80
def AppleTV_price : ℝ := 200
def AppleTV_tax_rate : ℝ := 0.08

def MacBook_quantity : ℕ := 50
def MacBook_price : ℝ := 1500
def MacBook_discount_rate : ℝ := 0.15

def total_quantity : ℕ := iPhone_quantity + iPad_quantity + AppleTV_quantity + MacBook_quantity

def total_cost : ℝ :=
  iPhone_quantity * iPhone_price * (1 + iPhone_tax_rate) +
  iPad_quantity * iPad_price * (1 - iPad_discount_rate) +
  AppleTV_quantity * AppleTV_price * (1 + AppleTV_tax_rate) +
  MacBook_quantity * MacBook_price * (1 - MacBook_discount_rate)

theorem average_cost_theorem :
  total_cost / total_quantity = 832.52 := by sorry

end NUMINAMATH_CALUDE_average_cost_theorem_l2015_201532


namespace NUMINAMATH_CALUDE_binary_10101_equals_21_l2015_201533

def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_10101_equals_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end NUMINAMATH_CALUDE_binary_10101_equals_21_l2015_201533


namespace NUMINAMATH_CALUDE_three_integer_chords_l2015_201587

/-- Represents a circle with a given radius and a point inside it. -/
structure CircleWithPoint where
  radius : ℝ
  distanceFromCenter : ℝ

/-- Counts the number of chords with integer lengths that contain the given point. -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem stating that for a circle with radius 13 and a point 5 units from the center,
    there are exactly 3 chords with integer lengths containing the point. -/
theorem three_integer_chords :
  let c := CircleWithPoint.mk 13 5
  countIntegerChords c = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_integer_chords_l2015_201587


namespace NUMINAMATH_CALUDE_factor_to_increase_average_l2015_201593

theorem factor_to_increase_average (numbers : Finset ℝ) (factor : ℝ) : 
  Finset.card numbers = 5 →
  6 ∈ numbers →
  (Finset.sum numbers id) / 5 = 6.8 →
  ((Finset.sum numbers id) - 6 + 6 * factor) / 5 = 9.2 →
  factor = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_to_increase_average_l2015_201593


namespace NUMINAMATH_CALUDE_quadratic_root_range_l2015_201517

theorem quadratic_root_range (m : ℝ) : 
  (∃ x y : ℝ, x < 1 ∧ y > 1 ∧ 
   x^2 + (m-1)*x + m^2 - 2 = 0 ∧
   y^2 + (m-1)*y + m^2 - 2 = 0) →
  -2 < m ∧ m < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l2015_201517


namespace NUMINAMATH_CALUDE_reciprocal_problem_l2015_201505

theorem reciprocal_problem (x : ℚ) : 7 * x = 3 → 70 * (1 / x) = 490 / 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l2015_201505


namespace NUMINAMATH_CALUDE_cube_root_rationality_l2015_201582

theorem cube_root_rationality (a b : ℚ) (ha : 0 < a) (hb : 0 < b) 
  (h : ∃ (s : ℚ), s = (a^(1/3) + b^(1/3))) : 
  ∃ (r₁ r₂ : ℚ), r₁ = a^(1/3) ∧ r₂ = b^(1/3) := by
sorry

end NUMINAMATH_CALUDE_cube_root_rationality_l2015_201582


namespace NUMINAMATH_CALUDE_not_prime_sum_product_l2015_201501

theorem not_prime_sum_product (a b c d : ℕ) 
  (h_pos : 0 < d ∧ d < c ∧ c < b ∧ b < a) 
  (h_eq : a * c + b * d = (b + d - a + c) * (b + d + a - c)) : 
  ¬ Nat.Prime (a * b + c * d) := by
sorry

end NUMINAMATH_CALUDE_not_prime_sum_product_l2015_201501


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2015_201581

theorem smallest_n_congruence (n : ℕ) : 
  (∀ k < n, ¬(5^k ≡ k^5 [ZMOD 3])) ∧ (5^n ≡ n^5 [ZMOD 3]) ↔ n = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2015_201581


namespace NUMINAMATH_CALUDE_laundry_loads_required_l2015_201530

def num_families : ℕ := 3
def people_per_family : ℕ := 4
def vacation_days : ℕ := 7
def towels_per_person_per_day : ℕ := 1
def washing_machine_capacity : ℕ := 14

def total_people : ℕ := num_families * people_per_family
def total_towels : ℕ := total_people * vacation_days * towels_per_person_per_day

theorem laundry_loads_required :
  (total_towels + washing_machine_capacity - 1) / washing_machine_capacity = 6 := by
  sorry

end NUMINAMATH_CALUDE_laundry_loads_required_l2015_201530


namespace NUMINAMATH_CALUDE_orthogonal_vectors_m_l2015_201562

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, -1)

theorem orthogonal_vectors_m (m : ℝ) : 
  (a.1 + m * b.1, a.2 + m * b.2) • (a.1 - b.1, a.2 - b.2) = 0 → m = 23 / 3 :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_m_l2015_201562


namespace NUMINAMATH_CALUDE_solid_is_cone_l2015_201557

-- Define the properties of the solid
structure Solid where
  front_view_isosceles : Bool
  left_view_isosceles : Bool
  top_view_circle_with_center : Bool

-- Define what it means for a solid to be a cone
def is_cone (s : Solid) : Prop :=
  s.front_view_isosceles ∧ s.left_view_isosceles ∧ s.top_view_circle_with_center

-- Theorem statement
theorem solid_is_cone (s : Solid) 
  (h1 : s.front_view_isosceles = true) 
  (h2 : s.left_view_isosceles = true) 
  (h3 : s.top_view_circle_with_center = true) : 
  is_cone s := by sorry

end NUMINAMATH_CALUDE_solid_is_cone_l2015_201557


namespace NUMINAMATH_CALUDE_expansion_equality_constant_term_proof_l2015_201543

/-- The constant term in the expansion of (1/x^2 + 4x^2 + 4)^3 -/
def constantTerm : ℕ := 160

/-- The original expression (1/x^2 + 4x^2 + 4)^3 can be rewritten as (2x + 1/x)^6 -/
theorem expansion_equality (x : ℝ) (hx : x ≠ 0) :
  (1 / x^2 + 4 * x^2 + 4)^3 = (2 * x + 1 / x)^6 := by sorry

/-- The constant term in the expansion of (1/x^2 + 4x^2 + 4)^3 is equal to constantTerm -/
theorem constant_term_proof :
  constantTerm = 160 := by sorry

end NUMINAMATH_CALUDE_expansion_equality_constant_term_proof_l2015_201543


namespace NUMINAMATH_CALUDE_linear_equation_solution_l2015_201547

theorem linear_equation_solution (a : ℝ) :
  (∀ x, ax^2 + 5*x + 14 = 2*x^2 - 2*x + 3*a → x = -8/7) ∧
  (∃ m b, ∀ x, ax^2 + 5*x + 14 = 2*x^2 - 2*x + 3*a ↔ m*x + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l2015_201547


namespace NUMINAMATH_CALUDE_student_arrangement_theorem_l2015_201590

/-- The number of ways to arrange n students in a row -/
def arrange (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n students in a row with 2 specific students not at the ends -/
def arrangeNotAtEnds (n : ℕ) : ℕ :=
  arrange (n - 2) * (arrange (n - 3))

/-- The number of ways to arrange n students in a row with 2 specific students adjacent -/
def arrangeAdjacent (n : ℕ) : ℕ :=
  2 * arrange (n - 1)

/-- The number of ways to arrange n students in a row with 2 specific students not adjacent -/
def arrangeNotAdjacent (n : ℕ) : ℕ :=
  arrange n - arrangeAdjacent n

/-- The main theorem -/
theorem student_arrangement_theorem :
  (arrangeNotAtEnds 5 = 36) ∧
  (arrangeAdjacent 5 * arrangeNotAdjacent 3 = 24) :=
by sorry

end NUMINAMATH_CALUDE_student_arrangement_theorem_l2015_201590


namespace NUMINAMATH_CALUDE_roberto_outfits_l2015_201538

/-- The number of trousers Roberto has -/
def num_trousers : ℕ := 5

/-- The number of shirts Roberto has -/
def num_shirts : ℕ := 6

/-- The number of jackets Roberto has -/
def num_jackets : ℕ := 4

/-- The number of restricted outfits (combinations of the specific shirt and jacket that can't be worn together) -/
def num_restricted : ℕ := 1 * 1 * num_trousers

/-- The total number of possible outfits without restrictions -/
def total_outfits : ℕ := num_trousers * num_shirts * num_jackets

/-- The number of permissible outfits Roberto can put together -/
def permissible_outfits : ℕ := total_outfits - num_restricted

theorem roberto_outfits : permissible_outfits = 115 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l2015_201538


namespace NUMINAMATH_CALUDE_certain_number_proof_l2015_201598

theorem certain_number_proof (x : ℝ) : 
  (0.15 * x > 0.25 * 16 + 2) → (0.15 * x = 6) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2015_201598


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2015_201513

/-- Represents a repeating decimal with a single digit repeating -/
def SingleDigitRepeatingDecimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + repeating / 9

/-- Represents a repeating decimal with two digits repeating -/
def TwoDigitRepeatingDecimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + repeating / 99

theorem repeating_decimal_sum :
  SingleDigitRepeatingDecimal 0 3 + TwoDigitRepeatingDecimal 0 6 = 13 / 33 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2015_201513


namespace NUMINAMATH_CALUDE_friends_recycled_23_pounds_l2015_201539

/-- Represents the recycling scenario with Zoe and her friends -/
structure RecyclingScenario where
  pointsPerEightPounds : Nat
  zoeRecycled : Nat
  totalPoints : Nat

/-- Calculates the number of pounds Zoe's friends recycled -/
def friendsRecycled (scenario : RecyclingScenario) : Nat :=
  scenario.totalPoints * 8 - scenario.zoeRecycled

/-- Theorem stating that Zoe's friends recycled 23 pounds -/
theorem friends_recycled_23_pounds (scenario : RecyclingScenario)
  (h1 : scenario.pointsPerEightPounds = 1)
  (h2 : scenario.zoeRecycled = 25)
  (h3 : scenario.totalPoints = 6) :
  friendsRecycled scenario = 23 := by
  sorry

#eval friendsRecycled ⟨1, 25, 6⟩

end NUMINAMATH_CALUDE_friends_recycled_23_pounds_l2015_201539


namespace NUMINAMATH_CALUDE_sum_of_cubes_remainder_l2015_201578

def sum_of_cubes (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

def b : ℕ := 2

theorem sum_of_cubes_remainder (n : ℕ) (h : n = 2010) : 
  sum_of_cubes n % (b ^ 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_remainder_l2015_201578


namespace NUMINAMATH_CALUDE_number_problem_l2015_201577

theorem number_problem : ∃ x : ℝ, x > 0 ∧ 0.9 * x = (4/5 * 25) + 16 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2015_201577


namespace NUMINAMATH_CALUDE_dividend_calculation_l2015_201576

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 10 * quotient)
  (h2 : divisor = 5 * remainder)
  (h3 : remainder = 46) :
  divisor * quotient + remainder = 5336 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2015_201576


namespace NUMINAMATH_CALUDE_max_b_in_box_l2015_201572

theorem max_b_in_box (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (∀ a' b' c' : ℕ, (a' * b' * c' = 360) → (1 < c') → (c' < b') → (b' < a') → b' ≤ b) →
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_max_b_in_box_l2015_201572


namespace NUMINAMATH_CALUDE_scale_tower_height_l2015_201549

/-- Given a cylindrical tower and its scaled-down model, calculates the height of the model. -/
theorem scale_tower_height (actual_height : ℝ) (actual_volume : ℝ) (model_volume : ℝ) 
  (h1 : actual_height = 60) 
  (h2 : actual_volume = 80000)
  (h3 : model_volume = 0.5) :
  actual_height / Real.sqrt (actual_volume / model_volume) = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_scale_tower_height_l2015_201549


namespace NUMINAMATH_CALUDE_rectangle_midpoint_distances_l2015_201584

theorem rectangle_midpoint_distances (a b : ℝ) (ha : a = 3) (hb : b = 4) :
  let midpoint_distance (x y : ℝ) := Real.sqrt (x^2 + y^2)
  (midpoint_distance (a/2) 0) + (midpoint_distance a (b/2)) +
  (midpoint_distance (a/2) b) + (midpoint_distance 0 (b/2)) =
  3.5 + Real.sqrt 13 + Real.sqrt 18.25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_midpoint_distances_l2015_201584


namespace NUMINAMATH_CALUDE_total_problems_solved_l2015_201559

/-- The number of problems Seokjin initially solved -/
def initial_problems : ℕ := 12

/-- The number of additional problems Seokjin solved -/
def additional_problems : ℕ := 7

/-- Theorem: The total number of problems Seokjin solved is 19 -/
theorem total_problems_solved : initial_problems + additional_problems = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_problems_solved_l2015_201559


namespace NUMINAMATH_CALUDE_fruit_store_problem_l2015_201521

/-- The number of watermelons in a fruit store. -/
def num_watermelons : ℕ := by sorry

theorem fruit_store_problem :
  let apples : ℕ := 82
  let pears : ℕ := 90
  let tangerines : ℕ := 88
  let melons : ℕ := 84
  let total_fruits : ℕ := apples + pears + tangerines + melons + num_watermelons
  (total_fruits % 88 = 0) ∧ (total_fruits / 88 = 5) →
  num_watermelons = 96 := by sorry

end NUMINAMATH_CALUDE_fruit_store_problem_l2015_201521


namespace NUMINAMATH_CALUDE_circular_road_width_l2015_201589

theorem circular_road_width (r R : ℝ) (h1 : r = R / 3) (h2 : 2 * π * r + 2 * π * R = 88) :
  R - r = 22 / π := by
  sorry

end NUMINAMATH_CALUDE_circular_road_width_l2015_201589


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l2015_201552

theorem max_value_trig_expression (a b φ : ℝ) :
  (∀ θ : ℝ, a * Real.cos (θ + φ) + b * Real.sin (θ + φ) ≤ Real.sqrt (a^2 + b^2)) ∧
  (∃ θ : ℝ, a * Real.cos (θ + φ) + b * Real.sin (θ + φ) = Real.sqrt (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l2015_201552


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_l2015_201551

/-- The equation represents two lines -/
theorem equation_represents_two_lines :
  ∃ (a b c d : ℝ), a ≠ c ∧ b ≠ d ∧
  ∀ (x y : ℝ), x^2 - 50*y^2 - 10*x + 25 = 0 ↔ 
  ((x = a*y + b) ∨ (x = c*y + d)) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_l2015_201551


namespace NUMINAMATH_CALUDE_second_meeting_day_correct_l2015_201585

/-- Represents the number of days between visits for each schoolchild -/
def VisitSchedule : Fin 4 → ℕ
  | 0 => 4
  | 1 => 5
  | 2 => 6
  | 3 => 9

/-- The day when all schoolchildren meet for the second time -/
def SecondMeetingDay : ℕ := 360

theorem second_meeting_day_correct :
  SecondMeetingDay = 2 * Nat.lcm (VisitSchedule 0) (Nat.lcm (VisitSchedule 1) (Nat.lcm (VisitSchedule 2) (VisitSchedule 3))) :=
by sorry

#check second_meeting_day_correct

end NUMINAMATH_CALUDE_second_meeting_day_correct_l2015_201585


namespace NUMINAMATH_CALUDE_last_two_digits_of_sum_of_factorials_15_l2015_201518

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_of_sum_of_factorials_15 :
  last_two_digits (sum_of_factorials 15) = 13 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_sum_of_factorials_15_l2015_201518


namespace NUMINAMATH_CALUDE_triangle_properties_l2015_201565

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : a = 5) 
  (h3 : c = 6) 
  (h4 : Real.sin B = 3/5) :
  b = Real.sqrt 13 ∧ 
  Real.sin A = (3 * Real.sqrt 13) / 13 ∧ 
  Real.sin (2 * A + π/4) = (7 * Real.sqrt 2) / 26 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2015_201565


namespace NUMINAMATH_CALUDE_nested_square_root_value_l2015_201524

theorem nested_square_root_value :
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l2015_201524


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l2015_201588

/-- Represents different sampling methods -/
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

/-- Represents a population with different income levels -/
structure Population :=
  (total : ℕ)
  (high_income : ℕ)
  (middle_income : ℕ)
  (low_income : ℕ)

/-- Represents a sampling problem -/
structure SamplingProblem :=
  (population : Population)
  (sample_size : ℕ)

/-- Determines the best sampling method for a given problem -/
def best_sampling_method (problem : SamplingProblem) : SamplingMethod :=
  sorry

/-- The community population for problem 1 -/
def community : Population :=
  { total := 600
  , high_income := 100
  , middle_income := 380
  , low_income := 120 }

/-- Problem 1: Family income study -/
def problem1 : SamplingProblem :=
  { population := community
  , sample_size := 100 }

/-- Problem 2: Student seminar selection -/
def problem2 : SamplingProblem :=
  { population := { total := 15, high_income := 0, middle_income := 0, low_income := 0 }
  , sample_size := 3 }

theorem correct_sampling_methods :
  (best_sampling_method problem1 = SamplingMethod.Stratified) ∧
  (best_sampling_method problem2 = SamplingMethod.SimpleRandom) :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l2015_201588


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l2015_201571

/-- The number of pairs of shoes in the box -/
def num_pairs : ℕ := 7

/-- The total number of shoes in the box -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of ways to select two shoes from the box -/
def total_combinations : ℕ := (total_shoes * (total_shoes - 1)) / 2

/-- The number of ways to select a matching pair of shoes -/
def matching_pairs : ℕ := num_pairs

/-- The probability of selecting a matching pair of shoes -/
def probability : ℚ := matching_pairs / total_combinations

theorem matching_shoes_probability :
  probability = 1 / 13 := by sorry

end NUMINAMATH_CALUDE_matching_shoes_probability_l2015_201571


namespace NUMINAMATH_CALUDE_cat_shortest_distance_to_origin_l2015_201534

theorem cat_shortest_distance_to_origin :
  let center : ℝ × ℝ := (5, -2)
  let radius : ℝ := 8
  let origin : ℝ × ℝ := (0, 0)
  let distance_center_to_origin : ℝ := Real.sqrt ((center.1 - origin.1)^2 + (center.2 - origin.2)^2)
  ∀ p : ℝ × ℝ, (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 →
    Real.sqrt ((p.1 - origin.1)^2 + (p.2 - origin.2)^2) ≥ |distance_center_to_origin - radius| :=
by sorry

end NUMINAMATH_CALUDE_cat_shortest_distance_to_origin_l2015_201534


namespace NUMINAMATH_CALUDE_sequence_theorem_l2015_201529

/-- A positive sequence satisfying the given condition -/
def PositiveSequence (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n > 0

/-- The sum of the first n terms of the sequence -/
def S (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  (Finset.range n).sum (fun i => a ⟨i + 1, Nat.succ_pos i⟩)

/-- The main theorem -/
theorem sequence_theorem (a : ℕ+ → ℝ) (h_pos : PositiveSequence a)
    (h_cond : ∀ n : ℕ+, 2 * S a n = a n ^ 2 + a n) :
    ∀ n : ℕ+, a n = n := by
  sorry

end NUMINAMATH_CALUDE_sequence_theorem_l2015_201529


namespace NUMINAMATH_CALUDE_complex_ratio_l2015_201568

theorem complex_ratio (a b : ℝ) (h1 : a * b ≠ 0) :
  let z : ℂ := Complex.mk a b
  (∃ (k : ℝ), z * Complex.mk 1 (-2) = k) → a / b = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_ratio_l2015_201568


namespace NUMINAMATH_CALUDE_simplify_sqrt_plus_x_l2015_201514

theorem simplify_sqrt_plus_x (x : ℝ) (h : 1 < x ∧ x < 2) : 
  Real.sqrt ((x - 2)^2) + x = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_plus_x_l2015_201514


namespace NUMINAMATH_CALUDE_journey_speed_l2015_201597

/-- Proves that given a journey where 75% is traveled at 50 mph and 25% at S mph,
    if the average speed for the entire journey is 50 mph, then S must equal 50 mph. -/
theorem journey_speed (D : ℝ) (S : ℝ) (h1 : D > 0) :
  (D / ((0.75 * D / 50) + (0.25 * D / S)) = 50) → S = 50 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_l2015_201597


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2015_201599

def f (x : ℝ) : ℝ := 6*x^3 - 15*x^2 + 21*x - 23

theorem polynomial_remainder_theorem :
  ∃ (q : ℝ → ℝ), f = λ x => (3*x - 6) * q x + 7 :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2015_201599


namespace NUMINAMATH_CALUDE_correct_outfit_assignment_l2015_201555

-- Define the colors
inductive Color
  | White
  | Red
  | Blue

-- Define a person's outfit
structure Outfit :=
  (dress : Color)
  (shoes : Color)

-- Define the friends
inductive Friend
  | Nadya
  | Valya
  | Masha

def outfit_assignment : Friend → Outfit
  | Friend.Nadya => { dress := Color.Blue, shoes := Color.Blue }
  | Friend.Valya => { dress := Color.Red, shoes := Color.White }
  | Friend.Masha => { dress := Color.White, shoes := Color.Red }

theorem correct_outfit_assignment :
  -- Nadya's shoes match her dress
  (outfit_assignment Friend.Nadya).dress = (outfit_assignment Friend.Nadya).shoes ∧
  -- Valya's dress and shoes are not blue
  (outfit_assignment Friend.Valya).dress ≠ Color.Blue ∧
  (outfit_assignment Friend.Valya).shoes ≠ Color.Blue ∧
  -- Masha wears red shoes
  (outfit_assignment Friend.Masha).shoes = Color.Red ∧
  -- All dresses are different colors
  (outfit_assignment Friend.Nadya).dress ≠ (outfit_assignment Friend.Valya).dress ∧
  (outfit_assignment Friend.Nadya).dress ≠ (outfit_assignment Friend.Masha).dress ∧
  (outfit_assignment Friend.Valya).dress ≠ (outfit_assignment Friend.Masha).dress ∧
  -- All shoes are different colors
  (outfit_assignment Friend.Nadya).shoes ≠ (outfit_assignment Friend.Valya).shoes ∧
  (outfit_assignment Friend.Nadya).shoes ≠ (outfit_assignment Friend.Masha).shoes ∧
  (outfit_assignment Friend.Valya).shoes ≠ (outfit_assignment Friend.Masha).shoes := by
  sorry

end NUMINAMATH_CALUDE_correct_outfit_assignment_l2015_201555


namespace NUMINAMATH_CALUDE_log_inequality_condition_l2015_201535

theorem log_inequality_condition (a b : ℝ) : 
  (∀ a b, Real.log a > Real.log b → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(Real.log a > Real.log b)) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_condition_l2015_201535


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l2015_201504

/-- The number of ways to choose 7 starters from a volleyball team -/
def volleyball_starters_count : ℕ := 2376

/-- The total number of players in the team -/
def total_players : ℕ := 15

/-- The number of triplets in the team -/
def triplets_count : ℕ := 3

/-- The number of starters to be chosen -/
def starters_count : ℕ := 7

/-- The number of triplets that must be in the starting lineup -/
def required_triplets : ℕ := 2

theorem volleyball_team_selection :
  volleyball_starters_count = 
    (Nat.choose triplets_count required_triplets) * 
    (Nat.choose (total_players - triplets_count) (starters_count - required_triplets)) := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l2015_201504


namespace NUMINAMATH_CALUDE_expenditure_representation_l2015_201508

/-- Represents a monetary transaction in yuan -/
structure Transaction where
  amount : Int
  deriving Repr

/-- Defines an income transaction -/
def is_income (t : Transaction) : Prop := t.amount > 0

/-- Defines an expenditure transaction -/
def is_expenditure (t : Transaction) : Prop := t.amount < 0

/-- Theorem stating that an expenditure of 50 yuan should be represented as -50 yuan -/
theorem expenditure_representation :
  ∀ (t : Transaction),
    is_expenditure t → t.amount = 50 → t.amount = -50 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_representation_l2015_201508


namespace NUMINAMATH_CALUDE_expression_simplification_l2015_201580

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 2) (h3 : a ≠ -2) :
  ((a / (a - 2) - a / (a^2 - 2*a)) / (a + 2) * a) = (a^2 - a) / (a^2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2015_201580


namespace NUMINAMATH_CALUDE_percentage_decrease_l2015_201574

theorem percentage_decrease (x y z : ℝ) 
  (h1 : x = 1.2 * y) 
  (h2 : x = 0.6 * z) : 
  y = 0.5 * z := by
  sorry

end NUMINAMATH_CALUDE_percentage_decrease_l2015_201574


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l2015_201561

/-- The range of 'a' for which the ellipse x^2 + 4(y-a)^2 = 4 intersects with the parabola x^2 = 2y -/
theorem ellipse_parabola_intersection_range :
  ∀ (a : ℝ),
  (∃ (x y : ℝ), x^2 + 4*(y-a)^2 = 4 ∧ x^2 = 2*y) →
  -1 ≤ a ∧ a ≤ 17/8 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l2015_201561


namespace NUMINAMATH_CALUDE_sin_810_degrees_equals_one_l2015_201548

theorem sin_810_degrees_equals_one : Real.sin (810 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_810_degrees_equals_one_l2015_201548


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2015_201575

/-- The sum of 0.222... and 0.0202... equals 8/33 -/
theorem repeating_decimal_sum : 
  let a : ℚ := 2/9  -- represents 0.222...
  let b : ℚ := 2/99 -- represents 0.0202...
  a + b = 8/33 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2015_201575


namespace NUMINAMATH_CALUDE_ellipse_focal_property_l2015_201507

/-- A point on an ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / 16 + y^2 / 4 = 1

/-- The distance from a point to a focus -/
def distance_to_focus (P : PointOnEllipse) (F : ℝ × ℝ) : ℝ := sorry

/-- The foci of the ellipse -/
def foci : (ℝ × ℝ) × (ℝ × ℝ) := sorry

theorem ellipse_focal_property (P : PointOnEllipse) :
  distance_to_focus P (foci.1) = 3 →
  distance_to_focus P (foci.2) = 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_property_l2015_201507


namespace NUMINAMATH_CALUDE_paper_piles_problem_l2015_201567

theorem paper_piles_problem :
  ∃! N : ℕ,
    1000 < N ∧ N < 2000 ∧
    N % 2 = 1 ∧
    N % 3 = 1 ∧
    N % 4 = 1 ∧
    N % 5 = 1 ∧
    N % 6 = 1 ∧
    N % 7 = 1 ∧
    N % 8 = 1 ∧
    N % 41 = 0 :=
by sorry

end NUMINAMATH_CALUDE_paper_piles_problem_l2015_201567


namespace NUMINAMATH_CALUDE_airport_distance_l2015_201520

/-- Represents the problem of calculating the distance to the airport --/
def airport_distance_problem (initial_speed : ℝ) (speed_increase : ℝ) (late_time : ℝ) : Prop :=
  ∃ (distance : ℝ) (initial_time : ℝ),
    -- If he continued at initial speed, he'd be 1 hour late
    distance = initial_speed * (initial_time + 1) ∧
    -- The remaining distance at increased speed
    (distance - initial_speed) = (initial_speed + speed_increase) * (initial_time - late_time) ∧
    -- The total distance is 70 miles
    distance = 70

/-- The theorem stating that the airport is 70 miles away --/
theorem airport_distance :
  airport_distance_problem 40 20 (1/4) :=
sorry


end NUMINAMATH_CALUDE_airport_distance_l2015_201520


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2015_201510

/-- The function f(x) = α^(x-2) - 1 always passes through the point (2, 0) for any α > 0 and α ≠ 1 -/
theorem fixed_point_of_exponential_function (α : ℝ) (h1 : α > 0) (h2 : α ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ α^(x - 2) - 1
  f 2 = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2015_201510


namespace NUMINAMATH_CALUDE_necklace_length_theorem_l2015_201595

/-- The total length of a necklace made of overlapping paper pieces -/
def necklaceLength (n : ℕ) (pieceLength : ℝ) (overlap : ℝ) : ℝ :=
  n * (pieceLength - overlap)

/-- Theorem: The total length of a necklace made of 16 pieces of colored paper,
    each 10.4 cm long and overlapping by 3.5 cm, is equal to 110.4 cm -/
theorem necklace_length_theorem :
  necklaceLength 16 10.4 3.5 = 110.4 := by
  sorry

end NUMINAMATH_CALUDE_necklace_length_theorem_l2015_201595


namespace NUMINAMATH_CALUDE_min_value_of_sum_l2015_201596

theorem min_value_of_sum (x y z : ℝ) (h : x^2 + 2*y^2 + 5*z^2 = 22) :
  ∃ (m : ℝ), m = xy - yz - zx ∧ m ≥ (-55 - 11*Real.sqrt 5) / 10 ∧
  (∃ (x' y' z' : ℝ), x'^2 + 2*y'^2 + 5*z'^2 = 22 ∧
    x'*y' - y'*z' - z'*x' = (-55 - 11*Real.sqrt 5) / 10) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l2015_201596


namespace NUMINAMATH_CALUDE_angle_value_l2015_201558

theorem angle_value (PQR : ℝ) (x : ℝ) : 
  PQR = 90 → 2*x + x = PQR → x = 30 := by sorry

end NUMINAMATH_CALUDE_angle_value_l2015_201558


namespace NUMINAMATH_CALUDE_right_triangle_angles_l2015_201541

theorem right_triangle_angles (α β : Real) : 
  α > 0 → β > 0 → α + β = π / 2 →
  Real.tan α + Real.tan β + (Real.tan α)^2 + (Real.tan β)^2 + (Real.tan α)^3 + (Real.tan β)^3 = 70 →
  α = π / 2.4 ∧ β = π / 12 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_angles_l2015_201541


namespace NUMINAMATH_CALUDE_gcd_repeated_digits_l2015_201544

def is_repeated_digit (n : ℕ) : Prop :=
  ∃ (m : ℕ), 100 ≤ m ∧ m < 1000 ∧ n = 1001 * m

theorem gcd_repeated_digits :
  ∃ (d : ℕ), d > 0 ∧ 
  (∀ (n : ℕ), is_repeated_digit n → d ∣ n) ∧
  (∀ (k : ℕ), k > 0 → (∀ (n : ℕ), is_repeated_digit n → k ∣ n) → k ∣ d) :=
sorry

end NUMINAMATH_CALUDE_gcd_repeated_digits_l2015_201544


namespace NUMINAMATH_CALUDE_fencing_required_l2015_201579

theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : 
  area = 560 ∧ uncovered_side = 20 → 
  ∃ (width : ℝ), 
    area = uncovered_side * width ∧ 
    2 * width + uncovered_side = 76 := by
  sorry

end NUMINAMATH_CALUDE_fencing_required_l2015_201579


namespace NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l2015_201540

theorem cos_sum_of_complex_exponentials (θ φ : ℝ) :
  Complex.exp (θ * I) = 4/5 + 3/5 * I →
  Complex.exp (φ * I) = -5/13 + 12/13 * I →
  Real.cos (θ + φ) = -1/13 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l2015_201540


namespace NUMINAMATH_CALUDE_sqrt_12_plus_3_minus_pi_pow_0_plus_abs_1_minus_sqrt_3_equals_3_sqrt_3_l2015_201502

theorem sqrt_12_plus_3_minus_pi_pow_0_plus_abs_1_minus_sqrt_3_equals_3_sqrt_3 :
  Real.sqrt 12 + (3 - Real.pi) ^ (0 : ℕ) + |1 - Real.sqrt 3| = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_plus_3_minus_pi_pow_0_plus_abs_1_minus_sqrt_3_equals_3_sqrt_3_l2015_201502


namespace NUMINAMATH_CALUDE_equation_solution_l2015_201570

theorem equation_solution : 
  ∃ y : ℚ, y + 2/3 = 1/4 - 2/5 * 2 ∧ y = -511/420 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2015_201570


namespace NUMINAMATH_CALUDE_largest_number_l2015_201515

-- Define the numbers as real numbers
def A : ℝ := 8.03456
def B : ℝ := 8.034666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666
def C : ℝ := 8.034545454545454545454545454545454545454545454545454545454545454545454545454545454545454545454545454545
def D : ℝ := 8.034563456345634563456345634563456345634563456345634563456345634563456345634563456345634563456345634563456
def E : ℝ := 8.034560345603456034560345603456034560345603456034560345603456034560345603456034560345603456034560345603456

-- Theorem statement
theorem largest_number : B > A ∧ B > C ∧ B > D ∧ B > E := by sorry

end NUMINAMATH_CALUDE_largest_number_l2015_201515


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l2015_201525

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_transitivity 
  (m n : Line) (α β : Plane) 
  (h1 : perp m β) 
  (h2 : perp n β) 
  (h3 : perp n α) : 
  perp m α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l2015_201525


namespace NUMINAMATH_CALUDE_replacement_solution_concentration_l2015_201542

/-- Given an 80% chemical solution, if 50% of it is replaced with a solution
    of unknown concentration P%, resulting in a 50% chemical solution,
    then P% must be 20%. -/
theorem replacement_solution_concentration
  (original_concentration : ℝ)
  (replaced_fraction : ℝ)
  (final_concentration : ℝ)
  (replacement_concentration : ℝ)
  (h1 : original_concentration = 0.8)
  (h2 : replaced_fraction = 0.5)
  (h3 : final_concentration = 0.5)
  (h4 : final_concentration = (1 - replaced_fraction) * original_concentration
                            + replaced_fraction * replacement_concentration) :
  replacement_concentration = 0.2 := by
sorry

end NUMINAMATH_CALUDE_replacement_solution_concentration_l2015_201542


namespace NUMINAMATH_CALUDE_sandy_puppies_l2015_201573

def puppies_problem (initial_puppies : ℕ) (initial_spotted : ℕ) (new_puppies : ℕ) (new_spotted : ℕ) (given_away : ℕ) : Prop :=
  let initial_non_spotted := initial_puppies - initial_spotted
  let total_spotted := initial_spotted + new_spotted
  let total_non_spotted := initial_non_spotted + (new_puppies - new_spotted) - given_away
  let final_puppies := total_spotted + total_non_spotted
  final_puppies = 9

theorem sandy_puppies : puppies_problem 8 3 4 2 3 :=
by sorry

end NUMINAMATH_CALUDE_sandy_puppies_l2015_201573


namespace NUMINAMATH_CALUDE_mary_next_birthday_age_l2015_201500

theorem mary_next_birthday_age (m s d t : ℝ) : 
  m = 1.25 * s →
  s = 0.7 * d →
  t = 2 * s →
  m + s + d + t = 38 →
  ⌊m⌋ + 1 = 9 :=
sorry

end NUMINAMATH_CALUDE_mary_next_birthday_age_l2015_201500


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_l2015_201563

theorem mod_equivalence_unique : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 5 ∧ n ≡ -1723 [ZMOD 6] := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_l2015_201563


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l2015_201516

/-- A geometric sequence of positive integers -/
def GeometricSequence (a : ℕ+) (r : ℕ+) : ℕ → ℕ+
  | 0 => a
  | n + 1 => r * GeometricSequence a r n

theorem sixth_term_of_geometric_sequence 
  (a : ℕ+) (r : ℕ+) :
  a = 3 →
  GeometricSequence a r 4 = 243 →
  GeometricSequence a r 5 = 729 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l2015_201516


namespace NUMINAMATH_CALUDE_lent_amount_proof_l2015_201586

/-- The amount of money (in Rs.) that A lends to B -/
def lent_amount : ℝ := 1500

/-- The interest rate difference (in decimal form) between B's lending and borrowing rates -/
def interest_rate_diff : ℝ := 0.015

/-- The number of years for which the loan is considered -/
def years : ℝ := 3

/-- B's total gain (in Rs.) over the loan period -/
def total_gain : ℝ := 67.5

theorem lent_amount_proof :
  lent_amount * interest_rate_diff * years = total_gain :=
by sorry

end NUMINAMATH_CALUDE_lent_amount_proof_l2015_201586


namespace NUMINAMATH_CALUDE_mean_home_runs_l2015_201509

def num_players : List ℕ := [7, 5, 4, 2, 1]
def home_runs : List ℕ := [5, 6, 8, 9, 11]

theorem mean_home_runs : 
  (List.sum (List.zipWith (· * ·) num_players home_runs)) / (List.sum num_players) = 126 / 19 := by
  sorry

end NUMINAMATH_CALUDE_mean_home_runs_l2015_201509
