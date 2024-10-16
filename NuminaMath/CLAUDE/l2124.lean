import Mathlib

namespace NUMINAMATH_CALUDE_journey_distance_l2124_212453

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 10)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  ∃ (distance : ℝ), 
    distance = total_time * (speed1 + speed2) / 2 ∧ 
    distance = 224 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l2124_212453


namespace NUMINAMATH_CALUDE_liz_total_spent_l2124_212422

/-- The total amount spent by Liz on her baking purchases -/
def total_spent (recipe_book_cost : ℕ) (ingredient_cost : ℕ) (num_ingredients : ℕ) : ℕ :=
  let baking_dish_cost := 2 * recipe_book_cost
  let ingredients_total_cost := ingredient_cost * num_ingredients
  let apron_cost := recipe_book_cost + 1
  recipe_book_cost + baking_dish_cost + ingredients_total_cost + apron_cost

/-- Theorem stating that Liz spent $40 in total -/
theorem liz_total_spent : total_spent 6 3 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_liz_total_spent_l2124_212422


namespace NUMINAMATH_CALUDE_wire_length_proof_l2124_212440

theorem wire_length_proof (shorter_piece : ℝ) (longer_piece : ℝ) : 
  shorter_piece = 14.285714285714285 →
  shorter_piece = (2/5) * longer_piece →
  shorter_piece + longer_piece = 50 := by
sorry

end NUMINAMATH_CALUDE_wire_length_proof_l2124_212440


namespace NUMINAMATH_CALUDE_final_cost_calculation_l2124_212497

def washing_machine_cost : ℝ := 100
def dryer_cost : ℝ := washing_machine_cost - 30
def discount_rate : ℝ := 0.1

theorem final_cost_calculation :
  let total_cost : ℝ := washing_machine_cost + dryer_cost
  let discount_amount : ℝ := total_cost * discount_rate
  let final_cost : ℝ := total_cost - discount_amount
  final_cost = 153 := by sorry

end NUMINAMATH_CALUDE_final_cost_calculation_l2124_212497


namespace NUMINAMATH_CALUDE_pasture_rent_problem_l2124_212441

/-- Represents the number of oxen each person puts in the pasture -/
structure OxenCount where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the number of months each person's oxen graze -/
structure GrazingMonths where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total oxen-months for all three people -/
def totalOxenMonths (oxen : OxenCount) (months : GrazingMonths) : ℕ :=
  oxen.a * months.a + oxen.b * months.b + oxen.c * months.c

/-- Calculates a person's share of the rent based on their oxen-months -/
def rentShare (totalRent : ℚ) (oxenMonths : ℕ) (totalOxenMonths : ℕ) : ℚ :=
  totalRent * (oxenMonths : ℚ) / (totalOxenMonths : ℚ)

theorem pasture_rent_problem (totalRent : ℚ) (oxen : OxenCount) (months : GrazingMonths) 
    (h1 : totalRent = 175)
    (h2 : oxen.a = 10 ∧ oxen.b = 12 ∧ oxen.c = 15)
    (h3 : months.a = 7 ∧ months.b = 5)
    (h4 : rentShare totalRent (oxen.c * months.c) (totalOxenMonths oxen months) = 45) :
    months.c = 3 := by
  sorry

end NUMINAMATH_CALUDE_pasture_rent_problem_l2124_212441


namespace NUMINAMATH_CALUDE_largest_n_binomial_equality_l2124_212424

theorem largest_n_binomial_equality : 
  (∃ n : ℕ, (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n)) ∧
  (∀ m : ℕ, m > 7 → Nat.choose 10 3 + Nat.choose 10 4 ≠ Nat.choose 11 m) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_equality_l2124_212424


namespace NUMINAMATH_CALUDE_pi_sqrt_two_equality_l2124_212413

theorem pi_sqrt_two_equality : (π - 3.14)^0 + Real.sqrt ((Real.sqrt 2 - 1)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_pi_sqrt_two_equality_l2124_212413


namespace NUMINAMATH_CALUDE_simplify_radicals_l2124_212454

theorem simplify_radicals (y z : ℝ) (h : y ≥ 0 ∧ z ≥ 0) : 
  Real.sqrt (32 * y) * Real.sqrt (75 * z) * Real.sqrt (14 * y) = 40 * y * Real.sqrt (21 * z) := by
  sorry

end NUMINAMATH_CALUDE_simplify_radicals_l2124_212454


namespace NUMINAMATH_CALUDE_hexagon_triangle_area_ratio_l2124_212428

structure RegularHexagon where
  vertices : Finset (Fin 6 → ℝ × ℝ)
  is_regular : sorry
  is_divided : sorry

def center (h : RegularHexagon) : ℝ × ℝ := sorry

def small_triangle (h : RegularHexagon) (i j : Fin 6) (g : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

def large_triangle (h : RegularHexagon) (i j k : Fin 6) : Set (ℝ × ℝ) := sorry

def area (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem hexagon_triangle_area_ratio (h : RegularHexagon) :
  let g := center h
  let small_tri := small_triangle h 0 1 g
  let large_tri := large_triangle h 0 3 5
  area small_tri / area large_tri = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_hexagon_triangle_area_ratio_l2124_212428


namespace NUMINAMATH_CALUDE_paint_cost_per_kg_l2124_212401

/-- The cost of paint per kg given specific conditions -/
theorem paint_cost_per_kg (coverage : Real) (total_cost : Real) (side_length : Real) :
  coverage = 15 →
  total_cost = 200 →
  side_length = 5 →
  (total_cost / (6 * side_length^2 / coverage)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_paint_cost_per_kg_l2124_212401


namespace NUMINAMATH_CALUDE_geese_count_l2124_212467

/-- The number of ducks in the marsh -/
def num_ducks : ℝ := 37.0

/-- The difference between the number of geese and ducks -/
def geese_duck_difference : ℕ := 21

/-- The number of geese in the marsh -/
def num_geese : ℝ := num_ducks + geese_duck_difference

theorem geese_count : num_geese = 58 := by
  sorry

end NUMINAMATH_CALUDE_geese_count_l2124_212467


namespace NUMINAMATH_CALUDE_product_of_sums_and_differences_l2124_212438

theorem product_of_sums_and_differences (W X Y Z : ℝ) : 
  W = (Real.sqrt 2025 + Real.sqrt 2024) →
  X = (-Real.sqrt 2025 - Real.sqrt 2024) →
  Y = (Real.sqrt 2025 - Real.sqrt 2024) →
  Z = (Real.sqrt 2024 - Real.sqrt 2025) →
  W * X * Y * Z = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_and_differences_l2124_212438


namespace NUMINAMATH_CALUDE_only_three_divides_2002_power_l2124_212489

theorem only_three_divides_2002_power : 
  ∀ p : ℕ, Prime p → p < 17 → (p ∣ 2002^2002 - 1) ↔ p = 3 := by
sorry

end NUMINAMATH_CALUDE_only_three_divides_2002_power_l2124_212489


namespace NUMINAMATH_CALUDE_mean_equality_implies_values_l2124_212433

theorem mean_equality_implies_values (x y : ℝ) : 
  (2 + 11 + 6 + x) / 4 = (14 + 9 + y) / 3 → x = -35 ∧ y = -35 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_values_l2124_212433


namespace NUMINAMATH_CALUDE_smallest_integer_power_l2124_212474

theorem smallest_integer_power (x : ℕ) : (∀ y : ℕ, 27^y ≤ 3^24 → y < x) ∧ 27^x > 3^24 ↔ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_power_l2124_212474


namespace NUMINAMATH_CALUDE_same_color_probability_l2124_212469

/-- The number of pairs of shoes -/
def num_pairs : ℕ := 9

/-- The total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes to be selected -/
def selection_size : ℕ := 2

/-- The probability of selecting two shoes of the same color -/
theorem same_color_probability : 
  (num_pairs : ℚ) / (total_shoes.choose selection_size) = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2124_212469


namespace NUMINAMATH_CALUDE_sum_a_plus_c_equals_four_l2124_212431

/-- Represents a three-digit number in the form abc -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem sum_a_plus_c_equals_four :
  ∀ (a c : Nat),
  let num1 := ThreeDigitNumber.mk 2 a 3 (by sorry)
  let num2 := ThreeDigitNumber.mk 6 c 9 (by sorry)
  (num1.toNat + 427 = num2.toNat) →
  (num2.toNat % 3 = 0) →
  a + c = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_plus_c_equals_four_l2124_212431


namespace NUMINAMATH_CALUDE_stone_width_is_five_dm_l2124_212402

/-- Proves that the width of stones used to pave a hall is 5 decimeters -/
theorem stone_width_is_five_dm (hall_length : ℝ) (hall_width : ℝ) 
  (stone_length : ℝ) (num_stones : ℕ) :
  hall_length = 36 →
  hall_width = 15 →
  stone_length = 0.4 →
  num_stones = 2700 →
  ∃ (stone_width : ℝ),
    stone_width = 0.5 ∧
    hall_length * hall_width * 100 = num_stones * stone_length * stone_width :=
by sorry

end NUMINAMATH_CALUDE_stone_width_is_five_dm_l2124_212402


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2124_212463

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem inequality_solution_set 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_inc : ∀ x y, x < y → x ∈ [-1, 1] → y ∈ [-1, 1] → f x < f y)
  (h_dom : ∀ x, f x ≠ 0 → x ∈ [-1, 1]) :
  {x : ℝ | f (x - 1/2) + f (1/4 - x) < 0} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2124_212463


namespace NUMINAMATH_CALUDE_blue_easter_eggs_fraction_l2124_212478

theorem blue_easter_eggs_fraction 
  (purple_fraction : ℚ) 
  (purple_five_candy_ratio : ℚ) 
  (blue_five_candy_ratio : ℚ) 
  (five_candy_probability : ℚ) :
  purple_fraction = 1/5 →
  purple_five_candy_ratio = 1/2 →
  blue_five_candy_ratio = 1/4 →
  five_candy_probability = 3/10 →
  ∃ blue_fraction : ℚ, 
    blue_fraction = 4/5 ∧ 
    purple_fraction * purple_five_candy_ratio + blue_fraction * blue_five_candy_ratio = five_candy_probability :=
by sorry

end NUMINAMATH_CALUDE_blue_easter_eggs_fraction_l2124_212478


namespace NUMINAMATH_CALUDE_least_positive_congruence_l2124_212485

theorem least_positive_congruence :
  ∃! x : ℕ, x > 0 ∧ x + 5600 ≡ 325 [ZMOD 15] ∧ ∀ y : ℕ, y > 0 → y + 5600 ≡ 325 [ZMOD 15] → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_congruence_l2124_212485


namespace NUMINAMATH_CALUDE_S_min_value_l2124_212496

/-- The area function S(a) for a > 1 -/
noncomputable def S (a : ℝ) : ℝ := a^2 / Real.sqrt (a^2 - 1)

/-- Theorem stating the minimum value of S(a) -/
theorem S_min_value (a : ℝ) (h : a > 1) :
  ∃ (min_val : ℝ), min_val = 2 ∧ S (Real.sqrt 2) = min_val ∧ ∀ x > 1, S x ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_S_min_value_l2124_212496


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2124_212434

/-- The complex number -2i+1 corresponds to a point in the fourth quadrant of the complex plane -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := -2 * Complex.I + 1
  (z.re > 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2124_212434


namespace NUMINAMATH_CALUDE_hash_difference_l2124_212435

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x

-- Theorem statement
theorem hash_difference : (hash 7 4) - (hash 4 7) = -9 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_l2124_212435


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2124_212405

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Theorem statement
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 2 + a 3 + a 7 + a 8 = 8 →
  a 4 + a 6 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2124_212405


namespace NUMINAMATH_CALUDE_molecular_weight_N2O5_l2124_212457

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of a compound in g/mol -/
def molecular_weight (n_count : ℕ) (o_count : ℕ) : ℝ :=
  n_count * atomic_weight_N + o_count * atomic_weight_O

/-- Theorem stating that the molecular weight of N2O5 is 108.02 g/mol -/
theorem molecular_weight_N2O5 : 
  molecular_weight 2 5 = 108.02 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_N2O5_l2124_212457


namespace NUMINAMATH_CALUDE_greatest_number_neither_swimming_nor_soccer_l2124_212406

theorem greatest_number_neither_swimming_nor_soccer 
  (total_students : ℕ) 
  (swimming_fans : ℕ) 
  (soccer_fans : ℕ) 
  (h1 : total_students = 1460) 
  (h2 : swimming_fans = 33) 
  (h3 : soccer_fans = 36) : 
  ∃ (neither_fans : ℕ), 
    neither_fans ≤ total_students - (swimming_fans + soccer_fans) ∧ 
    neither_fans = 1391 :=
sorry

end NUMINAMATH_CALUDE_greatest_number_neither_swimming_nor_soccer_l2124_212406


namespace NUMINAMATH_CALUDE_isogonal_conjugate_is_conic_l2124_212446

/-- Trilinear coordinates -/
structure TrilinearCoord where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A triangle -/
structure Triangle where
  A : TrilinearCoord
  B : TrilinearCoord
  C : TrilinearCoord

/-- A line in trilinear coordinates -/
structure Line where
  p : ℝ
  q : ℝ
  r : ℝ

/-- Isogonal conjugation transformation -/
def isogonalConjugate (l : Line) : TrilinearCoord → Prop :=
  fun point => l.p * point.y * point.z + l.q * point.x * point.z + l.r * point.x * point.y = 0

/-- Definition of a conic section -/
def isConicSection (f : TrilinearCoord → Prop) : Prop := sorry

/-- The theorem to be proved -/
theorem isogonal_conjugate_is_conic (t : Triangle) (l : Line) 
  (h1 : l.p ≠ 0) (h2 : l.q ≠ 0) (h3 : l.r ≠ 0)
  (h4 : l.p * t.A.x + l.q * t.A.y + l.r * t.A.z ≠ 0)
  (h5 : l.p * t.B.x + l.q * t.B.y + l.r * t.B.z ≠ 0)
  (h6 : l.p * t.C.x + l.q * t.C.y + l.r * t.C.z ≠ 0) :
  isConicSection (isogonalConjugate l) ∧ 
  isogonalConjugate l t.A ∧ 
  isogonalConjugate l t.B ∧ 
  isogonalConjugate l t.C :=
sorry

end NUMINAMATH_CALUDE_isogonal_conjugate_is_conic_l2124_212446


namespace NUMINAMATH_CALUDE_train_speed_kmph_l2124_212488

/-- Converts speed from meters per second to kilometers per hour -/
def mps_to_kmph (speed_mps : ℝ) : ℝ :=
  speed_mps * 3.6

/-- The speed of the train in meters per second -/
def train_speed_mps : ℝ := 37.503

/-- Theorem: The train's speed in kilometers per hour is 135.0108 -/
theorem train_speed_kmph : mps_to_kmph train_speed_mps = 135.0108 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_kmph_l2124_212488


namespace NUMINAMATH_CALUDE_boat_distance_along_stream_l2124_212479

/-- The distance traveled by a boat along a stream in one hour -/
def distance_along_stream (boat_speed : ℝ) (against_stream_distance : ℝ) : ℝ :=
  boat_speed + (boat_speed - against_stream_distance)

/-- Theorem: The distance traveled by the boat along the stream in one hour is 8 km -/
theorem boat_distance_along_stream :
  distance_along_stream 5 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_along_stream_l2124_212479


namespace NUMINAMATH_CALUDE_a_c_inequality_l2124_212495

theorem a_c_inequality (a c : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : c > 1) :
  a * c + 1 < a + c := by
  sorry

end NUMINAMATH_CALUDE_a_c_inequality_l2124_212495


namespace NUMINAMATH_CALUDE_rock_collecting_contest_l2124_212444

/-- The rock collecting contest between Sydney and Conner --/
theorem rock_collecting_contest 
  (sydney_start : ℕ) 
  (conner_start : ℕ) 
  (sydney_day1 : ℕ) 
  (conner_day1_multiplier : ℕ) 
  (sydney_day2 : ℕ) 
  (conner_day2 : ℕ) 
  (sydney_day3_multiplier : ℕ)
  (h1 : sydney_start = 837)
  (h2 : conner_start = 723)
  (h3 : sydney_day1 = 4)
  (h4 : conner_day1_multiplier = 8)
  (h5 : sydney_day2 = 0)
  (h6 : conner_day2 = 123)
  (h7 : sydney_day3_multiplier = 2) :
  ∃ conner_day3 : ℕ, 
    conner_day3 ≥ 27 ∧ 
    conner_start + conner_day1_multiplier * sydney_day1 + conner_day2 + conner_day3 ≥ 
    sydney_start + sydney_day1 + sydney_day2 + sydney_day3_multiplier * (conner_day1_multiplier * sydney_day1) :=
by sorry

end NUMINAMATH_CALUDE_rock_collecting_contest_l2124_212444


namespace NUMINAMATH_CALUDE_combustible_ice_volume_scientific_notation_l2124_212416

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem combustible_ice_volume_scientific_notation :
  toScientificNotation 19400000000 = ScientificNotation.mk 1.94 10 (by norm_num) (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_combustible_ice_volume_scientific_notation_l2124_212416


namespace NUMINAMATH_CALUDE_bicycle_spoke_ratio_l2124_212450

theorem bicycle_spoke_ratio : 
  ∀ (front_spokes back_spokes : ℕ),
    front_spokes = 20 →
    front_spokes + back_spokes = 60 →
    (back_spokes : ℚ) / front_spokes = 2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_spoke_ratio_l2124_212450


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2124_212476

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + x^2 + 4

theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), f = λ x => (x - 2) * q x + 56 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2124_212476


namespace NUMINAMATH_CALUDE_car_speed_problem_l2124_212404

/-- The speed of Car A in miles per hour -/
def speed_A : ℝ := 58

/-- The speed of Car B in miles per hour -/
def speed_B : ℝ := 50

/-- The initial distance between Car A and Car B in miles -/
def initial_distance : ℝ := 16

/-- The final distance between Car A and Car B in miles -/
def final_distance : ℝ := 8

/-- The time taken for Car A to overtake Car B in hours -/
def time : ℝ := 3

theorem car_speed_problem :
  speed_A * time = speed_B * time + initial_distance + final_distance := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2124_212404


namespace NUMINAMATH_CALUDE_g_is_even_l2124_212465

-- Define the function f on ℝ
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem stating that g is an even function
theorem g_is_even : ∀ x : ℝ, g f x = g f (-x) := by
  sorry

end NUMINAMATH_CALUDE_g_is_even_l2124_212465


namespace NUMINAMATH_CALUDE_pentagon_distance_equality_l2124_212459

/-- A regular pentagon with vertices A1, A2, A3, A4, A5 -/
structure RegularPentagon where
  A1 : ℝ × ℝ
  A2 : ℝ × ℝ
  A3 : ℝ × ℝ
  A4 : ℝ × ℝ
  A5 : ℝ × ℝ
  is_regular : True  -- We assume this property without defining it explicitly

/-- The circumcircle of the regular pentagon -/
def circumcircle (p : RegularPentagon) : Set (ℝ × ℝ) :=
  sorry  -- Definition of the circumcircle

/-- The arc A1A5 of the circumcircle -/
def arcA1A5 (p : RegularPentagon) : Set (ℝ × ℝ) :=
  sorry  -- Definition of the arc A1A5

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry  -- Definition of Euclidean distance

/-- Statement of the theorem -/
theorem pentagon_distance_equality (p : RegularPentagon) (B : ℝ × ℝ)
    (h1 : B ∈ arcA1A5 p)
    (h2 : distance B p.A1 < distance B p.A5) :
    distance B p.A1 + distance B p.A3 + distance B p.A5 =
    distance B p.A2 + distance B p.A4 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_distance_equality_l2124_212459


namespace NUMINAMATH_CALUDE_divisible_by_two_and_three_implies_divisible_by_six_l2124_212472

theorem divisible_by_two_and_three_implies_divisible_by_six (n : ℕ) :
  (n % 2 = 0 ∧ n % 3 = 0) → n % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_two_and_three_implies_divisible_by_six_l2124_212472


namespace NUMINAMATH_CALUDE_marsha_second_package_distance_l2124_212409

/-- Represents the distance Marsha drives for her second package delivery -/
def second_package_distance : ℝ := 28

/-- Represents Marsha's total payment for the day -/
def total_payment : ℝ := 104

/-- Represents Marsha's payment per mile -/
def payment_per_mile : ℝ := 2

/-- Represents the distance Marsha drives for her first package delivery -/
def first_package_distance : ℝ := 10

theorem marsha_second_package_distance :
  second_package_distance = 28 ∧
  total_payment = payment_per_mile * (first_package_distance + second_package_distance + second_package_distance / 2) :=
by sorry

end NUMINAMATH_CALUDE_marsha_second_package_distance_l2124_212409


namespace NUMINAMATH_CALUDE_proposition_relationship_l2124_212400

theorem proposition_relationship :
  (∀ a b : ℝ, (a > b ∧ a⁻¹ > b⁻¹) → a > 0) ∧
  ¬(∀ a b : ℝ, a > 0 → (a > b ∧ a⁻¹ > b⁻¹)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l2124_212400


namespace NUMINAMATH_CALUDE_arc_length_45_degrees_l2124_212418

/-- Given a circle with circumference 100 meters, the length of an arc subtended by a central angle of 45° is 12.5 meters. -/
theorem arc_length_45_degrees (D : Real) (arc_EF : Real) :
  D = 100 → -- Circumference of circle D is 100 meters
  arc_EF = D * (45 / 360) → -- Arc length is proportional to the central angle
  arc_EF = 12.5 := by
sorry

end NUMINAMATH_CALUDE_arc_length_45_degrees_l2124_212418


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_sum_l2124_212407

-- Define an arithmetic sequence of positive integers
def arithmetic_sequence (a₁ : ℕ+) (d : ℕ) : ℕ → ℕ+
  | 0 => a₁
  | n + 1 => ⟨(arithmetic_sequence a₁ d n).val + d, by sorry⟩

-- Sum of the first n terms of an arithmetic sequence
def sum_arithmetic (a₁ : ℕ+) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁.val + (n - 1) * d) / 2

-- Theorem statement
theorem greatest_common_divisor_of_sum (a₁ : ℕ+) (d : ℕ) :
  6 = Nat.gcd (sum_arithmetic a₁ d 12) (Nat.gcd (sum_arithmetic (⟨a₁.val + 1, by sorry⟩) d 12)
    (sum_arithmetic (⟨a₁.val + 2, by sorry⟩) d 12)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_sum_l2124_212407


namespace NUMINAMATH_CALUDE_parallel_vectors_a_value_l2124_212410

def m (a : ℝ) : Fin 2 → ℝ := ![a, -2]
def n (a : ℝ) : Fin 2 → ℝ := ![1, 2-a]

theorem parallel_vectors_a_value :
  ∀ a : ℝ, (∃ k : ℝ, k ≠ 0 ∧ m a = k • n a) → (a = 1 + Real.sqrt 3 ∨ a = 1 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_a_value_l2124_212410


namespace NUMINAMATH_CALUDE_functions_equal_if_surjective_injective_and_greater_or_equal_l2124_212452

theorem functions_equal_if_surjective_injective_and_greater_or_equal
  (f g : ℕ → ℕ)
  (h_surj : Function.Surjective f)
  (h_inj : Function.Injective g)
  (h_ge : ∀ n : ℕ, f n ≥ g n) :
  ∀ n : ℕ, f n = g n := by
  sorry

end NUMINAMATH_CALUDE_functions_equal_if_surjective_injective_and_greater_or_equal_l2124_212452


namespace NUMINAMATH_CALUDE_mixed_oil_rate_l2124_212420

/-- The rate of mixed oil per litre given two different oils -/
theorem mixed_oil_rate (volume1 : ℝ) (price1 : ℝ) (volume2 : ℝ) (price2 : ℝ) :
  volume1 = 10 →
  price1 = 50 →
  volume2 = 5 →
  price2 = 66 →
  (volume1 * price1 + volume2 * price2) / (volume1 + volume2) = 55.33 := by
  sorry

end NUMINAMATH_CALUDE_mixed_oil_rate_l2124_212420


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2124_212427

/-- The sum of the infinite series ∑(n=1 to ∞) (3n+2)/(2^n) is equal to 8 -/
theorem infinite_series_sum : ∑' n, (3 * n + 2) / (2 ^ n) = 8 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2124_212427


namespace NUMINAMATH_CALUDE_expression_evaluation_l2124_212412

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := 3
  (a^2 * b^3 - 1/2 * (4*a*b + 6*a^2*b^3 - 1) + 2*(a*b - a^2*b^3)) = -53/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2124_212412


namespace NUMINAMATH_CALUDE_remainder_problem_l2124_212462

theorem remainder_problem (d : ℕ) (r : ℕ) (h1 : d > 1) 
  (h2 : 1083 % d = r) (h3 : 1455 % d = r) (h4 : 2345 % d = r) : 
  d - r = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2124_212462


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l2124_212468

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 2*x - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 4*x - 2

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) → y = 5*x - 5 :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l2124_212468


namespace NUMINAMATH_CALUDE_stratified_sample_sum_l2124_212458

def total_population : Nat := 100
def sample_size : Nat := 20
def stratum1_size : Nat := 10
def stratum2_size : Nat := 20

theorem stratified_sample_sum :
  let stratum1_sample := sample_size * stratum1_size / total_population
  let stratum2_sample := sample_size * stratum2_size / total_population
  stratum1_sample + stratum2_sample = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_sum_l2124_212458


namespace NUMINAMATH_CALUDE_line_equation_from_slope_and_intercept_l2124_212445

/-- The equation of a line with slope 2 and y-intercept 4 is y = 2x + 4 -/
theorem line_equation_from_slope_and_intercept :
  ∀ (x y : ℝ), (∃ (m b : ℝ), m = 2 ∧ b = 4 ∧ y = m * x + b) → y = 2 * x + 4 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_slope_and_intercept_l2124_212445


namespace NUMINAMATH_CALUDE_unique_number_exists_l2124_212493

theorem unique_number_exists : ∃! x : ℝ, x / 2 + x + 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_exists_l2124_212493


namespace NUMINAMATH_CALUDE_triangle_max_area_l2124_212464

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a * Real.cos C + c * Real.cos A = 3 →
  a^2 + c^2 = 9 + a*c →
  (∃ (S : ℝ), S = (1/2) * a * c * Real.sin B ∧
    ∀ (S' : ℝ), S' = (1/2) * a * c * Real.sin B → S' ≤ S) →
  S = (9 * Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2124_212464


namespace NUMINAMATH_CALUDE_AC_length_approx_l2124_212439

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
def satisfies_conditions (q : Quadrilateral) : Prop :=
  let dist := λ p1 p2 : ℝ × ℝ => Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  dist q.A q.B = 15 ∧ 
  dist q.D q.C = 24 ∧ 
  dist q.A q.D = 9

-- Theorem statement
theorem AC_length_approx (q : Quadrilateral) 
  (h : satisfies_conditions q) : 
  ∃ ε > 0, |dist q.A q.C - 30.7| < ε :=
sorry

#check AC_length_approx

end NUMINAMATH_CALUDE_AC_length_approx_l2124_212439


namespace NUMINAMATH_CALUDE_tom_barbados_cost_l2124_212425

/-- The total cost Tom has to pay for his trip to Barbados -/
def total_cost (num_vaccines : ℕ) (vaccine_cost : ℚ) (doctor_visit_cost : ℚ) 
  (insurance_coverage : ℚ) (trip_cost : ℚ) : ℚ :=
  let medical_cost := num_vaccines * vaccine_cost + doctor_visit_cost
  let insurance_payment := medical_cost * insurance_coverage
  let out_of_pocket_medical := medical_cost - insurance_payment
  out_of_pocket_medical + trip_cost

/-- Theorem stating the total cost Tom has to pay -/
theorem tom_barbados_cost : 
  total_cost 10 45 250 (4/5) 1200 = 1340 := by sorry

end NUMINAMATH_CALUDE_tom_barbados_cost_l2124_212425


namespace NUMINAMATH_CALUDE_infinitely_many_coprime_linear_combination_l2124_212432

theorem infinitely_many_coprime_linear_combination (a b n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hn : n > 0) (hab : Nat.gcd a b = 1) :
  Set.Infinite {k : ℕ | Nat.gcd (a * k + b) n = 1} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_coprime_linear_combination_l2124_212432


namespace NUMINAMATH_CALUDE_greatest_valid_number_l2124_212421

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ 
  ∃ k : ℕ, n = 9 * k + 2 ∧
  ∃ m : ℕ, n = 5 * m + 3

theorem greatest_valid_number : 
  is_valid_number 9962 ∧ ∀ n : ℕ, is_valid_number n → n ≤ 9962 :=
sorry

end NUMINAMATH_CALUDE_greatest_valid_number_l2124_212421


namespace NUMINAMATH_CALUDE_yellow_two_days_ago_white_tomorrow_dandelion_counts_l2124_212437

/-- Represents the state of dandelions in the meadow on a given day -/
structure DandelionState where
  yellow : ℕ
  white : ℕ

/-- The lifecycle of a dandelion -/
def dandelionLifecycle : ℕ := 3

/-- The state of dandelions yesterday -/
def yesterdayState : DandelionState := { yellow := 20, white := 14 }

/-- The state of dandelions today -/
def todayState : DandelionState := { yellow := 15, white := 11 }

/-- Theorem: The number of yellow dandelions the day before yesterday -/
theorem yellow_two_days_ago : ℕ := 25

/-- Theorem: The number of white dandelions tomorrow -/
theorem white_tomorrow : ℕ := 9

/-- Main theorem combining both results -/
theorem dandelion_counts : 
  (yellow_two_days_ago = yesterdayState.white + todayState.white) ∧
  (white_tomorrow = yesterdayState.yellow - todayState.white) := by
  sorry

end NUMINAMATH_CALUDE_yellow_two_days_ago_white_tomorrow_dandelion_counts_l2124_212437


namespace NUMINAMATH_CALUDE_smallest_n_divisible_l2124_212477

theorem smallest_n_divisible (n : ℕ) : 
  (∀ m : ℕ, m > 0 → m < n → ¬(20 ∣ (25 * m) ∧ 18 ∣ (25 * m) ∧ 24 ∣ (25 * m))) →
  (20 ∣ (25 * n) ∧ 18 ∣ (25 * n) ∧ 24 ∣ (25 * n)) →
  n = 36 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_l2124_212477


namespace NUMINAMATH_CALUDE_cyclic_win_sets_count_l2124_212436

/-- Represents a round-robin tournament -/
structure Tournament where
  n : ℕ  -- number of teams
  wins : ℕ  -- number of wins for each team
  losses : ℕ  -- number of losses for each team

/-- Conditions for the tournament -/
def tournament_conditions (t : Tournament) : Prop :=
  t.n * (t.n - 1) / 2 = t.wins * t.n ∧ 
  t.wins = 12 ∧ 
  t.losses = 8 ∧ 
  t.wins + t.losses = t.n - 1

/-- The number of sets of three teams with cyclic wins -/
def cyclic_win_sets (t : Tournament) : ℕ := sorry

/-- Main theorem -/
theorem cyclic_win_sets_count (t : Tournament) : 
  tournament_conditions t → cyclic_win_sets t = 868 := by sorry

end NUMINAMATH_CALUDE_cyclic_win_sets_count_l2124_212436


namespace NUMINAMATH_CALUDE_average_marks_of_failed_boys_l2124_212414

theorem average_marks_of_failed_boys
  (total_boys : ℕ)
  (overall_average : ℚ)
  (passed_average : ℚ)
  (passed_boys : ℕ)
  (h1 : total_boys = 120)
  (h2 : overall_average = 38)
  (h3 : passed_average = 39)
  (h4 : passed_boys = 115) :
  (total_boys * overall_average - passed_boys * passed_average) / (total_boys - passed_boys) = 15 := by
sorry

end NUMINAMATH_CALUDE_average_marks_of_failed_boys_l2124_212414


namespace NUMINAMATH_CALUDE_power_equation_solution_l2124_212448

theorem power_equation_solution (p : ℕ) : (81 ^ 10 : ℕ) = 3 ^ p → p = 40 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2124_212448


namespace NUMINAMATH_CALUDE_sequence_general_term_l2124_212487

theorem sequence_general_term (a : ℕ+ → ℝ) (h1 : a 1 = 2) 
  (h2 : ∀ n : ℕ+, n * a (n + 1) = (n + 1) * a n + 2) : 
  ∀ n : ℕ+, a n = 4 * n - 2 := by
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2124_212487


namespace NUMINAMATH_CALUDE_problem_equivalent_l2124_212447

theorem problem_equivalent : (16^1011) / 8 = 2^4033 := by
  sorry

end NUMINAMATH_CALUDE_problem_equivalent_l2124_212447


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l2124_212443

/-- The center coordinates of a circle with equation (x-h)^2 + (y-k)^2 = r^2 are (h, k) -/
theorem circle_center_coordinates (h k r : ℝ) :
  let circle_equation := fun (x y : ℝ) ↦ (x - h)^2 + (y - k)^2 = r^2
  circle_equation = fun (x y : ℝ) ↦ (x - 2)^2 + (y + 3)^2 = 1 →
  (h, k) = (2, -3) := by
  sorry

#check circle_center_coordinates

end NUMINAMATH_CALUDE_circle_center_coordinates_l2124_212443


namespace NUMINAMATH_CALUDE_total_options_is_twenty_l2124_212460

/-- The number of high-speed trains from location A to location B -/
def num_trains : ℕ := 5

/-- The number of ferries from location B to location C -/
def num_ferries : ℕ := 4

/-- The total number of travel options from location A to location C -/
def total_options : ℕ := num_trains * num_ferries

/-- Theorem stating that the total number of travel options is 20 -/
theorem total_options_is_twenty : total_options = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_options_is_twenty_l2124_212460


namespace NUMINAMATH_CALUDE_combined_8th_grade_percentage_is_21_11_percent_l2124_212490

-- Define the schools and their properties
def parkwood_students : ℕ := 150
def maplewood_students : ℕ := 120
def parkwood_8th_grade_percentage : ℚ := 18 / 100
def maplewood_8th_grade_percentage : ℚ := 25 / 100

-- Define the combined percentage of 8th grade students
def combined_8th_grade_percentage : ℚ := 
  (parkwood_8th_grade_percentage * parkwood_students + maplewood_8th_grade_percentage * maplewood_students) / 
  (parkwood_students + maplewood_students)

-- Theorem statement
theorem combined_8th_grade_percentage_is_21_11_percent : 
  combined_8th_grade_percentage = 2111 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_combined_8th_grade_percentage_is_21_11_percent_l2124_212490


namespace NUMINAMATH_CALUDE_five_thursdays_in_august_l2124_212415

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific date in July or August -/
structure Date where
  month : Nat
  day : Nat

/-- Function to get the day of the week for a given date -/
def dayOfWeek (d : Date) : DayOfWeek := sorry

/-- Function to check if a date is a Monday -/
def isMonday (d : Date) : Prop :=
  dayOfWeek d = DayOfWeek.Monday

/-- Function to check if a date is a Thursday -/
def isThursday (d : Date) : Prop :=
  dayOfWeek d = DayOfWeek.Thursday

/-- Theorem stating that if July has five Mondays, then August has five Thursdays -/
theorem five_thursdays_in_august
  (h1 : ∃ d1 d2 d3 d4 d5 : Date,
    d1.month = 7 ∧ d2.month = 7 ∧ d3.month = 7 ∧ d4.month = 7 ∧ d5.month = 7 ∧
    d1.day < d2.day ∧ d2.day < d3.day ∧ d3.day < d4.day ∧ d4.day < d5.day ∧
    isMonday d1 ∧ isMonday d2 ∧ isMonday d3 ∧ isMonday d4 ∧ isMonday d5)
  (h2 : ∀ d : Date, d.month = 7 → d.day ≤ 31)
  (h3 : ∀ d : Date, d.month = 8 → d.day ≤ 31) :
  ∃ d1 d2 d3 d4 d5 : Date,
    d1.month = 8 ∧ d2.month = 8 ∧ d3.month = 8 ∧ d4.month = 8 ∧ d5.month = 8 ∧
    d1.day < d2.day ∧ d2.day < d3.day ∧ d3.day < d4.day ∧ d4.day < d5.day ∧
    isThursday d1 ∧ isThursday d2 ∧ isThursday d3 ∧ isThursday d4 ∧ isThursday d5 :=
sorry

end NUMINAMATH_CALUDE_five_thursdays_in_august_l2124_212415


namespace NUMINAMATH_CALUDE_dream_sequence_sum_l2124_212475

/-- A sequence is a "dream sequence" if it satisfies the given equation -/
def isDreamSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, 1 / a (n + 1) - 2 / a n = 0

theorem dream_sequence_sum (b : ℕ → ℝ) :
  (∀ n, b n > 0) →  -- b is a positive sequence
  isDreamSequence (λ n => 1 / b n) →  -- 1/b_n is a dream sequence
  b 1 + b 2 + b 3 = 2 →  -- sum of first three terms is 2
  b 6 + b 7 + b 8 = 64 :=  -- sum of 6th, 7th, and 8th terms is 64
by
  sorry

end NUMINAMATH_CALUDE_dream_sequence_sum_l2124_212475


namespace NUMINAMATH_CALUDE_employee_count_l2124_212470

theorem employee_count (average_salary : ℕ) (new_average_salary : ℕ) (manager_salary : ℕ) :
  average_salary = 2400 →
  new_average_salary = 2500 →
  manager_salary = 4900 →
  ∃ n : ℕ, n * average_salary + manager_salary = (n + 1) * new_average_salary ∧ n = 24 :=
by sorry

end NUMINAMATH_CALUDE_employee_count_l2124_212470


namespace NUMINAMATH_CALUDE_secret_spread_theorem_l2124_212473

/-- The number of students who know the secret on a given day -/
def students_knowing_secret (day : ℕ) : ℕ :=
  (3^(day + 1) - 1) / 2

/-- The day of the week when 3280 students know the secret -/
def secret_spread_day : ℕ := 7

/-- Theorem stating that on the 7th day (Sunday), 3280 students know the secret -/
theorem secret_spread_theorem : 
  students_knowing_secret secret_spread_day = 3280 := by
  sorry

end NUMINAMATH_CALUDE_secret_spread_theorem_l2124_212473


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_9_l2124_212419

def is_divisible_by_9 (n : Nat) : Prop := n % 9 = 0

theorem missing_digit_divisible_by_9 :
  let n : Nat := 65304
  is_divisible_by_9 n ∧ 
  ∃ d : Nat, d < 10 ∧ n = 65000 + 300 + d * 10 + 4 :=
by sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_9_l2124_212419


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2124_212499

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x ≤ 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2124_212499


namespace NUMINAMATH_CALUDE_median_mode_difference_l2124_212466

def data : List ℕ := [12, 13, 14, 15, 15, 21, 21, 21, 32, 32, 38, 39, 40, 41, 42, 43, 53, 58, 59]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem median_mode_difference : |median data - mode data| = 11 := by sorry

end NUMINAMATH_CALUDE_median_mode_difference_l2124_212466


namespace NUMINAMATH_CALUDE_x_equals_five_l2124_212461

theorem x_equals_five (x : ℝ) (h : x - 2 = 3) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_five_l2124_212461


namespace NUMINAMATH_CALUDE_ducks_in_lake_l2124_212408

theorem ducks_in_lake (initial_ducks additional_ducks : ℕ) 
  (h1 : initial_ducks = 13)
  (h2 : additional_ducks = 20) :
  initial_ducks + additional_ducks = 33 := by
  sorry

end NUMINAMATH_CALUDE_ducks_in_lake_l2124_212408


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l2124_212484

/-- 
Given a geometric sequence {b_n} where b_n > 0 for all n and the common ratio q > 1,
prove that b₄ + b₈ > b₅ + b₇.
-/
theorem geometric_sequence_inequality (b : ℕ → ℝ) (q : ℝ) 
  (h_positive : ∀ n, b n > 0)
  (h_geometric : ∀ n, b (n + 1) = q * b n)
  (h_q_gt_one : q > 1) :
  b 4 + b 8 > b 5 + b 7 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l2124_212484


namespace NUMINAMATH_CALUDE_four_color_arrangement_l2124_212482

theorem four_color_arrangement : ∀ n : ℕ, n = 4 → (Nat.factorial n) = 24 := by
  sorry

end NUMINAMATH_CALUDE_four_color_arrangement_l2124_212482


namespace NUMINAMATH_CALUDE_unique_number_solution_l2124_212455

theorem unique_number_solution (a b c : ℕ) : 
  70 ≤ a ∧ a < 80 →
  60 ≤ b ∧ b < 70 →
  50 ≤ c ∧ c < 60 →
  a + b = 147 →
  120 ≤ a + c ∧ a + c < 130 →
  120 ≤ b + c ∧ b + c < 130 →
  a + c ≠ b + c →
  a = 78 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_solution_l2124_212455


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_given_remainders_l2124_212486

theorem smallest_positive_integer_with_given_remainders : ∃! n : ℕ+, 
  (n : ℤ) % 5 = 4 ∧
  (n : ℤ) % 7 = 6 ∧
  (n : ℤ) % 9 = 8 ∧
  (n : ℤ) % 11 = 10 ∧
  ∀ m : ℕ+, 
    (m : ℤ) % 5 = 4 ∧
    (m : ℤ) % 7 = 6 ∧
    (m : ℤ) % 9 = 8 ∧
    (m : ℤ) % 11 = 10 →
    n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_given_remainders_l2124_212486


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l2124_212492

theorem absolute_value_simplification (a b : ℝ) (h1 : a < 0) (h2 : a * b < 0) :
  |b - a + 1| - |a - b - 5| = -4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l2124_212492


namespace NUMINAMATH_CALUDE_cousin_calls_l2124_212483

/-- Represents the number of days in a leap year -/
def leapYearDays : ℕ := 366

/-- Represents the calling frequencies of the four cousins -/
def callingFrequencies : List ℕ := [2, 3, 4, 6]

/-- Calculates the number of days with at least one call in a leap year -/
def daysWithCalls (frequencies : List ℕ) (totalDays : ℕ) : ℕ :=
  sorry

theorem cousin_calls :
  daysWithCalls callingFrequencies leapYearDays = 244 :=
sorry

end NUMINAMATH_CALUDE_cousin_calls_l2124_212483


namespace NUMINAMATH_CALUDE_blocks_in_specific_box_l2124_212481

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks that can fit in the box -/
def blocksInBox (box : BoxDimensions) (block : BoxDimensions) : ℕ :=
  (box.length / block.length) * (box.width / block.width) * (box.height / block.height)

theorem blocks_in_specific_box :
  let box := BoxDimensions.mk 4 3 2
  let block := BoxDimensions.mk 3 1 1
  blocksInBox box block = 6 := by sorry

end NUMINAMATH_CALUDE_blocks_in_specific_box_l2124_212481


namespace NUMINAMATH_CALUDE_base_eight_sum_theorem_l2124_212423

/-- Converts a three-digit number in base 8 to its decimal representation -/
def baseEightToDecimal (a b c : ℕ) : ℕ := 64 * a + 8 * b + c

/-- Checks if a number is a valid non-zero digit in base 8 -/
def isValidBaseEightDigit (n : ℕ) : Prop := 0 < n ∧ n < 8

theorem base_eight_sum_theorem (A B C : ℕ) 
  (hA : isValidBaseEightDigit A) 
  (hB : isValidBaseEightDigit B) 
  (hC : isValidBaseEightDigit C) 
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (hSum : baseEightToDecimal A B C + baseEightToDecimal B C A + baseEightToDecimal C A B = baseEightToDecimal A A A) : 
  A + B + C = 8 := by
sorry

end NUMINAMATH_CALUDE_base_eight_sum_theorem_l2124_212423


namespace NUMINAMATH_CALUDE_math_team_combinations_l2124_212426

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem math_team_combinations : 
  let girls := 4
  let boys := 7
  let team_girls := 3
  let team_boys := 3
  (choose girls team_girls) * (choose boys team_boys) = 140 := by
sorry

end NUMINAMATH_CALUDE_math_team_combinations_l2124_212426


namespace NUMINAMATH_CALUDE_yz_minus_zx_minus_xy_l2124_212498

theorem yz_minus_zx_minus_xy (x y z : ℝ) 
  (h1 : x - y - z = 19) 
  (h2 : x^2 + y^2 + z^2 ≠ 19) : 
  y*z - z*x - x*y = 171 := by sorry

end NUMINAMATH_CALUDE_yz_minus_zx_minus_xy_l2124_212498


namespace NUMINAMATH_CALUDE_race_time_difference_l2124_212449

/-- Proves that in a 1000-meter race where runner A finishes in 90 seconds and is 100 meters ahead of runner B at the finish line, A beats B by 9 seconds. -/
theorem race_time_difference (race_length : ℝ) (a_time : ℝ) (distance_difference : ℝ) : 
  race_length = 1000 →
  a_time = 90 →
  distance_difference = 100 →
  (race_length / a_time) * (distance_difference / race_length) * a_time = 9 :=
by
  sorry

#check race_time_difference

end NUMINAMATH_CALUDE_race_time_difference_l2124_212449


namespace NUMINAMATH_CALUDE_x_value_at_y_25_l2124_212456

/-- The constant ratio between (4x - 5) and (2y + 20) -/
def k : ℚ := (4 * 1 - 5) / (2 * 5 + 20)

/-- Theorem stating that given the constant ratio k and the initial condition,
    x equals 2/3 when y equals 25 -/
theorem x_value_at_y_25 (x y : ℚ) 
  (h1 : (4 * x - 5) / (2 * y + 20) = k) 
  (h2 : x = 1 → y = 5) :
  y = 25 → x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_at_y_25_l2124_212456


namespace NUMINAMATH_CALUDE_profit_calculation_correct_l2124_212429

/-- Represents the profit distribution in a partnership business --/
structure ProfitDistribution where
  a_investment : ℚ
  b_investment : ℚ
  c_investment : ℚ
  a_period : ℚ
  b_period : ℚ
  c_period : ℚ
  c_profit : ℚ

/-- Calculates the profit shares and total profit for a given profit distribution --/
def calculate_profits (pd : ProfitDistribution) : 
  (ℚ × ℚ × ℚ × ℚ) :=
  sorry

/-- Theorem stating the correctness of profit calculation --/
theorem profit_calculation_correct (pd : ProfitDistribution) 
  (h1 : pd.a_investment = 2 * pd.b_investment)
  (h2 : pd.b_investment = 3 * pd.c_investment)
  (h3 : pd.a_period = 2 * pd.b_period)
  (h4 : pd.b_period = 3 * pd.c_period)
  (h5 : pd.c_profit = 3000) :
  calculate_profits pd = (108000, 27000, 3000, 138000) :=
  sorry

end NUMINAMATH_CALUDE_profit_calculation_correct_l2124_212429


namespace NUMINAMATH_CALUDE_halfway_between_one_third_and_one_fifth_l2124_212480

theorem halfway_between_one_third_and_one_fifth :
  (1 / 3 : ℚ) + (1 / 5 : ℚ) = 2 * (4 / 15 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_halfway_between_one_third_and_one_fifth_l2124_212480


namespace NUMINAMATH_CALUDE_sum_of_fraction_and_constant_l2124_212471

theorem sum_of_fraction_and_constant (x : Real) (h : x = 8.0) : 0.75 * x + 2 = 8.0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fraction_and_constant_l2124_212471


namespace NUMINAMATH_CALUDE_factors_of_2310_l2124_212411

theorem factors_of_2310 : Nat.card (Nat.divisors 2310) = 32 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_2310_l2124_212411


namespace NUMINAMATH_CALUDE_lunch_breakfast_difference_l2124_212494

def breakfast_cost : ℚ := 2 + 3 + 4 + 3.5 + 1.5

def lunch_base_cost : ℚ := 3.5 + 4 + 5.25 + 6 + 1 + 3

def service_charge (cost : ℚ) : ℚ := cost * (1 + 0.1)

def food_tax (cost : ℚ) : ℚ := cost * (1 + 0.05)

def lunch_total_cost : ℚ := food_tax (service_charge lunch_base_cost)

theorem lunch_breakfast_difference :
  lunch_total_cost - breakfast_cost = 12.28 := by sorry

end NUMINAMATH_CALUDE_lunch_breakfast_difference_l2124_212494


namespace NUMINAMATH_CALUDE_overlapping_squares_area_l2124_212417

/-- Given two squares with side length a that overlap such that one pair of vertices coincide,
    and the overlapping part forms a right triangle with an angle of 30°,
    the area of the non-overlapping part is 2(1 - √3/3)a². -/
theorem overlapping_squares_area (a : ℝ) (h : a > 0) :
  let overlap_angle : ℝ := 30 * π / 180
  let overlap_area : ℝ := a^2 * (Real.sin overlap_angle * Real.cos overlap_angle)
  let non_overlap_area : ℝ := 2 * (a^2 - overlap_area)
  non_overlap_area = 2 * (1 - Real.sqrt 3 / 3) * a^2 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_squares_area_l2124_212417


namespace NUMINAMATH_CALUDE_intersection_points_polar_l2124_212451

/-- The intersection points of ρ = 2sin θ and ρ cos θ = -√3/2 in polar coordinates -/
theorem intersection_points_polar (θ : Real) (h : 0 ≤ θ ∧ θ < 2 * Real.pi) :
  ∃ (ρ₁ ρ₂ θ₁ θ₂ : Real),
    (ρ₁ = 2 * Real.sin θ₁ ∧ ρ₁ * Real.cos θ₁ = -Real.sqrt 3 / 2) ∧
    (ρ₂ = 2 * Real.sin θ₂ ∧ ρ₂ * Real.cos θ₂ = -Real.sqrt 3 / 2) ∧
    ((ρ₁ = 1 ∧ θ₁ = 5 * Real.pi / 6) ∨ (ρ₁ = Real.sqrt 3 ∧ θ₁ = 2 * Real.pi / 3)) ∧
    ((ρ₂ = 1 ∧ θ₂ = 5 * Real.pi / 6) ∨ (ρ₂ = Real.sqrt 3 ∧ θ₂ = 2 * Real.pi / 3)) ∧
    ρ₁ ≠ ρ₂ :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_polar_l2124_212451


namespace NUMINAMATH_CALUDE_min_sum_factors_l2124_212442

def S (n : ℕ) : ℕ := (3 + 7 + 13 + (2*n + 2*n - 1))

theorem min_sum_factors (a b c : ℕ+) (h : S 10 = a * b * c) :
  ∃ (x y z : ℕ+), S 10 = x * y * z ∧ x + y + z ≤ a + b + c ∧ x + y + z = 68 :=
sorry

end NUMINAMATH_CALUDE_min_sum_factors_l2124_212442


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_12_l2124_212430

def is_valid_number (a b : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ a + b = 11 ∧ (a * 1000 + 520 + b) % 12 = 0

theorem four_digit_divisible_by_12 :
  ∀ a b : ℕ, is_valid_number a b → (a = 7 ∧ b = 4) ∨ (a = 3 ∧ b = 8) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_12_l2124_212430


namespace NUMINAMATH_CALUDE_intersection_A_B_union_B_complement_A_C_subset_complement_B_l2124_212491

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x < 9}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}

-- State the theorems
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 5} := by sorry

theorem union_B_complement_A : B ∪ Aᶜ = {x : ℝ | x ≤ 5 ∨ x ≥ 9} := by sorry

theorem C_subset_complement_B (a : ℝ) :
  C a ⊆ Bᶜ ↔ a < -4 ∨ a > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_B_complement_A_C_subset_complement_B_l2124_212491


namespace NUMINAMATH_CALUDE_simplify_inverse_sum_l2124_212403

theorem simplify_inverse_sum (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (y * z + x * z + x * y) / (x * y * z * (x + y + z)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_inverse_sum_l2124_212403
