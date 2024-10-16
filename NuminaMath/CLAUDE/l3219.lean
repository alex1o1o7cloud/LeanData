import Mathlib

namespace NUMINAMATH_CALUDE_expression_bounds_l3219_321936

theorem expression_bounds (x y z w : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + 
    Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ∧
  Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + 
    Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l3219_321936


namespace NUMINAMATH_CALUDE_power_division_rule_l3219_321998

theorem power_division_rule (n : ℕ) : 19^11 / 19^6 = 247609 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l3219_321998


namespace NUMINAMATH_CALUDE_percent_of_a_l3219_321944

theorem percent_of_a (a b c : ℝ) (h1 : c = 0.1 * b) (h2 : b = 2.5 * a) : c = 0.25 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_l3219_321944


namespace NUMINAMATH_CALUDE_divisibility_by_forty_l3219_321946

theorem divisibility_by_forty (p : ℕ) (h_prime : Nat.Prime p) (h_ge_seven : p ≥ 7) :
  40 ∣ (p^2 - 1) ↔ p % 5 = 1 ∨ p % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_forty_l3219_321946


namespace NUMINAMATH_CALUDE_milk_pricing_markup_l3219_321980

/-- The wholesale price of milk in dollars -/
def wholesale_price : ℝ := 4

/-- The price paid by the customer with a 5% discount, in dollars -/
def discounted_price : ℝ := 4.75

/-- The discount percentage as a decimal -/
def discount_rate : ℝ := 0.05

/-- The percentage above wholesale price that we need to prove -/
def markup_percentage : ℝ := 25

theorem milk_pricing_markup :
  let original_price := discounted_price / (1 - discount_rate)
  let markup := original_price - wholesale_price
  markup / wholesale_price * 100 = markup_percentage := by
  sorry

end NUMINAMATH_CALUDE_milk_pricing_markup_l3219_321980


namespace NUMINAMATH_CALUDE_cars_meeting_time_l3219_321923

theorem cars_meeting_time (highway_length : ℝ) (speed1 speed2 : ℝ) (h1 : highway_length = 333)
  (h2 : speed1 = 54) (h3 : speed2 = 57) : 
  (highway_length / (speed1 + speed2)) = 3 := by
sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l3219_321923


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_3_l3219_321941

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The first line: 3y - a = 9x + 1 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := 3 * y - a = 9 * x + 1

/-- The second line: y - 2 = (2a - 3)x -/
def line2 (a : ℝ) (x y : ℝ) : Prop := y - 2 = (2 * a - 3) * x

theorem parallel_lines_a_equals_3 :
  ∀ a : ℝ, (∀ x y : ℝ, line1 a x y ↔ line2 a x y) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_3_l3219_321941


namespace NUMINAMATH_CALUDE_xyz_equals_four_l3219_321978

theorem xyz_equals_four (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 
  x * y * z = 4 := by
sorry

end NUMINAMATH_CALUDE_xyz_equals_four_l3219_321978


namespace NUMINAMATH_CALUDE_ellen_painted_ten_roses_l3219_321970

/-- The time it takes to paint different types of flowers and vines --/
structure PaintingTimes where
  lily : ℕ
  rose : ℕ
  orchid : ℕ
  vine : ℕ

/-- The number of each type of flower and vine painted --/
structure FlowerCounts where
  lilies : ℕ
  roses : ℕ
  orchids : ℕ
  vines : ℕ

/-- Calculates the total time spent painting based on the painting times and flower counts --/
def totalPaintingTime (times : PaintingTimes) (counts : FlowerCounts) : ℕ :=
  times.lily * counts.lilies +
  times.rose * counts.roses +
  times.orchid * counts.orchids +
  times.vine * counts.vines

/-- Theorem: Given the painting times and flower counts, prove that Ellen painted 10 roses --/
theorem ellen_painted_ten_roses
  (times : PaintingTimes)
  (counts : FlowerCounts)
  (h1 : times.lily = 5)
  (h2 : times.rose = 7)
  (h3 : times.orchid = 3)
  (h4 : times.vine = 2)
  (h5 : counts.lilies = 17)
  (h6 : counts.orchids = 6)
  (h7 : counts.vines = 20)
  (h8 : totalPaintingTime times counts = 213) :
  counts.roses = 10 :=
sorry

end NUMINAMATH_CALUDE_ellen_painted_ten_roses_l3219_321970


namespace NUMINAMATH_CALUDE_gcd_triple_existence_l3219_321935

theorem gcd_triple_existence (S : Set ℕ+) 
  (h_infinite : Set.Infinite S)
  (h_distinct_gcd : ∃ (a b c d : ℕ+), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    Nat.gcd a.val b.val ≠ Nat.gcd c.val d.val) :
  ∃ (x y z : ℕ+), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    Nat.gcd x.val y.val = Nat.gcd y.val z.val ∧ 
    Nat.gcd y.val z.val ≠ Nat.gcd z.val x.val :=
sorry

end NUMINAMATH_CALUDE_gcd_triple_existence_l3219_321935


namespace NUMINAMATH_CALUDE_library_books_theorem_l3219_321951

-- Define the universe of books in the library
variable (Book : Type)

-- Define the property of being a new edition
variable (is_new_edition : Book → Prop)

-- Theorem stating that if not all books are new editions,
-- then there exists a book that is not a new edition and not all books are new editions
theorem library_books_theorem (h : ¬ ∀ b : Book, is_new_edition b) :
  (∃ b : Book, ¬ is_new_edition b) ∧ (¬ ∀ b : Book, is_new_edition b) :=
by sorry

end NUMINAMATH_CALUDE_library_books_theorem_l3219_321951


namespace NUMINAMATH_CALUDE_total_boxes_moved_l3219_321999

/-- The number of boxes a truck can hold -/
def boxes_per_truck : ℕ := 4

/-- The number of trips taken to move all boxes -/
def num_trips : ℕ := 218

/-- The total number of boxes moved -/
def total_boxes : ℕ := boxes_per_truck * num_trips

theorem total_boxes_moved :
  total_boxes = 872 := by sorry

end NUMINAMATH_CALUDE_total_boxes_moved_l3219_321999


namespace NUMINAMATH_CALUDE_equal_distribution_of_drawings_l3219_321902

theorem equal_distribution_of_drawings (total_drawings : ℕ) (num_neighbors : ℕ) 
  (h1 : total_drawings = 54) (h2 : num_neighbors = 6) :
  total_drawings / num_neighbors = 9 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_drawings_l3219_321902


namespace NUMINAMATH_CALUDE_fraction_cube_equality_l3219_321947

theorem fraction_cube_equality : (3 / 4 : ℚ) ^ 3 = 27 / 64 := by sorry

end NUMINAMATH_CALUDE_fraction_cube_equality_l3219_321947


namespace NUMINAMATH_CALUDE_gcf_60_75_l3219_321943

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_60_75_l3219_321943


namespace NUMINAMATH_CALUDE_A_power_15_minus_3_power_14_is_zero_l3219_321958

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 5; 0, 3]

theorem A_power_15_minus_3_power_14_is_zero :
  A^15 - 3 • A^14 = 0 := by sorry

end NUMINAMATH_CALUDE_A_power_15_minus_3_power_14_is_zero_l3219_321958


namespace NUMINAMATH_CALUDE_number_above_196_l3219_321949

/-- Represents the number of elements in the k-th row of the array -/
def elementsInRow (k : ℕ) : ℕ := 2 * k - 1

/-- Represents the sum of elements up to and including the k-th row -/
def sumUpToRow (k : ℕ) : ℕ := k^2

/-- Represents the first element in the k-th row -/
def firstElementInRow (k : ℕ) : ℕ := (k - 1)^2 + 1

/-- Represents the last element in the k-th row -/
def lastElementInRow (k : ℕ) : ℕ := k^2

/-- The theorem to be proved -/
theorem number_above_196 :
  ∃ (k : ℕ), 
    sumUpToRow k ≥ 196 ∧
    sumUpToRow (k-1) < 196 ∧
    lastElementInRow (k-1) = 169 := by
  sorry

end NUMINAMATH_CALUDE_number_above_196_l3219_321949


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3219_321921

theorem inequality_solution_set (x : ℝ) : 
  (8 - x^2 > 2*x) ↔ (-4 < x ∧ x < 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3219_321921


namespace NUMINAMATH_CALUDE_fraction_is_positive_integer_iff_p_18_l3219_321996

theorem fraction_is_positive_integer_iff_p_18 (p : ℕ+) :
  (∃ (n : ℕ+), (5 * p + 40 : ℚ) / (3 * p - 7 : ℚ) = n) ↔ p = 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_is_positive_integer_iff_p_18_l3219_321996


namespace NUMINAMATH_CALUDE_nancy_pencils_proof_l3219_321910

/-- The number of pencils Nancy placed in the drawer -/
def pencils_added (initial_pencils total_pencils : ℕ) : ℕ :=
  total_pencils - initial_pencils

theorem nancy_pencils_proof (initial_pencils total_pencils : ℕ) 
  (h1 : initial_pencils = 27)
  (h2 : total_pencils = 72) :
  pencils_added initial_pencils total_pencils = 45 := by
  sorry

#eval pencils_added 27 72

end NUMINAMATH_CALUDE_nancy_pencils_proof_l3219_321910


namespace NUMINAMATH_CALUDE_bank_account_problem_l3219_321987

/-- The bank account problem -/
theorem bank_account_problem (A E : ℝ) 
  (h1 : A > E)  -- Al has more money than Eliot
  (h2 : A - E = (A + E) / 12)  -- Difference is 1/12 of sum
  (h3 : A * 1.1 = E * 1.15 + 22)  -- After increase, Al has $22 more
  : E = 146.67 := by
  sorry

end NUMINAMATH_CALUDE_bank_account_problem_l3219_321987


namespace NUMINAMATH_CALUDE_track_length_l3219_321918

/-- The length of a circular track given specific running conditions -/
theorem track_length (first_lap : ℝ) (other_laps : ℝ) (avg_speed : ℝ) : 
  first_lap = 70 →
  other_laps = 85 →
  avg_speed = 5 →
  (3 : ℝ) * (first_lap + 2 * other_laps) * avg_speed / 3 = 400 := by
  sorry

end NUMINAMATH_CALUDE_track_length_l3219_321918


namespace NUMINAMATH_CALUDE_range_of_f_l3219_321974

def f (x : ℝ) := x^2 - 2*x + 3

theorem range_of_f :
  ∀ y ∈ Set.Icc 2 6, ∃ x ∈ Set.Icc 0 3, f x = y ∧
  ∀ x ∈ Set.Icc 0 3, f x ∈ Set.Icc 2 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3219_321974


namespace NUMINAMATH_CALUDE_seating_arrangements_l3219_321969

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def totalArrangements (n : ℕ) : ℕ := factorial n

def adjacentArrangements (n : ℕ) : ℕ := factorial (n - 1) * factorial 2

theorem seating_arrangements (n : ℕ) (h : n = 8) :
  totalArrangements n - adjacentArrangements n = 30240 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l3219_321969


namespace NUMINAMATH_CALUDE_systematic_sample_count_l3219_321920

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  total_population : ℕ
  sample_size : ℕ
  interval_start : ℕ
  interval_end : ℕ

/-- Calculates the number of sampled individuals within a given interval -/
def count_in_interval (s : SystematicSample) : ℕ :=
  ((s.interval_end - s.interval_start + 1) / (s.total_population / s.sample_size))

/-- Theorem stating that for the given parameters, 13 individuals are sampled from the interval -/
theorem systematic_sample_count (s : SystematicSample) 
  (h1 : s.total_population = 840)
  (h2 : s.sample_size = 42)
  (h3 : s.interval_start = 461)
  (h4 : s.interval_end = 720) :
  count_in_interval s = 13 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_count_l3219_321920


namespace NUMINAMATH_CALUDE_total_birds_on_fence_l3219_321929

def initial_birds : ℕ := 4
def additional_birds : ℕ := 6

theorem total_birds_on_fence :
  initial_birds + additional_birds = 10 := by sorry

end NUMINAMATH_CALUDE_total_birds_on_fence_l3219_321929


namespace NUMINAMATH_CALUDE_salmon_migration_l3219_321908

theorem salmon_migration (male_salmon female_salmon : ℕ) 
  (h1 : male_salmon = 712261)
  (h2 : female_salmon = 259378) :
  male_salmon + female_salmon = 971639 := by
  sorry

end NUMINAMATH_CALUDE_salmon_migration_l3219_321908


namespace NUMINAMATH_CALUDE_exactly_one_pair_probability_l3219_321961

def number_of_pairs : ℕ := 8
def shoes_drawn : ℕ := 4

def total_outcomes : ℕ := (Nat.choose (2 * number_of_pairs) shoes_drawn)

def favorable_outcomes : ℕ :=
  (Nat.choose number_of_pairs 1) *
  (Nat.choose (number_of_pairs - 1) 2) *
  (Nat.choose 2 1) *
  (Nat.choose 2 1)

theorem exactly_one_pair_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 24 / 65 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_pair_probability_l3219_321961


namespace NUMINAMATH_CALUDE_marco_dad_strawberries_l3219_321909

/-- The weight of additional strawberries found by Marco's dad -/
def additional_strawberries (initial_total final_marco final_dad : ℕ) : ℕ :=
  (final_marco + final_dad) - initial_total

theorem marco_dad_strawberries :
  additional_strawberries 22 36 16 = 30 := by
  sorry

end NUMINAMATH_CALUDE_marco_dad_strawberries_l3219_321909


namespace NUMINAMATH_CALUDE_regular_star_points_l3219_321971

/-- An n-pointed regular star polygon -/
structure RegularStar where
  n : ℕ
  angle_A : ℝ
  angle_B : ℝ
  h1 : angle_A = angle_B - 15
  h2 : n * angle_A = n * angle_B - 180

theorem regular_star_points (star : RegularStar) : star.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_star_points_l3219_321971


namespace NUMINAMATH_CALUDE_function_g_property_l3219_321966

theorem function_g_property (g : ℝ → ℝ) 
  (h1 : ∀ (b c : ℝ), c^2 * g b = b^2 * g c) 
  (h2 : g 3 ≠ 0) : 
  (g 6 - g 4) / g 3 = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_function_g_property_l3219_321966


namespace NUMINAMATH_CALUDE_linear_equation_condition_l3219_321963

theorem linear_equation_condition (a : ℝ) : 
  (∀ x, ∃ m k, (a - 2) * x^(|a| - 1) + 6 = m * x + k) ∧ (a - 2 ≠ 0) ↔ a = -2 :=
sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l3219_321963


namespace NUMINAMATH_CALUDE_modular_inverse_45_mod_47_l3219_321940

theorem modular_inverse_45_mod_47 :
  ∃ x : ℕ, x ≤ 46 ∧ (45 * x) % 47 = 1 ∧ x = 23 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_45_mod_47_l3219_321940


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3219_321990

theorem quadratic_factorization (x : ℝ) : x^2 - 4*x + 4 = (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3219_321990


namespace NUMINAMATH_CALUDE_donation_amount_l3219_321977

def barbara_stuffed_animals : ℕ := 9
def trish_stuffed_animals : ℕ := 2 * barbara_stuffed_animals
def sam_stuffed_animals : ℕ := barbara_stuffed_animals + 5
def linda_stuffed_animals : ℕ := sam_stuffed_animals - 7

def barbara_price : ℚ := 2
def trish_price : ℚ := (3:ℚ)/2
def sam_price : ℚ := (5:ℚ)/2
def linda_price : ℚ := 3

def discount_rate : ℚ := (1:ℚ)/10

theorem donation_amount (barbara_stuffed_animals : ℕ) (trish_stuffed_animals : ℕ) 
  (sam_stuffed_animals : ℕ) (linda_stuffed_animals : ℕ) (barbara_price : ℚ) 
  (trish_price : ℚ) (sam_price : ℚ) (linda_price : ℚ) (discount_rate : ℚ) :
  trish_stuffed_animals = 2 * barbara_stuffed_animals →
  sam_stuffed_animals = barbara_stuffed_animals + 5 →
  linda_stuffed_animals = sam_stuffed_animals - 7 →
  barbara_price = 2 →
  trish_price = (3:ℚ)/2 →
  sam_price = (5:ℚ)/2 →
  linda_price = 3 →
  discount_rate = (1:ℚ)/10 →
  (1 - discount_rate) * (barbara_stuffed_animals * barbara_price + 
    trish_stuffed_animals * trish_price + sam_stuffed_animals * sam_price + 
    linda_stuffed_animals * linda_price) = (909:ℚ)/10 := by
  sorry

end NUMINAMATH_CALUDE_donation_amount_l3219_321977


namespace NUMINAMATH_CALUDE_ratio_to_twelve_l3219_321934

theorem ratio_to_twelve : ∃ x : ℝ, (5 : ℝ) / 1 = x / 12 → x = 60 :=
by sorry

end NUMINAMATH_CALUDE_ratio_to_twelve_l3219_321934


namespace NUMINAMATH_CALUDE_problem_1_l3219_321950

theorem problem_1 (x y z : ℝ) : -x * y^2 * z^3 * (-x^2 * y)^3 = x^7 * y^5 * z^3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3219_321950


namespace NUMINAMATH_CALUDE_zero_smallest_natural_l3219_321905

theorem zero_smallest_natural : ∀ n : ℕ, 0 ≤ n := by
  sorry

end NUMINAMATH_CALUDE_zero_smallest_natural_l3219_321905


namespace NUMINAMATH_CALUDE_rupert_weight_l3219_321997

/-- Proves that Rupert weighs 35 kilograms given the conditions -/
theorem rupert_weight (antoinette_weight rupert_weight : ℕ) : 
  antoinette_weight = 63 → 
  antoinette_weight = 2 * rupert_weight - 7 → 
  rupert_weight = 35 := by
  sorry

end NUMINAMATH_CALUDE_rupert_weight_l3219_321997


namespace NUMINAMATH_CALUDE_no_such_polyhedron_l3219_321976

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  faces : ℕ
  vertices : ℕ
  edges : ℕ
  is_convex : Bool
  no_triangular_faces : Bool
  no_three_valent_vertices : Bool

/-- Euler's formula for polyhedra -/
def euler_formula (p : ConvexPolyhedron) : Prop :=
  p.faces + p.vertices - p.edges = 2

/-- Theorem: A convex polyhedron with no triangular faces and no three-valent vertices violates Euler's formula -/
theorem no_such_polyhedron (p : ConvexPolyhedron) 
  (h_convex : p.is_convex = true) 
  (h_no_tri : p.no_triangular_faces = true) 
  (h_no_three : p.no_three_valent_vertices = true) : 
  ¬(euler_formula p) := by
  sorry

end NUMINAMATH_CALUDE_no_such_polyhedron_l3219_321976


namespace NUMINAMATH_CALUDE_reciprocal_sum_pairs_l3219_321984

theorem reciprocal_sum_pairs : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ 
      p.1 > 0 ∧ p.2 > 0 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = (1 : ℚ) / 4) ∧
    s.card = 5 :=
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_pairs_l3219_321984


namespace NUMINAMATH_CALUDE_problem_solution_l3219_321928

theorem problem_solution (y : ℝ) (h : y + Real.sqrt (y^2 - 4) + 1 / (y - Real.sqrt (y^2 - 4)) = 24) :
  y^2 + Real.sqrt (y^4 - 4) + 1 / (y^2 + Real.sqrt (y^4 - 4)) = 1369/36 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3219_321928


namespace NUMINAMATH_CALUDE_fraction_exponent_product_l3219_321948

theorem fraction_exponent_product : (2 / 3 : ℚ)^4 * (1 / 5 : ℚ)^2 = 16 / 2025 := by
  sorry

end NUMINAMATH_CALUDE_fraction_exponent_product_l3219_321948


namespace NUMINAMATH_CALUDE_midpoint_sum_zero_l3219_321973

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (8, 10) and (-4, -14) is 0. -/
theorem midpoint_sum_zero : 
  let x1 : ℝ := 8
  let y1 : ℝ := 10
  let x2 : ℝ := -4
  let y2 : ℝ := -14
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 0 := by
sorry

end NUMINAMATH_CALUDE_midpoint_sum_zero_l3219_321973


namespace NUMINAMATH_CALUDE_number_of_boxes_l3219_321942

theorem number_of_boxes (eggs_per_box : ℕ) (total_eggs : ℕ) (h1 : eggs_per_box = 3) (h2 : total_eggs = 6) :
  total_eggs / eggs_per_box = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boxes_l3219_321942


namespace NUMINAMATH_CALUDE_z_shaped_area_l3219_321901

/-- The area of a Z-shaped region formed by subtracting two squares from a rectangle -/
theorem z_shaped_area (rectangle_length rectangle_width square1_side square2_side : ℝ) 
  (h1 : rectangle_length = 6)
  (h2 : rectangle_width = 4)
  (h3 : square1_side = 2)
  (h4 : square2_side = 1) :
  rectangle_length * rectangle_width - (square1_side^2 + square2_side^2) = 19 := by
  sorry

end NUMINAMATH_CALUDE_z_shaped_area_l3219_321901


namespace NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_l3219_321914

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane using a point and a normal vector
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Two planes are parallel -/
def are_parallel (p1 p2 : Plane3D) : Prop :=
  ∃ k : ℝ, p1.normal = k • p2.normal

/-- A plane is perpendicular to a line -/
def is_perpendicular_to (p : Plane3D) (l : Line3D) : Prop :=
  ∃ k : ℝ, p.normal = k • l.direction

/-- Theorem: Two planes perpendicular to the same line are parallel -/
theorem planes_perpendicular_to_line_are_parallel (p1 p2 : Plane3D) (l : Line3D)
  (h1 : is_perpendicular_to p1 l) (h2 : is_perpendicular_to p2 l) :
  are_parallel p1 p2 :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_l3219_321914


namespace NUMINAMATH_CALUDE_arithmetic_operations_l3219_321981

theorem arithmetic_operations : 
  (1 - 3 - (-8) + (-6) + 10 = 9) ∧ 
  (-12 * ((1/6) - (1/3) - (3/4)) = 11) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l3219_321981


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3219_321916

theorem point_in_second_quadrant :
  let x : ℝ := Real.sin (2014 * π / 180)
  let y : ℝ := Real.tan (2014 * π / 180)
  x < 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3219_321916


namespace NUMINAMATH_CALUDE_arrange_plants_under_lamps_count_l3219_321937

/-- Represents the number of ways to arrange plants under lamps -/
def arrange_plants_under_lamps : ℕ :=
  let num_plants : ℕ := 4
  let num_plant_types : ℕ := 3
  let num_lamps : ℕ := 4
  let num_lamp_colors : ℕ := 2
  
  -- All plants under same color lamp
  let all_under_one_color : ℕ := num_lamp_colors
  let three_under_one_color : ℕ := num_plants * num_lamp_colors
  
  -- Plants under different colored lamps
  let two_types_each_color : ℕ := (Nat.choose num_plant_types 2) * num_lamp_colors
  let one_type_alone : ℕ := num_plant_types * num_lamp_colors
  
  all_under_one_color + three_under_one_color + two_types_each_color + one_type_alone

/-- Theorem stating the correct number of ways to arrange plants under lamps -/
theorem arrange_plants_under_lamps_count :
  arrange_plants_under_lamps = 22 := by sorry

end NUMINAMATH_CALUDE_arrange_plants_under_lamps_count_l3219_321937


namespace NUMINAMATH_CALUDE_point_translation_to_origin_l3219_321954

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point by a given vector -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  ⟨p.x + dx, p.y + dy⟩

theorem point_translation_to_origin (A : Point) :
  translate A 3 2 = ⟨0, 0⟩ → A = ⟨-3, -2⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_translation_to_origin_l3219_321954


namespace NUMINAMATH_CALUDE_population_difference_after_two_years_l3219_321988

/-- The difference in population between city A and city C after 2 years -/
def population_difference (A B C : ℝ) : ℝ :=
  A * (1 + 0.03)^2 - C * (1 + 0.02)^2

/-- Theorem stating the difference in population after 2 years -/
theorem population_difference_after_two_years (A B C : ℝ) 
  (h : A + B = B + C + 5000) :
  population_difference A B C = 0.0205 * A + 5202 := by
  sorry

end NUMINAMATH_CALUDE_population_difference_after_two_years_l3219_321988


namespace NUMINAMATH_CALUDE_three_propositions_true_l3219_321922

-- Define reciprocals
def are_reciprocals (x y : ℝ) : Prop := x * y = 1

-- Define triangle area and congruence
def triangle_area (t : Set ℝ × Set ℝ × Set ℝ) : ℝ := sorry
def triangle_congruent (t1 t2 : Set ℝ × Set ℝ × Set ℝ) : Prop := sorry

-- Define quadratic equation solution existence
def has_real_solutions (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + m = 0

theorem three_propositions_true : 
  (∀ x y : ℝ, are_reciprocals x y → x * y = 1) ∧ 
  (∃ t1 t2 : Set ℝ × Set ℝ × Set ℝ, triangle_area t1 = triangle_area t2 ∧ ¬ triangle_congruent t1 t2) ∧
  (∀ m : ℝ, ¬ has_real_solutions m → m > 1) := by
  sorry

end NUMINAMATH_CALUDE_three_propositions_true_l3219_321922


namespace NUMINAMATH_CALUDE_no_valid_solutions_l3219_321925

theorem no_valid_solutions : ¬∃ (a b : ℝ), ∀ x, (a * x + b)^2 = 4 * x^2 + 4 * x + 4 := by sorry

end NUMINAMATH_CALUDE_no_valid_solutions_l3219_321925


namespace NUMINAMATH_CALUDE_max_qpn_value_l3219_321911

/-- Represents a two-digit number with equal digits -/
def TwoDigitEqualDigits (n : Nat) : Prop :=
  n ≥ 11 ∧ n ≤ 99 ∧ n % 11 = 0

/-- Represents a one-digit number -/
def OneDigit (n : Nat) : Prop :=
  n ≥ 1 ∧ n ≤ 9

/-- Represents a three-digit number -/
def ThreeDigits (n : Nat) : Prop :=
  n ≥ 100 ∧ n ≤ 999

theorem max_qpn_value (nn n qpn : Nat) 
  (h1 : TwoDigitEqualDigits nn)
  (h2 : OneDigit n)
  (h3 : ThreeDigits qpn)
  (h4 : nn * n = qpn) :
  qpn ≤ 396 :=
sorry

end NUMINAMATH_CALUDE_max_qpn_value_l3219_321911


namespace NUMINAMATH_CALUDE_sequence_calculation_l3219_321924

def x (n : ℕ) : ℕ := n^2 + n
def y (n : ℕ) : ℕ := 2 * n^2
def z (n : ℕ) : ℕ := n^3
def t (n : ℕ) : ℕ := 2^n

theorem sequence_calculation :
  (x 1 = 2 ∧ x 2 = 6 ∧ x 3 = 12 ∧ x 4 = 20) ∧
  (y 1 = 2 ∧ y 2 = 8 ∧ y 3 = 18 ∧ y 4 = 32) ∧
  (z 1 = 1 ∧ z 2 = 8 ∧ z 3 = 27 ∧ z 4 = 64) ∧
  (t 1 = 2 ∧ t 2 = 4 ∧ t 3 = 8 ∧ t 4 = 16) := by
  sorry

end NUMINAMATH_CALUDE_sequence_calculation_l3219_321924


namespace NUMINAMATH_CALUDE_egg_usage_ratio_l3219_321965

/-- Proves that the ratio of eggs used to total eggs bought is 1:2 --/
theorem egg_usage_ratio (total_dozen : ℕ) (broken : ℕ) (left : ℕ) : 
  total_dozen = 6 → broken = 15 → left = 21 → 
  (total_dozen * 12 - (left + broken)) * 2 = total_dozen * 12 := by
  sorry

end NUMINAMATH_CALUDE_egg_usage_ratio_l3219_321965


namespace NUMINAMATH_CALUDE_evening_milk_is_380_l3219_321955

/-- Represents the milk production and sales for Aunt May's farm --/
structure MilkProduction where
  morning : ℕ
  evening : ℕ
  leftover : ℕ
  sold : ℕ
  remaining : ℕ

/-- Calculates the evening milk production given the other parameters --/
def calculate_evening_milk (mp : MilkProduction) : ℕ :=
  mp.remaining + mp.sold - mp.morning - mp.leftover

/-- Theorem stating that the evening milk production is 380 gallons --/
theorem evening_milk_is_380 (mp : MilkProduction) 
  (h1 : mp.morning = 365)
  (h2 : mp.leftover = 15)
  (h3 : mp.sold = 612)
  (h4 : mp.remaining = 148) :
  calculate_evening_milk mp = 380 := by
  sorry

#eval calculate_evening_milk { morning := 365, evening := 0, leftover := 15, sold := 612, remaining := 148 }

end NUMINAMATH_CALUDE_evening_milk_is_380_l3219_321955


namespace NUMINAMATH_CALUDE_inequality_of_squares_and_sum_l3219_321933

theorem inequality_of_squares_and_sum (a b c : ℝ) :
  Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + a^2) ≥ Real.sqrt 2 * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_squares_and_sum_l3219_321933


namespace NUMINAMATH_CALUDE_trig_identity_l3219_321915

theorem trig_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3219_321915


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3219_321972

theorem infinite_series_sum : 
  ∑' (n : ℕ), (1 : ℝ) / (n * (n + 3)) = 1/3 + 1/6 + 1/9 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3219_321972


namespace NUMINAMATH_CALUDE_triangle_probability_theorem_l3219_321931

noncomputable def triangle_probability (XY : ℝ) (angle_XYZ : ℝ) : ℝ :=
  (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3

theorem triangle_probability_theorem (XY : ℝ) (angle_XYZ : ℝ) :
  XY = 12 →
  angle_XYZ = π / 6 →
  triangle_probability XY angle_XYZ = (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_probability_theorem_l3219_321931


namespace NUMINAMATH_CALUDE_a_minus_b_equals_thirteen_l3219_321930

theorem a_minus_b_equals_thirteen (a b : ℝ) 
  (ha : |a| = 8)
  (hb : |b| = 5)
  (ha_pos : a > 0)
  (hb_neg : b < 0) : 
  a - b = 13 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_thirteen_l3219_321930


namespace NUMINAMATH_CALUDE_tv_sales_effect_l3219_321903

theorem tv_sales_effect (price_reduction : Real) (sales_increase : Real) :
  price_reduction = 0.18 →
  sales_increase = 0.72 →
  let new_price_factor := 1 - price_reduction
  let new_sales_factor := 1 + sales_increase
  let net_effect := new_price_factor * new_sales_factor - 1
  net_effect * 100 = 41.04 := by
  sorry

end NUMINAMATH_CALUDE_tv_sales_effect_l3219_321903


namespace NUMINAMATH_CALUDE_complex_calculation_l3219_321907

theorem complex_calculation (a b : ℂ) (ha : a = 3 + 2*I) (hb : b = -2 + I) :
  4*a - 2*b = 16 + 6*I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l3219_321907


namespace NUMINAMATH_CALUDE_range_of_k_for_special_function_l3219_321913

theorem range_of_k_for_special_function (f : ℝ → ℝ) (k a b : ℝ) :
  (∀ x, f x = Real.sqrt (x + 2) + k) →
  a < b →
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y) →
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b) →
  k ∈ Set.Ioo (-9/4) (-2) := by
sorry

end NUMINAMATH_CALUDE_range_of_k_for_special_function_l3219_321913


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3219_321904

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 - i) / (1 + i) = (1 : ℂ) / 2 - (3 : ℂ) / 2 * i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3219_321904


namespace NUMINAMATH_CALUDE_benzene_molecular_weight_l3219_321956

-- Define atomic weights
def carbon_weight : ℝ := 12.01
def hydrogen_weight : ℝ := 1.008

-- Define the number of atoms in C6H6
def carbon_atoms : ℕ := 6
def hydrogen_atoms : ℕ := 6

-- Define the number of moles
def moles : ℝ := 4

-- Theorem statement
theorem benzene_molecular_weight :
  let benzene_weight := (carbon_atoms * carbon_weight + hydrogen_atoms * hydrogen_weight)
  moles * benzene_weight = 312.432 := by
  sorry

end NUMINAMATH_CALUDE_benzene_molecular_weight_l3219_321956


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3219_321982

theorem fraction_meaningful (m : ℝ) : 
  (∃ (x : ℝ), x = 1 / (m + 3)) ↔ m ≠ -3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3219_321982


namespace NUMINAMATH_CALUDE_range_of_m_range_of_a_l3219_321939

-- Define the propositions
def p (m : ℝ) : Prop := |m - 2| < 1
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*Real.sqrt 2*x + m = 0
def r (m a : ℝ) : Prop := a - 2 < m ∧ m < a + 1

-- Theorem 1
theorem range_of_m (m : ℝ) : p m ∧ ¬(q m) → 2 < m ∧ m < 3 := by sorry

-- Theorem 2
theorem range_of_a (a : ℝ) : 
  (∀ m : ℝ, p m → r m a) ∧ ¬(∀ m : ℝ, r m a → p m) → 
  2 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_m_range_of_a_l3219_321939


namespace NUMINAMATH_CALUDE_clown_balloons_l3219_321960

/-- The number of balloons the clown blew up initially -/
def initial_balloons : ℕ := sorry

/-- The number of additional balloons the clown blew up -/
def additional_balloons : ℕ := 13

/-- The total number of balloons the clown has now -/
def total_balloons : ℕ := 60

theorem clown_balloons : initial_balloons = 47 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l3219_321960


namespace NUMINAMATH_CALUDE_equation_solution_l3219_321985

theorem equation_solution : ∃ x : ℝ, 35 - (5 + 3) = 7 + x ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3219_321985


namespace NUMINAMATH_CALUDE_simplify_fraction_l3219_321962

theorem simplify_fraction : (121 : ℚ) / 13310 = 1 / 110 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3219_321962


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3219_321979

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ) :
  is_arithmetic_sequence a d →
  d > 0 →
  (a 1 + a 2 + a 3 = 15) →
  (a 1 * a 2 * a 3 = 80) →
  (a 11 + a 12 + a 13 = 105) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3219_321979


namespace NUMINAMATH_CALUDE_curve_line_intersection_l3219_321991

theorem curve_line_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2^2 = p.1 ∧ p.2 + 1 = k * p.1) → k = 0 ∨ k = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_curve_line_intersection_l3219_321991


namespace NUMINAMATH_CALUDE_expected_total_rainfall_l3219_321983

/-- Weather forecast for a single day -/
structure DailyForecast where
  sun_prob : ℝ
  rain_3in_prob : ℝ
  rain_8in_prob : ℝ
  prob_sum_is_one : sun_prob + rain_3in_prob + rain_8in_prob = 1
  probs_nonnegative : 0 ≤ sun_prob ∧ 0 ≤ rain_3in_prob ∧ 0 ≤ rain_8in_prob

/-- Calculate expected rainfall for a single day -/
def expected_daily_rainfall (forecast : DailyForecast) : ℝ :=
  forecast.sun_prob * 0 + forecast.rain_3in_prob * 3 + forecast.rain_8in_prob * 8

/-- The weather forecast for the week -/
def week_forecast : DailyForecast :=
  { sun_prob := 0.5
    rain_3in_prob := 0.3
    rain_8in_prob := 0.2
    prob_sum_is_one := by norm_num
    probs_nonnegative := by norm_num }

/-- Number of days in the forecast period -/
def forecast_days : ℕ := 5

/-- Theorem: The expected total rainfall for the week is 12.5 inches -/
theorem expected_total_rainfall :
  forecast_days * (expected_daily_rainfall week_forecast) = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_total_rainfall_l3219_321983


namespace NUMINAMATH_CALUDE_x_intercept_distance_l3219_321945

/-- Given two lines that intersect at (8, 20) with slopes 4 and 7 respectively,
    the distance between their x-intercepts is 15/7. -/
theorem x_intercept_distance (line1 line2 : ℝ → ℝ) : 
  (∀ x, line1 x = 4 * (x - 8) + 20) →
  (∀ x, line2 x = 7 * (x - 8) + 20) →
  let x1 := (line1 0 + 20) / 4 + 8
  let x2 := (line2 0 + 20) / 7 + 8
  |x1 - x2| = 15 / 7 := by
sorry

end NUMINAMATH_CALUDE_x_intercept_distance_l3219_321945


namespace NUMINAMATH_CALUDE_average_salary_calculation_l3219_321975

theorem average_salary_calculation (officer_salary : ℕ) (non_officer_salary : ℕ)
  (num_officers : ℕ) (num_non_officers : ℕ) :
  officer_salary = 430 →
  non_officer_salary = 110 →
  num_officers = 15 →
  num_non_officers = 465 →
  (officer_salary * num_officers + non_officer_salary * num_non_officers) / (num_officers + num_non_officers) = 120 := by
  sorry

#eval (430 * 15 + 110 * 465) / (15 + 465)

end NUMINAMATH_CALUDE_average_salary_calculation_l3219_321975


namespace NUMINAMATH_CALUDE_function_equality_l3219_321968

theorem function_equality (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x ≤ x) 
  (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_function_equality_l3219_321968


namespace NUMINAMATH_CALUDE_added_number_after_doubling_l3219_321993

theorem added_number_after_doubling (x : ℝ) (y : ℝ) (h : x = 4) : 3 * (2 * x + y) = 51 → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_added_number_after_doubling_l3219_321993


namespace NUMINAMATH_CALUDE_carmen_candle_usage_l3219_321938

/-- Calculates the number of candles needed for a given number of nights and burning hours per night. -/
def candles_needed (total_nights : ℕ) (hours_per_night : ℕ) (nights_per_candle_at_one_hour : ℕ) : ℕ :=
  total_nights * hours_per_night / nights_per_candle_at_one_hour

theorem carmen_candle_usage :
  candles_needed 24 2 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_carmen_candle_usage_l3219_321938


namespace NUMINAMATH_CALUDE_fraction_simplification_l3219_321957

theorem fraction_simplification (a b c : ℝ) (h : 2*a - 3*c - 4 + b ≠ 0) :
  (6*a^2 - 2*b^2 + 6*c^2 + a*b - 13*a*c - 4*b*c - 18*a - 5*b + 17*c + 12) / 
  (4*a^2 - b^2 + 9*c^2 - 12*a*c - 16*a + 24*c + 16) = 
  (3*a - 2*c - 3 + 2*b) / (2*a - 3*c - 4 + b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3219_321957


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3219_321927

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im ((1 - 2*i) / (2 + i)) = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3219_321927


namespace NUMINAMATH_CALUDE_rachel_furniture_assembly_time_l3219_321994

/-- Calculates the total time to assemble furniture -/
def total_assembly_time (chairs : ℕ) (tables : ℕ) (time_per_piece : ℕ) : ℕ :=
  (chairs + tables) * time_per_piece

/-- Theorem: The total assembly time for Rachel's furniture -/
theorem rachel_furniture_assembly_time :
  total_assembly_time 7 3 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_rachel_furniture_assembly_time_l3219_321994


namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l3219_321959

theorem indefinite_integral_proof (x : ℝ) :
  deriv (fun x => (2 - 3*x) * Real.exp (2*x)) = fun x => (1 - 6*x) * Real.exp (2*x) := by
  sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l3219_321959


namespace NUMINAMATH_CALUDE_orangeade_price_day2_l3219_321953

/-- Represents the price and composition of orangeade on two consecutive days -/
structure Orangeade where
  orange_juice : ℝ
  water_day1 : ℝ
  water_day2 : ℝ
  price_day1 : ℝ
  price_day2 : ℝ

/-- Theorem stating the conditions and the result to be proven -/
theorem orangeade_price_day2 (o : Orangeade) 
  (h1 : o.orange_juice = o.water_day1)
  (h2 : o.water_day2 = 2 * o.water_day1)
  (h3 : o.price_day1 = 0.3)
  (h4 : (o.orange_juice + o.water_day1) * o.price_day1 = 
        (o.orange_juice + o.water_day2) * o.price_day2) :
  o.price_day2 = 0.2 := by
  sorry

#check orangeade_price_day2

end NUMINAMATH_CALUDE_orangeade_price_day2_l3219_321953


namespace NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l3219_321912

theorem lcm_from_hcf_and_product (a b : ℕ+) :
  Nat.gcd a b = 20 →
  a * b = 2560 →
  Nat.lcm a b = 128 := by
sorry

end NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l3219_321912


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l3219_321906

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Represents the ball placement process -/
def ballPlacement (step : ℕ) : ℕ :=
  sorry

theorem ball_placement_theorem (step : ℕ) :
  step = 1024 →
  ballPlacement step = sumDigits (toBase7 step) :=
sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l3219_321906


namespace NUMINAMATH_CALUDE_solution_to_equation_l3219_321967

theorem solution_to_equation (x : ℝ) :
  Real.sqrt (4 * x^2 + 4 * x + 1) - Real.sqrt (4 * x^2 - 12 * x + 9) = 4 →
  x ≥ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3219_321967


namespace NUMINAMATH_CALUDE_min_difference_is_1747_l3219_321919

/-- Represents a valid digit assignment for the problem -/
structure DigitAssignment where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                  d ≠ e ∧ d ≠ f ∧
                  e ≠ f
  all_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0
  sum_constraint : 1000 * a + 100 * b + 10 * c + d + 10 * e + f = 2010

/-- The main theorem stating the minimum difference -/
theorem min_difference_is_1747 : 
  ∀ (assign : DigitAssignment), 
    1000 * assign.a + 100 * assign.b + 10 * assign.c + assign.d - (10 * assign.e + assign.f) = 1747 := by
  sorry

end NUMINAMATH_CALUDE_min_difference_is_1747_l3219_321919


namespace NUMINAMATH_CALUDE_g_negative_three_value_l3219_321995

theorem g_negative_three_value (g : ℝ → ℝ) (h : ∀ x : ℝ, g (5 * x - 7) = 8 * x + 2) :
  g (-3) = 8.4 := by
  sorry

end NUMINAMATH_CALUDE_g_negative_three_value_l3219_321995


namespace NUMINAMATH_CALUDE_max_score_is_43_l3219_321986

-- Define the sightseeing point type
structure SightseeingPoint where
  score : ℕ
  time : ℚ
  cost : ℕ

-- Define the list of sightseeing points
def sightseeingPoints : List SightseeingPoint := [
  ⟨10, 2/3, 1000⟩,
  ⟨7, 1/2, 700⟩,
  ⟨6, 1/3, 300⟩,
  ⟨8, 2/3, 800⟩,
  ⟨5, 1/4, 200⟩,
  ⟨9, 2/3, 900⟩,
  ⟨8, 1/2, 900⟩,
  ⟨8, 2/5, 600⟩,
  ⟨5, 1/5, 400⟩,
  ⟨6, 1/4, 600⟩
]

-- Define a function to check if a selection of points is valid
def isValidSelection (selection : List Bool) : Prop :=
  let selectedPoints := List.zipWith (λ s p => if s then some p else none) selection sightseeingPoints
  let totalTime := selectedPoints.filterMap id |>.map SightseeingPoint.time |>.sum
  let totalCost := selectedPoints.filterMap id |>.map SightseeingPoint.cost |>.sum
  totalTime < 3 ∧ totalCost ≤ 3500

-- Define a function to calculate the total score of a selection
def totalScore (selection : List Bool) : ℕ :=
  let selectedPoints := List.zipWith (λ s p => if s then some p else none) selection sightseeingPoints
  selectedPoints.filterMap id |>.map SightseeingPoint.score |>.sum

-- State the theorem
theorem max_score_is_43 :
  ∃ (selection : List Bool),
    selection.length = sightseeingPoints.length ∧
    isValidSelection selection ∧
    totalScore selection = 43 ∧
    ∀ (otherSelection : List Bool),
      otherSelection.length = sightseeingPoints.length →
      isValidSelection otherSelection →
      totalScore otherSelection ≤ 43 := by
  sorry

end NUMINAMATH_CALUDE_max_score_is_43_l3219_321986


namespace NUMINAMATH_CALUDE_simplify_sum_of_roots_l3219_321900

theorem simplify_sum_of_roots : 
  Real.sqrt (1 + 2) + Real.sqrt (1 + 2 + 3) + Real.sqrt (1 + 2 + 3 + 4) + Real.sqrt (1 + 2 + 3 + 4 + 5) + 2 = 
  Real.sqrt 3 + Real.sqrt 6 + Real.sqrt 10 + Real.sqrt 15 + 2 := by sorry

end NUMINAMATH_CALUDE_simplify_sum_of_roots_l3219_321900


namespace NUMINAMATH_CALUDE_det_B_equals_four_l3219_321932

theorem det_B_equals_four (b c : ℝ) (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B = ![![b, 2], ![-3, c]] →
  B + 2 * B⁻¹ = 0 →
  Matrix.det B = 4 := by
sorry

end NUMINAMATH_CALUDE_det_B_equals_four_l3219_321932


namespace NUMINAMATH_CALUDE_right_triangle_condition_l3219_321992

-- Define a triangle with angles α, β, and γ
structure Triangle where
  α : Real
  β : Real
  γ : Real
  angle_sum : α + β + γ = π
  positive_angles : 0 < α ∧ 0 < β ∧ 0 < γ

-- Theorem statement
theorem right_triangle_condition (t : Triangle) :
  Real.sin t.α + Real.cos t.α = Real.sin t.β + Real.cos t.β →
  t.α = π/2 ∨ t.β = π/2 ∨ t.γ = π/2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_condition_l3219_321992


namespace NUMINAMATH_CALUDE_cakes_baked_lunch_is_eight_l3219_321917

/-- The number of cakes baked during lunch today -/
def cakes_baked_lunch : ℕ := sorry

/-- The number of cakes sold during dinner -/
def cakes_sold_dinner : ℕ := 6

/-- The number of cakes baked yesterday -/
def cakes_baked_yesterday : ℕ := 3

/-- The number of cakes left -/
def cakes_left : ℕ := 2

/-- Theorem stating that the number of cakes baked during lunch today is 8 -/
theorem cakes_baked_lunch_is_eight :
  cakes_baked_lunch = 8 :=
by sorry

end NUMINAMATH_CALUDE_cakes_baked_lunch_is_eight_l3219_321917


namespace NUMINAMATH_CALUDE_a_greater_than_b_l3219_321989

theorem a_greater_than_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l3219_321989


namespace NUMINAMATH_CALUDE_platform_length_calculation_l3219_321952

/-- Given a train of length 300 meters that crosses a platform in 39 seconds
    and a signal pole in 14 seconds, prove that the length of the platform
    is approximately 535.77 meters. -/
theorem platform_length_calculation (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_time = 39)
  (h3 : pole_time = 14) :
  ∃ platform_length : ℝ, abs (platform_length - 535.77) < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_platform_length_calculation_l3219_321952


namespace NUMINAMATH_CALUDE_robin_afternoon_bottles_l3219_321926

/-- The number of bottles Robin drank in the morning -/
def morning_bottles : ℕ := 7

/-- The total number of bottles Robin drank -/
def total_bottles : ℕ := 14

/-- The number of bottles Robin drank in the afternoon -/
def afternoon_bottles : ℕ := total_bottles - morning_bottles

theorem robin_afternoon_bottles : afternoon_bottles = 7 := by
  sorry

end NUMINAMATH_CALUDE_robin_afternoon_bottles_l3219_321926


namespace NUMINAMATH_CALUDE_books_given_correct_l3219_321964

/-- The number of books Melissa gives to Jordan --/
def books_given : ℝ := 10.5

/-- Initial number of books Melissa had --/
def melissa_initial : ℕ := 123

/-- Initial number of books Jordan had --/
def jordan_initial : ℕ := 27

theorem books_given_correct :
  let melissa_final := melissa_initial - books_given
  let jordan_final := jordan_initial + books_given
  (melissa_initial + jordan_initial : ℝ) = melissa_final + jordan_final ∧
  melissa_final = 3 * jordan_final := by
  sorry

end NUMINAMATH_CALUDE_books_given_correct_l3219_321964
