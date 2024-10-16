import Mathlib

namespace NUMINAMATH_CALUDE_geometric_relations_l3125_312567

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Axioms
axiom different_lines {m n : Line} : m ≠ n
axiom different_planes {α β : Plane} : α ≠ β

-- Theorem
theorem geometric_relations 
  (m n : Line) (α β : Plane) :
  (perpendicular_plane m α ∧ perpendicular m n → 
    parallel_plane n α ∨ contains α n) ∧
  (parallel_planes α β ∧ perpendicular_plane n α ∧ parallel_plane m β → 
    perpendicular m n) ∧
  (parallel_plane m α ∧ perpendicular_plane n β ∧ perpendicular m n → 
    ¬(perpendicular_planes α β)) ∧
  (parallel_plane m α ∧ perpendicular_plane n β ∧ parallel m n → 
    perpendicular_planes α β) :=
by sorry


end NUMINAMATH_CALUDE_geometric_relations_l3125_312567


namespace NUMINAMATH_CALUDE_system_solution_l3125_312555

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 7 * x + y = 19
def equation2 (x y : ℝ) : Prop := x + 3 * y = 1
def equation3 (x y z : ℝ) : Prop := 2 * x + y - 4 * z = 10

-- Theorem statement
theorem system_solution (x y z : ℝ) :
  equation1 x y ∧ equation2 x y ∧ equation3 x y z →
  2 * x + y + 3 * z = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3125_312555


namespace NUMINAMATH_CALUDE_robin_cupcake_ratio_l3125_312578

/-- Given that Robin ate 4 cupcakes with chocolate sauce and 12 cupcakes in total,
    prove that the ratio of cupcakes with buttercream frosting to cupcakes with chocolate sauce is 2:1 -/
theorem robin_cupcake_ratio :
  let chocolate_cupcakes : ℕ := 4
  let total_cupcakes : ℕ := 12
  let buttercream_cupcakes : ℕ := total_cupcakes - chocolate_cupcakes
  (buttercream_cupcakes : ℚ) / chocolate_cupcakes = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_robin_cupcake_ratio_l3125_312578


namespace NUMINAMATH_CALUDE_max_triangles_is_eleven_l3125_312539

/-- Represents an equilateral triangle with a line segment connecting the midpoints of two sides -/
structure EquilateralTriangleWithMidline where
  side_length : ℝ
  midline_position : ℝ

/-- Represents the configuration of two overlapping equilateral triangles -/
structure OverlappingTriangles where
  triangle_a : EquilateralTriangleWithMidline
  triangle_b : EquilateralTriangleWithMidline
  overlap_distance : ℝ

/-- Counts the number of triangles formed in a given configuration -/
def count_triangles (config : OverlappingTriangles) : ℕ :=
  sorry

/-- Finds the maximum number of triangles formed during the overlap process -/
def max_triangles (triangle : EquilateralTriangleWithMidline) : ℕ :=
  sorry

/-- Main theorem: The maximum number of triangles formed is 11 -/
theorem max_triangles_is_eleven (triangle : EquilateralTriangleWithMidline) :
  max_triangles triangle = 11 :=
sorry

end NUMINAMATH_CALUDE_max_triangles_is_eleven_l3125_312539


namespace NUMINAMATH_CALUDE_probability_is_correct_l3125_312580

def total_vehicles : ℕ := 20000
def shattered_windshields : ℕ := 600

def probability_shattered_windshield : ℚ :=
  shattered_windshields / total_vehicles

theorem probability_is_correct : 
  probability_shattered_windshield = 3 / 100 := by sorry

end NUMINAMATH_CALUDE_probability_is_correct_l3125_312580


namespace NUMINAMATH_CALUDE_unique_congruence_solution_l3125_312536

theorem unique_congruence_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ 1000000 [ZMOD 9] := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_solution_l3125_312536


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3125_312575

/-- An arithmetic sequence with first term 11, common difference 4, and last term 107 has 25 terms -/
theorem arithmetic_sequence_terms (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 11 → d = 4 → aₙ = 107 → aₙ = a₁ + (n - 1) * d → n = 25 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3125_312575


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l3125_312506

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem set_intersection_theorem : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l3125_312506


namespace NUMINAMATH_CALUDE_cabin_price_calculation_l3125_312531

/-- The price of Alfonso's cabin that Gloria wants to buy -/
def cabin_price : ℕ := sorry

/-- Gloria's initial cash -/
def initial_cash : ℕ := 150

/-- Number of cypress trees Gloria has -/
def cypress_trees : ℕ := 20

/-- Number of pine trees Gloria has -/
def pine_trees : ℕ := 600

/-- Number of maple trees Gloria has -/
def maple_trees : ℕ := 24

/-- Price per cypress tree -/
def cypress_price : ℕ := 100

/-- Price per pine tree -/
def pine_price : ℕ := 200

/-- Price per maple tree -/
def maple_price : ℕ := 300

/-- Amount Gloria wants to have left after buying the cabin -/
def leftover_amount : ℕ := 350

/-- Total amount Gloria can get from selling her trees and her initial cash -/
def total_amount : ℕ :=
  initial_cash +
  cypress_trees * cypress_price +
  pine_trees * pine_price +
  maple_trees * maple_price

theorem cabin_price_calculation :
  cabin_price = total_amount - leftover_amount :=
by sorry

end NUMINAMATH_CALUDE_cabin_price_calculation_l3125_312531


namespace NUMINAMATH_CALUDE_bus_passengers_l3125_312557

/-- The number of people initially on the bus -/
def initial_people : ℕ := 4

/-- The number of people who got on the bus at the stop -/
def people_who_got_on : ℕ := 13

/-- The total number of people on the bus after the stop -/
def total_people : ℕ := initial_people + people_who_got_on

theorem bus_passengers : total_people = 17 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l3125_312557


namespace NUMINAMATH_CALUDE_complex_number_real_condition_l3125_312584

theorem complex_number_real_condition (a : ℝ) :
  (∃ (z : ℂ), z = (a + 1) + (a^2 - 1) * I ∧ z.im = 0) ↔ (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_complex_number_real_condition_l3125_312584


namespace NUMINAMATH_CALUDE_divisibility_of_x_l3125_312586

theorem divisibility_of_x (x y : ℕ+) (h1 : 2 * x ^ 2 - 1 = y ^ 15) (h2 : x > 1) :
  5 ∣ x.val := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_x_l3125_312586


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l3125_312519

open Set

theorem inequality_solution_sets (a b : ℝ) :
  {x : ℝ | a * x - b < 0} = Ioi 1 →
  {x : ℝ | (a * x + b) * (x - 3) > 0} = Ioo (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l3125_312519


namespace NUMINAMATH_CALUDE_lisa_walking_speed_l3125_312591

/-- The number of meters Lisa walks per minute -/
def meters_per_minute (total_distance : ℕ) (days : ℕ) (hours_per_day : ℕ) (minutes_per_hour : ℕ) : ℚ :=
  (total_distance : ℚ) / (days * hours_per_day * minutes_per_hour)

/-- Proof that Lisa walks 10 meters per minute -/
theorem lisa_walking_speed :
  let total_distance := 1200
  let days := 2
  let hours_per_day := 1
  let minutes_per_hour := 60
  meters_per_minute total_distance days hours_per_day minutes_per_hour = 10 := by
  sorry

#eval meters_per_minute 1200 2 1 60

end NUMINAMATH_CALUDE_lisa_walking_speed_l3125_312591


namespace NUMINAMATH_CALUDE_work_equals_2pi_l3125_312592

/-- The force field F --/
def F (x y : ℝ) : ℝ × ℝ := (x - y, 1)

/-- The curve L --/
def L : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4 ∧ p.2 ≥ 0}

/-- Starting point --/
def M : ℝ × ℝ := (2, 0)

/-- Ending point --/
def N : ℝ × ℝ := (-2, 0)

/-- Work done by force F along curve L from M to N --/
noncomputable def work : ℝ := sorry

theorem work_equals_2pi : work = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_work_equals_2pi_l3125_312592


namespace NUMINAMATH_CALUDE_flagpole_shadow_length_l3125_312527

/-- Given a flagpole and a building under similar shadow-casting conditions,
    prove that the flagpole's shadow length is 45 meters. -/
theorem flagpole_shadow_length
  (flagpole_height : ℝ)
  (building_height : ℝ)
  (building_shadow : ℝ)
  (h1 : flagpole_height = 18)
  (h2 : building_height = 24)
  (h3 : building_shadow = 60)
  (h4 : flagpole_height / flagpole_shadow = building_height / building_shadow) :
  flagpole_shadow = 45 :=
by
  sorry


end NUMINAMATH_CALUDE_flagpole_shadow_length_l3125_312527


namespace NUMINAMATH_CALUDE_complementary_sets_count_l3125_312523

/-- Represents a card with four attributes -/
structure Card where
  shape : Fin 3
  color : Fin 3
  shade : Fin 3
  pattern : Fin 3

/-- The deck of all possible cards -/
def deck : Finset Card := sorry

/-- Checks if three cards form a complementary set -/
def isComplementary (c1 c2 c3 : Card) : Prop := sorry

/-- The set of all complementary three-card sets -/
def complementarySets : Finset (Finset Card) := sorry

theorem complementary_sets_count :
  deck.card = 81 ∧ (∀ c1 c2 : Card, c1 ∈ deck → c2 ∈ deck → c1 = c2 ∨ c1.shape ≠ c2.shape ∨ c1.color ≠ c2.color ∨ c1.shade ≠ c2.shade ∨ c1.pattern ≠ c2.pattern) →
  complementarySets.card = 5400 := by
  sorry

end NUMINAMATH_CALUDE_complementary_sets_count_l3125_312523


namespace NUMINAMATH_CALUDE_congruence_problem_l3125_312594

theorem congruence_problem : ∃ (n : ℤ), 0 ≤ n ∧ n < 25 ∧ -175 ≡ n [ZMOD 25] ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3125_312594


namespace NUMINAMATH_CALUDE_sets_intersection_and_union_l3125_312533

def A : Set ℝ := {x | (x - 2) * (x + 5) < 0}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}

theorem sets_intersection_and_union :
  (A ∩ B = {x : ℝ | -5 < x ∧ x ≤ -1}) ∧
  (A ∪ (Bᶜ) = {x : ℝ | -5 < x ∧ x < 3}) := by sorry

end NUMINAMATH_CALUDE_sets_intersection_and_union_l3125_312533


namespace NUMINAMATH_CALUDE_constant_altitude_triangle_l3125_312563

/-- Given an equilateral triangle and a line through its center, prove the existence of a triangle
    with constant altitude --/
theorem constant_altitude_triangle (a : ℝ) (m : ℝ) :
  let A : ℝ × ℝ := (0, Real.sqrt 3 * a)
  let B : ℝ × ℝ := (-a, 0)
  let C : ℝ × ℝ := (a, 0)
  let O : ℝ × ℝ := (0, Real.sqrt 3 * a / 3)
  let N : ℝ × ℝ := (0, Real.sqrt 3 * a / 3)
  let M : ℝ × ℝ := (-Real.sqrt 3 * a / (3 * m), 0)
  let AM := Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)
  let BN := Real.sqrt ((N.1 - B.1)^2 + (N.2 - B.2)^2)
  let MN := Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2)
  ∃ (D E F : ℝ × ℝ),
    let h := Real.sqrt 6 * a / 3
    (E.1 - D.1)^2 + (E.2 - D.2)^2 = MN^2 ∧
    (F.1 - D.1)^2 + (F.2 - D.2)^2 = AM^2 ∧
    (F.1 - E.1)^2 + (F.2 - E.2)^2 = BN^2 ∧
    2 * (abs ((F.2 - E.2) * D.1 + (E.1 - F.1) * D.2 + (F.1 * E.2 - E.1 * F.2)) / Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)) = h :=
by
  sorry


end NUMINAMATH_CALUDE_constant_altitude_triangle_l3125_312563


namespace NUMINAMATH_CALUDE_multinomial_expansion_terms_l3125_312585

/-- The number of terms in the simplified multinomial expansion of (x+y+z)^10 -/
def multinomial_terms : ℕ := 66

/-- Theorem stating that the number of terms in the simplified multinomial expansion of (x+y+z)^10 is 66 -/
theorem multinomial_expansion_terms :
  multinomial_terms = 66 := by sorry

end NUMINAMATH_CALUDE_multinomial_expansion_terms_l3125_312585


namespace NUMINAMATH_CALUDE_amandas_car_round_trip_time_l3125_312543

/-- Given that:
    1. The bus takes 40 minutes to drive 80 miles to the beach.
    2. Amanda's car takes five fewer minutes than the bus for the same trip.
    Prove that Amanda's car takes 70 minutes to make a round trip to the beach. -/
theorem amandas_car_round_trip_time :
  let bus_time : ℕ := 40
  let car_time_difference : ℕ := 5
  let car_one_way_time : ℕ := bus_time - car_time_difference
  car_one_way_time * 2 = 70 := by sorry

end NUMINAMATH_CALUDE_amandas_car_round_trip_time_l3125_312543


namespace NUMINAMATH_CALUDE_measles_cases_1990_l3125_312554

/-- Calculates the number of measles cases in a given year, assuming a linear decrease from 1970 to 2000 -/
def measlesCases (year : ℕ) : ℕ :=
  let initialYear : ℕ := 1970
  let finalYear : ℕ := 2000
  let initialCases : ℕ := 480000
  let finalCases : ℕ := 600
  let yearsPassed : ℕ := year - initialYear
  let totalYears : ℕ := finalYear - initialYear
  let totalDecrease : ℕ := initialCases - finalCases
  let yearlyDecrease : ℕ := totalDecrease / totalYears
  initialCases - (yearsPassed * yearlyDecrease)

theorem measles_cases_1990 : measlesCases 1990 = 160400 := by
  sorry

end NUMINAMATH_CALUDE_measles_cases_1990_l3125_312554


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l3125_312507

theorem rectangle_length_proof (square_perimeter : ℝ) (rectangle_width : ℝ) :
  square_perimeter = 256 →
  rectangle_width = 32 →
  (square_perimeter / 4) ^ 2 = 2 * (rectangle_width * (square_perimeter / 4)) →
  square_perimeter / 4 = 64 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_proof_l3125_312507


namespace NUMINAMATH_CALUDE_difference_c_minus_a_l3125_312540

theorem difference_c_minus_a (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 90) : 
  c - a = 90 := by
sorry

end NUMINAMATH_CALUDE_difference_c_minus_a_l3125_312540


namespace NUMINAMATH_CALUDE_square_difference_65_35_l3125_312530

theorem square_difference_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_65_35_l3125_312530


namespace NUMINAMATH_CALUDE_exp_properties_l3125_312513

-- Define the exponential function
noncomputable def Exp : ℝ → ℝ := Real.exp

-- Theorem statement
theorem exp_properties :
  (∀ (a b x : ℝ), Exp ((a + b) * x) = Exp (a * x) * Exp (b * x)) ∧
  (∀ (x : ℝ) (k : ℕ), Exp (k * x) = (Exp x) ^ k) := by
  sorry

end NUMINAMATH_CALUDE_exp_properties_l3125_312513


namespace NUMINAMATH_CALUDE_distance_by_sea_l3125_312546

/-- The distance traveled by sea is the difference between the total distance and the distance by land -/
theorem distance_by_sea (total_distance land_distance : ℕ) (h1 : total_distance = 601) (h2 : land_distance = 451) :
  total_distance - land_distance = 150 := by
  sorry

end NUMINAMATH_CALUDE_distance_by_sea_l3125_312546


namespace NUMINAMATH_CALUDE_cube_sum_of_sum_and_product_l3125_312548

theorem cube_sum_of_sum_and_product (x y : ℝ) 
  (h1 : x + y = 10) (h2 : x * y = 11) : x^3 + y^3 = 670 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_of_sum_and_product_l3125_312548


namespace NUMINAMATH_CALUDE_tree_height_difference_l3125_312599

theorem tree_height_difference :
  let apple_tree_height : ℚ := 53 / 4
  let cherry_tree_height : ℚ := 147 / 8
  cherry_tree_height - apple_tree_height = 41 / 8 := by sorry

end NUMINAMATH_CALUDE_tree_height_difference_l3125_312599


namespace NUMINAMATH_CALUDE_base5_to_base7_conversion_l3125_312529

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def base10ToBase7 (n : ℕ) : ℕ := sorry

/-- Represents the number 412 in base 5 -/
def num_base5 : ℕ := 412

/-- Represents the number 212 in base 7 -/
def num_base7 : ℕ := 212

theorem base5_to_base7_conversion :
  base10ToBase7 (base5ToBase10 num_base5) = num_base7 := by
  sorry

end NUMINAMATH_CALUDE_base5_to_base7_conversion_l3125_312529


namespace NUMINAMATH_CALUDE_last_two_digits_product_l3125_312547

theorem last_two_digits_product (n : ℕ) : 
  (n % 100 ≥ 10) →  -- Ensure it's a two-digit number
  (n % 5 = 0) →     -- Divisible by 5
  ((n / 10) % 10 + n % 10 = 12) →  -- Sum of last two digits is 12
  ((n / 10) % 10 * (n % 10) = 35) :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l3125_312547


namespace NUMINAMATH_CALUDE_matrix_determinant_solution_l3125_312512

theorem matrix_determinant_solution (b : ℝ) (hb : b ≠ 0) :
  let y : ℝ := -b / 2
  ∃ (y : ℝ), Matrix.det
    ![![y + b, y, y],
      ![y, y + b, y],
      ![y, y, y + b]] = 0 ↔ y = -b / 2 := by
sorry

end NUMINAMATH_CALUDE_matrix_determinant_solution_l3125_312512


namespace NUMINAMATH_CALUDE_photo_arrangements_l3125_312545

/-- Represents the number of people in each category -/
structure People where
  teacher : Nat
  boys : Nat
  girls : Nat

/-- The total number of people -/
def total_people (p : People) : Nat :=
  p.teacher + p.boys + p.girls

/-- The number of arrangements for the given conditions -/
def arrangements (p : People) : Nat :=
  -- We'll define this function without implementation
  sorry

/-- Theorem stating the number of arrangements for the given problem -/
theorem photo_arrangements :
  let p := People.mk 1 2 2
  arrangements p = 24 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_l3125_312545


namespace NUMINAMATH_CALUDE_watercolor_painting_distribution_l3125_312566

theorem watercolor_painting_distribution (total_paintings : ℕ) (paintings_per_room : ℕ) (num_rooms : ℕ) : 
  total_paintings = 32 → paintings_per_room = 8 → num_rooms * paintings_per_room = total_paintings → num_rooms = 4 := by
  sorry

end NUMINAMATH_CALUDE_watercolor_painting_distribution_l3125_312566


namespace NUMINAMATH_CALUDE_max_pieces_is_112_l3125_312524

/-- Represents the dimensions of a rectangular cake cut into square pieces -/
structure CakeDimensions where
  m : ℕ
  n : ℕ

/-- Calculates the number of interior pieces in a cake -/
def interiorPieces (d : CakeDimensions) : ℕ :=
  if d.m > 2 ∧ d.n > 2 then (d.m - 2) * (d.n - 2) else 0

/-- Calculates the number of exterior pieces in a cake -/
def exteriorPieces (d : CakeDimensions) : ℕ :=
  d.m * d.n - interiorPieces d

/-- Checks if the cake satisfies the condition that exterior pieces are twice the interior pieces -/
def satisfiesCondition (d : CakeDimensions) : Prop :=
  exteriorPieces d = 2 * interiorPieces d

/-- The theorem stating that the maximum number of pieces under the given conditions is 112 -/
theorem max_pieces_is_112 :
  ∃ d : CakeDimensions, satisfiesCondition d ∧
  ∀ d' : CakeDimensions, satisfiesCondition d' → d.m * d.n ≥ d'.m * d'.n ∧ d.m * d.n = 112 :=
sorry

end NUMINAMATH_CALUDE_max_pieces_is_112_l3125_312524


namespace NUMINAMATH_CALUDE_mark_change_factor_l3125_312572

theorem mark_change_factor (n : ℕ) (original_avg new_avg : ℚ) (h1 : n = 10) (h2 : original_avg = 80) (h3 : new_avg = 160) :
  ∃ (factor : ℚ), factor * (n * original_avg) = n * new_avg ∧ factor = 2 := by
  sorry

end NUMINAMATH_CALUDE_mark_change_factor_l3125_312572


namespace NUMINAMATH_CALUDE_number_problem_l3125_312590

theorem number_problem (x : ℝ) : 3 * (2 * x + 9) = 57 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3125_312590


namespace NUMINAMATH_CALUDE_updated_mean_l3125_312544

theorem updated_mean (n : ℕ) (original_mean : ℝ) (decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 9 →
  (n * original_mean - n * decrement) / n = 191 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_l3125_312544


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_fifth_l3125_312525

theorem opposite_of_negative_one_fifth :
  -(-(1/5 : ℚ)) = 1/5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_fifth_l3125_312525


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l3125_312503

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ = 8 * Real.cos θ - 6 * Real.sin θ

-- Define the Cartesian equation
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = 0

-- Theorem statement
theorem polar_to_cartesian :
  ∀ (x y ρ θ : ℝ),
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →  -- Conversion between polar and Cartesian coordinates
  polar_equation ρ θ ↔ cartesian_equation x y :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l3125_312503


namespace NUMINAMATH_CALUDE_buffet_meal_combinations_l3125_312581

theorem buffet_meal_combinations : 
  let meat_options : ℕ := 3
  let vegetable_options : ℕ := 5
  let dessert_options : ℕ := 5
  let meat_selections : ℕ := 1
  let vegetable_selections : ℕ := 3
  let dessert_selections : ℕ := 2
  (meat_options.choose meat_selections) * 
  (vegetable_options.choose vegetable_selections) * 
  (dessert_options.choose dessert_selections) = 300 := by
sorry

end NUMINAMATH_CALUDE_buffet_meal_combinations_l3125_312581


namespace NUMINAMATH_CALUDE_lucky_lucy_theorem_l3125_312589

/-- The expression with parentheses -/
def expr_with_parentheses (a b c d e : ℤ) : ℤ := a + (b - (c + (d - e)))

/-- The expression without parentheses -/
def expr_without_parentheses (a b c d e : ℤ) : ℤ := a + b - c + d - e

/-- The theorem stating that the expressions are equal when e = 8 -/
theorem lucky_lucy_theorem (a b c d : ℤ) (ha : a = 2) (hb : b = 4) (hc : c = 6) (hd : d = 8) :
  ∃ e : ℤ, expr_with_parentheses a b c d e = expr_without_parentheses a b c d e ∧ e = 8 := by
  sorry

end NUMINAMATH_CALUDE_lucky_lucy_theorem_l3125_312589


namespace NUMINAMATH_CALUDE_vector_problems_l3125_312515

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

theorem vector_problems :
  (∃ k : ℝ, (a.1 + k * c.1, a.2 + k * c.2) • (2 * b.1 - a.1, 2 * b.2 - a.2) = 0 → k = -11/18) ∧
  (∃ d : ℝ × ℝ, ∃ t : ℝ, d = (t * c.1, t * c.2) ∧ d.1^2 + d.2^2 = 34 → 
    d = (4 * Real.sqrt 2, Real.sqrt 2) ∨ d = (-4 * Real.sqrt 2, -Real.sqrt 2)) :=
by sorry

#check vector_problems

end NUMINAMATH_CALUDE_vector_problems_l3125_312515


namespace NUMINAMATH_CALUDE_x_plus_y_equals_two_l3125_312559

theorem x_plus_y_equals_two (x y : ℝ) 
  (hx : (x - 1)^2017 + 2013 * (x - 1) = -1)
  (hy : (y - 1)^2017 + 2013 * (y - 1) = 1) : 
  x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_two_l3125_312559


namespace NUMINAMATH_CALUDE_rain_probability_l3125_312509

/-- The probability of rain on both Monday and Tuesday given specific conditions -/
theorem rain_probability (p_monday : ℝ) (p_tuesday : ℝ) (p_tuesday_given_no_monday : ℝ) :
  p_monday = 0.4 →
  p_tuesday = 0.3 →
  p_tuesday_given_no_monday = 0.5 →
  p_monday * p_tuesday = 0.12 :=
by sorry

end NUMINAMATH_CALUDE_rain_probability_l3125_312509


namespace NUMINAMATH_CALUDE_parallelogram_area_calculation_l3125_312576

/-- The area of the parallelogram formed by vectors u and z -/
def parallelogramArea (u z : Fin 2 → ℝ) : ℝ :=
  |u 0 * z 1 - u 1 * z 0|

/-- The problem statement -/
theorem parallelogram_area_calculation :
  let u : Fin 2 → ℝ := ![3, 4]
  let z : Fin 2 → ℝ := ![8, -1]
  parallelogramArea u z = 35 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_calculation_l3125_312576


namespace NUMINAMATH_CALUDE_intersecting_lines_k_value_l3125_312595

/-- Given two lines p and q that intersect at a point, prove the value of k -/
theorem intersecting_lines_k_value (k : ℝ) : 
  let p : ℝ → ℝ := λ x => -2 * x + 3
  let q : ℝ → ℝ := λ x => k * x + 9
  (p 6 = -9) ∧ (q 6 = -9) → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_k_value_l3125_312595


namespace NUMINAMATH_CALUDE_green_knights_magical_fraction_l3125_312504

/-- Represents the fraction of knights of a certain color who are magical -/
structure MagicalFraction where
  numerator : ℕ
  denominator : ℕ
  denominator_pos : denominator > 0

/-- Represents the distribution of knights in the kingdom -/
structure KnightDistribution where
  green_fraction : Rat
  yellow_fraction : Rat
  magical_fraction : Rat
  green_magical : MagicalFraction
  yellow_magical : MagicalFraction
  green_fraction_valid : green_fraction = 3 / 8
  yellow_fraction_valid : yellow_fraction = 5 / 8
  fractions_sum_to_one : green_fraction + yellow_fraction = 1
  magical_fraction_valid : magical_fraction = 1 / 5
  green_thrice_yellow : green_magical.numerator * yellow_magical.denominator = 
                        3 * yellow_magical.numerator * green_magical.denominator

theorem green_knights_magical_fraction 
  (k : KnightDistribution) : k.green_magical = MagicalFraction.mk 12 35 (by norm_num) := by
  sorry

end NUMINAMATH_CALUDE_green_knights_magical_fraction_l3125_312504


namespace NUMINAMATH_CALUDE_evaluate_expression_l3125_312517

theorem evaluate_expression : (3^3)^2 + 1 = 730 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3125_312517


namespace NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l3125_312538

/-- The solution set of the inequality (x+a)/(x^2+4x+3) > 0 --/
def SolutionSet (a : ℝ) : Set ℝ :=
  {x | (x + a) / (x^2 + 4*x + 3) > 0}

/-- The theorem stating that if the solution set is {x | x > -3, x ≠ -1}, then a = 1 --/
theorem solution_set_implies_a_equals_one :
  (∃ a : ℝ, SolutionSet a = {x : ℝ | x > -3 ∧ x ≠ -1}) →
  (∃ a : ℝ, SolutionSet a = {x : ℝ | x > -3 ∧ x ≠ -1} ∧ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l3125_312538


namespace NUMINAMATH_CALUDE_abc_product_l3125_312502

theorem abc_product (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_sum : a + b + c = 30) 
  (h_eq : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (420 : ℚ) / (a * b * c) = 1) : 
  a * b * c = 450 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l3125_312502


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3125_312573

theorem p_necessary_not_sufficient_for_q :
  (∃ x : ℝ, x < 1 ∧ ¬(x^2 + x - 2 < 0)) ∧
  (∀ x : ℝ, x^2 + x - 2 < 0 → x < 1) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3125_312573


namespace NUMINAMATH_CALUDE_R_has_smallest_d_l3125_312579

/-- Represents a square with four labeled sides --/
structure Square where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- The four squares given in the problem --/
def P : Square := { a := 2, b := 3, c := 10, d := 8 }
def Q : Square := { a := 8, b := 1, c := 2, d := 6 }
def R : Square := { a := 4, b := 5, c := 7, d := 1 }
def S : Square := { a := 7, b := 6, c := 5, d := 3 }

/-- Theorem stating that R has the smallest d value among the squares --/
theorem R_has_smallest_d : 
  R.d ≤ P.d ∧ R.d ≤ Q.d ∧ R.d ≤ S.d ∧ 
  (R.d < P.d ∨ R.d < Q.d ∨ R.d < S.d) := by
  sorry

end NUMINAMATH_CALUDE_R_has_smallest_d_l3125_312579


namespace NUMINAMATH_CALUDE_equation_solution_l3125_312514

theorem equation_solution : 
  ∃ (x : ℝ), x ≠ -4 ∧ (-x^2 = (4*x + 2) / (x + 4)) ↔ (x = -1 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3125_312514


namespace NUMINAMATH_CALUDE_sum_of_first_n_naturals_l3125_312568

theorem sum_of_first_n_naturals (n : ℕ) (h : n = 23) : 
  (List.range n).sum = 276 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_n_naturals_l3125_312568


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3125_312541

theorem complex_equation_solution (a b : ℝ) :
  (1 + 2 * Complex.I) * a + b = 2 * Complex.I → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3125_312541


namespace NUMINAMATH_CALUDE_sandy_bought_six_fish_l3125_312528

/-- The number of fish Sandy bought -/
def fish_bought (initial final : ℕ) : ℕ := final - initial

/-- Proof that Sandy bought 6 fish -/
theorem sandy_bought_six_fish :
  let initial := 26
  let final := 32
  fish_bought initial final = 6 := by
  sorry

end NUMINAMATH_CALUDE_sandy_bought_six_fish_l3125_312528


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3125_312534

/-- Given a geometric sequence {aₙ} where a₂a₃a₄ = 1 and a₆a₇a₈ = 64, prove that a₅ = 2 -/
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)  -- a is the sequence
  (h1 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)  -- a is a geometric sequence
  (h2 : a 2 * a 3 * a 4 = 1)  -- Condition 1
  (h3 : a 6 * a 7 * a 8 = 64)  -- Condition 2
  : a 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3125_312534


namespace NUMINAMATH_CALUDE_cookie_cutter_sides_l3125_312560

/-- The number of sides on a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides on a square -/
def square_sides : ℕ := 4

/-- The number of sides on a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of triangle-shaped cookie cutters -/
def num_triangles : ℕ := 6

/-- The number of square-shaped cookie cutters -/
def num_squares : ℕ := 4

/-- The number of hexagon-shaped cookie cutters -/
def num_hexagons : ℕ := 2

/-- The total number of sides on all cookie cutters -/
def total_sides : ℕ := num_triangles * triangle_sides + num_squares * square_sides + num_hexagons * hexagon_sides

theorem cookie_cutter_sides : total_sides = 46 := by
  sorry

end NUMINAMATH_CALUDE_cookie_cutter_sides_l3125_312560


namespace NUMINAMATH_CALUDE_chess_club_mixed_groups_l3125_312510

/-- Represents the chess club structure and game information -/
structure ChessClub where
  total_children : ℕ
  total_groups : ℕ
  children_per_group : ℕ
  boy_boy_games : ℕ
  girl_girl_games : ℕ

/-- Calculates the number of mixed groups in the chess club -/
def mixed_groups (club : ChessClub) : ℕ :=
  let total_games := club.total_groups * (club.children_per_group.choose 2)
  let mixed_games := total_games - club.boy_boy_games - club.girl_girl_games
  mixed_games / 2

/-- Theorem stating the number of mixed groups in the given scenario -/
theorem chess_club_mixed_groups :
  let club : ChessClub := {
    total_children := 90,
    total_groups := 30,
    children_per_group := 3,
    boy_boy_games := 30,
    girl_girl_games := 14
  }
  mixed_groups club = 23 := by sorry

end NUMINAMATH_CALUDE_chess_club_mixed_groups_l3125_312510


namespace NUMINAMATH_CALUDE_exists_non_increasing_f_l3125_312582

theorem exists_non_increasing_f :
  ∃ a : ℝ, a < 0 ∧
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧
  let f := fun x => a * x + Real.log x
  f x₁ ≥ f x₂ :=
sorry

end NUMINAMATH_CALUDE_exists_non_increasing_f_l3125_312582


namespace NUMINAMATH_CALUDE_parallelogram_area_l3125_312553

/-- The area of a parallelogram with base 20 meters and height 4 meters is 80 square meters. -/
theorem parallelogram_area : 
  let base : ℝ := 20
  let height : ℝ := 4
  let area := base * height
  area = 80 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3125_312553


namespace NUMINAMATH_CALUDE_lizzy_money_theorem_l3125_312520

def lizzy_money_problem (mother_gift uncle_gift father_gift candy_cost : ℕ) : Prop :=
  let initial_amount := mother_gift + father_gift
  let amount_after_spending := initial_amount - candy_cost
  let final_amount := amount_after_spending + uncle_gift
  final_amount = 140

theorem lizzy_money_theorem :
  lizzy_money_problem 80 70 40 50 := by
  sorry

end NUMINAMATH_CALUDE_lizzy_money_theorem_l3125_312520


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l3125_312549

/-- The intersection point of two lines with given angles of inclination -/
theorem intersection_point_of_lines (m n : ℝ → ℝ) (k₁ k₂ : ℝ) :
  (∀ x, m x = k₁ * x + 2) →
  (∀ x, n x = k₂ * x + Real.sqrt 3 + 1) →
  k₁ = Real.tan (π / 4) →
  k₂ = Real.tan (π / 3) →
  ∃ x y, m x = n x ∧ m x = y ∧ x = -1 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l3125_312549


namespace NUMINAMATH_CALUDE_floor_sqrt_26_squared_l3125_312597

theorem floor_sqrt_26_squared : ⌊Real.sqrt 26⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_26_squared_l3125_312597


namespace NUMINAMATH_CALUDE_no_upper_bound_for_positive_second_order_ratio_increasing_l3125_312532

open Set Real

-- Define the type for functions from (0, +∞) to ℝ
def PosRealFunc := { f : ℝ → ℝ // ∀ x, x > 0 → f x ≠ 0 }

-- Define second-order ratio increasing function
def SecondOrderRatioIncreasing (f : PosRealFunc) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f.val x / x^2 < f.val y / y^2

-- Define the theorem
theorem no_upper_bound_for_positive_second_order_ratio_increasing
  (f : PosRealFunc)
  (h1 : SecondOrderRatioIncreasing f)
  (h2 : ∀ x, x > 0 → f.val x > 0) :
  ¬∃ k, ∀ x, x > 0 → f.val x < k :=
sorry

end NUMINAMATH_CALUDE_no_upper_bound_for_positive_second_order_ratio_increasing_l3125_312532


namespace NUMINAMATH_CALUDE_glitched_clock_correct_time_l3125_312535

/-- Represents a 12-hour digital clock with a glitch that displays 7 instead of 5 -/
structure GlitchedClock where
  hours : Fin 12
  minutes : Fin 60

/-- Checks if a given hour is displayed correctly -/
def correctHour (h : Fin 12) : Bool :=
  h ≠ 5

/-- Checks if a given minute is displayed correctly -/
def correctMinute (m : Fin 60) : Bool :=
  m % 10 ≠ 5 ∧ m / 10 ≠ 5

/-- Calculates the fraction of the day the clock shows the correct time -/
def fractionCorrect : ℚ :=
  (11 : ℚ) / 12 * (54 : ℚ) / 60

theorem glitched_clock_correct_time :
  fractionCorrect = 33 / 40 := by
  sorry

#eval fractionCorrect

end NUMINAMATH_CALUDE_glitched_clock_correct_time_l3125_312535


namespace NUMINAMATH_CALUDE_negative495_terminates_as_225_l3125_312571

-- Define the set of possible answers
inductive PossibleAnswer
  | angle135  : PossibleAnswer
  | angle45   : PossibleAnswer
  | angle225  : PossibleAnswer
  | angleNeg225 : PossibleAnswer

-- Define a function to convert PossibleAnswer to real number (in degrees)
def toRealDegrees (a : PossibleAnswer) : ℝ :=
  match a with
  | PossibleAnswer.angle135   => 135
  | PossibleAnswer.angle45    => 45
  | PossibleAnswer.angle225   => 225
  | PossibleAnswer.angleNeg225 => -225

-- Define what it means for two angles to terminate in the same direction
def terminatesSameDirection (a b : ℝ) : Prop :=
  ∃ k : ℤ, a - b = 360 * (k : ℝ)

-- State the theorem
theorem negative495_terminates_as_225 :
  ∃ (answer : PossibleAnswer), terminatesSameDirection (-495) (toRealDegrees answer) ∧
  answer = PossibleAnswer.angle225 :=
sorry

end NUMINAMATH_CALUDE_negative495_terminates_as_225_l3125_312571


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3125_312550

theorem simplify_square_roots : 
  Real.sqrt 726 / Real.sqrt 242 + Real.sqrt 484 / Real.sqrt 121 = Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3125_312550


namespace NUMINAMATH_CALUDE_equation_solution_l3125_312598

theorem equation_solution :
  ∃ x : ℝ, (3639 + 11.95 - x^2 = 3054) ∧ (abs (x - 24.43) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3125_312598


namespace NUMINAMATH_CALUDE_angle_equality_l3125_312551

-- Define what it means for two angles to be vertical
def are_vertical_angles (A B : ℝ) : Prop := sorry

-- State the theorem that vertical angles are equal
axiom vertical_angles_are_equal : ∀ A B : ℝ, are_vertical_angles A B → A = B

-- The statement to be proved
theorem angle_equality (A B : ℝ) (h : are_vertical_angles A B) : A = B := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l3125_312551


namespace NUMINAMATH_CALUDE_different_color_chip_probability_l3125_312587

theorem different_color_chip_probability :
  let total_chips : ℕ := 12
  let red_chips : ℕ := 7
  let green_chips : ℕ := 5
  let prob_red : ℚ := red_chips / total_chips
  let prob_green : ℚ := green_chips / total_chips
  let prob_different_colors : ℚ := prob_red * prob_green + prob_green * prob_red
  prob_different_colors = 35 / 72 := by sorry

end NUMINAMATH_CALUDE_different_color_chip_probability_l3125_312587


namespace NUMINAMATH_CALUDE_total_insect_legs_l3125_312552

/-- The number of insects in the laboratory -/
def num_insects : ℕ := 6

/-- The number of legs per insect -/
def legs_per_insect : ℕ := 6

/-- Theorem: The total number of insect legs in the laboratory is 36 -/
theorem total_insect_legs : num_insects * legs_per_insect = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_insect_legs_l3125_312552


namespace NUMINAMATH_CALUDE_shortest_rope_part_l3125_312511

theorem shortest_rope_part (total_length : ℝ) (ratio1 ratio2 ratio3 : ℝ) 
  (h1 : total_length = 196.85)
  (h2 : ratio1 = 3.6)
  (h3 : ratio2 = 8.4)
  (h4 : ratio3 = 12) :
  let total_ratio := ratio1 + ratio2 + ratio3
  let shortest_part := (total_length / total_ratio) * ratio1
  shortest_part = 29.5275 := by
sorry

end NUMINAMATH_CALUDE_shortest_rope_part_l3125_312511


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l3125_312583

/-- A quadratic function -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- A linear function -/
def LinearFunction (m n : ℝ) : ℝ → ℝ := λ x ↦ m * x + n

/-- The theorem statement -/
theorem quadratic_function_theorem (a b c m n : ℝ) :
  let f := QuadraticFunction a b c
  let g := LinearFunction m n
  (f (-1) = 2) ∧ (g (-1) = 2) ∧ (f 2 = 5) ∧ (g 2 = 5) ∧
  (∃ x₀, ∀ x, f x₀ ≤ f x) ∧ (f x₀ = 1) →
  (f = λ x ↦ x^2 + 1) ∨ (f = λ x ↦ (1/9) * x^2 + (8/9) * x + 25/9) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l3125_312583


namespace NUMINAMATH_CALUDE_hexagon_coloring_count_l3125_312556

/-- Represents a coloring of the hexagon's outer disks -/
def Coloring := Fin 6 → Bool

/-- The group of symmetries of a regular hexagon -/
def HexagonSymmetry := Fin 12

/-- Checks if a coloring is fixed by a given symmetry -/
def is_fixed (c : Coloring) (s : HexagonSymmetry) : Bool :=
  sorry

/-- Counts the number of colorings fixed by a given symmetry -/
def fixed_count (s : HexagonSymmetry) : ℕ :=
  sorry

/-- The number of distinct colorings -/
def distinct_colorings : ℕ :=
  sorry

theorem hexagon_coloring_count : distinct_colorings = 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_coloring_count_l3125_312556


namespace NUMINAMATH_CALUDE_fraction_value_decreases_as_denominator_increases_l3125_312526

theorem fraction_value_decreases_as_denominator_increases 
  (ability : ℝ) (self_estimation : ℝ → ℝ) :
  ability > 0 → (∀ x y, x > 0 ∧ y > 0 ∧ x < y → self_estimation x > self_estimation y) →
  ∀ x y, x > 0 ∧ y > 0 ∧ x < y → ability / self_estimation x > ability / self_estimation y :=
sorry

end NUMINAMATH_CALUDE_fraction_value_decreases_as_denominator_increases_l3125_312526


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_odds_l3125_312564

theorem largest_of_three_consecutive_odds (a b c : ℤ) : 
  (a + b + c = 75) →  -- sum is 75
  (c - a = 4) →       -- difference between largest and smallest is 4
  (Odd a ∧ Odd b ∧ Odd c) →  -- all numbers are odd
  (b = a + 2 ∧ c = b + 2) →  -- numbers are consecutive
  (c = 27) :=         -- largest number is 27
by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_odds_l3125_312564


namespace NUMINAMATH_CALUDE_housing_development_l3125_312574

theorem housing_development (total : ℕ) (garage : ℕ) (pool : ℕ) (neither : ℕ) 
  (h_total : total = 90)
  (h_garage : garage = 50)
  (h_pool : pool = 40)
  (h_neither : neither = 35) :
  garage + pool - (total - neither) = 35 := by
  sorry

end NUMINAMATH_CALUDE_housing_development_l3125_312574


namespace NUMINAMATH_CALUDE_circle_radius_proof_l3125_312558

theorem circle_radius_proof (chord_length : ℝ) (center_to_intersection : ℝ) (ratio_left : ℝ) (ratio_right : ℝ) :
  chord_length = 18 →
  center_to_intersection = 7 →
  ratio_left = 2 * ratio_right →
  ratio_left + ratio_right = chord_length →
  ∃ (radius : ℝ), radius = 11 ∧ 
    (radius - center_to_intersection) * (radius + center_to_intersection) = ratio_left * ratio_right :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l3125_312558


namespace NUMINAMATH_CALUDE_rotational_function_example_rotational_function_value_rotational_function_symmetry_l3125_312569

/-- Definition of rotational functions -/
def rotational_functions (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₁ + a₂ = 0 ∧ b₁ = b₂ ∧ c₁ + c₂ = 0

/-- The rotational function of y = x² - 4x + 3 -/
theorem rotational_function_example :
  rotational_functions 1 (-4) 3 (-1) (-4) (-3) := by sorry

/-- If y = 5x² + (m+1)x + n and y = -5x² - nx - 3 are rotational functions, then (m+n)^2023 = -1 -/
theorem rotational_function_value (m n : ℝ) :
  rotational_functions 5 (m+1) n (-5) (-n) (-3) → (m+n)^2023 = -1 := by sorry

/-- The rotational function of y = 2(x-1)(x+3) passes through (-1,0), (3,0), and (0,6) -/
theorem rotational_function_symmetry :
  ∃ a b c : ℝ, rotational_functions 2 4 (-6) a b c ∧
  a*(-1)^2 + b*(-1) + c = 0 ∧
  a*3^2 + b*3 + c = 0 ∧
  c = 6 := by sorry

end NUMINAMATH_CALUDE_rotational_function_example_rotational_function_value_rotational_function_symmetry_l3125_312569


namespace NUMINAMATH_CALUDE_smallest_sausage_packages_l3125_312521

theorem smallest_sausage_packages (sausage_pack : ℕ) (bun_pack : ℕ) 
  (h1 : sausage_pack = 10) (h2 : bun_pack = 15) :
  ∃ n : ℕ, n > 0 ∧ sausage_pack * n % bun_pack = 0 ∧ 
  ∀ m : ℕ, m > 0 → sausage_pack * m % bun_pack = 0 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_sausage_packages_l3125_312521


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3125_312561

/-- The asymptote equation of a hyperbola with specific properties -/
theorem hyperbola_asymptote (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_focal : 2 * Real.sqrt (a^2 + b^2) = Real.sqrt 3 * (2 * a)) :
  ∃ (k : ℝ), k = Real.sqrt 2 ∧ 
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → y = k * x ∨ y = -k * x) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3125_312561


namespace NUMINAMATH_CALUDE_is_centre_of_hyperbola_l3125_312522

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 18 * x - 16 * y^2 + 64 * y - 143 = 0

/-- The centre of the hyperbola -/
def hyperbola_centre : ℝ × ℝ := (1, 2)

/-- Theorem stating that the given point is the centre of the hyperbola -/
theorem is_centre_of_hyperbola :
  let (h, k) := hyperbola_centre
  ∀ (a b : ℝ), hyperbola_equation (h + a) (k + b) ↔ hyperbola_equation (h - a) (k - b) :=
by sorry

end NUMINAMATH_CALUDE_is_centre_of_hyperbola_l3125_312522


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3125_312596

theorem isosceles_triangle_perimeter (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) →
  (∃ (base leg : ℝ), 
    (base^2 - 6*base + 8 = 0) ∧ 
    (leg^2 - 6*leg + 8 = 0) ∧
    (base ≠ leg) ∧
    (base + 2*leg = 10)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3125_312596


namespace NUMINAMATH_CALUDE_expression_simplification_l3125_312570

theorem expression_simplification (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  (x + 2) * (y - 2) - 2 * (x * y - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3125_312570


namespace NUMINAMATH_CALUDE_log_equation_l3125_312562

theorem log_equation : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_l3125_312562


namespace NUMINAMATH_CALUDE_total_oranges_count_l3125_312588

def initial_oranges : ℕ := 5
def received_oranges : ℕ := 3
def bought_skittles : ℕ := 9

theorem total_oranges_count : 
  initial_oranges + received_oranges = 8 :=
by sorry

end NUMINAMATH_CALUDE_total_oranges_count_l3125_312588


namespace NUMINAMATH_CALUDE_x_power_twelve_equals_one_l3125_312577

theorem x_power_twelve_equals_one (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_power_twelve_equals_one_l3125_312577


namespace NUMINAMATH_CALUDE_second_polygon_sides_l3125_312542

/-- 
Given two regular polygons with the same perimeter, where the first polygon has 24 sides
and a side length that is three times as long as the second polygon,
prove that the second polygon has 72 sides.
-/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 → 
  24 * (3 * s) = n * s → 
  n = 72 :=
by sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l3125_312542


namespace NUMINAMATH_CALUDE_sqrt_sum_floor_equality_l3125_312516

theorem sqrt_sum_floor_equality (n : ℤ) : 
  ⌊Real.sqrt (n : ℝ) + Real.sqrt ((n + 1) : ℝ)⌋ = ⌊Real.sqrt ((4 * n + 2) : ℝ)⌋ :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_floor_equality_l3125_312516


namespace NUMINAMATH_CALUDE_lasagna_mince_amount_l3125_312565

/-- Proves that the amount of ground mince used for each lasagna is 2 pounds -/
theorem lasagna_mince_amount 
  (total_dishes : ℕ) 
  (cottage_pie_mince : ℕ) 
  (total_mince : ℕ) 
  (cottage_pies : ℕ) 
  (h1 : total_dishes = 100)
  (h2 : cottage_pie_mince = 3)
  (h3 : total_mince = 500)
  (h4 : cottage_pies = 100) :
  (total_mince - cottage_pies * cottage_pie_mince) / (total_dishes - cottage_pies) = 2 := by
  sorry

#check lasagna_mince_amount

end NUMINAMATH_CALUDE_lasagna_mince_amount_l3125_312565


namespace NUMINAMATH_CALUDE_astronomical_unit_scientific_notation_l3125_312537

/-- One astronomical unit in kilometers -/
def astronomical_unit : ℝ := 1.496e9

/-- Scientific notation representation of one astronomical unit -/
def astronomical_unit_scientific : ℝ := 1.496 * 10^8

/-- Theorem stating that the astronomical unit can be expressed in scientific notation -/
theorem astronomical_unit_scientific_notation :
  astronomical_unit = astronomical_unit_scientific := by
  sorry

end NUMINAMATH_CALUDE_astronomical_unit_scientific_notation_l3125_312537


namespace NUMINAMATH_CALUDE_crayons_per_child_l3125_312518

/-- Given a group of children with crayons, prove that each child has 12 crayons. -/
theorem crayons_per_child (total_children : ℕ) (total_crayons : ℕ) 
  (h1 : total_children = 18) 
  (h2 : total_crayons = 216) : 
  total_crayons / total_children = 12 := by
  sorry

#check crayons_per_child

end NUMINAMATH_CALUDE_crayons_per_child_l3125_312518


namespace NUMINAMATH_CALUDE_no_square_root_representation_l3125_312593

theorem no_square_root_representation : ¬ ∃ (A B : ℤ), (A + B * Real.sqrt 3) ^ 2 = 99999 + 111111 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_no_square_root_representation_l3125_312593


namespace NUMINAMATH_CALUDE_subtraction_of_negatives_l3125_312500

theorem subtraction_of_negatives : -1 - 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_negatives_l3125_312500


namespace NUMINAMATH_CALUDE_carpet_width_l3125_312505

/-- Proves that a rectangular carpet covering 75% of a 48 sq ft room with a length of 9 ft has a width of 4 ft -/
theorem carpet_width (room_area : ℝ) (carpet_length : ℝ) (coverage_percent : ℝ) :
  room_area = 48 →
  carpet_length = 9 →
  coverage_percent = 0.75 →
  (room_area * coverage_percent) / carpet_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_carpet_width_l3125_312505


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l3125_312508

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : 
  r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l3125_312508


namespace NUMINAMATH_CALUDE_seven_at_eight_equals_28_div_9_l3125_312501

/-- The '@' operation for positive integers -/
def at_op (a b : ℕ+) : ℚ :=
  (a.val * b.val : ℚ) / (a.val + b.val + 3 : ℚ)

/-- Theorem: 7 @ 8 = 28/9 -/
theorem seven_at_eight_equals_28_div_9 : 
  at_op ⟨7, by norm_num⟩ ⟨8, by norm_num⟩ = 28 / 9 := by
  sorry

end NUMINAMATH_CALUDE_seven_at_eight_equals_28_div_9_l3125_312501
