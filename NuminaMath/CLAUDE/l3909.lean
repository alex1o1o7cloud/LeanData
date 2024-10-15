import Mathlib

namespace NUMINAMATH_CALUDE_midpoint_after_translation_l3909_390934

/-- Given points A, J, and H in a 2D coordinate system, and a translation vector,
    prove that the midpoint of A'H' after translation is as specified. -/
theorem midpoint_after_translation (A J H : ℝ × ℝ) (translation : ℝ × ℝ) :
  A = (3, 3) →
  J = (4, 8) →
  H = (7, 3) →
  translation = (-6, 3) →
  let A' := (A.1 + translation.1, A.2 + translation.2)
  let H' := (H.1 + translation.1, H.2 + translation.2)
  ((A'.1 + H'.1) / 2, (A'.2 + H'.2) / 2) = (-1, 6) := by
  sorry

end NUMINAMATH_CALUDE_midpoint_after_translation_l3909_390934


namespace NUMINAMATH_CALUDE_three_digit_permutations_l3909_390920

/-- The set of digits used in the problem -/
def digits : Finset Nat := {1, 2, 3}

/-- The number of digits used -/
def n : Nat := Finset.card digits

/-- The length of each permutation -/
def k : Nat := 3

/-- The number of permutations of the digits -/
def num_permutations : Nat := Nat.factorial n

theorem three_digit_permutations : num_permutations = 6 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_permutations_l3909_390920


namespace NUMINAMATH_CALUDE_expression_factorization_l3909_390911

theorem expression_factorization (x : ℝ) :
  (4 * x^3 + 64 * x^2 - 8) - (-6 * x^3 + 2 * x^2 - 8) = 2 * x^2 * (5 * x + 31) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l3909_390911


namespace NUMINAMATH_CALUDE_necessary_condition_for_inequality_l3909_390949

theorem necessary_condition_for_inequality (a b : ℝ) :
  a * Real.sqrt a + b * Real.sqrt b > a * Real.sqrt b + b * Real.sqrt a →
  a ≥ 0 ∧ b ≥ 0 ∧ a ≠ b :=
by sorry

end NUMINAMATH_CALUDE_necessary_condition_for_inequality_l3909_390949


namespace NUMINAMATH_CALUDE_book_cost_price_l3909_390954

theorem book_cost_price (final_price : ℝ) (profit_rate : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) 
  (h1 : final_price = 250)
  (h2 : profit_rate = 0.20)
  (h3 : tax_rate = 0.10)
  (h4 : discount_rate = 0.05) : 
  ∃ (cost_price : ℝ), cost_price = final_price / ((1 + profit_rate) * (1 - discount_rate) * (1 + tax_rate)) :=
by sorry

end NUMINAMATH_CALUDE_book_cost_price_l3909_390954


namespace NUMINAMATH_CALUDE_square_binomial_equality_l3909_390926

theorem square_binomial_equality (a b : ℝ) : 
  (-1 + a * b^2)^2 = 1 - 2 * a * b^2 + a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_equality_l3909_390926


namespace NUMINAMATH_CALUDE_difference_of_sums_l3909_390999

def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_5 (x : ℕ) : ℕ :=
  5 * ((x + 2) / 5)

def sum_rounded_to_5 (n : ℕ) : ℕ :=
  (n / 5) * (0 + 5 + 5 + 5 + 10)

theorem difference_of_sums (n : ℕ) (h : n = 200) : 
  (sum_to_n n) - (sum_rounded_to_5 n) = 19100 := by
  sorry

#check difference_of_sums

end NUMINAMATH_CALUDE_difference_of_sums_l3909_390999


namespace NUMINAMATH_CALUDE_largest_centrally_symmetric_polygon_area_l3909_390919

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A polygon in a 2D plane --/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Checks if a polygon is centrally symmetric --/
def isCentrallySymmetric (p : Polygon) : Prop := sorry

/-- Checks if a polygon is inside a triangle --/
def isInsideTriangle (p : Polygon) (t : Triangle) : Prop := sorry

/-- Calculates the area of a polygon --/
def area (p : Polygon) : ℝ := sorry

/-- Calculates the area of a triangle --/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The theorem to be proved --/
theorem largest_centrally_symmetric_polygon_area (t : Triangle) : 
  ∃ (p : Polygon), 
    isCentrallySymmetric p ∧ 
    isInsideTriangle p t ∧ 
    (∀ (q : Polygon), isCentrallySymmetric q → isInsideTriangle q t → area q ≤ area p) ∧
    area p = (2/3) * triangleArea t := by
  sorry

end NUMINAMATH_CALUDE_largest_centrally_symmetric_polygon_area_l3909_390919


namespace NUMINAMATH_CALUDE_lcm_24_36_45_l3909_390960

theorem lcm_24_36_45 : Nat.lcm (Nat.lcm 24 36) 45 = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_36_45_l3909_390960


namespace NUMINAMATH_CALUDE_carla_initial_marbles_l3909_390935

/-- The number of marbles Carla bought -/
def marbles_bought : ℕ := 134

/-- The number of marbles Carla has now -/
def marbles_now : ℕ := 187

/-- The number of marbles Carla started with -/
def marbles_start : ℕ := marbles_now - marbles_bought

theorem carla_initial_marbles : marbles_start = 53 := by
  sorry

end NUMINAMATH_CALUDE_carla_initial_marbles_l3909_390935


namespace NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l3909_390986

/-- Represents a ratio of three ingredients -/
structure Ratio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original ratio of ingredients -/
def original_ratio : Ratio :=
  { flour := 10, water := 6, sugar := 3 }

/-- The new ratio after adjusting flour to water and flour to sugar -/
def new_ratio : Ratio :=
  { flour := 20, water := 6, sugar := 12 }

/-- The amount of water in the new recipe -/
def new_water_amount : ℚ := 2

theorem sugar_amount_in_new_recipe :
  let sugar_amount := (new_ratio.sugar / new_ratio.water) * new_water_amount
  sugar_amount = 4 := by sorry

end NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l3909_390986


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3909_390957

/-- The minimum distance from the origin to a point on the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : 
  let line := {(x, y) : ℝ × ℝ | x + y = 4}
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧ 
    ∀ (P : ℝ × ℝ), P ∈ line → 
      Real.sqrt (P.1^2 + P.2^2) ≥ d ∧ 
      ∃ (Q : ℝ × ℝ), Q ∈ line ∧ Real.sqrt (Q.1^2 + Q.2^2) = d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3909_390957


namespace NUMINAMATH_CALUDE_joyce_farmland_l3909_390964

/-- Calculates the area of land suitable for growing vegetables given the size of the previous property, 
    the factor by which the new property is larger, and the size of a pond on the new property. -/
def land_for_vegetables (prev_property : ℝ) (size_factor : ℝ) (pond_size : ℝ) : ℝ :=
  prev_property * size_factor - pond_size

/-- Theorem stating that given a previous property of 2 acres, a new property 10 times larger, 
    and a 1-acre pond, the land suitable for growing vegetables is 19 acres. -/
theorem joyce_farmland : land_for_vegetables 2 10 1 = 19 := by
  sorry

end NUMINAMATH_CALUDE_joyce_farmland_l3909_390964


namespace NUMINAMATH_CALUDE_base_seven_addition_l3909_390906

/-- Given a base 7 addition problem 5XY₇ + 52₇ = 62X₇, prove that X + Y = 6 in base 10 -/
theorem base_seven_addition (X Y : ℕ) : 
  (5 * 7^2 + X * 7 + Y) + (5 * 7 + 2) = 6 * 7^2 + 2 * 7 + X → X + Y = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_addition_l3909_390906


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3909_390991

theorem simplify_trig_expression :
  let tan_45 : ℝ := 1
  let cot_45 : ℝ := 1
  (tan_45^3 + cot_45^3) / (tan_45 + cot_45) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3909_390991


namespace NUMINAMATH_CALUDE_cistern_filling_time_l3909_390922

theorem cistern_filling_time (empty_rate : ℝ) (combined_fill_time : ℝ) (fill_time : ℝ) : 
  empty_rate = 1 / 6 →
  combined_fill_time = 30 →
  1 / fill_time - empty_rate = 1 / combined_fill_time →
  fill_time = 5 := by
sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l3909_390922


namespace NUMINAMATH_CALUDE_range_of_a_l3909_390901

theorem range_of_a (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + 4 = 4*x*y) :
  (∀ a : ℝ, x*y + 1/2*a^2*x + a^2*y + a - 17 ≥ 0) ↔ 
  (∀ a : ℝ, a ≤ -3 ∨ a ≥ 5/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3909_390901


namespace NUMINAMATH_CALUDE_kite_diagonal_sum_less_than_largest_sides_sum_l3909_390902

/-- A kite is a quadrilateral with two pairs of adjacent sides of equal length -/
structure Kite where
  sides : Fin 4 → ℝ
  diagonals : Fin 2 → ℝ
  side_positive : ∀ i, sides i > 0
  diagonal_positive : ∀ i, diagonals i > 0
  adjacent_equal : sides 0 = sides 1 ∧ sides 2 = sides 3

theorem kite_diagonal_sum_less_than_largest_sides_sum (k : Kite) :
  k.diagonals 0 + k.diagonals 1 < 
  (max (k.sides 0) (k.sides 2)) + (max (k.sides 1) (k.sides 3)) + 
  (min (max (k.sides 0) (k.sides 2)) (max (k.sides 1) (k.sides 3))) :=
sorry

end NUMINAMATH_CALUDE_kite_diagonal_sum_less_than_largest_sides_sum_l3909_390902


namespace NUMINAMATH_CALUDE_consecutive_even_integers_l3909_390995

theorem consecutive_even_integers (a b c d e : ℤ) : 
  (∃ n : ℤ, a = n ∧ b = n + 2 ∧ c = n + 4 ∧ d = n + 6 ∧ e = n + 8) →  -- five consecutive even integers
  (a + e = 204) →  -- sum of first and last is 204
  (a + b + c + d + e = 510) ∧ (a = 98)  -- sum is 510 and smallest is 98
  := by sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_l3909_390995


namespace NUMINAMATH_CALUDE_lineup_combinations_l3909_390940

def total_players : ℕ := 15
def selected_players : ℕ := 2
def lineup_size : ℕ := 5

theorem lineup_combinations :
  Nat.choose (total_players - selected_players) (lineup_size - selected_players) = 286 := by
  sorry

end NUMINAMATH_CALUDE_lineup_combinations_l3909_390940


namespace NUMINAMATH_CALUDE_gym_attendance_proof_l3909_390924

theorem gym_attendance_proof (W A S : ℕ) : 
  (W + A + S) + 8 = 30 → W + A + S = 22 := by
  sorry

end NUMINAMATH_CALUDE_gym_attendance_proof_l3909_390924


namespace NUMINAMATH_CALUDE_beach_house_pool_problem_l3909_390923

theorem beach_house_pool_problem (total_people : ℕ) (legs_in_pool : ℕ) (legs_per_person : ℕ) :
  total_people = 26 →
  legs_in_pool = 34 →
  legs_per_person = 2 →
  total_people - (legs_in_pool / legs_per_person) = 9 := by
sorry

end NUMINAMATH_CALUDE_beach_house_pool_problem_l3909_390923


namespace NUMINAMATH_CALUDE_prism_with_hole_volume_formula_l3909_390925

/-- The volume of a rectangular prism with a hole running through it -/
def prism_with_hole_volume (x : ℝ) : ℝ :=
  let large_prism_volume := (x + 8) * (x + 6) * 4
  let hole_volume := (2*x - 4) * (x - 3) * 4
  large_prism_volume - hole_volume

/-- Theorem stating the volume of the prism with a hole -/
theorem prism_with_hole_volume_formula (x : ℝ) :
  prism_with_hole_volume x = -4*x^2 + 96*x + 144 :=
by sorry

end NUMINAMATH_CALUDE_prism_with_hole_volume_formula_l3909_390925


namespace NUMINAMATH_CALUDE_mikes_books_l3909_390985

/-- Mike's book counting problem -/
theorem mikes_books (initial_books bought_books : ℕ) :
  initial_books = 35 →
  bought_books = 56 →
  initial_books + bought_books = 91 := by
  sorry

end NUMINAMATH_CALUDE_mikes_books_l3909_390985


namespace NUMINAMATH_CALUDE_slope_characterization_l3909_390930

/-- The set of possible slopes for a line with y-intercept (0,3) intersecting the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt (16/405) ∨ m ≥ Real.sqrt (16/405)}

/-- The equation of the line with slope m and y-intercept (0,3) -/
def line_equation (m : ℝ) (x : ℝ) : ℝ := m * x + 3

/-- The equation of the ellipse 4x^2 + 25y^2 = 100 -/
def ellipse_equation (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

/-- Theorem stating that the set of possible slopes for a line with y-intercept (0,3) 
    intersecting the ellipse 4x^2 + 25y^2 = 100 is (-∞, -√(16/405)] ∪ [√(16/405), ∞) -/
theorem slope_characterization :
  ∀ m : ℝ, (∃ x : ℝ, ellipse_equation x (line_equation m x)) ↔ m ∈ possible_slopes := by
  sorry

end NUMINAMATH_CALUDE_slope_characterization_l3909_390930


namespace NUMINAMATH_CALUDE_fencing_calculation_l3909_390962

/-- The total fencing length for a square playground and a rectangular garden -/
def total_fencing (playground_side : ℝ) (garden_length garden_width : ℝ) : ℝ :=
  4 * playground_side + 2 * (garden_length + garden_width)

/-- Theorem: The total fencing for a playground with side 27 yards and a garden of 12 by 9 yards is 150 yards -/
theorem fencing_calculation :
  total_fencing 27 12 9 = 150 := by
  sorry

end NUMINAMATH_CALUDE_fencing_calculation_l3909_390962


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l3909_390909

theorem floor_ceiling_sum : ⌊(0.999 : ℝ)⌋ + ⌈(2.001 : ℝ)⌉ = 3 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l3909_390909


namespace NUMINAMATH_CALUDE_car_repair_cost_l3909_390965

theorem car_repair_cost (total_cost : ℝ) (num_parts : ℕ) (labor_rate : ℝ) (work_hours : ℝ)
  (h1 : total_cost = 220)
  (h2 : num_parts = 2)
  (h3 : labor_rate = 0.5)
  (h4 : work_hours = 6) :
  (total_cost - labor_rate * work_hours * 60) / num_parts = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_repair_cost_l3909_390965


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l3909_390997

theorem empty_solution_set_range (k : ℝ) : 
  (∀ x : ℝ, ¬(k * x^2 + 2 * k * x + 2 < 0)) ↔ (0 ≤ k ∧ k ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l3909_390997


namespace NUMINAMATH_CALUDE_weight_of_barium_iodide_l3909_390992

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of moles of Barium iodide -/
def moles_BaI2 : ℝ := 4

/-- The molecular weight of Barium iodide (BaI2) in g/mol -/
def molecular_weight_BaI2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_I

/-- The total weight of Barium iodide in grams -/
def total_weight_BaI2 : ℝ := moles_BaI2 * molecular_weight_BaI2

theorem weight_of_barium_iodide :
  total_weight_BaI2 = 1564.52 := by sorry

end NUMINAMATH_CALUDE_weight_of_barium_iodide_l3909_390992


namespace NUMINAMATH_CALUDE_circle_radius_with_inscribed_dodecagon_l3909_390951

theorem circle_radius_with_inscribed_dodecagon (Q : ℝ) (R : ℝ) : 
  (R > 0) → 
  (π * R^2 = Q + 3 * R^2) → 
  R = Real.sqrt (Q / (π - 3)) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_with_inscribed_dodecagon_l3909_390951


namespace NUMINAMATH_CALUDE_largest_k_for_inequality_l3909_390937

theorem largest_k_for_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_3 : a + b + c = 3) :
  (∀ k : ℝ, 0 < k → k ≤ 5 → a^3 + b^3 + c^3 - 3 ≥ k * (3 - a*b - b*c - c*a)) ∧
  (∃ a₀ b₀ c₀ : ℝ, 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ + b₀ + c₀ = 3 ∧
    a₀^3 + b₀^3 + c₀^3 - 3 = 5 * (3 - a₀*b₀ - b₀*c₀ - c₀*a₀)) :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_inequality_l3909_390937


namespace NUMINAMATH_CALUDE_other_denomination_is_70_l3909_390955

/-- Proves that the other denomination of travelers checks is $70 --/
theorem other_denomination_is_70 
  (total_checks : ℕ)
  (total_worth : ℕ)
  (known_denomination : ℕ)
  (known_count : ℕ)
  (remaining_average : ℕ)
  (h1 : total_checks = 30)
  (h2 : total_worth = 1800)
  (h3 : known_denomination = 50)
  (h4 : known_count = 15)
  (h5 : remaining_average = 70)
  (h6 : known_count * known_denomination + (total_checks - known_count) * remaining_average = total_worth) :
  ∃ (other_denomination : ℕ), other_denomination = 70 ∧ 
    known_count * known_denomination + (total_checks - known_count) * other_denomination = total_worth :=
by sorry

end NUMINAMATH_CALUDE_other_denomination_is_70_l3909_390955


namespace NUMINAMATH_CALUDE_dormitory_allocation_l3909_390917

/-- The number of ways to assign n students to two dormitories with at least k students in each -/
def allocation_schemes (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items -/
def choose (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of ways to arrange r items from n items -/
def arrange (n : ℕ) (r : ℕ) : ℕ := sorry

theorem dormitory_allocation :
  allocation_schemes 7 2 = 112 :=
by
  sorry

#check dormitory_allocation

end NUMINAMATH_CALUDE_dormitory_allocation_l3909_390917


namespace NUMINAMATH_CALUDE_necessary_to_sufficient_contrapositive_l3909_390904

theorem necessary_to_sufficient_contrapositive (p q : Prop) :
  (q → p) → (¬p → ¬q) :=
by
  sorry

end NUMINAMATH_CALUDE_necessary_to_sufficient_contrapositive_l3909_390904


namespace NUMINAMATH_CALUDE_room_length_calculation_l3909_390973

/-- Given a room with width 2.75 m and a floor paving cost of 600 per sq. metre
    resulting in a total cost of 10725, the length of the room is 6.5 meters. -/
theorem room_length_calculation (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  width = 2.75 ∧ cost_per_sqm = 600 ∧ total_cost = 10725 →
  total_cost = (6.5 * width * cost_per_sqm) :=
by sorry

end NUMINAMATH_CALUDE_room_length_calculation_l3909_390973


namespace NUMINAMATH_CALUDE_left_movement_denoted_negative_l3909_390905

/-- Represents the direction of movement -/
inductive Direction
| Left
| Right

/-- Represents a movement with a distance and direction -/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Denotes a movement as a signed real number -/
def denoteMovement (m : Movement) : ℝ :=
  match m.direction with
  | Direction.Right => m.distance
  | Direction.Left => -m.distance

theorem left_movement_denoted_negative (d : ℝ) (h : d > 0) :
  denoteMovement { distance := d, direction := Direction.Right } = d →
  denoteMovement { distance := d, direction := Direction.Left } = -d :=
by
  sorry

#check left_movement_denoted_negative

end NUMINAMATH_CALUDE_left_movement_denoted_negative_l3909_390905


namespace NUMINAMATH_CALUDE_perp_to_same_plane_implies_parallel_perp_to_two_planes_implies_parallel_l3909_390969

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perp : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Define non-coincidence
variable (non_coincident_lines : Line → Line → Prop)
variable (non_coincident_planes : Plane → Plane → Prop)

-- Theorem 1: If two lines are perpendicular to the same plane, then they are parallel
theorem perp_to_same_plane_implies_parallel 
  (a b : Line) (α : Plane) 
  (h1 : non_coincident_lines a b) 
  (h2 : perp a α) (h3 : perp b α) : 
  parallel a b := by sorry

-- Theorem 2: If a line is perpendicular to two planes, then those planes are parallel
theorem perp_to_two_planes_implies_parallel 
  (a : Line) (α β : Plane) 
  (h1 : non_coincident_planes α β) 
  (h2 : perp a α) (h3 : perp a β) : 
  plane_parallel α β := by sorry

end NUMINAMATH_CALUDE_perp_to_same_plane_implies_parallel_perp_to_two_planes_implies_parallel_l3909_390969


namespace NUMINAMATH_CALUDE_platform_walk_probability_l3909_390998

/-- The number of platforms at the train station -/
def num_platforms : ℕ := 16

/-- The distance between adjacent platforms in feet -/
def platform_distance : ℕ := 200

/-- The maximum walking distance we're interested in -/
def max_walk_distance : ℕ := 800

/-- The probability of walking 800 feet or less between two randomly assigned platforms -/
theorem platform_walk_probability : 
  let total_assignments := num_platforms * (num_platforms - 1)
  let favorable_assignments := 
    (2 * 4 * 8) +  -- Edge platforms (1-4 and 13-16) have 8 choices each
    (8 * 10)       -- Central platforms (5-12) have 10 choices each
  (favorable_assignments : ℚ) / total_assignments = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_platform_walk_probability_l3909_390998


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l3909_390976

theorem polynomial_multiplication (x : ℝ) : (x + 1) * (x^2 - x + 1) = x^3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l3909_390976


namespace NUMINAMATH_CALUDE_one_minus_repeating_third_equals_two_thirds_l3909_390978

def repeating_decimal_one_third : ℚ := 1/3

theorem one_minus_repeating_third_equals_two_thirds :
  1 - repeating_decimal_one_third = 2/3 := by sorry

end NUMINAMATH_CALUDE_one_minus_repeating_third_equals_two_thirds_l3909_390978


namespace NUMINAMATH_CALUDE_fraction_integrality_l3909_390979

theorem fraction_integrality (a b c : ℤ) 
  (h : ∃ (n : ℤ), (a * b : ℚ) / c + (a * c : ℚ) / b + (b * c : ℚ) / a = n) :
  (∃ (n1 : ℤ), (a * b : ℚ) / c = n1) ∧ 
  (∃ (n2 : ℤ), (a * c : ℚ) / b = n2) ∧ 
  (∃ (n3 : ℤ), (b * c : ℚ) / a = n3) := by
sorry

end NUMINAMATH_CALUDE_fraction_integrality_l3909_390979


namespace NUMINAMATH_CALUDE_jungkook_money_l3909_390928

def initial_amount (notebook_cost pencil_cost remaining : ℕ) : Prop :=
  ∃ (total : ℕ),
    notebook_cost = total / 2 ∧
    pencil_cost = (total - notebook_cost) / 2 ∧
    remaining = total - notebook_cost - pencil_cost ∧
    remaining = 750

theorem jungkook_money : 
  ∀ (notebook_cost pencil_cost remaining : ℕ),
    initial_amount notebook_cost pencil_cost remaining →
    ∃ (total : ℕ), total = 3000 :=
by
  sorry

end NUMINAMATH_CALUDE_jungkook_money_l3909_390928


namespace NUMINAMATH_CALUDE_crayons_count_l3909_390972

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 41

/-- The number of crayons Sam added to the drawer -/
def added_crayons : ℕ := 12

/-- The total number of crayons after Sam's addition -/
def total_crayons : ℕ := initial_crayons + added_crayons

theorem crayons_count : total_crayons = 53 := by
  sorry

end NUMINAMATH_CALUDE_crayons_count_l3909_390972


namespace NUMINAMATH_CALUDE_school_averages_l3909_390933

theorem school_averages 
  (J L : ℕ) -- Number of boys at Jefferson and Lincoln
  (j l : ℕ) -- Number of girls at Jefferson and Lincoln
  (h1 : (68 * J + 73 * j) / (J + j) = 70) -- Jefferson combined average
  (h2 : (68 * J + 78 * L) / (J + L) = 76) -- Boys combined average
  (h3 : J = (3 * j) / 2) -- Derived from h1
  (h4 : J = L) -- Derived from h2
  (h5 : l = j) -- Assumption of equal girls at both schools
  : ((73 * j + 85 * l) / (j + l) = 79) ∧ 
    ((78 * L + 85 * l) / (L + l) = 808/10) :=
by sorry

end NUMINAMATH_CALUDE_school_averages_l3909_390933


namespace NUMINAMATH_CALUDE_sum_of_eight_smallest_multiples_of_12_l3909_390970

theorem sum_of_eight_smallest_multiples_of_12 : 
  (Finset.range 8).sum (λ i => 12 * (i + 1)) = 432 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_eight_smallest_multiples_of_12_l3909_390970


namespace NUMINAMATH_CALUDE_batsman_running_percentage_l3909_390950

def batsman_score_percentage (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) : ℚ :=
  let boundary_runs := 4 * boundaries
  let six_runs := 6 * sixes
  let runs_without_running := boundary_runs + six_runs
  let runs_by_running := total_runs - runs_without_running
  (runs_by_running : ℚ) / total_runs * 100

theorem batsman_running_percentage :
  batsman_score_percentage 120 3 8 = 50 := by sorry

end NUMINAMATH_CALUDE_batsman_running_percentage_l3909_390950


namespace NUMINAMATH_CALUDE_water_added_to_container_l3909_390966

theorem water_added_to_container (capacity initial_percentage final_fraction : ℝ) 
  (h1 : capacity = 80)
  (h2 : initial_percentage = 0.30)
  (h3 : final_fraction = 3/4) : 
  capacity * (final_fraction - initial_percentage) = 36 := by
sorry

end NUMINAMATH_CALUDE_water_added_to_container_l3909_390966


namespace NUMINAMATH_CALUDE_right_triangle_vector_relation_l3909_390993

/-- Given a right triangle ABC with ∠C = 90°, vector AB = (t, 1), and vector AC = (2, 3), prove that t = 5 -/
theorem right_triangle_vector_relation (t : ℝ) : 
  let A : ℝ × ℝ := (0, 0)  -- Assuming A is at the origin for simplicity
  let B : ℝ × ℝ := (t, 1)
  let C : ℝ × ℝ := (2, 3)
  let AB : ℝ × ℝ := (t - 0, 1 - 0)  -- Vector from A to B
  let AC : ℝ × ℝ := (2 - 0, 3 - 0)  -- Vector from A to C
  let BC : ℝ × ℝ := (2 - t, 3 - 1)  -- Vector from B to C
  (AC.1 * BC.1 + AC.2 * BC.2 = 0) →  -- Dot product of AC and BC is 0 (perpendicular)
  t = 5 := by
sorry


end NUMINAMATH_CALUDE_right_triangle_vector_relation_l3909_390993


namespace NUMINAMATH_CALUDE_two_circles_k_value_l3909_390990

/-- Two circles centered at the origin with given properties --/
structure TwoCircles where
  -- Radius of the larger circle
  R : ℝ
  -- Radius of the smaller circle
  r : ℝ
  -- Point P on the larger circle
  P : ℝ × ℝ
  -- Point S on the smaller circle
  S : ℝ × ℝ
  -- Distance QR
  QR : ℝ
  -- Conditions
  center_origin : True
  P_on_larger : P.1^2 + P.2^2 = R^2
  S_on_smaller : S.1^2 + S.2^2 = r^2
  S_on_y_axis : S.1 = 0
  radius_difference : R - r = QR

/-- Theorem stating the value of k for the given two circles --/
theorem two_circles_k_value (c : TwoCircles) (h1 : c.P = (10, 2)) (h2 : c.QR = 5) :
  ∃ k : ℝ, c.S = (0, k) ∧ (k = Real.sqrt 104 - 5 ∨ k = -(Real.sqrt 104 - 5)) := by
  sorry

end NUMINAMATH_CALUDE_two_circles_k_value_l3909_390990


namespace NUMINAMATH_CALUDE_xy_value_l3909_390938

theorem xy_value (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : x * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3909_390938


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l3909_390903

theorem unique_positive_integer_solution : ∃! (x : ℕ), x > 0 ∧ (4 * x)^2 - 3 * x = 1764 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l3909_390903


namespace NUMINAMATH_CALUDE_product_inequality_l3909_390941

theorem product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3909_390941


namespace NUMINAMATH_CALUDE_motel_rent_theorem_l3909_390982

/-- Represents the total rent charged by a motel on a Saturday night. -/
def TotalRent : ℕ → ℕ → ℕ 
  | r50, r60 => 50 * r50 + 60 * r60

/-- Represents the condition that changing 10 rooms from $60 to $50 reduces the rent by 25%. -/
def RentReductionCondition (r50 r60 : ℕ) : Prop :=
  4 * (TotalRent (r50 + 10) (r60 - 10)) = 3 * (TotalRent r50 r60)

theorem motel_rent_theorem :
  ∃ (r50 r60 : ℕ), RentReductionCondition r50 r60 ∧ TotalRent r50 r60 = 400 :=
sorry

end NUMINAMATH_CALUDE_motel_rent_theorem_l3909_390982


namespace NUMINAMATH_CALUDE_no_duplicates_on_diagonal_l3909_390980

/-- Represents a symmetric table with specific properties -/
structure SymmetricTable :=
  (size : Nat)
  (values : Fin size → Fin size → Fin size)
  (symmetric : ∀ i j, values i j = values j i)
  (distinct_rows : ∀ i j k, j ≠ k → values i j ≠ values i k)

/-- The main theorem stating that there are no duplicate numbers on the diagonal of symmetry -/
theorem no_duplicates_on_diagonal (t : SymmetricTable) (h : t.size = 101) :
  ∀ i j, i ≠ j → t.values i i ≠ t.values j j := by
  sorry

end NUMINAMATH_CALUDE_no_duplicates_on_diagonal_l3909_390980


namespace NUMINAMATH_CALUDE_five_people_six_chairs_l3909_390912

/-- The number of ways to arrange n people in m chairs in a row -/
def arrangements (n m : ℕ) : ℕ :=
  (m.factorial) / ((m - n).factorial)

/-- Theorem: There are 720 ways to arrange 5 people in a row of 6 chairs -/
theorem five_people_six_chairs : arrangements 5 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_five_people_six_chairs_l3909_390912


namespace NUMINAMATH_CALUDE_tan_315_degrees_l3909_390961

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l3909_390961


namespace NUMINAMATH_CALUDE_exhibition_arrangements_l3909_390936

def total_arrangements (n : ℕ) : ℕ := n.factorial

def adjacent_arrangements (n : ℕ) : ℕ := (n - 1).factorial * 2

theorem exhibition_arrangements :
  let n := 4
  let total := total_arrangements n
  let adjacent := adjacent_arrangements n
  total - adjacent = 12 := by sorry

end NUMINAMATH_CALUDE_exhibition_arrangements_l3909_390936


namespace NUMINAMATH_CALUDE_difference_of_roots_quadratic_l3909_390943

theorem difference_of_roots_quadratic (x : ℝ) : 
  let roots := {r : ℝ | r^2 - 9*r + 14 = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ |r₁ - r₂| = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_difference_of_roots_quadratic_l3909_390943


namespace NUMINAMATH_CALUDE_left_handed_jazz_lovers_count_l3909_390931

/-- Represents a club with members of different handedness and music preferences -/
structure Club where
  total : Nat
  leftHanded : Nat
  jazzLovers : Nat
  rightHandedNonJazz : Nat

/-- The number of left-handed jazz lovers in the club -/
def leftHandedJazzLovers (c : Club) : Nat :=
  c.leftHanded + c.jazzLovers - (c.total - c.rightHandedNonJazz)

/-- Theorem stating the number of left-handed jazz lovers in the specific club scenario -/
theorem left_handed_jazz_lovers_count (c : Club) 
  (h1 : c.total = 25)
  (h2 : c.leftHanded = 10)
  (h3 : c.jazzLovers = 18)
  (h4 : c.rightHandedNonJazz = 4) :
  leftHandedJazzLovers c = 7 := by
  sorry

#check left_handed_jazz_lovers_count

end NUMINAMATH_CALUDE_left_handed_jazz_lovers_count_l3909_390931


namespace NUMINAMATH_CALUDE_triangle_properties_l3909_390921

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.c / (t.b - t.a) = (Real.sin t.A + Real.sin t.B) / (Real.sin t.A + Real.sin t.C))
  (h2 : Real.sin t.C = 2 * Real.sin t.A)
  (h3 : 1/2 * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3)
  (h4 : t.b = Real.sqrt 3)
  (h5 : t.a * t.c = 1) : 
  t.B = 2 * Real.pi / 3 ∧ 
  t.a = 2 ∧ 
  t.c = 4 ∧ 
  t.a + t.b + t.c = 2 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3909_390921


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3909_390983

theorem sphere_surface_area (d : ℝ) (h : d = 4) : 
  (π * d^2) = 16 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3909_390983


namespace NUMINAMATH_CALUDE_larger_number_is_ten_l3909_390981

theorem larger_number_is_ten (x y : ℕ) (h1 : x * y = 30) (h2 : x + y = 13) : 
  max x y = 10 := by
sorry

end NUMINAMATH_CALUDE_larger_number_is_ten_l3909_390981


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_79_l3909_390987

theorem gcd_of_powers_of_79 : 
  Nat.Prime 79 → Nat.gcd (79^7 + 1) (79^7 + 79^2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_79_l3909_390987


namespace NUMINAMATH_CALUDE_parallel_perpendicular_implication_l3909_390974

/-- Two lines are parallel -/
def parallel (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular (l : Line) (α : Plane) : Prop := sorry

/-- Theorem: If two lines are parallel and one is perpendicular to a plane, 
    then the other is also perpendicular to that plane -/
theorem parallel_perpendicular_implication 
  (m n : Line) (α : Plane) : 
  parallel m n → perpendicular m α → perpendicular n α := by
  sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_implication_l3909_390974


namespace NUMINAMATH_CALUDE_total_books_l3909_390945

-- Define the number of books for each person
def betty_books (x : ℚ) : ℚ := x

def sister_books (x : ℚ) : ℚ := x + (1/4) * x

def cousin_books (x : ℚ) : ℚ := 2 * (sister_books x)

def friend_books (x y : ℚ) : ℚ := 
  betty_books x + sister_books x + cousin_books x - y

-- Theorem statement
theorem total_books (x y : ℚ) : 
  betty_books x + sister_books x + cousin_books x + friend_books x y = (19/2) * x - y := by
  sorry

end NUMINAMATH_CALUDE_total_books_l3909_390945


namespace NUMINAMATH_CALUDE_max_additional_bricks_l3909_390989

/-- Represents the weight capacity of a truck in terms of bags of sand -/
def sand_capacity : ℕ := 50

/-- Represents the weight capacity of a truck in terms of bricks -/
def brick_capacity : ℕ := 400

/-- Represents the number of bags of sand already in the truck -/
def sand_load : ℕ := 32

/-- Calculates the equivalent number of bricks for a given number of sand bags -/
def sand_to_brick_equiv (sand : ℕ) : ℕ :=
  (brick_capacity * sand) / sand_capacity

theorem max_additional_bricks : 
  sand_to_brick_equiv (sand_capacity - sand_load) = 144 := by
  sorry

end NUMINAMATH_CALUDE_max_additional_bricks_l3909_390989


namespace NUMINAMATH_CALUDE_vitamin_a_weekly_pills_l3909_390916

/-- The number of pills needed to meet the weekly recommended amount of Vitamin A -/
def pills_needed (vitamin_per_pill : ℕ) (daily_recommended : ℕ) (days_per_week : ℕ) : ℕ :=
  (daily_recommended * days_per_week) / vitamin_per_pill

/-- Proof that 28 pills are needed per week to meet the recommended Vitamin A intake -/
theorem vitamin_a_weekly_pills : pills_needed 50 200 7 = 28 := by
  sorry

end NUMINAMATH_CALUDE_vitamin_a_weekly_pills_l3909_390916


namespace NUMINAMATH_CALUDE_orange_trees_l3909_390910

theorem orange_trees (total_fruits : ℕ) (fruits_per_tree : ℕ) (remaining_ratio : ℚ) : 
  total_fruits = 960 →
  fruits_per_tree = 200 →
  remaining_ratio = 3/5 →
  (total_fruits : ℚ) / (remaining_ratio * fruits_per_tree) = 8 :=
by sorry

end NUMINAMATH_CALUDE_orange_trees_l3909_390910


namespace NUMINAMATH_CALUDE_prime_divisor_form_l3909_390953

theorem prime_divisor_form (p q : ℕ) (hp : Prime p) (hp2 : p > 2) (hq : Prime q) 
  (hdiv : q ∣ (2^p - 1)) : ∃ k : ℕ, q = 2*k*p + 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisor_form_l3909_390953


namespace NUMINAMATH_CALUDE_pages_ratio_day2_to_day1_l3909_390996

/-- Represents the number of pages read on each day --/
structure DailyPages where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Theorem stating the ratio of pages read on day 2 to day 1 --/
theorem pages_ratio_day2_to_day1 (pages : DailyPages) : 
  pages.day1 = 63 →
  pages.day3 = pages.day2 + 10 →
  pages.day4 = 29 →
  pages.day1 + pages.day2 + pages.day3 + pages.day4 = 354 →
  pages.day2 / pages.day1 = 2 := by
  sorry

#check pages_ratio_day2_to_day1

end NUMINAMATH_CALUDE_pages_ratio_day2_to_day1_l3909_390996


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l3909_390942

theorem solve_system_of_equations (x y : ℝ) : 
  (2 * x - y = 12) → (x = 5) → (y = -2) := by
  sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l3909_390942


namespace NUMINAMATH_CALUDE_almond_walnut_ratio_l3909_390918

/-- Given a mixture of almonds and walnuts, prove the ratio of almonds to walnuts -/
theorem almond_walnut_ratio 
  (total_weight : ℝ) 
  (almond_weight : ℝ) 
  (almond_parts : ℕ) 
  (h1 : total_weight = 280) 
  (h2 : almond_weight = 200) 
  (h3 : almond_parts = 5) :
  ∃ (walnut_parts : ℕ), 
    (almond_parts : ℝ) / walnut_parts = 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_almond_walnut_ratio_l3909_390918


namespace NUMINAMATH_CALUDE_lisa_income_percentage_l3909_390944

-- Define variables for incomes
variable (T : ℝ) -- Tim's income
variable (M : ℝ) -- Mary's income
variable (J : ℝ) -- Juan's income
variable (L : ℝ) -- Lisa's income

-- Define the conditions
variable (h1 : M = 1.60 * T) -- Mary's income is 60% more than Tim's
variable (h2 : T = 0.50 * J) -- Tim's income is 50% less than Juan's
variable (h3 : L = 1.30 * M) -- Lisa's income is 30% more than Mary's
variable (h4 : L = 0.75 * J) -- Lisa's income is 25% less than Juan's

-- Define the theorem
theorem lisa_income_percentage :
  (L / (M + J)) * 100 = 41.67 :=
sorry

end NUMINAMATH_CALUDE_lisa_income_percentage_l3909_390944


namespace NUMINAMATH_CALUDE_sum_of_tens_digits_l3909_390956

/-- Given single-digit numbers A, B, C, D such that A + B + C + D = 22,
    the sum of the tens digits of (A + B) and (C + D) equals 4. -/
theorem sum_of_tens_digits (A B C D : ℕ) (h1 : A < 10) (h2 : B < 10) (h3 : C < 10) (h4 : D < 10)
    (h5 : A + B + C + D = 22) : (A + B) / 10 + (C + D) / 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tens_digits_l3909_390956


namespace NUMINAMATH_CALUDE_alligator_count_l3909_390913

theorem alligator_count (crocodiles vipers total : ℕ) 
  (h1 : crocodiles = 22)
  (h2 : vipers = 5)
  (h3 : total = 50)
  (h4 : ∃ alligators : ℕ, crocodiles + alligators + vipers = total) :
  ∃ alligators : ℕ, alligators = 23 ∧ crocodiles + alligators + vipers = total :=
by sorry

end NUMINAMATH_CALUDE_alligator_count_l3909_390913


namespace NUMINAMATH_CALUDE_triangle_side_difference_l3909_390994

theorem triangle_side_difference (x : ℕ) : 
  (x > 0) →
  (x + 10 > 8) →
  (x + 8 > 10) →
  (10 + 8 > x) →
  (∃ (max min : ℕ), 
    (∀ y : ℕ, (y > 0 ∧ y + 10 > 8 ∧ y + 8 > 10 ∧ 10 + 8 > y) → y ≤ max ∧ y ≥ min) ∧
    (max - min = 14)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l3909_390994


namespace NUMINAMATH_CALUDE_angelina_walking_speed_l3909_390948

/-- Angelina's walking problem -/
theorem angelina_walking_speed 
  (distance_home_grocery : ℝ) 
  (distance_grocery_gym : ℝ) 
  (speed_home_grocery : ℝ) 
  (speed_grocery_gym : ℝ) 
  (time_difference : ℝ) :
  distance_home_grocery = 150 →
  distance_grocery_gym = 200 →
  speed_grocery_gym = 2 * speed_home_grocery →
  distance_home_grocery / speed_home_grocery - 
    distance_grocery_gym / speed_grocery_gym = time_difference →
  time_difference = 10 →
  speed_grocery_gym = 10 := by
sorry


end NUMINAMATH_CALUDE_angelina_walking_speed_l3909_390948


namespace NUMINAMATH_CALUDE_air_density_scientific_notation_l3909_390971

/-- The mass per unit volume of air in grams per cubic centimeter -/
def air_density : ℝ := 0.00124

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a positive real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem air_density_scientific_notation :
  to_scientific_notation air_density = ScientificNotation.mk 1.24 (-3) sorry :=
sorry

end NUMINAMATH_CALUDE_air_density_scientific_notation_l3909_390971


namespace NUMINAMATH_CALUDE_conic_section_is_hyperbola_l3909_390915

/-- The equation (x+7)^2 = (5y-6)^2 + 125 defines a hyperbola -/
theorem conic_section_is_hyperbola :
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
  ∀ (x y : ℝ), (x + 7)^2 = (5*y - 6)^2 + 125 ↔ a*x^2 + b*y^2 + c*x + d*y + e*x*y + f = 0 :=
by sorry

end NUMINAMATH_CALUDE_conic_section_is_hyperbola_l3909_390915


namespace NUMINAMATH_CALUDE_smallest_valid_n_l3909_390927

def is_valid (n : ℕ) : Prop :=
  Real.sqrt (n : ℝ) - Real.sqrt ((n - 1) : ℝ) < 0.01

theorem smallest_valid_n :
  (∀ m : ℕ, m < 2501 → ¬(is_valid m)) ∧ is_valid 2501 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l3909_390927


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3909_390963

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b ∧ b > 1 → log a 3 < log b 3) ∧
  (∃ a b, log a 3 < log b 3 ∧ ¬(a > b ∧ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3909_390963


namespace NUMINAMATH_CALUDE_local_max_at_two_l3909_390977

def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

theorem local_max_at_two (c : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), f c x ≤ f c 2) →
  c = 6 := by
sorry

end NUMINAMATH_CALUDE_local_max_at_two_l3909_390977


namespace NUMINAMATH_CALUDE_odd_function_sum_l3909_390952

/-- A function is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_sum (a b c : ℝ) (f : ℝ → ℝ) :
  IsOdd f →
  (∀ x, f x = x^2 * Real.sin x + c - 3) →
  (∀ x, x ∈ Set.Icc (a + 2) b → f x ≠ 0) →
  b > a + 2 →
  a + b + c = 1 := by sorry

end NUMINAMATH_CALUDE_odd_function_sum_l3909_390952


namespace NUMINAMATH_CALUDE_redo_profit_is_5000_l3909_390968

/-- Calculates the profit for Redo's horseshoe manufacturing --/
def calculate_profit (initial_outlay : ℕ) (cost_per_set : ℕ) (selling_price : ℕ) (num_sets : ℕ) : ℤ :=
  let revenue := num_sets * selling_price
  let manufacturing_costs := initial_outlay + (cost_per_set * num_sets)
  (revenue : ℤ) - manufacturing_costs

/-- Proves that the profit for Redo's horseshoe manufacturing is $5,000 --/
theorem redo_profit_is_5000 :
  calculate_profit 10000 20 50 500 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_redo_profit_is_5000_l3909_390968


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3909_390967

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) → 
  a ∈ Set.Iio 1 ∪ Set.Ioi 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3909_390967


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3909_390946

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (((43 - 2*x) ^ (1/4) : ℝ) + ((37 + 2*x) ^ (1/4) : ℝ) = 4) ↔ (x = -19 ∨ x = 21) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3909_390946


namespace NUMINAMATH_CALUDE_pump_fill_time_solution_l3909_390900

def pump_fill_time (P : ℝ) : Prop :=
  P > 0 ∧ (1 / P - 1 / 14 = 3 / 7)

theorem pump_fill_time_solution :
  ∃ P, pump_fill_time P ∧ P = 2 := by sorry

end NUMINAMATH_CALUDE_pump_fill_time_solution_l3909_390900


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3909_390907

theorem polynomial_factorization (a : ℝ) : 
  (a^2 - 4*a + 2) * (a^2 - 4*a + 6) + 4 = (a - 2)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3909_390907


namespace NUMINAMATH_CALUDE_total_weight_is_410_l3909_390908

/-- The number of A4 sheets Jane has -/
def num_a4_sheets : ℕ := 28

/-- The number of A3 sheets Jane has -/
def num_a3_sheets : ℕ := 27

/-- The weight of a single A4 sheet in grams -/
def weight_a4_sheet : ℕ := 5

/-- The weight of a single A3 sheet in grams -/
def weight_a3_sheet : ℕ := 10

/-- The total weight of all drawing papers in grams -/
def total_weight : ℕ := num_a4_sheets * weight_a4_sheet + num_a3_sheets * weight_a3_sheet

theorem total_weight_is_410 : total_weight = 410 := by sorry

end NUMINAMATH_CALUDE_total_weight_is_410_l3909_390908


namespace NUMINAMATH_CALUDE_speed_conversion_l3909_390932

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- The speed of the train in meters per second -/
def train_speed_mps : ℝ := 20.0016

/-- The speed of the train in kilometers per hour -/
def train_speed_kmph : ℝ := train_speed_mps * mps_to_kmph

theorem speed_conversion :
  train_speed_kmph = 72.00576 := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l3909_390932


namespace NUMINAMATH_CALUDE_quilt_block_shaded_fraction_l3909_390958

/-- Represents a quilt block -/
structure QuiltBlock where
  size : ℕ
  totalSquares : ℕ
  dividedSquares : ℕ
  shadedTrianglesPerSquare : ℕ

/-- The fraction of a quilt block that is shaded -/
def shadedFraction (q : QuiltBlock) : ℚ :=
  (q.dividedSquares * q.shadedTrianglesPerSquare : ℚ) / (2 * q.totalSquares : ℚ)

/-- Theorem: The shaded fraction of the specified quilt block is 1/8 -/
theorem quilt_block_shaded_fraction :
  let q : QuiltBlock := ⟨4, 16, 4, 1⟩
  shadedFraction q = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_quilt_block_shaded_fraction_l3909_390958


namespace NUMINAMATH_CALUDE_rational_expression_equality_inequality_system_solution_l3909_390959

-- Part 1
theorem rational_expression_equality (x : ℝ) (h : x ≠ 3) :
  (2*x + 4) / (x^2 - 6*x + 9) / ((2*x - 1) / (x - 3) - 1) = 2 / (x - 3) :=
sorry

-- Part 2
theorem inequality_system_solution (x : ℝ) :
  (5*x - 2 > 3*(x + 1) ∧ (1/2)*x - 1 ≥ 7 - (3/2)*x) ↔ x ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_rational_expression_equality_inequality_system_solution_l3909_390959


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l3909_390914

theorem ice_cream_flavors (n k : ℕ) (hn : n = 4) (hk : k = 4) :
  (n + k - 1).choose (k - 1) = 35 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l3909_390914


namespace NUMINAMATH_CALUDE_induction_step_for_even_numbers_l3909_390984

theorem induction_step_for_even_numbers (k : ℕ) (h_k_even : Even k) (h_k_ge_2 : k ≥ 2) :
  let n := k + 2
  Even n ∧ n > k :=
by sorry

end NUMINAMATH_CALUDE_induction_step_for_even_numbers_l3909_390984


namespace NUMINAMATH_CALUDE_circle_radius_from_distances_l3909_390939

theorem circle_radius_from_distances (max_distance min_distance : ℝ) 
  (h1 : max_distance = 11)
  (h2 : min_distance = 5) :
  ∃ (r : ℝ), (r = 3 ∨ r = 8) ∧ 
  ((max_distance - min_distance = 2 * r) ∨ (max_distance + min_distance = 2 * r)) := by
sorry

end NUMINAMATH_CALUDE_circle_radius_from_distances_l3909_390939


namespace NUMINAMATH_CALUDE_no_prime_sum_72_l3909_390947

theorem no_prime_sum_72 : ¬∃ (p q k : ℕ), Prime p ∧ Prime q ∧ p + q = 72 ∧ p * q = k := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_72_l3909_390947


namespace NUMINAMATH_CALUDE_tangent_and_cosine_relations_l3909_390929

theorem tangent_and_cosine_relations (θ : Real) (h : Real.tan θ = 2) :
  (Real.tan (π / 4 - θ) = -1 / 3) ∧ (Real.cos (2 * θ) = -3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_cosine_relations_l3909_390929


namespace NUMINAMATH_CALUDE_monas_weekly_miles_l3909_390988

/-- Represents the days of the week Mona bikes --/
inductive BikingDay
| Monday
| Wednesday
| Saturday

/-- Represents Mona's biking schedule --/
structure BikingSchedule where
  monday_miles : ℕ
  wednesday_miles : ℕ
  saturday_miles : ℕ

/-- Mona's actual biking schedule --/
def monas_schedule : BikingSchedule :=
  { monday_miles := 6
  , wednesday_miles := 12
  , saturday_miles := 12 }

/-- The total miles Mona bikes in a week --/
def total_miles (schedule : BikingSchedule) : ℕ :=
  schedule.monday_miles + schedule.wednesday_miles + schedule.saturday_miles

/-- Theorem stating that Mona bikes 30 miles each week --/
theorem monas_weekly_miles :
  total_miles monas_schedule = 30 ∧
  monas_schedule.wednesday_miles = 12 ∧
  monas_schedule.saturday_miles = 2 * monas_schedule.monday_miles ∧
  monas_schedule.monday_miles = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_monas_weekly_miles_l3909_390988


namespace NUMINAMATH_CALUDE_original_number_l3909_390975

theorem original_number (x : ℚ) : (3 * (x + 3) - 4) / 3 = 10 → x = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l3909_390975
