import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_perimeter_l3123_312350

theorem right_triangle_hypotenuse_and_perimeter :
  ∀ (a b c : ℝ),
  a = 60 →
  b = 80 →
  c^2 = a^2 + b^2 →
  c = 100 ∧ (a + b + c = 240) :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_perimeter_l3123_312350


namespace NUMINAMATH_CALUDE_equation_solution_l3123_312346

theorem equation_solution : ∃! x : ℝ, 5 * (3 * x + 2) - 2 = -2 * (1 - 7 * x) ∧ x = -10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3123_312346


namespace NUMINAMATH_CALUDE_cube_difference_of_sum_and_diff_l3123_312336

theorem cube_difference_of_sum_and_diff (x y : ℕ) 
  (sum_eq : x + y = 64) 
  (diff_eq : x - y = 16) 
  (x_pos : x > 0) 
  (y_pos : y > 0) : 
  x^3 - y^3 = 50176 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_of_sum_and_diff_l3123_312336


namespace NUMINAMATH_CALUDE_range_of_x_l3123_312392

theorem range_of_x (x : ℝ) : 
  (0 ≤ x ∧ x < 2 * Real.pi) → 
  (Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x) →
  (x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l3123_312392


namespace NUMINAMATH_CALUDE_garage_visitors_l3123_312339

/-- Given a number of cars, selections per car, and selections per client,
    calculate the number of clients who visited the garage. -/
def clientsVisited (numCars : ℕ) (selectionsPerCar : ℕ) (selectionsPerClient : ℕ) : ℕ :=
  (numCars * selectionsPerCar) / selectionsPerClient

/-- Theorem stating that given 15 cars, where each car is selected exactly 3 times,
    and each client selects 3 cars, the number of clients who visited the garage is 15. -/
theorem garage_visitors :
  clientsVisited 15 3 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_garage_visitors_l3123_312339


namespace NUMINAMATH_CALUDE_custom_dice_probability_l3123_312306

/-- Given a custom dice with odds of 5:7 for rolling a six, 
    the probability of not rolling a six is 7/12 -/
theorem custom_dice_probability (favorable : ℕ) (unfavorable : ℕ) 
    (h_odds : favorable = 5 ∧ unfavorable = 7) : 
    (unfavorable : ℚ) / ((favorable : ℚ) + (unfavorable : ℚ)) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_custom_dice_probability_l3123_312306


namespace NUMINAMATH_CALUDE_vector_expressions_equal_AD_l3123_312372

variable {V : Type*} [AddCommGroup V]

variable (A B C D M O : V)

theorem vector_expressions_equal_AD :
  (A - D + M - B) + (B - C + C - M) = A - D ∧
  (A - B + C - D) + (B - C) = A - D ∧
  (O - C) - (O - A) + (C - D) = A - D :=
by sorry

end NUMINAMATH_CALUDE_vector_expressions_equal_AD_l3123_312372


namespace NUMINAMATH_CALUDE_principal_amount_calculation_l3123_312370

/-- Calculate the principal amount given the difference between compound and simple interest -/
theorem principal_amount_calculation (interest_rate : ℝ) (compounding_frequency : ℕ) (time : ℝ) (interest_difference : ℝ) :
  interest_rate = 0.10 →
  compounding_frequency = 2 →
  time = 1 →
  interest_difference = 3.9999999999999147 →
  ∃ (principal : ℝ), 
    (principal * ((1 + interest_rate / compounding_frequency) ^ (compounding_frequency * time) - 1) - 
     principal * interest_rate * time = interest_difference) ∧
    (abs (principal - 1600) < 1) :=
by sorry

end NUMINAMATH_CALUDE_principal_amount_calculation_l3123_312370


namespace NUMINAMATH_CALUDE_harriet_round_trip_l3123_312358

/-- Harriet's round trip between A-ville and B-town -/
theorem harriet_round_trip 
  (d : ℝ) -- distance between A-ville and B-town in km
  (speed_to_b : ℝ) -- speed from A-ville to B-town in km/h
  (time_to_b : ℝ) -- time taken from A-ville to B-town in hours
  (total_time : ℝ) -- total round trip time in hours
  (h1 : d = speed_to_b * time_to_b) -- distance = speed * time for A-ville to B-town
  (h2 : speed_to_b = 100) -- speed from A-ville to B-town is 100 km/h
  (h3 : time_to_b = 3) -- time taken from A-ville to B-town is 3 hours
  (h4 : total_time = 5) -- total round trip time is 5 hours
  : d / (total_time - time_to_b) = 150 := by
  sorry

end NUMINAMATH_CALUDE_harriet_round_trip_l3123_312358


namespace NUMINAMATH_CALUDE_product_2000_sum_bounds_l3123_312371

theorem product_2000_sum_bounds (a b c d e : ℕ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) (hd : d > 1) (he : e > 1)
  (h_product : a * b * c * d * e = 2000) :
  (∃ (x y z w v : ℕ), x > 1 ∧ y > 1 ∧ z > 1 ∧ w > 1 ∧ v > 1 ∧ 
    x * y * z * w * v = 2000 ∧ x + y + z + w + v = 133) ∧
  (∃ (x y z w v : ℕ), x > 1 ∧ y > 1 ∧ z > 1 ∧ w > 1 ∧ v > 1 ∧ 
    x * y * z * w * v = 2000 ∧ x + y + z + w + v = 23) ∧
  (∀ (x y z w v : ℕ), x > 1 → y > 1 → z > 1 → w > 1 → v > 1 → 
    x * y * z * w * v = 2000 → 23 ≤ x + y + z + w + v ∧ x + y + z + w + v ≤ 133) :=
by sorry

end NUMINAMATH_CALUDE_product_2000_sum_bounds_l3123_312371


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3123_312368

/-- A geometric sequence with terms a_n -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The property that a_3 and a_7 are roots of x^2 - 4x + 3 = 0 -/
def roots_property (a : ℕ → ℝ) : Prop :=
  a 3 ^ 2 - 4 * a 3 + 3 = 0 ∧ a 7 ^ 2 - 4 * a 7 + 3 = 0

theorem geometric_sequence_fifth_term (a : ℕ → ℝ) :
  geometric_sequence a → roots_property a → a 5 = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3123_312368


namespace NUMINAMATH_CALUDE_max_primitive_dinosaur_cells_l3123_312337

/-- Represents a dinosaur as a tree -/
structure Dinosaur where
  cells : ℕ
  is_connected : Bool
  max_degree : ℕ

/-- Defines a primitive dinosaur -/
def is_primitive (d : Dinosaur) : Prop :=
  ∀ (d1 d2 : Dinosaur), d.cells ≠ d1.cells + d2.cells ∨ d1.cells < 2007 ∨ d2.cells < 2007

/-- The main theorem stating the maximum number of cells in a primitive dinosaur -/
theorem max_primitive_dinosaur_cells :
  ∀ (d : Dinosaur),
    d.cells ≥ 2007 →
    d.is_connected = true →
    d.max_degree = 4 →
    is_primitive d →
    d.cells ≤ 8025 :=
sorry

end NUMINAMATH_CALUDE_max_primitive_dinosaur_cells_l3123_312337


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l3123_312352

theorem max_value_sum_of_roots (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l3123_312352


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l3123_312345

theorem simplify_complex_fraction (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -3) :
  (x + 2 - 5 / (x - 2)) / ((x + 3) / (x - 2)) = x - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l3123_312345


namespace NUMINAMATH_CALUDE_solution_count_l3123_312386

/-- For any integer k > 1, there exist at least 3k + 1 distinct triples of positive integers (m, n, r) 
    satisfying the equation mn + nr + mr = k(m + n + r). -/
theorem solution_count (k : ℕ) (h : k > 1) : 
  ∃ S : Finset (ℕ × ℕ × ℕ), 
    (∀ (m n r : ℕ), (m, n, r) ∈ S → m > 0 ∧ n > 0 ∧ r > 0) ∧ 
    (∀ (m n r : ℕ), (m, n, r) ∈ S → m * n + n * r + m * r = k * (m + n + r)) ∧
    S.card ≥ 3 * k + 1 :=
sorry

end NUMINAMATH_CALUDE_solution_count_l3123_312386


namespace NUMINAMATH_CALUDE_jerry_remaining_money_l3123_312351

/-- Calculates the remaining money after expenses --/
def remaining_money (initial_amount video_games_cost snack_cost toy_original_cost toy_discount_percent : ℚ) : ℚ :=
  let toy_discount := toy_original_cost * (toy_discount_percent / 100)
  let toy_final_cost := toy_original_cost - toy_discount
  let total_spent := video_games_cost + snack_cost + toy_final_cost
  initial_amount - total_spent

/-- Theorem stating that Jerry's remaining money is $6 --/
theorem jerry_remaining_money :
  remaining_money 18 6 3 4 25 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jerry_remaining_money_l3123_312351


namespace NUMINAMATH_CALUDE_repeating_decimal_135_equals_5_37_l3123_312394

def repeating_decimal (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 999

theorem repeating_decimal_135_equals_5_37 :
  repeating_decimal 1 3 5 = 5 / 37 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_135_equals_5_37_l3123_312394


namespace NUMINAMATH_CALUDE_b_horses_count_l3123_312354

/-- The number of horses b put in the pasture -/
def H : ℕ := 16

/-- The total cost of the pasture in Rs -/
def total_cost : ℕ := 841

/-- The amount b should pay in Rs -/
def b_payment : ℕ := 348

/-- The number of horses a put in -/
def a_horses : ℕ := 12

/-- The number of months a's horses stayed -/
def a_months : ℕ := 8

/-- The number of months b's horses stayed -/
def b_months : ℕ := 9

/-- The number of horses c put in -/
def c_horses : ℕ := 18

/-- The number of months c's horses stayed -/
def c_months : ℕ := 6

theorem b_horses_count : 
  (b_payment : ℚ) / total_cost = 
  (H * b_months : ℚ) / (a_horses * a_months + H * b_months + c_horses * c_months) :=
by sorry

end NUMINAMATH_CALUDE_b_horses_count_l3123_312354


namespace NUMINAMATH_CALUDE_sin_cos_inverse_equation_l3123_312318

theorem sin_cos_inverse_equation (t : ℝ) :
  (Real.sin (2 * t) - Real.arcsin (2 * t))^2 + (Real.arccos (2 * t) - Real.cos (2 * t))^2 = 1 ↔
  ∃ k : ℤ, t = (π / 8) * (2 * ↑k + 1) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_inverse_equation_l3123_312318


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3123_312329

theorem simple_interest_problem (interest : ℚ) (rate : ℚ) (time : ℚ) (principal : ℚ) : 
  interest = 750 →
  rate = 6 / 100 →
  time = 5 →
  principal * rate * time = interest →
  principal = 2500 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3123_312329


namespace NUMINAMATH_CALUDE_dumbbell_system_weight_l3123_312317

/-- Represents the weight of a dumbbell pair in pounds -/
structure DumbbellPair where
  weight : ℕ

/-- Represents a multi-level dumbbell system -/
structure DumbbellSystem where
  pairs : List DumbbellPair

def total_weight (system : DumbbellSystem) : ℕ :=
  system.pairs.map (λ pair => 2 * pair.weight) |>.sum

theorem dumbbell_system_weight :
  ∀ (system : DumbbellSystem),
    system.pairs = [
      DumbbellPair.mk 3,
      DumbbellPair.mk 5,
      DumbbellPair.mk 8
    ] →
    total_weight system = 32 := by
  sorry

end NUMINAMATH_CALUDE_dumbbell_system_weight_l3123_312317


namespace NUMINAMATH_CALUDE_inequality_generalization_l3123_312398

theorem inequality_generalization (n : ℕ) (a : ℝ) :
  (∀ x : ℝ, x > 0 → x + a / x^n ≥ n + 1) → a = n^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_generalization_l3123_312398


namespace NUMINAMATH_CALUDE_triangle_construction_l3123_312321

/-- A line in a plane --/
structure Line where
  -- (We don't need to define the internal structure of a line for this statement)

/-- A point in a plane --/
structure Point where
  -- (We don't need to define the internal structure of a point for this statement)

/-- Represents a triangle --/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a point is on a given line --/
def Point.isOnLine (p : Point) (l : Line) : Prop :=
  sorry

/-- Checks if two points are on opposite sides of a line --/
def areOnOppositeSides (p1 p2 : Point) (l : Line) : Prop :=
  sorry

/-- Checks if a line is an altitude of a triangle --/
def Line.isAltitudeOf (l : Line) (t : Triangle) : Prop :=
  sorry

/-- Checks if a point is the foot of an altitude of a triangle --/
def Point.isAltitudeFoot (p : Point) (t : Triangle) : Prop :=
  sorry

/-- Checks if a line passes through the midpoint of a segment --/
def Line.passesThroughMidpoint (l : Line) (p1 p2 : Point) : Prop :=
  sorry

/-- Checks if a line is perpendicular to a segment --/
def Line.isPerpendicularTo (l : Line) (p1 p2 : Point) : Prop :=
  sorry

/-- The main theorem --/
theorem triangle_construction (l : Line) (E F : Point) 
  (h1 : areOnOppositeSides E F l) :
  (∃ (D : Point), 
    let t := Triangle.mk D E F
    (E.isAltitudeFoot t ∧ F.isAltitudeFoot t ∧ 
     ∃ (alt : Line), alt.isAltitudeOf t ∧ D.isOnLine alt ∧ alt = l)) ↔ 
  (¬l.passesThroughMidpoint E F ∨ 
   (l.passesThroughMidpoint E F ∧ l.isPerpendicularTo E F)) :=
sorry

end NUMINAMATH_CALUDE_triangle_construction_l3123_312321


namespace NUMINAMATH_CALUDE_cross_ratio_preserving_is_projective_l3123_312343

/-- A mapping between two lines -/
structure LineMapping (α : Type*) where
  to_fun : α → α

/-- The cross ratio of four points -/
def cross_ratio {α : Type*} [Field α] (x y z w : α) : α :=
  ((x - z) * (y - w)) / ((x - w) * (y - z))

/-- A mapping preserves cross ratio -/
def preserves_cross_ratio {α : Type*} [Field α] (f : LineMapping α) : Prop :=
  ∀ (x y z w : α), cross_ratio (f.to_fun x) (f.to_fun y) (f.to_fun z) (f.to_fun w) = cross_ratio x y z w

/-- Definition of a projective transformation -/
def is_projective {α : Type*} [Field α] (f : LineMapping α) : Prop :=
  ∃ (a b c d : α), (a * d - b * c ≠ 0) ∧
    (∀ x, f.to_fun x = (a * x + b) / (c * x + d))

/-- Main theorem: A cross-ratio preserving mapping is projective -/
theorem cross_ratio_preserving_is_projective {α : Type*} [Field α] (f : LineMapping α) :
  preserves_cross_ratio f → is_projective f :=
sorry

end NUMINAMATH_CALUDE_cross_ratio_preserving_is_projective_l3123_312343


namespace NUMINAMATH_CALUDE_f_of_f_10_eq_2_l3123_312300

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else Real.log x / Real.log 2

theorem f_of_f_10_eq_2 : f (f 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_10_eq_2_l3123_312300


namespace NUMINAMATH_CALUDE_sum_and_difference_squares_l3123_312326

theorem sum_and_difference_squares (x y : ℝ) (h1 : x + y = 24) (h2 : x - y = 8) :
  (x + y)^2 + (x - y)^2 = 640 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_difference_squares_l3123_312326


namespace NUMINAMATH_CALUDE_rowing_time_ratio_l3123_312335

/-- Proves that the ratio of time taken to row against the stream to the time taken to row in favor of the stream is 2:1, given that the boat's speed in still water is 3 times the stream's speed. -/
theorem rowing_time_ratio (B S D : ℝ) (h_positive : B > 0 ∧ S > 0 ∧ D > 0) (h_speed_ratio : B = 3 * S) :
  (D / (B - S)) / (D / (B + S)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_rowing_time_ratio_l3123_312335


namespace NUMINAMATH_CALUDE_cookie_drop_count_l3123_312327

/-- Represents the number of cookies of each type made by Alice and Bob --/
structure CookieCount where
  chocolate_chip : ℕ
  sugar : ℕ
  oatmeal_raisin : ℕ
  peanut_butter : ℕ
  snickerdoodle : ℕ
  white_chocolate_macadamia : ℕ

/-- Calculates the total number of cookies --/
def total_cookies (c : CookieCount) : ℕ :=
  c.chocolate_chip + c.sugar + c.oatmeal_raisin + c.peanut_butter + c.snickerdoodle + c.white_chocolate_macadamia

theorem cookie_drop_count 
  (initial_cookies : CookieCount)
  (initial_dropped : CookieCount)
  (additional_cookies : CookieCount)
  (final_edible_cookies : ℕ) :
  total_cookies initial_cookies + total_cookies additional_cookies - final_edible_cookies = 139 :=
by sorry

end NUMINAMATH_CALUDE_cookie_drop_count_l3123_312327


namespace NUMINAMATH_CALUDE_milk_water_ratio_in_mixed_vessel_l3123_312396

/-- Given three vessels with volumes in ratio 3:5:7 and milk-to-water ratios,
    prove the final milk-to-water ratio when mixed -/
theorem milk_water_ratio_in_mixed_vessel
  (v1 v2 v3 : ℚ)  -- Volumes of the three vessels
  (m1 w1 m2 w2 m3 w3 : ℚ)  -- Milk and water ratios in each vessel
  (hv : v1 / v2 = 3 / 5 ∧ v2 / v3 = 5 / 7)  -- Volume ratio condition
  (hr1 : m1 / w1 = 1 / 2)  -- Milk-to-water ratio in first vessel
  (hr2 : m2 / w2 = 3 / 2)  -- Milk-to-water ratio in second vessel
  (hr3 : m3 / w3 = 2 / 3)  -- Milk-to-water ratio in third vessel
  (hm1 : m1 + w1 = 1)  -- Normalization of ratios
  (hm2 : m2 + w2 = 1)
  (hm3 : m3 + w3 = 1) :
  (v1 * m1 + v2 * m2 + v3 * m3) / (v1 * w1 + v2 * w2 + v3 * w3) = 34 / 41 :=
sorry

end NUMINAMATH_CALUDE_milk_water_ratio_in_mixed_vessel_l3123_312396


namespace NUMINAMATH_CALUDE_bobs_speed_limit_l3123_312378

/-- Proves that Bob's speed must be less than 40 mph for Alice to arrive before him --/
theorem bobs_speed_limit (distance : ℝ) (alice_delay : ℝ) (alice_min_speed : ℝ) 
  (h_distance : distance = 180)
  (h_delay : alice_delay = 1/2)
  (h_min_speed : alice_min_speed = 45) :
  ∀ v : ℝ, (∃ (alice_speed : ℝ), 
    alice_speed > alice_min_speed ∧ 
    distance / alice_speed < distance / v - alice_delay) → 
  v < 40 := by
  sorry

end NUMINAMATH_CALUDE_bobs_speed_limit_l3123_312378


namespace NUMINAMATH_CALUDE_maryville_population_increase_l3123_312311

/-- The average number of people added each year in Maryville between 2000 and 2005 -/
def average_population_increase (pop_2000 pop_2005 : ℕ) : ℚ :=
  (pop_2005 - pop_2000 : ℚ) / 5

/-- Theorem stating that the average population increase in Maryville between 2000 and 2005 is 3400 -/
theorem maryville_population_increase :
  average_population_increase 450000 467000 = 3400 := by
  sorry

end NUMINAMATH_CALUDE_maryville_population_increase_l3123_312311


namespace NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l3123_312308

theorem sphere_radius_from_surface_area (S : ℝ) (r : ℝ) (h : S = 4 * Real.pi) :
  S = 4 * Real.pi * r^2 → r = 1 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l3123_312308


namespace NUMINAMATH_CALUDE_standard_pairs_parity_l3123_312384

/-- Represents a color on the chessboard -/
inductive Color
| Red
| Blue

/-- Represents a chessboard -/
structure Chessboard (m n : ℕ) where
  colors : Fin m → Fin n → Color
  m_ge_three : m ≥ 3
  n_ge_three : n ≥ 3

/-- Counts the number of blue squares on the boundary (excluding corners) -/
def countBlueBoundary (board : Chessboard m n) : ℕ :=
  sorry

/-- Counts the number of standard pairs on the chessboard -/
def countStandardPairs (board : Chessboard m n) : ℕ :=
  sorry

/-- Theorem stating that the parity of standard pairs is determined by the parity of blue boundary squares -/
theorem standard_pairs_parity (m n : ℕ) (board : Chessboard m n) :
  Even (countStandardPairs board) ↔ Even (countBlueBoundary board) :=
sorry

end NUMINAMATH_CALUDE_standard_pairs_parity_l3123_312384


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l3123_312342

open Set

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 < 0) ↔ (∀ x : ℝ, x^2 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l3123_312342


namespace NUMINAMATH_CALUDE_area_of_region_l3123_312373

/-- The equation of the curve enclosing the region -/
def curve_equation (x y : ℝ) : Prop :=
  x^2 - 10*x + 2*y + 50 = 25 + 7*y - y^2

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop :=
  y = x - 5

/-- The region above the line -/
def region_above_line (x y : ℝ) : Prop :=
  y > x - 5

/-- The theorem stating the area of the region -/
theorem area_of_region : 
  ∃ (A : ℝ), A = 25 * Real.pi / 4 ∧ 
  (∀ x y : ℝ, curve_equation x y ∧ region_above_line x y → 
    (x, y) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ A / Real.pi}) :=
sorry

end NUMINAMATH_CALUDE_area_of_region_l3123_312373


namespace NUMINAMATH_CALUDE_abc_condition_neither_sufficient_nor_necessary_l3123_312365

theorem abc_condition_neither_sufficient_nor_necessary :
  ¬ (∀ a b c : ℝ, a * b * c = 1 → 1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c ≤ a + b + c) ∧
  ¬ (∀ a b c : ℝ, 1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c ≤ a + b + c → a * b * c = 1) :=
by sorry

end NUMINAMATH_CALUDE_abc_condition_neither_sufficient_nor_necessary_l3123_312365


namespace NUMINAMATH_CALUDE_min_value_theorem_l3123_312309

/-- The hyperbola equation -/
def hyperbola (m n x y : ℝ) : Prop := x^2 / m - y^2 / n = 1

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The condition that the hyperbola and ellipse have the same foci -/
def same_foci (m n : ℝ) : Prop := m + n = 1

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_hyperbola : ∃ x y, hyperbola m n x y)
  (h_ellipse : ∃ x y, ellipse x y)
  (h_foci : same_foci m n) :
  (∀ m' n', m' > 0 → n' > 0 → same_foci m' n' → 4/m + 1/n ≤ 4/m' + 1/n') ∧ 
  (∃ m₀ n₀, m₀ > 0 ∧ n₀ > 0 ∧ same_foci m₀ n₀ ∧ 4/m₀ + 1/n₀ = 9) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3123_312309


namespace NUMINAMATH_CALUDE_identity_function_is_unique_solution_l3123_312356

theorem identity_function_is_unique_solution
  (f : ℕ → ℕ)
  (h : ∀ n : ℕ, f n + f (f n) + f (f (f n)) = 3 * n) :
  ∀ n : ℕ, f n = n :=
by sorry

end NUMINAMATH_CALUDE_identity_function_is_unique_solution_l3123_312356


namespace NUMINAMATH_CALUDE_exp_cos_inequality_l3123_312383

theorem exp_cos_inequality : 
  (Real.exp (Real.cos 1)) / (Real.cos 2 + 1) < 2 * Real.sqrt (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_exp_cos_inequality_l3123_312383


namespace NUMINAMATH_CALUDE_ellipse_intersection_slope_l3123_312375

/-- Given an ellipse ax^2 + by^2 = 1 intersecting with the line y = 1 - x,
    if the slope of the line passing through the origin and the midpoint
    of the intersection points is √3/2, then a/b = √3/2. -/
theorem ellipse_intersection_slope (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x₁ x₂ : ℝ, a * x₁^2 + b * (1 - x₁)^2 = 1 ∧
                a * x₂^2 + b * (1 - x₂)^2 = 1 ∧
                x₁ ≠ x₂) →
  ((b / (a + b)) / (a / (a + b)) = Real.sqrt 3 / 2) →
  a / b = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_slope_l3123_312375


namespace NUMINAMATH_CALUDE_johns_age_theorem_l3123_312395

theorem johns_age_theorem :
  ∀ (age : ℕ),
  (∃ (s : ℕ), (age - 2) = s^2) ∧ 
  (∃ (c : ℕ), (age + 2) = c^3) →
  age = 6 ∨ age = 123 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_age_theorem_l3123_312395


namespace NUMINAMATH_CALUDE_simplify_power_l3123_312325

theorem simplify_power (y : ℝ) : (3 * y^4)^4 = 81 * y^16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_l3123_312325


namespace NUMINAMATH_CALUDE_ratio_q_to_r_l3123_312385

/-- 
Given a total amount of 1210 divided among three persons p, q, and r,
where the ratio of p to q is 5:4 and r receives 400,
prove that the ratio of q to r is 9:10.
-/
theorem ratio_q_to_r (total : ℕ) (p q r : ℕ) (h1 : total = 1210) 
  (h2 : p + q + r = total) (h3 : 5 * q = 4 * p) (h4 : r = 400) :
  9 * r = 10 * q := by
  sorry


end NUMINAMATH_CALUDE_ratio_q_to_r_l3123_312385


namespace NUMINAMATH_CALUDE_distribution_plans_count_l3123_312334

/-- The number of ways to distribute 3 distinct items into 3 distinct boxes,
    where each box must contain at least one item -/
def distribution_count : ℕ := 12

/-- Theorem stating that the number of distribution plans is correct -/
theorem distribution_plans_count : distribution_count = 12 := by
  sorry

end NUMINAMATH_CALUDE_distribution_plans_count_l3123_312334


namespace NUMINAMATH_CALUDE_is_projection_matrix_l3123_312302

def projection_matrix (A : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  A * A = A

theorem is_projection_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![11/12, 12/25; 12/25, 13/25]
  projection_matrix A := by
  sorry

end NUMINAMATH_CALUDE_is_projection_matrix_l3123_312302


namespace NUMINAMATH_CALUDE_f_sum_over_sum_positive_l3123_312301

noncomputable def f (x : ℝ) : ℝ := x^3 - Real.log (Real.sqrt (x^2 + 1) - x)

theorem f_sum_over_sum_positive (a b : ℝ) (h : a + b ≠ 0) :
  (f a + f b) / (a + b) > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_over_sum_positive_l3123_312301


namespace NUMINAMATH_CALUDE_watermelon_seeds_count_l3123_312322

/-- Represents a watermelon with its properties -/
structure Watermelon :=
  (slices : ℕ)
  (black_seeds_per_slice : ℕ)
  (white_seeds_per_slice : ℕ)

/-- Calculates the total number of seeds in a watermelon -/
def total_seeds (w : Watermelon) : ℕ :=
  w.slices * (w.black_seeds_per_slice + w.white_seeds_per_slice)

/-- Theorem stating that a watermelon with 40 slices, 20 black seeds and 20 white seeds per slice has 1600 total seeds -/
theorem watermelon_seeds_count :
  ∀ (w : Watermelon),
  w.slices = 40 →
  w.black_seeds_per_slice = 20 →
  w.white_seeds_per_slice = 20 →
  total_seeds w = 1600 :=
by
  sorry


end NUMINAMATH_CALUDE_watermelon_seeds_count_l3123_312322


namespace NUMINAMATH_CALUDE_combine_terms_mn_zero_l3123_312381

theorem combine_terms_mn_zero (a b : ℝ) (m n : ℤ) :
  (∃ k : ℝ, ∃ p q : ℤ, -2 * a^m * b^4 + 5 * a^(n+2) * b^(2*m+n) = k * a^p * b^q) →
  m * n = 0 :=
sorry

end NUMINAMATH_CALUDE_combine_terms_mn_zero_l3123_312381


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_product_l3123_312324

theorem quadratic_equation_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x ≠ y) →
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9) →
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x * y = 20) →
  p + q = 87 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_product_l3123_312324


namespace NUMINAMATH_CALUDE_nickels_remaining_l3123_312382

def initial_nickels : ℕ := 87
def borrowed_nickels : ℕ := 75

theorem nickels_remaining (initial : ℕ) (borrowed : ℕ) :
  initial ≥ borrowed → initial - borrowed = initial_nickels - borrowed_nickels :=
by sorry

end NUMINAMATH_CALUDE_nickels_remaining_l3123_312382


namespace NUMINAMATH_CALUDE_farmer_feed_expenditure_l3123_312328

theorem farmer_feed_expenditure (initial_amount : ℝ) :
  (initial_amount * 0.4 / 0.5) + (initial_amount * 0.6) = 49 →
  initial_amount = 35 := by
sorry

end NUMINAMATH_CALUDE_farmer_feed_expenditure_l3123_312328


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3123_312303

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (6 * a^3 + 500 * a + 1001 = 0) →
  (6 * b^3 + 500 * b + 1001 = 0) →
  (6 * c^3 + 500 * c + 1001 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 500.5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3123_312303


namespace NUMINAMATH_CALUDE_kate_candy_count_l3123_312355

/-- Given a distribution of candy among four children (Kate, Robert, Bill, and Mary),
    prove that Kate gets 4 pieces of candy. -/
theorem kate_candy_count (kate robert bill mary : ℕ)
  (total : kate + robert + bill + mary = 20)
  (robert_kate : robert = kate + 2)
  (bill_mary : bill = mary - 6)
  (mary_robert : mary = robert + 2)
  (kate_bill : kate = bill + 2) :
  kate = 4 := by
  sorry

end NUMINAMATH_CALUDE_kate_candy_count_l3123_312355


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_inning_l3123_312307

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  average : Rat

/-- Calculates the new average after an additional inning -/
def newAverage (b : Batsman) (newInningRuns : Nat) : Rat :=
  (b.totalRuns + newInningRuns) / (b.innings + 1)

/-- Theorem: A batsman's average after 12th inning -/
theorem batsman_average_after_12th_inning
  (b : Batsman)
  (h1 : b.innings = 11)
  (h2 : newAverage b 60 = b.average + 4)
  : newAverage b 60 = 16 := by
  sorry

#check batsman_average_after_12th_inning

end NUMINAMATH_CALUDE_batsman_average_after_12th_inning_l3123_312307


namespace NUMINAMATH_CALUDE_binomial_square_constant_l3123_312305

/-- If 9x^2 - 18x + c is the square of a binomial, then c = 9 -/
theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x, 9*x^2 - 18*x + c = (a*x + b)^2) → c = 9 := by
sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l3123_312305


namespace NUMINAMATH_CALUDE_mindmaster_codes_l3123_312359

theorem mindmaster_codes (num_slots : ℕ) (num_colors : ℕ) : 
  num_slots = 5 → num_colors = 7 → num_colors ^ num_slots = 16807 := by
  sorry

end NUMINAMATH_CALUDE_mindmaster_codes_l3123_312359


namespace NUMINAMATH_CALUDE_average_of_middle_two_l3123_312366

theorem average_of_middle_two (total_avg : ℝ) (first_two_avg : ℝ) (last_two_avg : ℝ) :
  total_avg = 3.95 →
  first_two_avg = 3.4 →
  last_two_avg = 4.600000000000001 →
  (6 * total_avg - 2 * first_two_avg - 2 * last_two_avg) / 2 = 3.85 := by
  sorry

end NUMINAMATH_CALUDE_average_of_middle_two_l3123_312366


namespace NUMINAMATH_CALUDE_smallest_absolute_value_l3123_312374

theorem smallest_absolute_value : 
  let numbers : Finset ℚ := {-1/2, -2/3, 4, -5}
  ∀ x ∈ numbers, x ≠ -1/2 → abs (-1/2) < abs x :=
by sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_l3123_312374


namespace NUMINAMATH_CALUDE_physics_marks_l3123_312363

theorem physics_marks (P C M : ℝ) 
  (avg_all : (P + C + M) / 3 = 80)
  (avg_PM : (P + M) / 2 = 90)
  (avg_PC : (P + C) / 2 = 70) :
  P = 80 := by
sorry

end NUMINAMATH_CALUDE_physics_marks_l3123_312363


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l3123_312331

/-- Given two points A and B on a line, prove that the sum of the line's slope and y-intercept is 10. -/
theorem slope_intercept_sum (A B : ℝ × ℝ) : 
  A = (5, 6) → B = (8, 3) → 
  let m := (B.2 - A.2) / (B.1 - A.1)
  let b := A.2 - m * A.1
  m + b = 10 := by sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_l3123_312331


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l3123_312397

theorem greatest_integer_radius (r : ℕ) : 
  (∀ k : ℕ, k > r → ¬(Real.pi * k^2 < 30 * Real.pi ∧ 2 * Real.pi * k > 10 * Real.pi)) ∧
  (Real.pi * r^2 < 30 * Real.pi ∧ 2 * Real.pi * r > 10 * Real.pi) →
  r = 5 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l3123_312397


namespace NUMINAMATH_CALUDE_cad_to_jpy_exchange_l3123_312341

/-- The exchange rate from Canadian dollars (CAD) to Japanese yen (JPY) -/
def exchange_rate (cad : ℚ) (jpy : ℚ) : Prop :=
  5000 / 60 = jpy / cad

/-- The rounded exchange rate for 1 CAD in JPY -/
def rounded_rate (rate : ℚ) : ℕ :=
  (rate + 1/2).floor.toNat

theorem cad_to_jpy_exchange :
  ∃ (rate : ℚ), exchange_rate 1 rate ∧ rounded_rate rate = 83 := by
  sorry

end NUMINAMATH_CALUDE_cad_to_jpy_exchange_l3123_312341


namespace NUMINAMATH_CALUDE_nonnegative_solutions_count_l3123_312389

theorem nonnegative_solutions_count : 
  ∃! (n : ℕ), ∃ (x : ℝ), x ≥ 0 ∧ x^2 = -6*x ∧ n = 1 :=
by sorry

end NUMINAMATH_CALUDE_nonnegative_solutions_count_l3123_312389


namespace NUMINAMATH_CALUDE_equal_volume_equal_time_dig_time_equality_l3123_312399

/-- Represents the volume of earth to be dug -/
structure EarthVolume where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of earth given its dimensions -/
def volume (e : EarthVolume) : ℝ :=
  e.depth * e.length * e.breadth

/-- Theorem stating that equal volumes take equal time to dig -/
theorem equal_volume_equal_time (people : ℕ) (v1 v2 : EarthVolume) (days1 : ℝ) :
  volume v1 = volume v2 →
  days1 * (volume v2) = days1 * (volume v1) :=
by
  sorry

/-- Main theorem proving that digging 75 * 20 * 50 m³ takes 12 days
    if digging 100 * 25 * 30 m³ takes 12 days -/
theorem dig_time_equality :
  let v1 : EarthVolume := ⟨100, 25, 30⟩
  let v2 : EarthVolume := ⟨75, 20, 50⟩
  volume v1 = volume v2 →
  (12 : ℝ) = (12 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_equal_volume_equal_time_dig_time_equality_l3123_312399


namespace NUMINAMATH_CALUDE_store_revenue_l3123_312380

def shirt_price : ℝ := 10
def jeans_price : ℝ := 2 * shirt_price
def jacket_price : ℝ := 3 * jeans_price
def discount_rate : ℝ := 0.1

def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def num_jackets : ℕ := 15

def total_revenue : ℝ :=
  (num_shirts : ℝ) * shirt_price +
  (num_jeans : ℝ) * jeans_price +
  (num_jackets : ℝ) * jacket_price * (1 - discount_rate)

theorem store_revenue :
  total_revenue = 1210 := by sorry

end NUMINAMATH_CALUDE_store_revenue_l3123_312380


namespace NUMINAMATH_CALUDE_root_sum_of_coefficients_l3123_312323

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def isRoot (z : ℂ) (b c : ℝ) : Prop :=
  z^2 + b * z + c = 0

-- Theorem statement
theorem root_sum_of_coefficients :
  ∀ (b c : ℝ), isRoot (2 + i) b c → b + c = 1 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_of_coefficients_l3123_312323


namespace NUMINAMATH_CALUDE_cyclic_inequality_l3123_312338

theorem cyclic_inequality (x y z m n : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hm : m > 0) (hn : n > 0)
  (hmn : m + n ≥ 2) : 
  x * Real.sqrt (y * z * (x + m * y) * (x + n * z)) + 
  y * Real.sqrt (x * z * (y + m * x) * (y + n * z)) + 
  z * Real.sqrt (x * y * (z + m * x) * (z + n * y)) ≤ 
  (3 * (m + n) / 8) * (x + y) * (y + z) * (z + x) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l3123_312338


namespace NUMINAMATH_CALUDE_line_through_parabola_vertex_l3123_312333

theorem line_through_parabola_vertex (a : ℝ) : 
  (∃! (x y : ℝ), y = x^2 + a^2 ∧ y = 2*x + a ∧ ∀ (x' : ℝ), x'^2 + a^2 ≥ y) ↔ (a = 0 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_line_through_parabola_vertex_l3123_312333


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l3123_312344

theorem min_value_quadratic_form (x y z : ℝ) (h : 3 * x + 2 * y + z = 1) :
  ∃ (m : ℝ), m = 3 / 34 ∧ x^2 + 2 * y^2 + 3 * z^2 ≥ m ∧
  ∃ (x₀ y₀ z₀ : ℝ), 3 * x₀ + 2 * y₀ + z₀ = 1 ∧ x₀^2 + 2 * y₀^2 + 3 * z₀^2 = m :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l3123_312344


namespace NUMINAMATH_CALUDE_stacy_berries_l3123_312330

theorem stacy_berries (total : ℕ) (x : ℕ) : total = 1100 → x + 2*x + 8*x = total → 8*x = 800 :=
by
  sorry

end NUMINAMATH_CALUDE_stacy_berries_l3123_312330


namespace NUMINAMATH_CALUDE_triangle_area_is_2_sqrt_6_l3123_312304

/-- A triangle with integral sides and perimeter 12 --/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  perimeter_eq : a + b + c = 12
  triangle_ineq_ab : a + b > c
  triangle_ineq_bc : b + c > a
  triangle_ineq_ca : c + a > b

/-- The area of a triangle with integral sides and perimeter 12 is 2√6 --/
theorem triangle_area_is_2_sqrt_6 (t : Triangle) : 
  ∃ (area : ℝ), area = 2 * Real.sqrt 6 ∧ area = (t.a * t.b * Real.sin (π / 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_2_sqrt_6_l3123_312304


namespace NUMINAMATH_CALUDE_inequality_proof_l3123_312316

theorem inequality_proof (x : ℝ) (n : ℕ) 
  (h1 : |x| < 1) (h2 : n ≥ 2) : (1 + x)^n + (1 - x)^n < 2^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3123_312316


namespace NUMINAMATH_CALUDE_semicircle_area_comparison_l3123_312349

theorem semicircle_area_comparison : ∀ (short_side long_side : ℝ),
  short_side = 8 →
  long_side = 12 →
  let large_semicircle_area := π * (long_side / 2)^2
  let small_semicircle_area := π * (short_side / 2)^2
  large_semicircle_area = small_semicircle_area * 2.25 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_area_comparison_l3123_312349


namespace NUMINAMATH_CALUDE_art_museum_survey_l3123_312314

theorem art_museum_survey (V T E : ℕ) : 
  V = 6 * T →                 -- Total visitors = 6 * visitors who spent >10 minutes
  E = 3 * T / 5 →             -- 3/5 of visitors who spent >10 minutes both enjoyed and understood
  E = 180 →                   -- Number who enjoyed and understood = 180
  V = 1800 := by              -- Prove that total visitors = 1800
sorry

end NUMINAMATH_CALUDE_art_museum_survey_l3123_312314


namespace NUMINAMATH_CALUDE_ln_intersection_and_exponential_inequality_l3123_312369

open Real

theorem ln_intersection_and_exponential_inequality (m n : ℝ) (h : m < n) :
  (∃! x : ℝ, log x = x - 1) ∧
  ((exp n - exp m) / (n - m) > exp ((m + n) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ln_intersection_and_exponential_inequality_l3123_312369


namespace NUMINAMATH_CALUDE_triangleCount_is_sixteen_l3123_312376

/-- Represents a triangular grid with a given number of rows -/
structure TriangularGrid :=
  (rows : ℕ)

/-- Counts the number of small triangles in a triangular grid -/
def countSmallTriangles (grid : TriangularGrid) : ℕ :=
  (grid.rows * (grid.rows + 1)) / 2

/-- Counts the number of medium triangles in a triangular grid -/
def countMediumTriangles (grid : TriangularGrid) : ℕ :=
  (grid.rows - 1) * grid.rows / 2

/-- Counts the number of large triangles in a triangular grid -/
def countLargeTriangles (grid : TriangularGrid) : ℕ := 1

/-- Counts the number of extra large triangles in a triangular grid -/
def countExtraLargeTriangles (grid : TriangularGrid) : ℕ := 1

/-- Counts the total number of triangles in a triangular grid -/
def countTotalTriangles (grid : TriangularGrid) : ℕ :=
  countSmallTriangles grid + countMediumTriangles grid + 
  countLargeTriangles grid + countExtraLargeTriangles grid

theorem triangleCount_is_sixteen :
  ∀ (grid : TriangularGrid), grid.rows = 4 → countTotalTriangles grid = 16 := by
  sorry

end NUMINAMATH_CALUDE_triangleCount_is_sixteen_l3123_312376


namespace NUMINAMATH_CALUDE_coprime_linear_combination_l3123_312377

theorem coprime_linear_combination (a b n : ℕ+) (h1 : Nat.Coprime a b) (h2 : n > a * b) :
  ∃ (x y : ℕ+), n = a * x + b * y := by
sorry

end NUMINAMATH_CALUDE_coprime_linear_combination_l3123_312377


namespace NUMINAMATH_CALUDE_words_with_e_count_l3123_312364

/-- The number of letters in the alphabet we're using -/
def alphabet_size : ℕ := 5

/-- The length of the words we're forming -/
def word_length : ℕ := 4

/-- The number of letters in the alphabet excluding E -/
def alphabet_size_without_e : ℕ := 4

/-- The total number of possible words -/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of words without E -/
def words_without_e : ℕ := alphabet_size_without_e ^ word_length

/-- The number of words with at least one E -/
def words_with_e : ℕ := total_words - words_without_e

theorem words_with_e_count : words_with_e = 369 := by
  sorry

end NUMINAMATH_CALUDE_words_with_e_count_l3123_312364


namespace NUMINAMATH_CALUDE_inequality_proof_l3123_312360

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_sum : d * e * f + d * e + e * f + f * d = 4) : 
  ((a + b) * d * e + (b + c) * e * f + (c + a) * f * d)^2 ≥ 
  12 * (a * b * d * e + b * c * e * f + c * a * f * d) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3123_312360


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l3123_312391

theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 4 ∧ 
  (∃ (n : ℕ), 3 * b + 4 = n^2) ∧
  (∀ (k : ℕ), k > 4 ∧ k < b → ¬∃ (m : ℕ), 3 * k + 4 = m^2) ∧
  b = 7 := by
sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l3123_312391


namespace NUMINAMATH_CALUDE_semicircle_radius_prove_semicircle_radius_l3123_312319

theorem semicircle_radius : ℝ → Prop :=
fun r : ℝ =>
  (3 * (2 * r) + 2 * 12 = 2 * (2 * r) + 22 + 16 + 22) → r = 18

theorem prove_semicircle_radius : semicircle_radius 18 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_prove_semicircle_radius_l3123_312319


namespace NUMINAMATH_CALUDE_bottle_production_l3123_312388

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 16 such machines will produce 2880 bottles in 4 minutes. -/
theorem bottle_production 
  (machines : ℕ → ℕ) -- number of machines
  (bottles_per_minute : ℕ → ℕ) -- bottles produced per minute
  (h1 : machines 1 = 6)
  (h2 : bottles_per_minute 1 = 270)
  (h3 : ∀ n : ℕ, bottles_per_minute n = n * (bottles_per_minute 1 / machines 1)) :
  bottles_per_minute 16 * 4 = 2880 :=
by sorry

end NUMINAMATH_CALUDE_bottle_production_l3123_312388


namespace NUMINAMATH_CALUDE_complex_real_condition_l3123_312340

theorem complex_real_condition (a : ℝ) : 
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (((a^2 - 1) : ℂ) + (Complex.I : ℂ) * (a + 1)).im = 0 →
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l3123_312340


namespace NUMINAMATH_CALUDE_carlos_laundry_time_l3123_312315

/-- The total time for Carlos's laundry process -/
def laundry_time (wash_times : List Nat) (dry_times : List Nat) : Nat :=
  wash_times.sum + dry_times.sum

/-- Theorem stating that Carlos's laundry takes 380 minutes in total -/
theorem carlos_laundry_time :
  laundry_time [30, 45, 40, 50, 35] [85, 95] = 380 := by
  sorry

end NUMINAMATH_CALUDE_carlos_laundry_time_l3123_312315


namespace NUMINAMATH_CALUDE_power_multiplication_l3123_312387

theorem power_multiplication : 3^6 * 4^3 = 46656 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3123_312387


namespace NUMINAMATH_CALUDE_lucy_money_ratio_l3123_312362

/-- Proves that the ratio of money lost to initial amount is 1:3 given the conditions of Lucy's spending --/
theorem lucy_money_ratio (initial_amount : ℝ) (lost_amount : ℝ) (remainder : ℝ) (final_amount : ℝ) :
  initial_amount = 30 →
  remainder = initial_amount - lost_amount →
  final_amount = remainder - (1/4) * remainder →
  final_amount = 15 →
  lost_amount / initial_amount = 1/3 := by
sorry

end NUMINAMATH_CALUDE_lucy_money_ratio_l3123_312362


namespace NUMINAMATH_CALUDE_coefficient_sum_of_squares_l3123_312313

theorem coefficient_sum_of_squares (p q r s t u : ℤ) :
  (∀ x, 343 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 3506 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_sum_of_squares_l3123_312313


namespace NUMINAMATH_CALUDE_max_sum_is_33_l3123_312347

def numbers : List ℕ := [2, 5, 8, 11, 14]

structure LShape :=
  (a b c d e : ℕ)
  (in_numbers : {a, b, c, d, e} ⊆ numbers.toFinset)
  (horizontal_eq_vertical : a + b + e = a + c + e)

def sum (l : LShape) : ℕ := l.a + l.b + l.e

theorem max_sum_is_33 :
  ∃ (l : LShape), sum l = 33 ∧ ∀ (l' : LShape), sum l' ≤ 33 :=
sorry

end NUMINAMATH_CALUDE_max_sum_is_33_l3123_312347


namespace NUMINAMATH_CALUDE_intern_teacher_distribution_l3123_312361

/-- The number of ways to distribute n teachers among k classes with at least one teacher per class -/
def distribution_schemes (n k : ℕ) : ℕ :=
  if n < k then 0
  else (n - k + 1).choose k * (k - 1).choose (n - k)

/-- Theorem: There are 60 ways to distribute 5 intern teachers among 3 freshman classes with at least 1 teacher in each class -/
theorem intern_teacher_distribution : distribution_schemes 5 3 = 60 := by
  sorry


end NUMINAMATH_CALUDE_intern_teacher_distribution_l3123_312361


namespace NUMINAMATH_CALUDE_arccos_cos_eleven_l3123_312393

theorem arccos_cos_eleven : 
  Real.arccos (Real.cos 11) = 11 - 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_eleven_l3123_312393


namespace NUMINAMATH_CALUDE_example_rearrangements_l3123_312310

def word : String := "EXAMPLE"

def vowels : List Char := ['E', 'E', 'A']
def consonants : List Char := ['X', 'M', 'P', 'L']

def vowel_arrangements : ℕ := 3
def consonant_arrangements : ℕ := 24

theorem example_rearrangements :
  (vowel_arrangements * consonant_arrangements) = 72 :=
by sorry

end NUMINAMATH_CALUDE_example_rearrangements_l3123_312310


namespace NUMINAMATH_CALUDE_discount_tax_equivalence_l3123_312379

theorem discount_tax_equivalence (original_price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) :
  let discounted_price := original_price * (1 - discount_rate)
  let taxed_price := original_price * (1 + tax_rate)
  discounted_price * (1 + tax_rate) = taxed_price * (1 - discount_rate) :=
by sorry

#check discount_tax_equivalence 90 0.2 0.06

end NUMINAMATH_CALUDE_discount_tax_equivalence_l3123_312379


namespace NUMINAMATH_CALUDE_sin_transformation_l3123_312353

theorem sin_transformation (x : ℝ) : 
  Real.sin (2 * x + π / 3) = Real.sin (2 * (x - π / 12) + π / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_transformation_l3123_312353


namespace NUMINAMATH_CALUDE_monotone_increasing_interval_l3123_312357

/-- The function f(x) = ax / (x^2 + 1) is monotonically increasing on (-1, 1) for a > 0 -/
theorem monotone_increasing_interval (a : ℝ) (h : a > 0) :
  StrictMonoOn (fun x => a * x / (x^2 + 1)) (Set.Ioo (-1) 1) := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_interval_l3123_312357


namespace NUMINAMATH_CALUDE_no_valid_placement_l3123_312390

-- Define the chessboard
def Chessboard := Fin 8 × Fin 8

-- Define the piece types
inductive Piece
| Rook
| Knight
| Bishop

-- Define the placement function type
def Placement := Chessboard → Piece

-- Define the attack relations
def rook_attacks (a b : Chessboard) : Prop :=
  (a.1 = b.1 ∨ a.2 = b.2) ∧ a ≠ b

def knight_attacks (a b : Chessboard) : Prop :=
  (abs (a.1 - b.1) = 1 ∧ abs (a.2 - b.2) = 2) ∨
  (abs (a.1 - b.1) = 2 ∧ abs (a.2 - b.2) = 1)

def bishop_attacks (a b : Chessboard) : Prop :=
  abs (a.1 - b.1) = abs (a.2 - b.2) ∧ a ≠ b

-- Define the validity of a placement
def valid_placement (p : Placement) : Prop :=
  ∀ a b : Chessboard,
    (p a = Piece.Rook ∧ rook_attacks a b → p b = Piece.Knight) ∧
    (p a = Piece.Knight ∧ knight_attacks a b → p b = Piece.Bishop) ∧
    (p a = Piece.Bishop ∧ bishop_attacks a b → p b = Piece.Rook)

-- Theorem statement
theorem no_valid_placement : ¬∃ p : Placement, valid_placement p :=
  sorry

end NUMINAMATH_CALUDE_no_valid_placement_l3123_312390


namespace NUMINAMATH_CALUDE_rectangle_length_is_three_times_width_l3123_312348

/-- Represents the construction of a large square from six identical smaller squares and a rectangle -/
structure SquareConstruction where
  /-- Side length of each small square -/
  s : ℝ
  /-- Assertion that s is positive -/
  s_pos : 0 < s

/-- The length of the rectangle in the construction -/
def rectangleLength (c : SquareConstruction) : ℝ := 3 * c.s

/-- The width of the rectangle in the construction -/
def rectangleWidth (c : SquareConstruction) : ℝ := c.s

/-- The theorem stating that the length of the rectangle is 3 times its width -/
theorem rectangle_length_is_three_times_width (c : SquareConstruction) :
  rectangleLength c = 3 * rectangleWidth c := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_is_three_times_width_l3123_312348


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_13_l3123_312367

theorem greatest_three_digit_multiple_of_13 :
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 13 = 0 → n ≤ 988 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_13_l3123_312367


namespace NUMINAMATH_CALUDE_condition_for_reciprocal_inequality_l3123_312312

theorem condition_for_reciprocal_inequality (a : ℝ) :
  (∀ a, (1 / a > 1 → a < 1)) ∧ (∃ a, a < 1 ∧ 1 / a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_condition_for_reciprocal_inequality_l3123_312312


namespace NUMINAMATH_CALUDE_average_monthly_sales_l3123_312332

def january_sales : ℝ := 110
def february_sales : ℝ := 90
def march_sales : ℝ := 70
def april_sales : ℝ := 130
def may_sales : ℝ := 50
def total_months : ℕ := 5

def total_sales : ℝ := january_sales + february_sales + march_sales + april_sales + may_sales

theorem average_monthly_sales :
  total_sales / total_months = 90 := by sorry

end NUMINAMATH_CALUDE_average_monthly_sales_l3123_312332


namespace NUMINAMATH_CALUDE_expression_value_at_five_l3123_312320

theorem expression_value_at_five :
  let x : ℝ := 5
  (x^2 - 3*x - 4) / (x - 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_five_l3123_312320
