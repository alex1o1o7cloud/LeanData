import Mathlib

namespace NUMINAMATH_CALUDE_base_three_20121_equals_178_l667_66716

def base_three_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base_three_20121_equals_178 :
  base_three_to_ten [2, 0, 1, 2, 1] = 178 := by sorry

end NUMINAMATH_CALUDE_base_three_20121_equals_178_l667_66716


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l667_66750

def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count_theorem :
  sum_integers 20 40 + count_even_integers 20 40 = 641 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l667_66750


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l667_66717

theorem unique_positive_integer_solution : 
  ∃! (n : ℕ), n > 0 ∧ (30 - 6 * n > 18) := by sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l667_66717


namespace NUMINAMATH_CALUDE_earth_livable_fraction_l667_66743

/-- The fraction of the earth's surface not covered by water -/
def land_fraction : ℚ := 1/3

/-- The fraction of exposed land that is inhabitable -/
def inhabitable_fraction : ℚ := 1/3

/-- The fraction of the earth's surface that humans can live on -/
def livable_fraction : ℚ := land_fraction * inhabitable_fraction

theorem earth_livable_fraction :
  livable_fraction = 1/9 := by sorry

end NUMINAMATH_CALUDE_earth_livable_fraction_l667_66743


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l667_66704

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + 2*x*y = 8) :
  ∃ (m : ℝ), m = 4 ∧ ∀ z, z = x + 2*y → z ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l667_66704


namespace NUMINAMATH_CALUDE_least_nonprime_sum_l667_66713

theorem least_nonprime_sum (p : Nat) (h : Nat.Prime p) : ∃ (n : Nat), 
  (∀ (q : Nat), Nat.Prime q → ¬Nat.Prime (q^2 + n)) ∧ 
  (∀ (m : Nat), m < n → ∃ (r : Nat), Nat.Prime r ∧ Nat.Prime (r^2 + m)) :=
by
  sorry

#check least_nonprime_sum

end NUMINAMATH_CALUDE_least_nonprime_sum_l667_66713


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l667_66741

theorem contrapositive_equivalence (x : ℝ) :
  (x = 1 → x^2 - 3*x + 2 = 0) ↔ (x^2 - 3*x + 2 ≠ 0 → x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l667_66741


namespace NUMINAMATH_CALUDE_inequality_always_holds_l667_66724

theorem inequality_always_holds (α : ℝ) : 4 * Real.sin (3 * α) + 5 ≥ 4 * Real.cos (2 * α) + 5 * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l667_66724


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l667_66747

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 - a*b + b^2 = 0) :
  (a^6 + b^6) / (a + b)^6 = 1/18 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l667_66747


namespace NUMINAMATH_CALUDE_julia_remaining_money_l667_66712

def initial_amount : ℚ := 40
def game_fraction : ℚ := 1/2
def in_game_purchase_fraction : ℚ := 1/4

theorem julia_remaining_money :
  let amount_after_game := initial_amount * (1 - game_fraction)
  let final_amount := amount_after_game * (1 - in_game_purchase_fraction)
  final_amount = 15 := by sorry

end NUMINAMATH_CALUDE_julia_remaining_money_l667_66712


namespace NUMINAMATH_CALUDE_mosaic_configurations_l667_66765

/-- Represents a tile in the mosaic --/
inductive Tile
| small : Tile  -- 1×1 tile
| large : Tile  -- 1×2 tile

/-- Represents a digit in the number 2021 --/
inductive Digit
| two : Digit
| zero : Digit
| one : Digit

/-- The number of cells used by each digit --/
def digit_cells (d : Digit) : Nat :=
  match d with
  | Digit.two => 13
  | Digit.zero => 18
  | Digit.one => 8

/-- The total number of tiles available --/
def available_tiles : Nat × Nat := (4, 24)  -- (small tiles, large tiles)

/-- A configuration of tiles for a single digit --/
def DigitConfiguration := List Tile

/-- A configuration of tiles for the entire number 2021 --/
def NumberConfiguration := List DigitConfiguration

/-- Checks if a digit configuration is valid for a given digit --/
def is_valid_digit_config (d : Digit) (config : DigitConfiguration) : Prop := sorry

/-- Checks if a number configuration is valid --/
def is_valid_number_config (config : NumberConfiguration) : Prop := sorry

/-- Counts the number of valid configurations --/
def count_valid_configs : Nat := sorry

/-- The main theorem --/
theorem mosaic_configurations :
  count_valid_configs = 6517 := sorry

end NUMINAMATH_CALUDE_mosaic_configurations_l667_66765


namespace NUMINAMATH_CALUDE_paint_mixer_production_time_l667_66729

/-- A paint mixer's production rate and time to complete a job -/
theorem paint_mixer_production_time 
  (days_for_some_drums : ℕ) 
  (total_drums : ℕ) 
  (total_days : ℕ) 
  (h1 : days_for_some_drums = 3)
  (h2 : total_drums = 360)
  (h3 : total_days = 60) :
  total_days = total_drums / (total_drums / total_days) :=
by sorry

end NUMINAMATH_CALUDE_paint_mixer_production_time_l667_66729


namespace NUMINAMATH_CALUDE_y_intercept_of_specific_line_l667_66797

/-- A line is defined by its slope and a point it passes through -/
structure Line where
  slope : ℚ
  point : ℚ × ℚ

/-- The y-intercept of a line is the y-coordinate where the line crosses the y-axis -/
def y_intercept (l : Line) : ℚ := 
  l.point.2 - l.slope * l.point.1

theorem y_intercept_of_specific_line : 
  let l : Line := { slope := -3/2, point := (4, 0) }
  y_intercept l = 6 := by
  sorry

#check y_intercept_of_specific_line

end NUMINAMATH_CALUDE_y_intercept_of_specific_line_l667_66797


namespace NUMINAMATH_CALUDE_mode_is_nine_l667_66791

def digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def frequency : Nat → Nat
| 0 => 8
| 1 => 8
| 2 => 12
| 3 => 11
| 4 => 10
| 5 => 8
| 6 => 9
| 7 => 8
| 8 => 12
| 9 => 14
| _ => 0

def is_mode (x : Nat) : Prop :=
  x ∈ digits ∧ ∀ y ∈ digits, frequency x ≥ frequency y

theorem mode_is_nine : is_mode 9 := by
  sorry

end NUMINAMATH_CALUDE_mode_is_nine_l667_66791


namespace NUMINAMATH_CALUDE_least_pennies_count_l667_66789

theorem least_pennies_count (a : ℕ) : 
  (a > 0) → 
  (a % 7 = 1) → 
  (a % 3 = 0) → 
  (∀ b : ℕ, b > 0 → b % 7 = 1 → b % 3 = 0 → a ≤ b) → 
  a = 15 := by
sorry

end NUMINAMATH_CALUDE_least_pennies_count_l667_66789


namespace NUMINAMATH_CALUDE_original_class_size_l667_66754

theorem original_class_size (x : ℕ) : 
  (40 * x + 15 * 32 = (x + 15) * 36) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_original_class_size_l667_66754


namespace NUMINAMATH_CALUDE_map_scale_theorem_l667_66782

/-- Represents the scale of a map as a ratio of 1 to some natural number. -/
structure MapScale where
  ratio : ℕ
  property : ratio > 0

/-- Calculates the map scale given the real distance and the corresponding map distance. -/
def calculate_map_scale (real_distance : ℕ) (map_distance : ℕ) : MapScale :=
  { ratio := real_distance / map_distance
    property := sorry }

theorem map_scale_theorem (real_km : ℕ) (map_cm : ℕ) 
  (h1 : real_km = 30) (h2 : map_cm = 20) : 
  (calculate_map_scale (real_km * 100000) map_cm).ratio = 150000 := by
  sorry

#check map_scale_theorem

end NUMINAMATH_CALUDE_map_scale_theorem_l667_66782


namespace NUMINAMATH_CALUDE_added_number_proof_l667_66749

theorem added_number_proof (x : ℝ) : 
  (((2 * (62.5 + x)) / 5) - 5 = 22) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_added_number_proof_l667_66749


namespace NUMINAMATH_CALUDE_identical_car_in_kindergarten_l667_66730

-- Define the properties of a car
structure Car where
  color : String
  size : String
  hasTrailer : Bool

-- Define the boys and their car collections
def Misha : List Car := [
  { color := "green", size := "small", hasTrailer := false },
  { color := "unknown", size := "small", hasTrailer := false },
  { color := "unknown", size := "unknown", hasTrailer := true }
]

def Vitya : List Car := [
  { color := "unknown", size := "unknown", hasTrailer := false },
  { color := "green", size := "small", hasTrailer := true }
]

def Kolya : List Car := [
  { color := "unknown", size := "big", hasTrailer := false },
  { color := "blue", size := "small", hasTrailer := true }
]

-- Define the theorem
theorem identical_car_in_kindergarten :
  ∃ (c : Car),
    c ∈ Misha ∧ c ∈ Vitya ∧ c ∈ Kolya ∧
    c.color = "green" ∧ c.size = "big" ∧ c.hasTrailer = false :=
by
  sorry

end NUMINAMATH_CALUDE_identical_car_in_kindergarten_l667_66730


namespace NUMINAMATH_CALUDE_quadratic_function_unique_form_l667_66751

/-- A quadratic function is a function of the form f(x) = ax² + bx + c where a ≠ 0 -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_unique_form
  (f : ℝ → ℝ)
  (hquad : QuadraticFunction f)
  (hf0 : f 0 = 1)
  (hfdiff : ∀ x, f (x + 1) - f x = 4 * x) :
  ∀ x, f x = 2 * x^2 - 2 * x + 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_form_l667_66751


namespace NUMINAMATH_CALUDE_cube_root_problem_l667_66768

theorem cube_root_problem (a : ℕ) : a^3 = 21 * 25 * 45 * 49 → a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l667_66768


namespace NUMINAMATH_CALUDE_special_number_prime_iff_l667_66764

/-- Represents a natural number formed by one digit 7 and n-1 digits 1 -/
def special_number (n : ℕ) : ℕ :=
  7 * 10^(n-1) + (10^(n-1) - 1) / 9

/-- Predicate to check if a natural number is prime -/
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

/-- The main theorem stating that only n = 1 and n = 2 satisfy the condition -/
theorem special_number_prime_iff (n : ℕ) :
  (∀ k : ℕ, k ≤ n → is_prime (special_number k)) ↔ n = 1 ∨ n = 2 :=
sorry

end NUMINAMATH_CALUDE_special_number_prime_iff_l667_66764


namespace NUMINAMATH_CALUDE_average_of_w_x_z_l667_66787

theorem average_of_w_x_z (w x y z a : ℝ) 
  (h1 : 2/w + 2/x + 2/z = 2/y)
  (h2 : w*x*z = y)
  (h3 : w + x + z = a) :
  (w + x + z) / 3 = a / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_w_x_z_l667_66787


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l667_66705

theorem quadratic_roots_relation (b c : ℝ) : 
  (∃ p q : ℝ, 
    (3 * p^2 - 5 * p - 7 = 0) ∧ 
    (3 * q^2 - 5 * q - 7 = 0) ∧ 
    ((p + 2)^2 + b * (p + 2) + c = 0) ∧
    ((q + 2)^2 + b * (q + 2) + c = 0)) →
  c = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l667_66705


namespace NUMINAMATH_CALUDE_solve_system_l667_66790

theorem solve_system (u v : ℝ) 
  (eq1 : 3 * u - 7 * v = 29)
  (eq2 : 5 * u + 3 * v = -9) :
  u + v = -3.363 := by sorry

end NUMINAMATH_CALUDE_solve_system_l667_66790


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l667_66767

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

-- Part 1
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) → (2 < x ∧ x < 3) :=
sorry

-- Part 2
theorem range_of_a_when_not_p_implies_not_q :
  ∀ a : ℝ, (∀ x : ℝ, ¬(p x a) → (x ≤ 2 ∨ x > 3)) ∧
           (∃ x : ℝ, (x ≤ 2 ∨ x > 3) ∧ p x a) →
  (1 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l667_66767


namespace NUMINAMATH_CALUDE_orange_marble_probability_l667_66744

/-- Represents the contents of a bag of marbles -/
structure BagContents where
  white : ℕ := 0
  black : ℕ := 0
  orange : ℕ := 0
  green : ℕ := 0
  blue : ℕ := 0

/-- Calculates the total number of marbles in a bag -/
def BagContents.total (bag : BagContents) : ℕ :=
  bag.white + bag.black + bag.orange + bag.green + bag.blue

/-- The contents of Bag A -/
def bagA : BagContents := ⟨4, 5, 0, 0, 0⟩

/-- The contents of Bag B -/
def bagB : BagContents := ⟨0, 0, 7, 5, 3⟩

/-- The contents of Bag C -/
def bagC : BagContents := ⟨0, 0, 4, 4, 2⟩

/-- The probability of drawing an orange marble as the second marble -/
theorem orange_marble_probability :
  let probWhiteA := bagA.white / bagA.total
  let probBlackA := bagA.black / bagA.total
  let probOrangeB := bagB.orange / bagB.total
  let probOrangeC := bagC.orange / bagC.total
  probWhiteA * probOrangeB + probBlackA * probOrangeC = 58 / 135 := by
  sorry


end NUMINAMATH_CALUDE_orange_marble_probability_l667_66744


namespace NUMINAMATH_CALUDE_number_of_c_animals_l667_66703

/-- Given the number of (A) and (B) animals, and the relationship between (A), (B), and (C) animals,
    prove that the number of (C) animals is 5. -/
theorem number_of_c_animals (a b : ℕ) (h1 : a = 45) (h2 : b = 32) 
    (h3 : b + c = a - 8) : c = 5 :=
by sorry

end NUMINAMATH_CALUDE_number_of_c_animals_l667_66703


namespace NUMINAMATH_CALUDE_concert_attendance_l667_66734

-- Define the total number of people at the concert
variable (P : ℕ)

-- Define the conditions
def second_band_audience : ℚ := 2/3
def first_band_audience : ℚ := 1/3
def under_30_second_band : ℚ := 1/2
def women_under_30_second_band : ℚ := 3/5
def men_under_30_second_band : ℕ := 20

-- Theorem statement
theorem concert_attendance : 
  second_band_audience + first_band_audience = 1 →
  (second_band_audience * under_30_second_band * (1 - women_under_30_second_band)) * P = men_under_30_second_band →
  P = 150 :=
by sorry

end NUMINAMATH_CALUDE_concert_attendance_l667_66734


namespace NUMINAMATH_CALUDE_batsman_total_score_l667_66756

/-- Represents a batsman's score in cricket -/
structure BatsmanScore where
  boundaries : ℕ
  sixes : ℕ
  runningPercentage : ℚ

/-- Calculates the total score of a batsman -/
def totalScore (score : BatsmanScore) : ℕ :=
  sorry

theorem batsman_total_score (score : BatsmanScore) 
  (h1 : score.boundaries = 6)
  (h2 : score.sixes = 4)
  (h3 : score.runningPercentage = 60/100) :
  totalScore score = 120 := by
  sorry

end NUMINAMATH_CALUDE_batsman_total_score_l667_66756


namespace NUMINAMATH_CALUDE_water_depth_l667_66783

theorem water_depth (ron_height dean_height water_depth : ℕ) : 
  ron_height = 13 →
  dean_height = ron_height + 4 →
  water_depth = 15 * dean_height →
  water_depth = 255 := by
sorry

end NUMINAMATH_CALUDE_water_depth_l667_66783


namespace NUMINAMATH_CALUDE_elena_garden_lilies_l667_66722

/-- The number of lilies in Elena's garden -/
def num_lilies : ℕ := 8

/-- The number of tulips in Elena's garden -/
def num_tulips : ℕ := 5

/-- The number of petals each lily has -/
def petals_per_lily : ℕ := 6

/-- The number of petals each tulip has -/
def petals_per_tulip : ℕ := 3

/-- The total number of petals in Elena's garden -/
def total_petals : ℕ := 63

theorem elena_garden_lilies :
  num_lilies * petals_per_lily + num_tulips * petals_per_tulip = total_petals :=
by sorry

end NUMINAMATH_CALUDE_elena_garden_lilies_l667_66722


namespace NUMINAMATH_CALUDE_function_is_identity_l667_66733

/-- A function satisfying the given functional equation for all positive real numbers -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    (z + 1) * f (x + y) = f (x * f z + y) + f (y * f z + x)

/-- The main theorem: if f satisfies the equation, then f(x) = x for all positive real numbers -/
theorem function_is_identity {f : ℝ → ℝ} (hf : SatisfiesEquation f) :
    ∀ x : ℝ, x > 0 → f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_is_identity_l667_66733


namespace NUMINAMATH_CALUDE_zorg_game_threshold_l667_66778

theorem zorg_game_threshold : ∃ (n : ℕ), n = 40 ∧ ∀ (m : ℕ), m < n → (m * (m + 1)) / 2 ≤ 20 * m :=
by sorry

end NUMINAMATH_CALUDE_zorg_game_threshold_l667_66778


namespace NUMINAMATH_CALUDE_quiz_goal_achievement_l667_66721

theorem quiz_goal_achievement (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (current_as : ℕ) : 
  total_quizzes = 60 →
  goal_percentage = 3/4 →
  completed_quizzes = 40 →
  current_as = 27 →
  (total_quizzes - completed_quizzes) - 
    (↑(total_quizzes) * goal_percentage - current_as).ceil = 2 := by
  sorry

end NUMINAMATH_CALUDE_quiz_goal_achievement_l667_66721


namespace NUMINAMATH_CALUDE_nested_radical_value_l667_66775

theorem nested_radical_value :
  ∃ y : ℝ, y = Real.sqrt (4 + y) → y = (1 + Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l667_66775


namespace NUMINAMATH_CALUDE_tetrahedron_volume_and_height_l667_66735

/-- Tetrahedron with given vertices -/
structure Tetrahedron where
  A₁ : ℝ × ℝ × ℝ := (-1, 2, -3)
  A₂ : ℝ × ℝ × ℝ := (4, -1, 0)
  A₃ : ℝ × ℝ × ℝ := (2, 1, -2)
  A₄ : ℝ × ℝ × ℝ := (3, 4, 5)

/-- Calculate the volume of the tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ := by sorry

/-- Calculate the height from A₄ to face A₁A₂A₃ -/
def tetrahedronHeight (t : Tetrahedron) : ℝ := by sorry

/-- Main theorem: Volume and height of the tetrahedron -/
theorem tetrahedron_volume_and_height (t : Tetrahedron) :
  tetrahedronVolume t = 20 / 3 ∧ tetrahedronHeight t = 5 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_and_height_l667_66735


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l667_66700

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ x y : ℝ, x = -2 ∧ y = 6 ∧ a^(x + 2) + 5 = y :=
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l667_66700


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l667_66763

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem stating the relationship between a₄, a₇, and a₁₀ in a geometric sequence -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : geometric_sequence a) :
  a 4 * a 10 = 9 → a 7 = 3 ∨ a 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l667_66763


namespace NUMINAMATH_CALUDE_range_of_a_l667_66723

theorem range_of_a (x y a : ℝ) (h1 : x + y + 3 = x * y) (h2 : x > 0) (h3 : y > 0)
  (h4 : ∀ x y : ℝ, x > 0 → y > 0 → x + y + 3 = x * y → (x + y)^2 - a*(x + y) + 1 ≥ 0) :
  a ≤ 37/6 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l667_66723


namespace NUMINAMATH_CALUDE_no_intersection_l667_66795

/-- The function representing y = |3x + 6| -/
def f (x : ℝ) : ℝ := |3 * x + 6|

/-- The function representing y = -|4x - 3| -/
def g (x : ℝ) : ℝ := -|4 * x - 3|

/-- Theorem stating that there are no intersection points between f and g -/
theorem no_intersection :
  ¬∃ (x : ℝ), f x = g x :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l667_66795


namespace NUMINAMATH_CALUDE_desired_interest_percentage_l667_66732

theorem desired_interest_percentage 
  (face_value : ℝ) 
  (dividend_rate : ℝ) 
  (market_value : ℝ) 
  (h1 : face_value = 56) 
  (h2 : dividend_rate = 0.09) 
  (h3 : market_value = 42) : 
  (dividend_rate * face_value) / market_value = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_desired_interest_percentage_l667_66732


namespace NUMINAMATH_CALUDE_cricket_captain_age_l667_66770

theorem cricket_captain_age (team_size : ℕ) (captain_age wicket_keeper_age : ℕ) 
  (team_average : ℚ) (remaining_average : ℚ) :
  team_size = 11 →
  wicket_keeper_age = captain_age + 3 →
  team_average = 25 →
  remaining_average = team_average - 1 →
  (team_size : ℚ) * team_average = 
    (team_size - 2 : ℚ) * remaining_average + captain_age + wicket_keeper_age →
  captain_age = 28 := by
  sorry

end NUMINAMATH_CALUDE_cricket_captain_age_l667_66770


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l667_66762

/-- An isosceles triangle with side lengths 3 and 4 has a perimeter of either 10 or 11 -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), 
  (a = 3 ∨ a = 4) → 
  (b = 3 ∨ b = 4) → 
  (c = 3 ∨ c = 4) →
  (a = b ∨ b = c ∨ a = c) → 
  (a + b > c ∧ b + c > a ∧ a + c > b) →
  (a + b + c = 10 ∨ a + b + c = 11) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l667_66762


namespace NUMINAMATH_CALUDE_tan_105_degrees_l667_66726

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l667_66726


namespace NUMINAMATH_CALUDE_coin_jar_problem_l667_66793

theorem coin_jar_problem (y : ℕ) :
  (5 * y + 10 * y + 25 * y = 1440) → y = 36 :=
by sorry

end NUMINAMATH_CALUDE_coin_jar_problem_l667_66793


namespace NUMINAMATH_CALUDE_birthday_pigeonhole_l667_66798

theorem birthday_pigeonhole (n : ℕ) (h : n = 50) :
  ∃ (m : ℕ) (S : Finset (Fin n)), S.card ≥ 5 ∧ (∀ i ∈ S, (i : ℕ) % 12 + 1 = m) :=
sorry

end NUMINAMATH_CALUDE_birthday_pigeonhole_l667_66798


namespace NUMINAMATH_CALUDE_sodium_chloride_formation_l667_66725

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  product3 : String

-- Define the molar quantities
def moles_NaHSO3 : ℚ := 2
def moles_HCl : ℚ := 2

-- Define the reaction
def sodium_bisulfite_reaction : Reaction :=
  { reactant1 := "NaHSO3"
  , reactant2 := "HCl"
  , product1 := "NaCl"
  , product2 := "H2O"
  , product3 := "SO2" }

-- Theorem statement
theorem sodium_chloride_formation 
  (r : Reaction) 
  (h1 : r = sodium_bisulfite_reaction) 
  (h2 : moles_NaHSO3 = moles_HCl) :
  moles_NaHSO3 = 2 → 2 = (let moles_NaCl := moles_NaHSO3; moles_NaCl) :=
by
  sorry

end NUMINAMATH_CALUDE_sodium_chloride_formation_l667_66725


namespace NUMINAMATH_CALUDE_negation_of_p_l667_66748

-- Define the set M
def M : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define the original proposition p
def p : Prop := ∃ x ∈ M, x^2 - x - 2 < 0

-- Statement: The negation of p is equivalent to ∀x ∈ M, x^2 - x - 2 ≥ 0
theorem negation_of_p : ¬p ↔ ∀ x ∈ M, x^2 - x - 2 ≥ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_p_l667_66748


namespace NUMINAMATH_CALUDE_student_transportation_l667_66786

theorem student_transportation (total : ℚ) 
  (bus car scooter skateboard : ℚ) 
  (h1 : total = 1)
  (h2 : bus = 1/3)
  (h3 : car = 1/5)
  (h4 : scooter = 1/6)
  (h5 : skateboard = 1/8) :
  total - (bus + car + scooter + skateboard) = 7/40 := by
  sorry

end NUMINAMATH_CALUDE_student_transportation_l667_66786


namespace NUMINAMATH_CALUDE_no_equal_sum_partition_l667_66759

/-- A group of four consecutive natural numbers -/
structure NumberGroup :=
  (start : ℕ)
  (h : start > 0 ∧ start ≤ 69)

/-- The product of four consecutive natural numbers starting from n -/
def groupProduct (g : NumberGroup) : ℕ :=
  g.start * (g.start + 1) * (g.start + 2) * (g.start + 3)

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- A partition of 72 consecutive natural numbers into 18 groups -/
def Partition := Fin 18 → NumberGroup

/-- The theorem stating that no partition exists where all groups have the same sum of digits of their product -/
theorem no_equal_sum_partition :
  ¬ ∃ (p : Partition), ∃ (s : ℕ), ∀ i : Fin 18, sumOfDigits (groupProduct (p i)) = s :=
sorry

end NUMINAMATH_CALUDE_no_equal_sum_partition_l667_66759


namespace NUMINAMATH_CALUDE_total_hangers_count_l667_66752

def pink_hangers : ℕ := 7
def green_hangers : ℕ := 4
def blue_hangers : ℕ := green_hangers - 1
def yellow_hangers : ℕ := blue_hangers - 1
def orange_hangers : ℕ := 2 * pink_hangers
def purple_hangers : ℕ := yellow_hangers + 3
def red_hangers : ℕ := purple_hangers / 2

theorem total_hangers_count :
  pink_hangers + green_hangers + blue_hangers + yellow_hangers +
  orange_hangers + purple_hangers + red_hangers = 37 := by
  sorry

end NUMINAMATH_CALUDE_total_hangers_count_l667_66752


namespace NUMINAMATH_CALUDE_equation_solution_exists_l667_66788

theorem equation_solution_exists : ∃ (MA TE TI KA : ℕ),
  MA < 10 ∧ TE < 10 ∧ TI < 10 ∧ KA < 10 ∧
  MA ≠ TE ∧ MA ≠ TI ∧ MA ≠ KA ∧ TE ≠ TI ∧ TE ≠ KA ∧ TI ≠ KA ∧
  MA * TE * MA * TI * KA = 2016000 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l667_66788


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l667_66771

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (2 * x + 9) = 11 → x = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l667_66771


namespace NUMINAMATH_CALUDE_tablet_count_l667_66766

theorem tablet_count : 
  ∀ (n : ℕ) (x y : ℕ),
  -- Lenovo (x), Samsung (x+6), and Huawei (y) make up less than a third of the total
  (2*x + y + 6 < n/3) →
  -- Apple iPads are three times as many as Huawei tablets
  (n - 2*x - y - 6 = 3*y) →
  -- If Lenovo tablets were tripled, there would be 59 Apple iPads
  (n - 3*x - (x+6) - y = 59) →
  (n = 94) := by
sorry

end NUMINAMATH_CALUDE_tablet_count_l667_66766


namespace NUMINAMATH_CALUDE_percentage_calculation_l667_66799

theorem percentage_calculation (P : ℝ) : 
  0.15 * 0.30 * P * 5600 = 126 → P = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l667_66799


namespace NUMINAMATH_CALUDE_volume_increase_rectangular_prism_l667_66737

theorem volume_increase_rectangular_prism 
  (l w h : ℝ) 
  (l_increase : ℝ) 
  (w_increase : ℝ) 
  (h_increase : ℝ) 
  (hl : l_increase = 0.15) 
  (hw : w_increase = 0.20) 
  (hh : h_increase = 0.10) :
  let new_volume := (l * (1 + l_increase)) * (w * (1 + w_increase)) * (h * (1 + h_increase))
  let original_volume := l * w * h
  let volume_increase_percentage := (new_volume - original_volume) / original_volume * 100
  volume_increase_percentage = 51.8 := by
sorry

end NUMINAMATH_CALUDE_volume_increase_rectangular_prism_l667_66737


namespace NUMINAMATH_CALUDE_amoeba_count_14_days_l667_66719

/-- Calculates the number of amoebas after a given number of days -/
def amoeba_count (days : ℕ) : ℕ :=
  if days ≤ 2 then 2^(days - 1)
  else 5 * 2^(days - 3)

/-- The number of amoebas after 14 days is 10240 -/
theorem amoeba_count_14_days : amoeba_count 14 = 10240 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_14_days_l667_66719


namespace NUMINAMATH_CALUDE_bill_caroline_age_ratio_l667_66731

/-- Given the ages of Bill and Caroline, prove their age ratio -/
theorem bill_caroline_age_ratio :
  ∀ (bill_age caroline_age : ℕ),
  bill_age = 17 →
  bill_age + caroline_age = 26 →
  ∃ (n : ℕ), bill_age = n * caroline_age - 1 →
  (bill_age : ℚ) / caroline_age = 17 / 9 := by
  sorry

end NUMINAMATH_CALUDE_bill_caroline_age_ratio_l667_66731


namespace NUMINAMATH_CALUDE_milk_water_ratio_l667_66757

theorem milk_water_ratio (initial_volume : ℝ) (water_added : ℝ) 
  (milk : ℝ) (water : ℝ) : 
  initial_volume = 115 →
  water_added = 46 →
  milk + water = initial_volume →
  milk / (water + water_added) = 3 / 4 →
  milk / water = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_milk_water_ratio_l667_66757


namespace NUMINAMATH_CALUDE_total_nail_polishes_l667_66772

/-- The number of nail polishes each person has -/
structure NailPolishes where
  kim : ℕ
  heidi : ℕ
  karen : ℕ
  laura : ℕ
  simon : ℕ

/-- The conditions of the nail polish problem -/
def nail_polish_conditions (np : NailPolishes) : Prop :=
  np.kim = 25 ∧
  np.heidi = np.kim + 8 ∧
  np.karen = np.kim - 6 ∧
  np.laura = 2 * np.kim ∧
  np.simon = (np.kim / 2 + 10)

/-- The theorem stating the total number of nail polishes -/
theorem total_nail_polishes (np : NailPolishes) :
  nail_polish_conditions np →
  np.heidi + np.karen + np.laura + np.simon = 125 :=
by
  sorry


end NUMINAMATH_CALUDE_total_nail_polishes_l667_66772


namespace NUMINAMATH_CALUDE_lcm_4_8_9_10_l667_66708

theorem lcm_4_8_9_10 : Nat.lcm 4 (Nat.lcm 8 (Nat.lcm 9 10)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_4_8_9_10_l667_66708


namespace NUMINAMATH_CALUDE_symmetric_point_6_1_l667_66755

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The origin point (0, 0) -/
def origin : Point2D := ⟨0, 0⟩

/-- Function to find the symmetric point with respect to the origin -/
def symmetricPoint (p : Point2D) : Point2D :=
  ⟨-p.x, -p.y⟩

/-- Theorem: The point symmetric to (6, 1) with respect to the origin is (-6, -1) -/
theorem symmetric_point_6_1 :
  symmetricPoint ⟨6, 1⟩ = ⟨-6, -1⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_6_1_l667_66755


namespace NUMINAMATH_CALUDE_sum_of_roots_equal_one_l667_66728

theorem sum_of_roots_equal_one : 
  ∃ (x₁ x₂ : ℝ), (x₁ + 2) * (x₁ - 3) = 16 ∧ 
                 (x₂ + 2) * (x₂ - 3) = 16 ∧ 
                 x₁ + x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equal_one_l667_66728


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l667_66745

theorem square_garden_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 200 →
  side^2 = area →
  perimeter = 4 * side →
  perimeter = 40 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l667_66745


namespace NUMINAMATH_CALUDE_wrench_force_calculation_l667_66776

/-- Given two wrenches with different handle lengths, calculate the force required for the second wrench -/
theorem wrench_force_calculation (l₁ l₂ f₁ : ℝ) (h₁ : l₁ > 0) (h₂ : l₂ > 0) (h₃ : f₁ > 0) :
  let f₂ := (l₁ * f₁) / l₂
  l₁ = 12 ∧ f₁ = 450 ∧ l₂ = 18 → f₂ = 300 := by
  sorry

#check wrench_force_calculation

end NUMINAMATH_CALUDE_wrench_force_calculation_l667_66776


namespace NUMINAMATH_CALUDE_magnitude_z_equals_magnitude_iz_l667_66792

theorem magnitude_z_equals_magnitude_iz (z : ℂ) : Complex.abs z = Complex.abs (Complex.I * z) := by
  sorry

end NUMINAMATH_CALUDE_magnitude_z_equals_magnitude_iz_l667_66792


namespace NUMINAMATH_CALUDE_calories_per_bar_is_48_l667_66707

-- Define the total number of calories
def total_calories : ℕ := 2016

-- Define the number of candy bars
def num_candy_bars : ℕ := 42

-- Define the function to calculate calories per candy bar
def calories_per_bar : ℚ := total_calories / num_candy_bars

-- Theorem to prove
theorem calories_per_bar_is_48 : calories_per_bar = 48 := by
  sorry

end NUMINAMATH_CALUDE_calories_per_bar_is_48_l667_66707


namespace NUMINAMATH_CALUDE_wendy_cupcakes_l667_66714

/-- Represents the number of pastries in Wendy's bake sale scenario -/
structure BakeSale where
  cupcakes : ℕ
  cookies : ℕ
  pastries_left : ℕ
  pastries_sold : ℕ

/-- The theorem stating the number of cupcakes Wendy baked -/
theorem wendy_cupcakes (b : BakeSale) 
  (h1 : b.cupcakes + b.cookies = b.pastries_left + b.pastries_sold)
  (h2 : b.cookies = 29)
  (h3 : b.pastries_left = 24)
  (h4 : b.pastries_sold = 9) :
  b.cupcakes = 4 := by
  sorry

end NUMINAMATH_CALUDE_wendy_cupcakes_l667_66714


namespace NUMINAMATH_CALUDE_bruce_shopping_theorem_l667_66760

/-- Calculates the remaining money after Bruce's shopping trip. -/
def remaining_money (initial_amount : ℕ) (shirt_price : ℕ) (num_shirts : ℕ) (pants_price : ℕ) : ℕ :=
  initial_amount - (shirt_price * num_shirts + pants_price)

/-- Theorem stating that Bruce has $20 left after his shopping trip. -/
theorem bruce_shopping_theorem :
  remaining_money 71 5 5 26 = 20 := by
  sorry

#eval remaining_money 71 5 5 26

end NUMINAMATH_CALUDE_bruce_shopping_theorem_l667_66760


namespace NUMINAMATH_CALUDE_no_triangle_with_given_conditions_l667_66709

theorem no_triangle_with_given_conditions : ¬∃ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧  -- positive sides
  (c = 0.2 * a) ∧            -- shortest side is 20% of longest
  (b = 0.25 * (a + b + c)) ∧ -- third side is 25% of perimeter
  (a + b > c ∧ a + c > b ∧ b + c > a) -- triangle inequality
  := by sorry

end NUMINAMATH_CALUDE_no_triangle_with_given_conditions_l667_66709


namespace NUMINAMATH_CALUDE_hard_candy_food_coloring_l667_66794

/-- Proves that the amount of food colouring needed for each hard candy is 30ml -/
theorem hard_candy_food_coloring (lollipop_coloring : ℕ) (gummy_coloring : ℕ) 
  (num_lollipops : ℕ) (num_hard_candies : ℕ) (num_gummy_candies : ℕ) 
  (total_coloring : ℕ) :
  lollipop_coloring = 8 →
  gummy_coloring = 3 →
  num_lollipops = 150 →
  num_hard_candies = 20 →
  num_gummy_candies = 50 →
  total_coloring = 1950 →
  (total_coloring - (num_lollipops * lollipop_coloring + num_gummy_candies * gummy_coloring)) / num_hard_candies = 30 :=
by sorry

end NUMINAMATH_CALUDE_hard_candy_food_coloring_l667_66794


namespace NUMINAMATH_CALUDE_range_of_m_l667_66758

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*m*x + 1 ≠ 0) →  -- negation of q
  (abs (m + 1) ≤ 2) →               -- p
  (-1 < m ∧ m < 1) := by            -- conclusion
sorry

end NUMINAMATH_CALUDE_range_of_m_l667_66758


namespace NUMINAMATH_CALUDE_arithmetic_sequence_unique_n_l667_66769

/-- An arithmetic sequence with n terms, where a₁ is the first term and d is the common difference. -/
structure ArithmeticSequence where
  n : ℕ
  a₁ : ℚ
  d : ℚ

/-- The sum of the first k terms of an arithmetic sequence. -/
def sum_first_k (seq : ArithmeticSequence) (k : ℕ) : ℚ :=
  k / 2 * (2 * seq.a₁ + (k - 1) * seq.d)

/-- The sum of the last k terms of an arithmetic sequence. -/
def sum_last_k (seq : ArithmeticSequence) (k : ℕ) : ℚ :=
  k / 2 * (2 * (seq.a₁ + (seq.n - 1) * seq.d) - (k - 1) * seq.d)

/-- The sum of all terms in an arithmetic sequence. -/
def sum_all (seq : ArithmeticSequence) : ℚ :=
  seq.n / 2 * (2 * seq.a₁ + (seq.n - 1) * seq.d)

theorem arithmetic_sequence_unique_n :
  ∀ seq : ArithmeticSequence,
    sum_first_k seq 3 = 34 →
    sum_last_k seq 3 = 146 →
    sum_all seq = 390 →
    seq.n = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_unique_n_l667_66769


namespace NUMINAMATH_CALUDE_cousins_distribution_l667_66710

/-- The number of ways to distribute n indistinguishable objects into k distinct boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 5 cousins -/
def num_cousins : ℕ := 5

/-- There are 4 rooms -/
def num_rooms : ℕ := 4

/-- The number of ways to distribute the cousins into the rooms -/
def num_distributions : ℕ := distribute num_cousins num_rooms

theorem cousins_distribution :
  num_distributions = 66 := by sorry

end NUMINAMATH_CALUDE_cousins_distribution_l667_66710


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l667_66711

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) is on the line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line has equal intercepts on both axes -/
def Line.hasEqualIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c ≠ 0 ∧ l.c / l.a = - l.c / l.b

theorem line_through_point_with_equal_intercepts :
  ∃ (l : Line), l.contains (-1) 2 ∧ l.hasEqualIntercepts ∧
  ((l.a = 2 ∧ l.b = 1 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -1)) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l667_66711


namespace NUMINAMATH_CALUDE_largest_r_for_sequence_convergence_l667_66701

theorem largest_r_for_sequence_convergence (r : ℝ) : r > 2 →
  ∃ (a : ℕ → ℕ+), (∀ n, (a n : ℝ) ≤ a (n + 2) ∧ (a (n + 2) : ℝ) ≤ Real.sqrt ((a n : ℝ)^2 + r * (a (n + 1) : ℝ))) ∧
  ¬∃ M, ∀ n ≥ M, a (n + 2) = a n :=
by sorry

end NUMINAMATH_CALUDE_largest_r_for_sequence_convergence_l667_66701


namespace NUMINAMATH_CALUDE_worker_a_time_l667_66736

/-- Proves that Worker A takes 10 hours to do a job alone, given the conditions of the problem -/
theorem worker_a_time (time_b time_together : ℝ) : 
  time_b = 15 → 
  time_together = 6 → 
  (1 / 10 : ℝ) + (1 / time_b) = (1 / time_together) := by
  sorry

end NUMINAMATH_CALUDE_worker_a_time_l667_66736


namespace NUMINAMATH_CALUDE_f_m_plus_n_eq_zero_l667_66739

noncomputable def f (x : ℝ) : ℝ := x * Real.sin (abs x) + Real.log ((2019 - x) / (2019 + x))

theorem f_m_plus_n_eq_zero 
  (m n : ℝ) 
  (h1 : ∀ x ∈ Set.Icc (-2018 : ℝ) 2018, m < f x ∧ f x < n) 
  (h2 : ∀ x : ℝ, x ∉ Set.Icc (-2018 : ℝ) 2018 → ¬(m < f x ∧ f x < n)) :
  f (m + n) = 0 := by
sorry

end NUMINAMATH_CALUDE_f_m_plus_n_eq_zero_l667_66739


namespace NUMINAMATH_CALUDE_intersection_complement_equals_singleton_l667_66773

open Set

universe u

def U : Finset ℕ := {1, 2, 3, 4}

theorem intersection_complement_equals_singleton
  (A B : Finset ℕ)
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : (U \ (A ∪ B)) = {4})
  (h4 : B = {1, 2}) :
  A ∩ (U \ B) = {3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_singleton_l667_66773


namespace NUMINAMATH_CALUDE_student_ratio_proof_l667_66784

theorem student_ratio_proof (m n : ℕ) (a b : ℝ) (α β : ℝ) 
  (h1 : α = 3 / 4)
  (h2 : β = 19 / 20)
  (h3 : a = α * b)
  (h4 : a = β * (a * m + b * n) / (m + n)) :
  m / n = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_student_ratio_proof_l667_66784


namespace NUMINAMATH_CALUDE_cube_score_is_40_l667_66720

/-- Represents the score for a unit cube based on the number of painted faces. -/
def score (painted_faces : Nat) : Int :=
  match painted_faces with
  | 3 => 3
  | 2 => 2
  | 1 => 1
  | 0 => -7
  | _ => 0  -- This case should never occur in our problem

/-- The size of one side of the cube. -/
def cube_size : Nat := 4

/-- The total number of unit cubes in the large cube. -/
def total_cubes : Nat := cube_size ^ 3

/-- The number of corner cubes (with 3 painted faces). -/
def corner_cubes : Nat := 8

/-- The number of edge cubes (with 2 painted faces), excluding corners. -/
def edge_cubes : Nat := 12 * (cube_size - 2)

/-- The number of face cubes (with 1 painted face), excluding edges and corners. -/
def face_cubes : Nat := 6 * (cube_size - 2) ^ 2

/-- The number of internal cubes (with 0 painted faces). -/
def internal_cubes : Nat := (cube_size - 2) ^ 3

theorem cube_score_is_40 :
  (corner_cubes * score 3 +
   edge_cubes * score 2 +
   face_cubes * score 1 +
   internal_cubes * score 0) = 40 ∧
  corner_cubes + edge_cubes + face_cubes + internal_cubes = total_cubes :=
sorry

end NUMINAMATH_CALUDE_cube_score_is_40_l667_66720


namespace NUMINAMATH_CALUDE_max_k_value_l667_66780

theorem max_k_value (x y k : ℝ) (pos_x : x > 0) (pos_y : y > 0) (pos_k : k > 0)
  (eq_condition : 3 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 7) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l667_66780


namespace NUMINAMATH_CALUDE_janice_earnings_this_week_l667_66779

/-- Calculates Janice's weekly earnings based on her work schedule and wages -/
def janice_weekly_earnings (regular_days : ℕ) (regular_wage : ℕ) (overtime_shifts : ℕ) (overtime_bonus : ℕ) : ℕ :=
  regular_days * regular_wage + overtime_shifts * overtime_bonus

/-- Proves that Janice's weekly earnings are $195 given her work schedule -/
theorem janice_earnings_this_week :
  janice_weekly_earnings 5 30 3 15 = 195 := by
  sorry

#eval janice_weekly_earnings 5 30 3 15

end NUMINAMATH_CALUDE_janice_earnings_this_week_l667_66779


namespace NUMINAMATH_CALUDE_ratio_problem_l667_66742

theorem ratio_problem (A B C : ℝ) (h1 : A + B + C = 98) (h2 : B / C = 5 / 8) (h3 : B = 30) :
  A / B = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l667_66742


namespace NUMINAMATH_CALUDE_basketball_win_rate_l667_66785

theorem basketball_win_rate (initial_wins : Nat) (initial_games : Nat) 
  (remaining_games : Nat) (target_win_rate : Rat) :
  initial_wins = 35 →
  initial_games = 45 →
  remaining_games = 55 →
  target_win_rate = 3/4 →
  ∃ (remaining_wins : Nat),
    remaining_wins = 40 ∧
    (initial_wins + remaining_wins : Rat) / (initial_games + remaining_games) = target_win_rate :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_win_rate_l667_66785


namespace NUMINAMATH_CALUDE_gcd_sum_and_sum_squares_minus_product_l667_66718

theorem gcd_sum_and_sum_squares_minus_product (a b : ℤ) :
  Int.gcd a b = 1 → Int.gcd (a + b) (a^2 + b^2 - a*b) = 1 ∨ Int.gcd (a + b) (a^2 + b^2 - a*b) = 3 :=
by sorry

end NUMINAMATH_CALUDE_gcd_sum_and_sum_squares_minus_product_l667_66718


namespace NUMINAMATH_CALUDE_emerson_rowing_distance_l667_66746

/-- The total distance covered by Emerson on his rowing trip -/
def total_distance (first_part second_part third_part : ℕ) : ℕ :=
  first_part + second_part + third_part

/-- Theorem stating that Emerson's total rowing distance is 39 miles -/
theorem emerson_rowing_distance :
  total_distance 6 15 18 = 39 := by
  sorry

end NUMINAMATH_CALUDE_emerson_rowing_distance_l667_66746


namespace NUMINAMATH_CALUDE_range_of_a_for_equation_solution_l667_66740

theorem range_of_a_for_equation_solution (a : ℝ) : 
  (∃ x : ℝ, (a + Real.cos x) * (a - Real.sin x) = 1) ↔ 
  (a ∈ Set.Icc (-1 - Real.sqrt 2 / 2) (-1 + Real.sqrt 2 / 2) ∪ 
   Set.Icc (1 - Real.sqrt 2 / 2) (1 + Real.sqrt 2 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_equation_solution_l667_66740


namespace NUMINAMATH_CALUDE_finite_solutions_of_exponential_equation_l667_66702

theorem finite_solutions_of_exponential_equation :
  ∃ (S : Finset (ℕ × ℕ × ℕ × ℕ)),
    ∀ (x y z n : ℕ), (2^x : ℕ) + 5^y - 31^z = n.factorial → (x, y, z, n) ∈ S := by
  sorry

end NUMINAMATH_CALUDE_finite_solutions_of_exponential_equation_l667_66702


namespace NUMINAMATH_CALUDE_van_rental_cost_equation_l667_66753

theorem van_rental_cost_equation (x : ℝ) (h : x > 2) :
  180 / (x - 2) - 180 / x = 3 :=
sorry

end NUMINAMATH_CALUDE_van_rental_cost_equation_l667_66753


namespace NUMINAMATH_CALUDE_unique_intersection_point_l667_66777

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 5*x^2 + 10*x + 20

-- State the theorem
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (-4, -4) :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l667_66777


namespace NUMINAMATH_CALUDE_triangle_division_possible_l667_66796

/-- Represents a part of the triangle -/
structure TrianglePart where
  numbers : List Nat
  area : Nat

/-- Represents the entire triangle -/
structure Triangle where
  parts : List TrianglePart

/-- The sum of numbers in a triangle part -/
def sumPart (part : TrianglePart) : Nat :=
  part.numbers.sum

/-- The total sum of all numbers in the triangle -/
def totalSum (triangle : Triangle) : Nat :=
  triangle.parts.map sumPart |>.sum

/-- Check if all parts have equal sums -/
def equalSums (triangle : Triangle) : Prop :=
  ∀ i j, i < triangle.parts.length → j < triangle.parts.length → 
    sumPart (triangle.parts.get ⟨i, by sorry⟩) = sumPart (triangle.parts.get ⟨j, by sorry⟩)

/-- Check if all parts have different areas -/
def differentAreas (triangle : Triangle) : Prop :=
  ∀ i j, i < triangle.parts.length → j < triangle.parts.length → i ≠ j → 
    (triangle.parts.get ⟨i, by sorry⟩).area ≠ (triangle.parts.get ⟨j, by sorry⟩).area

/-- The main theorem -/
theorem triangle_division_possible : 
  ∃ (t : Triangle), totalSum t = 63 ∧ t.parts.length = 3 ∧ equalSums t ∧ differentAreas t :=
sorry

end NUMINAMATH_CALUDE_triangle_division_possible_l667_66796


namespace NUMINAMATH_CALUDE_min_value_product_l667_66738

theorem min_value_product (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x/y + y/z + z/x + y/x + z/y + x/z = 10) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) ≥ 37 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l667_66738


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l667_66781

/-- A trinomial ax^2 + bx + c is a perfect square if there exist p and q such that
    ax^2 + bx + c = (px + q)^2 for all x. -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

/-- If 4x^2 + mx + 25 is a perfect square trinomial, then m = 20. -/
theorem perfect_square_trinomial_condition (m : ℝ) :
  is_perfect_square_trinomial 4 m 25 → m = 20 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l667_66781


namespace NUMINAMATH_CALUDE_hours_to_weeks_l667_66727

/-- Proves that 2016 hours is equivalent to 12 weeks -/
theorem hours_to_weeks : 
  (∀ (week : ℕ) (day : ℕ) (hour : ℕ), 
    (1 : ℕ) * week = 7 * day ∧ 
    (1 : ℕ) * day = 24 * hour) → 
  2016 = 12 * (7 * 24) :=
by sorry

end NUMINAMATH_CALUDE_hours_to_weeks_l667_66727


namespace NUMINAMATH_CALUDE_crayons_lost_or_given_away_l667_66774

theorem crayons_lost_or_given_away (initial_crayons end_crayons : ℕ) 
  (h1 : initial_crayons = 479)
  (h2 : end_crayons = 134) :
  initial_crayons - end_crayons = 345 :=
by sorry

end NUMINAMATH_CALUDE_crayons_lost_or_given_away_l667_66774


namespace NUMINAMATH_CALUDE_smallest_valid_staircase_sum_of_digits_90_l667_66706

def is_valid_staircase (n : ℕ) : Prop :=
  ⌈(n : ℚ) / 2⌉ - ⌈(n : ℚ) / 3⌉ = 15

theorem smallest_valid_staircase :
  ∀ m : ℕ, m < 90 → ¬(is_valid_staircase m) ∧ is_valid_staircase 90 :=
by sorry

theorem sum_of_digits_90 : (9 : ℕ) = (9 : ℕ) + (0 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_staircase_sum_of_digits_90_l667_66706


namespace NUMINAMATH_CALUDE_three_true_inequalities_l667_66715

theorem three_true_inequalities
  (x y a b : ℝ)
  (hx_pos : x > 0)
  (hy_pos : y > 0)
  (ha_pos : a > 0)
  (hb_pos : b > 0)
  (hx : x^2 < a^2)
  (hy : y^2 < b^2) :
  (x^2 + y^2 < a^2 + b^2) ∧
  (x^2 * y^2 < a^2 * b^2) ∧
  (x^2 / y^2 < a^2 / b^2) ∧
  ¬(∀ x y a b, x > 0 → y > 0 → a > 0 → b > 0 → x^2 < a^2 → y^2 < b^2 → x^2 - y^2 < a^2 - b^2) :=
by sorry

end NUMINAMATH_CALUDE_three_true_inequalities_l667_66715


namespace NUMINAMATH_CALUDE_max_s_value_l667_66761

/-- Definition of the lucky number t for a given s -/
def lucky_number (s : ℕ) : ℕ :=
  let x := s / 100 - 1
  let y := (s / 10) % 10
  if y ≤ 6 then
    1000 * (x + 1) + 100 * y + 30 + y + 3
  else
    1000 * (x + 1) + 100 * y + 30 + y - 7

/-- Definition of the function F for a given lucky number N -/
def F (N : ℕ) : ℚ :=
  let ab := N / 100
  let dc := N % 100
  (ab - dc) / 3

/-- Theorem stating the maximum value of s satisfying all conditions -/
theorem max_s_value : 
  ∃ (s : ℕ), s = 913 ∧ 
  (∀ (x y : ℕ), s = 100 * x + 10 * y + 103 → 
    x ≥ y ∧ 
    y ≤ 8 ∧ 
    x ≤ 8 ∧ 
    (lucky_number s) % 17 = 5 ∧ 
    (F (lucky_number s)).den = 1) ∧
  (∀ (s' : ℕ), s' > s → 
    ¬(∃ (x' y' : ℕ), s' = 100 * x' + 10 * y' + 103 ∧ 
      x' ≥ y' ∧ 
      y' ≤ 8 ∧ 
      x' ≤ 8 ∧ 
      (lucky_number s') % 17 = 5 ∧ 
      (F (lucky_number s')).den = 1)) :=
sorry

end NUMINAMATH_CALUDE_max_s_value_l667_66761
