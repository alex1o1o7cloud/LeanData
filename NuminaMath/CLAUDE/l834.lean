import Mathlib

namespace NUMINAMATH_CALUDE_second_train_speed_l834_83490

/-- Given two trains starting from the same station, traveling in the same direction
    on parallel tracks for 8 hours, with one train moving at 11 mph and ending up
    160 miles behind the other train, prove that the speed of the second train is 31 mph. -/
theorem second_train_speed (v : ℝ) : 
  v > 0 → -- The speed of the second train is positive
  (v * 8 - 11 * 8 = 160) → -- Distance difference after 8 hours
  v = 31 :=
by sorry

end NUMINAMATH_CALUDE_second_train_speed_l834_83490


namespace NUMINAMATH_CALUDE_solve_system_l834_83472

theorem solve_system (x y b : ℚ) : 
  (4 * x + 2 * y = b) → 
  (3 * x + 7 * y = 3 * b) → 
  (y = 3) → 
  (b = 22 / 3) := by
sorry

end NUMINAMATH_CALUDE_solve_system_l834_83472


namespace NUMINAMATH_CALUDE_angle_relation_l834_83491

theorem angle_relation (α β : Real) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π)
  (h3 : Real.tan (α - β) = 1/3) (h4 : Real.tan β = 1/7) :
  2 * α - β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_relation_l834_83491


namespace NUMINAMATH_CALUDE_new_students_average_age_l834_83437

/-- Proves that the average age of new students is 32 years given the conditions of the problem -/
theorem new_students_average_age
  (original_average : ℝ)
  (new_students : ℕ)
  (new_average : ℝ)
  (original_strength : ℕ)
  (h1 : original_average = 40)
  (h2 : new_students = 12)
  (h3 : new_average = 36)
  (h4 : original_strength = 12) :
  (original_strength : ℝ) * original_average + (new_students : ℝ) * 32 =
    ((original_strength + new_students) : ℝ) * new_average :=
by sorry

end NUMINAMATH_CALUDE_new_students_average_age_l834_83437


namespace NUMINAMATH_CALUDE_quadratic_function_property_l834_83498

theorem quadratic_function_property (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : f 0 = f 4)
  (h3 : f 0 > f 1) :
  a > 0 ∧ 4 * a + b = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l834_83498


namespace NUMINAMATH_CALUDE_floor_coverage_l834_83439

/-- A type representing a rectangular floor -/
structure RectangularFloor where
  m : ℕ
  n : ℕ
  h_m : m > 3
  h_n : n > 3

/-- A predicate that determines if a floor can be fully covered by 2x4 tiles -/
def canBeCovered (floor : RectangularFloor) : Prop :=
  floor.m % 2 = 0 ∧ floor.n % 2 = 0

/-- Theorem stating that a rectangular floor can be fully covered by 2x4 tiles 
    if and only if both dimensions are even -/
theorem floor_coverage (floor : RectangularFloor) :
  canBeCovered floor ↔ (floor.m % 2 = 0 ∧ floor.n % 2 = 0) := by
  sorry

#check floor_coverage

end NUMINAMATH_CALUDE_floor_coverage_l834_83439


namespace NUMINAMATH_CALUDE_addition_problems_l834_83402

theorem addition_problems :
  (15 + (-22) = -7) ∧
  ((-13) + (-8) = -21) ∧
  ((-0.9) + 1.5 = 0.6) ∧
  (1/2 + (-2/3) = -1/6) := by
  sorry

end NUMINAMATH_CALUDE_addition_problems_l834_83402


namespace NUMINAMATH_CALUDE_num_paths_is_126_l834_83475

/-- The number of paths from A to C passing through B on a grid -/
def num_paths_through_B : ℕ :=
  let a_to_b_right := 5
  let a_to_b_down := 2
  let b_to_c_right := 2
  let b_to_c_down := 2
  let paths_a_to_b := Nat.choose (a_to_b_right + a_to_b_down) a_to_b_right
  let paths_b_to_c := Nat.choose (b_to_c_right + b_to_c_down) b_to_c_right
  paths_a_to_b * paths_b_to_c

/-- Theorem stating the number of paths from A to C passing through B is 126 -/
theorem num_paths_is_126 : num_paths_through_B = 126 := by
  sorry

end NUMINAMATH_CALUDE_num_paths_is_126_l834_83475


namespace NUMINAMATH_CALUDE_fraction_addition_l834_83416

theorem fraction_addition : (3 : ℚ) / 5 + (2 : ℚ) / 5 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_l834_83416


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l834_83440

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l834_83440


namespace NUMINAMATH_CALUDE_pinwheel_area_is_six_l834_83450

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a kite in the pinwheel -/
structure Kite where
  center : GridPoint
  vertex1 : GridPoint
  vertex2 : GridPoint
  vertex3 : GridPoint

/-- Represents a pinwheel shape -/
structure Pinwheel where
  kites : Fin 4 → Kite
  grid_size : Nat
  h_grid_size : grid_size = 5

/-- Calculates the area of a pinwheel -/
noncomputable def pinwheel_area (p : Pinwheel) : ℝ :=
  sorry

/-- Theorem stating that the area of the described pinwheel is 6 square units -/
theorem pinwheel_area_is_six (p : Pinwheel) : pinwheel_area p = 6 := by
  sorry

end NUMINAMATH_CALUDE_pinwheel_area_is_six_l834_83450


namespace NUMINAMATH_CALUDE_envelope_count_l834_83469

/-- Proves that the number of envelopes sent is 850, given the weight of one envelope and the total weight. -/
theorem envelope_count (envelope_weight : ℝ) (total_weight_kg : ℝ) : 
  envelope_weight = 8.5 →
  total_weight_kg = 7.225 →
  (total_weight_kg * 1000) / envelope_weight = 850 := by
  sorry

end NUMINAMATH_CALUDE_envelope_count_l834_83469


namespace NUMINAMATH_CALUDE_cloth_square_cutting_l834_83485

/-- Proves that a 29 cm by 40 cm cloth can be cut into at most 280 squares of 4 square centimeters each. -/
theorem cloth_square_cutting (cloth_width : ℕ) (cloth_length : ℕ) 
  (square_area : ℕ) (max_squares : ℕ) : 
  cloth_width = 29 → 
  cloth_length = 40 → 
  square_area = 4 → 
  max_squares = 280 → 
  (cloth_width / 2) * (cloth_length / 2) ≤ max_squares :=
by
  sorry

#check cloth_square_cutting

end NUMINAMATH_CALUDE_cloth_square_cutting_l834_83485


namespace NUMINAMATH_CALUDE_distance_before_meeting_l834_83448

/-- The distance between two boats one minute before they meet -/
theorem distance_before_meeting (v1 v2 d : ℝ) (hv1 : v1 = 4) (hv2 : v2 = 20) (hd : d = 20) :
  let t := d / (v1 + v2)  -- Time to meet
  let distance_per_minute := (v1 + v2) / 60
  (t - 1/60) * (v1 + v2) = 0.4
  := by sorry

end NUMINAMATH_CALUDE_distance_before_meeting_l834_83448


namespace NUMINAMATH_CALUDE_square_sum_equals_eight_l834_83407

theorem square_sum_equals_eight (a b c : ℝ) 
  (sum_condition : a + b + c = 4)
  (product_sum_condition : a * b + b * c + a * c = 4) :
  a^2 + b^2 + c^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_eight_l834_83407


namespace NUMINAMATH_CALUDE_gumball_distribution_l834_83441

/-- Represents the number of gumballs each person has -/
structure Gumballs :=
  (joanna : ℕ)
  (jacques : ℕ)
  (julia : ℕ)

/-- Calculates the total number of gumballs -/
def total_gumballs (g : Gumballs) : ℕ :=
  g.joanna + g.jacques + g.julia

/-- Represents the purchase multipliers for each person -/
structure PurchaseMultipliers :=
  (joanna : ℕ)
  (jacques : ℕ)
  (julia : ℕ)

/-- Calculates the number of gumballs after purchases -/
def after_purchase (initial : Gumballs) (multipliers : PurchaseMultipliers) : Gumballs :=
  { joanna := initial.joanna + initial.joanna * multipliers.joanna,
    jacques := initial.jacques + initial.jacques * multipliers.jacques,
    julia := initial.julia + initial.julia * multipliers.julia }

/-- Theorem statement -/
theorem gumball_distribution 
  (initial : Gumballs) 
  (multipliers : PurchaseMultipliers) :
  initial.joanna = 40 ∧ 
  initial.jacques = 60 ∧ 
  initial.julia = 80 ∧
  multipliers.joanna = 5 ∧
  multipliers.jacques = 3 ∧
  multipliers.julia = 2 →
  let final := after_purchase initial multipliers
  (final.joanna = 240 ∧ 
   final.jacques = 240 ∧ 
   final.julia = 240) ∧
  (total_gumballs final / 3 = 240) :=
by sorry

end NUMINAMATH_CALUDE_gumball_distribution_l834_83441


namespace NUMINAMATH_CALUDE_equation_solution_l834_83467

theorem equation_solution (a : ℝ) : (a + 3) ^ (a + 1) = 1 ↔ a = -1 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l834_83467


namespace NUMINAMATH_CALUDE_flight_duration_sum_l834_83470

/-- Represents a time with hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes, accounting for day change -/
def timeDiffMinutes (t1 t2 : Time) : ℕ :=
  let totalMinutes1 := t1.hours * 60 + t1.minutes
  let totalMinutes2 := t2.hours * 60 + t2.minutes
  if totalMinutes2 < totalMinutes1 then
    (24 * 60 - totalMinutes1) + totalMinutes2
  else
    totalMinutes2 - totalMinutes1

theorem flight_duration_sum (departure : Time) (arrival : Time) (h m : ℕ) :
  departure.hours = 17 ∧ departure.minutes = 30 ∧
  arrival.hours = 2 ∧ arrival.minutes = 15 ∧
  0 < m ∧ m < 60 ∧
  timeDiffMinutes departure arrival + 3 * 60 = h * 60 + m →
  h + m = 56 := by
  sorry

end NUMINAMATH_CALUDE_flight_duration_sum_l834_83470


namespace NUMINAMATH_CALUDE_expected_sticky_corn_l834_83427

theorem expected_sticky_corn (total_corn : ℕ) (sticky_corn : ℕ) (sample_size : ℕ) :
  total_corn = 140 →
  sticky_corn = 56 →
  sample_size = 40 →
  (sample_size * sticky_corn) / total_corn = 16 := by
  sorry

end NUMINAMATH_CALUDE_expected_sticky_corn_l834_83427


namespace NUMINAMATH_CALUDE_function_properties_l834_83494

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (2 * ω * x - Real.pi / 6) + 2 * (Real.cos (ω * x))^2 - 1

theorem function_properties (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x, f ω (x + Real.pi / ω) = f ω x) : 
  ω = 1 ∧ 
  (∀ x ∈ Set.Icc 0 (7 * Real.pi / 12), f ω x ≤ 1) ∧
  (∀ x ∈ Set.Icc 0 (7 * Real.pi / 12), f ω x ≥ -Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc 0 (7 * Real.pi / 12), f ω x = 1) ∧
  (∃ x ∈ Set.Icc 0 (7 * Real.pi / 12), f ω x = -Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l834_83494


namespace NUMINAMATH_CALUDE_intersection_tangent_line_l834_83496

theorem intersection_tangent_line (x₀ : ℝ) (hx₀ : x₀ ≠ 0) (h : Real.tan x₀ = -x₀) :
  (x₀^2 + 1) * (1 + Real.cos (2 * x₀)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_tangent_line_l834_83496


namespace NUMINAMATH_CALUDE_min_value_of_f_l834_83488

/-- The function f(x) = -(x-1)³ + 12x + a - 1 -/
def f (x a : ℝ) : ℝ := -(x-1)^3 + 12*x + a - 1

/-- The interval [a, b] -/
def closed_interval (a b : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ b}

theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ closed_interval (-2) 2, ∀ y ∈ closed_interval (-2) 2, f y a ≤ f x a) ∧
  (∃ x ∈ closed_interval (-2) 2, f x a = 20) →
  (∃ x ∈ closed_interval (-2) 2, f x a = -7 ∧ ∀ y ∈ closed_interval (-2) 2, -7 ≤ f y a) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l834_83488


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l834_83428

/-- Given a circle D with equation x^2 + 10x + 2y^2 - 8y = 18,
    prove that the sum of its center coordinates and radius is -3 + √38 -/
theorem circle_center_radius_sum :
  ∃ (a b r : ℝ), 
    (∀ (x y : ℝ), x^2 + 10*x + 2*y^2 - 8*y = 18 ↔ (x - a)^2 + (y - b)^2 = r^2) ∧
    a + b + r = -3 + Real.sqrt 38 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l834_83428


namespace NUMINAMATH_CALUDE_tournament_permutation_exists_l834_83443

/-- Represents the result of a match between two players -/
inductive MatchResult
  | Win
  | Loss

/-- Represents a tournament with n players -/
structure Tournament (n : Nat) where
  /-- The result of the match between player i and player j -/
  result : Fin n → Fin n → MatchResult

/-- A permutation of players -/
def PlayerPermutation (n : Nat) := Fin n → Fin n

/-- Checks if a player satisfies the condition with their neighbors -/
def satisfiesCondition (t : Tournament 1000) (p : PlayerPermutation 1000) (i : Fin 998) : Prop :=
  (t.result (p i) (p (i + 1)) = MatchResult.Win ∧ t.result (p i) (p (i + 2)) = MatchResult.Win) ∨
  (t.result (p i) (p (i + 1)) = MatchResult.Loss ∧ t.result (p i) (p (i + 2)) = MatchResult.Loss)

/-- The main theorem -/
theorem tournament_permutation_exists (t : Tournament 1000) :
  ∃ (p : PlayerPermutation 1000), ∀ (i : Fin 998), satisfiesCondition t p i := by
  sorry

end NUMINAMATH_CALUDE_tournament_permutation_exists_l834_83443


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l834_83451

theorem quadratic_inequality_equivalence (x : ℝ) :
  x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l834_83451


namespace NUMINAMATH_CALUDE_line_curve_hyperbola_l834_83486

variable (a b : ℝ)

theorem line_curve_hyperbola (h1 : a ≠ 0) (h2 : b ≠ 0) :
  ∃ (x y : ℝ), (a * x - y + b = 0) ∧ (b * x^2 + a * y^2 = a * b) →
  ∃ (A B : ℝ), A > 0 ∧ B > 0 ∧ ∀ (x y : ℝ), x^2 / A - y^2 / B = 1 :=
sorry

end NUMINAMATH_CALUDE_line_curve_hyperbola_l834_83486


namespace NUMINAMATH_CALUDE_square_and_cube_roots_l834_83447

-- Define square root
def is_square_root (x y : ℝ) : Prop := y^2 = x

-- Define cube root
def is_cube_root (x y : ℝ) : Prop := y^3 = x

-- Define self square root
def is_self_square_root (x : ℝ) : Prop := x^2 = x

theorem square_and_cube_roots :
  (∃ y : ℝ, y < 0 ∧ is_square_root 2 y) ∧
  (is_cube_root (-1) (-1)) ∧
  (is_square_root 100 10) ∧
  (∀ x : ℝ, is_self_square_root x ↔ (x = 0 ∨ x = 1)) :=
by sorry

end NUMINAMATH_CALUDE_square_and_cube_roots_l834_83447


namespace NUMINAMATH_CALUDE_translation_theorem_l834_83409

/-- Represents a point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def apply_translation (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_theorem :
  let A : Point := { x := -1, y := 0 }
  let B : Point := { x := 1, y := 2 }
  let A1 : Point := { x := 2, y := -1 }
  let t : Translation := { dx := A1.x - A.x, dy := A1.y - A.y }
  let B1 : Point := apply_translation B t
  B1 = { x := 4, y := 1 } := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l834_83409


namespace NUMINAMATH_CALUDE_salt_solution_concentration_l834_83426

/-- Proves that adding a specific amount of pure salt to a given salt solution results in the desired concentration -/
theorem salt_solution_concentration 
  (initial_weight : Real) 
  (initial_concentration : Real) 
  (added_salt : Real) 
  (final_concentration : Real) : 
  initial_weight = 100 ∧ 
  initial_concentration = 0.1 ∧ 
  added_salt = 28.571428571428573 ∧ 
  final_concentration = 0.3 →
  (initial_concentration * initial_weight + added_salt) / (initial_weight + added_salt) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_salt_solution_concentration_l834_83426


namespace NUMINAMATH_CALUDE_x_fourth_plus_y_fourth_l834_83432

theorem x_fourth_plus_y_fourth (x y : ℕ+) (h : y * x^2 + x * y^2 = 70) : x^4 + y^4 = 641 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_y_fourth_l834_83432


namespace NUMINAMATH_CALUDE_cube_squared_equals_sixth_power_l834_83449

theorem cube_squared_equals_sixth_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_cube_squared_equals_sixth_power_l834_83449


namespace NUMINAMATH_CALUDE_deposit_ratio_l834_83435

def mark_deposit : ℚ := 88
def total_deposit : ℚ := 400

def bryan_deposit : ℚ := total_deposit - mark_deposit

theorem deposit_ratio :
  ∃ (n : ℚ), n > 1 ∧ bryan_deposit < n * mark_deposit →
  (bryan_deposit / mark_deposit) = 39 / 11 := by
sorry

end NUMINAMATH_CALUDE_deposit_ratio_l834_83435


namespace NUMINAMATH_CALUDE_even_number_in_rows_l834_83425

/-- Definition of the triangle table -/
def triangle_table : ℕ → ℤ → ℕ
| 1, 0 => 1
| n, k => if n > 1 ∧ abs k < n then
            triangle_table (n-1) (k-1) + triangle_table (n-1) k + triangle_table (n-1) (k+1)
          else 0

/-- Theorem: From the third row onward, each row contains at least one even number -/
theorem even_number_in_rows (n : ℕ) (h : n ≥ 3) :
  ∃ k : ℤ, Even (triangle_table n k) := by sorry

end NUMINAMATH_CALUDE_even_number_in_rows_l834_83425


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l834_83480

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) - a n = d

-- Define a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b / a = r ∧ c / b = r

-- Theorem statement
theorem arithmetic_geometric_ratio 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : geometric_sequence (a 2) (a 3) (a 6)) : 
  ∃ r : ℝ, r = 5 / 3 ∧ (a 3) / (a 2) = r ∧ (a 6) / (a 3) = r :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l834_83480


namespace NUMINAMATH_CALUDE_same_last_six_digits_l834_83433

/-- Given a positive integer N where N and N^2 both end in the same sequence
    of six digits abcdef in base 10 (with a ≠ 0), prove that the five-digit
    number abcde is equal to 48437. -/
theorem same_last_six_digits (N : ℕ) : 
  (N > 0) →
  (∃ (a b c d e f : ℕ), 
    a ≠ 0 ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧
    N % 1000000 = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f ∧
    (N^2) % 1000000 = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f) →
  (∃ (a b c d e : ℕ),
    a ≠ 0 ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
    a * 10000 + b * 1000 + c * 100 + d * 10 + e = 48437) :=
by sorry

end NUMINAMATH_CALUDE_same_last_six_digits_l834_83433


namespace NUMINAMATH_CALUDE_cube_surface_area_l834_83446

/-- The surface area of a cube with edge length 2a cm is 24a² cm² -/
theorem cube_surface_area (a : ℝ) (h : a > 0) : 
  6 * (2 * a) ^ 2 = 24 * a ^ 2 := by
  sorry

#check cube_surface_area

end NUMINAMATH_CALUDE_cube_surface_area_l834_83446


namespace NUMINAMATH_CALUDE_frank_reading_speed_l834_83410

/-- The number of days Frank took to finish all books -/
def total_days : ℕ := 492

/-- The total number of books Frank read -/
def total_books : ℕ := 41

/-- The number of days it took Frank to finish each book -/
def days_per_book : ℚ := total_days / total_books

/-- Theorem stating that Frank took 12 days to finish each book -/
theorem frank_reading_speed : days_per_book = 12 := by
  sorry

end NUMINAMATH_CALUDE_frank_reading_speed_l834_83410


namespace NUMINAMATH_CALUDE_biology_marks_proof_l834_83430

def english_marks : ℕ := 72
def math_marks : ℕ := 45
def physics_marks : ℕ := 72
def chemistry_marks : ℕ := 77
def average_marks : ℚ := 68.2
def total_subjects : ℕ := 5

theorem biology_marks_proof :
  ∃ (biology_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / total_subjects = average_marks ∧
    biology_marks = 75 := by
  sorry

end NUMINAMATH_CALUDE_biology_marks_proof_l834_83430


namespace NUMINAMATH_CALUDE_dormitory_arrangements_l834_83400

def num_students : ℕ := 7
def min_per_dorm : ℕ := 2

-- Function to calculate the number of arrangements
def calculate_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  sorry

theorem dormitory_arrangements :
  calculate_arrangements num_students min_per_dorm 2 = 60 :=
sorry

end NUMINAMATH_CALUDE_dormitory_arrangements_l834_83400


namespace NUMINAMATH_CALUDE_train_arrival_interval_l834_83459

def train_interval (passengers_per_hour : ℕ) (passengers_left : ℕ) (passengers_taken : ℕ) : ℕ :=
  60 / (passengers_per_hour / (passengers_left + passengers_taken))

theorem train_arrival_interval :
  train_interval 6240 200 320 = 5 := by
  sorry

end NUMINAMATH_CALUDE_train_arrival_interval_l834_83459


namespace NUMINAMATH_CALUDE_coordinate_sum_theorem_l834_83481

/-- Given a function f where f(3) = 4, the sum of the coordinates of the point (x, y) 
    satisfying 4y = 2f(2x) + 7 is equal to 5.25. -/
theorem coordinate_sum_theorem (f : ℝ → ℝ) (hf : f 3 = 4) :
  ∃ (x y : ℝ), 4 * y = 2 * f (2 * x) + 7 ∧ x + y = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_theorem_l834_83481


namespace NUMINAMATH_CALUDE_ribbons_shipment_count_l834_83462

/-- The number of ribbons that arrived in the shipment before lunch -/
def ribbons_in_shipment (initial : ℕ) (morning : ℕ) (afternoon : ℕ) (final : ℕ) : ℕ :=
  (afternoon + final) - (initial - morning)

/-- Theorem stating that the number of ribbons in the shipment is 4 -/
theorem ribbons_shipment_count :
  ribbons_in_shipment 38 14 16 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ribbons_shipment_count_l834_83462


namespace NUMINAMATH_CALUDE_max_queens_101_88_l834_83434

/-- Represents a chessboard with a red corner -/
structure RedCornerBoard :=
  (size : Nat)
  (red_size : Nat)
  (h_size : size > red_size)

/-- Represents the maximum number of non-attacking queens on a RedCornerBoard -/
def max_queens (board : RedCornerBoard) : Nat :=
  2 * (board.size - board.red_size)

/-- Theorem stating the maximum number of non-attacking queens on a 101x101 board with 88x88 red corner -/
theorem max_queens_101_88 :
  let board : RedCornerBoard := ⟨101, 88, by norm_num⟩
  max_queens board = 26 := by
  sorry

#eval max_queens ⟨101, 88, by norm_num⟩

end NUMINAMATH_CALUDE_max_queens_101_88_l834_83434


namespace NUMINAMATH_CALUDE_least_months_to_double_amount_l834_83453

/-- The amount owed after t months -/
def amount_owed (initial_amount : ℝ) (interest_rate : ℝ) (t : ℕ) : ℝ :=
  initial_amount * (1 + interest_rate) ^ t

/-- The theorem stating that 25 is the least number of months to double the borrowed amount -/
theorem least_months_to_double_amount : 
  let initial_amount : ℝ := 1500
  let interest_rate : ℝ := 0.03
  let double_amount : ℝ := 2 * initial_amount
  ∀ t : ℕ, t < 25 → amount_owed initial_amount interest_rate t ≤ double_amount ∧
  amount_owed initial_amount interest_rate 25 > double_amount :=
by sorry

end NUMINAMATH_CALUDE_least_months_to_double_amount_l834_83453


namespace NUMINAMATH_CALUDE_fraction_equivalence_l834_83473

theorem fraction_equivalence : ∃ n : ℤ, (2 + n) / (7 + n) = 3 / 4 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l834_83473


namespace NUMINAMATH_CALUDE_vitamin_a_weekly_pills_l834_83438

/-- Calculates the number of pills needed for a week's supply of Vitamin A -/
def weekly_vitamin_pills (daily_recommended : ℕ) (mg_per_pill : ℕ) : ℕ :=
  (daily_recommended / mg_per_pill) * 7

/-- Theorem stating that 28 pills are needed for a week's supply of Vitamin A -/
theorem vitamin_a_weekly_pills :
  weekly_vitamin_pills 200 50 = 28 := by
  sorry

#eval weekly_vitamin_pills 200 50

end NUMINAMATH_CALUDE_vitamin_a_weekly_pills_l834_83438


namespace NUMINAMATH_CALUDE_triangle_tangent_ratio_l834_83460

theorem triangle_tangent_ratio (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- Acute triangle condition
  A + B + C = π →  -- Triangle angle sum
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a / (2 * Real.sin A) = b / (2 * Real.sin B) →  -- Law of sines
  a / (2 * Real.sin A) = c / (2 * Real.sin C) →  -- Law of sines
  a / b + b / a = 6 * Real.cos C →  -- Given condition
  Real.tan C / Real.tan A + Real.tan C / Real.tan B = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_tangent_ratio_l834_83460


namespace NUMINAMATH_CALUDE_light_reflection_l834_83417

/-- Given a ray of light reflecting off a line, this theorem proves the equations of the incident and reflected rays. -/
theorem light_reflection (A B : ℝ × ℝ) (reflecting_line : ℝ → ℝ → Prop) :
  A = (2, 3) →
  B = (1, 1) →
  (∀ x y, reflecting_line x y ↔ x + y + 1 = 0) →
  ∃ (incident_ray reflected_ray : ℝ → ℝ → Prop),
    (∀ x y, incident_ray x y ↔ 5*x - 4*y + 2 = 0) ∧
    (∀ x y, reflected_ray x y ↔ 4*x - 5*y + 1 = 0) ∧
    (∃ C : ℝ × ℝ, incident_ray C.1 C.2 ∧ reflecting_line C.1 C.2 ∧ reflected_ray C.1 C.2) :=
by sorry

end NUMINAMATH_CALUDE_light_reflection_l834_83417


namespace NUMINAMATH_CALUDE_special_sequence_250th_term_l834_83406

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- Predicate to check if a number is a multiple of 3 -/
def is_multiple_of_three (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 3 * m

/-- The sequence of positive integers omitting perfect squares and multiples of 3 -/
def special_sequence : ℕ → ℕ :=
  sorry

/-- The 250th term of the special sequence is 350 -/
theorem special_sequence_250th_term :
  special_sequence 250 = 350 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_250th_term_l834_83406


namespace NUMINAMATH_CALUDE_point_on_line_trig_identity_l834_83408

/-- 
Given a point P with coordinates (cos θ, sin θ) that lies on the line 2x + y = 0,
prove that cos 2θ + (1/2) sin 2θ = -1.
-/
theorem point_on_line_trig_identity (θ : Real) 
  (h : 2 * Real.cos θ + Real.sin θ = 0) : 
  Real.cos (2 * θ) + (1/2) * Real.sin (2 * θ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_trig_identity_l834_83408


namespace NUMINAMATH_CALUDE_garrick_nickels_count_l834_83471

/-- The number of cents in a dime -/
def dime_value : ℕ := 10

/-- The number of cents in a quarter -/
def quarter_value : ℕ := 25

/-- The number of cents in a nickel -/
def nickel_value : ℕ := 5

/-- The number of cents in a penny -/
def penny_value : ℕ := 1

/-- The number of dimes Cindy tossed -/
def cindy_dimes : ℕ := 5

/-- The number of quarters Eric flipped -/
def eric_quarters : ℕ := 3

/-- The number of pennies Ivy dropped -/
def ivy_pennies : ℕ := 60

/-- The total amount of money in the pond in cents -/
def total_cents : ℕ := 200

/-- The number of nickels Garrick threw into the pond -/
def garrick_nickels : ℕ := (total_cents - (cindy_dimes * dime_value + eric_quarters * quarter_value + ivy_pennies * penny_value)) / nickel_value

theorem garrick_nickels_count : garrick_nickels = 3 := by
  sorry

end NUMINAMATH_CALUDE_garrick_nickels_count_l834_83471


namespace NUMINAMATH_CALUDE_factor_expression_l834_83404

theorem factor_expression (b : ℝ) : 45 * b^2 + 135 * b^3 = 45 * b^2 * (1 + 3 * b) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l834_83404


namespace NUMINAMATH_CALUDE_initial_lives_proof_l834_83444

/-- Represents the number of lives Kaleb had initially -/
def initial_lives : ℕ := 98

/-- Represents the number of lives Kaleb lost -/
def lives_lost : ℕ := 25

/-- Represents the number of lives Kaleb had remaining -/
def remaining_lives : ℕ := 73

/-- Theorem stating that the initial number of lives equals the sum of remaining lives and lives lost -/
theorem initial_lives_proof : initial_lives = remaining_lives + lives_lost := by
  sorry

end NUMINAMATH_CALUDE_initial_lives_proof_l834_83444


namespace NUMINAMATH_CALUDE_closer_to_d_probability_l834_83442

/-- Triangle DEF with side lengths -/
structure Triangle (DE EF FD : ℝ) where
  side_positive : 0 < DE ∧ 0 < EF ∧ 0 < FD
  triangle_inequality : DE + EF > FD ∧ EF + FD > DE ∧ FD + DE > EF

/-- The region closer to D than to E or F -/
def CloserToD (t : Triangle DE EF FD) : Set (ℝ × ℝ) := sorry

theorem closer_to_d_probability (t : Triangle 8 6 10) : 
  MeasureTheory.volume (CloserToD t) = (1/4) * MeasureTheory.volume (Set.univ : Set (ℝ × ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_closer_to_d_probability_l834_83442


namespace NUMINAMATH_CALUDE_stair_climbing_time_l834_83477

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The time taken to climb stairs -/
theorem stair_climbing_time : arithmetic_sum 15 8 7 = 273 := by
  sorry

end NUMINAMATH_CALUDE_stair_climbing_time_l834_83477


namespace NUMINAMATH_CALUDE_probability_of_specific_arrangement_l834_83484

theorem probability_of_specific_arrangement (n : ℕ) (r : ℕ) : 
  n = 4 → r = 2 → (1 : ℚ) / (n! / r!) = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_arrangement_l834_83484


namespace NUMINAMATH_CALUDE_negative_double_greater_than_negative_abs_l834_83463

theorem negative_double_greater_than_negative_abs :
  -(-(1/9 : ℚ)) > -|(-(1/9 : ℚ))| := by sorry

end NUMINAMATH_CALUDE_negative_double_greater_than_negative_abs_l834_83463


namespace NUMINAMATH_CALUDE_permutation_and_exponent_inequalities_l834_83454

theorem permutation_and_exponent_inequalities 
  (i m n : ℕ) 
  (h1 : 1 < i) 
  (h2 : i ≤ m) 
  (h3 : m < n) : 
  n * (m.factorial / (m - i).factorial) < m * (n.factorial / (n - i).factorial) ∧ 
  (1 + m : ℝ) ^ n > (1 + n : ℝ) ^ m := by
  sorry

end NUMINAMATH_CALUDE_permutation_and_exponent_inequalities_l834_83454


namespace NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_l834_83493

/-- The maximum area of an equilateral triangle inscribed in a 12 by 5 rectangle -/
theorem max_area_equilateral_triangle_in_rectangle :
  ∃ (A : ℝ),
    A = (15 : ℝ) * Real.sqrt 3 - 10 ∧
    (∀ (s : ℝ),
      s > 0 →
      s ≤ 5 →
      s * Real.sqrt 3 ≤ 12 →
      (Real.sqrt 3 / 4) * s^2 ≤ A) :=
by sorry

end NUMINAMATH_CALUDE_max_area_equilateral_triangle_in_rectangle_l834_83493


namespace NUMINAMATH_CALUDE_heat_of_neutralization_instruments_l834_83479

-- Define the set of available instruments
inductive Instrument
  | Balance
  | MeasuringCylinder
  | Beaker
  | Burette
  | Thermometer
  | TestTube
  | AlcoholLamp

-- Define the requirements for the heat of neutralization experiment
structure ExperimentRequirements where
  needsWeighing : Bool
  needsHeating : Bool
  reactionContainer : Instrument
  volumeMeasurementTool : Instrument
  temperatureMeasurementTool : Instrument

-- Define the correct set of instruments
def correctInstruments : Set Instrument :=
  {Instrument.MeasuringCylinder, Instrument.Beaker, Instrument.Thermometer}

-- Define the heat of neutralization experiment requirements
def heatOfNeutralizationRequirements : ExperimentRequirements :=
  { needsWeighing := false
  , needsHeating := false
  , reactionContainer := Instrument.Beaker
  , volumeMeasurementTool := Instrument.MeasuringCylinder
  , temperatureMeasurementTool := Instrument.Thermometer
  }

-- Theorem statement
theorem heat_of_neutralization_instruments :
  correctInstruments = 
    { i : Instrument | i = heatOfNeutralizationRequirements.volumeMeasurementTool ∨
                       i = heatOfNeutralizationRequirements.reactionContainer ∨
                       i = heatOfNeutralizationRequirements.temperatureMeasurementTool } :=
by sorry

end NUMINAMATH_CALUDE_heat_of_neutralization_instruments_l834_83479


namespace NUMINAMATH_CALUDE_forum_questions_per_hour_l834_83452

/-- Proves that given the conditions of the forum, the average number of questions posted by each user per hour is 3 -/
theorem forum_questions_per_hour (members : ℕ) (total_posts_per_day : ℕ) 
  (h1 : members = 200)
  (h2 : total_posts_per_day = 57600) : 
  (total_posts_per_day / (24 * members)) / 4 = 3 := by
  sorry

#check forum_questions_per_hour

end NUMINAMATH_CALUDE_forum_questions_per_hour_l834_83452


namespace NUMINAMATH_CALUDE_min_distinct_lines_for_31_links_l834_83415

/-- A polygonal chain in a plane -/
structure PolygonalChain where
  links : ℕ
  non_self_intersecting : Bool
  adjacent_links_not_collinear : Bool

/-- The minimum number of distinct lines needed to contain all links of a polygonal chain -/
def min_distinct_lines (chain : PolygonalChain) : ℕ := sorry

/-- Theorem: For a non-self-intersecting polygonal chain with 31 links where adjacent links are not collinear, 
    the minimum number of distinct lines that can contain all links is 9 -/
theorem min_distinct_lines_for_31_links : 
  ∀ (chain : PolygonalChain), 
    chain.links = 31 ∧ 
    chain.non_self_intersecting = true ∧ 
    chain.adjacent_links_not_collinear = true → 
    min_distinct_lines chain = 9 := by sorry

end NUMINAMATH_CALUDE_min_distinct_lines_for_31_links_l834_83415


namespace NUMINAMATH_CALUDE_austin_friday_hours_l834_83424

/-- Represents the problem of Austin saving for a bicycle --/
def bicycle_savings (hourly_rate : ℚ) (monday_hours : ℚ) (wednesday_hours : ℚ) (total_weeks : ℕ) (bicycle_cost : ℚ) : Prop :=
  let monday_earnings := hourly_rate * monday_hours
  let wednesday_earnings := hourly_rate * wednesday_hours
  let weekly_earnings := monday_earnings + wednesday_earnings
  let total_earnings_without_friday := weekly_earnings * total_weeks
  let remaining_earnings_needed := bicycle_cost - total_earnings_without_friday
  let friday_hours := remaining_earnings_needed / (hourly_rate * total_weeks)
  friday_hours = 3

/-- Theorem stating that Austin needs to work 3 hours on Fridays --/
theorem austin_friday_hours : 
  bicycle_savings 5 2 1 6 180 := by sorry

end NUMINAMATH_CALUDE_austin_friday_hours_l834_83424


namespace NUMINAMATH_CALUDE_expression_value_polynomial_simplification_l834_83497

-- Part 1
theorem expression_value : (1/2)^(-2) - 0.01^(-1) + (-1 - 1/7)^0 = -95 := by sorry

-- Part 2
theorem polynomial_simplification (x : ℝ) : (x-2)*(x+1) - (x-1)^2 = x - 3 := by sorry

end NUMINAMATH_CALUDE_expression_value_polynomial_simplification_l834_83497


namespace NUMINAMATH_CALUDE_expand_expression_l834_83466

theorem expand_expression (x y : ℝ) : (2*x - 3*y + 1) * (2*x + 3*y - 1) = 4*x^2 - 9*y^2 + 6*y - 1 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l834_83466


namespace NUMINAMATH_CALUDE_fraction_sum_equals_two_l834_83445

theorem fraction_sum_equals_two (a b : ℝ) (h : a ≠ b) : 
  (2 * a) / (a - b) + (2 * b) / (b - a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_two_l834_83445


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_both_zero_l834_83468

theorem sum_of_squares_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_both_zero_l834_83468


namespace NUMINAMATH_CALUDE_ratio_x_y_is_two_l834_83423

theorem ratio_x_y_is_two (x y a : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (eq1 : x^3 + Real.log x + 2*a^2 = 0) 
  (eq2 : 4*y^3 + Real.log (Real.sqrt y) + Real.log (Real.sqrt 2) + a^2 = 0) : 
  x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_x_y_is_two_l834_83423


namespace NUMINAMATH_CALUDE_objective_function_range_l834_83489

-- Define the constraint set
def ConstraintSet (x y : ℝ) : Prop :=
  x + 2*y ≥ 2 ∧ 2*x + y ≤ 4 ∧ 4*x - y ≥ -1

-- Define the objective function
def ObjectiveFunction (x y : ℝ) : ℝ := 3*x - y

-- State the theorem
theorem objective_function_range :
  ∀ x y : ℝ, ConstraintSet x y →
  ∃ z_min z_max : ℝ, z_min = -3/2 ∧ z_max = 6 ∧
  z_min ≤ ObjectiveFunction x y ∧ ObjectiveFunction x y ≤ z_max :=
sorry

end NUMINAMATH_CALUDE_objective_function_range_l834_83489


namespace NUMINAMATH_CALUDE_longest_segment_in_quarter_circle_l834_83401

theorem longest_segment_in_quarter_circle (r : ℝ) (h : r = 9) :
  let sector_chord_length_squared := 2 * r^2
  sector_chord_length_squared = 162 := by sorry

end NUMINAMATH_CALUDE_longest_segment_in_quarter_circle_l834_83401


namespace NUMINAMATH_CALUDE_megan_initial_albums_l834_83411

/-- The number of albums Megan initially put in her shopping cart -/
def initial_albums : ℕ := sorry

/-- The number of albums Megan removed from her cart -/
def removed_albums : ℕ := 2

/-- The number of songs in each album -/
def songs_per_album : ℕ := 7

/-- The total number of songs Megan bought -/
def total_songs : ℕ := 42

/-- Theorem stating that Megan initially put 8 albums in her shopping cart -/
theorem megan_initial_albums :
  initial_albums = 8 :=
by sorry

end NUMINAMATH_CALUDE_megan_initial_albums_l834_83411


namespace NUMINAMATH_CALUDE_sum_mod_seven_l834_83476

theorem sum_mod_seven : (1001 + 1002 + 1003 + 1004 + 1005) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_seven_l834_83476


namespace NUMINAMATH_CALUDE_unique_number_with_sum_of_largest_divisors_3333_l834_83482

/-- The largest divisor of a natural number is the number itself -/
def largest_divisor (n : ℕ) : ℕ := n

/-- The second largest divisor of an even natural number is half of the number -/
def second_largest_divisor (n : ℕ) : ℕ := n / 2

/-- The property that the sum of the two largest divisors of n is 3333 -/
def sum_of_largest_divisors_is_3333 (n : ℕ) : Prop :=
  largest_divisor n + second_largest_divisor n = 3333

theorem unique_number_with_sum_of_largest_divisors_3333 :
  ∀ n : ℕ, sum_of_largest_divisors_is_3333 n → n = 2222 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_sum_of_largest_divisors_3333_l834_83482


namespace NUMINAMATH_CALUDE_work_completion_time_l834_83436

theorem work_completion_time (original_men : ℕ) (added_men : ℕ) (time_reduction : ℕ) : 
  original_men = 40 →
  added_men = 8 →
  time_reduction = 10 →
  ∃ (original_time : ℕ), 
    original_time * original_men = (original_time - time_reduction) * (original_men + added_men) ∧
    original_time = 60 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l834_83436


namespace NUMINAMATH_CALUDE_tom_candy_l834_83420

def candy_problem (initial : ℕ) (from_friend : ℕ) (bought : ℕ) : Prop :=
  initial + from_friend + bought = 19

theorem tom_candy : candy_problem 2 7 10 := by sorry

end NUMINAMATH_CALUDE_tom_candy_l834_83420


namespace NUMINAMATH_CALUDE_h1n1_diameter_scientific_notation_l834_83421

theorem h1n1_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000081 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ n = -8 :=
sorry

end NUMINAMATH_CALUDE_h1n1_diameter_scientific_notation_l834_83421


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l834_83465

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem unique_two_digit_number :
  ∃! n : ℕ, is_two_digit n ∧
    tens_digit n + 2 = ones_digit n ∧
    3 * (tens_digit n * ones_digit n) = n ∧
    n = 24 := by sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l834_83465


namespace NUMINAMATH_CALUDE_boys_without_notebooks_l834_83464

theorem boys_without_notebooks
  (total_boys : ℕ)
  (total_with_notebooks : ℕ)
  (girls_with_notebooks : ℕ)
  (h1 : total_boys = 24)
  (h2 : total_with_notebooks = 30)
  (h3 : girls_with_notebooks = 18) :
  total_boys - (total_with_notebooks - girls_with_notebooks) = 12 :=
by sorry

end NUMINAMATH_CALUDE_boys_without_notebooks_l834_83464


namespace NUMINAMATH_CALUDE_product_of_two_primes_not_prime_l834_83478

/-- A number is prime if it's greater than 1 and its only positive divisors are 1 and itself -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem product_of_two_primes_not_prime (a b : ℤ) :
  isPrime (Int.natAbs (a * b)) → ¬(isPrime (Int.natAbs a) ∧ isPrime (Int.natAbs b)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_two_primes_not_prime_l834_83478


namespace NUMINAMATH_CALUDE_two_points_determine_line_l834_83455

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem: Two distinct points determine a unique line
theorem two_points_determine_line (p1 p2 : Point2D) (h : p1 ≠ p2) :
  ∃! l : Line2D, pointOnLine p1 l ∧ pointOnLine p2 l :=
sorry

end NUMINAMATH_CALUDE_two_points_determine_line_l834_83455


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l834_83413

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_third_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 1 + a 5 = 16) :
  a 3 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l834_83413


namespace NUMINAMATH_CALUDE_largest_common_value_l834_83414

theorem largest_common_value (n m : ℕ) : 
  (∃ n m : ℕ, 479 = 2 + 3 * n ∧ 479 = 3 + 7 * m) ∧ 
  (∀ k : ℕ, k < 500 → k > 479 → ¬(∃ p q : ℕ, k = 2 + 3 * p ∧ k = 3 + 7 * q)) := by
sorry

end NUMINAMATH_CALUDE_largest_common_value_l834_83414


namespace NUMINAMATH_CALUDE_fourth_power_of_nested_sqrt_l834_83495

theorem fourth_power_of_nested_sqrt (y : ℝ) :
  y = Real.sqrt (3 + Real.sqrt (3 + Real.sqrt 5)) →
  y^4 = 12 + 6 * Real.sqrt (3 + Real.sqrt 5) + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_of_nested_sqrt_l834_83495


namespace NUMINAMATH_CALUDE_rectangle_area_preservation_l834_83461

theorem rectangle_area_preservation (L W : ℝ) (h : L > 0 ∧ W > 0) :
  ∃ x : ℝ, x > 0 ∧ x < 100 ∧
  (L * (1 - x / 100)) * (W * 1.25) = L * W ∧
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_preservation_l834_83461


namespace NUMINAMATH_CALUDE_train_crossing_time_l834_83499

/-- Given a train and platform with specific dimensions and time to pass,
    prove the time it takes for the train to cross a point (tree) -/
theorem train_crossing_time
  (train_length : ℝ)
  (platform_length : ℝ)
  (time_to_pass_platform : ℝ)
  (h1 : train_length = 1200)
  (h2 : platform_length = 700)
  (h3 : time_to_pass_platform = 190)
  : (train_length / ((train_length + platform_length) / time_to_pass_platform)) = 120 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l834_83499


namespace NUMINAMATH_CALUDE_orthogonal_projection_area_l834_83431

/-- A plane polygon -/
structure PlanePolygon where
  area : ℝ

/-- An orthogonal projection of a plane polygon onto another plane -/
structure OrthogonalProjection (P : PlanePolygon) where
  area : ℝ
  angle : ℝ  -- Angle between the original plane and the projection plane

/-- 
Theorem: The area of the orthogonal projection of a plane polygon 
onto a plane is equal to the area of the polygon being projected, 
multiplied by the cosine of the angle between the projection plane 
and the plane of the polygon.
-/
theorem orthogonal_projection_area 
  (P : PlanePolygon) (proj : OrthogonalProjection P) : 
  proj.area = P.area * Real.cos proj.angle := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_projection_area_l834_83431


namespace NUMINAMATH_CALUDE_jelly_ratio_l834_83422

def jelly_problem (grape strawberry raspberry plum : ℕ) : Prop :=
  grape = 2 * strawberry ∧
  raspberry = 2 * plum ∧
  plum = 6 ∧
  strawberry = 18 ∧
  raspberry * 3 = grape

theorem jelly_ratio :
  ∀ grape strawberry raspberry plum : ℕ,
  jelly_problem grape strawberry raspberry plum →
  raspberry * 3 = grape :=
by
  sorry

end NUMINAMATH_CALUDE_jelly_ratio_l834_83422


namespace NUMINAMATH_CALUDE_f_g_2_equals_22_l834_83483

-- Define the functions g and f
def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3*x - 2

-- State the theorem
theorem f_g_2_equals_22 : f (g 2) = 22 := by
  sorry

end NUMINAMATH_CALUDE_f_g_2_equals_22_l834_83483


namespace NUMINAMATH_CALUDE_unique_integer_triples_l834_83458

theorem unique_integer_triples : 
  {(a, b, c) : ℕ × ℕ × ℕ | 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a ≤ b ∧ b ≤ c ∧
    a + b + c + a*b + b*c + c*a = a*b*c + 1} 
  = {(2, 5, 8), (3, 4, 13)} := by sorry

end NUMINAMATH_CALUDE_unique_integer_triples_l834_83458


namespace NUMINAMATH_CALUDE_square_triangle_apothem_ratio_l834_83456

theorem square_triangle_apothem_ratio :
  ∀ (s t : ℝ),
  s > 0 → t > 0 →
  s * Real.sqrt 2 = 9 * t →  -- diagonal of square = 3 * perimeter of triangle
  s * s = 2 * s →           -- apothem of square = area of square
  (s / 2) / ((Real.sqrt 3 / 2 * t) / 3) = 9 * Real.sqrt 6 / 4 :=
by sorry

end NUMINAMATH_CALUDE_square_triangle_apothem_ratio_l834_83456


namespace NUMINAMATH_CALUDE_phi_tau_ge_n_l834_83412

/-- The number of divisors of a positive integer n -/
def tau (n : ℕ+) : ℕ := sorry

/-- Euler's totient function for a positive integer n -/
def phi (n : ℕ+) : ℕ := sorry

/-- For any positive integer n, the product of φ(n) and τ(n) is greater than or equal to n -/
theorem phi_tau_ge_n (n : ℕ+) : phi n * tau n ≥ n := by sorry

end NUMINAMATH_CALUDE_phi_tau_ge_n_l834_83412


namespace NUMINAMATH_CALUDE_constant_is_arithmetic_l834_83487

def is_constant_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a n = a m

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem constant_is_arithmetic :
  ∀ a : ℕ → ℝ, is_constant_sequence a → is_arithmetic_sequence a :=
by
  sorry

end NUMINAMATH_CALUDE_constant_is_arithmetic_l834_83487


namespace NUMINAMATH_CALUDE_gcd_problems_l834_83457

theorem gcd_problems :
  (Nat.gcd 63 84 = 21) ∧ (Nat.gcd 351 513 = 27) := by
  sorry

end NUMINAMATH_CALUDE_gcd_problems_l834_83457


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l834_83419

theorem max_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (∀ m : ℝ, (1 / a + 1 / b ≥ m) → m ≤ 4) ∧ 
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 1 ∧ 1 / a + 1 / b = 4) :=
sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l834_83419


namespace NUMINAMATH_CALUDE_graph_intersects_x_equals_one_at_most_once_f_equals_g_l834_83429

-- Define a general function type
def RealFunction := ℝ → ℝ

-- Statement 1: The graph of y = f(x) intersects with x = 1 at most at one point
theorem graph_intersects_x_equals_one_at_most_once (f : RealFunction) :
  ∃! y, f 1 = y :=
sorry

-- Statement 2: f(x) = x^2 - 2x + 1 and g(t) = t^2 - 2t + 1 are the same function
def f (x : ℝ) : ℝ := x^2 - 2*x + 1
def g (t : ℝ) : ℝ := t^2 - 2*t + 1

theorem f_equals_g : f = g :=
sorry

end NUMINAMATH_CALUDE_graph_intersects_x_equals_one_at_most_once_f_equals_g_l834_83429


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l834_83403

/-- Proves that given a boat with a speed of 20 km/hr in still water and a stream with
    speed of 6 km/hr, if the boat travels the same time downstream as it does to
    travel 14 km upstream, then the distance traveled downstream is 26 km. -/
theorem boat_downstream_distance
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (upstream_distance : ℝ)
  (h1 : boat_speed = 20)
  (h2 : stream_speed = 6)
  (h3 : upstream_distance = 14)
  (h4 : (upstream_distance / (boat_speed - stream_speed)) =
        (downstream_distance / (boat_speed + stream_speed))) :
  downstream_distance = 26 :=
by
  sorry


end NUMINAMATH_CALUDE_boat_downstream_distance_l834_83403


namespace NUMINAMATH_CALUDE_not_perfect_square_123_ones_l834_83418

def number_with_ones (n : ℕ) : ℕ :=
  (10^n - 1) * 10^n + 123

theorem not_perfect_square_123_ones :
  ∀ n : ℕ, ∃ k : ℕ, (number_with_ones n) ≠ k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_123_ones_l834_83418


namespace NUMINAMATH_CALUDE_circus_investment_revenue_l834_83405

/-- A circus production investment problem -/
theorem circus_investment_revenue (overhead : ℕ) (production_cost : ℕ) (break_even_performances : ℕ) :
  overhead = 81000 →
  production_cost = 7000 →
  break_even_performances = 9 →
  (overhead + break_even_performances * production_cost) / break_even_performances = 16000 :=
by sorry

end NUMINAMATH_CALUDE_circus_investment_revenue_l834_83405


namespace NUMINAMATH_CALUDE_twenty_first_figure_squares_l834_83474

/-- The number of squares in the nth figure of the sequence -/
def num_squares (n : ℕ) : ℕ := n^2 + (n-1)^2

/-- The theorem stating that the 21st figure has 841 squares -/
theorem twenty_first_figure_squares : num_squares 21 = 841 := by
  sorry

end NUMINAMATH_CALUDE_twenty_first_figure_squares_l834_83474


namespace NUMINAMATH_CALUDE_min_value_theorem_solution_set_theorem_l834_83492

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |x - 1|

-- Theorem for part (1)
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = f (-1)) :
  (2/a + 1/b) ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = f (-1) ∧ 2/a₀ + 1/b₀ = 8 := by
  sorry

-- Theorem for part (2)
theorem solution_set_theorem (x : ℝ) :
  f x > 1/2 ↔ x < 5/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_solution_set_theorem_l834_83492
