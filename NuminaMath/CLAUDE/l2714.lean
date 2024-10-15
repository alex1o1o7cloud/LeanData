import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_two_primes_24_l2714_271410

theorem sum_of_two_primes_24 : ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p + q = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_primes_24_l2714_271410


namespace NUMINAMATH_CALUDE_value_of_expression_l2714_271425

theorem value_of_expression (α : Real) (h : 4 * Real.sin α - 3 * Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + 2 * Real.sin (2 * α)) = 25 / 64 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2714_271425


namespace NUMINAMATH_CALUDE_problem_G6_1_l2714_271460

theorem problem_G6_1 (p : ℝ) : 
  p = (21^3 - 11^3) / (21^2 + 21*11 + 11^2) → p = 10 := by
  sorry


end NUMINAMATH_CALUDE_problem_G6_1_l2714_271460


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l2714_271423

theorem fruit_seller_apples : ∀ (original : ℕ),
  (original : ℝ) * 0.6 = 420 → original = 700 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l2714_271423


namespace NUMINAMATH_CALUDE_fraction_zero_implies_a_negative_two_l2714_271464

theorem fraction_zero_implies_a_negative_two (a : ℝ) : 
  (a^2 - 4) / (a - 2) = 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_a_negative_two_l2714_271464


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l2714_271477

theorem matrix_equation_solution : 
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]
  N^3 - 3 • N^2 + 2 • N = !![2, 6; 3, 1] :=
by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l2714_271477


namespace NUMINAMATH_CALUDE_no_solution_for_floor_equation_l2714_271427

theorem no_solution_for_floor_equation :
  ∀ x : ℝ, ⌊x⌋ + ⌊2*x⌋ + ⌊4*x⌋ + ⌊8*x⌋ + ⌊16*x⌋ + ⌊32*x⌋ ≠ 12345 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_floor_equation_l2714_271427


namespace NUMINAMATH_CALUDE_product_of_roots_is_root_of_sextic_l2714_271481

theorem product_of_roots_is_root_of_sextic (a b c d : ℝ) : 
  a^4 + a^3 - 1 = 0 → 
  b^4 + b^3 - 1 = 0 → 
  c^4 + c^3 - 1 = 0 → 
  d^4 + d^3 - 1 = 0 → 
  (a * b)^6 + (a * b)^4 + (a * b)^3 - (a * b)^2 - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_is_root_of_sextic_l2714_271481


namespace NUMINAMATH_CALUDE_line_slope_equals_k_l2714_271437

/-- 
Given a line passing through points (-1, -4) and (4, k),
if the slope of the line is equal to k, then k = 1.
-/
theorem line_slope_equals_k (k : ℝ) : 
  (k - (-4)) / (4 - (-1)) = k → k = 1 := by sorry

end NUMINAMATH_CALUDE_line_slope_equals_k_l2714_271437


namespace NUMINAMATH_CALUDE_theater_seat_count_l2714_271470

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increase : ℕ
  last_row_seats : ℕ

/-- Calculates the number of rows in the theater -/
def number_of_rows (t : Theater) : ℕ :=
  (t.last_row_seats - t.first_row_seats) / t.seat_increase + 1

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := number_of_rows t
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- The theater described in the problem -/
def problem_theater : Theater :=
  { first_row_seats := 15
  , seat_increase := 2
  , last_row_seats := 53 }

theorem theater_seat_count :
  total_seats problem_theater = 680 := by
  sorry

end NUMINAMATH_CALUDE_theater_seat_count_l2714_271470


namespace NUMINAMATH_CALUDE_return_trip_time_l2714_271454

/-- Calculates the return trip time given the outbound trip details -/
theorem return_trip_time (outbound_time : ℝ) (outbound_speed : ℝ) (speed_increase : ℝ) : 
  outbound_time = 6 →
  outbound_speed = 60 →
  speed_increase = 12 →
  (outbound_time * outbound_speed) / (outbound_speed + speed_increase) = 5 := by
  sorry

end NUMINAMATH_CALUDE_return_trip_time_l2714_271454


namespace NUMINAMATH_CALUDE_mod_congruence_unique_n_l2714_271482

theorem mod_congruence_unique_n (a b : ℤ) 
  (ha : a ≡ 22 [ZMOD 50])
  (hb : b ≡ 78 [ZMOD 50]) :
  ∃! n : ℤ, 150 ≤ n ∧ n ≤ 201 ∧ (a - b) ≡ n [ZMOD 50] ∧ n = 194 :=
sorry

end NUMINAMATH_CALUDE_mod_congruence_unique_n_l2714_271482


namespace NUMINAMATH_CALUDE_division_problem_l2714_271436

theorem division_problem (divisor quotient remainder : ℕ) 
  (h1 : divisor = 30)
  (h2 : quotient = 9)
  (h3 : remainder = 1) :
  divisor * quotient + remainder = 271 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l2714_271436


namespace NUMINAMATH_CALUDE_min_product_of_three_l2714_271467

def S : Finset Int := {-8, -6, -4, 0, 3, 5, 7}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdiff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  x * y * z = -280 ∧ 
  ∀ (p q r : Int), p ∈ S → q ∈ S → r ∈ S → p ≠ q → q ≠ r → p ≠ r → 
  p * q * r ≥ -280 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_three_l2714_271467


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2714_271496

theorem inequality_equivalence (x : ℝ) : 
  (∀ y : ℝ, y > 0 → (4 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y)) / (x + y) > 3 * x^2 * y) ↔ 
  (x > 0 ∧ x < 4) := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2714_271496


namespace NUMINAMATH_CALUDE_derivative_even_implies_a_equals_three_l2714_271400

/-- Given a function f(x) = x³ + (a-3)x² + αx, prove that if its derivative f'(x) is an even function, then a = 3 -/
theorem derivative_even_implies_a_equals_three (a α : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 + (a - 3) * x^2 + α * x
  let f' : ℝ → ℝ := λ x ↦ deriv f x
  (∀ x, f' (-x) = f' x) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_even_implies_a_equals_three_l2714_271400


namespace NUMINAMATH_CALUDE_boy_squirrel_walnuts_l2714_271443

theorem boy_squirrel_walnuts (initial_walnuts : ℕ) (boy_gathered : ℕ) (girl_brought : ℕ) (girl_ate : ℕ) (final_walnuts : ℕ) :
  initial_walnuts = 12 →
  girl_brought = 5 →
  girl_ate = 2 →
  final_walnuts = 20 →
  final_walnuts = initial_walnuts + boy_gathered - 1 + girl_brought - girl_ate →
  boy_gathered = 6 := by
sorry

end NUMINAMATH_CALUDE_boy_squirrel_walnuts_l2714_271443


namespace NUMINAMATH_CALUDE_even_sum_odd_vertices_l2714_271420

/-- Represents a country on the spherical map -/
structure Country where
  color : Fin 4  -- 0: red, 1: yellow, 2: blue, 3: green
  vertices : ℕ

/-- Represents the spherical map -/
structure SphericalMap where
  countries : List Country
  neighbor_relation : Country → Country → Prop

/-- The number of countries with odd vertices for a given color -/
def num_odd_vertices (m : SphericalMap) (c : Fin 4) : ℕ :=
  (m.countries.filter (λ country => country.color = c ∧ country.vertices % 2 = 1)).length

theorem even_sum_odd_vertices (m : SphericalMap) :
  (num_odd_vertices m 0 + num_odd_vertices m 2) % 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_even_sum_odd_vertices_l2714_271420


namespace NUMINAMATH_CALUDE_tiktok_house_theorem_l2714_271471

/-- Represents a 3x3 grid of bloggers --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Represents a day's arrangement of bloggers --/
def DailyArrangement := Fin 9 → Fin 3 × Fin 3

/-- Represents the three days of arrangements --/
def ThreeDayArrangements := Fin 3 → DailyArrangement

/-- Checks if two positions in the grid are adjacent --/
def are_adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- Counts the number of unique pairs formed over three days --/
def count_unique_pairs (arrangements : ThreeDayArrangements) : ℕ :=
  sorry

/-- The main theorem to be proved --/
theorem tiktok_house_theorem (arrangements : ThreeDayArrangements) :
  count_unique_pairs arrangements < (9 * 8) / 2 := by
  sorry

end NUMINAMATH_CALUDE_tiktok_house_theorem_l2714_271471


namespace NUMINAMATH_CALUDE_christina_total_driving_time_l2714_271441

-- Define the total journey distance
def total_distance : ℝ := 210

-- Define the speed limits for each segment
def speed_limit_1 : ℝ := 30
def speed_limit_2 : ℝ := 40
def speed_limit_3 : ℝ := 50
def speed_limit_4 : ℝ := 60

-- Define the distances covered in the second and third segments
def distance_2 : ℝ := 120
def distance_3 : ℝ := 50

-- Define the time spent in the second and third segments
def time_2 : ℝ := 3
def time_3 : ℝ := 1

-- Define Christina's driving time function
def christina_driving_time : ℝ := by sorry

-- Theorem statement
theorem christina_total_driving_time :
  christina_driving_time = 100 / 60 := by sorry

end NUMINAMATH_CALUDE_christina_total_driving_time_l2714_271441


namespace NUMINAMATH_CALUDE_three_hundred_thousand_squared_minus_million_l2714_271408

theorem three_hundred_thousand_squared_minus_million : (300000 * 300000) - 1000000 = 89990000000 := by
  sorry

end NUMINAMATH_CALUDE_three_hundred_thousand_squared_minus_million_l2714_271408


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_of_f_l2714_271463

-- Define the function f(x) = |x+2|
def f (x : ℝ) : ℝ := |x + 2|

-- State the theorem
theorem monotonic_increasing_interval_of_f :
  {x : ℝ | ∀ y, x ≤ y → f x ≤ f y} = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_of_f_l2714_271463


namespace NUMINAMATH_CALUDE_max_fold_length_less_than_eight_l2714_271456

theorem max_fold_length_less_than_eight (length width : ℝ) 
  (h_length : length = 6) (h_width : width = 5) : 
  Real.sqrt (length^2 + width^2) < 8 := by
  sorry

end NUMINAMATH_CALUDE_max_fold_length_less_than_eight_l2714_271456


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l2714_271465

theorem circle_radius_from_area :
  ∀ (r : ℝ), r > 0 → π * r^2 = 64 * π → r = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l2714_271465


namespace NUMINAMATH_CALUDE_matrix_commutation_result_l2714_271424

/-- Given two 2x2 matrices A and B, where A is fixed and B has variable entries,
    prove that if AB = BA and 4y ≠ z, then (x - w) / (z - 4y) = 0. -/
theorem matrix_commutation_result (x y z w : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]
  (A * B = B * A) → (4 * y ≠ z) → (x - w) / (z - 4 * y) = 0 := by
  sorry


end NUMINAMATH_CALUDE_matrix_commutation_result_l2714_271424


namespace NUMINAMATH_CALUDE_min_value_of_g_l2714_271402

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem min_value_of_g (f g : ℝ → ℝ) 
  (h_odd : IsOdd f) (h_even : IsEven g) 
  (h_sum : ∀ x, f x + g x = 2^x) : 
  ∃ m, m = 1 ∧ ∀ x, g x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_g_l2714_271402


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_300_l2714_271483

/-- Given a natural number n, returns the sum of digits in its binary representation -/
def sumOfBinaryDigits (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem: The sum of digits in the binary representation of 300 is 4 -/
theorem sum_of_binary_digits_300 : sumOfBinaryDigits 300 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_300_l2714_271483


namespace NUMINAMATH_CALUDE_pattern_continuation_l2714_271426

theorem pattern_continuation (h1 : 1 = 6) (h2 : 2 = 12) (h3 : 3 = 18) (h4 : 4 = 24) (h5 : 5 = 30) : 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_pattern_continuation_l2714_271426


namespace NUMINAMATH_CALUDE_furniture_assembly_time_l2714_271430

def chairs : ℕ := 2
def tables : ℕ := 2
def time_per_piece : ℕ := 8

def total_pieces : ℕ := chairs + tables

def total_time : ℕ := total_pieces * time_per_piece

theorem furniture_assembly_time : total_time = 32 := by
  sorry

end NUMINAMATH_CALUDE_furniture_assembly_time_l2714_271430


namespace NUMINAMATH_CALUDE_max_popsicles_with_budget_l2714_271469

def single_price : ℚ := 3/2
def box3_price : ℚ := 3
def box7_price : ℚ := 5
def budget : ℚ := 12

def max_popsicles (s p3 p7 : ℕ) : ℕ := s + 3 * p3 + 7 * p7

def valid_purchase (s p3 p7 : ℕ) : Prop :=
  single_price * s + box3_price * p3 + box7_price * p7 ≤ budget

theorem max_popsicles_with_budget :
  ∃ (s p3 p7 : ℕ), valid_purchase s p3 p7 ∧
    max_popsicles s p3 p7 = 15 ∧
    ∀ (s' p3' p7' : ℕ), valid_purchase s' p3' p7' →
      max_popsicles s' p3' p7' ≤ 15 := by sorry

end NUMINAMATH_CALUDE_max_popsicles_with_budget_l2714_271469


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2714_271439

theorem imaginary_part_of_z (z : ℂ) (h : Complex.abs (z + 2 * Complex.I) = Complex.abs z) :
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2714_271439


namespace NUMINAMATH_CALUDE_line_bisected_by_M_l2714_271484

-- Define the lines and point
def l₁ (x y : ℝ) : Prop := x - 3 * y + 10 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y - 8 = 0
def M : ℝ × ℝ := (0, 1)

-- Define the line we want to prove
def target_line (x y : ℝ) : Prop := y = -1/3 * x + 1

-- Theorem statement
theorem line_bisected_by_M :
  ∃ (A B : ℝ × ℝ),
    l₁ A.1 A.2 ∧
    l₂ B.1 B.2 ∧
    target_line A.1 A.2 ∧
    target_line B.1 B.2 ∧
    target_line M.1 M.2 ∧
    M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) :=
  sorry


end NUMINAMATH_CALUDE_line_bisected_by_M_l2714_271484


namespace NUMINAMATH_CALUDE_solution_property_l2714_271497

theorem solution_property (m n : ℝ) (hm : m ≠ 0) (h : m^2 + n*m - m = 0) : m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_property_l2714_271497


namespace NUMINAMATH_CALUDE_problem_statement_l2714_271458

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + x^2 + (a - 1) * x + 1

/-- The theorem statement -/
theorem problem_statement (a : ℝ) :
  (∀ x > 0, Real.exp x > x + 1) →
  (∀ x > 0, f a x ≤ x^4 + Real.exp x) →
  a ≤ Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2714_271458


namespace NUMINAMATH_CALUDE_min_value_of_a_l2714_271407

theorem min_value_of_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 → 1/x + a/y ≥ 4) → 
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_a_l2714_271407


namespace NUMINAMATH_CALUDE_all_zeros_assignment_l2714_271498

/-- Represents a vertex in the triangular grid -/
structure Vertex (n : ℕ) where
  x : Fin (n + 1)
  y : Fin (n + 1)
  h : x.val + y.val ≤ n

/-- Represents an assignment of real numbers to vertices -/
def Assignment (n : ℕ) := Vertex n → ℝ

/-- Checks if three vertices form a triangle parallel to the sides of the main triangle -/
def is_parallel_triangle (n : ℕ) (v1 v2 v3 : Vertex n) : Prop :=
  ∃ (dx dy : Fin (n + 1)), 
    (v2.x = v1.x + dx ∧ v2.y = v1.y) ∧
    (v3.x = v1.x ∧ v3.y = v1.y + dy)

/-- The main theorem -/
theorem all_zeros_assignment {n : ℕ} (h : n ≥ 3) 
  (f : Assignment n) 
  (sum_zero : ∀ (v1 v2 v3 : Vertex n), 
    is_parallel_triangle n v1 v2 v3 → f v1 + f v2 + f v3 = 0) :
  ∀ v : Vertex n, f v = 0 := by sorry

end NUMINAMATH_CALUDE_all_zeros_assignment_l2714_271498


namespace NUMINAMATH_CALUDE_even_digits_512_base5_l2714_271435

/-- Converts a natural number from base 10 to base 5 --/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers --/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of even digits in the base-5 representation of 512 is 3 --/
theorem even_digits_512_base5 : countEvenDigits (toBase5 512) = 3 := by
  sorry

end NUMINAMATH_CALUDE_even_digits_512_base5_l2714_271435


namespace NUMINAMATH_CALUDE_decimal_addition_l2714_271478

theorem decimal_addition : (0.9 : ℝ) + 0.99 = 1.89 := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_l2714_271478


namespace NUMINAMATH_CALUDE_alex_sweaters_l2714_271444

/-- The number of shirts Alex has to wash -/
def num_shirts : ℕ := 18

/-- The number of pants Alex has to wash -/
def num_pants : ℕ := 12

/-- The number of jeans Alex has to wash -/
def num_jeans : ℕ := 13

/-- The maximum number of items the washing machine can wash per cycle -/
def items_per_cycle : ℕ := 15

/-- The duration of each washing cycle in minutes -/
def cycle_duration : ℕ := 45

/-- The total time needed to wash all clothes in minutes -/
def total_wash_time : ℕ := 180

/-- The theorem stating that Alex has 17 sweaters to wash -/
theorem alex_sweaters : 
  ∃ (num_sweaters : ℕ), 
    (num_shirts + num_pants + num_jeans + num_sweaters) = 
    (total_wash_time / cycle_duration * items_per_cycle) ∧ 
    num_sweaters = 17 := by
  sorry

end NUMINAMATH_CALUDE_alex_sweaters_l2714_271444


namespace NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l2714_271452

/-- The length of a bridge that a train can cross, given the train's length, speed, and crossing time. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof that a 500m train traveling at 42 km/h crosses a bridge of approximately 200.2m in 60 seconds. -/
theorem specific_bridge_length : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |bridge_length 500 42 60 - 200.2| < ε :=
sorry

end NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l2714_271452


namespace NUMINAMATH_CALUDE_ratio_of_distances_l2714_271499

/-- Given four points P, Q, R, and S on a line (in that order), with distances PQ = 3, QR = 7, and PS = 22,
    prove that the ratio of PR to QS is 10/19. -/
theorem ratio_of_distances (P Q R S : ℝ) (h_order : P < Q ∧ Q < R ∧ R < S) 
  (h_PQ : Q - P = 3) (h_QR : R - Q = 7) (h_PS : S - P = 22) : 
  (R - P) / (S - Q) = 10 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_distances_l2714_271499


namespace NUMINAMATH_CALUDE_tank_capacity_l2714_271403

theorem tank_capacity : ∀ (T : ℚ),
  (3/4 : ℚ) * T + 7 = (7/8 : ℚ) * T →
  T = 56 := by sorry

end NUMINAMATH_CALUDE_tank_capacity_l2714_271403


namespace NUMINAMATH_CALUDE_divisibility_of_fifth_powers_l2714_271457

theorem divisibility_of_fifth_powers (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (y - z) * (z - x) * (x - y)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_fifth_powers_l2714_271457


namespace NUMINAMATH_CALUDE_simplification_equivalence_simplified_is_quadratic_trinomial_l2714_271414

-- Define the original polynomial
def original_polynomial (x : ℝ) : ℝ := 2*x^2 - 5*x + x^2 - 4*x + 5

-- Define the simplified polynomial
def simplified_polynomial (x : ℝ) : ℝ := 3*x^2 - 9*x + 5

-- Theorem stating that the simplified polynomial is equivalent to the original
theorem simplification_equivalence :
  ∀ x, original_polynomial x = simplified_polynomial x :=
by sorry

-- Define what it means for a polynomial to be quadratic
def is_quadratic (p : ℝ → ℝ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ ∀ x, p x = a*x^2 + b*x + c

-- Define what it means for a polynomial to have exactly three terms
def has_three_terms (p : ℝ → ℝ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ∀ x, p x = a*x^2 + b*x + c

-- Theorem stating that the simplified polynomial is a quadratic trinomial
theorem simplified_is_quadratic_trinomial :
  is_quadratic simplified_polynomial ∧ has_three_terms simplified_polynomial :=
by sorry

end NUMINAMATH_CALUDE_simplification_equivalence_simplified_is_quadratic_trinomial_l2714_271414


namespace NUMINAMATH_CALUDE_meeting_participants_ratio_l2714_271438

/-- Given information about participants in a meeting, prove the ratio of female democrats to total female participants -/
theorem meeting_participants_ratio :
  let total_participants : ℕ := 810
  let female_democrats : ℕ := 135
  let male_democrat_ratio : ℚ := 1/4
  let total_democrat_ratio : ℚ := 1/3
  ∃ (female_participants male_participants : ℕ),
    female_participants + male_participants = total_participants ∧
    female_democrats + male_democrat_ratio * male_participants = total_democrat_ratio * total_participants ∧
    female_democrats / female_participants = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_meeting_participants_ratio_l2714_271438


namespace NUMINAMATH_CALUDE_cut_rectangle_perimeter_example_l2714_271434

/-- The perimeter of a rectangle with squares cut from its corners -/
def cut_rectangle_perimeter (length width cut : ℝ) : ℝ :=
  2 * (length + width)

/-- Theorem: The perimeter of a 12x5 cm rectangle with 2x2 cm squares cut from each corner is 34 cm -/
theorem cut_rectangle_perimeter_example :
  cut_rectangle_perimeter 12 5 2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_cut_rectangle_perimeter_example_l2714_271434


namespace NUMINAMATH_CALUDE_video_game_cost_l2714_271451

/-- If two identical video games cost $50 in total, then seven of these video games will cost $175. -/
theorem video_game_cost (cost_of_two : ℝ) (h : cost_of_two = 50) :
  7 * (cost_of_two / 2) = 175 := by
  sorry

end NUMINAMATH_CALUDE_video_game_cost_l2714_271451


namespace NUMINAMATH_CALUDE_rabbit_position_final_position_l2714_271450

theorem rabbit_position (n : ℕ) : 
  1 + n * (n + 1) / 2 = (n + 1) * (n + 2) / 2 := by sorry

theorem final_position : 
  (2020 + 1) * (2020 + 2) / 2 = 2041211 := by sorry

end NUMINAMATH_CALUDE_rabbit_position_final_position_l2714_271450


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2714_271445

/-- An arithmetic sequence with first term a-1 and common difference 2 has the general formula a_n = a + 2n - 3 -/
theorem arithmetic_sequence_formula (a : ℝ) :
  let a_n := fun (n : ℕ) => a - 1 + 2 * (n - 1)
  ∀ n : ℕ, a_n n = a + 2 * n - 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2714_271445


namespace NUMINAMATH_CALUDE_altitude_least_integer_l2714_271480

theorem altitude_least_integer (a b : ℝ) (h : a = 5 ∧ b = 12) : 
  ∃ (L : ℝ), L = (a * b) / (2 * Real.sqrt (a^2 + b^2)) ∧ 
  (∀ (n : ℤ), (n : ℝ) > L → n ≥ 5) ∧ (4 : ℝ) < L :=
sorry

end NUMINAMATH_CALUDE_altitude_least_integer_l2714_271480


namespace NUMINAMATH_CALUDE_smallest_t_is_four_l2714_271487

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def smallest_valid_t : ℕ → Prop
  | t => is_valid_triangle 7.5 11 (t : ℝ) ∧ 
         ∀ k : ℕ, k < t → ¬is_valid_triangle 7.5 11 (k : ℝ)

theorem smallest_t_is_four : smallest_valid_t 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_t_is_four_l2714_271487


namespace NUMINAMATH_CALUDE_absolute_value_at_two_l2714_271476

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- Condition that g is a third-degree polynomial with specific absolute values -/
def SatisfiesCondition (g : ThirdDegreePolynomial) : Prop :=
  ∃ (a b c d : ℝ), ∀ x, g x = a * x^3 + b * x^2 + c * x + d ∧
  (|g 0| = 10) ∧ (|g 1| = 10) ∧ (|g 3| = 10) ∧
  (|g 4| = 10) ∧ (|g 5| = 10) ∧ (|g 8| = 10)

/-- Theorem stating that if g satisfies the condition, then |g(2)| = 20 -/
theorem absolute_value_at_two
  (g : ThirdDegreePolynomial)
  (h : SatisfiesCondition g) :
  |g 2| = 20 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_at_two_l2714_271476


namespace NUMINAMATH_CALUDE_expand_expression_l2714_271419

theorem expand_expression (x y : ℝ) : (2 * x - 5) * (3 * y + 15) = 6 * x * y + 30 * x - 15 * y - 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2714_271419


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_value_l2714_271404

theorem quadratic_root_implies_a_value (a : ℝ) : 
  (∃ (z : ℂ), z = 2 + I ∧ z^2 - 4*z + a = 0) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_value_l2714_271404


namespace NUMINAMATH_CALUDE_prob_at_least_three_same_l2714_271488

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 4

/-- The probability of rolling a specific value on a single die -/
def probSingleDie : ℚ := 1 / numSides

/-- The probability that at least three out of four fair six-sided dice show the same value -/
def probAtLeastThreeSame : ℚ := 7 / 72

/-- Theorem stating that the probability of at least three out of four fair six-sided dice 
    showing the same value is 7/72 -/
theorem prob_at_least_three_same :
  probAtLeastThreeSame = 
    (1 * probSingleDie ^ 3) + -- Probability of all four dice showing same value
    (4 * probSingleDie ^ 2 * (1 - probSingleDie)) -- Probability of exactly three dice showing same value
  := by sorry

end NUMINAMATH_CALUDE_prob_at_least_three_same_l2714_271488


namespace NUMINAMATH_CALUDE_max_distance_unit_circle_l2714_271418

/-- The maximum distance between any two points on the unit circle is 2 -/
theorem max_distance_unit_circle : 
  ∀ (α β : ℝ), 
  let P := (Real.cos α, Real.sin α)
  let Q := (Real.cos β, Real.sin β)
  ∃ (maxDist : ℝ), maxDist = 2 ∧ 
    ∀ (α' β' : ℝ), 
    let P' := (Real.cos α', Real.sin α')
    let Q' := (Real.cos β', Real.sin β')
    Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≤ maxDist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_unit_circle_l2714_271418


namespace NUMINAMATH_CALUDE_four_students_seven_seats_l2714_271409

/-- The number of ways to arrange students in seats with adjacent empty seats -/
def seating_arrangements (total_seats : ℕ) (students : ℕ) (adjacent_empty : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 480 ways to arrange 4 students in 7 seats with 2 adjacent empty seats -/
theorem four_students_seven_seats : seating_arrangements 7 4 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_four_students_seven_seats_l2714_271409


namespace NUMINAMATH_CALUDE_radioactive_balls_identification_l2714_271479

/-- Represents a strategy for identifying radioactive balls -/
structure Strategy where
  num_tests : ℕ
  -- Other fields omitted for simplicity

/-- Represents the outcome of applying a strategy -/
inductive Outcome
  | IdentifiedBoth
  | NotIdentified

/-- Applies a strategy to a set of balls and returns the outcome -/
def apply_strategy (s : Strategy) (total_balls : ℕ) (radioactive_balls : ℕ) : Outcome :=
  sorry

theorem radioactive_balls_identification
  (total_balls : ℕ)
  (radioactive_balls : ℕ)
  (h_total : total_balls = 11)
  (h_radioactive : radioactive_balls = 2) :
  (∀ s : Strategy, s.num_tests < 7 → ∃ outcome, outcome = Outcome.NotIdentified) ∧
  (∃ s : Strategy, s.num_tests = 7 ∧ apply_strategy s total_balls radioactive_balls = Outcome.IdentifiedBoth) :=
sorry

end NUMINAMATH_CALUDE_radioactive_balls_identification_l2714_271479


namespace NUMINAMATH_CALUDE_unique_solution_l2714_271412

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (a + b) - f (a * b) = f a * f b - 1

/-- The theorem stating that the only function satisfying the equation is f(n) = n + 1 -/
theorem unique_solution (f : ℤ → ℤ) (h : SatisfiesEquation f) :
  ∀ n : ℤ, f n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2714_271412


namespace NUMINAMATH_CALUDE_bianca_albums_l2714_271468

theorem bianca_albums (total_pics : ℕ) (main_album_pics : ℕ) (pics_per_album : ℕ) : 
  total_pics = 33 → main_album_pics = 27 → pics_per_album = 2 → 
  (total_pics - main_album_pics) / pics_per_album = 3 := by
  sorry

end NUMINAMATH_CALUDE_bianca_albums_l2714_271468


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l2714_271433

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_tenth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 19 = -18) :
  a 10 = -9 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l2714_271433


namespace NUMINAMATH_CALUDE_andy_position_after_2021_moves_l2714_271446

-- Define the ant's position as a pair of integers
def Position := ℤ × ℤ

-- Define the direction as an enumeration
inductive Direction
| North
| East
| South
| West

-- Define the initial position and direction
def initial_position : Position := (10, -10)
def initial_direction : Direction := Direction.North

-- Define a function to calculate the movement distance for a given move number
def movement_distance (move_number : ℕ) : ℕ :=
  (move_number / 4 : ℕ) + 1

-- Define a function to update the direction after a right turn
def turn_right (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.East
  | Direction.East => Direction.South
  | Direction.South => Direction.West
  | Direction.West => Direction.North

-- Define a function to update the position based on direction and distance
def move (pos : Position) (dir : Direction) (dist : ℤ) : Position :=
  match dir with
  | Direction.North => (pos.1, pos.2 + dist)
  | Direction.East => (pos.1 + dist, pos.2)
  | Direction.South => (pos.1, pos.2 - dist)
  | Direction.West => (pos.1 - dist, pos.2)

-- Define a function to simulate the ant's movement for a given number of moves
def simulate_movement (num_moves : ℕ) : Position :=
  sorry -- Actual implementation would go here

-- State the theorem
theorem andy_position_after_2021_moves :
  simulate_movement 2021 = (10, 496) := by sorry

end NUMINAMATH_CALUDE_andy_position_after_2021_moves_l2714_271446


namespace NUMINAMATH_CALUDE_correct_answers_for_86_points_min_correct_for_first_prize_l2714_271428

/-- Represents a math competition with given parameters -/
structure MathCompetition where
  total_questions : ℕ
  full_score : ℕ
  correct_points : ℕ
  wrong_points : ℤ
  unanswered_points : ℕ

/-- Theorem for part (1) of the problem -/
theorem correct_answers_for_86_points (comp : MathCompetition)
    (h1 : comp.total_questions = 25)
    (h2 : comp.full_score = 100)
    (h3 : comp.correct_points = 4)
    (h4 : comp.wrong_points = -1)
    (h5 : comp.unanswered_points = 0)
    (h6 : ∃ (x : ℕ), x * comp.correct_points + (comp.total_questions - 1 - x) * comp.wrong_points = 86) :
    ∃ (x : ℕ), x = 22 ∧ x * comp.correct_points + (comp.total_questions - 1 - x) * comp.wrong_points = 86 :=
  sorry

/-- Theorem for part (2) of the problem -/
theorem min_correct_for_first_prize (comp : MathCompetition)
    (h1 : comp.total_questions = 25)
    (h2 : comp.full_score = 100)
    (h3 : comp.correct_points = 4)
    (h4 : comp.wrong_points = -1)
    (h5 : comp.unanswered_points = 0) :
    ∃ (x : ℕ), x ≥ 23 ∧ ∀ (y : ℕ), y * comp.correct_points + (comp.total_questions - y) * comp.wrong_points ≥ 90 → y ≥ x :=
  sorry

end NUMINAMATH_CALUDE_correct_answers_for_86_points_min_correct_for_first_prize_l2714_271428


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l2714_271461

theorem ratio_of_sum_to_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (h : x + y = 7 * (x - y)) : x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l2714_271461


namespace NUMINAMATH_CALUDE_fraction_addition_l2714_271421

theorem fraction_addition : (2 : ℚ) / 3 + (1 : ℚ) / 6 = (5 : ℚ) / 6 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_l2714_271421


namespace NUMINAMATH_CALUDE_matrix_power_four_l2714_271490

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_power_four :
  A^4 = !![0, -9; 9, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l2714_271490


namespace NUMINAMATH_CALUDE_sum_medians_gt_four_times_circumradius_l2714_271417

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define properties of a triangle
def Triangle.isNonObtuse (t : Triangle) : Prop := sorry

def Triangle.medians (t : Triangle) : ℝ × ℝ × ℝ := sorry

def Triangle.circumradius (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem sum_medians_gt_four_times_circumradius 
  (t : Triangle) (h : t.isNonObtuse) : 
  let (m₁, m₂, m₃) := t.medians
  m₁ + m₂ + m₃ > 4 * t.circumradius :=
by
  sorry

end NUMINAMATH_CALUDE_sum_medians_gt_four_times_circumradius_l2714_271417


namespace NUMINAMATH_CALUDE_solve_system_l2714_271406

theorem solve_system (x y : ℤ) (h1 : x + y = 260) (h2 : x - y = 200) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2714_271406


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2714_271432

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 6

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (3*x + y - 7 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2714_271432


namespace NUMINAMATH_CALUDE_problem_statement_l2714_271493

open Real

variable (a b : ℝ)

theorem problem_statement (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a) * (b + 1/b) > 4 ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → sqrt (1 + a) + sqrt (1 + b) ≤ sqrt 6) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → (a * b + 4 * a + b) / (4 * a + b) ≤ 10 / 9) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2714_271493


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2714_271401

theorem fraction_multiplication : (1 / 3 : ℚ) * (3 / 4 : ℚ) * (4 / 5 : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2714_271401


namespace NUMINAMATH_CALUDE_sapling_growth_l2714_271447

/-- The height of a sapling after n years -/
def sapling_height (n : ℕ) : ℝ :=
  1.5 + 0.2 * n

/-- Theorem: The height of the sapling after n years is 1.5 + 0.2n meters -/
theorem sapling_growth (n : ℕ) :
  sapling_height n = 1.5 + 0.2 * n := by
  sorry

end NUMINAMATH_CALUDE_sapling_growth_l2714_271447


namespace NUMINAMATH_CALUDE_decagon_adjacent_vertices_probability_l2714_271405

/-- A decagon is a polygon with 10 vertices -/
def Decagon : Type := Nat

/-- The number of vertices in a decagon -/
def num_vertices : Nat := 10

/-- The number of adjacent vertices for each vertex in a decagon -/
def num_adjacent : Nat := 2

/-- The probability of choosing two distinct adjacent vertices in a decagon -/
def prob_adjacent_vertices (d : Decagon) : Rat :=
  num_adjacent / (num_vertices - 1)

theorem decagon_adjacent_vertices_probability :
  ∀ d : Decagon, prob_adjacent_vertices d = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_decagon_adjacent_vertices_probability_l2714_271405


namespace NUMINAMATH_CALUDE_two_copy_machines_output_l2714_271442

/-- Calculates the total number of copies made by two copy machines in a given time -/
def total_copies (rate1 rate2 time : ℕ) : ℕ :=
  rate1 * time + rate2 * time

/-- Proves that two copy machines with given rates produce 3300 copies in 30 minutes -/
theorem two_copy_machines_output : total_copies 35 75 30 = 3300 := by
  sorry

end NUMINAMATH_CALUDE_two_copy_machines_output_l2714_271442


namespace NUMINAMATH_CALUDE_taxi_average_speed_l2714_271411

/-- The average speed of a taxi that travels 100 kilometers in 1 hour and 15 minutes is 80 kilometers per hour. -/
theorem taxi_average_speed :
  let distance : ℝ := 100 -- distance in kilometers
  let time : ℝ := 1.25 -- time in hours (1 hour and 15 minutes = 1.25 hours)
  let average_speed := distance / time
  average_speed = 80 := by sorry

end NUMINAMATH_CALUDE_taxi_average_speed_l2714_271411


namespace NUMINAMATH_CALUDE_cubic_monotone_increasing_l2714_271455

/-- A cubic function f(x) = ax³ - x² + x - 5 is monotonically increasing on ℝ if and only if a ≥ 1/3 -/
theorem cubic_monotone_increasing (a : ℝ) :
  (∀ x : ℝ, HasDerivAt (fun x => a * x^3 - x^2 + x - 5) (3 * a * x^2 - 2 * x + 1) x) →
  (∀ x y : ℝ, x < y → (a * x^3 - x^2 + x - 5) < (a * y^3 - y^2 + y - 5)) ↔
  a ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_monotone_increasing_l2714_271455


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l2714_271449

/-- Represents a cricket team -/
structure CricketTeam where
  numPlayers : Nat
  captainAge : Nat
  captainBattingAvg : Nat
  wicketKeeperAge : Nat
  wicketKeeperBattingAvg : Nat
  youngestPlayerBattingAvg : Nat

/-- Calculate the average age of the team -/
def averageTeamAge (team : CricketTeam) : Rat :=
  sorry

theorem cricket_team_average_age 
  (team : CricketTeam)
  (h1 : team.numPlayers = 11)
  (h2 : team.captainAge = 25)
  (h3 : team.captainBattingAvg = 45)
  (h4 : team.wicketKeeperAge = team.captainAge + 5)
  (h5 : team.wicketKeeperBattingAvg = 35)
  (h6 : team.youngestPlayerBattingAvg = 42)
  (h7 : ∀ (remainingPlayersAvgAge : Rat),
        remainingPlayersAvgAge = (averageTeamAge team - 1) ∧
        (team.captainAge + team.wicketKeeperAge + remainingPlayersAvgAge * (team.numPlayers - 2)) / team.numPlayers = averageTeamAge team)
  (h8 : ∃ (youngestPlayerAge : Nat),
        youngestPlayerAge ≤ team.wicketKeeperAge - 15 ∧
        youngestPlayerAge > 0) :
  averageTeamAge team = 23 :=
sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l2714_271449


namespace NUMINAMATH_CALUDE_min_sum_squares_l2714_271474

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Define the line
def line (x y : ℝ) : Prop := y = x - 1

-- Define a point on the line
structure PointOnLine where
  x : ℝ
  y : ℝ
  on_line : line x y

-- Define the diameter AB
structure Diameter where
  A : ℝ × ℝ
  B : ℝ × ℝ
  is_diameter : ∀ (x y : ℝ), circle_C x y → 
    (x - A.1)^2 + (y - A.2)^2 + (x - B.1)^2 + (y - B.2)^2 = 4

-- Theorem statement
theorem min_sum_squares (d : Diameter) :
  ∃ (min : ℝ), min = 6 ∧ 
  ∀ (P : PointOnLine), 
    (P.x - d.A.1)^2 + (P.y - d.A.2)^2 + (P.x - d.B.1)^2 + (P.y - d.B.2)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2714_271474


namespace NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l2714_271486

/-- The maximum y-coordinate of a point on the curve r = sin 3θ is 4√3/9 -/
theorem max_y_coordinate_sin_3theta :
  let r : ℝ → ℝ := λ θ => Real.sin (3 * θ)
  let y : ℝ → ℝ := λ θ => r θ * Real.sin θ
  ∃ (max_y : ℝ), (∀ θ, y θ ≤ max_y) ∧ (max_y = 4 * Real.sqrt 3 / 9) := by
  sorry

end NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l2714_271486


namespace NUMINAMATH_CALUDE_deepak_age_l2714_271475

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 5 / 2 →
  rahul_age + 6 = 26 →
  deepak_age = 8 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l2714_271475


namespace NUMINAMATH_CALUDE_max_sqrt_sum_l2714_271413

theorem max_sqrt_sum (x : ℝ) (h : 0 ≤ x ∧ x ≤ 20) :
  Real.sqrt (x + 16) + Real.sqrt (20 - x) + 2 * Real.sqrt x ≤ 8 * Real.sqrt 3 / 3 ∧
  ∃ y : ℝ, 0 ≤ y ∧ y ≤ 20 ∧ Real.sqrt (y + 16) + Real.sqrt (20 - y) + 2 * Real.sqrt y = 8 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_sqrt_sum_l2714_271413


namespace NUMINAMATH_CALUDE_b_income_less_than_others_l2714_271448

structure Income where
  c : ℝ
  a : ℝ
  b_salary : ℝ
  b_commission : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

def Income.b_total (i : Income) : ℝ := i.b_salary + i.b_commission

def Income.others_total (i : Income) : ℝ := i.a + i.c + i.d + i.e + i.f

def valid_income (i : Income) : Prop :=
  i.a = i.c * 1.2 ∧
  i.b_salary = i.a * 1.25 ∧
  i.b_commission = (i.a + i.c) * 0.05 ∧
  i.d = i.b_total * 0.85 ∧
  i.e = i.c * 1.1 ∧
  i.f = (i.b_total + i.e) / 2

theorem b_income_less_than_others (i : Income) (h : valid_income i) :
  i.b_total < i.others_total ∧ i.b_commission = i.c * 0.11 :=
sorry

end NUMINAMATH_CALUDE_b_income_less_than_others_l2714_271448


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2714_271422

theorem polynomial_factorization (a b m n : ℝ) : 
  (3 * a^2 - 6 * a * b + 3 * b^2 = 3 * (a - b)^2) ∧ 
  (4 * m^2 - 9 * n^2 = (2 * m - 3 * n) * (2 * m + 3 * n)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2714_271422


namespace NUMINAMATH_CALUDE_point_D_transformation_l2714_271415

def rotate_90_clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def transform_point (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_axis (reflect_x_axis (rotate_90_clockwise p))

theorem point_D_transformation :
  transform_point (4, -3) = (3, 4) := by
  sorry

end NUMINAMATH_CALUDE_point_D_transformation_l2714_271415


namespace NUMINAMATH_CALUDE_expression_value_l2714_271459

theorem expression_value :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 4 + 6 - 8 + 10 - 12 + 14) = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2714_271459


namespace NUMINAMATH_CALUDE_tom_tim_ratio_l2714_271485

structure TypingSpeed where
  tim : ℝ
  tom : ℝ

def combined_speed (s : TypingSpeed) : ℝ := s.tim + s.tom

def increased_speed (s : TypingSpeed) : ℝ := s.tim + 1.4 * s.tom

theorem tom_tim_ratio (s : TypingSpeed) 
  (h1 : combined_speed s = 20)
  (h2 : increased_speed s = 24) : 
  s.tom / s.tim = 1 := by
  sorry

end NUMINAMATH_CALUDE_tom_tim_ratio_l2714_271485


namespace NUMINAMATH_CALUDE_total_travel_time_travel_time_calculation_l2714_271416

/-- Calculates the total travel time between two towns given specific conditions -/
theorem total_travel_time (total_distance : ℝ) (initial_fraction : ℝ) (lunch_time : ℝ) 
  (second_fraction : ℝ) (pit_stop_time : ℝ) (speed_increase : ℝ) : ℝ :=
  let initial_distance := initial_fraction * total_distance
  let initial_speed := initial_distance
  let remaining_distance := total_distance - initial_distance
  let second_distance := second_fraction * remaining_distance
  let final_distance := remaining_distance - second_distance
  let final_speed := initial_speed + speed_increase
  initial_fraction + lunch_time + (second_distance / initial_speed) + 
  pit_stop_time + (final_distance / final_speed)

/-- The total travel time between the two towns is 5.25 hours -/
theorem travel_time_calculation : 
  total_travel_time 200 (1/4) 1 (1/2) (1/2) 10 = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_time_travel_time_calculation_l2714_271416


namespace NUMINAMATH_CALUDE_no_solution_exists_l2714_271494

theorem no_solution_exists : ¬ ∃ (m n : ℤ), 5 * m^2 - 6 * m * n + 7 * n^2 = 1985 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2714_271494


namespace NUMINAMATH_CALUDE_least_number_with_remainder_4_l2714_271466

def is_valid_divisor (n : ℕ) : Prop := n > 0 ∧ 252 % n = 0

theorem least_number_with_remainder_4 : 
  (∀ x : ℕ, is_valid_divisor x → 256 % x = 4) ∧ 
  (∀ n : ℕ, n < 256 → ∃ y : ℕ, is_valid_divisor y ∧ n % y ≠ 4) :=
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_4_l2714_271466


namespace NUMINAMATH_CALUDE_b_completion_time_l2714_271431

/-- The number of days A needs to complete the entire work -/
def a_total_days : ℚ := 15

/-- The number of days A actually works -/
def a_worked_days : ℚ := 5

/-- The number of days B needs to complete the entire work -/
def b_total_days : ℚ := 9/2

/-- The fraction of work completed by A -/
def a_work_completed : ℚ := a_worked_days / a_total_days

/-- The fraction of work B needs to complete -/
def b_work_to_complete : ℚ := 1 - a_work_completed

/-- The fraction of work B completes per day -/
def b_work_per_day : ℚ := 1 / b_total_days

/-- The number of days B needs to complete the remaining work -/
def b_days_needed : ℚ := b_work_to_complete / b_work_per_day

theorem b_completion_time : b_days_needed = 3 := by
  sorry

end NUMINAMATH_CALUDE_b_completion_time_l2714_271431


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l2714_271472

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * a + 4 * b + 4 * c = 180) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) 
  (h3 : a = 10) : 
  2 * (a * b + b * c + c * a) = 1400 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l2714_271472


namespace NUMINAMATH_CALUDE_icosahedron_edges_l2714_271491

/-- A regular icosahedron is a convex polyhedron with 20 congruent equilateral triangular faces -/
def RegularIcosahedron : Type := sorry

/-- The number of edges in a polyhedron -/
def num_edges (p : RegularIcosahedron) : ℕ := sorry

/-- Theorem: A regular icosahedron has 30 edges -/
theorem icosahedron_edges :
  ∀ (i : RegularIcosahedron), num_edges i = 30 := by sorry

end NUMINAMATH_CALUDE_icosahedron_edges_l2714_271491


namespace NUMINAMATH_CALUDE_only_45_increases_ninefold_l2714_271489

/-- A function that inserts a zero between the tens and units digits of a natural number -/
def insertZero (n : ℕ) : ℕ :=
  10 * (n / 10) * 10 + n % 10

/-- The property that a number increases ninefold when a zero is inserted between its digits -/
def increasesNinefold (n : ℕ) : Prop :=
  insertZero n = 9 * n

theorem only_45_increases_ninefold :
  ∀ n : ℕ, n ≠ 0 → (increasesNinefold n ↔ n = 45) :=
sorry

end NUMINAMATH_CALUDE_only_45_increases_ninefold_l2714_271489


namespace NUMINAMATH_CALUDE_certain_number_existence_and_uniqueness_l2714_271453

theorem certain_number_existence_and_uniqueness :
  ∃! x : ℚ, x / 3 + x + 3 = 63 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_existence_and_uniqueness_l2714_271453


namespace NUMINAMATH_CALUDE_final_numbers_correct_l2714_271440

/-- The number of elements in the initial sequence -/
def n : ℕ := 2022

/-- The number of operations performed -/
def operations : ℕ := (n - 2) / 2

/-- The arithmetic mean operation on squares -/
def arithmetic_mean_operation (x : ℕ) : ℕ := x^2 + 1

/-- The final two numbers after applying the arithmetic mean operation -/
def final_numbers : Fin 2 → ℕ
| 0 => arithmetic_mean_operation 1011 + operations
| 1 => arithmetic_mean_operation 1012 + operations

theorem final_numbers_correct :
  final_numbers 0 = 1023131 ∧ final_numbers 1 = 1025154 := by sorry

end NUMINAMATH_CALUDE_final_numbers_correct_l2714_271440


namespace NUMINAMATH_CALUDE_cooldrink_mixture_l2714_271495

/-- Amount of Cool-drink B added to create a mixture with 10% jasmine water -/
theorem cooldrink_mixture (total_volume : ℝ) (cooldrink_a_volume : ℝ) (jasmine_water_added : ℝ) (fruit_juice_added : ℝ)
  (cooldrink_a_jasmine_percent : ℝ) (cooldrink_a_fruit_percent : ℝ)
  (cooldrink_b_jasmine_percent : ℝ) (cooldrink_b_fruit_percent : ℝ)
  (final_jasmine_percent : ℝ) :
  total_volume = 150 →
  cooldrink_a_volume = 80 →
  jasmine_water_added = 8 →
  fruit_juice_added = 20 →
  cooldrink_a_jasmine_percent = 0.12 →
  cooldrink_a_fruit_percent = 0.88 →
  cooldrink_b_jasmine_percent = 0.05 →
  cooldrink_b_fruit_percent = 0.95 →
  final_jasmine_percent = 0.10 →
  ∃ cooldrink_b_volume : ℝ,
    cooldrink_b_volume = 136 ∧
    (cooldrink_a_volume * cooldrink_a_jasmine_percent + cooldrink_b_volume * cooldrink_b_jasmine_percent + jasmine_water_added) / 
    (cooldrink_a_volume + cooldrink_b_volume + jasmine_water_added + fruit_juice_added) = final_jasmine_percent :=
by
  sorry

end NUMINAMATH_CALUDE_cooldrink_mixture_l2714_271495


namespace NUMINAMATH_CALUDE_factor_implies_m_value_l2714_271429

theorem factor_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, ∃ k : ℝ, x^2 - m*x - 40 = (x + 5) * k) → m = 13 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_m_value_l2714_271429


namespace NUMINAMATH_CALUDE_abc_inequality_l2714_271492

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b * c * (a + b + c + Real.sqrt (a^2 + b^2 + c^2))) / 
  ((a^2 + b^2 + c^2) * (a * b + b * c + a * c)) ≤ (3 + Real.sqrt 3) / 9 :=
sorry

end NUMINAMATH_CALUDE_abc_inequality_l2714_271492


namespace NUMINAMATH_CALUDE_profit_share_difference_l2714_271473

theorem profit_share_difference (a b c : ℕ) (b_profit : ℕ) : 
  a = 8000 → b = 10000 → c = 12000 → b_profit = 1400 →
  ∃ (a_profit c_profit : ℕ), 
    a_profit * b = b_profit * a ∧ 
    c_profit * b = b_profit * c ∧ 
    c_profit - a_profit = 560 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_difference_l2714_271473


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2714_271462

theorem cube_root_equation_solution :
  ∃! x : ℚ, Real.rpow (5 + x) (1/3 : ℝ) = 4/3 :=
by
  use -71/27
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2714_271462
