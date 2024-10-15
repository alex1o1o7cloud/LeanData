import Mathlib

namespace NUMINAMATH_CALUDE_vector_dot_product_roots_l661_66157

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.sqrt 3 * (Real.cos x)^2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, -2)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem vector_dot_product_roots (x₁ x₂ : ℝ) :
  0 < x₁ ∧ x₁ < x₂ ∧ x₂ < π ∧
  dot_product (m x₁) (n x₁) = 1/2 - Real.sqrt 3 ∧
  dot_product (m x₂) (n x₂) = 1/2 - Real.sqrt 3 →
  Real.sin (x₁ - x₂) = -Real.sqrt 15 / 4 := by
sorry

end NUMINAMATH_CALUDE_vector_dot_product_roots_l661_66157


namespace NUMINAMATH_CALUDE_min_distinct_values_l661_66161

theorem min_distinct_values (n : ℕ) (mode_freq : ℕ) (second_freq : ℕ) 
  (h1 : n = 3000)
  (h2 : mode_freq = 15)
  (h3 : second_freq = 14)
  (h4 : ∀ k : ℕ, k ≠ mode_freq → k ≤ second_freq) :
  (∃ x : ℕ, x * mode_freq + x * second_freq + (n - x * mode_freq - x * second_freq) ≤ n ∧ 
   ∀ y : ℕ, y < x → y * mode_freq + y * second_freq + (n - y * mode_freq - y * second_freq) > n) →
  x = 232 := by
sorry

end NUMINAMATH_CALUDE_min_distinct_values_l661_66161


namespace NUMINAMATH_CALUDE_kitchen_tiles_l661_66100

/-- The number of tiles needed to cover a rectangular floor -/
def tiles_needed (floor_length floor_width tile_area : ℕ) : ℕ :=
  (floor_length * floor_width) / tile_area

/-- Proof that 576 tiles are needed for the given floor and tile specifications -/
theorem kitchen_tiles :
  tiles_needed 48 72 6 = 576 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_tiles_l661_66100


namespace NUMINAMATH_CALUDE_square_sum_of_roots_l661_66174

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

-- State the theorem
theorem square_sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = 19) : (a + b)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_roots_l661_66174


namespace NUMINAMATH_CALUDE_vector_subtraction_l661_66125

/-- Given two vectors OM and ON in ℝ², prove that the vector MN has coordinates (-8, 1) -/
theorem vector_subtraction (OM ON : ℝ × ℝ) (h1 : OM = (3, -2)) (h2 : ON = (-5, -1)) :
  ON - OM = (-8, 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l661_66125


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l661_66180

theorem complex_fraction_simplification :
  let z₁ : ℂ := -2 + 5*I
  let z₂ : ℂ := 6 - 3*I
  z₁ / z₂ = -9/15 + 8/15*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l661_66180


namespace NUMINAMATH_CALUDE_bolded_area_percentage_l661_66167

theorem bolded_area_percentage (s : ℝ) (h : s > 0) : 
  let square_area := s^2
  let total_area := 4 * square_area
  let bolded_area_1 := (1/2) * square_area
  let bolded_area_2 := (1/2) * square_area
  let bolded_area_3 := (1/8) * square_area
  let bolded_area_4 := (1/4) * square_area
  let total_bolded_area := bolded_area_1 + bolded_area_2 + bolded_area_3 + bolded_area_4
  (total_bolded_area / total_area) * 100 = 100/3
:= by sorry

end NUMINAMATH_CALUDE_bolded_area_percentage_l661_66167


namespace NUMINAMATH_CALUDE_max_M_value_l661_66187

theorem max_M_value (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (eq1 : x - 2*y = z - 2*u) (eq2 : 2*y*z = u*x) (h_zy : z ≥ y) :
  ∃ M : ℝ, M > 0 ∧ M ≤ z/y ∧ ∀ N : ℝ, (N > 0 ∧ N ≤ z/y → N ≤ M) ∧ M = 6 + 4*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_M_value_l661_66187


namespace NUMINAMATH_CALUDE_complex_division_l661_66162

theorem complex_division (z₁ z₂ : ℂ) : z₁ = 1 + I ∧ z₂ = 2 * I → z₂ / z₁ = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l661_66162


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_9_15_45_l661_66178

theorem gcf_lcm_sum_9_15_45 : ∃ (C D : ℕ),
  (C = Nat.gcd 9 (Nat.gcd 15 45)) ∧
  (D = Nat.lcm 9 (Nat.lcm 15 45)) ∧
  (C + D = 60) := by
sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_9_15_45_l661_66178


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l661_66112

/-- Given a geometric sequence {a_n} where a_1 = 3 and 4a_1, 2a_2, a_3 form an arithmetic sequence,
    prove that a_3 + a_4 + a_5 = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 3 →                            -- a_1 = 3
  4 * a 1 - 2 * a 2 = 2 * a 2 - a 3 →  -- arithmetic sequence condition
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l661_66112


namespace NUMINAMATH_CALUDE_book_cost_price_l661_66130

/-- The cost price of a book sold for $200 with a 20% profit is $166.67 -/
theorem book_cost_price (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 200 ∧ profit_percentage = 20 → 
  (selling_price / (1 + profit_percentage / 100) : ℝ) = 166.67 := by
sorry

end NUMINAMATH_CALUDE_book_cost_price_l661_66130


namespace NUMINAMATH_CALUDE_remainder_theorem_l661_66190

theorem remainder_theorem (x : ℝ) : ∃ (Q : ℝ → ℝ) (R : ℝ → ℝ),
  (∀ x, x^100 = (x^2 - 3*x + 2) * Q x + R x) ∧
  (∃ a b, R = fun x ↦ a * x + b) ∧
  R = fun x ↦ 2^100 * (x - 1) - (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l661_66190


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l661_66170

theorem arithmetic_sequence_ratio (x y d₁ d₂ : ℝ) (h₁ : d₁ ≠ 0) (h₂ : d₂ ≠ 0) 
  (h₃ : x + 4 * d₁ = y) (h₄ : x + 5 * d₂ = y) : d₁ / d₂ = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l661_66170


namespace NUMINAMATH_CALUDE_f_properties_l661_66128

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - 8^x)

theorem f_properties :
  (∀ x, f x ≠ 0 → x ≤ 2/3) ∧
  (∀ y, (∃ x, f x = y) → 0 ≤ y ∧ y < 2) ∧
  (∀ x, f x ≤ 1 → Real.log 3 / Real.log 8 ≤ x ∧ x ≤ 2/3) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l661_66128


namespace NUMINAMATH_CALUDE_vertex_in_fourth_quadrant_l661_66153

-- Define the line y = x + m
def line (x m : ℝ) : ℝ := x + m

-- Define the parabola y = (x + m)^2 - 1
def parabola (x m : ℝ) : ℝ := (x + m)^2 - 1

-- Define what it means for a line to pass through the first, third, and fourth quadrants
def passes_through_134 (m : ℝ) : Prop :=
  ∃ (x1 x3 x4 : ℝ), 
    (x1 > 0 ∧ line x1 m > 0) ∧
    (x3 < 0 ∧ line x3 m < 0) ∧
    (x4 > 0 ∧ line x4 m < 0)

-- Define the fourth quadrant
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Theorem statement
theorem vertex_in_fourth_quadrant (m : ℝ) :
  passes_through_134 m → in_fourth_quadrant (-m) (-1) :=
sorry

end NUMINAMATH_CALUDE_vertex_in_fourth_quadrant_l661_66153


namespace NUMINAMATH_CALUDE_florist_roses_l661_66165

/-- The number of roses a florist has after selling some and picking more. -/
def roses_after_selling_and_picking (initial : ℕ) (sold : ℕ) (picked : ℕ) : ℕ :=
  initial - sold + picked

/-- Theorem: Given the specific numbers from the problem, 
    the florist ends up with 40 roses. -/
theorem florist_roses : roses_after_selling_and_picking 37 16 19 = 40 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_l661_66165


namespace NUMINAMATH_CALUDE_area_between_parallel_chords_l661_66102

theorem area_between_parallel_chords (r : ℝ) (d : ℝ) (h1 : r = 8) (h2 : d = 8) :
  let chord_length := 2 * Real.sqrt (r ^ 2 - (d / 2) ^ 2)
  let segment_area := (1 / 3) * π * r ^ 2 - (1 / 2) * chord_length * (d / 2)
  2 * segment_area = 32 * Real.sqrt 3 + 64 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_area_between_parallel_chords_l661_66102


namespace NUMINAMATH_CALUDE_basketball_success_rate_l661_66149

theorem basketball_success_rate (p : ℝ) 
  (h : 1 - p^2 = 16/25) : p = 3/5 := by sorry

end NUMINAMATH_CALUDE_basketball_success_rate_l661_66149


namespace NUMINAMATH_CALUDE_smallest_integer_with_conditions_l661_66113

/-- Represents a natural number as a list of its digits in reverse order -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  have : n / 10 < n := sorry
  (n % 10) :: digits (n / 10)

/-- Checks if the digits of a number are in strictly increasing order -/
def increasing_digits (n : ℕ) : Prop :=
  List.Pairwise (· < ·) (digits n)

/-- Calculates the sum of squares of digits of a number -/
def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  (digits n).map (λ d => d * d) |> List.sum

/-- Calculates the product of digits of a number -/
def product_of_digits (n : ℕ) : ℕ :=
  (digits n).prod

/-- The main theorem -/
theorem smallest_integer_with_conditions :
  ∃ n : ℕ,
    (∀ m : ℕ, m < n →
      (sum_of_squares_of_digits m ≠ 85 ∨
       ¬increasing_digits m)) ∧
    sum_of_squares_of_digits n = 85 ∧
    increasing_digits n ∧
    product_of_digits n = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_conditions_l661_66113


namespace NUMINAMATH_CALUDE_allan_plum_count_l661_66117

/-- The number of plums Sharon has -/
def sharon_plums : ℕ := 7

/-- The difference between Sharon's plums and Allan's plums -/
def plum_difference : ℕ := 3

/-- The number of plums Allan has -/
def allan_plums : ℕ := sharon_plums - plum_difference

theorem allan_plum_count : allan_plums = 4 := by
  sorry

end NUMINAMATH_CALUDE_allan_plum_count_l661_66117


namespace NUMINAMATH_CALUDE_simplify_expression_l661_66152

theorem simplify_expression : 
  (Real.sqrt 8 + Real.sqrt 12) - (2 * Real.sqrt 3 - Real.sqrt 2) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l661_66152


namespace NUMINAMATH_CALUDE_hikers_speed_hikers_speed_specific_l661_66110

/-- The problem of determining a hiker's speed given specific conditions involving a cyclist -/
theorem hikers_speed (cyclist_speed : ℝ) (cyclist_travel_time : ℝ) (hiker_catch_up_time : ℝ) : ℝ :=
  let hiker_speed := (cyclist_speed * cyclist_travel_time) / hiker_catch_up_time
  by
    -- Assuming:
    -- 1. The hiker walks at a constant rate.
    -- 2. A cyclist passes the hiker, traveling in the same direction at 'cyclist_speed'.
    -- 3. The cyclist stops after 'cyclist_travel_time'.
    -- 4. The hiker continues walking at her constant rate.
    -- 5. The cyclist waits 'hiker_catch_up_time' until the hiker catches up.
    
    -- Prove: hiker_speed = 20/3

    sorry

/-- The specific instance of the hiker's speed problem -/
theorem hikers_speed_specific : hikers_speed 20 (1/12) (1/4) = 20/3 :=
  by sorry

end NUMINAMATH_CALUDE_hikers_speed_hikers_speed_specific_l661_66110


namespace NUMINAMATH_CALUDE_school_A_percentage_l661_66131

theorem school_A_percentage (total : ℕ) (science_percent : ℚ) (non_science : ℕ) :
  total = 300 →
  science_percent = 30 / 100 →
  non_science = 42 →
  ∃ (school_A_percent : ℚ),
    school_A_percent = 20 / 100 ∧
    non_science = (1 - science_percent) * (school_A_percent * total) :=
by sorry

end NUMINAMATH_CALUDE_school_A_percentage_l661_66131


namespace NUMINAMATH_CALUDE_working_hours_growth_equation_l661_66179

-- Define the initial and final average working hours
def initial_hours : ℝ := 40
def final_hours : ℝ := 48.4

-- Define the growth rate variable
variable (x : ℝ)

-- State the theorem
theorem working_hours_growth_equation :
  initial_hours * (1 + x)^2 = final_hours := by
  sorry

end NUMINAMATH_CALUDE_working_hours_growth_equation_l661_66179


namespace NUMINAMATH_CALUDE_festival_average_surfers_l661_66106

/-- The average number of surfers at the Rip Curl Myrtle Beach Surf Festival -/
def average_surfers (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) : ℚ :=
  (day1 + day2 + day3) / 3

/-- Theorem: The average number of surfers at the Festival for three days is 1400 -/
theorem festival_average_surfers :
  let day1 : ℕ := 1500
  let day2 : ℕ := day1 + 600
  let day3 : ℕ := day1 * 2 / 5
  average_surfers day1 day2 day3 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_festival_average_surfers_l661_66106


namespace NUMINAMATH_CALUDE_parallelogram_d_coordinates_l661_66169

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a vector in 2D space
structure Vector2D where
  x : ℝ
  y : ℝ

-- Define a parallelogram
structure Parallelogram where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

def vector_between_points (p1 p2 : Point2D) : Vector2D :=
  { x := p2.x - p1.x, y := p2.y - p1.y }

theorem parallelogram_d_coordinates :
  ∀ (ABCD : Parallelogram),
    ABCD.A = { x := 1, y := 2 } →
    ABCD.B = { x := -2, y := 0 } →
    vector_between_points ABCD.A ABCD.C = { x := 2, y := -3 } →
    ABCD.D = { x := 6, y := 1 } :=
by
  sorry


end NUMINAMATH_CALUDE_parallelogram_d_coordinates_l661_66169


namespace NUMINAMATH_CALUDE_unknown_number_proof_l661_66155

theorem unknown_number_proof (n : ℕ) 
  (h1 : Nat.lcm 24 n = 168) 
  (h2 : Nat.gcd 24 n = 4) : 
  n = 28 := by
sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l661_66155


namespace NUMINAMATH_CALUDE_garrett_roses_count_l661_66133

/-- The number of red roses Mrs. Santiago has -/
def santiago_roses : ℕ := 58

/-- The difference in the number of roses between Mrs. Santiago and Mrs. Garrett -/
def difference : ℕ := 34

/-- The number of red roses Mrs. Garrett has -/
def garrett_roses : ℕ := santiago_roses - difference

theorem garrett_roses_count : garrett_roses = 24 := by
  sorry

end NUMINAMATH_CALUDE_garrett_roses_count_l661_66133


namespace NUMINAMATH_CALUDE_equation_solution_l661_66166

theorem equation_solution : ∃! x : ℝ, 3 * x - 4 = -2 * x + 11 ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l661_66166


namespace NUMINAMATH_CALUDE_repeating_decimal_value_l661_66175

/-- The repeating decimal 0.0000253253325333... -/
def x : ℚ := 253 / 990000

/-- The result of (10^7 - 10^5) * x -/
def result : ℚ := (10^7 - 10^5) * x

/-- Theorem stating that the result is equal to 253/990 -/
theorem repeating_decimal_value : result = 253 / 990 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_value_l661_66175


namespace NUMINAMATH_CALUDE_max_omega_for_monotonic_sin_l661_66145

theorem max_omega_for_monotonic_sin (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.sin (ω * x)) →
  ω > 0 →
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 3 → f x < f y) →
  ω ≤ 3 / 2 ∧ ∀ ω' > 3 / 2, ∃ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 3 ∧ f x ≥ f y :=
by sorry

end NUMINAMATH_CALUDE_max_omega_for_monotonic_sin_l661_66145


namespace NUMINAMATH_CALUDE_dans_pokemon_cards_l661_66105

/-- The number of Pokemon cards Dan has -/
def dans_cards : ℕ := 41

/-- Sally's initial number of Pokemon cards -/
def sallys_initial_cards : ℕ := 27

/-- The number of Pokemon cards Sally bought -/
def cards_sally_bought : ℕ := 20

/-- The difference between Sally's and Dan's cards -/
def card_difference : ℕ := 6

theorem dans_pokemon_cards :
  sallys_initial_cards + cards_sally_bought = dans_cards + card_difference :=
sorry

end NUMINAMATH_CALUDE_dans_pokemon_cards_l661_66105


namespace NUMINAMATH_CALUDE_inequality_proof_l661_66120

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  (x^2 / (y + z) + y^2 / (z + x) + z^2 / (x + y)) ≥ 
  (1/2) * ((x^2 + y^2) / (x + y) + (y^2 + z^2) / (y + z) + (z^2 + x^2) / (z + x)) ∧
  (1/2) * ((x^2 + y^2) / (x + y) + (y^2 + z^2) / (y + z) + (z^2 + x^2) / (z + x)) ≥ (x + y + z) / 2 ∧
  (x + y + z) / 2 ≥ 3/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l661_66120


namespace NUMINAMATH_CALUDE_unique_number_with_18_factors_l661_66136

/-- The number of positive factors of n -/
def num_factors (n : ℕ) : ℕ := sorry

theorem unique_number_with_18_factors (x : ℕ) : 
  num_factors x = 18 ∧ 
  18 ∣ x ∧ 
  24 ∣ x → 
  x = 288 := by sorry

end NUMINAMATH_CALUDE_unique_number_with_18_factors_l661_66136


namespace NUMINAMATH_CALUDE_single_digit_square_5929_l661_66154

theorem single_digit_square_5929 :
  ∃! (A : ℕ), A < 10 ∧ (10 * A + A)^2 = 5929 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_single_digit_square_5929_l661_66154


namespace NUMINAMATH_CALUDE_grey_cats_count_l661_66148

/-- The number of grey cats in a house after a series of events -/
def grey_cats_after_events : ℕ :=
  let initial_total : ℕ := 16
  let initial_white : ℕ := 2
  let initial_black : ℕ := (25 * initial_total) / 100
  let black_after_leaving : ℕ := initial_black / 2
  let white_after_arrival : ℕ := initial_white + 2
  let initial_grey : ℕ := initial_total - initial_white - initial_black
  initial_grey + 1

/-- Theorem stating the number of grey cats after the events -/
theorem grey_cats_count : grey_cats_after_events = 11 := by
  sorry

end NUMINAMATH_CALUDE_grey_cats_count_l661_66148


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l661_66176

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2) :
  (1 - 1 / a) / ((a^2 - 1) / a) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l661_66176


namespace NUMINAMATH_CALUDE_det_of_specific_matrix_l661_66159

theorem det_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![6, -2; -3, 5]
  Matrix.det A = 24 := by
sorry

end NUMINAMATH_CALUDE_det_of_specific_matrix_l661_66159


namespace NUMINAMATH_CALUDE_sharon_trip_distance_l661_66183

def normal_time : ℝ := 200
def reduced_speed_time : ℝ := 310
def speed_reduction : ℝ := 30

def trip_distance : ℝ := 220

theorem sharon_trip_distance :
  let normal_speed := trip_distance / normal_time
  let reduced_speed := normal_speed - speed_reduction / 60
  (trip_distance / 3) / normal_speed + (2 * trip_distance / 3) / reduced_speed = reduced_speed_time :=
by sorry

end NUMINAMATH_CALUDE_sharon_trip_distance_l661_66183


namespace NUMINAMATH_CALUDE_runners_meet_time_l661_66168

/-- The length of the circular track in meters -/
def track_length : ℝ := 400

/-- The speeds of the three runners in meters per second -/
def runner_speeds : Fin 3 → ℝ
  | 0 => 5
  | 1 => 5.5
  | 2 => 6

/-- The time in seconds for the runners to meet again at the starting point -/
def meeting_time : ℝ := 800

theorem runners_meet_time :
  ∀ (i : Fin 3), ∃ (n : ℕ), (runner_speeds i * meeting_time) = n * track_length :=
sorry

end NUMINAMATH_CALUDE_runners_meet_time_l661_66168


namespace NUMINAMATH_CALUDE_heather_final_blocks_l661_66115

-- Define the initial number of blocks Heather has
def heather_initial : ℝ := 86.0

-- Define the number of blocks Jose shares
def jose_shares : ℝ := 41.0

-- Theorem statement
theorem heather_final_blocks : 
  heather_initial + jose_shares = 127.0 := by
  sorry

end NUMINAMATH_CALUDE_heather_final_blocks_l661_66115


namespace NUMINAMATH_CALUDE_remainder_three_pow_twenty_mod_seven_l661_66172

theorem remainder_three_pow_twenty_mod_seven : 3^20 % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_pow_twenty_mod_seven_l661_66172


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l661_66198

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 12 * x^2 + 7 * y^2 = 4620 ↔
    ((x = 7 ∨ x = -7) ∧ (y = 24 ∨ y = -24)) ∨
    ((x = 14 ∨ x = -14) ∧ (y = 18 ∨ y = -18)) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l661_66198


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l661_66101

theorem condition_necessary_not_sufficient : 
  (∀ x : ℝ, x = 0 → (2*x - 1)*x = 0) ∧ 
  ¬(∀ x : ℝ, (2*x - 1)*x = 0 → x = 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l661_66101


namespace NUMINAMATH_CALUDE_digit_101_of_7_over_26_l661_66164

theorem digit_101_of_7_over_26 : ∃ (d : ℕ), d = 2 ∧ 
  (∃ (a : ℕ → ℕ), 
    (∀ n, a n < 10) ∧ 
    (∀ n, (7 * 10^(n+1)) / 26 % 10 = a n) ∧ 
    a 100 = d) := by
  sorry

end NUMINAMATH_CALUDE_digit_101_of_7_over_26_l661_66164


namespace NUMINAMATH_CALUDE_train_length_l661_66132

/-- The length of a train given its speed and time to pass an observer -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 144 → time_s = 10 → speed_kmh * (1000 / 3600) * time_s = 400 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l661_66132


namespace NUMINAMATH_CALUDE_walnut_problem_l661_66160

/-- Calculates the final number of walnuts in the main burrow after the actions of three squirrels. -/
def final_walnut_count (initial : ℕ) (boy_gather boy_drop boy_hide : ℕ)
  (girl_bring girl_eat girl_give girl_lose girl_knock : ℕ)
  (third_gather third_drop third_hide third_return third_give : ℕ) : ℕ :=
  initial + boy_gather - boy_drop - boy_hide +
  girl_bring - girl_eat - girl_give - girl_lose - girl_knock +
  third_return

/-- The final number of walnuts in the main burrow is 44. -/
theorem walnut_problem :
  final_walnut_count 30 20 4 8 15 5 4 3 2 10 1 3 6 1 = 44 := by
  sorry

end NUMINAMATH_CALUDE_walnut_problem_l661_66160


namespace NUMINAMATH_CALUDE_children_per_seat_l661_66185

theorem children_per_seat (total_children : ℕ) (total_seats : ℕ) 
  (h1 : total_children = 58) (h2 : total_seats = 29) : 
  total_children / total_seats = 2 := by
sorry

end NUMINAMATH_CALUDE_children_per_seat_l661_66185


namespace NUMINAMATH_CALUDE_rate_of_discount_l661_66139

/-- Calculate the rate of discount given the marked price and selling price -/
theorem rate_of_discount (marked_price selling_price : ℝ) :
  marked_price = 200 →
  selling_price = 120 →
  (marked_price - selling_price) / marked_price * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_rate_of_discount_l661_66139


namespace NUMINAMATH_CALUDE_f_properties_l661_66151

noncomputable def f (x : ℝ) : ℝ := 1/2 * (Real.cos x)^2 + (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + 1

theorem f_properties :
  let period : ℝ := Real.pi
  let max_value : ℝ := 7/4
  let min_value : ℝ := (5 + Real.sqrt 3) / 4
  let interval : Set ℝ := Set.Icc (Real.pi / 12) (Real.pi / 4)
  (∀ x : ℝ, f (x + period) = f x) ∧
  (∀ t : ℝ, t > 0 → (∀ x : ℝ, f (x + t) = f x) → t ≥ period) ∧
  (∃ x ∈ interval, f x = max_value ∧ ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, f x = min_value ∧ ∀ y ∈ interval, f y ≥ f x) ∧
  (f (Real.pi / 6) = max_value) ∧
  (f (Real.pi / 12) = min_value) ∧
  (f (Real.pi / 4) = min_value) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l661_66151


namespace NUMINAMATH_CALUDE_afternoon_email_count_l661_66163

/-- Represents the number of emails Jack received at different times of the day -/
structure EmailCount where
  morning : ℕ
  afternoon : ℕ
  evening : ℕ

/-- The theorem stating that Jack received 7 emails in the afternoon -/
theorem afternoon_email_count (e : EmailCount) 
  (h1 : e.morning = 10)
  (h2 : e.evening = 17)
  (h3 : e.morning = e.afternoon + 3) :
  e.afternoon = 7 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_email_count_l661_66163


namespace NUMINAMATH_CALUDE_not_kth_power_consecutive_product_l661_66138

theorem not_kth_power_consecutive_product (m k : ℕ) (hk : k > 1) :
  ¬ ∃ (a : ℤ), m * (m + 1) = a^k := by
  sorry

end NUMINAMATH_CALUDE_not_kth_power_consecutive_product_l661_66138


namespace NUMINAMATH_CALUDE_optimal_quadruple_l661_66135

def is_valid_quadruple (k l m n : ℕ) : Prop :=
  k > l ∧ l > m ∧ m > n

def sum_inverse (k l m n : ℕ) : ℚ :=
  1 / k + 1 / l + 1 / m + 1 / n

theorem optimal_quadruple :
  ∀ k l m n : ℕ,
    is_valid_quadruple k l m n →
    sum_inverse k l m n < 1 →
    sum_inverse k l m n ≤ sum_inverse 43 7 3 2 :=
by sorry

end NUMINAMATH_CALUDE_optimal_quadruple_l661_66135


namespace NUMINAMATH_CALUDE_complex_number_line_l661_66143

theorem complex_number_line (z : ℂ) (h : 2 * (1 + Complex.I) * z = 1 - Complex.I) :
  z.im = -1/2 * z.re := by
  sorry

end NUMINAMATH_CALUDE_complex_number_line_l661_66143


namespace NUMINAMATH_CALUDE_five_digit_numbers_count_l661_66108

theorem five_digit_numbers_count : 
  (Finset.filter (fun n : Nat => 
    n ≥ 10000 ∧ n < 100000 ∧ 
    (n / 10000) ≠ 5 ∧
    (n % 10) ≠ 2 ∧
    (Finset.card (Finset.image (fun i => (n / (10 ^ i)) % 10) (Finset.range 5))) = 5
  ) (Finset.range 100000)).card = 8 * 9 * 8 * 7 * 6 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_numbers_count_l661_66108


namespace NUMINAMATH_CALUDE_scientific_notation_of_316000000_l661_66199

/-- Definition of scientific notation -/
def is_scientific_notation (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ ∃ (x : ℝ), x = a * (10 : ℝ) ^ n

/-- The number to be represented in scientific notation -/
def number : ℝ := 316000000

/-- Theorem stating that 316000000 in scientific notation is 3.16 × 10^8 -/
theorem scientific_notation_of_316000000 :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n ∧ number = a * (10 : ℝ) ^ n ∧ a = 3.16 ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_316000000_l661_66199


namespace NUMINAMATH_CALUDE_cubic_root_sum_l661_66156

theorem cubic_root_sum (k m : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 7*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  k + m = 22 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l661_66156


namespace NUMINAMATH_CALUDE_total_marbles_count_l661_66188

/-- The number of marbles Connie has -/
def connie_marbles : ℕ := 39

/-- The number of marbles Juan has -/
def juan_marbles : ℕ := connie_marbles + 25

/-- The number of marbles Maria has -/
def maria_marbles : ℕ := 2 * juan_marbles

/-- The total number of marbles for all three people -/
def total_marbles : ℕ := connie_marbles + juan_marbles + maria_marbles

theorem total_marbles_count : total_marbles = 231 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_count_l661_66188


namespace NUMINAMATH_CALUDE_some_negative_numbers_satisfy_inequality_l661_66129

theorem some_negative_numbers_satisfy_inequality :
  (∃ x : ℝ, x < 0 ∧ (1 + x) * (1 - 9 * x) > 0) ↔
  (∃ x₀ : ℝ, x₀ < 0 ∧ (1 + x₀) * (1 - 9 * x₀) > 0) :=
by sorry

end NUMINAMATH_CALUDE_some_negative_numbers_satisfy_inequality_l661_66129


namespace NUMINAMATH_CALUDE_batch_not_qualified_l661_66142

-- Define the parameters of the normal distribution
def mean : ℝ := 4
def std_dev : ℝ := 0.5  -- sqrt(0.25)

-- Define the measured diameter
def measured_diameter : ℝ := 5.7

-- Define a function to determine if a batch is qualified
def is_qualified (x : ℝ) : Prop :=
  (x - mean) / std_dev ≤ 3 ∧ (x - mean) / std_dev ≥ -3

-- Theorem statement
theorem batch_not_qualified : ¬(is_qualified measured_diameter) :=
sorry

end NUMINAMATH_CALUDE_batch_not_qualified_l661_66142


namespace NUMINAMATH_CALUDE_sqrt_eight_equals_two_sqrt_two_l661_66171

theorem sqrt_eight_equals_two_sqrt_two : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_equals_two_sqrt_two_l661_66171


namespace NUMINAMATH_CALUDE_square_semicircle_perimeter_l661_66195

theorem square_semicircle_perimeter : 
  let square_side : ℝ := 2 / Real.pi
  let semicircle_diameter : ℝ := square_side
  let full_circle_circumference : ℝ := Real.pi * semicircle_diameter
  let region_perimeter : ℝ := 2 * full_circle_circumference
  region_perimeter = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_semicircle_perimeter_l661_66195


namespace NUMINAMATH_CALUDE_chongqing_population_scientific_notation_l661_66124

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The population of Chongqing at the end of 2022 -/
def chongqing_population : ℕ := 32000000

/-- Converts a natural number to its scientific notation representation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem chongqing_population_scientific_notation :
  to_scientific_notation chongqing_population =
    ScientificNotation.mk 3.2 7 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_chongqing_population_scientific_notation_l661_66124


namespace NUMINAMATH_CALUDE_number_with_specific_remainders_l661_66191

theorem number_with_specific_remainders (n : ℕ) :
  ∃ (x : ℕ+), 
    x > 1 ∧ 
    n % x = 2 ∧ 
    (2 * n) % x = 4 → 
    x = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_with_specific_remainders_l661_66191


namespace NUMINAMATH_CALUDE_exists_polynomial_composition_l661_66192

-- Define the polynomials P and Q
variable (K : Type*) [Field K]
variable (P Q : K → K)

-- Define the condition for the existence of R
variable (R : K → K → K)
variable (h : ∀ x y, P x - P y = R x y * (Q x - Q y))

-- Theorem statement
theorem exists_polynomial_composition :
  ∃ S : K → K, ∀ x, P x = S (Q x) := by
  sorry

end NUMINAMATH_CALUDE_exists_polynomial_composition_l661_66192


namespace NUMINAMATH_CALUDE_inequality_solution_set_l661_66141

theorem inequality_solution_set (x : ℝ) : (x + 1) / (x - 1) ≤ 0 ↔ x ∈ Set.Icc (-1) 1 \ {1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l661_66141


namespace NUMINAMATH_CALUDE_last_triangle_perimeter_l661_66104

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  valid : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Generates the next triangle in the sequence based on the current triangle -/
def nextTriangle (T : Triangle) : Option Triangle := sorry

/-- The sequence of triangles starting from T₁ -/
def triangleSequence : ℕ → Option Triangle
  | 0 => some ⟨1003, 1004, 1005, sorry⟩
  | n + 1 => (triangleSequence n).bind nextTriangle

/-- The perimeter of a triangle -/
def perimeter (T : Triangle) : ℝ := T.a + T.b + T.c

/-- Finds the last existing triangle in the sequence -/
def lastTriangle : Option Triangle := sorry

theorem last_triangle_perimeter :
  ∀ T, lastTriangle = some T → perimeter T = 753 / 128 := by sorry

end NUMINAMATH_CALUDE_last_triangle_perimeter_l661_66104


namespace NUMINAMATH_CALUDE_income_left_percentage_man_income_left_l661_66116

/-- Given a man's spending pattern, calculate the percentage of income left --/
theorem income_left_percentage (total_income : ℝ) (food_percent : ℝ) (education_percent : ℝ) 
  (transport_percent : ℝ) (rent_percent : ℝ) : ℝ :=
  let initial_expenses := food_percent + education_percent + transport_percent
  let remaining_after_initial := 100 - initial_expenses
  let rent_amount := rent_percent * remaining_after_initial / 100
  let total_expenses := initial_expenses + rent_amount
  100 - total_expenses

/-- Prove that the man is left with 12.6% of his income --/
theorem man_income_left :
  income_left_percentage 100 42 18 12 55 = 12.6 := by
  sorry

end NUMINAMATH_CALUDE_income_left_percentage_man_income_left_l661_66116


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l661_66197

-- Define the trajectory C
def C (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y + Real.sqrt 3)^2) + Real.sqrt (x^2 + (y - Real.sqrt 3)^2) = 4

-- Define the intersection line
def intersectionLine (x y : ℝ) : Prop := y = (1/2) * x

theorem trajectory_and_intersection :
  -- The equation of trajectory C
  (∀ x y, C x y ↔ x^2 + y^2/4 = 1) ∧
  -- The length of chord AB
  (∃ x₁ y₁ x₂ y₂, 
    C x₁ y₁ ∧ C x₂ y₂ ∧ 
    intersectionLine x₁ y₁ ∧ intersectionLine x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l661_66197


namespace NUMINAMATH_CALUDE_max_digit_sum_18_l661_66186

/-- Represents a digit (1 to 9) -/
def Digit := {d : ℕ // 1 ≤ d ∧ d ≤ 9}

/-- Calculates the value of a number with n identical digits -/
def digitSum (d : Digit) (n : ℕ) : ℕ := d.val * ((10^n - 1) / 9)

/-- The main theorem -/
theorem max_digit_sum_18 :
  ∃ (a b c : Digit) (n₁ n₂ : ℕ+),
    n₁ ≠ n₂ ∧
    digitSum c (2 * n₁) - digitSum b n₁ = (digitSum a n₁)^2 ∧
    digitSum c (2 * n₂) - digitSum b n₂ = (digitSum a n₂)^2 ∧
    ∀ (a' b' c' : Digit),
      (∃ (m₁ m₂ : ℕ+), m₁ ≠ m₂ ∧
        digitSum c' (2 * m₁) - digitSum b' m₁ = (digitSum a' m₁)^2 ∧
        digitSum c' (2 * m₂) - digitSum b' m₂ = (digitSum a' m₂)^2) →
      a'.val + b'.val + c'.val ≤ a.val + b.val + c.val ∧
      a.val + b.val + c.val = 18 :=
by sorry

end NUMINAMATH_CALUDE_max_digit_sum_18_l661_66186


namespace NUMINAMATH_CALUDE_problem_solution_l661_66144

theorem problem_solution (x y m a b : ℝ) : 
  (∃ k : ℤ, (x - 1 = k^2 * 4)) →
  ((4 * x + y)^(1/3) = 3) →
  (m^2 = y - x) →
  (5 + m = a + b) →
  (∃ n : ℤ, a = n) →
  (0 < b) →
  (b < 1) →
  (m = Real.sqrt 2 ∧ a - (Real.sqrt 2 - b)^2 = 5) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l661_66144


namespace NUMINAMATH_CALUDE_different_set_l661_66181

def set_A : Set ℝ := {x | x = 1}
def set_B : Set ℝ := {x | x^2 = 1}
def set_C : Set ℝ := {1}
def set_D : Set ℝ := {y | (y - 1)^2 = 0}

theorem different_set :
  (set_A = set_C) ∧ (set_A = set_D) ∧ (set_C = set_D) ∧ (set_B ≠ set_A) ∧ (set_B ≠ set_C) ∧ (set_B ≠ set_D) :=
sorry

end NUMINAMATH_CALUDE_different_set_l661_66181


namespace NUMINAMATH_CALUDE_complex_equation_solution_l661_66194

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) / (z - Complex.I) = Complex.I → z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l661_66194


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l661_66119

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, x1 = -1 + Real.sqrt 5 ∧ x2 = -1 - Real.sqrt 5 ∧
    x1^2 + 2*x1 - 4 = 0 ∧ x2^2 + 2*x2 - 4 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = 3 ∧ x2 = -2 ∧
    2*x1 - 6 = x1*(3-x1) ∧ 2*x2 - 6 = x2*(3-x2)) :=
by
  sorry

#check quadratic_equations_solutions

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l661_66119


namespace NUMINAMATH_CALUDE_multiply_sum_power_l661_66118

theorem multiply_sum_power (n : ℕ) (h : n > 0) :
  n * (n^n + 1) = n^(n + 1) + n :=
by sorry

end NUMINAMATH_CALUDE_multiply_sum_power_l661_66118


namespace NUMINAMATH_CALUDE_perfect_square_count_l661_66137

theorem perfect_square_count : 
  ∃! (count : ℕ), ∃ (S : Finset ℕ), 
    (Finset.card S = count) ∧ 
    (∀ n, n ∈ S ↔ ∃ x : ℤ, (4:ℤ)^n - 15 = x^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_count_l661_66137


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l661_66111

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 1 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x | x > 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l661_66111


namespace NUMINAMATH_CALUDE_largest_sum_and_simplification_l661_66127

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/2, 1/3 + 1/9, 1/3 + 1/6]
  (∀ s ∈ sums, s ≤ (1/3 + 1/2)) ∧ (1/3 + 1/2 = 5/6) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_and_simplification_l661_66127


namespace NUMINAMATH_CALUDE_current_price_calculation_l661_66147

/-- The current unit price after price adjustments -/
def current_price (x : ℝ) : ℝ := (1 - 0.25) * (x + 10)

/-- Theorem stating that the current price calculation is correct -/
theorem current_price_calculation (x : ℝ) : 
  current_price x = (1 - 0.25) * (x + 10) := by
  sorry

end NUMINAMATH_CALUDE_current_price_calculation_l661_66147


namespace NUMINAMATH_CALUDE_jalapeno_slices_per_pepper_l661_66134

/-- The number of jalapeno strips required per sandwich -/
def strips_per_sandwich : ℕ := 4

/-- The time in minutes between serving each sandwich -/
def minutes_per_sandwich : ℕ := 5

/-- The number of hours the shop operates per day -/
def operating_hours : ℕ := 8

/-- The number of jalapeno peppers required for a full day of operation -/
def peppers_per_day : ℕ := 48

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem jalapeno_slices_per_pepper : 
  (operating_hours * minutes_per_hour / minutes_per_sandwich) * strips_per_sandwich / peppers_per_day = 8 := by
  sorry

end NUMINAMATH_CALUDE_jalapeno_slices_per_pepper_l661_66134


namespace NUMINAMATH_CALUDE_angle_of_inclination_sqrt3_l661_66109

theorem angle_of_inclination_sqrt3 :
  let line : ℝ → ℝ := λ x ↦ Real.sqrt 3 * x - 2
  let slope : ℝ := Real.sqrt 3
  let angle_of_inclination : ℝ := Real.arctan slope
  angle_of_inclination = π / 3 := by sorry

end NUMINAMATH_CALUDE_angle_of_inclination_sqrt3_l661_66109


namespace NUMINAMATH_CALUDE_inscribed_squares_segment_product_l661_66158

theorem inscribed_squares_segment_product :
  ∀ (a b : ℝ),
    (∃ (inner_area outer_area : ℝ),
      inner_area = 16 ∧
      outer_area = 18 ∧
      (∃ (inner_side outer_side : ℝ),
        inner_side^2 = inner_area ∧
        outer_side^2 = outer_area ∧
        a + b = outer_side ∧
        (a^2 + b^2) = inner_side^2)) →
    a * b = -7 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_segment_product_l661_66158


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l661_66173

/-- A line passing through two points (2, 9) and (4, 13) intersects the y-axis at (0, 5) -/
theorem line_intersection_y_axis :
  ∀ (f : ℝ → ℝ),
  (f 2 = 9) →
  (f 4 = 13) →
  (∀ x y, f x = y ↔ y = 2*x + 5) →
  f 0 = 5 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l661_66173


namespace NUMINAMATH_CALUDE_three_good_pairs_l661_66126

-- Define a structure for a line in slope-intercept form
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the lines
def L1 : Line := { slope := 2, intercept := 3 }
def L2 : Line := { slope := 2, intercept := 3 }
def L3 : Line := { slope := 4, intercept := -2 }
def L4 : Line := { slope := -4, intercept := 3 }
def L5 : Line := { slope := -4, intercept := 3 }

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

-- Define what it means for two lines to be perpendicular
def perpendicular (l1 l2 : Line) : Prop := l1.slope * l2.slope = -1

-- Define what it means for two lines to be "good"
def good (l1 l2 : Line) : Prop := parallel l1 l2 ∨ perpendicular l1 l2

-- The main theorem
theorem three_good_pairs :
  ∃ (pairs : List (Line × Line)),
    pairs.length = 3 ∧
    (∀ p ∈ pairs, good p.1 p.2) ∧
    (∀ l1 l2 : Line, l1 ≠ l2 → good l1 l2 → (l1, l2) ∈ pairs ∨ (l2, l1) ∈ pairs) :=
by
  sorry

end NUMINAMATH_CALUDE_three_good_pairs_l661_66126


namespace NUMINAMATH_CALUDE_largest_repeated_product_365_l661_66121

def is_eight_digit_repeated (n : ℕ) : Prop :=
  100000000 > n ∧ n ≥ 10000000 ∧ 
  ∃ (a b c d : ℕ), n = a * 10000000 + b * 1000000 + c * 100000 + d * 10000 + 
                    a * 1000 + b * 100 + c * 10 + d

theorem largest_repeated_product_365 : 
  (∀ m : ℕ, m > 273863 → ¬(is_eight_digit_repeated (m * 365))) ∧ 
  is_eight_digit_repeated (273863 * 365) := by
sorry

#eval 273863 * 365  -- Should output 99959995

end NUMINAMATH_CALUDE_largest_repeated_product_365_l661_66121


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_six_l661_66103

theorem largest_four_digit_divisible_by_six :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 6 = 0 → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_six_l661_66103


namespace NUMINAMATH_CALUDE_min_value_of_function_l661_66107

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  x + 2 / (2 * x + 1) - 3 / 2 ≥ 0 ∧
  ∃ y > 0, y + 2 / (2 * y + 1) - 3 / 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l661_66107


namespace NUMINAMATH_CALUDE_union_of_sets_l661_66122

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {2, 3}
  A ∪ B = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l661_66122


namespace NUMINAMATH_CALUDE_vector_collinearity_l661_66123

/-- Given vectors in ℝ², prove that if 3a + b is collinear with c, then x = -4 -/
theorem vector_collinearity (a b c : ℝ × ℝ) (x : ℝ) :
  a = (-2, 0) →
  b = (2, 1) →
  c = (x, 1) →
  ∃ (k : ℝ), k • (3 • a + b) = c →
  x = -4 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l661_66123


namespace NUMINAMATH_CALUDE_seventh_root_of_unity_product_l661_66189

theorem seventh_root_of_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^5 - 1) * (r^6 - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_of_unity_product_l661_66189


namespace NUMINAMATH_CALUDE_girls_in_class_l661_66114

theorem girls_in_class (total : ℕ) (prob : ℚ) (boys : ℕ) (girls : ℕ) : 
  total = 25 →
  prob = 3 / 25 →
  boys + girls = total →
  (boys.choose 2 : ℚ) / (total.choose 2 : ℚ) = prob →
  girls = 16 :=
sorry

end NUMINAMATH_CALUDE_girls_in_class_l661_66114


namespace NUMINAMATH_CALUDE_unique_solution_for_diophantine_equation_l661_66150

theorem unique_solution_for_diophantine_equation :
  ∃! (a b : ℕ), 
    Nat.Prime a ∧ 
    b > 0 ∧ 
    9 * (2 * a + b)^2 = 509 * (4 * a + 511 * b) ∧
    a = 251 ∧ 
    b = 7 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_diophantine_equation_l661_66150


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l661_66196

theorem fraction_equality_implies_numerator_equality 
  (a b c : ℝ) (h : c ≠ 0) : a / c = b / c → a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l661_66196


namespace NUMINAMATH_CALUDE_face_mask_profit_l661_66184

/-- Calculates the total profit from selling face masks given specific conditions --/
theorem face_mask_profit : 
  let original_price : ℝ := 10
  let discount1 : ℝ := 0.2
  let discount2 : ℝ := 0.3
  let discount3 : ℝ := 0.4
  let packs1 : ℕ := 20
  let packs2 : ℕ := 30
  let packs3 : ℕ := 40
  let masks_per_pack : ℕ := 5
  let sell_price1 : ℝ := 0.75
  let sell_price2 : ℝ := 0.85
  let sell_price3 : ℝ := 0.95

  let cost1 : ℝ := original_price * (1 - discount1)
  let cost2 : ℝ := original_price * (1 - discount2)
  let cost3 : ℝ := original_price * (1 - discount3)

  let total_cost : ℝ := cost1 + cost2 + cost3

  let revenue1 : ℝ := (packs1 * masks_per_pack : ℝ) * sell_price1
  let revenue2 : ℝ := (packs2 * masks_per_pack : ℝ) * sell_price2
  let revenue3 : ℝ := (packs3 * masks_per_pack : ℝ) * sell_price3

  let total_revenue : ℝ := revenue1 + revenue2 + revenue3

  let total_profit : ℝ := total_revenue - total_cost

  total_profit = 371.5 := by sorry

end NUMINAMATH_CALUDE_face_mask_profit_l661_66184


namespace NUMINAMATH_CALUDE_index_cards_per_student_l661_66193

theorem index_cards_per_student 
  (num_classes : ℕ) 
  (students_per_class : ℕ) 
  (total_packs : ℕ) 
  (h1 : num_classes = 6) 
  (h2 : students_per_class = 30) 
  (h3 : total_packs = 360) : 
  total_packs / (num_classes * students_per_class) = 2 := by
  sorry

end NUMINAMATH_CALUDE_index_cards_per_student_l661_66193


namespace NUMINAMATH_CALUDE_angles_between_plane_and_legs_l661_66140

/-- Given a right triangle with an acute angle α and a plane through the smallest median
    forming an angle β with the triangle's plane, this theorem states the angles between
    the plane and the legs of the triangle. -/
theorem angles_between_plane_and_legs (α β : Real) 
  (h_acute : 0 < α ∧ α < Real.pi / 2)
  (h_right_triangle : True)  -- Placeholder for the right triangle condition
  (h_smallest_median : True) -- Placeholder for the smallest median condition
  (h_plane_angle : True)     -- Placeholder for the plane angle condition
  : ∃ (γ θ : Real),
    γ = Real.arcsin (Real.sin β * Real.cos α) ∧
    θ = Real.arcsin (Real.sin β * Real.sin α) :=
by sorry

end NUMINAMATH_CALUDE_angles_between_plane_and_legs_l661_66140


namespace NUMINAMATH_CALUDE_number_with_quotient_and_remainder_l661_66177

theorem number_with_quotient_and_remainder (x : ℕ) : 
  (x / 7 = 4) ∧ (x % 7 = 6) → x = 34 := by
  sorry

end NUMINAMATH_CALUDE_number_with_quotient_and_remainder_l661_66177


namespace NUMINAMATH_CALUDE_divisibility_conditions_l661_66182

theorem divisibility_conditions (n : ℕ) (hn : n ≥ 1) :
  (n ∣ 2^n - 1 ↔ n = 1) ∧
  (n % 2 = 1 ∧ n ∣ 3^n + 1 ↔ n = 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_conditions_l661_66182


namespace NUMINAMATH_CALUDE_coprime_elements_bound_l661_66146

/-- The number of elements in [1, n] coprime to M -/
def h (M n : ℕ+) : ℕ := sorry

/-- The proportion of numbers in [1, M] coprime to M -/
def β (M : ℕ+) : ℚ := (h M M : ℚ) / M

/-- ω(M) is the number of distinct prime factors of M -/
def ω (M : ℕ+) : ℕ := sorry

theorem coprime_elements_bound (M : ℕ+) :
  ∃ S : Finset ℕ+,
    S.card ≥ M / 3 ∧
    ∀ n ∈ S, n ≤ M ∧
    |h M n - β M * n| ≤ Real.sqrt (β M * 2^(ω M - 3)) + 1 :=
  sorry

end NUMINAMATH_CALUDE_coprime_elements_bound_l661_66146
