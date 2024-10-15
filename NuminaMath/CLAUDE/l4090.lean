import Mathlib

namespace NUMINAMATH_CALUDE_units_digit_of_power_l4090_409042

theorem units_digit_of_power (n : ℕ) : (147 ^ 25) ^ 50 ≡ 9 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_l4090_409042


namespace NUMINAMATH_CALUDE_olivias_house_height_l4090_409003

/-- The height of Olivia's house in feet -/
def house_height : ℕ := 81

/-- The length of the shadow cast by Olivia's house in feet -/
def house_shadow : ℕ := 70

/-- The height of the flagpole in feet -/
def flagpole_height : ℕ := 35

/-- The length of the shadow cast by the flagpole in feet -/
def flagpole_shadow : ℕ := 30

/-- The height of the bush in feet -/
def bush_height : ℕ := 14

/-- The length of the shadow cast by the bush in feet -/
def bush_shadow : ℕ := 12

theorem olivias_house_height :
  (house_height : ℚ) / house_shadow = flagpole_height / flagpole_shadow ∧
  (house_height : ℚ) / house_shadow = bush_height / bush_shadow ∧
  house_height = 81 :=
sorry

end NUMINAMATH_CALUDE_olivias_house_height_l4090_409003


namespace NUMINAMATH_CALUDE_equation_holds_l4090_409016

theorem equation_holds : Real.sqrt (2 + Real.sqrt (3 + Real.sqrt 0)) = (2 + Real.sqrt 0) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l4090_409016


namespace NUMINAMATH_CALUDE_system_solution_l4090_409000

theorem system_solution (a b : ℝ) (h : a ≠ b) :
  ∃! (x y : ℝ), (a + 1) * x + (a - 1) * y = a ∧ (b + 1) * x + (b - 1) * y = b ∧ x = (1 : ℝ) / 2 ∧ y = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4090_409000


namespace NUMINAMATH_CALUDE_angle_triple_complement_l4090_409098

theorem angle_triple_complement (x : ℝ) : x = 3 * (90 - x) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l4090_409098


namespace NUMINAMATH_CALUDE_baseball_team_size_l4090_409085

/-- Calculates the number of players on a team given the total points, 
    points scored by one player, and points scored by each other player -/
def team_size (total_points : ℕ) (one_player_points : ℕ) (other_player_points : ℕ) : ℕ :=
  (total_points - one_player_points) / other_player_points + 1

/-- Theorem stating that for the given conditions, the team size is 6 -/
theorem baseball_team_size : 
  team_size 68 28 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_size_l4090_409085


namespace NUMINAMATH_CALUDE_sin_equation_solutions_l4090_409071

/-- The number of solutions to 2sin³x - 5sin²x + 2sinx = 0 in [0, 2π] is 5 -/
theorem sin_equation_solutions : 
  let f : ℝ → ℝ := λ x => 2 * Real.sin x ^ 3 - 5 * Real.sin x ^ 2 + 2 * Real.sin x
  ∃! (s : Finset ℝ), s.card = 5 ∧ 
    (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f x = 0) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f x = 0 → x ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_sin_equation_solutions_l4090_409071


namespace NUMINAMATH_CALUDE_vector_dot_product_equation_l4090_409059

/-- Given vectors a, b, and c satisfying certain conditions, prove that x = 4 -/
theorem vector_dot_product_equation (a b c : ℝ × ℝ) (x : ℝ) 
  (ha : a = (1, 1))
  (hb : b = (2, 5))
  (hc : c = (3, x))
  (h_dot : ((8 • a - b) • c) = 30) :
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_vector_dot_product_equation_l4090_409059


namespace NUMINAMATH_CALUDE_fraction_product_cubed_simplify_fraction_cube_l4090_409050

theorem fraction_product_cubed (a b c d : ℚ) :
  (a / b) ^ 3 * (c / d) ^ 3 = ((a * c) / (b * d)) ^ 3 :=
by sorry

theorem simplify_fraction_cube :
  (5 / 8) ^ 3 * (2 / 3) ^ 3 = 125 / 1728 :=
by sorry

end NUMINAMATH_CALUDE_fraction_product_cubed_simplify_fraction_cube_l4090_409050


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l4090_409099

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l4090_409099


namespace NUMINAMATH_CALUDE_exists_number_with_property_l4090_409096

-- Define small numbers
def isSmall (n : ℕ) : Prop := n ≤ 150

-- Define the property we're looking for
def hasProperty (N : ℕ) : Prop :=
  ∃ (a b : ℕ), isSmall a ∧ isSmall b ∧ b = a + 1 ∧
  ¬(N % a = 0) ∧ ¬(N % b = 0) ∧
  ∀ k, isSmall k → k ≠ a → k ≠ b → N % k = 0

-- Theorem statement
theorem exists_number_with_property :
  ∃ N : ℕ, hasProperty N :=
sorry

end NUMINAMATH_CALUDE_exists_number_with_property_l4090_409096


namespace NUMINAMATH_CALUDE_b_2017_equals_1_l4090_409054

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def b (n : ℕ) : ℕ := fibonacci n % 3

theorem b_2017_equals_1 : b 2017 = 1 := by sorry

end NUMINAMATH_CALUDE_b_2017_equals_1_l4090_409054


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l4090_409048

theorem unique_solution_exponential_equation :
  ∃! (x y z t : ℕ+), 12^(x:ℕ) + 13^(y:ℕ) - 14^(z:ℕ) = 2013^(t:ℕ) ∧ 
    x = 1 ∧ y = 3 ∧ z = 2 ∧ t = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l4090_409048


namespace NUMINAMATH_CALUDE_power_negative_one_equals_half_l4090_409072

-- Define the theorem
theorem power_negative_one_equals_half : 2^(-1 : ℤ) = (1/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_power_negative_one_equals_half_l4090_409072


namespace NUMINAMATH_CALUDE_melanie_balloons_l4090_409038

def joan_balloons : ℕ := 40
def total_balloons : ℕ := 81

theorem melanie_balloons : total_balloons - joan_balloons = 41 := by
  sorry

end NUMINAMATH_CALUDE_melanie_balloons_l4090_409038


namespace NUMINAMATH_CALUDE_fort_blocks_count_l4090_409047

/-- Represents the dimensions of a rectangular fort --/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed for a fort with given dimensions and specifications --/
def calculateFortBlocks (d : FortDimensions) : ℕ :=
  let totalVolume := d.length * d.width * d.height
  let internalLength := d.length - 2
  let internalWidth := d.width - 2
  let internalHeight := d.height - 1
  let internalVolume := internalLength * internalWidth * internalHeight
  let partitionVolume := 1 * internalWidth * internalHeight
  totalVolume - internalVolume + partitionVolume

/-- Theorem stating that a fort with the given dimensions requires 458 blocks --/
theorem fort_blocks_count :
  let fortDims : FortDimensions := ⟨14, 12, 6⟩
  calculateFortBlocks fortDims = 458 := by
  sorry

#eval calculateFortBlocks ⟨14, 12, 6⟩

end NUMINAMATH_CALUDE_fort_blocks_count_l4090_409047


namespace NUMINAMATH_CALUDE_complex_multiplication_l4090_409011

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l4090_409011


namespace NUMINAMATH_CALUDE_inequality_proof_l4090_409063

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  64 * (a * b * c * d + 1) / (a + b + c + d)^2 ≤ 
  a^2 + b^2 + c^2 + d^2 + 1/a^2 + 1/b^2 + 1/c^2 + 1/d^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4090_409063


namespace NUMINAMATH_CALUDE_min_gumballs_for_four_is_ten_l4090_409019

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat

/-- The minimum number of gumballs needed to guarantee 4 of the same color -/
def minGumballsForFour (machine : GumballMachine) : Nat :=
  10

/-- Theorem stating that for a machine with 9 red, 7 white, and 8 blue gumballs,
    the minimum number of gumballs needed to guarantee 4 of the same color is 10 -/
theorem min_gumballs_for_four_is_ten (machine : GumballMachine)
    (h_red : machine.red = 9)
    (h_white : machine.white = 7)
    (h_blue : machine.blue = 8) :
    minGumballsForFour machine = 10 := by
  sorry


end NUMINAMATH_CALUDE_min_gumballs_for_four_is_ten_l4090_409019


namespace NUMINAMATH_CALUDE_train_distance_theorem_l4090_409017

/-- The distance between two trains traveling in opposite directions -/
def distance_between_trains (speed_a speed_b : ℝ) (time_a time_b : ℝ) : ℝ :=
  speed_a * time_a + speed_b * time_b

/-- Theorem: The distance between the trains is 1284 miles -/
theorem train_distance_theorem :
  distance_between_trains 56 23 18 12 = 1284 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_theorem_l4090_409017


namespace NUMINAMATH_CALUDE_quadratic_roots_l4090_409070

theorem quadratic_roots (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let f (x : ℝ) := 3*a*x^2 + 2*(a + b)*x + (b + c)
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l4090_409070


namespace NUMINAMATH_CALUDE_area_of_triangle_MOI_l4090_409078

/-- Triangle ABC with given side lengths --/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (ab_length : dist A B = 15)
  (ac_length : dist A C = 8)
  (bc_length : dist B C = 7)

/-- Circumcenter of a triangle --/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Incenter of a triangle --/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- Point M: center of circle tangent to sides AC, BC, and circumcircle --/
def point_M (t : Triangle) : ℝ × ℝ := sorry

/-- Area of a triangle given three points --/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem --/
theorem area_of_triangle_MOI (t : Triangle) :
  triangle_area (circumcenter t) (incenter t) (point_M t) = 7/4 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_MOI_l4090_409078


namespace NUMINAMATH_CALUDE_M_equals_N_l4090_409067

/-- The set M of integers of the form 12m + 8n + 4l where m, n, l are integers -/
def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}

/-- The set N of integers of the form 20p + 16q + 12r where p, q, r are integers -/
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

/-- Theorem stating that M equals N -/
theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l4090_409067


namespace NUMINAMATH_CALUDE_equation_solution_l4090_409033

theorem equation_solution (k : ℝ) : 
  (∃ x : ℝ, x = -5 ∧ (1 : ℝ) / 2023 * x - 2 = 3 * x + k) →
  (∃ y : ℝ, y = -3 ∧ (1 : ℝ) / 2023 * (2 * y + 1) - 5 = 6 * y + k) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4090_409033


namespace NUMINAMATH_CALUDE_smallest_sum_with_factors_and_perfect_square_l4090_409082

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_sum_with_factors_and_perfect_square :
  ∃ (a b : ℕ+),
    num_factors a = 15 ∧
    num_factors b = 20 ∧
    is_perfect_square (a.val + b.val) ∧
    ∀ (c d : ℕ+),
      num_factors c = 15 →
      num_factors d = 20 →
      is_perfect_square (c.val + d.val) →
      a.val + b.val ≤ c.val + d.val ∧
      a.val + b.val = 576 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_with_factors_and_perfect_square_l4090_409082


namespace NUMINAMATH_CALUDE_clara_triple_anna_age_l4090_409035

def anna_current_age : ℕ := 54
def clara_current_age : ℕ := 80

theorem clara_triple_anna_age :
  ∃ (years_ago : ℕ), 
    clara_current_age - years_ago = 3 * (anna_current_age - years_ago) ∧
    years_ago = 41 :=
by sorry

end NUMINAMATH_CALUDE_clara_triple_anna_age_l4090_409035


namespace NUMINAMATH_CALUDE_det_specific_matrix_l4090_409022

theorem det_specific_matrix (x : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![x + 2, x + 1, x; x, x + 2, x + 1; x + 1, x, x + 2]
  Matrix.det A = x^2 + 11*x + 9 := by
sorry

end NUMINAMATH_CALUDE_det_specific_matrix_l4090_409022


namespace NUMINAMATH_CALUDE_amusement_park_admission_l4090_409051

theorem amusement_park_admission (child_fee : ℚ) (adult_fee : ℚ) (total_fee : ℚ) (num_children : ℕ) :
  child_fee = 3/2 →
  adult_fee = 4 →
  total_fee = 810 →
  num_children = 180 →
  ∃ (num_adults : ℕ), 
    (child_fee * num_children + adult_fee * num_adults = total_fee) ∧
    (num_children + num_adults = 315) :=
by
  sorry

end NUMINAMATH_CALUDE_amusement_park_admission_l4090_409051


namespace NUMINAMATH_CALUDE_compound_statement_false_l4090_409014

theorem compound_statement_false (p q : Prop) :
  ¬(p ∧ q) → (¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_compound_statement_false_l4090_409014


namespace NUMINAMATH_CALUDE_sample_size_calculation_l4090_409055

theorem sample_size_calculation (num_classes : ℕ) (papers_per_class : ℕ) : 
  num_classes = 8 → papers_per_class = 12 → num_classes * papers_per_class = 96 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l4090_409055


namespace NUMINAMATH_CALUDE_calculation_proof_system_of_equations_proof_l4090_409060

-- Problem 1
theorem calculation_proof : (-1)^2023 + Real.sqrt 9 - Real.pi^0 + Real.sqrt (1/8) * Real.sqrt 32 = 3 := by
  sorry

-- Problem 2
theorem system_of_equations_proof :
  ∃ (x y : ℝ), 2*x - y = 5 ∧ 3*x + 2*y = 11 ∧ x = 3 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_system_of_equations_proof_l4090_409060


namespace NUMINAMATH_CALUDE_line_slope_l4090_409091

/-- Given a line with equation y - 3 = 4(x + 1), its slope is 4 -/
theorem line_slope (x y : ℝ) : y - 3 = 4 * (x + 1) → (y - 3) / (x - (-1)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l4090_409091


namespace NUMINAMATH_CALUDE_oak_trees_remaining_l4090_409076

/-- The number of oak trees remaining after cutting down damaged trees -/
def remaining_oak_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that the number of oak trees remaining is 7 -/
theorem oak_trees_remaining :
  remaining_oak_trees 9 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_remaining_l4090_409076


namespace NUMINAMATH_CALUDE_trailing_zeros_15_factorial_base_15_l4090_409043

/-- The number of trailing zeros in n! in base b --/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ := sorry

/-- The factorial of a natural number --/
def factorial (n : ℕ) : ℕ := sorry

/-- 15 factorial --/
def factorial15 : ℕ := factorial 15

theorem trailing_zeros_15_factorial_base_15 :
  trailingZeros factorial15 15 = 3 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_15_factorial_base_15_l4090_409043


namespace NUMINAMATH_CALUDE_g_13_equals_205_l4090_409058

def g (n : ℕ) : ℕ := n^2 + n + 23

theorem g_13_equals_205 : g 13 = 205 := by
  sorry

end NUMINAMATH_CALUDE_g_13_equals_205_l4090_409058


namespace NUMINAMATH_CALUDE_complex_magnitude_sum_reciprocals_l4090_409005

theorem complex_magnitude_sum_reciprocals (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_sum_reciprocals_l4090_409005


namespace NUMINAMATH_CALUDE_count_auspicious_dragon_cards_l4090_409020

/-- The number of ways to select 4 digits from 0 to 9 and arrange them in ascending order -/
def auspicious_dragon_cards : ℕ := sorry

/-- Theorem stating that the number of Auspicious Dragon Cards is 210 -/
theorem count_auspicious_dragon_cards : auspicious_dragon_cards = 210 := by sorry

end NUMINAMATH_CALUDE_count_auspicious_dragon_cards_l4090_409020


namespace NUMINAMATH_CALUDE_smallest_abs_z_l4090_409007

theorem smallest_abs_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z + 6 * Complex.I) = 17) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs (w - 8) + Complex.abs (w + 6 * Complex.I) = 17 ∧ Complex.abs w = 48 / 17 :=
sorry

end NUMINAMATH_CALUDE_smallest_abs_z_l4090_409007


namespace NUMINAMATH_CALUDE_ticket_ratio_l4090_409013

/-- Prove the ratio of Peyton's tickets to Tate's total tickets -/
theorem ticket_ratio :
  let tate_initial : ℕ := 32
  let tate_bought : ℕ := 2
  let total_tickets : ℕ := 51
  let tate_total : ℕ := tate_initial + tate_bought
  let peyton_tickets : ℕ := total_tickets - tate_total
  (peyton_tickets : ℚ) / tate_total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ticket_ratio_l4090_409013


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l4090_409087

/-- Represents the number of complete books that can be read given the reading speed, book length, and available time. -/
def booksRead (readingSpeed : ℕ) (bookLength : ℕ) (availableTime : ℕ) : ℕ :=
  (readingSpeed * availableTime) / bookLength

/-- Theorem stating that Robert can read 2 complete 360-page books in 8 hours at a speed of 120 pages per hour. -/
theorem robert_reading_capacity :
  booksRead 120 360 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_reading_capacity_l4090_409087


namespace NUMINAMATH_CALUDE_percentage_problem_l4090_409090

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.1 * 500 - 5 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l4090_409090


namespace NUMINAMATH_CALUDE_grace_earnings_l4090_409046

/-- Calculates the number of weeks needed to earn a target amount given a weekly rate and biweekly payment schedule. -/
def weeksToEarn (weeklyRate : ℕ) (targetAmount : ℕ) : ℕ :=
  let biweeklyEarnings := weeklyRate * 2
  let numPayments := targetAmount / biweeklyEarnings
  numPayments * 2

/-- Proves that it takes 6 weeks to earn 1800 dollars with a weekly rate of 300 dollars and biweekly payments. -/
theorem grace_earnings : weeksToEarn 300 1800 = 6 := by
  sorry

end NUMINAMATH_CALUDE_grace_earnings_l4090_409046


namespace NUMINAMATH_CALUDE_distance_maximum_at_halfway_l4090_409021

-- Define a square in 2D space
structure Square :=
  (side : ℝ)
  (center : ℝ × ℝ)

-- Define a runner's position on the square
structure RunnerPosition :=
  (square : Square)
  (t : ℝ)  -- Parameter representing time or position along the path (0 ≤ t ≤ 4)

-- Function to calculate the runner's coordinates
def runnerCoordinates (pos : RunnerPosition) : ℝ × ℝ :=
  sorry

-- Function to calculate the straight-line distance from the starting point
def distanceFromStart (pos : RunnerPosition) : ℝ :=
  sorry

theorem distance_maximum_at_halfway (s : Square) :
  ∃ (t_max : ℝ), t_max = 2 ∧
  ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 4 →
    distanceFromStart ⟨s, t⟩ ≤ distanceFromStart ⟨s, t_max⟩ :=
sorry

end NUMINAMATH_CALUDE_distance_maximum_at_halfway_l4090_409021


namespace NUMINAMATH_CALUDE_optimal_transport_plan_l4090_409052

/-- Represents the transportation problem for fruits A, B, and C -/
structure FruitTransport where
  total_trucks : ℕ
  total_tons : ℕ
  tons_per_truck_A : ℕ
  tons_per_truck_B : ℕ
  tons_per_truck_C : ℕ
  profit_per_ton_A : ℕ
  profit_per_ton_B : ℕ
  profit_per_ton_C : ℕ

/-- Calculates the profit for a given transportation plan -/
def calculate_profit (ft : FruitTransport) (trucks_A trucks_B trucks_C : ℕ) : ℕ :=
  trucks_A * ft.tons_per_truck_A * ft.profit_per_ton_A +
  trucks_B * ft.tons_per_truck_B * ft.profit_per_ton_B +
  trucks_C * ft.tons_per_truck_C * ft.profit_per_ton_C

/-- The main theorem stating the optimal transportation plan and maximum profit -/
theorem optimal_transport_plan (ft : FruitTransport)
  (h1 : ft.total_trucks = 20)
  (h2 : ft.total_tons = 100)
  (h3 : ft.tons_per_truck_A = 6)
  (h4 : ft.tons_per_truck_B = 5)
  (h5 : ft.tons_per_truck_C = 4)
  (h6 : ft.profit_per_ton_A = 500)
  (h7 : ft.profit_per_ton_B = 600)
  (h8 : ft.profit_per_ton_C = 400) :
  ∃ (trucks_A trucks_B trucks_C : ℕ),
    trucks_A + trucks_B + trucks_C = ft.total_trucks ∧
    trucks_A * ft.tons_per_truck_A + trucks_B * ft.tons_per_truck_B + trucks_C * ft.tons_per_truck_C = ft.total_tons ∧
    trucks_A ≥ 2 ∧ trucks_B ≥ 2 ∧ trucks_C ≥ 2 ∧
    trucks_A = 2 ∧ trucks_B = 16 ∧ trucks_C = 2 ∧
    calculate_profit ft trucks_A trucks_B trucks_C = 57200 ∧
    ∀ (a b c : ℕ), a + b + c = ft.total_trucks →
      a * ft.tons_per_truck_A + b * ft.tons_per_truck_B + c * ft.tons_per_truck_C = ft.total_tons →
      a ≥ 2 → b ≥ 2 → c ≥ 2 →
      calculate_profit ft a b c ≤ calculate_profit ft trucks_A trucks_B trucks_C :=
by sorry

end NUMINAMATH_CALUDE_optimal_transport_plan_l4090_409052


namespace NUMINAMATH_CALUDE_orange_bin_count_l4090_409018

theorem orange_bin_count (initial : ℕ) (removed : ℕ) (added : ℕ) : 
  initial = 40 → removed = 37 → added = 7 → initial - removed + added = 10 := by
  sorry

end NUMINAMATH_CALUDE_orange_bin_count_l4090_409018


namespace NUMINAMATH_CALUDE_functional_equation_solution_l4090_409056

/-- A continuous function satisfying the given functional equation is either constantly 0 or 1/2. -/
theorem functional_equation_solution (f : ℝ → ℝ) (hf : Continuous f)
  (h : ∀ x y : ℝ, f (x^2 - y^2) = f x^2 + f y^2) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l4090_409056


namespace NUMINAMATH_CALUDE_only_zero_solution_l4090_409015

theorem only_zero_solution (m n : ℤ) (h : 231 * m^2 = 130 * n^2) : m = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_only_zero_solution_l4090_409015


namespace NUMINAMATH_CALUDE_sin_negative_nine_half_pi_l4090_409032

theorem sin_negative_nine_half_pi : Real.sin (-9 * Real.pi / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_nine_half_pi_l4090_409032


namespace NUMINAMATH_CALUDE_problem_statement_l4090_409039

theorem problem_statement (a b c : ℕ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a ≥ b ∧ b ≥ c) (h3 : Nat.Prime ((a - c) / 2))
  (h4 : a^2 + b^2 + c^2 - 2*(a*b + b*c + c*a) = b) :
  Nat.Prime b ∨ ∃ k : ℕ, b = k^2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l4090_409039


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l4090_409074

theorem min_value_fraction_sum (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  x^2 / (y^2 - 1) + y^2 / (x^2 - 1) ≥ 4 ∧
  (x^2 / (y^2 - 1) + y^2 / (x^2 - 1) = 4 ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l4090_409074


namespace NUMINAMATH_CALUDE_union_of_sets_l4090_409077

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 3, 4}
  A ∪ B = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l4090_409077


namespace NUMINAMATH_CALUDE_expression_evaluation_l4090_409010

theorem expression_evaluation (a b : ℤ) (ha : a = -4) (hb : b = 3) :
  -2 * a - b^3 + 2 * a * b + b^2 = -34 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4090_409010


namespace NUMINAMATH_CALUDE_max_ratio_on_circle_l4090_409089

/-- A point with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- Definition of the circle x^2 + y^2 = 25 -/
def on_circle (p : IntPoint) : Prop :=
  p.x^2 + p.y^2 = 25

/-- Definition of irrational distance between two points -/
def irrational_distance (p q : IntPoint) : Prop :=
  ∃ (d : ℝ), d^2 = (p.x - q.x)^2 + (p.y - q.y)^2 ∧ Irrational d

/-- Theorem statement -/
theorem max_ratio_on_circle (P Q R S : IntPoint) :
  on_circle P → on_circle Q → on_circle R → on_circle S →
  irrational_distance P Q → irrational_distance R S →
  ∃ (ratio : ℝ), (∀ (d_PQ d_RS : ℝ),
    d_PQ^2 = (P.x - Q.x)^2 + (P.y - Q.y)^2 →
    d_RS^2 = (R.x - S.x)^2 + (R.y - S.y)^2 →
    d_PQ / d_RS ≤ ratio) ∧
  ratio = 5 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_ratio_on_circle_l4090_409089


namespace NUMINAMATH_CALUDE_difference_of_percentages_l4090_409081

theorem difference_of_percentages : (0.7 * 40) - ((4 / 5) * 25) = 8 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_percentages_l4090_409081


namespace NUMINAMATH_CALUDE_min_value_problem_l4090_409080

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 * x' + y' = 1 → 1 / x' + 4 / y' ≥ 1 / x + 4 / y) →
  1 / x + 4 / y = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l4090_409080


namespace NUMINAMATH_CALUDE_conditional_prob_B_given_A_l4090_409025

/-- The number of class officers -/
def total_officers : ℕ := 6

/-- The number of boys among the class officers -/
def num_boys : ℕ := 4

/-- The number of girls among the class officers -/
def num_girls : ℕ := 2

/-- The number of students selected -/
def num_selected : ℕ := 3

/-- Event A: "boy A being selected" -/
def event_A : Set (Fin total_officers) := sorry

/-- Event B: "girl B being selected" -/
def event_B : Set (Fin total_officers) := sorry

/-- The probability of event A -/
def prob_A : ℚ := 1 / 2

/-- The probability of both events A and B occurring -/
def prob_AB : ℚ := 1 / 5

/-- Theorem: The conditional probability P(B|A) is 2/5 -/
theorem conditional_prob_B_given_A : 
  (prob_AB / prob_A : ℚ) = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_conditional_prob_B_given_A_l4090_409025


namespace NUMINAMATH_CALUDE_g_2023_of_2_eq_2_l4090_409075

def g (x : ℚ) : ℚ := (2 - x) / (2 * x + 1)

def g_n : ℕ → ℚ → ℚ
  | 0, x => x
  | 1, x => g x
  | (n + 2), x => g (g_n (n + 1) x)

theorem g_2023_of_2_eq_2 : g_n 2023 2 = 2 := by sorry

end NUMINAMATH_CALUDE_g_2023_of_2_eq_2_l4090_409075


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l4090_409030

theorem geometric_series_first_term (a r : ℝ) (h1 : |r| < 1) 
  (h2 : a / (1 - r) = 30) (h3 : a^2 / (1 - r^2) = 90) : a = 60 / 11 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l4090_409030


namespace NUMINAMATH_CALUDE_sheets_per_pack_calculation_l4090_409041

/-- Represents the number of sheets in a pack of notebook paper -/
def sheets_per_pack : ℕ := 100

/-- Represents the number of pages Chip takes per day per class -/
def pages_per_day_per_class : ℕ := 2

/-- Represents the number of days Chip takes notes per week -/
def days_per_week : ℕ := 5

/-- Represents the number of classes Chip has -/
def num_classes : ℕ := 5

/-- Represents the number of weeks Chip has been taking notes -/
def num_weeks : ℕ := 6

/-- Represents the number of packs Chip used -/
def packs_used : ℕ := 3

theorem sheets_per_pack_calculation :
  sheets_per_pack = 
    (pages_per_day_per_class * days_per_week * num_classes * num_weeks) / packs_used :=
by sorry

end NUMINAMATH_CALUDE_sheets_per_pack_calculation_l4090_409041


namespace NUMINAMATH_CALUDE_johns_annual_epipen_cost_l4090_409034

/-- Represents the cost of EpiPens for John over a year -/
def annual_epipen_cost (epipen_cost : ℝ) (insurance_coverage : ℝ) (replacements_per_year : ℕ) : ℝ :=
  replacements_per_year * (1 - insurance_coverage) * epipen_cost

/-- Theorem stating that John's annual cost for EpiPens is $250 -/
theorem johns_annual_epipen_cost :
  annual_epipen_cost 500 0.75 2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_johns_annual_epipen_cost_l4090_409034


namespace NUMINAMATH_CALUDE_restaurant_friends_l4090_409002

theorem restaurant_friends (pre_cooked wings_cooked wings_per_person : ℕ) :
  pre_cooked = 2 →
  wings_cooked = 25 →
  wings_per_person = 3 →
  (pre_cooked + wings_cooked) / wings_per_person = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_friends_l4090_409002


namespace NUMINAMATH_CALUDE_coat_price_reduction_l4090_409094

/-- Given a coat with an original price and a reduction amount, calculate the percent reduction. -/
theorem coat_price_reduction (original_price reduction_amount : ℝ) 
  (h1 : original_price = 500)
  (h2 : reduction_amount = 200) :
  (reduction_amount / original_price) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_reduction_l4090_409094


namespace NUMINAMATH_CALUDE_cycle_not_divisible_by_three_l4090_409057

/-- A graph is a type with an edge relation -/
class Graph (V : Type) :=
  (adj : V → V → Prop)

/-- The degree of a vertex in a graph is the number of adjacent vertices -/
def degree {V : Type} [Graph V] (v : V) : ℕ := sorry

/-- A path in a graph is a list of vertices where each consecutive pair is adjacent -/
def is_path {V : Type} [Graph V] (p : List V) : Prop := sorry

/-- A cycle in a graph is a path where the first and last vertices are the same -/
def is_cycle {V : Type} [Graph V] (c : List V) : Prop := sorry

/-- The length of a path or cycle is the number of edges it contains -/
def length {V : Type} [Graph V] (p : List V) : ℕ := sorry

theorem cycle_not_divisible_by_three 
  {V : Type} [Graph V] 
  (h : ∀ v : V, degree v ≥ 3) : 
  ∃ c : List V, is_cycle c ∧ ¬(length c % 3 = 0) := by sorry

end NUMINAMATH_CALUDE_cycle_not_divisible_by_three_l4090_409057


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l4090_409097

theorem imaginary_part_of_reciprocal (i : ℂ) (h : i^2 = -1) :
  Complex.im (1 / (i - 2)) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l4090_409097


namespace NUMINAMATH_CALUDE_geometric_series_sum_l4090_409068

theorem geometric_series_sum : 
  let a := 2  -- first term
  let r := 3  -- common ratio
  let n := 7  -- number of terms
  a * (r^n - 1) / (r - 1) = 2186 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l4090_409068


namespace NUMINAMATH_CALUDE_meeting_2015_same_as_first_l4090_409006

/-- Represents a point on a line segment --/
structure Point :=
  (position : ℝ)

/-- Represents a person moving on the line segment --/
structure Person :=
  (startPoint : Point)
  (speed : ℝ)
  (startTime : ℝ)

/-- Represents a meeting between two people --/
def Meeting := ℕ → Point

/-- The movement pattern of two people as described in the problem --/
def movementPattern (person1 person2 : Person) : Meeting :=
  sorry

/-- Theorem stating that the 2015th meeting point is the same as the first meeting point --/
theorem meeting_2015_same_as_first 
  (person1 person2 : Person) (pattern : Meeting := movementPattern person1 person2) :
  pattern 2015 = pattern 1 :=
sorry

end NUMINAMATH_CALUDE_meeting_2015_same_as_first_l4090_409006


namespace NUMINAMATH_CALUDE_pyramid_levels_6_l4090_409069

/-- Defines the number of cubes in a pyramid with n levels -/
def pyramid_cubes (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Theorem stating that a pyramid with 6 levels contains 225 cubes -/
theorem pyramid_levels_6 : pyramid_cubes 6 = 225 := by sorry

end NUMINAMATH_CALUDE_pyramid_levels_6_l4090_409069


namespace NUMINAMATH_CALUDE_train_crossing_time_l4090_409065

/-- Proves that a train with given length and speed takes the calculated time to cross a post -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 150 →
  train_speed_kmh = 27 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l4090_409065


namespace NUMINAMATH_CALUDE_construction_company_higher_utility_l4090_409012

/-- Represents the quality of renovation work -/
structure Quality where
  value : ℝ
  nonneg : value ≥ 0

/-- Represents the cost of renovation work -/
structure Cost where
  value : ℝ
  nonneg : value ≥ 0

/-- Represents the amount of available information about the service provider -/
structure Information where
  value : ℝ
  nonneg : value ≥ 0

/-- Represents a renovation service provider -/
structure ServiceProvider where
  quality : Quality
  cost : Cost
  information : Information

/-- Utility function for renovation service -/
def utilityFunction (α β γ : ℝ) (sp : ServiceProvider) : ℝ :=
  α * sp.quality.value + β * sp.information.value - γ * sp.cost.value

/-- Theorem: Under certain conditions, a construction company can provide higher expected utility -/
theorem construction_company_higher_utility 
  (cc : ServiceProvider) -- construction company
  (prc : ServiceProvider) -- private repair crew
  (α β γ : ℝ) -- utility function parameters
  (h_α : α > 0) -- quality is valued positively
  (h_β : β > 0) -- information is valued positively
  (h_γ : γ > 0) -- cost is valued negatively
  (h_quality : cc.quality.value > prc.quality.value) -- company provides higher quality
  (h_info : cc.information.value > prc.information.value) -- company provides more information
  (h_cost : cc.cost.value > prc.cost.value) -- company is more expensive
  : ∃ (α β γ : ℝ), utilityFunction α β γ cc > utilityFunction α β γ prc :=
sorry

end NUMINAMATH_CALUDE_construction_company_higher_utility_l4090_409012


namespace NUMINAMATH_CALUDE_jellybean_probability_l4090_409049

/-- Probability of picking exactly 2 red jellybeans from a bowl -/
theorem jellybean_probability :
  let total_jellybeans : ℕ := 10
  let red_jellybeans : ℕ := 4
  let blue_jellybeans : ℕ := 1
  let white_jellybeans : ℕ := 5
  let picks : ℕ := 3
  
  -- Ensure the total number of jellybeans is correct
  total_jellybeans = red_jellybeans + blue_jellybeans + white_jellybeans →
  
  -- Calculate the probability
  (Nat.choose red_jellybeans 2 * (blue_jellybeans + white_jellybeans)) / 
  Nat.choose total_jellybeans picks = 3 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l4090_409049


namespace NUMINAMATH_CALUDE_common_chord_length_l4090_409083

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 3*x + 4*y - 18 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := 3*x - 4*y + 10 = 0

-- Theorem statement
theorem common_chord_length :
  ∃ (length : ℝ), 
    (∀ (x y : ℝ), circle1 x y ∧ circle2 x y → common_chord x y) ∧
    length = 4 :=
sorry

end NUMINAMATH_CALUDE_common_chord_length_l4090_409083


namespace NUMINAMATH_CALUDE_largest_common_term_less_than_800_l4090_409037

def arithmetic_progression_1 (n : ℕ) : ℤ := 4 + 5 * n
def arithmetic_progression_2 (m : ℕ) : ℤ := 7 + 8 * m

def is_common_term (a : ℤ) : Prop :=
  ∃ n m : ℕ, arithmetic_progression_1 n = a ∧ arithmetic_progression_2 m = a

theorem largest_common_term_less_than_800 :
  ∃ a : ℤ, is_common_term a ∧ a < 800 ∧ ∀ b : ℤ, is_common_term b ∧ b < 800 → b ≤ a :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_less_than_800_l4090_409037


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l4090_409092

/-- The perimeter of a semicircle with radius 12 is approximately 61.7 units. -/
theorem semicircle_perimeter_approx :
  let r : ℝ := 12
  let π_approx : ℝ := 3.14159
  let semicircle_perimeter := π_approx * r + 2 * r
  ∃ ε > 0, abs (semicircle_perimeter - 61.7) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l4090_409092


namespace NUMINAMATH_CALUDE_mean_of_five_integers_l4090_409044

theorem mean_of_five_integers (p q r s t : ℤ) 
  (h1 : (p + q + r) / 3 = 9)
  (h2 : (s + t) / 2 = 14) :
  (p + q + r + s + t) / 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_five_integers_l4090_409044


namespace NUMINAMATH_CALUDE_intersection_point_k_l4090_409064

-- Define the three lines
def line1 (x y : ℚ) : Prop := y = 4 * x - 1
def line2 (x y : ℚ) : Prop := y = -1/3 * x + 11
def line3 (x y k : ℚ) : Prop := y = 2 * x + k

-- Define the condition that all three lines intersect at the same point
def lines_intersect (k : ℚ) : Prop :=
  ∃ x y : ℚ, line1 x y ∧ line2 x y ∧ line3 x y k

-- Theorem statement
theorem intersection_point_k :
  ∃! k : ℚ, lines_intersect k ∧ k = 59/13 := by sorry

end NUMINAMATH_CALUDE_intersection_point_k_l4090_409064


namespace NUMINAMATH_CALUDE_oak_trees_planted_l4090_409027

/-- The number of oak trees planted today in the park -/
def trees_planted (current : ℕ) (final : ℕ) : ℕ := final - current

/-- Theorem stating that the number of oak trees planted today is 4 -/
theorem oak_trees_planted : trees_planted 5 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_planted_l4090_409027


namespace NUMINAMATH_CALUDE_lower_circle_radius_is_153_l4090_409053

/-- Configuration of circles and square between parallel lines -/
structure GeometricConfiguration where
  -- Distance between parallel lines
  line_distance : ℝ
  -- Side length of the square
  square_side : ℝ
  -- Radius of the upper circle
  upper_radius : ℝ
  -- The configuration satisfies the given conditions
  h1 : line_distance = 400
  h2 : square_side = 279
  h3 : upper_radius = 65

/-- Calculate the radius of the lower circle -/
def lower_circle_radius (config : GeometricConfiguration) : ℝ :=
  -- Placeholder for the actual calculation
  153

/-- Theorem stating that the radius of the lower circle is 153 units -/
theorem lower_circle_radius_is_153 (config : GeometricConfiguration) :
  lower_circle_radius config = 153 := by
  sorry

#check lower_circle_radius_is_153

end NUMINAMATH_CALUDE_lower_circle_radius_is_153_l4090_409053


namespace NUMINAMATH_CALUDE_cylindrical_to_rectangular_conversion_l4090_409008

theorem cylindrical_to_rectangular_conversion :
  let r : ℝ := 5
  let θ : ℝ := π / 3
  let z : ℝ := 2
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y, z) = (2.5, 5 * Real.sqrt 3 / 2, 2) :=
by sorry

end NUMINAMATH_CALUDE_cylindrical_to_rectangular_conversion_l4090_409008


namespace NUMINAMATH_CALUDE_kendra_shirts_theorem_l4090_409031

/-- Represents the number of shirts Kendra needs for two weeks -/
def shirts_needed : ℕ :=
  let school_days := 5
  let club_days := 3
  let saturday_shirts := 1
  let sunday_shirts := 2
  let weeks := 2
  (school_days + club_days + saturday_shirts + sunday_shirts) * weeks

/-- Theorem stating that Kendra needs 22 shirts to do laundry once every two weeks -/
theorem kendra_shirts_theorem : shirts_needed = 22 := by
  sorry

end NUMINAMATH_CALUDE_kendra_shirts_theorem_l4090_409031


namespace NUMINAMATH_CALUDE_fruit_arrangement_count_l4090_409079

/-- The number of ways to arrange fruits with constraints -/
def fruitArrangements (apples oranges bananas : ℕ) : ℕ :=
  (Nat.factorial (apples + oranges + bananas)) / 
  (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas) * 
  (Nat.choose (apples + bananas) apples)

/-- Theorem stating the number of fruit arrangements -/
theorem fruit_arrangement_count :
  fruitArrangements 4 2 2 = 18900 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangement_count_l4090_409079


namespace NUMINAMATH_CALUDE_quilt_material_requirement_l4090_409084

/-- Given that 7 quilts can be made with 21 yards of material,
    prove that 12 quilts require 36 yards of material. -/
theorem quilt_material_requirement : 
  (7 : ℚ) * (36 : ℚ) = (12 : ℚ) * (21 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_quilt_material_requirement_l4090_409084


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l4090_409073

theorem restaurant_bill_calculation (total_people adults kids : ℕ) (adult_meal_cost : ℚ) :
  total_people = adults + kids →
  total_people = 12 →
  kids = 7 →
  adult_meal_cost = 3 →
  adults * adult_meal_cost = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l4090_409073


namespace NUMINAMATH_CALUDE_hilt_snow_amount_l4090_409040

/-- The amount of snow at Brecknock Elementary School in inches -/
def school_snow : ℕ := 17

/-- The additional amount of snow at Mrs. Hilt's house compared to the school in inches -/
def additional_snow : ℕ := 12

/-- The total amount of snow at Mrs. Hilt's house in inches -/
def hilt_snow : ℕ := school_snow + additional_snow

/-- Theorem stating that the amount of snow at Mrs. Hilt's house is 29 inches -/
theorem hilt_snow_amount : hilt_snow = 29 := by sorry

end NUMINAMATH_CALUDE_hilt_snow_amount_l4090_409040


namespace NUMINAMATH_CALUDE_triangle_area_l4090_409066

/-- The area of a triangle with vertices at (2,-3), (-4,2), and (3,-7) is 19/2 -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, -3)
  let B : ℝ × ℝ := (-4, 2)
  let C : ℝ × ℝ := (3, -7)
  let area := abs ((C.1 - A.1) * (B.2 - A.2) - (C.2 - A.2) * (B.1 - A.1)) / 2
  area = 19 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l4090_409066


namespace NUMINAMATH_CALUDE_min_value_of_some_expression_l4090_409086

/-- The minimum value of |some expression| given the conditions -/
theorem min_value_of_some_expression :
  ∃ (f : ℝ → ℝ),
    (∀ x, |x - 4| + |x + 7| + |f x| ≥ 12) ∧
    (∃ x₀, |x₀ - 4| + |x₀ + 7| + |f x₀| = 12) →
    ∃ x₁, |f x₁| = 1 ∧ ∀ x, |f x| ≥ 1 :=
by sorry


end NUMINAMATH_CALUDE_min_value_of_some_expression_l4090_409086


namespace NUMINAMATH_CALUDE_negative_slope_decreasing_l4090_409095

/-- A linear function with negative slope -/
structure NegativeSlopeLinearFunction where
  k : ℝ
  b : ℝ
  h : k < 0

/-- The function corresponding to a NegativeSlopeLinearFunction -/
def NegativeSlopeLinearFunction.toFun (f : NegativeSlopeLinearFunction) : ℝ → ℝ := 
  fun x ↦ f.k * x + f.b

theorem negative_slope_decreasing (f : NegativeSlopeLinearFunction) 
    (x₁ x₂ : ℝ) (h : x₁ < x₂) : 
    f.toFun x₁ > f.toFun x₂ := by
  sorry

end NUMINAMATH_CALUDE_negative_slope_decreasing_l4090_409095


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l4090_409026

/-- 
Given a geometric sequence {a_n} with common ratio q,
prove that if a₂ = 1 and a₁ + a₃ = -2, then q = -1.
-/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 2 = 1 →                    -- a₂ = 1
  a 1 + a 3 = -2 →             -- a₁ + a₃ = -2
  q = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l4090_409026


namespace NUMINAMATH_CALUDE_stratified_sampling_girls_count_l4090_409093

theorem stratified_sampling_girls_count 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (girl_boy_diff : ℕ) : 
  total_students = 1750 →
  sample_size = 250 →
  girl_boy_diff = 20 →
  ∃ (girls_in_sample : ℕ) (boys_in_sample : ℕ),
    girls_in_sample + boys_in_sample = sample_size ∧
    boys_in_sample = girls_in_sample + girl_boy_diff ∧
    (girls_in_sample : ℚ) / (sample_size : ℚ) = 
      ((total_students - (boys_in_sample * total_students / sample_size)) : ℚ) / (total_students : ℚ) ∧
    total_students - (boys_in_sample * total_students / sample_size) = 805 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_girls_count_l4090_409093


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l4090_409009

theorem rectangle_dimensions (x : ℝ) : 
  (2*x - 3) * (3*x + 4) = 20*x - 12 → x = 7/2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l4090_409009


namespace NUMINAMATH_CALUDE_student_average_greater_than_true_average_l4090_409062

theorem student_average_greater_than_true_average (x y z : ℝ) (h : x < y ∧ y < z) :
  (x + y) / 2 + z > (x + y + z) / 3 := by
  sorry

end NUMINAMATH_CALUDE_student_average_greater_than_true_average_l4090_409062


namespace NUMINAMATH_CALUDE_new_combined_total_capacity_l4090_409029

/-- Represents a weightlifter's lifting capacities -/
structure Lifter where
  cleanAndJerk : ℝ
  snatch : ℝ

/-- Represents the improvement rates for a lifter -/
structure Improvement where
  cleanAndJerkRate : ℝ
  snatchRate : ℝ

/-- Calculates the new lifting capacities after improvement -/
def improve (lifter : Lifter) (imp : Improvement) : Lifter where
  cleanAndJerk := lifter.cleanAndJerk * (1 + imp.cleanAndJerkRate)
  snatch := lifter.snatch * (1 + imp.snatchRate)

/-- Calculates the total lifting capacity of a lifter -/
def totalCapacity (lifter : Lifter) : ℝ :=
  lifter.cleanAndJerk + lifter.snatch

/-- The main theorem to prove -/
theorem new_combined_total_capacity
  (john : Lifter)
  (alice : Lifter)
  (mark : Lifter)
  (johnImp : Improvement)
  (aliceImp : Improvement)
  (markImp : Improvement)
  (h1 : john.cleanAndJerk = 80)
  (h2 : john.snatch = 50)
  (h3 : alice.cleanAndJerk = 90)
  (h4 : alice.snatch = 55)
  (h5 : mark.cleanAndJerk = 100)
  (h6 : mark.snatch = 65)
  (h7 : johnImp.cleanAndJerkRate = 1)  -- doubled means 100% increase
  (h8 : johnImp.snatchRate = 0.8)
  (h9 : aliceImp.cleanAndJerkRate = 0.5)
  (h10 : aliceImp.snatchRate = 0.9)
  (h11 : markImp.cleanAndJerkRate = 0.75)
  (h12 : markImp.snatchRate = 0.7)
  : totalCapacity (improve john johnImp) +
    totalCapacity (improve alice aliceImp) +
    totalCapacity (improve mark markImp) = 775 := by
  sorry

end NUMINAMATH_CALUDE_new_combined_total_capacity_l4090_409029


namespace NUMINAMATH_CALUDE_cubic_fraction_simplification_l4090_409088

theorem cubic_fraction_simplification 
  (a b x : ℝ) 
  (h1 : x = a^3 / b^3) 
  (h2 : a ≠ b) 
  (h3 : b ≠ 0) : 
  (a^3 + b^3) / (a^3 - b^3) = (x + 1) / (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_fraction_simplification_l4090_409088


namespace NUMINAMATH_CALUDE_max_triangles_theorem_max_squares_theorem_l4090_409045

/-- The maximum number of identical, non-overlapping, and parallel triangles that can be fitted into a triangle -/
def max_triangles_in_triangle : ℕ := 6

/-- The maximum number of identical, non-overlapping, and parallel squares that can be fitted into a square -/
def max_squares_in_square : ℕ := 8

/-- Theorem stating the maximum number of identical, non-overlapping, and parallel triangles that can be fitted into a triangle -/
theorem max_triangles_theorem : max_triangles_in_triangle = 6 := by sorry

/-- Theorem stating the maximum number of identical, non-overlapping, and parallel squares that can be fitted into a square -/
theorem max_squares_theorem : max_squares_in_square = 8 := by sorry

end NUMINAMATH_CALUDE_max_triangles_theorem_max_squares_theorem_l4090_409045


namespace NUMINAMATH_CALUDE_min_abs_value_plus_constant_l4090_409036

theorem min_abs_value_plus_constant (x : ℝ) :
  ∀ y : ℝ, |x - 2| + 2023 ≤ |y - 2| + 2023 ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_value_plus_constant_l4090_409036


namespace NUMINAMATH_CALUDE_smallest_n_is_correct_smallest_n_satisfies_property_l4090_409001

/-- The smallest positive integer n with the given divisibility property -/
def smallest_n : ℕ := 13

/-- Proposition stating that smallest_n is the correct answer -/
theorem smallest_n_is_correct :
  ∀ (n : ℕ), n > 0 → 
  (∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 →
    x ∣ y^3 → y ∣ z^3 → z ∣ x^3 →
    x * y * z ∣ (x + y + z)^n) →
  n ≥ smallest_n :=
by sorry

/-- Proposition stating that smallest_n satisfies the required property -/
theorem smallest_n_satisfies_property :
  ∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 →
    x ∣ y^3 → y ∣ z^3 → z ∣ x^3 →
    x * y * z ∣ (x + y + z)^smallest_n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_is_correct_smallest_n_satisfies_property_l4090_409001


namespace NUMINAMATH_CALUDE_race_time_calculation_l4090_409004

/-- Given a race where runner A beats runner B by both distance and time, 
    this theorem proves the time taken by runner A to complete the race. -/
theorem race_time_calculation (race_distance : ℝ) (distance_diff : ℝ) (time_diff : ℝ) :
  race_distance = 1000 ∧ 
  distance_diff = 48 ∧ 
  time_diff = 12 →
  ∃ (time_A : ℝ), time_A = 250 ∧ 
    race_distance / time_A = (race_distance - distance_diff) / (time_A + time_diff) :=
by sorry

end NUMINAMATH_CALUDE_race_time_calculation_l4090_409004


namespace NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l4090_409028

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -1

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Theorem statement
theorem circle_tangent_to_parabola_directrix :
  ∀ x y : ℝ,
  parabola x y →
  (∃ t : ℝ, directrix t ∧ 
    ((x - focus.1)^2 + (y - focus.2)^2 = (t - focus.1)^2)) →
  circle_equation x y :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l4090_409028


namespace NUMINAMATH_CALUDE_sqrt_inequality_l4090_409023

theorem sqrt_inequality (x : ℝ) : 
  3 * x - 2 ≥ 0 → (|Real.sqrt (3 * x - 2) - 3| > 1 ↔ x > 6 ∨ (2/3 ≤ x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l4090_409023


namespace NUMINAMATH_CALUDE_range_of_b_l4090_409024

theorem range_of_b (b : ℝ) : 
  Real.sqrt ((b - 2)^2) = 2 - b ↔ b ∈ Set.Iic 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_b_l4090_409024


namespace NUMINAMATH_CALUDE_pet_ownership_percentages_l4090_409061

theorem pet_ownership_percentages (total_students : ℕ) (cat_owners : ℕ) (dog_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 75)
  (h3 : dog_owners = 125) :
  (cat_owners : ℚ) / total_students * 100 = 15 ∧
  (dog_owners : ℚ) / total_students * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_pet_ownership_percentages_l4090_409061
