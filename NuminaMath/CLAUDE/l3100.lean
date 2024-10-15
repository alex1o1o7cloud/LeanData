import Mathlib

namespace NUMINAMATH_CALUDE_subset_equivalence_l3100_310056

theorem subset_equivalence (φ A : Set α) (p q : Prop) :
  (φ ⊆ A ↔ (φ = A ∨ φ ⊂ A)) →
  (φ ⊆ A ↔ p ∨ q) :=
by sorry

end NUMINAMATH_CALUDE_subset_equivalence_l3100_310056


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_approximate_value_of_b_l3100_310042

theorem geometric_sequence_second_term (b : ℝ) : 
  b > 0 ∧ 
  (∃ r : ℝ, 30 * r = b ∧ b * r = 9/4) → 
  b^2 = 270/4 :=
by sorry

theorem approximate_value_of_b : 
  ∃ b : ℝ, b > 0 ∧ 
  (∃ r : ℝ, 30 * r = b ∧ b * r = 9/4) ∧ 
  (b^2 = 270/4) ∧ 
  (abs (b - 8.215838362) < 0.000000001) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_approximate_value_of_b_l3100_310042


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3100_310061

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (2 + Complex.I) / (1 + 3 * Complex.I)
  Complex.im z = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3100_310061


namespace NUMINAMATH_CALUDE_cube_roll_probability_l3100_310093

theorem cube_roll_probability (total_faces green_faces : ℕ) 
  (h1 : total_faces = 6)
  (h2 : green_faces = 3)
  (h3 : total_faces - green_faces = 3) : 
  (green_faces : ℚ) / total_faces = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cube_roll_probability_l3100_310093


namespace NUMINAMATH_CALUDE_gadget_production_75_workers_4_hours_l3100_310013

/-- Represents the production rate of a worker per hour -/
structure ProductionRate :=
  (gadgets : ℝ)
  (gizmos : ℝ)

/-- Calculates the total production given workers, hours, and rate -/
def totalProduction (workers : ℕ) (hours : ℕ) (rate : ProductionRate) : ProductionRate :=
  { gadgets := workers * hours * rate.gadgets,
    gizmos := workers * hours * rate.gizmos }

theorem gadget_production_75_workers_4_hours 
  (rate1 : ProductionRate)
  (rate2 : ProductionRate)
  (h1 : totalProduction 150 1 rate1 = { gadgets := 450, gizmos := 300 })
  (h2 : totalProduction 100 2 rate2 = { gadgets := 400, gizmos := 500 }) :
  (totalProduction 75 4 rate2).gadgets = 600 := by
sorry

end NUMINAMATH_CALUDE_gadget_production_75_workers_4_hours_l3100_310013


namespace NUMINAMATH_CALUDE_no_natural_solutions_for_cubic_equation_l3100_310018

theorem no_natural_solutions_for_cubic_equation :
  ¬∃ (x y z : ℕ), x^3 + 2*y^3 = 4*z^3 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_for_cubic_equation_l3100_310018


namespace NUMINAMATH_CALUDE_multiplication_subtraction_difference_l3100_310057

theorem multiplication_subtraction_difference : ∀ x : ℤ, 
  x = 11 → (3 * x) - (26 - x) = 18 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_difference_l3100_310057


namespace NUMINAMATH_CALUDE_power_of_two_properties_l3100_310054

theorem power_of_two_properties (n : ℕ) :
  (∃ k : ℕ, n = 3 * k ↔ 7 ∣ (2^n - 1)) ∧
  ¬(7 ∣ (2^n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_properties_l3100_310054


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3100_310014

theorem election_winner_percentage (total_votes : ℕ) (majority : ℕ) (winner_percentage : ℚ) : 
  total_votes = 435 →
  majority = 174 →
  winner_percentage = 70 / 100 →
  (winner_percentage * total_votes : ℚ) - ((1 - winner_percentage) * total_votes : ℚ) = majority :=
by sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3100_310014


namespace NUMINAMATH_CALUDE_one_lamp_position_l3100_310098

/-- Represents a position on the 5x5 grid -/
structure Position where
  x : Fin 5
  y : Fin 5

/-- Represents the state of the 5x5 grid of lamps -/
def Grid := Fin 5 → Fin 5 → Bool

/-- The operation of toggling a lamp and its adjacent lamps -/
def toggle (grid : Grid) (pos : Position) : Grid := sorry

/-- Checks if only one lamp is on in the grid -/
def onlyOneLampOn (grid : Grid) : Bool := sorry

/-- Checks if a position is either the center or directly diagonal to the center -/
def isCenterOrDiagonal (pos : Position) : Bool := sorry

/-- The main theorem: If only one lamp is on after a sequence of toggle operations,
    it must be in the center or directly diagonal to the center -/
theorem one_lamp_position (grid : Grid) (pos : Position) :
  (∃ (ops : List Position), onlyOneLampOn (ops.foldl toggle grid)) →
  (onlyOneLampOn grid ∧ grid pos.x pos.y = true) →
  isCenterOrDiagonal pos := sorry

end NUMINAMATH_CALUDE_one_lamp_position_l3100_310098


namespace NUMINAMATH_CALUDE_arrangement_count_l3100_310087

def valid_arrangements (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (fun k => (n.choose k) ^ 3)

theorem arrangement_count :
  valid_arrangements 4 =
    (Finset.sum (Finset.range 5) (fun k =>
      (Nat.choose 4 k) ^ 3)) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l3100_310087


namespace NUMINAMATH_CALUDE_base_conversion_sum_l3100_310029

-- Define the base conversions
def base_8_to_10 (n : ℕ) : ℕ := 
  2 * (8^2) + 5 * (8^1) + 4 * (8^0)

def base_3_to_10 (n : ℕ) : ℕ := 
  1 * (3^1) + 3 * (3^0)

def base_7_to_10 (n : ℕ) : ℕ := 
  2 * (7^2) + 3 * (7^1) + 2 * (7^0)

def base_5_to_10 (n : ℕ) : ℕ := 
  3 * (5^1) + 2 * (5^0)

-- Theorem statement
theorem base_conversion_sum :
  (base_8_to_10 254 / base_3_to_10 13) + (base_7_to_10 232 / base_5_to_10 32) = 35 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l3100_310029


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l3100_310002

theorem geometric_arithmetic_sequence_sum (x y z : ℝ) 
  (h1 : (4 * y)^2 = 15 * x * z)  -- Geometric sequence condition
  (h2 : 2 / y = 1 / x + 1 / z)   -- Arithmetic sequence condition
  : x / z + z / x = 34 / 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l3100_310002


namespace NUMINAMATH_CALUDE_johns_candy_store_spending_l3100_310008

def weekly_allowance : ℚ := 240 / 100

def arcade_spending : ℚ := (3 / 5) * weekly_allowance

def remaining_after_arcade : ℚ := weekly_allowance - arcade_spending

def toy_store_spending : ℚ := (1 / 3) * remaining_after_arcade

def candy_store_spending : ℚ := remaining_after_arcade - toy_store_spending

theorem johns_candy_store_spending :
  candy_store_spending = 64 / 100 := by sorry

end NUMINAMATH_CALUDE_johns_candy_store_spending_l3100_310008


namespace NUMINAMATH_CALUDE_dental_removal_fraction_l3100_310059

theorem dental_removal_fraction :
  ∀ (x : ℚ),
  (∃ (t₁ t₂ t₃ t₄ : ℕ),
    t₁ + t₂ + t₃ + t₄ = 4 ∧  -- Four adults
    (∀ i, t₁ ≤ i ∧ i ≤ t₄ → 32 = 32) ∧  -- Each adult has 32 teeth
    x * 32 + 3/8 * 32 + 1/2 * 32 + 4 = 40)  -- Total teeth removed
  → x = 1/4 := by
sorry

end NUMINAMATH_CALUDE_dental_removal_fraction_l3100_310059


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l3100_310012

/-- Calculates the sampling interval for systematic sampling -/
def samplingInterval (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  let adjustedPopSize := (populationSize / sampleSize) * sampleSize
  adjustedPopSize / sampleSize

/-- Proves that the sampling interval is 10 for the given problem -/
theorem systematic_sampling_interval :
  samplingInterval 123 12 = 10 := by
  sorry

#eval samplingInterval 123 12

end NUMINAMATH_CALUDE_systematic_sampling_interval_l3100_310012


namespace NUMINAMATH_CALUDE_exists_x0_implies_a_value_l3100_310024

noncomputable section

def f (a x : ℝ) : ℝ := x + Real.exp (x - a)

def g (a x : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem exists_x0_implies_a_value (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ - g a x₀ = 3) → a = -1 - Real.log 2 := by
  sorry

end

end NUMINAMATH_CALUDE_exists_x0_implies_a_value_l3100_310024


namespace NUMINAMATH_CALUDE_four_integers_average_l3100_310067

theorem four_integers_average (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a + b + c + d : ℚ) / 4 = 5 →
  ∀ w x y z : ℕ+, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
    (w + x + y + z : ℚ) / 4 = 5 →
    (d - a : ℤ) ≥ (z - w : ℤ) →
  ((b : ℚ) + c) / 2 = 5/2 := by
sorry

end NUMINAMATH_CALUDE_four_integers_average_l3100_310067


namespace NUMINAMATH_CALUDE_sams_walking_speed_l3100_310046

/-- Proves that Sam's walking speed is equal to Fred's given the problem conditions -/
theorem sams_walking_speed (total_distance : ℝ) (fred_speed : ℝ) (sam_distance : ℝ) :
  total_distance = 50 →
  fred_speed = 5 →
  sam_distance = 25 →
  let fred_distance := total_distance - sam_distance
  let time := fred_distance / fred_speed
  let sam_speed := sam_distance / time
  sam_speed = fred_speed :=
by sorry

end NUMINAMATH_CALUDE_sams_walking_speed_l3100_310046


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l3100_310037

theorem at_least_one_not_less_than_two (a b : ℕ) (h : a + b ≥ 3) : max a b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l3100_310037


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l3100_310035

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l3100_310035


namespace NUMINAMATH_CALUDE_greatest_x_value_l3100_310044

theorem greatest_x_value (x : ℝ) :
  (x^2 - x - 90) / (x - 9) = 2 / (x + 6) →
  x ≤ -8 + Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3100_310044


namespace NUMINAMATH_CALUDE_xyz_value_l3100_310094

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14) :
  x * y * z = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l3100_310094


namespace NUMINAMATH_CALUDE_no_solution_equation_l3100_310016

theorem no_solution_equation : ¬ ∃ (x y z : ℤ), x^3 + y^6 = 7*z + 3 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l3100_310016


namespace NUMINAMATH_CALUDE_not_necessarily_parallel_lines_l3100_310034

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersect_planes : Plane → Plane → Line → Prop)

-- State the theorem
theorem not_necessarily_parallel_lines 
  (α β : Plane) (m n : Line) 
  (h1 : α ≠ β) 
  (h2 : m ≠ n) 
  (h3 : parallel_line_plane m α) 
  (h4 : intersect_planes α β n) : 
  ¬ (∀ m n, parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_parallel_lines_l3100_310034


namespace NUMINAMATH_CALUDE_parabola_parameter_values_l3100_310055

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- Define point M
def M : ℝ × ℝ := (1, 1)

-- Define the distance from M to the directrix
def distance_to_directrix (a : ℝ) : ℝ := 2

theorem parabola_parameter_values :
  ∃ (a : ℝ), (parabola a (M.1) = M.2) ∧ 
             (distance_to_directrix a = 2) ∧ 
             (a = 1/4 ∨ a = -1/12) :=
by sorry

end NUMINAMATH_CALUDE_parabola_parameter_values_l3100_310055


namespace NUMINAMATH_CALUDE_one_pair_probability_l3100_310084

/-- The number of colors of socks -/
def num_colors : ℕ := 5

/-- The number of socks per color -/
def socks_per_color : ℕ := 2

/-- The total number of socks -/
def total_socks : ℕ := num_colors * socks_per_color

/-- The number of socks drawn -/
def drawn_socks : ℕ := 5

/-- The probability of drawing exactly one pair of the same color and three different colored socks -/
def probability_one_pair : ℚ := 20 / 21

theorem one_pair_probability :
  probability_one_pair = (num_colors.choose 4 * 4 * 8) / total_socks.choose drawn_socks :=
by sorry

end NUMINAMATH_CALUDE_one_pair_probability_l3100_310084


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l3100_310017

theorem five_digit_multiple_of_nine :
  ∃ (n : ℕ), n = 56781 ∧ n % 9 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l3100_310017


namespace NUMINAMATH_CALUDE_expression_value_l3100_310079

theorem expression_value (a b : ℝ) 
  (ha : a = 2 * Real.sin (45 * π / 180) + 1)
  (hb : b = 2 * Real.cos (45 * π / 180) - 1) :
  ((a^2 + b^2) / (2*a*b) - 1) / ((a^2 - b^2) / (a^2*b + a*b^2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3100_310079


namespace NUMINAMATH_CALUDE_smallest_four_digit_remainder_five_mod_six_l3100_310091

theorem smallest_four_digit_remainder_five_mod_six : 
  ∃ (n : ℕ), 
    (1000 ≤ n ∧ n ≤ 9999) ∧ 
    (n % 6 = 5) ∧
    (∀ m, (1000 ≤ m ∧ m ≤ 9999) → (m % 6 = 5) → n ≤ m) ∧
    n = 1001 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_remainder_five_mod_six_l3100_310091


namespace NUMINAMATH_CALUDE_turns_for_both_buckets_l3100_310071

/-- Represents the capacity of bucket Q -/
def capacity_Q : ℝ := 1

/-- Represents the capacity of bucket P -/
def capacity_P : ℝ := 3 * capacity_Q

/-- Represents the number of turns it takes bucket P to fill the drum -/
def turns_P : ℕ := 80

/-- Represents the capacity of the drum -/
def drum_capacity : ℝ := turns_P * capacity_P

/-- Represents the combined capacity of buckets P and Q -/
def combined_capacity : ℝ := capacity_P + capacity_Q

/-- 
Proves that the number of turns it takes for both buckets P and Q together 
to fill the drum is 60, given the conditions stated in the problem.
-/
theorem turns_for_both_buckets : 
  (drum_capacity / combined_capacity : ℝ) = 60 := by sorry

end NUMINAMATH_CALUDE_turns_for_both_buckets_l3100_310071


namespace NUMINAMATH_CALUDE_four_tellers_coins_l3100_310072

/-- Calculates the total number of coins for a given number of bank tellers -/
def totalCoins (numTellers : ℕ) (rollsPerTeller : ℕ) (coinsPerRoll : ℕ) : ℕ :=
  numTellers * rollsPerTeller * coinsPerRoll

/-- Theorem: Four bank tellers have 1000 coins in total -/
theorem four_tellers_coins :
  totalCoins 4 10 25 = 1000 := by
  sorry

#eval totalCoins 4 10 25  -- Should output 1000

end NUMINAMATH_CALUDE_four_tellers_coins_l3100_310072


namespace NUMINAMATH_CALUDE_root_quadratic_equation_l3100_310075

theorem root_quadratic_equation (a : ℝ) : 2 * a^2 = a + 4 → 4 * a^2 - 2 * a = 8 := by
  sorry

end NUMINAMATH_CALUDE_root_quadratic_equation_l3100_310075


namespace NUMINAMATH_CALUDE_lines_parallel_perpendicular_l3100_310062

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := x + (1 + a) * y + a - 1 = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0

-- Define parallel and perpendicular conditions
def parallel (a : ℝ) : Prop := a = 1
def perpendicular (a : ℝ) : Prop := a = -2/3

-- Theorem statement
theorem lines_parallel_perpendicular :
  (∀ a : ℝ, (∀ x y : ℝ, l₁ a x y ∧ l₂ a x y) → parallel a) ∧
  (∀ a : ℝ, (∀ x y : ℝ, l₁ a x y ∧ l₂ a x y) → perpendicular a) :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_perpendicular_l3100_310062


namespace NUMINAMATH_CALUDE_simplify_expression_l3100_310085

theorem simplify_expression (x y : ℝ) : 3 * x^2 - 2 * x * y - 3 * x^2 + 4 * x * y - 1 = 2 * x * y - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3100_310085


namespace NUMINAMATH_CALUDE_largest_odd_integer_with_coprime_primes_l3100_310019

theorem largest_odd_integer_with_coprime_primes : ∃ (n : ℕ), 
  n = 105 ∧ 
  n % 2 = 1 ∧
  (∀ k : ℕ, 1 < k → k < n → k % 2 = 1 → Nat.gcd k n = 1 → Nat.Prime k) ∧
  (∀ m : ℕ, m > n → m % 2 = 1 → 
    ∃ k : ℕ, 1 < k ∧ k < m ∧ k % 2 = 1 ∧ Nat.gcd k m = 1 ∧ ¬Nat.Prime k) :=
by sorry

end NUMINAMATH_CALUDE_largest_odd_integer_with_coprime_primes_l3100_310019


namespace NUMINAMATH_CALUDE_base8_digit_product_7890_l3100_310074

/-- Given a natural number n, returns the list of its digits in base 8 --/
def toBase8Digits (n : ℕ) : List ℕ :=
  sorry

/-- The product of a list of natural numbers --/
def listProduct (l : List ℕ) : ℕ :=
  sorry

theorem base8_digit_product_7890 :
  listProduct (toBase8Digits 7890) = 336 :=
sorry

end NUMINAMATH_CALUDE_base8_digit_product_7890_l3100_310074


namespace NUMINAMATH_CALUDE_completing_square_transformation_l3100_310064

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l3100_310064


namespace NUMINAMATH_CALUDE_cyclingProblemSolution_l3100_310083

/-- Natalia's cycling distances over four days --/
def cyclingProblem (tuesday : ℕ) : Prop :=
  let monday : ℕ := 40
  let wednesday : ℕ := tuesday / 2
  let thursday : ℕ := monday + wednesday
  monday + tuesday + wednesday + thursday = 180

/-- The solution to Natalia's cycling problem --/
theorem cyclingProblemSolution : ∃ (tuesday : ℕ), cyclingProblem tuesday ∧ tuesday = 33 := by
  sorry

#check cyclingProblemSolution

end NUMINAMATH_CALUDE_cyclingProblemSolution_l3100_310083


namespace NUMINAMATH_CALUDE_total_games_in_season_l3100_310038

theorem total_games_in_season (total_teams : ℕ) (teams_per_division : ℕ) 
  (h1 : total_teams = 16)
  (h2 : teams_per_division = 8)
  (h3 : total_teams = 2 * teams_per_division)
  (h4 : ∀ (division : Fin 2), ∀ (team : Fin teams_per_division),
    (division.val = 0 → 
      (teams_per_division - 1) * 2 + teams_per_division = 22) ∧
    (division.val = 1 → 
      (teams_per_division - 1) * 2 + teams_per_division = 22)) :
  total_teams * 22 / 2 = 176 := by
sorry

end NUMINAMATH_CALUDE_total_games_in_season_l3100_310038


namespace NUMINAMATH_CALUDE_carnation_count_l3100_310000

theorem carnation_count (total_flowers : ℕ) (roses : ℕ) (carnations : ℕ) : 
  total_flowers = 10 → roses = 5 → total_flowers = roses + carnations → carnations = 5 := by
  sorry

end NUMINAMATH_CALUDE_carnation_count_l3100_310000


namespace NUMINAMATH_CALUDE_baduk_stone_difference_l3100_310050

theorem baduk_stone_difference (total : ℕ) (white : ℕ) (h1 : total = 928) (h2 : white = 713) :
  white - (total - white) = 498 := by
  sorry

end NUMINAMATH_CALUDE_baduk_stone_difference_l3100_310050


namespace NUMINAMATH_CALUDE_expand_expression_l3100_310090

theorem expand_expression (x y : ℝ) : (3*x + 5) * (4*y^2 + 15) = 12*x*y^2 + 45*x + 20*y^2 + 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3100_310090


namespace NUMINAMATH_CALUDE_stairs_climbing_time_l3100_310040

def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem stairs_climbing_time : arithmeticSum 30 8 6 = 300 := by
  sorry

end NUMINAMATH_CALUDE_stairs_climbing_time_l3100_310040


namespace NUMINAMATH_CALUDE_cos_pi_plus_alpha_implies_sin_5pi_over_2_minus_alpha_l3100_310033

theorem cos_pi_plus_alpha_implies_sin_5pi_over_2_minus_alpha 
  (α : Real) 
  (h : Real.cos (Real.pi + α) = -1/3) : 
  Real.sin ((5/2) * Real.pi - α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_plus_alpha_implies_sin_5pi_over_2_minus_alpha_l3100_310033


namespace NUMINAMATH_CALUDE_product_difference_squares_l3100_310026

theorem product_difference_squares : (3 + Real.sqrt 7) * (3 - Real.sqrt 7) = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_squares_l3100_310026


namespace NUMINAMATH_CALUDE_inverse_function_ln_l3100_310023

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1)

noncomputable def g (x : ℝ) : ℝ := Real.exp x + 1

theorem inverse_function_ln (x : ℝ) (hx : x > 2) :
  Function.Injective f ∧
  Function.Surjective f ∧
  (∀ y, y > 0 → g y > 2) ∧
  (∀ y, y > 0 → f (g y) = y) ∧
  (∀ x, x > 2 → g (f x) = x) := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_ln_l3100_310023


namespace NUMINAMATH_CALUDE_problem_solution_l3100_310047

theorem problem_solution (x y : ℝ) : (x + 3)^2 + Real.sqrt (2 - y) = 0 → (x + y)^2021 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3100_310047


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3100_310036

/-- Given a hyperbola with equation x^2 - ty^2 = 3t and focal distance 6, its eccentricity is √6/2 -/
theorem hyperbola_eccentricity (t : ℝ) :
  (∃ (x y : ℝ), x^2 - t*y^2 = 3*t) →  -- Hyperbola equation
  (∃ (c : ℝ), c = 3) →  -- Focal distance is 6, so half of it (c) is 3
  (∃ (e : ℝ), e = (Real.sqrt 6) / 2 ∧ e = (Real.sqrt (t + 1))) -- Eccentricity
  :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3100_310036


namespace NUMINAMATH_CALUDE_tank_capacity_l3100_310066

theorem tank_capacity (initial_fraction : ℚ) (final_fraction : ℚ) (added_amount : ℚ) :
  initial_fraction = 5 / 8 →
  final_fraction = 19 / 24 →
  added_amount = 15 →
  ∃ (total_capacity : ℚ),
    initial_fraction * total_capacity + added_amount = final_fraction * total_capacity ∧
    total_capacity = 90 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l3100_310066


namespace NUMINAMATH_CALUDE_rectangle_area_l3100_310092

/-- Given a square with side length 15 and a rectangle with length 18 and diagonal 27,
    prove that the area of the rectangle is 216 when its perimeter equals the square's perimeter. -/
theorem rectangle_area (square_side : ℝ) (rect_length rect_diagonal : ℝ) :
  square_side = 15 →
  rect_length = 18 →
  rect_diagonal = 27 →
  4 * square_side = 2 * rect_length + 2 * (rect_diagonal ^ 2 - rect_length ^ 2).sqrt →
  rect_length * (rect_diagonal ^ 2 - rect_length ^ 2).sqrt = 216 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3100_310092


namespace NUMINAMATH_CALUDE_barbara_initial_candies_l3100_310060

/-- The number of candies Barbara bought -/
def candies_bought : ℕ := 18

/-- The total number of candies Barbara has after buying more -/
def total_candies : ℕ := 27

/-- The number of candies Barbara had initially -/
def initial_candies : ℕ := total_candies - candies_bought

theorem barbara_initial_candies : initial_candies = 9 := by
  sorry

end NUMINAMATH_CALUDE_barbara_initial_candies_l3100_310060


namespace NUMINAMATH_CALUDE_min_communication_size_l3100_310030

/-- Represents a set of cards with positive numbers -/
def CardSet := Finset ℕ+

/-- The number of cards -/
def n : ℕ := 100

/-- A function that takes a set of cards and returns a set of communicated values -/
def communicate (cards : CardSet) : Finset ℕ := sorry

/-- Predicate to check if a set of communicated values uniquely determines the original card set -/
def uniquely_determines (comms : Finset ℕ) (cards : CardSet) : Prop := sorry

theorem min_communication_size :
  ∀ (cards : CardSet),
  (cards.card = n) →
  ∃ (comms : Finset ℕ),
    (communicate cards = comms) ∧
    (uniquely_determines comms cards) ∧
    (comms.card = n + 1) ∧
    (∀ (comms' : Finset ℕ),
      (communicate cards = comms') ∧
      (uniquely_determines comms' cards) →
      (comms'.card ≥ n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_min_communication_size_l3100_310030


namespace NUMINAMATH_CALUDE_opposite_of_pi_l3100_310032

theorem opposite_of_pi : -π = -π := by sorry

end NUMINAMATH_CALUDE_opposite_of_pi_l3100_310032


namespace NUMINAMATH_CALUDE_tristan_study_hours_l3100_310043

/-- Represents the number of hours Tristan studies each day of the week -/
structure StudyHours where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ
  saturday : ℝ
  sunday : ℝ

/-- Theorem stating the number of hours Tristan studies from Wednesday to Friday -/
theorem tristan_study_hours (h : StudyHours) : 
  h.monday = 4 ∧ 
  h.tuesday = 2 * h.monday ∧ 
  h.wednesday = h.thursday ∧ 
  h.thursday = h.friday ∧ 
  h.monday + h.tuesday + h.wednesday + h.thursday + h.friday + h.saturday + h.sunday = 25 ∧ 
  h.saturday = h.sunday → 
  h.wednesday = 13/3 := by
sorry

#eval 13/3  -- To show the result is approximately 4.33

end NUMINAMATH_CALUDE_tristan_study_hours_l3100_310043


namespace NUMINAMATH_CALUDE_man_walking_speed_l3100_310011

/-- The speed of a man walking alongside a train --/
theorem man_walking_speed (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) :
  train_length = 900 →
  crossing_time = 53.99568034557235 →
  train_speed_kmh = 63 →
  ∃ (man_speed : ℝ), abs (man_speed - 0.832) < 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_man_walking_speed_l3100_310011


namespace NUMINAMATH_CALUDE_ratio_transformation_l3100_310009

theorem ratio_transformation (x : ℚ) : 
  (2 + x) / (3 + x) = 4 / 5 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_transformation_l3100_310009


namespace NUMINAMATH_CALUDE_basketball_game_result_l3100_310049

/-- Represents a basketball team --/
structure Team where
  initial_score : ℕ
  baskets_scored : ℕ
  basket_value : ℕ

/-- Calculates the final score of a team --/
def final_score (team : Team) : ℕ := team.initial_score + team.baskets_scored * team.basket_value

/-- The basketball game scenario --/
def basketball_game_scenario : Prop :=
  let hornets : Team := { initial_score := 86, baskets_scored := 2, basket_value := 2 }
  let fireflies : Team := { initial_score := 74, baskets_scored := 7, basket_value := 3 }
  final_score fireflies - final_score hornets = 5

/-- Theorem stating the result of the basketball game --/
theorem basketball_game_result : basketball_game_scenario := by sorry

end NUMINAMATH_CALUDE_basketball_game_result_l3100_310049


namespace NUMINAMATH_CALUDE_intersection_points_form_diameter_l3100_310027

/-- Two circles in a plane -/
structure TwoCircles where
  S₁ : Set (ℝ × ℝ)
  S₂ : Set (ℝ × ℝ)

/-- Intersection points of the two circles -/
def intersection_points (tc : TwoCircles) : Set (ℝ × ℝ) :=
  tc.S₁ ∩ tc.S₂

/-- Tangent line to a circle at a point -/
def tangent_line (S : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Radius of a circle -/
def radius (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- Inner arc of a circle -/
def inner_arc (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- Line passing through two points -/
def line_through (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Diameter of a circle -/
def is_diameter (S : Set (ℝ × ℝ)) (p q : ℝ × ℝ) : Prop := sorry

/-- Main theorem -/
theorem intersection_points_form_diameter
  (tc : TwoCircles)
  (A B : ℝ × ℝ)
  (h_AB : A ∈ intersection_points tc ∧ B ∈ intersection_points tc)
  (h_tangent : tangent_line tc.S₁ A = radius tc.S₂ ∧ tangent_line tc.S₁ B = radius tc.S₂)
  (C : ℝ × ℝ)
  (h_C : C ∈ inner_arc tc.S₁)
  (K L : ℝ × ℝ)
  (h_K : K ∈ line_through A C ∩ tc.S₂)
  (h_L : L ∈ line_through B C ∩ tc.S₂) :
  is_diameter tc.S₂ K L := by sorry

end NUMINAMATH_CALUDE_intersection_points_form_diameter_l3100_310027


namespace NUMINAMATH_CALUDE_symmetric_axis_after_transformation_l3100_310069

/-- Given a function f(x) = √3 sin(x - π/6) + cos(x - π/6), 
    after stretching the horizontal coordinate to twice its original length 
    and shifting the graph π/6 units to the left, 
    one symmetric axis of the resulting function is at x = 5π/6 -/
theorem symmetric_axis_after_transformation (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sqrt 3 * Real.sin (x - π/6) + Real.cos (x - π/6)
  let g : ℝ → ℝ := λ x => f ((x + π/6) / 2)
  ∃ (k : ℤ), g (5*π/6 + 2*π*k) = g (5*π/6 - 2*π*k) := by
  sorry


end NUMINAMATH_CALUDE_symmetric_axis_after_transformation_l3100_310069


namespace NUMINAMATH_CALUDE_smallest_cube_with_more_than_half_remaining_l3100_310003

theorem smallest_cube_with_more_than_half_remaining : 
  ∀ n : ℕ, n > 0 → ((n : ℚ) - 4)^3 > (n : ℚ)^3 / 2 ↔ n ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_with_more_than_half_remaining_l3100_310003


namespace NUMINAMATH_CALUDE_parabola_kite_sum_l3100_310031

/-- The sum of coefficients for two parabolas forming a kite -/
theorem parabola_kite_sum (a b : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    -- Parabola 1 intersects x-axis
    a * x₁^2 - 4 = 0 ∧ 
    a * x₂^2 - 4 = 0 ∧ 
    x₁ ≠ x₂ ∧
    -- Parabola 2 intersects x-axis
    6 - b * x₁^2 = 0 ∧ 
    6 - b * x₂^2 = 0 ∧
    -- Parabolas intersect y-axis
    y₁ = -4 ∧
    y₂ = 6 ∧
    -- Area of kite formed by intersection points
    (1/2) * (x₂ - x₁) * (y₂ - y₁) = 16) →
  a + b = 3.9 := by
sorry

end NUMINAMATH_CALUDE_parabola_kite_sum_l3100_310031


namespace NUMINAMATH_CALUDE_four_times_three_equals_thirtyone_l3100_310020

-- Define the multiplication operation based on the given condition
def special_mult (a b : ℤ) : ℤ := a^2 + 2*a*b - b^2

-- State the theorem
theorem four_times_three_equals_thirtyone : special_mult 4 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_four_times_three_equals_thirtyone_l3100_310020


namespace NUMINAMATH_CALUDE_maximize_subsidy_l3100_310070

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.log (x + 1) - x / 10 + 1

theorem maximize_subsidy (m : ℝ) (h_m : m > 0) :
  let max_subsidy := fun x : ℝ => x ≥ 1 ∧ x ≤ 9 ∧ ∀ y, 1 ≤ y ∧ y ≤ 9 → f m x ≥ f m y
  (m ≤ 1/5 ∧ max_subsidy 1) ∨
  (1/5 < m ∧ m < 1 ∧ max_subsidy (10*m - 1)) ∨
  (m ≥ 1 ∧ max_subsidy 9) :=
by sorry

end NUMINAMATH_CALUDE_maximize_subsidy_l3100_310070


namespace NUMINAMATH_CALUDE_bob_benefit_reduction_l3100_310001

/-- Calculates the monthly reduction in housing benefit given a raise, work hours, and net increase --/
def monthly_benefit_reduction (raise_per_hour : ℚ) (hours_per_week : ℕ) (net_increase_per_week : ℚ) : ℚ :=
  4 * (raise_per_hour * hours_per_week - net_increase_per_week)

/-- Theorem stating that given the specific conditions, the monthly reduction in housing benefit is $60 --/
theorem bob_benefit_reduction :
  monthly_benefit_reduction (1/2) 40 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_bob_benefit_reduction_l3100_310001


namespace NUMINAMATH_CALUDE_vector_at_negative_one_l3100_310006

/-- A line in 3D space parameterized by t -/
structure ParametricLine where
  -- The vector on the line at t = 0
  v0 : ℝ × ℝ × ℝ
  -- The vector on the line at t = 1
  v1 : ℝ × ℝ × ℝ

/-- The vector on the line at a given t -/
def vectorAtT (line : ParametricLine) (t : ℝ) : ℝ × ℝ × ℝ :=
  let (x0, y0, z0) := line.v0
  let (x1, y1, z1) := line.v1
  (x0 + t * (x1 - x0), y0 + t * (y1 - y0), z0 + t * (z1 - z0))

theorem vector_at_negative_one (line : ParametricLine) 
  (h1 : line.v0 = (2, 6, 16)) 
  (h2 : line.v1 = (1, 1, 4)) : 
  vectorAtT line (-1) = (3, 11, 28) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_negative_one_l3100_310006


namespace NUMINAMATH_CALUDE_positive_real_power_difference_integer_l3100_310082

theorem positive_real_power_difference_integer (x : ℝ) (h1 : x > 0) 
  (h2 : ∃ (a b : ℤ), x^2012 - x^2001 = a ∧ x^2001 - x^1990 = b) : 
  ∃ (n : ℤ), x = n :=
sorry

end NUMINAMATH_CALUDE_positive_real_power_difference_integer_l3100_310082


namespace NUMINAMATH_CALUDE_vacation_cost_division_l3100_310077

theorem vacation_cost_division (total_cost : ℕ) (cost_difference : ℕ) : 
  total_cost = 720 →
  (total_cost / 4 + cost_difference) * 3 = total_cost →
  cost_difference = 60 →
  3 = total_cost / (total_cost / 4 + cost_difference) :=
by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_division_l3100_310077


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3100_310045

theorem trigonometric_equation_solution (x : ℝ) :
  2 * Real.cos x - 5 * Real.sin x = 3 →
  (3 * Real.sin x + 2 * Real.cos x = (-21 + 13 * Real.sqrt 145) / 58) ∨
  (3 * Real.sin x + 2 * Real.cos x = (-21 - 13 * Real.sqrt 145) / 58) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3100_310045


namespace NUMINAMATH_CALUDE_odd_digits_base4_233_l3100_310010

/-- Counts the number of odd digits in the base-4 representation of a natural number. -/
def countOddDigitsBase4 (n : ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-4 representation of 233 is 2. -/
theorem odd_digits_base4_233 : countOddDigitsBase4 233 = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_base4_233_l3100_310010


namespace NUMINAMATH_CALUDE_at_least_three_solutions_nine_solutions_for_2019_l3100_310095

/-- The number of solutions to the equation 1/x + 1/y = 1/a for positive integers x, y, and a > 1 -/
def num_solutions (a : ℕ) : ℕ := sorry

/-- The proposition that there are at least three distinct solutions for any a > 1 -/
theorem at_least_three_solutions (a : ℕ) (ha : a > 1) :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℕ),
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
    (1 : ℚ) / x₁ + (1 : ℚ) / y₁ = (1 : ℚ) / a ∧
    (1 : ℚ) / x₂ + (1 : ℚ) / y₂ = (1 : ℚ) / a ∧
    (1 : ℚ) / x₃ + (1 : ℚ) / y₃ = (1 : ℚ) / a :=
sorry

/-- The proposition that there are exactly 9 solutions when a = 2019 -/
theorem nine_solutions_for_2019 : num_solutions 2019 = 9 :=
sorry

end NUMINAMATH_CALUDE_at_least_three_solutions_nine_solutions_for_2019_l3100_310095


namespace NUMINAMATH_CALUDE_infinite_valid_moves_l3100_310015

-- Define the grid
def InfiniteSquareGrid := ℤ × ℤ

-- Define the directions
inductive Direction
| North
| South
| East
| West

-- Define a car
structure Car where
  position : InfiniteSquareGrid
  direction : Direction

-- Define the state of the grid
structure GridState where
  cars : Finset Car

-- Define a valid move
def validMove (state : GridState) (car : Car) : Prop :=
  car ∈ state.cars ∧
  (∀ other : Car, other ∈ state.cars → other.position ≠ car.position) ∧
  (∀ other : Car, other ∈ state.cars → 
    match car.direction with
    | Direction.North => other.position ≠ (car.position.1, car.position.2 + 1)
    | Direction.South => other.position ≠ (car.position.1, car.position.2 - 1)
    | Direction.East => other.position ≠ (car.position.1 + 1, car.position.2)
    | Direction.West => other.position ≠ (car.position.1 - 1, car.position.2)
  ) ∧
  (∀ other : Car, other ∈ state.cars →
    (car.direction = Direction.East ∧ other.direction = Direction.West → car.position.1 < other.position.1) ∧
    (car.direction = Direction.West ∧ other.direction = Direction.East → car.position.1 > other.position.1) ∧
    (car.direction = Direction.North ∧ other.direction = Direction.South → car.position.2 < other.position.2) ∧
    (car.direction = Direction.South ∧ other.direction = Direction.North → car.position.2 > other.position.2))

-- Define the theorem
theorem infinite_valid_moves (initialState : GridState) : 
  ∃ (moveSequence : ℕ → Car), 
    (∀ n : ℕ, validMove initialState (moveSequence n)) ∧
    (∀ car : Car, car ∈ initialState.cars → ∀ k : ℕ, ∃ n > k, moveSequence n = car) :=
sorry

end NUMINAMATH_CALUDE_infinite_valid_moves_l3100_310015


namespace NUMINAMATH_CALUDE_optimal_removal_l3100_310051

-- Define the grid
inductive Square
| a | b | c | d | e | f | g | j | k | l | m | n

-- Define the initial shape
def initial_shape : Set Square :=
  {Square.a, Square.b, Square.c, Square.d, Square.e, Square.f, Square.g, Square.j, Square.k, Square.l, Square.m, Square.n}

-- Define adjacency relation
def adjacent : Square → Square → Prop := sorry

-- Define connectivity
def is_connected (shape : Set Square) : Prop := sorry

-- Define perimeter calculation
def perimeter (shape : Set Square) : ℕ := sorry

-- Define the set of all possible pairs of squares to remove
def removable_pairs : Set (Square × Square) := sorry

theorem optimal_removal :
  ∀ (pair : Square × Square),
    pair ∈ removable_pairs →
    is_connected (initial_shape \ {pair.1, pair.2}) →
    perimeter (initial_shape \ {pair.1, pair.2}) ≤ 
    max (perimeter (initial_shape \ {Square.d, Square.k}))
        (perimeter (initial_shape \ {Square.e, Square.k})) :=
sorry

end NUMINAMATH_CALUDE_optimal_removal_l3100_310051


namespace NUMINAMATH_CALUDE_line_equations_l3100_310080

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := 2 * x + 4 * y - 1 = 0

-- Define a general line passing through a point
def line_through_point (a b c : ℝ) (x₀ y₀ : ℝ) : Prop :=
  a * x₀ + b * y₀ + c = 0

-- Define parallel lines
def parallel_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

-- Define perpendicular lines
def perpendicular_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ * a₂ + b₁ * b₂ = 0

theorem line_equations :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (∀ x y, l₁ x y ↔ a₁ * x + b₁ * y + c₁ = 0) ∧
    line_through_point a₂ b₂ c₂ 1 (-2) ∧
    ((parallel_lines a₁ b₁ c₁ a₂ b₂ c₂ →
      ∀ x y, a₂ * x + b₂ * y + c₂ = 0 ↔ x + 2 * y + 3 = 0) ∧
     (perpendicular_lines a₁ b₁ c₁ a₂ b₂ c₂ →
      ∀ x y, a₂ * x + b₂ * y + c₂ = 0 ↔ 2 * x - y - 4 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_equations_l3100_310080


namespace NUMINAMATH_CALUDE_tree_growth_rate_l3100_310007

theorem tree_growth_rate (h : ℝ) (initial_height : ℝ) (growth_period : ℕ) :
  initial_height = 4 →
  growth_period = 6 →
  initial_height + 6 * h = (initial_height + 4 * h) * (1 + 1/7) →
  h = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_rate_l3100_310007


namespace NUMINAMATH_CALUDE_expand_and_evaluate_l3100_310078

theorem expand_and_evaluate : 
  ∀ x : ℝ, (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 ∧ 
  (let x : ℝ := 5; 4 * x^2 + 4 * x - 24) = 96 := by sorry

end NUMINAMATH_CALUDE_expand_and_evaluate_l3100_310078


namespace NUMINAMATH_CALUDE_x_values_l3100_310039

def S : Set ℤ := {1, -1}

theorem x_values (a b c d e f : ℤ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S) (he : e ∈ S) (hf : f ∈ S) :
  {x | ∃ (a b c d e f : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ x = a - b + c - d + e - f} = {-6, -4, -2, 0, 2, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_x_values_l3100_310039


namespace NUMINAMATH_CALUDE_pond_capacity_l3100_310089

theorem pond_capacity 
  (normal_rate : ℝ) 
  (drought_factor : ℝ) 
  (fill_time : ℝ) 
  (h1 : normal_rate = 6) 
  (h2 : drought_factor = 2/3) 
  (h3 : fill_time = 50) : 
  normal_rate * drought_factor * fill_time = 200 := by
  sorry

end NUMINAMATH_CALUDE_pond_capacity_l3100_310089


namespace NUMINAMATH_CALUDE_log_sum_equality_l3100_310025

theorem log_sum_equality : 10^(Real.log 3 / Real.log 10) + Real.log 25 / Real.log 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l3100_310025


namespace NUMINAMATH_CALUDE_song_book_cost_l3100_310048

theorem song_book_cost (flute_cost music_stand_cost total_spent : ℚ)
  (h1 : flute_cost = 142.46)
  (h2 : music_stand_cost = 8.89)
  (h3 : total_spent = 158.35) :
  total_spent - (flute_cost + music_stand_cost) = 7.00 := by
  sorry

end NUMINAMATH_CALUDE_song_book_cost_l3100_310048


namespace NUMINAMATH_CALUDE_total_animals_count_l3100_310063

def animal_count : ℕ → ℕ → ℕ → ℕ
| snakes, arctic_foxes, leopards =>
  let bee_eaters := 12 * leopards
  let cheetahs := snakes / 3
  let alligators := 2 * (arctic_foxes + leopards)
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators

theorem total_animals_count :
  animal_count 100 80 20 = 673 :=
by sorry

end NUMINAMATH_CALUDE_total_animals_count_l3100_310063


namespace NUMINAMATH_CALUDE_age_difference_and_future_relation_l3100_310028

/-- A digit is a natural number from 0 to 9 -/
def Digit : Type := {n : ℕ // n ≤ 9}

/-- Jack's age given two digits -/
def jack_age (a b : Digit) : ℕ := 10 * a.val + b.val

/-- Bill's age given two digits -/
def bill_age (a b : Digit) : ℕ := b.val^2 + a.val

theorem age_difference_and_future_relation :
  ∃ (a b : Digit), 
    (jack_age a b - bill_age a b = 18) ∧ 
    (jack_age a b + 6 = 3 * (bill_age a b + 6)) := by
  sorry

end NUMINAMATH_CALUDE_age_difference_and_future_relation_l3100_310028


namespace NUMINAMATH_CALUDE_simplify_expression_l3100_310088

theorem simplify_expression (x y : ℝ) :
  5 * x - 3 * y + 9 * x^2 + 8 - (4 - 5 * x + 3 * y - 9 * x^2) =
  18 * x^2 + 10 * x - 6 * y + 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3100_310088


namespace NUMINAMATH_CALUDE_brush_width_ratio_l3100_310096

theorem brush_width_ratio (w l b : ℝ) (h1 : w = 4) (h2 : l = 9) : 
  b * Real.sqrt (w^2 + l^2) = (w * l) / 3 → l / b = 3 * Real.sqrt 97 / 4 := by
  sorry

end NUMINAMATH_CALUDE_brush_width_ratio_l3100_310096


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3100_310097

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. --/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 1 + 3 * a 8 + a 15 = 60) :
  2 * a 9 - a 10 = 12 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3100_310097


namespace NUMINAMATH_CALUDE_pythagorean_triple_example_l3100_310076

def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_example :
  (isPythagoreanTriple 6 8 10) ∧
  ¬(isPythagoreanTriple 2 3 4) ∧
  ¬(isPythagoreanTriple 4 5 6) ∧
  ¬(isPythagoreanTriple 7 8 9) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_example_l3100_310076


namespace NUMINAMATH_CALUDE_total_money_from_stone_sale_l3100_310021

def number_of_stones : ℕ := 8
def price_per_stone : ℕ := 1785

theorem total_money_from_stone_sale : number_of_stones * price_per_stone = 14280 := by
  sorry

end NUMINAMATH_CALUDE_total_money_from_stone_sale_l3100_310021


namespace NUMINAMATH_CALUDE_quadratic_equation_root_and_q_l3100_310058

theorem quadratic_equation_root_and_q (p q : ℝ) : 
  (∃ x : ℂ, 5 * x^2 + p * x + q = 0 ∧ x = 3 + 2*I) →
  q = 65 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_and_q_l3100_310058


namespace NUMINAMATH_CALUDE_root_product_theorem_l3100_310005

theorem root_product_theorem (y₁ y₂ y₃ y₄ y₅ : ℂ) : 
  (y₁^5 - 3*y₁^3 + 2 = 0) →
  (y₂^5 - 3*y₂^3 + 2 = 0) →
  (y₃^5 - 3*y₃^3 + 2 = 0) →
  (y₄^5 - 3*y₄^3 + 2 = 0) →
  (y₅^5 - 3*y₅^3 + 2 = 0) →
  ((y₁^2 - 3) * (y₂^2 - 3) * (y₃^2 - 3) * (y₄^2 - 3) * (y₅^2 - 3) = -32) :=
by sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3100_310005


namespace NUMINAMATH_CALUDE_greatest_common_divisor_540_126_under_60_l3100_310053

theorem greatest_common_divisor_540_126_under_60 : 
  Nat.gcd 540 126 < 60 ∧ 
  ∀ d : Nat, d ∣ 540 ∧ d ∣ 126 ∧ d < 60 → d ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_540_126_under_60_l3100_310053


namespace NUMINAMATH_CALUDE_last_term_before_one_l3100_310052

def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1 : ℝ) * d

theorem last_term_before_one (a : ℝ) (d : ℝ) (n : ℕ) :
  a = 100 ∧ d = -4 →
  arithmetic_sequence a d 25 > 1 ∧
  arithmetic_sequence a d 26 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_last_term_before_one_l3100_310052


namespace NUMINAMATH_CALUDE_sum_divisible_by_ten_l3100_310081

theorem sum_divisible_by_ten (n : ℕ) : 
  10 ∣ (n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2) ↔ n % 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_divisible_by_ten_l3100_310081


namespace NUMINAMATH_CALUDE_max_sum_with_product_2310_l3100_310041

theorem max_sum_with_product_2310 :
  ∀ A B C : ℕ+,
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2310 →
  A + B + C ≤ 48 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_product_2310_l3100_310041


namespace NUMINAMATH_CALUDE_evelyn_bottle_caps_l3100_310004

def initial_caps : ℕ := 18
def found_caps : ℕ := 63

theorem evelyn_bottle_caps : initial_caps + found_caps = 81 := by
  sorry

end NUMINAMATH_CALUDE_evelyn_bottle_caps_l3100_310004


namespace NUMINAMATH_CALUDE_cupcake_packages_l3100_310022

/-- Given the initial number of cupcakes, the number eaten, and the number of cupcakes per package,
    calculate the number of full packages that can be made. -/
def fullPackages (initial : ℕ) (eaten : ℕ) (perPackage : ℕ) : ℕ :=
  (initial - eaten) / perPackage

/-- Theorem stating that with 60 initial cupcakes, 22 eaten, and 10 cupcakes per package,
    the number of full packages is 3. -/
theorem cupcake_packages : fullPackages 60 22 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_packages_l3100_310022


namespace NUMINAMATH_CALUDE_special_integer_pairs_l3100_310068

theorem special_integer_pairs (a b : ℕ+) :
  (∃ (p : ℕ) (k : ℕ), Prime p ∧ a^2 + b + 1 = p^k) →
  (a^2 + b + 1) ∣ (b^2 - a^3 - 1) →
  ¬((a^2 + b + 1) ∣ (a + b - 1)^2) →
  ∃ (s : ℕ), s ≥ 2 ∧ a = 2^s ∧ b = 2^(2*s) - 1 :=
by sorry

end NUMINAMATH_CALUDE_special_integer_pairs_l3100_310068


namespace NUMINAMATH_CALUDE_average_problem_l3100_310073

-- Define the average of two numbers
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

-- Define the average of four numbers
def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem average_problem :
  avg4 (avg2 1 2) (avg2 3 1) (avg2 2 0) (avg2 1 1) = 11 / 8 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l3100_310073


namespace NUMINAMATH_CALUDE_average_entry_exit_time_is_200_l3100_310065

/-- Represents the position and movement of a car and storm -/
structure CarStormSystem where
  carSpeed : ℝ
  stormRadius : ℝ
  stormSpeedSouth : ℝ
  stormSpeedEast : ℝ
  initialNorthDistance : ℝ

/-- Calculates the average of the times when the car enters and exits the storm -/
def averageEntryExitTime (system : CarStormSystem) : ℝ :=
  200

/-- Theorem stating that the average entry/exit time is 200 minutes -/
theorem average_entry_exit_time_is_200 (system : CarStormSystem) 
  (h1 : system.carSpeed = 1)
  (h2 : system.stormRadius = 60)
  (h3 : system.stormSpeedSouth = 3/4)
  (h4 : system.stormSpeedEast = 1/4)
  (h5 : system.initialNorthDistance = 150) :
  averageEntryExitTime system = 200 := by
  sorry

end NUMINAMATH_CALUDE_average_entry_exit_time_is_200_l3100_310065


namespace NUMINAMATH_CALUDE_spinster_cat_ratio_l3100_310086

theorem spinster_cat_ratio :
  ∀ (s c : ℕ),
    s = 12 →
    c = s + 42 →
    ∃ (n : ℕ), n * s = 2 * c ∧ 9 * s = n * c :=
by
  sorry

end NUMINAMATH_CALUDE_spinster_cat_ratio_l3100_310086


namespace NUMINAMATH_CALUDE_total_fat_is_3600_l3100_310099

/-- Represents the fat content of different fish types and the number of fish served -/
structure FishData where
  herring_fat : ℕ
  eel_fat : ℕ
  pike_fat_extra : ℕ
  fish_count : ℕ

/-- Calculates the total fat content from all fish served -/
def total_fat (data : FishData) : ℕ :=
  data.fish_count * data.herring_fat +
  data.fish_count * data.eel_fat +
  data.fish_count * (data.eel_fat + data.pike_fat_extra)

/-- Theorem stating that the total fat content is 3600 oz given the specific fish data -/
theorem total_fat_is_3600 (data : FishData)
  (h1 : data.herring_fat = 40)
  (h2 : data.eel_fat = 20)
  (h3 : data.pike_fat_extra = 10)
  (h4 : data.fish_count = 40) :
  total_fat data = 3600 := by
  sorry

end NUMINAMATH_CALUDE_total_fat_is_3600_l3100_310099
