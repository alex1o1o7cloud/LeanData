import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_quadratic_linear_l4101_410191

theorem unique_solution_quadratic_linear (m : ℝ) :
  (∃! x : ℝ, x^2 = 4*x + m) ↔ m = -4 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_linear_l4101_410191


namespace NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l4101_410141

theorem quadratic_equation_two_distinct_roots (k : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  x₁^2 - (k + 3) * x₁ + 2 * k + 1 = 0 ∧
  x₂^2 - (k + 3) * x₂ + 2 * k + 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l4101_410141


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l4101_410170

theorem rectangular_prism_volume (l w h : ℝ) : 
  l > 0 → w > 0 → h > 0 →
  l * w = 15 → w * h = 10 → l * h = 6 →
  l * w * h = 30 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l4101_410170


namespace NUMINAMATH_CALUDE_amys_flash_drive_storage_l4101_410168

/-- Calculates the total storage space used on Amy's flash drive -/
def total_storage_space (music_files : ℝ) (music_size : ℝ) (video_files : ℝ) (video_size : ℝ) (picture_files : ℝ) (picture_size : ℝ) : ℝ :=
  music_files * music_size + video_files * video_size + picture_files * picture_size

/-- Theorem: The total storage space used on Amy's flash drive is 1116 MB -/
theorem amys_flash_drive_storage :
  total_storage_space 4 5 21 50 23 2 = 1116 := by
  sorry

end NUMINAMATH_CALUDE_amys_flash_drive_storage_l4101_410168


namespace NUMINAMATH_CALUDE_min_swaps_at_most_five_l4101_410148

/-- Represents a 4026-digit number composed of ones and twos -/
structure NumberConfig :=
  (ones_count : Nat)
  (twos_count : Nat)
  (total_digits : Nat)
  (h1 : ones_count = 2013)
  (h2 : twos_count = 2013)
  (h3 : total_digits = 4026)
  (h4 : ones_count + twos_count = total_digits)

/-- Represents the state of the number after some swaps -/
structure NumberState :=
  (config : NumberConfig)
  (ones_in_odd : Nat)
  (h : ones_in_odd ≤ config.ones_count)

/-- Checks if a NumberState is divisible by 11 -/
def isDivisibleBy11 (state : NumberState) : Prop :=
  (state.config.total_digits - 2 * state.ones_in_odd) % 11 = 0

/-- The minimum number of swaps required to make the number divisible by 11 -/
def minSwapsToDiv11 (state : NumberState) : Nat :=
  min (state.ones_in_odd % 11) ((11 - state.ones_in_odd % 11) % 11)

/-- The main theorem stating that the minimum number of swaps is at most 5 -/
theorem min_swaps_at_most_five (state : NumberState) : minSwapsToDiv11 state ≤ 5 := by
  sorry


end NUMINAMATH_CALUDE_min_swaps_at_most_five_l4101_410148


namespace NUMINAMATH_CALUDE_cubic_inequality_range_l4101_410104

theorem cubic_inequality_range (m : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, m * x^3 - x^2 + 4*x + 3 ≥ 0) ↔ m ∈ Set.Icc (-6 : ℝ) (-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_range_l4101_410104


namespace NUMINAMATH_CALUDE_set_intersection_implies_values_l4101_410127

theorem set_intersection_implies_values (a b : ℤ) : 
  let A : Set ℤ := {1, b, a + b}
  let B : Set ℤ := {a - b, a * b}
  A ∩ B = {-1, 0} →
  a = -1 ∧ b = 0 := by
sorry

end NUMINAMATH_CALUDE_set_intersection_implies_values_l4101_410127


namespace NUMINAMATH_CALUDE_sum_234_142_in_base4_l4101_410154

/-- Converts a natural number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list of digits represents a valid base 4 number -/
def isValidBase4 (digits : List ℕ) : Prop :=
  sorry

theorem sum_234_142_in_base4 :
  let sum := 234 + 142
  let base4Sum := toBase4 sum
  isValidBase4 base4Sum ∧ base4Sum = [1, 1, 0, 3, 0] := by
  sorry

end NUMINAMATH_CALUDE_sum_234_142_in_base4_l4101_410154


namespace NUMINAMATH_CALUDE_tennis_ball_box_capacity_l4101_410121

theorem tennis_ball_box_capacity :
  ∀ (total_balls : ℕ) (box_capacity : ℕ),
  (4 * box_capacity - 8 = total_balls) →
  (3 * box_capacity + 4 = total_balls) →
  box_capacity = 12 := by
sorry

end NUMINAMATH_CALUDE_tennis_ball_box_capacity_l4101_410121


namespace NUMINAMATH_CALUDE_monica_savings_l4101_410109

def weekly_savings : ℕ := 15
def weeks_to_fill : ℕ := 60
def repetitions : ℕ := 5

theorem monica_savings : weekly_savings * weeks_to_fill * repetitions = 4500 := by
  sorry

end NUMINAMATH_CALUDE_monica_savings_l4101_410109


namespace NUMINAMATH_CALUDE_garden_length_l4101_410123

/-- A rectangular garden with given perimeter and breadth has a specific length. -/
theorem garden_length (perimeter breadth : ℝ) (h_perimeter : perimeter = 600) (h_breadth : breadth = 150) :
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter → perimeter / 2 - breadth = 150 := by
sorry

end NUMINAMATH_CALUDE_garden_length_l4101_410123


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l4101_410136

/-- Represents a right circular cone -/
structure Cone where
  diameter : ℝ
  altitude : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ

/-- Predicate to check if a cylinder is inscribed in a cone -/
def is_inscribed (cylinder : Cylinder) (cone : Cone) : Prop :=
  -- This is a placeholder for the actual geometric condition
  True

theorem inscribed_cylinder_radius (cone : Cone) (cylinder : Cylinder) :
  cone.diameter = 8 →
  cone.altitude = 10 →
  is_inscribed cylinder cone →
  cylinder.radius * 2 = cylinder.radius * 2 →  -- Diameter equals height
  cylinder.radius = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l4101_410136


namespace NUMINAMATH_CALUDE_impossible_all_tails_l4101_410198

/-- Represents a 4x4 grid of binary values -/
def Grid := Matrix (Fin 4) (Fin 4) Bool

/-- Represents the possible flip operations -/
inductive FlipOperation
| Row : Fin 4 → FlipOperation
| Column : Fin 4 → FlipOperation
| Diagonal : Bool → Fin 4 → FlipOperation

/-- Initial configuration of the grid -/
def initialGrid : Grid :=
  Matrix.of (fun i j => if i = 0 ∧ j < 2 then true else false)

/-- Applies a flip operation to the grid -/
def applyFlip (g : Grid) (op : FlipOperation) : Grid :=
  sorry

/-- Checks if all values in the grid are false (tails) -/
def allTails (g : Grid) : Prop :=
  ∀ i j, g i j = false

/-- Main theorem: It's impossible to reach all tails from the initial configuration -/
theorem impossible_all_tails :
  ¬∃ (ops : List FlipOperation), allTails (ops.foldl applyFlip initialGrid) :=
  sorry

end NUMINAMATH_CALUDE_impossible_all_tails_l4101_410198


namespace NUMINAMATH_CALUDE_tenth_student_problems_l4101_410105

theorem tenth_student_problems (n : ℕ) : 
  -- Total number of students
  (10 : ℕ) > 0 →
  -- Each problem is solved by exactly 7 students
  ∃ p : ℕ, p > 0 ∧ (7 * p = 36 + n) →
  -- First 9 students each solved 4 problems
  (9 * 4 = 36) →
  -- The number of problems solved by the tenth student is n
  n ≤ p →
  -- Conclusion: The tenth student solved 6 problems
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_tenth_student_problems_l4101_410105


namespace NUMINAMATH_CALUDE_factorization_difference_l4101_410153

theorem factorization_difference (a b : ℤ) : 
  (∀ y : ℤ, 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) → 
  a - b = -7 := by
sorry

end NUMINAMATH_CALUDE_factorization_difference_l4101_410153


namespace NUMINAMATH_CALUDE_fraction_not_going_on_trip_l4101_410165

theorem fraction_not_going_on_trip :
  ∀ (S : ℝ) (J : ℝ),
    S > 0 →
    J = (2/3) * S →
    ((3/4) * J + (1/3) * S) / (J + S) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_not_going_on_trip_l4101_410165


namespace NUMINAMATH_CALUDE_total_water_needed_is_112_l4101_410158

/-- Calculates the total gallons of water needed for Nicole's fish tanks in four weeks -/
def water_needed_in_four_weeks : ℕ :=
  let num_tanks : ℕ := 4
  let first_tank_gallons : ℕ := 8
  let num_first_type_tanks : ℕ := 2
  let num_second_type_tanks : ℕ := num_tanks - num_first_type_tanks
  let second_tank_gallons : ℕ := first_tank_gallons - 2
  let weeks : ℕ := 4
  
  let weekly_total : ℕ := 
    first_tank_gallons * num_first_type_tanks + 
    second_tank_gallons * num_second_type_tanks
  
  weekly_total * weeks

/-- Theorem stating that the total gallons of water needed in four weeks is 112 -/
theorem total_water_needed_is_112 : water_needed_in_four_weeks = 112 := by
  sorry

end NUMINAMATH_CALUDE_total_water_needed_is_112_l4101_410158


namespace NUMINAMATH_CALUDE_smallest_digit_for_divisibility_l4101_410147

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def number_with_d (d : ℕ) : ℕ := 437000 + d * 1000 + 3

theorem smallest_digit_for_divisibility :
  ∃ (d : ℕ), d < 10 ∧ 
    is_divisible_by_9 (number_with_d d) ∧ 
    (∀ (d' : ℕ), d' < d → ¬is_divisible_by_9 (number_with_d d')) ∧
    d = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_digit_for_divisibility_l4101_410147


namespace NUMINAMATH_CALUDE_angle_measure_l4101_410186

theorem angle_measure : ∃ x : ℝ, 
  (0 < x) ∧ (x < 90) ∧ (90 - x = 2 * x + 15) ∧ (x = 25) :=
by sorry

end NUMINAMATH_CALUDE_angle_measure_l4101_410186


namespace NUMINAMATH_CALUDE_weight_of_larger_square_l4101_410108

/-- Represents the properties of a square piece of wood -/
structure WoodSquare where
  side : ℝ
  weight : ℝ

/-- Theorem stating the relationship between two wood squares of different sizes -/
theorem weight_of_larger_square
  (small : WoodSquare)
  (large : WoodSquare)
  (h1 : small.side = 4)
  (h2 : small.weight = 16)
  (h3 : large.side = 6)
  (h4 : large.weight = (large.side^2 / small.side^2) * small.weight) :
  large.weight = 36 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_larger_square_l4101_410108


namespace NUMINAMATH_CALUDE_hilary_kernels_to_shuck_l4101_410162

/-- Calculates the total number of kernels Hilary has to shuck --/
def total_kernels (ears_per_stalk : ℕ) (num_stalks : ℕ) (kernels_first_half : ℕ) (additional_kernels_second_half : ℕ) : ℕ :=
  let total_ears := ears_per_stalk * num_stalks
  let ears_per_half := total_ears / 2
  let kernels_second_half := kernels_first_half + additional_kernels_second_half
  ears_per_half * kernels_first_half + ears_per_half * kernels_second_half

/-- Theorem stating that Hilary has 237,600 kernels to shuck --/
theorem hilary_kernels_to_shuck :
  total_kernels 4 108 500 100 = 237600 := by
  sorry

end NUMINAMATH_CALUDE_hilary_kernels_to_shuck_l4101_410162


namespace NUMINAMATH_CALUDE_fraction_decomposition_l4101_410159

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 4/3) :
  (7 * x - 13) / (3 * x^2 + 2 * x - 8) = 27 / (10 * (x + 2)) - 11 / (10 * (3 * x - 4)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l4101_410159


namespace NUMINAMATH_CALUDE_water_pouring_proof_l4101_410111

/-- Represents the fraction of water remaining after n pourings -/
def remainingWater (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

theorem water_pouring_proof :
  remainingWater 28 = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_water_pouring_proof_l4101_410111


namespace NUMINAMATH_CALUDE_basketball_team_selection_l4101_410100

theorem basketball_team_selection (n m k : ℕ) (h1 : n = 18) (h2 : m = 2) (h3 : k = 8) :
  Nat.choose (n - m) (k - m) = 8008 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l4101_410100


namespace NUMINAMATH_CALUDE_reflection_result_l4101_410184

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point across the line y = x - 2 -/
def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2 + 2, p.1 - 2)

/-- The final position of point C after two reflections -/
def C_double_prime : ℝ × ℝ :=
  reflect_line (reflect_y_axis (5, 3))

theorem reflection_result :
  C_double_prime = (5, -7) :=
by sorry

end NUMINAMATH_CALUDE_reflection_result_l4101_410184


namespace NUMINAMATH_CALUDE_basketball_team_starters_l4101_410188

def total_players : ℕ := 15
def quadruplets : ℕ := 4
def starters : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem basketball_team_starters :
  (quadruplets * choose (total_players - quadruplets) (starters - 1)) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_starters_l4101_410188


namespace NUMINAMATH_CALUDE_square_perimeter_l4101_410149

theorem square_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 520 → perimeter = 8 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l4101_410149


namespace NUMINAMATH_CALUDE_alicia_tax_deduction_l4101_410163

/-- Calculates the local tax deduction in cents per hour given an hourly wage in dollars and a tax rate percentage. -/
def local_tax_deduction (hourly_wage : ℚ) (tax_rate_percent : ℚ) : ℚ :=
  hourly_wage * 100 * (tax_rate_percent / 100)

/-- Theorem: Given Alicia's hourly wage of $25 and a local tax rate of 2%, 
    the amount deducted for local taxes is 50 cents per hour. -/
theorem alicia_tax_deduction :
  local_tax_deduction 25 2 = 50 := by
  sorry

#eval local_tax_deduction 25 2

end NUMINAMATH_CALUDE_alicia_tax_deduction_l4101_410163


namespace NUMINAMATH_CALUDE_only_two_reduces_to_zero_l4101_410155

/-- A move on a table is either subtracting n from a column or multiplying a row by n -/
inductive Move (n : ℕ+)
  | subtract_column : Move n
  | multiply_row : Move n

/-- A table is a rectangular array of positive integers -/
def Table := Array (Array ℕ+)

/-- Apply a move to a table -/
def apply_move (t : Table) (m : Move n) : Table :=
  sorry

/-- A table is reducible to zero if there exists a sequence of moves that makes all entries zero -/
def reducible_to_zero (t : Table) (n : ℕ+) : Prop :=
  sorry

/-- The main theorem: n = 2 is the only value that allows any table to be reduced to zero -/
theorem only_two_reduces_to_zero :
  ∀ n : ℕ+, (∀ t : Table, reducible_to_zero t n) ↔ n = 2 :=
  sorry

end NUMINAMATH_CALUDE_only_two_reduces_to_zero_l4101_410155


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l4101_410195

theorem fractional_equation_solution :
  ∃ (x : ℝ), (x ≠ 0 ∧ x + 1 ≠ 0) ∧ (1 / x = 2 / (x + 1)) ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l4101_410195


namespace NUMINAMATH_CALUDE_maricela_production_l4101_410133

/-- Represents the orange production and juice sale scenario of the Morales sisters. -/
structure OrangeGroveSale where
  trees_per_sister : ℕ
  gabriela_oranges_per_tree : ℕ
  alba_oranges_per_tree : ℕ
  oranges_per_cup : ℕ
  price_per_cup : ℚ
  total_revenue : ℚ

/-- Calculates the number of oranges Maricela's trees must produce per tree. -/
def maricela_oranges_per_tree (sale : OrangeGroveSale) : ℚ :=
  sorry

/-- Theorem stating that given the conditions, Maricela's trees must produce 500 oranges per tree. -/
theorem maricela_production (sale : OrangeGroveSale) 
  (h1 : sale.trees_per_sister = 110)
  (h2 : sale.gabriela_oranges_per_tree = 600)
  (h3 : sale.alba_oranges_per_tree = 400)
  (h4 : sale.oranges_per_cup = 3)
  (h5 : sale.price_per_cup = 4)
  (h6 : sale.total_revenue = 220000) :
  maricela_oranges_per_tree sale = 500 := by
  sorry

end NUMINAMATH_CALUDE_maricela_production_l4101_410133


namespace NUMINAMATH_CALUDE_factorial_8_divisors_l4101_410140

theorem factorial_8_divisors : Nat.card (Nat.divisors (Nat.factorial 8)) = 96 := by sorry

end NUMINAMATH_CALUDE_factorial_8_divisors_l4101_410140


namespace NUMINAMATH_CALUDE_equation_solution_l4101_410126

theorem equation_solution : ∃ x : ℝ, 2 * x - 3 = 7 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4101_410126


namespace NUMINAMATH_CALUDE_abc12_paths_l4101_410130

/-- Represents the number of adjacent letters or numerals --/
def adjacent_count (letter : Char) : Nat :=
  match letter with
  | 'A' => 2  -- Number of B's adjacent to A
  | 'B' => 3  -- Number of C's adjacent to each B
  | 'C' => 2  -- Number of 1's adjacent to each C
  | '1' => 1  -- Number of 2's adjacent to each 1
  | _   => 0  -- For any other character

/-- Calculates the total number of paths to spell ABC12 --/
def total_paths : Nat :=
  adjacent_count 'A' * adjacent_count 'B' * adjacent_count 'C' * adjacent_count '1'

/-- Theorem stating that the number of paths to spell ABC12 is 12 --/
theorem abc12_paths : total_paths = 12 := by
  sorry

end NUMINAMATH_CALUDE_abc12_paths_l4101_410130


namespace NUMINAMATH_CALUDE_monotonic_cubic_implies_a_geq_one_l4101_410185

/-- A function f: ℝ → ℝ is monotonic if it is either monotonically increasing or monotonically decreasing -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

/-- The main theorem: if f(x) = (1/3)x³ + x² + ax - 5 is monotonic for all real x, then a ≥ 1 -/
theorem monotonic_cubic_implies_a_geq_one (a : ℝ) :
  Monotonic (fun x => (1/3) * x^3 + x^2 + a*x - 5) → a ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_monotonic_cubic_implies_a_geq_one_l4101_410185


namespace NUMINAMATH_CALUDE_min_value_sum_product_l4101_410189

theorem min_value_sum_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l4101_410189


namespace NUMINAMATH_CALUDE_tom_hockey_games_this_year_l4101_410138

/-- The number of hockey games Tom went to this year -/
def games_this_year (total_games : ℕ) (last_year_games : ℕ) : ℕ :=
  total_games - last_year_games

/-- Theorem stating that Tom went to 4 hockey games this year -/
theorem tom_hockey_games_this_year :
  games_this_year 13 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tom_hockey_games_this_year_l4101_410138


namespace NUMINAMATH_CALUDE_mango_rate_calculation_l4101_410156

def grape_weight : ℝ := 7
def grape_rate : ℝ := 68
def mango_weight : ℝ := 9
def total_paid : ℝ := 908

theorem mango_rate_calculation :
  (total_paid - grape_weight * grape_rate) / mango_weight = 48 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_calculation_l4101_410156


namespace NUMINAMATH_CALUDE_find_number_l4101_410164

theorem find_number : ∃ x : ℝ, (x / 23 - 67) * 2 = 102 ∧ x = 2714 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l4101_410164


namespace NUMINAMATH_CALUDE_polynomial_form_theorem_l4101_410139

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The condition that the polynomial P satisfies for all real a, b, c -/
def SatisfiesCondition (P : RealPolynomial) : Prop :=
  ∀ (a b c : ℝ), a*b + b*c + c*a = 0 → 
    P (a-b) + P (b-c) + P (c-a) = 2 * P (a+b+c)

/-- The theorem stating the form of polynomials satisfying the condition -/
theorem polynomial_form_theorem (P : RealPolynomial) 
  (h : SatisfiesCondition P) : 
  ∃ (α β : ℝ), ∀ (x : ℝ), P x = α * x^4 + β * x^2 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_form_theorem_l4101_410139


namespace NUMINAMATH_CALUDE_sequence_next_terms_l4101_410157

def sequence1 : List ℕ := [7, 11, 19, 35]
def sequence2 : List ℕ := [1, 4, 9, 16, 25]

def next_in_sequence1 (seq : List ℕ) : ℕ :=
  let diffs := List.zipWith (·-·) (seq.tail) seq
  let last_diff := diffs.getLast!
  seq.getLast! + (2 * last_diff)

def next_in_sequence2 (seq : List ℕ) : ℕ :=
  (seq.length + 1) ^ 2

theorem sequence_next_terms :
  next_in_sequence1 sequence1 = 67 ∧ next_in_sequence2 sequence2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sequence_next_terms_l4101_410157


namespace NUMINAMATH_CALUDE_sample_capacity_l4101_410182

/-- Given a sample divided into groups, prove that the sample capacity is 160
    when a certain group has a frequency of 20 and a frequency rate of 0.125. -/
theorem sample_capacity (n : ℕ) (frequency : ℕ) (frequency_rate : ℚ)
  (h1 : frequency = 20)
  (h2 : frequency_rate = 1/8)
  (h3 : (frequency : ℚ) / n = frequency_rate) :
  n = 160 := by
  sorry

end NUMINAMATH_CALUDE_sample_capacity_l4101_410182


namespace NUMINAMATH_CALUDE_particle_final_position_l4101_410146

/-- Represents the position of a particle -/
structure Position where
  x : Int
  y : Int

/-- Calculates the position of the particle after n steps -/
def particle_position (n : Nat) : Position :=
  sorry

/-- The number of complete rectangles after 2023 minutes -/
def complete_rectangles : Nat :=
  sorry

/-- The remaining time after completing the rectangles -/
def remaining_time : Nat :=
  sorry

theorem particle_final_position :
  particle_position (complete_rectangles + 1) = Position.mk 44 1 :=
sorry

end NUMINAMATH_CALUDE_particle_final_position_l4101_410146


namespace NUMINAMATH_CALUDE_tuesday_attendance_proof_l4101_410187

/-- The number of people who attended class on Tuesday -/
def tuesday_attendance : ℕ := sorry

/-- The number of people who attended class on Monday -/
def monday_attendance : ℕ := 10

/-- The number of people who attended class on Wednesday, Thursday, and Friday -/
def wednesday_to_friday_attendance : ℕ := 10

/-- The total number of days -/
def total_days : ℕ := 5

/-- The average attendance over all days -/
def average_attendance : ℕ := 11

theorem tuesday_attendance_proof :
  tuesday_attendance = 15 :=
by
  have h1 : monday_attendance + tuesday_attendance + 3 * wednesday_to_friday_attendance = average_attendance * total_days :=
    sorry
  sorry

end NUMINAMATH_CALUDE_tuesday_attendance_proof_l4101_410187


namespace NUMINAMATH_CALUDE_sqrt_eight_equals_two_sqrt_two_l4101_410196

theorem sqrt_eight_equals_two_sqrt_two : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_equals_two_sqrt_two_l4101_410196


namespace NUMINAMATH_CALUDE_lineArrangements_eq_36_l4101_410180

/-- The number of ways to arrange 3 students (who must stand together) and 2 teachers in a line -/
def lineArrangements : ℕ :=
  let studentsCount : ℕ := 3
  let teachersCount : ℕ := 2
  let unitsCount : ℕ := teachersCount + 1  -- Students count as one unit
  (Nat.factorial unitsCount) * (Nat.factorial studentsCount)

theorem lineArrangements_eq_36 : lineArrangements = 36 := by
  sorry

end NUMINAMATH_CALUDE_lineArrangements_eq_36_l4101_410180


namespace NUMINAMATH_CALUDE_marks_animals_legs_l4101_410178

/-- The number of legs of all animals owned by Mark -/
def total_legs (num_kangaroos : ℕ) (num_goats : ℕ) : ℕ :=
  2 * num_kangaroos + 4 * num_goats

/-- Theorem stating the total number of legs of Mark's animals -/
theorem marks_animals_legs : 
  let num_kangaroos : ℕ := 23
  let num_goats : ℕ := 3 * num_kangaroos
  total_legs num_kangaroos num_goats = 322 := by
  sorry

#check marks_animals_legs

end NUMINAMATH_CALUDE_marks_animals_legs_l4101_410178


namespace NUMINAMATH_CALUDE_aquafaba_for_angel_food_cakes_l4101_410122

/-- Proves that the number of tablespoons of aquafaba needed for two angel food cakes is 32 -/
theorem aquafaba_for_angel_food_cakes 
  (aquafaba_per_egg : ℕ) 
  (cakes : ℕ) 
  (egg_whites_per_cake : ℕ) 
  (h1 : aquafaba_per_egg = 2)
  (h2 : cakes = 2)
  (h3 : egg_whites_per_cake = 8) : 
  aquafaba_per_egg * cakes * egg_whites_per_cake = 32 :=
by sorry

end NUMINAMATH_CALUDE_aquafaba_for_angel_food_cakes_l4101_410122


namespace NUMINAMATH_CALUDE_count_valid_removal_sequences_for_specific_arrangement_l4101_410129

/-- Represents the arrangement of bricks -/
inductive BrickArrangement
| Empty : BrickArrangement
| Add : BrickArrangement → Nat → BrickArrangement

/-- Checks if a removal sequence is valid for a given arrangement -/
def isValidRemovalSequence (arrangement : BrickArrangement) (sequence : List Nat) : Prop := sorry

/-- Counts the number of valid removal sequences for a given arrangement -/
def countValidRemovalSequences (arrangement : BrickArrangement) : Nat := sorry

/-- The specific arrangement of 6 bricks as described in the problem -/
def specificArrangement : BrickArrangement := sorry

theorem count_valid_removal_sequences_for_specific_arrangement :
  countValidRemovalSequences specificArrangement = 10 := by sorry

end NUMINAMATH_CALUDE_count_valid_removal_sequences_for_specific_arrangement_l4101_410129


namespace NUMINAMATH_CALUDE_probability_5_odd_in_8_rolls_l4101_410115

def roll_die_8_times : ℕ := 8
def fair_die_sides : ℕ := 6
def odd_outcomes : ℕ := 3
def target_odd_rolls : ℕ := 5

theorem probability_5_odd_in_8_rolls : 
  (Nat.choose roll_die_8_times target_odd_rolls * (odd_outcomes ^ target_odd_rolls) * ((fair_die_sides - odd_outcomes) ^ (roll_die_8_times - target_odd_rolls))) / (fair_die_sides ^ roll_die_8_times) = 7 / 32 :=
sorry

end NUMINAMATH_CALUDE_probability_5_odd_in_8_rolls_l4101_410115


namespace NUMINAMATH_CALUDE_sum_of_cubes_plus_linear_positive_l4101_410151

theorem sum_of_cubes_plus_linear_positive
  (a b c : ℝ)
  (hab : a + b > 0)
  (hac : a + c > 0)
  (hbc : b + c > 0) :
  (a^3 + a) + (b^3 + b) + (c^3 + c) > 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_plus_linear_positive_l4101_410151


namespace NUMINAMATH_CALUDE_equation_solution_l4101_410143

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = 1/3 ∧ 
  (∀ x : ℝ, (x - 1)^2 + 2*x*(x - 1) = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4101_410143


namespace NUMINAMATH_CALUDE_number_of_newborns_l4101_410116

/-- Proves the number of newborns in a children's home --/
theorem number_of_newborns (total_children teenagers toddlers newborns : ℕ) : 
  total_children = 40 →
  teenagers = 5 * toddlers →
  toddlers = 6 →
  total_children = teenagers + toddlers + newborns →
  newborns = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_newborns_l4101_410116


namespace NUMINAMATH_CALUDE_smallest_difference_l4101_410132

def Digits : Finset ℕ := {2, 3, 4, 5, 6, 7, 8, 9}

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧ (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.image (λ i => (n / 10^i) % 10) {0,1,2,3})) = 4)

def valid_pair (a b : ℕ) : Prop :=
  is_valid_number a ∧ is_valid_number b ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.image (λ i => (a / 10^i) % 10) {0,1,2,3} ∪ Finset.image (λ i => (b / 10^i) % 10) {0,1,2,3})) = 8)

theorem smallest_difference :
  ∃ (a b : ℕ), valid_pair a b ∧
    (a > b) ∧
    (a - b = 247) ∧
    (∀ (c d : ℕ), valid_pair c d → c > d → c - d ≥ 247) :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_l4101_410132


namespace NUMINAMATH_CALUDE_parallelogram_distance_l4101_410167

/-- Given a parallelogram with base 10 feet, height 30 feet, and side length 60 feet,
    prove that the distance between the 60-foot sides is 5 feet. -/
theorem parallelogram_distance (base height side : ℝ) 
  (h_base : base = 10)
  (h_height : height = 30)
  (h_side : side = 60) :
  (base * height) / side = 5 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_distance_l4101_410167


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l4101_410199

theorem ratio_of_sum_and_difference (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (h_sum_diff : a + b = 4 * (a - b)) : a / b = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l4101_410199


namespace NUMINAMATH_CALUDE_prob_even_sum_is_11_20_l4101_410173

/-- Represents a wheel with a certain number of even and odd sections -/
structure Wheel where
  total : ℕ
  even : ℕ
  odd : ℕ
  valid : total = even + odd

/-- The probability of getting an even number on a wheel -/
def prob_even (w : Wheel) : ℚ :=
  w.even / w.total

/-- The probability of getting an odd number on a wheel -/
def prob_odd (w : Wheel) : ℚ :=
  w.odd / w.total

/-- The two wheels in the game -/
def wheel1 : Wheel := ⟨5, 2, 3, rfl⟩
def wheel2 : Wheel := ⟨4, 1, 3, rfl⟩

/-- The theorem to be proved -/
theorem prob_even_sum_is_11_20 :
  prob_even wheel1 * prob_even wheel2 + prob_odd wheel1 * prob_odd wheel2 = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_sum_is_11_20_l4101_410173


namespace NUMINAMATH_CALUDE_tim_speed_proof_l4101_410160

/-- Represents the initial distance between Tim and Élan in miles -/
def initial_distance : ℝ := 180

/-- Represents Élan's initial speed in mph -/
def elan_initial_speed : ℝ := 5

/-- Represents the distance Tim travels until meeting Élan in miles -/
def tim_travel_distance : ℝ := 120

/-- Represents Tim's initial speed in mph -/
def tim_initial_speed : ℝ := 40

theorem tim_speed_proof :
  ∃ (t : ℝ), 
    t > 0 ∧
    t + 2*t = tim_travel_distance ∧
    t = tim_initial_speed :=
by sorry

end NUMINAMATH_CALUDE_tim_speed_proof_l4101_410160


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l4101_410134

theorem quadratic_roots_problem (p : ℤ) : 
  (∃ u v : ℤ, u > 0 ∧ v > 0 ∧ 
   5 * u^2 - 5 * p * u + (66 * p - 1) = 0 ∧
   5 * v^2 - 5 * p * v + (66 * p - 1) = 0) →
  p = 76 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l4101_410134


namespace NUMINAMATH_CALUDE_investment_percentage_l4101_410192

/-- Proves that given the investment conditions, the percentage of the other investment is 7% -/
theorem investment_percentage (total_investment : ℝ) (investment_at_8_percent : ℝ) (total_interest : ℝ)
  (h1 : total_investment = 22000)
  (h2 : investment_at_8_percent = 17000)
  (h3 : total_interest = 1710) :
  (total_interest - investment_at_8_percent * 0.08) / (total_investment - investment_at_8_percent) = 0.07 := by
  sorry


end NUMINAMATH_CALUDE_investment_percentage_l4101_410192


namespace NUMINAMATH_CALUDE_star_two_three_solve_equation_l4101_410113

-- Define the new operation
def star (a b : ℝ) : ℝ := a^2 + 2*a*b

-- Theorem 1
theorem star_two_three : star 2 3 = 16 := by sorry

-- Theorem 2
theorem solve_equation (x : ℝ) : star (-2) x = -2 + x → x = 6/5 := by sorry

end NUMINAMATH_CALUDE_star_two_three_solve_equation_l4101_410113


namespace NUMINAMATH_CALUDE_distribute_5_3_l4101_410124

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_5_3 : distribute 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l4101_410124


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l4101_410142

/-- Calculates the time for a train to pass a platform given its length, time to cross a tree, and platform length -/
theorem train_platform_passing_time
  (train_length : ℝ)
  (tree_crossing_time : ℝ)
  (platform_length : ℝ)
  (h1 : train_length = 2000)
  (h2 : tree_crossing_time = 80)
  (h3 : platform_length = 1200) :
  (train_length + platform_length) / (train_length / tree_crossing_time) = 128 :=
by sorry

end NUMINAMATH_CALUDE_train_platform_passing_time_l4101_410142


namespace NUMINAMATH_CALUDE_square_difference_l4101_410118

theorem square_difference : 100^2 - 2 * 100 * 99 + 99^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l4101_410118


namespace NUMINAMATH_CALUDE_grouping_theorem_l4101_410197

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to divide 4 men and 3 women into a group of five 
    (with at least two men and two women) and a group of two -/
def groupingWays : ℕ :=
  choose 4 2 * choose 3 2 * choose 3 1

theorem grouping_theorem : groupingWays = 54 := by sorry

end NUMINAMATH_CALUDE_grouping_theorem_l4101_410197


namespace NUMINAMATH_CALUDE_fast_food_cost_l4101_410174

/-- The cost of items at a fast food restaurant -/
theorem fast_food_cost (H M F : ℝ) : 
  (3 * H + 5 * M + F = 23.50) → 
  (5 * H + 9 * M + F = 39.50) → 
  (2 * H + 2 * M + 2 * F = 15.00) :=
by sorry

end NUMINAMATH_CALUDE_fast_food_cost_l4101_410174


namespace NUMINAMATH_CALUDE_third_wall_length_l4101_410181

/-- Calculates the length of the third wall in a hall of mirrors. -/
theorem third_wall_length
  (total_glass : ℝ)
  (wall1_length wall1_height : ℝ)
  (wall2_length wall2_height : ℝ)
  (wall3_height : ℝ)
  (h1 : total_glass = 960)
  (h2 : wall1_length = 30 ∧ wall1_height = 12)
  (h3 : wall2_length = 30 ∧ wall2_height = 12)
  (h4 : wall3_height = 12)
  : ∃ (wall3_length : ℝ),
    total_glass = wall1_length * wall1_height + wall2_length * wall2_height + wall3_length * wall3_height
    ∧ wall3_length = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_third_wall_length_l4101_410181


namespace NUMINAMATH_CALUDE_integer_roots_cubic_l4101_410169

theorem integer_roots_cubic (b : ℤ) : 
  (∃ x : ℤ, x^3 - 2*x^2 + b*x + 6 = 0) ↔ b ∈ ({-25, -7, -5, 3, 13, 47} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_roots_cubic_l4101_410169


namespace NUMINAMATH_CALUDE_range_of_m_l4101_410128

/-- The set of x satisfying the condition p -/
def P : Set ℝ := {x | (x + 2) / (x - 10) ≤ 0}

/-- The set of x satisfying the condition q for a given m -/
def Q (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 < 0}

/-- p is a necessary but not sufficient condition for q -/
def NecessaryNotSufficient (m : ℝ) : Prop :=
  (∀ x, x ∈ Q m → x ∈ P) ∧ (∃ x, x ∈ P ∧ x ∉ Q m)

/-- The main theorem stating the range of m -/
theorem range_of_m :
  ∀ m, m > 0 → (NecessaryNotSufficient m ↔ m < 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l4101_410128


namespace NUMINAMATH_CALUDE_simplify_expression_l4101_410172

theorem simplify_expression (r : ℝ) : 150 * r - 70 * r + 25 = 80 * r + 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4101_410172


namespace NUMINAMATH_CALUDE_water_formation_l4101_410102

-- Define the molecules and their quantities
def HCl_moles : ℕ := 1
def NaHCO3_moles : ℕ := 1

-- Define the reaction equation
def reaction_equation : String := "HCl + NaHCO3 → NaCl + H2O + CO2"

-- Define the function to calculate water moles produced
def water_moles_produced (hcl : ℕ) (nahco3 : ℕ) : ℕ :=
  min hcl nahco3

-- Theorem statement
theorem water_formation :
  water_moles_produced HCl_moles NaHCO3_moles = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_water_formation_l4101_410102


namespace NUMINAMATH_CALUDE_largest_rhombus_diagonal_l4101_410166

/-- The diagonal of the largest rhombus inscribed in a circle with radius 10 cm is 20 cm. -/
theorem largest_rhombus_diagonal (r : ℝ) (h : r = 10) : 
  2 * r = 20 := by sorry

end NUMINAMATH_CALUDE_largest_rhombus_diagonal_l4101_410166


namespace NUMINAMATH_CALUDE_pipe_fill_time_l4101_410117

/-- The time it takes for a pipe to fill a tank without a leak, given:
  1. With the leak, it takes 12 hours to fill the tank.
  2. The leak alone can empty the full tank in 12 hours. -/
def fill_time_without_leak : ℝ := 6

/-- The time it takes to fill the tank with both the pipe and leak working -/
def fill_time_with_leak : ℝ := 12

/-- The time it takes for the leak to empty a full tank -/
def leak_empty_time : ℝ := 12

theorem pipe_fill_time :
  fill_time_without_leak = 6 ∧
  (1 / fill_time_without_leak - 1 / leak_empty_time = 1 / fill_time_with_leak) :=
sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l4101_410117


namespace NUMINAMATH_CALUDE_vertex_angle_and_side_not_determine_equilateral_l4101_410144

/-- A triangle with side lengths a, b, c and angles A, B, C (opposite to sides a, b, c respectively) -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- An equilateral triangle is a triangle with all sides equal -/
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- A vertex angle is any of the three angles in a triangle -/
def IsVertexAngle (t : Triangle) (angle : ℝ) : Prop :=
  angle = t.A ∨ angle = t.B ∨ angle = t.C

/-- Statement: Knowing a vertex angle and a side length is not sufficient to uniquely determine an equilateral triangle -/
theorem vertex_angle_and_side_not_determine_equilateral :
  ∃ (t1 t2 : Triangle) (angle side : ℝ),
    IsVertexAngle t1 angle ∧
    IsVertexAngle t2 angle ∧
    (t1.a = side ∨ t1.b = side ∨ t1.c = side) ∧
    (t2.a = side ∨ t2.b = side ∨ t2.c = side) ∧
    IsEquilateral t1 ∧
    ¬IsEquilateral t2 :=
  sorry

end NUMINAMATH_CALUDE_vertex_angle_and_side_not_determine_equilateral_l4101_410144


namespace NUMINAMATH_CALUDE_pens_to_pencils_ratio_l4101_410101

/-- Represents the contents of Tommy's pencil case -/
structure PencilCase where
  total : Nat
  pencils : Nat
  eraser : Nat
  pens : Nat

/-- Theorem stating the ratio of pens to pencils in Tommy's pencil case -/
theorem pens_to_pencils_ratio (case : PencilCase) 
  (h_total : case.total = 13)
  (h_pencils : case.pencils = 4)
  (h_eraser : case.eraser = 1)
  (h_sum : case.total = case.pencils + case.pens + case.eraser)
  (h_multiple : ∃ k : Nat, case.pens = k * case.pencils) :
  case.pens / case.pencils = 2 := by
  sorry

end NUMINAMATH_CALUDE_pens_to_pencils_ratio_l4101_410101


namespace NUMINAMATH_CALUDE_jake_weight_loss_l4101_410179

theorem jake_weight_loss (total_weight : ℕ) (jake_weight : ℕ) (weight_loss : ℕ) : 
  total_weight = 290 → 
  jake_weight = 196 → 
  jake_weight - weight_loss = 2 * (total_weight - jake_weight) → 
  weight_loss = 8 := by
  sorry

end NUMINAMATH_CALUDE_jake_weight_loss_l4101_410179


namespace NUMINAMATH_CALUDE_triangle_area_l4101_410110

/-- The area of a triangle with base 9 cm and height 12 cm is 54 cm². -/
theorem triangle_area : 
  ∀ (base height area : ℝ), 
  base = 9 → height = 12 → area = (1/2) * base * height → 
  area = 54 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l4101_410110


namespace NUMINAMATH_CALUDE_complementary_angle_theorem_l4101_410125

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary_angles (α β : ℝ) : Prop := α + β = 90

/-- Given complementary angles α and β where α = 40°, prove that β = 50° -/
theorem complementary_angle_theorem (α β : ℝ) 
  (h1 : complementary_angles α β) (h2 : α = 40) : β = 50 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angle_theorem_l4101_410125


namespace NUMINAMATH_CALUDE_custom_operation_properties_l4101_410193

-- Define the custom operation *
noncomputable def customMul (x y : ℝ) : ℝ :=
  if x = 0 then |y|
  else if y = 0 then |x|
  else if (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) then |x| + |y|
  else -(|x| + |y|)

-- Theorem statement
theorem custom_operation_properties :
  (∀ a : ℝ, customMul (-15) (customMul 3 0) = -18) ∧
  (∀ a : ℝ, 
    (a < 0 → customMul 3 a + a = 2 * a - 3) ∧
    (a = 0 → customMul 3 a + a = 3) ∧
    (a > 0 → customMul 3 a + a = 2 * a + 3)) :=
by sorry

end NUMINAMATH_CALUDE_custom_operation_properties_l4101_410193


namespace NUMINAMATH_CALUDE_fitness_center_member_ratio_l4101_410137

theorem fitness_center_member_ratio 
  (avg_female : ℝ) 
  (avg_male : ℝ) 
  (avg_total : ℝ) 
  (h1 : avg_female = 140) 
  (h2 : avg_male = 180) 
  (h3 : avg_total = 160) :
  ∃ (f m : ℝ), f > 0 ∧ m > 0 ∧ f / m = 1 ∧
  (f * avg_female + m * avg_male) / (f + m) = avg_total :=
by sorry

end NUMINAMATH_CALUDE_fitness_center_member_ratio_l4101_410137


namespace NUMINAMATH_CALUDE_flagpole_height_is_8_l4101_410145

/-- The height of the flagpole in meters. -/
def flagpole_height : ℝ := 8

/-- The length of the rope in meters. -/
def rope_length : ℝ := flagpole_height + 2

/-- The distance the rope is pulled away from the flagpole in meters. -/
def pull_distance : ℝ := 6

theorem flagpole_height_is_8 :
  flagpole_height = 8 ∧
  rope_length = flagpole_height + 2 ∧
  flagpole_height ^ 2 + pull_distance ^ 2 = rope_length ^ 2 :=
sorry

end NUMINAMATH_CALUDE_flagpole_height_is_8_l4101_410145


namespace NUMINAMATH_CALUDE_function_negative_on_interval_l4101_410152

/-- The function f(x) = x^2 + mx - 1 is negative on [m, m+1] iff m is in (-√2/2, 0) -/
theorem function_negative_on_interval (m : ℝ) : 
  (∀ x ∈ Set.Icc m (m + 1), x^2 + m*x - 1 < 0) ↔ 
  m ∈ Set.Ioo (-(Real.sqrt 2)/2) 0 :=
sorry

end NUMINAMATH_CALUDE_function_negative_on_interval_l4101_410152


namespace NUMINAMATH_CALUDE_max_volume_parallelepiped_l4101_410150

/-- The volume of a rectangular parallelepiped with square base of side length x
    and lateral faces with perimeter 6 -/
def volume (x : ℝ) : ℝ := x^2 * (3 - x)

/-- The maximum volume of a rectangular parallelepiped with square base
    and lateral faces with perimeter 6 is 4 -/
theorem max_volume_parallelepiped :
  ∃ (x : ℝ), x > 0 ∧ x < 3 ∧
  (∀ (y : ℝ), y > 0 → y < 3 → volume y ≤ volume x) ∧
  volume x = 4 := by sorry

end NUMINAMATH_CALUDE_max_volume_parallelepiped_l4101_410150


namespace NUMINAMATH_CALUDE_quadratic_b_value_l4101_410119

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := 2 * x^2 - b * x - 1

-- Define the property of passing through two points with the same y-coordinate
def passes_through (b : ℝ) : Prop :=
  ∃ y₀ : ℝ, f b 3 = y₀ ∧ f b 9 = y₀

-- Theorem statement
theorem quadratic_b_value :
  ∀ b : ℝ, passes_through b → b = 24 := by sorry

end NUMINAMATH_CALUDE_quadratic_b_value_l4101_410119


namespace NUMINAMATH_CALUDE_gauss_family_mean_age_is_correct_l4101_410114

/-- The mean age of the Gauss family children -/
def gauss_family_mean_age : ℚ :=
  let ages : List ℕ := [7, 7, 8, 14, 12, 15, 16]
  (ages.sum : ℚ) / ages.length

/-- Theorem stating that the mean age of the Gauss family children is 79/7 -/
theorem gauss_family_mean_age_is_correct : gauss_family_mean_age = 79 / 7 := by
  sorry

end NUMINAMATH_CALUDE_gauss_family_mean_age_is_correct_l4101_410114


namespace NUMINAMATH_CALUDE_parallel_planes_lines_l4101_410194

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_parallel_line : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_lines 
  (α β : Plane) (m n : Line) :
  parallel α β →
  line_parallel_plane m α →
  line_parallel_line n m →
  ¬ line_in_plane n β →
  line_parallel_plane n β :=
by sorry

end NUMINAMATH_CALUDE_parallel_planes_lines_l4101_410194


namespace NUMINAMATH_CALUDE_probability_all_truth_l4101_410135

theorem probability_all_truth (pA pB pC : ℝ) 
  (hA : 0 ≤ pA ∧ pA ≤ 1) 
  (hB : 0 ≤ pB ∧ pB ≤ 1) 
  (hC : 0 ≤ pC ∧ pC ≤ 1) 
  (hpA : pA = 0.8) 
  (hpB : pB = 0.6) 
  (hpC : pC = 0.75) : 
  pA * pB * pC = 0.27 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_truth_l4101_410135


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l4101_410120

/-- Definition of a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point -/
def given_point : Point2D :=
  { x := 3, y := -4 }

/-- Theorem: The given point lies in the fourth quadrant -/
theorem point_in_fourth_quadrant :
  fourth_quadrant given_point := by
  sorry


end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l4101_410120


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l4101_410112

theorem coefficient_x_cubed_in_expansion :
  let n : ℕ := 5
  let a : ℚ := 1
  let b : ℚ := 2
  let r : ℕ := 3
  let binomial_coeff := Nat.choose n r
  binomial_coeff * b^r = 80 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l4101_410112


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l4101_410131

/-- The quadratic equation (k-1)x^2 + 2x - 2 = 0 has two distinct real roots if and only if k > 1/2 and k ≠ 1 -/
theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (k - 1) * x₁^2 + 2 * x₁ - 2 = 0 ∧ (k - 1) * x₂^2 + 2 * x₂ - 2 = 0) ↔ 
  (k > 1/2 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l4101_410131


namespace NUMINAMATH_CALUDE_sixteen_points_configuration_unique_configuration_l4101_410106

/-- Represents a configuration of points on a line -/
structure LineConfiguration where
  totalPoints : ℕ
  pointA : ℕ
  pointB : ℕ

/-- Counts the number of segments that contain a given point -/
def segmentsContainingPoint (config : LineConfiguration) (point : ℕ) : ℕ :=
  (point - 1) * (config.totalPoints - point)

/-- The main theorem stating that the configuration with 16 points satisfies the given conditions -/
theorem sixteen_points_configuration :
  ∃ (config : LineConfiguration),
    config.totalPoints = 16 ∧
    segmentsContainingPoint config config.pointA = 50 ∧
    segmentsContainingPoint config config.pointB = 56 := by
  sorry

/-- Uniqueness theorem: there is only one configuration satisfying the conditions -/
theorem unique_configuration (config1 config2 : LineConfiguration) :
  segmentsContainingPoint config1 config1.pointA = 50 →
  segmentsContainingPoint config1 config1.pointB = 56 →
  segmentsContainingPoint config2 config2.pointA = 50 →
  segmentsContainingPoint config2 config2.pointB = 56 →
  config1.totalPoints = config2.totalPoints := by
  sorry

end NUMINAMATH_CALUDE_sixteen_points_configuration_unique_configuration_l4101_410106


namespace NUMINAMATH_CALUDE_rectangle_ratio_l4101_410183

theorem rectangle_ratio (w : ℝ) : 
  w > 0 → 
  2 * w + 2 * 10 = 30 → 
  w / 10 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l4101_410183


namespace NUMINAMATH_CALUDE_ways_to_top_teaching_building_l4101_410171

/-- A building with multiple floors and staircases -/
structure Building where
  floors : ℕ
  staircases_per_floor : ℕ

/-- The number of ways to go from the bottom floor to the top floor -/
def ways_to_top (b : Building) : ℕ :=
  b.staircases_per_floor ^ (b.floors - 1)

/-- The specific building in the problem -/
def teaching_building : Building :=
  { floors := 5, staircases_per_floor := 2 }

theorem ways_to_top_teaching_building :
  ways_to_top teaching_building = 2^4 := by
  sorry

#eval ways_to_top teaching_building

end NUMINAMATH_CALUDE_ways_to_top_teaching_building_l4101_410171


namespace NUMINAMATH_CALUDE_student_distribution_l4101_410161

theorem student_distribution (n : ℕ) (k : ℕ) (m : ℕ) (h1 : n = 12) (h2 : k = 3) (h3 : m = 4) :
  (Nat.choose n m) * (Nat.choose (n - m) m) * (Nat.choose (n - 2*m) m) = (Nat.choose n m) * (Nat.choose (n - m) m) * 1 :=
sorry

end NUMINAMATH_CALUDE_student_distribution_l4101_410161


namespace NUMINAMATH_CALUDE_translation_result_l4101_410177

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point left by a given amount -/
def translateLeft (p : Point) (d : ℝ) : Point :=
  { x := p.x - d, y := p.y }

/-- Translate a point up by a given amount -/
def translateUp (p : Point) (d : ℝ) : Point :=
  { x := p.x, y := p.y + d }

/-- The initial point A -/
def A : Point :=
  { x := 2, y := 3 }

/-- The final point after translation -/
def finalPoint : Point :=
  translateUp (translateLeft A 3) 2

theorem translation_result :
  finalPoint = { x := -1, y := 5 } := by sorry

end NUMINAMATH_CALUDE_translation_result_l4101_410177


namespace NUMINAMATH_CALUDE_club_officer_selection_l4101_410190

theorem club_officer_selection (total_members : Nat) (boys : Nat) (girls : Nat)
  (h1 : total_members = boys + girls)
  (h2 : boys = 18)
  (h3 : girls = 12)
  (h4 : boys > 0)
  (h5 : girls > 0) :
  boys * girls = 216 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_l4101_410190


namespace NUMINAMATH_CALUDE_expression_value_l4101_410176

theorem expression_value : 
  (45 + (23 / 89) * Real.sin (π / 6)) * (4 * (3 ^ 2) - 7 * ((-2) ^ 3)) = 4186 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l4101_410176


namespace NUMINAMATH_CALUDE_function_defined_range_l4101_410107

open Real

theorem function_defined_range (a : ℝ) :
  (∀ x ∈ Set.Iic 1, (1 + 2^x + 4^x * a) / 3 > 0) ↔ a > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_function_defined_range_l4101_410107


namespace NUMINAMATH_CALUDE_weight_order_l4101_410103

/-- Conversion factor from kilograms to grams -/
def kg_to_g : ℕ → ℕ := (· * 1000)

/-- Conversion factor from tonnes to grams -/
def t_to_g : ℕ → ℕ := (· * 1000000)

/-- Weight in grams -/
def weight_908g : ℕ := 908

/-- Weight in grams (9kg80g) -/
def weight_9kg80g : ℕ := kg_to_g 9 + 80

/-- Weight in grams (900kg) -/
def weight_900kg : ℕ := kg_to_g 900

/-- Weight in grams (0.09t) -/
def weight_009t : ℕ := t_to_g 0 + 90000

theorem weight_order :
  weight_908g < weight_9kg80g ∧
  weight_9kg80g < weight_009t ∧
  weight_009t < weight_900kg := by
  sorry

end NUMINAMATH_CALUDE_weight_order_l4101_410103


namespace NUMINAMATH_CALUDE_total_glows_is_569_l4101_410175

/-- The number of seconds between 1:57:58 am and 3:20:47 am -/
def time_duration : ℕ := 4969

/-- The interval at which Light A glows, in seconds -/
def light_a_interval : ℕ := 16

/-- The interval at which Light B glows, in seconds -/
def light_b_interval : ℕ := 35

/-- The interval at which Light C glows, in seconds -/
def light_c_interval : ℕ := 42

/-- The number of times Light A glows -/
def light_a_glows : ℕ := time_duration / light_a_interval

/-- The number of times Light B glows -/
def light_b_glows : ℕ := time_duration / light_b_interval

/-- The number of times Light C glows -/
def light_c_glows : ℕ := time_duration / light_c_interval

/-- The total number of glows for all light sources combined -/
def total_glows : ℕ := light_a_glows + light_b_glows + light_c_glows

theorem total_glows_is_569 : total_glows = 569 := by
  sorry

end NUMINAMATH_CALUDE_total_glows_is_569_l4101_410175
