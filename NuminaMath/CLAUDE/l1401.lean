import Mathlib

namespace trinomial_zeros_l1401_140194

theorem trinomial_zeros (a b : ℝ) (ha : a > 4) (hb : b > 4) :
  (a^2 - 4*b > 0) ∨ (b^2 - 4*a > 0) := by sorry

end trinomial_zeros_l1401_140194


namespace square_difference_of_sum_and_difference_l1401_140113

theorem square_difference_of_sum_and_difference (x y : ℝ) 
  (h_sum : x + y = 20) (h_diff : x - y = 8) : x^2 - y^2 = 160 := by
  sorry

end square_difference_of_sum_and_difference_l1401_140113


namespace square_areas_sum_l1401_140127

theorem square_areas_sum (a : ℝ) (h1 : a > 0) (h2 : (a + 4)^2 - a^2 = 80) : 
  a^2 + (a + 4)^2 = 208 := by
  sorry

end square_areas_sum_l1401_140127


namespace vector_equation_l1401_140191

variable {V : Type*} [AddCommGroup V]

theorem vector_equation (A B C : V) : (C - A) - (C - B) = B - A := by sorry

end vector_equation_l1401_140191


namespace closed_mul_l1401_140117

structure SpecialSet (S : Set ℝ) : Prop where
  one_mem : (1 : ℝ) ∈ S
  closed_sub : ∀ a b : ℝ, a ∈ S → b ∈ S → (a - b) ∈ S
  closed_inv : ∀ a : ℝ, a ∈ S → a ≠ 0 → (1 / a) ∈ S

theorem closed_mul {S : Set ℝ} (h : SpecialSet S) :
  ∀ a b : ℝ, a ∈ S → b ∈ S → (a * b) ∈ S := by
  sorry

end closed_mul_l1401_140117


namespace new_average_weight_l1401_140125

theorem new_average_weight (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 6 →
  a = 78 →
  (b + c + d + e) / 4 = 79 := by
sorry

end new_average_weight_l1401_140125


namespace sampling_methods_correct_l1401_140182

-- Define the sampling methods
inductive SamplingMethod
| SimpleRandom
| Stratified
| Systematic

-- Define the tasks
structure Task1 where
  total_products : Nat
  sample_size : Nat

structure Task2 where
  total_students : Nat
  first_year : Nat
  second_year : Nat
  third_year : Nat
  sample_size : Nat

structure Task3 where
  rows : Nat
  seats_per_row : Nat
  sample_size : Nat

-- Define the function to determine the most reasonable sampling method
def most_reasonable_sampling_method (task1 : Task1) (task2 : Task2) (task3 : Task3) : 
  (SamplingMethod × SamplingMethod × SamplingMethod) :=
  (SamplingMethod.SimpleRandom, SamplingMethod.Stratified, SamplingMethod.Systematic)

-- Theorem statement
theorem sampling_methods_correct (task1 : Task1) (task2 : Task2) (task3 : Task3) :
  task1.total_products = 30 ∧ task1.sample_size = 3 ∧
  task2.total_students = 2460 ∧ task2.first_year = 890 ∧ task2.second_year = 820 ∧ 
  task2.third_year = 810 ∧ task2.sample_size = 300 ∧
  task3.rows = 28 ∧ task3.seats_per_row = 32 ∧ task3.sample_size = 28 →
  most_reasonable_sampling_method task1 task2 task3 = 
    (SamplingMethod.SimpleRandom, SamplingMethod.Stratified, SamplingMethod.Systematic) :=
by
  sorry

end sampling_methods_correct_l1401_140182


namespace correct_average_after_adjustments_l1401_140156

theorem correct_average_after_adjustments (n : ℕ) (initial_avg : ℚ) 
  (error1 : ℚ) (wrong_num : ℚ) (correct_num : ℚ) :
  n = 10 →
  initial_avg = 40.2 →
  error1 = 17 →
  wrong_num = 13 →
  correct_num = 31 →
  (n : ℚ) * initial_avg - error1 - wrong_num + correct_num = n * 40.3 := by
  sorry

end correct_average_after_adjustments_l1401_140156


namespace complex_number_existence_l1401_140112

theorem complex_number_existence : ∃ (c : ℂ) (d : ℝ), c ≠ 0 ∧
  ∀ (z : ℂ), Complex.abs z = 1 → (1 + z + z^2 ≠ 0) →
    Complex.abs (Complex.abs (1 / (1 + z + z^2)) - Complex.abs (1 / (1 + z + z^2) - c)) = d :=
by sorry

end complex_number_existence_l1401_140112


namespace cube_root_of_one_eighth_l1401_140147

theorem cube_root_of_one_eighth (x : ℝ) : x^3 = 1/8 → x = 1/2 := by
  sorry

end cube_root_of_one_eighth_l1401_140147


namespace octagon_area_l1401_140152

/-- The area of a regular octagon inscribed in a circle with radius 3 units -/
theorem octagon_area (r : ℝ) (h : r = 3) : 
  let octagon_area := 8 * (1/2 * r^2 * Real.sin (π/4))
  octagon_area = 18 * Real.sqrt 2 := by
  sorry

end octagon_area_l1401_140152


namespace sum_of_coefficients_l1401_140166

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^9 = a₉*x^9 + a₈*x^8 + a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -2 :=
by sorry

end sum_of_coefficients_l1401_140166


namespace dots_on_line_l1401_140173

/-- The number of dots drawn on a line of given length at given intervals, excluding the beginning and end points. -/
def numDots (lineLength : ℕ) (interval : ℕ) : ℕ :=
  if interval = 0 then 0
  else (lineLength - interval) / interval

theorem dots_on_line (lineLength : ℕ) (interval : ℕ) 
  (h1 : lineLength = 30) 
  (h2 : interval = 5) : 
  numDots lineLength interval = 5 := by
  sorry

end dots_on_line_l1401_140173


namespace speed_increases_with_height_l1401_140196

/-- Represents a data point of height and time -/
structure DataPoint where
  height : ℝ
  time : ℝ

/-- The data set from the experiment -/
def dataSet : List DataPoint := [
  ⟨10, 4.23⟩, ⟨20, 3.00⟩, ⟨30, 2.45⟩, ⟨40, 2.13⟩, 
  ⟨50, 1.89⟩, ⟨60, 1.71⟩, ⟨70, 1.59⟩
]

/-- Theorem stating that average speed increases with height -/
theorem speed_increases_with_height :
  ∀ (d1 d2 : DataPoint), 
    d1 ∈ dataSet → d2 ∈ dataSet →
    d2.height > d1.height → 
    d2.height / d2.time > d1.height / d1.time :=
by sorry

end speed_increases_with_height_l1401_140196


namespace interval_relation_l1401_140109

theorem interval_relation : 
  (∀ x : ℝ, 3 < x ∧ x < 4 → 2 < x ∧ x < 5) ∧ 
  (∃ x : ℝ, 2 < x ∧ x < 5 ∧ ¬(3 < x ∧ x < 4)) :=
by sorry

end interval_relation_l1401_140109


namespace hexagon_area_equals_six_l1401_140162

/-- Given an equilateral triangle with area 4 and a regular hexagon with the same perimeter,
    prove that the area of the hexagon is 6. -/
theorem hexagon_area_equals_six (s t : ℝ) : 
  s > 0 → t > 0 → -- Positive side lengths
  3 * s = 6 * t → -- Equal perimeters
  s^2 * Real.sqrt 3 / 4 = 4 → -- Triangle area
  6 * (t^2 * Real.sqrt 3 / 4) = 6 := by
sorry


end hexagon_area_equals_six_l1401_140162


namespace base7_523_equals_base10_262_l1401_140186

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (a b c : ℕ) : ℕ :=
  a * 7^2 + b * 7^1 + c * 7^0

/-- The theorem stating that 523 in base-7 is equal to 262 in base-10 --/
theorem base7_523_equals_base10_262 : base7ToBase10 5 2 3 = 262 := by
  sorry

end base7_523_equals_base10_262_l1401_140186


namespace weekly_savings_l1401_140148

def hourly_rate_1 : ℚ := 20
def hourly_rate_2 : ℚ := 22
def subsidy : ℚ := 6
def hours_per_week : ℚ := 40

def weekly_cost_1 : ℚ := hourly_rate_1 * hours_per_week
def effective_hourly_rate_2 : ℚ := hourly_rate_2 - subsidy
def weekly_cost_2 : ℚ := effective_hourly_rate_2 * hours_per_week

theorem weekly_savings : weekly_cost_1 - weekly_cost_2 = 160 := by
  sorry

end weekly_savings_l1401_140148


namespace platform_length_l1401_140126

/-- Given a train of length 1200 m that takes 120 sec to pass a tree and 150 sec to pass a platform, 
    prove that the length of the platform is 300 m. -/
theorem platform_length 
  (train_length : ℝ) 
  (time_tree : ℝ) 
  (time_platform : ℝ) 
  (h1 : train_length = 1200)
  (h2 : time_tree = 120)
  (h3 : time_platform = 150) :
  let train_speed := train_length / time_tree
  let platform_length := train_speed * time_platform - train_length
  platform_length = 300 := by
sorry


end platform_length_l1401_140126


namespace pipe_cut_theorem_l1401_140111

theorem pipe_cut_theorem (total_length : ℝ) (difference : ℝ) (shorter_length : ℝ) : 
  total_length = 120 →
  difference = 22 →
  total_length = shorter_length + (shorter_length + difference) →
  shorter_length = 49 := by
sorry

end pipe_cut_theorem_l1401_140111


namespace largest_base3_3digit_in_base10_l1401_140177

/-- The largest three-digit number in base 3 -/
def largest_base3_3digit : ℕ := 2 * 3^2 + 2 * 3^1 + 2 * 3^0

/-- Theorem: The largest three-digit number in base 3, when converted to base 10, equals 26 -/
theorem largest_base3_3digit_in_base10 : largest_base3_3digit = 26 := by
  sorry

end largest_base3_3digit_in_base10_l1401_140177


namespace smallest_number_of_blocks_l1401_140180

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  height : ℕ
  length : ℕ

/-- Represents the dimensions of the wall -/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Calculates the number of blocks needed to build the wall -/
def blocksNeeded (wall : WallDimensions) (block : BlockDimensions) : ℕ :=
  let oddRowBlocks := wall.length / 2
  let evenRowBlocks := oddRowBlocks + 1
  let numRows := wall.height / block.height
  let oddRows := numRows / 2
  let evenRows := numRows - oddRows
  oddRows * oddRowBlocks + evenRows * evenRowBlocks

/-- The theorem stating the smallest number of blocks needed -/
theorem smallest_number_of_blocks 
  (wall : WallDimensions)
  (block : BlockDimensions)
  (h1 : wall.length = 120)
  (h2 : wall.height = 8)
  (h3 : block.height = 1)
  (h4 : block.length = 2 ∨ block.length = 1)
  (h5 : wall.length % 2 = 0) -- Ensures wall is even on the ends
  : blocksNeeded wall block = 484 :=
by
  sorry

end smallest_number_of_blocks_l1401_140180


namespace equivalent_discount_l1401_140142

theorem equivalent_discount (original_price : ℝ) 
  (first_discount second_discount : ℝ) 
  (h1 : first_discount = 0.3) 
  (h2 : second_discount = 0.2) :
  let price_after_first := original_price * (1 - first_discount)
  let final_price := price_after_first * (1 - second_discount)
  let equivalent_discount := 1 - (final_price / original_price)
  equivalent_discount = 0.44 := by sorry

end equivalent_discount_l1401_140142


namespace fourth_number_in_sequence_l1401_140149

theorem fourth_number_in_sequence (a b c d : ℝ) : 
  a / b = 5 / 3 ∧ 
  b / c = 3 / 4 ∧ 
  a + b + c = 108 ∧ 
  d - c = c - b ∧ 
  c - b = b - a 
  → d = 45 := by
  sorry

end fourth_number_in_sequence_l1401_140149


namespace average_marks_proof_l1401_140107

-- Define the marks for each subject
def physics : ℝ := 125
def chemistry : ℝ := 15
def mathematics : ℝ := 55

-- Define the conditions
theorem average_marks_proof :
  -- Average of all three subjects is 65
  (physics + chemistry + mathematics) / 3 = 65 ∧
  -- Average of physics and mathematics is 90
  (physics + mathematics) / 2 = 90 ∧
  -- Average of physics and chemistry is 70
  (physics + chemistry) / 2 = 70 ∧
  -- Physics marks are 125
  physics = 125 →
  -- Prove that chemistry is the subject that averages 70 with physics
  (physics + chemistry) / 2 = 70 :=
by sorry

end average_marks_proof_l1401_140107


namespace find_number_to_multiply_l1401_140100

theorem find_number_to_multiply : ∃ x : ℤ, 43 * x - 34 * x = 1215 :=
by sorry

end find_number_to_multiply_l1401_140100


namespace rational_solutions_quadratic_l1401_140121

theorem rational_solutions_quadratic (k : ℕ+) :
  (∃ x : ℚ, k * x^2 + 30 * x + k = 0) ↔ (k = 9 ∨ k = 15) := by
sorry

end rational_solutions_quadratic_l1401_140121


namespace cubic_equation_solutions_l1401_140161

theorem cubic_equation_solutions : 
  ∀ m n : ℤ, m^3 - n^3 = 2*m*n + 8 ↔ (m = 2 ∧ n = 0) ∨ (m = 0 ∧ n = -2) := by
  sorry

end cubic_equation_solutions_l1401_140161


namespace rational_equation_solution_l1401_140171

theorem rational_equation_solution :
  ∃ x : ℚ, (x + 11) / (x - 4) = (x - 3) / (x + 6) ↔ x = -9/4 := by
sorry

end rational_equation_solution_l1401_140171


namespace skateboard_padding_cost_increase_l1401_140128

/-- Calculates the percent increase in the combined cost of a skateboard and padding set. -/
theorem skateboard_padding_cost_increase 
  (skateboard_cost : ℝ) 
  (padding_cost : ℝ) 
  (skateboard_increase : ℝ) 
  (padding_increase : ℝ) : 
  skateboard_cost = 120 →
  padding_cost = 30 →
  skateboard_increase = 0.08 →
  padding_increase = 0.15 →
  let new_skateboard_cost := skateboard_cost * (1 + skateboard_increase)
  let new_padding_cost := padding_cost * (1 + padding_increase)
  let original_total := skateboard_cost + padding_cost
  let new_total := new_skateboard_cost + new_padding_cost
  (new_total - original_total) / original_total = 0.094 := by
  sorry

end skateboard_padding_cost_increase_l1401_140128


namespace function_identity_l1401_140122

theorem function_identity (f : ℕ → ℕ) : 
  (∀ n : ℕ, f n + f (f n) + f (f (f n)) = 3 * n) → 
  (∀ n : ℕ, f n = n) := by sorry

end function_identity_l1401_140122


namespace impossible_coin_probabilities_l1401_140102

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧ 
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) := by
  sorry

end impossible_coin_probabilities_l1401_140102


namespace quadratic_equation_roots_l1401_140157

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 4*x₁ - 4 = 0) ∧ (x₂^2 - 4*x₂ - 4 = 0) := by
  sorry

end quadratic_equation_roots_l1401_140157


namespace output_for_twelve_l1401_140199

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 25 then
    step1 - 5
  else
    step1 * 2

theorem output_for_twelve : function_machine 12 = 31 := by
  sorry

end output_for_twelve_l1401_140199


namespace chessboard_invariant_l1401_140183

/-- Represents a chessboard configuration -/
def Chessboard := Matrix (Fin 8) (Fin 8) Int

/-- Initial chessboard configuration -/
def initialBoard : Chessboard :=
  fun i j => if i = 1 ∧ j = 7 then -1 else 1

/-- Represents a move (changing signs in a row or column) -/
inductive Move
  | row (i : Fin 8)
  | col (j : Fin 8)

/-- Apply a move to a chessboard -/
def applyMove (b : Chessboard) (m : Move) : Chessboard :=
  match m with
  | Move.row i => fun r c => if r = i then -b r c else b r c
  | Move.col j => fun r c => if c = j then -b r c else b r c

/-- Apply a sequence of moves to a chessboard -/
def applyMoves (b : Chessboard) : List Move → Chessboard
  | [] => b
  | m :: ms => applyMoves (applyMove b m) ms

/-- Product of all numbers on the board -/
def boardProduct (b : Chessboard) : Int :=
  (Finset.univ.prod fun i => Finset.univ.prod fun j => b i j)

/-- Main theorem -/
theorem chessboard_invariant (moves : List Move) :
    boardProduct (applyMoves initialBoard moves) = -1 := by
  sorry

end chessboard_invariant_l1401_140183


namespace square_roots_problem_l1401_140189

theorem square_roots_problem (a : ℝ) :
  (∃ x > 0, (2*a - 1)^2 = x ∧ (a - 2)^2 = x) → (2*a - 1)^2 = 1 :=
by sorry

end square_roots_problem_l1401_140189


namespace conic_is_hyperbola_l1401_140195

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  (x + 5)^2 = (4*y - 3)^2 - 140

/-- Definition of a hyperbola -/
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
    ∀ x y, f x y ↔ a * x^2 + b * y^2 + c * x + d * y + e = 0

/-- Theorem stating that the given equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation :=
sorry

end conic_is_hyperbola_l1401_140195


namespace odd_divisibility_l1401_140165

theorem odd_divisibility (n : ℕ) (h : Odd (94 * n)) :
  ∃ k : ℕ, n * (n - 1) ^ ((n - 1) ^ n + 1) + n = k * ((n - 1) ^ n + 1) ^ 2 := by
  sorry

end odd_divisibility_l1401_140165


namespace hexagon_sixth_angle_l1401_140133

/-- The sum of interior angles in a hexagon -/
def hexagon_angle_sum : ℝ := 720

/-- The five known angles in the hexagon -/
def known_angles : List ℝ := [108, 130, 142, 105, 120]

/-- Theorem: In a hexagon where five of the interior angles measure 108°, 130°, 142°, 105°, and 120°, the measure of the sixth angle is 115°. -/
theorem hexagon_sixth_angle :
  hexagon_angle_sum - (known_angles.sum) = 115 := by
  sorry

end hexagon_sixth_angle_l1401_140133


namespace frog_dog_ratio_l1401_140193

theorem frog_dog_ratio (dogs : ℕ) (cats : ℕ) (frogs : ℕ) : 
  cats = (80 * dogs) / 100 →
  frogs = 160 →
  dogs + cats + frogs = 304 →
  frogs = 2 * dogs :=
by sorry

end frog_dog_ratio_l1401_140193


namespace circplus_comm_circplus_not_scalar_mult_circplus_zero_circplus_self_circplus_pos_l1401_140169

-- Define the ⊕ operation
def circplus (x y : ℝ) : ℝ := |x - y|^2

-- Theorem statements
theorem circplus_comm (x y : ℝ) : circplus x y = circplus y x := by sorry

theorem circplus_not_scalar_mult (x y : ℝ) : 
  2 * (circplus x y) ≠ circplus (2 * x) (2 * y) := by sorry

theorem circplus_zero (x : ℝ) : circplus x 0 = x^2 := by sorry

theorem circplus_self (x : ℝ) : circplus x x = 0 := by sorry

theorem circplus_pos (x y : ℝ) : x ≠ y → circplus x y > 0 := by sorry

end circplus_comm_circplus_not_scalar_mult_circplus_zero_circplus_self_circplus_pos_l1401_140169


namespace max_value_of_2x_plus_y_l1401_140159

theorem max_value_of_2x_plus_y (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) :
  ∃ (M : ℝ), M = Real.sqrt 11 ∧ 2 * x + y ≤ M ∧ ∃ (x₀ y₀ : ℝ), 3 * x₀^2 + 2 * y₀^2 ≤ 6 ∧ 2 * x₀ + y₀ = M :=
by sorry

end max_value_of_2x_plus_y_l1401_140159


namespace polynomial_expansion_l1401_140175

theorem polynomial_expansion (x : ℝ) : 
  (7 * x + 3) * (5 * x^2 + 4) = 35 * x^3 + 15 * x^2 + 28 * x + 12 := by
  sorry

end polynomial_expansion_l1401_140175


namespace absolute_value_inequality_solution_range_l1401_140114

theorem absolute_value_inequality_solution_range :
  (∃ (x : ℝ), |x - 5| + |x - 3| < m) → m > 2 := by
  sorry

end absolute_value_inequality_solution_range_l1401_140114


namespace largest_integer_with_remainder_l1401_140116

theorem largest_integer_with_remainder : ∃ n : ℕ, n = 94 ∧ 
  (∀ m : ℕ, m < 100 ∧ m % 6 = 4 → m ≤ n) ∧ 
  n < 100 ∧ 
  n % 6 = 4 :=
sorry

end largest_integer_with_remainder_l1401_140116


namespace oldest_bride_age_l1401_140160

theorem oldest_bride_age (bride_age groom_age : ℕ) : 
  bride_age = groom_age + 19 →
  bride_age + groom_age = 185 →
  bride_age = 102 := by
sorry

end oldest_bride_age_l1401_140160


namespace unique_solution_abs_equation_l1401_140134

theorem unique_solution_abs_equation :
  ∃! x : ℝ, |x - 20| + |x - 18| = |2*x - 36| :=
by
  -- The proof goes here
  sorry

end unique_solution_abs_equation_l1401_140134


namespace negative_eight_interpretations_l1401_140164

theorem negative_eight_interpretations :
  (-(- 8) = -(-8)) ∧
  (-(- 8) = -1 * (-8)) ∧
  (-(- 8) = |-8|) ∧
  (-(- 8) = 8) :=
by sorry

end negative_eight_interpretations_l1401_140164


namespace inequality_proof_l1401_140150

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ b)
  (h2 : b ≥ c)
  (h3 : c > 0)
  (h4 : a * b * c = 1)
  (h5 : a + b + c > 1/a + 1/b + 1/c) :
  a > 1 ∧ 1 > b := by
  sorry

end inequality_proof_l1401_140150


namespace johns_payment_ratio_l1401_140145

/-- Proves that the ratio of John's payment to the total cost for the first year is 1/2 --/
theorem johns_payment_ratio (
  num_members : ℕ)
  (join_fee : ℕ)
  (monthly_cost : ℕ)
  (johns_payment : ℕ)
  (h1 : num_members = 4)
  (h2 : join_fee = 4000)
  (h3 : monthly_cost = 1000)
  (h4 : johns_payment = 32000)
  : johns_payment / (num_members * (join_fee + 12 * monthly_cost)) = 1 / 2 := by
  sorry

end johns_payment_ratio_l1401_140145


namespace sqrt_one_plus_a_squared_is_quadratic_radical_l1401_140143

/-- A function is a quadratic radical if it's the square root of an expression 
    that yields a real number for all real values of its variable. -/
def is_quadratic_radical (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g x ≥ 0) ∧ (∀ x, f x = Real.sqrt (g x))

/-- The function f(a) = √(1 + a²) is a quadratic radical. -/
theorem sqrt_one_plus_a_squared_is_quadratic_radical :
  is_quadratic_radical (fun a => Real.sqrt (1 + a^2)) :=
by
  sorry


end sqrt_one_plus_a_squared_is_quadratic_radical_l1401_140143


namespace train_speed_proof_l1401_140135

/-- Proves that a train with given parameters has a specific speed -/
theorem train_speed_proof (train_length : ℝ) (crossing_time : ℝ) (total_length : ℝ) :
  train_length = 150 →
  crossing_time = 30 →
  total_length = 225 →
  (total_length - train_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_proof

end train_speed_proof_l1401_140135


namespace lara_baking_cookies_l1401_140130

/-- The number of baking trays Lara is using. -/
def num_trays : ℕ := 4

/-- The number of rows of cookies on each tray. -/
def rows_per_tray : ℕ := 5

/-- The number of cookies in each row. -/
def cookies_per_row : ℕ := 6

/-- The total number of cookies Lara is baking. -/
def total_cookies : ℕ := num_trays * rows_per_tray * cookies_per_row

theorem lara_baking_cookies : total_cookies = 120 := by
  sorry

end lara_baking_cookies_l1401_140130


namespace range_of_fraction_l1401_140174

theorem range_of_fraction (a b : ℝ) (ha : 1 < a ∧ a < 2) (hb : -2 < b ∧ b < -1) :
  ∃ (x : ℝ), -2 < x ∧ x < -1/2 ∧ (∃ (a' b' : ℝ), 1 < a' ∧ a' < 2 ∧ -2 < b' ∧ b' < -1 ∧ x = a' / b') ∧
  (∀ (y : ℝ), (∃ (a' b' : ℝ), 1 < a' ∧ a' < 2 ∧ -2 < b' ∧ b' < -1 ∧ y = a' / b') → -2 < y ∧ y < -1/2) :=
by sorry

end range_of_fraction_l1401_140174


namespace min_vertices_for_perpendicular_diagonals_l1401_140141

theorem min_vertices_for_perpendicular_diagonals : 
  (∀ k : ℕ, k < 28 → ¬(∃ m : ℕ, 2 * m = k ∧ m * (m - 1)^2 / 2 ≥ 1000)) ∧ 
  (∃ m : ℕ, 2 * m = 28 ∧ m * (m - 1)^2 / 2 ≥ 1000) := by
  sorry

end min_vertices_for_perpendicular_diagonals_l1401_140141


namespace geometric_sequence_general_term_l1401_140181

/-- A geometric sequence with common ratio 4 and sum of first three terms equal to 21 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 4 * a n) ∧ (a 1 + a 2 + a 3 = 21)

/-- The general term formula for the geometric sequence -/
theorem geometric_sequence_general_term (a : ℕ → ℝ) (h : GeometricSequence a) :
  ∀ n : ℕ, a n = 4^(n - 1) := by
  sorry

end geometric_sequence_general_term_l1401_140181


namespace algebraic_grid_difference_l1401_140140

/-- Represents a 3x3 grid of algebraic expressions -/
structure AlgebraicGrid (α : Type) [Ring α] where
  grid : Matrix (Fin 3) (Fin 3) α

/-- Checks if all rows, columns, and diagonals have the same sum -/
def isValidGrid {α : Type} [Ring α] (g : AlgebraicGrid α) : Prop :=
  let rowSum (i : Fin 3) := g.grid i 0 + g.grid i 1 + g.grid i 2
  let colSum (j : Fin 3) := g.grid 0 j + g.grid 1 j + g.grid 2 j
  let diag1Sum := g.grid 0 0 + g.grid 1 1 + g.grid 2 2
  let diag2Sum := g.grid 0 2 + g.grid 1 1 + g.grid 2 0
  ∀ i j : Fin 3, rowSum i = colSum j ∧ rowSum i = diag1Sum ∧ rowSum i = diag2Sum

theorem algebraic_grid_difference {α : Type} [CommRing α] (x : α) (M N : α) :
  let g : AlgebraicGrid α := {
    grid := λ i j =>
      if i = 0 ∧ j = 0 then M
      else if i = 0 ∧ j = 2 then x^2 - x - 1
      else if i = 1 ∧ j = 2 then x
      else if i = 2 ∧ j = 0 then x^2 - x
      else if i = 2 ∧ j = 1 then x - 1
      else if i = 2 ∧ j = 2 then N
      else 0  -- Other entries are not specified
  }
  isValidGrid g →
  M - N = -2*x^2 + 4*x :=
by
  sorry

end algebraic_grid_difference_l1401_140140


namespace eve_age_proof_l1401_140168

/-- Adam's current age -/
def adam_age : ℕ := 9

/-- Eve's current age -/
def eve_age : ℕ := 14

/-- Theorem stating Eve's age based on the given conditions -/
theorem eve_age_proof :
  (adam_age < eve_age) ∧
  (eve_age + 1 = 3 * (adam_age - 4)) ∧
  (adam_age = 9) →
  eve_age = 14 := by
sorry

end eve_age_proof_l1401_140168


namespace complex_number_simplification_l1401_140153

theorem complex_number_simplification :
  ∀ (i : ℂ), i^2 = -1 →
  3*i * (2 - 5*i) - (4 - 7*i) = 11 + 13*i :=
by
  sorry

end complex_number_simplification_l1401_140153


namespace sixteen_letters_with_both_l1401_140184

/-- Represents an alphabet with letters containing dots and straight lines -/
structure Alphabet :=
  (total : ℕ)
  (only_line : ℕ)
  (only_dot : ℕ)
  (both : ℕ)
  (all_have_feature : only_line + only_dot + both = total)

/-- The number of letters with both a dot and a straight line in the given alphabet -/
def letters_with_both (a : Alphabet) : ℕ := a.both

/-- Theorem stating that in the given alphabet, 16 letters contain both a dot and a straight line -/
theorem sixteen_letters_with_both (a : Alphabet) 
  (h1 : a.total = 50)
  (h2 : a.only_line = 30)
  (h3 : a.only_dot = 4) :
  letters_with_both a = 16 := by
  sorry

end sixteen_letters_with_both_l1401_140184


namespace city_fuel_efficiency_l1401_140110

/-- Represents the fuel efficiency of a car -/
structure CarFuelEfficiency where
  highway : ℝ  -- Miles per gallon on the highway
  city : ℝ     -- Miles per gallon in the city
  tank_size : ℝ -- Size of the fuel tank in gallons

/-- The conditions given in the problem -/
def problem_conditions (car : CarFuelEfficiency) : Prop :=
  car.highway * car.tank_size = 560 ∧
  car.city * car.tank_size = 336 ∧
  car.city = car.highway - 6

/-- The theorem to be proved -/
theorem city_fuel_efficiency (car : CarFuelEfficiency) :
  problem_conditions car → car.city = 9 := by
  sorry


end city_fuel_efficiency_l1401_140110


namespace circle_area_difference_l1401_140118

theorem circle_area_difference : 
  let r1 : ℝ := 30
  let c2 : ℝ := 30
  let area1 := π * r1^2
  let r2 := c2 / (2 * π)
  let area2 := π * r2^2
  area1 - area2 = (225 * (4 * π^2 - 1)) / π := by sorry

end circle_area_difference_l1401_140118


namespace f_increasing_iff_a_in_range_l1401_140105

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 4*a*x else (2*a + 3)*x - 4*a + 5

theorem f_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (1/2) (3/2) :=
sorry

end f_increasing_iff_a_in_range_l1401_140105


namespace hyperbola_circle_tangency_l1401_140155

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- A circle with center (0, 3) and parameter m -/
structure Circle (m : ℝ) where
  equation : ∀ x y : ℝ, x^2 + y^2 - 6*y + m = 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The asymptote of a hyperbola -/
def asymptote (h : Hyperbola a b) : Set (ℝ × ℝ) := sorry

/-- Tangency condition between a line and a circle -/
def is_tangent (l : Set (ℝ × ℝ)) (c : Circle m) : Prop := sorry

theorem hyperbola_circle_tangency 
  (a b : ℝ) 
  (h : Hyperbola a b) 
  (m : ℝ) 
  (c : Circle m) :
  eccentricity h = 3 →
  is_tangent (asymptote h) c →
  m = 8 := by sorry

end hyperbola_circle_tangency_l1401_140155


namespace tangent_circles_radius_l1401_140185

theorem tangent_circles_radius (r₁ r₂ d : ℝ) : 
  r₁ = 2 →
  d = 5 →
  (r₁ + r₂ = d ∨ |r₁ - r₂| = d) →
  r₂ = 3 ∨ r₂ = 7 := by
sorry

end tangent_circles_radius_l1401_140185


namespace round_table_seats_l1401_140188

/-- Represents a round table with equally spaced seats -/
structure RoundTable where
  total_seats : ℕ
  seat_numbers : Fin total_seats → ℕ

/-- Two seats are opposite if they are half the total number of seats apart -/
def opposite (t : RoundTable) (s1 s2 : Fin t.total_seats) : Prop :=
  (t.seat_numbers s2 - t.seat_numbers s1) % t.total_seats = t.total_seats / 2

theorem round_table_seats (t : RoundTable) (s1 s2 : Fin t.total_seats) :
  t.seat_numbers s1 = 10 ∧ t.seat_numbers s2 = 29 ∧ opposite t s1 s2 → t.total_seats = 38 :=
by
  sorry


end round_table_seats_l1401_140188


namespace parabola_c_value_l1401_140154

/-- A parabola with equation y = ax^2 + bx + c, vertex at (-3, -5), and passing through (-1, -4) -/
def Parabola (a b c : ℚ) : Prop :=
  ∀ x y : ℚ, y = a * x^2 + b * x + c →
  (∃ t : ℚ, y = a * (x + 3)^2 - 5) ∧  -- vertex form
  (-4 : ℚ) = a * (-1 + 3)^2 - 5       -- passes through (-1, -4)

/-- The value of c for the given parabola is -11/4 -/
theorem parabola_c_value (a b c : ℚ) (h : Parabola a b c) : c = -11/4 := by
  sorry

end parabola_c_value_l1401_140154


namespace min_empty_cells_is_three_l1401_140104

/-- Represents a triangular cell arrangement with grasshoppers -/
structure TriangularArrangement where
  up_cells : ℕ  -- Number of upward-pointing cells
  down_cells : ℕ  -- Number of downward-pointing cells
  has_more_up : up_cells = down_cells + 3

/-- The minimum number of empty cells after all grasshoppers have jumped -/
def min_empty_cells (arrangement : TriangularArrangement) : ℕ := 3

/-- Theorem stating that the minimum number of empty cells is always 3 -/
theorem min_empty_cells_is_three (arrangement : TriangularArrangement) :
  min_empty_cells arrangement = 3 := by
  sorry

end min_empty_cells_is_three_l1401_140104


namespace sacks_per_section_l1401_140144

/-- Given an orchard with 8 sections that produces 360 sacks of apples daily,
    prove that each section produces 45 sacks per day. -/
theorem sacks_per_section (sections : ℕ) (total_sacks : ℕ) (h1 : sections = 8) (h2 : total_sacks = 360) :
  total_sacks / sections = 45 := by
  sorry

end sacks_per_section_l1401_140144


namespace photos_sum_equals_total_l1401_140115

/-- The total number of photos collected by Tom, Tim, and Paul -/
def total_photos : ℕ := 152

/-- Tom's photos -/
def tom_photos : ℕ := 38

/-- Tim's photos -/
def tim_photos : ℕ := total_photos - 100

/-- Paul's photos -/
def paul_photos : ℕ := tim_photos + 10

/-- Theorem stating that the sum of individual photos equals the total photos -/
theorem photos_sum_equals_total : 
  tom_photos + tim_photos + paul_photos = total_photos := by sorry

end photos_sum_equals_total_l1401_140115


namespace average_children_in_families_with_children_l1401_140124

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_all : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 12)
  (h2 : average_all = 5/2)
  (h3 : childless_families = 2) :
  (total_families * average_all) / (total_families - childless_families) = 3 := by
sorry

end average_children_in_families_with_children_l1401_140124


namespace problem_solution_l1401_140192

theorem problem_solution (k : ℕ) (y : ℚ) 
  (h1 : (1/2)^18 * (1/81)^k = y)
  (h2 : k = 9) : 
  y = 1 / (2^18 * 3^36) := by
  sorry

end problem_solution_l1401_140192


namespace problem_1_problem_2_l1401_140151

-- Problem 1
theorem problem_1 : -1^2 - |(-2)| + (1/3 - 3/4) * 12 = -8 := by sorry

-- Problem 2
theorem problem_2 :
  ∃ (x y : ℚ), (x / 2 - (y + 1) / 3 = 1) ∧ (3 * x + 2 * y = 10) ∧ (x = 3) ∧ (y = 1/2) := by sorry

end problem_1_problem_2_l1401_140151


namespace mode_and_median_of_game_scores_l1401_140178

def game_scores : List Int := [20, 18, 23, 17, 20, 20, 18]

def mode (l : List Int) : Int := sorry

def median (l : List Int) : Int := sorry

theorem mode_and_median_of_game_scores :
  mode game_scores = 20 ∧ median game_scores = 20 := by sorry

end mode_and_median_of_game_scores_l1401_140178


namespace minimum_oranges_l1401_140163

theorem minimum_oranges : ∃ n : ℕ, n > 0 ∧ 
  (n % 5 = 1 ∧ n % 7 = 1 ∧ n % 10 = 1) ∧ 
  ∀ m : ℕ, m > 0 → (m % 5 = 1 ∧ m % 7 = 1 ∧ m % 10 = 1) → m ≥ 71 := by
  sorry

end minimum_oranges_l1401_140163


namespace negation_equivalence_l1401_140190

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2016 > 0) :=
by sorry

end negation_equivalence_l1401_140190


namespace two_distinct_roots_iff_p_condition_l1401_140198

theorem two_distinct_roots_iff_p_condition (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    ((x ≥ 0 → x^2 - 2*x - p = 0) ∧ (x < 0 → x^2 + 2*x - p = 0)) ∧
    ((y ≥ 0 → y^2 - 2*y - p = 0) ∧ (y < 0 → y^2 + 2*y - p = 0)))
  ↔ 
  (p > 0 ∨ p = -1) :=
sorry

end two_distinct_roots_iff_p_condition_l1401_140198


namespace tilly_bag_cost_l1401_140139

/-- Calculates the cost per bag for Tilly's business --/
def cost_per_bag (num_bags : ℕ) (selling_price : ℚ) (total_profit : ℚ) : ℚ :=
  (num_bags * selling_price - total_profit) / num_bags

/-- Proves that the cost per bag is $7 given the problem conditions --/
theorem tilly_bag_cost :
  cost_per_bag 100 10 300 = 7 := by
  sorry

end tilly_bag_cost_l1401_140139


namespace product_of_numbers_l1401_140108

theorem product_of_numbers (x y : ℝ) (sum_eq : x + y = 20) (sum_squares_eq : x^2 + y^2 = 200) :
  x * y = 100 := by
  sorry

end product_of_numbers_l1401_140108


namespace different_amounts_eq_127_l1401_140137

/-- Represents the number of coins of each denomination --/
structure CoinCounts where
  jiao_1 : Nat
  jiao_5 : Nat
  yuan_1 : Nat
  yuan_5 : Nat

/-- Calculates the number of different non-zero amounts that can be paid with the given coins --/
def differentAmounts (coins : CoinCounts) : Nat :=
  sorry

/-- The specific coin counts given in the problem --/
def problemCoins : CoinCounts :=
  { jiao_1 := 1
  , jiao_5 := 2
  , yuan_1 := 5
  , yuan_5 := 2 }

/-- Theorem stating that the number of different non-zero amounts is 127 --/
theorem different_amounts_eq_127 : differentAmounts problemCoins = 127 :=
  sorry

end different_amounts_eq_127_l1401_140137


namespace intersection_count_l1401_140103

-- Define the lines
def line1 (x y : ℝ) : Prop := 3*x + 4*y - 12 = 0
def line2 (x y : ℝ) : Prop := 5*x - 2*y - 10 = 0
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = 1

-- Define an intersection point
def is_intersection (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨
  (line1 x y ∧ line3 x) ∨
  (line1 x y ∧ line4 y) ∨
  (line2 x y ∧ line3 x) ∨
  (line2 x y ∧ line4 y) ∨
  (line3 x ∧ line4 y)

-- Theorem statement
theorem intersection_count :
  ∃ (p1 p2 : ℝ × ℝ), 
    p1 ≠ p2 ∧
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    ∀ (p : ℝ × ℝ), is_intersection p.1 p.2 → p = p1 ∨ p = p2 :=
sorry

end intersection_count_l1401_140103


namespace greatest_3digit_base9_divisible_by_7_l1401_140179

/-- Converts a base 9 number to base 10 --/
def base9ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 9^2 + tens * 9 + ones

/-- Checks if a number is a valid 3-digit base 9 number --/
def isValidBase9 (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 888

theorem greatest_3digit_base9_divisible_by_7 :
  ∃ (n : Nat), isValidBase9 n ∧ 
               base9ToBase10 n % 7 = 0 ∧
               ∀ (m : Nat), isValidBase9 m ∧ base9ToBase10 m % 7 = 0 → base9ToBase10 m ≤ base9ToBase10 n :=
by
  -- The proof goes here
  sorry

end greatest_3digit_base9_divisible_by_7_l1401_140179


namespace existence_of_triple_l1401_140123

theorem existence_of_triple (n : ℕ) :
  let A := Finset.range (2^(n+1))
  ∀ S : Finset ℕ, S ⊆ A → S.card = 2*n + 1 →
    ∃ a b c : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      (b * c : ℝ) < 2 * (a^2 : ℝ) ∧ 2 * (a^2 : ℝ) < 4 * (b * c : ℝ) :=
by sorry

end existence_of_triple_l1401_140123


namespace line_through_points_l1401_140176

/-- A line passing through two points (1,3) and (4,-2) can be represented by y = mx + b, where m + b = 3 -/
theorem line_through_points (m b : ℚ) : 
  (3 = m * 1 + b) → (-2 = m * 4 + b) → m + b = 3 := by
  sorry

end line_through_points_l1401_140176


namespace first_term_of_arithmetic_progression_l1401_140146

/-- Given an arithmetic progression with the 25th term equal to 173 and a common difference of 7,
    prove that the first term is 5. -/
theorem first_term_of_arithmetic_progression :
  ∀ (a : ℕ → ℤ),
    (∀ n : ℕ, a (n + 1) = a n + 7) →  -- Common difference is 7
    a 25 = 173 →                      -- 25th term is 173
    a 1 = 5 :=                        -- First term is 5
by
  sorry

end first_term_of_arithmetic_progression_l1401_140146


namespace cos_330_degrees_l1401_140170

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_degrees_l1401_140170


namespace reflected_point_spherical_coordinates_l1401_140167

/-- Given a point P with rectangular coordinates (x, y, z) and spherical coordinates (ρ, θ, φ),
    this function returns the spherical coordinates of the point Q(-x, y, z) -/
def spherical_coordinates_of_reflected_point (x y z ρ θ φ : Real) : Real × Real × Real :=
  sorry

/-- Theorem stating that if a point P has rectangular coordinates (x, y, z) and 
    spherical coordinates (3, 5π/6, π/4), then the point Q(-x, y, z) has 
    spherical coordinates (3, π/6, π/4) -/
theorem reflected_point_spherical_coordinates 
  (x y z : Real) 
  (h1 : x = 3 * Real.sin (π/4) * Real.cos (5*π/6))
  (h2 : y = 3 * Real.sin (π/4) * Real.sin (5*π/6))
  (h3 : z = 3 * Real.cos (π/4)) :
  spherical_coordinates_of_reflected_point x y z 3 (5*π/6) (π/4) = (3, π/6, π/4) := by
  sorry

end reflected_point_spherical_coordinates_l1401_140167


namespace increasing_function_inequality_range_l1401_140187

/-- Given an increasing function f defined on [0,+∞), 
    prove that the range of x satisfying f(2x-1) < f(1/3) is [1/2, 2/3). -/
theorem increasing_function_inequality_range 
  (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_domain : ∀ x, x ∈ Set.Ici (0 : ℝ) → f x ∈ Set.univ) :
  {x : ℝ | f (2*x - 1) < f (1/3)} = Set.Icc (1/2 : ℝ) (2/3) := by
sorry

end increasing_function_inequality_range_l1401_140187


namespace quadratic_equation_roots_l1401_140131

theorem quadratic_equation_roots (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 3*x + k = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁ * x₂ + 2*x₁ + 2*x₂ = 1) →
  k = -5 := by
  sorry

end quadratic_equation_roots_l1401_140131


namespace divisible_by_five_l1401_140132

theorem divisible_by_five (a b : ℕ) : 
  (5 ∣ a * b) → (5 ∣ a) ∨ (5 ∣ b) := by
  sorry

end divisible_by_five_l1401_140132


namespace anlu_temperature_difference_l1401_140172

/-- Given a temperature range from -3°C to 3°C in Anlu on a winter day,
    the temperature difference is 6°C. -/
theorem anlu_temperature_difference :
  let min_temp : ℤ := -3
  let max_temp : ℤ := 3
  (max_temp - min_temp : ℤ) = 6 := by sorry

end anlu_temperature_difference_l1401_140172


namespace right_triangle_sides_l1401_140138

/-- A right triangle with perimeter 60 and altitude to hypotenuse 12 has sides 15, 20, and 25. -/
theorem right_triangle_sides (a b c : ℝ) (h : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 →
  a + b + c = 60 →
  h = 12 →
  a^2 + b^2 = c^2 →
  a * b = 2 * h * c →
  (a = 15 ∧ b = 20 ∧ c = 25) ∨ (a = 20 ∧ b = 15 ∧ c = 25) :=
by sorry

end right_triangle_sides_l1401_140138


namespace imaginary_part_of_z_l1401_140106

theorem imaginary_part_of_z (z : ℂ) (h : (z + 1) / (1 - Complex.I) = Complex.I) : 
  Complex.im z = 1 := by sorry

end imaginary_part_of_z_l1401_140106


namespace retail_price_calculation_l1401_140197

def wholesale_price : ℝ := 90

def discount_rate : ℝ := 0.10

def profit_rate : ℝ := 0.20

def retail_price : ℝ := 120

theorem retail_price_calculation :
  let profit := profit_rate * wholesale_price
  let selling_price := wholesale_price + profit
  selling_price = retail_price * (1 - discount_rate) :=
by
  sorry

end retail_price_calculation_l1401_140197


namespace range_of_c_sum_of_squares_inequality_l1401_140129

-- Part I
theorem range_of_c (c : ℝ) (h1 : c > 0) 
  (h2 : ∀ x : ℝ, x + |x - 2*c| ≥ 2) : c ≥ 1 := by
  sorry

-- Part II
theorem sum_of_squares_inequality (p q r : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h_sum : p + q + r = 3) : p^2 + q^2 + r^2 ≥ 3 := by
  sorry

end range_of_c_sum_of_squares_inequality_l1401_140129


namespace k_range_l1401_140136

-- Define the function h
def h (x : ℝ) : ℝ := 5 * x - 3

-- Define the function k as a composition of h
def k (x : ℝ) : ℝ := h (h (h x))

-- State the theorem
theorem k_range :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3,
  ∃ y ∈ Set.Icc (-218 : ℝ) 282,
  k x = y ∧
  ∀ z, k x = z → z ∈ Set.Icc (-218 : ℝ) 282 :=
sorry

end k_range_l1401_140136


namespace isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l1401_140158

/-- An isosceles triangle with congruent sides of length 6 and perimeter 20 has a base of length 8 -/
theorem isosceles_triangle_base_length : ℝ → Prop :=
  fun base =>
    let congruent_side := 6
    let perimeter := 20
    (2 * congruent_side + base = perimeter) → base = 8

/-- Proof of the theorem -/
theorem isosceles_triangle_base_length_proof : isosceles_triangle_base_length 8 := by
  sorry

end isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l1401_140158


namespace least_multiple_of_13_greater_than_450_l1401_140119

theorem least_multiple_of_13_greater_than_450 :
  (∀ n : ℕ, n * 13 > 450 → n * 13 ≥ 455) ∧ 455 % 13 = 0 ∧ 455 > 450 := by
  sorry

end least_multiple_of_13_greater_than_450_l1401_140119


namespace earnings_ratio_l1401_140120

theorem earnings_ratio (mork_rate mindy_rate combined_rate : ℝ) 
  (h1 : mork_rate = 0.30)
  (h2 : mindy_rate = 0.20)
  (h3 : combined_rate = 0.225) : 
  ∃ (m k : ℝ), m > 0 ∧ k > 0 ∧ 
    (mindy_rate * m + mork_rate * k) / (m + k) = combined_rate ∧ 
    m / k = 3 := by
  sorry

end earnings_ratio_l1401_140120


namespace f_properties_l1401_140101

def f (x : ℝ) : ℝ := 1 - |x - x^2|

theorem f_properties :
  (∀ x, f x ≤ 1) ∧
  (f 0 = 1 ∧ f 1 = 1) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 1 - x + x^2) ∧
  (∀ x, (x < 0 ∨ x > 1) → f x = 1 + x - x^2) := by
  sorry

end f_properties_l1401_140101
