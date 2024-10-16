import Mathlib

namespace NUMINAMATH_CALUDE_polar_to_cartesian_coordinates_l52_5219

theorem polar_to_cartesian_coordinates :
  let r : ℝ := 2
  let θ : ℝ := π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 1 ∧ y = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_coordinates_l52_5219


namespace NUMINAMATH_CALUDE_cube_root_unity_inverse_l52_5261

/-- Given a complex cube root of unity ω, prove that (ω - ω⁻¹)⁻¹ = -(1 + 2ω²)/5 -/
theorem cube_root_unity_inverse (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω ≠ 1) :
  (ω - ω⁻¹)⁻¹ = -(1 + 2*ω^2)/5 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_inverse_l52_5261


namespace NUMINAMATH_CALUDE_problem_solution_l52_5222

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a / x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + a * x - 6 * log x

def h (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x + 4

theorem problem_solution :
  -- Part I
  (∀ a x, x > 0 → 
    (a ≥ 0 → (deriv (f a)) x > 0) ∧ 
    (a < 0 → ((0 < x ∧ x < -a) → (deriv (f a)) x < 0) ∧ 
             (x > -a → (deriv (f a)) x > 0))) ∧
  -- Part II
  (∀ a, (∀ x, x > 0 → (deriv (g a)) x ≥ 0) → a ≥ 5/2) ∧
  -- Part III
  (∀ m, (∃ x₁, 0 < x₁ ∧ x₁ < 1 ∧ 
        ∀ x₂, 1 ≤ x₂ ∧ x₂ ≤ 2 → g 2 x₁ ≥ h m x₂) → 
        m ≥ 8 - 5 * log 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l52_5222


namespace NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l52_5278

/-- Calculates the profit percentage of a middleman in a series of transactions -/
theorem cricket_bat_profit_percentage 
  (a_cost : ℝ) 
  (a_profit_percent : ℝ) 
  (c_price : ℝ) 
  (h1 : a_cost = 152)
  (h2 : a_profit_percent = 20)
  (h3 : c_price = 228) :
  let a_sell := a_cost * (1 + a_profit_percent / 100)
  let b_profit := c_price - a_sell
  let b_profit_percent := (b_profit / a_sell) * 100
  b_profit_percent = 25 := by
sorry


end NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l52_5278


namespace NUMINAMATH_CALUDE_f_1000_is_even_l52_5201

/-- A function that satisfies the given functional equation -/
def SatisfiesEquation (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → (f^[f n] n) = n^2 / (f (f n))

/-- Theorem stating that f(1000) is even for any function satisfying the equation -/
theorem f_1000_is_even (f : ℕ → ℕ) (h : SatisfiesEquation f) : 
  ∃ k : ℕ, f 1000 = 2 * k :=
sorry

end NUMINAMATH_CALUDE_f_1000_is_even_l52_5201


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_19_l52_5239

theorem greatest_three_digit_multiple_of_19 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → 19 ∣ n → n ≤ 988 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_19_l52_5239


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l52_5293

theorem arithmetic_calculation : 3 + (12 / 3 - 1)^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l52_5293


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l52_5236

/-- The number of varieties of wrapping paper -/
def wrapping_paper_varieties : ℕ := 8

/-- The number of colors of ribbon -/
def ribbon_colors : ℕ := 3

/-- The number of types of gift cards -/
def gift_card_types : ℕ := 4

/-- The total number of possible gift wrapping combinations -/
def total_combinations : ℕ := wrapping_paper_varieties * ribbon_colors * gift_card_types

/-- Theorem stating that the total number of combinations is 96 -/
theorem gift_wrapping_combinations :
  total_combinations = 96 := by sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l52_5236


namespace NUMINAMATH_CALUDE_octagon_diagonals_eq_twenty_l52_5211

/-- The number of diagonals in an octagon -/
def octagon_diagonals : ℕ :=
  let n := 8  -- number of vertices in an octagon
  let sides := 8  -- number of sides in an octagon
  (n * (n - 1)) / 2 - sides

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals_eq_twenty : octagon_diagonals = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_eq_twenty_l52_5211


namespace NUMINAMATH_CALUDE_f_at_one_eq_neg_7878_l52_5255

/-- The polynomial g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 20

/-- The polynomial f(x) -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 120*x + c

/-- Theorem stating that f(1) = -7878 under given conditions -/
theorem f_at_one_eq_neg_7878 (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g a x = 0 ∧ g a y = 0 ∧ g a z = 0) →  -- g has three distinct roots
  (∀ x : ℝ, g a x = 0 → f b c x = 0) →  -- Each root of g is a root of f
  g a (-100) = 0 →  -- -100 is a root of g
  f b c 1 = -7878 :=
by sorry

end NUMINAMATH_CALUDE_f_at_one_eq_neg_7878_l52_5255


namespace NUMINAMATH_CALUDE_binomial_floor_divisibility_l52_5277

theorem binomial_floor_divisibility (p n : ℕ) (hp : Nat.Prime p) :
  p ∣ (Nat.choose n p - n / p) := by
  sorry

end NUMINAMATH_CALUDE_binomial_floor_divisibility_l52_5277


namespace NUMINAMATH_CALUDE_rectangular_field_breadth_breadth_approximation_l52_5208

/-- The breadth of a rectangular field with length 90 meters, 
    whose area is equal to a square plot with diagonal 120 meters. -/
theorem rectangular_field_breadth : ℝ :=
  let rectangular_length : ℝ := 90
  let square_diagonal : ℝ := 120
  let square_side : ℝ := square_diagonal / Real.sqrt 2
  let square_area : ℝ := square_side ^ 2
  let rectangular_area : ℝ := square_area
  rectangular_area / rectangular_length

/-- The breadth of the rectangular field is approximately 80 meters. -/
theorem breadth_approximation (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, abs (rectangular_field_breadth - 80) < δ ∧ δ < ε :=
sorry

end NUMINAMATH_CALUDE_rectangular_field_breadth_breadth_approximation_l52_5208


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l52_5297

theorem boat_speed_ratio (boat_speed : ℝ) (current_speed : ℝ) 
  (h1 : boat_speed = 15)
  (h2 : current_speed = 5)
  (h3 : boat_speed > current_speed) :
  let downstream_speed := boat_speed + current_speed
  let upstream_speed := boat_speed - current_speed
  let avg_speed := 2 / (1 / downstream_speed + 1 / upstream_speed)
  avg_speed / boat_speed = 8 / 9 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l52_5297


namespace NUMINAMATH_CALUDE_identical_views_sphere_or_cube_l52_5253

-- Define a type for solids
structure Solid where
  -- Add any necessary properties

-- Define a function to represent the view of a solid
def view (s : Solid) : Set Point := sorry

-- Define spheres and cubes as specific types of solids
def Sphere : Solid := sorry
def Cube : Solid := sorry

-- Theorem stating that a solid with three identical views could be a sphere or a cube
theorem identical_views_sphere_or_cube (s : Solid) :
  (∃ v : Set Point, view s = v ∧ view s = v ∧ view s = v) →
  s = Sphere ∨ s = Cube :=
sorry

end NUMINAMATH_CALUDE_identical_views_sphere_or_cube_l52_5253


namespace NUMINAMATH_CALUDE_haley_weight_l52_5292

/-- Given the weights of Verna, Haley, and Sherry, prove Haley's weight -/
theorem haley_weight (V H S : ℝ) 
  (verna_haley : V = H + 17)
  (verna_sherry : V = S / 2)
  (total_weight : V + S = 360) :
  H = 103 := by
  sorry

end NUMINAMATH_CALUDE_haley_weight_l52_5292


namespace NUMINAMATH_CALUDE_game_question_count_l52_5228

theorem game_question_count (total_questions : ℕ) (correct_reward : ℕ) (incorrect_penalty : ℕ) 
  (h1 : total_questions = 50)
  (h2 : correct_reward = 7)
  (h3 : incorrect_penalty = 3)
  : ∃ (correct_answers : ℕ), 
    correct_answers * correct_reward = (total_questions - correct_answers) * incorrect_penalty ∧ 
    correct_answers = 15 := by
  sorry

end NUMINAMATH_CALUDE_game_question_count_l52_5228


namespace NUMINAMATH_CALUDE_rectangle_width_proof_l52_5263

theorem rectangle_width_proof (length width : ℝ) : 
  length = 24 →
  2 * length + 2 * width = 80 →
  length / width = 6 / 5 →
  width = 16 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_proof_l52_5263


namespace NUMINAMATH_CALUDE_system_of_equations_l52_5290

/-- Given a system of equations, prove the values of x, y, and z. -/
theorem system_of_equations : 
  let x := 80 * (1 + 0.11)
  let y := 120 * (1 - 0.15)
  let z := (0.4 * (x + y)) * (1 + 0.2)
  (x = 88.8) ∧ (y = 102) ∧ (z = 91.584) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_l52_5290


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l52_5251

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l52_5251


namespace NUMINAMATH_CALUDE_binary_equals_octal_l52_5241

-- Define the binary number
def binary_num : List Bool := [true, false, true, true, true, false]

-- Define the octal number
def octal_num : Nat := 56

-- Function to convert binary to decimal
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Function to convert decimal to octal
def decimal_to_octal (n : Nat) : Nat :=
  if n < 8 then n
  else 10 * (decimal_to_octal (n / 8)) + (n % 8)

-- Theorem stating that the binary number is equal to the octal number
theorem binary_equals_octal : 
  decimal_to_octal (binary_to_decimal binary_num) = octal_num := by
  sorry

end NUMINAMATH_CALUDE_binary_equals_octal_l52_5241


namespace NUMINAMATH_CALUDE_function_evaluation_l52_5288

/-- Given a function f(x) = x^2 + 1, prove that f(a+1) = a^2 + 2a + 2 -/
theorem function_evaluation (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 1
  f (a + 1) = a^2 + 2*a + 2 := by
sorry

end NUMINAMATH_CALUDE_function_evaluation_l52_5288


namespace NUMINAMATH_CALUDE_bug_return_probability_l52_5204

/-- Represents the probability of the bug being at the starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 - Q n) / 2

/-- The probability of returning to the starting vertex on the 12th move in a square -/
theorem bug_return_probability : Q 12 = 683 / 2048 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l52_5204


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l52_5296

/-- Given a geometric sequence where the last four terms are a, b, 243, 729,
    prove that the first term of the sequence is 3. -/
theorem geometric_sequence_first_term
  (a b : ℝ)
  (h1 : ∃ (r : ℝ), r ≠ 0 ∧ b = a * r ∧ 243 = b * r ∧ 729 = 243 * r)
  : ∃ (n : ℕ), 3 * (a / 243) ^ n = 1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l52_5296


namespace NUMINAMATH_CALUDE_train_speed_l52_5273

/-- Proves that a train with given length, crossing a bridge of given length in a specific time, has a specific speed in km/hr -/
theorem train_speed (train_length bridge_length : Real) (crossing_time : Real) :
  train_length = 110 →
  bridge_length = 132 →
  crossing_time = 24.198064154867613 →
  (train_length + bridge_length) / crossing_time * 3.6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l52_5273


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l52_5227

/-- The function f(x) = x^3 - x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem tangent_line_intersection (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = -1 ∧
  (f a x₁ = (a + 1) * x₁) ∧
  (f a x₂ = (a + 1) * x₂) ∧
  (∀ x : ℝ, f a x = (a + 1) * x → x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l52_5227


namespace NUMINAMATH_CALUDE_xiao_ming_reading_problem_l52_5279

/-- Represents the problem of finding the minimum number of pages to read per day -/
def min_pages_per_day (total_pages : ℕ) (total_days : ℕ) (initial_days : ℕ) (initial_pages_per_day : ℕ) : ℕ :=
  let remaining_days := total_days - initial_days
  let remaining_pages := total_pages - (initial_days * initial_pages_per_day)
  (remaining_pages + remaining_days - 1) / remaining_days

/-- Theorem stating the solution to Xiao Ming's reading problem -/
theorem xiao_ming_reading_problem :
  min_pages_per_day 72 10 2 5 = 8 :=
by sorry

end NUMINAMATH_CALUDE_xiao_ming_reading_problem_l52_5279


namespace NUMINAMATH_CALUDE_mean_score_approx_71_l52_5217

/-- Calculates the mean score of all students given the mean scores of two classes and the ratio of students in those classes. -/
def meanScoreAllStudents (morningMean afternoon_mean : ℚ) (morningStudents afternoonStudents : ℕ) : ℚ :=
  let totalStudents := morningStudents + afternoonStudents
  let totalScore := morningMean * morningStudents + afternoon_mean * afternoonStudents
  totalScore / totalStudents

/-- Proves that the mean score of all students is approximately 71 given the specified conditions. -/
theorem mean_score_approx_71 :
  ∃ (m a : ℕ), m > 0 ∧ a > 0 ∧ m = (5 * a) / 7 ∧ 
  abs (meanScoreAllStudents 78 65 m a - 71) < 1 :=
sorry


end NUMINAMATH_CALUDE_mean_score_approx_71_l52_5217


namespace NUMINAMATH_CALUDE_mn_sum_is_negative_two_l52_5238

theorem mn_sum_is_negative_two (m n : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 6 = (x-2)*(x-n)) → m + n = -2 := by
sorry

end NUMINAMATH_CALUDE_mn_sum_is_negative_two_l52_5238


namespace NUMINAMATH_CALUDE_max_distance_for_given_tires_l52_5294

/-- Represents the maximum distance a car can travel with one tire swap --/
def max_distance_with_swap (front_tire_life : ℕ) (rear_tire_life : ℕ) : ℕ :=
  front_tire_life + (rear_tire_life - front_tire_life) / 2

/-- Theorem: Given specific tire lifespans, the maximum distance with one swap is 48,000 km --/
theorem max_distance_for_given_tires :
  max_distance_with_swap 42000 56000 = 48000 := by
  sorry

#eval max_distance_with_swap 42000 56000

end NUMINAMATH_CALUDE_max_distance_for_given_tires_l52_5294


namespace NUMINAMATH_CALUDE_special_cylinder_lateral_area_l52_5245

/-- A cylinder with base area S and lateral surface that unfolds into a square -/
structure SpecialCylinder where
  S : ℝ
  baseArea : S > 0
  lateralSurfaceIsSquare : True

/-- The lateral surface area of a SpecialCylinder is 4πS -/
theorem special_cylinder_lateral_area (c : SpecialCylinder) :
  ∃ (lateralArea : ℝ), lateralArea = 4 * Real.pi * c.S := by
  sorry

end NUMINAMATH_CALUDE_special_cylinder_lateral_area_l52_5245


namespace NUMINAMATH_CALUDE_erased_grid_squares_l52_5291

/-- Represents a square grid with erased line segments -/
structure ErasedSquareGrid :=
  (size : Nat)
  (erasedLines : Nat)

/-- Counts the number of squares of a given size in the grid -/
def countSquares (grid : ErasedSquareGrid) (squareSize : Nat) : Nat :=
  sorry

/-- Calculates the total number of squares of all sizes in the grid -/
def totalSquares (grid : ErasedSquareGrid) : Nat :=
  sorry

/-- The main theorem stating that a 4x4 grid with 2 erased lines has 22 squares -/
theorem erased_grid_squares :
  let grid : ErasedSquareGrid := ⟨4, 2⟩
  totalSquares grid = 22 :=
by sorry

end NUMINAMATH_CALUDE_erased_grid_squares_l52_5291


namespace NUMINAMATH_CALUDE_bridge_length_l52_5218

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 265 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l52_5218


namespace NUMINAMATH_CALUDE_unique_A_for_multiple_of_9_l52_5275

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def sum_of_digits (A : ℕ) : ℕ := 2 + A + 3 + A

def four_digit_number (A : ℕ) : ℕ := 2000 + 100 * A + 30 + A

theorem unique_A_for_multiple_of_9 :
  ∃! A : ℕ, A < 10 ∧ is_multiple_of_9 (four_digit_number A) ∧ A = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_A_for_multiple_of_9_l52_5275


namespace NUMINAMATH_CALUDE_determinant_equals_x_squared_plus_y_squared_l52_5203

theorem determinant_equals_x_squared_plus_y_squared (x y : ℝ) : 
  Matrix.det !![1, x, y; 1, x - y, y; 1, x, y - x] = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equals_x_squared_plus_y_squared_l52_5203


namespace NUMINAMATH_CALUDE_pat_stickers_end_of_week_l52_5252

/-- The number of stickers Pat had at the end of the week -/
def total_stickers (initial : ℕ) (earned : ℕ) : ℕ := initial + earned

/-- Theorem: Pat had 61 stickers at the end of the week -/
theorem pat_stickers_end_of_week :
  total_stickers 39 22 = 61 := by
  sorry

end NUMINAMATH_CALUDE_pat_stickers_end_of_week_l52_5252


namespace NUMINAMATH_CALUDE_intersection_perpendicular_tangents_l52_5246

open Real

theorem intersection_perpendicular_tangents (a : ℝ) :
  ∃ x ∈ Set.Ioo 0 (π / 2),
    2 * sin x = a * cos x ∧
    (2 * cos x) * (-a * sin x) = -1 →
  a = 2 * sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_tangents_l52_5246


namespace NUMINAMATH_CALUDE_parakeet_to_kitten_ratio_l52_5298

-- Define the number of each type of pet
def num_puppies : ℕ := 2
def num_kittens : ℕ := 2
def num_parakeets : ℕ := 3

-- Define the cost of a parakeet
def parakeet_cost : ℕ := 10

-- Define the relationship between puppy and parakeet costs
def puppy_cost : ℕ := 3 * parakeet_cost

-- Define the total cost of all pets
def total_cost : ℕ := 130

-- Define the cost of a kitten (to be proved)
def kitten_cost : ℕ := (total_cost - num_puppies * puppy_cost - num_parakeets * parakeet_cost) / num_kittens

-- Theorem to prove the ratio of parakeet cost to kitten cost
theorem parakeet_to_kitten_ratio :
  parakeet_cost * 2 = kitten_cost :=
by sorry

end NUMINAMATH_CALUDE_parakeet_to_kitten_ratio_l52_5298


namespace NUMINAMATH_CALUDE_infinitely_many_satisfying_points_l52_5287

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2}

-- Define the diameter endpoints
def DiameterEndpoints (center : ℝ × ℝ) (radius : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((center.1 - radius, center.2), (center.1 + radius, center.2))

-- Define the distance squared between two points
def DistanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Define the set of points P satisfying the condition
def SatisfyingPoints (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | p ∈ Circle center radius ∧
               let (a, b) := DiameterEndpoints center radius
               DistanceSquared p a + DistanceSquared p b = 10}

-- Theorem statement
theorem infinitely_many_satisfying_points (center : ℝ × ℝ) :
  Set.Infinite (SatisfyingPoints center 2) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_satisfying_points_l52_5287


namespace NUMINAMATH_CALUDE_cubic_sum_problem_l52_5265

theorem cubic_sum_problem (a b c : ℝ) 
  (h1 : a + b + c = 7)
  (h2 : a * b + a * c + b * c = 11)
  (h3 : a * b * c = -6) :
  a^3 + b^3 + c^3 = 223 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_problem_l52_5265


namespace NUMINAMATH_CALUDE_rhombus_area_l52_5285

/-- The area of a rhombus with side length 4 and a 45-degree angle between adjacent sides is 8√2 -/
theorem rhombus_area (side : ℝ) (angle : ℝ) : 
  side = 4 → angle = π / 4 → 
  let area := side * side * Real.sin angle
  area = 8 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l52_5285


namespace NUMINAMATH_CALUDE_six_tangent_circles_l52_5212

-- Define the circles C₁ and C₂
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of being tangent
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

-- Define the problem setup
def problem_setup (C₁ C₂ : Circle) : Prop :=
  C₁.radius = 2 ∧
  C₂.radius = 2 ∧
  are_tangent C₁ C₂

-- Define a function to count tangent circles
def count_tangent_circles (C₁ C₂ : Circle) : ℕ :=
  sorry -- The actual counting logic would go here

-- The main theorem
theorem six_tangent_circles (C₁ C₂ : Circle) :
  problem_setup C₁ C₂ → count_tangent_circles C₁ C₂ = 6 :=
by sorry


end NUMINAMATH_CALUDE_six_tangent_circles_l52_5212


namespace NUMINAMATH_CALUDE_mole_winter_survival_l52_5243

/-- Represents the Mole's food storage --/
structure MoleStorage :=
  (grain : ℕ)
  (millet : ℕ)

/-- Represents a monthly consumption plan --/
inductive ConsumptionPlan
  | AllGrain
  | MixedDiet

/-- The Mole's winter survival problem --/
theorem mole_winter_survival 
  (initial_grain : ℕ)
  (storage_capacity : ℕ)
  (exchange_rate : ℕ)
  (winter_duration : ℕ)
  (h_initial_grain : initial_grain = 8)
  (h_storage_capacity : storage_capacity = 12)
  (h_exchange_rate : exchange_rate = 2)
  (h_winter_duration : winter_duration = 3)
  : ∃ (exchange_amount : ℕ) 
      (final_storage : MoleStorage) 
      (consumption_plan : Fin winter_duration → ConsumptionPlan),
    -- Exchange constraint
    exchange_amount ≤ initial_grain ∧
    -- Storage capacity constraint
    final_storage.grain + final_storage.millet ≤ storage_capacity ∧
    -- Exchange calculation
    final_storage.grain = initial_grain - exchange_amount ∧
    final_storage.millet = exchange_amount * exchange_rate ∧
    -- Survival constraint
    (∀ month : Fin winter_duration,
      (consumption_plan month = ConsumptionPlan.AllGrain → 
        ∃ remaining_storage : MoleStorage,
          remaining_storage.grain = final_storage.grain - 3 * (month.val + 1) ∧
          remaining_storage.millet = final_storage.millet) ∧
      (consumption_plan month = ConsumptionPlan.MixedDiet →
        ∃ remaining_storage : MoleStorage,
          remaining_storage.grain = final_storage.grain - (month.val + 1) ∧
          remaining_storage.millet = final_storage.millet - 3 * (month.val + 1))) ∧
    -- Final state
    ∃ final_state : MoleStorage,
      final_state.grain = 0 ∧ final_state.millet = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_mole_winter_survival_l52_5243


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l52_5244

/-- A quadratic function with a symmetry axis at x = -1 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  symmetry_axis : a ≠ 0 ∧ -|b| / (2 * a) = -1

/-- Three points on the parabola -/
structure ParabolaPoints where
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  on_parabola : ∀ (f : QuadraticFunction),
    f.a * (-14/3)^2 + |f.b| * (-14/3) + f.c = y₁ ∧
    f.a * (5/2)^2 + |f.b| * (5/2) + f.c = y₂ ∧
    f.a * 3^2 + |f.b| * 3 + f.c = y₃

/-- Theorem stating the relationship between y₁, y₂, and y₃ -/
theorem parabola_point_relationship (f : QuadraticFunction) (p : ParabolaPoints) :
  p.y₂ < p.y₁ ∧ p.y₁ < p.y₃ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l52_5244


namespace NUMINAMATH_CALUDE_lcm_of_26_and_16_l52_5216

theorem lcm_of_26_and_16 :
  let n : ℕ := 26
  let m : ℕ := 16
  let gcf : ℕ := 8
  Nat.lcm n m = 52 ∧ Nat.gcd n m = gcf :=
by sorry

end NUMINAMATH_CALUDE_lcm_of_26_and_16_l52_5216


namespace NUMINAMATH_CALUDE_complex_square_root_of_negative_one_l52_5247

theorem complex_square_root_of_negative_one (z : ℂ) : 
  (z - 1)^2 = -1 → z = 1 + I ∨ z = 1 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_root_of_negative_one_l52_5247


namespace NUMINAMATH_CALUDE_jellybean_probability_l52_5299

def total_jellybeans : ℕ := 15
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 3
def white_jellybeans : ℕ := 5
def green_jellybeans : ℕ := 2
def picked_jellybeans : ℕ := 4

theorem jellybean_probability : 
  (Nat.choose red_jellybeans 3 * Nat.choose (total_jellybeans - red_jellybeans) 1) / 
  Nat.choose total_jellybeans picked_jellybeans = 20 / 273 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_probability_l52_5299


namespace NUMINAMATH_CALUDE_some_magical_not_spooky_l52_5225

universe u

-- Define the types
variable {Creature : Type u}

-- Define the predicates
variable (Dragon : Creature → Prop)
variable (Magical : Creature → Prop)
variable (Spooky : Creature → Prop)

-- State the theorem
theorem some_magical_not_spooky
  (h1 : ∀ x, Dragon x → Magical x)
  (h2 : ∀ x, Spooky x → ¬ Dragon x) :
  ∃ x, Magical x ∧ ¬ Spooky x :=
sorry

end NUMINAMATH_CALUDE_some_magical_not_spooky_l52_5225


namespace NUMINAMATH_CALUDE_inverse_of_sixteen_point_six_periodic_l52_5223

/-- Given that 1 divided by a number is equal to 16.666666666666668,
    prove that the number is equal to 1/60. -/
theorem inverse_of_sixteen_point_six_periodic : ∃ x : ℚ, (1 : ℚ) / x = 16666666666666668 / 1000000000000000 ∧ x = 1 / 60 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_sixteen_point_six_periodic_l52_5223


namespace NUMINAMATH_CALUDE_range_of_G_l52_5283

/-- The function G(x) defined as |x+1|-|x-1| for all real x -/
def G (x : ℝ) : ℝ := |x + 1| - |x - 1|

/-- The range of G(x) is [-2,2] -/
theorem range_of_G : Set.range G = Set.Icc (-2) 2 := by sorry

end NUMINAMATH_CALUDE_range_of_G_l52_5283


namespace NUMINAMATH_CALUDE_sausage_division_ratio_l52_5240

/-- Represents the length of sausage remaining after each bite -/
def remaining_sausage : ℕ → ℚ
  | 0 => 1
  | n + 1 => if n % 2 = 0
              then (3/4) * remaining_sausage n
              else (2/3) * remaining_sausage n

/-- The ratio of sausage remaining approaches 1/2 as the number of bites increases -/
theorem sausage_division_ratio :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |remaining_sausage n - 1/2| < ε :=
sorry

#check sausage_division_ratio

end NUMINAMATH_CALUDE_sausage_division_ratio_l52_5240


namespace NUMINAMATH_CALUDE_product_of_solutions_l52_5260

theorem product_of_solutions (x₁ x₂ : ℝ) : 
  (|6 * x₁| + 5 = 47) → (|6 * x₂| + 5 = 47) → x₁ * x₂ = -49 := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l52_5260


namespace NUMINAMATH_CALUDE_circle_equation_l52_5221

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def pointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (h, k) := c.center
  (x - h)^2 + (y - k)^2 = c.radius^2

/-- Check if a circle is tangent to the y-axis -/
def tangentToYAxis (c : Circle) : Prop :=
  c.center.1 = c.radius

/-- Check if a point lies on the line x - 3y = 0 -/
def pointOnLine (p : ℝ × ℝ) : Prop :=
  p.1 - 3 * p.2 = 0

theorem circle_equation (c : Circle) :
  tangentToYAxis c ∧ 
  pointOnLine c.center ∧ 
  pointOnCircle c (6, 1) →
  c.center = (3, 1) ∧ c.radius = 3 :=
by sorry

#check circle_equation

end NUMINAMATH_CALUDE_circle_equation_l52_5221


namespace NUMINAMATH_CALUDE_g_sum_equals_one_l52_5258

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom func_property : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y
axiom f_nonzero : ∀ x : ℝ, f x ≠ 0
axiom f_equal : f 1 = f 2

-- State the theorem
theorem g_sum_equals_one : g (-1) + g 1 = 1 := by sorry

end NUMINAMATH_CALUDE_g_sum_equals_one_l52_5258


namespace NUMINAMATH_CALUDE_number_solution_l52_5259

theorem number_solution (z s : ℝ) (n : ℝ) : 
  z ≠ 0 → 
  z = Real.sqrt (n * z * s - 9 * s^2) → 
  z = 3 → 
  n = 3 + 3 * s := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l52_5259


namespace NUMINAMATH_CALUDE_horner_method_v3_l52_5248

def f (x : ℝ) : ℝ := 5*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

def horner_step (a : ℝ) (x : ℝ) (v : ℝ) : ℝ := a*x + v

def horner_v3 (x : ℝ) : ℝ :=
  let v0 := 5
  let v1 := horner_step 2 x v0
  let v2 := horner_step 3.5 x v1
  horner_step (-2.6) x v2

theorem horner_method_v3 :
  horner_v3 1 = 7.9 :=
sorry

end NUMINAMATH_CALUDE_horner_method_v3_l52_5248


namespace NUMINAMATH_CALUDE_y_derivative_l52_5276

/-- The function y in terms of x -/
def y (x : ℝ) : ℝ := (3 * x - 2) ^ 2

/-- The derivative of y with respect to x -/
def y' (x : ℝ) : ℝ := 6 * (3 * x - 2)

/-- Theorem stating that y' is the derivative of y -/
theorem y_derivative : ∀ x, deriv y x = y' x := by sorry

end NUMINAMATH_CALUDE_y_derivative_l52_5276


namespace NUMINAMATH_CALUDE_least_x_value_l52_5207

theorem least_x_value (x p : ℕ) (h1 : x > 0) (h2 : Nat.Prime p) 
  (h3 : ∃ q : ℕ, Nat.Prime q ∧ q ≠ 2 ∧ x = 12 * p * q) : x ≥ 72 := by
  sorry

end NUMINAMATH_CALUDE_least_x_value_l52_5207


namespace NUMINAMATH_CALUDE_smallest_additional_divisor_l52_5206

def divisors : Set Nat := {30, 48, 74, 100}

theorem smallest_additional_divisor :
  ∃ (n : Nat), n > 0 ∧ 
  (∀ m ∈ divisors, (44402 + 2) % m = 0) ∧
  (44402 + 2) % n = 0 ∧
  n ∉ divisors ∧
  (∀ k : Nat, 0 < k ∧ k < n → (44402 + 2) % k ≠ 0 ∨ k ∈ divisors) ∧
  n = 37 := by
  sorry

end NUMINAMATH_CALUDE_smallest_additional_divisor_l52_5206


namespace NUMINAMATH_CALUDE_delaney_missed_bus_l52_5220

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem delaney_missed_bus (busLeaveTime : Time) (travelTime : Nat) (leftHomeTime : Time) :
  busLeaveTime = { hours := 8, minutes := 0 } →
  travelTime = 30 →
  leftHomeTime = { hours := 7, minutes := 50 } →
  timeDifference (addMinutes leftHomeTime travelTime) busLeaveTime = 20 := by
  sorry

end NUMINAMATH_CALUDE_delaney_missed_bus_l52_5220


namespace NUMINAMATH_CALUDE_park_visitors_total_l52_5256

theorem park_visitors_total (saturday_visitors : ℕ) (sunday_extra : ℕ) : 
  saturday_visitors = 200 → sunday_extra = 40 → 
  saturday_visitors + (saturday_visitors + sunday_extra) = 440 := by
sorry

end NUMINAMATH_CALUDE_park_visitors_total_l52_5256


namespace NUMINAMATH_CALUDE_quadratic_minimum_l52_5209

def f (x : ℝ) := x^2 - 12*x + 28

theorem quadratic_minimum (x : ℝ) : f x ≥ f 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l52_5209


namespace NUMINAMATH_CALUDE_riza_age_proof_l52_5250

/-- Represents Riza's age when her son was born -/
def riza_age_at_birth : ℕ := 25

/-- Represents the current age of Riza's son -/
def son_current_age : ℕ := 40

/-- Represents the sum of Riza's and her son's current ages -/
def sum_of_ages : ℕ := 105

theorem riza_age_proof : 
  riza_age_at_birth + son_current_age + son_current_age = sum_of_ages := by
  sorry

end NUMINAMATH_CALUDE_riza_age_proof_l52_5250


namespace NUMINAMATH_CALUDE_variable_value_l52_5270

theorem variable_value : 
  ∀ (a n some_variable : ℤ) (x : ℝ),
  (3 * x + 2) * (2 * x - 7) = a * x^2 + some_variable * x + n →
  a - n + some_variable = 3 →
  some_variable = -17 := by
sorry

end NUMINAMATH_CALUDE_variable_value_l52_5270


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l52_5232

theorem factorial_fraction_simplification :
  (4 * Nat.factorial 6 + 20 * Nat.factorial 5 + 48 * Nat.factorial 4) / Nat.factorial 7 = 134 / 105 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l52_5232


namespace NUMINAMATH_CALUDE_cube_root_function_l52_5202

theorem cube_root_function (k : ℝ) : 
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y = k * x^(1/3)) →
  (4 * Real.sqrt 3 = k * 64^(1/3)) →
  (2 * Real.sqrt 3 = k * 8^(1/3)) := by sorry

end NUMINAMATH_CALUDE_cube_root_function_l52_5202


namespace NUMINAMATH_CALUDE_min_luxury_owners_l52_5205

structure Village where
  population : ℕ
  refrigerator_owners : Finset Nat
  television_owners : Finset Nat
  computer_owners : Finset Nat
  air_conditioner_owners : Finset Nat
  washing_machine_owners : Finset Nat
  microwave_owners : Finset Nat
  internet_owners : Finset Nat
  top_earners : Finset Nat

def Owlna (v : Village) : Prop :=
  v.refrigerator_owners.card = (67 * v.population) / 100 ∧
  v.television_owners.card = (74 * v.population) / 100 ∧
  v.computer_owners.card = (77 * v.population) / 100 ∧
  v.air_conditioner_owners.card = (83 * v.population) / 100 ∧
  v.washing_machine_owners.card = (55 * v.population) / 100 ∧
  v.microwave_owners.card = (48 * v.population) / 100 ∧
  v.internet_owners.card = (42 * v.population) / 100 ∧
  (v.television_owners ∩ v.computer_owners).card = (35 * v.population) / 100 ∧
  (v.washing_machine_owners ∩ v.microwave_owners).card = (30 * v.population) / 100 ∧
  (v.air_conditioner_owners ∩ v.refrigerator_owners).card = (27 * v.population) / 100 ∧
  v.top_earners.card = (10 * v.population) / 100 ∧
  (v.refrigerator_owners ∩ v.television_owners ∩ v.computer_owners ∩
   v.air_conditioner_owners ∩ v.washing_machine_owners ∩ v.microwave_owners ∩
   v.internet_owners) ⊆ v.top_earners

theorem min_luxury_owners (v : Village) (h : Owlna v) :
  (v.refrigerator_owners ∩ v.television_owners ∩ v.computer_owners ∩
   v.air_conditioner_owners ∩ v.washing_machine_owners ∩ v.microwave_owners ∩
   v.internet_owners ∩ v.top_earners).card = (10 * v.population) / 100 :=
by sorry

end NUMINAMATH_CALUDE_min_luxury_owners_l52_5205


namespace NUMINAMATH_CALUDE_min_value_of_objective_function_l52_5268

def objective_function (x y : ℝ) : ℝ := x + 3 * y

def constraint1 (x y : ℝ) : Prop := x + y - 2 ≥ 0
def constraint2 (x y : ℝ) : Prop := x - y - 2 ≤ 0
def constraint3 (y : ℝ) : Prop := y ≥ 1

theorem min_value_of_objective_function :
  ∀ x y : ℝ, constraint1 x y → constraint2 x y → constraint3 y →
  ∀ x' y' : ℝ, constraint1 x' y' → constraint2 x' y' → constraint3 y' →
  objective_function x y ≥ 4 ∧
  (∃ x₀ y₀ : ℝ, constraint1 x₀ y₀ ∧ constraint2 x₀ y₀ ∧ constraint3 y₀ ∧ objective_function x₀ y₀ = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_objective_function_l52_5268


namespace NUMINAMATH_CALUDE_evaluate_expression_l52_5281

theorem evaluate_expression : 
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) + 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l52_5281


namespace NUMINAMATH_CALUDE_different_color_probability_l52_5284

/-- The probability of drawing two balls of different colors from a box containing 2 red balls and 3 black balls -/
theorem different_color_probability (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) : 
  total_balls = 5 →
  red_balls = 2 →
  black_balls = 3 →
  (red_balls * black_balls : ℚ) / ((total_balls * (total_balls - 1)) / 2) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_different_color_probability_l52_5284


namespace NUMINAMATH_CALUDE_warehouse_boxes_theorem_l52_5233

/-- The number of boxes in two warehouses -/
def total_boxes (first_warehouse : ℕ) (second_warehouse : ℕ) : ℕ :=
  first_warehouse + second_warehouse

theorem warehouse_boxes_theorem (first_warehouse second_warehouse : ℕ) 
  (h1 : first_warehouse = 400)
  (h2 : first_warehouse = 2 * second_warehouse) : 
  total_boxes first_warehouse second_warehouse = 600 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_boxes_theorem_l52_5233


namespace NUMINAMATH_CALUDE_min_value_of_permutation_sum_l52_5282

theorem min_value_of_permutation_sum :
  ∀ (x₁ x₂ x₃ x₄ x₅ : ℕ),
  (x₁ :: x₂ :: x₃ :: x₄ :: x₅ :: []).Perm [1, 2, 3, 4, 5] →
  (∀ (y₁ y₂ y₃ y₄ y₅ : ℕ),
    (y₁ :: y₂ :: y₃ :: y₄ :: y₅ :: []).Perm [1, 2, 3, 4, 5] →
    x₁ + 2*x₂ + 3*x₃ + 4*x₄ + 5*x₅ ≤ y₁ + 2*y₂ + 3*y₃ + 4*y₄ + 5*y₅) →
  x₁ + 2*x₂ + 3*x₃ + 4*x₄ + 5*x₅ = 35 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_permutation_sum_l52_5282


namespace NUMINAMATH_CALUDE_root_sum_theorem_l52_5295

def cubic_equation (x : ℝ) : Prop := 60 * x^3 - 70 * x^2 + 24 * x - 2 = 0

theorem root_sum_theorem (p q r : ℝ) :
  cubic_equation p ∧ cubic_equation q ∧ cubic_equation r ∧
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  0 < p ∧ p < 2 ∧ 0 < q ∧ q < 2 ∧ 0 < r ∧ r < 2 →
  1 / (2 - p) + 1 / (2 - q) + 1 / (2 - r) = 116 / 15 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l52_5295


namespace NUMINAMATH_CALUDE_sqrt_of_neg_nine_l52_5231

-- Define the square root of a complex number
def complex_sqrt (z : ℂ) : Set ℂ :=
  {w : ℂ | w^2 = z}

-- Theorem statement
theorem sqrt_of_neg_nine :
  complex_sqrt (-9 : ℂ) = {3*I, -3*I} :=
sorry

end NUMINAMATH_CALUDE_sqrt_of_neg_nine_l52_5231


namespace NUMINAMATH_CALUDE_pen_sale_profit_percent_l52_5226

/-- Calculates the profit percent for a pen sale scenario -/
theorem pen_sale_profit_percent 
  (num_pens : ℕ)
  (purchase_price : ℕ)
  (discount_percent : ℚ)
  (h1 : num_pens = 60)
  (h2 : purchase_price = 46)
  (h3 : discount_percent = 1/100) :
  ∃ (profit_percent : ℚ), abs (profit_percent - 2913/10000) < 1/10000 := by
  sorry

end NUMINAMATH_CALUDE_pen_sale_profit_percent_l52_5226


namespace NUMINAMATH_CALUDE_sixteen_power_division_plus_two_l52_5230

theorem sixteen_power_division_plus_two (m : ℕ) : 
  m = 16^2023 → m / 8 + 2 = 2^8089 + 2 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_power_division_plus_two_l52_5230


namespace NUMINAMATH_CALUDE_impossibility_theorem_l52_5286

theorem impossibility_theorem (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) : 
  ¬(((1 - a) * b > 1/4) ∧ ((1 - b) * c > 1/4) ∧ ((1 - c) * a > 1/4)) := by
  sorry

end NUMINAMATH_CALUDE_impossibility_theorem_l52_5286


namespace NUMINAMATH_CALUDE_complex_sum_real_parts_l52_5280

theorem complex_sum_real_parts (a b : ℝ) (h : Complex.mk a b = Complex.I * (1 - Complex.I)) : a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_real_parts_l52_5280


namespace NUMINAMATH_CALUDE_solve_for_y_l52_5257

theorem solve_for_y (x y : ℤ) (h1 : x^2 - x + 6 = y + 2) (h2 : x = -8) : y = 76 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l52_5257


namespace NUMINAMATH_CALUDE_integer_fraction_condition_l52_5289

theorem integer_fraction_condition (n : ℤ) : 
  (∃ k : ℤ, 16 * (n^2 - n - 1)^2 = k * (2*n - 1)) ↔ 
  n = -12 ∨ n = -2 ∨ n = 0 ∨ n = 1 ∨ n = 3 ∨ n = 13 :=
sorry

end NUMINAMATH_CALUDE_integer_fraction_condition_l52_5289


namespace NUMINAMATH_CALUDE_minor_premise_is_proposition1_l52_5200

-- Define the propositions
def proposition1 : Prop := 0 < (1/2 : ℝ) ∧ (1/2 : ℝ) < 1
def proposition2 (a : ℝ) : Prop := ∀ x y : ℝ, x < y → a^y < a^x
def proposition3 : Prop := ∀ a : ℝ, 0 < a ∧ a < 1 → (∀ x y : ℝ, x < y → a^y < a^x)

-- Define the syllogism structure
structure Syllogism :=
  (major_premise : Prop)
  (minor_premise : Prop)
  (conclusion : Prop)

-- Theorem statement
theorem minor_premise_is_proposition1 :
  ∃ s : Syllogism, s.major_premise = proposition3 ∧
                   s.minor_premise = proposition1 ∧
                   s.conclusion = proposition2 (1/2) :=
sorry

end NUMINAMATH_CALUDE_minor_premise_is_proposition1_l52_5200


namespace NUMINAMATH_CALUDE_money_left_calculation_l52_5242

-- Define the initial amount, spent amount, and amount given to each friend
def initial_amount : ℚ := 5.10
def spent_on_sweets : ℚ := 1.05
def given_to_friend : ℚ := 1.00
def number_of_friends : ℕ := 2

-- Theorem to prove
theorem money_left_calculation :
  initial_amount - (spent_on_sweets + number_of_friends * given_to_friend) = 2.05 := by
  sorry


end NUMINAMATH_CALUDE_money_left_calculation_l52_5242


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l52_5213

theorem product_from_lcm_gcd (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 48) 
  (h_gcd : Nat.gcd a b = 8) : 
  a * b = 384 := by
sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l52_5213


namespace NUMINAMATH_CALUDE_smallest_x_value_l52_5262

theorem smallest_x_value (x y : ℕ+) (h : (9 : ℚ) / 10 = y / (275 + x)) : 
  x ≥ 5 ∧ ∃ (y' : ℕ+), (9 : ℚ) / 10 = y' / (275 + 5) :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l52_5262


namespace NUMINAMATH_CALUDE_system_solution_l52_5264

theorem system_solution :
  ∃ (a b c d : ℝ),
    (a * b + c + d = 3) ∧
    (b * c + d + a = 5) ∧
    (c * d + a + b = 2) ∧
    (d * a + b + c = 6) ∧
    (a = 2 ∧ b = 0 ∧ c = 0 ∧ d = 3) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l52_5264


namespace NUMINAMATH_CALUDE_product_sum_digits_base7_l52_5254

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Sums the digits of a number in base-7 --/
def sumDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem product_sum_digits_base7 :
  let a := 35
  let b := 52
  sumDigitsBase7 (toBase7 (toBase10 a * toBase10 b)) = 16 := by sorry

end NUMINAMATH_CALUDE_product_sum_digits_base7_l52_5254


namespace NUMINAMATH_CALUDE_matrix_equality_l52_5215

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = !![5, 2; -2, 3]) : 
  B * A = !![5, 2; -2, 3] := by sorry

end NUMINAMATH_CALUDE_matrix_equality_l52_5215


namespace NUMINAMATH_CALUDE_stock_profit_percentage_l52_5214

theorem stock_profit_percentage
  (total_stock : ℝ)
  (profit_percentage : ℝ)
  (loss_percentage : ℝ)
  (overall_loss : ℝ)
  (h1 : total_stock = 12500)
  (h2 : loss_percentage = 5)
  (h3 : overall_loss = 250)
  (h4 : (0.2 * total_stock * (1 + profit_percentage / 100) +
         0.8 * total_stock * (1 - loss_percentage / 100)) =
        total_stock - overall_loss) :
  profit_percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_stock_profit_percentage_l52_5214


namespace NUMINAMATH_CALUDE_rhombus_side_length_l52_5234

/-- The length of a side of a rhombus given one diagonal and its area -/
theorem rhombus_side_length 
  (d1 : ℝ) 
  (area : ℝ) 
  (h1 : d1 = 16) 
  (h2 : area = 327.90242451070714) : 
  ∃ (side : ℝ), abs (side - 37.73592452822641) < 1e-10 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l52_5234


namespace NUMINAMATH_CALUDE_craigs_apples_l52_5249

/-- 
Given:
- Craig's initial number of apples
- The number of apples Craig shares with Eugene
Prove that Craig's final number of apples is equal to the initial number minus the shared number.
-/
theorem craigs_apples (initial_apples shared_apples : ℕ) :
  initial_apples - shared_apples = initial_apples - shared_apples :=
by sorry

end NUMINAMATH_CALUDE_craigs_apples_l52_5249


namespace NUMINAMATH_CALUDE_mother_age_is_55_l52_5229

/-- The mother's age in years -/
def mother_age : ℕ := 55

/-- The daughter's age in years -/
def daughter_age : ℕ := mother_age - 27

theorem mother_age_is_55 :
  (mother_age = daughter_age + 27) ∧
  (mother_age - 1 = 2 * (daughter_age - 1)) →
  mother_age = 55 := by
  sorry

#check mother_age_is_55

end NUMINAMATH_CALUDE_mother_age_is_55_l52_5229


namespace NUMINAMATH_CALUDE_eleventh_inning_score_l52_5274

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : ℕ
  totalScore : ℕ

/-- Calculates the average score of a batsman -/
def average (stats : BatsmanStats) : ℚ :=
  stats.totalScore / stats.innings

/-- Theorem: Given the conditions, the score in the 11th inning is 110 runs -/
theorem eleventh_inning_score
  (stats10 : BatsmanStats)
  (stats11 : BatsmanStats)
  (h1 : stats10.innings = 10)
  (h2 : stats11.innings = 11)
  (h3 : average stats11 = 60)
  (h4 : average stats11 - average stats10 = 5) :
  stats11.totalScore - stats10.totalScore = 110 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_inning_score_l52_5274


namespace NUMINAMATH_CALUDE_gcd_105_88_l52_5267

theorem gcd_105_88 : Nat.gcd 105 88 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_88_l52_5267


namespace NUMINAMATH_CALUDE_range_of_m_l52_5224

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, m^2 * x^2 + 2*m*x - 4 < 2*x^2 + 4*x) → 
  -2 < m ∧ m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l52_5224


namespace NUMINAMATH_CALUDE_jacket_price_reduction_l52_5271

theorem jacket_price_reduction (x : ℝ) : 
  (1 - x) * (1 - 0.3) * (1 + 0.9047619047619048) = 1 → x = 0.25 := by
sorry

end NUMINAMATH_CALUDE_jacket_price_reduction_l52_5271


namespace NUMINAMATH_CALUDE_percentage_men_science_majors_l52_5237

/-- Given a college class, proves that 28% of men are science majors -/
theorem percentage_men_science_majors 
  (women_science_major_ratio : Real) 
  (non_science_ratio : Real) 
  (men_ratio : Real) 
  (h1 : women_science_major_ratio = 0.2)
  (h2 : non_science_ratio = 0.6)
  (h3 : men_ratio = 0.4) :
  (1 - non_science_ratio - women_science_major_ratio * (1 - men_ratio)) / men_ratio = 0.28 := by
  sorry

end NUMINAMATH_CALUDE_percentage_men_science_majors_l52_5237


namespace NUMINAMATH_CALUDE_tangent_condition_l52_5269

/-- The equation of the curve -/
def curve_eq (x y : ℝ) : Prop := y^2 - 4*x - 2*y + 1 = 0

/-- The equation of the line -/
def line_eq (k x y : ℝ) : Prop := y = k*x + 2

/-- The line is tangent to the curve -/
def is_tangent (k : ℝ) : Prop :=
  ∃! x y, curve_eq x y ∧ line_eq k x y

/-- The main theorem -/
theorem tangent_condition :
  ∀ k, is_tangent k ↔ (k = -2 + 2*Real.sqrt 2 ∨ k = -2 - 2*Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_condition_l52_5269


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_l52_5210

/-- An isosceles triangle with perimeter 13 and one side 3 has a base of 3 -/
theorem isosceles_triangle_base (a b c : ℝ) : 
  a + b + c = 13 →  -- perimeter is 13
  a = b →           -- isosceles condition
  (a = 3 ∨ b = 3 ∨ c = 3) →  -- one side is 3
  c = 3 :=          -- base is 3
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_l52_5210


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l52_5235

theorem sqrt_fraction_equality (x : ℝ) (h : x > 0) :
  Real.sqrt (x / (1 - (3 * x - 2) / (2 * x))) = Real.sqrt ((2 * x^2) / (2 - x)) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l52_5235


namespace NUMINAMATH_CALUDE_obtuse_triangles_in_100gon_l52_5272

/-- The number of vertices in the regular polygon -/
def n : ℕ := 100

/-- A function that determines if three vertices form an obtuse triangle in a regular n-gon -/
def is_obtuse (k l m : Fin n) : Prop :=
  (m - k : ℕ) % n > n / 4

/-- The number of ways to choose three vertices forming an obtuse triangle in a regular n-gon -/
def num_obtuse_triangles : ℕ := n * (n / 2 - 1).choose 2

/-- Theorem stating the number of obtuse triangles in a regular 100-gon -/
theorem obtuse_triangles_in_100gon :
  num_obtuse_triangles = 117600 :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangles_in_100gon_l52_5272


namespace NUMINAMATH_CALUDE_total_problems_l52_5266

/-- The number of problems Georgia completes in the first 20 minutes -/
def problems_first_20 : ℕ := 10

/-- The number of problems Georgia completes in the second 20 minutes -/
def problems_second_20 : ℕ := 2 * problems_first_20

/-- The number of problems Georgia has left to solve -/
def problems_left : ℕ := 45

/-- Theorem: The total number of problems on the test is 75 -/
theorem total_problems : 
  problems_first_20 + problems_second_20 + problems_left = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_problems_l52_5266
