import Mathlib

namespace NUMINAMATH_CALUDE_total_crayons_l381_38183

/-- The number of crayons each person has -/
structure CrayonCounts where
  wanda : ℕ
  dina : ℕ
  jacob : ℕ
  emma : ℕ
  xavier : ℕ
  hannah : ℕ

/-- The conditions of the problem -/
def crayon_problem (c : CrayonCounts) : Prop :=
  c.wanda = 62 ∧
  c.dina = 28 ∧
  c.jacob = c.dina - 2 ∧
  c.emma = 2 * c.wanda - 3 ∧
  c.xavier = ((c.jacob + c.dina) / 2) ^ 3 - 7 ∧
  c.hannah = (c.wanda + c.dina + c.jacob + c.emma + c.xavier) / 5

/-- The theorem to be proved -/
theorem total_crayons (c : CrayonCounts) : 
  crayon_problem c → c.wanda + c.dina + c.jacob + c.emma + c.xavier + c.hannah = 23895 := by
  sorry


end NUMINAMATH_CALUDE_total_crayons_l381_38183


namespace NUMINAMATH_CALUDE_square_difference_l381_38116

theorem square_difference : (39 : ℤ)^2 = 40^2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l381_38116


namespace NUMINAMATH_CALUDE_luke_bought_twelve_stickers_l381_38131

/-- The number of stickers Luke bought from the store -/
def stickers_bought (initial : ℕ) (birthday : ℕ) (given_away : ℕ) (used : ℕ) (remaining : ℕ) : ℕ :=
  remaining + given_away + used - initial - birthday

/-- Theorem stating that Luke bought 12 stickers from the store -/
theorem luke_bought_twelve_stickers :
  stickers_bought 20 20 5 8 39 = 12 := by
  sorry

end NUMINAMATH_CALUDE_luke_bought_twelve_stickers_l381_38131


namespace NUMINAMATH_CALUDE_sum_of_squares_l381_38142

theorem sum_of_squares (x y z : ℤ) 
  (sum_eq : x + y + z = 3)
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) :
  x^2 + y^2 + z^2 = 57 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l381_38142


namespace NUMINAMATH_CALUDE_parabola_and_circle_problem_l381_38167

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point K
def K : ℝ × ℝ := (-1, 0)

-- Define the line l passing through K
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m*y - 1

-- Define the condition for points A and B on the parabola and line l
def point_on_parabola_and_line (x y m : ℝ) : Prop :=
  parabola x y ∧ line_l m x y

-- Define the symmetry condition for points A and D
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = x₂ ∧ y₁ = -y₂

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - 1) * (x₂ - 1) + y₁ * y₂ = 8/9

-- Main theorem
theorem parabola_and_circle_problem
  (x₁ y₁ x₂ y₂ xd yd m : ℝ)
  (h₁ : point_on_parabola_and_line x₁ y₁ m)
  (h₂ : point_on_parabola_and_line x₂ y₂ m)
  (h₃ : symmetric_points x₁ y₁ xd yd)
  (h₄ : dot_product_condition x₁ y₁ x₂ y₂) :
  (∃ (k : ℝ), focus.1 = k * (x₂ - xd) + xd ∧ focus.2 = k * (y₂ + yd)) ∧
  (∃ (c : ℝ × ℝ) (r : ℝ), c = (1/9, 0) ∧ r = 2/3 ∧
    ∀ (x y : ℝ), (x - c.1)^2 + (y - c.2)^2 = r^2 ↔
      (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
        x = t * x₂ + (1-t) * K.1 ∧
        y = t * y₂ + (1-t) * K.2) ∨
      (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
        x = t * xd + (1-t) * K.1 ∧
        y = t * yd + (1-t) * K.2) ∨
      (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
        x = t * x₂ + (1-t) * xd ∧
        y = t * y₂ + (1-t) * yd)) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_circle_problem_l381_38167


namespace NUMINAMATH_CALUDE_proposition_implications_l381_38171

theorem proposition_implications (p q : Prop) 
  (h : ¬(¬p ∨ ¬q)) : (p ∧ q) ∧ (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_implications_l381_38171


namespace NUMINAMATH_CALUDE_max_value_of_a_l381_38163

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- State the theorem
theorem max_value_of_a :
  ∀ a : ℝ, (A a ∪ B a = Set.univ) → (∀ b : ℝ, (A b ∪ B b = Set.univ) → b ≤ a) → a = 2 := by
  sorry

-- Note: Set.univ represents the entire real number line (ℝ)

end NUMINAMATH_CALUDE_max_value_of_a_l381_38163


namespace NUMINAMATH_CALUDE_highest_elevation_l381_38128

/-- The elevation function of a particle projected vertically -/
def s (t : ℝ) : ℝ := 100 * t - 5 * t^2

/-- The initial velocity of the particle in meters per second -/
def initial_velocity : ℝ := 100

theorem highest_elevation :
  ∃ (t_max : ℝ), ∀ (t : ℝ), s t ≤ s t_max ∧ s t_max = 500 := by
  sorry

end NUMINAMATH_CALUDE_highest_elevation_l381_38128


namespace NUMINAMATH_CALUDE_sum_consecutive_odd_integers_to_25_l381_38169

/-- Sum of consecutive odd integers from 1 to n -/
def sumConsecutiveOddIntegers (n : ℕ) : ℕ :=
  let k := (n + 1) / 2
  k * k

/-- Theorem: The sum of consecutive odd integers from 1 to 25 is 169 -/
theorem sum_consecutive_odd_integers_to_25 :
  sumConsecutiveOddIntegers 25 = 169 := by
  sorry

#eval sumConsecutiveOddIntegers 25

end NUMINAMATH_CALUDE_sum_consecutive_odd_integers_to_25_l381_38169


namespace NUMINAMATH_CALUDE_p_min_value_l381_38134

/-- The quadratic function p(x) = x^2 + 6x + 5 -/
def p (x : ℝ) : ℝ := x^2 + 6*x + 5

/-- The minimum value of p(x) is -4 -/
theorem p_min_value : ∀ x : ℝ, p x ≥ -4 := by sorry

end NUMINAMATH_CALUDE_p_min_value_l381_38134


namespace NUMINAMATH_CALUDE_left_handed_fraction_conference_l381_38172

/-- Represents the fraction of left-handed participants for each country type -/
structure LeftHandedFractions where
  red : ℚ
  blue : ℚ
  green : ℚ
  yellow : ℚ

/-- Represents the ratio of participants from each country type -/
structure ParticipantRatio where
  red : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the fraction of left-handed participants given the ratio of participants
    and the fractions of left-handed participants for each country type -/
def leftHandedFraction (ratio : ParticipantRatio) (fractions : LeftHandedFractions) : ℚ :=
  (ratio.red * fractions.red + ratio.blue * fractions.blue +
   ratio.green * fractions.green + ratio.yellow * fractions.yellow) /
  (ratio.red + ratio.blue + ratio.green + ratio.yellow)

theorem left_handed_fraction_conference :
  let ratio : ParticipantRatio := ⟨10, 5, 3, 2⟩
  let fractions : LeftHandedFractions := ⟨37/100, 61/100, 26/100, 48/100⟩
  leftHandedFraction ratio fractions = 849/2000 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_fraction_conference_l381_38172


namespace NUMINAMATH_CALUDE_dice_sum_product_l381_38173

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 → 
  1 ≤ b ∧ b ≤ 6 → 
  1 ≤ c ∧ c ≤ 6 → 
  1 ≤ d ∧ d ≤ 6 → 
  a * b * c * d = 216 → 
  a + b + c + d ≠ 19 := by
sorry

end NUMINAMATH_CALUDE_dice_sum_product_l381_38173


namespace NUMINAMATH_CALUDE_sum_after_2023_operations_l381_38102

def starting_sequence : List Int := [7, 3, 5]

def operation (seq : List Int) : List Int :=
  seq ++ (List.zip seq (List.tail seq)).map (fun (a, b) => a - b)

def sum_after_n_operations (n : Nat) : Int :=
  n * 2 + (starting_sequence.sum)

theorem sum_after_2023_operations :
  sum_after_n_operations 2023 = 4061 := by sorry

end NUMINAMATH_CALUDE_sum_after_2023_operations_l381_38102


namespace NUMINAMATH_CALUDE_max_x5_value_l381_38113

theorem max_x5_value (x₁ x₂ x₃ x₄ x₅ : ℕ+) 
  (h : x₁ + x₂ + x₃ + x₄ + x₅ = x₁ * x₂ * x₃ * x₄ * x₅) : 
  x₅ ≤ 5 ∧ ∃ (a b c d : ℕ+), a + b + c + d + 5 = a * b * c * d * 5 := by
  sorry

end NUMINAMATH_CALUDE_max_x5_value_l381_38113


namespace NUMINAMATH_CALUDE_remaining_amount_after_buying_folders_l381_38125

def initial_amount : ℕ := 19
def folder_cost : ℕ := 2

theorem remaining_amount_after_buying_folders :
  initial_amount - (initial_amount / folder_cost * folder_cost) = 1 := by
sorry

end NUMINAMATH_CALUDE_remaining_amount_after_buying_folders_l381_38125


namespace NUMINAMATH_CALUDE_trig_equation_solution_l381_38186

theorem trig_equation_solution (t : ℝ) : 
  (2 * Real.cos (2 * t) + 5) * Real.cos t ^ 4 - (2 * Real.cos (2 * t) + 5) * Real.sin t ^ 4 = 3 ↔ 
  ∃ k : ℤ, t = π / 6 * (6 * ↑k + 1) ∨ t = π / 6 * (6 * ↑k - 1) :=
by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l381_38186


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l381_38109

theorem fraction_equation_solution (x y : ℝ) 
  (hx_nonzero : x ≠ 0) 
  (hx_not_one : x ≠ 1) 
  (hy_nonzero : y ≠ 0) 
  (hy_not_three : y ≠ 3) 
  (h_equation : (3 / x) + (2 / y) = 1 / 3) : 
  x = 9 * y / (y - 6) := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l381_38109


namespace NUMINAMATH_CALUDE_polynomial_expansion_l381_38119

/-- Proves the equality of the expanded polynomial expression -/
theorem polynomial_expansion (y : ℝ) : 
  (3 * y + 2) * (5 * y^12 - y^11 + 3 * y^10 + 2) = 
  15 * y^13 + 7 * y^12 + 7 * y^11 + 6 * y^10 + 6 * y + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l381_38119


namespace NUMINAMATH_CALUDE_seven_lines_intersection_impossibility_l381_38158

/-- The maximum number of intersections for n lines in a Euclidean plane -/
def max_intersections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of intersections required for a given number of triple and double intersections -/
def required_intersections (triple_points double_points : ℕ) : ℕ :=
  triple_points * 3 + double_points

theorem seven_lines_intersection_impossibility :
  let n_lines : ℕ := 7
  let min_triple_points : ℕ := 6
  let min_double_points : ℕ := 4
  required_intersections min_triple_points min_double_points > max_intersections n_lines := by
  sorry


end NUMINAMATH_CALUDE_seven_lines_intersection_impossibility_l381_38158


namespace NUMINAMATH_CALUDE_computer_sticker_price_l381_38191

theorem computer_sticker_price : 
  ∀ (x : ℝ), 
  (0.80 * x - 80 = 0.70 * x - 40 - 30) → 
  x = 700 := by
sorry

end NUMINAMATH_CALUDE_computer_sticker_price_l381_38191


namespace NUMINAMATH_CALUDE_max_cables_cut_theorem_l381_38174

/-- Represents a computer network with computers and cables -/
structure ComputerNetwork where
  num_computers : Nat
  num_cables : Nat
  num_clusters : Nat

/-- The initial state of the computer network -/
def initial_network : ComputerNetwork :=
  { num_computers := 200
  , num_cables := 345
  , num_clusters := 1 }

/-- The final state of the computer network after cable cutting -/
def final_network : ComputerNetwork :=
  { num_computers := 200
  , num_cables := initial_network.num_cables - 153
  , num_clusters := 8 }

/-- The maximum number of cables that can be cut -/
def max_cables_cut : Nat := 153

/-- Theorem stating the maximum number of cables that can be cut -/
theorem max_cables_cut_theorem :
  max_cables_cut = initial_network.num_cables - final_network.num_cables ∧
  final_network.num_clusters = 8 ∧
  final_network.num_cables ≥ final_network.num_computers - final_network.num_clusters :=
by sorry


end NUMINAMATH_CALUDE_max_cables_cut_theorem_l381_38174


namespace NUMINAMATH_CALUDE_f_comp_three_roots_l381_38180

/-- A quadratic function with a parameter c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 6*x + c

/-- The composition of f with itself -/
def f_comp (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- The theorem stating the condition for f(f(x)) to have exactly 3 distinct real roots -/
theorem f_comp_three_roots (c : ℝ) :
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f_comp c x = 0 ∧ f_comp c y = 0 ∧ f_comp c z = 0 ∧
    (∀ w : ℝ, f_comp c w = 0 → w = x ∨ w = y ∨ w = z)) ↔
  c = (11 - Real.sqrt 13) / 2 :=
sorry

end NUMINAMATH_CALUDE_f_comp_three_roots_l381_38180


namespace NUMINAMATH_CALUDE_triangle_area_l381_38153

theorem triangle_area (a b c : ℝ) (h1 : a = 17) (h2 : b = 144) (h3 : c = 145) :
  (1/2) * a * b = 1224 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l381_38153


namespace NUMINAMATH_CALUDE_parallel_vectors_difference_magnitude_l381_38111

theorem parallel_vectors_difference_magnitude :
  ∀ x : ℝ,
  let a : Fin 2 → ℝ := ![1, x]
  let b : Fin 2 → ℝ := ![2*x + 3, -x]
  (∃ (k : ℝ), a = k • b) →
  ‖a - b‖ = 2 ∨ ‖a - b‖ = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_difference_magnitude_l381_38111


namespace NUMINAMATH_CALUDE_exist_three_fractions_product_one_l381_38152

/-- The sequence of fractions from 1/2017 to 2017/1 -/
def fraction_sequence : Fin 2017 → Rat := λ i => (i + 1) / (2018 - (i + 1))

/-- Theorem: There exist three fractions in the sequence whose product is 1 -/
theorem exist_three_fractions_product_one :
  ∃ (i j k : Fin 2017), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    fraction_sequence i * fraction_sequence j * fraction_sequence k = 1 := by
  sorry

end NUMINAMATH_CALUDE_exist_three_fractions_product_one_l381_38152


namespace NUMINAMATH_CALUDE_hyperbola_condition_l381_38145

/-- Defines the equation of a conic section -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m - 1) + y^2 / (m - 2) = 1 ∧ (m - 1) * (m - 2) < 0

/-- Theorem stating the necessary and sufficient condition for the equation to represent a hyperbola -/
theorem hyperbola_condition (m : ℝ) :
  is_hyperbola m ↔ 1 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l381_38145


namespace NUMINAMATH_CALUDE_set_operations_l381_38155

def U : Set ℕ := {1,2,3,4,5,6,7,8}

def A : Set ℕ := {x | x^2 - 3*x + 2 = 0}

def B : Set ℕ := {x ∈ U | 1 ≤ x ∧ x ≤ 5}

def C : Set ℕ := {x ∈ U | 2 < x ∧ x < 9}

theorem set_operations :
  (A ∪ (B ∩ C) = {1,2,3,4,5}) ∧
  ((U \ B) ∪ (U \ C) = {1,2,6,7,8}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l381_38155


namespace NUMINAMATH_CALUDE_modulus_of_Z_l381_38168

-- Define the operation
def matrix_op (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem modulus_of_Z : ∃ (Z : ℂ), 
  (matrix_op Z i 1 i = 1 + i) ∧ (Complex.abs Z = Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_Z_l381_38168


namespace NUMINAMATH_CALUDE_maria_score_is_15_l381_38106

/-- Represents a quiz result -/
structure QuizResult where
  total_questions : Nat
  correct_answers : Nat
  incorrect_answers : Nat
  unanswered_questions : Nat
  deriving Repr

/-- Calculates the score for a quiz result -/
def calculate_score (result : QuizResult) : Nat :=
  result.correct_answers

/-- Maria's quiz result -/
def maria_result : QuizResult :=
  { total_questions := 20
  , correct_answers := 15
  , incorrect_answers := 3
  , unanswered_questions := 2
  }

theorem maria_score_is_15 :
  calculate_score maria_result = 15 ∧
  maria_result.total_questions = maria_result.correct_answers + maria_result.incorrect_answers + maria_result.unanswered_questions :=
by sorry

end NUMINAMATH_CALUDE_maria_score_is_15_l381_38106


namespace NUMINAMATH_CALUDE_class_average_problem_l381_38146

theorem class_average_problem (total_students : Nat) (high_scorers : Nat) (zero_scorers : Nat)
  (high_score : Nat) (class_average : Rat) :
  total_students = 27 →
  high_scorers = 5 →
  zero_scorers = 3 →
  high_score = 95 →
  class_average = 49.25925925925926 →
  let remaining_students := total_students - high_scorers - zero_scorers
  let total_score := class_average * total_students
  let high_scorers_total := high_scorers * high_score
  (total_score - high_scorers_total) / remaining_students = 45 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l381_38146


namespace NUMINAMATH_CALUDE_final_fish_count_l381_38122

def fish_count (day : ℕ) : ℕ :=
  match day with
  | 0 => 10  -- Initial number of fish
  | 1 => 30  -- Day 1: 10 * 3
  | 2 => 90  -- Day 2: 30 * 3
  | 3 => 270 -- Day 3: 90 * 3
  | 4 => 162 -- Day 4: (270 * 3) - (270 * 3 * 2 / 5)
  | 5 => 486 -- Day 5: 162 * 3
  | 6 => 834 -- Day 6: (486 * 3) - (486 * 3 * 3 / 7)
  | 7 => 2502 -- Day 7: 834 * 3
  | 8 => 7531 -- Day 8: (2502 * 3) + 25
  | 9 => 22593 -- Day 9: 7531 * 3
  | 10 => 33890 -- Day 10: (22593 * 3) - (22593 * 3 / 2)
  | 11 => 101670 -- Day 11: 33890 * 3
  | _ => 305010 -- Day 12: 101670 * 3

theorem final_fish_count :
  fish_count 12 + (3 * fish_count 12 + 5) = 1220045 := by
  sorry

#eval fish_count 12 + (3 * fish_count 12 + 5)

end NUMINAMATH_CALUDE_final_fish_count_l381_38122


namespace NUMINAMATH_CALUDE_height_difference_l381_38190

/-- Given heights of Jana, Jess, and Kelly, prove the height difference between Jess and Kelly. -/
theorem height_difference (jana_height jess_height : ℕ) : 
  jana_height = 74 →
  jess_height = 72 →
  ∃ kelly_height : ℕ, 
    jana_height = kelly_height + 5 ∧ 
    jess_height - kelly_height = 3 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l381_38190


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_l381_38149

theorem complex_sum_of_powers (x y : ℂ) (hxy : x ≠ 0 ∧ y ≠ 0) (h : x^2 + x*y + y^2 = 0) :
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_l381_38149


namespace NUMINAMATH_CALUDE_solve_lawn_mowing_problem_l381_38175

/-- Kaleb's lawn mowing business finances --/
def lawn_mowing_problem (spring_earnings summer_earnings final_amount : ℕ) : Prop :=
  let total_earnings := spring_earnings + summer_earnings
  let supplies_cost := total_earnings - final_amount
  supplies_cost = total_earnings - final_amount

theorem solve_lawn_mowing_problem :
  lawn_mowing_problem 4 50 50 = true :=
sorry

end NUMINAMATH_CALUDE_solve_lawn_mowing_problem_l381_38175


namespace NUMINAMATH_CALUDE_trapezoid_bc_length_l381_38135

/-- Represents a trapezoid ABCD with given properties -/
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  ab : ℝ
  cd : ℝ

/-- Theorem stating the length of BC in the trapezoid -/
theorem trapezoid_bc_length (t : Trapezoid) 
  (h_area : t.area = 180)
  (h_altitude : t.altitude = 8)
  (h_ab : t.ab = 14)
  (h_cd : t.cd = 20) :
  ∃ (bc : ℝ), bc = 22.5 - Real.sqrt 33 - 2 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_bc_length_l381_38135


namespace NUMINAMATH_CALUDE_intersection_A_B_l381_38192

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 < 0}

-- Define set B
def B : Set ℝ := {x | x > 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l381_38192


namespace NUMINAMATH_CALUDE_rope_folding_l381_38144

theorem rope_folding (n : ℕ) (original_length : ℝ) (h : n = 3) :
  let num_parts := 2^n
  let part_length := original_length / num_parts
  part_length = (1 / 8) * original_length := by
  sorry

end NUMINAMATH_CALUDE_rope_folding_l381_38144


namespace NUMINAMATH_CALUDE_factorization_equality_l381_38130

theorem factorization_equality (x y : ℝ) : 
  x^2 * (y^2 - 1) + 2 * x * (y^2 - 1) + (y^2 - 1) = (y + 1) * (y - 1) * (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l381_38130


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l381_38188

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) : 
  let z := Complex.abs (3 + 4 * i) / (1 - 2 * i)
  Complex.im z = 2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l381_38188


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l381_38187

theorem geometric_sequence_ratio_sum (k a₂ a₃ b₂ b₃ p r : ℝ) 
  (h1 : p ≠ 1)
  (h2 : r ≠ 1)
  (h3 : p ≠ r)
  (h4 : a₂ = k * p)
  (h5 : a₃ = k * p^2)
  (h6 : b₂ = k * r)
  (h7 : b₃ = k * r^2)
  (h8 : 3 * a₃ - 4 * b₃ = 5 * (3 * a₂ - 4 * b₂)) :
  p + r = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l381_38187


namespace NUMINAMATH_CALUDE_sequence_increasing_iff_a_in_range_l381_38147

def sequence_a (a : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 7 then (3 - a) * n - 3 else a^(n - 6)

theorem sequence_increasing_iff_a_in_range (a : ℝ) :
  (∀ n : ℕ, sequence_a a n ≤ sequence_a a (n + 1)) ↔ (9/4 < a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_sequence_increasing_iff_a_in_range_l381_38147


namespace NUMINAMATH_CALUDE_range_of_f_l381_38178

def f (x : ℝ) : ℝ := |x + 8| - |x - 3|

theorem range_of_f :
  Set.range f = Set.Icc (-11) 11 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l381_38178


namespace NUMINAMATH_CALUDE_line_k_value_l381_38176

/-- A line passes through the points (0, 3), (7, k), and (21, 2) -/
def line_passes_through (k : ℚ) : Prop :=
  ∃ m b : ℚ, 
    (3 = m * 0 + b) ∧ 
    (k = m * 7 + b) ∧ 
    (2 = m * 21 + b)

/-- Theorem: If a line passes through (0, 3), (7, k), and (21, 2), then k = 8/3 -/
theorem line_k_value : 
  ∀ k : ℚ, line_passes_through k → k = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_line_k_value_l381_38176


namespace NUMINAMATH_CALUDE_square_minus_one_divisible_by_three_l381_38124

theorem square_minus_one_divisible_by_three (x : ℤ) (h : ¬ 3 ∣ x) : 3 ∣ (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_square_minus_one_divisible_by_three_l381_38124


namespace NUMINAMATH_CALUDE_school_play_seating_l381_38185

theorem school_play_seating (rows : ℕ) (chairs_per_row : ℕ) (unoccupied : ℕ) : 
  rows = 40 → chairs_per_row = 20 → unoccupied = 10 → 
  rows * chairs_per_row - unoccupied = 790 := by
  sorry

end NUMINAMATH_CALUDE_school_play_seating_l381_38185


namespace NUMINAMATH_CALUDE_eighth_term_value_l381_38104

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the theorem
theorem eighth_term_value 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_sum : a 3 + 3 * a 8 + a 13 = 120) : 
  a 8 = 24 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_value_l381_38104


namespace NUMINAMATH_CALUDE_compute_expression_l381_38121

theorem compute_expression : 
  20 * (240 / 3 + 40 / 5 + 16 / 25 + 2) = 1772.8 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l381_38121


namespace NUMINAMATH_CALUDE_cut_square_corners_l381_38195

/-- Given a square with side length 24 units, if each corner is cut to form an isoscelos right
    triangle resulting in a smaller rectangle, then the total area of the four removed triangles
    is 288 square units. -/
theorem cut_square_corners (r s : ℝ) : 
  (r + s)^2 + (r - s)^2 = 24^2 → r^2 + s^2 = 288 := by sorry

end NUMINAMATH_CALUDE_cut_square_corners_l381_38195


namespace NUMINAMATH_CALUDE_rhombus_area_fraction_l381_38193

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rhombus defined by four vertices -/
structure Rhombus where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- The grid size -/
def gridSize : ℕ := 6

/-- The rhombus in question -/
def specialRhombus : Rhombus := {
  v1 := ⟨2, 2⟩,
  v2 := ⟨4, 2⟩,
  v3 := ⟨3, 3⟩,
  v4 := ⟨3, 1⟩
}

/-- Calculate the area of a rhombus -/
def rhombusArea (r : Rhombus) : ℝ := sorry

/-- Calculate the area of the grid -/
def gridArea : ℝ := gridSize ^ 2

/-- The main theorem to prove -/
theorem rhombus_area_fraction :
  rhombusArea specialRhombus / gridArea = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_fraction_l381_38193


namespace NUMINAMATH_CALUDE_triangle_problem_l381_38115

theorem triangle_problem (a b c A B C : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B →
  b = 2 →
  (1/2) * a * c * Real.sin B = Real.sqrt 3 →
  B = π/3 ∧ a = 2 ∧ c = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l381_38115


namespace NUMINAMATH_CALUDE_melinda_original_cost_l381_38117

/-- Represents the original cost of clothing items before tax and discounts -/
def original_cost (jeans_price shirt_price jacket_price : ℝ) : ℝ :=
  jeans_price + shirt_price + jacket_price

/-- The theorem stating the original cost of Melinda's purchase -/
theorem melinda_original_cost :
  original_cost 14.50 9.50 21.00 = 45.00 := by
  sorry

end NUMINAMATH_CALUDE_melinda_original_cost_l381_38117


namespace NUMINAMATH_CALUDE_numbers_with_2019_divisors_l381_38181

theorem numbers_with_2019_divisors (n : ℕ) : 
  n < 128^97 → (Finset.card (Nat.divisors n) = 2019) → 
  (n = 2^672 * 3^2 ∨ n = 2^672 * 5^2 ∨ n = 2^672 * 7^2 ∨ n = 2^672 * 11^2) :=
by sorry

end NUMINAMATH_CALUDE_numbers_with_2019_divisors_l381_38181


namespace NUMINAMATH_CALUDE_arrangement_theorem_l381_38179

/-- The number of permutations of n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to arrange 7 students with Student A not at either end. -/
def arrangement_count_1 : ℕ := 5 * permutations 6

/-- The number of ways to arrange 7 students with Student A not on the left end
    and Student B not on the right end. -/
def arrangement_count_2 : ℕ := permutations 6 + choose 5 1 * choose 5 1 * permutations 5

theorem arrangement_theorem :
  (arrangement_count_1 = 5 * permutations 6) ∧
  (arrangement_count_2 = permutations 6 + choose 5 1 * choose 5 1 * permutations 5) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l381_38179


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l381_38107

/-- Given k = 2012² + 2^2014, prove that (k² + 2^k) mod 10 = 5 -/
theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ) : k = 2012^2 + 2^2014 → (k^2 + 2^k) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l381_38107


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l381_38103

theorem system_of_equations_solution :
  ∃! (x y : ℝ), 4*x + 3*y = 6.4 ∧ 5*x - 6*y = -1.5 ∧ x = 11.3/13 ∧ y = 2.9232/3 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l381_38103


namespace NUMINAMATH_CALUDE_olympic_audience_conversion_l381_38148

def opening_ceremony_audience : ℕ := 316000000
def closing_ceremony_audience : ℕ := 236000000

def million_to_full_number (x : ℕ) : ℕ := x * 1000000
def million_to_billion (x : ℕ) : ℚ := x / 1000

/-- Rounds a rational number to one decimal place -/
def round_to_one_decimal (x : ℚ) : ℚ :=
  (x * 10).floor / 10

theorem olympic_audience_conversion :
  (million_to_full_number 316 = opening_ceremony_audience) ∧
  (round_to_one_decimal (million_to_billion closing_ceremony_audience) = 2.4) :=
sorry

end NUMINAMATH_CALUDE_olympic_audience_conversion_l381_38148


namespace NUMINAMATH_CALUDE_wellness_gym_ratio_l381_38170

theorem wellness_gym_ratio (f m : ℕ) (hf : f > 0) (hm : m > 0) :
  (35 : ℝ) * f + 30 * m = 32 * (f + m) →
  (f : ℝ) / m = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_wellness_gym_ratio_l381_38170


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l381_38141

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y+2)^2) + Real.sqrt ((x-6)^2 + (y-4)^2) = 14

-- Define what it means for a point to be on the conic
def point_on_conic (x y : ℝ) : Prop :=
  conic_equation x y

-- Define the foci of the conic
def focus1 : ℝ × ℝ := (0, -2)
def focus2 : ℝ × ℝ := (6, 4)

-- Theorem stating that the conic is an ellipse
theorem conic_is_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧
  ∀ (x y : ℝ), point_on_conic x y ↔
    (x - (focus1.1 + focus2.1) / 2)^2 / a^2 +
    (y - (focus1.2 + focus2.2) / 2)^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l381_38141


namespace NUMINAMATH_CALUDE_ab_leq_one_l381_38151

theorem ab_leq_one (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a + b = 2) : a * b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_leq_one_l381_38151


namespace NUMINAMATH_CALUDE_sum_of_fractions_l381_38136

theorem sum_of_fractions : (48 : ℚ) / 72 + (30 : ℚ) / 45 = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l381_38136


namespace NUMINAMATH_CALUDE_museum_admission_difference_l381_38184

theorem museum_admission_difference (men women free_admission : ℕ) 
  (h1 : men = 194)
  (h2 : women = 235)
  (h3 : free_admission = 68) :
  (men + women) - free_admission - free_admission = 293 := by
  sorry

end NUMINAMATH_CALUDE_museum_admission_difference_l381_38184


namespace NUMINAMATH_CALUDE_number_difference_l381_38161

theorem number_difference (a b : ℕ) : 
  a + b = 30000 →
  b = 10 * a + 5 →
  b - a = 24548 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l381_38161


namespace NUMINAMATH_CALUDE_min_difference_l381_38197

noncomputable def f (x : ℝ) : ℝ := Real.exp (4 * x - 1)

noncomputable def g (x : ℝ) : ℝ := 1/2 + Real.log (2 * x)

theorem min_difference (m n : ℝ) (h : f m = g n) :
  ∃ (m₀ n₀ : ℝ), f m₀ = g n₀ ∧ ∀ m' n', f m' = g n' → n₀ - m₀ ≤ n' - m' ∧ n₀ - m₀ = (1 + Real.log 2) / 4 :=
sorry

end NUMINAMATH_CALUDE_min_difference_l381_38197


namespace NUMINAMATH_CALUDE_root_equation_value_l381_38105

theorem root_equation_value (m : ℝ) : m^2 - m - 1 = 0 → m^2 - m + 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l381_38105


namespace NUMINAMATH_CALUDE_soda_box_cans_l381_38137

/-- The number of people attending the reunion -/
def attendees : ℕ := 5 * 12

/-- The number of cans each person consumes -/
def cans_per_person : ℕ := 2

/-- The cost of each box of soda in dollars -/
def cost_per_box : ℕ := 2

/-- The number of family members -/
def family_members : ℕ := 6

/-- The amount each family member pays in dollars -/
def payment_per_member : ℕ := 4

/-- The number of cans in each box -/
def cans_per_box : ℕ := 10

theorem soda_box_cans : 
  cans_per_box = (attendees * cans_per_person) / 
    ((family_members * payment_per_member) / cost_per_box) :=
by sorry

end NUMINAMATH_CALUDE_soda_box_cans_l381_38137


namespace NUMINAMATH_CALUDE_two_numbers_difference_l381_38164

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) :
  |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l381_38164


namespace NUMINAMATH_CALUDE_proposition_analysis_l381_38129

theorem proposition_analysis (m n : ℝ) : 
  (¬ (((m ≤ 0) ∨ (n ≤ 0)) → (m + n ≤ 0))) ∧ 
  ((m + n ≤ 0) → ((m ≤ 0) ∨ (n ≤ 0))) ∧
  (((m > 0) ∧ (n > 0)) → (m + n > 0)) ∧
  (¬ ((m + n > 0) → ((m > 0) ∧ (n > 0)))) ∧
  (((m + n ≤ 0) → ((m ≤ 0) ∨ (n ≤ 0))) ∧ ¬(((m ≤ 0) ∨ (n ≤ 0)) → (m + n ≤ 0))) :=
by sorry

end NUMINAMATH_CALUDE_proposition_analysis_l381_38129


namespace NUMINAMATH_CALUDE_abc_inequality_l381_38165

theorem abc_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l381_38165


namespace NUMINAMATH_CALUDE_angle_value_l381_38198

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem angle_value (a : ℕ → ℝ) (α : ℝ) :
  geometric_sequence a →
  (a 1 * a 1 - 2 * a 1 * Real.sin α - Real.sqrt 3 * Real.sin α = 0) →
  (a 8 * a 8 - 2 * a 8 * Real.sin α - Real.sqrt 3 * Real.sin α = 0) →
  ((a 1 + a 8) ^ 2 = 2 * a 3 * a 6 + 6) →
  (0 < α ∧ α < Real.pi / 2) →
  α = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_l381_38198


namespace NUMINAMATH_CALUDE_incorrect_to_correct_calculation_l381_38100

theorem incorrect_to_correct_calculation (x : ℝ) : x * 3 - 5 = 103 → (x / 3) - 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_to_correct_calculation_l381_38100


namespace NUMINAMATH_CALUDE_article_count_proof_l381_38199

/-- The number of articles we are considering -/
def X : ℕ := 50

/-- The number of articles sold at selling price -/
def sold_articles : ℕ := 35

/-- The gain percentage -/
def gain_percentage : ℚ := 42857142857142854 / 100000000000000000

theorem article_count_proof :
  (∃ (C S : ℚ), C > 0 ∧ S > 0 ∧
    X * C = sold_articles * S ∧
    (S - C) / C = gain_percentage) →
  X = 50 :=
by sorry

end NUMINAMATH_CALUDE_article_count_proof_l381_38199


namespace NUMINAMATH_CALUDE_shaded_square_area_l381_38150

/-- A configuration of four unit squares arranged in a 2x2 grid, each containing an inscribed equilateral triangle sharing an edge with the square. -/
structure SquareTriangleConfig where
  /-- The side length of each unit square -/
  unit_square_side : ℝ
  /-- The side length of each equilateral triangle -/
  triangle_side : ℝ
  /-- The side length of the larger square formed by the four unit squares -/
  large_square_side : ℝ
  /-- The side length of the shaded square formed by connecting triangle vertices -/
  shaded_square_side : ℝ
  /-- Condition: Each unit square has side length 1 -/
  unit_square_cond : unit_square_side = 1
  /-- Condition: The triangle side is equal to the unit square side -/
  triangle_side_cond : triangle_side = unit_square_side
  /-- Condition: The larger square has side length 2 -/
  large_square_cond : large_square_side = 2 * unit_square_side
  /-- Condition: The diagonal of the shaded square equals the side of the larger square -/
  shaded_square_diag_cond : shaded_square_side * Real.sqrt 2 = large_square_side

/-- The theorem stating that the area of the shaded square is 2 square units -/
theorem shaded_square_area (config : SquareTriangleConfig) : 
  config.shaded_square_side ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_square_area_l381_38150


namespace NUMINAMATH_CALUDE_vasya_no_purchase_days_l381_38143

theorem vasya_no_purchase_days :
  ∀ (x y z w : ℕ),
  x + y + z + w = 15 →
  9 * x + 4 * z = 30 →
  2 * y + z = 9 →
  w = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_vasya_no_purchase_days_l381_38143


namespace NUMINAMATH_CALUDE_subtracted_value_proof_l381_38112

theorem subtracted_value_proof (N : ℕ) (h : N = 2976) : ∃ V : ℚ, (N / 12 : ℚ) - V = 8 ∧ V = 240 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_proof_l381_38112


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l381_38194

theorem sqrt_product_equality : Real.sqrt 54 * Real.sqrt 32 * Real.sqrt 6 = 72 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l381_38194


namespace NUMINAMATH_CALUDE_elliptical_lines_l381_38132

-- Define the points M and N
def M : ℝ × ℝ := (-1, 0)
def N : ℝ × ℝ := (-1, 0)

-- Define the ellipse equation
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the condition for a point to be on a line
def is_on_line (x y : ℝ) (a b c : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Define what it means for a line to be an "elliptical line"
def is_elliptical_line (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, is_on_line x y a b c ∧ is_on_ellipse x y

theorem elliptical_lines :
  is_elliptical_line 1 (-1) 0 ∧ 
  is_elliptical_line 2 (-1) 1 ∧ 
  ¬is_elliptical_line 1 (-2) 6 ∧ 
  ¬is_elliptical_line 1 1 (-3) :=
sorry

end NUMINAMATH_CALUDE_elliptical_lines_l381_38132


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_sum_l381_38159

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_minimum_sum 
  (a : ℕ → ℝ) 
  (h_geom : GeometricSequence a) 
  (h_positive : ∀ n, a n > 0) 
  (h_product : a 3 * a 5 = 12) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x * y = 12 → x + y ≥ 4 * Real.sqrt 3) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = 12 ∧ x + y = 4 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_sum_l381_38159


namespace NUMINAMATH_CALUDE_function_set_property_l381_38189

/-- A set of functions from ℝ to ℝ satisfying a specific property -/
def FunctionSet : Type := {A : Set (ℝ → ℝ) // 
  ∀ (f₁ f₂ : ℝ → ℝ), f₁ ∈ A → f₂ ∈ A → 
    ∃ (f₃ : ℝ → ℝ), f₃ ∈ A ∧ 
      ∀ (x y : ℝ), f₁ (f₂ y - x) + 2 * x = f₃ (x + y)}

/-- The main theorem -/
theorem function_set_property (A : FunctionSet) :
  ∀ (f : ℝ → ℝ), f ∈ A.val → ∀ (x : ℝ), f (x - f x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_set_property_l381_38189


namespace NUMINAMATH_CALUDE_sam_and_dan_balloons_l381_38177

/-- The number of red balloons Sam and Dan have in total -/
def total_balloons (sam_initial : ℝ) (sam_given : ℝ) (dan : ℝ) : ℝ :=
  (sam_initial - sam_given) + dan

/-- Theorem stating the total number of red balloons Sam and Dan have -/
theorem sam_and_dan_balloons :
  total_balloons 46.0 10.0 16.0 = 52.0 := by
  sorry

end NUMINAMATH_CALUDE_sam_and_dan_balloons_l381_38177


namespace NUMINAMATH_CALUDE_prime_between_30_and_50_l381_38196

theorem prime_between_30_and_50 (n : ℕ) :
  Prime n →
  30 < n →
  n < 50 →
  n % 6 = 1 →
  n % 5 ≠ 0 →
  n = 31 ∨ n = 37 ∨ n = 43 := by
sorry

end NUMINAMATH_CALUDE_prime_between_30_and_50_l381_38196


namespace NUMINAMATH_CALUDE_gcf_72_120_l381_38156

theorem gcf_72_120 : Nat.gcd 72 120 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcf_72_120_l381_38156


namespace NUMINAMATH_CALUDE_solve_equation_l381_38108

theorem solve_equation (x : ℚ) : (2 * x + 3) / 5 = 11 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l381_38108


namespace NUMINAMATH_CALUDE_number_of_paths_l381_38162

theorem number_of_paths (paths_A_to_B paths_B_to_D paths_D_to_C : ℕ) 
  (direct_path_A_to_C : ℕ) :
  paths_A_to_B = 2 →
  paths_B_to_D = 3 →
  paths_D_to_C = 3 →
  direct_path_A_to_C = 1 →
  paths_A_to_B * paths_B_to_D * paths_D_to_C + direct_path_A_to_C = 19 :=
by sorry

end NUMINAMATH_CALUDE_number_of_paths_l381_38162


namespace NUMINAMATH_CALUDE_proposition_truth_l381_38114

theorem proposition_truth : 
  (∀ x : ℝ, x > 0 → (3 : ℝ) ^ x > (2 : ℝ) ^ x) ∧ 
  (∀ x : ℝ, x < 0 → (3 : ℝ) * x ≤ (2 : ℝ) * x) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l381_38114


namespace NUMINAMATH_CALUDE_sequence_formula_l381_38160

/-- Given a sequence {a_n} where the sum of the first n terms S_n satisfies S_n = 2a_n + 1,
    prove that the general formula for a_n is -2^(n-1) -/
theorem sequence_formula (a : ℕ → ℤ) (S : ℕ → ℤ) 
    (h : ∀ n, S n = 2 * a n + 1) :
  ∀ n, a n = -2^(n-1) := by
sorry

end NUMINAMATH_CALUDE_sequence_formula_l381_38160


namespace NUMINAMATH_CALUDE_log3_of_9_cubed_l381_38157

-- Define the logarithm function for base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem log3_of_9_cubed : log3 (9^3) = 6 := by sorry

end NUMINAMATH_CALUDE_log3_of_9_cubed_l381_38157


namespace NUMINAMATH_CALUDE_jake_earnings_l381_38118

/-- Jake's earnings calculation -/
theorem jake_earnings (jacob_hourly_rate : ℝ) (jake_daily_hours : ℝ) (days : ℝ) :
  jacob_hourly_rate = 6 →
  jake_daily_hours = 8 →
  days = 5 →
  (3 * jacob_hourly_rate * jake_daily_hours * days : ℝ) = 720 := by
  sorry

end NUMINAMATH_CALUDE_jake_earnings_l381_38118


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l381_38101

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 2 + a 3 + a 4 + a 5 + a 6 = 90) →
  (a 1 + a 7 = 36) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l381_38101


namespace NUMINAMATH_CALUDE_pauls_money_duration_l381_38138

/-- Given Paul's earnings and weekly spending, prove how long the money will last. -/
theorem pauls_money_duration (lawn_earnings weed_earnings weekly_spending : ℕ) 
  (h1 : lawn_earnings = 68)
  (h2 : weed_earnings = 13)
  (h3 : weekly_spending = 9) :
  (lawn_earnings + weed_earnings) / weekly_spending = 9 := by
  sorry

end NUMINAMATH_CALUDE_pauls_money_duration_l381_38138


namespace NUMINAMATH_CALUDE_points_collinear_l381_38182

-- Define the points
variable (A B C K : Point)

-- Define the shapes
variable (square1 square2 : Square)
variable (triangle : Triangle)

-- Define the properties
variable (triangle_isosceles : IsIsosceles triangle)
variable (K_on_triangle_side : OnSide K triangle)

-- Define the theorem
theorem points_collinear (h1 : triangle_isosceles) (h2 : K_on_triangle_side) : 
  Collinear A B C := by sorry

end NUMINAMATH_CALUDE_points_collinear_l381_38182


namespace NUMINAMATH_CALUDE_janice_earnings_l381_38154

/-- Calculates the total earnings for a week given specific working conditions --/
def calculate_earnings (weekday_hours : ℕ) (weekend_hours : ℕ) (holiday_hours : ℕ) : ℕ :=
  let weekday_rate := 10
  let weekend_rate := 12
  let holiday_rate := 2 * weekend_rate
  let weekday_earnings := weekday_hours * weekday_rate
  let weekend_earnings := weekend_hours * weekend_rate
  let holiday_earnings := holiday_hours * holiday_rate
  weekday_earnings + weekend_earnings + holiday_earnings

/-- Theorem stating that Janice's earnings for the given week are $720 --/
theorem janice_earnings : calculate_earnings 30 25 5 = 720 := by
  sorry

end NUMINAMATH_CALUDE_janice_earnings_l381_38154


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l381_38140

theorem quadratic_one_solution (b : ℝ) (hb : b ≠ 0) :
  (∃! x, 3 * x^2 + b * x + 12 = 0) →
  ((b = 12 ∧ ∃ x, 3 * x^2 + b * x + 12 = 0 ∧ x = -2) ∨
   (b = -12 ∧ ∃ x, 3 * x^2 + b * x + 12 = 0 ∧ x = 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l381_38140


namespace NUMINAMATH_CALUDE_water_needed_l381_38110

/-- Represents the recipe for lemonade tea --/
structure LemonadeTea where
  lemonJuice : ℝ
  sugar : ℝ
  water : ℝ
  tea : ℝ

/-- Checks if the recipe satisfies the given conditions --/
def isValidRecipe (recipe : LemonadeTea) : Prop :=
  recipe.water = 3 * recipe.sugar ∧
  recipe.sugar = 1.5 * recipe.lemonJuice ∧
  recipe.tea = (recipe.water + recipe.sugar + recipe.lemonJuice) / 6 ∧
  recipe.lemonJuice = 4

/-- Theorem stating that a valid recipe requires 18 cups of water --/
theorem water_needed (recipe : LemonadeTea) 
  (h : isValidRecipe recipe) : recipe.water = 18 := by
  sorry


end NUMINAMATH_CALUDE_water_needed_l381_38110


namespace NUMINAMATH_CALUDE_small_tub_cost_l381_38120

def total_cost : ℕ := 48
def num_large_tubs : ℕ := 3
def num_small_tubs : ℕ := 6
def cost_large_tub : ℕ := 6

theorem small_tub_cost : 
  ∃ (cost_small_tub : ℕ), 
    cost_small_tub * num_small_tubs + cost_large_tub * num_large_tubs = total_cost ∧
    cost_small_tub = 5 :=
by sorry

end NUMINAMATH_CALUDE_small_tub_cost_l381_38120


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_one_l381_38126

theorem fraction_zero_implies_x_one :
  ∀ x : ℝ, (x - 1) / (x - 5) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_one_l381_38126


namespace NUMINAMATH_CALUDE_max_d_value_l381_38139

def a (n : ℕ) : ℕ := n^3 + 4

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value : ∃ (k : ℕ), d k = 433 ∧ ∀ (n : ℕ), d n ≤ 433 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l381_38139


namespace NUMINAMATH_CALUDE_min_value_theorem_l381_38133

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3*b = 1) :
  (1/a + 1/(3*b)) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3*b₀ = 1 ∧ 1/a₀ + 1/(3*b₀) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l381_38133


namespace NUMINAMATH_CALUDE_sphere_surface_area_l381_38166

theorem sphere_surface_area (V : ℝ) (r : ℝ) (A : ℝ) : 
  V = 72 * Real.pi → 
  V = (4/3) * Real.pi * r^3 → 
  A = 4 * Real.pi * r^2 → 
  A = 36 * Real.pi * 2^(2/3) := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l381_38166


namespace NUMINAMATH_CALUDE_bamboo_volume_proof_l381_38123

/-- An arithmetic sequence of 9 terms -/
def ArithmeticSequence (a : Fin 9 → ℚ) : Prop :=
  ∃ d : ℚ, ∀ i j : Fin 9, a j - a i = (j - i : ℤ) • d

theorem bamboo_volume_proof (a : Fin 9 → ℚ) 
  (h_arith : ArithmeticSequence a)
  (h_bottom : a 0 + a 1 + a 2 = 4)
  (h_top : a 5 + a 6 + a 7 + a 8 = 3) :
  a 3 + a 4 = 2 + 3/22 := by
  sorry

end NUMINAMATH_CALUDE_bamboo_volume_proof_l381_38123


namespace NUMINAMATH_CALUDE_binomial_sum_identity_l381_38127

theorem binomial_sum_identity (p q n : ℕ+) :
  (∑' k, (Nat.choose (p + k) p) * (Nat.choose (q + n - k) q)) = Nat.choose (p + q + n + 1) (p + q + 1) :=
sorry

end NUMINAMATH_CALUDE_binomial_sum_identity_l381_38127
