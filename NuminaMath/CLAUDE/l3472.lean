import Mathlib

namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_equation_three_solutions_equation_four_solutions_l3472_347215

-- Equation 1
theorem equation_one_solutions (x : ℝ) : 
  (x + 3)^2 = (1 - 2*x)^2 ↔ x = 4 ∨ x = -2/3 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  (x + 1)^2 = 4*x ↔ x = 1 := by sorry

-- Equation 3
theorem equation_three_solutions (x : ℝ) :
  2*x^2 - 5*x + 1 = 0 ↔ x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4 := by sorry

-- Equation 4
theorem equation_four_solutions (x : ℝ) :
  (2*x - 1)^2 = x*(3*x + 2) - 7 ↔ x = 4 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_equation_three_solutions_equation_four_solutions_l3472_347215


namespace NUMINAMATH_CALUDE_max_candy_leftover_l3472_347222

theorem max_candy_leftover (x : ℕ+) : 
  ∃ (q r : ℕ), x = 12 * q + r ∧ 0 < r ∧ r ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l3472_347222


namespace NUMINAMATH_CALUDE_smallest_integer_solution_of_inequalities_l3472_347245

theorem smallest_integer_solution_of_inequalities :
  ∀ x : ℤ,
  (5 * x + 7 > 3 * (x + 1)) ∧
  (1 - (3/2) * x ≤ (1/2) * x - 1) →
  x ≥ 1 ∧
  ∀ y : ℤ, y < 1 →
    ¬((5 * y + 7 > 3 * (y + 1)) ∧
      (1 - (3/2) * y ≤ (1/2) * y - 1)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_of_inequalities_l3472_347245


namespace NUMINAMATH_CALUDE_problem_solution_l3472_347213

theorem problem_solution : ∀ (P Q Y : ℚ),
  P = 3012 / 4 →
  Q = P / 2 →
  Y = P - Q →
  Y = 376.5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3472_347213


namespace NUMINAMATH_CALUDE_equation_solutions_l3472_347278

theorem equation_solutions (x : ℝ) :
  x ∈ Set.Ioo 0 π ∧ (Real.sin x + Real.cos x) * Real.tan x = 2 * Real.cos x ↔
  x = (1/2) * (Real.arctan 3 + Real.arcsin (Real.sqrt 10 / 10)) ∨
  x = (1/2) * (π - Real.arcsin (Real.sqrt 10 / 10) + Real.arctan 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3472_347278


namespace NUMINAMATH_CALUDE_finite_painted_blocks_l3472_347229

theorem finite_painted_blocks : 
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    ∀ m n r : ℕ, 
      m * n * r = 2 * (m - 2) * (n - 2) * (r - 2) → 
      (m, n, r) ∈ S := by
sorry

end NUMINAMATH_CALUDE_finite_painted_blocks_l3472_347229


namespace NUMINAMATH_CALUDE_coffee_x_ratio_l3472_347223

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents a coffee mixture -/
structure CoffeeMixture where
  p : ℕ  -- amount of coffee p in lbs
  v : ℕ  -- amount of coffee v in lbs

def total_p : ℕ := 24
def total_v : ℕ := 25

def coffee_x : CoffeeMixture := { p := 20, v := 0 }
def coffee_y : CoffeeMixture := { p := 0, v := 0 }

def ratio_y : Ratio := { numerator := 1, denominator := 5 }

theorem coffee_x_ratio : 
  coffee_x.p * 1 = coffee_x.v * 4 := by sorry

end NUMINAMATH_CALUDE_coffee_x_ratio_l3472_347223


namespace NUMINAMATH_CALUDE_no_finite_algorithm_for_infinite_sum_l3472_347220

-- Define what an algorithm is
def Algorithm : Type := ℕ → ℕ

-- Define the property of finiteness for algorithms
def IsFinite (a : Algorithm) : Prop := ∃ n : ℕ, ∀ m : ℕ, m ≥ n → a m = a n

-- Define the infinite sum
def InfiniteSum : ℕ → ℕ
  | 0 => 0
  | n + 1 => InfiniteSum n + (n + 1)

-- Theorem: There is no finite algorithm that can calculate the infinite sum
theorem no_finite_algorithm_for_infinite_sum :
  ¬∃ (a : Algorithm), (IsFinite a) ∧ (∀ n : ℕ, a n = InfiniteSum n) :=
sorry

end NUMINAMATH_CALUDE_no_finite_algorithm_for_infinite_sum_l3472_347220


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3472_347299

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 19 participants, where each participant
    plays exactly one game with each of the remaining participants,
    the total number of games played is 171. --/
theorem chess_tournament_games :
  num_games 19 = 171 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3472_347299


namespace NUMINAMATH_CALUDE_january_savings_l3472_347221

def savings_sequence (initial : ℝ) (n : ℕ) : ℝ :=
  initial + 4 * (n - 1)

def total_savings (initial : ℝ) (months : ℕ) : ℝ :=
  (List.range months).map (savings_sequence initial) |>.sum

theorem january_savings (x : ℝ) : total_savings x 6 = 126 → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_january_savings_l3472_347221


namespace NUMINAMATH_CALUDE_two_positions_from_six_candidates_l3472_347293

/-- The number of ways to select two distinct positions from a group of candidates. -/
def selectTwoPositions (n : ℕ) : ℕ := n * (n - 1)

/-- The number of candidates. -/
def numCandidates : ℕ := 6

/-- The observed number of ways to select two positions. -/
def observedSelections : ℕ := 30

/-- Theorem stating that selecting 2 distinct positions from 6 candidates results in 30 possible selections. -/
theorem two_positions_from_six_candidates :
  selectTwoPositions numCandidates = observedSelections := by
  sorry


end NUMINAMATH_CALUDE_two_positions_from_six_candidates_l3472_347293


namespace NUMINAMATH_CALUDE_students_playing_neither_l3472_347209

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) 
  (h1 : total = 40)
  (h2 : football = 26)
  (h3 : tennis = 20)
  (h4 : both = 17) :
  total - (football + tennis - both) = 11 := by
sorry

end NUMINAMATH_CALUDE_students_playing_neither_l3472_347209


namespace NUMINAMATH_CALUDE_max_snacks_with_15_dollars_l3472_347286

/-- Represents the number of snacks that can be bought with a given amount of money -/
def maxSnacks (money : ℕ) : ℕ :=
  let singlePrice := 2  -- Price of a single snack
  let packOf4Price := 5  -- Price of a pack of 4 snacks
  let packOf7Price := 8  -- Price of a pack of 7 snacks
  -- Function to calculate the maximum number of snacks
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the maximum number of snacks that can be bought with $15 is 12 -/
theorem max_snacks_with_15_dollars :
  maxSnacks 15 = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_snacks_with_15_dollars_l3472_347286


namespace NUMINAMATH_CALUDE_lowest_sample_number_48_8_48_l3472_347281

/-- Calculates the lowest number in a systematic sample. -/
def lowestSampleNumber (totalPopulation : ℕ) (sampleSize : ℕ) (highestNumber : ℕ) : ℕ :=
  highestNumber - (totalPopulation / sampleSize) * (sampleSize - 1)

/-- Theorem: In a systematic sampling of 8 students from 48, with highest number 48, the lowest is 6. -/
theorem lowest_sample_number_48_8_48 :
  lowestSampleNumber 48 8 48 = 6 := by
  sorry

#eval lowestSampleNumber 48 8 48

end NUMINAMATH_CALUDE_lowest_sample_number_48_8_48_l3472_347281


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_42_l3472_347288

-- Define the original angle
def original_angle : ℝ := 42

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 90 - angle

-- Define the supplement of an angle
def supplement (angle : ℝ) : ℝ := 180 - angle

-- Theorem statement
theorem supplement_of_complement_of_42 : 
  supplement (complement original_angle) = 132 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_42_l3472_347288


namespace NUMINAMATH_CALUDE_area_of_N_region_l3472_347219

-- Define the plane region for point M
def plane_region_M (a b : ℝ) : Prop := sorry

-- Define the transformation from M to N
def transform_M_to_N (a b : ℝ) : ℝ × ℝ := (a + b, a - b)

-- Define the plane region for point N
def plane_region_N (x y : ℝ) : Prop := sorry

-- Theorem statement
theorem area_of_N_region : 
  ∀ (a b : ℝ), plane_region_M a b → 
  (∃ (S : Set (ℝ × ℝ)), (∀ (x y : ℝ), (x, y) ∈ S ↔ plane_region_N x y) ∧ 
                         MeasureTheory.volume S = 4) :=
sorry

end NUMINAMATH_CALUDE_area_of_N_region_l3472_347219


namespace NUMINAMATH_CALUDE_binary_multiplication_division_l3472_347235

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents the binary number 11010₂ -/
def a : Nat := binary_to_nat [true, true, false, true, false]

/-- Represents the binary number 11100₂ -/
def b : Nat := binary_to_nat [true, true, true, false, false]

/-- Represents the binary number 100₂ -/
def c : Nat := binary_to_nat [true, false, false]

/-- Represents the binary number 10101101₂ -/
def result : Nat := binary_to_nat [true, false, true, false, true, true, false, true]

/-- Theorem stating that 11010₂ × 11100₂ ÷ 100₂ = 10101101₂ -/
theorem binary_multiplication_division :
  a * b / c = result := by sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_l3472_347235


namespace NUMINAMATH_CALUDE_contrapositive_false_l3472_347277

theorem contrapositive_false : ¬(∀ x y : ℝ, (x ≤ 0 ∨ y ≤ 0) → x + y ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_false_l3472_347277


namespace NUMINAMATH_CALUDE_notebook_redistribution_l3472_347216

theorem notebook_redistribution (total_notebooks : ℕ) (initial_boxes : ℕ) (new_notebooks_per_box : ℕ) :
  total_notebooks = 1200 →
  initial_boxes = 30 →
  new_notebooks_per_box = 35 →
  total_notebooks % new_notebooks_per_box = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_notebook_redistribution_l3472_347216


namespace NUMINAMATH_CALUDE_expression_value_l3472_347246

theorem expression_value (x y : ℝ) (hx : x = 2) (hy : y = 5) :
  (x^4 + 2*y^2) / 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3472_347246


namespace NUMINAMATH_CALUDE_condition_property_l3472_347252

theorem condition_property (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 1 → x^2 + y^2 ≥ 2) ∧
  (∃ x y : ℝ, x^2 + y^2 ≥ 2 ∧ ¬(x ≥ 1 ∧ y ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_condition_property_l3472_347252


namespace NUMINAMATH_CALUDE_binomial_expansion_terms_l3472_347272

theorem binomial_expansion_terms (x a : ℚ) (n : ℕ) :
  (Nat.choose n 3 : ℚ) * x^(n - 3) * a^3 = 330 ∧
  (Nat.choose n 4 : ℚ) * x^(n - 4) * a^4 = 792 ∧
  (Nat.choose n 5 : ℚ) * x^(n - 5) * a^5 = 1716 →
  n = 7 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_terms_l3472_347272


namespace NUMINAMATH_CALUDE_largest_angle_in_ratio_triangle_l3472_347210

theorem largest_angle_in_ratio_triangle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ratio : b = 2 * a ∧ c = 3 * a) (h_sum : a + b + c = 180) :
  c = 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_ratio_triangle_l3472_347210


namespace NUMINAMATH_CALUDE_library_shelf_theorem_l3472_347290

/-- Represents the thickness of a biology book -/
def biology_thickness : ℝ := 1

/-- Represents the thickness of a history book -/
def history_thickness : ℝ := 2 * biology_thickness

/-- Represents the length of the shelf -/
def shelf_length : ℝ := 1

theorem library_shelf_theorem 
  (B G P Q F : ℕ) 
  (h_distinct : B ≠ G ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ F ∧ 
                G ≠ P ∧ G ≠ Q ∧ G ≠ F ∧ 
                P ≠ Q ∧ P ≠ F ∧ 
                Q ≠ F)
  (h_positive : B > 0 ∧ G > 0 ∧ P > 0 ∧ Q > 0 ∧ F > 0)
  (h_fill1 : B * biology_thickness + G * history_thickness = shelf_length)
  (h_fill2 : P * biology_thickness + Q * history_thickness = shelf_length)
  (h_fill3 : F * biology_thickness = shelf_length) :
  F = B + 2*G ∧ F = P + 2*Q :=
sorry

end NUMINAMATH_CALUDE_library_shelf_theorem_l3472_347290


namespace NUMINAMATH_CALUDE_f_at_one_eq_neg_7878_l3472_347205

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

end NUMINAMATH_CALUDE_f_at_one_eq_neg_7878_l3472_347205


namespace NUMINAMATH_CALUDE_opposite_teal_is_yellow_l3472_347244

-- Define the colors
inductive Color
| Blue | Orange | Yellow | Violet | Teal | Lime

-- Define the faces of a cube
inductive Face
| Top | Bottom | Front | Back | Left | Right

-- Define a cube as a function from Face to Color
def Cube := Face → Color

-- Define the property of opposite faces
def opposite (f1 f2 : Face) : Prop :=
  (f1 = Face.Top ∧ f2 = Face.Bottom) ∨
  (f1 = Face.Bottom ∧ f2 = Face.Top) ∨
  (f1 = Face.Left ∧ f2 = Face.Right) ∨
  (f1 = Face.Right ∧ f2 = Face.Left) ∨
  (f1 = Face.Front ∧ f2 = Face.Back) ∨
  (f1 = Face.Back ∧ f2 = Face.Front)

-- Define the views of the cube
def view1 (c : Cube) : Prop :=
  c Face.Top = Color.Blue ∧ c Face.Front = Color.Violet ∧ c Face.Right = Color.Yellow

def view2 (c : Cube) : Prop :=
  c Face.Top = Color.Blue ∧ c Face.Front = Color.Orange ∧ c Face.Right = Color.Yellow

def view3 (c : Cube) : Prop :=
  c Face.Top = Color.Blue ∧ c Face.Front = Color.Lime ∧ c Face.Right = Color.Yellow

-- Theorem statement
theorem opposite_teal_is_yellow (c : Cube) :
  (∃ f1 f2 : Face, c f1 = Color.Teal ∧ opposite f1 f2) →
  (∀ f : Face, c f ≠ Color.Teal → c f ≠ Color.Yellow) →
  view1 c → view2 c → view3 c →
  ∃ f : Face, c f = Color.Teal ∧ c (Face.Right) = Color.Yellow ∧ opposite f Face.Right :=
by sorry

end NUMINAMATH_CALUDE_opposite_teal_is_yellow_l3472_347244


namespace NUMINAMATH_CALUDE_yellow_peaches_to_add_result_l3472_347214

/-- The number of yellow peaches needed to be added to satisfy the condition -/
def yellow_peaches_to_add (red green yellow : ℕ) : ℕ :=
  2 * (red + green) - yellow

/-- Theorem stating the number of yellow peaches to be added -/
theorem yellow_peaches_to_add_result :
  yellow_peaches_to_add 7 8 15 = 15 := by
  sorry

end NUMINAMATH_CALUDE_yellow_peaches_to_add_result_l3472_347214


namespace NUMINAMATH_CALUDE_red_markers_count_l3472_347217

def total_markers : ℕ := 105
def blue_markers : ℕ := 64

theorem red_markers_count : 
  ∃ (red_markers : ℕ), red_markers = total_markers - blue_markers ∧ red_markers = 41 := by
  sorry

end NUMINAMATH_CALUDE_red_markers_count_l3472_347217


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l3472_347233

theorem smallest_constant_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b / (a + b + 2 * c) + b * c / (b + c + 2 * a) + c * a / (c + a + 2 * b) ≤ (1/4) * (a + b + c)) ∧
  ∀ k : ℝ, k > 0 → k < 1/4 →
    ∃ a' b' c' : ℝ, a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
      a' * b' / (a' + b' + 2 * c') + b' * c' / (b' + c' + 2 * a') + c' * a' / (c' + a' + 2 * b') > k * (a' + b' + c') :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l3472_347233


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3472_347260

theorem necessary_not_sufficient_condition (a b : ℝ) : 
  (∀ x y : ℝ, x < y → x < y + 1) ∧ 
  (∃ x y : ℝ, x < y + 1 ∧ ¬(x < y)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3472_347260


namespace NUMINAMATH_CALUDE_y1_less_than_y2_l3472_347275

/-- A linear function f(x) = x + 4 -/
def f (x : ℝ) : ℝ := x + 4

/-- The theorem states that for two points on the graph of f,
    if the x-coordinate of the first point is less than the x-coordinate of the second point,
    then the y-coordinate of the first point is less than the y-coordinate of the second point. -/
theorem y1_less_than_y2 (y1 y2 : ℝ) 
    (h1 : f (-1/2) = y1) 
    (h2 : f 1 = y2) : 
  y1 < y2 := by
  sorry

end NUMINAMATH_CALUDE_y1_less_than_y2_l3472_347275


namespace NUMINAMATH_CALUDE_simplification_proof_l3472_347259

theorem simplification_proof (x a : ℝ) :
  (3 * x^2 - 1 - 2*x - 5 + 3*x - x^2 = 2 * x^2 + x - 6) ∧
  (4 * (2 * a^2 - 1 + 2*a) - 3 * (a - 1 + a^2) = 5 * a^2 + 5*a - 1) :=
by sorry

end NUMINAMATH_CALUDE_simplification_proof_l3472_347259


namespace NUMINAMATH_CALUDE_x_range_l3472_347255

def p (x : ℝ) : Prop := x^2 - 4*x + 3 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

theorem x_range (x : ℝ) :
  (∀ y : ℝ, ¬(p y ∧ q y)) ∧ (∃ y : ℝ, p y ∨ q y) →
  ((1 < x ∧ x ≤ 2) ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_x_range_l3472_347255


namespace NUMINAMATH_CALUDE_reinforcement_size_l3472_347224

/-- Calculates the size of the reinforcement given the initial garrison size, 
    initial provision duration, days passed before reinforcement, and 
    remaining provision duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
                             (days_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let provisions := initial_garrison * initial_duration
  let provisions_left := initial_garrison * (initial_duration - days_passed)
  (provisions_left / remaining_duration) - initial_garrison

theorem reinforcement_size :
  calculate_reinforcement 2000 54 18 20 = 1600 := by
  sorry

#eval calculate_reinforcement 2000 54 18 20

end NUMINAMATH_CALUDE_reinforcement_size_l3472_347224


namespace NUMINAMATH_CALUDE_pencil_distribution_l3472_347249

/-- Given an initial number of pencils, number of containers, and additional pencils,
    calculate the number of pencils that can be evenly distributed per container. -/
def evenDistribution (initialPencils : ℕ) (containers : ℕ) (additionalPencils : ℕ) : ℕ :=
  (initialPencils + additionalPencils) / containers

/-- Prove that given 150 initial pencils, 5 containers, and 30 additional pencils,
    the number of pencils that can be evenly distributed between the containers
    after receiving additional pencils is 36. -/
theorem pencil_distribution :
  evenDistribution 150 5 30 = 36 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l3472_347249


namespace NUMINAMATH_CALUDE_circle_tangency_l3472_347297

/-- Definition of circle O₁ -/
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Definition of circle O₂ -/
def circle_O₂ (x y a : ℝ) : Prop := (x + 4)^2 + (y - a)^2 = 25

/-- The distance between the centers of two internally tangent circles
    is equal to the difference of their radii -/
def internally_tangent (a : ℝ) : Prop := 
  (4^2 + a^2).sqrt = 5 - 1

theorem circle_tangency (a : ℝ) 
  (h : internally_tangent a) : a = 0 := by sorry

end NUMINAMATH_CALUDE_circle_tangency_l3472_347297


namespace NUMINAMATH_CALUDE_A_equals_B_l3472_347273

def A : Set ℝ := {x | ∃ a : ℝ, x = 5 - 4*a + a^2}
def B : Set ℝ := {y | ∃ b : ℝ, y = 4*b^2 + 4*b + 2}

theorem A_equals_B : A = B := by sorry

end NUMINAMATH_CALUDE_A_equals_B_l3472_347273


namespace NUMINAMATH_CALUDE_remainder_double_n_l3472_347202

theorem remainder_double_n (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_double_n_l3472_347202


namespace NUMINAMATH_CALUDE_simplify_expression_l3472_347265

theorem simplify_expression (x y : ℝ) : 8*x + 5*y + 3 - 2*x + 9*y + 15 = 6*x + 14*y + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3472_347265


namespace NUMINAMATH_CALUDE_complex_product_l3472_347289

theorem complex_product (A B C : ℂ) : 
  A = 7 + 3*I ∧ B = I ∧ C = 7 - 3*I → A * B * C = 58 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_l3472_347289


namespace NUMINAMATH_CALUDE_original_number_proof_l3472_347208

theorem original_number_proof (r : ℝ) : 
  (1.20 * r - r) + (1.35 * r - r) - (r - 0.50 * r) = 110 → r = 2200 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3472_347208


namespace NUMINAMATH_CALUDE_p_not_sufficient_p_not_necessary_p_neither_sufficient_nor_necessary_l3472_347212

/-- Proposition p: x ≠ 2 and y ≠ 3 -/
def p (x y : ℝ) : Prop := x ≠ 2 ∧ y ≠ 3

/-- Proposition q: x + y ≠ 5 -/
def q (x y : ℝ) : Prop := x + y ≠ 5

/-- p is not a sufficient condition for q -/
theorem p_not_sufficient : ¬∀ x y : ℝ, p x y → q x y :=
sorry

/-- p is not a necessary condition for q -/
theorem p_not_necessary : ¬∀ x y : ℝ, q x y → p x y :=
sorry

/-- p is neither a sufficient nor a necessary condition for q -/
theorem p_neither_sufficient_nor_necessary : (¬∀ x y : ℝ, p x y → q x y) ∧ (¬∀ x y : ℝ, q x y → p x y) :=
sorry

end NUMINAMATH_CALUDE_p_not_sufficient_p_not_necessary_p_neither_sufficient_nor_necessary_l3472_347212


namespace NUMINAMATH_CALUDE_round_trip_distance_l3472_347271

/-- Proves that the total distance of a round trip is 2 miles given the specified conditions -/
theorem round_trip_distance
  (outbound_time : ℝ) (return_time : ℝ) (average_speed : ℝ)
  (h1 : outbound_time = 10) -- outbound time in minutes
  (h2 : return_time = 20) -- return time in minutes
  (h3 : average_speed = 4) -- average speed in miles per hour
  : (outbound_time + return_time) / 60 * average_speed = 2 := by
  sorry

#check round_trip_distance

end NUMINAMATH_CALUDE_round_trip_distance_l3472_347271


namespace NUMINAMATH_CALUDE_cattle_train_departure_time_l3472_347295

/-- Proves that the cattle train left 6 hours before the diesel train --/
theorem cattle_train_departure_time (cattle_speed diesel_speed : ℝ) 
  (time_difference total_time : ℝ) (total_distance : ℝ) : 
  cattle_speed = 56 →
  diesel_speed = cattle_speed - 33 →
  total_time = 12 →
  total_distance = 1284 →
  total_distance = diesel_speed * total_time + cattle_speed * total_time + cattle_speed * time_difference →
  time_difference = 6 := by
  sorry

end NUMINAMATH_CALUDE_cattle_train_departure_time_l3472_347295


namespace NUMINAMATH_CALUDE_pave_square_iff_integer_hypotenuse_l3472_347240

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ  -- Length of side AB
  b : ℕ  -- Length of side AC
  h : a^2 + b^2 > 0  -- Ensures the triangle is non-degenerate

/-- Checks if a square can be completely paved with a given right triangle -/
def can_pave_square (t : RightTriangle) : Prop :=
  ∃ (n : ℕ), ∃ (m : ℕ), m * (t.a * t.b) = 2 * n^2 * (t.a^2 + t.b^2)

/-- The main theorem: A square can be paved if and only if the hypotenuse is an integer -/
theorem pave_square_iff_integer_hypotenuse (t : RightTriangle) :
  can_pave_square t ↔ ∃ (k : ℕ), k^2 = t.a^2 + t.b^2 :=
sorry

end NUMINAMATH_CALUDE_pave_square_iff_integer_hypotenuse_l3472_347240


namespace NUMINAMATH_CALUDE_min_sum_of_product_2450_l3472_347269

theorem min_sum_of_product_2450 (a b c : ℕ+) (h : a * b * c = 2450) :
  (∀ x y z : ℕ+, x * y * z = 2450 → a + b + c ≤ x + y + z) ∧ a + b + c = 82 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_2450_l3472_347269


namespace NUMINAMATH_CALUDE_trapezoid_area_theorem_l3472_347253

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  diagonal1 : ℝ
  diagonal2 : ℝ
  midpoint_segment : ℝ

/-- The area of a trapezoid with the given properties -/
def trapezoid_area (t : Trapezoid) : ℝ := 6

/-- Theorem: The area of a trapezoid with diagonals 3 and 5, and midpoint segment 2, is 6 -/
theorem trapezoid_area_theorem (t : Trapezoid) 
  (h1 : t.diagonal1 = 3) 
  (h2 : t.diagonal2 = 5) 
  (h3 : t.midpoint_segment = 2) : 
  trapezoid_area t = 6 := by
  sorry

#check trapezoid_area_theorem

end NUMINAMATH_CALUDE_trapezoid_area_theorem_l3472_347253


namespace NUMINAMATH_CALUDE_permutations_of_four_l3472_347211

theorem permutations_of_four (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_four_l3472_347211


namespace NUMINAMATH_CALUDE_mean_score_approx_71_l3472_347263

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


end NUMINAMATH_CALUDE_mean_score_approx_71_l3472_347263


namespace NUMINAMATH_CALUDE_modulo_residue_problem_l3472_347282

theorem modulo_residue_problem :
  (312 + 6 * 51 + 8 * 187 + 5 * 34) % 17 = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulo_residue_problem_l3472_347282


namespace NUMINAMATH_CALUDE_lcm_of_26_and_16_l3472_347206

theorem lcm_of_26_and_16 :
  let n : ℕ := 26
  let m : ℕ := 16
  let gcf : ℕ := 8
  Nat.lcm n m = 52 ∧ Nat.gcd n m = gcf :=
by sorry

end NUMINAMATH_CALUDE_lcm_of_26_and_16_l3472_347206


namespace NUMINAMATH_CALUDE_smallest_eight_digit_four_fours_l3472_347241

def is_eight_digit (n : ℕ) : Prop := 10000000 ≤ n ∧ n ≤ 99999999

def count_digit (n : ℕ) (d : ℕ) : ℕ :=
  (n.digits 10).filter (· = d) |>.length

theorem smallest_eight_digit_four_fours : 
  ∀ n : ℕ, is_eight_digit n → count_digit n 4 = 4 → 10004444 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_eight_digit_four_fours_l3472_347241


namespace NUMINAMATH_CALUDE_polynomial_perfect_square_l3472_347237

theorem polynomial_perfect_square (k : ℚ) : 
  (∃ a : ℚ, ∀ x : ℚ, x^2 + 2*(k-9)*x + (k^2 + 3*k + 4) = (x + a)^2) ↔ k = 11/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_perfect_square_l3472_347237


namespace NUMINAMATH_CALUDE_find_number_l3472_347232

theorem find_number : ∃! x : ℝ, (8 * x + 5400) / 12 = 530 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3472_347232


namespace NUMINAMATH_CALUDE_problem_statement_l3472_347279

theorem problem_statement (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3 / 7) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = -1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3472_347279


namespace NUMINAMATH_CALUDE_cylinder_cut_surface_increase_l3472_347296

/-- Represents the possible shapes of the increased surface area when cutting a cylinder --/
inductive IncreasedSurfaceShape
  | Circle
  | Rectangle

/-- Represents a cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents a way to cut a cylinder into two equal parts --/
structure CutMethod where
  (cylinder : Cylinder)
  (increasedShape : IncreasedSurfaceShape)

/-- States that there exist at least two different ways to cut a cylinder 
    resulting in different increased surface area shapes --/
theorem cylinder_cut_surface_increase 
  (c : Cylinder) : 
  ∃ (cut1 cut2 : CutMethod), 
    cut1.cylinder = c ∧ 
    cut2.cylinder = c ∧ 
    cut1.increasedShape ≠ cut2.increasedShape :=
sorry

end NUMINAMATH_CALUDE_cylinder_cut_surface_increase_l3472_347296


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l3472_347294

theorem quadratic_inequality_roots (c : ℝ) : 
  (∀ x, -x^2 + c*x - 8 < 0 ↔ x < 2 ∨ x > 6) → c = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l3472_347294


namespace NUMINAMATH_CALUDE_border_area_is_198_l3472_347257

-- Define the photograph dimensions
def photo_height : ℕ := 12
def photo_width : ℕ := 15

-- Define the frame border width
def border_width : ℕ := 3

-- Define the function to calculate the area of the border
def border_area (h w b : ℕ) : ℕ :=
  (h + 2*b) * (w + 2*b) - h * w

-- Theorem statement
theorem border_area_is_198 :
  border_area photo_height photo_width border_width = 198 := by
  sorry

end NUMINAMATH_CALUDE_border_area_is_198_l3472_347257


namespace NUMINAMATH_CALUDE_remainder_after_addition_l3472_347280

theorem remainder_after_addition : Int.mod (3452179 + 50) 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_addition_l3472_347280


namespace NUMINAMATH_CALUDE_number_of_observations_l3472_347285

theorem number_of_observations (initial_mean : ℝ) (wrong_value : ℝ) (correct_value : ℝ) (new_mean : ℝ) : 
  initial_mean = 36 → 
  wrong_value = 23 → 
  correct_value = 46 → 
  new_mean = 36.5 → 
  ∃ n : ℕ, n * initial_mean + (correct_value - wrong_value) = n * new_mean ∧ n = 46 :=
by sorry

end NUMINAMATH_CALUDE_number_of_observations_l3472_347285


namespace NUMINAMATH_CALUDE_sine_matrix_det_zero_l3472_347274

/-- The determinant of a 3x3 matrix with sine entries is zero -/
theorem sine_matrix_det_zero : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.sin 1, Real.sin 2, Real.sin 3],
    ![Real.sin 4, Real.sin 5, Real.sin 6],
    ![Real.sin 7, Real.sin 8, Real.sin 9]
  ]
  Matrix.det A = 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_matrix_det_zero_l3472_347274


namespace NUMINAMATH_CALUDE_scale_division_l3472_347242

/-- Represents the length of an object in feet and inches -/
structure Length where
  feet : ℕ
  inches : ℕ
  h : inches < 12

/-- Converts a Length to total inches -/
def Length.toInches (l : Length) : ℕ := l.feet * 12 + l.inches

/-- Converts total inches to a Length -/
def inchesToLength (totalInches : ℕ) : Length :=
  { feet := totalInches / 12,
    inches := totalInches % 12,
    h := by
      apply Nat.mod_lt
      exact Nat.zero_lt_succ 11 }

theorem scale_division (scale : Length) (h : scale.feet = 6 ∧ scale.inches = 8) :
  let totalInches := scale.toInches
  let halfInches := totalInches / 2
  let halfLength := inchesToLength halfInches
  halfLength.feet = 3 ∧ halfLength.inches = 4 := by
  sorry


end NUMINAMATH_CALUDE_scale_division_l3472_347242


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_at_min_mn_l3472_347228

/-- Given that 1/m + 2/n = 1 with m > 0 and n > 0, prove that the eccentricity of the ellipse
    x²/m² + y²/n² = 1 is √3/2 when mn takes its minimum value. -/
theorem ellipse_eccentricity_at_min_mn (m n : ℝ) 
  (h1 : m > 0) (h2 : n > 0) (h3 : 1/m + 2/n = 1) : 
  let e := Real.sqrt (1 - (min m n)^2 / (max m n)^2)
  ∃ (x : ℝ), (x = mn) ∧ (∀ y : ℝ, y = m*n → x ≤ y) → e = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_at_min_mn_l3472_347228


namespace NUMINAMATH_CALUDE_intersection_A_B_l3472_347264

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}

-- Define set B
def B : Set ℝ := {-4, 1, 3, 5}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3472_347264


namespace NUMINAMATH_CALUDE_fibonacci_gcd_property_l3472_347276

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_gcd_property :
  Nat.gcd (fib 2017) (fib 99 * fib 101 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_gcd_property_l3472_347276


namespace NUMINAMATH_CALUDE_max_metro_speed_l3472_347200

/-- Represents the metro system and the students' travel scenario -/
structure MetroSystem where
  v : ℕ  -- Speed of metro trains in km/h
  S : ℝ  -- Distance between two nearest metro stations
  R : ℝ  -- Distance from home to nearest station

/-- Conditions for the metro system -/
def validMetroSystem (m : MetroSystem) : Prop :=
  m.S > 0 ∧ m.R > 0 ∧ m.R < m.S / 2

/-- Yegor's travel condition -/
def yegorCondition (m : MetroSystem) : Prop :=
  m.S / 24 > m.R / m.v

/-- Nikita's travel condition -/
def nikitaCondition (m : MetroSystem) : Prop :=
  m.S / 12 < (m.R + m.S) / m.v

/-- The maximum speed theorem -/
theorem max_metro_speed :
  ∃ (m : MetroSystem),
    validMetroSystem m ∧
    yegorCondition m ∧
    nikitaCondition m ∧
    (∀ (m' : MetroSystem),
      validMetroSystem m' ∧ yegorCondition m' ∧ nikitaCondition m' →
      m'.v ≤ m.v) ∧
    m.v = 23 := by
  sorry

end NUMINAMATH_CALUDE_max_metro_speed_l3472_347200


namespace NUMINAMATH_CALUDE_arctg_arcctg_comparison_l3472_347258

theorem arctg_arcctg_comparison : (5 * Real.sqrt 7) / 4 > Real.arctan (2 + Real.sqrt 5) + Real.arctan (1 / (2 - Real.sqrt 5)) := by
  sorry

end NUMINAMATH_CALUDE_arctg_arcctg_comparison_l3472_347258


namespace NUMINAMATH_CALUDE_total_rent_calculation_l3472_347226

/-- Calculates the total rent collected in a year for a rental building --/
theorem total_rent_calculation (total_units : ℕ) (occupancy_rate : ℚ) (rent_per_unit : ℕ) : 
  total_units = 100 → 
  occupancy_rate = 3/4 →
  rent_per_unit = 400 →
  (total_units : ℚ) * occupancy_rate * rent_per_unit * 12 = 360000 := by
  sorry

end NUMINAMATH_CALUDE_total_rent_calculation_l3472_347226


namespace NUMINAMATH_CALUDE_expand_binomials_l3472_347262

theorem expand_binomials (x : ℝ) : (7 * x + 9) * (3 * x + 4) = 21 * x^2 + 55 * x + 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l3472_347262


namespace NUMINAMATH_CALUDE_julio_fishing_l3472_347291

theorem julio_fishing (fish_per_hour : ℕ) (hours : ℕ) (lost_fish : ℕ) (total_fish : ℕ) : 
  hours = 9 → lost_fish = 15 → total_fish = 48 → fish_per_hour * hours - lost_fish = total_fish → fish_per_hour = 7 := by
sorry

end NUMINAMATH_CALUDE_julio_fishing_l3472_347291


namespace NUMINAMATH_CALUDE_prob_region_D_total_prob_is_one_l3472_347203

/-- Represents the regions on the wheel of fortune -/
inductive Region
| A
| B
| C
| D

/-- The probability function for the wheel of fortune -/
def P : Region → ℚ
| Region.A => 1/4
| Region.B => 1/3
| Region.C => 1/6
| Region.D => 1 - (1/4 + 1/3 + 1/6)

/-- The theorem stating that the probability of landing on region D is 1/4 -/
theorem prob_region_D : P Region.D = 1/4 := by
  sorry

/-- The sum of all probabilities is 1 -/
theorem total_prob_is_one : P Region.A + P Region.B + P Region.C + P Region.D = 1 := by
  sorry

end NUMINAMATH_CALUDE_prob_region_D_total_prob_is_one_l3472_347203


namespace NUMINAMATH_CALUDE_marcus_pies_l3472_347239

def pies_left (batch_size : ℕ) (num_batches : ℕ) (dropped_pies : ℕ) : ℕ :=
  batch_size * num_batches - dropped_pies

theorem marcus_pies :
  pies_left 5 7 8 = 27 := by
  sorry

end NUMINAMATH_CALUDE_marcus_pies_l3472_347239


namespace NUMINAMATH_CALUDE_product_of_real_parts_l3472_347261

theorem product_of_real_parts : ∃ (z₁ z₂ : ℂ),
  (z₁^2 - 4*z₁ = 3*Complex.I) ∧
  (z₂^2 - 4*z₂ = 3*Complex.I) ∧
  (z₁ ≠ z₂) ∧
  (Complex.re z₁ * Complex.re z₂ = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_product_of_real_parts_l3472_347261


namespace NUMINAMATH_CALUDE_exam_score_standard_deviations_l3472_347267

/-- Given an exam with mean score 76, where 60 is 2 standard deviations below the mean,
    prove that 100 is 3 standard deviations above the mean. -/
theorem exam_score_standard_deviations 
  (mean : ℝ) 
  (std_dev : ℝ) 
  (h1 : mean = 76) 
  (h2 : mean - 2 * std_dev = 60) 
  (h3 : mean + 3 * std_dev = 100) : 
  100 = mean + 3 * std_dev := by
  sorry

end NUMINAMATH_CALUDE_exam_score_standard_deviations_l3472_347267


namespace NUMINAMATH_CALUDE_no_divisible_by_six_l3472_347287

theorem no_divisible_by_six : ∀ y : ℕ, y < 10 → ¬(36000 + 100 * y + 25) % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_divisible_by_six_l3472_347287


namespace NUMINAMATH_CALUDE_circle_tangent_properties_l3472_347236

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define a point on the x-axis
def on_x_axis (Q : ℝ × ℝ) : Prop := Q.2 = 0

-- Define the tangent property
def are_tangents (Q A B : ℝ × ℝ) : Prop := sorry

-- Define the distance between two points
def distance (P₁ P₂ : ℝ × ℝ) : ℝ := sorry

-- Define a line passing through a point
def line_passes_through (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop := l P

-- The main theorem
theorem circle_tangent_properties :
  ∀ (Q A B : ℝ × ℝ),
  circle_M A.1 A.2 ∧ circle_M B.1 B.2 ∧
  on_x_axis Q ∧
  are_tangents Q A B →
  (distance A B = 4 * Real.sqrt 2 / 3 → distance (0, 2) Q = 3) ∧
  (∃ (l : ℝ × ℝ → Prop), ∀ (Q' : ℝ × ℝ), on_x_axis Q' ∧ are_tangents Q' A B → 
    line_passes_through (0, 3/2) l) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_properties_l3472_347236


namespace NUMINAMATH_CALUDE_algae_growth_time_l3472_347266

/-- The growth factor of the algae population every 5 hours -/
def growth_factor : ℕ := 3

/-- The initial number of algae cells -/
def initial_cells : ℕ := 200

/-- The target number of algae cells -/
def target_cells : ℕ := 145800

/-- The time in hours for one growth cycle -/
def cycle_time : ℕ := 5

/-- The function to calculate the number of cells after a given number of cycles -/
def cells_after_cycles (n : ℕ) : ℕ :=
  initial_cells * growth_factor ^ n

/-- The theorem stating the time taken for the algae to grow to at least the target number of cells -/
theorem algae_growth_time : ∃ (t : ℕ), 
  cells_after_cycles (t / cycle_time) ≥ target_cells ∧ 
  ∀ (s : ℕ), s < t → cells_after_cycles (s / cycle_time) < target_cells :=
by sorry

end NUMINAMATH_CALUDE_algae_growth_time_l3472_347266


namespace NUMINAMATH_CALUDE_octal_sum_example_l3472_347268

/-- Represents a number in base 8 --/
def OctalNumber := Nat

/-- Converts a natural number to its octal representation --/
def toOctal (n : Nat) : OctalNumber := sorry

/-- Adds two octal numbers --/
def octalAdd (a b : OctalNumber) : OctalNumber := sorry

/-- Theorem: The sum of 356₈, 672₈, and 145₈ is 1477₈ in base 8 --/
theorem octal_sum_example : 
  octalAdd (octalAdd (toOctal 356) (toOctal 672)) (toOctal 145) = toOctal 1477 := by sorry

end NUMINAMATH_CALUDE_octal_sum_example_l3472_347268


namespace NUMINAMATH_CALUDE_count_lines_4x4_grid_l3472_347204

/-- A point in a 2D grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- A line in a 2D grid -/
structure GridLine where
  points : Set GridPoint

/-- A 4-by-4 grid of lattice points -/
def Grid4x4 : Set GridPoint :=
  {p | p.x < 4 ∧ p.y < 4}

/-- A function that determines if a line passes through at least two points in the grid -/
def passesThrough2Points (l : GridLine) (grid : Set GridPoint) : Prop :=
  (l.points ∩ grid).ncard ≥ 2

/-- The set of all lines that pass through at least two points in the 4-by-4 grid -/
def validLines : Set GridLine :=
  {l | passesThrough2Points l Grid4x4}

theorem count_lines_4x4_grid :
  (validLines).ncard = 88 := by sorry

end NUMINAMATH_CALUDE_count_lines_4x4_grid_l3472_347204


namespace NUMINAMATH_CALUDE_box_dimensions_l3472_347225

theorem box_dimensions (x : ℝ) 
  (h1 : x > 0)
  (h2 : ∃ (bow_length : ℝ), 6 * x + bow_length = 156)
  (h3 : ∃ (bow_length : ℝ), 7 * x + bow_length = 178) :
  x = 22 := by
sorry

end NUMINAMATH_CALUDE_box_dimensions_l3472_347225


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l3472_347201

/-- Theorem: For a rectangular field with length twice its width and perimeter 600 meters,
    the width is 100 meters and the length is 200 meters. -/
theorem rectangular_field_dimensions :
  ∀ (width length : ℝ),
  length = 2 * width →
  2 * (length + width) = 600 →
  width = 100 ∧ length = 200 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l3472_347201


namespace NUMINAMATH_CALUDE_linear_function_proof_l3472_347247

/-- A linear function passing through two given points -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  point1_x : ℝ
  point1_y : ℝ
  point2_x : ℝ
  point2_y : ℝ
  eq_at_point1 : point1_y = k * point1_x + b
  eq_at_point2 : point2_y = k * point2_x + b

/-- The specific linear function passing through (2,1) and (-3,6) -/
def specificLinearFunction : LinearFunction := {
  k := -1
  b := 3
  point1_x := 2
  point1_y := 1
  point2_x := -3
  point2_y := 6
  eq_at_point1 := by sorry
  eq_at_point2 := by sorry
}

theorem linear_function_proof :
  (specificLinearFunction.k = -1 ∧ specificLinearFunction.b = 3) ∧
  ¬(5 = specificLinearFunction.k * (-1) + specificLinearFunction.b) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_proof_l3472_347247


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l3472_347284

/-- The area of the region covered by two identical squares overlapping to form a regular octagon
    but not covered by a circle, given the circle's radius and π value. -/
theorem shaded_area_theorem (R : ℝ) (π : ℝ) (h1 : R = 60) (h2 : π = 3.14) :
  let total_square_area := 2 * R * R
  let circle_area := π * R * R
  total_square_area - circle_area = 3096 := by sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l3472_347284


namespace NUMINAMATH_CALUDE_min_bottles_to_fill_l3472_347256

theorem min_bottles_to_fill (large_capacity : ℕ) (small_capacity1 small_capacity2 : ℕ) :
  large_capacity = 720 ∧ small_capacity1 = 40 ∧ small_capacity2 = 45 →
  ∃ (x y : ℕ), x * small_capacity1 + y * small_capacity2 = large_capacity ∧
                x + y = 16 ∧
                ∀ (a b : ℕ), a * small_capacity1 + b * small_capacity2 = large_capacity →
                              x + y ≤ a + b :=
by sorry

end NUMINAMATH_CALUDE_min_bottles_to_fill_l3472_347256


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_19_l3472_347243

theorem consecutive_integers_sqrt_19 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 19) → (Real.sqrt 19 < b) → (a + b = 9) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_19_l3472_347243


namespace NUMINAMATH_CALUDE_angle_A_is_135_l3472_347254

/-- A trapezoid with specific angle relationships -/
structure SpecialTrapezoid where
  /-- The measure of angle A in degrees -/
  A : ℝ
  /-- The measure of angle B in degrees -/
  B : ℝ
  /-- The measure of angle C in degrees -/
  C : ℝ
  /-- The measure of angle D in degrees -/
  D : ℝ
  /-- AB is parallel to CD -/
  parallel : A + D = 180
  /-- Angle A is three times angle D -/
  A_eq_3D : A = 3 * D
  /-- Angle C is four times angle B -/
  C_eq_4B : C = 4 * B

/-- The measure of angle A in a special trapezoid is 135 degrees -/
theorem angle_A_is_135 (t : SpecialTrapezoid) : t.A = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_135_l3472_347254


namespace NUMINAMATH_CALUDE_initial_men_count_l3472_347283

/-- Represents the work completion scenario in a garment industry -/
structure WorkScenario where
  men : ℕ
  hours_per_day : ℕ
  days : ℕ

/-- Calculates the total man-hours for a given work scenario -/
def total_man_hours (scenario : WorkScenario) : ℕ :=
  scenario.men * scenario.hours_per_day * scenario.days

/-- The initial work scenario -/
def initial_scenario (initial_men : ℕ) : WorkScenario :=
  { men := initial_men, hours_per_day := 8, days := 10 }

/-- The second work scenario -/
def second_scenario : WorkScenario :=
  { men := 8, hours_per_day := 15, days := 8 }

/-- Theorem stating that the initial number of men is 12 -/
theorem initial_men_count : ∃ (initial_men : ℕ), 
  initial_men = 12 ∧ 
  total_man_hours (initial_scenario initial_men) = total_man_hours second_scenario :=
sorry

end NUMINAMATH_CALUDE_initial_men_count_l3472_347283


namespace NUMINAMATH_CALUDE_no_convex_polygon_from_regular_triangles_l3472_347218

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields

/-- A regular triangle -/
structure RegularTriangle where
  -- Add necessary fields

/-- Predicate to check if triangles are non-overlapping -/
def non_overlapping (T : List RegularTriangle) : Prop :=
  sorry

/-- Predicate to check if triangles are distinct -/
def distinct (T : List RegularTriangle) : Prop :=
  sorry

/-- Predicate to check if a polygon is composed of given triangles -/
def composed_of (P : ConvexPolygon) (T : List RegularTriangle) : Prop :=
  sorry

theorem no_convex_polygon_from_regular_triangles 
  (P : ConvexPolygon) (T : List RegularTriangle) :
  T.length ≥ 2 → non_overlapping T → distinct T → ¬(composed_of P T) :=
sorry

end NUMINAMATH_CALUDE_no_convex_polygon_from_regular_triangles_l3472_347218


namespace NUMINAMATH_CALUDE_inheritance_tax_problem_l3472_347230

theorem inheritance_tax_problem (x : ℝ) : 
  (0.25 * x) + (0.15 * (x - 0.25 * x)) = 15000 → x = 41379 :=
by sorry

end NUMINAMATH_CALUDE_inheritance_tax_problem_l3472_347230


namespace NUMINAMATH_CALUDE_painting_problem_l3472_347238

/-- The fraction of a wall that can be painted by two people working together in a given time -/
def combined_painting_fraction (rate1 rate2 time : ℚ) : ℚ :=
  (rate1 + rate2) * time

theorem painting_problem :
  let heidi_rate : ℚ := 1 / 60
  let linda_rate : ℚ := 1 / 40
  let work_time : ℚ := 12
  combined_painting_fraction heidi_rate linda_rate work_time = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_painting_problem_l3472_347238


namespace NUMINAMATH_CALUDE_triangle_trig_identity_l3472_347231

theorem triangle_trig_identity (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) (h1 : a = 4) (h2 : b = 7) (h3 : c = 5) :
  let α := Real.arccos ((c^2 + a^2 - b^2) / (2 * c * a))
  (Real.sin (α/2))^6 + (Real.cos (α/2))^6 = 7/25 := by sorry

end NUMINAMATH_CALUDE_triangle_trig_identity_l3472_347231


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3472_347270

theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : ∃ r : ℝ, b = a * r ∧ c = b * r) : 
  (a = 25 ∧ c = 1/4) → b = 5/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3472_347270


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3472_347248

/-- The lateral surface area of a cylinder with given base circumference and height -/
def lateral_surface_area (base_circumference : ℝ) (height : ℝ) : ℝ :=
  base_circumference * height

/-- Theorem: The lateral surface area of a cylinder with base circumference 5cm and height 2cm is 10 cm² -/
theorem cylinder_lateral_surface_area :
  lateral_surface_area 5 2 = 10 := by
sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3472_347248


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3472_347298

-- Define an isosceles triangle with one interior angle of 40°
structure IsoscelesTriangle where
  base_angle : ℝ
  is_isosceles : True
  has_40_degree_angle : base_angle = 40 ∨ 180 - 2 * base_angle = 40

-- Theorem stating that the vertex angle is either 40° or 100°
theorem isosceles_triangle_vertex_angle (t : IsoscelesTriangle) :
  (180 - 2 * t.base_angle = 40) ∨ (180 - 2 * t.base_angle = 100) :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3472_347298


namespace NUMINAMATH_CALUDE_average_of_eleven_numbers_l3472_347292

theorem average_of_eleven_numbers (first_six_avg : ℝ) (last_six_avg : ℝ) (sixth_number : ℝ) :
  first_six_avg = 19 →
  last_six_avg = 27 →
  sixth_number = 34 →
  (6 * first_six_avg + 6 * last_six_avg - sixth_number) / 11 = 22 :=
by sorry

end NUMINAMATH_CALUDE_average_of_eleven_numbers_l3472_347292


namespace NUMINAMATH_CALUDE_henry_twice_jills_age_l3472_347250

theorem henry_twice_jills_age (henry_age jill_age : ℕ) (years_ago : ℕ) : 
  henry_age + jill_age = 43 →
  henry_age = 27 →
  jill_age = 16 →
  henry_age - years_ago = 2 * (jill_age - years_ago) →
  years_ago = 5 := by
  sorry

end NUMINAMATH_CALUDE_henry_twice_jills_age_l3472_347250


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_25_23_l3472_347227

theorem half_abs_diff_squares_25_23 : (1 / 2 : ℝ) * |25^2 - 23^2| = 48 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_25_23_l3472_347227


namespace NUMINAMATH_CALUDE_express_regular_train_speed_ratio_l3472_347207

/-- The ratio of speeds between an express train and a regular train -/
def speed_ratio : ℝ := 2.5

/-- The time taken by the regular train from Moscow to St. Petersburg -/
def regular_train_time : ℝ := 10

/-- The time difference in arrival between regular and express trains -/
def arrival_time_difference : ℝ := 3

/-- The waiting time for the express train -/
def express_train_wait_time : ℝ := 3

/-- The time after departure when both trains are at the same distance from Moscow -/
def equal_distance_time : ℝ := 2

theorem express_regular_train_speed_ratio :
  ∀ (v_regular v_express : ℝ),
    v_regular > 0 →
    v_express > 0 →
    express_train_wait_time > 2.5 →
    v_express * equal_distance_time = v_regular * (express_train_wait_time + equal_distance_time) →
    v_express * (regular_train_time - arrival_time_difference - express_train_wait_time) = v_regular * regular_train_time →
    v_express / v_regular = speed_ratio := by
  sorry

end NUMINAMATH_CALUDE_express_regular_train_speed_ratio_l3472_347207


namespace NUMINAMATH_CALUDE_triangle_smallest_side_l3472_347251

theorem triangle_smallest_side (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_inequality : a^2 + b^2 > 5 * c^2) : c < a ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_triangle_smallest_side_l3472_347251


namespace NUMINAMATH_CALUDE_ten_machines_four_minutes_l3472_347234

/-- The number of bottles produced by a given number of machines in a given time -/
def bottles_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  let bottles_per_minute := (420 * machines) / 6
  bottles_per_minute * minutes

/-- Theorem stating that 10 machines produce 2800 bottles in 4 minutes -/
theorem ten_machines_four_minutes :
  bottles_produced 10 4 = 2800 := by
  sorry

#eval bottles_produced 10 4

end NUMINAMATH_CALUDE_ten_machines_four_minutes_l3472_347234
