import Mathlib

namespace tiffany_found_two_bags_l2442_244270

/-- The number of bags Tiffany found on the next day -/
def bags_found_next_day (bags_monday : ℕ) (total_bags : ℕ) : ℕ :=
  total_bags - bags_monday

/-- Theorem: Tiffany found 2 bags on the next day -/
theorem tiffany_found_two_bags :
  let bags_monday := 4
  let total_bags := 6
  bags_found_next_day bags_monday total_bags = 2 := by
  sorry

end tiffany_found_two_bags_l2442_244270


namespace quadratic_roots_expression_l2442_244255

theorem quadratic_roots_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (hroots : a^2 - c * a^2 + c = 0 ∧ b^2 - c * b^2 + c = 0) :
  (a * Real.sqrt (1 - 1 / b^2) + b * Real.sqrt (1 - 1 / a^2) = 2) ∨
  (a * Real.sqrt (1 - 1 / b^2) + b * Real.sqrt (1 - 1 / a^2) = -2) ∨
  (a * Real.sqrt (1 - 1 / b^2) + b * Real.sqrt (1 - 1 / a^2) = 0) :=
by sorry

end quadratic_roots_expression_l2442_244255


namespace smallest_m_for_nth_root_in_T_l2442_244258

def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

theorem smallest_m_for_nth_root_in_T : 
  (∀ n : ℕ, n ≥ 12 → ∃ z ∈ T, z^n = 1) ∧ 
  (∀ m : ℕ, m < 12 → ∃ n : ℕ, n ≥ m ∧ ∀ z ∈ T, z^n ≠ 1) :=
sorry

end smallest_m_for_nth_root_in_T_l2442_244258


namespace central_cell_removed_theorem_corner_cell_removed_theorem_l2442_244244

-- Define a 7x7 grid
def Grid := Fin 7 → Fin 7 → Bool

-- Define a domino placement
structure Domino where
  x : Fin 7
  y : Fin 7
  horizontal : Bool

-- Define a tiling of the grid
def Tiling := List Domino

-- Function to check if a tiling is valid for a given grid
def is_valid_tiling (g : Grid) (t : Tiling) : Prop := sorry

-- Function to count horizontal dominoes in a tiling
def count_horizontal (t : Tiling) : Nat := sorry

-- Function to count vertical dominoes in a tiling
def count_vertical (t : Tiling) : Nat := sorry

-- Define a grid with the central cell removed
def central_removed_grid : Grid := sorry

-- Define a grid with a corner cell removed
def corner_removed_grid : Grid := sorry

theorem central_cell_removed_theorem :
  ∃ t : Tiling, is_valid_tiling central_removed_grid t ∧
    count_horizontal t = count_vertical t := sorry

theorem corner_cell_removed_theorem :
  ¬∃ t : Tiling, is_valid_tiling corner_removed_grid t ∧
    count_horizontal t = count_vertical t := sorry

end central_cell_removed_theorem_corner_cell_removed_theorem_l2442_244244


namespace union_of_M_and_N_l2442_244265

def M : Set ℝ := {x : ℝ | x^2 - 3*x = 0}
def N : Set ℝ := {x : ℝ | x^2 - 5*x + 6 = 0}

theorem union_of_M_and_N : M ∪ N = {0, 2, 3} := by
  sorry

end union_of_M_and_N_l2442_244265


namespace consecutive_integers_product_sum_l2442_244212

theorem consecutive_integers_product_sum : ∃ (n : ℕ), 
  n > 0 ∧ n * (n + 1) = 1080 ∧ n + (n + 1) = 65 := by
  sorry

end consecutive_integers_product_sum_l2442_244212


namespace jessie_final_position_l2442_244226

/-- The number of steps Jessie takes in total -/
def total_steps : ℕ := 6

/-- The final position Jessie reaches -/
def final_position : ℕ := 24

/-- The number of steps to reach point x -/
def steps_to_x : ℕ := 4

/-- The number of steps from x to z -/
def steps_x_to_z : ℕ := 1

/-- The number of steps from z to y -/
def steps_z_to_y : ℕ := 1

/-- The length of each step -/
def step_length : ℚ := final_position / total_steps

/-- The position of point x -/
def x : ℚ := step_length * steps_to_x

/-- The position of point z -/
def z : ℚ := x + step_length * steps_x_to_z

/-- The position of point y -/
def y : ℚ := z + step_length * steps_z_to_y

theorem jessie_final_position : y = 24 := by
  sorry

end jessie_final_position_l2442_244226


namespace number_is_composite_l2442_244257

theorem number_is_composite : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 10^1962 + 1 = a * b := by
  sorry

end number_is_composite_l2442_244257


namespace triangle_side_length_l2442_244224

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  b = 2 * Real.sqrt 3 →
  B = 2 * π / 3 →
  C = π / 6 →
  a = 2 := by
  sorry

end triangle_side_length_l2442_244224


namespace total_bears_l2442_244233

/-- The number of bears in a national park --/
def bear_population (black white brown : ℕ) : ℕ :=
  black + white + brown

/-- Theorem: Given the conditions, the total bear population is 190 --/
theorem total_bears : ∀ (black white brown : ℕ),
  black = 60 →
  black = 2 * white →
  brown = black + 40 →
  bear_population black white brown = 190 := by
  sorry

end total_bears_l2442_244233


namespace midpoint_coordinate_product_l2442_244295

/-- Given that M(3,7) is the midpoint of CD and C(5,3) is one endpoint, 
    the product of the coordinates of point D is 11. -/
theorem midpoint_coordinate_product : 
  ∀ (D : ℝ × ℝ),
  (3, 7) = ((5 + D.1) / 2, (3 + D.2) / 2) →
  D.1 * D.2 = 11 := by
sorry

end midpoint_coordinate_product_l2442_244295


namespace only_two_subsets_implies_a_zero_or_one_l2442_244205

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x + 1 = 0}

-- State the theorem
theorem only_two_subsets_implies_a_zero_or_one :
  ∀ a : ℝ, (∀ S : Set ℝ, S ⊆ A a → (S = ∅ ∨ S = A a)) → (a = 0 ∨ a = 1) := by
  sorry

end only_two_subsets_implies_a_zero_or_one_l2442_244205


namespace number_puzzle_l2442_244269

theorem number_puzzle (x : ℤ) (h : x - 46 = 15) : x - 29 = 32 := by
  sorry

end number_puzzle_l2442_244269


namespace ellipse_a_range_l2442_244286

theorem ellipse_a_range (a b : ℝ) (e : ℝ) :
  a > b ∧ b > 0 ∧
  e ∈ Set.Icc (1 / Real.sqrt 3) (1 / Real.sqrt 2) ∧
  (∃ (M N : ℝ × ℝ),
    (M.1^2 / a^2 + M.2^2 / b^2 = 1) ∧
    (N.1^2 / a^2 + N.2^2 / b^2 = 1) ∧
    (M.2 = -M.1 + 1) ∧
    (N.2 = -N.1 + 1) ∧
    (M.1 * N.1 + M.2 * N.2 = 0)) →
  Real.sqrt 5 / 2 ≤ a ∧ a ≤ Real.sqrt 6 / 2 := by
sorry

end ellipse_a_range_l2442_244286


namespace sally_quarters_count_l2442_244230

/-- Given an initial number of quarters, quarters spent, and quarters found,
    calculate the final number of quarters Sally has. -/
def final_quarters (initial spent found : ℕ) : ℕ :=
  initial - spent + found

/-- Theorem stating that Sally's final number of quarters is 492 -/
theorem sally_quarters_count :
  final_quarters 760 418 150 = 492 := by
  sorry

end sally_quarters_count_l2442_244230


namespace distinct_arrangements_of_six_objects_l2442_244217

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem distinct_arrangements_of_six_objects : 
  factorial 6 = 720 := by sorry

end distinct_arrangements_of_six_objects_l2442_244217


namespace x_varies_as_eighth_power_of_z_l2442_244222

/-- Given that x varies as the fourth power of y, and y varies as the square of z,
    prove that x varies as the 8th power of z. -/
theorem x_varies_as_eighth_power_of_z
  (k : ℝ) (j : ℝ) (x y z : ℝ → ℝ)
  (h1 : ∀ t, x t = k * (y t)^4)
  (h2 : ∀ t, y t = j * (z t)^2) :
  ∃ m : ℝ, ∀ t, x t = m * (z t)^8 := by
  sorry

end x_varies_as_eighth_power_of_z_l2442_244222


namespace exists_unreachable_all_plus_configuration_l2442_244290

/-- Represents the sign in a cell: + or - -/
inductive Sign
| Plus
| Minus

/-- Represents an 8x8 grid of signs -/
def Grid := Fin 8 → Fin 8 → Sign

/-- Represents the allowed operations: flipping signs in 3x3 or 4x4 squares -/
def flip_square (g : Grid) (top_left : Fin 8 × Fin 8) (size : Fin 2) : Grid :=
  sorry

/-- Counts the number of minus signs in specific columns of the grid -/
def count_minus_outside_columns_3_6 (g : Grid) : Nat :=
  sorry

/-- Theorem stating that there exists a grid configuration that cannot be transformed to all plus signs -/
theorem exists_unreachable_all_plus_configuration :
  ∃ (initial : Grid), ¬∃ (final : Grid),
    (∀ i j, final i j = Sign.Plus) ∧
    (∃ (ops : List ((Fin 8 × Fin 8) × Fin 2)),
      final = ops.foldl (λ g (tl, s) => flip_square g tl s) initial) :=
  sorry

end exists_unreachable_all_plus_configuration_l2442_244290


namespace semicircle_segment_sum_l2442_244294

-- Define the semicircle and its properties
structure Semicircle where
  r : ℝ
  a : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  h_diameter : dist A B = 2 * r
  h_AT : a > 0 ∧ 2 * a < r / 2
  h_M_on_semicircle : dist M A * dist M B = r ^ 2
  h_N_on_semicircle : dist N A * dist N B = r ^ 2
  h_M_condition : dist M (0, -2 * a) / dist M A = 1
  h_N_condition : dist N (0, -2 * a) / dist N A = 1
  h_M_N_distinct : M ≠ N

-- State the theorem
theorem semicircle_segment_sum (s : Semicircle) : dist s.A s.M + dist s.A s.N = dist s.A s.B := by
  sorry

end semicircle_segment_sum_l2442_244294


namespace pizza_sales_l2442_244276

theorem pizza_sales (small_price large_price total_slices total_revenue : ℕ)
  (h1 : small_price = 150)
  (h2 : large_price = 250)
  (h3 : total_slices = 5000)
  (h4 : total_revenue = 1050000) :
  ∃ (small_slices large_slices : ℕ),
    small_slices + large_slices = total_slices ∧
    small_price * small_slices + large_price * large_slices = total_revenue ∧
    small_slices = 1500 :=
by sorry

end pizza_sales_l2442_244276


namespace coefficient_x_squared_in_expansion_l2442_244201

/-- The coefficient of x^2 in the expansion of (2x+1)^5 is 40 -/
theorem coefficient_x_squared_in_expansion : 
  (Finset.range 6).sum (fun k => 
    Nat.choose 5 k * (2^(5-k)) * (1^k) * 
    if 5 - k = 2 then 1 else 0) = 40 := by sorry

end coefficient_x_squared_in_expansion_l2442_244201


namespace sand_cone_weight_l2442_244273

/-- The weight of a sand cone given its dimensions and sand density -/
theorem sand_cone_weight (diameter : ℝ) (height_ratio : ℝ) (sand_density : ℝ) :
  diameter = 12 →
  height_ratio = 0.8 →
  sand_density = 100 →
  let radius := diameter / 2
  let height := height_ratio * diameter
  let volume := (1/3) * π * radius^2 * height
  volume * sand_density = 11520 * π :=
by sorry

end sand_cone_weight_l2442_244273


namespace lila_sticker_count_l2442_244206

/-- The number of stickers each person has -/
structure StickerCount where
  kristoff : ℕ
  riku : ℕ
  lila : ℕ

/-- The conditions of the sticker problem -/
def sticker_problem (s : StickerCount) : Prop :=
  s.kristoff = 85 ∧
  s.riku = 25 * s.kristoff ∧
  s.lila = 2 * (s.kristoff + s.riku)

/-- The theorem stating that Lila has 4420 stickers -/
theorem lila_sticker_count (s : StickerCount) 
  (h : sticker_problem s) : s.lila = 4420 := by
  sorry

end lila_sticker_count_l2442_244206


namespace nanjing_visitors_scientific_notation_l2442_244218

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem nanjing_visitors_scientific_notation :
  toScientificNotation 44300000 = ScientificNotation.mk 4.43 7 sorry := by
  sorry

end nanjing_visitors_scientific_notation_l2442_244218


namespace difference_of_squares_650_350_l2442_244215

theorem difference_of_squares_650_350 : 650^2 - 350^2 = 300000 := by
  sorry

end difference_of_squares_650_350_l2442_244215


namespace triangle_ratio_l2442_244274

/-- Triangle PQR with angle bisector PS intersecting MN at X -/
structure Triangle (P Q R S M N X : ℝ × ℝ) : Prop where
  m_on_pq : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • P + t • Q
  n_on_pr : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ N = (1 - t) • P + t • R
  ps_bisector : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (1 - t) • P + t • ((2/3) • Q + (1/3) • R)
  x_on_mn : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (1 - t) • M + t • N
  x_on_ps : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (1 - t) • P + t • S

/-- Given lengths in the triangle -/
structure TriangleLengths (P Q R S M N X : ℝ × ℝ) : Prop where
  pm_eq : ‖M - P‖ = 2
  mq_eq : ‖Q - M‖ = 6
  pn_eq : ‖N - P‖ = 3
  nr_eq : ‖R - N‖ = 9

/-- The main theorem -/
theorem triangle_ratio 
  (P Q R S M N X : ℝ × ℝ) 
  (h1 : Triangle P Q R S M N X) 
  (h2 : TriangleLengths P Q R S M N X) : 
  ‖X - P‖ / ‖S - P‖ = 1/4 :=
sorry

end triangle_ratio_l2442_244274


namespace exponent_addition_l2442_244277

theorem exponent_addition (a : ℝ) : a^3 * a^6 = a^9 := by
  sorry

end exponent_addition_l2442_244277


namespace basketball_non_gymnastics_percentage_l2442_244287

theorem basketball_non_gymnastics_percentage 
  (total : ℝ)
  (h_total_pos : total > 0)
  (h_basketball : total * (50 / 100) = total * 0.5)
  (h_gymnastics : total * (40 / 100) = total * 0.4)
  (h_both : (total * 0.5) * (30 / 100) = total * 0.15) :
  let non_gymnastics := total * 0.6
  let basketball_non_gymnastics := total * 0.35
  (basketball_non_gymnastics / non_gymnastics) * 100 = 58 := by
sorry

end basketball_non_gymnastics_percentage_l2442_244287


namespace power_function_continuous_l2442_244278

theorem power_function_continuous (n : ℕ+) :
  Continuous (fun x : ℝ => x ^ (n : ℕ)) :=
sorry

end power_function_continuous_l2442_244278


namespace perfect_square_condition_l2442_244282

theorem perfect_square_condition (n : ℕ) : 
  (∃ (k : ℕ), 2^(n+1) * n = k^2) ↔ 
  (∃ (m : ℕ), n = 2 * m^2) ∨ 
  (∃ (k : ℕ), n = k^2 ∧ k % 2 = 1) := by
sorry

end perfect_square_condition_l2442_244282


namespace polynomial_symmetry_l2442_244263

/-- Given a polynomial f(x) = ax^7 - bx^3 + cx - 5 where f(2) = 3, prove that f(-2) = -13 -/
theorem polynomial_symmetry (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^7 - b * x^3 + c * x - 5
  (f 2 = 3) → (f (-2) = -13) := by
sorry

end polynomial_symmetry_l2442_244263


namespace intersection_of_A_and_B_l2442_244228

def A : Set ℝ := {x | x + 1/2 ≥ 3/2}
def B : Set ℝ := {x | x^2 + x < 6}

theorem intersection_of_A_and_B : 
  A ∩ B = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x < 2)} := by sorry

end intersection_of_A_and_B_l2442_244228


namespace simplify_sqrt_expression_l2442_244202

theorem simplify_sqrt_expression :
  Real.sqrt 12 + 3 * Real.sqrt (1/3) = 3 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_expression_l2442_244202


namespace contractor_absent_days_l2442_244210

/-- Represents the contract details and calculates the number of absent days -/
def calculate_absent_days (total_days : ℕ) (pay_per_day : ℚ) (fine_per_day : ℚ) (total_pay : ℚ) : ℚ :=
  let worked_days := total_days - (total_pay - total_days * pay_per_day) / (pay_per_day + fine_per_day)
  total_days - worked_days

/-- Theorem stating that given the specific contract conditions, the number of absent days is 12 -/
theorem contractor_absent_days :
  let total_days : ℕ := 30
  let pay_per_day : ℚ := 25
  let fine_per_day : ℚ := 7.5
  let total_pay : ℚ := 360
  calculate_absent_days total_days pay_per_day fine_per_day total_pay = 12 := by
  sorry

#eval calculate_absent_days 30 25 7.5 360

end contractor_absent_days_l2442_244210


namespace quadratic_equation_solution_l2442_244243

theorem quadratic_equation_solution (p q : ℤ) (h1 : p + q = 2010) 
  (h2 : ∃ x1 x2 : ℤ, x1 > 0 ∧ x2 > 0 ∧ 67 * x1^2 + p * x1 + q = 0 ∧ 67 * x2^2 + p * x2 + q = 0) :
  p = -2278 := by sorry

end quadratic_equation_solution_l2442_244243


namespace roy_blue_pens_l2442_244240

/-- The number of blue pens Roy has -/
def blue_pens : ℕ := 2

/-- The number of black pens Roy has -/
def black_pens : ℕ := 2 * blue_pens

/-- The number of red pens Roy has -/
def red_pens : ℕ := 2 * black_pens - 2

/-- The total number of pens Roy has -/
def total_pens : ℕ := 12

theorem roy_blue_pens :
  blue_pens = 2 ∧
  black_pens = 2 * blue_pens ∧
  red_pens = 2 * black_pens - 2 ∧
  total_pens = blue_pens + black_pens + red_pens ∧
  total_pens = 12 := by
  sorry

end roy_blue_pens_l2442_244240


namespace factorial_equation_solution_l2442_244253

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_equation_solution :
  ∀ N : ℕ, N > 0 → (factorial 5 * factorial 9 = 12 * factorial N) → N = 10 := by
  sorry

end factorial_equation_solution_l2442_244253


namespace smallest_factor_of_36_l2442_244284

theorem smallest_factor_of_36 (a b c : ℤ) 
  (h1 : a * b * c = 36)
  (h2 : a + b + c = 4) :
  min a (min b c) = -4 :=
sorry

end smallest_factor_of_36_l2442_244284


namespace no_smallest_rational_l2442_244261

theorem no_smallest_rational : ¬ ∃ q : ℚ, ∀ r : ℚ, q ≤ r := by
  sorry

end no_smallest_rational_l2442_244261


namespace power_difference_sum_equals_six_l2442_244267

theorem power_difference_sum_equals_six : 3^2 - 2^2 + 1^2 = 6 := by
  sorry

end power_difference_sum_equals_six_l2442_244267


namespace base_with_six_digits_for_256_l2442_244227

theorem base_with_six_digits_for_256 :
  ∃! (b : ℕ), b > 0 ∧ b^5 ≤ 256 ∧ 256 < b^6 :=
by
  sorry

end base_with_six_digits_for_256_l2442_244227


namespace modified_tic_tac_toe_tie_probability_l2442_244249

theorem modified_tic_tac_toe_tie_probability 
  (amy_win_prob : ℚ) 
  (lily_win_prob : ℚ) 
  (h1 : amy_win_prob = 2/5) 
  (h2 : lily_win_prob = 1/4) 
  (h3 : amy_win_prob ≥ 2 * lily_win_prob ∨ lily_win_prob ≥ 2 * amy_win_prob) : 
  1 - (amy_win_prob + lily_win_prob) = 7/20 :=
by sorry

end modified_tic_tac_toe_tie_probability_l2442_244249


namespace unique_arrangement_l2442_244280

-- Define the shapes and colors
inductive Shape : Type
| Triangle : Shape
| Circle : Shape
| Rectangle : Shape
| Rhombus : Shape

inductive Color : Type
| Red : Color
| Blue : Color
| Yellow : Color
| Green : Color

-- Define the position type
inductive Position : Type
| First : Position
| Second : Position
| Third : Position
| Fourth : Position

-- Define the figure type
structure Figure :=
(shape : Shape)
(color : Color)
(position : Position)

def Arrangement := List Figure

-- Define the conditions
def redBetweenBlueAndGreen (arr : Arrangement) : Prop := sorry
def rhombusRightOfYellow (arr : Arrangement) : Prop := sorry
def circleRightOfTriangleAndRhombus (arr : Arrangement) : Prop := sorry
def triangleNotAtEdge (arr : Arrangement) : Prop := sorry
def blueAndYellowNotAdjacent (arr : Arrangement) : Prop := sorry

-- Define the correct arrangement
def correctArrangement : Arrangement := [
  ⟨Shape.Rectangle, Color.Yellow, Position.First⟩,
  ⟨Shape.Rhombus, Color.Green, Position.Second⟩,
  ⟨Shape.Triangle, Color.Red, Position.Third⟩,
  ⟨Shape.Circle, Color.Blue, Position.Fourth⟩
]

-- Theorem statement
theorem unique_arrangement :
  ∀ (arr : Arrangement),
    (redBetweenBlueAndGreen arr) →
    (rhombusRightOfYellow arr) →
    (circleRightOfTriangleAndRhombus arr) →
    (triangleNotAtEdge arr) →
    (blueAndYellowNotAdjacent arr) →
    (arr = correctArrangement) :=
by sorry

end unique_arrangement_l2442_244280


namespace million_is_ten_to_six_roundness_of_million_l2442_244298

/-- Roundness of a positive integer is the sum of the exponents in its prime factorization. -/
def roundness (n : ℕ) : ℕ := sorry

/-- 1,000,000 can be expressed as 10^6 -/
theorem million_is_ten_to_six : 1000000 = 10^6 := by sorry

/-- The roundness of 1,000,000 is 12 -/
theorem roundness_of_million : roundness 1000000 = 12 := by sorry

end million_is_ten_to_six_roundness_of_million_l2442_244298


namespace pyramid_base_edge_length_l2442_244203

theorem pyramid_base_edge_length 
  (r : ℝ) 
  (h : ℝ) 
  (hemisphere_radius : r = 3) 
  (pyramid_height : h = 4) 
  (hemisphere_tangent : True)  -- This represents the tangency condition
  : ∃ (s : ℝ), s = (12 * Real.sqrt 14) / 7 ∧ s > 0 := by
  sorry

end pyramid_base_edge_length_l2442_244203


namespace intersection_of_M_and_N_l2442_244275

def M : Set ℝ := {x | x < (1 : ℝ) / 2}
def N : Set ℝ := {x | x ≥ -4}

theorem intersection_of_M_and_N : M ∩ N = {x | -4 ≤ x ∧ x < (1 : ℝ) / 2} := by
  sorry

end intersection_of_M_and_N_l2442_244275


namespace min_difference_for_equal_f_values_l2442_244262

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then Real.log x else (1/2) * x + (1/2)

theorem min_difference_for_equal_f_values :
  ∃ (min_diff : ℝ),
    min_diff = 3 - 2 * Real.log 2 ∧
    ∀ (m n : ℝ), m < n → f m = f n → n - m ≥ min_diff :=
by sorry

end min_difference_for_equal_f_values_l2442_244262


namespace triangle_angle_measure_l2442_244204

theorem triangle_angle_measure (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  A + B + C = 180 →
  C = 2 * B →
  B = A / 3 →
  A = 90 := by sorry

end triangle_angle_measure_l2442_244204


namespace children_left_on_bus_l2442_244229

theorem children_left_on_bus (initial_children : ℕ) (difference : ℕ) : 
  initial_children = 41 →
  difference = 23 →
  initial_children - difference = 18 :=
by sorry

end children_left_on_bus_l2442_244229


namespace smallest_three_digit_number_l2442_244259

def digits : Finset Nat := {3, 0, 2, 5, 7}

def isValidNumber (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ 
  (Finset.card (Finset.filter (λ d => d ∈ digits) (Finset.image (λ i => (n / (10^i)) % 10) {0, 1, 2})) = 3)

def smallestValidNumber : Nat := 203

theorem smallest_three_digit_number :
  (isValidNumber smallestValidNumber) ∧
  (∀ n : Nat, isValidNumber n → n ≥ smallestValidNumber) :=
by sorry

end smallest_three_digit_number_l2442_244259


namespace exponential_growth_dominates_power_growth_l2442_244214

theorem exponential_growth_dominates_power_growth 
  (a : ℝ) (α : ℝ) (ha : a > 1) (hα : α > 0) :
  ∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > x₀, 
    (deriv (fun x => a^x) x) / a^x > (deriv (fun x => x^α) x) / x^α :=
sorry

end exponential_growth_dominates_power_growth_l2442_244214


namespace luncheon_cost_l2442_244221

/-- Given two luncheon bills, prove the cost of one sandwich, one coffee, and one pie --/
theorem luncheon_cost (s c p : ℚ) : 
  (5 * s + 8 * c + 2 * p = 510/100) →
  (6 * s + 11 * c + 2 * p = 645/100) →
  (s + c + p = 135/100) := by
  sorry

#check luncheon_cost

end luncheon_cost_l2442_244221


namespace equivalent_expression_l2442_244235

theorem equivalent_expression (x : ℝ) (hx : x < 0) :
  Real.sqrt ((x + 1) / (1 - (x - 2) / x)) = Complex.I * Real.sqrt (-((x^2 + x) / 2)) :=
by sorry

end equivalent_expression_l2442_244235


namespace fred_red_marbles_l2442_244216

/-- Represents the number of marbles of each color --/
structure MarbleCount where
  red : ℕ
  green : ℕ
  blue : ℕ

/-- The conditions of Fred's marble collection --/
def fredMarbles (m : MarbleCount) : Prop :=
  m.red + m.green + m.blue = 63 ∧
  m.blue = 6 ∧
  m.green = m.red / 2

/-- Theorem stating that Fred has 38 red marbles --/
theorem fred_red_marbles :
  ∃ m : MarbleCount, fredMarbles m ∧ m.red = 38 := by
  sorry

end fred_red_marbles_l2442_244216


namespace parallel_transitivity_l2442_244248

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define parallelism between two lines
def parallel (l1 l2 : Line) : Prop :=
  ∀ (x y : ℝ), l1 x y ↔ l2 x y

-- State the theorem
theorem parallel_transitivity (l1 l2 l3 : Line) :
  parallel l1 l3 → parallel l2 l3 → parallel l1 l2 := by
  sorry

end parallel_transitivity_l2442_244248


namespace sin_2alpha_value_l2442_244220

theorem sin_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : 2 * Real.cos (2*α) = Real.cos (α - π/4)) : 
  Real.sin (2*α) = 7/8 := by
sorry

end sin_2alpha_value_l2442_244220


namespace virginia_eggs_problem_l2442_244256

theorem virginia_eggs_problem (initial_eggs : ℕ) (amy_takes : ℕ) (john_takes : ℕ) (laura_takes : ℕ) 
  (h1 : initial_eggs = 372)
  (h2 : amy_takes = 15)
  (h3 : john_takes = 27)
  (h4 : laura_takes = 63) :
  initial_eggs - amy_takes - john_takes - laura_takes = 267 := by
  sorry

end virginia_eggs_problem_l2442_244256


namespace base_8_6_equality_l2442_244272

/-- Checks if a number is a valid digit in a given base -/
def isValidDigit (digit : ℕ) (base : ℕ) : Prop :=
  digit < base

/-- Converts a two-digit number from a given base to base 10 -/
def toBase10 (c d : ℕ) (base : ℕ) : ℕ :=
  base * c + d

/-- The main theorem stating that 0 is the only number satisfying the conditions -/
theorem base_8_6_equality (n : ℕ) : n > 0 → 
  (∃ (c d : ℕ), isValidDigit c 8 ∧ isValidDigit d 8 ∧ 
   isValidDigit c 6 ∧ isValidDigit d 6 ∧
   n = toBase10 c d 8 ∧ n = toBase10 d c 6) → n = 0 :=
by sorry

end base_8_6_equality_l2442_244272


namespace circle_area_equals_square_perimeter_l2442_244237

theorem circle_area_equals_square_perimeter (side_length : ℝ) (radius : ℝ) : 
  side_length = 25 → 4 * side_length = Real.pi * radius^2 → Real.pi * radius^2 = 100 :=
by sorry

end circle_area_equals_square_perimeter_l2442_244237


namespace females_with_advanced_degrees_l2442_244239

theorem females_with_advanced_degrees 
  (total_employees : ℕ) 
  (total_females : ℕ) 
  (employees_with_advanced_degrees : ℕ) 
  (males_with_college_only : ℕ) 
  (h1 : total_employees = 160) 
  (h2 : total_females = 90) 
  (h3 : employees_with_advanced_degrees = 80) 
  (h4 : males_with_college_only = 40) :
  total_females - (total_employees - employees_with_advanced_degrees - males_with_college_only) = 50 :=
by sorry

end females_with_advanced_degrees_l2442_244239


namespace largest_four_digit_divisible_by_24_l2442_244271

theorem largest_four_digit_divisible_by_24 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 24 ∣ n → n ≤ 9984 :=
by sorry

end largest_four_digit_divisible_by_24_l2442_244271


namespace quadratic_equation_roots_l2442_244236

theorem quadratic_equation_roots : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 4
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

end quadratic_equation_roots_l2442_244236


namespace power_mod_1111_l2442_244242

theorem power_mod_1111 : ∃ n : ℕ, 0 ≤ n ∧ n < 1111 ∧ 2^1110 ≡ n [ZMOD 1111] := by
  use 1024
  sorry

end power_mod_1111_l2442_244242


namespace problem_solution_l2442_244268

theorem problem_solution : (2210 - 2137)^2 + (2137 - 2028)^2 = 64 * 268.90625 := by sorry

end problem_solution_l2442_244268


namespace linear_function_characterization_l2442_244296

/-- A linear function f satisfying f(f(x)) = 16x - 15 -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  (∃ a b : ℝ, ∀ x, f x = a * x + b) ∧
  (∀ x, f (f x) = 16 * x - 15)

/-- The theorem stating that a linear function satisfying f(f(x)) = 16x - 15 
    must be either 4x - 3 or -4x + 5 -/
theorem linear_function_characterization (f : ℝ → ℝ) :
  LinearFunction f → 
  ((∀ x, f x = 4 * x - 3) ∨ (∀ x, f x = -4 * x + 5)) :=
by sorry

end linear_function_characterization_l2442_244296


namespace smallest_start_number_for_2520_divisibility_l2442_244250

theorem smallest_start_number_for_2520_divisibility : 
  ∃ (n : ℕ), n > 0 ∧ n ≤ 10 ∧ 
  (∀ (k : ℕ), n ≤ k ∧ k ≤ 10 → 2520 % k = 0) ∧
  (∀ (m : ℕ), m < n → ∃ (j : ℕ), m < j ∧ j ≤ 10 ∧ 2520 % j ≠ 0) :=
by sorry

end smallest_start_number_for_2520_divisibility_l2442_244250


namespace birthday_candles_sharing_l2442_244247

theorem birthday_candles_sharing (ambika_candles : ℕ) (aniyah_multiplier : ℕ) : 
  ambika_candles = 4 →
  aniyah_multiplier = 6 →
  ((ambika_candles + aniyah_multiplier * ambika_candles) / 2 : ℕ) = 14 :=
by sorry

end birthday_candles_sharing_l2442_244247


namespace speed_above_limit_l2442_244281

/-- Proves that given a travel distance of 150 miles, a travel time of 2 hours,
    and a speed limit of 60 mph, the difference between the average speed
    and the speed limit is 15 mph. -/
theorem speed_above_limit (distance : ℝ) (time : ℝ) (speed_limit : ℝ) :
  distance = 150 ∧ time = 2 ∧ speed_limit = 60 →
  distance / time - speed_limit = 15 := by
  sorry

end speed_above_limit_l2442_244281


namespace tobias_driveways_shoveled_l2442_244291

/-- Calculates the number of driveways shoveled by Tobias given his income and expenses. -/
theorem tobias_driveways_shoveled 
  (original_price : ℚ)
  (discount_rate : ℚ)
  (tax_rate : ℚ)
  (monthly_allowance : ℚ)
  (lawn_fee : ℚ)
  (driveway_fee : ℚ)
  (hourly_wage : ℚ)
  (remaining_money : ℚ)
  (months_saved : ℕ)
  (hours_worked : ℕ)
  (lawns_mowed : ℕ)
  (h1 : original_price = 95)
  (h2 : discount_rate = 1/10)
  (h3 : tax_rate = 1/20)
  (h4 : monthly_allowance = 5)
  (h5 : lawn_fee = 15)
  (h6 : driveway_fee = 7)
  (h7 : hourly_wage = 8)
  (h8 : remaining_money = 15)
  (h9 : months_saved = 3)
  (h10 : hours_worked = 10)
  (h11 : lawns_mowed = 4) :
  ∃ (driveways_shoveled : ℕ), driveways_shoveled = 7 :=
by sorry


end tobias_driveways_shoveled_l2442_244291


namespace number_ratio_problem_l2442_244252

/-- Given three numbers satisfying specific conditions, prove their ratios -/
theorem number_ratio_problem (a b c : ℚ) : 
  a + b + c = 98 → 
  b = 30 → 
  c = (8/5) * b → 
  a / b = 2/3 := by
  sorry

end number_ratio_problem_l2442_244252


namespace point_in_fourth_quadrant_l2442_244264

theorem point_in_fourth_quadrant (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ x y : ℝ, x ≠ y ∧ a * x^2 - x - 1/4 = 0 ∧ a * y^2 - y - 1/4 = 0) :
  (a + 1 > 0) ∧ (-3 - a < 0) := by
  sorry


end point_in_fourth_quadrant_l2442_244264


namespace quadratic_root_value_l2442_244292

theorem quadratic_root_value (m : ℝ) : 
  (1 : ℝ)^2 + (1 : ℝ) - m = 0 → m = 2 := by
  sorry

end quadratic_root_value_l2442_244292


namespace sin_cos_pi_12_l2442_244285

theorem sin_cos_pi_12 : 2 * Real.sin (π / 12) * Real.cos (π / 12) = 1 / 2 := by
  sorry

end sin_cos_pi_12_l2442_244285


namespace adventure_distance_l2442_244283

/-- The square of the distance between two points, where one travels east and the other travels north. -/
def squareDistance (eastDistance : ℝ) (northDistance : ℝ) : ℝ :=
  eastDistance ^ 2 + northDistance ^ 2

/-- Theorem: The square of the distance between two points, where one travels 18 km east
    and the other travels 10 km north, is 424 km². -/
theorem adventure_distance : squareDistance 18 10 = 424 := by
  sorry

end adventure_distance_l2442_244283


namespace merchant_loss_l2442_244245

/-- The total loss incurred by a merchant on a counterfeit transaction -/
def total_loss (purchase_cost : ℕ) (additional_price : ℕ) : ℕ :=
  purchase_cost + additional_price

/-- Theorem stating that under the given conditions, the total loss is 92 yuan -/
theorem merchant_loss :
  let purchase_cost : ℕ := 80
  let additional_price : ℕ := 12
  total_loss purchase_cost additional_price = 92 := by
  sorry

#check merchant_loss

end merchant_loss_l2442_244245


namespace complex_magnitude_one_l2442_244209

theorem complex_magnitude_one (z : ℂ) (h : 11 * z^10 + 10*Complex.I * z^9 + 10*Complex.I * z - 11 = 0) : Complex.abs z = 1 := by
  sorry

end complex_magnitude_one_l2442_244209


namespace unique_triple_prime_l2442_244299

theorem unique_triple_prime (p : ℕ) : 
  (p > 0 ∧ Nat.Prime p ∧ Nat.Prime (p + 4) ∧ Nat.Prime (p + 8)) ↔ p = 3 := by
  sorry

end unique_triple_prime_l2442_244299


namespace jacob_age_jacob_age_proof_l2442_244232

theorem jacob_age : ℕ → Prop :=
  fun j : ℕ =>
    ∃ t : ℕ,
      t = j / 2 ∧  -- Tony's age is half of Jacob's age
      t + 6 = 18 ∧ -- In 6 years, Tony will be 18 years old
      j = 24       -- Jacob's current age is 24

-- The proof of the theorem
theorem jacob_age_proof : ∃ j : ℕ, jacob_age j :=
  sorry

end jacob_age_jacob_age_proof_l2442_244232


namespace total_fish_catch_l2442_244234

def fishing_competition (jackson_daily : ℕ) (jonah_daily : ℕ) (george_catches : List ℕ) 
  (lily_catches : List ℕ) (alex_diff : ℕ) : Prop :=
  george_catches.length = 5 ∧ 
  lily_catches.length = 4 ∧
  ∀ i, i < 5 → List.get? (george_catches) i ≠ none ∧
  ∀ i, i < 4 → List.get? (lily_catches) i ≠ none ∧
  (jackson_daily * 5 + jonah_daily * 5 + george_catches.sum + lily_catches.sum + 
    (george_catches.map (λ x => x - alex_diff)).sum) = 159

theorem total_fish_catch : 
  fishing_competition 6 4 [8, 12, 7, 9, 11] [5, 6, 9, 5] 2 := by
  sorry

end total_fish_catch_l2442_244234


namespace system_solution_l2442_244241

theorem system_solution (x y k : ℝ) : 
  (x + 2*y = 7 + k) → 
  (5*x - y = k) → 
  (y = -x) → 
  (k = -6) := by
sorry

end system_solution_l2442_244241


namespace wage_decrease_percentage_l2442_244208

theorem wage_decrease_percentage (wages_last_week : ℝ) (x : ℝ) : 
  wages_last_week > 0 →
  (0.2 * wages_last_week * (1 - x / 100) = 0.6999999999999999 * (0.2 * wages_last_week)) →
  x = 30 := by
  sorry

end wage_decrease_percentage_l2442_244208


namespace store_constraints_equivalence_l2442_244231

/-- Represents the constraints on product purchases in a store. -/
def StoreConstraints (x : ℝ) : Prop :=
  let productACost : ℝ := 8
  let productBCost : ℝ := 2
  let productBQuantity : ℝ := 2 * x - 4
  let totalItems : ℝ := x + productBQuantity
  let totalCost : ℝ := productACost * x + productBCost * productBQuantity
  (totalItems ≥ 32) ∧ (totalCost ≤ 148)

/-- Theorem stating that the given system of inequalities correctly represents the store constraints. -/
theorem store_constraints_equivalence (x : ℝ) :
  StoreConstraints x ↔ (x + (2 * x - 4) ≥ 32 ∧ 8 * x + 2 * (2 * x - 4) ≤ 148) :=
by sorry

end store_constraints_equivalence_l2442_244231


namespace smallest_frood_number_l2442_244225

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem smallest_frood_number : 
  ∀ n : ℕ, n > 0 → (n < 10 → sum_first_n n ≤ 5 * n) ∧ (sum_first_n 10 > 5 * 10) :=
sorry

end smallest_frood_number_l2442_244225


namespace chicken_increase_l2442_244211

/-- The increase in chickens is the sum of chickens bought on two days -/
theorem chicken_increase (initial : ℕ) (day1 : ℕ) (day2 : ℕ) :
  day1 + day2 = (initial + day1 + day2) - initial :=
by sorry

end chicken_increase_l2442_244211


namespace expression_unbounded_l2442_244293

theorem expression_unbounded (M : ℝ) (hM : M > 0) :
  ∃ x y z : ℝ, -1 < x ∧ x < 1 ∧ -1 < y ∧ y < 1 ∧ -1 < z ∧ z < 1 ∧
    (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)) +
     1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))) > M :=
by sorry

end expression_unbounded_l2442_244293


namespace sum_of_coordinates_equals_eight_l2442_244246

def point_C : ℝ × ℝ := (3, 4)

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def point_D : ℝ × ℝ := reflect_over_y_axis point_C

theorem sum_of_coordinates_equals_eight :
  point_C.1 + point_C.2 + point_D.1 + point_D.2 = 8 := by sorry

end sum_of_coordinates_equals_eight_l2442_244246


namespace exists_tangent_circle_l2442_244279

-- Define the basic geometric objects
structure Point :=
  (x y : ℝ)

structure Line :=
  (a b c : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define the given objects
variable (M : Point)
variable (l : Line)
variable (S : Circle)

-- Define the tangency and passing through relations
def isTangentToLine (c : Circle) (l : Line) : Prop := sorry
def isTangentToCircle (c1 c2 : Circle) : Prop := sorry
def passesThrough (c : Circle) (p : Point) : Prop := sorry

-- Theorem statement
theorem exists_tangent_circle :
  ∃ (Ω : Circle),
    passesThrough Ω M ∧
    isTangentToLine Ω l ∧
    isTangentToCircle Ω S :=
sorry

end exists_tangent_circle_l2442_244279


namespace absolute_value_of_negative_l2442_244238

theorem absolute_value_of_negative (a : ℝ) (h : a < 0) : |a| = -a := by
  sorry

end absolute_value_of_negative_l2442_244238


namespace sin_shift_l2442_244260

theorem sin_shift (x : ℝ) : Real.sin (x + π/3) = Real.sin (x + π/3) := by sorry

end sin_shift_l2442_244260


namespace smallest_winning_number_l2442_244219

theorem smallest_winning_number : ∃ N : ℕ, 
  (N = 46) ∧ 
  (0 ≤ N) ∧ 
  (N ≤ 999) ∧ 
  (9 * N - 80 < 1000) ∧ 
  (27 * N - 240 ≥ 1000) ∧ 
  (∀ k : ℕ, k < N → (9 * k - 80 ≥ 1000 ∨ 27 * k - 240 < 1000)) :=
by sorry

end smallest_winning_number_l2442_244219


namespace unique_positive_zero_implies_negative_a_l2442_244289

/-- The function f(x) = ax³ + 3x² + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 1

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem unique_positive_zero_implies_negative_a :
  ∀ a : ℝ, (∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ > 0) → a < 0 :=
by sorry

end unique_positive_zero_implies_negative_a_l2442_244289


namespace relay_race_second_leg_time_l2442_244207

theorem relay_race_second_leg_time 
  (first_leg_time : ℝ) 
  (average_time : ℝ) 
  (h1 : first_leg_time = 58) 
  (h2 : average_time = 42) : 
  let second_leg_time := 2 * average_time - first_leg_time
  second_leg_time = 26 := by
  sorry

end relay_race_second_leg_time_l2442_244207


namespace percentage_problem_l2442_244266

theorem percentage_problem (P : ℝ) (x : ℝ) (h1 : x = 412.5) 
  (h2 : (P / 100) * x = (1 / 3) * x + 110) : P = 60 := by
  sorry

end percentage_problem_l2442_244266


namespace rent_utilities_percentage_after_raise_l2442_244288

theorem rent_utilities_percentage_after_raise (initial_income : ℝ) 
  (initial_percentage : ℝ) (salary_increase : ℝ) : 
  initial_income = 1000 →
  initial_percentage = 40 →
  salary_increase = 600 →
  let initial_amount := initial_income * (initial_percentage / 100)
  let new_income := initial_income + salary_increase
  (initial_amount / new_income) * 100 = 25 := by
sorry

end rent_utilities_percentage_after_raise_l2442_244288


namespace triangle_base_length_l2442_244297

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 3 → height = 3 → area = (base * height) / 2 → base = 2 :=
by sorry

end triangle_base_length_l2442_244297


namespace simplify_expression_l2442_244254

theorem simplify_expression (x y : ℝ) (h : y = Real.sqrt (x - 2) + Real.sqrt (2 - x) + 2) :
  |y - Real.sqrt 3| - (x - 2 + Real.sqrt 2)^2 = -Real.sqrt 3 := by
  sorry

end simplify_expression_l2442_244254


namespace files_per_folder_l2442_244200

theorem files_per_folder (initial_files : ℝ) (additional_files : ℝ) (num_folders : ℝ) :
  let total_files := initial_files + additional_files
  total_files / num_folders = (initial_files + additional_files) / num_folders :=
by sorry

end files_per_folder_l2442_244200


namespace work_completion_time_l2442_244251

/-- 
Given a group of ladies that can complete a piece of work in 12 days,
prove that a group with twice as many ladies will complete half of the work in 3 days.
-/
theorem work_completion_time (num_ladies : ℕ) (total_work : ℝ) : 
  (num_ladies * 12 : ℝ) * total_work = 12 * total_work →
  ((2 * num_ladies : ℝ) * 3) * (total_work / 2) = 12 * (total_work / 2) :=
by sorry

end work_completion_time_l2442_244251


namespace arithmetic_sequence_difference_l2442_244223

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d > 0 ∧
  (∀ n, a n > 0) ∧
  (a 1 = 1) ∧
  (∀ n, a (n + 1) = a n + d) ∧
  (a 3 * a 11 = (a 4 + 5/2)^2)

/-- Theorem stating the difference between two terms -/
theorem arithmetic_sequence_difference
  (a : ℕ → ℝ) (m n : ℕ) (h : ArithmeticSequence a) (h_diff : m - n = 8) :
  a m - a n = 12 :=
sorry

end arithmetic_sequence_difference_l2442_244223


namespace golden_apples_weight_l2442_244213

theorem golden_apples_weight (kidney_apples : ℕ) (canada_apples : ℕ) (sold_apples : ℕ) (left_apples : ℕ) 
  (h1 : kidney_apples = 23)
  (h2 : canada_apples = 14)
  (h3 : sold_apples = 36)
  (h4 : left_apples = 38) :
  ∃ golden_apples : ℕ, 
    kidney_apples + golden_apples + canada_apples = sold_apples + left_apples ∧ 
    golden_apples = 37 := by
  sorry

end golden_apples_weight_l2442_244213
