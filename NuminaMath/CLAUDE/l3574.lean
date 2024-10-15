import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_enclosure_count_l3574_357464

theorem rectangle_enclosure_count :
  let horizontal_lines : ℕ := 5
  let vertical_lines : ℕ := 5
  let choose_horizontal : ℕ := 2
  let choose_vertical : ℕ := 2
  (Nat.choose horizontal_lines choose_horizontal) * (Nat.choose vertical_lines choose_vertical) = 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_enclosure_count_l3574_357464


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3574_357406

theorem triangle_angle_measure (A B C : Real) (a b c : Real) (S : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (A + B + C = π) →
  (a > 0) → (b > 0) → (c > 0) →
  -- A = 2B
  (A = 2 * B) →
  -- Area S = a²/4
  (S = a^2 / 4) →
  -- Area formula
  (S = (1/2) * b * c * Real.sin A) →
  -- Sine law
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  -- Conclusion: A is either π/2 or π/4
  (A = π/2 ∨ A = π/4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3574_357406


namespace NUMINAMATH_CALUDE_cubic_equation_ratio_l3574_357484

/-- Given a cubic equation ax³ + bx² + cx + d = 0 with roots 1, 2, and 3,
    prove that c/d = -11/6 -/
theorem cubic_equation_ratio (a b c d : ℝ) : 
  (∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) →
  c / d = -11 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_ratio_l3574_357484


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3574_357427

theorem complex_fraction_equality : 
  let numerator := ((5 / 2) ^ 2 / (1 / 2) ^ 3) * (5 / 2) ^ 2
  let denominator := ((5 / 3) ^ 4 * (1 / 2) ^ 2) / (2 / 3) ^ 3
  numerator / denominator = 48 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3574_357427


namespace NUMINAMATH_CALUDE_history_paper_pages_l3574_357494

/-- Calculates the total number of pages in a paper given the number of days and pages per day -/
def total_pages (days : ℕ) (pages_per_day : ℕ) : ℕ :=
  days * pages_per_day

/-- Proves that a paper due in 3 days with 21 pages written per day has 63 pages in total -/
theorem history_paper_pages : total_pages 3 21 = 63 := by
  sorry

end NUMINAMATH_CALUDE_history_paper_pages_l3574_357494


namespace NUMINAMATH_CALUDE_farmer_land_ownership_l3574_357426

/-- Proves that a farmer owns 5000 acres of land given the described land usage -/
theorem farmer_land_ownership : ∀ (total_land : ℝ),
  (0.9 * total_land * 0.1 + 0.9 * total_land * 0.8 + 450 = 0.9 * total_land) →
  total_land = 5000 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_ownership_l3574_357426


namespace NUMINAMATH_CALUDE_volleyball_team_combinations_l3574_357444

theorem volleyball_team_combinations (n : ℕ) (k : ℕ) (h1 : n = 14) (h2 : k = 6) :
  Nat.choose n k = 3003 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_combinations_l3574_357444


namespace NUMINAMATH_CALUDE_min_value_alpha_gamma_l3574_357467

open Complex

theorem min_value_alpha_gamma (f : ℂ → ℂ) (α γ : ℂ) :
  (∀ z, f z = (4 + I) * z^2 + α * z + γ) →
  (f 1).im = 0 →
  (f I).im = 0 →
  ∃ (α₀ γ₀ : ℂ), abs α₀ + abs γ₀ = Real.sqrt 2 ∧ 
    ∀ (α' γ' : ℂ), (∀ z, f z = (4 + I) * z^2 + α' * z + γ') →
      (f 1).im = 0 → (f I).im = 0 → abs α' + abs γ' ≥ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_alpha_gamma_l3574_357467


namespace NUMINAMATH_CALUDE_sum_of_first_n_natural_numbers_l3574_357459

theorem sum_of_first_n_natural_numbers (n : ℕ) : 
  (∃ (k : ℕ), k > 0 ∧ k < 10 ∧ n * (n + 1) / 2 = 111 * k) ↔ n = 36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_n_natural_numbers_l3574_357459


namespace NUMINAMATH_CALUDE_binary_1011_is_11_l3574_357440

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, x) => acc + if x then 2^i else 0) 0

theorem binary_1011_is_11 :
  binary_to_decimal [true, true, false, true] = 11 := by
  sorry

end NUMINAMATH_CALUDE_binary_1011_is_11_l3574_357440


namespace NUMINAMATH_CALUDE_optimal_selection_uses_golden_ratio_l3574_357430

/-- The optimal selection method popularized by Hua Luogeng --/
def OptimalSelectionMethod : Type := Unit

/-- The concept used in the optimal selection method --/
def ConceptUsed : Type := Unit

/-- The golden ratio --/
def GoldenRatio : Type := Unit

/-- The optimal selection method was popularized by Hua Luogeng --/
axiom hua_luogeng_popularized : OptimalSelectionMethod

/-- The concept used in the optimal selection method is the golden ratio --/
theorem optimal_selection_uses_golden_ratio : 
  ConceptUsed = GoldenRatio := by sorry

end NUMINAMATH_CALUDE_optimal_selection_uses_golden_ratio_l3574_357430


namespace NUMINAMATH_CALUDE_gcd_45123_32768_l3574_357424

theorem gcd_45123_32768 : Nat.gcd 45123 32768 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45123_32768_l3574_357424


namespace NUMINAMATH_CALUDE_sum_of_selected_flowerbeds_l3574_357405

/-- The number of seeds in each flowerbed -/
def seeds : Fin 9 → ℕ
  | 0 => 18  -- 1st flowerbed
  | 1 => 22  -- 2nd flowerbed
  | 2 => 30  -- 3rd flowerbed
  | 3 => 2 * seeds 0  -- 4th flowerbed
  | 4 => seeds 2  -- 5th flowerbed
  | 5 => seeds 1 / 2  -- 6th flowerbed
  | 6 => seeds 0  -- 7th flowerbed
  | 7 => seeds 3  -- 8th flowerbed
  | 8 => seeds 2 - 1  -- 9th flowerbed

theorem sum_of_selected_flowerbeds : seeds 0 + seeds 4 + seeds 8 = 77 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_selected_flowerbeds_l3574_357405


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3574_357478

/-- Theorem: For a hyperbola x^2 - y^2/a^2 = 1 with a > 0, if its asymptotes are y = ± 2x, then a = 2 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x^2 - y^2/a^2 = 1 → (y = 2*x ∨ y = -2*x)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3574_357478


namespace NUMINAMATH_CALUDE_complex_multiplication_subtraction_l3574_357486

theorem complex_multiplication_subtraction : ∃ (i : ℂ), i^2 = -1 ∧ (4 - 3*i) * (2 + 5*i) - (6 - 2*i) = 17 + 16*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_subtraction_l3574_357486


namespace NUMINAMATH_CALUDE_general_term_is_2n_l3574_357457

/-- An increasing arithmetic sequence with specific properties -/
def IncreasingArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) > a n) ∧ 
  (∃ d > 0, ∀ n, a (n + 1) = a n + d) ∧
  (a 1 = 2) ∧
  (a 2 ^ 2 = a 5 + 6)

/-- The general term of the sequence is 2n -/
theorem general_term_is_2n (a : ℕ → ℝ) 
    (h : IncreasingArithmeticSequence a) : 
    ∀ n : ℕ, a n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_general_term_is_2n_l3574_357457


namespace NUMINAMATH_CALUDE_point_coordinates_l3574_357469

def is_valid_point (x y : ℝ) : Prop :=
  |y| = 1 ∧ |x| = 2

theorem point_coordinates :
  ∀ x y : ℝ, is_valid_point x y ↔ (x = 2 ∧ y = 1) ∨ (x = 2 ∧ y = -1) ∨ (x = -2 ∧ y = 1) ∨ (x = -2 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l3574_357469


namespace NUMINAMATH_CALUDE_construct_triangle_from_equilateral_vertices_l3574_357492

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is acute-angled -/
def isAcute (t : Triangle) : Prop :=
  sorry

/-- Checks if three points form an equilateral triangle -/
def isEquilateral (A B C : Point) : Prop :=
  sorry

/-- Main theorem: Given an acute-angled triangle A₁B₁C₁, there exists a unique triangle ABC
    such that A₁, B₁, and C₁ are the vertices of equilateral triangles drawn outward
    on the sides BC, CA, and AB respectively -/
theorem construct_triangle_from_equilateral_vertices
  (A₁ B₁ C₁ : Point) (h : isAcute (Triangle.mk A₁ B₁ C₁)) :
  ∃! (ABC : Triangle),
    isEquilateral ABC.B ABC.C A₁ ∧
    isEquilateral ABC.C ABC.A B₁ ∧
    isEquilateral ABC.A ABC.B C₁ :=
  sorry

end NUMINAMATH_CALUDE_construct_triangle_from_equilateral_vertices_l3574_357492


namespace NUMINAMATH_CALUDE_mr_a_loss_l3574_357487

/-- Calculates the total loss for Mr. A in a house transaction --/
def calculate_loss (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : ℝ :=
  let first_sale_price := initial_value * (1 - loss_percent)
  let second_sale_price := first_sale_price * (1 + gain_percent)
  second_sale_price - initial_value

/-- Theorem stating that Mr. A loses $2040 in the house transaction --/
theorem mr_a_loss :
  calculate_loss 12000 0.15 0.20 = 2040 := by sorry

end NUMINAMATH_CALUDE_mr_a_loss_l3574_357487


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3574_357437

theorem polynomial_remainder (f : ℝ → ℝ) (a b c d : ℝ) (h : a ≠ b) :
  (∃ g : ℝ → ℝ, ∀ x, f x = (x - a) * g x + c) →
  (∃ h : ℝ → ℝ, ∀ x, f x = (x - b) * h x + d) →
  ∃ k : ℝ → ℝ, ∀ x, f x = (x - a) * (x - b) * k x + ((c - d) / (a - b)) * x + ((a * d - b * c) / (a - b)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3574_357437


namespace NUMINAMATH_CALUDE_sum_of_factors_l3574_357417

theorem sum_of_factors (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
  r ≠ s ∧ r ≠ t ∧ 
  s ≠ t → 
  (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = -120 →
  p + q + r + s + t = 27 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l3574_357417


namespace NUMINAMATH_CALUDE_pigeonhole_principle_on_sequence_l3574_357439

theorem pigeonhole_principle_on_sequence (n : ℕ) : 
  ∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ 2*n ∧ (i + i) % (2*n) = (j + j) % (2*n) := by
  sorry

end NUMINAMATH_CALUDE_pigeonhole_principle_on_sequence_l3574_357439


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l3574_357432

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l3574_357432


namespace NUMINAMATH_CALUDE_min_moves_to_black_l3574_357416

/-- Represents a chessboard with alternating colors -/
structure Chessboard :=
  (size : Nat)
  (alternating : Bool)

/-- Represents a move on the chessboard -/
structure Move :=
  (top_left : Nat × Nat)
  (bottom_right : Nat × Nat)

/-- Function to apply a move to a chessboard -/
def apply_move (board : Chessboard) (move : Move) : Chessboard := sorry

/-- Function to check if all squares are black -/
def all_black (board : Chessboard) : Bool := sorry

/-- Theorem stating the minimum number of moves required -/
theorem min_moves_to_black (board : Chessboard) :
  board.size = 98 ∧ board.alternating →
  (∃ (moves : List Move), all_black (moves.foldl apply_move board) ∧ moves.length = 98) ∧
  (∀ (moves : List Move), all_black (moves.foldl apply_move board) → moves.length ≥ 98) :=
sorry

end NUMINAMATH_CALUDE_min_moves_to_black_l3574_357416


namespace NUMINAMATH_CALUDE_special_sequence_values_l3574_357485

/-- An increasing sequence of natural numbers satisfying a_{a_k} = 3k for any k. -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n m, n < m → a n < a m) ∧ 
  (∀ k, a (a k) = 3 * k)

theorem special_sequence_values (a : ℕ → ℕ) (h : SpecialSequence a) :
  a 100 = 181 ∧ a 1983 = 3762 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_values_l3574_357485


namespace NUMINAMATH_CALUDE_compute_fraction_power_l3574_357409

theorem compute_fraction_power : 9 * (1/7)^4 = 9/2401 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_power_l3574_357409


namespace NUMINAMATH_CALUDE_savings_calculation_l3574_357445

theorem savings_calculation (total : ℚ) (furniture_fraction : ℚ) (tv_cost : ℚ) : 
  furniture_fraction = 3 / 4 →
  tv_cost = 300 →
  (1 - furniture_fraction) * total = tv_cost →
  total = 1200 := by
sorry

end NUMINAMATH_CALUDE_savings_calculation_l3574_357445


namespace NUMINAMATH_CALUDE_molecular_weight_proof_l3574_357448

/-- The molecular weight of C7H6O2 -/
def molecular_weight_C7H6O2 : ℝ := 122

/-- The number of moles in the given sample -/
def num_moles : ℝ := 9

/-- The total molecular weight of the given sample -/
def total_weight : ℝ := 1098

/-- Theorem: The molecular weight of one mole of C7H6O2 is 122 g/mol -/
theorem molecular_weight_proof :
  molecular_weight_C7H6O2 = total_weight / num_moles :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_proof_l3574_357448


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l3574_357488

/-- Number of ways to distribute indistinguishable balls among distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 84 ways to distribute 6 indistinguishable balls among 4 distinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 84 := by sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l3574_357488


namespace NUMINAMATH_CALUDE_prob_sum_18_three_dice_l3574_357421

-- Define a die as having 6 faces
def die_faces : ℕ := 6

-- Define the number of dice
def num_dice : ℕ := 3

-- Define the target sum
def target_sum : ℕ := 18

-- Define the probability of rolling a specific number on a single die
def single_die_prob : ℚ := 1 / die_faces

-- Statement to prove
theorem prob_sum_18_three_dice : 
  (single_die_prob ^ num_dice : ℚ) = 1 / 216 := by sorry

end NUMINAMATH_CALUDE_prob_sum_18_three_dice_l3574_357421


namespace NUMINAMATH_CALUDE_tank_capacity_l3574_357476

/-- The capacity of a tank with specific inlet and outlet pipe properties -/
theorem tank_capacity
  (outlet_time : ℝ)
  (inlet_rate : ℝ)
  (both_pipes_time : ℝ)
  (h1 : outlet_time = 5)
  (h2 : inlet_rate = 8)
  (h3 : both_pipes_time = 8) :
  ∃ (capacity : ℝ), capacity = 1280 ∧
    capacity / outlet_time - (inlet_rate * 60) = capacity / both_pipes_time :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l3574_357476


namespace NUMINAMATH_CALUDE_number_of_adults_at_play_l3574_357477

/-- The number of adults attending a play, given ticket prices and conditions. -/
theorem number_of_adults_at_play : ℕ :=
  let num_children : ℕ := 7
  let adult_ticket_price : ℕ := 11
  let child_ticket_price : ℕ := 7
  let extra_adult_cost : ℕ := 50
  9

#check number_of_adults_at_play

end NUMINAMATH_CALUDE_number_of_adults_at_play_l3574_357477


namespace NUMINAMATH_CALUDE_max_area_rectangle_l3574_357449

/-- Represents a rectangle with integer dimensions. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The perimeter of the rectangle. -/
def Rectangle.perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The area of the rectangle. -/
def Rectangle.area (r : Rectangle) : ℕ := r.length * r.width

/-- Predicate to check if a number is even. -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Theorem stating the maximum area of a rectangle with given constraints. -/
theorem max_area_rectangle :
  ∀ r : Rectangle,
    r.perimeter = 40 →
    isEven r.length →
    r.area ≤ 100 ∧
    (r.area = 100 ↔ r.length = 10 ∧ r.width = 10) := by
  sorry

#check max_area_rectangle

end NUMINAMATH_CALUDE_max_area_rectangle_l3574_357449


namespace NUMINAMATH_CALUDE_factor_expression_l3574_357404

theorem factor_expression (c : ℝ) : 189 * c^2 + 27 * c - 36 = 9 * (3 * c - 1) * (7 * c + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3574_357404


namespace NUMINAMATH_CALUDE_is_circle_center_l3574_357433

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y - 55 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (3, -1)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center : ∀ (x y : ℝ), 
  circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 65 :=
by sorry

end NUMINAMATH_CALUDE_is_circle_center_l3574_357433


namespace NUMINAMATH_CALUDE_staircase_extension_l3574_357454

/-- Calculates the number of additional toothpicks needed to extend a staircase -/
def additional_toothpicks (initial_steps : ℕ) (final_steps : ℕ) (initial_toothpicks : ℕ) : ℕ :=
  let base_increase := initial_toothpicks / initial_steps + 2
  let num_new_steps := final_steps - initial_steps
  (num_new_steps * (2 * base_increase + (num_new_steps - 1) * 2)) / 2

theorem staircase_extension :
  additional_toothpicks 4 7 28 = 42 :=
by sorry

end NUMINAMATH_CALUDE_staircase_extension_l3574_357454


namespace NUMINAMATH_CALUDE_jose_peanuts_l3574_357431

theorem jose_peanuts (jose_peanuts kenya_peanuts : ℕ) 
  (h1 : kenya_peanuts = 133)
  (h2 : kenya_peanuts = jose_peanuts + 48) : 
  jose_peanuts = 85 := by
  sorry

end NUMINAMATH_CALUDE_jose_peanuts_l3574_357431


namespace NUMINAMATH_CALUDE_smallest_area_is_40_l3574_357462

/-- A rectangle with even side lengths that can be divided into squares and dominoes -/
structure CheckeredRectangle where
  width : Nat
  height : Nat
  has_square : Bool
  has_domino : Bool
  width_even : Even width
  height_even : Even height
  both_types : has_square ∧ has_domino

/-- The area of a CheckeredRectangle -/
def area (r : CheckeredRectangle) : Nat :=
  r.width * r.height

/-- Theorem stating the smallest possible area of a valid CheckeredRectangle is 40 -/
theorem smallest_area_is_40 :
  ∀ r : CheckeredRectangle, area r ≥ 40 ∧ ∃ r' : CheckeredRectangle, area r' = 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_area_is_40_l3574_357462


namespace NUMINAMATH_CALUDE_greatest_b_value_l3574_357480

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 8*x - 15 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 8*5 - 15 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l3574_357480


namespace NUMINAMATH_CALUDE_polynomial_has_three_distinct_integer_roots_l3574_357481

def polynomial (x : ℤ) : ℤ := x^5 + 3*x^4 - 4044118*x^3 - 12132362*x^2 - 12132363*x - 2011^2

theorem polynomial_has_three_distinct_integer_roots :
  ∃ (a b c : ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∀ x : ℤ, polynomial x = 0 ↔ x = a ∨ x = b ∨ x = c) :=
sorry

end NUMINAMATH_CALUDE_polynomial_has_three_distinct_integer_roots_l3574_357481


namespace NUMINAMATH_CALUDE_corrected_mean_l3574_357497

/-- Given a set of observations, calculate the corrected mean after fixing an error in one observation -/
theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n > 0 →
  let original_sum := n * original_mean
  let difference := correct_value - incorrect_value
  let corrected_sum := original_sum + difference
  (corrected_sum / n) = 36.14 →
  n = 50 ∧ original_mean = 36 ∧ incorrect_value = 23 ∧ correct_value = 30 :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_l3574_357497


namespace NUMINAMATH_CALUDE_dubblefud_product_l3574_357489

/-- Represents the number of points for each chip color in the game of Dubblefud -/
structure ChipPoints where
  yellow : ℕ
  blue : ℕ
  green : ℕ

/-- Represents the number of chips for each color in a selection -/
structure ChipSelection where
  yellow : ℕ
  blue : ℕ
  green : ℕ

/-- The theorem statement for the Dubblefud game problem -/
theorem dubblefud_product (points : ChipPoints) (selection : ChipSelection) :
  points.yellow = 2 →
  points.blue = 4 →
  points.green = 5 →
  selection.blue = selection.green →
  selection.yellow = 4 →
  (points.yellow * selection.yellow) *
  (points.blue * selection.blue) *
  (points.green * selection.green) =
  72 * selection.blue :=
by sorry

end NUMINAMATH_CALUDE_dubblefud_product_l3574_357489


namespace NUMINAMATH_CALUDE_rabbit_carrots_l3574_357435

theorem rabbit_carrots : ∀ (rabbit_holes fox_holes : ℕ),
  rabbit_holes = fox_holes + 2 →
  5 * rabbit_holes = 6 * fox_holes →
  5 * rabbit_holes = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_rabbit_carrots_l3574_357435


namespace NUMINAMATH_CALUDE_reflection_of_A_across_x_axis_l3574_357463

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the original point A
def A : Point := (1, -2)

-- Define reflection across x-axis
def reflect_x (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem to prove
theorem reflection_of_A_across_x_axis :
  reflect_x A = (1, 2) := by sorry

end NUMINAMATH_CALUDE_reflection_of_A_across_x_axis_l3574_357463


namespace NUMINAMATH_CALUDE_regular_octagon_diagonal_length_l3574_357402

/-- The length of a diagonal in a regular octagon -/
theorem regular_octagon_diagonal_length :
  ∀ (side_length : ℝ),
  side_length > 0 →
  ∃ (diagonal_length : ℝ),
  diagonal_length = side_length * Real.sqrt (2 + Real.sqrt 2) ∧
  diagonal_length^2 = 2 * side_length^2 + side_length^2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_diagonal_length_l3574_357402


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l3574_357407

theorem triangle_side_calculation (A B C : Real) (a b : Real) : 
  B = π / 6 → -- 30° in radians
  C = 7 * π / 12 → -- 105° in radians
  A = π / 4 → -- 45° in radians (derived from B + C + A = π)
  a = 4 →
  b = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l3574_357407


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3574_357479

theorem trigonometric_inequality (θ : Real) (h : 0 < θ ∧ θ < π/4) :
  3 * Real.cos θ + 1 / Real.sin θ + Real.sqrt 3 * Real.tan θ + 2 * Real.sin θ ≥ 4 * (3 * Real.sqrt 3) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l3574_357479


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3574_357458

theorem average_speed_calculation (d₁ d₂ d₃ v₁ v₂ v₃ : ℝ) 
  (h₁ : d₁ = 30) (h₂ : d₂ = 50) (h₃ : d₃ = 40)
  (h₄ : v₁ = 30) (h₅ : v₂ = 50) (h₆ : v₃ = 60) : 
  (d₁ + d₂ + d₃) / ((d₁ / v₁) + (d₂ / v₂) + (d₃ / v₃)) = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3574_357458


namespace NUMINAMATH_CALUDE_tangent_line_problem_l3574_357422

theorem tangent_line_problem (a : ℝ) : 
  (∃ (m : ℝ), 
    (∀ x y : ℝ, y = x^3 → (y - 0 = m * (x - 1) → (x = 1 ∨ (y = m * (x - 1) ∧ 3 * x^2 = m)))) ∧
    (∀ x y : ℝ, y = a * x^2 + (15/4) * x - 9 → (y - 0 = m * (x - 1) → (x = 1 ∨ (y = m * (x - 1) ∧ 2 * a * x + 15/4 = m)))))
  → a = -25/64 ∨ a = -1 := by
  sorry

#check tangent_line_problem

end NUMINAMATH_CALUDE_tangent_line_problem_l3574_357422


namespace NUMINAMATH_CALUDE_factors_of_60_l3574_357473

/-- The number of positive factors of 60 -/
def num_factors_60 : ℕ := sorry

/-- Theorem stating that the number of positive factors of 60 is 12 -/
theorem factors_of_60 : num_factors_60 = 12 := by sorry

end NUMINAMATH_CALUDE_factors_of_60_l3574_357473


namespace NUMINAMATH_CALUDE_smallest_n_for_shared_vertex_triangles_l3574_357412

/-- A two-coloring of a complete graph -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Fin 2

/-- Predicate for a monochromatic triangle in a two-colored complete graph -/
def MonochromaticTriangle (n : ℕ) (c : TwoColoring n) (v₁ v₂ v₃ : Fin n) : Prop :=
  v₁ ≠ v₂ ∧ v₂ ≠ v₃ ∧ v₁ ≠ v₃ ∧
  c v₁ v₂ = c v₂ v₃ ∧ c v₂ v₃ = c v₁ v₃

/-- Predicate for two monochromatic triangles sharing exactly one vertex -/
def SharedVertexTriangles (n : ℕ) (c : TwoColoring n) : Prop :=
  ∃ (v₁ v₂ v₃ v₄ v₅ : Fin n),
    MonochromaticTriangle n c v₁ v₂ v₃ ∧
    MonochromaticTriangle n c v₁ v₄ v₅ ∧
    v₂ ≠ v₄ ∧ v₂ ≠ v₅ ∧ v₃ ≠ v₄ ∧ v₃ ≠ v₅

/-- The main theorem: 9 is the smallest n such that any two-coloring of K_n contains two monochromatic triangles sharing exactly one vertex -/
theorem smallest_n_for_shared_vertex_triangles :
  (∀ c : TwoColoring 9, SharedVertexTriangles 9 c) ∧
  (∀ m : ℕ, m < 9 → ∃ c : TwoColoring m, ¬SharedVertexTriangles m c) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_shared_vertex_triangles_l3574_357412


namespace NUMINAMATH_CALUDE_set_union_problem_l3574_357471

theorem set_union_problem (A B : Set ℕ) (m : ℕ) :
  A = {1, 2, 3, 4} →
  B = {m, 4, 7, 8} →
  A ∩ B = {1, 4} →
  A ∪ B = {1, 2, 3, 4, 7, 8} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l3574_357471


namespace NUMINAMATH_CALUDE_pentagon_perimeter_l3574_357465

/-- Given a pentagon ABCDE where:
  - ΔABE, ΔBCE, and ΔCDE are right-angled triangles
  - ∠AEB = 45°
  - ∠BEC = 60°
  - ∠CED = 45°
  - AE = 40
Prove that the perimeter of pentagon ABCDE is 140 + (40√3)/3 -/
theorem pentagon_perimeter (A B C D E : ℝ × ℝ) : 
  let angle (p q r : ℝ × ℝ) := Real.arccos ((p.1 - q.1) * (r.1 - q.1) + (p.2 - q.2) * (r.2 - q.2)) / 
    (((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt * ((r.1 - q.1)^2 + (r.2 - q.2)^2).sqrt)
  let dist (p q : ℝ × ℝ) := ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt
  let perimeter := dist A B + dist B C + dist C D + dist D E + dist E A
  angle A E B = π/4 ∧ 
  angle B E C = π/3 ∧ 
  angle C E D = π/4 ∧
  angle B A E = π/2 ∧
  angle C B E = π/2 ∧
  angle D C E = π/2 ∧
  dist A E = 40 →
  perimeter = 140 + 40 * Real.sqrt 3 / 3 := by
sorry


end NUMINAMATH_CALUDE_pentagon_perimeter_l3574_357465


namespace NUMINAMATH_CALUDE_line_properties_l3574_357455

-- Define the lines
def l1 (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 1 = 0
def l2 (a x y : ℝ) : Prop := a * x + y = 1

-- Define perpendicularity
def perpendicular (a : ℝ) : Prop := Real.sqrt 3 * a + 1 = 0

-- Define angle of inclination
def angle_of_inclination (θ : ℝ) : Prop := θ = 2 * Real.pi / 3

-- Define distance from origin to line
def distance_to_origin (a : ℝ) (d : ℝ) : Prop := 
  d = 1 / Real.sqrt (a^2 + 1)

theorem line_properties (a : ℝ) :
  perpendicular a →
  angle_of_inclination (Real.arctan (-Real.sqrt 3)) ∧
  distance_to_origin a (Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l3574_357455


namespace NUMINAMATH_CALUDE_triangle_inequality_l3574_357420

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 * (-a + b + c) + b^2 * (a - b + c) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3574_357420


namespace NUMINAMATH_CALUDE_tangent_line_proof_l3574_357474

noncomputable def f (x : ℝ) : ℝ := -(1/2) * x + Real.log x

theorem tangent_line_proof :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ 
  f x₀ = (1/2) * x₀ - 1 ∧
  deriv f x₀ = 1/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_proof_l3574_357474


namespace NUMINAMATH_CALUDE_max_value_product_max_value_achieved_l3574_357446

theorem max_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_two : a + b + c = 2) : 
  a^2 * b^3 * c^2 ≤ 128/2187 := by
sorry

theorem max_value_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 2 ∧ 
  a^2 * b^3 * c^2 > 128/2187 - ε := by
sorry

end NUMINAMATH_CALUDE_max_value_product_max_value_achieved_l3574_357446


namespace NUMINAMATH_CALUDE_factorization_equality_l3574_357472

theorem factorization_equality (a y : ℝ) : a^2 * y - 4 * y = y * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3574_357472


namespace NUMINAMATH_CALUDE_score_three_points_count_l3574_357413

/-- Represents the number of items to be matched -/
def n : ℕ := 4

/-- Represents the number of points awarded for a correct match -/
def correct_points : ℕ := 3

/-- Represents the number of points awarded for an incorrect match -/
def incorrect_points : ℕ := 0

/-- The total number of ways to match exactly one item correctly and the rest incorrectly -/
def ways_to_score_three_points : ℕ := n

theorem score_three_points_count :
  ways_to_score_three_points = n := by
  sorry

end NUMINAMATH_CALUDE_score_three_points_count_l3574_357413


namespace NUMINAMATH_CALUDE_sue_votes_l3574_357410

/-- Given 1000 total votes and Sue receiving 35% of the votes, prove that Sue received 350 votes. -/
theorem sue_votes (total_votes : ℕ) (sue_percentage : ℚ) (h1 : total_votes = 1000) (h2 : sue_percentage = 35 / 100) :
  ↑total_votes * sue_percentage = 350 := by
  sorry

end NUMINAMATH_CALUDE_sue_votes_l3574_357410


namespace NUMINAMATH_CALUDE_solve_for_b_l3574_357447

theorem solve_for_b (b : ℚ) (h : b + b/4 = 10/4) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l3574_357447


namespace NUMINAMATH_CALUDE_toms_restaurant_bill_l3574_357491

/-- The total bill for a group at Tom's Restaurant -/
def total_bill (adults children meal_cost : ℕ) : ℕ :=
  (adults + children) * meal_cost

/-- Theorem: The bill for 2 adults and 5 children with $8 meals is $56 -/
theorem toms_restaurant_bill : total_bill 2 5 8 = 56 := by
  sorry

end NUMINAMATH_CALUDE_toms_restaurant_bill_l3574_357491


namespace NUMINAMATH_CALUDE_schur_inequality_special_case_l3574_357493

theorem schur_inequality_special_case (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b - c) * (b + c - a) * (c + a - b) ≤ a * b * c := by
  sorry

end NUMINAMATH_CALUDE_schur_inequality_special_case_l3574_357493


namespace NUMINAMATH_CALUDE_claire_cakes_l3574_357498

/-- The number of cakes Claire can make -/
def num_cakes (packages_per_cake : ℕ) (price_per_package : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent / price_per_package) / packages_per_cake

theorem claire_cakes : num_cakes 2 3 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_claire_cakes_l3574_357498


namespace NUMINAMATH_CALUDE_toothpicks_for_ten_base_triangles_l3574_357452

/-- The number of toothpicks needed to construct a large equilateral triangle -/
def toothpicks_needed (base_triangles : ℕ) : ℕ :=
  let total_triangles := base_triangles * (base_triangles + 1) / 2
  let total_sides := 3 * total_triangles
  let shared_sides := (total_sides - 3 * base_triangles) / 2
  let boundary_sides := 3 * base_triangles
  shared_sides + boundary_sides

/-- Theorem stating that 98 toothpicks are needed for a large equilateral triangle with 10 small triangles on its base -/
theorem toothpicks_for_ten_base_triangles :
  toothpicks_needed 10 = 98 := by
  sorry


end NUMINAMATH_CALUDE_toothpicks_for_ten_base_triangles_l3574_357452


namespace NUMINAMATH_CALUDE_second_week_rainfall_l3574_357443

/-- Rainfall during the first two weeks of January in Springdale -/
def total_rainfall : ℝ := 20

/-- Ratio of second week's rainfall to first week's rainfall -/
def rainfall_ratio : ℝ := 1.5

/-- Theorem: The rainfall during the second week was 12 inches -/
theorem second_week_rainfall : 
  ∃ (first_week second_week : ℝ),
    first_week + second_week = total_rainfall ∧
    second_week = rainfall_ratio * first_week ∧
    second_week = 12 := by
  sorry

end NUMINAMATH_CALUDE_second_week_rainfall_l3574_357443


namespace NUMINAMATH_CALUDE_periodic_function_theorem_l3574_357499

/-- A function f is periodic with period b if f(x + b) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, f (x + b) = f x

/-- The functional equation property for f -/
def HasFunctionalEquation (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (x + a) = 1/2 + Real.sqrt (f x - (f x)^2)

theorem periodic_function_theorem (f : ℝ → ℝ) (a : ℝ) (ha : a > 0) 
    (h : HasFunctionalEquation f a) : 
    IsPeriodic f (2 * a) := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_theorem_l3574_357499


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3574_357495

theorem algebraic_expression_value (a b : ℝ) (h : a - b = 2) :
  a^3 - 2*a^2*b + a*b^2 - 4*a = 0 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3574_357495


namespace NUMINAMATH_CALUDE_bridge_length_l3574_357453

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = 255 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * crossing_time :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l3574_357453


namespace NUMINAMATH_CALUDE_choose_four_from_nine_l3574_357460

theorem choose_four_from_nine :
  Nat.choose 9 4 = 126 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_nine_l3574_357460


namespace NUMINAMATH_CALUDE_inverse_of_A_squared_l3574_357415

def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, -1; 2, 1]

theorem inverse_of_A_squared :
  A⁻¹ = !![3, -1; 2, 1] →
  (A^2)⁻¹ = !![7, -4; 8, -1] := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_squared_l3574_357415


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l3574_357403

theorem inscribed_quadrilateral_fourth_side 
  (r : ℝ) 
  (a b c d : ℝ) 
  (θ : ℝ) 
  (h1 : r = 150 * Real.sqrt 2)
  (h2 : a = 150 ∧ b = 150 ∧ c = 150)
  (h3 : θ = 120 * π / 180) : 
  d = 375 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l3574_357403


namespace NUMINAMATH_CALUDE_florist_roses_count_l3574_357466

/-- Calculates the total number of roses after picking two batches -/
def total_roses (initial : Float) (batch1 : Float) (batch2 : Float) : Float :=
  initial + batch1 + batch2

/-- Theorem stating that given the specific numbers from the problem, 
    the total number of roses is 72.0 -/
theorem florist_roses_count : 
  total_roses 37.0 16.0 19.0 = 72.0 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_count_l3574_357466


namespace NUMINAMATH_CALUDE_train_optimization_l3574_357470

/-- Represents the relationship between carriages and round trips -/
def round_trips (x : ℝ) : ℝ := -2 * x + 24

/-- Represents the total number of passengers transported per day -/
def passengers (x : ℝ) : ℝ := 110 * x * round_trips x

/-- The optimal number of carriages -/
def optimal_carriages : ℝ := 6

/-- The optimal number of round trips -/
def optimal_trips : ℝ := round_trips optimal_carriages

/-- The maximum number of passengers per day -/
def max_passengers : ℝ := passengers optimal_carriages

theorem train_optimization :
  (round_trips 4 = 16) →
  (round_trips 7 = 10) →
  (∀ x, round_trips x = -2 * x + 24) →
  (optimal_carriages = 6) →
  (optimal_trips = 12) →
  (max_passengers = 7920) →
  (∀ x, passengers x ≤ max_passengers) :=
by sorry

end NUMINAMATH_CALUDE_train_optimization_l3574_357470


namespace NUMINAMATH_CALUDE_dartboard_angle_l3574_357496

theorem dartboard_angle (P : ℝ) (θ : ℝ) : 
  P = 1/8 → θ = P * 360 → θ = 45 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_angle_l3574_357496


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3574_357423

/-- An arithmetic sequence satisfying a specific condition -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ d : ℝ, ∀ k : ℕ, a (k + 1) = a k + d

/-- The specific condition for the sequence -/
def SequenceCondition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) + a n = 4 * n

theorem arithmetic_sequence_first_term
  (a : ℕ → ℝ)
  (h1 : ArithmeticSequence a)
  (h2 : SequenceCondition a) :
  a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3574_357423


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l3574_357418

/-- The number of unique arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 120

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of 'A's in "BANANA" -/
def num_a : ℕ := 3

/-- Theorem stating that the number of unique arrangements of "BANANA" is correct -/
theorem banana_arrangement_count :
  banana_arrangements = (total_letters.factorial) / (num_a.factorial) :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l3574_357418


namespace NUMINAMATH_CALUDE_basketball_weight_l3574_357450

theorem basketball_weight (basketball_weight bicycle_weight : ℝ) 
  (h1 : 9 * basketball_weight = 6 * bicycle_weight)
  (h2 : 4 * bicycle_weight = 120) : 
  basketball_weight = 20 := by
sorry

end NUMINAMATH_CALUDE_basketball_weight_l3574_357450


namespace NUMINAMATH_CALUDE_max_jogs_is_six_l3574_357425

/-- Represents the quantity of each item Bill can buy --/
structure Purchase where
  jags : Nat
  jigs : Nat
  jogs : Nat

/-- Calculates the total cost of a purchase --/
def totalCost (p : Purchase) : Nat :=
  p.jags * 1 + p.jigs * 2 + p.jogs * 7

/-- Checks if a purchase satisfies all conditions --/
def isValidPurchase (p : Purchase) : Prop :=
  p.jags ≥ 1 ∧ p.jigs ≥ 1 ∧ p.jogs ≥ 1 ∧ totalCost p = 50

/-- Theorem stating that the maximum number of jogs Bill can buy is 6 --/
theorem max_jogs_is_six :
  (∃ p : Purchase, isValidPurchase p ∧ p.jogs = 6) ∧
  (∀ p : Purchase, isValidPurchase p → p.jogs ≤ 6) :=
sorry

end NUMINAMATH_CALUDE_max_jogs_is_six_l3574_357425


namespace NUMINAMATH_CALUDE_equation_solution_l3574_357456

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (12 + 3*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 ∧ 
  x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3574_357456


namespace NUMINAMATH_CALUDE_solution_difference_l3574_357461

theorem solution_difference (r s : ℝ) : 
  ((6 * r - 18) / (r^2 + 3*r - 18) = r + 3) →
  ((6 * s - 18) / (s^2 + 3*s - 18) = s + 3) →
  r ≠ s →
  r > s →
  r - s = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l3574_357461


namespace NUMINAMATH_CALUDE_sum_last_two_digits_fibonacci_factorial_series_l3574_357436

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem sum_last_two_digits_fibonacci_factorial_series :
  (fibonacci_factorial_series.map (λ n => last_two_digits (factorial n))).sum = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_last_two_digits_fibonacci_factorial_series_l3574_357436


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3574_357483

-- Define the condition |a-1| + |a| ≤ 1
def condition (a : ℝ) : Prop := abs (a - 1) + abs a ≤ 1

-- Define the property that y = a^x is decreasing on ℝ
def is_decreasing (a : ℝ) : Prop := ∀ x y : ℝ, x < y → a^x > a^y

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, is_decreasing a → condition a) ∧
  (∃ a : ℝ, condition a ∧ ¬is_decreasing a) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3574_357483


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l3574_357490

/-- The minimum number of additional coins needed for unique distribution -/
def min_additional_coins (friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (friends * (friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum additional coins needed for Alex's distribution -/
theorem alex_coin_distribution (friends : ℕ) (initial_coins : ℕ) 
  (h1 : friends = 15) (h2 : initial_coins = 94) :
  min_additional_coins friends initial_coins = 26 := by
  sorry

#eval min_additional_coins 15 94

end NUMINAMATH_CALUDE_alex_coin_distribution_l3574_357490


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l3574_357434

theorem smallest_positive_integer_with_remainders : ∃ (b : ℕ), b > 0 ∧
  b % 3 = 2 ∧ b % 5 = 3 ∧ 
  ∀ (x : ℕ), x > 0 ∧ x % 3 = 2 ∧ x % 5 = 3 → b ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l3574_357434


namespace NUMINAMATH_CALUDE_min_sum_distances_to_lines_l3574_357411

/-- The minimum sum of distances from a point on the parabola y² = 4x to two specific lines -/
theorem min_sum_distances_to_lines : 
  let parabola := {P : ℝ × ℝ | P.2^2 = 4 * P.1}
  let line1 := {P : ℝ × ℝ | 4 * P.1 - 3 * P.2 + 6 = 0}
  let line2 := {P : ℝ × ℝ | P.1 = -1}
  let dist_to_line1 (P : ℝ × ℝ) := |4 * P.1 - 3 * P.2 + 6| / Real.sqrt (4^2 + (-3)^2)
  let dist_to_line2 (P : ℝ × ℝ) := |P.1 + 1|
  ∃ (min_dist : ℝ), min_dist = 2 ∧ 
    ∀ (P : ℝ × ℝ), P ∈ parabola → 
      dist_to_line1 P + dist_to_line2 P ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_sum_distances_to_lines_l3574_357411


namespace NUMINAMATH_CALUDE_grandfather_money_calculation_l3574_357408

def birthday_money_problem (aunt_money grandfather_money total_money bank_money : ℕ) : Prop :=
  aunt_money = 75 ∧
  bank_money = 45 ∧
  bank_money = total_money / 5 ∧
  total_money = aunt_money + grandfather_money

theorem grandfather_money_calculation :
  ∀ aunt_money grandfather_money total_money bank_money,
  birthday_money_problem aunt_money grandfather_money total_money bank_money →
  grandfather_money = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_grandfather_money_calculation_l3574_357408


namespace NUMINAMATH_CALUDE_surrounding_circles_radius_l3574_357442

/-- The radius of the central circle -/
def central_radius : ℝ := 2

/-- The number of surrounding circles -/
def num_surrounding_circles : ℕ := 4

/-- Predicate that checks if all circles are touching each other -/
def circles_touching (r : ℝ) : Prop :=
  ∃ (centers : Fin num_surrounding_circles → ℝ × ℝ),
    ∀ (i j : Fin num_surrounding_circles),
      i ≠ j → ‖centers i - centers j‖ = 2 * r ∧
    ∀ (i : Fin num_surrounding_circles),
      ‖centers i‖ = central_radius + r

/-- Theorem stating that the radius of surrounding circles is 2 -/
theorem surrounding_circles_radius :
  ∃ (r : ℝ), r > 0 ∧ circles_touching r → r = 2 :=
sorry

end NUMINAMATH_CALUDE_surrounding_circles_radius_l3574_357442


namespace NUMINAMATH_CALUDE_absolute_value_expression_l3574_357451

theorem absolute_value_expression (x : ℝ) (h : x < 0) : 
  |x - 3 * Real.sqrt ((x - 2)^2)| = 6 - 4*x := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l3574_357451


namespace NUMINAMATH_CALUDE_proposition_logic_l3574_357475

theorem proposition_logic (p q : Prop) :
  (p ∨ q) → ¬p → (¬p ∧ q) := by sorry

end NUMINAMATH_CALUDE_proposition_logic_l3574_357475


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l3574_357400

/-- The sum of the intercepts of the line 2x - 3y + 6 = 0 on the coordinate axes is -1 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (2 * x - 3 * y + 6 = 0) → 
  (∃ x_intercept y_intercept : ℝ, 
    (2 * x_intercept + 6 = 0) ∧ 
    (-3 * y_intercept + 6 = 0) ∧ 
    (x_intercept + y_intercept = -1)) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l3574_357400


namespace NUMINAMATH_CALUDE_xiao_ying_performance_l3574_357438

def regular_weight : ℝ := 0.20
def midterm_weight : ℝ := 0.30
def final_weight : ℝ := 0.50
def regular_score : ℝ := 85
def midterm_score : ℝ := 90
def final_score : ℝ := 92

def semester_performance : ℝ :=
  regular_weight * regular_score +
  midterm_weight * midterm_score +
  final_weight * final_score

theorem xiao_ying_performance :
  semester_performance = 90 := by sorry

end NUMINAMATH_CALUDE_xiao_ying_performance_l3574_357438


namespace NUMINAMATH_CALUDE_parabola_properties_line_parabola_intersection_l3574_357419

/-- Parabola C: y^2 = -4x -/
def parabola (x y : ℝ) : Prop := y^2 = -4*x

/-- Line l: y = kx - k + 2, passing through (1, 2) -/
def line (k x y : ℝ) : Prop := y = k*x - k + 2

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (-1, 0)

/-- Directrix of the parabola -/
def directrix (x : ℝ) : Prop := x = 1

/-- Distance from focus to directrix -/
def focus_directrix_distance : ℝ := 2

/-- Theorem about the parabola and its properties -/
theorem parabola_properties :
  (∀ x y, parabola x y → (focus.1 = -1 ∧ focus.2 = 0)) ∧
  (∀ x, directrix x ↔ x = 1) ∧
  focus_directrix_distance = 2 :=
sorry

/-- Theorem about the intersection of the line and parabola -/
theorem line_parabola_intersection (k : ℝ) :
  (∀ x y, parabola x y ∧ line k x y →
    (k = 0 ∨ k = 1 + Real.sqrt 2 ∨ k = 1 - Real.sqrt 2) ↔
      (∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ line k p.1 p.2)) ∧
  ((1 - Real.sqrt 2 < k ∧ k < 1 + Real.sqrt 2) ↔
    (∃ p q : ℝ × ℝ, p ≠ q ∧ parabola p.1 p.2 ∧ parabola q.1 q.2 ∧ line k p.1 p.2 ∧ line k q.1 q.2)) ∧
  (k > 1 + Real.sqrt 2 ∨ k < 1 - Real.sqrt 2) ↔
    (∀ x y, ¬(parabola x y ∧ line k x y)) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_line_parabola_intersection_l3574_357419


namespace NUMINAMATH_CALUDE_green_light_probability_theorem_l3574_357428

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of arriving during the green light -/
def greenLightProbability (d : TrafficLightDuration) : ℚ :=
  d.green / (d.red + d.yellow + d.green)

/-- Theorem stating the probability of arriving during the green light
    for the given traffic light durations -/
theorem green_light_probability_theorem (d : TrafficLightDuration) 
  (h1 : d.red = 30)
  (h2 : d.yellow = 5)
  (h3 : d.green = 40) :
  greenLightProbability d = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_green_light_probability_theorem_l3574_357428


namespace NUMINAMATH_CALUDE_sine_cosine_identity_l3574_357414

theorem sine_cosine_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (170 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_identity_l3574_357414


namespace NUMINAMATH_CALUDE_triangle_area_l3574_357429

/-- Triangle XYZ with given properties has area 35√7/2 -/
theorem triangle_area (X Y Z : Real) (r R : Real) (h1 : r = 3) (h2 : R = 12) 
  (h3 : 3 * Real.cos Y = Real.cos X + Real.cos Z) : 
  ∃ (area : Real), area = (35 * Real.sqrt 7) / 2 ∧ 
  area = r * (Real.sin X * R + Real.sin Y * R + Real.sin Z * R) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3574_357429


namespace NUMINAMATH_CALUDE_twenty_percent_value_l3574_357468

theorem twenty_percent_value (x : ℝ) : 1.2 * x = 600 → 0.2 * x = 100 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_value_l3574_357468


namespace NUMINAMATH_CALUDE_difference_largest_smallest_l3574_357482

def digits : List Nat := [9, 3, 1, 2, 6, 4]

def max_occurrences : Nat := 2

def largest_number : Nat := 99664332211

def smallest_number : Nat := 1122334699

theorem difference_largest_smallest :
  largest_number - smallest_number = 98541997512 :=
by sorry

end NUMINAMATH_CALUDE_difference_largest_smallest_l3574_357482


namespace NUMINAMATH_CALUDE_dirk_amulet_selling_days_l3574_357441

/-- Represents the problem of calculating the number of days Dirk sold amulets. -/
def amulet_problem (amulets_per_day : ℕ) (selling_price : ℚ) (cost_price : ℚ) 
  (faire_cut_percentage : ℚ) (total_profit : ℚ) : Prop :=
  let revenue_per_amulet : ℚ := selling_price
  let profit_per_amulet : ℚ := selling_price - cost_price
  let faire_cut_per_amulet : ℚ := faire_cut_percentage * revenue_per_amulet
  let net_profit_per_amulet : ℚ := profit_per_amulet - faire_cut_per_amulet
  let net_profit_per_day : ℚ := net_profit_per_amulet * amulets_per_day
  let days : ℚ := total_profit / net_profit_per_day
  days = 2

/-- Theorem stating the solution to Dirk's amulet selling problem. -/
theorem dirk_amulet_selling_days : 
  amulet_problem 25 40 30 (1/10) 300 := by
  sorry

end NUMINAMATH_CALUDE_dirk_amulet_selling_days_l3574_357441


namespace NUMINAMATH_CALUDE_train_length_l3574_357401

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 36 * (5/18) → time = 40 → speed * time = 400 := by sorry

end NUMINAMATH_CALUDE_train_length_l3574_357401
