import Mathlib

namespace NUMINAMATH_CALUDE_no_solution_for_seven_power_plus_cube_divisible_by_nine_l1154_115492

theorem no_solution_for_seven_power_plus_cube_divisible_by_nine :
  ∀ n : ℕ, n ≥ 1 → ¬(9 ∣ 7^n + n^3) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_seven_power_plus_cube_divisible_by_nine_l1154_115492


namespace NUMINAMATH_CALUDE_unknown_score_is_66_l1154_115419

def scores : List ℕ := [65, 70, 78, 85, 92]

def is_integer (n : ℚ) : Prop := ∃ m : ℤ, n = m

theorem unknown_score_is_66 (x : ℕ) 
  (h1 : is_integer ((scores.sum + x) / 6))
  (h2 : x % 6 = 0)
  (h3 : x ≥ 60 ∧ x ≤ 100) :
  x = 66 := by sorry

end NUMINAMATH_CALUDE_unknown_score_is_66_l1154_115419


namespace NUMINAMATH_CALUDE_square_area_with_rectangle_division_l1154_115452

theorem square_area_with_rectangle_division (x : ℝ) (h1 : x > 0) : 
  let rectangle_area := 14
  let square_side := 4 * x
  let rectangle_width := x
  let rectangle_length := 3 * x
  rectangle_area = rectangle_width * rectangle_length →
  (square_side)^2 = 224/3 := by
sorry

end NUMINAMATH_CALUDE_square_area_with_rectangle_division_l1154_115452


namespace NUMINAMATH_CALUDE_inequality_proof_l1154_115438

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 2015) :
  (a + b) / (a^2 + b^2) + (b + c) / (b^2 + c^2) + (c + a) / (c^2 + a^2) ≤ (Real.sqrt a + Real.sqrt b + Real.sqrt c) / Real.sqrt 2015 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1154_115438


namespace NUMINAMATH_CALUDE_unique_square_pattern_l1154_115479

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  n.hundreds * 100 + n.tens * 10 + n.ones

/-- Checks if a number satisfies the squaring pattern -/
def satisfiesSquarePattern (n : ThreeDigitNumber) : Prop :=
  let square := n.toNat * n.toNat
  -- Add conditions here that check if the square follows the pattern
  -- This is a placeholder and should be replaced with actual conditions
  true

/-- The main theorem stating that 748 is the only number satisfying the conditions -/
theorem unique_square_pattern : 
  ∃! n : ThreeDigitNumber, satisfiesSquarePattern n ∧ n.toNat = 748 := by
  sorry

#check unique_square_pattern

end NUMINAMATH_CALUDE_unique_square_pattern_l1154_115479


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1154_115436

theorem condition_sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 1 → x^2 + y^2 ≥ 2) ∧
  (∃ x y : ℝ, x^2 + y^2 ≥ 2 ∧ ¬(x ≥ 1 ∧ y ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1154_115436


namespace NUMINAMATH_CALUDE_ratio_equality_l1154_115449

theorem ratio_equality : (240 : ℚ) / 1547 / (2 / 13) = (5 : ℚ) / 34 / (7 / 48) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1154_115449


namespace NUMINAMATH_CALUDE_max_shaded_area_trapezoid_l1154_115497

/-- Given a trapezoid ABCD with bases of length a and b, and area 1,
    the maximum area of the shaded region formed by moving points on the bases is ab / (a+b)^2 -/
theorem max_shaded_area_trapezoid (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let trapezoid_area : ℝ := 1
  ∃ (max_area : ℝ), max_area = a * b / (a + b)^2 ∧
    ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ b →
      (x * y / ((a + b) * (x + y)) + (a - x) * (b - y) / ((a + b) * (a + b - x - y))) / (a + b) ≤ max_area :=
sorry

end NUMINAMATH_CALUDE_max_shaded_area_trapezoid_l1154_115497


namespace NUMINAMATH_CALUDE_smallest_mu_inequality_l1154_115473

theorem smallest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  (∀ μ : ℝ, (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + 4*b^2 + 4*c^2 + d^2 ≥ 2*a*b + μ*b*c + 2*c*d) → μ ≥ 6) ∧
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + 4*b^2 + 4*c^2 + d^2 ≥ 2*a*b + 6*b*c + 2*c*d) :=
by sorry

end NUMINAMATH_CALUDE_smallest_mu_inequality_l1154_115473


namespace NUMINAMATH_CALUDE_work_completion_time_l1154_115481

/-- Represents the number of days it takes for B to complete the entire work -/
def days_for_B (days_for_A days_A_worked days_B_remaining : ℕ) : ℚ :=
  (4 * days_for_A * days_B_remaining) / (3 * days_for_A - 3 * days_A_worked)

theorem work_completion_time :
  days_for_B 40 10 45 = 60 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1154_115481


namespace NUMINAMATH_CALUDE_cube_sum_inverse_l1154_115421

theorem cube_sum_inverse (x R S : ℝ) (hx : x ≠ 0) : 
  x + 1 / x = R → x^3 + 1 / x^3 = S → S = R^3 - 3 * R :=
by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inverse_l1154_115421


namespace NUMINAMATH_CALUDE_circle_tangent_range_l1154_115402

/-- The range of k values for which a circle x²+y²+2x-4y+k-2=0 allows
    two tangents from the point (1, 2) -/
theorem circle_tangent_range : 
  ∀ k : ℝ, 
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + k - 2 = 0 ∧ 
   ∃ (t₁ t₂ : ℝ × ℝ), t₁ ≠ t₂ ∧ 
   (t₁.1 - 1)^2 + (t₁.2 - 2)^2 = (t₂.1 - 1)^2 + (t₂.2 - 2)^2 ∧
   (t₁.1^2 + t₁.2^2 + 2*t₁.1 - 4*t₁.2 + k - 2 = 0) ∧
   (t₂.1^2 + t₂.2^2 + 2*t₂.1 - 4*t₂.2 + k - 2 = 0)) ↔ 
  (3 < k ∧ k < 7) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_range_l1154_115402


namespace NUMINAMATH_CALUDE_power_calculation_l1154_115439

theorem power_calculation : 3^15 * 9^5 / 27^6 = 3^7 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1154_115439


namespace NUMINAMATH_CALUDE_perpendicular_lines_l1154_115476

/-- Two lines ax+y-1=0 and x-y+3=0 are perpendicular if and only if a = 1 -/
theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, (a*x + y = 1 ∧ x - y = -3) → 
   ((-a) * 1 = -1)) ↔ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l1154_115476


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1154_115493

theorem min_value_quadratic (x y : ℝ) (h : x + y = 5) :
  x^2 - x*y + y^2 ≥ 25/4 ∧ ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 5 ∧ x₀^2 - x₀*y₀ + y₀^2 = 25/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1154_115493


namespace NUMINAMATH_CALUDE_equal_earnings_l1154_115450

theorem equal_earnings (t : ℝ) : 
  (t - 4) * (3 * t - 7) = (3 * t - 12) * (t - 3) → t = 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_earnings_l1154_115450


namespace NUMINAMATH_CALUDE_oil_remaining_l1154_115433

theorem oil_remaining (x₁ x₂ x₃ : ℕ) : 
  x₁ > 0 → x₂ > 0 → x₃ > 0 →
  3 * x₁ = 2 * x₂ →
  5 * x₁ = 3 * x₃ →
  30 - (x₁ + x₂ + x₃) = 5 :=
by sorry

end NUMINAMATH_CALUDE_oil_remaining_l1154_115433


namespace NUMINAMATH_CALUDE_number_decrease_proof_l1154_115495

theorem number_decrease_proof (x v : ℝ) : 
  x > 0 → x = 7 → x - v = 21 * (1/x) → v = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_decrease_proof_l1154_115495


namespace NUMINAMATH_CALUDE_parabola_properties_l1154_115486

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem parabola_properties (a b c : ℝ) 
  (h_a_nonzero : a ≠ 0)
  (h_c_gt_3 : c > 3)
  (h_passes_through : parabola a b c 5 = 0)
  (h_symmetry_axis : -b / (2 * a) = 2) :
  (a * b * c < 0) ∧ 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ parabola a b c x₁ = 2 ∧ parabola a b c x₂ = 2) ∧
  (a < -3/5) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l1154_115486


namespace NUMINAMATH_CALUDE_highway_traffic_l1154_115475

/-- The number of vehicles involved in accidents per 100 million vehicles -/
def accident_rate : ℕ := 96

/-- The total number of vehicles involved in accidents last year -/
def total_accidents : ℕ := 2880

/-- The number of vehicles (in billions) that traveled on the highway last year -/
def vehicles_traveled : ℕ := 3

theorem highway_traffic :
  vehicles_traveled * 1000000000 = (total_accidents * 100000000) / accident_rate := by
  sorry

end NUMINAMATH_CALUDE_highway_traffic_l1154_115475


namespace NUMINAMATH_CALUDE_sequence_count_mod_l1154_115456

def sequence_count (n : ℕ) (max : ℕ) : ℕ :=
  let m := Nat.choose (max - n + n) n
  m / 3

theorem sequence_count_mod (n : ℕ) (max : ℕ) : 
  sequence_count n max % 1000 = 662 :=
sorry

#check sequence_count_mod 10 2018

end NUMINAMATH_CALUDE_sequence_count_mod_l1154_115456


namespace NUMINAMATH_CALUDE_function_properties_and_triangle_l1154_115474

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.cos x)^2 + 3

theorem function_properties_and_triangle (A B C : ℝ) (a b c : ℝ) :
  (∀ x, f x ≤ 4) ∧ 
  (∀ ε > 0, ∃ T > 0, T ≤ π ∧ ∀ x, f (x + T) = f x) ∧
  c = Real.sqrt 3 →
  f C = 4 →
  Real.sin A = 2 * Real.sin B →
  a = 2 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_and_triangle_l1154_115474


namespace NUMINAMATH_CALUDE_min_width_proof_l1154_115444

/-- The minimum width of a rectangular area with given constraints -/
def min_width : ℝ := 10

/-- The length of the rectangular area -/
def length (w : ℝ) : ℝ := w + 20

/-- The area of the rectangular area -/
def area (w : ℝ) : ℝ := w * length w

theorem min_width_proof :
  (∀ w : ℝ, w > 0 → area w ≥ 150 → w ≥ min_width) ∧
  (area min_width ≥ 150) ∧
  (min_width > 0) :=
sorry

end NUMINAMATH_CALUDE_min_width_proof_l1154_115444


namespace NUMINAMATH_CALUDE_phoenix_airport_on_time_rate_l1154_115437

def total_flights : ℕ := 8
def late_flights : ℕ := 1
def initial_on_time_flights : ℕ := 3
def subsequent_on_time_flights : ℕ := 4
def target_rate : ℚ := 4/5

def on_time_rate (total : ℕ) (on_time : ℕ) : ℚ :=
  (on_time : ℚ) / (total : ℚ)

theorem phoenix_airport_on_time_rate :
  on_time_rate total_flights (initial_on_time_flights + subsequent_on_time_flights) > target_rate := by
  sorry

end NUMINAMATH_CALUDE_phoenix_airport_on_time_rate_l1154_115437


namespace NUMINAMATH_CALUDE_functional_equation_bound_l1154_115490

/-- Given real-valued functions f and g defined on ℝ satisfying certain conditions,
    prove that |g(y)| ≤ 1 for all y. -/
theorem functional_equation_bound (f g : ℝ → ℝ)
  (h1 : ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y)
  (h2 : ∀ x, f x ≠ 0)
  (h3 : ∀ x, |f x| ≤ 1) :
  ∀ y, |g y| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_bound_l1154_115490


namespace NUMINAMATH_CALUDE_product_diversity_l1154_115451

theorem product_diversity (n k : ℕ+) :
  ∃ (m : ℕ), m = n + k - 2 ∧
  ∀ (A : Finset ℝ) (B : Finset ℝ),
    A.card = k ∧ B.card = n →
    (A.product B).card ≥ m ∧
    ∀ (m' : ℕ), m' > m →
      ∃ (A' : Finset ℝ) (B' : Finset ℝ),
        A'.card = k ∧ B'.card = n ∧ (A'.product B').card < m' :=
by sorry

end NUMINAMATH_CALUDE_product_diversity_l1154_115451


namespace NUMINAMATH_CALUDE_midpoint_vector_equation_l1154_115443

/-- Given two points P₁ and P₂ in ℝ², prove that the point P satisfying 
    the vector equation P₁P - PP₂ = 0 has coordinates (1, 1) -/
theorem midpoint_vector_equation (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (-1, 2) → P₂ = (3, 0) → (P.1 - P₁.1, P.2 - P₁.2) = (P₂.1 - P.1, P₂.2 - P.2) → 
  P = (1, 1) := by
sorry

end NUMINAMATH_CALUDE_midpoint_vector_equation_l1154_115443


namespace NUMINAMATH_CALUDE_evaluate_P_l1154_115459

/-- The polynomial P(x) = x^3 - 6x^2 - 5x + 4 -/
def P (x : ℝ) : ℝ := x^3 - 6*x^2 - 5*x + 4

/-- Theorem stating that under given conditions, P(y) = -22 -/
theorem evaluate_P (y z : ℝ) (h : ∀ n : ℝ, z * P y = P (y - n) + P (y + n)) : P y = -22 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_P_l1154_115459


namespace NUMINAMATH_CALUDE_min_additional_teddy_bears_l1154_115469

def teddy_bears : ℕ := 37
def row_size : ℕ := 8

theorem min_additional_teddy_bears :
  let next_multiple := ((teddy_bears + row_size - 1) / row_size) * row_size
  next_multiple - teddy_bears = 3 := by
sorry

end NUMINAMATH_CALUDE_min_additional_teddy_bears_l1154_115469


namespace NUMINAMATH_CALUDE_cube_sum_minus_product_eq_2003_l1154_115477

theorem cube_sum_minus_product_eq_2003 : 
  {(x, y, z) : ℤ × ℤ × ℤ | x^3 + y^3 + z^3 - 3*x*y*z = 2003} = 
  {(668, 668, 667), (668, 667, 668), (667, 668, 668)} := by
sorry

end NUMINAMATH_CALUDE_cube_sum_minus_product_eq_2003_l1154_115477


namespace NUMINAMATH_CALUDE_simplify_sqrt_2_simplify_complex_sqrt_l1154_115429

-- Part 1
theorem simplify_sqrt_2 : 2 * Real.sqrt 2 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

-- Part 2
theorem simplify_complex_sqrt : 
  Real.sqrt 2 * Real.sqrt 10 / (1 / Real.sqrt 5) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_2_simplify_complex_sqrt_l1154_115429


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_l1154_115430

theorem inequality_solution_implies_m (m : ℝ) : 
  (∀ x, mx + 2 > 0 ↔ x < 2) → m = -1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_l1154_115430


namespace NUMINAMATH_CALUDE_second_player_wins_l1154_115468

/-- Represents the state of the game -/
structure GameState where
  grid_size : Nat
  piece1_pos : Nat
  piece2_pos : Nat

/-- Defines a valid move in the game -/
inductive Move
  | One
  | Two

/-- Applies a move to a game state -/
def apply_move (state : GameState) (player : Nat) (move : Move) : GameState :=
  match player, move with
  | 1, Move.One => { state with piece1_pos := state.piece1_pos + 1 }
  | 1, Move.Two => { state with piece1_pos := state.piece1_pos + 2 }
  | 2, Move.One => { state with piece2_pos := state.piece2_pos - 1 }
  | 2, Move.Two => { state with piece2_pos := state.piece2_pos - 2 }
  | _, _ => state

/-- Checks if a move is valid -/
def is_valid_move (state : GameState) (player : Nat) (move : Move) : Prop :=
  match player, move with
  | 1, Move.One => state.piece1_pos + 1 < state.piece2_pos
  | 1, Move.Two => state.piece1_pos + 2 < state.piece2_pos
  | 2, Move.One => state.piece1_pos < state.piece2_pos - 1
  | 2, Move.Two => state.piece1_pos < state.piece2_pos - 2
  | _, _ => False

/-- Checks if the game is over -/
def is_game_over (state : GameState) : Prop :=
  state.piece2_pos - state.piece1_pos <= 1

/-- Checks if the number of empty squares between pieces is a multiple of 3 -/
def is_multiple_of_three (state : GameState) : Prop :=
  (state.piece2_pos - state.piece1_pos - 1) % 3 = 0

/-- Theorem: The second player has a winning strategy if and only if
    the number of empty squares between pieces is always a multiple of 3
    after the second player's move -/
theorem second_player_wins (initial_state : GameState)
  (h_initial : initial_state.grid_size = 20 ∧
               initial_state.piece1_pos = 1 ∧
               initial_state.piece2_pos = 20) :
  (∀ (game_state : GameState),
   ∀ (move1 : Move),
   is_valid_move game_state 1 move1 →
   ∃ (move2 : Move),
   is_valid_move (apply_move game_state 1 move1) 2 move2 ∧
   is_multiple_of_three (apply_move (apply_move game_state 1 move1) 2 move2)) ↔
  (∃ (strategy : GameState → Move),
   ∀ (game_state : GameState),
   ¬is_game_over game_state →
   is_valid_move game_state 2 (strategy game_state) ∧
   is_multiple_of_three (apply_move game_state 2 (strategy game_state))) :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l1154_115468


namespace NUMINAMATH_CALUDE_zero_of_f_l1154_115427

/-- The function f(x) = (x+1)^2 -/
def f (x : ℝ) : ℝ := (x + 1)^2

/-- The zero of f(x) is -1 -/
theorem zero_of_f : f (-1) = 0 := by sorry

end NUMINAMATH_CALUDE_zero_of_f_l1154_115427


namespace NUMINAMATH_CALUDE_hedgehog_strawberries_l1154_115412

theorem hedgehog_strawberries : 
  ∀ (num_hedgehogs num_baskets strawberries_per_basket : ℕ) 
    (remaining_fraction : ℚ),
  num_hedgehogs = 2 →
  num_baskets = 3 →
  strawberries_per_basket = 900 →
  remaining_fraction = 2 / 9 →
  ∃ (strawberries_eaten_per_hedgehog : ℕ),
    strawberries_eaten_per_hedgehog = 1050 ∧
    (num_baskets * strawberries_per_basket) * (1 - remaining_fraction) = 
      num_hedgehogs * strawberries_eaten_per_hedgehog :=
by sorry

end NUMINAMATH_CALUDE_hedgehog_strawberries_l1154_115412


namespace NUMINAMATH_CALUDE_optimal_selling_price_l1154_115435

def initial_purchase_price : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_sales_volume : ℝ := 500
def price_increase : ℝ → ℝ := λ x => x
def sales_volume : ℝ → ℝ := λ x => initial_sales_volume - 10 * price_increase x
def selling_price : ℝ → ℝ := λ x => initial_selling_price + price_increase x
def profit : ℝ → ℝ := λ x => (selling_price x * sales_volume x) - (initial_purchase_price * sales_volume x)

theorem optimal_selling_price :
  ∃ x : ℝ, (∀ y : ℝ, profit y ≤ profit x) ∧ selling_price x = 70 := by
  sorry

end NUMINAMATH_CALUDE_optimal_selling_price_l1154_115435


namespace NUMINAMATH_CALUDE_star_placement_impossible_l1154_115461

/-- Represents a grid of cells that may contain stars. -/
def Grid := Fin 10 → Fin 10 → Bool

/-- Checks if a 2x2 square starting at (i, j) contains exactly two stars. -/
def has_two_stars_2x2 (grid : Grid) (i j : Fin 10) : Prop :=
  (grid i j).toNat + (grid i (j+1)).toNat + (grid (i+1) j).toNat + (grid (i+1) (j+1)).toNat = 2

/-- Checks if a 3x1 rectangle starting at (i, j) contains exactly one star. -/
def has_one_star_3x1 (grid : Grid) (i j : Fin 10) : Prop :=
  (grid i j).toNat + (grid (i+1) j).toNat + (grid (i+2) j).toNat = 1

/-- The main theorem stating the impossibility of the star placement. -/
theorem star_placement_impossible : 
  ¬∃ (grid : Grid), 
    (∀ i j : Fin 9, has_two_stars_2x2 grid i j) ∧ 
    (∀ i : Fin 8, ∀ j : Fin 10, has_one_star_3x1 grid i j) :=
sorry

end NUMINAMATH_CALUDE_star_placement_impossible_l1154_115461


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l1154_115400

/-- The complex number z defined as i(-2 + i) -/
def z : ℂ := Complex.I * (Complex.mk (-2) 1)

/-- Predicate to check if a complex number is in the third quadrant -/
def in_third_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im < 0

/-- Theorem stating that z is in the third quadrant -/
theorem z_in_third_quadrant : in_third_quadrant z := by sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l1154_115400


namespace NUMINAMATH_CALUDE_valid_numbers_l1154_115414

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 
    a < b ∧ 
    b ≤ 9 ∧
    n = 10 * a + b ∧ 
    n = (b - a + 1) * (a + b) / 2

theorem valid_numbers : 
  {n : ℕ | is_valid_number n} = {14, 26, 37, 48, 59} := by sorry

end NUMINAMATH_CALUDE_valid_numbers_l1154_115414


namespace NUMINAMATH_CALUDE_cody_game_expense_l1154_115499

theorem cody_game_expense (initial_amount birthday_gift final_amount : ℕ) 
  (h1 : initial_amount = 45)
  (h2 : birthday_gift = 9)
  (h3 : final_amount = 35) :
  initial_amount + birthday_gift - final_amount = 19 :=
by sorry

end NUMINAMATH_CALUDE_cody_game_expense_l1154_115499


namespace NUMINAMATH_CALUDE_kelly_carrot_harvest_l1154_115466

/-- Represents the number of carrots harvested from each bed -/
structure CarrotHarvest where
  bed1 : ℕ
  bed2 : ℕ
  bed3 : ℕ

/-- Calculates the total weight of carrots in pounds -/
def totalWeight (harvest : CarrotHarvest) (carrotsPerPound : ℕ) : ℕ :=
  (harvest.bed1 + harvest.bed2 + harvest.bed3) / carrotsPerPound

/-- Theorem stating that Kelly's carrot harvest weighs 39 pounds -/
theorem kelly_carrot_harvest :
  let harvest := CarrotHarvest.mk 55 101 78
  let carrotsPerPound := 6
  totalWeight harvest carrotsPerPound = 39 := by
  sorry


end NUMINAMATH_CALUDE_kelly_carrot_harvest_l1154_115466


namespace NUMINAMATH_CALUDE_donation_theorem_l1154_115440

def donation_problem (total_donation : ℚ) 
  (community_pantry_fraction : ℚ) 
  (crisis_fund_fraction : ℚ) 
  (contingency_amount : ℚ) : Prop :=
  let remaining := total_donation - (community_pantry_fraction * total_donation) - (crisis_fund_fraction * total_donation)
  let livelihood_amount := remaining - contingency_amount
  livelihood_amount / remaining = 1 / 4

theorem donation_theorem : 
  donation_problem 240 (1/3) (1/2) 30 := by
  sorry

end NUMINAMATH_CALUDE_donation_theorem_l1154_115440


namespace NUMINAMATH_CALUDE_at_least_one_nonnegative_l1154_115498

theorem at_least_one_nonnegative (x y z : ℝ) :
  max (x^2 + y + 1/4) (max (y^2 + z + 1/4) (z^2 + x + 1/4)) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_nonnegative_l1154_115498


namespace NUMINAMATH_CALUDE_divisibility_of_fifth_power_differences_l1154_115471

theorem divisibility_of_fifth_power_differences (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (x - y) * (y - z) * (z - x)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_fifth_power_differences_l1154_115471


namespace NUMINAMATH_CALUDE_fraction_equality_implies_constants_l1154_115408

theorem fraction_equality_implies_constants (a b : ℝ) :
  (∀ x : ℝ, x ≠ -b → x ≠ -36 → x ≠ -30 → 
    (x - a) / (x + b) = (x^2 - 45*x + 504) / (x^2 + 66*x - 1080)) →
  a = 18 ∧ b = 30 ∧ a + b = 48 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_constants_l1154_115408


namespace NUMINAMATH_CALUDE_unique_solution_xy_l1154_115483

theorem unique_solution_xy (x y : ℝ) 
  (h1 : x^2 + y^2 = 2)
  (h2 : x^2 / (2 - y) + y^2 / (2 - x) = 2) :
  x = 1 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_xy_l1154_115483


namespace NUMINAMATH_CALUDE_f_odd_implies_b_one_f_monotone_increasing_l1154_115496

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := Real.log (Real.sqrt (4 * x^2 + b) + 2 * x) / Real.log 2

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem f_odd_implies_b_one (b : ℝ) :
  is_odd (f b) → b = 1 := by sorry

theorem f_monotone_increasing (b : ℝ) :
  ∀ x₁ x₂, x₁ < x₂ → f b x₁ < f b x₂ := by sorry

end NUMINAMATH_CALUDE_f_odd_implies_b_one_f_monotone_increasing_l1154_115496


namespace NUMINAMATH_CALUDE_troys_home_distance_l1154_115420

/-- The distance between Troy's home and school -/
def troys_distance : ℝ := 75

/-- The distance between Emily's home and school -/
def emilys_distance : ℝ := 98

/-- The additional distance Emily walks compared to Troy in five days -/
def additional_distance : ℝ := 230

/-- The number of days -/
def days : ℕ := 5

theorem troys_home_distance :
  troys_distance = 75 ∧
  emilys_distance = 98 ∧
  additional_distance = 230 ∧
  days = 5 →
  days * (2 * emilys_distance) - days * (2 * troys_distance) = additional_distance :=
by sorry

end NUMINAMATH_CALUDE_troys_home_distance_l1154_115420


namespace NUMINAMATH_CALUDE_reciprocal_of_opposite_negative_two_thirds_l1154_115453

theorem reciprocal_of_opposite_negative_two_thirds :
  (-(- (2 : ℚ) / 3))⁻¹ = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_opposite_negative_two_thirds_l1154_115453


namespace NUMINAMATH_CALUDE_point_on_line_equal_intercepts_l1154_115410

/-- A line passing through (-2, -3) with equal x and y intercepts -/
def line_with_equal_intercepts (x y : ℝ) : Prop :=
  x + y = 5

/-- The point (-2, -3) lies on the line -/
theorem point_on_line : line_with_equal_intercepts (-2) (-3) := by sorry

/-- The line has equal intercepts on x and y axes -/
theorem equal_intercepts :
  ∃ a : ℝ, a > 0 ∧ line_with_equal_intercepts a 0 ∧ line_with_equal_intercepts 0 a := by sorry

end NUMINAMATH_CALUDE_point_on_line_equal_intercepts_l1154_115410


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l1154_115487

-- Define the sets S and P
def S : Set ℝ := {x | x^2 - 3*x - 10 < 0}
def P (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2*a + 15}

-- Theorem statement
theorem subset_implies_a_range (a : ℝ) : S ⊆ P a → a ∈ Set.Icc (-5) (-3) := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l1154_115487


namespace NUMINAMATH_CALUDE_probability_six_consecutive_heads_l1154_115494

/-- The number of ways to get at least 6 consecutive heads in 9 coin flips -/
def consecutiveHeadsCount : ℕ := 49

/-- The total number of possible outcomes when flipping a coin 9 times -/
def totalOutcomes : ℕ := 512

/-- A fair coin is flipped 9 times. This theorem states that the probability
    of getting at least 6 consecutive heads is 49/512. -/
theorem probability_six_consecutive_heads :
  (consecutiveHeadsCount : ℚ) / totalOutcomes = 49 / 512 := by
  sorry

end NUMINAMATH_CALUDE_probability_six_consecutive_heads_l1154_115494


namespace NUMINAMATH_CALUDE_mikes_music_store_spending_l1154_115446

/-- The amount Mike spent on the trumpet -/
def trumpet_cost : ℚ := 145.16

/-- The amount Mike spent on the song book -/
def song_book_cost : ℚ := 5.84

/-- The total amount Mike spent at the music store -/
def total_spent : ℚ := trumpet_cost + song_book_cost

/-- Theorem stating that the total amount Mike spent is $151.00 -/
theorem mikes_music_store_spending :
  total_spent = 151.00 := by sorry

end NUMINAMATH_CALUDE_mikes_music_store_spending_l1154_115446


namespace NUMINAMATH_CALUDE_runner_speed_problem_l1154_115411

theorem runner_speed_problem (total_distance : ℝ) (total_time : ℝ) (first_segment_distance : ℝ) (first_segment_speed : ℝ) (last_segment_distance : ℝ) :
  total_distance = 16 →
  total_time = 1.5 →
  first_segment_distance = 10 →
  first_segment_speed = 12 →
  last_segment_distance = 6 →
  (last_segment_distance / (total_time - (first_segment_distance / first_segment_speed))) = 9 := by
  sorry

end NUMINAMATH_CALUDE_runner_speed_problem_l1154_115411


namespace NUMINAMATH_CALUDE_game_ends_in_finite_steps_l1154_115422

/-- Represents the state of a bowl in the game -/
inductive BowlState
| Empty : BowlState
| NonEmpty : BowlState

/-- Represents the game state -/
def GameState (n : ℕ) := Fin n → BowlState

/-- Function to place a bean in a bowl -/
def placeBeanInBowl (k : ℕ) (n : ℕ) : Fin n :=
  ⟨k * (k + 1) / 2, sorry⟩

/-- Predicate to check if a number is a power of 2 -/
def isPowerOfTwo (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

/-- Theorem stating the condition for the game to end in finite steps -/
theorem game_ends_in_finite_steps (n : ℕ) :
  (∃ k : ℕ, ∀ i : Fin n, (placeBeanInBowl k n).val = i.val → 
    ∃ m : ℕ, m ≤ k ∧ (placeBeanInBowl m n).val = i.val) ↔ 
  isPowerOfTwo n :=
sorry


end NUMINAMATH_CALUDE_game_ends_in_finite_steps_l1154_115422


namespace NUMINAMATH_CALUDE_alice_apples_l1154_115418

theorem alice_apples (A : ℕ) : 
  A > 2 →
  A % 9 = 2 →
  A % 10 = 2 →
  A % 11 = 2 →
  (∀ B : ℕ, B > 2 → B % 9 = 2 → B % 10 = 2 → B % 11 = 2 → A ≤ B) →
  A = 992 := by
sorry

end NUMINAMATH_CALUDE_alice_apples_l1154_115418


namespace NUMINAMATH_CALUDE_max_angle_C_in_triangle_l1154_115404

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a² + b² = 2c², then the maximum value of angle C is π/3 -/
theorem max_angle_C_in_triangle (a b c : ℝ) (h : a^2 + b^2 = 2*c^2) :
  ∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧ 
    A + B + C = π ∧
    c = Real.sqrt (a^2 + b^2 - 2*a*b*Real.cos C) ∧
    C ≤ π/3 ∧
    (C = π/3 → a = b) := by
  sorry

end NUMINAMATH_CALUDE_max_angle_C_in_triangle_l1154_115404


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l1154_115415

theorem degree_to_radian_conversion (π : Real) :
  (60 : Real) * (π / 180) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l1154_115415


namespace NUMINAMATH_CALUDE_stan_playlist_sufficient_stan_playlist_sufficient_proof_l1154_115426

theorem stan_playlist_sufficient (total_run_time : ℕ) 
  (songs_3min songs_4min songs_6min : ℕ) 
  (max_songs_per_category : ℕ) 
  (min_favorite_songs : ℕ) 
  (favorite_song_length : ℕ) : Prop :=
  total_run_time = 90 ∧
  songs_3min ≥ 10 ∧
  songs_4min ≥ 12 ∧
  songs_6min ≥ 15 ∧
  max_songs_per_category = 7 ∧
  min_favorite_songs = 3 ∧
  favorite_song_length = 4 →
  ∃ (playlist_3min playlist_4min playlist_6min : ℕ),
    playlist_3min ≤ max_songs_per_category ∧
    playlist_4min ≤ max_songs_per_category ∧
    playlist_6min ≤ max_songs_per_category ∧
    playlist_4min ≥ min_favorite_songs ∧
    playlist_3min * 3 + playlist_4min * 4 + playlist_6min * 6 ≥ total_run_time

theorem stan_playlist_sufficient_proof : stan_playlist_sufficient 90 10 12 15 7 3 4 := by
  sorry

end NUMINAMATH_CALUDE_stan_playlist_sufficient_stan_playlist_sufficient_proof_l1154_115426


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1154_115416

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.abs (1 + Complex.I)) : 
  z.im = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1154_115416


namespace NUMINAMATH_CALUDE_remainder_sum_mod_nine_l1154_115480

theorem remainder_sum_mod_nine (a b c : ℕ) : 
  0 < a ∧ a < 10 ∧
  0 < b ∧ b < 10 ∧
  0 < c ∧ c < 10 ∧
  (a * b * c) % 9 = 1 ∧
  (4 * c) % 9 = 5 ∧
  (7 * b) % 9 = (4 + b) % 9 →
  (a + b + c) % 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_nine_l1154_115480


namespace NUMINAMATH_CALUDE_min_value_inequality_l1154_115491

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 5) :
  (9 / x) + (25 / y) + (49 / z) ≥ 45 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 5 ∧ (9 / x) + (25 / y) + (49 / z) = 45 :=
by sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1154_115491


namespace NUMINAMATH_CALUDE_brick_width_calculation_l1154_115482

/-- Calculates the width of a brick given the wall dimensions, brick dimensions, and number of bricks --/
theorem brick_width_calculation (wall_length wall_height wall_thickness : ℝ)
                                (brick_length brick_height : ℝ)
                                (num_bricks : ℕ) :
  wall_length = 800 →
  wall_height = 600 →
  wall_thickness = 22.5 →
  brick_length = 125 →
  brick_height = 6 →
  num_bricks = 1280 →
  ∃ (brick_width : ℝ),
    brick_width = 11.25 ∧
    wall_length * wall_height * wall_thickness =
    num_bricks * brick_length * brick_width * brick_height :=
by
  sorry

#check brick_width_calculation

end NUMINAMATH_CALUDE_brick_width_calculation_l1154_115482


namespace NUMINAMATH_CALUDE_sqrt_comparison_l1154_115424

theorem sqrt_comparison : Real.sqrt 10 - Real.sqrt 6 < Real.sqrt 7 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_comparison_l1154_115424


namespace NUMINAMATH_CALUDE_project_completion_time_project_completion_time_solution_l1154_115464

/-- Represents the project completion time problem -/
theorem project_completion_time 
  (initial_workers : ℕ) 
  (initial_days : ℕ) 
  (additional_workers : ℕ) 
  (efficiency_improvement : ℚ) : ℕ :=
  let total_work := initial_workers * initial_days
  let new_workers := initial_workers + additional_workers
  let new_efficiency := 1 + efficiency_improvement
  let new_daily_work := new_workers * new_efficiency
  ⌊(total_work / new_daily_work : ℚ)⌋₊
    
/-- The solution to the specific problem instance -/
theorem project_completion_time_solution :
  project_completion_time 10 20 5 (1/10) = 12 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_project_completion_time_solution_l1154_115464


namespace NUMINAMATH_CALUDE_books_remaining_correct_l1154_115445

/-- Calculates the number of books remaining on the shelf by the evening. -/
def books_remaining (initial : ℕ) (borrowed_lunch : ℕ) (added : ℕ) (borrowed_evening : ℕ) : ℕ :=
  initial - borrowed_lunch + added - borrowed_evening

/-- Proves that the number of books remaining on the shelf by the evening is correct. -/
theorem books_remaining_correct (initial : ℕ) (borrowed_lunch : ℕ) (added : ℕ) (borrowed_evening : ℕ)
    (h1 : initial = 100)
    (h2 : borrowed_lunch = 50)
    (h3 : added = 40)
    (h4 : borrowed_evening = 30) :
    books_remaining initial borrowed_lunch added borrowed_evening = 60 := by
  sorry

end NUMINAMATH_CALUDE_books_remaining_correct_l1154_115445


namespace NUMINAMATH_CALUDE_characterization_of_valid_n_l1154_115458

def floor_sqrt (n : ℕ) : ℕ := Nat.sqrt n

def is_valid (n : ℕ) : Prop :=
  (n > 0) ∧
  (∃ k₁ : ℕ, n - 4 = k₁ * (floor_sqrt n - 2)) ∧
  (∃ k₂ : ℕ, n + 4 = k₂ * (floor_sqrt n + 2))

def special_set : Set ℕ := {2, 4, 11, 20, 31, 36, 44}

def general_form (a : ℕ) : ℕ := a^2 + 2*a - 4

theorem characterization_of_valid_n :
  ∀ n : ℕ, is_valid n ↔ (n ∈ special_set ∨ ∃ a : ℕ, a > 2 ∧ n = general_form a) :=
sorry

end NUMINAMATH_CALUDE_characterization_of_valid_n_l1154_115458


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l1154_115457

theorem fraction_equality_sum (M N : ℚ) :
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + 2 * N = 330 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l1154_115457


namespace NUMINAMATH_CALUDE_inverse_of_proposition_l1154_115484

theorem inverse_of_proposition :
  (∀ a b : ℝ, a = -2*b → a^2 = 4*b^2) →
  (∀ a b : ℝ, a^2 = 4*b^2 → a = -2*b) :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_proposition_l1154_115484


namespace NUMINAMATH_CALUDE_stating_min_sides_for_rotation_l1154_115455

/-- The rotation angle in degrees -/
def rotation_angle : ℚ := 25 + 30 / 60

/-- The fraction of a full circle that the rotation represents -/
def rotation_fraction : ℚ := rotation_angle / 360

/-- The minimum number of sides for the polygons -/
def min_sides : ℕ := 240

/-- 
  Theorem stating that the minimum number of sides for two identical polygons
  that coincide when one is rotated by 25°30' is 240
-/
theorem min_sides_for_rotation :
  ∀ n : ℕ, 
    (n > 0 ∧ (rotation_fraction * n).den = 1) → 
    n ≥ min_sides :=
sorry

end NUMINAMATH_CALUDE_stating_min_sides_for_rotation_l1154_115455


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt_40_l1154_115441

theorem closest_integer_to_sqrt_40 :
  ∃ (n : ℤ), n = 6 ∧ ∀ (m : ℤ), |m - Real.sqrt 40| ≥ |n - Real.sqrt 40| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt_40_l1154_115441


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1154_115460

theorem sum_of_coefficients (a b c : ℕ+) : 
  (∃ (k : ℚ), k * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) = 
    (a * Real.sqrt 6 + b * Real.sqrt 8) / c) →
  (∀ (x y z : ℕ+), 
    (∃ (l : ℚ), l * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) = 
      (x * Real.sqrt 6 + y * Real.sqrt 8) / z) → 
    c ≤ z) →
  a + b + c = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1154_115460


namespace NUMINAMATH_CALUDE_fixed_point_of_line_l1154_115417

theorem fixed_point_of_line (m : ℝ) :
  (∀ m : ℝ, ∃! p : ℝ × ℝ, m * p.1 + p.2 - 1 + 2 * m = 0) →
  (∃ p : ℝ × ℝ, p = (-2, 1) ∧ ∀ m : ℝ, m * p.1 + p.2 - 1 + 2 * m = 0) :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_l1154_115417


namespace NUMINAMATH_CALUDE_elvis_song_writing_time_l1154_115403

/-- Given Elvis's album production parameters, prove the time to write each song. -/
theorem elvis_song_writing_time
  (total_songs : ℕ)
  (studio_time_hours : ℕ)
  (recording_time_per_song : ℕ)
  (total_editing_time : ℕ)
  (h1 : total_songs = 15)
  (h2 : studio_time_hours = 7)
  (h3 : recording_time_per_song = 18)
  (h4 : total_editing_time = 45) :
  (studio_time_hours * 60 - recording_time_per_song * total_songs - total_editing_time) / total_songs = 7 :=
by sorry

end NUMINAMATH_CALUDE_elvis_song_writing_time_l1154_115403


namespace NUMINAMATH_CALUDE_ariels_fish_count_l1154_115431

theorem ariels_fish_count (total : ℕ) (male_ratio : ℚ) (female_count : ℕ) 
  (h1 : male_ratio = 2/3)
  (h2 : female_count = 15)
  (h3 : ↑female_count = (1 - male_ratio) * ↑total) : 
  total = 45 := by
  sorry

end NUMINAMATH_CALUDE_ariels_fish_count_l1154_115431


namespace NUMINAMATH_CALUDE_smallest_two_digit_number_one_more_than_multiple_l1154_115447

theorem smallest_two_digit_number_one_more_than_multiple (n : ℕ) : n = 71 ↔ 
  (n ≥ 10 ∧ n < 100) ∧ 
  (∃ k : ℕ, n = 2 * k + 1 ∧ n = 5 * k + 1 ∧ n = 7 * k + 1) ∧
  (∀ m : ℕ, m < n → ¬(m ≥ 10 ∧ m < 100 ∧ ∃ k : ℕ, m = 2 * k + 1 ∧ m = 5 * k + 1 ∧ m = 7 * k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_number_one_more_than_multiple_l1154_115447


namespace NUMINAMATH_CALUDE_matthew_hotdogs_l1154_115448

/-- The number of hotdogs Ella wants -/
def ella_hotdogs : ℕ := 2

/-- The number of hotdogs Emma wants -/
def emma_hotdogs : ℕ := 2

/-- The number of hotdogs Luke wants -/
def luke_hotdogs : ℕ := 2 * (ella_hotdogs + emma_hotdogs)

/-- The number of hotdogs Hunter wants -/
def hunter_hotdogs : ℕ := (3 * (ella_hotdogs + emma_hotdogs)) / 2

/-- The total number of hotdogs Matthew needs to cook -/
def total_hotdogs : ℕ := ella_hotdogs + emma_hotdogs + luke_hotdogs + hunter_hotdogs

theorem matthew_hotdogs : total_hotdogs = 14 := by
  sorry

end NUMINAMATH_CALUDE_matthew_hotdogs_l1154_115448


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1154_115478

theorem angle_measure_proof (AOB BOC : Real) : 
  AOB + BOC = 180 →  -- adjacent supplementary angles
  AOB = BOC + 18 →   -- AOB is 18° larger than BOC
  AOB = 99 :=        -- prove that AOB is 99°
by sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1154_115478


namespace NUMINAMATH_CALUDE_square_in_M_l1154_115488

/-- The set of functions f: ℝ → ℝ with the property that there exist real numbers a and k (k ≠ 0)
    such that f(a+x) = kf(a-x) for all x ∈ ℝ -/
def M : Set (ℝ → ℝ) :=
  {f | ∃ (a k : ℝ), k ≠ 0 ∧ ∀ x, f (a + x) = k * f (a - x)}

/-- The square function -/
def square : ℝ → ℝ := fun x ↦ x^2

/-- Theorem: The square function belongs to set M -/
theorem square_in_M : square ∈ M := by sorry

end NUMINAMATH_CALUDE_square_in_M_l1154_115488


namespace NUMINAMATH_CALUDE_dartboard_double_score_angle_l1154_115463

theorem dartboard_double_score_angle (num_regions : ℕ) (probability : ℚ) :
  num_regions = 6 →
  probability = 1 / 8 →
  (360 : ℚ) * probability = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_dartboard_double_score_angle_l1154_115463


namespace NUMINAMATH_CALUDE_dough_perimeter_l1154_115401

theorem dough_perimeter (dough_width : ℕ) (mold_side : ℕ) (unused_width : ℕ) (total_cookies : ℕ) :
  dough_width = 34 →
  mold_side = 4 →
  unused_width = 2 →
  total_cookies = 24 →
  let used_width := dough_width - unused_width
  let molds_across := used_width / mold_side
  let molds_along := total_cookies / molds_across
  let dough_length := molds_along * mold_side
  2 * dough_width + 2 * dough_length = 92 := by
  sorry

end NUMINAMATH_CALUDE_dough_perimeter_l1154_115401


namespace NUMINAMATH_CALUDE_harriet_miles_l1154_115467

theorem harriet_miles (total_miles : ℕ) (katarina_miles : ℕ) 
  (h1 : total_miles = 195)
  (h2 : katarina_miles = 51)
  (h3 : ∃ x : ℕ, x * 3 + katarina_miles = total_miles) :
  ∃ harriet_miles : ℕ, harriet_miles = 48 ∧ 
    harriet_miles * 3 + katarina_miles = total_miles := by
  sorry

end NUMINAMATH_CALUDE_harriet_miles_l1154_115467


namespace NUMINAMATH_CALUDE_curve_tangent_product_l1154_115413

/-- Given a curve y = ax³ + bx where the point (2, 2) lies on the curve
    and the slope of the tangent line at this point is 9,
    prove that the product ab equals -3. -/
theorem curve_tangent_product (a b : ℝ) : 
  (2 : ℝ) = a * (2 : ℝ)^3 + b * (2 : ℝ) → -- Point (2, 2) lies on the curve
  (9 : ℝ) = 3 * a * (2 : ℝ)^2 + b →       -- Slope of tangent at (2, 2) is 9
  a * b = -3 := by
sorry

end NUMINAMATH_CALUDE_curve_tangent_product_l1154_115413


namespace NUMINAMATH_CALUDE_two_roots_theorem_l1154_115472

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y → f x < f y

def has_exactly_two_roots (f : ℝ → ℝ) : Prop :=
  ∃ a b, a < b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b

theorem two_roots_theorem (f : ℝ → ℝ) 
  (h1 : even_function f)
  (h2 : monotone_increasing_nonneg f)
  (h3 : f 1 * f 2 < 0) :
  has_exactly_two_roots f :=
sorry

end NUMINAMATH_CALUDE_two_roots_theorem_l1154_115472


namespace NUMINAMATH_CALUDE_other_number_from_hcf_lcm_l1154_115406

theorem other_number_from_hcf_lcm (A B : ℕ+) : 
  Nat.gcd A B = 12 → 
  Nat.lcm A B = 396 → 
  A = 24 → 
  B = 198 := by
sorry

end NUMINAMATH_CALUDE_other_number_from_hcf_lcm_l1154_115406


namespace NUMINAMATH_CALUDE_missed_number_l1154_115405

theorem missed_number (n : ℕ) (incorrect_sum correct_sum missed_number : ℕ) :
  n > 0 →
  incorrect_sum = 575 →
  correct_sum = n * (n + 1) / 2 →
  correct_sum = 595 →
  incorrect_sum + missed_number = correct_sum →
  missed_number = 20 := by
  sorry

end NUMINAMATH_CALUDE_missed_number_l1154_115405


namespace NUMINAMATH_CALUDE_range_of_trigonometric_function_l1154_115442

theorem range_of_trigonometric_function :
  ∀ x : ℝ, 0 ≤ Real.cos x ^ 4 + Real.cos x * Real.sin x + Real.sin x ^ 4 ∧
           Real.cos x ^ 4 + Real.cos x * Real.sin x + Real.sin x ^ 4 ≤ 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_trigonometric_function_l1154_115442


namespace NUMINAMATH_CALUDE_problem_solution_l1154_115462

theorem problem_solution : 
  (∀ x : ℝ, (Real.sqrt 24 - Real.sqrt 6) / Real.sqrt 3 - (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) = Real.sqrt 2 - 1) ∧ 
  (∀ x : ℝ, 2 * x^3 - 16 = 0 ↔ x = 2) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1154_115462


namespace NUMINAMATH_CALUDE_frog_reaches_boundary_in_three_hops_l1154_115425

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Represents the possible directions of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Defines whether a position is on the boundary of the grid -/
def is_boundary (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Defines a single hop movement on the grid -/
def hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨min 3 (p.x + 1), p.y⟩
  | Direction.Down => ⟨max 0 (p.x - 1), p.y⟩
  | Direction.Left => ⟨p.x, max 0 (p.y - 1)⟩
  | Direction.Right => ⟨p.x, min 3 (p.y + 1)⟩

/-- Calculates the probability of reaching the boundary within n hops -/
def prob_reach_boundary (start : Position) (n : Nat) : ℝ :=
  sorry

theorem frog_reaches_boundary_in_three_hops :
  prob_reach_boundary ⟨1, 1⟩ 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_frog_reaches_boundary_in_three_hops_l1154_115425


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1154_115485

theorem complex_equation_solution (a : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a * i) / (2 - i) = (1 - 2*i) / 5) : a = -1 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1154_115485


namespace NUMINAMATH_CALUDE_harry_hours_worked_l1154_115454

/-- Represents the payment structure and hours worked for an employee -/
structure Employee where
  baseHours : ℕ  -- Number of hours paid at base rate
  baseRate : ℝ   -- Base hourly rate
  overtimeRate : ℝ  -- Overtime hourly rate
  hoursWorked : ℕ  -- Total hours worked

/-- Calculates the total pay for an employee -/
def totalPay (e : Employee) : ℝ :=
  let baseAmount := min e.hoursWorked e.baseHours * e.baseRate
  let overtimeHours := max (e.hoursWorked - e.baseHours) 0
  baseAmount + overtimeHours * e.overtimeRate

/-- The main theorem to prove -/
theorem harry_hours_worked 
  (x : ℝ) 
  (harry : Employee) 
  (james : Employee) :
  harry.baseHours = 12 ∧ 
  harry.baseRate = x ∧ 
  harry.overtimeRate = 1.5 * x ∧
  james.baseHours = 40 ∧ 
  james.baseRate = x ∧ 
  james.overtimeRate = 2 * x ∧
  james.hoursWorked = 41 ∧
  totalPay harry = totalPay james →
  harry.hoursWorked = 32 := by
  sorry


end NUMINAMATH_CALUDE_harry_hours_worked_l1154_115454


namespace NUMINAMATH_CALUDE_transformed_stddev_l1154_115407

variable {n : ℕ}
variable (a : Fin n → ℝ)
variable (S : ℝ)

def variance (x : Fin n → ℝ) : ℝ := sorry

def stdDev (x : Fin n → ℝ) : ℝ := sorry

theorem transformed_stddev 
  (h : variance a = S^2) : 
  stdDev (fun i => 2 * a i - 3) = 2 * S := by sorry

end NUMINAMATH_CALUDE_transformed_stddev_l1154_115407


namespace NUMINAMATH_CALUDE_freshman_psych_liberal_arts_percentage_l1154_115409

def college_population (total : ℝ) : Prop :=
  let freshmen := 0.5 * total
  let int_freshmen := 0.3 * freshmen
  let dom_freshmen := 0.7 * freshmen
  let int_lib_arts := 0.4 * int_freshmen
  let dom_lib_arts := 0.35 * dom_freshmen
  let int_psych_lib_arts := 0.2 * int_lib_arts
  let dom_psych_lib_arts := 0.25 * dom_lib_arts
  let total_psych_lib_arts := int_psych_lib_arts + dom_psych_lib_arts
  total_psych_lib_arts / total = 0.04

theorem freshman_psych_liberal_arts_percentage :
  ∀ total : ℝ, total > 0 → college_population total :=
sorry

end NUMINAMATH_CALUDE_freshman_psych_liberal_arts_percentage_l1154_115409


namespace NUMINAMATH_CALUDE_probability_two_white_balls_correct_l1154_115465

/-- The probability of having two white balls in an urn, given the conditions of the problem -/
def probability_two_white_balls (n : ℕ) : ℚ :=
  (4:ℚ)^n / ((2:ℚ) * (3:ℚ)^n + (4:ℚ)^n)

/-- The theorem stating the probability of having two white balls in the urn -/
theorem probability_two_white_balls_correct (n : ℕ) :
  let total_balls : ℕ := 4
  let draws : ℕ := 2 * n
  let white_draws : ℕ := n
  probability_two_white_balls n = (4:ℚ)^n / ((2:ℚ) * (3:ℚ)^n + (4:ℚ)^n) :=
by sorry

end NUMINAMATH_CALUDE_probability_two_white_balls_correct_l1154_115465


namespace NUMINAMATH_CALUDE_complementary_angles_can_be_both_acute_l1154_115434

-- Define what it means for two angles to be complementary
def complementary (a b : ℝ) : Prop := a + b = 90

-- Define what it means for an angle to be acute
def acute (a : ℝ) : Prop := 0 < a ∧ a < 90

theorem complementary_angles_can_be_both_acute :
  ∃ (a b : ℝ), complementary a b ∧ acute a ∧ acute b :=
sorry

end NUMINAMATH_CALUDE_complementary_angles_can_be_both_acute_l1154_115434


namespace NUMINAMATH_CALUDE_stationery_box_sheets_l1154_115470

/-- Represents a box of stationery -/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- The scenario described in the problem -/
def stationery_scenario (box : StationeryBox) : Prop :=
  (box.sheets - box.envelopes = 30) ∧ 
  (2 * box.envelopes = box.sheets)

/-- The theorem to prove -/
theorem stationery_box_sheets : 
  ∀ (box : StationeryBox), stationery_scenario box → box.sheets = 60 := by
  sorry

end NUMINAMATH_CALUDE_stationery_box_sheets_l1154_115470


namespace NUMINAMATH_CALUDE_january_display_144_l1154_115423

/-- Rose display sequence with a constant increase -/
structure RoseSequence where
  october : ℕ
  november : ℕ
  december : ℕ
  february : ℕ
  constant_increase : ℕ
  increase_consistent : 
    november - october = constant_increase ∧
    december - november = constant_increase ∧
    february - (december + constant_increase) = constant_increase

/-- The number of roses displayed in January given a rose sequence -/
def january_roses (seq : RoseSequence) : ℕ :=
  seq.december + seq.constant_increase

/-- Theorem stating that for the given rose sequence, January displays 144 roses -/
theorem january_display_144 (seq : RoseSequence) 
  (h_oct : seq.october = 108)
  (h_nov : seq.november = 120)
  (h_dec : seq.december = 132)
  (h_feb : seq.february = 156) :
  january_roses seq = 144 := by
  sorry


end NUMINAMATH_CALUDE_january_display_144_l1154_115423


namespace NUMINAMATH_CALUDE_exists_four_mutually_acquainted_l1154_115432

/-- Represents the acquaintance relation between people --/
def Acquainted (n : ℕ) := Fin n → Fin n → Prop

/-- The property that among every 3 people, at least 2 are acquainted --/
def AtLeastTwoAcquainted (n : ℕ) (acq : Acquainted n) : Prop :=
  ∀ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c →
    acq a b ∨ acq b c ∨ acq a c

/-- A subset of 4 mutually acquainted people --/
def FourMutuallyAcquainted (n : ℕ) (acq : Acquainted n) : Prop :=
  ∃ a b c d : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
    acq a b ∧ acq a c ∧ acq a d ∧ acq b c ∧ acq b d ∧ acq c d

/-- The main theorem --/
theorem exists_four_mutually_acquainted :
  ∀ (acq : Acquainted 9),
    AtLeastTwoAcquainted 9 acq →
    FourMutuallyAcquainted 9 acq :=
by
  sorry


end NUMINAMATH_CALUDE_exists_four_mutually_acquainted_l1154_115432


namespace NUMINAMATH_CALUDE_garden_tilling_time_l1154_115489

/-- Represents a rectangular obstacle in the garden -/
structure Obstacle where
  length : ℝ
  width : ℝ

/-- Represents the garden plot and tilling parameters -/
structure GardenPlot where
  shortBase : ℝ
  longBase : ℝ
  height : ℝ
  tillerWidth : ℝ
  tillingRate : ℝ
  obstacles : List Obstacle
  extraTimePerObstacle : ℝ

/-- Calculates the time required to till the garden plot -/
def tillingTime (plot : GardenPlot) : ℝ :=
  sorry

/-- Theorem stating the tilling time for the given garden plot -/
theorem garden_tilling_time :
  let plot : GardenPlot := {
    shortBase := 135,
    longBase := 170,
    height := 90,
    tillerWidth := 2.5,
    tillingRate := 1.5 / 3,
    obstacles := [
      { length := 20, width := 10 },
      { length := 15, width := 30 },
      { length := 10, width := 15 }
    ],
    extraTimePerObstacle := 15
  }
  abs (tillingTime plot - 173.08) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_garden_tilling_time_l1154_115489


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l1154_115428

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A * B * C = 3003 →
  A + B + C ≤ 49 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l1154_115428
