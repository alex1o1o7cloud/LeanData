import Mathlib

namespace NUMINAMATH_CALUDE_tens_digit_of_nine_power_2023_l2165_216536

theorem tens_digit_of_nine_power_2023 : 9^2023 % 100 = 29 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_nine_power_2023_l2165_216536


namespace NUMINAMATH_CALUDE_sport_gender_relationship_l2165_216540

/-- The critical value of K² for P(K² ≥ k) = 0.05 -/
def critical_value : ℝ := 3.841

/-- The observed value of K² -/
def observed_value : ℝ := 4.892

/-- The significance level -/
def significance_level : ℝ := 0.05

/-- The sample size -/
def sample_size : ℕ := 200

/-- Theorem stating that the observed value exceeds the critical value,
    allowing us to conclude a relationship between liking the sport and gender
    with 1 - significance_level confidence -/
theorem sport_gender_relationship :
  observed_value > critical_value →
  ∃ (confidence_level : ℝ), confidence_level = 1 - significance_level ∧
    confidence_level > 0.95 ∧
    (∃ (relationship : Prop), relationship) :=
by
  sorry

end NUMINAMATH_CALUDE_sport_gender_relationship_l2165_216540


namespace NUMINAMATH_CALUDE_x_equals_2_valid_l2165_216523

/-- Represents an assignment statement -/
inductive AssignmentStatement
| constant : ℕ → AssignmentStatement
| variable : String → ℕ → AssignmentStatement
| consecutive : String → String → ℕ → AssignmentStatement
| expression : String → String → ℕ → AssignmentStatement

/-- Checks if an assignment statement is valid -/
def isValidAssignment (stmt : AssignmentStatement) : Prop :=
  match stmt with
  | AssignmentStatement.variable _ _ => True
  | _ => False

theorem x_equals_2_valid :
  isValidAssignment (AssignmentStatement.variable "x" 2) = True :=
by sorry

end NUMINAMATH_CALUDE_x_equals_2_valid_l2165_216523


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2165_216589

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ+,
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 2 + (1 : ℚ) / z →
  (x = 1 ∧ y = 2 ∧ z = 1) ∨
  (x = 2 ∧ ((y = 1 ∧ z = 1) ∨ (y = z ∧ y ≥ 2))) ∨
  (x = 3 ∧ ((y = 3 ∧ z = 6) ∨ (y = 4 ∧ z = 12) ∨ (y = 5 ∧ z = 30) ∨ (y = 2 ∧ z = 3))) ∨
  (x ≥ 4 ∧ y ≥ 4 → False) :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2165_216589


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2165_216584

theorem fraction_to_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 0.650 :=
by
  sorry

#check fraction_to_decimal 13 320

end NUMINAMATH_CALUDE_fraction_to_decimal_l2165_216584


namespace NUMINAMATH_CALUDE_tangent_line_and_extrema_l2165_216528

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_and_extrema :
  let tangent_line (x : ℝ) := 1
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ f 0) ∧
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ f (Real.pi / 2)) ∧
  (HasDerivAt f (tangent_line 0 - f 0) 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_extrema_l2165_216528


namespace NUMINAMATH_CALUDE_number_equivalence_l2165_216544

theorem number_equivalence : ∃ x : ℕ, 
  x = 1 ∧ 
  x = 62 ∧ 
  x = 363 ∧ 
  x = 3634 ∧ 
  x = 365 ∧ 
  36 = 2 ∧ 
  x = 2 :=
by sorry

end NUMINAMATH_CALUDE_number_equivalence_l2165_216544


namespace NUMINAMATH_CALUDE_adults_who_ate_proof_l2165_216527

/-- Represents the number of adults who had their meal -/
def adults_who_ate : ℕ := sorry

/-- The total number of adults in the group -/
def total_adults : ℕ := 55

/-- The total number of children in the group -/
def total_children : ℕ := 70

/-- The meal capacity for adults -/
def meal_capacity_adults : ℕ := 70

/-- The meal capacity for children -/
def meal_capacity_children : ℕ := 90

/-- The number of children that can be fed with remaining food after some adults eat -/
def remaining_children_fed : ℕ := 72

theorem adults_who_ate_proof :
  adults_who_ate = 14 ∧
  adults_who_ate ≤ total_adults ∧
  (meal_capacity_adults - adults_who_ate) * meal_capacity_children / meal_capacity_adults = remaining_children_fed :=
sorry

end NUMINAMATH_CALUDE_adults_who_ate_proof_l2165_216527


namespace NUMINAMATH_CALUDE_trishas_walk_distance_l2165_216510

/-- The total distance Trisha walked during her vacation in New York City -/
theorem trishas_walk_distance :
  let distance_hotel_to_postcard : ℚ := 0.1111111111111111
  let distance_postcard_to_tshirt : ℚ := 0.1111111111111111
  let distance_tshirt_to_hotel : ℚ := 0.6666666666666666
  distance_hotel_to_postcard + distance_postcard_to_tshirt + distance_tshirt_to_hotel = 0.8888888888888888 := by
  sorry

end NUMINAMATH_CALUDE_trishas_walk_distance_l2165_216510


namespace NUMINAMATH_CALUDE_harmonic_mean_of_1_and_5040_l2165_216514

def harmonic_mean (a b : ℚ) : ℚ := 2 * a * b / (a + b)

theorem harmonic_mean_of_1_and_5040 :
  harmonic_mean 1 5040 = 10080 / 5041 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_of_1_and_5040_l2165_216514


namespace NUMINAMATH_CALUDE_max_value_of_s_l2165_216516

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 8)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 12) :
  s ≤ 2 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_s_l2165_216516


namespace NUMINAMATH_CALUDE_lisa_likes_one_last_digit_l2165_216586

def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def divisible_by_2 (n : ℕ) : Prop := n % 2 = 0

def last_digit (n : ℕ) : ℕ := n % 10

theorem lisa_likes_one_last_digit :
  ∃! d : ℕ, d < 10 ∧ ∀ n : ℕ, last_digit n = d → (divisible_by_5 n ∧ divisible_by_2 n) :=
by
  sorry

end NUMINAMATH_CALUDE_lisa_likes_one_last_digit_l2165_216586


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l2165_216538

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ y : ℝ, (3 - 4*I) * (a + b*I) = y*I) : a/b = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l2165_216538


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_problem_1_l2165_216564

theorem quadratic_roots_sum_and_product (a b c x₁ x₂ : ℝ) (ha : a ≠ 0) :
  a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 →
  x₁ + x₂ = -b / a ∧ x₁ * x₂ = c / a :=
by sorry

theorem problem_1 (x₁ x₂ : ℝ) :
  5 * x₁^2 + 10 * x₁ - 1 = 0 ∧ 5 * x₂^2 + 10 * x₂ - 1 = 0 →
  x₁ + x₂ = -2 ∧ x₁ * x₂ = -1/5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_problem_1_l2165_216564


namespace NUMINAMATH_CALUDE_door_height_problem_l2165_216577

theorem door_height_problem (pole_length width height diagonal : ℝ) : 
  pole_length > 0 ∧
  width > 0 ∧
  height > 0 ∧
  pole_length = width + 4 ∧
  pole_length = height + 2 ∧
  pole_length = diagonal ∧
  diagonal^2 = width^2 + height^2
  → height = 8 := by
  sorry

end NUMINAMATH_CALUDE_door_height_problem_l2165_216577


namespace NUMINAMATH_CALUDE_share_distribution_l2165_216555

theorem share_distribution (total : ℝ) (maya annie saiji : ℝ) : 
  total = 900 →
  maya = (1/2) * annie →
  annie = (1/2) * saiji →
  total = maya + annie + saiji →
  saiji = 900 * (4/7) :=
by sorry

end NUMINAMATH_CALUDE_share_distribution_l2165_216555


namespace NUMINAMATH_CALUDE_prob_diff_games_l2165_216590

/-- Probability of getting heads on a single toss of the biased coin -/
def p_heads : ℚ := 3/4

/-- Probability of getting tails on a single toss of the biased coin -/
def p_tails : ℚ := 1/4

/-- Probability of winning Game A -/
def p_win_game_a : ℚ := 
  4 * (p_heads^3 * p_tails) + p_heads^4

/-- Probability of winning Game B -/
def p_win_game_b : ℚ := 
  (p_heads^2 + p_tails^2)^2

/-- The difference in probabilities between winning Game A and Game B -/
theorem prob_diff_games : p_win_game_a - p_win_game_b = 89/256 := by
  sorry

end NUMINAMATH_CALUDE_prob_diff_games_l2165_216590


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2165_216556

theorem imaginary_part_of_z (z : ℂ) (h : z - Complex.I = (4 - 2 * Complex.I) / (1 + 2 * Complex.I)) : 
  z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2165_216556


namespace NUMINAMATH_CALUDE_ratio_of_a_to_c_l2165_216595

theorem ratio_of_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 8) :
  a / c = 7.5 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_a_to_c_l2165_216595


namespace NUMINAMATH_CALUDE_price_theorem_min_bottles_theorem_l2165_216567

-- Define variables
def peanut_oil_price : ℝ := sorry
def corn_oil_price : ℝ := sorry
def peanut_oil_sell_price : ℝ := 60
def peanut_oil_purchased : ℕ := 50

-- Define conditions
axiom condition1 : 20 * peanut_oil_price + 30 * corn_oil_price = 2200
axiom condition2 : 30 * peanut_oil_price + 10 * corn_oil_price = 1900

-- Define theorems to prove
theorem price_theorem :
  peanut_oil_price = 50 ∧ corn_oil_price = 40 :=
sorry

theorem min_bottles_theorem :
  ∀ n : ℕ, n * peanut_oil_sell_price > peanut_oil_purchased * peanut_oil_price →
  n ≥ 42 :=
sorry

end NUMINAMATH_CALUDE_price_theorem_min_bottles_theorem_l2165_216567


namespace NUMINAMATH_CALUDE_fraction_simplification_complex_fraction_simplification_l2165_216570

-- Problem 1
theorem fraction_simplification (a b : ℝ) (h : a ≠ b) : 
  (a / (a - b)) - (b / (a + b)) = (a^2 + b^2) / (a^2 - b^2) :=
sorry

-- Problem 2
theorem complex_fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  ((x - 2) / (x - 1)) / ((x^2 - 4*x + 4) / (x^2 - 1)) + ((1 - x) / (x - 2)) = 2 / (x - 2) :=
sorry

end NUMINAMATH_CALUDE_fraction_simplification_complex_fraction_simplification_l2165_216570


namespace NUMINAMATH_CALUDE_cosine_half_angle_l2165_216551

theorem cosine_half_angle (α : Real) (h : Real.cos (α/2)^2 = 1/3) : 
  Real.cos α = -1/3 := by
sorry

end NUMINAMATH_CALUDE_cosine_half_angle_l2165_216551


namespace NUMINAMATH_CALUDE_product_equality_l2165_216542

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Proves that if a * (reversed b) = 143, then a * b = 143 -/
theorem product_equality (a b : ℕ) 
  (ha : 100 ≤ a ∧ a < 1000) 
  (hb : 10 ≤ b ∧ b < 100) 
  (h : a * (reverse_digits b) = 143) : 
  a * b = 143 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l2165_216542


namespace NUMINAMATH_CALUDE_certain_number_equation_l2165_216594

theorem certain_number_equation : ∃! x : ℝ, 16 * x + 17 * x + 20 * x + 11 = 170 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l2165_216594


namespace NUMINAMATH_CALUDE_binomial_9_5_l2165_216588

theorem binomial_9_5 : Nat.choose 9 5 = 756 := by
  sorry

end NUMINAMATH_CALUDE_binomial_9_5_l2165_216588


namespace NUMINAMATH_CALUDE_xy_xz_yz_bounds_l2165_216501

theorem xy_xz_yz_bounds (x y z : ℝ) (h : 3 * (x + y + z) = x^2 + y^2 + z^2) :
  (∃ (a b c : ℝ), a + b + c = x + y + z ∧ a * b + b * c + c * a = 27) ∧
  (∃ (d e f : ℝ), d + e + f = x + y + z ∧ d * e + e * f + f * d = 0) ∧
  (∀ (u v w : ℝ), u + v + w = x + y + z → u * v + v * w + w * u ≤ 27) ∧
  (∀ (u v w : ℝ), u + v + w = x + y + z → u * v + v * w + w * u ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_xy_xz_yz_bounds_l2165_216501


namespace NUMINAMATH_CALUDE_total_rectangles_is_176_l2165_216581

/-- The number of gray cells in the picture -/
def total_gray_cells : ℕ := 40

/-- The number of "blue" cells (a subset of gray cells) -/
def blue_cells : ℕ := 36

/-- The number of "red" cells (the remaining subset of gray cells) -/
def red_cells : ℕ := total_gray_cells - blue_cells

/-- The number of unique rectangles containing each blue cell -/
def rectangles_per_blue_cell : ℕ := 4

/-- The number of unique rectangles containing each red cell -/
def rectangles_per_red_cell : ℕ := 8

/-- The total number of checkered rectangles containing exactly one gray cell -/
def total_rectangles : ℕ := blue_cells * rectangles_per_blue_cell + red_cells * rectangles_per_red_cell

theorem total_rectangles_is_176 : total_rectangles = 176 := by
  sorry

end NUMINAMATH_CALUDE_total_rectangles_is_176_l2165_216581


namespace NUMINAMATH_CALUDE_height_on_hypotenuse_l2165_216506

/-- Given a right triangle with legs of lengths 3 and 4, 
    the height on the hypotenuse is 12/5 -/
theorem height_on_hypotenuse (a b c h : ℝ) : 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → h * c = 2 * (a * b / 2) → h = 12/5 := by sorry

end NUMINAMATH_CALUDE_height_on_hypotenuse_l2165_216506


namespace NUMINAMATH_CALUDE_lcm_of_180_504_169_l2165_216598

def a : ℕ := 180
def b : ℕ := 504
def c : ℕ := 169

theorem lcm_of_180_504_169 : 
  Nat.lcm (Nat.lcm a b) c = 2^3 * 3^2 * 5 * 7 * 13^2 := by sorry

end NUMINAMATH_CALUDE_lcm_of_180_504_169_l2165_216598


namespace NUMINAMATH_CALUDE_range_of_a_l2165_216517

theorem range_of_a (a : ℝ) (h_a_pos : a > 0) : 
  (((∀ x y : ℝ, x < y → a^x > a^y) ∧ ¬(∀ x : ℝ, x^2 - 3*a*x + 1 > 0)) ∨
   (¬(∀ x y : ℝ, x < y → a^x > a^y) ∧ (∀ x : ℝ, x^2 - 3*a*x + 1 > 0))) →
  (2/3 ≤ a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2165_216517


namespace NUMINAMATH_CALUDE_cube_construction_proof_l2165_216554

/-- Represents a piece of cardboard with foldable and glueable edges -/
structure CardboardPiece where
  foldable_edges : Set (Nat × Nat)
  glueable_edges : Set (Nat × Nat)

/-- Represents a pair of cardboard pieces -/
structure CardboardOption where
  piece1 : CardboardPiece
  piece2 : CardboardPiece

/-- Checks if a CardboardOption can form a cube -/
def can_form_cube (option : CardboardOption) : Prop := sorry

/-- The set of all given options -/
def options : Set CardboardOption := sorry

/-- Option (e) from the given set -/
def option_e : CardboardOption := sorry

theorem cube_construction_proof :
  ∀ opt ∈ options, can_form_cube opt ↔ opt = option_e := by sorry

end NUMINAMATH_CALUDE_cube_construction_proof_l2165_216554


namespace NUMINAMATH_CALUDE_g_of_two_equals_fourteen_l2165_216526

-- Define g as a function from ℝ to ℝ
def g : ℝ → ℝ := sorry

-- State the theorem
theorem g_of_two_equals_fourteen :
  (∀ x : ℝ, g (3 * x - 4) = 4 * x + 6) →
  g 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_g_of_two_equals_fourteen_l2165_216526


namespace NUMINAMATH_CALUDE_simplify_expression_l2165_216597

theorem simplify_expression (a : ℝ) (h : a > 1) :
  (1 - a) * Real.sqrt (1 / (a - 1)) = -Real.sqrt (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2165_216597


namespace NUMINAMATH_CALUDE_circle_intersection_existence_l2165_216512

/-- Given a circle with diameter 2R and a line perpendicular to the diameter at distance a from one endpoint,
    this theorem states the conditions for the existence of points C on the circle and D on the perpendicular line
    such that CD = l. -/
theorem circle_intersection_existence (R a l : ℝ) : 
  (∃ (C D : ℝ × ℝ), 
    C.1^2 + C.2^2 = R^2 ∧ 
    D.1 = a ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = l^2) ↔ 
  ((0 < a ∧ a < 2*R ∧ l < 2*R - a) ∨
   (a > 2*R ∧ R > 0 ∧ l > a - 2*R) ∨
   (-2*R < a ∧ a < 0 ∧ l^2 ≥ -8*R*a ∧ l < 2*R - a) ∨
   (a < -2*R ∧ R < 0 ∧ l > 2*R - a)) :=
by sorry


end NUMINAMATH_CALUDE_circle_intersection_existence_l2165_216512


namespace NUMINAMATH_CALUDE_min_turns_for_1000_pieces_l2165_216513

/-- Represents the state of the game with black and white pieces on a circumference. -/
structure GameState where
  black : ℕ
  white : ℕ

/-- Represents a player's turn in the game. -/
inductive Turn
  | PlayerA
  | PlayerB

/-- Defines the rules for removing pieces based on the current player's turn. -/
def removePieces (state : GameState) (turn : Turn) : GameState :=
  match turn with
  | Turn.PlayerA => { black := state.black, white := state.white + 2 * state.black }
  | Turn.PlayerB => { black := state.black + 2 * state.white, white := state.white }

/-- Checks if the game has ended (only one color remains). -/
def isGameOver (state : GameState) : Bool :=
  state.black = 0 || state.white = 0

/-- Calculates the minimum number of turns required to end the game. -/
def minTurnsToEnd (initialState : GameState) : ℕ :=
  sorry

/-- Theorem stating that for 1000 initial pieces, the minimum number of turns to end the game is 8. -/
theorem min_turns_for_1000_pieces :
  ∃ (black white : ℕ), black + white = 1000 ∧ minTurnsToEnd { black := black, white := white } = 8 :=
  sorry

end NUMINAMATH_CALUDE_min_turns_for_1000_pieces_l2165_216513


namespace NUMINAMATH_CALUDE_weight_replacement_l2165_216563

theorem weight_replacement (initial_count : Nat) (weight_increase : ℝ) (new_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 5 →
  new_weight = 105 →
  (new_weight - (initial_count * weight_increase)) = 65 := by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l2165_216563


namespace NUMINAMATH_CALUDE_kamal_age_problem_l2165_216518

theorem kamal_age_problem (k s : ℕ) : 
  k - 8 = 4 * (s - 8) →
  k + 8 = 2 * (s + 8) →
  k = 40 :=
by sorry

end NUMINAMATH_CALUDE_kamal_age_problem_l2165_216518


namespace NUMINAMATH_CALUDE_f_value_at_2_l2165_216515

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_neg (x : ℝ) : ℝ := -x^2 + x

theorem f_value_at_2 (f : ℝ → ℝ) 
    (h_odd : is_odd_function f)
    (h_neg : ∀ x < 0, f x = f_neg x) : 
  f 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_f_value_at_2_l2165_216515


namespace NUMINAMATH_CALUDE_even_function_extension_l2165_216559

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_extension (f : ℝ → ℝ) (h_even : IsEven f) 
  (h_def : ∀ x > 0, f x = x * (1 + x)) :
  ∀ x < 0, f x = x * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_even_function_extension_l2165_216559


namespace NUMINAMATH_CALUDE_quadratic_root_form_l2165_216530

/-- The quadratic equation 2x^2 - 5x - 4 = 0 -/
def quadratic_equation (x : ℝ) : Prop := 2 * x^2 - 5 * x - 4 = 0

/-- The roots of the equation in the form (m ± √n) / p -/
def root_form (m n p : ℕ) (x : ℝ) : Prop :=
  ∃ (sign : Bool), x = (m + if sign then 1 else -1 * Real.sqrt n) / p

/-- m, n, and p are coprime -/
def coprime (m n p : ℕ) : Prop := Nat.gcd m (Nat.gcd n p) = 1

theorem quadratic_root_form :
  ∃ (m n p : ℕ), 
    (∀ x : ℝ, quadratic_equation x → root_form m n p x) ∧
    coprime m n p ∧
    n = 57 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_form_l2165_216530


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2165_216529

theorem sum_of_fractions : (7 : ℚ) / 12 + (3 : ℚ) / 8 = (23 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2165_216529


namespace NUMINAMATH_CALUDE_total_cars_produced_l2165_216546

/-- Given that a car company produced 3,884 cars in North America and 2,871 cars in Europe,
    prove that the total number of cars produced is 6,755. -/
theorem total_cars_produced (north_america : ℕ) (europe : ℕ) 
  (h1 : north_america = 3884) (h2 : europe = 2871) : 
  north_america + europe = 6755 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_produced_l2165_216546


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2165_216507

theorem right_triangle_side_length 
  (A B C : Real) 
  (BC : Real) 
  (h1 : A = Real.pi / 2) 
  (h2 : BC = 10) 
  (h3 : Real.tan C = 3 * Real.cos B) : 
  ∃ AB : Real, AB = 20 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2165_216507


namespace NUMINAMATH_CALUDE_olivers_money_l2165_216568

/-- Oliver's money calculation -/
theorem olivers_money (initial_amount spent_amount received_amount : ℕ) :
  initial_amount = 33 →
  spent_amount = 4 →
  received_amount = 32 →
  initial_amount - spent_amount + received_amount = 61 :=
by
  sorry

end NUMINAMATH_CALUDE_olivers_money_l2165_216568


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l2165_216531

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def point_on_parabola (A : ℝ × ℝ) : Prop :=
  parabola A.1 A.2

-- Define the dot product condition
def dot_product_condition (A : ℝ × ℝ) : Prop :=
  let O := (0, 0)
  let F := focus
  (A.1 - O.1) * (F.1 - A.1) + (A.2 - O.2) * (F.2 - A.2) = -4

-- Theorem statement
theorem parabola_point_coordinates :
  ∀ A : ℝ × ℝ,
  point_on_parabola A →
  dot_product_condition A →
  (A = (1, 2) ∨ A = (1, -2)) :=
sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l2165_216531


namespace NUMINAMATH_CALUDE_luncheon_cost_theorem_l2165_216511

/-- Represents the cost of a luncheon item -/
structure LuncheonItem where
  price : ℚ

/-- Represents a luncheon order -/
structure Luncheon where
  sandwiches : ℕ
  coffee : ℕ
  pie : ℕ
  total : ℚ

/-- The theorem to be proved -/
theorem luncheon_cost_theorem (s : LuncheonItem) (c : LuncheonItem) (p : LuncheonItem) 
  (l1 : Luncheon) (l2 : Luncheon) : 
  l1.sandwiches = 2 ∧ l1.coffee = 5 ∧ l1.pie = 2 ∧ l1.total = 25/4 ∧
  l2.sandwiches = 5 ∧ l2.coffee = 8 ∧ l2.pie = 3 ∧ l2.total = 121/10 →
  s.price + c.price + p.price = 31/20 := by
  sorry

#eval 31/20  -- This should evaluate to 1.55

end NUMINAMATH_CALUDE_luncheon_cost_theorem_l2165_216511


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2165_216522

def a (n : ℕ+) : ℤ := 2^n.val - (-1)^n.val

theorem arithmetic_sequence_properties :
  (∃ n : ℕ+, a n + a (n + 2) = 2 * a (n + 1) ∧ n = 2) ∧
  (∃ n₂ n₃ : ℕ+, n₂ < n₃ ∧ a 1 + a n₃ = 2 * a n₂ ∧ n₃ - n₂ = 1) ∧
  (∀ t : ℕ+, t > 3 →
    ¬∃ (n : Fin t → ℕ+), (∀ i j : Fin t, i < j → n i < n j) ∧
      (∀ i : Fin (t - 2), 2 * a (n (i + 1)) = a (n i) + a (n (i + 2)))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2165_216522


namespace NUMINAMATH_CALUDE_dataset_manipulation_result_l2165_216558

def calculate_final_dataset_size (initial_size : ℕ) : ℕ :=
  let size_after_increase := initial_size + (initial_size * 15 / 100)
  let size_after_addition := size_after_increase + 40
  let size_after_removal := size_after_addition - (size_after_addition / 6)
  let final_size := size_after_removal - (size_after_removal * 10 / 100)
  final_size

theorem dataset_manipulation_result :
  calculate_final_dataset_size 300 = 289 := by
  sorry

end NUMINAMATH_CALUDE_dataset_manipulation_result_l2165_216558


namespace NUMINAMATH_CALUDE_power_equality_l2165_216579

theorem power_equality (x : ℝ) : (1/8 : ℝ) * 2^36 = 8^x → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2165_216579


namespace NUMINAMATH_CALUDE_john_can_lift_2800_pounds_l2165_216592

-- Define the given values
def original_squat : ℝ := 135
def training_increase : ℝ := 265
def bracer_multiplier : ℝ := 7  -- 600% increase means multiplying by 7 (1 + 6)

-- Define the calculation steps
def new_squat : ℝ := original_squat + training_increase
def final_lift : ℝ := new_squat * bracer_multiplier

-- Theorem statement
theorem john_can_lift_2800_pounds : 
  final_lift = 2800 := by sorry

end NUMINAMATH_CALUDE_john_can_lift_2800_pounds_l2165_216592


namespace NUMINAMATH_CALUDE_factor_expression_l2165_216578

theorem factor_expression (y z : ℝ) : 64 - 16 * y^2 * z^2 = 16 * (2 - y*z) * (2 + y*z) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2165_216578


namespace NUMINAMATH_CALUDE_derivative_at_two_l2165_216502

open Real

theorem derivative_at_two (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x > 0, f x = x^2 + 3 * x * (deriv f 2) - log x) : 
  deriv f 2 = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_two_l2165_216502


namespace NUMINAMATH_CALUDE_shaded_area_of_partitioned_isosceles_right_triangle_l2165_216547

theorem shaded_area_of_partitioned_isosceles_right_triangle 
  (leg_length : ℝ) 
  (num_partitions : ℕ) 
  (num_shaded : ℕ) : 
  leg_length = 8 → 
  num_partitions = 16 → 
  num_shaded = 10 → 
  (1 / 2 * leg_length * leg_length) * (num_shaded / num_partitions) = 20 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_of_partitioned_isosceles_right_triangle_l2165_216547


namespace NUMINAMATH_CALUDE_fixed_fee_calculation_l2165_216545

/-- Internet subscription with fixed monthly fee and per-hour usage fee -/
structure InternetSubscription where
  fixed_fee : ℝ
  hourly_fee : ℝ

/-- Bill calculation for a given month -/
def monthly_bill (s : InternetSubscription) (hours : ℝ) : ℝ :=
  s.fixed_fee + s.hourly_fee * hours

theorem fixed_fee_calculation (s : InternetSubscription) 
  (feb_hours mar_hours : ℝ) :
  monthly_bill s feb_hours = 18.60 →
  monthly_bill s mar_hours = 30.90 →
  mar_hours = 3 * feb_hours →
  s.fixed_fee = 12.45 := by
  sorry

#eval 12.45 -- To display the result

end NUMINAMATH_CALUDE_fixed_fee_calculation_l2165_216545


namespace NUMINAMATH_CALUDE_speed_in_still_water_l2165_216587

/-- The speed of a man in still water given his upstream and downstream speeds -/
theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) :
  upstream_speed = 20 →
  downstream_speed = 80 →
  (upstream_speed + downstream_speed) / 2 = 50 := by
  sorry

#check speed_in_still_water

end NUMINAMATH_CALUDE_speed_in_still_water_l2165_216587


namespace NUMINAMATH_CALUDE_transformation_solvable_l2165_216520

/-- A transformation that replaces two numbers with their product -/
def transformation (numbers : List ℝ) (i j : Nat) : List ℝ :=
  if i < numbers.length ∧ j < numbers.length ∧ i ≠ j then
    let product := numbers[i]! * numbers[j]!
    numbers.set i product |>.set j product
  else
    numbers

/-- Predicate to check if all numbers in the list are the same -/
def allSame (numbers : List ℝ) : Prop :=
  ∀ i j, i < numbers.length → j < numbers.length → numbers[i]! = numbers[j]!

/-- The main theorem stating when the problem is solvable -/
theorem transformation_solvable (n : ℕ) :
  (∃ (numbers : List ℝ) (k : ℕ), numbers.length = n ∧ 
   ∃ (transformations : List (ℕ × ℕ)), 
     allSame (transformations.foldl (λ acc (i, j) => transformation acc i j) numbers)) ↔ 
  (n % 2 = 0 ∨ n = 1) :=
sorry

end NUMINAMATH_CALUDE_transformation_solvable_l2165_216520


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l2165_216533

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l2165_216533


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2165_216571

-- Define the properties of the polygon
def perimeter : ℝ := 150
def side_length : ℝ := 15

-- Theorem statement
theorem regular_polygon_sides : 
  perimeter / side_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2165_216571


namespace NUMINAMATH_CALUDE_passengers_at_third_station_l2165_216582

/-- Calculates the number of passengers at the third station given the initial number of passengers and the changes at each station. -/
def passengersAtThirdStation (initialPassengers : ℕ) : ℕ :=
  let afterFirstDrop := initialPassengers - initialPassengers / 3
  let afterFirstAdd := afterFirstDrop + 280
  let afterSecondDrop := afterFirstAdd - afterFirstAdd / 2
  afterSecondDrop + 12

/-- Theorem stating that given 270 initial passengers, the number of passengers at the third station is 242. -/
theorem passengers_at_third_station :
  passengersAtThirdStation 270 = 242 := by
  sorry

#eval passengersAtThirdStation 270

end NUMINAMATH_CALUDE_passengers_at_third_station_l2165_216582


namespace NUMINAMATH_CALUDE_equilateral_triangle_reflection_theorem_l2165_216583

/-- Represents a ray path in an equilateral triangle -/
structure RayPath where
  n : ℕ  -- number of reflections
  returns_to_start : Bool  -- whether the ray returns to the starting point
  passes_through_vertices : Bool  -- whether the ray passes through other vertices

/-- Checks if a number is a valid reflection count -/
def is_valid_reflection_count (n : ℕ) : Prop :=
  (n % 6 = 1 ∨ n % 6 = 5) ∧ n ≠ 5 ∧ n ≠ 17

/-- Main theorem: Characterizes valid reflection counts in an equilateral triangle -/
theorem equilateral_triangle_reflection_theorem :
  ∀ (path : RayPath),
    path.returns_to_start ∧ ¬path.passes_through_vertices ↔
    is_valid_reflection_count path.n :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_reflection_theorem_l2165_216583


namespace NUMINAMATH_CALUDE_power_two_greater_than_square_l2165_216566

theorem power_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_square_l2165_216566


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_area_ratio_l2165_216525

/-- The ratio of the perimeter to the area of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_perimeter_area_ratio :
  let side_length : ℝ := 6
  let perimeter : ℝ := 3 * side_length
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  perimeter / area = 2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_area_ratio_l2165_216525


namespace NUMINAMATH_CALUDE_solution_values_l2165_216503

/-- A quadratic function f(x) = ax^2 - 2(a+1)x + b where a and b are real numbers. -/
def f (a b x : ℝ) : ℝ := a * x^2 - 2 * (a + 1) * x + b

/-- The property that the solution set of f(x) < 0 is (1,2) -/
def solution_set_property (a b : ℝ) : Prop :=
  ∀ x, f a b x < 0 ↔ 1 < x ∧ x < 2

/-- Theorem stating that if the solution set property holds, then a = 2 and b = 4 -/
theorem solution_values (a b : ℝ) (h : solution_set_property a b) : a = 2 ∧ b = 4 := by
  sorry


end NUMINAMATH_CALUDE_solution_values_l2165_216503


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2165_216575

def A : Set ℕ := {2, 3}
def B : Set ℕ := {3, 4}

theorem union_of_A_and_B : A ∪ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2165_216575


namespace NUMINAMATH_CALUDE_simplify_expression_l2165_216562

theorem simplify_expression (x : ℝ) : (5 - 4 * x) - (2 + 5 * x) = 3 - 9 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2165_216562


namespace NUMINAMATH_CALUDE_bucket_6_5_full_l2165_216572

/-- Represents the state of water in buckets -/
structure BucketState where
  bucket_2_5 : Real
  bucket_3 : Real
  bucket_5_6 : Real
  bucket_6_5 : Real

/-- Represents the capacity of each bucket -/
def bucket_capacities : BucketState :=
  { bucket_2_5 := 2.5
  , bucket_3 := 3
  , bucket_5_6 := 5.6
  , bucket_6_5 := 6.5 }

/-- Performs the water pouring operations as described in the problem -/
def pour_water (initial : BucketState) : BucketState :=
  sorry

/-- The theorem to be proved -/
theorem bucket_6_5_full (initial : BucketState) :
  let final := pour_water initial
  final.bucket_6_5 = bucket_capacities.bucket_6_5 ∧
  ∀ ε > 0, final.bucket_6_5 + ε > bucket_capacities.bucket_6_5 :=
sorry

end NUMINAMATH_CALUDE_bucket_6_5_full_l2165_216572


namespace NUMINAMATH_CALUDE_proposition_truth_l2165_216534

theorem proposition_truth : (∀ x ∈ Set.Ioo 0 (Real.pi / 2), Real.sin x - x < 0) ∧
  ¬(∃ x₀ ∈ Set.Ioi 0, (2 : ℝ) ^ x₀ = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l2165_216534


namespace NUMINAMATH_CALUDE_state_returns_sold_l2165_216504

-- Define the prices and quantities
def federal_price : ℕ := 50
def state_price : ℕ := 30
def quarterly_price : ℕ := 80
def federal_quantity : ℕ := 60
def quarterly_quantity : ℕ := 10
def total_revenue : ℕ := 4400

-- Define the function to calculate total revenue
def calculate_revenue (state_quantity : ℕ) : ℕ :=
  federal_price * federal_quantity +
  state_price * state_quantity +
  quarterly_price * quarterly_quantity

-- Theorem statement
theorem state_returns_sold : 
  ∃ (state_quantity : ℕ), calculate_revenue state_quantity = total_revenue ∧ state_quantity = 20 := by
  sorry

end NUMINAMATH_CALUDE_state_returns_sold_l2165_216504


namespace NUMINAMATH_CALUDE_range_of_a_l2165_216553

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2*x + Real.exp x - 1 / Real.exp x

theorem range_of_a (a : ℝ) :
  (f (2*a - 3) + f (a^2) ≤ 0) ↔ -3 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2165_216553


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2165_216585

theorem inequality_solution_set (x : ℝ) : 
  (((2 * x - 3) / (x + 4) > (4 * x + 1) / (3 * x - 2)) ↔ 
  (x > -4 ∧ x < (17 - Real.sqrt 201) / 4) ∨ 
  (x > (17 + Real.sqrt 201) / 4 ∧ x < 2 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2165_216585


namespace NUMINAMATH_CALUDE_arrangement_theorem_l2165_216509

def number_of_arrangements (num_men : ℕ) (num_women : ℕ) : ℕ :=
  let group_of_four_two_men := Nat.choose num_men 2 * Nat.choose num_women 2
  let group_of_four_one_man := Nat.choose num_men 1 * Nat.choose num_women 3
  group_of_four_two_men + group_of_four_one_man

theorem arrangement_theorem :
  number_of_arrangements 5 4 = 80 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l2165_216509


namespace NUMINAMATH_CALUDE_min_value_sqrt_and_reciprocal_l2165_216548

theorem min_value_sqrt_and_reciprocal (x : ℝ) (h : x > 0) :
  4 * Real.sqrt x + 4 / x ≥ 8 ∧ ∃ y > 0, 4 * Real.sqrt y + 4 / y = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_and_reciprocal_l2165_216548


namespace NUMINAMATH_CALUDE_distribute_eq_choose_l2165_216569

/-- The number of ways to distribute n indistinguishable objects into k distinct boxes -/
def distribute (n k : ℕ+) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- Theorem stating that the number of ways to distribute n indistinguishable objects
    into k distinct boxes is equal to (n+k-1) choose (k-1) -/
theorem distribute_eq_choose (n k : ℕ+) :
  distribute n k = Nat.choose (n + k - 1) (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_distribute_eq_choose_l2165_216569


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_24_l2165_216596

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_8_with_digit_sum_24 :
  ∀ n : ℕ, is_three_digit n → n % 8 = 0 → sum_of_digits n = 24 → n ≤ 888 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_8_with_digit_sum_24_l2165_216596


namespace NUMINAMATH_CALUDE_xyz_equation_solutions_l2165_216591

theorem xyz_equation_solutions (n : ℕ+) (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  ∃! k : ℕ, k = 3 * (n + 1) ∧
  ∃ S : Finset (ℕ × ℕ × ℕ),
    S.card = k ∧
    ∀ (x y z : ℕ), (x, y, z) ∈ S ↔ 
      x > 0 ∧ y > 0 ∧ z > 0 ∧ 
      x * y * z = p ^ (n : ℕ) * (x + y + z) :=
sorry

end NUMINAMATH_CALUDE_xyz_equation_solutions_l2165_216591


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l2165_216505

-- Define the number of sides in a nonagon
def nonagon_sides : ℕ := 9

-- Define the number of diagonals in a nonagon
def nonagon_diagonals : ℕ := 27

-- Define a function to calculate combinations
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem nonagon_diagonal_intersection_probability :
  let total_diagonal_pairs := choose nonagon_diagonals 2
  let intersecting_diagonal_pairs := choose nonagon_sides 4
  (intersecting_diagonal_pairs : ℚ) / total_diagonal_pairs = 14 / 39 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l2165_216505


namespace NUMINAMATH_CALUDE_trapezoid_area_example_l2165_216580

/-- Represents a trapezoid with sides a, b, c, d where a is parallel to c -/
structure Trapezoid :=
  (a b c d : ℝ)

/-- Calculates the area of a trapezoid -/
def trapezoidArea (t : Trapezoid) : ℝ :=
  sorry

theorem trapezoid_area_example :
  let t := Trapezoid.mk 52 20 65 11
  trapezoidArea t = 594 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_example_l2165_216580


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2165_216500

/-- A geometric sequence with first term 512 and 8th term 2 has 6th term equal to 16 -/
theorem geometric_sequence_sixth_term : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) = a n * (a 8 / a 7)) →  -- Geometric sequence property
  a 1 = 512 →                            -- First term is 512
  a 8 = 2 →                              -- 8th term is 2
  a 6 = 16 :=                            -- 6th term is 16
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2165_216500


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2165_216561

theorem sum_of_fractions : 
  (1 / (2 * 3 * 4 : ℚ)) + (1 / (3 * 4 * 5 : ℚ)) + (1 / (4 * 5 * 6 : ℚ)) + 
  (1 / (5 * 6 * 7 : ℚ)) + (1 / (6 * 7 * 8 : ℚ)) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2165_216561


namespace NUMINAMATH_CALUDE_part_one_part_two_l2165_216576

-- Define the inequality
def inequality (a b x : ℝ) : Prop := a * x^2 - b ≥ 2 * x - a * x

-- Define the solution set
def solution_set (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ -1

-- Theorem for part (1)
theorem part_one (a b : ℝ) : 
  (∀ x, inequality a b x ↔ solution_set x) → a = -1 ∧ b = 2 := by sorry

-- Define the second inequality
def inequality_two (a x : ℝ) : Prop := (a * x - 2) * (x + 1) ≥ 0

-- Define the solution sets for part (2)
def solution_set_one (a x : ℝ) : Prop := 2 / a ≤ x ∧ x ≤ -1
def solution_set_two (x : ℝ) : Prop := x = -1
def solution_set_three (a x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 2 / a

-- Theorem for part (2)
theorem part_two (a : ℝ) (h : a < 0) :
  (∀ x, inequality_two a x ↔ 
    ((-2 < a ∧ a < 0 ∧ solution_set_one a x) ∨
     (a = -2 ∧ solution_set_two x) ∨
     (a < -2 ∧ solution_set_three a x))) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2165_216576


namespace NUMINAMATH_CALUDE_divisibility_condition_implies_a_geq_neg_one_l2165_216552

theorem divisibility_condition_implies_a_geq_neg_one (a : ℤ) :
  (∃ x y : ℕ+, x ≠ y ∧ (a * x * y + 1 : ℤ) ∣ (a * x^2 + 1)^2) →
  a ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_implies_a_geq_neg_one_l2165_216552


namespace NUMINAMATH_CALUDE_midpoint_product_and_distance_l2165_216593

/-- Given that C is the midpoint of segment AB, prove that xy = -12 and d = 4√5 -/
theorem midpoint_product_and_distance (x y : ℝ) :
  (4 : ℝ) = (2 + x) / 2 →
  (2 : ℝ) = (6 + y) / 2 →
  x * y = -12 ∧ Real.sqrt ((x - 2)^2 + (y - 6)^2) = 4 * Real.sqrt 5 := by
  sorry

#check midpoint_product_and_distance

end NUMINAMATH_CALUDE_midpoint_product_and_distance_l2165_216593


namespace NUMINAMATH_CALUDE_diamond_calculation_l2165_216541

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_calculation : 
  let x := diamond (diamond 3 4) 5
  let y := diamond 3 (diamond 4 5)
  x - y = -71 / 380 := by sorry

end NUMINAMATH_CALUDE_diamond_calculation_l2165_216541


namespace NUMINAMATH_CALUDE_jills_age_l2165_216573

theorem jills_age (henry_age jill_age : ℕ) : 
  henry_age + jill_age = 43 →
  henry_age - 5 = 2 * (jill_age - 5) →
  jill_age = 16 := by
sorry

end NUMINAMATH_CALUDE_jills_age_l2165_216573


namespace NUMINAMATH_CALUDE_cubic_roots_coefficients_relation_l2165_216508

theorem cubic_roots_coefficients_relation 
  (a b c d : ℝ) (x₁ x₂ x₃ : ℝ) 
  (h : a ≠ 0) 
  (h_roots : ∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) : 
  (x₁ + x₂ + x₃ = -b / a) ∧ 
  (x₁ * x₂ + x₁ * x₃ + x₂ * x₃ = c / a) ∧ 
  (x₁ * x₂ * x₃ = -d / a) := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_coefficients_relation_l2165_216508


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2165_216550

/-- The minimum value of 1/m + 1/n given the conditions -/
theorem min_value_reciprocal_sum (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1)
  (h_line : 2*m + 2*n = 1) (h_positive : m*n > 0) :
  1/m + 1/n ≥ 8 ∧ ∃ (m n : ℝ), 2*m + 2*n = 1 ∧ m*n > 0 ∧ 1/m + 1/n = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2165_216550


namespace NUMINAMATH_CALUDE_reflected_arcs_area_l2165_216524

/-- The area of the region bounded by 8 reflected arcs in a regular octagon inscribed in a circle -/
theorem reflected_arcs_area (s : ℝ) (h : s = 2) : 
  let r := Real.sqrt (2 * Real.sqrt 2)
  let sector_area := π * r^2 / 8
  let triangle_area := s^2 / 4
  let reflected_arc_area := sector_area - triangle_area
  8 * reflected_arc_area = 2 * π * Real.sqrt 2 - 8 := by
  sorry

end NUMINAMATH_CALUDE_reflected_arcs_area_l2165_216524


namespace NUMINAMATH_CALUDE_sequence_and_sum_properties_l2165_216557

def sequence_a (n : ℕ) : ℤ :=
  4 * n - 25

def sum_S (n : ℕ) : ℤ :=
  n * (sequence_a 1 + sequence_a n) / 2

theorem sequence_and_sum_properties :
  (sequence_a 3 = -13) ∧
  (∀ n > 1, sequence_a n = sequence_a (n - 1) + 4) ∧
  (sequence_a 1 = -21) ∧
  (sequence_a 2 = -17) ∧
  (∀ n, sequence_a n = 4 * n - 25) ∧
  (∀ k, sum_S 6 ≤ sum_S k) ∧
  (sum_S 6 = -66) := by
  sorry

end NUMINAMATH_CALUDE_sequence_and_sum_properties_l2165_216557


namespace NUMINAMATH_CALUDE_cookie_average_is_14_l2165_216537

def cookie_counts : List Nat := [8, 10, 12, 15, 16, 17, 20]

def average (list : List Nat) : Rat :=
  (list.sum : Rat) / list.length

theorem cookie_average_is_14 :
  average cookie_counts = 14 := by
  sorry

end NUMINAMATH_CALUDE_cookie_average_is_14_l2165_216537


namespace NUMINAMATH_CALUDE_cylindrical_can_volume_condition_l2165_216543

/-- The value of y that satisfies the volume condition for a cylindrical can --/
theorem cylindrical_can_volume_condition (π : ℝ) (h : π > 0) : 
  ∃! y : ℝ, y > 0 ∧ 
    π * (5 + y)^2 * (4 + y) = π * (5 + 2*y)^2 * 4 ∧
    y = Real.sqrt 76 - 5 := by
  sorry

end NUMINAMATH_CALUDE_cylindrical_can_volume_condition_l2165_216543


namespace NUMINAMATH_CALUDE_fib_50_mod_5_l2165_216560

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_50_mod_5 : fib 50 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_50_mod_5_l2165_216560


namespace NUMINAMATH_CALUDE_base_conversion_1729_to_base_7_l2165_216521

theorem base_conversion_1729_to_base_7 :
  (5 * 7^3 + 0 * 7^2 + 2 * 7^1 + 0 * 7^0 : ℕ) = 1729 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_1729_to_base_7_l2165_216521


namespace NUMINAMATH_CALUDE_job_filling_combinations_l2165_216565

def num_resumes : ℕ := 30
def num_unsuitable : ℕ := 20
def num_job_openings : ℕ := 5

theorem job_filling_combinations :
  (num_resumes - num_unsuitable).factorial / (num_resumes - num_unsuitable - num_job_openings).factorial = 30240 :=
by sorry

end NUMINAMATH_CALUDE_job_filling_combinations_l2165_216565


namespace NUMINAMATH_CALUDE_recreation_spending_comparison_l2165_216519

theorem recreation_spending_comparison (last_week_wages : ℝ) : 
  let last_week_recreation := 0.15 * last_week_wages
  let this_week_wages := 0.90 * last_week_wages
  let this_week_recreation := 0.30 * this_week_wages
  (this_week_recreation / last_week_recreation) * 100 = 180 := by
sorry

end NUMINAMATH_CALUDE_recreation_spending_comparison_l2165_216519


namespace NUMINAMATH_CALUDE_journey_time_proof_l2165_216535

/-- Proves that a journey of 336 km, with the first half traveled at 21 km/hr
    and the second half at 24 km/hr, takes 15 hours to complete. -/
theorem journey_time_proof (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_distance = 336 ∧ speed1 = 21 ∧ speed2 = 24 →
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_proof_l2165_216535


namespace NUMINAMATH_CALUDE_multiply_add_equality_l2165_216549

theorem multiply_add_equality : (-3) * 2 + 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_add_equality_l2165_216549


namespace NUMINAMATH_CALUDE_remainder_divisibility_l2165_216539

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 13 * k + 4) → (∃ m : ℤ, N = 39 * m + 4) :=
sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l2165_216539


namespace NUMINAMATH_CALUDE_ladybugs_with_spots_l2165_216599

theorem ladybugs_with_spots (total : ℕ) (without_spots : ℕ) (with_spots : ℕ) : 
  total = 67082 → without_spots = 54912 → total = with_spots + without_spots → 
  with_spots = 12170 := by
sorry

end NUMINAMATH_CALUDE_ladybugs_with_spots_l2165_216599


namespace NUMINAMATH_CALUDE_green_shirt_percentage_l2165_216532

theorem green_shirt_percentage 
  (total_students : ℕ) 
  (blue_percent : ℚ) 
  (red_percent : ℚ) 
  (other_count : ℕ) 
  (h1 : total_students = 900) 
  (h2 : blue_percent = 44/100) 
  (h3 : red_percent = 28/100) 
  (h4 : other_count = 162) :
  (total_students - (blue_percent * total_students + red_percent * total_students + other_count : ℚ)) / total_students = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_green_shirt_percentage_l2165_216532


namespace NUMINAMATH_CALUDE_min_value_expression_l2165_216574

theorem min_value_expression (a x : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 15) (h3 : a ≤ x) (h4 : x ≤ 15) :
  ∃ (min_x : ℝ), min_x = 15 ∧
    ∀ y, a ≤ y ∧ y ≤ 15 →
      |y - a| + |y - 15| + |y - a - 15| ≥ |min_x - a| + |min_x - 15| + |min_x - a - 15| ∧
      |min_x - a| + |min_x - 15| + |min_x - a - 15| = 15 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2165_216574
