import Mathlib

namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l72_7285

/-- Stewart Farm problem -/
theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ) (sheep_food horse_food : ℝ),
  sheep * 7 = horses →
  horse_food = 230 →
  horses * horse_food = 12880 →
  sheep_food = 150 →
  sheep * sheep_food = 6300 →
  sheep = 8 := by
sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l72_7285


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l72_7207

/-- Given a conic section with equation x²/m + y² = 1 and eccentricity √7, prove that m = -6 -/
theorem conic_section_eccentricity (m : ℝ) : 
  (∃ (x y : ℝ), x^2/m + y^2 = 1) →  -- Condition 1: Conic section equation
  (∃ (e : ℝ), e = Real.sqrt 7 ∧ e^2 = (1 - m)/1) →  -- Condition 2: Eccentricity
  m = -6 := by sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l72_7207


namespace NUMINAMATH_CALUDE_calculate_expression_l72_7213

theorem calculate_expression : 2^2 + |-3| - Real.sqrt 25 = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l72_7213


namespace NUMINAMATH_CALUDE_triangle_area_and_length_l72_7239

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively, and point D as the midpoint of BC -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ × ℝ

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem triangle_area_and_length (t : Triangle) :
  (t.c * Real.cos t.B = Real.sqrt 3 * t.b * Real.sin t.C) →
  (t.a^2 * Real.sin t.C = 4 * Real.sqrt 3 * Real.sin t.A) →
  (area t = Real.sqrt 3) ∧
  (t.a = 2 * Real.sqrt 3 → t.b = Real.sqrt 7 → t.c > t.b →
   distance (0, 0) t.D = Real.sqrt 13) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_and_length_l72_7239


namespace NUMINAMATH_CALUDE_root_transformation_l72_7216

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 3*r₁^2 + 8 = 0) ∧ 
  (r₂^3 - 3*r₂^2 + 8 = 0) ∧ 
  (r₃^3 - 3*r₃^2 + 8 = 0) → 
  ((2*r₁)^3 - 6*(2*r₁)^2 + 64 = 0) ∧
  ((2*r₂)^3 - 6*(2*r₂)^2 + 64 = 0) ∧
  ((2*r₃)^3 - 6*(2*r₃)^2 + 64 = 0) := by
sorry

end NUMINAMATH_CALUDE_root_transformation_l72_7216


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l72_7237

theorem smallest_undefined_inverse (a : ℕ) : 
  (∀ b < 14, (Nat.gcd b 70 = 1 ∨ Nat.gcd b 84 = 1)) ∧ 
  (Nat.gcd 14 70 > 1 ∧ Nat.gcd 14 84 > 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l72_7237


namespace NUMINAMATH_CALUDE_max_blocks_in_box_l72_7280

/-- Represents the dimensions of a rectangular box. -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of a block. -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of blocks that can fit in a box. -/
def maxBlocksFit (box : BoxDimensions) (block : BlockDimensions) : ℕ :=
  let boxVolume := box.length * box.width * box.height
  let blockVolume := block.length * block.width * block.height
  boxVolume / blockVolume

/-- Theorem stating that the maximum number of 3×1×1 blocks that can fit in a 4×3×2 box is 8. -/
theorem max_blocks_in_box :
  let box := BoxDimensions.mk 4 3 2
  let block := BlockDimensions.mk 3 1 1
  maxBlocksFit box block = 8 := by
  sorry

#eval maxBlocksFit (BoxDimensions.mk 4 3 2) (BlockDimensions.mk 3 1 1)

end NUMINAMATH_CALUDE_max_blocks_in_box_l72_7280


namespace NUMINAMATH_CALUDE_geese_count_l72_7283

def geese_problem (n : ℕ) : Prop :=
  -- The number of geese is an integer (implied by ℕ)
  -- After each lake, the number of remaining geese is an integer
  (∀ k : ℕ, k ≤ 7 → ∃ m : ℕ, n * 2^(7 - k) - (2^(7 - k) - 1) = m) ∧
  -- The process continues for exactly 7 lakes
  -- At each lake, half of the remaining geese plus half a goose land (implied by the formula)
  -- After 7 lakes, no geese remain
  n * 2^0 - (2^0 - 1) = 0

theorem geese_count : ∃ n : ℕ, geese_problem n ∧ n = 127 := by
  sorry

end NUMINAMATH_CALUDE_geese_count_l72_7283


namespace NUMINAMATH_CALUDE_investment_proof_l72_7244

def total_investment : ℝ := 3000
def part_one_investment : ℝ := 800
def part_one_interest_rate : ℝ := 0.10
def total_yearly_interest : ℝ := 256

theorem investment_proof :
  ∃ (part_two_interest_rate : ℝ),
    part_one_investment * part_one_interest_rate +
    (total_investment - part_one_investment) * part_two_interest_rate =
    total_yearly_interest :=
by sorry

end NUMINAMATH_CALUDE_investment_proof_l72_7244


namespace NUMINAMATH_CALUDE_gavin_blue_shirts_l72_7297

/-- The number of blue shirts Gavin has -/
def blue_shirts (total : ℕ) (green : ℕ) : ℕ := total - green

theorem gavin_blue_shirts :
  let total_shirts : ℕ := 23
  let green_shirts : ℕ := 17
  blue_shirts total_shirts green_shirts = 6 := by
sorry

end NUMINAMATH_CALUDE_gavin_blue_shirts_l72_7297


namespace NUMINAMATH_CALUDE_points_per_game_l72_7298

theorem points_per_game (total_points : ℕ) (num_games : ℕ) (points_per_game : ℕ) : 
  total_points = 24 → 
  num_games = 6 → 
  total_points = num_games * points_per_game → 
  points_per_game = 4 := by
sorry

end NUMINAMATH_CALUDE_points_per_game_l72_7298


namespace NUMINAMATH_CALUDE_largest_integer_m_l72_7267

theorem largest_integer_m (x m : ℝ) : 
  (3 : ℝ) / 3 + 2 * m < -3 → 
  ∀ k : ℤ, (k : ℝ) > m → k ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_largest_integer_m_l72_7267


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_all_ones_l72_7201

def is_all_ones (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 1

theorem smallest_multiplier_for_all_ones :
  ∃! N : ℕ, (N > 0) ∧ 
    is_all_ones (999999 * N) ∧
    (∀ m : ℕ, m > 0 → is_all_ones (999999 * m) → N ≤ m) ∧
    N = 111112 := by sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_all_ones_l72_7201


namespace NUMINAMATH_CALUDE_product_sum_equality_l72_7258

theorem product_sum_equality : 15 * 35 + 45 * 15 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_equality_l72_7258


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l72_7223

/-- Given a geometric sequence {a_n} with S_3 = 13/9 and S_6 = 364/9, 
    prove that a_n = (1/6) * 3^(n-1) for all n ≥ 1 -/
theorem geometric_sequence_formula (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  S 3 = 13/9 →
  S 6 = 364/9 →
  (∀ n, S n = (a 1 * (1 - (a 2 / a 1)^n)) / (1 - a 2 / a 1)) →  -- Sum formula for geometric sequence
  ∀ n : ℕ, n ≥ 1 → a n = (1/6) * 3^(n-1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l72_7223


namespace NUMINAMATH_CALUDE_eight_sum_product_theorem_l72_7227

theorem eight_sum_product_theorem : 
  ∃ (a b c d e f g h : ℤ), 
    (a + b + c + d + e + f + g + h = 8) ∧ 
    (a * b * c * d * e * f * g * h = 8) :=
sorry

end NUMINAMATH_CALUDE_eight_sum_product_theorem_l72_7227


namespace NUMINAMATH_CALUDE_function_periodicity_l72_7251

variable (a : ℝ)
variable (f : ℝ → ℝ)

theorem function_periodicity
  (h : ∀ x, f (x + a) = (1 + f x) / (1 - f x)) :
  ∀ x, f (x + 4 * a) = f x :=
by sorry

end NUMINAMATH_CALUDE_function_periodicity_l72_7251


namespace NUMINAMATH_CALUDE_partnership_investment_l72_7220

/-- Given the investments of partners A and B, the total profit, and A's share of the profit,
    calculate the investment of partner C in a partnership business. -/
theorem partnership_investment (a b total_profit a_profit : ℕ) (ha : a = 6300) (hb : b = 4200) 
    (h_total_profit : total_profit = 12200) (h_a_profit : a_profit = 3660) : 
    ∃ c : ℕ, c = 10490 ∧ a * total_profit = a_profit * (a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_partnership_investment_l72_7220


namespace NUMINAMATH_CALUDE_percentage_difference_l72_7265

theorem percentage_difference (C : ℝ) (A B : ℝ) 
  (hA : A = 0.7 * C) 
  (hB : B = 0.63 * C) : 
  (A - B) / A = 0.1 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l72_7265


namespace NUMINAMATH_CALUDE_binomial_17_4_l72_7292

theorem binomial_17_4 : Nat.choose 17 4 = 2380 := by
  sorry

end NUMINAMATH_CALUDE_binomial_17_4_l72_7292


namespace NUMINAMATH_CALUDE_square_plus_one_positive_l72_7230

theorem square_plus_one_positive (a : ℝ) : 0 < a^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_positive_l72_7230


namespace NUMINAMATH_CALUDE_figure_area_bound_l72_7287

-- Define the unit square
def UnitSquare : Set (Real × Real) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the property of the figure
def ValidFigure (F : Set (Real × Real)) : Prop :=
  F ⊆ UnitSquare ∧
  ∀ p q : Real × Real, p ∈ F → q ∈ F → dist p q ≠ 0.001

-- Define the area of a set
noncomputable def area (S : Set (Real × Real)) : Real :=
  sorry

-- State the theorem
theorem figure_area_bound {F : Set (Real × Real)} (hF : ValidFigure F) :
  area F ≤ 0.34 ∧ area F ≤ 0.287 :=
sorry

end NUMINAMATH_CALUDE_figure_area_bound_l72_7287


namespace NUMINAMATH_CALUDE_total_cost_calculation_l72_7233

def regular_admission : ℚ := 8
def early_discount_percentage : ℚ := 25 / 100
def student_discount_percentage : ℚ := 10 / 100
def total_people : ℕ := 6
def students : ℕ := 2

def discounted_price : ℚ := regular_admission * (1 - early_discount_percentage)
def student_price : ℚ := discounted_price * (1 - student_discount_percentage)

theorem total_cost_calculation :
  let non_student_cost := (total_people - students) * discounted_price
  let student_cost := students * student_price
  non_student_cost + student_cost = 348 / 10 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l72_7233


namespace NUMINAMATH_CALUDE_factorial_ratio_l72_7281

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 10 = 132 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l72_7281


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l72_7203

theorem quadratic_inequality_solution_set (a b c : ℝ) (h1 : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c < 0) → (a < 0 ∧ b^2 - 4*a*c < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l72_7203


namespace NUMINAMATH_CALUDE_little_zeta_name_combinations_l72_7253

theorem little_zeta_name_combinations : ∃ n : ℕ, n = 300 ∧ 
  n = (Finset.range 26).sum (λ i => 25 - i) := by
  sorry

end NUMINAMATH_CALUDE_little_zeta_name_combinations_l72_7253


namespace NUMINAMATH_CALUDE_shifted_quadratic_sum_l72_7249

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 5

/-- The shifted quadratic function -/
def g (x : ℝ) : ℝ := f (x - 3)

/-- The coefficients of the shifted function -/
def a : ℝ := 3
def b : ℝ := -20
def c : ℝ := 38

theorem shifted_quadratic_sum :
  g x = a * x^2 + b * x + c ∧ a + b + c = 21 := by sorry

end NUMINAMATH_CALUDE_shifted_quadratic_sum_l72_7249


namespace NUMINAMATH_CALUDE_square_condition_l72_7247

def is_square (x : ℕ) : Prop := ∃ t : ℕ, x = t^2

def floor_div (n m : ℕ) : ℕ := n / m

def expression (n : ℕ) : ℕ :=
  let k := Nat.log2 n
  (List.range (k+1)).foldl (λ acc i => acc * floor_div n (2^i)) 1 + 2 * 4^(k / 2)

theorem square_condition (n : ℕ) : 
  n > 0 → 
  (∃ k : ℕ, 2^k ≤ n ∧ n < 2^(k+1)) → 
  is_square (expression n) → 
  n = 2 ∨ n = 4 := by
sorry

#eval expression 2  -- Expected: 4 (which is 2^2)
#eval expression 4  -- Expected: 16 (which is 4^2)

end NUMINAMATH_CALUDE_square_condition_l72_7247


namespace NUMINAMATH_CALUDE_first_three_average_l72_7238

theorem first_three_average (a b c d : ℝ) : 
  a = 33 →
  d = 18 →
  (b + c + d) / 3 = 15 →
  (a + b + c) / 3 = 20 := by
sorry

end NUMINAMATH_CALUDE_first_three_average_l72_7238


namespace NUMINAMATH_CALUDE_davids_biology_marks_l72_7226

/-- Given David's marks in four subjects and his average marks, calculate his marks in Biology. -/
theorem davids_biology_marks
  (english_marks : ℕ)
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (average_marks : ℚ)
  (h1 : english_marks = 96)
  (h2 : math_marks = 98)
  (h3 : physics_marks = 99)
  (h4 : chemistry_marks = 100)
  (h5 : average_marks = 98.2)
  (h6 : average_marks = (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / 5) :
  biology_marks = 98 := by
  sorry

#check davids_biology_marks

end NUMINAMATH_CALUDE_davids_biology_marks_l72_7226


namespace NUMINAMATH_CALUDE_matrix_P_satisfies_conditions_l72_7221

theorem matrix_P_satisfies_conditions : 
  let P : Matrix (Fin 2) (Fin 2) ℚ := !![2, -2/3; 3, -4]
  (P.mulVec ![4, 0] = ![8, 12]) ∧ 
  (P.mulVec ![2, -3] = ![2, -6]) := by
  sorry

end NUMINAMATH_CALUDE_matrix_P_satisfies_conditions_l72_7221


namespace NUMINAMATH_CALUDE_hotel_loss_calculation_l72_7290

def hotel_loss (expenses : ℝ) (payment_ratio : ℝ) : ℝ :=
  expenses - (payment_ratio * expenses)

theorem hotel_loss_calculation (expenses : ℝ) (payment_ratio : ℝ) 
  (h1 : expenses = 100)
  (h2 : payment_ratio = 3/4) :
  hotel_loss expenses payment_ratio = 25 := by
  sorry

end NUMINAMATH_CALUDE_hotel_loss_calculation_l72_7290


namespace NUMINAMATH_CALUDE_factorize_quadratic_l72_7234

theorem factorize_quadratic (a : ℝ) : a^2 - 8*a + 15 = (a-3)*(a-5) := by
  sorry

end NUMINAMATH_CALUDE_factorize_quadratic_l72_7234


namespace NUMINAMATH_CALUDE_xyz_mod_9_l72_7293

theorem xyz_mod_9 (x y z : ℕ) : 
  x < 9 → y < 9 → z < 9 →
  (x + 3*y + 2*z) % 9 = 0 →
  (3*x + 2*y + z) % 9 = 5 →
  (2*x + y + 3*z) % 9 = 5 →
  (x*y*z) % 9 = 0 := by
sorry

end NUMINAMATH_CALUDE_xyz_mod_9_l72_7293


namespace NUMINAMATH_CALUDE_uniform_rod_weight_l72_7204

/-- Represents the weight of a uniform rod -/
def rod_weight (length : ℝ) (weight_per_meter : ℝ) : ℝ :=
  length * weight_per_meter

/-- Theorem: For a uniform rod where 9 m weighs 34.2 kg, 11.25 m of the same rod weighs 42.75 kg -/
theorem uniform_rod_weight :
  ∀ (weight_per_meter : ℝ),
    rod_weight 9 weight_per_meter = 34.2 →
    rod_weight 11.25 weight_per_meter = 42.75 := by
  sorry

end NUMINAMATH_CALUDE_uniform_rod_weight_l72_7204


namespace NUMINAMATH_CALUDE_negative_abs_negative_three_l72_7210

theorem negative_abs_negative_three : -|-3| = -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_negative_three_l72_7210


namespace NUMINAMATH_CALUDE_comic_collection_overtake_l72_7212

/-- The number of months after which LaShawn's collection becomes 1.5 times Kymbrea's --/
def months_to_overtake : ℕ := 70

/-- Kymbrea's initial number of comic books --/
def kymbrea_initial : ℕ := 40

/-- LaShawn's initial number of comic books --/
def lashawn_initial : ℕ := 25

/-- Kymbrea's monthly collection rate --/
def kymbrea_rate : ℕ := 3

/-- LaShawn's monthly collection rate --/
def lashawn_rate : ℕ := 5

/-- Theorem stating that after the specified number of months, 
    LaShawn's collection is 1.5 times Kymbrea's --/
theorem comic_collection_overtake :
  (lashawn_initial + lashawn_rate * months_to_overtake : ℚ) = 
  1.5 * (kymbrea_initial + kymbrea_rate * months_to_overtake) :=
by sorry

end NUMINAMATH_CALUDE_comic_collection_overtake_l72_7212


namespace NUMINAMATH_CALUDE_tv_price_reduction_l72_7225

theorem tv_price_reduction (x : ℝ) : 
  (1 - x/100)^2 = 1 - 19/100 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_reduction_l72_7225


namespace NUMINAMATH_CALUDE_three_digit_difference_l72_7288

theorem three_digit_difference (a b c : ℕ) (h1 : a ≥ 1) (h2 : a ≤ 9) (h3 : b ≥ 0) (h4 : b ≤ 9) (h5 : c ≥ 0) (h6 : c ≤ 9) (h7 : a = c + 2) :
  (100 * a + 10 * b + c) - (100 * c + 10 * b + a) = 198 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_difference_l72_7288


namespace NUMINAMATH_CALUDE_sawing_time_l72_7263

/-- Given that sawing a steel bar into 2 pieces takes 2 minutes,
    this theorem proves that sawing the same bar into 6 pieces takes 10 minutes. -/
theorem sawing_time (time_for_two_pieces : ℕ) (pieces : ℕ) : 
  time_for_two_pieces = 2 → pieces = 6 → (pieces - 1) * (time_for_two_pieces / (2 - 1)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sawing_time_l72_7263


namespace NUMINAMATH_CALUDE_cookie_problem_solution_l72_7241

/-- Represents the number of cookies decorated by each person in one cycle -/
structure DecoratingCycle where
  grandmother : ℕ
  mary : ℕ
  john : ℕ

/-- Represents the problem setup -/
structure CookieDecoratingProblem where
  cycle : DecoratingCycle
  trays : ℕ
  cookies_per_tray : ℕ
  grandmother_time_per_cookie : ℕ

def solve_cookie_problem (problem : CookieDecoratingProblem) :
  (ℕ × ℕ × ℕ) :=
sorry

theorem cookie_problem_solution
  (problem : CookieDecoratingProblem)
  (h_cycle : problem.cycle = ⟨5, 3, 2⟩)
  (h_trays : problem.trays = 5)
  (h_cookies_per_tray : problem.cookies_per_tray = 12)
  (h_grandmother_time : problem.grandmother_time_per_cookie = 4) :
  solve_cookie_problem problem = (4, 140, 40) :=
sorry

end NUMINAMATH_CALUDE_cookie_problem_solution_l72_7241


namespace NUMINAMATH_CALUDE_least_addend_for_divisibility_least_addend_for_1156_and_97_l72_7250

theorem least_addend_for_divisibility (n m : ℕ) (h : n > 0) : 
  ∃ (x : ℕ), (n + x) % m = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % m ≠ 0 :=
sorry

theorem least_addend_for_1156_and_97 : 
  ∃ (x : ℕ), (1156 + x) % 97 = 0 ∧ ∀ (y : ℕ), y < x → (1156 + y) % 97 ≠ 0 ∧ x = 8 :=
sorry

end NUMINAMATH_CALUDE_least_addend_for_divisibility_least_addend_for_1156_and_97_l72_7250


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_reciprocals_of_first_four_primes_l72_7268

def first_four_primes : List Nat := [2, 3, 5, 7]

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  let reciprocals := first_four_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length) = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_reciprocals_of_first_four_primes_l72_7268


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l72_7257

def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x^2 - 3*x - 4 < 0}

theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x : ℝ | 1 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l72_7257


namespace NUMINAMATH_CALUDE_remainder_equality_l72_7255

def r (n : ℕ) : ℕ := n % 6

theorem remainder_equality (n : ℕ) : 
  r (2 * n + 3) = r (5 * n + 6) ↔ ∃ k : ℤ, n = 2 * k - 1 := by sorry

end NUMINAMATH_CALUDE_remainder_equality_l72_7255


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l72_7296

/-- The volume of a cube given its space diagonal -/
theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 6 * Real.sqrt 3) :
  (d / Real.sqrt 3) ^ 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l72_7296


namespace NUMINAMATH_CALUDE_average_visitors_theorem_l72_7218

/-- Calculates the average number of visitors per day in a 30-day month starting on Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let numSundays := 4
  let numOtherDays := 30 - numSundays
  let totalVisitors := numSundays * sundayVisitors + numOtherDays * otherDayVisitors
  totalVisitors / 30

/-- Proves that the average number of visitors per day is 188 given the specified conditions -/
theorem average_visitors_theorem :
  averageVisitorsPerDay 500 140 = 188 := by
  sorry

end NUMINAMATH_CALUDE_average_visitors_theorem_l72_7218


namespace NUMINAMATH_CALUDE_friction_force_on_rotated_board_l72_7272

/-- The friction force on a block on a rotated rectangular board -/
theorem friction_force_on_rotated_board 
  (m g : ℝ) 
  (α β : ℝ) 
  (h_α_acute : 0 < α ∧ α < π / 2) 
  (h_β_acute : 0 < β ∧ β < π / 2) :
  ∃ F : ℝ, F = m * g * Real.sqrt (1 - Real.cos α ^ 2 * Real.cos β ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_friction_force_on_rotated_board_l72_7272


namespace NUMINAMATH_CALUDE_green_blue_difference_l72_7235

/-- Represents the number of parts for each color in the ratio --/
structure ColorRatio :=
  (blue : ℕ)
  (yellow : ℕ)
  (green : ℕ)
  (red : ℕ)

/-- Calculates the total number of parts in the ratio --/
def totalParts (ratio : ColorRatio) : ℕ :=
  ratio.blue + ratio.yellow + ratio.green + ratio.red

/-- Calculates the number of disks for a given color based on the ratio and total disks --/
def disksPerColor (ratio : ColorRatio) (color : ℕ) (totalDisks : ℕ) : ℕ :=
  color * (totalDisks / totalParts ratio)

theorem green_blue_difference (totalDisks : ℕ) (ratio : ColorRatio) :
  totalDisks = 180 →
  ratio = ⟨3, 7, 8, 9⟩ →
  disksPerColor ratio ratio.green totalDisks - disksPerColor ratio ratio.blue totalDisks = 35 :=
by sorry

end NUMINAMATH_CALUDE_green_blue_difference_l72_7235


namespace NUMINAMATH_CALUDE_x_squared_positive_necessary_not_sufficient_l72_7205

theorem x_squared_positive_necessary_not_sufficient :
  (∀ x : ℝ, x > 0 → x^2 > 0) ∧
  (∃ x : ℝ, x^2 > 0 ∧ x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_positive_necessary_not_sufficient_l72_7205


namespace NUMINAMATH_CALUDE_parabola_h_values_l72_7295

/-- Represents a parabola of the form y = -(x - h)² -/
def Parabola (h : ℝ) : ℝ → ℝ := fun x ↦ -((x - h)^2)

/-- The domain of x values -/
def Domain : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

theorem parabola_h_values (h : ℝ) :
  (∀ x ∈ Domain, Parabola h x ≤ -1) ∧
  (∃ x ∈ Domain, Parabola h x = -1) →
  h = 2 ∨ h = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_h_values_l72_7295


namespace NUMINAMATH_CALUDE_friendly_iff_ge_seven_l72_7254

def is_friendly (n : ℕ) : Prop :=
  n ≥ 2 ∧
  ∃ (A : Fin n → Set (Fin n)),
    (∀ i, i ∉ A i) ∧
    (∀ i j, i ≠ j → (i ∈ A j ↔ j ∉ A i)) ∧
    (∀ i j, (A i ∩ A j).Nonempty)

theorem friendly_iff_ge_seven :
  ∀ n : ℕ, is_friendly n ↔ n ≥ 7 := by sorry

end NUMINAMATH_CALUDE_friendly_iff_ge_seven_l72_7254


namespace NUMINAMATH_CALUDE_largest_satisfying_number_l72_7228

/-- A function that returns all possible two-digit numbers from three digits -/
def twoDigitNumbers (a b c : Nat) : List Nat :=
  [10*a+b, 10*a+c, 10*b+a, 10*b+c, 10*c+a, 10*c+b]

/-- The property that a three-digit number satisfies the given conditions -/
def satisfiesCondition (n : Nat) : Prop :=
  ∃ a b c : Nat,
    n = 100*a + 10*b + c ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (twoDigitNumbers a b c).sum = n

theorem largest_satisfying_number :
  satisfiesCondition 396 ∧
  ∀ m : Nat, satisfiesCondition m → m ≤ 396 :=
sorry

end NUMINAMATH_CALUDE_largest_satisfying_number_l72_7228


namespace NUMINAMATH_CALUDE_c_range_l72_7214

def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y

def q (c : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (1/2) 2 → x + 1/x > 1/c

def range_c (c : ℝ) : Prop := (0 < c ∧ c ≤ 1/2) ∨ c ≥ 1

theorem c_range (c : ℝ) (h1 : c > 0) (h2 : (p c ∨ q c) ∧ ¬(p c ∧ q c)) : range_c c := by
  sorry

end NUMINAMATH_CALUDE_c_range_l72_7214


namespace NUMINAMATH_CALUDE_missing_number_l72_7270

theorem missing_number (n : ℕ) : 
  (∀ k : ℕ, k < n → k * (k + 1) / 2 ≤ 575) ∧ 
  (n * (n + 1) / 2 > 575) → 
  n * (n + 1) / 2 - 575 = 20 := by
sorry

end NUMINAMATH_CALUDE_missing_number_l72_7270


namespace NUMINAMATH_CALUDE_bike_ride_time_l72_7275

/-- Given a constant speed where 2 miles are covered in 6 minutes,
    prove that the time required to travel 5 miles at the same speed is 15 minutes. -/
theorem bike_ride_time (speed : ℝ) (h1 : speed > 0) (h2 : 2 / speed = 6) : 5 / speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_bike_ride_time_l72_7275


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l72_7248

def P : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def Q : Set (ℝ × ℝ) := {q | q.1 - q.2 = 4}

theorem intersection_of_P_and_Q : P ∩ Q = {(3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l72_7248


namespace NUMINAMATH_CALUDE_production_order_machines_correct_initial_machines_l72_7261

/-- The number of machines initially used to complete a production order -/
def initial_machines : ℕ := 3

/-- The time (in hours) to complete the order with the initial number of machines -/
def initial_time : ℕ := 44

/-- The time (in hours) to complete the order with one additional machine -/
def reduced_time : ℕ := 33

/-- The production rate of a single machine (assumed to be constant) -/
def machine_rate : ℚ := 1 / initial_machines / initial_time

theorem production_order_machines :
  (initial_machines * machine_rate * initial_time : ℚ) =
  ((initial_machines + 1) * machine_rate * reduced_time : ℚ) :=
sorry

theorem correct_initial_machines :
  initial_machines = 3 :=
sorry

end NUMINAMATH_CALUDE_production_order_machines_correct_initial_machines_l72_7261


namespace NUMINAMATH_CALUDE_church_trip_distance_l72_7206

def trip_distance (speed1 speed2 speed3 : Real) (time : Real) : Real :=
  (speed1 * time + speed2 * time + speed3 * time)

theorem church_trip_distance :
  let speed1 : Real := 16
  let speed2 : Real := 12
  let speed3 : Real := 20
  let time : Real := 15 / 60
  trip_distance speed1 speed2 speed3 time = 12 := by
  sorry

end NUMINAMATH_CALUDE_church_trip_distance_l72_7206


namespace NUMINAMATH_CALUDE_nina_running_distance_l72_7278

theorem nina_running_distance : 
  let first_run : ℝ := 0.08
  let second_run_part1 : ℝ := 0.08
  let second_run_part2 : ℝ := 0.67
  first_run + second_run_part1 + second_run_part2 = 0.83 := by
sorry

end NUMINAMATH_CALUDE_nina_running_distance_l72_7278


namespace NUMINAMATH_CALUDE_fly_path_bounded_l72_7260

/-- Represents a right triangle room -/
structure RightTriangleRoom where
  hypotenuse : ℝ
  hypotenuse_positive : hypotenuse > 0

/-- Represents a fly's path in the room -/
structure FlyPath where
  room : RightTriangleRoom
  num_turns : ℕ
  start_in_acute_angle : Bool

/-- Theorem: A fly's path in a right triangle room with hypotenuse 5 meters and 10 turns cannot exceed 10 meters -/
theorem fly_path_bounded (path : FlyPath) 
  (h1 : path.room.hypotenuse = 5)
  (h2 : path.num_turns = 10)
  (h3 : path.start_in_acute_angle = true) :
  ∃ (max_distance : ℝ), max_distance ≤ 10 ∧ 
  ∀ (actual_distance : ℝ), actual_distance ≤ max_distance :=
sorry

end NUMINAMATH_CALUDE_fly_path_bounded_l72_7260


namespace NUMINAMATH_CALUDE_biased_coin_probability_l72_7224

/-- The probability of getting exactly k successes in n trials of a binomial experiment -/
def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 9 heads in 12 flips of a biased coin with 1/3 probability of landing heads -/
theorem biased_coin_probability : 
  binomialProbability 12 9 (1/3) = 1760/531441 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l72_7224


namespace NUMINAMATH_CALUDE_button_numbers_l72_7277

theorem button_numbers (x y : ℕ) 
  (h1 : y - x = 480) 
  (h2 : y = 4 * x + 30) : 
  y = 630 := by
  sorry

end NUMINAMATH_CALUDE_button_numbers_l72_7277


namespace NUMINAMATH_CALUDE_point_coordinates_l72_7242

theorem point_coordinates (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : 1 < b) (hb' : b < 2) :
  (0 < a/2 ∧ a/2 < 1/2 ∧ 2 < b+1 ∧ b+1 < 3) ∧
  (-1 < a-1 ∧ a-1 < 0 ∧ 0 < b/2 ∧ b/2 < 1) ∧
  (-1 < -a ∧ -a < 0 ∧ -2 < -b ∧ -b < -1) ∧
  (0 < 1-a ∧ 1-a < 1 ∧ 0 < b-1 ∧ b-1 < 1) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l72_7242


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l72_7259

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 10*x - 24) → (∃ y : ℝ, y^2 = 10*y - 24 ∧ x + y = 10) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l72_7259


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l72_7215

-- Define the cone
def cone_base_diameter : ℝ := 16
def cone_vertex_angle : ℝ := 90

-- Define the sphere
def sphere_touches_lateral_surfaces : Prop := sorry
def sphere_rests_on_table : Prop := sorry

-- Calculate the volume of the sphere
noncomputable def sphere_volume : ℝ := 
  let base_radius := cone_base_diameter / 2
  let cone_height := base_radius * 2
  let sphere_radius := base_radius / Real.sqrt 2
  (4 / 3) * Real.pi * (sphere_radius ^ 3)

-- Theorem statement
theorem inscribed_sphere_volume 
  (h1 : cone_vertex_angle = 90)
  (h2 : sphere_touches_lateral_surfaces)
  (h3 : sphere_rests_on_table) :
  sphere_volume = (512 * Real.sqrt 2 * Real.pi) / 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l72_7215


namespace NUMINAMATH_CALUDE_remainder_a_37_mod_45_l72_7222

def sequence_number (n : ℕ) : ℕ :=
  -- Definition of a_n: integer obtained by writing all integers from 1 to n sequentially
  sorry

theorem remainder_a_37_mod_45 : sequence_number 37 % 45 = 37 := by
  sorry

end NUMINAMATH_CALUDE_remainder_a_37_mod_45_l72_7222


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l72_7202

theorem complex_fraction_equality : 
  let a := 3 + 1/3 + 2.5
  let b := 2.5 - (1 + 1/3)
  let c := 4.6 - (2 + 1/3)
  let d := 4.6 + (2 + 1/3)
  let e := 5.2
  let f := 0.05 / (1/7 - 0.125) + 5.7
  (a / b * c / d * e) / f = 5/34 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l72_7202


namespace NUMINAMATH_CALUDE_complex_reciprocal_sum_l72_7262

theorem complex_reciprocal_sum (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_reciprocal_sum_l72_7262


namespace NUMINAMATH_CALUDE_cube_paint_equality_l72_7208

/-- The number of unit cubes with exactly one face painted in a cube of side length n -/
def one_face_painted (n : ℕ) : ℕ := 6 * (n - 2)^2

/-- The number of unit cubes with exactly two faces painted in a cube of side length n -/
def two_faces_painted (n : ℕ) : ℕ := 12 * (n - 2)

theorem cube_paint_equality (n : ℕ) (h : n > 3) :
  one_face_painted n = two_faces_painted n ↔ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_paint_equality_l72_7208


namespace NUMINAMATH_CALUDE_remaining_legos_l72_7236

theorem remaining_legos (initial : ℕ) (lost : ℕ) (remaining : ℕ) : 
  initial = 2080 → lost = 17 → remaining = initial - lost → remaining = 2063 := by
sorry

end NUMINAMATH_CALUDE_remaining_legos_l72_7236


namespace NUMINAMATH_CALUDE_express_train_speed_l72_7282

/-- 
Given two trains traveling towards each other from towns 390 km apart,
where the freight train travels 30 km/h slower than the express train,
and they pass each other after 3 hours,
prove that the speed of the express train is 80 km/h.
-/
theorem express_train_speed (distance : ℝ) (time : ℝ) (speed_difference : ℝ) 
  (h1 : distance = 390)
  (h2 : time = 3)
  (h3 : speed_difference = 30) : 
  ∃ (express_speed : ℝ), 
    express_speed * time + (express_speed - speed_difference) * time = distance ∧ 
    express_speed = 80 := by
  sorry

end NUMINAMATH_CALUDE_express_train_speed_l72_7282


namespace NUMINAMATH_CALUDE_n_is_composite_l72_7294

/-- The number of zeros in the given number -/
def num_zeros : ℕ := 2^1974 + 2^1000 - 1

/-- The number to be proven composite -/
def n : ℕ := 10^(num_zeros + 1) + 1

/-- Theorem stating that n is composite -/
theorem n_is_composite : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b :=
sorry

end NUMINAMATH_CALUDE_n_is_composite_l72_7294


namespace NUMINAMATH_CALUDE_negation_equivalence_l72_7243

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l72_7243


namespace NUMINAMATH_CALUDE_thirty_sided_polygon_diagonals_l72_7240

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem thirty_sided_polygon_diagonals :
  num_diagonals 30 = 405 := by sorry

end NUMINAMATH_CALUDE_thirty_sided_polygon_diagonals_l72_7240


namespace NUMINAMATH_CALUDE_expression_simplification_l72_7271

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (1 - x / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l72_7271


namespace NUMINAMATH_CALUDE_percentage_enrolled_in_biology_l72_7284

theorem percentage_enrolled_in_biology (total_students : ℕ) (not_enrolled : ℕ) 
  (h1 : total_students = 880) (h2 : not_enrolled = 594) :
  (((total_students - not_enrolled : ℝ) / total_students) * 100) = 
    (880 - 594 : ℝ) / 880 * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_enrolled_in_biology_l72_7284


namespace NUMINAMATH_CALUDE_money_constraints_l72_7256

theorem money_constraints (a b : ℝ) 
  (eq_constraint : 5 * a - b = 60)
  (ineq_constraint : 6 * a + b < 90) :
  a < 13.64 ∧ b < 8.18 := by
sorry

end NUMINAMATH_CALUDE_money_constraints_l72_7256


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l72_7245

theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + b + c = 1) :
  a^2 + b^2 + c^2 ≥ 1/3 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l72_7245


namespace NUMINAMATH_CALUDE_circle_properties_l72_7231

/-- Given that this equation represents a circle for real m, prove the statements about m, r, and the circle's center -/
theorem circle_properties (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*(m + 3)*x + 2*(1 - 4*m^2)*y + 16*m^4 + 9 = 0 →
    ∃ r : ℝ, (x - (m + 3))^2 + (y - (4*m^2 - 1))^2 = r^2) →
  (-1 < m ∧ m < 1) ∧
  (∃ r : ℝ, 0 < r ∧ r ≤ Real.sqrt 2 ∧
    ∀ x y : ℝ, x^2 + y^2 - 2*(m + 3)*x + 2*(1 - 4*m^2)*y + 16*m^4 + 9 = 0 →
      (x - (m + 3))^2 + (y - (4*m^2 - 1))^2 = r^2) ∧
  (∃ x y : ℝ, -1 < x ∧ x < 4 ∧ y = 4*(x - 3)^2 - 1 ∧
    x^2 + y^2 - 2*(m + 3)*x + 2*(1 - 4*m^2)*y + 16*m^4 + 9 = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l72_7231


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l72_7276

/-- 
Given an isosceles right triangle that, when folded twice along the altitude to its hypotenuse, 
results in a smaller isosceles right triangle with leg length 2 cm, 
prove that the area of the original triangle is 4 square centimeters.
-/
theorem isosceles_right_triangle_area (a : ℝ) (h1 : a > 0) : 
  (a / Real.sqrt 2 = 2) → (1 / 2 * a * a = 4) := by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l72_7276


namespace NUMINAMATH_CALUDE_xyz_value_l72_7291

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (eq4 : x * y + x * z + y * z = 8)
  (eq5 : x + y + z = 4) :
  x * y * z = 10 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l72_7291


namespace NUMINAMATH_CALUDE_inequality_proof_l72_7289

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 + b^3)^(1/3) < (a^2 + b^2)^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l72_7289


namespace NUMINAMATH_CALUDE_value_calculation_l72_7266

theorem value_calculation (n : ℝ) (v : ℝ) (h : n = 50) : 0.20 * n - 4 = v → v = 6 := by
  sorry

end NUMINAMATH_CALUDE_value_calculation_l72_7266


namespace NUMINAMATH_CALUDE_circle_radius_l72_7217

theorem circle_radius (x y : ℝ) (h : x + y = 72 * Real.pi) :
  ∃ r : ℝ, r > 0 ∧ x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l72_7217


namespace NUMINAMATH_CALUDE_inverse_square_problem_l72_7200

-- Define the relationship between x and y
def inverse_square_relation (k : ℝ) (x y : ℝ) : Prop :=
  x = k / (y ^ 2)

theorem inverse_square_problem (k : ℝ) :
  inverse_square_relation k 1 3 →
  inverse_square_relation k (1/9) 9 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_square_problem_l72_7200


namespace NUMINAMATH_CALUDE_circle_origin_outside_l72_7274

theorem circle_origin_outside (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - x + y + m = 0 → (x^2 + y^2 > 0)) → 
  (0 < m ∧ m < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_circle_origin_outside_l72_7274


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l72_7232

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- Theorem: The intersection point of the parabola y = x^2 - 3x + 2 with the y-axis is (0, 2) -/
theorem parabola_y_axis_intersection :
  ∃ (y : ℝ), f 0 = y ∧ y = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l72_7232


namespace NUMINAMATH_CALUDE_part_one_part_two_l72_7264

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 12 = 0}
def B : Set ℝ := {x | x^2 - 2*x - 8 = 0}
def C (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

-- Part I: Prove that if A = B, then a = 2
theorem part_one : A 2 = B :=
sorry

-- Part II: Prove that if B ∪ C = B, then m ∈ {-1/4, 0, 1/2}
theorem part_two (m : ℝ) : B ∪ C m = B → m ∈ ({-1/4, 0, 1/2} : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l72_7264


namespace NUMINAMATH_CALUDE_max_product_of_primes_sum_l72_7252

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem max_product_of_primes_sum (p : List ℕ) (h : p = primes) :
  ∃ (a b c d e f g h : ℕ),
    a ∈ p ∧ b ∈ p ∧ c ∈ p ∧ d ∈ p ∧ e ∈ p ∧ f ∈ p ∧ g ∈ p ∧ h ∈ p ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h ∧
    (a + b + c + d) * (e + f + g + h) = 1480 ∧
    ∀ (a' b' c' d' e' f' g' h' : ℕ),
      a' ∈ p → b' ∈ p → c' ∈ p → d' ∈ p → e' ∈ p → f' ∈ p → g' ∈ p → h' ∈ p →
      a' ≠ b' → a' ≠ c' → a' ≠ d' → a' ≠ e' → a' ≠ f' → a' ≠ g' → a' ≠ h' →
      b' ≠ c' → b' ≠ d' → b' ≠ e' → b' ≠ f' → b' ≠ g' → b' ≠ h' →
      c' ≠ d' → c' ≠ e' → c' ≠ f' → c' ≠ g' → c' ≠ h' →
      d' ≠ e' → d' ≠ f' → d' ≠ g' → d' ≠ h' →
      e' ≠ f' → e' ≠ g' → e' ≠ h' →
      f' ≠ g' → f' ≠ h' →
      g' ≠ h' →
      (a' + b' + c' + d') * (e' + f' + g' + h') ≤ 1480 :=
by sorry

end NUMINAMATH_CALUDE_max_product_of_primes_sum_l72_7252


namespace NUMINAMATH_CALUDE_cos_180_degrees_l72_7286

theorem cos_180_degrees : Real.cos (π) = -1 := by sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l72_7286


namespace NUMINAMATH_CALUDE_even_odd_square_sum_l72_7299

theorem even_odd_square_sum (a b : ℕ) :
  (Even (a * b) → ∃ c d : ℕ, a^2 + b^2 + c^2 = d^2) ∧
  (Odd (a * b) → ¬∃ c d : ℕ, a^2 + b^2 + c^2 = d^2) :=
by sorry

end NUMINAMATH_CALUDE_even_odd_square_sum_l72_7299


namespace NUMINAMATH_CALUDE_cistern_fill_time_l72_7279

/-- Time to fill cistern with all pipes open simultaneously -/
theorem cistern_fill_time (fill_time_A fill_time_B empty_time_C : ℝ) 
  (h_A : fill_time_A = 45)
  (h_B : fill_time_B = 60)
  (h_C : empty_time_C = 72) : 
  (1 / ((1 / fill_time_A) + (1 / fill_time_B) - (1 / empty_time_C))) = 40 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l72_7279


namespace NUMINAMATH_CALUDE_trig_identity_l72_7209

theorem trig_identity (x : Real) : 
  (Real.cos x)^4 + (Real.sin x)^4 + 3*(Real.sin x)^2*(Real.cos x)^2 = 
  (Real.cos x)^6 + (Real.sin x)^6 + 4*(Real.sin x)^2*(Real.cos x)^2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l72_7209


namespace NUMINAMATH_CALUDE_complement_A_subset_B_l72_7273

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 1}

-- Define set B
def B : Set ℝ := {y | y ≥ 0}

-- Define the complement of A
def complementA : Set ℝ := {x | x ≥ 1}

theorem complement_A_subset_B : complementA ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_complement_A_subset_B_l72_7273


namespace NUMINAMATH_CALUDE_studio_audience_size_l72_7246

theorem studio_audience_size :
  ∀ (total : ℕ) (envelope_ratio winner_ratio : ℚ) (winners : ℕ),
    envelope_ratio = 2/5 →
    winner_ratio = 1/5 →
    winners = 8 →
    (envelope_ratio * winner_ratio * total : ℚ) = winners →
    total = 100 := by
  sorry

end NUMINAMATH_CALUDE_studio_audience_size_l72_7246


namespace NUMINAMATH_CALUDE_max_k_value_l72_7269

open Real

noncomputable def f (x : ℝ) := exp x - x - 2

theorem max_k_value :
  ∃ (k : ℤ), k = 2 ∧
  (∀ (x : ℝ), x > 0 → (x - ↑k) * (exp x - 1) + x + 1 > 0) ∧
  (∀ (m : ℤ), m > 2 → ∃ (y : ℝ), y > 0 ∧ (y - ↑m) * (exp y - 1) + y + 1 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l72_7269


namespace NUMINAMATH_CALUDE_right_triangle_in_circle_l72_7229

theorem right_triangle_in_circle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Positive lengths
  c = 10 →  -- Diameter (and hypotenuse) is 10
  b = 8 →   -- One leg is 8
  a * a + b * b = c * c →  -- Pythagorean theorem
  a = 6 := by sorry

end NUMINAMATH_CALUDE_right_triangle_in_circle_l72_7229


namespace NUMINAMATH_CALUDE_all_drawings_fit_three_notebooks_l72_7211

/-- Proves that all drawings fit in three notebooks after reorganization --/
theorem all_drawings_fit_three_notebooks 
  (initial_notebooks : Nat) 
  (pages_per_notebook : Nat) 
  (initial_drawings_per_page : Nat) 
  (new_drawings_per_page : Nat) 
  (h1 : initial_notebooks = 5)
  (h2 : pages_per_notebook = 60)
  (h3 : initial_drawings_per_page = 8)
  (h4 : new_drawings_per_page = 15) :
  (initial_notebooks * pages_per_notebook * initial_drawings_per_page) ≤ 
  (3 * pages_per_notebook * new_drawings_per_page) := by
  sorry

#check all_drawings_fit_three_notebooks

end NUMINAMATH_CALUDE_all_drawings_fit_three_notebooks_l72_7211


namespace NUMINAMATH_CALUDE_function_domain_constraint_l72_7219

theorem function_domain_constraint (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x : ℝ, f x = (x - 7)^(1/3) / (a * x^2 + 4 * a * x + 3)) →
  (∀ x : ℝ, f x ≠ 0) →
  (0 < a ∧ a < 3/4) :=
sorry

end NUMINAMATH_CALUDE_function_domain_constraint_l72_7219
