import Mathlib

namespace NUMINAMATH_CALUDE_optimal_income_maximizes_take_home_pay_l169_16991

/-- Represents the take-home pay as a function of the tax rate x -/
def takeHomePay (x : ℝ) : ℝ := 1000 * (x + 10) - 10 * x * (x + 10)

/-- The optimal tax rate that maximizes take-home pay -/
def optimalRate : ℝ := 45

/-- The income corresponding to the optimal tax rate -/
def optimalIncome : ℝ := (optimalRate + 10) * 1000

theorem optimal_income_maximizes_take_home_pay :
  optimalIncome = 55000 ∧
  ∀ x : ℝ, takeHomePay x ≤ takeHomePay optimalRate :=
sorry

end NUMINAMATH_CALUDE_optimal_income_maximizes_take_home_pay_l169_16991


namespace NUMINAMATH_CALUDE_inverse_composition_equals_neg_eight_ninths_l169_16930

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 7

-- Define the inverse function g⁻¹
noncomputable def g_inv (y : ℝ) : ℝ := (y - 7) / 3

-- Theorem statement
theorem inverse_composition_equals_neg_eight_ninths :
  g_inv (g_inv 20) = -8/9 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_equals_neg_eight_ninths_l169_16930


namespace NUMINAMATH_CALUDE_pats_password_length_l169_16945

/-- Represents the structure of Pat's computer password -/
structure PasswordStructure where
  lowercase_count : ℕ
  uppercase_and_numbers_count : ℕ
  symbol_count : ℕ

/-- Calculates the total number of characters in Pat's password -/
def total_characters (p : PasswordStructure) : ℕ :=
  p.lowercase_count + p.uppercase_and_numbers_count + p.symbol_count

/-- Theorem stating the total number of characters in Pat's password -/
theorem pats_password_length :
  ∃ (p : PasswordStructure),
    p.lowercase_count = 8 ∧
    p.uppercase_and_numbers_count = p.lowercase_count / 2 ∧
    p.symbol_count = 2 ∧
    total_characters p = 14 := by
  sorry

end NUMINAMATH_CALUDE_pats_password_length_l169_16945


namespace NUMINAMATH_CALUDE_veranda_area_is_136_l169_16999

/-- Calculates the area of a veranda surrounding a rectangular room. -/
def verandaArea (roomLength roomWidth verandaWidth : ℝ) : ℝ :=
  (roomLength + 2 * verandaWidth) * (roomWidth + 2 * verandaWidth) - roomLength * roomWidth

/-- Theorem stating that the area of the veranda is 136 m² given the specified dimensions. -/
theorem veranda_area_is_136 :
  let roomLength : ℝ := 18
  let roomWidth : ℝ := 12
  let verandaWidth : ℝ := 2
  verandaArea roomLength roomWidth verandaWidth = 136 := by
  sorry

#eval verandaArea 18 12 2

end NUMINAMATH_CALUDE_veranda_area_is_136_l169_16999


namespace NUMINAMATH_CALUDE_certain_number_problem_l169_16904

theorem certain_number_problem (x : ℝ) : 
  ((x + 10) * 2) / 2 - 2 = 88 / 2 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l169_16904


namespace NUMINAMATH_CALUDE_equation_roots_l169_16920

/-- The equation in question -/
def equation (x k : ℂ) : Prop :=
  x / (x + 3) + x / (x + 4) = k * x

/-- The condition for having exactly two distinct complex roots -/
def has_two_distinct_roots (k : ℂ) : Prop :=
  ∃ (x₁ x₂ : ℂ), x₁ ≠ x₂ ∧ 
    equation x₁ k ∧ equation x₂ k ∧
    ∀ x, equation x k → (x = x₁ ∨ x = x₂)

/-- The main theorem -/
theorem equation_roots (k : ℂ) :
  has_two_distinct_roots k ↔ (k = 2*I ∨ k = -2*I) :=
sorry

end NUMINAMATH_CALUDE_equation_roots_l169_16920


namespace NUMINAMATH_CALUDE_bounded_figure_at_most_one_center_no_figure_exactly_two_centers_finite_set_at_most_three_almost_centers_l169_16958

-- Define a type for figures
structure Figure where
  isBounded : Bool

-- Define a type for sets of points
structure PointSet where
  isFinite : Bool

-- Define a function to count centers of symmetry
def countCentersOfSymmetry (f : Figure) : Nat :=
  sorry

-- Define a function to count almost centers of symmetry
def countAlmostCentersOfSymmetry (s : PointSet) : Nat :=
  sorry

-- Theorem 1: A bounded figure has at most one center of symmetry
theorem bounded_figure_at_most_one_center (f : Figure) (h : f.isBounded = true) :
  countCentersOfSymmetry f ≤ 1 :=
sorry

-- Theorem 2: No figure can have exactly two centers of symmetry
theorem no_figure_exactly_two_centers (f : Figure) :
  countCentersOfSymmetry f ≠ 2 :=
sorry

-- Theorem 3: A finite set of points has at most 3 almost centers of symmetry
theorem finite_set_at_most_three_almost_centers (s : PointSet) (h : s.isFinite = true) :
  countAlmostCentersOfSymmetry s ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_bounded_figure_at_most_one_center_no_figure_exactly_two_centers_finite_set_at_most_three_almost_centers_l169_16958


namespace NUMINAMATH_CALUDE_inequality_proof_l169_16992

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l169_16992


namespace NUMINAMATH_CALUDE_equation_solution_l169_16979

theorem equation_solution :
  ∃ k : ℚ, (3 * k - 4) / (k + 7) = 2 / 5 ↔ k = 34 / 13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l169_16979


namespace NUMINAMATH_CALUDE_average_of_numbers_l169_16949

def numbers : List ℕ := [12, 13, 14, 510, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers :
  (numbers.sum : ℚ) / numbers.length = 125790 := by sorry

end NUMINAMATH_CALUDE_average_of_numbers_l169_16949


namespace NUMINAMATH_CALUDE_word_count_is_370_l169_16984

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of five-letter words with exactly two A's and at least one O -/
def word_count : ℕ :=
  let a_freq := 6
  let e_freq := 4
  let i_freq := 5
  let o_freq := 3
  let u_freq := 2
  let word_length := 5
  let a_count := 2
  let remaining_letters := word_length - a_count
  let ways_to_place_a := choose word_length a_count
  let ways_to_place_o_and_others := 
    (choose remaining_letters 1) * (e_freq + i_freq + u_freq)^2 +
    (choose remaining_letters 2) * (e_freq + i_freq + u_freq) +
    (choose remaining_letters 3)
  ways_to_place_a * ways_to_place_o_and_others

theorem word_count_is_370 : word_count = 370 := by
  sorry

end NUMINAMATH_CALUDE_word_count_is_370_l169_16984


namespace NUMINAMATH_CALUDE_matrix_power_50_l169_16978

theorem matrix_power_50 (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A = ![![1, 1], ![0, 1]] →
  A ^ 50 = ![![1, 50], ![0, 1]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_50_l169_16978


namespace NUMINAMATH_CALUDE_coin_flip_probability_l169_16959

def n : ℕ := 12
def k : ℕ := 4
def p : ℚ := 1/2

theorem coin_flip_probability :
  (n.choose k) * p^k * (1 - p)^(n - k) = 495/4096 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l169_16959


namespace NUMINAMATH_CALUDE_triangle_theorem_l169_16910

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) 
  (ha : t.a = 2)
  (hc : t.c = 3)
  (hcosB : Real.cos t.angleB = 1/4) :
  t.b = Real.sqrt 10 ∧ Real.sin (2 * t.angleC) = (3 * Real.sqrt 15) / 16 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l169_16910


namespace NUMINAMATH_CALUDE_xyz_value_l169_16909

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) :
  x * y * z = 10 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l169_16909


namespace NUMINAMATH_CALUDE_f_min_max_on_interval_l169_16917

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem f_min_max_on_interval :
  let a : ℝ := 0
  let b : ℝ := 2 * Real.pi
  ∃ (x_min x_max : ℝ), a ≤ x_min ∧ x_min ≤ b ∧ a ≤ x_max ∧ x_max ≤ b ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x_min ≤ f x) ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f x_max) ∧
    f x_min = -3 * Real.pi / 2 ∧
    f x_max = Real.pi / 2 + 2 :=
  sorry

end NUMINAMATH_CALUDE_f_min_max_on_interval_l169_16917


namespace NUMINAMATH_CALUDE_proposition_truth_values_l169_16976

theorem proposition_truth_values (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬(¬p)) : 
  ¬q := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l169_16976


namespace NUMINAMATH_CALUDE_min_value_of_M_l169_16918

theorem min_value_of_M (a b : ℝ) (h1 : a > b) (h2 : a * b = 1) :
  let M := (a^2 + b^2) / (a - b)
  ∀ x, M ≥ x → x ≥ 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_M_l169_16918


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l169_16990

theorem inequality_system_solution_set :
  let S := {x : ℝ | x - 2 ≥ -5 ∧ 3*x < x + 2}
  S = {x : ℝ | -3 ≤ x ∧ x < 1} := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l169_16990


namespace NUMINAMATH_CALUDE_dried_mushroom_weight_l169_16943

/-- 
Given:
- Fresh mushrooms contain 90% water by weight
- Dried mushrooms contain 12% water by weight
- We start with 22 kg of fresh mushrooms

Prove that the weight of dried mushrooms obtained is 2.5 kg
-/
theorem dried_mushroom_weight (fresh_water_content : ℝ) (dried_water_content : ℝ) 
  (fresh_weight : ℝ) (dried_weight : ℝ) :
  fresh_water_content = 0.90 →
  dried_water_content = 0.12 →
  fresh_weight = 22 →
  dried_weight = 2.5 →
  dried_weight = (1 - fresh_water_content) * fresh_weight / (1 - dried_water_content) :=
by sorry

end NUMINAMATH_CALUDE_dried_mushroom_weight_l169_16943


namespace NUMINAMATH_CALUDE_judy_spending_l169_16944

def carrot_price : ℕ := 1
def milk_price : ℕ := 3
def pineapple_price : ℕ := 4
def flour_price : ℕ := 5
def ice_cream_price : ℕ := 7

def carrot_quantity : ℕ := 5
def milk_quantity : ℕ := 4
def pineapple_quantity : ℕ := 2
def flour_quantity : ℕ := 2

def coupon_discount : ℕ := 10
def coupon_threshold : ℕ := 40

def shopping_total : ℕ :=
  carrot_price * carrot_quantity +
  milk_price * milk_quantity +
  pineapple_price * (pineapple_quantity / 2) +
  flour_price * flour_quantity +
  ice_cream_price

theorem judy_spending :
  shopping_total = 38 :=
by sorry

end NUMINAMATH_CALUDE_judy_spending_l169_16944


namespace NUMINAMATH_CALUDE_sphere_to_great_circle_area_ratio_l169_16947

/-- The ratio of the area of a sphere to the area of its great circle is 4 -/
theorem sphere_to_great_circle_area_ratio :
  ∀ (R : ℝ), R > 0 →
  (4 * π * R^2) / (π * R^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_sphere_to_great_circle_area_ratio_l169_16947


namespace NUMINAMATH_CALUDE_probability_all_digits_different_l169_16922

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def all_digits_different (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

def count_three_digit_numbers : ℕ := 999 - 100 + 1

def count_three_digit_same_digits : ℕ := 9

theorem probability_all_digits_different :
  (count_three_digit_numbers - count_three_digit_same_digits : ℚ) / count_three_digit_numbers = 99/100 :=
sorry

end NUMINAMATH_CALUDE_probability_all_digits_different_l169_16922


namespace NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l169_16980

theorem arithmetic_progression_of_primes (a : ℕ → ℕ) (d : ℕ) :
  (∀ i ∈ Finset.range 15, Nat.Prime (a i)) →
  (∀ i ∈ Finset.range 14, a (i + 1) = a i + d) →
  d > 0 →
  a 0 > 15 →
  d > 30000 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l169_16980


namespace NUMINAMATH_CALUDE_jack_king_queen_probability_l169_16940

theorem jack_king_queen_probability : 
  let deck_size : ℕ := 52
  let jack_count : ℕ := 4
  let king_count : ℕ := 4
  let queen_count : ℕ := 4
  let prob_jack : ℚ := jack_count / deck_size
  let prob_king : ℚ := king_count / (deck_size - 1)
  let prob_queen : ℚ := queen_count / (deck_size - 2)
  prob_jack * prob_king * prob_queen = 8 / 16575 :=
by sorry

end NUMINAMATH_CALUDE_jack_king_queen_probability_l169_16940


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l169_16938

theorem count_integers_satisfying_inequality :
  ∃! (S : Finset Int),
    (∀ n ∈ S, -11 ≤ n ∧ n ≤ 11 ∧ (n - 2) * (n + 4) * (n + 8) < 0) ∧
    (∀ n, -11 ≤ n ∧ n ≤ 11 ∧ (n - 2) * (n + 4) * (n + 8) < 0 → n ∈ S) ∧
    S.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l169_16938


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l169_16954

-- Define the ellipse parameters
def a : ℝ := 2
def b : ℝ := 1

-- Define the point M
def M : ℝ × ℝ := (2, 1)

-- Define the ellipse equation
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define a line passing through M
def line_through_M (k : ℝ) (x y : ℝ) : Prop :=
  y - M.2 = k * (x - M.1)

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  M.1 = (x₁ + x₂) / 2 ∧ M.2 = (y₁ + y₂) / 2

-- Theorem statement
theorem ellipse_line_intersection :
  ∃ (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
    is_on_ellipse x₁ y₁ ∧
    is_on_ellipse x₂ y₂ ∧
    line_through_M k x₁ y₁ ∧
    line_through_M k x₂ y₂ ∧
    is_midpoint x₁ y₁ x₂ y₂ ∧
    k = -1/2 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l169_16954


namespace NUMINAMATH_CALUDE_fib_100_mod_5_l169_16975

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Theorem statement
theorem fib_100_mod_5 : fib 99 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_100_mod_5_l169_16975


namespace NUMINAMATH_CALUDE_accidental_addition_l169_16988

theorem accidental_addition (x : ℕ) : x + 65 = 125 → x + 95 = 155 := by
  sorry

end NUMINAMATH_CALUDE_accidental_addition_l169_16988


namespace NUMINAMATH_CALUDE_y_values_l169_16973

theorem y_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 72) :
  let y := ((x - 3)^2 * (x + 2)) / (2 * x - 4)
  y = 9 ∨ y = 3.6 :=
by sorry

end NUMINAMATH_CALUDE_y_values_l169_16973


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l169_16989

-- Define the sets M and N
def M : Set ℝ := {x | (x + 2) * (x - 1) < 0}
def N : Set ℝ := {x | x + 1 < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | -2 < x ∧ x < -1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l169_16989


namespace NUMINAMATH_CALUDE_inequality_solution_l169_16981

theorem inequality_solution (x : ℝ) :
  (6 * x^2 + 9 * x - 48) / ((3 * x + 5) * (x - 2)) < 0 ↔ 
  -4 < x ∧ x < -5/3 ∧ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l169_16981


namespace NUMINAMATH_CALUDE_age_problem_l169_16902

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 22 →
  b = 8 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l169_16902


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l169_16916

def A : Set ℝ := {x | x^2 ≤ 1}
def B : Set ℝ := {x | (x-2)/x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l169_16916


namespace NUMINAMATH_CALUDE_square_sum_equation_l169_16923

theorem square_sum_equation (x y : ℝ) : 
  (x^2 + y^2)^2 = x^2 + y^2 + 12 → x^2 + y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equation_l169_16923


namespace NUMINAMATH_CALUDE_magic_polynomial_bound_l169_16903

open Polynomial
open Nat

theorem magic_polynomial_bound (n : ℕ) (P : Polynomial ℚ) 
  (h_deg : degree P = n) (h_irr : Irreducible P) :
  ∃ (s : Finset (Polynomial ℚ)), 
    (∀ Q ∈ s, degree Q < n ∧ (P ∣ (P.comp Q))) ∧ 
    (∀ Q : Polynomial ℚ, degree Q < n → (P ∣ (P.comp Q)) → Q ∈ s) ∧
    s.card ≤ n := by
  sorry

end NUMINAMATH_CALUDE_magic_polynomial_bound_l169_16903


namespace NUMINAMATH_CALUDE_banana_bunches_l169_16962

theorem banana_bunches (total_bananas : ℕ) (known_bunches : ℕ) (known_bananas_per_bunch : ℕ) 
  (unknown_bunches : ℕ) (h1 : total_bananas = 83) (h2 : known_bunches = 6) 
  (h3 : known_bananas_per_bunch = 8) (h4 : unknown_bunches = 5) : 
  (total_bananas - known_bunches * known_bananas_per_bunch) / unknown_bunches = 7 := by
  sorry

end NUMINAMATH_CALUDE_banana_bunches_l169_16962


namespace NUMINAMATH_CALUDE_sisyphus_earning_zero_l169_16966

/-- Represents the state of the stone boxes and Sisyphus's earnings -/
structure BoxState where
  a : ℕ  -- number of stones in box A
  b : ℕ  -- number of stones in box B
  c : ℕ  -- number of stones in box C
  x : ℤ  -- Sisyphus's earnings (can be negative)

/-- Represents a move of a stone from one box to another -/
inductive Move
  | AB : Move  -- from A to B
  | AC : Move  -- from A to C
  | BA : Move  -- from B to A
  | BC : Move  -- from B to C
  | CA : Move  -- from C to A
  | CB : Move  -- from C to B

/-- Applies a move to a BoxState -/
def applyMove (state : BoxState) (move : Move) : BoxState :=
  match move with
  | Move.AB => { state with 
      a := state.a - 1, 
      b := state.b + 1, 
      x := state.x + (state.a - state.b - 1) }
  | Move.AC => { state with 
      a := state.a - 1, 
      c := state.c + 1, 
      x := state.x + (state.a - state.c - 1) }
  | Move.BA => { state with 
      b := state.b - 1, 
      a := state.a + 1, 
      x := state.x + (state.b - state.a - 1) }
  | Move.BC => { state with 
      b := state.b - 1, 
      c := state.c + 1, 
      x := state.x + (state.b - state.c - 1) }
  | Move.CA => { state with 
      c := state.c - 1, 
      a := state.a + 1, 
      x := state.x + (state.c - state.a - 1) }
  | Move.CB => { state with 
      c := state.c - 1, 
      b := state.b + 1, 
      x := state.x + (state.c - state.b - 1) }

/-- Theorem: The greatest possible earning of Sisyphus is 0 -/
theorem sisyphus_earning_zero 
  (initial : BoxState) 
  (moves : List Move) 
  (h1 : moves.length = 24 * 365 * 1000) -- 1000 years of hourly moves
  (h2 : (moves.foldl applyMove initial).a = initial.a) -- stones return to initial state
  (h3 : (moves.foldl applyMove initial).b = initial.b)
  (h4 : (moves.foldl applyMove initial).c = initial.c) :
  (moves.foldl applyMove initial).x ≤ 0 :=
sorry

#check sisyphus_earning_zero

end NUMINAMATH_CALUDE_sisyphus_earning_zero_l169_16966


namespace NUMINAMATH_CALUDE_students_in_class_l169_16901

/-- The number of students in a class that needs to earn a certain number of points for eating vegetables. -/
def number_of_students (total_points : ℕ) (points_per_vegetable : ℕ) (num_weeks : ℕ) (vegetables_per_week : ℕ) : ℕ :=
  total_points / (points_per_vegetable * num_weeks * vegetables_per_week)

/-- Theorem stating that there are 25 students in the class given the problem conditions. -/
theorem students_in_class : 
  number_of_students 200 2 2 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_in_class_l169_16901


namespace NUMINAMATH_CALUDE_jane_final_score_l169_16964

/-- Calculates the final score in a card game --/
def final_score (rounds : ℕ) (points_per_win : ℕ) (points_lost : ℕ) : ℕ :=
  rounds * points_per_win - points_lost

/-- Theorem: Jane's final score in the card game --/
theorem jane_final_score :
  let rounds : ℕ := 8
  let points_per_win : ℕ := 10
  let points_lost : ℕ := 20
  final_score rounds points_per_win points_lost = 60 := by
  sorry


end NUMINAMATH_CALUDE_jane_final_score_l169_16964


namespace NUMINAMATH_CALUDE_cube_root_of_27_l169_16965

theorem cube_root_of_27 (x : ℝ) (h : (Real.sqrt x) ^ 3 = 27) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_27_l169_16965


namespace NUMINAMATH_CALUDE_achieve_target_average_l169_16971

/-- Represents Gage's skating schedule and target average -/
structure SkatingSchedule where
  days_with_80_min : Nat
  days_with_100_min : Nat
  target_average : Nat
  total_days : Nat

/-- Calculates the total skating time for the given schedule -/
def total_skating_time (schedule : SkatingSchedule) (last_day_minutes : Nat) : Nat :=
  schedule.days_with_80_min * 80 + 
  schedule.days_with_100_min * 100 + 
  last_day_minutes

/-- Theorem stating that skating 140 minutes on the 8th day achieves the target average -/
theorem achieve_target_average (schedule : SkatingSchedule) 
    (h1 : schedule.days_with_80_min = 4)
    (h2 : schedule.days_with_100_min = 3)
    (h3 : schedule.target_average = 95)
    (h4 : schedule.total_days = 8) :
  total_skating_time schedule 140 / schedule.total_days = schedule.target_average := by
  sorry

#eval total_skating_time { days_with_80_min := 4, days_with_100_min := 3, target_average := 95, total_days := 8 } 140

end NUMINAMATH_CALUDE_achieve_target_average_l169_16971


namespace NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l169_16921

/-- Given that the total marks in physics, chemistry, and mathematics is 150 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 75. -/
theorem average_marks_chemistry_mathematics (P C M : ℝ)
  (h : P + C + M = P + 150) :
  (C + M) / 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l169_16921


namespace NUMINAMATH_CALUDE_baseball_distribution_l169_16914

theorem baseball_distribution (total : ℕ) (classes : ℕ) (h1 : total = 43) (h2 : classes = 6) :
  total % classes = 1 := by
  sorry

end NUMINAMATH_CALUDE_baseball_distribution_l169_16914


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l169_16932

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fifth_term
    (a : ℕ → ℝ)
    (h_geometric : IsGeometricSequence a)
    (h_sum1 : a 1 + a 2 = 4)
    (h_sum2 : a 2 + a 3 = 12) :
    a 5 = 81 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l169_16932


namespace NUMINAMATH_CALUDE_function_value_at_nine_l169_16906

-- Define the function f(x) = k * x^(1/2)
def f (k : ℝ) (x : ℝ) : ℝ := k * (x ^ (1/2))

-- State the theorem
theorem function_value_at_nine (k : ℝ) : 
  f k 16 = 6 → f k 9 = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_nine_l169_16906


namespace NUMINAMATH_CALUDE_sum_of_common_elements_l169_16937

/-- Arithmetic progression with first term 4 and common difference 3 -/
def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n

/-- Geometric progression with first term 20 and common ratio 2 -/
def geometric_progression (k : ℕ) : ℕ := 20 * 2^k

/-- The sequence of common elements between the arithmetic and geometric progressions -/
def common_sequence (n : ℕ) : ℕ := 40 * 4^n

theorem sum_of_common_elements :
  (Finset.range 10).sum common_sequence = 13981000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_common_elements_l169_16937


namespace NUMINAMATH_CALUDE_actual_weight_loss_percentage_l169_16957

-- Define the weight loss challenge scenario
def weight_loss_challenge (W : ℝ) (actual_loss_percent : ℝ) (clothes_add_percent : ℝ) (measured_loss_percent : ℝ) : Prop :=
  let final_weight := W * (1 - actual_loss_percent / 100 + clothes_add_percent / 100)
  final_weight = W * (1 - measured_loss_percent / 100)

-- Theorem statement
theorem actual_weight_loss_percentage 
  (W : ℝ) (actual_loss_percent : ℝ) (clothes_add_percent : ℝ) (measured_loss_percent : ℝ)
  (h1 : W > 0)
  (h2 : clothes_add_percent = 2)
  (h3 : measured_loss_percent = 8.2)
  (h4 : weight_loss_challenge W actual_loss_percent clothes_add_percent measured_loss_percent) :
  actual_loss_percent = 10.2 := by
sorry


end NUMINAMATH_CALUDE_actual_weight_loss_percentage_l169_16957


namespace NUMINAMATH_CALUDE_three_digit_number_theorem_l169_16970

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds_bound : hundreds < 10
  h_tens_bound : tens < 10
  h_ones_bound : ones < 10

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The reversed value of a three-digit number -/
def ThreeDigitNumber.reversedValue (n : ThreeDigitNumber) : Nat :=
  100 * n.ones + 10 * n.tens + n.hundreds

theorem three_digit_number_theorem (n : ThreeDigitNumber) 
    (h_sum : n.hundreds + n.tens + n.ones = 10)
    (h_middle : n.tens = n.hundreds + n.ones)
    (h_reverse : n.reversedValue = n.value + 99) :
    n.value = 203 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_theorem_l169_16970


namespace NUMINAMATH_CALUDE_decreasing_function_property_l169_16987

-- Define a real-valued function f on the positive real numbers
variable (f : ℝ → ℝ)

-- Define the property of f being decreasing on (0, +∞)
def IsDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ 0 < y ∧ x < y → f y < f x

-- State the theorem
theorem decreasing_function_property
  (h : IsDecreasingOn f) : f 3 < f 2 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_property_l169_16987


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l169_16963

theorem cubic_root_ratio (p q r s : ℝ) (h : p ≠ 0) :
  (∀ x, p * x^3 + q * x^2 + r * x + s = 0 ↔ x = -1 ∨ x = 3 ∨ x = 4) →
  r / s = -5 / 12 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l169_16963


namespace NUMINAMATH_CALUDE_polynomial_symmetry_condition_l169_16950

/-- A polynomial function of degree 4 -/
def polynomial (a b c d e : ℝ) (x : ℝ) : ℝ :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

/-- Symmetry condition for a function -/
def isSymmetric (f : ℝ → ℝ) : Prop :=
  ∃ t : ℝ, ∀ x : ℝ, f x = f (2 * t - x)

theorem polynomial_symmetry_condition
  (a b c d e : ℝ) (h : a ≠ 0) :
  isSymmetric (polynomial a b c d e) ↔ b^3 - a*b*c + 8*a^2*d = 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_condition_l169_16950


namespace NUMINAMATH_CALUDE_definite_integral_x_squared_l169_16982

theorem definite_integral_x_squared : ∫ x in (-1)..(1), x^2 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_x_squared_l169_16982


namespace NUMINAMATH_CALUDE_tom_car_washing_earnings_l169_16926

/-- Represents the amount of money Tom had last week in dollars -/
def initial_amount : ℕ := 74

/-- Represents the amount of money Tom has now in dollars -/
def current_amount : ℕ := 86

/-- Represents the amount of money Tom made washing cars in dollars -/
def money_made : ℕ := current_amount - initial_amount

theorem tom_car_washing_earnings :
  money_made = current_amount - initial_amount :=
by sorry

end NUMINAMATH_CALUDE_tom_car_washing_earnings_l169_16926


namespace NUMINAMATH_CALUDE_two_numbers_difference_l169_16928

theorem two_numbers_difference (a b : ℕ) (h1 : a < b) (h2 : a + b = 20000) 
  (h3 : a % 9 = 0 ∨ b % 9 = 0) (h4 : 2 * a + 6 = b) : b - a = 6670 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l169_16928


namespace NUMINAMATH_CALUDE_dairy_factory_profit_comparison_l169_16993

/-- Represents the profit calculation for a dairy factory --/
theorem dairy_factory_profit_comparison :
  let total_milk : ℝ := 20
  let fresh_milk_profit : ℝ := 500
  let yogurt_profit : ℝ := 1000
  let milk_powder_profit : ℝ := 1800
  let yogurt_capacity : ℝ := 6
  let milk_powder_capacity : ℝ := 2
  let days : ℝ := 4

  let plan_one_profit : ℝ := 
    (milk_powder_capacity * days * milk_powder_profit) + 
    ((total_milk - milk_powder_capacity * days) * fresh_milk_profit)

  let plan_two_milk_powder_days : ℝ := 
    (total_milk - yogurt_capacity * days) / (yogurt_capacity - milk_powder_capacity)
  
  let plan_two_yogurt_days : ℝ := days - plan_two_milk_powder_days

  let plan_two_profit : ℝ := 
    (plan_two_milk_powder_days * milk_powder_capacity * milk_powder_profit) + 
    (plan_two_yogurt_days * yogurt_capacity * yogurt_profit)

  plan_two_profit > plan_one_profit := by sorry

end NUMINAMATH_CALUDE_dairy_factory_profit_comparison_l169_16993


namespace NUMINAMATH_CALUDE_ranch_problem_l169_16948

theorem ranch_problem : ∃! (s c : ℕ), s > 0 ∧ c > 0 ∧ 35 * s + 40 * c = 1200 ∧ c > s := by
  sorry

end NUMINAMATH_CALUDE_ranch_problem_l169_16948


namespace NUMINAMATH_CALUDE_gulbis_count_l169_16935

/-- The number of gulbis in one dureum -/
def fish_per_dureum : ℕ := 20

/-- The number of dureums of gulbis -/
def num_dureums : ℕ := 156

/-- The total number of gulbis -/
def total_gulbis : ℕ := num_dureums * fish_per_dureum

theorem gulbis_count : total_gulbis = 3120 := by
  sorry

end NUMINAMATH_CALUDE_gulbis_count_l169_16935


namespace NUMINAMATH_CALUDE_gcd_1994_powers_and_product_l169_16967

theorem gcd_1994_powers_and_product : 
  Nat.gcd (1994^1994 + 1994^1995) (1994 * 1995) = 1994 * 1995 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1994_powers_and_product_l169_16967


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l169_16911

theorem contrapositive_equivalence :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔
  (∀ x y : ℝ, ¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l169_16911


namespace NUMINAMATH_CALUDE_g_of_3_equals_4_l169_16933

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Theorem statement
theorem g_of_3_equals_4 : g 3 = 4 := by sorry

end NUMINAMATH_CALUDE_g_of_3_equals_4_l169_16933


namespace NUMINAMATH_CALUDE_line_inclination_sine_l169_16924

/-- Given a straight line 3x - 4y + 5 = 0 with angle of inclination α, prove that sin(α) = 3/5 -/
theorem line_inclination_sine (x y : ℝ) (α : ℝ) 
  (h : 3 * x - 4 * y + 5 = 0) 
  (h_incl : α = Real.arctan (3 / 4)) : 
  Real.sin α = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_sine_l169_16924


namespace NUMINAMATH_CALUDE_mode_is_two_hours_l169_16995

/-- Represents the time spent on volunteer activities -/
inductive VolunteerTime
  | OneHour
  | OneAndHalfHours
  | TwoHours
  | TwoAndHalfHours
  | ThreeHours

/-- The number of students for each volunteer time category -/
def studentCount : VolunteerTime → Nat
  | VolunteerTime.OneHour => 20
  | VolunteerTime.OneAndHalfHours => 32
  | VolunteerTime.TwoHours => 38
  | VolunteerTime.TwoAndHalfHours => 8
  | VolunteerTime.ThreeHours => 2

/-- The total number of students -/
def totalStudents : Nat := 100

/-- The mode of the data set -/
def dataMode : VolunteerTime := VolunteerTime.TwoHours

theorem mode_is_two_hours :
  (∀ t : VolunteerTime, studentCount dataMode ≥ studentCount t) ∧
  dataMode = VolunteerTime.TwoHours :=
by sorry

end NUMINAMATH_CALUDE_mode_is_two_hours_l169_16995


namespace NUMINAMATH_CALUDE_train_speed_calculation_l169_16955

/-- Given a train of length 150 meters passing an oak tree in 9.99920006399488 seconds,
    prove that its speed is 54.00287976961843 km/hr. -/
theorem train_speed_calculation (train_length : Real) (time_to_pass : Real) :
  train_length = 150 →
  time_to_pass = 9.99920006399488 →
  (train_length / time_to_pass) * 3.6 = 54.00287976961843 := by
  sorry

#eval (150 / 9.99920006399488) * 3.6

end NUMINAMATH_CALUDE_train_speed_calculation_l169_16955


namespace NUMINAMATH_CALUDE_total_flowers_collected_l169_16934

/-- The maximum number of flowers each person can pick --/
def max_flowers : ℕ := 50

/-- The number of tulips Arwen picked --/
def arwen_tulips : ℕ := 20

/-- The number of roses Arwen picked --/
def arwen_roses : ℕ := 18

/-- The number of sunflowers Arwen picked --/
def arwen_sunflowers : ℕ := 6

/-- The number of tulips Elrond picked --/
def elrond_tulips : ℕ := 2 * arwen_tulips

/-- The number of roses Elrond picked --/
def elrond_roses : ℕ := min (3 * arwen_roses) (max_flowers - elrond_tulips)

/-- The number of tulips Galadriel picked --/
def galadriel_tulips : ℕ := min (3 * elrond_tulips) max_flowers

/-- The number of roses Galadriel picked --/
def galadriel_roses : ℕ := min (2 * arwen_roses) (max_flowers - galadriel_tulips)

/-- The number of sunflowers Legolas picked --/
def legolas_sunflowers : ℕ := arwen_sunflowers

/-- The number of roses Legolas picked --/
def legolas_roses : ℕ := (max_flowers - legolas_sunflowers) / 2

/-- The number of tulips Legolas picked --/
def legolas_tulips : ℕ := (max_flowers - legolas_sunflowers) / 2

theorem total_flowers_collected :
  arwen_tulips + arwen_roses + arwen_sunflowers +
  elrond_tulips + elrond_roses +
  galadriel_tulips + galadriel_roses +
  legolas_sunflowers + legolas_roses + legolas_tulips = 194 := by
  sorry

#eval arwen_tulips + arwen_roses + arwen_sunflowers +
  elrond_tulips + elrond_roses +
  galadriel_tulips + galadriel_roses +
  legolas_sunflowers + legolas_roses + legolas_tulips

end NUMINAMATH_CALUDE_total_flowers_collected_l169_16934


namespace NUMINAMATH_CALUDE_f_is_mapping_from_A_to_B_l169_16952

def A : Set ℕ := {0, 1, 2, 4}
def B : Set ℚ := {1/2, 0, 1, 2, 6, 8}

def f (x : ℕ) : ℚ := 2^(x - 1)

theorem f_is_mapping_from_A_to_B : ∀ x ∈ A, f x ∈ B := by
  sorry

end NUMINAMATH_CALUDE_f_is_mapping_from_A_to_B_l169_16952


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l169_16919

theorem necessary_not_sufficient_condition :
  (∀ x : ℝ, x^2 - 2*x - 3 < 0 → -2 < x ∧ x < 3) ∧
  ¬(∀ x : ℝ, -2 < x ∧ x < 3 → x^2 - 2*x - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l169_16919


namespace NUMINAMATH_CALUDE_circle_tangent_intersection_ratio_l169_16972

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are externally tangent at a point -/
def ExternallyTangent (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Two circles intersect at a point -/
def Intersect (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Distance between two points -/
def Distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

theorem circle_tangent_intersection_ratio
  (Γ₁ Γ₂ Γ₃ Γ₄ : Circle)
  (P A B C D : ℝ × ℝ)
  (h1 : Γ₁ ≠ Γ₂ ∧ Γ₁ ≠ Γ₃ ∧ Γ₁ ≠ Γ₄ ∧ Γ₂ ≠ Γ₃ ∧ Γ₂ ≠ Γ₄ ∧ Γ₃ ≠ Γ₄)
  (h2 : ExternallyTangent Γ₁ Γ₃ P)
  (h3 : ExternallyTangent Γ₂ Γ₄ P)
  (h4 : Intersect Γ₁ Γ₂ A)
  (h5 : Intersect Γ₂ Γ₃ B)
  (h6 : Intersect Γ₃ Γ₄ C)
  (h7 : Intersect Γ₄ Γ₁ D)
  (h8 : A ≠ P ∧ B ≠ P ∧ C ≠ P ∧ D ≠ P) :
  (Distance A B * Distance B C) / (Distance A D * Distance D C) = 
  (Distance P B)^2 / (Distance P D)^2 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_intersection_ratio_l169_16972


namespace NUMINAMATH_CALUDE_road_travel_rate_l169_16968

/-- The rate per square meter for traveling roads on a rectangular lawn -/
theorem road_travel_rate (lawn_length lawn_width road_width total_cost : ℕ) :
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_width = 10 ∧ 
  total_cost = 5200 →
  (total_cost : ℚ) / ((lawn_length * road_width + lawn_width * road_width - road_width * road_width) : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_road_travel_rate_l169_16968


namespace NUMINAMATH_CALUDE_probability_sum_three_l169_16977

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numSides * numSides

/-- The number of ways to roll a sum of 3 with two dice -/
def favorableOutcomes : ℕ := 2

/-- The probability of rolling a sum of 3 with two fair six-sided dice -/
theorem probability_sum_three (numSides : ℕ) (totalOutcomes : ℕ) (favorableOutcomes : ℕ) :
  numSides = 6 →
  totalOutcomes = numSides * numSides →
  favorableOutcomes = 2 →
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_three_l169_16977


namespace NUMINAMATH_CALUDE_nearest_town_distance_l169_16936

theorem nearest_town_distance (d : ℝ) : 
  (¬ (d ≥ 8)) → (¬ (d ≤ 7)) → (¬ (d ≤ 6)) → (d > 7 ∧ d < 8) :=
by
  sorry

end NUMINAMATH_CALUDE_nearest_town_distance_l169_16936


namespace NUMINAMATH_CALUDE_power_of_negative_square_l169_16915

theorem power_of_negative_square (m : ℝ) : (-m^2)^4 = m^8 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_square_l169_16915


namespace NUMINAMATH_CALUDE_not_like_terms_example_l169_16997

/-- Definition of a monomial -/
structure Monomial (α : Type*) [CommRing α] :=
  (coeff : α)
  (vars : List (α × ℕ))

/-- Definition of like terms -/
def are_like_terms {α : Type*} [CommRing α] (m1 m2 : Monomial α) : Prop :=
  m1.vars.map Prod.fst = m2.vars.map Prod.fst ∧
  m1.vars.map Prod.snd = m2.vars.map Prod.snd

/-- The main theorem -/
theorem not_like_terms_example {α : Type*} [CommRing α] :
  ¬ are_like_terms 
    (Monomial.mk 7 [(a, 2), (n, 1)])
    (Monomial.mk (-9) [(a, 1), (n, 2)]) :=
sorry

end NUMINAMATH_CALUDE_not_like_terms_example_l169_16997


namespace NUMINAMATH_CALUDE_symmetric_point_on_parabola_l169_16985

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y = (x-h)^2 + k -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- Checks if a point lies on a given parabola -/
def isOnParabola (p : Point) (parab : Parabola) : Prop :=
  p.y = (p.x - parab.h)^2 + parab.k

/-- Finds the symmetric point with respect to the axis of symmetry of a parabola -/
def symmetricPoint (p : Point) (parab : Parabola) : Point :=
  ⟨2 * parab.h - p.x, p.y⟩

theorem symmetric_point_on_parabola (parab : Parabola) (p : Point) :
  isOnParabola p parab → p.x = -1 → 
  symmetricPoint p parab = Point.mk 3 6 := by
  sorry

#check symmetric_point_on_parabola

end NUMINAMATH_CALUDE_symmetric_point_on_parabola_l169_16985


namespace NUMINAMATH_CALUDE_tank_capacity_l169_16913

/-- Represents the properties of a tank with a leak and an inlet pipe. -/
structure Tank where
  capacity : ℝ
  leak_empty_time : ℝ
  inlet_rate : ℝ
  combined_empty_time : ℝ

/-- Theorem stating that a tank with given properties has a capacity of 1080 litres. -/
theorem tank_capacity (t : Tank)
  (h1 : t.leak_empty_time = 4)
  (h2 : t.inlet_rate = 6)
  (h3 : t.combined_empty_time = 12) :
  t.capacity = 1080 :=
by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l169_16913


namespace NUMINAMATH_CALUDE_max_value_fraction_l169_16907

theorem max_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -2) (hy : 0 ≤ y ∧ y ≤ 4) :
  (x + y) / x ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l169_16907


namespace NUMINAMATH_CALUDE_polynomial_equality_l169_16960

/-- Given polynomials h and k such that h(x) + k(x) = 3x^2 + 2x - 5 and h(x) = x^4 - 3x^2 + 1,
    prove that k(x) = -x^4 + 6x^2 + 2x - 6 -/
theorem polynomial_equality (x : ℝ) (h k : ℝ → ℝ) 
    (h_def : h = fun x => x^4 - 3*x^2 + 1)
    (sum_eq : ∀ x, h x + k x = 3*x^2 + 2*x - 5) :
  k x = -x^4 + 6*x^2 + 2*x - 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l169_16960


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l169_16956

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3}
def B : Set Nat := {3, 4}

theorem complement_intersection_theorem : 
  (U \ A) ∩ (U \ B) = {2, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l169_16956


namespace NUMINAMATH_CALUDE_historians_contemporaries_probability_l169_16996

/-- Represents the number of years in the time period --/
def totalYears : ℕ := 300

/-- Represents the lifespan of each historian --/
def lifespan : ℕ := 80

/-- Represents the probability space of possible birth year combinations --/
def totalPossibilities : ℕ := totalYears * totalYears

/-- Represents the number of favorable outcomes (contemporaneous birth year combinations) --/
def favorableOutcomes : ℕ := totalPossibilities - 2 * ((totalYears - lifespan) * (totalYears - lifespan) / 2)

/-- The probability of two historians being contemporaries --/
def probabilityOfContemporaries : ℚ := favorableOutcomes / totalPossibilities

theorem historians_contemporaries_probability :
  probabilityOfContemporaries = 104 / 225 := by
  sorry

end NUMINAMATH_CALUDE_historians_contemporaries_probability_l169_16996


namespace NUMINAMATH_CALUDE_sum_of_valid_starting_numbers_l169_16939

def machine_rule (n : ℕ) : ℕ :=
  if n % 2 = 0 then n + 5 else 2 * n

def iterate_machine (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => machine_rule (iterate_machine n k)

def valid_starting_numbers : List ℕ :=
  (List.range 55).filter (λ n => iterate_machine n 4 = 54)

theorem sum_of_valid_starting_numbers :
  valid_starting_numbers.sum = 39 :=
sorry

end NUMINAMATH_CALUDE_sum_of_valid_starting_numbers_l169_16939


namespace NUMINAMATH_CALUDE_johns_running_distance_l169_16900

/-- The distance John ran each morning -/
def daily_distance (total_distance : ℕ) (days : ℕ) : ℚ :=
  total_distance / days

theorem johns_running_distance :
  daily_distance 10200 6 = 1700 := by
  sorry

end NUMINAMATH_CALUDE_johns_running_distance_l169_16900


namespace NUMINAMATH_CALUDE_specific_tree_height_l169_16986

/-- Represents the height of a tree after a given number of years -/
def tree_height (initial_height : ℝ) (annual_growth : ℝ) (years : ℝ) : ℝ :=
  initial_height + annual_growth * years

/-- Theorem stating the height of a specific tree after x years -/
theorem specific_tree_height (x : ℝ) :
  tree_height 2.5 0.22 x = 2.5 + 0.22 * x := by
  sorry

end NUMINAMATH_CALUDE_specific_tree_height_l169_16986


namespace NUMINAMATH_CALUDE_bus_schedule_hours_l169_16908

/-- The number of hours per day that buses leave the station -/
def hours_per_day (total_buses : ℕ) (days : ℕ) (buses_per_hour : ℕ) : ℚ :=
  (total_buses : ℚ) / (days : ℚ) / (buses_per_hour : ℚ)

/-- Theorem stating that under given conditions, buses leave the station for 12 hours per day -/
theorem bus_schedule_hours (total_buses : ℕ) (days : ℕ) (buses_per_hour : ℕ)
    (h1 : total_buses = 120)
    (h2 : days = 5)
    (h3 : buses_per_hour = 2) :
    hours_per_day total_buses days buses_per_hour = 12 := by
  sorry

end NUMINAMATH_CALUDE_bus_schedule_hours_l169_16908


namespace NUMINAMATH_CALUDE_cube_painted_faces_l169_16961

/-- Calculates the number of unit cubes with exactly one painted side in a painted cube of given side length -/
def painted_faces (side_length : ℕ) : ℕ :=
  if side_length ≤ 2 then 0
  else 6 * (side_length - 2)^2

/-- The problem statement -/
theorem cube_painted_faces :
  painted_faces 5 = 54 := by
  sorry

end NUMINAMATH_CALUDE_cube_painted_faces_l169_16961


namespace NUMINAMATH_CALUDE_f_monotone_implies_a_range_l169_16927

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 3) * x - 1 else Real.log x / Real.log a

-- Define the property of being monotonically increasing
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem f_monotone_implies_a_range (a : ℝ) :
  MonotonicallyIncreasing (f a) → 3 < a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_implies_a_range_l169_16927


namespace NUMINAMATH_CALUDE_convex_ngon_diagonals_and_triangles_l169_16994

/-- A convex n-gon where no three diagonals intersect at the same point -/
structure ConvexNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_convex : sorry
  no_triple_intersection : sorry

/-- The number of diagonals in a convex n-gon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of internal triangles formed by sides and diagonals of a convex n-gon -/
def num_internal_triangles (n : ℕ) : ℕ :=
  Nat.choose n 3 + 4 * Nat.choose n 4 + 5 * Nat.choose n 5 + Nat.choose n 6

/-- Theorem stating the number of diagonals and internal triangles in a convex n-gon -/
theorem convex_ngon_diagonals_and_triangles (n : ℕ) (A : ConvexNGon n) :
  (num_diagonals n = n * (n - 3) / 2) ∧
  (num_internal_triangles n = Nat.choose n 3 + 4 * Nat.choose n 4 + 5 * Nat.choose n 5 + Nat.choose n 6) :=
by sorry

end NUMINAMATH_CALUDE_convex_ngon_diagonals_and_triangles_l169_16994


namespace NUMINAMATH_CALUDE_quadrilateral_area_l169_16925

/-- A quadrilateral with vertices at (3,-1), (-1,4), (2,3), and (9,9) -/
def Quadrilateral : List (ℝ × ℝ) := [(3, -1), (-1, 4), (2, 3), (9, 9)]

/-- One side of the quadrilateral is horizontal -/
axiom horizontal_side : ∃ (a b : ℝ) (y : ℝ), ((a, y) ∈ Quadrilateral ∧ (b, y) ∈ Quadrilateral) ∧ a ≠ b

/-- The area of the quadrilateral -/
def area : ℝ := 22.5

/-- Theorem: The area of the quadrilateral is 22.5 -/
theorem quadrilateral_area : 
  let vertices := Quadrilateral
  area = (1/2) * abs (
    (vertices[0].1 * vertices[1].2 + vertices[1].1 * vertices[2].2 + 
     vertices[2].1 * vertices[3].2 + vertices[3].1 * vertices[0].2) - 
    (vertices[1].1 * vertices[0].2 + vertices[2].1 * vertices[1].2 + 
     vertices[3].1 * vertices[2].2 + vertices[0].1 * vertices[3].2)
  ) := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l169_16925


namespace NUMINAMATH_CALUDE_trapezoid_area_l169_16946

-- Define the trapezoid ABCD
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  height : ℝ

-- Define the circle γ
structure Circle where
  radius : ℝ
  center_in_trapezoid : Bool
  tangent_to_AB_BC_DA : Bool
  arc_angle : ℝ

-- Define the problem
def trapezoid_circle_problem (ABCD : Trapezoid) (γ : Circle) : Prop :=
  ABCD.AB = 10 ∧
  ABCD.CD = 15 ∧
  γ.radius = 6 ∧
  γ.center_in_trapezoid = true ∧
  γ.tangent_to_AB_BC_DA = true ∧
  γ.arc_angle = 120

-- Theorem statement
theorem trapezoid_area (ABCD : Trapezoid) (γ : Circle) :
  trapezoid_circle_problem ABCD γ →
  (ABCD.AB + ABCD.CD) * ABCD.height / 2 = 225 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l169_16946


namespace NUMINAMATH_CALUDE_school_start_time_proof_l169_16912

structure SchoolCommute where
  normalTime : ℕ
  redLightStops : ℕ
  redLightTime : ℕ
  constructionTime : ℕ
  departureTime : Nat × Nat
  lateMinutes : ℕ

def schoolStartTime (c : SchoolCommute) : Nat × Nat :=
  sorry

theorem school_start_time_proof (c : SchoolCommute) 
  (h1 : c.normalTime = 30)
  (h2 : c.redLightStops = 4)
  (h3 : c.redLightTime = 3)
  (h4 : c.constructionTime = 10)
  (h5 : c.departureTime = (7, 15))
  (h6 : c.lateMinutes = 7) :
  schoolStartTime c = (8, 0) :=
by
  sorry

end NUMINAMATH_CALUDE_school_start_time_proof_l169_16912


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l169_16953

theorem negative_fraction_comparison : -3/5 > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l169_16953


namespace NUMINAMATH_CALUDE_area_of_absolute_value_equation_l169_16942

def enclosed_area (f : ℝ × ℝ → ℝ) : ℝ := sorry

theorem area_of_absolute_value_equation :
  enclosed_area (fun (x, y) => |x| + |3 * y| + |x - y| - 20) = 200 / 3 := by sorry

end NUMINAMATH_CALUDE_area_of_absolute_value_equation_l169_16942


namespace NUMINAMATH_CALUDE_intersection_trajectory_l169_16931

/-- 
Given points A(a,0) and B(b,0) on the x-axis and a point C(0,c) on the y-axis,
prove that the trajectory of the intersection point of line l (passing through O(0,0) 
and perpendicular to AC) and line BC is described by the given equation.
-/
theorem intersection_trajectory 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hab : a ≠ b) :
  ∃ (x y : ℝ → ℝ), 
    ∀ (c : ℝ), c ≠ 0 →
      (x c - b/2)^2 / (b^2/4) + (y c)^2 / (a*b/4) = 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_trajectory_l169_16931


namespace NUMINAMATH_CALUDE_spencer_sessions_per_day_l169_16983

/-- Represents the jumping routine of Spencer --/
structure JumpingRoutine where
  jumps_per_minute : ℕ
  minutes_per_session : ℕ
  total_jumps : ℕ
  total_days : ℕ

/-- Calculates the number of sessions per day for Spencer's jumping routine --/
def sessions_per_day (routine : JumpingRoutine) : ℚ :=
  (routine.total_jumps / routine.total_days) / (routine.jumps_per_minute * routine.minutes_per_session)

/-- Theorem stating that Spencer's jumping routine results in 2 sessions per day --/
theorem spencer_sessions_per_day :
  let routine := JumpingRoutine.mk 4 10 400 5
  sessions_per_day routine = 2 := by
  sorry

end NUMINAMATH_CALUDE_spencer_sessions_per_day_l169_16983


namespace NUMINAMATH_CALUDE_linear_function_decreasing_values_l169_16951

theorem linear_function_decreasing_values (x₁ : ℝ) : 
  let f := fun (x : ℝ) => -3 * x + 1
  let y₁ := f x₁
  let y₂ := f (x₁ + 1)
  let y₃ := f (x₁ + 2)
  y₃ < y₂ ∧ y₂ < y₁ := by
sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_values_l169_16951


namespace NUMINAMATH_CALUDE_power_of_power_l169_16941

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l169_16941


namespace NUMINAMATH_CALUDE_inequality_theorem_l169_16905

theorem inequality_theorem (φ : Real) (h : φ ∈ Set.Ioo 0 (Real.pi / 2)) :
  Real.sin (Real.cos φ) < Real.cos φ ∧ Real.cos φ < Real.cos (Real.sin φ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l169_16905


namespace NUMINAMATH_CALUDE_mod_seven_equality_l169_16929

theorem mod_seven_equality : (47 ^ 2049 - 18 ^ 2049) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mod_seven_equality_l169_16929


namespace NUMINAMATH_CALUDE_customers_not_buying_coffee_l169_16974

theorem customers_not_buying_coffee (total_customers : ℕ) (coffee_fraction : ℚ) : 
  total_customers = 25 → coffee_fraction = 3/5 → 
  total_customers - (coffee_fraction * total_customers).floor = 10 :=
by sorry

end NUMINAMATH_CALUDE_customers_not_buying_coffee_l169_16974


namespace NUMINAMATH_CALUDE_shooter_probability_l169_16998

theorem shooter_probability (p : ℝ) (h : p = 2/3) :
  1 - (1 - p)^3 = 26/27 := by
  sorry

end NUMINAMATH_CALUDE_shooter_probability_l169_16998


namespace NUMINAMATH_CALUDE_subsidy_at_160_max_revenue_l169_16969

/-- Represents the monthly sales and revenue model for Wang Hua's clothing business -/
structure ClothingBusinessModel where
  costPrice : ℝ
  subsidy : ℝ
  demandFunction : ℝ → ℝ
  revenueFunction : ℝ → ℝ

/-- The specific model for Wang Hua's business -/
def wangHuaModel : ClothingBusinessModel :=
  { costPrice := 100
    subsidy := 20
    demandFunction := λ x => -3 * x + 900
    revenueFunction := λ x => (x - 80) * (-3 * x + 900) }

/-- Theorem for the government subsidy at a specific selling price -/
theorem subsidy_at_160 (model : ClothingBusinessModel := wangHuaModel) :
  model.subsidy * model.demandFunction 160 = 8400 := by sorry

/-- Theorem for the maximum revenue and the price at which it occurs -/
theorem max_revenue (model : ClothingBusinessModel := wangHuaModel) :
  (∃ x_max : ℝ, x_max = 190 ∧ 
   (∀ x : ℝ, model.revenueFunction x ≤ model.revenueFunction x_max) ∧
   model.revenueFunction x_max = 36300) := by sorry

end NUMINAMATH_CALUDE_subsidy_at_160_max_revenue_l169_16969
