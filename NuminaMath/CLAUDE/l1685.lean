import Mathlib

namespace NUMINAMATH_CALUDE_andrey_solved_half_l1685_168581

theorem andrey_solved_half (N : ℕ) (x : ℕ) : 
  (N - x - (N - x) / 3 = N / 3) → 
  (x : ℚ) / N = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_andrey_solved_half_l1685_168581


namespace NUMINAMATH_CALUDE_transposition_changes_cycles_even_permutation_iff_even_diff_l1685_168512

/-- A permutation of numbers 1 to n -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- Number of cycles in a permutation -/
def numCycles (σ : Permutation n) : ℕ := sorry

/-- Perform a transposition on a permutation -/
def transpose (σ : Permutation n) (i j : Fin n) : Permutation n := sorry

/-- A permutation is even -/
def isEven (σ : Permutation n) : Prop := sorry

theorem transposition_changes_cycles (n : ℕ) (σ : Permutation n) (i j : Fin n) :
  ∃ k : ℤ, k = 1 ∨ k = -1 ∧ numCycles (transpose σ i j) = numCycles σ + k :=
sorry

theorem even_permutation_iff_even_diff (n : ℕ) (σ : Permutation n) :
  isEven σ ↔ Even (n - numCycles σ) :=
sorry

end NUMINAMATH_CALUDE_transposition_changes_cycles_even_permutation_iff_even_diff_l1685_168512


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l1685_168596

/-- The line y = kx + 2 intersects the hyperbola x^2 - y^2 = 2 at exactly one point if and only if k = ±1 or k = ±√3 -/
theorem line_hyperbola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 + 2 ∧ p.1^2 - p.2^2 = 2) ↔ 
  (k = 1 ∨ k = -1 ∨ k = Real.sqrt 3 ∨ k = -Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l1685_168596


namespace NUMINAMATH_CALUDE_f_mapping_result_l1685_168575

def A : Set (ℝ × ℝ) := Set.univ

def B : Set (ℝ × ℝ) := Set.univ

def f : (ℝ × ℝ) → (ℝ × ℝ) := λ (x, y) ↦ (x - y, x + y)

theorem f_mapping_result : f (-1, 2) = (-3, 1) := by
  sorry

end NUMINAMATH_CALUDE_f_mapping_result_l1685_168575


namespace NUMINAMATH_CALUDE__l1685_168570

def main_theorem (f : ℝ → ℝ) (h1 : ∀ p q, f (p + q) = f p * f q) (h2 : f 1 = 3) : 
  (f 1 ^ 2 + f 2) / f 1 + (f 2 ^ 2 + f 4) / f 3 + (f 3 ^ 2 + f 6) / f 5 + 
  (f 4 ^ 2 + f 8) / f 7 + (f 5 ^ 2 + f 10) / f 9 = 30 := by
  sorry

end NUMINAMATH_CALUDE__l1685_168570


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1685_168555

-- First expression
theorem simplify_expression_1 (x y : ℝ) :
  4 * y^2 + 3 * x - 5 + 6 - 4 * x - 2 * y^2 = 2 * y^2 - x + 1 := by sorry

-- Second expression
theorem simplify_expression_2 (m n : ℝ) :
  3/2 * (m^2 - m*n) - 2 * (m*n + m^2) = -1/2 * m^2 - 7/2 * m*n := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1685_168555


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l1685_168576

/-- A trinomial of the form ax^2 + bx + c is a perfect square if and only if
    there exist real numbers p and q such that ax^2 + bx + c = (px + q)^2 -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_condition (k : ℝ) :
  IsPerfectSquareTrinomial 1 (-k) 9 → k = 6 ∨ k = -6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l1685_168576


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l1685_168510

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log (2 * x)

theorem min_distance_between_curves : 
  ∃ (d : ℝ), d = Real.sqrt 2 * (1 - Real.log 2) ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → 
  Real.sqrt ((x - y)^2 + (f x - g y)^2) ≥ d :=
sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l1685_168510


namespace NUMINAMATH_CALUDE_min_value_abc_l1685_168577

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 + 2*a*b + 2*a*c + 4*b*c = 12) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → x^2 + 2*x*y + 2*x*z + 4*y*z = 12 → 
  a + b + c ≤ x + y + z ∧ a + b + c ≥ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abc_l1685_168577


namespace NUMINAMATH_CALUDE_not_p_or_q_false_implies_at_least_one_true_l1685_168528

theorem not_p_or_q_false_implies_at_least_one_true (p q : Prop) :
  ¬(¬(p ∨ q)) → (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_q_false_implies_at_least_one_true_l1685_168528


namespace NUMINAMATH_CALUDE_even_sine_function_phi_l1685_168580

/-- Given a function f(x) = sin((x + φ) / 3) where φ ∈ [0, 2π],
    prove that if f is even, then φ = 3π/2 -/
theorem even_sine_function_phi (φ : ℝ) (h1 : φ ∈ Set.Icc 0 (2 * Real.pi)) :
  (∀ x, Real.sin ((x + φ) / 3) = Real.sin ((-x + φ) / 3)) →
  φ = 3 * Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_even_sine_function_phi_l1685_168580


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l1685_168552

/-- Converts a list of binary digits to a natural number -/
def binary_to_nat (digits : List Bool) : ℕ :=
  digits.foldl (fun acc d => 2 * acc + if d then 1 else 0) 0

/-- Converts a natural number to a list of binary digits -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true, false, true]  -- 101101₂
  let b := [true, true, false, true]               -- 1101₂
  let product := [true, false, false, false, true, false, false, false, true, true, true]  -- 10001000111₂
  binary_to_nat a * binary_to_nat b = binary_to_nat product := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l1685_168552


namespace NUMINAMATH_CALUDE_stratified_sampling_third_year_students_l1685_168587

theorem stratified_sampling_third_year_students 
  (total_students : ℕ) 
  (third_year_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 1600) 
  (h2 : third_year_students = 400) 
  (h3 : sample_size = 160) :
  (sample_size * third_year_students) / total_students = 40 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_year_students_l1685_168587


namespace NUMINAMATH_CALUDE_bakery_rolls_combinations_l1685_168592

theorem bakery_rolls_combinations :
  let total_rolls : ℕ := 8
  let num_kinds : ℕ := 4
  let rolls_to_distribute : ℕ := total_rolls - num_kinds
  (Nat.choose (rolls_to_distribute + num_kinds - 1) (num_kinds - 1)) = 35 := by
  sorry

end NUMINAMATH_CALUDE_bakery_rolls_combinations_l1685_168592


namespace NUMINAMATH_CALUDE_min_value_theorem_l1685_168571

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_bisect : ∀ (x y : ℝ), 2*x + y - 2 = 0 → x^2 + y^2 - 2*a*x - 4*b*y + 1 = 0 → 
    ∃ (x' y' : ℝ), x'^2 + y'^2 - 2*a*x' - 4*b*y' + 1 = 0 ∧ 2*x' + y' - 2 ≠ 0) : 
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → 2/a' + 1/(2*b') ≥ 9/2) ∧ 
  (∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ 2/a' + 1/(2*b') = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1685_168571


namespace NUMINAMATH_CALUDE_typing_time_proof_l1685_168560

def typing_speed : ℕ := 38
def paper_length : ℕ := 4560
def minutes_per_hour : ℕ := 60

theorem typing_time_proof :
  (paper_length / typing_speed : ℚ) / minutes_per_hour = 2 := by
  sorry

end NUMINAMATH_CALUDE_typing_time_proof_l1685_168560


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l1685_168578

-- Define the sets N and M
def N (q : ℝ) : Set ℝ := {x | x^2 + 6*x - q = 0}
def M (p : ℝ) : Set ℝ := {x | x^2 - p*x + 6 = 0}

-- State the theorem
theorem intersection_implies_sum (p q : ℝ) :
  M p ∩ N q = {2} → p + q = 21 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l1685_168578


namespace NUMINAMATH_CALUDE_odd_number_as_difference_of_squares_l1685_168548

theorem odd_number_as_difference_of_squares (n : ℤ) : 
  2 * n + 1 = (n + 1)^2 - n^2 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_as_difference_of_squares_l1685_168548


namespace NUMINAMATH_CALUDE_equivalent_discount_l1685_168565

theorem equivalent_discount (original_price : ℝ) (h : original_price > 0) :
  let first_discount := 0.15
  let second_discount := 0.25
  let price_after_first := original_price * (1 - first_discount)
  let price_after_second := price_after_first * (1 - second_discount)
  let equivalent_discount := 1 - (price_after_second / original_price)
  equivalent_discount = 0.3625 := by
sorry

end NUMINAMATH_CALUDE_equivalent_discount_l1685_168565


namespace NUMINAMATH_CALUDE_problem_statement_l1685_168568

theorem problem_statement (x y a : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x + y + 8 = x * y) (h2 : ∀ x y, x > 0 → y > 0 → x + y + 8 = x * y → (x + y)^2 - a*(x + y) + 1 ≥ 0) : 
  a ≤ 65/8 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1685_168568


namespace NUMINAMATH_CALUDE_spy_arrangement_exists_l1685_168529

-- Define the board
def Board := Fin 6 → Fin 6 → Bool

-- Define the direction a spy can face
inductive Direction
| North
| East
| South
| West

-- Define a spy's position and direction
structure Spy where
  row : Fin 6
  col : Fin 6
  dir : Direction

-- Define the visibility function for a spy
def canSee (s : Spy) (r : Fin 6) (c : Fin 6) : Prop :=
  match s.dir with
  | Direction.North => 
      (s.row > r && s.row - r ≤ 2 && s.col = c) || 
      (s.row = r && (s.col = c + 1 || s.col + 1 = c))
  | Direction.East => 
      (s.col < c && c - s.col ≤ 2 && s.row = r) || 
      (s.col = c && (s.row = r + 1 || s.row + 1 = r))
  | Direction.South => 
      (s.row < r && r - s.row ≤ 2 && s.col = c) || 
      (s.row = r && (s.col = c + 1 || s.col + 1 = c))
  | Direction.West => 
      (s.col > c && s.col - c ≤ 2 && s.row = r) || 
      (s.col = c && (s.row = r + 1 || s.row + 1 = r))

-- Define a valid arrangement of spies
def validArrangement (spies : List Spy) : Prop :=
  spies.length = 18 ∧
  ∀ s1 s2, s1 ∈ spies → s2 ∈ spies → s1 ≠ s2 →
    ¬(canSee s1 s2.row s2.col) ∧ ¬(canSee s2 s1.row s1.col)

-- Theorem: There exists a valid arrangement of 18 spies
theorem spy_arrangement_exists : ∃ spies : List Spy, validArrangement spies := by
  sorry

end NUMINAMATH_CALUDE_spy_arrangement_exists_l1685_168529


namespace NUMINAMATH_CALUDE_math_problem_distribution_l1685_168505

theorem math_problem_distribution :
  let num_problems : ℕ := 7
  let num_friends : ℕ := 12
  (num_friends ^ num_problems : ℕ) = 35831808 :=
by sorry

end NUMINAMATH_CALUDE_math_problem_distribution_l1685_168505


namespace NUMINAMATH_CALUDE_quilt_transformation_l1685_168525

/-- Given a rectangular quilt with width 6 feet and an unknown length, and a square quilt with side length 12 feet, 
    if their areas are equal, then the length of the rectangular quilt is 24 feet. -/
theorem quilt_transformation (length : ℝ) : 
  (6 * length = 12 * 12) → length = 24 := by
  sorry

end NUMINAMATH_CALUDE_quilt_transformation_l1685_168525


namespace NUMINAMATH_CALUDE_x_plus_y_values_l1685_168554

theorem x_plus_y_values (x y : ℝ) (hx : |x| = 3) (hy : |y| = 6) (hxy : x > y) :
  (x + y = -3 ∨ x + y = -9) ∧ ∀ z, (x + y = z → z = -3 ∨ z = -9) :=
by sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l1685_168554


namespace NUMINAMATH_CALUDE_necessary_sufficient_condition_l1685_168522

theorem necessary_sufficient_condition (a b : ℝ) :
  a * |a + b| < |a| * (a + b) ↔ a < 0 ∧ b > -a := by
  sorry

end NUMINAMATH_CALUDE_necessary_sufficient_condition_l1685_168522


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l1685_168524

theorem simplify_sqrt_sum : 
  Real.sqrt (10 + 6 * Real.sqrt 3) + Real.sqrt (10 - 6 * Real.sqrt 3) = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l1685_168524


namespace NUMINAMATH_CALUDE_grocery_stock_problem_l1685_168579

theorem grocery_stock_problem (asparagus_bundles : ℕ) (asparagus_price : ℚ)
  (grape_boxes : ℕ) (grape_price : ℚ)
  (apple_price : ℚ) (total_worth : ℚ) :
  asparagus_bundles = 60 →
  asparagus_price = 3 →
  grape_boxes = 40 →
  grape_price = (5/2) →
  apple_price = (1/2) →
  total_worth = 630 →
  (total_worth - (asparagus_bundles * asparagus_price + grape_boxes * grape_price)) / apple_price = 700 := by
  sorry

end NUMINAMATH_CALUDE_grocery_stock_problem_l1685_168579


namespace NUMINAMATH_CALUDE_right_triangle_with_three_isosceles_l1685_168527

/-- A right-angled triangle that can be divided into three isosceles triangles has acute angles of 22.5° and 67.5°. -/
theorem right_triangle_with_three_isosceles (α β : Real) : 
  α + β = 90 → -- The sum of acute angles in a right triangle is 90°
  (∃ (γ : Real), γ = 90 ∧ 2*α + 2*α = γ) → -- One of the isosceles triangles has a right angle and two equal angles of 2α
  (α = 22.5 ∧ β = 67.5) := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_with_three_isosceles_l1685_168527


namespace NUMINAMATH_CALUDE_product_mod_seven_l1685_168543

theorem product_mod_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l1685_168543


namespace NUMINAMATH_CALUDE_third_chapter_pages_l1685_168589

theorem third_chapter_pages (total_pages first_chapter second_chapter : ℕ) 
  (h1 : total_pages = 125)
  (h2 : first_chapter = 66)
  (h3 : second_chapter = 35) :
  total_pages - (first_chapter + second_chapter) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_third_chapter_pages_l1685_168589


namespace NUMINAMATH_CALUDE_greatest_piece_length_l1685_168537

theorem greatest_piece_length (rope1 rope2 rope3 : ℕ) 
  (h1 : rope1 = 28) (h2 : rope2 = 45) (h3 : rope3 = 63) : 
  Nat.gcd rope1 (Nat.gcd rope2 rope3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_greatest_piece_length_l1685_168537


namespace NUMINAMATH_CALUDE_sqrt_720_simplification_l1685_168557

theorem sqrt_720_simplification : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_720_simplification_l1685_168557


namespace NUMINAMATH_CALUDE_hexagonal_grid_triangles_l1685_168586

/-- Represents a hexagonal grid -/
structure HexagonalGrid :=
  (units : ℕ)

/-- Counts the number of triangles in a hexagonal grid -/
def count_triangles (grid : HexagonalGrid) : ℕ :=
  sorry

/-- Theorem: A hexagonal grid with 10 units contains 20 triangles -/
theorem hexagonal_grid_triangles :
  ∀ (grid : HexagonalGrid), grid.units = 10 → count_triangles grid = 20 :=
by sorry

end NUMINAMATH_CALUDE_hexagonal_grid_triangles_l1685_168586


namespace NUMINAMATH_CALUDE_parabola_chord_slope_l1685_168513

/-- Given a parabola y² = 2px, a point Q(q, 0) where q < 0, and a line x = s where p > 0 and s > 0,
    this theorem proves the slope of a chord through Q that intersects the parabola at two points
    equidistant from the line x = s. -/
theorem parabola_chord_slope (p s q : ℝ) (hp : p > 0) (hs : s > 0) (hq : q < 0) (h_feasible : s ≥ -q) :
  ∃ m : ℝ, m ^ 2 = p / (s - q) ∧
    ∀ x y : ℝ, y ^ 2 = 2 * p * x →
      y = m * (x - q) →
      ∃ x' y' : ℝ, y' ^ 2 = 2 * p * x' ∧
                  y' = m * (x' - q) ∧
                  x + x' = 2 * s :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_chord_slope_l1685_168513


namespace NUMINAMATH_CALUDE_revenue_maximized_at_optimal_price_l1685_168542

/-- Revenue function for gadget sales -/
def R (p : ℝ) : ℝ := p * (200 - 4 * p)

/-- The price that maximizes revenue -/
def optimal_price : ℝ := 25

theorem revenue_maximized_at_optimal_price :
  ∀ p : ℝ, p ≤ 40 → R p ≤ R optimal_price := by sorry

end NUMINAMATH_CALUDE_revenue_maximized_at_optimal_price_l1685_168542


namespace NUMINAMATH_CALUDE_solution_system_l1685_168569

theorem solution_system (x y m n : ℤ) : 
  x = 2 ∧ y = -3 ∧ x + y = m ∧ 2 * x - y = n → m - n = -8 := by
  sorry

end NUMINAMATH_CALUDE_solution_system_l1685_168569


namespace NUMINAMATH_CALUDE_function_range_l1685_168562

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (Real.exp x + x - a)

theorem function_range (a : ℝ) :
  (∃ y₀ : ℝ, y₀ ∈ Set.Icc (-1) 1 ∧ f a (f a y₀) = y₀) →
  a ∈ Set.Icc 1 (Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_function_range_l1685_168562


namespace NUMINAMATH_CALUDE_f_properties_l1685_168507

-- Define the function f
def f (x : ℝ) := -x^2 - 4*x + 1

-- Theorem statement
theorem f_properties :
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≤ m ∧ (∃ (x₀ : ℝ), f x₀ = m) ∧ m = 5) ∧
  (∀ (x y : ℝ), x < y ∧ y < -2 → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1685_168507


namespace NUMINAMATH_CALUDE_solution_set_min_value_l1685_168582

-- Part I
def f (x : ℝ) : ℝ := |3 * x - 1| + |x + 3|

theorem solution_set (x : ℝ) : f x ≥ 4 ↔ x ≤ 0 ∨ x ≥ 1/2 := by sorry

-- Part II
def g (b c x : ℝ) : ℝ := |x - b| + |x + c|

theorem min_value (b c : ℝ) (hb : b > 0) (hc : c > 0) 
  (h_min : ∃ (x : ℝ), ∀ (y : ℝ), g b c x ≤ g b c y) 
  (h_eq : ∃ (x : ℝ), g b c x = 1) :
  (1 / b + 1 / c) ≥ 4 ∧ ∃ (b₀ c₀ : ℝ), 1 / b₀ + 1 / c₀ = 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_min_value_l1685_168582


namespace NUMINAMATH_CALUDE_max_sum_squares_l1685_168597

theorem max_sum_squares (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + 2*b + 3*c = 1) :
  ∃ (max : ℝ), max = 1 ∧ a^2 + b^2 + c^2 ≤ max ∧ ∃ (a' b' c' : ℝ), 
    0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + 2*b' + 3*c' = 1 ∧ a'^2 + b'^2 + c'^2 = max :=
sorry

end NUMINAMATH_CALUDE_max_sum_squares_l1685_168597


namespace NUMINAMATH_CALUDE_toy_store_revenue_ratio_l1685_168500

theorem toy_store_revenue_ratio :
  ∀ (N D J : ℝ),
  J = (1/3) * N →
  D = 3.75 * ((N + J) / 2) →
  N / D = 2/5 := by
sorry

end NUMINAMATH_CALUDE_toy_store_revenue_ratio_l1685_168500


namespace NUMINAMATH_CALUDE_businessmen_drinks_l1685_168503

theorem businessmen_drinks (total : ℕ) (coffee : ℕ) (tea : ℕ) (both : ℕ) :
  total = 30 →
  coffee = 15 →
  tea = 13 →
  both = 6 →
  total - (coffee + tea - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_drinks_l1685_168503


namespace NUMINAMATH_CALUDE_train_speed_l1685_168501

theorem train_speed 
  (n : ℝ) 
  (a : ℝ) 
  (b : ℝ) 
  (c : ℝ) 
  (h1 : n > 0) 
  (h2 : a > c) 
  (h3 : b > 0) : 
  ∃ (speed : ℝ), speed = (b * (n + 1)) / (a - c) := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1685_168501


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1685_168546

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 + 5 * x^3 - 3 * x + 7) + (-x^4 + 4 * x^2 - 5 * x + 2) =
  x^4 + 5 * x^3 + 4 * x^2 - 8 * x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1685_168546


namespace NUMINAMATH_CALUDE_floor_sqrt_33_squared_l1685_168502

theorem floor_sqrt_33_squared : ⌊Real.sqrt 33⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_33_squared_l1685_168502


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1685_168553

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z := i^2018 / (i^2019 - 1)
  Complex.im z = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1685_168553


namespace NUMINAMATH_CALUDE_water_jars_theorem_l1685_168506

theorem water_jars_theorem (S L : ℚ) (h1 : S > 0) (h2 : L > 0) (h3 : S ≠ L) : 
  (1/3 : ℚ) * S = (1/2 : ℚ) * L → (1/2 : ℚ) * L + (1/3 : ℚ) * S = L := by
  sorry

end NUMINAMATH_CALUDE_water_jars_theorem_l1685_168506


namespace NUMINAMATH_CALUDE_purple_marble_probability_l1685_168504

structure Bag where
  red : ℕ
  green : ℕ
  orange : ℕ
  purple : ℕ

def bagX : Bag := { red := 5, green := 3, orange := 0, purple := 0 }
def bagY : Bag := { red := 0, green := 0, orange := 8, purple := 2 }
def bagZ : Bag := { red := 0, green := 0, orange := 3, purple := 7 }

def total_marbles (b : Bag) : ℕ := b.red + b.green + b.orange + b.purple

def prob_red (b : Bag) : ℚ := b.red / (total_marbles b)
def prob_green (b : Bag) : ℚ := b.green / (total_marbles b)
def prob_purple (b : Bag) : ℚ := b.purple / (total_marbles b)

theorem purple_marble_probability :
  let p_red_X := prob_red bagX
  let p_green_X := prob_green bagX
  let p_purple_Y := prob_purple bagY
  let p_purple_Z := prob_purple bagZ
  p_red_X * p_purple_Y + p_green_X * p_purple_Z = 31 / 80 := by
  sorry

end NUMINAMATH_CALUDE_purple_marble_probability_l1685_168504


namespace NUMINAMATH_CALUDE_fraction_zero_solution_l1685_168511

theorem fraction_zero_solution (x : ℝ) : 
  (x^2 - 4) / (x + 4) = 0 ∧ x ≠ -4 → x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_solution_l1685_168511


namespace NUMINAMATH_CALUDE_circle_parabola_intersection_l1685_168550

theorem circle_parabola_intersection (b : ℝ) : 
  (∃ (a : ℝ), -- center of the circle (a, b)
    (∃ (r : ℝ), r > 0 ∧ -- radius of the circle
      (∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = r^2 → -- equation of the circle
        ((y = 3/4 * x^2) ∨ (x = 0 ∧ y = 0) ∨ (y = 3/4 * x + b)) -- intersections
      ) ∧
      (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ -- two distinct intersection points
        (3/4 * x1^2 = 3/4 * x1 + b) ∧ 
        (3/4 * x2^2 = 3/4 * x2 + b)
      )
    )
  ) → b = 25/12 :=
by sorry

end NUMINAMATH_CALUDE_circle_parabola_intersection_l1685_168550


namespace NUMINAMATH_CALUDE_smallest_solution_biquadratic_l1685_168533

theorem smallest_solution_biquadratic (x : ℝ) :
  x^4 - 26*x^2 + 169 = 0 → x ≥ -Real.sqrt 13 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_biquadratic_l1685_168533


namespace NUMINAMATH_CALUDE_sean_final_houses_l1685_168563

/-- Calculates the final number of houses Sean has after a series of transactions in Monopoly. -/
def final_houses (initial : ℕ) (traded_for_money : ℕ) (bought : ℕ) (traded_for_marvin : ℕ) (sold_for_atlantic : ℕ) (traded_for_hotels : ℕ) : ℕ :=
  initial - traded_for_money + bought - traded_for_marvin - sold_for_atlantic - traded_for_hotels

/-- Theorem stating that Sean ends up with 20 houses after the given transactions. -/
theorem sean_final_houses : 
  final_houses 45 15 18 5 7 16 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sean_final_houses_l1685_168563


namespace NUMINAMATH_CALUDE_range_of_sum_l1685_168518

def f (x : ℝ) := |2 - x^2|

theorem range_of_sum (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  ∃ (y : ℝ), 2 < y ∧ y < 2 * Real.sqrt 2 ∧ y = a + b :=
sorry

end NUMINAMATH_CALUDE_range_of_sum_l1685_168518


namespace NUMINAMATH_CALUDE_tax_revenue_consumption_relation_l1685_168551

/-- Proves that a 40% tax reduction and 25% revenue decrease results in a 25% consumption increase -/
theorem tax_revenue_consumption_relation 
  (T : ℝ) -- Original tax rate
  (C : ℝ) -- Original consumption
  (h1 : T > 0) -- Assumption: Original tax rate is positive
  (h2 : C > 0) -- Assumption: Original consumption is positive
  : 
  let new_tax := 0.6 * T -- New tax rate after 40% reduction
  let new_revenue := 0.75 * T * C -- New revenue after 25% decrease
  let new_consumption := new_revenue / new_tax -- New consumption
  new_consumption = 1.25 * C -- Proves 25% increase in consumption
  := by sorry

end NUMINAMATH_CALUDE_tax_revenue_consumption_relation_l1685_168551


namespace NUMINAMATH_CALUDE_power_product_equals_power_sum_l1685_168564

theorem power_product_equals_power_sum (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_power_sum_l1685_168564


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a5_l1685_168561

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 + a 8 = 12 → a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a5_l1685_168561


namespace NUMINAMATH_CALUDE_no_solution_exists_l1685_168549

theorem no_solution_exists : ¬∃ (a b : ℕ+), 
  (a * b + 90 = 24 * Nat.lcm a b + 15 * Nat.gcd a b) ∧ 
  (Nat.gcd a b = 3) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1685_168549


namespace NUMINAMATH_CALUDE_snow_probability_l1685_168574

theorem snow_probability (p : ℝ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 5) :
  1 - (1 - p)^n = 1023/1024 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l1685_168574


namespace NUMINAMATH_CALUDE_squirrel_acorns_l1685_168591

theorem squirrel_acorns (num_squirrels : ℕ) (acorns_collected : ℕ) (acorns_needed_per_squirrel : ℕ) :
  num_squirrels = 5 →
  acorns_collected = 575 →
  acorns_needed_per_squirrel = 130 →
  (num_squirrels * acorns_needed_per_squirrel - acorns_collected) / num_squirrels = 15 :=
by sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l1685_168591


namespace NUMINAMATH_CALUDE_parabola_translation_l1685_168584

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk 3 0 0
  let translated := translate original 2 (-3)
  y = 3 * x^2 → y = 3 * (x - 2)^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l1685_168584


namespace NUMINAMATH_CALUDE_max_ranked_participants_l1685_168517

/-- The maximum number of participants that can be awarded a rank in a chess tournament -/
theorem max_ranked_participants (n : ℕ) (rank_threshold : ℚ) : 
  n = 30 →
  rank_threshold = 60 / 100 →
  ∃ (max_ranked : ℕ), max_ranked = 23 ∧ 
    (∀ (ranked : ℕ), 
      ranked ≤ n ∧
      (ranked : ℚ) * rank_threshold * (n - 1 : ℚ) ≤ (n * (n - 1) / 2 : ℚ) →
      ranked ≤ max_ranked) :=
by sorry

end NUMINAMATH_CALUDE_max_ranked_participants_l1685_168517


namespace NUMINAMATH_CALUDE_f_continuous_at_2_l1685_168532

def f (x : ℝ) := -2 * x^2 - 5

theorem f_continuous_at_2 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by sorry

end NUMINAMATH_CALUDE_f_continuous_at_2_l1685_168532


namespace NUMINAMATH_CALUDE_present_worth_calculation_l1685_168514

/-- Calculates the present worth given the banker's gain, interest rate, and time period -/
def present_worth (bankers_gain : ℚ) (interest_rate : ℚ) (time : ℚ) : ℚ :=
  bankers_gain / (interest_rate * time)

/-- Theorem stating that under given conditions, the present worth is 120 -/
theorem present_worth_calculation :
  let bankers_gain : ℚ := 24
  let interest_rate : ℚ := 1/10  -- 10% as a rational number
  let time : ℚ := 2
  present_worth bankers_gain interest_rate time = 120 := by
sorry

#eval present_worth 24 (1/10) 2

end NUMINAMATH_CALUDE_present_worth_calculation_l1685_168514


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1685_168558

/-- A point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- A parabola defined by a focus point and a directrix line -/
structure Parabola :=
  (focus : Point) (directrix : Line)

/-- Represents the intersection points between a line and a parabola -/
inductive Intersection
  | NoIntersection
  | OnePoint (p : Point)
  | TwoPoints (p1 p2 : Point)

/-- 
Given a point F (focus), a line L, and a line D (directrix) in a plane,
there exists a construction method to find the intersection points (if any)
between L and the parabola defined by focus F and directrix D.
-/
theorem parabola_line_intersection
  (F : Point) (L D : Line) :
  ∃ (construct : Point → Line → Line → Intersection),
    construct F L D = Intersection.NoIntersection ∨
    (∃ p, construct F L D = Intersection.OnePoint p) ∨
    (∃ p1 p2, construct F L D = Intersection.TwoPoints p1 p2) :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l1685_168558


namespace NUMINAMATH_CALUDE_range_of_a_l1685_168545

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + x - 2 > 0
def q (x a : ℝ) : Prop := x > a

-- Define what it means for q to be sufficient but not necessary for p
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, q x a → p x) ∧ ¬(∀ x, p x → q x a)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  sufficient_not_necessary a → a ∈ Set.Ici 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1685_168545


namespace NUMINAMATH_CALUDE_hamburger_sales_proof_l1685_168559

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The average number of hamburgers sold per day -/
def average_daily_sales : ℕ := 9

/-- The total number of hamburgers sold in a week -/
def total_weekly_sales : ℕ := days_in_week * average_daily_sales

theorem hamburger_sales_proof : total_weekly_sales = 63 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_sales_proof_l1685_168559


namespace NUMINAMATH_CALUDE_cattle_train_speed_l1685_168585

/-- Represents the problem of determining the speed of a cattle train given specific conditions --/
theorem cattle_train_speed : ∀ (v : ℝ),
  (v > 0) →  -- The speed is positive
  (6 * v + 12 * v + 12 * (v - 33) = 1284) →  -- Total distance equation
  (v = 56) :=  -- The speed of the cattle train is 56 mph
by
  sorry

#check cattle_train_speed

end NUMINAMATH_CALUDE_cattle_train_speed_l1685_168585


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1685_168593

theorem unique_solution_quadratic (p : ℝ) (hp : p ≠ 0) :
  (∃! x : ℝ, p * x^2 - 8 * x + 2 = 0) ↔ p = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1685_168593


namespace NUMINAMATH_CALUDE_min_swaps_for_geese_order_l1685_168572

def initial_order : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
def final_order : List ℕ := List.range 20 |>.map (· + 1)

def count_inversions (l : List ℕ) : ℕ :=
  l.foldr (fun x acc => acc + (l.filter (· < x) |>.filter (fun y => l.indexOf y > l.indexOf x) |>.length)) 0

def min_swaps_to_sort (l : List ℕ) : ℕ := count_inversions l

theorem min_swaps_for_geese_order :
  min_swaps_to_sort initial_order = 55 :=
sorry

end NUMINAMATH_CALUDE_min_swaps_for_geese_order_l1685_168572


namespace NUMINAMATH_CALUDE_book_cost_problem_l1685_168536

theorem book_cost_problem (book_price : ℝ) : 
  (3 * book_price = 45) → (7 * book_price = 105) := by
  sorry

end NUMINAMATH_CALUDE_book_cost_problem_l1685_168536


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1685_168590

theorem sum_of_fractions : (3 : ℚ) / 7 + 9 / 12 = 33 / 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1685_168590


namespace NUMINAMATH_CALUDE_solve_equation_l1685_168538

theorem solve_equation : ∃ x : ℚ, (3 * x + 15 = (1/3) * (7 * x + 42)) ∧ (x = -3/2) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1685_168538


namespace NUMINAMATH_CALUDE_train_passing_time_l1685_168523

/-- Proves that a train of given length and speed takes a specific time to pass a stationary object. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 150 →
  train_speed_kmh = 36 →
  passing_time = 15 →
  passing_time = train_length / (train_speed_kmh * (5/18)) := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l1685_168523


namespace NUMINAMATH_CALUDE_four_X_three_l1685_168539

/-- The operation X defined for any two real numbers -/
def X (a b : ℝ) : ℝ := b + 7*a - a^3 + 2*b

/-- Theorem stating that 4 X 3 = -27 -/
theorem four_X_three : X 4 3 = -27 := by
  sorry

end NUMINAMATH_CALUDE_four_X_three_l1685_168539


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1685_168508

/-- The complex number z = i / (1 - i) is located in the second quadrant of the complex plane. -/
theorem complex_number_in_second_quadrant :
  let z : ℂ := Complex.I / (1 - Complex.I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1685_168508


namespace NUMINAMATH_CALUDE_t_range_l1685_168588

theorem t_range (t α β a : ℝ) :
  (t = Real.cos β ^ 3 + (α / 2) * Real.cos β) →
  (a ≤ t) →
  (t ≤ α - 5 * Real.cos β) →
  (-2/3 ≤ t ∧ t ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_t_range_l1685_168588


namespace NUMINAMATH_CALUDE_custom_mult_value_l1685_168531

/-- Custom multiplication operation for non-zero integers -/
def custom_mult (a b : ℤ) : ℚ :=
  1 / a + 1 / b

theorem custom_mult_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a + b = 12 → a * b = 32 → custom_mult a b = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_value_l1685_168531


namespace NUMINAMATH_CALUDE_percentage_difference_l1685_168516

theorem percentage_difference (x y z n : ℝ) : 
  x = 8 * y ∧ 
  y = 2 * |z - n| ∧ 
  z = 1.1 * n → 
  (x - y) / x * 100 = 87.5 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l1685_168516


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1685_168547

theorem complex_fraction_equality : 1 + (1 / (1 + (1 / (1 + 1)))) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1685_168547


namespace NUMINAMATH_CALUDE_chess_game_draw_probability_l1685_168540

theorem chess_game_draw_probability (prob_A_win prob_A_not_lose : ℝ) 
  (h1 : prob_A_win = 0.4)
  (h2 : prob_A_not_lose = 0.9) : 
  prob_A_not_lose - prob_A_win = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_draw_probability_l1685_168540


namespace NUMINAMATH_CALUDE_copy_machines_output_l1685_168519

/-- The number of copies made by two machines in a given time -/
def total_copies (rate1 rate2 time : ℕ) : ℕ :=
  rate1 * time + rate2 * time

/-- Theorem stating that two machines with rates 25 and 55 copies per minute
    make 2400 copies in 30 minutes -/
theorem copy_machines_output : total_copies 25 55 30 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_copy_machines_output_l1685_168519


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l1685_168595

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0006 + 0.00007 = 23467 / 100000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l1685_168595


namespace NUMINAMATH_CALUDE_class_group_size_l1685_168521

theorem class_group_size (boys girls groups : ℕ) 
  (h_boys : boys = 9) 
  (h_girls : girls = 12) 
  (h_groups : groups = 7) : 
  (boys + girls) / groups = 3 := by
sorry

end NUMINAMATH_CALUDE_class_group_size_l1685_168521


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1685_168544

/-- A line passing through (b, 0) and tangent to a circle of radius r centered at (0, 0),
    forming a triangle in the first quadrant with area S, has the equation rx - bry - rb = 0 --/
theorem tangent_line_equation (b r S : ℝ) (hb : b > 0) (hr : r > 0) (hS : S > 0) :
  ∃ (x y : ℝ → ℝ), 
    (∀ t, x t = b ∧ y t = 0) →  -- Line passes through (b, 0)
    (∃ t, (x t)^2 + (y t)^2 = r^2) →  -- Line touches the circle
    (∃ h, S = (1/2) * b * h) →  -- Triangle area
    (∀ t, r * (x t) - b * r * (y t) - r * b = 0) :=  -- Equation of the line
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1685_168544


namespace NUMINAMATH_CALUDE_H_surjective_l1685_168541

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2| - |x - 2| + x

-- State the theorem
theorem H_surjective : Function.Surjective H := by sorry

end NUMINAMATH_CALUDE_H_surjective_l1685_168541


namespace NUMINAMATH_CALUDE_tan_derivative_l1685_168594

open Real

theorem tan_derivative (x : ℝ) : deriv tan x = 1 / (cos x)^2 := by
  sorry

end NUMINAMATH_CALUDE_tan_derivative_l1685_168594


namespace NUMINAMATH_CALUDE_cats_given_away_l1685_168573

/-- Proves that the number of cats given away is 14, given the initial and remaining cat counts -/
theorem cats_given_away (initial_cats : ℝ) (remaining_cats : ℕ) 
  (h1 : initial_cats = 17.0) (h2 : remaining_cats = 3) : 
  initial_cats - remaining_cats = 14 := by
  sorry

end NUMINAMATH_CALUDE_cats_given_away_l1685_168573


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l1685_168556

theorem negative_fractions_comparison : -3/4 < -2/3 := by sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l1685_168556


namespace NUMINAMATH_CALUDE_complex_product_example_l1685_168526

-- Define the complex numbers
def z₁ : ℂ := 3 + 4 * Complex.I
def z₂ : ℂ := -2 - 3 * Complex.I

-- State the theorem
theorem complex_product_example : z₁ * z₂ = -18 - 17 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_example_l1685_168526


namespace NUMINAMATH_CALUDE_treble_double_plus_five_l1685_168567

theorem treble_double_plus_five (initial_number : ℕ) : initial_number = 15 → 
  3 * (2 * initial_number + 5) = 105 := by
  sorry

end NUMINAMATH_CALUDE_treble_double_plus_five_l1685_168567


namespace NUMINAMATH_CALUDE_round_trip_distance_l1685_168599

/-- Calculates the total distance of a round trip given the times for each leg and the average speed -/
theorem round_trip_distance (t1 t2 : ℚ) (avg_speed : ℚ) (h1 : t1 = 15/60) (h2 : t2 = 25/60) (h3 : avg_speed = 3) :
  (t1 + t2) * avg_speed = 2 := by
  sorry

#check round_trip_distance

end NUMINAMATH_CALUDE_round_trip_distance_l1685_168599


namespace NUMINAMATH_CALUDE_easter_egg_hunt_l1685_168530

/-- The number of eggs found by Cheryl exceeds the combined total of eggs found by Kevin, Bonnie, and George by 29. -/
theorem easter_egg_hunt (kevin bonnie george cheryl : ℕ) 
  (h1 : kevin = 5) 
  (h2 : bonnie = 13) 
  (h3 : george = 9) 
  (h4 : cheryl = 56) : 
  cheryl - (kevin + bonnie + george) = 29 := by
  sorry

end NUMINAMATH_CALUDE_easter_egg_hunt_l1685_168530


namespace NUMINAMATH_CALUDE_willey_farm_land_allocation_l1685_168535

/-- The Willey Farm Collective land allocation problem -/
theorem willey_farm_land_allocation :
  let corn_cost : ℝ := 42
  let wheat_cost : ℝ := 35
  let total_capital : ℝ := 165200
  let wheat_acres : ℝ := 3400
  let corn_acres : ℝ := (total_capital - wheat_cost * wheat_acres) / corn_cost
  corn_acres + wheat_acres = 4500 := by
  sorry

end NUMINAMATH_CALUDE_willey_farm_land_allocation_l1685_168535


namespace NUMINAMATH_CALUDE_S_min_value_l1685_168509

/-- The function S defined on real numbers x and y -/
def S (x y : ℝ) : ℝ := 2 * x^2 - x*y + y^2 + 2*x + 3*y

/-- Theorem stating that S has a minimum value of -4 -/
theorem S_min_value :
  (∀ x y : ℝ, S x y ≥ -4) ∧ (∃ x y : ℝ, S x y = -4) :=
sorry

end NUMINAMATH_CALUDE_S_min_value_l1685_168509


namespace NUMINAMATH_CALUDE_profit_share_calculation_l1685_168598

theorem profit_share_calculation (investment_A investment_B investment_C : ℕ)
  (profit_difference_AC : ℕ) (profit_share_B : ℕ) :
  investment_A = 6000 →
  investment_B = 8000 →
  investment_C = 10000 →
  profit_difference_AC = 500 →
  profit_share_B = 1000 :=
by sorry

end NUMINAMATH_CALUDE_profit_share_calculation_l1685_168598


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1685_168566

/-- Proves that if an article is sold at 800 with a profit of 25%, then its cost price is 640. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 800)
  (h2 : profit_percentage = 25) :
  let cost_price := selling_price / (1 + profit_percentage / 100)
  cost_price = 640 := by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1685_168566


namespace NUMINAMATH_CALUDE_triangle_angle_c_l1685_168583

theorem triangle_angle_c (A B C : ℝ) (m n : ℝ × ℝ) : 
  0 < C ∧ C < π →
  A + B + C = π →
  m = (Real.sqrt 3 * Real.sin A, Real.sin B) →
  n = (Real.cos B, Real.sqrt 3 * Real.cos A) →
  m.1 * n.1 + m.2 * n.2 = 1 + Real.cos (A + B) →
  C = 2 * π / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l1685_168583


namespace NUMINAMATH_CALUDE_cameron_house_paintable_area_l1685_168520

/-- Calculates the total paintable area of walls in multiple bedrooms -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - unpaintable_area
  num_bedrooms * paintable_area

/-- Theorem stating that the total paintable area of walls in Cameron's house is 1840 square feet -/
theorem cameron_house_paintable_area :
  total_paintable_area 4 15 12 10 80 = 1840 := by
  sorry

#eval total_paintable_area 4 15 12 10 80

end NUMINAMATH_CALUDE_cameron_house_paintable_area_l1685_168520


namespace NUMINAMATH_CALUDE_each_girl_receives_two_dollars_l1685_168534

def debt : ℕ := 40

def lulu_savings : ℕ := 6

def nora_savings : ℕ := 5 * lulu_savings

def tamara_savings : ℕ := nora_savings / 3

def total_savings : ℕ := tamara_savings + nora_savings + lulu_savings

def remaining_money : ℕ := total_savings - debt

theorem each_girl_receives_two_dollars : 
  remaining_money / 3 = 2 := by sorry

end NUMINAMATH_CALUDE_each_girl_receives_two_dollars_l1685_168534


namespace NUMINAMATH_CALUDE_total_prom_cost_is_correct_l1685_168515

/-- Calculates the total cost of prom services for Keesha -/
def total_prom_cost : ℝ :=
  let updo_cost : ℝ := 50
  let updo_discount : ℝ := 0.1
  let manicure_cost : ℝ := 30
  let pedicure_cost : ℝ := 35
  let pedicure_discount : ℝ := 0.5
  let makeup_cost : ℝ := 40
  let makeup_tax : ℝ := 0.07
  let facial_cost : ℝ := 60
  let facial_discount : ℝ := 0.15
  let tip_rate : ℝ := 0.2

  let hair_total : ℝ := (updo_cost * (1 - updo_discount)) * (1 + tip_rate)
  let nails_total : ℝ := (manicure_cost + pedicure_cost * pedicure_discount) * (1 + tip_rate)
  let makeup_total : ℝ := (makeup_cost * (1 + makeup_tax)) * (1 + tip_rate)
  let facial_total : ℝ := (facial_cost * (1 - facial_discount)) * (1 + tip_rate)

  hair_total + nails_total + makeup_total + facial_total

/-- Theorem stating that the total cost of prom services for Keesha is $223.56 -/
theorem total_prom_cost_is_correct : total_prom_cost = 223.56 := by
  sorry

end NUMINAMATH_CALUDE_total_prom_cost_is_correct_l1685_168515
