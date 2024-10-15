import Mathlib

namespace NUMINAMATH_CALUDE_range_of_m_l3228_322880

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, ¬(Real.sin x + Real.cos x > m)) ∧ 
  (∀ x : ℝ, x^2 + m*x + 1 > 0) ↔ 
  -Real.sqrt 2 ≤ m ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3228_322880


namespace NUMINAMATH_CALUDE_divisor_is_one_l3228_322871

theorem divisor_is_one (x d : ℕ) (k n : ℤ) : 
  x % d = 5 →
  (x + 17) % 41 = 22 →
  x = k * d + 5 →
  x = 41 * n + 5 →
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_divisor_is_one_l3228_322871


namespace NUMINAMATH_CALUDE_favorite_number_is_25_l3228_322833

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def digit_diff (n : ℕ) : ℕ := Int.natAbs ((n / 10) - (n % 10))

def has_unique_digit (n : ℕ) : Prop :=
  ∀ m : ℕ, is_two_digit m → is_perfect_square m → m ≠ n →
    (n / 10 ≠ m / 10 ∧ n / 10 ≠ m % 10) ∨ (n % 10 ≠ m / 10 ∧ n % 10 ≠ m % 10)

def non_unique_sum (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ n ∧ is_two_digit m ∧ digit_sum m = digit_sum n

def non_unique_diff (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ n ∧ is_two_digit m ∧ digit_diff m = digit_diff n

theorem favorite_number_is_25 :
  ∃! n : ℕ, is_two_digit n ∧ is_perfect_square n ∧ has_unique_digit n ∧
    non_unique_sum n ∧ non_unique_diff n ∧ n = 25 :=
by sorry

end NUMINAMATH_CALUDE_favorite_number_is_25_l3228_322833


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3228_322867

theorem inequality_equivalence (x : ℝ) : 
  (2*x + 3)/(3*x + 5) > (4*x + 1)/(x + 4) ↔ -4 < x ∧ x < -5/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3228_322867


namespace NUMINAMATH_CALUDE_correspondence_C_is_mapping_l3228_322892

def is_mapping (A B : Type) (f : A → B) : Prop :=
  ∀ x : A, ∃! y : B, f x = y

theorem correspondence_C_is_mapping :
  let A := Nat
  let B := { x : Int // x = -1 ∨ x = 0 ∨ x = 1 }
  let f : A → B := λ x => ⟨(-1)^x, by sorry⟩
  is_mapping A B f := by sorry

end NUMINAMATH_CALUDE_correspondence_C_is_mapping_l3228_322892


namespace NUMINAMATH_CALUDE_sector_central_angle_l3228_322815

theorem sector_central_angle (r : ℝ) (p : ℝ) (h1 : r = 10) (h2 : p = 45) :
  let θ := (p - 2 * r) / r
  θ = 2.5 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3228_322815


namespace NUMINAMATH_CALUDE_victory_circle_count_l3228_322848

/-- Represents the different types of medals -/
inductive Medal
  | Gold
  | Silver
  | Bronze
  | Titanium
  | Copper

/-- Represents a runner in the race -/
structure Runner :=
  (position : Nat)
  (medal : Option Medal)

/-- Represents a victory circle configuration -/
def VictoryCircle := List Runner

/-- The number of runners in the race -/
def num_runners : Nat := 8

/-- The maximum number of medals that can be awarded -/
def max_medals : Nat := 5

/-- The minimum number of medals that can be awarded -/
def min_medals : Nat := 3

/-- Generates all possible victory circles for the given scenarios -/
def generate_victory_circles : List VictoryCircle := sorry

/-- Counts the number of unique victory circles -/
def count_victory_circles (circles : List VictoryCircle) : Nat := sorry

/-- Main theorem: The number of different victory circles is 28 -/
theorem victory_circle_count :
  count_victory_circles generate_victory_circles = 28 := by sorry

end NUMINAMATH_CALUDE_victory_circle_count_l3228_322848


namespace NUMINAMATH_CALUDE_rootsOfTwo_is_well_defined_set_rootsOfTwo_has_two_elements_l3228_322826

-- Define the set of real number roots of x^2 = 2
def rootsOfTwo : Set ℝ := {x : ℝ | x^2 = 2}

-- Theorem stating that rootsOfTwo is a well-defined set
theorem rootsOfTwo_is_well_defined_set : 
  ∃ (S : Set ℝ), S = rootsOfTwo ∧ (∀ x : ℝ, x ∈ S ↔ x^2 = 2) :=
by
  sorry

-- Theorem stating that rootsOfTwo contains exactly two elements
theorem rootsOfTwo_has_two_elements :
  ∃ (a b : ℝ), a ≠ b ∧ rootsOfTwo = {a, b} :=
by
  sorry

end NUMINAMATH_CALUDE_rootsOfTwo_is_well_defined_set_rootsOfTwo_has_two_elements_l3228_322826


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3228_322863

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if there exists a point C(0, √(2b)) such that the perpendicular bisector of AC
    (where A is the left vertex) passes through B (the right vertex),
    then the eccentricity of the hyperbola is √10/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ x^2 / a^2 - y^2 / b^2
  let A : ℝ × ℝ := (-a, 0)
  let B : ℝ × ℝ := (a, 0)
  let C : ℝ × ℝ := (0, Real.sqrt (2 * b^2))
  f B = 1 ∧ f A = 1 ∧ f C = -1 ∧
  (∃ M : ℝ × ℝ, M.1 = (A.1 + C.1) / 2 ∧ M.2 = (A.2 + C.2) / 2 ∧
    (B.2 - M.2) * (C.1 - A.1) = (B.1 - M.1) * (C.2 - A.2)) →
  Real.sqrt (a^2 + b^2) / a = Real.sqrt 10 / 2 := by
sorry


end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3228_322863


namespace NUMINAMATH_CALUDE_composite_sum_of_fourth_power_and_64_power_l3228_322869

theorem composite_sum_of_fourth_power_and_64_power (n : ℕ) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 64^n = a * b :=
sorry

end NUMINAMATH_CALUDE_composite_sum_of_fourth_power_and_64_power_l3228_322869


namespace NUMINAMATH_CALUDE_rhinestones_needed_proof_l3228_322847

/-- Given a total number of rhinestones needed, calculate the number still needed
    after buying one-third and finding one-fifth of the total. -/
def rhinestones_still_needed (total : ℕ) : ℕ :=
  total - (total / 3) - (total / 5)

/-- Theorem stating that for 45 rhinestones, the number still needed is 21. -/
theorem rhinestones_needed_proof :
  rhinestones_still_needed 45 = 21 := by
  sorry

#eval rhinestones_still_needed 45

end NUMINAMATH_CALUDE_rhinestones_needed_proof_l3228_322847


namespace NUMINAMATH_CALUDE_binomial_expansion_terms_l3228_322825

theorem binomial_expansion_terms (x a : ℝ) (n : ℕ) : 
  (Nat.choose n 1 * x^(n-1) * a = 56) →
  (Nat.choose n 2 * x^(n-2) * a^2 = 168) →
  (Nat.choose n 3 * x^(n-3) * a^3 = 336) →
  n = 5 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_terms_l3228_322825


namespace NUMINAMATH_CALUDE_triangle_side_not_eight_l3228_322811

/-- A triangle with side lengths a, b, and c exists if and only if the sum of any two sides is greater than the third side for all combinations. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: In a triangle with side lengths 3, 5, and x, x cannot be 8. -/
theorem triangle_side_not_eight :
  ¬ (triangle_inequality 3 5 8) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_not_eight_l3228_322811


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_sum_l3228_322896

theorem geometric_sequence_common_ratio_sum 
  (k : ℝ) (p r : ℝ) (h_distinct : p ≠ r) (h_nonzero : k ≠ 0) 
  (h_equation : k * p^2 - k * r^2 = 3 * (k * p - k * r)) : 
  p + r = 3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_sum_l3228_322896


namespace NUMINAMATH_CALUDE_root_sum_squares_reciprocal_l3228_322872

theorem root_sum_squares_reciprocal (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 49/36 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_reciprocal_l3228_322872


namespace NUMINAMATH_CALUDE_path_order_paths_through_A_paths_through_B_total_paths_correct_l3228_322844

-- Define the grid points
inductive GridPoint
| X | Y | A | B | C | D | E | F | G

-- Define a function to count paths through a point
def pathsThrough (p : GridPoint) : ℕ := sorry

-- Total number of paths from X to Y
def totalPaths : ℕ := 924

-- Theorem stating the order of points based on number of paths
theorem path_order :
  pathsThrough GridPoint.A > pathsThrough GridPoint.F ∧
  pathsThrough GridPoint.F > pathsThrough GridPoint.C ∧
  pathsThrough GridPoint.C > pathsThrough GridPoint.G ∧
  pathsThrough GridPoint.G > pathsThrough GridPoint.E ∧
  pathsThrough GridPoint.E > pathsThrough GridPoint.D ∧
  pathsThrough GridPoint.D > pathsThrough GridPoint.B :=
by sorry

-- Theorem stating that the sum of paths through A and the point below X equals totalPaths
theorem paths_through_A :
  pathsThrough GridPoint.A = totalPaths / 2 :=
by sorry

-- Theorem stating that there's only one path through B
theorem paths_through_B :
  pathsThrough GridPoint.B = 1 :=
by sorry

-- Theorem stating that the total number of paths is correct
theorem total_paths_correct :
  (pathsThrough GridPoint.A) * 2 = totalPaths :=
by sorry

end NUMINAMATH_CALUDE_path_order_paths_through_A_paths_through_B_total_paths_correct_l3228_322844


namespace NUMINAMATH_CALUDE_average_decrease_rate_proof_optimal_price_reduction_proof_l3228_322875

-- Define the initial price, final price, and years of decrease
def initial_price : ℝ := 200
def final_price : ℝ := 162
def years_of_decrease : ℕ := 2

-- Define the daily sales and profit parameters
def initial_daily_sales : ℕ := 20
def price_reduction_step : ℝ := 5
def sales_increase_per_step : ℕ := 10
def daily_profit : ℝ := 1150

-- Define the average yearly decrease rate
def average_decrease_rate : ℝ := 0.1

-- Define the optimal price reduction
def optimal_price_reduction : ℝ := 15

-- Theorem for the average yearly decrease rate
theorem average_decrease_rate_proof :
  initial_price * (1 - average_decrease_rate) ^ years_of_decrease = final_price :=
sorry

-- Theorem for the optimal price reduction
theorem optimal_price_reduction_proof :
  let new_price := initial_price - optimal_price_reduction
  let new_sales := initial_daily_sales + (optimal_price_reduction / price_reduction_step) * sales_increase_per_step
  (new_price - final_price) * new_sales = daily_profit :=
sorry

end NUMINAMATH_CALUDE_average_decrease_rate_proof_optimal_price_reduction_proof_l3228_322875


namespace NUMINAMATH_CALUDE_cecilia_B_count_l3228_322877

/-- The number of students who received a 'B' in Mrs. Cecilia's class -/
def students_with_B_cecilia (jacob_total : ℕ) (jacob_B : ℕ) (cecilia_total : ℕ) (cecilia_absent : ℕ) : ℕ :=
  let jacob_proportion : ℚ := jacob_B / jacob_total
  let cecilia_present : ℕ := cecilia_total - cecilia_absent
  ⌊(jacob_proportion * cecilia_present : ℚ)⌋₊

theorem cecilia_B_count :
  students_with_B_cecilia 20 12 30 6 = 14 :=
by sorry

end NUMINAMATH_CALUDE_cecilia_B_count_l3228_322877


namespace NUMINAMATH_CALUDE_stratified_sample_sophomores_l3228_322806

/-- Represents the number of sophomores in a stratified sample -/
def sophomores_in_sample (total_students : ℕ) (total_sophomores : ℕ) (sample_size : ℕ) : ℕ :=
  (sample_size * total_sophomores) / total_students

/-- Theorem: In a school with 1500 students, of which 600 are sophomores,
    a stratified sample of 100 students should include 40 sophomores -/
theorem stratified_sample_sophomores :
  sophomores_in_sample 1500 600 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_sophomores_l3228_322806


namespace NUMINAMATH_CALUDE_function_inequality_l3228_322885

-- Define a differentiable function f
variable (f : ℝ → ℝ)

-- Assume f is differentiable
variable (hf : Differentiable ℝ f)

-- Assume f'(x) < f(x) for all x in ℝ
variable (h : ∀ x : ℝ, deriv f x < f x)

-- Theorem statement
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x : ℝ, deriv f x < f x) : 
  f 1 < Real.exp 1 * f 0 ∧ f 2014 < Real.exp 2014 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3228_322885


namespace NUMINAMATH_CALUDE_cubic_roots_determinant_l3228_322834

theorem cubic_roots_determinant (p q r : ℝ) (a b c : ℝ) : 
  (a^3 - p*a^2 + q*a - r = 0) →
  (b^3 - p*b^2 + q*b - r = 0) →
  (c^3 - p*c^2 + q*c - r = 0) →
  let matrix : Matrix (Fin 3) (Fin 3) ℝ := !![a, 1, 1; 1, b, 1; 1, 1, c]
  Matrix.det matrix = r - p + 2 := by sorry

end NUMINAMATH_CALUDE_cubic_roots_determinant_l3228_322834


namespace NUMINAMATH_CALUDE_anas_dresses_l3228_322899

theorem anas_dresses (ana lisa : ℕ) : 
  lisa = ana + 18 → 
  ana + lisa = 48 → 
  ana = 15 := by
sorry

end NUMINAMATH_CALUDE_anas_dresses_l3228_322899


namespace NUMINAMATH_CALUDE_gray_opposite_black_l3228_322816

-- Define the colors
inductive Color
| A -- Aqua
| B -- Black
| C -- Crimson
| D -- Dark Blue
| E -- Emerald
| F -- Fuchsia
| G -- Gray
| H -- Hazel

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  faces : List Face
  adjacent : Color → Color → Prop
  opposite : Color → Color → Prop

-- Define the problem conditions
axiom cube_has_eight_faces : ∀ (c : Cube), c.faces.length = 8

axiom aqua_adjacent_to_dark_blue_and_emerald : 
  ∀ (c : Cube), c.adjacent Color.A Color.D ∧ c.adjacent Color.A Color.E

-- The theorem to prove
theorem gray_opposite_black (c : Cube) : c.opposite Color.G Color.B := by
  sorry


end NUMINAMATH_CALUDE_gray_opposite_black_l3228_322816


namespace NUMINAMATH_CALUDE_apples_sale_theorem_l3228_322882

/-- Calculate the total money made from selling boxes of apples -/
def total_money_from_apples (total_apples : ℕ) (apples_per_box : ℕ) (price_per_box : ℕ) : ℕ :=
  ((total_apples / apples_per_box) * price_per_box)

/-- Theorem: Given 275 apples, with 20 apples per box sold at 8,000 won each,
    the total money made from selling all full boxes is 104,000 won -/
theorem apples_sale_theorem :
  total_money_from_apples 275 20 8000 = 104000 := by
  sorry

end NUMINAMATH_CALUDE_apples_sale_theorem_l3228_322882


namespace NUMINAMATH_CALUDE_existence_of_multiple_factorizations_l3228_322801

def V_n (n : ℕ) := {m : ℕ | ∃ k : ℕ, k ≥ 1 ∧ m = 1 + k * n}

def irreducible_in_V_n (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ p q : ℕ, p ∈ V_n n → q ∈ V_n n → p * q ≠ m

theorem existence_of_multiple_factorizations (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ V_n n ∧
    ∃ (factors1 factors2 : List ℕ),
      factors1 ≠ factors2 ∧
      (∀ f ∈ factors1, irreducible_in_V_n n f) ∧
      (∀ f ∈ factors2, irreducible_in_V_n n f) ∧
      r = factors1.prod ∧
      r = factors2.prod :=
  sorry

end NUMINAMATH_CALUDE_existence_of_multiple_factorizations_l3228_322801


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3228_322851

/-- A function f: ℝ⁺* → ℝ⁺* satisfying the functional equation f(x) f(y f(x)) = f(x + y) for all x, y > 0 -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f x > 0 → f (y * f x) > 0 → f x * f (y * f x) = f (x + y)

/-- The theorem stating that functions satisfying the given functional equation
    are either of the form f(x) = 1/(1 + ax) for some a > 0, or f(x) = 1 -/
theorem functional_equation_solution (f : ℝ → ℝ) :
  FunctionalEquation f →
  (∃ a : ℝ, a > 0 ∧ ∀ x, x > 0 → f x = 1 / (1 + a * x)) ∨
  (∀ x, x > 0 → f x = 1) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3228_322851


namespace NUMINAMATH_CALUDE_pyramid_solution_l3228_322812

structure NumberPyramid where
  row1_left : ℕ
  row1_right : ℕ
  row2_left : ℕ
  row2_right : ℕ
  row3_left : ℕ
  row3_middle : ℕ
  row3_right : ℕ

def is_valid_pyramid (p : NumberPyramid) : Prop :=
  p.row2_left = p.row1_left + p.row1_right ∧
  p.row2_right = p.row1_right + 660 ∧
  p.row3_left = p.row2_left * p.row1_left ∧
  p.row3_middle = p.row2_left * p.row2_right ∧
  p.row3_right = p.row2_right * 660

theorem pyramid_solution :
  ∃ (p : NumberPyramid), is_valid_pyramid p ∧ 
    p.row3_left = 28 ∧ p.row3_right = 630 ∧ p.row2_left = 13 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_solution_l3228_322812


namespace NUMINAMATH_CALUDE_count_distinct_values_l3228_322874

def is_pythagorean_triple (a b c : ℕ) : Prop := a^2 + b^2 = c^2

def satisfies_conditions (f : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, f n ∣ n^2016) ∧
  (∀ a b c : ℕ, is_pythagorean_triple a b c → f a * f b = f c)

theorem count_distinct_values :
  ∃ (S : Finset ℕ),
    (∀ f : ℕ → ℕ, satisfies_conditions f →
      (f 2014 + f 2 - f 2016) ∈ S) ∧
    S.card = 2^2017 - 1 :=
sorry

end NUMINAMATH_CALUDE_count_distinct_values_l3228_322874


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_gt_one_l3228_322858

theorem solution_set_reciprocal_gt_one (x : ℝ) : 1 / x > 1 ↔ 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_gt_one_l3228_322858


namespace NUMINAMATH_CALUDE_fishing_problem_l3228_322891

/-- The number of fish Xiaohua caught -/
def xiaohua_fish : ℕ := 26

/-- The number of fish Xiaobai caught -/
def xiaobai_fish : ℕ := 4

/-- The condition when Xiaohua gives 2 fish to Xiaobai -/
def condition1 (x y : ℕ) : Prop :=
  y - 2 = 4 * (x + 2)

/-- The condition when Xiaohua gives 6 fish to Xiaobai -/
def condition2 (x y : ℕ) : Prop :=
  y - 6 = 2 * (x + 6)

theorem fishing_problem :
  condition1 xiaobai_fish xiaohua_fish ∧
  condition2 xiaobai_fish xiaohua_fish := by
  sorry


end NUMINAMATH_CALUDE_fishing_problem_l3228_322891


namespace NUMINAMATH_CALUDE_player5_score_breakdown_l3228_322873

/-- Represents the scoring breakdown for a basketball player -/
structure PlayerScore where
  threes : Nat
  twos : Nat
  frees : Nat

/-- Calculates the total points scored by a player -/
def totalPoints (score : PlayerScore) : Nat :=
  3 * score.threes + 2 * score.twos + score.frees

theorem player5_score_breakdown :
  ∀ (team_total : Nat) (other_players_total : Nat),
    team_total = 75 →
    other_players_total = 61 →
    ∃ (score : PlayerScore),
      totalPoints score = team_total - other_players_total ∧
      score.threes ≥ 2 ∧
      score.twos ≥ 1 ∧
      score.frees ≤ 4 ∧
      score.threes = 2 ∧
      score.twos = 2 ∧
      score.frees = 4 :=
by sorry

end NUMINAMATH_CALUDE_player5_score_breakdown_l3228_322873


namespace NUMINAMATH_CALUDE_x_root_of_quadratic_with_integer_coeff_l3228_322894

/-- Given distinct real numbers x and y with equal fractional parts and equal fractional parts of their cubes,
    x is a root of a quadratic equation with integer coefficients. -/
theorem x_root_of_quadratic_with_integer_coeff
  (x y : ℝ)
  (h_distinct : x ≠ y)
  (h_frac_eq : x - ⌊x⌋ = y - ⌊y⌋)
  (h_frac_cube_eq : x^3 - ⌊x^3⌋ = y^3 - ⌊y^3⌋) :
  ∃ (a b c : ℤ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧ (a * x^2 + b * x + c : ℝ) = 0 :=
sorry

end NUMINAMATH_CALUDE_x_root_of_quadratic_with_integer_coeff_l3228_322894


namespace NUMINAMATH_CALUDE_y_equal_at_one_y_diff_five_at_two_l3228_322862

-- Define the functions y₁ and y₂
def y₁ (x : ℝ) : ℝ := -2 * x + 3
def y₂ (x : ℝ) : ℝ := 3 * x - 2

-- Theorem 1: y₁ = y₂ when x = 1
theorem y_equal_at_one : y₁ 1 = y₂ 1 := by sorry

-- Theorem 2: y₁ + 5 = y₂ when x = 2
theorem y_diff_five_at_two : y₁ 2 + 5 = y₂ 2 := by sorry

end NUMINAMATH_CALUDE_y_equal_at_one_y_diff_five_at_two_l3228_322862


namespace NUMINAMATH_CALUDE_largest_number_l3228_322855

theorem largest_number : 
  let numbers : List ℝ := [0.9791, 0.97019, 0.97909, 0.971, 0.97109]
  ∀ x ∈ numbers, x ≤ 0.9791 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3228_322855


namespace NUMINAMATH_CALUDE_rectangle_problem_l3228_322820

theorem rectangle_problem (x : ℝ) :
  (∃ a b : ℝ, 
    a > 0 ∧ b > 0 ∧
    a = 2 * b ∧
    2 * (a + b) = x ∧
    a * b = x) →
  x = 18 := by
sorry

end NUMINAMATH_CALUDE_rectangle_problem_l3228_322820


namespace NUMINAMATH_CALUDE_distance_to_origin_l3228_322876

/-- The distance from the point (3, -4) to the origin (0, 0) in the Cartesian coordinate system is 5. -/
theorem distance_to_origin : Real.sqrt (3^2 + (-4)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l3228_322876


namespace NUMINAMATH_CALUDE_students_present_l3228_322888

theorem students_present (total : ℕ) (absent_percent : ℚ) (present : ℕ) : 
  total = 50 → 
  absent_percent = 12 / 100 → 
  present = total - (total * (absent_percent : ℚ)).floor → 
  present = 44 := by
sorry

end NUMINAMATH_CALUDE_students_present_l3228_322888


namespace NUMINAMATH_CALUDE_prob_non_defective_pencils_l3228_322824

/-- The probability of selecting 5 non-defective pencils from a box of 12 pencils
    where 4 are defective is 7/99. -/
theorem prob_non_defective_pencils :
  let total_pencils : ℕ := 12
  let defective_pencils : ℕ := 4
  let selected_pencils : ℕ := 5
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  (Nat.choose non_defective_pencils selected_pencils : ℚ) /
  (Nat.choose total_pencils selected_pencils : ℚ) = 7 / 99 := by
sorry

end NUMINAMATH_CALUDE_prob_non_defective_pencils_l3228_322824


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_l3228_322817

theorem algebraic_expression_simplification (a : ℝ) :
  a = 2 * Real.sin (60 * π / 180) + 3 →
  (a + 1) / (a - 3) - (a - 3) / (a + 2) / ((a^2 - 6*a + 9) / (a^2 - 4)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_l3228_322817


namespace NUMINAMATH_CALUDE_initial_investment_rate_is_five_percent_l3228_322868

/-- Proves that given specific investment conditions, the initial investment rate is 5% --/
theorem initial_investment_rate_is_five_percent
  (initial_investment : ℝ)
  (additional_investment : ℝ)
  (additional_rate : ℝ)
  (total_income_rate : ℝ)
  (h1 : initial_investment = 2800)
  (h2 : additional_investment = 1400)
  (h3 : additional_rate = 0.08)
  (h4 : total_income_rate = 0.06)
  (h5 : initial_investment * x + additional_investment * additional_rate = 
        (initial_investment + additional_investment) * total_income_rate) :
  x = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_initial_investment_rate_is_five_percent_l3228_322868


namespace NUMINAMATH_CALUDE_certain_number_problem_l3228_322839

theorem certain_number_problem (x : ℝ) : 
  (0.20 * x) - (1/3) * (0.20 * x) = 24 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3228_322839


namespace NUMINAMATH_CALUDE_integral_2x_minus_1_l3228_322822

theorem integral_2x_minus_1 : ∫ x in (0 : ℝ)..3, (2*x - 1) = 6 := by sorry

end NUMINAMATH_CALUDE_integral_2x_minus_1_l3228_322822


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3228_322883

/-- An arithmetic sequence {a_n} with the given properties -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 1/3)
  (h3 : a 2 + a 5 = 4)
  (h4 : ∃ n : ℕ, a n = 27) :
  ∃ n : ℕ, n = 9 ∧ a n = 27 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3228_322883


namespace NUMINAMATH_CALUDE_min_teachers_is_six_l3228_322881

/-- Represents the number of subjects for each discipline -/
structure SubjectCounts where
  maths : Nat
  physics : Nat
  chemistry : Nat

/-- Represents the constraints of the teaching system -/
structure TeachingSystem where
  subjects : SubjectCounts
  max_subjects_per_teacher : Nat
  specialized : Bool

/-- Calculates the minimum number of teachers required -/
def min_teachers_required (system : TeachingSystem) : Nat :=
  if system.specialized then
    let maths_teachers := (system.subjects.maths + system.max_subjects_per_teacher - 1) / system.max_subjects_per_teacher
    let physics_teachers := (system.subjects.physics + system.max_subjects_per_teacher - 1) / system.max_subjects_per_teacher
    let chemistry_teachers := (system.subjects.chemistry + system.max_subjects_per_teacher - 1) / system.max_subjects_per_teacher
    maths_teachers + physics_teachers + chemistry_teachers
  else
    let total_subjects := system.subjects.maths + system.subjects.physics + system.subjects.chemistry
    (total_subjects + system.max_subjects_per_teacher - 1) / system.max_subjects_per_teacher

/-- The main theorem stating that the minimum number of teachers required is 6 -/
theorem min_teachers_is_six (system : TeachingSystem) 
  (h1 : system.subjects = { maths := 6, physics := 5, chemistry := 5 })
  (h2 : system.max_subjects_per_teacher = 4)
  (h3 : system.specialized = true) : 
  min_teachers_required system = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_teachers_is_six_l3228_322881


namespace NUMINAMATH_CALUDE_absent_children_l3228_322832

/-- Proves that the number of absent children is 70 given the conditions of the problem -/
theorem absent_children (total_children : ℕ) (sweets_per_child : ℕ) (extra_sweets : ℕ) : 
  total_children = 190 →
  sweets_per_child = 38 →
  extra_sweets = 14 →
  (total_children - (total_children - sweets_per_child * total_children / (sweets_per_child - extra_sweets))) = 70 := by
  sorry

end NUMINAMATH_CALUDE_absent_children_l3228_322832


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3228_322849

theorem rectangular_to_polar_conversion :
  let x : ℝ := 8
  let y : ℝ := 2 * Real.sqrt 2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  (r = 6 * Real.sqrt 2) ∧ 
  (θ = Real.arctan (Real.sqrt 2 / 4)) ∧
  (r > 0) ∧ 
  (0 ≤ θ) ∧ 
  (θ < 2 * Real.pi) := by
sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3228_322849


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3228_322879

/-- A line that is tangent to a circle and intersects a parabola -/
structure TangentLine where
  -- The line equation: ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- The circle equation: x^4 + y^2 = 8
  circle : (x y : ℝ) → x^4 + y^2 = 8
  -- The parabola equation: y^2 = 4x
  parabola : (x y : ℝ) → y^2 = 4*x
  -- The line is tangent to the circle
  is_tangent : ∃ (x y : ℝ), a*x + b*y + c = 0 ∧ x^4 + y^2 = 8
  -- The line intersects the parabola at two points
  intersects_parabola : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    a*x₁ + b*y₁ + c = 0 ∧ y₁^2 = 4*x₁ ∧
    a*x₂ + b*y₂ + c = 0 ∧ y₂^2 = 4*x₂
  -- The circle passes through the origin
  origin_on_circle : 0^4 + 0^2 = 8

/-- The theorem stating the equation of the tangent line -/
theorem tangent_line_equation (l : TangentLine) : 
  (l.a = 1 ∧ l.b = -1 ∧ l.c = -4) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -4) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3228_322879


namespace NUMINAMATH_CALUDE_albert_oranges_l3228_322837

/-- The number of boxes Albert has -/
def num_boxes : ℕ := 7

/-- The number of oranges in each box -/
def oranges_per_box : ℕ := 5

/-- The total number of oranges Albert has -/
def total_oranges : ℕ := num_boxes * oranges_per_box

theorem albert_oranges : total_oranges = 35 := by
  sorry

end NUMINAMATH_CALUDE_albert_oranges_l3228_322837


namespace NUMINAMATH_CALUDE_correct_average_l3228_322821

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ 
  incorrect_avg = 46 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 65 →
  (n : ℚ) * incorrect_avg - incorrect_num + correct_num = n * 50 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l3228_322821


namespace NUMINAMATH_CALUDE_f_max_min_values_f_max_min_m_neg_f_max_min_m_0_to_4_f_max_min_m_gt_4_l3228_322852

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + m - 1

-- Define the domain
def domain : Set ℝ := Set.Icc 0 4

-- Theorem for the maximum and minimum values
theorem f_max_min_values (m : ℝ) :
  (∀ x ∈ domain, f m x ≥ (m - 1) ∧ f m x ≤ (15 - 7*m)) ∨
  ((∀ x ∈ domain, f m x ≥ (-m^2 + m - 1)) ∧
   ((0 ≤ m ∧ m ≤ 2 → ∀ x ∈ domain, f m x ≤ (15 - 7*m)) ∧
    (2 ≤ m ∧ m ≤ 4 → ∀ x ∈ domain, f m x ≤ (m - 1)))) ∨
  (∀ x ∈ domain, f m x ≥ (15 - 7*m) ∧ f m x ≤ (m - 1)) :=
by sorry

-- Helper theorems for each case
theorem f_max_min_m_neg (m : ℝ) (hm : m < 0) :
  ∀ x ∈ domain, f m x ≥ (m - 1) ∧ f m x ≤ (15 - 7*m) :=
by sorry

theorem f_max_min_m_0_to_4 (m : ℝ) (hm : 0 ≤ m ∧ m ≤ 4) :
  (∀ x ∈ domain, f m x ≥ (-m^2 + m - 1)) ∧
  ((0 ≤ m ∧ m ≤ 2 → ∀ x ∈ domain, f m x ≤ (15 - 7*m)) ∧
   (2 ≤ m ∧ m ≤ 4 → ∀ x ∈ domain, f m x ≤ (m - 1))) :=
by sorry

theorem f_max_min_m_gt_4 (m : ℝ) (hm : m > 4) :
  ∀ x ∈ domain, f m x ≥ (15 - 7*m) ∧ f m x ≤ (m - 1) :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_values_f_max_min_m_neg_f_max_min_m_0_to_4_f_max_min_m_gt_4_l3228_322852


namespace NUMINAMATH_CALUDE_infinitely_many_non_square_plus_prime_numbers_l3228_322800

theorem infinitely_many_non_square_plus_prime_numbers :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ n ∈ S, ¬∃ (m : ℤ) (p : ℕ), Nat.Prime p ∧ n = m^2 + p := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_non_square_plus_prime_numbers_l3228_322800


namespace NUMINAMATH_CALUDE_binomial_20_4_l3228_322841

theorem binomial_20_4 : Nat.choose 20 4 = 4845 := by sorry

end NUMINAMATH_CALUDE_binomial_20_4_l3228_322841


namespace NUMINAMATH_CALUDE_dressing_ratio_l3228_322813

def ranch_cases : ℕ := 28
def caesar_cases : ℕ := 4

theorem dressing_ratio : 
  (ranch_cases / caesar_cases : ℚ) = 7 / 1 := by
  sorry

end NUMINAMATH_CALUDE_dressing_ratio_l3228_322813


namespace NUMINAMATH_CALUDE_negative_ten_meters_westward_l3228_322805

-- Define the direction as an enumeration
inductive Direction
  | East
  | West

-- Define a function to convert a signed distance to a direction and magnitude
def interpretDistance (d : ℤ) : Direction × ℕ :=
  if d ≥ 0 then (Direction.East, d.natAbs) else (Direction.West, d.natAbs)

-- State the theorem
theorem negative_ten_meters_westward :
  interpretDistance (-10) = (Direction.West, 10) := by
  sorry

end NUMINAMATH_CALUDE_negative_ten_meters_westward_l3228_322805


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3228_322866

theorem arithmetic_mean_of_special_set : 
  let S : Finset ℕ := Finset.range 9
  let special_number (n : ℕ) : ℕ := n * ((10^n - 1) / 9)
  let sum_of_set : ℕ := S.sum special_number
  sum_of_set / 9 = 123456790 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3228_322866


namespace NUMINAMATH_CALUDE_part_I_part_II_l3228_322886

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

-- Define the complement of A relative to ℝ
def C_R_A : Set ℝ := {x | x ≤ 3 ∨ x ≥ 7}

-- Theorem for part (I)
theorem part_I : (C_R_A ∩ B) = {x | (2 < x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

-- Theorem for part (II)
theorem part_II (a : ℝ) : C a ⊆ (A ∪ B) → a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l3228_322886


namespace NUMINAMATH_CALUDE_min_value_expression_l3228_322831

theorem min_value_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + y = 2) :
  ((x + 1)^2 + 3) / (x + 2) + y^2 / (y + 1) ≥ 14/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3228_322831


namespace NUMINAMATH_CALUDE_three_lines_intersection_l3228_322838

/-- Three distinct lines in 2D space -/
structure ThreeLines where
  a : ℝ
  b : ℝ
  l₁ : ℝ → ℝ → ℝ := λ x y => a * x + 2 * b * y + 3 * (a + b + 1)
  l₂ : ℝ → ℝ → ℝ := λ x y => b * x + 2 * (a + b + 1) * y + 3 * a
  l₃ : ℝ → ℝ → ℝ := λ x y => (a + b + 1) * x + 2 * a * y + 3 * b
  distinct : l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₃ ≠ l₁

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point lying on a line -/
def PointOnLine (p : Point) (l : ℝ → ℝ → ℝ) : Prop :=
  l p.x p.y = 0

/-- Definition of three lines intersecting at a single point -/
def IntersectAtSinglePoint (lines : ThreeLines) : Prop :=
  ∃! p : Point, PointOnLine p lines.l₁ ∧ PointOnLine p lines.l₂ ∧ PointOnLine p lines.l₃

/-- Theorem statement -/
theorem three_lines_intersection (lines : ThreeLines) :
  IntersectAtSinglePoint lines ↔ lines.a + lines.b = -1/2 := by sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l3228_322838


namespace NUMINAMATH_CALUDE_lucas_running_speed_l3228_322819

theorem lucas_running_speed :
  let eugene_speed : ℚ := 5
  let brianna_speed : ℚ := (3 / 4) * eugene_speed
  let katie_speed : ℚ := (4 / 3) * brianna_speed
  let lucas_speed : ℚ := (5 / 6) * katie_speed
  lucas_speed = 25 / 6 := by sorry

end NUMINAMATH_CALUDE_lucas_running_speed_l3228_322819


namespace NUMINAMATH_CALUDE_inequality_always_true_l3228_322884

theorem inequality_always_true : ∀ x : ℝ, 3 * x - 5 ≤ 12 - 2 * x + x^2 := by sorry

end NUMINAMATH_CALUDE_inequality_always_true_l3228_322884


namespace NUMINAMATH_CALUDE_two_red_cards_probability_l3228_322860

/-- The number of cards in the deck -/
def total_cards : ℕ := 65

/-- The number of red cards in the deck -/
def red_cards : ℕ := 39

/-- The number of ways to choose 2 cards from the deck -/
def total_combinations : ℕ := total_cards.choose 2

/-- The number of ways to choose 2 red cards from the red cards -/
def red_combinations : ℕ := red_cards.choose 2

/-- The probability of drawing two red cards in the first two draws -/
def probability : ℚ := red_combinations / total_combinations

theorem two_red_cards_probability :
  probability = 741 / 2080 := by sorry

end NUMINAMATH_CALUDE_two_red_cards_probability_l3228_322860


namespace NUMINAMATH_CALUDE_quadruple_solution_l3228_322808

theorem quadruple_solution :
  ∀ (a b c d : ℝ), 
    a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
    a^2 * b = c →
    b * c^2 = a →
    c * a^2 = b →
    a + b + c = d →
    a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadruple_solution_l3228_322808


namespace NUMINAMATH_CALUDE_couponA_provides_greatest_discount_l3228_322836

-- Define the coupon discount functions
def couponA (price : Real) : Real := 0.12 * price

def couponB (price : Real) : Real := 25

def couponC (price : Real) : Real := 0.15 * (price - 150)

def couponD (price : Real) : Real := 0.1 * price + 13.5

-- Define the listed price
def listedPrice : Real := 229.95

-- Theorem statement
theorem couponA_provides_greatest_discount :
  couponA listedPrice > couponB listedPrice ∧
  couponA listedPrice > couponC listedPrice ∧
  couponA listedPrice > couponD listedPrice := by
  sorry

end NUMINAMATH_CALUDE_couponA_provides_greatest_discount_l3228_322836


namespace NUMINAMATH_CALUDE_max_books_borrowed_l3228_322850

theorem max_books_borrowed (total_students : Nat) (zero_books : Nat) (one_book : Nat) (two_books : Nat) 
  (avg_books : Nat) (h1 : total_students = 20) (h2 : zero_books = 3) (h3 : one_book = 9) 
  (h4 : two_books = 4) (h5 : avg_books = 2) : ∃ (max_books : Nat), max_books = 14 ∧ 
  max_books = total_students * avg_books - (zero_books * 0 + one_book * 1 + two_books * 2 + 
  (total_students - zero_books - one_book - two_books - 1) * 3) := by
  sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l3228_322850


namespace NUMINAMATH_CALUDE_power_problem_l3228_322897

theorem power_problem (a m n : ℕ) (h1 : a ^ m = 3) (h2 : a ^ n = 2) :
  a ^ (3 * m + 2 * n) = 108 := by
  sorry

end NUMINAMATH_CALUDE_power_problem_l3228_322897


namespace NUMINAMATH_CALUDE_round_trip_speed_calculation_l3228_322854

/-- Proves that given a round trip with total distance 72 miles, total time 7 hours,
    and return speed 18 miles per hour, the outbound speed is 7.2 miles per hour. -/
theorem round_trip_speed_calculation (total_distance : ℝ) (total_time : ℝ) (return_speed : ℝ) :
  total_distance = 72 ∧ total_time = 7 ∧ return_speed = 18 →
  ∃ outbound_speed : ℝ,
    outbound_speed = 7.2 ∧
    total_distance / 2 / outbound_speed + total_distance / 2 / return_speed = total_time :=
by sorry

end NUMINAMATH_CALUDE_round_trip_speed_calculation_l3228_322854


namespace NUMINAMATH_CALUDE_statements_evaluation_l3228_322895

theorem statements_evaluation :
  (∃ a b : ℝ, a > b ∧ ¬(a^2 > b^2)) ∧
  (∀ a b : ℝ, |a| > |b| → a^2 > b^2) ∧
  (∃ a b c : ℝ, a > b ∧ ¬(a*c^2 > b*c^2)) ∧
  (∀ a b : ℝ, a > b ∧ b > 0 → 1/a < 1/b) := by
  sorry


end NUMINAMATH_CALUDE_statements_evaluation_l3228_322895


namespace NUMINAMATH_CALUDE_save_fraction_is_one_seventh_l3228_322889

/-- Represents the worker's financial situation over a year --/
structure WorkerFinances where
  monthly_pay : ℝ
  save_fraction : ℝ
  months : ℕ := 12

/-- The conditions of the problem as described --/
def valid_finances (w : WorkerFinances) : Prop :=
  w.monthly_pay > 0 ∧
  w.save_fraction > 0 ∧
  w.save_fraction < 1 ∧
  w.months * w.save_fraction * w.monthly_pay = 2 * (1 - w.save_fraction) * w.monthly_pay

/-- The main theorem stating that the save fraction is 1/7 --/
theorem save_fraction_is_one_seventh (w : WorkerFinances) 
  (h : valid_finances w) : w.save_fraction = 1 / 7 := by
  sorry

#check save_fraction_is_one_seventh

end NUMINAMATH_CALUDE_save_fraction_is_one_seventh_l3228_322889


namespace NUMINAMATH_CALUDE_shifted_function_eq_l3228_322865

def original_function (x : ℝ) : ℝ := 2 * x

def vertical_shift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  fun x => f x - shift

def shifted_function : ℝ → ℝ := vertical_shift original_function 2

theorem shifted_function_eq : shifted_function = fun x => 2 * x - 2 := by sorry

end NUMINAMATH_CALUDE_shifted_function_eq_l3228_322865


namespace NUMINAMATH_CALUDE_vector_ratio_theorem_l3228_322853

theorem vector_ratio_theorem (a b : ℝ × ℝ) :
  let angle := Real.pi / 3
  let magnitude (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)
  let sum := (a.1 + b.1, a.2 + b.2)
  (∃ (d : ℝ), magnitude b = magnitude a + d ∧ magnitude sum = magnitude a + 2*d) →
  (a.1 * b.1 + a.2 * b.2 = magnitude a * magnitude b * Real.cos angle) →
  ∃ (k : ℝ), k > 0 ∧ magnitude a = 3*k ∧ magnitude b = 5*k ∧ magnitude sum = 7*k := by
sorry

end NUMINAMATH_CALUDE_vector_ratio_theorem_l3228_322853


namespace NUMINAMATH_CALUDE_min_sum_given_reciprocal_sum_l3228_322828

theorem min_sum_given_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 1) + 1 / (y + 1) = 1 / 2) :
  x + y ≥ 6 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 1 / (a + 1) + 1 / (b + 1) = 1 / 2 ∧ a + b = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_given_reciprocal_sum_l3228_322828


namespace NUMINAMATH_CALUDE_price_decrease_sales_increase_l3228_322823

/-- Given a price decrease and revenue increase, calculate the increase in number of items sold -/
theorem price_decrease_sales_increase
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_decrease_percentage : ℝ)
  (revenue_increase_percentage : ℝ)
  (h_price_decrease : price_decrease_percentage = 20)
  (h_revenue_increase : revenue_increase_percentage = 28.000000000000025)
  (h_positive_price : original_price > 0)
  (h_positive_quantity : original_quantity > 0) :
  let new_price := original_price * (1 - price_decrease_percentage / 100)
  let new_quantity := original_quantity * (1 + revenue_increase_percentage / 100) / (1 - price_decrease_percentage / 100)
  let quantity_increase_percentage := (new_quantity / original_quantity - 1) * 100
  ∃ ε > 0, |quantity_increase_percentage - 60| < ε :=
sorry

end NUMINAMATH_CALUDE_price_decrease_sales_increase_l3228_322823


namespace NUMINAMATH_CALUDE_travis_payment_l3228_322810

/-- Calculates the payment for Travis given the specified conditions --/
def calculate_payment (total_bowls : ℕ) (fixed_fee : ℕ) (safe_delivery_fee : ℕ) (penalty : ℕ) (lost_bowls : ℕ) (broken_bowls : ℕ) : ℕ :=
  let damaged_bowls := lost_bowls + broken_bowls
  let safe_bowls := total_bowls - damaged_bowls
  let safe_delivery_payment := safe_bowls * safe_delivery_fee
  let total_payment := safe_delivery_payment + fixed_fee
  let penalty_amount := damaged_bowls * penalty
  total_payment - penalty_amount

/-- Theorem stating that Travis should be paid $1825 given the specified conditions --/
theorem travis_payment :
  calculate_payment 638 100 3 4 12 15 = 1825 := by
  sorry

end NUMINAMATH_CALUDE_travis_payment_l3228_322810


namespace NUMINAMATH_CALUDE_river_joe_pricing_l3228_322803

/-- River Joe's Seafood Diner pricing problem -/
theorem river_joe_pricing
  (total_orders : ℕ)
  (total_revenue : ℚ)
  (catfish_price : ℚ)
  (popcorn_shrimp_orders : ℕ)
  (h1 : total_orders = 26)
  (h2 : total_revenue = 133.5)
  (h3 : catfish_price = 6)
  (h4 : popcorn_shrimp_orders = 9) :
  ∃ (popcorn_shrimp_price : ℚ),
    popcorn_shrimp_price = 3.5 ∧
    total_revenue = (total_orders - popcorn_shrimp_orders) * catfish_price +
                    popcorn_shrimp_orders * popcorn_shrimp_price :=
by sorry

end NUMINAMATH_CALUDE_river_joe_pricing_l3228_322803


namespace NUMINAMATH_CALUDE_parabola_directrix_l3228_322846

/-- Given a parabola with equation y² = 16x, its directrix has equation x = -4 -/
theorem parabola_directrix (x y : ℝ) : 
  (y^2 = 16*x) → (∃ p : ℝ, p = 4 ∧ x = -p) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3228_322846


namespace NUMINAMATH_CALUDE_no_maximum_value_l3228_322843

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def symmetric_about_point (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (2*a - x) = 2*b - f x

theorem no_maximum_value (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_sym : symmetric_about_point f 1 1) : 
  ¬ ∃ M, ∀ x, f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_no_maximum_value_l3228_322843


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l3228_322898

theorem simplify_and_evaluate_expression :
  let x : ℝ := Real.sqrt 5 - 2
  (2 / (x^2 - 4)) / (1 - x / (x - 2)) = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l3228_322898


namespace NUMINAMATH_CALUDE_cash_percentage_proof_l3228_322835

def total_amount : ℝ := 7428.57
def raw_materials : ℝ := 5000
def machinery : ℝ := 200

theorem cash_percentage_proof :
  let spent := raw_materials + machinery
  let cash := total_amount - spent
  let percentage := (cash / total_amount) * 100
  ∀ ε > 0, |percentage - 29.99| < ε :=
by sorry

end NUMINAMATH_CALUDE_cash_percentage_proof_l3228_322835


namespace NUMINAMATH_CALUDE_two_roses_more_expensive_than_three_carnations_l3228_322893

/-- The price of a single rose in yuan -/
def rose_price : ℝ := sorry

/-- The price of a single carnation in yuan -/
def carnation_price : ℝ := sorry

/-- The combined price of 6 roses and 3 carnations -/
def combined_price_1 : ℝ := 6 * rose_price + 3 * carnation_price

/-- The combined price of 4 roses and 5 carnations -/
def combined_price_2 : ℝ := 4 * rose_price + 5 * carnation_price

/-- Theorem stating that the price of 2 roses is higher than the price of 3 carnations -/
theorem two_roses_more_expensive_than_three_carnations 
  (h1 : combined_price_1 > 24)
  (h2 : combined_price_2 < 22) :
  2 * rose_price > 3 * carnation_price :=
by sorry

end NUMINAMATH_CALUDE_two_roses_more_expensive_than_three_carnations_l3228_322893


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_reciprocal_l3228_322827

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 7*x^2 + 10*x - 6

-- State the theorem
theorem roots_sum_of_squares_reciprocal :
  ∃ (a b c : ℝ), (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (1 / a^2 + 1 / b^2 + 1 / c^2 = 46 / 9) := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_reciprocal_l3228_322827


namespace NUMINAMATH_CALUDE_max_pieces_of_pie_l3228_322864

def is_valid_assignment (p k u s o i r g : ℕ) : Prop :=
  p ≠ k ∧ p ≠ u ∧ p ≠ s ∧ p ≠ o ∧ p ≠ i ∧ p ≠ r ∧ p ≠ g ∧
  k ≠ u ∧ k ≠ s ∧ k ≠ o ∧ k ≠ i ∧ k ≠ r ∧ k ≠ g ∧
  u ≠ s ∧ u ≠ o ∧ u ≠ i ∧ u ≠ r ∧ u ≠ g ∧
  s ≠ o ∧ s ≠ i ∧ s ≠ r ∧ s ≠ g ∧
  o ≠ i ∧ o ≠ r ∧ o ≠ g ∧
  i ≠ r ∧ i ≠ g ∧
  r ≠ g ∧
  p ≠ 0 ∧ k ≠ 0

def pirog (p i r o g : ℕ) : ℕ := p * 10000 + i * 1000 + r * 100 + o * 10 + g

def kusok (k u s o k : ℕ) : ℕ := k * 10000 + u * 1000 + s * 100 + o * 10 + k

theorem max_pieces_of_pie :
  ∀ p i r o g k u s n,
    is_valid_assignment p k u s o i r g →
    pirog p i r o g = n * kusok k u s o k →
    n ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_pieces_of_pie_l3228_322864


namespace NUMINAMATH_CALUDE_hypotenuse_length_l3228_322814

-- Define the points A and B
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (-2, 4)
def O : ℝ × ℝ := (0, 0)

-- Define the properties
theorem hypotenuse_length :
  -- A and B are on the graph of y = x^2
  (A.2 = A.1^2) →
  (B.2 = B.1^2) →
  -- Triangle ABO forms a right triangle at O
  ((A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 0) →
  -- A and B are symmetric about the y-axis
  (A.1 = -B.1) →
  (A.2 = B.2) →
  -- The x-coordinate of A is 2
  (A.1 = 2) →
  -- The length of hypotenuse AB is 4
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l3228_322814


namespace NUMINAMATH_CALUDE_minimal_edge_count_l3228_322829

/-- A graph with 7 vertices satisfying the given conditions -/
structure MinimalGraph where
  -- The set of vertices
  V : Finset ℕ
  -- The set of edges
  E : Finset (Finset ℕ)
  -- There are exactly 7 vertices
  vertex_count : V.card = 7
  -- Each edge connects exactly two vertices
  edge_valid : ∀ e ∈ E, e.card = 2 ∧ e ⊆ V
  -- Among any three vertices, at least two are connected
  connected_condition : ∀ {a b c}, a ∈ V → b ∈ V → c ∈ V → a ≠ b → b ≠ c → a ≠ c →
    {a, b} ∈ E ∨ {b, c} ∈ E ∨ {a, c} ∈ E

/-- The theorem stating that the minimal number of edges is 9 -/
theorem minimal_edge_count (G : MinimalGraph) : G.E.card = 9 := by
  sorry

end NUMINAMATH_CALUDE_minimal_edge_count_l3228_322829


namespace NUMINAMATH_CALUDE_percentage_problem_l3228_322842

theorem percentage_problem (a b c : ℝ) : 
  a = 0.8 * b → 
  c = 1.4 * b → 
  c - a = 72 → 
  a = 96 ∧ b = 120 ∧ c = 168 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3228_322842


namespace NUMINAMATH_CALUDE_sqrt_x_plus_inverse_sqrt_x_l3228_322878

theorem sqrt_x_plus_inverse_sqrt_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_inverse_sqrt_x_l3228_322878


namespace NUMINAMATH_CALUDE_triple_solution_l3228_322830

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def is_solution (a b c : ℕ+) : Prop :=
  (a.val > 0 ∧ b.val > 0 ∧ c.val > 0) ∧
  is_integer ((a + b : ℚ)^4 / c + (b + c : ℚ)^4 / a + (c + a : ℚ)^4 / b) ∧
  Nat.Prime (a + b + c)

theorem triple_solution :
  ∀ a b c : ℕ+, is_solution a b c ↔
    ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (2, 2, 1) ∨ (a, b, c) = (6, 3, 2)) ∨
    ((a, b, c) = (1, 2, 2) ∨ (a, b, c) = (2, 1, 2) ∨ (a, b, c) = (2, 2, 1)) ∨
    ((a, b, c) = (6, 3, 2) ∨ (a, b, c) = (6, 2, 3) ∨ (a, b, c) = (3, 6, 2) ∨
     (a, b, c) = (3, 2, 6) ∨ (a, b, c) = (2, 6, 3) ∨ (a, b, c) = (2, 3, 6)) :=
by sorry

end NUMINAMATH_CALUDE_triple_solution_l3228_322830


namespace NUMINAMATH_CALUDE_max_profit_increase_2008_l3228_322890

def profit_growth : Fin 10 → ℝ
  | ⟨0, _⟩ => 20
  | ⟨1, _⟩ => 40
  | ⟨2, _⟩ => 60
  | ⟨3, _⟩ => 65
  | ⟨4, _⟩ => 80
  | ⟨5, _⟩ => 85
  | ⟨6, _⟩ => 90
  | ⟨7, _⟩ => 95
  | ⟨8, _⟩ => 100
  | ⟨9, _⟩ => 80

def year_from_index (i : Fin 10) : ℕ := 2000 + 2 * i.val

def profit_increase (i : Fin 9) : ℝ := profit_growth (i.succ) - profit_growth i

theorem max_profit_increase_2008 :
  ∃ (i : Fin 9), year_from_index i.succ = 2008 ∧
  ∀ (j : Fin 9), profit_increase i ≥ profit_increase j :=
by sorry

end NUMINAMATH_CALUDE_max_profit_increase_2008_l3228_322890


namespace NUMINAMATH_CALUDE_relationship_x_y_l3228_322804

theorem relationship_x_y (x y : ℝ) 
  (h1 : 3 * x - 2 * y > 4 * x + 1) 
  (h2 : 2 * x + 3 * y < 5 * y - 2) : 
  x < 1 - y := by
sorry

end NUMINAMATH_CALUDE_relationship_x_y_l3228_322804


namespace NUMINAMATH_CALUDE_base_6_divisibility_l3228_322807

def base_6_to_decimal (y : ℕ) : ℕ := 2 * 6^3 + 4 * 6^2 + y * 6 + 2

def is_valid_base_6_digit (y : ℕ) : Prop := y ≤ 5

theorem base_6_divisibility (y : ℕ) : 
  is_valid_base_6_digit y → (base_6_to_decimal y % 13 = 0 ↔ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_base_6_divisibility_l3228_322807


namespace NUMINAMATH_CALUDE_ten_factorial_divided_by_four_factorial_l3228_322861

theorem ten_factorial_divided_by_four_factorial :
  (10 : ℕ).factorial / (4 : ℕ).factorial = 151200 := by
  have h1 : (10 : ℕ).factorial = 3628800 := by sorry
  sorry

end NUMINAMATH_CALUDE_ten_factorial_divided_by_four_factorial_l3228_322861


namespace NUMINAMATH_CALUDE_alien_martian_limb_difference_l3228_322802

/-- The number of arms an alien has -/
def alien_arms : ℕ := 3

/-- The number of legs an alien has -/
def alien_legs : ℕ := 8

/-- The number of arms a Martian has -/
def martian_arms : ℕ := 2 * alien_arms

/-- The number of legs a Martian has -/
def martian_legs : ℕ := alien_legs / 2

/-- The total number of limbs an alien has -/
def alien_limbs : ℕ := alien_arms + alien_legs

/-- The total number of limbs a Martian has -/
def martian_limbs : ℕ := martian_arms + martian_legs

/-- The number of aliens and Martians being compared -/
def group_size : ℕ := 5

theorem alien_martian_limb_difference :
  group_size * alien_limbs - group_size * martian_limbs = 5 := by
  sorry

end NUMINAMATH_CALUDE_alien_martian_limb_difference_l3228_322802


namespace NUMINAMATH_CALUDE_probability_of_defective_product_l3228_322845

/-- Given a product with three grades (first-grade, second-grade, and defective),
    prove that the probability of selecting a defective product is 0.05,
    given the probabilities of selecting first-grade and second-grade products. -/
theorem probability_of_defective_product
  (p_first : ℝ)
  (p_second : ℝ)
  (h_first : p_first = 0.65)
  (h_second : p_second = 0.3)
  (h_nonneg_first : 0 ≤ p_first)
  (h_nonneg_second : 0 ≤ p_second)
  (h_sum_le_one : p_first + p_second ≤ 1) :
  1 - (p_first + p_second) = 0.05 := by
sorry

end NUMINAMATH_CALUDE_probability_of_defective_product_l3228_322845


namespace NUMINAMATH_CALUDE_function_value_theorem_l3228_322859

/-- Given a function f(x) = ax³ - bx + |x| - 1 where f(-8) = 3, prove that f(8) = 11 -/
theorem function_value_theorem (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 - b * x + |x| - 1
  (f (-8) = 3) → (f 8 = 11) := by
sorry

end NUMINAMATH_CALUDE_function_value_theorem_l3228_322859


namespace NUMINAMATH_CALUDE_solve_manuscript_typing_l3228_322840

def manuscript_typing_problem (total_pages : ℕ) (twice_revised : ℕ) (first_typing_cost : ℕ) (revision_cost : ℕ) (total_cost : ℕ) : Prop :=
  ∃ (once_revised : ℕ),
    once_revised + twice_revised ≤ total_pages ∧
    first_typing_cost * total_pages + revision_cost * once_revised + 2 * revision_cost * twice_revised = total_cost ∧
    once_revised = 30

theorem solve_manuscript_typing :
  manuscript_typing_problem 100 20 5 4 780 :=
sorry

end NUMINAMATH_CALUDE_solve_manuscript_typing_l3228_322840


namespace NUMINAMATH_CALUDE_sum_in_base8_l3228_322818

def base8_to_decimal (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.foldr (fun d acc => acc * 8 + d) 0

theorem sum_in_base8 :
  let a := base8_to_decimal 245
  let b := base8_to_decimal 174
  let c := base8_to_decimal 354
  let sum := a + b + c
  base8_to_decimal 1015 = sum := by
sorry

end NUMINAMATH_CALUDE_sum_in_base8_l3228_322818


namespace NUMINAMATH_CALUDE_student_count_difference_l3228_322887

/-- Represents the number of students in each grade level -/
structure StudentCounts where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- The problem statement -/
theorem student_count_difference (counts : StudentCounts) : 
  counts.freshmen + counts.sophomores + counts.juniors + counts.seniors = 800 →
  counts.juniors = 208 →
  counts.sophomores = 200 →
  counts.seniors = 160 →
  counts.freshmen - counts.sophomores = 32 := by
  sorry

end NUMINAMATH_CALUDE_student_count_difference_l3228_322887


namespace NUMINAMATH_CALUDE_bicycle_sampling_is_systematic_l3228_322870

-- Define the sampling method
structure SamplingMethod where
  location : String
  selectionCriteria : String

-- Define systematic sampling
def isSystematicSampling (method : SamplingMethod) : Prop :=
  method.location = "main road" ∧ 
  method.selectionCriteria = "6-digit license plate numbers"

-- Define the specific sampling method used in the problem
def bicycleSamplingMethod : SamplingMethod :=
  { location := "main road"
  , selectionCriteria := "6-digit license plate numbers" }

-- Theorem statement
theorem bicycle_sampling_is_systematic :
  isSystematicSampling bicycleSamplingMethod :=
by sorry


end NUMINAMATH_CALUDE_bicycle_sampling_is_systematic_l3228_322870


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l3228_322809

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x^2 - 2*x - 2

-- State the theorem
theorem derivative_f_at_1 :
  HasDerivAt f 3 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l3228_322809


namespace NUMINAMATH_CALUDE_factor_and_divisor_properties_l3228_322856

theorem factor_and_divisor_properties :
  (∃ n : ℕ, 25 = 5 * n) ∧
  (∃ m : ℕ, 171 = 9 * m) ∧
  ¬(209 % 19 = 0 ∧ 57 % 19 ≠ 0) ∧
  (90 % 30 = 0 ∨ 75 % 30 = 0) ∧
  ¬(51 % 17 = 0 ∧ 68 % 17 ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_factor_and_divisor_properties_l3228_322856


namespace NUMINAMATH_CALUDE_janes_shopping_theorem_l3228_322857

theorem janes_shopping_theorem :
  ∀ (s f : ℕ),
  s + f = 7 →
  (90 * s + 60 * f) % 100 = 0 →
  s = 4 :=
by sorry

end NUMINAMATH_CALUDE_janes_shopping_theorem_l3228_322857
