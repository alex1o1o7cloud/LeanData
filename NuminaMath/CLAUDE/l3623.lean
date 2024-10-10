import Mathlib

namespace sum_of_repeating_decimals_l3623_362369

def repeating_decimal_1 : ℚ := 1/3
def repeating_decimal_2 : ℚ := 4/99
def repeating_decimal_3 : ℚ := 5/999

theorem sum_of_repeating_decimals :
  repeating_decimal_1 + repeating_decimal_2 + repeating_decimal_3 = 42/111 := by
  sorry

end sum_of_repeating_decimals_l3623_362369


namespace stratified_sampling_l3623_362390

theorem stratified_sampling (total_employees : ℕ) (employees_over_30 : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 49)
  (h2 : employees_over_30 = 14)
  (h3 : sample_size = 7) :
  ↑employees_over_30 / ↑total_employees * ↑sample_size = 2 :=
by sorry

end stratified_sampling_l3623_362390


namespace dot_product_zero_on_diagonal_l3623_362310

/-- A square with side length 1 -/
structure UnitSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_unit_square : A.1 + 1 = B.1 ∧ A.2 + 1 = B.2 ∧
                   C.1 = B.1 ∧ C.2 = B.2 + 1 ∧
                   D.1 = A.1 ∧ D.2 = C.2

/-- A point on the diagonal AC of a unit square -/
def PointOnDiagonal (square : UnitSquare) : Type :=
  {P : ℝ × ℝ // ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    P.1 = square.A.1 + t * (square.C.1 - square.A.1) ∧
    P.2 = square.A.2 + t * (square.C.2 - square.A.2)}

/-- Vector from point A to point P -/
def vec_AP (square : UnitSquare) (P : PointOnDiagonal square) : ℝ × ℝ :=
  (P.val.1 - square.A.1, P.val.2 - square.A.2)

/-- Vector from point P to point B -/
def vec_PB (square : UnitSquare) (P : PointOnDiagonal square) : ℝ × ℝ :=
  (square.B.1 - P.val.1, square.B.2 - P.val.2)

/-- Vector from point P to point D -/
def vec_PD (square : UnitSquare) (P : PointOnDiagonal square) : ℝ × ℝ :=
  (square.D.1 - P.val.1, square.D.2 - P.val.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem dot_product_zero_on_diagonal (square : UnitSquare) (P : PointOnDiagonal square) :
  dot_product (vec_AP square P) (vec_PB square P + vec_PD square P) = 0 := by
  sorry

end dot_product_zero_on_diagonal_l3623_362310


namespace existence_of_integers_satisfying_inequality_l3623_362318

theorem existence_of_integers_satisfying_inequality :
  ∃ (a b : ℤ), (2003 : ℝ) < (a : ℝ) + (b : ℝ) * Real.sqrt 2 ∧ 
  (a : ℝ) + (b : ℝ) * Real.sqrt 2 < 2003.01 := by
  sorry

end existence_of_integers_satisfying_inequality_l3623_362318


namespace n_has_five_digits_l3623_362349

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 15 -/
axiom n_div_15 : 15 ∣ n

/-- n^2 is a perfect fourth power -/
axiom n_sq_fourth_power : ∃ k : ℕ, n^2 = k^4

/-- n^4 is a perfect square -/
axiom n_fourth_square : ∃ m : ℕ, n^4 = m^2

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_smallest : ∀ k : ℕ, k > 0 → (15 ∣ k) → (∃ a : ℕ, k^2 = a^4) → (∃ b : ℕ, k^4 = b^2) → n ≤ k

/-- The number of digits in n -/
def digits (m : ℕ) : ℕ := sorry

/-- Theorem stating that n has 5 digits -/
theorem n_has_five_digits : digits n = 5 := sorry

end n_has_five_digits_l3623_362349


namespace range_of_m_l3623_362367

def p (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ a = 8 - m ∧ b = 2 * m - 1

def q (m : ℝ) : Prop :=
  (m + 1) * (m - 2) < 0

theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ Set.Ioo (-1 : ℝ) (1/2) ∪ Set.Ico 2 3 :=
sorry

end range_of_m_l3623_362367


namespace smallest_valid_assembly_is_four_l3623_362376

/-- Represents a modified cube with snaps and receptacles -/
structure ModifiedCube :=
  (snaps : Fin 2)
  (receptacles : Fin 4)

/-- Represents an assembly of modified cubes -/
structure CubeAssembly :=
  (cubes : List ModifiedCube)
  (all_snaps_covered : Bool)
  (only_receptacles_visible : Bool)

/-- Returns true if the assembly is valid according to the problem constraints -/
def is_valid_assembly (assembly : CubeAssembly) : Prop :=
  assembly.all_snaps_covered ∧ assembly.only_receptacles_visible

/-- The smallest number of cubes needed for a valid assembly -/
def smallest_valid_assembly : ℕ := 4

/-- Theorem stating that the smallest valid assembly consists of 4 cubes -/
theorem smallest_valid_assembly_is_four :
  ∀ (assembly : CubeAssembly),
    is_valid_assembly assembly →
    assembly.cubes.length ≥ smallest_valid_assembly :=
by sorry

end smallest_valid_assembly_is_four_l3623_362376


namespace composite_has_at_least_three_factors_l3623_362374

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ k ∣ n

/-- The number of factors of a natural number -/
def NumberOfFactors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem composite_has_at_least_three_factors (n : ℕ) (h : IsComposite n) :
    NumberOfFactors n ≥ 3 := by
  sorry

end composite_has_at_least_three_factors_l3623_362374


namespace azalea_profit_l3623_362395

/-- Calculates the profit from a sheep farm given the number of sheep, shearing cost, wool per sheep, and price per pound of wool. -/
def sheep_farm_profit (num_sheep : ℕ) (shearing_cost : ℕ) (wool_per_sheep : ℕ) (price_per_pound : ℕ) : ℕ :=
  num_sheep * wool_per_sheep * price_per_pound - shearing_cost

/-- Proves that Azalea's profit from her sheep farm is $38,000 -/
theorem azalea_profit : sheep_farm_profit 200 2000 10 20 = 38000 := by
  sorry

end azalea_profit_l3623_362395


namespace circle_placement_possible_l3623_362313

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square -/
structure Square where
  center : Point

/-- The main theorem -/
theorem circle_placement_possible
  (rect : Rectangle)
  (squares : Finset Square)
  (h_rect_dim : rect.width = 20 ∧ rect.height = 25)
  (h_squares_count : squares.card = 120) :
  ∃ (p : Point),
    (0.5 ≤ p.x ∧ p.x ≤ rect.width - 0.5) ∧
    (0.5 ≤ p.y ∧ p.y ≤ rect.height - 0.5) ∧
    ∀ (s : Square), s ∈ squares →
      (p.x - s.center.x)^2 + (p.y - s.center.y)^2 ≥ 1 :=
sorry

end circle_placement_possible_l3623_362313


namespace complement_of_union_l3623_362350

def S : Finset Nat := {1,2,3,4,5,6,7,8,9,10}
def A : Finset Nat := {2,4,6,8,10}
def B : Finset Nat := {3,6,9}

theorem complement_of_union (S A B : Finset Nat) :
  S = {1,2,3,4,5,6,7,8,9,10} →
  A = {2,4,6,8,10} →
  B = {3,6,9} →
  S \ (A ∪ B) = {1,5,7} :=
by sorry

end complement_of_union_l3623_362350


namespace product_expansion_l3623_362377

theorem product_expansion (x : ℝ) : 
  (7 * x^2 + 3) * (5 * x^3 + 2 * x + 1) = 35 * x^5 + 29 * x^3 + 7 * x^2 + 6 * x + 3 := by
  sorry

end product_expansion_l3623_362377


namespace newborn_count_l3623_362302

theorem newborn_count (total_children : ℕ) (toddlers : ℕ) : 
  total_children = 40 → 
  toddlers = 6 → 
  total_children = toddlers + 5 * toddlers + (total_children - toddlers - 5 * toddlers) → 
  (total_children - toddlers - 5 * toddlers) = 4 := by
sorry

end newborn_count_l3623_362302


namespace smallest_prime_with_prime_digit_sum_l3623_362315

def digit_sum (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_prime_digit_sum :
  ∃ (p : Nat), is_prime p ∧ 
               is_prime (digit_sum p) ∧ 
               digit_sum p > 10 ∧ 
               (∀ q : Nat, q < p → ¬(is_prime q ∧ is_prime (digit_sum q) ∧ digit_sum q > 10)) ∧
               p = 29 := by
  sorry

end smallest_prime_with_prime_digit_sum_l3623_362315


namespace largest_even_digit_multiple_of_11_l3623_362371

def is_even_digit (d : ℕ) : Prop := d % 2 = 0 ∧ d < 10

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_even_digit d

theorem largest_even_digit_multiple_of_11 :
  ∀ n : ℕ,
    n < 10000 →
    has_only_even_digits n →
    n % 11 = 0 →
    n ≤ 8800 :=
sorry

end largest_even_digit_multiple_of_11_l3623_362371


namespace intersection_size_l3623_362306

/-- Given a finite universe U and two subsets A and B, 
    this theorem calculates the size of their intersection. -/
theorem intersection_size 
  (U A B : Finset ℕ) 
  (h1 : A ⊆ U) 
  (h2 : B ⊆ U) 
  (h3 : Finset.card U = 215)
  (h4 : Finset.card A = 170)
  (h5 : Finset.card B = 142)
  (h6 : Finset.card (U \ (A ∪ B)) = 38) :
  Finset.card (A ∩ B) = 135 := by
sorry

end intersection_size_l3623_362306


namespace two_sector_area_l3623_362351

/-- The area of a figure formed by two sectors of a circle -/
theorem two_sector_area (r : ℝ) (angle1 angle2 : ℝ) (h1 : r = 15) (h2 : angle1 = 90) (h3 : angle2 = 45) :
  (angle1 / 360) * π * r^2 + (angle2 / 360) * π * r^2 = 84.375 * π := by
  sorry

#check two_sector_area

end two_sector_area_l3623_362351


namespace passing_marks_l3623_362330

/-- The number of marks for passing an exam, given conditions about failing and passing candidates. -/
theorem passing_marks (T : ℝ) (P : ℝ) : 
  (0.4 * T = P - 40) →  -- Condition 1
  (0.6 * T = P + 20) →  -- Condition 2
  P = 160 := by
sorry

end passing_marks_l3623_362330


namespace cube_root_strict_mono_l3623_362340

theorem cube_root_strict_mono {a b : ℝ} (h : a < b) : ¬(a^(1/3) ≥ b^(1/3)) := by
  sorry

end cube_root_strict_mono_l3623_362340


namespace exists_even_in_sequence_l3623_362339

/-- A sequence of natural numbers where each subsequent number is obtained by adding one of its non-zero digits to the previous number. -/
def DigitAdditionSequence : Type :=
  ℕ → ℕ

/-- Property that defines the sequence: each subsequent number is obtained by adding one of its non-zero digits to the previous number. -/
def IsValidSequence (seq : DigitAdditionSequence) : Prop :=
  ∀ n : ℕ, ∃ d : ℕ, d > 0 ∧ d < 10 ∧ seq (n + 1) = seq n + d

/-- Theorem stating that there exists an even number in the sequence. -/
theorem exists_even_in_sequence (seq : DigitAdditionSequence) (h : IsValidSequence seq) :
  ∃ n : ℕ, Even (seq n) := by
  sorry

end exists_even_in_sequence_l3623_362339


namespace function_range_l3623_362363

theorem function_range (t : ℝ) : 
  (∃ x : ℝ, t ≤ x ∧ x ≤ t + 2 ∧ x^2 + t*x - 12 ≤ 0) → 
  -4 ≤ t ∧ t ≤ Real.sqrt 6 := by
  sorry

end function_range_l3623_362363


namespace min_quotient_four_digit_number_l3623_362346

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  (∃ a b c d : ℕ, n = 1000 * a + 100 * b + 10 * c + d ∧
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (Even a ∨ Even b ∨ Even c ∨ Even d) ∧
    (Even a ∧ Even b ∨ Even a ∧ Even c ∨ Even a ∧ Even d ∨
     Even b ∧ Even c ∨ Even b ∧ Even d ∨ Even c ∧ Even d))

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

theorem min_quotient_four_digit_number :
  ∀ n : ℕ, is_valid_number n → (n : ℚ) / (digit_sum n : ℚ) ≥ 87 :=
by sorry

end min_quotient_four_digit_number_l3623_362346


namespace medical_team_arrangements_l3623_362332

/-- The number of male doctors --/
def num_male_doctors : ℕ := 6

/-- The number of female nurses --/
def num_female_nurses : ℕ := 3

/-- The number of medical teams --/
def num_teams : ℕ := 3

/-- The number of male doctors per team --/
def doctors_per_team : ℕ := 2

/-- The number of female nurses per team --/
def nurses_per_team : ℕ := 1

/-- The number of distinct locations --/
def num_locations : ℕ := 3

/-- The total number of arrangements --/
def total_arrangements : ℕ := 540

theorem medical_team_arrangements :
  (num_male_doctors.choose doctors_per_team *
   (num_male_doctors - doctors_per_team).choose doctors_per_team *
   (num_male_doctors - 2 * doctors_per_team).choose doctors_per_team) /
  num_teams.factorial *
  num_teams.factorial *
  num_teams.factorial = total_arrangements :=
sorry

end medical_team_arrangements_l3623_362332


namespace rectangle_count_6x5_grid_l3623_362307

/-- Represents a grid of lines in a coordinate plane -/
structure Grid :=
  (horizontal_lines : ℕ)
  (vertical_lines : ℕ)

/-- Represents a point in a 2D coordinate plane -/
structure Point :=
  (x : ℕ)
  (y : ℕ)

/-- Counts the number of ways to form a rectangle enclosing a given point -/
def count_rectangles (g : Grid) (p : Point) : ℕ :=
  sorry

/-- Theorem stating the number of ways to form a rectangle enclosing (3, 4) in a 6x5 grid -/
theorem rectangle_count_6x5_grid :
  let g : Grid := ⟨6, 5⟩
  let p : Point := ⟨3, 4⟩
  count_rectangles g p = 24 :=
sorry

end rectangle_count_6x5_grid_l3623_362307


namespace last_three_average_l3623_362387

theorem last_three_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 62 →
  (list.take 4).sum / 4 = 54 →
  (list.drop 4).sum / 3 = 72.67 := by
sorry

end last_three_average_l3623_362387


namespace trigonometric_expression_equality_l3623_362359

theorem trigonometric_expression_equality : 
  (Real.tan (150 * π / 180)) * (Real.cos (-210 * π / 180)) * (Real.sin (-420 * π / 180)) / 
  ((Real.sin (1050 * π / 180)) * (Real.cos (-600 * π / 180))) = -Real.sqrt 3 := by
  sorry

end trigonometric_expression_equality_l3623_362359


namespace china_space_station_orbit_height_scientific_notation_l3623_362319

theorem china_space_station_orbit_height_scientific_notation :
  let orbit_height : ℝ := 400000
  orbit_height = 4 * (10 : ℝ)^5 := by sorry

end china_space_station_orbit_height_scientific_notation_l3623_362319


namespace inequality_solution_set_l3623_362355

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ ((m - 1) * x < Real.sqrt (4 * x - x^2))) → 
  m = 2 := by
sorry

end inequality_solution_set_l3623_362355


namespace min_area_bounded_by_curve_and_lines_l3623_362333

noncomputable section

-- Define the curve C
def f (x : ℝ) : ℝ := 1 / (1 + x^2)

-- Define the area function T(α)
def T (α : ℝ) : ℝ :=
  Real.arctan α + Real.arctan (1 / α) - α / (1 + α^2)

-- Theorem statement
theorem min_area_bounded_by_curve_and_lines (α : ℝ) (h : α > 0) :
  ∃ (min_area : ℝ), min_area = π / 2 - 1 / 2 ∧
  ∀ β > 0, T β ≥ min_area :=
sorry

end min_area_bounded_by_curve_and_lines_l3623_362333


namespace bc_over_a_is_zero_l3623_362312

theorem bc_over_a_is_zero (a b c : ℝ) 
  (h1 : a = 2*b + Real.sqrt 2)
  (h2 : a*b + (Real.sqrt 3 / 2)*c^2 + 1/4 = 0) : 
  b*c/a = 0 := by
  sorry

end bc_over_a_is_zero_l3623_362312


namespace allowance_calculation_l3623_362380

theorem allowance_calculation (card_cost sticker_box_cost : ℚ) 
  (total_sticker_packs : ℕ) (h1 : card_cost = 10) 
  (h2 : sticker_box_cost = 2) (h3 : total_sticker_packs = 4) : 
  (card_cost + sticker_box_cost * (total_sticker_packs / 2)) / 2 = 9 := by
  sorry

end allowance_calculation_l3623_362380


namespace axis_of_symmetry_at_1_5_l3623_362378

/-- A function g is symmetric about x = 1.5 if g(x) = g(3-x) for all x -/
def IsSymmetricAbout1_5 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (3 - x)

/-- The line x = 1.5 is an axis of symmetry for g if g is symmetric about x = 1.5 -/
theorem axis_of_symmetry_at_1_5 (g : ℝ → ℝ) (h : IsSymmetricAbout1_5 g) :
  ∀ x y, g x = y → g (3 - x) = y :=
by sorry

end axis_of_symmetry_at_1_5_l3623_362378


namespace tangent_point_relation_l3623_362317

-- Define the curve and tangent line
def curve (x a b : ℝ) : ℝ := x^3 + a*x + b
def tangent_line (x k : ℝ) : ℝ := k*x + 1

-- State the theorem
theorem tangent_point_relation (a b k : ℝ) : 
  (∃ x y, x = 1 ∧ y = 3 ∧ 
    curve x a b = y ∧ 
    tangent_line x k = y ∧
    (∀ x', curve x' a b = tangent_line x' k → x' = x)) →
  2*a + b = 1 := by
sorry

end tangent_point_relation_l3623_362317


namespace inequality_proof_l3623_362347

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a / (a^2 + 2)) + (b / (b^2 + 2)) + (c / (c^2 + 2)) ≤ 1 := by
  sorry

end inequality_proof_l3623_362347


namespace integer_square_root_l3623_362358

theorem integer_square_root (x : ℤ) : 
  (∃ n : ℤ, n ≥ 0 ∧ n^2 = x^2 - x + 1) ↔ (x = 0 ∨ x = 1) :=
by sorry

end integer_square_root_l3623_362358


namespace cube_dimensions_l3623_362343

-- Define the surface area of the cube
def surface_area : ℝ := 864

-- Theorem stating the side length and diagonal of the cube
theorem cube_dimensions (s d : ℝ) : 
  (6 * s^2 = surface_area) → 
  (s = 12) ∧ 
  (d = 12 * Real.sqrt 3) := by
  sorry

end cube_dimensions_l3623_362343


namespace triangle_height_l3623_362308

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 12 → area = 30 → area = (base * height) / 2 → height = 5 := by
  sorry

end triangle_height_l3623_362308


namespace horner_v2_value_l3623_362354

/-- Horner's method for a polynomial --/
def horner_step (x : ℝ) (a b : ℝ) : ℝ := a * x + b

/-- The polynomial f(x) = x^6 + 6x^4 + 9x^2 + 208 --/
def f (x : ℝ) : ℝ := x^6 + 6*x^4 + 9*x^2 + 208

/-- Theorem: The value of v₂ in Horner's method for f(x) at x = -4 is 22 --/
theorem horner_v2_value :
  let x : ℝ := -4
  let v0 : ℝ := 1
  let v1 : ℝ := horner_step x v0 0
  let v2 : ℝ := horner_step x v1 6
  v2 = 22 := by sorry

end horner_v2_value_l3623_362354


namespace tangent_slope_at_negative_one_l3623_362335

def curve (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2

def tangent_slope (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2

theorem tangent_slope_at_negative_one (a : ℝ) :
  tangent_slope a (-1) = Real.tan (π/4) → a = 1/3 := by
  sorry

end tangent_slope_at_negative_one_l3623_362335


namespace ab_relation_to_a_over_b_l3623_362384

theorem ab_relation_to_a_over_b (a b : ℝ) (h : a * b ≠ 0) :
  ¬(∀ a b, a * b > 1 → a > 1 / b) ∧
  ¬(∀ a b, a > 1 / b → a * b > 1) := by
  sorry

end ab_relation_to_a_over_b_l3623_362384


namespace van_capacity_l3623_362305

theorem van_capacity (students : ℕ) (adults : ℕ) (vans : ℕ) : 
  students = 22 → adults = 2 → vans = 3 → (students + adults) / vans = 8 := by
  sorry

end van_capacity_l3623_362305


namespace sin_plus_2cos_equals_two_fifths_l3623_362399

theorem sin_plus_2cos_equals_two_fifths (a : ℝ) (α : ℝ) :
  a < 0 →
  (∃ (x y : ℝ), x = -3*a ∧ y = 4*a ∧ Real.sin α = y / Real.sqrt (x^2 + y^2) ∧ Real.cos α = x / Real.sqrt (x^2 + y^2)) →
  Real.sin α + 2 * Real.cos α = 2/5 := by
sorry

end sin_plus_2cos_equals_two_fifths_l3623_362399


namespace g_27_is_zero_l3623_362360

/-- A function satisfying the given property -/
def special_function (g : ℕ → ℕ) : Prop :=
  ∀ a b c : ℕ, 3 * g (a^2 + b^2 + c^2) = (g a)^2 + (g b)^2 + (g c)^2

/-- The theorem stating that g(27) = 0 for any function satisfying the special property -/
theorem g_27_is_zero (g : ℕ → ℕ) (h : special_function g) : g 27 = 0 := by
  sorry

#check g_27_is_zero

end g_27_is_zero_l3623_362360


namespace diana_wins_probability_l3623_362342

def diana_die : ℕ := 6
def apollo_die : ℕ := 4

def favorable_outcomes : ℕ := 14
def total_outcomes : ℕ := diana_die * apollo_die

theorem diana_wins_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 12 :=
sorry

end diana_wins_probability_l3623_362342


namespace power_of_five_l3623_362361

theorem power_of_five (n : ℕ) : 5^n = 5 * 25^2 * 125^3 → n = 14 := by
  sorry

end power_of_five_l3623_362361


namespace unique_three_digit_number_l3623_362372

theorem unique_three_digit_number : ∃! x : ℕ, 
  100 ≤ x ∧ x < 1000 ∧ 
  (∃ k : ℤ, x - 7 = 7 * k) ∧
  (∃ l : ℤ, x - 8 = 8 * l) ∧
  (∃ m : ℤ, x - 9 = 9 * m) :=
by sorry

end unique_three_digit_number_l3623_362372


namespace existence_of_special_integers_l3623_362300

theorem existence_of_special_integers (k : ℕ+) :
  ∃ x y : ℤ, x % 7 ≠ 0 ∧ y % 7 ≠ 0 ∧ x^2 + 6*y^2 = 7^(k : ℕ) := by
  sorry

end existence_of_special_integers_l3623_362300


namespace unique_solution_is_sqrt_2_l3623_362316

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else Real.log x / Real.log 4

theorem unique_solution_is_sqrt_2 :
  ∃! x, x > 1 ∧ f x = 1/4 :=
by
  sorry

end unique_solution_is_sqrt_2_l3623_362316


namespace intersection_A_B_l3623_362365

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (-x^2 + 2*x)}

-- Define set B
def B : Set ℝ := {x | |x| ≤ 1}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end intersection_A_B_l3623_362365


namespace amaya_viewing_time_l3623_362382

/-- Represents the total time Amaya spent watching the movie -/
def total_viewing_time (
  segment1 segment2 segment3 segment4 segment5 : ℕ
) (rewind1 rewind2 rewind3 rewind4 : ℕ) : ℕ :=
  segment1 + segment2 + segment3 + segment4 + segment5 +
  rewind1 + rewind2 + rewind3 + rewind4

/-- Theorem stating that the total viewing time is 170 minutes -/
theorem amaya_viewing_time :
  total_viewing_time 35 45 25 15 20 5 7 10 8 = 170 := by
  sorry

end amaya_viewing_time_l3623_362382


namespace total_food_service_employees_l3623_362328

/-- Represents the number of employees trained for each restaurant combination --/
structure RestaurantTraining where
  b : ℕ  -- Trained for family buffet only
  d : ℕ  -- Trained for dining room only
  s : ℕ  -- Trained for snack bar only
  bd : ℕ -- Trained for family buffet and dining room
  bs : ℕ -- Trained for family buffet and snack bar
  ds : ℕ -- Trained for dining room and snack bar
  bds : ℕ -- Trained for all three restaurants

/-- Calculates the total number of employees trained for each restaurant --/
def total_per_restaurant (rt : RestaurantTraining) : (ℕ × ℕ × ℕ) :=
  (rt.b + rt.bd + rt.bs + rt.bds,
   rt.d + rt.bd + rt.ds + rt.bds,
   rt.s + rt.bs + rt.ds + rt.bds)

/-- Calculates the total number of food service employees --/
def total_employees (rt : RestaurantTraining) : ℕ :=
  rt.b + rt.d + rt.s + rt.bd + rt.bs + rt.ds + rt.bds

/-- Theorem stating the total number of food service employees --/
theorem total_food_service_employees :
  ∀ (rt : RestaurantTraining),
    total_per_restaurant rt = (15, 18, 12) →
    rt.bd + rt.bs + rt.ds = 4 →
    rt.bds = 1 →
    total_employees rt = 39 := by
  sorry


end total_food_service_employees_l3623_362328


namespace sam_tutoring_hours_l3623_362352

/-- Sam's hourly rate for Math tutoring -/
def hourly_rate : ℕ := 10

/-- Sam's earnings for the first month -/
def first_month_earnings : ℕ := 200

/-- The additional amount Sam earned in the second month compared to the first -/
def second_month_increase : ℕ := 150

/-- The total number of hours Sam spent tutoring for two months -/
def total_hours : ℕ := 55

/-- Theorem stating that given the conditions, Sam spent 55 hours tutoring over two months -/
theorem sam_tutoring_hours :
  hourly_rate * total_hours = first_month_earnings + (first_month_earnings + second_month_increase) :=
by sorry

end sam_tutoring_hours_l3623_362352


namespace dividend_divisor_calculation_l3623_362357

/-- Given a dividend of 73648 and a divisor of 874, prove that the result of subtracting
    the product of the divisor and the sum of the quotient's digits from the dividend
    is equal to 63160. -/
theorem dividend_divisor_calculation : 
  let dividend : Nat := 73648
  let divisor : Nat := 874
  let quotient : Nat := dividend / divisor
  let remainder : Nat := dividend % divisor
  let sum_of_digits : Nat := (quotient / 10) + (quotient % 10)
  73648 - (sum_of_digits * 874) = 63160 := by
  sorry

#eval 73648 - ((73648 / 874 / 10 + 73648 / 874 % 10) * 874)

end dividend_divisor_calculation_l3623_362357


namespace problem_statement_l3623_362398

theorem problem_statement (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h1 : a + 2 / b = b + 2 / c) (h2 : b + 2 / c = c + 2 / a) :
  (a + 2 / b)^2 + (b + 2 / c)^2 + (c + 2 / a)^2 = 6 := by
sorry

end problem_statement_l3623_362398


namespace unique_number_with_remainder_l3623_362391

theorem unique_number_with_remainder (n : ℕ) : n < 5000 ∧ 
  (∀ k ∈ Finset.range 9, n % (k + 2) = 1) ↔ n = 2521 := by
  sorry

end unique_number_with_remainder_l3623_362391


namespace dance_move_ratio_l3623_362353

/-- Frank's dance move sequence --/
structure DanceMove where
  initial_back : ℤ
  first_forward : ℤ
  second_back : ℤ
  final_forward : ℤ

/-- The dance move Frank performs --/
def franks_move : DanceMove :=
  { initial_back := 5
  , first_forward := 10
  , second_back := 2
  , final_forward := 4 }

/-- The final position relative to the starting point --/
def final_position (move : DanceMove) : ℤ :=
  -move.initial_back + move.first_forward - move.second_back + move.final_forward

/-- The theorem stating the ratio of final forward steps to second back steps --/
theorem dance_move_ratio (move : DanceMove) : 
  final_position move = 7 → 
  (move.final_forward : ℚ) / move.second_back = 2 := by
  sorry

#eval final_position franks_move

end dance_move_ratio_l3623_362353


namespace integer_root_count_theorem_l3623_362320

/-- A polynomial of degree 5 with integer coefficients -/
def IntPolynomial5 (x b c d e f : ℤ) : ℤ := x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f

/-- The set of possible numbers of integer roots for IntPolynomial5 -/
def PossibleRootCounts : Set ℕ := {0, 1, 2, 5}

/-- Theorem stating that the number of integer roots of IntPolynomial5 is in PossibleRootCounts -/
theorem integer_root_count_theorem (b c d e f : ℤ) :
  ∃ (n : ℕ), n ∈ PossibleRootCounts ∧
  (∃ (roots : List ℤ), (∀ x ∈ roots, IntPolynomial5 x b c d e f = 0) ∧
                       roots.length = n) :=
sorry

end integer_root_count_theorem_l3623_362320


namespace gcd_of_polynomial_and_linear_term_l3623_362386

theorem gcd_of_polynomial_and_linear_term (b : ℤ) (h : 1620 ∣ b) :
  Nat.gcd (Int.natAbs (b^2 + 11*b + 36)) (Int.natAbs (b + 6)) = 6 := by
  sorry

end gcd_of_polynomial_and_linear_term_l3623_362386


namespace freddy_age_l3623_362311

theorem freddy_age (matthew rebecca freddy : ℕ) : 
  matthew + rebecca + freddy = 35 →
  matthew = rebecca + 2 →
  freddy = matthew + 4 →
  ∃ (x : ℕ), matthew = 4 * x ∧ rebecca = 5 * x ∧ freddy = 7 * x →
  freddy = 15 := by
sorry

end freddy_age_l3623_362311


namespace sequence_general_term_l3623_362368

def sequence_property (a : ℕ+ → ℕ+) : Prop :=
  (∀ m k : ℕ+, a (m^2) = (a m)^2) ∧
  (∀ m k : ℕ+, a (m^2 + k^2) = a m * a k)

theorem sequence_general_term (a : ℕ+ → ℕ+) (h : sequence_property a) :
  ∀ n : ℕ+, a n = 1 :=
sorry

end sequence_general_term_l3623_362368


namespace simplify_expression_1_simplify_expression_2_l3623_362348

-- Part 1
theorem simplify_expression_1 (a : ℝ) : a - 2*a + 3*a = 2*a := by
  sorry

-- Part 2
theorem simplify_expression_2 (x y : ℝ) : 3*(2*x - 7*y) - (4*x - 10*y) = 2*x - 11*y := by
  sorry

end simplify_expression_1_simplify_expression_2_l3623_362348


namespace problem_solution_l3623_362326

def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem problem_solution :
  (∀ a : ℝ, A ∪ B a = B a → a = 1) ∧
  (∀ C : Set ℝ, (∀ a : ℝ, A ∩ B a = B a → a ∈ C) → C = {a : ℝ | a ≤ -1 ∨ a = 1}) :=
by sorry

end problem_solution_l3623_362326


namespace tony_paint_area_l3623_362385

/-- The area Tony needs to paint on the wall -/
def area_to_paint (wall_height wall_length door_height door_width window_height window_width : ℝ) : ℝ :=
  wall_height * wall_length - (door_height * door_width + window_height * window_width)

/-- Theorem stating the area Tony needs to paint -/
theorem tony_paint_area :
  area_to_paint 10 15 3 5 2 3 = 129 := by
  sorry

end tony_paint_area_l3623_362385


namespace derek_age_l3623_362389

/-- Given the ages of Uncle Bob, Evan, and Derek, prove Derek's age -/
theorem derek_age (uncle_bob_age : ℕ) (evan_age : ℕ) (derek_age : ℕ) : 
  uncle_bob_age = 60 →
  evan_age = 2 * uncle_bob_age / 3 →
  derek_age = evan_age - 10 →
  derek_age = 30 := by
sorry

end derek_age_l3623_362389


namespace b_101_mod_49_l3623_362375

/-- The sequence b_n defined as 5^n + 7^n -/
def b (n : ℕ) : ℕ := 5^n + 7^n

/-- Theorem stating that b_101 is congruent to 12 modulo 49 -/
theorem b_101_mod_49 : b 101 ≡ 12 [MOD 49] := by
  sorry

end b_101_mod_49_l3623_362375


namespace inequality_proof_l3623_362323

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_cond : |Real.sqrt (a * b) - Real.sqrt (c * d)| ≤ 2) : 
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := by
  sorry

end inequality_proof_l3623_362323


namespace inequality_proof_l3623_362327

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / Real.sqrt b) + (b / Real.sqrt a) ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end inequality_proof_l3623_362327


namespace proposition_analysis_l3623_362309

def p : Prop := 6 ∣ 12
def q : Prop := 6 ∣ 24

theorem proposition_analysis :
  (p ∨ q) ∧ (p ∧ q) ∧ (¬¬p) := by sorry

end proposition_analysis_l3623_362309


namespace sine_cosine_unique_pair_l3623_362373

open Real

theorem sine_cosine_unique_pair :
  ∃! (c d : ℝ), 0 < c ∧ c < d ∧ d < π / 2 ∧
    sin (cos c) = c ∧ cos (sin d) = d := by
  sorry

end sine_cosine_unique_pair_l3623_362373


namespace existence_of_close_pair_l3623_362381

-- Define a type for numbers between 0 and 1
def UnitInterval := {x : ℝ | 0 < x ∧ x < 1}

-- State the theorem
theorem existence_of_close_pair :
  ∀ (x y z : UnitInterval), ∃ (a b : UnitInterval), |a.val - b.val| ≤ 0.5 :=
sorry

end existence_of_close_pair_l3623_362381


namespace no_solution_exists_l3623_362396

theorem no_solution_exists (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : 0 < a₁) (h₂ : a₁ < a₂) (h₃ : a₂ < a₃) (h₄ : a₃ < a₄) :
  ¬ ∃ (k : ℝ) (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧
    x₁ + x₂ + x₃ + x₄ = 1 ∧
    a₁ * x₁ + a₂ * x₂ + a₃ * x₃ + a₄ * x₄ = k ∧
    a₁^2 * x₁ + a₂^2 * x₂ + a₃^2 * x₃ + a₄^2 * x₄ = k^2 :=
by
  sorry

end no_solution_exists_l3623_362396


namespace digits_of_3_pow_24_times_7_pow_36_l3623_362331

theorem digits_of_3_pow_24_times_7_pow_36 : ∃ n : ℕ, 
  n > 0 ∧ n < 10^32 ∧ 10^31 ≤ 3^24 * 7^36 ∧ 3^24 * 7^36 < 10^32 := by
  sorry

end digits_of_3_pow_24_times_7_pow_36_l3623_362331


namespace correct_payment_to_C_l3623_362379

/-- The amount to be paid to worker C -/
def payment_to_C (a_rate b_rate : ℚ) (total_payment : ℕ) (days_to_complete : ℕ) : ℚ :=
  let ab_rate := a_rate + b_rate
  let ab_work := ab_rate * days_to_complete
  let c_work := 1 - ab_work
  c_work * total_payment

/-- Theorem stating the correct payment to worker C -/
theorem correct_payment_to_C :
  payment_to_C (1/6) (1/8) 2400 3 = 300 := by sorry

end correct_payment_to_C_l3623_362379


namespace sin_shift_l3623_362334

theorem sin_shift (x : ℝ) : Real.sin (2 * x + π / 3) = Real.sin (2 * (x + π / 6)) := by
  sorry

end sin_shift_l3623_362334


namespace row_arrangement_counts_l3623_362324

/-- Represents a person in the row -/
inductive Person : Type
| A | B | C | D | E

/-- A row is a permutation of five people -/
def Row := Fin 5 → Person

/-- Checks if A and B are adjacent with B to the right of A in a given row -/
def adjacent_AB (row : Row) : Prop :=
  ∃ i : Fin 4, row i = Person.A ∧ row (i.succ) = Person.B

/-- Checks if A, B, and C are in order from left to right in a given row -/
def ABC_in_order (row : Row) : Prop :=
  ∃ i j k : Fin 5, i < j ∧ j < k ∧ 
    row i = Person.A ∧ row j = Person.B ∧ row k = Person.C

/-- The main theorem to be proved -/
theorem row_arrangement_counts :
  (∃! (s : Finset Row), s.card = 24 ∧ ∀ row ∈ s, adjacent_AB row) ∧
  (∃! (s : Finset Row), s.card = 20 ∧ ∀ row ∈ s, ABC_in_order row) :=
sorry

end row_arrangement_counts_l3623_362324


namespace least_sum_of_exponents_l3623_362345

def target_number : ℕ := 3124

def is_sum_of_distinct_powers_of_two (n : ℕ) (exponents : List ℕ) : Prop :=
  n = (exponents.map (fun e => 2^e)).sum ∧ exponents.Nodup

theorem least_sum_of_exponents :
  ∃ (exponents : List ℕ),
    is_sum_of_distinct_powers_of_two target_number exponents ∧
    ∀ (other_exponents : List ℕ),
      is_sum_of_distinct_powers_of_two target_number other_exponents →
      exponents.sum ≤ other_exponents.sum ∧
      exponents.sum = 32 :=
sorry

end least_sum_of_exponents_l3623_362345


namespace pool_capacity_l3623_362314

theorem pool_capacity : 
  ∀ (initial_fraction final_fraction added_volume total_capacity : ℚ),
  initial_fraction = 1 / 8 →
  final_fraction = 2 / 3 →
  added_volume = 210 →
  (final_fraction - initial_fraction) * total_capacity = added_volume →
  total_capacity = 5040 / 13 := by
sorry

end pool_capacity_l3623_362314


namespace geometric_sequence_sum_l3623_362303

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100 →
  a 4 + a 6 = 10 := by
  sorry

end geometric_sequence_sum_l3623_362303


namespace B_and_C_complementary_l3623_362362

open Set

-- Define the sample space for a fair cubic die
def Ω : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define event B
def B : Set Nat := {n ∈ Ω | n ≤ 3}

-- Define event C
def C : Set Nat := {n ∈ Ω | n ≥ 4}

-- Theorem statement
theorem B_and_C_complementary : B ∪ C = Ω ∧ B ∩ C = ∅ := by
  sorry

end B_and_C_complementary_l3623_362362


namespace smallest_square_area_l3623_362356

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The smallest square that can contain two non-overlapping rectangles -/
def smallest_containing_square (r1 r2 : Rectangle) : ℕ := 
  max (r1.width + r2.width) (max r1.height r2.height)

/-- Theorem: The smallest square containing a 3×5 and a 4×6 rectangle has area 49 -/
theorem smallest_square_area : 
  let r1 : Rectangle := ⟨3, 5⟩
  let r2 : Rectangle := ⟨4, 6⟩
  (smallest_containing_square r1 r2)^2 = 49 := by
sorry

#eval (smallest_containing_square ⟨3, 5⟩ ⟨4, 6⟩)^2

end smallest_square_area_l3623_362356


namespace circle_center_l3623_362338

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8*y = 0

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Theorem stating that (3, -4) is the center of the circle defined by CircleEquation -/
theorem circle_center : 
  ∃ (c : CircleCenter), c.x = 3 ∧ c.y = -4 ∧ 
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - c.x)^2 + (y - c.y)^2 = 25 :=
sorry

end circle_center_l3623_362338


namespace rica_spent_fraction_l3623_362364

theorem rica_spent_fraction (total_prize : ℝ) (rica_fraction : ℝ) (rica_left : ℝ) : 
  total_prize = 1000 →
  rica_fraction = 3/8 →
  rica_left = 300 →
  (total_prize * rica_fraction - rica_left) / (total_prize * rica_fraction) = 1/5 :=
by sorry

end rica_spent_fraction_l3623_362364


namespace floor_abs_sum_equals_57_l3623_362325

theorem floor_abs_sum_equals_57 : ⌊|(-57.85 : ℝ) + 0.1|⌋ = 57 := by sorry

end floor_abs_sum_equals_57_l3623_362325


namespace royalties_for_420_tax_l3623_362322

/-- Calculates the tax on royalties based on the given rules -/
def calculateTax (royalties : ℕ) : ℚ :=
  if royalties ≤ 800 then 0
  else if royalties ≤ 4000 then (royalties - 800) * 14 / 100
  else royalties * 11 / 100

/-- Theorem stating that 3800 yuan in royalties results in 420 yuan tax -/
theorem royalties_for_420_tax : calculateTax 3800 = 420 := by sorry

end royalties_for_420_tax_l3623_362322


namespace hawks_total_points_l3623_362337

theorem hawks_total_points (touchdowns : ℕ) (points_per_touchdown : ℕ) 
  (h1 : touchdowns = 3) 
  (h2 : points_per_touchdown = 7) : 
  touchdowns * points_per_touchdown = 21 := by
  sorry

end hawks_total_points_l3623_362337


namespace sufficient_not_necessary_condition_l3623_362393

theorem sufficient_not_necessary_condition :
  (∀ b : ℝ, b ∈ Set.Ioo 0 4 → ∀ x : ℝ, b * x^2 - b * x + 1 > 0) ∧
  (∃ b : ℝ, b ∉ Set.Ioo 0 4 ∧ ∀ x : ℝ, b * x^2 - b * x + 1 > 0) := by
  sorry

end sufficient_not_necessary_condition_l3623_362393


namespace equation_equivalence_l3623_362344

/-- An equation is homogeneous if for any solution (x, y), (rx, ry) is also a solution for any non-zero scalar r. -/
def IsHomogeneous (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ (x y r : ℝ), r ≠ 0 → f x y = 0 → f (r * x) (r * y) = 0

/-- The original equation -/
def OriginalEq (x y : ℝ) : ℝ := x^3 - 2*x^2*y + x*y^2 - 2*y^3

/-- The equivalent equation -/
def EquivalentEq (x y : ℝ) : Prop := x = 2*y

theorem equation_equivalence :
  IsHomogeneous OriginalEq →
  (∀ x y : ℝ, OriginalEq x y = 0 ↔ EquivalentEq x y) :=
sorry

end equation_equivalence_l3623_362344


namespace amount_added_l3623_362304

theorem amount_added (N A : ℝ) : 
  N = 1.375 → 
  0.6667 * N + A = 1.6667 → 
  A = 0.750025 := by
sorry

end amount_added_l3623_362304


namespace solution_of_inequality_1_solution_of_inequality_2_l3623_362341

-- Define the solution sets
def solution_set_1 : Set ℝ := {x | -5/2 < x ∧ x < 3}
def solution_set_2 : Set ℝ := {x | x < -2/3 ∨ x > 0}

-- Theorem for the first inequality
theorem solution_of_inequality_1 :
  {x : ℝ | 2*x^2 - x - 15 < 0} = solution_set_1 :=
by sorry

-- Theorem for the second inequality
theorem solution_of_inequality_2 :
  {x : ℝ | 2/x > -3} = solution_set_2 :=
by sorry

end solution_of_inequality_1_solution_of_inequality_2_l3623_362341


namespace triangle_side_length_l3623_362392

theorem triangle_side_length (A B C : ℝ) (b c : ℝ) :
  A = π / 3 →
  b = 16 →
  (1 / 2) * b * c * Real.sin A = 64 * Real.sqrt 3 →
  c = 16 := by
sorry

end triangle_side_length_l3623_362392


namespace dogwood_trees_after_five_years_l3623_362301

/-- Calculates the expected number of dogwood trees in the park after a given number of years -/
def expected_trees (initial_trees : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) 
                   (growth_rate_today : ℕ) (growth_rate_tomorrow : ℕ) (years : ℕ) : ℕ :=
  initial_trees + planted_today + planted_tomorrow + 
  (planted_today * growth_rate_today * years) + 
  (planted_tomorrow * growth_rate_tomorrow * years)

/-- Theorem stating the expected number of dogwood trees after 5 years -/
theorem dogwood_trees_after_five_years :
  expected_trees 39 41 20 2 4 5 = 130 := by
  sorry

#eval expected_trees 39 41 20 2 4 5

end dogwood_trees_after_five_years_l3623_362301


namespace geometric_series_equality_l3623_362366

theorem geometric_series_equality (n : ℕ) : n ≥ 1 → (
  let C : ℕ → ℚ := λ k => 1320 * (1 - 1 / 3^k)
  let D : ℕ → ℚ := λ k => 1008 * (1 - 1 / (-3)^k)
  (∃ k ≥ 1, C k = D k) ∧ (∀ m ≥ 1, m < n → C m ≠ D m) → n = 2
) := by sorry

end geometric_series_equality_l3623_362366


namespace smallest_integer_quadratic_inequality_l3623_362329

theorem smallest_integer_quadratic_inequality :
  ∃ (n : ℤ), (∀ (m : ℤ), m^2 - 9*m + 18 ≥ 0 → n ≤ m) ∧ (n^2 - 9*n + 18 ≥ 0) ∧ n = 3 := by
  sorry

end smallest_integer_quadratic_inequality_l3623_362329


namespace team_formation_count_l3623_362397

def male_doctors : ℕ := 5
def female_doctors : ℕ := 4
def team_size : ℕ := 3

def team_formations : ℕ := 
  (Nat.choose male_doctors 2 * Nat.choose female_doctors 1) + 
  (Nat.choose male_doctors 1 * Nat.choose female_doctors 2)

theorem team_formation_count : team_formations = 70 := by
  sorry

end team_formation_count_l3623_362397


namespace min_value_of_linear_combination_l3623_362388

theorem min_value_of_linear_combination (x y : ℝ) : 
  3 * x^2 + 3 * y^2 = 20 * x + 10 * y + 10 → 
  5 * x + 6 * y ≥ 122 := by
sorry

end min_value_of_linear_combination_l3623_362388


namespace union_complement_problem_l3623_362321

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem union_complement_problem : A ∪ (U \ B) = {0, 1, 2, 3} := by sorry

end union_complement_problem_l3623_362321


namespace cos_five_pi_fourth_plus_x_l3623_362370

theorem cos_five_pi_fourth_plus_x (x : ℝ) (h : Real.sin (π / 4 - x) = -1 / 5) :
  Real.cos (5 * π / 4 + x) = 1 / 5 := by sorry

end cos_five_pi_fourth_plus_x_l3623_362370


namespace only_C_not_like_terms_l3623_362394

-- Define a structure for a term
structure Term where
  coefficient : ℚ
  x_exponent : ℕ
  y_exponent : ℕ
  m_exponent : ℕ
  n_exponent : ℕ
  deriving Repr

-- Define a function to check if two terms are like terms
def are_like_terms (t1 t2 : Term) : Prop :=
  t1.x_exponent = t2.x_exponent ∧
  t1.y_exponent = t2.y_exponent ∧
  t1.m_exponent = t2.m_exponent ∧
  t1.n_exponent = t2.n_exponent

-- Define the terms from the problem
def term_A1 : Term := ⟨-1, 2, 1, 0, 0⟩  -- -x²y
def term_A2 : Term := ⟨2, 2, 1, 0, 0⟩   -- 2yx²
def term_B1 : Term := ⟨2, 0, 0, 0, 0⟩   -- 2πR (treating π and R as constants)
def term_B2 : Term := ⟨1, 0, 0, 0, 0⟩   -- π²R (treating π and R as constants)
def term_C1 : Term := ⟨-1, 0, 0, 2, 1⟩  -- -m²n
def term_C2 : Term := ⟨1/2, 0, 0, 1, 2⟩ -- 1/2mn²
def term_D1 : Term := ⟨1, 0, 0, 0, 0⟩   -- 2³ (8)
def term_D2 : Term := ⟨1, 0, 0, 0, 0⟩   -- 3² (9)

-- Theorem stating that only pair C contains terms that are not like terms
theorem only_C_not_like_terms :
  are_like_terms term_A1 term_A2 ∧
  are_like_terms term_B1 term_B2 ∧
  ¬(are_like_terms term_C1 term_C2) ∧
  are_like_terms term_D1 term_D2 :=
sorry

end only_C_not_like_terms_l3623_362394


namespace x_twenty_percent_greater_than_98_l3623_362383

theorem x_twenty_percent_greater_than_98 (x : ℝ) :
  x = 98 * (1 + 20 / 100) → x = 117.6 := by
  sorry

end x_twenty_percent_greater_than_98_l3623_362383


namespace game_playing_time_l3623_362336

theorem game_playing_time (num_children : ℕ) (game_duration : ℕ) (players_at_once : ℕ) :
  num_children = 8 →
  game_duration = 120 →  -- 2 hours in minutes
  players_at_once = 2 →
  (game_duration * players_at_once) % num_children = 0 →
  (game_duration * players_at_once) / num_children = 30 :=
by sorry

end game_playing_time_l3623_362336
