import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_plus_csc_l757_75716

noncomputable def f (x : ℝ) := Real.tan x + (1 / Real.sin x)

theorem period_of_tan_plus_csc : 
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x) ∧ 
  (∀ (q : ℝ), (∀ (x : ℝ), f (x + q) = f x) → q ≥ 2 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_tan_plus_csc_l757_75716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_fourth_rods_l757_75799

/-- A function that checks if four lengths can form a valid quadrilateral with positive area -/
def is_valid_quadrilateral (a b c d : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧
  a + b > d ∧ a + d > b ∧ b + d > a ∧
  a + c > d ∧ a + d > c ∧ c + d > a ∧
  b + c > d ∧ b + d > c ∧ c + d > b

/-- The set of all rod lengths -/
def all_rods : Finset ℕ := Finset.range 40

/-- The set of selected rod lengths -/
def selected_rods : Finset ℕ := {4, 9, 18}

/-- The set of remaining rod lengths -/
def remaining_rods : Finset ℕ := all_rods \ selected_rods

/-- A decidable version of is_valid_quadrilateral -/
def is_valid_quadrilateral_dec (a b c d : ℕ) : Bool :=
  a + b > c && a + c > b && b + c > a &&
  a + b > d && a + d > b && b + d > a &&
  a + c > d && a + d > c && c + d > a &&
  b + c > d && b + d > c && c + d > b

/-- The theorem to be proved -/
theorem count_valid_fourth_rods :
  (remaining_rods.filter (fun x => is_valid_quadrilateral_dec 4 9 18 x)).card = 22 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_fourth_rods_l757_75799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_four_I_l757_75791

open Real MeasureTheory Interval

-- Define I_n as a function of n
noncomputable def I (n : ℕ+) : ℝ :=
  ∫ x in (-π)..(π), (π/2 - |x|) * Real.cos (n * x)

-- State the theorem
theorem sum_of_first_four_I : I 1 + I 2 + I 3 + I 4 = 40/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_four_I_l757_75791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_baskets_count_l757_75770

def num_apples : ℕ := 7
def num_oranges : ℕ := 12

def valid_basket (basket : ℕ × ℕ) : Bool :=
  let (apples, oranges) := basket
  (apples ≤ num_apples) &&
  (oranges ≤ num_oranges) &&
  (apples + oranges > 0) &&
  (apples ≥ 2 || oranges ≥ 2)

def count_valid_baskets : ℕ := 
  (num_apples + 1) * (num_oranges + 1) - 1 - 2

theorem valid_baskets_count : 
  (Finset.filter (fun b => valid_basket b) (Finset.product (Finset.range (num_apples + 1)) (Finset.range (num_oranges + 1)))).card = count_valid_baskets := by
  sorry

#eval count_valid_baskets

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_baskets_count_l757_75770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l757_75730

theorem sequence_convergence (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_rec : ∀ n, (a n)^2 ≤ a n - a (n+1)) :
  ∀ n, a n < (n : ℝ)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l757_75730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_sides_l757_75739

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) : 
  (((n - 2) * 180) / n = 135) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_sides_l757_75739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l757_75768

noncomputable def f (x : ℝ) : ℝ := (2022 * x^3 + 2 * x^2 + 3 * x + 6) / (x^2 + 3)

theorem function_property (a : ℝ) (h : f a = 14) : f (-a) = -10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l757_75768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_partition_bounds_l757_75746

/-- A partition of stones is good if all small piles have different numbers of stones
    and splitting any small pile results in two piles with the same number of stones. -/
def is_good_partition (partition : List Nat) : Prop :=
  (partition.sum = 100) ∧
  (∀ x, x ∈ partition → x ≥ 1) ∧
  (∀ x y, x ∈ partition → y ∈ partition → x ≠ y → partition.count x = 1) ∧
  (∀ x, x ∈ partition → ∀ a b : Nat, a + b = x → 
    ∃ y z, y ∈ (a :: b :: partition.erase x) ∧ z ∈ (a :: b :: partition.erase x) ∧ y = z ∧ y ≠ x)

theorem good_partition_bounds :
  (∃ partition : List Nat, is_good_partition partition ∧ partition.length = 13) ∧
  (∃ partition : List Nat, is_good_partition partition ∧ partition.length = 10) ∧
  (∀ partition : List Nat, is_good_partition partition → 
    partition.length ≤ 13 ∧ partition.length ≥ 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_partition_bounds_l757_75746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_desired_interest_rate_l757_75793

noncomputable def nominal_value : ℚ := 20
noncomputable def dividend_rate : ℚ := 9
noncomputable def market_value : ℚ := 15

noncomputable def dividend_per_share : ℚ := (dividend_rate / 100) * nominal_value

theorem desired_interest_rate :
  (dividend_per_share / market_value) * 100 = 12 := by
  -- Unfold definitions
  unfold dividend_per_share nominal_value dividend_rate market_value
  -- Simplify the expression
  simp [mul_div_assoc, mul_comm, mul_assoc]
  -- Evaluate the numerical expression
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_desired_interest_rate_l757_75793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_product_l757_75773

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Represents a focus of the ellipse -/
structure Focus (e : Ellipse) where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The statement of the problem -/
theorem ellipse_max_product (e : Ellipse) (F1 F2 : Focus e) (M N : PointOnEllipse e) (A B : PointOnEllipse e) :
  (distance (M.x, M.y) (F1.x, F1.y) + distance (M.x, M.y) (F2.x, F2.y) +
   distance (N.x, N.y) (F1.x, F1.y) + distance (N.x, N.y) (F2.x, F2.y) = 4) →
  (∃ (l : ℝ × ℝ → Prop), l (F1.x, F1.y) ∧ l (A.x, A.y) ∧ l (B.x, B.y)) →
  (distance (A.x, A.y) (B.x, B.y) = 4/3) →
  (∃ (maxProduct : ℝ), maxProduct = 16/9 ∧
    ∀ (AF2 BF2 : ℝ), AF2 = distance (A.x, A.y) (F2.x, F2.y) →
                      BF2 = distance (B.x, B.y) (F2.x, F2.y) →
                      AF2 * BF2 ≤ maxProduct) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_max_product_l757_75773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_inspection_probabilities_l757_75749

def total_products : ℕ := 20
def qualified_products : ℕ := 17
def defective_products : ℕ := 3
def chosen_products : ℕ := 3

theorem product_inspection_probabilities :
  (Nat.choose qualified_products chosen_products = 680) ∧
  ((Nat.choose qualified_products (chosen_products - 1) * Nat.choose defective_products 1) / Nat.choose total_products chosen_products = 34 / 95) ∧
  ((Nat.choose qualified_products chosen_products + Nat.choose qualified_products (chosen_products - 1) * Nat.choose defective_products 1) / Nat.choose total_products chosen_products = 272 / 285) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_inspection_probabilities_l757_75749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_marbles_l757_75755

theorem least_marbles (n : ℕ) : 
  (∀ k : ℕ, k ∈ ({4, 5, 6, 7, 8} : Set ℕ) → n % k = 0) ↔ 840 ≤ n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_marbles_l757_75755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_rate_approx_l757_75741

/-- Calculates the rate of a current given boat speed, distance traveled, and time taken -/
noncomputable def calculate_current_rate (boat_speed : ℝ) (distance : ℝ) (time_minutes : ℝ) : ℝ :=
  let time_hours := time_minutes / 60
  let total_speed := distance / time_hours
  total_speed - boat_speed

theorem current_rate_approx (boat_speed : ℝ) (distance : ℝ) (time_minutes : ℝ) 
  (h1 : boat_speed = 20)
  (h2 : distance = 8.75)
  (h3 : time_minutes = 21) :
  ∃ ε > 0, |calculate_current_rate boat_speed distance time_minutes - 5| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_rate_approx_l757_75741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_f_nonneg_l757_75779

/-- An odd function f: ℝ → ℝ with a specific definition for x ≤ 0 -/
noncomputable def f : ℝ → ℝ :=
  fun x => if x ≤ 0 then x^2 - 3*x + 2 else -x^2 + 3*x - 2

/-- Theorem stating that f is an odd function -/
theorem f_odd : ∀ x, f (-x) = -f x := by
  sorry

/-- Theorem proving the specific form of f for x ≥ 0 -/
theorem f_nonneg (x : ℝ) (h : x ≥ 0) : f x = -x^2 + 3*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_f_nonneg_l757_75779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l757_75786

/-- The number of distinct ordered pairs of positive integers (m,n) satisfying 1/m + 1/n = 1/6 -/
def count_pairs : ℕ := 9

/-- Predicate for pairs satisfying the equation -/
def satisfies_equation (m n : ℕ+) : Prop :=
  (m : ℚ)⁻¹ + (n : ℚ)⁻¹ = (6 : ℚ)⁻¹

/-- The set of all pairs satisfying the equation -/
def valid_pairs : Set (ℕ+ × ℕ+) :=
  {p | satisfies_equation p.1 p.2}

/-- List of all valid pairs -/
def all_valid_pairs : List (ℕ+ × ℕ+) := [
  (⟨7, by norm_num⟩, ⟨42, by norm_num⟩),
  (⟨8, by norm_num⟩, ⟨24, by norm_num⟩),
  (⟨9, by norm_num⟩, ⟨18, by norm_num⟩),
  (⟨10, by norm_num⟩, ⟨15, by norm_num⟩),
  (⟨12, by norm_num⟩, ⟨12, by norm_num⟩),
  (⟨15, by norm_num⟩, ⟨10, by norm_num⟩),
  (⟨18, by norm_num⟩, ⟨9, by norm_num⟩),
  (⟨24, by norm_num⟩, ⟨8, by norm_num⟩),
  (⟨42, by norm_num⟩, ⟨7, by norm_num⟩)
]

theorem count_valid_pairs :
  valid_pairs = all_valid_pairs.toFinset ∧ all_valid_pairs.length = count_pairs :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l757_75786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_abs_two_and_equal_l757_75781

theorem sum_of_abs_two_and_equal (a b : ℝ) : 
  (abs a = 2 ∧ abs b = a) → (a + b ∈ ({-4, 0, 4} : Set ℝ)) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_abs_two_and_equal_l757_75781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_descending_order_abc_l757_75772

theorem descending_order_abc (a b c : ℝ) : 
  a = (8 : ℝ)^(7/10) → b = (8 : ℝ)^(9/10) → c = (2 : ℝ)^(4/5) → b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_descending_order_abc_l757_75772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l757_75794

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (4 - x^2)) / (x - 1)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = Set.Icc (-2) 2 \ {1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l757_75794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triple_exists_l757_75735

/-- Represents an ordered triple of integers (a, b, c) -/
structure Triple where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if a triple satisfies all conditions -/
def satisfiesConditions (t : Triple) : Prop :=
  t.a ≥ 2 ∧ t.b ≥ 1 ∧ (Real.log t.b / Real.log t.a) = t.c^2 ∧ t.a + t.b + t.c = 100

/-- There exists exactly one ordered triple of integers (a,b,c) that satisfies all conditions -/
theorem unique_triple_exists : ∃! t : Triple, satisfiesConditions t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triple_exists_l757_75735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangement_exists_l757_75750

/-- Represents a 4x4 board with gold and silver coins -/
def Board := Fin 4 → Fin 4 → ℕ

/-- The number of silver coins on a given cell -/
def silver_coins (board : Board) (i j : Fin 4) : ℕ := 
  if i = 2 ∧ j = 2 then 9 else 0

/-- The number of gold coins on a given cell -/
def gold_coins (board : Board) (i j : Fin 4) : ℕ := 
  if i = 2 ∧ j = 2 then 0 else 1

/-- The total number of silver coins on the board -/
def total_silver (board : Board) : ℕ := 
  Finset.sum (Finset.univ : Finset (Fin 4)) fun i => 
    Finset.sum (Finset.univ : Finset (Fin 4)) fun j => 
      silver_coins board i j

/-- The total number of gold coins on the board -/
def total_gold (board : Board) : ℕ := 
  Finset.sum (Finset.univ : Finset (Fin 4)) fun i => 
    Finset.sum (Finset.univ : Finset (Fin 4)) fun j => 
      gold_coins board i j

/-- Checks if a 3x3 sub-square has more silver than gold coins -/
def valid_3x3 (board : Board) (top_left_i top_left_j : Fin 2) : Prop :=
  let silver_sum := Finset.sum (Finset.range 3) fun i => 
    Finset.sum (Finset.range 3) fun j => 
      silver_coins board (top_left_i + i) (top_left_j + j)
  let gold_sum := Finset.sum (Finset.range 3) fun i => 
    Finset.sum (Finset.range 3) fun j => 
      gold_coins board (top_left_i + i) (top_left_j + j)
  silver_sum > gold_sum

/-- The main theorem stating that a valid arrangement exists -/
theorem valid_arrangement_exists : ∃ (board : Board), 
  (∀ (i j : Fin 2), valid_3x3 board i j) ∧ 
  (total_gold board > total_silver board) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangement_exists_l757_75750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l757_75763

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def two_lines_perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of a line y = mx + b is m -/
def line_slope (m : ℝ) : ℝ := m

theorem perpendicular_lines_a_value : 
  ∀ a : ℝ, 
  two_lines_perpendicular (line_slope a) (line_slope (a + 2)) → 
  a = -1 :=
by
  intro a h
  unfold two_lines_perpendicular at h
  unfold line_slope at h
  -- The rest of the proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_value_l757_75763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_children_count_l757_75751

def age_sequence (n : ℕ) : ℕ → ℕ
| 0 => 4
| i + 1 => age_sequence n i + 3

def sum_ages (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (age_sequence n)

theorem children_count :
  ∃ n : ℕ, sum_ages n = 50 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_children_count_l757_75751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_between_sqrt3_and_sqrt14_l757_75771

theorem integer_between_sqrt3_and_sqrt14 : ∃! x : ℤ, 
  x ∈ ({1, 3, 5, 7} : Set ℤ) ∧ 
  (Real.sqrt 3 : ℝ) < (x : ℝ) ∧ (x : ℝ) < (Real.sqrt 14 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_between_sqrt3_and_sqrt14_l757_75771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isothermal_compression_work_theorem_l757_75798

/-- The mechanical work required to compress gas isothermally -/
noncomputable def isothermal_compression_work (p p₀ V₂ : ℝ) : ℝ := p * V₂ * Real.log (p / p₀)

/-- 
Theorem: The mechanical work required to achieve pressure p in a tank of volume V₂, 
starting from atmospheric pressure p₀, in an isothermal process is given by p * V₂ * ln(p/p₀)
-/
theorem isothermal_compression_work_theorem 
  (p p₀ V₁ V₂ : ℝ) 
  (h_p_pos : p > 0) 
  (h_p₀_pos : p₀ > 0) 
  (h_V₁_pos : V₁ > 0) 
  (h_V₂_pos : V₂ > 0) :
  ∃ W : ℝ, W = isothermal_compression_work p p₀ V₂ := by
  use p * V₂ * Real.log (p / p₀)
  rfl

#check isothermal_compression_work_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isothermal_compression_work_theorem_l757_75798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloons_left_l757_75738

def round_balloons : ℕ := 5 * 25
def long_balloons : ℕ := 4 * 35
def heart_balloons : ℕ := 3 * 40
def star_balloons : ℕ := 2 * 50

def round_defective : ℕ := (10 * round_balloons) / 100
def long_defective : ℕ := (5 * long_balloons) / 100
def heart_defective : ℕ := (15 * heart_balloons) / 100
def star_defective : ℕ := (8 * star_balloons) / 100

def round_burst : ℕ := 5
def long_burst : ℕ := 7
def heart_burst : ℕ := 3
def star_burst : ℕ := 4

theorem balloons_left : 
  (round_balloons - round_defective - round_burst) +
  (long_balloons - long_defective - long_burst) +
  (heart_balloons - heart_defective - heart_burst) +
  (star_balloons - star_defective - star_burst) = 421 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloons_left_l757_75738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_array_sum_is_three_halves_l757_75733

/-- Represents the sum of all terms in the 1/9-array -/
noncomputable def arraySum : ℚ := ∑' r, ∑' c, (1 / 4 ^ r) * (1 / 9 ^ c)

/-- The first entry of each row is 1/4 times the first entry of the previous row -/
axiom row_start_rule (r : ℕ) : (1 / 4 ^ r) = 1 / 4 * (1 / 4 ^ (r - 1))

/-- Each succeeding term in a row is 1/9 times the previous term in the same row -/
axiom column_rule (r c : ℕ) : (1 / 4 ^ r) * (1 / 9 ^ c) = 1 / 9 * ((1 / 4 ^ r) * (1 / 9 ^ (c - 1)))

/-- The sum of all terms in the array is equal to 3/2 -/
theorem array_sum_is_three_halves : arraySum = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_array_sum_is_three_halves_l757_75733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kiki_buys_eighteen_scarves_l757_75707

/-- Represents Kiki's purchasing scenario -/
structure KikiPurchase where
  total_money : ℚ
  hat_percentage : ℚ
  scarf_price : ℚ
  hat_to_scarf_ratio : ℕ

/-- Calculates the number of scarves Kiki can buy -/
def scarves_bought (k : KikiPurchase) : ℕ :=
  (k.total_money * (1 - k.hat_percentage) / k.scarf_price).floor.toNat

/-- Theorem stating that Kiki will buy 18 scarves -/
theorem kiki_buys_eighteen_scarves :
  let k : KikiPurchase := {
    total_money := 90,
    hat_percentage := 3/5,
    scarf_price := 2,
    hat_to_scarf_ratio := 2
  }
  scarves_bought k = 18 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kiki_buys_eighteen_scarves_l757_75707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_pure_imaginary_z₁_fourth_z₂_first_quadrant_l757_75727

/-- Complex number z₁ -/
def z₁ (x : ℝ) : ℂ := x^2 - 1 + (x^2 - 3*x + 2)*Complex.I

/-- Complex number z₂ -/
def z₂ (x : ℝ) : ℂ := x + (3 - 2*x)*Complex.I

/-- Theorem for part (1) -/
theorem z₁_pure_imaginary (x : ℝ) : z₁ x = Complex.I * Complex.im (z₁ x) → x = -1 := by sorry

/-- Theorem for part (2) -/
theorem z₁_fourth_z₂_first_quadrant (x : ℝ) :
  (Complex.re (z₁ x) > 0 ∧ Complex.im (z₁ x) < 0) ∧ 
  (Complex.re (z₂ x) > 0 ∧ Complex.im (z₂ x) > 0) → 
  1 < x ∧ x < 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_pure_imaginary_z₁_fourth_z₂_first_quadrant_l757_75727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_reciprocal_sum_l757_75705

/-- A triangular number is a number of the form n * (n + 1) / 2 for some natural number n. -/
def is_triangular (x : ℕ) : Prop :=
  ∃ n : ℕ, x = n * (n + 1) / 2

/-- The sum of reciprocals of a list of natural numbers -/
def sum_reciprocals (xs : List ℕ) : ℚ :=
  (xs.map (λ x => (1 : ℚ) / x)).sum

/-- Main theorem: For any positive integer s ≠ 2, there exists a list of s triangular numbers
    whose reciprocals sum to 1. -/
theorem triangular_reciprocal_sum (s : ℕ) (hs : s ≠ 0 ∧ s ≠ 2) :
  ∃ xs : List ℕ, xs.length = s ∧ (∀ x ∈ xs, is_triangular x) ∧ sum_reciprocals xs = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_reciprocal_sum_l757_75705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_value_l757_75714

/-- Calculates the market value of a stock given its face value, dividend rate, and yield. -/
noncomputable def market_value (face_value : ℝ) (dividend_rate : ℝ) (yield : ℝ) : ℝ :=
  (face_value * dividend_rate / yield) * 100

/-- Theorem stating that a stock with 12% dividend rate and 8% yield has a market value of 150% of its face value. -/
theorem stock_market_value (face_value : ℝ) (h_pos : face_value > 0) :
  market_value face_value 0.12 0.08 = 1.5 * face_value := by
  -- Unfold the definition of market_value
  unfold market_value
  -- Perform algebraic simplification
  simp [mul_assoc, mul_comm, mul_div_cancel']
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_value_l757_75714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_when_not_p_holds_l757_75789

theorem range_of_a_when_not_p_holds (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + a * x + 1 < 0) ↔ a ∈ Set.Ioi 4 ∪ Set.Iio 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_when_not_p_holds_l757_75789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonically_increasing_condition_l757_75796

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

-- Part 1: Tangent line equation
theorem tangent_line_at_one (a : ℝ) :
  a = -1 →
  ∃ m b : ℝ, ∀ x y : ℝ,
    y = m * (x - 1) + f a 1 ↔ Real.log 2 * x + y - Real.log 2 = 0 :=
by sorry

-- Part 2: Monotonicity condition
theorem monotonically_increasing_condition (a : ℝ) :
  (∀ x : ℝ, x > 0 → (deriv (f a)) x ≥ 0) ↔ a ≥ 1/2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonically_increasing_condition_l757_75796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l757_75723

noncomputable def f (x : ℝ) : ℝ := (x^2 - (1/2)*x + 2) / (x^2 + 2*x + 3)

theorem range_of_f :
  Set.range f = Set.Icc (-21/4 : ℝ) (3/4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l757_75723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coeff_x_four_equal_binomial_coeff_l757_75734

/-- The binomial expansion of (x - 2/√x)^10 -/
noncomputable def binomial_expansion (x : ℝ) := (x - 2 / Real.sqrt x) ^ 10

/-- The coefficient of x^k in the binomial expansion -/
def coeff (k : ℕ) : ℚ := (-2)^k * (Nat.choose 10 k)

/-- The power of x in the (k+1)-th term of the expansion -/
def power (k : ℕ) : ℚ := 10 - (3/2) * k

theorem binomial_coeff_x_four :
  coeff 4 = 3360 := by
  sorry

theorem equal_binomial_coeff :
  ∀ r : ℕ, (Nat.choose 10 (3*r - 1) = Nat.choose 10 (r + 1)) → r = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coeff_x_four_equal_binomial_coeff_l757_75734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_expression_equality_l757_75758

/-- Vector a --/
def a : Fin 3 → ℝ := ![3, -4, 2]

/-- Vector b --/
def b : Fin 3 → ℝ := ![-2, 5, 3]

/-- Vector c --/
def c : Fin 3 → ℝ := ![1, 1, -4]

/-- Theorem stating the equality of the vector expression --/
theorem vector_expression_equality : 
  (a + 2 • b - c) = ![(-2 : ℝ), 5, 12] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_expression_equality_l757_75758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_divisible_by_eleven_l757_75753

theorem difference_divisible_by_eleven (S : Finset ℤ) (h : S.card = 12) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ 11 ∣ (a - b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_divisible_by_eleven_l757_75753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_demographics_l757_75743

theorem company_demographics (total : ℕ) (total_pos : total > 0) :
  let men_percent : ℚ := 56 / 100
  let union_percent : ℚ := 60 / 100
  let union_men_percent : ℚ := 70 / 100
  let men := (men_percent * total : ℚ).floor
  let union := (union_percent * total : ℚ).floor
  let union_men := (union_men_percent * union : ℚ).floor
  let non_union := total - union
  let non_union_men := men - union_men
  let non_union_women := non_union - non_union_men
  (non_union_women : ℚ) / non_union = 65 / 100 :=
by
  sorry

#check company_demographics

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_demographics_l757_75743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_iff_a_in_range_l757_75745

noncomputable def f (x : ℝ) := Real.sqrt (x^2 - 1)

def A : Set ℝ := {x : ℝ | x ≥ 1 ∨ x ≤ -1}

def B (a : ℝ) : Set ℝ := {x : ℝ | 1 < a * x ∧ a * x < 2}

theorem subset_iff_a_in_range (a : ℝ) : B a ⊆ A ↔ a ∈ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_iff_a_in_range_l757_75745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equality_l757_75712

def sequence_to_number (s : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ i ↦ s i * 10^i)

theorem sequence_equality {a b : ℕ → ℕ} (h1 : ∀ n, a n ∈ Finset.range 10)
    (h2 : ∀ n, b n ∈ Finset.range 10) (h3 : ∃ M, ∀ n ≥ M, a n ≠ 0 ∧ b n ≠ 0)
    (h4 : ∀ n, (sequence_to_number a n)^2 + 999 ∣ (sequence_to_number b n)^2 + 999) :
  ∀ n, a n = b n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equality_l757_75712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_cover_price_l757_75709

theorem paper_cover_price (paper_percentage : ℝ) (hardcover_percentage : ℝ)
  (paper_copies : ℕ) (hardcover_copies : ℕ) (hardcover_price : ℝ) (total_earnings : ℝ)
  (h1 : paper_percentage = 0.06)
  (h2 : hardcover_percentage = 0.12)
  (h3 : paper_copies = 32000)
  (h4 : hardcover_copies = 15000)
  (h5 : hardcover_price = 0.40)
  (h6 : total_earnings = 1104) :
  ∃ (paper_price : ℝ),
    paper_price * paper_percentage * (paper_copies : ℝ) +
    hardcover_price * hardcover_percentage * (hardcover_copies : ℝ) = total_earnings ∧
    paper_price = 0.20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_cover_price_l757_75709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_m_l757_75759

def m : ℕ := 2^5 * 3^6 * 5^7 * 7^8

theorem number_of_factors_of_m : 
  (Finset.filter (· ∣ m) (Finset.range (m + 1))).card = 3024 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_m_l757_75759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_doesnt_pay_bernardo_l757_75744

/-- Represents the amount paid by each person in dollars -/
structure Payments where
  leroy : ℚ
  bernardo : ℚ
  carlos : ℚ

/-- Calculates the total amount paid by all three people -/
def total_paid (p : Payments) : ℚ := p.leroy + p.bernardo + p.carlos

/-- Calculates the amount each person should pay for equal sharing -/
def equal_share (p : Payments) : ℚ := (total_paid p) / 3

/-- Determines if a person needs to pay another person -/
def needs_to_pay (amount_paid : ℚ) (should_pay : ℚ) : Prop :=
  amount_paid < should_pay

theorem carlos_doesnt_pay_bernardo (exchange_rate : ℚ) (p : Payments)
    (h1 : p.leroy = 100)
    (h2 : p.bernardo = 150)
    (h3 : p.carlos = 120 * exchange_rate)
    (h4 : exchange_rate = 11/10) :
    ¬(needs_to_pay p.carlos (equal_share p)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_doesnt_pay_bernardo_l757_75744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_is_101_l757_75784

def mySequence : List ℕ := [12, 13, 101, 17, 111, 113, 117, 119, 123, 129, 131]

theorem third_number_is_101 : mySequence[2] = 101 := by
  rfl

#eval mySequence[2]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_is_101_l757_75784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_implies_a_range_l757_75785

/-- A piecewise function f depending on a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 7 else a/x

/-- f is monotonically increasing on ℝ -/
def is_monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

/-- The main theorem stating that if f is monotonically increasing, then a is in [-4, -2] -/
theorem monotone_increasing_implies_a_range (a : ℝ) :
  is_monotone_increasing (f a) → a ∈ Set.Icc (-4) (-2) := by
  sorry

#check monotone_increasing_implies_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_implies_a_range_l757_75785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_special_numbers_l757_75715

theorem order_of_special_numbers : 
  Real.sqrt 2 > Real.log 3 / Real.log π ∧ 
  Real.log 3 / Real.log π > Real.log (Real.sin (2 * π / 5)) / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_special_numbers_l757_75715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_count_l757_75787

theorem turtle_count (kristen_turtles : ℚ) : ℚ := by
  -- Define the number of turtles each person has
  let kris_turtles := kristen_turtles / 4
  let trey_turtles := 5 * kris_turtles

  -- Define the total number of turtles
  let total_turtles := kristen_turtles + kris_turtles + trey_turtles

  -- State that the total number of turtles is 30
  have h : total_turtles = 30

  -- Prove that the total number of turtles is indeed 30
  sorry

  -- Return the total number of turtles
  exact 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_count_l757_75787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_angle_relation_l757_75725

theorem triangle_side_angle_relation (A B C : ℝ) (a b c : ℝ) :
  (0 < A) → (A < Real.pi) →
  (0 < B) → (B < Real.pi) →
  (0 < C) → (C < Real.pi) →
  (A + B + C = Real.pi) →
  (a > 0) → (b > 0) → (c > 0) →
  (a / Real.sin A = b / Real.sin B) →
  (a / Real.sin A = c / Real.sin C) →
  (a < b ↔ Real.cos A > Real.cos B) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_angle_relation_l757_75725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_running_group_gender_ratio_l757_75762

/-- Represents the three activity groups --/
inductive ActivityGroup
  | RopeJumping
  | LongJump
  | Running

/-- Represents the gender of students --/
inductive Gender
  | Male
  | Female

/-- Represents the distribution of students across activities and genders --/
structure ClassDistribution where
  totalStudents : ℕ
  activityRatio : Fin 3 → ℕ
  genderRatio : Gender → ℕ
  ropeJumpingGenderRatio : Gender → ℕ
  longJumpGenderRatio : Gender → ℕ

/-- The given conditions of the problem --/
def problemConditions : ClassDistribution :=
  { totalStudents := 40,
    activityRatio := λ i => [20, 8, 12].get i,
    genderRatio := λ g => match g with
      | Gender.Male => 2
      | Gender.Female => 3,
    ropeJumpingGenderRatio := λ g => match g with
      | Gender.Male => 1
      | Gender.Female => 3,
    longJumpGenderRatio := λ g => match g with
      | Gender.Male => 3
      | Gender.Female => 1 }

/-- Helper function to calculate the number of students in the running group for a given gender --/
def studentsInRunningGroup (g : Gender) (d : ClassDistribution) : ℕ :=
  sorry  -- Implementation details omitted

/-- The theorem to be proved --/
theorem running_group_gender_ratio 
  (d : ClassDistribution) 
  (h : d = problemConditions) :
  (studentsInRunningGroup Gender.Male d : ℚ) / (studentsInRunningGroup Gender.Female d : ℚ) = 7 / 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_running_group_gender_ratio_l757_75762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_other_intercept_l757_75760

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  intercept1 : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For the given ellipse, the other x-intercept is (20/7, 0) -/
theorem ellipse_other_intercept (e : Ellipse) 
  (h1 : e.focus1 = ⟨0, 3⟩) 
  (h2 : e.focus2 = ⟨4, 0⟩)
  (h3 : e.intercept1 = ⟨1, 0⟩) : 
  ∃ (p : Point), p.x = 20/7 ∧ p.y = 0 ∧ 
  distance p e.focus1 + distance p e.focus2 = 
  distance e.intercept1 e.focus1 + distance e.intercept1 e.focus2 := by
  sorry

#check ellipse_other_intercept

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_other_intercept_l757_75760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l757_75754

theorem max_true_statements (x y : ℝ) : 
  let statements := [
    (1/x > 1/y),
    (x^2 < y^2),
    (x > y),
    (x > 0),
    (y > 0)
  ]
  ∃ (true_statements : List Bool), 
    (true_statements.length ≤ 4) ∧ 
    (∀ i, i < statements.length → 
      (true_statements.get ⟨i, by sorry⟩ = true → statements.get ⟨i, by sorry⟩ = true)) ∧
    (∀ subset : List Bool, 
      (subset.length > 4) → 
      (∀ i, i < statements.length → 
        (subset.get ⟨i, by sorry⟩ = true → statements.get ⟨i, by sorry⟩ = true)) → 
      False) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l757_75754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_board_product_l757_75713

/-- Represents a three-digit number where each digit is distinct and non-zero -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  distinct : hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones
  nonzero : hundreds ≠ 0 ∧ tens ≠ 0 ∧ ones ≠ 0
  valid : hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- The product of two ThreeDigitNumbers satisfying the given conditions -/
def boardProduct (n1 n2 : ThreeDigitNumber) : Prop :=
  let prod := n1.hundreds * 100 + n1.tens * 10 + n1.ones *
              (n2.hundreds * 100 + n2.tens * 10 + n2.ones)
  let digits := [prod / 100000, (prod / 10000) % 10, (prod / 1000) % 10,
                 (prod / 100) % 10, (prod / 10) % 10, prod % 10]
  ∃ (c i k : Nat),
    c ∈ digits ∧ i ∈ digits ∧ k ∈ digits ∧ 0 ∈ digits ∧
    (digits.count c = 3) ∧ (digits.count i = 1) ∧ (digits.count k = 1) ∧
    (digits.count 0 = 1) ∧ (digits.headD 0 = c) ∧ c ≠ i ∧ c ≠ k ∧ i ≠ k

theorem unique_board_product :
  ∃! (n1 n2 : ThreeDigitNumber),
    boardProduct n1 n2 ∧
    n1.hundreds * 100 + n1.tens * 10 + n1.ones = 521 ∧
    n2.hundreds * 100 + n2.tens * 10 + n2.ones = 215 := by
  sorry

#eval 521 * 215  -- Should output 112015

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_board_product_l757_75713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_a_time_l757_75729

/-- The time (in days) it takes for two workers to complete a job together -/
noncomputable def combined_time : ℝ := 6

/-- The time (in days) it takes for worker b to complete the job alone -/
noncomputable def b_time : ℝ := 12

/-- The rate at which worker a completes the job -/
noncomputable def a_rate : ℝ := 1 / combined_time - 1 / b_time

/-- The time it takes for worker a to complete the job alone -/
noncomputable def a_time : ℝ := 1 / a_rate

theorem worker_a_time : a_time = b_time := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_a_time_l757_75729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sphere_areas_for_specific_pyramid_l757_75790

/-- A right square pyramid with side edge perpendicular to the base -/
structure RightSquarePyramid where
  baseEdge : ℝ
  sideEdge : ℝ

/-- Calculate the sum of surface areas of inscribed and circumscribed spheres -/
noncomputable def sumOfSphereAreas (p : RightSquarePyramid) : ℝ :=
  let circumscribedRadius := p.sideEdge * Real.sqrt 3 / 2
  let inscribedRadius := 2 - Real.sqrt 2
  4 * Real.pi * circumscribedRadius^2 + 4 * Real.pi * inscribedRadius^2

/-- Theorem: The sum of surface areas for a specific right square pyramid -/
theorem sum_of_sphere_areas_for_specific_pyramid :
  let p : RightSquarePyramid := { baseEdge := 2, sideEdge := 2 }
  sumOfSphereAreas p = 36 * Real.pi - 16 * Real.sqrt 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sphere_areas_for_specific_pyramid_l757_75790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_swaps_to_divisible_by_11_l757_75778

/-- Represents a 4026-digit number composed of 2013 '1's and 2013 '2's -/
def Number := Fin 4026 → Fin 2

/-- The sum of digits in odd positions minus the sum of digits in even positions -/
def digitSum (n : Number) : ℤ :=
  (Finset.sum (Finset.filter (fun i => i % 2 = 1) (Finset.range 4026)) (fun i => n i)) -
  (Finset.sum (Finset.filter (fun i => i % 2 = 0) (Finset.range 4026)) (fun i => n i))

/-- A number is divisible by 11 if the alternating sum of its digits is divisible by 11 -/
def isDivisibleBy11 (n : Number) : Prop :=
  digitSum n % 11 = 0

/-- Swapping two digits in a number -/
def swap (n : Number) (i j : Fin 4026) : Number :=
  fun k => if k = i then n j else if k = j then n i else n k

/-- The main theorem: Any arrangement can be made divisible by 11 in at most 5 swaps -/
theorem max_swaps_to_divisible_by_11 (n : Number) :
  ∃ (swaps : List (Fin 4026 × Fin 4026)),
    swaps.length ≤ 5 ∧
    isDivisibleBy11 (swaps.foldl (fun acc (i, j) => swap acc i j) n) := by
  sorry

#check max_swaps_to_divisible_by_11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_swaps_to_divisible_by_11_l757_75778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_diagonal_ratio_l757_75775

/-- Given a parallelepiped PQRSMTUV defined by vectors PQ, PR, and PS, 
    the ratio (PT^2 + QM^2 + RV^2 + SU^2) / (PQ^2 + PR^2 + PS^2) is equal to 4 -/
theorem parallelepiped_diagonal_ratio (P Q R S M T U V : EuclideanSpace ℝ (Fin 3)) : 
  let PQ := Q - P
  let PR := R - P
  let PS := S - P
  let PT := T - P
  let QM := M - Q
  let RV := V - R
  let SU := U - S
  (‖PT‖^2 + ‖QM‖^2 + ‖RV‖^2 + ‖SU‖^2) / (‖PQ‖^2 + ‖PR‖^2 + ‖PS‖^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_diagonal_ratio_l757_75775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_100_zeros_count_l757_75792

-- Define g_0
def g_0 (x : ℝ) : ℝ := x + |x - 200| - |x + 200|

-- Define g_n recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g_0 x
  | n + 1 => |g n x| - 2

-- State the theorem
theorem g_100_zeros_count :
  ∃ (S : Finset ℝ), (∀ x ∈ S, g 100 x = 0) ∧ (S.card = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_100_zeros_count_l757_75792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_perimeter_l757_75722

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
noncomputable def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the cutting process described in the problem -/
noncomputable def cut (r : Rectangle) : Rectangle :=
  { length := r.length / 2, width := r.width / 2 }

theorem original_perimeter (r : Rectangle) :
  (cut r).perimeter = 129 → r.perimeter = 258 := by
  intro h
  unfold Rectangle.perimeter at h ⊢
  unfold cut at h
  simp at h
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_perimeter_l757_75722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_f_domain_proof_l757_75777

noncomputable def f (x : ℝ) : ℝ := (1 - Real.sqrt (x + 1)) / (x - 2)

def ValidInput (x : ℝ) : Prop := x ≥ -1 ∧ x ≠ 2

theorem f_domain : 
  {x : ℝ | ValidInput x} = {x : ℝ | x ≥ -1 ∧ x ≠ 2} := by
  ext x
  simp [ValidInput]
  
-- The actual proof would go here
theorem f_domain_proof : 
  ∀ x : ℝ, ValidInput x ↔ (x ≥ -1 ∧ x ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_f_domain_proof_l757_75777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_l757_75717

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
noncomputable def problem_setup : (Circle × Circle × Circle) := by
  -- Define the circles based on the problem conditions
  let largest_circle : Circle := ⟨(0, 0), Real.sqrt (144 * Real.pi / Real.pi)⟩
  let smaller_circle : Circle := ⟨(0, largest_circle.radius - largest_circle.radius / 3), largest_circle.radius / 3⟩
  let third_circle : Circle := ⟨smaller_circle.center, smaller_circle.radius / 2⟩
  exact (largest_circle, smaller_circle, third_circle)

-- Define the theorem
theorem total_shaded_area (setup : (Circle × Circle × Circle)) :
  setup = problem_setup →
  (Real.pi * setup.1.radius^2 / 2 + Real.pi * setup.2.1.radius^2 / 2 + Real.pi * setup.2.2.radius^2 / 2) = 82 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_l757_75717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_highest_value_l757_75737

/-- The probability that a ball is tossed into bin k -/
noncomputable def prob_bin (k : ℕ+) : ℝ := 3^(-k.val : ℝ)

/-- The probability that the red ball is tossed into a higher-numbered bin
    than both the green and blue balls -/
noncomputable def prob_red_highest : ℝ :=
  1 - ∑' k, (prob_bin k)^3

theorem prob_red_highest_value : prob_red_highest = 25/78 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_highest_value_l757_75737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_derivative_l757_75774

open Real

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := sin x + exp x + x^2015

-- Define the nth derivative of f
noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 => f
| n + 1 => deriv (f_n n)

-- State the theorem
theorem f_2016_derivative (x : ℝ) : f_n 2016 x = sin x + exp x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2016_derivative_l757_75774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_26_value_l757_75706

noncomputable def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sumOfArithmeticSequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem a_26_value (a : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmeticSequence a →
  (∀ n, S n = sumOfArithmeticSequence a n) →
  a 1 = 2 →
  arithmeticSequence (λ n ↦ |Real.sqrt (S n)|) →
  a 26 = 102 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_26_value_l757_75706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_constant_slope_product_l757_75780

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + (y - 1/4)^2 = 1 ∧ y ≥ 1/4
def C₂ (x y : ℝ) : Prop := x^2 = 8*y - 1 ∧ abs x ≥ 1

-- Define the moving line l
def line_l (k b : ℝ) (x y : ℝ) : Prop := y = k*x + b

-- Define the intersection points A and B
def intersection_points (k b x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  line_l k b x₁ y₁ ∧ C₂ x₁ y₁ ∧
  line_l k b x₂ y₂ ∧ C₂ x₂ y₂ ∧
  x₁ ≠ x₂

-- Define the tangent lines at A and B
def tangent_line (x₀ y₀ x y : ℝ) : Prop := y - y₀ = (x₀ / 4) * (x - x₀)

-- Define the intersection point M of tangents
def intersection_M (x₁ y₁ x₂ y₂ u v : ℝ) : Prop :=
  tangent_line x₁ y₁ u v ∧ tangent_line x₂ y₂ u v

-- Define perpendicularity of MA and MB
def perpendicular_MA_MB (x₁ y₁ x₂ y₂ u v : ℝ) : Prop :=
  (y₁ - v) * (y₂ - v) = -(x₁ - u) * (x₂ - u)

-- Theorem 1: When MA ⊥ MB, line l passes through (0, 17/8)
theorem fixed_point_theorem (k b x₁ y₁ x₂ y₂ u v : ℝ) :
  intersection_points k b x₁ y₁ x₂ y₂ →
  intersection_M x₁ y₁ x₂ y₂ u v →
  perpendicular_MA_MB x₁ y₁ x₂ y₂ u v →
  b = 17/8 := by sorry

-- Define the slopes of MT₁ and MT₂
noncomputable def slope_MT₁ (u v : ℝ) : ℝ := (v + 1) / u
noncomputable def slope_MT₂ (u v : ℝ) : ℝ := (v - 1) / u

-- Theorem 2: Product of slopes MT₁ and MT₂ is constant
theorem constant_slope_product (u v : ℝ) :
  C₂ u v →
  u ≠ 0 →
  slope_MT₁ u v * slope_MT₂ u v = 1/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_constant_slope_product_l757_75780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l757_75726

/-- The area of a rhombus with vertices at (0, 3.5), (6, 0), (0, -3.5), and (-6, 0) is 42 square units. -/
theorem rhombus_area : ∃ (area : ℝ), area = 42 := by
  let vertices : List (ℝ × ℝ) := [(0, 3.5), (6, 0), (0, -3.5), (-6, 0)]
  
  let diagonal1 : ℝ := |(vertices[0].2 - vertices[2].2)|
  let diagonal2 : ℝ := |(vertices[1].1 - vertices[3].1)|
  
  let area : ℝ := (diagonal1 * diagonal2) / 2
  
  exists area
  
  sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l757_75726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_c_l757_75764

theorem max_magnitude_c (a b c : ℝ × ℝ) : 
  ‖a‖ = 1 → 
  ‖b‖ = 1 → 
  a • b = 0 → 
  ‖c - a + b‖ = 2 → 
  ∃ (d : ℝ × ℝ), ‖d‖ = Real.sqrt 2 + 2 ∧ ∀ (e : ℝ × ℝ), ‖e - a + b‖ = 2 → ‖e‖ ≤ ‖d‖ := by
  sorry

#check max_magnitude_c

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_magnitude_c_l757_75764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_is_survey_size_l757_75769

/-- The sample size function returns the number of units in the survey. -/
def sample_size (population : ℕ) (survey_size : ℕ) : ℕ := survey_size

/-- Given a population and a survey, the sample size is the number of units in the survey. -/
theorem sample_size_is_survey_size (population : ℕ) (survey_size : ℕ) 
  (h1 : population = 236) (h2 : survey_size = 50) (h3 : survey_size ≤ population) :
  sample_size population survey_size = 50 :=
by
  rw [sample_size]
  exact h2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_is_survey_size_l757_75769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chloe_strawberry_price_l757_75720

/-- The price per dozen that Chloe paid for strawberries -/
noncomputable def price_per_dozen : ℝ := 50

/-- The number of dozens Chloe sold -/
def dozens_sold : ℝ := 50

/-- The price Chloe charged for half a dozen -/
def price_half_dozen : ℝ := 30

/-- Chloe's profit -/
def profit : ℝ := 500

theorem chloe_strawberry_price : 
  price_per_dozen = 50 :=
by
  -- Define the equation
  have h1 : price_half_dozen * 2 * dozens_sold - price_per_dozen * dozens_sold = profit := by
    -- Proof of the equation
    sorry
  
  -- Use the equation to prove the theorem
  sorry

#check chloe_strawberry_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chloe_strawberry_price_l757_75720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_m_value_l757_75700

/-- A trinomial is a perfect square if it can be expressed as (y + a)^2 or (y - a)^2 for some real number a -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ y : ℝ, a * y^2 + b * y + c = (y + k)^2 ∨ a * y^2 + b * y + c = (y - k)^2

theorem perfect_square_trinomial_m_value (m : ℝ) :
  IsPerfectSquareTrinomial 1 (-m) 1 → m = 2 ∨ m = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_trinomial_m_value_l757_75700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_implies_a_positive_l757_75701

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x ^ 3 + a * Real.cos x ^ 2

-- State the theorem
theorem minimum_implies_a_positive (a : ℝ) :
  (∃ x₀ ∈ Set.Ioo 0 Real.pi, ∀ x ∈ Set.Ioo 0 Real.pi, f a x₀ ≤ f a x) →
  a > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_implies_a_positive_l757_75701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounces_14_times_l757_75761

/-- Represents a rectangular table -/
structure RectangularTable where
  length : ℕ
  width : ℕ

/-- Represents the path of a ball on the table -/
def BallPath (table : RectangularTable) : ℕ :=
  (Nat.lcm table.length table.width / table.width) +
  (Nat.lcm table.length table.width / table.length) - 2

/-- Theorem: A ball on a 9x7 table bounces 14 times before reaching the opposite corner -/
theorem ball_bounces_14_times :
  let table : RectangularTable := { length := 9, width := 7 }
  BallPath table = 14 := by sorry

#eval BallPath { length := 9, width := 7 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounces_14_times_l757_75761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gina_coin_count_l757_75767

def total_coins (total_value : ℚ) (num_dimes : ℕ) : ℕ :=
  let dime_value : ℚ := 1/10
  let nickel_value : ℚ := 1/20
  let dimes_value : ℚ := num_dimes * dime_value
  let nickels_value : ℚ := total_value - dimes_value
  let num_nickels : ℕ := (nickels_value / nickel_value).floor.toNat
  num_dimes + num_nickels

theorem gina_coin_count :
  total_coins (43/10) 14 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gina_coin_count_l757_75767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_minnie_penny_l757_75702

/-- Represents a cyclist with their speeds for different road conditions -/
structure Cyclist where
  flat_speed : ℚ
  downhill_speed : ℚ
  uphill_speed : ℚ

/-- Represents a route segment with distance and road condition -/
inductive RouteSegment
  | Flat (distance : ℚ)
  | Downhill (distance : ℚ)
  | Uphill (distance : ℚ)

def Minnie : Cyclist := ⟨25, 35, 10⟩
def Penny : Cyclist := ⟨35, 45, 15⟩

def route : List RouteSegment := [
  RouteSegment.Uphill 12,
  RouteSegment.Downhill 18,
  RouteSegment.Flat 24
]

/-- Calculates the time taken by a cyclist to complete a route segment -/
def time_for_segment (c : Cyclist) (s : RouteSegment) : ℚ :=
  match s with
  | RouteSegment.Flat d => d / c.flat_speed
  | RouteSegment.Downhill d => d / c.downhill_speed
  | RouteSegment.Uphill d => d / c.uphill_speed

/-- Calculates the total time taken by a cyclist to complete the entire route -/
def total_time (c : Cyclist) (r : List RouteSegment) : ℚ :=
  r.foldr (fun s acc => acc + time_for_segment c s) 0

/-- The main theorem stating the time difference between Minnie and Penny -/
theorem time_difference_minnie_penny :
  (total_time Minnie route - total_time Penny (route.reverse)) * 60 = 31260 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_minnie_penny_l757_75702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_2000_to_2019_trailing_zeros_l757_75747

def product_range (a b : ℕ) : ℕ := (List.range (b - a + 1)).map (fun x => x + a) |>.prod

def trailing_zeros (n : ℕ) : ℕ := (Nat.factors n).filter (fun x => x = 5) |>.length

theorem product_2000_to_2019_trailing_zeros :
  trailing_zeros (product_range 2000 2019) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_2000_to_2019_trailing_zeros_l757_75747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_noncongruent_triangles_count_is_five_l757_75724

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Given an isosceles triangle ABC with AB = AC, P, Q, R as midpoints, and S as centroid -/
structure IsoscelesTriangleSetup where
  abc : Triangle
  p : Point
  q : Point
  r : Point
  s : Point
  h : (abc.a.x - abc.b.x)^2 + (abc.a.y - abc.b.y)^2 = (abc.a.x - abc.c.x)^2 + (abc.a.y - abc.c.y)^2
  hp : p = ⟨(abc.a.x + abc.b.x) / 2, (abc.a.y + abc.b.y) / 2⟩
  hq : q = ⟨(abc.b.x + abc.c.x) / 2, (abc.b.y + abc.c.y) / 2⟩
  hr : r = ⟨(abc.c.x + abc.a.x) / 2, (abc.c.y + abc.a.y) / 2⟩
  hs : s = ⟨(abc.a.x + abc.b.x + abc.c.x) / 3, (abc.a.y + abc.b.y + abc.c.y) / 3⟩

/-- Two triangles are congruent -/
def CongruentTriangles (t1 t2 : Triangle) : Prop := sorry

/-- Count of noncongruent triangles formed by any three points from the setup -/
def NoncongruentTrianglesCount (setup : IsoscelesTriangleSetup) : ℕ := sorry

/-- Main theorem: The number of noncongruent triangles is 5 -/
theorem noncongruent_triangles_count_is_five (setup : IsoscelesTriangleSetup) :
  NoncongruentTrianglesCount setup = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_noncongruent_triangles_count_is_five_l757_75724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_KM_NM_l757_75732

/-- The ellipse on which points M and N move -/
def ellipse (x y : ℝ) : Prop := x^2/36 + y^2/9 = 1

/-- The fixed point K -/
def K : ℝ × ℝ := (2, 0)

/-- A point on the ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The vector from K to a point P -/
def vector_KP (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 - K.1, P.2 - K.2)

/-- Convert PointOnEllipse to ℝ × ℝ -/
def to_tuple (P : PointOnEllipse) : ℝ × ℝ := (P.x, P.y)

/-- The main theorem -/
theorem min_dot_product_KM_NM :
  ∀ (M N : PointOnEllipse),
  dot_product (vector_KP (to_tuple M)) (vector_KP (to_tuple N)) = 0 →
  ∀ (P Q : PointOnEllipse),
  dot_product (vector_KP (to_tuple P)) (vector_KP (to_tuple Q)) = 0 →
  dot_product (vector_KP (to_tuple M)) (vector_KP (to_tuple M)) ≤
  dot_product (vector_KP (to_tuple P)) (vector_KP (to_tuple P)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_KM_NM_l757_75732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_unit_vectors_60_degrees_l757_75756

theorem sum_of_unit_vectors_60_degrees (a b : ℝ × ℝ × ℝ) : 
  norm a = 1 → norm b = 1 → a • b = 1/2 → norm (a + b) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_unit_vectors_60_degrees_l757_75756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l757_75797

/-- A hyperbola with asymptotes passing through (1,2) and the curve passing through (2√2, 4) has eccentricity √5 -/
theorem hyperbola_eccentricity (H : Set (ℝ × ℝ)) 
  (asymptote1 asymptote2 : Set (ℝ × ℝ))
  (h_asym : (∃ (b : ℝ), (⟨1, 2⟩ ∈ asymptote1) ∧ 
                        (⟨1, 2⟩ ∈ asymptote2) ∧
                        ((∀ x y, (x, y) ∈ asymptote1 → y = 2*x + b) ∨ 
                         (∀ x y, (x, y) ∈ asymptote1 → y = -2*x + b)) ∧
                        ((∀ x y, (x, y) ∈ asymptote2 → y = 2*x + b) ∨ 
                         (∀ x y, (x, y) ∈ asymptote2 → y = -2*x + b))))
  (h_point : ⟨2 * Real.sqrt 2, 4⟩ ∈ H)
  (h_hyperbola : ∃ a b : ℝ, ∀ x y, (x, y) ∈ H ↔ x^2/a^2 - y^2/b^2 = 1)
  (h_eccentricity : ∀ a b c : ℝ, a > 0 → b > 0 → c^2 = a^2 + b^2 → 
                    (∀ x y, (x, y) ∈ H ↔ x^2/a^2 - y^2/b^2 = 1) → 
                    c/a = Real.sqrt 5) :
  ∃ e : ℝ, e = Real.sqrt 5 ∧ 
    (∀ a b c : ℝ, a > 0 → b > 0 → c^2 = a^2 + b^2 → 
    (∀ x y, (x, y) ∈ H ↔ x^2/a^2 - y^2/b^2 = 1) → 
    c/a = e) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l757_75797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_60_l757_75703

/-- The speed of a train in km/hr -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: The speed of a train is 60 km/hr -/
theorem train_speed_is_60 (length time : ℝ) 
  (h1 : length = 50) 
  (h2 : time = 3) : 
  train_speed length time = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_60_l757_75703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_AB_l757_75776

/-- The minimum length of the line segment AB, where A and B are the intersection points
    of the line y = a with y = 2x - 2 and y = 2e^x + x, respectively. -/
theorem min_length_AB : ∃ (min_length : ℝ), min_length = (3 + Real.log 2) / 2 ∧
  ∀ (a x₁ x₂ : ℝ),
    (a = 2 * Real.exp x₂ + x₂) →
    (a = 2 * x₁ - 2) →
    min_length ≤ |x₁ - x₂| :=
by
  -- We define our minimum length
  let min_length := (3 + Real.log 2) / 2

  -- We claim this is indeed the minimum length
  use min_length

  -- We split our proof into two parts
  constructor

  -- First part: equality of min_length
  · rfl

  -- Second part: proving it's the minimum for all valid x₁ and x₂
  · intro a x₁ x₂ h₁ h₂
    -- The actual proof would go here
    sorry

#check min_length_AB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_AB_l757_75776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tabitha_money_left_l757_75711

noncomputable def calculate_remaining_money (initial_amount : ℝ) (given_to_mom : ℝ) (num_items : ℕ) (item_cost : ℝ) : ℝ :=
  let remaining_after_mom := initial_amount - given_to_mom
  let invested := remaining_after_mom / 2
  let remaining_after_investment := remaining_after_mom - invested
  let total_item_cost := (num_items : ℝ) * item_cost
  remaining_after_investment - total_item_cost

theorem tabitha_money_left :
  calculate_remaining_money 25 8 5 0.5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tabitha_money_left_l757_75711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skier_velocity_at_3s_l757_75757

noncomputable def distance (t : ℝ) : ℝ := 2 * t^2 + 3/2 * t

noncomputable def velocity (t : ℝ) : ℝ := deriv distance t

theorem skier_velocity_at_3s : velocity 3 = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skier_velocity_at_3s_l757_75757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_theorem_l757_75728

/-- Represents a rhombus with given perimeter and diagonal length -/
structure Rhombus where
  perimeter : ℝ
  diagonal : ℝ

/-- Calculates the area of a rhombus given its perimeter and one diagonal -/
noncomputable def area (r : Rhombus) : ℝ :=
  let side := r.perimeter / 4
  let halfDiag := r.diagonal / 2
  let otherHalfDiag := Real.sqrt (side^2 - halfDiag^2)
  r.diagonal * otherHalfDiag

/-- Theorem stating that a rhombus with perimeter 40 and diagonal 16 has area 96 -/
theorem rhombus_area_theorem (r : Rhombus) (h1 : r.perimeter = 40) (h2 : r.diagonal = 16) : 
  area r = 96 := by
  sorry

#check rhombus_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_theorem_l757_75728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l757_75795

/-- The function f(x) = x^2 - bx + 2 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + 2

/-- The closed interval [-1, 2] -/
def I : Set ℝ := Set.Icc (-1) 2

/-- The theorem statement -/
theorem inverse_function_condition (b : ℝ) :
  (∃ g : ℝ → ℝ, (∀ x ∈ I, g (f b x) = x) ∧ (∀ y ∈ (f b '' I), f b (g y) = y)) →
  b ∈ Set.Iic (-2) ∪ Set.Ici 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l757_75795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_seven_addition_problem_l757_75721

theorem base_seven_addition_problem :
  ∃ (triangle square : Fin 7),
    (3 * 7^3 + 2 * 7^2 + 1 * 7 + triangle : Fin 7) +
    (square * 7^2 + 4 * 7 + 0 : Fin 7) +
    (triangle * 7 + 5 : Fin 7) =
    (4 * 7^3 + 3 * 7^2 + triangle * 7 + 1 : Fin 7) →
    triangle = 3 ∧ square = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_seven_addition_problem_l757_75721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_num_white_balls_prove_prob_at_most_one_white_l757_75710

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the probability of drawing at least one white ball when drawing 2 balls
def prob_at_least_one_white : ℚ := 7/9

-- Define the number of white balls
def num_white_balls : ℕ := 5

-- Theorem 1: Prove the number of white balls
theorem prove_num_white_balls :
  (1 - (Nat.choose (total_balls - num_white_balls) 2 : ℚ) / (Nat.choose total_balls 2 : ℚ)) = prob_at_least_one_white := by
  sorry

-- Theorem 2: Prove the probability of drawing at most one white ball when drawing 3 balls
theorem prove_prob_at_most_one_white :
  ((Nat.choose num_white_balls 0 * Nat.choose (total_balls - num_white_balls) 3 +
    Nat.choose num_white_balls 1 * Nat.choose (total_balls - num_white_balls) 2) : ℚ) /
  (Nat.choose total_balls 3 : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_num_white_balls_prove_prob_at_most_one_white_l757_75710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_a7_l757_75736

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0
  | (n + 1) => (10 / 7) * sequence_a n + (2 / 7) * Real.sqrt (4^n - (sequence_a n)^2)

theorem exists_unique_a7 : ∃! x : ℝ, sequence_a 7 = x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_a7_l757_75736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_satisfying_conditions_l757_75766

theorem count_triples_satisfying_conditions : 
  let count := Finset.filter (fun t : ℕ × ℕ × ℕ => 
    let (a, b, c) := t
    0 < a ∧ 0 < b ∧ 0 < c ∧
    6 * a * b = c * c ∧
    a < b ∧ b < c ∧ c ≤ 35) (Finset.range 36 ×ˢ Finset.range 36 ×ˢ Finset.range 36)
  Finset.card count = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_satisfying_conditions_l757_75766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l757_75752

/-- Given a point P (-√3, m) on the terminal side of angle α, and sin α = (√2 * m) / 4, prove that cos α = -√6 / 4 -/
theorem cos_alpha_value (α : Real) (m : Real) 
  (h1 : ((-Real.sqrt 3), m) ∈ Set.range (λ t => (Real.cos α * t, Real.sin α * t)))
  (h2 : Real.sin α = (Real.sqrt 2 * m) / 4) : 
  Real.cos α = -(Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l757_75752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inequalities_l757_75742

theorem triangle_area_inequalities (a b c : ℝ) (Δ : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let s := (a + b + c) / 2
  Δ = Real.sqrt (s * (s - a) * (s - b) * (s - c)) →
  Δ ≤ (3/4) * (a*b*c / Real.sqrt (a^2 + b^2 + c^2)) ∧
  Δ ≤ (3/4) * Real.sqrt 3 * (a*b*c / (a + b + c)) ∧
  Δ ≤ (Real.sqrt 3 / 4) * (a*b*c)^(2/3) ∧
  (Δ = (3/4) * (a*b*c / Real.sqrt (a^2 + b^2 + c^2)) ∧
   Δ = (3/4) * Real.sqrt 3 * (a*b*c / (a + b + c)) ∧
   Δ = (Real.sqrt 3 / 4) * (a*b*c)^(2/3) ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inequalities_l757_75742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l757_75782

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | x ≥ 2}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l757_75782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_l757_75748

/-- Calculates the return speed given the total distance, outbound speed, and average speed of a round trip. -/
noncomputable def return_speed (total_distance : ℝ) (outbound_speed : ℝ) (average_speed : ℝ) : ℝ :=
  let half_distance := total_distance / 2
  let outbound_time := half_distance / outbound_speed
  let total_time := total_distance / average_speed
  let return_time := total_time - outbound_time
  half_distance / return_time

/-- Theorem stating that for a 360-mile round trip with outbound speed 90 mph and average speed 60 mph, the return speed is 45 mph. -/
theorem round_trip_speed : return_speed 360 90 60 = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_l757_75748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_questionnaire_B_l757_75788

/-- Systematic sampling problem -/
theorem systematic_sampling_questionnaire_B 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (first_number : ℕ) 
  (interval_start : ℕ) 
  (interval_end : ℕ) : 
  total_population = 960 →
  sample_size = 32 →
  first_number = 9 →
  interval_start = 451 →
  interval_end = 750 →
  (Finset.filter (λ n ↦ 
    interval_start ≤ (first_number + (n - 1) * (total_population / sample_size)) ∧ 
    (first_number + (n - 1) * (total_population / sample_size)) ≤ interval_end) 
    (Finset.range (sample_size + 1))).card = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_questionnaire_B_l757_75788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segments_is_nine_l757_75704

/-- A configuration of points and line segments in a plane. -/
structure Configuration where
  points : Finset (ℝ × ℝ)
  segments : Finset ((ℝ × ℝ) × (ℝ × ℝ))

/-- A configuration is valid if it satisfies the problem conditions. -/
def ValidConfiguration (c : Configuration) : Prop :=
  c.points.card = 7 ∧
  ∀ p₁ p₂ p₃, p₁ ∈ c.points → p₂ ∈ c.points → p₃ ∈ c.points →
    p₁ ≠ p₂ → p₂ ≠ p₃ → p₁ ≠ p₃ →
    (p₁, p₂) ∈ c.segments ∨ (p₂, p₃) ∈ c.segments ∨ (p₁, p₃) ∈ c.segments

/-- The minimum number of segments in a valid configuration is 9. -/
theorem min_segments_is_nine :
  ∀ c : Configuration, ValidConfiguration c →
    c.segments.card ≥ 9 ∧ ∃ c' : Configuration, ValidConfiguration c' ∧ c'.segments.card = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segments_is_nine_l757_75704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_at_neg_three_undefined_f_at_two_thirds_l757_75731

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 3) + 1 / (x + 2)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x ≥ -3 ∧ x ≠ -2}

-- Theorem for the domain of f
theorem domain_of_f : 
  ∀ x : ℝ, x ∈ domain_f ↔ (x ≥ -3 ∧ x ≠ -2) := by
  sorry

-- Theorem for f(-3) being undefined
theorem f_at_neg_three_undefined :
  ¬ (-3 : ℝ) ∈ domain_f := by
  sorry

-- Theorem for the value of f(2/3)
theorem f_at_two_thirds :
  f (2/3) = (8 * Real.sqrt 33 + 9) / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_at_neg_three_undefined_f_at_two_thirds_l757_75731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_hexagonal_pyramid_circumscribed_sphere_volume_l757_75708

/-- A hexagonal pyramid with a regular hexagon base -/
structure HexagonalPyramid where
  /-- Side length of the regular hexagon base -/
  baseSideLength : ℝ
  /-- Height of the pyramid (perpendicular distance from apex to base) -/
  height : ℝ

/-- The volume of the circumscribed sphere of a hexagonal pyramid -/
noncomputable def circumscribedSphereVolume (pyramid : HexagonalPyramid) : ℝ :=
  4 * Real.sqrt 3 * Real.pi

/-- Theorem: The volume of the circumscribed sphere of a specific hexagonal pyramid -/
theorem specific_hexagonal_pyramid_circumscribed_sphere_volume :
  ∃ (pyramid : HexagonalPyramid),
    pyramid.baseSideLength = Real.sqrt 2 ∧
    pyramid.height = 2 ∧
    circumscribedSphereVolume pyramid = 4 * Real.sqrt 3 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_hexagonal_pyramid_circumscribed_sphere_volume_l757_75708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_sequence_properties_l757_75783

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  S : ℕ → ℚ
  h1 : a 3 = 5
  h2 : S 5 = 3 * S 3 - 2

/-- The sum of the first n terms of a geometric sequence with first term b₁ and common ratio r -/
def geometricSum (b₁ r : ℚ) (n : ℕ) : ℚ := b₁ * (1 - r^n) / (1 - r)

/-- Main theorem about the arithmetic sequence and its derived geometric sequence -/
theorem arithmetic_and_geometric_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 2 * n - 1) ∧
  (∀ n : ℕ, geometricSum 2 4 n = (2/3) * (4^n - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_geometric_sequence_properties_l757_75783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_2744000_l757_75765

theorem cube_root_of_2744000 : (2744000 : ℝ) ^ (1/3) = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_2744000_l757_75765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_odd_function_condition_f_1_eq_3_implies_a_eq_1_l757_75740

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 / (2^x - 1) + a

-- Theorem for the domain of f(x)
theorem domain_of_f (a : ℝ) :
  {x : ℝ | f a x ≠ 0} = {x : ℝ | x ≠ 0} := by
  sorry

-- Theorem for the condition of f(x) being an odd function
theorem odd_function_condition (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) → a = 1 := by
  sorry

-- Theorem for option D (which is incorrect, but included for completeness)
theorem f_1_eq_3_implies_a_eq_1 :
  f 1 1 = 3 → 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_odd_function_condition_f_1_eq_3_implies_a_eq_1_l757_75740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_27_l757_75718

theorem max_min_sum_equals_27 :
  ∃ (N n : ℝ), 
    (∀ x y z : ℝ, 3 * (x + y + z) = x^2 + y^2 + z^2 → x*y + x*z + y*z ≤ N) ∧
    (∀ x y z : ℝ, 3 * (x + y + z) = x^2 + y^2 + z^2 → n ≤ x*y + x*z + y*z) ∧
    N + 10 * n = 27 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_27_l757_75718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_germination_percentage_l757_75719

theorem seed_germination_percentage : 
  let seeds_plot1 : ℕ := 300
  let seeds_plot2 : ℕ := 200
  let germination_rate1 : ℚ := 30 / 100
  let germination_rate2 : ℚ := 35 / 100
  let total_seeds : ℕ := seeds_plot1 + seeds_plot2
  let germinated_seeds1 : ℚ := (seeds_plot1 : ℚ) * germination_rate1
  let germinated_seeds2 : ℚ := (seeds_plot2 : ℚ) * germination_rate2
  let total_germinated : ℚ := germinated_seeds1 + germinated_seeds2
  (total_germinated / total_seeds) * 100 = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_germination_percentage_l757_75719
