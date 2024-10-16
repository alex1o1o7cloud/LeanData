import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_is_square_of_binomial_l1025_102525

/-- The polynomial 4x^2 + 16x + 16 is the square of a binomial. -/
theorem polynomial_is_square_of_binomial :
  ∃ (r s : ℝ), ∀ x, 4 * x^2 + 16 * x + 16 = (r * x + s)^2 :=
sorry

end NUMINAMATH_CALUDE_polynomial_is_square_of_binomial_l1025_102525


namespace NUMINAMATH_CALUDE_expression_equals_zero_l1025_102545

theorem expression_equals_zero : (-3)^3 + (-3)^2 * 3^1 + 3^2 * (-3)^1 + 3^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l1025_102545


namespace NUMINAMATH_CALUDE_store_earnings_calculation_l1025_102532

/-- Represents the earnings from selling bottled drinks in a country store. -/
def storeEarnings (colaCost juiceCost waterCost : ℝ) (colaSold juiceSold waterSold : ℕ) : ℝ :=
  colaCost * colaSold + juiceCost * juiceSold + waterCost * waterSold

/-- Theorem stating that the store's earnings from selling the specified quantities of drinks at given prices is $88. -/
theorem store_earnings_calculation :
  storeEarnings 3 1.5 1 15 12 25 = 88 := by
  sorry

end NUMINAMATH_CALUDE_store_earnings_calculation_l1025_102532


namespace NUMINAMATH_CALUDE_shaded_area_square_l1025_102562

theorem shaded_area_square (a : ℝ) (h : a = 4) : 
  let square_area := a ^ 2
  let shaded_area := square_area / 2
  shaded_area = 8 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_square_l1025_102562


namespace NUMINAMATH_CALUDE_cost_difference_l1025_102524

def dan_money : ℕ := 5
def chocolate_cost : ℕ := 3
def candy_bar_cost : ℕ := 7

theorem cost_difference : candy_bar_cost - chocolate_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_l1025_102524


namespace NUMINAMATH_CALUDE_problem_proof_l1025_102511

theorem problem_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ x*y > a*b) → a*b ≤ 1/8 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → 1/x + 8/y ≥ 25) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → x^2 + 4*y^2 ≥ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_problem_proof_l1025_102511


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l1025_102508

theorem largest_n_for_equation : 
  (∃ n : ℕ, 
    (∀ m : ℕ, m > n → 
      ¬∃ x y z : ℕ+, m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10) ∧
    (∃ x y z : ℕ+, n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10)) ∧
  (∀ n : ℕ, 
    (∀ m : ℕ, m > n → 
      ¬∃ x y z : ℕ+, m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10) ∧
    (∃ x y z : ℕ+, n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10) →
    n = 4) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l1025_102508


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l1025_102548

/-- Given a line L1: 2x + 3y = 9, prove that a line L2 perpendicular to L1 with y-intercept 5 has x-intercept -10/3 -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := fun x y ↦ 2 * x + 3 * y = 9
  let m1 : ℝ := -2 / 3  -- slope of L1
  let m2 : ℝ := -1 / m1  -- slope of perpendicular line
  let L2 : ℝ → ℝ → Prop := fun x y ↦ y = m2 * x + 5  -- equation of perpendicular line
  let x_intercept : ℝ := -10 / 3
  (∀ x y, L2 x y → y = 0 → x = x_intercept) :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l1025_102548


namespace NUMINAMATH_CALUDE_actual_miles_traveled_l1025_102500

/-- Represents an odometer that skips the digit 7 -/
structure FaultyOdometer where
  current_reading : Nat
  skipped_digit : Nat

/-- Calculates the number of skipped readings up to a given number -/
def count_skipped_readings (n : Nat) : Nat :=
  -- Implementation details omitted
  sorry

/-- Theorem: The actual miles traveled when the faulty odometer reads 003008 is 2194 -/
theorem actual_miles_traveled (o : FaultyOdometer) 
  (h1 : o.current_reading = 3008)
  (h2 : o.skipped_digit = 7) : 
  o.current_reading - count_skipped_readings o.current_reading = 2194 := by
  sorry

end NUMINAMATH_CALUDE_actual_miles_traveled_l1025_102500


namespace NUMINAMATH_CALUDE_sloth_shoe_theorem_l1025_102564

/-- The number of feet a sloth has -/
def sloth_feet : ℕ := 3

/-- The number of complete sets of shoes desired -/
def desired_sets : ℕ := 5

/-- The number of sets of shoes already owned -/
def owned_sets : ℕ := 1

/-- Calculate the number of pairs of shoes needed to be purchased -/
def shoes_to_buy : ℕ :=
  (desired_sets * sloth_feet - owned_sets * sloth_feet) / 2

theorem sloth_shoe_theorem : shoes_to_buy = 6 := by
  sorry

end NUMINAMATH_CALUDE_sloth_shoe_theorem_l1025_102564


namespace NUMINAMATH_CALUDE_quadratic_contradiction_l1025_102595

theorem quadratic_contradiction : ¬ ∃ (a b c : ℝ), 
  ((a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ b > 0 ∧ c > 0)) ∧
  ((a < 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ b < 0 ∧ c < 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_contradiction_l1025_102595


namespace NUMINAMATH_CALUDE_layoff_plans_count_l1025_102554

def staff_count : ℕ := 10
def layoff_count : ℕ := 4

/-- The number of ways to select 4 people out of 10 for layoff, 
    where two specific people (A and B) cannot both be kept -/
def layoff_plans : ℕ := Nat.choose (staff_count - 2) layoff_count + 
                        2 * Nat.choose (staff_count - 2) (layoff_count - 1)

theorem layoff_plans_count : layoff_plans = 182 := by
  sorry

end NUMINAMATH_CALUDE_layoff_plans_count_l1025_102554


namespace NUMINAMATH_CALUDE_triangle_side_length_l1025_102540

theorem triangle_side_length (A B : Real) (b : Real) (hA : A = 60 * π / 180) (hB : B = 45 * π / 180) (hb : b = Real.sqrt 2) :
  ∃ a : Real, a = Real.sqrt 3 ∧ a * Real.sin B = b * Real.sin A := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1025_102540


namespace NUMINAMATH_CALUDE_spontaneous_low_temp_signs_l1025_102578

/-- Represents the change in enthalpy -/
def ΔH : ℝ := sorry

/-- Represents the change in entropy -/
def ΔS : ℝ := sorry

/-- Represents temperature -/
def T : ℝ := sorry

/-- Represents the change in Gibbs free energy -/
def ΔG (T : ℝ) : ℝ := ΔH - T * ΔS

/-- Represents that the reaction is spontaneous -/
def is_spontaneous (T : ℝ) : Prop := ΔG T < 0

/-- Represents that the reaction is spontaneous only at low temperatures -/
def spontaneous_at_low_temp : Prop :=
  ∃ T₀ > 0, ∀ T, 0 < T → T < T₀ → is_spontaneous T

theorem spontaneous_low_temp_signs :
  spontaneous_at_low_temp → ΔH < 0 ∧ ΔS < 0 := by
  sorry

end NUMINAMATH_CALUDE_spontaneous_low_temp_signs_l1025_102578


namespace NUMINAMATH_CALUDE_geometric_sequence_a8_l1025_102533

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a8 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 = 2 →
  a 3 * a 4 = 32 →
  a 8 = 128 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a8_l1025_102533


namespace NUMINAMATH_CALUDE_det_A_eq_neg_46_l1025_102503

def A : Matrix (Fin 3) (Fin 3) ℝ := !![2, -4, 5; 0, 6, -2; 3, -1, 2]

theorem det_A_eq_neg_46 : Matrix.det A = -46 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_neg_46_l1025_102503


namespace NUMINAMATH_CALUDE_shower_water_usage_l1025_102580

theorem shower_water_usage (total : ℕ) (remy : ℕ) (h1 : total = 33) (h2 : remy = 25) :
  ∃ (M : ℕ), remy = M * (total - remy) + 1 ∧ M = 3 := by
sorry

end NUMINAMATH_CALUDE_shower_water_usage_l1025_102580


namespace NUMINAMATH_CALUDE_sixth_term_is_thirteen_l1025_102521

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 3 = 7 ∧ 
  a 5 = a 2 + 6

/-- The 6th term of the arithmetic sequence is 13 -/
theorem sixth_term_is_thirteen 
  (a : ℕ → ℝ) 
  (h : ArithmeticSequence a) : 
  a 6 = 13 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_is_thirteen_l1025_102521


namespace NUMINAMATH_CALUDE_b_is_killer_l1025_102571

-- Define the characters
inductive Character : Type
| A : Character
| B : Character
| C : Character

-- Define the actions
def poisoned_water (x y : Character) : Prop := x = Character.A ∧ y = Character.C
def made_hole (x y : Character) : Prop := x = Character.B ∧ y = Character.C
def died_of_thirst (x : Character) : Prop := x = Character.C

-- Define the killer
def is_killer (x : Character) : Prop := x = Character.B

-- Theorem statement
theorem b_is_killer 
  (h1 : poisoned_water Character.A Character.C)
  (h2 : made_hole Character.B Character.C)
  (h3 : died_of_thirst Character.C) :
  is_killer Character.B :=
sorry

end NUMINAMATH_CALUDE_b_is_killer_l1025_102571


namespace NUMINAMATH_CALUDE_min_distance_exp_ln_l1025_102518

/-- The minimum distance between a point on y = e^x and a point on y = ln(x) -/
theorem min_distance_exp_ln : ∀ (P Q : ℝ × ℝ),
  (∃ x : ℝ, P = (x, Real.exp x)) →
  (∃ y : ℝ, Q = (Real.exp y, y)) →
  ∃ d : ℝ, d = Real.sqrt 2 ∧ ∀ P' Q', 
    (∃ x' : ℝ, P' = (x', Real.exp x')) →
    (∃ y' : ℝ, Q' = (Real.exp y', y')) →
    d ≤ Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_exp_ln_l1025_102518


namespace NUMINAMATH_CALUDE_integer_in_3_rows_and_3_cols_l1025_102590

/-- Represents a 21x21 array of integers -/
def Array21x21 := Fin 21 → Fin 21 → Int

/-- Predicate to check if a row has at most 6 different integers -/
def row_at_most_6_different (arr : Array21x21) (row : Fin 21) : Prop :=
  (Finset.univ.image (fun col => arr row col)).card ≤ 6

/-- Predicate to check if a column has at most 6 different integers -/
def col_at_most_6_different (arr : Array21x21) (col : Fin 21) : Prop :=
  (Finset.univ.image (fun row => arr row col)).card ≤ 6

/-- Predicate to check if an integer appears in at least 3 rows -/
def in_at_least_3_rows (arr : Array21x21) (n : Int) : Prop :=
  (Finset.univ.filter (fun row => ∃ col, arr row col = n)).card ≥ 3

/-- Predicate to check if an integer appears in at least 3 columns -/
def in_at_least_3_cols (arr : Array21x21) (n : Int) : Prop :=
  (Finset.univ.filter (fun col => ∃ row, arr row col = n)).card ≥ 3

theorem integer_in_3_rows_and_3_cols (arr : Array21x21) 
  (h_rows : ∀ row, row_at_most_6_different arr row)
  (h_cols : ∀ col, col_at_most_6_different arr col) :
  ∃ n : Int, in_at_least_3_rows arr n ∧ in_at_least_3_cols arr n := by
  sorry

end NUMINAMATH_CALUDE_integer_in_3_rows_and_3_cols_l1025_102590


namespace NUMINAMATH_CALUDE_decagon_sign_change_impossible_l1025_102528

/-- Represents a point in the decagon with a sign -/
structure Point where
  sign : Int
  deriving Repr

/-- Represents the decagon with its points -/
structure Decagon where
  points : List Point
  deriving Repr

/-- An operation that can change signs on a side or diagonal -/
inductive Operation
  | Side : Nat → Operation
  | Diagonal : Nat → Nat → Operation
  deriving Repr

/-- Apply an operation to a decagon -/
def applyOperation (d : Decagon) (op : Operation) : Decagon :=
  sorry

/-- Check if all signs in the decagon are negative -/
def allNegative (d : Decagon) : Bool :=
  sorry

/-- Initialize a decagon with all positive signs -/
def initDecagon : Decagon :=
  sorry

theorem decagon_sign_change_impossible :
  ∀ (ops : List Operation),
    allNegative (ops.foldl applyOperation initDecagon) = false :=
  sorry

end NUMINAMATH_CALUDE_decagon_sign_change_impossible_l1025_102528


namespace NUMINAMATH_CALUDE_repeating_decimal_calculation_l1025_102501

/-- Represents a repeating decimal where the digits after the decimal point repeat indefinitely -/
def repeating_decimal (n : ℕ) (d : ℕ) : ℚ := n / d

theorem repeating_decimal_calculation :
  let x := repeating_decimal 27 100000
  (10^5 - 10^3) * x = 26.73 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_calculation_l1025_102501


namespace NUMINAMATH_CALUDE_contrapositive_isosceles_angles_l1025_102587

-- Define a triangle ABC
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the property of being an isosceles triangle
def isIsosceles (t : Triangle) : Prop := sorry

-- Define the property of having two equal interior angles
def hasTwoEqualAngles (t : Triangle) : Prop := sorry

-- State the theorem
theorem contrapositive_isosceles_angles (t : Triangle) :
  (¬ isIsosceles t → ¬ hasTwoEqualAngles t) ↔
  (hasTwoEqualAngles t → isIsosceles t) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_isosceles_angles_l1025_102587


namespace NUMINAMATH_CALUDE_partnership_profit_l1025_102537

/-- Calculates the profit of a business partnership given the investments and profit sharing rules -/
theorem partnership_profit (mary_investment mike_investment : ℚ) 
  (h1 : mary_investment = 700)
  (h2 : mike_investment = 300)
  (h3 : mary_investment + mike_investment > 0) :
  ∃ (P : ℚ), 
    (P / 6 + 7 * (2 * P / 3) / 10) - (P / 6 + 3 * (2 * P / 3) / 10) = 800 ∧ 
    P = 3000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_l1025_102537


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1025_102522

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a + a₁*(x + 2) + a₂*(x + 2)^2 + a₃*(x + 2)^3 + 
    a₄*(x + 2)^4 + a₅*(x + 2)^5 + a₆*(x + 2)^6 + a₇*(x + 2)^7 + a₈*(x + 2)^8 + 
    a₉*(x + 2)^9 + a₁₀*(x + 2)^10 + a₁₁*(x + 2)^11) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1025_102522


namespace NUMINAMATH_CALUDE_tape_recorder_cost_l1025_102509

theorem tape_recorder_cost :
  ∃ (x : ℕ) (p : ℝ),
    x > 2 ∧
    170 < p ∧ p < 195 ∧
    p / (x - 2) - p / x = 1 ∧
    p = 180 := by
  sorry

end NUMINAMATH_CALUDE_tape_recorder_cost_l1025_102509


namespace NUMINAMATH_CALUDE_sum_of_numbers_ge_1_1_l1025_102530

theorem sum_of_numbers_ge_1_1 : 
  let numbers : List ℝ := [1.4, 9/10, 1.2, 0.5, 13/10]
  let filtered_numbers := numbers.filter (λ x => x ≥ 1.1)
  filtered_numbers.sum = 3.9 := by sorry

end NUMINAMATH_CALUDE_sum_of_numbers_ge_1_1_l1025_102530


namespace NUMINAMATH_CALUDE_distance_between_points_l1025_102584

theorem distance_between_points : 
  let p₁ : ℝ × ℝ := (3, 4)
  let p₂ : ℝ × ℝ := (8, -6)
  Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2) = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1025_102584


namespace NUMINAMATH_CALUDE_chocolate_problem_l1025_102572

theorem chocolate_problem (cost_price selling_price : ℝ) 
  (h1 : cost_price * 81 = selling_price * 45)
  (h2 : (selling_price - cost_price) / cost_price = 0.8) :
  81 = 81 := by sorry

end NUMINAMATH_CALUDE_chocolate_problem_l1025_102572


namespace NUMINAMATH_CALUDE_average_difference_l1025_102526

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = (x + 50 + 45) / 3 + 5 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l1025_102526


namespace NUMINAMATH_CALUDE_sin_30_plus_cos_60_l1025_102514

theorem sin_30_plus_cos_60 : Real.sin (30 * π / 180) + Real.cos (60 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_plus_cos_60_l1025_102514


namespace NUMINAMATH_CALUDE_vincent_sticker_packs_l1025_102594

theorem vincent_sticker_packs (yesterday_packs today_extra_packs : ℕ) :
  yesterday_packs = 15 →
  today_extra_packs = 10 →
  yesterday_packs + (yesterday_packs + today_extra_packs) = 40 := by
  sorry

end NUMINAMATH_CALUDE_vincent_sticker_packs_l1025_102594


namespace NUMINAMATH_CALUDE_only_one_proposition_is_true_l1025_102541

-- Define the basic types
def Solid : Type := Unit
def View : Type := Unit

-- Define the properties
def has_three_identical_views (s : Solid) : Prop := sorry
def is_cube (s : Solid) : Prop := sorry
def front_view_is_rectangle (s : Solid) : Prop := sorry
def top_view_is_rectangle (s : Solid) : Prop := sorry
def is_cuboid (s : Solid) : Prop := sorry
def all_views_are_rectangles (s : Solid) : Prop := sorry
def front_view_is_isosceles_trapezoid (s : Solid) : Prop := sorry
def side_view_is_isosceles_trapezoid (s : Solid) : Prop := sorry
def is_frustum (s : Solid) : Prop := sorry

-- Define the propositions
def proposition1 : Prop := ∀ s : Solid, has_three_identical_views s → is_cube s
def proposition2 : Prop := ∀ s : Solid, front_view_is_rectangle s ∧ top_view_is_rectangle s → is_cuboid s
def proposition3 : Prop := ∀ s : Solid, all_views_are_rectangles s → is_cuboid s
def proposition4 : Prop := ∀ s : Solid, front_view_is_isosceles_trapezoid s ∧ side_view_is_isosceles_trapezoid s → is_frustum s

-- Theorem statement
theorem only_one_proposition_is_true : 
  (¬proposition1 ∧ ¬proposition2 ∧ proposition3 ∧ ¬proposition4) ∧
  (proposition1 → (¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4)) ∧
  (proposition2 → (¬proposition1 ∧ ¬proposition3 ∧ ¬proposition4)) ∧
  (proposition4 → (¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3)) :=
sorry

end NUMINAMATH_CALUDE_only_one_proposition_is_true_l1025_102541


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l1025_102555

theorem book_arrangement_theorem :
  let total_books : ℕ := 7
  let identical_books : ℕ := 3
  let different_books : ℕ := 4
  (total_books = identical_books + different_books) →
  (Nat.factorial total_books / Nat.factorial identical_books = 840) := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l1025_102555


namespace NUMINAMATH_CALUDE_tiles_required_to_cover_floor_l1025_102579

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℚ
  width : ℚ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℚ :=
  d.length * d.width

/-- Converts inches to feet -/
def inchesToFeet (inches : ℚ) : ℚ :=
  inches / 12

/-- The dimensions of the floor in feet -/
def floorDimensions : Dimensions :=
  { length := 12, width := 9 }

/-- The dimensions of a tile in inches -/
def tileDimensions : Dimensions :=
  { length := 8, width := 6 }

/-- Theorem stating that 324 tiles are required to cover the floor -/
theorem tiles_required_to_cover_floor :
  (area floorDimensions) / (area { length := inchesToFeet tileDimensions.length,
                                   width := inchesToFeet tileDimensions.width }) = 324 := by
  sorry

end NUMINAMATH_CALUDE_tiles_required_to_cover_floor_l1025_102579


namespace NUMINAMATH_CALUDE_cistern_fill_time_l1025_102597

/-- If a cistern can be emptied by a tap in 10 hours, and when both this tap and another tap
    are opened simultaneously the cistern gets filled in 20/3 hours, then the time it takes
    for the other tap alone to fill the cistern is 4 hours. -/
theorem cistern_fill_time (empty_rate : ℝ) (combined_fill_time : ℝ) (fill_time : ℝ) :
  empty_rate = 10 →
  combined_fill_time = 20 / 3 →
  1 / fill_time - 1 / empty_rate = 1 / combined_fill_time →
  fill_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l1025_102597


namespace NUMINAMATH_CALUDE_terror_arrangements_count_l1025_102553

/-- The number of unique arrangements of the letters in "TERROR" -/
def terror_arrangements : ℕ := 180

/-- The total number of letters in "TERROR" -/
def total_letters : ℕ := 6

/-- The number of R's in "TERROR" -/
def num_r : ℕ := 2

/-- The number of E's in "TERROR" -/
def num_e : ℕ := 2

/-- Theorem stating that the number of unique arrangements of the letters in "TERROR" is 180 -/
theorem terror_arrangements_count : 
  terror_arrangements = (Nat.factorial total_letters) / ((Nat.factorial num_r) * (Nat.factorial num_e)) :=
by sorry

end NUMINAMATH_CALUDE_terror_arrangements_count_l1025_102553


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l1025_102566

theorem rhombus_diagonal_length (area : ℝ) (ratio : ℚ) (shorter_diagonal : ℝ) : 
  area = 144 →
  ratio = 4/3 →
  shorter_diagonal = 6 * Real.sqrt 6 →
  area = (1/2) * shorter_diagonal * (ratio * shorter_diagonal) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l1025_102566


namespace NUMINAMATH_CALUDE_club_officer_selection_l1025_102574

/-- The number of ways to choose officers in a club -/
def choose_officers (total_members : ℕ) (newest_members : ℕ) : ℕ :=
  total_members * newest_members * (total_members - 2) * (total_members - 3) * (total_members - 4)

/-- Theorem: The number of ways to choose 5 officers in a club of 12 members -/
theorem club_officer_selection :
  choose_officers 12 4 = 34560 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_l1025_102574


namespace NUMINAMATH_CALUDE_fraction_exists_for_all_n_infinitely_many_n_without_fraction_l1025_102567

-- Part (a)
theorem fraction_exists_for_all_n (n : ℕ+) :
  ∃ (a b : ℤ), 0 < b ∧ b ≤ Real.sqrt n + 1 ∧ Real.sqrt n ≤ (a : ℝ) / b ∧ (a : ℝ) / b ≤ Real.sqrt (n + 1) :=
sorry

-- Part (b)
theorem infinitely_many_n_without_fraction :
  ∃ (S : Set ℕ+), Set.Infinite S ∧
  ∀ (n : ℕ+), n ∈ S →
    ∀ (a b : ℤ), (0 < b ∧ (b : ℝ) ≤ Real.sqrt n) →
      ¬(Real.sqrt n ≤ (a : ℝ) / b ∧ (a : ℝ) / b ≤ Real.sqrt (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_fraction_exists_for_all_n_infinitely_many_n_without_fraction_l1025_102567


namespace NUMINAMATH_CALUDE_ellipse_properties_l1025_102581

/-- An ellipse with specific properties -/
structure SpecificEllipse where
  foci_on_y_axis : Bool
  center_at_origin : Bool
  minor_axis_length : ℝ
  eccentricity : ℝ

/-- A line passing through a point -/
structure Line where
  point : ℝ × ℝ

/-- A point satisfying a specific condition -/
structure SpecialPoint where
  coords : ℝ × ℝ
  condition : Bool

/-- Theorem about the specific ellipse and related geometric properties -/
theorem ellipse_properties (e : SpecificEllipse) (l : Line) (m : SpecialPoint) :
  e.foci_on_y_axis ∧
  e.center_at_origin ∧
  e.minor_axis_length = 2 * Real.sqrt 3 ∧
  e.eccentricity = 1 / 2 ∧
  l.point = (0, 3) ∧
  m.coords = (2, 0) ∧
  m.condition →
  (∃ (x y : ℝ), y^2 / 4 + x^2 / 3 = 1) ∧
  (∃ (d : ℝ), 0 ≤ d ∧ d < (48 + 8 * Real.sqrt 15) / 21) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1025_102581


namespace NUMINAMATH_CALUDE_unique_solution_iff_nonzero_l1025_102589

theorem unique_solution_iff_nonzero (a : ℝ) :
  (a ≠ 0) ↔ (∃! x : ℝ, a * x = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_nonzero_l1025_102589


namespace NUMINAMATH_CALUDE_fraction_equality_l1025_102535

theorem fraction_equality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : 0 / b = b / c) (h2 : b / c = 1 / a) :
  (a + b - c) / (a - b + c) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1025_102535


namespace NUMINAMATH_CALUDE_zero_function_theorem_l1025_102550

-- Define the function type
def NonNegativeFunction := { f : ℝ → ℝ // ∀ x ≥ 0, f x ≥ 0 }

-- State the theorem
theorem zero_function_theorem (f : NonNegativeFunction) 
  (h_diff : Differentiable ℝ (fun x => f.val x))
  (h_initial : f.val 0 = 0)
  (h_deriv : ∀ x ≥ 0, (deriv f.val) (x^2) = f.val x) :
  ∀ x ≥ 0, f.val x = 0 := by
sorry

end NUMINAMATH_CALUDE_zero_function_theorem_l1025_102550


namespace NUMINAMATH_CALUDE_complex_equation_l1025_102516

theorem complex_equation : Complex.I ^ 3 + 2 * Complex.I = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_l1025_102516


namespace NUMINAMATH_CALUDE_equation_solution_l1025_102515

theorem equation_solution :
  ∃! y : ℝ, (4 * y - 2) / (5 * y - 5) = 3 / 4 ∧ y = -7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1025_102515


namespace NUMINAMATH_CALUDE_arrangements_with_pair_eq_10080_l1025_102523

/-- The number of ways to arrange n people in a line. -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange 8 people in a line where two specific individuals must always stand next to each other. -/
def arrangements_with_pair : ℕ :=
  factorial 7 * factorial 2

theorem arrangements_with_pair_eq_10080 :
  arrangements_with_pair = 10080 := by sorry

end NUMINAMATH_CALUDE_arrangements_with_pair_eq_10080_l1025_102523


namespace NUMINAMATH_CALUDE_not_p_or_q_l1025_102513

-- Define proposition p
def p : Prop := ∀ (A B C : ℝ) (sinA sinB : ℝ),
  (sinA = Real.sin A ∧ sinB = Real.sin B) →
  (A > B → sinA > sinB) ∧ ¬(sinA > sinB → A > B)

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 + 2*x + 2 ≤ 0

-- Theorem to prove
theorem not_p_or_q : ¬p ∨ q := by sorry

end NUMINAMATH_CALUDE_not_p_or_q_l1025_102513


namespace NUMINAMATH_CALUDE_terms_before_50_l1025_102529

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem terms_before_50 (a₁ a₂ : ℝ) (h₁ : a₁ = 100) (h₂ : a₂ = 95) :
  let d := a₂ - a₁
  let n := (a₁ - 50) / (-d) + 1
  ⌊n⌋ - 1 = 10 := by sorry

end NUMINAMATH_CALUDE_terms_before_50_l1025_102529


namespace NUMINAMATH_CALUDE_foil_covered_prism_width_l1025_102577

/-- The width of a foil-covered rectangular prism -/
theorem foil_covered_prism_width :
  ∀ (inner_length inner_width inner_height : ℝ),
    inner_length * inner_width * inner_height = 128 →
    inner_width = 2 * inner_length →
    inner_width = 2 * inner_height →
    ∃ (outer_width : ℝ),
      outer_width = 4 * (2 : ℝ)^(1/3) + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_foil_covered_prism_width_l1025_102577


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1025_102517

theorem quadratic_equation_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x - 2 = 0 ∧ x = 2) → 
  (b = -1 ∧ ∃ y : ℝ, y^2 + b*y - 2 = 0 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1025_102517


namespace NUMINAMATH_CALUDE_complex_magnitude_l1025_102582

theorem complex_magnitude (z : ℂ) (h : 2 + z = (2 - z) * Complex.I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1025_102582


namespace NUMINAMATH_CALUDE_solution_set_is_circle_minus_point_l1025_102596

theorem solution_set_is_circle_minus_point :
  ∀ (x y a : ℝ),
  (a * x + y = 2 * a + 3 ∧ x - a * y = a + 4) ↔
  ((x - 3)^2 + (y - 1)^2 = 5 ∧ (x, y) ≠ (2, -1)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_is_circle_minus_point_l1025_102596


namespace NUMINAMATH_CALUDE_min_colors_is_23_l1025_102575

/-- Represents a coloring arrangement for 8 boxes with 6 balls each -/
structure ColorArrangement where
  n : ℕ  -- Number of colors
  boxes : Fin 8 → Finset (Fin n)
  all_boxes_size_six : ∀ i, (boxes i).card = 6
  no_duplicate_colors : ∀ i j, i ≠ j → (boxes i ∩ boxes j).card ≤ 1

/-- The minimum number of colors needed for a valid ColorArrangement -/
def min_colors : ℕ := 23

/-- Theorem stating that 23 is the minimum number of colors needed -/
theorem min_colors_is_23 :
  (∃ arrangement : ColorArrangement, arrangement.n = min_colors) ∧
  (∀ arrangement : ColorArrangement, arrangement.n ≥ min_colors) :=
sorry

end NUMINAMATH_CALUDE_min_colors_is_23_l1025_102575


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l1025_102593

theorem fourth_root_equation_solutions :
  let f : ℝ → ℝ := λ x => (x ^ (1/4 : ℝ)) - 15 / (8 - x ^ (1/4 : ℝ))
  {x : ℝ | f x = 0} = {625, 81} := by
sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l1025_102593


namespace NUMINAMATH_CALUDE_sum_of_squares_l1025_102563

theorem sum_of_squares (a b n : ℕ) (h : ∃ k : ℕ, a^2 + 2*n*b^2 = k^2) :
  ∃ x y : ℕ, a^2 + n*b^2 = x^2 + y^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1025_102563


namespace NUMINAMATH_CALUDE_even_sum_impossible_both_odd_l1025_102507

theorem even_sum_impossible_both_odd (n m : ℤ) (h : Even (n^2 + m^2 + n*m)) : 
  ¬(Odd n ∧ Odd m) := by
  sorry

end NUMINAMATH_CALUDE_even_sum_impossible_both_odd_l1025_102507


namespace NUMINAMATH_CALUDE_no_divisible_polynomial_values_l1025_102557

theorem no_divisible_polynomial_values : ¬∃ (m n : ℤ), 
  0 < m ∧ m < n ∧ 
  (n ∣ (m^2 + m - 70)) ∧ 
  ((n + 1) ∣ ((m + 1)^2 + (m + 1) - 70)) := by
  sorry

end NUMINAMATH_CALUDE_no_divisible_polynomial_values_l1025_102557


namespace NUMINAMATH_CALUDE_line_slope_and_intercept_l1025_102573

theorem line_slope_and_intercept :
  ∀ (k b : ℝ),
  (∀ x y : ℝ, 3 * x + 2 * y + 6 = 0 ↔ y = k * x + b) →
  k = -3/2 ∧ b = -3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_and_intercept_l1025_102573


namespace NUMINAMATH_CALUDE_sqrt_three_times_sqrt_six_l1025_102506

theorem sqrt_three_times_sqrt_six : Real.sqrt 3 * Real.sqrt 6 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_times_sqrt_six_l1025_102506


namespace NUMINAMATH_CALUDE_fourth_root_over_sixth_root_of_seven_l1025_102512

theorem fourth_root_over_sixth_root_of_seven (x : ℝ) :
  (7 ^ (1/4)) / (7 ^ (1/6)) = 7 ^ (1/12) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_over_sixth_root_of_seven_l1025_102512


namespace NUMINAMATH_CALUDE_vector_subtraction_l1025_102531

/-- Given two vectors AB and AC in 2D space, prove that BC is their difference. -/
theorem vector_subtraction (AB AC : Fin 2 → ℝ) (h1 : AB = ![3, 6]) (h2 : AC = ![1, 2]) :
  AB - AC = ![-2, -4] := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1025_102531


namespace NUMINAMATH_CALUDE_fixed_point_coordinates_l1025_102527

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation (3+k)x-2y+1-k=0 passes through the point A for any real k -/
def passes_through (A : Point) : Prop :=
  ∀ k : ℝ, (3 + k) * A.x - 2 * A.y + 1 - k = 0

/-- The fixed point A that the line passes through for all k has coordinates (1, 2) -/
theorem fixed_point_coordinates : 
  ∃ A : Point, passes_through A ∧ A.x = 1 ∧ A.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_coordinates_l1025_102527


namespace NUMINAMATH_CALUDE_domain_intersection_l1025_102559

-- Define the domain of y = √(4-x²)
def domain_sqrt (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

-- Define the domain of y = ln(1-x)
def domain_ln (x : ℝ) : Prop := x < 1

-- Theorem statement
theorem domain_intersection :
  {x : ℝ | domain_sqrt x ∧ domain_ln x} = {x : ℝ | -2 ≤ x ∧ x < 1} :=
sorry

end NUMINAMATH_CALUDE_domain_intersection_l1025_102559


namespace NUMINAMATH_CALUDE_store_pricing_l1025_102502

/-- The price of a chair in dollars -/
def chair_price : ℝ := 60 - 52.5

/-- The price of a table in dollars -/
def table_price : ℝ := 52.5

/-- The price of 2 chairs and 1 table in dollars -/
def two_chairs_one_table : ℝ := 2 * chair_price + table_price

/-- The price of 1 chair and 2 tables in dollars -/
def one_chair_two_tables : ℝ := chair_price + 2 * table_price

theorem store_pricing :
  two_chairs_one_table / one_chair_two_tables = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_store_pricing_l1025_102502


namespace NUMINAMATH_CALUDE_decreasing_power_function_l1025_102576

theorem decreasing_power_function (m : ℝ) : 
  (m^2 - m - 1 > 0) ∧ (m^2 - 2*m - 3 < 0) ↔ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_power_function_l1025_102576


namespace NUMINAMATH_CALUDE_coordinates_product_l1025_102588

/-- Given points A and M, where M is one-third of the way from A to B, 
    prove that the product of B's coordinates is -85 -/
theorem coordinates_product (A M : ℝ × ℝ) (h1 : A = (4, 2)) (h2 : M = (1, 7)) : 
  let B := (3 * M.1 - 2 * A.1, 3 * M.2 - 2 * A.2)
  B.1 * B.2 = -85 := by sorry

end NUMINAMATH_CALUDE_coordinates_product_l1025_102588


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l1025_102510

theorem consecutive_odd_integers_sum (a : ℤ) : 
  (∃ (b c : ℤ), b = a + 2 ∧ c = a + 4 ∧ a + c = 100) →
  a + (a + 2) + (a + 4) = 150 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l1025_102510


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l1025_102547

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 - 6 * x + m = 0) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l1025_102547


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l1025_102504

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem extreme_value_implies_a_equals_5 (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -3 ∧ |x + 3| < ε → f a x ≤ f a (-3) ∨ f a x ≥ f a (-3)) →
  a = 5 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l1025_102504


namespace NUMINAMATH_CALUDE_toy_ratio_after_removal_l1025_102552

/-- Proves that given 134 total toys, with 90 initially red, after removing 2 red toys, 
    the ratio of red to white toys is 2:1. -/
theorem toy_ratio_after_removal (total : ℕ) (initial_red : ℕ) (removed : ℕ) : 
  total = 134 → initial_red = 90 → removed = 2 →
  (initial_red - removed) / (total - initial_red) = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_toy_ratio_after_removal_l1025_102552


namespace NUMINAMATH_CALUDE_room_length_l1025_102565

/-- Given a rectangular room with width 4 meters and a paving cost of 950 per square meter
    resulting in a total cost of 20900, the length of the room is 5.5 meters. -/
theorem room_length (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (length : ℝ) : 
  width = 4 →
  cost_per_sqm = 950 →
  total_cost = 20900 →
  total_cost = cost_per_sqm * (length * width) →
  length = 5.5 := by
sorry


end NUMINAMATH_CALUDE_room_length_l1025_102565


namespace NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_is_tight_l1025_102505

theorem min_value_of_expression (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 9 * Real.sqrt 5 / 5 := by sorry

theorem lower_bound_is_tight : 
  ∃ (x : ℝ), (x^2 + 9) / Real.sqrt (x^2 + 5) = 9 * Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_is_tight_l1025_102505


namespace NUMINAMATH_CALUDE_min_draws_for_pair_of_each_color_l1025_102546

/-- Represents the number of items of a given color -/
structure ColorCount where
  count : Nat

/-- Represents the box containing hats and gloves -/
structure Box where
  red : ColorCount
  green : ColorCount
  orange : ColorCount

/-- Calculates the minimum number of draws required for a given color -/
def minDrawsForColor (c : ColorCount) : Nat :=
  c.count + 1

/-- Calculates the total minimum draws required for all colors -/
def totalMinDraws (box : Box) : Nat :=
  minDrawsForColor box.red + minDrawsForColor box.green + minDrawsForColor box.orange

/-- The main theorem to be proved -/
theorem min_draws_for_pair_of_each_color (box : Box) 
  (h_red : box.red.count = 41)
  (h_green : box.green.count = 23)
  (h_orange : box.orange.count = 11) :
  totalMinDraws box = 78 := by
  sorry

#eval totalMinDraws { red := { count := 41 }, green := { count := 23 }, orange := { count := 11 } }

end NUMINAMATH_CALUDE_min_draws_for_pair_of_each_color_l1025_102546


namespace NUMINAMATH_CALUDE_team_size_l1025_102569

theorem team_size (average_age : ℝ) (leader_age : ℝ) (average_age_without_leader : ℝ) 
  (h1 : average_age = 25)
  (h2 : leader_age = 45)
  (h3 : average_age_without_leader = 23) :
  ∃ n : ℕ, n * average_age = (n - 1) * average_age_without_leader + leader_age ∧ n = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_team_size_l1025_102569


namespace NUMINAMATH_CALUDE_sine_tangent_comparison_l1025_102598

open Real

theorem sine_tangent_comparison (α : ℝ) (h : 0 < α ∧ α < π / 2) : 
  sin α < tan α ∧ (deriv sin) α < (deriv tan) α := by sorry

end NUMINAMATH_CALUDE_sine_tangent_comparison_l1025_102598


namespace NUMINAMATH_CALUDE_number_of_payment_ways_l1025_102543

/-- Represents the number of ways to pay 16 rubles using 10-ruble, 2-ruble, and 1-ruble coins. -/
def payment_ways : ℕ := 13

/-- Represents the total amount to be paid in rubles. -/
def total_amount : ℕ := 16

/-- Represents the value of a 10-ruble coin. -/
def ten_ruble : ℕ := 10

/-- Represents the value of a 2-ruble coin. -/
def two_ruble : ℕ := 2

/-- Represents the value of a 1-ruble coin. -/
def one_ruble : ℕ := 1

/-- Represents the minimum number of coins of each type available. -/
def min_coins : ℕ := 21

/-- Theorem stating that the number of ways to pay 16 rubles is 13. -/
theorem number_of_payment_ways :
  payment_ways = (Finset.filter
    (fun n : ℕ × ℕ × ℕ => n.1 * ten_ruble + n.2.1 * two_ruble + n.2.2 * one_ruble = total_amount)
    (Finset.product (Finset.range (min_coins + 1))
      (Finset.product (Finset.range (min_coins + 1)) (Finset.range (min_coins + 1))))).card :=
by sorry

end NUMINAMATH_CALUDE_number_of_payment_ways_l1025_102543


namespace NUMINAMATH_CALUDE_sum_of_abs_values_l1025_102536

theorem sum_of_abs_values (a b : ℝ) : 
  (|a| = 3 ∧ |b| = 5 ∧ a > b) → (a + b = -2 ∨ a + b = -8) := by
sorry

end NUMINAMATH_CALUDE_sum_of_abs_values_l1025_102536


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1025_102538

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 120 -/
def SequenceSum (a : ℕ → ℚ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property
  (a : ℕ → ℚ)
  (h1 : ArithmeticSequence a)
  (h2 : SequenceSum a) :
  a 9 - (1/3) * a 11 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1025_102538


namespace NUMINAMATH_CALUDE_jean_speed_is_45_over_46_l1025_102585

/-- Represents the hiking scenario with Chantal and Jean --/
structure HikingScenario where
  speed_first_third : ℝ
  speed_uphill : ℝ
  break_time : ℝ
  speed_downhill : ℝ
  meeting_point : ℝ

/-- Calculates Jean's average speed given a hiking scenario --/
def jeanAverageSpeed (scenario : HikingScenario) : ℝ :=
  sorry

/-- Theorem stating that Jean's average speed is 45/46 miles per hour --/
theorem jean_speed_is_45_over_46 (scenario : HikingScenario) :
  scenario.speed_first_third = 5 ∧
  scenario.speed_uphill = 3 ∧
  scenario.break_time = 1/6 ∧
  scenario.speed_downhill = 4 ∧
  scenario.meeting_point = 3/2 →
  jeanAverageSpeed scenario = 45/46 :=
sorry

end NUMINAMATH_CALUDE_jean_speed_is_45_over_46_l1025_102585


namespace NUMINAMATH_CALUDE_expression_value_l1025_102551

theorem expression_value : (0.5 : ℝ)^3 - (0.1 : ℝ)^3 / (0.5 : ℝ)^2 + 0.05 + (0.1 : ℝ)^2 = 0.181 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1025_102551


namespace NUMINAMATH_CALUDE_cos_2018pi_minus_pi_sixth_l1025_102583

theorem cos_2018pi_minus_pi_sixth : 
  Real.cos (2018 * Real.pi - Real.pi / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_2018pi_minus_pi_sixth_l1025_102583


namespace NUMINAMATH_CALUDE_farmer_apples_l1025_102520

/-- The number of apples the farmer gave away -/
def apples_given_away : ℕ := 88

/-- The number of apples the farmer has left -/
def apples_left : ℕ := 39

/-- The initial number of apples the farmer had -/
def initial_apples : ℕ := apples_given_away + apples_left

theorem farmer_apples : initial_apples = 127 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l1025_102520


namespace NUMINAMATH_CALUDE_quadratic_roots_and_integer_k_l1025_102534

/-- Represents a quadratic equation of the form kx^2 + (k-2)x - 2 = 0 --/
def QuadraticEquation (k : ℝ) : ℝ → Prop :=
  fun x => k * x^2 + (k - 2) * x - 2 = 0

theorem quadratic_roots_and_integer_k :
  ∀ k : ℝ, k ≠ 0 →
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ QuadraticEquation k x₁ ∧ QuadraticEquation k x₂) ∧
    (∃ k' : ℤ, k' ∈ ({-2, -1, 1, 2} : Set ℤ) ∧
      ∃ x₁ x₂ : ℤ, QuadraticEquation (k' : ℝ) x₁ ∧ QuadraticEquation (k' : ℝ) x₂) :=
by sorry

#check quadratic_roots_and_integer_k

end NUMINAMATH_CALUDE_quadratic_roots_and_integer_k_l1025_102534


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1025_102542

theorem min_value_sum_reciprocals (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_two : a + b + c + d = 2) :
  (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1025_102542


namespace NUMINAMATH_CALUDE_log_8_1000_equals_inverse_log_10_2_l1025_102519

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define log_10 as the natural logarithm divided by ln(10)
noncomputable def log_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_8_1000_equals_inverse_log_10_2 :
  log 8 1000 = 1 / log_10 2 := by
  sorry

end NUMINAMATH_CALUDE_log_8_1000_equals_inverse_log_10_2_l1025_102519


namespace NUMINAMATH_CALUDE_g_of_g_is_even_l1025_102570

def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

theorem g_of_g_is_even (g : ℝ → ℝ) (h : is_even_function g) : is_even_function (g ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_g_of_g_is_even_l1025_102570


namespace NUMINAMATH_CALUDE_solve_for_i_l1025_102549

-- Define the equation as a function of x and i
def equation (x i : ℝ) : Prop :=
  (10 * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / i

-- State the theorem
theorem solve_for_i :
  ∃ i : ℝ, equation 0.3 i ∧ abs (i - 2.9993) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_i_l1025_102549


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l1025_102544

theorem triangle_ABC_properties (A B C : Real) (a b c : Real) :
  -- Given conditions
  (a * Real.cos B + b * Real.cos A = -2 * c * Real.cos C) →
  (c = Real.sqrt 7) →
  (b = 2) →
  -- Conclusions
  (C = 2 * Real.pi / 3) ∧
  (1/2 * a * b * Real.sin C = Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l1025_102544


namespace NUMINAMATH_CALUDE_sum_of_m_values_l1025_102561

theorem sum_of_m_values (x y z m : ℝ) : 
  x ≠ y ∧ y ≠ z ∧ x ≠ z →
  x / (2 - y) = m ∧ y / (2 - z) = m ∧ z / (2 - x) = m →
  ∃ m₁ m₂ : ℝ, m₁ + m₂ = 2 ∧ (∀ m' : ℝ, m' = m₁ ∨ m' = m₂ ↔ 
    x / (2 - y) = m' ∧ y / (2 - z) = m' ∧ z / (2 - x) = m') :=
by sorry

end NUMINAMATH_CALUDE_sum_of_m_values_l1025_102561


namespace NUMINAMATH_CALUDE_distinct_triangles_count_l1025_102558

/-- The maximum exponent for the line segment lengths -/
def max_exponent : ℕ := 10

/-- The set of line segment lengths -/
def segment_lengths : Set ℕ := {n | ∃ k : ℕ, 0 ≤ k ∧ k ≤ max_exponent ∧ n = 2^k}

/-- A function to check if three lengths can form a nondegenerate triangle -/
def is_nondegenerate_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The number of distinct nondegenerate triangles -/
def num_distinct_triangles : ℕ := Nat.choose (max_exponent + 1) 2

theorem distinct_triangles_count :
  num_distinct_triangles = 55 := by sorry

end NUMINAMATH_CALUDE_distinct_triangles_count_l1025_102558


namespace NUMINAMATH_CALUDE_sum_of_integers_l1025_102556

theorem sum_of_integers (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t →
  (9 - p) * (9 - q) * (9 - r) * (9 - s) * (9 - t) = -120 →
  p + q + r + s + t = 32 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1025_102556


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1025_102568

-- Define the hyperbola
structure Hyperbola where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  passes_through : ℝ × ℝ  -- point that the hyperbola passes through

-- Define the standard equation of a hyperbola
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

-- Theorem statement
theorem hyperbola_equation (h : Hyperbola) :
  h.a = 2 * Real.sqrt 5 →
  h.passes_through = (5, -2) →
  ∀ x y : ℝ, standard_equation h x y ↔ x^2 / 20 - y^2 / 16 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1025_102568


namespace NUMINAMATH_CALUDE_fuel_for_three_trips_l1025_102560

/-- Calculates the total fuel needed for a series of trips given a fuel consumption rate -/
def totalFuelNeeded (fuelRate : ℝ) (trips : List ℝ) : ℝ :=
  fuelRate * (trips.sum)

/-- Proves that the total fuel needed for three specific trips is 550 liters -/
theorem fuel_for_three_trips :
  let fuelRate : ℝ := 5
  let trips : List ℝ := [50, 35, 25]
  totalFuelNeeded fuelRate trips = 550 := by
  sorry

#check fuel_for_three_trips

end NUMINAMATH_CALUDE_fuel_for_three_trips_l1025_102560


namespace NUMINAMATH_CALUDE_total_children_l1025_102599

theorem total_children (happy : ℕ) (sad : ℕ) (neutral : ℕ) 
  (boys : ℕ) (girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ) (neutral_boys : ℕ)
  (h1 : happy = 30)
  (h2 : sad = 10)
  (h3 : neutral = 20)
  (h4 : boys = 16)
  (h5 : girls = 44)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neutral_boys = 4) :
  boys + girls = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_children_l1025_102599


namespace NUMINAMATH_CALUDE_point_y_coordinate_l1025_102539

/-- A straight line in the xy-plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given line has slope 2 and y-intercept 2 -/
def given_line : Line :=
  { slope := 2, y_intercept := 2 }

/-- The x-coordinate of the point in question is 239 -/
def given_x : ℝ := 239

/-- A point is on a line if its coordinates satisfy the line equation -/
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

/-- Theorem: The point on the given line with x-coordinate 239 has y-coordinate 480 -/
theorem point_y_coordinate :
  ∃ p : Point, p.x = given_x ∧ point_on_line p given_line ∧ p.y = 480 :=
sorry

end NUMINAMATH_CALUDE_point_y_coordinate_l1025_102539


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1025_102586

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) :
  let side := Real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)
  4 * side = 40 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1025_102586


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1025_102591

theorem sin_cos_identity : 
  Real.sin (75 * π / 180) * Real.cos (30 * π / 180) - 
  Real.sin (15 * π / 180) * Real.sin (150 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1025_102591


namespace NUMINAMATH_CALUDE_tea_mixture_price_l1025_102592

theorem tea_mixture_price (price1 price2 price3 mixture_price : ℝ) 
  (h1 : price1 = 126)
  (h2 : price3 = 175.5)
  (h3 : mixture_price = 153)
  (h4 : price1 + price2 + 2 * price3 = 4 * mixture_price) :
  price2 = 135 := by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l1025_102592
