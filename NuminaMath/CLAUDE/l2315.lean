import Mathlib

namespace solve_for_d_l2315_231526

theorem solve_for_d (x d : ℝ) (h1 : x = 0.3) 
  (h2 : (10 * x + 2) / 4 - (d * x - 6) / 18 = (2 * x + 4) / 3) : d = 3 := by
  sorry

end solve_for_d_l2315_231526


namespace sibling_ages_sum_l2315_231541

theorem sibling_ages_sum (a b : ℕ+) : 
  a < b → 
  a * b * b * b = 216 → 
  a + b + b + b = 19 := by
sorry

end sibling_ages_sum_l2315_231541


namespace f_symmetry_l2315_231525

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2 + 3 * Real.sin x + 2

theorem f_symmetry (a : ℝ) (h : f a = 1) : f (-a) = 3 := by
  sorry

end f_symmetry_l2315_231525


namespace distinct_values_x9_mod_999_l2315_231571

theorem distinct_values_x9_mod_999 : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, n < 999) ∧ 
  (∀ x : ℕ, ∃ n ∈ S, x^9 ≡ n [ZMOD 999]) ∧
  Finset.card S = 15 :=
sorry

end distinct_values_x9_mod_999_l2315_231571


namespace right_triangle_existence_l2315_231500

/-- A right triangle with hypotenuse c and angle bisector f of the right angle -/
structure RightTriangle where
  c : ℝ  -- Length of hypotenuse
  f : ℝ  -- Length of angle bisector of right angle
  c_pos : c > 0
  f_pos : f > 0

/-- The condition for the existence of a right triangle given its hypotenuse and angle bisector -/
def constructible (t : RightTriangle) : Prop :=
  t.f < t.c / 2

/-- Theorem stating the condition for the existence of a right triangle -/
theorem right_triangle_existence (t : RightTriangle) :
  constructible t ↔ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = t.c^2 ∧ 
    t.f = (a * b) / (a + b) :=
sorry

end right_triangle_existence_l2315_231500


namespace dress_discount_price_l2315_231554

theorem dress_discount_price (d : ℝ) (h : d > 0) : 
  d * (1 - 0.45) * (1 - 0.4) = d * 0.33 := by
sorry

end dress_discount_price_l2315_231554


namespace nails_for_smaller_planks_eq_eight_l2315_231597

/-- The number of large planks used for the walls -/
def large_planks : ℕ := 13

/-- The number of nails needed for each large plank -/
def nails_per_plank : ℕ := 17

/-- The total number of nails needed for the house wall -/
def total_nails : ℕ := 229

/-- The number of nails needed for smaller planks -/
def nails_for_smaller_planks : ℕ := total_nails - (large_planks * nails_per_plank)

theorem nails_for_smaller_planks_eq_eight :
  nails_for_smaller_planks = 8 := by
  sorry

end nails_for_smaller_planks_eq_eight_l2315_231597


namespace isosceles_right_triangle_area_l2315_231550

theorem isosceles_right_triangle_area 
  (h : ℝ) -- hypotenuse length
  (is_isosceles_right : True) -- condition that the triangle is isosceles right
  (hyp_length : h = 6 * Real.sqrt 2) : -- condition for the hypotenuse length
  (1/2) * ((h / Real.sqrt 2) ^ 2) = 18 := by
sorry

end isosceles_right_triangle_area_l2315_231550


namespace max_product_of_roots_l2315_231511

/-- Given a quadratic equation 5x^2 - 10x + m = 0 with real roots,
    the maximum value of the product of its roots is 1. -/
theorem max_product_of_roots :
  ∀ m : ℝ,
  (∃ x : ℝ, 5 * x^2 - 10 * x + m = 0) →
  (∀ k : ℝ, (∃ x : ℝ, 5 * x^2 - 10 * x + k = 0) → m / 5 ≥ k / 5) →
  m / 5 = 1 :=
by sorry

end max_product_of_roots_l2315_231511


namespace range_of_a_l2315_231528

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → x^3 - a*x + 1 ≥ 0) → 
  0 ≤ a ∧ a ≤ 3 * (2 : ℝ)^(1/3) / 2 :=
sorry

end range_of_a_l2315_231528


namespace a_range_l2315_231549

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

theorem a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) →
  (3/2 ≤ a ∧ a < 2) :=
sorry

end a_range_l2315_231549


namespace parking_lot_car_ratio_l2315_231544

theorem parking_lot_car_ratio :
  let red_cars : ℕ := 28
  let black_cars : ℕ := 75
  (red_cars : ℚ) / black_cars = 28 / 75 :=
by sorry

end parking_lot_car_ratio_l2315_231544


namespace village_population_percentage_l2315_231534

theorem village_population_percentage : 
  let total_population : ℕ := 25600
  let part_population : ℕ := 23040
  (part_population : ℚ) / total_population * 100 = 90 := by sorry

end village_population_percentage_l2315_231534


namespace subtract_squared_terms_l2315_231546

theorem subtract_squared_terms (x : ℝ) : 3 * x^2 - 4 * x^2 = -x^2 := by
  sorry

end subtract_squared_terms_l2315_231546


namespace mary_coins_l2315_231548

theorem mary_coins (dimes quarters : ℕ) 
  (h1 : quarters = 2 * dimes + 7)
  (h2 : (0.10 : ℚ) * dimes + (0.25 : ℚ) * quarters = 10.15) : quarters = 35 := by
  sorry

end mary_coins_l2315_231548


namespace system_solution_unique_equation_no_solution_l2315_231584

-- Problem 1
theorem system_solution_unique (x y : ℝ) : 
  x - 3*y = 4 ∧ 2*x - y = 3 ↔ x = 1 ∧ y = -1 :=
sorry

-- Problem 2
theorem equation_no_solution : 
  ¬∃ x : ℝ, (x ≠ 2) ∧ (1 / (x - 2) + 3 = (1 - x) / (2 - x)) :=
sorry

end system_solution_unique_equation_no_solution_l2315_231584


namespace math_only_students_l2315_231578

theorem math_only_students (total : ℕ) (math foreign_lang science : Finset ℕ) :
  total = 120 →
  (∀ s, s ∈ math ∪ foreign_lang ∪ science) →
  math.card = 85 →
  foreign_lang.card = 65 →
  science.card = 50 →
  (math ∩ foreign_lang ∩ science).card = 20 →
  (math \ (foreign_lang ∪ science)).card = 52 :=
by sorry

end math_only_students_l2315_231578


namespace alicia_singles_stats_l2315_231530

/-- Represents a baseball player's hit statistics -/
structure HitStats where
  total : ℕ
  homeRuns : ℕ
  triples : ℕ
  doubles : ℕ

/-- Calculates the number of singles and their percentage of total hits -/
def singlesStats (stats : HitStats) : (ℕ × ℚ) :=
  let singles := stats.total - (stats.homeRuns + stats.triples + stats.doubles)
  let percentage := (singles : ℚ) / (stats.total : ℚ) * 100
  (singles, percentage)

/-- Theorem: Given Alicia's hit statistics, prove that she had 38 singles
    which constitute 76% of her total hits -/
theorem alicia_singles_stats :
  let alicia : HitStats := ⟨50, 2, 3, 7⟩
  singlesStats alicia = (38, 76) := by sorry

end alicia_singles_stats_l2315_231530


namespace x_value_when_one_in_set_l2315_231507

theorem x_value_when_one_in_set (x : ℝ) : 
  (1 ∈ ({x, x^2} : Set ℝ)) → x ≠ x^2 → x = -1 := by
  sorry

end x_value_when_one_in_set_l2315_231507


namespace sum_of_occurrences_l2315_231510

theorem sum_of_occurrences (a₀ a₁ a₂ a₃ a₄ : ℕ) 
  (sum_constraint : a₀ + a₁ + a₂ + a₃ + a₄ = 5)
  (value_constraint : 0*a₀ + 1*a₁ + 2*a₂ + 3*a₃ + 4*a₄ = 5) :
  a₀ + a₁ + a₂ + a₃ = 5 := by
  sorry

end sum_of_occurrences_l2315_231510


namespace integer_decimal_parts_sum_l2315_231536

theorem integer_decimal_parts_sum (m n : ℝ) : 
  (∃ k : ℤ, 7 + Real.sqrt 13 = k + m ∧ k ≤ 7 + Real.sqrt 13 ∧ 7 + Real.sqrt 13 < k + 1) →
  (∃ j : ℤ, Real.sqrt 13 = j + n ∧ j ≤ Real.sqrt 13 ∧ Real.sqrt 13 < j + 1) →
  m + n = 7 + Real.sqrt 13 :=
by sorry

end integer_decimal_parts_sum_l2315_231536


namespace hayley_sticker_distribution_l2315_231519

def distribute_stickers (total_stickers : ℕ) (num_friends : ℕ) : ℕ :=
  total_stickers / num_friends

theorem hayley_sticker_distribution :
  let total_stickers : ℕ := 72
  let num_friends : ℕ := 9
  distribute_stickers total_stickers num_friends = 8 := by sorry

end hayley_sticker_distribution_l2315_231519


namespace smaller_pyramid_volume_l2315_231596

/-- The volume of a smaller pyramid cut from a right rectangular pyramid -/
theorem smaller_pyramid_volume
  (base_length : ℝ) (base_width : ℝ) (slant_edge : ℝ) (cut_height : ℝ)
  (h_base_length : base_length = 10 * Real.sqrt 2)
  (h_base_width : base_width = 6 * Real.sqrt 2)
  (h_slant_edge : slant_edge = 12)
  (h_cut_height : cut_height = 4) :
  ∃ (volume : ℝ),
    volume = 20 * ((2 * Real.sqrt 19 - 4) / (2 * Real.sqrt 19))^3 * (2 * Real.sqrt 19 - 4) :=
by sorry

end smaller_pyramid_volume_l2315_231596


namespace smallest_square_area_l2315_231582

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum side length of a square that can contain two rectangles,
    one of which is rotated 90 degrees -/
def minSquareSide (r1 r2 : Rectangle) : ℕ :=
  max (max r1.width r1.height) (max r2.width r2.height)

/-- Theorem stating the smallest possible area of the square -/
theorem smallest_square_area (r1 r2 : Rectangle)
  (h1 : r1 = ⟨4, 2⟩ ∨ r1 = ⟨2, 4⟩)
  (h2 : r2 = ⟨5, 3⟩ ∨ r2 = ⟨3, 5⟩) :
  (minSquareSide r1 r2) ^ 2 = 25 := by
  sorry

end smallest_square_area_l2315_231582


namespace necklace_packing_condition_l2315_231552

/-- Represents a necklace of cubes -/
structure CubeNecklace where
  n : ℕ
  numCubes : ℕ
  isLooped : Bool

/-- Represents a cubic box -/
structure CubicBox where
  edgeLength : ℕ

/-- Predicate to check if a necklace can be packed into a box -/
def canBePacked (necklace : CubeNecklace) (box : CubicBox) : Prop :=
  necklace.numCubes = box.edgeLength ^ 3 ∧
  necklace.isLooped = true

/-- Theorem stating the condition for packing the necklace -/
theorem necklace_packing_condition (n : ℕ) :
  let necklace := CubeNecklace.mk n (n^3) true
  let box := CubicBox.mk n
  canBePacked necklace box ↔ Even n :=
sorry

end necklace_packing_condition_l2315_231552


namespace fib_70_mod_10_l2315_231595

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Fibonacci sequence modulo 10 -/
def fibMod10 (n : ℕ) : ℕ := fib n % 10

/-- Period of Fibonacci sequence modulo 10 -/
def fibMod10Period : ℕ := 60

theorem fib_70_mod_10 :
  fibMod10 70 = 5 := by sorry

end fib_70_mod_10_l2315_231595


namespace count_six_digit_numbers_with_at_least_two_zeros_l2315_231501

/-- The number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers with no zeros -/
def numbers_with_no_zeros : ℕ := 531441

/-- The number of 6-digit numbers with exactly one zero -/
def numbers_with_one_zero : ℕ := 295245

/-- The number of 6-digit numbers with at least two zeros -/
def numbers_with_at_least_two_zeros : ℕ := 
  total_six_digit_numbers - (numbers_with_no_zeros + numbers_with_one_zero)

theorem count_six_digit_numbers_with_at_least_two_zeros : 
  numbers_with_at_least_two_zeros = 73314 := by
  sorry

end count_six_digit_numbers_with_at_least_two_zeros_l2315_231501


namespace smallest_fourth_number_l2315_231524

/-- Given three two-digit numbers and a fourth unknown two-digit number,
    if the sum of the digits of all four numbers is 1/4 of their total sum,
    then the smallest possible value for the unknown number is 70. -/
theorem smallest_fourth_number (x : ℕ) :
  x ≥ 10 ∧ x < 100 →
  (34 + 21 + 63 + x : ℕ) = 4 * ((3 + 4 + 2 + 1 + 6 + 3 + (x / 10) + (x % 10)) : ℕ) →
  ∀ y : ℕ, y ≥ 10 ∧ y < 100 →
    (34 + 21 + 63 + y : ℕ) = 4 * ((3 + 4 + 2 + 1 + 6 + 3 + (y / 10) + (y % 10)) : ℕ) →
    x ≤ y →
  x = 70 :=
by sorry

end smallest_fourth_number_l2315_231524


namespace martha_butterflies_l2315_231504

theorem martha_butterflies (total : ℕ) (blue : ℕ) (yellow : ℕ) (black : ℕ) : 
  total = 19 → 
  blue = 2 * yellow → 
  blue = 6 → 
  black = total - (blue + yellow) → 
  black = 10 := by
sorry

end martha_butterflies_l2315_231504


namespace abie_chips_count_l2315_231563

theorem abie_chips_count (initial : Nat) (given : Nat) (bought : Nat) (final : Nat) : 
  initial = 20 → given = 4 → bought = 6 → final = initial - given + bought → final = 22 := by
  sorry

end abie_chips_count_l2315_231563


namespace team_b_size_l2315_231575

/-- Proves that Team B has 9 people given the competition conditions -/
theorem team_b_size (team_a_avg : ℝ) (team_b_avg : ℝ) (total_avg : ℝ) (size_diff : ℕ) :
  team_a_avg = 75 →
  team_b_avg = 73 →
  total_avg = 73.5 →
  size_diff = 6 →
  ∃ (x : ℕ), x + size_diff = 9 ∧
    (team_a_avg * x + team_b_avg * (x + size_diff)) / (x + (x + size_diff)) = total_avg :=
by
  sorry

#check team_b_size

end team_b_size_l2315_231575


namespace tree_planting_activity_l2315_231570

theorem tree_planting_activity (boys girls : ℕ) : 
  (boys = 2 * girls + 15) →
  (girls = boys / 3 + 6) →
  (boys = 81 ∧ girls = 33) :=
by sorry

end tree_planting_activity_l2315_231570


namespace adjacent_i_probability_l2315_231506

theorem adjacent_i_probability : 
  let total_letters : ℕ := 10
  let unique_letters : ℕ := 9
  let repeated_letter : ℕ := 1
  let favorable_arrangements : ℕ := unique_letters.factorial
  let total_arrangements : ℕ := total_letters.factorial / (repeated_letter + 1).factorial
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 5 := by
sorry

end adjacent_i_probability_l2315_231506


namespace min_value_theorem_l2315_231579

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : Real.log (a + b) = 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → Real.log (x + y) = 0 → 2/x + 3/y ≥ 5 + 2 * Real.sqrt 6) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ Real.log (x + y) = 0 ∧ 2/x + 3/y = 5 + 2 * Real.sqrt 6) :=
by sorry

end min_value_theorem_l2315_231579


namespace perpendicular_nonzero_vectors_exist_l2315_231565

theorem perpendicular_nonzero_vectors_exist :
  ∃ (a b : ℝ × ℝ), a ≠ (0, 0) ∧ b ≠ (0, 0) ∧ a.1 * b.1 + a.2 * b.2 = 0 :=
by
  sorry

end perpendicular_nonzero_vectors_exist_l2315_231565


namespace trigonometric_identities_l2315_231573

theorem trigonometric_identities :
  (Real.cos (2 * Real.pi / 5) - Real.cos (4 * Real.pi / 5) = Real.sqrt 5 / 2) ∧
  (Real.sin (2 * Real.pi / 7) + Real.sin (4 * Real.pi / 7) - Real.sin (6 * Real.pi / 7) = Real.sqrt 7 / 2) := by
  sorry

end trigonometric_identities_l2315_231573


namespace inverse_composition_l2315_231533

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv g_inv : ℝ → ℝ)

-- State the given condition
axiom condition : ∀ x, f_inv (g x) = 5 * x + 3

-- State the theorem to be proved
theorem inverse_composition : g_inv (f (-7)) = -2 := by sorry

end inverse_composition_l2315_231533


namespace sum_equals_350_l2315_231529

theorem sum_equals_350 : 124 + 129 + 106 + 141 + 237 - 500 + 113 = 350 := by
  sorry

end sum_equals_350_l2315_231529


namespace exists_correct_coloring_l2315_231518

/-- Represents the color of a square on the board -/
inductive Color
| White
| Black

/-- Represents a position on the board -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the game board -/
def Board := Position → Color

/-- Checks if two positions are adjacent -/
def adjacent (p1 p2 : Position) : Bool :=
  (p1.row = p2.row ∧ (p1.col + 1 = p2.col ∨ p2.col + 1 = p1.col)) ∨
  (p1.col = p2.col ∧ (p1.row + 1 = p2.row ∨ p2.row + 1 = p1.row))

/-- Checks if a position is within the 4x8 board -/
def validPosition (p : Position) : Bool :=
  p.row < 4 ∧ p.col < 8

/-- Inverts the color -/
def invertColor (c : Color) : Color :=
  match c with
  | Color.White => Color.Black
  | Color.Black => Color.White

/-- Applies a move to the board -/
def applyMove (board : Board) (topLeft : Position) : Board :=
  λ p => if p.row ∈ [topLeft.row, topLeft.row + 1] ∧ 
            p.col ∈ [topLeft.col, topLeft.col + 1]
         then invertColor (board p)
         else board p

/-- Checks if the board is correctly colored -/
def isCorrectlyColored (board : Board) : Prop :=
  ∀ p1 p2, validPosition p1 ∧ validPosition p2 ∧ adjacent p1 p2 →
    board p1 ≠ board p2

/-- The main theorem to prove -/
theorem exists_correct_coloring :
  ∃ (finalBoard : Board),
    (∃ (moves : List Position), 
      finalBoard = (moves.foldl applyMove (λ _ => Color.White)) ∧
      isCorrectlyColored finalBoard) :=
sorry

end exists_correct_coloring_l2315_231518


namespace time_after_minutes_l2315_231512

def minutes_after_midnight : ℕ := 2345

def hours_in_day : ℕ := 24

def minutes_in_hour : ℕ := 60

def start_date : String := "January 1, 2022"

theorem time_after_minutes (m : ℕ) (h : m = minutes_after_midnight) :
  (start_date, m) = ("January 2", 15 * minutes_in_hour + 5) := by sorry

end time_after_minutes_l2315_231512


namespace trapezoid_area_theorem_l2315_231543

/-- Represents a trapezoid with diagonals and sum of bases -/
structure Trapezoid where
  diagonal1 : ℝ
  diagonal2 : ℝ
  sum_of_bases : ℝ

/-- Calculates the area of a trapezoid given its diagonals and sum of bases -/
def area (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating that a trapezoid with diagonals 12 and 6, and sum of bases 14, has an area of 16√5 -/
theorem trapezoid_area_theorem :
  let t : Trapezoid := { diagonal1 := 12, diagonal2 := 6, sum_of_bases := 14 }
  area t = 16 * Real.sqrt 5 := by
  sorry

end trapezoid_area_theorem_l2315_231543


namespace sandys_age_l2315_231557

/-- Given that Molly is 16 years older than Sandy and their ages are in the ratio 7:9, prove that Sandy is 56 years old. -/
theorem sandys_age (sandy_age molly_age : ℕ) : 
  molly_age = sandy_age + 16 →
  sandy_age * 9 = molly_age * 7 →
  sandy_age = 56 := by
  sorry

end sandys_age_l2315_231557


namespace matrix_equation_proof_l2315_231559

theorem matrix_equation_proof : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-21, -2; 13, 1]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![71/14, -109/14; -43/14, 67/14]
  N * A = B := by sorry

end matrix_equation_proof_l2315_231559


namespace f_value_at_2_l2315_231581

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem f_value_at_2 : 
  (∀ x : ℝ, f (2 * x + 1) = x^2 - 2*x) → f 2 = -3/4 := by sorry

end f_value_at_2_l2315_231581


namespace correct_loan_amounts_l2315_231589

/-- Represents the loan amounts and interest rates for a company's two types of loans. -/
structure LoanInfo where
  typeA : ℝ  -- Amount of Type A loan in yuan
  typeB : ℝ  -- Amount of Type B loan in yuan
  rateA : ℝ  -- Annual interest rate for Type A loan
  rateB : ℝ  -- Annual interest rate for Type B loan

/-- Theorem stating the correct loan amounts given the problem conditions. -/
theorem correct_loan_amounts (loan : LoanInfo) : 
  loan.typeA = 200000 ∧ loan.typeB = 300000 ↔ 
  loan.typeA + loan.typeB = 500000 ∧ 
  loan.rateA * loan.typeA + loan.rateB * loan.typeB = 44000 ∧
  loan.rateA = 0.1 ∧ 
  loan.rateB = 0.08 := by
  sorry

end correct_loan_amounts_l2315_231589


namespace max_product_l2315_231586

def digits : List ℕ := [1, 3, 5, 8, 9]

def is_valid_combination (a b c d e : ℕ) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def product (a b c d e : ℕ) : ℕ := (100 * a + 10 * b + c) * (10 * d + e)

theorem max_product :
  ∀ a b c d e : ℕ,
    is_valid_combination a b c d e →
    product a b c d e ≤ product 9 3 1 8 5 :=
by sorry

end max_product_l2315_231586


namespace angle_triple_supplement_l2315_231514

theorem angle_triple_supplement (x : ℝ) : 
  (x = 3 * (180 - x)) → x = 135 := by sorry

end angle_triple_supplement_l2315_231514


namespace negation_of_existential_proposition_l2315_231553

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^3 > x) ↔ (∀ x : ℝ, x^3 ≤ x) := by
  sorry

end negation_of_existential_proposition_l2315_231553


namespace matrix_equation_satisfied_l2315_231517

/-- The matrix M that satisfies the given equation -/
def M : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]

/-- The right-hand side matrix of the equation -/
def RHS : Matrix (Fin 2) (Fin 2) ℝ := !![10, 20; 5, 10]

/-- Theorem stating that M satisfies the given matrix equation -/
theorem matrix_equation_satisfied :
  M^3 - 4 • M^2 + 5 • M = RHS := by sorry

end matrix_equation_satisfied_l2315_231517


namespace power_sum_equals_zero_l2315_231520

theorem power_sum_equals_zero : (-1 : ℤ) ^ (5^2) + (1 : ℤ) ^ (2^5) = 0 := by
  sorry

end power_sum_equals_zero_l2315_231520


namespace constant_term_expansion_l2315_231594

/-- The constant term in the expansion of (1/x + 2x)^6 is 160 -/
theorem constant_term_expansion : ∃ c : ℕ, c = 160 ∧ 
  ∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, (λ x => (1/x + 2*x)^6) = (λ x => c + x * f x)) := by
  sorry

end constant_term_expansion_l2315_231594


namespace misread_number_correction_l2315_231576

theorem misread_number_correction (n : ℕ) (incorrect_avg correct_avg misread_value : ℚ) 
  (h1 : n = 10)
  (h2 : incorrect_avg = 14)
  (h3 : correct_avg = 15)
  (h4 : misread_value = 26) : 
  ∃ (actual_value : ℚ), 
    (n : ℚ) * correct_avg - (n : ℚ) * incorrect_avg = misread_value - actual_value ∧ 
    actual_value = 16 := by
  sorry

end misread_number_correction_l2315_231576


namespace cable_length_l2315_231583

/-- The length of a curve defined by the intersection of a sphere and a plane --/
theorem cable_length (x y z : ℝ) : 
  x + y + z = 10 →
  x * y + y * z + z * x = 18 →
  ∃ (curve_length : ℝ), curve_length = 4 * Real.pi * Real.sqrt (23 / 3) :=
by sorry

end cable_length_l2315_231583


namespace odd_function_extension_l2315_231522

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem odd_function_extension :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is an odd function
  (∀ x : ℝ, x ≥ 0 → f x = x^2 - 2*x) →  -- f(x) = x^2 - 2x when x ≥ 0
  (∀ x : ℝ, x < 0 → f x = -x^2 + 2*x) :=  -- f(x) = -x^2 + 2x when x < 0
by sorry

end odd_function_extension_l2315_231522


namespace geometric_sequence_product_l2315_231567

theorem geometric_sequence_product (a r : ℝ) (n : ℕ) (h_even : Even n) :
  let S := a * (1 - r^n) / (1 - r)
  let S' := (1 / (2*a)) * (r^n - 1) / (r - 1) * r^(1-n)
  let P := (2*a)^n * r^(n*(n-1)/2)
  P = (S * S')^(n/2) := by sorry

end geometric_sequence_product_l2315_231567


namespace nala_seashell_count_l2315_231539

/-- The number of seashells Nala found on Monday -/
def monday_shells : ℕ := 5

/-- The number of seashells Nala found on Tuesday -/
def tuesday_shells : ℕ := 7

/-- The number of seashells Nala discarded on Tuesday -/
def tuesday_discarded : ℕ := 3

/-- The number of seashells Nala found on Wednesday relative to Monday -/
def wednesday_multiplier : ℕ := 2

/-- The fraction of seashells Nala discarded on Wednesday -/
def wednesday_discard_fraction : ℚ := 1/2

/-- The number of seashells Nala found on Thursday relative to Tuesday -/
def thursday_multiplier : ℕ := 3

/-- The total number of unbroken seashells Nala has by the end of Thursday -/
def total_shells : ℕ := 35

theorem nala_seashell_count : 
  monday_shells + 
  (tuesday_shells - tuesday_discarded) + 
  (wednesday_multiplier * monday_shells - Nat.floor (↑(wednesday_multiplier * monday_shells) * wednesday_discard_fraction)) + 
  (thursday_multiplier * tuesday_shells) = total_shells := by
  sorry

end nala_seashell_count_l2315_231539


namespace ferris_wheel_capacity_l2315_231516

theorem ferris_wheel_capacity (total_people : ℕ) (num_seats : ℕ) (people_per_seat : ℕ) : 
  total_people = 18 → num_seats = 2 → people_per_seat = total_people / num_seats → people_per_seat = 9 :=
by
  sorry

end ferris_wheel_capacity_l2315_231516


namespace unique_solution_quadratic_l2315_231502

theorem unique_solution_quadratic (c : ℝ) (h : c ≠ 0) :
  (∃! b : ℝ, b > 0 ∧ (∃! x : ℝ, x^2 + 3 * (b + 1/b) * x + c = 0)) ↔ c = 9 := by
  sorry

end unique_solution_quadratic_l2315_231502


namespace polynomial_equality_constants_l2315_231523

theorem polynomial_equality_constants (k1 k2 k3 : ℤ) : 
  (∀ x : ℝ, -x^4 - (k1 + 11)*x^3 - k2*x^2 - 8*x - k3 = -(x - 2)*(x^3 - 6*x^2 + 8*x - 4)) ↔ 
  (k1 = -19 ∧ k2 = 20 ∧ k3 = 8) := by
sorry

end polynomial_equality_constants_l2315_231523


namespace negation_of_proposition_l2315_231572

theorem negation_of_proposition :
  (¬ ∀ (a : ℝ) (n : ℕ), n > 0 → (a ≠ n → a * n ≠ 2 * n)) ↔
  (∃ (a : ℝ) (n : ℕ), n > 0 ∧ a ≠ n ∧ a * n = 2 * n) :=
by sorry

end negation_of_proposition_l2315_231572


namespace rhombus_longest_diagonal_l2315_231591

/-- Given a rhombus with area 150 square units and diagonals in the ratio 4:3,
    prove that the length of the longest diagonal is 20 units. -/
theorem rhombus_longest_diagonal (area : ℝ) (d₁ d₂ : ℝ) : 
  area = 150 →
  d₁ / d₂ = 4 / 3 →
  area = (1 / 2) * d₁ * d₂ →
  d₁ > d₂ →
  d₁ = 20 := by
sorry

end rhombus_longest_diagonal_l2315_231591


namespace joes_fast_food_cost_l2315_231545

/-- Calculates the cost of a purchase at Joe's Fast Food -/
def calculate_cost (sandwich_price : ℕ) (soda_price : ℕ) (sandwich_count : ℕ) (soda_count : ℕ) (bulk_discount : ℕ) (bulk_threshold : ℕ) : ℕ :=
  let total_items := sandwich_count + soda_count
  let subtotal := sandwich_price * sandwich_count + soda_price * soda_count
  if total_items > bulk_threshold then subtotal - bulk_discount else subtotal

/-- The cost of purchasing 6 sandwiches and 6 sodas at Joe's Fast Food is 37 dollars -/
theorem joes_fast_food_cost : calculate_cost 4 3 6 6 5 10 = 37 := by
  sorry

end joes_fast_food_cost_l2315_231545


namespace line_through_points_circle_through_points_circle_center_on_y_axis_l2315_231513

-- Define the points A and B
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 3)

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + (y-2)^2 = 2

-- Theorem for the line equation
theorem line_through_points : 
  line_eq A.1 A.2 ∧ line_eq B.1 B.2 := by sorry

-- Theorem for the circle equation
theorem circle_through_points : 
  circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 := by sorry

-- Theorem to prove the center of the circle is on the y-axis
theorem circle_center_on_y_axis : 
  ∃ y : ℝ, ∀ x : ℝ, circle_eq 0 y → circle_eq x y → x = 0 := by sorry

end line_through_points_circle_through_points_circle_center_on_y_axis_l2315_231513


namespace book_arrangement_problem_l2315_231562

/-- Represents the number of arrangements of books on a shelf. -/
def num_arrangements (n : ℕ) (chinese : ℕ) (math : ℕ) (physics : ℕ) : ℕ := sorry

/-- Theorem stating the number of arrangements for the given problem. -/
theorem book_arrangement_problem :
  num_arrangements 5 2 2 1 = 48 :=
sorry

end book_arrangement_problem_l2315_231562


namespace lattice_point_bounds_l2315_231535

/-- The minimum number of points in ℤ^d such that any set of these points
    will contain n points whose centroid is a lattice point -/
def f (n d : ℕ) : ℕ :=
  sorry

theorem lattice_point_bounds (n d : ℕ) (hn : n > 0) (hd : d > 0) :
  (n - 1) * 2^d + 1 ≤ f n d ∧ f n d ≤ (n - 1) * n^d + 1 :=
by sorry

end lattice_point_bounds_l2315_231535


namespace intersection_point_first_quadrant_l2315_231590

-- Define the quadratic and linear functions
def f (x : ℝ) : ℝ := x^2 - x - 5
def g (x : ℝ) : ℝ := 2*x - 1

-- Define the first quadrant
def first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

-- Theorem statement
theorem intersection_point_first_quadrant :
  ∃! p : ℝ × ℝ, first_quadrant p ∧ f p.1 = g p.1 ∧ f p.1 = p.2 ∧ p = (4, 7) :=
sorry

end intersection_point_first_quadrant_l2315_231590


namespace problems_per_worksheet_l2315_231569

/-- Given a set of worksheets with some graded and some problems left to grade,
    calculate the number of problems per worksheet. -/
theorem problems_per_worksheet
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (problems_left : ℕ)
  (h1 : total_worksheets = 14)
  (h2 : graded_worksheets = 7)
  (h3 : problems_left = 14)
  (h4 : graded_worksheets < total_worksheets) :
  problems_left / (total_worksheets - graded_worksheets) = 2 :=
by
  sorry

#check problems_per_worksheet

end problems_per_worksheet_l2315_231569


namespace minimum_value_of_a_l2315_231599

theorem minimum_value_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 := by
  sorry

end minimum_value_of_a_l2315_231599


namespace midpoint_coordinate_product_l2315_231505

/-- The product of the coordinates of the midpoint of a line segment with endpoints (5, -3) and (-7, 11) is -4. -/
theorem midpoint_coordinate_product : 
  let x1 : ℝ := 5
  let y1 : ℝ := -3
  let x2 : ℝ := -7
  let y2 : ℝ := 11
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x * midpoint_y = -4 := by
  sorry

end midpoint_coordinate_product_l2315_231505


namespace saras_quarters_l2315_231585

/-- The number of quarters Sara has after receiving some from her dad -/
def total_quarters (initial_quarters given_quarters : ℝ) : ℝ :=
  initial_quarters + given_quarters

/-- Theorem stating that Sara's total quarters is the sum of her initial quarters and those given by her dad -/
theorem saras_quarters (initial_quarters given_quarters : ℝ) :
  total_quarters initial_quarters given_quarters = initial_quarters + given_quarters :=
by sorry

end saras_quarters_l2315_231585


namespace sqrt_1001_irreducible_l2315_231551

theorem sqrt_1001_irreducible : ∀ a b : ℕ, a * a = 1001 * (b * b) → a = 1001 ∧ b = 1 := by
  sorry

end sqrt_1001_irreducible_l2315_231551


namespace eating_contest_l2315_231577

/-- Eating contest problem -/
theorem eating_contest (hot_dog_weight burger_weight pie_weight : ℕ)
  (noah_burgers jacob_pies mason_hotdogs : ℕ) :
  hot_dog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  jacob_pies = noah_burgers - 3 →
  mason_hotdogs = 3 * jacob_pies →
  noah_burgers = 8 →
  mason_hotdogs * hot_dog_weight = 30 := by
sorry

end eating_contest_l2315_231577


namespace weight_problem_l2315_231532

theorem weight_problem (c d e f : ℝ) 
  (h1 : c + d = 330)
  (h2 : d + e = 290)
  (h3 : e + f = 310) :
  c + f = 350 := by
sorry

end weight_problem_l2315_231532


namespace sets_theorem_l2315_231555

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 = 0}
def B : Set ℝ := {y | ∃ x, y = x^2 - 4}

-- Statement to prove
theorem sets_theorem :
  (A ∩ B = A) ∧ (A ∪ B = B) := by sorry

end sets_theorem_l2315_231555


namespace f1_times_g0_l2315_231538

-- Define f as an odd function on ℝ
def f : ℝ → ℝ := sorry

-- Define g as an even function on ℝ
def g : ℝ → ℝ := sorry

-- Define the relationship between f and g
axiom fg_relation : ∀ x : ℝ, f x - g x = 2^x

-- Define the property of odd function
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Define the property of even function
axiom g_even : ∀ x : ℝ, g (-x) = g x

-- Theorem to prove
theorem f1_times_g0 : f 1 * g 0 = -3/4 := by sorry

end f1_times_g0_l2315_231538


namespace shirt_cost_l2315_231508

-- Define the number of $10 bills
def num_10_bills : ℕ := 2

-- Define the number of $20 bills
def num_20_bills : ℕ := num_10_bills + 1

-- Define the value of a $10 bill
def value_10_bill : ℕ := 10

-- Define the value of a $20 bill
def value_20_bill : ℕ := 20

-- Theorem: The cost of the shirt is $80
theorem shirt_cost : 
  num_10_bills * value_10_bill + num_20_bills * value_20_bill = 80 := by
  sorry

end shirt_cost_l2315_231508


namespace right_triangle_properties_l2315_231587

-- Define a right-angled triangle ABC
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  h_right_angle : angleA + angleB = π / 2
  h_sides : c^2 = a^2 + b^2
  h_not_equal : a ≠ b

-- State the theorem
theorem right_triangle_properties (t : RightTriangle) :
  (Real.tan t.angleA * Real.tan t.angleB ≠ 1) ∧
  (Real.sin t.angleA = t.a / t.c) ∧
  (t.c^2 - t.a^2 = t.b^2) ∧
  (t.c = t.b / Real.cos t.angleA) :=
by sorry

end right_triangle_properties_l2315_231587


namespace man_upstream_speed_l2315_231560

/-- Calculates the upstream speed of a man given his downstream speed and the stream speed -/
def upstream_speed (downstream_speed stream_speed : ℝ) : ℝ :=
  downstream_speed - 2 * stream_speed

/-- Theorem stating that given a downstream speed of 13 kmph and a stream speed of 2.5 kmph, 
    the upstream speed is 8 kmph -/
theorem man_upstream_speed :
  upstream_speed 13 2.5 = 8 := by
  sorry

end man_upstream_speed_l2315_231560


namespace evaluate_expression_l2315_231566

theorem evaluate_expression (x : ℕ) (h : x = 3) : x^2 + x * (x^(Nat.factorial x)) = 2196 :=
by sorry

end evaluate_expression_l2315_231566


namespace fraction_operation_equivalence_l2315_231527

theorem fraction_operation_equivalence (x : ℚ) :
  x * (5/6) / (2/7) = x * (35/12) := by
sorry

end fraction_operation_equivalence_l2315_231527


namespace multiplication_proof_l2315_231564

theorem multiplication_proof : 287 * 23 = 6601 := by
  sorry

end multiplication_proof_l2315_231564


namespace exponent_multiplication_l2315_231531

theorem exponent_multiplication (x : ℝ) : x^3 * (2*x^4) = 2*x^7 := by
  sorry

end exponent_multiplication_l2315_231531


namespace carton_length_is_30_inches_l2315_231588

/-- Proves that the length of a carton is 30 inches given specific dimensions and constraints -/
theorem carton_length_is_30_inches 
  (carton_width : ℕ) 
  (carton_height : ℕ)
  (soap_length : ℕ) 
  (soap_width : ℕ) 
  (soap_height : ℕ)
  (max_soap_boxes : ℕ)
  (h1 : carton_width = 42)
  (h2 : carton_height = 60)
  (h3 : soap_length = 7)
  (h4 : soap_width = 6)
  (h5 : soap_height = 5)
  (h6 : max_soap_boxes = 360) :
  ∃ (carton_length : ℕ), carton_length = 30 ∧ 
    carton_length * carton_width * carton_height = 
    max_soap_boxes * soap_length * soap_width * soap_height :=
by
  sorry

end carton_length_is_30_inches_l2315_231588


namespace integer_count_inequality_l2315_231593

theorem integer_count_inequality (x : ℤ) : 
  (Finset.filter (fun i => (i - 2)^2 ≤ 4) (Finset.range 10)).card = 5 := by
  sorry

end integer_count_inequality_l2315_231593


namespace systematic_sample_smallest_element_l2315_231580

/-- Represents a systematic sample -/
structure SystematicSample where
  total : ℕ
  sampleSize : ℕ
  interval : ℕ
  containsElement : ℕ

/-- The smallest element in a systematic sample -/
def smallestElement (s : SystematicSample) : ℕ :=
  s.interval * (s.containsElement / s.interval)

theorem systematic_sample_smallest_element 
  (s : SystematicSample) 
  (h1 : s.total = 360)
  (h2 : s.sampleSize = 30)
  (h3 : s.interval = s.total / s.sampleSize)
  (h4 : s.containsElement = 105)
  (h5 : s.containsElement ≤ s.total)
  : smallestElement s = 96 := by
  sorry

end systematic_sample_smallest_element_l2315_231580


namespace chocolate_squares_multiple_l2315_231574

theorem chocolate_squares_multiple (mike_squares jenny_squares : ℕ) 
  (h1 : mike_squares = 20) 
  (h2 : jenny_squares = 65) 
  (h3 : ∃ m : ℕ, jenny_squares = mike_squares * m + 5) : 
  ∃ m : ℕ, m = 3 ∧ jenny_squares = mike_squares * m + 5 := by
sorry

end chocolate_squares_multiple_l2315_231574


namespace exists_divisible_by_13_in_79_consecutive_l2315_231568

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: In any sequence of 79 consecutive positive integers, 
    there exists at least one integer whose sum of digits is divisible by 13 -/
theorem exists_divisible_by_13_in_79_consecutive (start : ℕ) : 
  ∃ k ∈ Finset.range 79, (sum_of_digits (start + k)) % 13 = 0 := by sorry

end exists_divisible_by_13_in_79_consecutive_l2315_231568


namespace root_sum_powers_l2315_231542

theorem root_sum_powers (α β : ℝ) : 
  α^2 - 5*α + 6 = 0 → β^2 - 5*β + 6 = 0 → 3*α^3 + 10*β^4 = 2305 := by
sorry

end root_sum_powers_l2315_231542


namespace reflected_light_equation_l2315_231540

/-- The incident light line -/
def incident_line (x y : ℝ) : Prop := 2 * x - y + 6 = 0

/-- The reflection line -/
def reflection_line (x y : ℝ) : Prop := y = x

/-- The reflected light line -/
def reflected_line (x y : ℝ) : Prop := x + 2 * y + 18 = 0

/-- 
Given an incident light line 2x - y + 6 = 0 striking the line y = x, 
prove that the reflected light line has the equation x + 2y + 18 = 0.
-/
theorem reflected_light_equation :
  ∀ x y : ℝ, incident_line x y ∧ reflection_line x y → reflected_line x y :=
by sorry

end reflected_light_equation_l2315_231540


namespace sum_of_roots_for_f_l2315_231503

def f (x : ℝ) : ℝ := (4*x)^2 - (4*x) + 2

theorem sum_of_roots_for_f (z : ℝ) : 
  (∃ z₁ z₂, f z₁ = 10 ∧ f z₂ = 10 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = 1/16) :=
sorry

end sum_of_roots_for_f_l2315_231503


namespace quadratic_touches_x_axis_at_one_point_l2315_231561

/-- A quadratic function g(x) = x^2 - 6x + k -/
def g (k : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + k

/-- The discriminant of the quadratic function g -/
def discriminant (k : ℝ) : ℝ := (-6)^2 - 4*1*k

/-- Theorem: The value of k that makes g(x) touch the x-axis at exactly one point is 9 -/
theorem quadratic_touches_x_axis_at_one_point :
  ∃ (k : ℝ), (discriminant k = 0) ∧ (k = 9) := by sorry

end quadratic_touches_x_axis_at_one_point_l2315_231561


namespace triangle_side_length_triangle_angle_relation_l2315_231521

-- Define the triangle ABC
structure Triangle where
  a : ℝ  -- side opposite to angle A
  b : ℝ  -- side opposite to angle B
  c : ℝ  -- side opposite to angle C
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C

-- Theorem 1
theorem triangle_side_length (t : Triangle) 
  (h1 : t.a = 3 * t.c) 
  (h2 : t.b = Real.sqrt 2) 
  (h3 : Real.cos t.B = 2/3) : 
  t.c = Real.sqrt 3 / 3 := by
  sorry

-- Theorem 2
theorem triangle_angle_relation (t : Triangle) 
  (h : Real.sin t.A / t.a = Real.cos t.B / (2 * t.b)) : 
  Real.sin (t.B + π/2) = 2 * Real.sqrt 5 / 5 := by
  sorry

end triangle_side_length_triangle_angle_relation_l2315_231521


namespace box_third_side_l2315_231547

/-- A rectangular box with known properties -/
structure Box where
  cubes : ℕ  -- Number of cubes that fit in the box
  cube_volume : ℕ  -- Volume of each cube in cubic centimetres
  side1 : ℕ  -- Length of first known side in centimetres
  side2 : ℕ  -- Length of second known side in centimetres

/-- The length of the third side of the box -/
def third_side (b : Box) : ℚ :=
  (b.cubes * b.cube_volume : ℚ) / (b.side1 * b.side2)

/-- Theorem stating that the third side of the given box is 6 centimetres -/
theorem box_third_side :
  let b : Box := { cubes := 24, cube_volume := 27, side1 := 9, side2 := 12 }
  third_side b = 6 := by sorry

end box_third_side_l2315_231547


namespace sqrt_256_squared_plus_100_l2315_231537

theorem sqrt_256_squared_plus_100 : (Real.sqrt 256)^2 + 100 = 356 := by
  sorry

end sqrt_256_squared_plus_100_l2315_231537


namespace walking_distance_l2315_231556

theorem walking_distance (speed1 speed2 time_diff : ℝ) (h1 : speed1 = 4)
  (h2 : speed2 = 3) (h3 : time_diff = 1/2) :
  let distance := speed1 * (time_diff + distance / speed2)
  distance = 6 := by sorry

end walking_distance_l2315_231556


namespace malvina_card_sum_l2315_231515

open Real MeasureTheory

theorem malvina_card_sum : ∀ x : ℝ,
  90 * π / 180 < x ∧ x < π →
  (∀ y : ℝ, 90 * π / 180 < y ∧ y < π →
    sin y > 0 ∧ cos y < 0 ∧ tan y < 0) →
  (∫ y in Set.Icc (90 * π / 180) π, sin y) = 1 := by
  sorry

end malvina_card_sum_l2315_231515


namespace hyperbola_from_ellipse_and_asymptote_l2315_231558

/-- Given ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

/-- Given asymptote equation -/
def asymptote_equation (x y : ℝ) : Prop := x - Real.sqrt 2 * y = 0

/-- Hyperbola equation to be proved -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / (8 * Real.sqrt 3 / 3) - y^2 / (4 * Real.sqrt 3 / 3) = 1

theorem hyperbola_from_ellipse_and_asymptote :
  ∀ x y : ℝ,
  (∃ a b : ℝ, ellipse_equation a b ∧
    (∀ c d : ℝ, hyperbola_equation c d → (c - a)^2 + (d - b)^2 = (c + a)^2 + (d + b)^2)) →
  asymptote_equation x y →
  hyperbola_equation x y :=
sorry

end hyperbola_from_ellipse_and_asymptote_l2315_231558


namespace positive_expression_l2315_231598

theorem positive_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  max ((a + b + c)^2 - 8*a*c) (max ((a + b + c)^2 - 8*b*c) ((a + b + c)^2 - 8*a*b)) > 0 := by
  sorry

end positive_expression_l2315_231598


namespace sin_cos_three_eighths_pi_l2315_231592

theorem sin_cos_three_eighths_pi (π : Real) :
  Real.sin (3 * π / 8) * Real.cos (π / 8) = (2 + Real.sqrt 2) / 4 := by
  sorry

end sin_cos_three_eighths_pi_l2315_231592


namespace intersection_of_M_and_N_l2315_231509

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by sorry

end intersection_of_M_and_N_l2315_231509
