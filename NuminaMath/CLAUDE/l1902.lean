import Mathlib

namespace NUMINAMATH_CALUDE_meadow_business_revenue_l1902_190207

/-- Represents Meadow's diaper business --/
structure DiaperBusiness where
  boxes_per_week : ℕ
  packs_per_box : ℕ
  diapers_per_pack : ℕ
  price_per_diaper : ℕ

/-- Calculates the total money made from selling all diapers --/
def total_money_made (business : DiaperBusiness) : ℕ :=
  business.boxes_per_week * business.packs_per_box * business.diapers_per_pack * business.price_per_diaper

/-- Theorem stating that Meadow's business makes $960000 from selling all diapers --/
theorem meadow_business_revenue :
  let meadow_business : DiaperBusiness := {
    boxes_per_week := 30,
    packs_per_box := 40,
    diapers_per_pack := 160,
    price_per_diaper := 5
  }
  total_money_made meadow_business = 960000 := by
  sorry

end NUMINAMATH_CALUDE_meadow_business_revenue_l1902_190207


namespace NUMINAMATH_CALUDE_last_locker_opened_l1902_190298

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Represents the process of toggling lockers -/
def toggle_lockers (n : ℕ) (k : ℕ) : List LockerState → List LockerState :=
  sorry

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- Finds the largest perfect square less than or equal to a given number -/
def largest_perfect_square_le (n : ℕ) : ℕ :=
  sorry

theorem last_locker_opened (num_lockers : ℕ) (num_lockers_eq : num_lockers = 500) :
  largest_perfect_square_le num_lockers = 484 :=
sorry

end NUMINAMATH_CALUDE_last_locker_opened_l1902_190298


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l1902_190206

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex nonagon has 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l1902_190206


namespace NUMINAMATH_CALUDE_perfume_price_increase_l1902_190226

theorem perfume_price_increase (x : ℝ) : 
  let original_price : ℝ := 1200
  let increased_price : ℝ := original_price * (1 + x / 100)
  let final_price : ℝ := increased_price * (1 - 15 / 100)
  final_price = original_price - 78 → x = 10 :=
by sorry

end NUMINAMATH_CALUDE_perfume_price_increase_l1902_190226


namespace NUMINAMATH_CALUDE_project_completion_time_l1902_190294

theorem project_completion_time (team_a_time team_b_time team_c_time total_time : ℝ) 
  (h1 : team_a_time = 10)
  (h2 : team_b_time = 15)
  (h3 : team_c_time = 20)
  (h4 : total_time = 6) :
  (1 - (1 / team_b_time + 1 / team_c_time) * total_time) / (1 / team_a_time) = 3 := by
  sorry

#check project_completion_time

end NUMINAMATH_CALUDE_project_completion_time_l1902_190294


namespace NUMINAMATH_CALUDE_parabola_ratio_l1902_190289

/-- Given a parabola y = ax² + bx + c passing through points (-1, 1) and (3, 1),
    prove that a/b = -2 -/
theorem parabola_ratio (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 1 → x = -1 ∨ x = 3) →
  a / b = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_ratio_l1902_190289


namespace NUMINAMATH_CALUDE_polygon_sides_l1902_190290

theorem polygon_sides (n : ℕ) (sum_angles : ℝ) : 
  sum_angles = 2002 → 
  (n - 2) * 180 - 360 < sum_angles ∧ sum_angles < (n - 2) * 180 →
  n = 14 ∨ n = 15 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l1902_190290


namespace NUMINAMATH_CALUDE_triangle_properties_l1902_190244

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.A + t.B + t.C = Real.pi)
  (h2 : t.a * Real.cos t.B = (2 * t.c - t.b) * Real.cos t.A) : 
  t.A = Real.pi / 3 ∧ 
  (∃ (max : Real), max = Real.sqrt 3 ∧ 
    ∀ (x : Real), x = Real.sin t.B + Real.sin t.C → x ≤ max) ∧
  (t.A = t.B ∧ t.B = t.C) := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l1902_190244


namespace NUMINAMATH_CALUDE_large_pizza_has_16_slices_l1902_190297

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := sorry

/-- The number of large pizzas -/
def num_large_pizzas : ℕ := 2

/-- The number of small pizzas -/
def num_small_pizzas : ℕ := 2

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := 8

/-- The total number of slices eaten -/
def total_slices_eaten : ℕ := 48

theorem large_pizza_has_16_slices :
  num_large_pizzas * large_pizza_slices + num_small_pizzas * small_pizza_slices = total_slices_eaten →
  large_pizza_slices = 16 := by
  sorry

end NUMINAMATH_CALUDE_large_pizza_has_16_slices_l1902_190297


namespace NUMINAMATH_CALUDE_problem_statement_l1902_190251

theorem problem_statement (a b c d k m : ℕ) 
  (h1 : d * a = b * c)
  (h2 : a + d = 2^k)
  (h3 : b + c = 2^m) :
  a = 1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1902_190251


namespace NUMINAMATH_CALUDE_matrix_equality_implies_fraction_l1902_190287

/-- Given two 2x2 matrices A and B, where A is [[2, 5], [3, 7]] and B is [[a, b], [c, d]],
    if AB = BA and 5b ≠ c, then (a - d) / (c - 5b) = 6c / (5a + 22c) -/
theorem matrix_equality_implies_fraction (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 5; 3, 7]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A * B = B * A) → (5 * b ≠ c) → 
  (a - d) / (c - 5 * b) = 6 * c / (5 * a + 22 * c) := by
  sorry

end NUMINAMATH_CALUDE_matrix_equality_implies_fraction_l1902_190287


namespace NUMINAMATH_CALUDE_marked_squares_rearrangement_l1902_190235

/-- Represents a square table with marked cells -/
structure MarkedTable (n : ℕ) where
  marks : Finset ((Fin n) × (Fin n))
  mark_count : marks.card = 110

/-- Represents a permutation of rows and columns -/
structure TablePermutation (n : ℕ) where
  row_perm : Equiv.Perm (Fin n)
  col_perm : Equiv.Perm (Fin n)

/-- Checks if a cell is on or above the main diagonal -/
def is_on_or_above_diagonal {n : ℕ} (i j : Fin n) : Prop :=
  i.val ≤ j.val

/-- Applies a permutation to a marked cell -/
def apply_perm {n : ℕ} (perm : TablePermutation n) (cell : (Fin n) × (Fin n)) : (Fin n) × (Fin n) :=
  (perm.row_perm cell.1, perm.col_perm cell.2)

/-- Theorem: For any 100x100 table with 110 marked squares, there exists a permutation
    that places all marked squares on or above the main diagonal -/
theorem marked_squares_rearrangement :
  ∀ (t : MarkedTable 100),
  ∃ (perm : TablePermutation 100),
  ∀ cell ∈ t.marks,
  is_on_or_above_diagonal (apply_perm perm cell).1 (apply_perm perm cell).2 :=
sorry

end NUMINAMATH_CALUDE_marked_squares_rearrangement_l1902_190235


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_10_l1902_190285

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  S : ℕ+ → ℚ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1
  sum_def : ∀ n : ℕ+, S n = (n : ℚ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_10 (seq : ArithmeticSequence) 
  (h1 : seq.a 3 = 16) (h2 : seq.S 20 = 20) : seq.S 10 = 110 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_10_l1902_190285


namespace NUMINAMATH_CALUDE_polynomial_coefficient_F_l1902_190270

def polynomial (x E F G H : ℤ) : ℤ := x^6 - 14*x^5 + E*x^4 + F*x^3 + G*x^2 + H*x + 36

def roots : List ℤ := [3, 3, 2, 2, 2, 2]

theorem polynomial_coefficient_F (E F G H : ℤ) :
  (∀ r ∈ roots, polynomial r E F G H = 0) →
  (List.sum roots = 14) →
  (∀ r ∈ roots, r > 0) →
  F = -248 := by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_F_l1902_190270


namespace NUMINAMATH_CALUDE_f_f_zero_l1902_190246

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 3 * x^2 - 4
  else if x = 0 then Real.pi
  else 0

theorem f_f_zero : f (f 0) = 3 * Real.pi^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_f_f_zero_l1902_190246


namespace NUMINAMATH_CALUDE_ab_difference_l1902_190241

theorem ab_difference (A B C : ℕ) : 
  A > 0 → B > 0 → C > 0 →
  (1212017 * 100 * A + 1212017 * 10 * B + 1212017 * C) % 45 = 0 →
  ∃ (max_AB min_AB : ℕ),
    (∀ A' B' C' : ℕ, 
      A' > 0 → B' > 0 → C' > 0 →
      (1212017 * 100 * A' + 1212017 * 10 * B' + 1212017 * C') % 45 = 0 →
      A' * 10 + B' ≤ max_AB) ∧
    (∀ A' B' C' : ℕ, 
      A' > 0 → B' > 0 → C' > 0 →
      (1212017 * 100 * A' + 1212017 * 10 * B' + 1212017 * C') % 45 = 0 →
      A' * 10 + B' ≥ min_AB) ∧
    max_AB - min_AB = 85 :=
by sorry

end NUMINAMATH_CALUDE_ab_difference_l1902_190241


namespace NUMINAMATH_CALUDE_scientific_notation_of_113700_l1902_190273

theorem scientific_notation_of_113700 :
  (113700 : ℝ) = 1.137 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_113700_l1902_190273


namespace NUMINAMATH_CALUDE_abs_eq_sum_implies_zero_l1902_190238

theorem abs_eq_sum_implies_zero (x y : ℝ) :
  |x - y^2| = x + y^2 → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_sum_implies_zero_l1902_190238


namespace NUMINAMATH_CALUDE_factorization_equality_l1902_190243

theorem factorization_equality (a b x y : ℝ) :
  a^2 * b * (x - y)^3 - a * b^2 * (y - x)^2 = a * b * (x - y)^2 * (a * x - a * y - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1902_190243


namespace NUMINAMATH_CALUDE_larrys_coincidence_l1902_190281

theorem larrys_coincidence (a b c d e : ℝ) 
  (ha : a = 5) (hb : b = 3) (hc : c = 6) (hd : d = 4) :
  a - b + c + d - e = a - (b - (c + (d - e))) :=
by sorry

end NUMINAMATH_CALUDE_larrys_coincidence_l1902_190281


namespace NUMINAMATH_CALUDE_log_ratio_equality_l1902_190220

theorem log_ratio_equality : (Real.log 2 / Real.log 3) / (Real.log 8 / Real.log 9) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_equality_l1902_190220


namespace NUMINAMATH_CALUDE_square_difference_emily_calculation_l1902_190277

theorem square_difference (n : ℕ) : (n - 1)^2 = n^2 - (2*n - 1) := by sorry

theorem emily_calculation : 39^2 = 40^2 - 79 := by sorry

end NUMINAMATH_CALUDE_square_difference_emily_calculation_l1902_190277


namespace NUMINAMATH_CALUDE_simplify_expression_l1902_190222

theorem simplify_expression (a b : ℝ) :
  (30 * a + 45 * b) + (15 * a + 40 * b) - (20 * a + 55 * b) + (5 * a - 10 * b) = 30 * a + 20 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1902_190222


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_S_l1902_190260

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if all digits in a natural number are different -/
def all_digits_different (n : ℕ) : Prop := sorry

/-- The theorem stating the smallest possible sum of digits of S -/
theorem smallest_sum_of_digits_S : 
  ∀ (a b : ℕ), 
    100 ≤ a ∧ a < 1000 ∧ 
    100 ≤ b ∧ b < 1000 ∧
    all_digits_different a ∧ 
    all_digits_different b ∧
    all_digits_different (a + b) ∧
    (a + b < 1000) →
    (∃ (S : ℕ), S = a + b ∧ sum_of_digits S = 4 ∧ 
      ∀ (T : ℕ), T = a + b → sum_of_digits T ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_S_l1902_190260


namespace NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l1902_190213

theorem quadratic_roots_real_and_equal : ∃ x : ℝ, 
  (∀ y : ℝ, y^2 + 4*y*Real.sqrt 2 + 8 = 0 ↔ y = x) ∧ 
  (x^2 + 4*x*Real.sqrt 2 + 8 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l1902_190213


namespace NUMINAMATH_CALUDE_probability_of_arithmetic_progression_l1902_190232

/-- Represents an 8-sided die -/
def Die := Fin 8

/-- The number of dice rolled -/
def numDice : ℕ := 4

/-- Checks if a list of four numbers forms an arithmetic progression with common difference 2 -/
def isArithmeticProgression (nums : List ℕ) : Prop :=
  nums.length = numDice ∧
  ∃ a : ℕ, nums = [a, a + 2, a + 4, a + 6]

/-- The set of all possible outcomes when rolling four 8-sided dice -/
def allOutcomes : Finset (List Die) :=
  sorry

/-- The set of favorable outcomes (those forming the desired arithmetic progression) -/
def favorableOutcomes : Finset (List Die) :=
  sorry

/-- The probability of obtaining a favorable outcome -/
theorem probability_of_arithmetic_progression :
  (Finset.card favorableOutcomes : ℚ) / (Finset.card allOutcomes : ℚ) = 3 / 256 :=
sorry

end NUMINAMATH_CALUDE_probability_of_arithmetic_progression_l1902_190232


namespace NUMINAMATH_CALUDE_total_bacon_needed_l1902_190228

/-- The number of eggs on each breakfast plate -/
def eggs_per_plate : ℕ := 2

/-- The number of customers ordering breakfast plates -/
def num_customers : ℕ := 14

/-- The number of bacon strips on each breakfast plate -/
def bacon_per_plate : ℕ := 2 * eggs_per_plate

/-- The total number of bacon strips needed -/
def total_bacon : ℕ := num_customers * bacon_per_plate

theorem total_bacon_needed : total_bacon = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_bacon_needed_l1902_190228


namespace NUMINAMATH_CALUDE_vector_at_zero_given_two_points_l1902_190230

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  -- The vector on the line at parameter t
  vector : ℝ → ℝ × ℝ

theorem vector_at_zero_given_two_points (L : ParameterizedLine) :
  L.vector 1 = (2, 3) →
  L.vector 4 = (8, -5) →
  L.vector 0 = (0, 17/3) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_zero_given_two_points_l1902_190230


namespace NUMINAMATH_CALUDE_number_equation_solution_l1902_190204

theorem number_equation_solution : ∃ x : ℝ, (0.68 * x - 5) / 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1902_190204


namespace NUMINAMATH_CALUDE_power_of_256_three_fourths_l1902_190227

theorem power_of_256_three_fourths : (256 : ℝ) ^ (3/4) = 64 := by sorry

end NUMINAMATH_CALUDE_power_of_256_three_fourths_l1902_190227


namespace NUMINAMATH_CALUDE_tree_leaves_problem_l1902_190263

/-- The number of leaves remaining after dropping 1/10 of leaves n times -/
def leavesRemaining (initialLeaves : ℕ) (n : ℕ) : ℚ :=
  initialLeaves * (9/10)^n

/-- The proposition that a tree with the given leaf-dropping pattern initially had 311 leaves -/
theorem tree_leaves_problem : ∃ (initialLeaves : ℕ),
  (leavesRemaining initialLeaves 4).num = 204 * (leavesRemaining initialLeaves 4).den ∧
  initialLeaves = 311 := by
  sorry


end NUMINAMATH_CALUDE_tree_leaves_problem_l1902_190263


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1902_190240

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (x - 3) * (x + 2) = k + 3 * x) ↔ k = -10 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1902_190240


namespace NUMINAMATH_CALUDE_cosine_function_properties_l1902_190200

/-- Given function f(x) = cos(ωx + φ) -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

theorem cosine_function_properties (ω φ : ℝ) 
  (h_ω_pos : ω > 0)
  (h_φ_range : 0 < φ ∧ φ < π / 2)
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x)
  (h_value : f ω φ (π / 3) = -Real.sqrt 3 / 2) :
  (ω = 2 ∧ φ = π / 6) ∧
  (∀ x, f ω φ x > 1 / 2 ↔ ∃ k : ℤ, k * π - π / 4 < x ∧ x < k * π + π / 12) :=
by sorry

end NUMINAMATH_CALUDE_cosine_function_properties_l1902_190200


namespace NUMINAMATH_CALUDE_range_of_f_l1902_190295

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- State the theorem
theorem range_of_f :
  ∀ y ∈ Set.Icc 1 10, ∃ x ∈ Set.Icc 1 5, f x = y ∧
  ∀ x ∈ Set.Icc 1 5, f x ∈ Set.Icc 1 10 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l1902_190295


namespace NUMINAMATH_CALUDE_B_and_C_complementary_l1902_190202

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

end NUMINAMATH_CALUDE_B_and_C_complementary_l1902_190202


namespace NUMINAMATH_CALUDE_car_stopping_distance_l1902_190269

/-- Represents the distance traveled by a car in feet during each second after brakes are applied -/
def braking_sequence : ℕ → ℤ
  | 0 => 28
  | n + 1 => braking_sequence n - 7

/-- Calculates the total distance traveled by the car until it stops -/
def total_distance : ℕ → ℤ
  | 0 => 28
  | n + 1 => total_distance n + braking_sequence (n + 1)

/-- The number of seconds it takes for the car to stop -/
def stopping_time : ℕ := 3

theorem car_stopping_distance :
  total_distance stopping_time = 70 :=
sorry

end NUMINAMATH_CALUDE_car_stopping_distance_l1902_190269


namespace NUMINAMATH_CALUDE_inequality_proof_l1902_190248

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : 
  (Real.sqrt (b^2 - a*c)) / a < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1902_190248


namespace NUMINAMATH_CALUDE_remaining_distance_to_grandma_l1902_190215

theorem remaining_distance_to_grandma (total_distance driven_first driven_second : ℕ) 
  (h1 : total_distance = 78)
  (h2 : driven_first = 35)
  (h3 : driven_second = 18) : 
  total_distance - (driven_first + driven_second) = 25 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_to_grandma_l1902_190215


namespace NUMINAMATH_CALUDE_original_average_marks_l1902_190258

/-- Given a class of students, proves that if doubling each student's mark
    results in a new average of 140, then the original average mark was 70. -/
theorem original_average_marks (n : ℕ) (original_avg : ℝ) :
  n > 0 →
  2 * original_avg = 140 →
  original_avg = 70 :=
by sorry

end NUMINAMATH_CALUDE_original_average_marks_l1902_190258


namespace NUMINAMATH_CALUDE_power_product_cube_l1902_190262

theorem power_product_cube (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_product_cube_l1902_190262


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1902_190264

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_properties (a b c : ℝ) 
  (h1 : QuadraticFunction a b c 1 = -a/2)
  (h2 : a > 0)
  (h3 : ∀ x, QuadraticFunction a b c x < 1 ↔ 0 < x ∧ x < 3) :
  (QuadraticFunction a b c = fun x ↦ (2/3) * x^2 - 2 * x + 1) ∧
  (∃ x, 0 < x ∧ x < 2 ∧ QuadraticFunction a b c x = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1902_190264


namespace NUMINAMATH_CALUDE_problem_solution_l1902_190209

def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem problem_solution :
  (∀ a : ℝ, A ∪ B a = B a → a = 1) ∧
  (∀ C : Set ℝ, (∀ a : ℝ, A ∩ B a = B a → a ∈ C) → C = {a : ℝ | a ≤ -1 ∨ a = 1}) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1902_190209


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_not_regular_l1902_190221

/-- A pyramid with a regular polygon base and all edges of equal length -/
structure RegularPyramid (n : ℕ) where
  /-- The number of sides of the base polygon -/
  base_sides : n > 2
  /-- The length of each edge of the pyramid -/
  edge_length : ℝ
  /-- The edge length is positive -/
  edge_positive : edge_length > 0

/-- Theorem stating that a hexagonal pyramid cannot have all edges of equal length -/
theorem hexagonal_pyramid_not_regular : ¬∃ (p : RegularPyramid 6), True :=
sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_not_regular_l1902_190221


namespace NUMINAMATH_CALUDE_cube_iff_diagonal_perpendicular_l1902_190253

/-- A rectangular parallelepiped -/
structure RectangularParallelepiped where
  -- Add necessary fields and properties here

/-- Predicate for a rectangular parallelepiped being a cube -/
def is_cube (S : RectangularParallelepiped) : Prop :=
  sorry

/-- Predicate for the diagonal perpendicularity property -/
def diagonal_perpendicular_property (S : RectangularParallelepiped) : Prop :=
  sorry

/-- Theorem stating the equivalence of the cube property and the diagonal perpendicularity property -/
theorem cube_iff_diagonal_perpendicular (S : RectangularParallelepiped) :
  is_cube S ↔ diagonal_perpendicular_property S :=
sorry

end NUMINAMATH_CALUDE_cube_iff_diagonal_perpendicular_l1902_190253


namespace NUMINAMATH_CALUDE_range_of_a_for_inequality_l1902_190217

theorem range_of_a_for_inequality (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ 2^x * (3*x + a) < 1) ↔ a < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_inequality_l1902_190217


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l1902_190245

theorem smallest_non_factor_product (a b : ℕ+) : 
  a ≠ b →
  a ∣ 48 →
  b ∣ 48 →
  ¬(a * b ∣ 48) →
  (∀ (c d : ℕ+), c ≠ d → c ∣ 48 → d ∣ 48 → ¬(c * d ∣ 48) → a * b ≤ c * d) →
  a * b = 18 := by
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l1902_190245


namespace NUMINAMATH_CALUDE_unique_triangle_l1902_190208

-- Define the properties of the triangle
def is_valid_triangle (a b c : ℕ) : Prop :=
  ∃ (p q : ℕ) (m n k : ℕ+),
    Prime p ∧ Prime q ∧
    a = p ^ (m : ℕ) ∧
    b = q ^ (n : ℕ) ∧
    c = 2 * k + 1 ∧
    a * a + b * b = c * c

-- State the theorem
theorem unique_triangle :
  ∀ a b c : ℕ,
    is_valid_triangle a b c →
    ((a = 3 ∧ b = 4) ∨ (a = 4 ∧ b = 3)) ∧ c = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_triangle_l1902_190208


namespace NUMINAMATH_CALUDE_largest_number_in_block_l1902_190293

/-- Represents a 2x3 block of numbers in a 10-column table -/
structure NumberBlock where
  first_number : ℕ
  deriving Repr

/-- The sum of numbers in a 2x3 block -/
def block_sum (block : NumberBlock) : ℕ :=
  6 * block.first_number + 36

theorem largest_number_in_block (block : NumberBlock) 
  (h1 : block.first_number ≥ 1)
  (h2 : block.first_number + 12 ≤ 100)
  (h3 : block_sum block = 480) :
  (block.first_number + 12 = 86) :=
sorry

end NUMINAMATH_CALUDE_largest_number_in_block_l1902_190293


namespace NUMINAMATH_CALUDE_no_valid_tiling_l1902_190223

/-- Represents a tile on the grid -/
inductive Tile
  | OneByFour : Tile
  | TwoByTwo : Tile

/-- Represents a position on the 8x8 grid -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a placement of a tile on the grid -/
structure Placement :=
  (tile : Tile)
  (position : Position)

/-- Represents a tiling of the 8x8 grid -/
def Tiling := List Placement

/-- Checks if a tiling is valid (covers the entire grid without overlaps) -/
def isValidTiling (t : Tiling) : Prop := sorry

/-- Checks if a tiling uses exactly 15 1x4 tiles and 1 2x2 tile -/
def hasCorrectTileCount (t : Tiling) : Prop := sorry

/-- The main theorem stating that no valid tiling exists with the given constraints -/
theorem no_valid_tiling :
  ¬ ∃ (t : Tiling), isValidTiling t ∧ hasCorrectTileCount t := by
  sorry

end NUMINAMATH_CALUDE_no_valid_tiling_l1902_190223


namespace NUMINAMATH_CALUDE_fry_costs_60_cents_l1902_190250

-- Define the costs in cents
def burger_cost : ℕ := 80
def soda_cost : ℕ := 60

-- Define the total costs of Alice's and Bill's purchases in cents
def alice_total : ℕ := 420
def bill_total : ℕ := 340

-- Define the function to calculate the cost of a fry
def fry_cost : ℕ :=
  alice_total - 3 * burger_cost - 2 * soda_cost

-- Theorem to prove
theorem fry_costs_60_cents :
  fry_cost = 60 ∧
  2 * burger_cost + soda_cost + 2 * fry_cost = bill_total :=
by sorry

end NUMINAMATH_CALUDE_fry_costs_60_cents_l1902_190250


namespace NUMINAMATH_CALUDE_cubic_root_function_l1902_190296

theorem cubic_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * (64 : ℝ)^(1/3) ∧ y = 4 * Real.sqrt 3) →
  k * (8 : ℝ)^(1/3) = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_function_l1902_190296


namespace NUMINAMATH_CALUDE_equal_split_contribution_l1902_190233

def earnings : List ℝ := [18, 22, 30, 38, 45]

theorem equal_split_contribution (total : ℝ) (equal_share : ℝ) :
  total = earnings.sum →
  equal_share = total / 5 →
  45 - equal_share = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_equal_split_contribution_l1902_190233


namespace NUMINAMATH_CALUDE_inequality_proof_l1902_190210

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / Real.sqrt b) + (b / Real.sqrt a) ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1902_190210


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1902_190229

theorem sufficient_not_necessary_condition :
  (∀ b : ℝ, b ∈ Set.Ioo 0 4 → ∀ x : ℝ, b * x^2 - b * x + 1 > 0) ∧
  (∃ b : ℝ, b ∉ Set.Ioo 0 4 ∧ ∀ x : ℝ, b * x^2 - b * x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1902_190229


namespace NUMINAMATH_CALUDE_impossible_valid_arrangement_l1902_190249

/-- Represents the colors of chips -/
inductive Color
| Blue
| Red
| Green

/-- Represents a circular arrangement of chips -/
def CircularArrangement := List Color

/-- Represents a swap operation -/
inductive SwapOperation
| BlueRed
| BlueGreen

/-- Initial arrangement of chips -/
def initial_arrangement : CircularArrangement :=
  (List.replicate 40 Color.Blue) ++ (List.replicate 30 Color.Red) ++ (List.replicate 20 Color.Green)

/-- Checks if an arrangement has no adjacent chips of the same color -/
def is_valid_arrangement (arr : CircularArrangement) : Bool :=
  sorry

/-- Applies a swap operation to an arrangement -/
def apply_swap (arr : CircularArrangement) (op : SwapOperation) : CircularArrangement :=
  sorry

/-- Theorem stating that it's impossible to achieve a valid arrangement -/
theorem impossible_valid_arrangement :
  ∀ (ops : List SwapOperation),
    let final_arrangement := ops.foldl apply_swap initial_arrangement
    ¬ (is_valid_arrangement final_arrangement) :=
  sorry

end NUMINAMATH_CALUDE_impossible_valid_arrangement_l1902_190249


namespace NUMINAMATH_CALUDE_clock_angle_at_two_thirty_l1902_190234

/-- The measure of the smaller angle formed by the hour-hand and minute-hand of a clock at 2:30 -/
def clock_angle : ℝ := 105

/-- The number of degrees in a full circle on a clock -/
def full_circle : ℝ := 360

/-- The number of hours on a clock -/
def clock_hours : ℕ := 12

/-- The hour component of the time -/
def hour : ℕ := 2

/-- The minute component of the time -/
def minute : ℕ := 30

theorem clock_angle_at_two_thirty :
  clock_angle = min (|hour_angle - minute_angle|) (full_circle - |hour_angle - minute_angle|) :=
by
  sorry
where
  /-- The angle of the hour hand from 12 o'clock position -/
  hour_angle : ℝ := (hour + minute / 60) * (full_circle / clock_hours)
  /-- The angle of the minute hand from 12 o'clock position -/
  minute_angle : ℝ := minute * (full_circle / 60)

#check clock_angle_at_two_thirty

end NUMINAMATH_CALUDE_clock_angle_at_two_thirty_l1902_190234


namespace NUMINAMATH_CALUDE_milburg_grown_ups_l1902_190255

/-- The number of grown-ups in Milburg -/
def grown_ups (total_population children : ℕ) : ℕ :=
  total_population - children

/-- Proof that the number of grown-ups in Milburg is 5256 -/
theorem milburg_grown_ups :
  grown_ups 8243 2987 = 5256 := by
  sorry

end NUMINAMATH_CALUDE_milburg_grown_ups_l1902_190255


namespace NUMINAMATH_CALUDE_total_crayons_l1902_190212

/-- Given that each child has 12 crayons and there are 18 children, 
    prove that the total number of crayons is 216. -/
theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) 
  (h1 : crayons_per_child = 12) (h2 : num_children = 18) : 
  crayons_per_child * num_children = 216 := by
sorry

end NUMINAMATH_CALUDE_total_crayons_l1902_190212


namespace NUMINAMATH_CALUDE_tangent_point_on_reciprocal_curve_l1902_190291

/-- Prove that the point of tangency on y = 1/x, where the tangent line passes through (0,2), is (1,1) -/
theorem tangent_point_on_reciprocal_curve :
  ∀ m n : ℝ,
  (n = 1 / m) →                         -- Point (m,n) is on the curve y = 1/x
  (2 - n) / m = -1 / (m^2) →            -- Tangent line passes through (0,2) with slope -1/m^2
  (m = 1 ∧ n = 1) :=                    -- The point of tangency is (1,1)
by sorry

end NUMINAMATH_CALUDE_tangent_point_on_reciprocal_curve_l1902_190291


namespace NUMINAMATH_CALUDE_sequence_a11_value_l1902_190259

/-- Given a sequence {aₙ} with sum of first n terms Sₙ,
    prove that a₁₁ = -2 given 4Sₙ = 2aₙ - n² + 7n for all positive integers n -/
theorem sequence_a11_value
  (a : ℕ → ℤ)  -- Sequence {aₙ}
  (S : ℕ → ℤ)  -- Sum function Sₙ
  (h : ∀ n : ℕ, n > 0 → 4 * S n = 2 * a n - n^2 + 7 * n) :  -- Given condition
  a 11 = -2 :=
sorry

end NUMINAMATH_CALUDE_sequence_a11_value_l1902_190259


namespace NUMINAMATH_CALUDE_function_range_l1902_190203

theorem function_range (t : ℝ) : 
  (∃ x : ℝ, t ≤ x ∧ x ≤ t + 2 ∧ x^2 + t*x - 12 ≤ 0) → 
  -4 ≤ t ∧ t ≤ Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l1902_190203


namespace NUMINAMATH_CALUDE_sum_with_rearrangement_not_all_nines_l1902_190275

def digit_sum (n : ℕ) : ℕ := sorry

def is_digit_rearrangement (n m : ℕ) : Prop :=
  digit_sum n = digit_sum m

def repeated_nines (k : ℕ) : ℕ := sorry

theorem sum_with_rearrangement_not_all_nines (n : ℕ) :
  ∀ m : ℕ, is_digit_rearrangement n m → n + m ≠ repeated_nines 125 := by sorry

end NUMINAMATH_CALUDE_sum_with_rearrangement_not_all_nines_l1902_190275


namespace NUMINAMATH_CALUDE_power_plus_mod_five_l1902_190272

theorem power_plus_mod_five : (2^2018 + 2019) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_plus_mod_five_l1902_190272


namespace NUMINAMATH_CALUDE_distance_center_to_plane_l1902_190266

/-- Given a sphere and three points on its surface, calculate the distance from the center to the plane of the triangle formed by the points. -/
theorem distance_center_to_plane (S : Real) (AB BC AC : Real) (h1 : S = 20 * Real.pi) (h2 : BC = 2 * Real.sqrt 3) (h3 : AB = 2) (h4 : AC = 2) : 
  ∃ d : Real, d = 1 ∧ d = Real.sqrt (((S / (4 * Real.pi))^(1/2 : Real))^2 - (BC / (2 * Real.sin (Real.arccos ((AC^2 + AB^2 - BC^2) / (2 * AC * AB)))))^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_center_to_plane_l1902_190266


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1902_190267

/-- Given a geometric sequence {a_n} with common ratio q and sum of first n terms S_n,
    prove that if q = 2 and S_5 = 1, then S_10 = 33. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- sum formula
  q = 2 →
  S 5 = 1 →
  S 10 = 33 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1902_190267


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l1902_190265

theorem shaded_area_fraction (total_squares : ℕ) (shaded_squares : ℕ) :
  total_squares = 6 →
  shaded_squares = 2 →
  (shaded_squares : ℚ) / total_squares = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l1902_190265


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1902_190282

theorem sin_alpha_value (α : Real) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 4) 
  (h3 : Real.sin α * Real.cos α = 3 * Real.sqrt 7 / 16) : 
  Real.sin α = Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1902_190282


namespace NUMINAMATH_CALUDE_race_finish_orders_l1902_190284

theorem race_finish_orders (n : Nat) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_orders_l1902_190284


namespace NUMINAMATH_CALUDE_solve_equation_l1902_190280

theorem solve_equation (x y : ℝ) (h1 : y = (x^2 - 9) / (x - 3)) 
  (h2 : y = 3*x) (h3 : x ≠ 3) : x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1902_190280


namespace NUMINAMATH_CALUDE_work_multiple_l1902_190286

/-- If a person can complete one unit of work in 5 days, and takes 15 days to complete 
    a certain amount of the same type of work, then the amount of work completed in 15 days 
    is 3 times the original unit of work. -/
theorem work_multiple (original_days : ℕ) (new_days : ℕ) (work_multiple : ℚ) :
  original_days = 5 →
  new_days = 15 →
  work_multiple = (new_days : ℚ) / (original_days : ℚ) →
  work_multiple = 3 := by
sorry

end NUMINAMATH_CALUDE_work_multiple_l1902_190286


namespace NUMINAMATH_CALUDE_average_difference_l1902_190257

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 35) 
  (h2 : (b + c) / 2 = 80) : 
  c - a = 90 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l1902_190257


namespace NUMINAMATH_CALUDE_local_max_condition_l1902_190279

theorem local_max_condition (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ IsLocalMax (fun x => Real.exp x + a * x) x) →
  a < -1 := by sorry

end NUMINAMATH_CALUDE_local_max_condition_l1902_190279


namespace NUMINAMATH_CALUDE_largest_similar_triangle_exists_l1902_190274

-- Define the types for points and triangles
def Point : Type := ℝ × ℝ
def Triangle : Type := Point × Point × Point

-- Define the properties of the triangles
axiom similar_triangles (T1 T2 : Triangle) : Prop
axiom point_on_line (P Q R : Point) : Prop
axiom triangle_area (T : Triangle) : ℝ

-- Define the given triangles
variable (A B : Triangle)

-- Define the constructed triangle
variable (M : Triangle)

-- Define the conditions
variable (h1 : point_on_line (A.1) (M.2.1) (M.2.2))
variable (h2 : point_on_line (A.2.1) (A.1) (A.2.2))
variable (h3 : point_on_line (A.2.2) (A.1) (A.2.1))
variable (h4 : similar_triangles M B)

-- State the theorem
theorem largest_similar_triangle_exists :
  ∃ (M : Triangle), 
    point_on_line (A.1) (M.2.1) (M.2.2) ∧
    point_on_line (A.2.1) (A.1) (A.2.2) ∧
    point_on_line (A.2.2) (A.1) (A.2.1) ∧
    similar_triangles M B ∧
    ∀ (M' : Triangle), 
      (point_on_line (A.1) (M'.2.1) (M'.2.2) ∧
       point_on_line (A.2.1) (A.1) (A.2.2) ∧
       point_on_line (A.2.2) (A.1) (A.2.1) ∧
       similar_triangles M' B) →
      triangle_area M ≥ triangle_area M' :=
sorry

end NUMINAMATH_CALUDE_largest_similar_triangle_exists_l1902_190274


namespace NUMINAMATH_CALUDE_ariels_age_multiplier_l1902_190252

theorem ariels_age_multiplier :
  let current_age : ℕ := 5
  let years_passed : ℕ := 15
  let future_age : ℕ := current_age + years_passed
  ∃ (multiplier : ℕ), future_age = multiplier * current_age ∧ multiplier = 4 :=
by sorry

end NUMINAMATH_CALUDE_ariels_age_multiplier_l1902_190252


namespace NUMINAMATH_CALUDE_centroid_maximizes_dist_product_l1902_190268

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- Distance from a point to a line defined by two points --/
def distToLine (P : Point) (A B : Point) : ℝ := sorry

/-- The centroid of a triangle --/
def centroid (t : Triangle) : Point := sorry

/-- Product of distances from a point to the sides of a triangle --/
def distProduct (P : Point) (t : Triangle) : ℝ := 
  distToLine P t.A t.B * distToLine P t.B t.C * distToLine P t.C t.A

/-- Predicate to check if a point is inside a triangle --/
def isInside (P : Point) (t : Triangle) : Prop := sorry

theorem centroid_maximizes_dist_product (t : Triangle) :
  ∀ P, isInside P t → distProduct P t ≤ distProduct (centroid t) t :=
sorry

end NUMINAMATH_CALUDE_centroid_maximizes_dist_product_l1902_190268


namespace NUMINAMATH_CALUDE_triangle_value_l1902_190205

theorem triangle_value (triangle p : ℝ) 
  (eq1 : 2 * triangle + p = 72)
  (eq2 : triangle + p + 2 * triangle = 128) :
  triangle = 56 := by
sorry

end NUMINAMATH_CALUDE_triangle_value_l1902_190205


namespace NUMINAMATH_CALUDE_product_sum_inequality_l1902_190201

theorem product_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_inequality_l1902_190201


namespace NUMINAMATH_CALUDE_card_58_is_six_l1902_190225

/-- Represents a playing card value -/
inductive CardValue
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Converts a natural number to a card value -/
def natToCardValue (n : ℕ) : CardValue :=
  match n % 13 with
  | 0 => CardValue.Ace
  | 1 => CardValue.Two
  | 2 => CardValue.Three
  | 3 => CardValue.Four
  | 4 => CardValue.Five
  | 5 => CardValue.Six
  | 6 => CardValue.Seven
  | 7 => CardValue.Eight
  | 8 => CardValue.Nine
  | 9 => CardValue.Ten
  | 10 => CardValue.Jack
  | 11 => CardValue.Queen
  | _ => CardValue.King

theorem card_58_is_six :
  natToCardValue 57 = CardValue.Six :=
by sorry

end NUMINAMATH_CALUDE_card_58_is_six_l1902_190225


namespace NUMINAMATH_CALUDE_cats_in_academy_l1902_190242

/-- The number of cats that can jump -/
def jump : ℕ := 40

/-- The number of cats that can climb -/
def climb : ℕ := 25

/-- The number of cats that can hunt -/
def hunt : ℕ := 30

/-- The number of cats that can jump and climb -/
def jump_and_climb : ℕ := 10

/-- The number of cats that can climb and hunt -/
def climb_and_hunt : ℕ := 15

/-- The number of cats that can jump and hunt -/
def jump_and_hunt : ℕ := 12

/-- The number of cats that can do all three skills -/
def all_skills : ℕ := 5

/-- The number of cats that cannot perform any skills -/
def no_skills : ℕ := 6

/-- The total number of cats in the academy -/
def total_cats : ℕ := 69

theorem cats_in_academy :
  total_cats = jump + climb + hunt - jump_and_climb - climb_and_hunt - jump_and_hunt + all_skills + no_skills := by
  sorry

end NUMINAMATH_CALUDE_cats_in_academy_l1902_190242


namespace NUMINAMATH_CALUDE_order_of_abc_l1902_190231

theorem order_of_abc : 
  let a : ℝ := 1 / (6 * Real.sqrt 15)
  let b : ℝ := (3/4) * Real.sin (1/60)
  let c : ℝ := Real.log (61/60)
  b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l1902_190231


namespace NUMINAMATH_CALUDE_rectangle_length_l1902_190236

theorem rectangle_length (l w : ℝ) (h1 : l = 4 * w) (h2 : l * w = 100) : l = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l1902_190236


namespace NUMINAMATH_CALUDE_sum_of_divisors_143_l1902_190224

theorem sum_of_divisors_143 : (Finset.filter (· ∣ 143) (Finset.range 144)).sum id = 168 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_143_l1902_190224


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l1902_190254

theorem painted_cube_theorem (n : ℕ) (h1 : n > 2) :
  (6 * (n - 2)^2 = (n - 2)^3) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l1902_190254


namespace NUMINAMATH_CALUDE_kim_cherry_difference_l1902_190276

/-- The number of questions Nicole answered correctly -/
def nicole_correct : ℕ := 22

/-- The number of questions Cherry answered correctly -/
def cherry_correct : ℕ := 17

/-- The number of questions Kim answered correctly -/
def kim_correct : ℕ := nicole_correct + 3

theorem kim_cherry_difference : kim_correct - cherry_correct = 8 := by
  sorry

end NUMINAMATH_CALUDE_kim_cherry_difference_l1902_190276


namespace NUMINAMATH_CALUDE_sqrt_two_divided_by_sqrt_two_minus_one_l1902_190288

theorem sqrt_two_divided_by_sqrt_two_minus_one :
  Real.sqrt 2 / (Real.sqrt 2 - 1) = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_divided_by_sqrt_two_minus_one_l1902_190288


namespace NUMINAMATH_CALUDE_max_value_of_S_l1902_190237

theorem max_value_of_S (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let S := min x (min (y + 1/x) (1/y))
  ∃ (max_S : ℝ), max_S = Real.sqrt 2 ∧
    (∀ x' y' : ℝ, x' > 0 → y' > 0 → 
      min x' (min (y' + 1/x') (1/y')) ≤ max_S) ∧
    S = max_S ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_S_l1902_190237


namespace NUMINAMATH_CALUDE_total_food_service_employees_l1902_190211

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


end NUMINAMATH_CALUDE_total_food_service_employees_l1902_190211


namespace NUMINAMATH_CALUDE_harry_age_l1902_190283

/-- Represents the ages of the people in the problem -/
structure Ages where
  kiarra : ℕ
  bea : ℕ
  job : ℕ
  figaro : ℕ
  harry : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.kiarra = 2 * ages.bea ∧
  ages.job = 3 * ages.bea ∧
  ages.figaro = ages.job + 7 ∧
  2 * ages.harry = ages.figaro ∧
  ages.kiarra = 30

/-- The theorem stating that under the given conditions, Harry's age is 26 -/
theorem harry_age (ages : Ages) :
  problem_conditions ages → ages.harry = 26 := by
  sorry

end NUMINAMATH_CALUDE_harry_age_l1902_190283


namespace NUMINAMATH_CALUDE_water_tank_capacity_l1902_190292

theorem water_tank_capacity (c : ℝ) (h1 : c > 0) :
  (1 / 5 : ℝ) * c + 6 = (1 / 3 : ℝ) * c → c = 45 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l1902_190292


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1902_190218

-- Define what a quadratic equation is
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific function we're checking
def f (x : ℝ) : ℝ := x^2 + 6*x

-- Theorem statement
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1902_190218


namespace NUMINAMATH_CALUDE_fourth_power_sum_l1902_190271

theorem fourth_power_sum (a b t : ℝ) 
  (h1 : a + b = t) 
  (h2 : a^2 + b^2 = t) 
  (h3 : a^3 + b^3 = t) : 
  a^4 + b^4 = t := by sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l1902_190271


namespace NUMINAMATH_CALUDE_economics_test_absentees_l1902_190299

theorem economics_test_absentees (total_students : Nat) 
  (correct_q1 : Nat) (correct_q2 : Nat) (correct_both : Nat) :
  total_students = 30 →
  correct_q1 = 25 →
  correct_q2 = 22 →
  correct_both = 22 →
  correct_both = correct_q2 →
  total_students - correct_q2 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_economics_test_absentees_l1902_190299


namespace NUMINAMATH_CALUDE_vasya_counts_more_apples_CD_l1902_190261

/-- Represents the number of apple trees around the circular lake -/
def n : ℕ := sorry

/-- Represents the total number of apples on all trees -/
def m : ℕ := sorry

/-- Represents the number of trees Vasya counts from A to B -/
def vasya_trees_AB : ℕ := n / 3

/-- Represents the number of trees Petya counts from A to B -/
def petya_trees_AB : ℕ := 2 * n / 3

/-- Represents the number of apples Vasya counts from A to B -/
def vasya_apples_AB : ℕ := m / 8

/-- Represents the number of apples Petya counts from A to B -/
def petya_apples_AB : ℕ := 7 * m / 8

/-- Represents the number of trees Vasya counts from B to C -/
def vasya_trees_BC : ℕ := n / 3

/-- Represents the number of trees Petya counts from B to C -/
def petya_trees_BC : ℕ := 2 * n / 3

/-- Represents the number of apples Vasya counts from B to C -/
def vasya_apples_BC : ℕ := m / 8

/-- Represents the number of apples Petya counts from B to C -/
def petya_apples_BC : ℕ := 7 * m / 8

/-- Represents the number of trees Vasya counts from C to D -/
def vasya_trees_CD : ℕ := n / 3

/-- Represents the number of trees Petya counts from C to D -/
def petya_trees_CD : ℕ := 2 * n / 3

/-- Theorem stating that Vasya counts 3 times more apples than Petya from C to D -/
theorem vasya_counts_more_apples_CD :
  (m - vasya_apples_AB - vasya_apples_BC) = 3 * (m - petya_apples_AB - petya_apples_BC) :=
by sorry

end NUMINAMATH_CALUDE_vasya_counts_more_apples_CD_l1902_190261


namespace NUMINAMATH_CALUDE_ahead_of_schedule_l1902_190278

/-- Represents the worker's production plan -/
def WorkerPlan (total_parts : ℕ) (total_days : ℕ) (initial_rate : ℕ) (initial_days : ℕ) (x : ℕ) : Prop :=
  initial_rate * initial_days + (total_days - initial_days) * x > total_parts

/-- Theorem stating the condition for completing the task ahead of schedule -/
theorem ahead_of_schedule (x : ℕ) :
  WorkerPlan 408 15 24 3 x ↔ 24 * 3 + (15 - 3) * x > 408 :=
by sorry

end NUMINAMATH_CALUDE_ahead_of_schedule_l1902_190278


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l1902_190214

/-- Definition of an ellipse with semi-major axis 4 and semi-minor axis 3 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 16) + (p.2^2 / 9) = 1}

/-- The right focus of the ellipse -/
def F₂ : ℝ × ℝ := sorry

/-- The left focus of the ellipse -/
def F₁ : ℝ × ℝ := sorry

/-- Theorem: The perimeter of triangle AF₁B is 16 for any A and B on the ellipse -/
theorem ellipse_triangle_perimeter 
  (A B : ℝ × ℝ) 
  (hA : A ∈ Ellipse) 
  (hB : B ∈ Ellipse) : 
  dist A F₁ + dist B F₁ + dist A B = 16 := 
sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l1902_190214


namespace NUMINAMATH_CALUDE_power_product_simplification_l1902_190219

theorem power_product_simplification :
  (-3/2 : ℚ)^2023 * (-2/3 : ℚ)^2022 = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_power_product_simplification_l1902_190219


namespace NUMINAMATH_CALUDE_tan_negative_405_degrees_l1902_190247

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_405_degrees_l1902_190247


namespace NUMINAMATH_CALUDE_probability_heart_joker_value_l1902_190239

/-- A deck of cards with 54 cards total, including 13 hearts and 2 jokers -/
structure Deck :=
  (total : Nat)
  (hearts : Nat)
  (jokers : Nat)
  (h_total : total = 54)
  (h_hearts : hearts = 13)
  (h_jokers : jokers = 2)

/-- The probability of drawing a heart first and a joker second from the deck -/
def probability_heart_joker (d : Deck) : ℚ :=
  (d.hearts : ℚ) / d.total * d.jokers / (d.total - 1)

/-- Theorem stating the probability of drawing a heart first and a joker second -/
theorem probability_heart_joker_value (d : Deck) :
  probability_heart_joker d = 13 / 1419 := by
  sorry

#eval (13 : ℚ) / 1419

end NUMINAMATH_CALUDE_probability_heart_joker_value_l1902_190239


namespace NUMINAMATH_CALUDE_exists_term_with_nine_l1902_190216

/-- Represents an arithmetic progression with natural number first term and common difference -/
structure ArithmeticProgression where
  first_term : ℕ
  common_difference : ℕ

/-- Predicate to check if a natural number contains the digit 9 -/
def contains_digit_nine (n : ℕ) : Prop := sorry

/-- Theorem stating that there exists a term in the arithmetic progression containing the digit 9 -/
theorem exists_term_with_nine (ap : ArithmeticProgression) : 
  ∃ (k : ℕ), contains_digit_nine (ap.first_term + k * ap.common_difference) := by sorry

end NUMINAMATH_CALUDE_exists_term_with_nine_l1902_190216


namespace NUMINAMATH_CALUDE_deposit_ratio_is_39_11_l1902_190256

/-- Represents the deposit amounts and their ratio -/
structure DepositRatio where
  mark : ℚ
  bryan : ℚ
  total : ℚ
  ratio : ℚ

/-- Theorem stating the deposit ratio given the conditions -/
theorem deposit_ratio_is_39_11 (d : DepositRatio) 
  (h1 : d.mark = 88)
  (h2 : d.bryan < d.mark * (d.bryan / d.mark).floor)
  (h3 : d.mark + d.bryan = d.total)
  (h4 : d.total = 400)
  (h5 : d.ratio = d.bryan / d.mark) :
  d.ratio = 39 / 11 := by
  sorry

end NUMINAMATH_CALUDE_deposit_ratio_is_39_11_l1902_190256
