import Mathlib

namespace NUMINAMATH_CALUDE_mike_final_cards_l3273_327396

def mike_cards (initial : ℕ) (received : ℕ) (traded : ℕ) : ℕ :=
  initial + received - traded

theorem mike_final_cards :
  mike_cards 64 18 20 = 62 := by
  sorry

end NUMINAMATH_CALUDE_mike_final_cards_l3273_327396


namespace NUMINAMATH_CALUDE_number_divided_by_three_l3273_327399

theorem number_divided_by_three : ∃ x : ℤ, x / 3 = x - 24 ∧ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_three_l3273_327399


namespace NUMINAMATH_CALUDE_success_rate_paradox_l3273_327397

structure Player :=
  (name : String)
  (attempts_season1 : ℕ)
  (successes_season1 : ℕ)
  (attempts_season2 : ℕ)
  (successes_season2 : ℕ)

def success_rate (attempts : ℕ) (successes : ℕ) : ℚ :=
  if attempts = 0 then 0 else (successes : ℚ) / (attempts : ℚ)

def combined_success_rate (p : Player) : ℚ :=
  success_rate (p.attempts_season1 + p.attempts_season2) (p.successes_season1 + p.successes_season2)

theorem success_rate_paradox (p1 p2 : Player) :
  (success_rate p1.attempts_season1 p1.successes_season1 > success_rate p2.attempts_season1 p2.successes_season1) ∧
  (success_rate p1.attempts_season2 p1.successes_season2 > success_rate p2.attempts_season2 p2.successes_season2) ∧
  (combined_success_rate p1 < combined_success_rate p2) :=
sorry

end NUMINAMATH_CALUDE_success_rate_paradox_l3273_327397


namespace NUMINAMATH_CALUDE_triangle_side_length_l3273_327371

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = Real.sqrt 3) 
  (h2 : B = π / 4) 
  (h3 : A = π / 3) 
  (h4 : C = π - A - B) 
  (h5 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h6 : 0 < A ∧ A < π) 
  (h7 : 0 < B ∧ B < π) 
  (h8 : 0 < C ∧ C < π) 
  (h9 : a / Real.sin A = b / Real.sin B) 
  (h10 : a / Real.sin A = c / Real.sin C) 
  (h11 : c^2 = a^2 + b^2 - 2*a*b*Real.cos C) : 
  c = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3273_327371


namespace NUMINAMATH_CALUDE_walking_time_calculation_l3273_327328

/-- Proves that given a distance that takes 40 minutes to cover at a speed of 16.5 kmph,
    it will take 165 minutes to cover the same distance at a speed of 4 kmph. -/
theorem walking_time_calculation (distance : ℝ) : 
  distance = 16.5 * (40 / 60) → distance / 4 * 60 = 165 := by
  sorry

end NUMINAMATH_CALUDE_walking_time_calculation_l3273_327328


namespace NUMINAMATH_CALUDE_probability_a_squared_geq_4b_l3273_327384

-- Define the set of numbers
def S : Set Nat := {1, 2, 3, 4}

-- Define the condition
def condition (a b : Nat) : Prop := a^2 ≥ 4*b

-- Define the total number of ways to select two numbers
def total_selections : Nat := 12

-- Define the number of favorable selections
def favorable_selections : Nat := 6

-- State the theorem
theorem probability_a_squared_geq_4b :
  (favorable_selections : ℚ) / total_selections = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_a_squared_geq_4b_l3273_327384


namespace NUMINAMATH_CALUDE_non_yellow_houses_count_l3273_327336

-- Define the number of houses of each color
def yellow_houses : ℕ := 30
def green_houses : ℕ := 90
def red_houses : ℕ := 70
def blue_houses : ℕ := 60
def pink_houses : ℕ := 50

-- State the theorem
theorem non_yellow_houses_count :
  -- Conditions
  (green_houses = 3 * yellow_houses) →
  (red_houses = yellow_houses + 40) →
  (green_houses = 90) →
  (blue_houses = (green_houses + yellow_houses) / 2) →
  (pink_houses = red_houses / 2 + 15) →
  -- Conclusion
  (green_houses + red_houses + blue_houses + pink_houses = 270) :=
by
  sorry

end NUMINAMATH_CALUDE_non_yellow_houses_count_l3273_327336


namespace NUMINAMATH_CALUDE_stating_ladder_of_twos_theorem_l3273_327332

/-- 
A function that represents the number of distinct integers obtainable 
from a ladder of n twos by placing nested parentheses.
-/
def ladder_of_twos (n : ℕ) : ℕ :=
  if n ≥ 3 then 2^(n-3) else 0

/-- 
Theorem stating that for n ≥ 3, the number of distinct integers obtainable 
from a ladder of n twos by placing nested parentheses is 2^(n-3).
-/
theorem ladder_of_twos_theorem (n : ℕ) (h : n ≥ 3) : 
  ladder_of_twos n = 2^(n-3) := by
  sorry

end NUMINAMATH_CALUDE_stating_ladder_of_twos_theorem_l3273_327332


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l3273_327302

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l3273_327302


namespace NUMINAMATH_CALUDE_star_op_power_equality_l3273_327370

def star_op (a b : ℕ+) : ℕ+ := a ^ (b.val ^ 2)

theorem star_op_power_equality (a b n : ℕ+) :
  (star_op a b) ^ n.val = star_op a (n * b) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_star_op_power_equality_l3273_327370


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l3273_327373

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l3273_327373


namespace NUMINAMATH_CALUDE_max_value_of_a_l3273_327355

theorem max_value_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, a ≤ (1 - x) / x + Real.log x) → 
  a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3273_327355


namespace NUMINAMATH_CALUDE_intersection_area_of_bisected_octahedron_l3273_327358

-- Define a regular octahedron
structure RegularOctahedron :=
  (side_length : ℝ)

-- Define the intersection polygon
structure IntersectionPolygon :=
  (octahedron : RegularOctahedron)
  (is_parallel : Bool)
  (is_bisecting : Bool)

-- Define the area of the intersection polygon
def intersection_area (p : IntersectionPolygon) : ℝ :=
  sorry

-- Theorem statement
theorem intersection_area_of_bisected_octahedron 
  (o : RegularOctahedron) 
  (p : IntersectionPolygon) 
  (h1 : o.side_length = 2) 
  (h2 : p.octahedron = o) 
  (h3 : p.is_parallel = true) 
  (h4 : p.is_bisecting = true) : 
  intersection_area p = 9 * Real.sqrt 3 / 8 :=
sorry

end NUMINAMATH_CALUDE_intersection_area_of_bisected_octahedron_l3273_327358


namespace NUMINAMATH_CALUDE_table_loss_percentage_l3273_327301

theorem table_loss_percentage 
  (cost_price selling_price : ℝ) 
  (h1 : 15 * cost_price = 20 * selling_price) 
  (discount_rate : ℝ) (h2 : discount_rate = 0.1)
  (tax_rate : ℝ) (h3 : tax_rate = 0.08) : 
  (cost_price * (1 - discount_rate) - selling_price * (1 + tax_rate)) / cost_price = 0.09 := by
sorry

end NUMINAMATH_CALUDE_table_loss_percentage_l3273_327301


namespace NUMINAMATH_CALUDE_divisibility_of_fraction_l3273_327382

theorem divisibility_of_fraction (a b n : ℕ) (h1 : a ≠ b) (h2 : n ∣ (a^n - b^n)) :
  n ∣ ((a^n - b^n) / (a - b)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_fraction_l3273_327382


namespace NUMINAMATH_CALUDE_complete_square_equivalence_l3273_327369

theorem complete_square_equivalence :
  let f₁ : ℝ → ℝ := λ x ↦ x^2 - 2*x + 3
  let f₂ : ℝ → ℝ := λ x ↦ 3*x^2 + 6*x - 1
  let f₃ : ℝ → ℝ := λ x ↦ -2*x^2 + 3*x - 2
  let g₁ : ℝ → ℝ := λ x ↦ (x - 1)^2 + 2
  let g₂ : ℝ → ℝ := λ x ↦ 3*(x + 1)^2 - 4
  let g₃ : ℝ → ℝ := λ x ↦ -2*(x - 3/4)^2 - 7/8
  (∀ x : ℝ, f₁ x = g₁ x) ∧
  (∀ x : ℝ, f₂ x = g₂ x) ∧
  (∀ x : ℝ, f₃ x = g₃ x) := by
  sorry

end NUMINAMATH_CALUDE_complete_square_equivalence_l3273_327369


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sin_sum_l3273_327392

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sin_sum
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 7 + a 13 = 4 * Real.pi) :
  Real.sin (a 2 + a 12) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sin_sum_l3273_327392


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l3273_327346

theorem quadratic_equation_equivalence :
  ∃ (m n : ℝ), (∀ x, 4 * x^2 + 8 * x - 448 = 0 ↔ (x + m)^2 = n) ∧ m = 1 ∧ n = 113 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l3273_327346


namespace NUMINAMATH_CALUDE_unique_solution_exponential_system_l3273_327366

theorem unique_solution_exponential_system :
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 →
  (x^y = z ∧ y^z = x ∧ z^x = y) →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_system_l3273_327366


namespace NUMINAMATH_CALUDE_preferred_sequence_bound_l3273_327341

/-- A sequence of matrices is preferred if it satisfies certain conditions -/
def IsPreferred (k n : ℕ) (A : Fin k → Matrix (Fin n) (Fin n) ℝ) : Prop :=
  ∀ i : Fin k, A i ^ 2 ≠ 0 ∧ ∀ j : Fin k, i ≠ j → A i * A j = 0

/-- The main theorem: for any preferred sequence, k ≤ n -/
theorem preferred_sequence_bound {k n : ℕ} (hk : k > 0) (hn : n > 0)
  (A : Fin k → Matrix (Fin n) (Fin n) ℝ) (h : IsPreferred k n A) :
  k ≤ n := by
  sorry


end NUMINAMATH_CALUDE_preferred_sequence_bound_l3273_327341


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_self_l3273_327393

theorem product_of_fractions_equals_self (n : ℝ) (h : n > 0) : 
  n = (4/5 * n) * (5/6 * n) → n = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_self_l3273_327393


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l3273_327344

theorem square_of_binomial_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) → a = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l3273_327344


namespace NUMINAMATH_CALUDE_power_of_product_cube_l3273_327349

theorem power_of_product_cube (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_cube_l3273_327349


namespace NUMINAMATH_CALUDE_crayon_selection_ways_l3273_327313

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of crayons in the box. -/
def total_crayons : ℕ := 15

/-- The number of crayons Karl selects. -/
def selected_crayons : ℕ := 5

/-- Theorem stating that selecting 5 crayons from 15 crayons can be done in 3003 ways. -/
theorem crayon_selection_ways : binomial total_crayons selected_crayons = 3003 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_ways_l3273_327313


namespace NUMINAMATH_CALUDE_algebraic_expression_evaluation_l3273_327395

theorem algebraic_expression_evaluation :
  ∀ (a b : ℝ), 
  (2 * a * (-1)^3 - 3 * b * (-1) + 8 = 18) → 
  (9 * b - 6 * a + 2 = 32) := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_evaluation_l3273_327395


namespace NUMINAMATH_CALUDE_break_even_price_correct_l3273_327368

/-- The price per kilogram to sell fruits without loss or profit -/
def break_even_price : ℝ := 2.6

/-- The price per jin that results in a loss -/
def loss_price : ℝ := 1.2

/-- The price per jin that results in a profit -/
def profit_price : ℝ := 1.5

/-- The amount of loss when selling at loss_price -/
def loss_amount : ℝ := 4

/-- The amount of profit when selling at profit_price -/
def profit_amount : ℝ := 8

/-- Conversion factor from jin to kilogram -/
def jin_to_kg : ℝ := 0.5

theorem break_even_price_correct :
  ∃ (weight : ℝ),
    weight * (break_even_price * jin_to_kg) = weight * loss_price + loss_amount ∧
    weight * (break_even_price * jin_to_kg) = weight * profit_price - profit_amount :=
by sorry

end NUMINAMATH_CALUDE_break_even_price_correct_l3273_327368


namespace NUMINAMATH_CALUDE_batsman_average_l3273_327324

/-- Proves that given a batsman's average of 45 runs in 25 matches and an overall average of 38.4375 in 32 matches, the average runs scored in the last 7 matches is 15. -/
theorem batsman_average (first_25_avg : ℝ) (total_32_avg : ℝ) (first_25_matches : ℕ) (total_matches : ℕ) :
  first_25_avg = 45 →
  total_32_avg = 38.4375 →
  first_25_matches = 25 →
  total_matches = 32 →
  let last_7_matches := total_matches - first_25_matches
  let total_runs := total_32_avg * total_matches
  let first_25_runs := first_25_avg * first_25_matches
  let last_7_runs := total_runs - first_25_runs
  last_7_runs / last_7_matches = 15 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l3273_327324


namespace NUMINAMATH_CALUDE_cookie_difference_l3273_327300

/-- The number of chocolate chip cookies Helen baked yesterday -/
def helen_choc_yesterday : ℕ := 19

/-- The number of raisin cookies Helen baked this morning -/
def helen_raisin_today : ℕ := 231

/-- The number of chocolate chip cookies Helen baked this morning -/
def helen_choc_today : ℕ := 237

/-- The number of oatmeal cookies Helen baked this morning -/
def helen_oatmeal_today : ℕ := 107

/-- The number of chocolate chip cookies Giselle baked -/
def giselle_choc : ℕ := 156

/-- The number of raisin cookies Giselle baked -/
def giselle_raisin : ℕ := 89

/-- The number of chocolate chip cookies Timmy baked -/
def timmy_choc : ℕ := 135

/-- The number of oatmeal cookies Timmy baked -/
def timmy_oatmeal : ℕ := 246

theorem cookie_difference : 
  (helen_choc_yesterday + helen_choc_today + giselle_choc + timmy_choc) - 
  (helen_raisin_today + giselle_raisin) = 227 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l3273_327300


namespace NUMINAMATH_CALUDE_complement_A_union_B_eq_R_A_union_B_eq_A_l3273_327322

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ 3 - 2*a}

-- Part 1
theorem complement_A_union_B_eq_R (a : ℝ) :
  (Set.univ \ A) ∪ B a = Set.univ ↔ a ≤ 0 := by sorry

-- Part 2
theorem A_union_B_eq_A (a : ℝ) :
  A ∪ B a = A ↔ a ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_eq_R_A_union_B_eq_A_l3273_327322


namespace NUMINAMATH_CALUDE_product_expansion_l3273_327331

theorem product_expansion {R : Type*} [CommRing R] (x : R) :
  (3 * x + 4) * (2 * x^2 + x + 6) = 6 * x^3 + 11 * x^2 + 22 * x + 24 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l3273_327331


namespace NUMINAMATH_CALUDE_find_a_minus_b_l3273_327320

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -3 * x + 5
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem find_a_minus_b (a b : ℝ) : 
  (∀ x, h a b x = x - 7) → a - b = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_a_minus_b_l3273_327320


namespace NUMINAMATH_CALUDE_initial_bananas_count_l3273_327312

/-- The number of bananas Raj has eaten -/
def bananas_eaten : ℕ := 70

/-- The number of bananas left on the tree after Raj cut some -/
def bananas_left_on_tree : ℕ := 100

/-- The number of bananas in Raj's basket -/
def bananas_in_basket : ℕ := 2 * bananas_eaten

/-- The total number of bananas Raj cut from the tree -/
def bananas_cut : ℕ := bananas_eaten + bananas_in_basket

/-- The initial number of bananas on the tree -/
def initial_bananas : ℕ := bananas_cut + bananas_left_on_tree

theorem initial_bananas_count : initial_bananas = 310 := by
  sorry

end NUMINAMATH_CALUDE_initial_bananas_count_l3273_327312


namespace NUMINAMATH_CALUDE_shopping_time_calculation_l3273_327338

def total_trip_time : ℕ := 90
def wait_for_cart : ℕ := 3
def wait_for_employee : ℕ := 13
def wait_for_stocker : ℕ := 14
def wait_in_line : ℕ := 18

theorem shopping_time_calculation :
  total_trip_time - (wait_for_cart + wait_for_employee + wait_for_stocker + wait_in_line) = 42 := by
  sorry

end NUMINAMATH_CALUDE_shopping_time_calculation_l3273_327338


namespace NUMINAMATH_CALUDE_complement_M_intersect_P_l3273_327391

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | |x - 1/2| ≤ 5/2}
def P : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- State the theorem
theorem complement_M_intersect_P :
  (U \ M) ∩ P = {x | 3 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_P_l3273_327391


namespace NUMINAMATH_CALUDE_robot_types_count_l3273_327342

theorem robot_types_count (shapes : ℕ) (colors : ℕ) (h1 : shapes = 3) (h2 : colors = 4) :
  shapes * colors = 12 := by
  sorry

end NUMINAMATH_CALUDE_robot_types_count_l3273_327342


namespace NUMINAMATH_CALUDE_area_range_of_special_triangle_l3273_327377

/-- Given an acute triangle ABC where angles A, B, C form an arithmetic sequence
    and the side opposite to angle B has length √3, prove that the area S of the triangle
    satisfies √3/2 < S ≤ 3√3/4. -/
theorem area_range_of_special_triangle (A B C : Real) (a b c : Real) (S : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- ABC is an acute triangle
  A + B + C = π ∧  -- sum of angles in a triangle
  2 * B = A + C ∧  -- A, B, C form an arithmetic sequence
  b = Real.sqrt 3 ∧  -- side opposite to B has length √3
  S = (1 / 2) * a * c * Real.sin B ∧  -- area formula
  a * Real.sin B = b * Real.sin A ∧  -- sine law
  c * Real.sin B = b * Real.sin C  -- sine law
  →
  Real.sqrt 3 / 2 < S ∧ S ≤ 3 * Real.sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_area_range_of_special_triangle_l3273_327377


namespace NUMINAMATH_CALUDE_no_integer_solution_quadratic_l3273_327379

theorem no_integer_solution_quadratic (x : ℤ) : x^2 + 3 ≥ 2*x := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_quadratic_l3273_327379


namespace NUMINAMATH_CALUDE_inequality_region_is_triangle_l3273_327381

/-- The region described by a system of inequalities -/
def InequalityRegion (x y : ℝ) : Prop :=
  x + y - 1 ≤ 0 ∧ -x + y - 1 ≤ 0 ∧ y ≥ -1

/-- The triangle with vertices (0, 1), (2, -1), and (-2, -1) -/
def Triangle (x y : ℝ) : Prop :=
  (x = 0 ∧ y = 1) ∨ (x = 2 ∧ y = -1) ∨ (x = -2 ∧ y = -1) ∨
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    ((x = 2*t - 2 ∧ y = -1) ∨
     (x = 2*t ∧ y = -t) ∨
     (x = -2*t ∧ y = t)))

theorem inequality_region_is_triangle :
  ∀ x y : ℝ, InequalityRegion x y ↔ Triangle x y :=
by sorry

end NUMINAMATH_CALUDE_inequality_region_is_triangle_l3273_327381


namespace NUMINAMATH_CALUDE_factorization_equality_l3273_327357

theorem factorization_equality (a b : ℝ) : 4*a - a*b^2 = a*(2+b)*(2-b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3273_327357


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l3273_327339

theorem range_of_a_minus_b (a b : ℝ) (ha : -2 < a ∧ a < 3) (hb : 1 < b ∧ b < 2) :
  ∀ x, (∃ (a' b' : ℝ), -2 < a' ∧ a' < 3 ∧ 1 < b' ∧ b' < 2 ∧ x = a' - b') ↔ -4 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l3273_327339


namespace NUMINAMATH_CALUDE_function_range_l3273_327345

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Define the domain
def domain : Set ℝ := { x | -3 ≤ x ∧ x ≤ 2 }

-- State the theorem
theorem function_range :
  { y | ∃ x ∈ domain, f x = y } = { y | 1 ≤ y ∧ y ≤ 17 } := by sorry

end NUMINAMATH_CALUDE_function_range_l3273_327345


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l3273_327306

/-- 
Proves that the profit percent is 32% when selling an article at a certain price, 
given that selling at 2/3 of that price results in a 12% loss.
-/
theorem profit_percent_calculation 
  (P : ℝ) -- The selling price
  (C : ℝ) -- The cost price
  (h : (2/3) * P = 0.88 * C) -- Condition: selling at 2/3 of P results in a 12% loss
  : (P - C) / C * 100 = 32 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l3273_327306


namespace NUMINAMATH_CALUDE_quadratic_function_k_value_l3273_327387

theorem quadratic_function_k_value (a b c k : ℤ) :
  let g := λ (x : ℤ) => a * x^2 + b * x + c
  (g 2 = 0) →
  (110 < g 9) →
  (g 9 < 120) →
  (130 < g 10) →
  (g 10 < 140) →
  (6000 * k < g 100) →
  (g 100 < 6000 * (k + 1)) →
  k = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_k_value_l3273_327387


namespace NUMINAMATH_CALUDE_f_of_tan_squared_plus_one_l3273_327329

noncomputable def f (x : ℝ) : ℝ :=
  1 / ((x / (x - 1)) - 1)

theorem f_of_tan_squared_plus_one (t : ℝ) (h : 0 ≤ t ∧ t ≤ π/2) :
  f (Real.tan t ^ 2 + 1) = (Real.sin (2 * t)) ^ 2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_f_of_tan_squared_plus_one_l3273_327329


namespace NUMINAMATH_CALUDE_some_fast_animals_are_pets_l3273_327334

-- Define our universe
variable (U : Type)

-- Define our predicates
variable (Wolf FastAnimal Pet : U → Prop)

-- State the theorem
theorem some_fast_animals_are_pets
  (h1 : ∀ x, Wolf x → FastAnimal x)
  (h2 : ∃ x, Pet x ∧ Wolf x) :
  ∃ x, FastAnimal x ∧ Pet x :=
sorry

end NUMINAMATH_CALUDE_some_fast_animals_are_pets_l3273_327334


namespace NUMINAMATH_CALUDE_function_properties_l3273_327347

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos (x / 2))^2 - Real.sqrt 3 * Real.sin x

theorem function_properties :
  ∃ (a : ℝ),
    (π / 2 < a ∧ a < π) ∧
    f (a - π / 3) = 1 / 3 →
    (∀ (x : ℝ), f (x + 2 * π) = f x) ∧
    (∀ (y : ℝ), -1 ≤ y ∧ y ≤ 3 ↔ ∃ (x : ℝ), f x = y) ∧
    (Real.cos (2 * a)) / (1 + Real.cos (2 * a) - Real.sin (2 * a)) = (1 - 2 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3273_327347


namespace NUMINAMATH_CALUDE_square_circle_circumradius_infinite_l3273_327372

/-- The radius of the circumcircle of a square with side length 1 and an inscribed circle 
    with diameter equal to the square's diagonal is infinite. -/
theorem square_circle_circumradius_infinite :
  let square : Set (ℝ × ℝ) := {p | p.1 ∈ [0, 1] ∧ p.2 ∈ [0, 1]}
  let circle : Set (ℝ × ℝ) := {p | (p.1 - 0.5)^2 + p.2^2 ≤ 0.5^2}
  let figure : Set (ℝ × ℝ) := square ∪ circle
  ¬ ∃ (r : ℝ), r > 0 ∧ ∃ (c : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ figure → (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 :=
by sorry


end NUMINAMATH_CALUDE_square_circle_circumradius_infinite_l3273_327372


namespace NUMINAMATH_CALUDE_min_sum_values_l3273_327359

theorem min_sum_values (a b x y : ℝ) : 
  a > 0 → b > 0 → x > 0 → y > 0 →
  a + b = 10 →
  a / x + b / y = 1 →
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → a / x' + b / y' = 1 → x' + y' ≥ 16) →
  x + y = 16 →
  ((a = 1 ∧ b = 9) ∨ (a = 9 ∧ b = 1)) := by
sorry

end NUMINAMATH_CALUDE_min_sum_values_l3273_327359


namespace NUMINAMATH_CALUDE_translation_of_quadratic_l3273_327308

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := (x - 2)^2 - 4

/-- The translated quadratic function -/
def g (x : ℝ) : ℝ := (x - 1)^2 - 2

/-- Theorem stating that g is the result of translating f one unit left and two units up -/
theorem translation_of_quadratic :
  ∀ x : ℝ, g x = f (x - 1) + 2 := by sorry

end NUMINAMATH_CALUDE_translation_of_quadratic_l3273_327308


namespace NUMINAMATH_CALUDE_train_length_l3273_327305

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (time_seconds : ℝ) : 
  speed_kmph = 360 → time_seconds = 5 → speed_kmph * (5 / 18) * time_seconds = 500 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3273_327305


namespace NUMINAMATH_CALUDE_correct_equation_transformation_l3273_327367

theorem correct_equation_transformation (x : ℝ) : x - 1 = 4 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_transformation_l3273_327367


namespace NUMINAMATH_CALUDE_min_value_line_circle_l3273_327316

/-- Given a line ax + by + c - 1 = 0 that passes through the center of the circle x^2 + y^2 - 2y - 5 = 0,
    prove that the minimum value of 4/b + 1/c is 9, where b > 0 and c > 0. -/
theorem min_value_line_circle (a b c : ℝ) (hb : b > 0) (hc : c > 0) :
  (∀ x y : ℝ, a * x + b * y + c - 1 = 0 → x^2 + y^2 - 2*y - 5 = 0) →
  (∃ x y : ℝ, a * x + b * y + c - 1 = 0 ∧ x^2 + y^2 - 2*y - 5 = 0) →
  (∀ b' c' : ℝ, b' > 0 → c' > 0 → 4 / b' + 1 / c' ≥ 9) ∧
  (∃ b' c' : ℝ, b' > 0 ∧ c' > 0 ∧ 4 / b' + 1 / c' = 9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_line_circle_l3273_327316


namespace NUMINAMATH_CALUDE_series_sum_l3273_327310

/-- The sum of the series Σ(3^(2^k) / (9^(2^k) - 1)) from k = 0 to infinity is 1/2 -/
theorem series_sum : 
  ∑' k, (3 ^ (2 ^ k) : ℝ) / ((9 : ℝ) ^ (2 ^ k) - 1) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_series_sum_l3273_327310


namespace NUMINAMATH_CALUDE_min_side_length_l3273_327362

theorem min_side_length (EF HG : ℝ) (EG HF : ℝ) (h1 : EF = 7) (h2 : EG = 15) (h3 : HG = 10) (h4 : HF = 25) :
  ∀ FG : ℝ, (FG > EG - EF ∧ FG > HF - HG) → FG ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_l3273_327362


namespace NUMINAMATH_CALUDE_pizza_combinations_l3273_327330

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  n + Nat.choose n 2 + Nat.choose n 3 = 92 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l3273_327330


namespace NUMINAMATH_CALUDE_infinite_fraction_value_l3273_327325

theorem infinite_fraction_value : 
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = (3 + Real.sqrt 69) / 2 := by
sorry

end NUMINAMATH_CALUDE_infinite_fraction_value_l3273_327325


namespace NUMINAMATH_CALUDE_repeating_decimal_eq_fraction_l3273_327394

/-- The repeating decimal 0.4̅5̅6̅ as a rational number -/
def repeating_decimal : ℚ := 0.4 + (56 : ℚ) / 990

/-- The fraction 226/495 -/
def fraction : ℚ := 226 / 495

/-- Theorem stating that the repeating decimal 0.4̅5̅6̅ is equal to the fraction 226/495 -/
theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_eq_fraction_l3273_327394


namespace NUMINAMATH_CALUDE_f_neg_three_eq_neg_two_l3273_327340

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the property of f being an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the symmetry condition between f and g
def symmetric_about_y_eq_x_plus_1 (f g : ℝ → ℝ) : Prop :=
  ∀ x y, g x = y ↔ f (y - 1) = x - 1

-- State the theorem
theorem f_neg_three_eq_neg_two
  (h_odd : is_odd f)
  (h_sym : symmetric_about_y_eq_x_plus_1 f g)
  (h_g : g 1 = 4) :
  f (-3) = -2 :=
sorry

end NUMINAMATH_CALUDE_f_neg_three_eq_neg_two_l3273_327340


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_l3273_327380

theorem mean_equality_implies_z (z : ℚ) : 
  (4 + 16 + 20) / 3 = (2 * 4 + z) / 2 → z = 56 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_l3273_327380


namespace NUMINAMATH_CALUDE_avery_egg_cartons_l3273_327314

/-- Given the number of chickens, eggs per chicken, and number of filled cartons,
    calculate the number of eggs per carton. -/
def eggs_per_carton (num_chickens : ℕ) (eggs_per_chicken : ℕ) (num_cartons : ℕ) : ℕ :=
  (num_chickens * eggs_per_chicken) / num_cartons

/-- Prove that with 20 chickens laying 6 eggs each, filling 10 cartons results in 12 eggs per carton. -/
theorem avery_egg_cartons : eggs_per_carton 20 6 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_avery_egg_cartons_l3273_327314


namespace NUMINAMATH_CALUDE_library_visitors_average_l3273_327354

def average_visitors (total_visitors : ℕ) (days : ℕ) : ℚ :=
  (total_visitors : ℚ) / (days : ℚ)

theorem library_visitors_average (
  sunday_visitors : ℕ)
  (weekday_visitors : ℕ)
  (weekend_visitors : ℕ)
  (special_event_visitors : ℕ)
  (h1 : sunday_visitors = 660)
  (h2 : weekday_visitors = 280)
  (h3 : weekend_visitors = 350)
  (h4 : special_event_visitors = 120)
  : average_visitors (
    4 * sunday_visitors +
    17 * weekday_visitors +
    8 * weekend_visitors +
    special_event_visitors
  ) 30 = 344 := by
  sorry

#eval average_visitors (
  4 * 660 +
  17 * 280 +
  8 * 350 +
  120
) 30

end NUMINAMATH_CALUDE_library_visitors_average_l3273_327354


namespace NUMINAMATH_CALUDE_reinforced_grid_30x8_toothpicks_l3273_327350

/-- Calculates the total number of toothpicks in a reinforced rectangular grid. -/
def total_toothpicks (height width : ℕ) : ℕ :=
  let internal_horizontal := (height + 1) * width
  let internal_vertical := (width + 1) * height
  let external_horizontal := 2 * width
  let external_vertical := 2 * (height + 2)
  internal_horizontal + internal_vertical + external_horizontal + external_vertical

/-- Theorem stating that a reinforced rectangular grid of 30x8 toothpicks uses 598 toothpicks. -/
theorem reinforced_grid_30x8_toothpicks :
  total_toothpicks 30 8 = 598 := by
  sorry

#eval total_toothpicks 30 8  -- Should output 598

end NUMINAMATH_CALUDE_reinforced_grid_30x8_toothpicks_l3273_327350


namespace NUMINAMATH_CALUDE_total_boxes_is_6200_l3273_327307

/-- The number of boxes in Warehouse D -/
def warehouse_d : ℕ := 800

/-- The number of boxes in Warehouse C -/
def warehouse_c : ℕ := warehouse_d - 200

/-- The number of boxes in Warehouse B -/
def warehouse_b : ℕ := 2 * warehouse_c

/-- The number of boxes in Warehouse A -/
def warehouse_a : ℕ := 3 * warehouse_b

/-- The total number of boxes in all four warehouses -/
def total_boxes : ℕ := warehouse_a + warehouse_b + warehouse_c + warehouse_d

theorem total_boxes_is_6200 : total_boxes = 6200 := by
  sorry

end NUMINAMATH_CALUDE_total_boxes_is_6200_l3273_327307


namespace NUMINAMATH_CALUDE_sum_of_four_integers_l3273_327304

theorem sum_of_four_integers (a b c d : ℤ) :
  (a + b + c) / 3 + d = 8 ∧
  (a + b + d) / 3 + c = 12 ∧
  (a + c + d) / 3 + b = 32 / 3 ∧
  (b + c + d) / 3 + a = 28 / 3 →
  a + b + c + d = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_integers_l3273_327304


namespace NUMINAMATH_CALUDE_age_difference_l3273_327383

/-- Represents the ages of four people: Patrick, Michael, Monica, and Nathan. -/
structure Ages where
  patrick : ℝ
  michael : ℝ
  monica : ℝ
  nathan : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.patrick / ages.michael = 3 / 5 ∧
  ages.michael / ages.monica = 3 / 5 ∧
  ages.monica / ages.nathan = 4 / 7 ∧
  ages.patrick + ages.michael + ages.monica + ages.nathan = 142

/-- The theorem stating the difference between Patrick's and Nathan's ages -/
theorem age_difference (ages : Ages) (h : satisfies_conditions ages) :
  ∃ ε > 0, |ages.patrick - ages.nathan - 1.46| < ε :=
sorry

end NUMINAMATH_CALUDE_age_difference_l3273_327383


namespace NUMINAMATH_CALUDE_no_simultaneous_perfect_squares_l3273_327319

theorem no_simultaneous_perfect_squares (a b : ℕ+) : 
  ¬(∃ (x y : ℕ), (a.val^2 + 4*b.val = x^2) ∧ (b.val^2 + 4*a.val = y^2)) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_perfect_squares_l3273_327319


namespace NUMINAMATH_CALUDE_cindys_calculation_l3273_327315

theorem cindys_calculation (x : ℝ) : (x - 10) / 5 = 50 → (x - 5) / 10 = 25.5 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l3273_327315


namespace NUMINAMATH_CALUDE_fraction_equality_l3273_327309

theorem fraction_equality (a b : ℝ) (h : b / a = 1 / 2) : (a + b) / a = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3273_327309


namespace NUMINAMATH_CALUDE_total_sales_over_three_days_l3273_327348

def friday_sales : ℕ := 30

def saturday_sales : ℕ := 2 * friday_sales

def sunday_sales : ℕ := saturday_sales - 15

theorem total_sales_over_three_days : 
  friday_sales + saturday_sales + sunday_sales = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_over_three_days_l3273_327348


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l3273_327390

/-- The lateral area of a cylinder with diameter and height both equal to 4 is 16π. -/
theorem cylinder_lateral_area : 
  ∀ (d h : ℝ), d = 4 → h = 4 → 2 * π * (d / 2) * h = 16 * π :=
by
  sorry


end NUMINAMATH_CALUDE_cylinder_lateral_area_l3273_327390


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l3273_327327

theorem quadratic_equation_condition (m : ℝ) : 
  (∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, (m - 3) * x^2 + m * x + (-2 * m - 2) = a * x^2 + b * x + c) ↔ 
  m = -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l3273_327327


namespace NUMINAMATH_CALUDE_unique_triple_solution_l3273_327352

theorem unique_triple_solution : 
  ∃! (x y z : ℝ), x + y = 4 ∧ x * y - z^2 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l3273_327352


namespace NUMINAMATH_CALUDE_last_three_digits_of_8_to_108_l3273_327388

theorem last_three_digits_of_8_to_108 : 8^108 ≡ 38 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_8_to_108_l3273_327388


namespace NUMINAMATH_CALUDE_problem_solution_l3273_327374

theorem problem_solution (x y : ℝ) 
  (h1 : x = 103) 
  (h2 : x^3*y - 4*x^2*y + 4*x*y = 515400) : 
  y = 1/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3273_327374


namespace NUMINAMATH_CALUDE_probability_three_teachers_same_gate_l3273_327356

-- Define the number of teachers and gates
def num_teachers : ℕ := 12
def num_gates : ℕ := 3
def teachers_per_gate : ℕ := 4

-- Define the probability function
noncomputable def probability_same_gate : ℚ :=
  3 / 55

-- Theorem statement
theorem probability_three_teachers_same_gate :
  probability_same_gate = 3 / 55 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_three_teachers_same_gate_l3273_327356


namespace NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l3273_327323

theorem factorial_of_factorial_divided_by_factorial :
  (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = 25852016738884976640000 := by
  sorry

end NUMINAMATH_CALUDE_factorial_of_factorial_divided_by_factorial_l3273_327323


namespace NUMINAMATH_CALUDE_probability_of_standard_lamp_l3273_327351

/-- The probability of a lamp being from factory 1 -/
def p_factory1 : ℝ := 0.45

/-- The probability of a lamp being from factory 2 -/
def p_factory2 : ℝ := 0.40

/-- The probability of a lamp being from factory 3 -/
def p_factory3 : ℝ := 0.15

/-- The probability of a lamp being standard given it's from factory 1 -/
def p_standard_given_factory1 : ℝ := 0.70

/-- The probability of a lamp being standard given it's from factory 2 -/
def p_standard_given_factory2 : ℝ := 0.80

/-- The probability of a lamp being standard given it's from factory 3 -/
def p_standard_given_factory3 : ℝ := 0.81

/-- The theorem stating the probability of purchasing a standard lamp -/
theorem probability_of_standard_lamp :
  p_factory1 * p_standard_given_factory1 +
  p_factory2 * p_standard_given_factory2 +
  p_factory3 * p_standard_given_factory3 = 0.7565 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_standard_lamp_l3273_327351


namespace NUMINAMATH_CALUDE_balance_is_132_l3273_327318

/-- Calculates the balance of a bank account after two years with given parameters. -/
def balance_after_two_years (initial_deposit : ℝ) (interest_rate : ℝ) (annual_deposit : ℝ) : ℝ :=
  let balance_year_one := initial_deposit * (1 + interest_rate) + annual_deposit
  balance_year_one * (1 + interest_rate) + annual_deposit

/-- Theorem stating that given specific parameters, the balance after two years is $132. -/
theorem balance_is_132 :
  balance_after_two_years 100 0.1 10 = 132 := by
  sorry

#eval balance_after_two_years 100 0.1 10

end NUMINAMATH_CALUDE_balance_is_132_l3273_327318


namespace NUMINAMATH_CALUDE_smallest_z_l3273_327311

theorem smallest_z (x y z : ℤ) : 
  x < y → y < z → 
  (2 * y = x + z) →  -- arithmetic progression
  (z * z = x * y) →  -- geometric progression
  (∀ w : ℤ, (∃ a b c : ℤ, a < b ∧ b < w ∧ 2 * b = a + w ∧ w * w = a * b) → w ≥ z) →
  z = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_z_l3273_327311


namespace NUMINAMATH_CALUDE_prism_volume_l3273_327386

/-- The volume of a right rectangular prism given its face areas -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 36)
  (h2 : a * c = 72)
  (h3 : b * c = 48) :
  a * b * c = 352.8 := by
sorry

end NUMINAMATH_CALUDE_prism_volume_l3273_327386


namespace NUMINAMATH_CALUDE_f_of_2_eq_0_l3273_327303

-- Define the function f
def f : ℝ → ℝ := fun x => (x - 1)^2 - 1

-- State the theorem
theorem f_of_2_eq_0 : f 2 = 0 := by sorry

end NUMINAMATH_CALUDE_f_of_2_eq_0_l3273_327303


namespace NUMINAMATH_CALUDE_lillys_daily_savings_l3273_327378

/-- Proves that the daily savings amount is $2 given the conditions of Lilly's flower-buying plan for Maria's birthday. -/
theorem lillys_daily_savings 
  (saving_period : ℕ) 
  (flower_cost : ℚ) 
  (total_flowers : ℕ) 
  (h1 : saving_period = 22)
  (h2 : flower_cost = 4)
  (h3 : total_flowers = 11) : 
  (total_flowers : ℚ) * flower_cost / saving_period = 2 := by
  sorry

end NUMINAMATH_CALUDE_lillys_daily_savings_l3273_327378


namespace NUMINAMATH_CALUDE_triangle_side_length_l3273_327361

/-- In a triangle ABC, given side lengths a and c, and angle A, prove that side length b has a specific value. -/
theorem triangle_side_length (a c b : ℝ) (A : ℝ) : 
  a = 3 → c = Real.sqrt 3 → A = π / 3 → 
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A → 
  b = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3273_327361


namespace NUMINAMATH_CALUDE_remainder_2519_div_3_l3273_327365

theorem remainder_2519_div_3 : 2519 % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2519_div_3_l3273_327365


namespace NUMINAMATH_CALUDE_median_length_l3273_327375

/-- A tetrahedron with vertex D at the origin and right angles at D -/
structure RightTetrahedron where
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ
  right_angles : sorry
  DA_length : ‖A‖ = 1
  DB_length : ‖B‖ = 2
  DC_length : ‖C‖ = 3

/-- The median of a tetrahedron from vertex D -/
def tetrahedron_median (t : RightTetrahedron) : ℝ := sorry

/-- Theorem: The length of the median from D in the specified tetrahedron is √6/3 -/
theorem median_length (t : RightTetrahedron) : 
  tetrahedron_median t = Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_median_length_l3273_327375


namespace NUMINAMATH_CALUDE_steak_eaten_l3273_327337

theorem steak_eaten (original_weight : ℝ) (burned_fraction : ℝ) (eaten_fraction : ℝ) : 
  original_weight = 30 ∧ 
  burned_fraction = 0.5 ∧ 
  eaten_fraction = 0.8 → 
  original_weight * (1 - burned_fraction) * eaten_fraction = 12 := by
  sorry

end NUMINAMATH_CALUDE_steak_eaten_l3273_327337


namespace NUMINAMATH_CALUDE_living_room_walls_count_l3273_327364

/-- The number of walls in Eric's living room -/
def living_room_walls : ℕ := 7

/-- The time Eric spent removing wallpaper from one wall in the dining room (in hours) -/
def time_per_wall : ℕ := 2

/-- The total time it will take Eric to remove wallpaper from the living room (in hours) -/
def total_time : ℕ := 14

/-- Theorem stating that the number of walls in Eric's living room is 7 -/
theorem living_room_walls_count :
  living_room_walls = total_time / time_per_wall :=
by sorry

end NUMINAMATH_CALUDE_living_room_walls_count_l3273_327364


namespace NUMINAMATH_CALUDE_stretch_circle_to_ellipse_l3273_327385

/-- Given a circle A and a stretch transformation, prove the equation of the resulting curve C -/
theorem stretch_circle_to_ellipse (x y x' y' : ℝ) :
  (x^2 + y^2 = 1) →  -- Circle A equation
  (x' = 2*x) →       -- Stretch transformation for x
  (y' = 3*y) →       -- Stretch transformation for y
  (x'^2 / 4 + y'^2 / 9 = 1) -- Resulting curve C equation
:= by sorry

end NUMINAMATH_CALUDE_stretch_circle_to_ellipse_l3273_327385


namespace NUMINAMATH_CALUDE_eve_can_discover_secret_number_l3273_327317

theorem eve_can_discover_secret_number :
  ∀ x : ℕ, ∃ (k : ℕ) (n : Fin k → ℕ),
    ∀ y : ℕ, (∀ i : Fin k, Prime (x + n i) ↔ Prime (y + n i)) → x = y :=
sorry

end NUMINAMATH_CALUDE_eve_can_discover_secret_number_l3273_327317


namespace NUMINAMATH_CALUDE_quadratic_equations_properties_l3273_327333

theorem quadratic_equations_properties (b c : ℤ) 
  (x₁ x₂ x₁' x₂' : ℤ) :
  (x₁^2 + b*x₁ + c = 0) →
  (x₂^2 + b*x₂ + c = 0) →
  (x₁'^2 + c*x₁' + b = 0) →
  (x₂'^2 + c*x₂' + b = 0) →
  (x₁ * x₂ > 0) →
  (x₁' * x₂' > 0) →
  (
    (x₁ < 0 ∧ x₂ < 0 ∧ x₁' < 0 ∧ x₂' < 0) ∧
    (b - 1 ≤ c ∧ c ≤ b + 1) ∧
    ((b = 4 ∧ c = 4) ∨ (b = 5 ∧ c = 6) ∨ (b = 6 ∧ c = 5))
  ) := by sorry

end NUMINAMATH_CALUDE_quadratic_equations_properties_l3273_327333


namespace NUMINAMATH_CALUDE_antimatter_prescription_fulfillment_l3273_327398

theorem antimatter_prescription_fulfillment :
  ∃ (x y z : ℕ), x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧
  (11 : ℝ) * x + 1.1 * y + 0.11 * z = 20.13 := by
  sorry

end NUMINAMATH_CALUDE_antimatter_prescription_fulfillment_l3273_327398


namespace NUMINAMATH_CALUDE_quadratic_sum_l3273_327363

/-- Given a quadratic expression 4x^2 - 8x + 5, when expressed in the form a(x - h)^2 + k,
    the sum a + h + k equals 6 -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a h k : ℝ), (4 * x^2 - 8 * x + 5 = a * (x - h)^2 + k) ∧ (a + h + k = 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3273_327363


namespace NUMINAMATH_CALUDE_binomial_square_condition_l3273_327321

theorem binomial_square_condition (b : ℚ) : 
  (∃ (p q : ℚ), ∀ x, b * x^2 + 20 * x + 9 = (p * x + q)^2) → b = 100 / 9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_condition_l3273_327321


namespace NUMINAMATH_CALUDE_percentage_problem_l3273_327360

theorem percentage_problem (P : ℝ) : P = 20 → (P / 100) * 680 = 0.4 * 140 + 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3273_327360


namespace NUMINAMATH_CALUDE_person_A_silver_sheets_l3273_327343

-- Define the exchange rates
def red_to_gold_rate : ℚ := 5 / 2
def gold_to_red_and_silver_rate : ℚ := 1

-- Define the initial number of sheets
def initial_red_sheets : ℕ := 3
def initial_gold_sheets : ℕ := 3

-- Define the function to calculate the total silver sheets
def total_silver_sheets : ℕ :=
  let gold_to_silver := initial_gold_sheets
  let red_to_silver := (initial_red_sheets + initial_gold_sheets) / 3 * 2
  gold_to_silver + red_to_silver

-- Theorem statement
theorem person_A_silver_sheets :
  total_silver_sheets = 7 :=
sorry

end NUMINAMATH_CALUDE_person_A_silver_sheets_l3273_327343


namespace NUMINAMATH_CALUDE_circle_equation_l3273_327335

/-- Given a circle with radius 5 and a line l: x + 2y - 3 = 0 tangent to the circle at point P(1,1),
    prove that the equations of the circle are:
    (x-1-√5)² + (y-1-2√5)² = 25 and (x-1+√5)² + (y-1+2√5)² = 25 -/
theorem circle_equation (x y : ℝ) :
  let r : ℝ := 5
  let l : ℝ → ℝ → ℝ := fun x y ↦ x + 2*y - 3
  let P : ℝ × ℝ := (1, 1)
  (∃ (center : ℝ × ℝ), (center.1 - P.1)^2 + (center.2 - P.2)^2 = r^2 ∧
    l P.1 P.2 = 0 ∧
    (∀ (t : ℝ), t ≠ 0 → l (P.1 + t) (P.2 + t * ((center.2 - P.2) / (center.1 - P.1))) ≠ 0)) →
  ((x - (1 - Real.sqrt 5))^2 + (y - (1 - 2 * Real.sqrt 5))^2 = 25) ∨
  ((x - (1 + Real.sqrt 5))^2 + (y - (1 + 2 * Real.sqrt 5))^2 = 25) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3273_327335


namespace NUMINAMATH_CALUDE_infinite_slips_with_same_number_l3273_327326

-- Define a type for slip numbers
def SlipNumber : Type := ℕ

-- Define the set of all slips
def AllSlips : Set SlipNumber := Set.univ

-- Define the property that any infinite subset has at least two slips with the same number
def HasDuplicatesInInfiniteSubsets (S : Set SlipNumber) : Prop :=
  ∀ (T : Set SlipNumber), T ⊆ S → T.Infinite → ∃ (n : SlipNumber), (∃ (s t : SlipNumber), s ∈ T ∧ t ∈ T ∧ s ≠ t ∧ n = s ∧ n = t)

-- State the theorem
theorem infinite_slips_with_same_number :
  AllSlips.Infinite →
  HasDuplicatesInInfiniteSubsets AllSlips →
  ∃ (n : SlipNumber), {s : SlipNumber | s ∈ AllSlips ∧ n = s}.Infinite :=
by sorry

end NUMINAMATH_CALUDE_infinite_slips_with_same_number_l3273_327326


namespace NUMINAMATH_CALUDE_trapezoid_area_l3273_327389

theorem trapezoid_area (large_square_side : ℝ) (small_square_side : ℝ) :
  large_square_side = 4 →
  small_square_side = 1 →
  let large_square_area := large_square_side ^ 2
  let small_square_area := small_square_side ^ 2
  let total_trapezoid_area := large_square_area - small_square_area
  let num_trapezoids := 4
  (total_trapezoid_area / num_trapezoids : ℝ) = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3273_327389


namespace NUMINAMATH_CALUDE_film_festival_theorem_l3273_327353

theorem film_festival_theorem (n : ℕ) (m : ℕ) : 
  -- Total number of films
  n > 0 →
  -- Total number of viewers (2m, where m is the number of men/women)
  m > 0 →
  -- Each film is liked by exactly 8 viewers
  -- Each viewer likes the same number of films
  -- The total number of "likes" is 8n
  8 * n = 2 * m * (8 * n / (2 * m)) →
  -- At least 3/7 of the films are liked by at least two men
  ∃ (k : ℕ), k ≥ (3 * n + 6) / 7 ∧ 
    (∀ (i : ℕ), i < k → ∃ (male_viewers : ℕ), male_viewers ≥ 2 ∧ male_viewers ≤ 8) :=
by
  sorry

end NUMINAMATH_CALUDE_film_festival_theorem_l3273_327353


namespace NUMINAMATH_CALUDE_train_length_l3273_327376

/-- The length of a train given its speed and time to cross a bridge -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 205 →
  crossing_time = 30 →
  (train_speed * crossing_time) - bridge_length = 170 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3273_327376
