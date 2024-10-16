import Mathlib

namespace NUMINAMATH_CALUDE_convince_jury_l3579_357957

-- Define the types of people
inductive PersonType
| Knight
| Liar
| Normal

-- Define the properties of a person
structure Person where
  type : PersonType
  guilty : Bool

-- Define the statement made by the person
def statement (p : Person) : Prop :=
  p.guilty ∧ p.type = PersonType.Liar

-- Define what it means for a person to be consistent with their statement
def consistent (p : Person) : Prop :=
  (p.type = PersonType.Knight ∧ statement p) ∨
  (p.type = PersonType.Liar ∧ ¬statement p) ∨
  (p.type = PersonType.Normal ∧ statement p)

-- Theorem to prove
theorem convince_jury :
  ∃ (p : Person), consistent p ∧ ¬p.guilty ∧ p.type ≠ PersonType.Knight :=
sorry

end NUMINAMATH_CALUDE_convince_jury_l3579_357957


namespace NUMINAMATH_CALUDE_success_permutations_count_l3579_357951

/-- The number of distinct permutations of the multiset {S, S, S, U, C, C, E} -/
def successPermutations : ℕ :=
  Nat.factorial 7 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of SUCCESS is 420 -/
theorem success_permutations_count : successPermutations = 420 := by
  sorry

end NUMINAMATH_CALUDE_success_permutations_count_l3579_357951


namespace NUMINAMATH_CALUDE_power_factorial_inequality_l3579_357907

theorem power_factorial_inequality (n : ℕ) : 2^n * n.factorial < (n + 1)^n := by
  sorry

end NUMINAMATH_CALUDE_power_factorial_inequality_l3579_357907


namespace NUMINAMATH_CALUDE_sum_of_integers_l3579_357929

theorem sum_of_integers (x y : ℕ+) (h1 : x - y = 8) (h2 : x * y = 180) : x + y = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3579_357929


namespace NUMINAMATH_CALUDE_two_triangles_exist_l3579_357935

/-- A triangle with a given angle, height, and circumradius. -/
structure SpecialTriangle where
  /-- One of the angles of the triangle -/
  angle : ℝ
  /-- The height corresponding to one side of the triangle -/
  height : ℝ
  /-- The radius of the circumcircle -/
  circumradius : ℝ
  /-- The angle is positive and less than π -/
  angle_pos : 0 < angle
  angle_lt_pi : angle < π
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The circumradius is positive -/
  circumradius_pos : 0 < circumradius

/-- There exist two distinct triangles satisfying the given conditions -/
theorem two_triangles_exist (α m r : ℝ) 
  (h_α_pos : 0 < α) (h_α_lt_pi : α < π) 
  (h_m_pos : 0 < m) (h_r_pos : 0 < r) : 
  ∃ (t1 t2 : SpecialTriangle), t1 ≠ t2 ∧ 
    t1.angle = α ∧ t1.height = m ∧ t1.circumradius = r ∧
    t2.angle = α ∧ t2.height = m ∧ t2.circumradius = r := by
  sorry

end NUMINAMATH_CALUDE_two_triangles_exist_l3579_357935


namespace NUMINAMATH_CALUDE_abc_inequality_l3579_357922

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define a, b, and c
noncomputable def a : ℝ := log (3/4) (log 3 4)
noncomputable def b : ℝ := (3/4) ^ (1/2 : ℝ)
noncomputable def c : ℝ := (4/3) ^ (1/2 : ℝ)

-- Theorem statement
theorem abc_inequality : a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l3579_357922


namespace NUMINAMATH_CALUDE_pascal_triangle_row20_sum_l3579_357974

theorem pascal_triangle_row20_sum : Nat.choose 20 4 + Nat.choose 20 5 = 20349 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_row20_sum_l3579_357974


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_five_l3579_357910

theorem fraction_zero_implies_x_equals_five (x : ℝ) : 
  (x^2 - 25) / (x + 5) = 0 ∧ x + 5 ≠ 0 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_five_l3579_357910


namespace NUMINAMATH_CALUDE_rope_cutting_probability_l3579_357930

theorem rope_cutting_probability : 
  let rope_length : ℝ := 6
  let num_nodes : ℕ := 5
  let num_parts : ℕ := 6
  let min_segment_length : ℝ := 2

  let part_length : ℝ := rope_length / num_parts
  let favorable_cuts : ℕ := (num_nodes - 2)
  
  (favorable_cuts : ℝ) / num_nodes = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_rope_cutting_probability_l3579_357930


namespace NUMINAMATH_CALUDE_impossible_to_make_all_divisible_by_three_l3579_357928

/-- Represents the state of numbers on the vertices of a 2018-sided polygon -/
def PolygonState := Fin 2018 → ℤ

/-- The initial state of the polygon -/
def initial_state : PolygonState :=
  fun i => if i.val = 2017 then 1 else 0

/-- The sum of all numbers on the vertices -/
def vertex_sum (state : PolygonState) : ℤ :=
  (Finset.univ.sum fun i => state i)

/-- Represents a legal move on the polygon -/
inductive LegalMove
  | add_subtract (i j : Fin 2018) : LegalMove

/-- Apply a legal move to a given state -/
def apply_move (state : PolygonState) (move : LegalMove) : PolygonState :=
  match move with
  | LegalMove.add_subtract i j =>
      fun k => if k = i then state k + 1
               else if k = j then state k - 1
               else state k

/-- Predicate to check if all numbers are divisible by 3 -/
def all_divisible_by_three (state : PolygonState) : Prop :=
  ∀ i, state i % 3 = 0

theorem impossible_to_make_all_divisible_by_three :
  ¬∃ (moves : List LegalMove), 
    all_divisible_by_three (moves.foldl apply_move initial_state) :=
  sorry


end NUMINAMATH_CALUDE_impossible_to_make_all_divisible_by_three_l3579_357928


namespace NUMINAMATH_CALUDE_series_convergence_l3579_357901

/-- The infinite series ∑(k=1 to ∞) [k(k+1)/(2*3^k)] converges to 3/2 -/
theorem series_convergence : 
  ∑' k, (k * (k + 1) : ℝ) / (2 * 3^k) = 3/2 := by sorry

end NUMINAMATH_CALUDE_series_convergence_l3579_357901


namespace NUMINAMATH_CALUDE_original_car_cost_l3579_357931

/-- Proves that the original cost of a car is 42000 given the repair cost, selling price, and profit percentage. -/
theorem original_car_cost (repair_cost selling_price profit_percent : ℝ) : 
  repair_cost = 8000 →
  selling_price = 64900 →
  profit_percent = 29.8 →
  ∃ (original_cost : ℝ), 
    original_cost = 42000 ∧
    selling_price = (original_cost + repair_cost) * (1 + profit_percent / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_original_car_cost_l3579_357931


namespace NUMINAMATH_CALUDE_derivative_sum_positive_l3579_357927

open Real

noncomputable def f (a b x : ℝ) : ℝ := a * log x - b * x - 1 / x

theorem derivative_sum_positive (a : ℝ) (h_a : a > 0) (x₁ x₂ : ℝ) 
  (h_x₁ : x₁ > 0) (h_x₂ : x₂ > 0) (h_neq : x₁ ≠ x₂) :
  ∃ b : ℝ, f a b x₁ = f a b x₂ → 
    (deriv (f a b) x₁ + deriv (f a b) x₂ > 0) := by
  sorry

end NUMINAMATH_CALUDE_derivative_sum_positive_l3579_357927


namespace NUMINAMATH_CALUDE_inequality_condition_l3579_357946

theorem inequality_condition (A B C : ℝ) :
  (∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) ↔
  |A - B + C| ≤ 2 * Real.sqrt (A * C) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l3579_357946


namespace NUMINAMATH_CALUDE_fraction_calculation_l3579_357909

theorem fraction_calculation : (0.5^3) / (0.05^2) = 50 := by sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3579_357909


namespace NUMINAMATH_CALUDE_equation_positive_root_l3579_357932

theorem equation_positive_root (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (3 * x - 1) / (x + 1) - m / (x + 1) = 1) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_root_l3579_357932


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3579_357980

/-- The perimeter of an equilateral triangle whose area is numerically equal to twice its side length is 8√3. -/
theorem equilateral_triangle_perimeter : 
  ∀ s : ℝ, s > 0 → (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3579_357980


namespace NUMINAMATH_CALUDE_cone_radii_sum_l3579_357940

/-- Given a circle with radius 5 divided into three sectors with area ratios 1:2:3,
    used as lateral surfaces of three cones with base radii r₁, r₂, and r₃ respectively,
    prove that r₁ + r₂ + r₃ = 5. -/
theorem cone_radii_sum (r₁ r₂ r₃ : ℝ) : 
  (2 * π * r₁ = (1 / 6) * 2 * π * 5) → 
  (2 * π * r₂ = (2 / 6) * 2 * π * 5) → 
  (2 * π * r₃ = (3 / 6) * 2 * π * 5) → 
  r₁ + r₂ + r₃ = 5 := by
  sorry

end NUMINAMATH_CALUDE_cone_radii_sum_l3579_357940


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l3579_357917

-- Define the function f
def f (x : ℝ) : ℝ := (1 - 2*x)^10

-- State the theorem
theorem derivative_f_at_1 : 
  deriv f 1 = 20 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l3579_357917


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l3579_357906

/-- Given two sequences (aₖ) and (bₖ) satisfying certain conditions, 
    prove that aₖ > bₖ for all k between 2 and n-1 inclusive. -/
theorem arithmetic_geometric_inequality (n : ℕ) (a b : ℕ → ℝ) 
  (h_n : n ≥ 3)
  (h_a_arith : ∀ k l : ℕ, k < l → l ≤ n → a l - a k = (l - k) * (a 2 - a 1))
  (h_b_geom : ∀ k l : ℕ, k < l → l ≤ n → b l / b k = (b 2 / b 1) ^ (l - k))
  (h_a_pos : ∀ k : ℕ, k ≤ n → 0 < a k)
  (h_b_pos : ∀ k : ℕ, k ≤ n → 0 < b k)
  (h_a_inc : ∀ k : ℕ, k < n → a k < a (k + 1))
  (h_b_inc : ∀ k : ℕ, k < n → b k < b (k + 1))
  (h_eq_first : a 1 = b 1)
  (h_eq_last : a n = b n) :
  ∀ k : ℕ, 2 ≤ k → k < n → a k > b k :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l3579_357906


namespace NUMINAMATH_CALUDE_one_third_of_360_l3579_357943

theorem one_third_of_360 : (360 : ℝ) * (1 / 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_360_l3579_357943


namespace NUMINAMATH_CALUDE_no_two_roots_exist_l3579_357948

-- Define the equation as a function of x, y, and a
def equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*x = |x - a| - 1

-- Theorem statement
theorem no_two_roots_exist :
  ¬ ∃ (a : ℝ), ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≠ x₂ ∧ 
    equation x₁ y₁ a ∧ 
    equation x₂ y₂ a ∧ 
    (∀ (x y : ℝ), equation x y a → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
sorry

end NUMINAMATH_CALUDE_no_two_roots_exist_l3579_357948


namespace NUMINAMATH_CALUDE_correct_statements_are_1_and_3_l3579_357903

-- Define the proof methods
inductive ProofMethod
| Synthetic
| Analytic
| Contradiction

-- Define the properties of proof methods
def isCauseToEffect (m : ProofMethod) : Prop := m = ProofMethod.Synthetic
def isEffectToCause (m : ProofMethod) : Prop := m = ProofMethod.Analytic
def isDirectMethod (m : ProofMethod) : Prop := m = ProofMethod.Synthetic ∨ m = ProofMethod.Analytic
def isIndirectMethod (m : ProofMethod) : Prop := m = ProofMethod.Contradiction

-- Define the statements
def statement1 : Prop := isCauseToEffect ProofMethod.Synthetic
def statement2 : Prop := isIndirectMethod ProofMethod.Analytic
def statement3 : Prop := isEffectToCause ProofMethod.Analytic
def statement4 : Prop := isDirectMethod ProofMethod.Contradiction

-- Theorem to prove
theorem correct_statements_are_1_and_3 :
  (statement1 ∧ statement3) ∧ (¬statement2 ∧ ¬statement4) :=
sorry

end NUMINAMATH_CALUDE_correct_statements_are_1_and_3_l3579_357903


namespace NUMINAMATH_CALUDE_parabola_equation_l3579_357987

/-- Given a parabola y^2 = 2px where p > 0, if a point P(2, y_0) on the parabola
    has a distance of 4 from the directrix, then the equation of the parabola is y^2 = 8x -/
theorem parabola_equation (p : ℝ) (y_0 : ℝ) (h1 : p > 0) (h2 : y_0^2 = 2*p*2) 
  (h3 : p/2 + 2 = 4) : 
  ∀ x y : ℝ, y^2 = 2*p*x ↔ y^2 = 8*x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3579_357987


namespace NUMINAMATH_CALUDE_product_sum_difference_l3579_357976

theorem product_sum_difference (a b : ℤ) (h1 : b = 8) (h2 : b - a = 3) :
  a * b - 2 * (a + b) = 14 := by sorry

end NUMINAMATH_CALUDE_product_sum_difference_l3579_357976


namespace NUMINAMATH_CALUDE_cake_muffin_mix_probability_l3579_357959

theorem cake_muffin_mix_probability (total_buyers : ℕ) (cake_buyers : ℕ) (muffin_buyers : ℕ) (both_buyers : ℕ)
  (h1 : total_buyers = 100)
  (h2 : cake_buyers = 50)
  (h3 : muffin_buyers = 40)
  (h4 : both_buyers = 19) :
  (total_buyers - (cake_buyers + muffin_buyers - both_buyers)) / total_buyers = 29 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cake_muffin_mix_probability_l3579_357959


namespace NUMINAMATH_CALUDE_train_speed_l3579_357918

/-- The speed of a train given specific conditions involving a jogger --/
theorem train_speed (jogger_speed : ℝ) (jogger_ahead : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 →
  jogger_ahead = 120 →
  train_length = 120 →
  passing_time = 24 →
  ∃ (train_speed : ℝ), train_speed = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3579_357918


namespace NUMINAMATH_CALUDE_string_cheese_cost_is_10_cents_l3579_357926

/-- The cost of each piece of string cheese in cents -/
def string_cheese_cost (num_packs : ℕ) (cheeses_per_pack : ℕ) (total_cost : ℚ) : ℚ :=
  (total_cost * 100) / (num_packs * cheeses_per_pack)

/-- Theorem: The cost of each piece of string cheese is 10 cents -/
theorem string_cheese_cost_is_10_cents :
  string_cheese_cost 3 20 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_string_cheese_cost_is_10_cents_l3579_357926


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l3579_357963

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x + y = 3
def equation2 (x y : ℝ) : Prop := 2 * (x + y) - y = 5

-- Theorem stating that (2, 1) is the solution
theorem solution_satisfies_system :
  equation1 2 1 ∧ equation2 2 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l3579_357963


namespace NUMINAMATH_CALUDE_find_number_l3579_357934

/-- Given the equation (47% of 1442 - 36% of N) + 66 = 6, prove that N = 2049.28 --/
theorem find_number (N : ℝ) : (0.47 * 1442 - 0.36 * N) + 66 = 6 → N = 2049.28 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3579_357934


namespace NUMINAMATH_CALUDE_hexagon_area_l3579_357991

-- Define the hexagon vertices
def hexagon_vertices : List (ℝ × ℝ) := [(0,0), (1,4), (3,4), (4,0), (3,-4), (1,-4)]

-- Function to calculate the area of a polygon given its vertices
def polygon_area (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem hexagon_area : polygon_area hexagon_vertices = 24 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_l3579_357991


namespace NUMINAMATH_CALUDE_no_four_digit_perfect_square_palindromes_l3579_357952

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem no_four_digit_perfect_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end NUMINAMATH_CALUDE_no_four_digit_perfect_square_palindromes_l3579_357952


namespace NUMINAMATH_CALUDE_pots_per_vertical_stack_l3579_357981

theorem pots_per_vertical_stack (total_pots : ℕ) (num_shelves : ℕ) (sets_per_shelf : ℕ) : 
  total_pots = 60 → num_shelves = 4 → sets_per_shelf = 3 → 
  (total_pots / (num_shelves * sets_per_shelf) : ℕ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_pots_per_vertical_stack_l3579_357981


namespace NUMINAMATH_CALUDE_even_function_order_l3579_357947

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 6 * m * x + 2

-- State the theorem
theorem even_function_order (m : ℝ) :
  (∀ x, f m x = f m (-x)) →
  f m (-2) < f m 1 ∧ f m 1 < f m 0 := by
sorry


end NUMINAMATH_CALUDE_even_function_order_l3579_357947


namespace NUMINAMATH_CALUDE_divisible_by_seven_l3579_357995

theorem divisible_by_seven (x y : ℕ+) (a b : ℕ) 
  (h1 : 3 * x.val + 4 * y.val = a ^ 2)
  (h2 : 4 * x.val + 3 * y.val = b ^ 2) : 
  7 ∣ x.val ∧ 7 ∣ y.val := by
sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l3579_357995


namespace NUMINAMATH_CALUDE_average_equation_solution_l3579_357966

theorem average_equation_solution (x : ℝ) : 
  (1/3 : ℝ) * ((x + 8) + (5*x + 3) + (3*x + 4)) = 4*x + 1 → x = 4 := by
sorry

end NUMINAMATH_CALUDE_average_equation_solution_l3579_357966


namespace NUMINAMATH_CALUDE_integral_of_f_l3579_357938

-- Define the function f(x) = |x + 2|
def f (x : ℝ) : ℝ := |x + 2|

-- State the theorem
theorem integral_of_f : ∫ x in (-4)..3, f x = 29/2 := by sorry

end NUMINAMATH_CALUDE_integral_of_f_l3579_357938


namespace NUMINAMATH_CALUDE_third_home_donation_l3579_357942

/-- Represents the donation amounts in cents to avoid floating-point issues -/
def total_donation : ℕ := 70000
def first_home_donation : ℕ := 24500
def second_home_donation : ℕ := 22500

/-- The donation to the third home is the difference between the total donation
    and the sum of donations to the first two homes -/
theorem third_home_donation :
  total_donation - first_home_donation - second_home_donation = 23000 := by
  sorry

end NUMINAMATH_CALUDE_third_home_donation_l3579_357942


namespace NUMINAMATH_CALUDE_paintings_distribution_l3579_357973

/-- Given a total number of paintings, number of rooms, and paintings kept in a private study,
    calculate the number of paintings placed in each room. -/
def paintings_per_room (total : ℕ) (rooms : ℕ) (kept : ℕ) : ℕ :=
  (total - kept) / rooms

/-- Theorem stating that given 47 total paintings, 6 rooms, and 5 paintings kept in a private study,
    the number of paintings placed in each room is 7. -/
theorem paintings_distribution :
  paintings_per_room 47 6 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_paintings_distribution_l3579_357973


namespace NUMINAMATH_CALUDE_smallest_complex_magnitude_l3579_357968

theorem smallest_complex_magnitude (z : ℂ) (h : Complex.abs (z - 9) + Complex.abs (z - (0 + 4*I)) = 15) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 36 / Real.sqrt 97 :=
sorry

end NUMINAMATH_CALUDE_smallest_complex_magnitude_l3579_357968


namespace NUMINAMATH_CALUDE_pushup_percentage_l3579_357920

def jumping_jacks : ℕ := 12
def pushups : ℕ := 8
def situps : ℕ := 20

def total_exercises : ℕ := jumping_jacks + pushups + situps

theorem pushup_percentage :
  (pushups : ℚ) / (total_exercises : ℚ) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_pushup_percentage_l3579_357920


namespace NUMINAMATH_CALUDE_total_stairs_climbed_l3579_357933

def samir_stairs : ℕ := 318

def veronica_stairs : ℕ := samir_stairs / 2 + 18

theorem total_stairs_climbed : samir_stairs + veronica_stairs = 495 := by
  sorry

end NUMINAMATH_CALUDE_total_stairs_climbed_l3579_357933


namespace NUMINAMATH_CALUDE_factorial_100_trailing_zeros_l3579_357996

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- 100! has 24 trailing zeros -/
theorem factorial_100_trailing_zeros :
  trailingZeros 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_factorial_100_trailing_zeros_l3579_357996


namespace NUMINAMATH_CALUDE_min_value_parallel_vectors_l3579_357982

/-- Given vectors a and b, where a is parallel to b, prove the minimum value of 3^x + 9^y + 2 -/
theorem min_value_parallel_vectors (x y : ℝ) :
  let a : Fin 2 → ℝ := ![3 - x, y]
  let b : Fin 2 → ℝ := ![2, 1]
  (∃ (k : ℝ), a = k • b) →
  (∀ (x' y' : ℝ), 3^x' + 9^y' + 2 ≥ 6 * Real.sqrt 3 + 2) ∧
  (∃ (x₀ y₀ : ℝ), 3^x₀ + 9^y₀ + 2 = 6 * Real.sqrt 3 + 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_parallel_vectors_l3579_357982


namespace NUMINAMATH_CALUDE_fraction_upper_bound_l3579_357944

theorem fraction_upper_bound (x : ℝ) (h : x > 0) : x / (x^2 + 3*x + 1) ≤ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_upper_bound_l3579_357944


namespace NUMINAMATH_CALUDE_sum_zero_from_absolute_value_inequalities_l3579_357921

theorem sum_zero_from_absolute_value_inequalities (a b c : ℝ) 
  (h1 : |a| ≥ |b+c|) 
  (h2 : |b| ≥ |c+a|) 
  (h3 : |c| ≥ |a+b|) : 
  a + b + c = 0 := by 
sorry

end NUMINAMATH_CALUDE_sum_zero_from_absolute_value_inequalities_l3579_357921


namespace NUMINAMATH_CALUDE_box_fits_40_blocks_l3579_357908

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℕ
  width : ℕ
  length : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ :=
  d.height * d.width * d.length

/-- Calculates how many smaller objects can fit into a larger object -/
def fitCount (larger smaller : Dimensions) : ℕ :=
  (volume larger) / (volume smaller)

theorem box_fits_40_blocks : 
  let box := Dimensions.mk 8 10 12
  let block := Dimensions.mk 3 2 4
  fitCount box block = 40 := by
  sorry

#eval fitCount (Dimensions.mk 8 10 12) (Dimensions.mk 3 2 4)

end NUMINAMATH_CALUDE_box_fits_40_blocks_l3579_357908


namespace NUMINAMATH_CALUDE_complex_power_sum_l3579_357925

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^500 + 1/(z^500) = 2 * Real.cos (100 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3579_357925


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l3579_357992

/-- The perimeter of a semicircle with radius 3.1 cm is equal to π * 3.1 + 6.2 cm. -/
theorem semicircle_perimeter :
  let r : Real := 3.1
  let perimeter := π * r + 2 * r
  perimeter = π * 3.1 + 6.2 := by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l3579_357992


namespace NUMINAMATH_CALUDE_initial_storks_count_storks_on_fence_l3579_357997

/-- Given a fence with birds and storks, prove the initial number of storks. -/
theorem initial_storks_count (initial_birds : ℕ) (additional_birds : ℕ) (stork_bird_difference : ℕ) : ℕ :=
  let final_birds := initial_birds + additional_birds
  let storks := final_birds + stork_bird_difference
  storks

/-- Prove that the number of storks initially on the fence is 6. -/
theorem storks_on_fence :
  initial_storks_count 2 3 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_storks_count_storks_on_fence_l3579_357997


namespace NUMINAMATH_CALUDE_same_day_ticket_cost_l3579_357949

/-- Proves that the cost of same-day tickets is $30 given the specified conditions -/
theorem same_day_ticket_cost
  (total_tickets : ℕ)
  (total_receipts : ℕ)
  (advance_ticket_cost : ℕ)
  (advance_tickets_sold : ℕ)
  (h1 : total_tickets = 60)
  (h2 : total_receipts = 1600)
  (h3 : advance_ticket_cost = 20)
  (h4 : advance_tickets_sold = 20) :
  (total_receipts - advance_ticket_cost * advance_tickets_sold) / (total_tickets - advance_tickets_sold) = 30 :=
by sorry

end NUMINAMATH_CALUDE_same_day_ticket_cost_l3579_357949


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3579_357983

theorem quadratic_real_roots (a b : ℝ) : 
  (∃ x : ℝ, x^2 + 2*(1+a)*x + (3*a^2 + 4*a*b + 4*b^2 + 2) = 0) ↔ (a = 1 ∧ b = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3579_357983


namespace NUMINAMATH_CALUDE_birds_left_after_week_l3579_357978

/-- Calculates the number of birds left in a poultry farm after a week of disease -/
def birdsLeftAfterWeek (initialChickens initialTurkeys initialGuineaFowls : ℕ)
                       (dailyLossChickens dailyLossTurkeys dailyLossGuineaFowls : ℕ) : ℕ :=
  let daysInWeek : ℕ := 7
  let chickensLeft := initialChickens - daysInWeek * dailyLossChickens
  let turkeysLeft := initialTurkeys - daysInWeek * dailyLossTurkeys
  let guineaFowlsLeft := initialGuineaFowls - daysInWeek * dailyLossGuineaFowls
  chickensLeft + turkeysLeft + guineaFowlsLeft

theorem birds_left_after_week :
  birdsLeftAfterWeek 300 200 80 20 8 5 = 349 := by
  sorry

end NUMINAMATH_CALUDE_birds_left_after_week_l3579_357978


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_l3579_357900

/-- The sum of the infinite geometric series 5/3 - 5/8 + 25/128 - 125/1024 + ... -/
def infiniteGeometricSeriesSum : ℚ := 8/3

/-- The first term of the geometric series -/
def firstTerm : ℚ := 5/3

/-- The common ratio of the geometric series -/
def commonRatio : ℚ := 3/8

/-- Theorem stating that the sum of the infinite geometric series is 8/3 -/
theorem infinite_geometric_series_sum :
  infiniteGeometricSeriesSum = firstTerm / (1 - commonRatio) :=
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_l3579_357900


namespace NUMINAMATH_CALUDE_gcf_of_60_and_75_l3579_357999

theorem gcf_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_60_and_75_l3579_357999


namespace NUMINAMATH_CALUDE_sports_lottery_combinations_and_cost_l3579_357986

/-- The number of ways to choose 3 consecutive numbers from 01 to 17 -/
def consecutive_three : ℕ := 15

/-- The number of ways to choose 2 consecutive numbers from 19 to 29 -/
def consecutive_two : ℕ := 10

/-- The number of ways to choose 1 number from 30 to 36 -/
def single_number : ℕ := 7

/-- The cost of each bet in yuan -/
def bet_cost : ℕ := 2

/-- The total number of combinations -/
def total_combinations : ℕ := consecutive_three * consecutive_two * single_number

/-- The total cost in yuan -/
def total_cost : ℕ := total_combinations * bet_cost

theorem sports_lottery_combinations_and_cost :
  total_combinations = 1050 ∧ total_cost = 2100 := by
  sorry

end NUMINAMATH_CALUDE_sports_lottery_combinations_and_cost_l3579_357986


namespace NUMINAMATH_CALUDE_abs_sum_minimum_l3579_357916

theorem abs_sum_minimum (x : ℝ) : 
  |x + 3| + |x + 6| + |x + 8| ≥ 5 ∧ ∃ y : ℝ, |y + 3| + |y + 6| + |y + 8| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_minimum_l3579_357916


namespace NUMINAMATH_CALUDE_factors_of_81_l3579_357955

theorem factors_of_81 : Finset.card (Nat.divisors 81) = 5 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_81_l3579_357955


namespace NUMINAMATH_CALUDE_equation_solution_l3579_357984

theorem equation_solution :
  ∃ x : ℚ, (1 / 3 + 1 / x = 2 / 3) ∧ (x = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3579_357984


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3579_357971

def tournament_games (n : ℕ) : ℕ := n * (n - 1)

theorem chess_tournament_games :
  tournament_games 17 * 2 = 544 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3579_357971


namespace NUMINAMATH_CALUDE_constant_geometric_sequence_l3579_357937

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = q * b n

theorem constant_geometric_sequence
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_geom : is_geometric_sequence b)
  (h_relation : ∀ n : ℕ, a (n + 1) / a n = b n) :
  ∀ n : ℕ, b n = 1 := by
sorry

end NUMINAMATH_CALUDE_constant_geometric_sequence_l3579_357937


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3579_357977

/-- Given a geometric sequence {a_n} where all terms are positive,
    if a_3 * a_5 + a_2 * a_10 + 2 * a_4 * a_6 = 100,
    then a_4 + a_6 = 10 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geo : ∀ n m : ℕ, a (n + m) = a n * (a 2) ^ (m - 1))
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_sum : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100) :
  a 4 + a 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3579_357977


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_less_than_100_l3579_357923

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n = 99 ∧ 9 ∣ n ∧ n < 100 ∧ ∀ (m : ℕ), 9 ∣ m → m < 100 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_less_than_100_l3579_357923


namespace NUMINAMATH_CALUDE_max_b_value_l3579_357936

theorem max_b_value (a b : ℤ) (h : (127 : ℚ) / a - (16 : ℚ) / b = 1) : b ≤ 2016 := by
  sorry

end NUMINAMATH_CALUDE_max_b_value_l3579_357936


namespace NUMINAMATH_CALUDE_player_A_wins_l3579_357998

/-- Represents a card with a digit from 0 to 6 -/
inductive Card : Type
| zero | one | two | three | four | five | six

/-- Represents a player in the game -/
inductive Player : Type
| A | B

/-- Represents the state of the game -/
structure GameState :=
(remaining_cards : List Card)
(player_A_cards : List Card)
(player_B_cards : List Card)
(current_player : Player)

/-- Checks if a list of cards can form a number divisible by 17 -/
def can_form_divisible_by_17 (cards : List Card) : Bool :=
  sorry

/-- Determines the winner of the game given optimal play -/
def optimal_play_winner (initial_state : GameState) : Player :=
  sorry

/-- The main theorem stating that Player A wins with optimal play -/
theorem player_A_wins :
  ∀ (initial_state : GameState),
    initial_state.remaining_cards = [Card.zero, Card.one, Card.two, Card.three, Card.four, Card.five, Card.six] →
    initial_state.player_A_cards = [] →
    initial_state.player_B_cards = [] →
    initial_state.current_player = Player.A →
    optimal_play_winner initial_state = Player.A :=
  sorry

end NUMINAMATH_CALUDE_player_A_wins_l3579_357998


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3579_357939

theorem absolute_value_inequality (x : ℝ) (h : x ≠ 2) :
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3579_357939


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l3579_357905

theorem fraction_equality_sum (M N : ℚ) : 
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + N = 183 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l3579_357905


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l3579_357979

/-- The custom operation ⊗ -/
def tensor (a b c d : ℂ) : ℂ := a * c - b * d

/-- The complex number z satisfying the given equation -/
noncomputable def z : ℂ := sorry

/-- The statement to prove -/
theorem z_in_second_quadrant :
  tensor z (1 - 2*I) (-1) (1 + I) = 0 →
  z.re < 0 ∧ z.im > 0 := by sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l3579_357979


namespace NUMINAMATH_CALUDE_rectangle_to_parallelogram_perimeter_l3579_357961

/-- A rectangle is transformed into a parallelogram while maintaining the same perimeter -/
theorem rectangle_to_parallelogram_perimeter (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) :
  let rectangle_perimeter := 2 * (a + b)
  let parallelogram_perimeter := 2 * (a + b)
  rectangle_perimeter = parallelogram_perimeter :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_parallelogram_perimeter_l3579_357961


namespace NUMINAMATH_CALUDE_day_statistics_order_l3579_357962

/-- Represents the frequency distribution of days in a non-leap year -/
def day_frequency (n : ℕ) : ℕ :=
  if n ≤ 28 then 12
  else if n ≤ 30 then 11
  else if n = 31 then 6
  else 0

/-- The total number of days in a non-leap year -/
def total_days : ℕ := 365

/-- The median of modes for the day distribution -/
def median_of_modes : ℚ := 14.5

/-- The median of the day distribution -/
def median : ℕ := 13

/-- The mean of the day distribution -/
def mean : ℚ := 5707 / 365

theorem day_statistics_order :
  median_of_modes < median ∧ (median : ℚ) < mean :=
sorry

end NUMINAMATH_CALUDE_day_statistics_order_l3579_357962


namespace NUMINAMATH_CALUDE_softball_players_l3579_357990

/-- The number of softball players in a games hour -/
theorem softball_players (cricket hockey football total : ℕ) 
  (h1 : cricket = 10)
  (h2 : hockey = 12)
  (h3 : football = 16)
  (h4 : total = 51)
  (h5 : total = cricket + hockey + football + softball) : 
  softball = 13 := by
  sorry

end NUMINAMATH_CALUDE_softball_players_l3579_357990


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3579_357988

theorem opposite_of_negative_fraction (n : ℕ) (hn : n ≠ 0) :
  -(-(1 : ℚ) / n) = 1 / n :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3579_357988


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3579_357958

/-- Given positive integers A, B, C, and integer D, where A, B, C form an arithmetic sequence,
    B, C, D form a geometric sequence, and C/B = 7/3, the smallest possible value of A + B + C + D is 76. -/
theorem smallest_sum_of_sequence (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →
  (∃ r : ℤ, C - B = B - A) →  -- arithmetic sequence condition
  (∃ q : ℚ, C = B * q ∧ D = C * q) →  -- geometric sequence condition
  C = (7 * B) / 3 →
  (∀ A' B' C' D' : ℤ, 
    A' > 0 → B' > 0 → C' > 0 →
    (∃ r : ℤ, C' - B' = B' - A') →
    (∃ q : ℚ, C' = B' * q ∧ D' = C' * q) →
    C' = (7 * B') / 3 →
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 76 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3579_357958


namespace NUMINAMATH_CALUDE_additional_earnings_calculation_l3579_357911

/-- Represents the financial data for a company's quarterly earnings and dividends. -/
structure CompanyFinancials where
  expectedEarnings : ℝ
  actualEarnings : ℝ
  additionalDividendRate : ℝ

/-- Calculates the additional earnings per share based on the company's financial data. -/
def additionalEarnings (cf : CompanyFinancials) : ℝ :=
  cf.actualEarnings - cf.expectedEarnings

/-- Theorem stating that the additional earnings per share is the difference between
    actual and expected earnings. -/
theorem additional_earnings_calculation (cf : CompanyFinancials) 
    (h1 : cf.expectedEarnings = 0.80)
    (h2 : cf.actualEarnings = 1.10)
    (h3 : cf.additionalDividendRate = 0.04) :
    additionalEarnings cf = 0.30 := by
  sorry

end NUMINAMATH_CALUDE_additional_earnings_calculation_l3579_357911


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_l3579_357913

theorem gcd_of_powers_of_two : Nat.gcd (2^2016 - 1) (2^2008 - 1) = 2^8 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_l3579_357913


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l3579_357970

def U : Set ℕ := {1, 2, 3, 4, 5}

def A : Set ℕ := {x ∈ U | x^2 - 3*x + 2 = 0}

def B : Set ℕ := {x ∈ U | ∃ a ∈ A, x = 2*a}

theorem complement_of_A_union_B (h : Set ℕ) : 
  h = U \ (A ∪ B) → h = {3, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l3579_357970


namespace NUMINAMATH_CALUDE_chef_pies_l3579_357985

theorem chef_pies (apple_pies pecan_pies pumpkin_pies total_pies : ℕ) 
  (h1 : apple_pies = 2)
  (h2 : pecan_pies = 4)
  (h3 : total_pies = 13)
  (h4 : total_pies = apple_pies + pecan_pies + pumpkin_pies) :
  pumpkin_pies = 7 := by
  sorry

end NUMINAMATH_CALUDE_chef_pies_l3579_357985


namespace NUMINAMATH_CALUDE_max_intersection_points_fifth_degree_polynomials_l3579_357975

/-- A fifth-degree polynomial function with leading coefficient 1 -/
def FifthDegreePolynomial (a b c d e : ℝ) : ℝ → ℝ := 
  λ x => x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- The difference between two fifth-degree polynomials where one has an additional -x^3 term -/
def PolynomialDifference (p q : ℝ → ℝ) : ℝ → ℝ :=
  λ x => p x - q x

theorem max_intersection_points_fifth_degree_polynomials :
  ∀ (a₁ b₁ c₁ d₁ e₁ a₂ b₂ c₂ d₂ e₂ : ℝ),
  let p := FifthDegreePolynomial a₁ b₁ c₁ d₁ e₁
  let q := FifthDegreePolynomial a₂ (b₂ - 1) c₂ d₂ e₂
  let diff := PolynomialDifference p q
  (∀ x : ℝ, diff x = 0 → x = 0) ∧
  (∃ x : ℝ, diff x = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_intersection_points_fifth_degree_polynomials_l3579_357975


namespace NUMINAMATH_CALUDE_triangle_2_3_4_l3579_357945

-- Define the triangle operation
def triangle (a b c : ℝ) : ℝ := b^3 - 5*a*c

-- Theorem statement
theorem triangle_2_3_4 : triangle 2 3 4 = -13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_2_3_4_l3579_357945


namespace NUMINAMATH_CALUDE_man_son_age_difference_l3579_357912

/-- Given a man and his son, proves that the man is 20 years older than his son. -/
theorem man_son_age_difference (man_age son_age : ℕ) : 
  son_age = 18 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_difference_l3579_357912


namespace NUMINAMATH_CALUDE_fifth_power_sum_l3579_357965

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a*x + b*y = 2)
  (h2 : a*x^2 + b*y^2 = 5)
  (h3 : a*x^3 + b*y^3 = 15)
  (h4 : a*x^4 + b*y^4 = 35) :
  a*x^5 + b*y^5 = 10 := by
sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l3579_357965


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l3579_357993

def quadratic_function (x : ℝ) : ℝ := (x - 1)^2

theorem y1_greater_than_y2 (y₁ y₂ : ℝ) 
  (h1 : y₁ = quadratic_function 3)
  (h2 : y₂ = quadratic_function 1) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l3579_357993


namespace NUMINAMATH_CALUDE_smallest_positive_a_l3579_357953

-- Define the equation
def equation (x a : ℚ) : Prop :=
  (((x - a) / 2 + (x - 2*a) / 3) / ((x + 4*a) / 5 - (x + 3*a) / 4)) =
  (((x - 3*a) / 4 + (x - 4*a) / 5) / ((x + 2*a) / 3 - (x + a) / 2))

-- Define what it means for the equation to have an integer root
def has_integer_root (a : ℚ) : Prop :=
  ∃ x : ℤ, equation x a

-- State the theorem
theorem smallest_positive_a : 
  (∀ a : ℚ, 0 < a ∧ a < 419/421 → ¬ has_integer_root a) ∧ 
  has_integer_root (419/421) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_a_l3579_357953


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3579_357915

theorem inequality_system_solution (x : ℝ) :
  (1 - 3 * (x - 1) < 8 - x) ∧ ((x - 3) / 2 + 3 ≥ x) → -2 < x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3579_357915


namespace NUMINAMATH_CALUDE_soccer_field_area_l3579_357941

theorem soccer_field_area (w l : ℝ) (h1 : l = 3 * w - 30) (h2 : 2 * (w + l) = 880) :
  w * l = 37906.25 := by
  sorry

end NUMINAMATH_CALUDE_soccer_field_area_l3579_357941


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3579_357972

theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 5) (hb : b = 10) (hc : c = 20) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 20 * (7 - Real.sqrt 10) / 39 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3579_357972


namespace NUMINAMATH_CALUDE_house_rent_fraction_l3579_357989

theorem house_rent_fraction (salary : ℝ) 
  (food_fraction : ℝ) (conveyance_fraction : ℝ) (left_amount : ℝ) (food_conveyance_expense : ℝ)
  (h1 : food_fraction = 3/10)
  (h2 : conveyance_fraction = 1/8)
  (h3 : left_amount = 1400)
  (h4 : food_conveyance_expense = 3400)
  (h5 : food_fraction * salary + conveyance_fraction * salary = food_conveyance_expense)
  (h6 : salary - (food_fraction * salary + conveyance_fraction * salary + left_amount) = 
        salary * (1 - food_fraction - conveyance_fraction - left_amount / salary)) :
  1 - food_fraction - conveyance_fraction - left_amount / salary = 2/5 := by
sorry

end NUMINAMATH_CALUDE_house_rent_fraction_l3579_357989


namespace NUMINAMATH_CALUDE_tangent_line_and_hyperbola_l3579_357904

/-- Given two functions f(x) = x + 4 and g(x) = k/x that are tangent to each other, 
    prove that k = -4 -/
theorem tangent_line_and_hyperbola (k : ℝ) :
  (∃ x : ℝ, x + 4 = k / x ∧ 
   ∀ y : ℝ, y ≠ x → (y + 4 - k / y) * (x - y) ≠ 0) → 
  k = -4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_hyperbola_l3579_357904


namespace NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l3579_357919

theorem sphere_cylinder_equal_area (r : ℝ) : 
  (4 : ℝ) * Real.pi * r^2 = (2 : ℝ) * Real.pi * 4 * 8 → r = 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l3579_357919


namespace NUMINAMATH_CALUDE_spam_price_theorem_l3579_357967

-- Define the constants from the problem
def peanut_butter_price : ℝ := 5
def bread_price : ℝ := 2
def spam_cans : ℕ := 12
def peanut_butter_jars : ℕ := 3
def bread_loaves : ℕ := 4
def total_paid : ℝ := 59

-- Define the theorem
theorem spam_price_theorem :
  ∃ (spam_price : ℝ),
    spam_price * spam_cans +
    peanut_butter_price * peanut_butter_jars +
    bread_price * bread_loaves = total_paid ∧
    spam_price = 3 :=
by sorry

end NUMINAMATH_CALUDE_spam_price_theorem_l3579_357967


namespace NUMINAMATH_CALUDE_certain_number_proof_l3579_357956

theorem certain_number_proof (y : ℝ) : 
  (0.20 * 1050 = 0.15 * y - 15) → y = 1500 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3579_357956


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l3579_357960

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 17) 
  (h2 : x * y = 6) : 
  x^2 + y^2 = 301 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l3579_357960


namespace NUMINAMATH_CALUDE_selection_theorem_l3579_357914

/-- Represents the number of students who can play only chess -/
def chess_only : ℕ := 2

/-- Represents the number of students who can play only Go -/
def go_only : ℕ := 3

/-- Represents the number of students who can play both chess and Go -/
def both : ℕ := 4

/-- Represents the total number of students -/
def total_students : ℕ := chess_only + go_only + both

/-- Calculates the number of ways to select two students for chess and Go competitions -/
def selection_ways : ℕ :=
  chess_only * go_only +  -- One from chess_only, one from go_only
  both * go_only +        -- One from both for chess, one from go_only
  chess_only * both +     -- One from chess_only, one from both for Go
  (both * (both - 1)) / 2 -- Two from both (combination)

/-- Theorem stating that the number of ways to select students is 32 -/
theorem selection_theorem : selection_ways = 32 := by sorry

end NUMINAMATH_CALUDE_selection_theorem_l3579_357914


namespace NUMINAMATH_CALUDE_fish_problem_l3579_357950

theorem fish_problem (ken_fish : ℕ) (kendra_fish : ℕ) :
  ken_fish = 2 * kendra_fish - 3 →
  ken_fish + kendra_fish = 87 →
  kendra_fish = 30 := by
sorry

end NUMINAMATH_CALUDE_fish_problem_l3579_357950


namespace NUMINAMATH_CALUDE_marias_piggy_bank_l3579_357964

/-- Represents the number of coins of each type -/
structure CoinCount where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ

/-- Calculates the total value of coins in dollars -/
def totalValue (coins : CoinCount) : ℚ :=
  0.10 * coins.dimes + 0.25 * coins.quarters + 0.05 * coins.nickels

/-- The problem statement -/
theorem marias_piggy_bank (initialCoins : CoinCount) :
  initialCoins.dimes = 4 →
  initialCoins.quarters = 4 →
  totalValue { dimes := initialCoins.dimes,
               quarters := initialCoins.quarters + 5,
               nickels := initialCoins.nickels } = 3 →
  initialCoins.nickels = 7 := by
  sorry

end NUMINAMATH_CALUDE_marias_piggy_bank_l3579_357964


namespace NUMINAMATH_CALUDE_square_difference_divided_by_eleven_l3579_357969

theorem square_difference_divided_by_eleven : (121^2 - 110^2) / 11 = 231 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_eleven_l3579_357969


namespace NUMINAMATH_CALUDE_expand_expression_l3579_357924

theorem expand_expression (x : ℝ) : (16*x + 18 - 4*x^2) * 3*x = -12*x^3 + 48*x^2 + 54*x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3579_357924


namespace NUMINAMATH_CALUDE_ratio_proof_l3579_357994

theorem ratio_proof (x y z : ℚ) :
  (5 * x + 4 * y - 6 * z) / (4 * x - 5 * y + 7 * z) = 1 / 27 ∧
  (5 * x + 4 * y - 6 * z) / (6 * x + 5 * y - 4 * z) = 1 / 18 →
  ∃ (k : ℚ), x = 3 * k ∧ y = 4 * k ∧ z = 5 * k :=
by sorry

end NUMINAMATH_CALUDE_ratio_proof_l3579_357994


namespace NUMINAMATH_CALUDE_blue_pill_cost_is_correct_l3579_357954

/-- The cost of the blue pill in dollars -/
def blue_pill_cost : ℝ := 17

/-- The cost of the red pill in dollars -/
def red_pill_cost : ℝ := blue_pill_cost - 2

/-- The number of days for the treatment -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℝ := 672

theorem blue_pill_cost_is_correct :
  blue_pill_cost = 17 ∧
  red_pill_cost = blue_pill_cost - 2 ∧
  treatment_days * (blue_pill_cost + red_pill_cost) = total_cost := by
  sorry

#eval blue_pill_cost

end NUMINAMATH_CALUDE_blue_pill_cost_is_correct_l3579_357954


namespace NUMINAMATH_CALUDE_spinner_direction_final_direction_is_west_l3579_357902

-- Define the possible directions
inductive Direction
  | North
  | South
  | East
  | West

-- Define a function to rotate a direction
def rotate (d : Direction) (revolutions : ℚ) : Direction :=
  sorry

-- State the theorem
theorem spinner_direction (initial : Direction) 
  (clockwise : ℚ) (counterclockwise : ℚ) : Direction :=
  by
  -- Assume the initial direction is south
  have h1 : initial = Direction.South := by sorry
  -- Assume clockwise rotation is 3½ revolutions
  have h2 : clockwise = 7/2 := by sorry
  -- Assume counterclockwise rotation is 1¾ revolutions
  have h3 : counterclockwise = 7/4 := by sorry
  -- Prove that the final direction is west
  sorry

-- The main theorem
theorem final_direction_is_west :
  spinner_direction Direction.South (7/2) (7/4) = Direction.West :=
  by sorry

end NUMINAMATH_CALUDE_spinner_direction_final_direction_is_west_l3579_357902
