import Mathlib

namespace NUMINAMATH_CALUDE_birth_year_proof_l1409_140921

/-- A person born in the first half of the 19th century whose age was x in the year x^2 was born in 1806 -/
theorem birth_year_proof (x : ℕ) (h1 : 1800 < x^2) (h2 : x^2 < 1850) (h3 : x^2 - x = 1806) : 
  x^2 - x = 1806 := by sorry

end NUMINAMATH_CALUDE_birth_year_proof_l1409_140921


namespace NUMINAMATH_CALUDE_total_books_l1409_140916

/-- The number of books each person has -/
structure Books where
  beatrix : ℕ
  alannah : ℕ
  queen : ℕ

/-- The conditions of the problem -/
def book_conditions (b : Books) : Prop :=
  b.beatrix = 30 ∧
  b.alannah = b.beatrix + 20 ∧
  b.queen = b.alannah + (b.alannah / 5)

/-- The theorem to prove -/
theorem total_books (b : Books) :
  book_conditions b → b.beatrix + b.alannah + b.queen = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l1409_140916


namespace NUMINAMATH_CALUDE_line_symmetry_l1409_140995

-- Define the lines
def l (x y : ℝ) : Prop := x - y - 1 = 0
def l₁ (x y : ℝ) : Prop := 2*x - y - 2 = 0
def l₂ (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (f g h : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y → ∃ x' y', g x' y' ∧ h x y ∧
    ((x + x') / 2, (y + y') / 2) ∈ {(a, b) | f a b}

-- Theorem statement
theorem line_symmetry :
  symmetric_wrt l l₁ l₂ :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l1409_140995


namespace NUMINAMATH_CALUDE_fraction_simplification_l1409_140914

theorem fraction_simplification :
  ((3^2008)^2 - (3^2006)^2) / ((3^2007)^2 - (3^2005)^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1409_140914


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l1409_140924

def line (x : ℝ) : ℝ := x - 2

theorem y_intercept_of_line :
  line 0 = -2 := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l1409_140924


namespace NUMINAMATH_CALUDE_supplement_of_supplement_58_l1409_140923

def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_supplement_58 :
  supplement (supplement 58) = 58 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_supplement_58_l1409_140923


namespace NUMINAMATH_CALUDE_rectangle_x_value_l1409_140985

/-- Given a rectangle with vertices (x, 1), (1, 1), (1, -4), and (x, -4) and area 30, prove that x = -5 -/
theorem rectangle_x_value (x : ℝ) : 
  let vertices := [(x, 1), (1, 1), (1, -4), (x, -4)]
  let width := 1 - (-4)
  let area := 30
  let length := area / width
  x = 1 - length → x = -5 := by sorry

end NUMINAMATH_CALUDE_rectangle_x_value_l1409_140985


namespace NUMINAMATH_CALUDE_no_geometric_progression_of_2n_plus_1_l1409_140958

theorem no_geometric_progression_of_2n_plus_1 :
  ¬ ∃ (k m n : ℕ), k ≠ m ∧ m ≠ n ∧ k ≠ n ∧
    (2^m + 1)^2 = (2^k + 1) * (2^n + 1) :=
sorry

end NUMINAMATH_CALUDE_no_geometric_progression_of_2n_plus_1_l1409_140958


namespace NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_1_l1409_140980

theorem sin_50_plus_sqrt3_tan_10_equals_1 :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_1_l1409_140980


namespace NUMINAMATH_CALUDE_f_max_value_l1409_140975

/-- The quadratic function f(x) = -2x^2 - 8x + 16 -/
def f (x : ℝ) : ℝ := -2 * x^2 - 8 * x + 16

/-- The maximum value of f(x) -/
def max_value : ℝ := 24

/-- The x-coordinate where f(x) achieves its maximum value -/
def max_point : ℝ := -2

theorem f_max_value :
  (∀ x : ℝ, f x ≤ max_value) ∧ f max_point = max_value := by sorry

end NUMINAMATH_CALUDE_f_max_value_l1409_140975


namespace NUMINAMATH_CALUDE_toy_store_shelves_l1409_140953

/-- Calculates the number of shelves needed to display stuffed bears in a toy store. -/
def shelves_needed (initial_stock : ℕ) (new_shipment : ℕ) (bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

/-- Proves that given the initial conditions, the number of shelves needed is 4. -/
theorem toy_store_shelves :
  shelves_needed 6 18 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_shelves_l1409_140953


namespace NUMINAMATH_CALUDE_abs_x_less_than_2_sufficient_not_necessary_l1409_140901

theorem abs_x_less_than_2_sufficient_not_necessary :
  (∀ x : ℝ, (|x| < 2 ↔ -2 < x ∧ x < 2)) →
  (∀ x : ℝ, (x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3)) →
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧
  ¬(∀ x : ℝ, x^2 - x - 6 < 0 → |x| < 2) :=
by sorry

end NUMINAMATH_CALUDE_abs_x_less_than_2_sufficient_not_necessary_l1409_140901


namespace NUMINAMATH_CALUDE_nate_optimal_speed_l1409_140966

/-- The speed at which Nate should drive to arrive just in time -/
def optimal_speed : ℝ := 48

/-- The time it takes for Nate to arrive on time -/
def on_time : ℝ := 5

/-- The distance Nate needs to travel -/
def distance : ℝ := 240

theorem nate_optimal_speed :
  (distance = 40 * (on_time + 1)) ∧
  (distance = 60 * (on_time - 1)) →
  optimal_speed = distance / on_time :=
by sorry

end NUMINAMATH_CALUDE_nate_optimal_speed_l1409_140966


namespace NUMINAMATH_CALUDE_banana_count_l1409_140990

theorem banana_count (total : ℕ) (apple_multiplier persimmon_multiplier : ℕ) 
  (h1 : total = 210)
  (h2 : apple_multiplier = 4)
  (h3 : persimmon_multiplier = 3) :
  ∃ (banana_count : ℕ), 
    banana_count * (apple_multiplier + persimmon_multiplier) = total ∧ 
    banana_count = 30 := by
  sorry

end NUMINAMATH_CALUDE_banana_count_l1409_140990


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l1409_140926

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distributeBalls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 41 ways to distribute 5 distinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : distributeBalls 5 3 = 41 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l1409_140926


namespace NUMINAMATH_CALUDE_vector_parallel_implies_x_eq_half_l1409_140907

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

-- Define the parallel condition
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v 0 * w 1 = k * v 1 * w 0

-- State the theorem
theorem vector_parallel_implies_x_eq_half :
  ∀ x : ℝ, are_parallel (a + 2 • b x) (2 • a - 2 • b x) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_x_eq_half_l1409_140907


namespace NUMINAMATH_CALUDE_infinitely_many_primes_composite_l1409_140986

theorem infinitely_many_primes_composite (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ¬Nat.Prime (a * p + b)} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_composite_l1409_140986


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1409_140964

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x * (x - 2) < 0}
def B : Set ℝ := {x : ℝ | Real.log x > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1409_140964


namespace NUMINAMATH_CALUDE_system_equation_solution_l1409_140981

theorem system_equation_solution :
  ∃ (x y : ℝ),
    (4 * x + y = 15) ∧
    (x + 4 * y = 18) ∧
    (13 * x^2 + 14 * x * y + 13 * y^2 = 438.6) := by
  sorry

end NUMINAMATH_CALUDE_system_equation_solution_l1409_140981


namespace NUMINAMATH_CALUDE_two_fifths_in_nine_thirds_l1409_140906

theorem two_fifths_in_nine_thirds : (9 / 3) / (2 / 5) = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_in_nine_thirds_l1409_140906


namespace NUMINAMATH_CALUDE_Q_no_real_roots_l1409_140976

def Q (x : ℝ) : ℝ := x^6 - 3*x^5 + 6*x^4 - 6*x^3 - x + 8

theorem Q_no_real_roots : ∀ x : ℝ, Q x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_Q_no_real_roots_l1409_140976


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l1409_140965

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = 1)
  (h_sum : a 3 + a 4 + a 5 + a 6 = 20) :
  a 8 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l1409_140965


namespace NUMINAMATH_CALUDE_complement_of_union_M_P_l1409_140996

open Set

-- Define the universal set U as the real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x ≤ 1}

-- Define set P
def P : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_of_union_M_P : 
  (M ∪ P)ᶜ = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_M_P_l1409_140996


namespace NUMINAMATH_CALUDE_inequality_problem_l1409_140922

theorem inequality_problem (a b x y : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
  (h_sum : a + b = 1) : 
  (a*x + b*y) * (b*x + a*y) ≥ x*y := by
sorry

end NUMINAMATH_CALUDE_inequality_problem_l1409_140922


namespace NUMINAMATH_CALUDE_sector_area_l1409_140963

/-- Given a circular sector with central angle 1 radian and circumference 6,
    prove that its area is 2. -/
theorem sector_area (θ : Real) (c : Real) (h1 : θ = 1) (h2 : c = 6) :
  let r := c / 3
  (1/2) * r^2 * θ = 2 := by sorry

end NUMINAMATH_CALUDE_sector_area_l1409_140963


namespace NUMINAMATH_CALUDE_debate_team_combinations_l1409_140950

theorem debate_team_combinations (n : ℕ) (k : ℕ) : n = 7 → k = 4 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_combinations_l1409_140950


namespace NUMINAMATH_CALUDE_quadratic_ratio_l1409_140931

/-- Given a quadratic polynomial x^2 + 1500x + 1800, prove that when written in the form (x+a)^2 + d,
    the ratio d/a equals -560700/750. -/
theorem quadratic_ratio (x : ℝ) :
  ∃ (a d : ℝ), x^2 + 1500*x + 1800 = (x + a)^2 + d ∧ d / a = -560700 / 750 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l1409_140931


namespace NUMINAMATH_CALUDE_smallest_c_for_no_five_l1409_140932

theorem smallest_c_for_no_five : ∃ c : ℤ, (∀ x : ℝ, x^2 + c*x + 10 ≠ 5) ∧ 
  (∀ c' : ℤ, c' < c → ∃ x : ℝ, x^2 + c'*x + 10 = 5) :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_for_no_five_l1409_140932


namespace NUMINAMATH_CALUDE_prime_sum_product_l1409_140915

theorem prime_sum_product : ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p + q = 95 ∧ p * q = 178 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_product_l1409_140915


namespace NUMINAMATH_CALUDE_area_of_specific_quadrilateral_l1409_140941

/-- The area of a quadrilateral can be calculated using the Shoelace formula -/
def quadrilateralArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  let x1 := v1.1
  let y1 := v1.2
  let x2 := v2.1
  let y2 := v2.2
  let x3 := v3.1
  let y3 := v3.2
  let x4 := v4.1
  let y4 := v4.2
  0.5 * abs ((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

theorem area_of_specific_quadrilateral :
  quadrilateralArea (2, 1) (4, 3) (7, 1) (4, 6) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_specific_quadrilateral_l1409_140941


namespace NUMINAMATH_CALUDE_cubic_polynomial_unique_l1409_140977

/-- A monic cubic polynomial with real coefficients -/
def cubic_polynomial (a b c : ℝ) : ℝ → ℂ :=
  fun x => x^3 + a*x^2 + b*x + c

theorem cubic_polynomial_unique 
  (q : ℝ → ℂ) 
  (h_monic : ∀ x, q x = x^3 + (q 1 - 1) * x^2 + (q 1 - q 0 - 1) * x + q 0)
  (h_root : q (5 - 3*I) = 0)
  (h_const : q 0 = 81) :
  ∀ x, q x = x^3 - (79/16)*x^2 - (17/8)*x + 81 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_unique_l1409_140977


namespace NUMINAMATH_CALUDE_smallest_sum_is_four_ninths_l1409_140927

theorem smallest_sum_is_four_ninths :
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/6, 1/3 + 1/7, 1/3 + 1/9]
  (∀ s ∈ sums, 1/3 + 1/9 ≤ s) ∧ (1/3 + 1/9 = 4/9) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_is_four_ninths_l1409_140927


namespace NUMINAMATH_CALUDE_regular_polygon_45_degree_exterior_angle_is_octagon_l1409_140962

/-- A regular polygon with exterior angles of 45° is a regular octagon -/
theorem regular_polygon_45_degree_exterior_angle_is_octagon :
  ∀ (n : ℕ), n > 2 →
  (360 / n : ℚ) = 45 →
  n = 8 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_45_degree_exterior_angle_is_octagon_l1409_140962


namespace NUMINAMATH_CALUDE_decimal_point_shift_l1409_140920

theorem decimal_point_shift (x : ℝ) : 10 * x = x + 37.89 → 100 * x = 421 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_shift_l1409_140920


namespace NUMINAMATH_CALUDE_ten_parabolas_regions_l1409_140900

/-- The number of regions a circle can be divided into by n parabolas -/
def circle_regions (n : ℕ) : ℕ := 2 * n^2 + 1

/-- Theorem stating that 10 parabolas divide a circle into 201 regions -/
theorem ten_parabolas_regions : circle_regions 10 = 201 := by
  sorry

end NUMINAMATH_CALUDE_ten_parabolas_regions_l1409_140900


namespace NUMINAMATH_CALUDE_special_function_at_zero_l1409_140992

/-- A function satisfying f(x + y) = f(x) + f(y) - xy for all real x and y, with f(1) = 1 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y - x * y) ∧ (f 1 = 1)

/-- Theorem: For a special function f, f(0) = 0 -/
theorem special_function_at_zero {f : ℝ → ℝ} (hf : special_function f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_zero_l1409_140992


namespace NUMINAMATH_CALUDE_quadratic_sum_l1409_140982

/-- A quadratic function with vertex (h, k) and passing through point (x₀, y₀) -/
def quadratic_function (a b c h k x₀ y₀ : ℝ) : Prop :=
  ∀ x, a * (x - h)^2 + k = a * x^2 + b * x + c ∧
  a * (x₀ - h)^2 + k = y₀

theorem quadratic_sum (a b c : ℝ) :
  quadratic_function a b c 2 3 3 2 →
  a + b + 2 * c = 2 := by
  sorry

#check quadratic_sum

end NUMINAMATH_CALUDE_quadratic_sum_l1409_140982


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l1409_140946

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem fifth_term_of_sequence (a₁ d : ℤ) :
  arithmetic_sequence a₁ d 20 = 12 →
  arithmetic_sequence a₁ d 21 = 15 →
  arithmetic_sequence a₁ d 5 = -33 :=
by sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l1409_140946


namespace NUMINAMATH_CALUDE_total_apples_is_eleven_l1409_140949

/-- The number of apples Marin has -/
def marin_apples : ℕ := 9

/-- The number of apples Donald has -/
def donald_apples : ℕ := 2

/-- The total number of apples Marin and Donald have together -/
def total_apples : ℕ := marin_apples + donald_apples

/-- Proof that the total number of apples is 11 -/
theorem total_apples_is_eleven : total_apples = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_is_eleven_l1409_140949


namespace NUMINAMATH_CALUDE_modulo_equivalence_unique_solution_l1409_140929

theorem modulo_equivalence_unique_solution : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2839 [ZMOD 10] ∧ n = 1 := by sorry

end NUMINAMATH_CALUDE_modulo_equivalence_unique_solution_l1409_140929


namespace NUMINAMATH_CALUDE_love_logic_l1409_140956

-- Define the propositions
variable (B : Prop) -- "I love Betty"
variable (J : Prop) -- "I love Jane"

-- State the theorem
theorem love_logic (h1 : B ∨ J) (h2 : B → J) : J ∧ ¬(B ↔ True) :=
  sorry


end NUMINAMATH_CALUDE_love_logic_l1409_140956


namespace NUMINAMATH_CALUDE_half_distance_time_l1409_140973

/-- Represents the total distance of Tony's errands in miles -/
def total_distance : ℝ := 10 + 15 + 5 + 20 + 25

/-- Represents Tony's constant speed in miles per hour -/
def speed : ℝ := 50

/-- Theorem stating that the time taken to drive half the total distance at the given speed is 0.75 hours -/
theorem half_distance_time : (total_distance / 2) / speed = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_half_distance_time_l1409_140973


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l1409_140983

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 + 4 * (a - 2) * x + 1 ≠ 0

def prop_q (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + a * x + 1)

def range_of_a : Set ℝ := Set.Iic (-2) ∪ Set.Ioo 1 2 ∪ Set.Ici 3

theorem range_of_a_theorem :
  (∀ a : ℝ, (prop_p a ∧ ¬prop_q a) ∨ (¬prop_p a ∧ prop_q a)) →
  {a : ℝ | prop_p a ∨ prop_q a} = range_of_a := by sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l1409_140983


namespace NUMINAMATH_CALUDE_find_a_interest_rate_l1409_140987

-- Define constants
def total_amount : ℝ := 10000
def years : ℝ := 2
def b_interest_rate : ℝ := 18
def interest_difference : ℝ := 360
def b_amount : ℝ := 4000

-- Define variables
variable (a_amount : ℝ) (a_interest_rate : ℝ)

-- Theorem statement
theorem find_a_interest_rate :
  a_amount + b_amount = total_amount →
  (a_amount * a_interest_rate * years) / 100 = (b_amount * b_interest_rate * years) / 100 + interest_difference →
  a_interest_rate = 15 := by
  sorry

end NUMINAMATH_CALUDE_find_a_interest_rate_l1409_140987


namespace NUMINAMATH_CALUDE_emir_needs_two_more_dollars_l1409_140930

/-- The amount of additional money Emir needs to buy three books --/
def additional_money_needed (dictionary_cost cookbook_cost dinosaur_book_cost savings : ℕ) : ℕ :=
  (dictionary_cost + cookbook_cost + dinosaur_book_cost) - savings

/-- Theorem: Emir needs $2 more to buy all three books --/
theorem emir_needs_two_more_dollars : 
  additional_money_needed 5 5 11 19 = 2 := by
  sorry

end NUMINAMATH_CALUDE_emir_needs_two_more_dollars_l1409_140930


namespace NUMINAMATH_CALUDE_mod_eight_equivalence_l1409_140913

theorem mod_eight_equivalence : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -3737 [ZMOD 8] ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_mod_eight_equivalence_l1409_140913


namespace NUMINAMATH_CALUDE_initial_average_mark_l1409_140918

/-- Proves that the initial average mark of a class is 80, given specific conditions --/
theorem initial_average_mark (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) : 
  total_students = 10 →
  excluded_students = 5 →
  excluded_avg = 70 →
  remaining_avg = 90 →
  (total_students * (total_students * remaining_avg - excluded_students * excluded_avg)) / 
    (total_students * (total_students - excluded_students)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_mark_l1409_140918


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l1409_140904

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the line
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * x - Real.sqrt 3

-- Define point M
def point_M : ℝ × ℝ := (1, 0)

-- Define the midpoint condition
def is_midpoint (m a b : ℝ × ℝ) : Prop :=
  m.1 = (a.1 + b.1) / 2 ∧ m.2 = (a.2 + b.2) / 2

-- Main theorem
theorem parabola_intersection_theorem :
  ∃ (p : ℝ) (a b : ℝ × ℝ),
    parabola p b.1 b.2 ∧
    line b.1 b.2 ∧
    is_midpoint point_M a b →
    p = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l1409_140904


namespace NUMINAMATH_CALUDE_coffee_cost_calculation_l1409_140952

/-- Calculates the weekly coffee cost for a household -/
def weekly_coffee_cost (people : ℕ) (cups_per_day : ℕ) (oz_per_cup : ℚ) (cost_per_oz : ℚ) : ℚ :=
  people * cups_per_day * oz_per_cup * cost_per_oz * 7

theorem coffee_cost_calculation :
  let people : ℕ := 4
  let cups_per_day : ℕ := 2
  let oz_per_cup : ℚ := 1/2
  let cost_per_oz : ℚ := 5/4
  weekly_coffee_cost people cups_per_day oz_per_cup cost_per_oz = 35 := by
  sorry

#eval weekly_coffee_cost 4 2 (1/2) (5/4)

end NUMINAMATH_CALUDE_coffee_cost_calculation_l1409_140952


namespace NUMINAMATH_CALUDE_aubrey_gum_count_l1409_140902

theorem aubrey_gum_count (john_gum : ℕ) (cole_gum : ℕ) (aubrey_gum : ℕ) 
  (h1 : john_gum = 54)
  (h2 : cole_gum = 45)
  (h3 : john_gum + cole_gum + aubrey_gum = 33 * 3) :
  aubrey_gum = 0 := by
sorry

end NUMINAMATH_CALUDE_aubrey_gum_count_l1409_140902


namespace NUMINAMATH_CALUDE_sum_of_integers_l1409_140954

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 7)
  (eq2 : y - z + w = 8)
  (eq3 : z - w + x = 4)
  (eq4 : w - x + y = 3) :
  x + y + z + w = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1409_140954


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cos_relation_l1409_140947

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_cos_relation (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 + a 5 + a 9 = 8 * Real.pi → Real.cos (a 3 + a 7) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cos_relation_l1409_140947


namespace NUMINAMATH_CALUDE_y1_gt_y2_iff_x_gt_neg_one_fifth_l1409_140968

/-- Given y₁ = a^(2x+1), y₂ = a^(-3x), a > 0, and a > 1, y₁ > y₂ if and only if x > -1/5 -/
theorem y1_gt_y2_iff_x_gt_neg_one_fifth (a x : ℝ) (h1 : a > 0) (h2 : a > 1) :
  a^(2*x + 1) > a^(-3*x) ↔ x > -1/5 := by
  sorry

end NUMINAMATH_CALUDE_y1_gt_y2_iff_x_gt_neg_one_fifth_l1409_140968


namespace NUMINAMATH_CALUDE_f_properties_l1409_140910

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / (x + 1)

def tangent_slope (a : ℝ) (x : ℝ) : ℝ :=
  1 / x - a / ((x + 1) ^ 2)

def critical_point (a : ℝ) : ℝ :=
  (a - 2 + Real.sqrt ((a - 2) ^ 2 + 4)) / 2

theorem f_properties (a : ℝ) (h : a ≥ 0) :
  (tangent_slope 3 1 = 1/4) ∧
  (∀ x > 0, f a x ≤ (2016 - a) * x^3 + (x^2 + a - 1) / (x + 1) →
    (∃ x > 0, (tangent_slope a x = 0) → 4 < a ∧ a ≤ 2016)) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l1409_140910


namespace NUMINAMATH_CALUDE_fraction_inequality_counterexample_l1409_140960

theorem fraction_inequality_counterexample : 
  ∃ (a b c d : ℝ), (a / b > c / d) ∧ (b / a ≥ d / c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_counterexample_l1409_140960


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1409_140925

theorem smallest_prime_divisor_of_sum (n : ℕ) (m : ℕ) :
  2 = Nat.minFac (3^25 + 11^19) := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1409_140925


namespace NUMINAMATH_CALUDE_product_of_solutions_l1409_140934

theorem product_of_solutions (x : ℝ) : (|x| = 3 * (|x| - 2)) → ∃ y : ℝ, (|y| = 3 * (|y| - 2)) ∧ x * y = -9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l1409_140934


namespace NUMINAMATH_CALUDE_sum_of_squares_l1409_140974

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1409_140974


namespace NUMINAMATH_CALUDE_acid_concentration_solution_l1409_140951

/-- Represents the acid concentration problem with three flasks of acid and one of water -/
def AcidConcentrationProblem (acid1 acid2 acid3 : ℝ) (concentration1 concentration2 : ℝ) : Prop :=
  let water1 := acid1 / concentration1 - acid1
  let water2 := acid2 / concentration2 - acid2
  let total_water := water1 + water2
  let concentration3 := acid3 / (acid3 + total_water)
  (acid1 = 10) ∧ 
  (acid2 = 20) ∧ 
  (acid3 = 30) ∧ 
  (concentration1 = 0.05) ∧ 
  (concentration2 = 70/300) ∧ 
  (concentration3 = 0.105)

/-- Theorem stating the solution to the acid concentration problem -/
theorem acid_concentration_solution : 
  ∃ (acid1 acid2 acid3 concentration1 concentration2 : ℝ),
  AcidConcentrationProblem acid1 acid2 acid3 concentration1 concentration2 :=
by
  sorry

end NUMINAMATH_CALUDE_acid_concentration_solution_l1409_140951


namespace NUMINAMATH_CALUDE_custom_calculator_results_l1409_140928

-- Define the custom operation *
noncomputable def customOp (a b : ℤ) : ℤ := 2 * a - b

-- Properties of the custom operation
axiom prop_i (a : ℤ) : customOp a a = a
axiom prop_ii (a : ℤ) : customOp a 0 = 2 * a
axiom prop_iii (a b c d : ℤ) : customOp a b + customOp c d = customOp (a + c) (b + d)

-- Theorem to prove
theorem custom_calculator_results :
  (customOp 2 3 + customOp 0 3 = -2) ∧ (customOp 1024 48 = 2000) := by
  sorry

end NUMINAMATH_CALUDE_custom_calculator_results_l1409_140928


namespace NUMINAMATH_CALUDE_lizard_to_gecko_ratio_l1409_140994

/-- Represents the number of bugs eaten by each animal -/
structure BugsEaten where
  gecko : ℕ
  lizard : ℕ
  frog : ℕ
  toad : ℕ

/-- Conditions of the bug-eating scenario -/
def bugEatingScenario (b : BugsEaten) : Prop :=
  b.gecko = 12 ∧
  b.frog = 3 * b.lizard ∧
  b.toad = (3 * b.lizard) + (3 * b.lizard) / 2 ∧
  b.gecko + b.lizard + b.frog + b.toad = 63

/-- The ratio of bugs eaten by the lizard to bugs eaten by the gecko is 1:2 -/
theorem lizard_to_gecko_ratio (b : BugsEaten) 
  (h : bugEatingScenario b) : b.lizard * 2 = b.gecko := by
  sorry

#check lizard_to_gecko_ratio

end NUMINAMATH_CALUDE_lizard_to_gecko_ratio_l1409_140994


namespace NUMINAMATH_CALUDE_room_width_is_seven_l1409_140984

/-- Represents the dimensions and features of a room -/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ
  doorCount : ℕ
  doorArea : ℝ
  largeWindowCount : ℕ
  largeWindowArea : ℝ
  smallWindowCount : ℕ
  smallWindowArea : ℝ
  paintCostPerSqM : ℝ
  totalPaintCost : ℝ

/-- Calculates the paintable area of the room -/
def paintableArea (r : Room) : ℝ :=
  2 * (r.height * r.length + r.height * r.width) -
  (r.doorCount * r.doorArea + r.largeWindowCount * r.largeWindowArea + r.smallWindowCount * r.smallWindowArea)

/-- Theorem stating that the width of the room is 7 meters -/
theorem room_width_is_seven (r : Room) 
  (h1 : r.length = 10)
  (h2 : r.height = 5)
  (h3 : r.doorCount = 2)
  (h4 : r.doorArea = 3)
  (h5 : r.largeWindowCount = 1)
  (h6 : r.largeWindowArea = 3)
  (h7 : r.smallWindowCount = 2)
  (h8 : r.smallWindowArea = 1.5)
  (h9 : r.paintCostPerSqM = 3)
  (h10 : r.totalPaintCost = 474)
  (h11 : paintableArea r * r.paintCostPerSqM = r.totalPaintCost) :
  r.width = 7 := by
  sorry

end NUMINAMATH_CALUDE_room_width_is_seven_l1409_140984


namespace NUMINAMATH_CALUDE_product_divisible_by_sum_implies_inequality_l1409_140998

theorem product_divisible_by_sum_implies_inequality (m n : ℕ+) 
  (h : (m + n : ℕ) ∣ (m * n : ℕ)) : 
  (m : ℕ) + n ≤ n^2 := by
sorry

end NUMINAMATH_CALUDE_product_divisible_by_sum_implies_inequality_l1409_140998


namespace NUMINAMATH_CALUDE_smallest_b_quadratic_inequality_l1409_140936

theorem smallest_b_quadratic_inequality :
  let f : ℝ → ℝ := fun b => -3 * b^2 + 13 * b - 10
  ∃ b_min : ℝ, b_min = -2/3 ∧
    (∀ b : ℝ, f b ≥ 0 → b ≥ b_min) ∧
    f b_min ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_quadratic_inequality_l1409_140936


namespace NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l1409_140943

theorem no_real_solution_for_log_equation :
  ¬ ∃ x : ℝ, (Real.log (x + 5) + Real.log (2 * x - 2) = Real.log (2 * x^2 + x - 10)) ∧ 
  (x + 5 > 0) ∧ (2 * x - 2 > 0) ∧ (2 * x^2 + x - 10 > 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l1409_140943


namespace NUMINAMATH_CALUDE_picture_area_l1409_140989

/-- The area of a picture on a sheet of paper with given dimensions and margins. -/
theorem picture_area (paper_width paper_length margin : ℝ) 
  (hw : paper_width = 8.5)
  (hl : paper_length = 10)
  (hm : margin = 1.5) : 
  (paper_width - 2 * margin) * (paper_length - 2 * margin) = 38.5 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l1409_140989


namespace NUMINAMATH_CALUDE_parabola_focus_l1409_140903

/-- The parabola with equation y² = -8x has its focus at (-2, 0) -/
theorem parabola_focus (x y : ℝ) :
  y^2 = -8*x → (x + 2)^2 + y^2 = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l1409_140903


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_to_8_l1409_140939

theorem coefficient_x3y5_in_expansion_of_x_plus_y_to_8 :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k : ℕ) * (if k = 3 then 1 else 0)) = 56 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_of_x_plus_y_to_8_l1409_140939


namespace NUMINAMATH_CALUDE_tray_height_l1409_140933

theorem tray_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) : 
  side_length = 120 →
  cut_distance = 5 →
  cut_angle = 45 →
  ∃ (height : ℝ), height = 5 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_tray_height_l1409_140933


namespace NUMINAMATH_CALUDE_is_quadratic_equation_l1409_140970

theorem is_quadratic_equation (x : ℝ) : ∃ (a b c : ℝ), a ≠ 0 ∧ 3*(x-1)^2 = 2*(x-1) ↔ a*x^2 + b*x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_is_quadratic_equation_l1409_140970


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l1409_140972

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem book_arrangement_proof :
  let total_books : ℕ := 9
  let arabic_books : ℕ := 2
  let french_books : ℕ := 3
  let english_books : ℕ := 4
  let arabic_group : ℕ := 1
  let english_group : ℕ := 1
  let total_groups : ℕ := arabic_group + english_group + french_books

  (factorial total_groups) * (factorial arabic_books) * (factorial english_books) = 5760 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l1409_140972


namespace NUMINAMATH_CALUDE_unique_representation_of_nonnegative_integers_l1409_140909

theorem unique_representation_of_nonnegative_integers (n : ℕ) :
  ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_representation_of_nonnegative_integers_l1409_140909


namespace NUMINAMATH_CALUDE_reciprocal_of_opposite_l1409_140959

theorem reciprocal_of_opposite (x : ℝ) (h : x ≠ 0) : 
  (-(1 / x)) = 1 / (-x) :=
sorry

end NUMINAMATH_CALUDE_reciprocal_of_opposite_l1409_140959


namespace NUMINAMATH_CALUDE_polynomial_identity_l1409_140993

theorem polynomial_identity 
  (a b c x : ℝ) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  a^2 * ((x-b)*(x-c)) / ((a-b)*(a-c)) + 
  b^2 * ((x-c)*(x-a)) / ((b-c)*(b-a)) + 
  c^2 * ((x-a)*(x-b)) / ((c-a)*(c-b)) = x^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1409_140993


namespace NUMINAMATH_CALUDE_dog_catches_fox_dog_catches_fox_at_120m_l1409_140961

/-- The distance at which a dog catches a fox given their jump lengths and frequencies -/
theorem dog_catches_fox (initial_distance : ℝ) (dog_jump : ℝ) (fox_jump : ℝ) 
  (dog_jumps_per_unit : ℕ) (fox_jumps_per_unit : ℕ) : ℝ :=
  let dog_distance_per_unit := dog_jump * dog_jumps_per_unit
  let fox_distance_per_unit := fox_jump * fox_jumps_per_unit
  let net_gain_per_unit := dog_distance_per_unit - fox_distance_per_unit
  let time_units := initial_distance / net_gain_per_unit
  dog_distance_per_unit * time_units

/-- Proof that the dog catches the fox at 120 meters from the starting point -/
theorem dog_catches_fox_at_120m : 
  dog_catches_fox 30 2 1 2 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_dog_catches_fox_dog_catches_fox_at_120m_l1409_140961


namespace NUMINAMATH_CALUDE_range_of_a_l1409_140997

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x, x^2 + (a-1)*x + a^2 > 0

-- Define proposition q
def q (a : ℝ) : Prop := ∀ x y, x < y → (2*a^2 - a)^x < (2*a^2 - a)^y

-- Theorem statement
theorem range_of_a :
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∀ a : ℝ, (1/3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1/2) ↔ (p a ∨ q a) ∧ ¬(p a ∧ q a)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1409_140997


namespace NUMINAMATH_CALUDE_total_paintable_area_is_1200_l1409_140942

/-- Represents the dimensions of a bedroom --/
structure BedroomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total wall area of a bedroom --/
def totalWallArea (dim : BedroomDimensions) : ℝ :=
  2 * (dim.length * dim.height + dim.width * dim.height)

/-- Calculates the paintable wall area of a bedroom --/
def paintableWallArea (dim : BedroomDimensions) (nonPaintableArea : ℝ) : ℝ :=
  totalWallArea dim - nonPaintableArea

/-- The main theorem stating the total paintable area of all bedrooms --/
theorem total_paintable_area_is_1200 
  (bedroom1 : BedroomDimensions)
  (bedroom2 : BedroomDimensions)
  (bedroom3 : BedroomDimensions)
  (nonPaintable1 nonPaintable2 nonPaintable3 : ℝ) :
  bedroom1.length = 14 ∧ bedroom1.width = 11 ∧ bedroom1.height = 9 ∧
  bedroom2.length = 13 ∧ bedroom2.width = 12 ∧ bedroom2.height = 9 ∧
  bedroom3.length = 15 ∧ bedroom3.width = 10 ∧ bedroom3.height = 9 ∧
  nonPaintable1 = 50 ∧ nonPaintable2 = 55 ∧ nonPaintable3 = 45 →
  paintableWallArea bedroom1 nonPaintable1 + 
  paintableWallArea bedroom2 nonPaintable2 + 
  paintableWallArea bedroom3 nonPaintable3 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_is_1200_l1409_140942


namespace NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l1409_140911

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ -9/4}
def B : Set ℝ := {x | x < 0}

-- Define set difference
def setDifference (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}

-- Define symmetric difference
def symmetricDifference (M N : Set ℝ) : Set ℝ := 
  (setDifference M N) ∪ (setDifference N M)

-- Theorem statement
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {x | x ≥ 0 ∨ x < -9/4} := by
  sorry

end NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l1409_140911


namespace NUMINAMATH_CALUDE_percentage_increase_proof_l1409_140955

def original_earnings : ℚ := 60
def new_earnings : ℚ := 80

theorem percentage_increase_proof :
  (new_earnings - original_earnings) / original_earnings = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_proof_l1409_140955


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1409_140938

theorem system_solution_ratio :
  ∃ (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0),
    x + 10*y + 5*z = 0 ∧
    2*x + 5*y + 4*z = 0 ∧
    3*x + 6*y + 5*z = 0 ∧
    y*z / (x^2) = -3/49 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1409_140938


namespace NUMINAMATH_CALUDE_percentage_problem_l1409_140919

theorem percentage_problem (p : ℝ) (h1 : 0.25 * 820 = (p/100) * 1500 - 20) : p = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1409_140919


namespace NUMINAMATH_CALUDE_jenny_money_l1409_140967

theorem jenny_money (original : ℚ) : 
  (4/7 : ℚ) * original = 24 → (1/2 : ℚ) * original = 21 := by
sorry

end NUMINAMATH_CALUDE_jenny_money_l1409_140967


namespace NUMINAMATH_CALUDE_tank_insulation_problem_l1409_140905

theorem tank_insulation_problem (x : ℝ) : 
  x > 0 →  -- Ensure x is positive
  (14 * x + 20) * 20 = 1520 → 
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_tank_insulation_problem_l1409_140905


namespace NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_nine_l1409_140957

theorem difference_of_cubes_divisible_by_nine (a b : ℤ) :
  ∃ k : ℤ, (2*a + 1)^3 - (2*b + 1)^3 = 9*k :=
sorry

end NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_nine_l1409_140957


namespace NUMINAMATH_CALUDE_range_of_3a_minus_b_l1409_140945

theorem range_of_3a_minus_b (a b : ℝ) 
  (h1 : -1 < a + b ∧ a + b < 3) 
  (h2 : 2 < a - b ∧ a - b < 4) : 
  (∃ (x y : ℝ), (x = a ∧ y = b) ∧ 3*x - y = 3) ∧ 
  (∃ (x y : ℝ), (x = a ∧ y = b) ∧ 3*x - y = 11) ∧
  (∀ (x y : ℝ), (x = a ∧ y = b) → 3 ≤ 3*x - y ∧ 3*x - y ≤ 11) :=
sorry

end NUMINAMATH_CALUDE_range_of_3a_minus_b_l1409_140945


namespace NUMINAMATH_CALUDE_equation_equivalence_and_domain_x_domain_l1409_140948

-- Define the original equation
def original_equation (x y : ℝ) : Prop :=
  x = (2 * y + 1) / (y - 2)

-- Define the inverted equation
def inverted_equation (x y : ℝ) : Prop :=
  y = (2 * x + 1) / (x - 2)

-- Theorem stating the equivalence of the equations and the domain of x
theorem equation_equivalence_and_domain :
  ∀ x y : ℝ, original_equation x y ↔ (inverted_equation x y ∧ x ≠ 2) :=
by
  sorry

-- Theorem stating the domain of x
theorem x_domain : ∀ x : ℝ, (∃ y : ℝ, original_equation x y) → x ≠ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_and_domain_x_domain_l1409_140948


namespace NUMINAMATH_CALUDE_percentage_same_grade_is_42_5_l1409_140908

/-- The total number of students in the class -/
def total_students : ℕ := 40

/-- The number of students who received an 'A' on both tests -/
def same_grade_A : ℕ := 3

/-- The number of students who received a 'B' on both tests -/
def same_grade_B : ℕ := 5

/-- The number of students who received a 'C' on both tests -/
def same_grade_C : ℕ := 6

/-- The number of students who received a 'D' on both tests -/
def same_grade_D : ℕ := 2

/-- The number of students who received an 'E' on both tests -/
def same_grade_E : ℕ := 1

/-- The total number of students who received the same grade on both tests -/
def total_same_grade : ℕ := same_grade_A + same_grade_B + same_grade_C + same_grade_D + same_grade_E

/-- The percentage of students who received the same grade on both tests -/
def percentage_same_grade : ℚ := (total_same_grade : ℚ) / (total_students : ℚ) * 100

theorem percentage_same_grade_is_42_5 : percentage_same_grade = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_same_grade_is_42_5_l1409_140908


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1409_140912

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x + 2

theorem tangent_line_equation :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m : ℝ := (Real.cos x₀ + Real.exp x₀)
  ∀ x y : ℝ, y = m * (x - x₀) + y₀ ↔ y = 2 * x + 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1409_140912


namespace NUMINAMATH_CALUDE_tens_digit_of_19_power_1987_l1409_140971

theorem tens_digit_of_19_power_1987 : ∃ n : ℕ, 19^1987 ≡ 30 + n [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_19_power_1987_l1409_140971


namespace NUMINAMATH_CALUDE_twenty_bananas_equal_twelve_pears_l1409_140991

/-- The cost relationship between bananas, apples, and pears at Hector's Healthy Habits -/
structure FruitCosts where
  banana : ℚ
  apple : ℚ
  pear : ℚ
  banana_apple_ratio : 4 * banana = 3 * apple
  apple_pear_ratio : 5 * apple = 4 * pear

/-- Theorem stating that 20 bananas cost the same as 12 pears -/
theorem twenty_bananas_equal_twelve_pears (c : FruitCosts) : 20 * c.banana = 12 * c.pear := by
  sorry

end NUMINAMATH_CALUDE_twenty_bananas_equal_twelve_pears_l1409_140991


namespace NUMINAMATH_CALUDE_stating_four_of_a_kind_hands_l1409_140969

/-- Represents the number of distinct values in a standard deck of cards. -/
def num_values : ℕ := 13

/-- Represents the number of distinct suits in a standard deck of cards. -/
def num_suits : ℕ := 4

/-- Represents the total number of cards in a standard deck. -/
def total_cards : ℕ := num_values * num_suits

/-- Represents the number of cards in a hand. -/
def hand_size : ℕ := 5

/-- 
Theorem stating that the number of 5-card hands containing four cards of the same value 
in a standard 52-card deck is equal to 624.
-/
theorem four_of_a_kind_hands : 
  (num_values : ℕ) * (total_cards - num_suits : ℕ) = 624 := by
  sorry


end NUMINAMATH_CALUDE_stating_four_of_a_kind_hands_l1409_140969


namespace NUMINAMATH_CALUDE_amy_game_score_l1409_140979

theorem amy_game_score (points_per_treasure : ℕ) (treasures_level1 : ℕ) (treasures_level2 : ℕ) : 
  points_per_treasure = 4 →
  treasures_level1 = 6 →
  treasures_level2 = 2 →
  points_per_treasure * treasures_level1 + points_per_treasure * treasures_level2 = 32 := by
sorry

end NUMINAMATH_CALUDE_amy_game_score_l1409_140979


namespace NUMINAMATH_CALUDE_sequence_inequality_l1409_140940

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hn1 : a (n + 1) = 0) 
  (h_ineq : ∀ k : ℕ, k ≥ 1 → k ≤ n → a (k - 1) - 2 * a k + a (k + 1) ≤ 1) :
  ∀ k : ℕ, k ≤ n + 1 → a k ≤ k * (n + 1 - k) / 2 :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1409_140940


namespace NUMINAMATH_CALUDE_circle_power_theorem_l1409_140944

/-- The power of a point with respect to a circle -/
def power (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : ℝ :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 - radius^2

theorem circle_power_theorem (k : ℝ) (hk : k < 0) :
  (∃ p : ℝ × ℝ, power (0, 0) 1 p = k) ∧
  ¬(∀ k : ℝ, k < 0 → ∃ q : ℝ × ℝ, power (0, 0) 1 q = -k) :=
by sorry

end NUMINAMATH_CALUDE_circle_power_theorem_l1409_140944


namespace NUMINAMATH_CALUDE_mom_shirt_purchase_l1409_140935

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The number of packages Mom needs to buy -/
def packages_to_buy : ℚ := 11.83333333

/-- The total number of t-shirts Mom wants to buy -/
def total_shirts : ℕ := 71

theorem mom_shirt_purchase :
  ⌊(packages_to_buy * shirts_per_package : ℚ)⌋ = total_shirts := by
  sorry

end NUMINAMATH_CALUDE_mom_shirt_purchase_l1409_140935


namespace NUMINAMATH_CALUDE_trivia_team_selection_l1409_140917

theorem trivia_team_selection (total_students : ℕ) (num_groups : ℕ) (students_per_group : ℕ) :
  total_students = 17 →
  num_groups = 3 →
  students_per_group = 4 →
  total_students - (num_groups * students_per_group) = 5 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_selection_l1409_140917


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_positive_n_value_l1409_140988

theorem quadratic_equation_unique_solution (n : ℝ) : 
  (∃! x : ℝ, 5 * x^2 + n * x + 45 = 0) → n = 30 ∨ n = -30 :=
by sorry

theorem positive_n_value (n : ℝ) : 
  (∃! x : ℝ, 5 * x^2 + n * x + 45 = 0) → n > 0 → n = 30 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_positive_n_value_l1409_140988


namespace NUMINAMATH_CALUDE_min_magnitude_in_A_l1409_140937

def a : Fin 3 → ℝ := ![1, 2, 3]
def b : Fin 3 → ℝ := ![1, -1, 1]

def A : Set (Fin 3 → ℝ) :=
  {x | ∃ k : ℤ, x = fun i => a i + k * b i}

theorem min_magnitude_in_A :
  ∃ x ∈ A, ∀ y ∈ A, ‖x‖ ≤ ‖y‖ ∧ ‖x‖ = Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_min_magnitude_in_A_l1409_140937


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l1409_140999

-- Define the function f
def f (a b c d e : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e

-- State the theorem
theorem polynomial_value_theorem (a b c d e : ℝ) :
  f a b c d e (-1) = 2 → 16 * a - 8 * b + 4 * c - 2 * d + e = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l1409_140999


namespace NUMINAMATH_CALUDE_circle_radius_l1409_140978

theorem circle_radius (x y : ℝ) :
  x > 0 ∧ y > 0 ∧ 
  (∃ r : ℝ, r > 0 ∧ x = π * r^2 ∧ y = 2 * π * r) ∧
  x + y = 72 * π →
  ∃ r : ℝ, r = 6 ∧ x = π * r^2 ∧ y = 2 * π * r :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l1409_140978
