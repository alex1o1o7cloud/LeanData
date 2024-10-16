import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_15th_set_l3749_374940

/-- The first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ :=
  1 + (n * (n - 1)) / 2

/-- The number of elements in the nth set -/
def set_size (n : ℕ) : ℕ := n

/-- The sum of elements in the nth set -/
def S (n : ℕ) : ℕ :=
  let a := first_element n
  let l := a + set_size n - 1
  (set_size n * (a + l)) / 2

/-- Theorem: The sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l3749_374940


namespace NUMINAMATH_CALUDE_melanie_dimes_l3749_374963

theorem melanie_dimes (initial : ℕ) (from_dad : ℕ) (total : ℕ) (from_mom : ℕ) : 
  initial = 7 → from_dad = 8 → total = 19 → from_mom = total - (initial + from_dad) → from_mom = 4 := by sorry

end NUMINAMATH_CALUDE_melanie_dimes_l3749_374963


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l3749_374901

theorem hexagon_angle_measure :
  ∀ (a b c d e : ℝ),
    a = 135 ∧ b = 150 ∧ c = 120 ∧ d = 130 ∧ e = 100 →
    ∃ (q : ℝ),
      q = 85 ∧
      a + b + c + d + e + q = 720 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l3749_374901


namespace NUMINAMATH_CALUDE_product_first_three_terms_l3749_374900

/-- An arithmetic sequence with given properties -/
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  (a 8 = 20) ∧ (∀ n : ℕ, a (n + 1) = a n + 2)

/-- Theorem stating the product of the first three terms -/
theorem product_first_three_terms (a : ℕ → ℕ) (h : ArithmeticSequence a) :
  a 1 * a 2 * a 3 = 480 := by
  sorry


end NUMINAMATH_CALUDE_product_first_three_terms_l3749_374900


namespace NUMINAMATH_CALUDE_parabola_area_theorem_l3749_374914

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given a parabola y^2 = 2px (p > 0) with a point A(m,1) on it,
    and a point B on the directrix such that AB is perpendicular to the directrix,
    if the area of triangle AOB (where O is the origin) is 1/2, then p = 1. -/
theorem parabola_area_theorem (C : Parabola) (A : Point) (m : ℝ) :
  A.x = m →
  A.y = 1 →
  A.y^2 = 2 * C.p * A.x →
  (∃ B : Point, B.y = -C.p/2 ∧ (A.x - B.x) * (A.y - B.y) = 0) →
  (1/2 * m * 1 + 1/2 * (C.p/2) * 1 = 1/2) →
  C.p = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_area_theorem_l3749_374914


namespace NUMINAMATH_CALUDE_common_factor_is_gcf_l3749_374996

-- Define the expression
def expression (a b c : ℤ) : ℤ := 8 * a^3 * b^2 - 12 * a * b^3 * c + 2 * a * b

-- Define the common factor
def common_factor (a b : ℤ) : ℤ := 2 * a * b

-- Theorem statement
theorem common_factor_is_gcf (a b c : ℤ) :
  (∃ k₁ k₂ k₃ : ℤ, 
    expression a b c = common_factor a b * (k₁ + k₂ + k₃) ∧
    k₁ = 4 * a^2 * b ∧
    k₂ = -6 * b^2 * c ∧
    k₃ = 1) ∧
  (∀ d : ℤ, d ∣ expression a b c → d ∣ common_factor a b ∨ d = 1 ∨ d = -1) :=
sorry

end NUMINAMATH_CALUDE_common_factor_is_gcf_l3749_374996


namespace NUMINAMATH_CALUDE_simplify_expression_l3749_374976

theorem simplify_expression (x y : ℝ) (h : y ≠ 0) :
  let P := x^2 + y^2
  let Q := x^2 - y^2
  ((P + 3*Q) / (P - Q)) - ((P - 3*Q) / (P + Q)) = (2*x^4 - y^4) / (x^2 * y^2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3749_374976


namespace NUMINAMATH_CALUDE_smallest_ellipse_area_l3749_374920

/-- The smallest area of an ellipse containing two specific circles -/
theorem smallest_ellipse_area (p q : ℝ) (h_ellipse : ∀ (x y : ℝ), x^2 / p^2 + y^2 / q^2 = 1 → 
  ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) :
  ∃ (m : ℝ), m = 3 * Real.sqrt 3 / 2 ∧ 
    ∀ (p' q' : ℝ), (∀ (x y : ℝ), x^2 / p'^2 + y^2 / q'^2 = 1 → 
      ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) → 
    p' * q' * Real.pi ≥ m * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_ellipse_area_l3749_374920


namespace NUMINAMATH_CALUDE_angle_sum_from_tangents_l3749_374952

theorem angle_sum_from_tangents (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.tan α = 2 → 
  Real.tan β = 3 → 
  α + β = 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_from_tangents_l3749_374952


namespace NUMINAMATH_CALUDE_star_property_l3749_374932

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element
  | five : Element

-- Define the operation *
def star : Element → Element → Element
  | Element.one, Element.one => Element.one
  | Element.one, Element.two => Element.two
  | Element.one, Element.three => Element.three
  | Element.one, Element.four => Element.four
  | Element.one, Element.five => Element.five
  | Element.two, Element.one => Element.two
  | Element.two, Element.two => Element.one
  | Element.two, Element.three => Element.five
  | Element.two, Element.four => Element.three
  | Element.two, Element.five => Element.four
  | Element.three, Element.one => Element.three
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.two
  | Element.three, Element.four => Element.five
  | Element.three, Element.five => Element.one
  | Element.four, Element.one => Element.four
  | Element.four, Element.two => Element.five
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.two
  | Element.four, Element.five => Element.three
  | Element.five, Element.one => Element.five
  | Element.five, Element.two => Element.three
  | Element.five, Element.three => Element.four
  | Element.five, Element.four => Element.one
  | Element.five, Element.five => Element.two

theorem star_property : 
  star (star Element.three Element.five) (star Element.two Element.four) = Element.three := by
  sorry

end NUMINAMATH_CALUDE_star_property_l3749_374932


namespace NUMINAMATH_CALUDE_symmetric_point_l3749_374974

/-- Given two points A and B in a plane, find the symmetric point A' of A with respect to B. -/
theorem symmetric_point (A B : ℝ × ℝ) (A' : ℝ × ℝ) : 
  A = (2, 1) → B = (-3, 7) → 
  (B.1 = (A.1 + A'.1) / 2 ∧ B.2 = (A.2 + A'.2) / 2) →
  A' = (-8, 13) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_l3749_374974


namespace NUMINAMATH_CALUDE_unique_solution_for_abc_l3749_374941

theorem unique_solution_for_abc (a b c : ℝ) 
  (ha : a > 2) (hb : b > 2) (hc : c > 2)
  (heq : (a - 1)^2 / (b + c + 1) + (b + 1)^2 / (c + a - 1) + (c + 5)^2 / (a + b - 5) = 49) :
  a = 10.5 ∧ b = 10 ∧ c = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_abc_l3749_374941


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l3749_374975

theorem infinitely_many_solutions (b : ℝ) : 
  (∀ x : ℝ, 3 * (5 + b * x) = 18 * x + 15) → b = 6 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l3749_374975


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_half_l3749_374937

theorem cos_sin_sum_equals_half : 
  Real.cos (263 * π / 180) * Real.cos (203 * π / 180) + 
  Real.sin (83 * π / 180) * Real.sin (23 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_half_l3749_374937


namespace NUMINAMATH_CALUDE_fred_took_233_marbles_l3749_374942

/-- The number of black marbles Fred took from Sara -/
def marbles_taken (initial_black_marbles remaining_black_marbles : ℕ) : ℕ :=
  initial_black_marbles - remaining_black_marbles

/-- Proof that Fred took 233 black marbles from Sara -/
theorem fred_took_233_marbles :
  marbles_taken 792 559 = 233 := by
  sorry

end NUMINAMATH_CALUDE_fred_took_233_marbles_l3749_374942


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3749_374951

theorem trigonometric_identity (α : ℝ) : 
  Real.sin (10 * α) * Real.sin (8 * α) + Real.sin (8 * α) * Real.sin (6 * α) - Real.sin (4 * α) * Real.sin (2 * α) = 
  2 * Real.cos (2 * α) * Real.sin (6 * α) * Real.sin (10 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3749_374951


namespace NUMINAMATH_CALUDE_complex_point_location_l3749_374987

theorem complex_point_location (a b : ℝ) (i : ℂ) (h : i * i = -1) 
  (eq : a + i = (b + i) * (2 - i)) : 
  a > 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_location_l3749_374987


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3749_374971

def U : Set ℕ := {x | x > 0 ∧ x < 9}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {7, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3749_374971


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l3749_374981

theorem quadratic_equation_equivalence :
  ∃ (r : ℝ), ∀ (x : ℝ), (4 * x^2 - 8 * x - 288 = 0) ↔ ((x + r)^2 = 73) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l3749_374981


namespace NUMINAMATH_CALUDE_matrix_equation_l3749_374965

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 12, 4]
def N : Matrix (Fin 2) (Fin 2) ℚ := !![46/7, -58/7; -26/7, 34/7]

theorem matrix_equation : N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_l3749_374965


namespace NUMINAMATH_CALUDE_monotonicity_intervals_no_increasing_intervals_l3749_374946

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - 2*a*x + a^2

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1/x + 2*x - 2*a

theorem monotonicity_intervals (x : ℝ) (h : x > 0) :
  let a := 2
  (f_deriv a x > 0 ↔ (x < (2 - Real.sqrt 2) / 2 ∨ x > (2 + Real.sqrt 2) / 2)) ∧
  (f_deriv a x < 0 ↔ ((2 - Real.sqrt 2) / 2 < x ∧ x < (2 + Real.sqrt 2) / 2)) :=
sorry

theorem no_increasing_intervals (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f_deriv a x ≤ 0) ↔ a ≥ 19/6 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_intervals_no_increasing_intervals_l3749_374946


namespace NUMINAMATH_CALUDE_complex_power_to_rectangular_l3749_374945

theorem complex_power_to_rectangular : 
  (3 * Complex.cos (Real.pi / 6) + 3 * Complex.I * Complex.sin (Real.pi / 6)) ^ 4 = 
  Complex.mk (-40.5) (40.5 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_to_rectangular_l3749_374945


namespace NUMINAMATH_CALUDE_exists_1992_gon_l3749_374991

/-- A convex polygon with n sides that is circumscribable about a circle -/
structure CircumscribablePolygon (n : ℕ) where
  sides : Fin n → ℝ
  convex : sorry
  circumscribable : sorry

/-- The condition that the side lengths are 1, 2, 3, ..., n in some order -/
def valid_side_lengths (n : ℕ) (p : CircumscribablePolygon n) : Prop :=
  ∃ (σ : Equiv (Fin n) (Fin n)), ∀ i, p.sides i = (σ i).val + 1

/-- The main theorem stating the existence of a 1992-sided circumscribable polygon
    with side lengths 1, 2, 3, ..., 1992 in some order -/
theorem exists_1992_gon :
  ∃ (p : CircumscribablePolygon 1992), valid_side_lengths 1992 p :=
sorry

end NUMINAMATH_CALUDE_exists_1992_gon_l3749_374991


namespace NUMINAMATH_CALUDE_representation_of_1917_l3749_374917

theorem representation_of_1917 : ∃ (a b c : ℤ), 1917 = a^2 - b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_representation_of_1917_l3749_374917


namespace NUMINAMATH_CALUDE_solve_inequality_for_x_find_k_range_l3749_374948

-- Part 1
theorem solve_inequality_for_x (x : ℝ) :
  (|1 - x * 2| > |x - 2|) ↔ (x < -1 ∨ x > 1) := by sorry

-- Part 2
theorem find_k_range (k : ℝ) :
  (∀ x y : ℝ, |x| < 1 → |y| < 1 → |1 - k*x*y| > |k*x - y|) ↔ 
  (k ≥ -1 ∧ k ≤ 1) := by sorry

end NUMINAMATH_CALUDE_solve_inequality_for_x_find_k_range_l3749_374948


namespace NUMINAMATH_CALUDE_range_of_m_l3749_374982

-- Define the propositions p and q
def p (m : ℝ) : Prop := 4 < m ∧ m < 10

def q (m : ℝ) : Prop := 8 < m ∧ m < 12

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ (4 < m ∧ m ≤ 8) ∨ (10 ≤ m ∧ m < 12) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3749_374982


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l3749_374995

theorem unique_modular_congruence :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 8] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l3749_374995


namespace NUMINAMATH_CALUDE_square_side_length_l3749_374962

theorem square_side_length (area : ℚ) (side : ℚ) (h1 : area = 9/16) (h2 : side * side = area) : side = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3749_374962


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l3749_374934

theorem at_least_one_greater_than_one (a b : ℝ) :
  (a + b > 2 → max a b > 1) ∧ (a * b > 1 → max a b > 1) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l3749_374934


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3749_374913

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r ∧ a n > 0

theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  (a 1) * (a 19) = 16 →
  (a 1) + (a 19) = 10 →
  (a 8) * (a 10) * (a 12) = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3749_374913


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3749_374985

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a < 0 ∧ b < 0 → a + b < 0) ∧
  ∃ (x y : ℝ), x + y < 0 ∧ ¬(x < 0 ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3749_374985


namespace NUMINAMATH_CALUDE_tan_seven_pi_fourth_l3749_374957

theorem tan_seven_pi_fourth : Real.tan (7 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_fourth_l3749_374957


namespace NUMINAMATH_CALUDE_custom_op_result_l3749_374923

def custom_op (a b : ℚ) : ℚ := (a + b) / (a - b)

theorem custom_op_result : custom_op (custom_op 8 6) 12 = -19/5 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l3749_374923


namespace NUMINAMATH_CALUDE_charity_race_fundraising_l3749_374930

/-- Proves that the amount raised by each of the ten students is $20 -/
theorem charity_race_fundraising
  (total_students : ℕ)
  (special_students : ℕ)
  (regular_amount : ℕ)
  (total_raised : ℕ)
  (h1 : total_students = 30)
  (h2 : special_students = 10)
  (h3 : regular_amount = 30)
  (h4 : total_raised = 800)
  (h5 : total_raised = special_students * X + (total_students - special_students) * regular_amount)
  : X = 20 := by
  sorry

end NUMINAMATH_CALUDE_charity_race_fundraising_l3749_374930


namespace NUMINAMATH_CALUDE_shorts_price_is_6_l3749_374977

/-- The price of a single jacket in dollars -/
def jacket_price : ℕ := 10

/-- The number of jackets bought -/
def num_jackets : ℕ := 3

/-- The price of a single pair of pants in dollars -/
def pants_price : ℕ := 12

/-- The number of pairs of pants bought -/
def num_pants : ℕ := 4

/-- The number of pairs of shorts bought -/
def num_shorts : ℕ := 2

/-- The total amount spent in dollars -/
def total_spent : ℕ := 90

theorem shorts_price_is_6 :
  ∃ (shorts_price : ℕ),
    shorts_price * num_shorts + 
    jacket_price * num_jackets + 
    pants_price * num_pants = total_spent ∧
    shorts_price = 6 :=
by sorry

end NUMINAMATH_CALUDE_shorts_price_is_6_l3749_374977


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3749_374908

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 1 + a 2 + a 3 = 6) →
  (a 7 + a 8 + a 9 = 24) →
  (a 4 + a 5 + a 6 = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3749_374908


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3749_374978

/-- The value of 'a' for a circle with equation x^2 + y^2 - 2ax + 2y + 1 = 0,
    where the line y = -x + 1 passes through its center. -/
theorem circle_line_intersection (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 - 2*a*x + 2*y + 1 = 0 ∧ 
               y = -x + 1 ∧ 
               ∀ x' y' : ℝ, x'^2 + y'^2 - 2*a*x' + 2*y' + 1 = 0 → 
                 (x - x')^2 + (y - y')^2 ≤ (x' - a)^2 + (y' + 1)^2) → 
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3749_374978


namespace NUMINAMATH_CALUDE_transistor_count_2002_and_2010_l3749_374926

/-- Moore's law: number of transistors doubles every two years -/
def moores_law (initial : ℕ) (years : ℕ) : ℕ :=
  initial * (2 ^ (years / 2))

/-- The year when the initial count was recorded -/
def initial_year : ℕ := 1992

/-- The initial number of transistors -/
def initial_transistors : ℕ := 2000000

theorem transistor_count_2002_and_2010 :
  (moores_law initial_transistors (2002 - initial_year) = 64000000) ∧
  (moores_law initial_transistors (2010 - initial_year) = 1024000000) := by
  sorry

end NUMINAMATH_CALUDE_transistor_count_2002_and_2010_l3749_374926


namespace NUMINAMATH_CALUDE_chess_piece_paths_l3749_374969

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem chess_piece_paths :
  let num_segments : ℕ := 15
  let steps_per_segment : ℕ := 6
  let ways_per_segment : ℕ := fibonacci (steps_per_segment + 1)
  num_segments * ways_per_segment = 195 :=
by sorry

end NUMINAMATH_CALUDE_chess_piece_paths_l3749_374969


namespace NUMINAMATH_CALUDE_inverse_exists_mod_prime_wilsons_theorem_l3749_374904

-- Define primality
def isPrime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

-- Part 1: Inverse exists for non-zero elements modulo prime
theorem inverse_exists_mod_prime (p k : ℕ) (hp : isPrime p) (hk : ¬(p ∣ k)) :
  ∃ l : ℕ, k * l ≡ 1 [ZMOD p] :=
sorry

-- Part 2: Wilson's theorem
theorem wilsons_theorem (n : ℕ) :
  isPrime n ↔ (Nat.factorial (n - 1)) ≡ -1 [ZMOD n] :=
sorry

end NUMINAMATH_CALUDE_inverse_exists_mod_prime_wilsons_theorem_l3749_374904


namespace NUMINAMATH_CALUDE_book_pages_from_digits_l3749_374919

/-- Given a book with pages numbered consecutively starting from 1,
    this function calculates the total number of digits used to number all pages. -/
def totalDigits (n : ℕ) : ℕ :=
  let oneDigit := min n 9
  let twoDigit := max 0 (min n 99 - 9)
  let threeDigit := max 0 (n - 99)
  oneDigit + 2 * twoDigit + 3 * threeDigit

/-- Theorem stating that a book with 672 digits used for page numbering has 260 pages. -/
theorem book_pages_from_digits :
  ∃ (n : ℕ), totalDigits n = 672 ∧ n = 260 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_from_digits_l3749_374919


namespace NUMINAMATH_CALUDE_min_degree_of_g_l3749_374955

variable (x : ℝ)
variable (f g h : ℝ → ℝ)

def is_polynomial (p : ℝ → ℝ) : Prop := sorry

def degree (p : ℝ → ℝ) : ℕ := sorry

theorem min_degree_of_g 
  (hpoly : is_polynomial f ∧ is_polynomial g ∧ is_polynomial h)
  (heq : ∀ x, 2 * f x + 5 * g x = h x)
  (hf : degree f = 7)
  (hh : degree h = 10) :
  degree g ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_min_degree_of_g_l3749_374955


namespace NUMINAMATH_CALUDE_student_selection_methods_l3749_374950

theorem student_selection_methods (first_year second_year third_year : ℕ) 
  (h1 : first_year = 3) 
  (h2 : second_year = 5) 
  (h3 : third_year = 4) : 
  first_year + second_year + third_year = 12 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_methods_l3749_374950


namespace NUMINAMATH_CALUDE_total_splash_width_l3749_374970

/-- Represents the splash width of different rock types -/
def splash_width (rock_type : String) : ℚ :=
  match rock_type with
  | "pebble" => 1/4
  | "rock" => 1/2
  | "boulder" => 2
  | "mini-boulder" => 1
  | "large_pebble" => 1/3
  | _ => 0

/-- Calculates the total splash width for a given rock type and count -/
def total_splash (rock_type : String) (count : ℕ) : ℚ :=
  (splash_width rock_type) * count

/-- Theorem: The total width of splashes is 14 meters -/
theorem total_splash_width :
  (total_splash "pebble" 8) +
  (total_splash "rock" 4) +
  (total_splash "boulder" 3) +
  (total_splash "mini-boulder" 2) +
  (total_splash "large_pebble" 6) = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_splash_width_l3749_374970


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l3749_374922

theorem complex_magnitude_product : Complex.abs ((7 - 24 * Complex.I) * (3 + 4 * Complex.I)) = 125 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l3749_374922


namespace NUMINAMATH_CALUDE_solution_theorem_l3749_374921

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * a * x + a^2 - 3

theorem solution_theorem :
  -- Part I
  (∃ (l b : ℝ), ∀ x, f 3 x < 0 ↔ l < x ∧ x < b) ∧
  (∃ (l : ℝ), ∀ x, f 3 x < 0 ↔ l < x ∧ x < 2) ∧
  -- Part II
  (∀ a : ℝ, a < 0 → (∀ x, -3 ≤ x ∧ x ≤ 3 → f a x < 4) → -7/4 < a) ∧
  (∀ a : ℝ, a < 0 → (∀ x, -3 ≤ x ∧ x ≤ 3 → f a x < 4) → a < 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_theorem_l3749_374921


namespace NUMINAMATH_CALUDE_expression_value_l3749_374961

theorem expression_value : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3749_374961


namespace NUMINAMATH_CALUDE_impossibility_of_swapping_all_stones_l3749_374918

/-- Represents a Go board -/
structure GoBoard where
  size : Nat
  total_stones : Nat

/-- Represents the state of stones on a Go board -/
inductive StoneState
  | Black
  | White

/-- Represents a stone on the Go board -/
structure Stone where
  position : Nat × Nat
  color : StoneState

/-- Definition of a valid Go board configuration -/
def is_valid_board (board : GoBoard) (stones : List Stone) : Prop :=
  board.size = 19 ∧
  board.total_stones = board.size * board.size ∧
  stones.length = board.total_stones ∧
  ∀ (i j : Nat), i < board.size ∧ j < board.size →
    ∃ (s : Stone), s ∈ stones ∧ s.position = (i, j) ∧
      (∀ (n : Stone), n ∈ stones ∧ n.position ∈ [(i-1, j), (i+1, j), (i, j-1), (i, j+1)] →
        n.color ≠ s.color)

/-- Theorem stating the impossibility of swapping all stone positions -/
theorem impossibility_of_swapping_all_stones (board : GoBoard) (stones : List Stone) :
  is_valid_board board stones →
  ¬∃ (new_stones : List Stone),
    is_valid_board board new_stones ∧
    (∀ (s : Stone), s ∈ stones →
      ∃ (n : Stone), n ∈ new_stones ∧ n.position = s.position ∧ n.color ≠ s.color) :=
by sorry

end NUMINAMATH_CALUDE_impossibility_of_swapping_all_stones_l3749_374918


namespace NUMINAMATH_CALUDE_square_root_range_l3749_374902

theorem square_root_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 4) → x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_square_root_range_l3749_374902


namespace NUMINAMATH_CALUDE_length_of_projected_segment_l3749_374986

/-- Given two points A and B on the y-axis, and their respective projections A' and B' on the line y = x,
    with AA' and BB' intersecting at point C, prove that the length of A'B' is 2.5√2. -/
theorem length_of_projected_segment (A B A' B' C : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 15) →
  C = (3, 9) →
  (A'.1 = A'.2) →
  (B'.1 = B'.2) →
  (∃ t : ℝ, A + t • (C - A) = A') →
  (∃ s : ℝ, B + s • (C - B) = B') →
  ‖A' - B'‖ = 2.5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_length_of_projected_segment_l3749_374986


namespace NUMINAMATH_CALUDE_parabola_theorem_l3749_374931

-- Define a parabola type
structure Parabola where
  equation : ℝ → ℝ → Prop
  directrix : ℝ → ℝ → Prop

-- Define the conditions for the parabola
def parabola_conditions (p : Parabola) : Prop :=
  -- Vertex at origin
  p.equation 0 0 ∧
  -- Passes through (-3, 2)
  p.equation (-3) 2 ∧
  -- Axis of symmetry along coordinate axis (implied by the equation forms)
  (∃ (a : ℝ), ∀ (x y : ℝ), p.equation x y ↔ y^2 = a * x) ∨
  (∃ (b : ℝ), ∀ (x y : ℝ), p.equation x y ↔ x^2 = b * y)

-- Define the possible equations and directrices
def parabola1 : Parabola :=
  { equation := λ x y => y^2 = -4/3 * x
    directrix := λ x y => x = 1/3 }

def parabola2 : Parabola :=
  { equation := λ x y => x^2 = 9/2 * y
    directrix := λ x y => y = -9/8 }

-- Theorem statement
theorem parabola_theorem :
  ∀ (p : Parabola), parabola_conditions p →
    (p = parabola1 ∨ p = parabola2) :=
sorry

end NUMINAMATH_CALUDE_parabola_theorem_l3749_374931


namespace NUMINAMATH_CALUDE_box_2_neg2_3_l3749_374968

/-- Definition of the box operation for integers -/
def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

/-- Theorem stating that the box operation applied to 2, -2, and 3 equals 69/4 -/
theorem box_2_neg2_3 : box 2 (-2) 3 = 69/4 := by sorry

end NUMINAMATH_CALUDE_box_2_neg2_3_l3749_374968


namespace NUMINAMATH_CALUDE_existence_of_separated_points_l3749_374967

/-- A type representing a segment in a plane -/
structure Segment where
  -- Add necessary fields

/-- A type representing a point in a plane -/
structure Point where
  -- Add necessary fields

/-- Checks if two segments are parallel -/
def are_parallel (s1 s2 : Segment) : Prop :=
  sorry

/-- Checks if two segments intersect -/
def intersect (s1 s2 : Segment) : Prop :=
  sorry

/-- Checks if a segment separates two points -/
def separates (s : Segment) (p1 p2 : Point) : Prop :=
  sorry

/-- Main theorem -/
theorem existence_of_separated_points (n : ℕ) (segments : Fin (n^2) → Segment)
  (h1 : ∀ i j, i ≠ j → ¬(are_parallel (segments i) (segments j)))
  (h2 : ∀ i j, i ≠ j → ¬(intersect (segments i) (segments j))) :
  ∃ (points : Fin n → Point),
    ∀ i j, i ≠ j → ∃ k, separates (segments k) (points i) (points j) :=
sorry

end NUMINAMATH_CALUDE_existence_of_separated_points_l3749_374967


namespace NUMINAMATH_CALUDE_eva_apple_count_l3749_374938

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks Eva needs to eat an apple -/
def weeks : ℕ := 2

/-- The number of apples Eva should buy -/
def apples_to_buy : ℕ := days_per_week * weeks

theorem eva_apple_count : apples_to_buy = 14 := by
  sorry

end NUMINAMATH_CALUDE_eva_apple_count_l3749_374938


namespace NUMINAMATH_CALUDE_remainder_783245_div_7_l3749_374912

theorem remainder_783245_div_7 : 783245 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_783245_div_7_l3749_374912


namespace NUMINAMATH_CALUDE_tenth_meeting_position_l3749_374905

/-- Represents a robot on a circular track -/
structure Robot where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Represents the state of the system -/
structure State where
  robotA : Robot
  robotB : Robot
  position : ℝ  -- Position on the track (0 ≤ position < 8)
  meetings : ℕ

/-- Updates the state after a meeting -/
def updateState (s : State) : State :=
  { s with
    robotB := { s.robotB with direction := !s.robotB.direction }
    meetings := s.meetings + 1
  }

/-- Simulates the movement of robots until they meet 10 times -/
def simulate (initialState : State) : ℝ :=
  sorry

theorem tenth_meeting_position (initialA initialB : Robot) :
  let initialState : State :=
    { robotA := initialA
      robotB := initialB
      position := 0
      meetings := 0
    }
  simulate initialState = 0 :=
sorry

end NUMINAMATH_CALUDE_tenth_meeting_position_l3749_374905


namespace NUMINAMATH_CALUDE_min_cars_in_group_l3749_374907

theorem min_cars_in_group (total : ℕ) 
  (no_ac : ℕ) 
  (racing_stripes : ℕ) 
  (ac_no_stripes : ℕ) : 
  no_ac = 47 →
  racing_stripes ≥ 53 →
  ac_no_stripes ≤ 47 →
  total ≥ 100 :=
by
  sorry

end NUMINAMATH_CALUDE_min_cars_in_group_l3749_374907


namespace NUMINAMATH_CALUDE_melanie_trout_l3749_374973

def melanie_catch : ℕ → ℕ → Prop
| m, t => t = 2 * m

theorem melanie_trout (tom_catch : ℕ) (h : melanie_catch 8 tom_catch) (h2 : tom_catch = 16) : 
  8 = 8 := by sorry

end NUMINAMATH_CALUDE_melanie_trout_l3749_374973


namespace NUMINAMATH_CALUDE_triangle_side_decomposition_l3749_374915

/-- Given a triangle with side lengths a, b, and c, there exist positive numbers x, y, and z
    such that a = y + z, b = x + z, and c = x + y -/
theorem triangle_side_decomposition (a b c : ℝ) (h_triangle : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ a = y + z ∧ b = x + z ∧ c = x + y :=
sorry

end NUMINAMATH_CALUDE_triangle_side_decomposition_l3749_374915


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3749_374983

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃! (a : ℝ), i * (1 + a * i) = 2 + i :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3749_374983


namespace NUMINAMATH_CALUDE_professors_arrangement_count_l3749_374966

/-- The number of ways to arrange professors among students. -/
def arrange_professors (num_students : ℕ) (num_professors : ℕ) : ℕ :=
  Nat.descFactorial (num_students - 1) num_professors

/-- Theorem stating that arranging 3 professors among 6 students results in 60 possibilities. -/
theorem professors_arrangement_count :
  arrange_professors 6 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_professors_arrangement_count_l3749_374966


namespace NUMINAMATH_CALUDE_toms_age_ratio_l3749_374956

theorem toms_age_ratio (T M : ℚ) : 
  (∃ (children_sum : ℚ), 
    children_sum = T ∧ 
    T - M = 3 * (children_sum - 4 * M)) → 
  T / M = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l3749_374956


namespace NUMINAMATH_CALUDE_m_eq_one_sufficient_not_necessary_l3749_374928

-- Define the lines l1 and l2 as functions of x and y
def l1 (m : ℝ) (x y : ℝ) : Prop := m * x + y + 3 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := (3 * m - 2) * x + m * y + 2 = 0

-- Define parallel lines
def parallel (m : ℝ) : Prop := ∀ x y, l1 m x y ↔ l2 m x y

-- Theorem statement
theorem m_eq_one_sufficient_not_necessary :
  (∀ m : ℝ, m = 1 → parallel m) ∧ 
  (∃ m : ℝ, m ≠ 1 ∧ parallel m) :=
sorry

end NUMINAMATH_CALUDE_m_eq_one_sufficient_not_necessary_l3749_374928


namespace NUMINAMATH_CALUDE_birds_flew_away_l3749_374947

theorem birds_flew_away (original : ℝ) (remaining : ℝ) (flew_away : ℝ) : 
  original = 21.0 → remaining = 7 → flew_away = original - remaining → flew_away = 14.0 := by
  sorry

end NUMINAMATH_CALUDE_birds_flew_away_l3749_374947


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l3749_374944

theorem smallest_integer_with_given_remainders : ∃ M : ℕ,
  (M > 0) ∧
  (M % 4 = 3) ∧
  (M % 5 = 4) ∧
  (M % 6 = 5) ∧
  (M % 7 = 6) ∧
  (M % 8 = 7) ∧
  (M % 9 = 8) ∧
  (∀ n : ℕ, n > 0 ∧
    n % 4 = 3 ∧
    n % 5 = 4 ∧
    n % 6 = 5 ∧
    n % 7 = 6 ∧
    n % 8 = 7 ∧
    n % 9 = 8 → n ≥ M) ∧
  M = 2519 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l3749_374944


namespace NUMINAMATH_CALUDE_cube_volume_l3749_374972

theorem cube_volume (s : ℝ) (h : s * s = 64) : s * s * s = 512 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l3749_374972


namespace NUMINAMATH_CALUDE_A_simplified_A_value_when_x_plus_one_squared_is_six_l3749_374939

-- Define the polynomial A
def A (x : ℝ) : ℝ := (x + 2)^2 + (1 - x) * (2 + x) - 3

-- Theorem for the simplified form of A
theorem A_simplified (x : ℝ) : A x = 3 * x + 3 := by sorry

-- Theorem for the value of A when (x+1)^2 = 6
theorem A_value_when_x_plus_one_squared_is_six :
  ∃ x : ℝ, (x + 1)^2 = 6 ∧ (A x = 3 * Real.sqrt 6 ∨ A x = -3 * Real.sqrt 6) := by sorry

end NUMINAMATH_CALUDE_A_simplified_A_value_when_x_plus_one_squared_is_six_l3749_374939


namespace NUMINAMATH_CALUDE_total_marigolds_sold_l3749_374984

/-- The number of marigolds sold during a three-day sale -/
def marigolds_sold (day1 day2 day3 : ℕ) : ℕ := day1 + day2 + day3

/-- Theorem stating the total number of marigolds sold during the sale -/
theorem total_marigolds_sold :
  let day1 := 14
  let day2 := 25
  let day3 := 2 * day2
  marigolds_sold day1 day2 day3 = 89 := by
  sorry

end NUMINAMATH_CALUDE_total_marigolds_sold_l3749_374984


namespace NUMINAMATH_CALUDE_inequality_range_l3749_374964

theorem inequality_range (a : ℝ) : 
  (∀ (x θ : ℝ), θ ∈ Set.Icc 0 (Real.pi / 2) → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≤ Real.sqrt 6 ∨ a ≥ 7/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3749_374964


namespace NUMINAMATH_CALUDE_valid_topping_combinations_l3749_374993

/-- Represents the number of cheese options --/
def cheese_options : ℕ := 3

/-- Represents the number of meat options --/
def meat_options : ℕ := 4

/-- Represents the number of vegetable options --/
def vegetable_options : ℕ := 5

/-- Represents that peppers is one of the vegetable options --/
axiom peppers_is_vegetable : vegetable_options > 0

/-- Represents that pepperoni is one of the meat options --/
axiom pepperoni_is_meat : meat_options > 0

/-- Calculates the total number of combinations without restrictions --/
def total_combinations : ℕ := cheese_options * meat_options * vegetable_options

/-- Represents the number of invalid combinations (pepperoni with peppers) --/
def invalid_combinations : ℕ := 1

/-- Theorem stating the total number of valid topping combinations --/
theorem valid_topping_combinations : 
  total_combinations - invalid_combinations = 59 := by sorry

end NUMINAMATH_CALUDE_valid_topping_combinations_l3749_374993


namespace NUMINAMATH_CALUDE_store_a_more_cost_effective_l3749_374909

/-- Represents the cost of purchasing tennis equipment from two different stores -/
def tennis_purchase_cost (x : ℝ) : Prop :=
  x > 40 ∧ 
  (25 * x + 3000 < 22.5 * x + 3600) = (x > 120)

/-- Theorem stating that Store A is more cost-effective when x = 100 -/
theorem store_a_more_cost_effective : tennis_purchase_cost 100 := by
  sorry

end NUMINAMATH_CALUDE_store_a_more_cost_effective_l3749_374909


namespace NUMINAMATH_CALUDE_chessboard_ratio_l3749_374903

/-- The number of squares on an n x n chessboard -/
def num_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The number of rectangles on a chessboard with m horizontal and n vertical lines -/
def num_rectangles (m n : ℕ) : ℕ := (m.choose 2) * (n.choose 2)

theorem chessboard_ratio :
  (num_squares 9 : ℚ) / (num_rectangles 10 10 : ℚ) = 19 / 135 := by sorry

end NUMINAMATH_CALUDE_chessboard_ratio_l3749_374903


namespace NUMINAMATH_CALUDE_intersecting_line_theorem_l3749_374949

/-- Given points A and B, and a line y = ax intersecting segment AB at point C,
    prove that if AC = 2CB, then a = 1 -/
theorem intersecting_line_theorem (a : ℝ) : 
  let A : ℝ × ℝ := (7, 1)
  let B : ℝ × ℝ := (1, 4)
  ∃ (C : ℝ × ℝ), 
    (C.2 = a * C.1) ∧  -- C is on the line y = ax
    (∃ t : ℝ, 0 < t ∧ t < 1 ∧ C = (1 - t) • A + t • B) ∧  -- C is on segment AB
    ((C.1 - A.1, C.2 - A.2) = (2 * (B.1 - C.1), 2 * (B.2 - C.2)))  -- AC = 2CB
    → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_line_theorem_l3749_374949


namespace NUMINAMATH_CALUDE_no_prime_divisor_l3749_374958

theorem no_prime_divisor : ¬ ∃ (p : ℕ), Prime p ∧ p > 1 ∧ p ∣ (1255 - 8) ∧ p ∣ (1490 - 11) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_divisor_l3749_374958


namespace NUMINAMATH_CALUDE_workshop_technicians_l3749_374988

theorem workshop_technicians 
  (total_workers : ℕ) 
  (avg_salary_all : ℚ) 
  (avg_salary_tech : ℚ) 
  (avg_salary_others : ℚ) 
  (h1 : total_workers = 20)
  (h2 : avg_salary_all = 750)
  (h3 : avg_salary_tech = 900)
  (h4 : avg_salary_others = 700) :
  ∃ (num_technicians : ℕ), 
    num_technicians * avg_salary_tech + (total_workers - num_technicians) * avg_salary_others = 
    total_workers * avg_salary_all ∧ 
    num_technicians = 5 := by
  sorry

end NUMINAMATH_CALUDE_workshop_technicians_l3749_374988


namespace NUMINAMATH_CALUDE_mary_bike_rental_cost_l3749_374933

/-- Calculates the total cost of bike rental given the fixed fee, hourly rate, and duration. -/
def bikeRentalCost (fixedFee : ℕ) (hourlyRate : ℕ) (duration : ℕ) : ℕ :=
  fixedFee + hourlyRate * duration

/-- Theorem stating that the bike rental cost for Mary is $80 -/
theorem mary_bike_rental_cost :
  bikeRentalCost 17 7 9 = 80 := by
  sorry

end NUMINAMATH_CALUDE_mary_bike_rental_cost_l3749_374933


namespace NUMINAMATH_CALUDE_add_and_round_to_nearest_ten_l3749_374960

def round_to_nearest_ten (n : ℤ) : ℤ :=
  10 * ((n + 5) / 10)

theorem add_and_round_to_nearest_ten : round_to_nearest_ten (58 + 29) = 90 := by
  sorry

end NUMINAMATH_CALUDE_add_and_round_to_nearest_ten_l3749_374960


namespace NUMINAMATH_CALUDE_male_average_score_l3749_374999

theorem male_average_score (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (overall_average : ℚ) (female_average : ℚ) :
  total_students = male_students + female_students →
  total_students = 28 →
  male_students = 8 →
  female_students = 20 →
  overall_average = 90 →
  female_average = 92 →
  (total_students : ℚ) * overall_average = 
    (male_students : ℚ) * ((total_students : ℚ) * overall_average - (female_students : ℚ) * female_average) / (male_students : ℚ) + 
    (female_students : ℚ) * female_average →
  ((total_students : ℚ) * overall_average - (female_students : ℚ) * female_average) / (male_students : ℚ) = 85 :=
by sorry

end NUMINAMATH_CALUDE_male_average_score_l3749_374999


namespace NUMINAMATH_CALUDE_average_weight_b_c_l3749_374953

theorem average_weight_b_c (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 41)
  (h3 : b = 33) :
  (b + c) / 2 = 43 := by
sorry

end NUMINAMATH_CALUDE_average_weight_b_c_l3749_374953


namespace NUMINAMATH_CALUDE_sum_210_72_in_base5_l3749_374997

/-- Converts a decimal number to its base 5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of base 5 digits to a decimal number -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_210_72_in_base5 :
  toBase5 (210 + 72) = [2, 0, 6, 2] := by
  sorry

end NUMINAMATH_CALUDE_sum_210_72_in_base5_l3749_374997


namespace NUMINAMATH_CALUDE_martha_apples_l3749_374954

theorem martha_apples (jane_apples james_apples martha_remaining martha_to_give : ℕ) :
  jane_apples = 5 →
  james_apples = jane_apples + 2 →
  martha_remaining = 4 →
  martha_to_give = 4 →
  jane_apples + james_apples + martha_remaining + martha_to_give = 20 :=
by sorry

end NUMINAMATH_CALUDE_martha_apples_l3749_374954


namespace NUMINAMATH_CALUDE_function_property_l3749_374959

/-- Strictly increasing function from ℕ+ to ℕ+ -/
def StrictlyIncreasing (f : ℕ+ → ℕ+) : Prop :=
  ∀ x y : ℕ+, x < y → f x < f y

/-- The image of a function f : ℕ+ → ℕ+ -/
def Image (f : ℕ+ → ℕ+) : Set ℕ+ :=
  {y : ℕ+ | ∃ x : ℕ+, f x = y}

theorem function_property (f g : ℕ+ → ℕ+) 
  (h1 : StrictlyIncreasing f)
  (h2 : StrictlyIncreasing g)
  (h3 : Image f ∪ Image g = Set.univ)
  (h4 : Image f ∩ Image g = ∅)
  (h5 : ∀ n : ℕ+, g n = f (f n) + 1) :
  f 240 = 388 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3749_374959


namespace NUMINAMATH_CALUDE_distributive_property_subtraction_l3749_374992

theorem distributive_property_subtraction (a b c : ℝ) : a - (b + c) = a - b - c := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_subtraction_l3749_374992


namespace NUMINAMATH_CALUDE_cost_per_bag_is_seven_l3749_374979

/-- Calculates the cost per bag given the number of bags, selling price, and desired profit --/
def cost_per_bag (num_bags : ℕ) (selling_price : ℚ) (desired_profit : ℚ) : ℚ :=
  (num_bags * selling_price - desired_profit) / num_bags

/-- Theorem: Given 100 bags sold at $10 each with a $300 profit, the cost per bag is $7 --/
theorem cost_per_bag_is_seven :
  cost_per_bag 100 10 300 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_bag_is_seven_l3749_374979


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3749_374935

theorem infinite_series_sum : 
  let r := (1 : ℝ) / 1950
  let S := ∑' n, n * r^(n-1)
  S = 3802500 / 3802601 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3749_374935


namespace NUMINAMATH_CALUDE_groceries_expense_l3749_374925

def monthly_salary : ℕ := 20000
def savings_percentage : ℚ := 1/10
def savings_amount : ℕ := 2000
def rent : ℕ := 5000
def milk : ℕ := 1500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 2500

theorem groceries_expense (h1 : savings_amount = monthly_salary * savings_percentage) 
  (h2 : savings_amount = 2000) : 
  monthly_salary - (rent + milk + education + petrol + miscellaneous + savings_amount) = 6500 := by
  sorry

end NUMINAMATH_CALUDE_groceries_expense_l3749_374925


namespace NUMINAMATH_CALUDE_lawyer_fee_ratio_l3749_374906

/-- Lawyer fee calculation and payment ratio problem --/
theorem lawyer_fee_ratio :
  let upfront_fee : ℕ := 1000
  let hourly_rate : ℕ := 100
  let court_hours : ℕ := 50
  let prep_hours : ℕ := 2 * court_hours
  let total_fee : ℕ := upfront_fee + hourly_rate * (court_hours + prep_hours)
  let john_payment : ℕ := 8000
  let brother_payment : ℕ := total_fee - john_payment
  brother_payment * 2 = total_fee := by sorry

end NUMINAMATH_CALUDE_lawyer_fee_ratio_l3749_374906


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l3749_374943

theorem sqrt_fraction_simplification :
  Real.sqrt (7^2 + 24^2) / Real.sqrt (49 + 16) = (25 * Real.sqrt 65) / 65 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l3749_374943


namespace NUMINAMATH_CALUDE_oldest_child_age_l3749_374980

theorem oldest_child_age (average_age : ℝ) (age1 age2 age3 : ℕ) :
  average_age = 9 ∧ age1 = 5 ∧ age2 = 8 ∧ age3 = 11 →
  ∃ (age4 : ℕ), (age1 + age2 + age3 + age4 : ℝ) / 4 = average_age ∧ age4 = 12 :=
by sorry

end NUMINAMATH_CALUDE_oldest_child_age_l3749_374980


namespace NUMINAMATH_CALUDE_tangent_line_slope_is_one_l3749_374989

/-- The slope of a line passing through (-1, 0) and tangent to y = e^x is 1 -/
theorem tangent_line_slope_is_one :
  ∀ (a : ℝ), 
    (∃ (k : ℝ), 
      (∀ x, k * (x + 1) = Real.exp x → x = a) ∧ 
      k * (a + 1) = Real.exp a ∧
      k = Real.exp a) →
    k = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_line_slope_is_one_l3749_374989


namespace NUMINAMATH_CALUDE_range_of_a_l3749_374911

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3749_374911


namespace NUMINAMATH_CALUDE_relationship_abc_l3749_374936

theorem relationship_abc (a b c : ℝ) : 
  a = 2 → b = Real.log 9 → c = 2 * Real.sin (9 * π / 5) → a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l3749_374936


namespace NUMINAMATH_CALUDE_tom_july_books_l3749_374998

/-- The number of books Tom read in May -/
def may_books : ℕ := 2

/-- The number of books Tom read in June -/
def june_books : ℕ := 6

/-- The total number of books Tom read -/
def total_books : ℕ := 18

/-- The number of books Tom read in July -/
def july_books : ℕ := total_books - may_books - june_books

theorem tom_july_books : july_books = 10 := by
  sorry

end NUMINAMATH_CALUDE_tom_july_books_l3749_374998


namespace NUMINAMATH_CALUDE_system_solution_l3749_374924

theorem system_solution (x y z : ℝ) : 
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x^3 = 2*y^2 - z ∧
  y^3 = 2*z^2 - x ∧
  z^3 = 2*x^2 - y →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3749_374924


namespace NUMINAMATH_CALUDE_factorization_problems_l3749_374927

theorem factorization_problems (x y a : ℝ) : 
  (-3 * x^3 * y + 6 * x^2 * y^2 - 3 * x * y^3 = -3 * x * y * (x - y)^2) ∧
  ((a^2 + 9)^2 - 36 * a^2 = (a + 3)^2 * (a - 3)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l3749_374927


namespace NUMINAMATH_CALUDE_raffle_ticket_cost_l3749_374916

theorem raffle_ticket_cost (x : ℚ) : 
  (25 * x + 30 + 20 = 100) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_raffle_ticket_cost_l3749_374916


namespace NUMINAMATH_CALUDE_number_of_combinations_is_736_l3749_374910

/-- Represents the number of different ways to occupy planets given the specified conditions --/
def number_of_combinations : ℕ :=
  let earth_like_planets : ℕ := 7
  let mars_like_planets : ℕ := 6
  let earth_like_units : ℕ := 3
  let mars_like_units : ℕ := 1
  let total_units : ℕ := 15

  -- The actual calculation of combinations
  0 -- placeholder, replace with actual calculation

/-- Theorem stating that the number of combinations is 736 --/
theorem number_of_combinations_is_736 : number_of_combinations = 736 := by
  sorry


end NUMINAMATH_CALUDE_number_of_combinations_is_736_l3749_374910


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l3749_374929

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y + 1) * (y - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l3749_374929


namespace NUMINAMATH_CALUDE_coeff_x_cubed_expansion_l3749_374990

/-- The coefficient of x^3 in the expansion of (x^2 - x + 1)^10 -/
def coeff_x_cubed : ℤ := -210

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coeff_x_cubed_expansion :
  coeff_x_cubed = binomial 10 8 * binomial 2 1 * (-1) + binomial 10 7 * binomial 3 3 * (-1) :=
sorry

end NUMINAMATH_CALUDE_coeff_x_cubed_expansion_l3749_374990


namespace NUMINAMATH_CALUDE_triangle_angle_C_l3749_374994

theorem triangle_angle_C (A B C : ℝ) (h : (Real.cos A + Real.sin A) * (Real.cos B + Real.sin B) = 2) :
  A + B + C = Real.pi → C = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l3749_374994
