import Mathlib

namespace limit_sqrt_minus_one_over_x_l750_75086

theorem limit_sqrt_minus_one_over_x (f : ℝ → ℝ) (h : ∀ x ≠ 0, f x = (1 - Real.sqrt (x + 1)) / x) :
  Filter.Tendsto f (Filter.atTop.comap (fun x => 1 / x)) (nhds (-1/2)) := by
sorry

end limit_sqrt_minus_one_over_x_l750_75086


namespace cubic_polynomial_problem_l750_75083

/-- Given a cubic equation and a polynomial P satisfying certain conditions, 
    prove that P has a specific form. -/
theorem cubic_polynomial_problem (a b c : ℝ) (P : ℝ → ℝ) : 
  (∀ x, x^3 - 4*x^2 + x - 1 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  (∃ p q r s : ℝ, ∀ x, P x = p*x^3 + q*x^2 + r*x + s) →
  P a = b + c →
  P b = a + c →
  P c = a + b →
  P (a + b + c) = -20 →
  ∀ x, P x = (-20*x^3 + 80*x^2 - 23*x + 32) / 3 := by
sorry

end cubic_polynomial_problem_l750_75083


namespace product_and_sum_of_factors_l750_75094

theorem product_and_sum_of_factors : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 1540 ∧ 
  a + b = 97 := by
  sorry

end product_and_sum_of_factors_l750_75094


namespace sum_of_squares_l750_75067

/-- A structure representing a set of four-digit numbers formed from four distinct digits. -/
structure FourDigitSet where
  digits : Finset Nat
  first_number : Nat
  second_last_number : Nat
  (digit_count : digits.card = 4)
  (distinct_digits : ∀ d ∈ digits, d < 10)
  (number_count : (digits.powerset.filter (λ s : Finset Nat => s.card = 4)).card = 18)
  (ascending_order : first_number < second_last_number)
  (first_is_square : ∃ n : Nat, first_number = n ^ 2)
  (second_last_is_square : ∃ n : Nat, second_last_number = n ^ 2)

/-- The theorem stating that the sum of the first and second-last numbers is 10890. -/
theorem sum_of_squares (s : FourDigitSet) : s.first_number + s.second_last_number = 10890 := by
  sorry

end sum_of_squares_l750_75067


namespace milk_students_l750_75092

theorem milk_students (total_students : ℕ) (soda_students : ℕ) (milk_percent : ℚ) (soda_percent : ℚ) :
  soda_percent = 1/2 →
  milk_percent = 3/10 →
  soda_students = 90 →
  (milk_percent / soda_percent) * soda_students = 54 :=
by sorry

end milk_students_l750_75092


namespace consecutive_four_plus_one_is_square_l750_75066

theorem consecutive_four_plus_one_is_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end consecutive_four_plus_one_is_square_l750_75066


namespace total_roses_planted_l750_75008

/-- The number of roses planted by Uncle Welly over three days -/
def roses_planted (day1 day2 day3 : ℕ) : ℕ := day1 + day2 + day3

/-- Theorem stating the total number of roses planted -/
theorem total_roses_planted :
  let day1 := 50
  let day2 := day1 + 20
  let day3 := 2 * day1
  roses_planted day1 day2 day3 = 220 := by sorry

end total_roses_planted_l750_75008


namespace unique_two_digit_number_mod_13_l750_75043

theorem unique_two_digit_number_mod_13 :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ (13 * n) % 100 = 42 :=
by
  -- The proof goes here
  sorry

end unique_two_digit_number_mod_13_l750_75043


namespace ratio_of_sums_l750_75010

theorem ratio_of_sums (p q r u v w : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end ratio_of_sums_l750_75010


namespace refrigerator_savings_l750_75032

def cash_price : ℕ := 8000
def deposit : ℕ := 3000
def num_installments : ℕ := 30
def installment_amount : ℕ := 300

theorem refrigerator_savings : 
  deposit + num_installments * installment_amount - cash_price = 4000 := by
  sorry

end refrigerator_savings_l750_75032


namespace dolphin_training_hours_l750_75042

theorem dolphin_training_hours (num_dolphins : ℕ) (training_hours_per_dolphin : ℕ) (num_trainers : ℕ) 
  (h1 : num_dolphins = 12)
  (h2 : training_hours_per_dolphin = 5)
  (h3 : num_trainers = 4)
  (h4 : num_trainers > 0) :
  (num_dolphins * training_hours_per_dolphin) / num_trainers = 15 := by
sorry

end dolphin_training_hours_l750_75042


namespace dogs_left_over_l750_75077

theorem dogs_left_over (total_dogs : ℕ) (num_houses : ℕ) (h1 : total_dogs = 50) (h2 : num_houses = 17) : 
  total_dogs - (num_houses * (total_dogs / num_houses)) = 16 := by
sorry

end dogs_left_over_l750_75077


namespace f_geq_f1_iff_a_in_range_l750_75020

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1 then (1/3) * x^3 - a * x + 1
  else if x ≥ 1 then a * Real.log x
  else 0  -- This case should never occur in our problem, but Lean requires it for completeness

-- State the theorem
theorem f_geq_f1_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ f a 1) ↔ (0 < a ∧ a ≤ 4/3) :=
by sorry

end f_geq_f1_iff_a_in_range_l750_75020


namespace conic_is_circle_l750_75026

-- Define the equation
def conic_equation (x y : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 49

-- Theorem stating that the equation represents a circle
theorem conic_is_circle :
  ∃ (h k r : ℝ), r > 0 ∧ 
  (∀ (x y : ℝ), conic_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2) :=
sorry

end conic_is_circle_l750_75026


namespace tower_of_threes_greater_than_tower_of_twos_l750_75074

-- Define a function to represent the tower of exponents
def tower (base : ℕ) (height : ℕ) : ℕ :=
  match height with
  | 0 => 1
  | n + 1 => base ^ (tower base n)

-- State the theorem
theorem tower_of_threes_greater_than_tower_of_twos :
  tower 3 99 > tower 2 100 :=
sorry

end tower_of_threes_greater_than_tower_of_twos_l750_75074


namespace ratio_problem_l750_75068

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end ratio_problem_l750_75068


namespace factorization_equality_l750_75013

theorem factorization_equality (a : ℝ) : -3*a + 12*a^2 - 12*a^3 = -3*a*(1-2*a)^2 := by
  sorry

end factorization_equality_l750_75013


namespace opposite_number_theorem_l750_75018

theorem opposite_number_theorem (a b c : ℝ) : 
  -((-a + b - c)) = c - a - b := by sorry

end opposite_number_theorem_l750_75018


namespace tire_price_proof_l750_75037

/-- The regular price of a tire -/
def regular_price : ℝ := 126

/-- The promotional price for three tires -/
def promotional_price : ℝ := 315

/-- The promotion discount on the third tire -/
def discount : ℝ := 0.5

theorem tire_price_proof :
  2 * regular_price + discount * regular_price = promotional_price :=
by sorry

end tire_price_proof_l750_75037


namespace largest_package_size_l750_75064

theorem largest_package_size (alex_folders jamie_folders : ℕ) 
  (h1 : alex_folders = 60) (h2 : jamie_folders = 90) : 
  Nat.gcd alex_folders jamie_folders = 30 := by
  sorry

end largest_package_size_l750_75064


namespace triangle_perimeter_and_shape_l750_75038

/-- Given a triangle ABC with side lengths a, b, and c satisfying certain conditions,
    prove that its perimeter is 17 and it is an isosceles triangle. -/
theorem triangle_perimeter_and_shape (a b c : ℝ) : 
  (b - 5)^2 + (c - 7)^2 = 0 →
  |a - 3| = 2 →
  a + b + c = 17 ∧ a = b := by
  sorry

end triangle_perimeter_and_shape_l750_75038


namespace sum_of_squares_divisible_by_seven_l750_75099

theorem sum_of_squares_divisible_by_seven (x y : ℤ) : 
  (7 ∣ x^2 + y^2) → (7 ∣ x) ∧ (7 ∣ y) := by
  sorry

end sum_of_squares_divisible_by_seven_l750_75099


namespace triangle_acute_if_tan_product_positive_l750_75001

/-- Given a triangle ABC with internal angles A, B, and C, 
    if the product of their tangents is positive, 
    then the triangle is acute. -/
theorem triangle_acute_if_tan_product_positive 
  (A B C : Real) 
  (h_triangle : A + B + C = π) 
  (h_positive : Real.tan A * Real.tan B * Real.tan C > 0) : 
  A < π/2 ∧ B < π/2 ∧ C < π/2 := by
  sorry

end triangle_acute_if_tan_product_positive_l750_75001


namespace third_side_length_valid_l750_75063

theorem third_side_length_valid (a b c : ℝ) : 
  a = 2 → b = 4 → c = 4 → 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) := by
  sorry

end third_side_length_valid_l750_75063


namespace slices_per_pizza_l750_75015

theorem slices_per_pizza (total_pizzas : ℕ) (total_slices : ℕ) (h1 : total_pizzas = 7) (h2 : total_slices = 14) :
  total_slices / total_pizzas = 2 := by
  sorry

end slices_per_pizza_l750_75015


namespace simplify_fraction_multiplication_l750_75031

theorem simplify_fraction_multiplication : (405 : ℚ) / 1215 * 27 = 9 := by
  sorry

end simplify_fraction_multiplication_l750_75031


namespace smallest_equal_packs_l750_75055

theorem smallest_equal_packs (pencil_pack : Nat) (eraser_pack : Nat) : 
  pencil_pack = 5 → eraser_pack = 7 → 
  (∃ n : Nat, n > 0 ∧ ∃ m : Nat, n * eraser_pack = m * pencil_pack ∧ 
  ∀ k : Nat, k > 0 → k * eraser_pack = m * pencil_pack → n ≤ k) → n = 5 := by
sorry

end smallest_equal_packs_l750_75055


namespace square_sum_problem_l750_75059

theorem square_sum_problem (a b : ℝ) (h1 : a + b = -9) (h2 : a = 30 / b) : a^2 + b^2 = 61 := by
  sorry

end square_sum_problem_l750_75059


namespace sean_needs_six_packs_l750_75093

def bedroom_bulbs : ℕ := 2
def bathroom_bulbs : ℕ := 1
def kitchen_bulbs : ℕ := 1
def basement_bulbs : ℕ := 4
def bulbs_per_pack : ℕ := 2

def total_non_garage_bulbs : ℕ := bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs

def garage_bulbs : ℕ := total_non_garage_bulbs / 2

def total_bulbs : ℕ := total_non_garage_bulbs + garage_bulbs

theorem sean_needs_six_packs : (total_bulbs + bulbs_per_pack - 1) / bulbs_per_pack = 6 := by
  sorry

end sean_needs_six_packs_l750_75093


namespace smallest_solution_quartic_equation_l750_75080

theorem smallest_solution_quartic_equation :
  let f : ℝ → ℝ := λ x => x^4 - 40*x^2 + 144
  ∃ (x : ℝ), f x = 0 ∧ (∀ y : ℝ, f y = 0 → x ≤ y) ∧ x = -6 :=
by sorry

end smallest_solution_quartic_equation_l750_75080


namespace a_100_value_l750_75034

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a n - a (n + 1) = 2

theorem a_100_value (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 3 = 6) : 
  a 100 = -188 := by
  sorry

end a_100_value_l750_75034


namespace initial_term_range_l750_75029

/-- A strictly increasing sequence satisfying the given recursive formula -/
def StrictlyIncreasingSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) > a n) ∧ 
  (∀ n, a (n + 1) = (4 * a n - 2) / (a n + 1))

/-- The theorem stating that the initial term of the sequence must be in (1, 2) -/
theorem initial_term_range (a : ℕ → ℝ) :
  StrictlyIncreasingSequence a → 1 < a 1 ∧ a 1 < 2 := by
  sorry


end initial_term_range_l750_75029


namespace initial_men_count_l750_75016

/-- The number of days it takes for the initial group to complete the work -/
def initial_days : ℕ := 70

/-- The number of days it takes for 40 men to complete the work -/
def new_days : ℕ := 63

/-- The number of men in the new group -/
def new_men : ℕ := 40

/-- The amount of work is constant and can be represented as men * days -/
axiom work_constant (m1 m2 : ℕ) (d1 d2 : ℕ) : m1 * d1 = m2 * d2

/-- The theorem to be proved -/
theorem initial_men_count : ∃ x : ℕ, x * initial_days = new_men * new_days ∧ x = 36 := by
  sorry

end initial_men_count_l750_75016


namespace exponent_zero_equals_one_f_equals_S_l750_75082

-- Option C
theorem exponent_zero_equals_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

-- Option D
def f (x : ℝ) : ℝ := x^2
def S (t : ℝ) : ℝ := t^2

theorem f_equals_S : f = S := by sorry

end exponent_zero_equals_one_f_equals_S_l750_75082


namespace paper_pieces_l750_75007

/-- The number of pieces of paper after n tears -/
def num_pieces (n : ℕ) : ℕ := 3 * n + 1

/-- Theorem stating the number of pieces after n tears -/
theorem paper_pieces (n : ℕ) : 
  (∀ k : ℕ, k ≤ n → num_pieces k = num_pieces (k - 1) + 3) → 
  num_pieces n = 3 * n + 1 :=
by sorry

end paper_pieces_l750_75007


namespace hyperbola_equation_and_minimum_distance_l750_75005

structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

def on_hyperbola (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

def asymptotic_equation (h : Hyperbola) : Prop :=
  h.b = Real.sqrt 3 * h.a

def point_on_hyperbola (h : Hyperbola) : Prop :=
  on_hyperbola h (Real.sqrt 5) (Real.sqrt 3)

def perpendicular_vectors (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem hyperbola_equation_and_minimum_distance 
  (h : Hyperbola) 
  (h_asymptotic : asymptotic_equation h)
  (h_point : point_on_hyperbola h) :
  (h.a = 2 ∧ h.b = 2 * Real.sqrt 3) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    on_hyperbola h x₁ y₁ → 
    on_hyperbola h x₂ y₂ → 
    perpendicular_vectors x₁ y₁ x₂ y₂ → 
    x₁^2 + y₁^2 + x₂^2 + y₂^2 ≥ 24) :=
  sorry

end hyperbola_equation_and_minimum_distance_l750_75005


namespace tv_show_sampling_interval_l750_75053

/-- Calculate the sampling interval for system sampling --/
def sampling_interval (total_population : ℕ) (sample_size : ℕ) : ℕ :=
  total_population / sample_size

/-- Theorem: The sampling interval for selecting 10 viewers from 10,000 is 1000 --/
theorem tv_show_sampling_interval :
  sampling_interval 10000 10 = 1000 := by
  sorry

#eval sampling_interval 10000 10

end tv_show_sampling_interval_l750_75053


namespace x_intercept_of_line_l750_75012

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℚ) : 4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end x_intercept_of_line_l750_75012


namespace polyhedron_relations_l750_75075

structure Polyhedron where
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  V : ℕ  -- number of vertices
  n : ℕ  -- number of sides in each face
  m : ℕ  -- number of edges meeting at each vertex

theorem polyhedron_relations (P : Polyhedron) : 
  (P.n * P.F = 2 * P.E) ∧ 
  (P.m * P.V = 2 * P.E) ∧ 
  (P.V + P.F = P.E + 2) ∧ 
  ¬(P.m * P.F = 2 * P.E) := by
  sorry

end polyhedron_relations_l750_75075


namespace fewer_twos_to_hundred_l750_75089

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_to_hundred_l750_75089


namespace unique_preimage_of_triple_l750_75003

-- Define v₂ function
def v₂ (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n.log 2)

-- Define the properties of function f
def has_properties (f : ℕ → ℕ) : Prop :=
  (∀ x : ℕ, f x ≤ 3 * x) ∧ 
  (∀ x y : ℕ, v₂ (f x + f y) = v₂ (x + y))

-- State the theorem
theorem unique_preimage_of_triple (f : ℕ → ℕ) (h : has_properties f) :
  ∀ a : ℕ, ∃! x : ℕ, f x = 3 * a :=
sorry

end unique_preimage_of_triple_l750_75003


namespace equation_roots_imply_a_range_l750_75065

open Real

theorem equation_roots_imply_a_range (m : ℝ) (a : ℝ) (e : ℝ) :
  m > 0 →
  e > 0 →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁ + a * (2 * x₁ + 2 * m - 4 * e * x₁) * (log (x₁ + m) - log x₁) = 0 ∧
    x₂ + a * (2 * x₂ + 2 * m - 4 * e * x₂) * (log (x₂ + m) - log x₂) = 0) →
  a > 1 / (2 * e) :=
by sorry

end equation_roots_imply_a_range_l750_75065


namespace matchsticks_left_l750_75036

def totalMatchsticks : ℕ := 50
def elvisSquareMatchsticks : ℕ := 4
def ralphSquareMatchsticks : ℕ := 8
def zoeyTriangleMatchsticks : ℕ := 6
def elvisMaxMatchsticks : ℕ := 20
def ralphMaxMatchsticks : ℕ := 20
def zoeyMaxMatchsticks : ℕ := 15
def maxTotalShapes : ℕ := 9

theorem matchsticks_left : 
  ∃ (elvisShapes ralphShapes zoeyShapes : ℕ),
    elvisShapes * elvisSquareMatchsticks ≤ elvisMaxMatchsticks ∧
    ralphShapes * ralphSquareMatchsticks ≤ ralphMaxMatchsticks ∧
    zoeyShapes * zoeyTriangleMatchsticks ≤ zoeyMaxMatchsticks ∧
    elvisShapes + ralphShapes + zoeyShapes = maxTotalShapes ∧
    totalMatchsticks - (elvisShapes * elvisSquareMatchsticks + 
                        ralphShapes * ralphSquareMatchsticks + 
                        zoeyShapes * zoeyTriangleMatchsticks) = 2 :=
by sorry

end matchsticks_left_l750_75036


namespace arithmetic_progression_same_digit_sum_l750_75049

/-- Sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Arithmetic progression with first term a and common difference d -/
def arithmeticProgression (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

theorem arithmetic_progression_same_digit_sum (a d : ℕ) :
  ∃ m n : ℕ, m ≠ n ∧ 
    digitSum (arithmeticProgression a d m) = digitSum (arithmeticProgression a d n) := by
  sorry

end arithmetic_progression_same_digit_sum_l750_75049


namespace least_number_for_divisibility_l750_75091

theorem least_number_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((2496 + y) % 7 = 0 ∧ (2496 + y) % 11 = 0)) ∧ 
  (2496 + x) % 7 = 0 ∧ (2496 + x) % 11 = 0 → 
  x = 37 := by sorry

end least_number_for_divisibility_l750_75091


namespace max_sum_of_squares_l750_75051

/-- The distance from a real number to the nearest integer -/
noncomputable def distToNearestInt (x : ℝ) : ℝ := min (x - ⌊x⌋) (⌈x⌉ - x)

/-- The sum of squares of x * (distance of x to nearest integer) -/
noncomputable def sumOfSquares (xs : Finset ℝ) : ℝ :=
  Finset.sum xs (λ x => (x * distToNearestInt x)^2)

/-- The maximum value of the sum of squares given the constraints -/
theorem max_sum_of_squares (n : ℕ) :
  ∃ (xs : Finset ℝ),
    (∀ x ∈ xs, 0 ≤ x) ∧
    (Finset.sum xs id = n) ∧
    (Finset.card xs = n) ∧
    (∀ ys : Finset ℝ,
      (∀ y ∈ ys, 0 ≤ y) →
      (Finset.sum ys id = n) →
      (Finset.card ys = n) →
      sumOfSquares ys ≤ sumOfSquares xs) ∧
    (sumOfSquares xs = (n^2 - n + 1/2) / 4) := by
  sorry

end max_sum_of_squares_l750_75051


namespace train_bridge_crossing_time_l750_75040

/-- Proves that a train of given length, traveling at a given speed, will take the calculated time to cross a bridge of given length. -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (bridge_length : ℝ)
  (train_speed_kmph : ℝ)
  (h1 : train_length = 100)
  (h2 : bridge_length = 200)
  (h3 : train_speed_kmph = 36)
  : (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 30 :=
by sorry

end train_bridge_crossing_time_l750_75040


namespace fred_paper_count_l750_75006

theorem fred_paper_count (initial_sheets received_sheets given_sheets : ℕ) :
  initial_sheets = 212 →
  received_sheets = 307 →
  given_sheets = 156 →
  initial_sheets + received_sheets - given_sheets = 363 := by
  sorry

end fred_paper_count_l750_75006


namespace number_divided_by_002_equals_50_l750_75024

theorem number_divided_by_002_equals_50 :
  ∃ x : ℝ, x / 0.02 = 50 ∧ x = 1 := by
  sorry

end number_divided_by_002_equals_50_l750_75024


namespace edward_final_lives_l750_75028

/-- Calculates the final number of lives Edward has after completing three stages of a game. -/
def final_lives (initial_lives : ℕ) 
                (stage1_loss stage1_gain : ℕ) 
                (stage2_loss stage2_gain : ℕ) 
                (stage3_loss stage3_gain : ℕ) : ℕ :=
  initial_lives - stage1_loss + stage1_gain - stage2_loss + stage2_gain - stage3_loss + stage3_gain

/-- Theorem stating that Edward's final number of lives is 23 given the specified conditions. -/
theorem edward_final_lives : 
  final_lives 50 18 7 10 5 13 2 = 23 := by
  sorry


end edward_final_lives_l750_75028


namespace solution_pairs_l750_75069

theorem solution_pairs : 
  ∀ (x y : ℕ), 2^(2*x+1) + 2^x + 1 = y^2 ↔ (x = 0 ∧ y = 2) ∨ (x = 4 ∧ y = 23) :=
by sorry

end solution_pairs_l750_75069


namespace scientific_notation_of_400_million_l750_75002

theorem scientific_notation_of_400_million :
  (400000000 : ℝ) = 4 * 10^8 := by
  sorry

end scientific_notation_of_400_million_l750_75002


namespace min_dimension_sum_for_2310_volume_l750_75030

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- The volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- The sum of the dimensions of a box -/
def dimensionSum (d : BoxDimensions) : ℕ := d.length + d.width + d.height

/-- Theorem stating that the minimum sum of dimensions for a box with volume 2310 is 42 -/
theorem min_dimension_sum_for_2310_volume :
  (∃ (d : BoxDimensions), boxVolume d = 2310) →
  (∀ (d : BoxDimensions), boxVolume d = 2310 → dimensionSum d ≥ 42) ∧
  (∃ (d : BoxDimensions), boxVolume d = 2310 ∧ dimensionSum d = 42) :=
by sorry

end min_dimension_sum_for_2310_volume_l750_75030


namespace two_thirds_of_number_l750_75057

theorem two_thirds_of_number (y : ℝ) : (2 / 3) * y = 40 → y = 60 := by
  sorry

end two_thirds_of_number_l750_75057


namespace intermediate_root_existence_l750_75071

theorem intermediate_root_existence (a b c x₁ x₂ : ℝ) 
  (ha : a ≠ 0)
  (hx₁ : a * x₁^2 + b * x₁ + c = 0)
  (hx₂ : -a * x₂^2 + b * x₂ + c = 0) :
  ∃ x₃ : ℝ, (a / 2) * x₃^2 + b * x₃ + c = 0 ∧ 
    ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) := by
  sorry

end intermediate_root_existence_l750_75071


namespace jerry_logs_count_l750_75056

/-- The number of logs Jerry gets from cutting trees -/
def total_logs : ℕ :=
  let pine_logs_per_tree : ℕ := 80
  let maple_logs_per_tree : ℕ := 60
  let walnut_logs_per_tree : ℕ := 100
  let pine_trees_cut : ℕ := 8
  let maple_trees_cut : ℕ := 3
  let walnut_trees_cut : ℕ := 4
  pine_logs_per_tree * pine_trees_cut +
  maple_logs_per_tree * maple_trees_cut +
  walnut_logs_per_tree * walnut_trees_cut

theorem jerry_logs_count : total_logs = 1220 := by
  sorry

end jerry_logs_count_l750_75056


namespace distance_traveled_l750_75085

/-- Given a person traveling at a constant speed for a certain time,
    prove that the distance traveled is equal to the product of speed and time. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 25) (h2 : time = 5) :
  speed * time = 125 := by
  sorry

end distance_traveled_l750_75085


namespace smallest_sum_of_four_consecutive_primes_divisible_by_four_l750_75004

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_primes (p₁ p₂ p₃ p₄ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧
  ∀ q : ℕ, (is_prime q ∧ p₁ < q ∧ q < p₄) → (q = p₂ ∨ q = p₃)

theorem smallest_sum_of_four_consecutive_primes_divisible_by_four :
  ∃ p₁ p₂ p₃ p₄ : ℕ,
    consecutive_primes p₁ p₂ p₃ p₄ ∧
    (p₁ + p₂ + p₃ + p₄) % 4 = 0 ∧
    p₁ + p₂ + p₃ + p₄ = 36 ∧
    ∀ q₁ q₂ q₃ q₄ : ℕ,
      consecutive_primes q₁ q₂ q₃ q₄ →
      (q₁ + q₂ + q₃ + q₄) % 4 = 0 →
      q₁ + q₂ + q₃ + q₄ ≥ 36 :=
sorry

end smallest_sum_of_four_consecutive_primes_divisible_by_four_l750_75004


namespace golden_ratio_logarithm_l750_75041

theorem golden_ratio_logarithm (r s : ℝ) (hr : r > 0) (hs : s > 0) :
  (Real.log r / Real.log 4 = Real.log s / Real.log 18) ∧
  (Real.log s / Real.log 18 = Real.log (r + s) / Real.log 24) →
  s / r = (1 + Real.sqrt 5) / 2 := by
sorry

end golden_ratio_logarithm_l750_75041


namespace jenn_savings_problem_l750_75011

/-- Given information about Jenn's savings for a bike purchase --/
theorem jenn_savings_problem (num_jars : ℕ) (bike_cost : ℕ) (leftover : ℕ) :
  num_jars = 5 →
  bike_cost = 180 →
  leftover = 20 →
  (∃ (quarters_per_jar : ℕ),
    quarters_per_jar * num_jars = (bike_cost + leftover) * 4 ∧
    quarters_per_jar = 160) :=
by sorry

end jenn_savings_problem_l750_75011


namespace five_year_compound_interest_l750_75025

/-- Calculates the final amount after compound interest --/
def compound_interest (m : ℝ) (a : ℝ) (n : ℕ) : ℝ :=
  m * (1 + a) ^ n

/-- Theorem: After 5 years of compound interest, the final amount is m(1+a)^5 --/
theorem five_year_compound_interest (m : ℝ) (a : ℝ) :
  compound_interest m a 5 = m * (1 + a) ^ 5 :=
by
  sorry

end five_year_compound_interest_l750_75025


namespace expression_evaluation_l750_75035

theorem expression_evaluation :
  let x : ℕ := 3
  (x + x * x^(x^2)) * 3 = 177156 := by sorry

end expression_evaluation_l750_75035


namespace football_angles_l750_75097

-- Define the football structure
structure Football :=
  (edge_length : ℝ)
  (pentagon_sides : ℕ)
  (hexagon_sides : ℕ)
  (pentagons_per_hexagon : ℕ)

-- Define the angles between faces
def angle_between_hexagons (f : Football) : ℝ := sorry
def angle_between_hexagon_and_pentagon (f : Football) : ℝ := sorry

-- Theorem statement
theorem football_angles 
  (f : Football) 
  (h1 : f.edge_length = 1)
  (h2 : f.pentagon_sides = 5)
  (h3 : f.hexagon_sides = 6)
  (h4 : f.pentagons_per_hexagon = 5) :
  ∃ (α β : ℝ), 
    α = angle_between_hexagons f ∧
    β = angle_between_hexagon_and_pentagon f ∧
    ∃ (t1 t2 : ℝ → ℝ), 
      (t1 = Real.tan) ∧ 
      (t2 = Real.tan) ∧
      (t1 α = (Real.sqrt (3 * 3 - 2 * 2)) / 2) ∧
      (t2 β = (Real.sqrt (5 - 2 * Real.sqrt 5)) / (3 - Real.sqrt 5)) :=
sorry

end football_angles_l750_75097


namespace alice_cookies_l750_75009

/-- Given that Alice can make 24 cookies with 4 cups of flour,
    this theorem proves that she can make 30 cookies with 5 cups of flour. -/
theorem alice_cookies (cookies_four : ℕ) (flour_four : ℕ) (flour_five : ℕ)
  (h1 : cookies_four = 24)
  (h2 : flour_four = 4)
  (h3 : flour_five = 5)
  : (cookies_four * flour_five) / flour_four = 30 := by
  sorry

end alice_cookies_l750_75009


namespace max_value_of_x_plus_inverse_l750_75072

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  (∀ y : ℝ, y > 0 → 13 = y^2 + 1/y^2 → x + 1/x ≥ y + 1/y) → x + 1/x = Real.sqrt 15 :=
sorry

end max_value_of_x_plus_inverse_l750_75072


namespace watch_cost_price_l750_75021

theorem watch_cost_price (loss_percent : ℚ) (gain_percent : ℚ) (price_difference : ℚ) :
  loss_percent = 16 →
  gain_percent = 4 →
  price_difference = 140 →
  ∃ (cost_price : ℚ),
    (cost_price * (1 - loss_percent / 100)) + price_difference = cost_price * (1 + gain_percent / 100) ∧
    cost_price = 700 :=
by sorry

end watch_cost_price_l750_75021


namespace right_triangle_area_l750_75073

/-- A right triangle with vertices at (0, 0), (0, 10), and (-10, 0), 
    and two points (-3, 7) and (-7, 3) on its hypotenuse. -/
structure RightTriangle where
  -- Define the vertices
  v1 : ℝ × ℝ := (0, 0)
  v2 : ℝ × ℝ := (0, 10)
  v3 : ℝ × ℝ := (-10, 0)
  -- Define the points on the hypotenuse
  p1 : ℝ × ℝ := (-3, 7)
  p2 : ℝ × ℝ := (-7, 3)
  -- Ensure the triangle is right-angled
  is_right_angle : (v2.1 - v1.1) * (v3.1 - v1.1) + (v2.2 - v1.2) * (v3.2 - v1.2) = 0
  -- Ensure the points lie on the hypotenuse
  p1_on_hypotenuse : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p1 = (t * v2.1 + (1 - t) * v3.1, t * v2.2 + (1 - t) * v3.2)
  p2_on_hypotenuse : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ p2 = (s * v2.1 + (1 - s) * v3.1, s * v2.2 + (1 - s) * v3.2)

/-- The area of the right triangle is 50 square units. -/
theorem right_triangle_area (t : RightTriangle) : 
  (1/2) * abs (t.v2.1 * t.v3.2 - t.v3.1 * t.v2.2) = 50 := by
  sorry

end right_triangle_area_l750_75073


namespace condition_relation_l750_75014

theorem condition_relation (a : ℝ) :
  (∀ a, a > 0 → a^2 + a ≥ 0) ∧
  (∃ a, a^2 + a ≥ 0 ∧ ¬(a > 0)) :=
by sorry

end condition_relation_l750_75014


namespace sqrt_simplification_exists_l750_75046

theorem sqrt_simplification_exists :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * Real.sqrt (b / a) = Real.sqrt (a * b) :=
sorry

end sqrt_simplification_exists_l750_75046


namespace intersection_implies_m_equals_three_l750_75070

def A (m : ℝ) : Set ℝ := {1, 2, m}
def B : Set ℝ := {3, 4}

theorem intersection_implies_m_equals_three (m : ℝ) :
  A m ∩ B = {3} → m = 3 := by
  sorry

end intersection_implies_m_equals_three_l750_75070


namespace max_of_min_values_l750_75088

/-- The function f(x) for a given m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 8*m + 4

/-- The minimum value of f(x) for a given m -/
def min_value (m : ℝ) : ℝ := f m m

/-- The function representing all minimum values of f(x) for different m -/
def g (m : ℝ) : ℝ := -m^2 + 8*m + 4

/-- The maximum of all minimum values of f(x) -/
theorem max_of_min_values :
  (⨆ (m : ℝ), min_value m) = 20 :=
sorry

end max_of_min_values_l750_75088


namespace base8_sum_l750_75087

/-- Base 8 representation of a three-digit number -/
def base8Rep (x y z : ℕ) : ℕ := 64 * x + 8 * y + z

/-- Proposition: If X, Y, and Z are non-zero distinct digits in base 8 such that 
    XYZ₈ + YZX₈ + ZXY₈ = XXX0₈, then Y + Z = 7₈ -/
theorem base8_sum (X Y Z : ℕ) 
  (h1 : X ≠ 0 ∧ Y ≠ 0 ∧ Z ≠ 0)
  (h2 : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z)
  (h3 : X < 8 ∧ Y < 8 ∧ Z < 8)
  (h4 : base8Rep X Y Z + base8Rep Y Z X + base8Rep Z X Y = 8 * base8Rep X X X) :
  Y + Z = 7 := by
sorry

end base8_sum_l750_75087


namespace quadratic_inequality_range_l750_75052

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3/8 < 0) → -3 < k ∧ k < 0 := by
  sorry

end quadratic_inequality_range_l750_75052


namespace basketball_game_students_l750_75039

/-- The total number of students in a basketball game given the number of 5th graders and a ratio of 6th to 5th graders -/
def total_students (fifth_graders : ℕ) (ratio : ℕ) : ℕ :=
  fifth_graders + ratio * fifth_graders

/-- Theorem stating that given 12 5th graders and 6 times as many 6th graders, the total number of students is 84 -/
theorem basketball_game_students :
  total_students 12 6 = 84 := by
  sorry

end basketball_game_students_l750_75039


namespace correct_recommendation_count_l750_75023

/-- Represents the number of recommendation spots for each language -/
structure RecommendationSpots :=
  (russian : Nat)
  (japanese : Nat)
  (spanish : Nat)

/-- Represents the number of male and female candidates -/
structure Candidates :=
  (males : Nat)
  (females : Nat)

/-- Calculate the number of different recommendation plans -/
def countRecommendationPlans (spots : RecommendationSpots) (candidates : Candidates) : Nat :=
  sorry

theorem correct_recommendation_count :
  let spots := RecommendationSpots.mk 2 2 1
  let candidates := Candidates.mk 3 2
  countRecommendationPlans spots candidates = 24 :=
by sorry

end correct_recommendation_count_l750_75023


namespace goods_train_speed_l750_75081

/-- The speed of a goods train passing a woman in an opposite moving train -/
theorem goods_train_speed
  (woman_train_speed : ℝ)
  (passing_time : ℝ)
  (goods_train_length : ℝ)
  (h1 : woman_train_speed = 25)
  (h2 : passing_time = 3)
  (h3 : goods_train_length = 140) :
  ∃ (goods_train_speed : ℝ),
    goods_train_speed = 143 ∧
    (goods_train_length / passing_time) * 3.6 = woman_train_speed + goods_train_speed :=
by sorry

end goods_train_speed_l750_75081


namespace book_sale_loss_percentage_l750_75019

/-- Given two books with specified costs and selling conditions, prove the loss percentage on the first book. -/
theorem book_sale_loss_percentage
  (total_cost : ℝ)
  (cost_book1 : ℝ)
  (gain_percentage : ℝ)
  (h_total_cost : total_cost = 480)
  (h_cost_book1 : cost_book1 = 280)
  (h_gain_percentage : gain_percentage = 19)
  (h_same_selling_price : ∃ (selling_price : ℝ),
    selling_price = cost_book1 * (1 - (loss_percentage / 100)) ∧
    selling_price = (total_cost - cost_book1) * (1 + (gain_percentage / 100)))
  : ∃ (loss_percentage : ℝ), loss_percentage = 15 := by
  sorry


end book_sale_loss_percentage_l750_75019


namespace brocard_angle_inequalities_l750_75033

/-- The Brocard angle of a triangle -/
def brocard_angle (α β γ : ℝ) : ℝ := sorry

/-- Theorem: Brocard angle inequalities -/
theorem brocard_angle_inequalities (α β γ : ℝ) (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) 
  (hsum : α + β + γ = π) :
  let φ := brocard_angle α β γ
  (φ^3 ≤ (α - φ) * (β - φ) * (γ - φ)) ∧ (8 * φ^3 ≤ α * β * γ) := by sorry

end brocard_angle_inequalities_l750_75033


namespace complex_rational_sum_l750_75048

def separate_and_sum (a b c d : ℚ) : ℚ :=
  let int_part := a.floor + b.floor + c.floor + d.floor
  let frac_part := (a - a.floor) + (b - b.floor) + (c - c.floor) + (d - d.floor)
  int_part + frac_part

theorem complex_rational_sum :
  separate_and_sum (-206) (401 + 3/4) (-204 - 2/3) (-1 - 1/2) = -10 - 5/12 :=
by sorry

end complex_rational_sum_l750_75048


namespace cricketer_wickets_before_match_l750_75084

/-- Represents a cricketer's bowling statistics -/
structure CricketerStats where
  wickets : ℕ
  runs : ℕ
  avg : ℚ

/-- Calculates the new average after a match -/
def newAverage (stats : CricketerStats) (newWickets : ℕ) (newRuns : ℕ) : ℚ :=
  (stats.runs + newRuns) / (stats.wickets + newWickets)

theorem cricketer_wickets_before_match 
  (stats : CricketerStats)
  (h1 : stats.avg = 12.4)
  (h2 : newAverage stats 5 26 = 12) :
  stats.wickets = 85 := by
  sorry

end cricketer_wickets_before_match_l750_75084


namespace parabola_directrix_l750_75027

/-- The directrix of the parabola y = 3x^2 + 6x + 5 is y = 23/12 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), y = 3 * x^2 + 6 * x + 5 → 
  ∃ (k : ℝ), k = 23/12 ∧ (∀ (x₀ : ℝ), (x - x₀)^2 = 4 * (1/12) * (y - k)) :=
by sorry

end parabola_directrix_l750_75027


namespace solutions_of_equation_l750_75060

theorem solutions_of_equation (x : ℝ) : x * (x - 1) = x ↔ x = 0 ∨ x = 2 := by sorry

end solutions_of_equation_l750_75060


namespace orangeade_price_day2_l750_75096

structure Orangeade where
  orange_juice : ℝ
  water : ℝ
  price_per_glass : ℝ
  glasses_sold : ℝ

def revenue (o : Orangeade) : ℝ := o.price_per_glass * o.glasses_sold

theorem orangeade_price_day2 (day1 day2 : Orangeade) :
  day1.orange_juice > 0 →
  day1.orange_juice = day1.water →
  day2.orange_juice = day1.orange_juice →
  day2.water = 2 * day1.water →
  day1.price_per_glass = 0.9 →
  revenue day1 = revenue day2 →
  day2.glasses_sold = (3/2) * day1.glasses_sold →
  day2.price_per_glass = 0.6 := by
  sorry

end orangeade_price_day2_l750_75096


namespace safari_park_animal_difference_l750_75054

theorem safari_park_animal_difference :
  let safari_lions : ℕ := 100
  let safari_snakes : ℕ := safari_lions / 2
  let savanna_lions : ℕ := safari_lions * 2
  let savanna_snakes : ℕ := safari_snakes * 3
  let safari_giraffes : ℕ := safari_snakes - (savanna_lions + savanna_snakes + safari_giraffes + 20 - 410)
  safari_snakes - safari_giraffes = 10 := by
  sorry

end safari_park_animal_difference_l750_75054


namespace kozlov_inequality_l750_75098

theorem kozlov_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end kozlov_inequality_l750_75098


namespace range_of_a_l750_75044

theorem range_of_a (a b c : ℝ) 
  (sum_eq : a + b + c = 2)
  (sum_sq_eq : a^2 + b^2 + c^2 = 4)
  (order : a > b ∧ b > c) :
  a ∈ Set.Ioo (2/3) 2 :=
sorry

end range_of_a_l750_75044


namespace energy_usage_is_96_watts_l750_75050

/-- Calculate total energy usage for three lights over a given time period -/
def totalEnergyUsage (baseWatts : ℕ) (hours : ℕ) : ℕ :=
  let lightA := baseWatts * hours
  let lightB := 3 * lightA
  let lightC := 4 * lightA
  lightA + lightB + lightC

/-- Theorem: The total energy usage for the given scenario is 96 watts -/
theorem energy_usage_is_96_watts :
  totalEnergyUsage 6 2 = 96 := by sorry

end energy_usage_is_96_watts_l750_75050


namespace quadratic_solution_set_l750_75047

/-- Given a quadratic function f(x) = x^2 + bx + c, 
    this theorem states that if the solution set of f(x) < 0 
    is the open interval (1, 3), then b + c = -1. -/
theorem quadratic_solution_set (b c : ℝ) : 
  ({x : ℝ | x^2 + b*x + c < 0} = {x : ℝ | 1 < x ∧ x < 3}) → 
  b + c = -1 := by
  sorry

end quadratic_solution_set_l750_75047


namespace number_of_girls_in_class_l750_75045

theorem number_of_girls_in_class (total_students : ℕ) (girls_ratio : ℚ) : 
  total_students = 35 →
  girls_ratio = 0.4 →
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    girls = (girls_ratio * boys).floor ∧
    girls = 10 := by
  sorry

end number_of_girls_in_class_l750_75045


namespace log_2_base_10_bound_l750_75076

theorem log_2_base_10_bound (h1 : 10^3 = 1000) (h2 : 10^5 = 100000)
  (h3 : 2^12 = 4096) (h4 : 2^15 = 32768) (h5 : 2^17 = 131072) :
  5/17 < Real.log 2 / Real.log 10 := by
  sorry

end log_2_base_10_bound_l750_75076


namespace unpainted_cubes_in_6x6x6_l750_75058

/-- Represents a cube composed of unit cubes -/
structure Cube where
  size : Nat
  total_units : Nat
  painted_per_face : Nat

/-- Calculates the number of unpainted unit cubes in a cube with painted cross patterns on each face -/
def unpainted_cubes (c : Cube) : Nat :=
  c.total_units - (c.painted_per_face * 6 - 24 - 12)

/-- Theorem stating the number of unpainted cubes in the specific 6x6x6 cube -/
theorem unpainted_cubes_in_6x6x6 :
  let c : Cube := { size := 6, total_units := 216, painted_per_face := 10 }
  unpainted_cubes c = 180 := by
  sorry

end unpainted_cubes_in_6x6x6_l750_75058


namespace james_shoe_purchase_l750_75090

theorem james_shoe_purchase (price1 price2 : ℝ) : 
  price1 = 40 →
  price2 = 60 →
  let cheaper_price := min price1 price2
  let discounted_price2 := price2 - cheaper_price / 2
  let total_before_extra_discount := price1 + discounted_price2
  let extra_discount := total_before_extra_discount / 4
  let final_price := total_before_extra_discount - extra_discount
  final_price = 45 := by sorry

end james_shoe_purchase_l750_75090


namespace inequality_solution_set_l750_75000

theorem inequality_solution_set (x : ℝ) : (3 * x - 1) / (2 - x) ≥ 1 ↔ 3 / 4 ≤ x ∧ x < 2 := by
  sorry

end inequality_solution_set_l750_75000


namespace john_net_profit_l750_75078

def gross_income : ℝ := 30000
def car_purchase_price : ℝ := 20000
def monthly_maintenance_cost : ℝ := 300
def annual_insurance_cost : ℝ := 1200
def tire_replacement_cost : ℝ := 400
def car_trade_in_value : ℝ := 6000
def tax_rate : ℝ := 0.15

def total_maintenance_cost : ℝ := monthly_maintenance_cost * 12
def car_depreciation : ℝ := car_purchase_price - car_trade_in_value
def total_expenses : ℝ := total_maintenance_cost + annual_insurance_cost + tire_replacement_cost + car_depreciation
def taxes : ℝ := tax_rate * gross_income
def net_profit : ℝ := gross_income - total_expenses - taxes

theorem john_net_profit : net_profit = 6300 := by
  sorry

end john_net_profit_l750_75078


namespace bottom_face_points_l750_75017

/-- Represents the number of points on each face of a cube -/
structure CubePoints where
  front : ℕ
  back : ℕ
  left : ℕ
  right : ℕ
  top : ℕ
  bottom : ℕ

/-- Theorem stating the number of points on the bottom face of the cube -/
theorem bottom_face_points (c : CubePoints) 
  (opposite_sum : c.front + c.back = 13 ∧ c.left + c.right = 13 ∧ c.top + c.bottom = 13)
  (front_left_top_sum : c.front + c.left + c.top = 16)
  (top_right_back_sum : c.top + c.right + c.back = 24) :
  c.bottom = 6 := by
  sorry

end bottom_face_points_l750_75017


namespace min_value_quadratic_min_value_achieved_min_value_points_l750_75062

theorem min_value_quadratic (x y : ℝ) : 2*x^2 + 2*y^2 - 8*x + 6*y + 28 ≥ 10.5 :=
by sorry

theorem min_value_achieved : ∃ (x y : ℝ), 2*x^2 + 2*y^2 - 8*x + 6*y + 28 = 10.5 :=
by sorry

theorem min_value_points : 2*2^2 + 2*(-3/2)^2 - 8*2 + 6*(-3/2) + 28 = 10.5 :=
by sorry

end min_value_quadratic_min_value_achieved_min_value_points_l750_75062


namespace largest_quantity_l750_75079

theorem largest_quantity (a b c d e : ℝ) 
  (eq1 : a = b + 3)
  (eq2 : b = c - 4)
  (eq3 : c = d + 5)
  (eq4 : d = e - 6) :
  e ≥ a ∧ e ≥ b ∧ e ≥ c ∧ e ≥ d := by
  sorry

end largest_quantity_l750_75079


namespace tree_planting_theorem_l750_75022

/-- The number of trees planted by 4th graders -/
def trees_4th : ℕ := 30

/-- The number of trees planted by 5th graders -/
def trees_5th : ℕ := 2 * trees_4th

/-- The number of trees planted by 6th graders -/
def trees_6th : ℕ := 3 * trees_5th - 30

/-- The total number of trees planted by all grades -/
def total_trees : ℕ := trees_4th + trees_5th + trees_6th

theorem tree_planting_theorem : total_trees = 240 := by
  sorry

end tree_planting_theorem_l750_75022


namespace arithmetic_sequence_a7_l750_75061

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) (h_arith : ArithmeticSequence a)
    (h_a4 : a 4 = 4) (h_sum : a 3 + a 8 = 5) : a 7 = 1 := by
  sorry

end arithmetic_sequence_a7_l750_75061


namespace fran_required_speed_l750_75095

/-- Calculates the required average speed for Fran to travel the same distance as Joann -/
theorem fran_required_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5) :
  (joann_speed * joann_time) / fran_time = 120 / 7 :=
by sorry

end fran_required_speed_l750_75095
