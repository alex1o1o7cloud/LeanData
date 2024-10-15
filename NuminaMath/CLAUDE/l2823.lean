import Mathlib

namespace NUMINAMATH_CALUDE_newscast_advertising_time_l2823_282327

theorem newscast_advertising_time (total_time national_news international_news sports weather : ℕ)
  (h_total : total_time = 30)
  (h_national : national_news = 12)
  (h_international : international_news = 5)
  (h_sports : sports = 5)
  (h_weather : weather = 2) :
  total_time - (national_news + international_news + sports + weather) = 6 := by
  sorry

end NUMINAMATH_CALUDE_newscast_advertising_time_l2823_282327


namespace NUMINAMATH_CALUDE_ab_value_l2823_282333

theorem ab_value (a b : ℝ) (h : Real.sqrt (a - 1) + b^2 - 4*b + 4 = 0) : a * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2823_282333


namespace NUMINAMATH_CALUDE_gloria_cypress_price_l2823_282348

/-- The amount Gloria gets for each cypress tree -/
def cypress_price : ℕ := sorry

theorem gloria_cypress_price :
  let cabin_price : ℕ := 129000
  let initial_cash : ℕ := 150
  let num_cypress : ℕ := 20
  let num_pine : ℕ := 600
  let num_maple : ℕ := 24
  let maple_price : ℕ := 300
  let pine_price : ℕ := 200
  let remaining_cash : ℕ := 350

  cypress_price * num_cypress + 
  pine_price * num_pine + 
  maple_price * num_maple + 
  initial_cash = 
  cabin_price + remaining_cash →
  
  cypress_price = 100 :=
by sorry

end NUMINAMATH_CALUDE_gloria_cypress_price_l2823_282348


namespace NUMINAMATH_CALUDE_jacob_calorie_limit_l2823_282377

/-- Jacob's calorie intake and limit problem -/
theorem jacob_calorie_limit :
  ∀ (breakfast lunch dinner total_eaten planned_limit : ℕ),
    breakfast = 400 →
    lunch = 900 →
    dinner = 1100 →
    total_eaten = breakfast + lunch + dinner →
    total_eaten = planned_limit + 600 →
    planned_limit = 1800 := by
  sorry

end NUMINAMATH_CALUDE_jacob_calorie_limit_l2823_282377


namespace NUMINAMATH_CALUDE_a_21_value_l2823_282397

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The property of being a geometric sequence -/
def IsGeometric (b : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem a_21_value
  (a b : Sequence)
  (h1 : a 1 = 1)
  (h2 : IsGeometric b)
  (h3 : ∀ n : ℕ, b n = a (n + 1) / a n)
  (h4 : b 10 * b 11 = 52) :
  a 21 = 4 := by
sorry

end NUMINAMATH_CALUDE_a_21_value_l2823_282397


namespace NUMINAMATH_CALUDE_unique_solution_gcd_system_l2823_282370

theorem unique_solution_gcd_system (a b c : ℕ+) :
  a + b = (Nat.gcd a b)^2 ∧
  b + c = (Nat.gcd b c)^2 ∧
  c + a = (Nat.gcd c a)^2 →
  a = 2 ∧ b = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_gcd_system_l2823_282370


namespace NUMINAMATH_CALUDE_f_solutions_when_a_neg_one_f_monotonic_increasing_iff_f_max_min_when_a_one_l2823_282395

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (x - 1) * |x - a|

-- Part 1
theorem f_solutions_when_a_neg_one :
  ∀ x : ℝ, f (-1) x = 1 ↔ (x ≤ -1 ∨ x = 1) :=
sorry

-- Part 2
theorem f_monotonic_increasing_iff :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≥ 1/3 :=
sorry

-- Part 3
theorem f_max_min_when_a_one :
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f 1 x ≤ 1) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ f 1 x = 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f 1 x ≥ -1) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ f 1 x = -1) :=
sorry

end NUMINAMATH_CALUDE_f_solutions_when_a_neg_one_f_monotonic_increasing_iff_f_max_min_when_a_one_l2823_282395


namespace NUMINAMATH_CALUDE_f_inf_fixed_point_l2823_282325

variable {A : Type*} [Fintype A]
variable (f : A → A)

def f_n : ℕ → (Set A) → Set A
  | 0, S => S
  | n + 1, S => f '' (f_n n S)

def f_inf (S : Set A) : Set A :=
  ⋂ n, f_n f n S

theorem f_inf_fixed_point (S : Set A) :
  f '' (f_inf f S) = f_inf f S := by sorry

end NUMINAMATH_CALUDE_f_inf_fixed_point_l2823_282325


namespace NUMINAMATH_CALUDE_bobs_final_score_l2823_282362

/-- Bob's math knowledge competition score calculation -/
theorem bobs_final_score :
  let points_per_correct : ℕ := 5
  let points_per_incorrect : ℕ := 2
  let correct_answers : ℕ := 18
  let incorrect_answers : ℕ := 2
  let total_score := points_per_correct * correct_answers - points_per_incorrect * incorrect_answers
  total_score = 86 := by
  sorry

end NUMINAMATH_CALUDE_bobs_final_score_l2823_282362


namespace NUMINAMATH_CALUDE_complex_fraction_product_l2823_282355

theorem complex_fraction_product (a b : ℝ) : 
  (1 + Complex.I) / (1 - Complex.I) = Complex.mk a b → a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_product_l2823_282355


namespace NUMINAMATH_CALUDE_negation_equivalence_l2823_282369

theorem negation_equivalence :
  (¬ ∃ x : ℝ, Real.exp x > x) ↔ (∀ x : ℝ, Real.exp x ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2823_282369


namespace NUMINAMATH_CALUDE_s_tends_to_infinity_l2823_282334

/-- Sum of digits in the decimal expansion of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- s_n is the sum of digits in the decimal expansion of 2^n -/
def s (n : ℕ) : ℕ := sum_of_digits (2^n)

/-- The sequence (s_n) tends to infinity -/
theorem s_tends_to_infinity : ∀ k : ℕ, ∃ N : ℕ, ∀ n ≥ N, s n ≥ k := by
  sorry

end NUMINAMATH_CALUDE_s_tends_to_infinity_l2823_282334


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l2823_282360

/-- A quadratic function with vertex form parameters -/
def quad_vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

/-- A quadratic function in standard form -/
def quad_standard_form (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem stating the equivalence of the quadratic function with given properties -/
theorem quadratic_equivalence :
  ∃ (a b c : ℝ),
    (∀ x, quad_vertex_form (1/2) 4 3 x = quad_standard_form a b c x) ∧
    quad_standard_form a b c 2 = 5 ∧
    a = 1/2 ∧ b = -4 ∧ c = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l2823_282360


namespace NUMINAMATH_CALUDE_highest_divisible_digit_l2823_282381

theorem highest_divisible_digit : 
  ∃ (a : ℕ), a ≤ 9 ∧ 
  (∀ (b : ℕ), b ≤ 9 → 365 * 100 * b + 16 ≡ 0 [MOD 8] → b ≤ a) ∧
  (365 * 100 * a + 16 ≡ 0 [MOD 8]) :=
by sorry

end NUMINAMATH_CALUDE_highest_divisible_digit_l2823_282381


namespace NUMINAMATH_CALUDE_jesse_blocks_l2823_282321

theorem jesse_blocks (cityscape farmhouse zoo fence1 fence2 fence3 left : ℕ) 
  (h1 : cityscape = 80)
  (h2 : farmhouse = 123)
  (h3 : zoo = 95)
  (h4 : fence1 = 57)
  (h5 : fence2 = 43)
  (h6 : fence3 = 62)
  (h7 : left = 84) :
  cityscape + farmhouse + zoo + fence1 + fence2 + fence3 + left = 544 := by
  sorry

end NUMINAMATH_CALUDE_jesse_blocks_l2823_282321


namespace NUMINAMATH_CALUDE_find_b_value_l2823_282303

/-- Given the equation a * b * c = ( √ ( a + 2 ) ( b + 3 ) ) / ( c + 1 ),
    when a = 6, c = 3, and the left-hand side of the equation equals 3,
    prove that b = 15. -/
theorem find_b_value (a b c : ℝ) :
  a = 6 →
  c = 3 →
  a * b * c = ( Real.sqrt ((a + 2) * (b + 3)) ) / (c + 1) →
  a * b * c = 3 →
  b = 15 := by
  sorry


end NUMINAMATH_CALUDE_find_b_value_l2823_282303


namespace NUMINAMATH_CALUDE_tangent_line_of_odd_cubic_l2823_282374

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = x^3 + (a-1)x^2 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (a-1)*x^2 + a*x

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*(a-1)*x + a

/-- The tangent line equation at (0,0) is y = mx, where m is the slope at x = 0 -/
def TangentLineAt0 (f : ℝ → ℝ) (f' : ℝ → ℝ) : ℝ → ℝ := fun x ↦ (f' 0) * x

theorem tangent_line_of_odd_cubic (a : ℝ) :
  IsOdd (f a) → TangentLineAt0 (f a) (f' a) = fun x ↦ x := by sorry

end NUMINAMATH_CALUDE_tangent_line_of_odd_cubic_l2823_282374


namespace NUMINAMATH_CALUDE_sum_and_average_of_squares_of_multiples_of_7_l2823_282372

def multiples_of_7 (n : ℕ) : List ℕ :=
  List.range n |>.map (· * 7 + 7)

def sum_of_squares (lst : List ℕ) : ℕ :=
  lst.map (· ^ 2) |>.sum

theorem sum_and_average_of_squares_of_multiples_of_7 :
  let lst := multiples_of_7 10
  let sum := sum_of_squares lst
  let avg := (sum : ℚ) / 10
  sum = 16865 ∧ avg = 1686.5 := by sorry

end NUMINAMATH_CALUDE_sum_and_average_of_squares_of_multiples_of_7_l2823_282372


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2823_282320

theorem triangle_abc_properties (A B C : Real) (R : Real) (BC AC : Real) :
  0 < R →
  0 < BC →
  0 < AC →
  C = 3 * Real.pi / 4 →
  Real.sin (A + C) = (BC / R) * Real.cos (A + B) →
  (1 / 2) * BC * AC * Real.sin C = 1 →
  (BC * AC = AC * (2 * BC)) ∧ 
  (AC * BC = A + B) ∧
  AC ^ 2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2823_282320


namespace NUMINAMATH_CALUDE_min_sum_abcd_l2823_282385

theorem min_sum_abcd (a b c d : ℕ) (h : a * b + b * c + c * d + d * a = 707) :
  a + b + c + d ≥ 108 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_abcd_l2823_282385


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l2823_282349

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log x

theorem derivative_f_at_one :
  deriv f 1 = Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l2823_282349


namespace NUMINAMATH_CALUDE_nicholas_crackers_l2823_282368

theorem nicholas_crackers (marcus_crackers : ℕ) (mona_crackers : ℕ) (nicholas_crackers : ℕ)
  (h1 : marcus_crackers = 27)
  (h2 : marcus_crackers = 3 * mona_crackers)
  (h3 : nicholas_crackers = mona_crackers + 6) :
  nicholas_crackers = 15 := by
sorry

end NUMINAMATH_CALUDE_nicholas_crackers_l2823_282368


namespace NUMINAMATH_CALUDE_complex_modulus_proof_l2823_282380

theorem complex_modulus_proof (z z₁ z₂ : ℂ) 
  (h₁ : z₁ ≠ z₂)
  (h₂ : z₁^2 = -2 - 2 * Complex.I * Real.sqrt 3)
  (h₃ : z₂^2 = -2 - 2 * Complex.I * Real.sqrt 3)
  (h₄ : Complex.abs (z - z₁) = 4)
  (h₅ : Complex.abs (z - z₂) = 4) :
  Complex.abs z = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_proof_l2823_282380


namespace NUMINAMATH_CALUDE_problem_solution_l2823_282305

/-- Proposition A: The solution set of x^2 + (a-1)x + a^2 ≤ 0 with respect to x is empty -/
def proposition_a (a : ℝ) : Prop :=
  ∀ x, x^2 + (a-1)*x + a^2 > 0

/-- Proposition B: The function y = (2a^2 - a)^x is increasing -/
def proposition_b (a : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → (2*a^2 - a)^x₁ < (2*a^2 - a)^x₂

/-- The set of real numbers a for which at least one of A or B is true -/
def at_least_one_true (a : ℝ) : Prop :=
  proposition_a a ∨ proposition_b a

/-- The set of real numbers a for which exactly one of A or B is true -/
def exactly_one_true (a : ℝ) : Prop :=
  (proposition_a a ∧ ¬proposition_b a) ∨ (¬proposition_a a ∧ proposition_b a)

theorem problem_solution :
  (∀ a, at_least_one_true a ↔ (a < -1/2 ∨ a > 1/3)) ∧
  (∀ a, exactly_one_true a ↔ (1/3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1/2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2823_282305


namespace NUMINAMATH_CALUDE_largest_element_in_S_l2823_282332

def a : ℝ := -4

def S : Set ℝ := { -2 * a^2, 5 * a, 40 / a, 3 * a^2, 2 }

theorem largest_element_in_S : ∀ x ∈ S, x ≤ (3 * a^2) := by
  sorry

end NUMINAMATH_CALUDE_largest_element_in_S_l2823_282332


namespace NUMINAMATH_CALUDE_vector_problem_l2823_282359

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (2, 3)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), v.1 * w.2 = c * v.2 * w.1

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_problem (k m : ℝ) :
  (parallel (3 • a - b) (a + k • b) → k = -1/3) ∧
  (perpendicular a (m • a - b) → m = -4/5) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l2823_282359


namespace NUMINAMATH_CALUDE_rocket_arrangements_l2823_282364

def word : String := "ROCKET"

theorem rocket_arrangements : 
  (∃ (c : Char), c ∈ word.data ∧ 
    (word.data.count c = 2) ∧ 
    (∀ (d : Char), d ∈ word.data ∧ d ≠ c → word.data.count d = 1)) →
  (Nat.factorial (word.length + 1) / 2 = 2520) :=
by sorry

end NUMINAMATH_CALUDE_rocket_arrangements_l2823_282364


namespace NUMINAMATH_CALUDE_intersection_of_planes_and_line_l2823_282365

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relation for intersection between planes
variable (intersect : Plane → Plane → Prop)

-- Define the relation for perpendicularity between planes
variable (perpendicular : Plane → Plane → Prop)

-- Define the relation for a line lying in a plane
variable (lies_in : Line → Plane → Prop)

-- Define the relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Define the relation for perpendicular lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the theorem
theorem intersection_of_planes_and_line 
  (α β : Plane) (m : Line)
  (h1 : intersect α β)
  (h2 : ¬ perpendicular α β)
  (h3 : lies_in m α) :
  (∃ (n : Line), lies_in n β ∧ ¬ (∀ (n : Line), lies_in n β → parallel m n)) ∧
  (∃ (p : Line), lies_in p β ∧ perpendicular_lines m p) :=
sorry

end NUMINAMATH_CALUDE_intersection_of_planes_and_line_l2823_282365


namespace NUMINAMATH_CALUDE_unpainted_cubes_count_l2823_282363

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_units : Nat
  unpainted_center_size : Nat

/-- Calculates the number of unpainted unit cubes in the given PaintedCube -/
def count_unpainted_cubes (cube : PaintedCube) : Nat :=
  sorry

/-- Theorem stating that a 6x6x6 cube with 2x2 unpainted centers has 72 unpainted unit cubes -/
theorem unpainted_cubes_count (cube : PaintedCube) 
  (h1 : cube.size = 6)
  (h2 : cube.total_units = 216)
  (h3 : cube.unpainted_center_size = 2) : 
  count_unpainted_cubes cube = 72 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_count_l2823_282363


namespace NUMINAMATH_CALUDE_solve_apples_problem_l2823_282394

def apples_problem (initial_apples : ℕ) (given_to_father : ℕ) (apples_per_person : ℕ) : Prop :=
  let remaining_apples := initial_apples - given_to_father
  let friends := (remaining_apples - apples_per_person) / apples_per_person
  friends = 4

theorem solve_apples_problem :
  apples_problem 55 10 9 :=
by sorry

end NUMINAMATH_CALUDE_solve_apples_problem_l2823_282394


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l2823_282378

/-- A geometric sequence with a_1 = 1 and a_9 = 3 has a_5 = √3 -/
theorem geometric_sequence_a5 (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →
  a 9 = 3 →
  a 5 = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l2823_282378


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_a_equals_two_l2823_282342

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Given two vectors m and n, if they are parallel, then the second component of n is 2 -/
theorem parallel_vectors_imply_a_equals_two (m n : ℝ × ℝ) 
    (hm : m = (2, 1)) 
    (hn : ∃ a : ℝ, n = (4, a)) 
    (h_parallel : are_parallel m n) : 
    n.2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_a_equals_two_l2823_282342


namespace NUMINAMATH_CALUDE_total_cost_calculation_l2823_282316

theorem total_cost_calculation (sandwich_price soda_price : ℚ) 
  (sandwich_quantity soda_quantity : ℕ) : 
  sandwich_price = 245/100 →
  soda_price = 87/100 →
  sandwich_quantity = 2 →
  soda_quantity = 4 →
  (sandwich_price * sandwich_quantity + soda_price * soda_quantity : ℚ) = 838/100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l2823_282316


namespace NUMINAMATH_CALUDE_employee_share_l2823_282340

theorem employee_share (total_profit : ℝ) (num_employees : ℕ) (employer_percentage : ℝ) :
  total_profit = 50 ∧ num_employees = 9 ∧ employer_percentage = 0.1 →
  (total_profit - (employer_percentage * total_profit)) / num_employees = 5 := by
  sorry

end NUMINAMATH_CALUDE_employee_share_l2823_282340


namespace NUMINAMATH_CALUDE_placemat_length_l2823_282393

theorem placemat_length (r : ℝ) (n : ℕ) (w : ℝ) (y : ℝ) : 
  r = 5 → n = 8 → w = 1 → 
  y = 2 * r * Real.sin ((π / n) / 2) →
  y = 10 * Real.sin (5 * π / 16) :=
by sorry

end NUMINAMATH_CALUDE_placemat_length_l2823_282393


namespace NUMINAMATH_CALUDE_triangle_problem_l2823_282358

theorem triangle_problem (A B C a b c : Real) (t : Real) :
  -- Conditions
  (A + B + C = π) →
  (2 * B = A + C) →
  (b = Real.sqrt 7) →
  (a = 3) →
  (t = Real.sin A * Real.sin C) →
  -- Conclusions
  (c = 4 ∧ t ≤ Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2823_282358


namespace NUMINAMATH_CALUDE_factorial_sum_remainder_20_l2823_282351

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_sum_remainder_20 :
  (factorial 1 + factorial 2 + factorial 3 + factorial 4) % 20 = 13 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_remainder_20_l2823_282351


namespace NUMINAMATH_CALUDE_equation_value_proof_l2823_282309

theorem equation_value_proof (x y z w : ℝ) 
  (eq1 : 4 * x * z + y * w = 4)
  (eq2 : (2 * x + y) * (2 * z + w) = 20) :
  x * w + y * z = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_value_proof_l2823_282309


namespace NUMINAMATH_CALUDE_city_C_highest_growth_l2823_282391

structure City where
  name : String
  pop1970 : ℕ
  pop1980 : ℕ

def cities : List City := [
  { name := "A", pop1970 := 40, pop1980 := 50 },
  { name := "B", pop1970 := 50, pop1980 := 70 },
  { name := "C", pop1970 := 70, pop1980 := 100 },
  { name := "D", pop1970 := 100, pop1980 := 130 },
  { name := "E", pop1970 := 120, pop1980 := 160 }
]

def growthRatio (city : City) : ℚ :=
  city.pop1980 / city.pop1970

theorem city_C_highest_growth :
  ∃ c ∈ cities, c.name = "C" ∧
  ∀ other ∈ cities, growthRatio c ≥ growthRatio other :=
by sorry

end NUMINAMATH_CALUDE_city_C_highest_growth_l2823_282391


namespace NUMINAMATH_CALUDE_marble_selection_ways_l2823_282314

def total_marbles : ℕ := 20
def red_marbles : ℕ := 3
def green_marbles : ℕ := 3
def blue_marbles : ℕ := 2
def special_marbles : ℕ := red_marbles + green_marbles + blue_marbles
def other_marbles : ℕ := total_marbles - special_marbles
def chosen_marbles : ℕ := 5
def chosen_special : ℕ := 2

theorem marble_selection_ways :
  (Nat.choose red_marbles 2 +
   Nat.choose green_marbles 2 +
   Nat.choose blue_marbles 2 +
   Nat.choose red_marbles 1 * Nat.choose green_marbles 1 +
   Nat.choose red_marbles 1 * Nat.choose blue_marbles 1 +
   Nat.choose green_marbles 1 * Nat.choose blue_marbles 1) *
  Nat.choose other_marbles (chosen_marbles - chosen_special) = 6160 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l2823_282314


namespace NUMINAMATH_CALUDE_average_difference_l2823_282347

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((10 + 80 + x) / 3) + 5 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l2823_282347


namespace NUMINAMATH_CALUDE_obtuse_triangle_consecutive_sides_l2823_282345

theorem obtuse_triangle_consecutive_sides :
  ∀ (a b c : ℕ), 
    (a < b) → 
    (b < c) → 
    (c = a + 2) → 
    (a^2 + b^2 < c^2) → 
    (a = 2 ∧ b = 3 ∧ c = 4) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_consecutive_sides_l2823_282345


namespace NUMINAMATH_CALUDE_min_value_theorem_l2823_282328

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  9 ≤ 4*a + b ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 1/b₀ = 1 ∧ 4*a₀ + b₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2823_282328


namespace NUMINAMATH_CALUDE_parallel_vectors_linear_combination_l2823_282317

/-- Given two parallel plane vectors a and b, prove their linear combination -/
theorem parallel_vectors_linear_combination (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, m]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  (2 • a + 3 • b : Fin 2 → ℝ) = ![-4, -8] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_linear_combination_l2823_282317


namespace NUMINAMATH_CALUDE_number_equation_l2823_282354

theorem number_equation (x : ℤ) : 8 * x + 64 = 336 ↔ x = 34 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l2823_282354


namespace NUMINAMATH_CALUDE_sum_of_numbers_l2823_282306

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 9375) (h4 : y / x = 15) : 
  x + y = 400 := by sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l2823_282306


namespace NUMINAMATH_CALUDE_sum_of_digits_3_plus_4_pow_17_l2823_282384

/-- The sum of the tens digit and the ones digit of (3+4)^17 in integer form is 7 -/
theorem sum_of_digits_3_plus_4_pow_17 : 
  let n : ℕ := (3 + 4)^17
  let tens_digit : ℕ := (n / 10) % 10
  let ones_digit : ℕ := n % 10
  tens_digit + ones_digit = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_3_plus_4_pow_17_l2823_282384


namespace NUMINAMATH_CALUDE_gardener_path_tiles_l2823_282396

def park_width : ℕ := 13
def park_length : ℕ := 19

theorem gardener_path_tiles :
  ∀ (avoid : ℕ), avoid = 1 →
  (park_width + park_length - Nat.gcd park_width park_length) - avoid = 30 := by
sorry

end NUMINAMATH_CALUDE_gardener_path_tiles_l2823_282396


namespace NUMINAMATH_CALUDE_binomial_18_6_l2823_282338

theorem binomial_18_6 : Nat.choose 18 6 = 4767 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_6_l2823_282338


namespace NUMINAMATH_CALUDE_kopeck_ruble_equivalence_l2823_282304

/-- Represents the denominations of coins available in kopecks -/
def coin_denominations : List ℕ := [1, 2, 5, 10, 20, 50, 100]

/-- Represents a collection of coins, where each natural number is the count of coins for the corresponding denomination -/
def Coins := List ℕ

/-- Calculates the total value of a collection of coins in kopecks -/
def total_value (coins : Coins) : ℕ :=
  List.sum (List.zipWith (· * ·) coins coin_denominations)

/-- Calculates the total number of coins in a collection -/
def total_count (coins : Coins) : ℕ :=
  List.sum coins

theorem kopeck_ruble_equivalence (k m : ℕ) (coins : Coins) 
    (h1 : total_count coins = k)
    (h2 : total_value coins = m) :
  ∃ (new_coins : Coins), total_count new_coins = m ∧ total_value new_coins = k * 100 := by
  sorry

end NUMINAMATH_CALUDE_kopeck_ruble_equivalence_l2823_282304


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l2823_282336

theorem y_in_terms_of_x (x y : ℝ) (h : y - 2*x = 5) : y = 2*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l2823_282336


namespace NUMINAMATH_CALUDE_added_number_after_doubling_l2823_282318

theorem added_number_after_doubling (x : ℝ) : 
  3 * (2 * 7 + x) = 69 → x = 9 := by
sorry

end NUMINAMATH_CALUDE_added_number_after_doubling_l2823_282318


namespace NUMINAMATH_CALUDE_propositions_truth_values_l2823_282388

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def is_solution (x : ℚ) : Prop := x^2 + x - 2 = 0

theorem propositions_truth_values :
  (is_prime 3 ∨ is_even 3) ∧
  ¬(is_prime 3 ∧ is_even 3) ∧
  ¬(¬is_prime 3) ∧
  (is_solution (-2) ∨ is_solution 1) ∧
  (is_solution (-2) ∧ is_solution 1) ∧
  ¬(¬is_solution (-2)) := by
  sorry

end NUMINAMATH_CALUDE_propositions_truth_values_l2823_282388


namespace NUMINAMATH_CALUDE_sophie_savings_l2823_282335

/-- The amount of money saved in a year by not buying dryer sheets -/
def money_saved_per_year (loads_per_week : ℕ) (sheets_per_load : ℕ) (sheets_per_box : ℕ) (cost_per_box : ℚ) : ℚ :=
  let sheets_per_week : ℕ := loads_per_week * sheets_per_load
  let sheets_per_year : ℕ := sheets_per_week * 52
  let boxes_per_year : ℚ := (sheets_per_year : ℚ) / (sheets_per_box : ℚ)
  boxes_per_year * cost_per_box

/-- Theorem stating the amount of money saved per year -/
theorem sophie_savings : money_saved_per_year 4 1 104 (11/2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_sophie_savings_l2823_282335


namespace NUMINAMATH_CALUDE_coins_missing_fraction_l2823_282352

-- Define the initial number of coins
variable (x : ℚ)

-- Define the fractions based on the problem conditions
def lost_fraction : ℚ := 1 / 3
def found_fraction : ℚ := 5 / 6
def spent_fraction : ℚ := 1 / 4

-- Define the fraction of coins still missing
def missing_fraction : ℚ := 
  spent_fraction + (lost_fraction - lost_fraction * found_fraction)

-- Theorem to prove
theorem coins_missing_fraction : missing_fraction = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_coins_missing_fraction_l2823_282352


namespace NUMINAMATH_CALUDE_retirement_plan_ratio_l2823_282367

/-- Represents the number of workers in each category -/
structure WorkerCounts where
  men : ℕ
  women : ℕ
  withPlan : ℕ
  withoutPlan : ℕ

/-- Represents the percentages of workers in different categories -/
structure WorkerPercentages where
  womenWithoutPlan : ℚ
  menWithPlan : ℚ

/-- The main theorem about the ratio of workers without a retirement plan -/
theorem retirement_plan_ratio
  (counts : WorkerCounts)
  (percentages : WorkerPercentages)
  (h1 : counts.men = 120)
  (h2 : counts.women = 180)
  (h3 : percentages.womenWithoutPlan = 3/5)
  (h4 : percentages.menWithPlan = 2/5)
  (h5 : counts.men + counts.women = counts.withPlan + counts.withoutPlan)
  (h6 : percentages.womenWithoutPlan * counts.withoutPlan = counts.women - percentages.menWithPlan * counts.withPlan)
  (h7 : (1 - percentages.womenWithoutPlan) * counts.withoutPlan = counts.men - percentages.menWithPlan * counts.withPlan) :
  counts.withoutPlan * 13 = (counts.withPlan + counts.withoutPlan) * 9 :=
sorry

end NUMINAMATH_CALUDE_retirement_plan_ratio_l2823_282367


namespace NUMINAMATH_CALUDE_shaded_area_approx_l2823_282310

/-- The area of a 4 x 6 rectangle minus a circle with diameter 2 is approximately 21 -/
theorem shaded_area_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  (4 * 6 : ℝ) - Real.pi * (2 / 2)^2 = 21 + ε := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_approx_l2823_282310


namespace NUMINAMATH_CALUDE_expand_expression_l2823_282357

theorem expand_expression (x : ℝ) : 20 * (3 * x - 4) = 60 * x - 80 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2823_282357


namespace NUMINAMATH_CALUDE_range_of_k_l2823_282382

theorem range_of_k (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 + c^2 = 16) (h2 : b^2 + c^2 = 25) :
  let k := a^2 + b^2
  9 < k ∧ k < 41 := by sorry

end NUMINAMATH_CALUDE_range_of_k_l2823_282382


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2823_282311

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : GeometricSequence a) 
  (h_sum : a 4 + a 8 = -3) : 
  a 6 * (a 2 + 2 * a 6 + a 10) = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2823_282311


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2823_282353

/-- The eccentricity of a hyperbola with equation x^2/a^2 - y^2/b^2 = 1 (a > 0, b > 0)
    is √2, given that one of its asymptotes is parallel to the line x - y + 3 = 0. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_asymptote : ∃ (k : ℝ), b / a = k ∧ 1 = k) : 
    Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2823_282353


namespace NUMINAMATH_CALUDE_pictures_per_album_l2823_282356

/-- Given the number of pictures uploaded from a phone and a camera, and the number of albums,
    prove that the number of pictures in each album is correct. -/
theorem pictures_per_album
  (phone_pics : ℕ)
  (camera_pics : ℕ)
  (num_albums : ℕ)
  (h1 : phone_pics = 35)
  (h2 : camera_pics = 5)
  (h3 : num_albums = 5)
  (h4 : num_albums > 0) :
  (phone_pics + camera_pics) / num_albums = 8 := by
sorry

end NUMINAMATH_CALUDE_pictures_per_album_l2823_282356


namespace NUMINAMATH_CALUDE_range_of_m_l2823_282390

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | -1 < x ∧ x < m + 1}

-- State the theorem
theorem range_of_m (m : ℝ) : B m ⊂ A → -2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2823_282390


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l2823_282350

theorem complex_fraction_evaluation : 
  1 / (3 + 1 / (3 + 1 / (3 - 1 / (3 + 1 / (2 * (3 + 2 / 5)))))) = 968/3191 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l2823_282350


namespace NUMINAMATH_CALUDE_line_segment_param_sum_squares_l2823_282379

/-- 
Given a line segment connecting (-2,7) and (3,11), parameterized by x = at + b and y = ct + d 
where 0 ≤ t ≤ 1 and t = 0 corresponds to (-2,7), the sum a^2 + b^2 + c^2 + d^2 equals 94.
-/
theorem line_segment_param_sum_squares : 
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    (a * t + b = -2 * (1 - t) + 3 * t) ∧ 
    (c * t + d = 7 * (1 - t) + 11 * t)) →
  b = -2 →
  d = 7 →
  a^2 + b^2 + c^2 + d^2 = 94 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_param_sum_squares_l2823_282379


namespace NUMINAMATH_CALUDE_min_value_abs_plus_two_l2823_282329

theorem min_value_abs_plus_two (a : ℚ) : |a - 1| + 2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abs_plus_two_l2823_282329


namespace NUMINAMATH_CALUDE_inequality_solution_l2823_282392

theorem inequality_solution (m : ℝ) : 
  (∃ (a : ℝ), a = 5 ∧ 
   ∃ (x : ℝ), |x - 1| - |x + m| ≥ a ∧ 
   ∀ (b : ℝ), (∃ (y : ℝ), |y - 1| - |y + m| ≥ b) → b ≤ a) → 
  m = 4 ∨ m = -6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2823_282392


namespace NUMINAMATH_CALUDE_cookies_left_to_take_home_l2823_282337

-- Define the initial number of cookies
def initial_cookies : ℕ := 120

-- Define the number of cookies in a dozen
def cookies_per_dozen : ℕ := 12

-- Define the number of dozens sold in the morning
def morning_dozens_sold : ℕ := 3

-- Define the number of cookies sold during lunch
def lunch_cookies_sold : ℕ := 57

-- Define the number of cookies sold in the afternoon
def afternoon_cookies_sold : ℕ := 16

-- Theorem statement
theorem cookies_left_to_take_home :
  initial_cookies - (morning_dozens_sold * cookies_per_dozen + lunch_cookies_sold + afternoon_cookies_sold) = 11 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_to_take_home_l2823_282337


namespace NUMINAMATH_CALUDE_function_difference_l2823_282301

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (k : ℝ) (x : ℝ) : ℝ := x^2 + k * x - 8

-- State the theorem
theorem function_difference (k : ℝ) : 
  f 5 - g k 5 = 20 → k = 53 / 5 := by
  sorry

end NUMINAMATH_CALUDE_function_difference_l2823_282301


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2823_282344

/-- The line equation passing through a fixed point -/
def line_equation (k x y : ℝ) : Prop :=
  (k + 1) * x - (2 * k - 1) * y + 3 * k = 0

/-- Theorem stating that the line passes through (-1, 1) for all k -/
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, line_equation k (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2823_282344


namespace NUMINAMATH_CALUDE_john_calculation_l2823_282339

theorem john_calculation (n : ℕ) (h : n = 40) : n^2 - (n - 1)^2 = 2*n - 1 := by
  sorry

#check john_calculation

end NUMINAMATH_CALUDE_john_calculation_l2823_282339


namespace NUMINAMATH_CALUDE_third_root_of_polynomial_l2823_282387

/-- Given a polynomial ax^3 + (a + 3b)x^2 + (b - 4a)x + (10 - a) with roots -3 and 4,
    prove that the third root is -17/10 -/
theorem third_root_of_polynomial (a b : ℝ) :
  (∀ x : ℝ, a * x^3 + (a + 3*b) * x^2 + (b - 4*a) * x + (10 - a) = 0 ↔ x = -3 ∨ x = 4 ∨ x = -17/10) :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_polynomial_l2823_282387


namespace NUMINAMATH_CALUDE_pool_wall_area_ratio_l2823_282373

theorem pool_wall_area_ratio :
  let pool_radius : ℝ := 20
  let wall_width : ℝ := 4
  let pool_area := π * pool_radius^2
  let total_area := π * (pool_radius + wall_width)^2
  let wall_area := total_area - pool_area
  wall_area / pool_area = 11 / 25 := by sorry

end NUMINAMATH_CALUDE_pool_wall_area_ratio_l2823_282373


namespace NUMINAMATH_CALUDE_coordinate_square_area_l2823_282330

/-- A square in the coordinate plane with y-coordinates between 3 and 8 -/
structure CoordinateSquare where
  lowest_y : ℝ
  highest_y : ℝ
  is_square : lowest_y = 3 ∧ highest_y = 8

/-- The area of a CoordinateSquare is 25 -/
theorem coordinate_square_area (s : CoordinateSquare) : (s.highest_y - s.lowest_y) ^ 2 = 25 :=
sorry

end NUMINAMATH_CALUDE_coordinate_square_area_l2823_282330


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2823_282386

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: A regular decagon has 35 diagonals -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2823_282386


namespace NUMINAMATH_CALUDE_organization_size_l2823_282312

/-- Represents a committee in the organization -/
def Committee := Fin 6

/-- Represents a member of the organization -/
structure Member where
  committees : Finset Committee
  member_in_three : committees.card = 3

/-- The organization with its members -/
structure Organization where
  members : Finset Member
  all_triples_covered : ∀ (c1 c2 c3 : Committee), c1 ≠ c2 → c2 ≠ c3 → c1 ≠ c3 →
    ∃! m : Member, m ∈ members ∧ c1 ∈ m.committees ∧ c2 ∈ m.committees ∧ c3 ∈ m.committees

theorem organization_size (org : Organization) : org.members.card = 20 := by
  sorry

end NUMINAMATH_CALUDE_organization_size_l2823_282312


namespace NUMINAMATH_CALUDE_daves_age_ratio_l2823_282376

theorem daves_age_ratio (D N : ℚ) : 
  (D > 0) → 
  (N > 0) → 
  (∃ (a b c d : ℚ), a + b + c + d = D) → -- Combined ages of four children equal D
  (D - N = 3 * (D - 4 * N)) → -- N years ago, Dave's age was thrice the sum of children's ages
  D / N = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_daves_age_ratio_l2823_282376


namespace NUMINAMATH_CALUDE_problem_statement_l2823_282343

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = -3) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = -5932 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2823_282343


namespace NUMINAMATH_CALUDE_exists_all_met_l2823_282300

-- Define a type for participants
variable (Participant : Type)

-- Define a relation for "has met"
variable (has_met : Participant → Participant → Prop)

-- Define the number of participants
variable (n : ℕ)

-- Assume there are at least 4 participants
variable (h_n : n ≥ 4)

-- Define the set of all participants
variable (participants : Finset Participant)

-- Assume the number of participants matches n
variable (h_card : participants.card = n)

-- State the condition that among any 4 participants, one has met the other 3
variable (h_four_met : ∀ (a b c d : Participant), a ∈ participants → b ∈ participants → c ∈ participants → d ∈ participants →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (has_met a b ∧ has_met a c ∧ has_met a d) ∨
  (has_met b a ∧ has_met b c ∧ has_met b d) ∨
  (has_met c a ∧ has_met c b ∧ has_met c d) ∨
  (has_met d a ∧ has_met d b ∧ has_met d c))

-- Theorem statement
theorem exists_all_met :
  ∃ (x : Participant), x ∈ participants ∧ ∀ (y : Participant), y ∈ participants → y ≠ x → has_met x y :=
sorry

end NUMINAMATH_CALUDE_exists_all_met_l2823_282300


namespace NUMINAMATH_CALUDE_min_S_value_l2823_282315

/-- Represents a 10x10 table arrangement of numbers from 1 to 100 -/
def Arrangement := Fin 10 → Fin 10 → Fin 100

/-- Checks if two positions in the table are adjacent -/
def isAdjacent (p1 p2 : Fin 10 × Fin 10) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Checks if an arrangement satisfies the adjacent sum condition -/
def satisfiesCondition (arr : Arrangement) (S : ℕ) : Prop :=
  ∀ p1 p2 : Fin 10 × Fin 10, isAdjacent p1 p2 →
    (arr p1.1 p1.2).val + (arr p2.1 p2.2).val ≤ S

/-- The main theorem stating the minimum value of S -/
theorem min_S_value :
  (∃ (arr : Arrangement), satisfiesCondition arr 106) ∧
  (∀ S : ℕ, S < 106 → ¬∃ (arr : Arrangement), satisfiesCondition arr S) :=
sorry

end NUMINAMATH_CALUDE_min_S_value_l2823_282315


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l2823_282375

/-- Given 10 people in an elevator with an average weight of 165 lbs, 
    prove that if an 11th person enters and increases the average weight to 170 lbs, 
    then the weight of the 11th person is 220 lbs. -/
theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℝ) 
  (new_avg_weight : ℝ) (new_person_weight : ℝ) :
  initial_people = 10 →
  initial_avg_weight = 165 →
  new_avg_weight = 170 →
  (initial_people * initial_avg_weight + new_person_weight) / (initial_people + 1) = new_avg_weight →
  new_person_weight = 220 :=
by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l2823_282375


namespace NUMINAMATH_CALUDE_equal_sum_and_product_sets_l2823_282307

theorem equal_sum_and_product_sets : ∃ (S₁ S₂ : Finset ℕ),
  S₁ ≠ S₂ ∧
  S₁.card = 8 ∧
  S₂.card = 8 ∧
  (S₁.sum id = S₁.prod id) ∧
  (S₂.sum id = S₂.prod id) :=
by
  sorry

end NUMINAMATH_CALUDE_equal_sum_and_product_sets_l2823_282307


namespace NUMINAMATH_CALUDE_intersection_distance_l2823_282319

/-- Given a line y = kx - 2 intersecting a parabola y^2 = 8x at two points,
    if the x-coordinate of the midpoint of these points is 2,
    then the distance between these points is 2√15. -/
theorem intersection_distance (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ ≠ x₂ ∧
    y₁^2 = 8*x₁ ∧ y₂^2 = 8*x₂ ∧
    y₁ = k*x₁ - 2 ∧ y₂ = k*x₂ - 2 ∧
    (x₁ + x₂) / 2 = 2) →
  ∃ A B : ℝ × ℝ, 
    A.1 ≠ B.1 ∧
    A.2^2 = 8*A.1 ∧ B.2^2 = 8*B.1 ∧
    A.2 = k*A.1 - 2 ∧ B.2 = k*B.1 - 2 ∧
    (A.1 + B.1) / 2 = 2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2 : ℝ) = 2 * (15^(1/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l2823_282319


namespace NUMINAMATH_CALUDE_log_sine_absolute_value_sum_l2823_282324

theorem log_sine_absolute_value_sum (x : ℝ) (θ : ℝ) 
  (h : Real.log x / Real.log 2 = 2 + Real.sin θ) : 
  |x + 1| + |x - 10| = 11 := by
  sorry

end NUMINAMATH_CALUDE_log_sine_absolute_value_sum_l2823_282324


namespace NUMINAMATH_CALUDE_system_solution_l2823_282308

theorem system_solution :
  let x : ℚ := -89/43
  let y : ℚ := -202/129
  (4 * x - 3 * y = -14) ∧ (5 * x + 7 * y = 3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2823_282308


namespace NUMINAMATH_CALUDE_students_not_playing_sports_l2823_282322

theorem students_not_playing_sports (total : ℕ) (basketball : ℕ) (volleyball : ℕ) (both : ℕ) : 
  total = 20 ∧ 
  basketball = total / 2 ∧ 
  volleyball = total * 2 / 5 ∧ 
  both = total / 10 → 
  total - (basketball + volleyball - both) = 4 :=
by sorry

end NUMINAMATH_CALUDE_students_not_playing_sports_l2823_282322


namespace NUMINAMATH_CALUDE_consecutive_color_draw_probability_l2823_282398

def numTan : ℕ := 4
def numPink : ℕ := 3
def numViolet : ℕ := 5
def totalChips : ℕ := numTan + numPink + numViolet

theorem consecutive_color_draw_probability :
  (numTan.factorial * numPink.factorial * numViolet.factorial) / totalChips.factorial = 1 / 27720 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_color_draw_probability_l2823_282398


namespace NUMINAMATH_CALUDE_inequality_proof_l2823_282346

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2823_282346


namespace NUMINAMATH_CALUDE_birdseed_supply_l2823_282383

/-- Represents a box of birdseed -/
structure BirdseedBox where
  totalAmount : ℕ
  typeAAmount : ℕ
  typeBAmount : ℕ

/-- Represents a bird's weekly seed consumption -/
structure BirdConsumption where
  totalAmount : ℕ
  typeAPercentage : ℚ
  typeBPercentage : ℚ

/-- The problem statement -/
theorem birdseed_supply (pantryBoxes : List BirdseedBox)
  (parrot cockatiel canary : BirdConsumption) :
  pantryBoxes.length = 5 →
  (pantryBoxes.map (·.typeAAmount)).sum ≥ 650 →
  (pantryBoxes.map (·.typeBAmount)).sum ≥ 675 →
  parrot.totalAmount = 100 ∧ parrot.typeAPercentage = 3/5 ∧ parrot.typeBPercentage = 2/5 →
  cockatiel.totalAmount = 50 ∧ cockatiel.typeAPercentage = 1/2 ∧ cockatiel.typeBPercentage = 1/2 →
  canary.totalAmount = 25 ∧ canary.typeAPercentage = 2/5 ∧ canary.typeBPercentage = 3/5 →
  ∃ (weeks : ℕ), weeks ≥ 6 ∧
    (pantryBoxes.map (·.typeAAmount)).sum ≥ weeks * (parrot.totalAmount * parrot.typeAPercentage +
      cockatiel.totalAmount * cockatiel.typeAPercentage +
      canary.totalAmount * canary.typeAPercentage) ∧
    (pantryBoxes.map (·.typeBAmount)).sum ≥ weeks * (parrot.totalAmount * parrot.typeBPercentage +
      cockatiel.totalAmount * cockatiel.typeBPercentage +
      canary.totalAmount * canary.typeBPercentage) := by
  sorry


end NUMINAMATH_CALUDE_birdseed_supply_l2823_282383


namespace NUMINAMATH_CALUDE_f_eval_one_l2823_282313

/-- The polynomial g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 20

/-- The polynomial f(x) -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 200*x + c

/-- Theorem stating that f(1) = -28417 given the conditions -/
theorem f_eval_one (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g a x = 0 ∧ g a y = 0 ∧ g a z = 0) →
  (∀ x : ℝ, g a x = 0 → f b c x = 0) →
  f b c 1 = -28417 := by
  sorry

end NUMINAMATH_CALUDE_f_eval_one_l2823_282313


namespace NUMINAMATH_CALUDE_probability_king_ace_standard_deck_l2823_282341

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (kings : Nat)
  (aces : Nat)

/-- The probability of drawing a King as the top card and an Ace as the second card -/
def probability_king_ace (d : Deck) : Rat :=
  (d.kings : Rat) / d.total_cards * d.aces / (d.total_cards - 1)

/-- Theorem: The probability of drawing a King as the top card and an Ace as the second card
    in a standard 52-card deck is 4/663 -/
theorem probability_king_ace_standard_deck :
  probability_king_ace ⟨52, 4, 4⟩ = 4 / 663 := by
  sorry

#eval probability_king_ace ⟨52, 4, 4⟩

end NUMINAMATH_CALUDE_probability_king_ace_standard_deck_l2823_282341


namespace NUMINAMATH_CALUDE_odd_sum_probability_l2823_282361

structure Wheel where
  total : ℕ
  even : ℕ
  odd : ℕ
  even_plus_odd : even + odd = total

def probability_odd_sum (a b : Wheel) : ℚ :=
  (a.even * b.odd + a.odd * b.even : ℚ) / (a.total * b.total : ℚ)

theorem odd_sum_probability 
  (a b : Wheel)
  (ha : a.even = a.odd)
  (hb : b.even = 3 * b.odd)
  (hta : a.total = 8)
  (htb : b.total = 8) :
  probability_odd_sum a b = 1/2 :=
sorry

end NUMINAMATH_CALUDE_odd_sum_probability_l2823_282361


namespace NUMINAMATH_CALUDE_product_digits_sum_base7_l2823_282331

/-- Converts a base 7 number to decimal --/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a number in base 7 --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of digits in base 7 of the product of 16₇ and 21₇ is equal to 3₇ --/
theorem product_digits_sum_base7 : 
  sumOfDigitsBase7 (toBase7 (toDecimal 16 * toDecimal 21)) = 3 := by sorry

end NUMINAMATH_CALUDE_product_digits_sum_base7_l2823_282331


namespace NUMINAMATH_CALUDE_no_division_between_valid_numbers_l2823_282389

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ d₅ d₆ d₇ : ℕ),
    d₁ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₂ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₃ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₄ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₅ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₆ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₇ ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₁ ≠ d₅ ∧ d₁ ≠ d₆ ∧ d₁ ≠ d₇ ∧
    d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₂ ≠ d₅ ∧ d₂ ≠ d₆ ∧ d₂ ≠ d₇ ∧
    d₃ ≠ d₄ ∧ d₃ ≠ d₅ ∧ d₃ ≠ d₆ ∧ d₃ ≠ d₇ ∧
    d₄ ≠ d₅ ∧ d₄ ≠ d₆ ∧ d₄ ≠ d₇ ∧
    d₅ ≠ d₆ ∧ d₅ ≠ d₇ ∧
    d₆ ≠ d₇ ∧
    n = d₁ * 1000000 + d₂ * 100000 + d₃ * 10000 + d₄ * 1000 + d₅ * 100 + d₆ * 10 + d₇

theorem no_division_between_valid_numbers :
  ∀ a b : ℕ, is_valid_number a → is_valid_number b → a ≠ b → ¬(a ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_no_division_between_valid_numbers_l2823_282389


namespace NUMINAMATH_CALUDE_constant_phi_is_cone_l2823_282326

-- Define spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the set of points satisfying φ = c
def ConstantPhiSet (c : ℝ) : Set SphericalCoord :=
  {p : SphericalCoord | p.φ = c}

-- Define a cone (we'll use a simplified definition for this statement)
def Cone : Set SphericalCoord := sorry

-- Theorem statement
theorem constant_phi_is_cone (c : ℝ) : 
  ConstantPhiSet c = Cone := by sorry

end NUMINAMATH_CALUDE_constant_phi_is_cone_l2823_282326


namespace NUMINAMATH_CALUDE_factor_expression_l2823_282323

theorem factor_expression (a b c : ℝ) :
  a^4*(b^3 - c^3) + b^4*(c^3 - a^3) + c^4*(a^3 - b^3) = 
  (a-b)*(b-c)*(c-a)*(a^2 + a*b + a*c + b^2 + b*c + c^2) := by
sorry

end NUMINAMATH_CALUDE_factor_expression_l2823_282323


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l2823_282302

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_percentage : ℚ)
  (candidate_valid_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15 / 100)
  (h3 : candidate_valid_votes = 357000) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 75 / 100 := by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l2823_282302


namespace NUMINAMATH_CALUDE_f_range_l2823_282399

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then |x| - 1 else Real.sin x ^ 2

theorem f_range :
  Set.range f = Set.Ioi (-1) := by sorry

end NUMINAMATH_CALUDE_f_range_l2823_282399


namespace NUMINAMATH_CALUDE_remaining_slices_for_phill_l2823_282371

/-- Represents the number of slices in a pizza -/
def total_slices : ℕ := 8

/-- Represents the number of slices given to friends -/
def given_slices : ℕ := 7

/-- Theorem stating that the remaining slices for Phill is 1 -/
theorem remaining_slices_for_phill : total_slices - given_slices = 1 := by
  sorry

end NUMINAMATH_CALUDE_remaining_slices_for_phill_l2823_282371


namespace NUMINAMATH_CALUDE_expand_expression_l2823_282366

theorem expand_expression (x : ℝ) : (17 * x + 21) * (3 * x) = 51 * x^2 + 63 * x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2823_282366
