import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3421_342176

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 10 * x + 6
  ∃ x₁ x₂ : ℝ, x₁ = 5/3 + Real.sqrt 7/3 ∧ 
             x₂ = 5/3 - Real.sqrt 7/3 ∧
             f x₁ = 0 ∧ f x₂ = 0 ∧
             ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3421_342176


namespace NUMINAMATH_CALUDE_final_brand_z_percentage_l3421_342164

/-- Represents the state of the fuel tank -/
structure TankState where
  brandZ : ℚ  -- Amount of Brand Z gasoline
  brandX : ℚ  -- Amount of Brand X gasoline

/-- Fills the tank with Brand Z gasoline -/
def fillWithZ (s : TankState) : TankState :=
  { brandZ := s.brandZ + (1 - s.brandZ - s.brandX), brandX := s.brandX }

/-- Fills the tank with Brand X gasoline -/
def fillWithX (s : TankState) : TankState :=
  { brandZ := s.brandZ, brandX := s.brandX + (1 - s.brandZ - s.brandX) }

/-- Empties the tank by the given fraction -/
def emptyTank (s : TankState) (fraction : ℚ) : TankState :=
  { brandZ := s.brandZ * (1 - fraction), brandX := s.brandX * (1 - fraction) }

/-- The main theorem stating the final percentage of Brand Z gasoline -/
theorem final_brand_z_percentage : 
  let s0 := TankState.mk 1 0  -- Initial state: full of Brand Z
  let s1 := fillWithX (emptyTank s0 (3/4))  -- 3/4 empty, fill with X
  let s2 := fillWithZ (emptyTank s1 (1/2))  -- 1/2 empty, fill with Z
  let s3 := fillWithX (emptyTank s2 (1/2))  -- 1/2 empty, fill with X
  s3.brandZ / (s3.brandZ + s3.brandX) = 5/16 := by
  sorry

#eval (5/16 : ℚ) * 100  -- Should evaluate to 31.25

end NUMINAMATH_CALUDE_final_brand_z_percentage_l3421_342164


namespace NUMINAMATH_CALUDE_proportionality_check_l3421_342178

-- Define the concept of direct and inverse proportionality
def is_directly_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

def is_inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k / x

-- Define the equations
def eq_A (x y : ℝ) : Prop := 2*x + 3*y = 5
def eq_B (x y : ℝ) : Prop := 7*x*y = 14
def eq_C (x y : ℝ) : Prop := x = 7*y + 1
def eq_D (x y : ℝ) : Prop := 4*x + 2*y = 8
def eq_E (x y : ℝ) : Prop := x/y = 5

-- Theorem statement
theorem proportionality_check :
  (¬ ∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_A x y ↔ y = f x) ∧
    (is_directly_proportional f ∨ is_inversely_proportional f)) ∧
  (¬ ∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_D x y ↔ y = f x) ∧
    (is_directly_proportional f ∨ is_inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_B x y ↔ y = f x) ∧
    (is_directly_proportional f ∨ is_inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_C x y ↔ y = f x) ∧
    (is_directly_proportional f ∨ is_inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_E x y ↔ y = f x) ∧
    (is_directly_proportional f ∨ is_inversely_proportional f)) :=
by sorry

end NUMINAMATH_CALUDE_proportionality_check_l3421_342178


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l3421_342148

/-- A line with equal intercepts on both coordinate axes passing through (-3, -2) -/
structure EqualInterceptLine where
  -- The slope of the line
  k : ℝ
  -- The y-intercept of the line
  b : ℝ
  -- The line passes through (-3, -2)
  point_condition : -2 = k * (-3) + b
  -- The line has equal intercepts on both axes
  equal_intercepts : k * b + b = b

/-- The equation of an EqualInterceptLine is either 2x - 3y = 0 or x + y + 5 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (∀ x y, y = l.k * x + l.b → 2 * x - 3 * y = 0) ∨
  (∀ x y, y = l.k * x + l.b → x + y + 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l3421_342148


namespace NUMINAMATH_CALUDE_binary_11010_equals_octal_32_l3421_342130

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (λ (i, bit) acc => acc + if bit then 2^i else 0) 0

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_11010_equals_octal_32 :
  decimal_to_octal (binary_to_decimal [true, true, false, true, false]) = [3, 2] := by
  sorry

end NUMINAMATH_CALUDE_binary_11010_equals_octal_32_l3421_342130


namespace NUMINAMATH_CALUDE_inequality_proof_l3421_342189

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (sum_eq_two : a + b + c + d = 2) :
  (a^2 / (a^2 + 1)^2) + (b^2 / (b^2 + 1)^2) + 
  (c^2 / (c^2 + 1)^2) + (d^2 / (d^2 + 1)^2) ≤ 16/25 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3421_342189


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l3421_342113

/-- Given a sequence {a_n} where the sum of the first n terms is S_n = n^2 + 3n,
    prove that the sum of the 6th, 7th, and 8th terms is 48. -/
theorem sum_of_specific_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
    (h : ∀ n, S n = n^2 + 3*n) :
  a 6 + a 7 + a 8 = 48 := by
sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l3421_342113


namespace NUMINAMATH_CALUDE_davids_crunches_l3421_342111

/-- Given that David did 17 less crunches than Zachary, and Zachary did 62 crunches,
    prove that David did 45 crunches. -/
theorem davids_crunches (zachary_crunches : ℕ) (david_difference : ℤ) 
  (h1 : zachary_crunches = 62)
  (h2 : david_difference = -17) :
  zachary_crunches + david_difference = 45 :=
by sorry

end NUMINAMATH_CALUDE_davids_crunches_l3421_342111


namespace NUMINAMATH_CALUDE_function_is_exponential_base_3_l3421_342184

-- Define the properties of the function f
def satisfies_functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x * f y

def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem function_is_exponential_base_3 (f : ℝ → ℝ) 
  (h1 : satisfies_functional_equation f)
  (h2 : monotonically_increasing f) :
  ∀ x, f x = 3^x :=
sorry

end NUMINAMATH_CALUDE_function_is_exponential_base_3_l3421_342184


namespace NUMINAMATH_CALUDE_problem_1_l3421_342146

theorem problem_1 : Real.sqrt 9 + (-2)^3 - Real.cos (π / 3) = -11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3421_342146


namespace NUMINAMATH_CALUDE_min_c_for_unique_solution_l3421_342134

/-- The system of equations -/
def system (x y c : ℝ) : Prop :=
  8 * (x + 7)^4 + (y - 4)^4 = c ∧ (x + 4)^4 + 8 * (y - 7)^4 = c

/-- The existence of a unique solution for the system -/
def has_unique_solution (c : ℝ) : Prop :=
  ∃! x y, system x y c

/-- The theorem stating the minimum value of c for a unique solution -/
theorem min_c_for_unique_solution :
  ∀ c, has_unique_solution c → c ≥ 24 ∧ has_unique_solution 24 :=
sorry

end NUMINAMATH_CALUDE_min_c_for_unique_solution_l3421_342134


namespace NUMINAMATH_CALUDE_square_inscribed_in_circle_l3421_342193

theorem square_inscribed_in_circle (r : ℝ) (S : ℝ) :
  r > 0 →
  r^2 * π = 16 * π →
  S = (2 * r)^2 / 2 →
  S = 32 :=
by sorry

end NUMINAMATH_CALUDE_square_inscribed_in_circle_l3421_342193


namespace NUMINAMATH_CALUDE_negative_product_expression_B_l3421_342135

theorem negative_product_expression_B : 
  let a : ℚ := -9
  let b : ℚ := 1/8
  let c : ℚ := -4/7
  let d : ℚ := 7
  let e : ℚ := -1/3
  a * b * c * d * e < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_product_expression_B_l3421_342135


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l3421_342131

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l3421_342131


namespace NUMINAMATH_CALUDE_sqrt_x_squared_plus_6x_plus_9_l3421_342117

theorem sqrt_x_squared_plus_6x_plus_9 (x : ℝ) (h : x = Real.sqrt 5 - 3) :
  Real.sqrt (x^2 + 6*x + 9) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_squared_plus_6x_plus_9_l3421_342117


namespace NUMINAMATH_CALUDE_jam_cost_proof_l3421_342106

/-- The cost of jam used for all sandwiches --/
def jam_cost (N B J H : ℕ+) : ℚ :=
  (N * J * 7 : ℚ) / 100

/-- The total cost of ingredients for all sandwiches --/
def total_cost (N B J H : ℕ+) : ℚ :=
  (N * (6 * B + 7 * J + 4 * H) : ℚ) / 100

theorem jam_cost_proof (N B J H : ℕ+) (h1 : N > 1) (h2 : total_cost N B J H = 462/100) :
  jam_cost N B J H = 462/100 := by
  sorry

end NUMINAMATH_CALUDE_jam_cost_proof_l3421_342106


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l3421_342171

theorem rectangle_area_increase (L W P : ℝ) (h : L > 0) (h' : W > 0) (h'' : P > 0) :
  (L * (1 + P)) * (W * (1 + P)) = 4 * (L * W) → P = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l3421_342171


namespace NUMINAMATH_CALUDE_concert_ticket_revenue_l3421_342133

/-- Calculate the total revenue from concert ticket sales --/
theorem concert_ticket_revenue : 
  let original_price : ℚ := 20
  let first_group_size : ℕ := 10
  let second_group_size : ℕ := 20
  let third_group_size : ℕ := 15
  let first_discount : ℚ := 0.4
  let second_discount : ℚ := 0.15
  let third_premium : ℚ := 0.1
  let first_group_revenue := first_group_size * (original_price * (1 - first_discount))
  let second_group_revenue := second_group_size * (original_price * (1 - second_discount))
  let third_group_revenue := third_group_size * (original_price * (1 + third_premium))
  let total_revenue := first_group_revenue + second_group_revenue + third_group_revenue
  total_revenue = 790 := by
  sorry


end NUMINAMATH_CALUDE_concert_ticket_revenue_l3421_342133


namespace NUMINAMATH_CALUDE_candidate_count_l3421_342126

theorem candidate_count (total : ℕ) (selected_A selected_B : ℕ) : 
  selected_A = (6 * total) / 100 →
  selected_B = (7 * total) / 100 →
  selected_B = selected_A + 81 →
  total = 8100 := by
sorry

end NUMINAMATH_CALUDE_candidate_count_l3421_342126


namespace NUMINAMATH_CALUDE_no_consecutive_solution_l3421_342141

theorem no_consecutive_solution : ¬ ∃ (a b c d e f : ℕ), 
  (b = a + 1) ∧ (c = a + 2) ∧ (d = a + 3) ∧ (e = a + 4) ∧ (f = a + 5) ∧
  (a * b^c * d + e^f * a * b = 2015) := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_solution_l3421_342141


namespace NUMINAMATH_CALUDE_rectangle_DC_length_l3421_342150

/-- Represents a rectangle ABCF with points E and D on FC -/
structure Rectangle :=
  (AB : ℝ)
  (AF : ℝ)
  (FE : ℝ)
  (area_ABDE : ℝ)

/-- The length of DC in the rectangle -/
def length_DC (r : Rectangle) : ℝ :=
  -- Definition of DC length
  sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem rectangle_DC_length (r : Rectangle) 
  (h1 : r.AB = 30)
  (h2 : r.AF = 14)
  (h3 : r.FE = 5)
  (h4 : r.area_ABDE = 266) :
  length_DC r = 17 :=
sorry

end NUMINAMATH_CALUDE_rectangle_DC_length_l3421_342150


namespace NUMINAMATH_CALUDE_consecutive_integers_squares_sum_l3421_342154

theorem consecutive_integers_squares_sum : ∃ a : ℕ,
  (a > 0) ∧
  ((a - 1) * a * (a + 1) = 8 * (3 * a)) ∧
  ((a - 1)^2 + a^2 + (a + 1)^2 = 77) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_squares_sum_l3421_342154


namespace NUMINAMATH_CALUDE_problem_statement_l3421_342144

theorem problem_statement (x y : ℚ) (hx : x = 5/6) (hy : y = 6/5) :
  (1/3) * x^8 * y^5 - x^2 * y = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3421_342144


namespace NUMINAMATH_CALUDE_trig_expression_equality_l3421_342107

theorem trig_expression_equality : 
  (Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + Real.cos (160 * π / 180) * Real.cos (110 * π / 180)) /
  (Real.sin (24 * π / 180) * Real.cos (6 * π / 180) + Real.cos (156 * π / 180) * Real.cos (96 * π / 180)) =
  (1 - Real.sin (40 * π / 180)) / (1 - Real.sin (48 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l3421_342107


namespace NUMINAMATH_CALUDE_no_primes_in_list_l3421_342177

/-- Represents a number formed by repeating 57 a certain number of times -/
def repeatedNumber (repetitions : ℕ) : ℕ :=
  57 * ((10^(2*repetitions) - 1) / 99)

/-- The list of numbers formed by repeating 57 from 1 to n times -/
def numberList (n : ℕ) : List ℕ :=
  List.map repeatedNumber (List.range n)

/-- Counts the number of prime numbers in the list -/
def countPrimes (list : List ℕ) : ℕ :=
  (list.filter Nat.Prime).length

theorem no_primes_in_list (n : ℕ) : countPrimes (numberList n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_list_l3421_342177


namespace NUMINAMATH_CALUDE_expression_equivalence_l3421_342170

theorem expression_equivalence (x y z : ℝ) :
  let P := x + y
  let Q := x - y
  ((P + Q + z) / (P - Q - z) - (P - Q - z) / (P + Q + z)) = 
    (4 * (x^2 + y^2 + x*z)) / ((2*y - z) * (2*x + z)) := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l3421_342170


namespace NUMINAMATH_CALUDE_base6_subtraction_addition_l3421_342127

-- Define a function to convert base-6 to decimal
def base6ToDecimal (n : ℕ) : ℕ := sorry

-- Define a function to convert decimal to base-6
def decimalToBase6 (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem base6_subtraction_addition :
  decimalToBase6 (base6ToDecimal 655 - base6ToDecimal 222 + base6ToDecimal 111) = 544 := by
  sorry

end NUMINAMATH_CALUDE_base6_subtraction_addition_l3421_342127


namespace NUMINAMATH_CALUDE_no_covering_compact_rationals_l3421_342157

theorem no_covering_compact_rationals :
  ¬ (∃ (A : ℕ → Set ℝ),
    (∀ n, IsCompact (A n)) ∧
    (∀ n, A n ⊆ Set.range (Rat.cast : ℚ → ℝ)) ∧
    (∀ K : Set ℝ, IsCompact K → K ⊆ Set.range (Rat.cast : ℚ → ℝ) →
      ∃ m, K ⊆ A m)) :=
by sorry

end NUMINAMATH_CALUDE_no_covering_compact_rationals_l3421_342157


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3421_342140

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3421_342140


namespace NUMINAMATH_CALUDE_square_root_of_ten_thousand_l3421_342167

theorem square_root_of_ten_thousand : Real.sqrt 10000 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_ten_thousand_l3421_342167


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l3421_342179

/-- The quadratic function f(x) = -(x+1)^2 - 8 -/
def f (x : ℝ) : ℝ := -(x + 1)^2 - 8

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (-1, -8)

/-- Theorem: The vertex of the quadratic function f(x) = -(x+1)^2 - 8 is at the point (-1, -8) -/
theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≤ f (vertex.1)) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l3421_342179


namespace NUMINAMATH_CALUDE_one_hundred_twentieth_letter_l3421_342195

def letter_pattern (n : ℕ) : Char :=
  match n % 4 with
  | 0 => 'D'
  | 1 => 'A'
  | 2 => 'B'
  | 3 => 'C'
  | _ => 'D'  -- This case is unreachable, but Lean requires it for exhaustiveness

theorem one_hundred_twentieth_letter :
  letter_pattern 120 = 'D' := by
  sorry

end NUMINAMATH_CALUDE_one_hundred_twentieth_letter_l3421_342195


namespace NUMINAMATH_CALUDE_five_Y_three_equals_four_l3421_342196

-- Define the Y operation
def Y (a b : ℤ) : ℤ := a^2 - 2*a*b + b^2

-- Theorem statement
theorem five_Y_three_equals_four : Y 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_five_Y_three_equals_four_l3421_342196


namespace NUMINAMATH_CALUDE_no_consistent_solution_l3421_342151

theorem no_consistent_solution :
  ¬ ∃ (x y : ℕ+) (z : ℤ),
    (∃ (q : ℕ), x = 11 * y + 4 ∧ x = 11 * q + 4) ∧
    (∃ (q : ℕ), 2 * x = 8 * (3 * y) + 3 ∧ 2 * x = 8 * q + 3) ∧
    (∃ (q : ℕ), x + z = 17 * (2 * y) + 5 ∧ x + z = 17 * q + 5) :=
by sorry

end NUMINAMATH_CALUDE_no_consistent_solution_l3421_342151


namespace NUMINAMATH_CALUDE_hawks_first_half_score_l3421_342122

/-- Represents the score of a basketball team in a game with two halves -/
structure TeamScore where
  first_half : ℕ
  second_half : ℕ

/-- Represents the final scores of two teams in a basketball game -/
structure GameScore where
  eagles : TeamScore
  hawks : TeamScore

/-- The conditions of the basketball game -/
def game_conditions (game : GameScore) : Prop :=
  let eagles_total := game.eagles.first_half + game.eagles.second_half
  let hawks_total := game.hawks.first_half + game.hawks.second_half
  eagles_total + hawks_total = 120 ∧
  eagles_total = hawks_total + 16 ∧
  game.hawks.second_half = game.hawks.first_half + 8

theorem hawks_first_half_score (game : GameScore) :
  game_conditions game → game.hawks.first_half = 22 := by
  sorry

#check hawks_first_half_score

end NUMINAMATH_CALUDE_hawks_first_half_score_l3421_342122


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3421_342121

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 4) / 9 = 4 / (x - 9) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3421_342121


namespace NUMINAMATH_CALUDE_fraction_equality_l3421_342142

theorem fraction_equality (a b : ℚ) (h : a / 5 = b / 3) : (a - b) / (3 * a) = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3421_342142


namespace NUMINAMATH_CALUDE_slope_range_l3421_342115

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l passing through (0,2) with slope k
def line (x y k : ℝ) : Prop := y = k * x + 2

-- Define the condition for intersection points
def intersects (k : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ, 
  x₁ ≠ x₂ ∧ ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line x₁ y₁ k ∧ line x₂ y₂ k

-- Define the acute angle condition
def acute_angle (k : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂ : ℝ, 
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line x₁ y₁ k ∧ line x₂ y₂ k → 
  x₁ * x₂ + y₁ * y₂ > 0

-- Main theorem
theorem slope_range : 
  ∀ k : ℝ, intersects k ∧ acute_angle k ↔ 
  (k > Real.sqrt 3 / 2 ∧ k < 2) ∨ (k < -Real.sqrt 3 / 2 ∧ k > -2) := by
  sorry

end NUMINAMATH_CALUDE_slope_range_l3421_342115


namespace NUMINAMATH_CALUDE_keith_cantaloupes_l3421_342187

/-- The number of cantaloupes grown by Keith, given the total number of cantaloupes
    and the numbers grown by Fred and Jason. -/
theorem keith_cantaloupes (total : ℕ) (fred : ℕ) (jason : ℕ) 
    (h_total : total = 65) 
    (h_fred : fred = 16) 
    (h_jason : jason = 20) : 
  total - (fred + jason) = 29 := by
  sorry

end NUMINAMATH_CALUDE_keith_cantaloupes_l3421_342187


namespace NUMINAMATH_CALUDE_conditional_probability_wind_rain_l3421_342198

/-- Given probabilities of events A and B, and their intersection,
    prove that the conditional probability P(B|A) is 3/4 -/
theorem conditional_probability_wind_rain 
  (P_A P_B P_AB : ℝ) 
  (h_A : P_A = 0.4)
  (h_B : P_B = 0.5)
  (h_AB : P_AB = 0.3) :
  P_AB / P_A = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_wind_rain_l3421_342198


namespace NUMINAMATH_CALUDE_bank_deposit_theorem_l3421_342116

/-- Calculates the actual amount of principal and interest after one year,
    given an initial deposit, annual interest rate, and interest tax rate. -/
def actual_amount (initial_deposit : ℝ) (interest_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  initial_deposit + (1 - tax_rate) * interest_rate * initial_deposit

theorem bank_deposit_theorem (x : ℝ) :
  actual_amount x 0.0225 0.2 = (0.8 * 0.0225 * x + x) := by sorry

end NUMINAMATH_CALUDE_bank_deposit_theorem_l3421_342116


namespace NUMINAMATH_CALUDE_books_sold_l3421_342183

theorem books_sold (initial_books : ℕ) (new_books : ℕ) (final_books : ℕ) 
  (h1 : initial_books = 34)
  (h2 : new_books = 7)
  (h3 : final_books = 24) :
  initial_books - (initial_books - new_books - final_books) = 17 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_l3421_342183


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3421_342149

theorem complex_fraction_simplification :
  (1 + 3*Complex.I) / (1 - Complex.I) = -1 + 2*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3421_342149


namespace NUMINAMATH_CALUDE_fred_car_wash_earnings_l3421_342118

/-- The amount of money Fred made washing cars -/
def fred_earnings (initial_amount final_amount : ℕ) : ℕ :=
  final_amount - initial_amount

/-- Theorem: Fred made 63 dollars washing cars -/
theorem fred_car_wash_earnings : fred_earnings 23 86 = 63 := by
  sorry

end NUMINAMATH_CALUDE_fred_car_wash_earnings_l3421_342118


namespace NUMINAMATH_CALUDE_shaded_region_area_l3421_342101

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- A line defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- The shaded region formed by two intersecting lines and a right triangle -/
structure ShadedRegion where
  line1 : Line
  line2 : Line

def area_of_shaded_region (region : ShadedRegion) : ℚ :=
  sorry

theorem shaded_region_area :
  let line1 := Line.mk (Point.mk 0 5) (Point.mk 10 2)
  let line2 := Line.mk (Point.mk 2 6) (Point.mk 9 0)
  let region := ShadedRegion.mk line1 line2
  area_of_shaded_region region = 151425 / 3136 := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_area_l3421_342101


namespace NUMINAMATH_CALUDE_correct_factorization_l3421_342132

theorem correct_factorization (x y : ℝ) : x^2 - 2*x*y + x = x*(x - 2*y + 1) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l3421_342132


namespace NUMINAMATH_CALUDE_value_of_a_value_of_b_when_perpendicular_distance_when_parallel_l3421_342160

-- Define the lines
def l1 (a : ℝ) : ℝ → ℝ → Prop := λ x y => a * x + 2 * y - 1 = 0
def l2 (b : ℝ) : ℝ → ℝ → Prop := λ x y => x + b * y - 3 = 0

-- Define the angle of inclination
def angle_of_inclination (l : ℝ → ℝ → Prop) : ℝ := sorry

-- Define perpendicularity
def perpendicular (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry

-- Define parallelism
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry

-- Define distance between parallel lines
def distance_between_parallel_lines (l1 l2 : ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statements
theorem value_of_a (a : ℝ) : 
  angle_of_inclination (l1 a) = π / 4 → a = -2 := by sorry

theorem value_of_b_when_perpendicular (b : ℝ) : 
  perpendicular (l1 (-2)) (l2 b) → b = 1 := by sorry

theorem distance_when_parallel (b : ℝ) : 
  parallel (l1 (-2)) (l2 b) → 
  distance_between_parallel_lines (l1 (-2)) (l2 b) = 7 * Real.sqrt 2 / 4 := by sorry

end NUMINAMATH_CALUDE_value_of_a_value_of_b_when_perpendicular_distance_when_parallel_l3421_342160


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3421_342125

open Set

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State the theorem
theorem solution_set_of_inequality 
  (h1 : ∀ x, f' x < f x) 
  (h2 : f 1 = Real.exp 1) :
  {x : ℝ | f (Real.log x) > x} = Ioo 0 (Real.exp 1) := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3421_342125


namespace NUMINAMATH_CALUDE_incorrect_number_value_l3421_342174

theorem incorrect_number_value (n : ℕ) (initial_avg correct_avg incorrect_value : ℚ) 
  (h1 : n = 10)
  (h2 : initial_avg = 46)
  (h3 : correct_avg = 51)
  (h4 : incorrect_value = 25) :
  let correct_value := n * correct_avg - (n * initial_avg - incorrect_value)
  correct_value = 75 := by sorry

end NUMINAMATH_CALUDE_incorrect_number_value_l3421_342174


namespace NUMINAMATH_CALUDE_ratio_problem_l3421_342165

theorem ratio_problem (a b c : ℚ) 
  (h1 : c / b = 4)
  (h2 : b / a = 2)
  (h3 : c = 20 - 7 * b) :
  a = 10 / 11 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3421_342165


namespace NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l3421_342181

theorem sqrt_six_div_sqrt_two_eq_sqrt_three : 
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_div_sqrt_two_eq_sqrt_three_l3421_342181


namespace NUMINAMATH_CALUDE_alice_bob_meet_time_l3421_342190

def circlePoints : ℕ := 15
def aliceMove : ℕ := 4
def bobMove : ℕ := 8  -- Equivalent clockwise movement

theorem alice_bob_meet_time :
  let relativeMove := (bobMove - aliceMove) % circlePoints
  ∃ n : ℕ, n > 0 ∧ (n * relativeMove) % circlePoints = 0 ∧
  ∀ m : ℕ, 0 < m ∧ m < n → (m * relativeMove) % circlePoints ≠ 0 ∧
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_alice_bob_meet_time_l3421_342190


namespace NUMINAMATH_CALUDE_noon_temperature_l3421_342162

def morning_temp : ℤ := 4
def temp_drop : ℤ := 10

theorem noon_temperature :
  morning_temp - temp_drop = -6 := by
  sorry

end NUMINAMATH_CALUDE_noon_temperature_l3421_342162


namespace NUMINAMATH_CALUDE_min_value_expression_l3421_342102

theorem min_value_expression (x y : ℝ) :
  x^2 - 6*x*Real.sin y - 9*(Real.cos y)^2 ≥ -9 ∧
  ∃ (x y : ℝ), x^2 - 6*x*Real.sin y - 9*(Real.cos y)^2 = -9 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3421_342102


namespace NUMINAMATH_CALUDE_banana_solution_l3421_342137

/-- Represents the banana cutting scenario -/
def banana_problem (initial_bananas : ℕ) (cut_bananas : ℕ) (eaten_bananas : ℕ) : Prop :=
  initial_bananas ≥ cut_bananas ∧
  cut_bananas > eaten_bananas ∧
  cut_bananas - eaten_bananas = 2 * (initial_bananas - cut_bananas)

/-- Theorem stating the solution to the banana problem -/
theorem banana_solution :
  ∃ (cut_bananas : ℕ),
    banana_problem 310 cut_bananas 70 ∧
    310 - cut_bananas = 100 := by
  sorry

end NUMINAMATH_CALUDE_banana_solution_l3421_342137


namespace NUMINAMATH_CALUDE_point_on_transformed_graph_l3421_342194

/-- Given a function g: ℝ → ℝ, prove that if (2,5) lies on the graph of y = g(x),
    then (1,8) lies on the graph of 4y = 5g(3x-1) + 7, and the sum of the coordinates of (1,8) is 9. -/
theorem point_on_transformed_graph (g : ℝ → ℝ) (h : g 2 = 5) :
  4 * 8 = 5 * g (3 * 1 - 1) + 7 ∧ 1 + 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_point_on_transformed_graph_l3421_342194


namespace NUMINAMATH_CALUDE_domain_of_f_l3421_342197

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define our function f(x) = lg(x+1)
noncomputable def f (x : ℝ) := lg (x + 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | f x = f x} = {x : ℝ | x > -1} :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l3421_342197


namespace NUMINAMATH_CALUDE_triangle_inequality_reciprocal_l3421_342124

theorem triangle_inequality_reciprocal (a b c : ℝ) 
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  1 / (a + c) + 1 / (b + c) > 1 / (a + b) ∧
  1 / (a + c) + 1 / (a + b) > 1 / (b + c) ∧
  1 / (b + c) + 1 / (a + b) > 1 / (a + c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_reciprocal_l3421_342124


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3421_342100

theorem angle_measure_proof : ∃! x : ℝ, 0 < x ∧ x < 90 ∧ x + (3 * x^2 + 10) = 90 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3421_342100


namespace NUMINAMATH_CALUDE_probability_one_defective_l3421_342158

/-- The probability of selecting exactly one defective product from a batch -/
theorem probability_one_defective (total : ℕ) (defective : ℕ) : 
  total = 40 →
  defective = 12 →
  (Nat.choose (total - defective) 1 * Nat.choose defective 1) / Nat.choose total 2 = 28 / 65 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_defective_l3421_342158


namespace NUMINAMATH_CALUDE_alcohol_dilution_l3421_342161

theorem alcohol_dilution (initial_volume : ℝ) (initial_alcohol_percentage : ℝ) (added_water : ℝ) :
  initial_volume = 15 →
  initial_alcohol_percentage = 20 →
  added_water = 5 →
  let initial_alcohol := initial_volume * (initial_alcohol_percentage / 100)
  let new_volume := initial_volume + added_water
  let new_alcohol_percentage := (initial_alcohol / new_volume) * 100
  new_alcohol_percentage = 15 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_dilution_l3421_342161


namespace NUMINAMATH_CALUDE_prob_one_rectification_prob_at_least_one_closed_l3421_342166

-- Define the number of canteens
def num_canteens : ℕ := 4

-- Define the probability of passing inspection before rectification
def prob_pass_before : ℝ := 0.5

-- Define the probability of passing inspection after rectification
def prob_pass_after : ℝ := 0.8

-- Theorem for the probability that exactly one canteen needs rectification
theorem prob_one_rectification :
  (num_canteens.choose 1 : ℝ) * prob_pass_before^(num_canteens - 1) * (1 - prob_pass_before) = 0.25 := by
  sorry

-- Theorem for the probability that at least one canteen is closed
theorem prob_at_least_one_closed :
  1 - (1 - (1 - prob_pass_before) * (1 - prob_pass_after))^num_canteens = 0.34 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_rectification_prob_at_least_one_closed_l3421_342166


namespace NUMINAMATH_CALUDE_greatest_lower_bound_system_l3421_342120

theorem greatest_lower_bound_system (x y z u : ℕ+) 
  (h1 : x ≥ y) 
  (h2 : x + y = z + u) 
  (h3 : 2 * x * y = z * u) : 
  ∃ m : ℝ, m = 3 + 2 * Real.sqrt 2 ∧ 
  (∀ a b c d : ℕ+, a ≥ b → a + b = c + d → 2 * a * b = c * d → (a : ℝ) / b ≥ m) ∧
  (∀ ε > 0, ∃ a b c d : ℕ+, a ≥ b ∧ a + b = c + d ∧ 2 * a * b = c * d ∧ (a : ℝ) / b < m + ε) :=
sorry

end NUMINAMATH_CALUDE_greatest_lower_bound_system_l3421_342120


namespace NUMINAMATH_CALUDE_trapezoid_mn_length_l3421_342185

/-- Represents a trapezoid ABCD with points M and N on its sides -/
structure Trapezoid (α : Type*) [LinearOrderedField α] :=
  (a b : α)  -- lengths of BC and AD respectively
  (area_ratio : α)  -- ratio of areas of MBCN to MADN

/-- 
  Given a trapezoid ABCD with BC = a and AD = b, 
  if MN is parallel to AD and the areas of trapezoids MBCN and MADN are in the ratio 1:5, 
  then MN = sqrt((5a^2 + b^2) / 6)
-/
theorem trapezoid_mn_length 
  {α : Type*} [LinearOrderedField α] (t : Trapezoid α) 
  (h_ratio : t.area_ratio = 1/5) :
  ∃ mn : α, mn^2 = (5*t.a^2 + t.b^2) / 6 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_mn_length_l3421_342185


namespace NUMINAMATH_CALUDE_sum_has_48_divisors_l3421_342155

def sum_of_numbers : ℕ := 9240 + 8820

theorem sum_has_48_divisors : Nat.card (Nat.divisors sum_of_numbers) = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_has_48_divisors_l3421_342155


namespace NUMINAMATH_CALUDE_space_station_cost_share_l3421_342169

/-- Calculates the individual share of a project cost -/
def calculate_share (total_cost : ℕ) (total_population : ℕ) : ℚ :=
  (total_cost : ℚ) / ((total_population : ℚ) / 2)

theorem space_station_cost_share :
  let total_cost : ℕ := 50000000000 -- $50 billion in dollars
  let total_population : ℕ := 400000000 -- 400 million people
  calculate_share total_cost total_population = 250 := by sorry

end NUMINAMATH_CALUDE_space_station_cost_share_l3421_342169


namespace NUMINAMATH_CALUDE_fraction_value_l3421_342152

theorem fraction_value : (0.5 ^ 4) / (0.05 ^ 2.5) = 559.06 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l3421_342152


namespace NUMINAMATH_CALUDE_power_sum_of_i_l3421_342103

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23 + i^47 = -2*i := by sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l3421_342103


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3421_342109

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 3 / 4 → Nat.gcd a b = 5 → Nat.lcm a b = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3421_342109


namespace NUMINAMATH_CALUDE_lion_king_cost_l3421_342186

theorem lion_king_cost (
  lion_king_earnings : ℝ)
  (star_wars_cost : ℝ)
  (star_wars_earnings : ℝ)
  (h1 : lion_king_earnings = 200)
  (h2 : star_wars_cost = 25)
  (h3 : star_wars_earnings = 405)
  (h4 : lion_king_earnings - (lion_king_earnings - (star_wars_earnings - star_wars_cost) / 2) = 10) :
  ∃ (lion_king_cost : ℝ), lion_king_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_lion_king_cost_l3421_342186


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3421_342175

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3421_342175


namespace NUMINAMATH_CALUDE_special_1992_gon_exists_l3421_342180

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : Fin n → ℝ
  convex : sorry -- Condition for convexity

/-- An inscribed circle in a polygon -/
structure InscribedCircle {n : ℕ} (p : ConvexPolygon n) where
  center : ℝ × ℝ
  radius : ℝ
  touches_all_sides : sorry -- Condition that the circle touches all sides

/-- The theorem stating the existence of the special 1992-gon -/
theorem special_1992_gon_exists : ∃ (p : ConvexPolygon 1992),
  (∃ (σ : Equiv (Fin 1992) (Fin 1992)), ∀ i, p.sides i = σ i + 1) ∧
  ∃ (c : InscribedCircle p), True :=
sorry

end NUMINAMATH_CALUDE_special_1992_gon_exists_l3421_342180


namespace NUMINAMATH_CALUDE_one_face_colored_count_l3421_342153

/-- Represents a cube that has been painted and cut into smaller cubes -/
structure PaintedCube where
  edge_count : Nat
  is_painted : Bool

/-- Counts the number of small cubes with exactly one face colored -/
def count_one_face_colored (cube : PaintedCube) : Nat :=
  sorry

/-- Theorem: A cube painted on all faces and cut into 5x5x5 smaller cubes
    will have 54 small cubes with exactly one face colored -/
theorem one_face_colored_count (cube : PaintedCube) :
  cube.edge_count = 5 → cube.is_painted → count_one_face_colored cube = 54 := by
  sorry

end NUMINAMATH_CALUDE_one_face_colored_count_l3421_342153


namespace NUMINAMATH_CALUDE_altitude_intersection_property_l3421_342143

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a line is an altitude of a triangle -/
def isAltitude (t : Triangle) (p1 p2 : Point) : Prop := sorry

/-- Checks if two lines intersect at a point -/
def intersectAt (p1 p2 p3 p4 p5 : Point) : Prop := sorry

theorem altitude_intersection_property (ABC : Triangle) (D E H : Point) :
  isAcute ABC →
  isAltitude ABC A D →
  isAltitude ABC B E →
  intersectAt A D B E H →
  distance H D = 3 →
  distance H E = 4 →
  ∃ (BD DC AE EC : ℝ),
    BD * DC - AE * EC = 3 * distance A D - 7 := by
  sorry

end NUMINAMATH_CALUDE_altitude_intersection_property_l3421_342143


namespace NUMINAMATH_CALUDE_or_true_if_one_true_l3421_342139

theorem or_true_if_one_true (p q : Prop) (h : p ∨ q) : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_or_true_if_one_true_l3421_342139


namespace NUMINAMATH_CALUDE_sequence_difference_l3421_342114

theorem sequence_difference (a : ℕ → ℤ) 
  (h : ∀ n : ℕ, a (n + 1) - a n - n = 0) : 
  a 2017 - a 2016 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_l3421_342114


namespace NUMINAMATH_CALUDE_probability_site_in_statistics_l3421_342168

def letters_statistics : List Char := ['S', 'T', 'A', 'T', 'I', 'S', 'T', 'I', 'C', 'S']
def letters_site : List Char := ['S', 'I', 'T', 'E']

def count_in_statistics (c : Char) : Nat :=
  (letters_statistics.filter (· = c)).length

def is_in_site (c : Char) : Bool :=
  letters_site.contains c

def favorable_outcomes : Nat :=
  (letters_statistics.filter is_in_site).length

def total_outcomes : Nat :=
  letters_statistics.length

theorem probability_site_in_statistics :
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_site_in_statistics_l3421_342168


namespace NUMINAMATH_CALUDE_range_symmetric_range_b_decreasing_l3421_342172

-- Define the function f
def f (a b x : ℝ) : ℝ := -2 * x^2 + a * x + b

-- Theorem for part (1)
theorem range_symmetric (a b : ℝ) :
  f a b 2 = -3 →
  (∀ x : ℝ, f a b (1 + x) = f a b (1 - x)) →
  (∀ x : ℝ, x ∈ Set.Icc (-2) 3 → f a b x ∈ Set.Icc (-19) (-1)) :=
sorry

-- Theorem for part (2)
theorem range_b_decreasing (a b : ℝ) :
  f a b 2 = -3 →
  (∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x ≤ y → f a b x ≥ f a b y) →
  b ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_range_symmetric_range_b_decreasing_l3421_342172


namespace NUMINAMATH_CALUDE_solve_for_y_l3421_342145

theorem solve_for_y (x y : ℝ) 
  (eq1 : 9823 + x = 13200) 
  (eq2 : x = y / 3 + 37.5) : 
  y = 10018.5 := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l3421_342145


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3421_342128

theorem polynomial_factorization :
  ∀ x : ℝ, x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3421_342128


namespace NUMINAMATH_CALUDE_half_AB_equals_2_1_l3421_342192

def MA : ℝ × ℝ := (-2, 4)
def MB : ℝ × ℝ := (2, 6)

theorem half_AB_equals_2_1 : (1 / 2 : ℝ) • (MB - MA) = (2, 1) := by sorry

end NUMINAMATH_CALUDE_half_AB_equals_2_1_l3421_342192


namespace NUMINAMATH_CALUDE_only_four_points_l3421_342108

/-- A configuration of n points in the plane with associated real numbers -/
structure PointConfiguration (n : ℕ) where
  points : Fin n → ℝ × ℝ
  r : Fin n → ℝ

/-- The area of a triangle given by three points -/
def triangleArea (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

/-- Three points are collinear if the area of the triangle they form is zero -/
def collinear (p₁ p₂ p₃ : ℝ × ℝ) : Prop :=
  triangleArea p₁ p₂ p₃ = 0

/-- A valid configuration satisfies the problem conditions -/
def validConfiguration {n : ℕ} (config : PointConfiguration n) : Prop :=
  (n > 3) ∧
  (∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ¬collinear (config.points i) (config.points j) (config.points k)) ∧
  (∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    triangleArea (config.points i) (config.points j) (config.points k) =
      config.r i + config.r j + config.r k)

/-- The main theorem: The only valid configuration is for n = 4 -/
theorem only_four_points :
  ∀ n : ℕ, (∃ config : PointConfiguration n, validConfiguration config) → n = 4 :=
sorry

end NUMINAMATH_CALUDE_only_four_points_l3421_342108


namespace NUMINAMATH_CALUDE_jack_classic_authors_l3421_342123

/-- The number of books each classic author has -/
def books_per_author : ℕ := 33

/-- The total number of classic books in Jack's collection -/
def total_classic_books : ℕ := 198

/-- The number of classic authors in Jack's collection -/
def number_of_authors : ℕ := total_classic_books / books_per_author

theorem jack_classic_authors :
  number_of_authors = 6 :=
sorry

end NUMINAMATH_CALUDE_jack_classic_authors_l3421_342123


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3421_342119

theorem rectangle_diagonal (a b d : ℝ) : 
  a = 13 →
  a * b = 142.40786495134319 →
  d^2 = a^2 + b^2 →
  d = 17 := by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3421_342119


namespace NUMINAMATH_CALUDE_fiction_books_count_l3421_342156

theorem fiction_books_count (total : ℕ) (picture_books : ℕ) : 
  total = 35 → picture_books = 11 → ∃ (fiction : ℕ), 
    fiction + (fiction + 4) + 2 * fiction + picture_books = total ∧ fiction = 5 := by
  sorry

end NUMINAMATH_CALUDE_fiction_books_count_l3421_342156


namespace NUMINAMATH_CALUDE_circle_area_circumference_ratio_l3421_342159

theorem circle_area_circumference_ratio (r₁ r₂ : ℝ) (h : π * r₁^2 / (π * r₂^2) = 49 / 64) :
  (2 * π * r₁) / (2 * π * r₂) = 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_circle_area_circumference_ratio_l3421_342159


namespace NUMINAMATH_CALUDE_x_squared_mod_25_l3421_342136

theorem x_squared_mod_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 2 * x ≡ 22 [ZMOD 25]) :
  x^2 ≡ 9 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_25_l3421_342136


namespace NUMINAMATH_CALUDE_divisible_by_prime_l3421_342147

/-- Sequence of polynomials Q -/
def Q : ℕ → (ℤ → ℤ)
| 0 => λ x => 1
| 1 => λ x => x
| (n + 2) => λ x => x * Q (n + 1) x + (n + 1) * Q n x

/-- Theorem statement -/
theorem divisible_by_prime (p : ℕ) (hp : p.Prime) (hp2 : p > 2) :
  ∀ x : ℤ, (Q p x - x ^ p) % p = 0 := by sorry

end NUMINAMATH_CALUDE_divisible_by_prime_l3421_342147


namespace NUMINAMATH_CALUDE_opposite_faces_sum_seven_l3421_342191

-- Define a type for the faces of a die
inductive DieFace : Type
  | one : DieFace
  | two : DieFace
  | three : DieFace
  | four : DieFace
  | five : DieFace
  | six : DieFace

-- Define a function to get the numeric value of a face
def faceValue : DieFace → Nat
  | DieFace.one => 1
  | DieFace.two => 2
  | DieFace.three => 3
  | DieFace.four => 4
  | DieFace.five => 5
  | DieFace.six => 6

-- Define a function to get the opposite face
def oppositeFace : DieFace → DieFace
  | DieFace.one => DieFace.six
  | DieFace.two => DieFace.five
  | DieFace.three => DieFace.four
  | DieFace.four => DieFace.three
  | DieFace.five => DieFace.two
  | DieFace.six => DieFace.one

-- Theorem: The sum of values on opposite faces is always 7
theorem opposite_faces_sum_seven (face : DieFace) :
  faceValue face + faceValue (oppositeFace face) = 7 := by
  sorry


end NUMINAMATH_CALUDE_opposite_faces_sum_seven_l3421_342191


namespace NUMINAMATH_CALUDE_sum_reciprocal_squares_geq_sum_squares_l3421_342182

theorem sum_reciprocal_squares_geq_sum_squares 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (sum_eq_3 : a + b + c = 3) :
  1/a^2 + 1/b^2 + 1/c^2 ≥ a^2 + b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_squares_geq_sum_squares_l3421_342182


namespace NUMINAMATH_CALUDE_right_triangle_legs_l3421_342199

theorem right_triangle_legs (c n : ℝ) (h1 : c > 0) (h2 : n > 0) : 
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧ 
    a^2 + b^2 = c^2 ∧ 
    (n / Real.sqrt 3)^2 = a * b * (1 - ((a + b) / c)^2) ∧
    a = n / 2 ∧ b = c * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l3421_342199


namespace NUMINAMATH_CALUDE_watch_cost_calculation_l3421_342110

/-- The cost of a watch, given the amount saved and the additional amount needed. -/
def watch_cost (saved : ℕ) (additional_needed : ℕ) : ℕ :=
  saved + additional_needed

/-- Theorem: The cost of the watch is $55, given Connie saved $39 and needs $16 more. -/
theorem watch_cost_calculation : watch_cost 39 16 = 55 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_calculation_l3421_342110


namespace NUMINAMATH_CALUDE_unique_solution_l3421_342105

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y : ℝ, f (2 * x - 2 * y) + x = f (3 * x) - f (2 * y) + k * y

/-- The theorem stating the unique solution to the functional equation -/
theorem unique_solution :
  ∃! f : ℝ → ℝ, ∃ k : ℝ, SatisfiesEquation f k ∧ f = id ∧ k = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3421_342105


namespace NUMINAMATH_CALUDE_problem_solution_l3421_342188

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 3 - f a 2 = 1) :
  (∃ m_lower m_upper : ℝ, m_lower = 2/3 ∧ m_upper = 7 ∧
    ∀ m : ℝ, m_lower < m ∧ m < m_upper ↔ f a (3*m - 2) < f a (2*m + 5)) ∧
  (∃ x : ℝ, x = 4 ∧ f a (x - 2/x) = Real.log (7/2) / Real.log (3/2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3421_342188


namespace NUMINAMATH_CALUDE_second_train_speed_l3421_342104

/-- Calculates the speed of the second train given the parameters of two trains passing each other --/
theorem second_train_speed 
  (train1_length : ℝ)
  (train1_speed : ℝ)
  (train2_length : ℝ)
  (time_to_cross : ℝ)
  (h1 : train1_length = 420)
  (h2 : train1_speed = 72)
  (h3 : train2_length = 640)
  (h4 : time_to_cross = 105.99152067834574)
  : ∃ (train2_speed : ℝ), train2_speed = 36 := by
  sorry

end NUMINAMATH_CALUDE_second_train_speed_l3421_342104


namespace NUMINAMATH_CALUDE_hexagon_circle_visibility_l3421_342163

theorem hexagon_circle_visibility (s : ℝ) (r : ℝ) (h1 : s = 3) (h2 : r > 0) : 
  let a := s * Real.sqrt 3 / 2
  (2 * Real.pi * r / 3) / (2 * Real.pi * r) = 1 / 3 → r = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_circle_visibility_l3421_342163


namespace NUMINAMATH_CALUDE_count_equations_l3421_342129

-- Define a function to check if an expression is an equation
def is_equation (expr : String) : Bool :=
  match expr with
  | "5 + 3 = 8" => false
  | "a = 0" => true
  | "y^2 - 2y" => false
  | "x - 3 = 8" => true
  | _ => false

-- Define the list of expressions
def expressions : List String :=
  ["5 + 3 = 8", "a = 0", "y^2 - 2y", "x - 3 = 8"]

-- Theorem to prove
theorem count_equations :
  (expressions.filter is_equation).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_equations_l3421_342129


namespace NUMINAMATH_CALUDE_complement_of_union_l3421_342173

universe u

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def A : Set ℕ := {1,3,5,7}
def B : Set ℕ := {2,4,5}

theorem complement_of_union : 
  (U \ (A ∪ B)) = {6,8} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3421_342173


namespace NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l3421_342112

theorem cone_sphere_volume_ratio (r : ℝ) (h : ℝ) : 
  r > 0 → h = 2 * r → 
  (1 / 3 * π * r^2 * h) / (4 / 3 * π * r^3) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l3421_342112


namespace NUMINAMATH_CALUDE_prism_volume_l3421_342138

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 10)
  (h2 : b * c = 15)
  (h3 : c * a = 18) :
  a * b * c = 30 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l3421_342138
