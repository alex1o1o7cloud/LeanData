import Mathlib

namespace NUMINAMATH_CALUDE_complex_magnitude_in_complement_range_l1832_183259

theorem complex_magnitude_in_complement_range (a : ℝ) : 
  let z : ℂ := 1 + a * Complex.I
  let M : Set ℝ := {x | x > 2}
  Complex.abs z ∈ (Set.Iic 2) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_in_complement_range_l1832_183259


namespace NUMINAMATH_CALUDE_range_of_m_l1832_183280

theorem range_of_m (m : ℝ) : m ≥ 3 ↔ 
  (∀ x : ℝ, (|2*x + 1| ≤ 3 → x^2 - 2*x + 1 - m^2 ≤ 0) ∧ 
  (∃ x : ℝ, |2*x + 1| > 3 ∧ x^2 - 2*x + 1 - m^2 > 0)) ∧ 
  m > 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1832_183280


namespace NUMINAMATH_CALUDE_divisible_by_120_l1832_183224

theorem divisible_by_120 (n : ℤ) : ∃ k : ℤ, n^5 - 5*n^3 + 4*n = 120*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_120_l1832_183224


namespace NUMINAMATH_CALUDE_number_problem_l1832_183225

theorem number_problem (n : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * n = 15 → (40/100 : ℝ) * n = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1832_183225


namespace NUMINAMATH_CALUDE_last_number_proof_l1832_183217

theorem last_number_proof (a b c d : ℝ) : 
  (a + b + c) / 3 = 20 → 
  (b + c + d) / 3 = 15 → 
  a = 33 → 
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_last_number_proof_l1832_183217


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l1832_183218

theorem parallel_vectors_t_value (t : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, t]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, b i = k * a i)) → t = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l1832_183218


namespace NUMINAMATH_CALUDE_painter_problem_l1832_183288

/-- Calculates the total number of rooms to be painted given the painting time per room,
    number of rooms already painted, and remaining painting time. -/
def total_rooms_to_paint (time_per_room : ℕ) (rooms_painted : ℕ) (remaining_time : ℕ) : ℕ :=
  rooms_painted + remaining_time / time_per_room

/-- Proves that the total number of rooms to be painted is 10 given the specific conditions. -/
theorem painter_problem :
  let time_per_room : ℕ := 8
  let rooms_painted : ℕ := 8
  let remaining_time : ℕ := 16
  total_rooms_to_paint time_per_room rooms_painted remaining_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_painter_problem_l1832_183288


namespace NUMINAMATH_CALUDE_measure_string_l1832_183205

theorem measure_string (string_length : ℚ) (h : string_length = 2/3) :
  string_length - (1/4 * string_length) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_measure_string_l1832_183205


namespace NUMINAMATH_CALUDE_als_initial_investment_l1832_183248

theorem als_initial_investment (a b c : ℝ) : 
  a + b + c = 2000 →
  3*a + 2*b + 2*c = 3500 →
  a = 500 := by
sorry

end NUMINAMATH_CALUDE_als_initial_investment_l1832_183248


namespace NUMINAMATH_CALUDE_saree_price_theorem_l1832_183251

/-- The original price of sarees before discounts -/
def original_price : ℝ := 550

/-- The first discount rate -/
def discount1 : ℝ := 0.18

/-- The second discount rate -/
def discount2 : ℝ := 0.12

/-- The final sale price after both discounts -/
def final_price : ℝ := 396.88

/-- Theorem stating that the original price of sarees is approximately 550,
    given the final price after two successive discounts -/
theorem saree_price_theorem :
  ∃ ε > 0, abs (original_price - (final_price / ((1 - discount1) * (1 - discount2)))) < ε :=
sorry

end NUMINAMATH_CALUDE_saree_price_theorem_l1832_183251


namespace NUMINAMATH_CALUDE_anoop_join_time_l1832_183216

/-- Prove that Anoop joined after 6 months given the investment conditions -/
theorem anoop_join_time (arjun_investment : ℕ) (anoop_investment : ℕ) (total_months : ℕ) :
  arjun_investment = 20000 →
  anoop_investment = 40000 →
  total_months = 12 →
  ∃ x : ℕ, 
    (arjun_investment * total_months = anoop_investment * (total_months - x)) ∧
    x = 6 := by
  sorry

end NUMINAMATH_CALUDE_anoop_join_time_l1832_183216


namespace NUMINAMATH_CALUDE_coefficient_of_monomial_l1832_183229

def monomial : ℚ × (ℕ × ℕ × ℕ) := (-2/9, (1, 4, 2))

theorem coefficient_of_monomial :
  (monomial.fst : ℚ) = -2/9 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_monomial_l1832_183229


namespace NUMINAMATH_CALUDE_lucas_class_size_l1832_183238

theorem lucas_class_size : ∃! x : ℕ, 
  70 < x ∧ x < 120 ∧ 
  x % 6 = 4 ∧ 
  x % 5 = 2 ∧ 
  x % 7 = 3 ∧
  x = 148 := by
  sorry

end NUMINAMATH_CALUDE_lucas_class_size_l1832_183238


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l1832_183276

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 32) :
  let side_length := face_perimeter / 4
  let volume := side_length ^ 3
  volume = 512 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l1832_183276


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l1832_183261

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- First line equation: y = x + 1 -/
def line1 (x y : ℝ) : Prop := y = x + 1

/-- Second line equation: y = -x + 1 -/
def line2 (x y : ℝ) : Prop := y = -x + 1

/-- The theorem stating that the intersection point of the two lines is (0, 1) -/
theorem intersection_point_of_lines : 
  ∃ p : IntersectionPoint, line1 p.x p.y ∧ line2 p.x p.y ∧ p.x = 0 ∧ p.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l1832_183261


namespace NUMINAMATH_CALUDE_estimate_value_l1832_183297

theorem estimate_value : 
  3 < (2 * Real.sqrt 2 + Real.sqrt 6) * Real.sqrt (1/2) ∧ 
  (2 * Real.sqrt 2 + Real.sqrt 6) * Real.sqrt (1/2) < 4 := by
  sorry

end NUMINAMATH_CALUDE_estimate_value_l1832_183297


namespace NUMINAMATH_CALUDE_tan_product_seventh_pi_l1832_183265

theorem tan_product_seventh_pi : 
  Real.tan (π / 7) * Real.tan (2 * π / 7) * Real.tan (3 * π / 7) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_seventh_pi_l1832_183265


namespace NUMINAMATH_CALUDE_constant_term_g_l1832_183287

-- Define polynomials f, g, and h
variable (f g h : ℝ[X])

-- Define the conditions
axiom h_def : h = f * g
axiom f_constant : f.coeff 0 = 5
axiom h_constant : h.coeff 0 = -10
axiom g_quadratic : g.degree ≤ 2

-- Theorem to prove
theorem constant_term_g : g.coeff 0 = -2 := by sorry

end NUMINAMATH_CALUDE_constant_term_g_l1832_183287


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l1832_183263

/-- Given a real number a, a function f, and its derivative f', 
    prove that the tangent line at the origin has slope -3 
    when f'(x) is an even function. -/
theorem tangent_line_at_origin (a : ℝ) 
  (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 + a*x^2 + (a-3)*x) 
  (h2 : ∀ x, (deriv f) x = f' x) 
  (h3 : ∀ x, f' x = f' (-x)) : 
  (deriv f) 0 = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l1832_183263


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l1832_183201

def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 10*x^3 + 20*x^2 - 5*x - 21

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := by sorry

theorem polynomial_remainder (x : ℝ) :
  ∃ q : ℝ → ℝ, f x = (x - 2) * q x + 33 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l1832_183201


namespace NUMINAMATH_CALUDE_tan_105_degrees_l1832_183208

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l1832_183208


namespace NUMINAMATH_CALUDE_greatest_3digit_base9_divisible_by_7_l1832_183220

/-- Converts a base 9 number to base 10 --/
def base9ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 9^2 + tens * 9 + ones

/-- Checks if a number is a valid 3-digit base 9 number --/
def isValidBase9 (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 888

theorem greatest_3digit_base9_divisible_by_7 :
  ∃ (n : Nat), isValidBase9 n ∧ 
               base9ToBase10 n % 7 = 0 ∧
               ∀ (m : Nat), isValidBase9 m ∧ base9ToBase10 m % 7 = 0 → base9ToBase10 m ≤ base9ToBase10 n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_3digit_base9_divisible_by_7_l1832_183220


namespace NUMINAMATH_CALUDE_exists_coplanar_even_sum_l1832_183289

-- Define a cube as a set of 8 integers (representing the labels on vertices)
def Cube := Fin 8 → ℤ

-- Define a function to check if a set of four vertices is coplanar
def isCoplanar (v1 v2 v3 v4 : Fin 8) : Prop := sorry

-- Define a function to check if the sum of four integers is even
def sumIsEven (a b c d : ℤ) : Prop :=
  (a + b + c + d) % 2 = 0

-- Theorem statement
theorem exists_coplanar_even_sum (cube : Cube) :
  ∃ (v1 v2 v3 v4 : Fin 8), isCoplanar v1 v2 v3 v4 ∧ sumIsEven (cube v1) (cube v2) (cube v3) (cube v4) := by
  sorry

end NUMINAMATH_CALUDE_exists_coplanar_even_sum_l1832_183289


namespace NUMINAMATH_CALUDE_todds_initial_gum_l1832_183252

theorem todds_initial_gum (initial : ℕ) : 
  (∃ (after_steve after_emily : ℕ),
    after_steve = initial + 16 ∧
    after_emily = after_steve - 12 ∧
    after_emily = 54) →
  initial = 50 := by
sorry

end NUMINAMATH_CALUDE_todds_initial_gum_l1832_183252


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l1832_183292

theorem min_value_trig_expression : 
  ∀ x : ℝ, (Real.sin x)^8 + (Real.cos x)^8 + 1 ≥ (9/10) * ((Real.sin x)^6 + (Real.cos x)^6 + 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l1832_183292


namespace NUMINAMATH_CALUDE_equation_solution_l1832_183284

theorem equation_solution : ∃ x : ℝ, 4*x + 6*x = 360 - 10*(x - 4) ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1832_183284


namespace NUMINAMATH_CALUDE_flowers_per_bouquet_l1832_183221

theorem flowers_per_bouquet (total_flowers : ℕ) (wilted_flowers : ℕ) (num_bouquets : ℕ) :
  total_flowers = 88 →
  wilted_flowers = 48 →
  num_bouquets = 8 →
  (total_flowers - wilted_flowers) / num_bouquets = 5 :=
by sorry

end NUMINAMATH_CALUDE_flowers_per_bouquet_l1832_183221


namespace NUMINAMATH_CALUDE_quadratic_inequality_integer_set_l1832_183262

theorem quadratic_inequality_integer_set :
  {x : ℤ | x^2 - 3*x - 4 < 0} = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_integer_set_l1832_183262


namespace NUMINAMATH_CALUDE_least_expensive_trip_l1832_183281

-- Define the triangle
def triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 5000^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 4000^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 - (C.1 - A.1)^2 - (C.2 - A.2)^2

-- Define travel costs
def car_cost (distance : ℝ) : ℝ := 0.20 * distance
def train_cost (distance : ℝ) : ℝ := 150 + 0.15 * distance

-- Define the total trip cost
def trip_cost (AB BC CA : ℝ) (mode_AB mode_BC mode_CA : Bool) : ℝ :=
  (if mode_AB then train_cost AB else car_cost AB) +
  (if mode_BC then train_cost BC else car_cost BC) +
  (if mode_CA then train_cost CA else car_cost CA)

-- Theorem statement
theorem least_expensive_trip (A B C : ℝ × ℝ) :
  triangle A B C →
  ∃ (mode_AB mode_BC mode_CA : Bool),
    ∀ (other_mode_AB other_mode_BC other_mode_CA : Bool),
      trip_cost 5000 22500 4000 mode_AB mode_BC mode_CA ≤
      trip_cost 5000 22500 4000 other_mode_AB other_mode_BC other_mode_CA ∧
      trip_cost 5000 22500 4000 mode_AB mode_BC mode_CA = 5130 :=
sorry

end NUMINAMATH_CALUDE_least_expensive_trip_l1832_183281


namespace NUMINAMATH_CALUDE_problem_statement_l1832_183228

theorem problem_statement (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 6) :
  a * b^2 - a^2 * b = -24 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1832_183228


namespace NUMINAMATH_CALUDE_square_difference_formula_l1832_183230

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 9/17) (h2 : x - y = 1/19) : x^2 - y^2 = 9/323 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l1832_183230


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1832_183295

-- Define the solution sets
def solution_set_1 : Set ℝ := {x : ℝ | (x + 3) * (x - 1) = 0}
def solution_set_2 : Set ℝ := {x : ℝ | x - 1 = 0}

-- State the theorem
theorem necessary_but_not_sufficient :
  (solution_set_2 ⊆ solution_set_1) ∧ (solution_set_2 ≠ solution_set_1) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1832_183295


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1832_183234

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that in an arithmetic sequence where a₂ + 2a₆ + a₁₀ = 120, a₃ + a₉ = 60. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) 
    (h_sum : a 2 + 2 * a 6 + a 10 = 120) : a 3 + a 9 = 60 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1832_183234


namespace NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l1832_183272

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_k_value
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h1 : a 4 + a 7 + a 10 = 17)
  (h2 : a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 + a 13 + a 14 = 77)
  (h3 : ∃ k : ℕ, a k = 13) :
  ∃ k : ℕ, a k = 13 ∧ k = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l1832_183272


namespace NUMINAMATH_CALUDE_sum_six_smallest_multiples_of_11_l1832_183244

theorem sum_six_smallest_multiples_of_11 : 
  (Finset.range 6).sum (fun i => 11 * (i + 1)) = 231 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_smallest_multiples_of_11_l1832_183244


namespace NUMINAMATH_CALUDE_travel_agency_comparison_l1832_183254

/-- Represents the cost function for a travel agency --/
def CostFunction := Nat → Nat

/-- The full ticket price --/
def fullPrice : Nat := 240

/-- Cost function for Agency A --/
def costA : CostFunction := fun x => 120 * x + 240

/-- Cost function for Agency B --/
def costB : CostFunction := fun x => 144 * (x + 1)

theorem travel_agency_comparison :
  (∀ x, costA x = 120 * x + 240) ∧
  (∀ x, costB x = 144 * (x + 1)) ∧
  (costA 10 < costB 10) ∧
  (costA 4 = costB 4) :=
sorry

end NUMINAMATH_CALUDE_travel_agency_comparison_l1832_183254


namespace NUMINAMATH_CALUDE_price_before_discount_l1832_183200

theorem price_before_discount (reduced_price : ℝ) (discount_rate : ℝ) (original_price : ℝ) : 
  reduced_price = 620 → 
  discount_rate = 0.5 → 
  reduced_price = original_price * (1 - discount_rate) → 
  original_price = 1240 := by
sorry

end NUMINAMATH_CALUDE_price_before_discount_l1832_183200


namespace NUMINAMATH_CALUDE_girl_walking_distance_l1832_183285

/-- The distance traveled by a girl walking at a constant speed for a given time. -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: A girl walking at 5 kmph for 6 hours travels 30 kilometers. -/
theorem girl_walking_distance :
  let speed : ℝ := 5
  let time : ℝ := 6
  distance_traveled speed time = 30 := by
  sorry

end NUMINAMATH_CALUDE_girl_walking_distance_l1832_183285


namespace NUMINAMATH_CALUDE_power_five_remainder_l1832_183298

theorem power_five_remainder (n : ℕ) : (5^1234 : ℕ) % 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_five_remainder_l1832_183298


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1832_183210

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 99) 
  (h2 : a*b + b*c + a*c = 131) : 
  a + b + c = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1832_183210


namespace NUMINAMATH_CALUDE_factory_production_l1832_183207

/-- Represents the production of toys in a factory -/
structure ToyProduction where
  days_per_week : ℕ
  toys_per_day : ℕ

/-- Calculates the total number of toys produced per week -/
def toys_per_week (prod : ToyProduction) : ℕ :=
  prod.days_per_week * prod.toys_per_day

/-- Theorem: Given the conditions, the factory produces 5505 toys per week -/
theorem factory_production :
  ∀ (prod : ToyProduction),
  prod.days_per_week = 5 →
  prod.toys_per_day = 1101 →
  toys_per_week prod = 5505 :=
by
  sorry

end NUMINAMATH_CALUDE_factory_production_l1832_183207


namespace NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l1832_183299

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space defined by ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point satisfies the equation 9x^2 - 25y^2 = 0 -/
def satisfiesEquation (p : Point2D) : Prop :=
  9 * p.x^2 - 25 * p.y^2 = 0

/-- Checks if a point lies on a given line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The two lines that form the solution -/
def line1 : Line2D := { a := 3, b := -5, c := 0 }
def line2 : Line2D := { a := 3, b := 5, c := 0 }

/-- Theorem stating that the equation represents a pair of straight lines -/
theorem equation_represents_pair_of_lines :
  ∀ p : Point2D, satisfiesEquation p ↔ (pointOnLine p line1 ∨ pointOnLine p line2) :=
sorry


end NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l1832_183299


namespace NUMINAMATH_CALUDE_f_minimum_f_le_g_iff_exists_three_roots_l1832_183231

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := x * Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x

-- Statement 1: f has a minimum at x = 1/e
theorem f_minimum : ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x := by sorry

-- Statement 2: f(x) ≤ g(x) for all x > 0 iff a ≥ 1
theorem f_le_g_iff (a : ℝ) : (∀ x > 0, f x ≤ g a x) ↔ a ≥ 1 := by sorry

-- Statement 3: When a = 1/8, there exists m such that 3f(x)/(4x) + m + g(x) = 0 has three distinct real roots iff 7/8 < m < 15/8 - 3/4 * ln 3
theorem exists_three_roots :
  ∃ (m : ℝ), (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (3 * f x) / (4 * x) + m + g (1/8) x = 0 ∧
    (3 * f y) / (4 * y) + m + g (1/8) y = 0 ∧
    (3 * f z) / (4 * z) + m + g (1/8) z = 0) ↔
  (7/8 < m ∧ m < 15/8 - 3/4 * Real.log 3) := by sorry

end

end NUMINAMATH_CALUDE_f_minimum_f_le_g_iff_exists_three_roots_l1832_183231


namespace NUMINAMATH_CALUDE_fraction_division_simplify_fraction_divide_fractions_l1832_183246

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem simplify_fraction (n d : ℤ) (hd : d ≠ 0) :
  (n / d : ℚ) = (n / gcd n d) / (d / gcd n d) :=
by sorry

theorem divide_fractions : (5 / 6 : ℚ) / (-9 / 10) = -25 / 27 :=
by sorry

end NUMINAMATH_CALUDE_fraction_division_simplify_fraction_divide_fractions_l1832_183246


namespace NUMINAMATH_CALUDE_acid_concentration_percentage_l1832_183223

/-- 
Given a solution with 1.6 litres of pure acid in 8 litres of total volume,
prove that the percentage concentration of the acid is 20%.
-/
theorem acid_concentration_percentage (pure_acid : ℝ) (total_volume : ℝ) :
  pure_acid = 1.6 →
  total_volume = 8 →
  (pure_acid / total_volume) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_acid_concentration_percentage_l1832_183223


namespace NUMINAMATH_CALUDE_geometric_sequence_a1_l1832_183237

/-- A monotonically decreasing geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 0 < q ∧ q < 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a1 (a : ℕ → ℝ) :
  GeometricSequence a →
  a 3 = 1 →
  a 2 + a 4 = 5/2 →
  a 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a1_l1832_183237


namespace NUMINAMATH_CALUDE_third_grade_boys_count_l1832_183227

/-- The number of third-grade boys in an elementary school -/
def third_grade_boys (total : ℕ) (fourth_grade_excess : ℕ) (third_grade_girl_deficit : ℕ) : ℕ :=
  let third_graders := (total - fourth_grade_excess) / 2
  let third_grade_boys := (third_graders + third_grade_girl_deficit) / 2
  third_grade_boys

/-- Theorem stating the number of third-grade boys given the conditions -/
theorem third_grade_boys_count :
  third_grade_boys 531 31 22 = 136 :=
by sorry

end NUMINAMATH_CALUDE_third_grade_boys_count_l1832_183227


namespace NUMINAMATH_CALUDE_problem_solution_l1832_183239

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -12)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 15) :
  b / (a + b) + c / (b + c) + a / (c + a) = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1832_183239


namespace NUMINAMATH_CALUDE_doubled_container_volume_l1832_183279

/-- The volume of a container after doubling its dimensions -/
def doubled_volume (original_volume : ℝ) : ℝ := 8 * original_volume

/-- Theorem: Doubling the dimensions of a 3-gallon container results in a 24-gallon container -/
theorem doubled_container_volume : doubled_volume 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_doubled_container_volume_l1832_183279


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l1832_183222

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- The measure of each interior angle in a regular octagon -/
def octagon_interior_angle : ℝ := 135

/-- Theorem: Each interior angle of a regular octagon measures 135 degrees -/
theorem regular_octagon_interior_angle :
  (180 * (octagon_sides - 2 : ℝ)) / octagon_sides = octagon_interior_angle :=
sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l1832_183222


namespace NUMINAMATH_CALUDE_fraction_problem_l1832_183282

theorem fraction_problem (t k : ℚ) (f : ℚ) : 
  t = f * (k - 32) → t = 75 → k = 167 → f = 5/9 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1832_183282


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1832_183269

theorem min_value_of_expression (x y : ℝ) : (x * y - 2)^2 + (x + y - 1)^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1832_183269


namespace NUMINAMATH_CALUDE_robin_camera_pictures_l1832_183206

/-- The number of pictures Robin uploaded from her camera -/
def pictures_from_camera (total_albums : ℕ) (pictures_per_album : ℕ) (pictures_from_phone : ℕ) : ℕ :=
  total_albums * pictures_per_album - pictures_from_phone

/-- Theorem stating that Robin uploaded 5 pictures from her camera -/
theorem robin_camera_pictures :
  pictures_from_camera 5 8 35 = 5 := by
  sorry

end NUMINAMATH_CALUDE_robin_camera_pictures_l1832_183206


namespace NUMINAMATH_CALUDE_construction_problem_l1832_183294

/-- Represents the construction plan for a quarter --/
structure ConstructionPlan where
  ordinary : ℝ
  elevated : ℝ
  tunnel : ℝ

/-- Represents the cost per kilometer for each type of construction --/
structure CostPerKm where
  ordinary : ℝ
  elevated : ℝ
  tunnel : ℝ

/-- Calculates the total cost of a construction plan given the cost per kilometer --/
def totalCost (plan : ConstructionPlan) (cost : CostPerKm) : ℝ :=
  plan.ordinary * cost.ordinary + plan.elevated * cost.elevated + plan.tunnel * cost.tunnel

theorem construction_problem (a : ℝ) :
  let q1_plan : ConstructionPlan := { ordinary := 32, elevated := 21, tunnel := 3 }
  let q1_cost : CostPerKm := { ordinary := 1, elevated := 2, tunnel := 4 }
  let q2_plan : ConstructionPlan := { ordinary := 32 - 9*a, elevated := 21 - 2*a, tunnel := 3 + a }
  let q2_cost : CostPerKm := { ordinary := 1, elevated := 2 + 0.5*a, tunnel := 4 }
  
  (∀ x, x ≤ 3 → 56 - 32 - x ≥ 7*x) ∧ 
  (totalCost q1_plan q1_cost = totalCost q2_plan q2_cost) →
  a = 3/2 := by sorry

end NUMINAMATH_CALUDE_construction_problem_l1832_183294


namespace NUMINAMATH_CALUDE_prob_two_red_cards_l1832_183270

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)

/-- Defines a standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    suits := 4,
    cards_per_suit := 13,
    red_suits := 2,
    black_suits := 2 }

/-- The probability of drawing a red card from the deck -/
def prob_red_card (d : Deck) : ℚ :=
  (d.red_suits * d.cards_per_suit) / d.total_cards

/-- Theorem: The probability of drawing two red cards in succession with replacement is 1/4 -/
theorem prob_two_red_cards (d : Deck) (h : d = standard_deck) :
  (prob_red_card d) * (prob_red_card d) = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_prob_two_red_cards_l1832_183270


namespace NUMINAMATH_CALUDE_david_average_marks_l1832_183256

def david_marks : List ℕ := [76, 65, 82, 67, 85]

theorem david_average_marks :
  (david_marks.sum / david_marks.length : ℚ) = 75 := by sorry

end NUMINAMATH_CALUDE_david_average_marks_l1832_183256


namespace NUMINAMATH_CALUDE_value_144_is_square_iff_b_gt_4_l1832_183247

/-- The value of 144 in base b -/
def value_in_base_b (b : ℕ) : ℕ := b^2 + 4*b + 4

/-- A number is a perfect square if it has an integer square root -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m^2 = n

theorem value_144_is_square_iff_b_gt_4 (b : ℕ) :
  is_perfect_square (value_in_base_b b) ↔ b > 4 :=
sorry

end NUMINAMATH_CALUDE_value_144_is_square_iff_b_gt_4_l1832_183247


namespace NUMINAMATH_CALUDE_geometric_number_difference_l1832_183268

/-- A geometric number is a 3-digit number with distinct digits forming a geometric sequence,
    and the middle digit is odd. -/
def IsGeometricNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    Odd b ∧
    b * b = a * c

theorem geometric_number_difference :
  ∃ (min max : ℕ),
    IsGeometricNumber min ∧
    IsGeometricNumber max ∧
    (∀ n, IsGeometricNumber n → min ≤ n ∧ n ≤ max) ∧
    max - min = 220 := by
  sorry

end NUMINAMATH_CALUDE_geometric_number_difference_l1832_183268


namespace NUMINAMATH_CALUDE_dragon_can_be_defeated_l1832_183232

/-- Represents the possible strikes and their corresponding regrowth --/
inductive Strike : Type
| one : Strike
| seventeen : Strike
| twentyone : Strike
| thirtythree : Strike

/-- Returns the number of heads chopped for a given strike --/
def heads_chopped (s : Strike) : ℕ :=
  match s with
  | Strike.one => 1
  | Strike.seventeen => 17
  | Strike.twentyone => 21
  | Strike.thirtythree => 33

/-- Returns the number of heads that grow back for a given strike --/
def heads_regrown (s : Strike) : ℕ :=
  match s with
  | Strike.one => 10
  | Strike.seventeen => 14
  | Strike.twentyone => 0
  | Strike.thirtythree => 48

/-- Represents the state of the dragon --/
structure DragonState :=
  (heads : ℕ)

/-- Applies a strike to the dragon state --/
def apply_strike (state : DragonState) (s : Strike) : DragonState :=
  let new_heads := state.heads - heads_chopped s + heads_regrown s
  ⟨max new_heads 0⟩

/-- Theorem: There exists a sequence of strikes that defeats the dragon --/
theorem dragon_can_be_defeated :
  ∃ (sequence : List Strike), (sequence.foldl apply_strike ⟨2000⟩).heads = 0 :=
sorry

end NUMINAMATH_CALUDE_dragon_can_be_defeated_l1832_183232


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_leq_eight_l1832_183204

theorem sqrt_meaningful_iff_leq_eight (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 8 - x) ↔ x ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_leq_eight_l1832_183204


namespace NUMINAMATH_CALUDE_distribute_six_books_l1832_183240

/-- The number of ways to distribute n different books among two people, 
    with each person getting one book. -/
def distribute_books (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem stating that distributing 6 different books among two people, 
    with each person getting one book, can be done in 30 different ways. -/
theorem distribute_six_books : distribute_books 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_books_l1832_183240


namespace NUMINAMATH_CALUDE_f_x_plus_one_l1832_183286

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 + 4*(x + 1) - 5

-- State the theorem
theorem f_x_plus_one (x : ℝ) : f (x + 1) = x^2 + 8*x + 7 := by
  sorry

end NUMINAMATH_CALUDE_f_x_plus_one_l1832_183286


namespace NUMINAMATH_CALUDE_distinct_nonneg_inequality_l1832_183233

theorem distinct_nonneg_inequality (a b c : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  a^2 + b^2 + c^2 > Real.sqrt (a*b*c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_distinct_nonneg_inequality_l1832_183233


namespace NUMINAMATH_CALUDE_matt_profit_l1832_183212

/-- Represents a baseball card with its value -/
structure Card where
  value : ℕ

/-- Represents a trade of cards -/
structure Trade where
  cardsGiven : List Card
  cardsReceived : List Card

def initialCards : List Card := List.replicate 8 ⟨6⟩

def trade1 : Trade := {
  cardsGiven := [⟨6⟩, ⟨6⟩],
  cardsReceived := [⟨2⟩, ⟨2⟩, ⟨2⟩, ⟨9⟩]
}

def trade2 : Trade := {
  cardsGiven := [⟨2⟩, ⟨6⟩],
  cardsReceived := [⟨5⟩, ⟨5⟩, ⟨8⟩]
}

def trade3 : Trade := {
  cardsGiven := [⟨5⟩, ⟨9⟩],
  cardsReceived := [⟨3⟩, ⟨3⟩, ⟨3⟩, ⟨10⟩, ⟨1⟩]
}

def cardValue (c : Card) : ℕ := c.value

def tradeProfit (t : Trade) : ℤ :=
  (t.cardsReceived.map cardValue).sum - (t.cardsGiven.map cardValue).sum

theorem matt_profit :
  (tradeProfit trade1 + tradeProfit trade2 + tradeProfit trade3 : ℤ) = 19 := by
  sorry

end NUMINAMATH_CALUDE_matt_profit_l1832_183212


namespace NUMINAMATH_CALUDE_candy_distribution_l1832_183274

theorem candy_distribution (anna_candy_per_house : ℕ) (anna_houses : ℕ) 
  (billy_candy_per_house : ℕ) (candy_difference : ℕ) :
  anna_candy_per_house = 14 →
  anna_houses = 60 →
  billy_candy_per_house = 11 →
  anna_candy_per_house * anna_houses = billy_candy_per_house * (anna_houses + 15) + candy_difference →
  anna_houses + 15 = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1832_183274


namespace NUMINAMATH_CALUDE_student_rank_problem_l1832_183257

/-- Given a total number of students and a student's rank from the right,
    calculates the student's rank from the left. -/
def rank_from_left (total : ℕ) (rank_from_right : ℕ) : ℕ :=
  total - rank_from_right + 1

/-- Proves that for 21 total students and a student ranked 16th from the right,
    the student's rank from the left is 6. -/
theorem student_rank_problem :
  rank_from_left 21 16 = 6 := by
  sorry

#eval rank_from_left 21 16

end NUMINAMATH_CALUDE_student_rank_problem_l1832_183257


namespace NUMINAMATH_CALUDE_inequality_theorem_l1832_183215

theorem inequality_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (b^2 / a) + (a^2 / b) ≥ a + b := by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1832_183215


namespace NUMINAMATH_CALUDE_goals_theorem_l1832_183266

/-- The total number of goals scored in the league against Barca -/
def total_goals : ℕ := 300

/-- The number of players who scored goals -/
def num_players : ℕ := 2

/-- The number of goals scored by each player -/
def goals_per_player : ℕ := 30

/-- The percentage of total goals scored by the two players -/
def percentage : ℚ := 1/5

theorem goals_theorem (h1 : num_players * goals_per_player = (percentage * total_goals).num) :
  total_goals = 300 := by
  sorry

end NUMINAMATH_CALUDE_goals_theorem_l1832_183266


namespace NUMINAMATH_CALUDE_apple_distribution_l1832_183290

theorem apple_distribution (total_apples : ℕ) (num_friends : ℕ) (apples_per_friend : ℕ) : 
  total_apples = 9 → num_friends = 3 → total_apples / num_friends = apples_per_friend → apples_per_friend = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l1832_183290


namespace NUMINAMATH_CALUDE_mary_earnings_proof_l1832_183273

/-- Calculates Mary's weekly earnings after deductions --/
def maryWeeklyEarnings (
  maxHours : Nat)
  (regularRate : ℚ)
  (overtimeRateIncrease : ℚ)
  (additionalRateIncrease : ℚ)
  (regularHours : Nat)
  (overtimeHours : Nat)
  (taxRate1 : ℚ)
  (taxRate2 : ℚ)
  (taxRate3 : ℚ)
  (taxThreshold1 : ℚ)
  (taxThreshold2 : ℚ)
  (insuranceFee : ℚ)
  (weekendBonus : ℚ)
  (weekendShiftHours : Nat) : ℚ :=
  sorry

theorem mary_earnings_proof :
  maryWeeklyEarnings 70 10 0.3 0.6 40 20 0.15 0.1 0.25 400 600 50 75 8 = 691.25 := by
  sorry

end NUMINAMATH_CALUDE_mary_earnings_proof_l1832_183273


namespace NUMINAMATH_CALUDE_multiples_of_six_ending_in_four_l1832_183277

theorem multiples_of_six_ending_in_four (n : ℕ) : 
  (∃ m : ℕ, m = 10) ↔ 
  (∀ k : ℕ, (6 * k < 600 ∧ (6 * k) % 10 = 4) → k ≤ n) ∧ 
  (∃ (k₁ k₂ : ℕ), k₁ ≤ n ∧ k₂ ≤ n ∧ k₁ ≠ k₂ ∧ 
    6 * k₁ < 600 ∧ (6 * k₁) % 10 = 4 ∧ 
    6 * k₂ < 600 ∧ (6 * k₂) % 10 = 4) :=
by sorry

end NUMINAMATH_CALUDE_multiples_of_six_ending_in_four_l1832_183277


namespace NUMINAMATH_CALUDE_line_equation_proof_l1832_183235

/-- A line in the xy-plane passing through two points -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- The line passing through (2,3) and (4,4) -/
def line_through_points : Line :=
  { a := 1, b := 8, c := -26 }

theorem line_equation_proof :
  (line_through_points.contains 2 3) ∧
  (line_through_points.contains 4 4) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1832_183235


namespace NUMINAMATH_CALUDE_eighth_term_is_six_l1832_183219

/-- An arithmetic progression with given conditions -/
structure ArithmeticProgression where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 2
  sum_condition : a 3 + a 6 = 8

/-- The 8th term of the arithmetic progression is 6 -/
theorem eighth_term_is_six (ap : ArithmeticProgression) : ap.a 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_six_l1832_183219


namespace NUMINAMATH_CALUDE_value_of_T_l1832_183267

theorem value_of_T : ∃ T : ℚ, (1/2 : ℚ) * (1/7 : ℚ) * T = (1/3 : ℚ) * (1/5 : ℚ) * 60 ∧ T = 56 := by
  sorry

end NUMINAMATH_CALUDE_value_of_T_l1832_183267


namespace NUMINAMATH_CALUDE_sixteen_horses_walking_legs_l1832_183242

/-- Given a number of horses and an equal number of men, with half riding and half walking,
    calculate the number of legs walking on the ground. -/
def legs_walking (num_horses : ℕ) : ℕ :=
  let num_men := num_horses
  let num_walking_men := num_men / 2
  let men_legs := num_walking_men * 2
  let horse_legs := num_horses * 4
  men_legs + horse_legs

/-- Theorem stating that with 16 horses and men, half riding and half walking,
    there are 80 legs walking on the ground. -/
theorem sixteen_horses_walking_legs :
  legs_walking 16 = 80 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_horses_walking_legs_l1832_183242


namespace NUMINAMATH_CALUDE_dog_area_theorem_l1832_183293

/-- Represents a rectangular obstruction -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the position where the dog is tied -/
structure TiePoint where
  distance_from_midpoint : ℝ

/-- Calculates the area accessible by a dog tied to a point near a rectangular obstruction -/
def accessible_area (rect : Rectangle) (tie : TiePoint) (rope_length : ℝ) : ℝ :=
  sorry

/-- Theorem stating the accessible area for the given problem -/
theorem dog_area_theorem (rect : Rectangle) (tie : TiePoint) (rope_length : ℝ) :
  rect.length = 20 ∧ rect.width = 10 ∧ tie.distance_from_midpoint = 5 ∧ rope_length = 10 →
  accessible_area rect tie rope_length = 62.5 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_dog_area_theorem_l1832_183293


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l1832_183271

/-- A geometric sequence of positive integers -/
def GeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℕ)
  (h_geometric : GeometricSequence a)
  (h_first : a 1 = 3)
  (h_fifth : a 5 = 243) :
  a 6 = 729 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l1832_183271


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l1832_183214

theorem smallest_m_for_integral_solutions : ∃ (m : ℕ), 
  (m > 0) ∧ 
  (∃ (x : ℤ), 12 * x^2 - m * x + 432 = 0) ∧
  (∀ (k : ℕ), k > 0 ∧ k < m → ¬∃ (y : ℤ), 12 * y^2 - k * y + 432 = 0) ∧
  m = 144 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l1832_183214


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_6_not_4_l1832_183203

theorem smallest_two_digit_multiple_of_6_not_4 :
  ∃ n : ℕ, 
    n ≥ 10 ∧ n < 100 ∧  -- two-digit positive integer
    n % 6 = 0 ∧         -- multiple of 6
    n % 4 ≠ 0 ∧         -- not a multiple of 4
    (∀ m : ℕ, m ≥ 10 ∧ m < 100 ∧ m % 6 = 0 ∧ m % 4 ≠ 0 → n ≤ m) ∧  -- smallest such number
    n = 18 :=           -- the number is 18
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_6_not_4_l1832_183203


namespace NUMINAMATH_CALUDE_equation_solution_l1832_183226

theorem equation_solution : 
  ∃ x : ℝ, (5 * 1.6 - (2 * x) / 1.3 = 4) ∧ (x = 2.6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1832_183226


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1832_183278

theorem sqrt_equation_solution : ∃ x : ℝ, x = 1225 / 36 ∧ Real.sqrt x + Real.sqrt (x + 4) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1832_183278


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1832_183275

theorem fraction_multiplication : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5060 = 759 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1832_183275


namespace NUMINAMATH_CALUDE_a6_is_2_in_factorial_base_of_1735_l1832_183211

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def factorial_base_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  (n / factorial k) % (k + 1)

theorem a6_is_2_in_factorial_base_of_1735 :
  factorial_base_coefficient 1735 6 = 2 := by sorry

end NUMINAMATH_CALUDE_a6_is_2_in_factorial_base_of_1735_l1832_183211


namespace NUMINAMATH_CALUDE_lcm_18_24_l1832_183250

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l1832_183250


namespace NUMINAMATH_CALUDE_probability_not_perfect_power_l1832_183283

/-- A number is a perfect power if it can be expressed as x^y where x and y are integers and y > 1 -/
def IsPerfectPower (n : ℕ) : Prop :=
  ∃ x y : ℕ, y > 1 ∧ n = x^y

/-- The count of numbers from 1 to 200 that are not perfect powers -/
def CountNotPerfectPower : ℕ := 179

theorem probability_not_perfect_power :
  (CountNotPerfectPower : ℚ) / 200 = 179 / 200 :=
sorry

end NUMINAMATH_CALUDE_probability_not_perfect_power_l1832_183283


namespace NUMINAMATH_CALUDE_smallest_element_mean_l1832_183296

/-- The arithmetic mean of the smallest number in all r-element subsets of {1, 2, ..., n} -/
def f (r n : ℕ+) : ℚ :=
  (n + 1) / (r + 1)

/-- Theorem stating that f(r, n) is the arithmetic mean of the smallest number
    in all r-element subsets of {1, 2, ..., n} -/
theorem smallest_element_mean (r n : ℕ+) (h : r ≤ n) :
  f r n = (Finset.sum (Finset.range (n - r + 1)) (fun a => a * (Nat.choose (n - a) (r - 1)))) /
          (Nat.choose n r) :=
sorry

end NUMINAMATH_CALUDE_smallest_element_mean_l1832_183296


namespace NUMINAMATH_CALUDE_quiz_contest_orderings_l1832_183258

theorem quiz_contest_orderings (n : ℕ) (h : n = 5) : Nat.factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_quiz_contest_orderings_l1832_183258


namespace NUMINAMATH_CALUDE_lcm_150_540_l1832_183264

theorem lcm_150_540 : Nat.lcm 150 540 = 2700 := by
  sorry

end NUMINAMATH_CALUDE_lcm_150_540_l1832_183264


namespace NUMINAMATH_CALUDE_total_yellow_marbles_l1832_183213

/-- The total number of yellow marbles given the number of marbles each person has -/
def total_marbles (mary_marbles joan_marbles john_marbles : ℕ) : ℕ :=
  mary_marbles + joan_marbles + john_marbles

/-- Theorem stating that the total number of yellow marbles is 19 -/
theorem total_yellow_marbles :
  total_marbles 9 3 7 = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_yellow_marbles_l1832_183213


namespace NUMINAMATH_CALUDE_red_bacon_bits_count_l1832_183236

def salad_problem (mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits : ℕ) : Prop :=
  mushrooms = 3 ∧
  cherry_tomatoes = 2 * mushrooms ∧
  pickles = 4 * cherry_tomatoes ∧
  bacon_bits = 4 * pickles ∧
  red_bacon_bits = bacon_bits / 3

theorem red_bacon_bits_count : ∃ (mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits : ℕ),
  salad_problem mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits ∧ red_bacon_bits = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_red_bacon_bits_count_l1832_183236


namespace NUMINAMATH_CALUDE_units_digit_of_product_units_digit_of_27_times_68_l1832_183245

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The product of two natural numbers has the same units digit as the product of their units digits -/
theorem units_digit_of_product (a b : ℕ) :
  unitsDigit (a * b) = unitsDigit (unitsDigit a * unitsDigit b) := by sorry

theorem units_digit_of_27_times_68 :
  unitsDigit (27 * 68) = 6 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_units_digit_of_27_times_68_l1832_183245


namespace NUMINAMATH_CALUDE_alpha_30_sufficient_not_necessary_for_sin_half_l1832_183253

open Real

theorem alpha_30_sufficient_not_necessary_for_sin_half :
  (∃ α : ℝ, α = 30 * π / 180 → sin α = 1/2) ∧
  (∃ α : ℝ, sin α = 1/2 ∧ α ≠ 30 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_alpha_30_sufficient_not_necessary_for_sin_half_l1832_183253


namespace NUMINAMATH_CALUDE_prob_both_white_is_zero_l1832_183241

/-- Two boxes containing marbles -/
structure TwoBoxes where
  box1 : Finset ℕ
  box2 : Finset ℕ
  total_marbles : box1.card + box2.card = 36
  box1_black : ∀ m ∈ box1, m = 0  -- 0 represents black marbles
  prob_both_black : (box1.card : ℚ) / 36 * (box2.filter (λ m => m = 0)).card / box2.card = 18 / 25

/-- The probability of drawing two white marbles -/
def prob_both_white (boxes : TwoBoxes) : ℚ :=
  (boxes.box1.filter (λ m => m ≠ 0)).card / boxes.box1.card *
  (boxes.box2.filter (λ m => m ≠ 0)).card / boxes.box2.card

theorem prob_both_white_is_zero (boxes : TwoBoxes) : prob_both_white boxes = 0 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_white_is_zero_l1832_183241


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1832_183255

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36 →
  a 5 + a 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1832_183255


namespace NUMINAMATH_CALUDE_cookout_bun_packs_l1832_183249

/-- Calculate the number of bun packs needed for a cookout --/
theorem cookout_bun_packs 
  (total_friends : ℕ) 
  (burgers_per_guest : ℕ) 
  (non_meat_eaters : ℕ) 
  (no_bread_eaters : ℕ) 
  (gluten_free_friends : ℕ) 
  (nut_allergy_friends : ℕ)
  (regular_buns_per_pack : ℕ) 
  (gluten_free_buns_per_pack : ℕ) 
  (nut_free_buns_per_pack : ℕ)
  (h1 : total_friends = 35)
  (h2 : burgers_per_guest = 3)
  (h3 : non_meat_eaters = 7)
  (h4 : no_bread_eaters = 4)
  (h5 : gluten_free_friends = 3)
  (h6 : nut_allergy_friends = 1)
  (h7 : regular_buns_per_pack = 15)
  (h8 : gluten_free_buns_per_pack = 6)
  (h9 : nut_free_buns_per_pack = 5) :
  (((total_friends - non_meat_eaters) * burgers_per_guest - no_bread_eaters * burgers_per_guest + regular_buns_per_pack - 1) / regular_buns_per_pack = 5) ∧ 
  ((gluten_free_friends * burgers_per_guest + gluten_free_buns_per_pack - 1) / gluten_free_buns_per_pack = 2) ∧
  ((nut_allergy_friends * burgers_per_guest + nut_free_buns_per_pack - 1) / nut_free_buns_per_pack = 1) :=
by sorry

end NUMINAMATH_CALUDE_cookout_bun_packs_l1832_183249


namespace NUMINAMATH_CALUDE_min_value_shifted_quadratic_l1832_183243

/-- Given a quadratic function f(x) = x^2 + 4x + 7 - a with minimum value 2,
    prove that g(x) = f(x - 2015) also has minimum value 2 -/
theorem min_value_shifted_quadratic (a : ℝ) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = x^2 + 4*x + 7 - a) ∧ 
   (∃ m, m = 2 ∧ ∀ x, f x ≥ m)) →
  (∃ (g : ℝ → ℝ), (∀ x, g x = (x - 2015)^2 + 4*(x - 2015) + 7 - a) ∧ 
   (∃ m, m = 2 ∧ ∀ x, g x ≥ m)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_shifted_quadratic_l1832_183243


namespace NUMINAMATH_CALUDE_complement_union_A_B_l1832_183209

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 2}
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

theorem complement_union_A_B : 
  (U \ (A ∪ B)) = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l1832_183209


namespace NUMINAMATH_CALUDE_average_first_12_even_numbers_l1832_183260

theorem average_first_12_even_numbers : 
  let first_12_even : List ℕ := List.range 12 |>.map (fun n => 2 * (n + 1))
  (first_12_even.sum / first_12_even.length : ℚ) = 13 := by
  sorry

end NUMINAMATH_CALUDE_average_first_12_even_numbers_l1832_183260


namespace NUMINAMATH_CALUDE_moon_speed_in_km_per_hour_l1832_183291

/-- The speed of the moon in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 1.04

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Converts speed from kilometers per second to kilometers per hour -/
def km_per_sec_to_km_per_hour (speed_km_per_sec : ℝ) : ℝ :=
  speed_km_per_sec * (seconds_per_hour : ℝ)

theorem moon_speed_in_km_per_hour :
  km_per_sec_to_km_per_hour moon_speed_km_per_sec = 3744 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_in_km_per_hour_l1832_183291


namespace NUMINAMATH_CALUDE_quadratic_through_origin_l1832_183202

/-- Given a quadratic function f(x) = ax^2 + x + a(a-2) that passes through the origin,
    prove that a = 2 -/
theorem quadratic_through_origin (a : ℝ) (h1 : a ≠ 0) :
  (∀ x, a*x^2 + x + a*(a-2) = 0 → x = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_through_origin_l1832_183202
