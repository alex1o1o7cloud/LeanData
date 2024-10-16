import Mathlib

namespace NUMINAMATH_CALUDE_sequence_formula_l4013_401313

theorem sequence_formula (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
  (h : ∀ n : ℕ+, S n = 2 * a n - 1) : 
  ∀ n : ℕ+, a n = 2^(n.val - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l4013_401313


namespace NUMINAMATH_CALUDE_only_odd_divisor_of_3_pow_n_plus_1_l4013_401320

theorem only_odd_divisor_of_3_pow_n_plus_1 :
  ∀ n : ℕ, Odd n → (n ∣ 3^n + 1) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_odd_divisor_of_3_pow_n_plus_1_l4013_401320


namespace NUMINAMATH_CALUDE_total_profit_is_27_l4013_401361

/-- Given the following conditions:
  1. Natasha has 3 times as much money as Carla
  2. Carla has twice as much money as Cosima
  3. Natasha has $60
  4. Sergio has 1.5 times as much money as Cosima
  5. Natasha buys 4 items at $15 each
  6. Carla buys 6 items at $10 each
  7. Cosima buys 5 items at $8 each
  8. Sergio buys 3 items at $12 each
  9. Profit margins: Natasha 10%, Carla 15%, Cosima 12%, Sergio 20%

  Prove that the total profit after selling all goods is $27. -/
theorem total_profit_is_27 (natasha_money carla_money cosima_money sergio_money : ℚ)
  (natasha_items carla_items cosima_items sergio_items : ℕ)
  (natasha_price carla_price cosima_price sergio_price : ℚ)
  (natasha_margin carla_margin cosima_margin sergio_margin : ℚ) :
  natasha_money = 60 ∧
  natasha_money = 3 * carla_money ∧
  carla_money = 2 * cosima_money ∧
  sergio_money = 1.5 * cosima_money ∧
  natasha_items = 4 ∧
  carla_items = 6 ∧
  cosima_items = 5 ∧
  sergio_items = 3 ∧
  natasha_price = 15 ∧
  carla_price = 10 ∧
  cosima_price = 8 ∧
  sergio_price = 12 ∧
  natasha_margin = 0.1 ∧
  carla_margin = 0.15 ∧
  cosima_margin = 0.12 ∧
  sergio_margin = 0.2 →
  natasha_items * natasha_price * natasha_margin +
  carla_items * carla_price * carla_margin +
  cosima_items * cosima_price * cosima_margin +
  sergio_items * sergio_price * sergio_margin = 27 := by
  sorry


end NUMINAMATH_CALUDE_total_profit_is_27_l4013_401361


namespace NUMINAMATH_CALUDE_rationalize_denominator_1_l4013_401383

theorem rationalize_denominator_1 (a b c : ℝ) :
  a / (b - Real.sqrt c + a) = (a * (b + a + Real.sqrt c)) / ((b + a)^2 - c) :=
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_1_l4013_401383


namespace NUMINAMATH_CALUDE_tree_increase_l4013_401393

theorem tree_increase (initial_trees : ℕ) (increase_percentage : ℚ) : 
  initial_trees = 120 →
  increase_percentage = 5.5 / 100 →
  initial_trees + ⌊(increase_percentage * initial_trees : ℚ)⌋ = 126 := by
sorry

end NUMINAMATH_CALUDE_tree_increase_l4013_401393


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l4013_401300

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_solution
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_sum : a 1 + a 2 + a 3 = 21)
  (h_product : a 1 * a 2 * a 3 = 216)
  (h_product_35 : a 3 * a 5 = 18)
  (h_product_48 : a 4 * a 8 = 72) :
  (∃ n : ℕ, a n = 3 * 2^(n-1) ∨ a n = 12 * (1/2)^(n-1)) ∧
  (∃ q : ℝ, q = Real.sqrt 2 ∨ q = -Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l4013_401300


namespace NUMINAMATH_CALUDE_quadratic_function_range_l4013_401308

def f (x : ℝ) : ℝ := -x^2 + x + 2

theorem quadratic_function_range (a : ℝ) :
  (∀ x₁ x₂, a ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ a + 3 → f x₁ < f x₂) ∧
  (∃ x, a ≤ x ∧ x ≤ a + 3 ∧ f x = -4) →
  -5 ≤ a ∧ a ≤ -5/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l4013_401308


namespace NUMINAMATH_CALUDE_student_average_correct_l4013_401368

theorem student_average_correct (p q r : ℝ) (wp wq wr : ℝ) :
  wp + wq + wr = 6 →
  wp = 2 →
  wq = 1 →
  wr = 3 →
  p < q →
  q < r →
  (wp * p + wq * q + wr * r) / (wp + wq + wr) =
  ((wp + wq) * ((wp * p + wq * q) / (wp + wq)) + wr * r) / (wp + wq + wr) :=
by sorry

end NUMINAMATH_CALUDE_student_average_correct_l4013_401368


namespace NUMINAMATH_CALUDE_optimal_fence_dimensions_l4013_401316

/-- Represents the dimensions of a rectangular plot -/
structure PlotDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the total fence length for a given plot -/
def totalFenceLength (d : PlotDimensions) : ℝ :=
  3 * d.length + 2 * d.width

/-- Theorem stating the optimal dimensions for minimal fence length -/
theorem optimal_fence_dimensions :
  ∃ (d : PlotDimensions),
    d.length * d.width = 294 ∧
    d.length = 14 ∧
    d.width = 21 ∧
    ∀ (d' : PlotDimensions),
      d'.length * d'.width = 294 →
      totalFenceLength d ≤ totalFenceLength d' := by
  sorry

end NUMINAMATH_CALUDE_optimal_fence_dimensions_l4013_401316


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l4013_401387

/-- The volume of a cube with a space diagonal of 6√3 units is 216 cubic units. -/
theorem cube_volume_from_space_diagonal :
  ∀ (s : ℝ), s > 0 → s * Real.sqrt 3 = 6 * Real.sqrt 3 → s^3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l4013_401387


namespace NUMINAMATH_CALUDE_complex_number_problem_l4013_401317

theorem complex_number_problem (z : ℂ) 
  (h1 : ∃ (r1 : ℝ), z / (1 + z^2) = r1)
  (h2 : ∃ (r2 : ℝ), z^2 / (1 + z) = r2) :
  z = -1/2 + (Complex.I * Real.sqrt 3)/2 ∨ 
  z = -1/2 - (Complex.I * Real.sqrt 3)/2 :=
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l4013_401317


namespace NUMINAMATH_CALUDE_negation_existence_equivalence_l4013_401303

theorem negation_existence_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x = x - 1) ↔
  (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) := by
sorry

end NUMINAMATH_CALUDE_negation_existence_equivalence_l4013_401303


namespace NUMINAMATH_CALUDE_max_profit_at_half_l4013_401365

/-- The profit function for a souvenir sale after process improvement -/
def profit_function (x : ℝ) : ℝ := 500 * (1 + 4*x - x^2 - 4*x^3)

/-- The theorem stating the maximum profit and the corresponding price increase -/
theorem max_profit_at_half :
  ∃ (max_profit : ℝ),
    (∀ x : ℝ, 0 < x → x < 1 → profit_function x ≤ max_profit) ∧
    profit_function (1/2) = max_profit ∧
    max_profit = 11125 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_half_l4013_401365


namespace NUMINAMATH_CALUDE_first_customer_headphones_l4013_401344

/-- The cost of one MP3 player -/
def mp3_cost : ℕ := 120

/-- The cost of one set of headphones -/
def headphone_cost : ℕ := 30

/-- The number of MP3 players bought by the first customer -/
def first_customer_mp3 : ℕ := 5

/-- The total cost of the first customer's purchase -/
def first_customer_total : ℕ := 840

/-- The number of MP3 players bought by the second customer -/
def second_customer_mp3 : ℕ := 3

/-- The number of headphone sets bought by the second customer -/
def second_customer_headphones : ℕ := 4

/-- The total cost of the second customer's purchase -/
def second_customer_total : ℕ := 480

theorem first_customer_headphones :
  ∃ h : ℕ, first_customer_mp3 * mp3_cost + h * headphone_cost = first_customer_total ∧
          h = 8 :=
by sorry

end NUMINAMATH_CALUDE_first_customer_headphones_l4013_401344


namespace NUMINAMATH_CALUDE_remainder_theorem_l4013_401360

theorem remainder_theorem (n : ℤ) : n % 7 = 3 → (5 * n - 12) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4013_401360


namespace NUMINAMATH_CALUDE_uniform_price_calculation_l4013_401391

/-- Represents the price of the uniform in Rupees -/
def uniform_price : ℝ := 200

/-- Represents the full year service pay in Rupees -/
def full_year_pay : ℝ := 800

/-- Represents the actual service duration in months -/
def actual_service : ℝ := 9

/-- Represents the full year service duration in months -/
def full_year : ℝ := 12

/-- Represents the actual payment received in Rupees -/
def actual_payment : ℝ := 400

theorem uniform_price_calculation :
  uniform_price = full_year_pay * (actual_service / full_year) - actual_payment :=
by sorry

end NUMINAMATH_CALUDE_uniform_price_calculation_l4013_401391


namespace NUMINAMATH_CALUDE_investment_difference_l4013_401312

def initial_investment : ℕ := 2000

def alice_multiplier : ℕ := 2
def bob_multiplier : ℕ := 5

def alice_final (initial : ℕ) : ℕ := initial * alice_multiplier
def bob_final (initial : ℕ) : ℕ := initial * bob_multiplier

theorem investment_difference : 
  bob_final initial_investment - alice_final initial_investment = 6000 := by
  sorry

end NUMINAMATH_CALUDE_investment_difference_l4013_401312


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l4013_401377

/-- For a right triangle with legs a and b, hypotenuse c, and an inscribed circle of radius r -/
def RightTriangleWithInscribedCircle (a b c r : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ a^2 + b^2 = c^2

/-- The radius of the inscribed circle in a right triangle is equal to (a + b - c) / 2 -/
theorem inscribed_circle_radius 
  (a b c r : ℝ) 
  (h : RightTriangleWithInscribedCircle a b c r) : 
  r = (a + b - c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l4013_401377


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l4013_401390

/-- Represents a batch of parts with different classes -/
structure Batch :=
  (total : ℕ)
  (first_class : ℕ)
  (second_class : ℕ)
  (third_class : ℕ)

/-- Represents a sampling process -/
structure Sampling :=
  (batch : Batch)
  (sample_size : ℕ)

/-- The probability of selecting an individual part in systematic sampling -/
def selection_probability (s : Sampling) : ℚ :=
  s.sample_size / s.batch.total

/-- Theorem stating the probability of selecting each part in the given scenario -/
theorem systematic_sampling_probability (b : Batch) (s : Sampling) :
  b.total = 120 →
  b.first_class = 24 →
  b.second_class = 36 →
  b.third_class = 60 →
  s.batch = b →
  s.sample_size = 20 →
  selection_probability s = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_probability_l4013_401390


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4013_401351

theorem polynomial_simplification (x : ℝ) :
  (15 * x^10 + 10 * x^9 + 5 * x^8) + (3 * x^12 + 2 * x^10 + x^9 + 3 * x^7 + 4 * x^4 + 6 * x^2 + 9) =
  3 * x^12 + 17 * x^10 + 11 * x^9 + 5 * x^8 + 3 * x^7 + 4 * x^4 + 6 * x^2 + 9 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l4013_401351


namespace NUMINAMATH_CALUDE_integer_sum_l4013_401314

theorem integer_sum (x y : ℕ+) : x - y = 14 → x * y = 48 → x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_l4013_401314


namespace NUMINAMATH_CALUDE_coin_count_theorem_l4013_401392

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | Quarter
  | Dollar

/-- The value of each coin type in cents --/
def coinValue (c : CoinType) : ℕ :=
  match c with
  | .Penny => 1
  | .Nickel => 5
  | .Dime => 10
  | .Quarter => 25
  | .Dollar => 100

/-- The set of all coin types --/
def allCoinTypes : List CoinType := [.Penny, .Nickel, .Dime, .Quarter, .Dollar]

theorem coin_count_theorem (n : ℕ) 
    (h1 : n > 0)
    (h2 : (List.sum (List.map (fun c => coinValue c * n) allCoinTypes)) = 351) :
    List.length allCoinTypes * n = 15 := by
  sorry

end NUMINAMATH_CALUDE_coin_count_theorem_l4013_401392


namespace NUMINAMATH_CALUDE_gcd_91_49_l4013_401321

theorem gcd_91_49 : Nat.gcd 91 49 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_91_49_l4013_401321


namespace NUMINAMATH_CALUDE_unique_solution_cubic_system_l4013_401325

theorem unique_solution_cubic_system :
  ∃! (x y z : ℝ),
    x = y^3 + y - 8 ∧
    y = z^3 + z - 8 ∧
    z = x^3 + x - 8 ∧
    x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_system_l4013_401325


namespace NUMINAMATH_CALUDE_divisibility_problem_l4013_401350

theorem divisibility_problem (n m k : ℕ) (h1 : n = 859722) (h2 : m = 456) (h3 : k = 54) :
  (n + k) % m = 0 :=
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l4013_401350


namespace NUMINAMATH_CALUDE_positive_integer_pairs_l4013_401378

theorem positive_integer_pairs (a b : ℕ+) :
  (∃ k : ℤ, (a.val^3 * b.val - 1) = k * (a.val + 1)) ∧
  (∃ m : ℤ, (b.val^3 * a.val + 1) = m * (b.val - 1)) →
  ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_pairs_l4013_401378


namespace NUMINAMATH_CALUDE_committee_selection_l4013_401389

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l4013_401389


namespace NUMINAMATH_CALUDE_complement_A_union_B_when_a_3_A_intersect_B_equals_B_iff_l4013_401398

-- Define set A
def A : Set ℝ := {x | x^2 ≤ 3*x + 10}

-- Define set B as a function of a
def B (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}

-- Statement 1
theorem complement_A_union_B_when_a_3 :
  (Set.univ \ A) ∪ (B 3) = {x | x ≤ -2 ∨ (4 ≤ x ∧ x ≤ 7)} := by sorry

-- Statement 2
theorem A_intersect_B_equals_B_iff (a : ℝ) :
  A ∩ (B a) = B a ↔ a < 2 := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_when_a_3_A_intersect_B_equals_B_iff_l4013_401398


namespace NUMINAMATH_CALUDE_root_shift_theorem_l4013_401354

/-- Given a, b, and c are roots of x³ - 5x + 7 = 0, prove that a+3, b+3, and c+3 are roots of x³ - 9x² + 22x - 5 = 0 -/
theorem root_shift_theorem (a b c : ℝ) : 
  (a^3 - 5*a + 7 = 0) → 
  (b^3 - 5*b + 7 = 0) → 
  (c^3 - 5*c + 7 = 0) → 
  ((a+3)^3 - 9*(a+3)^2 + 22*(a+3) - 5 = 0) ∧
  ((b+3)^3 - 9*(b+3)^2 + 22*(b+3) - 5 = 0) ∧
  ((c+3)^3 - 9*(c+3)^2 + 22*(c+3) - 5 = 0) := by
  sorry


end NUMINAMATH_CALUDE_root_shift_theorem_l4013_401354


namespace NUMINAMATH_CALUDE_sqrt_gt_3x_iff_l4013_401379

theorem sqrt_gt_3x_iff (x : ℝ) (h : x > 0) : 
  Real.sqrt x > 3 * x ↔ 0 < x ∧ x < 1/9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_gt_3x_iff_l4013_401379


namespace NUMINAMATH_CALUDE_break_even_proof_l4013_401381

def initial_investment : ℝ := 10410
def manufacturing_cost : ℝ := 2.65
def selling_price : ℝ := 20

def break_even_point : ℕ := 600

theorem break_even_proof :
  ⌈initial_investment / (selling_price - manufacturing_cost)⌉ = break_even_point :=
by sorry

end NUMINAMATH_CALUDE_break_even_proof_l4013_401381


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4013_401323

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  d : ℤ
  h1 : a 4 = -15
  h2 : d = 3
  h3 : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def SumOfTerms (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n * (seq.a 1 + seq.a n)) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 3 * n - 27) ∧
  ∃ min : ℤ, min = -108 ∧ ∀ n : ℕ, SumOfTerms seq n ≥ min :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4013_401323


namespace NUMINAMATH_CALUDE_student_arrangement_l4013_401339

theorem student_arrangement (n : ℕ) (k : ℕ) (m : ℕ) : 
  n = 7 → k = 6 → m = 3 → 
  (n.choose k) * (k.choose m) * ((k - m).choose m) = 140 :=
sorry

end NUMINAMATH_CALUDE_student_arrangement_l4013_401339


namespace NUMINAMATH_CALUDE_triangle_perimeter_l4013_401332

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the properties of the triangle
def isOnCircumference (P Q : ℝ × ℝ) : Prop := sorry

def angleEqual (P Q R : ℝ × ℝ) : Prop := sorry

def distance (A B : ℝ × ℝ) : ℝ := sorry

def perimeter (t : Triangle) : ℝ :=
  distance t.P t.Q + distance t.Q t.R + distance t.R t.P

-- State the theorem
theorem triangle_perimeter (t : Triangle) :
  isOnCircumference t.P t.Q →
  angleEqual t.P t.Q t.R →
  distance t.Q t.R = 8 →
  distance t.P t.R = 10 →
  perimeter t = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l4013_401332


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l4013_401334

theorem matrix_equation_proof :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -7; 9, 3]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![44/7, -57/7; -39/14, 51/14]
  N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l4013_401334


namespace NUMINAMATH_CALUDE_carbon_neutrality_time_l4013_401306

/-- Carbon neutrality time calculation -/
theorem carbon_neutrality_time (a : ℝ) (b : ℝ) :
  a > 0 →
  a * b^7 = (4/5) * a →
  ∃ t : ℝ, t ≥ 42 ∧ a * b^t = (1/4) * a :=
by
  sorry

end NUMINAMATH_CALUDE_carbon_neutrality_time_l4013_401306


namespace NUMINAMATH_CALUDE_f_properties_l4013_401359

def f (x : ℝ) : ℝ := x^2 * (x - 3) * (x + 1)

theorem f_properties :
  (∀ x, f x = f (-x)) ∧ 
  f (-1) = 0 ∧ 
  f 3 = 0 := by
sorry

end NUMINAMATH_CALUDE_f_properties_l4013_401359


namespace NUMINAMATH_CALUDE_u_converges_immediately_l4013_401324

def u : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 3 * u n - 3 * (u n)^2

theorem u_converges_immediately :
  ∀ k : ℕ, |u k - 1/2| ≤ 1/2^20 := by
  sorry

end NUMINAMATH_CALUDE_u_converges_immediately_l4013_401324


namespace NUMINAMATH_CALUDE_problem_solution_l4013_401335

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xyz : x * y * z = 1)
  (h_x_z : x + 1 / z = 4)
  (h_y_x : y + 1 / x = 20) :
  z + 1 / y = 26 / 79 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4013_401335


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l4013_401352

theorem no_integer_solutions_for_equation :
  ¬ ∃ (x y : ℤ), 15 * x^2 - 7 * y^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l4013_401352


namespace NUMINAMATH_CALUDE_gaokao_probability_l4013_401384

/-- The probability of choosing both Physics and History in the Gaokao exam -/
theorem gaokao_probability (p_physics_not_history p_history_not_physics : ℝ) 
  (h1 : p_physics_not_history = 0.5)
  (h2 : p_history_not_physics = 0.3) :
  1 - p_physics_not_history - p_history_not_physics = 0.2 := by sorry

end NUMINAMATH_CALUDE_gaokao_probability_l4013_401384


namespace NUMINAMATH_CALUDE_sin_cos_identity_l4013_401382

theorem sin_cos_identity (x : ℝ) : 
  Real.sin x * Real.cos x + Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3 / 2 = 
  Real.sin (2 * x + π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l4013_401382


namespace NUMINAMATH_CALUDE_integer_pair_gcd_equation_l4013_401374

theorem integer_pair_gcd_equation :
  ∀ x y : ℕ+, 
    (x.val * y.val * Nat.gcd x.val y.val = x.val + y.val + (Nat.gcd x.val y.val)^2) ↔ 
    ((x, y) = (2, 2) ∨ (x, y) = (2, 3) ∨ (x, y) = (3, 2)) := by
  sorry

end NUMINAMATH_CALUDE_integer_pair_gcd_equation_l4013_401374


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4013_401348

def i : ℂ := Complex.I

theorem complex_fraction_simplification :
  (2 - 3 * i) / (1 + 4 * i) = -10/17 - 11/17 * i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4013_401348


namespace NUMINAMATH_CALUDE_system_solution_l4013_401356

theorem system_solution : ∃ (a b c : ℝ), 
  (a^2 * b^2 - a^2 - a*b + 1 = 0) ∧ 
  (a^2 * c - a*b - a - c = 0) ∧ 
  (a*b*c = -1) ∧ 
  (a = -1) ∧ (b = -1) ∧ (c = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4013_401356


namespace NUMINAMATH_CALUDE_division_sum_theorem_l4013_401341

theorem division_sum_theorem (n d : ℕ) (h1 : n = 55) (h2 : d = 11) :
  n + d + (n / d) = 71 := by
  sorry

end NUMINAMATH_CALUDE_division_sum_theorem_l4013_401341


namespace NUMINAMATH_CALUDE_lace_cost_is_36_l4013_401302

/-- The cost of lace per meter in dollars -/
def lace_cost_per_meter : ℚ := 6

/-- The length of each cuff in centimeters -/
def cuff_length : ℚ := 50

/-- The number of cuffs -/
def num_cuffs : ℕ := 2

/-- The length of the hem in centimeters -/
def hem_length : ℚ := 300

/-- The length of the waist in centimeters -/
def waist_length : ℚ := hem_length / 3

/-- The number of ruffles for the neck -/
def num_ruffles : ℕ := 5

/-- The length of lace used for each ruffle in centimeters -/
def ruffle_length : ℚ := 20

/-- The total cost of lace for trimming the dress -/
def total_lace_cost : ℚ :=
  let total_length_cm : ℚ := 
    cuff_length * num_cuffs + hem_length + waist_length + ruffle_length * num_ruffles
  let total_length_m : ℚ := total_length_cm / 100
  total_length_m * lace_cost_per_meter

theorem lace_cost_is_36 : total_lace_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_lace_cost_is_36_l4013_401302


namespace NUMINAMATH_CALUDE_parallel_line_slope_l4013_401301

/-- A point P with coordinates depending on a parameter p -/
def P (p : ℝ) : ℝ × ℝ := (2*p, -4*p + 1)

/-- The line y = kx + 2 -/
def line (k : ℝ) (x : ℝ) : ℝ := k*x + 2

/-- The theorem stating that k must be -2 for the line to be parallel to the locus of P -/
theorem parallel_line_slope (k : ℝ) : 
  (∀ p : ℝ, P p ∉ {xy : ℝ × ℝ | xy.2 = line k xy.1}) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l4013_401301


namespace NUMINAMATH_CALUDE_abs_minus_one_lt_two_iff_product_lt_zero_l4013_401338

theorem abs_minus_one_lt_two_iff_product_lt_zero (x : ℝ) :
  |x - 1| < 2 ↔ (x + 1) * (x - 3) < 0 := by sorry

end NUMINAMATH_CALUDE_abs_minus_one_lt_two_iff_product_lt_zero_l4013_401338


namespace NUMINAMATH_CALUDE_no_fermat_in_sequence_l4013_401363

/-- The general term of the second-order arithmetic sequence -/
def a (n k : ℕ) : ℕ := (k - 2) * n * (n - 1) / 2 + n

/-- Fermat number of order m -/
def fermat (m : ℕ) : ℕ := 2^(2^m) + 1

/-- Statement: There are no Fermat numbers in the sequence for k > 2 -/
theorem no_fermat_in_sequence (k : ℕ) (h : k > 2) :
  ∀ (n m : ℕ), a n k ≠ fermat m :=
sorry

end NUMINAMATH_CALUDE_no_fermat_in_sequence_l4013_401363


namespace NUMINAMATH_CALUDE_quadratic_vertex_l4013_401331

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The vertex of a parabola -/
structure Vertex where
  x : ℝ
  y : ℝ

/-- Theorem: If f(2) = 3 and x = 2 is the axis of symmetry for f(x) = ax^2 + bx + c,
    then the vertex of the parabola is at (2, 3) -/
theorem quadratic_vertex (a b c : ℝ) :
  let f := QuadraticFunction a b c
  f 2 = 3 → -- f(2) = 3
  (∀ x, f (4 - x) = f x) → -- x = 2 is the axis of symmetry
  Vertex.mk 2 3 = Vertex.mk (2 : ℝ) (f 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l4013_401331


namespace NUMINAMATH_CALUDE_max_volume_triangular_pyramid_l4013_401370

/-- A triangular pyramid P-ABC with given side lengths -/
structure TriangularPyramid where
  PA : ℝ
  PB : ℝ
  AB : ℝ
  BC : ℝ
  CA : ℝ

/-- The volume of a triangular pyramid -/
def volume (t : TriangularPyramid) : ℝ := sorry

/-- The maximum volume of a triangular pyramid with specific side lengths -/
theorem max_volume_triangular_pyramid :
  ∀ t : TriangularPyramid,
  t.PA = 3 ∧ t.PB = 3 ∧ t.AB = 2 ∧ t.BC = 2 ∧ t.CA = 2 →
  volume t ≤ 2 * Real.sqrt 6 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_volume_triangular_pyramid_l4013_401370


namespace NUMINAMATH_CALUDE_area_of_triangle_PQR_l4013_401369

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def circleP : Circle := { center := (0, 1), radius := 1 }
def circleQ : Circle := { center := (3, 2), radius := 2 }
def circleR : Circle := { center := (4, 3), radius := 3 }

-- Define the line l (implicitly defined by the tangent points)

-- Define the theorem
theorem area_of_triangle_PQR :
  let P := circleP.center
  let Q := circleQ.center
  let R := circleR.center
  let area := abs ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2)
  area = Real.sqrt 6 - Real.sqrt 2 :=
sorry


end NUMINAMATH_CALUDE_area_of_triangle_PQR_l4013_401369


namespace NUMINAMATH_CALUDE_power_multiplication_l4013_401319

theorem power_multiplication (n : ℕ) : n * (n^(n - 1)) = n^n :=
by
  sorry

#check power_multiplication 3000

end NUMINAMATH_CALUDE_power_multiplication_l4013_401319


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l4013_401318

/-- Two-dimensional vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Parallelism of two vectors -/
def parallel (v w : Vec2D) : Prop :=
  ∃ (μ : ℝ), μ ≠ 0 ∧ v.x = μ * w.x ∧ v.y = μ * w.y

theorem parallel_vectors_x_value :
  ∀ (x : ℝ),
  let a : Vec2D := ⟨x, 1⟩
  let b : Vec2D := ⟨3, 6⟩
  parallel b a → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l4013_401318


namespace NUMINAMATH_CALUDE_divide_600_in_ratio_1_2_l4013_401333

def divide_in_ratio (total : ℚ) (ratio1 ratio2 : ℕ) : ℚ :=
  total * ratio1 / (ratio1 + ratio2)

theorem divide_600_in_ratio_1_2 :
  divide_in_ratio 600 1 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_divide_600_in_ratio_1_2_l4013_401333


namespace NUMINAMATH_CALUDE_negative_represents_spending_l4013_401385

/-- Represents a monetary transaction -/
inductive Transaction
| receive (amount : ℤ)
| spend (amount : ℤ)

/-- Converts a transaction to an integer representation -/
def transactionToInt : Transaction → ℤ
| Transaction.receive amount => amount
| Transaction.spend amount => -amount

theorem negative_represents_spending (t : Transaction) : 
  (∃ (a : ℤ), a > 0 ∧ transactionToInt (Transaction.receive a) = a) →
  (∀ (b : ℤ), b > 0 → transactionToInt (Transaction.spend b) = -b) :=
by sorry

end NUMINAMATH_CALUDE_negative_represents_spending_l4013_401385


namespace NUMINAMATH_CALUDE_roots_sum_greater_than_twice_zero_l4013_401309

noncomputable section

open Real

def f (x : ℝ) := x * log x
def g (x : ℝ) := x / exp x
def F (x : ℝ) := f x - g x
def m (x : ℝ) := min (f x) (g x)

theorem roots_sum_greater_than_twice_zero 
  (x₀ : ℝ) 
  (h₁ : 1 < x₀ ∧ x₀ < 2) 
  (h₂ : F x₀ = 0) 
  (h₃ : ∀ x, 1 < x ∧ x < 2 ∧ F x = 0 → x = x₀)
  (x₁ x₂ : ℝ) 
  (h₄ : 1 < x₁ ∧ x₁ < x₂)
  (h₅ : ∃ n, m x₁ = n ∧ m x₂ = n)
  : x₁ + x₂ > 2 * x₀ := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_greater_than_twice_zero_l4013_401309


namespace NUMINAMATH_CALUDE_cube_plane_difference_l4013_401357

/-- Represents a cube with points placed on each face -/
structure MarkedCube where
  -- Add necessary fields

/-- Represents a plane intersecting the cube -/
structure IntersectingPlane where
  -- Add necessary fields

/-- Represents a segment on the surface of the cube -/
structure SurfaceSegment where
  -- Add necessary fields

/-- The maximum number of planes required to create all possible segments -/
def max_planes (cube : MarkedCube) : ℕ := sorry

/-- The minimum number of planes required to create all possible segments -/
def min_planes (cube : MarkedCube) : ℕ := sorry

/-- All possible segments on the surface of the cube -/
def all_segments (cube : MarkedCube) : Set SurfaceSegment := sorry

/-- The set of segments created by a given set of planes -/
def segments_from_planes (cube : MarkedCube) (planes : Set IntersectingPlane) : Set SurfaceSegment := sorry

theorem cube_plane_difference (cube : MarkedCube) :
  max_planes cube - min_planes cube = 24 :=
sorry

end NUMINAMATH_CALUDE_cube_plane_difference_l4013_401357


namespace NUMINAMATH_CALUDE_coat_duration_proof_l4013_401375

/-- The duration (in years) for which the more expensive coat lasts -/
def duration_expensive_coat : ℕ := sorry

/-- The cost of the more expensive coat -/
def cost_expensive_coat : ℕ := 300

/-- The cost of the cheaper coat -/
def cost_cheaper_coat : ℕ := 120

/-- The duration (in years) for which the cheaper coat lasts -/
def duration_cheaper_coat : ℕ := 5

/-- The time period (in years) over which savings are calculated -/
def savings_period : ℕ := 30

/-- The amount saved over the savings period by choosing the more expensive coat -/
def savings_amount : ℕ := 120

theorem coat_duration_proof :
  duration_expensive_coat = 15 ∧
  cost_expensive_coat * savings_period / duration_expensive_coat +
    savings_amount =
  cost_cheaper_coat * savings_period / duration_cheaper_coat :=
by sorry

end NUMINAMATH_CALUDE_coat_duration_proof_l4013_401375


namespace NUMINAMATH_CALUDE_baker_usual_pastries_l4013_401372

/-- The number of pastries the baker usually sells -/
def usual_pastries : ℕ := sorry

/-- The number of loaves the baker usually sells -/
def usual_loaves : ℕ := 10

/-- The number of pastries sold today -/
def today_pastries : ℕ := 14

/-- The number of loaves sold today -/
def today_loaves : ℕ := 25

/-- The price of a pastry in dollars -/
def pastry_price : ℚ := 2

/-- The price of a loaf in dollars -/
def loaf_price : ℚ := 4

/-- The difference between today's sales and average sales in dollars -/
def sales_difference : ℚ := 48

theorem baker_usual_pastries : 
  usual_pastries = 20 :=
by sorry

end NUMINAMATH_CALUDE_baker_usual_pastries_l4013_401372


namespace NUMINAMATH_CALUDE_cuboid_volume_l4013_401310

/-- Given a cuboid with three side faces sharing a common vertex having areas 3, 5, and 15,
    prove that its volume is 15. -/
theorem cuboid_volume (a b c : ℝ) 
  (h1 : a * b = 3)
  (h2 : b * c = 5)
  (h3 : a * c = 15) : 
  a * b * c = 15 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_l4013_401310


namespace NUMINAMATH_CALUDE_functional_equation_implies_odd_l4013_401353

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y) = y * f x

/-- Theorem stating that f(-x) = -f(x) for functions satisfying the functional equation -/
theorem functional_equation_implies_odd (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_implies_odd_l4013_401353


namespace NUMINAMATH_CALUDE_no_prime_sum_10003_l4013_401388

/-- A function that returns the number of ways to write n as the sum of two primes -/
def countPrimeSumWays (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p)) (Finset.range n)).card

/-- Theorem stating that 10003 cannot be written as the sum of two primes -/
theorem no_prime_sum_10003 : countPrimeSumWays 10003 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_10003_l4013_401388


namespace NUMINAMATH_CALUDE_potato_sale_revenue_l4013_401371

/-- Calculates the revenue from selling potatoes given the total weight, damaged weight, bag size, and price per bag. -/
def potato_revenue (total_weight damaged_weight bag_size price_per_bag : ℕ) : ℕ :=
  let sellable_weight := total_weight - damaged_weight
  let num_bags := sellable_weight / bag_size
  num_bags * price_per_bag

/-- Theorem stating that the revenue from selling potatoes under given conditions is $9144. -/
theorem potato_sale_revenue :
  potato_revenue 6500 150 50 72 = 9144 := by
  sorry

end NUMINAMATH_CALUDE_potato_sale_revenue_l4013_401371


namespace NUMINAMATH_CALUDE_parallelogram_area_from_rectangle_l4013_401373

theorem parallelogram_area_from_rectangle (rectangle_width rectangle_length parallelogram_height : ℝ) 
  (hw : rectangle_width = 8)
  (hl : rectangle_length = 10)
  (hh : parallelogram_height = 9) :
  rectangle_width * parallelogram_height = 72 := by
  sorry

#check parallelogram_area_from_rectangle

end NUMINAMATH_CALUDE_parallelogram_area_from_rectangle_l4013_401373


namespace NUMINAMATH_CALUDE_miss_two_consecutive_probability_l4013_401364

/-- The probability of hitting a target in one shot. -/
def hit_probability : ℝ := 0.8

/-- The probability of missing a target in one shot. -/
def miss_probability : ℝ := 1 - hit_probability

/-- The probability of missing a target in two consecutive shots. -/
def miss_two_consecutive : ℝ := miss_probability * miss_probability

theorem miss_two_consecutive_probability :
  miss_two_consecutive = 0.04 := by sorry

end NUMINAMATH_CALUDE_miss_two_consecutive_probability_l4013_401364


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_count_l4013_401395

theorem systematic_sampling_interval_count 
  (total_employees : ℕ) 
  (sample_size : ℕ) 
  (interval_start : ℕ) 
  (interval_end : ℕ) 
  (h1 : total_employees = 840)
  (h2 : sample_size = 42)
  (h3 : interval_start = 481)
  (h4 : interval_end = 720) :
  (interval_end - interval_start + 1) / (total_employees / sample_size) = 12 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_count_l4013_401395


namespace NUMINAMATH_CALUDE_exists_valid_grid_l4013_401345

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Checks if all elements in a list are equal -/
def allEqual (l : List ℕ) : Prop :=
  l.all (· = l.head!)

/-- Checks if a grid satisfies the required properties -/
def validGrid (g : Grid) : Prop :=
  -- 0 is in the central position
  g 1 1 = 0 ∧
  -- Digits 1-8 are used exactly once each in the remaining positions
  (∀ i : Fin 9, i ≠ 0 → ∃! (r c : Fin 3), g r c = i.val) ∧
  -- The sum of digits in each row and each column is the same
  allEqual [
    g 0 0 + g 0 1 + g 0 2,
    g 1 0 + g 1 1 + g 1 2,
    g 2 0 + g 2 1 + g 2 2,
    g 0 0 + g 1 0 + g 2 0,
    g 0 1 + g 1 1 + g 2 1,
    g 0 2 + g 1 2 + g 2 2
  ]

/-- There exists a grid satisfying the required properties -/
theorem exists_valid_grid : ∃ g : Grid, validGrid g := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_grid_l4013_401345


namespace NUMINAMATH_CALUDE_min_value_sin_cos_squared_l4013_401305

theorem min_value_sin_cos_squared (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  ∃ (m : ℝ), m = -1/9 ∧ ∀ z, Real.sin z + Real.sin (y + z - x) = 1/3 →
    m ≤ Real.sin z + (Real.cos z)^2 :=
sorry

end NUMINAMATH_CALUDE_min_value_sin_cos_squared_l4013_401305


namespace NUMINAMATH_CALUDE_secret_number_count_l4013_401376

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def tens_digit (n : ℕ) : ℕ := n / 10

def units_digit (n : ℕ) : ℕ := n % 10

def secret_number (n : ℕ) : Prop :=
  is_two_digit n ∧
  Odd (tens_digit n) ∧
  Even (units_digit n) ∧
  n > 75 ∧
  n % 3 = 0

theorem secret_number_count : 
  ∃! (s : Finset ℕ), s.card = 3 ∧ ∀ n, n ∈ s ↔ secret_number n :=
sorry

end NUMINAMATH_CALUDE_secret_number_count_l4013_401376


namespace NUMINAMATH_CALUDE_textbook_selling_price_l4013_401366

/-- The selling price of a textbook, given its cost price and profit -/
theorem textbook_selling_price (cost_price profit : ℝ) (h1 : cost_price = 44) (h2 : profit = 11) :
  cost_price + profit = 55 := by
  sorry

#check textbook_selling_price

end NUMINAMATH_CALUDE_textbook_selling_price_l4013_401366


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a5_l4013_401380

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 1 + a 5 = 8 ∧
  a 4 = 7

/-- Theorem: In the given arithmetic sequence, a_5 = 10 -/
theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  a 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a5_l4013_401380


namespace NUMINAMATH_CALUDE_eleven_step_paths_through_F_l4013_401328

/-- A point on the 6x6 grid -/
structure Point where
  x : Nat
  y : Nat
  h_x : x ≤ 5
  h_y : y ≤ 5

/-- The number of paths between two points on the grid -/
def num_paths (start finish : Point) : Nat :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

theorem eleven_step_paths_through_F : 
  let E : Point := ⟨0, 5, by norm_num, by norm_num⟩
  let F : Point := ⟨3, 3, by norm_num, by norm_num⟩
  let G : Point := ⟨5, 0, by norm_num, by norm_num⟩
  (num_paths E F) * (num_paths F G) = 100 := by
  sorry

end NUMINAMATH_CALUDE_eleven_step_paths_through_F_l4013_401328


namespace NUMINAMATH_CALUDE_equation_solutions_l4013_401330

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 11 ∧ x₂ = 3 - Real.sqrt 11 ∧
    x₁^2 - 6*x₁ - 2 = 0 ∧ x₂^2 - 6*x₂ - 2 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = -1/2 ∧ y₂ = -2 ∧
    (2*y₁ + 1)^2 = -6*y₁ - 3 ∧ (2*y₂ + 1)^2 = -6*y₂ - 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4013_401330


namespace NUMINAMATH_CALUDE_jessica_bank_balance_l4013_401337

theorem jessica_bank_balance (B : ℝ) 
  (h1 : B * (2/5) = 200)  -- Condition: 2/5 of initial balance equals $200
  (h2 : B > 200)          -- Implicit condition: initial balance is greater than withdrawal
  : B - 200 + (B - 200) / 2 = 450 := by
  sorry

end NUMINAMATH_CALUDE_jessica_bank_balance_l4013_401337


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_over_2_l4013_401304

noncomputable def f (x : ℝ) : ℝ := Real.sin x / (Real.sin x + Real.cos x)

theorem derivative_f_at_pi_over_2 :
  deriv f (π / 2) = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_over_2_l4013_401304


namespace NUMINAMATH_CALUDE_chip_drawing_probability_l4013_401346

theorem chip_drawing_probability : 
  let total_chips : ℕ := 14
  let tan_chips : ℕ := 5
  let pink_chips : ℕ := 3
  let violet_chips : ℕ := 6
  let favorable_outcomes : ℕ := (Nat.factorial pink_chips) * (Nat.factorial tan_chips) * (Nat.factorial violet_chips)
  let total_outcomes : ℕ := Nat.factorial total_chips
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 168168 := by
sorry

end NUMINAMATH_CALUDE_chip_drawing_probability_l4013_401346


namespace NUMINAMATH_CALUDE_subtraction_problem_l4013_401349

theorem subtraction_problem : 2000000000000 - 1111111111111 = 888888888889 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l4013_401349


namespace NUMINAMATH_CALUDE_max_value_of_quadratic_l4013_401355

theorem max_value_of_quadratic (x : ℝ) (h1 : 0 < x) (h2 : x < 3/2) :
  x * (3 - 2*x) ≤ 9/8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_quadratic_l4013_401355


namespace NUMINAMATH_CALUDE_tetrahedron_vertices_tetrahedron_has_four_vertices_l4013_401322

/-- A tetrahedron is a three-dimensional geometric shape with four triangular faces. -/
structure Tetrahedron where
  -- No specific fields needed for this problem

/-- The number of vertices in a tetrahedron is 4. -/
theorem tetrahedron_vertices (t : Tetrahedron) : Nat := 4

/-- Proof that a tetrahedron has 4 vertices. -/
theorem tetrahedron_has_four_vertices (t : Tetrahedron) : tetrahedron_vertices t = 4 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_vertices_tetrahedron_has_four_vertices_l4013_401322


namespace NUMINAMATH_CALUDE_general_admission_tickets_l4013_401367

theorem general_admission_tickets (student_price general_price total_tickets total_revenue : ℕ) 
  (h1 : student_price = 4)
  (h2 : general_price = 6)
  (h3 : total_tickets = 525)
  (h4 : total_revenue = 2876) :
  ∃ (student_tickets general_tickets : ℕ),
    student_tickets + general_tickets = total_tickets ∧
    student_price * student_tickets + general_price * general_tickets = total_revenue ∧
    general_tickets = 388 := by
  sorry

end NUMINAMATH_CALUDE_general_admission_tickets_l4013_401367


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l4013_401394

-- Define a line by two points
def Line (x₁ y₁ x₂ y₂ : ℝ) := {(x, y) : ℝ × ℝ | ∃ t, x = x₁ + t * (x₂ - x₁) ∧ y = y₁ + t * (y₂ - y₁)}

-- Define the y-axis
def YAxis := {(x, y) : ℝ × ℝ | x = 0}

-- Theorem statement
theorem line_intersection_y_axis :
  ∃! p : ℝ × ℝ, p ∈ Line 2 9 4 15 ∧ p ∈ YAxis ∧ p = (0, 3) := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l4013_401394


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_45_l4013_401362

/-- Given a natural number n, returns the sum of n consecutive positive integers starting from a -/
def consecutive_sum (n : ℕ) (a : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- Predicate that checks if there exists a starting integer a such that n consecutive integers starting from a sum to 45 -/
def exists_consecutive_sum (n : ℕ) : Prop :=
  ∃ a : ℕ, a > 0 ∧ consecutive_sum n a = 45

theorem max_consecutive_integers_sum_45 :
  (∀ k : ℕ, k > 9 → ¬ exists_consecutive_sum k) ∧
  exists_consecutive_sum 9 :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_45_l4013_401362


namespace NUMINAMATH_CALUDE_complex_number_value_l4013_401399

theorem complex_number_value (z : ℂ) (h1 : z^2 = 6*z - 27 + 12*I) (h2 : ∃ (n : ℕ), Complex.abs z = n) :
  z = 3 + (Real.sqrt 6 + Real.sqrt 6 * I) ∨ z = 3 - (Real.sqrt 6 + Real.sqrt 6 * I) :=
sorry

end NUMINAMATH_CALUDE_complex_number_value_l4013_401399


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l4013_401315

-- Define the given quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 5 * x - 2

-- Define the solution set of the given inequality
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | 1/2 < x ∧ x < 2}

-- Define the second quadratic function
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 5 * x + a^2 - 1

theorem quadratic_inequality_problem (a : ℝ) :
  (∀ x, f a x > 0 ↔ x ∈ solution_set a) →
  (a = -2 ∧
   ∀ x, g a x > 0 ↔ -3 < x ∧ x < 1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l4013_401315


namespace NUMINAMATH_CALUDE_fraction_of_married_men_l4013_401342

theorem fraction_of_married_men (total : ℕ) (h1 : total > 0) : 
  let women := (60 : ℚ) / 100 * total
  let men := total - women
  let married := (60 : ℚ) / 100 * total
  let single_men := (3 : ℚ) / 4 * men
  (men - single_men) / men = (1 : ℚ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_married_men_l4013_401342


namespace NUMINAMATH_CALUDE_present_age_of_b_l4013_401347

theorem present_age_of_b (a b : ℕ) : 
  (a + 30 = 2 * (b - 30)) →  -- In 30 years, A will be twice as old as B was 30 years ago
  (a = b + 5) →              -- A is now 5 years older than B
  b = 95 :=                  -- The present age of B is 95
by sorry

end NUMINAMATH_CALUDE_present_age_of_b_l4013_401347


namespace NUMINAMATH_CALUDE_part_one_part_two_l4013_401343

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3 * x - 7 ≥ 8 - 2 * x}

-- Part 1
theorem part_one :
  let a := 2
  (A a ∩ B = {x | 3 ≤ x ∧ x < 4}) ∧
  ((Set.univ \ A a) ∪ (Set.univ \ B) = {x | x < 3 ∨ x ≥ 4}) := by sorry

-- Part 2
theorem part_two (a : ℝ) :
  A a ∩ B = A a → a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4013_401343


namespace NUMINAMATH_CALUDE_jacobs_gift_budget_l4013_401326

theorem jacobs_gift_budget (total_budget : ℕ) (num_friends : ℕ) (friend_gift_cost : ℕ) (num_parents : ℕ) :
  total_budget = 100 →
  num_friends = 8 →
  friend_gift_cost = 9 →
  num_parents = 2 →
  (total_budget - num_friends * friend_gift_cost) / num_parents = 14 :=
by sorry

end NUMINAMATH_CALUDE_jacobs_gift_budget_l4013_401326


namespace NUMINAMATH_CALUDE_total_books_l4013_401336

theorem total_books (tim_books sam_books alex_books : ℕ) 
  (h1 : tim_books = 44)
  (h2 : sam_books = 52)
  (h3 : alex_books = 65) :
  tim_books + sam_books + alex_books = 161 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l4013_401336


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_divisibility_l4013_401396

theorem infinitely_many_pairs_divisibility :
  ∀ n : ℕ, ∃ a b : ℤ, a > n ∧ (a * (a + 1)) ∣ (b^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_divisibility_l4013_401396


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l4013_401327

theorem complex_modulus_sqrt_two (x y : ℝ) :
  (Complex.I + 1) * x = Complex.I * y + 1 →
  Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l4013_401327


namespace NUMINAMATH_CALUDE_largest_divisor_of_product_l4013_401397

theorem largest_divisor_of_product (n : ℕ) (h : Even n) (h' : n > 0) :
  (∃ (k : ℕ), (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) = 105 * k) ∧
  (∀ (m : ℕ), m > 105 → ¬(∀ (n : ℕ), Even n → n > 0 →
    ∃ (k : ℕ), (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) = m * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_product_l4013_401397


namespace NUMINAMATH_CALUDE_polygon_perimeter_sum_l4013_401307

theorem polygon_perimeter_sum (n : ℕ) (x y c : ℝ) : 
  n ≥ 3 →
  x = (2 * n : ℝ) * (Real.tan (π / (n : ℝ))) * (c / (2 * π)) →
  y = (2 * n : ℝ) * (Real.sin (π / (n : ℝ))) * (c / (2 * π)) →
  (∀ θ : ℝ, 0 ≤ θ ∧ θ < π / 2 → Real.tan θ ≥ θ) →
  x + y ≥ 2 * c :=
by sorry

end NUMINAMATH_CALUDE_polygon_perimeter_sum_l4013_401307


namespace NUMINAMATH_CALUDE_apples_per_pie_l4013_401340

theorem apples_per_pie (initial_apples : ℕ) (handed_out : ℕ) (num_pies : ℕ) 
  (h1 : initial_apples = 86)
  (h2 : handed_out = 30)
  (h3 : num_pies = 7) :
  (initial_apples - handed_out) / num_pies = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_apples_per_pie_l4013_401340


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l4013_401311

theorem complex_magnitude_problem (z : ℂ) (h : (Complex.I / (1 + Complex.I)) * z = 1) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l4013_401311


namespace NUMINAMATH_CALUDE_price_A_base_correct_minimum_amount_spent_l4013_401386

-- Define the price of type A seedlings at the base
def price_A_base : ℝ := 20

-- Define the price of type B seedlings at the base
def price_B_base : ℝ := 30

-- Define the total number of bundles to purchase
def total_bundles : ℕ := 100

-- Define the discount rate
def discount_rate : ℝ := 0.9

-- Theorem for part 1
theorem price_A_base_correct :
  price_A_base * (300 / price_A_base) = 
  (5/4 * price_A_base) * (300 / (5/4 * price_A_base) + 3) := by sorry

-- Theorem for part 2
theorem minimum_amount_spent :
  let m := min (total_bundles / 2) total_bundles
  ∃ (n : ℕ), n ≤ total_bundles - n ∧
    discount_rate * (price_A_base * m + price_B_base * (total_bundles - m)) = 2250 := by sorry

end NUMINAMATH_CALUDE_price_A_base_correct_minimum_amount_spent_l4013_401386


namespace NUMINAMATH_CALUDE_cakes_served_today_l4013_401329

theorem cakes_served_today (lunch_cakes dinner_cakes : ℕ) 
  (h1 : lunch_cakes = 6) 
  (h2 : dinner_cakes = 9) : 
  lunch_cakes + dinner_cakes = 15 := by
  sorry

end NUMINAMATH_CALUDE_cakes_served_today_l4013_401329


namespace NUMINAMATH_CALUDE_cube_sum_squares_l4013_401358

theorem cube_sum_squares (a b t : ℝ) (h : a + b = t^2) :
  ∃ x y z : ℝ, 2 * (a^3 + b^3) = x^2 + y^2 + z^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_squares_l4013_401358
