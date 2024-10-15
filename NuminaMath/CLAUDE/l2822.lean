import Mathlib

namespace NUMINAMATH_CALUDE_multiplication_simplification_l2822_282221

theorem multiplication_simplification : 12 * (1 / 26) * 52 * 4 = 96 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_simplification_l2822_282221


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l2822_282274

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(38.2 : ℝ)⌉ = 35 := by sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l2822_282274


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2822_282242

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  n > 20 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 ∧ 
  n % 8 = 5 ∧ 
  (∀ m : ℕ, m > 20 ∧ m % 6 = 4 ∧ m % 7 = 3 ∧ m % 8 = 5 → m ≥ n) ∧
  n = 220 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2822_282242


namespace NUMINAMATH_CALUDE_polynomial_negative_roots_l2822_282253

theorem polynomial_negative_roots (q : ℝ) (hq : q > 2) :
  ∃ (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  x₁^4 + q*x₁^3 + 2*x₁^2 + q*x₁ + 1 = 0 ∧
  x₂^4 + q*x₂^3 + 2*x₂^2 + q*x₂ + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_negative_roots_l2822_282253


namespace NUMINAMATH_CALUDE_combined_mean_score_l2822_282229

/-- Given two sections of algebra students with different mean scores and a ratio of students between sections, calculate the combined mean score. -/
theorem combined_mean_score (mean1 mean2 : ℚ) (ratio : ℚ) : 
  mean1 = 92 →
  mean2 = 78 →
  ratio = 5/7 →
  (mean1 * ratio + mean2) / (ratio + 1) = 1006/12 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_score_l2822_282229


namespace NUMINAMATH_CALUDE_proportion_proof_l2822_282236

theorem proportion_proof (a b c d : ℝ) : 
  a = 3 →
  a / b = 0.6 →
  a * d = 12 →
  a / b = c / d →
  (a, b, c, d) = (3, 5, 2.4, 4) := by
  sorry

end NUMINAMATH_CALUDE_proportion_proof_l2822_282236


namespace NUMINAMATH_CALUDE_triple_angle_sine_sin_18_degrees_l2822_282277

open Real

-- Define the sum of sines formula
axiom sum_of_sines (α β : ℝ) : sin (α + β) = sin α * cos β + cos α * sin β

-- Define the double angle formula for sine
axiom double_angle_sine (α : ℝ) : sin (2 * α) = 2 * sin α * cos α

-- Define the relation between sine and cosine
axiom sine_cosine_relation (α : ℝ) : sin α = cos (π / 2 - α)

-- Theorem 1: Triple angle formula for sine
theorem triple_angle_sine (α : ℝ) : sin (3 * α) = 3 * sin α - 4 * (sin α)^3 := by sorry

-- Theorem 2: Value of sin 18°
theorem sin_18_degrees : sin (18 * π / 180) = (Real.sqrt 5 - 1) / 4 := by sorry

end NUMINAMATH_CALUDE_triple_angle_sine_sin_18_degrees_l2822_282277


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l2822_282223

theorem simplify_complex_expression :
  (3 * (Real.sqrt 3 + Real.sqrt 7)) / (4 * Real.sqrt (3 + Real.sqrt 5)) =
  Real.sqrt (224 - 22 * Real.sqrt 105) / 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l2822_282223


namespace NUMINAMATH_CALUDE_power_mod_23_l2822_282235

theorem power_mod_23 : 17^2001 % 23 = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_23_l2822_282235


namespace NUMINAMATH_CALUDE_min_four_digit_satisfying_condition_l2822_282227

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def satisfies_condition (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  let ab := 10 * a + b
  let cd := 10 * c + d
  (n + ab * cd) % 1111 = 0

theorem min_four_digit_satisfying_condition :
  ∃ (n : ℕ), is_four_digit n ∧ satisfies_condition n ∧
  ∀ (m : ℕ), is_four_digit m → satisfies_condition m → n ≤ m :=
by
  use 1729
  sorry

end NUMINAMATH_CALUDE_min_four_digit_satisfying_condition_l2822_282227


namespace NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_60_max_area_225_achievable_l2822_282258

/-- The maximum area of a rectangle with perimeter 60 is 225 -/
theorem max_area_rectangle_with_perimeter_60 :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * x + 2 * y = 60 →
  x * y ≤ 225 :=
by
  sorry

/-- The maximum area of 225 is achievable -/
theorem max_area_225_achievable :
  ∃ x y : ℝ,
  x > 0 ∧ y > 0 ∧
  2 * x + 2 * y = 60 ∧
  x * y = 225 :=
by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_60_max_area_225_achievable_l2822_282258


namespace NUMINAMATH_CALUDE_three_questions_identify_l2822_282237

/-- Represents a geometric figure -/
inductive GeometricFigure
  | Circle
  | Ellipse
  | Triangle
  | Square
  | Rectangle
  | Parallelogram
  | Trapezoid

/-- Represents a yes/no question about a geometric figure -/
def Question := GeometricFigure → Bool

/-- The set of all geometric figures -/
def FigureSet : Set GeometricFigure := {
  GeometricFigure.Circle,
  GeometricFigure.Ellipse,
  GeometricFigure.Triangle,
  GeometricFigure.Square,
  GeometricFigure.Rectangle,
  GeometricFigure.Parallelogram,
  GeometricFigure.Trapezoid
}

/-- A sequence of three questions -/
structure ThreeQuestions where
  q1 : Question
  q2 : Question
  q3 : Question

/-- Checks if a sequence of three questions can uniquely identify a figure -/
def canIdentify (qs : ThreeQuestions) (f : GeometricFigure) : Prop :=
  ∀ g : GeometricFigure, g ∈ FigureSet →
    (qs.q1 f = qs.q1 g ∧ qs.q2 f = qs.q2 g ∧ qs.q3 f = qs.q3 g) → f = g

/-- The main theorem: there exists a sequence of three questions that can identify any figure -/
theorem three_questions_identify :
  ∃ qs : ThreeQuestions, ∀ f : GeometricFigure, f ∈ FigureSet → canIdentify qs f := by
  sorry


end NUMINAMATH_CALUDE_three_questions_identify_l2822_282237


namespace NUMINAMATH_CALUDE_max_z_value_l2822_282249

theorem max_z_value (x y z : ℝ) (sum_eq : x + y + z = 3) (prod_eq : x*y + y*z + z*x = 2) :
  z ≤ 5/3 ∧ ∃ (x' y' z' : ℝ), x' + y' + z' = 3 ∧ x'*y' + y'*z' + z'*x' = 2 ∧ z' = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_max_z_value_l2822_282249


namespace NUMINAMATH_CALUDE_remainder_problem_l2822_282283

theorem remainder_problem (N : ℤ) : ∃ k : ℤ, N = 35 * k + 25 → ∃ m : ℤ, N = 15 * m + 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2822_282283


namespace NUMINAMATH_CALUDE_second_divisor_problem_l2822_282287

theorem second_divisor_problem : ∃ (D N k m : ℕ+), N = 39 * k + 17 ∧ N = D * m + 4 ∧ D = 13 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_problem_l2822_282287


namespace NUMINAMATH_CALUDE_max_pieces_in_box_l2822_282259

theorem max_pieces_in_box : 
  ∃ n : ℕ, n < 50 ∧ 
  (∃ k : ℕ, n = 4 * k + 2) ∧ 
  (∃ m : ℕ, n = 6 * m) ∧
  ∀ x : ℕ, x < 50 → 
    ((∃ k : ℕ, x = 4 * k + 2) ∧ (∃ m : ℕ, x = 6 * m)) → 
    x ≤ n :=
by sorry

end NUMINAMATH_CALUDE_max_pieces_in_box_l2822_282259


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2822_282214

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  (3 * X ^ 3 - 4 * X ^ 2 + 17 * X + 34 : Polynomial ℤ) = 
  (X - 7) * q + 986 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2822_282214


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2822_282232

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and an asymptote y = 4/3 * x is 5/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b / a = 4 / 3) :
  let c := Real.sqrt (a^2 + b^2)
  c / a = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2822_282232


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l2822_282290

/-- A geometric sequence with common ratio 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a n > 0) →
  a 3 * a 11 = 16 →
  a 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l2822_282290


namespace NUMINAMATH_CALUDE_wendy_phone_pictures_l2822_282289

/-- The number of pictures Wendy uploaded from her phone -/
def phone_pictures (total_albums : ℕ) (pictures_per_album : ℕ) (camera_pictures : ℕ) : ℕ :=
  total_albums * pictures_per_album - camera_pictures

/-- Proof that Wendy uploaded 22 pictures from her phone -/
theorem wendy_phone_pictures :
  phone_pictures 4 6 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_wendy_phone_pictures_l2822_282289


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2822_282250

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- a is the sequence
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- arithmetic sequence condition
  (h_a1 : a 1 = 2)  -- given a_1 = 2
  (h_a3 : a 3 = 8)  -- given a_3 = 8
  : a 2 - a 1 = 3 :=  -- prove that the common difference is 3
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2822_282250


namespace NUMINAMATH_CALUDE_intersection_M_N_l2822_282247

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}
def N : Set ℝ := {x | ∃ y, y = Real.log (2*x - x^2)}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2822_282247


namespace NUMINAMATH_CALUDE_f_neg_x_properties_l2822_282245

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem f_neg_x_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f (-x) > f (-y)) := by
  sorry

#check f_neg_x_properties

end NUMINAMATH_CALUDE_f_neg_x_properties_l2822_282245


namespace NUMINAMATH_CALUDE_domain_of_f_l2822_282264

noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((Real.log x - 2) * (x - Real.log x - 1))

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {1} ∪ Set.Ici (Real.exp 2) := by sorry

end NUMINAMATH_CALUDE_domain_of_f_l2822_282264


namespace NUMINAMATH_CALUDE_vector_properties_l2822_282262

-- Define plane vectors a and b
variable (a b : ℝ × ℝ)

-- Define the conditions
def condition1 : Prop := norm a = 1
def condition2 : Prop := norm b = 1
def condition3 : Prop := norm (2 • a + b) = Real.sqrt 6

-- Define the theorem
theorem vector_properties (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) :
  (a • b = 1/4) ∧ (norm (a + b) = Real.sqrt 10 / 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l2822_282262


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l2822_282273

theorem parallelogram_base_length 
  (area : ℝ) 
  (altitude : ℝ) 
  (base : ℝ) 
  (h1 : area = 450) 
  (h2 : altitude = 2 * base) 
  (h3 : area = base * altitude) : 
  base = 15 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l2822_282273


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2822_282219

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ a ∈ Set.Ioi 2 ∪ Set.Iio (-2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2822_282219


namespace NUMINAMATH_CALUDE_prob_more_heads_than_tails_fair_coin_l2822_282251

/-- A fair coin is a coin with equal probability of heads and tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def prob_k_heads (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1-p)^(n-k)

/-- The probability of getting more heads than tails in 3 flips of a fair coin -/
def prob_more_heads_than_tails (p : ℝ) : ℝ :=
  prob_k_heads 3 2 p + prob_k_heads 3 3 p

theorem prob_more_heads_than_tails_fair_coin :
  ∀ p : ℝ, fair_coin p → prob_more_heads_than_tails p = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_prob_more_heads_than_tails_fair_coin_l2822_282251


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l2822_282206

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  2 ∣ ((n - 2) + (n - 1) + n + (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l2822_282206


namespace NUMINAMATH_CALUDE_shirt_difference_l2822_282218

theorem shirt_difference (alex_shirts joe_shirts ben_shirts : ℕ) : 
  alex_shirts = 4 → 
  ben_shirts = 15 → 
  ben_shirts = joe_shirts + 8 → 
  joe_shirts - alex_shirts = 3 := by
sorry

end NUMINAMATH_CALUDE_shirt_difference_l2822_282218


namespace NUMINAMATH_CALUDE_total_population_l2822_282200

theorem total_population (b g t : ℕ) : 
  b = 4 * g ∧ g = 8 * t → b + g + t = 41 * t :=
by sorry

end NUMINAMATH_CALUDE_total_population_l2822_282200


namespace NUMINAMATH_CALUDE_product_of_numbers_l2822_282209

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 16) : x * y = 836 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2822_282209


namespace NUMINAMATH_CALUDE_square_root_problem_l2822_282212

theorem square_root_problem (a : ℝ) (n : ℝ) (h1 : n > 0) 
  (h2 : Real.sqrt n = a + 3) (h3 : Real.sqrt n = 2*a - 15) : n = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l2822_282212


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2822_282261

/-- Given real numbers x, y, and z forming a geometric sequence with -1 and -3,
    prove that their product equals -3√3 -/
theorem geometric_sequence_product (x y z : ℝ) 
  (h1 : ∃ (r : ℝ), x = -1 * r ∧ y = x * r ∧ z = y * r ∧ -3 = z * r) :
  x * y * z = -3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2822_282261


namespace NUMINAMATH_CALUDE_min_sum_of_coefficients_l2822_282280

theorem min_sum_of_coefficients (b c : ℕ+) 
  (h1 : ∃ (x y : ℝ), x ≠ y ∧ 2 * x^2 + b * x + c = 0 ∧ 2 * y^2 + b * y + c = 0)
  (h2 : ∃ (x y : ℝ), x - y = 30 ∧ 2 * x^2 + b * x + c = 0 ∧ 2 * y^2 + b * y + c = 0) :
  (∀ (b' c' : ℕ+), 
    (∃ (x y : ℝ), x ≠ y ∧ 2 * x^2 + b' * x + c' = 0 ∧ 2 * y^2 + b' * y + c' = 0) →
    (∃ (x y : ℝ), x - y = 30 ∧ 2 * x^2 + b' * x + c' = 0 ∧ 2 * y^2 + b' * y + c' = 0) →
    b'.val + c'.val ≥ b.val + c.val) →
  b.val + c.val = 126 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_coefficients_l2822_282280


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l2822_282207

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 5| + |x - 3|

-- State the theorem
theorem min_value_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 1/b = Real.sqrt 3) : 
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) ∧ m = 2) ∧ 
  (1/a^2 + 2/b^2 ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l2822_282207


namespace NUMINAMATH_CALUDE_unique_solution_fourth_root_equation_l2822_282222

/-- The equation √⁴(58 - 3x) + √⁴(26 + 3x) = 5 has a unique solution -/
theorem unique_solution_fourth_root_equation :
  ∃! x : ℝ, (58 - 3*x)^(1/4) + (26 + 3*x)^(1/4) = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_fourth_root_equation_l2822_282222


namespace NUMINAMATH_CALUDE_tangent_slope_implies_b_over_a_equals_two_l2822_282269

/-- A quadratic function f(x) = ax² + b with a tangent line of slope 2 at (1,3) -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b

/-- The derivative of f -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2 * a * x

theorem tangent_slope_implies_b_over_a_equals_two (a b : ℝ) :
  f a b 1 = 3 ∧ f_derivative a 1 = 2 → b / a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_b_over_a_equals_two_l2822_282269


namespace NUMINAMATH_CALUDE_circle_triangle_count_l2822_282265

def points : ℕ := 9

def total_combinations : ℕ := Nat.choose points 3

def consecutive_triangles : ℕ := points

def valid_triangles : ℕ := total_combinations - consecutive_triangles

theorem circle_triangle_count :
  valid_triangles = 75 := by sorry

end NUMINAMATH_CALUDE_circle_triangle_count_l2822_282265


namespace NUMINAMATH_CALUDE_total_time_at_least_5400_seconds_l2822_282208

/-- Represents an observer's record of lap times -/
structure Observer where
  lap_times : List Int
  time_difference : Int

/-- The proposition to be proved -/
theorem total_time_at_least_5400_seconds
  (observer1 observer2 : Observer)
  (h1 : observer1.time_difference = 1)
  (h2 : observer2.time_difference = -1)
  (h3 : observer1.lap_times.length = observer2.lap_times.length)
  (h4 : observer1.lap_times.length ≥ 29) :
  (List.sum observer1.lap_times + List.sum observer2.lap_times) ≥ 5400 :=
sorry

end NUMINAMATH_CALUDE_total_time_at_least_5400_seconds_l2822_282208


namespace NUMINAMATH_CALUDE_negation_and_absolute_value_l2822_282205

theorem negation_and_absolute_value : 
  (-(-2) = 2) ∧ (-|(-2)| = -2) := by
  sorry

end NUMINAMATH_CALUDE_negation_and_absolute_value_l2822_282205


namespace NUMINAMATH_CALUDE_roberto_chicken_investment_l2822_282282

def initial_cost : ℝ := 25 + 30 + 22 + 35
def weekly_feed_cost : ℝ := 1.5 + 1.3 + 1.1 + 0.9
def weekly_egg_production : ℕ := 4 + 3 + 5 + 2
def previous_egg_cost : ℝ := 2

def break_even_weeks : ℕ := 40

theorem roberto_chicken_investment (w : ℕ) :
  w = break_even_weeks ↔ 
  initial_cost + w * weekly_feed_cost = w * previous_egg_cost :=
sorry

end NUMINAMATH_CALUDE_roberto_chicken_investment_l2822_282282


namespace NUMINAMATH_CALUDE_simplify_fraction_l2822_282241

theorem simplify_fraction : (36 : ℚ) / 4536 = 1 / 126 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2822_282241


namespace NUMINAMATH_CALUDE_mrs_lovely_class_l2822_282297

/-- The number of students in Mrs. Lovely's class -/
def total_students : ℕ := 23

/-- The number of girls in the class -/
def girls : ℕ := 10

/-- The number of boys in the class -/
def boys : ℕ := girls + 3

/-- The total number of chocolates brought -/
def total_chocolates : ℕ := 500

/-- The number of chocolates left after distribution -/
def leftover_chocolates : ℕ := 10

theorem mrs_lovely_class :
  (girls * girls + boys * boys = total_chocolates - leftover_chocolates) ∧
  (girls + boys = total_students) := by
  sorry

end NUMINAMATH_CALUDE_mrs_lovely_class_l2822_282297


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2822_282284

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 2 * Complex.I ^ 2018 → z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2822_282284


namespace NUMINAMATH_CALUDE_oil_mixture_price_l2822_282267

/-- Given two oils mixed together, prove the price of the first oil -/
theorem oil_mixture_price (volume1 volume2 : ℝ) (price2 mix_price : ℝ) (h1 : volume1 = 10)
    (h2 : volume2 = 5) (h3 : price2 = 66) (h4 : mix_price = 58.67) :
    ∃ (price1 : ℝ), price1 = 55.005 ∧ 
    volume1 * price1 + volume2 * price2 = (volume1 + volume2) * mix_price := by
  sorry

end NUMINAMATH_CALUDE_oil_mixture_price_l2822_282267


namespace NUMINAMATH_CALUDE_equal_sum_sequence_2017_sum_l2822_282203

/-- An equal sum sequence with a given first term and common sum. -/
def EqualSumSequence (a : ℕ → ℕ) (first_term : ℕ) (common_sum : ℕ) : Prop :=
  a 1 = first_term ∧ ∀ n : ℕ, a n + a (n + 1) = common_sum

/-- The sum of the first n terms of a sequence. -/
def SequenceSum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum a

theorem equal_sum_sequence_2017_sum :
    ∀ a : ℕ → ℕ, EqualSumSequence a 2 5 → SequenceSum a 2017 = 5042 := by
  sorry

end NUMINAMATH_CALUDE_equal_sum_sequence_2017_sum_l2822_282203


namespace NUMINAMATH_CALUDE_function_extrema_l2822_282240

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x - 1/x + 1/x^2

theorem function_extrema (a : ℝ) :
  a ≠ 0 →
  (∃ (xmax xmin : ℝ), xmax > 0 ∧ xmin > 0 ∧
    (∀ x > 0, f a x ≤ f a xmax) ∧
    (∀ x > 0, f a x ≥ f a xmin)) ↔
  -1/8 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_function_extrema_l2822_282240


namespace NUMINAMATH_CALUDE_pool_and_deck_area_l2822_282276

/-- Calculates the total area of a rectangular pool and its surrounding deck. -/
def total_area (pool_length pool_width deck_width : ℝ) : ℝ :=
  (pool_length + 2 * deck_width) * (pool_width + 2 * deck_width)

/-- Proves that the total area of a specific rectangular pool and its deck is 728 square feet. -/
theorem pool_and_deck_area :
  total_area 20 22 3 = 728 := by
  sorry

end NUMINAMATH_CALUDE_pool_and_deck_area_l2822_282276


namespace NUMINAMATH_CALUDE_stratified_sample_sum_l2822_282270

/-- Represents the number of items in each food category -/
structure FoodCategories where
  grains : ℕ
  vegetableOils : ℕ
  animalDerived : ℕ
  fruitsAndVegetables : ℕ

/-- Calculates the total number of items across all categories -/
def totalItems (fc : FoodCategories) : ℕ :=
  fc.grains + fc.vegetableOils + fc.animalDerived + fc.fruitsAndVegetables

/-- Calculates the number of items to be sampled from a category in stratified sampling -/
def stratifiedSampleSize (categorySize sampleSize totalSize : ℕ) : ℕ :=
  (categorySize * sampleSize) / totalSize

/-- Theorem: In a stratified sample of 20 items from the given food categories,
    the sum of items from vegetable oils and fruits and vegetables is 6 -/
theorem stratified_sample_sum (fc : FoodCategories) 
    (h1 : fc.grains = 40)
    (h2 : fc.vegetableOils = 10)
    (h3 : fc.animalDerived = 30)
    (h4 : fc.fruitsAndVegetables = 20)
    (h5 : totalItems fc = 100)
    (sampleSize : ℕ)
    (h6 : sampleSize = 20) :
    stratifiedSampleSize fc.vegetableOils sampleSize (totalItems fc) +
    stratifiedSampleSize fc.fruitsAndVegetables sampleSize (totalItems fc) = 6 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_sum_l2822_282270


namespace NUMINAMATH_CALUDE_fraction_multiplication_invariance_l2822_282233

theorem fraction_multiplication_invariance (a b m : ℝ) (h : b ≠ 0) :
  ∀ x : ℝ, (a * (x - m)) / (b * (x - m)) = a / b ↔ x ≠ m :=
sorry

end NUMINAMATH_CALUDE_fraction_multiplication_invariance_l2822_282233


namespace NUMINAMATH_CALUDE_garden_perimeter_l2822_282243

/-- Represents a rectangular shape with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.length

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.length)

/-- Proves that the perimeter of the garden is 56 meters -/
theorem garden_perimeter (garden : Rectangle) (playground : Rectangle) : 
  garden.width = 16 → 
  playground.length = 16 → 
  garden.area = playground.area → 
  garden.perimeter = 56 → 
  garden.perimeter = 56 := by
  sorry

#check garden_perimeter

end NUMINAMATH_CALUDE_garden_perimeter_l2822_282243


namespace NUMINAMATH_CALUDE_some_number_value_l2822_282215

theorem some_number_value (some_number : ℝ) : 
  |9 - 8 * (3 - some_number)| - |5 - 11| = 75 → some_number = 12 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2822_282215


namespace NUMINAMATH_CALUDE_candle_burning_time_l2822_282272

-- Define the original length of the candle
def original_length : ℝ := 12

-- Define the rate of decrease in length per minute
def rate_of_decrease : ℝ := 0.08

-- Define the function for remaining length after x minutes
def remaining_length (x : ℝ) : ℝ := original_length - rate_of_decrease * x

-- Theorem statement
theorem candle_burning_time :
  ∃ (max_time : ℝ), max_time = 150 ∧ remaining_length max_time = 0 :=
sorry

end NUMINAMATH_CALUDE_candle_burning_time_l2822_282272


namespace NUMINAMATH_CALUDE_solve_system_l2822_282244

theorem solve_system (a b c d : ℤ) 
  (eq1 : a + b = c)
  (eq2 : b + c = 7)
  (eq3 : c + d = 10)
  (eq4 : c = 4) :
  a = 1 ∧ d = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2822_282244


namespace NUMINAMATH_CALUDE_custom_mult_example_l2822_282246

/-- Custom multiplication operation for rational numbers -/
def custom_mult (a b : ℚ) : ℚ := (a + b) / (1 - b)

/-- Theorem stating that (5 * 4) * 2 = 1 using the custom multiplication -/
theorem custom_mult_example : custom_mult (custom_mult 5 4) 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_example_l2822_282246


namespace NUMINAMATH_CALUDE_jimmy_stair_climbing_l2822_282260

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Jimmy's stair climbing problem -/
theorem jimmy_stair_climbing : arithmetic_sum 30 10 8 = 520 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_stair_climbing_l2822_282260


namespace NUMINAMATH_CALUDE_ratio_problem_l2822_282268

theorem ratio_problem (x y : ℝ) : 
  (0.60 / x = 6 / 2) ∧ (x / y = 8 / 12) → x = 0.20 ∧ y = 0.30 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2822_282268


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l2822_282201

theorem bicycle_cost_price (profit_A_to_B : ℝ) (loss_B_to_C : ℝ) (profit_C_to_D : ℝ) (loss_D_to_E : ℝ) (price_E : ℝ)
  (h1 : profit_A_to_B = 0.20)
  (h2 : loss_B_to_C = 0.15)
  (h3 : profit_C_to_D = 0.30)
  (h4 : loss_D_to_E = 0.10)
  (h5 : price_E = 285) :
  price_E / ((1 + profit_A_to_B) * (1 - loss_B_to_C) * (1 + profit_C_to_D) * (1 - loss_D_to_E)) =
  285 / (1.20 * 0.85 * 1.30 * 0.90) := by
sorry

#eval 285 / (1.20 * 0.85 * 1.30 * 0.90)

end NUMINAMATH_CALUDE_bicycle_cost_price_l2822_282201


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l2822_282291

theorem rectangle_dimension_change (L W : ℝ) (L' W' : ℝ) : 
  L > 0 ∧ W > 0 →  -- Ensure positive dimensions
  L' = 1.4 * L →   -- Length increased by 40%
  L * W = L' * W' → -- Area remains constant
  (W - W') / W = 0.2857 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l2822_282291


namespace NUMINAMATH_CALUDE_octahedron_sum_l2822_282286

/-- A regular octahedron with numbers from 1 to 12 on its vertices -/
structure NumberedOctahedron where
  /-- The assignment of numbers to vertices -/
  vertex_numbers : Fin 6 → Fin 12
  /-- The property that each number from 1 to 12 is used exactly once -/
  all_numbers_used : Function.Injective vertex_numbers

/-- The sum of numbers on a face of the octahedron -/
def face_sum (o : NumberedOctahedron) (face : Fin 8) : ℕ := sorry

/-- The property that all face sums are equal -/
def all_face_sums_equal (o : NumberedOctahedron) : Prop :=
  ∀ (face1 face2 : Fin 8), face_sum o face1 = face_sum o face2

theorem octahedron_sum (o : NumberedOctahedron) (h : all_face_sums_equal o) :
  ∃ (face : Fin 8), face_sum o face = 39 := by sorry

end NUMINAMATH_CALUDE_octahedron_sum_l2822_282286


namespace NUMINAMATH_CALUDE_cutting_theorem_l2822_282234

/-- Represents a string of pearls -/
structure PearlString where
  color : Bool  -- true for black, false for white
  length : Nat

/-- State of the cutting process -/
structure CuttingState where
  strings : List PearlString

/-- Cutting rules -/
def cut_strings (k : Nat) (state : CuttingState) : CuttingState := sorry

/-- Predicate to check if a state has a white pearl of length 1 -/
def has_single_white_pearl (state : CuttingState) : Prop := sorry

/-- Predicate to check if a state has a black pearl string of length > 1 -/
def has_multiple_black_pearls (state : CuttingState) : Prop := sorry

/-- The cutting process -/
def cutting_process (k : Nat) (b w : Nat) : CuttingState := sorry

/-- Main theorem -/
theorem cutting_theorem (k : Nat) (b w : Nat) 
  (h1 : k > 0) (h2 : b > w) (h3 : w > 1) :
  let final_state := cutting_process k b w
  has_single_white_pearl final_state → has_multiple_black_pearls final_state :=
by
  sorry


end NUMINAMATH_CALUDE_cutting_theorem_l2822_282234


namespace NUMINAMATH_CALUDE_intersection_equality_l2822_282279

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.1 + p.2^2 ≤ 0}
def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 ≥ p.1 + a}

-- State the theorem
theorem intersection_equality (a : ℝ) : M ∩ N a = M ↔ a ≤ 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l2822_282279


namespace NUMINAMATH_CALUDE_sum_of_g_10_and_neg_10_l2822_282266

/-- A function g defined as a polynomial of even degree -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^8 + b * x^6 - c * x^4 + 5

/-- Theorem stating that g(10) + g(-10) = 4 given g(10) = 2 -/
theorem sum_of_g_10_and_neg_10 (a b c : ℝ) (h : g a b c 10 = 2) :
  g a b c 10 + g a b c (-10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_g_10_and_neg_10_l2822_282266


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2822_282255

/-- An isosceles triangle with sides 6 and 3 has perimeter 15 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 6 → b = 3 → c = 6 →  -- Two sides are 6, one is 3
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  a = c →  -- Isosceles condition
  a + b + c = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2822_282255


namespace NUMINAMATH_CALUDE_simplify_fraction_l2822_282271

theorem simplify_fraction : (150 : ℚ) / 225 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2822_282271


namespace NUMINAMATH_CALUDE_most_likely_red_balls_l2822_282238

theorem most_likely_red_balls 
  (total_balls : ℕ) 
  (red_frequency : ℝ) 
  (h1 : total_balls = 20) 
  (h2 : 0 ≤ red_frequency ∧ red_frequency ≤ 1) 
  (h3 : red_frequency = 0.8) : 
  ⌊total_balls * red_frequency⌋ = 16 := by
sorry

end NUMINAMATH_CALUDE_most_likely_red_balls_l2822_282238


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l2822_282296

theorem polygon_sides_from_angle_sum (sum_of_angles : ℕ) (h : sum_of_angles = 1260) :
  ∃ n : ℕ, n ≥ 3 ∧ (n - 2) * 180 = sum_of_angles ∧ n = 9 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l2822_282296


namespace NUMINAMATH_CALUDE_range_of_m_l2822_282278

theorem range_of_m (m : ℝ) : 
  m ≠ 0 → 
  (∀ x : ℝ, x ≥ 4 → (m^2 * x - 1) / (m * x + 1) < 0) → 
  m < -1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2822_282278


namespace NUMINAMATH_CALUDE_age_difference_proof_l2822_282225

theorem age_difference_proof (patrick michael monica : ℕ) 
  (h1 : patrick * 5 = michael * 3)  -- Patrick and Michael's age ratio
  (h2 : michael * 4 = monica * 3)   -- Michael and Monica's age ratio
  (h3 : patrick + michael + monica = 88) -- Sum of ages
  : monica - patrick = 22 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2822_282225


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_plus_i_l2822_282294

theorem imaginary_part_of_i_times_one_plus_i : 
  Complex.im (Complex.I * (1 + Complex.I)) = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_plus_i_l2822_282294


namespace NUMINAMATH_CALUDE_richard_numbers_l2822_282224

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def all_digits_distinct (n : ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < 5 ∧ 0 ≤ j ∧ j < 5 → i ≠ j →
    (n / (10^i) % 10) ≠ (n / (10^j) % 10)

def all_digits_odd (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 5 → (n / (10^i) % 10) % 2 = 1

def all_digits_even (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 5 → (n / (10^i) % 10) % 2 = 0

def sum_starts_with_11_ends_with_1 (a b : ℕ) : Prop :=
  let sum := a + b
  110000 ≤ sum ∧ sum < 120000 ∧ sum % 10 = 1

def diff_starts_with_2_ends_with_11 (a b : ℕ) : Prop :=
  let diff := a - b
  20000 ≤ diff ∧ diff < 30000 ∧ diff % 100 = 11

theorem richard_numbers :
  ∃ (A B : ℕ),
    is_five_digit A ∧
    is_five_digit B ∧
    all_digits_distinct A ∧
    all_digits_distinct B ∧
    all_digits_odd A ∧
    all_digits_even B ∧
    sum_starts_with_11_ends_with_1 A B ∧
    diff_starts_with_2_ends_with_11 A B ∧
    A = 73591 ∧
    B = 46280 :=
by sorry

end NUMINAMATH_CALUDE_richard_numbers_l2822_282224


namespace NUMINAMATH_CALUDE_people_left_of_kolya_l2822_282263

/-- Given a class lineup with the following conditions:
  * There are 12 people to the right of Kolya
  * There are 20 people to the left of Sasha
  * There are 8 people to the right of Sasha
  Prove that there are 16 people to the left of Kolya -/
theorem people_left_of_kolya
  (right_of_kolya : ℕ)
  (left_of_sasha : ℕ)
  (right_of_sasha : ℕ)
  (h1 : right_of_kolya = 12)
  (h2 : left_of_sasha = 20)
  (h3 : right_of_sasha = 8) :
  left_of_sasha + right_of_sasha + 1 - right_of_kolya - 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_people_left_of_kolya_l2822_282263


namespace NUMINAMATH_CALUDE_first_hit_not_binomial_l2822_282228

/-- A random variable follows a binomial distribution -/
def is_binomial_distribution (X : ℕ → ℝ) : Prop :=
  ∃ (n : ℕ) (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧
    ∀ k, 0 ≤ k ∧ k ≤ n → X k = (n.choose k : ℝ) * p^k * (1-p)^(n-k)

/-- Computer virus infection scenario -/
def computer_infection (n : ℕ) : ℕ → ℝ := sorry

/-- First hit scenario -/
def first_hit : ℕ → ℝ := sorry

/-- Multiple shots scenario -/
def multiple_shots (n : ℕ) : ℕ → ℝ := sorry

/-- Car refueling scenario -/
def car_refueling : ℕ → ℝ := sorry

/-- Theorem stating that the first hit scenario is not a binomial distribution -/
theorem first_hit_not_binomial :
  is_binomial_distribution (computer_infection 10) ∧
  is_binomial_distribution (multiple_shots 10) ∧
  is_binomial_distribution car_refueling →
  ¬ is_binomial_distribution first_hit := by sorry

end NUMINAMATH_CALUDE_first_hit_not_binomial_l2822_282228


namespace NUMINAMATH_CALUDE_min_translation_for_symmetry_l2822_282281

open Real

theorem min_translation_for_symmetry :
  let f (x m : ℝ) := Real.sqrt 3 * cos (x + m) + sin (x + m)
  ∃ (min_m : ℝ), min_m > 0 ∧
    (∀ (m : ℝ), m > 0 → 
      (∀ (x : ℝ), f x m = f (-x) m) → m ≥ min_m) ∧
    (∀ (x : ℝ), f x min_m = f (-x) min_m) ∧
    min_m = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_min_translation_for_symmetry_l2822_282281


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l2822_282202

theorem inequality_system_integer_solutions (x : ℤ) :
  (2 * (x - 1) ≤ x + 3 ∧ (x + 1) / 3 < x - 1) ↔ (x = 3 ∨ x = 4 ∨ x = 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l2822_282202


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_one_l2822_282216

theorem product_of_fractions_equals_one :
  (5 / 3 : ℚ) * (6 / 10 : ℚ) * (15 / 9 : ℚ) * (12 / 20 : ℚ) *
  (25 / 15 : ℚ) * (18 / 30 : ℚ) * (35 / 21 : ℚ) * (24 / 40 : ℚ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_one_l2822_282216


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l2822_282217

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the number of red balls
def red_balls : ℕ := 4

-- Define the number of black balls
def black_balls : ℕ := 6

-- Theorem statement
theorem probability_of_red_ball :
  (red_balls : ℚ) / total_balls = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l2822_282217


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2822_282231

theorem cubic_roots_sum (p q r : ℝ) : 
  (p^3 + p^2 - 2*p - 1 = 0) →
  (q^3 + q^2 - 2*q - 1 = 0) →
  (r^3 + r^2 - 2*r - 1 = 0) →
  p*(q-r)^2 + q*(r-p)^2 + r*(p-q)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2822_282231


namespace NUMINAMATH_CALUDE_unique_solution_l2822_282211

theorem unique_solution (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (heq : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) :
  x = 12 ∧ y = 10 ∧ z = 8 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l2822_282211


namespace NUMINAMATH_CALUDE_continued_fraction_solution_l2822_282285

theorem continued_fraction_solution : 
  ∃ x : ℝ, x = 3 + 6 / (1 + 6 / x) ∧ x = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_solution_l2822_282285


namespace NUMINAMATH_CALUDE_num_measurable_weights_l2822_282298

/-- Represents the number of weights of each type -/
def num_weights : ℕ := 3

/-- Represents the weight values -/
def weight_values : List ℕ := [1, 5, 50]

/-- Represents the maximum number of weights that can be placed on one side of the scale -/
def max_weights_per_side : ℕ := num_weights * weight_values.length

/-- Represents the set of all possible weight combinations on one side of the scale -/
def weight_combinations : Finset (List ℕ) :=
  sorry

/-- Calculates the total weight of a combination -/
def total_weight (combination : List ℕ) : ℕ :=
  sorry

/-- Represents the set of all possible positive weight differences -/
def measurable_weights : Finset ℕ :=
  sorry

/-- The main theorem stating that the number of different positive weights
    that can be measured is 63 -/
theorem num_measurable_weights : measurable_weights.card = 63 :=
  sorry

end NUMINAMATH_CALUDE_num_measurable_weights_l2822_282298


namespace NUMINAMATH_CALUDE_exponent_simplification_l2822_282239

theorem exponent_simplification : 8^6 * 27^6 * 8^27 * 27^8 = 216^14 * 8^19 := by sorry

end NUMINAMATH_CALUDE_exponent_simplification_l2822_282239


namespace NUMINAMATH_CALUDE_existence_of_counterexample_l2822_282213

theorem existence_of_counterexample : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ |a - b| + (1 / (a - b)) < 2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_counterexample_l2822_282213


namespace NUMINAMATH_CALUDE_largest_n_with_conditions_l2822_282220

theorem largest_n_with_conditions : 
  ∃ (n : ℕ), n = 4513 ∧ 
  (∃ (m : ℕ), n^2 = (m+1)^3 - m^3) ∧
  (∃ (k : ℕ), 2*n + 99 = k^2) ∧
  (∀ (n' : ℕ), n' > n → 
    (¬∃ (m : ℕ), n'^2 = (m+1)^3 - m^3) ∨ 
    (¬∃ (k : ℕ), 2*n' + 99 = k^2)) := by
  sorry

#check largest_n_with_conditions

end NUMINAMATH_CALUDE_largest_n_with_conditions_l2822_282220


namespace NUMINAMATH_CALUDE_betty_age_l2822_282210

theorem betty_age (albert : ℕ) (mary : ℕ) (betty : ℕ)
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 22) :
  betty = 11 := by
  sorry

end NUMINAMATH_CALUDE_betty_age_l2822_282210


namespace NUMINAMATH_CALUDE_job_completion_time_l2822_282204

/-- Given two people working on a job, where the first person takes 3 hours and their combined
    work rate is 5/12 of the job per hour, prove that the second person takes 12 hours to
    complete the job individually. -/
theorem job_completion_time
  (time_person1 : ℝ)
  (combined_rate : ℝ)
  (h1 : time_person1 = 3)
  (h2 : combined_rate = 5 / 12)
  : ∃ (time_person2 : ℝ),
    time_person2 = 12 ∧
    1 / time_person1 + 1 / time_person2 = combined_rate :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_l2822_282204


namespace NUMINAMATH_CALUDE_olivias_initial_money_l2822_282299

/-- Calculates the initial amount of money Olivia had given the number of card packs, their prices, and the change received. -/
def initialMoney (basketballPacks : ℕ) (basketballPrice : ℕ) (baseballDecks : ℕ) (baseballPrice : ℕ) (change : ℕ) : ℕ :=
  basketballPacks * basketballPrice + baseballDecks * baseballPrice + change

/-- Proves that Olivia's initial amount of money was $50 given the problem conditions. -/
theorem olivias_initial_money :
  initialMoney 2 3 5 4 24 = 50 := by
  sorry

end NUMINAMATH_CALUDE_olivias_initial_money_l2822_282299


namespace NUMINAMATH_CALUDE_thirteenth_result_l2822_282226

theorem thirteenth_result (total_count : Nat) (total_avg first_avg last_avg : ℚ) :
  total_count = 25 →
  total_avg = 19 →
  first_avg = 14 →
  last_avg = 17 →
  (total_count * total_avg - 12 * first_avg - 12 * last_avg : ℚ) = 103 :=
by sorry

end NUMINAMATH_CALUDE_thirteenth_result_l2822_282226


namespace NUMINAMATH_CALUDE_position_interpretation_is_false_l2822_282254

/-- Represents a position in a grid -/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Interprets a position as (column, row) -/
def interpret (p : Position) : String :=
  s!"the {p.x}th column and the {p.y}th row"

/-- The statement to be proven false -/
def statement (p : Position) : String :=
  s!"the {p.y}th row and the {p.x}th column"

theorem position_interpretation_is_false : 
  statement (Position.mk 5 1) ≠ interpret (Position.mk 5 1) :=
sorry

end NUMINAMATH_CALUDE_position_interpretation_is_false_l2822_282254


namespace NUMINAMATH_CALUDE_rational_segment_existence_l2822_282230

theorem rational_segment_existence (f : ℚ → ℤ) : ∃ x y : ℚ, f x + f y ≤ 2 * f ((x + y) / 2) := by
  sorry

end NUMINAMATH_CALUDE_rational_segment_existence_l2822_282230


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2822_282252

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence. -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

/-- Given an arithmetic sequence with S₈ = 30 and S₄ = 7, prove that a₄ = 13/4. -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h₁ : S seq 8 = 30)
    (h₂ : S seq 4 = 7) : 
  seq.a 4 = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2822_282252


namespace NUMINAMATH_CALUDE_min_value_theorem_l2822_282288

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 2/b = 3) :
  (a + 1) * (b + 2) ≥ 50/9 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2822_282288


namespace NUMINAMATH_CALUDE_weighted_average_theorem_l2822_282295

def score1 : Rat := 55 / 100
def score2 : Rat := 67 / 100
def score3 : Rat := 76 / 100
def score4 : Rat := 82 / 100
def score5 : Rat := 85 / 100
def score6 : Rat := 48 / 60
def score7 : Rat := 150 / 200

def convertedScore6 : Rat := score6 * 100 / 60
def convertedScore7 : Rat := score7 * 100 / 200

def totalScores : Rat := score1 + score2 + score3 + score4 + score5 + convertedScore6 + convertedScore7
def numberOfScores : Nat := 7

theorem weighted_average_theorem :
  totalScores / numberOfScores = (55 + 67 + 76 + 82 + 85 + 80 + 75) / 7 := by sorry

end NUMINAMATH_CALUDE_weighted_average_theorem_l2822_282295


namespace NUMINAMATH_CALUDE_cubic_roots_relationship_l2822_282248

/-- The cubic polynomial f(x) -/
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4

/-- The cubic polynomial h(x) -/
def h (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Theorem stating the relationship between f and h and the values of a, b, and c -/
theorem cubic_roots_relationship (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  (∀ x : ℝ, f x = 0 → h a b c (x^3) = 0) →
  a = -6 ∧ b = -9 ∧ c = 20 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_relationship_l2822_282248


namespace NUMINAMATH_CALUDE_negation_of_true_is_false_l2822_282275

theorem negation_of_true_is_false (p : Prop) : p → ¬p = False := by
  sorry

end NUMINAMATH_CALUDE_negation_of_true_is_false_l2822_282275


namespace NUMINAMATH_CALUDE_light_reflection_l2822_282257

-- Define points M and P
def M : ℝ × ℝ := (-1, 3)
def P : ℝ × ℝ := (1, 0)

-- Define the reflecting lines
def x_axis (x y : ℝ) : Prop := y = 0
def reflecting_line (x y : ℝ) : Prop := x + y = 4

-- Define the light rays
def l1 : ℝ × ℝ → Prop := sorry
def l2 : ℝ × ℝ → Prop := sorry
def l3 : ℝ × ℝ → Prop := sorry

-- Define the reflection operation
def reflect (line : ℝ × ℝ → Prop) (ray : ℝ × ℝ → Prop) : ℝ × ℝ → Prop := sorry

-- State the theorem
theorem light_reflection :
  (∀ x y, l2 (x, y) ↔ y = 3/2 * (x - 1)) ∧
  (∀ x y, l3 (x, y) ↔ 2*x - 3*y + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_light_reflection_l2822_282257


namespace NUMINAMATH_CALUDE_cubic_fraction_equals_10_04_l2822_282256

theorem cubic_fraction_equals_10_04 :
  let a : ℝ := 6
  let b : ℝ := 3
  let c : ℝ := 2
  (a^3 + b^3 + c^3) / (a^2 - a*b + b^2 - b*c + c^2) = 10.04 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_10_04_l2822_282256


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l2822_282292

theorem cubic_expression_evaluation : (3^3 - 3) - (4^3 - 4) + (5^3 - 5) = 84 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l2822_282292


namespace NUMINAMATH_CALUDE_triangle_area_l2822_282293

/-- Given a triangle with perimeter 36 and inradius 2.5, its area is 45 -/
theorem triangle_area (p : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : p = 36) -- perimeter is 36
  (h2 : r = 2.5) -- inradius is 2.5
  (h3 : A = r * (p / 2)) -- area formula: A = r * s, where s is semiperimeter (p / 2)
  : A = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2822_282293
