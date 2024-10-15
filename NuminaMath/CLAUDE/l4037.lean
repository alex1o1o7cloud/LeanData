import Mathlib

namespace NUMINAMATH_CALUDE_no_solution_implies_a_equals_negative_two_l4037_403711

theorem no_solution_implies_a_equals_negative_two (a : ℝ) : 
  (∀ x y : ℝ, ¬(a * x + 2 * y = a + 2 ∧ 2 * x + a * y = 2 * a)) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_equals_negative_two_l4037_403711


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4037_403715

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sin θ)^4 / a + (Real.cos θ)^4 / b = 1 / (a + b) →
  (Real.sin θ)^8 / a^3 + (Real.cos θ)^8 / b^3 = 1 / (a + b)^3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4037_403715


namespace NUMINAMATH_CALUDE_four_birds_joined_l4037_403768

/-- The number of birds that joined the fence -/
def birds_joined (initial_birds final_birds : ℕ) : ℕ :=
  final_birds - initial_birds

/-- Proof that 4 birds joined the fence -/
theorem four_birds_joined :
  let initial_birds : ℕ := 2
  let final_birds : ℕ := 6
  birds_joined initial_birds final_birds = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_birds_joined_l4037_403768


namespace NUMINAMATH_CALUDE_linear_function_properties_l4037_403747

/-- A linear function f(x) = k(x + 2) where k ≠ 0 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * (x + 2)

/-- g is f shifted 2 units upwards -/
def g (k : ℝ) (x : ℝ) : ℝ := f k x + 2

theorem linear_function_properties (k : ℝ) (h : k ≠ 0) :
  (f k (-2) = 0) ∧
  (g k 1 = -2 → k = -4/3) ∧
  (0 > f k 0 ∧ f k 0 > -2 → -1 < k ∧ k < 0) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l4037_403747


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_l4037_403735

theorem quadratic_is_perfect_square : ∃ (a b : ℝ), ∀ x : ℝ, x^2 - 18*x + 81 = (a*x + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_l4037_403735


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l4037_403752

/-- A parabola with vertex at the origin passing through (-2, 4) -/
structure Parabola where
  /-- The equation of the parabola is either x^2 = ay or y^2 = bx for some a, b ∈ ℝ -/
  equation : (∃ a : ℝ, ∀ x y : ℝ, y = a * x^2) ∨ (∃ b : ℝ, ∀ x y : ℝ, x = b * y^2)
  /-- The parabola passes through the point (-2, 4) -/
  point : (∃ a : ℝ, 4 = a * (-2)^2) ∨ (∃ b : ℝ, -2 = b * 4^2)

/-- The standard equation of the parabola is either x^2 = y or y^2 = -8x -/
theorem parabola_standard_equation (p : Parabola) :
  (∀ x y : ℝ, y = x^2) ∨ (∀ x y : ℝ, x = -8 * y^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l4037_403752


namespace NUMINAMATH_CALUDE_absolute_value_sum_equality_l4037_403737

theorem absolute_value_sum_equality (x y : ℝ) : 
  (|x + y| = |x| + |y|) ↔ x * y ≥ 0 := by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_equality_l4037_403737


namespace NUMINAMATH_CALUDE_inequality_solution_and_abs_inequality_l4037_403758

def f (x : ℝ) := |x - 1|

theorem inequality_solution_and_abs_inequality (a b : ℝ) :
  (∀ x, f x + f (x + 4) ≥ 8 ↔ x ≤ -5 ∨ x ≥ 3) ∧
  (|a| < 1 → |b| < 1 → a ≠ 0 → f (a * b) > |a| * f (b / a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_and_abs_inequality_l4037_403758


namespace NUMINAMATH_CALUDE_sum_xy_value_l4037_403743

theorem sum_xy_value (x y : ℝ) (h1 : x + 2*y = 5) (h2 : (x + y) / 3 = 1.222222222222222) :
  x + y = 3.666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_sum_xy_value_l4037_403743


namespace NUMINAMATH_CALUDE_sum_remainder_l4037_403793

theorem sum_remainder (a b c d e : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9)
  (he : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_remainder_l4037_403793


namespace NUMINAMATH_CALUDE_solution_set_f_positive_range_of_a_l4037_403734

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

-- Theorem for the solution set of f(x) > 0
theorem solution_set_f_positive :
  {x : ℝ | f x > 0} = {x : ℝ | x < -1/3 ∨ x > 3} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∃ x₀ : ℝ, f x₀ + 2*a^2 < 4*a} = {a : ℝ | -1/2 < a ∧ a < 5/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_positive_range_of_a_l4037_403734


namespace NUMINAMATH_CALUDE_largest_package_size_l4037_403741

theorem largest_package_size (john_markers alex_markers : ℕ) 
  (h1 : john_markers = 36) (h2 : alex_markers = 60) : 
  Nat.gcd john_markers alex_markers = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l4037_403741


namespace NUMINAMATH_CALUDE_bucket_filling_time_l4037_403748

theorem bucket_filling_time (total_time : ℝ) (h : total_time = 135) : 
  (2 / 3 : ℝ) * total_time = 90 := by
  sorry

end NUMINAMATH_CALUDE_bucket_filling_time_l4037_403748


namespace NUMINAMATH_CALUDE_cube_iff_greater_l4037_403788

theorem cube_iff_greater (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end NUMINAMATH_CALUDE_cube_iff_greater_l4037_403788


namespace NUMINAMATH_CALUDE_cake_division_l4037_403751

theorem cake_division (n_cakes : ℕ) (n_girls : ℕ) (share : ℚ) :
  n_cakes = 11 →
  n_girls = 6 →
  share = 1 + 1/2 + 1/4 + 1/12 →
  ∃ (division : List (List ℚ)),
    (∀ piece ∈ division.join, piece ≠ 1/6) ∧
    (division.length = n_girls) ∧
    (∀ girl_share ∈ division, girl_share.sum = share) ∧
    (division.join.sum = n_cakes) :=
by sorry

end NUMINAMATH_CALUDE_cake_division_l4037_403751


namespace NUMINAMATH_CALUDE_function_graph_overlap_l4037_403775

theorem function_graph_overlap (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, a^(x/2) = 2^(-x/2)) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_graph_overlap_l4037_403775


namespace NUMINAMATH_CALUDE_order_of_exponents_l4037_403705

theorem order_of_exponents :
  let a : ℝ := (36 : ℝ) ^ (1/5)
  let b : ℝ := (3 : ℝ) ^ (4/3)
  let c : ℝ := (9 : ℝ) ^ (2/5)
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_order_of_exponents_l4037_403705


namespace NUMINAMATH_CALUDE_value_of_y_l4037_403757

theorem value_of_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1/y) (eq2 : y = 1 + 1/x) : y = (1 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l4037_403757


namespace NUMINAMATH_CALUDE_decagon_diagonals_l4037_403704

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l4037_403704


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l4037_403744

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  a₁_eq_1 : a 1 = 1
  geometric_subseq : (a 3)^2 = a 1 * a 9

/-- The general term of the arithmetic sequence is either n or 1 -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = n) ∨ (∀ n : ℕ, seq.a n = 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l4037_403744


namespace NUMINAMATH_CALUDE_coefficient_x_squared_l4037_403708

theorem coefficient_x_squared (x y : ℝ) : 
  let expansion := (x - 2 * y^3) * (x + 1/y)^5
  ∃ (a b c : ℝ), expansion = a * x^3 + (-20) * x^2 + b * x + c :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_l4037_403708


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l4037_403791

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := x = y^2 / 10 + 5 / 2

/-- The focus of the parabola -/
def parabola_focus : ℝ × ℝ := (5, 0)

/-- The directrix of the parabola is the x-axis -/
def parabola_directrix (x : ℝ) : ℝ × ℝ := (x, 0)

theorem hyperbola_parabola_intersection :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    hyperbola x₁ y₁ ∧ parabola x₁ y₁ ∧
    hyperbola x₂ y₂ ∧ parabola x₂ y₂ ∧
    (x₁, y₁) ≠ (x₂, y₂) :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l4037_403791


namespace NUMINAMATH_CALUDE_megan_popsicles_l4037_403772

/-- The number of Popsicles Megan can finish in a given time period --/
def popsicles_finished (total_minutes : ℕ) (popsicle_time : ℕ) (break_time : ℕ) (break_interval : ℕ) : ℕ :=
  let effective_minutes := total_minutes - (total_minutes / (break_interval * 60)) * break_time
  (effective_minutes / popsicle_time : ℕ)

/-- Theorem stating the number of Popsicles Megan can finish in 5 hours and 40 minutes --/
theorem megan_popsicles :
  popsicles_finished 340 20 5 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_megan_popsicles_l4037_403772


namespace NUMINAMATH_CALUDE_min_max_expression_l4037_403750

theorem min_max_expression (a b c : ℝ) 
  (eq1 : a^2 + a*b + b^2 = 19)
  (eq2 : b^2 + b*c + c^2 = 19) :
  (∃ x y z : ℝ, x^2 + x*y + y^2 = 19 ∧ y^2 + y*z + z^2 = 19 ∧ z^2 + z*x + x^2 = 0) ∧
  (∀ x y z : ℝ, x^2 + x*y + y^2 = 19 → y^2 + y*z + z^2 = 19 → z^2 + z*x + x^2 ≤ 76) :=
by
  sorry

end NUMINAMATH_CALUDE_min_max_expression_l4037_403750


namespace NUMINAMATH_CALUDE_decrease_by_one_point_five_l4037_403783

/-- Represents a linear regression equation of the form y = a + bx -/
structure LinearRegression where
  a : ℝ
  b : ℝ

/-- The change in y when x increases by one unit in a linear regression -/
def change_in_y (lr : LinearRegression) : ℝ := -lr.b

/-- Theorem: In the given linear regression, when x increases by one unit, y decreases by 1.5 units -/
theorem decrease_by_one_point_five :
  let lr : LinearRegression := { a := 2, b := -1.5 }
  change_in_y lr = -1.5 := by sorry

end NUMINAMATH_CALUDE_decrease_by_one_point_five_l4037_403783


namespace NUMINAMATH_CALUDE_union_of_P_and_Q_l4037_403730

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def Q : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Define the union of P and Q
def PUnionQ : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Theorem statement
theorem union_of_P_and_Q : P ∪ Q = PUnionQ := by
  sorry

end NUMINAMATH_CALUDE_union_of_P_and_Q_l4037_403730


namespace NUMINAMATH_CALUDE_cookie_sales_ratio_l4037_403786

theorem cookie_sales_ratio : 
  ∀ (goal : ℕ) (first third fourth fifth left : ℕ),
    goal ≥ 150 →
    first = 5 →
    third = 10 →
    fifth = 10 →
    left = 75 →
    goal - left = first + 4 * first + third + fourth + fifth →
    fourth / third = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_sales_ratio_l4037_403786


namespace NUMINAMATH_CALUDE_cube_of_square_of_third_smallest_prime_l4037_403723

def third_smallest_prime : ℕ := sorry

theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime ^ 2) ^ 3 = 15625 := by sorry

end NUMINAMATH_CALUDE_cube_of_square_of_third_smallest_prime_l4037_403723


namespace NUMINAMATH_CALUDE_age_ratio_is_four_thirds_l4037_403796

-- Define the current ages of Arun and Deepak
def arun_current_age : ℕ := 26 - 6
def deepak_current_age : ℕ := 15

-- Define the ratio of their ages
def age_ratio : ℚ := arun_current_age / deepak_current_age

-- Theorem to prove
theorem age_ratio_is_four_thirds : age_ratio = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_is_four_thirds_l4037_403796


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l4037_403764

theorem logarithmic_equation_solution :
  ∃! x : ℝ, x > 0 ∧ 3^x = x + 50 := by sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l4037_403764


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l4037_403784

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l4037_403784


namespace NUMINAMATH_CALUDE_perry_vs_phil_l4037_403779

-- Define the number of games won by each player
def phil_games : ℕ := 12
def charlie_games : ℕ := phil_games - 3
def dana_games : ℕ := charlie_games + 2
def perry_games : ℕ := dana_games + 5

-- Theorem statement
theorem perry_vs_phil : perry_games = phil_games + 4 := by
  sorry

end NUMINAMATH_CALUDE_perry_vs_phil_l4037_403779


namespace NUMINAMATH_CALUDE_equation_solution_l4037_403761

theorem equation_solution (x : ℝ) (a b : ℕ) :
  (x^2 + 5*x + 5/x + 1/x^2 = 40) →
  (x = a + Real.sqrt b) →
  (a > 0 ∧ b > 0) →
  (a + b = 11) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l4037_403761


namespace NUMINAMATH_CALUDE_servant_service_duration_l4037_403798

def yearly_payment : ℕ := 800
def uniform_price : ℕ := 300
def actual_payment : ℕ := 600

def months_served : ℕ := 7

theorem servant_service_duration :
  yearly_payment = 800 ∧
  uniform_price = 300 ∧
  actual_payment = 600 →
  months_served = 7 :=
by sorry

end NUMINAMATH_CALUDE_servant_service_duration_l4037_403798


namespace NUMINAMATH_CALUDE_unique_common_term_l4037_403759

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => x (n + 1) + 2 * x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * y (n + 1) + 3 * y n

theorem unique_common_term : ∀ n : ℕ, x n = y n → x n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_common_term_l4037_403759


namespace NUMINAMATH_CALUDE_halfway_between_one_sixth_and_one_fourth_l4037_403765

theorem halfway_between_one_sixth_and_one_fourth : 
  (1/6 : ℚ) / 2 + (1/4 : ℚ) / 2 = 5/24 := by sorry

end NUMINAMATH_CALUDE_halfway_between_one_sixth_and_one_fourth_l4037_403765


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_5_to_1994_l4037_403771

theorem rightmost_three_digits_of_5_to_1994 : 5^1994 % 1000 = 625 := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_5_to_1994_l4037_403771


namespace NUMINAMATH_CALUDE_derivative_x_ln_x_l4037_403781

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem derivative_x_ln_x (x : ℝ) (h : x > 0) : 
  deriv f x = 1 + Real.log x :=
sorry

end NUMINAMATH_CALUDE_derivative_x_ln_x_l4037_403781


namespace NUMINAMATH_CALUDE_divisibility_by_240_l4037_403736

theorem divisibility_by_240 (p : ℕ) (hp : Nat.Prime p) (hp_ge_7 : p ≥ 7) :
  (240 : ℕ) ∣ (p^4 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_240_l4037_403736


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_composite_sum_l4037_403792

theorem diophantine_equation_solutions :
  {(m, n) : ℕ × ℕ | 5 * m + 8 * n = 120} = {(24, 0), (16, 5), (8, 10), (0, 15)} := by sorry

theorem composite_sum :
  ∀ (a b c : ℕ+), c > 1 → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / c →
  (∃ d : ℕ, 1 < d ∧ d < a + c ∧ (a + c) % d = 0) ∨
  (∃ d : ℕ, 1 < d ∧ d < b + c ∧ (b + c) % d = 0) := by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_composite_sum_l4037_403792


namespace NUMINAMATH_CALUDE_rope_length_proof_l4037_403774

/-- The length of a rope after being folded in half twice -/
def folded_length : ℝ := 10

/-- The number of times the rope is folded in half -/
def fold_count : ℕ := 2

/-- Calculates the original length of the rope before folding -/
def original_length : ℝ := folded_length * (2 ^ fold_count)

/-- Proves that the original length of the rope is 40 centimeters -/
theorem rope_length_proof : original_length = 40 := by
  sorry

end NUMINAMATH_CALUDE_rope_length_proof_l4037_403774


namespace NUMINAMATH_CALUDE_candy_bar_sales_l4037_403732

/-- The number of candy bars Marvin sold -/
def marvins_candy_bars : ℕ := 35

theorem candy_bar_sales : 
  let candy_bar_price : ℕ := 2
  let tinas_candy_bars : ℕ := 3 * marvins_candy_bars
  let marvins_revenue : ℕ := candy_bar_price * marvins_candy_bars
  let tinas_revenue : ℕ := candy_bar_price * tinas_candy_bars
  tinas_revenue = marvins_revenue + 140 → marvins_candy_bars = 35 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_sales_l4037_403732


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l4037_403702

theorem absolute_value_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 10 * a * b) :
  |((a + b) / (a - b))| = Real.sqrt (3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l4037_403702


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4037_403742

theorem polynomial_simplification (p : ℝ) : 
  (4 * p^4 + 2 * p^3 - 7 * p^2 + 3 * p - 5) + (-3 * p^4 - 2 * p^3 + 8 * p^2 - 4 * p + 6) = 
  p^4 + p^2 - p + 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l4037_403742


namespace NUMINAMATH_CALUDE_pencil_distribution_l4037_403717

theorem pencil_distribution (x y : ℕ+) (h1 : 3 * x < 48) (h2 : 48 < 4 * x) 
  (h3 : 4 * y < 48) (h4 : 48 < 5 * y) : 
  (3 * x < 48 ∧ 48 < 4 * x) ∧ (4 * y < 48 ∧ 48 < 5 * y) := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l4037_403717


namespace NUMINAMATH_CALUDE_problem_statement_l4037_403790

theorem problem_statement (x y : ℝ) (h : (x + y - 2020) * (2023 - x - y) = 2) :
  (x + y - 2020)^2 * (2023 - x - y)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4037_403790


namespace NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l4037_403773

theorem product_from_lcm_and_gcd (a b : ℤ) : 
  lcm a b = 36 → gcd a b = 6 → a * b = 216 := by sorry

end NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l4037_403773


namespace NUMINAMATH_CALUDE_longest_tape_l4037_403714

theorem longest_tape (red_tape blue_tape yellow_tape : ℚ) 
  (h_red : red_tape = 11/6)
  (h_blue : blue_tape = 7/4)
  (h_yellow : yellow_tape = 13/8) :
  red_tape > blue_tape ∧ red_tape > yellow_tape := by
  sorry

end NUMINAMATH_CALUDE_longest_tape_l4037_403714


namespace NUMINAMATH_CALUDE_decaf_coffee_percentage_l4037_403763

/-- Proves that the percentage of decaffeinated coffee in the initial stock is 40% --/
theorem decaf_coffee_percentage
  (initial_stock : ℝ)
  (additional_purchase : ℝ)
  (decaf_percent_additional : ℝ)
  (decaf_percent_total : ℝ)
  (h1 : initial_stock = 400)
  (h2 : additional_purchase = 100)
  (h3 : decaf_percent_additional = 60)
  (h4 : decaf_percent_total = 44)
  (h5 : decaf_percent_total / 100 * (initial_stock + additional_purchase) =
        (initial_stock * x / 100) + (additional_purchase * decaf_percent_additional / 100)) :
  x = 40 := by
  sorry

end NUMINAMATH_CALUDE_decaf_coffee_percentage_l4037_403763


namespace NUMINAMATH_CALUDE_difference_of_squares_l4037_403797

theorem difference_of_squares (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l4037_403797


namespace NUMINAMATH_CALUDE_no_natural_n_power_of_two_l4037_403739

theorem no_natural_n_power_of_two : ∀ n : ℕ, ¬∃ k : ℕ, 6 * n^2 + 5 * n = 2^k := by sorry

end NUMINAMATH_CALUDE_no_natural_n_power_of_two_l4037_403739


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l4037_403755

/-- Given a parabola with focus F(a,0) where a < 0, its standard equation is y^2 = 4ax -/
theorem parabola_standard_equation (a : ℝ) (h : a < 0) :
  ∃ (x y : ℝ), y^2 = 4*a*x :=
sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l4037_403755


namespace NUMINAMATH_CALUDE_intersection_distance_l4037_403780

theorem intersection_distance : ∃ (C D : ℝ × ℝ),
  (C.2 = 2 ∧ C.2 = 3 * C.1^2 + 2 * C.1 - 5) ∧
  (D.2 = 2 ∧ D.2 = 3 * D.1^2 + 2 * D.1 - 5) ∧
  C ≠ D ∧
  |C.1 - D.1| = 2 * Real.sqrt 22 / 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l4037_403780


namespace NUMINAMATH_CALUDE_equation_solution_l4037_403745

theorem equation_solution : 
  ∃! x : ℝ, x ≠ 3 ∧ x + 25 / (x - 3) = -8 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l4037_403745


namespace NUMINAMATH_CALUDE_semicircle_tangent_circle_and_triangle_l4037_403728

/-- Given a semicircle with diameter AB and center O, where AO = OB = R,
    and two semicircles drawn over AO and BO, this theorem proves:
    1. The radius of the circle tangent to all three semicircles is R/3
    2. The sides of the triangle formed by the tangency points are 2R/5 and (R/5)√10 -/
theorem semicircle_tangent_circle_and_triangle (R : ℝ) (R_pos : R > 0) :
  ∃ (r a b : ℝ),
    r = R / 3 ∧
    2 * a = 2 * R / 5 ∧
    b = (R / 5) * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_tangent_circle_and_triangle_l4037_403728


namespace NUMINAMATH_CALUDE_integer_solutions_of_system_l4037_403785

theorem integer_solutions_of_system : 
  ∀ x y z t : ℤ, 
    (x * z - 2 * y * t = 3 ∧ x * t + y * z = 1) ↔ 
    ((x, y, z, t) = (1, 0, 3, 1) ∨ 
     (x, y, z, t) = (-1, 0, -3, -1) ∨ 
     (x, y, z, t) = (3, 1, 1, 0) ∨ 
     (x, y, z, t) = (-3, -1, -1, 0)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_system_l4037_403785


namespace NUMINAMATH_CALUDE_determinant_max_value_l4037_403782

theorem determinant_max_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → a * b - 1 ≥ x * y - 1) →
  a * b - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_determinant_max_value_l4037_403782


namespace NUMINAMATH_CALUDE_arithmetic_reciprocal_sequence_l4037_403767

theorem arithmetic_reciprocal_sequence (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (harith : ∃ d ≠ 0, b = a + d ∧ c = a + 2*d) :
  ¬(∃ r ≠ 0, (1/b - 1/a) = r ∧ (1/c - 1/b) = r) ∧
  ¬(∃ q ≠ 1, (1/b) / (1/a) = q ∧ (1/c) / (1/b) = q) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_reciprocal_sequence_l4037_403767


namespace NUMINAMATH_CALUDE_jogger_speed_l4037_403712

/-- The speed of a jogger given specific conditions involving a train --/
theorem jogger_speed (train_length : ℝ) (initial_distance : ℝ) (train_speed : ℝ) (passing_time : ℝ)
  (h1 : train_length = 120)
  (h2 : initial_distance = 120)
  (h3 : train_speed = 45)
  (h4 : passing_time = 24)
  : ∃ (jogger_speed : ℝ), jogger_speed = 9 := by
  sorry


end NUMINAMATH_CALUDE_jogger_speed_l4037_403712


namespace NUMINAMATH_CALUDE_probability_theorem_l4037_403766

-- Define the total number of students
def total_students : ℕ := 20

-- Define the fraction of students interested in the career
def interested_fraction : ℚ := 4 / 5

-- Define the number of interested students
def interested_students : ℕ := (interested_fraction * total_students).num.toNat

-- Define the function to calculate the probability
def probability_at_least_one_interested : ℚ :=
  1 - (total_students - interested_students) * (total_students - interested_students - 1) /
      (total_students * (total_students - 1))

-- Theorem statement
theorem probability_theorem :
  probability_at_least_one_interested = 92 / 95 :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l4037_403766


namespace NUMINAMATH_CALUDE_tg_ctg_equation_solution_l4037_403769

theorem tg_ctg_equation_solution (x : ℝ) :
  (∀ n : ℤ, x ≠ (n : ℝ) * π / 2) →
  (Real.tan x ^ 4 + (1 / Real.tan x) ^ 4 = (82 / 9) * (Real.tan x * Real.tan (2 * x) + 1) * Real.cos (2 * x)) ↔
  ∃ k : ℤ, x = π / 6 * ((3 * k : ℝ) + 1) ∨ x = π / 6 * ((3 * k : ℝ) - 1) :=
by sorry

end NUMINAMATH_CALUDE_tg_ctg_equation_solution_l4037_403769


namespace NUMINAMATH_CALUDE_append_five_to_two_digit_number_l4037_403722

/-- Given a two-digit number with tens digit t and units digit u,
    appending the digit 5 results in the number 100t + 10u + 5 -/
theorem append_five_to_two_digit_number (t u : ℕ) :
  let original := 10 * t + u
  let appended := original * 10 + 5
  appended = 100 * t + 10 * u + 5 := by
sorry

end NUMINAMATH_CALUDE_append_five_to_two_digit_number_l4037_403722


namespace NUMINAMATH_CALUDE_x_in_terms_of_y_and_k_l4037_403726

theorem x_in_terms_of_y_and_k (x y k : ℝ) :
  x / (x - k) = (y^2 + 3*y + 2) / (y^2 + 3*y + 1) →
  x = k*y^2 + 3*k*y + 2*k := by
  sorry

end NUMINAMATH_CALUDE_x_in_terms_of_y_and_k_l4037_403726


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l4037_403777

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l4037_403777


namespace NUMINAMATH_CALUDE_game_probability_specific_case_l4037_403753

def game_probability (total_rounds : ℕ) 
  (alex_prob : ℚ) (chelsea_prob : ℚ) (mel_prob : ℚ)
  (alex_wins : ℕ) (chelsea_wins : ℕ) (mel_wins : ℕ) : ℚ :=
  (alex_prob ^ alex_wins) * 
  (chelsea_prob ^ chelsea_wins) * 
  (mel_prob ^ mel_wins) * 
  (Nat.choose total_rounds alex_wins).choose chelsea_wins

theorem game_probability_specific_case : 
  game_probability 8 (5/12) (1/3) (1/4) 3 4 1 = 625/9994 := by
  sorry

end NUMINAMATH_CALUDE_game_probability_specific_case_l4037_403753


namespace NUMINAMATH_CALUDE_complex_set_sum_l4037_403738

/-- A set of complex numbers with closure under multiplication property -/
def ClosedMultSet (S : Set ℂ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x * y) ∈ S

theorem complex_set_sum (a b c d : ℂ) :
  let S := {a, b, c, d}
  ClosedMultSet S →
  a^2 = 1 →
  b^2 = 1 →
  c^2 = b →
  b + c + d = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_set_sum_l4037_403738


namespace NUMINAMATH_CALUDE_abc_inequalities_l4037_403725

theorem abc_inequalities (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  (2 * a * b + b * c + c * a + c^2 / 2 ≤ 1 / 2) ∧ 
  ((a^2 + c^2) / b + (b^2 + a^2) / c + (c^2 + b^2) / a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequalities_l4037_403725


namespace NUMINAMATH_CALUDE_division_of_fractions_calculate_fraction_division_l4037_403799

theorem division_of_fractions (a b c : ℚ) (hb : b ≠ 0) :
  a / (c / b) = (a * b) / c :=
by sorry

theorem calculate_fraction_division :
  (4 : ℚ) / (5 / 7) = 28 / 5 :=
by sorry

end NUMINAMATH_CALUDE_division_of_fractions_calculate_fraction_division_l4037_403799


namespace NUMINAMATH_CALUDE_sum_289_37_base4_l4037_403762

/-- Converts a natural number to its base 4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- Checks if a list of natural numbers represents a valid base 4 number -/
def isValidBase4 (l : List ℕ) : Prop :=
  ∀ d ∈ l, d < 4

theorem sum_289_37_base4 :
  let sum := 289 + 37
  let base4Sum := toBase4 sum
  isValidBase4 base4Sum ∧ base4Sum = [1, 1, 0, 1, 2] := by sorry

end NUMINAMATH_CALUDE_sum_289_37_base4_l4037_403762


namespace NUMINAMATH_CALUDE_inverse_of_B_squared_l4037_403703

theorem inverse_of_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![3, -2; 0, 5]) : 
  (B^2)⁻¹ = !![9, -16; 0, 25] := by sorry

end NUMINAMATH_CALUDE_inverse_of_B_squared_l4037_403703


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l4037_403749

theorem unique_congruence_in_range : ∃! n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n ≡ 12345 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l4037_403749


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l4037_403713

theorem absolute_value_equation_solution :
  let f : ℝ → ℝ := λ x => 3*x + 6
  let g : ℝ → ℝ := λ x => |(-20 + x^2)|
  ∃ x₁ x₂ : ℝ, x₁ = (3 + Real.sqrt 113) / 2 ∧
              x₂ = (3 - Real.sqrt 113) / 2 ∧
              (∀ x : ℝ, f x = g x ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l4037_403713


namespace NUMINAMATH_CALUDE_floor_sum_example_l4037_403707

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l4037_403707


namespace NUMINAMATH_CALUDE_trivia_competition_score_l4037_403733

theorem trivia_competition_score :
  ∀ (total_members absent_members points_per_member : ℕ),
    total_members = 120 →
    absent_members = 37 →
    points_per_member = 24 →
    (total_members - absent_members) * points_per_member = 1992 :=
by
  sorry

end NUMINAMATH_CALUDE_trivia_competition_score_l4037_403733


namespace NUMINAMATH_CALUDE_pet_shop_solution_l4037_403789

/-- Represents the pet shop inventory --/
structure PetShop where
  kittens : ℕ
  hamsters : ℕ
  birds : ℕ
  puppies : ℕ

/-- The initial state of the pet shop --/
def initial_state : PetShop :=
  { kittens := 45,
    hamsters := 30,
    birds := 60,
    puppies := 15 }

/-- The final state of the pet shop after changes --/
def final_state : PetShop :=
  { kittens := initial_state.kittens,
    hamsters := initial_state.hamsters,
    birds := initial_state.birds + 10,
    puppies := initial_state.puppies - 5 }

/-- Theorem stating the correctness of the solution --/
theorem pet_shop_solution :
  (initial_state.kittens + initial_state.hamsters + initial_state.birds + initial_state.puppies = 150) ∧
  (3 * initial_state.hamsters = 2 * initial_state.kittens) ∧
  (initial_state.birds = initial_state.hamsters + 30) ∧
  (4 * initial_state.puppies = initial_state.birds) ∧
  (final_state.kittens + final_state.hamsters + final_state.birds + final_state.puppies = 155) ∧
  (final_state.kittens = 45) ∧
  (final_state.hamsters = 30) ∧
  (final_state.birds = 70) ∧
  (final_state.puppies = 10) := by
  sorry


end NUMINAMATH_CALUDE_pet_shop_solution_l4037_403789


namespace NUMINAMATH_CALUDE_existence_of_n_l4037_403794

theorem existence_of_n (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h_cd : c * d = 1) :
  ∃ n : ℤ, (a * b : ℝ) ≤ (n : ℝ)^2 ∧ (n : ℝ)^2 ≤ (a + c) * (b + d) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_l4037_403794


namespace NUMINAMATH_CALUDE_like_terms_imply_expression_value_l4037_403727

theorem like_terms_imply_expression_value :
  ∀ (a b : ℤ),
  (2 : ℤ) = 1 - a →
  (5 : ℤ) = 3 * b - 1 →
  5 * a * b^2 - (6 * a^2 * b - 3 * (a * b^2 + 2 * a^2 * b)) = -32 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_imply_expression_value_l4037_403727


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l4037_403706

/-- Proves that given a car's average speed and first hour speed, we can determine the second hour speed -/
theorem car_speed_second_hour 
  (average_speed : ℝ) 
  (first_hour_speed : ℝ) 
  (h1 : average_speed = 55) 
  (h2 : first_hour_speed = 65) : 
  ∃ (second_hour_speed : ℝ), 
    second_hour_speed = 45 ∧ 
    average_speed = (first_hour_speed + second_hour_speed) / 2 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l4037_403706


namespace NUMINAMATH_CALUDE_exponent_rule_l4037_403787

theorem exponent_rule (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_rule_l4037_403787


namespace NUMINAMATH_CALUDE_least_value_of_x_l4037_403778

theorem least_value_of_x (x p : ℕ) (h1 : x > 0) (h2 : Nat.Prime p) 
  (h3 : ∃ (k : ℕ), k > 0 ∧ x = 11 * p * k ∧ Nat.Prime k ∧ Even k) : 
  x ≥ 44 ∧ ∃ (x₀ : ℕ), x₀ ≥ 44 ∧ 
    (∃ (p₀ : ℕ), Nat.Prime p₀ ∧ ∃ (k₀ : ℕ), k₀ > 0 ∧ x₀ = 11 * p₀ * k₀ ∧ Nat.Prime k₀ ∧ Even k₀) :=
by sorry

end NUMINAMATH_CALUDE_least_value_of_x_l4037_403778


namespace NUMINAMATH_CALUDE_rectangular_pen_max_area_l4037_403776

/-- The perimeter of the rectangular pen -/
def perimeter : ℝ := 60

/-- The maximum possible area of a rectangular pen with the given perimeter -/
def max_area : ℝ := 225

/-- Theorem: The maximum area of a rectangular pen with a perimeter of 60 feet is 225 square feet -/
theorem rectangular_pen_max_area : 
  ∀ (width height : ℝ), 
  width > 0 → height > 0 → 
  2 * (width + height) = perimeter → 
  width * height ≤ max_area :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_pen_max_area_l4037_403776


namespace NUMINAMATH_CALUDE_point_on_axes_l4037_403770

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The coordinate axes in 2D space -/
def CoordinateAxes : Set Point2D :=
  {p : Point2D | p.x = 0 ∨ p.y = 0}

/-- Theorem: If xy = 0, then P(x,y) is located on the coordinate axes -/
theorem point_on_axes (p : Point2D) (h : p.x * p.y = 0) : p ∈ CoordinateAxes := by
  sorry

end NUMINAMATH_CALUDE_point_on_axes_l4037_403770


namespace NUMINAMATH_CALUDE_parabola_translation_l4037_403724

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation :
  let original := Parabola.mk (-5) 0 1
  let translated := translate original (-1) (-2)
  translated = Parabola.mk (-5) 10 (-1) := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l4037_403724


namespace NUMINAMATH_CALUDE_double_fibonacci_sum_convergence_l4037_403701

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def G (n : ℕ) : ℕ := 2 * fibonacci n

theorem double_fibonacci_sum_convergence :
  (∑' n, (G n : ℝ) / 5^n) = 10/19 := by sorry

end NUMINAMATH_CALUDE_double_fibonacci_sum_convergence_l4037_403701


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_multiples_of_three_l4037_403718

theorem sum_of_three_consecutive_multiples_of_three (a b c : ℕ) : 
  a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0 ∧  -- a, b, c are multiples of 3
  b = a + 3 ∧ c = b + 3 ∧               -- a, b, c are consecutive
  c = 27 →                              -- the largest number is 27
  a + b + c = 72 :=                     -- the sum is 72
by sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_multiples_of_three_l4037_403718


namespace NUMINAMATH_CALUDE_positive_range_of_even_function_l4037_403720

def evenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem positive_range_of_even_function
  (f : ℝ → ℝ)
  (f' : ℝ → ℝ)
  (h_even : evenFunction f)
  (h_deriv : ∀ x ≠ 0, HasDerivAt f (f' x) x)
  (h_zero : f (-1) = 0)
  (h_ineq : ∀ x > 0, x * f' x - f x < 0) :
  {x : ℝ | f x > 0} = Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1 := by
sorry

end NUMINAMATH_CALUDE_positive_range_of_even_function_l4037_403720


namespace NUMINAMATH_CALUDE_multiply_by_nine_l4037_403795

theorem multiply_by_nine (A B : ℕ) (h1 : 1 ≤ A ∧ A ≤ 9) (h2 : B ≤ 9) :
  (10 * A + B) * 9 = ((10 * A + B) - (A + 1)) * 10 + (10 - B) := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_nine_l4037_403795


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l4037_403746

-- Problem 1
theorem problem_1 : 123^2 - 124 * 122 = 1 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : (-2*a^2*b)^3 / (-a*b) * (1/2*a^2*b)^3 = a^11*b^5 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l4037_403746


namespace NUMINAMATH_CALUDE_price_difference_is_70_l4037_403700

-- Define the pricing structures and discount rates
def shop_x_base_price : ℝ := 1.25
def shop_y_base_price : ℝ := 2.75
def shop_x_discount_rate_80plus : ℝ := 0.10
def shop_y_bulk_price_80plus : ℝ := 2.00

-- Define the number of copies
def num_copies : ℕ := 80

-- Calculate the price for Shop X
def shop_x_price (copies : ℕ) : ℝ :=
  shop_x_base_price * copies * (1 - shop_x_discount_rate_80plus)

-- Calculate the price for Shop Y
def shop_y_price (copies : ℕ) : ℝ :=
  shop_y_bulk_price_80plus * copies

-- Theorem to prove
theorem price_difference_is_70 :
  shop_y_price num_copies - shop_x_price num_copies = 70 := by
  sorry


end NUMINAMATH_CALUDE_price_difference_is_70_l4037_403700


namespace NUMINAMATH_CALUDE_sin_2x_minus_pi_3_zeros_min_distance_l4037_403731

open Real

theorem sin_2x_minus_pi_3_zeros_min_distance (f : ℝ → ℝ) (h : ∀ x, f x = sin (2 * x - π / 3)) :
  ∀ a b : ℝ, a ≠ b → f a = 0 → f b = 0 → |a - b| ≥ π / 2 ∧ ∃ c d : ℝ, c ≠ d ∧ f c = 0 ∧ f d = 0 ∧ |c - d| = π / 2 :=
sorry

end NUMINAMATH_CALUDE_sin_2x_minus_pi_3_zeros_min_distance_l4037_403731


namespace NUMINAMATH_CALUDE_salary_restoration_l4037_403754

theorem salary_restoration (original_salary : ℝ) (reduced_salary : ℝ) : 
  reduced_salary = original_salary * (1 - 0.5) →
  reduced_salary * 2 = original_salary :=
by
  sorry

end NUMINAMATH_CALUDE_salary_restoration_l4037_403754


namespace NUMINAMATH_CALUDE_anthony_pencils_count_l4037_403760

/-- Given Anthony's initial pencils and Kathryn's gift, calculate Anthony's total pencils -/
def anthonyTotalPencils (initialPencils giftedPencils : ℕ) : ℕ :=
  initialPencils + giftedPencils

/-- Theorem: Anthony's total pencils is 65 given the initial conditions -/
theorem anthony_pencils_count :
  anthonyTotalPencils 9 56 = 65 := by
  sorry

end NUMINAMATH_CALUDE_anthony_pencils_count_l4037_403760


namespace NUMINAMATH_CALUDE_total_discount_calculation_l4037_403716

/-- Calculates the total discount percentage given a sale discount, coupon discount, and loyalty discount -/
theorem total_discount_calculation (original_price : ℝ) (sale_discount : ℝ) (coupon_discount : ℝ) (loyalty_discount : ℝ) :
  sale_discount = 1/3 →
  coupon_discount = 0.25 →
  loyalty_discount = 0.05 →
  let sale_price := original_price * (1 - sale_discount)
  let price_after_coupon := sale_price * (1 - coupon_discount)
  let final_price := price_after_coupon * (1 - loyalty_discount)
  (original_price - final_price) / original_price = 0.525 :=
by sorry

end NUMINAMATH_CALUDE_total_discount_calculation_l4037_403716


namespace NUMINAMATH_CALUDE_equation_solutions_l4037_403710

theorem equation_solutions :
  (∃ x : ℝ, x + 2*x = 12.6 ∧ x = 4.2) ∧
  (∃ x : ℝ, (1/4)*x + 1/2 = 3/5 ∧ x = 2/5) ∧
  (∃ x : ℝ, x + 1.3*x = 46 ∧ x = 20) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4037_403710


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l4037_403719

def k : ℕ := 2009^2 + 2^2009 - 3

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ := 2009^2 + 2^2009 - 3) :
  (k^2 + 2^k) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l4037_403719


namespace NUMINAMATH_CALUDE_original_price_is_360_l4037_403740

/-- The original price of a product satisfies two conditions:
    1. When sold at 75% of the original price, there's a loss of $12 per item.
    2. When sold at 90% of the original price, there's a profit of $42 per item. -/
theorem original_price_is_360 (price : ℝ) 
    (h1 : 0.75 * price + 12 = 0.9 * price - 42) : 
    price = 360 := by
  sorry

end NUMINAMATH_CALUDE_original_price_is_360_l4037_403740


namespace NUMINAMATH_CALUDE_spools_per_beret_l4037_403709

theorem spools_per_beret (total_spools : ℕ) (num_berets : ℕ) 
  (h1 : total_spools = 33) 
  (h2 : num_berets = 11) 
  (h3 : num_berets > 0) : 
  total_spools / num_berets = 3 := by
  sorry

end NUMINAMATH_CALUDE_spools_per_beret_l4037_403709


namespace NUMINAMATH_CALUDE_f_is_even_f_is_increasing_on_positive_l4037_403729

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- Theorem stating that f is an even function
theorem f_is_even : ∀ x : ℝ, f x = f (-x) := by sorry

-- Theorem stating that f is monotonically increasing on (0, +∞)
theorem f_is_increasing_on_positive : ∀ x y : ℝ, 0 < x → x < y → f x < f y := by sorry

end NUMINAMATH_CALUDE_f_is_even_f_is_increasing_on_positive_l4037_403729


namespace NUMINAMATH_CALUDE_parabola_directrix_l4037_403721

/-- The directrix of a parabola y² = 2px passing through (2, 2) -/
theorem parabola_directrix (p : ℝ) (h1 : p > 0) : 
  (∀ x y : ℝ, y^2 = 2*p*x → x = 2 → y = 2) → 
  (∃ k : ℝ, ∀ x y : ℝ, y^2 = 2*p*x → x = k) ∧ k = -1/2 :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l4037_403721


namespace NUMINAMATH_CALUDE_problem_solution_l4037_403756

-- Define the propositions
def p : Prop := ∃ k : ℤ, 0 = 2 * k
def q : Prop := ∃ k : ℤ, 3 = 2 * k

-- Theorem to prove
theorem problem_solution : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4037_403756
