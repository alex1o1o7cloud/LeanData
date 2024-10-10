import Mathlib

namespace power_of_power_l1335_133509

theorem power_of_power (a : ℝ) : (a^4)^3 = a^12 := by
  sorry

end power_of_power_l1335_133509


namespace hijk_is_square_l1335_133503

-- Define the points
variable (A B C D E F G H I J K : EuclideanSpace ℝ (Fin 2))

-- Define the squares
def is_square (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the midpoint
def is_midpoint (M P Q : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- State the theorem
theorem hijk_is_square 
  (h1 : is_square A B C D)
  (h2 : is_square D E F G)
  (h3 : A ≠ D ∧ B ≠ D ∧ C ≠ D ∧ E ≠ D ∧ F ≠ D ∧ G ≠ D)
  (h4 : is_midpoint H A G)
  (h5 : is_midpoint I G E)
  (h6 : is_midpoint J E C)
  (h7 : is_midpoint K C A) :
  is_square H I J K := by sorry

end hijk_is_square_l1335_133503


namespace variable_value_l1335_133569

theorem variable_value (w x v : ℝ) 
  (h1 : 2 / w + 2 / x = 2 / v) 
  (h2 : w * x = v) 
  (h3 : (w + x) / 2 = 0.5) : 
  v = 0.25 := by
sorry

end variable_value_l1335_133569


namespace circle_equation_l1335_133514

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (center : ℝ × ℝ), (center.1 - p.1)^2 + (center.2 - p.2)^2 = 4 ∧ 3 * center.1 - center.2 - 3 = 0}

-- Define points A and B
def point_A : ℝ × ℝ := (2, 5)
def point_B : ℝ × ℝ := (4, 3)

-- Theorem statement
theorem circle_equation : 
  (∀ p : ℝ × ℝ, p ∈ circle_C ↔ (p.1 - 2)^2 + (p.2 - 3)^2 = 4) ∧
  point_A ∈ circle_C ∧
  point_B ∈ circle_C :=
sorry

end circle_equation_l1335_133514


namespace job_completion_time_l1335_133533

theorem job_completion_time (p_rate q_rate : ℚ) (t : ℚ) : 
  p_rate = 1/4 →
  q_rate = 1/15 →
  t * (p_rate + q_rate) + 1/5 * p_rate = 1 →
  t = 3 :=
by sorry

end job_completion_time_l1335_133533


namespace real_number_line_bijection_l1335_133556

/-- A point on the number line -/
structure NumberLinePoint where
  position : ℝ

/-- The bijective function between real numbers and points on the number line -/
def realToPoint : ℝ → NumberLinePoint :=
  λ x ↦ ⟨x⟩

theorem real_number_line_bijection :
  Function.Bijective realToPoint :=
sorry

end real_number_line_bijection_l1335_133556


namespace complex_fraction_simplification_l1335_133541

theorem complex_fraction_simplification :
  let z : ℂ := (5 + 7*I) / (2 + 3*I)
  z = 31/13 - (1/13)*I := by sorry

end complex_fraction_simplification_l1335_133541


namespace quadratic_function_properties_l1335_133558

/-- A quadratic function passing through points (0,2) and (1,0) -/
def quadratic_function (x b c : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_function_properties :
  ∃ (b c : ℝ),
    (quadratic_function 0 b c = 2) ∧
    (quadratic_function 1 b c = 0) ∧
    (b = -3) ∧
    (c = 2) ∧
    (∀ x, quadratic_function x b c = (x - 3/2)^2 - 1/4) :=
by sorry

end quadratic_function_properties_l1335_133558


namespace pump_rate_calculation_l1335_133539

/-- Given two pumps operating for a total of 6 hours, with one pump rated at 250 gallons per hour
    and used for 3.5 hours, and a total volume pumped of 1325 gallons, the rate of the other pump
    is 180 gallons per hour. -/
theorem pump_rate_calculation (total_time : ℝ) (total_volume : ℝ) (pump2_rate : ℝ) (pump2_time : ℝ)
    (h1 : total_time = 6)
    (h2 : total_volume = 1325)
    (h3 : pump2_rate = 250)
    (h4 : pump2_time = 3.5) :
    (total_volume - pump2_rate * pump2_time) / (total_time - pump2_time) = 180 :=
by sorry

end pump_rate_calculation_l1335_133539


namespace triangle_properties_l1335_133506

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the properties and maximum area of the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : 4 * Real.cos t.C + Real.cos (2 * t.C) = 4 * Real.cos t.C * (Real.cos (t.C / 2))^2)
  (h2 : |t.b * Real.cos t.A - (1/2) * t.a * Real.cos t.B| = 2) : 
  t.C = π/3 ∧ 
  (∃ (S : ℝ), S ≤ 2 * Real.sqrt 3 ∧ 
    ∀ (S' : ℝ), S' = 1/2 * t.a * t.b * Real.sin t.C → S' ≤ S) := by
  sorry

end triangle_properties_l1335_133506


namespace special_polygon_diagonals_l1335_133555

/-- A polygon with 10 vertices, where 4 vertices lie on a straight line
    and the remaining 6 form a regular hexagon. -/
structure SpecialPolygon where
  vertices : Fin 10
  line_vertices : Fin 4
  hexagon_vertices : Fin 6

/-- The number of diagonals in the special polygon. -/
def num_diagonals (p : SpecialPolygon) : ℕ := 33

/-- Theorem stating that the number of diagonals in the special polygon is 33. -/
theorem special_polygon_diagonals (p : SpecialPolygon) : num_diagonals p = 33 := by
  sorry

end special_polygon_diagonals_l1335_133555


namespace arithmetic_sequence_common_difference_l1335_133571

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: If 2S_3 = 3S_2 + 6 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  seq.d = 2 := by
sorry

end arithmetic_sequence_common_difference_l1335_133571


namespace problem_1_problem_2_l1335_133567

theorem problem_1 : (1/2)⁻¹ + (Real.sqrt 2)^2 - 4 * |-(1/2)| = 2 := by sorry

theorem problem_2 (a : ℝ) (h : a = 2) : 
  (1 + 4 / (a - 1)) / ((a^2 + 6*a + 9) / (a^2 - a)) = 2/5 := by sorry

end problem_1_problem_2_l1335_133567


namespace intersection_implies_a_values_l1335_133593

-- Define sets A and B
def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {a, a^2 - 1}

-- State the theorem
theorem intersection_implies_a_values (a : ℝ) :
  (A ∩ B a = {1}) → (a = 1 ∨ a = Real.sqrt 2 ∨ a = -Real.sqrt 2) := by
  sorry

end intersection_implies_a_values_l1335_133593


namespace amy_hourly_rate_l1335_133585

/-- Calculates the hourly rate given total earnings, hours worked, and tips received. -/
def hourly_rate (total_earnings hours_worked tips : ℚ) : ℚ :=
  (total_earnings - tips) / hours_worked

/-- Proves that Amy's hourly rate is $2, given the conditions from the problem. -/
theorem amy_hourly_rate :
  let total_earnings : ℚ := 23
  let hours_worked : ℚ := 7
  let tips : ℚ := 9
  hourly_rate total_earnings hours_worked tips = 2 := by
  sorry

end amy_hourly_rate_l1335_133585


namespace largest_sum_of_squared_differences_l1335_133548

theorem largest_sum_of_squared_differences (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a > 0 ∧ b > 0 ∧ c > 0 →
  ∃ x y z : ℕ, (b + c - a = x^2) ∧ (c + a - b = y^2) ∧ (a + b - c = z^2) →
  a + b + c < 100 →
  a + b + c ≤ 91 :=
by sorry

end largest_sum_of_squared_differences_l1335_133548


namespace abfcde_perimeter_l1335_133540

/-- Represents a square with side length and perimeter -/
structure Square where
  side_length : ℝ
  perimeter : ℝ
  perimeter_eq : perimeter = 4 * side_length

/-- Represents the figure ABFCDE -/
structure ABFCDE where
  square : Square
  perimeter : ℝ

/-- The perimeter of ABFCDE is 80 inches, given a square with perimeter 64 inches -/
theorem abfcde_perimeter (s : Square) (fig : ABFCDE) 
  (h1 : s.perimeter = 64) 
  (h2 : fig.square = s) : 
  fig.perimeter = 80 :=
sorry

end abfcde_perimeter_l1335_133540


namespace base16_A987B_bits_bits_count_A987B_l1335_133595

def base16_to_decimal (n : String) : ℕ :=
  -- Implementation details omitted
  sorry

theorem base16_A987B_bits : 
  let decimal := base16_to_decimal "A987B"
  2^19 ≤ decimal ∧ decimal < 2^20 := by
  sorry

theorem bits_count_A987B : 
  (Nat.log 2 (base16_to_decimal "A987B") + 1 : ℕ) = 20 := by
  sorry

end base16_A987B_bits_bits_count_A987B_l1335_133595


namespace power_negative_two_of_five_l1335_133528

theorem power_negative_two_of_five : 5^(-2 : ℤ) = (1 : ℚ) / 25 := by sorry

end power_negative_two_of_five_l1335_133528


namespace subset_implies_a_equals_one_l1335_133551

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_implies_a_equals_one :
  ∀ a : ℝ, A a ⊆ B a → a = 1 := by
  sorry

end subset_implies_a_equals_one_l1335_133551


namespace spinster_cat_difference_l1335_133587

theorem spinster_cat_difference (spinster_count : ℕ) (cat_count : ℕ) : 
  spinster_count = 14 →
  (2 : ℚ) / 7 = spinster_count / cat_count →
  cat_count > spinster_count →
  cat_count - spinster_count = 35 := by
sorry

end spinster_cat_difference_l1335_133587


namespace factorial_equation_solutions_l1335_133553

theorem factorial_equation_solutions :
  ∀ x y z : ℕ, 2^x + 5^y + 63 = z! → ((x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6)) :=
by sorry

end factorial_equation_solutions_l1335_133553


namespace digit_equality_l1335_133505

theorem digit_equality (a b c d e f : ℕ) 
  (h_a : a < 10) (h_b : b < 10) (h_c : c < 10) 
  (h_d : d < 10) (h_e : e < 10) (h_f : f < 10) :
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) -
  (100000 * f + 10000 * d + 1000 * e + 100 * b + 10 * c + a) ∣ 271 →
  b = d ∧ c = e := by
sorry

end digit_equality_l1335_133505


namespace sufficient_not_necessary_condition_l1335_133589

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > |b| → a^3 > b^3) ∧
  (∃ a b : ℝ, a^3 > b^3 ∧ a ≤ |b|) :=
by sorry

end sufficient_not_necessary_condition_l1335_133589


namespace vector_perpendicularity_l1335_133538

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : ℝ :=
  v.x * w.x + v.y * w.y

/-- Two vectors are perpendicular if their dot product is zero -/
def is_perpendicular (v w : Vector2D) : Prop :=
  dot_product v w = 0

/-- Unit vector in positive x direction -/
def i : Vector2D :=
  ⟨1, 0⟩

/-- Unit vector in positive y direction -/
def j : Vector2D :=
  ⟨0, 1⟩

/-- Vector addition -/
def add_vectors (v w : Vector2D) : Vector2D :=
  ⟨v.x + w.x, v.y + w.y⟩

/-- Vector subtraction -/
def subtract_vectors (v w : Vector2D) : Vector2D :=
  ⟨v.x - w.x, v.y - w.y⟩

/-- Scalar multiplication of a vector -/
def scalar_mult (k : ℝ) (v : Vector2D) : Vector2D :=
  ⟨k * v.x, k * v.y⟩

theorem vector_perpendicularity :
  let a := scalar_mult 2 i
  let b := add_vectors i j
  is_perpendicular (subtract_vectors a b) b := by
  sorry

end vector_perpendicularity_l1335_133538


namespace function_value_l1335_133580

/-- Given a function f where f(2x + 3) is defined and f(29) = 170,
    prove that f(2x + 3) = 170 for all x -/
theorem function_value (f : ℝ → ℝ) (h : f 29 = 170) : ∀ x, f (2 * x + 3) = 170 := by
  sorry

end function_value_l1335_133580


namespace units_digit_of_99_factorial_l1335_133550

theorem units_digit_of_99_factorial (n : ℕ) : n = 99 → n.factorial % 10 = 0 := by sorry

end units_digit_of_99_factorial_l1335_133550


namespace polynomial_factorization_l1335_133531

theorem polynomial_factorization (a b : ℝ) : 
  (∀ x, x^2 - 3*x + a = (x - 5) * (x - b)) → (a = -10 ∧ b = -2) := by
sorry

end polynomial_factorization_l1335_133531


namespace two_discount_equation_l1335_133500

/-- Proves the equation for a product's price after two consecutive discounts -/
theorem two_discount_equation (original_price final_price x : ℝ) :
  original_price = 400 →
  final_price = 225 →
  0 < x →
  x < 1 →
  original_price * (1 - x)^2 = final_price :=
by sorry

end two_discount_equation_l1335_133500


namespace largest_number_l1335_133566

theorem largest_number (a b c d e : ℝ) : 
  a = 17231 + 1 / 3251 →
  b = 17231 - 1 / 3251 →
  c = 17231 * (1 / 3251) →
  d = 17231 / (1 / 3251) →
  e = 17231.3251 →
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end largest_number_l1335_133566


namespace necessary_but_not_sufficient_condition_l1335_133562

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x > Real.exp 1 → x > 1) ∧ ∃ x, x > 1 ∧ x ≤ Real.exp 1 := by sorry

end necessary_but_not_sufficient_condition_l1335_133562


namespace arithmetic_sequence_middle_term_l1335_133513

/-- An arithmetic sequence with 5 terms -/
structure ArithmeticSequence5 where
  a : ℝ  -- first term
  b : ℝ  -- second term
  c : ℝ  -- third term (middle term)
  d : ℝ  -- fourth term
  e : ℝ  -- fifth term
  is_arithmetic : ∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r ∧ e = d + r

/-- The theorem stating that in an arithmetic sequence with first term 20, 
    last term 50, and middle term y, the value of y is 35 -/
theorem arithmetic_sequence_middle_term 
  (seq : ArithmeticSequence5) 
  (h1 : seq.a = 20) 
  (h2 : seq.e = 50) : 
  seq.c = 35 := by
sorry

end arithmetic_sequence_middle_term_l1335_133513


namespace joan_video_game_spending_l1335_133508

/-- The cost of the basketball game Joan purchased -/
def basketball_cost : ℚ := 5.2

/-- The cost of the racing game Joan purchased -/
def racing_cost : ℚ := 4.23

/-- The total amount Joan spent on video games -/
def total_spent : ℚ := basketball_cost + racing_cost

/-- Theorem stating that the total amount Joan spent on video games is $9.43 -/
theorem joan_video_game_spending :
  total_spent = 9.43 := by sorry

end joan_video_game_spending_l1335_133508


namespace problem_solution_l1335_133565

/-- The number of positive factors of n -/
def num_factors (n : ℕ) : ℕ := sorry

theorem problem_solution (y : ℕ) 
  (h1 : num_factors y = 18) 
  (h2 : 14 ∣ y) 
  (h3 : 18 ∣ y) : 
  y = 252 := by sorry

end problem_solution_l1335_133565


namespace age_difference_l1335_133526

theorem age_difference (alice_age carol_age betty_age : ℕ) : 
  carol_age = 5 * alice_age →
  carol_age = 2 * betty_age →
  betty_age = 6 →
  carol_age - alice_age = 10 := by
sorry

end age_difference_l1335_133526


namespace least_exponent_sum_for_400_l1335_133578

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def sum_of_distinct_powers_of_two (n : ℕ) (powers : List ℕ) : Prop :=
  (powers.length ≥ 2) ∧
  (∀ p ∈ powers, is_power_of_two p) ∧
  (powers.sum = n) ∧
  (powers.toFinset.card = powers.length)

def exponent_sum (powers : List ℕ) : ℕ :=
  (powers.map (λ p => (Nat.log p 2))).sum

theorem least_exponent_sum_for_400 :
  ∀ powers : List ℕ,
    sum_of_distinct_powers_of_two 400 powers →
    exponent_sum powers ≥ 19 :=
sorry

end least_exponent_sum_for_400_l1335_133578


namespace range_of_a_l1335_133594

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → |x + 1/x| > |a - 2| + 1) → 
  1 < a ∧ a < 3 := by
sorry

end range_of_a_l1335_133594


namespace decimal_to_fraction_l1335_133549

theorem decimal_to_fraction : (3.675 : ℚ) = 147 / 40 := by
  sorry

end decimal_to_fraction_l1335_133549


namespace remainder_theorem_l1335_133547

theorem remainder_theorem (x y q r : ℕ) (h1 : x = q * y + r) (h2 : r < y) :
  (x - 3 * q * y) % y = r := by
  sorry

end remainder_theorem_l1335_133547


namespace function_identity_l1335_133597

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ (m n : ℕ+), f (f m ^ 2 + 2 * f n ^ 2) = m ^ 2 + 2 * n ^ 2) : 
  ∀ (n : ℕ+), f n = n := by
  sorry

end function_identity_l1335_133597


namespace tan_theta_plus_pi_sixth_l1335_133518

theorem tan_theta_plus_pi_sixth (θ : Real) 
  (h1 : Real.sqrt 2 * Real.sin (θ - Real.pi/4) * Real.cos (Real.pi + θ) = Real.cos (2*θ))
  (h2 : Real.sin θ ≠ 0) : 
  Real.tan (θ + Real.pi/6) = 2 + Real.sqrt 3 := by
  sorry

end tan_theta_plus_pi_sixth_l1335_133518


namespace yellow_balls_count_l1335_133515

/-- Proves the number of yellow balls in a box given specific conditions -/
theorem yellow_balls_count (red yellow green : ℕ) : 
  red + yellow + green = 68 →
  yellow = 2 * red →
  3 * green = 4 * yellow →
  yellow = 24 := by
  sorry

end yellow_balls_count_l1335_133515


namespace two_lines_condition_l1335_133516

theorem two_lines_condition (m : ℝ) : 
  (∃ (a b c d : ℝ), ∀ (x y : ℝ), 
    (x^2 - m*y^2 + 2*x + 2*y = 0) ↔ ((a*x + b*y + c = 0) ∧ (a*x + b*y + d = 0))) 
  → m = 1 := by
sorry

end two_lines_condition_l1335_133516


namespace smallest_n_for_roots_of_unity_l1335_133501

/-- The polynomial z^5 - z^3 + 1 -/
def f (z : ℂ) : ℂ := z^5 - z^3 + 1

/-- n-th roots of unity -/
def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop := z^n = 1

/-- All roots of f are n-th roots of unity -/
def all_roots_are_nth_roots_of_unity (n : ℕ) : Prop :=
  ∀ z : ℂ, f z = 0 → is_nth_root_of_unity z n

theorem smallest_n_for_roots_of_unity :
  (∃ n : ℕ, n > 0 ∧ all_roots_are_nth_roots_of_unity n) ∧
  (∀ m : ℕ, m > 0 ∧ all_roots_are_nth_roots_of_unity m → m ≥ 30) :=
sorry

end smallest_n_for_roots_of_unity_l1335_133501


namespace third_root_of_cubic_l1335_133545

theorem third_root_of_cubic (a b : ℚ) (h : a ≠ 0) :
  (∃ x : ℚ, a * x^3 - (3*a - b) * x^2 + 2*(a + b) * x - (6 - 2*a) = 0) ∧
  (a * 1^3 - (3*a - b) * 1^2 + 2*(a + b) * 1 - (6 - 2*a) = 0) ∧
  (a * (-3)^3 - (3*a - b) * (-3)^2 + 2*(a + b) * (-3) - (6 - 2*a) = 0) →
  ∃ x : ℚ, x ≠ 1 ∧ x ≠ -3 ∧ a * x^3 - (3*a - b) * x^2 + 2*(a + b) * x - (6 - 2*a) = 0 ∧ x = 322/21 :=
by sorry

end third_root_of_cubic_l1335_133545


namespace union_of_M_and_N_l1335_133530

def M : Set Int := {-1, 0, 1}
def N : Set Int := {0, 1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} := by sorry

end union_of_M_and_N_l1335_133530


namespace union_implies_a_zero_l1335_133521

theorem union_implies_a_zero (a : ℝ) : 
  let A : Set ℝ := {1, a^2}
  let B : Set ℝ := {a, -1}
  A ∪ B = {-1, a, 1} → a = 0 := by
sorry

end union_implies_a_zero_l1335_133521


namespace sum_of_b_and_c_is_eleven_l1335_133577

theorem sum_of_b_and_c_is_eleven
  (a b c : ℕ+)
  (ha : a ≠ 1)
  (hb : b ≤ 9)
  (hc : c ≤ 9)
  (hbc : b ≠ c)
  (heq : (10 * a + b) * (10 * a + c) = 100 * a^2 + 110 * a + b * c) :
  b + c = 11 := by
  sorry

end sum_of_b_and_c_is_eleven_l1335_133577


namespace age_difference_l1335_133599

/-- Given three people a, b, and c, where b is twice as old as c, 
    the total of their ages is 12, and b is 4 years old, 
    prove that a is 2 years older than b. -/
theorem age_difference (a b c : ℕ) : 
  b = 2 * c →
  a + b + c = 12 →
  b = 4 →
  a = b + 2 := by
sorry

end age_difference_l1335_133599


namespace edward_tickets_l1335_133520

/-- The number of tickets Edward spent at the 'dunk a clown' booth -/
def spent_tickets : ℕ := 23

/-- The cost of each ride in tickets -/
def ride_cost : ℕ := 7

/-- The number of rides Edward could have gone on with the remaining tickets -/
def possible_rides : ℕ := 8

/-- The total number of tickets Edward bought at the state fair -/
def total_tickets : ℕ := spent_tickets + ride_cost * possible_rides

theorem edward_tickets : total_tickets = 79 := by sorry

end edward_tickets_l1335_133520


namespace cube_root_unity_sum_l1335_133529

/-- Given a nonreal cube root of unity ω, prove that (1 - ω + ω^2)^6 + (1 + ω - ω^2)^6 = 128 -/
theorem cube_root_unity_sum (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω ≠ 1) :
  (1 - ω + ω^2)^6 + (1 + ω - ω^2)^6 = 128 := by
  sorry

end cube_root_unity_sum_l1335_133529


namespace intersection_E_F_l1335_133544

open Set Real

def E : Set ℝ := {θ | cos θ < sin θ ∧ 0 ≤ θ ∧ θ ≤ 2 * π}
def F : Set ℝ := {θ | tan θ < sin θ}

theorem intersection_E_F : E ∩ F = Ioo (π / 2) π := by
  sorry

end intersection_E_F_l1335_133544


namespace pen_price_calculation_l1335_133507

theorem pen_price_calculation (num_pens num_pencils total_cost pencil_avg_price : ℝ) 
  (h1 : num_pens = 30)
  (h2 : num_pencils = 75)
  (h3 : total_cost = 750)
  (h4 : pencil_avg_price = 2) : 
  (total_cost - num_pencils * pencil_avg_price) / num_pens = 20 := by
  sorry

end pen_price_calculation_l1335_133507


namespace floor_equation_solutions_l1335_133575

theorem floor_equation_solutions :
  let S := {x : ℤ | ⌊(x : ℚ) / 2⌋ + ⌊(x : ℚ) / 4⌋ = x}
  S = {0, -3, -2, -5} := by
  sorry

end floor_equation_solutions_l1335_133575


namespace tree_planting_l1335_133559

theorem tree_planting (path_length : ℕ) (tree_distance : ℕ) (total_trees : ℕ) : 
  path_length = 50 →
  tree_distance = 2 →
  total_trees = 2 * (path_length / tree_distance + 1) →
  total_trees = 52 := by
sorry

end tree_planting_l1335_133559


namespace max_y_value_l1335_133557

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -2) : 
  ∃ (max_y : ℤ), (∀ (y' : ℤ), ∃ (x' : ℤ), x' * y' + 3 * x' + 2 * y' = -2 → y' ≤ max_y) ∧ max_y = 1 := by
  sorry

end max_y_value_l1335_133557


namespace f_increasing_range_l1335_133523

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * (x - 1)^2 + 1 else (a + 3) * x + 4 * a

theorem f_increasing_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔
  a ∈ Set.Icc (-2/5 : ℝ) 0 ∧ a ≠ 0 :=
sorry

end f_increasing_range_l1335_133523


namespace rectangular_field_perimeter_l1335_133543

theorem rectangular_field_perimeter : 
  ∀ (width length perimeter : ℝ),
  width = 60 →
  length = (7 / 5) * width →
  perimeter = 2 * (length + width) →
  perimeter = 288 := by
  sorry

end rectangular_field_perimeter_l1335_133543


namespace fraction_equality_l1335_133512

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (2 * x - 3 * y) / (x + 2 * y) = 3) : 
  (x - 2 * y) / (2 * x + 3 * y) = 11 / 15 := by
  sorry

end fraction_equality_l1335_133512


namespace last_locker_exists_l1335_133581

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Represents a corridor with a given number of lockers -/
def Corridor (n : Nat) := Fin n → LockerState

/-- Toggles the state of a locker -/
def toggleLocker (state : LockerState) : LockerState :=
  match state with
  | LockerState.Open => LockerState.Closed
  | LockerState.Closed => LockerState.Open

/-- Represents a single pass of toggling lockers with a given step size -/
def togglePass (c : Corridor 512) (step : Nat) : Corridor 512 :=
  sorry

/-- Represents the full toggling process until all lockers are open -/
def fullToggleProcess (c : Corridor 512) : Corridor 512 :=
  sorry

/-- Theorem stating that there exists a last locker to be opened -/
theorem last_locker_exists :
  ∃ (last : Fin 512), 
    ∀ (c : Corridor 512), 
      (fullToggleProcess c last = LockerState.Open) ∧ 
      (∀ (i : Fin 512), i.val > last.val → fullToggleProcess c i = LockerState.Open) :=
sorry

end last_locker_exists_l1335_133581


namespace sign_sum_theorem_l1335_133504

theorem sign_sum_theorem (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) :
  let sign (x : ℝ) := x / |x|
  let expr := sign p + sign q + sign r + sign (p * q * r) + sign (p * q)
  expr = 5 ∨ expr = 1 ∨ expr = -1 :=
by sorry

end sign_sum_theorem_l1335_133504


namespace rosa_phone_calls_l1335_133561

theorem rosa_phone_calls (last_week : ℝ) (this_week : ℝ) (total : ℝ) 
  (h1 : last_week = 10.2)
  (h2 : this_week = 8.6)
  (h3 : total = last_week + this_week) :
  total = 18.8 := by
sorry

end rosa_phone_calls_l1335_133561


namespace solution_to_linear_equation_l1335_133590

theorem solution_to_linear_equation :
  ∃ (x y : ℝ), 2 * x + y = 6 ∧ x = 2 ∧ y = 2 := by
  sorry

end solution_to_linear_equation_l1335_133590


namespace abc_sum_mod_five_l1335_133573

theorem abc_sum_mod_five (a b c : ℕ) : 
  0 < a ∧ a < 5 ∧ 
  0 < b ∧ b < 5 ∧ 
  0 < c ∧ c < 5 ∧ 
  (a * b * c) % 5 = 1 ∧ 
  (4 * c) % 5 = 3 ∧ 
  (3 * b) % 5 = (2 + b) % 5 → 
  (a + b + c) % 5 = 3 := by
sorry

end abc_sum_mod_five_l1335_133573


namespace combination_problem_l1335_133596

theorem combination_problem (n : ℕ) 
  (h : Nat.choose (n + 1) 7 - Nat.choose n 7 = Nat.choose n 8) : n = 14 := by
  sorry

end combination_problem_l1335_133596


namespace min_value_quadratic_form_l1335_133588

theorem min_value_quadratic_form (x y : ℝ) : x^2 - x*y + y^2 ≥ 0 ∧ 
  (x^2 - x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) := by
  sorry

end min_value_quadratic_form_l1335_133588


namespace scarlett_fruit_salad_berries_l1335_133560

/-- The weight of berries in Scarlett's fruit salad -/
def weight_of_berries (total_weight melon_weight : ℚ) : ℚ :=
  total_weight - melon_weight

/-- Proof that the weight of berries in Scarlett's fruit salad is 0.38 pounds -/
theorem scarlett_fruit_salad_berries :
  weight_of_berries (63/100) (1/4) = 38/100 := by
  sorry

end scarlett_fruit_salad_berries_l1335_133560


namespace another_square_possible_l1335_133592

/-- Represents a grid of cells -/
structure Grid :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a square cut out from the grid -/
structure Square :=
  (size : Nat)

/-- Function to check if another square can be cut out -/
def can_cut_another_square (g : Grid) (cut_squares : List Square) : Prop :=
  ∃ (new_square : Square), 
    new_square.size = 2 ∧ 
    (g.rows ≥ 2 ∧ g.cols ≥ 2) ∧
    (List.length cut_squares < (g.rows / 2) * (g.cols / 2))

/-- Theorem statement -/
theorem another_square_possible (g : Grid) (cut_squares : List Square) :
  g.rows = 29 ∧ g.cols = 29 ∧ 
  List.length cut_squares = 99 ∧
  ∀ s ∈ cut_squares, s.size = 2
  →
  can_cut_another_square g cut_squares :=
sorry

end another_square_possible_l1335_133592


namespace function_properties_l1335_133517

noncomputable def f (m n x : ℝ) : ℝ := m * x + n / x

theorem function_properties (m n : ℝ) :
  (∃ a, f m n 1 = a ∧ 3 + a - 8 = 0) →
  (m = 1 ∧ n = 4) ∧
  (∀ x, x < -2 → (deriv (f m n)) x > 0) ∧
  (∀ x, -2 < x → x < 0 → (deriv (f m n)) x < 0) ∧
  (∀ x, 0 < x → x < 2 → (deriv (f m n)) x < 0) ∧
  (∀ x, 2 < x → (deriv (f m n)) x > 0) ∧
  (∀ x, x ≠ 0 → (deriv (f m n)) x < 1) ∧
  (∀ α, (0 ≤ α ∧ α < π/4) ∨ (π/2 < α ∧ α < π) ↔ 
    ∃ x, x ≠ 0 ∧ Real.tan α = (deriv (f m n)) x) :=
by sorry

end function_properties_l1335_133517


namespace hyperbola_asymptotes_l1335_133511

/-- A hyperbola with equation mx^2 - y^2 = 1 and asymptotes y = ±3x has m = 9 -/
theorem hyperbola_asymptotes (m : ℝ) : 
  (∀ x y : ℝ, m * x^2 - y^2 = 1) → 
  (∀ x : ℝ, (∃ y : ℝ, y = 3 * x ∨ y = -3 * x) → m * x^2 - y^2 = 0) → 
  m = 9 := by
sorry

end hyperbola_asymptotes_l1335_133511


namespace complex_arithmetic_equality_l1335_133522

theorem complex_arithmetic_equality : 
  908 * 501 - (731 * 1389 - (547 * 236 + 842 * 731 - 495 * 361)) = 5448 := by
  sorry

end complex_arithmetic_equality_l1335_133522


namespace neznaika_claims_l1335_133534

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def divisibility_claims (n : ℕ) : List Bool :=
  [n % 3 = 0, n % 4 = 0, n % 5 = 0, n % 9 = 0, n % 10 = 0, n % 15 = 0, n % 18 = 0, n % 30 = 0]

theorem neznaika_claims (n : ℕ) : 
  is_two_digit n → (divisibility_claims n).count false = 4 → n = 36 ∨ n = 45 ∨ n = 72 := by
  sorry

end neznaika_claims_l1335_133534


namespace customers_in_other_countries_l1335_133524

/-- Represents the number of customers in different regions --/
structure CustomerDistribution where
  total : Nat
  usa : Nat
  canada : Nat

/-- Calculates the number of customers in other countries --/
def customersInOtherCountries (d : CustomerDistribution) : Nat :=
  d.total - (d.usa + d.canada)

/-- Theorem stating the number of customers in other countries --/
theorem customers_in_other_countries :
  let d : CustomerDistribution := {
    total := 7422,
    usa := 723,
    canada := 1297
  }
  customersInOtherCountries d = 5402 := by
  sorry

#eval customersInOtherCountries {total := 7422, usa := 723, canada := 1297}

end customers_in_other_countries_l1335_133524


namespace majka_numbers_unique_l1335_133570

/-- A three-digit number with alternating odd-even-odd digits -/
structure FunnyNumber :=
  (hundreds : Nat) (tens : Nat) (ones : Nat)
  (hundreds_odd : Odd hundreds)
  (tens_even : Even tens)
  (ones_odd : Odd ones)
  (is_three_digit : 100 ≤ hundreds * 100 + tens * 10 + ones ∧ hundreds * 100 + tens * 10 + ones < 1000)

/-- A three-digit number with alternating even-odd-even digits -/
structure CheerfulNumber :=
  (hundreds : Nat) (tens : Nat) (ones : Nat)
  (hundreds_even : Even hundreds)
  (tens_odd : Odd tens)
  (ones_even : Even ones)
  (is_three_digit : 100 ≤ hundreds * 100 + tens * 10 + ones ∧ hundreds * 100 + tens * 10 + ones < 1000)

/-- Convert a FunnyNumber to a natural number -/
def FunnyNumber.toNat (n : FunnyNumber) : Nat :=
  n.hundreds * 100 + n.tens * 10 + n.ones

/-- Convert a CheerfulNumber to a natural number -/
def CheerfulNumber.toNat (n : CheerfulNumber) : Nat :=
  n.hundreds * 100 + n.tens * 10 + n.ones

/-- The main theorem stating the unique solution to Majka's problem -/
theorem majka_numbers_unique (f : FunnyNumber) (c : CheerfulNumber) 
  (sum_eq : f.toNat + c.toNat = 1617)
  (product_ends_40 : (f.toNat * c.toNat) % 100 = 40)
  (all_digits_different : f.hundreds ≠ f.tens ∧ f.hundreds ≠ f.ones ∧ f.tens ≠ f.ones ∧
                          c.hundreds ≠ c.tens ∧ c.hundreds ≠ c.ones ∧ c.tens ≠ c.ones ∧
                          f.hundreds ≠ c.hundreds ∧ f.hundreds ≠ c.tens ∧ f.hundreds ≠ c.ones ∧
                          f.tens ≠ c.hundreds ∧ f.tens ≠ c.tens ∧ f.tens ≠ c.ones ∧
                          f.ones ≠ c.hundreds ∧ f.ones ≠ c.tens ∧ f.ones ≠ c.ones)
  (all_digits_nonzero : f.hundreds ≠ 0 ∧ f.tens ≠ 0 ∧ f.ones ≠ 0 ∧
                        c.hundreds ≠ 0 ∧ c.tens ≠ 0 ∧ c.ones ≠ 0) :
  f.toNat = 945 ∧ c.toNat = 672 ∧ f.toNat * c.toNat = 635040 := by
  sorry


end majka_numbers_unique_l1335_133570


namespace hexagon_area_sum_l1335_133527

theorem hexagon_area_sum (u v : ℤ) (hu : 0 < u) (hv : 0 < v) (huv : v < u) :
  let A : ℤ × ℤ := (u, v)
  let B : ℤ × ℤ := (v, u)
  let C : ℤ × ℤ := (-v, u)
  let D : ℤ × ℤ := (-v, -u)
  let E : ℤ × ℤ := (v, -u)
  let F : ℤ × ℤ := (-u, -v)
  let hexagon_area := 8 * u * v + |u^2 - u*v - v^2|
  hexagon_area = 802 → u + v = 27 := by
sorry

end hexagon_area_sum_l1335_133527


namespace solve_watermelon_problem_l1335_133532

def watermelon_problem (michael_weight : ℝ) (clay_multiplier : ℝ) (john_fraction : ℝ) : Prop :=
  let clay_weight := michael_weight * clay_multiplier
  let john_weight := clay_weight * john_fraction
  john_weight = 12

theorem solve_watermelon_problem :
  watermelon_problem 8 3 (1/2) :=
by
  sorry

end solve_watermelon_problem_l1335_133532


namespace bills_toddler_count_l1335_133563

/-- The number of toddlers Bill thinks he counted -/
def billsCount (actualCount doubleCount missedCount : ℕ) : ℕ :=
  actualCount + doubleCount - missedCount

/-- Theorem stating that Bill thinks he counted 26 toddlers -/
theorem bills_toddler_count :
  let actualCount : ℕ := 21
  let doubleCount : ℕ := 8
  let missedCount : ℕ := 3
  billsCount actualCount doubleCount missedCount = 26 := by
  sorry

end bills_toddler_count_l1335_133563


namespace original_purchase_price_l1335_133546

/-- Represents the original purchase price of the pants -/
def purchase_price : ℝ := sorry

/-- Represents the original selling price of the pants -/
def selling_price : ℝ := sorry

/-- The markup is 25% of the selling price -/
axiom markup_condition : selling_price = purchase_price + 0.25 * selling_price

/-- The new selling price after 20% decrease -/
def new_selling_price : ℝ := 0.8 * selling_price

/-- The gross profit is $5.40 -/
axiom gross_profit_condition : new_selling_price - purchase_price = 5.40

/-- Theorem stating that the original purchase price is $81 -/
theorem original_purchase_price : purchase_price = 81 := by sorry

end original_purchase_price_l1335_133546


namespace find_divisor_l1335_133598

theorem find_divisor : ∃ (d : ℕ), d = 675 ∧ 
  (9679 - 4) % d = 0 ∧ 
  ∀ (k : ℕ), 0 < k → k < 4 → (9679 - k) % d ≠ 0 := by
  sorry

end find_divisor_l1335_133598


namespace increase_by_percentage_increase_80_by_150_percent_l1335_133564

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_80_by_150_percent :
  80 * (1 + 150 / 100) = 200 := by sorry

end increase_by_percentage_increase_80_by_150_percent_l1335_133564


namespace average_equals_5x_minus_9_l1335_133535

theorem average_equals_5x_minus_9 (x : ℚ) : 
  (1/3 : ℚ) * ((x + 8) + (8*x + 3) + (3*x + 9)) = 5*x - 9 → x = 47/3 := by
sorry

end average_equals_5x_minus_9_l1335_133535


namespace completing_square_equivalence_l1335_133576

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 :=
by sorry

end completing_square_equivalence_l1335_133576


namespace polynomial_coefficient_sum_l1335_133510

theorem polynomial_coefficient_sum (a b c d : ℤ) :
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 - 2*x^3 + 3*x^2 + 4*x - 10) →
  a + b + c + d = 1 := by
sorry

end polynomial_coefficient_sum_l1335_133510


namespace fraction_order_l1335_133584

theorem fraction_order : 
  (21 : ℚ) / 17 < 18 / 13 ∧ 18 / 13 < 16 / 11 := by
  sorry

end fraction_order_l1335_133584


namespace sufficient_not_necessary_l1335_133586

theorem sufficient_not_necessary (a : ℝ) :
  (a < -1 → ∃ x₀ : ℝ, a * Real.cos x₀ + 1 < 0) ∧
  (∃ a : ℝ, a ≥ -1 ∧ ∃ x₀ : ℝ, a * Real.cos x₀ + 1 < 0) :=
sorry

end sufficient_not_necessary_l1335_133586


namespace max_value_expression_l1335_133572

theorem max_value_expression (a b c d : ℝ) 
  (ha : -6.5 ≤ a ∧ a ≤ 6.5)
  (hb : -6.5 ≤ b ∧ b ≤ 6.5)
  (hc : -6.5 ≤ c ∧ c ≤ 6.5)
  (hd : -6.5 ≤ d ∧ d ≤ 6.5) :
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 182 :=
sorry

end max_value_expression_l1335_133572


namespace horner_operations_for_f_l1335_133536

def f (x : ℝ) := 6 * x^6 + 5

def horner_operations (p : ℝ → ℝ) (degree : ℕ) : ℕ × ℕ :=
  (degree, degree)

theorem horner_operations_for_f :
  horner_operations f 6 = (6, 6) := by sorry

end horner_operations_for_f_l1335_133536


namespace quadratic_roots_property_l1335_133552

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  (x₁^2 + x₁ - 3 = 0) → 
  (x₂^2 + x₂ - 3 = 0) → 
  (x₁^3 - 4*x₂^2 + 19 = 0) := by
  sorry

end quadratic_roots_property_l1335_133552


namespace inequality_equivalence_l1335_133574

theorem inequality_equivalence (x y : ℝ) :
  (2 * y - 3 * x < Real.sqrt (9 * x^2 + 16)) ↔
  ((y < 4 * x ∧ x ≥ 0) ∨ (y < -x ∧ x < 0)) := by
  sorry

end inequality_equivalence_l1335_133574


namespace rectangle_cylinder_volume_ratio_l1335_133579

theorem rectangle_cylinder_volume_ratio :
  let rectangle_width : ℝ := 7
  let rectangle_height : ℝ := 9
  let cylinder1_height : ℝ := rectangle_height
  let cylinder1_circumference : ℝ := rectangle_width
  let cylinder2_height : ℝ := rectangle_width
  let cylinder2_circumference : ℝ := rectangle_height
  let cylinder1_volume : ℝ := (cylinder1_circumference ^ 2 * cylinder1_height) / (4 * Real.pi)
  let cylinder2_volume : ℝ := (cylinder2_circumference ^ 2 * cylinder2_height) / (4 * Real.pi)
  let larger_volume : ℝ := max cylinder1_volume cylinder2_volume
  let smaller_volume : ℝ := min cylinder1_volume cylinder2_volume
  larger_volume / smaller_volume = 1 / 7 := by
sorry

end rectangle_cylinder_volume_ratio_l1335_133579


namespace solution_set_eq_open_interval_l1335_133583

-- Define the logarithm function with base 10
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the solution set
def solution_set : Set ℝ := {x | log10 (x - 1) < 2}

-- State the theorem
theorem solution_set_eq_open_interval :
  solution_set = Set.Ioo 1 101 := by sorry

end solution_set_eq_open_interval_l1335_133583


namespace intersection_complement_equality_l1335_133537

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality : B ∩ (U \ A) = {3, 4} := by sorry

end intersection_complement_equality_l1335_133537


namespace factorization_equality_l1335_133568

theorem factorization_equality (z : ℝ) :
  70 * z^20 + 154 * z^40 + 224 * z^60 = 14 * z^20 * (5 + 11 * z^20 + 16 * z^40) := by
  sorry

end factorization_equality_l1335_133568


namespace abs_sum_inequality_l1335_133519

theorem abs_sum_inequality (k : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 3| > k) ↔ k < 4 :=
sorry

end abs_sum_inequality_l1335_133519


namespace windows_preference_count_survey_results_l1335_133542

/-- Represents the survey results of college students' computer brand preferences --/
structure SurveyResults where
  total : ℕ
  mac_preference : ℕ
  no_preference : ℕ
  both_preference : ℕ
  windows_preference : ℕ

/-- Theorem stating the number of students preferring Windows to Mac --/
theorem windows_preference_count (survey : SurveyResults) : 
  survey.total = 210 →
  survey.mac_preference = 60 →
  survey.no_preference = 90 →
  survey.both_preference = survey.mac_preference / 3 →
  survey.windows_preference = 40 := by
  sorry

/-- Main theorem proving the survey results --/
theorem survey_results : ∃ (survey : SurveyResults), 
  survey.total = 210 ∧
  survey.mac_preference = 60 ∧
  survey.no_preference = 90 ∧
  survey.both_preference = survey.mac_preference / 3 ∧
  survey.windows_preference = 40 := by
  sorry

end windows_preference_count_survey_results_l1335_133542


namespace logarithm_expression_equality_l1335_133525

theorem logarithm_expression_equality : 
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 - 5 ^ (Real.log 3 / Real.log 5) = -1 := by
  sorry

end logarithm_expression_equality_l1335_133525


namespace maximum_spent_l1335_133582

/-- Represents the denominations of money in fen -/
inductive Denomination
  | Yuan100
  | Yuan50
  | Yuan20
  | Yuan10
  | Yuan5
  | Yuan1
  | Jiao5
  | Jiao1
  | Fen5
  | Fen2
  | Fen1

/-- Converts a denomination to its value in fen -/
def denominationToFen (d : Denomination) : ℕ :=
  match d with
  | .Yuan100 => 10000
  | .Yuan50  => 5000
  | .Yuan20  => 2000
  | .Yuan10  => 1000
  | .Yuan5   => 500
  | .Yuan1   => 100
  | .Jiao5   => 50
  | .Jiao1   => 10
  | .Fen5    => 5
  | .Fen2    => 2
  | .Fen1    => 1

/-- Represents a set of banknotes or coins -/
structure Change where
  denominations : List Denomination
  distinct : denominations.Nodup

/-- The problem statement -/
theorem maximum_spent (initialAmount : ℕ) 
  (banknotes : Change) 
  (coins : Change) :
  (initialAmount = 10000) →
  (banknotes.denominations.length = 4) →
  (coins.denominations.length = 4) →
  (∀ d ∈ banknotes.denominations, denominationToFen d > 100) →
  (∀ d ∈ coins.denominations, denominationToFen d < 100) →
  ((banknotes.denominations.map denominationToFen).sum % 300 = 0) →
  ((coins.denominations.map denominationToFen).sum % 7 = 0) →
  (initialAmount - (banknotes.denominations.map denominationToFen).sum - 
   (coins.denominations.map denominationToFen).sum = 6337) :=
by sorry

end maximum_spent_l1335_133582


namespace min_value_expression_l1335_133591

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 1) (hab : a + b = 1) :
  ((2 * a + b) / (a * b) - 3) * c + Real.sqrt 2 / (c - 1) ≥ 4 + 2 * Real.sqrt 2 ∧
  (((2 * a + b) / (a * b) - 3) * c + Real.sqrt 2 / (c - 1) = 4 + 2 * Real.sqrt 2 ↔ 
    a = Real.sqrt 2 - 1 ∧ b = 2 - Real.sqrt 2 ∧ c = 1 + Real.sqrt 2 / 2) :=
by sorry

#check min_value_expression

end min_value_expression_l1335_133591


namespace stockholm_uppsala_distance_l1335_133502

/-- The scale factor of the map, representing km per cm -/
def scale : ℝ := 10

/-- The distance between Stockholm and Uppsala on the map in cm -/
def map_distance : ℝ := 35

/-- The actual distance between Stockholm and Uppsala in km -/
def actual_distance : ℝ := map_distance * scale

theorem stockholm_uppsala_distance : actual_distance = 350 := by
  sorry

end stockholm_uppsala_distance_l1335_133502


namespace geometric_sequence_specific_form_l1335_133554

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem states that for a geometric sequence satisfying certain conditions,
    its general term has a specific form. -/
theorem geometric_sequence_specific_form (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) 
    (h_a2 : a 2 = 1)
    (h_relation : a 3 * a 5 = 2 * a 7) :
    ∀ n : ℕ, a n = 1 / 2^(n - 2) := by
  sorry

end geometric_sequence_specific_form_l1335_133554
