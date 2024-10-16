import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2166_216661

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Theorem: In a geometric sequence, if a_3 * a_4 = 6, then a_2 * a_5 = 6 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_prod : a 3 * a 4 = 6) : a 2 * a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2166_216661


namespace NUMINAMATH_CALUDE_identify_roles_l2166_216662

-- Define the possible roles
inductive Role
  | Knight
  | Liar
  | Normal

-- Define the statements made by each individual
def statement_A : Prop := ∃ x, x = Role.Normal
def statement_B : Prop := statement_A
def statement_C : Prop := ¬∃ x, x = Role.Normal

-- Define the properties of each role
def always_true (r : Role) : Prop := r = Role.Knight
def always_false (r : Role) : Prop := r = Role.Liar
def can_be_either (r : Role) : Prop := r = Role.Normal

-- The main theorem
theorem identify_roles :
  ∃! (role_A role_B role_C : Role),
    -- Each person has a unique role
    role_A ≠ role_B ∧ role_B ≠ role_C ∧ role_A ≠ role_C ∧
    -- One of each role exists
    (always_true role_A ∨ always_true role_B ∨ always_true role_C) ∧
    (always_false role_A ∨ always_false role_B ∨ always_false role_C) ∧
    (can_be_either role_A ∨ can_be_either role_B ∨ can_be_either role_C) ∧
    -- Statements are consistent with roles
    ((always_true role_A → statement_A) ∧ (always_false role_A → ¬statement_A) ∧ (can_be_either role_A → True)) ∧
    ((always_true role_B → statement_B) ∧ (always_false role_B → ¬statement_B) ∧ (can_be_either role_B → True)) ∧
    ((always_true role_C → statement_C) ∧ (always_false role_C → ¬statement_C) ∧ (can_be_either role_C → True)) ∧
    -- The solution
    always_false role_A ∧ always_true role_B ∧ can_be_either role_C :=
by sorry


end NUMINAMATH_CALUDE_identify_roles_l2166_216662


namespace NUMINAMATH_CALUDE_line_passes_through_point_l2166_216642

theorem line_passes_through_point :
  let k : ℚ := 2/5
  let m : ℚ := 4/5
  let line_eq (x y : ℚ) := 2*k*x - m*y = 4
  line_eq 3 (-2) := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l2166_216642


namespace NUMINAMATH_CALUDE_distribute_six_balls_two_boxes_l2166_216601

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to distribute 6 distinguishable balls into 2 distinguishable boxes is 64 -/
theorem distribute_six_balls_two_boxes : distribute_balls 6 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_two_boxes_l2166_216601


namespace NUMINAMATH_CALUDE_min_value_expression_l2166_216612

theorem min_value_expression (x y : ℝ) :
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧
  ∃ (x y : ℝ), x^2 + y^2 - 8*x + 6*y + 25 = 0 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2166_216612


namespace NUMINAMATH_CALUDE_count_non_divisible_is_30_l2166_216686

/-- g(n) is the product of the proper positive integer divisors of n -/
def g (n : ℕ) : ℕ := sorry

/-- Counts the numbers n between 2 and 100 (inclusive) for which n does not divide g(n) -/
def count_non_divisible : ℕ := sorry

theorem count_non_divisible_is_30 : count_non_divisible = 30 := by sorry

end NUMINAMATH_CALUDE_count_non_divisible_is_30_l2166_216686


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l2166_216600

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 3 ∧ b = 4) ∨ (a = 3 ∧ c = 4) ∨ (b = 3 ∧ c = 4) →
  a^2 + b^2 = c^2 →
  c = 5 ∨ c = Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l2166_216600


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2166_216636

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

-- Define the conditions and theorem
theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a3 : a 3 = 2) 
  (h_a4a6 : a 4 * a 6 = 16) :
  (a 9 - a 11) / (a 5 - a 7) = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2166_216636


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l2166_216640

/-- Calculate the profit percent when buying 120 pens at the price of 95 pens and selling with a 2.5% discount -/
theorem profit_percent_calculation (marked_price : ℝ) (h_pos : marked_price > 0) : 
  let cost_price := 95 * marked_price
  let selling_price_per_pen := marked_price * (1 - 0.025)
  let total_selling_price := 120 * selling_price_per_pen
  let profit := total_selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  ∃ ε > 0, abs (profit_percent - 23.16) < ε :=
by sorry


end NUMINAMATH_CALUDE_profit_percent_calculation_l2166_216640


namespace NUMINAMATH_CALUDE_factorial_equation_l2166_216694

theorem factorial_equation : (Nat.factorial 6 - Nat.factorial 4) / Nat.factorial 5 = 29/5 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_l2166_216694


namespace NUMINAMATH_CALUDE_polynomial_equality_l2166_216691

theorem polynomial_equality (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ - a₁ + a₂ - a₃ + a₄ - a₅ = -243 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2166_216691


namespace NUMINAMATH_CALUDE_original_number_l2166_216697

theorem original_number (x : ℚ) : (1 / x) - 2 = 5 / 4 → x = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l2166_216697


namespace NUMINAMATH_CALUDE_triangle_properties_l2166_216644

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle with given properties -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 4)
  (h2 : t.b = 6)
  (h3 : Real.sin t.A = Real.sin (2 * t.B)) :
  Real.cos t.B = 1/3 ∧ 
  1/2 * t.a * t.c * Real.sin t.B = 8 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2166_216644


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l2166_216659

theorem probability_of_black_ball (prob_red prob_white : ℝ) 
  (h1 : prob_red = 0.42)
  (h2 : prob_white = 0.28)
  (h3 : prob_red + prob_white + prob_black = 1) :
  prob_black = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l2166_216659


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l2166_216605

theorem similar_triangles_leg_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (1/2) * a * b = 10 →
  a^2 + b^2 = 36 →
  (1/2) * c * d = 360 →
  c = 6 * a →
  d = 6 * b →
  c + d = 16 * Real.sqrt 30 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l2166_216605


namespace NUMINAMATH_CALUDE_fruit_basket_count_l2166_216655

/-- The number of ways to choose from n identical items -/
def chooseFromIdentical (n : ℕ) : ℕ := n + 1

/-- The number of fruit baskets with at least one fruit -/
def fruitBaskets (pears bananas : ℕ) : ℕ :=
  chooseFromIdentical pears * chooseFromIdentical bananas - 1

theorem fruit_basket_count :
  fruitBaskets 8 12 = 116 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l2166_216655


namespace NUMINAMATH_CALUDE_book_pages_count_l2166_216641

theorem book_pages_count :
  ∀ (P : ℕ),
  (P / 2 : ℕ) = P / 2 →  -- Half of the pages are filled with images
  (P - (P / 2 + 11)) / 2 = 19 →  -- Remaining pages after images and intro, half of which are text
  P = 98 :=
by sorry

end NUMINAMATH_CALUDE_book_pages_count_l2166_216641


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2166_216684

theorem gcd_of_three_numbers : Nat.gcd 10234 (Nat.gcd 14322 24570) = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2166_216684


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l2166_216633

theorem sum_of_coefficients_equals_one (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (1 - 2*x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l2166_216633


namespace NUMINAMATH_CALUDE_prob_neither_red_nor_green_is_one_third_l2166_216692

-- Define the number of pens of each color
def green_pens : ℕ := 5
def black_pens : ℕ := 6
def red_pens : ℕ := 7

-- Define the total number of pens
def total_pens : ℕ := green_pens + black_pens + red_pens

-- Define the probability of picking a pen that is neither red nor green
def prob_neither_red_nor_green : ℚ := black_pens / total_pens

-- Theorem statement
theorem prob_neither_red_nor_green_is_one_third :
  prob_neither_red_nor_green = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_neither_red_nor_green_is_one_third_l2166_216692


namespace NUMINAMATH_CALUDE_jason_initial_cards_l2166_216602

/-- The number of Pokemon cards Jason gave away -/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has left -/
def cards_left : ℕ := 4

/-- The initial number of Pokemon cards Jason had -/
def initial_cards : ℕ := cards_given_away + cards_left

theorem jason_initial_cards : initial_cards = 13 := by
  sorry

end NUMINAMATH_CALUDE_jason_initial_cards_l2166_216602


namespace NUMINAMATH_CALUDE_expansion_properties_l2166_216603

def n : ℕ := 5

def general_term (r : ℕ) : ℚ × ℤ → ℚ := λ (c, p) ↦ c * (2^(10 - r) * (-1)^r)

theorem expansion_properties :
  let tenth_term := general_term 9 (1, -8)
  let constant_term := general_term 5 (1, 0)
  let max_coeff_term := general_term 3 (1, 4)
  (tenth_term = -20) ∧
  (constant_term = -8064) ∧
  (max_coeff_term = -15360) ∧
  (∀ r : ℕ, r ≤ 10 → |general_term r (1, 10 - 2*r)| ≤ |max_coeff_term|) :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l2166_216603


namespace NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l2166_216647

theorem other_root_of_complex_quadratic (z : ℂ) :
  z^2 = -91 + 84*I ∧ z = 7 + 12*I → (-z) = -7 - 12*I ∧ (-z)^2 = -91 + 84*I := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l2166_216647


namespace NUMINAMATH_CALUDE_expression_simplification_l2166_216632

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 - 1) :
  2 * (a + Real.sqrt 3) * (a - Real.sqrt 3) - a * (a - Real.sqrt 2) + 6 = 5 - 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2166_216632


namespace NUMINAMATH_CALUDE_triangle_angle_c_value_l2166_216618

theorem triangle_angle_c_value (A B C x : ℝ) : 
  A = 45 ∧ B = 3 * x ∧ C = (1 / 2) * B ∧ A + B + C = 180 → C = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_value_l2166_216618


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2166_216685

theorem inequality_equivalence (x : ℝ) (h : x > 0) : 
  3/8 + |x - 14/24| < 8/12 ↔ 7/24 < x ∧ x < 7/8 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2166_216685


namespace NUMINAMATH_CALUDE_x_square_plus_reciprocal_l2166_216610

theorem x_square_plus_reciprocal (x : ℝ) (h : 31 = x^6 + 1/x^6) : 
  x^2 + 1/x^2 = (34 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_x_square_plus_reciprocal_l2166_216610


namespace NUMINAMATH_CALUDE_probability_not_spade_first_draw_l2166_216635

theorem probability_not_spade_first_draw (total_cards : ℕ) (spade_cards : ℕ) 
  (h1 : total_cards = 52) (h2 : spade_cards = 13) :
  (total_cards - spade_cards : ℚ) / total_cards = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_spade_first_draw_l2166_216635


namespace NUMINAMATH_CALUDE_P_less_than_Q_l2166_216663

theorem P_less_than_Q (a : ℝ) (h : a ≥ 0) : 
  Real.sqrt a + Real.sqrt (a + 7) < Real.sqrt (a + 3) + Real.sqrt (a + 4) := by
sorry

end NUMINAMATH_CALUDE_P_less_than_Q_l2166_216663


namespace NUMINAMATH_CALUDE_midpoint_of_fractions_l2166_216689

theorem midpoint_of_fractions :
  let a := 1 / 7
  let b := 1 / 9
  let midpoint := (a + b) / 2
  midpoint = 8 / 63 := by sorry

end NUMINAMATH_CALUDE_midpoint_of_fractions_l2166_216689


namespace NUMINAMATH_CALUDE_box_volume_cubes_l2166_216646

theorem box_volume_cubes (p : ℕ) (h : Prime p) : 
  p * (2 * p) * (3 * p) = 6 * p^3 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_cubes_l2166_216646


namespace NUMINAMATH_CALUDE_minimum_sum_a1_a5_min_value_a1_plus_a5_l2166_216695

/-- A positive geometric sequence -/
def PositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r > 0, ∀ k, a (k + 1) = r * a k

theorem minimum_sum_a1_a5 (a : ℕ → ℝ) (h : PositiveGeometricSequence a) 
    (h_prod : a 5 * a 4 * a 2 * a 1 = 16) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = 4 → x + y ≥ 4 :=
by sorry

theorem min_value_a1_plus_a5 (a : ℕ → ℝ) (h : PositiveGeometricSequence a) 
    (h_prod : a 5 * a 4 * a 2 * a 1 = 16) :
  ∃ m : ℝ, m = 4 ∧ a 1 + a 5 ≥ m ∧ ∃ seq : ℕ → ℝ, 
    PositiveGeometricSequence seq ∧ seq 5 * seq 4 * seq 2 * seq 1 = 16 ∧ seq 1 + seq 5 = m :=
by sorry

end NUMINAMATH_CALUDE_minimum_sum_a1_a5_min_value_a1_plus_a5_l2166_216695


namespace NUMINAMATH_CALUDE_corner_divisions_l2166_216690

/-- A corner made up of 3 squares -/
structure Corner :=
  (squares : Fin 3 → Square)

/-- Represents a division of the corner into equal parts -/
structure Division :=
  (parts : ℕ)
  (is_equal : Bool)

/-- Checks if a division of the corner into n parts is possible and equal -/
def is_valid_division (c : Corner) (n : ℕ) : Prop :=
  ∃ (d : Division), d.parts = n ∧ d.is_equal = true

/-- Theorem stating that the corner can be divided into 2, 3, and 4 equal parts -/
theorem corner_divisions (c : Corner) :
  (is_valid_division c 2) ∧ 
  (is_valid_division c 3) ∧ 
  (is_valid_division c 4) :=
sorry

end NUMINAMATH_CALUDE_corner_divisions_l2166_216690


namespace NUMINAMATH_CALUDE_henry_bought_two_fireworks_l2166_216638

/-- The number of fireworks Henry bought -/
def henrys_fireworks (total : ℕ) (last_year : ℕ) (friends : ℕ) : ℕ :=
  total - last_year - friends

/-- Proof that Henry bought 2 fireworks -/
theorem henry_bought_two_fireworks :
  henrys_fireworks 11 6 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_henry_bought_two_fireworks_l2166_216638


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2166_216693

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a₃ + a₁₁ = 22, then a₇ = 11 -/
theorem arithmetic_sequence_property (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) (h_sum : a 3 + a 11 = 22) : 
  a 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2166_216693


namespace NUMINAMATH_CALUDE_adams_fair_expense_l2166_216687

def fair_problem (initial_tickets : ℕ) (ferris_wheel_cost : ℕ) (roller_coaster_cost : ℕ) 
  (remaining_tickets : ℕ) (ticket_price : ℕ) (snack_price : ℕ) : Prop :=
  let used_tickets := initial_tickets - remaining_tickets
  let ride_cost := used_tickets * ticket_price
  let total_spent := ride_cost + snack_price
  total_spent = 99

theorem adams_fair_expense :
  fair_problem 13 2 3 4 9 18 := by
  sorry

end NUMINAMATH_CALUDE_adams_fair_expense_l2166_216687


namespace NUMINAMATH_CALUDE_expression_evaluation_l2166_216634

theorem expression_evaluation :
  let x : ℝ := 3 * Real.sqrt 3 + 2 * Real.sqrt 2
  let y : ℝ := 3 * Real.sqrt 3 - 2 * Real.sqrt 2
  ((x * (x + y) + 2 * y * (x + y)) / (x * y * (x + 2 * y))) / ((x * y) / (x + 2 * y)) = 108 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2166_216634


namespace NUMINAMATH_CALUDE_coin_sum_impossibility_l2166_216643

def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

def is_valid_sum (sum : ℕ) : Prop :=
  ∃ (n d q : ℕ), n + d + q = 6 ∧ n * nickel + d * dime + q * quarter = sum

theorem coin_sum_impossibility :
  is_valid_sum 40 ∧
  is_valid_sum 50 ∧
  is_valid_sum 60 ∧
  is_valid_sum 70 ∧
  ¬ is_valid_sum 30 :=
sorry

end NUMINAMATH_CALUDE_coin_sum_impossibility_l2166_216643


namespace NUMINAMATH_CALUDE_factorial_division_l2166_216666

theorem factorial_division : (Nat.factorial 8) / (Nat.factorial (8 - 2)) = 56 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l2166_216666


namespace NUMINAMATH_CALUDE_diamond_inequality_l2166_216683

def diamond (x y : ℝ) : ℝ := |x^2 - y^2|

theorem diamond_inequality : ∃ x y : ℝ, diamond (x + y) (x - y) ≠ diamond x y := by
  sorry

end NUMINAMATH_CALUDE_diamond_inequality_l2166_216683


namespace NUMINAMATH_CALUDE_train_speed_l2166_216682

/- Define the train length in meters -/
def train_length : ℝ := 160

/- Define the time taken to pass in seconds -/
def passing_time : ℝ := 8

/- Define the conversion factor from m/s to km/h -/
def ms_to_kmh : ℝ := 3.6

/- Theorem statement -/
theorem train_speed : 
  (train_length / passing_time) * ms_to_kmh = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2166_216682


namespace NUMINAMATH_CALUDE_vector_simplification_l2166_216608

variable {V : Type*} [AddCommGroup V]

theorem vector_simplification (P M N : V) : 
  (P - M) - (P - N) + (M - N) = (0 : V) := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l2166_216608


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_one_l2166_216660

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m1 m2 : ℝ) : Prop := m1 = m2

/-- The first line: y = ax - 2 -/
def line1 (a x : ℝ) : ℝ := a * x - 2

/-- The second line: y = (2-a)x + 1 -/
def line2 (a x : ℝ) : ℝ := (2 - a) * x + 1

theorem parallel_lines_imply_a_equals_one (a : ℝ) :
  parallel_lines a (2 - a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_one_l2166_216660


namespace NUMINAMATH_CALUDE_price_increase_l2166_216699

theorem price_increase (x : ℝ) : 
  (1 + x / 100) * (1 + x / 100) = 1 + 32.25 / 100 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_l2166_216699


namespace NUMINAMATH_CALUDE_leak_drain_time_l2166_216651

/-- Given a pump that can fill a tank in 2 hours without a leak,
    and takes 2 1/7 hours to fill the tank with a leak,
    the time it takes for the leak to drain the entire tank is 30 hours. -/
theorem leak_drain_time (fill_time_no_leak fill_time_with_leak : ℚ) : 
  fill_time_no_leak = 2 →
  fill_time_with_leak = 2 + 1 / 7 →
  (1 / (1 / fill_time_no_leak - 1 / fill_time_with_leak)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_leak_drain_time_l2166_216651


namespace NUMINAMATH_CALUDE_rotated_line_slope_l2166_216631

theorem rotated_line_slope (m : ℝ) (θ : ℝ) :
  m = -Real.sqrt 3 →
  θ = π / 3 →
  (m * Real.cos θ + Real.sin θ) / (Real.cos θ - m * Real.sin θ) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rotated_line_slope_l2166_216631


namespace NUMINAMATH_CALUDE_average_salary_problem_l2166_216681

/-- The average monthly salary problem -/
theorem average_salary_problem (initial_average : ℚ) (old_supervisor_salary : ℚ) 
  (new_supervisor_salary : ℚ) (num_workers : ℕ) (total_people : ℕ) 
  (h1 : initial_average = 430)
  (h2 : old_supervisor_salary = 870)
  (h3 : new_supervisor_salary = 780)
  (h4 : num_workers = 8)
  (h5 : total_people = num_workers + 1) :
  let total_initial_salary := initial_average * total_people
  let workers_salary := total_initial_salary - old_supervisor_salary
  let new_total_salary := workers_salary + new_supervisor_salary
  let new_average_salary := new_total_salary / total_people
  new_average_salary = 420 := by sorry

end NUMINAMATH_CALUDE_average_salary_problem_l2166_216681


namespace NUMINAMATH_CALUDE_max_leftover_grapes_l2166_216688

theorem max_leftover_grapes (n : ℕ) : 
  ∃ (q r : ℕ), n = 5 * q + r ∧ r < 5 ∧ r ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_leftover_grapes_l2166_216688


namespace NUMINAMATH_CALUDE_min_blocks_for_wall_l2166_216622

/-- Represents the dimensions of a wall --/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  length : ℚ
  height : ℕ

/-- Calculates the number of blocks needed for a wall --/
def calculateBlocksNeeded (wall : WallDimensions) (block1 : BlockDimensions) (block2 : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating the minimum number of blocks needed for the specified wall --/
theorem min_blocks_for_wall :
  let wall := WallDimensions.mk 120 8
  let block1 := BlockDimensions.mk 1 1
  let block2 := BlockDimensions.mk (3/2) 1
  calculateBlocksNeeded wall block1 block2 = 648 :=
sorry

end NUMINAMATH_CALUDE_min_blocks_for_wall_l2166_216622


namespace NUMINAMATH_CALUDE_max_correct_answers_l2166_216673

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (blank_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) : 
  total_questions = 60 →
  correct_points = 5 →
  blank_points = 0 →
  incorrect_points = -2 →
  total_score = 150 →
  (∃ (correct blank incorrect : ℕ),
    correct + blank + incorrect = total_questions ∧
    correct * correct_points + blank * blank_points + incorrect * incorrect_points = total_score) →
  (∀ (correct : ℕ),
    (∃ (blank incorrect : ℕ),
      correct + blank + incorrect = total_questions ∧
      correct * correct_points + blank * blank_points + incorrect * incorrect_points = total_score) →
    correct ≤ 38) :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l2166_216673


namespace NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l2166_216675

theorem sphere_radius_from_surface_area :
  ∀ (S R : ℝ), S = 4 * Real.pi → S = 4 * Real.pi * R^2 → R = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l2166_216675


namespace NUMINAMATH_CALUDE_absolute_value_equality_l2166_216621

theorem absolute_value_equality (x : ℝ) (h : x > 2) :
  |x - Real.sqrt ((x - 3)^2)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l2166_216621


namespace NUMINAMATH_CALUDE_expression_value_l2166_216674

theorem expression_value (x y z : ℚ) 
  (eq1 : 3 * x - 2 * y - 4 * z = 0)
  (eq2 : 2 * x + y - 9 * z = 0)
  (z_neq_zero : z ≠ 0) :
  (x^2 - 2*x*y) / (y^2 + 2*z^2) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2166_216674


namespace NUMINAMATH_CALUDE_amy_muffin_problem_l2166_216624

/-- Represents the number of muffins Amy brings to school each day -/
def muffins_sequence (first_day : ℕ) : ℕ → ℕ
| 0 => first_day
| n + 1 => muffins_sequence first_day n + 1

/-- Calculates the total number of muffins brought to school over 5 days -/
def total_muffins_brought (first_day : ℕ) : ℕ :=
  (List.range 5).map (muffins_sequence first_day) |>.sum

/-- Theorem stating the solution to Amy's muffin problem -/
theorem amy_muffin_problem :
  ∃ (first_day : ℕ),
    total_muffins_brought first_day = 22 - 7 ∧
    first_day = 1 := by
  sorry

end NUMINAMATH_CALUDE_amy_muffin_problem_l2166_216624


namespace NUMINAMATH_CALUDE_nose_spray_cost_l2166_216670

/-- Calculates the cost per nose spray in a "buy one get one free" promotion -/
def costPerNoseSpray (totalPaid : ℚ) (totalBought : ℕ) : ℚ :=
  totalPaid / (totalBought / 2)

theorem nose_spray_cost :
  let totalPaid : ℚ := 15
  let totalBought : ℕ := 10
  costPerNoseSpray totalPaid totalBought = 3 := by
  sorry

end NUMINAMATH_CALUDE_nose_spray_cost_l2166_216670


namespace NUMINAMATH_CALUDE_current_rate_calculation_l2166_216696

/-- Given a boat with speed in still water and its downstream travel details, 
    calculate the rate of the current. -/
theorem current_rate_calculation 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : boat_speed = 42)
  (h2 : downstream_distance = 33)
  (h3 : downstream_time = 44 / 60) : 
  (downstream_distance / downstream_time) - boat_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_current_rate_calculation_l2166_216696


namespace NUMINAMATH_CALUDE_joan_books_l2166_216676

theorem joan_books (initial_books sold_books : ℕ) 
  (h1 : initial_books = 33)
  (h2 : sold_books = 26) :
  initial_books - sold_books = 7 := by
  sorry

end NUMINAMATH_CALUDE_joan_books_l2166_216676


namespace NUMINAMATH_CALUDE_expand_and_subtract_fraction_division_l2166_216680

-- Part 1
theorem expand_and_subtract (m n : ℝ) :
  (2*m + 3*n)^2 - (2*m + n)*(2*m - n) = 12*m*n + 10*n^2 := by sorry

-- Part 2
theorem fraction_division (x y : ℝ) (hx : x ≠ 0) (hxy : x ≠ y) :
  (x - y) / x / (x + (y^2 - 2*x*y) / x) = 1 / (x - y) := by sorry

end NUMINAMATH_CALUDE_expand_and_subtract_fraction_division_l2166_216680


namespace NUMINAMATH_CALUDE_point_distance_3d_l2166_216607

/-- Given two points A(m, 2, 3) and B(1, -1, 1) in 3D space with distance √13 between them, m = 1 -/
theorem point_distance_3d (m : ℝ) : 
  let A : ℝ × ℝ × ℝ := (m, 2, 3)
  let B : ℝ × ℝ × ℝ := (1, -1, 1)
  (m - 1)^2 + 3^2 + 2^2 = 13 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_distance_3d_l2166_216607


namespace NUMINAMATH_CALUDE_four_cubic_yards_to_cubic_inches_l2166_216671

-- Define the conversion factors
def yard_to_foot : ℝ := 3
def foot_to_inch : ℝ := 12

-- Define the volume conversion function
def cubic_yards_to_cubic_inches (cubic_yards : ℝ) : ℝ :=
  cubic_yards * (yard_to_foot ^ 3) * (foot_to_inch ^ 3)

-- Theorem statement
theorem four_cubic_yards_to_cubic_inches :
  cubic_yards_to_cubic_inches 4 = 186624 := by
  sorry

end NUMINAMATH_CALUDE_four_cubic_yards_to_cubic_inches_l2166_216671


namespace NUMINAMATH_CALUDE_better_discount_order_l2166_216613

def original_price : ℝ := 30
def discount_amount : ℝ := 5
def discount_percent : ℝ := 0.25

def price_percent_then_amount : ℝ :=
  (1 - discount_percent) * original_price - discount_amount

def price_amount_then_percent : ℝ :=
  (1 - discount_percent) * (original_price - discount_amount)

theorem better_discount_order :
  price_percent_then_amount < price_amount_then_percent ∧
  price_amount_then_percent - price_percent_then_amount = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_better_discount_order_l2166_216613


namespace NUMINAMATH_CALUDE_abs_neg_four_minus_six_l2166_216698

theorem abs_neg_four_minus_six : |-4 - 6| = 10 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_four_minus_six_l2166_216698


namespace NUMINAMATH_CALUDE_modified_cube_properties_l2166_216668

/-- Represents a cube with removals as described in the problem -/
structure ModifiedCube where
  side_length : ℕ
  small_cube_size : ℕ
  center_removal_size : ℕ
  unit_removal : Bool

/-- Calculates the remaining volume after removals -/
def remaining_volume (c : ModifiedCube) : ℕ := sorry

/-- Calculates the surface area after removals -/
def surface_area (c : ModifiedCube) : ℕ := sorry

/-- The main theorem stating the properties of the modified cube -/
theorem modified_cube_properties :
  let c : ModifiedCube := {
    side_length := 12,
    small_cube_size := 2,
    center_removal_size := 2,
    unit_removal := true
  }
  remaining_volume c = 1463 ∧ surface_area c = 4598 := by sorry

end NUMINAMATH_CALUDE_modified_cube_properties_l2166_216668


namespace NUMINAMATH_CALUDE_journey_speed_proof_l2166_216672

/-- Proves that given a journey of 10 miles completed in 2 hours, 
    where the first 3 miles are traveled at speed v, 
    the next 3 miles at 3 mph, and the last 4 miles at 8 mph, 
    the value of v must be 6 mph. -/
theorem journey_speed_proof (v : ℝ) : 
  (3 / v + 3 / 3 + 4 / 8 = 2) → v = 6 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l2166_216672


namespace NUMINAMATH_CALUDE_perfect_apples_l2166_216664

theorem perfect_apples (total : ℕ) (small_fraction : ℚ) (unripe_fraction : ℚ) :
  total = 30 →
  small_fraction = 1/6 →
  unripe_fraction = 1/3 →
  (total : ℚ) - small_fraction * total - unripe_fraction * total = 15 := by
  sorry

end NUMINAMATH_CALUDE_perfect_apples_l2166_216664


namespace NUMINAMATH_CALUDE_expression_evaluation_l2166_216626

theorem expression_evaluation : 
  1 / (2 - Real.sqrt 3) - Real.pi ^ 0 - 2 * Real.cos (30 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2166_216626


namespace NUMINAMATH_CALUDE_problem_statement_l2166_216629

theorem problem_statement (x y : ℝ) : 
  |x - 2| + (y + 3)^2 = 0 → (x + y)^2020 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2166_216629


namespace NUMINAMATH_CALUDE_sequence_bounded_l2166_216625

/-- Given a sequence of nonnegative real numbers satisfying certain conditions, prove that it is bounded -/
theorem sequence_bounded (c : ℝ) (a : ℕ → ℝ) (hc : c > 2)
  (h1 : ∀ m n : ℕ, m ≥ 1 → n ≥ 1 → a (m + n) ≤ 2 * a m + 2 * a n)
  (h2 : ∀ k : ℕ, a (2^k) ≤ 1 / ((k : ℝ) + 1)^c)
  (h3 : ∀ n : ℕ, a n ≥ 0) :
  ∃ M : ℝ, ∀ n : ℕ, n ≥ 1 → a n ≤ M :=
sorry

end NUMINAMATH_CALUDE_sequence_bounded_l2166_216625


namespace NUMINAMATH_CALUDE_binomial_expectation_and_variance_l2166_216623

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectedValue (b : BinomialDistribution) : ℝ := b.n * b.p

/-- The variance of a binomial distribution -/
def variance (b : BinomialDistribution) : ℝ := b.n * b.p * (1 - b.p)

/-- Theorem: For a binomial distribution with n=200 and p=0.01, 
    the expected value is 2 and the variance is 1.98 -/
theorem binomial_expectation_and_variance :
  ∃ b : BinomialDistribution, 
    b.n = 200 ∧ 
    b.p = 0.01 ∧ 
    expectedValue b = 2 ∧ 
    variance b = 1.98 := by
  sorry


end NUMINAMATH_CALUDE_binomial_expectation_and_variance_l2166_216623


namespace NUMINAMATH_CALUDE_jessicas_initial_quarters_l2166_216656

/-- 
Given that Jessica received some quarters from her sister and now has a certain number of quarters,
this theorem proves the number of quarters Jessica had initially.
-/
theorem jessicas_initial_quarters 
  (quarters_from_sister : ℕ) -- Number of quarters Jessica received from her sister
  (current_quarters : ℕ) -- Number of quarters Jessica has now
  (h1 : quarters_from_sister = 3) -- Jessica received 3 quarters from her sister
  (h2 : current_quarters = 11) -- Jessica now has 11 quarters
  : current_quarters - quarters_from_sister = 8 := by
  sorry

end NUMINAMATH_CALUDE_jessicas_initial_quarters_l2166_216656


namespace NUMINAMATH_CALUDE_inscribed_rectangle_sides_l2166_216649

/-- A rectangle inscribed in a triangle -/
structure InscribedRectangle where
  -- Triangle dimensions
  triangleBase : ℝ
  triangleHeight : ℝ
  -- Rectangle side ratio
  rectRatio : ℝ
  -- Rectangle sides
  rectShortSide : ℝ
  rectLongSide : ℝ
  -- Conditions
  triangleBase_pos : 0 < triangleBase
  triangleHeight_pos : 0 < triangleHeight
  rectRatio_pos : 0 < rectRatio
  rectShortSide_pos : 0 < rectShortSide
  rectLongSide_pos : 0 < rectLongSide
  ratio_cond : rectLongSide / rectShortSide = 9 / 5
  inscribed_cond : rectLongSide ≤ triangleBase
  proportion_cond : (triangleHeight - rectShortSide) / triangleHeight = rectLongSide / triangleBase

/-- The sides of the inscribed rectangle are 10 and 18 -/
theorem inscribed_rectangle_sides (r : InscribedRectangle) 
    (h1 : r.triangleBase = 48) 
    (h2 : r.triangleHeight = 16) 
    (h3 : r.rectRatio = 9/5) : 
    r.rectShortSide = 10 ∧ r.rectLongSide = 18 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_sides_l2166_216649


namespace NUMINAMATH_CALUDE_max_quotient_value_l2166_216615

theorem max_quotient_value (a b : ℝ) 
  (ha : 210 ≤ a ∧ a ≤ 430) 
  (hb : 590 ≤ b ∧ b ≤ 1190) : 
  (∀ x y, 210 ≤ x ∧ x ≤ 430 ∧ 590 ≤ y ∧ y ≤ 1190 → y / x ≤ 1190 / 210) :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l2166_216615


namespace NUMINAMATH_CALUDE_max_of_min_is_sqrt_two_l2166_216650

theorem max_of_min_is_sqrt_two (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (⨅ z ∈ ({x, 1/y, y + 1/x} : Set ℝ), z) ≤ Real.sqrt 2 ∧
  ∃ x y, x > 0 ∧ y > 0 ∧ (⨅ z ∈ ({x, 1/y, y + 1/x} : Set ℝ), z) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_of_min_is_sqrt_two_l2166_216650


namespace NUMINAMATH_CALUDE_line_segment_param_sum_squares_l2166_216678

/-- Given a line segment connecting (1, -3) and (6, 4), parameterized by x = pt + q and y = rt + s
    where 0 ≤ t ≤ 1, and t = 0 corresponds to (1, -3), prove that p^2 + q^2 + r^2 + s^2 = 84 -/
theorem line_segment_param_sum_squares (p q r s : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = p * t + q ∧ y = r * t + s) →
  (q = 1 ∧ s = -3) →
  (p + q = 6 ∧ r + s = 4) →
  p^2 + q^2 + r^2 + s^2 = 84 := by
  sorry


end NUMINAMATH_CALUDE_line_segment_param_sum_squares_l2166_216678


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2166_216604

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- Predicate for a geometric sequence. -/
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The condition given in the problem. -/
def Condition (a : Sequence) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2

theorem condition_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometric a → Condition a) ∧
  (∃ a : Sequence, Condition a ∧ ¬IsGeometric a) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2166_216604


namespace NUMINAMATH_CALUDE_solve_equation_l2166_216677

theorem solve_equation (x y : ℝ) : y = 3 / (5 * x + 4) → y = 2 → x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2166_216677


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l2166_216658

theorem quadratic_roots_properties (a b m : ℝ) : 
  m > 0 → 
  2 * a^2 - 8 * a + m = 0 → 
  2 * b^2 - 8 * b + m = 0 → 
  (a^2 + b^2 ≥ 8) ∧ 
  (Real.sqrt a + Real.sqrt b ≤ 2 * Real.sqrt 2) ∧ 
  (1 / (a + 2) + 1 / (2 * b) ≥ (3 + 2 * Real.sqrt 2) / 12) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l2166_216658


namespace NUMINAMATH_CALUDE_base_8_sum_4321_l2166_216648

def base_8_sum (n : ℕ) : ℕ :=
  (n.digits 8).sum

theorem base_8_sum_4321 : base_8_sum 4321 = 9 := by sorry

end NUMINAMATH_CALUDE_base_8_sum_4321_l2166_216648


namespace NUMINAMATH_CALUDE_sugar_profit_problem_l2166_216653

/-- A merchant sells sugar with two different profit percentages --/
theorem sugar_profit_problem (total_sugar : ℝ) (sugar_at_known_profit : ℝ) (sugar_at_unknown_profit : ℝ)
  (known_profit_percentage : ℝ) (overall_profit_percentage : ℝ) (unknown_profit_percentage : ℝ)
  (h1 : total_sugar = 1000)
  (h2 : sugar_at_known_profit = 400)
  (h3 : sugar_at_unknown_profit = 600)
  (h4 : known_profit_percentage = 8)
  (h5 : overall_profit_percentage = 14)
  (h6 : total_sugar = sugar_at_known_profit + sugar_at_unknown_profit)
  (h7 : sugar_at_known_profit * (known_profit_percentage / 100) +
        sugar_at_unknown_profit * (unknown_profit_percentage / 100) =
        total_sugar * (overall_profit_percentage / 100)) :
  unknown_profit_percentage = 18 := by
sorry

end NUMINAMATH_CALUDE_sugar_profit_problem_l2166_216653


namespace NUMINAMATH_CALUDE_double_root_values_l2166_216654

/-- A polynomial with integer coefficients of the form x^4 + a₃x³ + a₂x² + a₁x + 18 -/
def P (a₃ a₂ a₁ : ℤ) (x : ℝ) : ℝ := x^4 + a₃*x^3 + a₂*x^2 + a₁*x + 18

/-- r is a double root of P if (x - r)² divides P -/
def is_double_root (r : ℤ) (a₃ a₂ a₁ : ℤ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, P a₃ a₂ a₁ x = (x - r)^2 * q x

theorem double_root_values (a₃ a₂ a₁ : ℤ) (r : ℤ) :
  is_double_root r a₃ a₂ a₁ → r = -3 ∨ r = -1 ∨ r = 1 ∨ r = 3 := by
  sorry

end NUMINAMATH_CALUDE_double_root_values_l2166_216654


namespace NUMINAMATH_CALUDE_max_cos_a_l2166_216627

theorem max_cos_a (a b : Real) (h : Real.cos (a + b) = Real.cos a - Real.cos b) :
  ∃ (max_cos_a : Real), max_cos_a = 1 ∧ ∀ x, Real.cos x ≤ max_cos_a :=
by sorry

end NUMINAMATH_CALUDE_max_cos_a_l2166_216627


namespace NUMINAMATH_CALUDE_faster_train_speed_l2166_216614

/-- The speed of the faster train given the conditions of the problem -/
def speed_of_faster_train (speed_difference : ℝ) (crossing_time : ℝ) (train_length : ℝ) : ℝ :=
  speed_difference * 2

/-- Theorem stating that the speed of the faster train is 72 kmph -/
theorem faster_train_speed :
  speed_of_faster_train 36 15 150 = 72 := by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l2166_216614


namespace NUMINAMATH_CALUDE_final_cake_count_l2166_216652

-- Define the problem parameters
def initial_cakes : ℕ := 110
def cakes_sold : ℕ := 75
def additional_cakes : ℕ := 76

-- Theorem statement
theorem final_cake_count :
  initial_cakes - cakes_sold + additional_cakes = 111 := by
  sorry

end NUMINAMATH_CALUDE_final_cake_count_l2166_216652


namespace NUMINAMATH_CALUDE_coin_balance_problem_l2166_216628

theorem coin_balance_problem :
  ∃ (a b c : ℕ),
    a + b + c = 99 ∧
    2 * a + 3 * b + c = 297 ∧
    3 * a + b + 2 * c = 297 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_balance_problem_l2166_216628


namespace NUMINAMATH_CALUDE_james_payment_is_18_l2166_216611

/-- James's meal cost -/
def james_meal : ℚ := 16

/-- Friend's meal cost -/
def friend_meal : ℚ := 14

/-- Tip percentage -/
def tip_percent : ℚ := 20 / 100

/-- Calculate James's payment given the meal costs and tip percentage -/
def calculate_james_payment (james_meal friend_meal tip_percent : ℚ) : ℚ :=
  let total_before_tip := james_meal + friend_meal
  let tip := tip_percent * total_before_tip
  let total_with_tip := total_before_tip + tip
  total_with_tip / 2

/-- Theorem stating that James's payment is $18 -/
theorem james_payment_is_18 :
  calculate_james_payment james_meal friend_meal tip_percent = 18 := by
  sorry

end NUMINAMATH_CALUDE_james_payment_is_18_l2166_216611


namespace NUMINAMATH_CALUDE_train_speed_l2166_216669

/-- Given a bridge and a train, calculate the train's speed in km/h -/
theorem train_speed (bridge_length train_length : ℝ) (crossing_time : ℝ) : 
  bridge_length = 200 →
  train_length = 100 →
  crossing_time = 60 →
  (bridge_length + train_length) / crossing_time * 3.6 = 18 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l2166_216669


namespace NUMINAMATH_CALUDE_parallel_vectors_angle_l2166_216665

theorem parallel_vectors_angle (α : Real) : 
  let a : Fin 2 → Real := ![1 - Real.cos α, Real.sqrt 3]
  let b : Fin 2 → Real := ![Real.sin α, 3]
  (∀ (i j : Fin 2), a i * b j = a j * b i) →  -- parallel condition
  0 < α → α < Real.pi / 2 →                   -- acute angle condition
  α = Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_angle_l2166_216665


namespace NUMINAMATH_CALUDE_rank_inequality_l2166_216606

variable {n : ℕ}
variable (A B : Matrix (Fin n) (Fin n) ℝ)

theorem rank_inequality (h1 : n ≥ 2) (h2 : B * B = B) :
  Matrix.rank (A * B - B * A) ≤ Matrix.rank (A * B + B * A) := by
  sorry

end NUMINAMATH_CALUDE_rank_inequality_l2166_216606


namespace NUMINAMATH_CALUDE_veranda_width_l2166_216645

/-- Proves that the width of a veranda surrounding a 20 m × 12 m rectangular room is 2 m,
    given that the area of the veranda is 144 m². -/
theorem veranda_width (room_length : ℝ) (room_width : ℝ) (veranda_area : ℝ) :
  room_length = 20 →
  room_width = 12 →
  veranda_area = 144 →
  ∃ w : ℝ, w > 0 ∧ (room_length + 2*w) * (room_width + 2*w) - room_length * room_width = veranda_area ∧ w = 2 :=
by sorry

end NUMINAMATH_CALUDE_veranda_width_l2166_216645


namespace NUMINAMATH_CALUDE_newspaper_circulation_estimate_l2166_216657

/-- Estimated circulation of a newspaper given survey results -/
theorem newspaper_circulation_estimate 
  (city_population : ℕ) 
  (survey_size : ℕ) 
  (buyers_in_survey : ℕ) 
  (h1 : city_population = 8000000)
  (h2 : survey_size = 2500)
  (h3 : buyers_in_survey = 500) :
  (buyers_in_survey : ℚ) / survey_size * (city_population / 10000) = 160 := by
  sorry

#check newspaper_circulation_estimate

end NUMINAMATH_CALUDE_newspaper_circulation_estimate_l2166_216657


namespace NUMINAMATH_CALUDE_larger_number_problem_l2166_216616

theorem larger_number_problem (x y : ℝ) : 
  x - y = 7 → x + y = 45 → x = 26 ∧ x > y := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2166_216616


namespace NUMINAMATH_CALUDE_even_blue_faces_count_l2166_216637

/-- Represents a cube with a certain number of blue faces -/
structure PaintedCube where
  blueFaces : Nat

/-- Represents the wooden block -/
structure WoodenBlock where
  length : Nat
  width : Nat
  height : Nat
  paintedSides : Nat

/-- Function to generate the list of cubes from a wooden block -/
def generateCubes (block : WoodenBlock) : List PaintedCube :=
  sorry

/-- Function to count cubes with even number of blue faces -/
def countEvenBlueFaces (cubes : List PaintedCube) : Nat :=
  sorry

/-- Main theorem -/
theorem even_blue_faces_count (block : WoodenBlock) 
    (h1 : block.length = 5)
    (h2 : block.width = 3)
    (h3 : block.height = 1)
    (h4 : block.paintedSides = 5) :
  countEvenBlueFaces (generateCubes block) = 5 := by
  sorry

end NUMINAMATH_CALUDE_even_blue_faces_count_l2166_216637


namespace NUMINAMATH_CALUDE_andrew_fruit_purchase_l2166_216619

/-- Calculates the total amount paid for fruits given the quantities and prices -/
def totalAmountPaid (grapeQuantity mangoQuantity grapePrice mangoPrice : ℕ) : ℕ :=
  grapeQuantity * grapePrice + mangoQuantity * mangoPrice

/-- Theorem stating that Andrew paid 975 for his fruit purchase -/
theorem andrew_fruit_purchase : 
  totalAmountPaid 6 9 74 59 = 975 := by
  sorry

end NUMINAMATH_CALUDE_andrew_fruit_purchase_l2166_216619


namespace NUMINAMATH_CALUDE_fraction_simplification_l2166_216617

theorem fraction_simplification (a x b : ℝ) (hb : b > 0) :
  (Real.sqrt b * (Real.sqrt (a^2 + x^2) - (x^2 - a^2) / Real.sqrt (a^2 + x^2))) / (b * (a^2 + x^2)) =
  (2 * a^2 * Real.sqrt b) / (b * (a^2 + x^2)^(3/2)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2166_216617


namespace NUMINAMATH_CALUDE_trigonometric_equality_l2166_216620

theorem trigonometric_equality : 
  4 * Real.sin (30 * π / 180) - Real.sqrt 2 * Real.cos (45 * π / 180) - 
  Real.sqrt 3 * Real.tan (30 * π / 180) + 2 * Real.sin (60 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l2166_216620


namespace NUMINAMATH_CALUDE_book_length_l2166_216679

theorem book_length (total_pages : ℕ) : 
  (2 : ℚ) / 3 * total_pages = (1 : ℚ) / 3 * total_pages + 20 → 
  total_pages = 60 := by
sorry

end NUMINAMATH_CALUDE_book_length_l2166_216679


namespace NUMINAMATH_CALUDE_exam_average_marks_l2166_216667

theorem exam_average_marks (total_boys : ℕ) (passed_boys : ℕ) (all_average : ℚ) (passed_average : ℚ) 
  (h1 : total_boys = 120)
  (h2 : passed_boys = 110)
  (h3 : all_average = 37)
  (h4 : passed_average = 39) :
  let failed_boys := total_boys - passed_boys
  let total_marks := total_boys * all_average
  let passed_marks := passed_boys * passed_average
  let failed_marks := total_marks - passed_marks
  failed_marks / failed_boys = 15 := by
sorry

end NUMINAMATH_CALUDE_exam_average_marks_l2166_216667


namespace NUMINAMATH_CALUDE_joggers_regain_sight_main_proof_l2166_216630

/-- The time it takes for two joggers to regain sight of each other after being obscured by a circular stadium --/
theorem joggers_regain_sight (steven_speed linda_speed : ℝ) 
  (path_distance stadium_diameter : ℝ) (initial_distance : ℝ) : ℝ :=
  let t : ℝ := 225
  sorry

/-- The main theorem that proves the time is 225 seconds --/
theorem main_proof : joggers_regain_sight 4 2 300 200 300 = 225 := by
  sorry

end NUMINAMATH_CALUDE_joggers_regain_sight_main_proof_l2166_216630


namespace NUMINAMATH_CALUDE_min_value_trig_expression_limit_approaches_min_value_l2166_216639

open Real

theorem min_value_trig_expression (θ : ℝ) (h : 0 < θ ∧ θ < π/2) :
  3 * cos θ + 1 / (2 * sin θ) + 2 * sqrt 2 * tan θ ≥ 3 * (3 ^ (1/3)) * (sqrt 2 ^ (1/3)) :=
by sorry

theorem limit_approaches_min_value :
  ∀ ε > 0, ∃ δ > 0, ∀ θ, 0 < θ ∧ θ < δ →
    abs ((3 * cos θ + 1 / (2 * sin θ) + 2 * sqrt 2 * tan θ) - 3 * (3 ^ (1/3)) * (sqrt 2 ^ (1/3))) < ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_limit_approaches_min_value_l2166_216639


namespace NUMINAMATH_CALUDE_max_tiles_theorem_l2166_216609

/-- Represents a rhombic tile with side length 1 and angles 60° and 120° -/
structure RhombicTile :=
  (side_length : ℝ := 1)
  (angle1 : ℝ := 60)
  (angle2 : ℝ := 120)

/-- Represents an equilateral triangle with side length n -/
structure EquilateralTriangle :=
  (side_length : ℕ)

/-- Calculates the maximum number of rhombic tiles that can fit in an equilateral triangle -/
def max_tiles_in_triangle (triangle : EquilateralTriangle) : ℕ :=
  (triangle.side_length^2 - triangle.side_length) / 2

/-- Theorem: The maximum number of rhombic tiles in an equilateral triangle is (n^2 - n) / 2 -/
theorem max_tiles_theorem (n : ℕ) (triangle : EquilateralTriangle) (tile : RhombicTile) :
  triangle.side_length = n →
  tile.side_length = 1 →
  tile.angle1 = 60 →
  tile.angle2 = 120 →
  max_tiles_in_triangle triangle = (n^2 - n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_tiles_theorem_l2166_216609
