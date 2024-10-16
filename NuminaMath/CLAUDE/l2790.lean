import Mathlib

namespace NUMINAMATH_CALUDE_min_value_x_plus_3y_l2790_279010

theorem min_value_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 3/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 3/b = 1 → x + 3*y ≤ a + 3*b ∧ ∃ c d : ℝ, c > 0 ∧ d > 0 ∧ 1/c + 3/d = 1 ∧ c + 3*d = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_3y_l2790_279010


namespace NUMINAMATH_CALUDE_max_min_difference_z_l2790_279057

theorem max_min_difference_z (x y z : ℝ) 
  (sum_eq : x + y + z = 3) 
  (sum_squares_eq : x^2 + y^2 + z^2 = 15) : 
  ∃ (z_max z_min : ℝ), 
    (∀ w, (∃ u v, u + v + w = 3 ∧ u^2 + v^2 + w^2 = 15) → w ≤ z_max) ∧
    (∀ w, (∃ u v, u + v + w = 3 ∧ u^2 + v^2 + w^2 = 15) → w ≥ z_min) ∧
    z_max - z_min = 8 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_z_l2790_279057


namespace NUMINAMATH_CALUDE_quadratic_root_sum_minus_product_l2790_279051

theorem quadratic_root_sum_minus_product (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 5 = 0 → 
  x₂^2 - 3*x₂ - 5 = 0 → 
  x₁ + x₂ - x₁ * x₂ = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_minus_product_l2790_279051


namespace NUMINAMATH_CALUDE_circle_problem_l2790_279055

-- Define the circles and points
def largeCircle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 100}
def smallCircle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 36}
def P : ℝ × ℝ := (6, 8)
def S (k : ℝ) : ℝ × ℝ := (0, k)

-- State the theorem
theorem circle_problem (k : ℝ) 
  (h1 : P ∈ largeCircle) 
  (h2 : S k ∈ smallCircle) 
  (h3 : (10 : ℝ) - (6 : ℝ) = 4) : 
  k = 6 := by sorry

end NUMINAMATH_CALUDE_circle_problem_l2790_279055


namespace NUMINAMATH_CALUDE_flowers_per_bouquet_is_nine_l2790_279059

/-- Calculates the number of flowers per bouquet given the initial number of seeds,
    the number of flowers killed, and the number of bouquets to be made. -/
def flowersPerBouquet (seedsPerColor : ℕ) (redKilled yellowKilled orangeKilled purpleKilled : ℕ)
    (numBouquets : ℕ) : ℕ :=
  let redSurvived := seedsPerColor - redKilled
  let yellowSurvived := seedsPerColor - yellowKilled
  let orangeSurvived := seedsPerColor - orangeKilled
  let purpleSurvived := seedsPerColor - purpleKilled
  let totalSurvived := redSurvived + yellowSurvived + orangeSurvived + purpleSurvived
  totalSurvived / numBouquets

/-- Theorem stating that the number of flowers per bouquet is 9 under the given conditions. -/
theorem flowers_per_bouquet_is_nine :
  flowersPerBouquet 125 45 61 30 40 36 = 9 := by
  sorry

#eval flowersPerBouquet 125 45 61 30 40 36

end NUMINAMATH_CALUDE_flowers_per_bouquet_is_nine_l2790_279059


namespace NUMINAMATH_CALUDE_arithmetic_sequence_500th_term_l2790_279087

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_500th_term 
  (p q : ℝ) 
  (h1 : arithmetic_sequence p 9 2 = 9)
  (h2 : arithmetic_sequence p 9 3 = 3*p - q^3)
  (h3 : arithmetic_sequence p 9 4 = 3*p + q^3) :
  arithmetic_sequence p 9 500 = 2005 - 2 * Real.rpow 2 (1/3) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_500th_term_l2790_279087


namespace NUMINAMATH_CALUDE_goods_train_speed_is_36_l2790_279038

/-- The speed of the goods train in km/h -/
def goods_train_speed : ℝ := 36

/-- The speed of the express train in km/h -/
def express_train_speed : ℝ := 90

/-- The time difference between the departure of the two trains in hours -/
def time_difference : ℝ := 6

/-- The time it takes for the express train to catch up with the goods train in hours -/
def catch_up_time : ℝ := 4

/-- Theorem stating that the speed of the goods train is 36 km/h -/
theorem goods_train_speed_is_36 :
  goods_train_speed * (time_difference + catch_up_time) = express_train_speed * catch_up_time :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_is_36_l2790_279038


namespace NUMINAMATH_CALUDE_remainder_proof_l2790_279092

theorem remainder_proof (g : ℕ) (h : g = 144) :
  (6215 % g = 23) ∧ (7373 % g = 29) ∧
  (∀ d : ℕ, d > g → (6215 % d ≠ 6215 % g ∨ 7373 % d ≠ 7373 % g)) := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l2790_279092


namespace NUMINAMATH_CALUDE_smallest_non_odd_unit_digit_l2790_279079

def OddUnitDigits : Set ℕ := {1, 3, 5, 7, 9}

def IsOdd (n : ℕ) : Prop := n % 2 = 1

def UnitsDigit (n : ℕ) : ℕ := n % 10

theorem smallest_non_odd_unit_digit :
  (∀ n : ℕ, IsOdd n → UnitsDigit n ∈ OddUnitDigits) →
  (∀ d : ℕ, d < 0 → d ∉ OddUnitDigits) →
  (∀ d : ℕ, 0 < d → d < 10 → d ∉ OddUnitDigits → 0 < d) →
  (0 ∉ OddUnitDigits ∧ ∀ d : ℕ, d < 10 → d ∉ OddUnitDigits → 0 ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_odd_unit_digit_l2790_279079


namespace NUMINAMATH_CALUDE_additional_tickets_needed_l2790_279027

def ferris_wheel_cost : ℕ := 5
def roller_coaster_cost : ℕ := 4
def bumper_cars_cost : ℕ := 4
def current_tickets : ℕ := 5

def total_cost : ℕ := ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost

theorem additional_tickets_needed : 
  (total_cost - current_tickets : ℕ) = 8 := by sorry

end NUMINAMATH_CALUDE_additional_tickets_needed_l2790_279027


namespace NUMINAMATH_CALUDE_prob_third_batch_given_two_standard_l2790_279066

/-- Represents the number of parts in each batch -/
def batch_size : ℕ := 20

/-- Represents the number of standard parts in the first batch -/
def standard_parts_batch1 : ℕ := 20

/-- Represents the number of standard parts in the second batch -/
def standard_parts_batch2 : ℕ := 15

/-- Represents the number of standard parts in the third batch -/
def standard_parts_batch3 : ℕ := 10

/-- Represents the probability of selecting a batch -/
def prob_select_batch : ℚ := 1 / 3

/-- Theorem stating the probability of selecting two standard parts from the third batch,
    given that two standard parts were selected consecutively from a randomly chosen batch -/
theorem prob_third_batch_given_two_standard : 
  (prob_select_batch * (standard_parts_batch3 / batch_size)^2) /
  (prob_select_batch * (standard_parts_batch1 / batch_size)^2 +
   prob_select_batch * (standard_parts_batch2 / batch_size)^2 +
   prob_select_batch * (standard_parts_batch3 / batch_size)^2) = 4 / 29 := by
  sorry

end NUMINAMATH_CALUDE_prob_third_batch_given_two_standard_l2790_279066


namespace NUMINAMATH_CALUDE_a_investment_value_l2790_279032

/-- Represents the investment and profit distribution in a partnership business --/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit_share : ℕ

/-- The theorem states that given the specific conditions of the partnership,
    a's investment must be 24000 --/
theorem a_investment_value (p : Partnership)
  (hb : p.b_investment = 32000)
  (hc : p.c_investment = 36000)
  (hp : p.total_profit = 92000)
  (hcs : p.c_profit_share = 36000)
  (h_profit_distribution : p.c_profit_share = p.c_investment * p.total_profit / (p.a_investment + p.b_investment + p.c_investment)) :
  p.a_investment = 24000 := by
  sorry


end NUMINAMATH_CALUDE_a_investment_value_l2790_279032


namespace NUMINAMATH_CALUDE_porter_buns_problem_l2790_279044

/-- The maximum number of buns that can be transported given the conditions -/
def max_buns_transported (total_buns : ℕ) (capacity : ℕ) (eaten_per_trip : ℕ) : ℕ :=
  total_buns - (2 * (total_buns / capacity) - 1) * eaten_per_trip

/-- Theorem stating that given the specific conditions, the maximum number of buns transported is 191 -/
theorem porter_buns_problem :
  max_buns_transported 200 40 1 = 191 := by
  sorry

end NUMINAMATH_CALUDE_porter_buns_problem_l2790_279044


namespace NUMINAMATH_CALUDE_tangent_slope_angle_at_one_zero_l2790_279040

/-- The slope angle of the tangent line to y = x^2 - x at (1, 0) is 45° -/
theorem tangent_slope_angle_at_one_zero :
  let f : ℝ → ℝ := λ x => x^2 - x
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let slope : ℝ := deriv f x₀
  Real.arctan slope * (180 / Real.pi) = 45 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_at_one_zero_l2790_279040


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l2790_279001

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l2790_279001


namespace NUMINAMATH_CALUDE_projection_theorem_l2790_279033

def v : Fin 3 → ℝ := ![1, 2, 3]
def proj_v : Fin 3 → ℝ := ![2, 4, 6]
def u : Fin 3 → ℝ := ![2, 1, -1]

theorem projection_theorem (w : Fin 3 → ℝ) 
  (hw : ∃ (k : ℝ), w = fun i => k * proj_v i) :
  let proj_u := (u • w) / (w • w) • w
  proj_u = fun i => (![1/14, 1/7, 3/14] : Fin 3 → ℝ) i := by
  sorry

#check projection_theorem

end NUMINAMATH_CALUDE_projection_theorem_l2790_279033


namespace NUMINAMATH_CALUDE_square_side_increase_l2790_279028

theorem square_side_increase (p : ℝ) : 
  (1 + p / 100)^2 = 1.21 → p = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_side_increase_l2790_279028


namespace NUMINAMATH_CALUDE_triangle_properties_l2790_279056

/-- An acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

/-- The theorem stating the properties of the specific triangle -/
theorem triangle_properties (t : AcuteTriangle) 
  (h1 : t.a = 2 * t.b * Real.sin t.A)
  (h2 : t.a = 3 * Real.sqrt 3)
  (h3 : t.c = 5) :
  t.B = π/6 ∧ t.b = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2790_279056


namespace NUMINAMATH_CALUDE_second_quadrant_trig_identity_l2790_279070

theorem second_quadrant_trig_identity (α : Real) 
  (h1 : π / 2 < α) (h2 : α < π) : 
  (2 * Real.sin α) / Real.sqrt (1 - Real.cos α ^ 2) + 
  Real.sqrt (1 - Real.sin α ^ 2) / Real.cos α = 1 := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_trig_identity_l2790_279070


namespace NUMINAMATH_CALUDE_expression_equality_l2790_279021

theorem expression_equality : 0.064^(-(1/3)) - (-7/8)^0 + 16^0.75 + 0.25^(1/2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2790_279021


namespace NUMINAMATH_CALUDE_equation_solutions_l2790_279048

theorem equation_solutions :
  (∃ x : ℝ, (3 + x) * (30 / 100) = 4.8 ∧ x = 13) ∧
  (∃ x : ℝ, 5 / x = (9 / 2) / (8 / 5) ∧ x = 16 / 9) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2790_279048


namespace NUMINAMATH_CALUDE_work_increase_with_absence_l2790_279012

/-- Given a total work W distributed among p persons, if 1/5 of the members are absent,
    the increase in work for each remaining person is W/(4p). -/
theorem work_increase_with_absence (W p : ℝ) (h : p > 0) :
  let original_work_per_person := W / p
  let remaining_persons := (4 / 5) * p
  let new_work_per_person := W / remaining_persons
  new_work_per_person - original_work_per_person = W / (4 * p) :=
by sorry

end NUMINAMATH_CALUDE_work_increase_with_absence_l2790_279012


namespace NUMINAMATH_CALUDE_equation_roots_property_l2790_279084

theorem equation_roots_property :
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧ 2 * x₁^2 - 5 = 20 ∧ 2 * x₂^2 - 5 = 20) ∧
  (∃ y₁ y₂ : ℝ, y₁ < 0 ∧ y₂ > 0 ∧ (3 * y₁ - 2)^2 = (2 * y₁ - 3)^2 ∧ (3 * y₂ - 2)^2 = (2 * y₂ - 3)^2) ∧
  (∃ z₁ z₂ : ℝ, z₁ < 0 ∧ z₂ > 0 ∧ (z₁^2 - 16 ≥ 0) ∧ (2 * z₁ - 2 ≥ 0) ∧ z₁^2 - 16 = 2 * z₁ - 2 ∧
                              (z₂^2 - 16 ≥ 0) ∧ (2 * z₂ - 2 ≥ 0) ∧ z₂^2 - 16 = 2 * z₂ - 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_property_l2790_279084


namespace NUMINAMATH_CALUDE_unique_positive_integer_l2790_279003

theorem unique_positive_integer : ∃! (n : ℕ), n > 0 ∧ 15 * n = n^2 + 56 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_l2790_279003


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2790_279089

theorem trigonometric_identity (a b : ℝ) : 
  (∀ x : ℝ, 2 * (Real.cos (x + b / 2))^2 - 2 * Real.sin (a * x - π / 2) * Real.cos (a * x - π / 2) = 1) ↔ 
  ((a = 1 ∧ ∃ k : ℤ, b = -3 * π / 2 + 2 * ↑k * π) ∨ 
   (a = -1 ∧ ∃ k : ℤ, b = 3 * π / 2 + 2 * ↑k * π)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2790_279089


namespace NUMINAMATH_CALUDE_average_of_arithmetic_sequence_l2790_279002

theorem average_of_arithmetic_sequence (z : ℝ) : 
  let seq := [5, 5 + 3*z, 5 + 6*z, 5 + 9*z, 5 + 12*z]
  (seq.sum / seq.length : ℝ) = 5 + 6*z := by sorry

end NUMINAMATH_CALUDE_average_of_arithmetic_sequence_l2790_279002


namespace NUMINAMATH_CALUDE_triangle_ratio_proof_l2790_279095

theorem triangle_ratio_proof (A B C D : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  let BD := Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2)
  let DC := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  AB = 8 →
  BC = 13 →
  AC = 10 →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (B.1 + t * (C.1 - B.1), B.2 + t * (C.2 - B.2))) →
  AD = 8 →
  BD / DC = 133 / 36 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_proof_l2790_279095


namespace NUMINAMATH_CALUDE_probability_three_white_balls_l2790_279081

def total_balls : ℕ := 15
def white_balls : ℕ := 7
def black_balls : ℕ := 8
def drawn_balls : ℕ := 3

theorem probability_three_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 7 / 91 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_white_balls_l2790_279081


namespace NUMINAMATH_CALUDE_p_sufficient_but_not_necessary_for_q_l2790_279016

-- Define the propositions
variable (p q r s : Prop)

-- Define the relationships between the propositions
variable (h1 : p → r)  -- p is sufficient for r
variable (h2 : r → s)  -- s is necessary for r
variable (h3 : s → q)  -- q is necessary for s

-- State the theorem
theorem p_sufficient_but_not_necessary_for_q :
  (p → q) ∧ ¬(q → p) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_but_not_necessary_for_q_l2790_279016


namespace NUMINAMATH_CALUDE_set_equality_proof_l2790_279041

universe u

def U : Set Nat := {1, 2, 3, 4}
def M : Set Nat := {1, 3, 4}
def N : Set Nat := {1, 2}

theorem set_equality_proof :
  ({2, 3, 4} : Set Nat) = (U \ M) ∪ (U \ N) := by sorry

end NUMINAMATH_CALUDE_set_equality_proof_l2790_279041


namespace NUMINAMATH_CALUDE_no_three_similar_piles_l2790_279053

theorem no_three_similar_piles : ¬∃ (x a b c : ℝ), 
  (x > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0) ∧ 
  (a + b + c = x) ∧
  (a ≤ Real.sqrt 2 * b ∧ b ≤ Real.sqrt 2 * a) ∧
  (a ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * a) ∧
  (b ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * b) := by
  sorry

end NUMINAMATH_CALUDE_no_three_similar_piles_l2790_279053


namespace NUMINAMATH_CALUDE_clothes_spending_fraction_l2790_279078

theorem clothes_spending_fraction (initial_amount : ℝ) (fraction_clothes : ℝ) : 
  initial_amount = 249.99999999999994 →
  (3/4 : ℝ) * (4/5 : ℝ) * (1 - fraction_clothes) * initial_amount = 100 →
  fraction_clothes = 11/15 := by
  sorry

end NUMINAMATH_CALUDE_clothes_spending_fraction_l2790_279078


namespace NUMINAMATH_CALUDE_equation_solution_l2790_279004

theorem equation_solution (x : ℂ) : 
  (x^2 + x + 1) / (x + 1) = x^2 + 2*x + 2 ↔ 
  (x = -1 ∨ x = (-1 + Complex.I * Real.sqrt 3) / 2 ∨ x = (-1 - Complex.I * Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2790_279004


namespace NUMINAMATH_CALUDE_equal_increment_implies_linear_l2790_279005

/-- A function with the property that equal increments in input correspond to equal increments in output -/
def EqualIncrementFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ x₃ x₄ : ℝ, x₂ - x₁ = x₄ - x₃ → f x₂ - f x₁ = f x₄ - f x₃

/-- The main theorem: if a function has the equal increment property, then it is linear -/
theorem equal_increment_implies_linear (f : ℝ → ℝ) (h : EqualIncrementFunction f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b :=
sorry

end NUMINAMATH_CALUDE_equal_increment_implies_linear_l2790_279005


namespace NUMINAMATH_CALUDE_peggy_needs_825_stamps_l2790_279076

/-- The number of stamps Peggy needs to add to have as many as Bert -/
def stamps_to_add (peggy_stamps ernie_stamps bert_stamps : ℕ) : ℕ :=
  bert_stamps - peggy_stamps

/-- Proof that Peggy needs to add 825 stamps to have as many as Bert -/
theorem peggy_needs_825_stamps : 
  ∀ (peggy_stamps ernie_stamps bert_stamps : ℕ),
    peggy_stamps = 75 →
    ernie_stamps = 3 * peggy_stamps →
    bert_stamps = 4 * ernie_stamps →
    stamps_to_add peggy_stamps ernie_stamps bert_stamps = 825 := by
  sorry

end NUMINAMATH_CALUDE_peggy_needs_825_stamps_l2790_279076


namespace NUMINAMATH_CALUDE_range_of_squared_sum_l2790_279017

theorem range_of_squared_sum (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + y = 1) :
  1/2 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_squared_sum_l2790_279017


namespace NUMINAMATH_CALUDE_overtime_pay_is_correct_l2790_279023

/-- Represents the time interval between minute and hour hand overlaps on a normal clock in minutes -/
def normal_overlap : ℚ := 720 / 11

/-- Represents the time interval between minute and hour hand overlaps on the slow clock in minutes -/
def slow_overlap : ℕ := 69

/-- Represents the normal workday duration in hours -/
def normal_workday : ℕ := 8

/-- Represents the regular hourly pay rate in dollars -/
def regular_rate : ℚ := 4

/-- Represents the overtime pay rate multiplier -/
def overtime_multiplier : ℚ := 3/2

/-- Theorem stating that the overtime pay is $2.60 given the specified conditions -/
theorem overtime_pay_is_correct :
  let actual_time_ratio : ℚ := slow_overlap / normal_overlap
  let actual_time_worked : ℚ := normal_workday * actual_time_ratio
  let overtime_hours : ℚ := actual_time_worked - normal_workday
  let overtime_pay : ℚ := overtime_hours * regular_rate * overtime_multiplier
  overtime_pay = 13/5 := by sorry

end NUMINAMATH_CALUDE_overtime_pay_is_correct_l2790_279023


namespace NUMINAMATH_CALUDE_jonies_cousins_ages_sum_l2790_279091

theorem jonies_cousins_ages_sum : ∃ (a b c d : ℕ),
  (0 < a ∧ a < 10) ∧
  (0 < b ∧ b < 10) ∧
  (0 < c ∧ c < 10) ∧
  (0 < d ∧ d < 10) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * b = 28 ∧
  c * d = 45 ∧
  a + b + c + d = 25 :=
by sorry

end NUMINAMATH_CALUDE_jonies_cousins_ages_sum_l2790_279091


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2790_279093

theorem floor_equation_solution (a b : ℝ) : 
  (∀ n : ℕ+, a * ⌊b * n⌋ = b * ⌊a * n⌋) ↔ 
  (a = 0 ∨ b = 0 ∨ (a = b ∧ ∃ m : ℤ, a = m)) := by
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2790_279093


namespace NUMINAMATH_CALUDE_smallest_max_sum_l2790_279008

theorem smallest_max_sum (a b c d e : ℕ+) (h_sum : a + b + c + d + e = 2500) :
  ∃ N : ℕ, N = max (a + b) (max (b + c) (max (c + d) (d + e))) ∧
  (∀ M : ℕ, M = max (a + b) (max (b + c) (max (c + d) (d + e))) → N ≤ M) ∧
  N = 834 :=
sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l2790_279008


namespace NUMINAMATH_CALUDE_parallel_line_equation_l2790_279073

/-- A line passing through a point and parallel to another line -/
theorem parallel_line_equation (x y : ℝ) : 
  (x - 2*y + 7 = 0) ↔ 
  (∃ (m b : ℝ), y = m*x + b ∧ m = (1/2) ∧ y = m*(x+1) + 3) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l2790_279073


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_57_degree_angle_l2790_279060

-- Define the original angle
def original_angle : ℝ := 57

-- Define complement
def complement (angle : ℝ) : ℝ := 90 - angle

-- Define supplement
def supplement (angle : ℝ) : ℝ := 180 - angle

-- Theorem statement
theorem supplement_of_complement_of_57_degree_angle :
  supplement (complement original_angle) = 147 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_57_degree_angle_l2790_279060


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2790_279035

theorem complex_equation_sum (a b : ℝ) :
  (2 : ℂ) - 2 * Complex.I^3 = a + b * Complex.I → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2790_279035


namespace NUMINAMATH_CALUDE_expression_simplification_l2790_279077

theorem expression_simplification (x : ℝ) 
  (hx : x ≠ 0 ∧ x ≠ 3 ∧ x ≠ 2) : 
  (x - 5) / (x - 3) - ((x^2 + 2*x + 1) / (x^2 + x)) / ((x + 1) / (x - 2)) = 
  -6 / (x^2 - 3*x) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l2790_279077


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2790_279031

/-- Given a complex number Z satisfying (1+i)Z = 2, prove that Z = 1 - i -/
theorem complex_equation_solution (Z : ℂ) (h : (1 + Complex.I) * Z = 2) : Z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2790_279031


namespace NUMINAMATH_CALUDE_pens_multiple_of_ten_l2790_279014

/-- Given that 920 pencils and some pens can be distributed equally among 10 students,
    prove that the number of pens must be a multiple of 10. -/
theorem pens_multiple_of_ten (num_pens : ℕ) (h : ∃ (pens_per_student : ℕ), num_pens = 10 * pens_per_student) :
  ∃ k : ℕ, num_pens = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_pens_multiple_of_ten_l2790_279014


namespace NUMINAMATH_CALUDE_lizette_stamps_l2790_279065

/-- Given that Lizette has 125 more stamps than Minerva and Minerva has 688 stamps,
    prove that Lizette has 813 stamps. -/
theorem lizette_stamps (minerva_stamps : ℕ) (lizette_extra : ℕ) 
  (h1 : minerva_stamps = 688)
  (h2 : lizette_extra = 125) : 
  minerva_stamps + lizette_extra = 813 := by
  sorry

end NUMINAMATH_CALUDE_lizette_stamps_l2790_279065


namespace NUMINAMATH_CALUDE_indeterminate_f_five_l2790_279020

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem indeterminate_f_five
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_shift : ∀ x, f 1 = f (x + 2) ∧ f (x + 2) = f x + f 2) :
  ¬∃ y, ∀ f, IsOdd f → (∀ x, f 1 = f (x + 2) ∧ f (x + 2) = f x + f 2) → f 5 = y :=
sorry

end NUMINAMATH_CALUDE_indeterminate_f_five_l2790_279020


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2790_279088

theorem condition_sufficient_not_necessary :
  (∃ x y : ℝ, x = 1 ∧ y = -1 → x * y = -1) ∧
  ¬(∀ x y : ℝ, x * y = -1 → x = 1 ∧ y = -1) :=
sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2790_279088


namespace NUMINAMATH_CALUDE_cat_food_finished_on_sunday_l2790_279009

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the number of cans consumed up to and including a given day -/
def cans_consumed (d : Day) : ℚ :=
  match d with
  | Day.Monday => 3/4
  | Day.Tuesday => 3/2
  | Day.Wednesday => 9/4
  | Day.Thursday => 3
  | Day.Friday => 15/4
  | Day.Saturday => 9/2
  | Day.Sunday => 21/4

/-- The amount of cat food Roy starts with -/
def initial_cans : ℚ := 8

theorem cat_food_finished_on_sunday :
  ∀ d : Day, cans_consumed d ≤ initial_cans ∧
  (d = Day.Sunday → cans_consumed d > initial_cans - 3/4) :=
by sorry

end NUMINAMATH_CALUDE_cat_food_finished_on_sunday_l2790_279009


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2790_279090

theorem book_arrangement_count : 
  let total_books : ℕ := 10
  let arabic_books : ℕ := 2
  let german_books : ℕ := 4
  let spanish_books : ℕ := 4
  let arrangements : ℕ := (arabic_books.factorial * spanish_books.factorial * (total_books - arabic_books - spanish_books + 2).factorial)
  arrangements = 34560 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2790_279090


namespace NUMINAMATH_CALUDE_chime_2400_date_l2790_279086

/-- Represents a date in the year 2004 --/
structure Date2004 where
  month : Nat
  day : Nat

/-- Represents a time of day --/
structure Time where
  hour : Nat
  minute : Nat

/-- Calculates the number of chimes for a given hour --/
def chimesForHour (hour : Nat) : Nat :=
  3 * (hour % 12 + if hour % 12 = 0 then 12 else 0)

/-- Calculates the total chimes from the start time to midnight --/
def chimesToMidnight (startTime : Time) : Nat :=
  sorry

/-- Calculates the total chimes for a full day --/
def chimesPerDay : Nat :=
  258

/-- Determines the date when the nth chime occurs --/
def dateOfNthChime (n : Nat) (startDate : Date2004) (startTime : Time) : Date2004 :=
  sorry

theorem chime_2400_date :
  dateOfNthChime 2400 ⟨2, 28⟩ ⟨17, 45⟩ = ⟨3, 7⟩ := by sorry

end NUMINAMATH_CALUDE_chime_2400_date_l2790_279086


namespace NUMINAMATH_CALUDE_sqrt_13_parts_sum_l2790_279072

theorem sqrt_13_parts_sum (a b : ℝ) : 
  (3 : ℝ) < Real.sqrt 13 ∧ Real.sqrt 13 < 4 →
  a = ⌊Real.sqrt 13⌋ →
  b = Real.sqrt 13 - ⌊Real.sqrt 13⌋ →
  a^2 + b - Real.sqrt 13 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_13_parts_sum_l2790_279072


namespace NUMINAMATH_CALUDE_beads_per_necklace_l2790_279094

theorem beads_per_necklace (members : ℕ) (necklaces_per_member : ℕ) (total_beads : ℕ) :
  members = 9 →
  necklaces_per_member = 2 →
  total_beads = 900 →
  total_beads / (members * necklaces_per_member) = 50 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_necklace_l2790_279094


namespace NUMINAMATH_CALUDE_work_completion_proof_l2790_279034

/-- The original number of men working on a project -/
def original_men : ℕ := 48

/-- The number of days it takes the original group to complete the work -/
def original_days : ℕ := 60

/-- The number of additional men added to the group -/
def additional_men : ℕ := 8

/-- The number of days it takes the larger group to complete the work -/
def new_days : ℕ := 50

/-- The amount of work to be completed -/
def work : ℝ := 1

theorem work_completion_proof :
  (original_men : ℝ) * work / original_days = 
  ((original_men + additional_men) : ℝ) * work / new_days :=
by sorry

#check work_completion_proof

end NUMINAMATH_CALUDE_work_completion_proof_l2790_279034


namespace NUMINAMATH_CALUDE_price_decrease_after_increase_l2790_279046

theorem price_decrease_after_increase (original_price : ℝ) (original_price_pos : original_price > 0) :
  let increased_price := original_price * 1.3
  let decrease_factor := 1 - (1 / 1.3)
  increased_price * (1 - decrease_factor) = original_price :=
by sorry

end NUMINAMATH_CALUDE_price_decrease_after_increase_l2790_279046


namespace NUMINAMATH_CALUDE_linda_savings_l2790_279018

theorem linda_savings (tv_cost : ℝ) (tv_fraction : ℝ) : 
  tv_cost = 300 → tv_fraction = 1/2 → tv_cost / tv_fraction = 600 := by
  sorry

end NUMINAMATH_CALUDE_linda_savings_l2790_279018


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_OBEC_l2790_279013

/-- A line with slope -3 passing through points A and B -/
def line1 (x y : ℝ) : Prop := y = -3 * x + 13

/-- A line passing through points C and D -/
def line2 (x y : ℝ) : Prop := y = -x + 7

/-- Point A on the x-axis -/
def A : ℝ × ℝ := (5, 0)

/-- Point B on the y-axis -/
def B : ℝ × ℝ := (0, 13)

/-- Point C on the x-axis -/
def C : ℝ × ℝ := (5, 0)

/-- Point D on the y-axis -/
def D : ℝ × ℝ := (0, 7)

/-- Point E where the lines intersect -/
def E : ℝ × ℝ := (3, 4)

/-- The area of quadrilateral OBEC -/
def area_OBEC : ℝ := 67.5

theorem area_of_quadrilateral_OBEC :
  line1 E.1 E.2 ∧ line2 E.1 E.2 →
  area_OBEC = (B.2 * E.1 + C.1 * E.2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_OBEC_l2790_279013


namespace NUMINAMATH_CALUDE_valentines_theorem_l2790_279006

theorem valentines_theorem (boys girls : ℕ) : 
  boys * girls = boys + girls + 40 → boys * girls = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_valentines_theorem_l2790_279006


namespace NUMINAMATH_CALUDE_car_distance_proof_l2790_279030

theorem car_distance_proof (D : ℝ) : 
  (D / 60 = D / 90 + 1/2) → D = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l2790_279030


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2790_279042

theorem arithmetic_expression_evaluation :
  (80 / 16) + (100 / 25) + ((6^2) * 3) - 300 - ((324 / 9) * 2) = -255 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2790_279042


namespace NUMINAMATH_CALUDE_point_not_in_second_quadrant_l2790_279063

theorem point_not_in_second_quadrant (n : ℝ) : ¬(n + 1 < 0 ∧ 2*n - 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_second_quadrant_l2790_279063


namespace NUMINAMATH_CALUDE_jack_morning_emails_l2790_279019

/-- Given that Jack received 3 emails in the afternoon, 1 email in the evening,
    and a total of 10 emails in the day, prove that he received 6 emails in the morning. -/
theorem jack_morning_emails
  (total : ℕ)
  (afternoon : ℕ)
  (evening : ℕ)
  (h1 : total = 10)
  (h2 : afternoon = 3)
  (h3 : evening = 1) :
  total - (afternoon + evening) = 6 :=
by sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l2790_279019


namespace NUMINAMATH_CALUDE_average_side_length_of_squares_l2790_279080

theorem average_side_length_of_squares (a b c : Real) 
  (ha : a = 25) (hb : b = 64) (hc : c = 144) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_side_length_of_squares_l2790_279080


namespace NUMINAMATH_CALUDE_souvenir_october_price_l2790_279098

/-- Represents the selling price and sales data of a souvenir --/
structure SouvenirSales where
  september_price : ℝ
  september_revenue : ℝ
  october_discount : ℝ
  october_volume_increase : ℕ
  october_revenue_increase : ℝ

/-- Calculates the October price of a souvenir given its sales data --/
def october_price (s : SouvenirSales) : ℝ :=
  s.september_price * (1 - s.october_discount)

/-- Theorem stating the October price of the souvenir --/
theorem souvenir_october_price (s : SouvenirSales) 
  (h1 : s.september_revenue = 2000)
  (h2 : s.october_discount = 0.1)
  (h3 : s.october_volume_increase = 20)
  (h4 : s.october_revenue_increase = 700) :
  october_price s = 45 := by
  sorry

end NUMINAMATH_CALUDE_souvenir_october_price_l2790_279098


namespace NUMINAMATH_CALUDE_x_squared_coefficient_l2790_279064

/-- The coefficient of x² in the expansion of (3x² + 4x + 5)(6x² + 7x + 8) is 82 -/
theorem x_squared_coefficient (x : ℝ) : 
  (3*x^2 + 4*x + 5) * (6*x^2 + 7*x + 8) = 18*x^4 + 39*x^3 + 82*x^2 + 67*x + 40 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_l2790_279064


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2790_279025

theorem polynomial_remainder (x : ℝ) : (x^14 + 1) % (x + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2790_279025


namespace NUMINAMATH_CALUDE_orange_shells_count_l2790_279075

theorem orange_shells_count (total shells purple pink yellow blue : ℕ) 
  (h1 : total = 65)
  (h2 : purple = 13)
  (h3 : pink = 8)
  (h4 : yellow = 18)
  (h5 : blue = 12) :
  total - (purple + pink + yellow + blue) = 14 := by
  sorry

end NUMINAMATH_CALUDE_orange_shells_count_l2790_279075


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_7_8_9_l2790_279096

theorem six_digit_divisible_by_7_8_9 :
  ∃ n : ℕ, 523000 ≤ n ∧ n ≤ 523999 ∧ 7 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_7_8_9_l2790_279096


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l2790_279024

theorem thirty_percent_less_than_ninety (x : ℝ) : x + x / 2 = 63 → x = 42 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l2790_279024


namespace NUMINAMATH_CALUDE_division_problem_l2790_279029

theorem division_problem (x : ℝ) : 100 / x = 400 → x = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2790_279029


namespace NUMINAMATH_CALUDE_exists_noninteger_zero_point_l2790_279054

/-- Definition of the polynomial p(x,y) -/
def p (b : Fin 12 → ℝ) (x y : ℝ) : ℝ :=
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 + 
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3 + 
  b 10 * x^4 + b 11 * y^4

/-- The theorem stating the existence of a non-integer point (r,s) where p(r,s) = 0 -/
theorem exists_noninteger_zero_point :
  ∃ (r s : ℝ), ¬(∃ m n : ℤ, (r : ℝ) = m ∧ (s : ℝ) = n) ∧
    ∀ (b : Fin 12 → ℝ), 
      p b 0 0 = 0 ∧ p b 1 0 = 0 ∧ p b (-1) 0 = 0 ∧ 
      p b 0 1 = 0 ∧ p b 0 (-1) = 0 ∧ p b 1 1 = 0 ∧ 
      p b 1 (-1) = 0 ∧ p b 2 2 = 0 ∧ p b (-1) (-1) = 0 →
      p b r s = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_noninteger_zero_point_l2790_279054


namespace NUMINAMATH_CALUDE_ninth_term_of_geometric_sequence_l2790_279061

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem ninth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_sixth : a 6 = 16)
  (h_twelfth : a 12 = 4) :
  a 9 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ninth_term_of_geometric_sequence_l2790_279061


namespace NUMINAMATH_CALUDE_arc_length_120_degrees_l2790_279037

/-- The arc length of a sector in a circle with radius π and central angle 120° -/
theorem arc_length_120_degrees (r : Real) (θ : Real) : 
  r = π → θ = 2 * π / 3 → 2 * π * r * (θ / (2 * π)) = 2 * π^2 / 3 := by
  sorry

#check arc_length_120_degrees

end NUMINAMATH_CALUDE_arc_length_120_degrees_l2790_279037


namespace NUMINAMATH_CALUDE_function_properties_l2790_279043

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 1 / Real.exp x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  Real.log ((3 - a) * Real.exp x + 1) - Real.log (3 * a) - 2 * x

theorem function_properties :
  (∃ (m : ℝ), m = 2 ∧ ∀ x : ℝ, x ≥ 0 → f x ≥ m) ∧
  (∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≥ 0 ∧ x₂ ≥ 0 → g a x₁ ≤ f x₂ - 2) →
    a ≥ 1 ∧ a ≤ 3) := by sorry

end NUMINAMATH_CALUDE_function_properties_l2790_279043


namespace NUMINAMATH_CALUDE_books_not_sold_percentage_l2790_279047

def initial_stock : ℕ := 1400
def monday_sales : ℕ := 62
def tuesday_sales : ℕ := 62
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

def percentage_not_sold : ℚ := (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

theorem books_not_sold_percentage :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |percentage_not_sold - 80.57| < ε :=
sorry

end NUMINAMATH_CALUDE_books_not_sold_percentage_l2790_279047


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2790_279099

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 2 * x^2 - m * x + n = 0 ∧ 2 * y^2 - m * y + n = 0 ∧ x + y = 10 ∧ x * y = 24) →
  m + n = 68 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2790_279099


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2790_279067

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 2) :
  let set := [1 - 1 / n, 1 - 2 / n] ++ List.replicate (n - 2) 1
  List.sum set / n = 1 - 3 / n^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2790_279067


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l2790_279050

-- Define a line type
structure Line where
  slope : ℝ
  yIntercept : ℝ

-- Define the line from the problem
def problemLine : Line := { slope := -1, yIntercept := 1 }

-- Define the third quadrant
def thirdQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 < 0 ∧ p.2 < 0}

-- Theorem statement
theorem line_not_in_third_quadrant :
  ∀ (x y : ℝ), (y = problemLine.slope * x + problemLine.yIntercept) →
  (x, y) ∉ thirdQuadrant :=
sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l2790_279050


namespace NUMINAMATH_CALUDE_least_multiple_of_25_greater_than_390_l2790_279069

theorem least_multiple_of_25_greater_than_390 :
  ∀ n : ℕ, n > 0 → 25 ∣ n → n > 390 → n ≥ 400 :=
by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_25_greater_than_390_l2790_279069


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l2790_279082

theorem midpoint_coordinate_product (p1 p2 : ℝ × ℝ) :
  let m := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  p1 = (10, -3) → p2 = (-4, 9) → m.1 * m.2 = 9 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l2790_279082


namespace NUMINAMATH_CALUDE_complement_A_inter_B_when_m_is_one_one_in_A_union_B_iff_m_in_range_l2790_279015

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 9}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m + 4}

-- Part I
theorem complement_A_inter_B_when_m_is_one :
  (A ∩ B 1)ᶜ = {x | x < 3 ∨ x ≥ 6} := by sorry

-- Part II
theorem one_in_A_union_B_iff_m_in_range (m : ℝ) :
  (1 ∈ A ∪ B m) ↔ (-3/2 < m ∧ m < 0) := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_when_m_is_one_one_in_A_union_B_iff_m_in_range_l2790_279015


namespace NUMINAMATH_CALUDE_sum_of_quotient_dividend_divisor_l2790_279011

theorem sum_of_quotient_dividend_divisor : 
  ∀ (N D : ℕ), 
  N = 50 → 
  D = 5 → 
  N + D + (N / D) = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_quotient_dividend_divisor_l2790_279011


namespace NUMINAMATH_CALUDE_baron_claim_max_l2790_279007

/-- A function that returns the number of possible good sets for a given n -/
def goodSets (n : ℕ) : ℕ := 2^(n-1)

/-- The maximum n for which Baron's claim can be true -/
def maxN : ℕ := 7

/-- Theorem stating that maxN is the largest n for which Baron's claim can be true -/
theorem baron_claim_max :
  (∀ k : ℕ, k > maxN → goodSets k > 80) ∧
  (goodSets maxN ≤ 80) :=
sorry

end NUMINAMATH_CALUDE_baron_claim_max_l2790_279007


namespace NUMINAMATH_CALUDE_expression_simplification_l2790_279071

theorem expression_simplification (x : ℤ) 
  (h1 : -1 ≤ x) (h2 : x < 2) (h3 : x ≠ 1) : 
  ((x + 1) / (x^2 - 1) + x / (x - 1)) / ((x + 1) / (x^2 - 2*x + 1)) = x - 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2790_279071


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l2790_279036

theorem magic_8_ball_probability : 
  let n : ℕ := 7  -- total number of questions
  let k : ℕ := 4  -- number of positive answers we're interested in
  let p : ℚ := 1/3  -- probability of a positive answer for each question
  Nat.choose n k * p^k * (1-p)^(n-k) = 280/2187 := by sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l2790_279036


namespace NUMINAMATH_CALUDE_integer_tuple_solution_l2790_279062

theorem integer_tuple_solution : 
  ∀ (a b c : ℤ), (a - b)^3 * (a + b)^2 = c^2 + 2*(a - b) + 1 ↔ (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = -1 ∧ b = 0 ∧ c = 0) := by
  sorry

end NUMINAMATH_CALUDE_integer_tuple_solution_l2790_279062


namespace NUMINAMATH_CALUDE_point_P_coordinates_l2790_279052

def P₁ : ℝ × ℝ := (2, -1)
def P₂ : ℝ × ℝ := (0, 5)

def on_extension_line (P₁ P₂ P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ P = (t • P₂.1 + (1 - t) • P₁.1, t • P₂.2 + (1 - t) • P₁.2)

def distance_ratio (P₁ P₂ P : ℝ × ℝ) : Prop :=
  (P.1 - P₁.1)^2 + (P.2 - P₁.2)^2 = 4 * ((P₂.1 - P.1)^2 + (P₂.2 - P.2)^2)

theorem point_P_coordinates :
  ∀ P : ℝ × ℝ, on_extension_line P₁ P₂ P → distance_ratio P₁ P₂ P → P = (-2, 11) :=
by sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l2790_279052


namespace NUMINAMATH_CALUDE_perfect_square_sum_partition_not_perfect_square_sum_partition_l2790_279026

/-- A partition of a set of natural numbers -/
def Partition (n : ℕ) := Fin 2 → Finset ℕ

/-- Predicate to check if a partition satisfies the perfect square sum property -/
def HasPerfectSquareSum (p : Partition n) : Prop :=
  ∃ (i : Fin 2) (a b : ℕ), a ≠ b ∧ a ∈ p i ∧ b ∈ p i ∧ ∃ (k : ℕ), a + b = k^2

/-- The main theorem stating the property holds for all n ≥ 15 -/
theorem perfect_square_sum_partition (n : ℕ) (h : n ≥ 15) :
  ∀ (p : Partition n), HasPerfectSquareSum p :=
sorry

/-- The property does not hold for n < 15 -/
theorem not_perfect_square_sum_partition (n : ℕ) (h : n < 15) :
  ∃ (p : Partition n), ¬HasPerfectSquareSum p :=
sorry

end NUMINAMATH_CALUDE_perfect_square_sum_partition_not_perfect_square_sum_partition_l2790_279026


namespace NUMINAMATH_CALUDE_vector_magnitude_l2790_279085

-- Define the vectors
def a (x : ℝ) : Fin 2 → ℝ := ![x, 2]
def b : Fin 2 → ℝ := ![2, 1]
def c (x : ℝ) : Fin 2 → ℝ := ![3, x]

-- Define the parallel condition
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ i, v i = k * w i

-- State the theorem
theorem vector_magnitude (x : ℝ) :
  parallel (a x) b →
  ‖b + c x‖ = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l2790_279085


namespace NUMINAMATH_CALUDE_ashley_amount_l2790_279068

theorem ashley_amount (ashley betty carlos dick elgin : ℕ) : 
  ashley + betty + carlos + dick + elgin = 86 →
  ashley = betty + 20 →
  (betty = carlos + 9 ∨ carlos = betty + 9) →
  (carlos = dick + 6 ∨ dick = carlos + 6) →
  (dick = elgin + 7 ∨ elgin = dick + 7) →
  elgin = ashley + 10 →
  ashley = 24 := by sorry

end NUMINAMATH_CALUDE_ashley_amount_l2790_279068


namespace NUMINAMATH_CALUDE_sqrt_product_equals_two_l2790_279022

theorem sqrt_product_equals_two : Real.sqrt 12 * Real.sqrt (1/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_two_l2790_279022


namespace NUMINAMATH_CALUDE_intersection_points_form_line_l2790_279097

theorem intersection_points_form_line (s : ℝ) : 
  ∃ (x y : ℝ), 2*x + 3*y = 8*s + 4 ∧ 3*x - 4*y = 9*s - 3 → 
  y = (20/59)*x + 60/59 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_form_line_l2790_279097


namespace NUMINAMATH_CALUDE_sum_of_three_squares_divisibility_l2790_279058

theorem sum_of_three_squares_divisibility (N : ℕ) :
  (∃ a b c : ℤ, (N : ℤ) = a^2 + b^2 + c^2 ∧ 3 ∣ a ∧ 3 ∣ b ∧ 3 ∣ c) →
  (∃ x y z : ℤ, (N : ℤ) = x^2 + y^2 + z^2 ∧ ¬(3 ∣ x) ∧ ¬(3 ∣ y) ∧ ¬(3 ∣ z)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_divisibility_l2790_279058


namespace NUMINAMATH_CALUDE_divisor_problem_l2790_279083

theorem divisor_problem :
  ∃! d : ℕ+, d > 5 ∧
  (∃ x q : ℤ, x = q * d.val + 5) ∧
  (∃ x p : ℤ, 4 * x = p * d.val + 6) :=
sorry

end NUMINAMATH_CALUDE_divisor_problem_l2790_279083


namespace NUMINAMATH_CALUDE_protein_content_lower_bound_l2790_279000

/-- Represents the protein content of a beverage can -/
structure BeverageCan where
  netWeight : ℝ
  proteinPercentage : ℝ

/-- Theorem: Given a beverage can with net weight 300 grams and protein content ≥ 0.6%,
    the protein content is at least 1.8 grams -/
theorem protein_content_lower_bound (can : BeverageCan)
    (h1 : can.netWeight = 300)
    (h2 : can.proteinPercentage ≥ 0.6) :
    can.netWeight * (can.proteinPercentage / 100) ≥ 1.8 := by
  sorry

#check protein_content_lower_bound

end NUMINAMATH_CALUDE_protein_content_lower_bound_l2790_279000


namespace NUMINAMATH_CALUDE_cans_bought_with_euros_l2790_279049

/-- The number of cans of soda that can be bought for a given amount of euros. -/
def cans_per_euros (T R E : ℚ) : ℚ :=
  (5 * E * T) / R

/-- Given that T cans of soda can be purchased for R quarters,
    and 1 euro is equivalent to 5 quarters,
    the number of cans of soda that can be bought for E euros is (5ET)/R -/
theorem cans_bought_with_euros (T R E : ℚ) (hT : T > 0) (hR : R > 0) (hE : E ≥ 0) :
  cans_per_euros T R E = (5 * E * T) / R :=
by sorry

end NUMINAMATH_CALUDE_cans_bought_with_euros_l2790_279049


namespace NUMINAMATH_CALUDE_factorial_ratio_2017_2016_l2790_279045

-- Define factorial operation
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_ratio_2017_2016 :
  factorial 2017 / factorial 2016 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_2017_2016_l2790_279045


namespace NUMINAMATH_CALUDE_dog_purchase_cost_l2790_279074

theorem dog_purchase_cost (current_amount additional_amount : ℕ) 
  (h1 : current_amount = 34)
  (h2 : additional_amount = 13) :
  current_amount + additional_amount = 47 := by
  sorry

end NUMINAMATH_CALUDE_dog_purchase_cost_l2790_279074


namespace NUMINAMATH_CALUDE_doctor_nurse_ratio_l2790_279039

theorem doctor_nurse_ratio (total : ℕ) (nurses : ℕ) (h1 : total = 280) (h2 : nurses = 180) :
  (total - nurses) / (Nat.gcd (total - nurses) nurses) = 5 ∧
  nurses / (Nat.gcd (total - nurses) nurses) = 9 :=
by sorry

end NUMINAMATH_CALUDE_doctor_nurse_ratio_l2790_279039
