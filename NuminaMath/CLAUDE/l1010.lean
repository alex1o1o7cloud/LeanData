import Mathlib

namespace exists_number_divisible_by_5_pow_1000_no_zero_digit_l1010_101088

theorem exists_number_divisible_by_5_pow_1000_no_zero_digit :
  ∃ n : ℕ, (5^1000 ∣ n) ∧ (∀ d : ℕ, d < 10 → d ≠ 0 → ∃ k : ℕ, n / 10^k % 10 = d) :=
sorry

end exists_number_divisible_by_5_pow_1000_no_zero_digit_l1010_101088


namespace train_speed_proof_l1010_101029

theorem train_speed_proof (train_length bridge_length crossing_time : Real) 
  (h1 : train_length = 110)
  (h2 : bridge_length = 170)
  (h3 : crossing_time = 16.7986561075114) : 
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  let speed_kmph := speed_ms * 3.6
  ⌊speed_kmph⌋ = 60 := by
  sorry

end train_speed_proof_l1010_101029


namespace probability_three_white_balls_l1010_101045

/-- The probability of drawing three white balls from a box containing 7 white balls and 8 black balls is 1/13. -/
theorem probability_three_white_balls (white_balls black_balls : ℕ) 
  (h1 : white_balls = 7) (h2 : black_balls = 8) : 
  (Nat.choose white_balls 3 : ℚ) / (Nat.choose (white_balls + black_balls) 3) = 1 / 13 := by
  sorry

#eval Nat.choose 7 3
#eval Nat.choose 15 3
#eval (35 : ℚ) / 455

end probability_three_white_balls_l1010_101045


namespace min_value_of_linear_function_l1010_101085

theorem min_value_of_linear_function :
  ∃ (m : ℝ), ∀ (x y : ℝ), 2*x + 3*y ≥ m ∧ ∃ (x₀ y₀ : ℝ), 2*x₀ + 3*y₀ = m :=
by sorry

end min_value_of_linear_function_l1010_101085


namespace initial_population_proof_l1010_101041

/-- The population change function over 5 years -/
def population_change (P : ℝ) : ℝ :=
  P * 0.9 * 1.1 * 0.9 * 1.15 * 0.75

/-- Theorem stating the initial population given the final population -/
theorem initial_population_proof : 
  ∃ P : ℕ, population_change (P : ℝ) = 4455 ∧ P = 5798 :=
sorry

end initial_population_proof_l1010_101041


namespace lemonade_stand_revenue_calculation_l1010_101077

/-- Calculates the gross revenue of a lemonade stand given total profit, babysitting income, and lemonade stand expenses. -/
def lemonade_stand_revenue (total_profit babysitting_income lemonade_expenses : ℤ) : ℤ :=
  (total_profit - babysitting_income) + lemonade_expenses

theorem lemonade_stand_revenue_calculation :
  lemonade_stand_revenue 44 31 34 = 47 := by
  sorry

end lemonade_stand_revenue_calculation_l1010_101077


namespace three_number_average_l1010_101078

theorem three_number_average (a b c : ℝ) 
  (h1 : a = 2 * b) 
  (h2 : a = 3 * c) 
  (h3 : a - c = 96) : 
  (a + b + c) / 3 = 88 := by
  sorry

end three_number_average_l1010_101078


namespace function_is_constant_one_l1010_101042

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ a b : ℕ, a > 0 ∧ b > 0 → f (a^2 + b^2) = f a * f b) ∧
  (∀ a : ℕ, a > 0 → f (a^2) = (f a)^2)

theorem function_is_constant_one (f : ℕ → ℕ) (h : is_valid_function f) :
  ∀ n : ℕ, n > 0 → f n = 1 :=
by sorry

end function_is_constant_one_l1010_101042


namespace probability_of_white_marble_l1010_101023

/-- Given a box of marbles with four colors, prove the probability of drawing a white marble. -/
theorem probability_of_white_marble (total_marbles : ℕ) 
  (p_green p_red_or_blue p_white : ℝ) : 
  total_marbles = 100 →
  p_green = 1/5 →
  p_red_or_blue = 0.55 →
  p_green + p_red_or_blue + p_white = 1 →
  p_white = 0.25 := by
  sorry

#check probability_of_white_marble

end probability_of_white_marble_l1010_101023


namespace arithmetic_expressions_correctness_l1010_101096

theorem arithmetic_expressions_correctness :
  (∀ a b c : ℚ, (a + b) + c = a + (b + c)) ∧
  (∃ a b c : ℚ, (a - b) - c ≠ a - (b - c)) ∧
  (∃ a b c : ℚ, (a + b) / c ≠ a + (b / c)) ∧
  (∃ a b c : ℚ, (a / b) / c ≠ a / (b / c)) :=
by sorry

end arithmetic_expressions_correctness_l1010_101096


namespace minimum_employees_needed_l1010_101055

/-- Represents the set of employees monitoring water pollution -/
def W : Finset Nat := sorry

/-- Represents the set of employees monitoring air pollution -/
def A : Finset Nat := sorry

/-- Represents the set of employees monitoring land pollution -/
def L : Finset Nat := sorry

theorem minimum_employees_needed : 
  (Finset.card W = 95) → 
  (Finset.card A = 80) → 
  (Finset.card L = 50) → 
  (Finset.card (W ∩ A) = 30) → 
  (Finset.card (A ∩ L) = 20) → 
  (Finset.card (W ∩ L) = 15) → 
  (Finset.card (W ∩ A ∩ L) = 10) → 
  Finset.card (W ∪ A ∪ L) = 170 := by
  sorry

end minimum_employees_needed_l1010_101055


namespace square_of_104_l1010_101005

theorem square_of_104 : (104 : ℕ)^2 = 10816 := by sorry

end square_of_104_l1010_101005


namespace trapezoid_perimeter_is_230_l1010_101087

/-- Represents a trapezoid ABCD with given properties -/
structure Trapezoid where
  BC : ℝ
  AP : ℝ
  DQ : ℝ
  AB : ℝ
  CD : ℝ
  AD_longer_than_BC : BC < AP + BC + DQ

/-- Calculates the perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.AB + t.BC + t.CD + (t.AP + t.BC + t.DQ)

/-- Theorem stating that the perimeter of the given trapezoid is 230 -/
theorem trapezoid_perimeter_is_230 (t : Trapezoid) 
    (h1 : t.BC = 60)
    (h2 : t.AP = 24)
    (h3 : t.DQ = 11)
    (h4 : t.AB = 40)
    (h5 : t.CD = 35) : 
  perimeter t = 230 := by
  sorry

#check trapezoid_perimeter_is_230

end trapezoid_perimeter_is_230_l1010_101087


namespace sqrt_of_sqrt_81_l1010_101090

theorem sqrt_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 3 := by sorry

end sqrt_of_sqrt_81_l1010_101090


namespace correct_operation_l1010_101062

theorem correct_operation (a : ℝ) : (-a + 2) * (-a - 2) = a^2 - 4 := by
  sorry

end correct_operation_l1010_101062


namespace mango_purchase_proof_l1010_101097

def grape_quantity : ℕ := 11
def grape_price : ℕ := 98
def mango_price : ℕ := 50
def total_payment : ℕ := 1428

def mango_quantity : ℕ := (total_payment - grape_quantity * grape_price) / mango_price

theorem mango_purchase_proof : mango_quantity = 7 := by
  sorry

end mango_purchase_proof_l1010_101097


namespace complex_second_quadrant_l1010_101009

theorem complex_second_quadrant (a : ℝ) : 
  let z : ℂ := a^2 * (1 + Complex.I) - a * (4 + Complex.I) - 6 * Complex.I
  (z.re < 0 ∧ z.im > 0) → (3 < a ∧ a < 4) := by
  sorry

end complex_second_quadrant_l1010_101009


namespace hyperbola_eccentricity_l1010_101027

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    if the distance from (4, 0) to its asymptote is √2,
    then its eccentricity is (2√14)/7 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (4 * b / Real.sqrt (a^2 + b^2) = Real.sqrt 2) →
  (Real.sqrt (a^2 + b^2) / a = 2 * Real.sqrt 14 / 7) :=
by sorry

end hyperbola_eccentricity_l1010_101027


namespace hill_climbing_time_l1010_101057

theorem hill_climbing_time 
  (descent_time : ℝ) 
  (average_speed_total : ℝ) 
  (average_speed_climbing : ℝ) 
  (h1 : descent_time = 2)
  (h2 : average_speed_total = 3.5)
  (h3 : average_speed_climbing = 2.625) :
  ∃ (climb_time : ℝ), 
    climb_time = 4 ∧ 
    average_speed_total = (2 * average_speed_climbing * climb_time) / (climb_time + descent_time) := by
  sorry

end hill_climbing_time_l1010_101057


namespace g_difference_l1010_101061

/-- Given a function g(n) = 1/2 * n^2 * (n+3), prove that g(s) - g(s-1) = 1/2 * (3s - 2) for any real number s. -/
theorem g_difference (s : ℝ) : 
  let g : ℝ → ℝ := λ n => (1/2) * n^2 * (n + 3)
  g s - g (s - 1) = (1/2) * (3*s - 2) := by
sorry

end g_difference_l1010_101061


namespace complement_intersection_theorem_l1010_101016

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x| ≥ 1}
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- Define the complement of A and B with respect to ℝ
def C_UA : Set ℝ := {x : ℝ | x ∉ A}
def C_UB : Set ℝ := {x : ℝ | x ∉ B}

-- Theorem statement
theorem complement_intersection_theorem :
  (C_UA ∩ C_UB) = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end complement_intersection_theorem_l1010_101016


namespace least_K_inequality_l1010_101067

theorem least_K_inequality (K : ℝ) : (∀ x y : ℝ, (1 + 20 * x^2) * (1 + 19 * y^2) ≥ K * x * y) ↔ K ≤ 8 * Real.sqrt 95 := by
  sorry

end least_K_inequality_l1010_101067


namespace wire_length_ratio_l1010_101064

/-- The ratio of wire lengths in cube frame construction -/
theorem wire_length_ratio (bonnie_wire_pieces : ℕ) (bonnie_wire_length : ℝ) 
  (roark_wire_length : ℝ) : 
  bonnie_wire_pieces = 12 →
  bonnie_wire_length = 8 →
  roark_wire_length = 0.5 →
  (bonnie_wire_length ^ 3) * (roark_wire_length ^ 3)⁻¹ * 
    (12 * roark_wire_length) * (bonnie_wire_pieces * bonnie_wire_length)⁻¹ = 256 →
  (bonnie_wire_pieces * bonnie_wire_length) * 
    ((bonnie_wire_length ^ 3) * (roark_wire_length ^ 3)⁻¹ * (12 * roark_wire_length))⁻¹ = 1 / 256 := by
  sorry

end wire_length_ratio_l1010_101064


namespace division_remainder_problem_l1010_101040

theorem division_remainder_problem : ∃ (r : ℕ), 15968 = 179 * 89 + r ∧ r < 179 := by
  sorry

end division_remainder_problem_l1010_101040


namespace smallest_angle_measure_l1010_101084

theorem smallest_angle_measure (ABC ABD : ℝ) (h1 : ABC = 40) (h2 : ABD = 30) :
  ∃ (CBD : ℝ), CBD = ABC - ABD ∧ CBD = 10 ∧ ∀ (x : ℝ), x ≥ 0 → x ≥ CBD :=
by sorry

end smallest_angle_measure_l1010_101084


namespace difference_calculation_l1010_101032

theorem difference_calculation (x y : ℝ) (hx : x = 497) (hy : y = 325) :
  2/5 * (3*x + 7*y) - 3/5 * (x * y) = -95408.6 := by
  sorry

end difference_calculation_l1010_101032


namespace scientific_notation_equality_l1010_101001

theorem scientific_notation_equality : (58000000000 : ℝ) = 5.8 * (10 ^ 10) := by
  sorry

end scientific_notation_equality_l1010_101001


namespace cos_thirteen_pi_fourths_l1010_101007

theorem cos_thirteen_pi_fourths : Real.cos (13 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end cos_thirteen_pi_fourths_l1010_101007


namespace simplify_expression_l1010_101073

theorem simplify_expression (z : ℝ) : 3 * (4 - 5 * z) - 2 * (2 + 3 * z) = 8 - 21 * z := by
  sorry

end simplify_expression_l1010_101073


namespace ellipse_t_squared_range_l1010_101093

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define point H
def H : ℝ × ℝ := (3, 0)

-- Define the condition for points A and B
def intersects_ellipse (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ∃ k : ℝ, A.2 - B.2 = k * (A.1 - B.1) ∧ A.2 = k * (A.1 - H.1) + H.2

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Define the vector relation
def vector_relation (O A B P : ℝ × ℝ) (t : ℝ) : Prop :=
  (A.1 - O.1, A.2 - O.2) + (B.1 - O.1, B.2 - O.2) = t • (P.1 - O.1, P.2 - O.2)

-- Define the distance condition
def distance_condition (P A B : ℝ × ℝ) : Prop :=
  ((P.1 - A.1)^2 + (P.2 - A.2)^2)^(1/2) - ((P.1 - B.1)^2 + (P.2 - B.2)^2)^(1/2) < Real.sqrt 3

theorem ellipse_t_squared_range :
  ∀ (O A B P : ℝ × ℝ) (t : ℝ),
    intersects_ellipse A B →
    P_condition P →
    vector_relation O A B P t →
    distance_condition P A B →
    20 - Real.sqrt 283 < t^2 ∧ t^2 < 4 :=
sorry

end ellipse_t_squared_range_l1010_101093


namespace team_size_is_five_l1010_101094

/-- The length of the relay race in meters -/
def relay_length : ℕ := 150

/-- The distance each team member runs in meters -/
def member_distance : ℕ := 30

/-- The number of people on the team -/
def team_size : ℕ := relay_length / member_distance

theorem team_size_is_five : team_size = 5 := by
  sorry

end team_size_is_five_l1010_101094


namespace total_legs_in_room_l1010_101039

/-- Represents the count of furniture items with their respective leg counts -/
structure FurnitureCount where
  four_leg_tables : ℕ
  four_leg_sofas : ℕ
  four_leg_chairs : ℕ
  three_leg_tables : ℕ
  one_leg_tables : ℕ
  two_leg_rocking_chairs : ℕ

/-- Calculates the total number of legs in the room -/
def total_legs (fc : FurnitureCount) : ℕ :=
  4 * fc.four_leg_tables +
  4 * fc.four_leg_sofas +
  4 * fc.four_leg_chairs +
  3 * fc.three_leg_tables +
  1 * fc.one_leg_tables +
  2 * fc.two_leg_rocking_chairs

/-- The given furniture configuration in the room -/
def room_furniture : FurnitureCount :=
  { four_leg_tables := 4
  , four_leg_sofas := 1
  , four_leg_chairs := 2
  , three_leg_tables := 3
  , one_leg_tables := 1
  , two_leg_rocking_chairs := 1 }

/-- Theorem stating that the total number of legs in the room is 40 -/
theorem total_legs_in_room : total_legs room_furniture = 40 := by
  sorry

end total_legs_in_room_l1010_101039


namespace max_positive_terms_is_seven_l1010_101003

/-- An arithmetic sequence with a positive first term where the sum of the first 3 terms 
    equals the sum of the first 11 terms. -/
structure ArithmeticSequence where
  a₁ : ℝ
  d : ℝ
  first_term_positive : 0 < a₁
  sum_equality : 3 * (2 * a₁ + 2 * d) = 11 * (2 * a₁ + 10 * d)

/-- The maximum number of terms that can be summed before reaching a non-positive term -/
def max_positive_terms (seq : ArithmeticSequence) : ℕ :=
  7

/-- Theorem stating that the maximum number of terms is correct -/
theorem max_positive_terms_is_seven (seq : ArithmeticSequence) :
  (max_positive_terms seq = 7) ∧
  (∀ n : ℕ, n ≤ 7 → seq.a₁ + (n - 1) * seq.d > 0) ∧
  (seq.a₁ + 7 * seq.d ≤ 0) :=
sorry

end max_positive_terms_is_seven_l1010_101003


namespace quadratic_equation_two_distinct_roots_l1010_101035

theorem quadratic_equation_two_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (∀ x : ℝ, (x - 1) * (x + 5) = 3 * x + 1 ↔ x = x₁ ∨ x = x₂) :=
sorry

end quadratic_equation_two_distinct_roots_l1010_101035


namespace companion_pair_expression_zero_l1010_101098

/-- Definition of companion number pairs -/
def is_companion_pair (a b : ℚ) : Prop :=
  a / 2 + b / 3 = (a + b) / 5

/-- Theorem: For any companion number pair (m,n), 
    the expression 14m-5n-[5m-3(3n-1)]+3 always evaluates to 0 -/
theorem companion_pair_expression_zero (m n : ℚ) 
  (h : is_companion_pair m n) : 
  14*m - 5*n - (5*m - 3*(3*n - 1)) + 3 = 0 := by
  sorry

end companion_pair_expression_zero_l1010_101098


namespace perfect_square_expression_l1010_101024

theorem perfect_square_expression (x y z : ℤ) :
  9 * (x^2 + y^2 + z^2)^2 - 8 * (x + y + z) * (x^3 + y^3 + z^3 - 3*x*y*z) =
  ((x + y + z)^2 - 6*(x*y + y*z + z*x))^2 := by
  sorry

end perfect_square_expression_l1010_101024


namespace single_point_ellipse_l1010_101018

/-- 
If the graph of 3x^2 + 4y^2 + 6x - 8y + c = 0 consists of a single point, then c = 7.
-/
theorem single_point_ellipse (c : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + 4 * p.2^2 + 6 * p.1 - 8 * p.2 + c = 0) → c = 7 := by
  sorry

end single_point_ellipse_l1010_101018


namespace greatest_four_digit_multiple_of_17_l1010_101050

theorem greatest_four_digit_multiple_of_17 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 17 ∣ n → n ≤ 9996 :=
by sorry

end greatest_four_digit_multiple_of_17_l1010_101050


namespace triangle_properties_l1010_101089

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  Real.cos C = 1 / 3 →
  c = 4 * Real.sqrt 2 →
  A = π / 3 ∧ 
  (1 / 2 * a * c * Real.sin B) = 4 * Real.sqrt 3 + 3 * Real.sqrt 2 :=
by sorry

end triangle_properties_l1010_101089


namespace coin_problem_l1010_101070

theorem coin_problem :
  let total_coins : ℕ := 56
  let total_value : ℕ := 440
  let coins_of_one_type : ℕ := 24
  let x : ℕ := total_coins - coins_of_one_type  -- number of 10-peso coins
  let y : ℕ := coins_of_one_type  -- number of 5-peso coins
  (x + y = total_coins) ∧ (10 * x + 5 * y = total_value) → y = 24 :=
by
  sorry

end coin_problem_l1010_101070


namespace triangle_incenter_distance_l1010_101038

/-- A triangle with sides a, b, and c, and incenter J -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  J : ℝ × ℝ

/-- The incircle of a triangle -/
structure Incircle where
  center : ℝ × ℝ
  radius : ℝ

/-- Given a triangle PQR with sides PQ = 30, PR = 29, and QR = 31,
    and J as the intersection of internal angle bisectors (incenter),
    prove that QJ = √(226 - r²), where r is the radius of the incircle -/
theorem triangle_incenter_distance (T : Triangle) (I : Incircle) :
  T.a = 30 ∧ T.b = 29 ∧ T.c = 31 ∧ 
  I.center = T.J ∧
  I.radius = r →
  ∃ (QJ : ℝ), QJ = Real.sqrt (226 - r^2) :=
sorry

end triangle_incenter_distance_l1010_101038


namespace first_year_exceeding_2_million_is_correct_l1010_101015

/-- The year when the R&D investment first exceeds 2 million yuan -/
def first_year_exceeding_2_million : ℕ := 2020

/-- The initial R&D investment in 2016 (in millions of yuan) -/
def initial_investment : ℝ := 1.3

/-- The annual increase rate of R&D investment -/
def annual_increase_rate : ℝ := 0.12

/-- The target R&D investment (in millions of yuan) -/
def target_investment : ℝ := 2.0

/-- Function to calculate the R&D investment for a given year -/
def investment_for_year (year : ℕ) : ℝ :=
  initial_investment * (1 + annual_increase_rate) ^ (year - 2016)

theorem first_year_exceeding_2_million_is_correct :
  (∀ y : ℕ, y < first_year_exceeding_2_million → investment_for_year y ≤ target_investment) ∧
  investment_for_year first_year_exceeding_2_million > target_investment :=
by sorry

end first_year_exceeding_2_million_is_correct_l1010_101015


namespace largest_n_for_trig_inequality_l1010_101030

theorem largest_n_for_trig_inequality : 
  (∃ (n : ℕ), n > 0 ∧ ∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 2/n) ∧ 
  (∀ (n : ℕ), n > 2 → ∃ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n < 2/n) :=
by sorry

end largest_n_for_trig_inequality_l1010_101030


namespace equation_has_four_solutions_l1010_101025

-- Define the equation
def equation (x : ℝ) : Prop := (3 * x^2 - 8)^2 = 49

-- Define a function that counts the number of distinct real solutions
def count_solutions : ℕ := sorry

-- Theorem statement
theorem equation_has_four_solutions : count_solutions = 4 := by sorry

end equation_has_four_solutions_l1010_101025


namespace lcm_of_18_28_45_65_l1010_101012

theorem lcm_of_18_28_45_65 : Nat.lcm 18 (Nat.lcm 28 (Nat.lcm 45 65)) = 16380 := by
  sorry

end lcm_of_18_28_45_65_l1010_101012


namespace isosceles_triangles_same_perimeter_l1010_101074

-- Define the properties of the triangles
def isIsosceles (a b c : ℝ) : Prop := (a = b) ∨ (b = c) ∨ (a = c)
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Define the theorem
theorem isosceles_triangles_same_perimeter 
  (c d : ℝ) 
  (h1 : isIsosceles 7 7 10) 
  (h2 : isIsosceles c c d) 
  (h3 : c ≠ d) 
  (h4 : perimeter 7 7 10 = 24) 
  (h5 : perimeter c c d = 24) :
  d = 2 := by sorry

end isosceles_triangles_same_perimeter_l1010_101074


namespace circle_area_ratio_l1010_101008

theorem circle_area_ratio (R S : ℝ) (hR : R > 0) (hS : S > 0) (h : R = 0.2 * S) :
  (π * (R / 2)^2) / (π * (S / 2)^2) = 0.04 := by
  sorry

end circle_area_ratio_l1010_101008


namespace complex_fraction_simplification_l1010_101083

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 : ℂ) / (1 + i) = 1 - i :=
by sorry

end complex_fraction_simplification_l1010_101083


namespace trout_calculation_l1010_101072

/-- The number of people fishing -/
def num_people : ℕ := 2

/-- The number of trout each person gets after splitting -/
def trout_per_person : ℕ := 9

/-- The total number of trout caught -/
def total_trout : ℕ := num_people * trout_per_person

theorem trout_calculation : total_trout = 18 := by
  sorry

end trout_calculation_l1010_101072


namespace acid_dilution_l1010_101000

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution results in a 25% acid solution. -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 →
  initial_concentration = 0.40 →
  added_water = 30 →
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + added_water) = final_concentration :=
by
  sorry

end acid_dilution_l1010_101000


namespace water_and_bottle_weights_l1010_101026

/-- The weight of one cup of water in grams -/
def cup_weight : ℝ := 80

/-- The weight of one empty bottle in grams -/
def bottle_weight : ℝ := 200

/-- The total weight of 3 cups of water and 1 empty bottle in grams -/
def weight_3cups_1bottle : ℝ := 440

/-- The total weight of 5 cups of water and 1 empty bottle in grams -/
def weight_5cups_1bottle : ℝ := 600

theorem water_and_bottle_weights :
  (3 * cup_weight + bottle_weight = weight_3cups_1bottle) ∧
  (5 * cup_weight + bottle_weight = weight_5cups_1bottle) := by
  sorry

end water_and_bottle_weights_l1010_101026


namespace corrected_mean_l1010_101065

def number_of_observations : ℕ := 100
def original_mean : ℝ := 125.6
def incorrect_observation1 : ℝ := 95.3
def incorrect_observation2 : ℝ := -15.9
def correct_observation1 : ℝ := 48.2
def correct_observation2 : ℝ := -35.7

theorem corrected_mean (n : ℕ) (om : ℝ) (io1 io2 co1 co2 : ℝ) :
  n = number_of_observations ∧
  om = original_mean ∧
  io1 = incorrect_observation1 ∧
  io2 = incorrect_observation2 ∧
  co1 = correct_observation1 ∧
  co2 = correct_observation2 →
  (n : ℝ) * om - (io1 + io2) + (co1 + co2) / n = 124.931 := by
  sorry

end corrected_mean_l1010_101065


namespace no_divisible_seven_digit_numbers_l1010_101056

/-- A function that checks if a number uses each of the digits 1-7 exactly once. -/
def usesDigits1To7Once (n : ℕ) : Prop :=
  ∃ (a b c d e f g : ℕ),
    n = a * 1000000 + b * 100000 + c * 10000 + d * 1000 + e * 100 + f * 10 + g ∧
    ({a, b, c, d, e, f, g} : Finset ℕ) = {1, 2, 3, 4, 5, 6, 7}

/-- Theorem stating that there are no two seven-digit numbers formed using
    digits 1-7 once each where one divides the other. -/
theorem no_divisible_seven_digit_numbers :
  ¬∃ (m n : ℕ), m ≠ n ∧ 
    usesDigits1To7Once m ∧ 
    usesDigits1To7Once n ∧ 
    m ∣ n :=
by sorry

end no_divisible_seven_digit_numbers_l1010_101056


namespace complex_equation_solution_l1010_101031

theorem complex_equation_solution (z : ℂ) :
  z * (1 - Complex.I) = 2 * Complex.I → z = -1 + Complex.I := by
  sorry

end complex_equation_solution_l1010_101031


namespace arcsin_equation_solution_l1010_101051

theorem arcsin_equation_solution (x : ℝ) : 
  Real.arcsin (3 * x) - Real.arcsin x = π / 6 → 
  x = 1 / Real.sqrt (40 - 12 * Real.sqrt 3) ∨ 
  x = -1 / Real.sqrt (40 - 12 * Real.sqrt 3) := by
sorry

end arcsin_equation_solution_l1010_101051


namespace complex_modulus_l1010_101079

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = 1 - 2 * Complex.I^3) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
sorry

end complex_modulus_l1010_101079


namespace fraction_sum_l1010_101069

theorem fraction_sum (a b : ℚ) (h : a / b = 1 / 3) : (a + b) / b = 4 / 3 := by
  sorry

end fraction_sum_l1010_101069


namespace group_size_l1010_101036

/-- The number of people in a group, given certain weight changes. -/
theorem group_size (avg_increase : ℝ) (old_weight new_weight : ℝ) (h1 : avg_increase = 1.5)
    (h2 : new_weight - old_weight = 6) : ℤ :=
  4

#check group_size

end group_size_l1010_101036


namespace family_ages_solution_l1010_101006

/-- Represents the ages of a family members -/
structure FamilyAges where
  son : ℕ
  daughter : ℕ
  man : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.man = ages.son + 20 ∧
  ages.man = ages.daughter + 15 ∧
  ages.man + 2 = 2 * (ages.son + 2) ∧
  ages.man + 2 = 3 * (ages.daughter + 2)

/-- Theorem stating that the given ages satisfy the problem conditions -/
theorem family_ages_solution :
  ∃ (ages : FamilyAges), satisfiesConditions ages ∧
    ages.son = 18 ∧ ages.daughter = 23 ∧ ages.man = 38 := by
  sorry

end family_ages_solution_l1010_101006


namespace subtraction_divisibility_implies_sum_l1010_101022

/-- Represents a three-digit number in the form xyz --/
structure ThreeDigitNumber where
  x : Nat
  y : Nat
  z : Nat
  x_nonzero : x ≠ 0
  digits_bound : x < 10 ∧ y < 10 ∧ z < 10

/-- Converts a ThreeDigitNumber to its numerical value --/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.x + 10 * n.y + n.z

theorem subtraction_divisibility_implies_sum (a b : Nat) :
  ∃ (num1 num2 : ThreeDigitNumber),
    num1.toNat = 407 + 10 * a ∧
    num2.toNat = 304 + 10 * b ∧
    830 - num1.toNat = num2.toNat ∧
    num2.toNat % 7 = 0 →
    a + b = 2 := by
  sorry

end subtraction_divisibility_implies_sum_l1010_101022


namespace domain_of_function_1_l1010_101033

theorem domain_of_function_1 (x : ℝ) : 
  (x ≥ 1 ∨ x < -1) ↔ (x ≠ -1 ∧ (x - 1) / (x + 1) ≥ 0) :=
sorry

#check domain_of_function_1

end domain_of_function_1_l1010_101033


namespace equation_solution_l1010_101044

theorem equation_solution (x : ℝ) : 5 * x^2 + 4 = 3 * x + 9 → (10 * x - 3)^2 = 109 := by
  sorry

end equation_solution_l1010_101044


namespace problem_1_l1010_101095

theorem problem_1 : -36 * (3/4 - 1/6 + 2/9 - 5/12) + |(-21/5) / (7/25)| = 61 := by
  sorry

end problem_1_l1010_101095


namespace f_is_power_function_l1010_101076

-- Define what a power function is
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Define the function we want to prove is a power function
def f (x : ℝ) : ℝ := x ^ (1/2)

-- Theorem statement
theorem f_is_power_function : is_power_function f := by
  sorry

end f_is_power_function_l1010_101076


namespace quadratic_real_root_condition_l1010_101049

/-- A quadratic equation x^2 + bx + 25 = 0 has at least one real root
    if and only if b ∈ (-∞, -10] ∪ [10, ∞) -/
theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by sorry

end quadratic_real_root_condition_l1010_101049


namespace abc_sum_mod_7_l1010_101053

theorem abc_sum_mod_7 (a b c : ℕ) : 
  0 < a ∧ a < 7 ∧ 
  0 < b ∧ b < 7 ∧ 
  0 < c ∧ c < 7 ∧ 
  (a * b * c) % 7 = 2 ∧ 
  (4 * c) % 7 = 3 ∧ 
  (7 * b) % 7 = (4 + b) % 7 → 
  (a + b + c) % 7 = 6 := by
sorry

end abc_sum_mod_7_l1010_101053


namespace factor_expression_l1010_101002

theorem factor_expression (x y z : ℝ) :
  (y^2 - z^2) * (1 + x*y) * (1 + x*z) + 
  (z^2 - x^2) * (1 + y*z) * (1 + x*y) + 
  (x^2 - y^2) * (1 + y*z) * (1 + x*z) = 
  (y - z) * (z - x) * (x - y) * (x*y*z + x + y + z) := by
  sorry

end factor_expression_l1010_101002


namespace value_of_expression_l1010_101017

theorem value_of_expression (m n : ℤ) (h : m - n = 2) : 
  (n - m)^3 - (m - n)^2 + 1 = -11 := by
sorry

end value_of_expression_l1010_101017


namespace quadratic_roots_l1010_101034

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = 3 ∧ x₂ = -1 ∧ 
  (x₁^2 - 2*x₁ - 3 = 0) ∧ (x₂^2 - 2*x₂ - 3 = 0) := by
  sorry

end quadratic_roots_l1010_101034


namespace frank_hamburger_sales_l1010_101019

/-- The number of additional hamburgers Frank needs to sell to reach his target revenue -/
def additional_hamburgers (target_revenue : ℕ) (price_per_hamburger : ℕ) (initial_sales : ℕ) : ℕ :=
  (target_revenue - price_per_hamburger * initial_sales) / price_per_hamburger

theorem frank_hamburger_sales : additional_hamburgers 50 5 6 = 4 := by
  sorry

end frank_hamburger_sales_l1010_101019


namespace smallest_number_divisible_l1010_101071

theorem smallest_number_divisible (n : ℕ) : n ≥ 58 →
  (∃ k : ℕ, n - 10 = 24 * k) →
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 10 = 24 * k)) :=
by sorry

end smallest_number_divisible_l1010_101071


namespace circle_in_square_area_ratio_l1010_101059

/-- The ratio of the area of a circle inscribed in a square 
    (where the circle's diameter is equal to the square's side length) 
    to the area of the square is π/4. -/
theorem circle_in_square_area_ratio : 
  ∀ s : ℝ, s > 0 → (π * (s/2)^2) / (s^2) = π/4 := by
sorry

end circle_in_square_area_ratio_l1010_101059


namespace max_candy_pieces_l1010_101066

theorem max_candy_pieces (n : ℕ) (avg : ℕ) (min_pieces : ℕ) :
  n = 30 →
  avg = 7 →
  min_pieces = 1 →
  ∃ (max_pieces : ℕ), max_pieces = n * avg - (n - 1) * min_pieces ∧
                       max_pieces = 181 :=
by sorry

end max_candy_pieces_l1010_101066


namespace chocolate_bars_per_box_l1010_101082

theorem chocolate_bars_per_box (total_bars : ℕ) (total_boxes : ℕ) (bars_per_box : ℕ) :
  total_bars = 475 →
  total_boxes = 19 →
  total_bars = total_boxes * bars_per_box →
  bars_per_box = 25 := by
  sorry

end chocolate_bars_per_box_l1010_101082


namespace quadrant_I_solution_l1010_101013

theorem quadrant_I_solution (c : ℝ) :
  (∃ x y : ℝ, x - y = 2 ∧ c * x + y = 3 ∧ x > 0 ∧ y > 0) ↔ -1 < c ∧ c < 3/2 := by
  sorry

end quadrant_I_solution_l1010_101013


namespace special_rectangle_AB_length_l1010_101028

/-- Represents a rectangle with specific properties -/
structure SpecialRectangle where
  AB : ℝ
  BC : ℝ
  PQ : ℝ
  XY : ℝ
  equalAreas : Bool
  PQparallelAB : Bool
  XYequation : Bool

/-- The theorem statement -/
theorem special_rectangle_AB_length
  (rect : SpecialRectangle)
  (h1 : rect.BC = 19)
  (h2 : rect.PQ = 87)
  (h3 : rect.equalAreas)
  (h4 : rect.PQparallelAB)
  (h5 : rect.XYequation) :
  rect.AB = 193 := by
  sorry

end special_rectangle_AB_length_l1010_101028


namespace coin_toss_probability_l1010_101086

theorem coin_toss_probability (n : ℕ) : (∀ k : ℕ, k < n → 1 - (1/2)^k < 15/16) ∧ 1 - (1/2)^n ≥ 15/16 → n = 4 := by
  sorry

end coin_toss_probability_l1010_101086


namespace average_b_c_is_fifty_l1010_101014

theorem average_b_c_is_fifty (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : c - a = 10) :
  (b + c) / 2 = 50 := by
sorry

end average_b_c_is_fifty_l1010_101014


namespace square_side_length_average_l1010_101063

theorem square_side_length_average (a b c : ℝ) (ha : a = 25) (hb : b = 64) (hc : c = 121) :
  (a.sqrt + b.sqrt + c.sqrt) / 3 = 8 := by
  sorry

end square_side_length_average_l1010_101063


namespace first_day_is_thursday_l1010_101047

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in a month -/
structure MonthDay where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the previous day of the week -/
def prevDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

/-- Theorem: If the 24th day of a month is a Saturday, then the 1st day of that month is a Thursday -/
theorem first_day_is_thursday (m : MonthDay) (h : m.day = 24 ∧ m.dayOfWeek = DayOfWeek.Saturday) :
  ∃ (firstDay : MonthDay), firstDay.day = 1 ∧ firstDay.dayOfWeek = DayOfWeek.Thursday :=
by sorry

end first_day_is_thursday_l1010_101047


namespace juniors_percentage_l1010_101011

/-- Represents the composition of students in a high school sample. -/
structure StudentSample where
  total : ℕ
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- Calculates the percentage of a part relative to the total. -/
def percentage (part : ℕ) (total : ℕ) : ℚ :=
  (part : ℚ) / (total : ℚ) * 100

/-- Theorem stating the percentage of juniors in the given student sample. -/
theorem juniors_percentage (sample : StudentSample) : 
  sample.total = 800 ∧ 
  sample.seniors = 160 ∧
  sample.sophomores = sample.total / 4 ∧
  sample.freshmen = sample.sophomores + 24 ∧
  sample.total = sample.freshmen + sample.sophomores + sample.juniors + sample.seniors →
  percentage sample.juniors sample.total = 27 := by
  sorry

end juniors_percentage_l1010_101011


namespace completing_square_equivalence_l1010_101080

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) := by
  sorry

end completing_square_equivalence_l1010_101080


namespace negation_of_existence_is_forall_l1010_101020

theorem negation_of_existence_is_forall :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) := by
  sorry

end negation_of_existence_is_forall_l1010_101020


namespace fraction_simplification_l1010_101021

theorem fraction_simplification (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := by
  sorry

end fraction_simplification_l1010_101021


namespace forgotten_angle_measure_l1010_101052

theorem forgotten_angle_measure (n : ℕ) (h : n > 2) :
  (n - 1) * 180 - 2017 = 143 :=
sorry

end forgotten_angle_measure_l1010_101052


namespace midpoint_octagon_area_ratio_l1010_101054

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry  -- Additional properties to ensure the octagon is regular

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
def midpointOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- The theorem stating that the area of the midpoint octagon is 3/4 of the original octagon -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpointOctagon o) = (3/4) * area o :=
sorry

end midpoint_octagon_area_ratio_l1010_101054


namespace prob_red_then_black_is_three_fourths_l1010_101099

/-- A deck of cards with red and black cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)
  (h_total : total_cards = red_cards + black_cards)
  (h_equal : red_cards = black_cards)

/-- The probability of drawing a red card first and a black card second -/
def prob_red_then_black (d : Deck) : ℚ :=
  (d.red_cards : ℚ) * d.black_cards / (d.total_cards * (d.total_cards - 1))

/-- Theorem: For a deck with 64 cards, half red and half black,
    the probability of drawing a red card first and a black card second is 3/4 -/
theorem prob_red_then_black_is_three_fourths (d : Deck) 
    (h_total : d.total_cards = 64) : prob_red_then_black d = 3/4 := by
  sorry

end prob_red_then_black_is_three_fourths_l1010_101099


namespace analysis_method_seeks_sufficient_conditions_l1010_101058

/-- The analysis method in mathematical proofs -/
def analysis_method (method : String) : Prop :=
  method = "starts from the conclusion to be proven and progressively seeks conditions that make the conclusion valid"

/-- The type of conditions sought by a proof method -/
inductive ConditionType
  | Sufficient
  | Necessary
  | NecessaryAndSufficient
  | Equivalent

/-- The conditions sought by a proof method -/
def seeks_conditions (method : String) (condition_type : ConditionType) : Prop :=
  analysis_method method → condition_type = ConditionType.Sufficient

theorem analysis_method_seeks_sufficient_conditions :
  ∀ (method : String),
  analysis_method method →
  seeks_conditions method ConditionType.Sufficient :=
by
  sorry


end analysis_method_seeks_sufficient_conditions_l1010_101058


namespace polynomial_factorization_l1010_101081

theorem polynomial_factorization (t : ℝ) :
  ∃ (a b c d : ℝ), ∀ (x : ℝ),
    x^4 + t*x^2 + 1 = (x^2 + a*x + b) * (x^2 + c*x + d) :=
by sorry

end polynomial_factorization_l1010_101081


namespace chess_tournament_games_l1010_101048

/-- The number of games in a chess tournament where each player plays twice against every other player. -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 17 players, where each player plays twice against every other player, the total number of games played is 272. -/
theorem chess_tournament_games :
  tournament_games 17 = 272 := by
  sorry

end chess_tournament_games_l1010_101048


namespace integral_equals_minus_eight_implies_a_equals_four_l1010_101037

theorem integral_equals_minus_eight_implies_a_equals_four (a : ℝ) :
  (∫ (x : ℝ) in -a..a, (2 * x - 1)) = -8 → a = 4 := by
  sorry

end integral_equals_minus_eight_implies_a_equals_four_l1010_101037


namespace square_cutout_l1010_101010

theorem square_cutout (N M : ℕ) (h : N^2 - M^2 = 79) : M = N - 1 := by
  sorry

end square_cutout_l1010_101010


namespace quadratic_roots_condition_l1010_101060

theorem quadratic_roots_condition (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 4 * x + 2 = 0 ∧ a * y^2 - 4 * y + 2 = 0) ↔ 
  (a ≤ 2 ∧ a ≠ 0) :=
by sorry

end quadratic_roots_condition_l1010_101060


namespace estimate_fish_population_l1010_101043

/-- Estimates the number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population
  (initially_tagged : ℕ)
  (second_catch : ℕ)
  (tagged_in_second_catch : ℕ)
  (h1 : initially_tagged = 100)
  (h2 : second_catch = 300)
  (h3 : tagged_in_second_catch = 15) :
  (initially_tagged * second_catch) / tagged_in_second_catch = 2000 := by
  sorry

#check estimate_fish_population

end estimate_fish_population_l1010_101043


namespace max_value_of_a_l1010_101092

theorem max_value_of_a : 
  (∃ a : ℝ, ∀ x : ℝ, x < a → x^2 - 2*x - 3 > 0) ∧ 
  (∀ a : ℝ, ∃ x : ℝ, x^2 - 2*x - 3 > 0 ∧ x ≥ a) →
  (∀ b : ℝ, (∀ x : ℝ, x < b → x^2 - 2*x - 3 > 0) → b ≤ -1) ∧
  (∀ x : ℝ, x < -1 → x^2 - 2*x - 3 > 0) :=
by sorry

end max_value_of_a_l1010_101092


namespace log_equation_proof_l1010_101091

theorem log_equation_proof (y : ℝ) (m : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → (Real.log 125 / Real.log 2 = m * y) → m = 9 := by
  sorry

end log_equation_proof_l1010_101091


namespace student_age_l1010_101046

theorem student_age (student_age man_age : ℕ) : 
  man_age = student_age + 26 →
  man_age + 2 = 2 * (student_age + 2) →
  student_age = 24 := by
sorry

end student_age_l1010_101046


namespace incorrect_inequality_implication_l1010_101068

theorem incorrect_inequality_implication : ¬ (∀ a b : ℝ, a > b → a^2 > b^2) := by
  sorry

end incorrect_inequality_implication_l1010_101068


namespace base_3_to_base_9_first_digit_l1010_101004

def base_3_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

def first_digit_base_9 (n : Nat) : Nat :=
  Nat.log 9 n

theorem base_3_to_base_9_first_digit :
  let y : Nat := base_3_to_10 [1, 1, 2, 2, 1, 1]
  first_digit_base_9 y = 4 := by sorry

end base_3_to_base_9_first_digit_l1010_101004


namespace min_value_expression_equality_achievable_l1010_101075

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^3 + 6 * b^3 + 27 * c^3 + 9 / (8 * a * b * c) ≥ 18 :=
by sorry

theorem equality_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    8 * a^3 + 6 * b^3 + 27 * c^3 + 9 / (8 * a * b * c) = 18 :=
by sorry

end min_value_expression_equality_achievable_l1010_101075
