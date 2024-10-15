import Mathlib

namespace NUMINAMATH_CALUDE_place_values_in_9890_l407_40750

theorem place_values_in_9890 : 
  ∃ (thousands hundreds tens : ℕ),
    9890 = thousands * 1000 + hundreds * 100 + tens * 10 + (9890 % 10) ∧
    thousands = 9 ∧
    hundreds = 8 ∧
    tens = 9 :=
by sorry

end NUMINAMATH_CALUDE_place_values_in_9890_l407_40750


namespace NUMINAMATH_CALUDE_paint_production_max_profit_l407_40711

/-- The paint production problem -/
theorem paint_production_max_profit :
  let material_A : ℝ := 120
  let material_B : ℝ := 90
  let total_production : ℝ := 150
  let type_A_material_A : ℝ := 0.6
  let type_A_material_B : ℝ := 0.7
  let type_A_profit : ℝ := 450
  let type_B_material_A : ℝ := 0.9
  let type_B_material_B : ℝ := 0.4
  let type_B_profit : ℝ := 500
  let profit (x : ℝ) := type_A_profit * x + type_B_profit * (total_production - x)
  ∀ x : ℝ, 
    (type_A_material_A * x + type_B_material_A * (total_production - x) ≤ material_A) →
    (type_A_material_B * x + type_B_material_B * (total_production - x) ≤ material_B) →
    profit x ≤ 72500 ∧ 
    (x = 50 → profit x = 72500) :=
by sorry

end NUMINAMATH_CALUDE_paint_production_max_profit_l407_40711


namespace NUMINAMATH_CALUDE_original_integer_is_21_l407_40796

theorem original_integer_is_21 (a b c d : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : (a + b + c) / 3 + d = 29)
  (h2 : (a + b + d) / 3 + c = 23)
  (h3 : (a + c + d) / 3 + b = 21)
  (h4 : (b + c + d) / 3 + a = 17) :
  a = 21 ∨ b = 21 ∨ c = 21 ∨ d = 21 := by
  sorry

end NUMINAMATH_CALUDE_original_integer_is_21_l407_40796


namespace NUMINAMATH_CALUDE_greater_fraction_l407_40764

theorem greater_fraction (x y : ℚ) (h_sum : x + y = 5/6) (h_prod : x * y = 1/8) :
  max x y = (5 + Real.sqrt 7) / 12 := by
  sorry

end NUMINAMATH_CALUDE_greater_fraction_l407_40764


namespace NUMINAMATH_CALUDE_only_third_equation_has_nontrivial_solution_l407_40723

theorem only_third_equation_has_nontrivial_solution :
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (Real.sqrt (a^2 + b^2) = a + 2*b) ∧
  (∀ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) → Real.sqrt (a^2 + b^2) ≠ a - b) ∧
  (∀ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) → Real.sqrt (a^2 + b^2) ≠ a^2 - b^2) ∧
  (∀ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) → Real.sqrt (a^2 + b^2) ≠ a^2*b - a*b^2) :=
by sorry

end NUMINAMATH_CALUDE_only_third_equation_has_nontrivial_solution_l407_40723


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l407_40712

/-- Theorem: When the sides of a rectangle are increased by 35%, the area increases by 82.25% -/
theorem rectangle_area_increase (L W : ℝ) (L_pos : L > 0) (W_pos : W > 0) :
  let original_area := L * W
  let new_length := L * 1.35
  let new_width := W * 1.35
  let new_area := new_length * new_width
  (new_area - original_area) / original_area * 100 = 82.25 := by
  sorry

#check rectangle_area_increase

end NUMINAMATH_CALUDE_rectangle_area_increase_l407_40712


namespace NUMINAMATH_CALUDE_B_power_87_l407_40729

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, -1, 0],
    ![1,  0, 0],
    ![0,  0, 0]]

theorem B_power_87 : B ^ 87 = ![![0,  1, 0],
                                 ![-1, 0, 0],
                                 ![0,  0, 0]] := by
  sorry

end NUMINAMATH_CALUDE_B_power_87_l407_40729


namespace NUMINAMATH_CALUDE_workers_savings_l407_40781

/-- A worker's savings problem -/
theorem workers_savings (monthly_pay : ℝ) (saving_fraction : ℝ) : 
  monthly_pay > 0 →
  saving_fraction > 0 →
  saving_fraction < 1 →
  (12 * saving_fraction * monthly_pay) = (4 * (1 - saving_fraction) * monthly_pay) →
  saving_fraction = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_workers_savings_l407_40781


namespace NUMINAMATH_CALUDE_tip_calculation_l407_40765

/-- Calculates the tip amount given the meal cost, tax rate, and payment amount. -/
def calculate_tip (meal_cost : ℝ) (tax_rate : ℝ) (payment : ℝ) : ℝ :=
  payment - (meal_cost * (1 + tax_rate))

/-- Proves that given a meal cost of $15.00, a tax rate of 20%, and a payment of $20.00, the tip amount is $2.00. -/
theorem tip_calculation :
  calculate_tip 15 0.2 20 = 2 := by
  sorry

#eval calculate_tip 15 0.2 20

end NUMINAMATH_CALUDE_tip_calculation_l407_40765


namespace NUMINAMATH_CALUDE_jasons_hardcover_books_l407_40742

/-- Proves that Jason has 70 hardcover books given the problem conditions --/
theorem jasons_hardcover_books :
  let bookcase_limit : ℕ := 80
  let hardcover_weight : ℚ := 1/2
  let textbook_count : ℕ := 30
  let textbook_weight : ℕ := 2
  let knickknack_count : ℕ := 3
  let knickknack_weight : ℕ := 6
  let over_limit : ℕ := 33
  
  let total_weight : ℕ := bookcase_limit + over_limit
  let textbook_total_weight : ℕ := textbook_count * textbook_weight
  let knickknack_total_weight : ℕ := knickknack_count * knickknack_weight
  let hardcover_total_weight : ℕ := total_weight - textbook_total_weight - knickknack_total_weight
  
  (hardcover_total_weight : ℚ) / hardcover_weight = 70 := by sorry

end NUMINAMATH_CALUDE_jasons_hardcover_books_l407_40742


namespace NUMINAMATH_CALUDE_cooks_selection_theorem_l407_40714

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem cooks_selection_theorem (total_people : ℕ) (cooks_needed : ℕ) (invalid_combinations : ℕ) :
  total_people = 10 →
  cooks_needed = 3 →
  invalid_combinations = choose 8 1 →
  choose total_people cooks_needed - invalid_combinations = 112 := by
sorry

end NUMINAMATH_CALUDE_cooks_selection_theorem_l407_40714


namespace NUMINAMATH_CALUDE_smallest_positive_coterminal_angle_l407_40792

/-- 
Given an angle of -660°, prove that the smallest positive angle 
with the same terminal side is 60°.
-/
theorem smallest_positive_coterminal_angle : 
  ∃ (k : ℤ), -660 + k * 360 = 60 ∧ 
  ∀ (m : ℤ), -660 + m * 360 > 0 → -660 + m * 360 ≥ 60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_coterminal_angle_l407_40792


namespace NUMINAMATH_CALUDE_last_digit_of_seven_power_seven_power_l407_40710

theorem last_digit_of_seven_power_seven_power (n : ℕ) : 7^(7^7) ≡ 3 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_seven_power_seven_power_l407_40710


namespace NUMINAMATH_CALUDE_solution_of_equations_solution_of_inequalities_l407_40713

-- Part 1: System of Equations
def system_of_equations (x y : ℝ) : Prop :=
  2 * x - y = 3 ∧ 3 * x + 2 * y = 22

theorem solution_of_equations : 
  ∃ x y : ℝ, system_of_equations x y ∧ x = 4 ∧ y = 5 := by sorry

-- Part 2: System of Inequalities
def system_of_inequalities (x : ℝ) : Prop :=
  (x - 2) / 2 + 1 < (x + 1) / 3 ∧ 5 * x + 1 ≥ 2 * (2 + x)

theorem solution_of_inequalities : 
  ∀ x : ℝ, system_of_inequalities x ↔ 1 ≤ x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_solution_of_equations_solution_of_inequalities_l407_40713


namespace NUMINAMATH_CALUDE_max_sum_cubes_l407_40709

theorem max_sum_cubes (e f g h i : ℝ) (h1 : e^4 + f^4 + g^4 + h^4 + i^4 = 5) :
  ∃ (M : ℝ), M = 5^(3/4) ∧ e^3 + f^3 + g^3 + h^3 + i^3 ≤ M ∧
  ∃ (e' f' g' h' i' : ℝ), e'^4 + f'^4 + g'^4 + h'^4 + i'^4 = 5 ∧
                          e'^3 + f'^3 + g'^3 + h'^3 + i'^3 = M :=
by sorry

end NUMINAMATH_CALUDE_max_sum_cubes_l407_40709


namespace NUMINAMATH_CALUDE_fair_haired_employees_percentage_l407_40770

theorem fair_haired_employees_percentage 
  (total_employees : ℕ) 
  (women_fair_hair_percentage : ℚ) 
  (fair_haired_women_percentage : ℚ) 
  (h1 : women_fair_hair_percentage = 10 / 100) 
  (h2 : fair_haired_women_percentage = 40 / 100) :
  (women_fair_hair_percentage * total_employees) / 
  (fair_haired_women_percentage * total_employees) = 25 / 100 := by
sorry

end NUMINAMATH_CALUDE_fair_haired_employees_percentage_l407_40770


namespace NUMINAMATH_CALUDE_first_job_earnings_is_52_l407_40707

/-- Represents Mike's weekly wages --/
def TotalWages : ℝ := 160

/-- Represents the hours Mike works at his second job --/
def SecondJobHours : ℝ := 12

/-- Represents the hourly rate for Mike's second job --/
def SecondJobRate : ℝ := 9

/-- Calculates the amount Mike earns from his second job --/
def SecondJobEarnings : ℝ := SecondJobHours * SecondJobRate

/-- Represents the amount Mike earns from his first job --/
def FirstJobEarnings : ℝ := TotalWages - SecondJobEarnings

/-- Proves that Mike's earnings from his first job is $52 --/
theorem first_job_earnings_is_52 : FirstJobEarnings = 52 := by
  sorry

end NUMINAMATH_CALUDE_first_job_earnings_is_52_l407_40707


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l407_40747

theorem circles_internally_tangent (r1 r2 : ℝ) (d : ℝ) : 
  r1 + r2 = 5 ∧ 
  r1 * r2 = 3 ∧ 
  d = 3 → 
  r1 < r2 ∧ r2 - r1 < d ∧ d < r1 + r2 := by
  sorry

#check circles_internally_tangent

end NUMINAMATH_CALUDE_circles_internally_tangent_l407_40747


namespace NUMINAMATH_CALUDE_tim_bodyguard_cost_l407_40784

/-- Calculates the total weekly cost for hiring bodyguards -/
def total_weekly_cost (num_bodyguards : ℕ) (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  num_bodyguards * hourly_rate * hours_per_day * days_per_week

/-- Proves that the total weekly cost for Tim's bodyguards is $2240 -/
theorem tim_bodyguard_cost :
  total_weekly_cost 2 20 8 7 = 2240 := by
  sorry

end NUMINAMATH_CALUDE_tim_bodyguard_cost_l407_40784


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l407_40721

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_prod : a 2 * a 4 = 1/2) :
  a 1 * a 3^2 * a 5 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l407_40721


namespace NUMINAMATH_CALUDE_custom_multiplication_prove_specific_case_l407_40741

theorem custom_multiplication (x y : ℤ) : x * y = x * y - 2 * (x + y) := by sorry

theorem prove_specific_case : 1 * (-3) = 1 := by sorry

end NUMINAMATH_CALUDE_custom_multiplication_prove_specific_case_l407_40741


namespace NUMINAMATH_CALUDE_negative_exponent_division_l407_40782

theorem negative_exponent_division (a : ℝ) : -a^6 / a^3 = -a^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_exponent_division_l407_40782


namespace NUMINAMATH_CALUDE_blue_contour_area_relation_l407_40746

/-- Represents the area of a blue contour on a sphere. -/
def blueContourArea (sphereRadius : ℝ) (contourArea : ℝ) : Prop :=
  contourArea ≥ 0 ∧ contourArea ≤ 4 * Real.pi * sphereRadius^2

/-- Theorem stating the relationship between blue contour areas on two concentric spheres. -/
theorem blue_contour_area_relation
  (r₁ : ℝ) (r₂ : ℝ) (a₁ : ℝ) (a₂ : ℝ)
  (h_r₁ : r₁ = 4)
  (h_r₂ : r₂ = 6)
  (h_a₁ : a₁ = 27)
  (h_positive : r₁ > 0 ∧ r₂ > 0)
  (h_contour₁ : blueContourArea r₁ a₁)
  (h_contour₂ : blueContourArea r₂ a₂)
  (h_proportion : a₁ / a₂ = (r₁ / r₂)^2) :
  a₂ = 60.75 :=
sorry

end NUMINAMATH_CALUDE_blue_contour_area_relation_l407_40746


namespace NUMINAMATH_CALUDE_solve_equation_l407_40787

theorem solve_equation (x : ℝ) (h : Real.sqrt (3 / x + 5) = 2) : x = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l407_40787


namespace NUMINAMATH_CALUDE_geometric_sequence_quadratic_root_l407_40748

theorem geometric_sequence_quadratic_root
  (a b c : ℝ)
  (h_geom : ∃ r : ℝ, b = a * r ∧ c = a * r^2)
  (h_order : a ≤ b ∧ b ≤ c)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_one_root : ∃! x : ℝ, a * x^2 + b * x + c = 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = -1/8 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_quadratic_root_l407_40748


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l407_40761

theorem algebraic_expression_value (m n : ℝ) (h : m ≠ n) 
  (h_equal : m^2 - 2*m + 3 = n^2 - 2*n + 3) : 
  let x := m + n
  (x^2 - 2*x + 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l407_40761


namespace NUMINAMATH_CALUDE_solve_prize_problem_l407_40728

def prize_problem (x y m n w : ℝ) : Prop :=
  x + 2*y = 40 ∧
  2*x + 3*y = 70 ∧
  m + n = 60 ∧
  m ≥ n/2 ∧
  w = m*x + n*y

theorem solve_prize_problem :
  ∀ x y m n w,
  prize_problem x y m n w →
  (x = 20 ∧ y = 10) ∧
  (∀ m' n' w',
    prize_problem x y m' n' w' →
    w ≤ w') ∧
  (m = 20 ∧ n = 40 ∧ w = 800) :=
by sorry

end NUMINAMATH_CALUDE_solve_prize_problem_l407_40728


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l407_40702

/-- Given a man's speed with the current and the speed of the current, 
    calculates the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that given the specific speeds in the problem, 
    the man's speed against the current is 12 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 22 5 = 12 := by
  sorry

#eval speed_against_current 22 5

end NUMINAMATH_CALUDE_mans_speed_against_current_l407_40702


namespace NUMINAMATH_CALUDE_square_area_proof_l407_40720

theorem square_area_proof (x : ℝ) 
  (h1 : 4 * x - 15 = 20 - 3 * x) : 
  (4 * x - 15) ^ 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l407_40720


namespace NUMINAMATH_CALUDE_triangle_type_l407_40795

theorem triangle_type (A B C : ℝ) (BC AC : ℝ) (h : BC * Real.cos A = AC * Real.cos B) :
  A = B ∨ A + B = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_type_l407_40795


namespace NUMINAMATH_CALUDE_janes_weekly_reading_l407_40759

/-- Represents the number of pages Jane reads on a given day -/
structure DailyReading where
  morning : ℕ
  lunch : ℕ
  evening : ℕ
  extra : ℕ

/-- Calculates the total pages read in a day -/
def totalPagesPerDay (d : DailyReading) : ℕ :=
  d.morning + d.lunch + d.evening + d.extra

/-- Represents Jane's weekly reading schedule -/
def weeklySchedule : List DailyReading :=
  [
    { morning := 5,  lunch := 0, evening := 10, extra := 0  }, -- Monday
    { morning := 7,  lunch := 0, evening := 8,  extra := 0  }, -- Tuesday
    { morning := 5,  lunch := 0, evening := 5,  extra := 0  }, -- Wednesday
    { morning := 7,  lunch := 0, evening := 8,  extra := 15 }, -- Thursday
    { morning := 10, lunch := 5, evening := 0,  extra := 0  }, -- Friday
    { morning := 12, lunch := 0, evening := 20, extra := 0  }, -- Saturday
    { morning := 12, lunch := 0, evening := 0,  extra := 0  }  -- Sunday
  ]

/-- Theorem: Jane reads 129 pages in total over one week -/
theorem janes_weekly_reading : 
  (weeklySchedule.map totalPagesPerDay).sum = 129 := by
  sorry

end NUMINAMATH_CALUDE_janes_weekly_reading_l407_40759


namespace NUMINAMATH_CALUDE_value_of_expression_l407_40788

theorem value_of_expression (x y z : ℝ) 
  (eq1 : 3 * x - 4 * y - z = 0)
  (eq2 : x + 4 * y - 15 * z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 + 3*x*y - y*z) / (y^2 + z^2) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l407_40788


namespace NUMINAMATH_CALUDE_length_MN_l407_40727

-- Define the points on the line
variable (A B C D M N : ℝ)

-- Define the conditions
axiom order : A < B ∧ B < C ∧ C < D
axiom midpoint_M : M = (A + C) / 2
axiom midpoint_N : N = (B + D) / 2
axiom length_AD : D - A = 68
axiom length_BC : C - B = 20

-- Theorem to prove
theorem length_MN : N - M = 24 := by sorry

end NUMINAMATH_CALUDE_length_MN_l407_40727


namespace NUMINAMATH_CALUDE_cyclic_triples_count_l407_40736

/-- Represents a round-robin tournament. -/
structure Tournament where
  n : ℕ  -- number of teams
  wins : ℕ  -- number of wins per team
  losses : ℕ  -- number of losses per team

/-- Calculates the number of cyclic triples in a tournament. -/
def cyclic_triples (t : Tournament) : ℕ :=
  if t.n * (t.n - 1) = 2 * (t.wins + t.losses) ∧ t.wins = 12 ∧ t.losses = 8
  then 665
  else 0

theorem cyclic_triples_count (t : Tournament) :
  t.n * (t.n - 1) = 2 * (t.wins + t.losses) →
  t.wins = 12 →
  t.losses = 8 →
  cyclic_triples t = 665 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_triples_count_l407_40736


namespace NUMINAMATH_CALUDE_negative_slope_implies_negative_correlation_l407_40703

/-- Represents a linear regression equation -/
structure LinearRegression where
  a : ℝ
  b : ℝ

/-- The correlation coefficient between two variables -/
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

/-- Theorem: Given a linear regression with negative slope, 
    the correlation coefficient is between -1 and 0 -/
theorem negative_slope_implies_negative_correlation 
  (reg : LinearRegression) 
  (x y : ℝ → ℝ) 
  (h_reg : ∀ t, y t = reg.a + reg.b * x t) 
  (h_neg : reg.b < 0) : 
  -1 < correlation_coefficient x y ∧ correlation_coefficient x y < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_slope_implies_negative_correlation_l407_40703


namespace NUMINAMATH_CALUDE_prime_pairs_theorem_l407_40733

def is_valid_pair (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ (7 * p + 1) % q = 0 ∧ (7 * q + 1) % p = 0

theorem prime_pairs_theorem : 
  ∀ p q : ℕ, is_valid_pair p q ↔ (p = 2 ∧ q = 3) ∨ (p = 2 ∧ q = 5) ∨ (p = 3 ∧ q = 11) :=
sorry

end NUMINAMATH_CALUDE_prime_pairs_theorem_l407_40733


namespace NUMINAMATH_CALUDE_arithmetic_equalities_l407_40768

theorem arithmetic_equalities : 
  (10 - 20 - (-7) + |(-2)|) = -1 ∧ 
  (48 * (-1/4) - (-36) / 4) = -3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_equalities_l407_40768


namespace NUMINAMATH_CALUDE_min_value_zero_iff_k_eq_one_l407_40763

/-- The quadratic expression in x and y with parameter k -/
def f (k x y : ℝ) : ℝ := 3*x^2 - 4*k*x*y + (2*k^2 + 1)*y^2 - 6*x - 2*y + 4

/-- The theorem stating that the minimum value of f is 0 iff k = 1 -/
theorem min_value_zero_iff_k_eq_one :
  (∃ (m : ℝ), m = 0 ∧ ∀ x y : ℝ, f 1 x y ≥ m) ∧
  (∀ k : ℝ, k ≠ 1 → ¬∃ (m : ℝ), m = 0 ∧ ∀ x y : ℝ, f k x y ≥ m) :=
sorry

end NUMINAMATH_CALUDE_min_value_zero_iff_k_eq_one_l407_40763


namespace NUMINAMATH_CALUDE_no_genetic_recombination_in_dna_replication_l407_40791

-- Define the basic types
def Cell : Type := String
def Process : Type := String

-- Define the specific cell and processes
def spermatogonialCell : Cell := "spermatogonial cell"
def geneticRecombination : Process := "genetic recombination"
def dnaUnwinding : Process := "DNA unwinding"
def geneMutation : Process := "gene mutation"
def proteinSynthesis : Process := "protein synthesis"

-- Define a function to represent whether a process occurs during DNA replication
def occursInDnaReplication (c : Cell) (p : Process) : Prop := sorry

-- State the theorem
theorem no_genetic_recombination_in_dna_replication :
  occursInDnaReplication spermatogonialCell dnaUnwinding ∧
  occursInDnaReplication spermatogonialCell geneMutation ∧
  occursInDnaReplication spermatogonialCell proteinSynthesis →
  ¬ occursInDnaReplication spermatogonialCell geneticRecombination :=
by sorry

end NUMINAMATH_CALUDE_no_genetic_recombination_in_dna_replication_l407_40791


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l407_40716

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  second_quadrant (-2 : ℝ) (3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l407_40716


namespace NUMINAMATH_CALUDE_sandy_savings_l407_40793

theorem sandy_savings (last_year_salary : ℝ) (last_year_savings_rate : ℝ)
  (salary_increase_rate : ℝ) (savings_increase_rate : ℝ) :
  last_year_savings_rate = 0.06 →
  salary_increase_rate = 0.10 →
  savings_increase_rate = 1.65 →
  (savings_increase_rate * last_year_savings_rate * last_year_salary) /
  (last_year_salary * (1 + salary_increase_rate)) = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_sandy_savings_l407_40793


namespace NUMINAMATH_CALUDE_farmer_shipped_six_boxes_last_week_l407_40797

/-- Represents the number of pomelos in a dozen -/
def dozen : ℕ := 12

/-- Represents the number of pomelos shipped last week -/
def pomelos_last_week : ℕ := 240

/-- Represents the number of boxes shipped this week -/
def boxes_this_week : ℕ := 20

/-- Represents the total number of dozens of pomelos shipped -/
def total_dozens : ℕ := 60

/-- Represents the number of boxes shipped last week -/
def boxes_last_week : ℕ := 6

/-- Proves that the farmer shipped 6 boxes last week given the conditions -/
theorem farmer_shipped_six_boxes_last_week :
  let total_pomelos := total_dozens * dozen
  let pomelos_this_week := total_pomelos - pomelos_last_week
  let pomelos_per_box := pomelos_this_week / boxes_this_week
  pomelos_last_week / pomelos_per_box = boxes_last_week :=
by sorry

end NUMINAMATH_CALUDE_farmer_shipped_six_boxes_last_week_l407_40797


namespace NUMINAMATH_CALUDE_zero_only_universal_prime_multiple_l407_40739

theorem zero_only_universal_prime_multiple : ∃! n : ℤ, ∀ p : ℕ, Prime p → ∃ k : ℤ, n * p = k * p :=
sorry

end NUMINAMATH_CALUDE_zero_only_universal_prime_multiple_l407_40739


namespace NUMINAMATH_CALUDE_lillian_initial_candies_l407_40752

-- Define the variables
def initial_candies : ℕ := sorry
def father_gave : ℕ := 5
def total_candies : ℕ := 93

-- State the theorem
theorem lillian_initial_candies : 
  initial_candies + father_gave = total_candies → initial_candies = 88 :=
by
  sorry

end NUMINAMATH_CALUDE_lillian_initial_candies_l407_40752


namespace NUMINAMATH_CALUDE_triangle_area_l407_40700

theorem triangle_area (a b c : ℝ) (h1 : a = 13) (h2 : b = 84) (h3 : c = 85) :
  (1/2) * a * b = 546 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l407_40700


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l407_40766

theorem rectangle_area (square_area : ℝ) (rectangle_length : ℝ) : ℝ :=
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_breath := (3 / 5) * circle_radius
  rectangle_length * rectangle_breath

theorem rectangle_area_proof :
  rectangle_area 2025 10 = 270 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l407_40766


namespace NUMINAMATH_CALUDE_dmv_waiting_time_l407_40735

/-- Calculates the additional waiting time at the DMV -/
theorem dmv_waiting_time (initial_wait : ℕ) (total_wait : ℕ) : 
  initial_wait = 20 →
  total_wait = 114 →
  total_wait = initial_wait + 4 * initial_wait + (total_wait - (initial_wait + 4 * initial_wait)) →
  total_wait - (initial_wait + 4 * initial_wait) = 34 :=
by sorry

end NUMINAMATH_CALUDE_dmv_waiting_time_l407_40735


namespace NUMINAMATH_CALUDE_max_y_value_l407_40757

theorem max_y_value (x y : ℤ) (h : x * y + 6 * x + 5 * y = -6) : 
  ∃ (max_y : ℤ), y ≤ max_y ∧ max_y = 18 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l407_40757


namespace NUMINAMATH_CALUDE_cycle_selling_price_l407_40744

/-- Given a cycle with a cost price of 1400 and sold at a loss of 25%, 
    prove that the selling price is 1050. -/
theorem cycle_selling_price 
  (cost_price : ℝ) 
  (loss_percentage : ℝ) 
  (h1 : cost_price = 1400) 
  (h2 : loss_percentage = 25) : 
  cost_price * (1 - loss_percentage / 100) = 1050 := by
sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l407_40744


namespace NUMINAMATH_CALUDE_field_trip_cost_l407_40775

/-- Calculate the total cost of renting buses and paying tolls for a field trip -/
theorem field_trip_cost (total_people : ℕ) (seats_per_bus : ℕ) 
  (rental_cost_per_bus : ℕ) (toll_per_bus : ℕ) : 
  total_people = 260 → 
  seats_per_bus = 41 → 
  rental_cost_per_bus = 300000 → 
  toll_per_bus = 7500 → 
  (((total_people + seats_per_bus - 1) / seats_per_bus) * 
   (rental_cost_per_bus + toll_per_bus)) = 2152500 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_cost_l407_40775


namespace NUMINAMATH_CALUDE_bugs_eat_flowers_l407_40740

/-- The number of flowers eaten by a group of bugs -/
def flowers_eaten (num_bugs : ℕ) (flowers_per_bug : ℕ) : ℕ :=
  num_bugs * flowers_per_bug

/-- Theorem: Given 3 bugs, each eating 2 flowers, the total number of flowers eaten is 6 -/
theorem bugs_eat_flowers :
  flowers_eaten 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bugs_eat_flowers_l407_40740


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l407_40789

theorem smallest_number_with_given_remainders : ∃! n : ℕ, 
  n > 0 ∧ 
  n % 3 = 2 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 3 ∧
  ∀ m : ℕ, m > 0 ∧ m % 3 = 2 ∧ m % 5 = 2 ∧ m % 7 = 3 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l407_40789


namespace NUMINAMATH_CALUDE_arithmetic_computation_l407_40799

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l407_40799


namespace NUMINAMATH_CALUDE_optimal_plan_is_correct_l407_40790

/-- Represents the number of cars a worker can install per month -/
structure WorkerProductivity where
  skilled : ℕ
  new : ℕ

/-- Represents the monthly salary of workers -/
structure WorkerSalary where
  skilled : ℕ
  new : ℕ

/-- Represents a recruitment plan -/
structure RecruitmentPlan where
  skilled : ℕ
  new : ℕ

def optimal_plan (prod : WorkerProductivity) (salary : WorkerSalary) : RecruitmentPlan :=
  sorry

theorem optimal_plan_is_correct (prod : WorkerProductivity) (salary : WorkerSalary) :
  let plan := optimal_plan prod salary
  prod.skilled * plan.skilled + prod.new * plan.new = 20 ∧
  ∀ other : RecruitmentPlan,
    prod.skilled * other.skilled + prod.new * other.new = 20 →
    salary.skilled * plan.skilled + salary.new * plan.new ≤
    salary.skilled * other.skilled + salary.new * other.new :=
by
  sorry

#check @optimal_plan_is_correct

end NUMINAMATH_CALUDE_optimal_plan_is_correct_l407_40790


namespace NUMINAMATH_CALUDE_simplify_expression_l407_40773

theorem simplify_expression (x : ℝ) : 
  (3 * x + 6 - 5 * x) / 3 = -2/3 * x + 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l407_40773


namespace NUMINAMATH_CALUDE_range_of_a_l407_40771

theorem range_of_a (a : ℝ) (ha : a ≠ 0) : 
  let A := {x : ℝ | x^2 - x - 6 < 0}
  let B := {x : ℝ | x^2 + 2*x - 8 ≥ 0}
  let C := {x : ℝ | x^2 - 4*a*x + 3*a^2 < 0}
  C ⊆ (A ∩ (Set.univ \ B)) →
  (0 < a ∧ a ≤ 2/3) ∨ (-2/3 ≤ a ∧ a < 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l407_40771


namespace NUMINAMATH_CALUDE_penelope_candy_count_l407_40786

/-- Given a ratio of M&M candies to Starbursts candies and a number of M&M candies,
    calculate the number of Starbursts candies. -/
def calculate_starbursts (mm_ratio : ℕ) (starbursts_ratio : ℕ) (mm_count : ℕ) : ℕ :=
  (mm_count / mm_ratio) * starbursts_ratio

/-- Theorem stating that given 5 M&M candies for every 3 Starbursts candies,
    and 25 M&M candies, there are 15 Starbursts candies. -/
theorem penelope_candy_count :
  calculate_starbursts 5 3 25 = 15 := by
  sorry

end NUMINAMATH_CALUDE_penelope_candy_count_l407_40786


namespace NUMINAMATH_CALUDE_band_percentage_of_ticket_price_l407_40731

/-- Proves that the band receives 70% of the ticket price, given the concert conditions -/
theorem band_percentage_of_ticket_price : 
  ∀ (attendance : ℕ) (ticket_price : ℕ) (band_members : ℕ) (member_earnings : ℕ),
    attendance = 500 →
    ticket_price = 30 →
    band_members = 4 →
    member_earnings = 2625 →
    (band_members * member_earnings : ℚ) / (attendance * ticket_price) = 70 / 100 := by
  sorry

end NUMINAMATH_CALUDE_band_percentage_of_ticket_price_l407_40731


namespace NUMINAMATH_CALUDE_p_and_q_true_l407_40755

theorem p_and_q_true (h : ¬(¬(p ∧ q))) : p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_true_l407_40755


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_B_P_l407_40706

-- Define the sets A, B, and P
def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def P : Set ℝ := {x | x ≤ 0 ∨ x ≥ 5/2}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

-- Theorem for (ᶜB) ∪ P
theorem union_complement_B_P : (Bᶜ : Set ℝ) ∪ P = {x : ℝ | x ≤ 0 ∨ x ≥ 5/2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_B_P_l407_40706


namespace NUMINAMATH_CALUDE_third_circle_radius_l407_40722

/-- Given three circles with the specified properties, the radius of the third circle is 15/14 -/
theorem third_circle_radius (r1 r2 r3 : ℝ) : 
  r1 = 2 →  -- Circle 1 has radius 2
  r2 = 3 →  -- Circle 2 has radius 3
  (∃ d : ℝ, d = r1 + r2) →  -- Circle 1 and Circle 2 are externally tangent
  (∃ k1 k2 k3 : ℝ, k1 = 1/r1 ∧ k2 = 1/r2 ∧ k3 = 1/r3 ∧ 
    k1 + k2 + k3 + 2 * Real.sqrt (k1 * k2 + k2 * k3 + k3 * k1) = 0) →  -- Descartes' theorem for externally tangent circles
  r3 = 15/14 := by
sorry

end NUMINAMATH_CALUDE_third_circle_radius_l407_40722


namespace NUMINAMATH_CALUDE_square_difference_theorem_l407_40779

theorem square_difference_theorem (x : ℝ) (h : (x + 2) * (x - 2) = 1221) : 
  x^2 = 1225 ∧ (x + 1) * (x - 1) = 1224 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l407_40779


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_13_and_3_l407_40734

theorem three_digit_divisible_by_13_and_3 : 
  (Finset.filter (fun n => n % 13 = 0 ∧ n % 3 = 0) (Finset.range 900)).card = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_13_and_3_l407_40734


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l407_40754

/-- The complex number z = sin 2 + i cos 2 is located in the fourth quadrant of the complex plane -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := Complex.mk (Real.sin 2) (Real.cos 2)
  z.re > 0 ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l407_40754


namespace NUMINAMATH_CALUDE_regular_dodecahedron_edges_l407_40760

/-- A regular dodecahedron is a polyhedron with 12 regular pentagonal faces -/
structure RegularDodecahedron where
  faces : Nat
  edges_per_face : Nat
  shared_edges : Nat

/-- Calculate the number of edges in a regular dodecahedron -/
def count_edges (d : RegularDodecahedron) : Nat :=
  (d.faces * d.edges_per_face) / d.shared_edges

/-- Theorem: A regular dodecahedron has 30 edges -/
theorem regular_dodecahedron_edges :
  ∀ d : RegularDodecahedron,
    d.faces = 12 →
    d.edges_per_face = 5 →
    d.shared_edges = 2 →
    count_edges d = 30 := by
  sorry

#check regular_dodecahedron_edges

end NUMINAMATH_CALUDE_regular_dodecahedron_edges_l407_40760


namespace NUMINAMATH_CALUDE_sum_of_terms_3_to_6_l407_40743

/-- Given a sequence {aₙ} where the sum of the first n terms is Sₙ = n² + 2n + 5,
    prove that a₃ + a₄ + a₅ + a₆ = 40 -/
theorem sum_of_terms_3_to_6 (a : ℕ → ℤ) (S : ℕ → ℤ) 
    (h : ∀ n : ℕ, S n = n^2 + 2*n + 5) : 
    a 3 + a 4 + a 5 + a 6 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_terms_3_to_6_l407_40743


namespace NUMINAMATH_CALUDE_power_of_128_l407_40780

theorem power_of_128 : (128 : ℝ) ^ (4/7 : ℝ) = 16 := by
  have h1 : (128 : ℝ) = 2^7 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_128_l407_40780


namespace NUMINAMATH_CALUDE_range_of_a_l407_40774

-- Define the set A
def A (a : ℝ) : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (a * x^2 + 2*(a-1)*x - 4)}

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, A a = Set.Ici 0) ↔ Set.Ici 0 = {a : ℝ | 0 ≤ a} := by sorry

end NUMINAMATH_CALUDE_range_of_a_l407_40774


namespace NUMINAMATH_CALUDE_chocolate_boxes_total_l407_40751

/-- The total number of chocolate pieces in multiple boxes -/
def total_pieces (boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  boxes * pieces_per_box

/-- Theorem: The total number of chocolate pieces in 6 boxes with 500 pieces each is 3000 -/
theorem chocolate_boxes_total :
  total_pieces 6 500 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_boxes_total_l407_40751


namespace NUMINAMATH_CALUDE_ticket_sales_proof_ticket_sales_result_l407_40745

theorem ticket_sales_proof (reduced_first_week : ℕ) (total_tickets : ℕ) : ℕ :=
  let reduced_price_tickets := reduced_first_week
  let full_price_tickets := 5 * reduced_price_tickets
  let total := reduced_price_tickets + full_price_tickets
  
  have h1 : reduced_first_week = 5400 := by sorry
  have h2 : total_tickets = 25200 := by sorry
  have h3 : total = total_tickets := by sorry
  
  full_price_tickets

theorem ticket_sales_result : ticket_sales_proof 5400 25200 = 21000 := by sorry

end NUMINAMATH_CALUDE_ticket_sales_proof_ticket_sales_result_l407_40745


namespace NUMINAMATH_CALUDE_art_arrangement_probability_l407_40701

/-- The total number of art pieces --/
def total_pieces : ℕ := 12

/-- The number of Escher prints --/
def escher_prints : ℕ := 4

/-- The number of Picasso prints --/
def picasso_prints : ℕ := 2

/-- The probability of the desired arrangement --/
def arrangement_probability : ℚ := 912 / 479001600

theorem art_arrangement_probability :
  let remaining_pieces := total_pieces - escher_prints
  let escher_block_positions := remaining_pieces + 1
  let escher_internal_arrangements := Nat.factorial escher_prints
  let picasso_positions := total_pieces - escher_prints + 1
  let valid_picasso_arrangements := 38
  (escher_block_positions * escher_internal_arrangements * valid_picasso_arrangements : ℚ) /
    Nat.factorial total_pieces = arrangement_probability := by
  sorry

end NUMINAMATH_CALUDE_art_arrangement_probability_l407_40701


namespace NUMINAMATH_CALUDE_total_prime_factors_l407_40777

def expression (a b c : ℕ) := (4^a) * (7^b) * (11^c)

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem total_prime_factors (a b c : ℕ) :
  a = 11 → b = 7 → c = 2 → is_prime 7 → is_prime 11 →
  (∃ n : ℕ, expression a b c = 2^(2*a) * 7^b * 11^c ∧ 
   n = (2*a) + b + c ∧ n = 31) :=
sorry

end NUMINAMATH_CALUDE_total_prime_factors_l407_40777


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l407_40776

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 1) + 1 / (b + 1) = 1) : 
  a + 2 * b ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l407_40776


namespace NUMINAMATH_CALUDE_friend_total_time_l407_40767

def my_reading_time : ℝ := 3 * 60 -- 3 hours in minutes
def my_writing_time : ℝ := 60 -- 1 hour in minutes
def friend_reading_speed_ratio : ℝ := 4 -- friend reads 4 times as fast

theorem friend_total_time (friend_reading_time friend_writing_time : ℝ) :
  friend_reading_time = my_reading_time / friend_reading_speed_ratio →
  friend_writing_time = my_writing_time →
  friend_reading_time + friend_writing_time = 105 := by
sorry

end NUMINAMATH_CALUDE_friend_total_time_l407_40767


namespace NUMINAMATH_CALUDE_max_value_of_expression_l407_40785

theorem max_value_of_expression (x : ℝ) :
  ∃ (max_x : ℝ), ∀ y, 1 - (y + 5)^2 ≤ 1 - (max_x + 5)^2 ∧ max_x = -5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l407_40785


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l407_40717

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and lines
variable (perp : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_transitivity 
  (α β : Plane) (m n : Line) 
  (h1 : perp_line_plane n α) 
  (h2 : perp_line_plane n β) 
  (h3 : perp_line_plane m α) : 
  perp_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l407_40717


namespace NUMINAMATH_CALUDE_yanna_kept_apples_l407_40778

def apples_kept (total bought : ℕ) (given_to_zenny given_to_andrea : ℕ) : ℕ :=
  bought - given_to_zenny - given_to_andrea

theorem yanna_kept_apples :
  apples_kept 60 18 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_yanna_kept_apples_l407_40778


namespace NUMINAMATH_CALUDE_correct_system_of_equations_l407_40704

/-- Represents the price of a basketball in yuan -/
def basketball_price : ℝ := sorry

/-- Represents the price of a soccer ball in yuan -/
def soccer_ball_price : ℝ := sorry

/-- The total cost of the purchase in yuan -/
def total_cost : ℝ := 445

/-- The number of basketballs purchased -/
def num_basketballs : ℕ := 3

/-- The number of soccer balls purchased -/
def num_soccer_balls : ℕ := 7

/-- The price difference between a basketball and a soccer ball in yuan -/
def price_difference : ℝ := 5

/-- Theorem stating that the system of equations correctly represents the given conditions -/
theorem correct_system_of_equations : 
  (num_basketballs * basketball_price + num_soccer_balls * soccer_ball_price = total_cost) ∧ 
  (basketball_price = soccer_ball_price + price_difference) := by
  sorry

end NUMINAMATH_CALUDE_correct_system_of_equations_l407_40704


namespace NUMINAMATH_CALUDE_min_value_x_plus_inverse_y_l407_40718

theorem min_value_x_plus_inverse_y (x y : ℝ) (h1 : x ≥ 3) (h2 : x - y = 1) :
  ∃ m : ℝ, m = 7/2 ∧ ∀ z : ℝ, z ≥ 3 → ∀ w : ℝ, z - w = 1 → z + 1/w ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_inverse_y_l407_40718


namespace NUMINAMATH_CALUDE_expression_value_l407_40726

theorem expression_value (x y : ℝ) 
  (eq1 : 3 * x + y = 7) 
  (eq2 : x + 3 * y = 8) : 
  10 * x^2 + 13 * x * y + 10 * y^2 = 113 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l407_40726


namespace NUMINAMATH_CALUDE_toothpicks_150th_stage_l407_40749

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ :=
  6 + (n - 1) * 4

/-- Theorem: The number of toothpicks in the 150th stage is 602 -/
theorem toothpicks_150th_stage : toothpicks 150 = 602 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_150th_stage_l407_40749


namespace NUMINAMATH_CALUDE_playground_children_l407_40753

/-- The number of children on a playground given the number of boys and girls -/
theorem playground_children (boys girls : ℕ) (h1 : boys = 40) (h2 : girls = 77) :
  boys + girls = 117 := by
  sorry

end NUMINAMATH_CALUDE_playground_children_l407_40753


namespace NUMINAMATH_CALUDE_rooster_count_l407_40769

/-- Given a chicken farm with roosters and hens, proves the number of roosters -/
theorem rooster_count (total : ℕ) (ratio : ℚ) (rooster_count : ℕ) : 
  total = 9000 →
  ratio = 2 / 1 →
  rooster_count = total * (ratio / (1 + ratio)) →
  rooster_count = 6000 := by
  sorry


end NUMINAMATH_CALUDE_rooster_count_l407_40769


namespace NUMINAMATH_CALUDE_tenths_of_2019_l407_40758

theorem tenths_of_2019 : (2019 : ℚ) / 10 = 201.9 := by
  sorry

end NUMINAMATH_CALUDE_tenths_of_2019_l407_40758


namespace NUMINAMATH_CALUDE_complex_fourth_power_l407_40772

theorem complex_fourth_power (i : ℂ) : i^2 = -1 → (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_l407_40772


namespace NUMINAMATH_CALUDE_two_digit_number_representation_l407_40756

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (num : TwoDigitNumber) : ℕ :=
  10 * num.tens + num.units

theorem two_digit_number_representation (n m : ℕ) (h : n ≥ 1 ∧ n ≤ 9 ∧ m ≤ 9) :
  let num : TwoDigitNumber := ⟨n, m, h⟩
  num.value = 10 * n + m := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_representation_l407_40756


namespace NUMINAMATH_CALUDE_triangle_side_roots_l407_40794

theorem triangle_side_roots (m : ℝ) : 
  (∃ a b c : ℝ, 
    (a - 1) * (a^2 - 2*a + m) = 0 ∧
    (b - 1) * (b^2 - 2*b + m) = 0 ∧
    (c - 1) * (c^2 - 2*c + m) = 0 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ a + c > b) →
  3/4 < m ∧ m ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_roots_l407_40794


namespace NUMINAMATH_CALUDE_apps_added_l407_40783

theorem apps_added (initial_apps : ℕ) (deleted_apps : ℕ) (final_apps : ℕ) :
  initial_apps = 10 →
  deleted_apps = 17 →
  final_apps = 4 →
  ∃ (added_apps : ℕ), (initial_apps + added_apps - deleted_apps = final_apps) ∧ (added_apps = 11) :=
by sorry

end NUMINAMATH_CALUDE_apps_added_l407_40783


namespace NUMINAMATH_CALUDE_square_function_difference_l407_40719

/-- For f(x) = x^2, prove that f(x) - f(x-1) = 2x - 1 for all real x -/
theorem square_function_difference (x : ℝ) : x^2 - (x-1)^2 = 2*x - 1 := by
  sorry

end NUMINAMATH_CALUDE_square_function_difference_l407_40719


namespace NUMINAMATH_CALUDE_market_equilibrium_and_max_revenue_l407_40762

-- Define the demand function
def demand_function (P : ℝ) : ℝ := 688 - 4 * P

-- Define the supply function (to be proven)
def supply_function (P : ℝ) : ℝ := 6 * P - 312

-- Define the tax revenue function
def tax_revenue (t : ℝ) (Q : ℝ) : ℝ := t * Q

-- Theorem statement
theorem market_equilibrium_and_max_revenue :
  -- Conditions
  let change_ratio : ℝ := 1.5
  let production_tax : ℝ := 90
  let producer_price : ℝ := 64

  -- Prove that the supply function is correct
  ∀ P, supply_function P = 6 * P - 312 ∧
  
  -- Prove that the maximum tax revenue is 8640
  ∃ t_optimal, 
    let Q_optimal := demand_function (producer_price + t_optimal)
    tax_revenue t_optimal Q_optimal = 8640 ∧
    ∀ t, tax_revenue t (demand_function (producer_price + t)) ≤ 8640 :=
by sorry

end NUMINAMATH_CALUDE_market_equilibrium_and_max_revenue_l407_40762


namespace NUMINAMATH_CALUDE_game_points_total_l407_40724

theorem game_points_total (eric_points mark_points samanta_points : ℕ) : 
  eric_points = 6 →
  mark_points = eric_points + eric_points / 2 →
  samanta_points = mark_points + 8 →
  eric_points + mark_points + samanta_points = 32 := by
sorry

end NUMINAMATH_CALUDE_game_points_total_l407_40724


namespace NUMINAMATH_CALUDE_shelf_adjustment_theorem_l407_40732

/-- The number of items on the shelf -/
def total_items : ℕ := 12

/-- The initial number of items on the upper layer -/
def initial_upper : ℕ := 4

/-- The initial number of items on the lower layer -/
def initial_lower : ℕ := 8

/-- The number of items to be moved from lower to upper layer -/
def items_to_move : ℕ := 2

/-- The number of ways to adjust the items -/
def adjustment_ways : ℕ := Nat.choose initial_lower items_to_move

theorem shelf_adjustment_theorem : adjustment_ways = 840 := by sorry

end NUMINAMATH_CALUDE_shelf_adjustment_theorem_l407_40732


namespace NUMINAMATH_CALUDE_monotonic_decreasing_implies_order_l407_40737

theorem monotonic_decreasing_implies_order (f : ℝ → ℝ) 
  (h : ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) :
  f 3 < f 2 ∧ f 2 < f 1 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_implies_order_l407_40737


namespace NUMINAMATH_CALUDE_number_wall_solution_l407_40798

structure NumberWall :=
  (x : ℤ)
  (a b c d : ℤ)
  (e f g : ℤ)
  (h i : ℤ)
  (j : ℤ)

def NumberWall.valid (w : NumberWall) : Prop :=
  w.e = w.x + w.a ∧
  w.f = w.a + w.b ∧
  w.g = w.b + w.c ∧
  w.d = w.c + w.d ∧
  w.h = w.e + w.f ∧
  w.i = w.g + w.d ∧
  w.j = w.h + w.i ∧
  w.a = 5 ∧
  w.b = 10 ∧
  w.c = 9 ∧
  w.d = 6 ∧
  w.i = 18 ∧
  w.j = 72

theorem number_wall_solution (w : NumberWall) (h : w.valid) : w.x = -50 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_solution_l407_40798


namespace NUMINAMATH_CALUDE_real_roots_iff_m_le_25_4_m_eq_6_when_condition_satisfied_l407_40730

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop := x^2 - 5*x + m = 0

-- Define the condition for real roots
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, quadratic_equation x m

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m ∧ quadratic_equation x₂ m

-- Define the additional condition for part 2
def satisfies_root_condition (x₁ x₂ : ℝ) : Prop := 3*x₁ - 2*x₂ = 5

-- Theorem 1: Equation has real roots iff m ≤ 25/4
theorem real_roots_iff_m_le_25_4 :
  ∀ m : ℝ, has_real_roots m ↔ m ≤ 25/4 :=
sorry

-- Theorem 2: If equation has two real roots satisfying the condition, then m = 6
theorem m_eq_6_when_condition_satisfied :
  ∀ m : ℝ, has_two_distinct_real_roots m →
  (∃ x₁ x₂ : ℝ, quadratic_equation x₁ m ∧ quadratic_equation x₂ m ∧ satisfies_root_condition x₁ x₂) →
  m = 6 :=
sorry

end NUMINAMATH_CALUDE_real_roots_iff_m_le_25_4_m_eq_6_when_condition_satisfied_l407_40730


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l407_40725

-- Define the circle from part 1
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the line x + y = 1
def line1 (x y : ℝ) : Prop := x + y = 1

-- Define the line y = -2x
def line2 (x y : ℝ) : Prop := y = -2 * x

-- Define the circle from part 2
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the line from part 2
def line3 (x y : ℝ) : Prop := x - 2 * y + 2 = 0

theorem circle_and_line_properties :
  -- Part 1
  (circle1 2 (-1)) ∧ 
  (∃ (x y : ℝ), circle1 x y ∧ line1 x y) ∧
  (∃ (x y : ℝ), circle1 x y ∧ line2 x y) ∧
  -- Part 2
  (¬ circle2 2 (-2)) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle2 x₁ y₁ ∧ circle2 x₂ y₂ ∧ 
    ((x₁ - 2) * (y₁ + 2) = 4) ∧ 
    ((x₂ - 2) * (y₂ + 2) = 4) ∧
    line3 x₁ y₁ ∧ line3 x₂ y₂) := by
  sorry

#check circle_and_line_properties

end NUMINAMATH_CALUDE_circle_and_line_properties_l407_40725


namespace NUMINAMATH_CALUDE_medium_apple_cost_l407_40738

/-- Proves that the cost of a medium apple is $2 given the conditions in the problem -/
theorem medium_apple_cost (small_apple_cost big_apple_cost total_cost : ℝ)
  (small_medium_count big_count : ℕ) :
  small_apple_cost = 1.5 →
  big_apple_cost = 3 →
  small_medium_count = 6 →
  big_count = 8 →
  total_cost = 45 →
  ∃ (medium_apple_cost : ℝ),
    small_apple_cost * (small_medium_count / 2) +
    medium_apple_cost * (small_medium_count / 2) +
    big_apple_cost * big_count = total_cost ∧
    medium_apple_cost = 2 :=
by sorry

end NUMINAMATH_CALUDE_medium_apple_cost_l407_40738


namespace NUMINAMATH_CALUDE_part_a_part_b_l407_40708

-- Define a type for our set of 100 positive numbers
def PositiveSet := Fin 100 → ℝ

-- Define the property that all numbers in the set are positive
def AllPositive (s : PositiveSet) : Prop :=
  ∀ i, s i > 0

-- Define the property that the sum of any 7 numbers is less than 7
def SumOfSevenLessThanSeven (s : PositiveSet) : Prop :=
  ∀ (i₁ i₂ i₃ i₄ i₅ i₆ i₇ : Fin 100),
    i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₁ ≠ i₅ ∧ i₁ ≠ i₆ ∧ i₁ ≠ i₇ ∧
    i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₂ ≠ i₅ ∧ i₂ ≠ i₆ ∧ i₂ ≠ i₇ ∧
    i₃ ≠ i₄ ∧ i₃ ≠ i₅ ∧ i₃ ≠ i₆ ∧ i₃ ≠ i₇ ∧
    i₄ ≠ i₅ ∧ i₄ ≠ i₆ ∧ i₄ ≠ i₇ ∧
    i₅ ≠ i₆ ∧ i₅ ≠ i₇ ∧
    i₆ ≠ i₇ →
    s i₁ + s i₂ + s i₃ + s i₄ + s i₅ + s i₆ + s i₇ < 7

-- Define the property that the sum of any 10 numbers is less than 10
def SumOfTenLessThanTen (s : PositiveSet) : Prop :=
  ∀ (i₁ i₂ i₃ i₄ i₅ i₆ i₇ i₈ i₉ i₁₀ : Fin 100),
    i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₁ ≠ i₅ ∧ i₁ ≠ i₆ ∧ i₁ ≠ i₇ ∧ i₁ ≠ i₈ ∧ i₁ ≠ i₉ ∧ i₁ ≠ i₁₀ ∧
    i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₂ ≠ i₅ ∧ i₂ ≠ i₆ ∧ i₂ ≠ i₇ ∧ i₂ ≠ i₈ ∧ i₂ ≠ i₉ ∧ i₂ ≠ i₁₀ ∧
    i₃ ≠ i₄ ∧ i₃ ≠ i₅ ∧ i₃ ≠ i₆ ∧ i₃ ≠ i₇ ∧ i₃ ≠ i₈ ∧ i₃ ≠ i₉ ∧ i₃ ≠ i₁₀ ∧
    i₄ ≠ i₅ ∧ i₄ ≠ i₆ ∧ i₄ ≠ i₇ ∧ i₄ ≠ i₈ ∧ i₄ ≠ i₉ ∧ i₄ ≠ i₁₀ ∧
    i₅ ≠ i₆ ∧ i₅ ≠ i₇ ∧ i₅ ≠ i₈ ∧ i₅ ≠ i₉ ∧ i₅ ≠ i₁₀ ∧
    i₆ ≠ i₇ ∧ i₆ ≠ i₈ ∧ i₆ ≠ i₉ ∧ i₆ ≠ i₁₀ ∧
    i₇ ≠ i₈ ∧ i₇ ≠ i₉ ∧ i₇ ≠ i₁₀ ∧
    i₈ ≠ i₉ ∧ i₈ ≠ i₁₀ ∧
    i₉ ≠ i₁₀ →
    s i₁ + s i₂ + s i₃ + s i₄ + s i₅ + s i₆ + s i₇ + s i₈ + s i₉ + s i₁₀ < 10

-- Theorem for part (a)
theorem part_a (s : PositiveSet) (h₁ : AllPositive s) (h₂ : SumOfSevenLessThanSeven s) :
  SumOfTenLessThanTen s := by
  sorry

-- Theorem for part (b)
theorem part_b :
  ¬∀ (s : PositiveSet), AllPositive s → SumOfTenLessThanTen s → SumOfSevenLessThanSeven s := by
  sorry

end NUMINAMATH_CALUDE_part_a_part_b_l407_40708


namespace NUMINAMATH_CALUDE_select_two_from_four_l407_40715

theorem select_two_from_four : Nat.choose 4 2 = 6 := by sorry

end NUMINAMATH_CALUDE_select_two_from_four_l407_40715


namespace NUMINAMATH_CALUDE_inverse_N_expression_l407_40705

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, 0; 2, -4]

theorem inverse_N_expression : 
  N⁻¹ = (1 / 12 : ℚ) • N + (1 / 12 : ℚ) • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end NUMINAMATH_CALUDE_inverse_N_expression_l407_40705
