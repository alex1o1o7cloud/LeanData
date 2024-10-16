import Mathlib

namespace NUMINAMATH_CALUDE_probability_two_red_balls_l3895_389509

def total_balls : ℕ := 10
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def drawn_balls : ℕ := 5

def probability_two_red : ℚ := 10 / 21

theorem probability_two_red_balls :
  (Nat.choose red_balls 2 * Nat.choose white_balls 3) / Nat.choose total_balls drawn_balls = probability_two_red :=
sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l3895_389509


namespace NUMINAMATH_CALUDE_second_term_value_l3895_389522

/-- A geometric sequence with sum of first n terms Sn = a·3^n - 2 -/
def GeometricSequence (a : ℝ) : ℕ → ℝ := sorry

/-- Sum of first n terms of the geometric sequence -/
def Sn (a : ℝ) (n : ℕ) : ℝ := a * 3^n - 2

/-- The second term of the sequence -/
def a2 (a : ℝ) : ℝ := GeometricSequence a 2

theorem second_term_value (a : ℝ) :
  a2 a = 12 :=
sorry

end NUMINAMATH_CALUDE_second_term_value_l3895_389522


namespace NUMINAMATH_CALUDE_parabola_hyperbola_triangle_l3895_389535

/-- Theorem: Value of p for a parabola and hyperbola forming an isosceles right triangle -/
theorem parabola_hyperbola_triangle (p a b : ℝ) : 
  p > 0 → a > 0 → b > 0 →
  (∀ x y, x^2 = 2*p*y) →
  (∀ x y, x^2/a^2 - y^2/b^2 = 1) →
  (∃ x₁ y₁ x₂ y₂ x₃ y₃,
    -- Points form an isosceles right triangle
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₂ - x₃)^2 + (y₂ - y₃)^2 ∧
    (x₁ - x₂) * (x₂ - x₃) + (y₁ - y₂) * (y₂ - y₃) = 0 ∧
    -- Area of the triangle is 1
    abs ((x₁ - x₃) * (y₂ - y₃) - (x₂ - x₃) * (y₁ - y₃)) / 2 = 1 ∧
    -- Points lie on the directrix and asymptotes
    y₁ = -p/2 ∧ y₂ = -p/2 ∧
    y₁ = b/a * x₁ ∧ y₃ = -b/a * x₃) →
  p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_triangle_l3895_389535


namespace NUMINAMATH_CALUDE_proof_by_contradiction_elements_l3895_389532

/-- Elements that can be used in a proof by contradiction -/
inductive ProofByContradictionElement : Type
  | NegationOfConclusion : ProofByContradictionElement
  | KnownConditions : ProofByContradictionElement
  | AxiomsTheoremsDefinitions : ProofByContradictionElement

/-- The set of elements used in proof by contradiction -/
def ProofByContradictionSet : Set ProofByContradictionElement :=
  {ProofByContradictionElement.NegationOfConclusion,
   ProofByContradictionElement.KnownConditions,
   ProofByContradictionElement.AxiomsTheoremsDefinitions}

/-- Theorem stating that the ProofByContradictionSet contains all necessary elements -/
theorem proof_by_contradiction_elements :
  ProofByContradictionElement.NegationOfConclusion ∈ ProofByContradictionSet ∧
  ProofByContradictionElement.KnownConditions ∈ ProofByContradictionSet ∧
  ProofByContradictionElement.AxiomsTheoremsDefinitions ∈ ProofByContradictionSet :=
by sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_elements_l3895_389532


namespace NUMINAMATH_CALUDE_fraction_equality_l3895_389508

theorem fraction_equality : (1877^2 - 1862^2) / (1880^2 - 1859^2) = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3895_389508


namespace NUMINAMATH_CALUDE_min_value_of_f_l3895_389594

/-- Given positive numbers a, b, c, x, y, z satisfying the conditions,
    the function f(x, y, z) has a minimum value of 1/2 -/
theorem min_value_of_f (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (eq1 : c * y + b * z = a)
  (eq2 : a * z + c * x = b)
  (eq3 : b * x + a * y = c) :
  ∀ (x' y' z' : ℝ), 0 < x' → 0 < y' → 0 < z' →
    x'^2 / (1 + x') + y'^2 / (1 + y') + z'^2 / (1 + z') ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3895_389594


namespace NUMINAMATH_CALUDE_elevator_max_velocity_l3895_389516

/-- Represents the state of the elevator at a given time -/
structure ElevatorState where
  time : ℝ
  velocity : ℝ

/-- The elevator's motion profile -/
def elevatorMotion : ℝ → ElevatorState := sorry

/-- The acceleration period of the elevator -/
def accelerationPeriod : Set ℝ := {t | 2 ≤ t ∧ t ≤ 4}

/-- The deceleration period of the elevator -/
def decelerationPeriod : Set ℝ := {t | 22 ≤ t ∧ t ≤ 24}

/-- The constant speed period of the elevator -/
def constantSpeedPeriod : Set ℝ := {t | 4 < t ∧ t < 22}

/-- The maximum downward velocity of the elevator -/
def maxDownwardVelocity : ℝ := sorry

theorem elevator_max_velocity :
  ∀ t ∈ constantSpeedPeriod,
    (elevatorMotion t).velocity = maxDownwardVelocity ∧
    ∀ s, (elevatorMotion s).velocity ≤ maxDownwardVelocity := by
  sorry

#check elevator_max_velocity

end NUMINAMATH_CALUDE_elevator_max_velocity_l3895_389516


namespace NUMINAMATH_CALUDE_expand_expression_l3895_389565

theorem expand_expression (x : ℝ) : (2*x - 3) * (4*x + 9) = 8*x^2 + 6*x - 27 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3895_389565


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3895_389559

-- Problem 1
theorem problem_1 (x : ℝ) : 
  x / (2 * x - 3) + 5 / (3 - 2 * x) = 4 ↔ x = 1 :=
sorry

-- Problem 2
theorem problem_2 : 
  ¬∃ (x : ℝ), (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 :=
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3895_389559


namespace NUMINAMATH_CALUDE_prime_with_integer_roots_l3895_389523

theorem prime_with_integer_roots (p : ℕ) : 
  Prime p → 
  (∃ x y : ℤ, x^2 + p*x - 530*p = 0 ∧ y^2 + p*y - 530*p = 0) → 
  43 < p ∧ p ≤ 53 := by
sorry

end NUMINAMATH_CALUDE_prime_with_integer_roots_l3895_389523


namespace NUMINAMATH_CALUDE_cakes_served_total_l3895_389576

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := 5

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := 6

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := 3

/-- The total number of cakes served over two days -/
def total_cakes : ℕ := lunch_cakes + dinner_cakes + yesterday_cakes

theorem cakes_served_total :
  total_cakes = 14 := by sorry

end NUMINAMATH_CALUDE_cakes_served_total_l3895_389576


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l3895_389544

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^17 + i^203 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l3895_389544


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3895_389583

theorem inequality_solution_set (x : ℝ) : 
  (3 - x < x - 1) ↔ (x > 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3895_389583


namespace NUMINAMATH_CALUDE_max_y_minus_x_l3895_389551

theorem max_y_minus_x (p q : ℕ+) (x y : ℤ) 
  (h : x * y = p * x + q * y) 
  (max_y : ∀ (y' : ℤ), x * y' = p * x + q * y' → y' ≤ y) : 
  y - x = (p - 1) * (q + 1) := by
sorry

end NUMINAMATH_CALUDE_max_y_minus_x_l3895_389551


namespace NUMINAMATH_CALUDE_cab_journey_time_l3895_389596

/-- The usual time for a cab to cover a journey -/
def usual_time : ℝ → Prop :=
  λ T => (6 / 5 * T = T + 15) ∧ (T = 75)

theorem cab_journey_time :
  ∃ T : ℝ, usual_time T :=
sorry

end NUMINAMATH_CALUDE_cab_journey_time_l3895_389596


namespace NUMINAMATH_CALUDE_intersection_A_B_l3895_389549

def U : Set Int := {-1, 3, 5, 7, 9}
def complement_A : Set Int := {-1, 9}
def B : Set Int := {3, 7, 9}

theorem intersection_A_B :
  let A := U \ complement_A
  (A ∩ B) = {3, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3895_389549


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3895_389566

def U : Set ℕ := {x | 1 < x ∧ x < 6}
def A : Set ℕ := {2, 3}
def B : Set ℕ := {2, 4, 5}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3895_389566


namespace NUMINAMATH_CALUDE_equation_solution_l3895_389580

theorem equation_solution : 
  ∃ x : ℚ, (x ≠ 2) ∧ (7 * x / (x - 2) + 4 / (x - 2) = 6 / (x - 2)) → x = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3895_389580


namespace NUMINAMATH_CALUDE_ordering_of_powers_l3895_389504

theorem ordering_of_powers : 3^15 < 4^12 ∧ 4^12 < 8^9 := by
  sorry

end NUMINAMATH_CALUDE_ordering_of_powers_l3895_389504


namespace NUMINAMATH_CALUDE_mixture_ratio_change_l3895_389589

/-- Given an initial mixture of milk and water, prove the new ratio after adding water -/
theorem mixture_ratio_change (initial_volume : ℚ) (initial_milk_ratio : ℚ) (initial_water_ratio : ℚ) 
  (added_water : ℚ) (new_milk_ratio : ℚ) (new_water_ratio : ℚ) : 
  initial_volume = 60 ∧ 
  initial_milk_ratio = 2 ∧ 
  initial_water_ratio = 1 ∧ 
  added_water = 60 ∧ 
  new_milk_ratio = 1 ∧ 
  new_water_ratio = 2 →
  let initial_milk := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let initial_water := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let new_water := initial_water + added_water
  new_milk_ratio / new_water_ratio = initial_milk / new_water :=
by
  sorry


end NUMINAMATH_CALUDE_mixture_ratio_change_l3895_389589


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l3895_389553

theorem stratified_sampling_proportion (total : ℕ) (males : ℕ) (females_selected : ℕ) :
  total = 220 →
  males = 60 →
  females_selected = 32 →
  (males / total : ℚ) = ((12 : ℕ) / (12 + females_selected) : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l3895_389553


namespace NUMINAMATH_CALUDE_definite_integral_problem_l3895_389515

theorem definite_integral_problem : 
  ∫ x in Set.Icc (π/4) (π/2), (x * Real.cos x + Real.sin x) / (x * Real.sin x)^2 = (4 * Real.sqrt 2 - 2) / π := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_problem_l3895_389515


namespace NUMINAMATH_CALUDE_work_completion_time_l3895_389539

/-- Given workers A and B, where A can finish a job in 4 days and B in 14 days,
    prove that after working together for 2 days and A leaving,
    B will take 5 more days to finish the job. -/
theorem work_completion_time 
  (days_A : ℝ) 
  (days_B : ℝ) 
  (days_together : ℝ) 
  (h1 : days_A = 4) 
  (h2 : days_B = 14) 
  (h3 : days_together = 2) : 
  (days_B - (1 - (days_together * (1 / days_A + 1 / days_B))) / (1 / days_B)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3895_389539


namespace NUMINAMATH_CALUDE_special_function_is_one_l3895_389568

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a > 0 ∧ b > 0 →
    (f (a^2 + b^2) = f a * f b) ∧
    (f (a^2) = (f a)^2)

/-- The main theorem stating that any function satisfying the conditions is constant 1 -/
theorem special_function_is_one (f : ℕ → ℕ) (h : SpecialFunction f) :
  ∀ n : ℕ, n > 0 → f n = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_function_is_one_l3895_389568


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_one_l3895_389511

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- Given two parallel lines y = ax - 2 and y = (2-a)x + 1, prove that a = 1 -/
theorem parallel_lines_a_equals_one :
  (∀ x y : ℝ, y = a * x - 2 ↔ y = (2 - a) * x + 1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_one_l3895_389511


namespace NUMINAMATH_CALUDE_complementary_angles_problem_l3895_389524

theorem complementary_angles_problem (C D : Real) : 
  C + D = 90 →  -- Angles C and D are complementary
  C = 3 * D →   -- The measure of angle C is 3 times angle D
  C = 67.5 :=   -- The measure of angle C is 67.5°
by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_problem_l3895_389524


namespace NUMINAMATH_CALUDE_largest_angle_of_triangle_l3895_389530

/-- Given a triangle PQR with side lengths p, q, and r satisfying certain conditions,
    prove that its largest angle is 120 degrees. -/
theorem largest_angle_of_triangle (p q r : ℝ) (h1 : p + 3*q + 3*r = p^2) (h2 : p + 3*q - 3*r = -1) :
  ∃ (P Q R : ℝ), 
    P + Q + R = 180 ∧ 
    0 < P ∧ 0 < Q ∧ 0 < R ∧
    P ≤ 120 ∧ Q ≤ 120 ∧ R = 120 :=
sorry

end NUMINAMATH_CALUDE_largest_angle_of_triangle_l3895_389530


namespace NUMINAMATH_CALUDE_equation_equivalence_l3895_389548

theorem equation_equivalence :
  ∀ x : ℝ, (x^2 - 2*x - 9 = 0) ↔ ((x - 1)^2 = 10) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3895_389548


namespace NUMINAMATH_CALUDE_polynomial_condition_implies_monomial_l3895_389545

/-- A polynomial with nonnegative coefficients and degree ≤ n -/
def NonNegPolynomial (n : ℕ) := {p : Polynomial ℝ // p.degree ≤ n ∧ ∀ i, 0 ≤ p.coeff i}

theorem polynomial_condition_implies_monomial {n : ℕ} (P : NonNegPolynomial n) :
  (∀ x : ℝ, x > 0 → P.val.eval x * P.val.eval (1/x) ≤ (P.val.eval 1)^2) →
  ∃ (k : ℕ) (a : ℝ), k ≤ n ∧ a ≥ 0 ∧ P.val = Polynomial.monomial k a :=
sorry

end NUMINAMATH_CALUDE_polynomial_condition_implies_monomial_l3895_389545


namespace NUMINAMATH_CALUDE_correct_number_of_sons_l3895_389574

/-- Represents the problem of dividing land among sons --/
structure LandDivision where
  total_land : ℝ  -- Total land in hectares
  hectare_to_sqm : ℝ  -- Conversion factor from hectare to square meters
  profit_area : ℝ  -- Area in square meters that yields a certain profit
  profit_per_quarter : ℝ  -- Profit in dollars per quarter for profit_area
  son_yearly_profit : ℝ  -- Yearly profit for each son in dollars

/-- Calculate the number of sons based on land division --/
def calculate_sons (ld : LandDivision) : ℕ :=
  sorry

/-- Theorem stating the correct number of sons --/
theorem correct_number_of_sons (ld : LandDivision) 
  (h1 : ld.total_land = 3)
  (h2 : ld.hectare_to_sqm = 10000)
  (h3 : ld.profit_area = 750)
  (h4 : ld.profit_per_quarter = 500)
  (h5 : ld.son_yearly_profit = 10000) :
  calculate_sons ld = 8 := by
    sorry

end NUMINAMATH_CALUDE_correct_number_of_sons_l3895_389574


namespace NUMINAMATH_CALUDE_other_communities_count_l3895_389537

theorem other_communities_count (total_boys : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total_boys = 850 →
  muslim_percent = 46 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  ∃ (other_boys : ℕ), other_boys = 136 ∧ 
    (↑other_boys : ℚ) / total_boys = 1 - (muslim_percent + hindu_percent + sikh_percent) :=
by sorry

end NUMINAMATH_CALUDE_other_communities_count_l3895_389537


namespace NUMINAMATH_CALUDE_sufficient_necessary_condition_l3895_389587

theorem sufficient_necessary_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) ↔ a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_necessary_condition_l3895_389587


namespace NUMINAMATH_CALUDE_jerrys_remaining_debt_l3895_389547

/-- Given Jerry's debt payments over two months, calculate his remaining debt -/
theorem jerrys_remaining_debt (total_debt : ℕ) (first_payment : ℕ) (additional_payment : ℕ) :
  total_debt = 50 →
  first_payment = 12 →
  additional_payment = 3 →
  total_debt - (first_payment + (first_payment + additional_payment)) = 23 :=
by sorry

end NUMINAMATH_CALUDE_jerrys_remaining_debt_l3895_389547


namespace NUMINAMATH_CALUDE_pear_sales_problem_l3895_389572

theorem pear_sales_problem (morning_sales afternoon_sales total_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  afternoon_sales = 320 →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 480 :=
by sorry

end NUMINAMATH_CALUDE_pear_sales_problem_l3895_389572


namespace NUMINAMATH_CALUDE_modulo_graph_intercepts_sum_l3895_389569

theorem modulo_graph_intercepts_sum (m : Nat) (x₀ y₀ : Nat) : m = 7 →
  0 ≤ x₀ → x₀ < m →
  0 ≤ y₀ → y₀ < m →
  (2 * x₀) % m = 1 % m →
  (3 * y₀ + 1) % m = 0 →
  x₀ + y₀ = 6 := by
sorry

end NUMINAMATH_CALUDE_modulo_graph_intercepts_sum_l3895_389569


namespace NUMINAMATH_CALUDE_positive_solution_x_l3895_389591

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y + 3 * x + 2 * y = 12)
  (eq2 : y * z + 5 * y + 3 * z = 15)
  (eq3 : x * z + 5 * x + 4 * z = 40)
  (x_pos : x > 0) : x = 4 := by
sorry

end NUMINAMATH_CALUDE_positive_solution_x_l3895_389591


namespace NUMINAMATH_CALUDE_gcd_78_143_l3895_389590

theorem gcd_78_143 : Nat.gcd 78 143 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_78_143_l3895_389590


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3895_389528

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ -3/5 < a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3895_389528


namespace NUMINAMATH_CALUDE_divisibility_in_base_system_l3895_389510

theorem divisibility_in_base_system : ∃! (b : ℕ), b ≥ 8 ∧ (∃ (q : ℕ), 7 * b + 2 = q * (2 * b^2 + 7 * b + 5)) ∧ b = 8 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_in_base_system_l3895_389510


namespace NUMINAMATH_CALUDE_complex_number_location_l3895_389500

theorem complex_number_location :
  let z : ℂ := (1 + Complex.I) * (1 - 2 * Complex.I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l3895_389500


namespace NUMINAMATH_CALUDE_method_two_more_swims_at_300_method_one_cheaper_for_40_plus_swims_l3895_389505

-- Define the cost functions for both methods
def cost_method_one (x : ℕ) : ℕ := 120 + 10 * x
def cost_method_two (x : ℕ) : ℕ := 15 * x

-- Theorem 1: For a total cost of 300 yuan, Method two allows more swims
theorem method_two_more_swims_at_300 :
  ∃ (x y : ℕ), cost_method_one x = 300 ∧ cost_method_two y = 300 ∧ y > x :=
sorry

-- Theorem 2: For 40 or more swims, Method one is less expensive
theorem method_one_cheaper_for_40_plus_swims :
  ∀ x : ℕ, x ≥ 40 → cost_method_one x < cost_method_two x :=
sorry

end NUMINAMATH_CALUDE_method_two_more_swims_at_300_method_one_cheaper_for_40_plus_swims_l3895_389505


namespace NUMINAMATH_CALUDE_profit_share_difference_l3895_389518

def investment_A : ℕ := 8000
def investment_B : ℕ := 10000
def investment_C : ℕ := 12000
def profit_share_B : ℕ := 2500

theorem profit_share_difference :
  let ratio_A : ℕ := investment_A / 2000
  let ratio_B : ℕ := investment_B / 2000
  let ratio_C : ℕ := investment_C / 2000
  let part_value : ℕ := profit_share_B / ratio_B
  let profit_A : ℕ := ratio_A * part_value
  let profit_C : ℕ := ratio_C * part_value
  profit_C - profit_A = 1000 := by sorry

end NUMINAMATH_CALUDE_profit_share_difference_l3895_389518


namespace NUMINAMATH_CALUDE_trivia_team_tryouts_l3895_389506

theorem trivia_team_tryouts (not_picked : ℕ) (num_groups : ℕ) (students_per_group : ℕ) : 
  not_picked = 9 → num_groups = 3 → students_per_group = 9 → 
  not_picked + num_groups * students_per_group = 36 := by
sorry

end NUMINAMATH_CALUDE_trivia_team_tryouts_l3895_389506


namespace NUMINAMATH_CALUDE_translation_preserves_shape_and_size_l3895_389593

-- Define a geometric figure
def GeometricFigure := Type

-- Define a translation operation
def translate (F : GeometricFigure) (v : ℝ × ℝ) : GeometricFigure := sorry

-- Define properties of a figure
def shape (F : GeometricFigure) : Type := sorry
def size (F : GeometricFigure) : ℝ := sorry

-- Theorem: Translation preserves shape and size
theorem translation_preserves_shape_and_size (F : GeometricFigure) (v : ℝ × ℝ) :
  (shape (translate F v) = shape F) ∧ (size (translate F v) = size F) := by sorry

end NUMINAMATH_CALUDE_translation_preserves_shape_and_size_l3895_389593


namespace NUMINAMATH_CALUDE_modified_morse_code_symbols_l3895_389555

/-- The number of distinct symbols for a given sequence length -/
def symbolCount (n : Nat) : Nat :=
  2^n

/-- The total number of distinct symbols for sequences of length 1 to 5 -/
def totalSymbols : Nat :=
  (symbolCount 1) + (symbolCount 2) + (symbolCount 3) + (symbolCount 4) + (symbolCount 5)

/-- Theorem: The total number of distinct symbols in modified Morse code for sequences
    of length 1 to 5 is 62 -/
theorem modified_morse_code_symbols :
  totalSymbols = 62 := by
  sorry

end NUMINAMATH_CALUDE_modified_morse_code_symbols_l3895_389555


namespace NUMINAMATH_CALUDE_problem_solution_l3895_389595

def p (x a : ℝ) : Prop := (x - 3*a) * (x - a) < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

theorem problem_solution :
  (∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3)) ∧
  (∀ a : ℝ, (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x) ↔ (1 ≤ a ∧ a ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3895_389595


namespace NUMINAMATH_CALUDE_power_function_through_point_l3895_389527

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x^α

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f →
  f 2 = Real.sqrt 2 / 2 →
  f 4 = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3895_389527


namespace NUMINAMATH_CALUDE_karen_cookies_l3895_389557

/-- Given Karen's cookie distribution, prove she kept 10 for herself -/
theorem karen_cookies (total : ℕ) (grandparents : ℕ) (class_size : ℕ) (per_person : ℕ)
  (h1 : total = 50)
  (h2 : grandparents = 8)
  (h3 : class_size = 16)
  (h4 : per_person = 2) :
  total - (grandparents + class_size * per_person) = 10 := by
  sorry

#eval 50 - (8 + 16 * 2)  -- Expected output: 10

end NUMINAMATH_CALUDE_karen_cookies_l3895_389557


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3895_389513

/-- Given two perpendicular vectors a = (3, -1) and b = (x, -2), prove that x = -2/3 -/
theorem perpendicular_vectors_x_value :
  let a : Fin 2 → ℝ := ![3, -1]
  let b : Fin 2 → ℝ := ![x, -2]
  (∀ i, i < 2 → a i * b i = 0) →
  x = -2/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3895_389513


namespace NUMINAMATH_CALUDE_yellow_second_probability_l3895_389564

/-- Represents the contents of a bag of marbles -/
structure BagContents where
  red : ℕ
  black : ℕ
  yellow : ℕ
  blue : ℕ

/-- Calculates the probability of drawing a yellow marble second given the bag contents and rules -/
def probability_yellow_second (bag_a bag_b bag_c : BagContents) : ℚ :=
  let total_a := bag_a.red + bag_a.black
  let total_b := bag_b.yellow + bag_b.blue
  let total_c := bag_c.yellow + bag_c.blue
  let prob_red_a := bag_a.red / total_a
  let prob_black_a := bag_a.black / total_a
  let prob_yellow_b := bag_b.yellow / total_b
  let prob_yellow_c := bag_c.yellow / total_c
  prob_red_a * prob_yellow_b + prob_black_a * prob_yellow_c

/-- Theorem stating that the probability of drawing a yellow marble second is 1/3 -/
theorem yellow_second_probability :
  let bag_a : BagContents := { red := 3, black := 6, yellow := 0, blue := 0 }
  let bag_b : BagContents := { red := 0, black := 0, yellow := 6, blue := 4 }
  let bag_c : BagContents := { red := 0, black := 0, yellow := 2, blue := 8 }
  probability_yellow_second bag_a bag_b bag_c = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_yellow_second_probability_l3895_389564


namespace NUMINAMATH_CALUDE_tax_savings_calculation_l3895_389597

/-- Calculates the differential savings when tax rate is lowered -/
def differential_savings (income : ℝ) (old_rate new_rate : ℝ) : ℝ :=
  income * (old_rate - new_rate)

/-- Theorem: The differential savings for a taxpayer with an annual income
    of $42,400, when the tax rate is reduced from 42% to 32%, is $4,240 -/
theorem tax_savings_calculation :
  differential_savings 42400 0.42 0.32 = 4240 := by
  sorry

end NUMINAMATH_CALUDE_tax_savings_calculation_l3895_389597


namespace NUMINAMATH_CALUDE_max_total_points_l3895_389562

/-- Represents the carnival game setup and Tiffany's current state -/
structure CarnivalGame where
  initial_money : ℕ := 3
  game_cost : ℕ := 1
  rings_per_game : ℕ := 5
  red_points : ℕ := 2
  green_points : ℕ := 3
  blue_points : ℕ := 5
  blue_success_rate : ℚ := 1/10
  time_limit : ℕ := 1
  games_played : ℕ := 2
  current_red : ℕ := 4
  current_green : ℕ := 5
  current_blue : ℕ := 1

/-- Calculates the maximum possible points for a single game -/
def max_points_per_game (game : CarnivalGame) : ℕ :=
  game.rings_per_game * game.blue_points

/-- Calculates the current total points -/
def current_total_points (game : CarnivalGame) : ℕ :=
  game.current_red * game.red_points +
  game.current_green * game.green_points +
  game.current_blue * game.blue_points

/-- Theorem: The maximum total points Tiffany can achieve in three games is 53 -/
theorem max_total_points (game : CarnivalGame) :
  current_total_points game + max_points_per_game game = 53 :=
sorry

end NUMINAMATH_CALUDE_max_total_points_l3895_389562


namespace NUMINAMATH_CALUDE_min_value_of_S_l3895_389558

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The curve C representing the trajectory of the circle's center -/
def C (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- The dot product of two vectors represented by points -/
def dot_product (p1 p2 : Point) : ℝ :=
  p1.x * p2.x + p1.y * p2.y

/-- The area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- The function S to be minimized -/
noncomputable def S (a b : Point) : ℝ :=
  let o : Point := ⟨0, 0⟩
  let f : Point := ⟨1, 0⟩
  triangle_area o f a + triangle_area o a b

/-- The main theorem stating the minimum value of S -/
theorem min_value_of_S :
  ∀ a b : Point,
  C a → C b →
  dot_product a b = -4 →
  S a b ≥ 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_S_l3895_389558


namespace NUMINAMATH_CALUDE_second_largest_divided_by_smallest_remainder_l3895_389541

theorem second_largest_divided_by_smallest_remainder : ∃ (a b c d : ℕ),
  (a = 10 ∧ b = 11 ∧ c = 12 ∧ d = 13) →
  (a < b ∧ b < c ∧ c < d) →
  c % a = 2 := by
sorry

end NUMINAMATH_CALUDE_second_largest_divided_by_smallest_remainder_l3895_389541


namespace NUMINAMATH_CALUDE_subtraction_problem_l3895_389529

theorem subtraction_problem (x : ℤ) (h : x - 46 = 15) : x - 29 = 32 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3895_389529


namespace NUMINAMATH_CALUDE_fixed_point_of_line_l3895_389503

/-- The fixed point through which the line (2k+1)x+(k-1)y+(7-k)=0 passes for all real k -/
theorem fixed_point_of_line (k : ℝ) : 
  ∃! p : ℝ × ℝ, ∀ k : ℝ, (2*k + 1) * p.1 + (k - 1) * p.2 + (7 - k) = 0 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_l3895_389503


namespace NUMINAMATH_CALUDE_relationship_abc_l3895_389577

theorem relationship_abc : 
  let a : ℝ := (3/7)^(2/7)
  let b : ℝ := (2/7)^(3/7)
  let c : ℝ := (2/7)^(2/7)
  a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3895_389577


namespace NUMINAMATH_CALUDE_james_chores_time_l3895_389567

/-- Proves that James spends 12 hours on his chores given the conditions -/
theorem james_chores_time (vacuum_time : ℝ) (other_chores_factor : ℝ) : 
  vacuum_time = 3 →
  other_chores_factor = 3 →
  vacuum_time + (other_chores_factor * vacuum_time) = 12 := by
  sorry

end NUMINAMATH_CALUDE_james_chores_time_l3895_389567


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l3895_389507

noncomputable def f (x : ℝ) := Real.log (x^2 + 2*x - 3) / Real.log (1/2)

theorem f_monotone_increasing :
  StrictMonoOn f (Set.Iio (-3)) := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l3895_389507


namespace NUMINAMATH_CALUDE_two_equal_intercept_lines_l3895_389521

/-- A line passing through (2, 3) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The intercept of the line on both axes -/
  intercept : ℝ
  /-- The line passes through (2, 3) -/
  passes_through : intercept - 2 = 3 * (intercept - intercept) / intercept

/-- There are exactly two lines passing through (2, 3) with equal intercepts on both axes -/
theorem two_equal_intercept_lines : 
  ∃! (s : Finset EqualInterceptLine), s.card = 2 ∧ 
  (∀ l : EqualInterceptLine, l ∈ s) ∧
  (∀ l : EqualInterceptLine, l ∈ s → l.intercept ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_two_equal_intercept_lines_l3895_389521


namespace NUMINAMATH_CALUDE_borrowed_sum_proof_l3895_389561

/-- 
Given a principal P borrowed at 8% per annum simple interest for 8 years,
if the interest I is equal to P - 900, then P = 2500.
-/
theorem borrowed_sum_proof (P : ℝ) (I : ℝ) : 
  (I = P * 8 * 8 / 100) →   -- Simple interest formula
  (I = P - 900) →           -- Given condition
  P = 2500 := by
sorry

end NUMINAMATH_CALUDE_borrowed_sum_proof_l3895_389561


namespace NUMINAMATH_CALUDE_unit_digit_of_sum_factorials_100_l3895_389579

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem unit_digit_of_sum_factorials_100 :
  sum_factorials 100 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_sum_factorials_100_l3895_389579


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l3895_389563

theorem quadratic_equation_proof (x₁ x₂ k : ℝ) : 
  (x₁^2 - 6*x₁ + k = 0) →
  (x₂^2 - 6*x₂ + k = 0) →
  (x₁^2 * x₂^2 - x₁ - x₂ = 115) →
  (k = -11 ∧ ((x₁ = 3 + 2*Real.sqrt 5 ∧ x₂ = 3 - 2*Real.sqrt 5) ∨ 
              (x₁ = 3 - 2*Real.sqrt 5 ∧ x₂ = 3 + 2*Real.sqrt 5))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l3895_389563


namespace NUMINAMATH_CALUDE_max_a_value_l3895_389502

theorem max_a_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = a * x^2 - a * x + 1) →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) →
  a ≤ 8 ∧ ∃ b : ℝ, b > 8 ∧ ∃ y : ℝ, 0 ≤ y ∧ y ≤ 1 ∧ |b * y^2 - b * y + 1| > 1 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l3895_389502


namespace NUMINAMATH_CALUDE_smallest_nth_root_of_unity_l3895_389520

theorem smallest_nth_root_of_unity : ∃ (n : ℕ), n > 0 ∧ 
  (∀ z : ℂ, z^6 - z^3 + 1 = 0 → z^n = 1) ∧
  (∀ m : ℕ, m > 0 → (∀ z : ℂ, z^6 - z^3 + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_smallest_nth_root_of_unity_l3895_389520


namespace NUMINAMATH_CALUDE_tv_tower_height_l3895_389556

/-- The height of a TV tower given specific angle measurements and distances -/
theorem tv_tower_height (angle_A : Real) (angle_B : Real) (angle_southwest : Real) (distance_AB : Real) :
  angle_A = π / 3 →  -- 60 degrees in radians
  angle_B = π / 4 →  -- 45 degrees in radians
  angle_southwest = π / 6 →  -- 30 degrees in radians
  distance_AB = 35 →
  ∃ (height : Real), height = 5 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_tv_tower_height_l3895_389556


namespace NUMINAMATH_CALUDE_BC_time_is_three_hours_l3895_389592

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 4
def work_rate_C : ℚ := 1 / 12

-- Define the combined work rate of A and C
def work_rate_AC : ℚ := 1 / 3

-- Define the time taken by B and C together
def time_BC : ℚ := 1 / (work_rate_B + work_rate_C)

-- Theorem statement
theorem BC_time_is_three_hours :
  work_rate_A = 1 / 4 →
  work_rate_B = 1 / 4 →
  work_rate_AC = 1 / 3 →
  work_rate_C = work_rate_AC - work_rate_A →
  time_BC = 3 := by
  sorry


end NUMINAMATH_CALUDE_BC_time_is_three_hours_l3895_389592


namespace NUMINAMATH_CALUDE_initial_deer_families_l3895_389546

/-- The number of deer families that stayed in the area -/
def families_stayed : ℕ := 45

/-- The number of deer families that moved out of the area -/
def families_moved_out : ℕ := 34

/-- The initial number of deer families in the area -/
def initial_families : ℕ := families_stayed + families_moved_out

theorem initial_deer_families : initial_families = 79 := by
  sorry

end NUMINAMATH_CALUDE_initial_deer_families_l3895_389546


namespace NUMINAMATH_CALUDE_walmart_complaints_l3895_389585

/-- The number of complaints received by a Walmart store over a period of days --/
def total_complaints (normal_rate : ℝ) (short_staffed_factor : ℝ) (checkout_broken_factor : ℝ) (days : ℝ) : ℝ :=
  normal_rate * short_staffed_factor * checkout_broken_factor * days

/-- Theorem stating that under given conditions, the total complaints over 3 days is 576 --/
theorem walmart_complaints :
  total_complaints 120 (4/3) 1.2 3 = 576 := by
  sorry

end NUMINAMATH_CALUDE_walmart_complaints_l3895_389585


namespace NUMINAMATH_CALUDE_least_multiple_of_primes_l3895_389501

theorem least_multiple_of_primes : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m → n ≤ m) ∧ 
  3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ n = 105 := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_primes_l3895_389501


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l3895_389573

theorem discount_percentage_proof (wholesale_price retail_price : ℝ) 
  (profit_percentage : ℝ) (h1 : wholesale_price = 81) 
  (h2 : retail_price = 108) (h3 : profit_percentage = 0.2) : 
  (retail_price - (wholesale_price + wholesale_price * profit_percentage)) / retail_price = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l3895_389573


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3895_389586

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (· + 1)

/-- Theorem: In a geometric sequence where the fourth term is 6! and the seventh term is 7!, the first term is 720/7. -/
theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geometric : IsGeometricSequence a)
  (h_fourth : a 4 = factorial 6)
  (h_seventh : a 7 = factorial 7) :
  a 1 = 720 / 7 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3895_389586


namespace NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_a_l3895_389588

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

-- Theorem 1: Solution set of f(x) ≤ 6
theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2: Range of a for f(x) ≥ |2x+a| - 4 when x ∈ [-1/2, 1]
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1/2) 1, f x ≥ |2*x + a| - 4) ↔ -7 ≤ a ∧ a ≤ 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_a_l3895_389588


namespace NUMINAMATH_CALUDE_ron_chocolate_cost_l3895_389526

/-- Calculates the cost of chocolate bars for a boy scout camp out -/
def chocolate_cost (chocolate_bar_price : ℚ) (sections_per_bar : ℕ) (num_scouts : ℕ) (smores_per_scout : ℕ) : ℚ :=
  let total_smores := num_scouts * smores_per_scout
  let bars_needed := (total_smores + sections_per_bar - 1) / sections_per_bar
  bars_needed * chocolate_bar_price

/-- Theorem: The cost of chocolate bars for Ron's boy scout camp out is $15.00 -/
theorem ron_chocolate_cost :
  chocolate_cost (3/2) 3 15 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ron_chocolate_cost_l3895_389526


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_sqrt_three_over_three_l3895_389519

theorem sqrt_sum_equals_seven_sqrt_three_over_three :
  Real.sqrt 12 + Real.sqrt (1/3) = 7 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_sqrt_three_over_three_l3895_389519


namespace NUMINAMATH_CALUDE_van_tire_mileage_l3895_389538

/-- Calculates the miles each tire is used given the total miles traveled,
    number of tires, and number of tires used at a time. -/
def miles_per_tire (total_miles : ℕ) (num_tires : ℕ) (tires_in_use : ℕ) : ℚ :=
  (total_miles * tires_in_use : ℚ) / num_tires

/-- Proves that for a van with 7 tires, where 6 are used at a time,
    and the van travels 42,000 miles with all tires equally worn,
    each tire is used for 36,000 miles. -/
theorem van_tire_mileage :
  miles_per_tire 42000 7 6 = 36000 := by sorry

end NUMINAMATH_CALUDE_van_tire_mileage_l3895_389538


namespace NUMINAMATH_CALUDE_solve_system_l3895_389584

theorem solve_system (x y : ℤ) (h1 : x + y = 280) (h2 : x - y = 200) : y = 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3895_389584


namespace NUMINAMATH_CALUDE_sum_must_be_odd_l3895_389571

theorem sum_must_be_odd (x y : ℤ) (h : 7 * x + 5 * y = 11111) : 
  ¬(Even (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_sum_must_be_odd_l3895_389571


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3895_389575

theorem arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + d) → a 3 + a 7 = 2 * a 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3895_389575


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3895_389512

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₁ + a₆ + a₁₁ = 3,
    prove that a₃ + a₉ = 2 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 6 + a 11 = 3) : 
  a 3 + a 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3895_389512


namespace NUMINAMATH_CALUDE_triangle_properties_l3895_389525

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_properties (t : Triangle) 
  (h1 : Real.sqrt 3 * t.a * Real.sin t.B - t.b * Real.cos t.A = t.b)
  (h2 : t.b + t.c = 4) :
  t.A = π / 3 ∧ 
  (∃ (min_a : ℝ), min_a = 2 ∧ 
    (∀ a', t.a = a' → a' ≥ min_a) ∧
    (t.a = min_a → Real.sqrt 3 / 2 * t.b * t.c = Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3895_389525


namespace NUMINAMATH_CALUDE_demand_proportion_for_constant_income_l3895_389570

theorem demand_proportion_for_constant_income
  (original_price original_demand : ℝ)
  (price_increase_factor : ℝ := 1.20)
  (demand_increase_factor : ℝ := 1.12)
  (h_price_positive : original_price > 0)
  (h_demand_positive : original_demand > 0) :
  let new_price := price_increase_factor * original_price
  let new_demand := (14 / 15) * original_demand
  new_price * new_demand = original_price * original_demand :=
by sorry

end NUMINAMATH_CALUDE_demand_proportion_for_constant_income_l3895_389570


namespace NUMINAMATH_CALUDE_stock_price_change_l3895_389531

theorem stock_price_change (x : ℝ) : 
  (1 - x / 100) * 1.1 = 1 + 4.499999999999993 / 100 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l3895_389531


namespace NUMINAMATH_CALUDE_hannah_dog_food_l3895_389599

/-- The amount of dog food Hannah needs to prepare daily for her five dogs -/
def total_dog_food (dog1_meal : ℝ) (dog1_freq : ℕ) (dog2_ratio : ℝ) (dog2_freq : ℕ)
  (dog3_extra : ℝ) (dog3_freq : ℕ) (dog4_ratio : ℝ) (dog4_freq : ℕ)
  (dog5_ratio : ℝ) (dog5_freq : ℕ) : ℝ :=
  (dog1_meal * dog1_freq) +
  (dog1_meal * dog2_ratio * dog2_freq) +
  ((dog1_meal * dog2_ratio + dog3_extra) * dog3_freq) +
  (dog4_ratio * (dog1_meal * dog2_ratio + dog3_extra) * dog4_freq) +
  (dog5_ratio * dog1_meal * dog5_freq)

/-- Theorem stating that Hannah needs to prepare 40.5 cups of dog food daily -/
theorem hannah_dog_food : total_dog_food 1.5 2 2 1 2.5 3 1.2 2 0.8 4 = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_hannah_dog_food_l3895_389599


namespace NUMINAMATH_CALUDE_distance_between_foci_l3895_389582

-- Define the three known endpoints of the ellipse's axes
def point1 : ℝ × ℝ := (-3, 5)
def point2 : ℝ × ℝ := (4, -3)
def point3 : ℝ × ℝ := (9, 5)

-- Define the ellipse based on these points
def ellipse_from_points (p1 p2 p3 : ℝ × ℝ) : Type := sorry

-- Theorem stating the distance between foci
theorem distance_between_foci 
  (e : ellipse_from_points point1 point2 point3) : 
  ∃ (f1 f2 : ℝ × ℝ), dist f1 f2 = 4 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_distance_between_foci_l3895_389582


namespace NUMINAMATH_CALUDE_sqrt_3_and_sqrt_1_3_same_type_l3895_389534

/-- Two quadratic radicals are of the same type if they have the same radicand after simplification -/
def same_type (a b : ℝ) : Prop :=
  ∃ (k₁ k₂ r : ℝ), k₁ > 0 ∧ k₂ > 0 ∧ r > 0 ∧ a = k₁ * Real.sqrt r ∧ b = k₂ * Real.sqrt r

/-- √3 and √(1/3) are of the same type -/
theorem sqrt_3_and_sqrt_1_3_same_type : same_type (Real.sqrt 3) (Real.sqrt (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_and_sqrt_1_3_same_type_l3895_389534


namespace NUMINAMATH_CALUDE_catherine_caps_proof_l3895_389598

/-- The number of bottle caps Nicholas starts with -/
def initial_caps : ℕ := 8

/-- The number of bottle caps Nicholas ends up with -/
def final_caps : ℕ := 93

/-- The number of bottle caps Catherine gave to Nicholas -/
def catherine_caps : ℕ := final_caps - initial_caps

theorem catherine_caps_proof : catherine_caps = 85 := by
  sorry

end NUMINAMATH_CALUDE_catherine_caps_proof_l3895_389598


namespace NUMINAMATH_CALUDE_equation1_solutions_equation2_solution_l3895_389560

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * (x + 1)^2 - 2 = 25
def equation2 (x : ℝ) : Prop := (x - 1)^3 = 64

-- Theorem for equation1
theorem equation1_solutions :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation1 x₁ ∧ equation1 x₂ ∧ x₁ = 2 ∧ x₂ = -4 :=
sorry

-- Theorem for equation2
theorem equation2_solution :
  ∃ x : ℝ, equation2 x ∧ x = 5 :=
sorry

end NUMINAMATH_CALUDE_equation1_solutions_equation2_solution_l3895_389560


namespace NUMINAMATH_CALUDE_one_absent_one_present_probability_l3895_389550

/-- The probability of a student being absent on any given day -/
def absent_prob : ℚ := 1 / 30

/-- The probability of a student being present on any given day -/
def present_prob : ℚ := 1 - absent_prob

/-- The probability that out of two randomly chosen students, exactly one is absent while the other is present -/
def one_absent_one_present_prob : ℚ := 2 * (absent_prob * present_prob)

theorem one_absent_one_present_probability :
  one_absent_one_present_prob = 58 / 900 := by sorry

end NUMINAMATH_CALUDE_one_absent_one_present_probability_l3895_389550


namespace NUMINAMATH_CALUDE_mikey_new_leaves_l3895_389514

/-- The number of new leaves that came to Mikey -/
def new_leaves (initial final : ℝ) : ℝ := final - initial

/-- Proof that Mikey received 112 new leaves -/
theorem mikey_new_leaves :
  let initial : ℝ := 356.0
  let final : ℝ := 468
  new_leaves initial final = 112 := by sorry

end NUMINAMATH_CALUDE_mikey_new_leaves_l3895_389514


namespace NUMINAMATH_CALUDE_sokka_fish_count_l3895_389578

theorem sokka_fish_count (aang_fish : ℕ) (toph_fish : ℕ) (average_fish : ℕ) (total_people : ℕ) :
  aang_fish = 7 →
  toph_fish = 12 →
  average_fish = 8 →
  total_people = 3 →
  ∃ sokka_fish : ℕ, sokka_fish = total_people * average_fish - (aang_fish + toph_fish) :=
by
  sorry

end NUMINAMATH_CALUDE_sokka_fish_count_l3895_389578


namespace NUMINAMATH_CALUDE_divisibility_by_81_invariant_under_reversal_l3895_389533

/-- A sequence of digits represented as a list of natural numbers. -/
def DigitSequence := List Nat

/-- Check if a number represented by a digit sequence is divisible by 81. -/
def isDivisibleBy81 (digits : DigitSequence) : Prop :=
  digits.foldl (fun acc d => (10 * acc + d) % 81) 0 = 0

/-- Reverse a digit sequence. -/
def reverseDigits (digits : DigitSequence) : DigitSequence :=
  digits.reverse

theorem divisibility_by_81_invariant_under_reversal
  (digits : DigitSequence)
  (h : digits.length = 2016)
  (h_divisible : isDivisibleBy81 digits) :
  isDivisibleBy81 (reverseDigits digits) := by
  sorry

#check divisibility_by_81_invariant_under_reversal

end NUMINAMATH_CALUDE_divisibility_by_81_invariant_under_reversal_l3895_389533


namespace NUMINAMATH_CALUDE_function_value_2009_l3895_389581

theorem function_value_2009 (f : ℝ → ℝ) 
  (h1 : f 3 = -Real.sqrt 3) 
  (h2 : ∀ x : ℝ, f (x + 2) * (1 - f x) = 1 + f x) : 
  f 2009 = 2 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_function_value_2009_l3895_389581


namespace NUMINAMATH_CALUDE_mean_temperature_l3895_389554

def temperatures : List ℤ := [-8, -5, -3, 0, 4, 2, 7]

theorem mean_temperature :
  (temperatures.sum : ℚ) / temperatures.length = -3/7 := by sorry

end NUMINAMATH_CALUDE_mean_temperature_l3895_389554


namespace NUMINAMATH_CALUDE_popcorn_shrimp_orders_l3895_389536

/-- Proves that the number of popcorn shrimp orders is 9 given the conditions -/
theorem popcorn_shrimp_orders 
  (catfish_cost : ℝ) 
  (shrimp_cost : ℝ) 
  (total_orders : ℕ) 
  (total_amount : ℝ) 
  (h1 : catfish_cost = 6)
  (h2 : shrimp_cost = 3.5)
  (h3 : total_orders = 26)
  (h4 : total_amount = 133.5) :
  ∃ (catfish_orders shrimp_orders : ℕ), 
    catfish_orders + shrimp_orders = total_orders ∧ 
    catfish_cost * (catfish_orders : ℝ) + shrimp_cost * (shrimp_orders : ℝ) = total_amount ∧
    shrimp_orders = 9 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_shrimp_orders_l3895_389536


namespace NUMINAMATH_CALUDE_negation_equivalence_l3895_389552

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3895_389552


namespace NUMINAMATH_CALUDE_expression_equality_l3895_389540

theorem expression_equality : 2 + 2/3 + 6.3 - (5/3 - (1 + 3/5)) = 8.9 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3895_389540


namespace NUMINAMATH_CALUDE_apple_distribution_l3895_389517

theorem apple_distribution (n : ℕ) (k : ℕ) (min_apples : ℕ) : 
  n = 24 → k = 3 → min_apples = 2 → 
  (Nat.choose (n - k * min_apples + k - 1) (k - 1)) = 190 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l3895_389517


namespace NUMINAMATH_CALUDE_binomial_distribution_parameters_l3895_389543

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial distribution -/
def expectedValue (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p

/-- Variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_distribution_parameters :
  ∃ (ξ : BinomialDistribution), expectedValue ξ = 12 ∧ variance ξ = 2.4 ∧ ξ.n = 15 ∧ ξ.p = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_parameters_l3895_389543


namespace NUMINAMATH_CALUDE_train_stop_time_l3895_389542

/-- Proves that a train with given speeds including and excluding stoppages
    stops for 20 minutes per hour. -/
theorem train_stop_time
  (speed_without_stops : ℝ)
  (speed_with_stops : ℝ)
  (h1 : speed_without_stops = 60)
  (h2 : speed_with_stops = 40)
  : (1 - speed_with_stops / speed_without_stops) * 60 = 20 :=
by sorry

end NUMINAMATH_CALUDE_train_stop_time_l3895_389542
