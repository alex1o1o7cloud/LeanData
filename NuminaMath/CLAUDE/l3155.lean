import Mathlib

namespace NUMINAMATH_CALUDE_log_36_2_in_terms_of_a_b_l3155_315548

/-- Given lg 2 = a and lg 3 = b, prove that log_36 2 = (a + b) / b -/
theorem log_36_2_in_terms_of_a_b (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 3 = b) :
  (Real.log 2) / (Real.log 36) = (a + b) / b := by
  sorry

end NUMINAMATH_CALUDE_log_36_2_in_terms_of_a_b_l3155_315548


namespace NUMINAMATH_CALUDE_als_original_portion_l3155_315520

theorem als_original_portion (a b c : ℝ) : 
  a + b + c = 1200 →
  a - 150 + 3*b + 3*c = 1800 →
  a = 825 :=
by sorry

end NUMINAMATH_CALUDE_als_original_portion_l3155_315520


namespace NUMINAMATH_CALUDE_perpendicular_implies_cos_value_triangle_implies_f_range_l3155_315503

noncomputable section

-- Define the vectors m and n
def m (x : ℝ) : Fin 2 → ℝ := ![Real.sqrt 3 * Real.sin (x/4), 1]
def n (x : ℝ) : Fin 2 → ℝ := ![Real.cos (x/4), Real.cos (x/4)^2]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define perpendicularity
def perpendicular (v w : Fin 2 → ℝ) : Prop := dot_product v w = 0

-- Define the function f
def f (x : ℝ) : ℝ := dot_product (m x) (n x)

-- Theorem 1
theorem perpendicular_implies_cos_value (x : ℝ) :
  perpendicular (m x) (n x) → Real.cos (2 * Real.pi / 3 - x) = -1/2 := by sorry

-- Theorem 2
theorem triangle_implies_f_range (A B C a b c : ℝ) :
  A + B + C = Real.pi →
  (2 * a - c) * Real.cos B = b * Real.cos C →
  0 < A →
  A < 2 * Real.pi / 3 →
  ∃ (y : ℝ), 1 < f A ∧ f A < 3/2 := by sorry

end

end NUMINAMATH_CALUDE_perpendicular_implies_cos_value_triangle_implies_f_range_l3155_315503


namespace NUMINAMATH_CALUDE_function_composition_problem_l3155_315505

theorem function_composition_problem (k b : ℝ) (f : ℝ → ℝ) :
  (k < 0) →
  (∀ x, f x = k * x + b) →
  (∀ x, f (f x) = 4 * x + 1) →
  (∀ x, f x = -2 * x - 1) :=
by sorry

end NUMINAMATH_CALUDE_function_composition_problem_l3155_315505


namespace NUMINAMATH_CALUDE_locus_of_midpoint_l3155_315517

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 13

-- Define a point P on the circle
def point_P (x y : ℝ) : Prop := circle_O x y

-- Define Q as the foot of the perpendicular from P to the y-axis
def point_Q (x y : ℝ) : Prop := x = 0

-- Define M as the midpoint of PQ
def point_M (x y px py : ℝ) : Prop := x = px / 2 ∧ y = py

-- Theorem statement
theorem locus_of_midpoint :
  ∀ (x y px py : ℝ),
  point_P px py →
  point_Q 0 py →
  point_M x y px py →
  (x^2 / (13/4) + y^2 / 13 = 1) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_midpoint_l3155_315517


namespace NUMINAMATH_CALUDE_smallest_number_l3155_315534

theorem smallest_number (a b c d e : ℚ) : 
  a = 3.4 ∧ b = 7/2 ∧ c = 1.7 ∧ d = 27/10 ∧ e = 2.9 →
  c ≤ a ∧ c ≤ b ∧ c ≤ d ∧ c ≤ e := by
sorry

end NUMINAMATH_CALUDE_smallest_number_l3155_315534


namespace NUMINAMATH_CALUDE_investment_interest_proof_l3155_315535

/-- Calculate compound interest --/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Calculate total interest earned --/
def total_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  compound_interest principal rate years - principal

theorem investment_interest_proof :
  let principal := 1500
  let rate := 0.08
  let years := 5
  ∃ ε > 0, abs (total_interest principal rate years - 704) < ε :=
by sorry

end NUMINAMATH_CALUDE_investment_interest_proof_l3155_315535


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l3155_315544

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i_is_zero :
  i^23456 + i^23457 + i^23458 + i^23459 = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l3155_315544


namespace NUMINAMATH_CALUDE_three_hour_charge_l3155_315533

/-- Represents the therapy pricing structure and calculates total charges --/
structure TherapyPricing where
  first_hour : ℝ
  subsequent_hour : ℝ
  service_fee_rate : ℝ
  first_hour_premium : ℝ
  eight_hour_total : ℝ

/-- Calculates the total charge for a given number of hours --/
def total_charge (p : TherapyPricing) (hours : ℕ) : ℝ :=
  let base_charge := p.first_hour + p.subsequent_hour * (hours - 1)
  base_charge * (1 + p.service_fee_rate)

/-- Theorem stating the total charge for 3 hours of therapy --/
theorem three_hour_charge (p : TherapyPricing) : 
  p.first_hour = p.subsequent_hour + p.first_hour_premium →
  p.service_fee_rate = 0.1 →
  p.first_hour_premium = 50 →
  total_charge p 8 = p.eight_hour_total →
  p.eight_hour_total = 900 →
  total_charge p 3 = 371.87 := by
  sorry

end NUMINAMATH_CALUDE_three_hour_charge_l3155_315533


namespace NUMINAMATH_CALUDE_power_equality_l3155_315527

theorem power_equality (n : ℕ) : 2^n = 8^20 → n = 60 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3155_315527


namespace NUMINAMATH_CALUDE_smaller_root_of_quadratic_l3155_315511

theorem smaller_root_of_quadratic (x : ℝ) : 
  (x + 1) * (x - 1) = 0 → x = -1 ∨ x = 1 → -1 ≤ 1 → -1 = min x (-x) := by
sorry

end NUMINAMATH_CALUDE_smaller_root_of_quadratic_l3155_315511


namespace NUMINAMATH_CALUDE_simplify_2A_minus_B_value_2A_minus_B_special_case_l3155_315507

/-- Definition of A in terms of a and b -/
def A (a b : ℝ) : ℝ := b^2 - a^2 + 5*a*b

/-- Definition of B in terms of a and b -/
def B (a b : ℝ) : ℝ := 3*a*b + 2*b^2 - a^2

/-- Theorem stating the simplified form of 2A - B -/
theorem simplify_2A_minus_B (a b : ℝ) : 2 * A a b - B a b = -a^2 + 7*a*b := by sorry

/-- Theorem stating the value of 2A - B when a = 1 and b = 2 -/
theorem value_2A_minus_B_special_case : 2 * A 1 2 - B 1 2 = 13 := by sorry

end NUMINAMATH_CALUDE_simplify_2A_minus_B_value_2A_minus_B_special_case_l3155_315507


namespace NUMINAMATH_CALUDE_max_garden_area_optimal_garden_exists_l3155_315526

/-- Represents a rectangular garden with three sides fenced -/
structure Garden where
  length : ℝ
  width : ℝ
  fence_length : ℝ
  fence_constraint : fence_length = length + 2 * width

/-- The area of a rectangular garden -/
def garden_area (g : Garden) : ℝ := g.length * g.width

/-- Theorem stating the maximum area of the garden under given constraints -/
theorem max_garden_area :
  ∀ g : Garden,
  g.fence_length = 160 →
  garden_area g ≤ 3200 ∧
  (garden_area g = 3200 ↔ g.length = 80 ∧ g.width = 40) :=
by sorry

/-- Existence of the optimal garden -/
theorem optimal_garden_exists :
  ∃ g : Garden, g.fence_length = 160 ∧ garden_area g = 3200 ∧ g.length = 80 ∧ g.width = 40 :=
by sorry

end NUMINAMATH_CALUDE_max_garden_area_optimal_garden_exists_l3155_315526


namespace NUMINAMATH_CALUDE_bus_speed_without_stoppages_l3155_315529

theorem bus_speed_without_stoppages 
  (speed_with_stoppages : ℝ) 
  (stop_time : ℝ) 
  (h1 : speed_with_stoppages = 45) 
  (h2 : stop_time = 10) : 
  speed_with_stoppages * (60 / (60 - stop_time)) = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_bus_speed_without_stoppages_l3155_315529


namespace NUMINAMATH_CALUDE_lucy_groceries_l3155_315519

/-- The number of packs of cookies Lucy bought -/
def cookies : ℕ := 4

/-- The number of packs of cake Lucy bought -/
def cake : ℕ := 22

/-- The number of packs of chocolate Lucy bought -/
def chocolate : ℕ := 16

/-- The total number of packs of groceries Lucy bought -/
def total_groceries : ℕ := cookies + cake + chocolate

theorem lucy_groceries : total_groceries = 42 := by
  sorry

end NUMINAMATH_CALUDE_lucy_groceries_l3155_315519


namespace NUMINAMATH_CALUDE_fourth_power_modulo_thirteen_l3155_315540

theorem fourth_power_modulo_thirteen (a d : ℤ) (h_pos : d > 0) 
  (h_div : d ∣ a^4 + a^3 + 2*a^2 - 4*a + 3) : 
  ∃ x : ℤ, d ≡ x^4 [ZMOD 13] := by
sorry

end NUMINAMATH_CALUDE_fourth_power_modulo_thirteen_l3155_315540


namespace NUMINAMATH_CALUDE_geometric_mean_relationship_l3155_315518

theorem geometric_mean_relationship (m : ℝ) : 
  (m = 4 → m^2 = 2 * 8) ∧ ¬(m^2 = 2 * 8 → m = 4) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_relationship_l3155_315518


namespace NUMINAMATH_CALUDE_smallest_difference_l3155_315502

def Digits : Finset Nat := {0, 3, 4, 7, 8}

def isValidArrangement (a b c d e : Nat) : Prop :=
  a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧ e ∈ Digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a ≠ 0

def difference (a b c d e : Nat) : Nat :=
  (100 * a + 10 * b + c) - (10 * d + e)

theorem smallest_difference :
  ∀ a b c d e,
    isValidArrangement a b c d e →
    difference a b c d e ≥ 339 :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_l3155_315502


namespace NUMINAMATH_CALUDE_motorcycle_toll_correct_l3155_315580

/-- Represents the weekly commute scenario for Geordie --/
structure CommuteScenario where
  workDaysPerWeek : ℕ
  carToll : ℚ
  mpg : ℚ
  commuteDistance : ℚ
  gasPrice : ℚ
  carTripsPerWeek : ℕ
  motorcycleTripsPerWeek : ℕ
  totalWeeklyCost : ℚ

/-- Calculates the motorcycle toll given a commute scenario --/
def calculateMotorcycleToll (scenario : CommuteScenario) : ℚ :=
  sorry

/-- Theorem stating that the calculated motorcycle toll is correct --/
theorem motorcycle_toll_correct (scenario : CommuteScenario) :
  scenario.workDaysPerWeek = 5 ∧
  scenario.carToll = 25/2 ∧
  scenario.mpg = 35 ∧
  scenario.commuteDistance = 14 ∧
  scenario.gasPrice = 15/4 ∧
  scenario.carTripsPerWeek = 3 ∧
  scenario.motorcycleTripsPerWeek = 2 ∧
  scenario.totalWeeklyCost = 118 →
  calculateMotorcycleToll scenario = 131/4 :=
sorry

end NUMINAMATH_CALUDE_motorcycle_toll_correct_l3155_315580


namespace NUMINAMATH_CALUDE_max_stamps_purchasable_l3155_315585

theorem max_stamps_purchasable (stamp_price : ℕ) (discounted_price : ℕ) (budget : ℕ) :
  stamp_price = 50 →
  discounted_price = 45 →
  budget = 5000 →
  (∀ n : ℕ, n ≤ 50 → n * stamp_price ≤ budget) →
  (∀ n : ℕ, n > 50 → 50 * stamp_price + (n - 50) * discounted_price ≤ budget) →
  (∃ n : ℕ, n = 105 ∧
    (∀ m : ℕ, m > n → 
      (m ≤ 50 → m * stamp_price > budget) ∧
      (m > 50 → 50 * stamp_price + (m - 50) * discounted_price > budget))) :=
by sorry

end NUMINAMATH_CALUDE_max_stamps_purchasable_l3155_315585


namespace NUMINAMATH_CALUDE_gcd_5280_2155_l3155_315516

theorem gcd_5280_2155 : Nat.gcd 5280 2155 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5280_2155_l3155_315516


namespace NUMINAMATH_CALUDE_base_number_proof_l3155_315582

theorem base_number_proof (base : ℝ) : base ^ 7 = 3 ^ 14 → base = 9 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l3155_315582


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_8_with_digit_sum_20_l3155_315560

/-- Returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Returns true if the number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_multiple_of_8_with_digit_sum_20 :
  ∀ n : ℕ, is_four_digit n → n % 8 = 0 → digit_sum n = 20 → n ≥ 1071 :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_8_with_digit_sum_20_l3155_315560


namespace NUMINAMATH_CALUDE_bob_pennies_l3155_315510

theorem bob_pennies (a b : ℕ) : 
  (b + 2 = 4 * (a - 2)) →
  (b - 2 = 3 * (a + 2)) →
  b = 62 :=
by sorry

end NUMINAMATH_CALUDE_bob_pennies_l3155_315510


namespace NUMINAMATH_CALUDE_candidates_per_state_l3155_315588

theorem candidates_per_state (candidates : ℕ) : 
  (candidates * 6 / 100 : ℚ) + 80 = candidates * 7 / 100 → candidates = 8000 := by
  sorry

end NUMINAMATH_CALUDE_candidates_per_state_l3155_315588


namespace NUMINAMATH_CALUDE_union_of_sets_l3155_315531

def A (m : ℝ) : Set ℝ := {2, 2^m}
def B (m n : ℝ) : Set ℝ := {m, n}

theorem union_of_sets (m n : ℝ) 
  (h : A m ∩ B m n = {1/4}) : 
  A m ∪ B m n = {2, -2, 1/4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3155_315531


namespace NUMINAMATH_CALUDE_quadratic_inequality_bound_l3155_315522

theorem quadratic_inequality_bound (d : ℝ) : 
  (∀ x : ℝ, x * (4 * x - 3) < d ↔ -5/2 < x ∧ x < 3) ↔ d = 39 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_bound_l3155_315522


namespace NUMINAMATH_CALUDE_salt_mixture_proof_l3155_315572

theorem salt_mixture_proof (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_volume : ℝ) (added_concentration : ℝ) (final_concentration : ℝ) :
  initial_volume = 40 ∧ 
  initial_concentration = 0.2 ∧ 
  added_volume = 40 ∧ 
  added_concentration = 0.6 ∧ 
  final_concentration = 0.4 →
  (initial_volume * initial_concentration + added_volume * added_concentration) / 
  (initial_volume + added_volume) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_salt_mixture_proof_l3155_315572


namespace NUMINAMATH_CALUDE_expected_value_coin_flip_l3155_315573

/-- The expected value of winnings for a coin flip game -/
theorem expected_value_coin_flip :
  let p_heads : ℚ := 2 / 5
  let p_tails : ℚ := 3 / 5
  let win_heads : ℚ := 5
  let loss_tails : ℚ := 3
  p_heads * win_heads - p_tails * loss_tails = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_coin_flip_l3155_315573


namespace NUMINAMATH_CALUDE_complex_real_condition_l3155_315543

theorem complex_real_condition (a : ℝ) : 
  (((1 : ℂ) + Complex.I) ^ 2 - a / Complex.I).im = 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l3155_315543


namespace NUMINAMATH_CALUDE_ellipse_sum_specific_l3155_315571

/-- The sum of the center coordinates and axis lengths of an ellipse -/
def ellipse_sum (h k a b : ℝ) : ℝ := h + k + a + b

/-- Theorem: The sum of center coordinates and axis lengths for a specific ellipse -/
theorem ellipse_sum_specific : ∃ (h k a b : ℝ), 
  (∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) ∧ 
  h = 3 ∧ 
  k = -5 ∧ 
  a = 7 ∧ 
  b = 4 ∧ 
  ellipse_sum h k a b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_specific_l3155_315571


namespace NUMINAMATH_CALUDE_largest_six_digit_with_product_60_l3155_315565

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def digit_product (n : ℕ) : ℕ := (n.digits 10).prod

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

theorem largest_six_digit_with_product_60 :
  ∃ M : ℕ, is_six_digit M ∧ 
           digit_product M = 60 ∧ 
           (∀ n : ℕ, is_six_digit n → digit_product n = 60 → n ≤ M) ∧
           digit_sum M = 15 := by
  sorry

end NUMINAMATH_CALUDE_largest_six_digit_with_product_60_l3155_315565


namespace NUMINAMATH_CALUDE_exists_linear_approximation_l3155_315506

/-- Cyclic distance in Fp -/
def cyclic_distance (p : ℕ) (x : Fin p) : ℕ :=
  min x.val (p - x.val)

/-- Almost additive function property -/
def almost_additive (p : ℕ) (f : Fin p → Fin p) : Prop :=
  ∀ x y : Fin p, cyclic_distance p (f (x + y) - f x - f y) < 100

/-- Main theorem -/
theorem exists_linear_approximation
  (p : ℕ) (hp : Nat.Prime p) (f : Fin p → Fin p) (hf : almost_additive p f) :
  ∃ m : Fin p, ∀ x : Fin p, cyclic_distance p (f x - m * x) < 1000 :=
sorry

end NUMINAMATH_CALUDE_exists_linear_approximation_l3155_315506


namespace NUMINAMATH_CALUDE_pastry_sale_revenue_l3155_315570

/-- Calculates the total money made from selling discounted pastries -/
theorem pastry_sale_revenue 
  (original_cupcake_price original_cookie_price : ℚ)
  (cupcakes_sold cookies_sold : ℕ)
  (h1 : original_cupcake_price = 3)
  (h2 : original_cookie_price = 2)
  (h3 : cupcakes_sold = 16)
  (h4 : cookies_sold = 8) :
  (cupcakes_sold : ℚ) * (original_cupcake_price / 2) + 
  (cookies_sold : ℚ) * (original_cookie_price / 2) = 32 := by
sorry


end NUMINAMATH_CALUDE_pastry_sale_revenue_l3155_315570


namespace NUMINAMATH_CALUDE_method1_is_optimal_l3155_315583

/-- Represents the three methods of division available to the economist. -/
inductive DivisionMethod
  | method1
  | method2
  | method3

/-- Represents a division of coins. -/
structure Division where
  total : ℕ
  part1 : ℕ
  part2 : ℕ
  part3 : ℕ
  part4 : ℕ

/-- The coin division problem. -/
def CoinDivisionProblem (n : ℕ) (method : DivisionMethod) : Prop :=
  -- The total number of coins is odd and greater than 4
  n % 2 = 1 ∧ n > 4 ∧
  -- There exists a valid division of coins
  ∃ (div : Division),
    -- The total number of coins is correct
    div.total = n ∧
    -- Each part has at least one coin
    div.part1 ≥ 1 ∧ div.part2 ≥ 1 ∧ div.part3 ≥ 1 ∧ div.part4 ≥ 1 ∧
    -- The sum of all parts equals the total
    div.part1 + div.part2 + div.part3 + div.part4 = n ∧
    -- The lawyer's initial division results in two parts with at least 2 coins each
    div.part1 + div.part2 ≥ 2 ∧ div.part3 + div.part4 ≥ 2 ∧
    -- Method 1 is the optimal strategy
    (method = DivisionMethod.method1 →
      div.part1 + div.part4 > div.part2 + div.part3 ∧
      div.part1 + div.part4 > div.part1 + div.part2 - 1)

/-- Theorem stating that Method 1 is the optimal strategy for the economist. -/
theorem method1_is_optimal (n : ℕ) :
  CoinDivisionProblem n DivisionMethod.method1 := by
  sorry


end NUMINAMATH_CALUDE_method1_is_optimal_l3155_315583


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3155_315595

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) + 2 * x = x * f y + 3 * f x

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → f (-1) = 7 → f (-1001) = -3493 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3155_315595


namespace NUMINAMATH_CALUDE_function_equality_l3155_315575

theorem function_equality (k : ℝ) (x : ℝ) (h1 : k > 0) (h2 : x ≠ Real.sqrt k) :
  (x^2 - k) / (x - Real.sqrt k) = 3 * x → x = Real.sqrt k / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l3155_315575


namespace NUMINAMATH_CALUDE_michaels_bills_l3155_315599

def total_amount : ℕ := 280
def bill_denomination : ℕ := 20

theorem michaels_bills : 
  total_amount / bill_denomination = 14 :=
by sorry

end NUMINAMATH_CALUDE_michaels_bills_l3155_315599


namespace NUMINAMATH_CALUDE_quadratic_equation_transformation_l3155_315568

theorem quadratic_equation_transformation (p q : ℝ) :
  (∀ x, 4 * x^2 - p * x + q = 0 ↔ (x - 1/4)^2 = 33/16) →
  q / p = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_transformation_l3155_315568


namespace NUMINAMATH_CALUDE_special_three_digit_numbers_l3155_315598

-- Define a three-digit number
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Define the sum of digits for a three-digit number
def sum_of_digits (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

-- Define the condition for the special property
def has_special_property (n : ℕ) : Prop :=
  n = sum_of_digits n + 2 * (sum_of_digits n)^2

-- Theorem statement
theorem special_three_digit_numbers : 
  ∀ n : ℕ, is_three_digit n ∧ has_special_property n ↔ n = 171 ∨ n = 465 ∨ n = 666 := by
  sorry

end NUMINAMATH_CALUDE_special_three_digit_numbers_l3155_315598


namespace NUMINAMATH_CALUDE_card_count_theorem_l3155_315597

/-- Represents the number of baseball cards each person has -/
structure CardCounts where
  brandon : ℕ
  malcom : ℕ
  ella : ℕ
  lily : ℕ
  mark : ℕ

/-- Calculates the final card counts after all transactions -/
def finalCardCounts (initial : CardCounts) : CardCounts :=
  let malcomToMark := (initial.malcom * 3) / 5
  let ellaToLily := initial.ella / 4
  { brandon := initial.brandon
  , malcom := initial.malcom - malcomToMark
  , ella := initial.ella - ellaToLily
  , lily := initial.lily + ellaToLily
  , mark := malcomToMark + 6 }

/-- Theorem stating the correctness of the final card counts -/
theorem card_count_theorem (initial : CardCounts) :
  initial.brandon = 20 →
  initial.malcom = initial.brandon + 12 →
  initial.ella = initial.malcom - 5 →
  initial.lily = 2 * initial.ella →
  initial.mark = 0 →
  let final := finalCardCounts initial
  final.brandon = 20 ∧
  final.malcom = 13 ∧
  final.ella = 21 ∧
  final.lily = 60 ∧
  final.mark = 25 :=
by
  sorry


end NUMINAMATH_CALUDE_card_count_theorem_l3155_315597


namespace NUMINAMATH_CALUDE_car_distance_traveled_l3155_315566

theorem car_distance_traveled (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 23 → time = 3 → distance = speed * time → distance = 69 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_traveled_l3155_315566


namespace NUMINAMATH_CALUDE_f_maximum_l3155_315545

/-- The quadratic function f(x) = -3x^2 + 9x + 24 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x + 24

/-- The point where f attains its maximum -/
def x_max : ℝ := 1.5

theorem f_maximum :
  ∀ x : ℝ, f x ≤ f x_max := by sorry

end NUMINAMATH_CALUDE_f_maximum_l3155_315545


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_nine_l3155_315542

theorem six_digit_divisible_by_nine :
  ∃! d : ℕ, d < 10 ∧ (135790 + d) % 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_nine_l3155_315542


namespace NUMINAMATH_CALUDE_midpoint_theorem_l3155_315593

/-- Given points A, B, and C in a 2D plane, where B is the midpoint of AC -/
structure Midpoint2D where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The midpoint condition for 2D points -/
def isMidpoint (m : Midpoint2D) : Prop :=
  m.B.1 = (m.A.1 + m.C.1) / 2 ∧ m.B.2 = (m.A.2 + m.C.2) / 2

/-- Theorem: If B(3,4) is the midpoint of AC where A(1,1), then C is (5,7) -/
theorem midpoint_theorem (m : Midpoint2D) 
    (h1 : m.A = (1, 1))
    (h2 : m.B = (3, 4))
    (h3 : isMidpoint m) :
    m.C = (5, 7) := by
  sorry


end NUMINAMATH_CALUDE_midpoint_theorem_l3155_315593


namespace NUMINAMATH_CALUDE_constant_sequence_l3155_315578

def sequence_condition (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ m n : ℕ, m > 0 → n > 0 → |a n - a m| ≤ (2 * m * n : ℝ) / ((m^2 + n^2) : ℝ)

theorem constant_sequence (a : ℕ → ℝ) (h : sequence_condition a) : ∀ n : ℕ, n > 0 → a n = 1 :=
sorry

end NUMINAMATH_CALUDE_constant_sequence_l3155_315578


namespace NUMINAMATH_CALUDE_exists_n_with_constant_term_l3155_315552

/-- A function that checks if the expansion of (x - 1/x³)ⁿ contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ r : ℕ, n = 4 * r

/-- Theorem stating that there exists an n between 3 and 16 (inclusive) 
    such that the expansion of (x - 1/x³)ⁿ contains a constant term -/
theorem exists_n_with_constant_term : 
  ∃ n : ℕ, 3 ≤ n ∧ n ≤ 16 ∧ has_constant_term n :=
sorry

end NUMINAMATH_CALUDE_exists_n_with_constant_term_l3155_315552


namespace NUMINAMATH_CALUDE_phone_price_reduction_l3155_315524

theorem phone_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 2000)
  (h2 : final_price = 1280)
  (h3 : final_price = initial_price * (1 - x)^2)
  (h4 : x > 0 ∧ x < 1) :
  x = 0.18 := by sorry

end NUMINAMATH_CALUDE_phone_price_reduction_l3155_315524


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3155_315536

/-- A geometric sequence with negative terms -/
def NegativeGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n < 0

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  NegativeGeometricSequence a →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36 →
  a 3 + a 5 = -6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3155_315536


namespace NUMINAMATH_CALUDE_remainder_sum_powers_mod_5_l3155_315547

theorem remainder_sum_powers_mod_5 : (9^5 + 11^6 + 12^7) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_powers_mod_5_l3155_315547


namespace NUMINAMATH_CALUDE_oranges_apples_balance_l3155_315594

/-- Given that 9 oranges have the same weight as 6 apples, 
    this function calculates the number of apples that would 
    balance the weight of a given number of oranges. -/
def applesEquivalent (numOranges : ℕ) : ℕ :=
  (2 * numOranges) / 3

/-- Theorem stating that 30 apples balance the weight of 45 oranges, 
    given the weight ratio between oranges and apples. -/
theorem oranges_apples_balance :
  applesEquivalent 45 = 30 := by
  sorry

end NUMINAMATH_CALUDE_oranges_apples_balance_l3155_315594


namespace NUMINAMATH_CALUDE_determinant_zero_l3155_315539

-- Define the cubic equation
def cubic_equation (x s t : ℝ) : Prop := x^3 + s*x^2 + t*x = 0

-- Define the determinant of the 3x3 matrix
def matrix_determinant (x y z : ℝ) : ℝ :=
  x * (z * y - x * x) - y * (y * y - x * z) + z * (y * z - z * x)

-- Theorem statement
theorem determinant_zero (x y z s t : ℝ) 
  (hx : cubic_equation x s t) 
  (hy : cubic_equation y s t) 
  (hz : cubic_equation z s t) : 
  matrix_determinant x y z = 0 := by sorry

end NUMINAMATH_CALUDE_determinant_zero_l3155_315539


namespace NUMINAMATH_CALUDE_distance_difference_l3155_315561

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Grayson's first leg speed in mph -/
def grayson_speed1 : ℝ := 25

/-- Grayson's first leg time in hours -/
def grayson_time1 : ℝ := 1

/-- Grayson's second leg speed in mph -/
def grayson_speed2 : ℝ := 20

/-- Grayson's second leg time in hours -/
def grayson_time2 : ℝ := 0.5

/-- Rudy's speed in mph -/
def rudy_speed : ℝ := 10

/-- Rudy's time in hours -/
def rudy_time : ℝ := 3

/-- The difference in distance traveled between Grayson and Rudy -/
theorem distance_difference : 
  distance grayson_speed1 grayson_time1 + distance grayson_speed2 grayson_time2 - 
  distance rudy_speed rudy_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l3155_315561


namespace NUMINAMATH_CALUDE_interval_intersection_l3155_315508

theorem interval_intersection (x : ℝ) : 
  (3/4 < x ∧ x < 5/4) ↔ (2 < 3*x ∧ 3*x < 4) ∧ (3 < 4*x ∧ 4*x < 5) := by
  sorry

end NUMINAMATH_CALUDE_interval_intersection_l3155_315508


namespace NUMINAMATH_CALUDE_power_of_two_unique_sum_of_squares_l3155_315501

/-- 
For any non-negative integer n, there exists a unique pair of non-negative integers (a, b),
up to order, such that 2^n = a² + b².
-/
theorem power_of_two_unique_sum_of_squares (n : ℕ) :
  ∃! (p : ℕ × ℕ), 2^n = p.1^2 + p.2^2 ∧ 
  (∀ (q : ℕ × ℕ), 2^n = q.1^2 + q.2^2 → (p = q ∨ p = (q.2, q.1))) :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_unique_sum_of_squares_l3155_315501


namespace NUMINAMATH_CALUDE_complement_M_equals_interval_l3155_315554

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x : ℝ | (2 - x) / (x + 3) < 0}

-- Define the complement of M in ℝ
def complement_M : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem complement_M_equals_interval : 
  (U \ M) = complement_M :=
sorry

end NUMINAMATH_CALUDE_complement_M_equals_interval_l3155_315554


namespace NUMINAMATH_CALUDE_radical_simplification_l3155_315558

theorem radical_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p^3) * Real.sqrt (3 * p^5) * Real.sqrt (4 * p^2) / Real.sqrt (2 * p) = 6 * p^(9/2) * Real.sqrt (5/2) :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l3155_315558


namespace NUMINAMATH_CALUDE_son_age_problem_l3155_315589

theorem son_age_problem (son_age father_age : ℕ) : 
  father_age = son_age + 46 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 44 := by
sorry

end NUMINAMATH_CALUDE_son_age_problem_l3155_315589


namespace NUMINAMATH_CALUDE_y_value_proof_l3155_315562

theorem y_value_proof (y : ℝ) (h : (9 : ℝ) / y^3 = y / 81) : y = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l3155_315562


namespace NUMINAMATH_CALUDE_textbook_savings_l3155_315521

/-- Calculates the savings when buying textbooks from alternative bookshops instead of the school bookshop -/
theorem textbook_savings 
  (math_school_price : ℝ) 
  (science_school_price : ℝ) 
  (literature_school_price : ℝ)
  (math_discount : ℝ) 
  (science_discount : ℝ) 
  (literature_discount : ℝ)
  (school_tax_rate : ℝ)
  (alt_tax_rate : ℝ)
  (shipping_cost : ℝ)
  (h1 : math_school_price = 45)
  (h2 : science_school_price = 60)
  (h3 : literature_school_price = 35)
  (h4 : math_discount = 0.2)
  (h5 : science_discount = 0.25)
  (h6 : literature_discount = 0.15)
  (h7 : school_tax_rate = 0.07)
  (h8 : alt_tax_rate = 0.06)
  (h9 : shipping_cost = 10) :
  let school_total := (math_school_price + science_school_price + literature_school_price) * (1 + school_tax_rate)
  let alt_total := (math_school_price * (1 - math_discount) + 
                    science_school_price * (1 - science_discount) + 
                    literature_school_price * (1 - literature_discount)) * (1 + alt_tax_rate) + shipping_cost
  school_total - alt_total = 22.4 := by
  sorry


end NUMINAMATH_CALUDE_textbook_savings_l3155_315521


namespace NUMINAMATH_CALUDE_second_discount_percentage_l3155_315500

theorem second_discount_percentage
  (list_price : ℝ)
  (final_price : ℝ)
  (first_discount : ℝ)
  (h1 : list_price = 150)
  (h2 : final_price = 105)
  (h3 : first_discount = 19.954259576901087)
  : ∃ (second_discount : ℝ), 
    abs (second_discount - 12.552) < 0.001 ∧
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l3155_315500


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3155_315579

theorem polar_to_rectangular_conversion :
  let r : ℝ := 3
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = 3 * Real.sqrt 2 / 2 ∧ y = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3155_315579


namespace NUMINAMATH_CALUDE_infinite_primes_quadratic_equation_l3155_315512

theorem infinite_primes_quadratic_equation :
  ∀ (S : Finset Nat), ∃ (p : Nat) (x y : Int),
    Prime p ∧ p ∉ S ∧ x^2 + x + 1 = p * y := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_quadratic_equation_l3155_315512


namespace NUMINAMATH_CALUDE_xy_sum_is_two_l3155_315509

theorem xy_sum_is_two (x y : ℝ) 
  (hx : (x - 1)^3 + 1997*(x - 1) = -1) 
  (hy : (y - 1)^3 + 1997*(y - 1) = 1) : 
  x + y = 2 := by sorry

end NUMINAMATH_CALUDE_xy_sum_is_two_l3155_315509


namespace NUMINAMATH_CALUDE_pasture_consumption_l3155_315584

/-- Represents the pasture scenario with cows and grass -/
structure Pasture where
  /-- Amount of grass each cow eats per day -/
  daily_consumption : ℝ
  /-- Daily growth rate of the grass -/
  daily_growth : ℝ
  /-- Original amount of grass in the pasture -/
  initial_grass : ℝ

/-- Given the conditions, proves that 94 cows will consume all grass in 28 days -/
theorem pasture_consumption (p : Pasture) : 
  (p.initial_grass + 25 * p.daily_growth = 100 * 25 * p.daily_consumption) →
  (p.initial_grass + 35 * p.daily_growth = 84 * 35 * p.daily_consumption) →
  (p.initial_grass + 28 * p.daily_growth = 94 * 28 * p.daily_consumption) :=
by sorry

end NUMINAMATH_CALUDE_pasture_consumption_l3155_315584


namespace NUMINAMATH_CALUDE_rabbits_per_cat_l3155_315596

theorem rabbits_per_cat (total_animals : ℕ) (num_cats : ℕ) (hares_per_rabbit : ℕ) :
  total_animals = 37 →
  num_cats = 4 →
  hares_per_rabbit = 3 →
  ∃ (rabbits_per_cat : ℕ),
    total_animals = 1 + num_cats + (num_cats * rabbits_per_cat) + (num_cats * rabbits_per_cat * hares_per_rabbit) ∧
    rabbits_per_cat = 2 := by
  sorry

end NUMINAMATH_CALUDE_rabbits_per_cat_l3155_315596


namespace NUMINAMATH_CALUDE_amelia_weekly_goal_l3155_315546

/-- Amelia's weekly Jet Bar sales goal -/
def weekly_goal (monday_sales tuesday_sales remaining : ℕ) : ℕ :=
  monday_sales + tuesday_sales + remaining

/-- Theorem: Amelia's weekly Jet Bar sales goal is 90 -/
theorem amelia_weekly_goal :
  ∀ (monday_sales tuesday_sales remaining : ℕ),
  monday_sales = 45 →
  tuesday_sales = monday_sales - 16 →
  remaining = 16 →
  weekly_goal monday_sales tuesday_sales remaining = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_amelia_weekly_goal_l3155_315546


namespace NUMINAMATH_CALUDE_sixteen_cows_days_to_finish_l3155_315556

/-- Represents the grass consumption scenario in a pasture -/
structure GrassConsumption where
  /-- Daily grass growth rate -/
  daily_growth : ℝ
  /-- Amount of grass each cow eats per day -/
  cow_consumption : ℝ
  /-- Original amount of grass in the pasture -/
  initial_grass : ℝ

/-- Theorem stating that 16 cows will take 18 days to finish the grass -/
theorem sixteen_cows_days_to_finish (gc : GrassConsumption) : 
  gc.initial_grass + 6 * gc.daily_growth = 24 * 6 * gc.cow_consumption →
  gc.initial_grass + 8 * gc.daily_growth = 21 * 8 * gc.cow_consumption →
  gc.initial_grass + 18 * gc.daily_growth = 16 * 18 * gc.cow_consumption := by
  sorry

#check sixteen_cows_days_to_finish

end NUMINAMATH_CALUDE_sixteen_cows_days_to_finish_l3155_315556


namespace NUMINAMATH_CALUDE_multiply_and_add_equality_l3155_315523

theorem multiply_and_add_equality : 45 * 56 + 54 * 45 = 4950 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_equality_l3155_315523


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3155_315581

theorem diophantine_equation_solution (x y z : ℤ) :
  5 * x^3 + 11 * y^3 + 13 * z^3 = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3155_315581


namespace NUMINAMATH_CALUDE_sin_1050_degrees_l3155_315559

theorem sin_1050_degrees : Real.sin (1050 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_1050_degrees_l3155_315559


namespace NUMINAMATH_CALUDE_slower_painter_start_time_painting_scenario_conditions_l3155_315532

/-- Proves that the slower painter starts at 6.6 hours past noon given the painting scenario conditions -/
theorem slower_painter_start_time :
  ∀ (start_time : ℝ),
    (start_time + 6 = start_time + 7) →  -- Both painters finish at the same time
    (start_time + 7 = 12.6) →            -- They finish at 0.6 past midnight
    start_time = 6.6 := by
  sorry

/-- Defines the time the slower painter starts in hours past noon -/
def slower_painter_start : ℝ := 6.6

/-- Defines the time the faster painter starts in hours past noon -/
def faster_painter_start : ℝ := slower_painter_start + 3

/-- Defines the time both painters finish in hours past noon -/
def finish_time : ℝ := 12.6

/-- Proves that the painting scenario conditions are satisfied -/
theorem painting_scenario_conditions :
  slower_painter_start + 6 = finish_time ∧
  faster_painter_start + 4 = finish_time ∧
  faster_painter_start = slower_painter_start + 3 := by
  sorry

end NUMINAMATH_CALUDE_slower_painter_start_time_painting_scenario_conditions_l3155_315532


namespace NUMINAMATH_CALUDE_triple_overlap_area_is_six_l3155_315525

/-- Represents a rectangular carpet with width and height in meters -/
structure Carpet where
  width : ℝ
  height : ℝ

/-- Represents the layout of carpets in a hall -/
structure CarpetLayout where
  hall_width : ℝ
  hall_height : ℝ
  carpet1 : Carpet
  carpet2 : Carpet
  carpet3 : Carpet

/-- Calculates the area of triple overlap given a carpet layout -/
def tripleOverlapArea (layout : CarpetLayout) : ℝ :=
  sorry

/-- Theorem stating that the area of triple overlap is 6 square meters for the given layout -/
theorem triple_overlap_area_is_six (layout : CarpetLayout) 
  (h1 : layout.hall_width = 10 ∧ layout.hall_height = 10)
  (h2 : layout.carpet1.width = 6 ∧ layout.carpet1.height = 8)
  (h3 : layout.carpet2.width = 6 ∧ layout.carpet2.height = 6)
  (h4 : layout.carpet3.width = 5 ∧ layout.carpet3.height = 7)
  (h5 : ∀ c1 c2 : Carpet, c1 ≠ c2 → ¬ (c1.width + c2.width > layout.hall_width ∨ c1.height + c2.height > layout.hall_height)) :
  tripleOverlapArea layout = 6 := by
  sorry

end NUMINAMATH_CALUDE_triple_overlap_area_is_six_l3155_315525


namespace NUMINAMATH_CALUDE_record_cost_thomas_record_cost_l3155_315592

theorem record_cost (num_books : ℕ) (book_price : ℚ) (num_records : ℕ) (leftover : ℚ) : ℚ :=
  let total_sale := num_books * book_price
  let spent_on_records := total_sale - leftover
  spent_on_records / num_records

theorem thomas_record_cost :
  record_cost 200 1.5 75 75 = 3 := by
  sorry

end NUMINAMATH_CALUDE_record_cost_thomas_record_cost_l3155_315592


namespace NUMINAMATH_CALUDE_complex_power_modulus_l3155_315549

theorem complex_power_modulus : Complex.abs ((2 - 3 * Complex.I) ^ 5) = 13 ^ (5/2) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l3155_315549


namespace NUMINAMATH_CALUDE_chip_rearrangement_l3155_315553

/-- Represents a color of a chip -/
inductive Color
  | Red
  | Green
  | Blue

/-- Represents a position in the rectangle -/
structure Position where
  row : Fin 3
  col : Nat

/-- Represents the state of the rectangle -/
def Rectangle (n : Nat) := Position → Color

/-- Checks if a given rectangle arrangement is valid -/
def isValidArrangement (n : Nat) (rect : Rectangle n) : Prop :=
  ∀ c : Color, ∀ i : Fin 3, ∃ j : Fin n, rect ⟨i, j⟩ = c

/-- Checks if a given rectangle arrangement satisfies the condition -/
def satisfiesCondition (n : Nat) (rect : Rectangle n) : Prop :=
  ∀ j : Fin n, ∀ c : Color, ∃ i : Fin 3, rect ⟨i, j⟩ = c

/-- The main theorem to be proved -/
theorem chip_rearrangement (n : Nat) :
  ∃ (rect : Rectangle n), isValidArrangement n rect ∧ satisfiesCondition n rect := by
  sorry


end NUMINAMATH_CALUDE_chip_rearrangement_l3155_315553


namespace NUMINAMATH_CALUDE_counterfeit_coin_location_l3155_315551

/-- Represents a coin that can be either genuine or counterfeit -/
inductive Coin
  | genuine
  | counterfeit

/-- Represents the result of a weighing operation -/
inductive WeighingResult
  | equal
  | notEqual

/-- Function to perform a weighing operation on two pairs of coins -/
def weighPairs (c1 c2 c3 c4 : Coin) : WeighingResult :=
  sorry

/-- Theorem stating that we can narrow down the location of the counterfeit coin -/
theorem counterfeit_coin_location
  (coins : Fin 6 → Coin)
  (h_one_counterfeit : ∃! i, coins i = Coin.counterfeit) :
  (weighPairs (coins 0) (coins 1) (coins 2) (coins 3) = WeighingResult.equal
    → (coins 4 = Coin.counterfeit ∨ coins 5 = Coin.counterfeit))
  ∧
  (weighPairs (coins 0) (coins 1) (coins 2) (coins 3) = WeighingResult.notEqual
    → (coins 0 = Coin.counterfeit ∨ coins 1 = Coin.counterfeit ∨
       coins 2 = Coin.counterfeit ∨ coins 3 = Coin.counterfeit)) :=
  sorry

end NUMINAMATH_CALUDE_counterfeit_coin_location_l3155_315551


namespace NUMINAMATH_CALUDE_cylinder_equation_l3155_315537

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying r = c in cylindrical coordinates -/
def CylindricalSurface (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = c}

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) : Prop :=
  ∃ c : ℝ, c > 0 ∧ S = CylindricalSurface c

theorem cylinder_equation (c : ℝ) (h : c > 0) :
  IsCylinder (CylindricalSurface c) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_equation_l3155_315537


namespace NUMINAMATH_CALUDE_loans_equal_at_start_l3155_315513

/-- Represents the loan details for a person -/
structure Loan where
  principal : ℝ
  dailyInterestRate : ℝ

/-- Calculates the balance of a loan after t days -/
def loanBalance (loan : Loan) (t : ℝ) : ℝ :=
  loan.principal * (1 + loan.dailyInterestRate * t)

theorem loans_equal_at_start (claudia bob diana : Loan)
  (h_claudia : claudia = { principal := 200, dailyInterestRate := 0.04 })
  (h_bob : bob = { principal := 300, dailyInterestRate := 0.03 })
  (h_diana : diana = { principal := 500, dailyInterestRate := 0.02 }) :
  ∃ t : ℝ, t = 0 ∧ loanBalance claudia t + loanBalance bob t = loanBalance diana t :=
sorry

end NUMINAMATH_CALUDE_loans_equal_at_start_l3155_315513


namespace NUMINAMATH_CALUDE_water_usage_difference_l3155_315563

/-- Proves the difference in daily water usage before and after installing a water recycling device -/
theorem water_usage_difference (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (b / a) - (b / (a + 4)) = (4 * b) / (a * (a + 4)) := by
  sorry

end NUMINAMATH_CALUDE_water_usage_difference_l3155_315563


namespace NUMINAMATH_CALUDE_problem_solution_l3155_315538

theorem problem_solution (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a + b = 100) (h4 : (3/10) * a = (1/5) * b) : b = 60 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3155_315538


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l3155_315530

/-- Calculates the loss percentage given the cost price and selling price. -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Theorem stating that for a radio with cost price 1500 and selling price 1275,
    the loss percentage is 15%. -/
theorem radio_loss_percentage :
  loss_percentage 1500 1275 = 15 := by sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l3155_315530


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3155_315541

/-- Given two vectors a and b in ℝ², prove that if they are parallel and 
    a = (2, -1) and b = (k, 5/2), then k = -5. -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) :
  a = (2, -1) →
  b = (k, 5/2) →
  (∃ (t : ℝ), t ≠ 0 ∧ a = t • b) →
  k = -5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3155_315541


namespace NUMINAMATH_CALUDE_cos_thirty_degrees_l3155_315590

theorem cos_thirty_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_thirty_degrees_l3155_315590


namespace NUMINAMATH_CALUDE_root_inequality_l3155_315515

noncomputable section

open Real

-- Define the functions f and g
def f (x : ℝ) := exp x + x - 2
def g (x : ℝ) := log x + x - 2

-- State the theorem
theorem root_inequality (a b : ℝ) (ha : f a = 0) (hb : g b = 0) :
  f a < f 1 ∧ f 1 < f b := by
  sorry

end

end NUMINAMATH_CALUDE_root_inequality_l3155_315515


namespace NUMINAMATH_CALUDE_last_two_digits_product_l3155_315514

theorem last_two_digits_product (A B : ℕ) : 
  (A * 10 + B) % 6 = 0 → A + B = 11 → A * B = 24 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l3155_315514


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l3155_315576

theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  0.00000043 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 4.3 ∧ n = -7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l3155_315576


namespace NUMINAMATH_CALUDE_decimal_period_11_13_l3155_315567

/-- The length of the smallest repeating block in the decimal expansion of a rational number -/
def decimal_period (n d : ℕ) : ℕ :=
  sorry

/-- Theorem: The length of the smallest repeating block in the decimal expansion of 11/13 is 6 -/
theorem decimal_period_11_13 : decimal_period 11 13 = 6 := by
  sorry

end NUMINAMATH_CALUDE_decimal_period_11_13_l3155_315567


namespace NUMINAMATH_CALUDE_similar_triangles_height_l3155_315587

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small = 3 →
  area_ratio = 4 →
  ∃ h_large : ℝ, h_large = 6 ∧ h_large / h_small = Real.sqrt area_ratio :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l3155_315587


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3155_315504

-- Define the equations
def equation1 (x : ℝ) : Prop := 7*x - 20 = 2*(3 - 3*x)
def equation2 (x : ℝ) : Prop := (2*x - 3)/5 = (3*x - 1)/2 + 1

-- Theorem for equation 1
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 2 := by sorry

-- Theorem for equation 2
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = -1 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3155_315504


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3155_315586

theorem complex_number_quadrant : 
  let z : ℂ := (Complex.I) / (2 * Complex.I - 1)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3155_315586


namespace NUMINAMATH_CALUDE_emma_harry_weight_l3155_315550

/-- Given the weights of pairs of students, prove the combined weight of Emma and Harry -/
theorem emma_harry_weight
  (emma_fiona : ℝ)
  (fiona_george : ℝ)
  (george_harry : ℝ)
  (h_emma_fiona : emma_fiona = 280)
  (h_fiona_george : fiona_george = 260)
  (h_george_harry : george_harry = 290) :
  ∃ (emma harry : ℝ),
    emma + harry = 310 ∧
    ∃ (fiona george : ℝ),
      emma + fiona = emma_fiona ∧
      fiona + george = fiona_george ∧
      george + harry = george_harry :=
sorry

end NUMINAMATH_CALUDE_emma_harry_weight_l3155_315550


namespace NUMINAMATH_CALUDE_snowflake_puzzle_solution_l3155_315555

-- Define the grid as a 3x3 matrix
def Grid := Matrix (Fin 3) (Fin 3) Nat

-- Define the valid numbers
def ValidNumbers : List Nat := [1, 2, 3, 4, 5, 6]

-- Define the function to check if a number is valid in a given position
def isValidPlacement (grid : Grid) (row col : Fin 3) (num : Nat) : Prop :=
  -- Check row
  (∀ j : Fin 3, j ≠ col → grid row j ≠ num) ∧
  -- Check column
  (∀ i : Fin 3, i ≠ row → grid i col ≠ num) ∧
  -- Check diagonal (if applicable)
  (row = col → ∀ i : Fin 3, i ≠ row → grid i i ≠ num) ∧
  (row + col = 2 → ∀ i : Fin 3, grid i (2 - i) ≠ num)

-- Define the partially filled grid (Figure 2)
def initialGrid : Grid := λ i j =>
  if i = 0 ∧ j = 0 then 3
  else if i = 2 ∧ j = 2 then 4
  else 0  -- 0 represents an empty cell

-- Define the positions of A, B, C, D
def posA : Fin 3 × Fin 3 := (0, 1)
def posB : Fin 3 × Fin 3 := (1, 0)
def posC : Fin 3 × Fin 3 := (1, 1)
def posD : Fin 3 × Fin 3 := (1, 2)

-- Theorem statement
theorem snowflake_puzzle_solution :
  ∀ (grid : Grid),
    (∀ i j, grid i j ∈ ValidNumbers ∪ {0}) →
    (∀ i j, initialGrid i j ≠ 0 → grid i j = initialGrid i j) →
    (∀ i j, grid i j ≠ 0 → isValidPlacement grid i j (grid i j)) →
    (grid posA.1 posA.2 = 2 ∧
     grid posB.1 posB.2 = 5 ∧
     grid posC.1 posC.2 = 1 ∧
     grid posD.1 posD.2 = 6) :=
  sorry

end NUMINAMATH_CALUDE_snowflake_puzzle_solution_l3155_315555


namespace NUMINAMATH_CALUDE_rals_current_age_l3155_315528

-- Define Suri's and Ral's ages as natural numbers
def suris_age : ℕ := sorry
def rals_age : ℕ := sorry

-- State the theorem
theorem rals_current_age : 
  (rals_age = 3 * suris_age) →   -- Ral is three times as old as Suri
  (suris_age + 3 = 16) →         -- In 3 years, Suri's current age will be 16
  rals_age = 39 :=               -- Ral's current age is 39
by sorry

end NUMINAMATH_CALUDE_rals_current_age_l3155_315528


namespace NUMINAMATH_CALUDE_girls_count_in_school_l3155_315574

theorem girls_count_in_school (total_students : ℕ) (boys_avg_age girls_avg_age school_avg_age : ℚ) : 
  total_students = 652 →
  boys_avg_age = 12 →
  girls_avg_age = 11 →
  school_avg_age = 11.75 →
  ∃ (girls_count : ℕ), 
    girls_count = 162 ∧ 
    (total_students - girls_count) * boys_avg_age + girls_count * girls_avg_age = total_students * school_avg_age :=
by sorry

end NUMINAMATH_CALUDE_girls_count_in_school_l3155_315574


namespace NUMINAMATH_CALUDE_megans_files_l3155_315577

theorem megans_files (deleted_files : ℕ) (files_per_folder : ℕ) (num_folders : ℕ) :
  deleted_files = 21 →
  files_per_folder = 8 →
  num_folders = 9 →
  deleted_files + (files_per_folder * num_folders) = 93 :=
by sorry

end NUMINAMATH_CALUDE_megans_files_l3155_315577


namespace NUMINAMATH_CALUDE_hyperbola_distance_l3155_315557

/-- Given a hyperbola with equation x²/25 - y²/9 = 1, prove that |ON| = 4 --/
theorem hyperbola_distance (M F₁ F₂ N O : ℝ × ℝ) : 
  (∀ x y, (x^2 / 25) - (y^2 / 9) = 1 → (x, y) = M) →  -- M is on the hyperbola
  (M.1 < 0) →  -- M is on the left branch
  ‖M - F₂‖ = 18 →  -- Distance from M to F₂ is 18
  N = (M + F₂) / 2 →  -- N is the midpoint of MF₂
  O = (0, 0) →  -- O is the origin
  ‖O - N‖ = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_distance_l3155_315557


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l3155_315564

/-- The area of the region between two concentric circles, where the radius of the larger circle
    is three times the radius of the smaller circle, and the diameter of the smaller circle is 6 units. -/
theorem shaded_area_between_circles (π : ℝ) : ℝ := by
  -- Define the diameter of the smaller circle
  let small_diameter : ℝ := 6
  -- Define the radius of the smaller circle
  let small_radius : ℝ := small_diameter / 2
  -- Define the radius of the larger circle
  let large_radius : ℝ := 3 * small_radius
  -- Define the area of the shaded region
  let shaded_area : ℝ := π * large_radius^2 - π * small_radius^2
  -- Prove that the shaded area equals 72π
  have : shaded_area = 72 * π := by sorry
  -- Return the result
  exact 72 * π

end NUMINAMATH_CALUDE_shaded_area_between_circles_l3155_315564


namespace NUMINAMATH_CALUDE_total_customers_l3155_315569

def customers_in_line (people_in_front : ℕ) : ℕ := people_in_front + 1

theorem total_customers (people_in_front : ℕ) : 
  people_in_front = 8 → customers_in_line people_in_front = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_customers_l3155_315569


namespace NUMINAMATH_CALUDE_max_bouquets_is_37_l3155_315591

/-- Represents the number of flowers available for each type -/
structure FlowerInventory where
  narcissus : ℕ
  chrysanthemum : ℕ
  tulip : ℕ
  lily : ℕ
  rose : ℕ

/-- Represents the constraints for creating a bouquet -/
structure BouquetConstraints where
  min_narcissus : ℕ
  min_chrysanthemum : ℕ
  min_tulip : ℕ
  max_lily_or_rose : ℕ
  max_total : ℕ

/-- Calculates the maximum number of bouquets that can be made -/
def maxBouquets (inventory : FlowerInventory) (constraints : BouquetConstraints) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of bouquets is 37 -/
theorem max_bouquets_is_37 :
  let inventory := FlowerInventory.mk 75 90 50 45 60
  let constraints := BouquetConstraints.mk 2 1 1 3 10
  maxBouquets inventory constraints = 37 := by sorry

end NUMINAMATH_CALUDE_max_bouquets_is_37_l3155_315591
