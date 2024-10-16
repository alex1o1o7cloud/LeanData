import Mathlib

namespace NUMINAMATH_CALUDE_dinner_bill_tip_percentage_l3114_311471

theorem dinner_bill_tip_percentage 
  (total_bill : ℝ)
  (num_friends : ℕ)
  (silas_payment : ℝ)
  (one_friend_payment : ℝ)
  (h1 : total_bill = 150)
  (h2 : num_friends = 6)
  (h3 : silas_payment = total_bill / 2)
  (h4 : one_friend_payment = 18)
  : (((one_friend_payment - (total_bill - silas_payment) / (num_friends - 1)) * (num_friends - 1)) / total_bill) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_dinner_bill_tip_percentage_l3114_311471


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3114_311414

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 + 5*x - 14 > 0} = {x : ℝ | x < -7 ∨ x > 2} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3114_311414


namespace NUMINAMATH_CALUDE_sqrt_of_four_l3114_311497

-- Define the square root function
def sqrt (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

-- Theorem statement
theorem sqrt_of_four : sqrt 4 = {2, -2} := by sorry

end NUMINAMATH_CALUDE_sqrt_of_four_l3114_311497


namespace NUMINAMATH_CALUDE_odometer_sum_of_squares_l3114_311449

/-- Represents the odometer reading as a three-digit number -/
structure OdometerReading where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds + tens + ones ≤ 9

/-- Represents the car's journey -/
structure CarJourney where
  duration : Nat
  avg_speed : Nat
  start_reading : OdometerReading
  end_reading : OdometerReading
  journey_valid : 
    duration = 8 ∧ 
    avg_speed = 65 ∧
    end_reading.hundreds = start_reading.ones ∧
    end_reading.tens = start_reading.tens ∧
    end_reading.ones = start_reading.hundreds

theorem odometer_sum_of_squares (journey : CarJourney) : 
  journey.start_reading.hundreds^2 + 
  journey.start_reading.tens^2 + 
  journey.start_reading.ones^2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_odometer_sum_of_squares_l3114_311449


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3114_311460

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : b = 7)
  (h4 : (a + b + c) / 3 = a + 15)
  (h5 : (a + b + c) / 3 = c - 10) :
  a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3114_311460


namespace NUMINAMATH_CALUDE_min_value_product_l3114_311407

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a/b + b/c + c/a + b/a + c/b + a/c = 10) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 38 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l3114_311407


namespace NUMINAMATH_CALUDE_carolyn_sticker_count_l3114_311437

/-- Given that Belle collected 97 stickers and Carolyn collected 18 fewer stickers than Belle,
    prove that Carolyn collected 79 stickers. -/
theorem carolyn_sticker_count :
  ∀ (belle_stickers carolyn_stickers : ℕ),
    belle_stickers = 97 →
    carolyn_stickers = belle_stickers - 18 →
    carolyn_stickers = 79 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_sticker_count_l3114_311437


namespace NUMINAMATH_CALUDE_exists_valid_surname_l3114_311484

/-- Represents the positions of letters in a 6-letter Russian surname --/
structure SurnameLetter where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ
  sixth : ℕ

/-- Conditions for the Russian writer's surname --/
def is_valid_surname (s : SurnameLetter) : Prop :=
  s.first = s.third ∧
  s.second = s.fourth ∧
  s.fifth = s.first + 9 ∧
  s.sixth = s.second + s.fourth - 2 ∧
  3 * s.first = s.second - 4 ∧
  s.first + s.second + s.third + s.fourth + s.fifth + s.sixth = 83

/-- The theorem stating the existence of a valid surname --/
theorem exists_valid_surname : ∃ (s : SurnameLetter), is_valid_surname s :=
sorry

end NUMINAMATH_CALUDE_exists_valid_surname_l3114_311484


namespace NUMINAMATH_CALUDE_gcd_n_cube_plus_25_and_n_plus_6_l3114_311467

theorem gcd_n_cube_plus_25_and_n_plus_6 (n : ℕ) (h : n > 2^5) :
  Nat.gcd (n^3 + 5^2) (n + 6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_cube_plus_25_and_n_plus_6_l3114_311467


namespace NUMINAMATH_CALUDE_w_coordinate_of_point_on_line_l3114_311411

/-- A 4D point -/
structure Point4D where
  x : ℝ
  y : ℝ
  z : ℝ
  w : ℝ

/-- Definition of the line passing through two points -/
def line_through (p q : Point4D) (t : ℝ) : Point4D :=
  { x := p.x + t * (q.x - p.x),
    y := p.y + t * (q.y - p.y),
    z := p.z + t * (q.z - p.z),
    w := p.w + t * (q.w - p.w) }

/-- The theorem to be proved -/
theorem w_coordinate_of_point_on_line : 
  let p1 : Point4D := {x := 3, y := 3, z := 2, w := 1}
  let p2 : Point4D := {x := 6, y := 2, z := 1, w := -1}
  ∃ t : ℝ, 
    let point := line_through p1 p2 t
    point.y = 4 ∧ point.w = 3 := by
  sorry

end NUMINAMATH_CALUDE_w_coordinate_of_point_on_line_l3114_311411


namespace NUMINAMATH_CALUDE_birch_trees_not_adjacent_probability_l3114_311485

def total_trees : ℕ := 14
def maple_trees : ℕ := 4
def oak_trees : ℕ := 5
def birch_trees : ℕ := 5

theorem birch_trees_not_adjacent_probability : 
  let total_arrangements := Nat.choose total_trees birch_trees
  let non_birch_trees := maple_trees + oak_trees
  let valid_arrangements := Nat.choose (non_birch_trees + 1) birch_trees
  (valid_arrangements : ℚ) / total_arrangements = 18 / 143 := by
  sorry

end NUMINAMATH_CALUDE_birch_trees_not_adjacent_probability_l3114_311485


namespace NUMINAMATH_CALUDE_retiree_benefit_theorem_l3114_311451

/-- Represents a bank customer --/
structure Customer where
  repayment_rate : ℝ
  monthly_income_stability : ℝ
  preferred_deposit_term : ℝ

/-- Represents a bank's financial metrics --/
structure BankMetrics where
  loan_default_risk : ℝ
  deposit_stability : ℝ
  long_term_liquidity : ℝ

/-- Calculates the benefit for a bank based on customer characteristics --/
def calculate_bank_benefit (c : Customer) : ℝ :=
  c.repayment_rate + c.monthly_income_stability + c.preferred_deposit_term

/-- Represents a retiree customer --/
def retiree : Customer where
  repayment_rate := 0.95
  monthly_income_stability := 0.9
  preferred_deposit_term := 5

/-- Represents an average customer --/
def average_customer : Customer where
  repayment_rate := 0.8
  monthly_income_stability := 0.7
  preferred_deposit_term := 2

/-- Theorem stating that offering special rates to retirees is beneficial for banks --/
theorem retiree_benefit_theorem :
  calculate_bank_benefit retiree > calculate_bank_benefit average_customer :=
by sorry

end NUMINAMATH_CALUDE_retiree_benefit_theorem_l3114_311451


namespace NUMINAMATH_CALUDE_chip_thickness_comparison_l3114_311479

theorem chip_thickness_comparison : 
  let a : ℝ := (1/3) * Real.sin (1/2)
  let b : ℝ := (1/2) * Real.sin (1/3)
  let c : ℝ := (1/3) * Real.cos (7/8)
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_chip_thickness_comparison_l3114_311479


namespace NUMINAMATH_CALUDE_fraction_subtraction_equality_l3114_311434

theorem fraction_subtraction_equality : 
  -1/8 - (1 + 1/3) - (-5/8) - (4 + 2/3) = -(11/2) := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_equality_l3114_311434


namespace NUMINAMATH_CALUDE_divisibility_proof_l3114_311452

theorem divisibility_proof (n : ℕ) : 
  (∃ k : ℤ, 32^(3*n) - 1312^n = 1966 * k) ∧ 
  (∃ m : ℤ, 843^(2*n+1) - 1099^(2*n+1) + 16^(4*n+2) = 1967 * m) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_proof_l3114_311452


namespace NUMINAMATH_CALUDE_A_when_one_is_element_B_is_zero_and_neg_one_third_l3114_311462

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x - 3 = 0}

-- Theorem 1: If 1 ∈ A, then A = {1, -3}
theorem A_when_one_is_element (a : ℝ) : 1 ∈ A a → A a = {1, -3} := by sorry

-- Define the set B
def B : Set ℝ := {a : ℝ | ∃! x, x ∈ A a}

-- Theorem 2: B = {0, -1/3}
theorem B_is_zero_and_neg_one_third : B = {0, -1/3} := by sorry

end NUMINAMATH_CALUDE_A_when_one_is_element_B_is_zero_and_neg_one_third_l3114_311462


namespace NUMINAMATH_CALUDE_farmers_income_2010_l3114_311478

/-- Farmers' income in a given year -/
structure FarmerIncome where
  wage : ℝ
  other : ℝ

/-- Calculate farmers' income after n years -/
def futureIncome (initial : FarmerIncome) (n : ℕ) : ℝ :=
  initial.wage * (1 + 0.06) ^ n + (initial.other + n * 320)

theorem farmers_income_2010 :
  let initial : FarmerIncome := { wage := 3600, other := 2700 }
  let income2010 := futureIncome initial 5
  8800 ≤ income2010 ∧ income2010 < 9200 := by
  sorry

end NUMINAMATH_CALUDE_farmers_income_2010_l3114_311478


namespace NUMINAMATH_CALUDE_prob_at_least_two_different_fruits_l3114_311423

def num_fruits : ℕ := 4
def num_meals : ℕ := 3

def prob_same_fruit_all_day : ℚ := (1 / num_fruits) ^ num_meals * num_fruits

theorem prob_at_least_two_different_fruits :
  1 - prob_same_fruit_all_day = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_different_fruits_l3114_311423


namespace NUMINAMATH_CALUDE_ribbons_shipment_count_l3114_311474

/-- The number of ribbons that arrived in the shipment before lunch -/
def ribbons_in_shipment (initial : ℕ) (morning : ℕ) (afternoon : ℕ) (final : ℕ) : ℕ :=
  (afternoon + final) - (initial - morning)

/-- Theorem stating that the number of ribbons in the shipment is 4 -/
theorem ribbons_shipment_count :
  ribbons_in_shipment 38 14 16 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ribbons_shipment_count_l3114_311474


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l3114_311464

theorem ratio_a_to_c (a b c : ℚ) 
  (h1 : a / b = 8 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l3114_311464


namespace NUMINAMATH_CALUDE_function_increasing_implies_a_leq_one_l3114_311416

/-- Given a function f(x) = e^(|x-a|), where a is a constant,
    if f(x) is increasing on [1, +∞), then a ≤ 1 -/
theorem function_increasing_implies_a_leq_one (a : ℝ) :
  (∀ x y, 1 ≤ x ∧ x < y → (Real.exp (|x - a|) < Real.exp (|y - a|))) →
  a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_increasing_implies_a_leq_one_l3114_311416


namespace NUMINAMATH_CALUDE_largest_number_with_conditions_l3114_311421

theorem largest_number_with_conditions : ∃ n : ℕ, n = 93 ∧
  n < 100 ∧
  n % 8 = 5 ∧
  n % 3 = 0 ∧
  ∀ m : ℕ, m < 100 → m % 8 = 5 → m % 3 = 0 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_conditions_l3114_311421


namespace NUMINAMATH_CALUDE_tank_capacity_l3114_311408

theorem tank_capacity (T : ℚ) : 
  (3/4 : ℚ) * T + 4 = (7/8 : ℚ) * T → T = 32 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l3114_311408


namespace NUMINAMATH_CALUDE_quadratic_completion_l3114_311444

theorem quadratic_completion (x : ℝ) : ∃ (a b : ℝ), x^2 - 6*x + 5 = 0 ↔ (x + a)^2 = b ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l3114_311444


namespace NUMINAMATH_CALUDE_brush_chess_pricing_l3114_311498

theorem brush_chess_pricing (x y : ℚ) : 
  (5 * x + 12 * y = 315) ∧ (8 * x + 6 * y = 240) → x = 15 ∧ y = 20 :=
by sorry

end NUMINAMATH_CALUDE_brush_chess_pricing_l3114_311498


namespace NUMINAMATH_CALUDE_decrement_calculation_l3114_311466

theorem decrement_calculation (n : ℕ) (original_mean updated_mean : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 200)
  (h3 : updated_mean = 194) :
  (n : ℝ) * original_mean - n * updated_mean = 6 * n := by
  sorry

end NUMINAMATH_CALUDE_decrement_calculation_l3114_311466


namespace NUMINAMATH_CALUDE_hyperbola_focus_l3114_311454

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 3 - x^2 / 6 = 1

/-- Definition of a focus of the hyperbola -/
def is_focus (x y : ℝ) : Prop :=
  ∃ (c : ℝ), c^2 = 3 + 6 ∧ (x = 0 ∧ (y = c ∨ y = -c))

/-- Theorem: One focus of the hyperbola has coordinates (0, 3) -/
theorem hyperbola_focus : ∃ (x y : ℝ), hyperbola_equation x y ∧ is_focus x y ∧ x = 0 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l3114_311454


namespace NUMINAMATH_CALUDE_domain_of_g_l3114_311459

-- Define the function f with domain [-3, 1]
def f : Set ℝ → Set ℝ := fun D ↦ {x | x ∈ D ∧ -3 ≤ x ∧ x ≤ 1}

-- Define the function g in terms of f
def g (f : Set ℝ → Set ℝ) : Set ℝ → Set ℝ := fun D ↦ {x | (x + 1) ∈ f D}

-- Theorem statement
theorem domain_of_g (D : Set ℝ) :
  g f D = {x : ℝ | -4 ≤ x ∧ x ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l3114_311459


namespace NUMINAMATH_CALUDE_eighteen_gon_symmetry_sum_l3114_311425

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle (in degrees) for rotational symmetry of a regular polygon -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ :=
  360 / n

theorem eighteen_gon_symmetry_sum :
  ∀ (p : RegularPolygon 18),
    (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 38 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_gon_symmetry_sum_l3114_311425


namespace NUMINAMATH_CALUDE_travel_theorem_l3114_311430

/-- Represents the scenario of two people traveling with a bicycle --/
structure TravelScenario where
  distance : ℝ
  walkSpeed : ℝ
  bikeSpeed : ℝ
  cLeaveTime : ℝ

/-- Checks if the travel scenario results in simultaneous arrival --/
def simultaneousArrival (scenario : TravelScenario) : Prop :=
  let t := scenario.distance / scenario.walkSpeed
  let meetTime := (scenario.distance * scenario.walkSpeed) / (scenario.bikeSpeed + scenario.walkSpeed)
  let cTravelTime := scenario.distance / scenario.walkSpeed - scenario.cLeaveTime
  t = meetTime + (scenario.distance - meetTime * scenario.bikeSpeed) / scenario.walkSpeed

/-- The main theorem to prove --/
theorem travel_theorem (scenario : TravelScenario) 
  (h1 : scenario.distance = 15)
  (h2 : scenario.walkSpeed = 6)
  (h3 : scenario.bikeSpeed = 15)
  (h4 : scenario.cLeaveTime = 3/11) :
  simultaneousArrival scenario := by
  sorry


end NUMINAMATH_CALUDE_travel_theorem_l3114_311430


namespace NUMINAMATH_CALUDE_sequences_properties_l3114_311412

/-- Definition of the first sequence -/
def seq1 (n : ℕ) : ℤ := (-2)^n

/-- Definition of the second sequence -/
def seq2 (m : ℕ) : ℤ := (-2)^(m-1)

/-- Definition of the third sequence -/
def seq3 (m : ℕ) : ℤ := (-2)^(m-1) - 1

/-- Theorem stating the properties of the sequences -/
theorem sequences_properties :
  (∀ n : ℕ, seq1 n = (-2)^n) ∧
  (∀ m : ℕ, seq3 m = seq2 m - 1) ∧
  (seq1 2019 + seq2 2019 + seq3 2019 = -1) :=
by sorry

end NUMINAMATH_CALUDE_sequences_properties_l3114_311412


namespace NUMINAMATH_CALUDE_binomial_10_3_l3114_311465

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem binomial_10_3 : binomial 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l3114_311465


namespace NUMINAMATH_CALUDE_farmer_chicken_sales_l3114_311438

def duck_price : ℕ := 10
def chicken_price : ℕ := 8
def ducks_sold : ℕ := 2

theorem farmer_chicken_sales : 
  ∃ (chickens_sold : ℕ),
    (duck_price * ducks_sold + chicken_price * chickens_sold) / 2 = 30 ∧
    chickens_sold = 5 := by
  sorry

end NUMINAMATH_CALUDE_farmer_chicken_sales_l3114_311438


namespace NUMINAMATH_CALUDE_sum_coordinates_of_D_l3114_311431

/-- Given a line segment CD where C(11, 4) and P(5, 10) is the midpoint,
    prove that the sum of the coordinates of point D is 15. -/
theorem sum_coordinates_of_D (C D P : ℝ × ℝ) : 
  C = (11, 4) →
  P = (5, 10) →
  P = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_D_l3114_311431


namespace NUMINAMATH_CALUDE_shells_added_correct_l3114_311495

/-- The amount of shells added to a bucket -/
def shells_added (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that the difference between the final and initial amounts
    equals the amount of shells added -/
theorem shells_added_correct (initial final : ℕ) (h : final ≥ initial) :
  shells_added initial final = final - initial :=
by
  sorry

end NUMINAMATH_CALUDE_shells_added_correct_l3114_311495


namespace NUMINAMATH_CALUDE_problem_solution_l3114_311447

theorem problem_solution (n : ℕ) (h1 : n > 30) (h2 : (4 * n - 1) ∣ (2002 * n)) : n = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3114_311447


namespace NUMINAMATH_CALUDE_complex_square_sum_l3114_311401

theorem complex_square_sum (a b : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (1 + 2 / Complex.I)^2 = a + b * Complex.I →
  a + b = -7 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_sum_l3114_311401


namespace NUMINAMATH_CALUDE_equation_solution_l3114_311468

theorem equation_solution : 
  ∃ x : ℚ, (x^2 + x + 1) / (x + 1) = x + 3 ∧ x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3114_311468


namespace NUMINAMATH_CALUDE_k_value_max_value_on_interval_l3114_311426

-- Define the function f(x) with parameter k
def f (k : ℝ) (x : ℝ) : ℝ := x^2 + (2*k - 3)*x + k^2 - 7

-- Theorem 1: k = 3 given the zeros of f(x)
theorem k_value (k : ℝ) : (f k (-1) = 0 ∧ f k (-2) = 0) → k = 3 := by sorry

-- Define the specific function f(x) = x^2 + 3x + 2
def f_specific (x : ℝ) : ℝ := x^2 + 3*x + 2

-- Theorem 2: Maximum value of f_specific on [-2, 2] is 12
theorem max_value_on_interval : 
  ∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f_specific x ≤ 12 ∧ ∃ y ∈ Set.Icc (-2) 2, f_specific y = 12 := by sorry

end NUMINAMATH_CALUDE_k_value_max_value_on_interval_l3114_311426


namespace NUMINAMATH_CALUDE_intersection_S_T_l3114_311486

def S : Set ℝ := {x | x + 1 ≥ 2}
def T : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_S_T : S ∩ T = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_l3114_311486


namespace NUMINAMATH_CALUDE_factor_expression_l3114_311403

/-- The expression a^3 (b^2 - c^2) + b^3 (c^2 - a^2) + c^3 (a^2 - b^2) 
    can be factored as (a - b)(b - c)(c - a) * (-(ab + ac + bc)) -/
theorem factor_expression (a b c : ℝ) :
  a^3 * (b^2 - c^2) + b^3 * (c^2 - a^2) + c^3 * (a^2 - b^2) =
  (a - b) * (b - c) * (c - a) * (-(a*b + a*c + b*c)) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3114_311403


namespace NUMINAMATH_CALUDE_fish_sample_count_l3114_311443

/-- Given a population of fish and a stratified sampling method, 
    calculate the number of black carp and common carp in the sample. -/
theorem fish_sample_count 
  (total_fish : ℕ) 
  (black_carp : ℕ) 
  (common_carp : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_fish = 200) 
  (h2 : black_carp = 20) 
  (h3 : common_carp = 40) 
  (h4 : sample_size = 20) : 
  (black_carp * sample_size / total_fish + common_carp * sample_size / total_fish : ℕ) = 6 := by
  sorry

#check fish_sample_count

end NUMINAMATH_CALUDE_fish_sample_count_l3114_311443


namespace NUMINAMATH_CALUDE_b_completes_in_20_days_l3114_311457

/-- The number of days it takes for person A to complete the work alone -/
def days_A : ℝ := 15

/-- The number of days A and B work together -/
def days_together : ℝ := 6

/-- The fraction of work left after A and B work together -/
def work_left : ℝ := 0.3

/-- The number of days it takes for person B to complete the work alone -/
def days_B : ℝ := 20

/-- Theorem stating that given the conditions, B can complete the work alone in 20 days -/
theorem b_completes_in_20_days :
  days_together * (1 / days_A + 1 / days_B) = 1 - work_left :=
sorry

end NUMINAMATH_CALUDE_b_completes_in_20_days_l3114_311457


namespace NUMINAMATH_CALUDE_exists_bijection_Z_to_H_l3114_311448

-- Define the set ℍ
def ℍ : Set ℚ :=
  { x | ∀ S : Set ℚ, 
    (1/2 ∈ S) → 
    (∀ y ∈ S, 1/(1+y) ∈ S ∧ y/(1+y) ∈ S) → 
    x ∈ S }

-- State the theorem
theorem exists_bijection_Z_to_H : ∃ f : ℤ → ℍ, Function.Bijective f := by
  sorry

end NUMINAMATH_CALUDE_exists_bijection_Z_to_H_l3114_311448


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3114_311428

theorem fraction_sum_equality (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 1 / (b - c) + 1 / (c - a) + 1 / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3114_311428


namespace NUMINAMATH_CALUDE_max_value_theorem_l3114_311439

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 5 * a + 2 * b < 100) :
  a * b * (100 - 5 * a - 2 * b) ≤ 78125 / 36 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 5 * a₀ + 2 * b₀ < 100 ∧
    a₀ * b₀ * (100 - 5 * a₀ - 2 * b₀) = 78125 / 36 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3114_311439


namespace NUMINAMATH_CALUDE_problem_statement_l3114_311405

theorem problem_statement : (-1)^53 + 2^(4^3 + 5^2 - 7^2) = 1099511627775 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3114_311405


namespace NUMINAMATH_CALUDE_cube_root_condition_square_root_condition_integer_part_condition_main_result_l3114_311482

def a : ℝ := 4
def b : ℝ := 2
def c : ℤ := 3

theorem cube_root_condition : (3 * a - 4) ^ (1/3 : ℝ) = 2 := by sorry

theorem square_root_condition : (a + 2 * b + 1) ^ (1/2 : ℝ) = 3 := by sorry

theorem integer_part_condition : c = Int.floor (Real.sqrt 15) := by sorry

theorem main_result : (a + b + c : ℝ) ^ (1/2 : ℝ) = 3 ∨ (a + b + c : ℝ) ^ (1/2 : ℝ) = -3 := by sorry

end NUMINAMATH_CALUDE_cube_root_condition_square_root_condition_integer_part_condition_main_result_l3114_311482


namespace NUMINAMATH_CALUDE_series_sum_210_l3114_311409

def series_sum (n : ℕ) : ℤ :=
  let groups := n / 3
  let last_term := 3 * (groups - 1)
  (groups : ℤ) * last_term / 2

theorem series_sum_210 :
  series_sum 210 = 7245 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_210_l3114_311409


namespace NUMINAMATH_CALUDE_female_officers_count_l3114_311492

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_duty_percentage : ℚ) :
  total_on_duty = 170 →
  female_on_duty_ratio = 1/2 →
  female_duty_percentage = 17/100 →
  ∃ (total_female : ℕ), total_female = 500 ∧ 
    (↑total_on_duty * female_on_duty_ratio : ℚ) = (↑total_female * female_duty_percentage : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_female_officers_count_l3114_311492


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l3114_311413

theorem complete_square_quadratic (a b c : ℝ) (h : a = 1 ∧ b = 6 ∧ c = 5) :
  ∃ (k : ℝ), (x + k)^2 - (x^2 + b*x + c) = 4 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l3114_311413


namespace NUMINAMATH_CALUDE_lenny_remaining_money_l3114_311470

-- Define the initial amount and expenses
def initial_amount : ℝ := 270
def console_price : ℝ := 149
def console_discount : ℝ := 0.15
def grocery_price : ℝ := 60
def grocery_discount : ℝ := 0.10
def lunch_price : ℝ := 30
def magazine_price : ℝ := 3.99

-- Define the function to calculate the remaining money
def remaining_money : ℝ :=
  initial_amount -
  (console_price * (1 - console_discount)) -
  (grocery_price * (1 - grocery_discount)) -
  lunch_price -
  magazine_price

-- Theorem to prove
theorem lenny_remaining_money :
  remaining_money = 55.36 := by sorry

end NUMINAMATH_CALUDE_lenny_remaining_money_l3114_311470


namespace NUMINAMATH_CALUDE_unique_a_value_l3114_311404

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + 4 = 0}

-- State the theorem
theorem unique_a_value :
  ∃! a : ℝ, (B a).Nonempty ∧ B a ⊆ A := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l3114_311404


namespace NUMINAMATH_CALUDE_robin_albums_l3114_311499

theorem robin_albums (total_pictures : ℕ) (pictures_per_album : ℕ) (h1 : total_pictures = 40) (h2 : pictures_per_album = 8) : total_pictures / pictures_per_album = 5 := by
  sorry

end NUMINAMATH_CALUDE_robin_albums_l3114_311499


namespace NUMINAMATH_CALUDE_largest_inscribed_square_side_length_largest_inscribed_square_side_length_proof_l3114_311427

/-- The side length of the largest square that can be inscribed in a square with side length 12,
    given two congruent equilateral triangles are inscribed as described in the problem. -/
theorem largest_inscribed_square_side_length : ℝ :=
  let outer_square_side : ℝ := 12
  let triangle_side : ℝ := 4 * Real.sqrt 6
  6 - Real.sqrt 6

/-- Proof that the calculated side length is correct -/
theorem largest_inscribed_square_side_length_proof :
  largest_inscribed_square_side_length = 6 - Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_square_side_length_largest_inscribed_square_side_length_proof_l3114_311427


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l3114_311450

theorem dogwood_tree_count (current_trees planted_trees : ℕ) : 
  current_trees = 34 → planted_trees = 49 → current_trees + planted_trees = 83 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l3114_311450


namespace NUMINAMATH_CALUDE_min_value_of_function_l3114_311433

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  let y := (x^2 + 3*x + 1) / x
  (∀ z, z > 0 → y ≤ (z^2 + 3*z + 1) / z) ∧ y = 5 := by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3114_311433


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_10_l3114_311480

theorem binomial_coefficient_20_10 
  (h1 : Nat.choose 18 8 = 43758)
  (h2 : Nat.choose 18 9 = 48620)
  (h3 : Nat.choose 18 10 = 43758) : 
  Nat.choose 20 10 = 184756 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_10_l3114_311480


namespace NUMINAMATH_CALUDE_negative_double_greater_than_negative_abs_l3114_311475

theorem negative_double_greater_than_negative_abs :
  -(-(1/9 : ℚ)) > -|(-(1/9 : ℚ))| := by sorry

end NUMINAMATH_CALUDE_negative_double_greater_than_negative_abs_l3114_311475


namespace NUMINAMATH_CALUDE_min_pizzas_to_earn_back_car_cost_l3114_311420

/-- The cost of the car John bought -/
def car_cost : ℕ := 6500

/-- The amount John receives for each pizza delivered -/
def income_per_pizza : ℕ := 12

/-- The amount John spends on gas for each pizza delivered -/
def gas_cost_per_pizza : ℕ := 4

/-- The amount John spends on maintenance for each pizza delivered -/
def maintenance_cost_per_pizza : ℕ := 1

/-- The minimum whole number of pizzas John must deliver to earn back the car cost -/
def min_pizzas : ℕ := 929

theorem min_pizzas_to_earn_back_car_cost :
  ∀ n : ℕ, n ≥ min_pizzas →
    n * (income_per_pizza - gas_cost_per_pizza - maintenance_cost_per_pizza) ≥ car_cost ∧
    ∀ m : ℕ, m < min_pizzas →
      m * (income_per_pizza - gas_cost_per_pizza - maintenance_cost_per_pizza) < car_cost :=
by sorry

end NUMINAMATH_CALUDE_min_pizzas_to_earn_back_car_cost_l3114_311420


namespace NUMINAMATH_CALUDE_num_quadrilaterals_is_495_l3114_311481

/-- The number of ways to choose 4 points from 12 distinct points on a circle's circumference to form convex quadrilaterals -/
def num_quadrilaterals : ℕ := Nat.choose 12 4

/-- Theorem stating that the number of quadrilaterals is 495 -/
theorem num_quadrilaterals_is_495 : num_quadrilaterals = 495 := by
  sorry

end NUMINAMATH_CALUDE_num_quadrilaterals_is_495_l3114_311481


namespace NUMINAMATH_CALUDE_f_monotonicity_f_shifted_even_f_positive_domain_l3114_311415

-- Define the function f(x) = lg|x-1|
noncomputable def f (x : ℝ) : ℝ := Real.log (abs (x - 1)) / Real.log 10

-- Statement 1: f(x) is monotonically decreasing on (-∞, 1) and increasing on (1, +∞)
theorem f_monotonicity :
  (∀ x y, x < y ∧ y < 1 → f x > f y) ∧
  (∀ x y, 1 < x ∧ x < y → f x < f y) := by sorry

-- Statement 2: f(x+1) is an even function
theorem f_shifted_even :
  ∀ x, f (x + 1) = f (-x + 1) := by sorry

-- Statement 3: If f(a) > 0, then a < 0 or a > 2
theorem f_positive_domain :
  ∀ a, f a > 0 → a < 0 ∨ a > 2 := by sorry

end NUMINAMATH_CALUDE_f_monotonicity_f_shifted_even_f_positive_domain_l3114_311415


namespace NUMINAMATH_CALUDE_turban_price_turban_price_is_70_l3114_311429

/-- The price of a turban given the following conditions:
  * The total salary for one year is Rs. 90 plus the turban
  * The servant works for 9 months (3/4 of a year)
  * The servant receives Rs. 50 and the turban after 9 months
-/
theorem turban_price : ℝ → Prop :=
  fun price =>
    let yearly_salary := 90 + price
    let worked_fraction := 3 / 4
    let received_salary := 50 + price
    worked_fraction * yearly_salary = received_salary

/-- The price of the turban is 70 rupees -/
theorem turban_price_is_70 : turban_price 70 := by
  sorry

end NUMINAMATH_CALUDE_turban_price_turban_price_is_70_l3114_311429


namespace NUMINAMATH_CALUDE_average_after_adding_constant_specific_average_problem_l3114_311477

theorem average_after_adding_constant (n : ℕ) (original_avg : ℚ) (added_const : ℚ) :
  n > 0 →
  let new_avg := original_avg + added_const
  new_avg = (n * original_avg + n * added_const) / n := by
  sorry

theorem specific_average_problem :
  let n : ℕ := 15
  let original_avg : ℚ := 40
  let added_const : ℚ := 10
  let new_avg := original_avg + added_const
  new_avg = 50 := by
  sorry

end NUMINAMATH_CALUDE_average_after_adding_constant_specific_average_problem_l3114_311477


namespace NUMINAMATH_CALUDE_sixth_term_is_three_l3114_311461

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_first_three : a 0 + a 1 + a 2 = 168
  specific_diff : a 1 - a 4 = 42

/-- The 6th term of the arithmetic progression is 3 -/
theorem sixth_term_is_three (ap : ArithmeticProgression) : ap.a 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_three_l3114_311461


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l3114_311463

theorem sqrt_sum_fractions : Real.sqrt (1 / 4 + 1 / 25) = Real.sqrt 29 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l3114_311463


namespace NUMINAMATH_CALUDE_odd_function_value_at_half_l3114_311476

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value_at_half
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_neg : ∀ x < 0, f x = 1 / (x + 1)) :
  f (1/2) = -2 := by
sorry

end NUMINAMATH_CALUDE_odd_function_value_at_half_l3114_311476


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3114_311406

theorem sum_of_reciprocals (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_sum : x + y + z = 3) :
  1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2) = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3114_311406


namespace NUMINAMATH_CALUDE_parabola_chord_length_l3114_311469

/-- Parabola struct representing y^2 = ax --/
structure Parabola where
  a : ℝ
  eq : ∀ x y : ℝ, y^2 = a * x

/-- Line struct representing y = m(x - h) + k --/
structure Line where
  m : ℝ
  h : ℝ
  k : ℝ
  eq : ∀ x y : ℝ, y = m * (x - h) + k

/-- The length of the chord AB formed by intersecting a parabola with a line --/
def chordLength (p : Parabola) (l : Line) : ℝ := sorry

theorem parabola_chord_length :
  let p : Parabola := { a := 3, eq := sorry }
  let f : ℝ × ℝ := (3/4, 0)
  let l : Line := { m := Real.sqrt 3 / 3, h := 3/4, k := 0, eq := sorry }
  chordLength p l = 12 := by sorry

end NUMINAMATH_CALUDE_parabola_chord_length_l3114_311469


namespace NUMINAMATH_CALUDE_paint_used_approx_253_33_l3114_311436

/-- Calculate the amount of paint used over five weeks given an initial amount and weekly usage fractions. -/
def paintUsed (initialPaint : ℝ) (week1Fraction week2Fraction week3Fraction week4Fraction week5Fraction : ℝ) : ℝ :=
  let remainingAfterWeek1 := initialPaint * (1 - week1Fraction)
  let remainingAfterWeek2 := remainingAfterWeek1 * (1 - week2Fraction)
  let remainingAfterWeek3 := remainingAfterWeek2 * (1 - week3Fraction)
  let remainingAfterWeek4 := remainingAfterWeek3 * (1 - week4Fraction)
  let usedInWeek5 := remainingAfterWeek4 * week5Fraction
  initialPaint - remainingAfterWeek4 + usedInWeek5

/-- Theorem stating that given the initial paint amount and weekly usage fractions, 
    the total paint used after five weeks is approximately 253.33 gallons. -/
theorem paint_used_approx_253_33 :
  ∃ ε > 0, ε < 0.01 ∧ 
  |paintUsed 360 (1/9) (1/5) (1/3) (1/4) (1/6) - 253.33| < ε :=
sorry

end NUMINAMATH_CALUDE_paint_used_approx_253_33_l3114_311436


namespace NUMINAMATH_CALUDE_quadratic_roots_distinct_real_l3114_311458

/-- Given a quadratic equation bx^2 - 3x√5 + d = 0 with real constants b and d,
    and a discriminant of 25, the roots are distinct and real. -/
theorem quadratic_roots_distinct_real (b d : ℝ) : 
  let discriminant := (-3 * Real.sqrt 5) ^ 2 - 4 * b * d
  ∀ x : ℝ, (b * x^2 - 3 * x * Real.sqrt 5 + d = 0 ∧ discriminant = 25) →
    ∃ y : ℝ, x ≠ y ∧ b * y^2 - 3 * y * Real.sqrt 5 + d = 0 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_distinct_real_l3114_311458


namespace NUMINAMATH_CALUDE_gcd_of_squared_sums_gcd_of_specific_squared_sums_l3114_311453

theorem gcd_of_squared_sums (a b c d e f : ℕ) : 
  Nat.gcd (a^2 + b^2 + c^2) (d^2 + e^2 + f^2) = 
  Nat.gcd ((a^2 + b^2 + c^2) - (d^2 + e^2 + f^2)) (d^2 + e^2 + f^2) :=
by sorry

theorem gcd_of_specific_squared_sums : 
  Nat.gcd (131^2 + 243^2 + 357^2) (130^2 + 242^2 + 358^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_squared_sums_gcd_of_specific_squared_sums_l3114_311453


namespace NUMINAMATH_CALUDE_factorial_series_diverges_l3114_311442

/-- The series Σ(k!/(2^k)) for k from 1 to infinity -/
def factorial_series (k : ℕ) : ℚ := (Nat.factorial k : ℚ) / (2 ^ k : ℚ)

/-- The statement that the factorial series diverges -/
theorem factorial_series_diverges : ¬ Summable factorial_series := by
  sorry

end NUMINAMATH_CALUDE_factorial_series_diverges_l3114_311442


namespace NUMINAMATH_CALUDE_initial_welders_correct_l3114_311432

/-- The number of welders initially working on the order -/
def initial_welders : ℕ := 16

/-- The number of days to complete the order with the initial number of welders -/
def total_days : ℕ := 8

/-- The number of welders that leave after the first day -/
def welders_leaving : ℕ := 9

/-- The number of additional days needed to complete the order after some welders leave -/
def additional_days : ℕ := 16

/-- The theorem stating that the initial number of welders is correct -/
theorem initial_welders_correct :
  (1 : ℚ) + additional_days * (initial_welders - welders_leaving) / initial_welders = total_days :=
by sorry

end NUMINAMATH_CALUDE_initial_welders_correct_l3114_311432


namespace NUMINAMATH_CALUDE_camp_total_boys_l3114_311472

structure Camp where
  totalBoys : ℕ
  schoolA : ℕ
  schoolB : ℕ
  schoolC : ℕ
  schoolAScience : ℕ
  schoolAMath : ℕ
  schoolBScience : ℕ
  schoolBEnglish : ℕ

def isValidCamp (c : Camp) : Prop :=
  c.schoolA + c.schoolB + c.schoolC = c.totalBoys ∧
  c.schoolA = c.totalBoys / 5 ∧
  c.schoolB = c.totalBoys / 4 ∧
  c.schoolC = c.totalBoys - c.schoolA - c.schoolB ∧
  c.schoolAScience = c.schoolA * 3 / 10 ∧
  c.schoolAMath = c.schoolA * 2 / 5 ∧
  c.schoolBScience = c.schoolB / 2 ∧
  c.schoolBEnglish = c.schoolB / 10 ∧
  c.schoolA - c.schoolAScience = 56 ∧
  c.schoolBEnglish = 35

theorem camp_total_boys (c : Camp) (h : isValidCamp c) : c.totalBoys = 400 := by
  sorry

end NUMINAMATH_CALUDE_camp_total_boys_l3114_311472


namespace NUMINAMATH_CALUDE_trip_time_difference_l3114_311493

theorem trip_time_difference (total_distance : ℝ) (first_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  average_speed = 40 →
  let first_half_time := (total_distance / 2) / first_half_speed
  let total_time := total_distance / average_speed
  let second_half_time := total_time - first_half_time
  (second_half_time - first_half_time) / first_half_time * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_difference_l3114_311493


namespace NUMINAMATH_CALUDE_largest_odd_five_digit_has_2_in_hundreds_place_l3114_311496

def Digits : Finset Nat := {1, 2, 3, 5, 8}

def is_odd (n : Nat) : Prop := n % 2 = 1

def is_five_digit (n : Nat) : Prop := 10000 ≤ n ∧ n < 100000

def uses_all_digits (n : Nat) (digits : Finset Nat) : Prop :=
  (Finset.card digits = 5) ∧
  (∀ d ∈ digits, ∃ i : Nat, (n / (10^i)) % 10 = d) ∧
  (∀ i : Nat, i < 5 → (n / (10^i)) % 10 ∈ digits)

def largest_odd_five_digit (n : Nat) : Prop :=
  is_odd n ∧
  is_five_digit n ∧
  uses_all_digits n Digits ∧
  ∀ m : Nat, is_odd m ∧ is_five_digit m ∧ uses_all_digits m Digits → m ≤ n

theorem largest_odd_five_digit_has_2_in_hundreds_place :
  ∃ n : Nat, largest_odd_five_digit n ∧ (n / 100) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_odd_five_digit_has_2_in_hundreds_place_l3114_311496


namespace NUMINAMATH_CALUDE_min_sum_squares_l3114_311424

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 4.8 ∧ x^2 + y^2 + z^2 ≥ m ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀^3 + y₀^3 + z₀^3 - 3*x₀*y₀*z₀ = 8 ∧ x₀^2 + y₀^2 + z₀^2 = m :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3114_311424


namespace NUMINAMATH_CALUDE_complex_sum_problem_l3114_311400

theorem complex_sum_problem (a b c d e f : ℂ) : 
  b = 4 →
  e = -a - c →
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) = 6 + 3 * Complex.I →
  d + f = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l3114_311400


namespace NUMINAMATH_CALUDE_work_completion_proof_l3114_311410

/-- The number of days it takes a to complete the work alone -/
def a_days : ℕ := 45

/-- The number of days it takes b to complete the work alone -/
def b_days : ℕ := 40

/-- The number of days b worked alone to complete the remaining work -/
def b_remaining_days : ℕ := 23

/-- The number of days a worked before leaving -/
def days_a_worked : ℕ := 9

theorem work_completion_proof :
  let total_work := 1
  let a_rate := total_work / a_days
  let b_rate := total_work / b_days
  let combined_rate := a_rate + b_rate
  combined_rate * days_a_worked + b_rate * b_remaining_days = total_work :=
by sorry

end NUMINAMATH_CALUDE_work_completion_proof_l3114_311410


namespace NUMINAMATH_CALUDE_binary_101111011_equals_379_l3114_311489

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101111011_equals_379 :
  binary_to_decimal [true, true, false, true, true, true, true, false, true] = 379 := by
  sorry

end NUMINAMATH_CALUDE_binary_101111011_equals_379_l3114_311489


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3114_311455

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 4 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 16) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3114_311455


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_m_l3114_311422

-- Define the function f(x) = |2x-1|
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -1/2 ∨ x > 3/2} := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f x + 2 * |x + 3| - 4 > m * x) → m ≤ -11 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_m_l3114_311422


namespace NUMINAMATH_CALUDE_peggy_total_dolls_l3114_311419

def initial_dolls : ℕ := 6
def grandmother_gift : ℕ := 30
def additional_dolls : ℕ := grandmother_gift / 2

theorem peggy_total_dolls :
  initial_dolls + grandmother_gift + additional_dolls = 51 := by
  sorry

end NUMINAMATH_CALUDE_peggy_total_dolls_l3114_311419


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3114_311402

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  (a > 0) → (b > 0) → (c > 0) →
  (a^2 + b^2 = c^2) →  -- right-angled triangle condition
  (a^2 + b^2 + c^2 = 4500) →  -- sum of squares condition
  (a = 3*b) →  -- one leg is three times the other
  c = 15 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3114_311402


namespace NUMINAMATH_CALUDE_train_crossing_time_l3114_311490

theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 431.25)
  (h3 : platform_crossing_time = 39) :
  let total_distance := train_length + platform_length
  let train_speed := total_distance / platform_crossing_time
  train_length / train_speed = 16 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3114_311490


namespace NUMINAMATH_CALUDE_first_dog_in_space_l3114_311473

/-- Represents a space mission -/
structure SpaceMission where
  date : String
  payload : String
  isFirst : Bool

/-- Represents the Sputnik 2 mission -/
def sputnik2 : SpaceMission :=
  { date := "November 3, 1957"
  , payload := "dog"
  , isFirst := true }

/-- The name of the first dog in space -/
def firstDogName : String := "Laika"

theorem first_dog_in_space :
  sputnik2.date = "November 3, 1957" →
  sputnik2.payload = "dog" →
  sputnik2.isFirst →
  firstDogName = "Laika" := by
  sorry

end NUMINAMATH_CALUDE_first_dog_in_space_l3114_311473


namespace NUMINAMATH_CALUDE_line_through_points_equation_l3114_311445

-- Define a line by two points
def Line (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • p1 + t • p2}

-- Define the equation of a line in the form ax + by + c = 0
def LineEquation (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Theorem statement
theorem line_through_points_equation :
  Line (3, 0) (0, 2) = LineEquation 2 3 (-6) := by sorry

end NUMINAMATH_CALUDE_line_through_points_equation_l3114_311445


namespace NUMINAMATH_CALUDE_f_equals_g_l3114_311440

/-- Given two functions f and g defined on real numbers,
    where f(x) = x^2 and g(x) = ∛(x^6),
    prove that f and g are equal for all real x. -/
theorem f_equals_g : ∀ x : ℝ, (fun x => x^2) x = (fun x => (x^6)^(1/3)) x := by
  sorry

end NUMINAMATH_CALUDE_f_equals_g_l3114_311440


namespace NUMINAMATH_CALUDE_representatives_selection_theorem_l3114_311435

def number_of_students : ℕ := 6
def number_of_representatives : ℕ := 3

def select_representatives (n m : ℕ) (at_least_one_from_set : ℕ) : ℕ :=
  sorry

theorem representatives_selection_theorem :
  select_representatives number_of_students number_of_representatives 2 = 96 :=
sorry

end NUMINAMATH_CALUDE_representatives_selection_theorem_l3114_311435


namespace NUMINAMATH_CALUDE_joan_seashells_l3114_311487

/-- The number of seashells Joan has after receiving some from Sam -/
def total_seashells (original : ℕ) (received : ℕ) : ℕ :=
  original + received

/-- Theorem: If Joan found 70 seashells and Sam gave her 27 seashells, 
    then Joan now has 97 seashells -/
theorem joan_seashells : total_seashells 70 27 = 97 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l3114_311487


namespace NUMINAMATH_CALUDE_running_contest_average_distance_l3114_311494

/-- The average distance run by two people given their individual distances -/
def average_distance (d1 d2 : ℕ) : ℚ :=
  (d1 + d2) / 2

theorem running_contest_average_distance :
  let block_length : ℕ := 200
  let johnny_laps : ℕ := 4
  let mickey_laps : ℕ := johnny_laps / 2
  let johnny_distance : ℕ := johnny_laps * block_length
  let mickey_distance : ℕ := mickey_laps * block_length
  average_distance johnny_distance mickey_distance = 600 := by
sorry

end NUMINAMATH_CALUDE_running_contest_average_distance_l3114_311494


namespace NUMINAMATH_CALUDE_ball_radius_from_hole_dimensions_l3114_311456

/-- Given a spherical ball partially submerged in a frozen surface,
    if the hole left after removing the ball is 30 cm across and 10 cm deep,
    then the radius of the ball is 16.25 cm. -/
theorem ball_radius_from_hole_dimensions (hole_width : ℝ) (hole_depth : ℝ) (ball_radius : ℝ) :
  hole_width = 30 →
  hole_depth = 10 →
  ball_radius = 16.25 := by
  sorry

end NUMINAMATH_CALUDE_ball_radius_from_hole_dimensions_l3114_311456


namespace NUMINAMATH_CALUDE_inequality_solution_l3114_311491

theorem inequality_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 5) (h3 : x ≠ 7) :
  (x - 1) * (x - 4) * (x - 6) / ((x - 2) * (x - 5) * (x - 7)) > 0 ↔
  x < 1 ∨ (2 < x ∧ x < 4) ∨ (5 < x ∧ x < 6) ∨ 7 < x :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3114_311491


namespace NUMINAMATH_CALUDE_fruit_juice_mixture_l3114_311483

/-- Given a 2-liter mixture that is 10% pure fruit juice, 
    adding 0.4 liters of pure fruit juice results in a 
    mixture that is 25% fruit juice -/
theorem fruit_juice_mixture : 
  let initial_volume : ℝ := 2
  let initial_percentage : ℝ := 0.1
  let added_volume : ℝ := 0.4
  let target_percentage : ℝ := 0.25
  let final_volume := initial_volume + added_volume
  let final_juice_volume := initial_volume * initial_percentage + added_volume
  final_juice_volume / final_volume = target_percentage :=
by sorry


end NUMINAMATH_CALUDE_fruit_juice_mixture_l3114_311483


namespace NUMINAMATH_CALUDE_triangle_side_length_l3114_311446

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 2 →
  c = 2 * Real.sqrt 3 →
  Real.cos A = Real.sqrt 3 / 2 →
  b < c →
  b = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3114_311446


namespace NUMINAMATH_CALUDE_janice_work_hours_janice_work_hours_unique_l3114_311441

/-- Calculates the total pay for a given number of hours worked -/
def totalPay (hours : ℕ) : ℕ :=
  if hours ≤ 40 then
    10 * hours
  else
    400 + 15 * (hours - 40)

/-- Theorem stating that 60 hours of work results in $700 pay -/
theorem janice_work_hours :
  totalPay 60 = 700 :=
by sorry

/-- Theorem stating that 60 is the unique number of hours that results in $700 pay -/
theorem janice_work_hours_unique :
  ∀ h : ℕ, totalPay h = 700 → h = 60 :=
by sorry

end NUMINAMATH_CALUDE_janice_work_hours_janice_work_hours_unique_l3114_311441


namespace NUMINAMATH_CALUDE_inlet_pipe_rate_l3114_311417

/-- Given a tank with the following properties:
    - Capacity of 4320 litres
    - Empties through a leak in 6 hours when full
    - Empties in 12 hours when both the leak and an inlet pipe are open
    This theorem proves that the rate at which the inlet pipe fills water is 6 litres per minute. -/
theorem inlet_pipe_rate (tank_capacity : ℝ) (leak_empty_time : ℝ) (combined_empty_time : ℝ)
  (h1 : tank_capacity = 4320)
  (h2 : leak_empty_time = 6)
  (h3 : combined_empty_time = 12) :
  let leak_rate := tank_capacity / leak_empty_time
  let net_empty_rate := tank_capacity / combined_empty_time
  let inlet_rate := leak_rate - net_empty_rate
  inlet_rate / 60 = 6 := by sorry

end NUMINAMATH_CALUDE_inlet_pipe_rate_l3114_311417


namespace NUMINAMATH_CALUDE_smallest_divisible_by_12_15_16_l3114_311488

theorem smallest_divisible_by_12_15_16 :
  ∃ (n : ℕ), n > 0 ∧ 12 ∣ n ∧ 15 ∣ n ∧ 16 ∣ n ∧
  ∀ (m : ℕ), m > 0 → 12 ∣ m → 15 ∣ m → 16 ∣ m → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_12_15_16_l3114_311488


namespace NUMINAMATH_CALUDE_product_difference_of_squares_l3114_311418

theorem product_difference_of_squares (m n : ℝ) : 
  m * n = ((m + n)/2)^2 - ((m - n)/2)^2 := by sorry

end NUMINAMATH_CALUDE_product_difference_of_squares_l3114_311418
