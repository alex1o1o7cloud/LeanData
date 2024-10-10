import Mathlib

namespace missing_digit_is_one_l947_94709

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def digit_sum (d : ℕ) : ℕ :=
  3 + 5 + 7 + 2 + d + 9

theorem missing_digit_is_one :
  ∀ d : ℕ, d < 10 →
    (is_divisible_by_3 (357200 + d * 10 + 9) ↔ d = 1) :=
by sorry

end missing_digit_is_one_l947_94709


namespace same_side_inequality_l947_94767

/-- Given that point P (a, b) and point Q (1, 2) are on the same side of the line 3x + 2y - 8 = 0,
    prove that 3a + 2b - 8 > 0 -/
theorem same_side_inequality (a b : ℝ) : 
  (∃ (k : ℝ), k > 0 ∧ (3*a + 2*b - 8) * (3*1 + 2*2 - 8) = k * (3*a + 2*b - 8)^2) →
  3*a + 2*b - 8 > 0 := by
sorry

end same_side_inequality_l947_94767


namespace min_sum_dimensions_of_box_l947_94781

/-- Given a rectangular box with positive integer dimensions and volume 2310,
    the minimum possible sum of its three dimensions is 42. -/
theorem min_sum_dimensions_of_box (l w h : ℕ+) : 
  l * w * h = 2310 → l.val + w.val + h.val ≥ 42 := by
  sorry

end min_sum_dimensions_of_box_l947_94781


namespace arithmetic_square_root_of_negative_four_squared_l947_94747

theorem arithmetic_square_root_of_negative_four_squared : Real.sqrt ((-4)^2) = 4 := by
  sorry

end arithmetic_square_root_of_negative_four_squared_l947_94747


namespace factorization_proof_l947_94787

theorem factorization_proof (x : ℝ) : 72 * x^5 - 90 * x^9 = 18 * x^5 * (4 - 5 * x^4) := by
  sorry

end factorization_proof_l947_94787


namespace arithmetic_sequence_problem_l947_94778

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a)
    (h_sum : a 3 + a 4 + a 5 = 3)
    (h_a8 : a 8 = 8) : 
  a 12 = 15 := by
  sorry

end arithmetic_sequence_problem_l947_94778


namespace max_value_interval_m_range_l947_94763

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem max_value_interval_m_range 
  (m : ℝ) 
  (h1 : ∃ (x : ℝ), m < x ∧ x < 8 - m^2 ∧ ∀ (y : ℝ), m < y ∧ y < 8 - m^2 → f y ≤ f x) :
  m ∈ Set.Ioc (-3) (-Real.sqrt 6) :=
sorry

end max_value_interval_m_range_l947_94763


namespace josh_string_cheese_cost_l947_94792

/-- The total cost of Josh's string cheese purchase, including tax and discount -/
def total_cost (pack1 pack2 pack3 : ℕ) (price_per_cheese : ℚ) (discount_rate tax_rate : ℚ) : ℚ :=
  let total_cheese := pack1 + pack2 + pack3
  let subtotal := (total_cheese : ℚ) * price_per_cheese
  let discounted_price := subtotal * (1 - discount_rate)
  let tax := discounted_price * tax_rate
  discounted_price + tax

/-- Theorem stating the total cost of Josh's purchase -/
theorem josh_string_cheese_cost :
  total_cost 18 22 24 (10 / 100) (5 / 100) (12 / 100) = (681 / 100) := by
  sorry

end josh_string_cheese_cost_l947_94792


namespace negation_equivalence_l947_94700

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Teenager : U → Prop)
variable (Responsible : U → Prop)

-- Theorem statement
theorem negation_equivalence :
  (∃ x, Teenager x ∧ ¬Responsible x) ↔ ¬(∀ x, Teenager x → Responsible x) := by
  sorry

end negation_equivalence_l947_94700


namespace quadrilateral_is_rectangle_l947_94706

/-- A quadrilateral in the complex plane -/
structure ComplexQuadrilateral where
  z₁ : ℂ
  z₂ : ℂ
  z₃ : ℂ
  z₄ : ℂ

/-- Predicate to check if a complex number has unit modulus -/
def hasUnitModulus (z : ℂ) : Prop := Complex.abs z = 1

/-- Predicate to check if a ComplexQuadrilateral is a rectangle -/
def isRectangle (q : ComplexQuadrilateral) : Prop :=
  -- Define what it means for a quadrilateral to be a rectangle
  -- This is a placeholder and should be properly defined
  True

/-- Main theorem: Under given conditions, the quadrilateral is a rectangle -/
theorem quadrilateral_is_rectangle (q : ComplexQuadrilateral) 
  (h₁ : hasUnitModulus q.z₁)
  (h₂ : hasUnitModulus q.z₂)
  (h₃ : hasUnitModulus q.z₃)
  (h₄ : hasUnitModulus q.z₄)
  (h_sum : q.z₁ + q.z₂ + q.z₃ + q.z₄ = 0) :
  isRectangle q :=
sorry

end quadrilateral_is_rectangle_l947_94706


namespace existence_of_special_number_l947_94739

def small_number (n : ℕ) : Prop := n ≤ 150

theorem existence_of_special_number :
  ∃ (N : ℕ) (a b : ℕ), 
    small_number a ∧ 
    small_number b ∧ 
    b = a + 1 ∧
    ¬(N % a = 0) ∧
    ¬(N % b = 0) ∧
    (∀ (k : ℕ), small_number k → k ≠ a → k ≠ b → N % k = 0) :=
sorry

end existence_of_special_number_l947_94739


namespace unique_f_three_l947_94795

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem unique_f_three (f : RealFunction) 
  (h : ∀ x y : ℝ, f x * f y - f (x + y) = x - y) : 
  f 3 = -3 := by sorry

end unique_f_three_l947_94795


namespace tangent_line_and_perpendicular_points_l947_94762

-- Define the function f(x) = x³ - x - 1
def f (x : ℝ) : ℝ := x^3 - x - 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem tangent_line_and_perpendicular_points (x y : ℝ) :
  -- The tangent line at (1, -1) has equation 2x - y - 3 = 0
  (x = 1 ∧ y = -1 → 2 * x - y - 3 = 0) ∧
  -- The points of tangency where the tangent line is perpendicular to y = -1/2x + 3
  -- are (1, -1) and (-1, -1)
  (f' x = 2 → (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1)) :=
by sorry

end tangent_line_and_perpendicular_points_l947_94762


namespace parallel_transitivity_l947_94756

-- Define a type for lines in a plane
variable (Line : Type)

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem parallel_transitivity (l1 l2 l3 : Line) :
  parallel l1 l3 → parallel l2 l3 → parallel l1 l2 :=
by
  sorry

#check parallel_transitivity

end parallel_transitivity_l947_94756


namespace number_of_pupils_l947_94744

theorem number_of_pupils (total_people : ℕ) (parents : ℕ) (pupils : ℕ) : 
  total_people = 676 → parents = 22 → pupils = total_people - parents → pupils = 654 := by
  sorry

end number_of_pupils_l947_94744


namespace bike_distance_l947_94796

/-- Theorem: A bike moving at a constant speed of 4 m/s for 8 seconds travels 32 meters. -/
theorem bike_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 4 → time = 8 → distance = speed * time → distance = 32 := by
  sorry

end bike_distance_l947_94796


namespace hotel_meal_spending_l947_94711

theorem hotel_meal_spending (total_persons : ℕ) (regular_spenders : ℕ) (regular_amount : ℕ) 
  (extra_amount : ℕ) (total_spent : ℕ) :
  total_persons = 9 →
  regular_spenders = 8 →
  regular_amount = 12 →
  extra_amount = 8 →
  total_spent = 117 →
  ∃ x : ℕ, (regular_spenders * regular_amount) + (x + extra_amount) = total_spent ∧ x = 13 :=
by sorry

end hotel_meal_spending_l947_94711


namespace distinct_arithmetic_sequences_l947_94749

/-- The largest prime power factor of a positive integer -/
def largest_prime_power_factor (n : ℕ+) : ℕ+ := sorry

/-- Check if two positive integers have the same largest prime power factor -/
def same_largest_prime_power_factor (m n : ℕ+) : Prop := 
  largest_prime_power_factor m = largest_prime_power_factor n

theorem distinct_arithmetic_sequences 
  (n : Fin 10000 → ℕ+) 
  (h_distinct : ∀ i j, i ≠ j → n i ≠ n j) 
  (h_same_factor : ∀ i j, same_largest_prime_power_factor (n i) (n j)) :
  ∃ a : Fin 10000 → ℤ, ∀ i j k l, i ≠ j → a i + k * (n i : ℤ) ≠ a j + l * (n j : ℤ) := by
    sorry

end distinct_arithmetic_sequences_l947_94749


namespace hike_distance_l947_94770

/-- The distance between two points given specific movement conditions -/
theorem hike_distance (A B C : ℝ × ℝ) : 
  B.1 - A.1 = 5 ∧ 
  B.2 - A.2 = 0 ∧ 
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 64 ∧ 
  C.1 - B.1 = C.2 - B.2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 89 + 40 * Real.sqrt 2 :=
by sorry

end hike_distance_l947_94770


namespace part_one_part_two_l947_94773

/-- Given a point M(2m+1, m-4) and N(5, 2) in the Cartesian coordinate system -/
def M (m : ℝ) : ℝ × ℝ := (2*m + 1, m - 4)
def N : ℝ × ℝ := (5, 2)

/-- Part 1: If MN is parallel to the x-axis, then M(13, 2) -/
theorem part_one (m : ℝ) : 
  (M m).2 = N.2 → M m = (13, 2) := by sorry

/-- Part 2: If M is 3 units to the right of the y-axis, then M(3, -3) -/
theorem part_two (m : ℝ) :
  (M m).1 = 3 → M m = (3, -3) := by sorry

end part_one_part_two_l947_94773


namespace unique_quadratic_polynomial_l947_94771

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  b : ℝ
  c : ℝ

/-- The roots of a quadratic polynomial -/
def roots (p : QuadraticPolynomial) : Set ℝ :=
  {x : ℝ | x^2 + p.b * x + p.c = 0}

/-- The set of coefficients of a quadratic polynomial -/
def coefficients (p : QuadraticPolynomial) : Set ℝ :=
  {1, p.b, p.c}

/-- The theorem stating that there exists exactly one quadratic polynomial
    satisfying the given conditions -/
theorem unique_quadratic_polynomial :
  ∃! p : QuadraticPolynomial, roots p = coefficients p :=
sorry

end unique_quadratic_polynomial_l947_94771


namespace remainder_theorem_l947_94733

theorem remainder_theorem (P D Q R D' Q' R' D'' S T : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R')
  (h3 : R' = S + T)
  (h4 : S = D'' * T) :
  P % (D * D' * D'') = D * R' + R :=
sorry

end remainder_theorem_l947_94733


namespace range_of_alpha_minus_beta_l947_94736

theorem range_of_alpha_minus_beta (α β : Real) 
  (h1 : -π ≤ α) (h2 : α ≤ β) (h3 : β ≤ π/2) :
  ∀ x, x ∈ Set.Icc (-3*π/2) 0 ↔ ∃ α' β', 
    -π ≤ α' ∧ α' ≤ β' ∧ β' ≤ π/2 ∧ x = α' - β' :=
by sorry

end range_of_alpha_minus_beta_l947_94736


namespace seed_mixture_weights_l947_94743

/-- Represents a seed mixture with percentages of different grass types -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ
  clover : ℝ
  sum_to_100 : ryegrass + bluegrass + fescue + clover = 100

/-- The final mixture of seeds -/
def FinalMixture (x y z : ℝ) : SeedMixture :=
  { ryegrass := 35
    bluegrass := 30
    fescue := 35
    clover := 0
    sum_to_100 := by norm_num }

/-- The seed mixtures X, Y, and Z -/
def X : SeedMixture :=
  { ryegrass := 40
    bluegrass := 50
    fescue := 0
    clover := 10
    sum_to_100 := by norm_num }

def Y : SeedMixture :=
  { ryegrass := 25
    bluegrass := 0
    fescue := 70
    clover := 5
    sum_to_100 := by norm_num }

def Z : SeedMixture :=
  { ryegrass := 30
    bluegrass := 20
    fescue := 50
    clover := 0
    sum_to_100 := by norm_num }

/-- The theorem stating the weights of seed mixtures X, Y, and Z in the final mixture -/
theorem seed_mixture_weights (x y z : ℝ) 
  (h_total : x + y + z = 8)
  (h_ratio : x / 3 = y / 2 ∧ x / 3 = z / 3)
  (h_final : FinalMixture x y z = 
    { ryegrass := (X.ryegrass * x + Y.ryegrass * y + Z.ryegrass * z) / 8
      bluegrass := (X.bluegrass * x + Y.bluegrass * y + Z.bluegrass * z) / 8
      fescue := (X.fescue * x + Y.fescue * y + Z.fescue * z) / 8
      clover := (X.clover * x + Y.clover * y + Z.clover * z) / 8
      sum_to_100 := sorry }) :
  x = 3 ∧ y = 2 ∧ z = 3 := by
  sorry

end seed_mixture_weights_l947_94743


namespace keith_picked_six_apples_l947_94703

/-- Given the number of apples picked by Mike, eaten by Nancy, and left in total,
    calculate the number of apples picked by Keith. -/
def keith_apples (mike_picked : ℝ) (nancy_ate : ℝ) (total_left : ℝ) : ℝ :=
  total_left - (mike_picked - nancy_ate)

/-- Theorem stating that Keith picked 6.0 apples given the problem conditions. -/
theorem keith_picked_six_apples :
  keith_apples 7.0 3.0 10 = 6.0 := by
  sorry

end keith_picked_six_apples_l947_94703


namespace rationalize_sum_l947_94715

/-- Represents a fraction with a cube root in the denominator -/
structure CubeRootFraction where
  numerator : ℚ
  denominator : ℚ
  root : ℕ

/-- Represents a rationalized fraction with a cube root in the numerator -/
structure RationalizedFraction where
  A : ℤ
  B : ℕ
  C : ℕ

/-- Checks if a number is not divisible by the cube of any prime -/
def not_divisible_by_cube_of_prime (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^3 ∣ n)

/-- Rationalizes a fraction with a cube root in the denominator -/
def rationalize (f : CubeRootFraction) : RationalizedFraction :=
  sorry

theorem rationalize_sum (f : CubeRootFraction) 
  (h : f = { numerator := 2, denominator := 3, root := 7 }) :
  let r := rationalize f
  r.A + r.B + r.C = 72 ∧ 
  r.C > 0 ∧
  not_divisible_by_cube_of_prime r.B :=
sorry

end rationalize_sum_l947_94715


namespace range_of_c_l947_94728

def is_decreasing (f : ℝ → ℝ) := ∀ x y, x < y → f y < f x

def has_two_distinct_real_roots (a b c : ℝ) := 
  b^2 - 4*a*c > 0

def proposition_p (c : ℝ) := is_decreasing (fun x ↦ c^x)

def proposition_q (c : ℝ) := has_two_distinct_real_roots 1 (2 * Real.sqrt c) (1/2)

theorem range_of_c (c : ℝ) 
  (h1 : c > 0) 
  (h2 : c ≠ 1) 
  (h3 : proposition_p c ∨ proposition_q c) 
  (h4 : ¬(proposition_p c ∧ proposition_q c)) : 
  c ∈ Set.Ioc 0 (1/2) ∪ Set.Ioi 1 := by
  sorry

end range_of_c_l947_94728


namespace anthony_transaction_percentage_l947_94758

theorem anthony_transaction_percentage (mabel_transactions cal_transactions anthony_transactions jade_transactions : ℕ) :
  mabel_transactions = 90 →
  cal_transactions = (2 : ℚ) / 3 * anthony_transactions →
  jade_transactions = cal_transactions + 16 →
  jade_transactions = 82 →
  (anthony_transactions : ℚ) / mabel_transactions - 1 = (1 : ℚ) / 10 := by
  sorry

end anthony_transaction_percentage_l947_94758


namespace factory_conditional_probability_l947_94717

/-- Represents the production data for a factory --/
structure FactoryData where
  total_parts : ℕ
  a_parts : ℕ
  a_qualified : ℕ
  b_parts : ℕ
  b_qualified : ℕ

/-- Calculates the conditional probability of a part being qualified given it was produced by A --/
def conditional_probability (data : FactoryData) : ℚ :=
  data.a_qualified / data.a_parts

/-- Theorem stating the conditional probability for the given problem --/
theorem factory_conditional_probability 
  (data : FactoryData)
  (h1 : data.total_parts = 100)
  (h2 : data.a_parts = 40)
  (h3 : data.a_qualified = 35)
  (h4 : data.b_parts = 60)
  (h5 : data.b_qualified = 50)
  (h6 : data.total_parts = data.a_parts + data.b_parts) :
  conditional_probability data = 7/8 := by
  sorry

end factory_conditional_probability_l947_94717


namespace functional_equation_solution_l947_94707

/-- A function satisfying the given properties -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x - y) = g x * g y) ∧
  (∀ x : ℝ, g x ≠ 0) ∧
  (g 0 = 1)

/-- Theorem stating that g(5) = e^5 for functions satisfying the given properties -/
theorem functional_equation_solution (g : ℝ → ℝ) (h : FunctionalEquation g) :
  g 5 = Real.exp 5 := by
  sorry

end functional_equation_solution_l947_94707


namespace custom_mul_seven_neg_two_l947_94761

/-- Custom multiplication operation for rational numbers -/
def custom_mul (a b : ℚ) : ℚ := b^2 - a

/-- Theorem stating that 7 * (-2) = -3 under the custom multiplication -/
theorem custom_mul_seven_neg_two : custom_mul 7 (-2) = -3 := by sorry

end custom_mul_seven_neg_two_l947_94761


namespace oil_bill_problem_l947_94760

/-- The oil bill problem -/
theorem oil_bill_problem (january_bill february_bill : ℚ) 
  (h1 : february_bill / january_bill = 3 / 2)
  (h2 : (february_bill + 30) / january_bill = 5 / 3) :
  january_bill = 180 := by
  sorry

end oil_bill_problem_l947_94760


namespace square_floor_tiles_l947_94776

theorem square_floor_tiles (black_tiles : ℕ) (total_tiles : ℕ) : 
  black_tiles = 101 → 
  (∃ (side_length : ℕ), 
    side_length * side_length = total_tiles ∧ 
    2 * side_length - 1 = black_tiles) → 
  total_tiles = 2601 :=
by
  sorry

end square_floor_tiles_l947_94776


namespace ellipse_equation_equivalence_l947_94779

/-- The equation of an ellipse with foci at (4,0) and (-4,0) -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + y^2) + Real.sqrt ((x + 4)^2 + y^2) = 10

/-- The simplified equation of the ellipse -/
def simplified_equation (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

/-- Theorem stating the equivalence of the two equations -/
theorem ellipse_equation_equivalence :
  ∀ x y : ℝ, ellipse_equation x y ↔ simplified_equation x y :=
sorry

end ellipse_equation_equivalence_l947_94779


namespace students_allowance_l947_94722

theorem students_allowance (allowance : ℚ) : 
  (allowance > 0) →
  (3/5 * allowance + 1/3 * (2/5 * allowance) + 2/5) = 3/2 := by
  sorry

end students_allowance_l947_94722


namespace fifteenth_student_age_l947_94731

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℝ) 
  (group1_size : Nat) 
  (avg_age_group1 : ℝ) 
  (group2_size : Nat) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : group1_size = 5)
  (h4 : avg_age_group1 = 12)
  (h5 : group2_size = 9)
  (h6 : avg_age_group2 = 16)
  (h7 : group1_size + group2_size + 1 = total_students) :
  ∃ (age_15th_student : ℝ), 
    age_15th_student = total_students * avg_age_all - 
      (group1_size * avg_age_group1 + group2_size * avg_age_group2) ∧ 
    age_15th_student = 21 := by
  sorry

end fifteenth_student_age_l947_94731


namespace root_relation_implies_coefficient_ratio_l947_94748

/-- Given two quadratic equations with roots related by a factor of 3, prove the ratio of coefficients -/
theorem root_relation_implies_coefficient_ratio
  (m n p : ℝ)
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (hp : p ≠ 0)
  (h_root_relation : ∀ x, x^2 + p*x + m = 0 → (3*x)^2 + m*(3*x) + n = 0) :
  n / p = -27 := by
  sorry

end root_relation_implies_coefficient_ratio_l947_94748


namespace vacation_payment_difference_l947_94737

/-- Represents the vacation expenses and payments for four people. -/
structure VacationExpenses where
  tom_paid : ℕ
  dorothy_paid : ℕ
  sammy_paid : ℕ
  nancy_paid : ℕ
  total_cost : ℕ
  equal_share : ℕ

/-- The given vacation expenses. -/
def given_expenses : VacationExpenses := {
  tom_paid := 150,
  dorothy_paid := 190,
  sammy_paid := 240,
  nancy_paid := 320,
  total_cost := 900,
  equal_share := 225
}

/-- Theorem stating the difference between Tom's and Dorothy's additional payments. -/
theorem vacation_payment_difference (e : VacationExpenses) 
  (h1 : e.total_cost = e.tom_paid + e.dorothy_paid + e.sammy_paid + e.nancy_paid)
  (h2 : e.equal_share = e.total_cost / 4)
  (h3 : e = given_expenses) :
  (e.equal_share - e.tom_paid) - (e.equal_share - e.dorothy_paid) = 40 := by
  sorry

end vacation_payment_difference_l947_94737


namespace semi_annual_compounding_l947_94721

noncomputable def compound_interest_frequency 
  (initial_investment : ℝ) 
  (annual_rate : ℝ) 
  (final_amount : ℝ) 
  (years : ℝ) : ℝ :=
  let r := annual_rate / 100
  ((final_amount / initial_investment) ^ (1 / (r * years)) - 1) / (r / years)

theorem semi_annual_compounding 
  (initial_investment : ℝ) 
  (annual_rate : ℝ) 
  (final_amount : ℝ) 
  (years : ℝ) 
  (h1 : initial_investment = 10000) 
  (h2 : annual_rate = 3.96) 
  (h3 : final_amount = 10815.83) 
  (h4 : years = 2) :
  ∃ ε > 0, |compound_interest_frequency initial_investment annual_rate final_amount years - 2| < ε :=
sorry

end semi_annual_compounding_l947_94721


namespace three_numbers_sum_l947_94713

theorem three_numbers_sum (x y z : ℝ) : 
  x ≤ y → y ≤ z →
  y = 5 →
  (x + y + z) / 3 = x + 10 →
  (x + y + z) / 3 = z - 15 →
  x + y + z = 30 := by sorry

end three_numbers_sum_l947_94713


namespace hyperbola_b_value_l947_94766

-- Define the hyperbola equation
def hyperbola (x y b : ℝ) : Prop := x^2 / 4 - y^2 / b^2 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := (Real.sqrt 3 * x + y = 0) ∧ (Real.sqrt 3 * x - y = 0)

-- Theorem statement
theorem hyperbola_b_value (b : ℝ) 
  (h1 : b > 0) 
  (h2 : ∀ x y, hyperbola x y b → asymptotes x y) : 
  b = 2 * Real.sqrt 3 := by
sorry

end hyperbola_b_value_l947_94766


namespace correct_average_after_mark_correction_l947_94708

theorem correct_average_after_mark_correction 
  (n : ℕ) 
  (initial_average : ℚ) 
  (wrong_mark correct_mark : ℚ) :
  n = 25 →
  initial_average = 100 →
  wrong_mark = 60 →
  correct_mark = 10 →
  (n * initial_average - (wrong_mark - correct_mark)) / n = 98 := by
sorry

end correct_average_after_mark_correction_l947_94708


namespace painted_area_calculation_l947_94793

/-- Given a rectangular exhibition space with specific dimensions and border widths,
    calculate the area of the painted region inside the border. -/
theorem painted_area_calculation (total_width total_length border_width_standard border_width_door : ℕ)
    (h1 : total_width = 100)
    (h2 : total_length = 150)
    (h3 : border_width_standard = 15)
    (h4 : border_width_door = 20) :
    (total_width - 2 * border_width_standard) * (total_length - border_width_standard - border_width_door) = 8050 :=
by sorry

end painted_area_calculation_l947_94793


namespace checkers_placement_divisibility_l947_94752

theorem checkers_placement_divisibility (p : Nat) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  (Nat.choose (p^2) p) % (p^5) = 0 := by
  sorry

end checkers_placement_divisibility_l947_94752


namespace penny_pudding_grains_l947_94720

-- Define the given conditions
def cans_per_tonne : ℕ := 25000
def grains_per_tonne : ℕ := 50000000

-- Define the function to calculate grains per can
def grains_per_can : ℕ := grains_per_tonne / cans_per_tonne

-- Theorem statement
theorem penny_pudding_grains :
  grains_per_can = 2000 :=
sorry

end penny_pudding_grains_l947_94720


namespace unbroken_seashells_l947_94726

theorem unbroken_seashells (total_seashells : ℕ) (broken_seashells : ℕ) 
  (h1 : total_seashells = 6) 
  (h2 : broken_seashells = 4) : 
  total_seashells - broken_seashells = 2 := by
  sorry

end unbroken_seashells_l947_94726


namespace rotation_equivalence_l947_94701

theorem rotation_equivalence (x : ℝ) : 
  (420 % 360 : ℝ) = (360 - x) % 360 → x < 360 → x = 300 := by
  sorry

end rotation_equivalence_l947_94701


namespace seventeen_in_binary_l947_94710

theorem seventeen_in_binary : 17 = 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0 := by
  sorry

end seventeen_in_binary_l947_94710


namespace roots_in_interval_l947_94702

theorem roots_in_interval (m : ℝ) : 
  (∀ x : ℝ, 4 * x^2 - (3 * m + 1) * x - m - 2 = 0 → -1 < x ∧ x < 2) ↔ 
  -3/2 < m ∧ m < 12/7 :=
by sorry

end roots_in_interval_l947_94702


namespace birthday_celebration_attendance_l947_94724

theorem birthday_celebration_attendance (total_guests : ℕ) 
  (women_ratio : ℚ) (men_count : ℕ) (men_left_ratio : ℚ) (children_left : ℕ) : 
  total_guests = 60 →
  women_ratio = 1/2 →
  men_count = 15 →
  men_left_ratio = 1/3 →
  children_left = 5 →
  ∃ (stayed : ℕ), stayed = 50 := by
  sorry

end birthday_celebration_attendance_l947_94724


namespace midpoint_locus_l947_94729

/-- Given a circle x^2 + y^2 = 1, point A(1,0), and triangle ABC inscribed in the circle
    with angle BAC = 60°, the locus of the midpoint of BC as BC moves on the circle
    is described by the equation x^2 + y^2 = 1/4 for x < 1/4 -/
theorem midpoint_locus (x y : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ),
    x1^2 + y1^2 = 1 ∧
    x2^2 + y2^2 = 1 ∧
    x = (x1 + x2) / 2 ∧
    y = (y1 + y2) / 2 ∧
    (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + y2^2 - (x1 - x2)^2 - (y1 - y2)^2 = 1) →
  x < 1/4 →
  x^2 + y^2 = 1/4 := by
sorry

end midpoint_locus_l947_94729


namespace no_real_solution_nonzero_z_l947_94719

theorem no_real_solution_nonzero_z (x y z : ℝ) : 
  x - y = 2 → xy + z^2 + 1 = 0 → z = 0 := by
  sorry

end no_real_solution_nonzero_z_l947_94719


namespace incorrect_denominator_clearing_l947_94750

theorem incorrect_denominator_clearing (x : ℝ) : 
  ¬((-((3*x+1)/2) - ((2*x-5)/6) > 1) ↔ (3*(3*x+1)+(2*x-5) > -6)) := by
  sorry

end incorrect_denominator_clearing_l947_94750


namespace easter_egg_hunt_friends_l947_94735

/-- Proves the number of friends at Shonda's Easter egg hunt --/
theorem easter_egg_hunt_friends (baskets : ℕ) (eggs_per_basket : ℕ) (eggs_per_person : ℕ)
  (shonda_kids : ℕ) (shonda : ℕ) (other_adults : ℕ) :
  baskets = 15 →
  eggs_per_basket = 12 →
  eggs_per_person = 9 →
  shonda_kids = 2 →
  shonda = 1 →
  other_adults = 7 →
  baskets * eggs_per_basket / eggs_per_person - (shonda_kids + shonda + other_adults) = 10 :=
by
  sorry


end easter_egg_hunt_friends_l947_94735


namespace range_of_a_given_quadratic_inequality_l947_94755

theorem range_of_a_given_quadratic_inequality (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (0 ≤ a ∧ a < 3) :=
sorry

end range_of_a_given_quadratic_inequality_l947_94755


namespace four_digit_number_property_l947_94794

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_valid_n (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n / 10 % 10 ≠ 0)

def split_n (n : ℕ) : ℕ × ℕ :=
  (n / 100, n % 100)

theorem four_digit_number_property (n : ℕ) 
  (h1 : is_valid_n n) 
  (h2 : let (A, B) := split_n n; is_two_digit A ∧ is_two_digit B)
  (h3 : let (A, B) := split_n n; n % (A * B) = 0) :
  n = 1734 ∨ n = 1352 := by
  sorry

end four_digit_number_property_l947_94794


namespace matrix_equals_five_l947_94716

-- Define the matrix
def matrix (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![3*x, 2; 2*x, 4*x]

-- Define the determinant of a 2x2 matrix
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Theorem statement
theorem matrix_equals_five (x : ℝ) : 
  det2x2 (matrix x 0 0) (matrix x 0 1) (matrix x 1 0) (matrix x 1 1) = 5 ↔ 
  x = 5/6 ∨ x = -1/2 := by
sorry

end matrix_equals_five_l947_94716


namespace min_odd_correct_answers_l947_94718

/-- Represents the number of correct answers a student can give -/
inductive CorrectAnswers
  | zero
  | one
  | two
  | three
  | four

/-- Represents the distribution of correct answers among students -/
structure AnswerDistribution where
  total : Nat
  zero : Nat
  one : Nat
  two : Nat
  three : Nat
  four : Nat
  sum_constraint : total = zero + one + two + three + four

/-- Checks if a distribution satisfies the problem constraints -/
def satisfies_constraints (d : AnswerDistribution) : Prop :=
  d.total = 50 ∧
  (∀ (s : Finset Nat), s.card = 40 → s.filter (λ i => i < d.total) ≠ ∅ → 
    (s.filter (λ i => i < d.three)).card ≥ 1) ∧
  (∀ (s : Finset Nat), s.card = 40 → s.filter (λ i => i < d.total) ≠ ∅ → 
    (s.filter (λ i => i < d.two)).card ≥ 2) ∧
  (∀ (s : Finset Nat), s.card = 40 → s.filter (λ i => i < d.total) ≠ ∅ → 
    (s.filter (λ i => i < d.one)).card ≥ 3) ∧
  (∀ (s : Finset Nat), s.card = 40 → s.filter (λ i => i < d.total) ≠ ∅ → 
    (s.filter (λ i => i < d.zero)).card ≥ 4)

/-- The main theorem to prove -/
theorem min_odd_correct_answers (d : AnswerDistribution) 
  (h : satisfies_constraints d) : d.one + d.three ≥ 23 := by
  sorry


end min_odd_correct_answers_l947_94718


namespace max_profit_at_85_optimal_selling_price_l947_94764

/-- Represents the profit function for the item sales --/
def profit (x : ℝ) : ℝ := (10 + x) * (400 - 20 * x) - 500

/-- Theorem stating that the maximum profit is achieved at a selling price of 85 yuan --/
theorem max_profit_at_85 :
  ∃ (x : ℝ), x > 0 ∧ x < 20 ∧
  ∀ (y : ℝ), y > 0 → y < 20 → profit x ≥ profit y ∧
  x + 80 = 85 := by
  sorry

/-- Corollary: The selling price that maximizes profit is 85 yuan --/
theorem optimal_selling_price : 
  ∃ (x : ℝ), x > 0 ∧ x < 20 ∧
  ∀ (y : ℝ), y > 0 → y < 20 → profit x ≥ profit y ∧
  x + 80 = 85 := by
  exact max_profit_at_85

end max_profit_at_85_optimal_selling_price_l947_94764


namespace units_digit_of_product_is_8_l947_94757

def first_four_composites : List Nat := [4, 6, 8, 9]

theorem units_digit_of_product_is_8 :
  (first_four_composites.prod % 10 = 8) := by sorry

end units_digit_of_product_is_8_l947_94757


namespace weight_measurement_l947_94753

def weights : List ℕ := [1, 3, 9, 27]

theorem weight_measurement (w : List ℕ := weights) :
  (∃ (S : List ℕ), S.sum = (List.sum w)) ∧
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ (List.sum w) → 
    ∃ (S : List ℕ), (∀ x ∈ S, x ∈ w) ∧ S.sum = n) :=
by sorry

end weight_measurement_l947_94753


namespace exhibit_visit_time_l947_94789

/-- Represents the time taken by each group to visit the exhibit -/
def group_time (students_per_group : ℕ) (time_per_student : ℕ) : ℕ :=
  students_per_group * time_per_student

/-- Calculates the total time for all groups to visit the exhibit -/
def total_exhibit_time (total_students : ℕ) (num_groups : ℕ) (group_times : List ℕ) : ℕ :=
  let students_per_group := total_students / num_groups
  (group_times.map (group_time students_per_group)).sum

/-- Theorem stating the total time for all groups to visit the exhibit -/
theorem exhibit_visit_time : 
  total_exhibit_time 30 5 [4, 5, 6, 7, 8] = 180 := by
  sorry

#eval total_exhibit_time 30 5 [4, 5, 6, 7, 8]

end exhibit_visit_time_l947_94789


namespace sum_of_digits_of_k_l947_94768

def k : ℕ := 10^30 - 54

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_k : sum_of_digits k = 11 := by sorry

end sum_of_digits_of_k_l947_94768


namespace total_sneaker_spending_l947_94705

/-- Geoff's sneaker spending over three days -/
def sneaker_spending (day1_spend : ℝ) : ℝ :=
  let day2_spend := 4 * day1_spend * (1 - 0.1)  -- 4 times day1 with 10% discount
  let day3_spend := 5 * day1_spend * (1 + 0.08) -- 5 times day1 with 8% tax
  day1_spend + day2_spend + day3_spend

/-- Theorem: Geoff's total sneaker spending over three days is $600 -/
theorem total_sneaker_spending :
  sneaker_spending 60 = 600 := by sorry

end total_sneaker_spending_l947_94705


namespace cattle_milk_production_l947_94777

/-- Represents the total milk production of a group of cows over a given number of days -/
def total_milk_production (num_cows : ℕ) (milk_per_cow : ℕ) (num_days : ℕ) : ℕ :=
  num_cows * milk_per_cow * num_days

/-- Proves that the total milk production of 150 cows over 12 days is 2655000 oz -/
theorem cattle_milk_production : 
  let total_cows : ℕ := 150
  let group1_cows : ℕ := 75
  let group2_cows : ℕ := 75
  let group1_milk_per_cow : ℕ := 1300
  let group2_milk_per_cow : ℕ := 1650
  let num_days : ℕ := 12
  total_milk_production group1_cows group1_milk_per_cow num_days + 
  total_milk_production group2_cows group2_milk_per_cow num_days = 2655000 :=
by
  sorry

end cattle_milk_production_l947_94777


namespace isosceles_triangle_m_values_l947_94765

/-- An isosceles triangle with side lengths satisfying a quadratic equation -/
structure IsoscelesTriangle where
  -- The length of side BC
  bc : ℝ
  -- The parameter m in the quadratic equation
  m : ℝ
  -- The roots of the quadratic equation x^2 - 10x + m = 0 represent the lengths of AB and AC
  root1 : ℝ
  root2 : ℝ
  -- Ensure that root1 and root2 are indeed roots of the equation
  eq1 : root1^2 - 10*root1 + m = 0
  eq2 : root2^2 - 10*root2 + m = 0
  -- Ensure that the triangle is isosceles (two sides are equal)
  isosceles : root1 = root2 ∨ (root1 = bc ∧ root2 = 10 - bc) ∨ (root2 = bc ∧ root1 = 10 - bc)
  -- Given condition that BC = 8
  bc_eq_8 : bc = 8

/-- The theorem stating that m is either 16 or 25 -/
theorem isosceles_triangle_m_values (t : IsoscelesTriangle) : t.m = 16 ∨ t.m = 25 := by
  sorry

end isosceles_triangle_m_values_l947_94765


namespace smallest_c_for_positive_quadratic_l947_94791

theorem smallest_c_for_positive_quadratic : 
  ∃ c : ℤ, (∀ x : ℝ, x^2 + c*x + 15 > 0) ∧ 
  (∀ d : ℤ, d < c → ∃ x : ℝ, x^2 + d*x + 15 ≤ 0) ∧ 
  c = -7 := by
  sorry

end smallest_c_for_positive_quadratic_l947_94791


namespace pen_sales_revenue_pen_sales_revenue_proof_l947_94775

theorem pen_sales_revenue : ℝ → Prop :=
  fun total_revenue =>
    ∀ (total_pens : ℕ) (displayed_pens : ℕ) (storeroom_pens : ℕ),
      (displayed_pens : ℝ) = 0.3 * total_pens ∧
      (storeroom_pens : ℝ) = 0.7 * total_pens ∧
      storeroom_pens = 210 ∧
      total_revenue = (displayed_pens : ℝ) * 2 →
      total_revenue = 180

-- The proof is omitted
theorem pen_sales_revenue_proof : pen_sales_revenue 180 := by
  sorry

end pen_sales_revenue_pen_sales_revenue_proof_l947_94775


namespace right_triangle_increase_sides_acute_l947_94745

/-- Given a right-angled triangle, increasing all sides by the same amount results in an acute-angled triangle -/
theorem right_triangle_increase_sides_acute (a b c k : ℝ) 
  (h_right : a^2 + b^2 = c^2) -- Original triangle is right-angled
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ k > 0) -- Sides and increase are positive
  : (a + k)^2 + (b + k)^2 > (c + k)^2 := by sorry

end right_triangle_increase_sides_acute_l947_94745


namespace hyperbola_vertex_distance_l947_94714

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ),
  x^2 / 121 - y^2 / 49 = 1 →
  ∃ (v1 v2 : ℝ × ℝ),
    v1 ∈ {p : ℝ × ℝ | p.1^2 / 121 - p.2^2 / 49 = 1} ∧
    v2 ∈ {p : ℝ × ℝ | p.1^2 / 121 - p.2^2 / 49 = 1} ∧
    v1 ≠ v2 ∧
    ∀ (v : ℝ × ℝ),
      v ∈ {p : ℝ × ℝ | p.1^2 / 121 - p.2^2 / 49 = 1} →
      v.2 = 0 →
      v = v1 ∨ v = v2 ∧
    Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = 22 :=
by
  sorry

end hyperbola_vertex_distance_l947_94714


namespace smallest_square_cover_l947_94759

/-- The side length of the smallest square that can be covered by 2-by-4 rectangles -/
def smallest_square_side : ℕ := 8

/-- The area of a 2-by-4 rectangle -/
def rectangle_area : ℕ := 2 * 4

/-- The number of 2-by-4 rectangles needed to cover the smallest square -/
def num_rectangles : ℕ := smallest_square_side^2 / rectangle_area

theorem smallest_square_cover :
  (∀ n : ℕ, n < smallest_square_side → n^2 % rectangle_area ≠ 0) ∧
  smallest_square_side^2 % rectangle_area = 0 ∧
  num_rectangles = 8 := by sorry

end smallest_square_cover_l947_94759


namespace minimum_average_score_for_target_l947_94798

def current_scores : List ℝ := [92, 81, 75, 65, 88]
def bonus_points : ℝ := 5
def target_increase : ℝ := 6

theorem minimum_average_score_for_target (new_test1 new_test2 : ℝ) :
  let current_avg := (current_scores.sum) / current_scores.length
  let new_avg := ((current_scores.sum + (new_test1 + bonus_points) + new_test2) / 
                  (current_scores.length + 2))
  let min_new_avg := (new_test1 + new_test2) / 2
  (new_avg = current_avg + target_increase) → min_new_avg ≥ 99 := by
  sorry

end minimum_average_score_for_target_l947_94798


namespace g_of_3_equals_5_l947_94740

-- Define the function g
def g (x : ℝ) : ℝ := 2 * (x - 2) + 3

-- State the theorem
theorem g_of_3_equals_5 : g 3 = 5 := by
  sorry

end g_of_3_equals_5_l947_94740


namespace distinct_arrangements_count_l947_94790

/-- A regular six-pointed star -/
structure SixPointedStar :=
  (points : Fin 12)

/-- The number of symmetries of a regular six-pointed star -/
def star_symmetries : ℕ := 12

/-- The number of distinct arrangements of 12 different objects on a regular six-pointed star,
    where reflections and rotations are considered equivalent -/
def distinct_arrangements : ℕ := Nat.factorial 12 / star_symmetries

theorem distinct_arrangements_count :
  distinct_arrangements = 39916800 := by
  sorry

end distinct_arrangements_count_l947_94790


namespace interest_rate_calculation_l947_94772

/-- Calculate the interest rate per annum given the principal, amount, and time period. -/
theorem interest_rate_calculation (principal amount : ℕ) (time : ℚ) :
  principal = 1100 →
  amount = 1232 →
  time = 12 / 5 →
  (amount - principal) * 100 / (principal * time) = 5 := by
  sorry

end interest_rate_calculation_l947_94772


namespace annual_price_decrease_l947_94742

def price_2001 : ℝ := 1950
def price_2009 : ℝ := 1670
def year_2001 : ℕ := 2001
def year_2009 : ℕ := 2009

theorem annual_price_decrease :
  (price_2001 - price_2009) / (year_2009 - year_2001 : ℝ) = 35 := by
  sorry

end annual_price_decrease_l947_94742


namespace georges_walk_l947_94780

/-- Given that George walks 1 mile to school at 3 mph normally, prove that
    if he walks the first 1/2 mile at 2 mph, he must run the last 1/2 mile
    at 6 mph to arrive at the same time. -/
theorem georges_walk (normal_distance : Real) (normal_speed : Real) 
  (first_half_distance : Real) (first_half_speed : Real) 
  (second_half_distance : Real) (second_half_speed : Real) :
  normal_distance = 1 ∧ 
  normal_speed = 3 ∧ 
  first_half_distance = 1/2 ∧ 
  first_half_speed = 2 ∧ 
  second_half_distance = 1/2 ∧
  normal_distance / normal_speed = 
    first_half_distance / first_half_speed + second_half_distance / second_half_speed →
  second_half_speed = 6 := by
  sorry

#check georges_walk

end georges_walk_l947_94780


namespace frog_jumps_theorem_l947_94727

/-- Represents a hexagon with vertices A, B, C, D, E, F -/
inductive Vertex
| A | B | C | D | E | F

/-- Represents the number of paths from A to C in n jumps -/
def paths_to_C (n : ℕ) : ℕ := (2^n - 1) / 3

/-- Represents the number of paths from A to C in n jumps avoiding D -/
def paths_to_C_avoiding_D (n : ℕ) : ℕ := 3^(n/2 - 1)

/-- Represents the probability of survival after n jumps with a mine at D -/
def survival_probability (n : ℕ) : ℚ := (3/4)^((n + 1)/2 - 1)

/-- The average lifespan of frogs -/
def average_lifespan : ℕ := 9

/-- Main theorem stating the properties of frog jumps on a hexagon -/
theorem frog_jumps_theorem :
  ∀ n : ℕ,
  (paths_to_C n = (2^n - 1) / 3) ∧
  (paths_to_C_avoiding_D n = 3^(n/2 - 1)) ∧
  (survival_probability n = (3/4)^((n + 1)/2 - 1)) ∧
  (average_lifespan = 9) :=
by sorry

end frog_jumps_theorem_l947_94727


namespace monotonic_increasing_range_l947_94754

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + x^2 + m*x + 1

-- State the theorem
theorem monotonic_increasing_range (m : ℝ) :
  (∀ x : ℝ, Monotone (f m)) → m ≥ 1/3 := by
  sorry

end monotonic_increasing_range_l947_94754


namespace blocks_added_l947_94784

def initial_blocks : ℕ := 35
def final_blocks : ℕ := 65

theorem blocks_added : final_blocks - initial_blocks = 30 := by
  sorry

end blocks_added_l947_94784


namespace jelly_beans_weight_l947_94738

theorem jelly_beans_weight (initial_weight : ℝ) : 
  initial_weight > 0 →
  2 * (4 * initial_weight) = 16 →
  initial_weight = 2 := by
sorry

end jelly_beans_weight_l947_94738


namespace other_root_of_quadratic_l947_94712

theorem other_root_of_quadratic (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - (a+b+c)*x + ab + bc + ca
  let ab := a + b
  let ab_bc_ca := ab + bc + ca
  f ab = 0 →
  ∃ k, f k = 0 ∧ k = (ab + bc + ca) / (a + b) :=
by
  sorry

end other_root_of_quadratic_l947_94712


namespace sequence_ratio_density_l947_94751

theorem sequence_ratio_density (a : ℕ → ℕ) 
  (h : ∀ n : ℕ, 0 < a (n + 1) - a n ∧ a (n + 1) - a n < Real.sqrt (a n)) :
  ∀ x y : ℝ, 0 < x → x < y → y < 1 → 
  ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ x < (a k : ℝ) / (a m : ℝ) ∧ (a k : ℝ) / (a m : ℝ) < y :=
by sorry

end sequence_ratio_density_l947_94751


namespace units_digit_of_k_squared_plus_two_to_k_l947_94723

def k : ℕ := 2017^2 + 2^2017

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ := k) : (k^2 + 2^k) % 10 = 3 := by
  sorry

end units_digit_of_k_squared_plus_two_to_k_l947_94723


namespace swimming_practice_months_l947_94786

def total_required_hours : ℕ := 1500
def completed_hours : ℕ := 180
def monthly_practice_hours : ℕ := 220

theorem swimming_practice_months :
  (total_required_hours - completed_hours) / monthly_practice_hours = 6 :=
by sorry

end swimming_practice_months_l947_94786


namespace james_tylenol_intake_l947_94725

/-- Calculates the total milligrams of Tylenol taken in a day -/
def tylenolPerDay (tabletsPerDose : ℕ) (mgPerTablet : ℕ) (hoursPerDose : ℕ) (hoursPerDay : ℕ) : ℕ :=
  let dosesPerDay := hoursPerDay / hoursPerDose
  let mgPerDose := tabletsPerDose * mgPerTablet
  dosesPerDay * mgPerDose

/-- Proves that James takes 3000 mg of Tylenol per day -/
theorem james_tylenol_intake :
  tylenolPerDay 2 375 6 24 = 3000 := by
  sorry

end james_tylenol_intake_l947_94725


namespace flowers_in_vase_l947_94746

theorem flowers_in_vase (initial_flowers : ℕ) (removed_flowers : ℕ) : 
  initial_flowers = 13 → removed_flowers = 7 → initial_flowers - removed_flowers = 6 := by
  sorry

end flowers_in_vase_l947_94746


namespace necessary_but_not_sufficient_l947_94734

theorem necessary_but_not_sufficient (A B C : Set α) (h : ∀ a, a ∈ A ↔ (a ∈ B ∧ a ∈ C)) :
  (∀ a, a ∈ A → a ∈ B) ∧ ¬(∀ a, a ∈ B → a ∈ A) :=
by sorry

end necessary_but_not_sufficient_l947_94734


namespace paige_finished_problems_l947_94785

/-- Calculates the number of finished homework problems -/
def finished_problems (total : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  total - (remaining_pages * problems_per_page)

/-- Theorem: Paige finished 47 problems -/
theorem paige_finished_problems :
  finished_problems 110 7 9 = 47 := by
  sorry

end paige_finished_problems_l947_94785


namespace fencing_required_l947_94769

/-- Calculates the fencing required for a rectangular field with one side uncovered -/
theorem fencing_required (length width area : ℝ) (h1 : length = 34) (h2 : area = 680) 
  (h3 : area = length * width) : 2 * width + length = 74 := by
  sorry

end fencing_required_l947_94769


namespace sqrt_inequality_l947_94788

theorem sqrt_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  Real.sqrt (a / d) > Real.sqrt (b / c) := by
  sorry

end sqrt_inequality_l947_94788


namespace investment_growth_l947_94741

/-- Calculates the final amount after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Proves that the given investment scenario results in the expected amount -/
theorem investment_growth (principal : ℝ) (rate : ℝ) (years : ℕ) 
  (h1 : principal = 2000)
  (h2 : rate = 0.05)
  (h3 : years = 18) :
  ∃ ε > 0, |compound_interest principal rate years - 4813.24| < ε :=
sorry

end investment_growth_l947_94741


namespace factorization_difference_l947_94774

theorem factorization_difference (a b : ℤ) : 
  (∀ y : ℝ, 5 * y^2 + 3 * y - 44 = (5 * y + a) * (y + b)) → 
  a - b = -15 := by
sorry

end factorization_difference_l947_94774


namespace point_B_coordinates_l947_94704

/-- Given point A and vector AB, find the coordinates of point B -/
theorem point_B_coordinates (A B : ℝ × ℝ × ℝ) (AB : ℝ × ℝ × ℝ) :
  A = (3, -1, 0) →
  AB = (2, 5, -3) →
  B = (5, 4, -3) :=
by sorry

end point_B_coordinates_l947_94704


namespace lindsey_september_savings_l947_94730

/-- The amount of money Lindsey saved in September -/
def september_savings : ℕ := sorry

/-- The amount of money Lindsey saved in October -/
def october_savings : ℕ := 37

/-- The amount of money Lindsey saved in November -/
def november_savings : ℕ := 11

/-- The amount of money Lindsey's mom gave her -/
def mom_gift : ℕ := 25

/-- The cost of the video game Lindsey bought -/
def video_game_cost : ℕ := 87

/-- The amount of money Lindsey had left after buying the video game -/
def money_left : ℕ := 36

/-- Theorem stating that Lindsey saved $50 in September -/
theorem lindsey_september_savings :
  september_savings = 50 ∧
  september_savings + october_savings + november_savings > 75 ∧
  september_savings + october_savings + november_savings + mom_gift = video_game_cost + money_left :=
sorry

end lindsey_september_savings_l947_94730


namespace max_min_values_of_f_l947_94797

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x^2 - 4 * x - 6)

theorem max_min_values_of_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max = 1 ∧
    min = 1 / Real.exp 8 :=
by sorry

end max_min_values_of_f_l947_94797


namespace two_solutions_for_x_squared_minus_y_squared_77_l947_94782

theorem two_solutions_for_x_squared_minus_y_squared_77 :
  ∃! n : ℕ, n > 0 ∧ 
  (∃ s : Finset (ℕ × ℕ), s.card = n ∧ 
    (∀ p ∈ s, p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 - p.2^2 = 77) ∧
    (∀ x y : ℕ, x > 0 → y > 0 → x^2 - y^2 = 77 → (x, y) ∈ s)) :=
sorry

end two_solutions_for_x_squared_minus_y_squared_77_l947_94782


namespace line_equation_through_point_with_slope_angle_l947_94799

/-- The equation of a line passing through a given point with a given slope angle -/
theorem line_equation_through_point_with_slope_angle 
  (x₀ y₀ : ℝ) (θ : ℝ) :
  x₀ = Real.sqrt 3 →
  y₀ = 1 →
  θ = π / 3 →
  ∃ (a b c : ℝ), 
    a * Real.sqrt 3 + b * 1 + c = 0 ∧
    a * x + b * y + c = 0 ∧
    a = Real.sqrt 3 ∧
    b = -1 ∧
    c = -2 :=
by sorry

end line_equation_through_point_with_slope_angle_l947_94799


namespace winnie_balloon_distribution_l947_94732

theorem winnie_balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 242) (h2 : num_friends = 12) : 
  total_balloons % num_friends = 2 := by
  sorry

end winnie_balloon_distribution_l947_94732


namespace first_digit_base9_of_122012_base3_l947_94783

/-- Converts a base 3 number to base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 3 * acc) 0

/-- Calculates the first digit of a number in base 9 -/
def firstDigitBase9 (n : Nat) : Nat :=
  if n < 9 then n else firstDigitBase9 (n / 9)

theorem first_digit_base9_of_122012_base3 :
  let y := base3ToBase10 [1, 2, 2, 0, 1, 2]
  firstDigitBase9 y = 5 := by
  sorry

end first_digit_base9_of_122012_base3_l947_94783
