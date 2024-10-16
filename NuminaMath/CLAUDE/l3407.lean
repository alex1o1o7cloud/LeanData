import Mathlib

namespace NUMINAMATH_CALUDE_correct_statements_count_l3407_340774

-- Define a structure for a statistical statement
structure StatStatement :=
  (id : Nat)
  (content : String)
  (isCorrect : Bool)

-- Define the four statements
def statement1 : StatStatement :=
  ⟨1, "Subtracting the same number from each datum in a data set does not change the mean or the variance", false⟩

def statement2 : StatStatement :=
  ⟨2, "In a survey of audience feedback in a theater, randomly selecting one row from 50 rows (equal number of people in each row) for the survey is an example of stratified sampling", false⟩

def statement3 : StatStatement :=
  ⟨3, "It is known that random variable X follows a normal distribution N(3,1), and P(2≤X≤4) = 0.6826, then P(X>4) is equal to 0.1587", true⟩

def statement4 : StatStatement :=
  ⟨4, "A unit has 750 employees, of which there are 350 young workers, 250 middle-aged workers, and 150 elderly workers. To understand the health status of the workers in the unit, stratified sampling is used to draw a sample. If there are 7 young workers in the sample, then the sample size is 15", true⟩

-- Define the list of all statements
def allStatements : List StatStatement := [statement1, statement2, statement3, statement4]

-- Theorem to prove
theorem correct_statements_count :
  (allStatements.filter (λ s => s.isCorrect)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_statements_count_l3407_340774


namespace NUMINAMATH_CALUDE_triangle_area_l3407_340735

/-- The area of a triangle with base 10 cm and height 3 cm is 15 cm² -/
theorem triangle_area : 
  let base : ℝ := 10
  let height : ℝ := 3
  let area : ℝ := (1/2) * base * height
  area = 15 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3407_340735


namespace NUMINAMATH_CALUDE_epsilon_delta_condition_l3407_340791

def f (x : ℝ) := x^2 + 1

theorem epsilon_delta_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, |x - 1| < b → |f x - 3| < a) ↔ b ≤ a / 2 := by
  sorry

end NUMINAMATH_CALUDE_epsilon_delta_condition_l3407_340791


namespace NUMINAMATH_CALUDE_cylinder_no_triangular_cross_section_l3407_340758

-- Define the type for geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | TriangularPrism
  | Cube

-- Define a function that determines if a solid can have a triangular cross-section
def canHaveTriangularCrossSection (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => False
  | _ => True

-- Theorem stating that only the Cylinder cannot have a triangular cross-section
theorem cylinder_no_triangular_cross_section :
  ∀ (solid : GeometricSolid),
    ¬(canHaveTriangularCrossSection solid) ↔ solid = GeometricSolid.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_cylinder_no_triangular_cross_section_l3407_340758


namespace NUMINAMATH_CALUDE_min_t_value_l3407_340780

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2 * Real.sqrt 2) ^ 2 + (y - 1) ^ 2 = 1

-- Define points A and B
def point_A (t : ℝ) : ℝ × ℝ := (-t, 0)
def point_B (t : ℝ) : ℝ × ℝ := (t, 0)

-- Define the condition for point P
def point_P_condition (P : ℝ × ℝ) (t : ℝ) : Prop :=
  circle_C P.1 P.2 ∧
  let AP := (P.1 + t, P.2)
  let BP := (P.1 - t, P.2)
  AP.1 * BP.1 + AP.2 * BP.2 = 0

-- State the theorem
theorem min_t_value :
  ∀ t : ℝ, t > 0 →
  (∃ P : ℝ × ℝ, point_P_condition P t) →
  (∀ t' : ℝ, t' > 0 ∧ (∃ P : ℝ × ℝ, point_P_condition P t') → t' ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_min_t_value_l3407_340780


namespace NUMINAMATH_CALUDE_parabola_intersection_difference_l3407_340788

/-- The difference between the larger and smaller x-coordinates of the intersection points of two parabolas -/
theorem parabola_intersection_difference : ∃ (a c : ℝ),
  (3 * a^2 - 6 * a + 2 = -2 * a^2 - 4 * a + 3) ∧
  (3 * c^2 - 6 * c + 2 = -2 * c^2 - 4 * c + 3) ∧
  (c ≥ a) ∧
  (c - a = 2 * Real.sqrt 6 / 5) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_difference_l3407_340788


namespace NUMINAMATH_CALUDE_yahs_to_bahs_500_l3407_340732

/-- Represents the exchange rate between bahs and rahs -/
def bah_to_rah_rate : ℚ := 16 / 10

/-- Represents the exchange rate between rahs and yahs -/
def rah_to_yah_rate : ℚ := 10 / 6

/-- Converts yahs to bahs -/
def yahs_to_bahs (yahs : ℚ) : ℚ :=
  yahs * (1 / rah_to_yah_rate) * (1 / bah_to_rah_rate)

theorem yahs_to_bahs_500 :
  yahs_to_bahs 500 = 187.5 := by sorry

end NUMINAMATH_CALUDE_yahs_to_bahs_500_l3407_340732


namespace NUMINAMATH_CALUDE_shoe_price_proof_l3407_340761

/-- The original price of the shoes -/
def P : ℝ := 200

/-- The price of each shirt -/
def shirt_price : ℝ := 80

/-- The number of shirts bought -/
def num_shirts : ℕ := 2

/-- The discount rate on the shoes -/
def shoe_discount : ℝ := 0.3

/-- The additional discount rate on the total -/
def total_discount : ℝ := 0.05

/-- The final amount paid after all discounts -/
def final_amount : ℝ := 285

/-- Theorem stating that given the conditions, the original price of the shoes is $200 -/
theorem shoe_price_proof : 
  (1 - total_discount) * ((1 - shoe_discount) * P + num_shirts * shirt_price) = final_amount :=
by sorry

end NUMINAMATH_CALUDE_shoe_price_proof_l3407_340761


namespace NUMINAMATH_CALUDE_insane_vampire_statement_l3407_340775

/-- Represents a being in Transylvania -/
inductive TransylvanianBeing
| Human
| Vampire

/-- Represents the mental state of a being -/
inductive MentalState
| Sane
| Insane

/-- Represents a Transylvanian entity with a mental state -/
structure Transylvanian :=
  (being : TransylvanianBeing)
  (state : MentalState)

/-- Predicate for whether a Transylvanian makes the statement "I am not a sane person" -/
def makesSanityStatement (t : Transylvanian) : Prop :=
  t.state = MentalState.Insane

/-- Theorem: A Transylvanian who states "I am not a sane person" must be an insane vampire -/
theorem insane_vampire_statement 
  (t : Transylvanian) 
  (h : makesSanityStatement t) : 
  t.being = TransylvanianBeing.Vampire ∧ t.state = MentalState.Insane :=
by sorry


end NUMINAMATH_CALUDE_insane_vampire_statement_l3407_340775


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l3407_340717

/-- A function that returns the set of digits of a natural number -/
def digits (n : ℕ) : Finset ℕ :=
  sorry

/-- A function that checks if a natural number is a six-digit number -/
def isSixDigit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

/-- The theorem stating that 142857 is the unique six-digit number satisfying the given conditions -/
theorem unique_six_digit_number :
  ∃! p : ℕ, isSixDigit p ∧
    (∀ i : Fin 6, isSixDigit ((i.val + 1) * p)) ∧
    (∀ i : Fin 6, digits ((i.val + 1) * p) = digits p) ∧
    p = 142857 :=
  sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l3407_340717


namespace NUMINAMATH_CALUDE_fraction_subtraction_equals_two_l3407_340787

theorem fraction_subtraction_equals_two (a : ℝ) (h : a ≠ 1) :
  (2 * a) / (a - 1) - 2 / (a - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_equals_two_l3407_340787


namespace NUMINAMATH_CALUDE_distance_between_points_l3407_340770

def point1 : ℝ × ℝ := (3, 7)
def point2 : ℝ × ℝ := (-5, 2)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 89 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3407_340770


namespace NUMINAMATH_CALUDE_problem_statement_l3407_340738

theorem problem_statement (x y : ℝ) (h1 : x - y = 3) (h2 : x * y = 2) :
  3 * x - 5 * x * y - 3 * y = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3407_340738


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3407_340743

theorem lcm_from_product_and_hcf (a b : ℕ+) 
  (h_product : a * b = 18750)
  (h_hcf : Nat.gcd a b = 25) :
  Nat.lcm a b = 750 := by
sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3407_340743


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l3407_340704

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_coefficient_problem (m : ℕ+) :
  let a := binomial (2 * m) m
  let b := binomial (2 * m + 1) m
  13 * a = 7 * b → m = 6 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l3407_340704


namespace NUMINAMATH_CALUDE_projection_onto_orthogonal_vector_l3407_340783

/-- Given orthogonal vectors a and b in R^2, and the projection of (4, -2) onto a,
    prove that the projection of (4, -2) onto b is (24/5, -2/5). -/
theorem projection_onto_orthogonal_vector 
  (a b : ℝ × ℝ) 
  (h_orthogonal : a.1 * b.1 + a.2 * b.2 = 0) 
  (h_proj_a : (4 : ℝ) * a.1 + (-2 : ℝ) * a.2 = (-4/5 : ℝ) * (a.1^2 + a.2^2)) :
  (4 : ℝ) * b.1 + (-2 : ℝ) * b.2 = (24/5 : ℝ) * (b.1^2 + b.2^2) :=
by sorry

end NUMINAMATH_CALUDE_projection_onto_orthogonal_vector_l3407_340783


namespace NUMINAMATH_CALUDE_max_ab_value_l3407_340709

-- Define the two quadratic functions
def f (x : ℝ) : ℝ := x^2 - 2*x + 2
def g (a b x : ℝ) : ℝ := -x^2 + a*x + b

-- State the theorem
theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x₀ : ℝ, f x₀ = g a b x₀ ∧ 
    (2*x₀ - 2) * (-2*x₀ + a) = -1) →
  ab ≤ 25/16 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀*b₀ = 25/16 :=
by
  sorry

end NUMINAMATH_CALUDE_max_ab_value_l3407_340709


namespace NUMINAMATH_CALUDE_smallest_number_game_l3407_340734

theorem smallest_number_game (alice_number : ℕ) (bob_number : ℕ) : 
  alice_number = 45 →
  (∀ p : ℕ, Nat.Prime p → p ∣ alice_number → p ∣ bob_number) →
  5 ∣ bob_number →
  bob_number > 0 →
  (∀ n : ℕ, n > 0 → (∀ p : ℕ, Nat.Prime p → p ∣ alice_number → p ∣ n) → 5 ∣ n → n ≥ bob_number) →
  bob_number = 15 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_game_l3407_340734


namespace NUMINAMATH_CALUDE_modulus_of_complex_quotient_l3407_340757

/-- The modulus of the complex number (4+3i)/(1-2i) is √5 -/
theorem modulus_of_complex_quotient :
  Complex.abs ((4 : ℂ) + 3 * Complex.I) / ((1 : ℂ) - 2 * Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_quotient_l3407_340757


namespace NUMINAMATH_CALUDE_max_additional_plates_l3407_340740

/-- Represents the sets of letters for license plates -/
structure LetterSets :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Calculates the number of possible license plates -/
def numPlates (s : LetterSets) : ℕ :=
  s.first * s.second * s.third

/-- The initial letter sets -/
def initialSets : LetterSets :=
  ⟨5, 3, 4⟩

/-- The number of new letters to be added -/
def newLetters : ℕ :=
  4

/-- Constraint: at least one letter must be added to second and third sets -/
def validDistribution (d : LetterSets) : Prop :=
  d.second > initialSets.second ∧ d.third > initialSets.third ∧
  d.first + d.second + d.third = initialSets.first + initialSets.second + initialSets.third + newLetters

theorem max_additional_plates :
  ∃ (d : LetterSets), validDistribution d ∧
    ∀ (d' : LetterSets), validDistribution d' →
      numPlates d - numPlates initialSets ≥ numPlates d' - numPlates initialSets ∧
      numPlates d - numPlates initialSets = 90 :=
by sorry

end NUMINAMATH_CALUDE_max_additional_plates_l3407_340740


namespace NUMINAMATH_CALUDE_carreys_fixed_amount_is_20_l3407_340728

/-- The fixed amount Carrey paid for the car rental -/
def carreys_fixed_amount : ℝ := 20

/-- The rate per kilometer for Carrey's rental -/
def carreys_rate_per_km : ℝ := 0.25

/-- The fixed amount Samuel paid for the car rental -/
def samuels_fixed_amount : ℝ := 24

/-- The rate per kilometer for Samuel's rental -/
def samuels_rate_per_km : ℝ := 0.16

/-- The number of kilometers driven by both Carrey and Samuel -/
def kilometers_driven : ℝ := 44.44444444444444

theorem carreys_fixed_amount_is_20 :
  carreys_fixed_amount + carreys_rate_per_km * kilometers_driven =
  samuels_fixed_amount + samuels_rate_per_km * kilometers_driven :=
sorry

end NUMINAMATH_CALUDE_carreys_fixed_amount_is_20_l3407_340728


namespace NUMINAMATH_CALUDE_polynomial_at_negative_two_l3407_340764

def polynomial (x : ℝ) : ℝ := 2 * x^4 - 3 * x^3 + x^2 - 2 * x + 4

theorem polynomial_at_negative_two :
  polynomial (-2) = 68 := by sorry

end NUMINAMATH_CALUDE_polynomial_at_negative_two_l3407_340764


namespace NUMINAMATH_CALUDE_range_of_m_l3407_340767

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - |x| else x^2 - 4*x + 3

theorem range_of_m (m : ℝ) : f (f m) ≥ 0 → m ∈ Set.Icc (-2) (2 + Real.sqrt 2) ∪ Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3407_340767


namespace NUMINAMATH_CALUDE_problem_solution_l3407_340751

-- Define the function f(x) = |x-a| + 3x
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

theorem problem_solution :
  (∀ x : ℝ, f 1 x > 3 * x + 2 ↔ (x > 3 ∨ x < -1)) ∧
  (∀ a : ℝ, a > 0 → (∀ x : ℝ, f a x ≤ 0 ↔ x ≤ -1) → a = 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3407_340751


namespace NUMINAMATH_CALUDE_port_vessel_ratio_l3407_340779

theorem port_vessel_ratio :
  ∀ (cargo sailboats fishing : ℕ),
    cargo + 4 + sailboats + fishing = 28 →
    sailboats = cargo + 6 →
    sailboats = 7 * fishing →
    cargo = 2 * 4 :=
by sorry

end NUMINAMATH_CALUDE_port_vessel_ratio_l3407_340779


namespace NUMINAMATH_CALUDE_line_slope_proof_l3407_340748

/-- Given a line passing through points P(-2, m) and Q(m, 4) with slope 1, prove that m = 1 -/
theorem line_slope_proof (m : ℝ) : 
  (4 - m) / (m - (-2)) = 1 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_proof_l3407_340748


namespace NUMINAMATH_CALUDE_orange_marbles_count_l3407_340701

/-- The number of orange marbles in a jar, given the total number of marbles,
    the number of red marbles, and that half of the marbles are blue. -/
def orangeMarbles (total : ℕ) (red : ℕ) (halfAreBlue : Bool) : ℕ :=
  total - (total / 2 + red)

/-- Theorem stating that there are 6 orange marbles in a jar with 24 total marbles,
    6 red marbles, and half of the marbles being blue. -/
theorem orange_marbles_count :
  orangeMarbles 24 6 true = 6 := by
  sorry

end NUMINAMATH_CALUDE_orange_marbles_count_l3407_340701


namespace NUMINAMATH_CALUDE_log_function_fixed_point_l3407_340703

-- Define the set of valid 'a' values
def ValidA := {a : ℝ | a ∈ Set.Ioo 0 1 ∪ Set.Ioi 1}

-- Define the logarithm function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a + 2

-- Theorem statement
theorem log_function_fixed_point (a : ℝ) (ha : a ∈ ValidA) :
  f a 1 = 2 :=
sorry

end NUMINAMATH_CALUDE_log_function_fixed_point_l3407_340703


namespace NUMINAMATH_CALUDE_special_triangle_perimeter_l3407_340710

/-- A triangle with integer side lengths satisfying specific conditions -/
structure SpecialTriangle where
  x : ℕ
  y : ℕ
  side_product : x * y = 105
  triangle_inequality : x + y > 13 ∧ x + 13 > y ∧ y + 13 > x

/-- The perimeter of the special triangle is 35 -/
theorem special_triangle_perimeter (t : SpecialTriangle) : 13 + t.x + t.y = 35 := by
  sorry

#check special_triangle_perimeter

end NUMINAMATH_CALUDE_special_triangle_perimeter_l3407_340710


namespace NUMINAMATH_CALUDE_problem_solution_l3407_340793

theorem problem_solution (A B : ℝ) 
  (h1 : 100 * A = 35^2 - 15^2) 
  (h2 : (A - 1)^6 = 27^B) : 
  A = 10 ∧ B = 4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3407_340793


namespace NUMINAMATH_CALUDE_unique_prime_f_l3407_340733

/-- The polynomial function f(n) = n^3 - 7n^2 + 18n - 10 -/
def f (n : ℕ) : ℤ := n^3 - 7*n^2 + 18*n - 10

/-- Theorem stating that there exists exactly one positive integer n such that f(n) is prime -/
theorem unique_prime_f : ∃! (n : ℕ+), Nat.Prime (Int.natAbs (f n)) := by sorry

end NUMINAMATH_CALUDE_unique_prime_f_l3407_340733


namespace NUMINAMATH_CALUDE_matrix_not_invertible_l3407_340707

def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2 + x, 9],
    ![4 - x, 10]]

theorem matrix_not_invertible (x : ℝ) :
  ¬(IsUnit (A x).det) ↔ x = 16 / 19 := by
  sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_l3407_340707


namespace NUMINAMATH_CALUDE_equal_division_theorem_l3407_340771

theorem equal_division_theorem (total : ℕ) (people : ℕ) (share : ℕ) : 
  total = 2400 → people = 4 → share * people = total → share = 600 := by
  sorry

end NUMINAMATH_CALUDE_equal_division_theorem_l3407_340771


namespace NUMINAMATH_CALUDE_digit_1234_is_4_l3407_340726

def decimal_sequence : ℕ → ℕ
  | 0 => 0  -- represents the decimal point
  | n+1 => 
    let k := (n-1) / 3 + 100
    if k ≤ 500 then
      match (n-1) % 3 with
      | 0 => k / 100
      | 1 => (k / 10) % 10
      | _ => k % 10
    else 0

theorem digit_1234_is_4 : decimal_sequence 1234 = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_1234_is_4_l3407_340726


namespace NUMINAMATH_CALUDE_sixteen_four_eight_calculation_l3407_340713

theorem sixteen_four_eight_calculation : (16^2 / 4^3) * 8^3 = 2048 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_four_eight_calculation_l3407_340713


namespace NUMINAMATH_CALUDE_tim_sleep_total_l3407_340739

theorem tim_sleep_total (sleep_first_two_days sleep_next_two_days : ℕ) 
  (h1 : sleep_first_two_days = 6 * 2)
  (h2 : sleep_next_two_days = 10 * 2) :
  sleep_first_two_days + sleep_next_two_days = 32 := by
  sorry

end NUMINAMATH_CALUDE_tim_sleep_total_l3407_340739


namespace NUMINAMATH_CALUDE_binomial_150_150_l3407_340759

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_l3407_340759


namespace NUMINAMATH_CALUDE_negation_of_universal_is_existential_l3407_340736

def A : Set ℤ := {x | ∃ k, x = 2*k + 1}
def B : Set ℤ := {x | ∃ k, x = 2*k}

theorem negation_of_universal_is_existential :
  ¬(∀ x ∈ A, 2*x ∈ B) ↔ ∃ x ∈ A, 2*x ∉ B :=
sorry

end NUMINAMATH_CALUDE_negation_of_universal_is_existential_l3407_340736


namespace NUMINAMATH_CALUDE_largest_number_l3407_340789

theorem largest_number : ∀ (a b c : ℝ), 
  a = -12.4 → b = -1.23 → c = -0.13 → 
  (0 ≥ a) ∧ (0 ≥ b) ∧ (0 ≥ c) ∧ (0 ≥ 0) :=
by
  sorry

#check largest_number

end NUMINAMATH_CALUDE_largest_number_l3407_340789


namespace NUMINAMATH_CALUDE_composite_power_sum_l3407_340737

theorem composite_power_sum (n : ℕ) (h : n % 6 = 4) : 3 ∣ (n^n + (n+1)^(n+1)) := by
  sorry

end NUMINAMATH_CALUDE_composite_power_sum_l3407_340737


namespace NUMINAMATH_CALUDE_max_value_2x_minus_y_l3407_340766

theorem max_value_2x_minus_y (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0) 
  (h2 : y + 1 ≥ 0) 
  (h3 : x + y + 1 ≤ 0) : 
  ∃ (max : ℝ), max = 1 ∧ ∀ z, 2*x - y ≤ z → z ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_2x_minus_y_l3407_340766


namespace NUMINAMATH_CALUDE_probability_four_white_balls_l3407_340768

/-- The probability of drawing 4 white balls from a box containing 7 white balls and 5 black balls -/
theorem probability_four_white_balls (white_balls black_balls drawn : ℕ) : 
  white_balls = 7 →
  black_balls = 5 →
  drawn = 4 →
  (Nat.choose white_balls drawn : ℚ) / (Nat.choose (white_balls + black_balls) drawn) = 7 / 99 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_white_balls_l3407_340768


namespace NUMINAMATH_CALUDE_divisible_by_33_pairs_count_l3407_340721

theorem divisible_by_33_pairs_count : 
  (Finset.filter (fun p : ℕ × ℕ => 
    1 ≤ p.1 ∧ p.1 ≤ p.2 ∧ p.2 ≤ 40 ∧ (p.1 * p.2) % 33 = 0) 
    (Finset.product (Finset.range 40) (Finset.range 41))).card = 64 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_33_pairs_count_l3407_340721


namespace NUMINAMATH_CALUDE_factorization_ax_squared_minus_a_l3407_340769

theorem factorization_ax_squared_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_ax_squared_minus_a_l3407_340769


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3407_340790

-- Define the given constants
def train_length : ℝ := 160
def bridge_length : ℝ := 215
def crossing_time : ℝ := 30

-- Define the speed conversion factor
def m_per_s_to_km_per_hr : ℝ := 3.6

-- Theorem statement
theorem train_speed_calculation :
  let total_distance := train_length + bridge_length
  let speed_m_per_s := total_distance / crossing_time
  let speed_km_per_hr := speed_m_per_s * m_per_s_to_km_per_hr
  speed_km_per_hr = 45 := by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l3407_340790


namespace NUMINAMATH_CALUDE_marly_soup_containers_l3407_340797

/-- The number of containers needed to store Marly's soup -/
def containers_needed (milk chicken_stock pureed_vegetables other_ingredients container_capacity : ℚ) : ℕ :=
  let total_soup := milk + chicken_stock + pureed_vegetables + other_ingredients
  (total_soup / container_capacity).ceil.toNat

/-- Proof that Marly needs 28 containers for his soup -/
theorem marly_soup_containers :
  containers_needed 15 (3 * 15) 5 4 (5/2) = 28 := by
  sorry

end NUMINAMATH_CALUDE_marly_soup_containers_l3407_340797


namespace NUMINAMATH_CALUDE_max_value_of_f_l3407_340746

def f (x : ℝ) : ℝ := -3 * (x - 2)^2 - 3

theorem max_value_of_f :
  ∃ (m : ℝ), m = -3 ∧ ∀ (x : ℝ), f x ≤ m := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3407_340746


namespace NUMINAMATH_CALUDE_train_length_l3407_340763

/-- Given a train traveling at 75 km/hour passing a 140-meter bridge in 24 seconds, 
    the length of the train is 360 meters. -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (passing_time : ℝ) :
  train_speed = 75 * (1000 / 3600) →
  bridge_length = 140 →
  passing_time = 24 →
  train_speed * passing_time - bridge_length = 360 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3407_340763


namespace NUMINAMATH_CALUDE_min_sum_of_product_2310_l3407_340749

theorem min_sum_of_product_2310 (a b c : ℕ+) (h : a * b * c = 2310) :
  ∃ (x y z : ℕ+), x * y * z = 2310 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 42 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_2310_l3407_340749


namespace NUMINAMATH_CALUDE_lincoln_county_houses_l3407_340745

/-- The total number of houses in Lincoln County after the housing boom -/
def total_houses (original : ℕ) (new_built : ℕ) : ℕ :=
  original + new_built

/-- Theorem: The total number of houses in Lincoln County after the housing boom is 118558 -/
theorem lincoln_county_houses :
  total_houses 20817 97741 = 118558 := by
  sorry

end NUMINAMATH_CALUDE_lincoln_county_houses_l3407_340745


namespace NUMINAMATH_CALUDE_teacher_instructions_l3407_340700

theorem teacher_instructions (x : ℤ) : 4 * (3 * (x + 3) - 2) = 4 * (3 * x + 7) := by
  sorry

end NUMINAMATH_CALUDE_teacher_instructions_l3407_340700


namespace NUMINAMATH_CALUDE_custom_mult_equation_solution_l3407_340776

-- Define the custom operation
def customMult (a b : ℝ) : ℝ := 4 * a * b

-- State the theorem
theorem custom_mult_equation_solution :
  ∀ x : ℝ, (customMult x x) + (customMult 2 x) - (customMult 2 4) = 0 → x = 2 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_equation_solution_l3407_340776


namespace NUMINAMATH_CALUDE_modified_system_solution_l3407_340782

theorem modified_system_solution 
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) 
  (h : a₁ * 8 + b₁ * 3 = c₁ ∧ a₂ * 8 + b₂ * 3 = c₂) :
  4 * a₁ * 10 + 3 * b₁ * 5 = 5 * c₁ ∧ 
  4 * a₂ * 10 + 3 * b₂ * 5 = 5 * c₂ :=
by sorry

end NUMINAMATH_CALUDE_modified_system_solution_l3407_340782


namespace NUMINAMATH_CALUDE_big_eighteen_games_l3407_340777

/-- Represents a basketball conference with the given structure -/
structure BasketballConference where
  num_divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculates the total number of conference games scheduled -/
def total_games (conf : BasketballConference) : Nat :=
  let total_teams := conf.num_divisions * conf.teams_per_division
  let games_per_team := (conf.teams_per_division - 1) * conf.intra_division_games + 
                        (total_teams - conf.teams_per_division) * conf.inter_division_games
  total_teams * games_per_team / 2

/-- The Big Eighteen Basketball Conference -/
def big_eighteen : BasketballConference :=
  { num_divisions := 3
  , teams_per_division := 6
  , intra_division_games := 3
  , inter_division_games := 1 }

theorem big_eighteen_games : total_games big_eighteen = 243 := by
  sorry

end NUMINAMATH_CALUDE_big_eighteen_games_l3407_340777


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l3407_340798

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + b = 2 * a * b) :
  ∀ x y, x > 0 → y > 0 → 3 * x + y = 2 * x * y → a + b ≤ x + y ∧ a + b = 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l3407_340798


namespace NUMINAMATH_CALUDE_complex_modulus_l3407_340741

theorem complex_modulus (z : ℂ) : z - Complex.I = 1 + Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3407_340741


namespace NUMINAMATH_CALUDE_yara_arrival_time_difference_l3407_340762

/-- Prove that Yara arrives 3 hours before Theon given their speeds and the destination distance -/
theorem yara_arrival_time_difference (theon_speed yara_speed destination : ℝ) 
  (h1 : theon_speed = 15)
  (h2 : yara_speed = 30)
  (h3 : destination = 90) : 
  destination / theon_speed - destination / yara_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_yara_arrival_time_difference_l3407_340762


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3407_340778

theorem solve_exponential_equation :
  ∃ n : ℕ, 16^n * 16^n * 16^n * 16^n = 256^4 ∧ n = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3407_340778


namespace NUMINAMATH_CALUDE_basketball_surface_area_l3407_340785

/-- The surface area of a sphere with circumference 30 inches is 900/π square inches. -/
theorem basketball_surface_area :
  let circumference : ℝ := 30
  let radius : ℝ := circumference / (2 * Real.pi)
  let surface_area : ℝ := 4 * Real.pi * radius^2
  surface_area = 900 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_basketball_surface_area_l3407_340785


namespace NUMINAMATH_CALUDE_square_difference_simplification_l3407_340786

theorem square_difference_simplification (a : ℝ) : (a + 1)^2 - a^2 = 2*a + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_simplification_l3407_340786


namespace NUMINAMATH_CALUDE_equation_solutions_l3407_340705

theorem equation_solutions :
  (∃ x : ℚ, 2 * x - 2 * (3 * x + 1) = 6 ∧ x = -2) ∧
  (∃ x : ℚ, (x + 1) / 2 - 1 = (2 - 3 * x) / 3 ∧ x = 7 / 9) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3407_340705


namespace NUMINAMATH_CALUDE_data_set_average_l3407_340720

theorem data_set_average (x : ℝ) : 
  (2 + 3 + 4 + x + 6) / 5 = 4 → x = 5 := by
sorry

end NUMINAMATH_CALUDE_data_set_average_l3407_340720


namespace NUMINAMATH_CALUDE_prob_at_least_one_of_three_l3407_340752

/-- The probability that at least one of three events occurs, given their individual probabilities -/
theorem prob_at_least_one_of_three (pA pB pC : ℝ) 
  (hA : 0 ≤ pA ∧ pA ≤ 1) 
  (hB : 0 ≤ pB ∧ pB ≤ 1) 
  (hC : 0 ≤ pC ∧ pC ≤ 1) 
  (h_pA : pA = 0.8) 
  (h_pB : pB = 0.6) 
  (h_pC : pC = 0.5) : 
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_of_three_l3407_340752


namespace NUMINAMATH_CALUDE_N_value_is_negative_twelve_point_five_l3407_340731

/-- Represents a grid with arithmetic sequences -/
structure ArithmeticGrid :=
  (row_first : ℚ)
  (col1_second : ℚ)
  (col1_third : ℚ)
  (col2_last : ℚ)
  (num_columns : ℕ)
  (num_rows : ℕ)

/-- Calculates the value of N in the arithmetic grid -/
def calculate_N (grid : ArithmeticGrid) : ℚ :=
  sorry

/-- Theorem stating that N equals -12.5 for the given grid -/
theorem N_value_is_negative_twelve_point_five :
  let grid : ArithmeticGrid := {
    row_first := 18,
    col1_second := 15,
    col1_third := 21,
    col2_last := -14,
    num_columns := 7,
    num_rows := 2
  }
  calculate_N grid = -12.5 := by sorry

end NUMINAMATH_CALUDE_N_value_is_negative_twelve_point_five_l3407_340731


namespace NUMINAMATH_CALUDE_parabola_c_value_l3407_340747

/-- A parabola that passes through the origin -/
def parabola_through_origin (c : ℝ) : ℝ → ℝ := λ x ↦ x^2 - 2*x + c - 4

/-- Theorem: For a parabola y = x^2 - 2x + c - 4 passing through the origin, c = 4 -/
theorem parabola_c_value :
  ∃ c : ℝ, (parabola_through_origin c 0 = 0) ∧ (c = 4) := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3407_340747


namespace NUMINAMATH_CALUDE_cantor_bernstein_l3407_340702

theorem cantor_bernstein {α β : Type*} (f : α → β) (g : β → α) 
  (hf : Function.Injective f) (hg : Function.Injective g) : 
  Nonempty (α ≃ β) :=
sorry

end NUMINAMATH_CALUDE_cantor_bernstein_l3407_340702


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l3407_340714

/-- Given line L1: 4x + 5y = 10 and line L2 perpendicular to L1 with y-intercept -3,
    prove that the x-intercept of L2 is 12/5 -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 4 * x + 5 * y = 10
  let L2 : ℝ → ℝ → Prop := λ x y ↦ ∃ m : ℝ, y = m * x - 3 ∧ m * (-4/5) = -1
  ∃ x : ℝ, L2 x 0 ∧ x = 12/5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l3407_340714


namespace NUMINAMATH_CALUDE_range_of_a_l3407_340750

/-- Proposition p: The equation a²x² + ax - 2 = 0 has a solution in the interval [-1, 1] -/
def p (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^2 * x^2 + a * x - 2 = 0

/-- Proposition q: There is only one real number x that satisfies x² + 2ax + 2a ≤ 0 -/
def q (a : ℝ) : Prop :=
  ∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0

/-- If both p and q are false, then -1 < a < 0 or 0 < a < 1 -/
theorem range_of_a (a : ℝ) : ¬(p a) ∧ ¬(q a) → (-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3407_340750


namespace NUMINAMATH_CALUDE_stating_tree_structure_equation_l3407_340711

/-- Represents a tree structure with a trunk, branches, and small branches. -/
structure TreeStructure where
  x : ℕ  -- number of branches grown by the trunk
  total : ℕ  -- total count of trunk, branches, and small branches
  h_total : total = x^2 + x + 1  -- relation between x and total

/-- 
Theorem stating that for a tree structure with 43 total elements,
the equation x^2 + x + 1 = 43 correctly represents the structure.
-/
theorem tree_structure_equation (t : TreeStructure) (h : t.total = 43) :
  t.x^2 + t.x + 1 = 43 := by
  sorry

end NUMINAMATH_CALUDE_stating_tree_structure_equation_l3407_340711


namespace NUMINAMATH_CALUDE_range_of_a_l3407_340760

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 6}

def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 3}

theorem range_of_a (a : ℝ) : (A ∪ B a = A) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3407_340760


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3407_340712

theorem inequality_solution_set (x : ℝ) :
  (Set.Icc (-2 : ℝ) 3 : Set ℝ) = {x | (x - 1)^2 * (x + 2) * (x - 3) ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3407_340712


namespace NUMINAMATH_CALUDE_a_100_equals_116_l3407_340730

/-- Sequence of positive integers not divisible by 7 -/
def a : ℕ → ℕ :=
  λ n => (n + (n - 1) / 6) + 1

theorem a_100_equals_116 : a 100 = 116 := by
  sorry

end NUMINAMATH_CALUDE_a_100_equals_116_l3407_340730


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l3407_340706

/-- Given an arithmetic progression where the sum of n terms is 5n + 4n^2,
    prove that the r-th term is 8r + 1 -/
theorem arithmetic_progression_rth_term (n : ℕ) (r : ℕ) :
  (∀ n, ∃ S : ℕ → ℕ, S n = 5*n + 4*n^2) →
  ∃ a : ℕ → ℕ, a r = 8*r + 1 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l3407_340706


namespace NUMINAMATH_CALUDE_construction_material_total_l3407_340719

theorem construction_material_total (gravel sand : ℝ) 
  (h1 : gravel = 5.91) (h2 : sand = 8.11) : 
  gravel + sand = 14.02 := by
  sorry

end NUMINAMATH_CALUDE_construction_material_total_l3407_340719


namespace NUMINAMATH_CALUDE_eight_balls_three_boxes_l3407_340718

def distribute_balls (n : ℕ) (k : ℕ) (min_first : ℕ) : ℕ :=
  -- Number of ways to distribute n indistinguishable balls into k distinguishable boxes,
  -- with the condition that the first box must contain at least min_first balls
  sorry

theorem eight_balls_three_boxes : distribute_balls 8 3 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_eight_balls_three_boxes_l3407_340718


namespace NUMINAMATH_CALUDE_mark_piggy_bank_problem_l3407_340744

/-- Given a total amount of money and a total number of bills (one and two dollar bills only),
    calculate the number of one dollar bills. -/
def one_dollar_bills (total_money : ℕ) (total_bills : ℕ) : ℕ :=
  total_bills - (total_money - total_bills)

/-- Theorem stating that given 87 dollars in total and 58 bills,
    the number of one dollar bills is 29. -/
theorem mark_piggy_bank_problem :
  one_dollar_bills 87 58 = 29 := by
  sorry

end NUMINAMATH_CALUDE_mark_piggy_bank_problem_l3407_340744


namespace NUMINAMATH_CALUDE_composite_s_l3407_340799

theorem composite_s (s : ℕ) (h1 : s ≥ 4) :
  (∃ a b c d : ℕ+, (a:ℕ) + b + c + d = s ∧ 
    (s ∣ a * b * c + a * b * d + a * c * d + b * c * d)) →
  ¬(Nat.Prime s) :=
by sorry

end NUMINAMATH_CALUDE_composite_s_l3407_340799


namespace NUMINAMATH_CALUDE_snail_noodles_problem_l3407_340754

/-- Snail noodles problem -/
theorem snail_noodles_problem 
  (price_A : ℝ) 
  (price_B : ℝ) 
  (quantity_A : ℝ) 
  (quantity_B : ℝ) 
  (h1 : price_A * quantity_A = 800)
  (h2 : price_B * quantity_B = 900)
  (h3 : price_B = 1.5 * price_A)
  (h4 : quantity_B = quantity_A - 2)
  (h5 : ∀ a : ℝ, 0 ≤ a ∧ a ≤ 15 → 
    90 * a + 135 * (30 - a) ≥ 90 * 15 + 135 * 15) :
  price_A = 100 ∧ price_B = 150 ∧ 
  (∃ (a : ℝ), 0 ≤ a ∧ a ≤ 15 ∧ 
    90 * a + 135 * (30 - a) = 3375 ∧
    ∀ (b : ℝ), 0 ≤ b ∧ b ≤ 15 → 
      90 * b + 135 * (30 - b) ≥ 3375) :=
sorry

end NUMINAMATH_CALUDE_snail_noodles_problem_l3407_340754


namespace NUMINAMATH_CALUDE_cubic_fraction_simplification_l3407_340755

theorem cubic_fraction_simplification (a b : ℝ) (h : a = 6 ∧ b = 6) :
  (a^3 + b^3) / (a^2 - a*b + b^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_simplification_l3407_340755


namespace NUMINAMATH_CALUDE_prime_4n_2n_1_implies_n_power_of_3_l3407_340727

-- Define a function to check if a number is prime
def isPrime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

-- Define a function to check if a number is a power of 3
def isPowerOf3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3^k

-- Theorem statement
theorem prime_4n_2n_1_implies_n_power_of_3 (n : ℕ) :
  n > 0 → isPrime (4^n + 2^n + 1) → isPowerOf3 n :=
by sorry

end NUMINAMATH_CALUDE_prime_4n_2n_1_implies_n_power_of_3_l3407_340727


namespace NUMINAMATH_CALUDE_function_non_negative_implies_a_value_l3407_340773

/-- Given a function f and a real number a, proves that if f satisfies certain conditions, then a = 2/3 -/
theorem function_non_negative_implies_a_value (a : ℝ) :
  (∀ x > 1 - 2*a, (Real.exp (x - a) - 1) * Real.log (x + 2*a - 1) ≥ 0) →
  a = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_function_non_negative_implies_a_value_l3407_340773


namespace NUMINAMATH_CALUDE_hulk_jump_theorem_l3407_340742

def jump_sequence : ℕ → ℕ
  | 0 => 2  -- We use 0-based indexing here
  | n + 1 => 2 * jump_sequence n + (n + 1)

theorem hulk_jump_theorem :
  (∀ k < 15, jump_sequence k ≤ 2000) ∧ jump_sequence 15 > 2000 := by
  sorry

#eval jump_sequence 15  -- To verify the result

end NUMINAMATH_CALUDE_hulk_jump_theorem_l3407_340742


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l3407_340724

/-- Represents the first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ := 3 + (n - 1) * n / 2

/-- Represents the last element of the nth set in the sequence -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- Represents the sum of elements in the nth set -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

theorem sum_of_15th_set : S 15 = 1725 := by sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l3407_340724


namespace NUMINAMATH_CALUDE_new_rectangle_area_l3407_340772

/-- Given a rectangle with sides a and b (a < b), prove that the area of a new rectangle
    with base (b + 2a) and height (b - a) is b^2 + ab - 2a^2 -/
theorem new_rectangle_area (a b : ℝ) (h : a < b) :
  (b + 2*a) * (b - a) = b^2 + a*b - 2*a^2 := by
  sorry

end NUMINAMATH_CALUDE_new_rectangle_area_l3407_340772


namespace NUMINAMATH_CALUDE_white_paint_calculation_l3407_340725

/-- Given the total amount of paint and the amounts of green and brown paint,
    calculate the amount of white paint needed. -/
theorem white_paint_calculation (total green brown : ℕ) (h1 : total = 69) 
    (h2 : green = 15) (h3 : brown = 34) : total - (green + brown) = 20 := by
  sorry

end NUMINAMATH_CALUDE_white_paint_calculation_l3407_340725


namespace NUMINAMATH_CALUDE_factorization_x_cubed_minus_9xy_squared_l3407_340784

theorem factorization_x_cubed_minus_9xy_squared (x y : ℝ) : 
  x^3 - 9*x*y^2 = x*(x+3*y)*(x-3*y) := by
sorry

end NUMINAMATH_CALUDE_factorization_x_cubed_minus_9xy_squared_l3407_340784


namespace NUMINAMATH_CALUDE_equation_equivalence_l3407_340796

theorem equation_equivalence (y : ℝ) (Q : ℝ) (h : 4 * (5 * y + 3 * Real.pi) = Q) :
  8 * (10 * y + 6 * Real.pi + 2 * Real.sqrt 3) = 4 * Q + 16 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3407_340796


namespace NUMINAMATH_CALUDE_min_value_expression_l3407_340765

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y = 4) :
  (x + 28 * y + 4) / (x * y) ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3407_340765


namespace NUMINAMATH_CALUDE_f_properties_l3407_340723

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.log x - a

theorem f_properties (a : ℝ) (h : a > 0) :
  -- Part 1
  (∃ (x : ℝ), x > 0 ∧ f a x = 0 ∧ ∀ (y : ℝ), y > 0 → f a y ≥ f a x) ∧
  (¬∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f a y ≤ f a x) ∧
  -- Part 2
  (∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 →
    1 / a < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ x₂ < a) ∧
  -- Part 3
  (∀ (x : ℝ), x > 0 → Real.exp (2 * x - 2) - Real.exp (x - 1) * Real.log x - x ≥ 0) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l3407_340723


namespace NUMINAMATH_CALUDE_complex_on_line_l3407_340708

/-- Given a complex number z = (2a-i)/i that corresponds to a point on the line x-y=0 in the complex plane, prove that a = 1/2 --/
theorem complex_on_line (a : ℝ) : 
  let z : ℂ := (2*a - Complex.I) / Complex.I
  (z.re - z.im = 0) → a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_complex_on_line_l3407_340708


namespace NUMINAMATH_CALUDE_inequality_proof_l3407_340795

theorem inequality_proof (a b : ℝ) (n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3407_340795


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l3407_340756

theorem power_fraction_simplification :
  (3^5 * 4^5) / 6^5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l3407_340756


namespace NUMINAMATH_CALUDE_truck_wheels_count_l3407_340722

/-- The toll formula for a truck crossing a bridge -/
def toll_formula (x : ℕ) : ℚ :=
  1.50 + 1.50 * (x - 2)

/-- The number of wheels on the front axle of the truck -/
def front_axle_wheels : ℕ := 2

/-- The number of wheels on each of the other axles of the truck -/
def other_axle_wheels : ℕ := 4

/-- Theorem stating that a truck with the given wheel configuration has 18 wheels in total -/
theorem truck_wheels_count :
  ∀ (x : ℕ), 
  x > 0 →
  toll_formula x = 6 →
  front_axle_wheels + (x - 1) * other_axle_wheels = 18 :=
by
  sorry


end NUMINAMATH_CALUDE_truck_wheels_count_l3407_340722


namespace NUMINAMATH_CALUDE_fourth_root_of_unity_l3407_340794

theorem fourth_root_of_unity : 
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 7 ∧ 
  (Complex.tan (π / 4) + Complex.I) / (Complex.tan (π / 4) - Complex.I) = 
  Complex.exp (Complex.I * (2 * n * π / 8)) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_unity_l3407_340794


namespace NUMINAMATH_CALUDE_triangle_properties_l3407_340729

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle --/
def TriangleConditions (t : Triangle) : Prop :=
  t.a * Real.cos t.B = 3 ∧
  t.b * Real.sin t.A = 4 ∧
  (1/2) * t.a * t.b * Real.sin t.C = 10

/-- Theorem stating the length of side a and the perimeter of the triangle --/
theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.a = 5 ∧ t.a + t.b + t.c = 10 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3407_340729


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3407_340715

theorem sum_of_three_numbers : ∀ (a b c : ℕ),
  b = 72 →
  a = 2 * b →
  c = a / 3 →
  a + b + c = 264 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3407_340715


namespace NUMINAMATH_CALUDE_sum_of_digits_of_1962_digit_number_divisible_by_9_l3407_340716

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A 1962-digit number is a natural number with 1962 digits -/
def is1962DigitNumber (n : ℕ) : Prop := sorry

theorem sum_of_digits_of_1962_digit_number_divisible_by_9 
  (N : ℕ) 
  (h1 : is1962DigitNumber N) 
  (h2 : N % 9 = 0) : 
  let a := sumOfDigits N
  let b := sumOfDigits a
  let c := sumOfDigits b
  c = 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_1962_digit_number_divisible_by_9_l3407_340716


namespace NUMINAMATH_CALUDE_expression_simplification_l3407_340753

theorem expression_simplification (b : ℝ) :
  ((2 * b + 6) - 5 * b) / 2 = -3/2 * b + 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3407_340753


namespace NUMINAMATH_CALUDE_similar_triangles_height_l3407_340781

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 → area_ratio = 9 →
  ∃ h_large : ℝ, h_large = h_small * Real.sqrt area_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l3407_340781


namespace NUMINAMATH_CALUDE_union_of_A_and_B_is_reals_l3407_340792

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 16 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

-- State the theorem
theorem union_of_A_and_B_is_reals : A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_is_reals_l3407_340792
