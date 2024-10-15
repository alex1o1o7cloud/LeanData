import Mathlib

namespace NUMINAMATH_CALUDE_optimal_method_is_random_then_stratified_l2499_249914

/-- Represents a sampling method -/
inductive SamplingMethod
  | Random
  | Stratified
  | RandomThenStratified
  | StratifiedThenRandom

/-- Represents a school with first-year classes -/
structure School where
  num_classes : Nat
  male_female_ratio : Real

/-- Represents the sampling scenario -/
structure SamplingScenario where
  school : School
  num_classes_to_sample : Nat

/-- Determines the optimal sampling method for a given scenario -/
def optimal_sampling_method (scenario : SamplingScenario) : SamplingMethod :=
  sorry

/-- Theorem stating that the optimal sampling method for the given scenario
    is to use random sampling first, then stratified sampling -/
theorem optimal_method_is_random_then_stratified 
  (scenario : SamplingScenario) 
  (h1 : scenario.school.num_classes = 16) 
  (h2 : scenario.num_classes_to_sample = 2) :
  optimal_sampling_method scenario = SamplingMethod.RandomThenStratified :=
sorry

end NUMINAMATH_CALUDE_optimal_method_is_random_then_stratified_l2499_249914


namespace NUMINAMATH_CALUDE_tyler_puppies_l2499_249998

/-- Given a person with a certain number of dogs, where each dog has a certain number of puppies,
    calculate the total number of puppies. -/
def total_puppies (num_dogs : ℕ) (puppies_per_dog : ℕ) : ℕ :=
  num_dogs * puppies_per_dog

/-- Theorem: A person with 15 dogs, where each dog has 5 puppies, has a total of 75 puppies. -/
theorem tyler_puppies : total_puppies 15 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_tyler_puppies_l2499_249998


namespace NUMINAMATH_CALUDE_min_value_quadratic_equation_l2499_249985

theorem min_value_quadratic_equation (a b : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + a*x + b - 3 = 0) →
  (∀ a' b' : ℝ, (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + a'*x + b' - 3 = 0) → 
    a^2 + (b - 4)^2 ≤ a'^2 + (b' - 4)^2) →
  a^2 + (b - 4)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_equation_l2499_249985


namespace NUMINAMATH_CALUDE_two_star_three_equals_one_l2499_249910

-- Define the ã — operation
def star_op (a b : ℤ) : ℤ := 2 * a - 3 * b + a * b

-- State the theorem
theorem two_star_three_equals_one :
  star_op 2 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_two_star_three_equals_one_l2499_249910


namespace NUMINAMATH_CALUDE_beef_weight_before_processing_l2499_249949

theorem beef_weight_before_processing 
  (initial_weight : ℝ) 
  (final_weight : ℝ) 
  (loss_percentage : ℝ) 
  (h1 : loss_percentage = 40) 
  (h2 : final_weight = 240) 
  (h3 : final_weight = initial_weight * (1 - loss_percentage / 100)) : 
  initial_weight = 400 := by
sorry

end NUMINAMATH_CALUDE_beef_weight_before_processing_l2499_249949


namespace NUMINAMATH_CALUDE_y_derivative_l2499_249926

noncomputable def y (x : ℝ) : ℝ := 
  (1/2) * Real.log ((1 + Real.cos x) / (1 - Real.cos x)) - 1 / Real.cos x - 1 / (3 * (Real.cos x)^3)

theorem y_derivative (x : ℝ) (h : Real.cos x ≠ 0) (h' : Real.sin x ≠ 0) : 
  deriv y x = -1 / (Real.sin x * (Real.cos x)^4) :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l2499_249926


namespace NUMINAMATH_CALUDE_max_value_of_trigonometric_expression_l2499_249960

theorem max_value_of_trigonometric_expression :
  let f : ℝ → ℝ := λ x => Real.sin (x + π/4) - Real.cos (x + π/3) + Real.sin (x + π/6)
  let domain : Set ℝ := {x | -π/4 ≤ x ∧ x ≤ 0}
  ∃ x ∈ domain, f x = 1 ∧ ∀ y ∈ domain, f y ≤ f x := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_trigonometric_expression_l2499_249960


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l2499_249986

theorem trig_expression_simplification (θ : ℝ) :
  (Real.tan (2 * Real.pi - θ) * Real.sin (-2 * Real.pi - θ) * Real.cos (6 * Real.pi - θ)) /
  (Real.cos (θ - Real.pi) * Real.sin (5 * Real.pi + θ)) = Real.tan θ :=
by sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l2499_249986


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2499_249911

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = 2 + Complex.I) : z = 1 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2499_249911


namespace NUMINAMATH_CALUDE_digit_swap_l2499_249984

theorem digit_swap (x : ℕ) (h : 9 < x ∧ x < 100) : 
  10 * (x % 10) + (x / 10) = 10 * (x % 10) + (x / 10) :=
by
  sorry

#check digit_swap

end NUMINAMATH_CALUDE_digit_swap_l2499_249984


namespace NUMINAMATH_CALUDE_f_has_three_roots_l2499_249918

def f (x : ℝ) := x^3 - 64*x

theorem f_has_three_roots : ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  ∀ x, f x = 0 → x = a ∨ x = b ∨ x = c := by
  sorry

end NUMINAMATH_CALUDE_f_has_three_roots_l2499_249918


namespace NUMINAMATH_CALUDE_sequence_properties_l2499_249951

def S (n : ℕ) : ℤ := 2 * n^2 - 10 * n

def a (n : ℕ) : ℤ := 4 * n - 5

theorem sequence_properties :
  (∀ n : ℕ, S (n + 1) - S n = a (n + 1)) ∧
  (∃ n : ℕ, ∀ m : ℕ, S m ≥ S n) ∧
  (∃ n : ℕ, S n = -12 ∧ ∀ m : ℕ, S m ≥ S n) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2499_249951


namespace NUMINAMATH_CALUDE_not_all_perfect_squares_l2499_249993

theorem not_all_perfect_squares (a b c : ℕ+) : 
  ¬(∃ (x y z : ℕ), x^2 = a^2 + b + c ∧ y^2 = b^2 + c + a ∧ z^2 = c^2 + a + b) := by
  sorry

end NUMINAMATH_CALUDE_not_all_perfect_squares_l2499_249993


namespace NUMINAMATH_CALUDE_smallest_base_for_27_l2499_249950

theorem smallest_base_for_27 : 
  ∃ (b : ℕ), b = 5 ∧ 
  (∀ (x : ℕ), x < b → ¬(x^2 ≤ 27 ∧ 27 < x^3)) ∧
  (b^2 ≤ 27 ∧ 27 < b^3) := by
sorry

end NUMINAMATH_CALUDE_smallest_base_for_27_l2499_249950


namespace NUMINAMATH_CALUDE_problem_solution_l2499_249971

theorem problem_solution (a b n : ℤ) : 
  a % 50 = 24 →
  b % 50 = 95 →
  150 ≤ n ∧ n ≤ 200 →
  (a - b) % 50 = n % 50 →
  n % 4 = 3 →
  n = 179 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2499_249971


namespace NUMINAMATH_CALUDE_women_decrease_l2499_249923

theorem women_decrease (initial_men : ℕ) (initial_women : ℕ) : 
  (initial_men : ℚ) / initial_women = 4 / 5 →
  initial_men + 2 = 14 →
  initial_women - 3 = 24 →
  initial_women - 24 = 3 := by
sorry

end NUMINAMATH_CALUDE_women_decrease_l2499_249923


namespace NUMINAMATH_CALUDE_crystal_beads_cost_l2499_249976

/-- The cost of one set of crystal beads -/
def crystal_cost : ℝ := sorry

/-- The cost of one set of metal beads -/
def metal_cost : ℝ := 10

/-- The number of crystal bead sets Nancy buys -/
def crystal_sets : ℕ := 1

/-- The number of metal bead sets Nancy buys -/
def metal_sets : ℕ := 2

/-- The total amount Nancy spends -/
def total_spent : ℝ := 29

theorem crystal_beads_cost :
  crystal_cost = 9 :=
by
  have h1 : crystal_cost + metal_cost * metal_sets = total_spent := sorry
  sorry

end NUMINAMATH_CALUDE_crystal_beads_cost_l2499_249976


namespace NUMINAMATH_CALUDE_factorial_calculation_l2499_249965

theorem factorial_calculation : (Nat.factorial 9 * Nat.factorial 5 * Nat.factorial 2) / (Nat.factorial 8 * Nat.factorial 6) = 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l2499_249965


namespace NUMINAMATH_CALUDE_congruence_solution_l2499_249924

theorem congruence_solution (n : ℤ) : 
  (15 * n) % 47 = 9 ↔ n % 47 = 18 := by sorry

end NUMINAMATH_CALUDE_congruence_solution_l2499_249924


namespace NUMINAMATH_CALUDE_imaginary_part_of_pure_imaginary_complex_l2499_249937

theorem imaginary_part_of_pure_imaginary_complex (a : ℝ) :
  let z : ℂ := (2 + a * Complex.I) / (3 - Complex.I)
  (∃ b : ℝ, z = b * Complex.I) → Complex.im z = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_pure_imaginary_complex_l2499_249937


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_47_plus_one_l2499_249992

theorem gcd_of_powers_of_47_plus_one (h : Nat.Prime 47) :
  Nat.gcd (47^6 + 1) (47^6 + 47^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_47_plus_one_l2499_249992


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2499_249956

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_abc : a + b + c = 3) : 
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2499_249956


namespace NUMINAMATH_CALUDE_defective_units_percentage_l2499_249997

/-- The percentage of defective units that are shipped for sale -/
def defective_shipped_percent : ℝ := 5

/-- The percentage of all units that are defective and shipped for sale -/
def total_defective_shipped_percent : ℝ := 0.4

/-- The percentage of all units that are defective -/
def defective_percent : ℝ := 8

theorem defective_units_percentage :
  defective_shipped_percent * defective_percent / 100 = total_defective_shipped_percent := by
  sorry

end NUMINAMATH_CALUDE_defective_units_percentage_l2499_249997


namespace NUMINAMATH_CALUDE_freshman_percentage_l2499_249916

-- Define the total number of students
variable (T : ℝ)
-- Define the fraction of freshmen (to be proven)
variable (F : ℝ)

-- Conditions from the problem
axiom liberal_arts : F * T * 0.5 = T * 0.1 / 0.5

-- Theorem to prove
theorem freshman_percentage : F = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_freshman_percentage_l2499_249916


namespace NUMINAMATH_CALUDE_perfect_square_discriminant_implies_rational_roots_rational_roots_implies_perfect_square_discriminant_all_odd_coefficients_no_rational_roots_l2499_249900

-- Define a structure for quadratic equations
structure QuadraticEquation where
  a : Int
  b : Int
  c : Int
  a_nonzero : a ≠ 0

-- Define the discriminant
def discriminant (eq : QuadraticEquation) : Int :=
  eq.b * eq.b - 4 * eq.a * eq.c

-- Define a perfect square
def is_perfect_square (n : Int) : Prop :=
  ∃ m : Int, n = m * m

-- Define rational roots
def has_rational_roots (eq : QuadraticEquation) : Prop :=
  ∃ p q : Int, q ≠ 0 ∧ eq.a * p * p + eq.b * p * q + eq.c * q * q = 0

-- Theorem 1
theorem perfect_square_discriminant_implies_rational_roots
  (eq : QuadraticEquation)
  (h : is_perfect_square (discriminant eq)) :
  has_rational_roots eq :=
sorry

-- Theorem 2
theorem rational_roots_implies_perfect_square_discriminant
  (eq : QuadraticEquation)
  (h : has_rational_roots eq) :
  is_perfect_square (discriminant eq) :=
sorry

-- Theorem 3
theorem all_odd_coefficients_no_rational_roots
  (eq : QuadraticEquation)
  (h1 : eq.a % 2 = 1)
  (h2 : eq.b % 2 = 1)
  (h3 : eq.c % 2 = 1) :
  ¬(has_rational_roots eq) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_discriminant_implies_rational_roots_rational_roots_implies_perfect_square_discriminant_all_odd_coefficients_no_rational_roots_l2499_249900


namespace NUMINAMATH_CALUDE_intersection_equals_nonnegative_reals_l2499_249917

-- Define set A
def A : Set ℝ := {x : ℝ | |x| = x}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 + x ≥ 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_nonnegative_reals :
  A_intersect_B = {x : ℝ | x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_intersection_equals_nonnegative_reals_l2499_249917


namespace NUMINAMATH_CALUDE_car_travel_time_l2499_249936

theorem car_travel_time (distance : ℝ) (speed : ℝ) (time_ratio : ℝ) (initial_time : ℝ) : 
  distance = 324 →
  speed = 36 →
  time_ratio = 3 / 2 →
  distance = speed * (time_ratio * initial_time) →
  initial_time = 6 := by
sorry

end NUMINAMATH_CALUDE_car_travel_time_l2499_249936


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l2499_249907

-- Define the function f
def f (a b x : ℝ) : ℝ := |x + a| + |2*x - b|

-- State the theorem
theorem min_value_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 1) 
  (hmin_exists : ∃ x, f a b x = 1) : 
  (2*a + b = 2) ∧ 
  (∀ t : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + 2*b ≥ t*a*b) → t ≤ 9/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l2499_249907


namespace NUMINAMATH_CALUDE_smallest_angle_in_special_right_triangle_l2499_249953

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The statement of the problem -/
theorem smallest_angle_in_special_right_triangle :
  ∀ a b : ℕ,
    a + b = 90 →
    a > b →
    isPrime a →
    isPrime b →
    isPrime (a - b) →
    b ≥ 17 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_special_right_triangle_l2499_249953


namespace NUMINAMATH_CALUDE_cube_equation_solution_l2499_249957

theorem cube_equation_solution :
  ∃! x : ℝ, (x - 5)^3 = (1/27)⁻¹ :=
by
  -- The unique solution is x = 8
  use 8
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l2499_249957


namespace NUMINAMATH_CALUDE_total_trophies_after_seven_years_l2499_249962

def michael_initial_trophies : ℕ := 100
def michael_yearly_increase : ℕ := 200
def years : ℕ := 7
def jack_multiplier : ℕ := 20

def michael_final_trophies : ℕ := michael_initial_trophies + michael_yearly_increase * years
def jack_final_trophies : ℕ := jack_multiplier * michael_initial_trophies + michael_final_trophies

theorem total_trophies_after_seven_years :
  michael_final_trophies + jack_final_trophies = 5000 := by
  sorry

end NUMINAMATH_CALUDE_total_trophies_after_seven_years_l2499_249962


namespace NUMINAMATH_CALUDE_apple_problem_l2499_249952

theorem apple_problem (x : ℚ) : 
  (((x / 2 + 10) * 2 / 3 + 2) / 2 + 1 = 12) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_apple_problem_l2499_249952


namespace NUMINAMATH_CALUDE_system_solution_l2499_249947

theorem system_solution : 
  ∃! (x y : ℝ), (3 * x + y = 2 ∧ 2 * x - 3 * y = 27) ∧ x = 3 ∧ y = -7 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2499_249947


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l2499_249929

theorem reciprocal_of_negative_two :
  ∀ x : ℚ, x * (-2) = 1 → x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l2499_249929


namespace NUMINAMATH_CALUDE_smallest_number_range_l2499_249909

theorem smallest_number_range (a b c d e : ℝ) 
  (distinct : a < b ∧ b < c ∧ c < d ∧ d < e)
  (sum1 : a + b = 20)
  (sum2 : a + c = 200)
  (sum3 : d + e = 2014)
  (sum4 : c + e = 2000) :
  -793 < a ∧ a < 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_range_l2499_249909


namespace NUMINAMATH_CALUDE_polynomial_irreducibility_equivalence_l2499_249954

theorem polynomial_irreducibility_equivalence 
  (f : Polynomial ℤ) : 
  Irreducible f ↔ Irreducible (f.map (algebraMap ℤ ℚ)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_irreducibility_equivalence_l2499_249954


namespace NUMINAMATH_CALUDE_rectangle_area_sum_l2499_249901

theorem rectangle_area_sum (a b : ℤ) (h1 : a > b) (h2 : b > 1) : 
  (2 * (a - b).natAbs * (a + b).natAbs = 50) → a + b = 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_sum_l2499_249901


namespace NUMINAMATH_CALUDE_modem_download_time_l2499_249974

theorem modem_download_time (time_a : ℝ) (speed_ratio : ℝ) (time_b : ℝ) : 
  time_a = 25.5 →
  speed_ratio = 0.17 →
  time_b = time_a / speed_ratio →
  time_b = 150 := by
sorry

end NUMINAMATH_CALUDE_modem_download_time_l2499_249974


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2499_249908

def A : Set ℝ := {x | x < -3}
def B : Set ℝ := {-5, -4, -3, 1}

theorem intersection_of_A_and_B : A ∩ B = {-5, -4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2499_249908


namespace NUMINAMATH_CALUDE_cubic_tangent_line_l2499_249961

/-- Given a cubic function f(x) = ax³ + bx + 1, if the tangent line
    at the point (1, f(1)) has the equation 4x - y - 1 = 0,
    then a + b = 2. -/
theorem cubic_tangent_line (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x + 1
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 + b
  (f' 1 = 4 ∧ f 1 = 3) → a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_tangent_line_l2499_249961


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l2499_249939

-- Define the function g
noncomputable def g : ℝ → ℤ
| x => if x > -3 then Int.ceil (1 / (x + 3))
       else if x < -3 then Int.floor (1 / (x + 3))
       else 0  -- This value doesn't matter as g is undefined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g : ∀ x : ℝ, x ≠ -3 → g x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l2499_249939


namespace NUMINAMATH_CALUDE_simplify_expression_l2499_249982

theorem simplify_expression : (5^8 + 3^7)*(0^5 - (-1)^5)^10 = 392812 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2499_249982


namespace NUMINAMATH_CALUDE_minimum_raft_capacity_l2499_249990

/-- Represents an animal with a specific weight -/
structure Animal where
  weight : ℕ

/-- Represents the raft with a weight capacity -/
structure Raft where
  capacity : ℕ

/-- Checks if a raft can carry at least two mice -/
def canCarryTwoMice (r : Raft) (mouseWeight : ℕ) : Prop :=
  r.capacity ≥ 2 * mouseWeight

/-- Checks if all animals can be transported given a raft capacity -/
def canTransportAll (r : Raft) (mice moles hamsters : List Animal) : Prop :=
  (mice ++ moles ++ hamsters).all (fun a => a.weight ≤ r.capacity)

theorem minimum_raft_capacity
  (mice : List Animal)
  (moles : List Animal)
  (hamsters : List Animal)
  (h_mice_count : mice.length = 5)
  (h_moles_count : moles.length = 3)
  (h_hamsters_count : hamsters.length = 4)
  (h_mice_weight : ∀ m ∈ mice, m.weight = 70)
  (h_moles_weight : ∀ m ∈ moles, m.weight = 90)
  (h_hamsters_weight : ∀ h ∈ hamsters, h.weight = 120)
  : ∃ (r : Raft), r.capacity = 140 ∧ 
    canCarryTwoMice r 70 ∧
    canTransportAll r mice moles hamsters :=
  sorry

#check minimum_raft_capacity

end NUMINAMATH_CALUDE_minimum_raft_capacity_l2499_249990


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2499_249967

theorem polynomial_factorization (a b m n : ℝ) 
  (h : |m - 4| + (n^2 - 8*n + 16) = 0) : 
  a^2 + 4*b^2 - m*a*b - n = (a - 2*b + 2) * (a - 2*b - 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2499_249967


namespace NUMINAMATH_CALUDE_max_sum_of_product_48_l2499_249903

theorem max_sum_of_product_48 :
  (∃ (x y : ℕ+), x * y = 48 ∧ x + y = 49) ∧
  (∀ (a b : ℕ+), a * b = 48 → a + b ≤ 49) := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_product_48_l2499_249903


namespace NUMINAMATH_CALUDE_three_lines_common_points_l2499_249913

/-- A line in 3D space --/
structure Line3D where
  -- Define a line using two points
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ
  ne : point1 ≠ point2

/-- The number of common points of three lines in 3D space --/
def commonPointCount (l1 l2 l3 : Line3D) : Nat :=
  sorry

/-- Three lines determine three planes --/
def determineThreePlanes (l1 l2 l3 : Line3D) : Prop :=
  sorry

theorem three_lines_common_points 
  (l1 l2 l3 : Line3D) 
  (h : determineThreePlanes l1 l2 l3) : 
  commonPointCount l1 l2 l3 = 0 ∨ commonPointCount l1 l2 l3 = 1 :=
sorry

end NUMINAMATH_CALUDE_three_lines_common_points_l2499_249913


namespace NUMINAMATH_CALUDE_ternary_10201_equals_100_l2499_249919

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

theorem ternary_10201_equals_100 :
  ternary_to_decimal [1, 0, 2, 0, 1] = 100 := by
  sorry

end NUMINAMATH_CALUDE_ternary_10201_equals_100_l2499_249919


namespace NUMINAMATH_CALUDE_sqrt_calculations_l2499_249999

theorem sqrt_calculations :
  (2 * Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27 = 3 * Real.sqrt 3) ∧
  ((Real.sqrt 18 - Real.sqrt 3) * Real.sqrt 12 = 6 * Real.sqrt 6 - 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l2499_249999


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2499_249921

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2499_249921


namespace NUMINAMATH_CALUDE_water_evaporation_rate_l2499_249996

/-- Proves that given a glass filled with 10 ounces of water, and 6% of the water
    evaporating over a 30-day period, the amount of water evaporated each day is 0.02 ounces. -/
theorem water_evaporation_rate (initial_water : ℝ) (days : ℕ) (evaporation_percentage : ℝ) :
  initial_water = 10 →
  days = 30 →
  evaporation_percentage = 6 →
  (initial_water * evaporation_percentage / 100) / days = 0.02 := by
  sorry


end NUMINAMATH_CALUDE_water_evaporation_rate_l2499_249996


namespace NUMINAMATH_CALUDE_partridge_family_allowance_l2499_249981

/-- The total weekly allowance for the Partridge family children -/
theorem partridge_family_allowance : 
  ∀ (younger_children older_children : ℕ) 
    (younger_allowance older_allowance : ℚ),
  younger_children = 3 →
  older_children = 2 →
  younger_allowance = 8 →
  older_allowance = 13 →
  (younger_children : ℚ) * younger_allowance + (older_children : ℚ) * older_allowance = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_partridge_family_allowance_l2499_249981


namespace NUMINAMATH_CALUDE_inverse_of_A_cubed_l2499_249987

theorem inverse_of_A_cubed (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A⁻¹ = ![![1, 4], ![-2, -7]] →
  (A^3)⁻¹ = ![![41, 140], ![-90, -335]] := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_cubed_l2499_249987


namespace NUMINAMATH_CALUDE_cone_base_radius_l2499_249995

/-- Given a cone whose lateral surface is a semicircle with radius 2,
    prove that the radius of the base of the cone is 1. -/
theorem cone_base_radius (r : ℝ) (h : r > 0) : r = 1 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l2499_249995


namespace NUMINAMATH_CALUDE_f_eq_g_l2499_249933

/-- The given polynomial function f(x, y, z) -/
def f (x y z : ℝ) : ℝ :=
  (y^2 - z^2) * (1 + x*y) * (1 + x*z) +
  (z^2 - x^2) * (1 + y*z) * (1 + x*y) +
  (x^2 - y^2) * (1 + y*z) * (1 + x*z)

/-- The factored form of the polynomial -/
def g (x y z : ℝ) : ℝ :=
  (y - z) * (z - x) * (x - y) * (x*y*z + x + y + z)

/-- Theorem stating that f and g are equivalent for all real x, y, and z -/
theorem f_eq_g : ∀ x y z : ℝ, f x y z = g x y z := by
  sorry

end NUMINAMATH_CALUDE_f_eq_g_l2499_249933


namespace NUMINAMATH_CALUDE_mabel_transactions_l2499_249972

theorem mabel_transactions : ∃ M : ℕ,
  let A := (11 * M) / 10  -- Anthony's transactions
  let C := (2 * A) / 3    -- Cal's transactions
  let J := C + 15         -- Jade's transactions
  J = 81 ∧ M = 90 := by
  sorry

end NUMINAMATH_CALUDE_mabel_transactions_l2499_249972


namespace NUMINAMATH_CALUDE_min_sum_squares_l2499_249925

def S : Set Int := {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_sum_squares (a b c d e f g h : Int) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
              b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
              c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
              d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
              e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
              f ≠ g ∧ f ≠ h ∧
              g ≠ h)
  (in_set : a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S)
  (sum_condition : e + f + g + h = 9) :
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 106 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2499_249925


namespace NUMINAMATH_CALUDE_marks_lost_per_incorrect_sum_l2499_249943

/-- Given Sandy's quiz results, prove the number of marks lost per incorrect sum --/
theorem marks_lost_per_incorrect_sum :
  ∀ (marks_per_correct : ℕ) 
    (total_attempts : ℕ) 
    (total_marks : ℕ) 
    (correct_sums : ℕ) 
    (marks_lost_per_incorrect : ℕ),
  marks_per_correct = 3 →
  total_attempts = 30 →
  total_marks = 60 →
  correct_sums = 24 →
  marks_lost_per_incorrect * (total_attempts - correct_sums) = 
    marks_per_correct * correct_sums - total_marks →
  marks_lost_per_incorrect = 2 :=
by sorry

end NUMINAMATH_CALUDE_marks_lost_per_incorrect_sum_l2499_249943


namespace NUMINAMATH_CALUDE_largest_product_in_S_largest_product_is_attained_l2499_249942

def S : Set Int := {-8, -3, 0, 2, 4}

theorem largest_product_in_S (a b : Int) : 
  a ∈ S → b ∈ S → a * b ≤ 24 := by sorry

theorem largest_product_is_attained : 
  ∃ (a b : Int), a ∈ S ∧ b ∈ S ∧ a * b = 24 := by sorry

end NUMINAMATH_CALUDE_largest_product_in_S_largest_product_is_attained_l2499_249942


namespace NUMINAMATH_CALUDE_fourth_root_64_times_cube_root_27_times_sqrt_9_l2499_249964

theorem fourth_root_64_times_cube_root_27_times_sqrt_9 :
  (64 : ℝ) ^ (1/4) * (27 : ℝ) ^ (1/3) * (9 : ℝ) ^ (1/2) = 18 * (2 : ℝ) ^ (1/2) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_64_times_cube_root_27_times_sqrt_9_l2499_249964


namespace NUMINAMATH_CALUDE_second_price_increase_l2499_249935

/-- Given an initial price increase of 20% followed by a second price increase,
    if the total price increase is 38%, then the second price increase is 15%. -/
theorem second_price_increase (P : ℝ) (x : ℝ) 
  (h1 : P > 0)
  (h2 : 1.20 * P * (1 + x / 100) = 1.38 * P) : 
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_second_price_increase_l2499_249935


namespace NUMINAMATH_CALUDE_jimmy_change_l2499_249904

def pen_cost : ℕ := 1
def notebook_cost : ℕ := 3
def folder_cost : ℕ := 5
def num_pens : ℕ := 3
def num_notebooks : ℕ := 4
def num_folders : ℕ := 2
def paid_amount : ℕ := 50

def total_cost : ℕ := 
  num_pens * pen_cost + num_notebooks * notebook_cost + num_folders * folder_cost

theorem jimmy_change : paid_amount - total_cost = 25 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_change_l2499_249904


namespace NUMINAMATH_CALUDE_remainder_x13_plus_1_div_x_minus_1_l2499_249955

theorem remainder_x13_plus_1_div_x_minus_1 (x : ℝ) : (x^13 + 1) % (x - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_x13_plus_1_div_x_minus_1_l2499_249955


namespace NUMINAMATH_CALUDE_same_solution_c_value_l2499_249922

theorem same_solution_c_value (x : ℚ) (c : ℚ) : 
  (3 * x + 5 = 1) ∧ (c * x + 8 = 6) → c = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_c_value_l2499_249922


namespace NUMINAMATH_CALUDE_jenny_bus_time_l2499_249945

/-- Represents the schedule of Jenny's day --/
structure Schedule where
  wakeUpTime : Nat  -- in minutes after midnight
  busToSchoolTime : Nat  -- in minutes after midnight
  numClasses : Nat
  classDuration : Nat  -- in minutes
  lunchDuration : Nat  -- in minutes
  extracurricularDuration : Nat  -- in minutes
  busHomeTime : Nat  -- in minutes after midnight

/-- Calculates the time Jenny spent on the bus given her schedule --/
def timeBusSpent (s : Schedule) : Nat :=
  (s.busHomeTime - s.busToSchoolTime) - 
  (s.numClasses * s.classDuration + s.lunchDuration + s.extracurricularDuration)

/-- Jenny's actual schedule --/
def jennySchedule : Schedule :=
  { wakeUpTime := 7 * 60
    busToSchoolTime := 8 * 60
    numClasses := 5
    classDuration := 45
    lunchDuration := 45
    extracurricularDuration := 90
    busHomeTime := 17 * 60 }

theorem jenny_bus_time : timeBusSpent jennySchedule = 180 := by
  sorry

end NUMINAMATH_CALUDE_jenny_bus_time_l2499_249945


namespace NUMINAMATH_CALUDE_line_symmetry_l2499_249978

/-- Given two lines in the form y = mx + b, this function checks if they are symmetrical about the x-axis -/
def symmetrical_about_x_axis (m1 b1 m2 b2 : ℝ) : Prop :=
  m1 = -m2 ∧ b1 = -b2

/-- The original line y = 3x - 4 -/
def original_line (x : ℝ) : ℝ := 3 * x - 4

/-- The proposed symmetrical line y = -3x + 4 -/
def symmetrical_line (x : ℝ) : ℝ := -3 * x + 4

theorem line_symmetry :
  symmetrical_about_x_axis 3 (-4) (-3) 4 :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l2499_249978


namespace NUMINAMATH_CALUDE_unit_digit_4137_pow_1289_l2499_249973

/-- The unit digit of a number -/
def unitDigit (n : ℕ) : ℕ := n % 10

/-- The unit digit pattern for powers of 7 repeats every 4 steps -/
def unitDigitPattern : Fin 4 → ℕ
  | 0 => 7
  | 1 => 9
  | 2 => 3
  | 3 => 1

theorem unit_digit_4137_pow_1289 :
  unitDigit ((4137 : ℕ) ^ 1289) = 7 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_4137_pow_1289_l2499_249973


namespace NUMINAMATH_CALUDE_prob_miss_at_least_once_prob_A_twice_B_once_l2499_249930

-- Define the probabilities of hitting the target
def prob_hit_A : ℚ := 2/3
def prob_hit_B : ℚ := 3/4

-- Define the number of shots for each part
def shots_part1 : ℕ := 3
def shots_part2 : ℕ := 2

-- Assume independence of shots
axiom independence : ∀ (n : ℕ), (prob_hit_A ^ n) = prob_hit_A * (prob_hit_A ^ (n - 1))

-- Part 1: Probability that Person A misses at least once in 3 shots
theorem prob_miss_at_least_once : 
  1 - (prob_hit_A ^ shots_part1) = 19/27 := by sorry

-- Part 2: Probability that A hits exactly twice and B hits exactly once in 2 shots each
theorem prob_A_twice_B_once :
  (prob_hit_A ^ 2) * (2 * prob_hit_B * (1 - prob_hit_B)) = 1/6 := by sorry

end NUMINAMATH_CALUDE_prob_miss_at_least_once_prob_A_twice_B_once_l2499_249930


namespace NUMINAMATH_CALUDE_no_real_solutions_l2499_249969

theorem no_real_solutions : ¬∃ x : ℝ, Real.sqrt (x + 7) - Real.sqrt (x - 5) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2499_249969


namespace NUMINAMATH_CALUDE_abc_sum_problem_l2499_249927

theorem abc_sum_problem (A B C : ℕ) : 
  A ≠ B → A ≠ C → B ≠ C →
  A < 10 → B < 10 → C < 10 →
  100 * A + 10 * B + C + 10 * A + B + A = C →
  C = 1 := by sorry

end NUMINAMATH_CALUDE_abc_sum_problem_l2499_249927


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2499_249906

/-- Two vectors in ℝ² -/
def a : ℝ × ℝ := (6, 2)
def b (k : ℝ) : ℝ × ℝ := (-3, k)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Theorem: If vectors a and b(k) are perpendicular, then k = 9 -/
theorem perpendicular_vectors (k : ℝ) : 
  dot_product a (b k) = 0 → k = 9 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2499_249906


namespace NUMINAMATH_CALUDE_equation_solution_l2499_249975

theorem equation_solution : ∃ r : ℚ, 23 - 5 = 3 * r + 2 ∧ r = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2499_249975


namespace NUMINAMATH_CALUDE_men_per_table_l2499_249928

theorem men_per_table 
  (num_tables : ℕ) 
  (women_per_table : ℕ) 
  (total_customers : ℕ) 
  (h1 : num_tables = 9) 
  (h2 : women_per_table = 7) 
  (h3 : total_customers = 90) : 
  (total_customers - num_tables * women_per_table) / num_tables = 3 := by
sorry

end NUMINAMATH_CALUDE_men_per_table_l2499_249928


namespace NUMINAMATH_CALUDE_optimal_start_time_maximizes_minimum_attention_l2499_249983

/-- Represents the attention index of students during a class -/
noncomputable def attentionIndex (x : ℝ) : ℝ :=
  if x ≤ 8 then 2 * x + 68
  else -1/8 * x^2 + 4 * x + 60

/-- The duration of the class in minutes -/
def classDuration : ℝ := 45

/-- The duration of the key explanation in minutes -/
def keyExplanationDuration : ℝ := 24

/-- The optimal start time for the key explanation -/
def optimalStartTime : ℝ := 4

theorem optimal_start_time_maximizes_minimum_attention :
  ∀ t : ℝ, 0 ≤ t ∧ t + keyExplanationDuration ≤ classDuration →
    (∀ x : ℝ, t ≤ x ∧ x ≤ t + keyExplanationDuration →
      attentionIndex x ≥ min (attentionIndex t) (attentionIndex (t + keyExplanationDuration))) →
    t = optimalStartTime := by sorry


end NUMINAMATH_CALUDE_optimal_start_time_maximizes_minimum_attention_l2499_249983


namespace NUMINAMATH_CALUDE_chair_carrying_trips_l2499_249966

/-- Proves that given 5 students, each carrying 5 chairs per trip, and a total of 250 chairs moved, the number of trips each student made is 10 -/
theorem chair_carrying_trips 
  (num_students : ℕ) 
  (chairs_per_trip : ℕ) 
  (total_chairs : ℕ) 
  (h1 : num_students = 5)
  (h2 : chairs_per_trip = 5)
  (h3 : total_chairs = 250) :
  (total_chairs / (num_students * chairs_per_trip) : ℕ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_chair_carrying_trips_l2499_249966


namespace NUMINAMATH_CALUDE_coin_problem_l2499_249934

def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def half_dollar_value : ℕ := 50

def total_coins : ℕ := 13
def total_value : ℕ := 163

theorem coin_problem (pennies nickels dimes quarters half_dollars : ℕ) 
  (h1 : pennies + nickels + dimes + quarters + half_dollars = total_coins)
  (h2 : pennies * penny_value + nickels * nickel_value + dimes * dime_value + 
        quarters * quarter_value + half_dollars * half_dollar_value = total_value)
  (h3 : pennies ≥ 1)
  (h4 : nickels ≥ 1)
  (h5 : dimes ≥ 1)
  (h6 : quarters ≥ 1)
  (h7 : half_dollars ≥ 1) :
  dimes = 3 := by
sorry

end NUMINAMATH_CALUDE_coin_problem_l2499_249934


namespace NUMINAMATH_CALUDE_relative_relationship_value_example_max_sum_given_relative_relationship_value_max_sum_achievable_l2499_249958

-- Define the relative relationship value
def relative_relationship_value (a b n : ℚ) : ℚ :=
  |a - n| + |b - n|

-- Theorem 1
theorem relative_relationship_value_example : 
  relative_relationship_value 2 (-5) 2 = 7 := by sorry

-- Theorem 2
theorem max_sum_given_relative_relationship_value :
  ∀ m n : ℚ, relative_relationship_value m n 2 = 2 → 
  m + n ≤ 6 := by sorry

-- Theorem to show that 6 is indeed achievable
theorem max_sum_achievable :
  ∃ m n : ℚ, relative_relationship_value m n 2 = 2 ∧ m + n = 6 := by sorry

end NUMINAMATH_CALUDE_relative_relationship_value_example_max_sum_given_relative_relationship_value_max_sum_achievable_l2499_249958


namespace NUMINAMATH_CALUDE_repeated_roots_coincide_l2499_249932

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic polynomial at a point x -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- A quadratic polynomial has a repeated root -/
def has_repeated_root (p : QuadraticPolynomial) : Prop :=
  ∃ r : ℝ, p.eval r = 0 ∧ (∀ x : ℝ, p.eval x = p.a * (x - r)^2)

/-- The sum of two quadratic polynomials -/
def add_poly (p q : QuadraticPolynomial) : QuadraticPolynomial :=
  ⟨p.a + q.a, p.b + q.b, p.c + q.c⟩

/-- Theorem: If P and Q are quadratic polynomials with repeated roots, 
    and P + Q also has a repeated root, then all these roots are equal -/
theorem repeated_roots_coincide (P Q : QuadraticPolynomial) 
  (hP : has_repeated_root P) 
  (hQ : has_repeated_root Q) 
  (hPQ : has_repeated_root (add_poly P Q)) : 
  ∃ r : ℝ, (∀ x : ℝ, P.eval x = P.a * (x - r)^2) ∧ 
            (∀ x : ℝ, Q.eval x = Q.a * (x - r)^2) ∧ 
            (∀ x : ℝ, (add_poly P Q).eval x = (P.a + Q.a) * (x - r)^2) := by
  sorry


end NUMINAMATH_CALUDE_repeated_roots_coincide_l2499_249932


namespace NUMINAMATH_CALUDE_final_water_level_approx_34cm_l2499_249905

/-- Represents the properties of a liquid in a cylindrical vessel -/
structure Liquid where
  density : ℝ
  initial_height : ℝ

/-- Represents a system of two connected cylindrical vessels with different liquids -/
structure ConnectedVessels where
  water : Liquid
  oil : Liquid

/-- Calculates the final water level in the first vessel after opening the valve -/
def final_water_level (vessels : ConnectedVessels) : ℝ :=
  sorry

/-- The theorem states that given the initial conditions, the final water level
    will be approximately 34 cm -/
theorem final_water_level_approx_34cm (vessels : ConnectedVessels)
  (h_water_density : vessels.water.density = 1000)
  (h_oil_density : vessels.oil.density = 700)
  (h_initial_height : vessels.water.initial_height = 40 ∧ vessels.oil.initial_height = 40) :
  ∃ ε > 0, |final_water_level vessels - 34| < ε :=
sorry

end NUMINAMATH_CALUDE_final_water_level_approx_34cm_l2499_249905


namespace NUMINAMATH_CALUDE_fifth_score_calculation_l2499_249944

theorem fifth_score_calculation (s1 s2 s3 s4 : ℕ) (avg : ℚ) (h1 : s1 = 65) (h2 : s2 = 67) (h3 : s3 = 76) (h4 : s4 = 82) (h5 : avg = 75) :
  ∃ (s5 : ℕ), (s1 + s2 + s3 + s4 + s5) / 5 = avg ∧ s5 = 85 := by
  sorry

end NUMINAMATH_CALUDE_fifth_score_calculation_l2499_249944


namespace NUMINAMATH_CALUDE_flagpole_height_l2499_249979

/-- Represents the height and shadow length of an object -/
structure Object where
  height : ℝ
  shadowLength : ℝ

/-- Given two objects under similar conditions, their height-to-shadow ratios are equal -/
def similarConditions (obj1 obj2 : Object) : Prop :=
  obj1.height / obj1.shadowLength = obj2.height / obj2.shadowLength

theorem flagpole_height
  (flagpole : Object)
  (building : Object)
  (h_flagpole_shadow : flagpole.shadowLength = 45)
  (h_building_height : building.height = 24)
  (h_building_shadow : building.shadowLength = 60)
  (h_similar : similarConditions flagpole building) :
  flagpole.height = 18 := by
  sorry


end NUMINAMATH_CALUDE_flagpole_height_l2499_249979


namespace NUMINAMATH_CALUDE_ralphs_cards_l2499_249902

/-- 
Given that Ralph initially collected some cards and his father gave him additional cards,
this theorem proves the total number of cards Ralph has.
-/
theorem ralphs_cards (initial_cards additional_cards : ℕ) 
  (h1 : initial_cards = 4)
  (h2 : additional_cards = 8) : 
  initial_cards + additional_cards = 12 := by
  sorry

end NUMINAMATH_CALUDE_ralphs_cards_l2499_249902


namespace NUMINAMATH_CALUDE_tangent_line_at_one_max_value_condition_a_range_condition_l2499_249912

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x

-- Define e as the base of natural logarithms
def e : ℝ := Real.exp 1

-- Theorem 1: Tangent line equation when a = 1
theorem tangent_line_at_one (x y : ℝ) :
  f 1 1 = 1 → 2 * x - y - 1 = 0 ↔ y - 1 = 2 * (x - 1) :=
sorry

-- Theorem 2: Value of a when maximum of f(x) is -2
theorem max_value_condition (a : ℝ) :
  (∃ x > 0, ∀ y > 0, f a x ≥ f a y) ∧ (∃ x > 0, f a x = -2) → a = -e :=
sorry

-- Theorem 3: Range of a when a < 0 and f(x) ≤ g(x) for x ∈ [1,e]
theorem a_range_condition (a : ℝ) :
  a < 0 ∧ (∀ x ∈ Set.Icc 1 e, f a x ≤ g a x) →
  a ∈ Set.Icc ((1 - 2*e) / (e^2 - e)) 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_max_value_condition_a_range_condition_l2499_249912


namespace NUMINAMATH_CALUDE_product_bound_l2499_249940

theorem product_bound (m : ℕ) (a : ℕ → ℕ) (h1 : ∀ i, i ∈ Finset.range m → a i > 0)
  (h2 : ∀ i, i ∈ Finset.range m → a i ≠ 10)
  (h3 : (Finset.range m).sum a = 10 * m) :
  ((Finset.range m).prod a) ^ (1 / m : ℝ) ≤ 3 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_product_bound_l2499_249940


namespace NUMINAMATH_CALUDE_video_card_upgrade_multiple_l2499_249991

theorem video_card_upgrade_multiple (computer_cost monitor_peripheral_ratio base_video_card_cost total_spent : ℚ) :
  computer_cost = 1500 →
  monitor_peripheral_ratio = 1/5 →
  base_video_card_cost = 300 →
  total_spent = 2100 →
  let monitor_peripheral_cost := computer_cost * monitor_peripheral_ratio
  let total_without_upgrade := computer_cost + monitor_peripheral_cost
  let upgraded_video_card_cost := total_spent - total_without_upgrade
  upgraded_video_card_cost / base_video_card_cost = 1 := by
  sorry

end NUMINAMATH_CALUDE_video_card_upgrade_multiple_l2499_249991


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2499_249968

/-- The line l intersects with the circle C if the distance from the center of C to l is less than the radius of C. -/
theorem line_intersects_circle (m : ℝ) : 
  let l : Set (ℝ × ℝ) := {(x, y) | m * x - y + 1 = 0}
  let C : Set (ℝ × ℝ) := {(x, y) | x^2 + (y-1)^2 = 5}
  let center : ℝ × ℝ := (0, 1)
  let radius : ℝ := Real.sqrt 5
  let distance_to_line (p : ℝ × ℝ) : ℝ := 
    abs (m * p.1 - p.2 + 1) / Real.sqrt (m^2 + 1)
  distance_to_line center < radius → 
  ∃ p, p ∈ l ∧ p ∈ C := by
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2499_249968


namespace NUMINAMATH_CALUDE_line_symmetry_l2499_249989

/-- A line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Symmetry condition for two lines with respect to y = x -/
def symmetric_about_y_eq_x (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = 1 ∧ l1.intercept + l2.intercept = 0

theorem line_symmetry (a b : ℝ) :
  let l1 : Line := ⟨a, 2⟩
  let l2 : Line := ⟨3, -b⟩
  symmetric_about_y_eq_x l1 l2 → a = 1/3 ∧ b = 6 := by
  sorry

#check line_symmetry

end NUMINAMATH_CALUDE_line_symmetry_l2499_249989


namespace NUMINAMATH_CALUDE_ratio_in_specific_arithmetic_sequence_l2499_249946

-- Define an arithmetic sequence
def is_arithmetic_sequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

-- Define our specific sequence
def our_sequence (s : ℕ → ℝ) (a m b : ℝ) : Prop :=
  s 0 = a ∧ s 1 = m ∧ s 2 = b ∧ s 3 = 3*m

-- State the theorem
theorem ratio_in_specific_arithmetic_sequence (s : ℕ → ℝ) (a m b : ℝ) :
  is_arithmetic_sequence s → our_sequence s a m b → b / a = -2 :=
by sorry

end NUMINAMATH_CALUDE_ratio_in_specific_arithmetic_sequence_l2499_249946


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_implies_fourth_plus_inverse_fourth_l2499_249994

theorem square_plus_inverse_square_implies_fourth_plus_inverse_fourth (x : ℝ) (h : x ≠ 0) :
  x^2 + (1/x^2) = 2 → x^4 + (1/x^4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_implies_fourth_plus_inverse_fourth_l2499_249994


namespace NUMINAMATH_CALUDE_min_sum_squares_l2499_249920

theorem min_sum_squares (x y z : ℝ) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 8 → 
  ∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → 
  x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧ 
  ∃ p q r : ℝ, p^3 + q^3 + r^3 - 3*p*q*r = 8 ∧ p^2 + q^2 + r^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2499_249920


namespace NUMINAMATH_CALUDE_kyle_age_l2499_249963

/-- Given the relationships between Kyle, Julian, Frederick, and Tyson's ages, prove Kyle's age. -/
theorem kyle_age (tyson_age : ℕ) (kyle_julian : ℕ) (julian_frederick : ℕ) (frederick_tyson : ℕ) :
  tyson_age = 20 →
  kyle_julian = 5 →
  julian_frederick = 20 →
  frederick_tyson = 2 →
  tyson_age * frederick_tyson - julian_frederick + kyle_julian = 25 :=
by sorry

end NUMINAMATH_CALUDE_kyle_age_l2499_249963


namespace NUMINAMATH_CALUDE_two_pump_fill_time_l2499_249941

/-- The time taken for two pumps to fill a tank together -/
theorem two_pump_fill_time (small_pump_rate large_pump_rate : ℝ) 
  (h1 : small_pump_rate = 1 / 2)
  (h2 : large_pump_rate = 3)
  (h3 : small_pump_rate > 0)
  (h4 : large_pump_rate > 0) :
  1 / (small_pump_rate + large_pump_rate) = 1 / 3.5 :=
by sorry

end NUMINAMATH_CALUDE_two_pump_fill_time_l2499_249941


namespace NUMINAMATH_CALUDE_stream_current_speed_l2499_249948

/-- Represents the scenario of a rower traveling upstream and downstream -/
structure RowerScenario where
  distance : ℝ
  rower_speed : ℝ
  current_speed : ℝ
  time_diff : ℝ

/-- Represents the scenario when the rower increases their speed -/
structure IncreasedSpeedScenario extends RowerScenario where
  speed_increase : ℝ
  new_time_diff : ℝ

/-- The theorem stating the speed of the stream's current given the conditions -/
theorem stream_current_speed 
  (scenario : RowerScenario)
  (increased : IncreasedSpeedScenario)
  (h1 : scenario.distance = 18)
  (h2 : scenario.time_diff = 4)
  (h3 : increased.speed_increase = 0.5)
  (h4 : increased.new_time_diff = 2)
  (h5 : scenario.distance / (scenario.rower_speed + scenario.current_speed) + scenario.time_diff = 
        scenario.distance / (scenario.rower_speed - scenario.current_speed))
  (h6 : scenario.distance / ((1 + increased.speed_increase) * scenario.rower_speed + scenario.current_speed) + 
        increased.new_time_diff = 
        scenario.distance / ((1 + increased.speed_increase) * scenario.rower_speed - scenario.current_speed))
  : scenario.current_speed = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_current_speed_l2499_249948


namespace NUMINAMATH_CALUDE_correct_mark_calculation_l2499_249959

theorem correct_mark_calculation (n : ℕ) (initial_avg correct_avg wrong_mark : ℝ) :
  n = 25 →
  initial_avg = 100 →
  wrong_mark = 60 →
  correct_avg = 98 →
  (n : ℝ) * initial_avg - wrong_mark + (n : ℝ) * correct_avg - (n : ℝ) * initial_avg = 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_mark_calculation_l2499_249959


namespace NUMINAMATH_CALUDE_keith_grew_six_turnips_l2499_249970

/-- The number of turnips Alyssa grew -/
def alyssas_turnips : ℕ := 9

/-- The total number of turnips Keith and Alyssa grew together -/
def total_turnips : ℕ := 15

/-- The number of turnips Keith grew -/
def keiths_turnips : ℕ := total_turnips - alyssas_turnips

theorem keith_grew_six_turnips : keiths_turnips = 6 := by
  sorry

end NUMINAMATH_CALUDE_keith_grew_six_turnips_l2499_249970


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l2499_249938

theorem angle_sum_theorem (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  (1 + Real.tan α) * (1 + Real.tan β) = 2 →
  α + β = π/4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l2499_249938


namespace NUMINAMATH_CALUDE_inverse_g_inverse_g_14_l2499_249931

def g (x : ℝ) : ℝ := 5 * x - 3

theorem inverse_g_inverse_g_14 : 
  (Function.invFun g) ((Function.invFun g) 14) = 32 / 25 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_inverse_g_14_l2499_249931


namespace NUMINAMATH_CALUDE_max_value_tan_cos_l2499_249977

open Real

theorem max_value_tan_cos (θ : Real) (h : 0 < θ ∧ θ < π/2) :
  ∃ (max : Real), max = 2 * (Real.sqrt ((-9 + Real.sqrt 117) / 2))^3 / 
    Real.sqrt (1 - (Real.sqrt ((-9 + Real.sqrt 117) / 2))^2) ∧
  ∀ (x : Real), 0 < x ∧ x < π/2 → 
    tan (x/2) * (1 - cos x) ≤ max := by sorry

end NUMINAMATH_CALUDE_max_value_tan_cos_l2499_249977


namespace NUMINAMATH_CALUDE_four_digit_integers_with_4_or_5_l2499_249915

/-- The number of four-digit positive integers -/
def four_digit_count : ℕ := 9000

/-- The number of options for the first digit when excluding 4 and 5 -/
def first_digit_options : ℕ := 7

/-- The number of options for each of the other three digits when excluding 4 and 5 -/
def other_digit_options : ℕ := 8

/-- The count of four-digit numbers without a 4 or 5 -/
def numbers_without_4_or_5 : ℕ := first_digit_options * other_digit_options * other_digit_options * other_digit_options

/-- The count of four-digit positive integers with at least one digit that is a 4 or a 5 -/
def numbers_with_4_or_5 : ℕ := four_digit_count - numbers_without_4_or_5

theorem four_digit_integers_with_4_or_5 : numbers_with_4_or_5 = 5416 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_integers_with_4_or_5_l2499_249915


namespace NUMINAMATH_CALUDE_problem_statement_l2499_249988

theorem problem_statement :
  (∃ x : ℝ, x^2 + 1 ≤ 2*x) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt ((x^2 + y^2)/2) ≥ (2*x*y)/(x + y)) ∧
  ¬(∀ x : ℝ, x ≠ 0 → x + 1/x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2499_249988


namespace NUMINAMATH_CALUDE_at_least_one_female_selection_l2499_249980

-- Define the total number of athletes
def total_athletes : ℕ := 10

-- Define the number of male athletes
def male_athletes : ℕ := 6

-- Define the number of female athletes
def female_athletes : ℕ := 4

-- Define the number of athletes to be selected
def selected_athletes : ℕ := 5

-- Theorem statement
theorem at_least_one_female_selection :
  (Nat.choose total_athletes selected_athletes) - (Nat.choose male_athletes selected_athletes) = 246 :=
sorry

end NUMINAMATH_CALUDE_at_least_one_female_selection_l2499_249980
