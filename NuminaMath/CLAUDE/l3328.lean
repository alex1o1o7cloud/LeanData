import Mathlib

namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3328_332833

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : x + y = 30) (h3 : x - y = 10) : 
  x = 8 → y = 25 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3328_332833


namespace NUMINAMATH_CALUDE_three_digit_ending_l3328_332803

theorem three_digit_ending (N : ℕ) (h1 : N > 0) (h2 : N % 1000 = N^2 % 1000) 
  (h3 : N % 1000 ≥ 100) : N % 1000 = 127 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_ending_l3328_332803


namespace NUMINAMATH_CALUDE_fishing_contest_result_l3328_332843

/-- The number of salmons Hazel caught -/
def hazel_catch : ℕ := 24

/-- The number of salmons Hazel's father caught -/
def father_catch : ℕ := 27

/-- The total number of salmons caught by Hazel and her father -/
def total_catch : ℕ := hazel_catch + father_catch

theorem fishing_contest_result : total_catch = 51 := by
  sorry

end NUMINAMATH_CALUDE_fishing_contest_result_l3328_332843


namespace NUMINAMATH_CALUDE_system_solution_l3328_332858

theorem system_solution :
  ∀ x y z t : ℝ,
  (x * y - t^2 = 9 ∧ x^2 + y^2 + z^2 = 18) →
  ((x = 3 ∧ y = 3 ∧ z = 0 ∧ t = 0) ∨ (x = -3 ∧ y = -3 ∧ z = 0 ∧ t = 0)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3328_332858


namespace NUMINAMATH_CALUDE_max_value_of_vector_expression_l3328_332871

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem max_value_of_vector_expression (a b c : V) 
  (ha : ‖a‖ = 2) (hb : ‖b‖ = 3) (hc : ‖c‖ = 4) :
  ‖a - 3 • b‖^2 + ‖b - 3 • c‖^2 + ‖c - 3 • a‖^2 + 2 * ‖a + b - c‖^2 ≤ 290 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_vector_expression_l3328_332871


namespace NUMINAMATH_CALUDE_sum_squared_equals_129_l3328_332880

theorem sum_squared_equals_129 (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 + a*b + b^2 = 25)
  (h2 : b^2 + b*c + c^2 = 49)
  (h3 : c^2 + c*a + a^2 = 64) :
  (a + b + c)^2 = 129 := by
  sorry

end NUMINAMATH_CALUDE_sum_squared_equals_129_l3328_332880


namespace NUMINAMATH_CALUDE_horizontal_distance_P_Q_l3328_332896

/-- The curve function -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

/-- Theorem stating the horizontal distance between P and Q -/
theorem horizontal_distance_P_Q : 
  ∀ (xp xq : ℝ), 
  f xp = 8 → 
  f xq = -1 → 
  (∀ x : ℝ, f x = -1 → |x - xp| ≥ |xq - xp|) → 
  |xq - xp| = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_horizontal_distance_P_Q_l3328_332896


namespace NUMINAMATH_CALUDE_product_mod_seven_l3328_332863

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l3328_332863


namespace NUMINAMATH_CALUDE_age_difference_l3328_332813

theorem age_difference : ∀ (a b : ℕ),
  (10 ≤ 10 * a + b) ∧ (10 * a + b < 100) ∧  -- Jack's age is two-digit
  (10 ≤ 10 * b + a) ∧ (10 * b + a < 100) ∧  -- Bill's age is two-digit
  (10 * a + b + 10 = 3 * (10 * b + a + 10))  -- In 10 years, Jack will be 3 times Bill's age
  → (10 * a + b) - (10 * b + a) = 54 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l3328_332813


namespace NUMINAMATH_CALUDE_total_pens_l3328_332879

def red_pens : ℕ := 65
def blue_pens : ℕ := 45
def black_pens : ℕ := 58

theorem total_pens : red_pens + blue_pens + black_pens = 168 := by
  sorry

end NUMINAMATH_CALUDE_total_pens_l3328_332879


namespace NUMINAMATH_CALUDE_unique_fixed_point_of_rotation_invariant_function_l3328_332817

/-- A function is rotation-invariant if rotating its graph by π/2 around the origin
    results in the same graph. -/
def RotationInvariant (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (-y) = x

theorem unique_fixed_point_of_rotation_invariant_function
  (f : ℝ → ℝ) (h : RotationInvariant f) :
  ∃! x, f x = x :=
sorry

end NUMINAMATH_CALUDE_unique_fixed_point_of_rotation_invariant_function_l3328_332817


namespace NUMINAMATH_CALUDE_C_power_50_l3328_332862

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 2; -8, -5]

theorem C_power_50 : C^50 = !![(-199 : ℤ), -100; 400, 199] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l3328_332862


namespace NUMINAMATH_CALUDE_projection_theorem_l3328_332831

def projection (v : ℝ × ℝ) : ℝ × ℝ := sorry

theorem projection_theorem :
  let p := projection
  p (1, -2) = (3/2, -3/2) →
  p (-4, 1) = (-5/2, 5/2) := by sorry

end NUMINAMATH_CALUDE_projection_theorem_l3328_332831


namespace NUMINAMATH_CALUDE_equidistant_function_c_squared_l3328_332869

/-- A complex function that is equidistant from its input and the origin -/
def EquidistantFunction (f : ℂ → ℂ) : Prop :=
  ∀ z : ℂ, Complex.abs (f z - z) = Complex.abs (f z)

theorem equidistant_function_c_squared
  (a c : ℝ)
  (f : ℂ → ℂ)
  (h1 : f = fun z ↦ (a + c * Complex.I) * z)
  (h2 : EquidistantFunction f)
  (h3 : Complex.abs (a + c * Complex.I) = 5) :
  c^2 = 24.75 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_function_c_squared_l3328_332869


namespace NUMINAMATH_CALUDE_salt_solution_replacement_l3328_332886

theorem salt_solution_replacement (original_salt_percentage : Real) 
  (replaced_fraction : Real) (final_salt_percentage : Real) 
  (replacing_salt_percentage : Real) : 
  original_salt_percentage = 13 →
  replaced_fraction = 1/4 →
  final_salt_percentage = 16 →
  (1 - replaced_fraction) * original_salt_percentage + 
    replaced_fraction * replacing_salt_percentage = final_salt_percentage →
  replacing_salt_percentage = 25 := by
sorry

end NUMINAMATH_CALUDE_salt_solution_replacement_l3328_332886


namespace NUMINAMATH_CALUDE_inequality_condition_l3328_332826

theorem inequality_condition (a b : ℝ) : 
  (a < b ∧ b < 0 → 1/a > 1/b) ∧ 
  ∃ a b : ℝ, 1/a > 1/b ∧ ¬(a < b ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l3328_332826


namespace NUMINAMATH_CALUDE_train_length_calculation_l3328_332878

theorem train_length_calculation (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : 
  jogger_speed = 9 →
  train_speed = 45 →
  initial_distance = 240 →
  passing_time = 37 →
  let relative_speed := (train_speed - jogger_speed) * (5 / 18)
  let distance_traveled := relative_speed * passing_time
  let train_length := distance_traveled - initial_distance
  train_length = 130 := by
sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3328_332878


namespace NUMINAMATH_CALUDE_cubic_roots_roots_product_l3328_332834

/-- Given a cubic equation x^3 - 7x^2 + 36 = 0 where the product of two of its roots is 18,
    prove that the roots are -2, 3, and 6. -/
theorem cubic_roots (x : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, x^3 - 7*x^2 + 36 = 0 ∧ 
   r₁ * r₂ = 18 ∧
   (x = r₁ ∨ x = r₂ ∨ x = r₃)) →
  (x = -2 ∨ x = 3 ∨ x = 6) :=
by sorry

/-- The product of all three roots of the cubic equation x^3 - 7x^2 + 36 = 0 is -36. -/
theorem roots_product (r₁ r₂ r₃ : ℝ) :
  r₁^3 - 7*r₁^2 + 36 = 0 ∧ 
  r₂^3 - 7*r₂^2 + 36 = 0 ∧ 
  r₃^3 - 7*r₃^2 + 36 = 0 →
  r₁ * r₂ * r₃ = -36 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_roots_product_l3328_332834


namespace NUMINAMATH_CALUDE_smallest_multiple_forty_satisfies_forty_is_smallest_l3328_332856

theorem smallest_multiple (y : ℕ) : y > 0 ∧ 800 ∣ (540 * y) → y ≥ 40 :=
sorry

theorem forty_satisfies : 800 ∣ (540 * 40) :=
sorry

theorem forty_is_smallest : ∀ y : ℕ, y > 0 ∧ 800 ∣ (540 * y) → y ≥ 40 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_forty_satisfies_forty_is_smallest_l3328_332856


namespace NUMINAMATH_CALUDE_green_ball_probability_l3328_332800

/-- Represents a container of balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The containers in the problem -/
def containerX : Container := ⟨3, 7⟩
def containerY : Container := ⟨7, 3⟩
def containerZ : Container := ⟨7, 3⟩

/-- The probability of selecting each container -/
def containerProb : ℚ := 1 / 3

/-- The probability of selecting a green ball -/
def greenBallProb : ℚ :=
  containerProb * greenProbability containerX +
  containerProb * greenProbability containerY +
  containerProb * greenProbability containerZ

theorem green_ball_probability :
  greenBallProb = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l3328_332800


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l3328_332874

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_implies_a_equals_one :
  ∀ a : ℝ, A a ⊆ B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l3328_332874


namespace NUMINAMATH_CALUDE_statements_are_equivalent_l3328_332809

-- Define propositions
variable (R : Prop) -- R represents "It rains"
variable (G : Prop) -- G represents "I go outside"

-- Define the original statement
def original_statement : Prop := ¬R → ¬G

-- Define the equivalent statement
def equivalent_statement : Prop := G → R

-- Theorem stating the logical equivalence
theorem statements_are_equivalent : original_statement R G ↔ equivalent_statement R G := by
  sorry

end NUMINAMATH_CALUDE_statements_are_equivalent_l3328_332809


namespace NUMINAMATH_CALUDE_total_assignment_plans_l3328_332857

def number_of_male_doctors : ℕ := 6
def number_of_female_doctors : ℕ := 4
def number_of_selected_male_doctors : ℕ := 3
def number_of_selected_female_doctors : ℕ := 2
def number_of_regions : ℕ := 5

def assignment_plans : ℕ := 12960

theorem total_assignment_plans :
  (number_of_male_doctors = 6) →
  (number_of_female_doctors = 4) →
  (number_of_selected_male_doctors = 3) →
  (number_of_selected_female_doctors = 2) →
  (number_of_regions = 5) →
  assignment_plans = 12960 :=
by sorry

end NUMINAMATH_CALUDE_total_assignment_plans_l3328_332857


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3328_332875

/-- Given two vectors a and b in ℝ², prove that if |a| = 3, |b| = 4, and the angle between them is 120°, then |a - b| = √13 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 3) 
  (h2 : ‖b‖ = 4) 
  (h3 : a.1 * b.1 + a.2 * b.2 = -6) : ‖a - b‖ = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3328_332875


namespace NUMINAMATH_CALUDE_polynomial_change_l3328_332840

/-- Given a polynomial f(x) = 2x^2 - 5 and a positive real number b,
    the change in the polynomial's value when x changes by ±b is 4bx ± 2b^2 -/
theorem polynomial_change (x b : ℝ) (h : b > 0) :
  let f : ℝ → ℝ := λ t ↦ 2 * t^2 - 5
  (f (x + b) - f x) = 4 * b * x + 2 * b^2 ∧
  (f (x - b) - f x) = -4 * b * x + 2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_change_l3328_332840


namespace NUMINAMATH_CALUDE_number_properties_l3328_332807

theorem number_properties (n : ℕ) (h : n > 0) :
  (∃ (factors : Set ℕ), Finite factors ∧ ∀ k ∈ factors, n % k = 0) ∧
  (∃ (multiples : Set ℕ), ¬Finite multiples ∧ ∀ m ∈ multiples, m % n = 0) ∧
  (∀ k : ℕ, k ∣ n → k ≥ 1) ∧
  (∀ k : ℕ, k ∣ n → k ≤ n) ∧
  (∀ m : ℕ, n ∣ m → m ≥ n) := by
sorry

end NUMINAMATH_CALUDE_number_properties_l3328_332807


namespace NUMINAMATH_CALUDE_last_locker_exists_l3328_332898

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Represents a corridor with a given number of lockers -/
def Corridor (n : Nat) := Fin n → LockerState

/-- Toggles the state of a locker -/
def toggleLocker (state : LockerState) : LockerState :=
  match state with
  | LockerState.Open => LockerState.Closed
  | LockerState.Closed => LockerState.Open

/-- Represents a single pass of toggling lockers with a given step size -/
def togglePass (c : Corridor 512) (step : Nat) : Corridor 512 :=
  sorry

/-- Represents the full toggling process until all lockers are open -/
def fullToggleProcess (c : Corridor 512) : Corridor 512 :=
  sorry

/-- Theorem stating that there exists a last locker to be opened -/
theorem last_locker_exists :
  ∃ (last : Fin 512), 
    ∀ (c : Corridor 512), 
      (fullToggleProcess c last = LockerState.Open) ∧ 
      (∀ (i : Fin 512), i.val > last.val → fullToggleProcess c i = LockerState.Open) :=
sorry

end NUMINAMATH_CALUDE_last_locker_exists_l3328_332898


namespace NUMINAMATH_CALUDE_part_one_solution_set_part_two_minimum_value_l3328_332895

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one_solution_set (x : ℝ) : 
  (f 1 x ≥ 4 - |x - 3|) ↔ (x ≤ 0 ∨ x ≥ 4) := by sorry

-- Part II
theorem part_two_minimum_value (a m n : ℝ) (h1 : m > 0) (h2 : n > 0) :
  (Set.Icc 0 2 = {x | f a x ≤ 1}) → 
  (1 / m + 1 / (2 * n) = a) → 
  (∀ k l, k > 0 → l > 0 → 1 / k + 1 / (2 * l) = a → m * n ≤ k * l) →
  m * n = 2 := by sorry

end NUMINAMATH_CALUDE_part_one_solution_set_part_two_minimum_value_l3328_332895


namespace NUMINAMATH_CALUDE_abc_sum_mod_five_l3328_332888

theorem abc_sum_mod_five (a b c : ℕ) : 
  0 < a ∧ a < 5 ∧ 
  0 < b ∧ b < 5 ∧ 
  0 < c ∧ c < 5 ∧ 
  (a * b * c) % 5 = 1 ∧ 
  (4 * c) % 5 = 3 ∧ 
  (3 * b) % 5 = (2 + b) % 5 → 
  (a + b + c) % 5 = 3 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_mod_five_l3328_332888


namespace NUMINAMATH_CALUDE_triangle_existence_condition_l3328_332815

/-- A triangle with given altitudes and median -/
structure Triangle where
  ma : ℝ  -- altitude to side a
  mb : ℝ  -- altitude to side b
  sc : ℝ  -- median to side c
  ma_pos : 0 < ma
  mb_pos : 0 < mb
  sc_pos : 0 < sc

/-- The existence condition for a triangle with given altitudes and median -/
def triangle_exists (t : Triangle) : Prop :=
  t.ma < 2 * t.sc ∧ t.mb < 2 * t.sc

/-- Theorem stating the necessary and sufficient condition for triangle existence -/
theorem triangle_existence_condition (t : Triangle) :
  ∃ (triangle : Triangle), triangle.ma = t.ma ∧ triangle.mb = t.mb ∧ triangle.sc = t.sc ↔ triangle_exists t :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_condition_l3328_332815


namespace NUMINAMATH_CALUDE_distinct_arrangements_apples_l3328_332859

def word_length : ℕ := 6
def repeated_letter_count : ℕ := 2
def single_letter_count : ℕ := 1
def number_of_single_letters : ℕ := 4

theorem distinct_arrangements_apples :
  (word_length.factorial) / (repeated_letter_count.factorial * (single_letter_count.factorial ^ number_of_single_letters)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_apples_l3328_332859


namespace NUMINAMATH_CALUDE_outfit_combinations_l3328_332806

def num_shirts : ℕ := 5
def num_pants : ℕ := 6
def num_hats : ℕ := 2

theorem outfit_combinations : num_shirts * num_pants * num_hats = 60 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l3328_332806


namespace NUMINAMATH_CALUDE_mosquito_blood_consumption_proof_l3328_332861

/-- The number of drops of blood per liter -/
def drops_per_liter : ℕ := 5000

/-- The number of liters of blood loss that leads to death -/
def lethal_blood_loss : ℕ := 3

/-- The number of mosquitoes that would cause death by feeding -/
def lethal_mosquito_count : ℕ := 750

/-- The number of drops of blood a single mosquito sucks in one feeding -/
def mosquito_blood_consumption : ℕ := 20

theorem mosquito_blood_consumption_proof :
  mosquito_blood_consumption = (drops_per_liter * lethal_blood_loss) / lethal_mosquito_count :=
by sorry

end NUMINAMATH_CALUDE_mosquito_blood_consumption_proof_l3328_332861


namespace NUMINAMATH_CALUDE_quadratic_properties_l3328_332854

/-- A quadratic function with the property that y > 0 for -2 < x < 3 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ∀ x : ℝ, -2 < x → x < 3 → 0 < a * x^2 + b * x + c

/-- The properties of the quadratic function that we want to prove -/
theorem quadratic_properties (f : QuadraticFunction) :
  f.b = -f.a ∧
  ∃ x₁ x₂ : ℝ, x₁ = -1/3 ∧ x₂ = 1/2 ∧
    f.c * x₁^2 - f.b * x₁ + f.a = 0 ∧
    f.c * x₂^2 - f.b * x₂ + f.a = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3328_332854


namespace NUMINAMATH_CALUDE_base16_A987B_bits_bits_count_A987B_l3328_332893

def base16_to_decimal (n : String) : ℕ :=
  -- Implementation details omitted
  sorry

theorem base16_A987B_bits : 
  let decimal := base16_to_decimal "A987B"
  2^19 ≤ decimal ∧ decimal < 2^20 := by
  sorry

theorem bits_count_A987B : 
  (Nat.log 2 (base16_to_decimal "A987B") + 1 : ℕ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_base16_A987B_bits_bits_count_A987B_l3328_332893


namespace NUMINAMATH_CALUDE_f_10_equals_222_l3328_332802

-- Define the function f
def f (x : ℝ) (y : ℝ) : ℝ := 2 * x^2 + y

-- State the theorem
theorem f_10_equals_222 (y : ℝ) (h : f 2 y = 30) : f 10 y = 222 := by
  sorry

end NUMINAMATH_CALUDE_f_10_equals_222_l3328_332802


namespace NUMINAMATH_CALUDE_sum_of_digits_of_square_of_ones_l3328_332804

/-- Given a natural number n, construct a number consisting of n ones -/
def numberWithOnes (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

theorem sum_of_digits_of_square_of_ones (n : ℕ) :
  sumOfDigits ((numberWithOnes n)^2) = n^2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_square_of_ones_l3328_332804


namespace NUMINAMATH_CALUDE_motel_billing_solution_l3328_332825

/-- Represents the motel's billing system -/
structure MotelBilling where
  flatFee : ℝ  -- Flat fee for the first night
  nightlyRate : ℝ  -- Fixed rate for subsequent nights

/-- Calculates the total cost for a stay -/
def totalCost (billing : MotelBilling) (nights : ℕ) : ℝ :=
  billing.flatFee + billing.nightlyRate * (nights - 1 : ℝ) -
    if nights > 4 then 25 else 0

/-- The motel billing system satisfies the given conditions -/
theorem motel_billing_solution :
  ∃ (billing : MotelBilling),
    totalCost billing 4 = 215 ∧
    totalCost billing 7 = 360 ∧
    billing.flatFee = 45 := by
  sorry


end NUMINAMATH_CALUDE_motel_billing_solution_l3328_332825


namespace NUMINAMATH_CALUDE_min_value_expression_l3328_332847

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (6 * a) / (b + 2 * c) + (6 * b) / (c + 2 * a) + (2 * c) / (a + 2 * b) + (6 * c) / (2 * a + b) ≥ 12 ∧
  ((6 * a) / (b + 2 * c) + (6 * b) / (c + 2 * a) + (2 * c) / (a + 2 * b) + (6 * c) / (2 * a + b) = 12 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3328_332847


namespace NUMINAMATH_CALUDE_distinguishable_arrangements_l3328_332822

def brown_tiles : ℕ := 1
def purple_tiles : ℕ := 1
def green_tiles : ℕ := 3
def yellow_tiles : ℕ := 2

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles

theorem distinguishable_arrangements :
  (Nat.factorial total_tiles) / 
  (Nat.factorial brown_tiles * Nat.factorial purple_tiles * 
   Nat.factorial green_tiles * Nat.factorial yellow_tiles) = 420 := by
  sorry

end NUMINAMATH_CALUDE_distinguishable_arrangements_l3328_332822


namespace NUMINAMATH_CALUDE_max_value_expression_l3328_332887

theorem max_value_expression (a b c d : ℝ) 
  (ha : -6.5 ≤ a ∧ a ≤ 6.5)
  (hb : -6.5 ≤ b ∧ b ≤ 6.5)
  (hc : -6.5 ≤ c ∧ c ≤ 6.5)
  (hd : -6.5 ≤ d ∧ d ≤ 6.5) :
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 182 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l3328_332887


namespace NUMINAMATH_CALUDE_largest_number_l3328_332889

theorem largest_number (a b c d e : ℝ) : 
  a = 17231 + 1 / 3251 →
  b = 17231 - 1 / 3251 →
  c = 17231 * (1 / 3251) →
  d = 17231 / (1 / 3251) →
  e = 17231.3251 →
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3328_332889


namespace NUMINAMATH_CALUDE_q_squared_minus_one_div_24_l3328_332839

/-- The largest prime number with 2023 digits -/
def q : ℕ := sorry

/-- q is prime -/
axiom q_prime : Nat.Prime q

/-- q has 2023 digits -/
axiom q_digits : q ≥ 10^2022 ∧ q < 10^2023

/-- q is the largest prime with 2023 digits -/
axiom q_largest : ∀ p, Nat.Prime p → (p ≥ 10^2022 ∧ p < 10^2023) → p ≤ q

/-- The theorem to be proved -/
theorem q_squared_minus_one_div_24 : 24 ∣ (q^2 - 1) := by sorry

end NUMINAMATH_CALUDE_q_squared_minus_one_div_24_l3328_332839


namespace NUMINAMATH_CALUDE_sqrt_9x_lt_3x_squared_iff_x_gt_1_l3328_332827

theorem sqrt_9x_lt_3x_squared_iff_x_gt_1 :
  ∀ x : ℝ, x > 0 → (Real.sqrt (9 * x) < 3 * x^2 ↔ x > 1) := by
sorry

end NUMINAMATH_CALUDE_sqrt_9x_lt_3x_squared_iff_x_gt_1_l3328_332827


namespace NUMINAMATH_CALUDE_f_has_zero_in_interval_l3328_332832

-- Define the function f(x) = x³ + 3x - 3
def f (x : ℝ) : ℝ := x^3 + 3*x - 3

-- Theorem statement
theorem f_has_zero_in_interval :
  ∃ c ∈ Set.Icc 0 1, f c = 0 :=
sorry

end NUMINAMATH_CALUDE_f_has_zero_in_interval_l3328_332832


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l3328_332814

/-- Represents a square with a given side length -/
structure Square (α : Type*) [LinearOrderedField α] where
  side_length : α

/-- Calculates the area of a square -/
def Square.area {α : Type*} [LinearOrderedField α] (s : Square α) : α :=
  s.side_length * s.side_length

/-- Represents the shaded regions in the square -/
structure ShadedRegions (α : Type*) [LinearOrderedField α] where
  small_square_side : α
  medium_square_side : α
  large_square_side : α

/-- Calculates the total shaded area -/
def ShadedRegions.total_area {α : Type*} [LinearOrderedField α] (sr : ShadedRegions α) : α :=
  sr.small_square_side * sr.small_square_side +
  (sr.medium_square_side * sr.medium_square_side - sr.small_square_side * sr.small_square_side) +
  (sr.large_square_side * sr.large_square_side - sr.medium_square_side * sr.medium_square_side)

/-- Theorem: The percentage of shaded area in square ABCD is (36/49) * 100 -/
theorem shaded_area_percentage
  (square : Square ℝ)
  (shaded : ShadedRegions ℝ)
  (h1 : square.side_length = 7)
  (h2 : shaded.small_square_side = 2)
  (h3 : shaded.medium_square_side = 4)
  (h4 : shaded.large_square_side = 6) :
  (shaded.total_area / square.area) * 100 = (36 / 49) * 100 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l3328_332814


namespace NUMINAMATH_CALUDE_tree_planting_l3328_332884

theorem tree_planting (path_length : ℕ) (tree_distance : ℕ) (total_trees : ℕ) : 
  path_length = 50 →
  tree_distance = 2 →
  total_trees = 2 * (path_length / tree_distance + 1) →
  total_trees = 52 := by
sorry

end NUMINAMATH_CALUDE_tree_planting_l3328_332884


namespace NUMINAMATH_CALUDE_class_transfer_equation_l3328_332830

theorem class_transfer_equation (x : ℕ) : 
  (∀ (total : ℕ), total = 98 → 
    (∀ (transfer : ℕ), transfer = 3 →
      (total - x) + transfer = x - transfer)) ↔ 
  (98 - x) + 3 = x - 3 :=
sorry

end NUMINAMATH_CALUDE_class_transfer_equation_l3328_332830


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_plus_five_l3328_332829

theorem half_abs_diff_squares_plus_five : 
  (|20^2 - 12^2| / 2 : ℝ) + 5 = 133 := by sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_plus_five_l3328_332829


namespace NUMINAMATH_CALUDE_prob_two_defective_out_of_three_l3328_332819

/-- The probability of selecting exactly 2 defective items out of 3 randomly chosen items
    from a set of 100 products containing 10 defective items. -/
theorem prob_two_defective_out_of_three (total_products : ℕ) (defective_items : ℕ) 
    (selected_items : ℕ) (h1 : total_products = 100) (h2 : defective_items = 10) 
    (h3 : selected_items = 3) :
  (Nat.choose defective_items 2 * Nat.choose (total_products - defective_items) 1) / 
  Nat.choose total_products selected_items = 27 / 1078 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_defective_out_of_three_l3328_332819


namespace NUMINAMATH_CALUDE_connor_date_cost_l3328_332877

/-- The cost of Connor's movie date -/
def movie_date_cost (ticket_price : ℚ) (ticket_quantity : ℕ) (combo_meal_price : ℚ) (candy_price : ℚ) (candy_quantity : ℕ) : ℚ :=
  ticket_price * ticket_quantity + combo_meal_price + candy_price * candy_quantity

/-- Theorem: Connor's movie date costs $36.00 -/
theorem connor_date_cost :
  movie_date_cost 10 2 11 (5/2) 2 = 36 :=
sorry

end NUMINAMATH_CALUDE_connor_date_cost_l3328_332877


namespace NUMINAMATH_CALUDE_system_solution_l3328_332848

theorem system_solution : ∃ (x y : ℝ), x = 4 ∧ y = 1 ∧ x - y = 3 ∧ 2*(x - y) = 6*y := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3328_332848


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3328_332864

theorem arithmetic_calculation : 8 / 2 - 3 + 2 * (4 - 3)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3328_332864


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l3328_332876

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_80_by_150_percent :
  80 * (1 + 150 / 100) = 200 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l3328_332876


namespace NUMINAMATH_CALUDE_scarlett_fruit_salad_berries_l3328_332885

/-- The weight of berries in Scarlett's fruit salad -/
def weight_of_berries (total_weight melon_weight : ℚ) : ℚ :=
  total_weight - melon_weight

/-- Proof that the weight of berries in Scarlett's fruit salad is 0.38 pounds -/
theorem scarlett_fruit_salad_berries :
  weight_of_berries (63/100) (1/4) = 38/100 := by
  sorry

end NUMINAMATH_CALUDE_scarlett_fruit_salad_berries_l3328_332885


namespace NUMINAMATH_CALUDE_milk_glass_density_ratio_l3328_332860

/-- Prove that the density of milk is 0.2 times the density of glass -/
theorem milk_glass_density_ratio 
  (m_CT : ℝ) -- mass of empty glass jar
  (m_M : ℝ)  -- mass of milk
  (V_CT : ℝ) -- volume of glass
  (V_M : ℝ)  -- volume of milk
  (h1 : m_CT + m_M = 3 * m_CT) -- mass of full jar is 3 times mass of empty jar
  (h2 : V_M = 10 * V_CT) -- volume of milk is 10 times volume of glass
  : m_M / V_M = 0.2 * (m_CT / V_CT) := by
  sorry

#check milk_glass_density_ratio

end NUMINAMATH_CALUDE_milk_glass_density_ratio_l3328_332860


namespace NUMINAMATH_CALUDE_marie_erasers_l3328_332870

theorem marie_erasers (initial : ℕ) (lost : ℕ) (final : ℕ) : 
  initial = 95 → lost = 42 → final = initial - lost → final = 53 := by sorry

end NUMINAMATH_CALUDE_marie_erasers_l3328_332870


namespace NUMINAMATH_CALUDE_mod_eight_power_difference_l3328_332824

theorem mod_eight_power_difference : (47^2023 - 22^2023) % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_mod_eight_power_difference_l3328_332824


namespace NUMINAMATH_CALUDE_subtract_from_21_to_get_8_l3328_332838

theorem subtract_from_21_to_get_8 : ∃ x : ℝ, 21 - x = 8 ∧ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_subtract_from_21_to_get_8_l3328_332838


namespace NUMINAMATH_CALUDE_cos_270_degrees_l3328_332810

theorem cos_270_degrees (h : ∀ θ, Real.cos (360 - θ) = Real.cos θ) : 
  Real.cos 270 = 0 := by
sorry

end NUMINAMATH_CALUDE_cos_270_degrees_l3328_332810


namespace NUMINAMATH_CALUDE_lcm_36_105_l3328_332811

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_105_l3328_332811


namespace NUMINAMATH_CALUDE_edwards_initial_spending_l3328_332866

/-- Given Edward's initial balance, additional spending, and final balance,
    prove the amount he spent initially. -/
theorem edwards_initial_spending
  (initial_balance : ℕ)
  (additional_spending : ℕ)
  (final_balance : ℕ)
  (h1 : initial_balance = 34)
  (h2 : additional_spending = 8)
  (h3 : final_balance = 17)
  : initial_balance - additional_spending - final_balance = 9 := by
  sorry

#check edwards_initial_spending

end NUMINAMATH_CALUDE_edwards_initial_spending_l3328_332866


namespace NUMINAMATH_CALUDE_tangent_property_reasoning_l3328_332835

-- Define the types of geometric objects
inductive GeometricObject
| Circle
| Line
| Sphere
| Plane

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical
| Transitive

-- Define the property of perpendicularity for 2D and 3D cases
def isPerpendicular (obj1 obj2 : GeometricObject) : Prop :=
  match obj1, obj2 with
  | GeometricObject.Line, GeometricObject.Line => true
  | GeometricObject.Line, GeometricObject.Plane => true
  | _, _ => false

-- Define the tangent property for 2D case
def tangentProperty2D (circle : GeometricObject) (tangentLine : GeometricObject) (centerToTangentLine : GeometricObject) : Prop :=
  circle = GeometricObject.Circle ∧
  tangentLine = GeometricObject.Line ∧
  centerToTangentLine = GeometricObject.Line ∧
  isPerpendicular tangentLine centerToTangentLine

-- Define the tangent property for 3D case
def tangentProperty3D (sphere : GeometricObject) (tangentPlane : GeometricObject) (centerToTangentLine : GeometricObject) : Prop :=
  sphere = GeometricObject.Sphere ∧
  tangentPlane = GeometricObject.Plane ∧
  centerToTangentLine = GeometricObject.Line ∧
  isPerpendicular tangentPlane centerToTangentLine

-- Theorem statement
theorem tangent_property_reasoning :
  (∃ (circle tangentLine centerToTangentLine : GeometricObject),
    tangentProperty2D circle tangentLine centerToTangentLine) →
  (∃ (sphere tangentPlane centerToTangentLine : GeometricObject),
    tangentProperty3D sphere tangentPlane centerToTangentLine) →
  (∀ (r : ReasoningType), r = ReasoningType.Analogical) :=
by sorry

end NUMINAMATH_CALUDE_tangent_property_reasoning_l3328_332835


namespace NUMINAMATH_CALUDE_division_problem_l3328_332853

theorem division_problem : (120 : ℚ) / ((6 : ℚ) / 2) = 40 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3328_332853


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l3328_332849

/-- Proves that in a rhombus with an area of 88 cm² and one diagonal of 11 cm, the length of the other diagonal is 16 cm. -/
theorem rhombus_diagonal_length (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h_area : area = 88) 
  (h_d1 : d1 = 11) 
  (h_rhombus_area : area = (d1 * d2) / 2) : d2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l3328_332849


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_11_l3328_332821

theorem least_three_digit_multiple_of_11 : ∃ (n : ℕ), n = 110 ∧ 
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 11 ∣ m → n ≤ m) ∧ 
  n ≥ 100 ∧ n < 1000 ∧ 11 ∣ n :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_11_l3328_332821


namespace NUMINAMATH_CALUDE_propositions_p_and_q_l3328_332841

theorem propositions_p_and_q : 
  (∃ a b : ℝ, a > b ∧ 1/a > 1/b) ∧ 
  (∀ x : ℝ, Real.sin x + Real.cos x < 3/2) := by
  sorry

end NUMINAMATH_CALUDE_propositions_p_and_q_l3328_332841


namespace NUMINAMATH_CALUDE_car_travel_distance_l3328_332872

def ring_travel (d1 d2 d4 total : ℕ) : Prop :=
  ∃ d3 : ℕ, d1 + d2 + d3 + d4 = total

theorem car_travel_distance :
  ∀ (d1 d2 d4 total : ℕ),
    d1 = 5 →
    d2 = 8 →
    d4 = 0 →
    total = 23 →
    ring_travel d1 d2 d4 total →
    ∃ d3 : ℕ, d3 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l3328_332872


namespace NUMINAMATH_CALUDE_quotient_problem_l3328_332899

theorem quotient_problem (L S Q : ℕ) : 
  L - S = 1365 → 
  L = 1620 → 
  L = S * Q + 15 → 
  Q = 6 := by sorry

end NUMINAMATH_CALUDE_quotient_problem_l3328_332899


namespace NUMINAMATH_CALUDE_intersection_on_y_axis_l3328_332894

/-- Given two lines l₁ and l₂, prove that if their intersection is on the y-axis, then C = -4 -/
theorem intersection_on_y_axis (A : ℝ) :
  ∃ (x y : ℝ),
    (A * x + 3 * y + C = 0) ∧
    (2 * x - 3 * y + 4 = 0) ∧
    (x = 0) →
    C = -4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_on_y_axis_l3328_332894


namespace NUMINAMATH_CALUDE_womens_average_age_l3328_332836

theorem womens_average_age (n : ℕ) (initial_avg : ℝ) :
  n = 8 ∧
  initial_avg > 0 ∧
  (n * initial_avg + 60) / n = initial_avg + 2 →
  60 / 2 = 30 := by
sorry

end NUMINAMATH_CALUDE_womens_average_age_l3328_332836


namespace NUMINAMATH_CALUDE_sum_exponents_15_factorial_l3328_332812

/-- The largest perfect square that divides n! -/
def largestPerfectSquareDivisor (n : ℕ) : ℕ := sorry

/-- The sum of the exponents of the prime factors of the square root of a number -/
def sumExponentsOfSquareRoot (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of the exponents of the prime factors of the square root
    of the largest perfect square that divides 15! is equal to 10 -/
theorem sum_exponents_15_factorial :
  sumExponentsOfSquareRoot (largestPerfectSquareDivisor 15) = 10 := by sorry

end NUMINAMATH_CALUDE_sum_exponents_15_factorial_l3328_332812


namespace NUMINAMATH_CALUDE_mineral_water_recycling_l3328_332816

/-- Calculates the total number of bottles that can be drunk given an initial number of bottles -/
def total_bottles_drunk (initial_bottles : ℕ) : ℕ :=
  sorry

/-- Calculates the initial number of bottles needed to drink a given total number of bottles -/
def initial_bottles_needed (total_drunk : ℕ) : ℕ :=
  sorry

theorem mineral_water_recycling :
  (total_bottles_drunk 1999 = 2665) ∧
  (initial_bottles_needed 3126 = 2345) :=
by sorry

end NUMINAMATH_CALUDE_mineral_water_recycling_l3328_332816


namespace NUMINAMATH_CALUDE_family_income_problem_l3328_332867

/-- Proves that in a family of 4 members with an average income of 10000,
    if three members earn 8000, 6000, and 11000 respectively,
    then the income of the fourth member is 15000. -/
theorem family_income_problem (family_size : ℕ) (average_income : ℕ) 
  (member1_income : ℕ) (member2_income : ℕ) (member3_income : ℕ) :
  family_size = 4 →
  average_income = 10000 →
  member1_income = 8000 →
  member2_income = 6000 →
  member3_income = 11000 →
  average_income * family_size - (member1_income + member2_income + member3_income) = 15000 :=
by sorry

end NUMINAMATH_CALUDE_family_income_problem_l3328_332867


namespace NUMINAMATH_CALUDE_statement_to_equation_l3328_332844

theorem statement_to_equation (a : ℝ) : 
  (3 * a + 5 = 4 * a) ↔ 
  (∃ x : ℝ, x = 3 * a + 5 ∧ x = 4 * a) :=
by sorry

end NUMINAMATH_CALUDE_statement_to_equation_l3328_332844


namespace NUMINAMATH_CALUDE_range_of_a_l3328_332892

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → |x + 1/x| > |a - 2| + 1) → 
  1 < a ∧ a < 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3328_332892


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3328_332855

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m > 24 ∧ ¬(m ∣ (n * (n + 1) * (n + 2) * (n + 3))) ∧
  ∀ k : ℕ, k ≤ 24 → (k ∣ (n * (n + 1) * (n + 2) * (n + 3))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3328_332855


namespace NUMINAMATH_CALUDE_expand_product_l3328_332845

theorem expand_product (x : ℝ) : 4 * (x + 3) * (2 * x + 7) = 8 * x^2 + 52 * x + 84 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3328_332845


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3328_332890

theorem problem_1 : (1/2)⁻¹ + (Real.sqrt 2)^2 - 4 * |-(1/2)| = 2 := by sorry

theorem problem_2 (a : ℝ) (h : a = 2) : 
  (1 + 4 / (a - 1)) / ((a^2 + 6*a + 9) / (a^2 - a)) = 2/5 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3328_332890


namespace NUMINAMATH_CALUDE_toothpicks_12th_stage_l3328_332842

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ := 3 * n

/-- Theorem: The 12th stage of the pattern contains 36 toothpicks -/
theorem toothpicks_12th_stage : toothpicks 12 = 36 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_12th_stage_l3328_332842


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3328_332891

theorem inequality_equivalence (x y : ℝ) :
  (2 * y - 3 * x < Real.sqrt (9 * x^2 + 16)) ↔
  ((y < 4 * x ∧ x ≥ 0) ∨ (y < -x ∧ x < 0)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3328_332891


namespace NUMINAMATH_CALUDE_solution_set_eq_open_interval_l3328_332881

-- Define the logarithm function with base 10
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the solution set
def solution_set : Set ℝ := {x | log10 (x - 1) < 2}

-- State the theorem
theorem solution_set_eq_open_interval :
  solution_set = Set.Ioo 1 101 := by sorry

end NUMINAMATH_CALUDE_solution_set_eq_open_interval_l3328_332881


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_l3328_332850

theorem mark_and_carolyn_money : 
  let mark_money : ℚ := 3/4
  let carolyn_money : ℚ := 3/10
  mark_money + carolyn_money = 21/20 := by sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_l3328_332850


namespace NUMINAMATH_CALUDE_min_value_abc_l3328_332801

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ 288 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' * b' * c' = 8 ∧
    (a' + 3 * b') * (b' + 3 * c') * (a' * c' + 2) = 288 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l3328_332801


namespace NUMINAMATH_CALUDE_total_cost_calculation_l3328_332818

def normal_pretzel_price : ℝ := 4
def discounted_pretzel_price : ℝ := 3.5
def normal_chip_price : ℝ := 7
def discounted_chip_price : ℝ := 6
def pretzel_discount_threshold : ℕ := 3
def chip_discount_threshold : ℕ := 2

def pretzel_packs_bought : ℕ := 3
def chip_packs_bought : ℕ := 4

def calculate_pretzel_cost (packs : ℕ) : ℝ :=
  if packs ≥ pretzel_discount_threshold then
    packs * discounted_pretzel_price
  else
    packs * normal_pretzel_price

def calculate_chip_cost (packs : ℕ) : ℝ :=
  if packs ≥ chip_discount_threshold then
    packs * discounted_chip_price
  else
    packs * normal_chip_price

theorem total_cost_calculation :
  calculate_pretzel_cost pretzel_packs_bought + calculate_chip_cost chip_packs_bought = 34.5 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l3328_332818


namespace NUMINAMATH_CALUDE_river_width_proof_l3328_332808

theorem river_width_proof (total_distance : ℝ) (prob_find : ℝ) (x : ℝ) : 
  total_distance = 500 →
  prob_find = 4/5 →
  x / total_distance = 1 - prob_find →
  x = 100 := by
sorry

end NUMINAMATH_CALUDE_river_width_proof_l3328_332808


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3328_332852

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

-- Define the point P
def point_P : ℝ × ℝ := (0, 2)

-- Define a line passing through point P
def line_through_P (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the chord length
def chord_length (k : ℝ) : ℝ := sorry

-- Define the arc ratio
def arc_ratio (k : ℝ) : ℝ := sorry

theorem circle_line_intersection :
  ∀ (k : ℝ),
  (chord_length k = 2 → (k = 0 ∨ k = 3/4)) ∧
  (arc_ratio k = 3/1 → (k = 1/3 ∨ k = -3)) :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3328_332852


namespace NUMINAMATH_CALUDE_volleyball_lineup_count_l3328_332805

/-- The number of ways to choose a starting lineup from a volleyball team. -/
def starting_lineup_count (total_players : ℕ) (lineup_size : ℕ) (captain_count : ℕ) : ℕ :=
  total_players * (Nat.choose (total_players - 1) (lineup_size - 1))

/-- Theorem: The number of ways to choose a starting lineup of 8 players
    (including one captain) from a team of 18 players is 350,064. -/
theorem volleyball_lineup_count :
  starting_lineup_count 18 8 1 = 350064 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_count_l3328_332805


namespace NUMINAMATH_CALUDE_calculate_savings_person_savings_l3328_332846

/-- Calculates a person's savings given their income sources and expenses --/
theorem calculate_savings (total_income : ℝ) 
  (source_a_percent source_b_percent source_c_percent : ℝ)
  (expense_a_percent expense_b_percent expense_c_percent : ℝ) : ℝ :=
  let source_a := source_a_percent * total_income
  let source_b := source_b_percent * total_income
  let source_c := source_c_percent * total_income
  let expense_a := expense_a_percent * source_a
  let expense_b := expense_b_percent * source_b
  let expense_c := expense_c_percent * source_c
  let total_expenses := expense_a + expense_b + expense_c
  total_income - total_expenses

/-- Proves that the person's savings is Rs. 19,005 given the specified conditions --/
theorem person_savings : 
  calculate_savings 21000 0.5 0.3 0.2 0.1 0.05 0.15 = 19005 := by
  sorry

end NUMINAMATH_CALUDE_calculate_savings_person_savings_l3328_332846


namespace NUMINAMATH_CALUDE_collinear_vectors_m_value_l3328_332851

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_m_value :
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (m, m - 2)
  collinear a b → m = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_value_l3328_332851


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3328_332882

/-- A quadratic function passing through points (0,2) and (1,0) -/
def quadratic_function (x b c : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_function_properties :
  ∃ (b c : ℝ),
    (quadratic_function 0 b c = 2) ∧
    (quadratic_function 1 b c = 0) ∧
    (b = -3) ∧
    (c = 2) ∧
    (∀ x, quadratic_function x b c = (x - 3/2)^2 - 1/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3328_332882


namespace NUMINAMATH_CALUDE_intersection_implies_a_values_l3328_332873

-- Define sets A and B
def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {a, a^2 - 1}

-- State the theorem
theorem intersection_implies_a_values (a : ℝ) :
  (A ∩ B a = {1}) → (a = 1 ∨ a = Real.sqrt 2 ∨ a = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_values_l3328_332873


namespace NUMINAMATH_CALUDE_four_parts_of_400_l3328_332868

theorem four_parts_of_400 (a b c d : ℝ) 
  (sum_eq_400 : a + b + c + d = 400)
  (parts_equal : a + 1 = b - 2 ∧ b - 2 = 3 * c ∧ 3 * c = d / 4)
  (positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  a = 62 ∧ b = 65 ∧ c = 21 ∧ d = 252 := by
sorry

end NUMINAMATH_CALUDE_four_parts_of_400_l3328_332868


namespace NUMINAMATH_CALUDE_y_values_from_x_equation_l3328_332823

theorem y_values_from_x_equation (x : ℝ) :
  x^2 + 5 * (x / (x - 3))^2 = 50 →
  ∃ y : ℝ, y = (x - 3)^2 * (x + 4) / (2*x - 5) ∧
    (∃ k : ℝ, (k = 5 + Real.sqrt 55 ∨ k = 5 - Real.sqrt 55 ∨
               k = 3 + 2 * Real.sqrt 6 ∨ k = 3 - 2 * Real.sqrt 6) ∧
              y = (k - 3)^2 * (k + 4) / (2*k - 5)) :=
by sorry

end NUMINAMATH_CALUDE_y_values_from_x_equation_l3328_332823


namespace NUMINAMATH_CALUDE_function_value_l3328_332897

/-- Given a function f where f(2x + 3) is defined and f(29) = 170,
    prove that f(2x + 3) = 170 for all x -/
theorem function_value (f : ℝ → ℝ) (h : f 29 = 170) : ∀ x, f (2 * x + 3) = 170 := by
  sorry

end NUMINAMATH_CALUDE_function_value_l3328_332897


namespace NUMINAMATH_CALUDE_sum_of_squares_l3328_332865

theorem sum_of_squares (x y z : ℤ) 
  (sum_eq : x + y + z = 3) 
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3328_332865


namespace NUMINAMATH_CALUDE_sum_of_integers_l3328_332820

theorem sum_of_integers (x y : ℕ+) (h1 : x.val - y.val = 8) (h2 : x.val * y.val = 56) :
  x.val + y.val = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3328_332820


namespace NUMINAMATH_CALUDE_inequality_implies_m_range_l3328_332837

theorem inequality_implies_m_range (m : ℝ) : 
  (∀ x > 0, (m * Real.exp x) / x ≥ 6 - 4 * x) → m ≥ 2 * Real.exp (-1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_m_range_l3328_332837


namespace NUMINAMATH_CALUDE_maximum_spent_l3328_332883

/-- Represents the denominations of money in fen -/
inductive Denomination
  | Yuan100
  | Yuan50
  | Yuan20
  | Yuan10
  | Yuan5
  | Yuan1
  | Jiao5
  | Jiao1
  | Fen5
  | Fen2
  | Fen1

/-- Converts a denomination to its value in fen -/
def denominationToFen (d : Denomination) : ℕ :=
  match d with
  | .Yuan100 => 10000
  | .Yuan50  => 5000
  | .Yuan20  => 2000
  | .Yuan10  => 1000
  | .Yuan5   => 500
  | .Yuan1   => 100
  | .Jiao5   => 50
  | .Jiao1   => 10
  | .Fen5    => 5
  | .Fen2    => 2
  | .Fen1    => 1

/-- Represents a set of banknotes or coins -/
structure Change where
  denominations : List Denomination
  distinct : denominations.Nodup

/-- The problem statement -/
theorem maximum_spent (initialAmount : ℕ) 
  (banknotes : Change) 
  (coins : Change) :
  (initialAmount = 10000) →
  (banknotes.denominations.length = 4) →
  (coins.denominations.length = 4) →
  (∀ d ∈ banknotes.denominations, denominationToFen d > 100) →
  (∀ d ∈ coins.denominations, denominationToFen d < 100) →
  ((banknotes.denominations.map denominationToFen).sum % 300 = 0) →
  ((coins.denominations.map denominationToFen).sum % 7 = 0) →
  (initialAmount - (banknotes.denominations.map denominationToFen).sum - 
   (coins.denominations.map denominationToFen).sum = 6337) :=
by sorry

end NUMINAMATH_CALUDE_maximum_spent_l3328_332883


namespace NUMINAMATH_CALUDE_jelly_cost_theorem_l3328_332828

/-- The cost of jelly for all sandwiches is $1.68 --/
theorem jelly_cost_theorem (N B J : ℕ) : 
  N > 1 → 
  B > 0 → 
  J > 0 → 
  N * (3 * B + 7 * J) = 336 → 
  (N * J * 7 : ℚ) / 100 = 1.68 := by
  sorry

end NUMINAMATH_CALUDE_jelly_cost_theorem_l3328_332828
