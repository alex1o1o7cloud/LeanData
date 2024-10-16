import Mathlib

namespace NUMINAMATH_CALUDE_f_derivative_at_one_l478_47844

-- Define the function f
def f (x : ℝ) (f'₁ : ℝ) : ℝ := x^2 + 2*x*f'₁ - 6

-- State the theorem
theorem f_derivative_at_one :
  ∃ f'₁ : ℝ, (∀ x, deriv (f · f'₁) x = 2*x + 2*f'₁) ∧ f'₁ = -2 :=
sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l478_47844


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l478_47853

def geometric_sequence (a : ℕ → ℝ) (x : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * (3 * x)

theorem fifth_term_of_sequence (a : ℕ → ℝ) (x : ℝ) :
  geometric_sequence a x →
  a 0 = 3 →
  a 1 = 9 * x →
  a 2 = 27 * x^2 →
  a 3 = 81 * x^3 →
  a 4 = 243 * x^4 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l478_47853


namespace NUMINAMATH_CALUDE_park_birds_difference_l478_47871

/-- The number of geese and ducks remaining at a park after some changes. -/
theorem park_birds_difference (initial_ducks : ℕ) (geese_leave : ℕ) : 
  let initial_geese := 2 * initial_ducks - 10
  let final_ducks := initial_ducks + 4
  let final_geese := initial_geese - (15 - 5)
  final_geese - final_ducks = 1 :=
by sorry

end NUMINAMATH_CALUDE_park_birds_difference_l478_47871


namespace NUMINAMATH_CALUDE_wendi_chicken_count_l478_47856

/-- The number of chickens Wendi has after various changes --/
def final_chicken_count (initial : ℕ) : ℕ :=
  let doubled := initial * 2
  let after_loss := doubled - 1
  let additional := 6
  after_loss + additional

/-- Theorem stating that starting with 4 chickens, Wendi ends up with 13 chickens --/
theorem wendi_chicken_count : final_chicken_count 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_wendi_chicken_count_l478_47856


namespace NUMINAMATH_CALUDE_train_length_proof_l478_47819

/-- Proves that the length of a train is 260 meters, given its speed and the time it takes to cross a platform of known length. -/
theorem train_length_proof (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (1000 / 3600) →
  platform_length = 260 →
  crossing_time = 26 →
  (train_speed * crossing_time) - platform_length = 260 :=
by sorry

end NUMINAMATH_CALUDE_train_length_proof_l478_47819


namespace NUMINAMATH_CALUDE_quadratic_inequality_quadratic_inequality_negative_m_l478_47835

theorem quadratic_inequality (m : ℝ) :
  (∀ x : ℝ, m * x^2 + (1 - m) * x + m - 2 ≥ -2) ↔ m ≥ 1/3 := by sorry

theorem quadratic_inequality_negative_m (m : ℝ) (hm : m < 0) :
  (∀ x : ℝ, m * x^2 + (1 - m) * x + m - 2 < m - 1) ↔
  ((m ≤ -1 ∧ (x < -1/m ∨ x > 1)) ∨
   (-1 < m ∧ m < 0 ∧ (x < 1 ∨ x > -1/m))) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_quadratic_inequality_negative_m_l478_47835


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l478_47840

theorem partial_fraction_decomposition (x : ℝ) (h : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  (4 * x - 2) / (x^3 - x) = 2 / x + 1 / (x - 1) - 3 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l478_47840


namespace NUMINAMATH_CALUDE_female_democrat_ratio_l478_47891

/-- Proves that the ratio of female democrats to total female participants is 1:2 --/
theorem female_democrat_ratio (total_participants : ℕ) (female_democrats : ℕ) 
  (h1 : total_participants = 870)
  (h2 : female_democrats = 145)
  (h3 : ∃ (male_participants : ℕ), 
    male_participants + (total_participants - male_participants) = total_participants ∧
    male_participants / 4 + female_democrats = total_participants / 3) :
  female_democrats * 2 = total_participants - (total_participants - female_democrats * 2) := by
  sorry

end NUMINAMATH_CALUDE_female_democrat_ratio_l478_47891


namespace NUMINAMATH_CALUDE_cylinder_radius_is_18_over_5_l478_47858

/-- A right circular cone with a right circular cylinder inscribed within it. -/
structure ConeWithCylinder where
  cone_diameter : ℝ
  cone_altitude : ℝ
  cylinder_radius : ℝ

/-- The conditions for our specific cone and cylinder. -/
def cone_cylinder_conditions (c : ConeWithCylinder) : Prop :=
  c.cone_diameter = 12 ∧
  c.cone_altitude = 18 ∧
  c.cylinder_radius * 2 = c.cylinder_radius * 2

theorem cylinder_radius_is_18_over_5 (c : ConeWithCylinder) 
  (h : cone_cylinder_conditions c) : c.cylinder_radius = 18 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_is_18_over_5_l478_47858


namespace NUMINAMATH_CALUDE_inverse_g_at_19_128_l478_47877

noncomputable def g (x : ℝ) : ℝ := (x^7 - 1) / 4

theorem inverse_g_at_19_128 :
  g⁻¹ (19/128) = (51/32)^(1/7) := by
sorry

end NUMINAMATH_CALUDE_inverse_g_at_19_128_l478_47877


namespace NUMINAMATH_CALUDE_min_shift_for_symmetry_l478_47881

open Real

theorem min_shift_for_symmetry (φ : ℝ) : 
  φ > 0 ∧ 
  (∀ x, sin (2 * (x - φ)) = sin (2 * (π / 3 - x))) →
  φ ≥ 5 * π / 12 :=
sorry

end NUMINAMATH_CALUDE_min_shift_for_symmetry_l478_47881


namespace NUMINAMATH_CALUDE_sum_of_eleventh_powers_l478_47815

/-- Given two real numbers a and b satisfying certain conditions, prove that a^11 + b^11 = 199 -/
theorem sum_of_eleventh_powers (a b : ℝ) : 
  (a + b = 1) →
  (a^2 + b^2 = 3) →
  (a^3 + b^3 = 4) →
  (a^4 + b^4 = 7) →
  (a^5 + b^5 = 11) →
  (∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) →
  a^11 + b^11 = 199 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_eleventh_powers_l478_47815


namespace NUMINAMATH_CALUDE_cardboard_pins_l478_47893

/-- Calculates the total number of pins used on a rectangular cardboard -/
def total_pins (length width pins_per_side : ℕ) : ℕ :=
  2 * pins_per_side * (length + width)

/-- Theorem: For a 34 * 14 cardboard with 35 pins per side, the total pins used is 140 -/
theorem cardboard_pins :
  total_pins 34 14 35 = 140 := by
  sorry

end NUMINAMATH_CALUDE_cardboard_pins_l478_47893


namespace NUMINAMATH_CALUDE_base_seven_43210_equals_10738_l478_47865

def base_seven_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_seven_43210_equals_10738 :
  base_seven_to_decimal [0, 1, 2, 3, 4] = 10738 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_43210_equals_10738_l478_47865


namespace NUMINAMATH_CALUDE_negation_of_forall_abs_plus_square_nonnegative_l478_47894

theorem negation_of_forall_abs_plus_square_nonnegative :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x : ℝ, |x| + x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_abs_plus_square_nonnegative_l478_47894


namespace NUMINAMATH_CALUDE_perpendicular_bisector_value_l478_47800

/-- A line that is a perpendicular bisector of a line segment passes through its midpoint -/
axiom perpendicular_bisector_passes_through_midpoint {x₁ y₁ x₂ y₂ : ℝ} (b : ℝ) :
  (∀ x y, x + y = b → (x - (x₁ + x₂) / 2)^2 + (y - (y₁ + y₂) / 2)^2 = ((x₂ - x₁)^2 + (y₂ - y₁)^2) / 4) →
  b = (x₁ + x₂) / 2 + (y₁ + y₂) / 2

/-- The value of b for the perpendicular bisector of the line segment from (2,1) to (8,7) -/
theorem perpendicular_bisector_value : 
  (∀ x y, x + y = b → (x - 5)^2 + (y - 4)^2 = 25) → b = 9 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_value_l478_47800


namespace NUMINAMATH_CALUDE_only_B_suitable_l478_47804

-- Define the structure for a sampling experiment
structure SamplingExperiment where
  totalSize : ℕ
  sampleSize : ℕ
  isWellMixed : Bool

-- Define the conditions for lottery method suitability
def isLotteryMethodSuitable (experiment : SamplingExperiment) : Prop :=
  experiment.totalSize ≤ 100 ∧ 
  experiment.sampleSize ≤ 10 ∧ 
  experiment.isWellMixed

-- Define the given sampling experiments
def experimentA : SamplingExperiment := ⟨5000, 600, true⟩
def experimentB : SamplingExperiment := ⟨36, 6, true⟩
def experimentC : SamplingExperiment := ⟨36, 6, false⟩
def experimentD : SamplingExperiment := ⟨5000, 10, true⟩

-- Theorem statement
theorem only_B_suitable : 
  ¬(isLotteryMethodSuitable experimentA) ∧
  (isLotteryMethodSuitable experimentB) ∧
  ¬(isLotteryMethodSuitable experimentC) ∧
  ¬(isLotteryMethodSuitable experimentD) :=
by sorry

end NUMINAMATH_CALUDE_only_B_suitable_l478_47804


namespace NUMINAMATH_CALUDE_solve_linear_equation_l478_47866

theorem solve_linear_equation (x : ℝ) (h : 7 - x = 12) : x = -5 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l478_47866


namespace NUMINAMATH_CALUDE_unique_number_l478_47825

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def reverse_digits (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

theorem unique_number : ∃! n : ℕ, 
  is_two_digit n ∧ 
  digit_sum n = 12 ∧ 
  reverse_digits (n + 36) = n :=
by sorry

end NUMINAMATH_CALUDE_unique_number_l478_47825


namespace NUMINAMATH_CALUDE_sarah_initial_money_l478_47863

def toy_car_price : ℕ := 11
def toy_car_quantity : ℕ := 2
def scarf_price : ℕ := 10
def beanie_price : ℕ := 14
def remaining_money : ℕ := 7

theorem sarah_initial_money :
  ∃ (initial_money : ℕ),
    initial_money = 
      remaining_money + beanie_price + scarf_price + (toy_car_price * toy_car_quantity) ∧
    initial_money = 53 := by
  sorry

end NUMINAMATH_CALUDE_sarah_initial_money_l478_47863


namespace NUMINAMATH_CALUDE_correct_quotient_l478_47879

theorem correct_quotient (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 12 = 70) : D / 21 = 40 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_l478_47879


namespace NUMINAMATH_CALUDE_lemonade_sale_duration_l478_47874

/-- 
Given that Stanley sells 4 cups of lemonade per hour and Carl sells 7 cups per hour,
prove that they sold lemonade for 3 hours if Carl sold 9 more cups than Stanley.
-/
theorem lemonade_sale_duration : ∃ h : ℕ, h > 0 ∧ 7 * h = 4 * h + 9 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_sale_duration_l478_47874


namespace NUMINAMATH_CALUDE_scheme2_more_cost_effective_l478_47802

/-- The cost of a teapot in yuan -/
def teapot_cost : ℝ := 25

/-- The cost of a tea cup in yuan -/
def teacup_cost : ℝ := 5

/-- The number of teapots the customer needs to buy -/
def num_teapots : ℕ := 4

/-- The discount percentage for Scheme 2 -/
def discount_percentage : ℝ := 0.94

/-- The cost calculation for Scheme 1 -/
def scheme1_cost (x : ℝ) : ℝ := 5 * x + 80

/-- The cost calculation for Scheme 2 -/
def scheme2_cost (x : ℝ) : ℝ := (teapot_cost * num_teapots + teacup_cost * x) * discount_percentage

/-- The number of tea cups for which we want to compare the schemes -/
def x : ℝ := 47

theorem scheme2_more_cost_effective : scheme2_cost x < scheme1_cost x := by
  sorry

end NUMINAMATH_CALUDE_scheme2_more_cost_effective_l478_47802


namespace NUMINAMATH_CALUDE_matrix_not_invertible_l478_47843

def matrix (x : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![2 + x, 9; 4 - x, 5]

theorem matrix_not_invertible (x : ℚ) :
  ¬(IsUnit (matrix x).det) ↔ x = 13/7 := by
  sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_l478_47843


namespace NUMINAMATH_CALUDE_least_xy_value_l478_47833

theorem least_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 8) : 
  (∀ a b : ℕ+, (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 8 → x * y ≤ a * b) ∧ x * y = 96 :=
sorry

end NUMINAMATH_CALUDE_least_xy_value_l478_47833


namespace NUMINAMATH_CALUDE_distribute_5_2_l478_47829

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 3 ways to distribute 5 indistinguishable balls into 2 indistinguishable boxes -/
theorem distribute_5_2 : distribute 5 2 = 3 := by sorry

end NUMINAMATH_CALUDE_distribute_5_2_l478_47829


namespace NUMINAMATH_CALUDE_common_root_quadratics_l478_47837

theorem common_root_quadratics (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ x^2 + x + a = 0) ↔ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_common_root_quadratics_l478_47837


namespace NUMINAMATH_CALUDE_problem_solution_l478_47851

theorem problem_solution (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) :
  a^2 - b^2 + 2*a*b = 64 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l478_47851


namespace NUMINAMATH_CALUDE_sequence_always_terminates_l478_47809

def last_digit (n : ℕ) : ℕ := n % 10

def remove_last_digit (n : ℕ) : ℕ := n / 10

def next_term (n : ℕ) : ℕ :=
  if n ≤ 5 then n
  else if last_digit n ≤ 5 then remove_last_digit n
  else 9 * n

def sequence_terminates (a₀ : ℕ) : Prop :=
  ∃ n : ℕ, (Nat.iterate next_term n a₀) ≤ 5

theorem sequence_always_terminates (a₀ : ℕ) : sequence_terminates a₀ := by
  sorry

#check sequence_always_terminates

end NUMINAMATH_CALUDE_sequence_always_terminates_l478_47809


namespace NUMINAMATH_CALUDE_system_solution_l478_47808

theorem system_solution (a b x y : ℝ) : 
  (2 * a - 3 * b = 13 ∧ 3 * a + 5 * b = 30.9 ∧ a = 8.3 ∧ b = 1.2) →
  (2 * (x + 2) - 3 * (y - 1) = 13 ∧ 3 * (x + 2) + 5 * (y - 1) = 30.9) →
  (x = 6.3 ∧ y = 2.2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l478_47808


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l478_47859

/-- Given two vectors a and b in R², where a = (4,8) and b = (x,4),
    if a is perpendicular to b, then x = -8. -/
theorem perpendicular_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (4, 8)
  let b : ℝ × ℝ := (x, 4)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = -8 :=
by
  sorry

#check perpendicular_vectors_x_value

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l478_47859


namespace NUMINAMATH_CALUDE_distance_on_quadratic_curve_l478_47842

/-- The distance between two points on a quadratic curve. -/
theorem distance_on_quadratic_curve (m n p x₁ x₂ : ℝ) :
  let y₁ := m * x₁^2 + n * x₁ + p
  let y₂ := m * x₂^2 + n * x₂ + p
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = (x₂ - x₁)^2 * (1 + m^2 * (x₂ + x₁)^2 + n^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_on_quadratic_curve_l478_47842


namespace NUMINAMATH_CALUDE_infinite_fibonacci_divisible_l478_47895

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: For any positive integer N, there are infinitely many Fibonacci numbers divisible by N -/
theorem infinite_fibonacci_divisible (N : ℕ) (hN : N > 0) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ k ∈ S, N ∣ fib k := by
  sorry

end NUMINAMATH_CALUDE_infinite_fibonacci_divisible_l478_47895


namespace NUMINAMATH_CALUDE_hare_wolf_distance_l478_47846

def track_length : ℝ := 200
def hare_speed : ℝ := 5
def wolf_speed : ℝ := 3
def time_elapsed : ℝ := 40

def distance_traveled (speed : ℝ) : ℝ := speed * time_elapsed

theorem hare_wolf_distance : 
  ∃ (initial_distance : ℝ), 
    (initial_distance = 40 ∨ initial_distance = 60) ∧
    (
      (distance_traveled hare_speed - distance_traveled wolf_speed) % track_length = 0 ∨
      (distance_traveled hare_speed - distance_traveled wolf_speed + initial_distance) % track_length = initial_distance
    ) :=
by sorry

end NUMINAMATH_CALUDE_hare_wolf_distance_l478_47846


namespace NUMINAMATH_CALUDE_largest_lcm_with_15_l478_47862

theorem largest_lcm_with_15 : 
  (Nat.lcm 15 2).max 
    ((Nat.lcm 15 3).max 
      ((Nat.lcm 15 5).max 
        ((Nat.lcm 15 6).max 
          ((Nat.lcm 15 9).max 
            (Nat.lcm 15 10))))) = 45 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_15_l478_47862


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l478_47801

theorem sum_of_x_and_y (x y : ℝ) (hx : |x| = 3) (hy : |y| = 4) (hxy : x * y > 0) :
  x + y = 7 ∨ x + y = -7 := by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l478_47801


namespace NUMINAMATH_CALUDE_stamps_bought_theorem_l478_47826

/-- The total number of stamps bought by Evariste and Sophie -/
def total_stamps (x y : ℕ) : ℕ := x + y

/-- The cost of Evariste's stamps in pence -/
def evariste_cost : ℕ := 110

/-- The cost of Sophie's stamps in pence -/
def sophie_cost : ℕ := 70

/-- The total amount spent in pence -/
def total_spent : ℕ := 1000

theorem stamps_bought_theorem (x y : ℕ) :
  x * evariste_cost + y * sophie_cost = total_spent →
  total_stamps x y = 12 := by
  sorry

#check stamps_bought_theorem

end NUMINAMATH_CALUDE_stamps_bought_theorem_l478_47826


namespace NUMINAMATH_CALUDE_no_simultaneous_integer_roots_l478_47823

theorem no_simultaneous_integer_roots :
  ¬ ∃ (b c : ℝ),
    (∃ (k l m n : ℤ),
      (k ≠ l ∧ m ≠ n) ∧
      (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = k ∨ x = l) ∧
      (∀ x : ℝ, 2*x^2 + (b+1)*x + (c+1) = 0 ↔ x = m ∨ x = n)) :=
by sorry

end NUMINAMATH_CALUDE_no_simultaneous_integer_roots_l478_47823


namespace NUMINAMATH_CALUDE_maggie_income_l478_47868

def office_rate : ℝ := 10
def tractor_rate : ℝ := 12
def tractor_hours : ℝ := 13
def office_hours : ℝ := 2 * tractor_hours

def total_income : ℝ := office_rate * office_hours + tractor_rate * tractor_hours

theorem maggie_income : total_income = 416 := by
  sorry

end NUMINAMATH_CALUDE_maggie_income_l478_47868


namespace NUMINAMATH_CALUDE_max_marks_calculation_l478_47845

/-- Proves that if a student scores 80% and receives 240 marks, the maximum possible marks in the examination is 300. -/
theorem max_marks_calculation (percentage : ℝ) (scored_marks : ℝ) (max_marks : ℝ) 
  (h1 : percentage = 0.80) 
  (h2 : scored_marks = 240) 
  (h3 : percentage * max_marks = scored_marks) : 
  max_marks = 300 := by
  sorry

end NUMINAMATH_CALUDE_max_marks_calculation_l478_47845


namespace NUMINAMATH_CALUDE_no_double_application_function_l478_47880

theorem no_double_application_function : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1987 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_function_l478_47880


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l478_47848

/-- Triangle ABC with given conditions -/
structure TriangleABC where
  /-- Point B has coordinates (4,4) -/
  B : ℝ × ℝ
  hB : B = (4, 4)
  
  /-- The equation of the angle bisector of ∠A is y = 0 -/
  angle_bisector : ℝ → ℝ
  h_angle_bisector : ∀ x, angle_bisector x = 0
  
  /-- The equation of the altitude from B to AC is x - 2y + 2 = 0 -/
  altitude : ℝ → ℝ
  h_altitude : ∀ x, altitude x = (x + 2) / 2

/-- The coordinates of point C in triangle ABC -/
def point_C (t : TriangleABC) : ℝ × ℝ := (10, -8)

/-- The area of triangle ABC -/
def area (t : TriangleABC) : ℝ := 48

/-- Main theorem: The coordinates of C and the area of triangle ABC are correct -/
theorem triangle_abc_properties (t : TriangleABC) : 
  (point_C t = (10, -8)) ∧ (area t = 48) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l478_47848


namespace NUMINAMATH_CALUDE_rectangle_formations_l478_47861

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of horizontal lines -/
def horizontal_lines : ℕ := 5

/-- The number of vertical lines -/
def vertical_lines : ℕ := 5

/-- The number of horizontal lines needed to form a rectangle -/
def horizontal_lines_needed : ℕ := 2

/-- The number of vertical lines needed to form a rectangle -/
def vertical_lines_needed : ℕ := 2

/-- The theorem stating the number of ways to form a rectangle -/
theorem rectangle_formations :
  (choose horizontal_lines horizontal_lines_needed) *
  (choose vertical_lines vertical_lines_needed) = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formations_l478_47861


namespace NUMINAMATH_CALUDE_third_row_is_10302_l478_47875

/-- Represents a 3x5 grid of numbers -/
def Grid := Fin 3 → Fin 5 → Nat

/-- Check if a grid satisfies the given conditions -/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i : Fin 3, ∃! j : Fin 5, g i j = 1) ∧
  (∀ i : Fin 3, ∃! j : Fin 5, g i j = 2) ∧
  (∀ i : Fin 3, ∃! j : Fin 5, g i j = 3) ∧
  (∀ j : Fin 5, ∃! i : Fin 3, g i j = 1) ∧
  (∀ j : Fin 5, ∃! i : Fin 3, g i j = 2) ∧
  (∀ j : Fin 5, ∃! i : Fin 3, g i j = 3)

/-- The sequence of numbers along the track -/
def track_sequence : Nat → Nat
| n => (n % 3) + 1

/-- The theorem stating that the third row of a valid grid is [1,0,3,0,2] -/
theorem third_row_is_10302 (g : Grid) (h : is_valid_grid g) :
  (g 2 0 = 1) ∧ (g 2 1 = 0) ∧ (g 2 2 = 3) ∧ (g 2 3 = 0) ∧ (g 2 4 = 2) :=
sorry

end NUMINAMATH_CALUDE_third_row_is_10302_l478_47875


namespace NUMINAMATH_CALUDE_largest_common_divisor_l478_47803

theorem largest_common_divisor : 
  let a := 924
  let b := 1386
  let c := 462
  Nat.gcd a (Nat.gcd b c) = 462 := by
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_l478_47803


namespace NUMINAMATH_CALUDE_max_garden_area_l478_47896

def garden_area (width : ℝ) : ℝ := 2 * width * width

def garden_perimeter (width : ℝ) : ℝ := 6 * width

theorem max_garden_area :
  ∃ (w : ℝ), w > 0 ∧ garden_perimeter w = 480 ∧
  ∀ (x : ℝ), x > 0 ∧ garden_perimeter x = 480 → garden_area x ≤ garden_area w ∧
  garden_area w = 12800 :=
sorry

end NUMINAMATH_CALUDE_max_garden_area_l478_47896


namespace NUMINAMATH_CALUDE_cable_car_arrangement_l478_47834

def adults : ℕ := 4
def children : ℕ := 2
def total_people : ℕ := adults + children
def max_cable_cars : ℕ := 3
def max_capacity : ℕ := 3

def arrangement_count : ℕ := sorry

theorem cable_car_arrangement :
  adults = 4 ∧ 
  children = 2 ∧ 
  total_people = adults + children ∧
  max_cable_cars = 3 ∧
  max_capacity = 3 →
  arrangement_count = 348 := by sorry

end NUMINAMATH_CALUDE_cable_car_arrangement_l478_47834


namespace NUMINAMATH_CALUDE_sphere_surface_area_in_dihedral_angle_l478_47841

/-- The surface area of a part of a sphere inside a dihedral angle -/
theorem sphere_surface_area_in_dihedral_angle 
  (R a α : ℝ) 
  (h_positive_R : R > 0)
  (h_positive_a : a > 0)
  (h_a_lt_R : a < R)
  (h_angle_range : 0 < α ∧ α < π) :
  let surface_area := 
    2 * R^2 * Real.arccos ((R * Real.cos α) / Real.sqrt (R^2 - a^2 * Real.sin α^2)) - 
    2 * R * a * Real.sin α * Real.arccos ((a * Real.cos α) / Real.sqrt (R^2 - a^2 * Real.sin α^2))
  surface_area > 0 ∧ surface_area < 4 * π * R^2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_in_dihedral_angle_l478_47841


namespace NUMINAMATH_CALUDE_parallelogram_sides_sum_l478_47899

theorem parallelogram_sides_sum (x y : ℝ) : 
  (5 * x - 7 = 14) → 
  (3 * y + 4 = 8 * y - 3) → 
  x + y = 5.6 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_sides_sum_l478_47899


namespace NUMINAMATH_CALUDE_cube_property_l478_47876

theorem cube_property : ∃! (n : ℕ), n > 0 ∧ ∃ (k : ℕ), n^3 + 2*n^2 + 9*n + 8 = k^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_property_l478_47876


namespace NUMINAMATH_CALUDE_ellipse_equation_l478_47820

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis

/-- The foci of the ellipse -/
def f1 : Point := ⟨-4, 0⟩
def f2 : Point := ⟨4, 0⟩

/-- Distance between foci -/
def focalDistance : ℝ := 8

/-- Maximum area of triangle PF₁F₂ -/
def maxTriangleArea : ℝ := 12

/-- Theorem: Given an ellipse with foci at (-4,0) and (4,0), and maximum area of triangle PF₁F₂ is 12,
    the equation of the ellipse is x²/25 + y²/9 = 1 -/
theorem ellipse_equation (e : Ellipse) : 
  (focalDistance = 8) → 
  (maxTriangleArea = 12) → 
  (e.a^2 = 25 ∧ e.b^2 = 9) := by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ellipse_equation_l478_47820


namespace NUMINAMATH_CALUDE_w_squared_value_l478_47878

theorem w_squared_value (w : ℝ) (h : (w + 15)^2 = (4*w + 6)*(2*w + 3)) : w^2 = 207/7 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l478_47878


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l478_47885

theorem sin_2alpha_value (α : Real) (h : Real.sin α - Real.cos α = 1/5) : 
  Real.sin (2 * α) = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l478_47885


namespace NUMINAMATH_CALUDE_largest_ball_on_torus_l478_47850

/-- The radius of the largest spherical ball that can be placed on top of a torus -/
def largest_ball_radius (inner_radius outer_radius : ℝ) : ℝ :=
  outer_radius - inner_radius

/-- The torus is formed by revolving a circle with radius 1 centered at (4,0,1) -/
def torus_center_radius : ℝ := 4

/-- The height of the torus center above the table -/
def torus_center_height : ℝ := 1

/-- Theorem: The radius of the largest spherical ball on a torus with inner radius 3 and outer radius 5 is 4 -/
theorem largest_ball_on_torus :
  largest_ball_radius 3 5 = 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_ball_on_torus_l478_47850


namespace NUMINAMATH_CALUDE_johnny_earnings_l478_47839

/-- Represents Johnny's daily work schedule and earnings --/
structure DailyWork where
  hours1 : ℕ
  rate1 : ℕ
  hours2 : ℕ
  rate2 : ℕ
  hours3 : ℕ
  rate3 : ℕ

/-- Calculates the total earnings for a given number of days --/
def totalEarnings (work : DailyWork) (days : ℕ) : ℕ :=
  days * (work.hours1 * work.rate1 + work.hours2 * work.rate2 + work.hours3 * work.rate3)

/-- Johnny's work schedule --/
def johnnysWork : DailyWork :=
  { hours1 := 3
  , rate1 := 7
  , hours2 := 2
  , rate2 := 10
  , hours3 := 4
  , rate3 := 12 }

theorem johnny_earnings :
  totalEarnings johnnysWork 5 = 445 := by
  sorry

end NUMINAMATH_CALUDE_johnny_earnings_l478_47839


namespace NUMINAMATH_CALUDE_parabola_directrix_l478_47864

/-- The directrix of the parabola x = -1/4 * y^2 is x = 1 -/
theorem parabola_directrix :
  ∀ (x y : ℝ), x = -(1/4) * y^2 → 
  ∃ (d : ℝ), d = 1 ∧ 
  ∀ (p : ℝ × ℝ), p.1 = -(1/4) * p.2^2 → 
  (p.1 - d)^2 = (p.1 - (-d))^2 + p.2^2 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l478_47864


namespace NUMINAMATH_CALUDE_cubic_root_sum_simplification_l478_47810

theorem cubic_root_sum_simplification :
  (((9 : ℝ) / 16 + 25 / 36 + 4 / 9) ^ (1/3 : ℝ)) = (245 : ℝ) ^ (1/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_simplification_l478_47810


namespace NUMINAMATH_CALUDE_point_transformation_identity_l478_47821

def rotateZ90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflectXY (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def rotateX90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -z, y)

def reflectYZ (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

theorem point_transformation_identity :
  let initial_point : ℝ × ℝ × ℝ := (2, 2, 2)
  let transformed_point := reflectYZ (rotateX90 (reflectXY (rotateZ90 initial_point)))
  transformed_point = initial_point := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_identity_l478_47821


namespace NUMINAMATH_CALUDE_min_fraction_sum_l478_47860

def Digits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem min_fraction_sum (A B C D : Nat) : 
  A ∈ Digits → B ∈ Digits → C ∈ Digits → D ∈ Digits →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  Nat.Prime B → Nat.Prime D →
  (∀ A' B' C' D' : Nat, 
    A' ∈ Digits → B' ∈ Digits → C' ∈ Digits → D' ∈ Digits →
    A' ≠ B' → A' ≠ C' → A' ≠ D' → B' ≠ C' → B' ≠ D' → C' ≠ D' →
    Nat.Prime B' → Nat.Prime D' →
    (A : ℚ) / (B : ℚ) + (C : ℚ) / (D : ℚ) ≤ (A' : ℚ) / (B' : ℚ) + (C' : ℚ) / (D' : ℚ)) →
  (A : ℚ) / (B : ℚ) + (C : ℚ) / (D : ℚ) = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l478_47860


namespace NUMINAMATH_CALUDE_point_and_tangent_line_l478_47828

def f (a t x : ℝ) : ℝ := x^3 + a*x
def g (b c t x : ℝ) : ℝ := b*x^2 + c
def h (a b c t x : ℝ) : ℝ := f a t x - g b c t x

theorem point_and_tangent_line (t : ℝ) (h_t : t ≠ 0) :
  ∃ (a b c : ℝ),
    (f a t t = 0) ∧
    (g b c t t = 0) ∧
    (∀ x, (deriv (f a t)) x = (deriv (g b c t)) x) ∧
    (∀ x ∈ Set.Ioo (-1) 3, StrictMonoOn (h a b c t) (Set.Ioo (-1) 3)) →
    (a = -t^2 ∧ b = t ∧ c = -t^3 ∧ (t ≤ -9 ∨ t ≥ 3)) :=
by sorry

end NUMINAMATH_CALUDE_point_and_tangent_line_l478_47828


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_squared_l478_47882

theorem square_plus_reciprocal_squared (x : ℝ) (h : x^2 + 1/x^2 = 2) :
  x^4 + 1/x^4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_squared_l478_47882


namespace NUMINAMATH_CALUDE_circle_intersection_angle_l478_47838

theorem circle_intersection_angle (r₁ r₂ r₃ : ℝ) (shaded_ratio : ℝ) :
  r₁ = 4 → r₂ = 3 → r₃ = 2 →
  shaded_ratio = 5 / 11 →
  ∃ θ : ℝ,
    θ > 0 ∧ θ < π / 2 ∧
    (θ * (r₁^2 + r₃^2) + (π - θ) * r₂^2) / (π * (r₁^2 + r₂^2 + r₃^2)) = shaded_ratio ∧
    θ = π / 176 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_angle_l478_47838


namespace NUMINAMATH_CALUDE_circle_m_range_l478_47898

-- Define the equation of the circle
def circle_equation (x y m : ℝ) : Prop :=
  x + y^2 - x + y + m = 0

-- State the theorem
theorem circle_m_range :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) → m < (1/2) :=
by sorry

end NUMINAMATH_CALUDE_circle_m_range_l478_47898


namespace NUMINAMATH_CALUDE_power_of_product_l478_47890

theorem power_of_product (a : ℝ) : (3 * a) ^ 3 = 27 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l478_47890


namespace NUMINAMATH_CALUDE_triple_product_sum_two_l478_47832

theorem triple_product_sum_two (x y z : ℝ) :
  (x * y + z = 2) ∧ (y * z + x = 2) ∧ (z * x + y = 2) →
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -2 ∧ y = -2 ∧ z = -2)) :=
by sorry

end NUMINAMATH_CALUDE_triple_product_sum_two_l478_47832


namespace NUMINAMATH_CALUDE_factorization_2x_squared_minus_4x_l478_47873

theorem factorization_2x_squared_minus_4x (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2x_squared_minus_4x_l478_47873


namespace NUMINAMATH_CALUDE_fraction_sum_l478_47897

theorem fraction_sum (x : ℝ) (h1 : x ≠ 1) (h2 : 2*x ≠ -3) (h3 : 2*x^2 + 5*x - 3 ≠ 0) : 
  (6*x - 8) / (2*x^2 + 5*x - 3) = (-2/5) / (x - 1) + (34/5) / (2*x + 3) := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_l478_47897


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l478_47813

theorem cos_x_plus_2y_equals_one 
  (x y : ℝ) 
  (h_x : x ∈ Set.Icc (-π/4) (π/4))
  (h_y : y ∈ Set.Icc (-π/4) (π/4))
  (h_eq1 : ∃ a : ℝ, x^3 + Real.sin x - 2*a = 0)
  (h_eq2 : ∃ a : ℝ, 4*y^3 + (1/2) * Real.sin (2*y) + a = 0) :
  Real.cos (x + 2*y) = 1 :=
by sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l478_47813


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_parabola_l478_47814

/-- Given two points A(1, 0) and B(b, 0), if there exists a point C on the parabola y^2 = 4x
    such that triangle ABC is equilateral, then b = 5 or b = -1/3 -/
theorem equilateral_triangle_on_parabola (b : ℝ) :
  (∃ (x y : ℝ), y^2 = 4*x ∧ 
    ((x - 1)^2 + y^2 = (x - b)^2 + y^2) ∧
    ((x - 1)^2 + y^2 = (b - 1)^2) ∧
    ((x - b)^2 + y^2 = (b - 1)^2)) →
  b = 5 ∨ b = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_on_parabola_l478_47814


namespace NUMINAMATH_CALUDE_solve_for_a_l478_47872

theorem solve_for_a (x : ℝ) (a : ℝ) (h1 : 2 * x - a - 5 = 0) (h2 : x = 3) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l478_47872


namespace NUMINAMATH_CALUDE_real_part_of_i_times_one_minus_i_l478_47870

theorem real_part_of_i_times_one_minus_i (i : ℂ) : 
  i * i = -1 → Complex.re (i * (1 - i)) = 1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_i_times_one_minus_i_l478_47870


namespace NUMINAMATH_CALUDE_problem_statement_l478_47811

theorem problem_statement (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a + b - 1 / (2 * a) - 2 / b = 3 / 2) 
  (h4 : a < 1) : 
  b > 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l478_47811


namespace NUMINAMATH_CALUDE_book_cost_problem_l478_47827

theorem book_cost_problem (total_cost : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) 
  (h1 : total_cost = 420)
  (h2 : loss_percent = 0.15)
  (h3 : gain_percent = 0.19)
  (h4 : ∃ (sell_price : ℝ), 
    sell_price = (1 - loss_percent) * (total_cost - x) ∧ 
    sell_price = (1 + gain_percent) * x) : 
  ∃ (x : ℝ), x = 245 ∧ x + (total_cost - x) = total_cost := by
sorry

end NUMINAMATH_CALUDE_book_cost_problem_l478_47827


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_one_and_six_l478_47886

theorem sqrt_equality_implies_one_and_six (a b : ℕ) (ha : a > 0) (hb : b > 0) (hlt : a < b) :
  (Real.sqrt (1 + Real.sqrt (40 + 24 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b) →
  a = 1 ∧ b = 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_one_and_six_l478_47886


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l478_47889

theorem parallelogram_side_length 
  (s : ℝ) 
  (side1 : ℝ) 
  (side2 : ℝ) 
  (angle : ℝ) 
  (area : ℝ) 
  (h1 : side1 = 3 * s) 
  (h2 : side2 = s) 
  (h3 : angle = π / 3) -- 60 degrees in radians
  (h4 : area = 9 * Real.sqrt 3) 
  (h5 : area = side1 * side2 * Real.sin angle) : 
  s = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l478_47889


namespace NUMINAMATH_CALUDE_isosceles_top_angle_l478_47888

-- Define an isosceles triangle
def IsIsosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

-- Define the sum of angles in a triangle
axiom angle_sum : ∀ (x y z : ℝ), x + y + z = 180

-- Theorem statement
theorem isosceles_top_angle (a b c : ℝ) 
  (h1 : IsIsosceles a b c) (h2 : a = 40 ∨ b = 40 ∨ c = 40) : 
  a = 40 ∨ b = 40 ∨ c = 40 ∨ a = 100 ∨ b = 100 ∨ c = 100 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_top_angle_l478_47888


namespace NUMINAMATH_CALUDE_leas_purchases_total_cost_l478_47869

/-- The total cost of Léa's purchases is $28, given that she bought one book for $16, 
    three binders for $2 each, and six notebooks for $1 each. -/
theorem leas_purchases_total_cost : 
  let book_cost : ℕ := 16
  let binder_cost : ℕ := 2
  let notebook_cost : ℕ := 1
  let num_binders : ℕ := 3
  let num_notebooks : ℕ := 6
  book_cost + num_binders * binder_cost + num_notebooks * notebook_cost = 28 :=
by sorry

end NUMINAMATH_CALUDE_leas_purchases_total_cost_l478_47869


namespace NUMINAMATH_CALUDE_fourth_roots_of_unity_solution_l478_47817

theorem fourth_roots_of_unity_solution (a b c d k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (eq1 : a * k^3 + b * k^2 + c * k + d = 0)
  (eq2 : b * k^3 + c * k^2 + d * k + a = 0) :
  k = 1 ∨ k = -1 ∨ k = I ∨ k = -I :=
sorry

end NUMINAMATH_CALUDE_fourth_roots_of_unity_solution_l478_47817


namespace NUMINAMATH_CALUDE_intersection_X_complement_Y_l478_47884

def U : Set ℝ := Set.univ

def X : Set ℝ := {x | x^2 - x = 0}

def Y : Set ℝ := {x | x^2 + x = 0}

theorem intersection_X_complement_Y : X ∩ (U \ Y) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_X_complement_Y_l478_47884


namespace NUMINAMATH_CALUDE_quadratic_polynomial_condition_l478_47847

/-- A quadratic polynomial of the form ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluate a quadratic polynomial at a given x -/
def QuadraticPolynomial.evaluate (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- A quadratic polynomial satisfies the given condition if
    f(a) = a, f(b) = b, and f(c) = c -/
def satisfies_condition (p : QuadraticPolynomial) : Prop :=
  p.evaluate p.a = p.a ∧
  p.evaluate p.b = p.b ∧
  p.evaluate p.c = p.c

/-- The theorem stating that only x^2 + x - 1 and x - 2 satisfy the condition -/
theorem quadratic_polynomial_condition :
  ∀ p : QuadraticPolynomial,
    satisfies_condition p →
      (p = ⟨1, 1, -1⟩ ∨ p = ⟨0, 1, -2⟩) :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_condition_l478_47847


namespace NUMINAMATH_CALUDE_equation_solution_l478_47830

theorem equation_solution : 
  ∃ x : ℝ, (64 + 5 * 12 / (x / 3) = 65) ∧ (x = 180) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l478_47830


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l478_47806

/-- A linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a linear function -/
def pointOnLinearFunction (p : Point) (f : LinearFunction) : Prop :=
  p.y = f.m * p.x + f.b

/-- The theorem to be proved -/
theorem y1_greater_than_y2
  (f : LinearFunction)
  (A B C : Point)
  (h1 : f.m ≠ 0)
  (h2 : f.b = 4)
  (h3 : A.x = -2)
  (h4 : B.x = 1)
  (h5 : B.y = 3)
  (h6 : C.x = 3)
  (h7 : pointOnLinearFunction A f)
  (h8 : pointOnLinearFunction B f)
  (h9 : pointOnLinearFunction C f) :
  A.y > C.y :=
sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l478_47806


namespace NUMINAMATH_CALUDE_union_equals_reals_l478_47836

def S : Set ℝ := {x | x < -1 ∨ x > 5}
def T (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 8}

theorem union_equals_reals (a : ℝ) : 
  S ∪ T a = Set.univ ↔ -3 < a ∧ a < -1 := by sorry

end NUMINAMATH_CALUDE_union_equals_reals_l478_47836


namespace NUMINAMATH_CALUDE_fish_population_estimate_l478_47816

theorem fish_population_estimate (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  initial_tagged = 60 →
  second_catch = 60 →
  tagged_in_second = 2 →
  (initial_tagged : ℚ) / (second_catch : ℚ) = (tagged_in_second : ℚ) / (initial_tagged : ℚ) →
  (initial_tagged * second_catch : ℚ) / tagged_in_second = 1800 :=
by
  sorry

#check fish_population_estimate

end NUMINAMATH_CALUDE_fish_population_estimate_l478_47816


namespace NUMINAMATH_CALUDE_value_of_a_l478_47812

/-- Given that 4 * ((a * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005,
    prove that a is approximately equal to 3.6 -/
theorem value_of_a (a : ℝ) : 
  4 * ((a * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005 → 
  ∃ ε > 0, |a - 3.6| < ε := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l478_47812


namespace NUMINAMATH_CALUDE_smallest_factor_l478_47805

theorem smallest_factor (n : ℕ) : n = 900 ↔ 
  (∀ m : ℕ, m > 0 → m < n → ¬(2^5 ∣ (936 * m) ∧ 3^3 ∣ (936 * m) ∧ 10^2 ∣ (936 * m))) ∧
  (2^5 ∣ (936 * n) ∧ 3^3 ∣ (936 * n) ∧ 10^2 ∣ (936 * n)) ∧
  (n > 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_l478_47805


namespace NUMINAMATH_CALUDE_max_intersections_15_10_l478_47857

/-- The maximum number of intersection points for segments connecting points on x and y axes -/
def max_intersections (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating the maximum number of intersections for 15 x-axis points and 10 y-axis points -/
theorem max_intersections_15_10 :
  max_intersections 15 10 = 4725 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_15_10_l478_47857


namespace NUMINAMATH_CALUDE_num_buses_is_ten_l478_47852

-- Define the given conditions
def total_people : ℕ := 342
def num_vans : ℕ := 9
def people_per_van : ℕ := 8
def people_per_bus : ℕ := 27

-- Define the function to calculate the number of buses
def calculate_buses : ℕ :=
  (total_people - num_vans * people_per_van) / people_per_bus

-- Theorem statement
theorem num_buses_is_ten : calculate_buses = 10 := by
  sorry

end NUMINAMATH_CALUDE_num_buses_is_ten_l478_47852


namespace NUMINAMATH_CALUDE_total_production_theorem_l478_47854

def week1_production : ℕ := 320
def week2_production : ℕ := 400
def week3_production : ℕ := 300

def total_3_weeks : ℕ := week1_production + week2_production + week3_production
def average_3_weeks : ℕ := total_3_weeks / 3
def total_4_weeks : ℕ := total_3_weeks + average_3_weeks

theorem total_production_theorem : total_4_weeks = 1360 := by
  sorry

end NUMINAMATH_CALUDE_total_production_theorem_l478_47854


namespace NUMINAMATH_CALUDE_range_of_a_l478_47824

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B (a : ℝ) : Set ℝ := {x | (2*x - a) / (x + 1) > 1}

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (A ⊂ B a ∧ A ≠ B a) → a < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l478_47824


namespace NUMINAMATH_CALUDE_bike_rental_problem_l478_47867

/-- Calculates the number of hours a bike was rented given the total payment, fixed fee, and hourly rate. -/
def rentedHours (totalPayment fixedFee hourlyRate : ℚ) : ℚ :=
  (totalPayment - fixedFee) / hourlyRate

theorem bike_rental_problem :
  let totalPayment : ℚ := 80
  let fixedFee : ℚ := 17
  let hourlyRate : ℚ := 7
  rentedHours totalPayment fixedFee hourlyRate = 9 := by
sorry

#eval rentedHours 80 17 7

end NUMINAMATH_CALUDE_bike_rental_problem_l478_47867


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l478_47849

theorem quadratic_roots_relation (m n p : ℝ) 
  (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0)
  (h : ∃ (r₁ r₂ : ℝ), (r₁ + r₂ = -p ∧ r₁ * r₂ = m) ∧ 
                      (3*r₁ + 3*r₂ = -m ∧ 9*r₁*r₂ = n)) :
  n / p = -27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l478_47849


namespace NUMINAMATH_CALUDE_village_population_after_events_l478_47818

theorem village_population_after_events (initial_population : ℕ) : 
  initial_population = 7600 → 
  (initial_population - initial_population / 10 - 
   (initial_population - initial_population / 10) / 4) = 5130 := by
sorry

end NUMINAMATH_CALUDE_village_population_after_events_l478_47818


namespace NUMINAMATH_CALUDE_cat_dog_ratio_l478_47831

def kennel (num_dogs : ℕ) (num_cats : ℕ) : Prop :=
  num_cats = num_dogs - 6 ∧ num_dogs = 18

theorem cat_dog_ratio (num_dogs num_cats : ℕ) :
  kennel num_dogs num_cats →
  (num_cats : ℚ) / (num_dogs : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cat_dog_ratio_l478_47831


namespace NUMINAMATH_CALUDE_parallel_intersecting_lines_c_is_zero_l478_47883

/-- Two lines that are parallel and intersect at a specific point -/
structure ParallelIntersectingLines where
  a : ℝ
  b : ℝ
  c : ℝ
  parallel : a / 2 = -2 / b
  intersect_x : 2 * a - 2 * (-4) = c
  intersect_y : 2 * 2 + b * (-4) = c

/-- The theorem stating that for such lines, c must be 0 -/
theorem parallel_intersecting_lines_c_is_zero (lines : ParallelIntersectingLines) : lines.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_parallel_intersecting_lines_c_is_zero_l478_47883


namespace NUMINAMATH_CALUDE_right_triangle_leg_l478_47892

theorem right_triangle_leg (h : Real) (angle : Real) :
  angle = Real.pi / 4 →
  h = 10 * Real.sqrt 2 →
  h * Real.sin angle = 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_l478_47892


namespace NUMINAMATH_CALUDE_min_value_expression_l478_47822

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_constraint : a + b + c = 13) : 
  (a^2 + b^3 + c^4 + 2019) / (10*b + 123*c + 26) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l478_47822


namespace NUMINAMATH_CALUDE_starting_number_with_20_multiples_of_5_l478_47807

theorem starting_number_with_20_multiples_of_5 :
  (∃! n : ℕ, n ≤ 100 ∧ 
    (∃ s : Finset ℕ, s.card = 20 ∧ 
      (∀ m ∈ s, n ≤ m ∧ m ≤ 100 ∧ m % 5 = 0) ∧
      (∀ m : ℕ, n ≤ m ∧ m ≤ 100 ∧ m % 5 = 0 → m ∈ s)) ∧
    (∀ k : ℕ, k < n → 
      ¬(∃ s : Finset ℕ, s.card = 20 ∧ 
        (∀ m ∈ s, k ≤ m ∧ m ≤ 100 ∧ m % 5 = 0) ∧
        (∀ m : ℕ, k ≤ m ∧ m ≤ 100 ∧ m % 5 = 0 → m ∈ s)))) ∧
  (∀ n : ℕ, (∃! n : ℕ, n ≤ 100 ∧ 
    (∃ s : Finset ℕ, s.card = 20 ∧ 
      (∀ m ∈ s, n ≤ m ∧ m ≤ 100 ∧ m % 5 = 0) ∧
      (∀ m : ℕ, n ≤ m ∧ m ≤ 100 ∧ m % 5 = 0 → m ∈ s)) ∧
    (∀ k : ℕ, k < n → 
      ¬(∃ s : Finset ℕ, s.card = 20 ∧ 
        (∀ m ∈ s, k ≤ m ∧ m ≤ 100 ∧ m % 5 = 0) ∧
        (∀ m : ℕ, k ≤ m ∧ m ≤ 100 ∧ m % 5 = 0 → m ∈ s)))) → n = 10) :=
by sorry

end NUMINAMATH_CALUDE_starting_number_with_20_multiples_of_5_l478_47807


namespace NUMINAMATH_CALUDE_a_equals_seven_l478_47855

theorem a_equals_seven (A B : Set ℝ) (a : ℝ) : 
  A = {1, 2, a} → B = {1, 7} → B ⊆ A → a = 7 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_seven_l478_47855


namespace NUMINAMATH_CALUDE_initial_files_count_l478_47887

theorem initial_files_count (organized_morning : ℕ) (to_organize_afternoon : ℕ) (missing : ℕ) :
  organized_morning = to_organize_afternoon ∧
  to_organize_afternoon = missing ∧
  to_organize_afternoon = 15 →
  2 * organized_morning + to_organize_afternoon + missing = 60 :=
by sorry

end NUMINAMATH_CALUDE_initial_files_count_l478_47887
