import Mathlib

namespace complex_point_equivalence_l285_28502

theorem complex_point_equivalence : 
  let z : ℂ := (Complex.I) / (1 + 3 * Complex.I)
  z = (3 : ℝ) / 10 + ((1 : ℝ) / 10) * Complex.I :=
by sorry

end complex_point_equivalence_l285_28502


namespace quadratic_equation_solution_l285_28556

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 6 ∧ 
  (x₁^2 - 7*x₁ + 6 = 0) ∧ (x₂^2 - 7*x₂ + 6 = 0) :=
by sorry

end quadratic_equation_solution_l285_28556


namespace hall_length_l285_28523

/-- Proves that given the conditions of the hall and verandah, the length of the hall is 20 meters -/
theorem hall_length (hall_breadth : ℝ) (verandah_width : ℝ) (flooring_rate : ℝ) (total_cost : ℝ) :
  hall_breadth = 15 →
  verandah_width = 2.5 →
  flooring_rate = 3.5 →
  total_cost = 700 →
  ∃ (hall_length : ℝ),
    hall_length = 20 ∧
    (hall_length + 2 * verandah_width) * (hall_breadth + 2 * verandah_width) -
    hall_length * hall_breadth = total_cost / flooring_rate :=
by sorry

end hall_length_l285_28523


namespace smallest_product_is_623_l285_28585

def Digits : Finset Nat := {7, 8, 9, 0}

def valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def two_digit_number (tens ones : Nat) : Nat :=
  10 * tens + ones

theorem smallest_product_is_623 :
  ∀ a b c d : Nat,
    valid_arrangement a b c d →
    (two_digit_number a b) * (two_digit_number c d) ≥ 623 :=
by sorry

end smallest_product_is_623_l285_28585


namespace fraction_addition_l285_28580

theorem fraction_addition : (1 : ℚ) / 6 + (5 : ℚ) / 12 = (7 : ℚ) / 12 := by
  sorry

end fraction_addition_l285_28580


namespace polynomial_factorization_l285_28577

theorem polynomial_factorization (k : ℤ) :
  ∃ (p q : Polynomial ℤ),
    Polynomial.degree p = 4 ∧
    Polynomial.degree q = 4 ∧
    (X : Polynomial ℤ)^8 + (4 * k^4 - 8 * k^2 + 2) * X^4 + 1 = p * q :=
sorry

end polynomial_factorization_l285_28577


namespace inverse_proposition_correct_l285_28568

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop :=
  a = b → |a| = |b|

-- Define the inverse proposition
def inverse_proposition (a b : ℝ) : Prop :=
  |a| = |b| → a = b

-- Theorem stating that the inverse_proposition is indeed the inverse of the original_proposition
theorem inverse_proposition_correct :
  ∀ a b : ℝ, inverse_proposition a b ↔ ¬(original_proposition a b) :=
by sorry

end inverse_proposition_correct_l285_28568


namespace gcd_problem_l285_28563

theorem gcd_problem (a : ℤ) (h : 2142 ∣ a) : 
  Nat.gcd (Int.natAbs ((a^2 + 11*a + 28) : ℤ)) (Int.natAbs ((a + 6) : ℤ)) = 2 := by
  sorry

end gcd_problem_l285_28563


namespace square_diagonal_l285_28554

theorem square_diagonal (area : ℝ) (h : area = 800) :
  ∃ (diagonal : ℝ), diagonal = 40 ∧ diagonal^2 = 2 * area := by
  sorry

end square_diagonal_l285_28554


namespace car_journey_downhill_distance_l285_28549

/-- Proves that a car traveling 100 km uphill at 30 km/hr and an unknown distance downhill
    at 60 km/hr, with an average speed of 36 km/hr for the entire journey,
    travels 50 km downhill. -/
theorem car_journey_downhill_distance
  (uphill_speed : ℝ) (downhill_speed : ℝ) (uphill_distance : ℝ) (average_speed : ℝ)
  (h1 : uphill_speed = 30)
  (h2 : downhill_speed = 60)
  (h3 : uphill_distance = 100)
  (h4 : average_speed = 36)
  : ∃ (downhill_distance : ℝ),
    (uphill_distance + downhill_distance) / ((uphill_distance / uphill_speed) + (downhill_distance / downhill_speed)) = average_speed
    ∧ downhill_distance = 50 :=
by sorry

end car_journey_downhill_distance_l285_28549


namespace point_line_plane_membership_l285_28558

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relations for a point being on a line and within a plane
variable (on_line : Point → Line → Prop)
variable (in_plane : Point → Plane → Prop)

-- Define specific points, line, and plane
variable (A E F : Point)
variable (l : Line)
variable (ABC : Plane)

-- State the theorem
theorem point_line_plane_membership :
  (on_line A l) ∧ (in_plane E ABC) ∧ (in_plane F ABC) :=
sorry

end point_line_plane_membership_l285_28558


namespace divisor_count_not_25323_or_25322_l285_28582

def sequential_number (n : ℕ) : ℕ :=
  -- Definition of the number formed by writing integers from 1 to n sequentially
  sorry

def count_divisors (n : ℕ) : ℕ :=
  -- Definition to count the number of divisors of n
  sorry

theorem divisor_count_not_25323_or_25322 :
  let N := sequential_number 1975
  (count_divisors N ≠ 25323) ∧ (count_divisors N ≠ 25322) := by
  sorry

end divisor_count_not_25323_or_25322_l285_28582


namespace mary_regular_hours_l285_28583

/-- Represents Mary's work schedule and pay structure --/
structure MaryWork where
  maxHours : Nat
  regularRate : ℝ
  overtimeRateIncrease : ℝ
  maxEarnings : ℝ

/-- Calculates Mary's earnings based on regular hours worked --/
def calculateEarnings (work : MaryWork) (regularHours : ℝ) : ℝ :=
  let overtimeRate := work.regularRate * (1 + work.overtimeRateIncrease)
  let overtimeHours := work.maxHours - regularHours
  regularHours * work.regularRate + overtimeHours * overtimeRate

/-- Theorem stating that Mary works 20 hours at her regular rate to maximize earnings --/
theorem mary_regular_hours (work : MaryWork)
    (h1 : work.maxHours = 50)
    (h2 : work.regularRate = 8)
    (h3 : work.overtimeRateIncrease = 0.25)
    (h4 : work.maxEarnings = 460) :
    ∃ (regularHours : ℝ), regularHours = 20 ∧
    calculateEarnings work regularHours = work.maxEarnings :=
  sorry


end mary_regular_hours_l285_28583


namespace allison_not_lowest_prob_l285_28509

/-- Represents a 6-sided cube with specific face values -/
structure Cube where
  faces : Fin 6 → ℕ

/-- Allison's cube with all faces showing 3 -/
def allison_cube : Cube :=
  ⟨λ _ => 3⟩

/-- Brian's cube with faces numbered 1 to 6 -/
def brian_cube : Cube :=
  ⟨λ i => i.val + 1⟩

/-- Noah's cube with three faces showing 1 and three faces showing 4 -/
def noah_cube : Cube :=
  ⟨λ i => if i.val < 3 then 1 else 4⟩

/-- The probability of rolling a value less than or equal to 3 on Brian's cube -/
def brian_prob_le_3 : ℚ :=
  1/2

/-- The probability of rolling a 4 on Noah's cube -/
def noah_prob_4 : ℚ :=
  1/2

/-- The probability of both Brian and Noah rolling lower than Allison -/
def prob_both_lower : ℚ :=
  1/6

theorem allison_not_lowest_prob :
  1 - prob_both_lower = 5/6 :=
sorry

end allison_not_lowest_prob_l285_28509


namespace horner_first_coefficient_l285_28517

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

-- Define Horner's method for a 5th degree polynomial
def horner_method (a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((((a₅ * x + a₄) * x + a₃) * x + a₂) * x + a₁) * x + a₀

-- Theorem statement
theorem horner_first_coefficient (x : ℝ) :
  ∃ (a₁ : ℝ), horner_method 0.5 4 0 (-3) a₁ (-1) x = f x ∧ a₁ = 1 :=
sorry

end horner_first_coefficient_l285_28517


namespace four_Z_one_l285_28514

/-- Define the Z operation -/
def Z (a b : ℝ) : ℝ := a^3 - 3*a^2*b + 3*a*b^2 - b^3

/-- Theorem: The value of 4 Z 1 is 27 -/
theorem four_Z_one : Z 4 1 = 27 := by sorry

end four_Z_one_l285_28514


namespace student_allowance_l285_28597

theorem student_allowance (allowance : ℝ) : 
  (allowance * 2/5 * 2/3 * 3/4 * 9/10 = 1.20) → 
  allowance = 60 := by
sorry

end student_allowance_l285_28597


namespace initial_average_is_16_l285_28503

def initial_average_problem (A : ℝ) : Prop :=
  -- Define the sum of 6 initial observations
  let initial_sum := 6 * A
  -- Define the sum of 7 observations after adding the new one
  let new_sum := initial_sum + 9
  -- The new average is A - 1
  new_sum / 7 = A - 1

theorem initial_average_is_16 :
  ∃ A : ℝ, initial_average_problem A ∧ A = 16 := by
  sorry

end initial_average_is_16_l285_28503


namespace log_28_5_l285_28513

theorem log_28_5 (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 7 = b) :
  (Real.log 5) / (Real.log 28) = (1 - a) / (2 * a + b) := by
  sorry

end log_28_5_l285_28513


namespace michaels_ride_l285_28584

/-- Calculates the total distance traveled by a cyclist given their speed and time -/
def total_distance (speed : ℚ) (time : ℚ) : ℚ :=
  speed * time

/-- Represents Michael's cycling scenario -/
theorem michaels_ride (total_time : ℚ) (speed : ℚ) 
    (h1 : total_time = 40) 
    (h2 : speed = 2 / 5) : 
  total_distance speed total_time = 16 := by
  sorry

#eval total_distance (2/5) 40

end michaels_ride_l285_28584


namespace integer_fraction_problem_l285_28527

theorem integer_fraction_problem (a b : ℕ+) :
  (a.val > 0) →
  (b.val > 0) →
  (∃ k : ℤ, (a.val^3 * b.val - 1) = k * (a.val + 1)) →
  (∃ m : ℤ, (b.val^3 * a.val + 1) = m * (b.val - 1)) →
  ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) := by
sorry

end integer_fraction_problem_l285_28527


namespace unique_n_mod_59_l285_28591

theorem unique_n_mod_59 : ∃! n : ℤ, 0 ≤ n ∧ n < 59 ∧ 58 * n % 59 = 20 % 59 ∧ n = 39 := by
  sorry

end unique_n_mod_59_l285_28591


namespace cubic_three_zeros_l285_28573

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * a * x^2 + a

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * a * x

theorem cubic_three_zeros (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) →
  a > 1 ∨ a < -1 := by
  sorry

end cubic_three_zeros_l285_28573


namespace exponential_function_passes_through_one_l285_28595

theorem exponential_function_passes_through_one (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x
  f 0 = 1 := by sorry

end exponential_function_passes_through_one_l285_28595


namespace ratio_equality_l285_28560

theorem ratio_equality (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4 ∧ c / 4 ≠ 0) : 
  (a + b) / c = 5 / 4 := by
sorry

end ratio_equality_l285_28560


namespace hillarys_craft_price_l285_28548

/-- Proves that the price of each craft is $12 given the conditions of Hillary's sales and deposits -/
theorem hillarys_craft_price :
  ∀ (price : ℕ),
  (3 * price + 7 = 18 + 25) →
  price = 12 := by
sorry

end hillarys_craft_price_l285_28548


namespace sum_of_ratio_terms_l285_28557

-- Define the points
variable (A B C D O P X Y : ℝ × ℝ)

-- Define the lengths
def length_AD : ℝ := 10
def length_AO : ℝ := 10
def length_OB : ℝ := 10
def length_BC : ℝ := 10
def length_AB : ℝ := 12
def length_DO : ℝ := 12
def length_OC : ℝ := 12

-- Define the conditions
axiom isosceles_DAO : length_AD = length_AO
axiom isosceles_AOB : length_AO = length_OB
axiom isosceles_OBC : length_OB = length_BC
axiom P_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B
axiom OP_perpendicular_AB : (O.1 - P.1) * (B.1 - A.1) + (O.2 - P.2) * (B.2 - A.2) = 0
axiom X_midpoint_AD : X = ((A.1 + D.1) / 2, (A.2 + D.2) / 2)
axiom Y_midpoint_BC : Y = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the areas of trapezoids
def area_ABYX : ℝ := sorry
def area_XYCD : ℝ := sorry

-- Define the ratio of areas
def ratio_areas : ℚ := sorry

-- Theorem to prove
theorem sum_of_ratio_terms : 
  ∃ (p q : ℕ), ratio_areas = p / q ∧ p + q = 12 :=
sorry

end sum_of_ratio_terms_l285_28557


namespace baker_cakes_sold_l285_28530

/-- Proves the number of cakes sold by a baker given certain conditions -/
theorem baker_cakes_sold (cake_price : ℕ) (pie_price : ℕ) (pies_sold : ℕ) (total_revenue : ℕ) :
  cake_price = 12 →
  pie_price = 7 →
  pies_sold = 126 →
  total_revenue = 6318 →
  ∃ cakes_sold : ℕ, cakes_sold * cake_price + pies_sold * pie_price = total_revenue ∧ cakes_sold = 453 :=
by sorry

end baker_cakes_sold_l285_28530


namespace angle_between_AO₂_and_CO₁_is_45_degrees_l285_28524

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the orthocenter H
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the incenter of a triangle
def incenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the angle between two lines given by two points each
def angle_between_lines (P₁ Q₁ P₂ Q₂ : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem angle_between_AO₂_and_CO₁_is_45_degrees 
  (ABC : Triangle) 
  (acute_angled : sorry) -- Condition: ABC is acute-angled
  (angle_B_30 : sorry) -- Condition: ∠B = 30°
  : 
  let H := orthocenter ABC
  let O₁ := incenter ABC.A ABC.B H
  let O₂ := incenter ABC.C ABC.B H
  angle_between_lines ABC.A O₂ ABC.C O₁ = 45 := by sorry

end angle_between_AO₂_and_CO₁_is_45_degrees_l285_28524


namespace quadratic_properties_l285_28570

def quadratic_function (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem quadratic_properties :
  (quadratic_function (-1) = 0) ∧
  (∀ x : ℝ, quadratic_function (1 + x) = quadratic_function (1 - x)) := by
  sorry

end quadratic_properties_l285_28570


namespace male_associate_or_full_tenured_percentage_l285_28522

structure University where
  total_professors : ℕ
  women_professors : ℕ
  tenured_professors : ℕ
  associate_or_full_professors : ℕ
  women_or_tenured_professors : ℕ
  male_associate_or_full_professors : ℕ

def University.valid (u : University) : Prop :=
  u.women_professors = (70 * u.total_professors) / 100 ∧
  u.tenured_professors = (70 * u.total_professors) / 100 ∧
  u.associate_or_full_professors = (50 * u.total_professors) / 100 ∧
  u.women_or_tenured_professors = (90 * u.total_professors) / 100 ∧
  u.male_associate_or_full_professors = (80 * u.associate_or_full_professors) / 100

theorem male_associate_or_full_tenured_percentage (u : University) (h : u.valid) :
  (u.tenured_professors - u.women_professors + (u.women_or_tenured_professors - u.total_professors)) * 100 / u.male_associate_or_full_professors = 50 := by
  sorry

end male_associate_or_full_tenured_percentage_l285_28522


namespace binomial_6_2_l285_28598

theorem binomial_6_2 : Nat.choose 6 2 = 15 := by
  sorry

end binomial_6_2_l285_28598


namespace triangle_midpoint_dot_product_l285_28546

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  AB = 10 ∧ AC = 6 ∧ BC = 8

-- Define the midpoint
def Midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define the dot product
def DotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem triangle_midpoint_dot_product 
  (A B C M : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : Midpoint M A B) : 
  DotProduct (M.1 - C.1, M.2 - C.2) (A.1 - C.1, A.2 - C.2) + 
  DotProduct (M.1 - C.1, M.2 - C.2) (B.1 - C.1, B.2 - C.2) = 50 := by
  sorry

end triangle_midpoint_dot_product_l285_28546


namespace alpha_beta_relation_l285_28576

open Real

theorem alpha_beta_relation (α β : ℝ) :
  π / 2 < α ∧ α < π ∧
  π / 2 < β ∧ β < π ∧
  (1 - cos (2 * α)) * (1 + sin β) = sin (2 * α) * cos β →
  2 * α + β = 5 * π / 2 := by
sorry

end alpha_beta_relation_l285_28576


namespace binary_110011_equals_51_l285_28593

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end binary_110011_equals_51_l285_28593


namespace partition_ratio_theorem_l285_28587

theorem partition_ratio_theorem (n : ℕ) : 
  (∃ (A B : Finset ℕ), 
    (A ∪ B = Finset.range (n^2 + 1) \ {0}) ∧ 
    (A ∩ B = ∅) ∧
    (A.card = B.card) ∧
    ((A.sum id) / (B.sum id) = 39 / 64)) ↔ 
  (∃ k : ℕ, n = 206 * k) ∧ 
  Even n :=
sorry

end partition_ratio_theorem_l285_28587


namespace inequality_range_l285_28559

theorem inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3/8 < 0) ↔ -3 < k ∧ k ≤ 0 :=
by sorry

end inequality_range_l285_28559


namespace manager_salary_calculation_l285_28550

/-- Calculates the manager's salary given the number of employees, their average salary,
    and the increase in average salary when the manager is included. -/
def manager_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) : ℚ :=
  (avg_salary + avg_increase) * (num_employees + 1) - avg_salary * num_employees

/-- Theorem stating that given 25 employees with an average salary of 2500,
    if adding a manager's salary increases the average by 400,
    then the manager's salary is 12900. -/
theorem manager_salary_calculation :
  manager_salary 25 2500 400 = 12900 := by
  sorry

end manager_salary_calculation_l285_28550


namespace arithmetic_calculation_l285_28574

theorem arithmetic_calculation : 5 * 7 + 6 * 9 + 13 * 2 + 4 * 6 = 139 := by
  sorry

end arithmetic_calculation_l285_28574


namespace least_integer_with_two_prime_factors_l285_28507

/-- A function that returns true if a number has exactly two prime factors -/
def has_two_prime_factors (n : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p * q

/-- The main theorem stating that 33 is the least positive integer satisfying the condition -/
theorem least_integer_with_two_prime_factors :
  (∀ m : ℕ, m > 0 ∧ m < 33 → ¬(has_two_prime_factors m ∧ has_two_prime_factors (m + 1) ∧ has_two_prime_factors (m + 2))) ∧
  (has_two_prime_factors 33 ∧ has_two_prime_factors 34 ∧ has_two_prime_factors 35) :=
by sorry

#check least_integer_with_two_prime_factors

end least_integer_with_two_prime_factors_l285_28507


namespace remainder_of_n_l285_28534

theorem remainder_of_n (n : ℕ) 
  (h1 : n^2 % 7 = 3) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 6 := by
sorry

end remainder_of_n_l285_28534


namespace exists_double_application_square_l285_28540

theorem exists_double_application_square :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n ^ 2 := by
  sorry

end exists_double_application_square_l285_28540


namespace baking_time_proof_l285_28571

-- Define the total baking time for 4 pans
def total_time : ℕ := 28

-- Define the number of pans
def num_pans : ℕ := 4

-- Define the time for one pan
def time_per_pan : ℕ := total_time / num_pans

-- Theorem to prove
theorem baking_time_proof : time_per_pan = 7 := by
  sorry

end baking_time_proof_l285_28571


namespace expected_value_of_marbles_l285_28508

/-- The set of marble numbers -/
def MarbleSet : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The number of marbles to draw -/
def DrawCount : ℕ := 3

/-- The sum of a combination of marbles -/
def CombinationSum (c : Finset ℕ) : ℕ := c.sum id

/-- The expected value of the sum of drawn marbles -/
noncomputable def ExpectedValue : ℚ :=
  (MarbleSet.powerset.filter (λ s => s.card = DrawCount)).sum CombinationSum /
   (MarbleSet.powerset.filter (λ s => s.card = DrawCount)).card

/-- Theorem: The expected value of the sum of three randomly drawn marbles is 10.5 -/
theorem expected_value_of_marbles : ExpectedValue = 21/2 := by
  sorry

end expected_value_of_marbles_l285_28508


namespace min_segments_for_perimeter_is_three_l285_28526

/-- Represents an octagon formed by cutting a smaller rectangle from a larger rectangle -/
structure CutOutOctagon where
  /-- The length of the larger rectangle -/
  outer_length : ℝ
  /-- The width of the larger rectangle -/
  outer_width : ℝ
  /-- The length of the smaller cut-out rectangle -/
  inner_length : ℝ
  /-- The width of the smaller cut-out rectangle -/
  inner_width : ℝ
  /-- Ensures the inner rectangle fits inside the outer rectangle -/
  h_inner_fits : inner_length < outer_length ∧ inner_width < outer_width

/-- The minimum number of line segment lengths required to calculate the perimeter of a CutOutOctagon -/
def min_segments_for_perimeter (oct : CutOutOctagon) : ℕ := 3

/-- Theorem stating that the minimum number of line segment lengths required to calculate
    the perimeter of a CutOutOctagon is always 3 -/
theorem min_segments_for_perimeter_is_three (oct : CutOutOctagon) :
  min_segments_for_perimeter oct = 3 := by
  sorry

end min_segments_for_perimeter_is_three_l285_28526


namespace min_distance_to_line_min_distance_achievable_l285_28519

/-- The minimum distance from the origin to a point on the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : ∀ m n : ℝ, m + n = 4 → Real.sqrt (m^2 + n^2) ≥ 2 * Real.sqrt 2 := by
  sorry

/-- The minimum distance 2√2 is achievable -/
theorem min_distance_achievable : ∃ m n : ℝ, m + n = 4 ∧ Real.sqrt (m^2 + n^2) = 2 * Real.sqrt 2 := by
  sorry

end min_distance_to_line_min_distance_achievable_l285_28519


namespace larger_number_of_pair_l285_28542

theorem larger_number_of_pair (a b : ℝ) (h1 : a - b = 5) (h2 : a + b = 37) :
  max a b = 21 := by sorry

end larger_number_of_pair_l285_28542


namespace fourth_root_unity_sum_l285_28581

theorem fourth_root_unity_sum (ζ : ℂ) (h : ζ^4 = 1) (h_nonreal : ζ ≠ 1 ∧ ζ ≠ -1) :
  (1 - ζ + ζ^3)^4 + (1 + ζ - ζ^3)^4 = -14 - 48 * I :=
by sorry

end fourth_root_unity_sum_l285_28581


namespace arithmetic_sequence_property_l285_28501

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 3 + a 5 + a 11 + a 13 = 80 →
  a 8 = 20 := by
sorry

end arithmetic_sequence_property_l285_28501


namespace abc_inequality_l285_28521

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 / (1 + a^2) + b^2 / (1 + b^2) + c^2 / (1 + c^2) = 1) :
  a * b * c ≤ Real.sqrt 2 / 4 := by
sorry

end abc_inequality_l285_28521


namespace problem_solution_l285_28579

theorem problem_solution (a b : ℝ) : 
  (|a| = 6 ∧ |b| = 2) →
  (((a * b > 0) → |a + b| = 8) ∧
   ((|a + b| = a + b) → (a - b = 4 ∨ a - b = 8))) := by
sorry

end problem_solution_l285_28579


namespace equation_solution_l285_28567

theorem equation_solution : ∃ x : ℝ, 6*x - 4*x = 380 - 10*(x + 2) ∧ x = 30 := by
  sorry

end equation_solution_l285_28567


namespace probability_is_one_fourteenth_l285_28572

/-- Represents a cube with side length 4 and two adjacent painted faces -/
structure PaintedCube :=
  (side_length : ℕ)
  (total_cubes : ℕ)
  (two_face_cubes : ℕ)
  (no_face_cubes : ℕ)

/-- The probability of selecting one cube with two painted faces and one with no painted faces -/
def select_probability (c : PaintedCube) : ℚ :=
  (c.two_face_cubes * c.no_face_cubes) / (c.total_cubes.choose 2)

/-- The theorem stating the probability is 1/14 -/
theorem probability_is_one_fourteenth (c : PaintedCube) 
  (h1 : c.side_length = 4)
  (h2 : c.total_cubes = 64)
  (h3 : c.two_face_cubes = 4)
  (h4 : c.no_face_cubes = 36) :
  select_probability c = 1 / 14 := by
  sorry

#eval select_probability { side_length := 4, total_cubes := 64, two_face_cubes := 4, no_face_cubes := 36 }

end probability_is_one_fourteenth_l285_28572


namespace integers_between_cubes_l285_28535

theorem integers_between_cubes : ∃ n : ℕ, n = (⌊(9.5 : ℝ)^3⌋ - ⌈(9.4 : ℝ)^3⌉ + 1) ∧ n = 27 := by
  sorry

end integers_between_cubes_l285_28535


namespace banana_purchase_cost_l285_28500

/-- The cost of bananas in dollars per three pounds -/
def banana_rate : ℚ := 3

/-- The amount of bananas in pounds to be purchased -/
def banana_amount : ℚ := 18

/-- The cost of purchasing the given amount of bananas -/
def banana_cost : ℚ := banana_amount * (banana_rate / 3)

theorem banana_purchase_cost : banana_cost = 18 := by
  sorry

end banana_purchase_cost_l285_28500


namespace expressions_equality_l285_28538

variable (a b c : ℝ)

theorem expressions_equality :
  (a - (b + c) = a - b - c) ∧
  (a + (-b - c) = a - b - c) ∧
  (a - (b - c) ≠ a - b - c) ∧
  ((-c) + (a - b) = a - b - c) := by
  sorry

end expressions_equality_l285_28538


namespace roots_product_minus_three_l285_28544

theorem roots_product_minus_three (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 7 * x₁ - 6 = 0) → 
  (3 * x₂^2 - 7 * x₂ - 6 = 0) → 
  (x₁ - 3) * (x₂ - 3) = 0 := by
  sorry

end roots_product_minus_three_l285_28544


namespace ratio_equality_l285_28511

theorem ratio_equality (x y : ℚ) (h : x / (2 * y) = 27) :
  (7 * x + 6 * y) / (x - 2 * y) = 96 / 13 := by
  sorry

end ratio_equality_l285_28511


namespace flour_to_add_l285_28525

/-- The total number of cups of flour required by the recipe. -/
def total_flour : ℕ := 7

/-- The number of cups of flour Mary has already added. -/
def flour_added : ℕ := 2

/-- The number of cups of flour Mary still needs to add. -/
def flour_needed : ℕ := total_flour - flour_added

theorem flour_to_add : flour_needed = 5 := by
  sorry

end flour_to_add_l285_28525


namespace share_of_y_is_36_l285_28510

/-- The share of y in rupees when a sum is divided among x, y, and z -/
def share_of_y (total : ℚ) (x_share : ℚ) (y_share : ℚ) (z_share : ℚ) : ℚ :=
  (y_share / x_share) * (total / (1 + y_share / x_share + z_share / x_share))

/-- Theorem: The share of y is 36 rupees given the problem conditions -/
theorem share_of_y_is_36 :
  share_of_y 156 1 (45/100) (1/2) = 36 := by
  sorry

#eval share_of_y 156 1 (45/100) (1/2)

end share_of_y_is_36_l285_28510


namespace shoe_shopping_cost_l285_28562

theorem shoe_shopping_cost 
  (price1 price2 price3 : ℝ) 
  (half_off_discount : ℝ → ℝ)
  (third_pair_discount : ℝ → ℝ)
  (extra_discount : ℝ → ℝ)
  (sales_tax : ℝ → ℝ)
  (h1 : price1 = 40)
  (h2 : price2 = 60)
  (h3 : price3 = 80)
  (h4 : half_off_discount x = x / 2)
  (h5 : third_pair_discount x = x * 0.7)
  (h6 : extra_discount x = x * 0.75)
  (h7 : sales_tax x = x * 1.08)
  : sales_tax (extra_discount (price1 + (price2 - half_off_discount price1) + third_pair_discount price3)) = 110.16 := by
  sorry

end shoe_shopping_cost_l285_28562


namespace fraction_equality_l285_28594

theorem fraction_equality : (18 - 6) / ((3 + 3) * 2) = 1 := by
  sorry

end fraction_equality_l285_28594


namespace square_root_five_expansion_l285_28569

theorem square_root_five_expansion 
  (a b m n : ℤ) 
  (h : a + b * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) : 
  a = m^2 + 5*n^2 ∧ b = 2*m*n := by
sorry

end square_root_five_expansion_l285_28569


namespace sum_of_fractions_l285_28533

theorem sum_of_fractions (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end sum_of_fractions_l285_28533


namespace total_price_calculation_l285_28515

/-- Calculates the total price of an order of ice-cream bars and sundaes -/
theorem total_price_calculation (ice_cream_bars sundaes : ℕ) (ice_cream_price sundae_price : ℚ) :
  ice_cream_bars = 125 →
  sundaes = 125 →
  ice_cream_price = 0.60 →
  sundae_price = 1.40 →
  ice_cream_bars * ice_cream_price + sundaes * sundae_price = 250 :=
by
  sorry

#check total_price_calculation

end total_price_calculation_l285_28515


namespace expression_value_l285_28578

theorem expression_value :
  let a : ℝ := 10
  let b : ℝ := 4
  let c : ℝ := 3
  (a - (b - c^2)) - ((a - b) - c^2) = 18 := by sorry

end expression_value_l285_28578


namespace ten_person_tournament_matches_l285_28590

/-- Calculate the number of matches in a round-robin tournament. -/
def roundRobinMatches (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem: A 10-person round-robin chess tournament has 45 matches. -/
theorem ten_person_tournament_matches :
  roundRobinMatches 10 = 45 := by
  sorry

#eval roundRobinMatches 10  -- Should output 45

end ten_person_tournament_matches_l285_28590


namespace infinite_primes_l285_28506

theorem infinite_primes : ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p) → 
  ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end infinite_primes_l285_28506


namespace perimeter_special_region_l285_28552

/-- The perimeter of a region bounded by three semicircular arcs and one three-quarter circular arc,
    constructed on the sides of a square with side length 1/π, is equal to 2.25. -/
theorem perimeter_special_region :
  let square_side : ℝ := 1 / Real.pi
  let semicircle_perimeter : ℝ := Real.pi * square_side / 2
  let three_quarter_circle_perimeter : ℝ := 3 * Real.pi * square_side / 4
  let total_perimeter : ℝ := 3 * semicircle_perimeter + three_quarter_circle_perimeter
  total_perimeter = 2.25 := by sorry

end perimeter_special_region_l285_28552


namespace geometric_sequence_property_l285_28555

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The terms a_2, (1/2)a_3, a_1 form an arithmetic sequence. -/
def ArithmeticSubsequence (a : ℕ → ℝ) : Prop :=
  a 2 - (1/2 * a 3) = (1/2 * a 3) - a 1

theorem geometric_sequence_property (a : ℕ → ℝ) :
  GeometricSequence a →
  ArithmeticSubsequence a →
  (a 3 + a 4) / (a 4 + a 5) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end geometric_sequence_property_l285_28555


namespace min_value_of_f_l285_28561

/-- The quadratic expression in x and y with parameter k -/
def f (k x y : ℝ) : ℝ := 9*x^2 - 6*k*x*y + (3*k^2 + 1)*y^2 - 6*x - 6*y + 7

/-- The theorem stating that k = 3 is the unique value for which f has a minimum of 1 -/
theorem min_value_of_f :
  ∃! k : ℝ, (∀ x y : ℝ, f k x y ≥ 1) ∧ (∃ x y : ℝ, f k x y = 1) ∧ k = 3 := by
  sorry

end min_value_of_f_l285_28561


namespace solution_part1_solution_part2_l285_28541

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |2*x - 1|

-- Theorem for part (1)
theorem solution_part1 : {x : ℝ | f x > 3 - 4*x} = {x : ℝ | x > 3/5} := by sorry

-- Theorem for part (2)
theorem solution_part2 : 
  (∀ x : ℝ, f x + |1 - x| ≥ 6*m^2 - 5*m) → 
  -1/6 ≤ m ∧ m ≤ 1 := by sorry

end solution_part1_solution_part2_l285_28541


namespace odd_periodic_function_value_l285_28551

-- Define an odd function with period 2
def is_odd_periodic_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2) = f x)

-- Theorem statement
theorem odd_periodic_function_value (f : ℝ → ℝ) 
  (h : is_odd_periodic_function f) : f 10 = 0 := by
  sorry

end odd_periodic_function_value_l285_28551


namespace rebus_solution_l285_28564

theorem rebus_solution : 
  ∃! (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧
    A = 4 ∧ B = 7 ∧ C = 6 := by
  sorry

end rebus_solution_l285_28564


namespace total_students_on_ride_l285_28589

theorem total_students_on_ride (seats_per_ride : ℕ) (empty_seats : ℕ) (num_rides : ℕ) : 
  seats_per_ride = 15 → empty_seats = 3 → num_rides = 18 →
  (seats_per_ride - empty_seats) * num_rides = 216 := by
  sorry

end total_students_on_ride_l285_28589


namespace positive_slope_implies_positive_correlation_l285_28520

/-- A linear regression model relating variables x and y. -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope
  equation : ∀ x y : ℝ, y = a + b * x

/-- Definition of positive linear correlation between two variables. -/
def positively_correlated (x y : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → y x₁ < y x₂

/-- Theorem stating that a linear regression with positive slope implies positive correlation. -/
theorem positive_slope_implies_positive_correlation
  (model : LinearRegression)
  (h_positive_slope : model.b > 0) :
  positively_correlated (λ x => x) (λ x => model.a + model.b * x) :=
by
  sorry


end positive_slope_implies_positive_correlation_l285_28520


namespace smallest_y_makes_perfect_cube_no_smaller_y_exists_smallest_y_is_perfect_cube_l285_28599

def x : ℕ := 7 * 24 * 54

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def smallest_y : ℕ := 1764

theorem smallest_y_makes_perfect_cube :
  (∀ y : ℕ, y < smallest_y → ¬ is_perfect_cube (x * y)) ∧
  is_perfect_cube (x * smallest_y) := by sorry

theorem no_smaller_y_exists (y : ℕ) (h : y < smallest_y) :
  ¬ is_perfect_cube (x * y) := by sorry

theorem smallest_y_is_perfect_cube :
  is_perfect_cube (x * smallest_y) := by sorry

end smallest_y_makes_perfect_cube_no_smaller_y_exists_smallest_y_is_perfect_cube_l285_28599


namespace min_discriminant_quadratic_trinomial_l285_28537

theorem min_discriminant_quadratic_trinomial (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c ≥ 0) →
  (∀ x, abs x < 1 → a * x^2 + b * x + c ≤ 1 / Real.sqrt (1 - x^2)) →
  b^2 - 4*a*c ≥ -4 ∧ ∃ a' b' c', b'^2 - 4*a'*c' = -4 :=
by sorry

end min_discriminant_quadratic_trinomial_l285_28537


namespace problem_solution_l285_28504

def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x < 4 ↔ x ∈ Set.Ioo (-2) 2) ∧
  (∃ x : ℝ, f x - |a - 1| < 0) ↔ a ∈ Set.Iio (-1) ∪ Set.Ioi 3 :=
by sorry

end problem_solution_l285_28504


namespace pears_left_l285_28588

def initial_pears : ℕ := 35
def given_pears : ℕ := 28

theorem pears_left : initial_pears - given_pears = 7 := by
  sorry

end pears_left_l285_28588


namespace complement_union_theorem_l285_28547

def U : Set ℕ := {x | 0 ≤ x ∧ x < 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {x ∈ U | x^2 + 4 = 5*x}

theorem complement_union_theorem : 
  (U \ A) ∪ (U \ B) = {0, 2, 3, 4, 5} := by sorry

end complement_union_theorem_l285_28547


namespace students_playing_neither_l285_28532

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) 
  (h1 : total = 35)
  (h2 : football = 26)
  (h3 : tennis = 20)
  (h4 : both = 17) :
  total - (football + tennis - both) = 6 :=
by
  sorry

end students_playing_neither_l285_28532


namespace square_garden_area_l285_28512

theorem square_garden_area (s : ℝ) (h1 : s > 0) : 
  (4 * s = 40) → (s^2 = 2 * (4 * s) + 20) → s^2 = 100 := by
  sorry

end square_garden_area_l285_28512


namespace imaginary_part_of_complex_fraction_l285_28543

/-- The imaginary part of 2i / (2 + i^3) is equal to 4/5 -/
theorem imaginary_part_of_complex_fraction :
  Complex.im (2 * Complex.I / (2 + Complex.I ^ 3)) = 4 / 5 := by
  sorry

end imaginary_part_of_complex_fraction_l285_28543


namespace average_price_approx_1_645_l285_28545

/-- Calculate the average price per bottle given the number and prices of large and small bottles, and a discount rate for large bottles. -/
def averagePricePerBottle (largeBottles smallBottles : ℕ) (largePricePerBottle smallPricePerBottle : ℚ) (discountRate : ℚ) : ℚ :=
  let largeCost := largeBottles * largePricePerBottle
  let largeDiscount := largeCost * discountRate
  let discountedLargeCost := largeCost - largeDiscount
  let smallCost := smallBottles * smallPricePerBottle
  let totalCost := discountedLargeCost + smallCost
  let totalBottles := largeBottles + smallBottles
  totalCost / totalBottles

/-- The average price per bottle is approximately $1.645 given the specific conditions. -/
theorem average_price_approx_1_645 :
  let largeBattles := 1325
  let smallBottles := 750
  let largePricePerBottle := 189/100  -- $1.89
  let smallPricePerBottle := 138/100  -- $1.38
  let discountRate := 5/100  -- 5%
  abs (averagePricePerBottle largeBattles smallBottles largePricePerBottle smallPricePerBottle discountRate - 1645/1000) < 1/1000 := by
  sorry


end average_price_approx_1_645_l285_28545


namespace largest_two_digit_prime_factor_of_binomial_l285_28565

theorem largest_two_digit_prime_factor_of_binomial :
  ∃ (p : ℕ), 
    p.Prime ∧ 
    10 ≤ p ∧ p < 100 ∧
    p ∣ Nat.choose 150 75 ∧
    (∀ q : ℕ, q.Prime → 10 ≤ q → q < 100 → q ∣ Nat.choose 150 75 → q ≤ p) ∧
    (∀ q : ℕ, q > p → ¬(q.Prime ∧ 10 ≤ q ∧ q < 100 ∧ q ∣ Nat.choose 150 75)) :=
by
  sorry

#check largest_two_digit_prime_factor_of_binomial

end largest_two_digit_prime_factor_of_binomial_l285_28565


namespace amanda_kimberly_distance_l285_28575

/-- The distance between Amanda's house and Kimberly's house -/
def distance : ℝ := 6

/-- The time Amanda spent walking -/
def walking_time : ℝ := 3

/-- Amanda's walking speed -/
def walking_speed : ℝ := 2

/-- Theorem: The distance between Amanda's house and Kimberly's house is 6 miles -/
theorem amanda_kimberly_distance : distance = walking_time * walking_speed := by
  sorry

end amanda_kimberly_distance_l285_28575


namespace area_of_four_isosceles_triangles_l285_28528

/-- The area of a figure composed of four isosceles triangles -/
theorem area_of_four_isosceles_triangles :
  ∀ (s : ℝ) (θ : ℝ),
  s = 1 →
  θ = 75 * π / 180 →
  2 * s^2 * Real.sin θ = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
  sorry

end area_of_four_isosceles_triangles_l285_28528


namespace laurent_series_expansion_l285_28596

open Complex

/-- The Laurent series expansion of f(z) = (z+2)/(z^2+4z+3) in the ring 2 < |z+1| < +∞ --/
theorem laurent_series_expansion (z : ℂ) (h : 2 < abs (z + 1)) :
  (z + 2) / (z^2 + 4*z + 3) = ∑' k, ((-2)^k + 1) / (z + 1)^(k + 1) := by sorry

end laurent_series_expansion_l285_28596


namespace inequality_solution_set_l285_28539

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x : ℝ, ax^2 - 5*x + b > 0 ↔ x < -1/3 ∨ x > 1/2) →
  (∀ x : ℝ, bx^2 - 5*x + a > 0 ↔ -3 < x ∧ x < 2) :=
by sorry

end inequality_solution_set_l285_28539


namespace remainder_2345678_div_5_l285_28516

theorem remainder_2345678_div_5 : 2345678 % 5 = 3 := by
  sorry

end remainder_2345678_div_5_l285_28516


namespace bottle_caps_difference_l285_28553

/-- Represents the number of bottle caps in various states --/
structure BottleCaps where
  thrown_away : ℕ
  found : ℕ
  final_collection : ℕ

/-- Theorem stating the difference between found and thrown away bottle caps --/
theorem bottle_caps_difference (caps : BottleCaps)
  (h1 : caps.thrown_away = 6)
  (h2 : caps.found = 50)
  (h3 : caps.final_collection = 60) :
  caps.found - caps.thrown_away = 44 := by
  sorry

end bottle_caps_difference_l285_28553


namespace meaningful_fraction_l285_28531

theorem meaningful_fraction (x : ℝ) : (x - 5)⁻¹ ≠ 0 ↔ x ≠ 5 := by sorry

end meaningful_fraction_l285_28531


namespace sum_first_eight_primes_ending_in_3_l285_28518

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def hasUnitsDigitOf3 (n : ℕ) : Prop := n % 10 = 3

def firstEightPrimesEndingIn3 : List ℕ := 
  [3, 13, 23, 43, 53, 73, 83, 103]

theorem sum_first_eight_primes_ending_in_3 :
  (∀ n ∈ firstEightPrimesEndingIn3, isPrime n ∧ hasUnitsDigitOf3 n) →
  (∀ p : ℕ, isPrime p → hasUnitsDigitOf3 p → 
    p ∉ firstEightPrimesEndingIn3 → 
    p > (List.maximum firstEightPrimesEndingIn3).getD 0) →
  List.sum firstEightPrimesEndingIn3 = 394 := by
sorry

end sum_first_eight_primes_ending_in_3_l285_28518


namespace cone_lateral_area_l285_28505

/-- Given a cone with base circumference 4π and slant height 3, its lateral area is 6π. -/
theorem cone_lateral_area (c : ℝ) (l : ℝ) (h1 : c = 4 * Real.pi) (h2 : l = 3) :
  let r := c / (2 * Real.pi)
  π * r * l = 6 * Real.pi := by
  sorry

end cone_lateral_area_l285_28505


namespace area_of_square_B_l285_28536

/-- Given a square A with diagonal x and a square B with diagonal 3x, 
    the area of square B is 9x^2/2 -/
theorem area_of_square_B (x : ℝ) :
  let diag_A := x
  let diag_B := 3 * diag_A
  let area_B := (diag_B^2) / 4
  area_B = 9 * x^2 / 2 := by
  sorry

end area_of_square_B_l285_28536


namespace janet_initial_lives_l285_28566

theorem janet_initial_lives :
  ∀ (initial : ℕ),
  (initial - 16 + 32 = 54) →
  initial = 38 :=
by sorry

end janet_initial_lives_l285_28566


namespace probability_red_ball_in_bag_l285_28586

/-- The probability of drawing a red ball from a bag -/
def probability_red_ball (total_balls : ℕ) (red_balls : ℕ) : ℚ :=
  red_balls / total_balls

/-- Theorem: The probability of drawing a red ball from an opaque bag with 5 balls, 2 of which are red, is 2/5 -/
theorem probability_red_ball_in_bag : probability_red_ball 5 2 = 2 / 5 := by
  sorry

end probability_red_ball_in_bag_l285_28586


namespace delta_cheaper_from_min_posters_gamma_cheaper_or_equal_before_min_posters_l285_28592

/-- Delta Printing Company's pricing function -/
def delta_price (n : ℕ) : ℝ := 40 + 7 * n

/-- Gamma Printing Company's pricing function -/
def gamma_price (n : ℕ) : ℝ := 11 * n

/-- The minimum number of posters for which Delta is cheaper than Gamma -/
def min_posters_for_delta : ℕ := 11

theorem delta_cheaper_from_min_posters :
  ∀ n : ℕ, n ≥ min_posters_for_delta → delta_price n < gamma_price n :=
sorry

theorem gamma_cheaper_or_equal_before_min_posters :
  ∀ n : ℕ, n < min_posters_for_delta → delta_price n ≥ gamma_price n :=
sorry

end delta_cheaper_from_min_posters_gamma_cheaper_or_equal_before_min_posters_l285_28592


namespace max_sum_given_constraints_l285_28529

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 104) 
  (h2 : x * y = 35) : 
  (x + y ≤ Real.sqrt 174) ∧ (∃ (a b : ℝ), a^2 + b^2 = 104 ∧ a * b = 35 ∧ a + b = Real.sqrt 174) :=
by sorry

end max_sum_given_constraints_l285_28529
