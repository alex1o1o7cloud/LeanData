import Mathlib

namespace petpals_center_total_cats_l169_16970

/-- Represents the PetPals Training Center -/
structure PetPalsCenter where
  jump : ℕ
  fetch : ℕ
  spin : ℕ
  jump_fetch : ℕ
  fetch_spin : ℕ
  jump_spin : ℕ
  all_three : ℕ
  none : ℕ

/-- The number of cats in the PetPals Training Center -/
def total_cats (center : PetPalsCenter) : ℕ :=
  center.all_three +
  (center.jump_fetch - center.all_three) +
  (center.fetch_spin - center.all_three) +
  (center.jump_spin - center.all_three) +
  (center.jump - center.jump_fetch - center.jump_spin + center.all_three) +
  (center.fetch - center.jump_fetch - center.fetch_spin + center.all_three) +
  (center.spin - center.jump_spin - center.fetch_spin + center.all_three) +
  center.none

/-- Theorem stating the total number of cats in the PetPals Training Center -/
theorem petpals_center_total_cats :
  ∀ (center : PetPalsCenter),
    center.jump = 60 →
    center.fetch = 40 →
    center.spin = 50 →
    center.jump_fetch = 25 →
    center.fetch_spin = 20 →
    center.jump_spin = 30 →
    center.all_three = 15 →
    center.none = 5 →
    total_cats center = 95 := by
  sorry

end petpals_center_total_cats_l169_16970


namespace sphere_volume_in_specific_cone_l169_16947

/-- A right circular cone with a sphere inscribed inside it. -/
structure ConeWithSphere where
  /-- The diameter of the cone's base in inches. -/
  base_diameter : ℝ
  /-- The vertex angle of the cross-section triangle perpendicular to the base. -/
  vertex_angle : ℝ

/-- Calculate the volume of the inscribed sphere in cubic inches. -/
def sphere_volume (cone : ConeWithSphere) : ℝ :=
  sorry

/-- Theorem stating the volume of the inscribed sphere in the specific cone. -/
theorem sphere_volume_in_specific_cone :
  let cone : ConeWithSphere := { base_diameter := 24, vertex_angle := 90 }
  sphere_volume cone = 288 * Real.pi := by
  sorry

end sphere_volume_in_specific_cone_l169_16947


namespace lisa_patricia_ratio_l169_16943

/-- Represents the money each person has -/
structure Money where
  patricia : ℕ
  lisa : ℕ
  charlotte : ℕ

/-- The conditions of the baseball card problem -/
def baseball_card_problem (m : Money) : Prop :=
  m.patricia = 6 ∧
  m.lisa = 2 * m.charlotte ∧
  m.patricia + m.lisa + m.charlotte = 51

theorem lisa_patricia_ratio (m : Money) :
  baseball_card_problem m →
  m.lisa / m.patricia = 5 := by
  sorry

end lisa_patricia_ratio_l169_16943


namespace part1_part2_part3_l169_16978

/-- Definition of X(n) function -/
def is_X_n_function (f : ℝ → ℝ) : Prop :=
  ∃ (n : ℝ), ∀ (x : ℝ), f (2 * n - x) = f x

/-- Part 1: Prove that |x| and x^2 - x are X(n) functions -/
theorem part1 :
  (is_X_n_function (fun x => |x|)) ∧
  (is_X_n_function (fun x => x^2 - x)) :=
sorry

/-- Part 2: Prove k = -1 for the given parabola conditions -/
theorem part2 (k : ℝ) :
  (∀ x, (x^2 + k - 4) * (x^2 + k - 4) ≤ 0 → 
   ((0 - x)^2 + (k - 4))^2 = 3 * (x^2 + k - 4)^2) →
  k = -1 :=
sorry

/-- Part 3: Prove t = -2 or t = 0 for the given quadratic function conditions -/
theorem part3 (a b t : ℝ) :
  (∀ x, (a*x^2 + b*x - 4) = (a*(2-x)^2 + b*(2-x) - 4)) →
  (a*(-1)^2 + b*(-1) - 4 = 2) →
  (∀ x ∈ Set.Icc t (t+4), a*x^2 + b*x - 4 ≥ -6) →
  (∃ x ∈ Set.Icc t (t+4), a*x^2 + b*x - 4 = 12) →
  (t = -2 ∨ t = 0) :=
sorry

end part1_part2_part3_l169_16978


namespace income_growth_equation_l169_16973

theorem income_growth_equation (x : ℝ) : 
  let initial_income : ℝ := 12000
  let final_income : ℝ := 14520
  initial_income * (1 + x)^2 = final_income := by
  sorry

end income_growth_equation_l169_16973


namespace temperature_difference_product_of_N_values_l169_16917

theorem temperature_difference (B D N : ℝ) : 
  B = D - N →
  (∃ k, k = (D - N + 10) - (D - 4) ∧ (k = 1 ∨ k = -1)) →
  (N = 13 ∨ N = 15) :=
sorry

theorem product_of_N_values : 
  (∃ N₁ N₂ : ℝ, (N₁ = 13 ∧ N₂ = 15) ∧ N₁ * N₂ = 195) :=
sorry

end temperature_difference_product_of_N_values_l169_16917


namespace profit_percentage_l169_16944

theorem profit_percentage (selling_price : ℝ) (cost_price : ℝ) (h : cost_price = 0.81 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (1 - 0.81) / 0.81 * 100 := by
sorry

end profit_percentage_l169_16944


namespace smallest_n_divisibility_l169_16908

theorem smallest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  18 ∣ n^2 ∧ 1152 ∣ n^3 ∧ 
  ∀ (m : ℕ), m > 0 → 18 ∣ m^2 → 1152 ∣ m^3 → n ≤ m :=
by
  use 72
  sorry

end smallest_n_divisibility_l169_16908


namespace min_value_of_sum_l169_16996

theorem min_value_of_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 5) :
  (9 / a + 16 / b + 25 / c) ≥ 30 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 5 ∧ 9 / a₀ + 16 / b₀ + 25 / c₀ = 30 :=
by sorry

end min_value_of_sum_l169_16996


namespace exponential_function_sum_of_extrema_l169_16984

theorem exponential_function_sum_of_extrema (a : ℝ) (h : a > 0) : 
  (∀ x ∈ Set.Icc 0 1, ∃ y, a^x = y) → 
  (a^0 + a^1 = 4/3) → 
  a = 1/3 := by
sorry

end exponential_function_sum_of_extrema_l169_16984


namespace snake_count_theorem_l169_16901

/-- Represents the number of pet owners for different combinations of pets --/
structure PetOwners where
  total : Nat
  onlyDogs : Nat
  onlyCats : Nat
  catsAndDogs : Nat
  catsDogsSnakes : Nat

/-- Given the pet ownership data, proves that the minimum number of snakes is 3
    and that the total number of snakes cannot be determined --/
theorem snake_count_theorem (po : PetOwners)
  (h1 : po.total = 79)
  (h2 : po.onlyDogs = 15)
  (h3 : po.onlyCats = 10)
  (h4 : po.catsAndDogs = 5)
  (h5 : po.catsDogsSnakes = 3) :
  ∃ (minSnakes : Nat), minSnakes = 3 ∧ 
  ¬∃ (totalSnakes : Nat), ∀ (n : Nat), n ≥ minSnakes → n = totalSnakes :=
by sorry

end snake_count_theorem_l169_16901


namespace existence_of_four_numbers_l169_16905

theorem existence_of_four_numbers (x y : ℝ) : 
  ∃ (a₁ a₂ a₃ a₄ : ℝ), x = a₁ + a₂ + a₃ + a₄ ∧ y = 1/a₁ + 1/a₂ + 1/a₃ + 1/a₄ := by
  sorry

end existence_of_four_numbers_l169_16905


namespace tan_five_pi_quarters_l169_16946

theorem tan_five_pi_quarters : Real.tan (5 * π / 4) = 1 := by
  sorry

end tan_five_pi_quarters_l169_16946


namespace base8_subtraction_l169_16992

/-- Converts a base-8 number represented as a list of digits to a natural number. -/
def base8ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a natural number to its base-8 representation as a list of digits. -/
def natToBase8 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- The subtraction operation in base 8. -/
def base8Sub (a b : List Nat) : List Nat :=
  natToBase8 (base8ToNat a - base8ToNat b)

theorem base8_subtraction :
  base8Sub [4, 5, 3] [3, 2, 6] = [1, 2, 5] := by sorry

end base8_subtraction_l169_16992


namespace derivative_equality_l169_16911

-- Define the function f
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + c

-- Define the derivative of f
noncomputable def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + b

-- Theorem statement
theorem derivative_equality (a b c : ℝ) :
  (f' a b 2 = 2) → (f' a b (-2) = 2) := by
  sorry

end derivative_equality_l169_16911


namespace chocolate_problem_l169_16977

/-- The number of chocolates in the cost price -/
def cost_chocolates : ℕ := 24

/-- The gain percentage -/
def gain_percent : ℚ := 1/2

/-- The number of chocolates in the selling price -/
def selling_chocolates : ℕ := 16

theorem chocolate_problem (C S : ℚ) (n : ℕ) 
  (h1 : C > 0) 
  (h2 : S > 0) 
  (h3 : n > 0) 
  (h4 : cost_chocolates * C = n * S) 
  (h5 : gain_percent = (S - C) / C) : 
  n = selling_chocolates := by
  sorry

#check chocolate_problem

end chocolate_problem_l169_16977


namespace composition_equality_condition_l169_16999

theorem composition_equality_condition (m n p q : ℝ) :
  let f : ℝ → ℝ := λ x ↦ m * x + n
  let g : ℝ → ℝ := λ x ↦ p * x + q
  (∀ x, f (g x) = g (f x)) ↔ n * (1 - p) = q * (1 - m) :=
by sorry

end composition_equality_condition_l169_16999


namespace max_abs_z_on_circle_l169_16916

theorem max_abs_z_on_circle (z : ℂ) (h : Complex.abs (z - (3 + 4*I)) = 1) :
  ∃ (w : ℂ), Complex.abs (w - (3 + 4*I)) = 1 ∧ ∀ (u : ℂ), Complex.abs (u - (3 + 4*I)) = 1 → Complex.abs u ≤ Complex.abs w ∧ Complex.abs w = 6 :=
sorry

end max_abs_z_on_circle_l169_16916


namespace original_price_from_decreased_price_l169_16926

/-- 
If an article's price after a 50% decrease is 620 (in some currency unit),
then its original price was 1240 (in the same currency unit).
-/
theorem original_price_from_decreased_price (decreased_price : ℝ) 
  (h : decreased_price = 620) : 
  ∃ (original_price : ℝ), 
    original_price * 0.5 = decreased_price ∧ 
    original_price = 1240 :=
by sorry

end original_price_from_decreased_price_l169_16926


namespace quadratic_equation_solution_l169_16969

theorem quadratic_equation_solution (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 4*x₁ - 2*m + 5 = 0 ∧ 
    x₂^2 - 4*x₂ - 2*m + 5 = 0 ∧
    x₁*x₂ + x₁ + x₂ = m^2 + 6) → 
  m = 1 := by sorry

end quadratic_equation_solution_l169_16969


namespace arithmetic_sequence_a12_l169_16960

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 (a : ℕ → ℚ) :
  ArithmeticSequence a →
  a 7 + a 9 = 16 →
  a 4 = 1 →
  a 12 = 15 := by
sorry

end arithmetic_sequence_a12_l169_16960


namespace A_intersect_B_l169_16919

def A : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {x | ∃ a ∈ A, x = 2*a - 1}

theorem A_intersect_B : A ∩ B = {1, 3} := by
  sorry

end A_intersect_B_l169_16919


namespace unique_solution_l169_16938

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  x^2 - 23*y + 66*z + 612 = 0 ∧
  y^2 + 62*x - 20*z + 296 = 0 ∧
  z^2 - 22*x + 67*y + 505 = 0

/-- The theorem stating that (-20, -22, -23) is the unique solution to the system -/
theorem unique_solution :
  ∃! (x y z : ℝ), system x y z ∧ x = -20 ∧ y = -22 ∧ z = -23 := by
  sorry

end unique_solution_l169_16938


namespace cubic_equation_roots_sum_l169_16909

theorem cubic_equation_roots_sum (a b c : ℝ) : 
  (a^3 - 6*a^2 + 11*a - 6 = 0) → 
  (b^3 - 6*b^2 + 11*b - 6 = 0) → 
  (c^3 - 6*c^2 + 11*c - 6 = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (1/a^3 + 1/b^3 + 1/c^3 = 251/216) := by
sorry

end cubic_equation_roots_sum_l169_16909


namespace loan_amount_calculation_l169_16990

/-- Calculates the total loan amount given the down payment, monthly payment, and loan duration in years. -/
def totalLoanAmount (downPayment : ℕ) (monthlyPayment : ℕ) (years : ℕ) : ℕ :=
  downPayment + monthlyPayment * (years * 12)

/-- Theorem stating that a loan with a $10,000 down payment and $600 monthly payments for 5 years totals $46,000. -/
theorem loan_amount_calculation :
  totalLoanAmount 10000 600 5 = 46000 := by
  sorry

end loan_amount_calculation_l169_16990


namespace inscribed_squares_ratio_l169_16939

/-- 
Given a right triangle with sides 6, 8, and 10, and an inscribed square with side length a 
where one vertex of the square coincides with the right angle of the triangle,
and an isosceles right triangle with legs 6 and 6, and an inscribed square with side length b 
where one side of the square lies on the hypotenuse of the triangle,
the ratio of a to b is √2/3.
-/
theorem inscribed_squares_ratio : 
  ∀ (a b : ℝ),
  (∃ (x y : ℝ), x + y = 10 ∧ x^2 + y^2 = 10^2 ∧ x * y = 48 ∧ a * (x - a) = a * (y - a)) →
  (∃ (z : ℝ), z^2 = 72 ∧ b + b = z) →
  a / b = Real.sqrt 2 / 3 := by
sorry

end inscribed_squares_ratio_l169_16939


namespace union_and_intersection_range_of_m_l169_16937

-- Define the sets A, B, and C
def A : Set ℝ := {x | -4 < x ∧ x < 2}
def B : Set ℝ := {x | x < -5 ∨ x > 1}
def C (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m + 1}

-- Theorem for part (1)
theorem union_and_intersection :
  (A ∪ B = {x | x < -5 ∨ x > -4}) ∧
  (A ∩ (Set.univ \ B) = {x | -4 < x ∧ x ≤ 1}) := by sorry

-- Theorem for part (2)
theorem range_of_m (m : ℝ) :
  (B ∩ C m = ∅) → (m ∈ Set.Icc (-4) 0) := by sorry

end union_and_intersection_range_of_m_l169_16937


namespace angle_position_l169_16950

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- An angle in the 2D plane -/
structure Angle where
  -- We don't need to define the internal structure of an angle for this problem

/-- The terminal side of an angle -/
def terminal_side (α : Angle) : Set Point :=
  sorry -- Definition not needed for the statement

/-- Predicate to check if a point is on the non-negative side of the y-axis -/
def on_nonnegative_y_side (p : Point) : Prop :=
  p.x = 0 ∧ p.y ≥ 0

theorem angle_position (α : Angle) (P : Point) :
  P ∈ terminal_side α →
  P = ⟨0, 3⟩ →
  ∃ (p : Point), p ∈ terminal_side α ∧ on_nonnegative_y_side p :=
sorry

end angle_position_l169_16950


namespace monkey_percentage_after_events_l169_16913

/-- Represents the counts of animals in the tree --/
structure AnimalCounts where
  monkeys : ℕ
  birds : ℕ
  squirrels : ℕ
  cats : ℕ

/-- Calculates the total number of animals --/
def totalAnimals (counts : AnimalCounts) : ℕ :=
  counts.monkeys + counts.birds + counts.squirrels + counts.cats

/-- Applies the events described in the problem --/
def applyEvents (initial : AnimalCounts) : AnimalCounts :=
  { monkeys := initial.monkeys,
    birds := initial.birds - 2 - 2,  -- 2 eaten by monkeys, 2 chased away
    squirrels := initial.squirrels - 1,  -- 1 chased away
    cats := initial.cats }

/-- Calculates the percentage of monkeys after the events --/
def monkeyPercentage (initial : AnimalCounts) : ℚ :=
  let final := applyEvents initial
  (final.monkeys : ℚ) / (totalAnimals final : ℚ) * 100

theorem monkey_percentage_after_events :
  let initial : AnimalCounts := { monkeys := 6, birds := 9, squirrels := 3, cats := 5 }
  monkeyPercentage initial = 100/3 := by sorry

end monkey_percentage_after_events_l169_16913


namespace decimal_arithmetic_proof_l169_16974

theorem decimal_arithmetic_proof : (5.92 + 2.4) - 3.32 = 5.00 := by
  sorry

end decimal_arithmetic_proof_l169_16974


namespace angle_between_polar_lines_eq_arctan_half_l169_16964

/-- The angle between two lines in polar coordinates -/
def angle_between_polar_lines (line1 : ℝ → ℝ → Prop) (line2 : ℝ → ℝ → Prop) : ℝ :=
  sorry

/-- First line equation in polar coordinates -/
def line1 (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ + 2 * Real.sin θ) = 1

/-- Second line equation in polar coordinates -/
def line2 (ρ θ : ℝ) : Prop :=
  ρ * Real.sin θ = 1

/-- Theorem stating the angle between the two given lines -/
theorem angle_between_polar_lines_eq_arctan_half :
  angle_between_polar_lines line1 line2 = Real.arctan (1 / 2) :=
sorry

end angle_between_polar_lines_eq_arctan_half_l169_16964


namespace distance_of_problem_lines_l169_16961

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  direction : ℝ × ℝ

/-- The distance between two parallel lines -/
def distance_between_parallel_lines (lines : ParallelLines) : ℝ :=
  sorry

/-- The specific parallel lines from the problem -/
def problem_lines : ParallelLines :=
  { point1 := (3, -4)
  , point2 := (-1, 1)
  , direction := (2, -5) }

theorem distance_of_problem_lines :
  distance_between_parallel_lines problem_lines = (150 * Real.sqrt 2) / 29 :=
sorry

end distance_of_problem_lines_l169_16961


namespace five_volunteers_four_events_l169_16972

/-- The number of ways to allocate volunteers to events --/
def allocationSchemes (volunteers : ℕ) (events : ℕ) : ℕ :=
  (volunteers.choose 2) * (events.factorial)

/-- Theorem stating the number of allocation schemes for 5 volunteers and 4 events --/
theorem five_volunteers_four_events :
  allocationSchemes 5 4 = 240 := by
  sorry

#eval allocationSchemes 5 4

end five_volunteers_four_events_l169_16972


namespace employee_count_l169_16982

theorem employee_count (avg_salary : ℝ) (salary_increase : ℝ) (manager_salary : ℝ)
  (h1 : avg_salary = 1500)
  (h2 : salary_increase = 600)
  (h3 : manager_salary = 14100) :
  ∃ n : ℕ, 
    (n : ℝ) * avg_salary + manager_salary = ((n : ℝ) + 1) * (avg_salary + salary_increase) ∧
    n = 20 :=
by sorry

end employee_count_l169_16982


namespace parallel_line_equation_l169_16991

/-- The curve function -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 6 * x - 4

theorem parallel_line_equation :
  let P : ℝ × ℝ := (-1, 2)
  let M : ℝ × ℝ := (1, 1)
  let m : ℝ := f' M.1  -- Slope of the tangent line at M
  let line (x y : ℝ) := 2 * x - y + 4 = 0
  (∀ x y, line x y ↔ y - P.2 = m * (x - P.1)) ∧  -- Point-slope form
  (f M.1 = M.2) ∧  -- M is on the curve
  (f' M.1 = m)  -- Slope at M equals the derivative
  := by sorry

end parallel_line_equation_l169_16991


namespace conference_handshakes_count_l169_16918

/-- The number of unique handshakes in a conference with specified conditions -/
def conferenceHandshakes (numCompanies : ℕ) (repsPerCompany : ℕ) : ℕ :=
  let totalPeople := numCompanies * repsPerCompany
  let handshakesPerPerson := totalPeople - repsPerCompany - 1
  (totalPeople * handshakesPerPerson) / 2

/-- Theorem: The number of handshakes in the specified conference is 250 -/
theorem conference_handshakes_count :
  conferenceHandshakes 5 5 = 250 := by
  sorry

#eval conferenceHandshakes 5 5

end conference_handshakes_count_l169_16918


namespace min_value_expression_l169_16954

theorem min_value_expression (n : ℕ+) : 
  (n : ℝ) / 2 + 24 / (n : ℝ) ≥ 7 ∧ 
  ∃ m : ℕ+, (m : ℝ) / 2 + 24 / (m : ℝ) = 7 :=
sorry

end min_value_expression_l169_16954


namespace trig_product_identities_l169_16930

open Real

theorem trig_product_identities (α : ℝ) :
  (1 + sin α = 2 * sin ((π/2 + α)/2) * cos ((π/2 - α)/2)) ∧
  (1 - sin α = 2 * cos ((π/2 + α)/2) * sin ((π/2 - α)/2)) ∧
  (1 + 2 * sin α = 4 * sin ((π/6 + α)/2) * cos ((π/6 - α)/2)) ∧
  (1 - 2 * sin α = 4 * cos ((π/6 + α)/2) * sin ((π/6 - α)/2)) ∧
  (1 + 2 * cos α = 4 * cos ((π/3 + α)/2) * cos ((π/3 - α)/2)) ∧
  (1 - 2 * cos α = -4 * sin ((π/3 + α)/2) * sin ((π/3 - α)/2)) :=
by sorry


end trig_product_identities_l169_16930


namespace solution_satisfies_system_l169_16998

def is_valid_solution (x : ℝ) (a b c d e f : ℝ) : Prop :=
  x ≥ 1 ∧
  a = x ∧ b = x ∧ c = 1/x ∧ d = x ∧ e = x ∧ f = 1/x ∧
  a = max (1/b) (1/c) ∧
  b = max (1/c) (1/d) ∧
  c = max (1/d) (1/e) ∧
  d = max (1/e) (1/f) ∧
  e = max (1/f) (1/a) ∧
  f = max (1/a) (1/b)

theorem solution_satisfies_system :
  ∀ x : ℝ, x > 0 → is_valid_solution x x x (1/x) x x (1/x) :=
by sorry

end solution_satisfies_system_l169_16998


namespace quadratic_expression_equals_724_l169_16904

theorem quadratic_expression_equals_724 
  (x y : ℝ) 
  (h1 : 4 * x + y = 18) 
  (h2 : x + 4 * y = 20) : 
  20 * x^2 + 16 * x * y + 20 * y^2 = 724 :=
by sorry

end quadratic_expression_equals_724_l169_16904


namespace number_of_trees_l169_16932

/-- The number of trees around the house. -/
def n : ℕ := 118

/-- The difference between Alexander's and Timur's starting points. -/
def start_diff : ℕ := 33 - 12

/-- The theorem stating the number of trees around the house. -/
theorem number_of_trees :
  ∃ k : ℕ, n + k = 105 - 12 + 8 ∧ start_diff = 33 - 12 := by
  sorry

end number_of_trees_l169_16932


namespace arc_length_for_45_degree_angle_l169_16923

/-- Given a circle with circumference 90 meters and a central angle of 45°,
    the length of the corresponding arc is 11.25 meters. -/
theorem arc_length_for_45_degree_angle (D : Real) (E F : Real) : 
  D = 90 →  -- circumference of circle D is 90 meters
  (E - F) = 45 * π / 180 →  -- central angle ∠EDF is 45° (converted to radians)
  D * (E - F) / (2 * π) = 11.25 :=  -- length of arc EF
by sorry

end arc_length_for_45_degree_angle_l169_16923


namespace cubic_polynomial_distinct_roots_condition_l169_16949

theorem cubic_polynomial_distinct_roots_condition (p q : ℝ) : 
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 + p*x + q = (x - a) * (x - b) * (x - c))) →
  p < 0 :=
by sorry

end cubic_polynomial_distinct_roots_condition_l169_16949


namespace arithmetic_sequence_ratio_l169_16940

/-- An arithmetic sequence with common difference d -/
def arithmeticSequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_ratio
  (a : ℕ → ℚ) (d : ℚ)
  (hd : d ≠ 0)
  (ha : arithmeticSequence a d)
  (hineq : (a 3)^2 ≠ (a 1) * (a 9)) :
  (a 3) / (a 6) = 1 / 2 := by
sorry

end arithmetic_sequence_ratio_l169_16940


namespace line_b_production_l169_16942

/-- Given three production lines A, B, and C forming an arithmetic sequence,
    prove that Line B produced 4400 units out of a total of 13200 units. -/
theorem line_b_production (total : ℕ) (a b c : ℕ) : 
  total = 13200 →
  a + b + c = total →
  ∃ (d : ℤ), a = b - d ∧ c = b + d →
  b = 4400 := by
  sorry

end line_b_production_l169_16942


namespace divisibility_by_a_squared_l169_16953

theorem divisibility_by_a_squared (a : ℤ) (n : ℕ) :
  ∃ k : ℤ, (a * n - 1) * (a + 1)^n + 1 = a^2 * k := by
  sorry

end divisibility_by_a_squared_l169_16953


namespace point_in_second_quadrant_l169_16980

def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  is_in_second_quadrant (-7 : ℝ) (3 : ℝ) :=
by sorry

end point_in_second_quadrant_l169_16980


namespace difference_of_squares_l169_16963

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end difference_of_squares_l169_16963


namespace inequality_proof_l169_16966

theorem inequality_proof (x y : ℝ) (hx : x < 0) (hy : y < 0) :
  x^4 / y^4 + y^4 / x^4 - x^2 / y^2 - y^2 / x^2 + x / y + y / x ≥ 2 := by
  sorry

end inequality_proof_l169_16966


namespace largest_two_digit_prime_factor_of_180_choose_90_l169_16929

theorem largest_two_digit_prime_factor_of_180_choose_90 : 
  ∃ (p : ℕ), 
    Prime p ∧ 
    10 ≤ p ∧ 
    p < 100 ∧ 
    p ∣ Nat.choose 180 90 ∧ 
    ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ Nat.choose 180 90 → q ≤ p ∧
    p = 59 :=
by sorry

end largest_two_digit_prime_factor_of_180_choose_90_l169_16929


namespace lcm_20_45_75_l169_16988

theorem lcm_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end lcm_20_45_75_l169_16988


namespace first_group_size_l169_16957

/-- Given a work that takes 25 days for some men to complete and 21 days for 50 men to complete,
    prove that the number of men in the first group is 42. -/
theorem first_group_size (days_first : ℕ) (days_second : ℕ) (men_second : ℕ) :
  days_first = 25 →
  days_second = 21 →
  men_second = 50 →
  (men_second * days_second : ℕ) = days_first * (42 : ℕ) :=
by sorry

end first_group_size_l169_16957


namespace number_of_installments_l169_16987

def cash_price : ℕ := 8000
def deposit : ℕ := 3000
def monthly_installment : ℕ := 300
def cash_saving : ℕ := 4000

theorem number_of_installments : 
  (cash_price + cash_saving - deposit) / monthly_installment = 30 := by
  sorry

end number_of_installments_l169_16987


namespace imo_42_inequality_l169_16936

theorem imo_42_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*a*c) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end imo_42_inequality_l169_16936


namespace fraction_problem_l169_16921

theorem fraction_problem (numerator : ℕ) : 
  (numerator : ℚ) / (2 * numerator + 4) = 3 / 7 → numerator = 12 := by
  sorry

end fraction_problem_l169_16921


namespace arithmetic_sequence_problem_l169_16902

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (n : ℕ) :
  (∀ k : ℕ, a (k + 2) - a (k + 1) = a (k + 1) - a k) →  -- arithmetic sequence condition
  a 1 = 1/3 →
  a 2 + a 5 = 4 →
  a n = 33 →
  n = 50 := by
sorry

end arithmetic_sequence_problem_l169_16902


namespace earth_orbit_radius_scientific_notation_l169_16993

theorem earth_orbit_radius_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 149000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.49 ∧ n = 8 := by
  sorry

end earth_orbit_radius_scientific_notation_l169_16993


namespace prism_volume_theorem_l169_16995

def prism_volume (AC_1 PQ phi : ℝ) (sin_phi cos_phi : ℝ) : Prop :=
  AC_1 = 3 ∧ 
  PQ = Real.sqrt 3 ∧ 
  phi = 30 * Real.pi / 180 ∧ 
  sin_phi = 1 / 2 ∧ 
  cos_phi = Real.sqrt 3 / 2 ∧ 
  ∃ (DL PK OK CL AL AC CC_1 : ℝ),
    DL = PK ∧
    DL = 1 / 2 * PQ * sin_phi ∧
    OK = 1 / 2 * PQ * cos_phi ∧
    CL / AL = (AC_1 / 2 - OK) / (AC_1 / 2 + OK) ∧
    AC = CL + AL ∧
    DL ^ 2 = CL * AL ∧
    CC_1 ^ 2 = AC_1 ^ 2 - AC ^ 2 ∧
    AC * DL * CC_1 = Real.sqrt 6 / 2

theorem prism_volume_theorem (AC_1 PQ phi sin_phi cos_phi : ℝ) :
  prism_volume AC_1 PQ phi sin_phi cos_phi → 
  ∃ (V : ℝ), V = Real.sqrt 6 / 2 := by
  sorry

end prism_volume_theorem_l169_16995


namespace quadratic_function_properties_l169_16920

def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 1) - f x = 2 * x + 1) ∧
  (∀ x, f (-x) = f x) ∧
  (f 0 = 1)

theorem quadratic_function_properties (f : ℝ → ℝ) (h : QuadraticFunction f) :
  (∀ x, f x = x^2 + 1) ∧
  (∀ x ∈ Set.Icc (-2) 1, f x ≤ 5) ∧
  (∀ x ∈ Set.Icc (-2) 1, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-2) 1, f x = 5) ∧
  (∃ x ∈ Set.Icc (-2) 1, f x = 1) :=
by sorry

end quadratic_function_properties_l169_16920


namespace surface_area_of_sliced_prism_l169_16914

/-- Right prism with equilateral triangle base -/
structure RightPrism :=
  (height : ℝ)
  (base_side_length : ℝ)

/-- Point on an edge of the prism -/
structure EdgePoint :=
  (distance_ratio : ℝ)

/-- Solid formed by slicing off part of the prism -/
structure SlicedSolid :=
  (prism : RightPrism)
  (point1 : EdgePoint)
  (point2 : EdgePoint)
  (point3 : EdgePoint)

/-- Surface area of the sliced solid -/
def surface_area (solid : SlicedSolid) : ℝ :=
  sorry

theorem surface_area_of_sliced_prism (solid : SlicedSolid) 
  (h1 : solid.prism.height = 24)
  (h2 : solid.prism.base_side_length = 18)
  (h3 : solid.point1.distance_ratio = 1/3)
  (h4 : solid.point2.distance_ratio = 1/3)
  (h5 : solid.point3.distance_ratio = 1/3) :
  surface_area solid = 96 + 9 * Real.sqrt 3 + 9 * Real.sqrt 15 :=
sorry

end surface_area_of_sliced_prism_l169_16914


namespace tv_weekly_cost_l169_16927

/-- Calculate the cost in cents to run a TV for a week -/
theorem tv_weekly_cost (tv_power : ℝ) (daily_usage : ℝ) (electricity_cost : ℝ) : 
  tv_power = 125 →
  daily_usage = 4 →
  electricity_cost = 14 →
  (tv_power * daily_usage * 7 / 1000 * electricity_cost) = 49 := by
  sorry

end tv_weekly_cost_l169_16927


namespace symmetric_points_sum_l169_16912

/-- Given two points A and B symmetric with respect to the y-axis, prove that the sum of their x-coordinates is the negative of the difference of their y-coordinates. -/
theorem symmetric_points_sum (m n : ℝ) : 
  (∃ (A B : ℝ × ℝ), A = (m, -3) ∧ B = (2, n) ∧ 
   (A.1 = -B.1) ∧ (A.2 = B.2)) → 
  m + n = -5 := by
sorry

end symmetric_points_sum_l169_16912


namespace find_n_l169_16985

theorem find_n : ∃ n : ℕ, (2^3 * 8^3 = 2^(2*n)) ∧ n = 6 := by
  sorry

end find_n_l169_16985


namespace ending_number_is_989_l169_16958

/-- A function that counts the number of integers between 0 and n (inclusive) 
    that do not contain the digit 1 in their decimal representation -/
def count_no_one (n : ℕ) : ℕ := sorry

/-- The theorem stating that 989 is the smallest positive integer n such that 
    there are exactly 728 integers between 0 and n (inclusive) that do not 
    contain the digit 1 -/
theorem ending_number_is_989 : 
  (∀ m : ℕ, m < 989 → count_no_one m < 728) ∧ count_no_one 989 = 728 :=
sorry

end ending_number_is_989_l169_16958


namespace locus_of_point_P_l169_16906

/-- The equation of an ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 + y^2/4 = 1

/-- A line with slope 1 -/
def line_slope_1 (x y x' y' : ℝ) : Prop := y = x + (y' - x')

/-- The locus equation -/
def locus (x y : ℝ) : Prop := 148*x^2 + 13*y^2 + 64*x*y - 20 = 0

/-- The theorem statement -/
theorem locus_of_point_P :
  ∀ (x' y' x1 y1 x2 y2 : ℝ),
  ellipse x1 y1 ∧ ellipse x2 y2 ∧  -- A and B are on the ellipse
  line_slope_1 x1 y1 x' y' ∧ line_slope_1 x2 y2 x' y' ∧  -- A, B, and P are on a line with slope 1
  x' = (x1 + 2*x2) / 3 ∧  -- AP = 2PB condition
  x1 < x2 →  -- Ensure A and B are distinct points
  locus x' y' :=
by sorry

end locus_of_point_P_l169_16906


namespace power_negative_two_m_squared_cubed_l169_16924

theorem power_negative_two_m_squared_cubed (m : ℝ) : (-2 * m^2)^3 = -8 * m^6 := by
  sorry

end power_negative_two_m_squared_cubed_l169_16924


namespace original_price_calculation_l169_16955

theorem original_price_calculation (P : ℝ) : 
  (P * (1 - 0.3) * (1 - 0.2) = 1120) → P = 2000 := by
  sorry

end original_price_calculation_l169_16955


namespace max_value_sum_of_roots_l169_16907

theorem max_value_sum_of_roots (a b c : ℝ) 
  (sum_eq_two : a + b + c = 2)
  (a_ge : a ≥ -1/2)
  (b_ge : b ≥ -1)
  (c_ge : c ≥ -2) :
  (∃ x y z : ℝ, x + y + z = 2 ∧ x ≥ -1/2 ∧ y ≥ -1 ∧ z ≥ -2 ∧
    Real.sqrt (4*x + 2) + Real.sqrt (4*y + 4) + Real.sqrt (4*z + 8) = Real.sqrt 66) ∧
  (∀ x y z : ℝ, x + y + z = 2 → x ≥ -1/2 → y ≥ -1 → z ≥ -2 →
    Real.sqrt (4*x + 2) + Real.sqrt (4*y + 4) + Real.sqrt (4*z + 8) ≤ Real.sqrt 66) :=
by sorry

end max_value_sum_of_roots_l169_16907


namespace quadratic_equations_common_root_l169_16915

/-- Given two quadratic equations with a common root, this theorem proves
    properties about the sum and product of the other two roots. -/
theorem quadratic_equations_common_root
  (a b : ℝ) (ha : a < 0) (hb : b < 0) (hab : a ≠ b)
  (h_common : ∃ x₀ : ℝ, x₀^2 + a*x₀ + b = 0 ∧ x₀^2 + b*x₀ + a = 0)
  (x₁ x₂ : ℝ)
  (h_x₁ : x₁^2 + a*x₁ + b = 0)
  (h_x₂ : x₂^2 + b*x₂ + a = 0) :
  (x₁ + x₂ = -1) ∧ (x₁ * x₂ ≤ 1/4) := by
  sorry

end quadratic_equations_common_root_l169_16915


namespace simplify_expression_l169_16933

theorem simplify_expression : 
  Real.sqrt 6 * (Real.sqrt 2 + Real.sqrt 3) - 3 * Real.sqrt (1/3) = Real.sqrt 3 + 3 * Real.sqrt 2 := by
  sorry

end simplify_expression_l169_16933


namespace jack_emails_l169_16934

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 3

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 5

/-- The total number of emails Jack received in the day -/
def total_emails : ℕ := morning_emails + afternoon_emails

theorem jack_emails : total_emails = 8 := by
  sorry

end jack_emails_l169_16934


namespace restaurant_customer_prediction_l169_16951

theorem restaurant_customer_prediction 
  (breakfast_customers : ℕ) 
  (lunch_customers : ℕ) 
  (dinner_customers : ℕ) 
  (h1 : breakfast_customers = 73)
  (h2 : lunch_customers = 127)
  (h3 : dinner_customers = 87) :
  2 * (breakfast_customers + lunch_customers + dinner_customers) = 574 := by
  sorry

end restaurant_customer_prediction_l169_16951


namespace area_of_circumcenter_quadrilateral_area_of_ABCD_proof_l169_16941

/-- The area of the quadrilateral formed by the circumcenters of four equilateral triangles
    erected on the sides of a unit square (one inside, three outside) -/
theorem area_of_circumcenter_quadrilateral : ℝ :=
  let square_side_length : ℝ := 1
  let triangle_side_length : ℝ := 1
  let inside_triangle_count : ℕ := 1
  let outside_triangle_count : ℕ := 3
  (3 + Real.sqrt 3) / 6

/-- Proof of the area of the quadrilateral ABCD -/
theorem area_of_ABCD_proof :
  area_of_circumcenter_quadrilateral = (3 + Real.sqrt 3) / 6 := by
  sorry

end area_of_circumcenter_quadrilateral_area_of_ABCD_proof_l169_16941


namespace ball_difference_l169_16959

theorem ball_difference (blue red : ℕ) 
  (h1 : red - 152 = (blue + 152) + 346) : 
  red - blue = 650 := by
sorry

end ball_difference_l169_16959


namespace composite_expression_l169_16925

theorem composite_expression (n : ℕ) (h : n > 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = 3^(2*n+1) - 2^(2*n+1) - 6^n := by
  sorry

end composite_expression_l169_16925


namespace balloon_count_l169_16967

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 40

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := 41

/-- The total number of blue balloons Joan and Melanie have together -/
def total_balloons : ℕ := joan_balloons + melanie_balloons

theorem balloon_count : total_balloons = 81 := by sorry

end balloon_count_l169_16967


namespace perimeter_of_modified_square_l169_16965

/-- The perimeter of figure ABFCDE formed by cutting a right isosceles triangle from a square and translating it -/
theorem perimeter_of_modified_square (side_length : ℝ) : 
  side_length > 0 →
  4 * side_length = 64 →
  let perimeter_ABFCDE := 4 * side_length + 2 * side_length * Real.sqrt 2
  perimeter_ABFCDE = 64 + 32 * Real.sqrt 2 := by sorry

end perimeter_of_modified_square_l169_16965


namespace shopkeeper_profit_percentage_l169_16986

theorem shopkeeper_profit_percentage 
  (cost_price selling_price_profit selling_price_loss : ℕ)
  (h1 : selling_price_loss = 540)
  (h2 : selling_price_profit = 900)
  (h3 : cost_price = 720)
  (h4 : selling_price_loss = (75 * cost_price) / 100) :
  (selling_price_profit - cost_price) * 100 / cost_price = 25 := by
sorry

end shopkeeper_profit_percentage_l169_16986


namespace fraction_chain_l169_16976

theorem fraction_chain (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 := by
sorry

end fraction_chain_l169_16976


namespace factory_cost_l169_16910

/-- Calculates the total cost of employing all workers for one day --/
def total_cost (total_employees : ℕ) 
  (group1_count : ℕ) (group1_rate : ℚ) (group1_regular_hours : ℕ)
  (group2_count : ℕ) (group2_rate : ℚ) (group2_regular_hours : ℕ)
  (group3_count : ℕ) (group3_rate : ℚ) (group3_regular_hours : ℕ) (group3_flat_rate : ℚ)
  (total_hours : ℕ) : ℚ :=
  let group1_cost := group1_count * (
    group1_rate * group1_regular_hours +
    group1_rate * 1.5 * (total_hours - group1_regular_hours)
  )
  let group2_cost := group2_count * (
    group2_rate * group2_regular_hours +
    group2_rate * 2 * (total_hours - group2_regular_hours)
  )
  let group3_cost := group3_count * (
    group3_rate * group3_regular_hours + group3_flat_rate
  )
  group1_cost + group2_cost + group3_cost

/-- Theorem stating the total cost for the given problem --/
theorem factory_cost : 
  total_cost 500 300 15 8 100 18 10 100 20 8 50 12 = 109200 := by
  sorry

end factory_cost_l169_16910


namespace min_max_problem_l169_16979

theorem min_max_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (xy = 15 → x + y ≥ 2 * Real.sqrt 15 ∧ (x + y = 2 * Real.sqrt 15 ↔ x = Real.sqrt 15 ∧ y = Real.sqrt 15)) ∧
  (x + y = 15 → x * y ≤ 225 / 4 ∧ (x * y = 225 / 4 ↔ x = 15 / 2 ∧ y = 15 / 2)) := by
  sorry

end min_max_problem_l169_16979


namespace octal_arithmetic_l169_16997

/-- Represents a number in base 8 as a list of digits (least significant digit first) -/
def OctalNumber := List Nat

/-- Addition of two octal numbers -/
def octalAdd (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Subtraction of two octal numbers -/
def octalSub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Convert a natural number to its octal representation -/
def toOctal (n : Nat) : OctalNumber :=
  sorry

/-- Convert an octal number to its decimal representation -/
def fromOctal (o : OctalNumber) : Nat :=
  sorry

theorem octal_arithmetic :
  let a := [2, 5, 6]  -- 652₈
  let b := [7, 4, 1]  -- 147₈
  let c := [3, 5]     -- 53₈
  let result := [0, 5] -- 50₈
  octalSub (octalAdd a b) c = result := by
  sorry

end octal_arithmetic_l169_16997


namespace exchange_rate_20_percent_increase_same_digits_l169_16945

/-- Represents an exchange rate as a pair of integers (whole, fraction) -/
structure ExchangeRate where
  whole : ℕ
  fraction : ℕ
  h_fraction : fraction < 100

/-- Checks if two exchange rates have the same digits in different order -/
def same_digits_different_order (x y : ExchangeRate) : Prop := sorry

/-- Calculates the 20% increase of an exchange rate -/
def increase_by_20_percent (x : ExchangeRate) : ExchangeRate := sorry

/-- Main theorem: There exists an exchange rate that, when increased by 20%,
    results in a new rate with the same digits in a different order -/
theorem exchange_rate_20_percent_increase_same_digits :
  ∃ (x : ExchangeRate), same_digits_different_order x (increase_by_20_percent x) := by
  sorry

end exchange_rate_20_percent_increase_same_digits_l169_16945


namespace square_of_binomial_l169_16962

theorem square_of_binomial (k : ℚ) : 
  (∃ t u : ℚ, ∀ x, k * x^2 + 28 * x + 9 = (t * x + u)^2) → k = 196 / 9 := by
  sorry

end square_of_binomial_l169_16962


namespace probability_multiple_of_7_l169_16928

def is_multiple_of_7 (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

def count_pairs (n : ℕ) : ℕ := n.choose 2

theorem probability_multiple_of_7 : 
  let total_pairs := count_pairs 100
  let valid_pairs := total_pairs - count_pairs (100 - 14)
  (valid_pairs : ℚ) / total_pairs = 259 / 990 := by sorry

end probability_multiple_of_7_l169_16928


namespace percentage_difference_l169_16975

theorem percentage_difference (w x y z : ℝ) 
  (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x = 1.2 * y) (hyz : y = 1.2 * z) (hwz : w = 1.152 * z) : 
  w = 0.8 * x := by
sorry

end percentage_difference_l169_16975


namespace point_coordinates_sum_l169_16952

theorem point_coordinates_sum (x : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 3)
  let slope : ℝ := (3 - 0) / (x - 0)
  slope = 3/4 → x + 3 = 7 := by
sorry

end point_coordinates_sum_l169_16952


namespace tea_mixture_price_l169_16931

/-- The price of the first variety of tea in Rs per kg -/
def price_first : ℝ := 126

/-- The price of the second variety of tea in Rs per kg -/
def price_second : ℝ := 135

/-- The price of the third variety of tea in Rs per kg -/
def price_third : ℝ := 175.5

/-- The price of the mixture in Rs per kg -/
def price_mixture : ℝ := 153

/-- The ratio of the first variety in the mixture -/
def ratio_first : ℝ := 1

/-- The ratio of the second variety in the mixture -/
def ratio_second : ℝ := 1

/-- The ratio of the third variety in the mixture -/
def ratio_third : ℝ := 2

/-- The total ratio sum -/
def ratio_total : ℝ := ratio_first + ratio_second + ratio_third

theorem tea_mixture_price :
  (ratio_first * price_first + ratio_second * price_second + ratio_third * price_third) / ratio_total = price_mixture := by
  sorry

end tea_mixture_price_l169_16931


namespace x_squared_is_quadratic_l169_16948

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² = 0 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem: x² = 0 is a quadratic equation -/
theorem x_squared_is_quadratic : is_quadratic_equation f := by
  sorry


end x_squared_is_quadratic_l169_16948


namespace quadratic_roots_relation_l169_16971

theorem quadratic_roots_relation (p q : ℝ) : 
  (∀ x, x^2 - p^2*x + p*q = 0 ↔ ∃ y, y^2 + p*y + q = 0 ∧ x = y + 1) →
  ((p = -1 ∧ q = -1) ∨ (p = 2 ∧ q = -1)) := by
  sorry

end quadratic_roots_relation_l169_16971


namespace right_triangle_hypotenuse_l169_16968

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → c = 10 :=
by
  sorry

end right_triangle_hypotenuse_l169_16968


namespace inequality_proof_l169_16900

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 := by
  sorry

end inequality_proof_l169_16900


namespace farm_feet_count_l169_16981

/-- Represents a farm with hens and cows -/
structure Farm where
  total_heads : ℕ
  num_hens : ℕ
  hen_feet : ℕ
  cow_feet : ℕ

/-- Calculates the total number of feet on the farm -/
def total_feet (f : Farm) : ℕ :=
  f.num_hens * f.hen_feet + (f.total_heads - f.num_hens) * f.cow_feet

/-- Theorem: Given a farm with 48 total animals, 24 hens, 2 feet per hen, and 4 feet per cow, 
    the total number of feet is 144 -/
theorem farm_feet_count : 
  ∀ (f : Farm), f.total_heads = 48 → f.num_hens = 24 → f.hen_feet = 2 → f.cow_feet = 4 
  → total_feet f = 144 := by
  sorry


end farm_feet_count_l169_16981


namespace line_slope_and_intercept_l169_16956

/-- Given a line with equation 3x + 4y + 5 = 0, prove its slope is -3/4 and y-intercept is -5/4 -/
theorem line_slope_and_intercept :
  let line := {(x, y) : ℝ × ℝ | 3 * x + 4 * y + 5 = 0}
  ∃ m b : ℝ, m = -3/4 ∧ b = -5/4 ∧ ∀ x y : ℝ, (x, y) ∈ line ↔ y = m * x + b :=
sorry

end line_slope_and_intercept_l169_16956


namespace probability_standard_deck_l169_16922

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (face_cards : Nat)
  (number_cards : Nat)

/-- Define a standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    face_cards := 12,
    number_cards := 40 }

/-- Calculate the probability of drawing a face card first and a number card second -/
def probability_face_then_number (d : Deck) : Rat :=
  (d.face_cards * d.number_cards : Rat) / (d.total_cards * (d.total_cards - 1))

/-- Theorem stating the probability for a standard deck -/
theorem probability_standard_deck :
  probability_face_then_number standard_deck = 40 / 221 := by
  sorry


end probability_standard_deck_l169_16922


namespace reciprocal_statements_l169_16983

def reciprocal (n : ℕ+) : ℚ := 1 / n.val

theorem reciprocal_statements : 
  (¬(reciprocal 4 + reciprocal 8 = reciprocal 12)) ∧
  (¬(reciprocal 9 - reciprocal 3 = reciprocal 6)) ∧
  (reciprocal 3 * reciprocal 9 = reciprocal 27) ∧
  ((reciprocal 15) / (reciprocal 3) = reciprocal 5) :=
by sorry

end reciprocal_statements_l169_16983


namespace square_sum_equals_four_l169_16903

theorem square_sum_equals_four (x y : ℝ) (h : x^2 + y^2 + x^2*y^2 - 4*x*y + 1 = 0) : 
  (x + y)^2 = 4 := by
  sorry

end square_sum_equals_four_l169_16903


namespace smallest_denominator_between_fractions_l169_16935

theorem smallest_denominator_between_fractions :
  ∀ (a b : ℕ), 
    a > 0 → b > 0 →
    (a : ℚ) / b > 6 / 17 →
    (a : ℚ) / b < 9 / 25 →
    b ≥ 14 :=
by
  sorry

end smallest_denominator_between_fractions_l169_16935


namespace same_color_probability_l169_16989

/-- The number of pairs of shoes -/
def num_pairs : ℕ := 6

/-- The total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes to be selected -/
def selected_shoes : ℕ := 2

/-- The probability of selecting two shoes of the same color -/
theorem same_color_probability :
  (num_pairs : ℚ) / (total_shoes.choose selected_shoes) = 1 / 11 := by
  sorry

end same_color_probability_l169_16989


namespace prank_week_combinations_l169_16994

/-- The number of available helpers for each day of the week -/
def helpers_per_day : List Nat := [1, 2, 3, 4, 1]

/-- The total number of possible combinations of helpers throughout the week -/
def total_combinations : Nat := List.prod helpers_per_day

/-- Theorem stating that the total number of combinations is 24 -/
theorem prank_week_combinations :
  total_combinations = 24 := by
  sorry

end prank_week_combinations_l169_16994
