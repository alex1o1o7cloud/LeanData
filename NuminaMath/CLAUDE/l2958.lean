import Mathlib

namespace difference_of_squares_l2958_295872

theorem difference_of_squares (x : ℝ) : 9 - 4 * x^2 = (3 - 2*x) * (3 + 2*x) := by
  sorry

end difference_of_squares_l2958_295872


namespace park_breadth_l2958_295877

/-- The breadth of a rectangular park given its perimeter and length -/
theorem park_breadth (perimeter length breadth : ℝ) : 
  perimeter = 1000 →
  length = 300 →
  perimeter = 2 * (length + breadth) →
  breadth = 200 := by
sorry

end park_breadth_l2958_295877


namespace delta_airlines_discount_percentage_l2958_295862

theorem delta_airlines_discount_percentage 
  (delta_price : ℝ) 
  (united_price : ℝ) 
  (united_discount : ℝ) 
  (price_difference : ℝ) :
  delta_price = 850 →
  united_price = 1100 →
  united_discount = 0.3 →
  price_difference = 90 →
  let united_discounted_price := united_price * (1 - united_discount)
  let delta_discounted_price := united_discounted_price - price_difference
  let delta_discount_amount := delta_price - delta_discounted_price
  let delta_discount_percentage := delta_discount_amount / delta_price
  delta_discount_percentage = 0.2 := by sorry

end delta_airlines_discount_percentage_l2958_295862


namespace arithmetic_sequence_max_sum_l2958_295822

/-- An arithmetic sequence with common difference -2 and S_3 = 21 reaches its maximum sum at n = 5 -/
theorem arithmetic_sequence_max_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n, a (n + 1) - a n = -2) →  -- Common difference is -2
  S 3 = 21 →                     -- S_3 = 21
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Definition of S_n for arithmetic sequence
  (∃ m, ∀ k, S k ≤ S m) →       -- S_n has a maximum value
  (∀ k, S k ≤ S 5) :=            -- The maximum occurs at n = 5
by sorry

end arithmetic_sequence_max_sum_l2958_295822


namespace solution_set_abs_inequality_l2958_295884

theorem solution_set_abs_inequality (x : ℝ) :
  (|2*x + 3| < 1) ↔ (-2 < x ∧ x < -1) :=
sorry

end solution_set_abs_inequality_l2958_295884


namespace students_in_front_of_yuna_l2958_295814

/-- Given a line of students with Yuna somewhere in the line, this theorem
    proves the number of students in front of Yuna. -/
theorem students_in_front_of_yuna 
  (total_students : ℕ) 
  (students_behind_yuna : ℕ) 
  (h1 : total_students = 25)
  (h2 : students_behind_yuna = 9) :
  total_students - (students_behind_yuna + 1) = 15 :=
by sorry

end students_in_front_of_yuna_l2958_295814


namespace jingJing_bought_four_notebooks_l2958_295830

/-- Represents the purchase of stationery items -/
structure StationeryPurchase where
  carbonPens : ℕ
  notebooks : ℕ
  pencilCases : ℕ

/-- Calculates the total cost of a stationery purchase -/
def totalCost (p : StationeryPurchase) : ℚ :=
  1.8 * p.carbonPens + 3.5 * p.notebooks + 4.2 * p.pencilCases

/-- Theorem stating that Jing Jing bought 4 notebooks -/
theorem jingJing_bought_four_notebooks :
  ∃ (p : StationeryPurchase),
    p.carbonPens > 0 ∧
    p.notebooks > 0 ∧
    p.pencilCases > 0 ∧
    totalCost p = 20 ∧
    p.notebooks = 4 :=
by sorry

end jingJing_bought_four_notebooks_l2958_295830


namespace museum_trip_buses_l2958_295826

/-- The number of people on the first bus -/
def first_bus : ℕ := 12

/-- The number of people on the second bus -/
def second_bus : ℕ := 2 * first_bus

/-- The number of people on the third bus -/
def third_bus : ℕ := second_bus - 6

/-- The number of people on the fourth bus -/
def fourth_bus : ℕ := first_bus + 9

/-- The total number of people going to the museum -/
def total_people : ℕ := 75

/-- The number of buses hired -/
def num_buses : ℕ := 4

theorem museum_trip_buses :
  first_bus + second_bus + third_bus + fourth_bus = total_people ∧
  num_buses = 4 := by sorry

end museum_trip_buses_l2958_295826


namespace roots_sum_powers_l2958_295833

theorem roots_sum_powers (c d : ℝ) : 
  c^2 - 5*c + 6 = 0 → d^2 - 5*d + 6 = 0 → c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 503 := by
  sorry

end roots_sum_powers_l2958_295833


namespace equation_solutions_l2958_295837

theorem equation_solutions :
  (∃ x : ℚ, 1 - 1 / (x - 5) = x / (x + 5) ∧ x = 15 / 2) ∧
  (∃ x : ℚ, 3 / (x - 1) - 2 / (x + 1) = 1 / (x^2 - 1) ∧ x = -4) := by
  sorry

end equation_solutions_l2958_295837


namespace max_tickets_buyable_l2958_295831

def regular_price : ℝ := 15
def discount_threshold : ℕ := 6
def discount_rate : ℝ := 0.1
def budget : ℝ := 120

def discounted_price : ℝ := regular_price * (1 - discount_rate)

def cost (n : ℕ) : ℝ :=
  if n ≤ discount_threshold then n * regular_price
  else n * discounted_price

theorem max_tickets_buyable :
  ∀ n : ℕ, cost n ≤ budget → n ≤ 8 ∧ cost 8 ≤ budget :=
sorry

end max_tickets_buyable_l2958_295831


namespace solution_t_l2958_295894

theorem solution_t : ∃ t : ℝ, (1 / (t + 2) + 2 * t / (t + 2) - 3 / (t + 2) = 1) ∧ t = 4 := by
  sorry

end solution_t_l2958_295894


namespace complex_subtraction_l2958_295871

theorem complex_subtraction (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 2 + I) :
  a - 3*b = -1 - 6*I := by
  sorry

end complex_subtraction_l2958_295871


namespace unique_number_with_three_prime_factors_l2958_295852

theorem unique_number_with_three_prime_factors (x n : ℕ) :
  x = 9^n - 1 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 7 ∧ q ≠ 7 ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ x → (r = p ∨ r = q ∨ r = 7))) →
  7 ∣ x →
  x = 728 :=
by sorry

end unique_number_with_three_prime_factors_l2958_295852


namespace perpendicular_lines_parallel_l2958_295846

-- Define the concept of a line in a plane
def Line : Type := ℝ × ℝ → Prop

-- Define perpendicularity relation between lines
def Perpendicular (l1 l2 : Line) : Prop := sorry

-- Define parallel relation between lines
def Parallel (l1 l2 : Line) : Prop := sorry

-- The main theorem
theorem perpendicular_lines_parallel (a b c : Line) :
  Perpendicular a b → Perpendicular c b → Parallel a c := by sorry

end perpendicular_lines_parallel_l2958_295846


namespace exponential_inequality_l2958_295853

open Real

theorem exponential_inequality (f : ℝ → ℝ) (h : ∀ x, f x = exp x) :
  (∀ a, (∀ x, f x ≥ exp 1 * x + a) ↔ a ≤ 0) := by
  sorry

end exponential_inequality_l2958_295853


namespace larger_number_proof_l2958_295881

theorem larger_number_proof (x y : ℝ) (h1 : x > y) (h2 : x - y = 3) (h3 : x^2 - y^2 = 39) : x = 8 := by
  sorry

end larger_number_proof_l2958_295881


namespace sum_of_fourth_powers_l2958_295817

theorem sum_of_fourth_powers (a b : ℝ) 
  (h1 : a^2 - b^2 = 10) 
  (h2 : a * b = 5) : 
  a^4 + b^4 = 150 := by
sorry

end sum_of_fourth_powers_l2958_295817


namespace race_runners_count_l2958_295890

theorem race_runners_count : ∃ n : ℕ, 
  n > 5 ∧ 
  5 * 8 + (n - 5) * 10 = 70 ∧ 
  n = 8 := by
sorry

end race_runners_count_l2958_295890


namespace fraction_zero_implies_x_equals_three_l2958_295899

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (x^2 - 9) / (x + 3) = 0 ∧ x + 3 ≠ 0 → x = 3 := by
sorry

end fraction_zero_implies_x_equals_three_l2958_295899


namespace melanie_dimes_count_l2958_295858

-- Define the initial number of dimes and the amounts given by family members
def initial_dimes : ℝ := 19
def dad_gave : ℝ := 39.5
def mom_gave : ℝ := 25.25
def brother_gave : ℝ := 15.75

-- Define the total number of dimes
def total_dimes : ℝ := initial_dimes + dad_gave + mom_gave + brother_gave

-- Theorem to prove
theorem melanie_dimes_count : total_dimes = 99.5 := by
  sorry

end melanie_dimes_count_l2958_295858


namespace modulus_of_z_l2958_295816

def complex_number (a b : ℝ) : ℂ := a + b * Complex.I

theorem modulus_of_z : 
  let z : ℂ := (7 - Complex.I) / (1 + Complex.I)
  Complex.abs z = 5 := by sorry

end modulus_of_z_l2958_295816


namespace quotient_change_l2958_295855

theorem quotient_change (A B : ℝ) (h : A / B = 0.514) : 
  (10 * A) / (B / 100) = 514 := by
sorry

end quotient_change_l2958_295855


namespace solve_equation_l2958_295835

theorem solve_equation (a : ℚ) (h : 2 * a + a / 2 = 9 / 2) : a = 9 / 5 := by
  sorry

end solve_equation_l2958_295835


namespace problem_solution_l2958_295876

theorem problem_solution (a b x y : ℝ) 
  (eq1 : 2*a*x + 2*b*y = 6)
  (eq2 : 3*a*x^2 + 3*b*y^2 = 21)
  (eq3 : 4*a*x^3 + 4*b*y^3 = 64)
  (eq4 : 5*a*x^4 + 5*b*y^4 = 210) :
  6*a*x^5 + 6*b*y^5 = 5372 := by
sorry

end problem_solution_l2958_295876


namespace real_roots_of_p_l2958_295896

/-- The polynomial under consideration -/
def p (x : ℝ) : ℝ := x^5 - 3*x^4 - x^2 + 3*x

/-- The set of real roots of the polynomial -/
def root_set : Set ℝ := {0, 1, 3}

/-- Theorem stating that root_set contains exactly the real roots of p -/
theorem real_roots_of_p :
  ∀ x : ℝ, x ∈ root_set ↔ p x = 0 :=
sorry

end real_roots_of_p_l2958_295896


namespace truncated_tetrahedron_lateral_area_l2958_295809

/-- Given a truncated tetrahedron with base area A₁, top area A₂ (where A₂ ≤ A₁),
    and sum of lateral face areas P, if the solid can be cut by a plane parallel
    to the base such that a sphere can be inscribed in each of the resulting sections,
    then P = (√A₁ + √A₂)(⁴√A₁ + ⁴√A₂)² -/
theorem truncated_tetrahedron_lateral_area
  (A₁ A₂ P : ℝ)
  (h₁ : 0 < A₁)
  (h₂ : 0 < A₂)
  (h₃ : A₂ ≤ A₁)
  (h₄ : ∃ (A : ℝ), 0 < A ∧ A < A₁ ∧ A > A₂ ∧
    ∃ (R₁ R₂ : ℝ), 0 < R₁ ∧ 0 < R₂ ∧
      A = Real.sqrt (A₁ * A₂) ∧
      (A / A₂) = (A₁ / A) ∧ (A / A₂) = (R₁ / R₂)^2) :
  P = (Real.sqrt A₁ + Real.sqrt A₂) * (Real.sqrt (Real.sqrt A₁) + Real.sqrt (Real.sqrt A₂))^2 := by
sorry

end truncated_tetrahedron_lateral_area_l2958_295809


namespace max_sum_is_27_l2958_295812

/-- Represents the arrangement of numbers in the grid -/
structure Arrangement where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ

/-- The set of available numbers -/
def availableNumbers : Finset ℕ := {2, 5, 8, 11, 14}

/-- Checks if an arrangement is valid according to the problem conditions -/
def isValidArrangement (arr : Arrangement) : Prop :=
  (arr.a ∈ availableNumbers) ∧
  (arr.b ∈ availableNumbers) ∧
  (arr.c ∈ availableNumbers) ∧
  (arr.d ∈ availableNumbers) ∧
  (arr.e ∈ availableNumbers) ∧
  (arr.f ∈ availableNumbers) ∧
  (arr.a + arr.b + arr.e = arr.c + arr.d + arr.f) ∧
  (arr.a + arr.c = arr.b + arr.d) ∧
  (arr.a + arr.c = arr.e + arr.f)

/-- The theorem to be proven -/
theorem max_sum_is_27 :
  ∀ (arr : Arrangement), isValidArrangement arr →
  (arr.a + arr.b + arr.e ≤ 27 ∧ arr.c + arr.d + arr.f ≤ 27) :=
by sorry

end max_sum_is_27_l2958_295812


namespace min_value_x_plus_y_l2958_295828

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y = x * y) :
  x + y ≥ 18 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + 8 * y = x * y ∧ x + y = 18 := by
  sorry

end min_value_x_plus_y_l2958_295828


namespace tax_free_items_cost_l2958_295818

/-- Given a total spend, sales tax, and tax rate, calculate the cost of tax-free items -/
def cost_of_tax_free_items (total_spend : ℚ) (sales_tax : ℚ) (tax_rate : ℚ) : ℚ :=
  total_spend - sales_tax / tax_rate

/-- Theorem: Given the specific values from the problem, the cost of tax-free items is 22 rupees -/
theorem tax_free_items_cost :
  let total_spend : ℚ := 25
  let sales_tax : ℚ := 30 / 100 -- 30 paise = 0.30 rupees
  let tax_rate : ℚ := 10 / 100 -- 10%
  cost_of_tax_free_items total_spend sales_tax tax_rate = 22 := by
  sorry

#eval cost_of_tax_free_items 25 (30/100) (10/100)

end tax_free_items_cost_l2958_295818


namespace g_five_equals_one_l2958_295878

def g_property (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x - y) = g x * g y) ∧
  (∀ x : ℝ, g x ≠ 0) ∧
  (∀ x : ℝ, g x = g (-x))

theorem g_five_equals_one (g : ℝ → ℝ) (h : g_property g) : g 5 = 1 := by
  sorry

end g_five_equals_one_l2958_295878


namespace characterization_of_functions_l2958_295850

-- Define the property P for a function f
def satisfies_property (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, n^2 + 4 * f n = (f (f n))^2

-- Define the three types of functions
def type1 (f : ℤ → ℤ) : Prop :=
  ∀ x : ℤ, f x = 1 + x

def type2 (f : ℤ → ℤ) : Prop :=
  ∃ a : ℤ, (∀ x ≤ a, f x = 1 - x) ∧ (∀ x > a, f x = 1 + x)

def type3 (f : ℤ → ℤ) : Prop :=
  f 0 = 0 ∧ (∀ x < 0, f x = 1 - x) ∧ (∀ x > 0, f x = 1 + x)

-- The main theorem
theorem characterization_of_functions (f : ℤ → ℤ) :
  satisfies_property f ↔ type1 f ∨ type2 f ∨ type3 f :=
sorry

end characterization_of_functions_l2958_295850


namespace fraction_simplification_l2958_295829

theorem fraction_simplification :
  (1 - 2 + 4 - 8 + 16 - 32 + 64) / (2 - 4 + 8 - 16 + 32 - 64 + 128) = 1 / 2 := by
  sorry

end fraction_simplification_l2958_295829


namespace banana_orange_equivalence_l2958_295866

/-- Given that 3/4 of 12 bananas are worth 9 oranges, 
    prove that 1/3 of 6 bananas are worth 2 oranges -/
theorem banana_orange_equivalence (banana orange : ℚ) 
  (h : (3/4 : ℚ) * 12 * banana = 9 * orange) : 
  (1/3 : ℚ) * 6 * banana = 2 * orange := by
  sorry

end banana_orange_equivalence_l2958_295866


namespace sum_of_specific_numbers_l2958_295898

theorem sum_of_specific_numbers : 7.52 + 12.23 = 19.75 := by
  sorry

end sum_of_specific_numbers_l2958_295898


namespace jackson_hermit_crabs_l2958_295848

/-- Given the conditions of Jackson's souvenir collection, prove that he collected 45 hermit crabs. -/
theorem jackson_hermit_crabs :
  ∀ (hermit_crabs spiral_shells starfish : ℕ),
  spiral_shells = 3 * hermit_crabs →
  starfish = 2 * spiral_shells →
  hermit_crabs + spiral_shells + starfish = 450 →
  hermit_crabs = 45 := by
sorry

end jackson_hermit_crabs_l2958_295848


namespace max_planes_15_points_l2958_295885

/-- The number of points in the space -/
def n : ℕ := 15

/-- A function to calculate the number of combinations of n things taken k at a time -/
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The maximum number of unique planes determined by n points in general position -/
def max_planes (n : ℕ) : ℕ := combination n 3

theorem max_planes_15_points :
  max_planes n = 455 :=
sorry

end max_planes_15_points_l2958_295885


namespace vector_expression_simplification_l2958_295889

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem vector_expression_simplification :
  (1/2 : ℝ) • (2 • a + 8 • b) - (4 • a - 2 • b) = 6 • b - 3 • a := by sorry

end vector_expression_simplification_l2958_295889


namespace sqrt_product_plus_one_equals_379_l2958_295856

theorem sqrt_product_plus_one_equals_379 :
  Real.sqrt ((21 : ℝ) * 20 * 19 * 18 + 1) = 379 := by
  sorry

end sqrt_product_plus_one_equals_379_l2958_295856


namespace common_intersection_point_l2958_295860

-- Define a type for points in a plane
variable {Point : Type}

-- Define a type for half-planes
variable {HalfPlane : Type}

-- Define a function to check if a point is in a half-plane
variable (in_half_plane : Point → HalfPlane → Prop)

-- Define a set of half-planes
variable {S : Set HalfPlane}

-- Theorem statement
theorem common_intersection_point 
  (h : ∀ (a b c : HalfPlane), a ∈ S → b ∈ S → c ∈ S → 
    ∃ (p : Point), in_half_plane p a ∧ in_half_plane p b ∧ in_half_plane p c) :
  ∃ (p : Point), ∀ (h : HalfPlane), h ∈ S → in_half_plane p h :=
sorry

end common_intersection_point_l2958_295860


namespace ab_value_l2958_295854

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 := by
  sorry

end ab_value_l2958_295854


namespace inequalities_for_positive_sum_two_l2958_295859

theorem inequalities_for_positive_sum_two (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : 
  a^2 + b^2 ≥ 2 ∧ 1/a + 1/b ≥ 2 := by sorry

end inequalities_for_positive_sum_two_l2958_295859


namespace soda_comparison_l2958_295806

theorem soda_comparison (J : ℝ) (L A : ℝ) 
  (h1 : L = J * 1.5)  -- Liliane has 50% more soda than Jacqueline
  (h2 : A = J * 1.25) -- Alice has 25% more soda than Jacqueline
  : L = A * 1.2       -- Liliane has 20% more soda than Alice
:= by sorry

end soda_comparison_l2958_295806


namespace soap_cost_theorem_l2958_295849

/-- Calculates the cost of soap for a year given the duration and price of a single bar -/
def soap_cost_for_year (months_per_bar : ℕ) (price_per_bar : ℚ) : ℚ :=
  (12 / months_per_bar) * price_per_bar

/-- Theorem stating that for soap lasting 2 months and costing $8.00, the yearly cost is $48.00 -/
theorem soap_cost_theorem : soap_cost_for_year 2 8 = 48 := by
  sorry

#eval soap_cost_for_year 2 8

end soap_cost_theorem_l2958_295849


namespace vector_coplanarity_theorem_point_coplanarity_theorem_l2958_295887

/-- A vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition of coplanarity for vectors -/
def coplanar_vectors (a b p : Vector3D) : Prop :=
  ∃ (x y : ℝ), p = Vector3D.mk (x * a.x + y * b.x) (x * a.y + y * b.y) (x * a.z + y * b.z)

/-- Definition of coplanarity for points -/
def coplanar_points (M A B P : Point3D) : Prop :=
  ∃ (x y : ℝ), 
    P.x - M.x = x * (A.x - M.x) + y * (B.x - M.x) ∧
    P.y - M.y = x * (A.y - M.y) + y * (B.y - M.y) ∧
    P.z - M.z = x * (A.z - M.z) + y * (B.z - M.z)

theorem vector_coplanarity_theorem (a b p : Vector3D) :
  (∃ (x y : ℝ), p = Vector3D.mk (x * a.x + y * b.x) (x * a.y + y * b.y) (x * a.z + y * b.z)) →
  coplanar_vectors a b p :=
by sorry

theorem point_coplanarity_theorem (M A B P : Point3D) :
  (∃ (x y : ℝ), 
    P.x - M.x = x * (A.x - M.x) + y * (B.x - M.x) ∧
    P.y - M.y = x * (A.y - M.y) + y * (B.y - M.y) ∧
    P.z - M.z = x * (A.z - M.z) + y * (B.z - M.z)) →
  coplanar_points M A B P :=
by sorry

end vector_coplanarity_theorem_point_coplanarity_theorem_l2958_295887


namespace system_equations_solutions_l2958_295891

theorem system_equations_solutions (a x y : ℝ) : 
  (x - y = 2*a + 1 ∧ 2*x + 3*y = 9*a - 8) →
  ((x = y → a = -1/2) ∧
   (x > 0 ∧ y < 0 ∧ x + y = 0 → a = 3/4)) := by
  sorry

end system_equations_solutions_l2958_295891


namespace phone_time_proof_l2958_295843

/-- 
Given a person who spends time on the phone for 5 days, 
doubling the time each day after the first, 
and spending a total of 155 minutes,
prove that they spent 5 minutes on the first day.
-/
theorem phone_time_proof (x : ℝ) : 
  x + 2*x + 4*x + 8*x + 16*x = 155 → x = 5 := by
  sorry

end phone_time_proof_l2958_295843


namespace inequality_preservation_l2958_295823

theorem inequality_preservation (a b : ℝ) (h : a > b) : a - 5 > b - 5 := by
  sorry

end inequality_preservation_l2958_295823


namespace largest_integer_prime_abs_quadratic_l2958_295808

theorem largest_integer_prime_abs_quadratic : 
  ∃ (x : ℤ), (∀ y : ℤ, y > x → ¬ Nat.Prime (Int.natAbs (4*y^2 - 39*y + 35))) ∧ 
  Nat.Prime (Int.natAbs (4*x^2 - 39*x + 35)) ∧ x = 6 := by
  sorry

end largest_integer_prime_abs_quadratic_l2958_295808


namespace xiao_zhang_four_vcd_probability_l2958_295870

/-- Represents the number of VCD and DVD discs for each person -/
structure DiscCount where
  vcd : ℕ
  dvd : ℕ

/-- The initial disc counts for Xiao Zhang and Xiao Wang -/
def initial_counts : DiscCount × DiscCount :=
  (⟨4, 3⟩, ⟨2, 1⟩)

/-- The total number of discs -/
def total_discs : ℕ :=
  (initial_counts.1.vcd + initial_counts.1.dvd +
   initial_counts.2.vcd + initial_counts.2.dvd)

/-- Theorem stating the probability of Xiao Zhang ending up with exactly 4 VCD discs -/
theorem xiao_zhang_four_vcd_probability :
  let (zhang, wang) := initial_counts
  let p_vcd_exchange := (zhang.vcd * wang.vcd : ℚ) / ((total_discs * (total_discs - 1)) / 2 : ℚ)
  let p_dvd_exchange := (zhang.dvd * wang.dvd : ℚ) / ((total_discs * (total_discs - 1)) / 2 : ℚ)
  p_vcd_exchange + p_dvd_exchange = 11 / 21 := by
  sorry

end xiao_zhang_four_vcd_probability_l2958_295870


namespace age_difference_l2958_295803

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 16) : A - C = 16 := by
  sorry

end age_difference_l2958_295803


namespace quadratic_always_positive_implies_a_greater_than_one_l2958_295886

theorem quadratic_always_positive_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + a > 0) → a > 1 := by
  sorry

end quadratic_always_positive_implies_a_greater_than_one_l2958_295886


namespace solve_exponential_equation_l2958_295895

theorem solve_exponential_equation (x : ℝ) :
  3^(x - 1) = (1 : ℝ) / 9 → x = -1 := by
  sorry

end solve_exponential_equation_l2958_295895


namespace wrappers_found_at_park_l2958_295811

/-- Represents the number of bottle caps Danny found at the park. -/
def bottle_caps_found : ℕ := 58

/-- Represents the number of wrappers Danny now has in his collection. -/
def wrappers_now : ℕ := 11

/-- Represents the number of bottle caps Danny now has in his collection. -/
def bottle_caps_now : ℕ := 12

/-- Represents the difference between bottle caps and wrappers Danny has now. -/
def cap_wrapper_difference : ℕ := 1

/-- Proves that the number of wrappers Danny found at the park is 11. -/
theorem wrappers_found_at_park : ℕ := by
  sorry

end wrappers_found_at_park_l2958_295811


namespace complex_number_in_fourth_quadrant_l2958_295827

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (Complex.I ^ 2016) / (3 + 2 * Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_fourth_quadrant_l2958_295827


namespace circle_radius_range_l2958_295868

-- Define the circle C
def Circle (r : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define points A and B
def A : ℝ × ℝ := (6, 0)
def B : ℝ × ℝ := (0, 8)

-- Define the line segment AB
def LineSegmentAB := {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • A + t • B}

-- Define the condition for points M and N
def ExistsMN (r : ℝ) (P : ℝ × ℝ) :=
  ∃ M N : ℝ × ℝ, M ∈ Circle r ∧ N ∈ Circle r ∧ P.1 - M.1 = N.1 - M.1 ∧ P.2 - M.2 = N.2 - M.2

-- State the theorem
theorem circle_radius_range :
  ∀ r : ℝ, (∀ P ∈ LineSegmentAB, ExistsMN r P) ↔ (8/3 ≤ r ∧ r < 12/5) :=
sorry

end circle_radius_range_l2958_295868


namespace geometric_sequence_iff_c_eq_neg_one_l2958_295805

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) (c : ℝ) : ℝ := 2^n + c

/-- The n-th term of the sequence a_n -/
def a (n : ℕ) (c : ℝ) : ℝ := S n c - S (n-1) c

/-- Predicate to check if a sequence is geometric -/
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, n > 0 → a (n+1) = r * a n

theorem geometric_sequence_iff_c_eq_neg_one (c : ℝ) :
  is_geometric (a · c) ↔ c = -1 :=
sorry

end geometric_sequence_iff_c_eq_neg_one_l2958_295805


namespace earliest_retirement_year_l2958_295807

/-- Represents the retirement eligibility rule -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- Represents the employee's age in a given year -/
def age_in_year (hire_year : ℕ) (hire_age : ℕ) (current_year : ℕ) : ℕ :=
  hire_age + (current_year - hire_year)

/-- Represents the employee's years of employment in a given year -/
def years_employed (hire_year : ℕ) (current_year : ℕ) : ℕ :=
  current_year - hire_year

/-- Theorem stating the earliest retirement year for the employee -/
theorem earliest_retirement_year 
  (hire_year : ℕ) 
  (hire_age : ℕ) 
  (retirement_year : ℕ) :
  hire_year = 1987 →
  hire_age = 32 →
  retirement_year = 2006 →
  (∀ y : ℕ, y < retirement_year → 
    ¬(rule_of_70 (age_in_year hire_year hire_age y) (years_employed hire_year y))) →
  rule_of_70 (age_in_year hire_year hire_age retirement_year) (years_employed hire_year retirement_year) :=
by
  sorry


end earliest_retirement_year_l2958_295807


namespace slope_range_l2958_295820

theorem slope_range (a : ℝ) : 
  (∃ x y : ℝ, (a^2 + 2*a)*x - y + 1 = 0 ∧ a^2 + 2*a < 0) ↔ 
  -2 < a ∧ a < 0 :=
sorry

end slope_range_l2958_295820


namespace probability_theorem_l2958_295838

/-- The set of ball numbers in the bag -/
def BallNumbers : Finset ℕ := {1, 2, 3, 4}

/-- The probability of drawing two balls with sum not exceeding 4 -/
def prob_sum_not_exceeding_4 : ℚ :=
  (Finset.filter (fun pair => pair.1 + pair.2 ≤ 4) (BallNumbers.product BallNumbers)).card /
  (BallNumbers.product BallNumbers).card

/-- The probability of drawing two balls with replacement where n < m + 2 -/
def prob_n_less_than_m_plus_2 : ℚ :=
  (Finset.filter (fun pair => pair.2 < pair.1 + 2) (BallNumbers.product BallNumbers)).card /
  (BallNumbers.product BallNumbers).card

theorem probability_theorem :
  prob_sum_not_exceeding_4 = 1/3 ∧ prob_n_less_than_m_plus_2 = 13/16 := by
  sorry

end probability_theorem_l2958_295838


namespace hostel_mess_expenditure_l2958_295813

/-- The original daily expenditure of a hostel mess given certain conditions -/
theorem hostel_mess_expenditure 
  (initial_students : ℕ) 
  (new_students : ℕ) 
  (expense_increase : ℕ) 
  (avg_expense_decrease : ℕ) 
  (h1 : initial_students = 35)
  (h2 : new_students = 7)
  (h3 : expense_increase = 42)
  (h4 : avg_expense_decrease = 1) : 
  ∃ (original_expenditure : ℕ), original_expenditure = 420 :=
by sorry

end hostel_mess_expenditure_l2958_295813


namespace machines_in_first_group_l2958_295893

/-- The number of machines in the first group -/
def num_machines : ℕ := 8

/-- The time taken by the first group to complete a job lot (in hours) -/
def time_first_group : ℕ := 6

/-- The number of machines in the second group -/
def num_machines_second : ℕ := 12

/-- The time taken by the second group to complete a job lot (in hours) -/
def time_second_group : ℕ := 4

/-- The work rate of a single machine (job lots per hour) -/
def work_rate : ℚ := 1 / (num_machines_second * time_second_group)

theorem machines_in_first_group :
  num_machines * work_rate * time_first_group = 1 :=
sorry

end machines_in_first_group_l2958_295893


namespace tan_five_pi_fourth_l2958_295875

theorem tan_five_pi_fourth : Real.tan (5 * π / 4) = 1 := by
  sorry

end tan_five_pi_fourth_l2958_295875


namespace greatest_x_value_l2958_295845

theorem greatest_x_value (x : ℤ) (h : 2.134 * (10 : ℝ) ^ (x : ℝ) < 220000) :
  x ≤ 5 ∧ 2.134 * (10 : ℝ) ^ (5 : ℝ) < 220000 := by
  sorry

end greatest_x_value_l2958_295845


namespace lawn_length_is_80_l2958_295874

/-- Represents a rectangular lawn with roads -/
structure LawnWithRoads where
  width : ℝ
  length : ℝ
  roadWidth : ℝ
  travelCost : ℝ
  totalCost : ℝ

/-- Calculates the area of the roads on the lawn -/
def roadArea (l : LawnWithRoads) : ℝ :=
  l.roadWidth * l.length + l.roadWidth * l.width - l.roadWidth * l.roadWidth

/-- Theorem: Given the specified conditions, the length of the lawn is 80 meters -/
theorem lawn_length_is_80 (l : LawnWithRoads) 
    (h1 : l.width = 60)
    (h2 : l.roadWidth = 10)
    (h3 : l.travelCost = 2)
    (h4 : l.totalCost = 2600)
    (h5 : l.totalCost = l.travelCost * roadArea l) :
  l.length = 80 := by
  sorry

end lawn_length_is_80_l2958_295874


namespace rectangular_plot_length_l2958_295865

theorem rectangular_plot_length 
  (metallic_cost : ℝ) 
  (wooden_cost : ℝ) 
  (gate_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : metallic_cost = 26.5)
  (h2 : wooden_cost = 22)
  (h3 : gate_cost = 240)
  (h4 : total_cost = 5600) :
  ∃ (breadth length : ℝ),
    length = breadth + 14 ∧ 
    (2 * length + breadth) * metallic_cost + breadth * wooden_cost + gate_cost = total_cost ∧
    length = 59.5 := by
  sorry

end rectangular_plot_length_l2958_295865


namespace line_intersects_ellipse_l2958_295815

/-- Given real numbers a and b where ab ≠ 0, prove that ax - y + b = 0 represents a line
    and bx² + ay² = ab represents an ellipse -/
theorem line_intersects_ellipse (a b : ℝ) (h : a * b ≠ 0) :
  ∃ (line : ℝ → ℝ) (ellipse : Set (ℝ × ℝ)),
    (∀ x y, ax - y + b = 0 ↔ y = line x) ∧
    (∀ x y, (x, y) ∈ ellipse ↔ b * x^2 + a * y^2 = a * b) :=
by sorry

end line_intersects_ellipse_l2958_295815


namespace not_in_fourth_quadrant_l2958_295851

-- Define the linear function
def f (x : ℝ) : ℝ := 2 * x + 1

-- Theorem: The function f does not pass through the fourth quadrant
theorem not_in_fourth_quadrant :
  ¬ ∃ (x y : ℝ), f x = y ∧ x > 0 ∧ y < 0 :=
by
  sorry


end not_in_fourth_quadrant_l2958_295851


namespace smallest_x_with_natural_percentages_l2958_295832

theorem smallest_x_with_natural_percentages :
  ∀ x : ℝ, x > 0 →
    (∃ n : ℕ, (45 / 100) * x = n) →
    (∃ m : ℕ, (24 / 100) * x = m) →
    x ≥ 100 / 3 ∧
    (∃ a b : ℕ, (45 / 100) * (100 / 3) = a ∧ (24 / 100) * (100 / 3) = b) :=
by sorry

end smallest_x_with_natural_percentages_l2958_295832


namespace total_jogging_time_l2958_295882

-- Define the number of weekdays
def weekdays : ℕ := 5

-- Define the regular jogging time per day in minutes
def regular_time : ℕ := 30

-- Define the extra time jogged on Tuesday in minutes
def extra_tuesday : ℕ := 5

-- Define the extra time jogged on Friday in minutes
def extra_friday : ℕ := 25

-- Define the total jogging time for the week in minutes
def total_time : ℕ := weekdays * regular_time + extra_tuesday + extra_friday

-- Theorem: The total jogging time for the week is equal to 3 hours
theorem total_jogging_time : total_time / 60 = 3 := by
  sorry

end total_jogging_time_l2958_295882


namespace maya_max_number_l2958_295842

theorem maya_max_number : ∃ (max : ℕ), max = 600 ∧ 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 3 * (300 - n) ≤ max :=
by sorry

end maya_max_number_l2958_295842


namespace fifth_subject_mark_l2958_295879

/-- Given a student's marks in four subjects and the average across five subjects,
    calculate the mark in the fifth subject. -/
theorem fifth_subject_mark (e m p c : ℕ) (avg : ℚ) (h1 : e = 90) (h2 : m = 92) (h3 : p = 85) (h4 : c = 87) (h5 : avg = 87.8) :
  ∃ (b : ℕ), (e + m + p + c + b : ℚ) / 5 = avg ∧ b = 85 := by
  sorry

#check fifth_subject_mark

end fifth_subject_mark_l2958_295879


namespace base_with_final_digit_one_l2958_295840

theorem base_with_final_digit_one : 
  ∃! b : ℕ, 2 ≤ b ∧ b ≤ 15 ∧ 648 % b = 1 :=
by sorry

end base_with_final_digit_one_l2958_295840


namespace tyrones_money_l2958_295824

def one_dollar_bills : ℕ := 2
def five_dollar_bills : ℕ := 1
def quarters : ℕ := 13
def dimes : ℕ := 20
def nickels : ℕ := 8
def pennies : ℕ := 35

def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10
def nickel_value : ℚ := 0.05
def penny_value : ℚ := 0.01

def total_money : ℚ := 
  one_dollar_bills + 
  5 * five_dollar_bills + 
  quarter_value * quarters + 
  dime_value * dimes + 
  nickel_value * nickels + 
  penny_value * pennies

theorem tyrones_money : total_money = 13 := by
  sorry

end tyrones_money_l2958_295824


namespace base_eight_47_to_base_ten_l2958_295867

/-- Converts a two-digit base-eight number to base-ten -/
def base_eight_to_ten (d1 d2 : Nat) : Nat :=
  d1 * 8 + d2

/-- The base-eight number 47 -/
def base_eight_47 : Nat × Nat := (4, 7)

theorem base_eight_47_to_base_ten :
  base_eight_to_ten base_eight_47.1 base_eight_47.2 = 39 := by
  sorry

end base_eight_47_to_base_ten_l2958_295867


namespace triangle_frame_stability_l2958_295810

/-- A bicycle frame is a structure used in bicycles. -/
structure BicycleFrame where
  shape : Type

/-- A triangle is a geometric shape with three sides and three angles. -/
inductive Triangle : Type where
  | mk : Triangle

/-- Stability is a property of structures that resist deformation under load. -/
def Stability : Prop := sorry

/-- A bicycle frame made in the shape of a triangle provides stability. -/
theorem triangle_frame_stability (frame : BicycleFrame) (h : frame.shape = Triangle) : 
  Stability :=
sorry

end triangle_frame_stability_l2958_295810


namespace fruit_selection_ways_l2958_295821

/-- The number of ways to choose k items from n distinct items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of fruits in the basket -/
def num_fruits : ℕ := 5

/-- The number of fruits to be selected -/
def num_selected : ℕ := 2

/-- Theorem: There are 10 ways to select 2 fruits from a basket of 5 fruits -/
theorem fruit_selection_ways : choose num_fruits num_selected = 10 := by
  sorry

end fruit_selection_ways_l2958_295821


namespace sara_marbles_l2958_295834

theorem sara_marbles (initial lost left : ℕ) : 
  lost = 7 → left = 3 → initial = lost + left → initial = 10 := by
sorry

end sara_marbles_l2958_295834


namespace savannah_gift_wrapping_l2958_295804

/-- Given the conditions of Savannah's gift wrapping, prove that the first roll wraps 3 gifts -/
theorem savannah_gift_wrapping (total_rolls : ℕ) (total_gifts : ℕ) (second_roll_gifts : ℕ) (third_roll_gifts : ℕ) 
  (h1 : total_rolls = 3)
  (h2 : total_gifts = 12)
  (h3 : second_roll_gifts = 5)
  (h4 : third_roll_gifts = 4) :
  total_gifts - (second_roll_gifts + third_roll_gifts) = 3 := by
  sorry

end savannah_gift_wrapping_l2958_295804


namespace division_problem_l2958_295892

theorem division_problem (n : ℕ) : 
  (n / 15 = 6) ∧ (n % 15 = 5) → n = 95 :=
by sorry

end division_problem_l2958_295892


namespace triangle_side_length_l2958_295880

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a = 3 ∧ c = 5 ∧ B = 2 * A ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  b = 2 * Real.sqrt 6 := by
sorry

end triangle_side_length_l2958_295880


namespace total_students_l2958_295857

/-- The number of students in each classroom -/
structure ClassroomSizes where
  tina : ℕ
  maura : ℕ
  zack : ℕ

/-- The conditions of the problem -/
def problem_conditions (sizes : ClassroomSizes) : Prop :=
  sizes.tina = sizes.maura ∧
  sizes.zack = (sizes.tina + sizes.maura) / 2 ∧
  sizes.zack = 23

/-- The theorem stating the total number of students -/
theorem total_students (sizes : ClassroomSizes) 
  (h : problem_conditions sizes) : 
  sizes.tina + sizes.maura + sizes.zack = 69 := by
  sorry

#check total_students

end total_students_l2958_295857


namespace juice_cans_for_two_dollars_l2958_295863

def anniversary_sale (original_price : ℕ) (discount : ℕ) (total_cost : ℕ) (ice_cream_count : ℕ) (juice_cans : ℕ) : Prop :=
  let sale_price := original_price - discount
  let ice_cream_total := sale_price * ice_cream_count
  let juice_cost := total_cost - ice_cream_total
  ∃ (cans_per_two_dollars : ℕ), 
    cans_per_two_dollars * (juice_cost / 2) = juice_cans ∧
    cans_per_two_dollars = 5

theorem juice_cans_for_two_dollars :
  anniversary_sale 12 2 24 2 10 → ∃ (x : ℕ), x = 5 :=
by
  sorry

end juice_cans_for_two_dollars_l2958_295863


namespace smallest_a1_l2958_295819

/-- A sequence of positive real numbers satisfying aₙ = 7aₙ₋₁ - n for n > 1 -/
def ValidSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n > 1, a n = 7 * a (n - 1) - n)

/-- The smallest possible value of a₁ in a valid sequence is 13/36 -/
theorem smallest_a1 :
    ∀ a : ℕ → ℝ, ValidSequence a → a 1 ≥ 13/36 ∧ ∃ a', ValidSequence a' ∧ a' 1 = 13/36 :=
  sorry

end smallest_a1_l2958_295819


namespace coeff_x6_q_cubed_is_15_l2958_295888

/-- The polynomial q(x) -/
def q (x : ℝ) : ℝ := x^4 + 5*x^2 - 4*x + 1

/-- The coefficient of x^6 in (q(x))^3 -/
def coeff_x6_q_cubed : ℝ := 15

/-- Theorem: The coefficient of x^6 in (q(x))^3 is 15 -/
theorem coeff_x6_q_cubed_is_15 : coeff_x6_q_cubed = 15 := by
  sorry

end coeff_x6_q_cubed_is_15_l2958_295888


namespace constant_term_g_l2958_295847

-- Define polynomials f, g, and h
variable (f g h : ℝ → ℝ)

-- Define the condition that h is the product of f and g
variable (h_def : ∀ x, h x = f x * g x)

-- Define the constant terms of f and h
variable (f_const : f 0 = 6)
variable (h_const : h 0 = -18)

-- Theorem statement
theorem constant_term_g : g 0 = -3 := by
  sorry

end constant_term_g_l2958_295847


namespace octagon_all_equal_l2958_295861

/-- Represents an octagon with numbers at each vertex -/
structure Octagon :=
  (vertices : Fin 8 → ℝ)

/-- Condition that each vertex number is the mean of its adjacent vertices -/
def mean_condition (o : Octagon) : Prop :=
  ∀ i : Fin 8, o.vertices i = (o.vertices (i - 1) + o.vertices (i + 1)) / 2

/-- Theorem stating that all vertex numbers must be equal -/
theorem octagon_all_equal (o : Octagon) (h : mean_condition o) : 
  ∀ i j : Fin 8, o.vertices i = o.vertices j :=
sorry

end octagon_all_equal_l2958_295861


namespace rhombus_perimeter_l2958_295844

/-- The perimeter of a rhombus with diagonals of 12 inches and 16 inches is 40 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 40 :=
by sorry

end rhombus_perimeter_l2958_295844


namespace claire_photos_l2958_295836

theorem claire_photos (lisa robert claire : ℕ) 
  (h1 : lisa = robert)
  (h2 : lisa = 3 * claire)
  (h3 : robert = claire + 10) :
  claire = 5 := by
sorry

end claire_photos_l2958_295836


namespace daily_profit_calculation_l2958_295864

theorem daily_profit_calculation (num_employees : ℕ) (employee_share : ℚ) (profit_share_percentage : ℚ) :
  num_employees = 9 →
  employee_share = 5 →
  profit_share_percentage = 9/10 →
  profit_share_percentage * ((num_employees : ℚ) * employee_share) = 50 :=
by
  sorry

end daily_profit_calculation_l2958_295864


namespace geometric_sequence_properties_l2958_295839

/-- Represents a geometric sequence with positive terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  pos_terms : ∀ n, a n > 0
  geom_prop : ∀ n, a (n + 1) = q * a n

/-- Sum of first n terms of a geometric sequence -/
def S (g : GeometricSequence) (n : ℕ) : ℝ := sorry

theorem geometric_sequence_properties (g : GeometricSequence) :
  (-1 : ℝ) < S g 5 ∧ S g 5 < S g 10 ∧  -- S_5 and S_10 are positive
  (S g 5 - (-1) = S g 10 - S g 5) →    -- -1, S_5, S_10 form an arithmetic sequence
  (S g 10 - 2 * S g 5 = 1) ∧           -- First result
  (∀ h : GeometricSequence, 
    ((-1 : ℝ) < S h 5 ∧ S h 5 < S h 10 ∧ 
     S h 5 - (-1) = S h 10 - S h 5) → 
    S g 15 - S g 10 ≤ S h 15 - S h 10) ∧  -- S_15 - S_10 is minimized for g
  (S g 15 - S g 10 = 4) :=              -- Minimum value is 4
by sorry

end geometric_sequence_properties_l2958_295839


namespace log_xy_z_in_terms_of_log_x_z_and_log_y_z_l2958_295800

theorem log_xy_z_in_terms_of_log_x_z_and_log_y_z
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.log z / Real.log (x * y) = (Real.log z / Real.log x * Real.log z / Real.log y) /
                                  (Real.log z / Real.log x + Real.log z / Real.log y) :=
by sorry

end log_xy_z_in_terms_of_log_x_z_and_log_y_z_l2958_295800


namespace sin_double_angle_for_point_on_terminal_side_l2958_295802

theorem sin_double_angle_for_point_on_terminal_side :
  ∀ α : ℝ,
  let P : ℝ × ℝ := (-4, -6 * Real.sin (150 * π / 180))
  (P.1 = -4 ∧ P.2 = -6 * Real.sin (150 * π / 180)) →
  Real.sin (2 * α) = 24/25 := by
  sorry

end sin_double_angle_for_point_on_terminal_side_l2958_295802


namespace equation_solutions_l2958_295883

def equation (x : ℝ) : Prop :=
  (17 * x - x^2) / (x + 2) * (x + (17 - x) / (x + 2)) = 48

theorem equation_solutions :
  {x : ℝ | equation x} = {3, 4, -10 + 4 * Real.sqrt 21, -10 - 4 * Real.sqrt 21} := by
  sorry

end equation_solutions_l2958_295883


namespace smallest_with_8_odd_10_even_divisors_l2958_295825

/-- A function that returns the number of positive odd integer divisors of a natural number -/
def num_odd_divisors (n : ℕ) : ℕ := sorry

/-- A function that returns the number of positive even integer divisors of a natural number -/
def num_even_divisors (n : ℕ) : ℕ := sorry

/-- The theorem stating that 53760 is the smallest positive integer with 8 odd divisors and 10 even divisors -/
theorem smallest_with_8_odd_10_even_divisors :
  ∀ n : ℕ, n > 0 →
    (num_odd_divisors n = 8 ∧ num_even_divisors n = 10) →
    n ≥ 53760 ∧
    (num_odd_divisors 53760 = 8 ∧ num_even_divisors 53760 = 10) := by
  sorry

end smallest_with_8_odd_10_even_divisors_l2958_295825


namespace expression_simplification_l2958_295897

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 - 2) :
  (a - 2) / (a - 1) / (a + 1 - 3 / (a - 1)) = Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_l2958_295897


namespace nonnegative_integer_pairs_l2958_295801

theorem nonnegative_integer_pairs (x y : ℕ) : (x * y + 2)^2 = x^2 + y^2 ↔ (x = 0 ∧ y = 2) ∨ (x = 2 ∧ y = 0) := by
  sorry

end nonnegative_integer_pairs_l2958_295801


namespace log_equation_solution_l2958_295841

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 + Real.log x / Real.log 27 = 7 →
  x = 3 ^ (42 / 11) := by
  sorry

end log_equation_solution_l2958_295841


namespace square_to_rectangle_ratio_l2958_295869

theorem square_to_rectangle_ratio : 
  ∀ (square_side : ℝ) (rectangle_base rectangle_height : ℝ),
  square_side = 3 →
  rectangle_base * rectangle_height = square_side^2 →
  rectangle_base = (square_side^2 + (square_side/2)^2).sqrt →
  (rectangle_height / rectangle_base) = 4/5 :=
by
  sorry

end square_to_rectangle_ratio_l2958_295869


namespace num_teachers_at_king_middle_school_l2958_295873

/-- The number of students at King Middle School -/
def num_students : ℕ := 1500

/-- The number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- The number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 5

/-- The number of students in each class -/
def students_per_class : ℕ := 25

/-- The number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- Theorem: The number of teachers at King Middle School is 72 -/
theorem num_teachers_at_king_middle_school : 
  (num_students * classes_per_student) / (students_per_class * classes_per_teacher) = 72 :=
by sorry

end num_teachers_at_king_middle_school_l2958_295873
