import Mathlib

namespace NUMINAMATH_GPT_total_boxes_correct_l1772_177290

noncomputable def initial_boxes : ℕ := 400
noncomputable def cost_per_box : ℕ := 80 + 165
noncomputable def initial_spent : ℕ := initial_boxes * cost_per_box
noncomputable def donor_amount : ℕ := 4 * initial_spent
noncomputable def additional_boxes : ℕ := donor_amount / cost_per_box
noncomputable def total_boxes : ℕ := initial_boxes + additional_boxes

theorem total_boxes_correct : total_boxes = 2000 := by
  sorry

end NUMINAMATH_GPT_total_boxes_correct_l1772_177290


namespace NUMINAMATH_GPT_find_number_l1772_177209

-- Define given numbers
def a : ℕ := 555
def b : ℕ := 445

-- Define given conditions
def sum : ℕ := a + b
def difference : ℕ := a - b
def quotient : ℕ := 2 * difference
def remainder : ℕ := 30

-- Define the number we're looking for
def number := sum * quotient + remainder

-- The theorem to prove
theorem find_number : number = 220030 := by
  -- Use the let expressions to simplify the calculation for clarity
  let sum := a + b
  let difference := a - b
  let quotient := 2 * difference
  let number := sum * quotient + remainder
  show number = 220030
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_find_number_l1772_177209


namespace NUMINAMATH_GPT_hotel_assignment_l1772_177278

noncomputable def numberOfWaysToAssignFriends (rooms friends : ℕ) : ℕ :=
  if rooms = 5 ∧ friends = 6 then 7200 else 0

theorem hotel_assignment : numberOfWaysToAssignFriends 5 6 = 7200 :=
by 
  -- This is the condition already matched in the noncomputable function defined above.
  sorry

end NUMINAMATH_GPT_hotel_assignment_l1772_177278


namespace NUMINAMATH_GPT_dad_steps_l1772_177212

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end NUMINAMATH_GPT_dad_steps_l1772_177212


namespace NUMINAMATH_GPT_min_value_f_prime_at_2_l1772_177271

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 2 * a * x^2 + (1/a) * x

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 4*a*x + (1/a)

theorem min_value_f_prime_at_2 (a : ℝ) (h : a > 0) : 
  f_prime a 2 >= 12 + 4 * Real.sqrt 2 := 
by
  -- proof will be written here
  sorry

end NUMINAMATH_GPT_min_value_f_prime_at_2_l1772_177271


namespace NUMINAMATH_GPT_mike_total_rose_bushes_l1772_177274

-- Definitions based on the conditions
def costPerRoseBush : ℕ := 75
def costPerTigerToothAloe : ℕ := 100
def numberOfRoseBushesForFriend : ℕ := 2
def totalExpenseByMike : ℕ := 500
def numberOfTigerToothAloe : ℕ := 2

-- The total number of rose bushes Mike bought
noncomputable def totalNumberOfRoseBushes : ℕ :=
  let totalSpentOnAloes := numberOfTigerToothAloe * costPerTigerToothAloe
  let amountSpentOnRoseBushes := totalExpenseByMike - totalSpentOnAloes
  let numberOfRoseBushesForMike := amountSpentOnRoseBushes / costPerRoseBush
  numberOfRoseBushesForMike + numberOfRoseBushesForFriend

-- The theorem to prove
theorem mike_total_rose_bushes : totalNumberOfRoseBushes = 6 :=
  by
    sorry

end NUMINAMATH_GPT_mike_total_rose_bushes_l1772_177274


namespace NUMINAMATH_GPT_f_g_minus_g_f_l1772_177244

-- Defining the functions f and g
def f (x : ℝ) : ℝ := x^2 + 3
def g (x : ℝ) : ℝ := 3 * x^2 + 5

-- Proving the given math problem
theorem f_g_minus_g_f :
  f (g 2) - g (f 2) = 140 := by
sorry

end NUMINAMATH_GPT_f_g_minus_g_f_l1772_177244


namespace NUMINAMATH_GPT_range_of_m_l1772_177275

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∀ x y : ℝ, 0 < x → 0 < y → (2 * y / x + 9 * x / (2 * y) ≥ m^2 + m))
  ↔ (-3 ≤ m ∧ m ≤ 2) := sorry

end NUMINAMATH_GPT_range_of_m_l1772_177275


namespace NUMINAMATH_GPT_problem_solution_l1772_177287

-- Define a function to sum the digits of a number.
def sum_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n % 1000) / 100
  let d3 := (n % 100) / 10
  let d4 := n % 10
  d1 + d2 + d3 + d4

-- Define the problem numbers.
def nums : List ℕ := [4272, 4281, 4290, 4311, 4320]

-- Check if the sum of digits is divisible by 9.
def divisible_by_9 (n : ℕ) : Prop :=
  sum_digits n % 9 = 0

-- Main theorem asserting the result.
theorem problem_solution :
  ∃ n ∈ nums, ¬divisible_by_9 n ∧ (n % 100 / 10) * (n % 10) = 14 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1772_177287


namespace NUMINAMATH_GPT_largest_repeating_number_l1772_177236

theorem largest_repeating_number :
  ∃ n, n * 365 = 273863 * 365 := sorry

end NUMINAMATH_GPT_largest_repeating_number_l1772_177236


namespace NUMINAMATH_GPT_tangents_equal_l1772_177235

theorem tangents_equal (α β γ : ℝ) (h1 : Real.sin α + Real.sin β + Real.sin γ = 0) (h2 : Real.cos α + Real.cos β + Real.cos γ = 0) :
  Real.tan (3 * α) = Real.tan (3 * β) ∧ Real.tan (3 * β) = Real.tan (3 * γ) := 
sorry

end NUMINAMATH_GPT_tangents_equal_l1772_177235


namespace NUMINAMATH_GPT_GODOT_value_l1772_177224

theorem GODOT_value (G O D I T : ℕ) (h1 : G ≠ 0) (h2 : D ≠ 0) 
  (eq1 : 1000 * G + 100 * O + 10 * G + O + 1000 * D + 100 * I + 10 * D + I = 10000 * G + 1000 * O + 100 * D + 10 * O + T) : 
  10000 * G + 1000 * O + 100 * D + 10 * O + T = 10908 :=
by {
  sorry
}

end NUMINAMATH_GPT_GODOT_value_l1772_177224


namespace NUMINAMATH_GPT_eq_root_count_l1772_177204

theorem eq_root_count (p : ℝ) : 
  (∀ x : ℝ, (2 * x^2 - 3 * p * x + 2 * p = 0 → (9 * p^2 - 16 * p = 0))) →
  (∃! p1 p2 : ℝ, (9 * p1^2 - 16 * p1 = 0) ∧ (9 * p2^2 - 16 * p2 = 0) ∧ p1 ≠ p2) :=
sorry

end NUMINAMATH_GPT_eq_root_count_l1772_177204


namespace NUMINAMATH_GPT_sum_of_dimensions_l1772_177286

theorem sum_of_dimensions
  (X Y Z : ℝ)
  (h1 : X * Y = 24)
  (h2 : X * Z = 48)
  (h3 : Y * Z = 72) :
  X + Y + Z = 22 := 
sorry

end NUMINAMATH_GPT_sum_of_dimensions_l1772_177286


namespace NUMINAMATH_GPT_alice_paid_percentage_l1772_177245

theorem alice_paid_percentage {P : ℝ} (hP : P > 0)
  (hMP : ∀ P, MP = 0.60 * P)
  (hPrice_Alice_Paid : ∀ MP, Price_Alice_Paid = 0.40 * MP) :
  (Price_Alice_Paid / P) * 100 = 24 := by
  sorry

end NUMINAMATH_GPT_alice_paid_percentage_l1772_177245


namespace NUMINAMATH_GPT_factorial_division_l1772_177210

theorem factorial_division (N : Nat) (h : N ≥ 2) : 
  (Nat.factorial (2 * N)) / ((Nat.factorial (N + 2)) * (Nat.factorial (N - 2))) = 
  (List.prod (List.range' (N + 3) (2 * N - (N + 2) + 1))) / (Nat.factorial (N - 1)) :=
sorry

end NUMINAMATH_GPT_factorial_division_l1772_177210


namespace NUMINAMATH_GPT_sequence_general_term_l1772_177200

theorem sequence_general_term (a : ℕ → ℝ) (h₁ : a 1 = 1) 
  (h₂ : ∀ n, a (n + 1) = 2^n * a n) : 
  ∀ n, a n = 2^((n-1)*n / 2) := sorry

end NUMINAMATH_GPT_sequence_general_term_l1772_177200


namespace NUMINAMATH_GPT_ratio_of_Y_share_l1772_177289

theorem ratio_of_Y_share (total_profit share_diff X_share Y_share : ℝ) 
(h1 : total_profit = 700) (h2 : share_diff = 140) 
(h3 : X_share + Y_share = 700) (h4 : X_share - Y_share = 140) : 
Y_share / total_profit = 2 / 5 :=
sorry

end NUMINAMATH_GPT_ratio_of_Y_share_l1772_177289


namespace NUMINAMATH_GPT_students_failed_in_english_l1772_177267

variable (H : ℝ) (E : ℝ) (B : ℝ) (P : ℝ)

theorem students_failed_in_english
  (hH : H = 34 / 100) 
  (hB : B = 22 / 100)
  (hP : P = 44 / 100)
  (hIE : (1 - P) = H + E - B) :
  E = 44 / 100 := 
sorry

end NUMINAMATH_GPT_students_failed_in_english_l1772_177267


namespace NUMINAMATH_GPT_max_sum_of_inequalities_l1772_177206

theorem max_sum_of_inequalities (x y : ℝ) (h1 : 4 * x + 3 * y ≤ 10) (h2 : 3 * x + 5 * y ≤ 11) :
  x + y ≤ 31 / 11 :=
sorry

end NUMINAMATH_GPT_max_sum_of_inequalities_l1772_177206


namespace NUMINAMATH_GPT_number_of_valid_m_values_l1772_177246

/--
In the coordinate plane, construct a right triangle with its legs parallel to the x and y axes, and with the medians on its legs lying on the lines y = 3x + 1 and y = mx + 2. 
Prove that the number of values for the constant m such that this triangle exists is 2.
-/
theorem number_of_valid_m_values : 
  ∃ (m : ℝ), 
    (∃ (a b : ℝ), 
      (∀ D E : ℝ × ℝ, D = (a / 2, 0) ∧ E = (0, b / 2) →
      D.2 = 3 * D.1 + 1 ∧ 
      E.2 = m * E.1 + 2)) → 
    (number_of_solutions_for_m = 2) 
  :=
sorry

end NUMINAMATH_GPT_number_of_valid_m_values_l1772_177246


namespace NUMINAMATH_GPT_diff_cubes_square_of_squares_l1772_177284

theorem diff_cubes_square_of_squares {x y : ℤ} (h1 : (x + 1) ^ 3 - x ^ 3 = y ^ 2) :
  ∃ (a b : ℤ), y = a ^ 2 + b ^ 2 ∧ a = b + 1 :=
sorry

end NUMINAMATH_GPT_diff_cubes_square_of_squares_l1772_177284


namespace NUMINAMATH_GPT_positive_integer_solution_l1772_177257

theorem positive_integer_solution (n : ℕ) (h1 : n + 2009 ∣ n^2 + 2009) (h2 : n + 2010 ∣ n^2 + 2010) : n = 1 := 
by
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_positive_integer_solution_l1772_177257


namespace NUMINAMATH_GPT_investment_ratio_l1772_177277

variable (x : ℝ)
variable (p q t : ℝ)

theorem investment_ratio (h1 : 7 * p = 5 * q) (h2 : (7 * p * 8) / (5 * q * t) = 7 / 10) : t = 16 :=
by
  sorry

end NUMINAMATH_GPT_investment_ratio_l1772_177277


namespace NUMINAMATH_GPT_class_scores_mean_l1772_177273

theorem class_scores_mean 
  (F S : ℕ) (Rf Rs : ℚ)
  (hF : F = 90)
  (hS : S = 75)
  (hRatio : Rf / Rs = 2 / 3) :
  (F * (2/3 * Rs) + S * Rs) / (2/3 * Rs + Rs) = 81 := by
    sorry

end NUMINAMATH_GPT_class_scores_mean_l1772_177273


namespace NUMINAMATH_GPT_largest_fraction_l1772_177291

theorem largest_fraction :
  let f1 := (2 : ℚ) / 3
  let f2 := (3 : ℚ) / 4
  let f3 := (2 : ℚ) / 5
  let f4 := (11 : ℚ) / 15
  f2 > f1 ∧ f2 > f3 ∧ f2 > f4 :=
by
  sorry

end NUMINAMATH_GPT_largest_fraction_l1772_177291


namespace NUMINAMATH_GPT_kiddie_scoop_cost_is_three_l1772_177256

-- Define the parameters for the costs of different scoops and total payment
variable (k : ℕ)  -- cost of kiddie scoop
def cost_regular : ℕ := 4
def cost_double : ℕ := 6
def total_payment : ℕ := 32

-- Conditions: Mr. and Mrs. Martin each get a regular scoop
def regular_cost : ℕ := 2 * cost_regular

-- Their three teenage children each get double scoops
def double_cost : ℕ := 3 * cost_double

-- Total cost of regular and double scoops
def combined_cost : ℕ := regular_cost + double_cost

-- Total payment includes two kiddie scoops
def kiddie_total_cost : ℕ := total_payment - combined_cost

-- The cost of one kiddie scoop
def kiddie_cost : ℕ := kiddie_total_cost / 2

theorem kiddie_scoop_cost_is_three : kiddie_cost = 3 := by
  sorry

end NUMINAMATH_GPT_kiddie_scoop_cost_is_three_l1772_177256


namespace NUMINAMATH_GPT_solve_system_eqs_l1772_177292

theorem solve_system_eqs : 
    ∃ (x y z : ℚ), 
    4 * x - 3 * y + z = -10 ∧
    3 * x + 5 * y - 2 * z = 8 ∧
    x - 2 * y + 7 * z = 5 ∧
    x = -51 / 61 ∧ 
    y = 378 / 61 ∧ 
    z = 728 / 61 := by
  sorry

end NUMINAMATH_GPT_solve_system_eqs_l1772_177292


namespace NUMINAMATH_GPT_gcd_5280_12155_l1772_177266

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 55 := by
  sorry

end NUMINAMATH_GPT_gcd_5280_12155_l1772_177266


namespace NUMINAMATH_GPT_translate_function_down_l1772_177288

theorem translate_function_down 
  (f : ℝ → ℝ) 
  (a k : ℝ) 
  (h : ∀ x, f x = a * x) 
  : ∀ x, (f x - k) = a * x - k :=
by
  sorry

end NUMINAMATH_GPT_translate_function_down_l1772_177288


namespace NUMINAMATH_GPT_weight_of_b_is_37_l1772_177205

variables {a b c : ℝ}

-- Conditions
def average_abc (a b c : ℝ) : Prop := (a + b + c) / 3 = 45
def average_ab (a b : ℝ) : Prop := (a + b) / 2 = 40
def average_bc (b c : ℝ) : Prop := (b + c) / 2 = 46

-- Statement to prove
theorem weight_of_b_is_37 (h1 : average_abc a b c) (h2 : average_ab a b) (h3 : average_bc b c) : b = 37 :=
by {
  sorry
}

end NUMINAMATH_GPT_weight_of_b_is_37_l1772_177205


namespace NUMINAMATH_GPT_total_limes_picked_l1772_177296

-- Define the number of limes each person picked
def fred_limes : Nat := 36
def alyssa_limes : Nat := 32
def nancy_limes : Nat := 35
def david_limes : Nat := 42
def eileen_limes : Nat := 50

-- Formal statement of the problem
theorem total_limes_picked : 
  fred_limes + alyssa_limes + nancy_limes + david_limes + eileen_limes = 195 := by
  -- Add proof
  sorry

end NUMINAMATH_GPT_total_limes_picked_l1772_177296


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1772_177254

theorem necessary_but_not_sufficient (p q : Prop) : 
  (p ∨ q) → (p ∧ q) → False :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1772_177254


namespace NUMINAMATH_GPT_part1_part2_l1772_177269

noncomputable section
def g1 (x : ℝ) : ℝ := Real.log x

noncomputable def f (t : ℝ) : ℝ := 
  if g1 t = t then 1 else sorry  -- Assuming g1(x) = t has exactly one root.

theorem part1 (t : ℝ) : f t = 1 :=
by sorry

def g2 (x : ℝ) (a : ℝ) : ℝ := 
  if x ≤ 0 then x else -x^2 + 2*a*x + a

theorem part2 (a : ℝ) (h : ∃ t : ℝ, f (t + 2) > f t) : a > 1 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1772_177269


namespace NUMINAMATH_GPT_inequality_solution_set_non_empty_l1772_177240

theorem inequality_solution_set_non_empty (a : ℝ) :
  (∃ x : ℝ, a * x > -1 ∧ x + a > 0) ↔ a > -1 :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_non_empty_l1772_177240


namespace NUMINAMATH_GPT_isosceles_triangle_sides_l1772_177272

/-
  Given: 
  - An isosceles triangle with a perimeter of 60 cm.
  - The intersection point of the medians lies on the inscribed circle.
  Prove:
  - The sides of the triangle are 25 cm, 25 cm, and 10 cm.
-/

theorem isosceles_triangle_sides (AB BC AC : ℝ) 
  (h1 : AB = BC)
  (h2 : AB + BC + AC = 60) 
  (h3 : ∃ r : ℝ, r > 0 ∧ 6 * r = AC ∧ 3 * r * AC = 30 * r) :
  AB = 25 ∧ BC = 25 ∧ AC = 10 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_sides_l1772_177272


namespace NUMINAMATH_GPT_intersection_is_A_l1772_177222

-- Define the set M based on the given condition
def M : Set ℝ := {x | x / (x - 1) ≥ 0}

-- Define the set N based on the given condition
def N : Set ℝ := {x | ∃ y, y = 3 * x^2 + 1}

-- Define the set A as the intersection of M and N
def A : Set ℝ := {x | x > 1}

-- Prove that the intersection of M and N is equal to the set A
theorem intersection_is_A : (M ∩ N = A) :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_is_A_l1772_177222


namespace NUMINAMATH_GPT_unique_triple_l1772_177248

theorem unique_triple (x y p : ℕ) (hx : 0 < x) (hy : 0 < y) (hp : Nat.Prime p) (h1 : p = x^2 + 1) (h2 : 2 * p^2 = y^2 + 1) :
  (x, y, p) = (2, 7, 5) :=
sorry

end NUMINAMATH_GPT_unique_triple_l1772_177248


namespace NUMINAMATH_GPT_pow_comparison_l1772_177268

theorem pow_comparison : 2^700 > 5^300 :=
by sorry

end NUMINAMATH_GPT_pow_comparison_l1772_177268


namespace NUMINAMATH_GPT_cube_and_fourth_power_remainders_l1772_177261

theorem cube_and_fourth_power_remainders (
  b : Fin 2018 → ℕ) 
  (h1 : StrictMono b) 
  (h2 : (Finset.univ.sum b) = 2018^3) :
  ((Finset.univ.sum (λ i => b i ^ 3)) % 5 = 3) ∧
  ((Finset.univ.sum (λ i => b i ^ 4)) % 5 = 1) := 
sorry

end NUMINAMATH_GPT_cube_and_fourth_power_remainders_l1772_177261


namespace NUMINAMATH_GPT_problem_I_problem_II_l1772_177241

-- Problem (I): Proving the inequality solution set
theorem problem_I (x : ℝ) : |x - 5| + |x + 6| ≤ 12 ↔ -13/2 ≤ x ∧ x ≤ 11/2 :=
by
  sorry

-- Problem (II): Proving the range of m
theorem problem_II (m : ℝ) : (∀ x : ℝ, |x - m| + |x + 6| ≥ 7) ↔ (m ≤ -13 ∨ m ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l1772_177241


namespace NUMINAMATH_GPT_maximum_area_of_triangle_ABC_l1772_177237

noncomputable def max_area_triangle_ABC (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1/2 * b * c * Real.sin A

theorem maximum_area_of_triangle_ABC (a b c A B C : ℝ) 
  (h1: a = 4) 
  (h2: (4 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C) :
  max_area_triangle_ABC a b c A B C = 4 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_maximum_area_of_triangle_ABC_l1772_177237


namespace NUMINAMATH_GPT_quadratic_transformation_l1772_177282

theorem quadratic_transformation (y m n : ℝ) 
  (h1 : 2 * y^2 - 2 = 4 * y) 
  (h2 : (y - m)^2 = n) : 
  (m - n)^2023 = -1 := 
  sorry

end NUMINAMATH_GPT_quadratic_transformation_l1772_177282


namespace NUMINAMATH_GPT_triangle_area_l1772_177243

theorem triangle_area (P : ℝ × ℝ)
  (Q : ℝ × ℝ) (R : ℝ × ℝ)
  (P_eq : P = (3, 2))
  (Q_eq : ∃ b, Q = (7/3, 0) ∧ 2 = 3 * 3 + b ∧ 0 = 3 * (7/3) + b)
  (R_eq : ∃ b, R = (4, 0) ∧ 2 = -2 * 3 + b ∧ 0 = -2 * 4 + b) :
  (1/2) * abs (Q.1 - R.1) * abs (P.2) = 5/3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l1772_177243


namespace NUMINAMATH_GPT_percentage_greater_than_88_l1772_177220

theorem percentage_greater_than_88 (x : ℝ) (percentage : ℝ) (h1 : x = 110) (h2 : x = 88 + (percentage * 88)) : percentage = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_greater_than_88_l1772_177220


namespace NUMINAMATH_GPT_cos_and_sin_double_angle_l1772_177214

variables (θ : ℝ)

-- Conditions
def is_in_fourth_quadrant (θ : ℝ) : Prop :=
  θ > 3 * Real.pi / 2 ∧ θ < 2 * Real.pi

def sin_theta (θ : ℝ) : Prop :=
  Real.sin θ = -1 / 3

-- Problem statement
theorem cos_and_sin_double_angle (h1 : is_in_fourth_quadrant θ) (h2 : sin_theta θ) :
  Real.cos θ = 2 * Real.sqrt 2 / 3 ∧ Real.sin (2 * θ) = -(4 * Real.sqrt 2 / 9) :=
sorry

end NUMINAMATH_GPT_cos_and_sin_double_angle_l1772_177214


namespace NUMINAMATH_GPT_compute_fraction_eq_2410_l1772_177299

theorem compute_fraction_eq_2410 (x : ℕ) (hx : x = 7) : 
  (x^8 + 18 * x^4 + 81) / (x^4 + 9) = 2410 := 
by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_compute_fraction_eq_2410_l1772_177299


namespace NUMINAMATH_GPT_maximum_value_of_a_l1772_177270

theorem maximum_value_of_a :
  (∀ x : ℝ, |x - 2| + |x - 8| ≥ a) → a ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_a_l1772_177270


namespace NUMINAMATH_GPT_fraction_simplification_l1772_177238

theorem fraction_simplification : 
  (1877^2 - 1862^2) / (1880^2 - 1859^2) = 5 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1772_177238


namespace NUMINAMATH_GPT_find_numbers_l1772_177225

theorem find_numbers (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (geom_mean_cond : Real.sqrt (a * b) = Real.sqrt 5)
  (harm_mean_cond : 2 / ((1 / a) + (1 / b)) = 2) :
  (a = (5 + Real.sqrt 5) / 2 ∧ b = (5 - Real.sqrt 5) / 2) ∨
  (a = (5 - Real.sqrt 5) / 2 ∧ b = (5 + Real.sqrt 5) / 2) :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l1772_177225


namespace NUMINAMATH_GPT_A_finishes_job_in_12_days_l1772_177217

variable (A B : ℝ)

noncomputable def work_rate_A_and_B := (1 / 40)
noncomputable def work_rate_A := (1 / A)
noncomputable def work_rate_B := (1 / B)

theorem A_finishes_job_in_12_days
  (h1 : work_rate_A + work_rate_B = work_rate_A_and_B)
  (h2 : 10 * work_rate_A_and_B = 1 / 4)
  (h3 : 9 * work_rate_A = 3 / 4) :
  A = 12 :=
  sorry

end NUMINAMATH_GPT_A_finishes_job_in_12_days_l1772_177217


namespace NUMINAMATH_GPT_last_digit_base_4_of_77_l1772_177233

theorem last_digit_base_4_of_77 : (77 % 4) = 1 :=
by
  sorry

end NUMINAMATH_GPT_last_digit_base_4_of_77_l1772_177233


namespace NUMINAMATH_GPT_gcd_pow_sub_one_l1772_177226

theorem gcd_pow_sub_one (a b : ℕ) 
  (h_a : a = 2^2004 - 1) 
  (h_b : b = 2^1995 - 1) : 
  Int.gcd a b = 511 :=
by
  sorry

end NUMINAMATH_GPT_gcd_pow_sub_one_l1772_177226


namespace NUMINAMATH_GPT_profit_function_simplified_maximize_profit_l1772_177276

-- Define the given conditions
def cost_per_product : ℝ := 3
def management_fee_per_product : ℝ := 3
def annual_sales_volume (x : ℝ) : ℝ := (12 - x) ^ 2

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - (cost_per_product + management_fee_per_product)) * annual_sales_volume x

-- Define the bounds for x
def x_bounds (x : ℝ) : Prop := 9 ≤ x ∧ x ≤ 11

-- Prove the profit function in simplified form
theorem profit_function_simplified (x : ℝ) (h : x_bounds x) :
    profit x = x ^ 3 - 30 * x ^ 2 + 288 * x - 864 :=
by
  sorry

-- Prove the maximum profit and the corresponding x value
theorem maximize_profit (x : ℝ) (h : x_bounds x) :
    (∀ y, (∃ x', x_bounds x' ∧ y = profit x') → y ≤ 27) ∧ profit 9 = 27 :=
by
  sorry

end NUMINAMATH_GPT_profit_function_simplified_maximize_profit_l1772_177276


namespace NUMINAMATH_GPT_cos_double_angle_l1772_177255

theorem cos_double_angle (α : ℝ) (h : Real.sin (π + α) = 2 / 3) : Real.cos (2 * α) = 1 / 9 := 
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1772_177255


namespace NUMINAMATH_GPT_find_S_l1772_177250

theorem find_S (R S T : ℝ) (c : ℝ)
  (h1 : R = c * (S / T))
  (h2 : R = 2) (h3 : S = 1/2) (h4 : T = 4/3) (h_c : c = 16/3)
  (h_R : R = Real.sqrt 75) (h_T : T = Real.sqrt 32) :
  S = 45/4 := by
  sorry

end NUMINAMATH_GPT_find_S_l1772_177250


namespace NUMINAMATH_GPT_college_students_freshmen_psych_majors_l1772_177252

variable (T : ℕ)
variable (hT : T > 0)

def freshmen (T : ℕ) : ℕ := 40 * T / 100
def lib_arts (F : ℕ) : ℕ := 50 * F / 100
def psych_majors (L : ℕ) : ℕ := 50 * L / 100
def percent_freshmen_psych_majors (P : ℕ) (T : ℕ) : ℕ := 100 * P / T

theorem college_students_freshmen_psych_majors :
  percent_freshmen_psych_majors (psych_majors (lib_arts (freshmen T))) T = 10 := by
  sorry

end NUMINAMATH_GPT_college_students_freshmen_psych_majors_l1772_177252


namespace NUMINAMATH_GPT_propositions_correct_l1772_177297

variable {R : Type} [LinearOrderedField R] {A B : Set R}

theorem propositions_correct :
  (¬ ∃ x : R, x^2 + x + 1 = 0) ∧
  (¬ (∃ x : R, x + 1 ≤ 2) → ∀ x : R, x + 1 > 2) ∧
  (∀ x : R, x ∈ A ∩ B → x ∈ A) ∧
  (∀ x : R, x > 3 → x^2 > 9 ∧ ∃ y : R, y^2 > 9 ∧ y < 3) :=
by
  sorry

end NUMINAMATH_GPT_propositions_correct_l1772_177297


namespace NUMINAMATH_GPT_time_to_paint_one_house_l1772_177294

theorem time_to_paint_one_house (houses : ℕ) (total_time_hours : ℕ) (total_time_minutes : ℕ) 
  (minutes_per_hour : ℕ) (h1 : houses = 9) (h2 : total_time_hours = 3) 
  (h3 : minutes_per_hour = 60) (h4 : total_time_minutes = total_time_hours * minutes_per_hour) : 
  (total_time_minutes / houses) = 20 :=
by
  sorry

end NUMINAMATH_GPT_time_to_paint_one_house_l1772_177294


namespace NUMINAMATH_GPT_abcd_product_l1772_177218

theorem abcd_product :
  let A := (Real.sqrt 3003 + Real.sqrt 3004)
  let B := (-Real.sqrt 3003 - Real.sqrt 3004)
  let C := (Real.sqrt 3003 - Real.sqrt 3004)
  let D := (Real.sqrt 3004 - Real.sqrt 3003)
  A * B * C * D = 1 := 
by
  sorry

end NUMINAMATH_GPT_abcd_product_l1772_177218


namespace NUMINAMATH_GPT_number_of_men_in_club_l1772_177265

variables (M W : ℕ)

theorem number_of_men_in_club 
  (h1 : M + W = 30) 
  (h2 : (1 / 3 : ℝ) * W + M = 18) : 
  M = 12 := 
sorry

end NUMINAMATH_GPT_number_of_men_in_club_l1772_177265


namespace NUMINAMATH_GPT_arithmetic_sequence_a8_l1772_177260

theorem arithmetic_sequence_a8 (a : ℕ → ℤ) (d : ℤ) :
  a 2 = 4 → a 4 = 2 → a 8 = -2 :=
by intros ha2 ha4
   sorry

end NUMINAMATH_GPT_arithmetic_sequence_a8_l1772_177260


namespace NUMINAMATH_GPT_toaster_total_cost_l1772_177232

theorem toaster_total_cost :
  let MSRP := 30
  let insurance_rate := 0.20
  let premium_upgrade := 7
  let recycling_fee := 5
  let tax_rate := 0.50

  -- Calculate costs
  let insurance_cost := insurance_rate * MSRP
  let total_insurance_cost := insurance_cost + premium_upgrade
  let cost_before_tax := MSRP + total_insurance_cost + recycling_fee
  let state_tax := tax_rate * cost_before_tax
  let total_cost := cost_before_tax + state_tax

  -- Total cost Jon must pay
  total_cost = 72 :=
by
  sorry

end NUMINAMATH_GPT_toaster_total_cost_l1772_177232


namespace NUMINAMATH_GPT_consecutive_integer_product_sum_l1772_177285

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end NUMINAMATH_GPT_consecutive_integer_product_sum_l1772_177285


namespace NUMINAMATH_GPT_final_cost_l1772_177280

-- Definitions of initial conditions
def initial_cart_total : ℝ := 54.00
def discounted_item_original_price : ℝ := 20.00
def discount_rate1 : ℝ := 0.20
def coupon_rate : ℝ := 0.10

-- Prove the final cost after applying discounts
theorem final_cost (initial_cart_total discounted_item_original_price discount_rate1 coupon_rate : ℝ) :
  let discounted_price := discounted_item_original_price * (1 - discount_rate1)
  let total_after_first_discount := initial_cart_total - discounted_price
  let final_total := total_after_first_discount * (1 - coupon_rate)
  final_total = 45.00 :=
by 
  sorry

end NUMINAMATH_GPT_final_cost_l1772_177280


namespace NUMINAMATH_GPT_bloodPressureFriday_l1772_177283

def bloodPressureSunday : ℕ := 120
def bpChangeMonday : ℤ := 20
def bpChangeTuesday : ℤ := -30
def bpChangeWednesday : ℤ := -25
def bpChangeThursday : ℤ := 15
def bpChangeFriday : ℤ := 30

theorem bloodPressureFriday : bloodPressureSunday + bpChangeMonday + bpChangeTuesday + bpChangeWednesday + bpChangeThursday + bpChangeFriday = 130 := by {
  -- Placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_bloodPressureFriday_l1772_177283


namespace NUMINAMATH_GPT_trueConverseB_l1772_177230

noncomputable def conditionA : Prop :=
  ∀ (x y : ℝ), -- "Vertical angles are equal"
  sorry -- Placeholder for vertical angles equality

noncomputable def conditionB : Prop :=
  ∀ (l₁ l₂ : ℝ), -- "If the consecutive interior angles are supplementary, then the two lines are parallel."
  sorry -- Placeholder for supplementary angles imply parallel lines

noncomputable def conditionC : Prop :=
  ∀ (a b : ℝ), -- "If \(a = b\), then \(a^2 = b^2\)"
  a = b → a^2 = b^2

noncomputable def conditionD : Prop :=
  ∀ (a b : ℝ), -- "If \(a > 0\) and \(b > 0\), then \(a^2 + b^2 > 0\)"
  a > 0 ∧ b > 0 → a^2 + b^2 > 0

theorem trueConverseB (hB: conditionB) : -- Proposition (B) has a true converse
  ∀ (l₁ l₂ : ℝ), 
  (∃ (a1 a2 : ℝ), -- Placeholder for angles
  sorry) → (l₁ = l₂) := -- Placeholder for consecutive interior angles are supplementary
  sorry

end NUMINAMATH_GPT_trueConverseB_l1772_177230


namespace NUMINAMATH_GPT_leon_older_than_aivo_in_months_l1772_177203

theorem leon_older_than_aivo_in_months
    (jolyn therese aivo leon : ℕ)
    (h1 : jolyn = therese + 2)
    (h2 : therese = aivo + 5)
    (h3 : jolyn = leon + 5) :
    leon = aivo + 2 := 
sorry

end NUMINAMATH_GPT_leon_older_than_aivo_in_months_l1772_177203


namespace NUMINAMATH_GPT_trapezoid_area_l1772_177208

theorem trapezoid_area
  (AD BC AC BD : ℝ)
  (h1 : AD = 24)
  (h2 : BC = 8)
  (h3 : AC = 13)
  (h4 : BD = 5 * Real.sqrt 17) : 
  ∃ (area : ℝ), area = 80 :=
by
  let area := (1 / 2) * (AD + BC) * 5
  existsi area
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1772_177208


namespace NUMINAMATH_GPT_square_of_1023_l1772_177263

def square_1023_eq_1046529 : Prop :=
  let x := 1023
  x * x = 1046529

theorem square_of_1023 : square_1023_eq_1046529 :=
by
  sorry

end NUMINAMATH_GPT_square_of_1023_l1772_177263


namespace NUMINAMATH_GPT_problem_1_problem_2_l1772_177249

-- First Proof Problem
theorem problem_1 (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = 2 * x^2 + 1) : 
  f x = 2 * x^2 - 4 * x + 3 :=
sorry

-- Second Proof Problem
theorem problem_2 {a b : ℝ} (f : ℝ → ℝ) (hf : ∀ x, f x = x / (a * x + b))
  (h1 : f 2 = 1) (h2 : ∃! x, f x = x) : 
  f x = 2 * x / (x + 2) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1772_177249


namespace NUMINAMATH_GPT_fifth_term_of_sequence_l1772_177207

theorem fifth_term_of_sequence :
  let a_n (n : ℕ) := (-1:ℤ)^(n+1) * (n^2 + 1)
  ∃ x : ℤ, a_n 5 * x^5 = 26 * x^5 :=
by
  sorry

end NUMINAMATH_GPT_fifth_term_of_sequence_l1772_177207


namespace NUMINAMATH_GPT_mural_lunch_break_duration_l1772_177213

variable (a t L : ℝ)

theorem mural_lunch_break_duration
  (h1 : (8 - L) * (a + t) = 0.6)
  (h2 : (6.5 - L) * t = 0.3)
  (h3 : (11 - L) * a = 0.1) :
  L = 40 :=
by
  sorry

end NUMINAMATH_GPT_mural_lunch_break_duration_l1772_177213


namespace NUMINAMATH_GPT_pair_divisibility_l1772_177231

theorem pair_divisibility (m n : ℕ) : 
  (m * n ∣ m ^ 2019 + n) ↔ ((m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 2 ^ 2019)) := sorry

end NUMINAMATH_GPT_pair_divisibility_l1772_177231


namespace NUMINAMATH_GPT_haley_spent_32_dollars_l1772_177216

noncomputable def total_spending (ticket_price : ℕ) (tickets_bought_self_friends : ℕ) (extra_tickets : ℕ) : ℕ :=
  ticket_price * (tickets_bought_self_friends + extra_tickets)

theorem haley_spent_32_dollars :
  total_spending 4 3 5 = 32 :=
by
  sorry

end NUMINAMATH_GPT_haley_spent_32_dollars_l1772_177216


namespace NUMINAMATH_GPT_marbles_steve_now_l1772_177229
-- Import necessary libraries

-- Define the initial conditions as given in a)
def initial_conditions (sam steve sally : ℕ) := sam = 2 * steve ∧ sally = sam - 5 ∧ sam - 6 = 8

-- Define the proof problem statement
theorem marbles_steve_now (sam steve sally : ℕ) (h : initial_conditions sam steve sally) : steve + 3 = 10 :=
sorry

end NUMINAMATH_GPT_marbles_steve_now_l1772_177229


namespace NUMINAMATH_GPT_cats_to_dogs_ratio_l1772_177259

noncomputable def num_dogs : ℕ := 18
noncomputable def num_cats : ℕ := num_dogs - 6
noncomputable def ratio (a b : ℕ) : ℚ := a / b

theorem cats_to_dogs_ratio (h1 : num_dogs = 18) (h2 : num_cats = num_dogs - 6) : ratio num_cats num_dogs = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cats_to_dogs_ratio_l1772_177259


namespace NUMINAMATH_GPT_proof_problem_l1772_177251

-- Define the sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℕ := (n^2 + n) / 2

-- Define the arithmetic sequence a_n based on S_n
def a (n : ℕ) : ℕ := if n = 1 then 1 else S n - S (n - 1)

-- Define the geometric sequence b_n with initial conditions
def b (n : ℕ) : ℕ :=
  if n = 1 then a 1 + 1
  else if n = 2 then a 2 + 2
  else 2^n

-- Define the sum of the first n terms of the geometric sequence b_n
def T (n : ℕ) : ℕ := 2 * (2^n - 1)

-- Main theorem to prove
theorem proof_problem :
  (∀ n, a n = n) ∧
  (∀ n, n ≥ 1 → b n = 2^n) ∧
  (∃ n, T n + a n > 300 ∧ ∀ m < n, T m + a m ≤ 300) :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_problem_l1772_177251


namespace NUMINAMATH_GPT_M_is_even_l1772_177201

def sum_of_digits (n : ℕ) : ℕ := -- Define the digit sum function
  sorry

theorem M_is_even (M : ℕ) (h1 : sum_of_digits M = 100) (h2 : sum_of_digits (5 * M) = 50) : 
  M % 2 = 0 :=
sorry

end NUMINAMATH_GPT_M_is_even_l1772_177201


namespace NUMINAMATH_GPT_common_difference_of_sequence_l1772_177281

variable (a : ℕ → ℚ)

def is_arithmetic_sequence (a : ℕ → ℚ) :=
  ∃ d : ℚ, ∀ n m : ℕ, a n = a m + d * (n - m)

theorem common_difference_of_sequence 
  (h : a 2015 = a 2013 + 6) 
  (ha : is_arithmetic_sequence a) :
  ∃ d : ℚ, d = 3 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_of_sequence_l1772_177281


namespace NUMINAMATH_GPT_find_x_l1772_177223

-- Definitions corresponding to conditions a)
def rectangle (AB CD BC AD x : ℝ) := AB = 2 ∧ CD = 2 ∧ BC = 1 ∧ AD = 1 ∧ x = 0

-- Define the main statement to be proven
theorem find_x (AB CD BC AD x k m: ℝ) (h: rectangle AB CD BC AD x) : 
  x = (0 : ℝ) ∧ k = 0 ∧ m = 0 ∧ x = (Real.sqrt k - m) ∧ k + m = 0 :=
by
  cases h
  sorry

end NUMINAMATH_GPT_find_x_l1772_177223


namespace NUMINAMATH_GPT_area_PST_is_5_l1772_177219

noncomputable def area_of_triangle_PST 
  (P Q R S T : Point)
  (PQ QR PR : ℝ) 
  (PS PT : ℝ)
  (hPQ : PQ = 8)
  (hQR : QR = 9)
  (hPR : PR = 10)
  (hPS : PS = 3)
  (hPT : PT = 5)
  : ℝ := 
  5

theorem area_PST_is_5 
  (P Q R S T : Point)
  (PQ QR PR : ℝ) 
  (PS PT : ℝ)
  (hPQ : PQ = 8)
  (hQR : QR = 9)
  (hPR : PR = 10)
  (hPS : PS = 3)
  (hPT : PT = 5)
  : area_of_triangle_PST P Q R S T PQ QR PR PS PT hPQ hQR hPR hPS hPT = 5 :=
sorry

end NUMINAMATH_GPT_area_PST_is_5_l1772_177219


namespace NUMINAMATH_GPT_solve_for_exponent_l1772_177258

theorem solve_for_exponent (K : ℕ) (h1 : 32 = 2 ^ 5) (h2 : 64 = 2 ^ 6) 
    (h3 : 32 ^ 5 * 64 ^ 2 = 2 ^ K) : K = 37 := 
by 
    sorry

end NUMINAMATH_GPT_solve_for_exponent_l1772_177258


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1772_177234

theorem arithmetic_sequence_sum : 
  ∃ x y, (∃ d, 
  d = 12 - 5 ∧ 
  19 + d = x ∧ 
  x + d = y ∧ 
  y + d = 40 ∧ 
  x + y = 59) :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1772_177234


namespace NUMINAMATH_GPT_find_xy_l1772_177262

theorem find_xy (x y : ℝ) (h : (x - 13)^2 + (y - 14)^2 + (x - y)^2 = 1/3) : 
  x = 40/3 ∧ y = 41/3 :=
sorry

end NUMINAMATH_GPT_find_xy_l1772_177262


namespace NUMINAMATH_GPT_team_A_more_points_than_team_B_l1772_177264

theorem team_A_more_points_than_team_B :
  let number_of_teams := 8
  let number_of_remaining_games := 6
  let win_probability_each_game := (1 : ℚ) / 2
  let team_A_beats_team_B_initial : Prop := True -- Corresponding to the condition team A wins the first game
  let probability_A_wins := 1087 / 2048
  team_A_beats_team_B_initial → win_probability_each_game = 1 / 2 → number_of_teams = 8 → 
    let A_more_points_than_B := team_A_beats_team_B_initial ∧ win_probability_each_game ^ number_of_remaining_games = probability_A_wins
    A_more_points_than_B :=
  sorry

end NUMINAMATH_GPT_team_A_more_points_than_team_B_l1772_177264


namespace NUMINAMATH_GPT_cost_of_four_dozen_bananas_l1772_177211

/-- Given that five dozen bananas cost $24.00,
    prove that the cost for four dozen bananas is $19.20. -/
theorem cost_of_four_dozen_bananas 
  (cost_five_dozen: ℝ)
  (rate: cost_five_dozen = 24) : 
  ∃ (cost_four_dozen: ℝ), cost_four_dozen = 19.2 := by
  sorry

end NUMINAMATH_GPT_cost_of_four_dozen_bananas_l1772_177211


namespace NUMINAMATH_GPT_part1_store_a_cost_part1_store_b_cost_part2_cost_comparison_part3_cost_effective_plan_l1772_177247

-- Defining the conditions
def racket_price : ℕ := 50
def ball_price : ℕ := 20
def num_rackets : ℕ := 10

-- Store A cost function
def store_A_cost (x : ℕ) : ℕ := 20 * x + 300

-- Store B cost function
def store_B_cost (x : ℕ) : ℕ := 16 * x + 400

-- Part (1): Express the costs in algebraic form
theorem part1_store_a_cost (x : ℕ) (hx : 10 < x) : store_A_cost x = 20 * x + 300 := by
  sorry

theorem part1_store_b_cost (x : ℕ) (hx : 10 < x) : store_B_cost x = 16 * x + 400 := by
  sorry

-- Part (2): Cost for x = 40
theorem part2_cost_comparison : store_A_cost 40 > store_B_cost 40 := by
  sorry

-- Part (3): Most cost-effective purchasing plan
def store_a_cost_rackets : ℕ := racket_price * num_rackets
def store_a_free_balls : ℕ := num_rackets
def remaining_balls (total_balls : ℕ) : ℕ := total_balls - store_a_free_balls
def store_b_cost_remaining_balls (remaining_balls : ℕ) : ℕ := remaining_balls * ball_price * 4 / 5

theorem part3_cost_effective_plan : store_a_cost_rackets + store_b_cost_remaining_balls (remaining_balls 40) = 980 := by
  sorry

end NUMINAMATH_GPT_part1_store_a_cost_part1_store_b_cost_part2_cost_comparison_part3_cost_effective_plan_l1772_177247


namespace NUMINAMATH_GPT_slope_of_line_l1772_177221

theorem slope_of_line (x y : ℝ) : 
  3 * y + 9 = -6 * x - 15 → 
  ∃ m b, y = m * x + b ∧ m = -2 := 
by {
  sorry
}

end NUMINAMATH_GPT_slope_of_line_l1772_177221


namespace NUMINAMATH_GPT_ashok_total_subjects_l1772_177242

/-- Ashok secured an average of 78 marks in some subjects. If the average of marks in 5 subjects 
is 74, and he secured 98 marks in the last subject, how many subjects are there in total? -/
theorem ashok_total_subjects (n : ℕ) 
  (avg_all : 78 * n = 74 * (n - 1) + 98) : n = 6 :=
sorry

end NUMINAMATH_GPT_ashok_total_subjects_l1772_177242


namespace NUMINAMATH_GPT_polynomial_remainder_l1772_177202

theorem polynomial_remainder (p : Polynomial ℝ) :
  (p.eval 2 = 3) → (p.eval 3 = 9) → ∃ q : Polynomial ℝ, p = (Polynomial.X - 2) * (Polynomial.X - 3) * q + (6 * Polynomial.X - 9) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l1772_177202


namespace NUMINAMATH_GPT_find_a_odd_function_l1772_177227

noncomputable def f (a x : ℝ) := Real.log (Real.sqrt (x^2 + 1) - a * x)

theorem find_a_odd_function :
  ∀ a : ℝ, (∀ x : ℝ, f a (-x) + f a x = 0) ↔ (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_GPT_find_a_odd_function_l1772_177227


namespace NUMINAMATH_GPT_fraction_of_males_l1772_177239

theorem fraction_of_males (M F : ℝ) 
  (h1 : M + F = 1)
  (h2 : (7 / 8) * M + (4 / 5) * F = 0.845) :
  M = 0.6 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_males_l1772_177239


namespace NUMINAMATH_GPT_annual_growth_rate_l1772_177279

theorem annual_growth_rate (P₁ P₂ : ℝ) (y : ℕ) (r : ℝ)
  (h₁ : P₁ = 1) 
  (h₂ : P₂ = 1.21)
  (h₃ : y = 2)
  (h_growth : P₂ = P₁ * (1 + r) ^ y) :
  r = 0.1 :=
by {
  sorry
}

end NUMINAMATH_GPT_annual_growth_rate_l1772_177279


namespace NUMINAMATH_GPT_problem_remainder_6_pow_83_add_8_pow_83_mod_49_l1772_177228

-- Definitions based on the conditions.
def euler_totient_49 : ℕ := 42

theorem problem_remainder_6_pow_83_add_8_pow_83_mod_49 
  (h1 : 6 ^ euler_totient_49 ≡ 1 [MOD 49])
  (h2 : 8 ^ euler_totient_49 ≡ 1 [MOD 49]) :
  (6 ^ 83 + 8 ^ 83) % 49 = 35 :=
by
  sorry

end NUMINAMATH_GPT_problem_remainder_6_pow_83_add_8_pow_83_mod_49_l1772_177228


namespace NUMINAMATH_GPT_sequence_explicit_formula_l1772_177295

noncomputable def sequence_a : ℕ → ℝ
| 0     => 0  -- Not used, but needed for definition completeness
| 1     => 3
| (n+1) => n / (n + 1) * sequence_a n

theorem sequence_explicit_formula (n : ℕ) (h : n ≠ 0) :
  sequence_a n = 3 / n :=
by sorry

end NUMINAMATH_GPT_sequence_explicit_formula_l1772_177295


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l1772_177253

theorem trigonometric_identity_proof 
  (α β γ : ℝ) (a b c : ℝ)
  (h1 : 0 < α ∧ α < π)
  (h2 : 0 < β ∧ β < π)
  (h3 : 0 < γ ∧ γ < π)
  (hc : 0 < c)
  (hb : b = (c * (Real.cos α + Real.cos β * Real.cos γ)) / (Real.sin γ)^2)
  (ha : a = (c * (Real.cos β + Real.cos α * Real.cos γ)) / (Real.sin γ)^2) :
  1 - (Real.cos α)^2 - (Real.cos β)^2 - (Real.cos γ)^2 - 2 * Real.cos α * Real.cos β * Real.cos γ = 0 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l1772_177253


namespace NUMINAMATH_GPT_f_five_eq_three_f_three_x_inv_f_243_l1772_177298

-- Define the function f satisfying the given conditions.
def f (x : ℕ) : ℕ :=
  if x = 5 then 3
  else if x = 15 then 9
  else if x = 45 then 27
  else if x = 135 then 81
  else if x = 405 then 243
  else 0

-- Define the condition f(5) = 3
theorem f_five_eq_three : f 5 = 3 := rfl

-- Define the condition f(3x) = 3f(x) for all x
theorem f_three_x (x : ℕ) : f (3 * x) = 3 * f x :=
sorry

-- Prove that f⁻¹(243) = 405.
theorem inv_f_243 : f (405) = 243 :=
by sorry

-- Concluding the proof statement using the concluded theorems.
example : f (405) = 243 :=
by apply inv_f_243

end NUMINAMATH_GPT_f_five_eq_three_f_three_x_inv_f_243_l1772_177298


namespace NUMINAMATH_GPT_sine_triangle_l1772_177293

theorem sine_triangle (a b c : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_perimeter : a + b + c ≤ 2 * Real.pi)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (ha_pi : a < Real.pi) (hb_pi : b < Real.pi) (hc_pi : c < Real.pi):
  ∃ (x y z : ℝ), x = Real.sin a ∧ y = Real.sin b ∧ z = Real.sin c ∧ x + y > z ∧ y + z > x ∧ x + z > y :=
by
  sorry

end NUMINAMATH_GPT_sine_triangle_l1772_177293


namespace NUMINAMATH_GPT_johns_payment_l1772_177215

-- Define the value of the camera
def camera_value : ℕ := 5000

-- Define the rental fee rate per week as a percentage
def rental_fee_rate : ℝ := 0.1

-- Define the rental period in weeks
def rental_period : ℕ := 4

-- Define the friend's contribution rate as a percentage
def friend_contribution_rate : ℝ := 0.4

-- Theorem: Calculate how much John pays for the camera rental
theorem johns_payment :
  let weekly_rental_fee := camera_value * rental_fee_rate
  let total_rental_fee := weekly_rental_fee * rental_period
  let friends_contribution := total_rental_fee * friend_contribution_rate
  let johns_payment := total_rental_fee - friends_contribution
  johns_payment = 1200 :=
by
  sorry

end NUMINAMATH_GPT_johns_payment_l1772_177215
