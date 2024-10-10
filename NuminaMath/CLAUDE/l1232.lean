import Mathlib

namespace fliers_calculation_l1232_123292

theorem fliers_calculation (initial_fliers : ℕ) : 
  (initial_fliers : ℚ) * (9/10) * (3/4) = 1350 →
  initial_fliers = 2000 :=
by sorry

end fliers_calculation_l1232_123292


namespace certain_amount_proof_l1232_123221

theorem certain_amount_proof (x : ℝ) (A : ℝ) (h1 : x = 840) (h2 : 0.25 * x = 0.15 * 1500 - A) : A = 15 := by
  sorry

end certain_amount_proof_l1232_123221


namespace factorization_equality_l1232_123218

-- Define the theorem
theorem factorization_equality {R : Type*} [Ring R] (a b : R) :
  2 * a^2 - a * b = a * (2 * a - b) := by
  sorry

end factorization_equality_l1232_123218


namespace football_pack_cost_proof_l1232_123252

/-- The cost of a pack of football cards -/
def football_pack_cost : ℝ := 2.73

/-- The number of football card packs bought -/
def football_packs : ℕ := 2

/-- The cost of a pack of Pokemon cards -/
def pokemon_pack_cost : ℝ := 4.01

/-- The cost of a deck of baseball cards -/
def baseball_deck_cost : ℝ := 8.95

/-- The total amount spent on cards -/
def total_spent : ℝ := 18.42

theorem football_pack_cost_proof :
  (football_pack_cost * football_packs) + pokemon_pack_cost + baseball_deck_cost = total_spent := by
  sorry

end football_pack_cost_proof_l1232_123252


namespace may_salary_is_6500_l1232_123231

/-- Calculates the salary for May given the average salaries and January's salary -/
def salary_may (avg_jan_to_apr avg_feb_to_may jan_salary : ℚ) : ℚ :=
  4 * avg_feb_to_may - (4 * avg_jan_to_apr - jan_salary)

/-- Proves that the salary for May is 6500 given the conditions -/
theorem may_salary_is_6500 :
  let avg_jan_to_apr : ℚ := 8000
  let avg_feb_to_may : ℚ := 8200
  let jan_salary : ℚ := 5700
  salary_may avg_jan_to_apr avg_feb_to_may jan_salary = 6500 :=
by
  sorry

#eval salary_may 8000 8200 5700

end may_salary_is_6500_l1232_123231


namespace max_sum_of_squares_l1232_123288

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 17 →
  a * b + c + d = 85 →
  a * d + b * c = 196 →
  c * d = 120 →
  a^2 + b^2 + c^2 + d^2 ≤ 918 :=
by sorry

end max_sum_of_squares_l1232_123288


namespace larger_sphere_radius_l1232_123298

theorem larger_sphere_radius (r : ℝ) (n : ℕ) (h : r = 2 ∧ n = 6) :
  (n * (4 / 3 * π * r^3) = 4 / 3 * π * (2 * Real.rpow 3 (1/3))^3) :=
by sorry

end larger_sphere_radius_l1232_123298


namespace a_eq_one_sufficient_not_necessary_l1232_123253

-- Define the sets P and M
def P : Set ℝ := {x | x > 2}
def M (a : ℝ) : Set ℝ := {x | x > a}

-- Define what it means for a condition to be sufficient
def is_sufficient (a : ℝ) : Prop := P ⊆ M a

-- Define what it means for a condition to be necessary
def is_necessary (a : ℝ) : Prop := ∀ b : ℝ, P ⊆ M b → a ≤ b

-- State the theorem
theorem a_eq_one_sufficient_not_necessary :
  (is_sufficient 1) ∧ ¬(is_necessary 1) := by sorry

end a_eq_one_sufficient_not_necessary_l1232_123253


namespace fruit_mix_problem_l1232_123262

theorem fruit_mix_problem (total : ℕ) (apples oranges bananas plums : ℕ) : 
  total = 240 →
  oranges = 3 * apples →
  bananas = 2 * oranges →
  plums = 5 * bananas →
  total = apples + oranges + bananas + plums →
  apples = 6 := by
sorry

end fruit_mix_problem_l1232_123262


namespace work_completion_proof_l1232_123271

/-- The number of days taken by the first group to complete the work -/
def days_first_group : ℕ := 18

/-- The number of men in the second group -/
def men_second_group : ℕ := 9

/-- The number of days taken by the second group to complete the work -/
def days_second_group : ℕ := 72

/-- The number of men in the first group -/
def men_first_group : ℕ := 36

theorem work_completion_proof :
  (men_first_group : ℚ) * days_first_group = men_second_group * days_second_group :=
sorry

end work_completion_proof_l1232_123271


namespace fractional_equation_solution_l1232_123254

theorem fractional_equation_solution : 
  ∃ x : ℝ, (3 / (x + 1) - 2 / (x - 1) = 0) ∧ (x = 5) := by
  sorry

end fractional_equation_solution_l1232_123254


namespace rectangle_area_18_l1232_123227

def rectangle_area (w l : ℕ+) : ℕ := w.val * l.val

theorem rectangle_area_18 :
  {p : ℕ+ × ℕ+ | rectangle_area p.1 p.2 = 18} =
  {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} := by
  sorry

end rectangle_area_18_l1232_123227


namespace S_value_l1232_123204

noncomputable def S : ℝ :=
  1 / (5 - Real.sqrt 23) + 1 / (Real.sqrt 23 - Real.sqrt 20) - 1 / (Real.sqrt 20 - 4) -
  1 / (4 - Real.sqrt 15) + 1 / (Real.sqrt 15 - Real.sqrt 12) - 1 / (Real.sqrt 12 - 3)

theorem S_value : S = 2 * Real.sqrt 23 - 2 := by sorry

end S_value_l1232_123204


namespace cosine_sum_lower_bound_l1232_123277

theorem cosine_sum_lower_bound (a b c : ℝ) :
  Real.cos (a - b) + Real.cos (b - c) + Real.cos (c - a) ≥ -3/2 := by
  sorry

end cosine_sum_lower_bound_l1232_123277


namespace f_negative_a_l1232_123268

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log (x + Real.sqrt (x^2 + 1)) / Real.log 10

theorem f_negative_a (a M : ℝ) (h : f a = M) : f (-a) = 2 * a^2 - M := by
  sorry

end f_negative_a_l1232_123268


namespace product_of_roots_quadratic_l1232_123242

theorem product_of_roots_quadratic (x : ℝ) : 
  (8 = -2 * x^2 - 6 * x) → (∃ α β : ℝ, (α * β = 4 ∧ 8 = -2 * α^2 - 6 * α ∧ 8 = -2 * β^2 - 6 * β)) :=
by sorry

end product_of_roots_quadratic_l1232_123242


namespace cylinder_not_unique_l1232_123200

theorem cylinder_not_unique (S V : ℝ) (h_pos_S : S > 0) (h_pos_V : V > 0)
  (h_inequality : S > 3 * (2 * π * V^2)^(1/3)) :
  ∃ (r₁ r₂ h₁ h₂ : ℝ),
    r₁ ≠ r₂ ∧
    2 * π * r₁ * h₁ + 2 * π * r₁^2 = S ∧
    2 * π * r₂ * h₂ + 2 * π * r₂^2 = S ∧
    π * r₁^2 * h₁ = V ∧
    π * r₂^2 * h₂ = V :=
by sorry

end cylinder_not_unique_l1232_123200


namespace point_on_y_axis_l1232_123276

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the y-axis
def on_y_axis (p : Point2D) : Prop := p.x = 0

-- Define our point P in terms of parameter a
def P (a : ℝ) : Point2D := ⟨2*a - 6, a + 1⟩

-- Theorem statement
theorem point_on_y_axis :
  ∃ a : ℝ, on_y_axis (P a) → P a = ⟨0, 4⟩ := by
  sorry

end point_on_y_axis_l1232_123276


namespace log_ratio_evaluation_l1232_123296

theorem log_ratio_evaluation : (Real.log 4 / Real.log 3) / (Real.log 8 / Real.log 9) = 4 / 3 := by
  sorry

end log_ratio_evaluation_l1232_123296


namespace base_of_equation_l1232_123203

theorem base_of_equation (x y : ℕ) (base : ℝ) : 
  x = 9 → 
  x - y = 9 → 
  (base ^ x) * (4 ^ y) = 19683 → 
  base = 3 := by
sorry

end base_of_equation_l1232_123203


namespace bart_firewood_consumption_l1232_123279

/-- The number of logs Bart burns per day -/
def logs_per_day (pieces_per_tree : ℕ) (trees_cut : ℕ) (days : ℕ) : ℚ :=
  (pieces_per_tree * trees_cut : ℚ) / days

theorem bart_firewood_consumption 
  (pieces_per_tree : ℕ) 
  (trees_cut : ℕ) 
  (days : ℕ) 
  (h1 : pieces_per_tree = 75)
  (h2 : trees_cut = 8)
  (h3 : days = 120) :
  logs_per_day pieces_per_tree trees_cut days = 5 := by
sorry

end bart_firewood_consumption_l1232_123279


namespace geometric_progression_proof_l1232_123233

theorem geometric_progression_proof (b q : ℝ) : 
  b + b*q + b*q^2 + b*q^3 = -40 ∧ 
  b^2 + (b*q)^2 + (b*q^2)^2 + (b*q^3)^2 = 3280 → 
  b = 2 ∧ q = -3 := by
sorry

end geometric_progression_proof_l1232_123233


namespace intersection_parallel_line_exists_specific_intersection_parallel_line_l1232_123282

/-- Given two lines l₁ and l₂ in the plane, and a third line l₃,
    this theorem states that there exists a line l that passes through
    the intersection of l₁ and l₂ and is parallel to l₃. -/
theorem intersection_parallel_line_exists (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ : ℝ) :
  ∃ (a b c : ℝ),
    -- l₁: a₁x + b₁y + c₁ = 0
    -- l₂: a₂x + b₂y + c₂ = 0
    -- l₃: a₃x + b₃y + c₃ = 0
    -- l: ax + by + c = 0
    -- l passes through the intersection of l₁ and l₂
    (∀ (x y : ℝ), a₁ * x + b₁ * y + c₁ = 0 ∧ a₂ * x + b₂ * y + c₂ = 0 → a * x + b * y + c = 0) ∧
    -- l is parallel to l₃
    (∃ (k : ℝ), k ≠ 0 ∧ a = k * a₃ ∧ b = k * b₃) :=
by
  sorry

/-- The specific instance of the theorem for the given problem -/
theorem specific_intersection_parallel_line :
  ∃ (a b c : ℝ),
    -- l₁: 2x + 3y - 5 = 0
    -- l₂: 3x - 2y - 3 = 0
    -- l₃: 2x + y - 3 = 0
    -- l: ax + by + c = 0
    (∀ (x y : ℝ), 2 * x + 3 * y - 5 = 0 ∧ 3 * x - 2 * y - 3 = 0 → a * x + b * y + c = 0) ∧
    (∃ (k : ℝ), k ≠ 0 ∧ a = k * 2 ∧ b = k * 1) ∧
    a = 26 ∧ b = -13 ∧ c = -29 :=
by
  sorry

end intersection_parallel_line_exists_specific_intersection_parallel_line_l1232_123282


namespace range_of_m_l1232_123224

def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*(m+1)*x + m*(m+1) > 0

theorem range_of_m (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m > 2 ∨ (-2 ≤ m ∧ m < -1) :=
by sorry

end range_of_m_l1232_123224


namespace xyz_inequality_l1232_123238

theorem xyz_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2*x*y*z = 1) : 
  8*x*y*z ≤ 1 ∧ 
  (8*x*y*z = 1 ↔ 
    ((x, y, z) = (1/2, 1/2, 1/2) ∨ 
     (x, y, z) = (-1/2, -1/2, 1/2) ∨ 
     (x, y, z) = (-1/2, 1/2, -1/2) ∨ 
     (x, y, z) = (1/2, -1/2, -1/2))) := by
  sorry

end xyz_inequality_l1232_123238


namespace unique_solution_l1232_123275

/-- The set of solutions for the system of equations x + y = 2 and x - y = 0 -/
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 2 ∧ p.1 - p.2 = 0}

/-- Theorem stating that the solution set contains only the point (1,1) -/
theorem unique_solution :
  solution_set = {(1, 1)} := by sorry

end unique_solution_l1232_123275


namespace perpendicular_necessary_not_sufficient_l1232_123219

-- Define the basic types
variable (P : Type) -- Type for points
variable (α : Set P) -- Type for planes
variable (l m : Set P) -- Type for lines

-- Define the geometric relations
variable (perpendicular : Set P → Set P → Prop) -- Perpendicular relation for lines and planes
variable (parallel : Set P → Set P → Prop) -- Parallel relation for lines and planes
variable (subset : Set P → Set P → Prop) -- Subset relation for lines and planes

-- State the theorem
theorem perpendicular_necessary_not_sufficient
  (h : perpendicular m α) :
  (∀ l, parallel l α → perpendicular l m) ∧
  ¬(∀ l, perpendicular l m → parallel l α) :=
by sorry

end perpendicular_necessary_not_sufficient_l1232_123219


namespace product_of_roots_l1232_123247

theorem product_of_roots (a b : ℝ) 
  (ha : a^2 - 4*a + 3 = 0) 
  (hb : b^2 - 4*b + 3 = 0) 
  (hab : a ≠ b) : 
  (a + 1) * (b + 1) = 8 := by
  sorry

end product_of_roots_l1232_123247


namespace system_solution_l1232_123206

theorem system_solution :
  ∃ (x y : ℝ), x + y = 1 ∧ 4 * x + y = 10 ∧ x = 3 ∧ y = -2 :=
by sorry

end system_solution_l1232_123206


namespace boat_speed_ratio_l1232_123210

def boat_speed : ℝ := 20
def current_speed : ℝ := 4
def distance : ℝ := 2

theorem boat_speed_ratio :
  let downstream_speed := boat_speed + current_speed
  let upstream_speed := boat_speed - current_speed
  let downstream_time := distance / downstream_speed
  let upstream_time := distance / upstream_speed
  let total_time := downstream_time + upstream_time
  let total_distance := 2 * distance
  let average_speed := total_distance / total_time
  (average_speed / boat_speed) = 24 / 25 := by
sorry

end boat_speed_ratio_l1232_123210


namespace unique_digit_solution_l1232_123246

/-- Represents a six-digit number as a list of digits -/
def SixDigitNumber := List Nat

/-- Converts a three-digit number to a six-digit number -/
def toSixDigit (n : Nat) : SixDigitNumber :=
  sorry

/-- Converts a list of digits to a natural number -/
def fromDigits (digits : List Nat) : Nat :=
  sorry

/-- Checks if all digits in a list are distinct -/
def allDistinct (digits : List Nat) : Prop :=
  sorry

/-- Theorem: Unique solution for the given digit equation system -/
theorem unique_digit_solution :
  ∃! (A B C D E F : Nat),
    A ∈ Finset.range 10 ∧
    B ∈ Finset.range 10 ∧
    C ∈ Finset.range 10 ∧
    D ∈ Finset.range 10 ∧
    E ∈ Finset.range 10 ∧
    F ∈ Finset.range 10 ∧
    allDistinct [A, B, C, D, E, F] ∧
    fromDigits [A, B, C] ^ 2 = fromDigits (toSixDigit (fromDigits [D, A, E, C, F, B])) ∧
    fromDigits [C, B, A] ^ 2 = fromDigits (toSixDigit (fromDigits [E, D, C, A, B, F])) ∧
    A = 3 ∧ B = 6 ∧ C = 4 ∧ D = 1 ∧ E = 2 ∧ F = 9 :=
  by sorry

end unique_digit_solution_l1232_123246


namespace product_726_4_base9_l1232_123287

/-- Convert a base-9 number represented as a list of digits to a natural number. -/
def base9ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 9 * acc + d) 0

/-- Convert a natural number to its base-9 representation as a list of digits. -/
def natToBase9 (n : Nat) : List Nat :=
  if n < 9 then [n]
  else (n % 9) :: natToBase9 (n / 9)

/-- Theorem stating that the product of 726₉ and 4₉ is equal to 3216₉ in base 9. -/
theorem product_726_4_base9 :
  base9ToNat [6, 2, 7] * base9ToNat [4] = base9ToNat [6, 1, 2, 3] := by
  sorry

#eval base9ToNat [6, 2, 7] * base9ToNat [4] == base9ToNat [6, 1, 2, 3]

end product_726_4_base9_l1232_123287


namespace quadratic_inequality_solution_sets_l1232_123273

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) := a * x^2 - b * x - 1

-- Define the solution set of the first inequality
def solution_set (a b : ℝ) := {x : ℝ | f a b x ≥ 0}

-- State the theorem
theorem quadratic_inequality_solution_sets 
  (a b : ℝ) 
  (h1 : solution_set a b = Set.Icc (-1/2) (-1/3)) :
  {x : ℝ | x^2 - b*x - a < 0} = Set.Ioo 2 3 := by
  sorry

end quadratic_inequality_solution_sets_l1232_123273


namespace sum_of_common_ratios_is_seven_l1232_123243

/-- Given two nonconstant geometric sequences with first term k and different common ratios p and r,
    if a₃ - b₃ = 7(a₂ - b₂), then p + r = 7. -/
theorem sum_of_common_ratios_is_seven
  (k : ℝ) (p r : ℝ) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) :
  k * p^2 - k * r^2 = 7 * (k * p - k * r) → p + r = 7 := by
  sorry


end sum_of_common_ratios_is_seven_l1232_123243


namespace minus_one_power_difference_l1232_123285

theorem minus_one_power_difference : (-1)^2024 - (-1)^2023 = 2 := by
  sorry

end minus_one_power_difference_l1232_123285


namespace quadratic_polynomial_special_value_l1232_123265

/-- A quadratic polynomial -/
def QuadraticPolynomial (α : Type*) [Field α] := α → α

/-- Property: [q(x)]^2 - x^2 is divisible by (x - 2)(x + 2)(x - 5) -/
def HasSpecialDivisibility (q : QuadraticPolynomial ℝ) : Prop :=
  ∃ r : ℝ → ℝ, ∀ x : ℝ, (q x)^2 - x^2 = (x - 2) * (x + 2) * (x - 5) * (r x)

theorem quadratic_polynomial_special_value 
  (q : QuadraticPolynomial ℝ) 
  (h : HasSpecialDivisibility q) : 
  q 10 = 110 / 7 := by
  sorry

end quadratic_polynomial_special_value_l1232_123265


namespace sin_135_degrees_l1232_123207

theorem sin_135_degrees : Real.sin (135 * π / 180) = 1 / Real.sqrt 2 := by
  sorry

end sin_135_degrees_l1232_123207


namespace inequality_ratio_l1232_123260

theorem inequality_ratio (a b : ℝ) (h1 : a < b) (h2 : b < 0) : b / a < a / b := by
  sorry

end inequality_ratio_l1232_123260


namespace open_box_volume_proof_l1232_123215

/-- Calculates the volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
def open_box_volume (sheet_length sheet_width cut_size : ℝ) : ℝ :=
  (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size

/-- Proves that the volume of the open box formed from a 48 m x 36 m sheet with 6 m x 6 m corner cuts is 5184 m³. -/
theorem open_box_volume_proof :
  open_box_volume 48 36 6 = 5184 := by
  sorry

#eval open_box_volume 48 36 6

end open_box_volume_proof_l1232_123215


namespace boutique_packaging_combinations_l1232_123241

theorem boutique_packaging_combinations :
  let wrapping_paper_designs : ℕ := 10
  let ribbon_colors : ℕ := 5
  let gift_card_varieties : ℕ := 6
  let decorative_tag_types : ℕ := 2
  wrapping_paper_designs * ribbon_colors * gift_card_varieties * decorative_tag_types = 600 :=
by sorry

end boutique_packaging_combinations_l1232_123241


namespace inscribed_octagon_area_is_200_l1232_123201

/-- The area of an octagon inscribed in a square, where each vertex of the octagon
    bisects the sides of the square and the perimeter of the square is 80 centimeters. -/
def inscribedOctagonArea (square_perimeter : ℝ) (octagon_bisects_square : Prop) : ℝ :=
  sorry

/-- Theorem stating that the area of the inscribed octagon is 200 square centimeters. -/
theorem inscribed_octagon_area_is_200 :
  inscribedOctagonArea 80 true = 200 := by sorry

end inscribed_octagon_area_is_200_l1232_123201


namespace prob_A_third_try_prob_at_least_one_success_l1232_123205

/-- Probability of 甲 solving the cube within 30 seconds -/
def prob_A : ℝ := 0.8

/-- Probability of 乙 solving the cube within 30 seconds -/
def prob_B : ℝ := 0.6

/-- Each attempt is independent -/
axiom attempts_independent : True

/-- Probability of 甲 succeeding on the third try -/
theorem prob_A_third_try : 
  (1 - prob_A) * (1 - prob_A) * prob_A = 0.032 := by sorry

/-- Probability of at least one person succeeding on the first try -/
theorem prob_at_least_one_success : 
  1 - (1 - prob_A) * (1 - prob_B) = 0.92 := by sorry

end prob_A_third_try_prob_at_least_one_success_l1232_123205


namespace caleb_picked_less_than_kayla_l1232_123269

/-- The number of apples picked by Kayla -/
def kayla_apples : ℕ := 20

/-- The number of apples picked by Suraya -/
def suraya_apples : ℕ := kayla_apples + 7

/-- The number of apples picked by Caleb -/
def caleb_apples : ℕ := suraya_apples - 12

theorem caleb_picked_less_than_kayla : kayla_apples - caleb_apples = 5 := by
  sorry

end caleb_picked_less_than_kayla_l1232_123269


namespace factor_expression_l1232_123258

theorem factor_expression (x y : ℝ) : x * y^2 - 4 * x = x * (y + 2) * (y - 2) := by
  sorry

end factor_expression_l1232_123258


namespace number_difference_l1232_123230

theorem number_difference (a b : ℕ) (h1 : a + b = 22904) (h2 : b % 5 = 0) (h3 : b = 7 * a) : 
  b - a = 17178 := by
sorry

end number_difference_l1232_123230


namespace arithmetic_sequence_term_count_l1232_123248

/-- 
Given an arithmetic sequence with:
- First term a = 2
- Last term l = 2008
- Common difference d = 3

Prove that the number of terms in the sequence is 669.
-/
theorem arithmetic_sequence_term_count : 
  ∀ (a l d n : ℕ), 
    a = 2 → 
    l = 2008 → 
    d = 3 → 
    l = a + (n - 1) * d → 
    n = 669 := by
  sorry

end arithmetic_sequence_term_count_l1232_123248


namespace range_of_c_l1232_123244

-- Define the propositions p and q
def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y
def q (c : ℝ) : Prop := 1 - 2*c < 0

-- State the theorem
theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1) 
  (h3 : (p c ∨ q c) ∧ ¬(p c ∧ q c)) : 
  (c ∈ Set.Ioc 0 (1/2)) ∨ (c ∈ Set.Ioi 1) := by
  sorry

end range_of_c_l1232_123244


namespace min_paper_toys_l1232_123217

/-- Represents the number of paper toys that can be made from one sheet -/
structure PaperToys where
  boats : Nat
  planes : Nat

/-- The number of paper toys that can be made from one sheet -/
def sheet_capacity : PaperToys :=
  { boats := 8, planes := 6 }

/-- The minimum number of paper toys that can be made -/
def min_toys : Nat :=
  sheet_capacity.boats

theorem min_paper_toys :
  ∀ (n : Nat), n ≥ min_toys →
  ∃ (b p : Nat), b > 0 ∧ n = b * sheet_capacity.boats + p * sheet_capacity.planes :=
by sorry

end min_paper_toys_l1232_123217


namespace vehicle_speed_problem_l1232_123228

/-- Represents the problem of determining initial and final speeds of a vehicle --/
theorem vehicle_speed_problem
  (total_distance : ℝ)
  (initial_distance : ℝ)
  (initial_time : ℝ)
  (late_time : ℝ)
  (early_time : ℝ)
  (h1 : total_distance = 280)
  (h2 : initial_distance = 112)
  (h3 : initial_time = 2)
  (h4 : late_time = 0.5)
  (h5 : early_time = 0.5)
  : ∃ (initial_speed final_speed : ℝ),
    initial_speed = initial_distance / initial_time ∧
    final_speed = (total_distance - initial_distance) / (
      (total_distance / initial_speed - late_time) - initial_time
    ) ∧
    initial_speed = 56 ∧
    final_speed = 84 := by
  sorry


end vehicle_speed_problem_l1232_123228


namespace number_comparisons_l1232_123240

theorem number_comparisons : 
  (97430 < 100076) ∧ 
  (67500000 > 65700000) ∧ 
  (2648050 > 2648005) ∧ 
  (45000000 = 45000000) := by
  sorry

end number_comparisons_l1232_123240


namespace sin_negative_600_degrees_l1232_123280

theorem sin_negative_600_degrees : Real.sin ((-600 : ℝ) * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_negative_600_degrees_l1232_123280


namespace red_marbles_in_bag_l1232_123223

theorem red_marbles_in_bag (total_marbles : ℕ) 
  (prob_two_non_red : ℚ) (red_marbles : ℕ) : 
  total_marbles = 48 →
  prob_two_non_red = 9/16 →
  (((total_marbles - red_marbles : ℚ) / total_marbles) ^ 2 = prob_two_non_red) →
  red_marbles = 12 := by
  sorry

end red_marbles_in_bag_l1232_123223


namespace person_height_from_shadows_l1232_123212

/-- Given a tree and a person casting shadows under the same lighting conditions,
    calculate the person's height based on the tree's height and shadow lengths. -/
theorem person_height_from_shadows 
  (tree_height : ℝ) (tree_shadow : ℝ) (person_shadow : ℝ) 
  (tree_height_pos : tree_height > 0)
  (tree_shadow_pos : tree_shadow > 0)
  (person_shadow_pos : person_shadow > 0)
  (h_tree : tree_height = 40 ∧ tree_shadow = 10)
  (h_person_shadow : person_shadow = 15 / 12) -- Convert 15 inches to feet
  : (tree_height / tree_shadow) * person_shadow = 5 := by
  sorry

#check person_height_from_shadows

end person_height_from_shadows_l1232_123212


namespace expression_values_l1232_123259

theorem expression_values (a b : ℝ) (h : (2 * a) / (a + b) + b / (a - b) = 2) :
  (3 * a - b) / (a + 5 * b) = 3 ∨ (3 * a - b) / (a + 5 * b) = 1 :=
by sorry

end expression_values_l1232_123259


namespace optimal_swap_theorem_l1232_123256

/-- The distance at which car tires should be swapped to wear out equally -/
def optimal_swap_distance : ℝ := 9375

/-- The total distance a front tire can travel before wearing out -/
def front_tire_lifespan : ℝ := 25000

/-- The total distance a rear tire can travel before wearing out -/
def rear_tire_lifespan : ℝ := 15000

/-- Theorem stating that swapping tires at the optimal distance results in equal wear -/
theorem optimal_swap_theorem :
  let remaining_front := (3/5) * (front_tire_lifespan - optimal_swap_distance)
  let remaining_rear := (5/3) * (rear_tire_lifespan - optimal_swap_distance)
  remaining_front = remaining_rear := by sorry

end optimal_swap_theorem_l1232_123256


namespace pipe_fill_time_l1232_123281

theorem pipe_fill_time (fill_rate_B fill_rate_both : ℝ) 
  (hB : fill_rate_B = 1 / 15)
  (hBoth : fill_rate_both = 1 / 6)
  (hSum : fill_rate_B + (1 / fill_time_A) = fill_rate_both) :
  fill_time_A = 10 := by
  sorry

end pipe_fill_time_l1232_123281


namespace odd_function_sum_l1232_123294

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_f1 : f 1 = 8) :
  f 2008 + f 2009 + f 2010 = 8 :=
sorry

end odd_function_sum_l1232_123294


namespace three_cubes_volume_and_area_l1232_123237

/-- Calculates the volume of a cube given its edge length -/
def cube_volume (edge_length : ℝ) : ℝ := edge_length ^ 3

/-- Calculates the surface area of a cube given its edge length -/
def cube_surface_area (edge_length : ℝ) : ℝ := 6 * edge_length ^ 2

/-- Theorem about the total volume and surface area of three cubic boxes -/
theorem three_cubes_volume_and_area (edge1 edge2 edge3 : ℝ) 
  (h1 : edge1 = 3) (h2 : edge2 = 5) (h3 : edge3 = 6) : 
  cube_volume edge1 + cube_volume edge2 + cube_volume edge3 = 368 ∧ 
  cube_surface_area edge1 + cube_surface_area edge2 + cube_surface_area edge3 = 420 := by
  sorry

#check three_cubes_volume_and_area

end three_cubes_volume_and_area_l1232_123237


namespace chess_games_ratio_l1232_123235

/-- Given a chess player who played 44 games and won 16 of them, 
    prove that the ratio of games lost to games won is 7:4 -/
theorem chess_games_ratio (total_games : ℕ) (games_won : ℕ) 
  (h1 : total_games = 44) (h2 : games_won = 16) :
  (total_games - games_won) / games_won = 7 / 4 := by
  sorry

end chess_games_ratio_l1232_123235


namespace bounds_on_y_l1232_123234

-- Define the equations
def eq1 (x y : ℝ) : Prop := x^2 - 6*x + 2*y = 0
def eq2 (x y : ℝ) : Prop := 3*x^2 + 12*x - 2*y - 4 = 0
def eq3 (x y : ℝ) : Prop := y = 2*x / (1 + x^2)
def eq4 (x y : ℝ) : Prop := y = (2*x - 1) / (x^2 + 2*x + 1)

-- Define the theorem
theorem bounds_on_y :
  ∀ x y : ℝ,
  eq1 x y ∧ eq2 x y ∧ eq3 x y ∧ eq4 x y →
  y ≤ 4.5 ∧ y ≥ -8 ∧ -1 ≤ y ∧ y ≤ 1 ∧ y ≤ 1/3 :=
by sorry

end bounds_on_y_l1232_123234


namespace dodgeball_assistant_count_l1232_123263

theorem dodgeball_assistant_count :
  ∀ (total_students : ℕ) (boys girls : ℕ),
    total_students = 27 →
    boys + girls < total_students →
    boys % 4 = 0 →
    girls % 6 = 0 →
    boys / 2 + girls / 3 = girls / 2 + boys / 4 →
    (total_students - (boys + girls) = 7) ∨
    (total_students - (boys + girls) = 17) :=
by sorry

end dodgeball_assistant_count_l1232_123263


namespace symmetric_point_on_x_axis_l1232_123270

/-- Given a point P(m-1, m+1) that lies on the x-axis, 
    prove that its symmetric point with respect to the x-axis has coordinates (-2, 0) -/
theorem symmetric_point_on_x_axis (m : ℝ) :
  (m + 1 = 0) →  -- P lies on the x-axis
  ((-2 : ℝ), (0 : ℝ)) = (m - 1, -(m + 1)) :=
by sorry

end symmetric_point_on_x_axis_l1232_123270


namespace new_books_bought_l1232_123214

/-- Given Kaleb's initial number of books, the number of books he sold, and his final number of books,
    prove that the number of new books he bought is equal to the difference between his final number
    of books and the number of books he had after selling some. -/
theorem new_books_bought (initial_books sold_books final_books : ℕ) :
  initial_books = 34 →
  sold_books = 17 →
  final_books = 24 →
  final_books - (initial_books - sold_books) = 7 :=
by sorry

end new_books_bought_l1232_123214


namespace f_properties_l1232_123232

noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 - x

theorem f_properties :
  -- Part 1: Monotonicity
  (∀ x ≥ 1, ∀ y ≥ x, f y ≤ f x) ∧
  -- Part 2: Inequality for a ≥ 2
  (∀ a ≥ 2, ∀ x > 0, f x < (a/2 - 1) * x^2 + a * x - 1) ∧
  -- Part 3: Inequality for x1 and x2
  (∀ x1 > 0, ∀ x2 > 0,
    f x1 + f x2 + 2 * (x1^2 + x2^2) + x1 * x2 = 0 →
    x1 + x2 ≥ (Real.sqrt 5 - 1) / 2) :=
by sorry

end f_properties_l1232_123232


namespace tims_contribution_l1232_123202

-- Define the initial number of bales
def initial_bales : ℕ := 28

-- Define the final number of bales
def final_bales : ℕ := 54

-- Define Tim's contribution
def tims_bales : ℕ := final_bales - initial_bales

-- Theorem to prove
theorem tims_contribution : tims_bales = 26 := by sorry

end tims_contribution_l1232_123202


namespace x_y_existence_l1232_123297

theorem x_y_existence : ∃ (x y : ℝ), x / 7 = 5 / 14 ∧ x / 7 + y = 10 := by
  sorry

end x_y_existence_l1232_123297


namespace three_As_theorem_l1232_123255

-- Define the set of students
inductive Student : Type
  | Alan : Student
  | Beth : Student
  | Carlos : Student
  | Diana : Student
  | Emma : Student

-- Define a function to represent whether a student got an A
def gotA : Student → Prop := sorry

-- Define the implications stated by each student
axiom alan_implication : gotA Student.Alan → gotA Student.Beth
axiom beth_implication : gotA Student.Beth → (gotA Student.Carlos ∧ gotA Student.Emma)
axiom carlos_implication : gotA Student.Carlos → gotA Student.Diana
axiom diana_implication : gotA Student.Diana → gotA Student.Emma

-- Define a function to count how many students got an A
def count_A : (Student → Prop) → Nat := sorry

-- State the theorem
theorem three_As_theorem :
  (count_A gotA = 3) →
  ((gotA Student.Beth ∧ gotA Student.Carlos ∧ gotA Student.Emma) ∨
   (gotA Student.Carlos ∧ gotA Student.Diana ∧ gotA Student.Emma)) :=
by sorry

end three_As_theorem_l1232_123255


namespace min_tosses_for_heads_l1232_123283

theorem min_tosses_for_heads (p : ℝ) (h_p : p = 1/2) :
  ∃ n : ℕ, n ≥ 1 ∧
  (∀ k : ℕ, k ≥ n → 1 - p^k ≥ 15/16) ∧
  (∀ k : ℕ, k < n → 1 - p^k < 15/16) ∧
  n = 4 :=
sorry

end min_tosses_for_heads_l1232_123283


namespace bacterium_diameter_nanometers_l1232_123226

/-- Conversion factor from meters to nanometers -/
def meters_to_nanometers : ℝ := 10^9

/-- The diameter of the bacterium in meters -/
def bacterium_diameter_meters : ℝ := 0.00000285

/-- Theorem stating that the diameter of the bacterium in nanometers is 2850 -/
theorem bacterium_diameter_nanometers :
  bacterium_diameter_meters * meters_to_nanometers = 2850 := by
  sorry

#eval bacterium_diameter_meters * meters_to_nanometers

end bacterium_diameter_nanometers_l1232_123226


namespace number_relationship_l1232_123264

theorem number_relationship (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (ha2 : a^2 = 2) (hb3 : b^3 = 3) (hc4 : c^4 = 4) (hd5 : d^5 = 5) :
  a = c ∧ c < d ∧ d < b :=
sorry

end number_relationship_l1232_123264


namespace parallel_vectors_k_value_l1232_123225

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_k_value :
  ∀ k : ℝ, 
  let a : ℝ × ℝ := (1, k)
  let b : ℝ × ℝ := (9, k - 6)
  are_parallel a b → k = -3/4 := by
sorry

end parallel_vectors_k_value_l1232_123225


namespace triangle_properties_l1232_123222

open Real

theorem triangle_properties (a b c A B C : ℝ) (h1 : 0 < A) (h2 : A < π) 
  (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π) 
  (h7 : b * cos A = Real.sqrt 3 * a * sin B) (h8 : a = 1) :
  A = π / 6 ∧ 
  (∃ (S : ℝ), S = (2 + Real.sqrt 3) / 4 ∧ 
    ∀ (S' : ℝ), S' = 1 / 2 * b * c * sin A → S' ≤ S) := by
  sorry

end triangle_properties_l1232_123222


namespace calorie_difference_per_dollar_l1232_123251

/-- Calculates the difference in calories per dollar between burgers and burritos -/
theorem calorie_difference_per_dollar : 
  let burrito_count : ℕ := 10
  let burrito_price : ℚ := 6
  let burrito_calories : ℕ := 120
  let burger_count : ℕ := 5
  let burger_price : ℚ := 8
  let burger_calories : ℕ := 400
  let burrito_calories_per_dollar := (burrito_count * burrito_calories : ℚ) / burrito_price
  let burger_calories_per_dollar := (burger_count * burger_calories : ℚ) / burger_price
  burger_calories_per_dollar - burrito_calories_per_dollar = 50
:= by sorry


end calorie_difference_per_dollar_l1232_123251


namespace goods_lost_percentage_l1232_123272

-- Define the profit percentage
def profit_percentage : ℝ := 0.10

-- Define the loss percentage on selling price
def loss_percentage_on_selling_price : ℝ := 0.12

-- Theorem to prove
theorem goods_lost_percentage (original_value : ℝ) (original_value_positive : original_value > 0) :
  let selling_price := original_value * (1 + profit_percentage)
  let loss_value := selling_price * loss_percentage_on_selling_price
  let goods_lost_percentage := (loss_value / original_value) * 100
  goods_lost_percentage = 13.2 := by
  sorry

end goods_lost_percentage_l1232_123272


namespace teal_more_blue_l1232_123213

/-- The number of people surveyed -/
def total_surveyed : ℕ := 150

/-- The number of people who believe teal is "more green" -/
def more_green : ℕ := 80

/-- The number of people who believe teal is both "more green" and "more blue" -/
def both : ℕ := 40

/-- The number of people who think teal is neither "more green" nor "more blue" -/
def neither : ℕ := 20

/-- The number of people who believe teal is "more blue" -/
def more_blue : ℕ := total_surveyed - (more_green - both) - both - neither

theorem teal_more_blue : more_blue = 90 := by
  sorry

end teal_more_blue_l1232_123213


namespace convex_pentagon_arithmetic_angles_l1232_123245

/-- A convex pentagon with angles in arithmetic progression has each angle greater than 36° -/
theorem convex_pentagon_arithmetic_angles (α γ : ℝ) (h_convex : α + 4*γ < π) 
  (h_sum : 5*α + 10*γ = 3*π) : α > π/5 := by
  sorry

end convex_pentagon_arithmetic_angles_l1232_123245


namespace f_symmetry_l1232_123250

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x - 3

-- State the theorem
theorem f_symmetry (a b c : ℝ) : f a b c (-3) = 7 → f a b c 3 = -13 := by
  sorry

end f_symmetry_l1232_123250


namespace quadratic_factorization_l1232_123239

theorem quadratic_factorization (a b : ℤ) :
  (∀ y : ℝ, 4 * y^2 - 3 * y - 28 = (4 * y + a) * (y + b)) →
  a - b = -11 := by
sorry

end quadratic_factorization_l1232_123239


namespace triangle_area_l1232_123278

/-- The area of a triangle with vertices A(2, 2), B(8, 2), and C(4, 10) is 24 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (8, 2)
  let C : ℝ × ℝ := (4, 10)
  let area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  area = 24 := by sorry

end triangle_area_l1232_123278


namespace ellipse_equation_l1232_123229

-- Define the hyperbola E1
def E1 (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the ellipse E2
def E2 (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the condition that a > b > 0
def ellipse_condition (a b : ℝ) : Prop := a > b ∧ b > 0

-- Define the common focus condition
def common_focus (E1 E2 : ℝ → ℝ → Prop) : Prop := 
  ∃ (x y : ℝ), E1 x y ∧ E2 x y

-- Define the intersection condition
def intersect_in_quadrants (E1 E2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), E1 x1 y1 ∧ E2 x1 y1 ∧ E1 x2 y2 ∧ E2 x2 y2 ∧
    x1 > 0 ∧ y1 > 0 ∧ x2 > 0 ∧ y2 < 0

-- Define the condition that chord MN passes through focus F2
def chord_through_focus (E1 E2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), E1 x1 y1 ∧ E2 x1 y1 ∧ E1 x2 y2 ∧ E2 x2 y2 ∧
    (y1 - y2) / (x1 - x2) = (y1 - 0) / (x1 - 3)

theorem ellipse_equation :
  ∀ (a b : ℝ),
    ellipse_condition a b →
    common_focus E1 (E2 · · a b) →
    intersect_in_quadrants E1 (E2 · · a b) →
    chord_through_focus E1 (E2 · · a b) →
    a^2 = 81/4 ∧ b^2 = 45/4 :=
sorry

end ellipse_equation_l1232_123229


namespace line_equation_correct_l1232_123208

/-- The y-intercept of the line 2x + y + 2 = 0 -/
def y_intercept : ℝ := -2

/-- The point A through which line l passes -/
def point_A : ℝ × ℝ := (2, 0)

/-- The equation of line l -/
def line_equation (x y : ℝ) : Prop := x - y - 2 = 0

theorem line_equation_correct :
  (line_equation point_A.1 point_A.2) ∧
  (line_equation 0 y_intercept) ∧
  (∀ x y : ℝ, line_equation x y → (2 * x + y + 2 = 0 → y = y_intercept)) :=
sorry

end line_equation_correct_l1232_123208


namespace square_root_equation_solution_l1232_123299

/-- Given a positive real number x equal to 3.3333333333333335, prove that the equation
    x * 10 / y = x^2 is satisfied when y = 3. -/
theorem square_root_equation_solution (x : ℝ) (hx : x = 3.3333333333333335) :
  ∃ y : ℝ, y = 3 ∧ x * 10 / y = x^2 := by
  sorry

end square_root_equation_solution_l1232_123299


namespace number_of_friends_l1232_123261

theorem number_of_friends (total_cards : ℕ) (cards_per_friend : ℕ) (h1 : total_cards = 455) (h2 : cards_per_friend = 91) :
  total_cards / cards_per_friend = 5 := by
  sorry

end number_of_friends_l1232_123261


namespace f_negative_a_is_zero_l1232_123291

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x + 1

-- State the theorem
theorem f_negative_a_is_zero (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end f_negative_a_is_zero_l1232_123291


namespace f_difference_at_3_and_neg_3_l1232_123220

-- Define the function f
def f (x : ℝ) : ℝ := x^6 + x^2 + 7*x

-- State the theorem
theorem f_difference_at_3_and_neg_3 : f 3 - f (-3) = 42 := by sorry

end f_difference_at_3_and_neg_3_l1232_123220


namespace q_zero_value_l1232_123266

-- Define polynomials p, q, and r
variable (p q r : ℝ → ℝ)

-- Define the relationship between p, q, and r
axiom relation : ∀ x, r x = p x * q x + 2

-- Define the constant terms of p and r
axiom p_constant : p 0 = 6
axiom r_constant : r 0 = 5

-- Theorem to prove
theorem q_zero_value : q 0 = 1/2 := by sorry

end q_zero_value_l1232_123266


namespace arithmetic_sequence_sum_l1232_123286

def arithmetic_sum (a₁ aₙ : Int) (d : Int) : Int :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum :
  arithmetic_sum (-41) 1 2 = -440 := by
  sorry

end arithmetic_sequence_sum_l1232_123286


namespace divisible_by_three_l1232_123293

theorem divisible_by_three (n : ℕ) : 
  (3 ∣ n * 2^n + 1) ↔ (n % 6 = 1 ∨ n % 6 = 2) :=
by sorry

end divisible_by_three_l1232_123293


namespace quadratic_inequality_solution_set_l1232_123211

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-3 : ℝ) (1/2) = {x : ℝ | c * x^2 + b * x + a < 0}) :
  {x : ℝ | a * x^2 + b * x + c ≥ 0} = Set.Icc (-1/3 : ℝ) 2 :=
by sorry

end quadratic_inequality_solution_set_l1232_123211


namespace union_of_A_and_B_complement_of_intersection_A_and_B_l1232_123290

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 * x - 8 ≥ 0}

-- Theorem for A ∪ B
theorem union_of_A_and_B : A ∪ B = {x | x ≥ 3} := by sorry

-- Theorem for (A ∩ B)ᶜ
theorem complement_of_intersection_A_and_B : (A ∩ B)ᶜ = {x | x < 4 ∨ x ≥ 10} := by sorry

end union_of_A_and_B_complement_of_intersection_A_and_B_l1232_123290


namespace min_value_of_expression_l1232_123284

theorem min_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z * (x + y + z) = 1) : 
  (x + y) * (y + z) ≥ 2 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ 
    x' * y' * z' * (x' + y' + z') = 1 ∧ (x' + y') * (y' + z') = 2 :=
by sorry

end min_value_of_expression_l1232_123284


namespace special_geometric_sequence_a0_l1232_123236

/-- A geometric sequence with a special sum property -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  isGeometric : ∀ n : ℕ, a (n + 2) * a n = (a (n + 1))^2
  sumProperty : ∀ n : ℕ, (Finset.range n).sum a = 5^(n + 1) + a 0 - 5

/-- The value of a₀ in a SpecialGeometricSequence is -5 -/
theorem special_geometric_sequence_a0 (seq : SpecialGeometricSequence) : seq.a 0 = -5 := by
  sorry


end special_geometric_sequence_a0_l1232_123236


namespace cauchy_schwarz_iag_equivalence_l1232_123295

theorem cauchy_schwarz_iag_equivalence :
  (∀ (a b c d : ℝ), (a * c + b * d)^2 ≤ (a^2 + b^2) * (c^2 + d^2)) ↔
  (∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) ≤ (x + y) / 2) :=
by sorry

end cauchy_schwarz_iag_equivalence_l1232_123295


namespace no_real_roots_equation_3_l1232_123257

theorem no_real_roots_equation_3 
  (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (geom_seq : b^2 = a*c)
  (real_roots_1 : a^2 ≥ 4)
  (no_real_roots_2 : b^2 < 8) :
  c^2 < 16 :=
sorry

end no_real_roots_equation_3_l1232_123257


namespace same_color_probability_l1232_123216

theorem same_color_probability (total_balls green_balls red_balls : ℕ) 
  (h_total : total_balls = green_balls + red_balls)
  (h_green : green_balls = 6)
  (h_red : red_balls = 4) : 
  (green_balls / total_balls) ^ 2 + (red_balls / total_balls) ^ 2 = 13 / 25 := by
  sorry

end same_color_probability_l1232_123216


namespace no_squares_end_in_seven_l1232_123249

theorem no_squares_end_in_seven : 
  ∀ n : ℕ, ¬(∃ m : ℕ, m * m = 10 * n + 7) :=
by
  sorry

end no_squares_end_in_seven_l1232_123249


namespace smallest_base_for_perfect_square_l1232_123267

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_base_for_perfect_square :
  ∀ b : ℕ, b > 6 →
    (∀ k : ℕ, k > 6 ∧ k < b → ¬ is_perfect_square (4 * k + 5)) →
    is_perfect_square (4 * b + 5) →
    b = 11 := by sorry

end smallest_base_for_perfect_square_l1232_123267


namespace larger_number_proof_l1232_123209

theorem larger_number_proof (a b : ℕ) 
  (hcf_cond : Nat.gcd a b = 23)
  (lcm_cond : Nat.lcm a b = 23 * 13 * 16) :
  max a b = 368 := by
sorry

end larger_number_proof_l1232_123209


namespace marble_probabilities_and_total_l1232_123289

/-- The probability of drawing a white marble -/
def prob_white : ℚ := 1/4

/-- The probability of drawing a green marble -/
def prob_green : ℚ := 2/7

/-- The probability of drawing either a red or blue marble -/
def prob_red_or_blue : ℚ := 13/28

/-- The total number of marbles in the box -/
def total_marbles : ℕ := 28

/-- Theorem stating that the given probabilities sum to 1 and the total number of marbles is 28 -/
theorem marble_probabilities_and_total : 
  prob_white + prob_green + prob_red_or_blue = 1 ∧ total_marbles = 28 := by sorry

end marble_probabilities_and_total_l1232_123289


namespace stratified_sample_class_size_l1232_123274

/-- Given two classes with a total of 100 students, if a stratified random sample
    of 10 students contains 4 from one class, then the other class has 60 students. -/
theorem stratified_sample_class_size (total : ℕ) (sample_size : ℕ) (class_a_sample : ℕ) :
  total = 100 →
  sample_size = 10 →
  class_a_sample = 4 →
  (total - (class_a_sample * total / sample_size) : ℕ) = 60 := by
  sorry

end stratified_sample_class_size_l1232_123274
