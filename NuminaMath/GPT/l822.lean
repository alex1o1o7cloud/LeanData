import Mathlib

namespace complement_N_subset_M_l822_82206

-- Definitions for the sets M and N
def M : Set ℝ := {x | x * (x - 3) < 0}
def N : Set ℝ := {x | x < 1 ∨ x ≥ 3}

-- Complement of N in ℝ
def complement_N : Set ℝ := {x | ¬(x < 1 ∨ x ≥ 3)}

-- The theorem stating that complement_N is a subset of M
theorem complement_N_subset_M : complement_N ⊆ M :=
by
  sorry

end complement_N_subset_M_l822_82206


namespace euler_sum_of_squares_euler_sum_of_quads_l822_82228

theorem euler_sum_of_squares :
  ∑' n : ℕ, 1 / (n.succ : ℚ)^2 = π^2 / 6 := sorry

theorem euler_sum_of_quads :
  ∑' n : ℕ, 1 / (n.succ : ℚ)^4 = π^4 / 90 := sorry

end euler_sum_of_squares_euler_sum_of_quads_l822_82228


namespace total_people_present_l822_82270

/-- This definition encapsulates all the given conditions: 
    The number of parents, pupils, staff members, and performers. -/
def num_parents : ℕ := 105
def num_pupils : ℕ := 698
def num_staff : ℕ := 45
def num_performers : ℕ := 32

/-- Theorem stating that the total number of people present in the program is 880 
    given the stated conditions. -/
theorem total_people_present : num_parents + num_pupils + num_staff + num_performers = 880 :=
by 
  /- We can use Lean's capabilities to verify the arithmetics. -/
  sorry

end total_people_present_l822_82270


namespace cost_difference_is_120_l822_82290

-- Define the monthly costs and duration
def rent_monthly_cost : ℕ := 20
def buy_monthly_cost : ℕ := 30
def months_in_a_year : ℕ := 12

-- Annual cost definitions
def annual_rent_cost : ℕ := rent_monthly_cost * months_in_a_year
def annual_buy_cost : ℕ := buy_monthly_cost * months_in_a_year

-- The main theorem to prove the difference in annual cost is $120
theorem cost_difference_is_120 : annual_buy_cost - annual_rent_cost = 120 := by
  sorry

end cost_difference_is_120_l822_82290


namespace value_of_x_l822_82231

theorem value_of_x (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3 / 8 := 
by
  sorry

end value_of_x_l822_82231


namespace megatek_manufacturing_percentage_proof_l822_82235

def megatek_employee_percentage
  (total_degrees_in_circle : ℕ)
  (manufacturing_degrees : ℕ) : ℚ :=
  (manufacturing_degrees / total_degrees_in_circle : ℚ) * 100

theorem megatek_manufacturing_percentage_proof (h1 : total_degrees_in_circle = 360)
  (h2 : manufacturing_degrees = 54) :
  megatek_employee_percentage total_degrees_in_circle manufacturing_degrees = 15 := 
by
  sorry

end megatek_manufacturing_percentage_proof_l822_82235


namespace events_mutually_exclusive_but_not_opposite_l822_82267

inductive Card
| black
| red
| white

inductive Person
| A
| B
| C

def event_A_gets_red (distribution : Person → Card) : Prop :=
  distribution Person.A = Card.red

def event_B_gets_red (distribution : Person → Card) : Prop :=
  distribution Person.B = Card.red

theorem events_mutually_exclusive_but_not_opposite (distribution : Person → Card) :
  event_A_gets_red distribution ∧ event_B_gets_red distribution → False :=
by sorry

end events_mutually_exclusive_but_not_opposite_l822_82267


namespace consequent_in_ratio_4_6_l822_82288

theorem consequent_in_ratio_4_6 (h : 4 = 6 * (20 / x)) : x = 30 := 
by
  have h' : 4 * x = 6 * 20 := sorry -- cross-multiplication
  have h'' : x = 120 / 4 := sorry -- solving for x
  have hx : x = 30 := sorry -- simplifying 120 / 4

  exact hx

end consequent_in_ratio_4_6_l822_82288


namespace proof_problem_l822_82224

open Set

variable {R : Set ℝ} (A B : Set ℝ) (complement_B : Set ℝ)

-- Defining set A
def setA : Set ℝ := { x | 1 < x ∧ x < 3 }

-- Defining set B based on the given functional relationship
def setB : Set ℝ := { x | 2 < x } 

-- Defining the complement of set B (in the universal set R)
def complementB : Set ℝ := { x | x ≤ 2 }

-- The intersection we need to prove is equivalent to the given answer
def intersection_result : Set ℝ := { x | 1 < x ∧ x ≤ 2 }

-- The theorem statement (no proof)
theorem proof_problem : setA ∩ complementB = intersection_result := 
by
  sorry

end proof_problem_l822_82224


namespace general_formula_for_sequence_l822_82292

noncomputable def a_n (n : ℕ) : ℕ := sorry
noncomputable def S_n (n : ℕ) : ℕ := sorry

theorem general_formula_for_sequence {n : ℕ} (hn: n > 0)
  (h1: ∀ n, a_n n > 0)
  (h2: ∀ n, 4 * S_n n = (a_n n)^2 + 2 * (a_n n))
  : a_n n = 2 * n := sorry

end general_formula_for_sequence_l822_82292


namespace correct_result_l822_82275

theorem correct_result (x : ℤ) (h : x * 3 - 5 = 103) : (x / 3) - 5 = 7 :=
sorry

end correct_result_l822_82275


namespace exam_room_selection_l822_82237

theorem exam_room_selection (rooms : List ℕ) (n : ℕ) 
    (fifth_room_selected : 5 ∈ rooms) (twentyfirst_room_selected : 21 ∈ rooms) :
    rooms = [5, 13, 21, 29, 37, 45, 53, 61] → 
    37 ∈ rooms ∧ 53 ∈ rooms :=
by
  sorry

end exam_room_selection_l822_82237


namespace remainder_of_product_l822_82255

open Nat

theorem remainder_of_product (a b : ℕ) (ha : a % 5 = 4) (hb : b % 5 = 3) :
  (a * b) % 5 = 2 :=
by
  sorry

end remainder_of_product_l822_82255


namespace no_intersection_of_curves_l822_82262

theorem no_intersection_of_curves :
  ∀ x y : ℝ, ¬ (3 * x^2 + 2 * y^2 = 4 ∧ 6 * x^2 + 3 * y^2 = 9) :=
by sorry

end no_intersection_of_curves_l822_82262


namespace equation_of_parallel_line_l822_82272

noncomputable def is_parallel (m₁ m₂ : ℝ) := m₁ = m₂

theorem equation_of_parallel_line (m : ℝ) (b : ℝ) (x₀ y₀ : ℝ) (a b1 c : ℝ) :
  is_parallel m (1 / 2) → y₀ = -1 → x₀ = 0 → 
  (a = 1 ∧ b1 = -2 ∧ c = -2) →
  a * x₀ + b1 * y₀ + c = 0 :=
by
  intros h_parallel hy hx habc
  sorry

end equation_of_parallel_line_l822_82272


namespace product_of_powers_l822_82245

theorem product_of_powers :
  ((-1 : Int)^3) * ((-2 : Int)^2) = -4 := by
  sorry

end product_of_powers_l822_82245


namespace large_diagonal_proof_l822_82285

variable (a b : ℝ) (α : ℝ)
variable (h₁ : a < b)
variable (h₂ : 1 < a) -- arbitrary positive scalar to make obtuse properties hold

noncomputable def large_diagonal_length : ℝ :=
  Real.sqrt (a^2 + b^2 + 2 * b * (Real.cos α * Real.sqrt (a^2 - b^2 * Real.sin α^2) + b * Real.sin α^2))

theorem large_diagonal_proof
  (h₃ : 90 < α + Real.arcsin (b * Real.sin α / a)) :
  large_diagonal_length a b α = Real.sqrt (a^2 + b^2 + 2 * b * (Real.cos α * Real.sqrt (a^2 - b^2 * Real.sin α^2) + b * Real.sin α^2)) :=
sorry

end large_diagonal_proof_l822_82285


namespace volume_of_box_l822_82276

theorem volume_of_box (l w h : ℝ) (h1 : l * w = 24) (h2 : w * h = 16) (h3 : l * h = 6) :
  l * w * h = 48 :=
by
  sorry

end volume_of_box_l822_82276


namespace sum_midpoints_x_sum_midpoints_y_l822_82264

-- Defining the problem conditions
variables (a b c d e f : ℝ)
-- Sum of the x-coordinates of the triangle vertices is 15
def sum_x_coords (a b c : ℝ) : Prop := a + b + c = 15
-- Sum of the y-coordinates of the triangle vertices is 12
def sum_y_coords (d e f : ℝ) : Prop := d + e + f = 12

-- Proving the sum of x-coordinates of midpoints of sides is 15
theorem sum_midpoints_x (h1 : sum_x_coords a b c) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := 
by  
  sorry

-- Proving the sum of y-coordinates of midpoints of sides is 12
theorem sum_midpoints_y (h2 : sum_y_coords d e f) : 
  (d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 12 := 
by  
  sorry

end sum_midpoints_x_sum_midpoints_y_l822_82264


namespace large_paintings_count_l822_82209

-- Define the problem conditions
def paint_per_large : Nat := 3
def paint_per_small : Nat := 2
def small_paintings : Nat := 4
def total_paint : Nat := 17

-- Question to find number of large paintings (L)
theorem large_paintings_count :
  ∃ L : Nat, (paint_per_large * L + paint_per_small * small_paintings = total_paint) → L = 3 :=
by
  -- Placeholder for the proof
  sorry

end large_paintings_count_l822_82209


namespace gifts_from_Pedro_l822_82216

theorem gifts_from_Pedro (gifts_from_Emilio gifts_from_Jorge total_gifts : ℕ)
  (h1 : gifts_from_Emilio = 11)
  (h2 : gifts_from_Jorge = 6)
  (h3 : total_gifts = 21) :
  total_gifts - (gifts_from_Emilio + gifts_from_Jorge) = 4 := by
  sorry

end gifts_from_Pedro_l822_82216


namespace find_number_l822_82256

theorem find_number (x : ℝ) (h : x = 12) : ( ( 17.28 / x ) / ( 3.6 * 0.2 ) ) = 2 := 
by
  -- Proof will be here
  sorry

end find_number_l822_82256


namespace wait_time_at_least_8_l822_82233

-- Define the conditions
variables (p₀ p : ℝ) (r x : ℝ)

-- Given conditions
def initial_BAC := p₀ = 89
def BAC_after_2_hours := p = 61
def BAC_decrease := p = p₀ * (Real.exp (r * x))
def decrease_in_2_hours := p = 89 * (Real.exp (r * 2))

-- The main goal to prove the time required is at least 8 hours
theorem wait_time_at_least_8 (h1 : p₀ = 89) (h2 : p = 61) (h3 : p = p₀ * Real.exp (r * x)) (h4 : 61 = 89 * Real.exp (2 * r)) : 
  ∃ x, 89 * Real.exp (r * x) < 20 ∧ x ≥ 8 :=
sorry

end wait_time_at_least_8_l822_82233


namespace johns_total_pay_l822_82227

-- Define the given conditions
def lastYearBonus : ℝ := 10000
def CAGR : ℝ := 0.05
def numYears : ℕ := 1
def projectsCompleted : ℕ := 8
def bonusPerProject : ℝ := 2000
def thisYearSalary : ℝ := 200000

-- Define the calculation for the first part of the bonus using the CAGR formula
def firstPartBonus (presentValue : ℝ) (growthRate : ℝ) (years : ℕ) : ℝ :=
  presentValue * (1 + growthRate)^years

-- Define the calculation for the second part of the bonus
def secondPartBonus (numProjects : ℕ) (bonusPerProject : ℝ) : ℝ :=
  numProjects * bonusPerProject

-- Define the total pay calculation
def totalPay (salary : ℝ) (bonus1 : ℝ) (bonus2 : ℝ) : ℝ :=
  salary + bonus1 + bonus2

-- The proof statement, given the conditions, prove the total pay is $226,500
theorem johns_total_pay : totalPay thisYearSalary (firstPartBonus lastYearBonus CAGR numYears) (secondPartBonus projectsCompleted bonusPerProject) = 226500 := 
by
  -- insert proof here
  sorry

end johns_total_pay_l822_82227


namespace calculate_expression_l822_82214

theorem calculate_expression : 5 * 7 + 9 * 4 - 36 / 3 + 48 / 4 = 71 := by
  sorry

end calculate_expression_l822_82214


namespace eval_expression_l822_82284

noncomputable def T := (1 / (Real.sqrt 10 - Real.sqrt 8)) + (1 / (Real.sqrt 8 - Real.sqrt 6)) + (1 / (Real.sqrt 6 - Real.sqrt 4))

theorem eval_expression : T = (Real.sqrt 10 + 2 * Real.sqrt 8 + 2 * Real.sqrt 6 + 2) / 2 := 
by
  sorry

end eval_expression_l822_82284


namespace age_twice_in_years_l822_82241

theorem age_twice_in_years (x : ℕ) : (40 + x = 2 * (12 + x)) → x = 16 :=
by {
  sorry
}

end age_twice_in_years_l822_82241


namespace problem_statement_l822_82225

theorem problem_statement (n : ℕ) : (-1 : ℤ) ^ n * (-1) ^ (2 * n + 1) * (-1) ^ (n + 1) = 1 := 
by
  sorry

end problem_statement_l822_82225


namespace gift_bag_combinations_l822_82274

theorem gift_bag_combinations (giftBags tissuePapers tags : ℕ) (h1 : giftBags = 10) (h2 : tissuePapers = 4) (h3 : tags = 5) : 
  giftBags * tissuePapers * tags = 200 := 
by 
  sorry

end gift_bag_combinations_l822_82274


namespace find_original_number_l822_82204

def original_four_digit_number (N : ℕ) : Prop :=
  N >= 1000 ∧ N < 10000 ∧ (70000 + N) - (10 * N + 7) = 53208

theorem find_original_number (N : ℕ) (h : original_four_digit_number N) : N = 1865 :=
by
  sorry

end find_original_number_l822_82204


namespace total_dolphins_correct_l822_82243

-- Define the initial number of dolphins
def initialDolphins : Nat := 65

-- Define the multiplier for the dolphins joining from the river
def joiningMultiplier : Nat := 3

-- Define the total number of dolphins after joining
def totalDolphins : Nat := initialDolphins + (joiningMultiplier * initialDolphins)

-- Prove that the total number of dolphins is 260
theorem total_dolphins_correct : totalDolphins = 260 := by
  sorry

end total_dolphins_correct_l822_82243


namespace two_pow_add_three_perfect_square_two_pow_add_one_perfect_square_l822_82263

theorem two_pow_add_three_perfect_square (n : ℕ) :
  ∃ k, 2^n + 3 = k^2 ↔ n = 0 :=
by {
  sorry
}

theorem two_pow_add_one_perfect_square (n : ℕ) :
  ∃ k, 2^n + 1 = k^2 ↔ n = 3 :=
by {
  sorry
}

end two_pow_add_three_perfect_square_two_pow_add_one_perfect_square_l822_82263


namespace Eugene_buys_two_pairs_of_shoes_l822_82208

theorem Eugene_buys_two_pairs_of_shoes :
  let tshirt_price : ℕ := 20
  let pants_price : ℕ := 80
  let shoes_price : ℕ := 150
  let discount_rate : ℕ := 10
  let discounted_price (price : ℕ) := price - (price * discount_rate / 100)
  let total_price (count1 count2 count3 : ℕ) (price1 price2 price3 : ℕ) :=
    (count1 * price1) + (count2 * price2) + (count3 * price3)
  let total_amount_paid : ℕ := 558
  let tshirts_bought : ℕ := 4
  let pants_bought : ℕ := 3
  let amount_left := total_amount_paid - discounted_price (tshirts_bought * tshirt_price + pants_bought * pants_price)
  let shoes_bought := amount_left / discounted_price shoes_price
  shoes_bought = 2 := 
sorry

end Eugene_buys_two_pairs_of_shoes_l822_82208


namespace derivative_at_1_l822_82217

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2^x) / x^2

theorem derivative_at_1 : (deriv f 1) = 2 * Real.log 2 - 3 := 
sorry

end derivative_at_1_l822_82217


namespace prove_f2_l822_82229

def func_condition (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x ^ 2 - y) + 2 * c * f x * y

theorem prove_f2 (c : ℝ) (f : ℝ → ℝ)
  (hf : func_condition f c) :
  (f 2 = 0 ∨ f 2 = 4) ∧ (2 * (if f 2 = 0 then 4 else if f 2 = 4 then 4 else 0) = 8) :=
by {
  sorry
}

end prove_f2_l822_82229


namespace count_valid_n_l822_82282

theorem count_valid_n:
  ( ∃ f: ℕ → ℕ, ∀ n, (0 < n ∧ n < 2012 → 7 ∣ (2^n - n^2) ↔ 7 ∣ (f n)) ∧ f 2012 = 576) → 
  ∃ valid_n_count: ℕ, valid_n_count = 576 := 
sorry

end count_valid_n_l822_82282


namespace quadratic_other_root_is_three_l822_82252

-- Steps for creating the Lean statement following the identified conditions
variable (b : ℝ)

theorem quadratic_other_root_is_three (h1 : ∀ x : ℝ, x^2 - 2 * x - b = 0 → (x = -1 ∨ x = 3)) : 
  ∀ x : ℝ, x^2 - 2 * x - b = 0 → x = -1 ∨ x = 3 :=
by
  -- The proof is omitted
  exact h1

end quadratic_other_root_is_three_l822_82252


namespace price_of_large_slice_is_250_l822_82277

noncomputable def priceOfLargeSlice (totalSlices soldSmallSlices totalRevenue smallSlicePrice: ℕ) : ℕ :=
  let totalRevenueSmallSlices := soldSmallSlices * smallSlicePrice
  let totalRevenueLargeSlices := totalRevenue - totalRevenueSmallSlices
  let soldLargeSlices := totalSlices - soldSmallSlices
  totalRevenueLargeSlices / soldLargeSlices

theorem price_of_large_slice_is_250 :
  priceOfLargeSlice 5000 2000 1050000 150 = 250 :=
by
  sorry

end price_of_large_slice_is_250_l822_82277


namespace sum_mod_17_l822_82289

/--
Given the sum of the numbers 82, 83, 84, 85, 86, 87, 88, and 89, and the divisor 17,
prove that the remainder when dividing this sum by 17 is 11.
-/
theorem sum_mod_17 : (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17 = 11 :=
by
  sorry

end sum_mod_17_l822_82289


namespace valbonne_middle_school_l822_82265

theorem valbonne_middle_school (students : Finset ℕ) (h : students.card = 367) :
  ∃ (date1 date2 : ℕ), date1 ≠ date2 ∧ date1 = date2 ∧ date1 ∈ students ∧ date2 ∈ students :=
by {
  sorry
}

end valbonne_middle_school_l822_82265


namespace total_cost_price_l822_82238

variables (C_table C_chair C_shelf : ℝ)

axiom h1 : 1.24 * C_table = 8091
axiom h2 : 1.18 * C_chair = 5346
axiom h3 : 1.30 * C_shelf = 11700

theorem total_cost_price :
  C_table + C_chair + C_shelf = 20055.51 :=
sorry

end total_cost_price_l822_82238


namespace original_decimal_number_l822_82211

theorem original_decimal_number (x : ℝ) (h₁ : 0 < x) (h₂ : 100 * x = 9 * (1 / x)) : x = 3 / 10 :=
by
  sorry

end original_decimal_number_l822_82211


namespace remainder_when_divided_by_39_l822_82250

theorem remainder_when_divided_by_39 (N : ℤ) (h1 : ∃ k : ℤ, N = 13 * k + 3) : N % 39 = 3 :=
sorry

end remainder_when_divided_by_39_l822_82250


namespace max_intersections_circle_quadrilateral_max_intersections_correct_l822_82234

-- Define the intersection property of a circle and a line segment
def max_intersections_per_side (circle : Type) (line_segment : Type) : ℕ := 2

-- Define a quadrilateral as a shape having four sides
def sides_of_quadrilateral : ℕ := 4

-- The theorem stating the maximum number of intersection points between a circle and a quadrilateral
theorem max_intersections_circle_quadrilateral (circle : Type) (quadrilateral : Type) : Prop :=
  max_intersections_per_side circle quadrilateral * sides_of_quadrilateral = 8

-- Proof is skipped with 'sorry'
theorem max_intersections_correct (circle : Type) (quadrilateral : Type) :
  max_intersections_circle_quadrilateral circle quadrilateral :=
by
  sorry

end max_intersections_circle_quadrilateral_max_intersections_correct_l822_82234


namespace problem1_problem2_problem3_problem4_l822_82296

-- Problem 1
theorem problem1 : ∃ n : ℕ, n = 3^4 ∧ n = 81 :=
by
  sorry

-- Problem 2
theorem problem2 : ∃ n : ℕ, n = (Nat.choose 4 2) * 6 ∧ n = 36 :=
by
  sorry

-- Problem 3
theorem problem3 : ∃ n : ℕ, n = Nat.choose 4 2 ∧ n = 6 :=
by
  sorry

-- Problem 4
theorem problem4 : ∃ n : ℕ, n = 1 + (Nat.choose 4 1 + Nat.choose 4 2 / 2) + 6 ∧ n = 14 :=
by
  sorry

end problem1_problem2_problem3_problem4_l822_82296


namespace inequality_proof_l822_82242

theorem inequality_proof
  (x y : ℝ) (h1 : x^2 + x * y + y^2 = (x + y)^2 - x * y) 
  (h2 : x + y ≥ 2 * Real.sqrt (x * y)) : 
  x + y + Real.sqrt (x * y) ≤ 3 * (x + y - Real.sqrt (x * y)) := 
by
  sorry

end inequality_proof_l822_82242


namespace smallest_determinant_and_min_ab_l822_82248

def determinant (a b : ℤ) : ℤ :=
  36 * b - 81 * a

theorem smallest_determinant_and_min_ab :
  (∃ (a b : ℤ), 0 < determinant a b ∧ determinant a b = 9 ∧ ∀ a' b', determinant a' b' = 9 → a' + b' ≥ a + b) ∧
  (∃ (a b : ℤ), a = 3 ∧ b = 7) :=
sorry

end smallest_determinant_and_min_ab_l822_82248


namespace max_temp_difference_l822_82279

-- Define the highest and lowest temperatures
def highest_temp : ℤ := 3
def lowest_temp : ℤ := -3

-- State the theorem for maximum temperature difference
theorem max_temp_difference : highest_temp - lowest_temp = 6 := 
by 
  -- Provide the proof here
  sorry

end max_temp_difference_l822_82279


namespace sin_identity_l822_82201

open Real

noncomputable def alpha : ℝ := π  -- since we are considering angles in radians

theorem sin_identity (h1 : sin α = 3/5) (h2 : π/2 < α ∧ α < 3 * π / 2) :
  sin (5 * π / 2 - α) = -4 / 5 :=
by sorry

end sin_identity_l822_82201


namespace min_value_sin6_cos6_l822_82244

open Real

theorem min_value_sin6_cos6 (x : ℝ) : sin x ^ 6 + 2 * cos x ^ 6 ≥ 2 / 3 :=
by
  sorry

end min_value_sin6_cos6_l822_82244


namespace probability_green_or_blue_l822_82281

-- Define the properties of the 10-sided die
def total_faces : ℕ := 10
def red_faces : ℕ := 4
def yellow_faces : ℕ := 3
def green_faces : ℕ := 2
def blue_faces : ℕ := 1

-- Define the number of favorable outcomes
def favorable_outcomes : ℕ := green_faces + blue_faces

-- Define the probability function
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- The theorem to prove
theorem probability_green_or_blue :
  probability favorable_outcomes total_faces = 3 / 10 :=
by
  sorry

end probability_green_or_blue_l822_82281


namespace maximum_integer_value_of_a_l822_82286

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + (2 - a) * x - a * Real.log x

theorem maximum_integer_value_of_a (a : ℝ) (h : ∀ x ≥ 1, f x a > 0) : a ≤ 2 :=
sorry

end maximum_integer_value_of_a_l822_82286


namespace total_new_people_last_year_l822_82269

-- Define the number of new people born and the number of people immigrated
def new_people_born : ℕ := 90171
def people_immigrated : ℕ := 16320

-- Prove that the total number of new people is 106491
theorem total_new_people_last_year : new_people_born + people_immigrated = 106491 := by
  sorry

end total_new_people_last_year_l822_82269


namespace count_house_numbers_l822_82215

def isPrime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def twoDigitPrimesBetween40And60 : List ℕ :=
  [41, 43, 47, 53, 59]

theorem count_house_numbers : 
  ∃ n : ℕ, n = 20 ∧ 
  ∀ (AB CD : ℕ), 
  AB ∈ twoDigitPrimesBetween40And60 → 
  CD ∈ twoDigitPrimesBetween40And60 → 
  AB ≠ CD → 
  true :=
by
  sorry

end count_house_numbers_l822_82215


namespace yellow_scores_l822_82259

theorem yellow_scores (W B : ℕ) 
  (h₁ : W / B = 7 / 6)
  (h₂ : (2 / 3 : ℚ) * (W - B) = 4) : 
  W + B = 78 :=
sorry

end yellow_scores_l822_82259


namespace max_value_is_one_l822_82297

noncomputable def max_expression (a b : ℝ) : ℝ :=
(a + b) ^ 2 / (a ^ 2 + 2 * a * b + b ^ 2)

theorem max_value_is_one {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  max_expression a b ≤ 1 :=
sorry

end max_value_is_one_l822_82297


namespace problem_integer_condition_l822_82222

theorem problem_integer_condition (a : ℤ) (h1 : 0 ≤ a ∧ a ≤ 14)
  (h2 : (235935623 * 74^0 + 2 * 74^1 + 6 * 74^2 + 5 * 74^3 + 3 * 74^4 + 9 * 74^5 + 
         5 * 74^6 + 3 * 74^7 + 2 * 74^8 - a) % 15 = 0) : a = 0 :=
by
  sorry

end problem_integer_condition_l822_82222


namespace eval_diff_squares_l822_82212

theorem eval_diff_squares : 81^2 - 49^2 = 4160 :=
by
  sorry

end eval_diff_squares_l822_82212


namespace emma_age_when_sister_is_56_l822_82258

theorem emma_age_when_sister_is_56 (e s : ℕ) (he : e = 7) (hs : s = e + 9) : 
  (s + (56 - s) - 9 = 47) :=
by {
  sorry
}

end emma_age_when_sister_is_56_l822_82258


namespace rectangle_breadth_l822_82246

/-- The breadth of the rectangle is 10 units given that
1. The length of the rectangle is two-fifths of the radius of a circle.
2. The radius of the circle is equal to the side of the square.
3. The area of the square is 1225 sq. units.
4. The area of the rectangle is 140 sq. units. -/
theorem rectangle_breadth (r l b : ℝ) (h_radius : r = 35) (h_length : l = (2 / 5) * r) (h_square : 35 * 35 = 1225) (h_area_rect : l * b = 140) : b = 10 :=
by
  sorry

end rectangle_breadth_l822_82246


namespace possible_values_of_m_l822_82253

open Set

variable (A B : Set ℤ)
variable (m : ℤ)

theorem possible_values_of_m (h₁ : A = {1, 2, m * m}) (h₂ : B = {1, m}) (h₃ : B ⊆ A) :
  m = 0 ∨ m = 2 :=
  sorry

end possible_values_of_m_l822_82253


namespace point_C_number_l822_82226

theorem point_C_number (B C: ℝ) (h1 : B = 3) (h2 : |C - B| = 2) :
  C = 1 ∨ C = 5 := 
by {
  sorry
}

end point_C_number_l822_82226


namespace eleven_pow_603_mod_500_eq_331_l822_82293

theorem eleven_pow_603_mod_500_eq_331 : 11^603 % 500 = 331 := by
  sorry

end eleven_pow_603_mod_500_eq_331_l822_82293


namespace flour_for_recipe_l822_82200

theorem flour_for_recipe (flour_needed shortening_have : ℚ)
  (flour_ratio shortening_ratio : ℚ) 
  (ratio : flour_ratio / shortening_ratio = 5)
  (shortening_used : shortening_ratio = 2 / 3) :
  flour_needed = 10 / 3 := 
by 
  sorry

end flour_for_recipe_l822_82200


namespace abs_y_lt_inequality_sum_l822_82260

-- Problem (1)
theorem abs_y_lt {
  x y : ℝ
} (h1 : |x - y| < 1) (h2 : |2 * x + y| < 1) :
  |y| < 1 := by
  sorry

-- Problem (2)
theorem inequality_sum {
  a b c d : ℝ
} (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b) + 1 / (b - c) + 1 / (c - d)) ≥ 9 / (a - d) := by
  sorry

end abs_y_lt_inequality_sum_l822_82260


namespace sequence_periodicity_l822_82236

variable {a b : ℕ → ℤ}

theorem sequence_periodicity (h : ∀ n ≥ 3, 
    (a n - a (n - 1)) * (a n - a (n - 2)) + 
    (b n - b (n - 1)) * (b n - b (n - 2)) = 0) : 
    ∃ k > 0, a k + b k = a (k + 2018) + b (k + 2018) := 
    by
    sorry

end sequence_periodicity_l822_82236


namespace prove_a_zero_l822_82218

-- Define two natural numbers a and b
variables (a b : ℕ)

-- Condition: For every natural number n, 2^n * a + b is a perfect square
def condition := ∀ n : ℕ, ∃ k : ℕ, 2^n * a + b = k^2

-- Statement to prove: a = 0
theorem prove_a_zero (h : condition a b) : a = 0 := sorry

end prove_a_zero_l822_82218


namespace min_value_of_expression_l822_82278

noncomputable def quadratic_function_min_value (a b c : ℝ) : ℝ :=
  (3 * (a * 1^2 + b * 1 + c) + 6 * (a * 0^2 + b * 0 + c) - (a * (-1)^2 + b * (-1) + c)) /
  ((a * 0^2 + b * 0 + c) - (a * (-2)^2 + b * (-2) + c))

theorem min_value_of_expression (a b c : ℝ)
  (h1 : b > 2 * a)
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0)
  (h3 : a > 0) :
  quadratic_function_min_value a b c = 12 :=
sorry

end min_value_of_expression_l822_82278


namespace units_sold_to_customer_c_l822_82283

theorem units_sold_to_customer_c 
  (initial_units : ℕ)
  (defective_units : ℕ)
  (units_a : ℕ)
  (units_b : ℕ)
  (units_c : ℕ)
  (h_initial : initial_units = 20)
  (h_defective : defective_units = 5)
  (h_units_a : units_a = 3)
  (h_units_b : units_b = 5)
  (h_non_defective : initial_units - defective_units = 15)
  (h_sold_all : units_a + units_b + units_c = 15) :
  units_c = 7 := by
  -- use sorry to skip the proof
  sorry

end units_sold_to_customer_c_l822_82283


namespace laptop_cost_l822_82295

theorem laptop_cost
  (C : ℝ) (down_payment := 0.2 * C + 20) (installments_paid := 65 * 4) (balance_after_4_months := 520)
  (h : C - (down_payment + installments_paid) = balance_after_4_months) :
  C = 1000 :=
by
  sorry

end laptop_cost_l822_82295


namespace range_b_values_l822_82207

theorem range_b_values (f g : ℝ → ℝ) (a b : ℝ) 
  (hf : ∀ x, f x = Real.exp x - 1) 
  (hg : ∀ x, g x = -x^2 + 4*x - 3) 
  (h : f a = g b) : 
  b ∈ Set.univ :=
by sorry

end range_b_values_l822_82207


namespace rectangle_difference_l822_82268

theorem rectangle_difference (A d x y : ℝ) (h1 : x * y = A) (h2 : x^2 + y^2 = d^2) :
  x - y = 2 * Real.sqrt A := 
sorry

end rectangle_difference_l822_82268


namespace books_read_by_Megan_l822_82240

theorem books_read_by_Megan 
    (M : ℕ)
    (Kelcie : ℕ := M / 4)
    (Greg : ℕ := 2 * (M / 4) + 9)
    (total : M + Kelcie + Greg = 65) :
  M = 32 :=
by sorry

end books_read_by_Megan_l822_82240


namespace cos_17pi_over_4_l822_82223

theorem cos_17pi_over_4 : Real.cos (17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end cos_17pi_over_4_l822_82223


namespace problem_85_cube_plus_3_85_square_plus_3_85_plus_1_l822_82251

theorem problem_85_cube_plus_3_85_square_plus_3_85_plus_1 :
  85^3 + 3 * (85^2) + 3 * 85 + 1 = 636256 := 
sorry

end problem_85_cube_plus_3_85_square_plus_3_85_plus_1_l822_82251


namespace eggs_given_by_Andrew_l822_82294

variable (total_eggs := 222)
variable (eggs_to_buy := 67)
variable (eggs_given : ℕ)

theorem eggs_given_by_Andrew :
  eggs_given = total_eggs - eggs_to_buy ↔ eggs_given = 155 := 
by 
  sorry

end eggs_given_by_Andrew_l822_82294


namespace solution_set_of_quadratic_inequality_2_l822_82261

-- Definitions
variables {a b c x : ℝ}
def quadratic_inequality_1 (a b c x : ℝ) := a * x^2 + b * x + c < 0
def quadratic_inequality_2 (a b c x : ℝ) := a * x^2 - b * x + c > 0

-- Conditions
axiom condition_1 : ∀ x, quadratic_inequality_1 a b c x ↔ (x < -2 ∨ x > -1/2)
axiom condition_2 : a < 0
axiom condition_3 : ∃ x, a * x^2 + b * x + c = 0 ∧ (x = -2 ∨ x = -1/2)
axiom condition_4 : b = 5 * a / 2
axiom condition_5 : c = a

-- Proof Problem
theorem solution_set_of_quadratic_inequality_2 : ∀ x, quadratic_inequality_2 a b c x ↔ (1/2 < x ∧ x < 2) :=
by
  -- Proof goes here
  sorry

end solution_set_of_quadratic_inequality_2_l822_82261


namespace compound_interest_l822_82266

noncomputable def final_amount (P : ℕ) (r : ℚ) (t : ℕ) :=
  P * ((1 : ℚ) + r) ^ t

theorem compound_interest : 
  final_amount 20000 0.20 10 = 123834.73 := 
by 
  sorry

end compound_interest_l822_82266


namespace find_x_l822_82202

theorem find_x (x : ℚ) : (8 + 12 + 24) / 3 = (16 + x) / 2 → x = 40 / 3 :=
by
  intro h
  sorry

end find_x_l822_82202


namespace quadratic_function_correct_value_l822_82254

noncomputable def quadratic_function_value (a b x x1 x2 : ℝ) :=
  a * x^2 + b * x + 5

theorem quadratic_function_correct_value
  (a b x1 x2 : ℝ)
  (h_a : a ≠ 0)
  (h_A : quadratic_function_value a b x1 x1 x2 = 2002)
  (h_B : quadratic_function_value a b x2 x1 x2 = 2002) :
  quadratic_function_value a b (x1 + x2) x1 x2 = 5 :=
by
  sorry

end quadratic_function_correct_value_l822_82254


namespace emily_jumping_game_l822_82249

def tiles_number (n : ℕ) : Prop :=
  n % 2 = 1 ∧ n % 3 = 2 ∧ n % 5 = 2

theorem emily_jumping_game : tiles_number 47 :=
by
  unfold tiles_number
  sorry

end emily_jumping_game_l822_82249


namespace blue_first_red_second_probability_l822_82271

-- Define the initial conditions
def initial_red_marbles : ℕ := 4
def initial_white_marbles : ℕ := 6
def initial_blue_marbles : ℕ := 2
def total_marbles : ℕ := initial_red_marbles + initial_white_marbles + initial_blue_marbles

-- Probability calculation under the given conditions
def probability_blue_first : ℚ := initial_blue_marbles / total_marbles
def remaining_marbles_after_blue : ℕ := total_marbles - 1
def remaining_red_marbles : ℕ := initial_red_marbles
def probability_red_second_given_blue_first : ℚ := remaining_red_marbles / remaining_marbles_after_blue

-- Combined probability
def combined_probability : ℚ := probability_blue_first * probability_red_second_given_blue_first

-- The statement to be proved
theorem blue_first_red_second_probability :
  combined_probability = 2 / 33 :=
sorry

end blue_first_red_second_probability_l822_82271


namespace Martha_knitting_grandchildren_l822_82247

theorem Martha_knitting_grandchildren (T_hat T_scarf T_mittens T_socks T_sweater T_total : ℕ)
  (h_hat : T_hat = 2) (h_scarf : T_scarf = 3) (h_mittens : T_mittens = 2)
  (h_socks : T_socks = 3) (h_sweater : T_sweater = 6) (h_total : T_total = 48) :
  (T_total / (T_hat + T_scarf + T_mittens + T_socks + T_sweater)) = 3 := by
  sorry

end Martha_knitting_grandchildren_l822_82247


namespace polynomial_root_arithmetic_sequence_l822_82203

theorem polynomial_root_arithmetic_sequence :
  (∃ (a d : ℝ), 
    (64 * (a - d)^3 + 144 * (a - d)^2 + 92 * (a - d) + 15 = 0) ∧
    (64 * a^3 + 144 * a^2 + 92 * a + 15 = 0) ∧
    (64 * (a + d)^3 + 144 * (a + d)^2 + 92 * (a + d) + 15 = 0) ∧
    (2 * d = 1)) := sorry

end polynomial_root_arithmetic_sequence_l822_82203


namespace smallest_integer_base_cube_l822_82210

theorem smallest_integer_base_cube (b : ℤ) (h1 : b > 5) (h2 : ∃ k : ℤ, 1 * b + 2 = k^3) : b = 6 :=
sorry

end smallest_integer_base_cube_l822_82210


namespace panels_per_home_panels_needed_per_home_l822_82230

theorem panels_per_home (P : ℕ) (total_homes : ℕ) (shortfall : ℕ) (homes_installed : ℕ) :
  total_homes = 20 →
  shortfall = 50 →
  homes_installed = 15 →
  (P - shortfall) / homes_installed = P / total_homes →
  P = 200 :=
by
  intro h1 h2 h3 h4
  sorry

theorem panels_needed_per_home :
  (200 / 20) = 10 :=
by
  sorry

end panels_per_home_panels_needed_per_home_l822_82230


namespace solution_set_of_inequality_l822_82232

theorem solution_set_of_inequality (f : ℝ → ℝ) :
  (∀ x, f x = f (-x)) →
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ x2 → f x2 ≤ f x1) →
  (f 1 = 0) →
  {x : ℝ | f (x - 3) ≥ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 4} :=
by
  intros h_even h_mono h_f1
  sorry

end solution_set_of_inequality_l822_82232


namespace sin_bound_l822_82287

theorem sin_bound (a : ℝ) (h : ¬ ∃ x : ℝ, Real.sin x > a) : a ≥ 1 := 
sorry

end sin_bound_l822_82287


namespace road_trip_ratio_l822_82213

-- Problem Definitions
variable (x d3 total grand_total : ℕ)
variable (hx1 : total = x + 2 * x + d3 + 2 * (x + 2 * x + d3))
variable (hx2 : d3 = 40)
variable (hx3 : total = 560)
variable (hx4 : grand_total = d3 / x)

-- Proof Statement
theorem road_trip_ratio (hx1 : total = x + 2 * x + d3 + 2 * (x + 2 * x + d3)) 
  (hx2 : d3 = 40) (hx3 : total = 560) : grand_total = 9 / 11 := by
  sorry

end road_trip_ratio_l822_82213


namespace age_double_in_years_l822_82220

theorem age_double_in_years (S M X: ℕ) (h1: M = S + 22) (h2: S = 20) (h3: M + X = 2 * (S + X)) : X = 2 :=
by 
  sorry

end age_double_in_years_l822_82220


namespace percentage_more_than_cost_price_l822_82291

noncomputable def SP : ℝ := 7350
noncomputable def CP : ℝ := 6681.818181818181

theorem percentage_more_than_cost_price : 
  (SP - CP) / CP * 100 = 10 :=
by
  sorry

end percentage_more_than_cost_price_l822_82291


namespace division_multiplication_l822_82273

theorem division_multiplication : (0.25 / 0.005) * 2 = 100 := 
by 
  sorry

end division_multiplication_l822_82273


namespace matrix_power_50_l822_82205

-- Defining the matrix A.
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![5, 1], 
    ![-12, -3]]

-- Statement of the theorem
theorem matrix_power_50 :
  A ^ 50 = ![![301, 50], 
               ![-900, -301]] :=
by
  sorry

end matrix_power_50_l822_82205


namespace bus_speed_including_stoppages_l822_82219

theorem bus_speed_including_stoppages
  (speed_without_stoppages : ℝ)
  (stoppage_time : ℝ)
  (remaining_time_ratio : ℝ)
  (h1 : speed_without_stoppages = 12)
  (h2 : stoppage_time = 0.5)
  (h3 : remaining_time_ratio = 1 - stoppage_time) :
  (speed_without_stoppages * remaining_time_ratio) = 6 := 
by
  sorry

end bus_speed_including_stoppages_l822_82219


namespace f_neg_m_l822_82298

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

-- State the problem as a theorem
theorem f_neg_m (a b m : ℝ) (h : f a b m = 6) : f a b (-m) = -4 :=
by
  -- Proof is not required
  sorry

end f_neg_m_l822_82298


namespace packing_peanuts_per_large_order_l822_82239

/-- Definitions of conditions as stated -/
def large_orders : ℕ := 3
def small_orders : ℕ := 4
def total_peanuts_used : ℕ := 800
def peanuts_per_small : ℕ := 50

/-- The statement to prove, ensuring all conditions are utilized in the definitions -/
theorem packing_peanuts_per_large_order : 
  ∃ L, large_orders * L + small_orders * peanuts_per_small = total_peanuts_used ∧ L = 200 := 
by
  use 200
  -- Adding the necessary proof steps
  have h1 : large_orders = 3 := rfl
  have h2 : small_orders = 4 := rfl
  have h3 : peanuts_per_small = 50 := rfl
  have h4 : total_peanuts_used = 800 := rfl
  sorry

end packing_peanuts_per_large_order_l822_82239


namespace prove_range_of_p_l822_82221

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x - 1

def A (x : ℝ) : Prop := x > 2
def no_pre_image_in_A (p : ℝ) : Prop := ∀ x, A x → f x ≠ p

theorem prove_range_of_p (p : ℝ) : no_pre_image_in_A p ↔ p > -1 := by
  sorry

end prove_range_of_p_l822_82221


namespace geometric_sequence_common_ratio_l822_82280

-- Define the geometric sequence with properties
def increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ a n < a (n + 1)

-- Main theorem
theorem geometric_sequence_common_ratio {a : ℕ → ℝ} {q : ℝ} (h_seq : increasing_geometric_sequence a q) (h_a1 : a 0 > 0) (h_eqn : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) :
  q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l822_82280


namespace photos_to_cover_poster_l822_82299

/-
We are given a poster of dimensions 3 feet by 5 feet, and photos of dimensions 3 inches by 5 inches.
We need to prove that the number of such photos required to cover the poster is 144.
-/

-- Convert feet to inches
def feet_to_inches(feet : ℕ) : ℕ := 12 * feet

-- Dimensions of the poster in inches
def poster_height_in_inches := feet_to_inches 3
def poster_width_in_inches := feet_to_inches 5

-- Area of the poster
def poster_area : ℕ := poster_height_in_inches * poster_width_in_inches

-- Dimensions and area of one photo in inches
def photo_height := 3
def photo_width := 5
def photo_area : ℕ := photo_height * photo_width

-- Number of photos required to cover the poster
def number_of_photos : ℕ := poster_area / photo_area

-- Theorem stating the required number of photos is 144
theorem photos_to_cover_poster : number_of_photos = 144 := by
  -- Proof is omitted
  sorry

end photos_to_cover_poster_l822_82299


namespace inequality_C_false_l822_82257

theorem inequality_C_false (a b : ℝ) (ha : 1 < a) (hb : 1 < b) : (1 / a) ^ (1 / b) ≤ 1 := 
sorry

end inequality_C_false_l822_82257
