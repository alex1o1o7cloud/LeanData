import Mathlib

namespace total_hotdogs_sold_l1752_175239

theorem total_hotdogs_sold (small_hotdogs large_hotdogs : ℕ) 
  (h1 : small_hotdogs = 58) 
  (h2 : large_hotdogs = 21) : 
  small_hotdogs + large_hotdogs = 79 := by
  sorry

end total_hotdogs_sold_l1752_175239


namespace sum_first_10_even_integers_l1752_175249

/-- The sum of the first n positive even integers -/
def sum_first_n_even_integers (n : ℕ) : ℕ :=
  2 * n * (n + 1)

/-- Theorem: The sum of the first 10 positive even integers is 110 -/
theorem sum_first_10_even_integers :
  sum_first_n_even_integers 10 = 110 := by
  sorry

end sum_first_10_even_integers_l1752_175249


namespace unique_square_divisible_by_three_in_range_l1752_175274

theorem unique_square_divisible_by_three_in_range : ∃! y : ℕ, 
  (∃ x : ℕ, y = x^2) ∧ 
  (∃ k : ℕ, y = 3 * k) ∧ 
  50 < y ∧ y < 120 :=
by
  -- The proof goes here
  sorry

end unique_square_divisible_by_three_in_range_l1752_175274


namespace part_one_part_two_l1752_175285

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a + 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Part 1
theorem part_one : (Set.univ \ P 3) ∩ Q = {x | -2 ≤ x ∧ x < 4} := by sorry

-- Part 2
theorem part_two : {a : ℝ | P a ⊂ Q ∧ P a ≠ ∅} = {a : ℝ | 0 ≤ a ∧ a ≤ 2} := by sorry

end part_one_part_two_l1752_175285


namespace unique_solution_condition_l1752_175280

theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, x^2 - a * abs x + a^2 - 3 = 0) ↔ a = -Real.sqrt 3 := by
sorry

end unique_solution_condition_l1752_175280


namespace pi_over_three_irrational_l1752_175299

theorem pi_over_three_irrational : Irrational (π / 3) :=
by
  sorry

end pi_over_three_irrational_l1752_175299


namespace square_difference_l1752_175210

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 6) :
  (x - y)^2 = 57 := by
  sorry

end square_difference_l1752_175210


namespace max_notebooks_purchase_l1752_175291

theorem max_notebooks_purchase (notebook_price : ℕ) (available_money : ℚ) : 
  notebook_price = 45 → available_money = 40.5 → 
  ∃ max_notebooks : ℕ, max_notebooks = 90 ∧ 
  (max_notebooks : ℚ) * (notebook_price : ℚ) / 100 ≤ available_money ∧
  ∀ n : ℕ, (n : ℚ) * (notebook_price : ℚ) / 100 ≤ available_money → n ≤ max_notebooks :=
by sorry

end max_notebooks_purchase_l1752_175291


namespace linda_cookie_distribution_l1752_175289

/-- Calculates the number of cookies per student given the problem conditions -/
def cookies_per_student (classmates : ℕ) (cookies_per_batch : ℕ) 
  (choc_chip_batches : ℕ) (oatmeal_batches : ℕ) (additional_batches : ℕ) : ℕ :=
  let total_cookies := (choc_chip_batches + oatmeal_batches + additional_batches) * cookies_per_batch
  total_cookies / classmates

/-- Proves that given the problem conditions, each student receives 10 cookies -/
theorem linda_cookie_distribution : 
  cookies_per_student 24 (4 * 12) 2 1 2 = 10 := by
  sorry

end linda_cookie_distribution_l1752_175289


namespace f_range_on_interval_l1752_175260

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.exp x * (Real.sin x + Real.cos x)

theorem f_range_on_interval :
  let a := 0
  let b := Real.pi / 2
  ∃ (min max : ℝ), 
    (∀ x ∈ Set.Icc a b, f x ≥ min ∧ f x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc a b, f x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc a b, f x₂ = max) ∧
    min = 1/2 ∧
    max = (1/2) * Real.exp (Real.pi / 2) :=
by sorry

end f_range_on_interval_l1752_175260


namespace ellipse_tangent_property_l1752_175221

/-- Ellipse passing through a point with specific tangent properties -/
theorem ellipse_tangent_property (m : ℝ) (r : ℝ) (h_m : m > 0) (h_r : r > 0) :
  (∃ (E F : ℝ × ℝ),
    -- E and F are on the ellipse
    (E.1^2 / 4 + E.2^2 / m = 1) ∧
    (F.1^2 / 4 + F.2^2 / m = 1) ∧
    -- A is on the ellipse
    (1^2 / 4 + (3/2)^2 / m = 1) ∧
    -- Slopes form arithmetic sequence
    (∃ (k : ℝ),
      (F.2 - 3/2) / (F.1 - 1) = k ∧
      (E.2 - 3/2) / (E.1 - 1) = -k ∧
      (F.2 - E.2) / (F.1 - E.1) = 3*k) ∧
    -- AE and AF are tangent to the circle
    ((1 - 2)^2 + (3/2 - 3/2)^2 = r^2)) →
  r = Real.sqrt 37 / 37 :=
by sorry

end ellipse_tangent_property_l1752_175221


namespace social_media_ratio_l1752_175269

/-- Represents the daily phone usage in hours -/
def daily_phone_usage : ℝ := 16

/-- Represents the weekly social media usage in hours -/
def weekly_social_media_usage : ℝ := 56

/-- Represents the number of days in a week -/
def days_in_week : ℝ := 7

/-- Theorem: The ratio of daily time spent on social media to total daily time spent on phone is 1:2 -/
theorem social_media_ratio : 
  (weekly_social_media_usage / days_in_week) / daily_phone_usage = 1 / 2 := by
  sorry

end social_media_ratio_l1752_175269


namespace f_derivative_at_1_l1752_175257

-- Define the function f
def f (x : ℝ) : ℝ := (2023 - 2022 * x) ^ 3

-- State the theorem
theorem f_derivative_at_1 : 
  (deriv f) 1 = -6066 := by sorry

end f_derivative_at_1_l1752_175257


namespace cubic_roots_sum_product_l1752_175245

theorem cubic_roots_sum_product (α β γ : ℂ) (u v w : ℂ) : 
  (∀ x : ℂ, x^3 + 5*x^2 + 7*x - 13 = (x - α) * (x - β) * (x - γ)) →
  (∀ x : ℂ, x^3 + u*x^2 + v*x + w = (x - (α + β)) * (x - (β + γ)) * (x - (γ + α))) →
  w = 48 := by
sorry

end cubic_roots_sum_product_l1752_175245


namespace complement_of_union_equals_five_l1752_175233

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 2}

-- Define set N
def N : Finset Nat := {3, 4}

-- Theorem statement
theorem complement_of_union_equals_five :
  (U \ (M ∪ N)) = {5} := by sorry

end complement_of_union_equals_five_l1752_175233


namespace min_value_theorem_l1752_175297

theorem min_value_theorem (x y : ℝ) (h : x^2 * y^2 + y^4 = 1) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 ∧ ∀ (z w : ℝ), z^2 * w^2 + w^4 = 1 → x^2 + 3 * y^2 ≤ z^2 + 3 * w^2 :=
sorry

end min_value_theorem_l1752_175297


namespace triangle_area_implies_q_value_l1752_175219

/-- Given a triangle ABC with vertices A(3, 15), B(15, 0), and C(0, q), 
    prove that if the area of the triangle is 50, then q = 125/12 -/
theorem triangle_area_implies_q_value (q : ℝ) : 
  let A : ℝ × ℝ := (3, 15)
  let B : ℝ × ℝ := (15, 0)
  let C : ℝ × ℝ := (0, q)
  let triangle_area := (abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2))) / 2
  triangle_area = 50 → q = 125 / 12 := by
  sorry

#check triangle_area_implies_q_value

end triangle_area_implies_q_value_l1752_175219


namespace tire_purchase_cost_total_cost_proof_l1752_175205

/-- Calculates the total cost of purchasing tires with given prices and tax rate -/
theorem tire_purchase_cost (num_tires : ℕ) (price1 : ℚ) (price2 : ℚ) (tax_rate : ℚ) : ℚ :=
  let first_group_cost := min num_tires 4 * price1
  let second_group_cost := max (num_tires - 4) 0 * price2
  let subtotal := first_group_cost + second_group_cost
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Proves that the total cost of purchasing 8 tires with given prices and tax rate is 3.78 -/
theorem total_cost_proof :
  tire_purchase_cost 8 (1/2) (2/5) (1/20) = 189/50 :=
by sorry

end tire_purchase_cost_total_cost_proof_l1752_175205


namespace inverse_proportion_problem_l1752_175216

/-- Given that x and y are inversely proportional, prove that y = -56.25 when x = -12,
    given that x = 3y when x + y = 60 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) : 
  (∀ x' y', x' * y' = k) →  -- x and y are inversely proportional
  (∃ x₀ y₀, x₀ = 3 * y₀ ∧ x₀ + y₀ = 60) →  -- when their sum is 60, x is three times y
  (x = -12 → y = -56.25) :=  -- y = -56.25 when x = -12
by sorry

end inverse_proportion_problem_l1752_175216


namespace sin_cos_difference_equals_negative_half_l1752_175262

theorem sin_cos_difference_equals_negative_half :
  Real.sin (119 * π / 180) * Real.cos (91 * π / 180) - 
  Real.sin (91 * π / 180) * Real.sin (29 * π / 180) = -1/2 :=
by sorry

end sin_cos_difference_equals_negative_half_l1752_175262


namespace product_49_sum_0_l1752_175235

theorem product_49_sum_0 (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
  a * b * c * d = 49 → 
  a + b + c + d = 0 := by
sorry

end product_49_sum_0_l1752_175235


namespace greatest_divisible_by_cubes_l1752_175283

theorem greatest_divisible_by_cubes : ∃ (n : ℕ), n = 60 ∧ 
  (∀ (m : ℕ), m^3 ≤ n → n % m = 0) ∧
  (∀ (k : ℕ), k > n → ∃ (m : ℕ), m^3 ≤ k ∧ k % m ≠ 0) :=
by sorry

end greatest_divisible_by_cubes_l1752_175283


namespace four_fours_theorem_l1752_175201

def is_valid_expression (e : ℕ → ℕ) : Prop :=
  ∃ (a b c d f : ℕ), 
    (a = 4 ∧ b = 4 ∧ c = 4 ∧ d = 4 ∧ f = 4) ∧
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 22 → e n = n)

theorem four_fours_theorem :
  ∃ e : ℕ → ℕ, is_valid_expression e :=
sorry

end four_fours_theorem_l1752_175201


namespace equation_proof_l1752_175248

theorem equation_proof : 529 - 2 * 23 * 8 + 64 = 225 := by
  sorry

end equation_proof_l1752_175248


namespace job_completion_time_solution_l1752_175225

/-- Represents the time taken by three machines working together to complete a job -/
def job_completion_time (y : ℝ) : Prop :=
  let machine_a_time := y + 4
  let machine_b_time := y + 3
  let machine_c_time := 3 * y
  (1 / machine_a_time) + (1 / machine_b_time) + (1 / machine_c_time) = 1 / y

/-- Proves that the job completion time satisfies the given equation -/
theorem job_completion_time_solution :
  ∃ y : ℝ, job_completion_time y ∧ y = (-14 + Real.sqrt 296) / 10 := by
  sorry

end job_completion_time_solution_l1752_175225


namespace quadratic_roots_sum_and_product_l1752_175295

theorem quadratic_roots_sum_and_product (α β : ℝ) : 
  α ≠ β →
  α^2 - 5*α - 2 = 0 →
  β^2 - 5*β - 2 = 0 →
  α + β + α*β = 3 := by
sorry

end quadratic_roots_sum_and_product_l1752_175295


namespace intersection_A_B_union_complements_l1752_175241

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x ≤ 3} := by sorry

-- Theorem for (C_U A) ∪ (C_U B)
theorem union_complements : (Set.univ \ A) ∪ (Set.univ \ B) = {x | x ≤ 1 ∨ x > 3} := by sorry

end intersection_A_B_union_complements_l1752_175241


namespace f_min_value_existence_l1752_175215

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then -a * x + 3 else (x - 3) * Real.exp x + Real.exp 2

theorem f_min_value_existence (a : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m) →
  (∃ (a' : ℝ), 0 ≤ a' ∧ a' ≤ Real.sqrt 3 ∧
    ∀ (x : ℝ), f a' x ≥ f a' (2 : ℝ)) ∧
  (∀ (a' : ℝ), a' > Real.sqrt 3 →
    ¬∃ (m : ℝ), ∀ (x : ℝ), f a' x ≥ m) :=
sorry

end f_min_value_existence_l1752_175215


namespace car_speed_problem_l1752_175277

/-- Given two cars starting from the same point and traveling in opposite directions,
    this theorem proves that if one car travels at 60 mph and after 4.66666666667 hours
    they are 490 miles apart, then the speed of the other car must be 45 mph. -/
theorem car_speed_problem (v : ℝ) : 
  (v * (14/3) + 60 * (14/3) = 490) → v = 45 := by
  sorry

end car_speed_problem_l1752_175277


namespace law_school_students_l1752_175247

/-- The number of students in the business school -/
def business_students : ℕ := 500

/-- The number of sibling pairs -/
def sibling_pairs : ℕ := 30

/-- The probability of selecting a sibling pair -/
def sibling_pair_probability : ℚ := 7500000000000001 / 100000000000000000

/-- Theorem stating the number of law students -/
theorem law_school_students (L : ℕ) : 
  (sibling_pairs : ℚ) / (business_students * L) = sibling_pair_probability → 
  L = 8000 := by
  sorry

end law_school_students_l1752_175247


namespace sum_of_solutions_eq_sixteen_l1752_175244

theorem sum_of_solutions_eq_sixteen : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 8)^2 = 16 ∧ (x₂ - 8)^2 = 16 ∧ x₁ + x₂ = 16 := by
  sorry

end sum_of_solutions_eq_sixteen_l1752_175244


namespace largest_k_for_inequality_l1752_175273

theorem largest_k_for_inequality (a b c : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c) 
  (h3 : a * b + b * c + c * a = 0) 
  (h4 : a * b * c = 1) :
  (∀ k : ℝ, (∀ a b c : ℝ, a ≤ b → b ≤ c → a * b + b * c + c * a = 0 → a * b * c = 1 → 
    |a + b| ≥ k * |c|) → k ≤ 4) ∧
  (∀ a b c : ℝ, a ≤ b → b ≤ c → a * b + b * c + c * a = 0 → a * b * c = 1 → 
    |a + b| ≥ 4 * |c|) :=
by sorry

end largest_k_for_inequality_l1752_175273


namespace grid_sum_theorem_l1752_175259

/-- A 3x3 grid represented as a function from (Fin 3 × Fin 3) to ℕ -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- The sum of numbers on the main diagonal of the grid -/
def mainDiagonalSum (g : Grid) : ℕ :=
  g 0 0 + g 1 1 + g 2 2

/-- The sum of numbers on the other diagonal of the grid -/
def otherDiagonalSum (g : Grid) : ℕ :=
  g 0 2 + g 1 1 + g 2 0

/-- The sum of numbers not on either diagonal -/
def nonDiagonalSum (g : Grid) : ℕ :=
  g 0 1 + g 1 0 + g 1 2 + g 2 1 + g 1 1

/-- The theorem statement -/
theorem grid_sum_theorem (g : Grid) :
  (∀ i j, g i j ∈ Finset.range 10) →
  (mainDiagonalSum g = 7) →
  (otherDiagonalSum g = 21) →
  (nonDiagonalSum g = 25) := by
  sorry

end grid_sum_theorem_l1752_175259


namespace set_separation_iff_disjoint_l1752_175253

universe u

theorem set_separation_iff_disjoint {U : Type u} (A B : Set U) :
  (∃ C : Set U, A ⊆ C ∧ B ⊆ Cᶜ) ↔ A ∩ B = ∅ := by
  sorry

end set_separation_iff_disjoint_l1752_175253


namespace boarding_students_count_l1752_175258

theorem boarding_students_count (x : ℕ) (students : ℕ) : 
  (students = 4 * x + 10) →  -- If each dormitory houses 4 people with 10 left over
  (6 * (x - 1) + 1 ≤ students) →  -- Lower bound when housing 6 per dormitory
  (students ≤ 6 * (x - 1) + 5) →  -- Upper bound when housing 6 per dormitory
  (students = 34 ∨ students = 38) :=
by sorry

end boarding_students_count_l1752_175258


namespace at_op_difference_l1752_175284

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - 3 * x + y

-- Theorem statement
theorem at_op_difference : (at_op 5 6) - (at_op 6 5) = 4 := by
  sorry

end at_op_difference_l1752_175284


namespace fraction_sum_inequality_l1752_175264

theorem fraction_sum_inequality (a b c : ℝ) (h : a * b * c = 1) :
  (1 / (2 * a^2 + b^2 + 3)) + (1 / (2 * b^2 + c^2 + 3)) + (1 / (2 * c^2 + a^2 + 3)) ≤ 1/2 := by
  sorry

end fraction_sum_inequality_l1752_175264


namespace circular_plate_arrangement_l1752_175237

def arrangement_count (blue red green yellow : ℕ) : ℕ :=
  sorry

theorem circular_plate_arrangement :
  arrangement_count 6 3 2 1 = 22680 :=
sorry

end circular_plate_arrangement_l1752_175237


namespace bottle_t_cost_l1752_175287

/-- The cost of Bottle T given the conditions of the problem -/
theorem bottle_t_cost :
  let bottle_r_capsules : ℕ := 250
  let bottle_r_cost : ℚ := 625 / 100  -- $6.25 represented as a rational number
  let bottle_t_capsules : ℕ := 100
  let cost_per_capsule_diff : ℚ := 5 / 1000  -- $0.005 represented as a rational number
  let bottle_r_cost_per_capsule : ℚ := bottle_r_cost / bottle_r_capsules
  let bottle_t_cost_per_capsule : ℚ := bottle_r_cost_per_capsule - cost_per_capsule_diff
  bottle_t_cost_per_capsule * bottle_t_capsules = 2 := by
sorry

end bottle_t_cost_l1752_175287


namespace polynomial_division_remainder_l1752_175261

theorem polynomial_division_remainder (k : ℝ) : 
  (∀ x : ℝ, ∃ q : ℝ, 3 * x^3 - k * x^2 + 4 = (3 * x - 1) * q + 5) → k = -8 := by
  sorry

end polynomial_division_remainder_l1752_175261


namespace inequality_and_equality_condition_l1752_175286

theorem inequality_and_equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x + y^2 / x ≥ 2 * y ∧ (x + y^2 / x = 2 * y ↔ x = y) := by
  sorry

end inequality_and_equality_condition_l1752_175286


namespace sin_cos_sum_bound_l1752_175236

theorem sin_cos_sum_bound (θ : Real) (h1 : π/2 < θ) (h2 : θ < π) (h3 : Real.sin (θ/2) < Real.cos (θ/2)) :
  -Real.sqrt 2 < Real.sin (θ/2) + Real.cos (θ/2) ∧ Real.sin (θ/2) + Real.cos (θ/2) < -1 := by
  sorry

end sin_cos_sum_bound_l1752_175236


namespace sofia_shopping_cost_l1752_175232

/-- The cost of all items Sofia buys at the department store -/
theorem sofia_shopping_cost :
  let shirt_cost : ℕ := 7
  let shoes_cost : ℕ := shirt_cost + 3
  let two_shirts_and_shoes_cost : ℕ := 2 * shirt_cost + shoes_cost
  let bag_cost : ℕ := two_shirts_and_shoes_cost / 2
  let total_cost : ℕ := 2 * shirt_cost + shoes_cost + bag_cost
  total_cost = 36 := by
  sorry

end sofia_shopping_cost_l1752_175232


namespace removed_number_is_34_l1752_175229

/-- Given n consecutive natural numbers starting from 1, if one number x is removed
    and the average of the remaining numbers is 152/7, then x = 34. -/
theorem removed_number_is_34 (n : ℕ) (x : ℕ) :
  (x ≥ 1 ∧ x ≤ n) →
  (n * (n + 1) / 2 - x) / (n - 1) = 152 / 7 →
  x = 34 := by
  sorry

end removed_number_is_34_l1752_175229


namespace count_is_thirty_l1752_175296

/-- 
Counts the number of non-negative integers n less than 120 for which 
there exists an integer m divisible by 4 such that the roots of 
x^2 - nx + m = 0 are consecutive non-negative integers.
-/
def count_valid_n : ℕ := by
  sorry

/-- The main theorem stating that the count is equal to 30 -/
theorem count_is_thirty : count_valid_n = 30 := by
  sorry

end count_is_thirty_l1752_175296


namespace essay_writing_rate_l1752_175267

/-- Proves that the writing rate for the first two hours must be 400 words per hour 
    given the conditions of the essay writing problem. -/
theorem essay_writing_rate (total_words : ℕ) (total_hours : ℕ) (later_rate : ℕ) 
    (h1 : total_words = 1200)
    (h2 : total_hours = 4)
    (h3 : later_rate = 200) : 
  ∃ (initial_rate : ℕ), 
    initial_rate * 2 + later_rate * (total_hours - 2) = total_words ∧ 
    initial_rate = 400 := by
  sorry

end essay_writing_rate_l1752_175267


namespace proportional_function_quadrants_l1752_175246

/-- A function passes through the first and third quadrants if for any non-zero x,
    x and f(x) have the same sign. -/
def passes_through_first_and_third_quadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (x > 0 ∧ f x > 0) ∨ (x < 0 ∧ f x < 0)

/-- Theorem: If the graph of y = kx passes through the first and third quadrants,
    then k is positive. -/
theorem proportional_function_quadrants (k : ℝ) :
  passes_through_first_and_third_quadrants (λ x => k * x) → k > 0 := by
  sorry

end proportional_function_quadrants_l1752_175246


namespace shorter_segment_length_l1752_175218

-- Define the triangle ABC
def Triangle (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the angle bisector
def AngleBisector (a b c ae ec : ℝ) := ae / ec = a / b

theorem shorter_segment_length 
  (a b c : ℝ) 
  (h_triangle : Triangle a b c)
  (h_ratio : ∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k)
  (h_ab_length : c = 24)
  (ae ec : ℝ)
  (h_bisector : AngleBisector a b c ae ec)
  (h_sum : ae + ec = c)
  (h_ae_shorter : ae ≤ ec) :
  ae = 72/7 :=
sorry

end shorter_segment_length_l1752_175218


namespace bucket_capacities_solution_l1752_175217

/-- Represents the capacities of three buckets A, B, and C. -/
structure BucketCapacities where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given capacities satisfy the problem conditions. -/
def satisfiesConditions (caps : BucketCapacities) : Prop :=
  caps.a + caps.b + caps.c = 1440 ∧
  caps.a + (1/5) * caps.b = caps.c ∧
  caps.b + (1/3) * caps.a = caps.c

/-- Theorem stating that the unique solution satisfying the conditions is (480, 400, 560). -/
theorem bucket_capacities_solution :
  ∃! (caps : BucketCapacities), satisfiesConditions caps ∧ 
    caps.a = 480 ∧ caps.b = 400 ∧ caps.c = 560 := by
  sorry

end bucket_capacities_solution_l1752_175217


namespace weight_10_moles_CaH2_l1752_175214

/-- The molecular weight of CaH2 in g/mol -/
def molecular_weight_CaH2 : ℝ := 40.08 + 2 * 1.008

/-- The total weight of a given number of moles of CaH2 in grams -/
def total_weight_CaH2 (moles : ℝ) : ℝ := moles * molecular_weight_CaH2

/-- Theorem stating that 10 moles of CaH2 weigh 420.96 grams -/
theorem weight_10_moles_CaH2 : total_weight_CaH2 10 = 420.96 := by sorry

end weight_10_moles_CaH2_l1752_175214


namespace weight_of_aluminum_oxide_l1752_175242

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of aluminum atoms in one molecule of aluminum oxide -/
def Al_count : ℕ := 2

/-- The number of oxygen atoms in one molecule of aluminum oxide -/
def O_count : ℕ := 3

/-- The number of moles of aluminum oxide -/
def moles_Al2O3 : ℝ := 5

/-- The molecular weight of aluminum oxide in g/mol -/
def molecular_weight_Al2O3 : ℝ := Al_count * atomic_weight_Al + O_count * atomic_weight_O

/-- The total weight of the given amount of aluminum oxide in grams -/
def total_weight_Al2O3 : ℝ := moles_Al2O3 * molecular_weight_Al2O3

theorem weight_of_aluminum_oxide :
  total_weight_Al2O3 = 509.8 := by
  sorry

end weight_of_aluminum_oxide_l1752_175242


namespace parabola_tangent_hyperbola_l1752_175223

/-- The value of m for which the parabola y = 2x^2 + 3 is tangent to the hyperbola 4y^2 - mx^2 = 9 -/
def tangent_value : ℝ := 48

/-- The equation of the parabola -/
def parabola (x y : ℝ) : Prop := y = 2 * x^2 + 3

/-- The equation of the hyperbola -/
def hyperbola (m x y : ℝ) : Prop := 4 * y^2 - m * x^2 = 9

/-- The parabola is tangent to the hyperbola when m equals the tangent_value -/
theorem parabola_tangent_hyperbola :
  ∃ (x y : ℝ), parabola x y ∧ hyperbola tangent_value x y ∧
  ∀ (x' y' : ℝ), parabola x' y' ∧ hyperbola tangent_value x' y' → (x', y') = (x, y) :=
sorry

end parabola_tangent_hyperbola_l1752_175223


namespace hexagon_area_ratio_l1752_175213

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- A point on a side of the hexagon -/
def SidePoint (h : RegularHexagon) (i : Fin 6) := ℝ × ℝ

/-- The ratio of areas of two polygons -/
def AreaRatio (p1 p2 : Set (ℝ × ℝ)) : ℚ := sorry

theorem hexagon_area_ratio 
  (ABCDEF : RegularHexagon)
  (P : SidePoint ABCDEF 0) (Q : SidePoint ABCDEF 1) (R : SidePoint ABCDEF 2)
  (S : SidePoint ABCDEF 3) (T : SidePoint ABCDEF 4) (U : SidePoint ABCDEF 5)
  (h_P : P = (2/3 : ℝ) • ABCDEF.vertices 0 + (1/3 : ℝ) • ABCDEF.vertices 1)
  (h_Q : Q = (2/3 : ℝ) • ABCDEF.vertices 1 + (1/3 : ℝ) • ABCDEF.vertices 2)
  (h_R : R = (2/3 : ℝ) • ABCDEF.vertices 2 + (1/3 : ℝ) • ABCDEF.vertices 3)
  (h_S : S = (2/3 : ℝ) • ABCDEF.vertices 3 + (1/3 : ℝ) • ABCDEF.vertices 4)
  (h_T : T = (2/3 : ℝ) • ABCDEF.vertices 4 + (1/3 : ℝ) • ABCDEF.vertices 5)
  (h_U : U = (2/3 : ℝ) • ABCDEF.vertices 5 + (1/3 : ℝ) • ABCDEF.vertices 0) :
  let inner_hexagon := {ABCDEF.vertices 0, R, ABCDEF.vertices 2, T, ABCDEF.vertices 4, P}
  let outer_hexagon := {ABCDEF.vertices i | i : Fin 6}
  AreaRatio inner_hexagon outer_hexagon = 4/9 := by
  sorry

end hexagon_area_ratio_l1752_175213


namespace triangle_angles_theorem_l1752_175222

noncomputable def triangle_angles (a b c : ℝ) : ℝ × ℝ × ℝ := sorry

theorem triangle_angles_theorem :
  let side1 := 3
  let side2 := 3
  let side3 := Real.sqrt 8 - Real.sqrt 3
  let (angle_A, angle_B, angle_C) := triangle_angles side1 side2 side3
  angle_C = Real.arccos ((7 / 18) + (2 * Real.sqrt 6 / 9)) ∧
  angle_A = (π - angle_C) / 2 ∧
  angle_B = (π - angle_C) / 2 :=
sorry

end triangle_angles_theorem_l1752_175222


namespace expected_regions_100_l1752_175288

/-- The number of points on the circle -/
def n : ℕ := 100

/-- The probability that two randomly chosen chords intersect inside the circle -/
def p_intersect : ℚ := 1/3

/-- The expected number of regions bounded by straight lines when n points are picked 
    independently and uniformly at random on a circle, and connected by line segments -/
def expected_regions (n : ℕ) : ℚ :=
  1 + p_intersect * (n.choose 2 - 3 * n)

theorem expected_regions_100 : 
  expected_regions n = 1651 := by sorry

end expected_regions_100_l1752_175288


namespace largest_power_of_two_dividing_difference_of_fourth_powers_l1752_175281

theorem largest_power_of_two_dividing_difference_of_fourth_powers :
  ∃ k : ℕ, (2^k : ℕ) = 128 ∧ (2^k : ℕ) ∣ (17^4 - 15^4) ∧
  ∀ m : ℕ, 2^m ∣ (17^4 - 15^4) → m ≤ k :=
by sorry

end largest_power_of_two_dividing_difference_of_fourth_powers_l1752_175281


namespace sin_range_on_interval_l1752_175254

theorem sin_range_on_interval :
  let f : ℝ → ℝ := λ x ↦ Real.sin x
  let S : Set ℝ := { x | -π/4 ≤ x ∧ x ≤ 3*π/4 }
  f '' S = { y | -Real.sqrt 2 / 2 ≤ y ∧ y ≤ 1 } := by
  sorry

end sin_range_on_interval_l1752_175254


namespace triangle_area_l1752_175272

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  c^2 = (a - b)^2 + 6 →
  C = π/3 →
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_area_l1752_175272


namespace lava_lamp_probability_l1752_175211

/-- The number of red lava lamps -/
def num_red : ℕ := 4

/-- The number of blue lava lamps -/
def num_blue : ℕ := 4

/-- The number of green lava lamps -/
def num_green : ℕ := 4

/-- The total number of lava lamps -/
def total_lamps : ℕ := num_red + num_blue + num_green

/-- The number of lamps that are turned on -/
def num_on : ℕ := 6

/-- The probability of the leftmost lamp being green and off, and the rightmost lamp being blue and on -/
def prob_specific_arrangement : ℚ := 80 / 1313

theorem lava_lamp_probability :
  prob_specific_arrangement = (Nat.choose (total_lamps - 2) num_red * Nat.choose (total_lamps - 2 - num_red) (num_blue - 1) * Nat.choose (total_lamps - 1) (num_on - 1)) /
  (Nat.choose total_lamps num_red * Nat.choose (total_lamps - num_red) num_blue * Nat.choose total_lamps num_on) :=
sorry

end lava_lamp_probability_l1752_175211


namespace four_tire_repair_cost_l1752_175227

/-- The total cost for repairing a given number of tires -/
def total_cost (repair_cost : ℚ) (sales_tax : ℚ) (num_tires : ℕ) : ℚ :=
  (repair_cost + sales_tax) * num_tires

/-- Theorem: The total cost for repairing 4 tires is $30 -/
theorem four_tire_repair_cost :
  total_cost 7 0.5 4 = 30 := by
  sorry

end four_tire_repair_cost_l1752_175227


namespace quadratic_function_determination_l1752_175228

open Real

/-- Given real numbers a, b, c, and functions f and g,
    if the maximum value of g(x) is 2 when -1 ≤ x ≤ 1,
    then f(x) = 2x^2 - 1 -/
theorem quadratic_function_determination
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^2 + b * x + c)
  (h_g : ∀ x, g x = a * x + b)
  (h_max : ∀ x ∈ Set.Icc (-1) 1, g x ≤ 2)
  (h_reaches_max : ∃ x ∈ Set.Icc (-1) 1, g x = 2) :
  ∀ x, f x = 2 * x^2 - 1 := by
  sorry

end quadratic_function_determination_l1752_175228


namespace orientation_count_equals_product_of_combinations_l1752_175271

/-- The number of ways to orient 40 unit segments for zero sum --/
def orientationCount : ℕ := sorry

/-- The total number of unit segments --/
def totalSegments : ℕ := 40

/-- The number of horizontal (or vertical) segments --/
def segmentsPerDirection : ℕ := 20

/-- The number of segments that need to be positive in each direction for zero sum --/
def positiveSegmentsPerDirection : ℕ := 10

theorem orientation_count_equals_product_of_combinations : 
  orientationCount = Nat.choose segmentsPerDirection positiveSegmentsPerDirection * 
                     Nat.choose segmentsPerDirection positiveSegmentsPerDirection := by sorry

end orientation_count_equals_product_of_combinations_l1752_175271


namespace walnut_problem_l1752_175230

theorem walnut_problem (a b c : ℕ) : 
  28 * a + 30 * b + 31 * c = 365 → a + b + c = 12 :=
by
  sorry

end walnut_problem_l1752_175230


namespace percentage_difference_l1752_175243

theorem percentage_difference (total : ℝ) (z_share : ℝ) (x_premium : ℝ) : 
  total = 555 → z_share = 150 → x_premium = 0.25 →
  ∃ y_share : ℝ, 
    y_share = (total - z_share) / (2 + x_premium) ∧
    (y_share - z_share) / z_share = 0.2 := by
  sorry

end percentage_difference_l1752_175243


namespace greatest_integer_with_gcf_five_l1752_175292

def is_valid (n : ℕ) : Prop :=
  n < 150 ∧ Nat.gcd n 30 = 5

theorem greatest_integer_with_gcf_five : 
  (∀ m, is_valid m → m ≤ 145) ∧ is_valid 145 := by sorry

end greatest_integer_with_gcf_five_l1752_175292


namespace negative_fraction_comparison_l1752_175298

theorem negative_fraction_comparison : -3/5 < -1/5 := by
  sorry

end negative_fraction_comparison_l1752_175298


namespace wills_calories_burned_per_minute_l1752_175208

/-- Calories burned per minute while jogging -/
def calories_burned_per_minute (initial_calories net_calories jogging_duration_minutes : ℕ) : ℚ :=
  (initial_calories - net_calories : ℚ) / jogging_duration_minutes

/-- Theorem stating the calories burned per minute for Will's specific case -/
theorem wills_calories_burned_per_minute :
  calories_burned_per_minute 900 600 30 = 10 := by
  sorry

end wills_calories_burned_per_minute_l1752_175208


namespace greatest_odd_integer_below_sqrt_50_l1752_175234

theorem greatest_odd_integer_below_sqrt_50 :
  ∀ x : ℕ, x % 2 = 1 → x^2 < 50 → x ≤ 7 :=
by sorry

end greatest_odd_integer_below_sqrt_50_l1752_175234


namespace eight_power_ten_sum_equals_two_power_y_l1752_175275

theorem eight_power_ten_sum_equals_two_power_y (y : ℕ) :
  8^10 + 8^10 + 8^10 + 8^10 + 8^10 + 8^10 + 8^10 + 8^10 = 2^y → y = 33 := by
  sorry

end eight_power_ten_sum_equals_two_power_y_l1752_175275


namespace cosine_sine_fraction_equals_negative_tangent_l1752_175293

theorem cosine_sine_fraction_equals_negative_tangent (α : ℝ) :
  (Real.cos α - Real.cos (3 * α) + Real.cos (5 * α) - Real.cos (7 * α)) / 
  (Real.sin α + Real.sin (3 * α) + Real.sin (5 * α) + Real.sin (7 * α)) = 
  -Real.tan α := by
  sorry

end cosine_sine_fraction_equals_negative_tangent_l1752_175293


namespace floor_plus_twice_eq_33_l1752_175250

theorem floor_plus_twice_eq_33 :
  ∃! x : ℝ, (⌊x⌋ : ℝ) + 2 * x = 33 :=
by sorry

end floor_plus_twice_eq_33_l1752_175250


namespace abs_sum_greater_than_abs_l1752_175278

theorem abs_sum_greater_than_abs (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : a * b + b * c + a * c = 0) 
  (h4 : a * b * c = 1) : 
  |a + b| > |c| := by
sorry

end abs_sum_greater_than_abs_l1752_175278


namespace equation_solutions_l1752_175251

theorem equation_solutions :
  (∃ x : ℚ, 8 * x = -2 * (x + 5) ∧ x = -1) ∧
  (∃ x : ℚ, (x - 1) / 4 = (5 * x - 7) / 6 + 1 ∧ x = -1 / 7) := by
  sorry

end equation_solutions_l1752_175251


namespace intersection_point_expression_l1752_175212

theorem intersection_point_expression (m n : ℝ) : 
  n = m - 2022 → 
  n = -2022 / m → 
  (2022 / m) + ((m^2 - 2022*m) / n) = 2022 := by
sorry

end intersection_point_expression_l1752_175212


namespace boys_to_girls_ratio_l1752_175255

/-- Proves that the ratio of boys to girls in a school with 90 students, of which 60 are girls, is 1:2 -/
theorem boys_to_girls_ratio (total_students : Nat) (girls : Nat) (h1 : total_students = 90) (h2 : girls = 60) :
  (total_students - girls) / girls = 1 / 2 := by
  sorry

end boys_to_girls_ratio_l1752_175255


namespace log_equation_solution_l1752_175200

-- Define the logarithm function for base 2
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem log_equation_solution :
  ∃ x : ℝ, log2 (x + 3) + 2 * log2 5 = 4 ∧ x = -59 / 25 := by
  sorry

end log_equation_solution_l1752_175200


namespace smallest_number_with_same_prime_factors_l1752_175270

def is_prime_factor (p n : ℕ) : Prop :=
  Nat.Prime p ∧ n % p = 0

def has_all_prime_factors (m n : ℕ) : Prop :=
  ∀ p, is_prime_factor p n → is_prime_factor p m

theorem smallest_number_with_same_prime_factors (n : ℕ) (hn : n = 36) :
  ∃ m : ℕ, m = 6 ∧
    has_all_prime_factors m n ∧
    ∀ k : ℕ, k < m → ¬(has_all_prime_factors k n) :=
by sorry

end smallest_number_with_same_prime_factors_l1752_175270


namespace cubic_factorization_l1752_175282

theorem cubic_factorization (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end cubic_factorization_l1752_175282


namespace bella_ella_meeting_l1752_175202

/-- The distance between Bella's and Ella's houses in feet -/
def distance : ℕ := 15840

/-- The length of Bella's step in feet -/
def step_length : ℕ := 3

/-- The ratio of Ella's speed to Bella's speed -/
def speed_ratio : ℕ := 5

/-- The number of steps Bella takes before meeting Ella -/
def steps_taken : ℕ := 880

theorem bella_ella_meeting :
  distance = 15840 ∧
  step_length = 3 ∧
  speed_ratio = 5 →
  steps_taken = 880 :=
by sorry

end bella_ella_meeting_l1752_175202


namespace divisor_power_difference_l1752_175238

theorem divisor_power_difference (k : ℕ) : 
  (15 ^ k ∣ 823435) → 5 ^ k - k ^ 5 = 4 := by
  sorry

end divisor_power_difference_l1752_175238


namespace photo_arrangement_l1752_175240

theorem photo_arrangement (n_male : ℕ) (n_female : ℕ) : 
  n_male = 4 → n_female = 2 → (
    (3 : ℕ) *           -- ways to place "甲" in middle positions
    (4 : ℕ).factorial * -- ways to arrange remaining units
    (2 : ℕ).factorial   -- ways to arrange female students within their unit
  ) = 144 := by
  sorry

end photo_arrangement_l1752_175240


namespace smallest_tangent_circle_slope_l1752_175220

/-- Circle ω₁ -/
def ω₁ (x y : ℝ) : Prop := x^2 + y^2 + 12*x - 20*y - 100 = 0

/-- Circle ω₂ -/
def ω₂ (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 20*y + 196 = 0

/-- A circle is externally tangent to ω₂ -/
def externally_tangent_ω₂ (x y r : ℝ) : Prop :=
  r + 8 = Real.sqrt ((x - 6)^2 + (y - 10)^2)

/-- A circle is internally tangent to ω₁ -/
def internally_tangent_ω₁ (x y r : ℝ) : Prop :=
  16 - r = Real.sqrt ((x + 6)^2 + (y - 10)^2)

/-- The main theorem -/
theorem smallest_tangent_circle_slope :
  ∃ (m : ℝ), m > 0 ∧ m^2 = 160/99 ∧
  (∀ (a : ℝ), a > 0 → a < m →
    ¬∃ (x y r : ℝ), y = a*x ∧
      externally_tangent_ω₂ x y r ∧
      internally_tangent_ω₁ x y r) ∧
  (∃ (x y r : ℝ), y = m*x ∧
    externally_tangent_ω₂ x y r ∧
    internally_tangent_ω₁ x y r) :=
sorry

end smallest_tangent_circle_slope_l1752_175220


namespace solve_equation_l1752_175268

theorem solve_equation (X : ℝ) : 
  (X^3).sqrt = 81 * (81^(1/12)) → X = 3^(14/9) := by
  sorry

end solve_equation_l1752_175268


namespace unique_element_condition_l1752_175279

def A (a : ℝ) : Set ℝ := {x | a * x^2 - 2 * x - 1 = 0}

theorem unique_element_condition (a : ℝ) : (∃! x, x ∈ A a) ↔ (a = 0 ∨ a = -1) := by
  sorry

end unique_element_condition_l1752_175279


namespace certain_number_proof_l1752_175207

theorem certain_number_proof : ∃ n : ℕ, (73 * n) % 8 = 7 ∧ n = 7 := by
  sorry

end certain_number_proof_l1752_175207


namespace divisibility_of_sum_l1752_175209

theorem divisibility_of_sum : 
  let x : ℕ := 50 + 100 + 140 + 180 + 320 + 400 + 5000
  (x % 5 = 0 ∧ x % 10 = 0) ∧ (x % 20 ≠ 0 ∧ x % 40 ≠ 0) :=
by sorry

end divisibility_of_sum_l1752_175209


namespace investment_sum_l1752_175204

/-- Given a sum of money invested for 2 years, if increasing the interest rate by 3% 
    results in 300 more rupees of interest, then the original sum invested must be 5000 rupees. -/
theorem investment_sum (P : ℝ) (R : ℝ) : 
  (P * (R + 3) * 2) / 100 = (P * R * 2) / 100 + 300 → P = 5000 := by
sorry

end investment_sum_l1752_175204


namespace cylinder_side_surface_diagonal_l1752_175224

/-- Given a cylinder with height 8 feet and base perimeter 6 feet,
    prove that the diagonal of the rectangular plate forming its side surface is 10 feet. -/
theorem cylinder_side_surface_diagonal (h : ℝ) (p : ℝ) (d : ℝ) :
  h = 8 →
  p = 6 →
  d = (h^2 + p^2)^(1/2) →
  d = 10 :=
by sorry

end cylinder_side_surface_diagonal_l1752_175224


namespace roots_greater_than_three_l1752_175206

/-- For a quadratic equation x^2 - 6ax + (2 - 2a + 9a^2) = 0, both roots are greater than 3 
    if and only if a > 11/9 -/
theorem roots_greater_than_three (a : ℝ) : 
  (∀ x : ℝ, x^2 - 6*a*x + (2 - 2*a + 9*a^2) = 0 → x > 3) ↔ a > 11/9 := by
  sorry

end roots_greater_than_three_l1752_175206


namespace examination_statements_l1752_175226

/-- Represents a statistical population -/
structure Population where
  size : ℕ

/-- Represents a sample from a population -/
structure Sample (pop : Population) where
  size : ℕ
  h_size_le : size ≤ pop.size

/-- The given examination scenario -/
def examination_scenario : Prop :=
  ∃ (pop : Population) (sample : Sample pop),
    pop.size = 70000 ∧
    sample.size = 1000 ∧
    (sample.size = 1000 → sample.size = 1000) ∧
    (pop.size = 70000 → pop.size = 70000)

/-- The statements to be proved -/
theorem examination_statements (h : examination_scenario) :
  ∃ (pop : Population) (sample : Sample pop),
    pop.size = 70000 ∧
    sample.size = 1000 ∧
    (Sample pop → True) ∧  -- Statement 1
    (pop.size = 70000 → True) ∧  -- Statement 3
    (sample.size = 1000 → True)  -- Statement 4
    := by sorry

end examination_statements_l1752_175226


namespace min_value_fraction_l1752_175265

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y - 3 = 0) :
  (x + 2 * y) / (x * y) ≥ 3 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ - 3 = 0 ∧ (x₀ + 2 * y₀) / (x₀ * y₀) = 3 :=
by sorry

end min_value_fraction_l1752_175265


namespace correct_practice_times_l1752_175256

/-- Represents the practice schedule and time spent on instruments in a month -/
structure PracticeSchedule where
  piano_daily_minutes : ℕ
  violin_daily_minutes : ℕ
  flute_daily_minutes : ℕ
  violin_days_per_week : ℕ
  flute_days_per_week : ℕ
  weeks_with_6_days : ℕ
  weeks_with_7_days : ℕ

/-- Calculates the total practice time for each instrument in the given month -/
def calculate_practice_time (schedule : PracticeSchedule) :
  ℕ × ℕ × ℕ :=
  let total_days := schedule.weeks_with_6_days * 6 + schedule.weeks_with_7_days * 7
  let violin_total_days := schedule.violin_days_per_week * (schedule.weeks_with_6_days + schedule.weeks_with_7_days)
  let flute_total_days := schedule.flute_days_per_week * (schedule.weeks_with_6_days + schedule.weeks_with_7_days)
  (schedule.piano_daily_minutes * total_days,
   schedule.violin_daily_minutes * violin_total_days,
   schedule.flute_daily_minutes * flute_total_days)

/-- Theorem stating the correct practice times for each instrument -/
theorem correct_practice_times (schedule : PracticeSchedule)
  (h1 : schedule.piano_daily_minutes = 25)
  (h2 : schedule.violin_daily_minutes = 3 * schedule.piano_daily_minutes)
  (h3 : schedule.flute_daily_minutes = schedule.violin_daily_minutes / 2)
  (h4 : schedule.violin_days_per_week = 5)
  (h5 : schedule.flute_days_per_week = 4)
  (h6 : schedule.weeks_with_6_days = 2)
  (h7 : schedule.weeks_with_7_days = 2) :
  calculate_practice_time schedule = (650, 1500, 600) :=
sorry


end correct_practice_times_l1752_175256


namespace unique_solution_quartic_l1752_175263

theorem unique_solution_quartic (n : ℤ) : 
  (∃! x : ℝ, 4 * x^4 + n * x^2 + 4 = 0) ↔ (n = 8 ∨ n = -8) :=
sorry

end unique_solution_quartic_l1752_175263


namespace gcd_12345_54321_l1752_175290

theorem gcd_12345_54321 : Nat.gcd 12345 54321 = 1 := by
  sorry

end gcd_12345_54321_l1752_175290


namespace unique_remainder_mod_nine_l1752_175266

theorem unique_remainder_mod_nine : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1111 ≡ n [ZMOD 9] := by
  sorry

end unique_remainder_mod_nine_l1752_175266


namespace solve_for_x_l1752_175294

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- State the theorem
theorem solve_for_x (x : ℝ) : euro x (euro 4 5) = 480 → x = 6 := by
  sorry

end solve_for_x_l1752_175294


namespace fourth_month_sales_l1752_175231

def sales_1 : ℕ := 5400
def sales_2 : ℕ := 9000
def sales_3 : ℕ := 6300
def sales_5 : ℕ := 4500
def sales_6 : ℕ := 1200
def average_sale : ℕ := 5600
def num_months : ℕ := 6

theorem fourth_month_sales :
  ∃ (sales_4 : ℕ), 
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = average_sale ∧
    sales_4 = 8200 := by
  sorry

end fourth_month_sales_l1752_175231


namespace candy_shop_ratio_l1752_175276

/-- Proves that the ratio of cherry sours to lemon sours is 4:5 given the conditions of the candy shop problem -/
theorem candy_shop_ratio :
  ∀ (total cherry orange lemon : ℕ),
  total = 96 →
  cherry = 32 →
  orange = total / 4 →
  total = cherry + orange + lemon →
  (cherry : ℚ) / lemon = 4 / 5 := by
sorry

end candy_shop_ratio_l1752_175276


namespace pure_imaginary_power_l1752_175252

theorem pure_imaginary_power (a : ℝ) (z : ℂ) : 
  z = a + (a + 1) * Complex.I → (z.im ≠ 0 ∧ z.re = 0) → z^2010 = -1 := by
  sorry

end pure_imaginary_power_l1752_175252


namespace locus_of_centers_l1752_175203

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C₂ with equation (x - 3)² + y² = 9 -/
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

/-- A circle is externally tangent to C₁ if the distance between their centers
    equals the sum of their radii -/
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C₂ if the distance between their centers
    equals the difference of their radii -/
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 3)^2 + b^2 = (3 - r)^2

/-- The locus of centers (a,b) of circles externally tangent to C₁ and internally tangent to C₂
    satisfies the equation 28a² + 64b² - 84a - 49 = 0 -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) → 
  28 * a^2 + 64 * b^2 - 84 * a - 49 = 0 := by
  sorry

end locus_of_centers_l1752_175203
