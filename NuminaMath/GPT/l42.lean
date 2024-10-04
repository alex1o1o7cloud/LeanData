import Mathlib

namespace a_100_value_l42_42224

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0     => 0    -- using 0-index for convenience
| (n+1) => a n + 4

-- Prove the value of the 100th term in the sequence
theorem a_100_value : a 100 = 397 := 
by {
  -- proof would go here
  sorry
}

end a_100_value_l42_42224


namespace greatest_divisible_by_13_l42_42304

theorem greatest_divisible_by_13 (A B C : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (hC : 0 ≤ C ∧ C ≤ 9)
  (distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) : (10000 * A + 1000 * B + 100 * C + 10 * B + A = 96769) 
  ↔ (10000 * A + 1000 * B + 100 * C + 10 * B + A) % 13 = 0 :=
sorry

end greatest_divisible_by_13_l42_42304


namespace encode_message_correct_l42_42527

/-- Encoding mappings in the old system -/
def old_encoding : char → string
| 'A' := "11"
| 'B' := "011"
| 'C' := "0"
| _ := ""

/-- Encoding mappings in the new system -/
def new_encoding : char → string
| 'A' := "21"
| 'B' := "122"
| 'C' := "1"
| _ := ""

/-- Decoding the old encoded message to a string of characters -/
def decode_old_message : string → list char
| "011011010011" := ['A', 'B', 'C', 'B', 'A']
| _ := []

/-- Encode a list of characters using the new encoding -/
def encode_new_message : list char → string
| ['A', 'B', 'C', 'B', 'A'] := "211221121"
| _ := ""

/-- Proving that decoding the old message and re-encoding it gives the correct new encoded message -/
theorem encode_message_correct :
  encode_new_message (decode_old_message "011011010011") = "211221121" :=
by sorry

end encode_message_correct_l42_42527


namespace part_I_part_II_l42_42743

variable (a b c : ℝ)

theorem part_I (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : ∀ x : ℝ, |x + a| + |x - b| + c ≥ 4) : a + b + c = 4 :=
sorry

theorem part_II (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 4) : (1/4) * a^2 + (1/9) * b^2 + c^2 ≥ 8/7 :=
sorry

end part_I_part_II_l42_42743


namespace jan_discount_percentage_l42_42624

theorem jan_discount_percentage :
  ∃ percent_discount : ℝ,
    ∀ (roses_bought dozen : ℕ) (rose_cost amount_paid : ℝ),
      roses_bought = 5 * dozen → dozen = 12 →
      rose_cost = 6 →
      amount_paid = 288 →
      (roses_bought * rose_cost - amount_paid) / (roses_bought * rose_cost) * 100 = percent_discount →
      percent_discount = 20 :=
by
  sorry

end jan_discount_percentage_l42_42624


namespace sum_D_E_correct_sum_of_all_possible_values_of_D_E_l42_42610

theorem sum_D_E_correct :
  ∀ (D E : ℕ), (D < 10) → (E < 10) →
  (∃ k : ℕ, (10^8 * D + 4650000 + 1000 * E + 32) = 7 * k) →
  D + E = 1 ∨ D + E = 8 ∨ D + E = 15 :=
by sorry

theorem sum_of_all_possible_values_of_D_E :
  (1 + 8 + 15) = 24 :=
by norm_num

end sum_D_E_correct_sum_of_all_possible_values_of_D_E_l42_42610


namespace part1_part2_l42_42426

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l42_42426


namespace complex_point_quadrant_l42_42781

open Complex

theorem complex_point_quadrant :
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  (0 < z.re) ∧ (0 < z.im) :=
by 
  let z := (1 + 3 * Complex.i) * (3 - Complex.i)
  split
  -- prove the real part is positive
  sorry
  -- prove the imaginary part is positive
  sorry

end complex_point_quadrant_l42_42781


namespace division_of_fractions_l42_42533

theorem division_of_fractions :
  (5 / 6 : ℚ) / (11 / 12) = 10 / 11 := by
  sorry

end division_of_fractions_l42_42533


namespace problem_part1_problem_part2_l42_42751

theorem problem_part1 (k m : ℝ) :
  (∀ x : ℝ, (|k|-3) * x^2 - (k-3) * x + 2*m + 1 = 0 → (|k|-3 = 0 ∧ k ≠ 3)) →
  k = -3 :=
sorry

theorem problem_part2 (k m : ℝ) :
  ((∃ x1 x2 : ℝ, 
     ((|k|-3) * x1^2 - (k-3) * x1 + 2*m + 1 = 0) ∧
     (3 * x2 - 2 = 4 - 5 * x2 + 2 * x2) ∧
     x1 = -x2) →
  (∀ x : ℝ, (|k|-3) * x^2 - (k-3) * x + 2*m + 1 = 0 → (|k|-3 = 0 ∧ x = -1)) →
  (k = -3 ∧ m = 5/2)) :=
sorry

end problem_part1_problem_part2_l42_42751


namespace product_of_square_roots_l42_42992
-- Importing the necessary Lean library

-- Declare the mathematical problem in Lean 4
theorem product_of_square_roots (x : ℝ) (hx : 0 ≤ x) :
  Real.sqrt (40 * x) * Real.sqrt (5 * x) * Real.sqrt (18 * x) = 60 * x * Real.sqrt (3 * x) :=
by
  sorry

end product_of_square_roots_l42_42992


namespace min_books_borrowed_l42_42213

theorem min_books_borrowed 
    (h1 : 12 * 1 = 12) 
    (h2 : 10 * 2 = 20) 
    (h3 : 2 = 2) 
    (h4 : 32 = 32) 
    (h5 : (32 * 2 = 64))
    (h6 : ∀ x, x ≤ 11) :
    ∃ (x : ℕ), (8 * x = 32) ∧ x ≤ 11 := 
  sorry

end min_books_borrowed_l42_42213


namespace ancient_chinese_wine_problem_l42_42221

theorem ancient_chinese_wine_problem:
  ∃ x: ℝ, 10 * x + 3 * (5 - x) = 30 :=
by
  sorry

end ancient_chinese_wine_problem_l42_42221


namespace expression_simplifies_to_10_over_7_l42_42545

def complex_expression : ℚ :=
  1 + 1 / (2 + 1 / (1 + 2))

theorem expression_simplifies_to_10_over_7 : 
  complex_expression = 10 / 7 :=
by
  sorry

end expression_simplifies_to_10_over_7_l42_42545


namespace gross_profit_percentage_l42_42737

theorem gross_profit_percentage :
  ∀ (selling_price wholesale_cost : ℝ),
  selling_price = 28 →
  wholesale_cost = 24.14 →
  (selling_price - wholesale_cost) / wholesale_cost * 100 = 15.99 :=
by
  intros selling_price wholesale_cost h1 h2
  rw [h1, h2]
  norm_num
  sorry

end gross_profit_percentage_l42_42737


namespace find_gamma_delta_l42_42822

theorem find_gamma_delta (γ δ : ℝ) (h : ∀ x : ℝ, (x - γ) / (x + δ) = (x^2 - 90 * x + 1980) / (x^2 + 60 * x - 3240)) : 
  γ + δ = 140 :=
sorry

end find_gamma_delta_l42_42822


namespace michael_bunnies_l42_42484

theorem michael_bunnies (total_pets : ℕ) (percent_dogs percent_cats : ℕ) (h1 : total_pets = 36) (h2 : percent_dogs = 25) (h3 : percent_cats = 50) : total_pets * (100 - percent_dogs - percent_cats) / 100 = 9 :=
by
  -- 25% of 36 is 9
  rw [h1, h2, h3]
  norm_num
  sorry

end michael_bunnies_l42_42484


namespace increased_sales_type_B_l42_42835

-- Definitions for sales equations
def store_A_sales (x y : ℝ) : Prop :=
  60 * x + 15 * y = 3600

def store_B_sales (x y : ℝ) : Prop :=
  40 * x + 60 * y = 4400

-- Definition for the price of clothing items
def price_A (x : ℝ) : Prop :=
  x = 50

def price_B (y : ℝ) : Prop :=
  y = 40

-- Definition for the increased sales in May for type A
def may_sales_A (x : ℝ) : Prop :=
  100 * x * 1.2 = 6000

-- Definition to prove percentage increase for type B sales in May
noncomputable def percentage_increase_B (x y : ℝ) : ℝ :=
  ((4500 - (100 * y * 0.4)) / (100 * y * 0.4)) * 100

theorem increased_sales_type_B (x y : ℝ)
  (h1 : store_A_sales x y)
  (h2 : store_B_sales x y)
  (hA : price_A x)
  (hB : price_B y)
  (hMayA : may_sales_A x) :
  percentage_increase_B x y = 50 :=
sorry

end increased_sales_type_B_l42_42835


namespace each_person_gets_4_roses_l42_42080

def ricky_roses_total : Nat := 40
def roses_stolen : Nat := 4
def people : Nat := 9
def remaining_roses : Nat := ricky_roses_total - roses_stolen
def roses_per_person : Nat := remaining_roses / people

theorem each_person_gets_4_roses : roses_per_person = 4 := by
  sorry

end each_person_gets_4_roses_l42_42080


namespace property_P_difference_l42_42877

noncomputable def f (n : ℕ) : ℕ :=
  if n % 2 = 0 then 
    6 * 2^(n / 2) - n - 5 
  else 
    4 * 2^((n + 1) / 2) - n - 5

theorem property_P_difference : f 9 - f 8 = 31 := by
  sorry

end property_P_difference_l42_42877


namespace power_sum_eq_nine_l42_42232

theorem power_sum_eq_nine {m n p q : ℕ} (h : ∀ x > 0, (x + 1)^m / x^n - 1 = (x + 1)^p / x^q) :
  (m^2 + 2 * n + p)^(2 * q) = 9 :=
sorry

end power_sum_eq_nine_l42_42232


namespace part1_part2_l42_42430

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l42_42430


namespace minimum_value_of_f_l42_42605

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f x ≤ f y) ∧ f x = -1 :=
by
  sorry

end minimum_value_of_f_l42_42605


namespace feeding_ways_correct_l42_42134

def total_feeding_ways : Nat :=
  (5 * 6 * (5 * 4 * 3 * 2 * 1)^2)

theorem feeding_ways_correct :
  total_feeding_ways = 432000 :=
by
  -- Proof is omitted here
  sorry

end feeding_ways_correct_l42_42134


namespace simplify_expression_l42_42248

theorem simplify_expression (m : ℤ) : 
  ((7 * m + 3) - 3 * m * 2) * 4 + (5 - 2 / 4) * (8 * m - 12) = 40 * m - 42 :=
by 
  sorry

end simplify_expression_l42_42248


namespace find_f3_l42_42338

theorem find_f3 {f : ℝ → ℝ} (h : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 5) : f 3 = -2 :=
by
  sorry

end find_f3_l42_42338


namespace shaded_area_l42_42105

theorem shaded_area (d : ℝ) (k : ℝ) (π : ℝ) (r : ℝ)
  (h_diameter : d = 6) 
  (h_radius_large : k = 5)
  (h_small_radius: r = d / 2) :
  ((π * (k * r)^2) - (π * r^2)) = 216 * π :=
by
  sorry

end shaded_area_l42_42105


namespace max_x_minus_y_l42_42187

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  (∃ z : ℝ, z = x - y ∧ (∀ w : ℝ, w = x - y → w ≤ 1 + 3 * real.sqrt 2) ∧ (∃ u : ℝ, u = x - y ∧ u = 1 + 3 * real.sqrt 2)) :=
sorry

end max_x_minus_y_l42_42187


namespace certain_value_of_101n_squared_l42_42052

theorem certain_value_of_101n_squared 
  (n : ℤ) 
  (h : ∀ (n : ℤ), 101 * n^2 ≤ 4979 → n ≤ 7) : 
  4979 = 101 * 7^2 :=
by {
  /- proof goes here -/
  sorry
}

end certain_value_of_101n_squared_l42_42052


namespace angle_sum_at_F_l42_42773

theorem angle_sum_at_F (x y z w v : ℝ) (h : x + y + z + w + v = 360) : 
  x = 360 - y - z - w - v := by
  sorry

end angle_sum_at_F_l42_42773


namespace gcd_840_1764_l42_42676

theorem gcd_840_1764 : Int.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l42_42676


namespace instantaneous_velocity_at_1_l42_42303

noncomputable def particle_displacement (t : ℝ) : ℝ := t + Real.log t

theorem instantaneous_velocity_at_1 : 
  let v := fun t => deriv (particle_displacement) t
  v 1 = 2 :=
by
  sorry

end instantaneous_velocity_at_1_l42_42303


namespace solve_inequality_I_solve_inequality_II_l42_42604

def f (x : ℝ) : ℝ := |x - 1| - |2 * x + 3|

theorem solve_inequality_I (x : ℝ) : f x > 2 ↔ -2 < x ∧ x < -4 / 3 :=
by sorry

theorem solve_inequality_II (a : ℝ) : ∀ x, f x ≤ (3 / 2) * a^2 - a ↔ a ≥ 5 / 3 :=
by sorry

end solve_inequality_I_solve_inequality_II_l42_42604


namespace father_l42_42126

-- Define the variables
variables (F S : ℕ)

-- Define the conditions
def condition1 : Prop := F = 4 * S
def condition2 : Prop := F + 20 = 2 * (S + 20)
def condition3 : Prop := S = 10

-- Statement of the problem
theorem father's_age (h1 : condition1 F S) (h2 : condition2 F S) (h3 : condition3 S) : F = 40 :=
by sorry

end father_l42_42126


namespace maximum_value_of_x_minus_y_l42_42170

noncomputable def max_value_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : ℝ :=
1 + 3 * Real.sqrt 2

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
    x - y ≤ max_value_x_minus_y x y h := sorry

end maximum_value_of_x_minus_y_l42_42170


namespace area_of_complex_polygon_l42_42275

-- Defining the problem
def area_of_polygon (side1 side2 side3 : ℝ) (rot1 rot2 : ℝ) : ℝ :=
  -- This is a placeholder definition.
  -- In a complete proof, here we would calculate the area based on the input conditions.
  sorry

-- Main theorem statement
theorem area_of_complex_polygon :
  area_of_polygon 4 5 6 (π / 4) (-π / 6) = 72 :=
by sorry

end area_of_complex_polygon_l42_42275


namespace part1_l42_42433

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

theorem part1 {x : ℝ} : f x ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) := by
  sorry

end part1_l42_42433


namespace volleyball_final_probability_l42_42503

/-- The championship finals of the Chinese Volleyball Super League adopts a seven-game four-win system,
which means that if one team wins four games first, that team will be the overall champion, and the competition will end.
Each team has a 1/2 probability of winning each game. 
The ticket revenue for the first game is 5 million yuan, and each subsequent game's revenue increases by 100,000 yuan compared to the previous game.
This theorem states that the probability that the total ticket revenue for the finals is exactly 45 million yuan is 5/16. -/
theorem volleyball_final_probability :
  let p_win : ℚ := 1 / 2,
      revenue_fn : ℕ → ℕ := λ n, 5 * n + (n * (n - 1)) / 2,
      total_revenue : ℚ := 45,
      n : ℕ := 6 in
  revenue_fn n = 45 → 
  let p_total_revenue : ℚ := (nat.choose 5 3 * p_win^5 + nat.choose 5 2 * p_win^5) in
  p_total_revenue = 5 / 16 :=
by
  sorry

end volleyball_final_probability_l42_42503


namespace max_value_x_minus_y_l42_42177

theorem max_value_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_value_x_minus_y_l42_42177


namespace triangle_area_ordering_l42_42860

variable (m n p : ℚ)

theorem triangle_area_ordering (hm : m = 15 / 2) (hn : n = 13 / 2) (hp : p = 7) : n < p ∧ p < m := by
  sorry

end triangle_area_ordering_l42_42860


namespace rock_paper_scissors_l42_42674

open Nat

-- Definitions based on problem conditions
def personA_movement (x y z : ℕ) : ℤ :=
  3 * (x : ℤ) - 2 * (y : ℤ) + (z : ℤ)

def personB_movement (x y z : ℕ) : ℤ :=
  3 * (y : ℤ) - 2 * (x : ℤ) + (z : ℤ)

def total_rounds (x y z : ℕ) : ℕ :=
  x + y + z

-- Problem statement
theorem rock_paper_scissors (x y z : ℕ) 
  (h1 : total_rounds x y z = 15)
  (h2 : personA_movement x y z = 17)
  (h3 : personB_movement x y z = 2) : x = 7 :=
by
  sorry

end rock_paper_scissors_l42_42674


namespace girls_ran_miles_l42_42116

def boys_laps : ℕ := 34
def extra_laps : ℕ := 20
def lap_distance : ℚ := 1 / 6
def girls_laps : ℕ := boys_laps + extra_laps

theorem girls_ran_miles : girls_laps * lap_distance = 9 := 
by 
  sorry

end girls_ran_miles_l42_42116


namespace boys_girls_students_l42_42670

theorem boys_girls_students (total_students : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ)
  (h1 : total_students = 100)
  (h2 : ratio_boys = 3)
  (h3 : ratio_girls = 2) :
  3 * (total_students / (ratio_boys + ratio_girls)) - 2 * (total_students / (ratio_boys + ratio_girls)) = 20 :=
by
  sorry

end boys_girls_students_l42_42670


namespace find_the_number_l42_42114

theorem find_the_number :
  ∃ x : ℕ, (x + 720) / 125 = 7392 / 462 ∧ x = 1280 := 
  sorry

end find_the_number_l42_42114


namespace unused_combinations_eq_40_l42_42312

-- Defining the basic parameters
def num_resources : ℕ := 6
def total_combinations : ℕ := 2 ^ num_resources
def used_combinations : ℕ := 23

-- Calculating the number of unused combinations
theorem unused_combinations_eq_40 : total_combinations - 1 - used_combinations = 40 := by
  sorry

end unused_combinations_eq_40_l42_42312


namespace Lindsay_has_26_more_black_brown_dolls_than_blonde_l42_42926

def blonde_dolls : Nat := 4
def brown_dolls : Nat := 4 * blonde_dolls
def black_dolls : Nat := brown_dolls - 2
def total_black_brown_dolls : Nat := black_dolls + brown_dolls
def extra_black_brown_dolls (blonde_dolls black_dolls brown_dolls : Nat) : Nat :=
  total_black_brown_dolls - blonde_dolls

theorem Lindsay_has_26_more_black_brown_dolls_than_blonde :
  extra_black_brown_dolls blonde_dolls black_dolls brown_dolls = 26 := by
  sorry

end Lindsay_has_26_more_black_brown_dolls_than_blonde_l42_42926


namespace halfway_fraction_l42_42678

theorem halfway_fraction (a b : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 7) :
  ((a + b) / 2) = 41 / 56 :=
by
  rw [h_a, h_b]
  sorry

end halfway_fraction_l42_42678


namespace find_angle_C_find_sin_A_plus_sin_B_l42_42226

open Real

namespace TriangleProblem

variables (a b c : ℝ) (A B C : ℝ)

def sides_opposite_angles (a b c : ℝ) (A B C : ℝ) : Prop :=
  c^2 = a^2 + b^2 + a * b

def given_c (c : ℝ) : Prop :=
  c = 4 * sqrt 7

def perimeter (a b c : ℝ) : Prop :=
  a + b + c = 12 + 4 * sqrt 7

theorem find_angle_C (a b c A B C : ℝ)
  (h1 : sides_opposite_angles a b c A B C) : 
  C = 2 * pi / 3 :=
sorry

theorem find_sin_A_plus_sin_B (a b c A B C : ℝ)
  (h1 : sides_opposite_angles a b c A B C)
  (h2 : given_c c)
  (h3 : perimeter a b c) : 
  sin A + sin B = 3 * sqrt 21 / 28 :=
sorry

end TriangleProblem

end find_angle_C_find_sin_A_plus_sin_B_l42_42226


namespace tan_difference_l42_42590

theorem tan_difference (x y : ℝ) (hx : Real.tan x = 3) (hy : Real.tan y = 2) :
  Real.tan (x - y) = 1 / 7 := 
  sorry

end tan_difference_l42_42590


namespace cos_theta_seven_l42_42030

theorem cos_theta_seven {θ : ℝ} (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -8383 / 98304 :=
by
  sorry

end cos_theta_seven_l42_42030


namespace halfway_fraction_l42_42685

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/7) : (a + b) / 2 = 41/56 :=
by
  sorry

end halfway_fraction_l42_42685


namespace line_eqn_with_given_conditions_l42_42088

theorem line_eqn_with_given_conditions : 
  ∃(m c : ℝ), (∀ x y : ℝ, y = m*x + c → x + y - 3 = 0) ↔ 
  ∀ x y, x + y = 3 :=
sorry

end line_eqn_with_given_conditions_l42_42088


namespace remainder_98_mul_102_div_11_l42_42962

theorem remainder_98_mul_102_div_11 : (98 * 102) % 11 = 7 := by
  sorry

end remainder_98_mul_102_div_11_l42_42962


namespace train_speed_l42_42845

theorem train_speed
  (distance_meters : ℝ := 400)
  (time_seconds : ℝ := 12)
  (distance_kilometers : ℝ := distance_meters / 1000)
  (time_hours : ℝ := time_seconds / 3600) :
  distance_kilometers / time_hours = 120 := by
  sorry

end train_speed_l42_42845


namespace product_of_four_integers_l42_42589

theorem product_of_four_integers:
  ∃ (A B C D : ℚ) (x : ℚ), 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧
  A + B + C + D = 40 ∧
  A - 3 = x ∧ B + 3 = x ∧ C / 2 = x ∧ D * 2 = x ∧
  A * B * C * D = (9089600 / 6561) := by
  sorry

end product_of_four_integers_l42_42589


namespace cos_seven_theta_l42_42043

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -953 / 1024 :=
by sorry

end cos_seven_theta_l42_42043


namespace radius_of_circle_passing_through_ABC_incircle_center_l42_42791

variable {a : ℝ} {α : ℝ} (hα : 0 < α) (hα_lt_π : α < real.pi)

theorem radius_of_circle_passing_through_ABC_incircle_center
  {AB : ℝ} {α : ℝ} (hAB : AB = a) (hC : ∠ ABC = α) :
  ∃ R : ℝ, R = a / (2 * real.cos (α / 2)) :=
sorry

end radius_of_circle_passing_through_ABC_incircle_center_l42_42791


namespace pentomino_reflectional_count_l42_42028

def is_reflectional (p : Pentomino) : Prop := sorry -- Define reflectional symmetry property
def is_rotational (p : Pentomino) : Prop := sorry -- Define rotational symmetry property

theorem pentomino_reflectional_count :
  ∀ (P : Finset Pentomino),
  P.card = 15 →
  (∃ (R : Finset Pentomino), R.card = 2 ∧ (∀ p ∈ R, is_rotational p ∧ ¬ is_reflectional p)) →
  (∃ (S : Finset Pentomino), S.card = 7 ∧ (∀ p ∈ S, is_reflectional p)) :=
by
  sorry -- Proof not required as per instructions

end pentomino_reflectional_count_l42_42028


namespace smallest_n_for_gcd_lcm_l42_42588

theorem smallest_n_for_gcd_lcm (n a b : ℕ) (h_gcd : Nat.gcd a b = 999) (h_lcm : Nat.lcm a b = Nat.factorial n) :
  n = 37 := sorry

end smallest_n_for_gcd_lcm_l42_42588


namespace smallest_n_integer_l42_42636

theorem smallest_n_integer (m n : ℕ) (s : ℝ) (h_m : m = (n + s)^4) (h_n_pos : 0 < n) (h_s_range : 0 < s ∧ s < 1 / 2000) : n = 8 := 
by
  sorry

end smallest_n_integer_l42_42636


namespace count_positive_integers_l42_42608

noncomputable def count_satisfying_n : ℕ :=
  Nat.factorization (2^9 * 3^4 * 5^4 * 7^2 * 11)

theorem count_positive_integers (n : ℕ) :
  ( ∃ m : ℕ, n = 3 * m ∧ Nat.lcm 5040 n = 3 * Nat.gcd 479001600 n ) ↔ n = 600 :=
by
  sorry

end count_positive_integers_l42_42608


namespace probability_blue_face_eq_one_third_l42_42900

-- Define the necessary conditions
def numberOfFaces : Nat := 12
def numberOfBlueFaces : Nat := 4

-- Define the term representing the probability
def probabilityOfBlueFace : ℚ := numberOfBlueFaces / numberOfFaces

-- The theorem to prove that the probability is 1/3
theorem probability_blue_face_eq_one_third :
  probabilityOfBlueFace = (1 : ℚ) / 3 :=
  by
  sorry

end probability_blue_face_eq_one_third_l42_42900


namespace fifteenth_odd_multiple_of_5_l42_42688

theorem fifteenth_odd_multiple_of_5 : ∃ (n : Nat), n = 15 → 10 * n - 5 = 145 :=
by
  intro n hn
  have h : 10 * 15 - 5 = 145 := by
    calc
      10 * 15 - 5 = 150 - 5 : by rw (Nat.mul_eq_mul_left n 10)
                ... = 145    : by rfl
  exact ⟨15, h⟩
  sorry

end fifteenth_odd_multiple_of_5_l42_42688


namespace part1_solution_set_part2_range_a_l42_42392

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l42_42392


namespace power_mod_eq_remainder_l42_42825

theorem power_mod_eq_remainder (b m e : ℕ) (hb : b = 17) (hm : m = 23) (he : e = 2090) : 
  b^e % m = 12 := 
  by sorry

end power_mod_eq_remainder_l42_42825


namespace find_m_for_parallel_lines_l42_42440

open Real

theorem find_m_for_parallel_lines :
  ∀ (m : ℝ),
    (∀ x y : ℝ, 3 * x + 4 * y - 3 = 0 → 6 * x + m * y + 14 = 0 → 3 * m = 4 * 6) →
    m = 8 :=
by
  intro m h
  have H : 3 * m = 4 * 6 := h 0 0 sorry sorry
  linarith

end find_m_for_parallel_lines_l42_42440


namespace base_conversion_problem_l42_42066

theorem base_conversion_problem (n d : ℕ) (hn : 0 < n) (hd : d < 10) 
  (h1 : 3 * n^2 + 2 * n + d = 263) (h2 : 3 * n^2 + 2 * n + 4 = 253 + 6 * d) : 
  n + d = 11 :=
by
  sorry

end base_conversion_problem_l42_42066


namespace inequality_proof_l42_42875

theorem inequality_proof (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : 
  (1 / a + 4 / (1 - a) ≥ 9) := 
sorry

end inequality_proof_l42_42875


namespace smallest_number_of_marbles_l42_42138

-- Define the conditions
variables (r w b g n : ℕ)
def valid_total (r w b g n : ℕ) := r + w + b + g = n
def valid_probability_4r (r w b g n : ℕ) := r * (r-1) * (r-2) * (r-3) = 24 * (w) * (r * (r - 1) * (r - 2) / 6)
def valid_probability_1w3r (r w b g n : ℕ) := r * (r-1) * (r-2) * (r-3) = 24 * (w * b * (r * (r - 1) / 2))
def valid_probability_1w1b2r (r w b g n : ℕ) := w * b * (r * (r - 1) / 2) = w * b * g * r

theorem smallest_number_of_marbles :
  ∃ n r w b g, valid_total r w b g n ∧
  valid_probability_4r r w b g n ∧
  valid_probability_1w3r r w b g n ∧
  valid_probability_1w1b2r r w b g n ∧ 
  n = 21 :=
  sorry

end smallest_number_of_marbles_l42_42138


namespace find_land_area_l42_42703

variable (L : ℝ) -- cost of land per square meter
variable (B : ℝ) -- cost of bricks per 1000 bricks
variable (R : ℝ) -- cost of roof tiles per tile
variable (numBricks : ℝ) -- number of bricks needed
variable (numTiles : ℝ) -- number of roof tiles needed
variable (totalCost : ℝ) -- total construction cost

theorem find_land_area (h1 : L = 50) 
                       (h2 : B = 100)
                       (h3 : R = 10) 
                       (h4 : numBricks = 10000) 
                       (h5 : numTiles = 500) 
                       (h6 : totalCost = 106000) : 
                       ∃ x : ℝ, 50 * x + (numBricks / 1000) * B + numTiles * R = totalCost ∧ x = 2000 := 
by 
  use 2000
  simp [h1, h2, h3, h4, h5, h6]
  norm_num
  done

end find_land_area_l42_42703


namespace part1_solution_set_part2_range_of_a_l42_42414

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l42_42414


namespace find_interval_l42_42862

theorem find_interval (x : ℝ) : x^2 + 3*x < 10 ↔ -5 < x ∧ x < 2 :=
by
  sorry

end find_interval_l42_42862


namespace ian_money_left_l42_42894

theorem ian_money_left
  (hours_worked : ℕ)
  (hourly_rate : ℕ)
  (spending_percentage : ℚ)
  (total_earnings : ℕ)
  (amount_spent : ℕ)
  (amount_left : ℕ)
  (h_worked : hours_worked = 8)
  (h_rate : hourly_rate = 18)
  (h_spending : spending_percentage = 0.5)
  (h_earnings : total_earnings = hours_worked * hourly_rate)
  (h_spent : amount_spent = total_earnings * spending_percentage)
  (h_left : amount_left = total_earnings - amount_spent) :
  amount_left = 72 := 
  sorry

end ian_money_left_l42_42894


namespace part1_solution_set_part2_range_of_a_l42_42419

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l42_42419


namespace part1_solution_set_part2_range_l42_42357

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l42_42357


namespace max_x_minus_y_l42_42174

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : (x - y) ≤ 1 + 3 * (Real.sqrt 2) :=
sorry

end max_x_minus_y_l42_42174


namespace fraction_halfway_between_fraction_halfway_between_l42_42682

theorem fraction_halfway_between : (3/4 : ℚ) < (5/7 : ℚ) :=
by linarith

theorem fraction_halfway_between : (41 / 56 : ℚ) = (1 / 2) * ((3 / 4) + (5 / 7)) :=
by sorry

end fraction_halfway_between_fraction_halfway_between_l42_42682


namespace ducks_remaining_l42_42094

theorem ducks_remaining (D_0 : ℕ) (D_0_eq : D_0 = 320) :
  let D_1 := D_0 - D_0 / 4,
      D_2 := D_1 - D_1 / 6,
      D_3 := D_2 - (3 * D_2) / 10 in
  D_3 = 140 :=
by
  sorry

end ducks_remaining_l42_42094


namespace misha_total_shots_l42_42110

theorem misha_total_shots (x y : ℕ) 
  (h1 : 18 * x + 5 * y = 99) 
  (h2 : 2 * x + y = 15) 
  (h3 : (15 / 0.9375 : ℝ) = 16) : 
  (¬(x = 0) ∧ ¬(y = 24)) ->
  16 = 16 :=
by
  sorry

end misha_total_shots_l42_42110


namespace purely_imaginary_iff_m_eq_1_l42_42349

theorem purely_imaginary_iff_m_eq_1 (m : ℝ) :
  (m^2 - 1 = 0 ∧ m + 1 ≠ 0) → m = 1 :=
by
  sorry

end purely_imaginary_iff_m_eq_1_l42_42349


namespace men_per_table_l42_42306

theorem men_per_table (total_tables : ℕ) (women_per_table : ℕ) (total_customers : ℕ) (total_women : ℕ)
    (h1 : total_tables = 9)
    (h2 : women_per_table = 7)
    (h3 : total_customers = 90)
    (h4 : total_women = women_per_table * total_tables)
    (h5 : total_women + total_men = total_customers) :
  total_men / total_tables = 3 :=
by
  have total_women := 7 * 9
  have total_men := 90 - total_women
  exact sorry

end men_per_table_l42_42306


namespace part1_solution_set_part2_range_of_a_l42_42416

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l42_42416


namespace channels_taken_away_l42_42229

theorem channels_taken_away (X : ℕ) : 
  (150 - X + 12 - 10 + 8 + 7 = 147) -> X = 20 :=
by
  sorry

end channels_taken_away_l42_42229


namespace parallelogram_height_l42_42216

variable (base height area : ℝ)
variable (h_eq_diag : base = 30)
variable (h_eq_area : area = 600)

theorem parallelogram_height :
  (height = 20) ↔ (base * height = area) :=
by
  sorry

end parallelogram_height_l42_42216


namespace smallest_x_for_multiple_l42_42280

theorem smallest_x_for_multiple (x : ℕ) (h720 : 720 = 2^4 * 3^2 * 5) (h1250 : 1250 = 2 * 5^4) : 
  (∃ x, (x > 0) ∧ (1250 ∣ (720 * x))) → x = 125 :=
by
  sorry

end smallest_x_for_multiple_l42_42280


namespace complex_quadrant_is_first_l42_42787

variable (z1 z2 : ℂ)
def point_quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First quadrant"
  else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
  else "Fourth quadrant"

theorem complex_quadrant_is_first :
  point_quadrant ((1 + 3i) * (3 - i)) = "First quadrant" :=
by
  sorry

end complex_quadrant_is_first_l42_42787


namespace probability_is_two_fifths_l42_42522

-- Define the set of integers
def S : Finset ℤ := {-10, -7, 0, 5, 8}

-- The total number of ways to choose two different integers from S
def total_pairs : ℕ := Finset.card (S.powersetLen 2)

-- The number of successful outcomes (choosing one negative and one positive integer)
def successful_pairs : ℕ := 4

-- The probability that the product of two chosen integers is negative
def probability_neg_product : ℚ := successful_pairs / total_pairs

theorem probability_is_two_fifths :
  probability_neg_product = 2 / 5 :=
by
  -- This part is intentionally left as "sorry" to align with the instructions.
  sorry

end probability_is_two_fifths_l42_42522


namespace kaylin_is_younger_by_five_l42_42793

def Freyja_age := 10
def Kaylin_age := 33
def Eli_age := Freyja_age + 9
def Sarah_age := 2 * Eli_age
def age_difference := Sarah_age - Kaylin_age

theorem kaylin_is_younger_by_five : age_difference = 5 := 
by
  show 5 = Sarah_age - Kaylin_age
  sorry

end kaylin_is_younger_by_five_l42_42793


namespace part1_l42_42435

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

theorem part1 {x : ℝ} : f x ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) := by
  sorry

end part1_l42_42435


namespace total_birds_in_store_l42_42709

def num_bird_cages := 4
def parrots_per_cage := 8
def parakeets_per_cage := 2
def birds_per_cage := parrots_per_cage + parakeets_per_cage
def total_birds := birds_per_cage * num_bird_cages

theorem total_birds_in_store : total_birds = 40 :=
  by sorry

end total_birds_in_store_l42_42709


namespace find_value_divide_subtract_l42_42129

theorem find_value_divide_subtract :
  (Number = 8 * 156 + 2) → 
  (CorrectQuotient = Number / 5) → 
  (Value = CorrectQuotient - 3) → 
  Value = 247 :=
by
  intros h1 h2 h3
  sorry

end find_value_divide_subtract_l42_42129


namespace relationship_among_terms_l42_42447

theorem relationship_among_terms (a : ℝ) (h : a ^ 2 + a < 0) : 
  -a > a ^ 2 ∧ a ^ 2 > -a ^ 2 ∧ -a ^ 2 > a :=
sorry

end relationship_among_terms_l42_42447


namespace min_expr_value_l42_42880

theorem min_expr_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 2) :
  (∃ a, a = (4 / (x + 2) + (3 * x - 7) / (3 * y + 4)) ∧ a ≥ 0) → 
  (∀ (u v : ℝ), u = x + 2 → v = 3 * y + 4 → u * v = 16) →
  (4 / (x + 2) + (3 * x - 7) / (3 * y + 4)) ≥ 11 / 16 :=
sorry

end min_expr_value_l42_42880


namespace find_x_in_equation_l42_42734

theorem find_x_in_equation :
  ∃ x : ℝ, 2.5 * ( (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) ) = 2000.0000000000002 ∧ x = 0.3 :=
by 
  sorry

end find_x_in_equation_l42_42734


namespace tangent_sum_problem_l42_42019

theorem tangent_sum_problem
  (α β : ℝ)
  (h_eq_root : ∃ (x y : ℝ), (x = Real.tan α) ∧ (y = Real.tan β) ∧ (6*x^2 - 5*x + 1 = 0) ∧ (6*y^2 - 5*y + 1 = 0))
  (h_range_α : 0 < α ∧ α < π/2)
  (h_range_β : π < β ∧ β < 3*π/2) :
  (Real.tan (α + β) = 1) ∧ (α + β = 5*π/4) := 
sorry

end tangent_sum_problem_l42_42019


namespace solve_for_x_l42_42699

theorem solve_for_x (x : ℝ) (h : (15 - 2 + (x / 1)) / 2 * 8 = 77) : x = 6.25 :=
by
  sorry

end solve_for_x_l42_42699


namespace simplest_common_denominator_of_fractions_l42_42947

noncomputable def simplestCommonDenominator (a b : ℕ) (x y : ℕ) : ℕ := 6 * (x ^ 2) * (y ^ 3)

theorem simplest_common_denominator_of_fractions :
  simplestCommonDenominator 2 6 x y = 6 * x^2 * y^3 :=
by
  sorry

end simplest_common_denominator_of_fractions_l42_42947


namespace Krishan_has_4046_l42_42663

variable (Ram Gopal Krishan : ℕ) -- Define the variables

-- Conditions given in the problem
axiom ratio_Ram_Gopal : Ram * 17 = Gopal * 7
axiom ratio_Gopal_Krishan : Gopal * 17 = Krishan * 7
axiom Ram_value : Ram = 686

-- This is the goal to prove
theorem Krishan_has_4046 : Krishan = 4046 :=
by
  -- Here is where the proof would go
  sorry

end Krishan_has_4046_l42_42663


namespace find_M_range_of_a_l42_42005

def Δ (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∉ B }

def A : Set ℝ := { x | 4 * x^2 + 9 * x + 2 < 0 }

def B : Set ℝ := { x | -1 < x ∧ x < 2 }

def M : Set ℝ := Δ B A

def P (a: ℝ) : Set ℝ := { x | (x - 2 * a) * (x + a - 2) < 0 }

theorem find_M :
  M = { x | -1/4 ≤ x ∧ x < 2 } :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ M → x ∈ P a) →
  a < -1/8 ∨ a > 9/4 :=
sorry

end find_M_range_of_a_l42_42005


namespace possible_sums_of_digits_l42_42119

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def all_digits_nonzero (A : ℕ) : Prop :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

def reverse_number (A : ℕ) : ℕ :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  1000 * d + 100 * c + 10 * b + a

def sum_of_digits (A : ℕ) : ℕ :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  a + b + c + d

theorem possible_sums_of_digits (A B : ℕ) 
  (h_four_digit : is_four_digit_number A) 
  (h_nonzero_digits : all_digits_nonzero A) 
  (h_reverse : B = reverse_number A) 
  (h_divisible : (A + B) % 109 = 0) : 
  sum_of_digits A = 14 ∨ sum_of_digits A = 23 ∨ sum_of_digits A = 28 := 
sorry

end possible_sums_of_digits_l42_42119


namespace second_polygon_sides_l42_42524

-- Conditions as definitions
def perimeter_first_polygon (s : ℕ) := 50 * (3 * s)
def perimeter_second_polygon (N s : ℕ) := N * s
def same_perimeter (s N : ℕ) := perimeter_first_polygon s = perimeter_second_polygon N s

-- Theorem statement
theorem second_polygon_sides (s N : ℕ) :
  same_perimeter s N → N = 150 :=
by
  sorry

end second_polygon_sides_l42_42524


namespace bus_driver_total_compensation_l42_42701

theorem bus_driver_total_compensation :
  let regular_rate := 16
  let regular_hours := 40
  let overtime_hours := 60 - regular_hours
  let overtime_rate := regular_rate + 0.75 * regular_rate
  let regular_pay := regular_hours * regular_rate
  let overtime_pay := overtime_hours * overtime_rate
  let total_compensation := regular_pay + overtime_pay
  total_compensation = 1200 := by
  sorry

end bus_driver_total_compensation_l42_42701


namespace smallest_number_l42_42291

theorem smallest_number:
  ∃ n : ℕ, (∀ d ∈ [12, 16, 18, 21, 28, 35, 39], (n - 7) % d = 0) ∧ n = 65527 :=
by
  sorry

end smallest_number_l42_42291


namespace max_value_g_l42_42733

noncomputable def g (x : ℝ) := 4 * x - x ^ 4

theorem max_value_g : 
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.sqrt 4 ∧
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ Real.sqrt 4 → g y ≤ 3 :=
sorry

end max_value_g_l42_42733


namespace circus_tickets_l42_42940

variable (L U : ℕ)

theorem circus_tickets (h1 : L + U = 80) (h2 : 30 * L + 20 * U = 2100) : L = 50 :=
by
  sorry

end circus_tickets_l42_42940


namespace cos_theta_seven_l42_42032

theorem cos_theta_seven {θ : ℝ} (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -8383 / 98304 :=
by
  sorry

end cos_theta_seven_l42_42032


namespace sum_of_integers_l42_42899

theorem sum_of_integers (x y : ℕ) (h1 : x = y + 3) (h2 : x^3 - y^3 = 63) : x + y = 5 :=
by
  sorry

end sum_of_integers_l42_42899


namespace root_exists_between_a_and_b_l42_42494

variable {α : Type*} [LinearOrderedField α]

theorem root_exists_between_a_and_b (a b p q : α) (h₁ : a^2 + p * a + q = 0) (h₂ : b^2 - p * b - q = 0) (h₃ : q ≠ 0) :
  ∃ c, a < c ∧ c < b ∧ (c^2 + 2 * p * c + 2 * q = 0) := by
  sorry

end root_exists_between_a_and_b_l42_42494


namespace tens_digit_of_2013_squared_minus_2013_l42_42686

theorem tens_digit_of_2013_squared_minus_2013 : (2013^2 - 2013) % 100 / 10 = 5 := by
  sorry

end tens_digit_of_2013_squared_minus_2013_l42_42686


namespace part1_l42_42438

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

theorem part1 {x : ℝ} : f x ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) := by
  sorry

end part1_l42_42438


namespace cos_sq_sin_sq_equiv_l42_42934

-- Given angle in degrees converted to radians
def deg_to_rad (d: ℝ) : ℝ := d * (Real.pi / 180)

-- Helper Definitions for cos^2 and sin^2 involving 18 degrees
def cos_sq_18 : ℝ := (Real.cos (deg_to_rad 18)) ^ 2
def sin_sq_18 : ℝ := (Real.sin (deg_to_rad 18)) ^ 2

theorem cos_sq_sin_sq_equiv : 4 * cos_sq_18 - 1 = 1 / (4 * sin_sq_18) :=
by 
  sorry

end cos_sq_sin_sq_equiv_l42_42934


namespace find_difference_in_ticket_costs_l42_42072

-- Conditions
def num_adults : ℕ := 9
def num_children : ℕ := 7
def cost_adult_ticket : ℕ := 11
def cost_child_ticket : ℕ := 7

def total_cost_adults : ℕ := num_adults * cost_adult_ticket
def total_cost_children : ℕ := num_children * cost_child_ticket
def total_tickets : ℕ := num_adults + num_children

-- Discount conditions (not needed for this proof since they don't apply)
def apply_discount (total_tickets : ℕ) (total_cost : ℕ) : ℕ :=
  if total_tickets >= 10 ∧ total_tickets <= 12 then
    total_cost * 9 / 10
  else if total_tickets >= 13 ∧ total_tickets <= 15 then
    total_cost * 85 / 100
  else
    total_cost

-- The main statement to prove
theorem find_difference_in_ticket_costs : total_cost_adults - total_cost_children = 50 := by
  sorry

end find_difference_in_ticket_costs_l42_42072


namespace fifteenth_odd_multiple_of_5_l42_42689

theorem fifteenth_odd_multiple_of_5 :
  (∃ n: ℕ, n = 15 ∧ (10 * n - 5 = 145)) :=
begin
  use 15,
  split,
  { refl },
  { norm_num }
end

end fifteenth_odd_multiple_of_5_l42_42689


namespace part1_solution_set_part2_range_l42_42350

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l42_42350


namespace subcommittee_ways_l42_42553

theorem subcommittee_ways (R D : ℕ) (hR : R = 8) (hD : D = 10) :
  let chooseR := Nat.choose 8 3,
      chooseChair := Nat.choose 10 1,
      chooseRestD := Nat.choose 9 1 in
  chooseR * chooseChair * chooseRestD = 5040 :=
by
  intro chooseR chooseChair chooseRestD
  rw [hR, hD]
  have h1 : chooseR = Nat.choose 8 3 := rfl
  have h2 : chooseChair = Nat.choose 10 1 := rfl
  have h3 : chooseRestD = Nat.choose 9 1 := rfl
  rw [h1, h2, h3]
  sorry

end subcommittee_ways_l42_42553


namespace tangent_line_eq_monotonic_intervals_extremes_f_l42_42884

variables {a x : ℝ}

noncomputable def f (a x : ℝ) : ℝ := -1/3 * x^3 + 2 * a * x^2 - 3 * a^2 * x
noncomputable def f' (a x : ℝ) : ℝ := -x^2 + 4 * a * x - 3 * a^2

theorem tangent_line_eq {a : ℝ} (h : a = -1) : (∃ y, y = f (-1) (-2) ∧ 3 * x - 3 * y + 8 = 0) := sorry

theorem monotonic_intervals_extremes {a : ℝ} (h : 0 < a) :
  (∀ x, (a < x ∧ x < 3 * a → 0 < f' a x) ∧ 
        (x < a ∨ 3 * a < x → f' a x < 0) ∧ 
        (f a (3 * a) = 0 ∧ f a a = -4/3 * a^3)) := sorry

theorem f'_inequality_range (h1 : ∀ x, 2 * a ≤ x ∧ x ≤ 2 * a + 2 → |f' a x| ≤ 3 * a) :
  (1 ≤ a ∧ a ≤ 3) := sorry

end tangent_line_eq_monotonic_intervals_extremes_f_l42_42884


namespace total_cars_l42_42483

-- Definitions for the conditions
def cathy_cars : ℕ := 5
def lindsey_cars : ℕ := cathy_cars + 4
def carol_cars : ℕ := 2 * cathy_cars
def susan_cars : ℕ := carol_cars - 2

-- Lean theorem statement
theorem total_cars : cathy_cars + lindsey_cars + carol_cars + susan_cars = 32 := by
  sorry

end total_cars_l42_42483


namespace die_vanishing_probability_and_floor_value_l42_42091

/-
Given conditions:
1. The die has four faces labeled 0, 1, 2, 3.
2. When the die lands on a face labeled:
   - 0: the die vanishes.
   - 1: nothing happens (one die remains).
   - 2: the die replicates into 2 dice.
   - 3: the die replicates into 3 dice.
3. All dice (original and replicas) will continuously be rolled.
Prove:
  The value of ⌊10/p⌋ is 24, where p is the probability that all dice will eventually disappear.
-/

theorem die_vanishing_probability_and_floor_value : 
  ∃ (p : ℝ), 
  (p^3 + p^2 - 3 * p + 1 = 0 ∧ 0 ≤ p ∧ p ≤ 1 ∧ p = Real.sqrt 2 - 1) 
  ∧ ⌊10 / p⌋ = 24 := 
    sorry

end die_vanishing_probability_and_floor_value_l42_42091


namespace five_digit_number_unique_nonzero_l42_42214

theorem five_digit_number_unique_nonzero (a b c d e : ℕ) (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) (h2 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) (h3 : (100 * a + 10 * b + c) * 7 = 100 * c + 10 * d + e) : a = 1 ∧ b = 2 ∧ c = 9 ∧ d = 4 ∧ e = 6 :=
by
  sorry

end five_digit_number_unique_nonzero_l42_42214


namespace radioactive_decay_minimum_years_l42_42979

noncomputable def min_years (a : ℝ) (n : ℕ) : Prop :=
  (a * (1 - 3 / 4) ^ n ≤ a * 1 / 100)

theorem radioactive_decay_minimum_years (a : ℝ) (h : 0 < a) : ∃ n : ℕ, min_years a n ∧ n = 4 :=
by {
  sorry
}

end radioactive_decay_minimum_years_l42_42979


namespace coffee_table_price_correct_l42_42631

-- Conditions
def sofa_cost : ℕ := 1250
def armchair_cost_each : ℕ := 425
def num_armchairs : ℕ := 2
def total_invoice : ℕ := 2430

-- Question: What is the price of the coffee table?
def coffee_table_price : ℕ := total_invoice - (sofa_cost + num_armchairs * armchair_cost_each)

-- Proof statement (to be completed)
theorem coffee_table_price_correct : coffee_table_price = 330 := by
  sorry

end coffee_table_price_correct_l42_42631


namespace measure_of_side_XY_l42_42276

theorem measure_of_side_XY 
  (a b c : ℝ) 
  (Area : ℝ)
  (h1 : a = 30)
  (h2 : b = 60)
  (h3 : c = 90)
  (h4 : a + b + c = 180)
  (h_area : Area = 36)
  : (∀ (XY YZ XZ : ℝ), XY = 4.56) :=
by
  sorry

end measure_of_side_XY_l42_42276


namespace teacher_works_days_in_month_l42_42566

theorem teacher_works_days_in_month (P : ℕ) (W : ℕ) (M : ℕ) (T : ℕ) (H1 : P = 5) (H2 : W = 5) (H3 : M = 6) (H4 : T = 3600) : 
  (T / M) / (P * W) = 24 :=
by
  sorry

end teacher_works_days_in_month_l42_42566


namespace unused_combinations_eq_40_l42_42313

-- Defining the basic parameters
def num_resources : ℕ := 6
def total_combinations : ℕ := 2 ^ num_resources
def used_combinations : ℕ := 23

-- Calculating the number of unused combinations
theorem unused_combinations_eq_40 : total_combinations - 1 - used_combinations = 40 := by
  sorry

end unused_combinations_eq_40_l42_42313


namespace abs_diff_of_two_numbers_l42_42666

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) : |x - y| = 4 :=
by 
  calc |x - y| = _ 
  sorry

end abs_diff_of_two_numbers_l42_42666


namespace half_of_1_point_6_times_10_pow_6_l42_42534

theorem half_of_1_point_6_times_10_pow_6 : (1.6 * 10^6) / 2 = 8 * 10^5 :=
by
  sorry

end half_of_1_point_6_times_10_pow_6_l42_42534


namespace total_cars_all_own_l42_42479

theorem total_cars_all_own :
  ∀ (C L S K : ℕ), 
  (C = 5) →
  (L = C + 4) →
  (K = 2 * C) →
  (S = K - 2) →
  (C + L + K + S = 32) :=
by
  intros C L S K
  intro hC
  intro hL
  intro hK
  intro hS
  sorry

end total_cars_all_own_l42_42479


namespace option_D_correct_l42_42136

noncomputable def y1 (x : ℝ) : ℝ := 1 / x
noncomputable def y2 (x : ℝ) : ℝ := x^2
noncomputable def y3 (x : ℝ) : ℝ := (1 / 2)^x
noncomputable def y4 (x : ℝ) : ℝ := 1 / x^2

theorem option_D_correct :
  (∀ x : ℝ, y4 x = y4 (-x)) ∧ (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → y4 x₁ > y4 x₂) :=
by
  sorry

end option_D_correct_l42_42136


namespace polygon_sides_l42_42055

theorem polygon_sides (n : ℕ) : 
  (180 * (n - 2) / 360 = 5 / 2) → n = 7 :=
by
  sorry

end polygon_sides_l42_42055


namespace other_point_on_circle_l42_42790

noncomputable def circle_center_radius (p : ℝ × ℝ) (r : ℝ) : Prop :=
  dist p (0, 0) = r

theorem other_point_on_circle (r : ℝ) (h : r = 16) (point_on_circle : circle_center_radius (16, 0) r) :
  circle_center_radius (-16, 0) r :=
by
  sorry

end other_point_on_circle_l42_42790


namespace find_factor_l42_42302

theorem find_factor (x f : ℕ) (h1 : x = 15) (h2 : (2 * x + 5) * f = 105) : f = 3 :=
sorry

end find_factor_l42_42302


namespace find_ratio_l42_42842

-- Definitions
noncomputable def cost_per_gram_A : ℝ := 0.01
noncomputable def cost_per_gram_B : ℝ := 0.008
noncomputable def new_cost_per_gram_A : ℝ := 0.011
noncomputable def new_cost_per_gram_B : ℝ := 0.0072

def total_weight : ℝ := 1000

-- Theorem statement
theorem find_ratio (x y : ℝ) (h1 : x + y = total_weight)
    (h2 : cost_per_gram_A * x + cost_per_gram_B * y = new_cost_per_gram_A * x + new_cost_per_gram_B * y) :
    x / y = 4 / 5 :=
by
  sorry

end find_ratio_l42_42842


namespace boat_distance_against_stream_l42_42058

theorem boat_distance_against_stream 
  (v_b : ℝ)
  (v_s : ℝ)
  (distance_downstream : ℝ)
  (t : ℝ)
  (speed_downstream : v_s + v_b = 11)
  (speed_still_water : v_b = 8)
  (time : t = 1) :
  (v_b - (11 - v_b)) * t = 5 :=
by
  -- Here we're given the initial conditions and have to show the final distance against the stream is 5 km
  sorry

end boat_distance_against_stream_l42_42058


namespace initial_walnut_trees_l42_42956

/-- 
  Given there are 29 walnut trees in the park after cutting down 13 walnut trees, 
  prove that initially there were 42 walnut trees in the park.
-/
theorem initial_walnut_trees (cut_walnut_trees remaining_walnut_trees initial_walnut_trees : ℕ) 
  (h₁ : cut_walnut_trees = 13)
  (h₂ : remaining_walnut_trees = 29)
  (h₃ : initial_walnut_trees = cut_walnut_trees + remaining_walnut_trees) :
  initial_walnut_trees = 42 := 
sorry

end initial_walnut_trees_l42_42956


namespace smallest_n_condition_l42_42772

theorem smallest_n_condition :
  ∃ n ≥ 2, ∃ (a : Fin n → ℤ), (Finset.sum Finset.univ a = 1990) ∧ (Finset.univ.prod a = 1990) ∧ (n = 5) :=
by
  sorry

end smallest_n_condition_l42_42772


namespace initial_number_of_kids_l42_42099

theorem initial_number_of_kids (joined kids_total initial : ℕ) (h1 : joined = 22) (h2 : kids_total = 36) (h3 : kids_total = initial + joined) : initial = 14 :=
by 
  -- Proof goes here
  sorry

end initial_number_of_kids_l42_42099


namespace part1_solution_set_part2_range_of_a_l42_42381

-- Part 1
theorem part1_solution_set (x : ℝ) : (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x + 3| > -a) ↔ (a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_of_a_l42_42381


namespace range_of_a_proof_l42_42973

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

theorem range_of_a_proof (a : ℝ) : range_of_a a ↔ 0 ≤ a ∧ a < 4 :=
by
  sorry

end range_of_a_proof_l42_42973


namespace cos_identity_l42_42873

theorem cos_identity (θ : ℝ) (h : Real.cos (π / 6 + θ) = (Real.sqrt 3) / 3) : 
  Real.cos (5 * π / 6 - θ) = - (Real.sqrt 3 / 3) :=
by
  sorry

end cos_identity_l42_42873


namespace smallest_possible_b_l42_42256

theorem smallest_possible_b 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a - b = 8) 
  (h4 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  b = 4 := 
sorry

end smallest_possible_b_l42_42256


namespace arithmetic_sequence_a5_l42_42222

variable (a : ℕ → ℝ)

-- Conditions translated to Lean definitions
def cond1 : Prop := a 3 = 7
def cond2 : Prop := a 9 = 19

-- Theorem statement that needs to be proved
theorem arithmetic_sequence_a5 (h1 : cond1 a) (h2 : cond2 a) : a 5 = 11 :=
sorry

end arithmetic_sequence_a5_l42_42222


namespace mean_value_z_l42_42942

theorem mean_value_z (z : ℚ) (h : (7 + 10 + 23) / 3 = (18 + z) / 2) : z = 26 / 3 :=
by
  sorry

end mean_value_z_l42_42942


namespace part1_solution_set_part2_range_a_l42_42388

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l42_42388


namespace new_total_weight_correct_l42_42515

-- Definitions based on the problem statement
variables (R S k : ℝ)
def ram_original_weight : ℝ := 2 * k
def shyam_original_weight : ℝ := 5 * k
def ram_new_weight : ℝ := 1.10 * (ram_original_weight k)
def shyam_new_weight : ℝ := 1.17 * (shyam_original_weight k)

-- Definition for total original weight and increased weight
def total_original_weight : ℝ := ram_original_weight k + shyam_original_weight k
def total_weight_increased : ℝ := 1.15 * total_original_weight k
def new_total_weight : ℝ := ram_new_weight k + shyam_new_weight k

-- The proof statement
theorem new_total_weight_correct :
  new_total_weight k = total_weight_increased k :=
by
  sorry

end new_total_weight_correct_l42_42515


namespace probability_event_A_l42_42051

open Classical
noncomputable theory

-- Define the finite set of university graduates
def graduates := {A, B, C, D, E}

-- Define the event "either A or B is hired" as a subset
def event_A (s : Set graduates) : Prop := A ∈ s ∨ B ∈ s

-- Define the number of ways to choose 3 out of 5
def comb_5_3 : ℕ := Nat.choose 5 3

-- Define the number of ways to choose 3 out of {C, D, E}
def comb_3_3 : ℕ := Nat.choose 3 3

-- Define the probability of the complement event 
def P_compl_A : ℚ := comb_3_3 / comb_5_3

-- Define the probability of event A 
def P_A : ℚ := 1 - P_compl_A

-- The theorem to prove
theorem probability_event_A : P_A = 9 / 10 := by
  sorry

end probability_event_A_l42_42051


namespace increased_work_l42_42115

variable (W p : ℕ)

theorem increased_work (hW : W > 0) (hp : p > 0) : 
  (W / (7 * p / 8)) - (W / p) = W / (7 * p) := 
sorry

end increased_work_l42_42115


namespace encoded_message_correct_l42_42528

def old_message := "011011010011"
def new_message := "211221121"
def encoding_rules : Π (ch : Char), String
| 'A' => "21"
| 'B' => "122"
| 'C' => "1"
| _   => ""

theorem encoded_message_correct :
  (decode old_message = "ABCBA") ∧ (encode "ABCBA" = new_message) :=
by
  -- Proof will go here
  sorry

def decode : String → String := sorry  -- Provide implementation
def encode : String → String := sorry  -- Provide implementation

end encoded_message_correct_l42_42528


namespace nested_inverse_value_l42_42258

def f (x : ℝ) : ℝ := 5 * x + 6

noncomputable def f_inv (y : ℝ) : ℝ := (y - 6) / 5

theorem nested_inverse_value :
  f_inv (f_inv 16) = -4/5 :=
by
  sorry

end nested_inverse_value_l42_42258


namespace maximum_value_of_f_l42_42972

noncomputable def f (a x : ℝ) : ℝ := (1 + x) ^ a - a * x

theorem maximum_value_of_f (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  ∃ x : ℝ, x > -1 ∧ ∀ y : ℝ, y > -1 → f a y ≤ f a x ∧ f a x = 1 :=
by {
  sorry
}

end maximum_value_of_f_l42_42972


namespace ducks_remaining_after_three_nights_l42_42095

def initial_ducks : ℕ := 320
def first_night_ducks (initial_ducks : ℕ) : ℕ := initial_ducks - (initial_ducks / 4)
def second_night_ducks (first_night_ducks : ℕ) : ℕ := first_night_ducks - (first_night_ducks / 6)
def third_night_ducks (second_night_ducks : ℕ) : ℕ := second_night_ducks - (second_night_ducks * 30 / 100)

theorem ducks_remaining_after_three_nights : 
  third_night_ducks (second_night_ducks (first_night_ducks initial_ducks)) = 140 :=
by
  -- Proof goes here
  sorry

end ducks_remaining_after_three_nights_l42_42095


namespace part1_part2_l42_42359

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l42_42359


namespace treasure_contains_645_coins_max_leftover_coins_when_choosing_93_pirates_l42_42659

namespace PirateTreasure

-- Given conditions
def num_pirates_excl_captain := 100
def max_coins := 1000
def remaining_coins_99_pirates := 51
def remaining_coins_77_pirates := 29

-- Problem Part (a): Prove the number of coins in treasure
theorem treasure_contains_645_coins : 
  ∃ (N : ℕ), N < max_coins ∧ (N % 99 = remaining_coins_99_pirates ∧ N % 77 = remaining_coins_77_pirates) ∧ N = 645 :=
  sorry

-- Problem Part (b): Prove the number of pirates Barbaroxa should choose
theorem max_leftover_coins_when_choosing_93_pirates :
  ∃ (n : ℕ), n ≤ num_pirates_excl_captain ∧ (∀ k, k ≤ num_pirates_excl_captain → (645 % k) ≤ (645 % k) ∧ n = 93) :=
  sorry

end PirateTreasure

end treasure_contains_645_coins_max_leftover_coins_when_choosing_93_pirates_l42_42659


namespace boys_in_classroom_l42_42517

-- Definitions of the conditions
def total_children := 45
def girls_fraction := 1 / 3

-- The theorem we want to prove
theorem boys_in_classroom : (2 / 3) * total_children = 30 := by
  sorry

end boys_in_classroom_l42_42517


namespace part1_solution_set_part2_range_of_a_l42_42420

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l42_42420


namespace algebraic_expression_eval_l42_42692

theorem algebraic_expression_eval (a b c : ℝ) (h : a * (-5:ℝ)^4 + b * (-5)^2 + c = 3): 
  a * (5:ℝ)^4 + b * (5)^2 + c = 3 :=
by
  sorry

end algebraic_expression_eval_l42_42692


namespace coffee_remaining_after_shrink_l42_42562

-- Definitions of conditions in the problem
def shrink_factor : ℝ := 0.5
def cups_before_shrink : ℕ := 5
def ounces_per_cup_before_shrink : ℝ := 8

-- Definition of the total ounces of coffee remaining after shrinking
def ounces_per_cup_after_shrink : ℝ := ounces_per_cup_before_shrink * shrink_factor
def total_ounces_after_shrink : ℝ := cups_before_shrink * ounces_per_cup_after_shrink

-- The proof statement
theorem coffee_remaining_after_shrink :
  total_ounces_after_shrink = 20 :=
by
  -- Omitting the proof as only the statement is needed
  sorry

end coffee_remaining_after_shrink_l42_42562


namespace probability_of_graduate_degree_l42_42289

-- Define the conditions as Lean statements
variable (k m : ℕ)
variable (G := 1 * k) 
variable (C := 2 * m) 
variable (N1 := 8 * k) -- from the ratio G:N = 1:8
variable (N2 := 3 * m) -- from the ratio C:N = 2:3

-- Least common multiple (LCM) of 8 and 3 is 24
-- Therefore, determine specific values for G, C, and N
-- Given these updates from solution steps we set:
def G_scaled : ℕ := 3
def C_scaled : ℕ := 16
def N_scaled : ℕ := 24

-- Total number of college graduates
def total_college_graduates : ℕ := G_scaled + C_scaled

-- Probability q of picking a college graduate with a graduate degree
def q : ℚ := G_scaled / total_college_graduates

-- Lean proof statement for equivalence
theorem probability_of_graduate_degree : 
  q = 3 / 19 := by
sorry

end probability_of_graduate_degree_l42_42289


namespace exists_n_not_represented_l42_42064

theorem exists_n_not_represented (a b c d : ℤ) (a_gt_14 : a > 14)
  (h1 : 0 ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ a) :
  ∃ (n : ℕ), ¬ ∃ (x y z : ℤ), n = x * (a * x + b) + y * (a * y + c) + z * (a * z + d) :=
sorry

end exists_n_not_represented_l42_42064


namespace triangle_perimeter_is_26_l42_42748

-- Define the lengths of the medians as given conditions
def median1 := 3
def median2 := 4
def median3 := 6

-- Define the perimeter of the triangle
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- The theorem to prove that the perimeter is 26 cm
theorem triangle_perimeter_is_26 :
  perimeter (2 * median1) (2 * median2) (2 * median3) = 26 :=
by
  -- Calculation follows directly from the definition
  sorry

end triangle_perimeter_is_26_l42_42748


namespace raisin_cost_fraction_l42_42993

theorem raisin_cost_fraction
  (R : ℚ) -- cost of a pound of raisins in dollars
  (cost_of_nuts : ℚ)
  (total_cost_raisins : ℚ)
  (total_cost_nuts : ℚ) :
  cost_of_nuts = 3 * R →
  total_cost_raisins = 5 * R →
  total_cost_nuts = 4 * cost_of_nuts →
  (total_cost_raisins / (total_cost_raisins + total_cost_nuts)) = 5 / 17 :=
by
  sorry

end raisin_cost_fraction_l42_42993


namespace circle_radius_l42_42559

theorem circle_radius (M N : ℝ) (h1 : M / N = 20) :
  ∃ r : ℝ, M = π * r^2 ∧ N = 2 * π * r ∧ r = 40 :=
by
  sorry

end circle_radius_l42_42559


namespace complex_quadrant_l42_42786

theorem complex_quadrant (a b : ℝ) (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) 
  (h : z1 * z2 = a + b * complex.I) : a > 0 ∧ b > 0 :=
by
  sorry

end complex_quadrant_l42_42786


namespace complete_the_square_l42_42694

theorem complete_the_square (x : ℝ) : 
  (∃ a b : ℝ, (x^2 + 10 * x - 3 = 0) → ((x + a)^2 = b) ∧ b = 28) :=
sorry

end complete_the_square_l42_42694


namespace f1_g1_eq_one_l42_42340

-- Definitions of even and odd functions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- Given statement to be proved
theorem f1_g1_eq_one (f g : ℝ → ℝ) (h_even : even_function f) (h_odd : odd_function g)
    (h_diff : ∀ x, f x - g x = x^3 + x^2 + 1) : f 1 + g 1 = 1 :=
  sorry

end f1_g1_eq_one_l42_42340


namespace compute_f_2_neg3_neg1_l42_42898

def f (p q r : ℤ) : ℚ := (r + p : ℚ) / (r - q + 1 : ℚ)

theorem compute_f_2_neg3_neg1 : f 2 (-3) (-1) = 1 / 3 := 
by
  sorry

end compute_f_2_neg3_neg1_l42_42898


namespace solution_l42_42895

-- Define the conditions
def equation (x : ℝ) : Prop :=
  (x / 15) = (15 / x)

theorem solution (x : ℝ) : equation x → x = 15 ∨ x = -15 :=
by
  intros h
  -- The proof would go here.
  sorry

end solution_l42_42895


namespace factorial_base_9_zeroes_l42_42890

-- Main statement 
theorem factorial_base_9_zeroes (n : ℕ) (h : n = 15) : 
  let factors_of_3 := (nat.factorial 15).multiplicity 3 in   
   (factors_of_3 / 2).to_nat = 3 :=
by
  have : factors_of_3 = nat.factorial 15 / (9 ^ 3),
  { sorry }
  rw this
  sorry
  -- Continue steps to complete the proof

end factorial_base_9_zeroes_l42_42890


namespace problem_l42_42161

noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

noncomputable def f_deriv (x : ℝ) : ℝ := - (1 / x^2) * Real.cos x - (1 / x) * Real.sin x

theorem problem (h_pi_ne_zero : Real.pi ≠ 0) (h_pi_div_two_ne_zero : Real.pi / 2 ≠ 0) :
  f Real.pi + f_deriv (Real.pi / 2) = -3 / Real.pi  := by
  sorry

end problem_l42_42161


namespace maximum_value_of_x_minus_y_l42_42168

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
begin
  sorry
end

end maximum_value_of_x_minus_y_l42_42168


namespace exists_int_squares_l42_42473

theorem exists_int_squares (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  ∃ x y : ℤ, (a^2 + b^2)^n = x^2 + y^2 :=
by
  sorry

end exists_int_squares_l42_42473


namespace delta_max_success_ratio_l42_42778

theorem delta_max_success_ratio (y w x z : ℤ) (h1 : 360 + 240 = 600)
  (h2 : 0 < x ∧ x < y ∧ z < w)
  (h3 : y + w = 600)
  (h4 : (x : ℚ) / y < (200 : ℚ) / 360)
  (h5 : (z : ℚ) / w < (160 : ℚ) / 240)
  (h6 : (360 : ℚ) / 600 = 3 / 5)
  (h7 : (x + z) < 166) :
  (x + z : ℚ) / 600 ≤ 166 / 600 := 
sorry

end delta_max_success_ratio_l42_42778


namespace symmetry_with_respect_to_line_x_eq_1_l42_42586

theorem symmetry_with_respect_to_line_x_eq_1 (f : ℝ → ℝ) :
  ∀ x, f (x - 1) = f (1 - x) ↔ x - 1 = 1 - x :=
by
  sorry

end symmetry_with_respect_to_line_x_eq_1_l42_42586


namespace coin_probability_l42_42309

theorem coin_probability (p : ℚ) 
  (P_X_3 : ℚ := 10 * p^3 * (1 - p)^2)
  (P_X_4 : ℚ := 5 * p^4 * (1 - p))
  (P_X_5 : ℚ := p^5)
  (w : ℚ := P_X_3 + P_X_4 + P_X_5) :
  w = 5 / 16 → p = 1 / 4 :=
by
  sorry

end coin_probability_l42_42309


namespace ratio_of_lemons_l42_42070

theorem ratio_of_lemons :
  ∃ (L J E I : ℕ), 
  L = 5 ∧ 
  J = L + 6 ∧ 
  J = E / 3 ∧ 
  E = I / 2 ∧ 
  L + J + E + I = 115 ∧ 
  J / E = 1 / 3 :=
by
  sorry

end ratio_of_lemons_l42_42070


namespace interest_rate_first_part_l42_42495

theorem interest_rate_first_part (A A1 A2 I : ℝ) (r : ℝ) :
  A = 3200 →
  A1 = 800 →
  A2 = A - A1 →
  I = 144 →
  (A1 * r / 100 + A2 * 5 / 100 = I) →
  r = 3 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end interest_rate_first_part_l42_42495


namespace cos_identity_l42_42596

open Real

theorem cos_identity
  (θ : ℝ)
  (h1 : cos ((5 * π) / 12 + θ) = 3 / 5)
  (h2 : -π < θ ∧ θ < -π / 2) :
  cos ((π / 12) - θ) = -4 / 5 :=
by
  sorry

end cos_identity_l42_42596


namespace incorrect_vertex_is_false_l42_42887

-- Definition of the given parabola
def parabola (x : ℝ) : ℝ := -2 * (x - 2)^2 + 1

-- Define the incorrect hypothesis: Vertex at (-2, 1)
def incorrect_vertex (x y : ℝ) : Prop := (x, y) = (-2, 1)

-- Proposition to prove that the vertex is not at (-2, 1)
theorem incorrect_vertex_is_false : ¬ ∃ x y, (x, y) = (-2, 1) ∧ parabola x = y :=
by
  sorry

end incorrect_vertex_is_false_l42_42887


namespace surface_area_of_equal_volume_cube_l42_42980

def vol_rect_prism (l w h : ℝ) : ℝ := l * w * h
def surface_area_cube (s : ℝ) : ℝ := 6 * s^2

theorem surface_area_of_equal_volume_cube :
  (vol_rect_prism 5 5 45 = surface_area_cube 10.5) :=
by
  sorry

end surface_area_of_equal_volume_cube_l42_42980


namespace part1_solution_set_part2_range_a_l42_42389

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l42_42389


namespace unique_function_l42_42583

theorem unique_function (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f (x + 1) ≥ f x + 1) 
  (h2 : ∀ x y : ℝ, f (x * y) ≥ f x * f y) : 
  ∀ x : ℝ, f x = x := 
sorry

end unique_function_l42_42583


namespace part1_solution_set_part2_range_of_a_l42_42385

-- Part 1
theorem part1_solution_set (x : ℝ) : (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x + 3| > -a) ↔ (a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_of_a_l42_42385


namespace price_of_coffee_table_l42_42629

-- Define the given values
def price_sofa : ℕ := 1250
def price_armchair : ℕ := 425
def num_armchairs : ℕ := 2
def total_invoice : ℕ := 2430

-- Define the target value (price of the coffee table)
def price_coffee_table : ℕ := 330

-- The theorem to prove
theorem price_of_coffee_table :
  total_invoice = price_sofa + num_armchairs * price_armchair + price_coffee_table :=
by sorry

end price_of_coffee_table_l42_42629


namespace slices_served_today_l42_42844

-- Definitions based on conditions from part a)
def slices_lunch_today : ℕ := 7
def slices_dinner_today : ℕ := 5

-- Proof statement based on part c)
theorem slices_served_today : slices_lunch_today + slices_dinner_today = 12 := 
by
  sorry

end slices_served_today_l42_42844


namespace coffee_table_price_correct_l42_42630

-- Conditions
def sofa_cost : ℕ := 1250
def armchair_cost_each : ℕ := 425
def num_armchairs : ℕ := 2
def total_invoice : ℕ := 2430

-- Question: What is the price of the coffee table?
def coffee_table_price : ℕ := total_invoice - (sofa_cost + num_armchairs * armchair_cost_each)

-- Proof statement (to be completed)
theorem coffee_table_price_correct : coffee_table_price = 330 := by
  sorry

end coffee_table_price_correct_l42_42630


namespace cinema_chairs_l42_42719

theorem cinema_chairs (chairs_between : ℕ) (h : chairs_between = 30) :
  chairs_between + 2 = 32 := by
  sorry

end cinema_chairs_l42_42719


namespace Ian_money_left_l42_42891

-- Definitions based on the conditions
def hours_worked : ℕ := 8
def rate_per_hour : ℕ := 18
def total_money_made : ℕ := hours_worked * rate_per_hour
def money_left : ℕ := total_money_made / 2

-- The statement to be proved 
theorem Ian_money_left : money_left = 72 :=
by
  sorry

end Ian_money_left_l42_42891


namespace complex_expression_result_l42_42920

open Complex

theorem complex_expression_result (s : ℂ) (h1 : s^7 = 1) (h2 : s ≠ 1):
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) * (s^6 - 1) = 8 := 
by
  sorry

end complex_expression_result_l42_42920


namespace max_x_minus_y_l42_42176

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : (x - y) ≤ 1 + 3 * (Real.sqrt 2) :=
sorry

end max_x_minus_y_l42_42176


namespace min_value_of_ab_l42_42606

theorem min_value_of_ab {a b : ℝ} (ha : a > 0) (hb : b > 0)
    (h : 1 / a + 1 / b = 1) : a + b ≥ 4 :=
sorry

end min_value_of_ab_l42_42606


namespace ratio_of_x_intercepts_l42_42824

theorem ratio_of_x_intercepts (c : ℝ) (u v : ℝ) (h1 : c ≠ 0) 
  (h2 : u = -c / 8) (h3 : v = -c / 4) : u / v = 1 / 2 :=
by {
  sorry
}

end ratio_of_x_intercepts_l42_42824


namespace multiple_of_four_diff_multiple_of_four_diff_multiple_of_two_l42_42936

variable (a b : ℤ)
variable (h1 : a % 4 = 0) 
variable (h2 : b % 8 = 0)

theorem multiple_of_four (h1 : a % 4 = 0) (h2 : b % 8 = 0) : b % 4 = 0 := by
  sorry

theorem diff_multiple_of_four (h1 : a % 4 = 0) (h2 : b % 8 = 0) : (a - b) % 4 = 0 := by
  sorry

theorem diff_multiple_of_two (h1 : a % 4 = 0) (h2 : b % 8 = 0) : (a - b) % 2 = 0 := by
  sorry

end multiple_of_four_diff_multiple_of_four_diff_multiple_of_two_l42_42936


namespace range_of_a_l42_42638

noncomputable def f (x : ℝ) : ℝ := (2^x - 2^(-x)) / 2
noncomputable def g (x : ℝ) : ℝ := (2^x + 2^(-x)) / 2

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ set.Icc 1 2, a * f x + g (2 * x) ≥ 0) → 
  a ≥ -17 / 6 :=
sorry

end range_of_a_l42_42638


namespace lcm_16_24_l42_42537

/-
  Prove that the least common multiple (LCM) of 16 and 24 is 48.
-/
theorem lcm_16_24 : Nat.lcm 16 24 = 48 :=
by
  sorry

end lcm_16_24_l42_42537


namespace andrew_cookies_per_day_l42_42331

/-- Number of days in May --/
def days_in_may : ℤ := 31

/-- Cost per cookie in dollars --/
def cost_per_cookie : ℤ := 15

/-- Total amount spent by Andrew on cookies in dollars --/
def total_amount_spent : ℤ := 1395

/-- Total number of cookies purchased by Andrew --/
def total_cookies : ℤ := total_amount_spent / cost_per_cookie

/-- Number of cookies purchased per day --/
def cookies_per_day : ℤ := total_cookies / days_in_may

theorem andrew_cookies_per_day : cookies_per_day = 3 := by
  sorry

end andrew_cookies_per_day_l42_42331


namespace original_flow_rate_l42_42308

theorem original_flow_rate (x : ℝ) (h : 2 = 0.6 * x - 1) : x = 5 :=
by
  sorry

end original_flow_rate_l42_42308


namespace bob_cleaning_time_l42_42463

-- Define the conditions
def timeAlice : ℕ := 30
def fractionBob : ℚ := 1 / 3

-- Define the proof problem
theorem bob_cleaning_time : (fractionBob * timeAlice : ℚ) = 10 := by
  sorry

end bob_cleaning_time_l42_42463


namespace sin_pi_plus_alpha_l42_42197

/-- Given that \(\sin \left(\frac{\pi}{2}+\alpha \right) = \frac{3}{5}\)
    and \(\alpha \in (0, \frac{\pi}{2})\),
    prove that \(\sin(\pi + \alpha) = -\frac{4}{5}\). -/
theorem sin_pi_plus_alpha (α : ℝ) (h1 : Real.sin (Real.pi / 2 + α) = 3 / 5)
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin (Real.pi + α) = -4 / 5 := 
  sorry

end sin_pi_plus_alpha_l42_42197


namespace batsman_average_after_17th_inning_l42_42287

theorem batsman_average_after_17th_inning (A : ℝ) :
  let total_runs_after_17_innings := 16 * A + 87
  let new_average := total_runs_after_17_innings / 17
  new_average = A + 3 → 
  (A + 3) = 39 :=
by
  sorry

end batsman_average_after_17th_inning_l42_42287


namespace problem1_problem2_l42_42022

-- Problem 1
theorem problem1 (f : ℝ → ℝ) (x : ℝ) (h : ∀ x, f x = abs (x - 1)) :
  f x ≥ (1/2) * (x + 1) ↔ (x ≤ 1/3) ∨ (x ≥ 3) :=
sorry

-- Problem 2
theorem problem2 (g : ℝ → ℝ) (A : Set ℝ) (a : ℝ) 
  (h1 : ∀ x, g x = abs (x - a) - abs (x - 2))
  (h2 : A ⊆ Set.Icc (-1 : ℝ) 3) :
  (1 ≤ a ∧ a < 2) ∨ (2 ≤ a ∧ a ≤ 3) :=
sorry

end problem1_problem2_l42_42022


namespace find_n_given_combination_l42_42874

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem find_n_given_combination : ∃ n : ℕ, binomial_coefficient (n+1) 2 = 21 ↔ n = 6 := by
  sorry

end find_n_given_combination_l42_42874


namespace equal_parts_count_l42_42712

def scale_length_in_inches : ℕ := (7 * 12) + 6
def part_length_in_inches : ℕ := 18
def number_of_parts (total_length part_length : ℕ) : ℕ := total_length / part_length

theorem equal_parts_count :
  number_of_parts scale_length_in_inches part_length_in_inches = 5 :=
by
  sorry

end equal_parts_count_l42_42712


namespace number_of_bunnies_l42_42485

theorem number_of_bunnies (total_pets : ℕ) (dogs_percentage : ℚ) (cats_percentage : ℚ) (rest_are_bunnies : total_pets = 36 ∧ dogs_percentage = 25 / 100 ∧ cats_percentage = 50 / 100) :
  let dogs := dogs_percentage * total_pets;
      cats := cats_percentage * total_pets;
      bunnies := total_pets - (dogs + cats)
  in bunnies = 9 :=
by
  sorry

end number_of_bunnies_l42_42485


namespace max_difference_value_l42_42195

noncomputable def max_difference (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9) : ℝ :=
  x - y

theorem max_difference_value : ∀ (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9),
  max_difference x y h ≤ 1 + 3 * real.sqrt 2 :=
by
  sorry

end max_difference_value_l42_42195


namespace prize_winners_l42_42218

theorem prize_winners (total_people : ℕ) (percent_envelope : ℝ) (percent_win : ℝ) 
  (h_total : total_people = 100) (h_percent_envelope : percent_envelope = 0.40) 
  (h_percent_win : percent_win = 0.20) : 
  (percent_win * (percent_envelope * total_people)) = 8 := by
  sorry

end prize_winners_l42_42218


namespace ratio_is_one_third_l42_42954

-- Definitions based on given conditions
def total_students : ℕ := 90
def initial_cafeteria_students : ℕ := (2 * total_students) / 3
def initial_outside_students : ℕ := total_students - initial_cafeteria_students
def moved_cafeteria_to_outside : ℕ := 3
def final_cafeteria_students : ℕ := 67
def students_ran_inside : ℕ := final_cafeteria_students - (initial_cafeteria_students - moved_cafeteria_to_outside)

-- Ratio calculation as a proof statement
def ratio_ran_inside_to_outside : ℚ := students_ran_inside / initial_outside_students

-- Proof that the ratio is 1/3
theorem ratio_is_one_third : ratio_ran_inside_to_outside = 1 / 3 :=
by sorry -- Proof omitted

end ratio_is_one_third_l42_42954


namespace Jill_arrives_9_minutes_later_l42_42623

theorem Jill_arrives_9_minutes_later
  (distance : ℝ)
  (Jack_speed : ℝ)
  (Jill_speed : ℝ)
  (h1 : distance = 1)
  (h2 : Jack_speed = 10)
  (h3 : Jill_speed = 4) :
  ((distance / Jill_speed) - (distance / Jack_speed)) * 60 = 9 := by
  -- Placeholder for the proof
  sorry

end Jill_arrives_9_minutes_later_l42_42623


namespace code_transformation_l42_42529

def old_to_new_encoding (s : String) : String := sorry

theorem code_transformation :
  old_to_new_encoding "011011010011" = "211221121" := sorry

end code_transformation_l42_42529


namespace sleepySquirrelNutsPerDay_l42_42803

def twoBusySquirrelsNutsPerDay : ℕ := 2 * 30
def totalDays : ℕ := 40
def totalNuts : ℕ := 3200

theorem sleepySquirrelNutsPerDay 
  (s  : ℕ) 
  (h₁ : 2 * 30 * totalDays + s * totalDays = totalNuts) 
  : s = 20 := 
  sorry

end sleepySquirrelNutsPerDay_l42_42803


namespace ratio_ad_bc_l42_42896

theorem ratio_ad_bc (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 5 * c) (h3 : c = 3 * d) : 
  (a * d) / (b * c) = 4 / 3 := 
by 
  sorry

end ratio_ad_bc_l42_42896


namespace math_problem_l42_42239

theorem math_problem (a b : ℕ) (x y : ℚ) (h1 : a = 10) (h2 : b = 11) (h3 : x = 1.11) (h4 : y = 1.01) :
  ∃ k : ℕ, k * y = 2.02 ∧ (a * x + b * y - k * y = 20.19) :=
by {
  sorry
}

end math_problem_l42_42239


namespace number_of_balls_sold_l42_42491

-- Let n be the number of balls sold
variable (n : ℕ)

-- The given conditions
def selling_price : ℕ := 720
def cost_price_per_ball : ℕ := 60
def loss := 5 * cost_price_per_ball

-- Prove that if the selling price of 'n' balls is Rs. 720 and 
-- the loss is equal to the cost price of 5 balls, then the 
-- number of balls sold (n) is 17.
theorem number_of_balls_sold (h1 : selling_price = 720) 
                             (h2 : cost_price_per_ball = 60) 
                             (h3 : loss = 5 * cost_price_per_ball) 
                             (hsale : n * cost_price_per_ball - selling_price = loss) : 
  n = 17 := 
by
  sorry

end number_of_balls_sold_l42_42491


namespace janet_total_action_figures_l42_42626

/-- Janet owns 10 action figures, sells 6, gets 4 more in better condition,
and then receives twice her current collection from her brother.
We need to prove she ends up with 24 action figures. -/
theorem janet_total_action_figures :
  let initial := 10 in
  let after_selling := initial - 6 in
  let after_acquiring_better := after_selling + 4 in
  let from_brother := 2 * after_acquiring_better in
  after_acquiring_better + from_brother = 24 :=
by
  -- Proof would go here
  sorry

end janet_total_action_figures_l42_42626


namespace new_encoded_message_is_correct_l42_42532

def oldEncodedMessage : String := "011011010011"
def newEncodedMessage : String := "211221121"

def decodeOldEncoding (s : String) : String := 
  -- Function to decode the old encoded message to "ABCBA"
  if s = "011011010011" then "ABCBA" else "unknown"

def encodeNewEncoding (s : String) : String :=
  -- Function to encode "ABCBA" to "211221121"
  s.replace "A" "21".replace "B" "122".replace "C" "1"

theorem new_encoded_message_is_correct : 
  encodeNewEncoding (decodeOldEncoding oldEncodedMessage) = newEncodedMessage := 
by sorry

end new_encoded_message_is_correct_l42_42532


namespace misha_students_count_l42_42236

theorem misha_students_count :
  (∀ n : ℕ, n = 60 → (exists better worse : ℕ, better = n - 1 ∧  worse = n - 1)) →
  (∀ n : ℕ, n = 60 → (better + worse + 1 = 119)) :=
by
  sorry

end misha_students_count_l42_42236


namespace part1_part2_l42_42365

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l42_42365


namespace triangle_area_qin_jiushao_l42_42937

theorem triangle_area_qin_jiushao (a b c : ℝ) (h1: a = 2) (h2: b = 3) (h3: c = Real.sqrt 13) :
  Real.sqrt ((1 / 4) * (a^2 * b^2 - (1 / 4) * (a^2 + b^2 - c^2)^2)) = 3 :=
by
  -- Hypotheses
  rw [h1, h2, h3]
  sorry

end triangle_area_qin_jiushao_l42_42937


namespace density_of_second_part_l42_42988

theorem density_of_second_part (ρ₁ : ℝ) (V₁ V : ℝ) (m₁ m : ℝ) (h₁ : ρ₁ = 2700) (h₂ : V₁ = 0.25 * V) (h₃ : m₁ = 0.4 * m) :
  (0.6 * m) / (0.75 * V) = 2160 :=
by
  --- Proof omitted
  sorry

end density_of_second_part_l42_42988


namespace total_cars_l42_42476

-- Definitions of the conditions
def cathy_cars : Nat := 5

def carol_cars : Nat := 2 * cathy_cars

def susan_cars : Nat := carol_cars - 2

def lindsey_cars : Nat := cathy_cars + 4

-- The theorem statement (problem)
theorem total_cars : cathy_cars + carol_cars + susan_cars + lindsey_cars = 32 :=
by
  -- sorry is added to skip the proof
  sorry

end total_cars_l42_42476


namespace full_tank_capacity_l42_42718

theorem full_tank_capacity (speed : ℝ) (gas_usage_per_mile : ℝ) (time : ℝ) (gas_used_fraction : ℝ) (distance_per_tank : ℝ) (gallons_used : ℝ)
  (h1 : speed = 50)
  (h2 : gas_usage_per_mile = 1 / 30)
  (h3 : time = 5)
  (h4 : gas_used_fraction = 0.8333333333333334)
  (h5 : distance_per_tank = speed * time)
  (h6 : gallons_used = distance_per_tank * gas_usage_per_mile)
  (h7 : gallons_used = 0.8333333333333334 * 10) :
  distance_per_tank / 30 / 0.8333333333333334 = 10 :=
by sorry

end full_tank_capacity_l42_42718


namespace eccentricity_of_ellipse_l42_42740

noncomputable def ellipse_eccentricity {a b : ℝ} (h : a > b > 0) (P F1 F2 : ℝ × ℝ)
  (h1 : P ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1})
  (angle_F1PF2 : ∠ F1 P F2 = 120) (dist_condition : dist P F1 = 3 * dist P F2) : Real :=
  let c := sqrt ((13 / 4^2) * (dist P F2)^2) / 2 in
  c / (2 * dist P F2 / 4)

theorem eccentricity_of_ellipse {a b : ℝ} (h : a > b > 0) (P F1 F2 : ℝ × ℝ)
  (h1 : P ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1})
  (angle_F1PF2 : ∠ F1 P F2 = 120) (dist_condition : dist P F1 = 3 * dist P F2) :
  ellipse_eccentricity h P F1 F2 h1 angle_F1PF2 dist_condition = sqrt 13 / 4 := sorry

end eccentricity_of_ellipse_l42_42740


namespace speed_ratio_l42_42986

noncomputable def k_value {u v x y : ℝ} (h_uv : u > 0) (h_v : v > 0) (h_x : x > 0) (h_y : y > 0) 
  (h_ratio : u / v = ((x + y) / (u - v)) / ((x + y) / (u + v))) : ℝ :=
  1 + Real.sqrt 2

theorem speed_ratio (u v x y : ℝ) (h_uv : u > 0) (h_v : v > 0) (h_x : x > 0) (h_y : y > 0) 
  (h_ratio : u / v = ((x + y) / (u - v)) / ((x + y) / (u + v))) : 
  u / v = k_value h_uv h_v h_x h_y h_ratio :=
sorry

end speed_ratio_l42_42986


namespace percent_of_1600_l42_42556

theorem percent_of_1600 (x : ℝ) (h1 : 0.25 * 1600 = 400) (h2 : x / 100 * 400 = 20) : x = 5 :=
sorry

end percent_of_1600_l42_42556


namespace no_real_solution_for_x_l42_42344

theorem no_real_solution_for_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 8) (h2 : y + 1 / x = 7 / 20) : false :=
by sorry

end no_real_solution_for_x_l42_42344


namespace log_inequality_l42_42150

theorem log_inequality (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ 1) (h4 : y ≠ 1) :
    (Real.log y / Real.log x + Real.log x / Real.log y > 2) →
    (x ≠ y ∧ ((x > 1 ∧ y > 1) ∨ (x < 1 ∧ y < 1))) :=
by
    sorry

end log_inequality_l42_42150


namespace ellas_coins_worth_l42_42010

theorem ellas_coins_worth :
  ∀ (n d : ℕ), n + d = 18 → n = d + 2 → 5 * n + 10 * d = 130 := by
  intros n d h1 h2
  sorry

end ellas_coins_worth_l42_42010


namespace part1_part2_l42_42409

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) :=
by
  sorry

end part1_part2_l42_42409


namespace determine_B_l42_42607

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (h1 : (A ∪ B)ᶜ = {1})
variable (h2 : A ∩ Bᶜ = {3})

theorem determine_B : B = {2, 4, 5} :=
by
  sorry

end determine_B_l42_42607


namespace trig_equalities_l42_42644

theorem trig_equalities (α β γ : ℝ) (h1 : Real.cos α = Real.tan β) (h2 : Real.cos β = Real.tan γ) (h3 : Real.cos γ = Real.tan α) :
  (Real.sin α)^2 = (Real.sin β)^2 ∧ (Real.sin β)^2 = (Real.sin γ)^2 ∧ (Real.sin γ)^2 = 4 * (Real.sin (Real.pi / 10))^2 ∧ 
  (Real.cos α)^4 = (Real.cos β)^4 ∧ (Real.cos β)^4 = (Real.cos γ)^4 :=
begin
  sorry
end

end trig_equalities_l42_42644


namespace maria_carrots_l42_42235

theorem maria_carrots :
  ∀ (picked initially thrownOut moreCarrots totalLeft : ℕ),
    initially = 48 →
    thrownOut = 11 →
    totalLeft = 52 →
    moreCarrots = totalLeft - (initially - thrownOut) →
    moreCarrots = 15 :=
by
  intros
  sorry

end maria_carrots_l42_42235


namespace smallest_possible_b_l42_42257

theorem smallest_possible_b 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a - b = 8) 
  (h4 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  b = 4 := 
sorry

end smallest_possible_b_l42_42257


namespace percent_first_shift_participating_l42_42731

variable (total_employees_in_company : ℕ)
variable (first_shift_employees : ℕ)
variable (second_shift_employees : ℕ)
variable (third_shift_employees : ℕ)
variable (second_shift_percent_participating : ℚ)
variable (third_shift_percent_participating : ℚ)
variable (overall_percent_participating : ℚ)
variable (first_shift_percent_participating : ℚ)

theorem percent_first_shift_participating :
  total_employees_in_company = 150 →
  first_shift_employees = 60 →
  second_shift_employees = 50 →
  third_shift_employees = 40 →
  second_shift_percent_participating = 0.40 →
  third_shift_percent_participating = 0.10 →
  overall_percent_participating = 0.24 →
  first_shift_percent_participating = (12 / 60) →
  first_shift_percent_participating = 0.20 := 
by 
  intros t_e f_s_e s_s_e t_s_e s_s_p_p t_s_p_p o_p_p f_s_p_p
  -- Sorry, here would be the place for the actual proof
  sorry

end percent_first_shift_participating_l42_42731


namespace max_k_value_l42_42325

theorem max_k_value :
  ∀ (a b c : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = a * b + b * c + c * a →
  (a + b + c) * (1 / (a + b) + 1 / (b + c) + 1 / (c + a) - 1) ≥ 1 :=
by
  intros a b c ha hb hc habc_eq
  sorry

end max_k_value_l42_42325


namespace total_eggs_l42_42272

theorem total_eggs (students : ℕ) (eggs_per_student : ℕ) (h1 : students = 7) (h2 : eggs_per_student = 8) :
  students * eggs_per_student = 56 :=
by
  sorry

end total_eggs_l42_42272


namespace brenda_num_cookies_per_box_l42_42000

def numCookiesPerBox (trays : ℕ) (cookiesPerTray : ℕ) (costPerBox : ℚ) (totalSpent : ℚ) : ℚ :=
  let totalCookies := trays * cookiesPerTray
  let numBoxes := totalSpent / costPerBox
  totalCookies / numBoxes

theorem brenda_num_cookies_per_box :
  numCookiesPerBox 3 80 3.5 14 = 60 := by
  sorry

end brenda_num_cookies_per_box_l42_42000


namespace lyle_payment_l42_42706

def pen_cost : ℝ := 1.50

def notebook_cost : ℝ := 3 * pen_cost

def cost_for_4_notebooks : ℝ := 4 * notebook_cost

theorem lyle_payment : cost_for_4_notebooks = 18.00 :=
by
  sorry

end lyle_payment_l42_42706


namespace part1_solution_set_part2_range_l42_42351

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l42_42351


namespace zucchini_pounds_l42_42246

theorem zucchini_pounds :
  let eggplants_pounds := 5
  let eggplants_cost_per_pound := 2.00
  let tomatoes_pounds := 4
  let tomatoes_cost_per_pound := 3.50
  let onions_pounds := 3
  let onions_cost_per_pound := 1.00
  let basil_pounds := 1
  let basil_cost_per_half_pound := 2.50
  let quarts := 4
  let cost_per_quart := 10.00
  let total_cost := quarts * cost_per_quart
  let cost_of_eggplants := eggplants_pounds * eggplants_cost_per_pound
  let cost_of_tomatoes := tomatoes_pounds * tomatoes_cost_per_pound
  let cost_of_onions := onions_pounds * onions_cost_per_pound
  let cost_of_basil := basil_pounds * (basil_cost_per_half_pound * 2)
  let other_ingredients_cost := cost_of_eggplants + cost_of_tomatoes + cost_of_onions + cost_of_basil
  let cost_of_zucchini := total_cost - other_ingredients_cost
  let zucchini_cost_per_pound := 2.00
  let pounds_of_zucchini := cost_of_zucchini / zucchini_cost_per_pound
  pounds_of_zucchini = 4 :=
by
  sorry

end zucchini_pounds_l42_42246


namespace range_for_a_l42_42600

theorem range_for_a (a : ℝ) : 
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by
  sorry

end range_for_a_l42_42600


namespace value_of_b_l42_42966

theorem value_of_b : (15^2 * 9^2 * 356 = 6489300) :=
by 
  sorry

end value_of_b_l42_42966


namespace girls_with_rulers_l42_42452

theorem girls_with_rulers 
  (total_students : ℕ) (students_with_rulers : ℕ) (boys_with_set_squares : ℕ) 
  (total_girls : ℕ) (student_count : total_students = 50) 
  (ruler_count : students_with_rulers = 28) 
  (boys_with_set_squares_count : boys_with_set_squares = 14) 
  (girl_count : total_girls = 31) 
  : total_girls - (total_students - students_with_rulers - boys_with_set_squares) = 23 := 
by
  sorry

end girls_with_rulers_l42_42452


namespace evaluate_expression_l42_42540

theorem evaluate_expression : ((2^2 - 2) - (3^2 - 3) + (4^2 - 4) = 8) :=
sorry

end evaluate_expression_l42_42540


namespace dice_probability_l42_42542

noncomputable def probability_same_face_in_single_roll : ℝ :=
  (1 / 6)^10

noncomputable def probability_not_all_same_face_in_single_roll : ℝ :=
  1 - probability_same_face_in_single_roll

noncomputable def probability_not_all_same_face_in_five_rolls : ℝ :=
  probability_not_all_same_face_in_single_roll^5

noncomputable def probability_at_least_one_all_same_face : ℝ :=
  1 - probability_not_all_same_face_in_five_rolls

theorem dice_probability :
  probability_at_least_one_all_same_face = 1 - (1 - (1 / 6)^10)^5 :=
sorry

end dice_probability_l42_42542


namespace part1_part2_l42_42372

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l42_42372


namespace part1_part2_l42_42362

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l42_42362


namespace igor_reach_top_time_l42_42653

-- Define the conditions
def cabins_numbered_consecutively := (1, 99)
def igor_initial_cabin := 42
def first_aligned_cabin := 13
def second_aligned_cabin := 12
def alignment_time := 15
def total_cabins := 99
def expected_time := 17 * 60 + 15

-- State the problem as a theorem
theorem igor_reach_top_time :
  ∃ t, t = expected_time ∧
  -- Assume the cabins are numbered consecutively
  cabins_numbered_consecutively = (1, total_cabins) ∧
  -- Igor starts in cabin #42
  igor_initial_cabin = 42 ∧
  -- Cabin #42 first aligns with cabin #13, then aligns with cabin #12, 15 seconds later
  first_aligned_cabin = 13 ∧
  second_aligned_cabin = 12 ∧
  alignment_time = 15 :=
sorry

end igor_reach_top_time_l42_42653


namespace scientific_notation_of_114_trillion_l42_42460

theorem scientific_notation_of_114_trillion :
  (114 : ℝ) * 10^12 = (1.14 : ℝ) * 10^14 :=
by
  sorry

end scientific_notation_of_114_trillion_l42_42460


namespace rashmi_speed_second_day_l42_42079

noncomputable def rashmi_speed (distance speed1 time_late time_early : ℝ) : ℝ :=
  let time1 := distance / speed1
  let on_time := time1 - time_late / 60
  let time2 := on_time - time_early / 60
  distance / time2

theorem rashmi_speed_second_day :
  rashmi_speed 9.999999999999993 5 10 10 = 6 := by
  sorry

end rashmi_speed_second_day_l42_42079


namespace units_digit_expression_l42_42868

theorem units_digit_expression: 
  (8 * 19 * 1981 + 6^3 - 2^5) % 10 = 6 := 
by
  sorry

end units_digit_expression_l42_42868


namespace part1_solution_set_part2_range_a_l42_42386

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l42_42386


namespace find_sum_of_x_and_y_l42_42474

theorem find_sum_of_x_and_y (x y : ℝ) 
  (h1 : (x-1)^3 + 1997*(x-1) = -1)
  (h2 : (y-1)^3 + 1997*(y-1) = 1) :
  x + y = 2 :=
sorry

end find_sum_of_x_and_y_l42_42474


namespace remainder_of_3_pow_800_mod_17_l42_42282

theorem remainder_of_3_pow_800_mod_17 :
    (3 ^ 800) % 17 = 1 :=
by
    sorry

end remainder_of_3_pow_800_mod_17_l42_42282


namespace unique_function_f_l42_42863

theorem unique_function_f (f : ℝ → ℝ)
    (h1 : ∀ x : ℝ, f x = -f (-x))
    (h2 : ∀ x : ℝ, f (x + 1) = f x + 1)
    (h3 : ∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / x^2 * f x) :
    ∀ x : ℝ, f x = x := 
sorry

end unique_function_f_l42_42863


namespace update_year_l42_42124

def a (n : ℕ) : ℕ :=
  if n ≤ 7 then 2 * n + 2 else 16 * (5 / 4) ^ (n - 7)

noncomputable def S (n : ℕ) : ℕ :=
  if n ≤ 7 then n^2 + 3 * n else 80 * ((5 / 4) ^ (n - 7)) - 10

noncomputable def avg_maintenance_cost (n : ℕ) : ℚ :=
  (S n : ℚ) / n

theorem update_year (n : ℕ) (h : avg_maintenance_cost n > 12) : n = 9 :=
  by
  sorry

end update_year_l42_42124


namespace cos_seven_theta_l42_42044

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -953 / 1024 :=
by sorry

end cos_seven_theta_l42_42044


namespace luncheon_cost_l42_42654

theorem luncheon_cost (s c p : ℝ)
  (h1 : 2 * s + 5 * c + p = 3.00)
  (h2 : 5 * s + 8 * c + p = 5.40)
  (h3 : 3 * s + 4 * c + p = 3.60) :
  2 * s + 2 * c + p = 2.60 :=
sorry

end luncheon_cost_l42_42654


namespace remainder_when_2n_divided_by_4_l42_42829

theorem remainder_when_2n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (2 * n) % 4 = 2 :=
by
  sorry

end remainder_when_2n_divided_by_4_l42_42829


namespace find_a_l42_42069

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 2

theorem find_a (a : ℝ) (h : (3 * a * (-1 : ℝ)^2) = 3) : a = 1 :=
by
  sorry

end find_a_l42_42069


namespace total_time_to_make_cookies_l42_42333

def time_to_make_batter := 10
def baking_time := 15
def cooling_time := 15
def white_icing_time := 30
def chocolate_icing_time := 30

theorem total_time_to_make_cookies : 
  time_to_make_batter + baking_time + cooling_time + white_icing_time + chocolate_icing_time = 100 := 
by
  sorry

end total_time_to_make_cookies_l42_42333


namespace part1_solution_set_part2_range_of_a_l42_42415

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l42_42415


namespace no_integer_roots_l42_42635

theorem no_integer_roots (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) : 
  ¬ ∃ x : ℤ, a * x^2 + b * x + c = 0 :=
by {
  sorry
}

end no_integer_roots_l42_42635


namespace simplify_expression_l42_42082

theorem simplify_expression :
  (2^8 + 4^5) * (2^3 - (-2)^3)^2 = 327680 := by
  sorry

end simplify_expression_l42_42082


namespace vertical_line_intersect_parabola_ex1_l42_42984

theorem vertical_line_intersect_parabola_ex1 (m : ℝ) (h : ∀ y : ℝ, (-4 * y^2 + 2*y + 3 = m) → false) :
  m = 13 / 4 :=
sorry

end vertical_line_intersect_parabola_ex1_l42_42984


namespace hypotenuse_length_l42_42217

theorem hypotenuse_length (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 1450) (h2 : c^2 = a^2 + b^2) : 
  c = Real.sqrt 725 :=
by
  sorry

end hypotenuse_length_l42_42217


namespace evaluate_gg2_l42_42800

noncomputable def g (x : ℚ) : ℚ := 1 / (x^2) + (x^2) / (1 + x^2)

theorem evaluate_gg2 : g (g 2) = 530881 / 370881 :=
by
  sorry

end evaluate_gg2_l42_42800


namespace distinct_roots_iff_l42_42151

def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 + |x1| = 2 * Real.sqrt (3 + 2*a*x1 - 4*a)) ∧
                       (x2 + |x2| = 2 * Real.sqrt (3 + 2*a*x2 - 4*a))

theorem distinct_roots_iff (a : ℝ) :
  has_two_distinct_roots a ↔ (a ∈ Set.Ioo 0 (3 / 4 : ℝ) ∨ 3 < a) :=
sorry

end distinct_roots_iff_l42_42151


namespace part1_solution_set_part2_range_of_a_l42_42382

-- Part 1
theorem part1_solution_set (x : ℝ) : (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x + 3| > -a) ↔ (a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_of_a_l42_42382


namespace find_S2017_l42_42343

-- Setting up the given conditions and sequences
def a1 : ℤ := -2014
def S (n : ℕ) : ℚ := n * a1 + (n * (n - 1) / 2) * 2 -- Using the provided sum formula

theorem find_S2017
  (h1 : a1 = -2014)
  (h2 : (S 2014) / 2014 - (S 2008) / 2008 = 6) :
  S 2017 = 4034 := 
sorry

end find_S2017_l42_42343


namespace parallelepiped_analogy_l42_42283

-- Define plane figures and the concept of analogy for a parallelepiped 
-- (specifically here as a parallelogram) in space
inductive PlaneFigure where
  | triangle
  | parallelogram
  | trapezoid
  | rectangle

open PlaneFigure

/-- 
  Given the properties and definitions of a parallelepiped and plane figures,
  we want to show that the appropriate analogy for a parallelepiped in space
  is a parallelogram.
-/
theorem parallelepiped_analogy : 
  (analogy : PlaneFigure) = parallelogram :=
sorry

end parallelepiped_analogy_l42_42283


namespace prob_neither_alive_l42_42662

/-- Define the probability that a man will be alive for 10 more years -/
def prob_man_alive : ℚ := 1 / 4

/-- Define the probability that a wife will be alive for 10 more years -/
def prob_wife_alive : ℚ := 1 / 3

/-- Prove that the probability that neither the man nor his wife will be alive for 10 more years is 1/2 -/
theorem prob_neither_alive (p_man_alive p_wife_alive : ℚ)
    (h1 : p_man_alive = prob_man_alive) (h2 : p_wife_alive = prob_wife_alive) :
    (1 - p_man_alive) * (1 - p_wife_alive) = 1 / 2 :=
by
  sorry

end prob_neither_alive_l42_42662


namespace sum_inf_evaluation_eq_9_by_80_l42_42582

noncomputable def infinite_sum_evaluation : ℝ := ∑' n, (2 * n) / (n^4 + 16)

theorem sum_inf_evaluation_eq_9_by_80 :
  infinite_sum_evaluation = 9 / 80 :=
by
  sorry

end sum_inf_evaluation_eq_9_by_80_l42_42582


namespace erik_ate_more_pie_l42_42571

theorem erik_ate_more_pie :
  let erik_pies := 0.67
  let frank_pies := 0.33
  erik_pies - frank_pies = 0.34 :=
by
  sorry

end erik_ate_more_pie_l42_42571


namespace percent_unionized_men_is_70_l42_42618

open Real

def total_employees : ℝ := 100
def percent_men : ℝ := 0.5
def percent_unionized : ℝ := 0.6
def percent_women_nonunion : ℝ := 0.8
def percent_men_nonunion : ℝ := 0.2

def num_men := total_employees * percent_men
def num_unionized := total_employees * percent_unionized
def num_nonunion := total_employees - num_unionized
def num_men_nonunion := num_nonunion * percent_men_nonunion
def num_men_unionized := num_men - num_men_nonunion

theorem percent_unionized_men_is_70 :
  (num_men_unionized / num_unionized) * 100 = 70 := by
  sorry

end percent_unionized_men_is_70_l42_42618


namespace num_square_free_odds_l42_42137

noncomputable def is_square_free (m : ℕ) : Prop :=
  ∀ n : ℕ, n^2 ∣ m → n = 1

noncomputable def count_square_free_odds : ℕ :=
  (199 - 1) / 2 - (11 + 4 + 2 + 1 + 1 + 1)

theorem num_square_free_odds : count_square_free_odds = 79 := by
  sorry

end num_square_free_odds_l42_42137


namespace no_such_natural_number_exists_l42_42007

theorem no_such_natural_number_exists :
  ¬ ∃ (n s : ℕ), n = 2014 * s + 2014 ∧ n % s = 2014 ∧ (n / s) = 2014 :=
by
  sorry

end no_such_natural_number_exists_l42_42007


namespace find_x_l42_42975

theorem find_x (x : ℝ) (h : 0.40 * x = (1/3) * x + 110) : x = 1650 :=
sorry

end find_x_l42_42975


namespace alice_needs_136_life_vests_l42_42568

-- Definitions from the problem statement
def num_classes : ℕ := 4
def students_per_class : ℕ := 40
def instructors_per_class : ℕ := 10
def life_vest_probability : ℝ := 0.40

-- Calculate the total number of people
def total_people := num_classes * (students_per_class + instructors_per_class)

-- Calculate the expected number of students with life vests
def students_with_life_vests := (students_per_class : ℝ) * life_vest_probability
def total_students_with_life_vests := num_classes * students_with_life_vests

-- Calculate the number of life vests needed
def life_vests_needed := total_people - total_students_with_life_vests

-- Proof statement (missing the actual proof)
theorem alice_needs_136_life_vests : life_vests_needed = 136 := by
  sorry

end alice_needs_136_life_vests_l42_42568


namespace part1_solution_set_part2_range_l42_42352

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l42_42352


namespace total_onions_l42_42071

-- Define the number of onions grown by each individual
def nancy_onions : ℕ := 2
def dan_onions : ℕ := 9
def mike_onions : ℕ := 4

-- Proposition: The total number of onions grown is 15
theorem total_onions : (nancy_onions + dan_onions + mike_onions) = 15 := 
by sorry

end total_onions_l42_42071


namespace pencils_left_l42_42627

def initial_pencils := 4527
def given_to_dorothy := 1896
def given_to_samuel := 754
def given_to_alina := 307
def total_given := given_to_dorothy + given_to_samuel + given_to_alina
def remaining_pencils := initial_pencils - total_given

theorem pencils_left : remaining_pencils = 1570 := by
  sorry

end pencils_left_l42_42627


namespace sum_of_possible_values_of_x_l42_42756

theorem sum_of_possible_values_of_x (x : ℝ) (h : (x + 3) * (x - 4) = 22) : ∃ (x1 x2 : ℝ), x^2 - x - 34 = 0 ∧ x1 + x2 = 1 :=
by
  sorry

end sum_of_possible_values_of_x_l42_42756


namespace binary_calculation_l42_42146

theorem binary_calculation :
  let b1 := 0b110110
  let b2 := 0b101110
  let b3 := 0b100
  let expected_result := 0b11100011110
  ((b1 * b2) / b3) = expected_result := by
  sorry

end binary_calculation_l42_42146


namespace sum_of_six_digits_is_31_l42_42498

-- Problem constants and definitions
def digits : Set ℕ := {0, 2, 3, 4, 5, 7, 8, 9}

-- Problem conditions expressed as hypotheses
variables (a b c d e f g : ℕ)
variables (h1 : a ∈ digits) (h2 : b ∈ digits) (h3 : c ∈ digits) 
          (h4 : d ∈ digits) (h5 : e ∈ digits) (h6 : f ∈ digits) (h7 : g ∈ digits)
          (h8 : a ≠ b) (h9 : a ≠ c) (h10 : a ≠ d) (h11 : a ≠ e) (h12 : a ≠ f) (h13 : a ≠ g)
          (h14 : b ≠ c) (h15 : b ≠ d) (h16 : b ≠ e) (h17 : b ≠ f) (h18 : b ≠ g)
          (h19 : c ≠ d) (h20 : c ≠ e) (h21 : c ≠ f) (h22 : c ≠ g)
          (h23 : d ≠ e) (h24 : d ≠ f) (h25 : d ≠ g)
          (h26 : e ≠ f) (h27 : e ≠ g) (h28 : f ≠ g)
variable (shared : b = e)
variables (h29 : a + b + c = 24) (h30 : d + e + f + g = 14)

-- Proposition to be proved
theorem sum_of_six_digits_is_31 : a + b + c + d + e + f = 31 :=
by 
  sorry

end sum_of_six_digits_is_31_l42_42498


namespace gas_volume_at_12_l42_42158

variable (VolumeTemperature : ℕ → ℕ) -- a function representing the volume of gas at a given temperature 

axiom condition1 : ∀ t : ℕ, VolumeTemperature (t + 4) = VolumeTemperature t + 5

axiom condition2 : VolumeTemperature 28 = 35

theorem gas_volume_at_12 :
  VolumeTemperature 12 = 15 := 
sorry

end gas_volume_at_12_l42_42158


namespace no_real_roots_of_quadratic_l42_42021

def quadratic (a b c : ℝ) : ℝ × ℝ × ℝ := (a^2, b^2 + a^2 - c^2, b^2)

def discriminant (A B C : ℝ) : ℝ := B^2 - 4 * A * C

theorem no_real_roots_of_quadratic (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : c > 0)
  (h3 : |a - b| < c)
  : (discriminant (a^2) (b^2 + a^2 - c^2) (b^2)) < 0 :=
sorry

end no_real_roots_of_quadratic_l42_42021


namespace negation_universal_proposition_l42_42205

theorem negation_universal_proposition : 
  (¬ ∀ x : ℝ, x^2 - x < 0) = ∃ x : ℝ, x^2 - x ≥ 0 :=
by
  sorry

end negation_universal_proposition_l42_42205


namespace never_prime_except_three_l42_42578

theorem never_prime_except_three (p : ℕ) (hp : Nat.Prime p) :
  p^2 + 8 = 17 ∨ ∃ k, (k ≠ 1 ∧ k ≠ p^2 + 8 ∧ k ∣ (p^2 + 8)) := by
  sorry

end never_prime_except_three_l42_42578


namespace part1_solution_set_part2_range_of_a_l42_42383

-- Part 1
theorem part1_solution_set (x : ℝ) : (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x + 3| > -a) ↔ (a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_of_a_l42_42383


namespace number_is_24point2_l42_42572

noncomputable def certain_number (x : ℝ) : Prop :=
  0.12 * x = 2.904

theorem number_is_24point2 : certain_number 24.2 :=
by
  unfold certain_number
  sorry

end number_is_24point2_l42_42572


namespace boys_more_than_girls_l42_42669

-- Definitions of the conditions
def total_students : ℕ := 100
def boy_ratio : ℕ := 3
def girl_ratio : ℕ := 2

-- Statement of the problem
theorem boys_more_than_girls :
  (total_students * boy_ratio) / (boy_ratio + girl_ratio) - (total_students * girl_ratio) / (boy_ratio + girl_ratio) = 20 :=
by
  sorry

end boys_more_than_girls_l42_42669


namespace trapezium_area_correct_l42_42155

def a : ℚ := 20  -- Length of the first parallel side
def b : ℚ := 18  -- Length of the second parallel side
def h : ℚ := 20  -- Distance (height) between the parallel sides

def trapezium_area (a b h : ℚ) : ℚ :=
  (1/2) * (a + b) * h

theorem trapezium_area_correct : trapezium_area a b h = 380 := 
  by
    sorry  -- Proof goes here

end trapezium_area_correct_l42_42155


namespace find_m_parallel_l42_42443

def vector_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k • v) ∨ v = (k • u)

theorem find_m_parallel (m : ℝ) (a b : ℝ × ℝ) (h_a : a = (-1, 1)) (h_b : b = (3, m)) 
  (h_parallel : vector_parallel a (a.1 + b.1, a.2 + b.2)) : m = -3 := 
by 
  sorry

end find_m_parallel_l42_42443


namespace probability_none_l42_42959

variable {Ω : Type} [ProbabilitySpace Ω]

def PA : ℝ := 0.25
def PB : ℝ := 0.40
def PC : ℝ := 0.30
def PAB : ℝ := 0.15
def PBC : ℝ := 0.12
def PAC : ℝ := 0.10
def PABC : ℝ := 0.05

def PNone : ℝ := 
  1 - (PA + PB + PC - PAB - PBC - PAC + PABC)

theorem probability_none (h : PNone = 0.42) : 
  PA = 0.25 ∧ PB = 0.40 ∧ PC = 0.30 ∧ PAB = 0.15 ∧ PBC = 0.12 ∧ PAC = 0.10 ∧ PABC = 0.05 → 
  PNone = 0.42 :=
by 
  sorry

end probability_none_l42_42959


namespace complex_number_quadrant_l42_42784

open Complex

theorem complex_number_quadrant 
  (a b c d : ℤ) : 
  (a + b * Complex.I) * (c - d * Complex.I) = (a*c + b*d) + (a*d + b*c) * Complex.I → 
  (0 < (a*c + b*d) ∧ 0 < (a*d + b*c)) → 
  True := 
by
  intro h_mul h_coord
  sorry


end complex_number_quadrant_l42_42784


namespace cos_585_eq_neg_sqrt2_div_2_l42_42951

theorem cos_585_eq_neg_sqrt2_div_2 : Real.cos (585 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by sorry

end cos_585_eq_neg_sqrt2_div_2_l42_42951


namespace estimate_larger_than_difference_l42_42673

variable {x y : ℝ}

theorem estimate_larger_than_difference (h1 : x > y) (h2 : y > 0) :
    ⌈x⌉ - ⌊y⌋ > x - y := by
  sorry

end estimate_larger_than_difference_l42_42673


namespace max_difference_value_l42_42196

noncomputable def max_difference (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9) : ℝ :=
  x - y

theorem max_difference_value : ∀ (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9),
  max_difference x y h ≤ 1 + 3 * real.sqrt 2 :=
by
  sorry

end max_difference_value_l42_42196


namespace range_2a_minus_b_and_a_div_b_range_3x_minus_y_l42_42551

-- Proof for finding the range of 2a - b and a / b
theorem range_2a_minus_b_and_a_div_b (a b : ℝ) (h_a : 12 < a ∧ a < 60) (h_b : 15 < b ∧ b < 36) : 
  -12 < 2 * a - b ∧ 2 * a - b < 105 ∧ 1 / 3 < a / b ∧ a / b < 4 :=
by
  sorry

-- Proof for finding the range of 3x - y
theorem range_3x_minus_y (x y : ℝ) (h_xy_diff : -1 / 2 < x - y ∧ x - y < 1 / 2) (h_xy_sum : 0 < x + y ∧ x + y < 1) : 
  -1 < 3 * x - y ∧ 3 * x - y < 2 :=
by
  sorry

end range_2a_minus_b_and_a_div_b_range_3x_minus_y_l42_42551


namespace cos_theta_seven_l42_42031

theorem cos_theta_seven {θ : ℝ} (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -8383 / 98304 :=
by
  sorry

end cos_theta_seven_l42_42031


namespace total_cars_l42_42481

-- Definitions for the conditions
def cathy_cars : ℕ := 5
def lindsey_cars : ℕ := cathy_cars + 4
def carol_cars : ℕ := 2 * cathy_cars
def susan_cars : ℕ := carol_cars - 2

-- Lean theorem statement
theorem total_cars : cathy_cars + lindsey_cars + carol_cars + susan_cars = 32 := by
  sorry

end total_cars_l42_42481


namespace random_event_is_eventA_l42_42111

-- Definitions of conditions
def eventA : Prop := true  -- Tossing a coin and it lands either heads up or tails up is a random event
def eventB : Prop := (∀ (a b : ℝ), (b * a = b * a))  -- The area of a rectangle with sides of length a and b is ab is a certain event
def eventC : Prop := ∃ (defective_items : ℕ), (defective_items / 100 = 10 / 100)  -- Drawing 2 defective items from 100 parts with 10% defective parts is uncertain
def eventD : Prop := false -- Scoring 105 points in a regular 100-point system exam is an impossible event

-- The proof problem statement
theorem random_event_is_eventA : eventA ∧ ¬eventB ∧ ¬eventC ∧ ¬eventD := 
sorry

end random_event_is_eventA_l42_42111


namespace not_chosen_rate_l42_42558

theorem not_chosen_rate (sum : ℝ) (interest_15_percent : ℝ) (extra_interest : ℝ) : 
  sum = 7000 ∧ interest_15_percent = 2100 ∧ extra_interest = 420 →
  ∃ R : ℝ, (sum * 0.15 * 2 = interest_15_percent) ∧ 
           (interest_15_percent - (sum * R / 100 * 2) = extra_interest) ∧ 
           R = 12 := 
by {
  sorry
}

end not_chosen_rate_l42_42558


namespace abs_diff_probability_l42_42645

noncomputable def probability_abs_diff_gt_half : ℝ :=
1/4 * (0 + 1/2) + 1/4 * 1 + 1/16

theorem abs_diff_probability : probability_abs_diff_gt_half = 9/16 := by
  sorry

end abs_diff_probability_l42_42645


namespace total_players_is_59_l42_42619

-- Define the number of players from each sport.
def cricket_players : ℕ := 16
def hockey_players : ℕ := 12
def football_players : ℕ := 18
def softball_players : ℕ := 13

-- Define the total number of players as the sum of the above.
def total_players : ℕ :=
  cricket_players + hockey_players + football_players + softball_players

-- Prove that the total number of players is 59.
theorem total_players_is_59 :
  total_players = 59 :=
by
  unfold total_players
  unfold cricket_players
  unfold hockey_players
  unfold football_players
  unfold softball_players
  sorry

end total_players_is_59_l42_42619


namespace distinct_ordered_pairs_solution_l42_42730

theorem distinct_ordered_pairs_solution :
  (∃ n : ℕ, ∀ x y : ℕ, (x > 0 ∧ y > 0 ∧ x^4 * y^4 - 24 * x^2 * y^2 + 35 = 0) ↔ n = 1) :=
sorry

end distinct_ordered_pairs_solution_l42_42730


namespace age_of_b_l42_42965

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 47) : b = 18 := 
  sorry

end age_of_b_l42_42965


namespace evaluate_polynomial_at_3_l42_42103

def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 - 3*x + 2

theorem evaluate_polynomial_at_3 : f 3 = 2 :=
by
  sorry

end evaluate_polynomial_at_3_l42_42103


namespace freezer_temp_calculation_l42_42770

def refrigerator_temp : ℝ := 4
def freezer_temp (rt : ℝ) (d : ℝ) : ℝ := rt - d

theorem freezer_temp_calculation :
  (freezer_temp refrigerator_temp 22) = -18 :=
by
  sorry

end freezer_temp_calculation_l42_42770


namespace nathan_weeks_l42_42487

-- Define the conditions as per the problem
def hours_per_day_nathan : ℕ := 3
def days_per_week : ℕ := 7
def hours_per_week_nathan : ℕ := hours_per_day_nathan * days_per_week
def hours_per_day_tobias : ℕ := 5
def hours_one_week_tobias : ℕ := hours_per_day_tobias * days_per_week
def total_hours : ℕ := 77

-- The number of weeks Nathan played
def weeks_nathan (w : ℕ) : Prop :=
  hours_per_week_nathan * w + hours_one_week_tobias = total_hours

-- Prove the number of weeks Nathan played is 2
theorem nathan_weeks : ∃ w : ℕ, weeks_nathan w ∧ w = 2 :=
by
  use 2
  sorry

end nathan_weeks_l42_42487


namespace percentage_calculation_l42_42557

theorem percentage_calculation (P : ℕ) (h1 : 0.25 * 16 = 4) 
    (h2 : P / 100 * 40 = 6) : P = 15 :=
by 
    sorry

end percentage_calculation_l42_42557


namespace max_value_of_fraction_l42_42068

theorem max_value_of_fraction (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 4 * x^2 - 3 * x * y + y^2 = z) : 
  ∃ M, M = 1 ∧ ∀ (a b c : ℝ), (a > 0) → (b > 0) → (c > 0) → (4 * a^2 - 3 * a * b + b^2 = c) → (a * b / c ≤ M) :=
by
  sorry

end max_value_of_fraction_l42_42068


namespace preston_high_school_teachers_l42_42242

theorem preston_high_school_teachers 
  (num_students : ℕ)
  (classes_per_student : ℕ)
  (classes_per_teacher : ℕ)
  (students_per_class : ℕ)
  (teachers_per_class : ℕ)
  (H : num_students = 1500)
  (C : classes_per_student = 6)
  (T : classes_per_teacher = 5)
  (S : students_per_class = 30)
  (P : teachers_per_class = 1) : 
  (num_students * classes_per_student / students_per_class / classes_per_teacher = 60) :=
by sorry

end preston_high_school_teachers_l42_42242


namespace ratio_of_cereal_boxes_l42_42516

variable (F : ℕ) (S : ℕ) (T : ℕ) (k : ℚ)

def boxes_cereal : Prop :=
  F = 14 ∧
  F + S + T = 33 ∧
  S = k * (F : ℚ) ∧
  S = T - 5 → 
  S / F = 1 / 2

theorem ratio_of_cereal_boxes (F S T : ℕ) (k : ℚ) : 
  boxes_cereal F S T k :=
by
  sorry

end ratio_of_cereal_boxes_l42_42516


namespace sum_of_fractions_eq_one_l42_42316

variable {a b c d : ℝ} (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0)
          (h_equiv : (a * d + b * c) / (b * d) = (a * c) / (b * d))

theorem sum_of_fractions_eq_one : b / a + d / c = 1 :=
by sorry

end sum_of_fractions_eq_one_l42_42316


namespace diameter_of_circle_with_inscribed_right_triangle_l42_42828

theorem diameter_of_circle_with_inscribed_right_triangle (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (right_triangle : a^2 + b^2 = c^2) : c = 10 :=
by
  subst h1
  subst h2
  simp at right_triangle
  sorry

end diameter_of_circle_with_inscribed_right_triangle_l42_42828


namespace sum_of_roots_of_equation_l42_42764

theorem sum_of_roots_of_equation :
  (∀ x : ℝ, (x + 3) * (x - 4) = 22 → ∃ a b : ℝ, (x - a) * (x - b) = 0 ∧ a + b = 1) :=
by
  sorry

end sum_of_roots_of_equation_l42_42764


namespace find_y_from_eqns_l42_42613

theorem find_y_from_eqns (x y : ℝ) (h1 : x^2 - x + 6 = y + 2) (h2 : x = -5) : y = 34 :=
by {
  sorry
}

end find_y_from_eqns_l42_42613


namespace boys_more_than_girls_l42_42668

-- Definitions of the conditions
def total_students : ℕ := 100
def boy_ratio : ℕ := 3
def girl_ratio : ℕ := 2

-- Statement of the problem
theorem boys_more_than_girls :
  (total_students * boy_ratio) / (boy_ratio + girl_ratio) - (total_students * girl_ratio) / (boy_ratio + girl_ratio) = 20 :=
by
  sorry

end boys_more_than_girls_l42_42668


namespace negation_of_exists_x_lt_0_l42_42507

theorem negation_of_exists_x_lt_0 :
  (¬ ∃ x : ℝ, x + |x| < 0) ↔ (∀ x : ℝ, x + |x| ≥ 0) :=
by {
  sorry
}

end negation_of_exists_x_lt_0_l42_42507


namespace range_of_a_l42_42744

open Function

theorem range_of_a (f : ℝ → ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 0 → f x₁ > f x₂) (a : ℝ) (h_gt : f a > f 2) : a < -2 ∨ a > 2 :=
  sorry

end range_of_a_l42_42744


namespace initial_coloring_books_l42_42565

theorem initial_coloring_books
  (x : ℝ)
  (h1 : x - 20 = 80 / 4) :
  x = 40 :=
by
  sorry

end initial_coloring_books_l42_42565


namespace karen_start_time_late_l42_42911

theorem karen_start_time_late
  (karen_speed : ℝ := 60) -- Karen drives at 60 mph
  (tom_speed : ℝ := 45) -- Tom drives at 45 mph
  (tom_distance : ℝ := 24) -- Tom drives 24 miles before Karen wins
  (karen_lead : ℝ := 4) -- Karen needs to beat Tom by 4 miles
  : (60 * (24 / 45) - 60 * (28 / 60)) * 60 = 4 := by
  sorry

end karen_start_time_late_l42_42911


namespace lindsay_doll_count_l42_42924

theorem lindsay_doll_count :
  let B := 4 in                  -- Number of blonde-haired dolls
  let Br := 4 * B in             -- Number of brown-haired dolls
  let Bl := Br - 2 in            -- Number of black-haired dolls
  Bl + Br - B = 26 :=            -- Prove the combined excess of black and brown over blonde
begin 
  sorry 
end

end lindsay_doll_count_l42_42924


namespace max_x_minus_y_l42_42184

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  ∃ k ∈ ℝ, x - y ≤ k ∧ k = 1 + 3 * real.sqrt 2 :=
sorry

end max_x_minus_y_l42_42184


namespace arithmetic_sequence_angle_l42_42879

-- Define the conditions
variables (A B C a b c : ℝ)
-- The statement assumes that A, B, C form an arithmetic sequence
-- which implies 2B = A + C
-- We need to show that 1/(a + b) + 1/(b + c) = 3/(a + b + c)

theorem arithmetic_sequence_angle
  (h : 2 * B = A + C)
  (cos_rule : b^2 = c^2 + a^2 - 2 * c * a * Real.cos B):
    1 / (a + b) + 1 / (b + c) = 3 / (a + b + c) := sorry

end arithmetic_sequence_angle_l42_42879


namespace smallest_b_value_l42_42254

theorem smallest_b_value (a b : ℕ) (h1 : a - b = 8) (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : b = 3 := sorry

end smallest_b_value_l42_42254


namespace product_of_solutions_l42_42016

theorem product_of_solutions : ∀ y : ℝ, 
    (|y| = 3 * (|y| - 2)) → (y = 3 ∨ y = -3) → (3 * -3 = -9) :=
by
  intro y
  intro h
  intro hsol
  exact (3 * -3 = -9)
  sorry

end product_of_solutions_l42_42016


namespace lcm_of_coprimes_eq_product_l42_42931

theorem lcm_of_coprimes_eq_product (a b c : ℕ) (h_coprime_ab : Nat.gcd a b = 1) (h_coprime_bc : Nat.gcd b c = 1) (h_coprime_ca : Nat.gcd c a = 1) (h_product : a * b * c = 7429) :
  Nat.lcm (Nat.lcm a b) c = 7429 :=
by 
  sorry

end lcm_of_coprimes_eq_product_l42_42931


namespace part1_solution_set_part2_range_of_a_l42_42384

-- Part 1
theorem part1_solution_set (x : ℝ) : (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x + 3| > -a) ↔ (a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_of_a_l42_42384


namespace solution_set_inequality_range_a_inequality_l42_42885

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) - 2

theorem solution_set_inequality (x : ℝ) (a : ℝ) (h : a = 1) :
  f x a + abs (2*x - 3) > 0 ↔ (x < 2 / 3 ∨ 2 < x) := sorry

theorem range_a_inequality (a : ℝ) :
  (∀ x, f x a < abs (x - 3)) ↔ (1 < a ∧ a < 5) := sorry

end solution_set_inequality_range_a_inequality_l42_42885


namespace g_neg_one_l42_42345

variables {F : Type*} [Field F]

def odd_function (f : F → F) := ∀ x, f (-x) = -f x

variables (f : F → F) (g : F → F)

-- Given conditions
lemma given_conditions :
  (∀ x, f (-x) + (-x)^2 = -(f x + x^2)) ∧
  f 1 = 1 ∧
  (∀ x, g x = f x + 2) :=
sorry

-- Prove that g(-1) = -1
theorem g_neg_one :
  g (-1) = -1 :=
sorry

end g_neg_one_l42_42345


namespace chocolate_squares_remaining_l42_42711

theorem chocolate_squares_remaining (m : ℕ) : m * 6 - 21 = 45 :=
by
  sorry

end chocolate_squares_remaining_l42_42711


namespace C_eq_D_iff_n_eq_3_l42_42468

noncomputable def C (n : ℕ) : ℝ :=
  1000 * (1 - (1 / 3^n)) / (1 - 1 / 3)

noncomputable def D (n : ℕ) : ℝ :=
  2700 * (1 - (1 / (-3)^n)) / (1 + 1 / 3)

theorem C_eq_D_iff_n_eq_3 (n : ℕ) (h : 1 ≤ n) : C n = D n ↔ n = 3 :=
by
  unfold C D
  sorry

end C_eq_D_iff_n_eq_3_l42_42468


namespace part1_solution_part2_solution_l42_42396

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l42_42396


namespace min_exponent_binomial_l42_42908

theorem min_exponent_binomial (n : ℕ) (h1 : n > 0)
  (h2 : ∃ r : ℕ, (n.choose r) / (n.choose (r + 1)) = 5 / 7) : n = 11 :=
by {
-- Note: We are merely stating the theorem here according to the instructions,
-- the proof body is omitted and hence the use of 'sorry'.
sorry
}

end min_exponent_binomial_l42_42908


namespace sum_of_roots_of_equation_l42_42766

theorem sum_of_roots_of_equation :
  (∀ x : ℝ, (x + 3) * (x - 4) = 22 → ∃ a b : ℝ, (x - a) * (x - b) = 0 ∧ a + b = 1) :=
by
  sorry

end sum_of_roots_of_equation_l42_42766


namespace cells_after_10_days_l42_42702

theorem cells_after_10_days :
  let a := 4
  let r := 2
  let n := 10 / 2
  let a_n := a * r ^ (n - 1)
  a_n = 64 :=
by
  let a := 4
  let r := 2
  let n := 10 / 2
  let a_n := a * r ^ (n - 1)
  show a_n = 64
  sorry

end cells_after_10_days_l42_42702


namespace men_in_room_l42_42621
noncomputable def numMenInRoom (x : ℕ) : ℕ := 4 * x + 2

theorem men_in_room (x : ℕ) (h_initial_ratio : true) (h_after_events : true) (h_double_women : 2 * (5 * x - 3) = 24) :
  numMenInRoom x = 14 :=
sorry

end men_in_room_l42_42621


namespace dolls_proof_l42_42925

variable (blonde_dolls brown_dolls black_dolls : ℕ)

def given_conditions (blonde_dolls brown_dolls black_dolls : ℕ) : Prop :=
  blonde_dolls = 4 ∧
  brown_dolls = 4 * blonde_dolls ∧
  black_dolls = brown_dolls - 2

def question (blonde_dolls brown_dolls black_dolls : ℕ) : ℕ :=
  (brown_dolls + black_dolls) - blonde_dolls

theorem dolls_proof :
  ∀ (blonde_dolls brown_dolls black_dolls : ℕ),
  given_conditions blonde_dolls brown_dolls black_dolls →
  question blonde_dolls brown_dolls black_dolls = 26 :=
by
  intros blonde_dolls brown_dolls black_dolls h
  simp [given_conditions, question] at *
  split at h
  -- Now you'd need to either finish the proof or fill in the steps taken in the given solution.
  sorry

end dolls_proof_l42_42925


namespace identify_ATM_mistakes_additional_security_measures_l42_42963

-- Define the conditions as Boolean variables representing different mistakes and measures
variables (writing_PIN_on_card : Prop)
variables (using_ATM_despite_difficulty : Prop)
variables (believing_stranger : Prop)
variables (walking_away_without_card : Prop)
variables (use_trustworthy_locations : Prop)
variables (presence_during_transactions : Prop)
variables (enable_SMS_notifications : Prop)
variables (call_bank_for_suspicious_activities : Prop)
variables (be_cautious_of_fake_SMS_alerts : Prop)
variables (store_transaction_receipts : Prop)
variables (shield_PIN : Prop)
variables (use_chipped_cards : Prop)
variables (avoid_high_risk_ATMs : Prop)

-- Prove that the identified mistakes occur given the conditions
theorem identify_ATM_mistakes :
  writing_PIN_on_card ∧ using_ATM_despite_difficulty ∧ 
  believing_stranger ∧ walking_away_without_card := sorry

-- Prove that the additional security measures should be followed
theorem additional_security_measures :
  use_trustworthy_locations ∧ presence_during_transactions ∧ 
  enable_SMS_notifications ∧ call_bank_for_suspicious_activities ∧ 
  be_cautious_of_fake_SMS_alerts ∧ store_transaction_receipts ∧ 
  shield_PIN ∧ use_chipped_cards ∧ avoid_high_risk_ATMs := sorry

end identify_ATM_mistakes_additional_security_measures_l42_42963


namespace perimeter_difference_l42_42149

-- Define the dimensions of the two figures
def width1 : ℕ := 6
def height1 : ℕ := 3
def width2 : ℕ := 6
def height2 : ℕ := 2

-- Define the perimeters of the two figures
def perimeter1 : ℕ := 2 * (width1 + height1)
def perimeter2 : ℕ := 2 * (width2 + height2)

-- Prove the positive difference in perimeters is 2 units
theorem perimeter_difference : (perimeter1 - perimeter2) = 2 := by
  sorry

end perimeter_difference_l42_42149


namespace reading_time_difference_l42_42285

theorem reading_time_difference :
  let xanthia_reading_speed := 100 -- pages per hour
  let molly_reading_speed := 50 -- pages per hour
  let book_pages := 225
  let xanthia_time := book_pages / xanthia_reading_speed
  let molly_time := book_pages / molly_reading_speed
  let difference_in_hours := molly_time - xanthia_time
  let difference_in_minutes := difference_in_hours * 60
  difference_in_minutes = 135 := by
  sorry

end reading_time_difference_l42_42285


namespace value_of_r6_plus_s6_l42_42067

theorem value_of_r6_plus_s6 :
  ∀ r s : ℝ, (r^2 - 2 * r + Real.sqrt 2 = 0) ∧ (s^2 - 2 * s + Real.sqrt 2 = 0) →
  (r^6 + s^6 = 904 - 640 * Real.sqrt 2) :=
by
  intros r s h
  -- Proof skipped
  sorry

end value_of_r6_plus_s6_l42_42067


namespace arithmetic_series_sum_l42_42264

theorem arithmetic_series_sum (k : ℤ) : 
  let a₁ := k^2 + k + 1 
  let n := 2 * k + 3 
  let d := 1 
  let aₙ := a₁ + (n - 1) * d 
  let S_n := n / 2 * (a₁ + aₙ)
  S_n = 2 * k^3 + 7 * k^2 + 10 * k + 6 := 
by {
  sorry
}

end arithmetic_series_sum_l42_42264


namespace power_of_five_trailing_zeros_l42_42584

theorem power_of_five_trailing_zeros (n : ℕ) (h : n = 1968) : 
  ∃ k : ℕ, 5^n = 10^k ∧ k ≥ 1968 := 
by 
  sorry

end power_of_five_trailing_zeros_l42_42584


namespace sum_of_cubes_l42_42919

-- Definitions
noncomputable def p : ℂ := sorry
noncomputable def q : ℂ := sorry
noncomputable def r : ℂ := sorry

-- Roots conditions
axiom h_root_p : p^3 - 2 * p^2 + 3 * p - 4 = 0
axiom h_root_q : q^3 - 2 * q^2 + 3 * q - 4 = 0
axiom h_root_r : r^3 - 2 * r^2 + 3 * r - 4 = 0

-- Vieta's conditions
axiom h_sum : p + q + r = 2
axiom h_product_pairs : p * q + q * r + r * p = 3
axiom h_product : p * q * r = 4

-- Goal
theorem sum_of_cubes : p^3 + q^3 + r^3 = 2 :=
  sorry

end sum_of_cubes_l42_42919


namespace compute_expression_equals_375_l42_42321

theorem compute_expression_equals_375 : 15 * (30 / 6) ^ 2 = 375 := 
by 
  have frac_simplified : 30 / 6 = 5 := by sorry
  have power_calculated : 5 ^ 2 = 25 := by sorry
  have final_result : 15 * 25 = 375 := by sorry
  sorry

end compute_expression_equals_375_l42_42321


namespace quadrilateral_centroid_perimeter_l42_42500

-- Definition for the side length of the square and distances for points Q
def side_length : ℝ := 40
def EQ_dist : ℝ := 18
def FQ_dist : ℝ := 34

-- Theorem statement: Perimeter of the quadrilateral formed by centroids
theorem quadrilateral_centroid_perimeter :
  let centroid_perimeter := (4 * ((2 / 3) * side_length))
  centroid_perimeter = (320 / 3) := by
  sorry

end quadrilateral_centroid_perimeter_l42_42500


namespace minimum_boys_needed_l42_42991

theorem minimum_boys_needed (k n m : ℕ) (hn : n > 0) (hm : m > 0) (h : 100 * n + m * k = 10 * k) : n + m = 6 :=
by
  sorry

end minimum_boys_needed_l42_42991


namespace jane_percentage_bread_to_treats_l42_42062

variable (T J_b W_b W_t : ℕ) (P : ℕ)

-- Conditions as stated
axiom h1 : J_b = (P * T) / 100
axiom h2 : W_t = T / 2
axiom h3 : W_b = 3 * W_t
axiom h4 : W_b = 90
axiom h5 : J_b + W_b + T + W_t = 225

theorem jane_percentage_bread_to_treats : P = 75 :=
by
-- Proof skeleton
sorry

end jane_percentage_bread_to_treats_l42_42062


namespace rectangles_in_grid_squares_in_grid_l42_42970

theorem rectangles_in_grid (h_lines : ℕ) (v_lines : ℕ) : h_lines = 31 → v_lines = 31 → 
  (∃ rect_count : ℕ, rect_count = 216225) :=
by
  intros h_lines_eq v_lines_eq
  sorry

theorem squares_in_grid (n : ℕ) : n = 31 → (∃ square_count : ℕ, square_count = 6975) :=
by
  intros n_eq
  sorry

end rectangles_in_grid_squares_in_grid_l42_42970


namespace value_of_polynomial_l42_42691

theorem value_of_polynomial : 
  99^5 - 5 * 99^4 + 10 * 99^3 - 10 * 99^2 + 5 * 99 - 1 = 98^5 := by
  sorry

end value_of_polynomial_l42_42691


namespace first_donor_amount_l42_42929

theorem first_donor_amount
  (x second third fourth : ℝ)
  (h1 : second = 2 * x)
  (h2 : third = 3 * second)
  (h3 : fourth = 4 * third)
  (h4 : x + second + third + fourth = 132)
  : x = 4 := 
by 
  -- Simply add this line to make the theorem complete without proof.
  sorry

end first_donor_amount_l42_42929


namespace total_equipment_cost_l42_42514

-- Definitions of costs in USD
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.20
def socks_cost : ℝ := 6.80
def number_of_players : ℝ := 16

-- Statement to prove
theorem total_equipment_cost :
  number_of_players * (jersey_cost + shorts_cost + socks_cost) = 752 :=
by
  sorry

end total_equipment_cost_l42_42514


namespace find_tricycles_l42_42620

theorem find_tricycles (b t w : ℕ) 
  (sum_children : b + t + w = 10)
  (sum_wheels : 2 * b + 3 * t = 26) :
  t = 6 :=
by sorry

end find_tricycles_l42_42620


namespace log_sum_l42_42950

theorem log_sum : 2 * Real.log 2 + Real.log 25 = 2 := 
by 
  sorry

end log_sum_l42_42950


namespace part1_l42_42432

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

theorem part1 {x : ℝ} : f x ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) := by
  sorry

end part1_l42_42432


namespace even_function_inequality_l42_42971

variable {α : Type*} [LinearOrderedField α]

def is_even_function (f : α → α) : Prop := ∀ x, f x = f (-x)

-- The hypothesis and the assertion in Lean
theorem even_function_inequality
  (f : α → α)
  (h_even : is_even_function f)
  (h3_gt_1 : f 3 > f 1)
  : f (-1) < f 3 :=
sorry

end even_function_inequality_l42_42971


namespace dozens_of_golf_balls_l42_42101

theorem dozens_of_golf_balls (total_balls : ℕ) (dozen_size : ℕ) (h1 : total_balls = 156) (h2 : dozen_size = 12) : total_balls / dozen_size = 13 :=
by
  have h_total : total_balls = 156 := h1
  have h_size : dozen_size = 12 := h2
  sorry

end dozens_of_golf_balls_l42_42101


namespace initial_pens_count_l42_42695

theorem initial_pens_count (P : ℕ) (h : 2 * (P + 22) - 19 = 75) : P = 25 :=
by
  sorry

end initial_pens_count_l42_42695


namespace magic_grid_product_l42_42461

theorem magic_grid_product (p q r s t x : ℕ) 
  (h1: p * 6 * 3 = q * r * s)
  (h2: p * q * t = 6 * r * 2)
  (h3: p * r * x = 6 * 2 * t)
  (h4: q * 2 * 3 = r * s * x)
  (h5: t * 2 * x = 6 * s * 3)
  (h6: 6 * q * 3 = r * s * t)
  (h7: p * r * s = 6 * 2 * q)
  : x = 36 := 
by
  sorry

end magic_grid_product_l42_42461


namespace xy_sq_is_37_over_36_l42_42768

theorem xy_sq_is_37_over_36 (x y : ℚ) (h : 2002 * (x - 1)^2 + |x - 12 * y + 1| = 0) : x^2 + y^2 = 37 / 36 :=
sorry

end xy_sq_is_37_over_36_l42_42768


namespace construct_orthocenter_l42_42987

-- Define the problem setup
variables {A B C O D G H : Point}
variables (circ_circle : Circle)

-- Conditions 
def acute_non_isosceles_triangle (A B C : Point) : Prop := 
  ∃ α β γ : Angle, 
    α ≠ β ∧ β ≠ γ ∧ γ ≠ α ∧ 0 < α < π/2 ∧ 0 < β < π/2 ∧ 0 < γ < π/2

def midpoint (D : Point) (A B : Point) : Prop :=
  dist A D = dist D B ∧ A, D, B collinear

def circumscribed_circle (circ_circle : Circle) (A B C O : Point) : Prop :=
  circ_circle.center = O ∧ A, B, C on circ_circle

-- Prove the existence of orthocenter H at the intersection of lines AG and AD
theorem construct_orthocenter 
  (acute : acute_non_isosceles_triangle A B C)
  (circ : circumscribed_circle circ_circle A B C O)
  (midpt : midpoint D A B)
  : ∃ H : Point, intersection (line_from_to A G) (line_from_to A D) H := 
sorry -- Proof goes here

end construct_orthocenter_l42_42987


namespace max_difference_value_l42_42193

noncomputable def max_difference (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9) : ℝ :=
  x - y

theorem max_difference_value : ∀ (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9),
  max_difference x y h ≤ 1 + 3 * real.sqrt 2 :=
by
  sorry

end max_difference_value_l42_42193


namespace geometric_series_sum_eq_l42_42330

theorem geometric_series_sum_eq :
  let a := (1/3 : ℚ)
  let r := (1/3 : ℚ)
  let n := 8
  let S := a * (1 - r^n) / (1 - r)
  S = 3280 / 6561 :=
by
  sorry

end geometric_series_sum_eq_l42_42330


namespace num_ballpoint_pens_l42_42271

-- Define the total number of school supplies
def total_school_supplies : ℕ := 60

-- Define the number of pencils
def num_pencils : ℕ := 5

-- Define the number of notebooks
def num_notebooks : ℕ := 10

-- Define the number of erasers
def num_erasers : ℕ := 32

-- Define the number of ballpoint pens and prove it equals 13
theorem num_ballpoint_pens : total_school_supplies - (num_pencils + num_notebooks + num_erasers) = 13 :=
by
sorry

end num_ballpoint_pens_l42_42271


namespace no_positive_integer_solution_l42_42251

theorem no_positive_integer_solution (a b c d : ℕ) (h1 : a^2 + b^2 = c^2 - d^2) (h2 : a * b = c * d) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : false := 
by 
  sorry

end no_positive_integer_solution_l42_42251


namespace cos_theta_seven_l42_42033

theorem cos_theta_seven {θ : ℝ} (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -8383 / 98304 :=
by
  sorry

end cos_theta_seven_l42_42033


namespace base_number_pow_19_mod_10_l42_42536

theorem base_number_pow_19_mod_10 (x : ℕ) (h : x ^ 19 % 10 = 7) : x % 10 = 3 :=
sorry

end base_number_pow_19_mod_10_l42_42536


namespace range_of_a_l42_42056

theorem range_of_a (h : ∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1) : a > -1 :=
sorry

end range_of_a_l42_42056


namespace probability_of_real_roots_is_correct_l42_42128

open Real

def has_real_roots (m : ℝ) : Prop :=
  2 * m^2 - 8 ≥ 0 

def favorable_set : Set ℝ := {m | has_real_roots m}

def interval_length (a b : ℝ) : ℝ := b - a

noncomputable def probability_of_real_roots : ℝ :=
  interval_length (-4) (-2) + interval_length 2 3 / interval_length (-4) 3

theorem probability_of_real_roots_is_correct : probability_of_real_roots = 3 / 7 :=
by
  sorry

end probability_of_real_roots_is_correct_l42_42128


namespace probability_of_specific_draw_l42_42958

noncomputable def probability_first_ace_second_spade_third_3 : ℚ :=
  let prob_case1 := (3 / 52) * (12 / 51) * (4 / 50)
  let prob_case2 := (3 / 52) * (1 / 51) * (3 / 50)
  let prob_case3 := (1 / 52) * (12 / 51) * (4 / 50)
  let prob_case4 := (1 / 52) * (1 / 51) * (3 / 50)
  (prob_case1 + prob_case2 + prob_case3 + prob_case4)

theorem probability_of_specific_draw :
  probability_first_ace_second_spade_third_3 = 17 / 11050 :=
begin
  -- Skipping the proof steps
  sorry
end

end probability_of_specific_draw_l42_42958


namespace remainder_div_by_13_l42_42125

-- Define conditions
variable (N : ℕ)
variable (k : ℕ)

-- Given condition
def condition := N = 39 * k + 19

-- Goal statement
theorem remainder_div_by_13 (h : condition N k) : N % 13 = 6 :=
sorry

end remainder_div_by_13_l42_42125


namespace visited_iceland_l42_42215

variable (total : ℕ) (visitedNorway : ℕ) (visitedBoth : ℕ) (visitedNeither : ℕ)

theorem visited_iceland (h_total : total = 50)
                        (h_visited_norway : visitedNorway = 23)
                        (h_visited_both : visitedBoth = 21)
                        (h_visited_neither : visitedNeither = 23) :
                        (total - (visitedNorway - visitedBoth + visitedNeither) = 25) :=
  sorry

end visited_iceland_l42_42215


namespace product_of_three_numbers_l42_42665

theorem product_of_three_numbers :
  ∃ (x y z : ℚ), 
    (x + y + z = 30) ∧ 
    (x = 3 * (y + z)) ∧ 
    (y = 8 * z) ∧ 
    (x * y * z = 125) := 
by
  sorry

end product_of_three_numbers_l42_42665


namespace cos_seven_theta_l42_42047

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (7 * θ) = -160481 / 2097152 := by
  sorry

end cos_seven_theta_l42_42047


namespace number_of_insects_l42_42317

theorem number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) (h : total_legs = 54) (k : legs_per_insect = 6) :
  total_legs / legs_per_insect = 9 := by
  sorry

end number_of_insects_l42_42317


namespace necessary_but_not_sufficient_condition_l42_42754

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - 3 * x < 0) → (0 < x ∧ x < 4) :=
sorry

end necessary_but_not_sufficient_condition_l42_42754


namespace age_proof_l42_42563

   variable (x : ℝ)
   
   theorem age_proof (h : 3 * (x + 5) - 3 * (x - 5) = x) : x = 30 :=
   by
     sorry
   
end age_proof_l42_42563


namespace total_amount_paid_l42_42122

/-- Conditions -/
def days_in_may : Nat := 31
def rate_per_day : ℚ := 0.5
def days_book1_borrowed : Nat := 20
def days_book2_borrowed : Nat := 31
def days_book3_borrowed : Nat := 31

/-- Question and Proof -/
theorem total_amount_paid : rate_per_day * (days_book1_borrowed + days_book2_borrowed + days_book3_borrowed) = 41 := by
  sorry

end total_amount_paid_l42_42122


namespace rate_per_sqm_l42_42657

theorem rate_per_sqm (length width : ℝ) (cost : ℝ) (Area : ℝ := length * width) (rate : ℝ := cost / Area) 
  (h_length : length = 5.5) (h_width : width = 3.75) (h_cost : cost = 8250) : 
  rate = 400 :=
sorry

end rate_per_sqm_l42_42657


namespace x_can_be_positive_negative_or_zero_l42_42882

noncomputable
def characteristics_of_x (x y z w : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : w ≠ 0) 
  (h5 : (x / y) > (z / w)) (h6 : (y * w) > 0) : Prop :=
  ∃ r : ℝ, r = x

theorem x_can_be_positive_negative_or_zero (x y z w : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : w ≠ 0) (h5 : (x / y) > (z / w)) (h6 : (y * w) > 0) : 
  (characteristics_of_x x y z w h1 h2 h3 h4 h5 h6) :=
sorry

end x_can_be_positive_negative_or_zero_l42_42882


namespace max_x_minus_y_l42_42190

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l42_42190


namespace shed_width_l42_42314

theorem shed_width (backyard_length backyard_width shed_length area_needed : ℝ)
  (backyard_area : backyard_length * backyard_width = 260)
  (sod_area : area_needed = 245)
  (shed_dim : shed_length = 3) :
  (backyard_length * backyard_width - area_needed) / shed_length = 5 :=
by
  -- We need to prove the width of the shed given the conditions
  sorry

end shed_width_l42_42314


namespace find_seventh_number_l42_42117

-- Let's denote the 10 numbers as A1, A2, A3, A4, A5, A6, A7, A8, A9, A10.
variables {A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : ℝ}

-- The average of all 10 numbers is 60.
def avg_10 (A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : ℝ) := (A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10) / 10 = 60

-- The average of the first 6 numbers is 68.
def avg_first_6 (A1 A2 A3 A4 A5 A6 : ℝ) := (A1 + A2 + A3 + A4 + A5 + A6) / 6 = 68

-- The average of the last 6 numbers is 75.
def avg_last_6 (A5 A6 A7 A8 A9 A10 : ℝ) := (A5 + A6 + A7 + A8 + A9 + A10) / 6 = 75

-- Proving that the 7th number (A7) is 192.
theorem find_seventh_number (A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : ℝ) 
  (h1 : avg_10 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10) 
  (h2 : avg_first_6 A1 A2 A3 A4 A5 A6) 
  (h3 : avg_last_6 A5 A6 A7 A8 A9 A10) :
  A7 = 192 :=
by
  sorry

end find_seventh_number_l42_42117


namespace simplify_polynomial_l42_42646

theorem simplify_polynomial (y : ℝ) :
    (4 * y^10 + 6 * y^9 + 3 * y^8) + (2 * y^12 + 5 * y^10 + y^9 + y^7 + 4 * y^4 + 7 * y + 9) =
    2 * y^12 + 9 * y^10 + 7 * y^9 + 3 * y^8 + y^7 + 4 * y^4 + 7 * y + 9 := by
  sorry

end simplify_polynomial_l42_42646


namespace quadratic_roots_r_l42_42634

theorem quadratic_roots_r (a b m p r : ℚ) :
  (∀ x : ℚ, x^2 - m * x + 3 = 0 → (x = a ∨ x = b)) →
  (∀ x : ℚ, x^2 - p * x + r = 0 → (x = a + 1 / b ∨ x = b + 1 / a + 1)) →
  r = 19 / 3 :=
by
  sorry

end quadratic_roots_r_l42_42634


namespace determine_constant_l42_42577

theorem determine_constant (c : ℝ) :
  (∃ d : ℝ, 9 * x^2 - 24 * x + c = (3 * x + d)^2) ↔ c = 16 :=
by
  sorry

end determine_constant_l42_42577


namespace part1_part2_l42_42410

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) :=
by
  sorry

end part1_part2_l42_42410


namespace range_x_inequality_l42_42595

theorem range_x_inequality (a b x : ℝ) (ha : a ≠ 0) :
  (x ≥ 1/2) ∧ (x ≤ 5/2) →
  |a + b| + |a - b| ≥ |a| * (|x - 1| + |x - 2|) :=
by
  sorry

end range_x_inequality_l42_42595


namespace area_MNK_geq_quarter_area_ABC_l42_42637

theorem area_MNK_geq_quarter_area_ABC
  (O : Point)
  (A B C M : Point) 
  (h_triangle : IsAcuteTriangle O A B C)
  (h_M_on_AB : OnLine M A B) 
  (K : Point)
  (h_K_on_circ_AMO : OnCircumcircle K A M O)
  (h_K_on_AC : OnLine K A C)
  (N : Point)
  (h_N_on_circ_BMO : OnCircumcircle N B M O)
  (h_N_on_BC : OnLine N B C)
  : ∃ M, isMidpoint M A B ∧ Area (Triangle.mk M N K) = 1/4 * Area (Triangle.mk A B C) := sorry

end area_MNK_geq_quarter_area_ABC_l42_42637


namespace tank_ratio_l42_42720

variable (C D : ℝ)
axiom h1 : 3 / 4 * C = 2 / 5 * D

theorem tank_ratio : C / D = 8 / 15 := by
  sorry

end tank_ratio_l42_42720


namespace cos_seven_theta_l42_42045

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (7 * θ) = -160481 / 2097152 := by
  sorry

end cos_seven_theta_l42_42045


namespace find_D_l42_42843

-- Definitions
def divides (a b : ℕ) : Prop := ∃ k, b = a * k
def remainder (a b r : ℕ) : Prop := ∃ k, a = b * k + r

-- Problem Statement
theorem find_D {N D : ℕ} (h1 : remainder N D 75) (h2 : remainder N 37 1) : 
  D = 112 :=
by
  sorry

end find_D_l42_42843


namespace at_most_one_divisor_square_l42_42917

theorem at_most_one_divisor_square (p n : ℕ) (hp : Prime p) (hpn : p > 2) (hn_pos : 0 < n):
  ∃ (d : ℕ), (d ∈ divisors (p * n^2) ∧ ∃ (k : ℕ), k^2 = n^2 + d) → 
  ∀ (d1 d2 : ℕ), (d1 ∈ divisors (p * n^2) ∧ ∃ k1, k1^2 = n^2 + d1) → 
                 (d2 ∈ divisors (p * n^2) ∧ ∃ k2, k2^2 = n^2 + d2) → 
                 d1 = d2 :=
sorry

end at_most_one_divisor_square_l42_42917


namespace max_x_minus_y_l42_42181

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  ∃ k ∈ ℝ, x - y ≤ k ∧ k = 1 + 3 * real.sqrt 2 :=
sorry

end max_x_minus_y_l42_42181


namespace prime_dates_in_2008_l42_42237

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_prime_date (month day : ℕ) : Prop := is_prime month ∧ is_prime day

noncomputable def prime_dates_2008 : ℕ :=
  let prime_months := [2, 3, 5, 7, 11]
  let prime_days_31 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let prime_days_30 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let prime_days_29 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  prime_months.foldl (λ acc month => 
    acc + match month with
      | 2 => List.length prime_days_29
      | 3 | 5 | 7 => List.length prime_days_31
      | 11 => List.length prime_days_30
      | _ => 0
    ) 0

theorem prime_dates_in_2008 : 
  prime_dates_2008 = 53 :=
  sorry

end prime_dates_in_2008_l42_42237


namespace distance_behind_C_l42_42617

-- Conditions based on the problem
def distance_race : ℕ := 1000
def distance_B_when_A_finishes : ℕ := 50
def distance_C_when_B_finishes : ℕ := 100

-- Derived condition based on given problem details
def distance_run_by_B_when_A_finishes : ℕ := distance_race - distance_B_when_A_finishes
def distance_run_by_C_when_B_finishes : ℕ := distance_race - distance_C_when_B_finishes

-- Ratios
def ratio_B_to_A : ℚ := distance_run_by_B_when_A_finishes / distance_race
def ratio_C_to_B : ℚ := distance_run_by_C_when_B_finishes / distance_race

-- Combined ratio
def ratio_C_to_A : ℚ := ratio_C_to_B * ratio_B_to_A

-- Distance run by C when A finishes
def distance_run_by_C_when_A_finishes : ℚ := distance_race * ratio_C_to_A

-- Distance C is behind the finish line when A finishes
def distance_C_behind_when_A_finishes : ℚ := distance_race - distance_run_by_C_when_A_finishes

theorem distance_behind_C (d_race : ℕ) (d_BA : ℕ) (d_CB : ℕ)
  (hA : d_race = 1000) (hB : d_BA = 50) (hC : d_CB = 100) :
  distance_C_behind_when_A_finishes = 145 :=
  by sorry

end distance_behind_C_l42_42617


namespace processing_times_maximum_salary_l42_42549

def monthly_hours : ℕ := 8 * 25
def base_salary : ℕ := 800
def earnings_per_A : ℕ := 16
def earnings_per_B : ℕ := 12

theorem processing_times :
  ∃ (x y : ℕ),
    x + 3 * y = 5 ∧ 2 * x + 5 * y = 9 ∧ x = 2 ∧ y = 1 :=
by
  sorry

theorem maximum_salary :
  ∃ (a b W : ℕ),
    a ≥ 50 ∧ 
    b = monthly_hours - 2 * a ∧ 
    W = base_salary + earnings_per_A * a + earnings_per_B * b ∧ 
    a = 50 ∧ 
    b = 100 ∧ 
    W = 2800 :=
by
  sorry

end processing_times_maximum_salary_l42_42549


namespace cos_seven_theta_l42_42041

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -953 / 1024 :=
by sorry

end cos_seven_theta_l42_42041


namespace triangle_perimeter_is_26_l42_42749

-- Define the lengths of the medians as given conditions
def median1 := 3
def median2 := 4
def median3 := 6

-- Define the perimeter of the triangle
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- The theorem to prove that the perimeter is 26 cm
theorem triangle_perimeter_is_26 :
  perimeter (2 * median1) (2 * median2) (2 * median3) = 26 :=
by
  -- Calculation follows directly from the definition
  sorry

end triangle_perimeter_is_26_l42_42749


namespace sum_of_possible_values_of_x_l42_42757

theorem sum_of_possible_values_of_x (x : ℝ) (h : (x + 3) * (x - 4) = 22) : ∃ (x1 x2 : ℝ), x^2 - x - 34 = 0 ∧ x1 + x2 = 1 :=
by
  sorry

end sum_of_possible_values_of_x_l42_42757


namespace mutually_exclusive_both_miss_hitting_at_least_once_l42_42252

open Finset

variable {α : Type} [DecidableEq α]

/-- Definitions of the shooting events -/
def hitting_at_least_once (shots: Finset α) : Prop := (shots.card > 0)
def both_miss (shots: Finset α) : Prop := (shots.card = 0)

/-- Mutually exclusive events -/
def mutually_exclusive (A B : Finset α → Prop) : Prop :=
  ∀ shots, ¬ (A shots ∧ B shots)

-- Proof statement
theorem mutually_exclusive_both_miss_hitting_at_least_once :
  mutually_exclusive hitting_at_least_once both_miss :=
by
  intros shots
  unfold mutually_exclusive hitting_at_least_once both_miss
  sorry

end mutually_exclusive_both_miss_hitting_at_least_once_l42_42252


namespace butterflies_count_l42_42096

theorem butterflies_count (total_black_dots : ℕ) (black_dots_per_butterfly : ℕ) 
                          (h1 : total_black_dots = 4764) 
                          (h2 : black_dots_per_butterfly = 12) :
                          total_black_dots / black_dots_per_butterfly = 397 :=
by
  sorry

end butterflies_count_l42_42096


namespace part1_part2_l42_42412

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) :=
by
  sorry

end part1_part2_l42_42412


namespace right_triangle_construction_condition_l42_42996

theorem right_triangle_construction_condition (A B C : Point) (b d : ℝ) :
  AC = b → AC + BC - AB = d → b > d :=
by
  intro h1 h2
  sorry

end right_triangle_construction_condition_l42_42996


namespace find_a_l42_42995

noncomputable def g (x : ℝ) := 5 * x - 7

theorem find_a (a : ℝ) (h : g a = 0) : a = 7 / 5 :=
sorry

end find_a_l42_42995


namespace part1_solution_set_part2_range_l42_42354

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l42_42354


namespace sum_of_numbers_equal_16_l42_42848

theorem sum_of_numbers_equal_16 
  (a b c : ℕ) 
  (h1 : a * b = a * c - 1 ∨ a * b = b * c - 1 ∨ a * c = b * c - 1) 
  (h2 : a * b = a * c + 49 ∨ a * b = b * c + 49 ∨ a * c = b * c + 49) :
  a + b + c = 16 :=
sorry

end sum_of_numbers_equal_16_l42_42848


namespace sum_of_roots_l42_42762

-- Define the main condition
def equation (x : ℝ) : Prop :=
  (x + 3) * (x - 4) = 22

-- Define the statement we want to prove
theorem sum_of_roots : ∀ x1 x2 : ℝ, (equation x1 ∧ equation x2) → x1 + x2 = 1 := by
  intros x1 x2 h
  sorry

end sum_of_roots_l42_42762


namespace simplest_common_denominator_l42_42946

theorem simplest_common_denominator (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  LCM (2 * x^2 * y) (6 * x * y^3) = 6 * x^2 * y^3 := 
sorry

end simplest_common_denominator_l42_42946


namespace red_marbles_l42_42144

theorem red_marbles (R B : ℕ) (h₁ : B = R + 24) (h₂ : B = 5 * R) : R = 6 := by
  sorry

end red_marbles_l42_42144


namespace probability_event_l42_42960

-- Definitions of the conditions
def boxA := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
def boxB := {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28}

-- Definition of the desired event
def eventA(t : ℕ) := t ≤ 14
def eventB(t : ℕ) := t % 2 = 0 ∨ t > 25

-- Probability calculation
noncomputable def probability_boxA := (boxA.count eventA).toRat / boxA.card.toRat
noncomputable def probability_boxB := (boxB.count eventB).toRat / boxB.card.toRat
noncomputable def combined_probability := probability_boxA * probability_boxB

-- Lean theorem statement
theorem probability_event : combined_probability = 70 / 171 := by
  sorry

end probability_event_l42_42960


namespace part1_solution_part2_solution_l42_42403

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l42_42403


namespace binary101_to_decimal_l42_42323

theorem binary101_to_decimal :
  let binary_101 := 1 * 2^2 + 0 * 2^1 + 1 * 2^0
  binary_101 = 5 := 
by
  let binary_101 := 1 * 2^2 + 0 * 2^1 + 1 * 2^0
  show binary_101 = 5
  sorry

end binary101_to_decimal_l42_42323


namespace part1_solution_set_part2_range_a_l42_42391

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l42_42391


namespace value_of_neg_a_squared_sub_3a_l42_42448

variable (a : ℝ)
variable (h : a^2 + 3 * a - 5 = 0)

theorem value_of_neg_a_squared_sub_3a : -a^2 - 3*a = -5 :=
by
  sorry

end value_of_neg_a_squared_sub_3a_l42_42448


namespace correct_range_a_l42_42639

noncomputable def proposition_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
noncomputable def proposition_q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0

theorem correct_range_a (a : ℝ) :
  (¬ ∃ x, proposition_p a x → ¬ ∃ x, proposition_q x) →
  (a ≤ -4 ∨ a ≥ 2 ∨ a = 0) :=
sorry

end correct_range_a_l42_42639


namespace max_difference_value_l42_42194

noncomputable def max_difference (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9) : ℝ :=
  x - y

theorem max_difference_value : ∀ (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 9),
  max_difference x y h ≤ 1 + 3 * real.sqrt 2 :=
by
  sorry

end max_difference_value_l42_42194


namespace part1_part2_l42_42423

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l42_42423


namespace boat_man_mass_l42_42833

theorem boat_man_mass (L B h : ℝ) (rho g : ℝ): 
  L = 3 → B = 2 → h = 0.015 → rho = 1000 → g = 9.81 → (rho * L * B * h * g) / g = 9 :=
by
  intros
  simp_all
  sorry

end boat_man_mass_l42_42833


namespace emily_has_7_times_more_oranges_than_sandra_l42_42861

theorem emily_has_7_times_more_oranges_than_sandra
  (B S E : ℕ)
  (h1 : S = 3 * B)
  (h2 : B = 12)
  (h3 : E = 252) :
  ∃ k : ℕ, E = k * S ∧ k = 7 :=
by
  use 7
  sorry

end emily_has_7_times_more_oranges_than_sandra_l42_42861


namespace determine_a_square_binomial_l42_42729

theorem determine_a_square_binomial (a : ℝ) :
  (∃ r s : ℝ, ∀ x : ℝ, ax^2 + 24*x + 9 = (r*x + s)^2) → a = 16 :=
by
  sorry

end determine_a_square_binomial_l42_42729


namespace gcd_lcm_problem_part1_gcd_lcm_problem_part2_l42_42329

open Int

noncomputable def a1 := 5^2 * 7^4
noncomputable def a2 := 490 * 175

noncomputable def b1 := 2^5 * 3 * 7
noncomputable def b2 := 3^4 * 5^4 * 7^2
noncomputable def b3 := 10000

theorem gcd_lcm_problem_part1 : 
  gcd a1 a2 = 8575 ∧ Nat.lcm a1 a2 = 600250 := 
by
  sorry

theorem gcd_lcm_problem_part2 : 
  gcd (gcd b1 b2) b3 = 1 ∧ Nat.lcm b1 (Nat.lcm b2 b3) = 793881600 := 
by
  sorry

end gcd_lcm_problem_part1_gcd_lcm_problem_part2_l42_42329


namespace part1_part2_l42_42371

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l42_42371


namespace max_x_minus_y_l42_42175

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : (x - y) ≤ 1 + 3 * (Real.sqrt 2) :=
sorry

end max_x_minus_y_l42_42175


namespace average_marks_correct_l42_42697

-- Define the marks obtained in each subject
def english_marks := 86
def mathematics_marks := 85
def physics_marks := 92
def chemistry_marks := 87
def biology_marks := 95

-- Calculate total marks and average marks
def total_marks := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects := 5
def average_marks := total_marks / num_subjects

-- Prove that Dacid's average marks are 89
theorem average_marks_correct : average_marks = 89 := by
  sorry

end average_marks_correct_l42_42697


namespace sqrt_sin_cos_expression_l42_42286

theorem sqrt_sin_cos_expression (α β : ℝ) : 
  Real.sqrt ((1 - Real.sin α * Real.sin β)^2 - (Real.cos α * Real.cos β)^2) = |Real.sin α - Real.sin β| :=
sorry

end sqrt_sin_cos_expression_l42_42286


namespace find_cos_7theta_l42_42039

theorem find_cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = 1105 / 16384 :=
by
  sorry

end find_cos_7theta_l42_42039


namespace findInitialVolume_l42_42555

def initialVolume (V : ℝ) : Prop :=
  let newVolume := V + 18
  let initialSugar := 0.27 * V
  let addedSugar := 3.2
  let totalSugar := initialSugar + addedSugar
  let finalSugarPercentage := 0.26536312849162012
  finalSugarPercentage * newVolume = totalSugar 

theorem findInitialVolume : ∃ (V : ℝ), initialVolume V ∧ V = 340 := by
  use 340
  unfold initialVolume
  sorry

end findInitialVolume_l42_42555


namespace factorize_x_squared_minus_4_factorize_2mx_squared_minus_4mx_plus_2m_factorize_y_quad_l42_42154

-- Problem 1
theorem factorize_x_squared_minus_4 (x : ℝ) :
  x^2 - 4 = (x + 2) * (x - 2) :=
by { 
  sorry
}

-- Problem 2
theorem factorize_2mx_squared_minus_4mx_plus_2m (x m : ℝ) :
  2 * m * x^2 - 4 * m * x + 2 * m = 2 * m * (x - 1)^2 :=
by { 
  sorry
}

-- Problem 3
theorem factorize_y_quad (y : ℝ) :
  (y^2 - 1)^2 - 6 * (y^2 - 1) + 9 = (y + 2)^2 * (y - 2)^2 :=
by { 
  sorry
}

end factorize_x_squared_minus_4_factorize_2mx_squared_minus_4mx_plus_2m_factorize_y_quad_l42_42154


namespace boys_girls_students_l42_42671

theorem boys_girls_students (total_students : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ)
  (h1 : total_students = 100)
  (h2 : ratio_boys = 3)
  (h3 : ratio_girls = 2) :
  3 * (total_students / (ratio_boys + ratio_girls)) - 2 * (total_students / (ratio_boys + ratio_girls)) = 20 :=
by
  sorry

end boys_girls_students_l42_42671


namespace initial_kids_count_l42_42098

-- Define the initial number of kids as a variable
def init_kids (current_kids join_kids : Nat) : Nat :=
  current_kids - join_kids

-- Define the total current kids and kids joined
def current_kids : Nat := 36
def join_kids : Nat := 22

-- Prove that the initial number of kids was 14
theorem initial_kids_count : init_kids current_kids join_kids = 14 :=
by
  -- Proof skipped
  sorry

end initial_kids_count_l42_42098


namespace Nancy_more_pearl_beads_l42_42928

-- Define the problem conditions
def metal_beads_Nancy : ℕ := 40
def crystal_beads_Rose : ℕ := 20
def stone_beads_Rose : ℕ := crystal_beads_Rose * 2
def total_beads_needed : ℕ := 20 * 8
def total_Rose_beads : ℕ := crystal_beads_Rose + stone_beads_Rose
def pearl_beads_Nancy : ℕ := total_beads_needed - total_Rose_beads

-- State the theorem to prove
theorem Nancy_more_pearl_beads :
  pearl_beads_Nancy = metal_beads_Nancy + 60 :=
by
  -- We leave the proof as an exercise
  sorry

end Nancy_more_pearl_beads_l42_42928


namespace height_percentage_increase_l42_42767

theorem height_percentage_increase (B A : ℝ) 
  (hA : A = B * 0.8) : ((B - A) / A) * 100 = 25 := by
--   Given the condition that A's height is 20% less than B's height
--   translate into A = B * 0.8
--   We need to show ((B - A) / A) * 100 = 25
sorry

end height_percentage_increase_l42_42767


namespace max_initial_value_seq_l42_42867

theorem max_initial_value_seq :
  ∀ (x : Fin 1996 → ℝ),
    (∀ i : Fin 1996, 1 ≤ x i) →
    (x 0 = x 1995) →
    (∀ i : Fin 1995, x i + 2 / x i = 2 * x (i + 1) + 1 / x (i + 1)) →
    x 0 ≤ 2 ^ 997 :=
sorry

end max_initial_value_seq_l42_42867


namespace problem_f_x_sum_neg_l42_42552

open Function

-- Definitions for monotonic decreasing and odd properties of the function
def isOdd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def isMonotonicallyDecreasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f y ≤ f x

-- The main theorem to prove
theorem problem_f_x_sum_neg
  (f : ℝ → ℝ)
  (h_odd : isOdd f)
  (h_monotone : isMonotonicallyDecreasing f)
  (x₁ x₂ x₃ : ℝ)
  (h₁ : x₁ + x₂ > 0)
  (h₂ : x₂ + x₃ > 0)
  (h₃ : x₃ + x₁ > 0) :
  f x₁ + f x₂ + f x₃ < 0 :=
by
  sorry

end problem_f_x_sum_neg_l42_42552


namespace simplify_expression_l42_42888

theorem simplify_expression (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : a + c > b) :
  |a + b - c| - |b - a - c| = 2 * b - 2 * c :=
by
  sorry

end simplify_expression_l42_42888


namespace scientific_notation_of_274000000_l42_42074

theorem scientific_notation_of_274000000 :
  274000000 = 2.74 * 10^8 := by
  sorry

end scientific_notation_of_274000000_l42_42074


namespace part1_solution_set_part2_range_of_a_l42_42380

-- Part 1
theorem part1_solution_set (x : ℝ) : (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x + 3| > -a) ↔ (a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_of_a_l42_42380


namespace gcd_456_357_l42_42675

theorem gcd_456_357 : Nat.gcd 456 357 = 3 := by
  sorry

end gcd_456_357_l42_42675


namespace top_card_is_queen_probability_l42_42840

theorem top_card_is_queen_probability :
  let total_cards := 54
  let number_of_queens := 4
  (number_of_queens / total_cards) = (2 / 27) := by
    sorry

end top_card_is_queen_probability_l42_42840


namespace part1_part2_l42_42405

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) :=
by
  sorry

end part1_part2_l42_42405


namespace part1_solution_part2_solution_l42_42398

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l42_42398


namespace find_page_number_l42_42212

theorem find_page_number (n p : ℕ) (h1 : (n * (n + 1)) / 2 + 2 * p = 2046) : p = 15 :=
sorry

end find_page_number_l42_42212


namespace find_first_number_l42_42594

/-- Given a sequence of 6 numbers b_1, b_2, ..., b_6 such that:
  1. For n ≥ 2, b_{2n} = b_{2n-1}^2
  2. For n ≥ 2, b_{2n+1} = (b_{2n} * b_{2n-1})^2
And the sequence ends as: b_4 = 16, b_5 = 256, and b_6 = 65536,
prove that the first number b_1 is 1/2. -/
theorem find_first_number : 
  ∃ b : ℕ → ℝ, b 6 = 65536 ∧ b 5 = 256 ∧ b 4 = 16 ∧ 
  (∀ n ≥ 2, b (2 * n) = (b (2 * n - 1)) ^ 2) ∧
  (∀ n ≥ 2, b (2 * n + 1) = (b (2 * n) * b (2 * n - 1)) ^ 2) ∧ 
  b 1 = 1/2 :=
by
  sorry

end find_first_number_l42_42594


namespace max_value_x_minus_y_l42_42178

theorem max_value_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_value_x_minus_y_l42_42178


namespace a_in_A_l42_42795

def A := {x : ℝ | x ≥ 2 * Real.sqrt 2}
def a : ℝ := 3

theorem a_in_A : a ∈ A :=
by 
  sorry

end a_in_A_l42_42795


namespace least_possible_b_l42_42253

theorem least_possible_b (a b : ℕ) (p : ℕ) (prime_p : Nat.Prime p) (h_a_factors : ∃ k, a = p^k ∧ k + 1 = 3) (h_b_factors : ∃ m, b = p^m ∧ m + 1 = a) (h_divisible : b % a = 0) : 
  b = 8 := 
by 
  sorry

end least_possible_b_l42_42253


namespace soccer_ball_cost_l42_42120

theorem soccer_ball_cost (F S : ℝ) 
  (h1 : 3 * F + S = 155) 
  (h2 : 2 * F + 3 * S = 220) :
  S = 50 := 
sorry

end soccer_ball_cost_l42_42120


namespace probability_of_at_least_one_solving_l42_42102

variable (P1 P2 : ℝ)

theorem probability_of_at_least_one_solving : 
  (1 - (1 - P1) * (1 - P2)) = P1 + P2 - P1 * P2 := 
sorry

end probability_of_at_least_one_solving_l42_42102


namespace minimum_group_members_round_table_l42_42299

theorem minimum_group_members_round_table (n : ℕ) (h1 : ∀ (a : ℕ),  a < n) : 5 ≤ n :=
by
  sorry

end minimum_group_members_round_table_l42_42299


namespace jim_less_than_anthony_l42_42933

-- Definitions for the conditions
def scott_shoes : ℕ := 7

def anthony_shoes : ℕ := 3 * scott_shoes

def jim_shoes : ℕ := anthony_shoes - 2

-- Lean statement to prove the problem
theorem jim_less_than_anthony : anthony_shoes - jim_shoes = 2 := by
  sorry

end jim_less_than_anthony_l42_42933


namespace part1_part2_l42_42404

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) :=
by
  sorry

end part1_part2_l42_42404


namespace first_class_product_rate_l42_42599

theorem first_class_product_rate
  (total_products : ℕ)
  (pass_rate : ℝ)
  (first_class_rate_among_qualified : ℝ)
  (pass_rate_correct : pass_rate = 0.95)
  (first_class_rate_correct : first_class_rate_among_qualified = 0.2) :
  (first_class_rate_among_qualified * pass_rate : ℝ) = 0.19 :=
by
  rw [pass_rate_correct, first_class_rate_correct]
  norm_num


end first_class_product_rate_l42_42599


namespace find_b_l42_42523

def perpendicular_vectors (v1 v2 : ℝ × ℝ) : Prop := 
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_b (b : ℝ) :
  perpendicular_vectors ⟨-5, 11⟩ ⟨b, 3⟩ →
  b = 33 / 5 :=
by
  sorry

end find_b_l42_42523


namespace sequence_a_n_l42_42023

theorem sequence_a_n (a : ℕ → ℚ)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, a (n + 1) * (a n + 1) = a n) :
  a 6 = 1 / 6 :=
  sorry

end sequence_a_n_l42_42023


namespace gcd_108_450_l42_42865

theorem gcd_108_450 : Nat.gcd 108 450 = 18 :=
by
  sorry

end gcd_108_450_l42_42865


namespace books_ratio_l42_42135

-- Definitions based on the conditions
def Alyssa_books : Nat := 36
def Nancy_books : Nat := 252

-- Statement to prove
theorem books_ratio :
  (Nancy_books / Alyssa_books) = 7 := 
sorry

end books_ratio_l42_42135


namespace scholarship_amount_l42_42642

-- Definitions
def tuition_per_semester : ℕ := 22000
def parents_contribution : ℕ := tuition_per_semester / 2
def work_hours : ℕ := 200
def hourly_wage : ℕ := 10
def work_earnings : ℕ := work_hours * hourly_wage
def remaining_tuition : ℕ := tuition_per_semester - parents_contribution - work_earnings

-- Theorem to prove the scholarship amount
theorem scholarship_amount (S : ℕ) (h : 3 * S = remaining_tuition) : S = 3000 :=
by
  sorry

end scholarship_amount_l42_42642


namespace tan_alpha_equals_one_l42_42346

theorem tan_alpha_equals_one (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : Real.cos (α + β) = Real.sin (α - β))
  : Real.tan α = 1 := 
by
  sorry

end tan_alpha_equals_one_l42_42346


namespace snow_at_least_once_in_five_days_l42_42661

theorem snow_at_least_once_in_five_days :
  let p := (3/4 : ℚ)
  let q := (1/4 : ℚ)
  let probability_no_snow_in_five_days := q^5
  let probability_snow_at_least_once := 1 - probability_no_snow_in_five_days
    probability_snow_at_least_once = 1023 / 1024 :=
by {
  sorry
}

end snow_at_least_once_in_five_days_l42_42661


namespace triangle_area_solution_l42_42574

noncomputable def triangle_area (a b : ℝ) : ℝ := 
  let r := 6 -- radius of each circle
  let d := 2 -- derived distance
  let s := 2 * Real.sqrt 3 * d -- side length of the equilateral triangle
  let area := (Real.sqrt 3 / 4) * s^2 
  area

theorem triangle_area_solution : ∃ a b : ℝ, 
  triangle_area a b = 3 * Real.sqrt 3 ∧ 
  a + b = 27 := 
by 
  exists 27
  exists 3
  sorry

end triangle_area_solution_l42_42574


namespace simplify_polynomial_l42_42808

theorem simplify_polynomial :
  (3 * x ^ 4 - 2 * x ^ 3 + 5 * x ^ 2 - 8 * x + 10) + (7 * x ^ 5 - 3 * x ^ 4 + x ^ 3 - 7 * x ^ 2 + 2 * x - 2)
  = 7 * x ^ 5 - x ^ 3 - 2 * x ^ 2 - 6 * x + 8 :=
by sorry

end simplify_polynomial_l42_42808


namespace red_marbles_l42_42145

theorem red_marbles (R B : ℕ) (h₁ : B = R + 24) (h₂ : B = 5 * R) : R = 6 := by
  sorry

end red_marbles_l42_42145


namespace cos_seven_theta_l42_42042

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -953 / 1024 :=
by sorry

end cos_seven_theta_l42_42042


namespace multiple_of_4_difference_multiple_of_4_l42_42935

variables (a b : ℤ)

def is_multiple_of (x y : ℤ) : Prop := ∃ k : ℤ, x = k * y

axiom h1 : is_multiple_of a 4
axiom h2 : is_multiple_of b 8

theorem multiple_of_4 (b : ℤ) (h : is_multiple_of b 8) : is_multiple_of b 4 :=
by {
  unfold is_multiple_of at *,
  cases h with k hk,
  use k * 2,
  rw [hk],
  norm_num,
}

theorem difference_multiple_of_4 (a b : ℤ) (ha : is_multiple_of a 4) (hb : is_multiple_of b 4) : is_multiple_of (a - b) 4 :=
by {
  unfold is_multiple_of at *,
  cases ha with ka hka,
  cases hb with kb hkb,
  use (ka - kb),
  rw [hka, hkb, sub_mul, mul_sub],
}

end multiple_of_4_difference_multiple_of_4_l42_42935


namespace part1_monotonically_increasing_interval_part1_symmetry_axis_part2_find_a_b_l42_42602

noncomputable def f (a b x : ℝ) : ℝ :=
  a * (2 * (Real.cos (x/2))^2 + Real.sin x) + b

theorem part1_monotonically_increasing_interval (b : ℝ) (k : ℤ) :
  let f := f 1 b
  ∀ x, x ∈ Set.Icc (-3 * Real.pi / 4 + 2 * k * Real.pi) (Real.pi / 4 + 2 * k * Real.pi) ->
    f x <= f (x + Real.pi) :=
sorry

theorem part1_symmetry_axis (b : ℝ) (k : ℤ) :
  let f := f 1 b
  ∀ x, f x = f (2 * (Real.pi / 4 + k * Real.pi) - x) :=
sorry

theorem part2_find_a_b (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ Real.pi)
  (h2 : ∃ (a b : ℝ), ∀ x, x ∈ Set.Icc 0 Real.pi → (a > 0 ∧ 3 ≤ f a b x ∧ f a b x ≤ 4)) :
  (1 - Real.sqrt 2 < a ∧ a < 1 + Real.sqrt 2) ∧ b = 3 :=
sorry

end part1_monotonically_increasing_interval_part1_symmetry_axis_part2_find_a_b_l42_42602


namespace part1_solution_set_part2_range_l42_42353

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l42_42353


namespace part1_solution_set_part2_range_of_a_l42_42413

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l42_42413


namespace part1_solution_set_part2_range_of_a_l42_42378

-- Part 1
theorem part1_solution_set (x : ℝ) : (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x + 3| > -a) ↔ (a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_of_a_l42_42378


namespace movie_attendance_l42_42297

theorem movie_attendance (total_seats : ℕ) (empty_seats : ℕ) (h1 : total_seats = 750) (h2 : empty_seats = 218) :
  total_seats - empty_seats = 532 := by
  sorry

end movie_attendance_l42_42297


namespace pumpkin_weight_difference_l42_42504

theorem pumpkin_weight_difference (Brad: ℕ) (Jessica: ℕ) (Betty: ℕ) 
    (h1 : Brad = 54) 
    (h2 : Jessica = Brad / 2) 
    (h3 : Betty = Jessica * 4) 
    : (Betty - Jessica) = 81 := 
by
  sorry

end pumpkin_weight_difference_l42_42504


namespace complex_point_in_first_quadrant_l42_42780

/-- Prove that the point corresponding to the product of two complex numbers (1 + 3i) and (3 - i) 
    is in the first quadrant in the complex plane. 
    Conditions:
    - z1 = 1 + 3i
    - z2 = 3 - i
    - Expected result: the point (6, 8) is in the first quadrant
-/
noncomputable def z1 : ℂ := 1 + 3 * complex.I
noncomputable def z2 : ℂ := 3 - complex.I

theorem complex_point_in_first_quadrant (z1 z2 : ℂ) (h1 : z1 = 1 + 3 * complex.I) (h2 : z2 = 3 - complex.I) : 
  let z := z1 * z2 in z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_point_in_first_quadrant_l42_42780


namespace max_x_minus_y_l42_42183

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  ∃ k ∈ ℝ, x - y ≤ k ∧ k = 1 + 3 * real.sqrt 2 :=
sorry

end max_x_minus_y_l42_42183


namespace max_value_x_minus_y_l42_42179

theorem max_value_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_value_x_minus_y_l42_42179


namespace karen_starts_late_by_4_minutes_l42_42912

-- Define conditions as Lean 4 variables/constants
noncomputable def karen_speed : ℝ := 60 -- in mph
noncomputable def tom_speed : ℝ := 45 -- in mph
noncomputable def tom_distance : ℝ := 24 -- in miles
noncomputable def karen_lead : ℝ := 4 -- in miles

-- Main theorem statement
theorem karen_starts_late_by_4_minutes : 
  ∃ (minutes_late : ℝ), minutes_late = 4 :=
by
  -- Calculations based on given conditions provided in the problem
  let t := tom_distance / tom_speed -- Time for Tom to drive 24 miles
  let tk := (tom_distance + karen_lead) / karen_speed -- Time for Karen to drive 28 miles
  let time_difference := t - tk -- Time difference between Tom and Karen
  let minutes_late := time_difference * 60 -- Convert time difference to minutes
  existsi minutes_late -- Existential quantifier to state the existence of such a time
  have h : minutes_late = 4 := sorry -- Placeholder for demonstrating equality
  exact h

end karen_starts_late_by_4_minutes_l42_42912


namespace independence_test_purpose_l42_42506

theorem independence_test_purpose:
  ∀ (test: String), test = "independence test" → 
  ∀ (purpose: String), purpose = "to provide the reliability of the relationship between two categorical variables" →
  (test = "independence test" ∧ purpose = "to provide the reliability of the relationship between two categorical variables") :=
by
  intros test h_test purpose h_purpose
  exact ⟨h_test, h_purpose⟩

end independence_test_purpose_l42_42506


namespace range_of_b_for_monotonic_function_l42_42876

theorem range_of_b_for_monotonic_function :
  (∀ x : ℝ, (x^2 + 2 * b * x + b + 2) ≥ 0) ↔ (-1 ≤ b ∧ b ≤ 2) :=
by sorry

end range_of_b_for_monotonic_function_l42_42876


namespace num_distinct_factors_of_36_l42_42026

/-- Definition of the number 36. -/
def n : ℕ := 36

/-- Prime factorization of 36 is 2^2 * 3^2. -/
def prime_factors : List (ℕ × ℕ) := [(2, 2), (3, 2)]

/-- The number of distinct positive factors of 36 is 9. -/
theorem num_distinct_factors_of_36 : ∃ k : ℕ, k = 9 ∧ 
  ∀ d : ℕ, d ∣ n → d > 0 → List.mem d (List.range (n + 1)) :=
by
  sorry

end num_distinct_factors_of_36_l42_42026


namespace cube_volume_doubled_l42_42769

theorem cube_volume_doubled (a : ℝ) (h : a > 0) : 
  ((2 * a)^3 - a^3) / a^3 = 7 :=
by
  sorry

end cube_volume_doubled_l42_42769


namespace calc_expression_l42_42107

theorem calc_expression : (3.242 * 14) / 100 = 0.45388 :=
by
  sorry

end calc_expression_l42_42107


namespace vegetables_sold_mass_correct_l42_42837

-- Definitions based on the problem's conditions
def mass_carrots : ℕ := 15
def mass_zucchini : ℕ := 13
def mass_broccoli : ℕ := 8
def total_mass_vegetables := mass_carrots + mass_zucchini + mass_broccoli
def mass_of_vegetables_sold := total_mass_vegetables / 2

-- Theorem to be proved
theorem vegetables_sold_mass_correct : mass_of_vegetables_sold = 18 := by 
  sorry

end vegetables_sold_mass_correct_l42_42837


namespace arithmetic_mean_equality_l42_42011

variable (x y a b : ℝ)

theorem arithmetic_mean_equality (hx : x ≠ 0) (hy : y ≠ 0) :
  (1 / 2 * ((x + a) / y + (y - b) / x)) = (x^2 + a * x + y^2 - b * y) / (2 * x * y) :=
  sorry

end arithmetic_mean_equality_l42_42011


namespace remainder_of_2_pow_2005_mod_7_l42_42826

theorem remainder_of_2_pow_2005_mod_7 :
  2 ^ 2005 % 7 = 2 :=
sorry

end remainder_of_2_pow_2005_mod_7_l42_42826


namespace angle_B_cond1_area_range_cond1_angle_B_cond2_area_range_cond2_angle_B_cond3_area_range_cond3_l42_42148

variable (a b c A B C : ℝ)

-- Condition 1
def cond1 : Prop := b / a = (Real.cos B + 1) / (Real.sqrt 3 * Real.sin A)

-- Condition 2
def cond2 : Prop := 2 * b * Real.sin A = a * Real.tan B

-- Condition 3
def cond3 : Prop := (c - a = b * Real.cos A - a * Real.cos B)

-- Angle B and area of the triangle for Condition 1
theorem angle_B_cond1 (h : cond1 a b A B) : B = π / 3 := sorry

theorem area_range_cond1 (h : cond1 a b A B) (h_acute : ∀ (X : ℝ), 0 < X ∧ X < π / 2 → ∃ (c : ℝ), c = 2) : 
  ∃ S, S ∈ (Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 3)) := sorry

-- Angle B and area of the triangle for Condition 2
theorem angle_B_cond2 (h : cond2 a b A B) : B = π / 3 := sorry

theorem area_range_cond2 (h : cond2 a b A B) (h_acute : ∀ (X : ℝ), 0 < X ∧ X < π / 2 → ∃ (c : ℝ), c = 2) : 
  ∃ S, S ∈ (Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 3)) := sorry

-- Angle B and area of the triangle for Condition 3
theorem angle_B_cond3 (h : cond3 a b c A B) : B = π / 3 := sorry

theorem area_range_cond3 (h : cond3 a b c A B) (h_acute : ∀ (X : ℝ), 0 < X ∧ X < π / 2 → ∃ (c : ℝ), c = 2) : 
  ∃ S, S ∈ (Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 3)) := sorry

end angle_B_cond1_area_range_cond1_angle_B_cond2_area_range_cond2_angle_B_cond3_area_range_cond3_l42_42148


namespace price_of_coffee_table_l42_42628

-- Define the given values
def price_sofa : ℕ := 1250
def price_armchair : ℕ := 425
def num_armchairs : ℕ := 2
def total_invoice : ℕ := 2430

-- Define the target value (price of the coffee table)
def price_coffee_table : ℕ := 330

-- The theorem to prove
theorem price_of_coffee_table :
  total_invoice = price_sofa + num_armchairs * price_armchair + price_coffee_table :=
by sorry

end price_of_coffee_table_l42_42628


namespace part1_part2_l42_42429

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l42_42429


namespace rosy_has_14_fish_l42_42801

-- Define the number of Lilly's fish
def lilly_fish : ℕ := 10

-- Define the total number of fish
def total_fish : ℕ := 24

-- Define the number of Rosy's fish, which we need to prove equals 14
def rosy_fish : ℕ := total_fish - lilly_fish

-- Prove that Rosy has 14 fish
theorem rosy_has_14_fish : rosy_fish = 14 := by
  sorry

end rosy_has_14_fish_l42_42801


namespace problem_exists_integers_a_b_c_d_l42_42496

theorem problem_exists_integers_a_b_c_d :
  ∃ (a b c d : ℤ), 
  |a| > 1000000 ∧ |b| > 1000000 ∧ |c| > 1000000 ∧ |d| > 1000000 ∧
  (1 / (a:ℚ) + 1 / (b:ℚ) + 1 / (c:ℚ) + 1 / (d:ℚ) = 1 / (a * b * c * d : ℚ)) :=
sorry

end problem_exists_integers_a_b_c_d_l42_42496


namespace min_points_tenth_game_l42_42774

-- Defining the scores for each segment of games
def first_five_games : List ℕ := [18, 15, 13, 17, 19]
def next_four_games : List ℕ := [14, 20, 12, 21]

-- Calculating the total score after 9 games
def total_score_after_nine_games : ℕ := first_five_games.sum + next_four_games.sum

-- Defining the required total points after 10 games for an average greater than 17
def required_total_points := 171

-- Proving the number of points needed in the 10th game
theorem min_points_tenth_game (s₁ s₂ : List ℕ) (h₁ : s₁ = first_five_games) (h₂ : s₂ = next_four_games) :
    s₁.sum + s₂.sum + x ≥ required_total_points → x ≥ 22 :=
  sorry

end min_points_tenth_game_l42_42774


namespace find_cos_7theta_l42_42035

theorem find_cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = 1105 / 16384 :=
by
  sorry

end find_cos_7theta_l42_42035


namespace sum_of_roots_of_equation_l42_42765

theorem sum_of_roots_of_equation :
  (∀ x : ℝ, (x + 3) * (x - 4) = 22 → ∃ a b : ℝ, (x - a) * (x - b) = 0 ∧ a + b = 1) :=
by
  sorry

end sum_of_roots_of_equation_l42_42765


namespace halfway_fraction_l42_42684

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/7) : (a + b) / 2 = 41/56 :=
by
  sorry

end halfway_fraction_l42_42684


namespace fibonacci_odd_index_not_divisible_by_4k_plus_3_l42_42243

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_odd_index_not_divisible_by_4k_plus_3 (n k : ℕ) (p : ℕ) (h : p = 4 * k + 3) : ¬ (p ∣ fibonacci (2 * n - 1)) :=
by
  sorry

end fibonacci_odd_index_not_divisible_by_4k_plus_3_l42_42243


namespace initial_cows_l42_42957

theorem initial_cows {D C : ℕ}
  (h1 : C = 2 * D)
  (h2 : 161 = (3 * C) / 4 + D / 4) :
  C = 184 :=
by
  sorry

end initial_cows_l42_42957


namespace debate_team_boys_l42_42664

theorem debate_team_boys (total_groups : ℕ) (members_per_group : ℕ) (num_girls : ℕ) (total_members : ℕ) :
  total_groups = 8 →
  members_per_group = 4 →
  num_girls = 4 →
  total_members = total_groups * members_per_group →
  total_members - num_girls = 28 :=
by
  sorry

end debate_team_boys_l42_42664


namespace halfway_fraction_l42_42679

theorem halfway_fraction (a b : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 7) :
  ((a + b) / 2) = 41 / 56 :=
by
  rw [h_a, h_b]
  sorry

end halfway_fraction_l42_42679


namespace worker_assignment_l42_42985

theorem worker_assignment (x : ℕ) (y : ℕ) 
  (h1 : x + y = 90)
  (h2 : 2 * 15 * x = 3 * 8 * y) : 
  (x = 40 ∧ y = 50) := by
  sorry

end worker_assignment_l42_42985


namespace gcd_fact_plus_two_l42_42587

theorem gcd_fact_plus_two (n m : ℕ) (h1 : n = 6) (h2 : m = 8) :
  Nat.gcd (n.factorial + 2) (m.factorial + 2) = 2 :=
  sorry

end gcd_fact_plus_two_l42_42587


namespace intersection_M_N_l42_42640

def M : Set ℝ := { x | x^2 + x - 6 < 0 }
def N : Set ℝ := { x | |x - 1| ≤ 2 }

theorem intersection_M_N :
  M ∩ N = { x | -1 ≤ x ∧ x < 2 } :=
sorry

end intersection_M_N_l42_42640


namespace f_identity_l42_42065

def f (x : ℝ) : ℝ := (2 * x + 1)^5 - 5 * (2 * x + 1)^4 + 10 * (2 * x + 1)^3 - 10 * (2 * x + 1)^2 + 5 * (2 * x + 1) - 1

theorem f_identity (x : ℝ) : f x = 32 * x^5 :=
by
  -- the proof is omitted
  sorry

end f_identity_l42_42065


namespace brett_red_marbles_l42_42142

variables (r b : ℕ)

-- Define the conditions
axiom h1 : b = r + 24
axiom h2 : b = 5 * r

theorem brett_red_marbles : r = 6 :=
by
  sorry

end brett_red_marbles_l42_42142


namespace ball_arrangement_count_l42_42269

theorem ball_arrangement_count :
  let total_balls := 9
  let red_balls := 2
  let yellow_balls := 3
  let white_balls := 4
  let arrangements := Nat.factorial total_balls / (Nat.factorial red_balls * Nat.factorial yellow_balls * Nat.factorial white_balls)
  arrangements = 1260 :=
by
  sorry

end ball_arrangement_count_l42_42269


namespace total_cars_l42_42482

-- Definitions for the conditions
def cathy_cars : ℕ := 5
def lindsey_cars : ℕ := cathy_cars + 4
def carol_cars : ℕ := 2 * cathy_cars
def susan_cars : ℕ := carol_cars - 2

-- Lean theorem statement
theorem total_cars : cathy_cars + lindsey_cars + carol_cars + susan_cars = 32 := by
  sorry

end total_cars_l42_42482


namespace parallelogram_height_l42_42866

theorem parallelogram_height (A : ℝ) (b : ℝ) (h : ℝ) (h1 : A = 320) (h2 : b = 20) :
  h = A / b → h = 16 := by
  sorry

end parallelogram_height_l42_42866


namespace trigon_expr_correct_l42_42725

noncomputable def trigon_expr : ℝ :=
  1 / Real.sin (Real.pi / 6) - 4 * Real.sin (Real.pi / 3)

theorem trigon_expr_correct : trigon_expr = 2 - 2 * Real.sqrt 3 := by
  sorry

end trigon_expr_correct_l42_42725


namespace find_ellipse_eq_product_of_tangent_slopes_l42_42163

variables {a b : ℝ} {x y x0 y0 : ℝ}

-- Given conditions
def ellipse (a b : ℝ) := a > 0 ∧ b > 0 ∧ a > b ∧ (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → y = 1 ∧ y = 3 / 2)

def eccentricity (a b : ℝ) := b = (1 / 2) * a

def passes_through (x y : ℝ) := x = 1 ∧ y = 3 / 2

-- Part 1: Prove the equation of the ellipse
theorem find_ellipse_eq (a b : ℝ) (h_ellipse : ellipse a b) (h_eccentricity : eccentricity a b) (h_point : passes_through 1 (3/2)) :
    (x^2) / 4 + (y^2) / 3 = 1 :=
sorry

-- Circle equation definition
def circle (x y : ℝ) := x^2 + y^2 = 7

-- Part 2: Prove the product of the slopes of the tangent lines is constant
theorem product_of_tangent_slopes (P : ℝ × ℝ) (h_circle : circle P.1 P.2) : 
    ∀ k1 k2 : ℝ, (4 - P.1^2) * k1^2 + 6 * P.1 * P.2 * k1 + 3 - P.2^2 = 0 → 
    (4 - P.1^2) * k2^2 + 6 * P.1 * P.2 * k2 + 3 - P.2^2 = 0 → k1 * k2 = -1 :=
sorry

end find_ellipse_eq_product_of_tangent_slopes_l42_42163


namespace evaluate_expression_l42_42581

-- Definition of the conditions
def a : ℕ := 15
def b : ℕ := 19
def c : ℕ := 13

-- Problem statement
theorem evaluate_expression :
  (225 * (1 / a - 1 / b) + 361 * (1 / b - 1 / c) + 169 * (1 / c - 1 / a))
  /
  (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)) = a + b + c :=
by
  sorry

end evaluate_expression_l42_42581


namespace k_value_if_perfect_square_l42_42612

theorem k_value_if_perfect_square (k : ℤ) (x : ℝ) (h : ∃ (a : ℝ), x^2 + k * x + 25 = a^2) : k = 10 ∨ k = -10 := by
  sorry

end k_value_if_perfect_square_l42_42612


namespace maximum_value_of_x_minus_y_l42_42165

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
begin
  sorry
end

end maximum_value_of_x_minus_y_l42_42165


namespace collatz_conjecture_probability_l42_42651

noncomputable def collatz_sequence := [10, 5, 16, 8, 4, 2, 1]

def odd_numbers := collatz_sequence.filter (λ n, n % 2 = 1)

def choose (n k : ℕ) : ℕ := n.choose k

def probability_of_two_odd_numbers_3n_plus_1_conjecture
    (seq : List ℕ) (evens odds : List ℕ) : Prop :=
  let evens := seq.filter (λ n, n % 2 = 0)
  let odds := seq.filter (λ n, n % 2 = 1)
  (evens.length = 5) ∧
  (odds.length = 2) ∧
  (choose odds.length 2) / (choose seq.length 2) = 1 / 21

theorem collatz_conjecture_probability :
  probability_of_two_odd_numbers_3n_plus_1_conjecture collatz_sequence
  (collatz_sequence.filter (λ n, n % 2 = 0))
  odd_numbers :=
by
  sorry

end collatz_conjecture_probability_l42_42651


namespace proof_problem_l42_42974

noncomputable def initialEfficiencyOfOneMan : ℕ := sorry
noncomputable def initialEfficiencyOfOneWoman : ℕ := sorry
noncomputable def totalWork : ℕ := sorry

-- Condition (1): 10 men and 15 women together can complete the work in 6 days.
def condition1 := 10 * initialEfficiencyOfOneMan + 15 * initialEfficiencyOfOneWoman = totalWork / 6

-- Condition (2): The efficiency of men to complete the work decreases by 5% every day.
-- This condition is not directly measurable to our proof but noted as additional info.

-- Condition (3): The efficiency of women to complete the work increases by 3% every day.
-- This condition is not directly measurable to our proof but noted as additional info.

-- Condition (4): It takes 100 days for one man alone to complete the same work at his initial efficiency.
def condition4 := initialEfficiencyOfOneMan = totalWork / 100

-- Define the days required for one woman alone to complete the work at her initial efficiency.
noncomputable def daysForWomanToCompleteWork : ℕ := 225

-- Mathematically equivalent proof problem
theorem proof_problem : 
  condition1 ∧ condition4 → (totalWork / daysForWomanToCompleteWork = initialEfficiencyOfOneWoman) :=
by
  sorry

end proof_problem_l42_42974


namespace simplify_frac_l42_42083

theorem simplify_frac :
  (1 / (1 / (Real.sqrt 3 + 2) + 2 / (Real.sqrt 5 - 2))) = 
  (Real.sqrt 3 - 2 * Real.sqrt 5 - 2) :=
by
  sorry

end simplify_frac_l42_42083


namespace correct_new_encoding_l42_42530

def oldMessage : String := "011011010011"
def newMessage : String := "211221121"

def oldEncoding : Char → String
| 'A' => "11"
| 'B' => "011"
| 'C' => "0"
| _ => ""

def newEncoding : Char → String
| 'A' => "21"
| 'B' => "122"
| 'C' => "1"
| _ => ""

-- Define the decoded message based on the old encoding
def decodeOldMessage : String :=
  let rec decode (msg : String) : String :=
    if msg = "" then "" else
    if msg.endsWith "11" then decode (msg.dropRight 2) ++ "A"
    else if msg.endsWith "011" then decode (msg.dropRight 3) ++ "B"
    else if msg.endsWith "0" then decode (msg.dropRight 1) ++ "C"
    else ""
  decode oldMessage

-- Define the encoded message based on the new encoding
def encodeNewMessage (decodedMsg : String) : String :=
  decodedMsg.toList.map newEncoding |> String.join

-- Proof statement to verify the encoding and decoding
theorem correct_new_encoding : encodeNewMessage decodeOldMessage = newMessage := by
  sorry

end correct_new_encoding_l42_42530


namespace part1_solution_set_part2_range_of_a_l42_42418

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l42_42418


namespace polynomial_solution_l42_42085

theorem polynomial_solution (x : ℂ) (h : x^3 + x^2 + x + 1 = 0) : x^4 + 2 * x^3 + 2 * x^2 + 2 * x + 1 = 0 := 
by {
  sorry
}

end polynomial_solution_l42_42085


namespace brett_red_marbles_l42_42143

variables (r b : ℕ)

-- Define the conditions
axiom h1 : b = r + 24
axiom h2 : b = 5 * r

theorem brett_red_marbles : r = 6 :=
by
  sorry

end brett_red_marbles_l42_42143


namespace length_width_difference_l42_42131

noncomputable def width : ℝ := Real.sqrt (588 / 8)
noncomputable def length : ℝ := 4 * width
noncomputable def difference : ℝ := length - width

theorem length_width_difference : difference = 25.722 := by
  sorry

end length_width_difference_l42_42131


namespace cos_seven_theta_l42_42048

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (7 * θ) = -160481 / 2097152 := by
  sorry

end cos_seven_theta_l42_42048


namespace license_plates_possible_l42_42982

open Function Nat

theorem license_plates_possible :
  let characters := ['B', 'C', 'D', '1', '2', '2', '5']
  let license_plate_length := 4
  let plate_count_with_two_twos := (choose 4 2) * (choose 5 2 * 2!)
  let plate_count_with_one_two := (choose 4 1) * (choose 5 3 * 3!)
  let plate_count_with_no_twos := (choose 5 4) * 4!
  let plate_count_with_three_twos := (choose 4 3) * (choose 4 1)
  plate_count_with_two_twos + plate_count_with_one_two + plate_count_with_no_twos + plate_count_with_three_twos = 496 := 
  sorry

end license_plates_possible_l42_42982


namespace axis_of_symmetry_and_vertex_l42_42260

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

theorem axis_of_symmetry_and_vertex :
  (∃ (a : ℝ), (f a = -2 * (a - 1)^2 + 3) ∧ a = 1) ∧ ∃ v, (v = (1, 3) ∧ ∀ x, f x = -2 * (x - 1)^2 + 3) :=
sorry

end axis_of_symmetry_and_vertex_l42_42260


namespace ab_relationship_l42_42332

theorem ab_relationship (a b : ℝ) (n : ℕ) (h1 : a^n = a + 1) (h2 : b^(2*n) = b + 3*a) (h3 : n ≥ 2) (h4 : 0 < a) (h5 : 0 < b) :
  a > b ∧ a > 1 ∧ b > 1 :=
sorry

end ab_relationship_l42_42332


namespace boys_belong_to_other_communities_l42_42904

/-- In a school of 300 boys, if 44% are Muslims, 28% are Hindus, and 10% are Sikhs,
then the number of boys belonging to other communities is 54. -/
theorem boys_belong_to_other_communities
  (total_boys : ℕ)
  (percentage_muslims percentage_hindus percentage_sikhs : ℕ)
  (b : total_boys = 300)
  (m : percentage_muslims = 44)
  (h : percentage_hindus = 28)
  (s : percentage_sikhs = 10) :
  total_boys * ((100 - (percentage_muslims + percentage_hindus + percentage_sikhs)) / 100) = 54 := 
sorry

end boys_belong_to_other_communities_l42_42904


namespace required_more_visits_l42_42014

-- Define the conditions
def n := 395
def m := 2
def v1 := 135
def v2 := 112
def v3 := 97

-- Define the target statement
theorem required_more_visits : (n * m) - (v1 + v2 + v3) = 446 := by
  sorry

end required_more_visits_l42_42014


namespace arithmetic_sequence_property_l42_42219

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_property (h1 : is_arithmetic_sequence a)
                                     (h2 : a 3 + a 11 = 40) :
  a 6 - a 7 + a 8 = 20 :=
by
  sorry

end arithmetic_sequence_property_l42_42219


namespace total_cars_l42_42477

-- Definitions of the conditions
def cathy_cars : Nat := 5

def carol_cars : Nat := 2 * cathy_cars

def susan_cars : Nat := carol_cars - 2

def lindsey_cars : Nat := cathy_cars + 4

-- The theorem statement (problem)
theorem total_cars : cathy_cars + carol_cars + susan_cars + lindsey_cars = 32 :=
by
  -- sorry is added to skip the proof
  sorry

end total_cars_l42_42477


namespace maximum_distance_l42_42714

-- Given conditions for the problem.
def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def gasoline : ℝ := 23

-- Problem statement: prove the maximum distance on highway mileage.
theorem maximum_distance : highway_mpg * gasoline = 280.6 :=
sorry

end maximum_distance_l42_42714


namespace find_f_of_3_l42_42336

theorem find_f_of_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 1) = 3 * x - 5) : f 3 = -2 :=
by
  sorry

end find_f_of_3_l42_42336


namespace game_promises_total_hours_l42_42792

open Real

noncomputable def total_gameplay_hours (T : ℝ) : Prop :=
  let boring_gameplay := 0.80 * T
  let enjoyable_gameplay := 0.20 * T
  let expansion_hours := 30
  (enjoyable_gameplay + expansion_hours = 50) → (T = 100)

theorem game_promises_total_hours (T : ℝ) : total_gameplay_hours T :=
  sorry

end game_promises_total_hours_l42_42792


namespace num_factors_36_l42_42027

theorem num_factors_36 : ∀ (n : ℕ), n = 36 → (∃ (a b : ℕ), 36 = 2^a * 3^b ∧ a = 2 ∧ b = 2 ∧ (a + 1) * (b + 1) = 9) :=
by
  sorry

end num_factors_36_l42_42027


namespace cos_theta_seven_l42_42034

theorem cos_theta_seven {θ : ℝ} (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -8383 / 98304 :=
by
  sorry

end cos_theta_seven_l42_42034


namespace james_total_carrot_sticks_l42_42061

theorem james_total_carrot_sticks : 
  ∀ (before_dinner after_dinner : Nat), before_dinner = 22 → after_dinner = 15 → 
  before_dinner + after_dinner = 37 := 
by
  intros before_dinner after_dinner h1 h2
  rw [h1, h2]
  rfl

end james_total_carrot_sticks_l42_42061


namespace xy_sum_l42_42614

-- Define the problem conditions
variable (x y : ℚ)
variable (h1 : 1 / x + 1 / y = 4)
variable (h2 : 1 / x - 1 / y = -8)

-- Define the theorem to prove
theorem xy_sum : x + y = -1 / 3 := by
  sorry

end xy_sum_l42_42614


namespace problem_statement_l42_42727

open Finset

variable {α : Type*}

def T : Finset α := {a, b, c, d, e, f}

noncomputable def count_subsets_with_intersection (T : Finset α) (k : ℕ) : ℕ :=
  (choose T.card k) * (2 ^ (T.card - k)) / 2

theorem problem_statement : count_subsets_with_intersection T 3 = 80 :=
by sorry

end problem_statement_l42_42727


namespace arithmetic_mean_l42_42535

theorem arithmetic_mean (a b : ℚ) (h1 : a = 3/7) (h2 : b = 5/9) :
  (a + b) / 2 = 31/63 := 
by 
  sorry

end arithmetic_mean_l42_42535


namespace equal_segment_of_bisectors_and_circumcircle_l42_42090

theorem equal_segment_of_bisectors_and_circumcircle
  (A B C L K N : Point)
  (circumcircle : Circle)
  (h_circumcircle : Circumcircle A B C circumcircle)
  (h_BL : AngleBisector B L)
  (h_BL_extends : ExtendsTo BL circumcircle K)
  (h_ext_angle_bisector : ExternalAngleBisector B (line_through C A) N)
  (h_LN_extends : ExtendsTo LN N)
  (h_BK_BN_eq : BK = BN) :
  LN = 2 * circumcircle.radius := 
sorry

end equal_segment_of_bisectors_and_circumcircle_l42_42090


namespace max_x_minus_y_l42_42192

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l42_42192


namespace max_x_minus_y_l42_42185

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  (∃ z : ℝ, z = x - y ∧ (∀ w : ℝ, w = x - y → w ≤ 1 + 3 * real.sqrt 2) ∧ (∃ u : ℝ, u = x - y ∧ u = 1 + 3 * real.sqrt 2)) :=
sorry

end max_x_minus_y_l42_42185


namespace max_non_overlapping_triangles_l42_42716

variable (L : ℝ) (n : ℕ)
def equilateral_triangle (L : ℝ) := true   -- Placeholder definition for equilateral triangle 
def non_overlapping_interior := true        -- Placeholder definition for non-overlapping condition
def unit_triangle_orientation_shift := true -- Placeholder for orientation condition

theorem max_non_overlapping_triangles (L_pos : 0 < L)
                                    (h1 : equilateral_triangle L)
                                    (h2 : ∀ i, i < n → non_overlapping_interior)
                                    (h3 : ∀ i, i < n → unit_triangle_orientation_shift) :
                                    n ≤ (2 : ℝ) / 3 * L^2 := 
by 
  sorry

end max_non_overlapping_triangles_l42_42716


namespace total_cars_all_own_l42_42478

theorem total_cars_all_own :
  ∀ (C L S K : ℕ), 
  (C = 5) →
  (L = C + 4) →
  (K = 2 * C) →
  (S = K - 2) →
  (C + L + K + S = 32) :=
by
  intros C L S K
  intro hC
  intro hL
  intro hK
  intro hS
  sorry

end total_cars_all_own_l42_42478


namespace work_ratio_l42_42294

theorem work_ratio (r : ℕ) (w : ℕ) (m₁ m₂ d₁ d₂ : ℕ)
  (h₁ : m₁ = 5) 
  (h₂ : d₁ = 15) 
  (h₃ : m₂ = 3) 
  (h₄ : d₂ = 25)
  (h₅ : w = (m₁ * r * d₁) + (m₂ * r * d₂)) :
  ((m₁ * r * d₁):ℚ) / (w:ℚ) = 1 / 2 := by
  sorry

end work_ratio_l42_42294


namespace find_leftover_amount_l42_42465

open Nat

def octal_to_decimal (n : ℕ) : ℕ :=
  let digits := [5, 5, 5, 5]
  List.foldr (λ (d : ℕ) (acc : ℕ) => d + 8 * acc) 0 digits

def expenses_total : ℕ := 1200 + 800 + 400

theorem find_leftover_amount : 
  let initial_amount := octal_to_decimal 5555
  let final_amount := initial_amount - expenses_total
  final_amount = 525 := by
    sorry

end find_leftover_amount_l42_42465


namespace find_f_of_3_l42_42337

theorem find_f_of_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 1) = 3 * x - 5) : f 3 = -2 :=
by
  sorry

end find_f_of_3_l42_42337


namespace total_games_played_l42_42821

theorem total_games_played (n : ℕ) (h: n = 12) : ∑ (2 : ℕ) (choose 12 2 = 66 := by {
  sorry
}

end total_games_played_l42_42821


namespace initial_pinecones_l42_42955

theorem initial_pinecones (P : ℝ) :
  (0.20 * P + 2 * 0.20 * P + 0.25 * (0.40 * P) = 0.70 * P - 0.10 * P) ∧ (0.30 * P = 600) → P = 2000 :=
by
  intro h
  sorry

end initial_pinecones_l42_42955


namespace range_of_a_l42_42442

variable {R : Type*} [LinearOrderedField R]

def setA (a : R) : Set R := {x | x^2 - 2*x + a ≤ 0}

def setB : Set R := {x | x^2 - 3*x + 2 ≤ 0}

theorem range_of_a (a : R) (h : setB ⊆ setA a) : a ≤ 0 := sorry

end range_of_a_l42_42442


namespace triangle_perimeter_from_medians_l42_42746

theorem triangle_perimeter_from_medians (m1 m2 m3 : ℕ) (h1 : m1 = 3) (h2 : m2 = 4) (h3 : m3 = 6) :
  ∃ (p : ℕ), p = 26 :=
by sorry

end triangle_perimeter_from_medians_l42_42746


namespace red_marbles_count_l42_42334

theorem red_marbles_count (W Y G R : ℕ) (total_marbles : ℕ) 
(h1 : total_marbles = 50)
(h2 : W = 50 / 2)
(h3 : Y = 12)
(h4 : G = 12 - (12 * 0.5))
(h5 : W + Y + G + R = total_marbles)
: R = 7 :=
sorry

end red_marbles_count_l42_42334


namespace compare_abc_l42_42230

noncomputable def a : ℝ := 4 ^ (Real.log 2 / Real.log 3)
noncomputable def b : ℝ := 4 ^ (Real.log 6 / (2 * Real.log 3))
noncomputable def c : ℝ := 2 ^ (Real.sqrt 5)

theorem compare_abc : c > b ∧ b > a := 
by
  sorry

end compare_abc_l42_42230


namespace part1_solution_part2_solution_l42_42395

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l42_42395


namespace part1_solution_set_part2_range_a_l42_42394

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l42_42394


namespace solve_inequality_l42_42811

theorem solve_inequality (x : ℝ) : 
  3 * (2 * x - 1) - 2 * (x + 1) ≤ 1 → x ≤ 3 / 2 :=
by
  sorry

end solve_inequality_l42_42811


namespace possible_integer_roots_l42_42710

-- Define the general polynomial
def polynomial (b2 b1 : ℤ) (x : ℤ) : ℤ := x ^ 3 + b2 * x ^ 2 + b1 * x - 30

-- Statement: Prove the set of possible integer roots includes exactly the divisors of -30
theorem possible_integer_roots (b2 b1 : ℤ) :
  {r : ℤ | polynomial b2 b1 r = 0} = 
  {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30} :=
sorry

end possible_integer_roots_l42_42710


namespace steers_cows_unique_solution_l42_42284

-- Definition of the problem
def steers_and_cows_problem (s c : ℕ) : Prop :=
  25 * s + 26 * c = 1000 ∧ s > 0 ∧ c > 0

-- The theorem statement to be proved
theorem steers_cows_unique_solution :
  ∃! (s c : ℕ), steers_and_cows_problem s c ∧ c > s :=
sorry

end steers_cows_unique_solution_l42_42284


namespace part1_part2_l42_42424

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l42_42424


namespace max_x_minus_y_l42_42191

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l42_42191


namespace beads_to_remove_l42_42077

-- Definitions for the conditions given in the problem
def initial_blue_beads : Nat := 49
def initial_red_bead : Nat := 1
def total_initial_beads : Nat := initial_blue_beads + initial_red_bead
def target_blue_percentage : Nat := 90 -- percentage

-- The goal to prove
theorem beads_to_remove (initial_blue_beads : Nat) (initial_red_bead : Nat)
    (target_blue_percentage : Nat) : Nat :=
    let target_total_beads := (initial_red_bead * 100) / target_blue_percentage
    total_initial_beads - target_total_beads
-- Expected: beads_to_remove 49 1 90 = 40

example : beads_to_remove initial_blue_beads initial_red_bead target_blue_percentage = 40 := by 
    sorry

end beads_to_remove_l42_42077


namespace total_cars_all_own_l42_42480

theorem total_cars_all_own :
  ∀ (C L S K : ℕ), 
  (C = 5) →
  (L = C + 4) →
  (K = 2 * C) →
  (S = K - 2) →
  (C + L + K + S = 32) :=
by
  intros C L S K
  intro hC
  intro hL
  intro hK
  intro hS
  sorry

end total_cars_all_own_l42_42480


namespace smallest_b_value_l42_42255

theorem smallest_b_value (a b : ℕ) (h1 : a - b = 8) (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : b = 3 := sorry

end smallest_b_value_l42_42255


namespace fraction_halfway_between_fraction_halfway_between_l42_42681

theorem fraction_halfway_between : (3/4 : ℚ) < (5/7 : ℚ) :=
by linarith

theorem fraction_halfway_between : (41 / 56 : ℚ) = (1 / 2) * ((3 / 4) + (5 / 7)) :=
by sorry

end fraction_halfway_between_fraction_halfway_between_l42_42681


namespace divisibility_of_n_squared_plus_n_plus_two_l42_42341

-- Definition: n is a natural number.
def n (n : ℕ) : Prop := True

-- Theorem: For any natural number n, n^2 + n + 2 is always divisible by 2, but not necessarily divisible by 5.
theorem divisibility_of_n_squared_plus_n_plus_two (n : ℕ) : 
  (∃ k : ℕ, n^2 + n + 2 = 2 * k) ∧ (¬ ∃ m : ℕ, n^2 + n + 2 = 5 * m) :=
by
  sorry

end divisibility_of_n_squared_plus_n_plus_two_l42_42341


namespace hexagon_division_ratio_l42_42704

theorem hexagon_division_ratio
  (hex_area : ℝ)
  (hexagon : ∀ (A B C D E F : ℝ), hex_area = 8)
  (line_PQ_splits : ∀ (above_area below_area : ℝ), above_area = 4 ∧ below_area = 4)
  (below_PQ : ℝ)
  (unit_square_area : ∀ (unit_square : ℝ), unit_square = 1)
  (triangle_base : ℝ)
  (triangle_height : ℝ)
  (triangle_area : ∀ (base height : ℝ), triangle_base = 4 ∧ (base * height) / 2 = 3)
  (XQ QY : ℝ)
  (bases_sum : ∀ (XQ QY : ℝ), XQ + QY = 4) :
  XQ / QY = 2 / 3 :=
sorry

end hexagon_division_ratio_l42_42704


namespace part1_solution_set_part2_range_of_a_l42_42379

-- Part 1
theorem part1_solution_set (x : ℝ) : (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x + 3| > -a) ↔ (a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_of_a_l42_42379


namespace black_area_after_six_transformations_l42_42717

noncomputable def remaining_fraction_after_transformations (initial_fraction : ℚ) (transforms : ℕ) (reduction_factor : ℚ) : ℚ :=
  reduction_factor ^ transforms * initial_fraction

theorem black_area_after_six_transformations :
  remaining_fraction_after_transformations 1 6 (2 / 3) = 64 / 729 := 
by
  sorry

end black_area_after_six_transformations_l42_42717


namespace cos_seven_theta_l42_42049

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (7 * θ) = -160481 / 2097152 := by
  sorry

end cos_seven_theta_l42_42049


namespace number_of_balls_sold_l42_42490

-- Let n be the number of balls sold
variable (n : ℕ)

-- The given conditions
def selling_price : ℕ := 720
def cost_price_per_ball : ℕ := 60
def loss := 5 * cost_price_per_ball

-- Prove that if the selling price of 'n' balls is Rs. 720 and 
-- the loss is equal to the cost price of 5 balls, then the 
-- number of balls sold (n) is 17.
theorem number_of_balls_sold (h1 : selling_price = 720) 
                             (h2 : cost_price_per_ball = 60) 
                             (h3 : loss = 5 * cost_price_per_ball) 
                             (hsale : n * cost_price_per_ball - selling_price = loss) : 
  n = 17 := 
by
  sorry

end number_of_balls_sold_l42_42490


namespace find_cos_7theta_l42_42036

theorem find_cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = 1105 / 16384 :=
by
  sorry

end find_cos_7theta_l42_42036


namespace men_wages_l42_42293

def men := 5
def women := 5
def boys := 7
def total_wages := 90
def wage_man := 7.5

theorem men_wages (men women boys : ℕ) (total_wages wage_man : ℝ)
  (h1 : 5 = women) (h2 : women = boys) (h3 : 5 * wage_man + 1 * wage_man + 7 * wage_man = total_wages) :
  5 * wage_man = 37.5 :=
  sorry

end men_wages_l42_42293


namespace find_a_l42_42202

open Real

def ellipse (x y a : ℝ) : Prop := x^2 / 6 + y^2 / (a^2) = 1
def hyperbola (x y a : ℝ) : Prop := x^2 / a - y^2 / 4 = 1

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, ellipse x y a → hyperbola x y a → true) → a = 1 :=
by 
  sorry

end find_a_l42_42202


namespace part1_l42_42436

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

theorem part1 {x : ℝ} : f x ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) := by
  sorry

end part1_l42_42436


namespace sum_of_products_equal_l42_42806

theorem sum_of_products_equal 
  (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℝ)
  (h1 : a1 + a2 + a3 = b1 + b2 + b3)
  (h2 : b1 + b2 + b3 = c1 + c2 + c3)
  (h3 : c1 + c2 + c3 = a1 + b1 + c1)
  (h4 : a1 + b1 + c1 = a2 + b2 + c2)
  (h5 : a2 + b2 + c2 = a3 + b3 + c3) :
  a1 * b1 * c1 + a2 * b2 * c2 + a3 * b3 * c3 = a1 * a2 * a3 + b1 * b2 * b3 + c1 * c2 * c3 :=
by 
  sorry

end sum_of_products_equal_l42_42806


namespace part1_part2_l42_42407

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) :=
by
  sorry

end part1_part2_l42_42407


namespace karen_starts_late_l42_42914

def karen_speed := 60 -- Karen's speed in mph
def tom_speed := 45 -- Tom's speed in mph
def tom_distance := 24 -- Distance Tom drives in miles
def karen_lead := 4 -- Distance by which Karen beats Tom in miles

theorem karen_starts_late : 
  let t := tom_distance / tom_speed in -- Time Tom drives
  let t_k := (tom_distance + karen_lead) / karen_speed in -- Time Karen drives
  (t - t_k) * 60 = 4 := -- The time difference in minutes is 4
by
  sorry

end karen_starts_late_l42_42914


namespace find_f3_l42_42339

theorem find_f3 {f : ℝ → ℝ} (h : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 5) : f 3 = -2 :=
by
  sorry

end find_f3_l42_42339


namespace cos_seven_theta_l42_42040

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = -953 / 1024 :=
by sorry

end cos_seven_theta_l42_42040


namespace enclosed_area_l42_42856

noncomputable def calculateArea : ℝ :=
  ∫ (x : ℝ) in (1 / 2)..2, 1 / x

theorem enclosed_area : calculateArea = 2 * Real.log 2 :=
by
  sorry

end enclosed_area_l42_42856


namespace find_ab_l42_42750

theorem find_ab (a b : ℤ) :
  (∀ x : ℤ, a * (x - 3) + b * (3 * x + 1) = 5 * (x + 1)) →
  a = -1 ∧ b = 2 :=
by
  sorry

end find_ab_l42_42750


namespace u_less_than_v_l42_42511

noncomputable def f (u : ℝ) := (u + u^2 + u^3 + u^4 + u^5 + u^6 + u^7 + u^8) + 10 * u^9
noncomputable def g (v : ℝ) := (v + v^2 + v^3 + v^4 + v^5 + v^6 + v^7 + v^8 + v^9 + v^10) + 10 * v^11

theorem u_less_than_v
  (u v : ℝ)
  (hu : f u = 8)
  (hv : g v = 8) :
  u < v := 
sorry

end u_less_than_v_l42_42511


namespace average_of_remaining_two_l42_42288

theorem average_of_remaining_two (a1 a2 a3 a4 a5 a6 : ℝ)
    (h_avg6 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 3.95)
    (h_avg2_1 : (a1 + a2) / 2 = 3.4)
    (h_avg2_2 : (a3 + a4) / 2 = 3.85) :
    (a5 + a6) / 2 = 4.6 := 
sorry

end average_of_remaining_two_l42_42288


namespace find_correct_result_l42_42109

noncomputable def correct_result : Prop :=
  ∃ (x : ℝ), (-1.25 * x - 0.25 = 1.25 * x) ∧ (-1.25 * x = 0.125)

theorem find_correct_result : correct_result :=
  sorry

end find_correct_result_l42_42109


namespace integer_pairs_solution_l42_42732

def is_satisfied_solution (x y : ℤ) : Prop :=
  x^2 + y^2 = x + y + 2

theorem integer_pairs_solution :
  ∀ (x y : ℤ), is_satisfied_solution x y ↔ (x, y) = (-1, 0) ∨ (x, y) = (-1, 1) ∨ (x, y) = (0, -1) ∨ (x, y) = (0, 2) ∨ (x, y) = (1, -1) ∨ (x, y) = (1, 2) ∨ (x, y) = (2, 0) ∨ (x, y) = (2, 1) :=
by
  sorry

end integer_pairs_solution_l42_42732


namespace complex_quadrant_check_l42_42785

theorem complex_quadrant_check : 
  let z := (1 : ℂ) + 3 * complex.i
  let w := (3 : ℂ) - complex.i
  let result := z * w
  result.re > 0 ∧ result.im > 0 :=
by sorry

end complex_quadrant_check_l42_42785


namespace maximum_value_of_x_minus_y_l42_42171

noncomputable def max_value_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : ℝ :=
1 + 3 * Real.sqrt 2

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
    x - y ≤ max_value_x_minus_y x y h := sorry

end maximum_value_of_x_minus_y_l42_42171


namespace part1_part2_l42_42411

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) :=
by
  sorry

end part1_part2_l42_42411


namespace n_squared_plus_m_squared_odd_implies_n_plus_m_not_even_l42_42050

theorem n_squared_plus_m_squared_odd_implies_n_plus_m_not_even (n m : ℤ) (h : (n^2 + m^2) % 2 = 1) : (n + m) % 2 ≠ 0 := by
  sorry

end n_squared_plus_m_squared_odd_implies_n_plus_m_not_even_l42_42050


namespace odd_function_expression_l42_42881

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_expression (x : ℝ) (h1 : x < 0 → f x = x^2 - x) (h2 : ∀ x, f (-x) = -f x) (h3 : 0 < x) :
  f x = -x^2 - x :=
sorry

end odd_function_expression_l42_42881


namespace evaluate_expression_l42_42013

theorem evaluate_expression (a b c : ℝ) (h1 : a = 4) (h2 : b = -4) (h3 : c = 3) : (3 / (a + b + c) = 1) :=
by
  sorry

end evaluate_expression_l42_42013


namespace find_cos_7theta_l42_42038

theorem find_cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = 1105 / 16384 :=
by
  sorry

end find_cos_7theta_l42_42038


namespace max_marks_paper_one_l42_42123

theorem max_marks_paper_one (M : ℝ) : 
  (0.42 * M = 64) → (M = 152) :=
by
  sorry

end max_marks_paper_one_l42_42123


namespace area_of_shaded_region_l42_42223

theorem area_of_shaded_region :
  let inner_square_side_length := 3
  let triangle_base := 2
  let triangle_height := 1
  let number_of_triangles := 8
  let area_inner_square := inner_square_side_length * inner_square_side_length
  let area_one_triangle := (1/2) * triangle_base * triangle_height
  let total_area_triangles := number_of_triangles * area_one_triangle
  let total_area_shaded := area_inner_square + total_area_triangles
  total_area_shaded = 17 :=
sorry

end area_of_shaded_region_l42_42223


namespace remaining_paint_fraction_l42_42295

theorem remaining_paint_fraction (x : ℝ) (h : 1.2 * x = 1 / 2) : (1 / 2) - x = 1 / 12 :=
by 
  sorry

end remaining_paint_fraction_l42_42295


namespace part1_part2_l42_42118

-- Part (1) statement
theorem part1 {x : ℝ} : (|x - 1| + |x + 2| >= 5) ↔ (x <= -3 ∨ x >= 2) := 
sorry

-- Part (2) statement
theorem part2 (a : ℝ) : (∀ x : ℝ, (|a * x - 2| < 3 ↔ -5/3 < x ∧ x < 1/3)) → a = -3 :=
sorry

end part1_part2_l42_42118


namespace lower_bound_of_expression_l42_42871

theorem lower_bound_of_expression :
  ∃ L : ℤ, (∀ n : ℤ, ((-1 ≤ n ∧ n ≤ 8) → (L < 4 * n + 7 ∧ 4 * n + 7 < 40))) ∧ L = 1 :=
by {
  sorry
}

end lower_bound_of_expression_l42_42871


namespace dice_probability_l42_42543

theorem dice_probability :
  let outcomes : List ℕ := [2, 3, 4, 5]
  let total_possible_outcomes := 6 * 6 * 6
  let successful_outcomes := 4 * 4 * 4
  (successful_outcomes / total_possible_outcomes : ℚ) = 8 / 27 :=
by
  sorry

end dice_probability_l42_42543


namespace unused_types_l42_42311

theorem unused_types (total_resources : ℕ) (used_types : ℕ) (valid_types : ℕ) :
  total_resources = 6 → used_types = 23 → valid_types = 2^total_resources - 1 - used_types → valid_types = 40 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end unused_types_l42_42311


namespace two_rooks_non_attacking_two_kings_non_attacking_two_bishops_non_attacking_two_knights_non_attacking_two_queens_non_attacking_l42_42907

noncomputable def rooks_non_attacking : Nat :=
  8 * 8 * 7 * 7 / 2

theorem two_rooks_non_attacking : rooks_non_attacking = 1568 := by
  sorry

noncomputable def kings_non_attacking : Nat :=
  (4 * 60 + 24 * 58 + 36 * 55 + 24 * 55 + 4 * 50) / 2

theorem two_kings_non_attacking : kings_non_attacking = 1806 := by
  sorry

noncomputable def bishops_non_attacking : Nat :=
  (28 * 25 + 20 * 54 + 12 * 52 + 4 * 50) / 2

theorem two_bishops_non_attacking : bishops_non_attacking = 1736 := by
  sorry

noncomputable def knights_non_attacking : Nat :=
  (4 * 61 + 8 * 60 + 20 * 59 + 16 * 57 + 15 * 55) / 2

theorem two_knights_non_attacking : knights_non_attacking = 1848 := by
  sorry

noncomputable def queens_non_attacking : Nat :=
  (28 * 42 + 20 * 40 + 12 * 38 + 4 * 36) / 2

theorem two_queens_non_attacking : queens_non_attacking = 1288 := by
  sorry

end two_rooks_non_attacking_two_kings_non_attacking_two_bishops_non_attacking_two_knights_non_attacking_two_queens_non_attacking_l42_42907


namespace area_fraction_of_rhombus_in_square_l42_42076

theorem area_fraction_of_rhombus_in_square :
  let n := 7                 -- grid size
  let side_length := n - 1   -- side length of the square
  let square_area := side_length^2 -- area of the square
  let rhombus_side := Real.sqrt 2 -- side length of the rhombus
  let rhombus_area := 2      -- area of the rhombus
  (rhombus_area / square_area) = 1 / 18 := sorry

end area_fraction_of_rhombus_in_square_l42_42076


namespace system1_solution_system2_solution_l42_42810

-- Problem 1
theorem system1_solution (x z : ℤ) (h1 : 3 * x - 5 * z = 6) (h2 : x + 4 * z = -15) : x = -3 ∧ z = -3 :=
by
  sorry

-- Problem 2
theorem system2_solution (x y : ℚ) 
 (h1 : ((2 * x - 1) / 5) + ((3 * y - 2) / 4) = 2) 
 (h2 : ((3 * x + 1) / 5) - ((3 * y + 2) / 4) = 0) : x = 3 ∧ y = 2 :=
by
  sorry

end system1_solution_system2_solution_l42_42810


namespace find_b_l42_42204

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x b : ℝ) : ℝ := Real.log x + b

theorem find_b (b : ℝ) :
  (∃ (x1 x2 : ℝ), (f x1 = x2⁻¹) ∧ (1 - x1 = 0) ∧ (Real.log x2 + b - 1 = 0)) →
  b = 2 :=
by
  sorry

end find_b_l42_42204


namespace calculate_group5_students_l42_42505

variable (total_students : ℕ) (freq_group1 : ℕ) (sum_freq_group2_3 : ℝ) (freq_group4 : ℝ)

theorem calculate_group5_students
  (h1 : total_students = 50)
  (h2 : freq_group1 = 7)
  (h3 : sum_freq_group2_3 = 0.46)
  (h4 : freq_group4 = 0.2) :
  (total_students * (1 - (freq_group1 / total_students + sum_freq_group2_3 + freq_group4)) = 10) :=
by
  sorry

end calculate_group5_students_l42_42505


namespace part1_part2_l42_42370

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l42_42370


namespace tournament_total_games_l42_42805

def total_number_of_games (num_teams : ℕ) (group_size : ℕ) (num_groups : ℕ) (teams_for_knockout : ℕ) : ℕ :=
  let games_per_group := (group_size * (group_size - 1)) / 2
  let group_stage_games := num_groups * games_per_group
  let knockout_teams := num_groups * teams_for_knockout
  let knockout_games := knockout_teams - 1
  group_stage_games + knockout_games

theorem tournament_total_games : total_number_of_games 32 4 8 2 = 63 := by
  sorry

end tournament_total_games_l42_42805


namespace correct_multiplication_factor_l42_42838

theorem correct_multiplication_factor (x : ℕ) : ((139 * x) - 1251 = 139 * 34) → x = 43 := by
  sorry

end correct_multiplication_factor_l42_42838


namespace first_term_arithmetic_sequence_l42_42918

theorem first_term_arithmetic_sequence (S : ℕ → ℤ) (a : ℤ) (h1 : ∀ n, S n = (n * (2 * a + (n - 1) * 5)) / 2)
    (h2 : ∀ n m, (S (3 * n)) / (S m) = (S (3 * m)) / (S n)) : a = 5 / 2 := 
sorry

end first_term_arithmetic_sequence_l42_42918


namespace ian_money_left_l42_42893

theorem ian_money_left
  (hours_worked : ℕ)
  (hourly_rate : ℕ)
  (spending_percentage : ℚ)
  (total_earnings : ℕ)
  (amount_spent : ℕ)
  (amount_left : ℕ)
  (h_worked : hours_worked = 8)
  (h_rate : hourly_rate = 18)
  (h_spending : spending_percentage = 0.5)
  (h_earnings : total_earnings = hours_worked * hourly_rate)
  (h_spent : amount_spent = total_earnings * spending_percentage)
  (h_left : amount_left = total_earnings - amount_spent) :
  amount_left = 72 := 
  sorry

end ian_money_left_l42_42893


namespace length_of_goods_train_l42_42127

theorem length_of_goods_train 
  (speed_kmph : ℝ) (platform_length : ℝ) (time_sec : ℝ) (train_length : ℝ) 
  (h1 : speed_kmph = 72)
  (h2 : platform_length = 270) 
  (h3 : time_sec = 26) 
  (h4 : train_length = (speed_kmph * 1000 / 3600 * time_sec) - platform_length)
  : train_length = 250 := 
  by
    sorry

end length_of_goods_train_l42_42127


namespace xy_sum_l42_42615

-- Define the problem conditions
variable (x y : ℚ)
variable (h1 : 1 / x + 1 / y = 4)
variable (h2 : 1 / x - 1 / y = -8)

-- Define the theorem to prove
theorem xy_sum : x + y = -1 / 3 := by
  sorry

end xy_sum_l42_42615


namespace total_chairs_l42_42086

/-- Susan loves chairs. In her house, there are red chairs, yellow chairs, blue chairs, and green chairs.
    There are 5 red chairs. There are 4 times as many yellow chairs as red chairs.
    There are 2 fewer blue chairs than yellow chairs. The number of green chairs is half the sum of the number of red chairs and blue chairs (rounded down).
    We want to determine the total number of chairs in Susan's house. -/
theorem total_chairs (r y b g : ℕ) 
  (hr : r = 5)
  (hy : y = 4 * r) 
  (hb : b = y - 2) 
  (hg : g = (r + b) / 2) :
  r + y + b + g = 54 := 
sorry

end total_chairs_l42_42086


namespace part1_part2_l42_42369

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l42_42369


namespace cost_of_10_apples_l42_42227

-- Define the price for 10 apples as a variable
noncomputable def price_10_apples (P : ℝ) : ℝ := P

-- Theorem stating that the cost for 10 apples is the provided price
theorem cost_of_10_apples (P : ℝ) : price_10_apples P = P :=
  by
    sorry

end cost_of_10_apples_l42_42227


namespace conjecture_l42_42591

noncomputable def f (x : ℝ) : ℝ :=
  1 / (3^x + Real.sqrt 3)

theorem conjecture (x : ℝ) : f (-x) + f (1 + x) = Real.sqrt 3 / 3 := sorry

end conjecture_l42_42591


namespace ashton_remaining_items_l42_42140

variables (pencil_boxes : ℕ) (pens_boxes : ℕ) (pencils_per_box : ℕ) (pens_per_box : ℕ)
          (given_pencils_brother : ℕ) (distributed_pencils_friends : ℕ)
          (distributed_pens_friends : ℕ)

def total_initial_pencils := 3 * 14
def total_initial_pens := 2 * 10

def remaining_pencils := total_initial_pencils - 6 - 12
def remaining_pens := total_initial_pens - 8
def remaining_items := remaining_pencils + remaining_pens

theorem ashton_remaining_items : remaining_items = 36 :=
sorry

end ashton_remaining_items_l42_42140


namespace necessary_but_not_sufficient_l42_42841

noncomputable def necessary_but_not_sufficient_condition (x : ℝ) : Prop :=
  -2 < x ∧ x < 3 → x^2 - 2 * x - 3 < 0

theorem necessary_but_not_sufficient (x : ℝ) : 
-2 < x ∧ x < 3 → x^2 - 2 * x - 3 < 0 := 
by
  sorry

end necessary_but_not_sufficient_l42_42841


namespace ellipse_eq_and_line_eq_l42_42164

theorem ellipse_eq_and_line_eq
  (e : ℝ) (a b c xC yC: ℝ)
  (h_e : e = (Real.sqrt 3 / 2))
  (h_a : a = 2)
  (h_c : c = Real.sqrt 3)
  (h_b : b = Real.sqrt (a^2 - c^2))
  (h_ellipse : ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 = 1))
  (h_C_on_G : xC^2 / 4 + yC^2 = 1)
  (h_diameter_condition : ∀ (B : ℝ × ℝ), B = (0, 1) →
    ((2 * xC - yC + 1 = 0) →
    (xC = 0 ∧ yC = 1) ∨ (xC = -16 / 17 ∧ yC = -15 / 17)))
  : (∀ x y, (y = 2*x + 1) ↔ (x + 2*y - 2 = 0 ∨ 3*x - 10*y - 6 = 0)) :=
by
  sorry

end ellipse_eq_and_line_eq_l42_42164


namespace problem_statement_l42_42593

variable {f : ℝ → ℝ}

-- Condition 1: f(x) has domain ℝ (implicitly given by the type signature ωf)
-- Condition 2: f is decreasing on the interval (6, +∞)
def is_decreasing_on_6_infty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 6 < x → x < y → f x > f y

-- Condition 3: y = f(x + 6) is an even function
def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) = f (-x - 6)

-- The statement to prove
theorem problem_statement (h_decrease : is_decreasing_on_6_infty f) (h_even_shift : is_even_shifted f) : f 5 > f 8 :=
sorry

end problem_statement_l42_42593


namespace bounce_height_less_than_two_l42_42554

theorem bounce_height_less_than_two (k : ℕ) (h₀ : ℝ) (r : ℝ) (ε : ℝ) 
    (h₀_pos : h₀ = 20) (r_pos : r = 1/2) (ε_pos : ε = 2): 
  (h₀ * (r ^ k) < ε) ↔ k >= 4 := by
  sorry

end bounce_height_less_than_two_l42_42554


namespace total_cars_l42_42475

-- Definitions of the conditions
def cathy_cars : Nat := 5

def carol_cars : Nat := 2 * cathy_cars

def susan_cars : Nat := carol_cars - 2

def lindsey_cars : Nat := cathy_cars + 4

-- The theorem statement (problem)
theorem total_cars : cathy_cars + carol_cars + susan_cars + lindsey_cars = 32 :=
by
  -- sorry is added to skip the proof
  sorry

end total_cars_l42_42475


namespace problem_1_l42_42575

theorem problem_1 (x : ℝ) (h : x^2 = 2) : (3 * x)^2 - 4 * (x^3)^2 = -14 :=
by {
  sorry
}

end problem_1_l42_42575


namespace project_completion_in_16_days_l42_42550

noncomputable def a_work_rate : ℚ := 1 / 20
noncomputable def b_work_rate : ℚ := 1 / 30
noncomputable def c_work_rate : ℚ := 1 / 40
noncomputable def days_a_works (X: ℚ) : ℚ := X - 10
noncomputable def days_b_works (X: ℚ) : ℚ := X - 5
noncomputable def days_c_works (X: ℚ) : ℚ := X

noncomputable def total_work (X: ℚ) : ℚ :=
  (a_work_rate * days_a_works X) + (b_work_rate * days_b_works X) + (c_work_rate * days_c_works X)

theorem project_completion_in_16_days : total_work 16 = 1 := by
  sorry

end project_completion_in_16_days_l42_42550


namespace sum_local_values_l42_42698

theorem sum_local_values :
  let local_value_2 := 2000
  let local_value_3 := 300
  let local_value_4 := 40
  let local_value_5 := 5
  local_value_2 + local_value_3 + local_value_4 + local_value_5 = 2345 :=
by
  sorry

end sum_local_values_l42_42698


namespace coprime_integer_pairs_sum_285_l42_42207

theorem coprime_integer_pairs_sum_285 : 
  (∃ s : Finset (ℕ × ℕ), 
    ∀ p ∈ s, p.1 + p.2 = 285 ∧ Nat.gcd p.1 p.2 = 1 ∧ s.card = 72) := sorry

end coprime_integer_pairs_sum_285_l42_42207


namespace sum_square_divisors_positive_l42_42949

theorem sum_square_divisors_positive (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : 
  (a^2 + b^2) / c + (b^2 + c^2) / a + (c^2 + a^2) / b > 0 := 
by 
  sorry

end sum_square_divisors_positive_l42_42949


namespace bob_wins_l42_42776

-- Define the notion of nim-sum used in nim-games
def nim_sum (a b : ℕ) : ℕ := Nat.xor a b

-- Define nim-values for given walls based on size
def nim_value : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| 3 => 3
| 4 => 1
| 5 => 4
| 6 => 3
| 7 => 2
| _ => 0

-- Calculate the nim-value of a given configuration
def nim_config (c : List ℕ) : ℕ :=
c.foldl (λ acc n => nim_sum acc (nim_value n)) 0

-- Prove that the configuration (7, 3, 1) gives a nim-value of 0
theorem bob_wins : nim_config [7, 3, 1] = 0 := by
  sorry

end bob_wins_l42_42776


namespace avg_weight_of_class_l42_42270

def A_students : Nat := 36
def B_students : Nat := 44
def C_students : Nat := 50
def D_students : Nat := 30

def A_avg_weight : ℝ := 40
def B_avg_weight : ℝ := 35
def C_avg_weight : ℝ := 42
def D_avg_weight : ℝ := 38

def A_additional_students : Nat := 5
def A_additional_weight : ℝ := 10

def B_reduced_students : Nat := 7
def B_reduced_weight : ℝ := 8

noncomputable def total_weight_class : ℝ :=
  (A_students * A_avg_weight + A_additional_students * A_additional_weight) +
  (B_students * B_avg_weight - B_reduced_students * B_reduced_weight) +
  (C_students * C_avg_weight) +
  (D_students * D_avg_weight)

noncomputable def total_students_class : Nat :=
  A_students + B_students + C_students + D_students

noncomputable def avg_weight_class : ℝ :=
  total_weight_class / total_students_class

theorem avg_weight_of_class :
  avg_weight_class = 38.84 := by
    sorry

end avg_weight_of_class_l42_42270


namespace neither_drinkers_eq_nine_l42_42989

-- Define the number of businessmen at the conference
def total_businessmen : Nat := 30

-- Define the number of businessmen who drank coffee
def coffee_drinkers : Nat := 15

-- Define the number of businessmen who drank tea
def tea_drinkers : Nat := 13

-- Define the number of businessmen who drank both coffee and tea
def both_drinkers : Nat := 7

-- Prove the number of businessmen who drank neither coffee nor tea
theorem neither_drinkers_eq_nine : 
  total_businessmen - ((coffee_drinkers + tea_drinkers) - both_drinkers) = 9 := 
by
  sorry

end neither_drinkers_eq_nine_l42_42989


namespace negate_original_is_correct_l42_42943

-- Define the original proposition
def original_proposition (a b : ℕ) : Prop := (a * b = 0) → (a = 0 ∨ b = 0)

-- Define the negated proposition
def negated_proposition (a b : ℕ) : Prop := (a * b ≠ 0) → (a ≠ 0 ∧ b ≠ 0)

-- The theorem stating that the negation of the original proposition is the given negated proposition
theorem negate_original_is_correct (a b : ℕ) : ¬ original_proposition a b ↔ negated_proposition a b := by
  sorry

end negate_original_is_correct_l42_42943


namespace find_b_c_l42_42817

variable {b c : ℝ}

theorem find_b_c
  (h_bc_pos : (0 < b) ∧ (0 < c))
  (h_prod : ∀ (x1 x2 x3 x4 : ℝ), 
    (x1 * x2 * x3 * x4 = 1) ∧
    (x1 + x2 = -2 * b) ∧ (x1 * x2 = c) ∧
    (x3 + x4 = -2 * c) ∧ (x3 * x4 = b)) :
  b = 1 ∧ c = 1 := 
by
  sorry

end find_b_c_l42_42817


namespace find_value_of_s_l42_42580

theorem find_value_of_s
  (a b c w s p : ℕ)
  (h₁ : a + b = w)
  (h₂ : w + c = s)
  (h₃ : s + a = p)
  (h₄ : b + c + p = 16) :
  s = 8 :=
sorry

end find_value_of_s_l42_42580


namespace minute_hand_rotation_l42_42857

theorem minute_hand_rotation (h : ℕ) (radians_per_rotation : ℝ) : h = 5 → radians_per_rotation = 2 * Real.pi → - (h * radians_per_rotation) = -10 * Real.pi :=
by
  intros h_eq rp_eq
  rw [h_eq, rp_eq]
  sorry

end minute_hand_rotation_l42_42857


namespace abs_diff_of_numbers_l42_42667

theorem abs_diff_of_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) : |x - y| = 4 :=
by
  sorry

end abs_diff_of_numbers_l42_42667


namespace max_x_minus_y_l42_42173

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : (x - y) ≤ 1 + 3 * (Real.sqrt 2) :=
sorry

end max_x_minus_y_l42_42173


namespace percentage_of_x_l42_42967

theorem percentage_of_x (x : ℝ) (h : x > 0) : ((x / 5 + x / 25) / x) * 100 = 24 := 
by 
  sorry

end percentage_of_x_l42_42967


namespace not_neighboring_root_equation_x2_x_2_neighboring_root_equation_k_values_l42_42997

def is_neighboring_root_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁ * x₁ + b * x₁ + c = 0 ∧ a * x₂ * x₂ + b * x₂ + c = 0 
  ∧ (x₁ - x₂ = 1 ∨ x₂ - x₁ = 1)

theorem not_neighboring_root_equation_x2_x_2 : 
  ¬ is_neighboring_root_equation 1 1 (-2) :=
sorry

theorem neighboring_root_equation_k_values (k : ℝ) : 
  is_neighboring_root_equation 1 (-(k-3)) (-3*k) ↔ k = -2 ∨ k = -4 :=
sorry

end not_neighboring_root_equation_x2_x_2_neighboring_root_equation_k_values_l42_42997


namespace magic_square_y_l42_42455

theorem magic_square_y (a b c d e y : ℚ) (h1 : y - 61 = a) (h2 : 2 * y - 125 = b) 
    (h3 : y + 25 + 64 = 3 + (y - 61) + (2 * y - 125)) : y = 272 / 3 :=
by
  sorry

end magic_square_y_l42_42455


namespace election_valid_vote_counts_l42_42458

noncomputable def totalVotes : ℕ := 900000
noncomputable def invalidPercentage : ℝ := 0.25
noncomputable def validVotes : ℝ := totalVotes * (1.0 - invalidPercentage)
noncomputable def fractionA : ℝ := 7 / 15
noncomputable def fractionB : ℝ := 5 / 15
noncomputable def fractionC : ℝ := 3 / 15
noncomputable def validVotesA : ℝ := fractionA * validVotes
noncomputable def validVotesB : ℝ := fractionB * validVotes
noncomputable def validVotesC : ℝ := fractionC * validVotes

theorem election_valid_vote_counts :
  validVotesA = 315000 ∧ validVotesB = 225000 ∧ validVotesC = 135000 := by
  sorry

end election_valid_vote_counts_l42_42458


namespace max_x_minus_y_l42_42189

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l42_42189


namespace points_in_quadrants_l42_42266

theorem points_in_quadrants :
  ∀ (x y : ℝ), (y > 3 * x) → (y > 5 - 2 * x) → ((0 ≤ x ∧ 0 ≤ y) ∨ (x ≤ 0 ∧ 0 ≤ y)) :=
by
  intros x y h1 h2
  sorry

end points_in_quadrants_l42_42266


namespace part1_solution_part2_solution_l42_42402

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l42_42402


namespace probability_diamond_first_and_ace_or_king_second_l42_42277

-- Define the condition of the combined deck consisting of two standard decks (104 cards total)
def two_standard_decks := 104

-- Define the number of diamonds, aces, and kings in the combined deck
def number_of_diamonds := 26
def number_of_aces := 8
def number_of_kings := 8

-- Define the events for drawing cards
def first_card_is_diamond := (number_of_diamonds : ℕ) / (two_standard_decks : ℕ)
def second_card_is_ace_or_king_if_first_is_not_ace_or_king :=
  (16 / 103 : ℚ) -- 16 = 8 (aces) + 8 (kings)
def second_card_is_ace_or_king_if_first_is_ace_or_king :=
  (15 / 103 : ℚ) -- 15 = 7 (remaining aces) + 7 (remaining kings) + 1 (remaining ace or king of the same suit)

-- Define the probabilities of the combined event
def probability_first_is_non_ace_king_diamond_and_second_is_ace_or_king :=
  (22 / 104) * (16 / 103)
def probability_first_is_ace_or_king_diamond_and_second_is_ace_or_king :=
  (4 / 104) * (15 / 103)

-- Define the total probability combining both events
noncomputable def total_probability :=
  probability_first_is_non_ace_king_diamond_and_second_is_ace_or_king +
  probability_first_is_ace_or_king_diamond_and_second_is_ace_or_king

-- Theorem stating the desired probability result
theorem probability_diamond_first_and_ace_or_king_second :
  total_probability = (103 / 2678 : ℚ) :=
sorry

end probability_diamond_first_and_ace_or_king_second_l42_42277


namespace fifteenth_odd_multiple_of_5_is_145_l42_42690

def sequence_term (n : ℕ) : ℤ :=
  10 * n - 5

theorem fifteenth_odd_multiple_of_5_is_145 : sequence_term 15 = 145 :=
by
  sorry

end fifteenth_odd_multiple_of_5_is_145_l42_42690


namespace determine_uv_l42_42324

theorem determine_uv :
  ∃ u v : ℝ, (u = 5 / 17) ∧ (v = -31 / 17) ∧
    ((⟨3, -2⟩ : ℝ × ℝ) + u • ⟨5, 8⟩ = (⟨-1, 4⟩ : ℝ × ℝ) + v • ⟨-3, 2⟩) :=
by
  sorry

end determine_uv_l42_42324


namespace solution_of_system_l42_42739

theorem solution_of_system 
  (k : ℝ) (x y : ℝ)
  (h1 : (1 : ℝ) = 2 * 1 - 1)
  (h2 : (1 : ℝ) = k * 1)
  (h3 : k ≠ 0)
  (h4 : 2 * x - y = 1)
  (h5 : k * x - y = 0) : 
  x = 1 ∧ y = 1 :=
by
  sorry

end solution_of_system_l42_42739


namespace minimal_fraction_difference_l42_42471

theorem minimal_fraction_difference (p q : ℕ) (hp : 0 < p) (hq : 0 < q) 
  (h1 : 3 / 5 < p / q) (h2 : p / q < 2 / 3) (hmin: ∀ r s : ℕ, (3 / 5 < r / s ∧ r / s < 2 / 3 ∧ s < q) → false) :
  q - p = 11 := 
sorry

end minimal_fraction_difference_l42_42471


namespace find_N_l42_42093

theorem find_N (a b c N : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : N = a * b * c) (h2 : N = 8 * (a + b + c)) (h3 : c = a + b) : N = 272 :=
sorry

end find_N_l42_42093


namespace original_rent_eq_l42_42652

theorem original_rent_eq (R : ℝ)
  (h1 : 4 * 800 = 3200)
  (h2 : 4 * 850 = 3400)
  (h3 : 3400 - 3200 = 200)
  (h4 : 200 = 0.25 * R) : R = 800 := by
  sorry

end original_rent_eq_l42_42652


namespace part1_part2_l42_42374

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l42_42374


namespace find_a_l42_42502

noncomputable def f (x : ℝ) := x^2

theorem find_a (a : ℝ) (h : (1/2) * a^2 * (a/2) = 2) :
  a = 2 :=
sorry

end find_a_l42_42502


namespace fourth_root_squared_cubed_l42_42827

theorem fourth_root_squared_cubed (x : ℝ) (h : (x^(1/4))^2^3 = 1296) : x = 256 :=
sorry

end fourth_root_squared_cubed_l42_42827


namespace max_min_product_of_three_l42_42872

open List

theorem max_min_product_of_three (s : List Int) (h : s = [-1, -2, 3, 4]) : 
  ∃ (max min : Int), 
    max = 8 ∧ min = -24 ∧ 
    (∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c → a * b * c ≤ max) ∧
    (∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c → a * b * c ≥ min) := 
by
  sorry

end max_min_product_of_three_l42_42872


namespace part1_solution_part2_solution_l42_42400

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l42_42400


namespace max_value_of_e_l42_42923

theorem max_value_of_e (a b c d e : ℝ) 
  (h₁ : a + b + c + d + e = 8) 
  (h₂ : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  e ≤ 16 / 5 :=
sorry

end max_value_of_e_l42_42923


namespace shoe_length_increase_l42_42839

noncomputable def shoeSizeLength (l : ℕ → ℝ) (size : ℕ) : ℝ :=
  if size = 15 then 9.25
  else if size = 17 then 1.3 * l 8
  else l size

theorem shoe_length_increase :
  (forall l : ℕ → ℝ,
    (shoeSizeLength l 15 = 9.25) ∧
    (shoeSizeLength l 17 = 1.3 * (shoeSizeLength l 8)) ∧
    (forall n, shoeSizeLength l (n + 1) = shoeSizeLength l n + 0.25)
  ) :=
  sorry

end shoe_length_increase_l42_42839


namespace max_x_minus_y_l42_42188

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  (∃ z : ℝ, z = x - y ∧ (∀ w : ℝ, w = x - y → w ≤ 1 + 3 * real.sqrt 2) ∧ (∃ u : ℝ, u = x - y ∧ u = 1 + 3 * real.sqrt 2)) :=
sorry

end max_x_minus_y_l42_42188


namespace part1_solution_set_part2_range_l42_42355

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l42_42355


namespace solution_set_inequality_l42_42818

theorem solution_set_inequality (x : ℝ) : 3 * x - 2 > x → x > 1 := by
  sorry

end solution_set_inequality_l42_42818


namespace part1_part2_l42_42366

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l42_42366


namespace lower_limit_of_a_l42_42450

theorem lower_limit_of_a (a b : ℤ) (h_a : a < 26) (h_b1 : b > 14) (h_b2 : b < 31) (h_ineq : (4 : ℚ) / 3 ≤ a / b) : 
  20 ≤ a :=
by
  sorry

end lower_limit_of_a_l42_42450


namespace algebra_expression_value_l42_42771

theorem algebra_expression_value
  (x y : ℝ)
  (h : x - 2 * y + 2 = 5) : 4 * y - 2 * x + 1 = -5 :=
by sorry

end algebra_expression_value_l42_42771


namespace other_root_of_equation_l42_42492

theorem other_root_of_equation (m : ℝ) :
  (∃ (x : ℝ), 3 * x^2 + m * x = -2 ∧ x = -1) →
  (∃ (y : ℝ), 3 * y^2 + m * y + 2 = 0 ∧ y = -(-2 / 3)) :=
by
  sorry

end other_root_of_equation_l42_42492


namespace long_letter_time_ratio_l42_42501

-- Definitions based on conditions
def letters_per_month := (30 / 3 : Nat)
def regular_letter_pages := (20 / 10 : Nat)
def total_regular_pages := letters_per_month * regular_letter_pages
def long_letter_pages := 24 - total_regular_pages

-- Define the times and calculate the ratios
def time_spent_per_page_regular := (20 / regular_letter_pages : Nat)
def time_spent_per_page_long := (80 / long_letter_pages : Nat)
def time_ratio := time_spent_per_page_long / time_spent_per_page_regular

-- Theorem to prove the ratio
theorem long_letter_time_ratio : time_ratio = 2 := by
  sorry

end long_letter_time_ratio_l42_42501


namespace san_antonio_to_austin_buses_passed_l42_42721

def departure_schedule (departure_time_A_to_S departure_time_S_to_A travel_time : ℕ) : Prop :=
  ∀ t, (t < travel_time) →
       (∃ n, t = (departure_time_A_to_S + n * 60)) ∨
       (∃ m, t = (departure_time_S_to_A + m * 60)) →
       t < travel_time

theorem san_antonio_to_austin_buses_passed :
  let departure_time_A_to_S := 30  -- Austin to San Antonio buses leave every hour on the half-hour (e.g., 00:30, 1:30, ...)
  let departure_time_S_to_A := 0   -- San Antonio to Austin buses leave every hour on the hour (e.g., 00:00, 1:00, ...)
  let travel_time := 6 * 60        -- The trip takes 6 hours, or 360 minutes
  departure_schedule departure_time_A_to_S departure_time_S_to_A travel_time →
  ∃ count, count = 12 := 
by
  sorry

end san_antonio_to_austin_buses_passed_l42_42721


namespace x_quad_greater_l42_42247

theorem x_quad_greater (x : ℝ) : x^4 > x - 1/2 :=
sorry

end x_quad_greater_l42_42247


namespace fraction_addition_simplest_form_l42_42318

theorem fraction_addition_simplest_form :
  (7 / 8) + (3 / 5) = 59 / 40 :=
by sorry

end fraction_addition_simplest_form_l42_42318


namespace part1_l42_42431

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

theorem part1 {x : ℝ} : f x ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) := by
  sorry

end part1_l42_42431


namespace part1_part2_l42_42368

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l42_42368


namespace second_number_is_34_l42_42820

theorem second_number_is_34 (x y z : ℝ) (h1 : x + y + z = 120) 
  (h2 : x / y = 3 / 4) (h3 : y / z = 4 / 7) : y = 34 :=
by 
  sorry

end second_number_is_34_l42_42820


namespace evaluate_expression_l42_42012

theorem evaluate_expression : 3^5 * 6^5 * 3^6 * 6^6 = 18^11 :=
by sorry

end evaluate_expression_l42_42012


namespace rhombus_new_perimeter_l42_42262

theorem rhombus_new_perimeter (d1 d2 : ℝ) (scale : ℝ) (h_d1 : d1 = 10) (h_d2 : d2 = 24) (h_scale : scale = 0.5) : 
  4 * (scale * (Real.sqrt ((d1/2)^2 + (d2/2)^2))) = 26 := 
by
  sorry

end rhombus_new_perimeter_l42_42262


namespace sum_of_possible_values_of_x_l42_42755

theorem sum_of_possible_values_of_x (x : ℝ) (h : (x + 3) * (x - 4) = 22) : ∃ (x1 x2 : ℝ), x^2 - x - 34 = 0 ∧ x1 + x2 = 1 :=
by
  sorry

end sum_of_possible_values_of_x_l42_42755


namespace halfway_fraction_l42_42677

theorem halfway_fraction (a b : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 7) :
  ((a + b) / 2) = 41 / 56 :=
by
  rw [h_a, h_b]
  sorry

end halfway_fraction_l42_42677


namespace red_paint_four_times_blue_paint_total_painted_faces_is_1625_l42_42525

/-- Given a structure of twenty-five layers of cubes -/
def structure_layers := 25

/-- The number of painted faces from each vertical view -/
def vertical_faces_per_view : ℕ :=
  (structure_layers * (structure_layers + 1)) / 2

/-- The total number of red-painted faces (4 vertical views) -/
def total_red_faces : ℕ :=
  4 * vertical_faces_per_view

/-- The total number of blue-painted faces (1 top view) -/
def total_blue_faces : ℕ :=
  vertical_faces_per_view

theorem red_paint_four_times_blue_paint :
  total_red_faces = 4 * total_blue_faces :=
by sorry

theorem total_painted_faces_is_1625 :
  (4 * vertical_faces_per_view + vertical_faces_per_view) = 1625 :=
by sorry

end red_paint_four_times_blue_paint_total_painted_faces_is_1625_l42_42525


namespace equality_of_fractions_l42_42493

theorem equality_of_fractions
  (a b c x y z : ℝ)
  (h1 : a = b * z + c * y)
  (h2 : b = c * x + a * z)
  (h3 : c = a * y + b * x)
  (hx : x ≠ 1 ∧ x ≠ -1)
  (hy : y ≠ 1 ∧ y ≠ -1)
  (hz : z ≠ 1 ∧ z ≠ -1) :
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
  (a^2) / (1 - x^2) = (b^2) / (1 - y^2) ∧ (b^2) / (1 - y^2) = (c^2) / (1 - z^2) :=
by
  sorry

end equality_of_fractions_l42_42493


namespace sum_of_roots_l42_42761

-- Define the main condition
def equation (x : ℝ) : Prop :=
  (x + 3) * (x - 4) = 22

-- Define the statement we want to prove
theorem sum_of_roots : ∀ x1 x2 : ℝ, (equation x1 ∧ equation x2) → x1 + x2 = 1 := by
  intros x1 x2 h
  sorry

end sum_of_roots_l42_42761


namespace stickers_total_proof_l42_42952

def stickers_per_page : ℕ := 10
def number_of_pages : ℕ := 22
def total_stickers : ℕ := stickers_per_page * number_of_pages

theorem stickers_total_proof : total_stickers = 220 := by
  sorry

end stickers_total_proof_l42_42952


namespace max_x_minus_y_l42_42182

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  ∃ k ∈ ℝ, x - y ≤ k ∧ k = 1 + 3 * real.sqrt 2 :=
sorry

end max_x_minus_y_l42_42182


namespace max_true_statements_l42_42472

theorem max_true_statements {p q : ℝ} (hp : p > 0) (hq : q < 0) :
  ∀ (s1 s2 s3 s4 s5 : Prop), 
  s1 = (1 / p > 1 / q) →
  s2 = (p^3 > q^3) →
  s3 = (p^2 < q^2) →
  s4 = (p > 0) →
  s5 = (q < 0) →
  s1 ∧ s2 ∧ s4 ∧ s5 ∧ ¬s3 → 
  ∃ m : ℕ, m = 4 := 
by {
  sorry
}

end max_true_statements_l42_42472


namespace neg_sqrt_comparison_l42_42852

theorem neg_sqrt_comparison : -Real.sqrt 7 > -Real.sqrt 11 := by
  sorry

end neg_sqrt_comparison_l42_42852


namespace rowing_students_l42_42300

theorem rowing_students (X Y : ℕ) (N : ℕ) :
  (17 * X + 6 = N) →
  (10 * Y + 2 = N) →
  100 < N →
  N < 200 →
  5 ≤ X ∧ X ≤ 11 →
  10 ≤ Y ∧ Y ≤ 19 →
  N = 142 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end rowing_students_l42_42300


namespace part1_l42_42439

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

theorem part1 {x : ℝ} : f x ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) := by
  sorry

end part1_l42_42439


namespace karen_starts_late_by_4_minutes_l42_42913

-- Define conditions as Lean 4 variables/constants
noncomputable def karen_speed : ℝ := 60 -- in mph
noncomputable def tom_speed : ℝ := 45 -- in mph
noncomputable def tom_distance : ℝ := 24 -- in miles
noncomputable def karen_lead : ℝ := 4 -- in miles

-- Main theorem statement
theorem karen_starts_late_by_4_minutes : 
  ∃ (minutes_late : ℝ), minutes_late = 4 :=
by
  -- Calculations based on given conditions provided in the problem
  let t := tom_distance / tom_speed -- Time for Tom to drive 24 miles
  let tk := (tom_distance + karen_lead) / karen_speed -- Time for Karen to drive 28 miles
  let time_difference := t - tk -- Time difference between Tom and Karen
  let minutes_late := time_difference * 60 -- Convert time difference to minutes
  existsi minutes_late -- Existential quantifier to state the existence of such a time
  have h : minutes_late = 4 := sorry -- Placeholder for demonstrating equality
  exact h

end karen_starts_late_by_4_minutes_l42_42913


namespace mul_inv_800_mod_7801_l42_42944

theorem mul_inv_800_mod_7801 :
  ∃ x : ℕ, 0 ≤ x ∧ x < 7801 ∧ (800 * x) % 7801 = 1 := by
  use 3125
  dsimp
  norm_num1
  sorry

end mul_inv_800_mod_7801_l42_42944


namespace encoding_correctness_l42_42526

theorem encoding_correctness 
  (old_message : String)
  (new_encoding : Char → String)
  (decoded_message : String)
  (result : String) :
  old_message = "011011010011" →
  new_encoding 'A' = "21" →
  new_encoding 'B' = "122" →
  new_encoding 'C' = "1" →
  decoded_message = "ABCBA" →
  result = "211221121" →
  (encode (decode old_message) new_encoding) = result :=
by
  sorry

end encoding_correctness_l42_42526


namespace solution_set_of_inequality_l42_42598

variable {R : Type} [LinearOrderedField R] (f : R → R)

-- Conditions
def monotonically_increasing_on_nonnegatives := 
  ∀ x y : R, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

def odd_function_shifted_one := 
  ∀ x : R, f (-x) = 2 - f (x)

-- The problem
theorem solution_set_of_inequality
  (mono_inc : monotonically_increasing_on_nonnegatives f)
  (odd_shift : odd_function_shifted_one f) :
  {x : R | f (3 * x + 4) + f (1 - x) < 2} = {x : R | x < -5 / 2} :=
by
  sorry

end solution_set_of_inequality_l42_42598


namespace hex_product_l42_42906

def hex_to_dec (h : Char) : Nat :=
  match h with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | c   => c.toNat - '0'.toNat

noncomputable def dec_to_hex (n : Nat) : String :=
  let q := n / 16
  let r := n % 16
  let r_hex := if r < 10 then Char.ofNat (r + '0'.toNat) else Char.ofNat (r - 10 + 'A'.toNat)
  (if q > 0 then toString q else "") ++ Char.toString r_hex

theorem hex_product :
  dec_to_hex (hex_to_dec 'A' * hex_to_dec 'B') = "6E" :=
by
  sorry

end hex_product_l42_42906


namespace part1_part2_l42_42360

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l42_42360


namespace part1_part2_l42_42425

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l42_42425


namespace sum_of_roots_of_equation_l42_42763

theorem sum_of_roots_of_equation :
  (∀ x : ℝ, (x + 3) * (x - 4) = 22 → ∃ a b : ℝ, (x - a) * (x - b) = 0 ∧ a + b = 1) :=
by
  sorry

end sum_of_roots_of_equation_l42_42763


namespace find_k_value_l42_42451

theorem find_k_value (x y k : ℝ) 
  (h1 : x - 3 * y = k + 2) 
  (h2 : x - y = 4) 
  (h3 : 3 * x + y = -8) : 
  k = 12 := 
  by {
    sorry
  }

end find_k_value_l42_42451


namespace hat_cost_l42_42279

theorem hat_cost (total_hats blue_hat_cost green_hat_cost green_hats : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_hat_cost = 6)
  (h3 : green_hat_cost = 7)
  (h4 : green_hats = 20) :
  (total_hats - green_hats) * blue_hat_cost + green_hats * green_hat_cost = 530 := 
by sorry

end hat_cost_l42_42279


namespace jane_donuts_l42_42579

def croissant_cost := 60
def donut_cost := 90
def days := 6

theorem jane_donuts (c d k : ℤ) 
  (h1 : c + d = days)
  (h2 : donut_cost * d + croissant_cost * c = 100 * k + 50) :
  d = 3 :=
sorry

end jane_donuts_l42_42579


namespace find_constants_l42_42921

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x + 1

noncomputable def f_inv (x a b c : ℝ) : ℝ :=
  ( (x - a + Real.sqrt (x^2 - b * x + c)) / 2 )^(1/3) +
  ( (x - a - Real.sqrt (x^2 - b * x + c)) / 2 )^(1/3)

theorem find_constants (a b c : ℝ) (h1 : f_inv (1:ℝ) a b c = 0)
  (ha : a = 1) (hb : b = 2) (hc : c = 5) : a + 10 * b + 100 * c = 521 :=
by
  rw [ha, hb, hc]
  norm_num

end find_constants_l42_42921


namespace final_price_after_increase_and_decrease_l42_42969

variable (P : ℝ)

theorem final_price_after_increase_and_decrease (h : P > 0) : 
  let increased_price := P * 1.15
  let final_price := increased_price * 0.85
  final_price = P * 0.9775 :=
by
  sorry

end final_price_after_increase_and_decrease_l42_42969


namespace divisible_by_120_l42_42078

theorem divisible_by_120 (n : ℕ) : 120 ∣ n * (n^2 - 1) * (n^2 - 5 * n + 26) := sorry

end divisible_by_120_l42_42078


namespace sequence_eventually_periodic_l42_42547

open Nat

noncomputable def sum_prime_factors_plus_one (K : ℕ) : ℕ := 
  (K.factors.sum) + 1

theorem sequence_eventually_periodic (K : ℕ) (hK : K ≥ 9) :
  ∃ m n : ℕ, m ≠ n ∧ sum_prime_factors_plus_one^[m] K = sum_prime_factors_plus_one^[n] K := 
sorry

end sequence_eventually_periodic_l42_42547


namespace compute_fraction_power_l42_42994

theorem compute_fraction_power :
  8 * (1 / 4) ^ 4 = 1 / 32 := 
by
  sorry

end compute_fraction_power_l42_42994


namespace fifteenth_odd_multiple_of_five_l42_42687

theorem fifteenth_odd_multiple_of_five :
  ∃ a : ℕ, (∀ n : ℕ, a n = 5 + (n - 1) * 10) ∧ a 15 = 145 :=
by
  let a := λ n, 5 + (n - 1) * 10
  use a
  split
  { intros n,
    refl }
  { refl }
  sorry

end fifteenth_odd_multiple_of_five_l42_42687


namespace part1_l42_42437

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

theorem part1 {x : ℝ} : f x ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) := by
  sorry

end part1_l42_42437


namespace part1_part2_l42_42376

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l42_42376


namespace completing_square_to_simplify_eq_l42_42104

theorem completing_square_to_simplify_eq : 
  ∃ (c : ℝ), (∀ x : ℝ, x^2 - 6 * x + 4 = 0 ↔ (x - 3)^2 = c) :=
by
  use 5
  intro x
  constructor
  { intro h
    -- proof conversion process (skipped)
    sorry }
  { intro h
    -- reverse proof process (skipped)
    sorry }

end completing_square_to_simplify_eq_l42_42104


namespace sin_half_angle_identity_proof_tan_half_angle_identity_proof_cos_half_angle_identity_proof_l42_42696

noncomputable def sin_half_angle_identity (α β γ r R : ℝ) : Prop :=
  sin (α / 2) * sin (β / 2) * sin (γ / 2) = r / (4 * R)

noncomputable def tan_half_angle_identity (α β γ r p : ℝ) : Prop :=
  tan (α / 2) * tan (β / 2) * tan (γ / 2) = r / p

noncomputable def cos_half_angle_identity (α β γ p R : ℝ) : Prop :=
  cos (α / 2) * cos (β / 2) * cos (γ / 2) = p / (4 * R)

theorem sin_half_angle_identity_proof (α β γ r R : ℝ) : sin_half_angle_identity α β γ r R :=
sorry

theorem tan_half_angle_identity_proof (α β γ r p : ℝ) : tan_half_angle_identity α β γ r p :=
sorry

theorem cos_half_angle_identity_proof (α β γ p R : ℝ) : cos_half_angle_identity α β γ p R :=
sorry

end sin_half_angle_identity_proof_tan_half_angle_identity_proof_cos_half_angle_identity_proof_l42_42696


namespace binom_12_9_eq_220_l42_42573

open Nat

theorem binom_12_9_eq_220 : Nat.choose 12 9 = 220 := by
  sorry

end binom_12_9_eq_220_l42_42573


namespace functional_relationship_l42_42816

-- Define the conditions and question for Scenario ①
def scenario1 (x y k : ℝ) (h1 : k ≠ 0) : Prop :=
  y = k / x

-- Define the conditions and question for Scenario ②
def scenario2 (n S k : ℝ) (h2 : k ≠ 0) : Prop :=
  S = k / n

-- Define the conditions and question for Scenario ③
def scenario3 (t s k : ℝ) (h3 : k ≠ 0) : Prop :=
  s = k * t

-- The main theorem
theorem functional_relationship (x y n S t s k : ℝ) (h1 : k ≠ 0) :
  (scenario1 x y k h1) ∧ (scenario2 n S k h1) ∧ ¬(scenario3 t s k h1) := 
sorry

end functional_relationship_l42_42816


namespace total_days_2003_to_2006_l42_42609

theorem total_days_2003_to_2006 : 
  let days_2003 := 365
  let days_2004 := 366
  let days_2005 := 365
  let days_2006 := 365
  days_2003 + days_2004 + days_2005 + days_2006 = 1461 :=
by {
  sorry
}

end total_days_2003_to_2006_l42_42609


namespace maximum_value_of_x_minus_y_l42_42172

noncomputable def max_value_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : ℝ :=
1 + 3 * Real.sqrt 2

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
    x - y ≤ max_value_x_minus_y x y h := sorry

end maximum_value_of_x_minus_y_l42_42172


namespace part1_part2_l42_42406

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) :=
by
  sorry

end part1_part2_l42_42406


namespace third_jumper_height_l42_42932

/-- 
  Ravi can jump 39 inches high.
  Ravi can jump 1.5 times higher than the average height of three other jumpers.
  The three jumpers can jump 23 inches, 27 inches, and some unknown height x.
  Prove that the unknown height x is 28 inches.
-/
theorem third_jumper_height (x : ℝ) (h₁ : 39 = 1.5 * (23 + 27 + x) / 3) : 
  x = 28 :=
sorry

end third_jumper_height_l42_42932


namespace cut_half_meter_from_cloth_l42_42548

theorem cut_half_meter_from_cloth (initial_length : ℝ) (cut_length : ℝ) : 
  initial_length = 8 / 15 → cut_length = 1 / 30 → initial_length - cut_length = 1 / 2 := 
by
  intros h_initial h_cut
  sorry

end cut_half_meter_from_cloth_l42_42548


namespace find_first_number_l42_42814

theorem find_first_number
  (avg1 : (20 + 40 + 60) / 3 = 40)
  (avg2 : 40 - 4 = (x + 70 + 28) / 3)
  (sum_eq : x + 70 + 28 = 108) :
  x = 10 :=
by
  sorry

end find_first_number_l42_42814


namespace unused_types_l42_42310

theorem unused_types (total_resources : ℕ) (used_types : ℕ) (valid_types : ℕ) :
  total_resources = 6 → used_types = 23 → valid_types = 2^total_resources - 1 - used_types → valid_types = 40 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end unused_types_l42_42310


namespace total_resistance_l42_42220

theorem total_resistance (x y z : ℝ) (R_parallel r : ℝ)
    (hx : x = 3)
    (hy : y = 6)
    (hz : z = 4)
    (hR_parallel : 1 / R_parallel = 1 / x + 1 / y)
    (hr : r = R_parallel + z) :
    r = 6 := by
  sorry

end total_resistance_l42_42220


namespace sin_585_eq_neg_sqrt_two_div_two_l42_42853

theorem sin_585_eq_neg_sqrt_two_div_two : Real.sin (585 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_585_eq_neg_sqrt_two_div_two_l42_42853


namespace oil_drop_probability_l42_42152

theorem oil_drop_probability :
  let r_circle := 1 -- radius of the circle in cm
  let side_square := 0.5 -- side length of the square in cm
  let area_circle := π * r_circle^2
  let area_square := side_square * side_square
  (area_square / area_circle) = 1 / (4 * π) :=
by
  sorry

end oil_drop_probability_l42_42152


namespace max_value_x_minus_y_l42_42180

theorem max_value_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_value_x_minus_y_l42_42180


namespace solve_x_plus_Sx_eq_2001_l42_42796

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem solve_x_plus_Sx_eq_2001 (x : ℕ) (h : x + sum_of_digits x = 2001) : x = 1977 :=
  sorry

end solve_x_plus_Sx_eq_2001_l42_42796


namespace correct_operation_l42_42546

theorem correct_operation (x y : ℝ) : (x^3 * y^2 - y^2 * x^3 = 0) :=
by sorry

end correct_operation_l42_42546


namespace probability_not_hearing_favorite_song_l42_42003

noncomputable def num_ways_favor_not_heard_first_5_min : ℕ :=
  12! - (11! + 10!)

theorem probability_not_hearing_favorite_song :
  (num_ways_favor_not_heard_first_5_min : ℚ) / 12! = 10 / 11 := by
sorry

end probability_not_hearing_favorite_song_l42_42003


namespace mike_total_cards_l42_42804

variable (original_cards : ℕ) (birthday_cards : ℕ)

def initial_cards : ℕ := 64
def received_cards : ℕ := 18

theorem mike_total_cards :
  original_cards = 64 →
  birthday_cards = 18 →
  original_cards + birthday_cards = 82 :=
by
  intros
  sorry

end mike_total_cards_l42_42804


namespace find_constants_l42_42156

theorem find_constants
  (a_1 a_2 : ℚ)
  (h1 : 3 * a_1 - 3 * a_2 = 0)
  (h2 : 4 * a_1 + 7 * a_2 = 5) :
  a_1 = 5 / 11 ∧ a_2 = 5 / 11 :=
by
  sorry

end find_constants_l42_42156


namespace part1_solution_set_part2_range_a_l42_42387

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l42_42387


namespace triangle_proof_l42_42233

-- Defining the properties of triangle ABC
variable (A B C : Type) -- A, B, and C are points
variable [P : PointType A B C] -- Make use of a PointType class for triangle properties
variable (AB AC BC : ℝ) -- Define side lengths as real numbers

-- Introduction of the lengths
def side_AB : AB = 5 := sorry
def side_BC : BC = 4 := sorry

-- Definitions of angles at respective points
variable (angle_ABC angle_BAC angle_ACB : ℝ)

-- Propose statements
-- Statement (a)
def isosceles_triangle (AC : ℝ) (hAC : AC = 4) : angle_ABC > angle_BAC := sorry
-- Statement (c)
def degenerate_triangle (AC : ℝ) (hAC : AC = 2) : angle_ABC < angle_ACB := sorry
-- Proving statements a and c are incorrect
theorem triangle_proof : triangle_angles AB 5 BC 4 AC 4 ∧ isosceles_triangle AB BC AC → False ∧ 
                         triangle_angles AB 5 BC 4 AC 2 ∧ degenerate_triangle AB BC AC → False := 
begin
  sorry
end

end triangle_proof_l42_42233


namespace quadratic_has_two_distinct_real_roots_l42_42210

theorem quadratic_has_two_distinct_real_roots (k : ℝ) : 
  ((k - 1) * x^2 + 2 * x - 2 = 0) → (1 / 2 < k ∧ k ≠ 1) :=
sorry

end quadratic_has_two_distinct_real_roots_l42_42210


namespace find_common_ratio_sum_arithmetic_sequence_l42_42198

-- Conditions
variable {a : ℕ → ℝ}   -- a_n is a numeric sequence
variable (S : ℕ → ℝ)   -- S_n is the sum of the first n terms
variable {q : ℝ}       -- q is the common ratio
variable (k : ℕ)

-- Given: a_n is a geometric sequence with common ratio q, q ≠ 1, q ≠ 0
variable (h_geometric : ∀ n, a (n + 1) = a n * q)
variable (h_q_ne_one : q ≠ 1)
variable (h_q_ne_zero : q ≠ 0)

-- Given: S_n = a_1 * (1 - q^n) / (1 - q) when q ≠ 1 and q ≠ 0
variable (h_S : ∀ n, S n = a 1 * (1 - q ^ (n + 1)) / (1 - q))

-- Given: a_5, a_3, a_4 form an arithmetic sequence, so 2a_3 = a_5 + a_4
variable (h_arithmetic : 2 * a 3 = a 5 + a 4)

-- Prove part 1: common ratio q is -2
theorem find_common_ratio (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_arithmetic : 2 * a 3 = a 5 + a 4) 
  (h_q_ne_one : q ≠ 1) (h_q_ne_zero : q ≠ 0) : q = -2 :=
sorry

-- Prove part 2: S_(k+2), S_k, S_(k+1) form an arithmetic sequence
theorem sum_arithmetic_sequence (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_q_ne_one : q ≠ 1) (h_q_ne_zero : q ≠ 0)
  (h_S : ∀ n, S n = a 1 * (1 - q ^ (n + 1)) / (1 - q))
  (k : ℕ) : S (k + 2) + S k = 2 * S (k + 1) :=
sorry

end find_common_ratio_sum_arithmetic_sequence_l42_42198


namespace theater_seat_count_l42_42057

theorem theater_seat_count :
  ∃ n : ℕ, n < 60 ∧ n % 9 = 5 ∧ n % 6 = 3 ∧ n = 41 :=
sorry

end theater_seat_count_l42_42057


namespace chipmunk_families_went_away_l42_42153

theorem chipmunk_families_went_away :
  ∀ (total_families left_families went_away_families : ℕ),
  total_families = 86 →
  left_families = 21 →
  went_away_families = total_families - left_families →
  went_away_families = 65 :=
by
  intros total_families left_families went_away_families ht hl hw
  rw [ht, hl] at hw
  exact hw

end chipmunk_families_went_away_l42_42153


namespace f_odd_f_decreasing_f_max_min_l42_42470

noncomputable def f : ℝ → ℝ := sorry

lemma f_add (x y : ℝ) : f (x + y) = f x + f y := sorry
lemma f_neg1 : f (-1) = 2 := sorry
lemma f_positive_less_than_zero {x : ℝ} (hx : x > 0) : f x < 0 := sorry

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := sorry

theorem f_decreasing : ∀ x1 x2 : ℝ, x2 > x1 → f x2 < f x1 := sorry

theorem f_max_min : ∀ (f_max f_min : ℝ),
  f_max = f (-2) ∧ f_min = f 4 ∧
  f (-2) = 4 ∧ f 4 = -8 := sorry

end f_odd_f_decreasing_f_max_min_l42_42470


namespace original_flow_rate_l42_42307

theorem original_flow_rate (x : ℝ) (h : 2 = 0.6 * x - 1) : x = 5 :=
by
  sorry

end original_flow_rate_l42_42307


namespace find_x_l42_42206

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (8, 1/2 * x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : vector_a x = (8, 1/2 * x)) 
(h3 : vector_b x = (x, 1)) 
(h4 : ∀ k : ℝ, (vector_a x).1 = k * (vector_b x).1 ∧ 
                       (vector_a x).2 = k * (vector_b x).2) : 
                       x = 4 := sorry

end find_x_l42_42206


namespace chess_tournament_total_games_l42_42454

theorem chess_tournament_total_games (n : ℕ) (h : n = 10) : (n * (n - 1)) / 2 = 45 := by
  sorry

end chess_tournament_total_games_l42_42454


namespace animals_not_like_either_l42_42456

def total_animals : ℕ := 75
def animals_eat_carrots : ℕ := 26
def animals_like_hay : ℕ := 56
def animals_like_both : ℕ := 14

theorem animals_not_like_either : (total_animals - (animals_eat_carrots - animals_like_both + animals_like_hay - animals_like_both + animals_like_both)) = 7 := by
  sorry

end animals_not_like_either_l42_42456


namespace grain_remaining_l42_42981

def originalGrain : ℕ := 50870
def spilledGrain : ℕ := 49952
def remainingGrain : ℕ := 918

theorem grain_remaining : originalGrain - spilledGrain = remainingGrain := by
  -- calculations are omitted in the theorem statement
  sorry

end grain_remaining_l42_42981


namespace intersection_of_sets_l42_42752

noncomputable def A : Set ℝ := { x | x^2 - 1 > 0 }
noncomputable def B : Set ℝ := { x | Real.log x / Real.log 2 > 0 }

theorem intersection_of_sets :
  A ∩ B = { x | x > 1 } :=
by {
  sorry
}

end intersection_of_sets_l42_42752


namespace ad_minus_bc_divisible_by_2017_l42_42267

theorem ad_minus_bc_divisible_by_2017 
  (a b c d n : ℕ) 
  (h1 : (a * n + b) % 2017 = 0) 
  (h2 : (c * n + d) % 2017 = 0) : 
  (a * d - b * c) % 2017 = 0 :=
sorry

end ad_minus_bc_divisible_by_2017_l42_42267


namespace solve_problem_l42_42241

noncomputable def problem_statement : Prop :=
  ∀ (tons_to_pounds : ℕ) 
    (packet_weight_pounds : ℕ) 
    (packet_weight_ounces : ℕ)
    (num_packets : ℕ)
    (bag_capacity_tons : ℕ)
    (X : ℕ),
    tons_to_pounds = 2300 →
    packet_weight_pounds = 16 →
    packet_weight_ounces = 4 →
    num_packets = 1840 →
    bag_capacity_tons = 13 →
    X = (packet_weight_ounces * bag_capacity_tons * tons_to_pounds) / 
        ((bag_capacity_tons * tons_to_pounds) - (num_packets * packet_weight_pounds)) →
    X = 16

theorem solve_problem : problem_statement :=
by sorry

end solve_problem_l42_42241


namespace max_knights_cannot_be_all_liars_l42_42238

-- Define the conditions of the problem
structure Student :=
  (is_knight : Bool)
  (statement : String)

-- Define the function to check the truthfulness of statements
def is_truthful (s : Student) (conditions : List Student) : Bool :=
  -- Define how to check the statement based on conditions
  sorry

-- The maximum number of knights
theorem max_knights (N : ℕ) (students : List Student) (cond : ∀ s ∈ students, is_truthful s students = true ↔ s.is_knight) :
  ∃ M, M = N := by
  sorry

-- The school cannot be made up entirely of liars
theorem cannot_be_all_liars (N : ℕ) (students : List Student) (cond : ∀ s ∈ students, ¬is_truthful s students) :
  false := by
  sorry

end max_knights_cannot_be_all_liars_l42_42238


namespace ab_value_l42_42897

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 2) (h2 : a + b = 3) : a * b = 7/2 :=
by
  sorry

end ab_value_l42_42897


namespace find_f_when_x_lt_0_l42_42199

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_defined (f : ℝ → ℝ) : Prop :=
  ∀ x ≥ 0, f x = x^2 - 2 * x

theorem find_f_when_x_lt_0 (f : ℝ → ℝ) (h_odd : odd_function f) (h_defined : f_defined f) :
  ∀ x < 0, f x = -x^2 - 2 * x :=
by
  sorry

end find_f_when_x_lt_0_l42_42199


namespace inequality_solution_l42_42647

theorem inequality_solution (x : ℝ) :
  (2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 20) ↔
  (1 < x ∧ x < 2 ∨ 3 < x ∧ x < 6) :=
by
  sorry

end inequality_solution_l42_42647


namespace find_side_length_of_cube_l42_42133

theorem find_side_length_of_cube (n : ℕ) :
  (4 * n^2 = (1/3) * 6 * n^3) -> n = 2 :=
by
  sorry

end find_side_length_of_cube_l42_42133


namespace scientific_notation_of_population_l42_42660

theorem scientific_notation_of_population (population : Real) (h_pop : population = 6.8e6) :
    ∃ a n, (1 ≤ |a| ∧ |a| < 10) ∧ (population = a * 10^n) ∧ (a = 6.8) ∧ (n = 6) :=
by
  sorry

end scientific_notation_of_population_l42_42660


namespace balls_sold_l42_42489

theorem balls_sold (CP SP_total : ℕ) (loss : ℕ) (n : ℕ) :
  CP = 60 →
  SP_total = 720 →
  loss = 5 * CP →
  loss = n * CP - SP_total →
  n = 17 :=
by
  intros hCP hSP_total hloss htotal
  -- Your proof here
  sorry

end balls_sold_l42_42489


namespace cost_per_bar_l42_42927

variable (months_in_year : ℕ := 12)
variable (months_per_bar_of_soap : ℕ := 2)
variable (total_cost_for_year : ℕ := 48)

theorem cost_per_bar (h1 : months_per_bar_of_soap > 0)
                     (h2 : total_cost_for_year > 0) : 
    (total_cost_for_year / (months_in_year / months_per_bar_of_soap)) = 8 := 
by
  sorry

end cost_per_bar_l42_42927


namespace part1_part2_l42_42428

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l42_42428


namespace boys_in_classroom_l42_42520

theorem boys_in_classroom (total_children : ℕ) (girls_fraction : ℚ) (number_boys : ℕ) 
  (h1 : total_children = 45) (h2 : girls_fraction = 1/3) (h3 : number_boys = total_children - (total_children * girls_fraction).toNat) :
  number_boys = 30 :=
  by
    rw [h1, h2, h3]
    sorry

end boys_in_classroom_l42_42520


namespace part1_solution_set_part2_range_of_a_l42_42377

-- Part 1
theorem part1_solution_set (x : ℝ) : (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x + 3| > -a) ↔ (a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_of_a_l42_42377


namespace total_edge_length_of_parallelepiped_l42_42794

/-- Kolya has 440 identical cubes with a side length of 1 cm.
Kolya constructs a rectangular parallelepiped from these cubes 
and all edges have lengths of at least 5 cm. Prove 
that the total length of all edges of the rectangular parallelepiped is 96 cm. -/
theorem total_edge_length_of_parallelepiped {a b c : ℕ} 
  (h1 : a * b * c = 440) 
  (h2 : a ≥ 5) 
  (h3 : b ≥ 5) 
  (h4 : c ≥ 5) : 
  4 * (a + b + c) = 96 :=
sorry

end total_edge_length_of_parallelepiped_l42_42794


namespace sum_of_roots_l42_42760

-- Define the main condition
def equation (x : ℝ) : Prop :=
  (x + 3) * (x - 4) = 22

-- Define the statement we want to prove
theorem sum_of_roots : ∀ x1 x2 : ℝ, (equation x1 ∧ equation x2) → x1 + x2 = 1 := by
  intros x1 x2 h
  sorry

end sum_of_roots_l42_42760


namespace domain_tan_3x_sub_pi_over_4_l42_42263

noncomputable def domain_of_f : Set ℝ :=
  {x : ℝ | ∀ k : ℤ, x ≠ (k * Real.pi) / 3 + Real.pi / 4}

theorem domain_tan_3x_sub_pi_over_4 :
  ∀ x : ℝ, x ∈ domain_of_f ↔ ∀ k : ℤ, x ≠ (k * Real.pi) / 3 + Real.pi / 4 :=
by
  intro x
  sorry

end domain_tan_3x_sub_pi_over_4_l42_42263


namespace product_of_two_two_digit_numbers_greater_than_40_is_four_digit_l42_42509

-- Define the condition: both numbers are two-digit numbers greater than 40
def is_two_digit_and_greater_than_40 (n : ℕ) : Prop :=
  40 < n ∧ n < 100

-- Define the problem statement
theorem product_of_two_two_digit_numbers_greater_than_40_is_four_digit
  (a b : ℕ) (ha : is_two_digit_and_greater_than_40 a) (hb : is_two_digit_and_greater_than_40 b) :
  1000 ≤ a * b ∧ a * b < 10000 :=
by
  sorry

end product_of_two_two_digit_numbers_greater_than_40_is_four_digit_l42_42509


namespace rect_area_sum_eq_16_l42_42847

theorem rect_area_sum_eq_16 (a b c : ℕ) (h1 : |a * b - a * c| = 1) (h2 : |a * c - b * c| = 49) :
  a + b + c = 16 :=
sorry

end rect_area_sum_eq_16_l42_42847


namespace n_and_m_integers_and_n2_plus_m3_odd_then_n_plus_m_odd_l42_42611

theorem n_and_m_integers_and_n2_plus_m3_odd_then_n_plus_m_odd :
  ∀ (n m : ℤ), (n^2 + m^3) % 2 ≠ 0 → (n + m) % 2 = 1 :=
by sorry

end n_and_m_integers_and_n2_plus_m3_odd_then_n_plus_m_odd_l42_42611


namespace min_x_given_conditions_l42_42922

theorem min_x_given_conditions :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (100 : ℚ) / 151 < y / x ∧ y / x < (200 : ℚ) / 251 ∧ x = 3 :=
by
  sorry

end min_x_given_conditions_l42_42922


namespace shaded_area_correct_l42_42777

-- Definition of the grid dimensions
def grid_width : ℕ := 15
def grid_height : ℕ := 5

-- Definition of the heights of the shaded regions in segments
def shaded_height (x : ℕ) : ℕ :=
if x < 4 then 2
else if x < 9 then 3
else if x < 13 then 4
else if x < 15 then 5
else 0

-- Definition for the area of the entire grid
def grid_area : ℝ := grid_width * grid_height

-- Definition for the area of the unshaded triangle
def unshaded_triangle_area : ℝ := 0.5 * grid_width * grid_height

-- Definition for the area of the shaded region
def shaded_area : ℝ := grid_area - unshaded_triangle_area

-- The theorem to be proved
theorem shaded_area_correct : shaded_area = 37.5 :=
by
  sorry

end shaded_area_correct_l42_42777


namespace evaluate_expression_l42_42723

theorem evaluate_expression :
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5) + 1) = 107 :=
by
  -- The proof will go here.
  sorry

end evaluate_expression_l42_42723


namespace find_divisor_l42_42075

variable (dividend quotient remainder divisor : ℕ)

theorem find_divisor (h1 : dividend = 52) (h2 : quotient = 16) (h3 : remainder = 4) (h4 : dividend = divisor * quotient + remainder) : 
  divisor = 3 := by
  sorry

end find_divisor_l42_42075


namespace cade_marbles_now_l42_42851

def original_marbles : ℝ := 87.0
def added_marbles : ℝ := 8.0
def total_marbles : ℝ := original_marbles + added_marbles

theorem cade_marbles_now : total_marbles = 95.0 :=
by
  sorry

end cade_marbles_now_l42_42851


namespace quadratic_ineq_solution_l42_42513

theorem quadratic_ineq_solution (x : ℝ) : x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3 := 
sorry

end quadratic_ineq_solution_l42_42513


namespace complex_point_in_first_quadrant_l42_42783

theorem complex_point_in_first_quadrant : 
  let p := (1 + 3 * complex.i) * (3 - complex.i) in
  (p.re > 0) ∧ (p.im > 0) := 
by
  -- p = (1 + 3i)(3 - i)
  let p := (1 + 3 * complex.i) * (3 - complex.i)
  -- Prove that the real part of p is greater than 0
  have h_re : p.re = 6 := by sorry
  -- Prove that the imaginary part of p is greater than 0
  have h_im : p.im = 8 := by sorry
  -- Combine above results to show both parts are positive
  exact ⟨by linarith, by linarith⟩

end complex_point_in_first_quadrant_l42_42783


namespace product_expansion_l42_42722

theorem product_expansion (x : ℝ) : 2 * (x + 3) * (x + 4) = 2 * x^2 + 14 * x + 24 := 
by
  sorry

end product_expansion_l42_42722


namespace average_value_function_example_l42_42728

def average_value_function (f : ℝ → ℝ) (a b : ℝ) :=
  ∃ x0 : ℝ, a < x0 ∧ x0 < b ∧ f x0 = (f b - f a) / (b - a)

theorem average_value_function_example :
  average_value_function (λ x => x^2 - m * x - 1) (-1) (1) → 
  ∃ m : ℝ, 0 < m ∧ m < 2 :=
by
  intros h
  sorry

end average_value_function_example_l42_42728


namespace find_prime_triplets_l42_42328

def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_triplet (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ (p * (r + 1) = q * (r + 5))

theorem find_prime_triplets :
  { (p, q, r) | valid_triplet p q r } = {(3, 2, 7), (5, 3, 5), (7, 3, 2)} :=
by {
  sorry -- Proof is to be completed
}

end find_prime_triplets_l42_42328


namespace dropouts_correct_l42_42672

/-- Definition for initial racers, racers joining after 20 minutes, and racers at finish line. -/
def initial_racers : ℕ := 50
def joining_racers : ℕ := 30
def finishers : ℕ := 130

/-- Total racers after initial join and doubling. -/
def total_racers : ℕ := (initial_racers + joining_racers) * 2

/-- The number of people who dropped out before finishing the race. -/
def dropped_out : ℕ := total_racers - finishers

/-- Proof statement to show the number of people who dropped out before finishing is 30. -/
theorem dropouts_correct : dropped_out = 30 := by
  sorry

end dropouts_correct_l42_42672


namespace customers_remaining_l42_42567

theorem customers_remaining (init : ℕ) (left : ℕ) (remaining : ℕ) :
  init = 21 → left = 9 → remaining = 12 → init - left = remaining :=
by sorry

end customers_remaining_l42_42567


namespace squares_form_acute_triangle_l42_42983

theorem squares_form_acute_triangle (a b c x y z d : ℝ)
    (h_triangle : ∀ x y z : ℝ, (x > 0 ∧ y > 0 ∧ z > 0) → (x + y > z) ∧ (x + z > y) ∧ (y + z > x))
    (h_acute : ∀ x y z : ℝ, (x^2 + y^2 > z^2) ∧ (x^2 + z^2 > y^2) ∧ (y^2 + z^2 > x^2))
    (h_inscribed_squares : x = a ^ 2 * b * c / (d * a + b * c) ∧
                           y = b ^ 2 * a * c / (d * b + a * c) ∧
                           z = c ^ 2 * a * b / (d * c + a * b)) :
    (x + y > z) ∧ (x + z > y) ∧ (y + z > x) ∧
    (x^2 + y^2 > z^2) ∧ (x^2 + z^2 > y^2) ∧ (y^2 + z^2 > x^2) :=
sorry

end squares_form_acute_triangle_l42_42983


namespace parents_years_in_america_before_aziz_birth_l42_42141

noncomputable def aziz_birth_year (current_year : ℕ) (aziz_age : ℕ) : ℕ :=
  current_year - aziz_age

noncomputable def years_parents_in_america_before_aziz_birth (arrival_year : ℕ) (aziz_birth_year : ℕ) : ℕ :=
  aziz_birth_year - arrival_year

theorem parents_years_in_america_before_aziz_birth 
  (current_year : ℕ := 2021) 
  (aziz_age : ℕ := 36) 
  (arrival_year : ℕ := 1982) 
  (expected_years : ℕ := 3) :
  years_parents_in_america_before_aziz_birth arrival_year (aziz_birth_year current_year aziz_age) = expected_years :=
by 
  sorry

end parents_years_in_america_before_aziz_birth_l42_42141


namespace delta_gj_l42_42802

def vj := 120
def total := 770
def gj := total - vj

theorem delta_gj : gj - 5 * vj = 50 := by
  sorry

end delta_gj_l42_42802


namespace series_sum_to_4_l42_42209

theorem series_sum_to_4 (x : ℝ) (hx : ∑' n : ℕ, (n + 1) * x^n = 4) : x = 1 / 2 := 
sorry

end series_sum_to_4_l42_42209


namespace positive_diff_after_add_five_l42_42815

theorem positive_diff_after_add_five (y : ℝ) 
  (h : (45 + y)/2 = 32) : |45 - (y + 5)| = 21 :=
by 
  sorry

end positive_diff_after_add_five_l42_42815


namespace Ian_money_left_l42_42892

-- Definitions based on the conditions
def hours_worked : ℕ := 8
def rate_per_hour : ℕ := 18
def total_money_made : ℕ := hours_worked * rate_per_hour
def money_left : ℕ := total_money_made / 2

-- The statement to be proved 
theorem Ian_money_left : money_left = 72 :=
by
  sorry

end Ian_money_left_l42_42892


namespace part1_solution_set_part2_range_l42_42358

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l42_42358


namespace ellipse_x1_x2_squared_sum_eq_4_l42_42878

theorem ellipse_x1_x2_squared_sum_eq_4
  (x₁ y₁ x₂ y₂ : ℝ)
  (a b : ℝ)
  (ha : a = 2)
  (hb : b = 1)
  (hM : x₁^2 / a^2 + y₁^2 = 1)
  (hN : x₂^2 / a^2 + y₂^2 = 1)
  (h_slope_product : (y₁ / x₁) * (y₂ / x₂) = -1 / 4) :
  x₁^2 + x₂^2 = 4 :=
by
  sorry

end ellipse_x1_x2_squared_sum_eq_4_l42_42878


namespace triangle_shape_statements_l42_42797

theorem triangle_shape_statements (a b c : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (h : a^2 + b^2 + c^2 = ab + bc + ca) :
  (a = b ∧ b = c ∧ a = c) :=
by
  sorry 

end triangle_shape_statements_l42_42797


namespace roots_triple_relation_l42_42159

theorem roots_triple_relation (p q r α β : ℝ) (h1 : α + β = -q / p) (h2 : α * β = r / p) (h3 : β = 3 * α) :
  3 * q ^ 2 = 16 * p * r :=
sorry

end roots_triple_relation_l42_42159


namespace find_x_l42_42208

-- Definition of the problem
def infinite_series (x : ℝ) : ℝ := ∑' n : ℕ, (n + 1) * x^n

-- Given condition
axiom condition : infinite_series x = 4

-- Statement to prove
theorem find_x : (∃ x : ℝ, infinite_series x = 4) → x = 1/2 := by
  sorry

end find_x_l42_42208


namespace convert_polar_to_rectangular_example_l42_42004

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem convert_polar_to_rectangular_example :
  polar_to_rectangular 6 (5 * Real.pi / 2) = (0, 6) := by
  sorry

end convert_polar_to_rectangular_example_l42_42004


namespace solve_equation_l42_42250

theorem solve_equation (x : ℝ) (h₁ : x ≠ -11) (h₂ : x ≠ -5) (h₃ : x ≠ -12) (h₄ : x ≠ -4) :
  (1 / (x + 11) + 1 / (x + 5) = 1 / (x + 12) + 1 / (x + 4)) ↔ x = -8 :=
by
  sorry

end solve_equation_l42_42250


namespace average_speed_l42_42296

-- Define the speeds and times
def speed1 : ℝ := 120 -- km/h
def time1 : ℝ := 1 -- hour

def speed2 : ℝ := 150 -- km/h
def time2 : ℝ := 2 -- hours

def speed3 : ℝ := 80 -- km/h
def time3 : ℝ := 0.5 -- hour

-- Define the conversion factor
def km_to_miles : ℝ := 0.62

-- Calculate total distance (in kilometers)
def distance1 : ℝ := speed1 * time1
def distance2 : ℝ := speed2 * time2
def distance3 : ℝ := speed3 * time3

def total_distance_km : ℝ := distance1 + distance2 + distance3

-- Convert total distance to miles
def total_distance_miles : ℝ := total_distance_km * km_to_miles

-- Calculate total time (in hours)
def total_time : ℝ := time1 + time2 + time3

-- Final proof statement for average speed
theorem average_speed : total_distance_miles / total_time = 81.49 := by {
  sorry
}

end average_speed_l42_42296


namespace find_x_value_l42_42265

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x)
  else Real.log x * Real.log 81

theorem find_x_value (x : ℝ) (h : f x = 1 / 4) : x = 3 :=
sorry

end find_x_value_l42_42265


namespace find_k_l42_42585

theorem find_k (k : ℝ) :
    (∀ x : ℝ, 4 * x^2 + k * x + 4 ≠ 0) → k = 8 :=
sorry

end find_k_l42_42585


namespace drop_perpendicular_l42_42823

open Classical

-- Definitions for geometrical constructions on the plane
structure Point :=
(x : ℝ)
(y : ℝ)

structure Line :=
(p1 : Point)
(p2 : Point)

-- Condition 1: Drawing a line through two points
def draw_line (A B : Point) : Line := {
  p1 := A,
  p2 := B
}

-- Condition 2: Drawing a perpendicular line through a given point on a line
def draw_perpendicular (l : Line) (P : Point) : Line :=
-- Details of construction skipped, this function should return the perpendicular line
sorry

-- The problem: Given a point A and a line l not passing through A, construct the perpendicular from A to l
theorem drop_perpendicular : 
  ∀ (A : Point) (l : Line), ¬ (A = l.p1 ∨ A = l.p2) → ∃ (P : Point), ∃ (m : Line), (m = draw_perpendicular l P) ∧ (m.p1 = A) :=
by
  intros A l h
  -- Details of theorem-proof skipped, assert the existence of P and m as required
  sorry

end drop_perpendicular_l42_42823


namespace part1_part2_l42_42656

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 6

theorem part1 (a b : ℝ) (h1 : f a b 1 = 8) : a + b = 2 := by
  rw [f] at h1
  sorry

theorem part2 (a b : ℝ) (h1 : f a b (-1) = f a b 3) : f a b 2 = 6 := by
  rw [f] at h1
  sorry

end part1_part2_l42_42656


namespace people_off_second_eq_8_l42_42274

-- Initial number of people on the bus
def initial_people := 50

-- People who got off at the first stop
def people_off_first := 15

-- People who got on at the second stop
def people_on_second := 2

-- People who got off at the second stop (unknown, let's call it x)
variable (x : ℕ)

-- People who got off at the third stop
def people_off_third := 4

-- People who got on at the third stop
def people_on_third := 3

-- Number of people on the bus after the third stop
def people_after_third := 28

-- Equation formed by given conditions
def equation := initial_people - people_off_first - x + people_on_second - people_off_third + people_on_third = people_after_third

-- Goal: Prove the equation with given conditions results in x = 8
theorem people_off_second_eq_8 : equation x → x = 8 := by
  sorry

end people_off_second_eq_8_l42_42274


namespace at_least_two_equal_l42_42916

noncomputable def positive_reals (x y z : ℝ) : Prop :=
x > 0 ∧ y > 0 ∧ z > 0

noncomputable def triangle_inequality_for_n (x y z : ℝ) (n : ℕ) : Prop :=
(x^n + y^n > z^n) ∧ (y^n + z^n > x^n) ∧ (z^n + x^n > y^n)

theorem at_least_two_equal (x y z : ℝ) 
  (pos : positive_reals x y z) 
  (triangle_ineq: ∀ n : ℕ, n > 0 → triangle_inequality_for_n x y z n) : 
  x = y ∨ y = z ∨ z = x := 
sorry

end at_least_two_equal_l42_42916


namespace pounds_per_pie_l42_42081

-- Define the conditions
def total_weight : ℕ := 120
def applesauce_weight := total_weight / 2
def pies_weight := total_weight - applesauce_weight
def number_of_pies := 15

-- Define the required proof for pounds per pie
theorem pounds_per_pie :
  pies_weight / number_of_pies = 4 := by
  sorry

end pounds_per_pie_l42_42081


namespace dianas_roll_beats_apollos_max_l42_42999

/-- Define that Diana rolls a six-sided die -/
def DianaRolls := Fin 6

/-- Define that Apollo rolls two six-sided dice -/
def ApolloRolls := (Fin 6 × Fin 6)

/-- The probability that Diana's single die roll is higher than the maximum of Apollo's two rolls -/
def DianaBeatsApollo :=
  let outcomes : List (Fin 6 × Fin 6) := List.diag [0, 1, 2, 3, 4, 5]
  let diana_probability := 1 // 6
  let apollo_probabilities := [
    (1 / 36 : ℚ), (3 / 36 : ℚ),
    (5 / 36 : ℚ), (5 / 36 : ℚ),
    (7 / 36 : ℚ), (11 / 36 : ℚ)
  ]
  let beat_probability :=
    diana_probability * (apollo_probabilities.sum)
  beat_probability

/-- The proof that Diana's single die roll results in a higher number than the maximum of Apollo's two rolls
    has the probability 95/216 -/
theorem dianas_roll_beats_apollos_max :
  DianaBeatsApollo = (95 / 216 : ℚ) := sorry

end dianas_roll_beats_apollos_max_l42_42999


namespace part1_part2_l42_42361

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l42_42361


namespace jimin_shared_fruits_total_l42_42464

-- Define the quantities given in the conditions
def persimmons : ℕ := 2
def apples : ℕ := 7

-- State the theorem to be proved
theorem jimin_shared_fruits_total : persimmons + apples = 9 := by
  sorry

end jimin_shared_fruits_total_l42_42464


namespace first_player_wins_if_take_one_initial_l42_42953

theorem first_player_wins_if_take_one_initial :
  ∃ strategy : ℕ → ℕ, 
    (∀ n, strategy n = if n % 3 = 0 then 1 else 2) ∧ 
    strategy 99 = 1 ∧ 
    strategy 100 = 1 :=
sorry

end first_player_wins_if_take_one_initial_l42_42953


namespace sum_of_ages_now_l42_42108

variable (D A Al B : ℝ)

noncomputable def age_condition (D : ℝ) : Prop :=
  D = 16

noncomputable def alex_age_condition (A : ℝ) : Prop :=
  A = 60 - (30 - 16)

noncomputable def allison_age_condition (Al : ℝ) : Prop :=
  Al = 15 - (30 - 16)

noncomputable def bernard_age_condition (B A Al : ℝ) : Prop :=
  B = (A + Al) / 2

noncomputable def sum_of_ages (A Al B : ℝ) : ℝ :=
  A + Al + B

theorem sum_of_ages_now :
  age_condition D →
  alex_age_condition A →
  allison_age_condition Al →
  bernard_age_condition B A Al →
  sum_of_ages A Al B = 70.5 := by
  sorry

end sum_of_ages_now_l42_42108


namespace simplify_expression_l42_42655

theorem simplify_expression (a : ℝ) (h : a ≠ 0) : (a^9 * a^15) / a^3 = a^21 :=
by sorry

end simplify_expression_l42_42655


namespace quoted_price_of_shares_l42_42301

theorem quoted_price_of_shares :
  ∀ (investment nominal_value dividend_rate annual_income quoted_price : ℝ),
  investment = 4940 →
  nominal_value = 10 →
  dividend_rate = 14 →
  annual_income = 728 →
  quoted_price = 9.5 :=
by
  intros investment nominal_value dividend_rate annual_income quoted_price
  intros h_investment h_nominal_value h_dividend_rate h_annual_income
  sorry

end quoted_price_of_shares_l42_42301


namespace supply_lasts_for_8_months_l42_42228

-- Define the conditions
def pills_per_supply : ℕ := 120
def days_per_pill : ℕ := 2
def days_per_month : ℕ := 30

-- Define the function to calculate the duration in days
def supply_duration_in_days (pills : ℕ) (days_per_pill : ℕ) : ℕ :=
  pills * days_per_pill

-- Define the function to convert days to months
def days_to_months (days : ℕ) (days_per_month : ℕ) : ℕ :=
  days / days_per_month

-- Main statement to prove
theorem supply_lasts_for_8_months :
  days_to_months (supply_duration_in_days pills_per_supply days_per_pill) days_per_month = 8 :=
by
  sorry

end supply_lasts_for_8_months_l42_42228


namespace area_of_isosceles_trapezoid_l42_42831

variable (a b c d : ℝ) -- Variables for sides and bases of the trapezoid

-- Define isosceles trapezoid with given sides and bases
def is_isosceles_trapezoid (a b c d : ℝ) (h : ℝ) :=
  a = b ∧ c = 10 ∧ d = 16 ∧ (∃ (h : ℝ), a^2 = h^2 + ((d - c) / 2)^2 ∧ a = 5)

-- Lean theorem for the area of the isosceles trapezoid
theorem area_of_isosceles_trapezoid :
  ∀ (a b c d : ℝ) (h : ℝ), is_isosceles_trapezoid a b c d h
  → (1 / 2) * (c + d) * h = 52 :=
by
  sorry

end area_of_isosceles_trapezoid_l42_42831


namespace central_angle_of_sector_with_area_one_l42_42225

theorem central_angle_of_sector_with_area_one (θ : ℝ):
  (1 / 2) * θ = 1 → θ = 2 :=
by
  sorry

end central_angle_of_sector_with_area_one_l42_42225


namespace reduced_price_per_kg_l42_42305

-- Definitions
variables {P R Q : ℝ}

-- Conditions
axiom reduction_price : R = P * 0.82
axiom original_quantity : Q * P = 1080
axiom reduced_quantity : (Q + 8) * R = 1080

-- Proof statement
theorem reduced_price_per_kg : R = 24.30 :=
by {
  sorry
}

end reduced_price_per_kg_l42_42305


namespace possible_digits_C_multiple_of_5_l42_42870

theorem possible_digits_C_multiple_of_5 :
    ∃ (digits : Finset ℕ), (∀ x ∈ digits, x < 10) ∧ digits.card = 10 ∧ (∀ C ∈ digits, ∃ n : ℕ, n = 1000 + C * 100 + 35 ∧ n % 5 = 0) :=
by {
  sorry
}

end possible_digits_C_multiple_of_5_l42_42870


namespace lyle_payment_l42_42705

def pen_cost : ℝ := 1.50

def notebook_cost : ℝ := 3 * pen_cost

def cost_for_4_notebooks : ℝ := 4 * notebook_cost

theorem lyle_payment : cost_for_4_notebooks = 18.00 :=
by
  sorry

end lyle_payment_l42_42705


namespace possible_quadrilateral_areas_l42_42009

-- Define the problem set up
structure Point where
  x : ℝ
  y : ℝ

structure Square where
  side_length : ℝ
  A : Point
  B : Point
  C : Point
  D : Point

-- Defines the division points on each side of the square
def division_points (A B C D : Point) : List Point :=
  [
    -- Points on AB
    { x := 1, y := 4 }, { x := 2, y := 4 }, { x := 3, y := 4 },
    -- Points on BC
    { x := 4, y := 3 }, { x := 4, y := 2 }, { x := 4, y := 1 },
    -- Points on CD
    { x := 3, y := 0 }, { x := 2, y := 0 }, { x := 1, y := 0 },
    -- Points on DA
    { x := 0, y := 3 }, { x := 0, y := 2 }, { x := 0, y := 1 }
  ]

-- Possible areas calculation using the Shoelace Theorem
def quadrilateral_areas : List ℝ :=
  [6, 7, 7.5, 8, 8.5, 9, 10]

-- Math proof problem in Lean, we need to prove that the quadrilateral areas match the given values
theorem possible_quadrilateral_areas (ABCD : Square) (pts : List Point) :
    (division_points ABCD.A ABCD.B ABCD.C ABCD.D) = [
      { x := 1, y := 4 }, { x := 2, y := 4 }, { x := 3, y := 4 },
      { x := 4, y := 3 }, { x := 4, y := 2 }, { x := 4, y := 1 },
      { x := 3, y := 0 }, { x := 2, y := 0 }, { x := 1, y := 0 },
      { x := 0, y := 3 }, { x := 0, y := 2 }, { x := 0, y := 1 }
    ] → 
    (∃ areas, areas ⊆ quadrilateral_areas) := by
  sorry

end possible_quadrilateral_areas_l42_42009


namespace common_elements_count_l42_42633

theorem common_elements_count (S T : Set ℕ) (hS : S = {n | ∃ k : ℕ, k < 3000 ∧ n = 5 * (k + 1)})
    (hT : T = {n | ∃ k : ℕ, k < 3000 ∧ n = 8 * (k + 1)}) :
    S ∩ T = {n | ∃ m : ℕ, m < 375 ∧ n = 40 * (m + 1)} :=
by {
  sorry
}

end common_elements_count_l42_42633


namespace hypotenuse_length_l42_42457

theorem hypotenuse_length
    (a b c : ℝ)
    (h1: a^2 + b^2 + c^2 = 2450)
    (h2: b = a + 7)
    (h3: c^2 = a^2 + b^2) :
    c = 35 := sorry

end hypotenuse_length_l42_42457


namespace halfway_fraction_l42_42683

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/7) : (a + b) / 2 = 41/56 :=
by
  sorry

end halfway_fraction_l42_42683


namespace probability_inequality_l42_42234

noncomputable def Xi : Type := ℝ

axiom normal_distribution (μ σ : ℝ) : Xi → Prop

theorem probability_inequality (σ : ℝ) (h₁ : ∀ ξ, normal_distribution 2 σ ξ)
    (h₂ : P (λ ξ, ξ > 4) = 0.1) : P (λ ξ, ξ < 0) = 0.1 :=
by
  sorry

end probability_inequality_l42_42234


namespace integer_solution_of_floor_equation_l42_42864

theorem integer_solution_of_floor_equation (n : ℤ) : 
  (⌊n^2 / 4⌋ - ⌊n / 2⌋^2 = 5) ↔ (n = 11) :=
by sorry

end integer_solution_of_floor_equation_l42_42864


namespace quadratic_factorization_l42_42089

theorem quadratic_factorization (a b : ℕ) (h1 : x^2 - 18 * x + 72 = (x - a) * (x - b))
  (h2 : a > b) : 2 * b - a = 0 :=
sorry

end quadratic_factorization_l42_42089


namespace hyperbolas_same_asymptotes_l42_42869

theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1) ↔ (y^2 / 25 - x^2 / M = 1)) → M = 225 / 16 :=
by
  sorry

end hyperbolas_same_asymptotes_l42_42869


namespace karen_starts_late_l42_42915

def karen_speed := 60 -- Karen's speed in mph
def tom_speed := 45 -- Tom's speed in mph
def tom_distance := 24 -- Distance Tom drives in miles
def karen_lead := 4 -- Distance by which Karen beats Tom in miles

theorem karen_starts_late : 
  let t := tom_distance / tom_speed in -- Time Tom drives
  let t_k := (tom_distance + karen_lead) / karen_speed in -- Time Karen drives
  (t - t_k) * 60 = 4 := -- The time difference in minutes is 4
by
  sorry

end karen_starts_late_l42_42915


namespace find_y_value_l42_42029

variable (x y z k : ℝ)

-- Conditions
def inverse_relation_y (x y : ℝ) (k : ℝ) : Prop := 5 * y = k / (x^2)
def direct_relation_z (x z : ℝ) : Prop := 3 * z = x

-- Constant from conditions
def k_constant := 500

-- Problem statement
theorem find_y_value (h1 : inverse_relation_y 2 25 k_constant) (h2 : direct_relation_z 4 6) :
  y = 6.25 :=
by
  sorry

-- Auxiliary instance to fulfill the proof requirement
noncomputable def y_value : ℝ := 6.25

end find_y_value_l42_42029


namespace parabola_focus_coordinates_l42_42053

theorem parabola_focus_coordinates (h : ∀ y, y^2 = 4 * x) : ∃ x, x = 1 ∧ y = 0 := 
sorry

end parabola_focus_coordinates_l42_42053


namespace lyle_notebook_cost_l42_42708

theorem lyle_notebook_cost (pen_cost : ℝ) (notebook_multiplier : ℝ) (num_notebooks : ℕ) 
  (h_pen_cost : pen_cost = 1.50) (h_notebook_mul : notebook_multiplier = 3) 
  (h_num_notebooks : num_notebooks = 4) :
  (pen_cost * notebook_multiplier) * num_notebooks = 18 := 
  by
  sorry

end lyle_notebook_cost_l42_42708


namespace cassie_nails_claws_total_l42_42002

theorem cassie_nails_claws_total :
  let dogs := 4
  let parrots := 8
  let cats := 2
  let rabbits := 6
  let lizards := 5
  let tortoises := 3

  let dog_nails := dogs * 4 * 4

  let normal_parrots := 6
  let parrot_with_extra_toe := 1
  let parrot_missing_toe := 1
  let parrot_claws := (normal_parrots * 2 * 3) + (parrot_with_extra_toe * 2 * 4) + (parrot_missing_toe * 2 * 2)

  let normal_cats := 1
  let deformed_cat := 1
  let cat_toes := (1 * 4 * 5) + (1 * 4 * 4) + 1 

  let normal_rabbits := 5
  let deformed_rabbit := 1
  let rabbit_nails := (normal_rabbits * 4 * 9) + (3 * 9 + 2)

  let normal_lizards := 4
  let deformed_lizard := 1
  let lizard_toes := (normal_lizards * 4 * 5) + (deformed_lizard * 4 * 4)
  
  let normal_tortoises := 1
  let tortoise_with_extra_claw := 1
  let tortoise_missing_claw := 1
  let tortoise_claws := (normal_tortoises * 4 * 4) + (3 * 4 + 5) + (3 * 4 + 3)

  let total_nails_claws := dog_nails + parrot_claws + cat_toes + rabbit_nails + lizard_toes + tortoise_claws

  total_nails_claws = 524 :=
by
  sorry

end cassie_nails_claws_total_l42_42002


namespace probability_product_greater_than_zero_l42_42278

open ProbabilityTheory

-- Define the interval, probabilities, and the final probability
noncomputable def interval := Set.Icc (-30 : ℝ) 15
noncomputable def probability_pos := (15 / 45 : ℝ)
noncomputable def probability_neg := (30 / 45 : ℝ)
noncomputable def probability_product_gt_zero := (probability_pos ^ 2) + (probability_neg ^ 2)

-- Lean 4 statement for the proof
theorem probability_product_greater_than_zero :
  probability_product_gt_zero = (5 / 9 : ℝ) :=
by
  sorry

end probability_product_greater_than_zero_l42_42278


namespace arithmetic_sequence_ratio_l42_42650

theorem arithmetic_sequence_ratio
  (x y a1 a2 a3 b1 b2 b3 b4 : ℝ)
  (h1 : x ≠ y)
  (h2 : a1 = x + (1 * (a2 - a1)))
  (h3 : a2 = x + (2 * (a2 - a1)))
  (h4 : a3 = x + (3 * (a2 - a1)))
  (h5 : y = x + (4 * (a2 - a1)))
  (h6 : x = x)
  (h7 : b2 = x + (1 * (b3 - x)))
  (h8 : b3 = x + (2 * (b3 - x)))
  (h9 : y = x + (3 * (b3 - x)))
  (h10 : b4 = x + (4 * (b3 - x))) :
  (b4 - b3) / (a2 - a1) = 8 / 3 := by
  sorry

end arithmetic_sequence_ratio_l42_42650


namespace hannah_speed_l42_42444

theorem hannah_speed :
  ∃ H : ℝ, 
    (∀ t : ℝ, (t = 6) → d = 130) ∧ 
    (∀ t : ℝ, (t = 11) → d = 130) → 
    (d = 37 * 5 + H * 5) → 
    H = 15 := 
by 
  sorry

end hannah_speed_l42_42444


namespace find_j_of_scaled_quadratic_l42_42510

/- Define the given condition -/
def quadratic_expressed (p q r : ℝ) : Prop :=
  ∀ x : ℝ, p * x^2 + q * x + r = 5 * (x - 3)^2 + 15

/- State the theorem to be proved -/
theorem find_j_of_scaled_quadratic (p q r m j l : ℝ) (h_quad : quadratic_expressed p q r) :
  (∀ x : ℝ, 2 * p * x^2 + 2 * q * x + 2 * r = m * (x - j)^2 + l) → j = 3 :=
by
  intro h
  sorry

end find_j_of_scaled_quadratic_l42_42510


namespace pie_eating_contest_difference_l42_42902

-- Definition of given conditions
def num_students := 8
def emma_pies := 8
def sam_pies := 1

-- Statement to prove
theorem pie_eating_contest_difference :
  emma_pies - sam_pies = 7 :=
by
  -- Omitting the proof, as requested.
  sorry

end pie_eating_contest_difference_l42_42902


namespace solve_inequalities_l42_42648

theorem solve_inequalities :
  {x : ℤ | (x - 1) / 2 ≥ (x - 2) / 3 ∧ 2 * x - 5 < -3 * x} = {-1, 0} :=
by
  sorry

end solve_inequalities_l42_42648


namespace dark_chocolate_bars_sold_l42_42945

theorem dark_chocolate_bars_sold (W D : ℕ) (h₁ : 4 * D = 3 * W) (h₂ : W = 20) : D = 15 :=
by
  sorry

end dark_chocolate_bars_sold_l42_42945


namespace rank_of_matrix_A_is_2_l42_42017

def matrix_A : Matrix (Fin 4) (Fin 5) ℚ :=
  ![![3, -1, 1, 2, -8],
    ![7, -1, 2, 1, -12],
    ![11, -1, 3, 0, -16],
    ![10, -2, 3, 3, -20]]

theorem rank_of_matrix_A_is_2 : Matrix.rank matrix_A = 2 := by
  sorry

end rank_of_matrix_A_is_2_l42_42017


namespace find_monthly_salary_l42_42978

-- Definitions based on the conditions
def initial_saving_rate : ℝ := 0.25
def initial_expense_rate : ℝ := 1 - initial_saving_rate
def expense_increase_rate : ℝ := 1.25
def final_saving : ℝ := 300

-- Theorem: Prove the man's monthly salary
theorem find_monthly_salary (S : ℝ) (h1 : initial_saving_rate = 0.25)
  (h2 : initial_expense_rate = 0.75) (h3 : expense_increase_rate = 1.25)
  (h4 : final_saving = 300) : S = 4800 :=
by
  sorry

end find_monthly_salary_l42_42978


namespace part1_solution_set_part2_range_of_a_l42_42417

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l42_42417


namespace top_leftmost_rectangle_is_B_l42_42326

structure Rectangle :=
  (w x y z : ℕ)

def RectangleA := Rectangle.mk 5 1 9 2
def RectangleB := Rectangle.mk 2 0 6 3
def RectangleC := Rectangle.mk 6 7 4 1
def RectangleD := Rectangle.mk 8 4 3 5
def RectangleE := Rectangle.mk 7 3 8 0

-- Problem Statement: Given these rectangles, prove that the top leftmost rectangle is B.
theorem top_leftmost_rectangle_is_B 
  (A : Rectangle := RectangleA)
  (B : Rectangle := RectangleB)
  (C : Rectangle := RectangleC)
  (D : Rectangle := RectangleD)
  (E : Rectangle := RectangleE) : 
  B = Rectangle.mk 2 0 6 3 := 
sorry

end top_leftmost_rectangle_is_B_l42_42326


namespace circle_intersection_line_l42_42753

theorem circle_intersection_line (d : ℝ) :
  (∃ (x y : ℝ), (x - 5)^2 + (y + 2)^2 = 49 ∧ (x + 1)^2 + (y - 5)^2 = 25 ∧ x + y = d) ↔ d = 6.5 :=
by
  sorry

end circle_intersection_line_l42_42753


namespace validate_operation_l42_42964

theorem validate_operation (x y m a b : ℕ) :
  (2 * x - x ≠ 2) →
  (2 * m + 3 * m ≠ 5 * m^2) →
  (5 * xy - 4 * xy = xy) →
  (2 * a + 3 * b ≠ 5 * a * b) →
  (5 * xy - 4 * xy = xy) :=
by
  intros hA hB hC hD
  exact hC

end validate_operation_l42_42964


namespace polynomial_integer_roots_l42_42601

theorem polynomial_integer_roots
  (b c : ℤ)
  (x1 x2 x1' x2' : ℤ)
  (h_eq1 : x1 * x2 > 0)
  (h_eq2 : x1' * x2' > 0)
  (h_eq3 : x1^2 + b * x1 + c = 0)
  (h_eq4 : x2^2 + b * x2 + c = 0)
  (h_eq5 : x1'^2 + c * x1' + b = 0)
  (h_eq6 : x2'^2 + c * x2' + b = 0)
  : x1 < 0 ∧ x2 < 0 ∧ b - 1 ≤ c ∧ c ≤ b + 1 ∧ 
    ((b = 5 ∧ c = 6) ∨ (b = 6 ∧ c = 5) ∨ (b = 4 ∧ c = 4)) := 
sorry

end polynomial_integer_roots_l42_42601


namespace correct_new_encoding_l42_42531

def oldString : String := "011011010011"
def newString : String := "211221121"

def decodeOldEncoding (s : String) : String :=
  -- Decoding helper function
  sorry -- Implementation details are skipped here

def encodeNewEncoding (s : String) : String :=
  -- Encoding helper function
  sorry -- Implementation details are skipped here

axiom decodeOldEncoding_correctness :
  decodeOldEncoding oldString = "ABCBA"

axiom encodeNewEncoding_correctness :
  encodeNewEncoding "ABCBA" = newString

theorem correct_new_encoding :
  encodeNewEncoding (decodeOldEncoding oldString) = newString :=
by
  rw [decodeOldEncoding_correctness, encodeNewEncoding_correctness]
  sorry -- Proof steps are not required

end correct_new_encoding_l42_42531


namespace greatest_possible_y_l42_42812

theorem greatest_possible_y
  (x y : ℤ)
  (h : x * y + 7 * x + 6 * y = -14) : y ≤ 21 :=
sorry

end greatest_possible_y_l42_42812


namespace janet_total_l42_42625

-- Definitions based on the conditions
variable (initial_collect : ℕ) (sold : ℕ) (better_cond : ℕ)
variable (twice_size : ℕ)

-- The conditions from part a)
def janet_initial_collection := initial_collect = 10
def janet_sells := sold = 6
def janet_gets_better := better_cond = 4
def brother_gives := twice_size = 2 * (initial_collect - sold + better_cond)

-- The proof statement based on part c)
theorem janet_total (initial_collect sold better_cond twice_size : ℕ) : 
    janet_initial_collection initial_collect →
    janet_sells sold →
    janet_gets_better better_cond →
    brother_gives initial_collect sold better_cond twice_size →
    (initial_collect - sold + better_cond + twice_size = 24) :=
by
  intros h1 h2 h3 h4
  sorry

end janet_total_l42_42625


namespace tangent_line_eqn_when_a_zero_min_value_f_when_a_zero_range_of_a_for_x_ge_zero_exp_x_ln_x_plus_one_gt_x_sq_l42_42203

noncomputable def f (a : ℝ) (x : ℝ) := Real.exp x - 1 - x - a * x ^ 2

theorem tangent_line_eqn_when_a_zero :
  (∀ x, y = f 0 x → y - (Real.exp 1 - 2) = (Real.exp 1 - 1) * (x - 1)) :=
sorry

theorem min_value_f_when_a_zero :
  (∀ x : ℝ, f 0 x >= f 0 0) := 
sorry

theorem range_of_a_for_x_ge_zero (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → f a x ≥ 0) → (a ≤ 1/2) :=
sorry

theorem exp_x_ln_x_plus_one_gt_x_sq (x : ℝ) :
  x > 0 → ((Real.exp x - 1) * Real.log (x + 1) > x ^ 2) :=
sorry

end tangent_line_eqn_when_a_zero_min_value_f_when_a_zero_range_of_a_for_x_ge_zero_exp_x_ln_x_plus_one_gt_x_sq_l42_42203


namespace total_carrot_sticks_l42_42060

-- Define the number of carrot sticks James ate before and after dinner
def carrot_sticks_before_dinner : Nat := 22
def carrot_sticks_after_dinner : Nat := 15

-- Prove that the total number of carrot sticks James ate is 37
theorem total_carrot_sticks : carrot_sticks_before_dinner + carrot_sticks_after_dinner = 37 :=
  by sorry

end total_carrot_sticks_l42_42060


namespace total_runs_opponents_correct_l42_42976

-- Define the scoring conditions
def team_scores : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
def lost_games_scores : List ℕ := [3, 5, 7, 9, 11, 13]
def won_games_scores : List ℕ := [2, 4, 6, 8, 10, 12]

-- Define the total runs scored by opponents in lost games
def total_runs_lost_games : ℕ := (lost_games_scores.map (λ x => x + 1)).sum

-- Define the total runs scored by opponents in won games
def total_runs_won_games : ℕ := (won_games_scores.map (λ x => x / 2)).sum

-- Total runs scored by opponents (given)
def total_runs_opponents : ℕ := total_runs_lost_games + total_runs_won_games

-- The theorem to prove
theorem total_runs_opponents_correct : total_runs_opponents = 75 := by
  -- Proof goes here
  sorry

end total_runs_opponents_correct_l42_42976


namespace part1_part2_l42_42422

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l42_42422


namespace no_natural_number_divisible_2014_l42_42008

theorem no_natural_number_divisible_2014 : ¬∃ n s : ℕ, n = 2014 * s + 2014 := 
by
  -- Assume for contradiction that such numbers exist
  intro ⟨n, s, h⟩,
  -- Consider the transformed equation for contradiction
  have h' : n - s = 2013 * s + 2014, from sorry,
  -- Check the divisibility by 3 leading to contradiction
  have div_contr : (2013 * s + 2014) % 3 = 1, from sorry,
  -- Using the contradiction to close the proof
  sorry

end no_natural_number_divisible_2014_l42_42008


namespace abc_divisibility_l42_42886

theorem abc_divisibility (a b c : ℕ) (h1 : c ∣ a^b) (h2 : a ∣ b^c) (h3 : b ∣ c^a) : abc ∣ (a + b + c)^(a + b + c) := 
sorry

end abc_divisibility_l42_42886


namespace initial_number_of_kids_l42_42100

theorem initial_number_of_kids (joined kids_total initial : ℕ) (h1 : joined = 22) (h2 : kids_total = 36) (h3 : kids_total = initial + joined) : initial = 14 :=
by 
  -- Proof goes here
  sorry

end initial_number_of_kids_l42_42100


namespace jacket_final_price_l42_42315

theorem jacket_final_price 
  (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) (final_discount : ℝ)
  (price_after_first : ℝ := original_price * (1 - first_discount))
  (price_after_second : ℝ := price_after_first * (1 - second_discount))
  (final_price : ℝ := price_after_second * (1 - final_discount)) :
  original_price = 250 ∧ first_discount = 0.4 ∧ second_discount = 0.3 ∧ final_discount = 0.1 →
  final_price = 94.5 := 
by 
  sorry

end jacket_final_price_l42_42315


namespace equilateral_prism_lateral_edge_length_l42_42742

theorem equilateral_prism_lateral_edge_length
  (base_side_length : ℝ)
  (h_base : base_side_length = 1)
  (perpendicular_diagonals : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = base_side_length ∧ b = lateral_edge ∧ c = some_diagonal_length ∧ lateral_edge ≠ 0)
  : ∀ lateral_edge : ℝ, lateral_edge = (Real.sqrt 2) / 2 := sorry

end equilateral_prism_lateral_edge_length_l42_42742


namespace candy_sharing_l42_42446

theorem candy_sharing (Hugh_candy Tommy_candy Melany_candy shared_candy : ℕ) 
  (h1 : Hugh_candy = 8) (h2 : Tommy_candy = 6) (h3 : shared_candy = 7) :
  Hugh_candy + Tommy_candy + Melany_candy = 3 * shared_candy →
  Melany_candy = 7 :=
by
  intro h
  sorry

end candy_sharing_l42_42446


namespace garden_strawberry_yield_l42_42147

-- Definitions from the conditions
def garden_length : ℝ := 10
def garden_width : ℝ := 15
def plants_per_sq_ft : ℝ := 5
def strawberries_per_plant : ℝ := 12

-- Expected total number of strawberries
def expected_strawberries : ℝ := 9000

-- Proof statement
theorem garden_strawberry_yield : 
  (garden_length * garden_width * plants_per_sq_ft * strawberries_per_plant) = expected_strawberries :=
by sorry

end garden_strawberry_yield_l42_42147


namespace cannot_form_isosceles_triangle_l42_42268

theorem cannot_form_isosceles_triangle :
  ¬ ∃ (sticks : Finset ℕ) (a b c : ℕ), a ∈ sticks ∧ b ∈ sticks ∧ c ∈ sticks ∧
  a + b > c ∧ a + c > b ∧ b + c > a ∧ -- Triangle inequality
  (a = b ∨ b = c ∨ a = c) ∧ -- Isosceles condition
  sticks ⊆ {1, 2, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9} := sorry

end cannot_form_isosceles_triangle_l42_42268


namespace highest_financial_backing_l42_42570

-- Let x be the lowest level of financial backing
-- Define the five levels of backing as x, 6x, 36x, 216x, 1296x
-- Given that the total raised is $200,000

theorem highest_financial_backing (x : ℝ) 
  (h₁: 50 * x + 20 * 6 * x + 12 * 36 * x + 7 * 216 * x + 4 * 1296 * x = 200000) : 
  1296 * x = 35534 :=
sorry

end highest_financial_backing_l42_42570


namespace part1_part2_l42_42364

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l42_42364


namespace maximum_value_of_x_minus_y_l42_42166

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
begin
  sorry
end

end maximum_value_of_x_minus_y_l42_42166


namespace maximum_value_of_x_minus_y_l42_42167

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
begin
  sorry
end

end maximum_value_of_x_minus_y_l42_42167


namespace fraction_halfway_between_fraction_halfway_between_l42_42680

theorem fraction_halfway_between : (3/4 : ℚ) < (5/7 : ℚ) :=
by linarith

theorem fraction_halfway_between : (41 / 56 : ℚ) = (1 / 2) * ((3 / 4) + (5 / 7)) :=
by sorry

end fraction_halfway_between_fraction_halfway_between_l42_42680


namespace twice_total_credits_l42_42849

-- Define the variables and conditions
variables (Aria Emily Spencer Hannah : ℕ)
variables (h1 : Aria = 2 * Emily) 
variables (h2 : Emily = 2 * Spencer)
variables (h3 : Emily = 20)
variables (h4 : Hannah = 3 * Spencer)

-- Proof statement
theorem twice_total_credits : 2 * (Aria + Emily + Spencer + Hannah) = 200 :=
by 
  -- Proof steps are omitted with sorry
  sorry

end twice_total_credits_l42_42849


namespace part1_part2_l42_42373

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l42_42373


namespace ordering_of_a_b_c_l42_42160

noncomputable def a := 1 / Real.exp 1
noncomputable def b := Real.log 3 / 3
noncomputable def c := Real.log 4 / 4

-- We need to prove that the ordering is a > b > c.

theorem ordering_of_a_b_c : a > b ∧ b > c :=
by 
  sorry

end ordering_of_a_b_c_l42_42160


namespace f_inequality_l42_42469

def f (x : ℝ) : ℝ := sorry

axiom f_defined : ∀ x : ℝ, 0 < x → ∃ y : ℝ, f x = y

axiom f_additive : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x + f y

axiom f_positive : ∀ x : ℝ, x > 1 → f x > 0

axiom f_two : f 2 = 1

theorem f_inequality (x : ℝ) : 3 < x → x ≤ 4 → f x + f (x - 3) ≤ 2 :=
sorry

end f_inequality_l42_42469


namespace percent_gain_on_transaction_l42_42977

theorem percent_gain_on_transaction :
  ∀ (x : ℝ), (850 : ℝ) * x + (50 : ℝ) * (1.10 * ((850 : ℝ) * x / 800)) = 850 * x * (1 + 0.06875) := 
by
  intro x
  sorry

end percent_gain_on_transaction_l42_42977


namespace chairs_to_remove_l42_42560

/-- A conference hall is setting up seating for a lecture with specific conditions.
    Given the total number of chairs, chairs per row, and participants expected to attend,
    prove the number of chairs to be removed to have complete rows with the least number of empty seats. -/
theorem chairs_to_remove
  (chairs_per_row : ℕ) (total_chairs : ℕ) (expected_participants : ℕ)
  (h1 : chairs_per_row = 15)
  (h2 : total_chairs = 225)
  (h3 : expected_participants = 140) :
  total_chairs - (chairs_per_row * ((expected_participants + chairs_per_row - 1) / chairs_per_row)) = 75 :=
by
  sorry

end chairs_to_remove_l42_42560


namespace race_length_l42_42901

variables (L : ℕ)

def distanceCondition1 := L - 70
def distanceCondition2 := L - 100
def distanceCondition3 := L - 163

theorem race_length (h1 : distanceCondition1 = L - 70) 
                    (h2 : distanceCondition2 = L - 100) 
                    (h3 : distanceCondition3 = L - 163)
                    (h4 : (L - 70) / (L - 163) = (L) / (L - 100)) : 
  L = 1000 :=
sorry

end race_length_l42_42901


namespace balls_sold_l42_42488

theorem balls_sold (CP SP_total : ℕ) (loss : ℕ) (n : ℕ) :
  CP = 60 →
  SP_total = 720 →
  loss = 5 * CP →
  loss = n * CP - SP_total →
  n = 17 :=
by
  intros hCP hSP_total hloss htotal
  -- Your proof here
  sorry

end balls_sold_l42_42488


namespace coefficient_x3_expansion_l42_42162

noncomputable def normal_mean (μ : ℝ) (σ : ℝ) : ℝ := μ
noncomputable def normal_variance (μ σ : ℝ) : ℝ := σ^2
noncomputable def normal_distribution (X : ℝ) (μ σ : ℝ) := 
  ∀ x : ℝ, P(X ≤ x) = ∫ t in -∞..x, (1/(σ * sqrt (2 * π))) * exp (-(t-μ)^2 / (2*σ^2))

theorem coefficient_x3_expansion {X : ℝ} {a : ℝ} (hX : normal_distribution X 2 3) 
  (hProb : P(X ≤ 1) = P(X ≥ a)) : 
  a = 3 ∧ (∑ k in finset.range 6, choose 5 k * 3^(5-k) * (-1)^k) * 
  (∑ l in finset.range 3, choose 2 l * (6)^(2-l)) == 1620 :=
by
  sorry

end coefficient_x3_expansion_l42_42162


namespace contrapositive_inequality_l42_42342

theorem contrapositive_inequality (x : ℝ) :
  ((x + 2) * (x - 3) > 0) → (x < -2 ∨ x > 0) :=
by
  sorry

end contrapositive_inequality_l42_42342


namespace puppy_weight_is_3_8_l42_42130

noncomputable def puppy_weight_problem (p s l : ℝ) : Prop :=
  p + 2 * s + l = 38 ∧
  p + l = 3 * s ∧
  p + 2 * s = l

theorem puppy_weight_is_3_8 :
  ∃ p s l : ℝ, puppy_weight_problem p s l ∧ p = 3.8 :=
by
  sorry

end puppy_weight_is_3_8_l42_42130


namespace johns_videos_weekly_minutes_l42_42466

theorem johns_videos_weekly_minutes (daily_minutes weekly_minutes : ℕ) (short_video_length long_factor: ℕ) (short_videos_per_day long_videos_per_day days : ℕ)
  (h1 : daily_minutes = short_videos_per_day * short_video_length + long_videos_per_day * (long_factor * short_video_length))
  (h2 : weekly_minutes = daily_minutes * days)
  (h_short_videos_per_day : short_videos_per_day = 2)
  (h_long_videos_per_day : long_videos_per_day = 1)
  (h_short_video_length : short_video_length = 2)
  (h_long_factor : long_factor = 6)
  (h_weekly_minutes : weekly_minutes = 112):
  days = 7 :=
by
  sorry

end johns_videos_weekly_minutes_l42_42466


namespace geometric_sequence_solution_l42_42775

-- Define the geometric sequence a_n with a common ratio q and first term a_1
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 * q^n

-- Given conditions in the problem
variables {a : ℕ → ℝ} {q a1 : ℝ}

-- Common ratio is greater than 1
axiom ratio_gt_one : q > 1

-- Given conditions a_3a_7 = 72 and a_2 + a_8 = 27
axiom condition1 : a 3 * a 7 = 72
axiom condition2 : a 2 + a 8 = 27

-- Defining the property that we are looking to prove a_12 = 96
theorem geometric_sequence_solution :
  geometric_sequence a a1 q →
  a 12 = 96 :=
by
  -- This part of the proof would be filled in
  -- Show the conditions and relations leading to the solution a_12 = 96
  sorry

end geometric_sequence_solution_l42_42775


namespace part1_solution_part2_solution_l42_42399

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l42_42399


namespace evaluate_at_5_l42_42799

def f(x: ℝ) : ℝ := 3 * x^5 - 15 * x^4 + 27 * x^3 - 20 * x^2 - 72 * x + 40

theorem evaluate_at_5 : f 5 = 2515 :=
by
  sorry

end evaluate_at_5_l42_42799


namespace exist_circle_tangent_to_three_circles_l42_42273

variable (h1 k1 r1 h2 k2 r2 h3 k3 r3 h k r : ℝ)

def condition1 : Prop := (h - h1)^2 + (k - k1)^2 = (r + r1)^2
def condition2 : Prop := (h - h2)^2 + (k - k2)^2 = (r + r2)^2
def condition3 : Prop := (h - h3)^2 + (k - k3)^2 = (r + r3)^2

theorem exist_circle_tangent_to_three_circles : 
  ∃ (h k r : ℝ), condition1 h1 k1 r1 h k r ∧ condition2 h2 k2 r2 h k r ∧ condition3 h3 k3 r3 h k r :=
by
  sorry

end exist_circle_tangent_to_three_circles_l42_42273


namespace eight_div_pow_64_l42_42001

theorem eight_div_pow_64 (h : 64 = 8^2) : 8^15 / 64^7 = 8 := by
  sorry

end eight_div_pow_64_l42_42001


namespace boys_in_classroom_l42_42518

-- Definitions of the conditions
def total_children := 45
def girls_fraction := 1 / 3

-- The theorem we want to prove
theorem boys_in_classroom : (2 / 3) * total_children = 30 := by
  sorry

end boys_in_classroom_l42_42518


namespace part1_solution_set_part2_range_l42_42356

-- Definition of the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

-- Part (1): Solution set for a=1
theorem part1_solution_set (x : ℝ) : (-∞ < x ∧ x ≤ -4) ∨ (2 ≤ x ∧ x < ∞) ↔ f x 1 ≥ 6 := by
  sorry

-- Part (2): Range for a such that f(x) > -a
theorem part2_range (a : ℝ) : (-3/2 < a ∧ a < ∞) ↔ ∀ x, f x a > -a := by
  sorry

end part1_solution_set_part2_range_l42_42356


namespace find_ordered_pair_l42_42015

theorem find_ordered_pair (x y : ℚ) 
  (h1 : 3 * x - 18 * y = 2) 
  (h2 : 4 * y - x = 6) :
  x = -58 / 3 ∧ y = -10 / 3 :=
sorry

end find_ordered_pair_l42_42015


namespace cubic_inequality_solution_l42_42998

theorem cubic_inequality_solution (x : ℝ) (h : 0 ≤ x) : 
  x^3 - 9*x^2 - 16*x > 0 ↔ 16 < x := 
by 
  sorry

end cubic_inequality_solution_l42_42998


namespace karen_start_time_late_l42_42910

theorem karen_start_time_late
  (karen_speed : ℝ := 60) -- Karen drives at 60 mph
  (tom_speed : ℝ := 45) -- Tom drives at 45 mph
  (tom_distance : ℝ := 24) -- Tom drives 24 miles before Karen wins
  (karen_lead : ℝ := 4) -- Karen needs to beat Tom by 4 miles
  : (60 * (24 / 45) - 60 * (28 / 60)) * 60 = 4 := by
  sorry

end karen_start_time_late_l42_42910


namespace area_of_polygon_l42_42059

theorem area_of_polygon (side_length n : ℕ) (h1 : n = 36) (h2 : 36 * side_length = 72) (h3 : ∀ i, i < n → (∃ a, ∃ b, (a + b = 4) ∧ (i = 4 * a + b))) :
  (n / 4) * side_length ^ 2 = 144 :=
by
  sorry

end area_of_polygon_l42_42059


namespace problem_statement_l42_42807

variable {a b c d : ℝ}

theorem problem_statement (h : a * d - b * c = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + c * d ≠ 1 := 
sorry

end problem_statement_l42_42807


namespace prism_pyramid_fusion_l42_42643

theorem prism_pyramid_fusion :
  ∃ (result_faces result_edges result_vertices : ℕ),
    result_faces + result_edges + result_vertices = 28 ∧
    ((result_faces = 8 ∧ result_edges = 13 ∧ result_vertices = 7) ∨
    (result_faces = 7 ∧ result_edges = 12 ∧ result_vertices = 7)) :=
by
  sorry

end prism_pyramid_fusion_l42_42643


namespace point_in_first_quadrant_l42_42782

-- Define the complex numbers and the multiplication
def z1 : ℂ := 1 + 3 * Complex.i
def z2 : ℂ := 3 - Complex.i

-- State the theorem
theorem point_in_first_quadrant : (z1 * z2).re > 0 ∧ (z1 * z2).im > 0 := by
  -- Define the multiplication result
  let result := 6 + 8 * Complex.i
  have h : z1 * z2 = result := by 
    calc
      z1 * z2
        = (1 + 3 * Complex.i) * (3 - Complex.i) : by rfl
    ... = 1 * 3 + 1 * (-Complex.i) + 3 * Complex.i * 3 + 3 * Complex.i * (-Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i - 3 * (Complex.i * Complex.i) : by ring
    ... = 3 - Complex.i + 9 * Complex.i + 3 : by simp [Complex.i_sq]
    ... = 6 + 8 * Complex.i : by ring

  -- Prove the conditions for being in the first quadrant
  have h_re : (z1 * z2).re = 6 := by simp [h]
  have h_im : (z1 * z2).im = 8 := by simp [h]

  show (z1 * z2).re > 0 ∧ (z1 * z2).im > 0,
  from ⟨by {rw h_re, norm_num}, by {rw h_im, norm_num}⟩

end point_in_first_quadrant_l42_42782


namespace factorial_expression_evaluation_l42_42538

theorem factorial_expression_evaluation : (Real.sqrt ((Nat.factorial 5 * Nat.factorial 4) / Nat.factorial 2))^2 = 1440 :=
by
  sorry

end factorial_expression_evaluation_l42_42538


namespace power_mod_equality_l42_42320

theorem power_mod_equality (n : ℕ) : 
  (47 % 8 = 7) → (23 % 8 = 7) → (47 ^ 2500 - 23 ^ 2500) % 8 = 0 := 
by
  intro h1 h2
  sorry

end power_mod_equality_l42_42320


namespace inequality_solution_set_l42_42948

theorem inequality_solution_set (x : ℝ) :
  abs (1 + x + x^2 / 2) < 1 ↔ -2 < x ∧ x < 0 := by
  sorry

end inequality_solution_set_l42_42948


namespace speed_of_boat_in_still_water_l42_42290

variable (b s : ℝ) -- Speed of the boat in still water and speed of the stream

-- Condition 1: The boat goes 9 km along the stream in 1 hour
def boat_along_stream := b + s = 9

-- Condition 2: The boat goes 5 km against the stream in 1 hour
def boat_against_stream := b - s = 5

-- Theorem to prove: The speed of the boat in still water is 7 km/hr
theorem speed_of_boat_in_still_water : boat_along_stream b s → boat_against_stream b s → b = 7 := 
by
  sorry

end speed_of_boat_in_still_water_l42_42290


namespace sum_lent_eq_1100_l42_42113

def interest_rate : ℚ := 6 / 100

def period : ℕ := 8

def interest_amount (P : ℚ) : ℚ :=
  period * interest_rate * P

def total_interest_eq_principal_minus_572 (P: ℚ) : Prop :=
  interest_amount P = P - 572

theorem sum_lent_eq_1100 : ∃ P : ℚ, total_interest_eq_principal_minus_572 P ∧ P = 1100 :=
by
  use 1100
  sorry

end sum_lent_eq_1100_l42_42113


namespace total_pencils_correct_l42_42859

def pencils_per_child := 4
def num_children := 8
def total_pencils := pencils_per_child * num_children

theorem total_pencils_correct : total_pencils = 32 := by
  sorry

end total_pencils_correct_l42_42859


namespace sum_of_roots_l42_42759

-- Define the main condition
def equation (x : ℝ) : Prop :=
  (x + 3) * (x - 4) = 22

-- Define the statement we want to prove
theorem sum_of_roots : ∀ x1 x2 : ℝ, (equation x1 ∧ equation x2) → x1 + x2 = 1 := by
  intros x1 x2 h
  sorry

end sum_of_roots_l42_42759


namespace polar_coordinates_equivalence_l42_42201

theorem polar_coordinates_equivalence :
  ∀ (ρ θ1 θ2 : ℝ), θ1 = π / 3 ∧ θ2 = -5 * π / 3 →
  (ρ = 5) → 
  (ρ * Real.cos θ1 = ρ * Real.cos θ2 ∧ ρ * Real.sin θ1 = ρ * Real.sin θ2) :=
by
  sorry

end polar_coordinates_equivalence_l42_42201


namespace melody_initial_food_l42_42641

-- Conditions
variable (dogs : ℕ) (food_per_meal : ℚ) (meals_per_day : ℕ) (days_in_week : ℕ) (food_left : ℚ)
variable (initial_food : ℚ)

-- Values given in the problem statement
axiom h_dogs : dogs = 3
axiom h_food_per_meal : food_per_meal = 1/2
axiom h_meals_per_day : meals_per_day = 2
axiom h_days_in_week : days_in_week = 7
axiom h_food_left : food_left = 9

-- Theorem to prove
theorem melody_initial_food : initial_food = 30 :=
  sorry

end melody_initial_food_l42_42641


namespace system_of_equations_solution_l42_42499

theorem system_of_equations_solution 
  (x y z : ℤ) 
  (h1 : x^2 - y - z = 8) 
  (h2 : 4 * x + y^2 + 3 * z = -11) 
  (h3 : 2 * x - 3 * y + z^2 = -11) : 
  x = -3 ∧ y = 2 ∧ z = -1 :=
sorry

end system_of_equations_solution_l42_42499


namespace lyle_notebook_cost_l42_42707

theorem lyle_notebook_cost (pen_cost : ℝ) (notebook_multiplier : ℝ) (num_notebooks : ℕ) 
  (h_pen_cost : pen_cost = 1.50) (h_notebook_mul : notebook_multiplier = 3) 
  (h_num_notebooks : num_notebooks = 4) :
  (pen_cost * notebook_multiplier) * num_notebooks = 18 := 
  by
  sorry

end lyle_notebook_cost_l42_42707


namespace quadrilateral_angle_B_l42_42779

/-- In quadrilateral ABCD,
given that angle A + angle C = 150 degrees,
prove that angle B = 105 degrees. -/
theorem quadrilateral_angle_B (A C : ℝ) (B : ℝ) (h1 : A + C = 150) (h2 : A + B = 180) : B = 105 :=
by
  sorry

end quadrilateral_angle_B_l42_42779


namespace part1_part2_l42_42363

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l42_42363


namespace hexagon_area_ratio_l42_42106

open Real

theorem hexagon_area_ratio (r s : ℝ) (h_eq_diam : s = r * sqrt 3) :
    (let a1 := (3 * sqrt 3 / 2) * ((3 * r / 4) ^ 2)
     let a2 := (3 * sqrt 3 / 2) * r^2
     a1 / a2 = 9 / 16) :=
by
  sorry

end hexagon_area_ratio_l42_42106


namespace scientific_notation_of_18500000_l42_42459

-- Definition of scientific notation function
def scientific_notation (n : ℕ) : string := sorry

-- Problem statement
theorem scientific_notation_of_18500000 : 
  scientific_notation 18500000 = "1.85 × 10^7" :=
sorry

end scientific_notation_of_18500000_l42_42459


namespace cherries_cost_l42_42018

def cost_per_kg (total_cost kilograms : ℕ) : ℕ :=
  total_cost / kilograms

theorem cherries_cost 
  (genevieve_amount : ℕ) 
  (short_amount : ℕ)
  (total_kilograms : ℕ) 
  (total_cost : ℕ := genevieve_amount + short_amount) 
  (cost : ℕ := cost_per_kg total_cost total_kilograms) : 
  cost = 8 :=
by
  have h1 : genevieve_amount = 1600 := by sorry
  have h2 : short_amount = 400 := by sorry
  have h3 : total_kilograms = 250 := by sorry
  sorry

end cherries_cost_l42_42018


namespace moss_flower_pollen_scientific_notation_l42_42858

theorem moss_flower_pollen_scientific_notation (d : ℝ) (h : d = 0.0000084) : ∃ n : ℤ, d = 8.4 * 10^n ∧ n = -6 :=
by
  use -6
  rw [h]
  simp
  sorry

end moss_flower_pollen_scientific_notation_l42_42858


namespace differentiable_function_zero_l42_42632

theorem differentiable_function_zero
    (f : ℝ → ℝ)
    (h_diff : Differentiable ℝ f)
    (h_zero : f 0 = 0)
    (h_ineq : ∀ x : ℝ, 0 < |f x| ∧ |f x| < 1/2 → |deriv f x| ≤ |f x * Real.log (|f x|)|) :
    ∀ x : ℝ, f x = 0 :=
by
  sorry

end differentiable_function_zero_l42_42632


namespace domain_of_function_l42_42006

open Real

theorem domain_of_function : 
  ∀ x, 
    (x + 1 ≠ 0) ∧ 
    (-x^2 - 3 * x + 4 > 0) ↔ 
    (-4 < x ∧ x < -1) ∨ ( -1 < x ∧ x < 1) := 
by 
  sorry

end domain_of_function_l42_42006


namespace gcd_of_polynomials_l42_42348

theorem gcd_of_polynomials (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5959 * k) :
  Int.gcd (4 * b^2 + 73 * b + 156) (4 * b + 15) = 1 :=
by
  sorry

end gcd_of_polynomials_l42_42348


namespace min_value_fraction_l42_42020

theorem min_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : -3 ≤ y ∧ y ≤ 1) : (x + y) / x = 0.8 :=
by
  sorry

end min_value_fraction_l42_42020


namespace bunnies_count_l42_42486

def total_pets : ℕ := 36
def percent_bunnies : ℝ := 1 - 0.25 - 0.5
def number_of_bunnies : ℕ := total_pets * (percent_bunnies)

theorem bunnies_count :
  number_of_bunnies = 9 := by
  sorry

end bunnies_count_l42_42486


namespace positive_n_for_modulus_eq_l42_42736

theorem positive_n_for_modulus_eq (n : ℕ) (h_pos : 0 < n) (h_eq : Complex.abs (5 + (n : ℂ) * Complex.I) = 5 * Real.sqrt 26) : n = 25 :=
by
  sorry

end positive_n_for_modulus_eq_l42_42736


namespace percentage_change_l42_42968

def original_income (P T : ℝ) : ℝ :=
  P * T

def new_income (P T : ℝ) : ℝ :=
  (P * 1.3333) * (T * 0.6667)

theorem percentage_change (P T : ℝ) (hP : P ≠ 0) (hT : T ≠ 0) :
  ((new_income P T - original_income P T) / original_income P T) * 100 = -11.11 :=
by
  sorry

end percentage_change_l42_42968


namespace part1_l42_42434

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 3)

theorem part1 {x : ℝ} : f x ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) := by
  sorry

end part1_l42_42434


namespace exactly_2_std_devs_less_than_mean_l42_42259

noncomputable def mean : ℝ := 14.5
noncomputable def std_dev : ℝ := 1.5
noncomputable def value : ℝ := mean - 2 * std_dev

theorem exactly_2_std_devs_less_than_mean : value = 11.5 := by
  sorry

end exactly_2_std_devs_less_than_mean_l42_42259


namespace range_u_of_given_condition_l42_42592

theorem range_u_of_given_condition (x y : ℝ) (h : x^2 / 3 + y^2 = 1) :
  1 ≤ |2 * x + y - 4| + |3 - x - 2 * y| ∧ |2 * x + y - 4| + |3 - x - 2 * y| ≤ 13 := 
sorry

end range_u_of_given_condition_l42_42592


namespace total_driving_routes_l42_42990

def num_starting_points : ℕ := 4
def num_destinations : ℕ := 3

theorem total_driving_routes (h1 : ¬(num_starting_points = 0)) (h2 : ¬(num_destinations = 0)) : 
  num_starting_points * num_destinations = 12 :=
by
  sorry

end total_driving_routes_l42_42990


namespace ratio_of_x_l42_42449

theorem ratio_of_x (x : ℝ) (h : x = Real.sqrt 7 + Real.sqrt 6) :
    ((x + 1 / x) / (x - 1 / x)) = (Real.sqrt 7 / Real.sqrt 6) :=
by
  sorry

end ratio_of_x_l42_42449


namespace ferry_round_trip_time_increases_l42_42561

variable {S V a b : ℝ}

theorem ferry_round_trip_time_increases (h1 : V > 0) (h2 : a < b) (h3 : V > a) (h4 : V > b) :
  (S / (V + b) + S / (V - b)) > (S / (V + a) + S / (V - a)) :=
by sorry

end ferry_round_trip_time_increases_l42_42561


namespace matrix_power_A_100_l42_42467

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![0, 0, 1],![1, 0, 0],![0, 1, 0]]

theorem matrix_power_A_100 : A^100 = A := by sorry

end matrix_power_A_100_l42_42467


namespace percentage_of_number_l42_42240

variable (N P : ℝ)

theorem percentage_of_number 
  (h₁ : (1 / 4) * (1 / 3) * (2 / 5) * N = 10) 
  (h₂ : (P / 100) * N = 120) : 
  P = 40 := 
by 
  sorry

end percentage_of_number_l42_42240


namespace sum_of_possible_values_of_x_l42_42758

theorem sum_of_possible_values_of_x (x : ℝ) (h : (x + 3) * (x - 4) = 22) : ∃ (x1 x2 : ℝ), x^2 - x - 34 = 0 ∧ x1 + x2 = 1 :=
by
  sorry

end sum_of_possible_values_of_x_l42_42758


namespace students_taking_neither_l42_42073

-- Defining given conditions as Lean definitions
def total_students : ℕ := 70
def students_math : ℕ := 42
def students_physics : ℕ := 35
def students_chemistry : ℕ := 25
def students_math_physics : ℕ := 18
def students_math_chemistry : ℕ := 10
def students_physics_chemistry : ℕ := 8
def students_all_three : ℕ := 5

-- Define the problem to prove
theorem students_taking_neither : total_students
  - (students_math - students_math_physics - students_math_chemistry + students_all_three
    + students_physics - students_math_physics - students_physics_chemistry + students_all_three
    + students_chemistry - students_math_chemistry - students_physics_chemistry + students_all_three
    + students_math_physics - students_all_three
    + students_math_chemistry - students_all_three
    + students_physics_chemistry - students_all_three
    + students_all_three) = 0 := by
  sorry

end students_taking_neither_l42_42073


namespace second_solution_sugar_percentage_l42_42830

theorem second_solution_sugar_percentage
  (initial_solution_pct : ℝ)
  (second_solution_pct : ℝ)
  (initial_solution_amount : ℝ)
  (final_solution_pct : ℝ)
  (replaced_fraction : ℝ)
  (final_amount : ℝ) :
  initial_solution_pct = 0.1 →
  final_solution_pct = 0.17 →
  replaced_fraction = 1/4 →
  initial_solution_amount = 100 →
  final_amount = 100 →
  second_solution_pct = 0.38 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end second_solution_sugar_percentage_l42_42830


namespace max_x_minus_y_l42_42186

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : 
  (∃ z : ℝ, z = x - y ∧ (∀ w : ℝ, w = x - y → w ≤ 1 + 3 * real.sqrt 2) ∧ (∃ u : ℝ, u = x - y ∧ u = 1 + 3 * real.sqrt 2)) :=
sorry

end max_x_minus_y_l42_42186


namespace matrix_power_four_l42_42319

theorem matrix_power_four :
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![
    ![3 * Real.sqrt 2, -3],
    ![3, 3 * Real.sqrt 2]
  ]
  (A ^ 4 = ![
    ![ -81, 0],
    ![0, -81]
  ]) :=
by
  sorry

end matrix_power_four_l42_42319


namespace T_n_bounds_l42_42741

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 1

noncomputable def S_n (n : ℕ) : ℕ := n * (n + 2)

noncomputable def b_n (n : ℕ) : ℚ := 
if n ≤ 4 then 2 * n + 1
else 1 / (n * (n + 2))

noncomputable def T_n (n : ℕ) : ℚ := 
if n ≤ 4 then S_n n
else (24 : ℚ) + (1 / 2) * (1 / 5 + 1 / 6 - 1 / (n + 1 : ℚ) - 1 / (n + 2 : ℚ))

theorem T_n_bounds (n : ℕ) : 3 ≤ T_n n ∧ T_n n < 24 + 11 / 60 := by
  sorry

end T_n_bounds_l42_42741


namespace initial_kids_count_l42_42097

-- Define the initial number of kids as a variable
def init_kids (current_kids join_kids : Nat) : Nat :=
  current_kids - join_kids

-- Define the total current kids and kids joined
def current_kids : Nat := 36
def join_kids : Nat := 22

-- Prove that the initial number of kids was 14
theorem initial_kids_count : init_kids current_kids join_kids = 14 :=
by
  -- Proof skipped
  sorry

end initial_kids_count_l42_42097


namespace simplify_expression_l42_42809

variable (b c d x y : ℝ)

theorem simplify_expression :
  (cx * (b^2 * x^3 + 3 * b^2 * y^3 + c^3 * y^3) + dy * (b^2 * x^3 + 3 * c^3 * x^3 + c^3 * y^3)) / (cx + dy) 
  = b^2 * x^3 + 3 * c^2 * xy^3 + c^3 * y^3 :=
by sorry

end simplify_expression_l42_42809


namespace digits_solution_l42_42789

noncomputable def validate_reverse_multiplication
  (A B C D E : ℕ) : Prop :=
  (A * 10000 + B * 1000 + C * 100 + D * 10 + E) * 4 =
  (E * 10000 + D * 1000 + C * 100 + B * 10 + A)

theorem digits_solution :
  validate_reverse_multiplication 2 1 9 7 8 :=
by
  sorry

end digits_solution_l42_42789


namespace find_r_squared_l42_42508

noncomputable def parabola_intersect_circle_radius_squared : Prop :=
  ∀ (x y : ℝ), y = (x - 1)^2 ∧ x - 3 = (y + 2)^2 → (x - 3/2)^2 + (y + 3/2)^2 = 1/2

theorem find_r_squared : parabola_intersect_circle_radius_squared :=
sorry

end find_r_squared_l42_42508


namespace part1_solution_set_part2_range_a_l42_42393

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l42_42393


namespace pizza_fraction_eaten_l42_42112

theorem pizza_fraction_eaten :
  let a := (1 / 4 : ℚ)
  let r := (1 / 2 : ℚ)
  let n := 6
  (a * (1 - r ^ n) / (1 - r)) = 63 / 128 :=
by
  let a := (1 / 4 : ℚ)
  let r := (1 / 2 : ℚ)
  let n := 6
  sorry

end pizza_fraction_eaten_l42_42112


namespace concyclic_projections_of_concyclic_quad_l42_42231

variables {A B C D A' B' C' D' : Type*}

def are_concyclic (p1 p2 p3 p4: Type*) : Prop :=
  sorry -- Assume we have a definition for concyclic property of points

def are_orthogonal_projection (x y : Type*) (l : Type*) : Type* :=
  sorry -- Assume we have a definition for orthogonal projection of a point on line

theorem concyclic_projections_of_concyclic_quad
  (hABCD : are_concyclic A B C D)
  (hA'_proj : are_orthogonal_projection A A' (BD))
  (hC'_proj : are_orthogonal_projection C C' (BD))
  (hB'_proj : are_orthogonal_projection B B' (AC))
  (hD'_proj : are_orthogonal_projection D D' (AC)) :
  are_concyclic A' B' C' D' :=
sorry

end concyclic_projections_of_concyclic_quad_l42_42231


namespace solve_eq1_solve_eq2_solve_eq3_l42_42084

theorem solve_eq1 (x : ℝ) : 5 * x - 2.9 = 12 → x = 1.82 :=
by
  intro h
  -- Additional steps to verify should be here
  sorry

theorem solve_eq2 (x : ℝ) : 10.5 * x + 0.6 * x = 44 → x = 3 :=
by
  intro h
  -- Additional steps to verify should be here
  sorry

theorem solve_eq3 (x : ℝ) : 8 * x / 2 = 1.5 → x = 0.375 :=
by
  intro h
  -- Additional steps to verify should be here
  sorry

end solve_eq1_solve_eq2_solve_eq3_l42_42084


namespace percentage_increase_is_50_l42_42121

def initialNumber := 80
def finalNumber := 120

theorem percentage_increase_is_50 : ((finalNumber - initialNumber) / initialNumber : ℝ) * 100 = 50 := 
by 
  sorry

end percentage_increase_is_50_l42_42121


namespace part1_solution_set_part2_range_a_l42_42390

-- Part (1)
theorem part1_solution_set (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |x - 1| + |x + 3|) :
  a = 1 → {x : ℝ | f x ≥ 6} = {x | x ≤ -4} ∪ {x | x ≥ 2} :=
by
  intros h1
  rw h
  sorry

-- Part (2)
theorem part2_range_a (f : ℝ → ℝ) (h : ∀ x a, f x = |x - a| + |x + 3|) :
  {a : ℝ | ∀ x, f x > -a} = {a | a > -3 / 2} :=
by
  rw h
  sorry

end part1_solution_set_part2_range_a_l42_42390


namespace ab_zero_l42_42693

theorem ab_zero (a b : ℝ) (x : ℝ) (h : ∀ x : ℝ, a * x + b * x ^ 2 = -(a * (-x) + b * (-x) ^ 2)) : a * b = 0 :=
by
  sorry

end ab_zero_l42_42693


namespace width_of_bottom_trapezium_l42_42855

theorem width_of_bottom_trapezium (top_width : ℝ) (area : ℝ) (depth : ℝ) (bottom_width : ℝ) 
  (h_top_width : top_width = 10)
  (h_area : area = 640)
  (h_depth : depth = 80) :
  bottom_width = 6 :=
by
  -- Problem description: calculating the width of the bottom of the trapezium given the conditions.
  sorry

end width_of_bottom_trapezium_l42_42855


namespace factory_sample_capacity_l42_42836

theorem factory_sample_capacity (n : ℕ) (a_ratio b_ratio c_ratio : ℕ) 
  (total_ratio : a_ratio + b_ratio + c_ratio = 10) (a_sample : ℕ)
  (h : a_sample = 16) (h_ratio : a_ratio = 2) :
  n = 80 :=
by
  -- sample calculations proof would normally be here
  sorry

end factory_sample_capacity_l42_42836


namespace union_of_sets_l42_42024

variable (x : ℝ)

def A : Set ℝ := {x | 0 < x ∧ x < 3}
def B : Set ℝ := {x | -1 < x ∧ x < 2}
def target : Set ℝ := {x | -1 < x ∧ x < 3}

theorem union_of_sets : (A ∪ B) = target :=
by
  sorry

end union_of_sets_l42_42024


namespace circle_radius_triple_area_l42_42939

noncomputable def circle_radius (n : ℝ) : ℝ :=
  let r := (n * (Real.sqrt 3 + 1)) / 2
  r

theorem circle_radius_triple_area (r n : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) :
  r = (n * (Real.sqrt 3 + 1)) / 2 :=
by
  sorry

end circle_radius_triple_area_l42_42939


namespace part1_part2_l42_42375

noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1) ≥ 6 ↔ x ∈ set.Iic (-4) ∪ set.Ici 2 := 
sorry

theorem part2 (a : ℝ) :
  (∀ (x : ℝ), f x a > -a) ↔ a > (-(3/2)) := 
sorry

end part1_part2_l42_42375


namespace average_speed_of_Car_X_l42_42724

noncomputable def average_speed_CarX (V_x : ℝ) : Prop :=
  let head_start_time := 1.2
  let distance_traveled_by_CarX := 98
  let speed_CarY := 50
  let time_elapsed := distance_traveled_by_CarX / speed_CarY
  (distance_traveled_by_CarX / time_elapsed) = V_x

theorem average_speed_of_Car_X : average_speed_CarX 50 :=
  sorry

end average_speed_of_Car_X_l42_42724


namespace actual_cost_l42_42846

theorem actual_cost (x : ℝ) (h : 0.80 * x = 200) : x = 250 :=
sorry

end actual_cost_l42_42846


namespace part1_part2_l42_42367

open Real

def f (x a : ℝ) := abs (x - a) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 >= 6 → x ≤ -4 ∨ x ≥ 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x, f x a > -a) → a > -3 / 2 :=
  sorry

end part1_part2_l42_42367


namespace consecutive_integers_product_divisible_l42_42292

theorem consecutive_integers_product_divisible (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) :
  ∀ n : ℕ, ∃ (x y : ℕ), (n ≤ x) ∧ (x < n + b) ∧ (n ≤ y) ∧ (y < n + b) ∧ (x ≠ y) ∧ (a * b ∣ x * y) :=
by
  sorry

end consecutive_integers_product_divisible_l42_42292


namespace sum_of_first_5n_l42_42211

theorem sum_of_first_5n (n : ℕ) (h : (4 * n * (4 * n + 1)) / 2 = (2 * n * (2 * n + 1)) / 2 + 504) :
  (5 * n * (5 * n + 1)) / 2 = 1035 :=
sorry

end sum_of_first_5n_l42_42211


namespace total_votes_election_l42_42903

theorem total_votes_election (total_votes fiona_votes elena_votes devin_votes : ℝ) 
  (Fiona_fraction : fiona_votes = (4/15) * total_votes)
  (Elena_fiona : elena_votes = fiona_votes + 15)
  (Devin_elena : devin_votes = 2 * elena_votes)
  (total_eq : total_votes = fiona_votes + elena_votes + devin_votes) :
  total_votes = 675 := 
sorry

end total_votes_election_l42_42903


namespace solve_for_xy_l42_42249

theorem solve_for_xy (x y : ℝ) 
  (h1 : 0.05 * x + 0.07 * (30 + x) = 14.9)
  (h2 : 0.03 * y - 5.6 = 0.07 * x) : 
  x = 106.67 ∧ y = 435.567 := 
  by 
  sorry

end solve_for_xy_l42_42249


namespace find_a9_l42_42883

variable (a : ℕ → ℤ)

-- Condition 1: The sequence is arithmetic
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (d : ℤ), ∀ n, a (n + 1) = a n + d

-- Condition 2: Given a_4 = 5
def a4_value (a : ℕ → ℤ) : Prop :=
  a 4 = 5

-- Condition 3: Given a_5 = 4
def a5_value (a : ℕ → ℤ) : Prop :=
  a 5 = 4

-- Problem: Prove a_9 = 0
theorem find_a9 (h1 : arithmetic_sequence a) (h2 : a4_value a) (h3 : a5_value a) : a 9 = 0 := 
sorry

end find_a9_l42_42883


namespace find_dividend_l42_42139

theorem find_dividend (partial_product : ℕ) (remainder : ℕ) (divisor quotient : ℕ) :
  partial_product = 2015 → 
  remainder = 0 →
  divisor = 105 → 
  quotient = 197 → 
  divisor * quotient + remainder = partial_product → 
  partial_product * 10 = 20685 :=
by {
  -- Proof skipped
  sorry
}

end find_dividend_l42_42139


namespace greatest_divisor_l42_42157

theorem greatest_divisor (d : ℕ) (h1 : 4351 % d = 8) (h2 : 5161 % d = 10) : d = 1 :=
by
  -- Proof goes here
  sorry

end greatest_divisor_l42_42157


namespace find_f_2017_l42_42941

theorem find_f_2017 (f : ℤ → ℤ) (h_odd : ∀ x, f (-x) = -f x) (h_period : ∀ x, f (x + 3) = f x) (h_f_neg1 : f (-1) = 1) : 
  f 2017 = -1 :=
sorry

end find_f_2017_l42_42941


namespace total_length_of_board_l42_42832

-- Define variables for the lengths
variable (S L : ℝ)

-- Given conditions as Lean definitions
def condition1 : Prop := 2 * S = L + 4
def condition2 : Prop := S = 8.0

-- The goal is to prove the total length of the board is 20.0 feet
theorem total_length_of_board (h1 : condition1 S L) (h2 : condition2 S) : S + L = 20.0 := by
  sorry

end total_length_of_board_l42_42832


namespace final_position_is_east_8km_total_fuel_consumption_is_4_96liters_l42_42713

-- Define the travel distances
def travel_distances : List ℤ := [17, -9, 7, 11, -15, -3]

-- Define the fuel consumption rate
def fuel_consumption_rate : ℝ := 0.08

-- Theorem stating the final position
theorem final_position_is_east_8km :
  List.sum travel_distances = 8 :=
by
  sorry

-- Theorem stating the total fuel consumption
theorem total_fuel_consumption_is_4_96liters :
  (List.sum (travel_distances.map fun x => |x| : List ℝ)) * fuel_consumption_rate = 4.96 :=
by
  sorry

end final_position_is_east_8km_total_fuel_consumption_is_4_96liters_l42_42713


namespace range_m_inequality_l42_42603

theorem range_m_inequality (m : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → x^2 * Real.exp x < m) ↔ m > Real.exp 1 := 
  by
    sorry

end range_m_inequality_l42_42603


namespace part1_solution_part2_solution_l42_42401

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l42_42401


namespace total_weekly_cost_correct_l42_42909

def daily_consumption (cups_per_day : ℕ) (ounces_per_cup : ℝ) : ℝ :=
  cups_per_day * ounces_per_cup

def weekly_consumption (cups_per_day : ℕ) (ounces_per_cup : ℝ) (days_per_week : ℕ) : ℝ :=
  daily_consumption cups_per_day ounces_per_cup * days_per_week

def weekly_cost (weekly_ounces : ℝ) (cost_per_ounce : ℝ) : ℝ :=
  weekly_ounces * cost_per_ounce

def person_A_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 3 0.4 7) 1.40

def person_B_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 1 0.6 7) 1.20

def person_C_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 2 0.5 5) 1.35

def james_weekly_cost : ℝ :=
  weekly_cost (weekly_consumption 2 0.5 7) 1.25

def total_weekly_cost : ℝ :=
  person_A_weekly_cost + person_B_weekly_cost + person_C_weekly_cost + james_weekly_cost

theorem total_weekly_cost_correct : total_weekly_cost = 32.30 := by
  unfold total_weekly_cost person_A_weekly_cost person_B_weekly_cost person_C_weekly_cost james_weekly_cost
  unfold weekly_cost weekly_consumption daily_consumption
  sorry

end total_weekly_cost_correct_l42_42909


namespace part1_part2_l42_42427

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l42_42427


namespace sum_radical_conjugate_problem_statement_l42_42726

theorem sum_radical_conjugate (a : ℝ) (b : ℝ) : (a - b) + (a + b) = 2 * a :=
by sorry

theorem problem_statement : (12 - real.sqrt 2023) + (12 + real.sqrt 2023) = 24 :=
by sorry

end sum_radical_conjugate_problem_statement_l42_42726


namespace cone_volume_increase_l42_42616

open Real

noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h
noncomputable def new_height (h : ℝ) : ℝ := 2 * h
noncomputable def new_volume (r h : ℝ) : ℝ := cone_volume r (new_height h)

theorem cone_volume_increase (r h : ℝ) : new_volume r h = 2 * (cone_volume r h) :=
by
  sorry

end cone_volume_increase_l42_42616


namespace sheets_of_paper_in_each_box_l42_42063

theorem sheets_of_paper_in_each_box (E S : ℕ) (h1 : 2 * E + 40 = S) (h2 : 4 * (E - 40) = S) : S = 240 :=
by
  sorry

end sheets_of_paper_in_each_box_l42_42063


namespace price_of_case_l42_42025

variables (bottles_per_day : ℚ) (days : ℕ) (bottles_per_case : ℕ) (total_spent : ℚ)

def total_bottles_consumed (bottles_per_day : ℚ) (days : ℕ) : ℚ :=
  bottles_per_day * days

def cases_needed (total_bottles : ℚ) (bottles_per_case : ℕ) : ℚ :=
  total_bottles / bottles_per_case

def price_per_case (total_spent : ℚ) (cases : ℚ) : ℚ :=
  total_spent / cases

theorem price_of_case (h1 : bottles_per_day = 1/2)
                      (h2 : days = 240)
                      (h3 : bottles_per_case = 24)
                      (h4 : total_spent = 60) :
  price_per_case total_spent (cases_needed (total_bottles_consumed bottles_per_day days) bottles_per_case) = 12 := 
sorry

end price_of_case_l42_42025


namespace sequence_a100_l42_42512

theorem sequence_a100 :
  ∃ a : ℕ → ℕ, (a 1 = 1) ∧ (∀ m n : ℕ, 0 < m → 0 < n → a (n + m) = a n + a m + n * m) ∧ (a 100 = 5050) :=
by
  sorry

end sequence_a100_l42_42512


namespace inequality_solution_l42_42745

theorem inequality_solution (a c : ℝ) (h : ∀ x : ℝ, (1/3 < x ∧ x < 1/2) ↔ ax^2 + 5*x + c > 0) : a + c = -7 :=
sorry

end inequality_solution_l42_42745


namespace binary_to_decimal_1100_l42_42576

theorem binary_to_decimal_1100 : 
  (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 0 * 2^0) = 12 := 
by
  sorry

end binary_to_decimal_1100_l42_42576


namespace option_A_option_B_option_C_option_D_l42_42735

noncomputable def f (x : ℝ) : ℝ := 2 / x + Real.log x

theorem option_A : ∀ x : ℝ, 0 < x → f x > 0 := sorry

theorem option_B (a : ℝ) : 1 + Real.log 2 < a → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 0 < x1 ∧ 0 < x2 ∧ f x1 = a ∧ f x2 = a := sorry

noncomputable def g (x : ℝ) : ℝ := f x - x

theorem option_C : ∃! x : ℝ, 0 < x ∧ g x = 0 := sorry

theorem option_D : ¬ ( ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 0 < x1 ∧ 0 < x2 ∧ g x1 = 0 ∧ g x2 = 0 ) := sorry

end option_A_option_B_option_C_option_D_l42_42735


namespace total_amount_l42_42834

-- Define p, q, r and their shares
variables (p q r : ℕ)

-- Given conditions translated to Lean definitions
def ratio_pq := (5 * q) = (4 * p)
def ratio_qr := (9 * r) = (10 * q)
def r_share := r = 400

-- Statement to prove
theorem total_amount (hpq : ratio_pq p q) (hqr : ratio_qr q r) (hr : r_share r) :
  (p + q + r) = 1210 :=
by
  sorry

end total_amount_l42_42834


namespace geometric_series_sum_test_l42_42854

-- Let's define all necessary variables
variable (a : ℤ) (r : ℤ) (n : ℕ)

-- Define the geometric series sum formula
noncomputable def geometric_series_sum (a r : ℤ) (n : ℕ) : ℤ :=
  a * ((r ^ n - 1) / (r - 1))

-- Define the specific test case as per our conditions
theorem geometric_series_sum_test :
  geometric_series_sum (-2) 3 7 = -2186 :=
by
  sorry

end geometric_series_sum_test_l42_42854


namespace math_problem_l42_42597

theorem math_problem (a b c : ℚ) 
  (h1 : a * (-2) = 1)
  (h2 : |b + 2| = 5)
  (h3 : c = 5 - 6) :
  4 * a - b + 3 * c = -8 ∨ 4 * a - b + 3 * c = 2 :=
by
  sorry

end math_problem_l42_42597


namespace triangle_side_b_length_l42_42462

noncomputable def length_of_side_b (A B C a b c : ℝ) (h1 : a = 1)
  (h2 : Real.cos A = 4/5) (h3 : Real.cos C = 5/13) : Prop :=
  b = 21 / 13

theorem triangle_side_b_length (A B C a b c : ℝ) (h1 : a = 1)
  (h2 : Real.cos A = 4/5) (h3 : Real.cos C = 5/13) :
  length_of_side_b A B C a b c h1 h2 h3 :=
by
  sorry

end triangle_side_b_length_l42_42462


namespace part1_part2_l42_42408

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) :=
by
  sorry

end part1_part2_l42_42408


namespace extra_marks_15_l42_42700

theorem extra_marks_15 {T P : ℝ} (h1 : 0.30 * T = P - 30) (h2 : 0.45 * T = P + 15) (h3 : P = 120) : 
  0.45 * T - P = 15 := 
by
  sorry

end extra_marks_15_l42_42700


namespace find_number_of_men_l42_42622

noncomputable def initial_conditions (x : ℕ) : ℕ × ℕ :=
  let men := 4 * x
  let women := 5 * x
  (men, women)

theorem find_number_of_men (x : ℕ) : 
  let (initial_men, initial_women) := initial_conditions x in
  let men_after_entry := initial_men + 2 in
  let women_after_leaving := initial_women - 3 in
  2 * women_after_leaving = 24 →
  men_after_entry = 14 :=
by
  intros
  sorry

end find_number_of_men_l42_42622


namespace quadratic_root_m_l42_42054

theorem quadratic_root_m (m : ℝ) : (∃ x : ℝ, x^2 + x + m^2 - 1 = 0 ∧ x = 0) → (m = 1 ∨ m = -1) :=
by 
  sorry

end quadratic_root_m_l42_42054


namespace new_average_marks_l42_42905

theorem new_average_marks
  (orig_avg : ℕ) (num_papers : ℕ)
  (add_geography : ℕ) (add_history : ℕ)
  (H_orig_avg : orig_avg = 63)
  (H_num_papers : num_papers = 11)
  (H_add_geography : add_geography = 20)
  (H_add_history : add_history = 2) :
  (orig_avg * num_ppapers + add_geography + add_history) / num_papers = 65 :=
by
  -- Here would be the proof steps
  sorry

end new_average_marks_l42_42905


namespace part1_solution_set_part2_range_of_a_l42_42421

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l42_42421


namespace complex_point_in_first_quadrant_l42_42788

theorem complex_point_in_first_quadrant : 
  let z := (1 + 3 * complex.i) * (3 - complex.i) in
  (z.re > 0) ∧ (z.im > 0) :=
by {
  let z := (1 + 3 * complex.i) * (3 - complex.i),
  sorry
}

end complex_point_in_first_quadrant_l42_42788


namespace sum_of_fractions_l42_42322

def S_1 : List ℚ := List.range' 1 10 |>.map (λ n => n / 10)
def S_2 : List ℚ := List.replicate 4 (20 / 10)

def total_sum : ℚ := S_1.sum + S_2.sum

theorem sum_of_fractions : total_sum = 12.5 := by
  sorry

end sum_of_fractions_l42_42322


namespace well_depth_is_correct_l42_42564

noncomputable def depth_of_well : ℝ :=
  122500

theorem well_depth_is_correct (d t1 : ℝ) : 
  t1 = Real.sqrt (d / 20) ∧ 
  (d / 1100) + t1 = 10 →
  d = depth_of_well := 
by
  sorry

end well_depth_is_correct_l42_42564


namespace simplify_sqrt8_minus_sqrt2_l42_42497

theorem simplify_sqrt8_minus_sqrt2 :
  (Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2) :=
sorry

end simplify_sqrt8_minus_sqrt2_l42_42497


namespace num_of_veg_people_l42_42453

def only_veg : ℕ := 19
def both_veg_nonveg : ℕ := 12

theorem num_of_veg_people : only_veg + both_veg_nonveg = 31 := by 
  sorry

end num_of_veg_people_l42_42453


namespace min_surveyed_consumers_l42_42132

theorem min_surveyed_consumers (N : ℕ) 
    (h10 : ∃ k : ℕ, N = 10 * k)
    (h30 : ∃ l : ℕ, N = 10 * l) 
    (h40 : ∃ m : ℕ, N = 5 * m) : 
    N = 10 :=
by
  sorry

end min_surveyed_consumers_l42_42132


namespace fill_tank_with_reduced_bucket_capacity_l42_42521

theorem fill_tank_with_reduced_bucket_capacity (C : ℝ) :
    let original_buckets := 200
    let original_capacity := C
    let new_capacity := (4 / 5) * original_capacity
    let new_buckets := 250
    (original_buckets * original_capacity) = ((new_buckets) * new_capacity) :=
by
    sorry

end fill_tank_with_reduced_bucket_capacity_l42_42521


namespace sum_last_two_digits_l42_42281

theorem sum_last_two_digits (n : ℕ) (h1 : n = 20) : (9^n + 11^n) % 100 = 1 :=
by
  sorry

end sum_last_two_digits_l42_42281


namespace comparison_l42_42738

open Real

noncomputable def a := 5 * log (2 ^ exp 1)
noncomputable def b := 2 * log (5 ^ exp 1)
noncomputable def c := 10

theorem comparison : c > a ∧ a > b :=
by
  have a_def : a = 5 * log (2 ^ exp 1) := rfl
  have b_def : b = 2 * log (5 ^ exp 1) := rfl
  have c_def : c = 10 := rfl
  sorry -- Proof goes here

end comparison_l42_42738


namespace flight_up_speed_l42_42930

variable (v : ℝ) -- speed on the flight up
variable (d : ℝ) -- distance to mother's place

/--
Given:
1. The speed on the way home was 72 mph.
2. The average speed for the trip was 91 mph.

Prove:
The speed on the flight up was 123.62 mph.
-/
theorem flight_up_speed
  (h1 : 72 > 0)
  (h2 : 91 > 0)
  (avg_speed_def : 91 = (2 * d) / ((d / v) + (d / 72))) :
  v = 123.62 :=
by
  sorry

end flight_up_speed_l42_42930


namespace rectangular_area_l42_42813

theorem rectangular_area (length width : ℝ) (h₁ : length = 0.4) (h₂ : width = 0.22) :
  (length * width = 0.088) :=
by sorry

end rectangular_area_l42_42813


namespace a_c3_b3_equiv_zero_l42_42200

-- Definitions based on conditions
def cubic_eq_has_geom_progression_roots (a b c : ℝ) :=
  ∃ d q : ℝ, d ≠ 0 ∧ q ≠ 0 ∧ d + d * q + d * q^2 = -a ∧
    d^2 * q * (1 + q + q^2) = b ∧
    d^3 * q^3 = -c

-- Main theorem to prove
theorem a_c3_b3_equiv_zero (a b c : ℝ) :
  cubic_eq_has_geom_progression_roots a b c → a^3 * c - b^3 = 0 :=
by
  sorry

end a_c3_b3_equiv_zero_l42_42200


namespace correct_calculation_l42_42544

theorem correct_calculation (a : ℝ) : -3 * a - 2 * a = -5 * a :=
by
  sorry

end correct_calculation_l42_42544


namespace find_y_value_l42_42539

theorem find_y_value
  (y z : ℝ)
  (h1 : y + z + 175 = 360)
  (h2 : z = y + 10) :
  y = 88 :=
by
  sorry

end find_y_value_l42_42539


namespace sun_salutations_per_year_l42_42649

-- Definitions 
def sun_salutations_per_weekday : ℕ := 5
def weekdays_per_week : ℕ := 5
def weeks_per_year : ℕ := 52

-- Problem statement to prove
theorem sun_salutations_per_year :
  sun_salutations_per_weekday * weekdays_per_week * weeks_per_year = 1300 :=
by
  sorry

end sun_salutations_per_year_l42_42649


namespace quadratic_roots_diff_square_l42_42798

theorem quadratic_roots_diff_square :
  ∀ (d e : ℝ), (∀ x : ℝ, 4 * x^2 + 8 * x - 48 = 0 → (x = d ∨ x = e)) → (d - e)^2 = 49 :=
by
  intros d e h
  sorry

end quadratic_roots_diff_square_l42_42798


namespace factorial_15_base_9_zeroes_l42_42889

theorem factorial_15_base_9_zeroes :
  (∃ k, 15! % 9^k = 0 ∧ 15! % 9^(k+1) ≠ 0) ∧ 
  (∀ k', 15! % 9^(k'+1) = 0 ↔ k' < 3) := sorry

end factorial_15_base_9_zeroes_l42_42889


namespace cannot_bisect_segment_with_ruler_l42_42244

noncomputable def projective_transformation (A B M : Point) : Point :=
  -- This definition will use an unspecified projective transformation that leaves A and B invariant
  sorry

theorem cannot_bisect_segment_with_ruler (A B : Point) (method : Point -> Point -> Point) :
  (forall (phi : Point -> Point), phi A = A -> phi B = B -> phi (method A B) ≠ method A B) ->
  ¬ (exists (M : Point), method A B = M) := by
  sorry

end cannot_bisect_segment_with_ruler_l42_42244


namespace cos_seven_theta_l42_42046

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (7 * θ) = -160481 / 2097152 := by
  sorry

end cos_seven_theta_l42_42046


namespace boys_in_classroom_l42_42519

theorem boys_in_classroom (total_children : ℕ) (girls_fraction : ℚ) (number_boys : ℕ) 
  (h1 : total_children = 45) (h2 : girls_fraction = 1/3) (h3 : number_boys = total_children - (total_children * girls_fraction).toNat) :
  number_boys = 30 :=
  by
    rw [h1, h2, h3]
    sorry

end boys_in_classroom_l42_42519


namespace initial_legos_l42_42327

-- Definitions and conditions
def legos_won : ℝ := 17.0
def legos_now : ℝ := 2097.0

-- The statement to prove
theorem initial_legos : (legos_now - legos_won) = 2080 :=
by sorry

end initial_legos_l42_42327


namespace find_m_l42_42298

def g (n : ℤ) : ℤ :=
if n % 2 = 1 then n + 5 else n / 2

theorem find_m (m : ℤ) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 35) : m = 135 :=
sorry

end find_m_l42_42298


namespace time_in_1876_minutes_from_6AM_is_116PM_l42_42261

def minutesToTime (startTime : Nat) (minutesToAdd : Nat) : Nat × Nat :=
  let totalMinutes := startTime + minutesToAdd
  let totalHours := totalMinutes / 60
  let remainderMinutes := totalMinutes % 60
  let resultHours := (totalHours % 24)
  (resultHours, remainderMinutes)

theorem time_in_1876_minutes_from_6AM_is_116PM :
  minutesToTime (6 * 60) 1876 = (13, 16) :=
  sorry

end time_in_1876_minutes_from_6AM_is_116PM_l42_42261


namespace greatest_multiple_5_7_less_than_700_l42_42961

theorem greatest_multiple_5_7_less_than_700 :
  ∃ n, n < 700 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ (∀ m, m < 700 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≤ n) → n = 665 :=
by
  sorry

end greatest_multiple_5_7_less_than_700_l42_42961


namespace amanda_jogging_distance_l42_42569

/-- Amanda's jogging path and the distance calculation. -/
theorem amanda_jogging_distance:
  let east_leg := 1.5
  let northwest_leg := 2
  let southwest_leg := 1
  -- Convert runs to displacement components
  let nw_x := northwest_leg / Real.sqrt 2
  let nw_y := northwest_leg / Real.sqrt 2
  let sw_x := southwest_leg / Real.sqrt 2
  let sw_y := southwest_leg / Real.sqrt 2
  -- Calculate net displacements
  let net_east := east_leg - (nw_x + sw_x)
  let net_north := nw_y - sw_y
  -- Final distance back to starting point
  let distance := Real.sqrt (net_east^2 + net_north^2)
  distance = Real.sqrt ((1.5 - 3 * Real.sqrt 2 / 2)^2 + (Real.sqrt 2 / 2)^2) := sorry

end amanda_jogging_distance_l42_42569


namespace part1_solution_part2_solution_l42_42397

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l42_42397


namespace precious_stones_l42_42850

variable (total_amount : ℕ) (price_per_stone : ℕ) (number_of_stones : ℕ)

theorem precious_stones (h1 : total_amount = 14280) (h2 : price_per_stone = 1785) : number_of_stones = 8 :=
by
  sorry

end precious_stones_l42_42850


namespace find_b_l42_42347

noncomputable def h (x : ℝ) : ℝ := x^2 + 9
noncomputable def j (x : ℝ) : ℝ := x^2 + 1

theorem find_b (b : ℝ) (hjb : h (j b) = 15) (b_pos : b > 0) : b = Real.sqrt (Real.sqrt 6 - 1) := by
  sorry

end find_b_l42_42347


namespace triangle_perimeter_from_medians_l42_42747

theorem triangle_perimeter_from_medians (m1 m2 m3 : ℕ) (h1 : m1 = 3) (h2 : m2 = 4) (h3 : m3 = 6) :
  ∃ (p : ℕ), p = 26 :=
by sorry

end triangle_perimeter_from_medians_l42_42747


namespace missing_fraction_is_correct_l42_42819

theorem missing_fraction_is_correct :
  (1 / 3 + 1 / 2 + -5 / 6 + 1 / 5 + -9 / 20 + -9 / 20) = 0.45 - (23 / 20) :=
by
  sorry

end missing_fraction_is_correct_l42_42819


namespace greg_distance_work_to_market_l42_42445

-- Given conditions translated into definitions
def total_distance : ℝ := 40
def time_from_market_to_home : ℝ := 0.5  -- in hours
def speed_from_market_to_home : ℝ := 20  -- in miles per hour

-- Distance calculation from farmer's market to home
def distance_from_market_to_home := speed_from_market_to_home * time_from_market_to_home

-- Definition for the distance from workplace to the farmer's market
def distance_from_work_to_market := total_distance - distance_from_market_to_home

-- The theorem to be proved
theorem greg_distance_work_to_market : distance_from_work_to_market = 30 := by
  -- Skipping the detailed proof
  sorry

end greg_distance_work_to_market_l42_42445


namespace maximum_value_of_x_minus_y_l42_42169

noncomputable def max_value_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : ℝ :=
1 + 3 * Real.sqrt 2

theorem maximum_value_of_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) :
    x - y ≤ max_value_x_minus_y x y h := sorry

end maximum_value_of_x_minus_y_l42_42169


namespace complement_in_N_l42_42441

variable (M : Set ℕ) (N : Set ℕ)
def complement_N (M N : Set ℕ) : Set ℕ := { x ∈ N | x ∉ M }

theorem complement_in_N (M : Set ℕ) (N : Set ℕ) : 
  M = {2, 3, 4} → N = {0, 2, 3, 4, 5} → complement_N M N = {0, 5} :=
by
  intro hM hN
  subst hM
  subst hN 
  -- sorry is used to skip the proof
  sorry

end complement_in_N_l42_42441


namespace expr_divisible_by_120_l42_42245

theorem expr_divisible_by_120 (m : ℕ) : 120 ∣ (m^5 - 5 * m^3 + 4 * m) :=
sorry

end expr_divisible_by_120_l42_42245


namespace cost_of_paper_l42_42092

noncomputable def cost_of_paper_per_kg (edge_length : ℕ) (coverage_per_kg : ℕ) (expenditure : ℕ) : ℕ :=
  let surface_area := 6 * edge_length * edge_length
  let paper_needed := surface_area / coverage_per_kg
  expenditure / paper_needed

theorem cost_of_paper (h1 : edge_length = 10) (h2 : coverage_per_kg = 20) (h3 : expenditure = 1800) : 
  cost_of_paper_per_kg 10 20 1800 = 60 :=
by
  -- Using the hypothesis to directly derive the result.
  unfold cost_of_paper_per_kg
  sorry

end cost_of_paper_l42_42092


namespace panthers_score_l42_42938

theorem panthers_score (P : ℕ) (wildcats_score : ℕ := 36) (score_difference : ℕ := 19) (h : wildcats_score = P + score_difference) : P = 17 := by
  sorry

end panthers_score_l42_42938


namespace red_marbles_count_l42_42335

theorem red_marbles_count :
  ∀ (total marbles white yellow green red : ℕ),
    total = 50 →
    white = total / 2 →
    yellow = 12 →
    green = yellow / 2 →
    red = total - (white + yellow + green) →
    red = 7 :=
by
  intros total marbles white yellow green red Htotal Hwhite Hyellow Hgreen Hred
  sorry

end red_marbles_count_l42_42335


namespace min_vertices_in_hex_grid_l42_42715

-- Define a hexagonal grid and the condition on the midpoint property.
def hexagonal_grid (p : ℤ × ℤ) : Prop :=
  ∃ m n : ℤ, p = (m, n)

-- Statement: Prove that among any 9 points in a hexagonal grid, there are two points whose midpoint is also a grid point.
theorem min_vertices_in_hex_grid :
  ∀ points : Finset (ℤ × ℤ), points.card = 9 →
  (∃ p1 p2 : (ℤ × ℤ), p1 ∈ points ∧ p2 ∈ points ∧ 
  (∃ midpoint : ℤ × ℤ, hexagonal_grid midpoint ∧ midpoint = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2))) :=
by
  intros points h_points_card
  sorry

end min_vertices_in_hex_grid_l42_42715


namespace find_cos_7theta_l42_42037

theorem find_cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = 1105 / 16384 :=
by
  sorry

end find_cos_7theta_l42_42037


namespace roots_condition_l42_42658

theorem roots_condition (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 3 ∧ x2 < 3 ∧ x1^2 - m * x1 + 2 * m = 0 ∧ x2^2 - m * x2 + 2 * m = 0) ↔ m > 9 :=
by sorry

end roots_condition_l42_42658


namespace weight_difference_l42_42087

variables (W_A W_B W_C W_D W_E : ℝ)

def condition1 : Prop := (W_A + W_B + W_C) / 3 = 84
def condition2 : Prop := (W_A + W_B + W_C + W_D) / 4 = 80
def condition3 : Prop := (W_B + W_C + W_D + W_E) / 4 = 79
def condition4 : Prop := W_A = 80

theorem weight_difference (h1 : condition1 W_A W_B W_C) 
                          (h2 : condition2 W_A W_B W_C W_D) 
                          (h3 : condition3 W_B W_C W_D W_E) 
                          (h4 : condition4 W_A) : 
                          W_E - W_D = 8 :=
by
  sorry

end weight_difference_l42_42087


namespace calculate_original_lemon_price_l42_42541

variable (p_lemon_old p_lemon_new p_grape_old p_grape_new : ℝ)
variable (num_lemons num_grapes revenue : ℝ)

theorem calculate_original_lemon_price :
  ∀ (L : ℝ),
  -- conditions
  p_lemon_old = L ∧
  p_lemon_new = L + 4 ∧
  p_grape_old = 7 ∧
  p_grape_new = 9 ∧
  num_lemons = 80 ∧
  num_grapes = 140 ∧
  revenue = 2220 ->
  -- proof that the original price is 8
  p_lemon_old = 8 :=
by
  intros L h
  have h1 : p_lemon_new = L + 4 := h.2.1
  have h2 : p_grape_old = 7 := h.2.2.1
  have h3 : p_grape_new = 9 := h.2.2.2.1
  have h4 : num_lemons = 80 := h.2.2.2.2.1
  have h5 : num_grapes = 140 := h.2.2.2.2.2.1
  have h6 : revenue = 2220 := h.2.2.2.2.2.2
  sorry

end calculate_original_lemon_price_l42_42541
