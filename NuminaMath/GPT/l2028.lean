import Mathlib

namespace min_dist_AB_l2028_202837

-- Definitions of the conditions
structure Point3D where
  x : Float
  y : Float
  z : Float

def O := Point3D.mk 0 0 0
def B := Point3D.mk (Float.sqrt 3) (Float.sqrt 2) 2

def dist (P Q : Point3D) : Float :=
  Float.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)

-- Given points
variables (A : Point3D)
axiom AO_eq_1 : dist A O = 1

-- Minimum value of |AB|
theorem min_dist_AB : dist A B ≥ 2 := 
sorry

end min_dist_AB_l2028_202837


namespace perfect_squares_divide_l2028_202829

-- Define the problem and the conditions as Lean definitions
def numFactors (base exponent : ℕ) := (exponent / 2) + 1

def countPerfectSquareFactors : ℕ := 
  let choices2 := numFactors 2 3
  let choices3 := numFactors 3 5
  let choices5 := numFactors 5 7
  let choices7 := numFactors 7 9
  choices2 * choices3 * choices5 * choices7

theorem perfect_squares_divide (numFactors : (ℕ → ℕ → ℕ)) 
(countPerfectSquareFactors : ℕ) : countPerfectSquareFactors = 120 :=
by
  -- We skip the proof here
  -- Proof steps would go here if needed
  sorry

end perfect_squares_divide_l2028_202829


namespace trees_per_square_meter_l2028_202833

-- Definitions of the given conditions
def side_length : ℕ := 100
def total_trees : ℕ := 120000

def area_of_street : ℤ := side_length * side_length
def area_of_forest : ℤ := 3 * area_of_street

-- The question translated to Lean theorem statement
theorem trees_per_square_meter (h1: area_of_street = side_length * side_length)
    (h2: area_of_forest = 3 * area_of_street) 
    (h3: total_trees = 120000) : 
    (total_trees / area_of_forest) = 4 :=
sorry

end trees_per_square_meter_l2028_202833


namespace num_three_digit_powers_of_three_l2028_202803

theorem num_three_digit_powers_of_three : 
  ∃ n1 n2 : ℕ, 100 ≤ 3^n1 ∧ 3^n1 ≤ 999 ∧ 100 ≤ 3^n2 ∧ 3^n2 ≤ 999 ∧ n1 ≠ n2 ∧ 
  (∀ n : ℕ, 100 ≤ 3^n ∧ 3^n ≤ 999 → n = n1 ∨ n = n2) :=
sorry

end num_three_digit_powers_of_three_l2028_202803


namespace A_inter_complement_B_eq_set_minus_one_to_zero_l2028_202881

open Set

theorem A_inter_complement_B_eq_set_minus_one_to_zero :
  let U := @univ ℝ
  let A := {x : ℝ | x < 0}
  let B := {x : ℝ | x ≤ -1}
  A ∩ (U \ B) = {x : ℝ | -1 < x ∧ x < 0} := 
by
  sorry

end A_inter_complement_B_eq_set_minus_one_to_zero_l2028_202881


namespace inequality_solution_set_l2028_202839

theorem inequality_solution_set :
  {x : ℝ | x ≠ 0 ∧ x ≠ 2 ∧ (x / (x - 2) + (x + 3) / (3 * x) ≥ 4)} =
  {x : ℝ | 0 < x ∧ x ≤ 1 / 8} ∪ {x : ℝ | 2 < x ∧ x ≤ 6} :=
by
  -- Proof will go here
  sorry

end inequality_solution_set_l2028_202839


namespace ab_cd_value_l2028_202887

theorem ab_cd_value (a b c d : ℝ) 
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = 0) : 
  a * b + c * d = -31 :=
by
  sorry

end ab_cd_value_l2028_202887


namespace number_of_real_a_l2028_202812

open Int

-- Define the quadratic equation with integer roots
def quadratic_eq_with_integer_roots (a : ℝ) : Prop :=
  ∃ (r s : ℤ), r + s = -a ∧ r * s = 12 * a

-- Prove there are exactly 9 values of a such that the quadratic equation has only integer roots
theorem number_of_real_a (n : ℕ) : n = 9 ↔ ∃ (as : Finset ℝ), as.card = n ∧ ∀ a ∈ as, quadratic_eq_with_integer_roots a :=
by
  -- We can skip the proof with "sorry"
  sorry

end number_of_real_a_l2028_202812


namespace find_interest_rate_l2028_202818

-- Definitions for conditions
def principal : ℝ := 12500
def interest : ℝ := 1500
def time : ℝ := 1

-- Interest rate to prove
def interest_rate : ℝ := 0.12

-- Formal statement to prove
theorem find_interest_rate (P I T : ℝ) (hP : P = principal) (hI : I = interest) (hT : T = time) : I = P * interest_rate * T :=
by
  sorry

end find_interest_rate_l2028_202818


namespace quadratic_expression_l2028_202835

theorem quadratic_expression (x1 x2 : ℝ) (h1 : x1^2 - 3 * x1 + 1 = 0) (h2 : x2^2 - 3 * x2 + 1 = 0) : 
  x1^2 - 2 * x1 + x2 = 2 :=
sorry

end quadratic_expression_l2028_202835


namespace barley_percentage_is_80_l2028_202807

variables (T C : ℝ) -- Total land and cleared land
variables (B : ℝ) -- Percentage of cleared land planted with barley

-- Given conditions
def cleared_land (T : ℝ) : ℝ := 0.9 * T
def total_land_approx : ℝ := 1000
def potato_land (C : ℝ) : ℝ := 0.1 * C
def tomato_land : ℝ := 90
def barley_percentage (C : ℝ) (B : ℝ) : Prop := C - (potato_land C) - tomato_land = (B / 100) * C

-- Theorem statement to prove
theorem barley_percentage_is_80 :
  cleared_land total_land_approx = 900 → barley_percentage 900 80 :=
by
  intros hC
  rw [cleared_land, total_land_approx] at hC
  simp [barley_percentage, potato_land, tomato_land]
  sorry

end barley_percentage_is_80_l2028_202807


namespace average_episodes_per_year_l2028_202806

theorem average_episodes_per_year (total_years : ℕ) (n1 n2 n3 e1 e2 e3 : ℕ) 
  (h1 : total_years = 14)
  (h2 : n1 = 8) (h3 : e1 = 15)
  (h4 : n2 = 4) (h5 : e2 = 20)
  (h6 : n3 = 2) (h7 : e3 = 12) :
  (n1 * e1 + n2 * e2 + n3 * e3) / total_years = 16 := by
  -- Skip the proof steps
  sorry

end average_episodes_per_year_l2028_202806


namespace find_values_of_x_and_y_l2028_202802

-- Define the conditions
def first_condition (x : ℝ) : Prop := 0.75 / x = 5 / 7
def second_condition (y : ℝ) : Prop := y / 19 = 11 / 3

-- Define the main theorem to prove
theorem find_values_of_x_and_y (x y : ℝ) (h1 : first_condition x) (h2 : second_condition y) :
  x = 1.05 ∧ y = 209 / 3 := 
by 
  sorry

end find_values_of_x_and_y_l2028_202802


namespace value_of_expression_l2028_202896

theorem value_of_expression : 2 - (-2 : ℝ) ^ (-2 : ℝ) = 7 / 4 := 
by 
  sorry

end value_of_expression_l2028_202896


namespace length_of_room_l2028_202877

theorem length_of_room (area_in_sq_inches : ℕ) (length_of_side_in_feet : ℕ) (h1 : area_in_sq_inches = 14400)
  (h2 : length_of_side_in_feet * length_of_side_in_feet = area_in_sq_inches / 144) : length_of_side_in_feet = 10 :=
  by
  sorry

end length_of_room_l2028_202877


namespace find_x0_l2028_202813

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3
noncomputable def g (x : ℝ) : ℝ := 3 * x - 5

theorem find_x0 :
  (∃ x0 : ℝ, f (g x0) = 1) → (∃ x0 : ℝ, x0 = 4/3) :=
by
  sorry

end find_x0_l2028_202813


namespace ratio_of_ages_three_years_ago_l2028_202886

theorem ratio_of_ages_three_years_ago (k Y_c : ℕ) (h1 : 45 - 3 = k * (Y_c - 3)) (h2 : (45 + 7) + (Y_c + 7) = 83) : (45 - 3) / (Y_c - 3) = 2 :=
by {
  sorry
}

end ratio_of_ages_three_years_ago_l2028_202886


namespace john_got_rolls_l2028_202827

def cost_per_dozen : ℕ := 5
def money_spent : ℕ := 15
def rolls_per_dozen : ℕ := 12

theorem john_got_rolls : (money_spent / cost_per_dozen) * rolls_per_dozen = 36 :=
by sorry

end john_got_rolls_l2028_202827


namespace evaluate_expression_l2028_202872

theorem evaluate_expression : (532 * 532) - (531 * 533) = 1 := by
  sorry

end evaluate_expression_l2028_202872


namespace three_right_angled_triangles_l2028_202855

theorem three_right_angled_triangles 
  (a b c : ℕ)
  (h_area : 1/2 * (a * b) = 2 * (a + b + c))
  (h_pythagorean : a^2 + b^2 = c^2)
  (h_int_sides : a > 0 ∧ b > 0 ∧ c > 0) :
  (a = 9 ∧ b = 40 ∧ c = 41) ∨ 
  (a = 10 ∧ b = 24 ∧ c = 26) ∨ 
  (a = 12 ∧ b = 16 ∧ c = 20) := 
sorry

end three_right_angled_triangles_l2028_202855


namespace zookeeper_feeding_ways_l2028_202856

/-- We define the total number of ways the zookeeper can feed all the animals following the rules. -/
def feed_animal_ways : ℕ :=
  6 * 5^2 * 4^2 * 3^2 * 2^2 * 1^2

/-- Theorem statement: The number of ways to feed all the animals is 86400. -/
theorem zookeeper_feeding_ways : feed_animal_ways = 86400 :=
by
  sorry

end zookeeper_feeding_ways_l2028_202856


namespace initial_girls_is_11_l2028_202879

-- Definitions of initial parameters and transformations
def initially_girls_percent : ℝ := 0.35
def final_girls_percent : ℝ := 0.25
def three : ℝ := 3

-- 35% of the initial total is girls
def initially_girls (p : ℝ) : ℝ := initially_girls_percent * p
-- After three girls leave and three boys join, the count of girls
def final_girls (p : ℝ) : ℝ := initially_girls p - three

-- Using the condition that after the change, 25% are girls
def proof_problem : Prop := ∀ (p : ℝ), 
  (final_girls p) / p = final_girls_percent →
  (0.1 * p) = 3 → 
  initially_girls p = 11

-- The statement of the theorem to be proved in Lean 4
theorem initial_girls_is_11 : proof_problem := sorry

end initial_girls_is_11_l2028_202879


namespace solution_set_correct_l2028_202868

def inequality_solution (x : ℝ) : Prop :=
  (x - 1) * (x - 2) * (x - 3)^2 > 0

theorem solution_set_correct : 
  ∀ x : ℝ, inequality_solution x ↔ (x < 1 ∨ (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 3) ∨ x > 3) := 
by sorry

end solution_set_correct_l2028_202868


namespace solution_set_f_l2028_202897

noncomputable def f (x : ℝ) : ℝ := sorry -- The differentiable function f

axiom f_deriv_lt (x : ℝ) : deriv f x < x -- Condition on the derivative of f
axiom f_at_2 : f 2 = 1 -- Given f(2) = 1

theorem solution_set_f : ∀ x : ℝ, f x < (1 / 2) * x^2 - 1 ↔ x > 2 :=
by sorry

end solution_set_f_l2028_202897


namespace proof_problem_l2028_202869

theorem proof_problem (a1 a2 a3 : ℕ) (h1 : a1 = a2 - 1) (h2 : a3 = a2 + 1) : 
  a2^3 ∣ (a1 * a2 * a3 + a2) :=
by sorry

end proof_problem_l2028_202869


namespace gcd_calculation_l2028_202831

theorem gcd_calculation :
  let a := 97^7 + 1
  let b := 97^7 + 97^3 + 1
  gcd a b = 1 := by
  sorry

end gcd_calculation_l2028_202831


namespace arithmetic_sequence_sum_l2028_202819

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
  (∀ n, S n = n * (a 1 + a n) / 2) →
  a 4 + a 8 = 4 →
  S 11 + a 6 = 24 :=
by
  intros a S h1 h2
  sorry

end arithmetic_sequence_sum_l2028_202819


namespace calculate_fraction_l2028_202889

theorem calculate_fraction :
  (5 * 6 - 4) / 8 = 13 / 4 := 
by
  sorry

end calculate_fraction_l2028_202889


namespace value_of_g_3_l2028_202809

def g (x : ℚ) : ℚ := (x^2 + x + 1) / (5*x - 3)

theorem value_of_g_3 : g 3 = 13 / 12 :=
by
  -- Proof goes here
  sorry

end value_of_g_3_l2028_202809


namespace union_A_B_l2028_202894

def set_A : Set ℝ := { x | 1 / x ≤ 0 }
def set_B : Set ℝ := { x | x^2 - 1 < 0 }

theorem union_A_B : set_A ∪ set_B = { x | x < 1 } :=
by
  sorry

end union_A_B_l2028_202894


namespace negation_of_p_l2028_202884

-- Define the proposition p
def p : Prop := ∃ x : ℝ, x + 2 ≤ 0

-- Define the negation of p
def not_p : Prop := ∀ x : ℝ, x + 2 > 0

-- State the theorem that the negation of p is not_p
theorem negation_of_p : ¬ p = not_p := by 
  sorry -- Proof not provided

end negation_of_p_l2028_202884


namespace find_radius_of_sphere_l2028_202865

def radius_of_sphere (width : ℝ) (depth : ℝ) (r : ℝ) : Prop :=
  (width / 2) ^ 2 + (r - depth) ^ 2 = r ^ 2

theorem find_radius_of_sphere (r : ℝ) : radius_of_sphere 30 10 r → r = 16.25 :=
by
  intros h1
  -- sorry is a placeholder for the actual proof
  sorry

end find_radius_of_sphere_l2028_202865


namespace original_percentage_of_acid_l2028_202843

theorem original_percentage_of_acid 
  (a w : ℝ) 
  (h1 : a + w = 6) 
  (h2 : a / (a + w + 2) = 15 / 100) 
  (h3 : (a + 2) / (a + w + 4) = 25 / 100) :
  (a / 6) * 100 = 20 :=
  sorry

end original_percentage_of_acid_l2028_202843


namespace true_prop_count_l2028_202899

-- Define the propositions
def original_prop (x : ℝ) : Prop := x > -3 → x > -6
def converse (x : ℝ) : Prop := x > -6 → x > -3
def inverse (x : ℝ) : Prop := x ≤ -3 → x ≤ -6
def contrapositive (x : ℝ) : Prop := x ≤ -6 → x ≤ -3

-- The statement to prove
theorem true_prop_count (x : ℝ) : 
  (original_prop x → true) ∧ (contrapositive x → true) ∧ ¬(converse x) ∧ ¬(inverse x) → 
  (count_true_propositions = 2) :=
sorry

end true_prop_count_l2028_202899


namespace total_books_equals_45_l2028_202878

-- Define the number of books bought in each category
def adventure_books : ℝ := 13.0
def mystery_books : ℝ := 17.0
def crime_books : ℝ := 15.0

-- Total number of books bought
def total_books := adventure_books + mystery_books + crime_books

-- The theorem we need to prove
theorem total_books_equals_45 : total_books = 45.0 := by
  -- placeholder for the proof
  sorry

end total_books_equals_45_l2028_202878


namespace find_xyz_value_l2028_202890

noncomputable def xyz_satisfying_conditions (x y z : ℝ) : Prop :=
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧
  (x + 1/y = 5) ∧
  (y + 1/z = 2) ∧
  (z + 1/x = 3)

theorem find_xyz_value (x y z : ℝ) (h : xyz_satisfying_conditions x y z) : x * y * z = 1 :=
by
  sorry

end find_xyz_value_l2028_202890


namespace arithmetic_sequence_sum_l2028_202847

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m: ℕ, a (n + 1) - a n = a (m + 1) - a m

-- Sum of the first n terms of a sequence
def sum_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- Specific statement we want to prove
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a)
  (h_S9 : sum_seq a 9 = 72) :
  a 2 + a 4 + a 9 = 24 :=
sorry

end arithmetic_sequence_sum_l2028_202847


namespace sum_is_18_l2028_202845

/-- Define the distinct non-zero digits, Hen, Xin, Chun, satisfying the given equation. -/
theorem sum_is_18 (Hen Xin Chun : ℕ) (h1 : Hen ≠ Xin) (h2 : Xin ≠ Chun) (h3 : Hen ≠ Chun)
  (h4 : 1 ≤ Hen ∧ Hen ≤ 9) (h5 : 1 ≤ Xin ∧ Xin ≤ 9) (h6 : 1 ≤ Chun ∧ Chun ≤ 9) :
  Hen + Xin + Chun = 18 :=
sorry

end sum_is_18_l2028_202845


namespace average_rainfall_correct_l2028_202857

-- Define the leap year condition and days in February
def leap_year_february_days : ℕ := 29

-- Define total hours in a day
def hours_in_day : ℕ := 24

-- Define total rainfall in February 2012 in inches
def total_rainfall : ℕ := 420

-- Define total hours in February 2012
def total_hours_february : ℕ := leap_year_february_days * hours_in_day

-- Define the average rainfall calculation
def average_rainfall_per_hour : ℚ :=
  total_rainfall / total_hours_february

-- Theorem to prove the average rainfall is 35/58 inches per hour
theorem average_rainfall_correct :
  average_rainfall_per_hour = 35 / 58 :=
by 
  -- Placeholder for proof
  sorry

end average_rainfall_correct_l2028_202857


namespace perimeter_of_field_l2028_202859

theorem perimeter_of_field (b l : ℕ) (h1 : l = b + 30) (h2 : b * l = 18000) : 2 * (l + b) = 540 := 
by 
  -- Proof goes here
sorry

end perimeter_of_field_l2028_202859


namespace prime_divisors_6270_l2028_202853

theorem prime_divisors_6270 : 
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 11 ∧ p5 = 19 ∧ 
  (p1 * p2 * p3 * p4 * p5 = 6270) ∧ 
  (Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ Nat.Prime p5) ∧ 
  (∀ q, Nat.Prime q ∧ q ∣ 6270 → (q = p1 ∨ q = p2 ∨ q = p3 ∨ q = p4 ∨ q = p5)) := 
by 
  sorry

end prime_divisors_6270_l2028_202853


namespace cheaper_store_price_difference_in_cents_l2028_202870

theorem cheaper_store_price_difference_in_cents :
  let list_price : ℝ := 59.99
  let discount_budget_buys := list_price * 0.15
  let discount_frugal_finds : ℝ := 20
  let sale_price_budget_buys := list_price - discount_budget_buys
  let sale_price_frugal_finds := list_price - discount_frugal_finds
  let difference_in_price := sale_price_budget_buys - sale_price_frugal_finds
  let difference_in_cents := difference_in_price * 100
  difference_in_cents = 1099.15 :=
by
  sorry

end cheaper_store_price_difference_in_cents_l2028_202870


namespace total_word_count_is_5000_l2028_202880

def introduction : ℕ := 450
def conclusion : ℕ := 3 * introduction
def body_sections : ℕ := 4 * 800

def total_word_count : ℕ := introduction + conclusion + body_sections

theorem total_word_count_is_5000 : total_word_count = 5000 := 
by
  -- Lean proof code will go here.
  sorry

end total_word_count_is_5000_l2028_202880


namespace log_sum_even_l2028_202875

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

-- Define the condition for maximum value at x = 1
def has_max_value_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y : ℝ, f y ≤ f x

-- Main theorem statement: Prove that lg x + lg y is an even function
theorem log_sum_even (A ω φ : ℝ) (hA : 0 < A) (hω : 0 < ω) 
  (hf_max : has_max_value_at (f A ω φ) 1) : 
  ∀ x y : ℝ, Real.log x + Real.log y = Real.log y + Real.log x := by
  sorry

end log_sum_even_l2028_202875


namespace support_percentage_correct_l2028_202825

-- Define the total number of government employees and the percentage supporting the project
def num_gov_employees : ℕ := 150
def perc_gov_support : ℝ := 0.70

-- Define the total number of citizens and the percentage supporting the project
def num_citizens : ℕ := 800
def perc_citizens_support : ℝ := 0.60

-- Calculate the number of supporters among government employees
def gov_supporters : ℝ := perc_gov_support * num_gov_employees

-- Calculate the number of supporters among citizens
def citizens_supporters : ℝ := perc_citizens_support * num_citizens

-- Calculate the total number of people surveyed and the total number of supporters
def total_surveyed : ℝ := num_gov_employees + num_citizens
def total_supporters : ℝ := gov_supporters + citizens_supporters

-- Define the expected correct answer percentage
def correct_percentage_supporters : ℝ := 61.58

-- Prove that the percentage of overall supporters is equal to the expected correct percentage 
theorem support_percentage_correct :
  (total_supporters / total_surveyed * 100) = correct_percentage_supporters :=
by
  sorry

end support_percentage_correct_l2028_202825


namespace value_of_m_l2028_202882

noncomputable def has_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

noncomputable def has_no_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c < 0

theorem value_of_m (m : ℝ) :
  (has_distinct_real_roots 1 m 1 ∧ has_no_real_roots 4 (4 * (m + 2)) 1) ↔ (-3 < m ∧ m < -2) :=
by
  sorry

end value_of_m_l2028_202882


namespace relationship_between_x_y_z_l2028_202828

theorem relationship_between_x_y_z (x y z : ℕ) (a b c d : ℝ)
  (h1 : x ≤ y ∧ y ≤ z)
  (h2 : (x:ℝ)^a = 70^d ∧ (y:ℝ)^b = 70^d ∧ (z:ℝ)^c = 70^d)
  (h3 : 1/a + 1/b + 1/c = 1/d) :
  x + y = z := 
sorry

end relationship_between_x_y_z_l2028_202828


namespace number_of_solutions_l2028_202891

theorem number_of_solutions (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 5) :
  (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = x - 4 → x = -2 :=
sorry

end number_of_solutions_l2028_202891


namespace problem_equation_has_solution_l2028_202804

noncomputable def x (real_number : ℚ) : ℚ := 210 / 23

theorem problem_equation_has_solution (x_value : ℚ) : 
  (3 / 7) + (7 / x_value) = (10 / x_value) + (1 / 10) → 
  x_value = 210 / 23 :=
by
  intro h
  sorry

end problem_equation_has_solution_l2028_202804


namespace similar_triangle_legs_l2028_202815

theorem similar_triangle_legs {y : ℝ} 
  (h₁ : 12 / y = 9 / 6) : y = 8 :=
sorry

end similar_triangle_legs_l2028_202815


namespace statue_original_cost_l2028_202851

theorem statue_original_cost (selling_price : ℝ) (profit_percent : ℝ) (original_cost : ℝ) 
  (h1 : selling_price = 620) (h2 : profit_percent = 25) : 
  original_cost = 496 :=
by
  have h3 : profit_percent / 100 + 1 = 1.25 := by sorry
  have h4 : 1.25 * original_cost = selling_price := by sorry
  have h5 : original_cost = 620 / 1.25 := by sorry
  have h6 : 620 / 1.25 = 496 := by sorry
  exact sorry

end statue_original_cost_l2028_202851


namespace values_of_k_real_equal_roots_l2028_202893

theorem values_of_k_real_equal_roots (k : ℝ) : 
  (∃ k, (3 - 2 * k)^2 - 4 * 3 * 12 = 0 ∧ (k = -9 / 2 ∨ k = 15 / 2)) :=
by
  sorry

end values_of_k_real_equal_roots_l2028_202893


namespace solve_for_a_l2028_202866

def E (a b c : ℝ) : ℝ := a * b^2 + b * c + c

theorem solve_for_a : (E (-5/8) 3 2 = E (-5/8) 5 3) :=
by
  sorry

end solve_for_a_l2028_202866


namespace find_negative_a_l2028_202864

noncomputable def g (x : ℝ) : ℝ :=
if x ≤ 0 then -x else 3 * x - 22

theorem find_negative_a (a : ℝ) (ha : a < 0) :
  g (g (g 7)) = g (g (g a)) ↔ a = -23 / 3 :=
by
  sorry

end find_negative_a_l2028_202864


namespace brendan_cuts_yards_l2028_202898

theorem brendan_cuts_yards (x : ℝ) (h : 7 * 1.5 * x = 84) : x = 8 :=
sorry

end brendan_cuts_yards_l2028_202898


namespace train_distance_problem_l2028_202816

theorem train_distance_problem
  (Vx : ℝ) (Vy : ℝ) (t : ℝ) (distanceX : ℝ) 
  (h1 : Vx = 32) 
  (h2 : Vy = 160 / 3) 
  (h3 : 32 * t + (160 / 3) * t = 160) :
  distanceX = Vx * t → distanceX = 60 :=
by {
  sorry
}

end train_distance_problem_l2028_202816


namespace find_marks_in_biology_l2028_202805

-- Definitions based on conditions in a)
def marks_english : ℕ := 76
def marks_math : ℕ := 60
def marks_physics : ℕ := 72
def marks_chemistry : ℕ := 65
def num_subjects : ℕ := 5
def average_marks : ℕ := 71

-- The theorem that needs to be proved
theorem find_marks_in_biology : 
  let total_marks := marks_english + marks_math + marks_physics + marks_chemistry 
  let total_marks_all := average_marks * num_subjects
  let marks_biology := total_marks_all - total_marks
  marks_biology = 82 := 
by
  sorry

end find_marks_in_biology_l2028_202805


namespace diane_initial_amount_l2028_202861

theorem diane_initial_amount
  (X : ℝ)        -- the amount Diane started with
  (won_amount : ℝ := 65)
  (total_loss : ℝ := 215)
  (owing_friends : ℝ := 50)
  (final_amount := X + won_amount - total_loss - owing_friends) :
  X = 100 := 
by 
  sorry

end diane_initial_amount_l2028_202861


namespace rotate_A_180_about_B_l2028_202863

-- Define the points A, B, and C
def A : ℝ × ℝ := (-4, 1)
def B : ℝ × ℝ := (-1, 4)
def C : ℝ × ℝ := (-1, 1)

-- Define the 180 degrees rotation about B
def rotate_180_about (p q : ℝ × ℝ) : ℝ × ℝ :=
  let translated_p := (p.1 - q.1, p.2 - q.2) 
  let rotated_p := (-translated_p.1, -translated_p.2)
  (rotated_p.1 + q.1, rotated_p.2 + q.2)

-- Prove the image of point A after a 180 degrees rotation about point B
theorem rotate_A_180_about_B : rotate_180_about A B = (2, 7) :=
by
  sorry

end rotate_A_180_about_B_l2028_202863


namespace cube_root_approx_l2028_202848

open Classical

theorem cube_root_approx (n : ℤ) (x : ℝ) (h₁ : 2^n = x^3) (h₂ : abs (x - 50) <  1) : n = 17 := by
  sorry

end cube_root_approx_l2028_202848


namespace calculate_total_money_l2028_202841

theorem calculate_total_money (n100 n50 n10 : ℕ) 
  (h1 : n100 = 2) (h2 : n50 = 5) (h3 : n10 = 10) : 
  (n100 * 100 + n50 * 50 + n10 * 10 = 550) :=
by
  sorry

end calculate_total_money_l2028_202841


namespace total_bill_l2028_202810

/-
Ten friends dined at a restaurant and split the bill equally.
One friend, Chris, forgets his money.
Each of the remaining nine friends agreed to pay an extra $3 to cover Chris's share.
How much was the total bill?

Correct answer: 270
-/

theorem total_bill (t : ℕ) (h1 : ∀ x, t = 10 * x) (h2 : ∀ x, t = 9 * (x + 3)) : t = 270 := by
  sorry

end total_bill_l2028_202810


namespace larger_integer_l2028_202824

-- Definitions based on the given conditions
def two_integers (x : ℤ) (y : ℤ) :=
  y = 4 * x ∧ (x + 12) * 2 = y

-- Statement of the problem
theorem larger_integer (x : ℤ) (y : ℤ) (h : two_integers x y) : y = 48 :=
by sorry

end larger_integer_l2028_202824


namespace budget_allocation_degrees_l2028_202876

theorem budget_allocation_degrees :
  let microphotonics := 12.3
  let home_electronics := 17.8
  let food_additives := 9.4
  let gmo := 21.7
  let industrial_lubricants := 6.2
  let artificial_intelligence := 4.1
  let nanotechnology := 5.3
  let basic_astrophysics := 100 - (microphotonics + home_electronics + food_additives + gmo + industrial_lubricants + artificial_intelligence + nanotechnology)
  (basic_astrophysics * 3.6) + (artificial_intelligence * 3.6) + (nanotechnology * 3.6) = 117.36 :=
by
  sorry

end budget_allocation_degrees_l2028_202876


namespace distance_sum_is_ten_l2028_202808

noncomputable def angle_sum_distance (C A B : ℝ) (d : ℝ) (k : ℝ) : ℝ := 
  let h_A : ℝ := sorry -- replace with expression for h_A based on conditions
  let h_B : ℝ := sorry -- replace with expression for h_B based on conditions
  h_A + h_B

theorem distance_sum_is_ten 
  (A B C : ℝ) 
  (h : ℝ) 
  (k : ℝ) 
  (h_pos : h = 4) 
  (ratio_condition : h_A = 4 * h_B)
  : angle_sum_distance C A B h k = 10 := 
  sorry

end distance_sum_is_ten_l2028_202808


namespace team_total_score_l2028_202826

theorem team_total_score (Connor_score Amy_score Jason_score : ℕ)
  (h1 : Connor_score = 2)
  (h2 : Amy_score = Connor_score + 4)
  (h3 : Jason_score = 2 * Amy_score) :
  Connor_score + Amy_score + Jason_score = 20 :=
by
  sorry

end team_total_score_l2028_202826


namespace simplify_expression_l2028_202858

theorem simplify_expression : 
  2 ^ (-1: ℤ) + Real.sqrt 16 - (3 - Real.sqrt 3) ^ 0 + |Real.sqrt 2 - 1 / 2| = 3 + Real.sqrt 2 := by
  sorry

end simplify_expression_l2028_202858


namespace exponentiation_identity_l2028_202892

theorem exponentiation_identity :
  (5^4)^2 = 390625 :=
  by sorry

end exponentiation_identity_l2028_202892


namespace range_of_m_l2028_202844

theorem range_of_m {m : ℝ} (h1 : m^2 - 1 < 0) (h2 : m > 0) : 0 < m ∧ m < 1 :=
sorry

end range_of_m_l2028_202844


namespace star_contains_2011_l2028_202885

theorem star_contains_2011 :
  ∃ (n : ℕ), n = 183 ∧ 
  (∃ (seq : List ℕ), seq = List.range' (2003) 11 ∧ 2011 ∈ seq) :=
by
  sorry

end star_contains_2011_l2028_202885


namespace circle_equation_l2028_202838

-- Definitions for the given conditions
def line1 (x y : ℝ) : Prop := x + y + 2 = 0
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def line2 (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def is_solution (x y : ℝ) : Prop := x^2 + y^2 - 6 * x - 6 * y - 16 = 0

-- Problem statement in Lean
theorem circle_equation : ∃ x y : ℝ, 
  (line1 x y ∧ circle1 x y ∧ line2 (x / 2) (x / 2)) → is_solution x y :=
sorry

end circle_equation_l2028_202838


namespace ratio_new_circumference_to_original_diameter_l2028_202874

-- Define the problem conditions
variables (r k : ℝ) (hk : k > 0)

-- Define the Lean theorem to express the proof problem
theorem ratio_new_circumference_to_original_diameter (r k : ℝ) (hk : k > 0) :
  (π * (1 + k / r)) = (2 * π * (r + k)) / (2 * r) :=
by {
  -- Placeholder proof, to be filled in
  sorry
}

end ratio_new_circumference_to_original_diameter_l2028_202874


namespace file_size_l2028_202895

-- Definitions based on conditions
def upload_speed : ℕ := 8 -- megabytes per minute
def upload_time : ℕ := 20 -- minutes

-- Goal to prove
theorem file_size:
  (upload_speed * upload_time = 160) :=
by sorry

end file_size_l2028_202895


namespace sequence_geometric_and_general_term_sum_of_sequence_l2028_202850

theorem sequence_geometric_and_general_term (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ)
  (h1 : ∀ k : ℕ, S k = 2 * a k - k) : 
  (a 0 = 1) ∧ 
  (∀ k : ℕ, a (k + 1) = 2 * a k + 1) ∧ 
  (∀ k : ℕ, a k = 2^k - 1) :=
sorry

theorem sum_of_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ)
  (h1 : ∀ k : ℕ, a k = 2^k - 1)
  (h2 : ∀ k : ℕ, b k = 1 / a (k+1) + 1 / (a k * a (k+1))) :
  T n = 1 - 1 / (2^(n+1) - 1) :=
sorry

end sequence_geometric_and_general_term_sum_of_sequence_l2028_202850


namespace max_omega_l2028_202871

noncomputable def f (ω φ x : ℝ) := 2 * Real.sin (ω * x + φ)

theorem max_omega (ω φ : ℝ) (k k' : ℤ) (hω_pos : ω > 0) (hφ1 : 0 < φ)
  (hφ2 : φ < Real.pi / 2) (h1 : f ω φ (-Real.pi / 4) = 0)
  (h2 : ∀ x, f ω φ (Real.pi / 4 - x) = f ω φ (Real.pi / 4 + x))
  (h3 : ∀ x, x ∈ Set.Ioo (Real.pi / 18) (2 * Real.pi / 9) →
    Monotone (f ω φ)) :
  ω = 5 :=
sorry

end max_omega_l2028_202871


namespace profit_inequality_solution_l2028_202873

theorem profit_inequality_solution (x : ℝ) (h₁ : 1 ≤ x) (h₂ : x ≤ 10) :
  100 * 2 * (5 * x + 1 - 3 / x) ≥ 3000 ↔ 3 ≤ x ∧ x ≤ 10 :=
by
  sorry

end profit_inequality_solution_l2028_202873


namespace theodore_total_monthly_earning_l2028_202846

def total_earnings (stone_statues: Nat) (wooden_statues: Nat) (cost_stone: Nat) (cost_wood: Nat) (tax_rate: Rat) : Rat :=
  let pre_tax_earnings := stone_statues * cost_stone + wooden_statues * cost_wood
  let tax := tax_rate * pre_tax_earnings
  pre_tax_earnings - tax

theorem theodore_total_monthly_earning : total_earnings 10 20 20 5 0.10 = 270 :=
by
  sorry

end theodore_total_monthly_earning_l2028_202846


namespace age_difference_is_100_l2028_202801

-- Definition of the ages
variables {X Y Z : ℕ}

-- Conditions from the problem statement
axiom age_condition1 : X + Y > Y + Z
axiom age_condition2 : Z = X - 100

-- Proof to show the difference is 100 years
theorem age_difference_is_100 : (X + Y) - (Y + Z) = 100 :=
by sorry

end age_difference_is_100_l2028_202801


namespace bicycle_saves_time_l2028_202883

-- Define the conditions
def time_to_walk : ℕ := 98
def time_saved_by_bicycle : ℕ := 34

-- Prove the question equals the answer
theorem bicycle_saves_time :
  time_saved_by_bicycle = 34 := 
by
  sorry

end bicycle_saves_time_l2028_202883


namespace apples_from_C_to_D_l2028_202800

theorem apples_from_C_to_D (n m : ℕ)
  (h_tree_ratio : ∀ (P V : ℕ), P = 2 * V)
  (h_apple_ratio : ∀ (P V : ℕ), P = 7 * V)
  (trees_CD_Petya trees_CD_Vasya : ℕ)
  (h_trees_CD : trees_CD_Petya = 2 * trees_CD_Vasya)
  (apples_CD_Petya apples_CD_Vasya: ℕ)
  (h_apples_CD : apples_CD_Petya = (m / 4) ∧ apples_CD_Vasya = (3 * m / 4)) : 
  apples_CD_Vasya = 3 * apples_CD_Petya := by
  sorry

end apples_from_C_to_D_l2028_202800


namespace axis_of_symmetry_range_of_t_l2028_202822

section
variables (a b m n p t : ℝ)

-- Assume the given conditions
def parabola (x : ℝ) : ℝ := a * x ^ 2 + b * x

-- Part (1): Find the axis of symmetry
theorem axis_of_symmetry (h_a_pos : a > 0) 
    (hM : parabola a b 2 = m) 
    (hN : parabola a b 4 = n) 
    (hmn : m = n) : 
    -b / (2 * a) = 3 := 
  sorry

-- Part (2): Find the range of values for t
theorem range_of_t (h_a_pos : a > 0) 
    (hP : parabola a b (-1) = p)
    (axis : -b / (2 * a) = t) 
    (hmn_neg : m * n < 0) 
    (hmpn : m < p ∧ p < n) :
    1 < t ∧ t < 3 / 2 := 
  sorry
end

end axis_of_symmetry_range_of_t_l2028_202822


namespace number_of_teams_l2028_202823

theorem number_of_teams (n : ℕ) (h : n * (n - 1) / 2 = 36) : n = 9 :=
sorry

end number_of_teams_l2028_202823


namespace temperature_difference_l2028_202860

/-- The average temperature at the top of Mount Tai. -/
def T_top : ℝ := -9

/-- The average temperature at the foot of Mount Tai. -/
def T_foot : ℝ := -1

/-- The temperature difference between the average temperature at the foot and the top of Mount Tai is 8 degrees Celsius. -/
theorem temperature_difference : T_foot - T_top = 8 := by
  sorry

end temperature_difference_l2028_202860


namespace cost_of_fencing_each_side_l2028_202888

theorem cost_of_fencing_each_side (total_cost : ℕ) (x : ℕ) (h : total_cost = 276) (hx : 4 * x = total_cost) : x = 69 :=
by {
  sorry
}

end cost_of_fencing_each_side_l2028_202888


namespace number_of_truthful_dwarfs_l2028_202830

/-- 
Each of the 10 dwarfs either always tells the truth or always lies. 
Each dwarf likes exactly one type of ice cream: vanilla, chocolate, or fruit.
When asked, every dwarf raised their hand for liking vanilla ice cream.
When asked, 5 dwarfs raised their hand for liking chocolate ice cream.
When asked, only 1 dwarf raised their hand for liking fruit ice cream.
Prove that the number of truthful dwarfs is 4.
-/
theorem number_of_truthful_dwarfs (T L : ℕ) 
  (h1 : T + L = 10) 
  (h2 : T + 2 * L = 16) : 
  T = 4 := 
by
  -- Proof omitted
  sorry

end number_of_truthful_dwarfs_l2028_202830


namespace inequality_solution_set_l2028_202862

theorem inequality_solution_set (x : ℝ) : (x-1)/(x+2) > 1 → x < -2 := sorry

end inequality_solution_set_l2028_202862


namespace find_t_l2028_202836

theorem find_t :
  ∃ t : ℕ, 10 ≤ t ∧ t < 100 ∧ 13 * t % 100 = 52 ∧ t = 44 :=
by
  sorry

end find_t_l2028_202836


namespace find_p_l2028_202854

variable (A B C D p q u v w : ℝ)
variable (hu : u + v + w = -B / A)
variable (huv : u * v + v * w + w * u = C / A)
variable (huvw : u * v * w = -D / A)
variable (hpq : u^2 + v^2 = -p)
variable (hq : u^2 * v^2 = q)

theorem find_p (A B C D : ℝ) (u v w : ℝ) 
  (H1 : u + v + w = -B / A)
  (H2 : u * v + v * w + w * u = C / A)
  (H3 : u * v * w = -D / A)
  (H4 : v = -u - w)
  : p = (B^2 - 2 * C) / A^2 :=
by sorry

end find_p_l2028_202854


namespace mike_drive_average_rate_l2028_202849

open Real

variables (total_distance first_half_distance second_half_distance first_half_speed second_half_speed first_half_time second_half_time total_time avg_rate j : ℝ)

theorem mike_drive_average_rate :
  total_distance = 640 ∧
  first_half_distance = total_distance / 2 ∧
  second_half_distance = total_distance / 2 ∧
  first_half_speed = 80 ∧
  first_half_distance / first_half_speed = first_half_time ∧
  second_half_time = 3 * first_half_time ∧
  second_half_distance / second_half_time = second_half_speed ∧
  total_time = first_half_time + second_half_time ∧
  avg_rate = total_distance / total_time →
  j = 40 :=
by
  intro h
  sorry

end mike_drive_average_rate_l2028_202849


namespace part_one_part_two_l2028_202834

noncomputable def f (a x : ℝ) : ℝ :=
  |x + (1 / a)| + |x - a + 1|

theorem part_one (a : ℝ) (h : a > 0) (x : ℝ) : f a x ≥ 1 :=
sorry

theorem part_two (a : ℝ) (h : a > 0) : f a 3 < 11 / 2 → 2 < a ∧ a < (13 + 3 * Real.sqrt 17) / 4 :=
sorry

end part_one_part_two_l2028_202834


namespace veranda_area_correct_l2028_202842

-- Define the dimensions of the room.
def room_length : ℕ := 20
def room_width : ℕ := 12

-- Define the width of the veranda.
def veranda_width : ℕ := 2

-- Calculate the total dimensions with the veranda.
def total_length : ℕ := room_length + 2 * veranda_width
def total_width : ℕ := room_width + 2 * veranda_width

-- Calculate the area of the room and the total area including the veranda.
def room_area : ℕ := room_length * room_width
def total_area : ℕ := total_length * total_width

-- Prove that the area of the veranda is 144 m².
theorem veranda_area_correct : total_area - room_area = 144 := by
  sorry

end veranda_area_correct_l2028_202842


namespace area_of_square_on_PS_l2028_202817

-- Given parameters as conditions in the form of hypotheses
variables (PQ QR RS PS PR : ℝ)

-- Hypotheses based on problem conditions
def hypothesis1 : PQ^2 = 25 := sorry
def hypothesis2 : QR^2 = 49 := sorry
def hypothesis3 : RS^2 = 64 := sorry
def hypothesis4 : PR^2 = PQ^2 + QR^2 := sorry
def hypothesis5 : PS^2 = PR^2 - RS^2 := sorry

-- The main theorem we need to prove
theorem area_of_square_on_PS :
  PS^2 = 10 := 
by {
  sorry
}

end area_of_square_on_PS_l2028_202817


namespace tan_7pi_over_4_eq_neg1_l2028_202821

theorem tan_7pi_over_4_eq_neg1 : Real.tan (7 * Real.pi / 4) = -1 :=
  sorry

end tan_7pi_over_4_eq_neg1_l2028_202821


namespace isosceles_triangle_perimeter_l2028_202867

theorem isosceles_triangle_perimeter (a b : ℕ) (c : ℕ) 
  (h1 : a = 5) (h2 : b = 5) (h3 : c = 2) 
  (isosceles : a = b ∨ b = c ∨ c = a) 
  (triangle_inequality1 : a + b > c)
  (triangle_inequality2 : a + c > b)
  (triangle_inequality3 : b + c > a) : 
  a + b + c = 12 :=
  sorry

end isosceles_triangle_perimeter_l2028_202867


namespace solve_quadratic_eq_l2028_202852

theorem solve_quadratic_eq (x y z w d X Y Z W : ℤ) 
    (h1 : w % 2 = z % 2) 
    (h2 : x = 2 * d * (X * Z - Y * W))
    (h3 : y = 2 * d * (X * W + Y * Z))
    (h4 : z = d * (X^2 + Y^2 - Z^2 - W^2))
    (h5 : w = d * (X^2 + Y^2 + Z^2 + W^2)) :
    x^2 + y^2 + z^2 = w^2 :=
sorry

end solve_quadratic_eq_l2028_202852


namespace trader_profit_l2028_202840

theorem trader_profit (donation goal extra profit : ℝ) (half_profit : ℝ) 
  (H1 : donation = 310) (H2 : goal = 610) (H3 : extra = 180)
  (H4 : half_profit = profit / 2) 
  (H5 : half_profit + donation = goal + extra) : 
  profit = 960 := 
by
  sorry

end trader_profit_l2028_202840


namespace number_of_throwers_l2028_202814

theorem number_of_throwers (T N : ℕ) :
  (T + N = 61) ∧ ((2 * N) / 3 = 53 - T) → T = 37 :=
by 
  sorry

end number_of_throwers_l2028_202814


namespace pipes_fill_tank_in_8_hours_l2028_202820

theorem pipes_fill_tank_in_8_hours (A B C : ℝ) (hA : A = 1 / 56) (hB : B = 2 * A) (hC : C = 2 * B) :
  1 / (A + B + C) = 8 :=
by
  sorry

end pipes_fill_tank_in_8_hours_l2028_202820


namespace smallest_positive_period_minimum_value_of_f_center_of_symmetry_interval_of_increasing_l2028_202811

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x - Real.pi / 6)

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

theorem minimum_value_of_f :
  ∃ x, f x = -3 :=
sorry

theorem center_of_symmetry (k : ℤ) :
  ∃ p, (∀ x, f (p + x) = f (p - x)) ∧ p = (Real.pi / 12) + (k * Real.pi / 2) :=
sorry

theorem interval_of_increasing (k : ℤ) :
  ∃ a b, a = -(Real.pi / 6) + k * Real.pi ∧ b = (Real.pi / 3) + k * Real.pi ∧
  ∀ x, (a <= x ∧ x <= b) → StrictMonoOn f (Set.Icc a b) :=
sorry

end smallest_positive_period_minimum_value_of_f_center_of_symmetry_interval_of_increasing_l2028_202811


namespace max_months_to_build_l2028_202832

theorem max_months_to_build (a b c x : ℝ) (h1 : 1/a + 1/b = 1/6)
                            (h2 : 1/a + 1/c = 1/5)
                            (h3 : 1/c + 1/b = 1/4)
                            (h4 : (1/a + 1/b + 1/c) * x = 1) :
                            x = 4 :=
sorry

end max_months_to_build_l2028_202832
