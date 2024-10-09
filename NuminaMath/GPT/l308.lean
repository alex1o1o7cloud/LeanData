import Mathlib

namespace algebraic_expression_value_l308_30875

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 2 * x - 1 = 0) : x^3 - x^2 - 3 * x + 2 = 3 := 
by
  sorry

end algebraic_expression_value_l308_30875


namespace apples_for_pies_l308_30837

-- Define the conditions
def apples_per_pie : ℝ := 4.0
def number_of_pies : ℝ := 126.0

-- Define the expected answer
def number_of_apples : ℝ := number_of_pies * apples_per_pie

-- State the theorem to prove the question == answer given the conditions
theorem apples_for_pies : number_of_apples = 504 :=
by
  -- This is where the proof would go. Currently skipped.
  sorry

end apples_for_pies_l308_30837


namespace simplified_t_l308_30847

noncomputable def cuberoot (x : ℝ) : ℝ := x^(1/3)

theorem simplified_t (t : ℝ) (h : t = 1 / (3 - cuberoot 3)) : t = (3 + cuberoot 3) / 6 :=
by
  sorry

end simplified_t_l308_30847


namespace towel_percentage_decrease_l308_30835

theorem towel_percentage_decrease (L B : ℝ) (hL: L > 0) (hB: B > 0) :
  let OriginalArea := L * B
  let NewLength := 0.8 * L
  let NewBreadth := 0.8 * B
  let NewArea := NewLength * NewBreadth
  let PercentageDecrease := ((OriginalArea - NewArea) / OriginalArea) * 100
  PercentageDecrease = 36 :=
by
  sorry

end towel_percentage_decrease_l308_30835


namespace find_top_row_number_l308_30801

theorem find_top_row_number (x z : ℕ) (h1 : 8 = x * 2) (h2 : 16 = 2 * z)
  (h3 : 56 = 8 * 7) (h4 : 112 = 16 * 7) : x = 4 :=
by sorry

end find_top_row_number_l308_30801


namespace michael_number_l308_30823

theorem michael_number (m : ℕ) (h1 : m % 75 = 0) (h2 : m % 40 = 0) (h3 : 1000 < m) (h4 : m < 3000) :
  m = 1800 ∨ m = 2400 ∨ m = 3000 :=
sorry

end michael_number_l308_30823


namespace smallest_integer_greater_than_100_with_gcd_24_eq_4_l308_30871

theorem smallest_integer_greater_than_100_with_gcd_24_eq_4 :
  ∃ x : ℤ, x > 100 ∧ x % 24 = 4 ∧ (∀ y : ℤ, y > 100 ∧ y % 24 = 4 → x ≤ y) :=
sorry

end smallest_integer_greater_than_100_with_gcd_24_eq_4_l308_30871


namespace min_value_a1_l308_30872

noncomputable def is_geometric_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, ∀ n : ℕ, seq (n + 1) = r * seq n

theorem min_value_a1 (a1 a2 : ℕ) (seq : ℕ → ℕ)
  (h1 : is_geometric_sequence seq)
  (h2 : ∀ n : ℕ, seq n > 0)
  (h3 : seq 20 + seq 21 = 20^21) :
  ∃ a b : ℕ, a1 = 2^a * 5^b ∧ a + b = 24 :=
sorry

end min_value_a1_l308_30872


namespace simplify_radical_expression_l308_30831

noncomputable def simpl_radical_form (q : ℝ) : ℝ :=
  Real.sqrt (15 * q) * Real.sqrt (3 * q^2) * Real.sqrt (2 * q^3)

theorem simplify_radical_expression (q : ℝ) :
  simpl_radical_form q = 3 * q^3 * Real.sqrt 10 :=
by
  sorry

end simplify_radical_expression_l308_30831


namespace not_always_true_inequality_l308_30865

theorem not_always_true_inequality (x : ℝ) (hx : x > 0) : 2^x ≤ x^2 := sorry

end not_always_true_inequality_l308_30865


namespace basketball_holes_l308_30832

theorem basketball_holes (soccer_balls total_basketballs soccer_balls_with_hole balls_without_holes basketballs_without_holes: ℕ) 
  (h1: soccer_balls = 40) 
  (h2: total_basketballs = 15)
  (h3: soccer_balls_with_hole = 30) 
  (h4: balls_without_holes = 18) 
  (h5: basketballs_without_holes = 8) 
  : (total_basketballs - basketballs_without_holes = 7) := 
by
  sorry

end basketball_holes_l308_30832


namespace find_w_l308_30842

noncomputable def line_p(t : ℝ) : (ℝ × ℝ) := (2 + 3 * t, 5 + 2 * t)
noncomputable def line_q(u : ℝ) : (ℝ × ℝ) := (-3 + 3 * u, 7 + 2 * u)

def vector_DC(t u : ℝ) : ℝ × ℝ := ((2 + 3 * t) - (-3 + 3 * u), (5 + 2 * t) - (7 + 2 * u))

def w_condition (w1 w2 : ℝ) : Prop := w1 + w2 = 3

theorem find_w (t u : ℝ) :
  ∃ w1 w2 : ℝ, 
    w_condition w1 w2 ∧ 
    (∃ k : ℝ, 
      sorry -- This is a placeholder for the projection calculation
    )
    :=
  sorry -- This is a placeholder for the final proof

end find_w_l308_30842


namespace grocery_cost_l308_30899

def rent : ℕ := 1100
def utilities : ℕ := 114
def roommate_payment : ℕ := 757

theorem grocery_cost (total_payment : ℕ) (half_rent_utilities : ℕ) (half_groceries : ℕ) (total_groceries : ℕ) :
  total_payment = 757 →
  half_rent_utilities = (rent + utilities) / 2 →
  half_groceries = total_payment - half_rent_utilities →
  total_groceries = half_groceries * 2 →
  total_groceries = 300 :=
by
  intros
  sorry

end grocery_cost_l308_30899


namespace rachel_picked_apples_l308_30897

-- Define relevant variables based on problem conditions
variable (trees : ℕ) (apples_per_tree : ℕ) (remaining_apples : ℕ)
variable (total_apples_picked : ℕ)

-- Assume the given conditions
axiom num_trees : trees = 4
axiom apples_each_tree : apples_per_tree = 7
axiom apples_left : remaining_apples = 29

-- Define the number of apples picked
def total_apples_picked_def := trees * apples_per_tree

-- State the theorem to prove the total apples picked
theorem rachel_picked_apples :
  total_apples_picked_def trees apples_per_tree = 28 :=
by
  -- Proof omitted
  sorry

end rachel_picked_apples_l308_30897


namespace number_of_mixed_vegetable_plates_l308_30882

def cost_of_chapati := 6
def cost_of_rice := 45
def cost_of_mixed_vegetable := 70
def chapatis_ordered := 16
def rice_ordered := 5
def ice_cream_cups := 6 -- though not used, included for completeness
def total_amount_paid := 1111

def total_cost_of_known_items := (chapatis_ordered * cost_of_chapati) + (rice_ordered * cost_of_rice)
def amount_spent_on_mixed_vegetable := total_amount_paid - total_cost_of_known_items

theorem number_of_mixed_vegetable_plates : 
  amount_spent_on_mixed_vegetable / cost_of_mixed_vegetable = 11 := 
by sorry

end number_of_mixed_vegetable_plates_l308_30882


namespace gcd_problem_l308_30880

theorem gcd_problem (x : ℤ) (h : ∃ k, x = 2 * 2027 * k) :
  Int.gcd (3 * x ^ 2 + 47 * x + 101) (x + 23) = 1 :=
sorry

end gcd_problem_l308_30880


namespace correct_calculated_value_l308_30889

theorem correct_calculated_value (N : ℕ) (h : N ≠ 0) :
  N * 16 = 2048 * (N / 128) := by 
  sorry

end correct_calculated_value_l308_30889


namespace trigonometric_identity_l308_30861

theorem trigonometric_identity
  (x : ℝ)
  (h_tan : Real.tan x = -1/2) :
  Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x - 1 = -2 :=
sorry

end trigonometric_identity_l308_30861


namespace total_cost_proof_l308_30843

def uber_cost : ℤ := 22
def lyft_cost : ℤ := uber_cost - 3
def taxi_cost : ℤ := lyft_cost - 4
def tip : ℤ := (taxi_cost * 20) / 100
def total_cost : ℤ := taxi_cost + tip

theorem total_cost_proof :
  total_cost = 18 :=
by
  sorry

end total_cost_proof_l308_30843


namespace weight_of_lightest_weight_l308_30807

theorem weight_of_lightest_weight (x : ℕ) (y : ℕ) (h1 : 0 < y ∧ y < 9)
  (h2 : (10 : ℕ) * x + 45 - (x + y) = 2022) : x = 220 := by
  sorry

end weight_of_lightest_weight_l308_30807


namespace sum_m_n_is_55_l308_30898

theorem sum_m_n_is_55 (a b c : ℝ) (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1)
  (h1 : 5 / a = b + c) (h2 : 10 / b = c + a) (h3 : 13 / c = a + b) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : (a + b + c) = m / n) : m + n = 55 :=
  sorry

end sum_m_n_is_55_l308_30898


namespace problem_statement_l308_30821

variables (a b : ℝ)

-- Conditions: The lines \(x = \frac{1}{3}y + a\) and \(y = \frac{1}{3}x + b\) intersect at \((3, 1)\).
def lines_intersect_at (a b : ℝ) : Prop :=
  (3 = (1/3) * 1 + a) ∧ (1 = (1/3) * 3 + b)

-- Goal: Prove that \(a + b = \frac{8}{3}\)
theorem problem_statement (H : lines_intersect_at a b) : a + b = 8 / 3 :=
by
  sorry

end problem_statement_l308_30821


namespace alice_sold_20_pears_l308_30812

variables (S P C : ℝ)

theorem alice_sold_20_pears (h1 : C = 1.20 * P)
  (h2 : P = 0.50 * S)
  (h3 : S + P + C = 42) : S = 20 :=
by {
  -- mark the proof as incomplete with sorry
  sorry
}

end alice_sold_20_pears_l308_30812


namespace pyramid_volume_l308_30887

noncomputable def volume_of_pyramid (a b c : ℝ) : ℝ :=
  (1 / 3) * a * b * c * Real.sqrt 2

theorem pyramid_volume (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (A1 : ∃ x y, 1 / 2 * x * y = a^2) 
  (A2 : ∃ y z, 1 / 2 * y * z = b^2) 
  (A3 : ∃ x z, 1 / 2 * x * z = c^2)
  (h_perpendicular : True) :
  volume_of_pyramid a b c = (1 / 3) * a * b * c * Real.sqrt 2 :=
sorry

end pyramid_volume_l308_30887


namespace inequality_solution_l308_30870

theorem inequality_solution (x : ℝ) :
  (∃ x, 2 < x ∧ x < 3) ↔ ∃ x, (x-2)*(x-3)/(x^2 + 1) < 0 := by
  sorry

end inequality_solution_l308_30870


namespace math_problem_l308_30863

theorem math_problem (a b : ℝ) (h1 : 4 + a = 5 - b) (h2 : 5 + b = 8 + a) : 4 - a = 3 :=
by
  sorry

end math_problem_l308_30863


namespace minimum_3x_4y_l308_30802

theorem minimum_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 :=
by
  sorry

end minimum_3x_4y_l308_30802


namespace range_of_m_l308_30834

-- Definitions of propositions p and q
def p (m : ℝ) : Prop := 
  (2 * m - 3)^2 - 4 > 0

def q (m : ℝ) : Prop := 
  2 * m > 3

-- Theorem statement
theorem range_of_m (m : ℝ) : ¬ (p m ∧ q m) ∧ (p m ∨ q m) ↔ (m < 1 / 2 ∨ 3 / 2 < m ∧ m ≤ 5 / 2) :=
  sorry

end range_of_m_l308_30834


namespace length_of_second_offset_l308_30856

theorem length_of_second_offset 
  (d : ℝ) (offset1 : ℝ) (area : ℝ) (offset2 : ℝ) 
  (h1 : d = 40)
  (h2 : offset1 = 9)
  (h3 : area = 300) :
  offset2 = 6 :=
by
  sorry

end length_of_second_offset_l308_30856


namespace solve_equation_l308_30874

theorem solve_equation :
  ∀ x : ℝ, 18 / (x^2 - 9) - 3 / (x - 3) = 2 ↔ (x = 4.5 ∨ x = -3) :=
by
  sorry

end solve_equation_l308_30874


namespace calculate_total_people_l308_30873

-- Definitions given in the problem
def cost_per_adult_meal := 3
def num_kids := 7
def total_cost := 15

-- The target property to prove
theorem calculate_total_people : 
  (total_cost / cost_per_adult_meal) + num_kids = 12 := 
by 
  sorry

end calculate_total_people_l308_30873


namespace denomination_calculation_l308_30810

variables (total_money rs_50_count total_count rs_50_value remaining_count remaining_amount remaining_denomination_value : ℕ)

theorem denomination_calculation 
  (h1 : total_money = 10350)
  (h2 : rs_50_count = 97)
  (h3 : total_count = 108)
  (h4 : rs_50_value = 50)
  (h5 : remaining_count = total_count - rs_50_count)
  (h6 : remaining_amount = total_money - rs_50_count * rs_50_value)
  (h7 : remaining_denomination_value = remaining_amount / remaining_count) :
  remaining_denomination_value = 500 := 
sorry

end denomination_calculation_l308_30810


namespace max_value_of_f_l308_30800

noncomputable def f (theta x : ℝ) : ℝ :=
  (Real.cos theta)^2 - 2 * x * Real.cos theta - 1

noncomputable def M (x : ℝ) : ℝ :=
  if 0 <= x then 
    2 * x
  else 
    -2 * x

theorem max_value_of_f {x : ℝ} : 
  ∃ theta : ℝ, Real.cos theta ∈ [-1, 1] ∧ f theta x = M x :=
by
  sorry

end max_value_of_f_l308_30800


namespace identify_quadratic_equation_l308_30825

-- Definitions of the equations
def eqA : Prop := ∀ x : ℝ, x^2 + 1/x^2 = 4
def eqB : Prop := ∀ (a b x : ℝ), a*x^2 + b*x - 3 = 0
def eqC : Prop := ∀ x : ℝ, (x - 1)*(x + 2) = 1
def eqD : Prop := ∀ (x y : ℝ), 3*x^2 - 2*x*y - 5*y^2 = 0

-- Definition that identifies whether a given equation is a quadratic equation in one variable
def isQuadraticInOneVariable (eq : Prop) : Prop := 
  ∃ (a b c : ℝ) (a0 : a ≠ 0), ∀ x : ℝ, eq = (a * x^2 + b * x + c = 0)

theorem identify_quadratic_equation :
  isQuadraticInOneVariable eqC :=
by
  sorry

end identify_quadratic_equation_l308_30825


namespace sum_of_three_squares_l308_30851

variable (t s : ℝ)

-- Given equations
axiom h1 : 3 * t + 2 * s = 27
axiom h2 : 2 * t + 3 * s = 25

-- What we aim to prove
theorem sum_of_three_squares : 3 * s = 63 / 5 :=
by
  sorry

end sum_of_three_squares_l308_30851


namespace quadratic_has_real_roots_find_pos_m_l308_30830

-- Proof problem 1:
theorem quadratic_has_real_roots (m : ℝ) : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x^2 - 4 * m * x + 3 * m^2 = 0 :=
by
  sorry

-- Proof problem 2:
theorem find_pos_m (m x1 x2 : ℝ) (hm : x1 > x2) (h_diff : x1 - x2 = 2)
  (h_roots : ∀ m, (x^2 - 4*m*x + 3*m^2 = 0)) : m = 1 :=
by
  sorry

end quadratic_has_real_roots_find_pos_m_l308_30830


namespace tickets_sold_at_door_l308_30844

theorem tickets_sold_at_door :
  ∃ D : ℕ, ∃ A : ℕ, A + D = 800 ∧ (1450 * A + 2200 * D = 166400) ∧ D = 672 :=
by
  sorry

end tickets_sold_at_door_l308_30844


namespace sin_cos_from_tan_in_second_quadrant_l308_30808

theorem sin_cos_from_tan_in_second_quadrant (α : ℝ) 
  (h1 : Real.tan α = -2) 
  (h2 : α ∈ Set.Ioo (π / 2) π) : 
  Real.sin α = 2 * Real.sqrt 5 / 5 ∧ Real.cos α = -Real.sqrt 5 / 5 :=
by
  sorry

end sin_cos_from_tan_in_second_quadrant_l308_30808


namespace expression_value_l308_30854

theorem expression_value (a b : ℕ) (h₁ : a = 37) (h₂ : b = 12) : 
  (a + b)^2 - (a^2 + b^2) = 888 := by
  sorry

end expression_value_l308_30854


namespace product_is_correct_l308_30878

-- Define the numbers a and b
def a : ℕ := 72519
def b : ℕ := 9999

-- Theorem statement that proves the correctness of the product
theorem product_is_correct : a * b = 725117481 :=
by
  sorry

end product_is_correct_l308_30878


namespace polynomial_of_degree_2_l308_30803

noncomputable def polynomialSeq (n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
    ∃ (f_k f_k1 f_k2 : Polynomial ℝ),
      f_k ≠ Polynomial.C 0 ∧ (f_k * f_k1 = f_k1.comp f_k2)

theorem polynomial_of_degree_2 (n : ℕ) (h : n ≥ 3) :
  polynomialSeq n → 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
    ∃ f : Polynomial ℝ, f = Polynomial.X ^ 2 :=
sorry

end polynomial_of_degree_2_l308_30803


namespace sqrt_D_always_irrational_l308_30850

-- Definitions for consecutive even integers and D
def is_consecutive_even (p q : ℤ) : Prop :=
  ∃ k : ℤ, p = 2 * k ∧ q = 2 * k + 2

def D (p q : ℤ) : ℤ :=
  p^2 + q^2 + p * q^2

-- The main statement to prove
theorem sqrt_D_always_irrational (p q : ℤ) (h : is_consecutive_even p q) :
  ¬ ∃ r : ℤ, r * r = D p q :=
sorry

end sqrt_D_always_irrational_l308_30850


namespace candies_left_to_share_l308_30833

def initial_candies : Nat := 100
def sibling_count : Nat := 3
def candies_per_sibling : Nat := 10
def candies_Josh_eats : Nat := 16

theorem candies_left_to_share :
  let candies_given_to_siblings := sibling_count * candies_per_sibling;
  let candies_after_siblings := initial_candies - candies_given_to_siblings;
  let candies_given_to_friend := candies_after_siblings / 2;
  let candies_after_friend := candies_after_siblings - candies_given_to_friend;
  let candies_after_Josh := candies_after_friend - candies_Josh_eats;
  candies_after_Josh = 19 :=
by
  sorry

end candies_left_to_share_l308_30833


namespace sqrt_mult_pow_l308_30860

theorem sqrt_mult_pow (a : ℝ) (h_nonneg : 0 ≤ a) : (a^(2/3) * a^(1/5)) = a^(13/15) := by
  sorry

end sqrt_mult_pow_l308_30860


namespace stellar_hospital_multiple_births_l308_30852

/-- At Stellar Hospital, in a particular year, the multiple-birth statistics were such that sets of twins, triplets, and quintuplets accounted for 1200 of the babies born. 
There were twice as many sets of triplets as sets of quintuplets, and there were twice as many sets of twins as sets of triplets.
Determine how many of these 1200 babies were in sets of quintuplets. -/
theorem stellar_hospital_multiple_births 
    (a b c : ℕ)
    (h1 : b = 2 * c)
    (h2 : a = 2 * b)
    (h3 : 2 * a + 3 * b + 5 * c = 1200) :
    5 * c = 316 :=
by sorry

end stellar_hospital_multiple_births_l308_30852


namespace medicine_dosage_per_kg_l308_30888

theorem medicine_dosage_per_kg :
  ∀ (child_weight parts dose_per_part total_dose dose_per_kg : ℕ),
    (child_weight = 30) →
    (parts = 3) →
    (dose_per_part = 50) →
    (total_dose = parts * dose_per_part) →
    (dose_per_kg = total_dose / child_weight) →
    dose_per_kg = 5 :=
by
  intros child_weight parts dose_per_part total_dose dose_per_kg
  intros h1 h2 h3 h4 h5
  sorry

end medicine_dosage_per_kg_l308_30888


namespace kindergarten_solution_l308_30820

def kindergarten_cards (x y z t : ℕ) : Prop :=
  (x + y = 20) ∧ (z + t = 30) ∧ (y + z = 40) → (x + t = 10)

theorem kindergarten_solution : ∃ (x y z t : ℕ), kindergarten_cards x y z t :=
by {
  sorry
}

end kindergarten_solution_l308_30820


namespace quadrilateral_AD_length_l308_30827

noncomputable def length_AD (AB BC CD : ℝ) (angleB angleC : ℝ) : ℝ :=
  let AE := AB + BC * Real.cos angleC
  let CE := BC * Real.sin angleC
  let DE := CD - CE
  Real.sqrt (AE^2 + DE^2)

theorem quadrilateral_AD_length :
  let AB := 7
  let BC := 10
  let CD := 24
  let angleB := Real.pi / 2 -- 90 degrees in radians
  let angleC := Real.pi / 3 -- 60 degrees in radians
  length_AD AB BC CD angleB angleC = Real.sqrt (795 - 240 * Real.sqrt 3) :=
by
  sorry

end quadrilateral_AD_length_l308_30827


namespace expression_positive_intervals_l308_30858
open Real

theorem expression_positive_intervals (x : ℝ) :
  ((x + 2) * (x - 3) > 0) ↔ (x < -2 ∨ x > 3) := by
  sorry

end expression_positive_intervals_l308_30858


namespace polynomial_roots_l308_30866

-- Problem statement: prove that the roots of the given polynomial are {-1, 3, 3}
theorem polynomial_roots : 
  (λ x => x^3 - 5 * x^2 + 3 * x + 9) = (λ x => (x + 1) * (x - 3) ^ 2) :=
by
  sorry

end polynomial_roots_l308_30866


namespace sum_and_product_of_roots_l308_30885

-- Define the polynomial equation and the conditions on the roots
def cubic_eqn (x : ℝ) : Prop := 3 * x ^ 3 - 18 * x ^ 2 + 27 * x - 6 = 0

-- The Lean statement for the given problem
theorem sum_and_product_of_roots (p q r : ℝ) :
  cubic_eqn p ∧ cubic_eqn q ∧ cubic_eqn r →
  (p + q + r = 6) ∧ (p * q * r = 2) :=
by
  sorry

end sum_and_product_of_roots_l308_30885


namespace brenda_age_problem_l308_30814

variable (A B J : Nat)

theorem brenda_age_problem
  (h1 : A = 4 * B) 
  (h2 : J = B + 9) 
  (h3 : A = J) : 
  B = 3 := 
by 
  sorry

end brenda_age_problem_l308_30814


namespace manny_has_more_10_bills_than_mandy_l308_30868

theorem manny_has_more_10_bills_than_mandy :
  let mandy_bills_20 := 3
  let manny_bills_50 := 2
  let mandy_total_money := 20 * mandy_bills_20
  let manny_total_money := 50 * manny_bills_50
  let mandy_10_bills := mandy_total_money / 10
  let manny_10_bills := manny_total_money / 10
  mandy_10_bills < manny_10_bills →
  manny_10_bills - mandy_10_bills = 4 := sorry

end manny_has_more_10_bills_than_mandy_l308_30868


namespace value_of_a3_l308_30877

variable {a_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}

-- Given conditions
def S (n : ℕ) : ℤ := 2 * (n ^ 2) - 1
def a (n : ℕ) : ℤ := S n - S (n - 1)

-- The theorem to prove
theorem value_of_a3 : a 3 = 10 := by
  sorry

end value_of_a3_l308_30877


namespace gcd_90_270_l308_30864

theorem gcd_90_270 : Int.gcd 90 270 = 90 :=
by
  sorry

end gcd_90_270_l308_30864


namespace largest_possible_green_socks_l308_30816

/--
A box contains a mixture of green socks and yellow socks, with at most 2023 socks in total.
The probability of randomly pulling out two socks of the same color is exactly 1/3.
What is the largest possible number of green socks in the box? 
-/
theorem largest_possible_green_socks (g y : ℤ) (t : ℕ) (h : t ≤ 2023) 
  (prob_condition : (g * (g - 1) + y * (y - 1) = t * (t - 1) / 3)) : 
  g ≤ 990 :=
sorry

end largest_possible_green_socks_l308_30816


namespace no_nat_number_satisfies_l308_30884

theorem no_nat_number_satisfies (n : ℕ) : ¬ ((n^2 + 6 * n + 2019) % 100 = 0) :=
sorry

end no_nat_number_satisfies_l308_30884


namespace dave_apps_problem_l308_30848

theorem dave_apps_problem 
  (initial_apps : ℕ)
  (added_apps : ℕ)
  (final_apps : ℕ)
  (total_apps := initial_apps + added_apps)
  (deleted_apps := total_apps - final_apps) :
  initial_apps = 21 →
  added_apps = 89 →
  final_apps = 24 →
  (added_apps - deleted_apps = 3) :=
by
  intros
  sorry

end dave_apps_problem_l308_30848


namespace student_test_score_l308_30828

variable (C I : ℕ)

theorem student_test_score  
  (h1 : C + I = 100)
  (h2 : C - 2 * I = 64) :
  C = 88 :=
by
  -- Proof steps should go here
  sorry

end student_test_score_l308_30828


namespace find_vector_BC_l308_30818

structure Point2D where
  x : ℝ
  y : ℝ

def A : Point2D := ⟨0, 1⟩
def B : Point2D := ⟨3, 2⟩
def AC : Point2D := ⟨-4, -3⟩

def vector_add (p1 p2 : Point2D) : Point2D := ⟨p1.x + p2.x, p1.y + p2.y⟩
def vector_sub (p1 p2 : Point2D) : Point2D := ⟨p1.x - p2.x, p1.y - p2.y⟩

def C : Point2D := vector_add A AC
def BC : Point2D := vector_sub C B

theorem find_vector_BC : BC = ⟨-7, -4⟩ := by
  sorry

end find_vector_BC_l308_30818


namespace second_crew_tractors_l308_30890

theorem second_crew_tractors
    (total_acres : ℕ)
    (days : ℕ)
    (first_crew_days : ℕ)
    (first_crew_tractors : ℕ)
    (acres_per_tractor_per_day : ℕ)
    (remaining_days : ℕ)
    (remaining_acres_after_first_crew : ℕ)
    (second_crew_acres_per_tractor : ℕ) :
    total_acres = 1700 → days = 5 → first_crew_days = 2 → first_crew_tractors = 2 → 
    acres_per_tractor_per_day = 68 → remaining_days = 3 → 
    remaining_acres_after_first_crew = total_acres - (first_crew_tractors * acres_per_tractor_per_day * first_crew_days) → 
    second_crew_acres_per_tractor = acres_per_tractor_per_day * remaining_days → 
    (remaining_acres_after_first_crew / second_crew_acres_per_tractor = 7) := 
by
  sorry

end second_crew_tractors_l308_30890


namespace switches_assembled_are_correct_l308_30815

-- Definitions based on conditions
def total_payment : ℕ := 4700
def first_worker_payment : ℕ := 2000
def second_worker_per_switch_time_min : ℕ := 4
def third_worker_less_payment : ℕ := 300
def overtime_hours : ℕ := 5
def total_minutes (hours : ℕ) : ℕ := hours * 60

-- Function to calculate total switches assembled
noncomputable def total_switches_assembled :=
  let second_worker_payment := (total_payment - first_worker_payment + third_worker_less_payment) / 2
  let third_worker_payment := second_worker_payment - third_worker_less_payment
  let rate_per_switch := second_worker_payment / (total_minutes overtime_hours / second_worker_per_switch_time_min)
  let first_worker_switches := first_worker_payment / rate_per_switch
  let second_worker_switches := total_minutes overtime_hours / second_worker_per_switch_time_min
  let third_worker_switches := third_worker_payment / rate_per_switch
  first_worker_switches + second_worker_switches + third_worker_switches

-- Lean 4 statement to prove the problem
theorem switches_assembled_are_correct : 
  total_switches_assembled = 235 := by
  sorry

end switches_assembled_are_correct_l308_30815


namespace total_cost_l308_30881

def daily_rental_cost : ℝ := 25
def cost_per_mile : ℝ := 0.20
def duration_days : ℕ := 4
def distance_miles : ℕ := 400

theorem total_cost 
: (daily_rental_cost * duration_days + cost_per_mile * distance_miles) = 180 := 
by
  sorry

end total_cost_l308_30881


namespace value_of_m_l308_30805

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x

theorem value_of_m (a b m : ℝ) (h₀ : m ≠ 0)
  (h₁ : 3 * m^2 + 2 * a * m + b = 0)
  (h₂ : m^2 + a * m + b = 0)
  (h₃ : ∃ x, f x a b = 1/2) :
  m = 3/2 :=
by
  sorry

end value_of_m_l308_30805


namespace part1_part2_l308_30841

-- Define the function, assumptions, and the proof for the first part
theorem part1 (m : ℝ) (x : ℝ) :
  (∀ x > 1, -m * (0 * x + 1) * Real.log x + x - 0 ≥ 0) →
  m ≤ Real.exp 1 := sorry

-- Define the function, assumptions, and the proof for the second part
theorem part2 (x : ℝ) :
  (∀ x > 0, (x - 1) * (-(x + 1) * Real.log x + x - 1) ≤ 0) := sorry

end part1_part2_l308_30841


namespace profit_calculation_l308_30817

def totalProfit (totalMoney part1 interest1 interest2 time : ℕ) : ℕ :=
  let part2 := totalMoney - part1
  let interestFromPart1 := part1 * interest1 / 100 * time
  let interestFromPart2 := part2 * interest2 / 100 * time
  interestFromPart1 + interestFromPart2

theorem profit_calculation : 
  totalProfit 80000 70000 10 20 1 = 9000 :=
  by 
    -- Rather than providing a full proof, we insert 'sorry' as per the instruction.
    sorry

end profit_calculation_l308_30817


namespace range_of_m_l308_30853

theorem range_of_m (x y m : ℝ) (h1 : x - y = 2 * m + 7) (h2 : x + y = 4 * m - 3) 
  (h3 : x < 0) (h4 : y < 0) : m < -2 / 3 := 
by 
  sorry

end range_of_m_l308_30853


namespace kho_kho_only_l308_30896

variable (K H B : ℕ)

theorem kho_kho_only :
  (K + B = 10) ∧ (H + 5 = H + B) ∧ (B = 5) ∧ (K + H + B = 45) → H = 35 :=
by
  intros h
  sorry

end kho_kho_only_l308_30896


namespace triangle_equilateral_l308_30886

noncomputable def is_equilateral_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = b ∧ b = c ∧ A = B ∧ B = C

theorem triangle_equilateral 
  (a b c A B C : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * a * b * c) 
  (h2 : Real.sin A = 2 * Real.sin B * Real.cos C) : 
  is_equilateral_triangle a b c A B C :=
sorry

end triangle_equilateral_l308_30886


namespace monotonic_decreasing_fx_l308_30862

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem monotonic_decreasing_fx : ∀ (x : ℝ), (0 < x) ∧ (x < (1 / exp 1)) → deriv f x < 0 := 
by
  sorry

end monotonic_decreasing_fx_l308_30862


namespace hyperbola_equation_l308_30804

theorem hyperbola_equation {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (∀ {x y : ℝ}, x^2 / 12 + y^2 / 4 = 1 → True) →
  (∀ {x y : ℝ}, x^2 / a^2 - y^2 / b^2 = 1 → True) →
  (∀ {x y : ℝ}, y = Real.sqrt 3 * x → True) →
  (∃ k : ℝ, 4 < k ∧ k < 12 ∧ 2 = 12 - k ∧ 6 = k - 4) →
  a = 2 ∧ b = 6 := by
  intros h_ellipse h_hyperbola h_asymptote h_k
  sorry

end hyperbola_equation_l308_30804


namespace find_train_speed_l308_30811

variable (L V : ℝ)

-- Conditions
def condition1 := V = L / 10
def condition2 := V = (L + 600) / 30

-- Theorem statement
theorem find_train_speed (h1 : condition1 L V) (h2 : condition2 L V) : V = 30 :=
by
  sorry

end find_train_speed_l308_30811


namespace pure_imaginary_condition_l308_30829

-- Define the problem
theorem pure_imaginary_condition (θ : ℝ) :
  (∀ k : ℤ, θ = (3 * Real.pi / 4) + k * Real.pi) →
  ∀ z : ℂ, z = (Complex.cos θ - Complex.sin θ * Complex.I) * (1 + Complex.I) →
  ∃ k : ℤ, θ = (3 * Real.pi / 4) + k * Real.pi → 
  (Complex.re z = 0 ∧ Complex.im z ≠ 0) :=
  sorry

end pure_imaginary_condition_l308_30829


namespace find_a_l308_30891

def polys_are_integers (a b : Int) : Prop :=
  a ≠ 0 ∧ b ≠ 0

def factor_condition (a b : Int) : Prop :=
  ∀ x : ℝ, (x^2 - x - 1 = 0) → (a * x^19 + b * x^18 + 1 = 0)

theorem find_a (a b : Int) (h : polys_are_integers a b) (h_fac : factor_condition a b) : a = 1597 :=
by
  sorry

end find_a_l308_30891


namespace algebraic_expression_value_l308_30813

theorem algebraic_expression_value (a b : ℝ) (h : ∃ x : ℝ, x = 2 ∧ 3 * (a - x) = 2 * (b * x - 4)) :
  9 * a^2 - 24 * a * b + 16 * b^2 + 25 = 29 :=
by sorry

end algebraic_expression_value_l308_30813


namespace pipe_B_fill_time_l308_30822

variable (A B C : ℝ)
variable (fill_time : ℝ := 16)
variable (total_tank : ℝ := 1)

-- Conditions
axiom condition1 : A + B + C = (1 / fill_time)
axiom condition2 : A = 2 * B
axiom condition3 : B = 2 * C

-- Prove that B alone will take 56 hours to fill the tank
theorem pipe_B_fill_time : B = (1 / 56) :=
by sorry

end pipe_B_fill_time_l308_30822


namespace trader_total_discount_correct_l308_30806

theorem trader_total_discount_correct :
  let CP_A := 200
  let CP_B := 150
  let CP_C := 100
  let MSP_A := CP_A + 0.50 * CP_A
  let MSP_B := CP_B + 0.50 * CP_B
  let MSP_C := CP_C + 0.50 * CP_C
  let SP_A := 0.99 * CP_A
  let SP_B := 0.97 * CP_B
  let SP_C := 0.98 * CP_C
  let discount_A := MSP_A - SP_A
  let discount_B := MSP_B - SP_B
  let discount_C := MSP_C - SP_C
  let total_discount := discount_A + discount_B + discount_C
  total_discount = 233.5 := by sorry

end trader_total_discount_correct_l308_30806


namespace paint_left_for_solar_system_l308_30859

-- Definitions for the paint used
def Mary's_paint := 3
def Mike's_paint := Mary's_paint + 2
def Lucy's_paint := 4

-- Total original amount of paint
def original_paint := 25

-- Total paint used by Mary, Mike, and Lucy
def total_paint_used := Mary's_paint + Mike's_paint + Lucy's_paint

-- Theorem stating the amount of paint left for the solar system
theorem paint_left_for_solar_system : (original_paint - total_paint_used) = 13 :=
by
  sorry

end paint_left_for_solar_system_l308_30859


namespace decimal_to_vulgar_fraction_l308_30836

theorem decimal_to_vulgar_fraction :
  ∃ (n d : ℕ), (0.34 : ℝ) = (n : ℝ) / (d : ℝ) ∧ n = 17 :=
by
  sorry

end decimal_to_vulgar_fraction_l308_30836


namespace cori_age_proof_l308_30846

theorem cori_age_proof:
  ∃ (x : ℕ), (3 + x = (1 / 3) * (19 + x)) ∧ x = 5 :=
by
  sorry

end cori_age_proof_l308_30846


namespace longest_diagonal_length_l308_30824

-- Define the conditions
variables {a b : ℝ} (h_area : 135 = 1/2 * a * b) (h_ratio : a / b = 5 / 3)

-- Define the target to prove
theorem longest_diagonal_length (a b : ℝ) (h_area : 135 = 1/2 * a * b) (h_ratio : a / b = 5 / 3) :
    a = 15 * Real.sqrt 2 :=
sorry

end longest_diagonal_length_l308_30824


namespace answered_both_l308_30849

variables (A B : Type)
variables {test_takers : Type}

-- Defining the conditions
def pa : ℝ := 0.80  -- 80% answered first question correctly
def pb : ℝ := 0.75  -- 75% answered second question correctly
def pnone : ℝ := 0.05 -- 5% answered neither question correctly

-- Formal problem statement
theorem answered_both (test_takers: Type) : 
  (pa + pb - (1 - pnone)) = 0.60 :=
by
  sorry

end answered_both_l308_30849


namespace sum_of_cubes_of_real_roots_eq_11_l308_30839

-- Define the polynomial f(x) = x^3 - 2x^2 - x + 1
def poly (x : ℝ) : ℝ := x^3 - 2 * x^2 - x + 1

-- State that the polynomial has exactly three real roots
axiom three_real_roots : ∃ (x1 x2 x3 : ℝ), poly x1 = 0 ∧ poly x2 = 0 ∧ poly x3 = 0

-- Prove that the sum of the cubes of the real roots is 11
theorem sum_of_cubes_of_real_roots_eq_11 (x1 x2 x3 : ℝ)
  (hx1 : poly x1 = 0) (hx2 : poly x2 = 0) (hx3 : poly x3 = 0) : 
  x1^3 + x2^3 + x3^3 = 11 :=
by
  sorry

end sum_of_cubes_of_real_roots_eq_11_l308_30839


namespace box_dimensions_l308_30895

-- Given conditions
variables (a b c : ℕ)
axiom h1 : a + c = 17
axiom h2 : a + b = 13
axiom h3 : b + c = 20

theorem box_dimensions : a = 5 ∧ b = 8 ∧ c = 12 :=
by {
  -- These parts will contain the actual proof, which we omit for now
  sorry
}

end box_dimensions_l308_30895


namespace intersection_of_A_and_B_l308_30855

-- Given sets A and B
def A : Set ℤ := { -1, 0, 1, 2 }
def B : Set ℤ := { 0, 2, 3 }

-- Prove that the intersection of A and B is {0, 2}
theorem intersection_of_A_and_B : A ∩ B = {0, 2} := 
by
  sorry

end intersection_of_A_and_B_l308_30855


namespace minimum_wins_l308_30893

theorem minimum_wins (x y : ℕ) (h_score : 3 * x + y = 10) (h_games : x + y ≤ 7) (h_bounds : 0 < x ∧ x < 4) : x = 2 :=
by
  sorry

end minimum_wins_l308_30893


namespace acute_angles_45_degrees_l308_30892

-- Assuming quadrilaterals ABCD and A'B'C'D' such that sides of each lie on 
-- the perpendicular bisectors of the sides of the other. We want to prove that
-- the acute angles of A'B'C'D' are 45 degrees.

def convex_quadrilateral (Q : Type) := 
  ∃ (A B C D : Q), True -- Placeholder for a more detailed convex quadrilateral structure

def perpendicular_bisector (S1 S2 T1 T2: Type) := 
  ∃ (M : Type), True -- Placeholder for a more detailed perpendicular bisector structure

theorem acute_angles_45_degrees
  (Q1 Q2 : Type)
  (h1 : convex_quadrilateral Q1)
  (h2 : convex_quadrilateral Q2)
  (perp1 : perpendicular_bisector Q1 Q1 Q2 Q2)
  (perp2 : perpendicular_bisector Q2 Q2 Q1 Q1) :
  ∀ (θ : ℝ), θ = 45 := 
by
  sorry

end acute_angles_45_degrees_l308_30892


namespace worksheets_graded_l308_30869

theorem worksheets_graded (w : ℕ) (h1 : ∀ (n : ℕ), n = 3) (h2 : ∀ (n : ℕ), n = 15) (h3 : ∀ (p : ℕ), p = 24)  :
  w = 7 :=
sorry

end worksheets_graded_l308_30869


namespace transfer_equation_correct_l308_30840

theorem transfer_equation_correct (x : ℕ) :
  46 + x = 3 * (30 - x) := 
sorry

end transfer_equation_correct_l308_30840


namespace find_missing_number_l308_30819

theorem find_missing_number (x : ℕ) (h1 : (1 + 22 + 23 + 24 + x + 26 + 27 + 2) = 8 * 20) : x = 35 :=
  sorry

end find_missing_number_l308_30819


namespace crayons_divided_equally_l308_30838

theorem crayons_divided_equally (total_crayons : ℕ) (number_of_people : ℕ) (crayons_per_person : ℕ) 
  (h1 : total_crayons = 24) (h2 : number_of_people = 3) : 
  crayons_per_person = total_crayons / number_of_people → crayons_per_person = 8 :=
by
  intro h
  rw [h1, h2] at h
  have : 24 / 3 = 8 := by norm_num
  rw [this] at h
  exact h

end crayons_divided_equally_l308_30838


namespace trig_identity_l308_30809

theorem trig_identity (α : ℝ) :
  4.10 * (Real.cos (45 * Real.pi / 180 - α)) ^ 2 
  - (Real.cos (60 * Real.pi / 180 + α)) ^ 2 
  - Real.cos (75 * Real.pi / 180) * Real.sin (75 * Real.pi / 180 - 2 * α) 
  = Real.sin (2 * α) := 
sorry

end trig_identity_l308_30809


namespace range_of_x8_l308_30879

theorem range_of_x8 (x : ℕ → ℝ) (h1 : 0 ≤ x 1 ∧ x 1 ≤ x 2)
  (h_recurrence : ∀ n ≥ 1, x (n+2) = x (n+1) + x n)
  (h_x7 : 1 ≤ x 7 ∧ x 7 ≤ 2) : 
  (21/13 : ℝ) ≤ x 8 ∧ x 8 ≤ (13/4) :=
sorry

end range_of_x8_l308_30879


namespace find_certain_number_l308_30826

theorem find_certain_number (x : ℝ) (h : ((x^4) * 3.456789)^10 = 10^20) : x = 10 :=
sorry

end find_certain_number_l308_30826


namespace missing_number_l308_30894

theorem missing_number (mean : ℝ) (numbers : List ℝ) (x : ℝ) (h_mean : mean = 14.2) (h_numbers : numbers = [13.0, 8.0, 13.0, 21.0, 23.0]) :
  (numbers.sum + x) / (numbers.length + 1) = mean → x = 7.2 :=
by
  -- states the hypothesis about the mean calculation into the theorem structure
  intro h
  sorry

end missing_number_l308_30894


namespace evaluate_expression_l308_30867

theorem evaluate_expression (a b c : ℝ) : 
  (a / (30 - a) + b / (70 - b) + c / (75 - c) = 9) → 
  (6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 2.4) :=
by 
  sorry

end evaluate_expression_l308_30867


namespace find_polynomial_P_l308_30883

noncomputable def P (x : ℝ) : ℝ :=
  - (5/8) * x^3 + (5/2) * x^2 + (1/8) * x - 1

theorem find_polynomial_P 
  (α β γ : ℝ)
  (h_roots : ∀ {x: ℝ}, x^3 - 4 * x^2 + 6 * x + 8 = 0 → x = α ∨ x = β ∨ x = γ)
  (h1 : P α = β + γ)
  (h2 : P β = α + γ)
  (h3 : P γ = α + β)
  (h4 : P (α + β + γ) = -20) :
  P x = - (5/8) * x^3 + (5/2) * x^2 + (1/8) * x - 1 :=
by sorry

end find_polynomial_P_l308_30883


namespace minimum_value_is_16_l308_30857

noncomputable def minimum_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) : ℝ :=
  (x^3 / (y - 1) + y^3 / (x - 1))

theorem minimum_value_is_16 (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  minimum_value_expression x y hx hy ≥ 16 :=
sorry

end minimum_value_is_16_l308_30857


namespace smallest_perimeter_l308_30845

noncomputable def smallest_possible_perimeter : ℕ :=
  let n := 3
  n + (n + 1) + (n + 2)

theorem smallest_perimeter (n : ℕ) (h : n > 2) (ineq1 : n + (n + 1) > (n + 2)) 
  (ineq2 : n + (n + 2) > (n + 1)) (ineq3 : (n + 1) + (n + 2) > n) : 
  smallest_possible_perimeter = 12 :=
by
  sorry

end smallest_perimeter_l308_30845


namespace current_year_2021_l308_30876

variables (Y : ℤ)

def parents_moved_to_America := 1982
def Aziz_age := 36
def years_before_born := 3

theorem current_year_2021
  (h1 : parents_moved_to_America = 1982)
  (h2 : Aziz_age = 36)
  (h3 : years_before_born = 3)
  (h4 : Y - (Aziz_age) - (years_before_born) = 1982) : 
  Y = 2021 :=
by {
  sorry
}

end current_year_2021_l308_30876
